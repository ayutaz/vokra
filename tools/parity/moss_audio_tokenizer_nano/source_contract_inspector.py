#!/usr/bin/env python3
"""Inspect the pinned MOSS Audio Tokenizer Nano source contract.

This is an evidence collector, not a model validator. It materializes only
the seven non-weight files selected by the fixed Hugging Face revision. The
weight shard is authenticated from the expanded HF server tree/LFS identity
but is never downloaded. The Transformers probe is currently security-blocked
before any source or weight access because its former meta-device
implementation depended on a vulnerable third-party release. The
dependency-free metadata and source contract validators remain available for
separately authenticated audits.

The script records the existing source/weight owner sign-off but intentionally
leaves Python closure, API/runtime support, parity, and publication blocked. A
complete source inspection exits with status 2 so a
caller cannot accidentally continue into conversion or publication.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPOSITORY = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
REVISION = "6aa02b01e445cc585582cf0ba480bc3ea6c8dd68"
TRANSFORMERS_VERSION = "5.10.4"
SECURITY_ADVISORY = "BLOCKED_SECURITY_ADVISORY"
PAYLOAD_FILES = (
    ".gitattributes",
    "README.md",
    "__init__.py",
    "config.json",
    "configuration_moss_audio_tokenizer.py",
    "modeling_moss_audio_tokenizer.py",
    "model.safetensors.index.json",
    "model-00001-of-00001.safetensors",
)
MATERIALIZED_FILES = PAYLOAD_FILES[:-1]
WEIGHT_FILE = PAYLOAD_FILES[-1]
SOURCE_ROLES = {
    "configuration_moss_audio_tokenizer.py": "configuration",
    "modeling_moss_audio_tokenizer.py": "modeling",
}
EXPECTED_CONFIG = {
    "sampling_rate": 48_000,
    "downsample_rate": 3_840,
    "number_channels": 2,
}
EXPECTED_QUANTIZER = {
    "num_quantizers": 16,
    "codebook_size": 1_024,
    "codebook_dim": 8,
    "rvq_dim": 512,
    "output_dim": 768,
}
EXPECTED_MODEL_TYPE = "moss-audio-tokenizer"
EXPECTED_ARCHITECTURES = ["MossAudioTokenizerModel"]
EXPECTED_AUTO_MAP = {
    "AutoConfig": "configuration_moss_audio_tokenizer.MossAudioTokenizerConfig",
    "AutoModel": "modeling_moss_audio_tokenizer.MossAudioTokenizerModel",
}
# These fields are the shape-bearing subset of the official decoder_kwargs in
# config.json.  The complete byte identity is checked separately; keeping the
# shape contract here makes a changed upstream config fail before a meta probe
# can accidentally bless a different decoder topology.
EXPECTED_DECODER_LAYOUT = [
    {"module_type": "PatchedPretransform", "patch_size": 4},
    {"module_type": "Transformer", "input_dimension": 192, "output_dimension": 768},
    {"module_type": "PatchedPretransform", "patch_size": 2},
    {"module_type": "Transformer", "input_dimension": 384, "output_dimension": 768},
    {"module_type": "PatchedPretransform", "patch_size": 2},
    {"module_type": "Transformer", "input_dimension": 384, "output_dimension": 768},
    {"module_type": "PatchedPretransform", "patch_size": 2},
    {"module_type": "Transformer", "input_dimension": 384, "output_dimension": 240},
    {"module_type": "PatchedPretransform", "patch_size": 240},
]
EXPECTED_TAPS = [
    {"name": "quantizer", "shape": "1x768x2"},
    {"name": "decoder_0", "shape": "1x192x8"},
    {"name": "decoder_1", "shape": "1x768x8"},
    {"name": "decoder_2", "shape": "1x384x16"},
    {"name": "decoder_3", "shape": "1x768x16"},
    {"name": "decoder_4", "shape": "1x384x32"},
    {"name": "decoder_5", "shape": "1x768x32"},
    {"name": "decoder_6", "shape": "1x384x64"},
    {"name": "decoder_7", "shape": "1x240x64"},
    {"name": "decoder_8", "shape": "1x1x15360"},
]
EXPECTED_AUDIO_SHAPE = "1x2x7680"
EXPECTED_MODEL_INFO = {
    "id": REPOSITORY,
    "sha": REVISION,
    "private": False,
    "gated": False,
    "disabled": False,
    "cardData_license": "apache-2.0",
}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class InspectionError(RuntimeError):
    """An expected fail-closed inspection error."""


def reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InspectionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_json_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InspectionError(f"invalid JSON {path}: {error}") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as error:
        raise InspectionError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def file_hashes(path: Path) -> tuple[str, str]:
    """Return SHA-256 and the canonical Git blob SHA-1 in one pass."""

    try:
        size = path.stat().st_size
        sha256 = hashlib.sha256()
        git_sha1 = hashlib.sha1()
        git_sha1.update(f"blob {size}\0".encode("ascii"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                sha256.update(chunk)
                git_sha1.update(chunk)
    except OSError as error:
        raise InspectionError(f"cannot hash {path}: {error}") from error
    return sha256.hexdigest(), git_sha1.hexdigest()


def safe_relative_path(path: object) -> str:
    if not isinstance(path, str) or not path or "\\" in path or "\x00" in path:
        raise InspectionError(f"unsafe upstream path: {path!r}")
    if path.startswith("/") or ".." in Path(path).parts:
        raise InspectionError(f"unsafe upstream path: {path!r}")
    return path


def validate_model_info(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(EXPECTED_MODEL_INFO):
        raise InspectionError("HF model_info identity schema is not exact")
    for key, expected in EXPECTED_MODEL_INFO.items():
        if value.get(key) != expected:
            raise InspectionError(
                f"HF model_info.{key}={value.get(key)!r}, expected {expected!r}"
            )
    return dict(value)


def validate_vokra_checkout(root: Path, expected_head: str) -> dict[str, Any]:
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise InspectionError("Vokra checkout root must be an absolute real directory")
    if not HEX40.fullmatch(expected_head):
        raise InspectionError("expected Vokra checkout HEAD must be lowercase 40-hex")
    resolved_root = root.resolve(strict=True)
    if resolved_root.is_symlink():
        raise InspectionError("resolved Vokra checkout root must not be a symlink")
    try:
        head_result = subprocess.run(
            ["git", "-C", str(resolved_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        status_result = subprocess.run(
            ["git", "-C", str(resolved_root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise InspectionError(f"cannot inspect Vokra checkout: {error}") from error
    head = head_result.stdout.strip()
    clean = status_result.stdout == ""
    if not HEX40.fullmatch(head) or head != expected_head:
        raise InspectionError(f"Vokra checkout HEAD mismatch: {head!r} != {expected_head!r}")
    if not clean:
        raise InspectionError("Vokra checkout is not clean")
    return {"expected_head": expected_head, "head": head, "clean": True}


def file_row(path: Path, server_row: dict[str, Any] | None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise InspectionError(f"payload is not a regular file: {path.name}")
    size = path.stat().st_size
    digest, canonical_git_blob = file_hashes(path)
    row: dict[str, Any] = {
        "path": path.name,
        "role": SOURCE_ROLES.get(path.name, path.name.lower().replace(".", "_")),
        "bytes": size,
        "sha256": digest,
        "materialized": True,
        "content_not_downloaded": False,
        "status": "AUTHENTICATED",
    }
    if server_row is not None:
        if server_row.get("size") != size:
            raise InspectionError(
                f"server/materialized size mismatch for {path.name}: "
                f"{size} != {server_row.get('size')}"
            )
        lfs_sha = server_row.get("lfs_payload_sha256", server_row.get("lfs_sha256"))
        if lfs_sha is not None and digest != lfs_sha:
            raise InspectionError(
                f"materialized LFS SHA-256 mismatch for {path.name}: "
                f"{digest} != {lfs_sha}"
            )
        if lfs_sha is None:
            server_git_blob = server_row.get("git_blob_sha1")
            if not isinstance(server_git_blob, str) or not HEX40.fullmatch(server_git_blob):
                raise InspectionError(
                    f"materialized non-LFS file lacks a canonical Git blob SHA-1: {path.name}"
                )
            if canonical_git_blob != server_git_blob:
                raise InspectionError(
                    f"materialized Git blob SHA-1 mismatch for {path.name}: "
                    f"{canonical_git_blob} != {server_git_blob}"
                )
        row.update(
            {
                "git_blob_sha1": server_row.get("git_blob_sha1"),
                "canonical_git_blob_sha1": canonical_git_blob,
                "lfs_pointer_git_blob_sha1": server_row.get(
                    "lfs_pointer_git_blob_sha1"
                ),
                "lfs_payload_sha256": lfs_sha,
                "server_size": server_row.get("size"),
            }
        )
    return row


def canonical_remote_source(path: Path) -> str:
    resolved = path.resolve()
    try:
        index = resolved.parts.index("transformers_modules")
    except ValueError as error:
        raise InspectionError(
            f"custom source is outside transformers_modules: {resolved}"
        ) from error
    relative = Path(*resolved.parts[index:]).as_posix()
    if not relative.startswith("transformers_modules/"):
        raise InspectionError(f"invalid custom source path: {relative}")
    return relative


def source_identity(
    obj: object, label: str, snapshot: Path, expected_name: str
) -> dict[str, Any]:
    source = inspect.getsourcefile(obj)
    if source is None:
        raise InspectionError(f"{label} has no inspectable source file")
    source_path = Path(source).resolve()
    canonical = canonical_remote_source(source_path)
    if source_path.name != expected_name:
        raise InspectionError(
            f"{label} came from {source_path.name}, expected {expected_name}"
        )
    source_sha = sha256_file(source_path)
    snapshot_sha = sha256_file(snapshot / expected_name)
    if source_sha != snapshot_sha:
        raise InspectionError(
            f"{label} source differs from pinned snapshot: "
            f"{source_sha} != {snapshot_sha}"
        )
    return {
        "path": canonical,
        "filename": expected_name,
        "sha256": source_sha,
        "bytes": source_path.stat().st_size,
        "status": "AUTHENTICATED",
    }


def validate_snapshot(
    snapshot: Path, server_rows: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise InspectionError(f"snapshot directory is missing: {snapshot}")
    entries = list(snapshot.iterdir())
    for entry in entries:
        if entry.name == ".cache":
            if entry.is_symlink() or not entry.is_dir():
                raise InspectionError("snapshot .cache is not a real directory")
            continue
        if entry.name == WEIGHT_FILE:
            raise InspectionError("weight payload was materialized; source inspection is model-free")
        if entry.name not in MATERIALIZED_FILES:
            raise InspectionError(f"snapshot contains unexpected entry: {entry.name}")
        if entry.is_symlink() or not entry.is_file():
            raise InspectionError(f"snapshot payload is not regular: {entry.name}")
    missing = [name for name in MATERIALIZED_FILES if not (snapshot / name).is_file()]
    if missing:
        raise InspectionError(f"snapshot is missing payload files: {missing}")

    rows = [file_row(snapshot / name, server_rows.get(name)) for name in MATERIALIZED_FILES]
    weight_server = server_rows.get(WEIGHT_FILE)
    if weight_server is None:
        raise InspectionError(f"server-only weight identity is missing: {WEIGHT_FILE}")
    weight_size = weight_server.get("size")
    weight_lfs_sha = weight_server.get(
        "lfs_payload_sha256", weight_server.get("lfs_sha256")
    )
    if not isinstance(weight_size, int) or weight_size <= 0:
        raise InspectionError("server-only weight size is invalid")
    if not isinstance(weight_lfs_sha, str) or not HEX64.fullmatch(weight_lfs_sha):
        raise InspectionError(
            "server-only weight lacks an authenticated LFS payload SHA-256"
        )
    weight_pointer = weight_server.get("lfs_pointer_git_blob_sha1")
    if not isinstance(weight_pointer, str) or not HEX40.fullmatch(weight_pointer):
        raise InspectionError(
            "server-only weight lacks an authenticated LFS pointer Git identity"
        )
    rows.append(
        {
            "path": WEIGHT_FILE,
            "role": "weights",
            "server_bytes": weight_size,
            "lfs_pointer_git_blob_sha1": weight_server.get(
                "lfs_pointer_git_blob_sha1"
            ),
            "lfs_payload_sha256": weight_lfs_sha,
            "materialized": False,
            "content_not_downloaded": True,
            "status": "AUTHENTICATED_SERVER_IDENTITY_ONLY",
        }
    )
    config = load_json(snapshot / "config.json")
    if not isinstance(config, dict):
        raise InspectionError("config.json top level is not an object")
    if config.get("model_type") != EXPECTED_MODEL_TYPE:
        raise InspectionError(
            f"config.model_type={config.get('model_type')!r}, expected {EXPECTED_MODEL_TYPE!r}"
        )
    if config.get("architectures") != EXPECTED_ARCHITECTURES:
        raise InspectionError("config.architectures is not the official Nano model")
    if config.get("auto_map") != EXPECTED_AUTO_MAP:
        raise InspectionError("config.auto_map is not the official Nano API mapping")
    for key, expected in EXPECTED_CONFIG.items():
        if config.get(key) != expected:
            raise InspectionError(
                f"config.{key}={config.get(key)!r}, expected {expected!r}"
            )
    quantizer = config.get("quantizer_kwargs")
    if not isinstance(quantizer, dict):
        raise InspectionError("config.quantizer_kwargs is not an object")
    for key, expected in EXPECTED_QUANTIZER.items():
        if quantizer.get(key) != expected:
            raise InspectionError(
                f"config.quantizer_kwargs.{key}={quantizer.get(key)!r}, "
                f"expected {expected!r}"
            )
    decoder_kwargs = config.get("decoder_kwargs")
    if not isinstance(decoder_kwargs, list) or len(decoder_kwargs) != len(EXPECTED_DECODER_LAYOUT):
        raise InspectionError("config.decoder_kwargs does not have the official 9-stage layout")
    for index, expected in enumerate(EXPECTED_DECODER_LAYOUT):
        actual = decoder_kwargs[index]
        if not isinstance(actual, dict) or any(actual.get(key) != value for key, value in expected.items()):
            raise InspectionError(f"config.decoder_kwargs[{index}] shape contract drifted")
    index = load_json(snapshot / "model.safetensors.index.json")
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise InspectionError("checkpoint index has no weight_map")
    weight_map = index["weight_map"]
    if not weight_map or set(weight_map.values()) != {WEIGHT_FILE}:
        raise InspectionError("checkpoint index points outside the fixed Nano shard")
    return (
        rows,
        {
            "sampling_rate": EXPECTED_CONFIG["sampling_rate"],
            "downsample_rate": EXPECTED_CONFIG["downsample_rate"],
            "number_channels": EXPECTED_CONFIG["number_channels"],
            "quantizer_kwargs": EXPECTED_QUANTIZER,
            "model_type": EXPECTED_MODEL_TYPE,
            "architectures": EXPECTED_ARCHITECTURES,
            "auto_map": EXPECTED_AUTO_MAP,
            "decoder_layout": EXPECTED_DECODER_LAYOUT,
            "status": "AUTHENTICATED",
        },
        {"weight_map_entries": len(weight_map), "status": "AUTHENTICATED"},
    )


def server_tree(api: Any, repository: str, revision: str) -> tuple[str, dict[str, dict[str, Any]]]:
    info = api.model_info(repo_id=repository, revision=revision)
    resolved = getattr(info, "sha", None)
    if resolved != revision or not isinstance(resolved, str) or not HEX40.fullmatch(resolved):
        raise InspectionError(f"HF revision mismatch: {resolved!r} != {revision}")
    selected: dict[str, dict[str, Any]] = {}
    for item in api.list_repo_tree(
        repo_id=repository, revision=revision, recursive=True, expand=True
    ):
        kind = getattr(item, "type", None)
        if kind in {"directory", "folder", "dir"} or item.__class__.__name__ == "RepoFolder":
            continue
        path = safe_relative_path(getattr(item, "path", None))
        if path not in PAYLOAD_FILES:
            continue
        if kind not in {None, "file"} and item.__class__.__name__ != "RepoFile":
            raise InspectionError(f"unsupported selected tree member: {path}")
        size = getattr(item, "size", None)
        blob = getattr(item, "blob_id", None) or getattr(item, "oid", None)
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise InspectionError(f"invalid selected server size: {path}")
        if not isinstance(blob, str) or not re.fullmatch(r"[0-9a-f]{40}", blob):
            raise InspectionError(f"missing selected Git blob identity: {path}")
        lfs = getattr(item, "lfs", None)
        lfs_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        lfs_size = lfs.get("size") if isinstance(lfs, dict) else getattr(lfs, "size", None)
        if lfs_sha is not None:
            if not isinstance(lfs_sha, str) or not HEX64.fullmatch(lfs_sha):
                raise InspectionError(f"invalid selected LFS SHA-256: {path}")
            if lfs_size != size:
                raise InspectionError(f"selected LFS size mismatch: {path}")
            pointer = (
                "version https://git-lfs.github.com/spec/v1\n"
                f"oid sha256:{lfs_sha}\nsize {size}\n"
            ).encode()
            pointer_blob = hashlib.sha1(
                f"blob {len(pointer)}\0".encode() + pointer
            ).hexdigest()
            if pointer_blob != blob:
                raise InspectionError(f"selected LFS pointer Git blob mismatch: {path}")
            row = {
                "path": path,
                "size": size,
                "git_blob_sha1": None,
                "lfs_pointer_git_blob_sha1": blob,
                "lfs_payload_sha256": lfs_sha,
            }
        else:
            row = {
                "path": path,
                "size": size,
                "git_blob_sha1": blob,
                "lfs_pointer_git_blob_sha1": None,
                "lfs_payload_sha256": None,
            }
        if path in selected:
            raise InspectionError(f"duplicate selected server path: {path}")
        selected[path] = row
    missing = [name for name in PAYLOAD_FILES if name not in selected]
    if missing:
        raise InspectionError(f"selected files missing from server tree: {missing}")
    return resolved, selected


def shape(value: Any) -> str:
    if not hasattr(value, "shape"):
        raise InspectionError(f"tap is not shape-bearing: {type(value)!r}")
    axes = tuple(int(axis) for axis in value.shape)
    if not axes or any(axis <= 0 for axis in axes):
        raise InspectionError(f"invalid tap shape: {axes!r}")
    return "x".join(str(axis) for axis in axes)


def reserve_output(path: Path) -> None:
    """Reserve a fresh evidence directory without ever clobbering output."""

    if path.exists() or path.is_symlink():
        raise InspectionError(f"refusing pre-existing output: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise InspectionError(f"output parent is not a real directory: {path.parent}")
    try:
        path.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise InspectionError(f"refusing pre-existing output: {path}") from error


def api_and_shape_probe(snapshot: Path) -> dict[str, Any]:
    """Reject the unverified model-construction route before source access."""

    raise InspectionError(
        f"{SECURITY_ADVISORY}: official AutoModel meta construction has not "
        "been revalidated without the former meta-device helper on VAST"
    )



def blocked_manifest(
    *,
    repository: str,
    revision: str,
    resolved_revision: str | None,
    model_info: dict[str, Any] | None,
    files: list[dict[str, Any]] | None,
    config: dict[str, Any] | None,
    index: dict[str, Any] | None,
    route: dict[str, Any],
    vokra_checkout: dict[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    complete = False
    return {
        "schema": "vokra-moss-audio-tokenizer-nano-source-contract-v1",
        "status": "BLOCKED",
        "evidence_stage": "INSPECTION_ONLY",
        "inspection_status": (
            "AUTHENTICATED_EVIDENCE_COMPLETE" if complete else "INSPECTION_ERROR"
        ),
        "collection_status": "AUTHENTICATED" if complete else "INCOMPLETE",
        "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
        "cpu_status": "BLOCKED_UNRESOLVED_PYTHON_CLOSURE_API_RUNTIME_PARITY",
        "metal_status": "BLOCKED_BY_CPU",
        "parity_status": "NOT_RUN",
        "publication": "NO_UPLOAD",
        "vokra_checkout": vokra_checkout,
        "model": {
            "repository": repository,
            "requested_revision": revision,
            "resolved_revision": resolved_revision,
            "model_info": model_info,
        },
        "files": files,
        "config_contract": config,
        "checkpoint_index": index,
        "transformers_route": route,
        "license": {
            "status": "REVIEWED",
            "source_and_weight_review": "REVIEWED",
            "source_and_weight_license": "Apache-2.0",
            "source_and_weight_conclusion": "Commercial",
            "owner_signoff": "2026-08-01 yousan",
            "owner_signoff_citation": "docs/license-audit.md:671",
            "license_file_present": False,
            "hf_cardData_license": (
                model_info.get("cardData_license") if model_info is not None else None
            ),
        },
        "reference_contract": {
            "frames": 2,
            "quantizers": 16,
            "status": route.get("status"),
            "tap_shapes": route.get("taps"),
            "audio_shape": route.get("audio_shape"),
            "numeric_parity": "NOT_RUN",
        },
        "unresolved_gates": {
            "python_dependency_closure": "UNRESOLVED",
            "transformers_api_compatibility": "UNRESOLVED",
            "real_weight_runtime": "UNRESOLVED",
            "numerical_parity": "NOT_RUN",
            "overall_execution_approval": "NOT_APPROVED",
        },
        "error": error,
    }


def self_test() -> None:
    assert safe_relative_path("config.json") == "config.json"
    assert "LICENSE" not in MATERIALIZED_FILES
    assert ".gitattributes" in MATERIALIZED_FILES
    assert "__init__.py" in MATERIALIZED_FILES
    assert validate_model_info(EXPECTED_MODEL_INFO) == EXPECTED_MODEL_INFO
    assert EXPECTED_AUTO_MAP["AutoModel"].endswith("MossAudioTokenizerModel")
    assert len(EXPECTED_DECODER_LAYOUT) == 9
    assert EXPECTED_TAPS[-1] == {"name": "decoder_8", "shape": "1x1x15360"}
    try:
        validate_model_info({**EXPECTED_MODEL_INFO, "gated": True})
    except InspectionError:
        pass
    else:
        raise AssertionError("tampered HF model_info was accepted")
    sample_manifest = blocked_manifest(
        repository=REPOSITORY,
        revision=REVISION,
        resolved_revision=REVISION,
        model_info=EXPECTED_MODEL_INFO,
        files=[],
        config=None,
        index=None,
        route={"status": SECURITY_ADVISORY},
        vokra_checkout={"expected_head": "a" * 40, "head": "a" * 40, "clean": True},
        error=None,
    )
    assert sample_manifest["vokra_checkout"] == {
        "expected_head": "a" * 40,
        "head": "a" * 40,
        "clean": True,
    }
    assert sample_manifest["license"] == {
        "status": "REVIEWED",
        "source_and_weight_review": "REVIEWED",
        "source_and_weight_license": "Apache-2.0",
        "source_and_weight_conclusion": "Commercial",
        "owner_signoff": "2026-08-01 yousan",
        "owner_signoff_citation": "docs/license-audit.md:671",
        "license_file_present": False,
        "hf_cardData_license": "apache-2.0",
    }
    assert sample_manifest["cpu_status"] == "BLOCKED_UNRESOLVED_PYTHON_CLOSURE_API_RUNTIME_PARITY"
    assert sample_manifest["unresolved_gates"]["overall_execution_approval"] == "NOT_APPROVED"
    for bad_head in ("", "0" * 39, "G" * 40):
        try:
            validate_vokra_checkout(Path.cwd(), bad_head)
        except InspectionError:
            pass
        else:
            raise AssertionError(f"invalid expected checkout HEAD accepted: {bad_head!r}")
    with tempfile.TemporaryDirectory() as temporary:
        checkout = Path(temporary) / "checkout"
        checkout.mkdir()
        def git(*arguments: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                ["git", "-C", str(checkout), *arguments],
                check=True,
                capture_output=True,
                text=True,
            )
        git("init", "--quiet")
        git("config", "user.name", "MOSS Nano self-test")
        git("config", "user.email", "moss-nano-self-test@example.invalid")
        tracked = checkout / "tracked.txt"
        tracked.write_text("clean\n", encoding="utf-8")
        git("add", "tracked.txt")
        git("commit", "--quiet", "--no-gpg-sign", "-m", "initial")
        clean_head = git("rev-parse", "HEAD").stdout.strip()
        assert validate_vokra_checkout(checkout, clean_head) == {
            "expected_head": clean_head,
            "head": clean_head,
            "clean": True,
        }
        try:
            validate_vokra_checkout(checkout, "0" * 40)
        except InspectionError as error:
            assert "HEAD mismatch" in str(error)
        else:
            raise AssertionError("wrong valid checkout HEAD was accepted")
        tracked.write_text("dirty\n", encoding="utf-8")
        try:
            validate_vokra_checkout(checkout, clean_head)
        except InspectionError as error:
            assert "not clean" in str(error)
        else:
            raise AssertionError("dirty tracked checkout was accepted")
        tracked.write_text("clean\n", encoding="utf-8")
        (checkout / "untracked.txt").write_text("untracked\n", encoding="utf-8")
        try:
            validate_vokra_checkout(checkout, clean_head)
        except InspectionError as error:
            assert "not clean" in str(error)
        else:
            raise AssertionError("dirty untracked checkout was accepted")
        symlink = Path(temporary) / "checkout-link"
        symlink.symlink_to(checkout, target_is_directory=True)
        try:
            validate_vokra_checkout(symlink, clean_head)
        except InspectionError as error:
            assert "real directory" in str(error)
        else:
            raise AssertionError("symlink checkout path was accepted")
    for bad in ("", "/config.json", "../config.json", "a\\b", "a\x00b"):
        try:
            safe_relative_path(bad)
        except InspectionError:
            pass
        else:
            raise AssertionError(f"unsafe path accepted: {bad!r}")
    source = Path(__file__).read_text(encoding="utf-8")
    assert ("AutoModel." + "from_pretrained") not in source
    assert SECURITY_ADVISORY in source
    assert '"weights_loaded": False' in source
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "evidence"
        reserve_output(output)
        try:
            reserve_output(output)
        except InspectionError as error:
            assert "refusing pre-existing output" in str(error)
        else:
            raise AssertionError("pre-existing output was accepted")
    with tempfile.TemporaryDirectory() as temporary:
        snapshot = Path(temporary)
        config = {
            **EXPECTED_CONFIG,
            "model_type": EXPECTED_MODEL_TYPE,
            "architectures": EXPECTED_ARCHITECTURES,
            "auto_map": EXPECTED_AUTO_MAP,
            "quantizer_kwargs": EXPECTED_QUANTIZER,
            "decoder_kwargs": EXPECTED_DECODER_LAYOUT,
        }
        (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": WEIGHT_FILE}}), encoding="utf-8"
        )
        for name in MATERIALIZED_FILES:
            path = snapshot / name
            if not path.exists():
                path.write_bytes(name.encode("utf-8"))
        server_rows = {}
        for name in MATERIALIZED_FILES:
            path = snapshot / name
            server_rows[name] = {
                "path": name,
                "size": path.stat().st_size,
                "git_blob_sha1": file_hashes(path)[1],
                "lfs_pointer_git_blob_sha1": None,
                "lfs_payload_sha256": None,
            }
        server_rows[WEIGHT_FILE] = {
            "path": WEIGHT_FILE,
            "size": 123,
            "git_blob_sha1": None,
            "lfs_pointer_git_blob_sha1": "1" * 40,
            "lfs_payload_sha256": "2" * 64,
        }
        rows, _, _ = validate_snapshot(snapshot, server_rows)
        assert len(rows) == len(PAYLOAD_FILES)
        assert rows[-1]["materialized"] is False
        assert rows[-1]["content_not_downloaded"] is True
        for label, mutate in (
            ("auto-map", lambda value: value.update(auto_map={"AutoModel": "wrong.Model"})),
            ("decoder-layout", lambda value: value["decoder_kwargs"][0].update(patch_size=8)),
        ):
            tampered_config = json.loads(json.dumps(config))
            mutate(tampered_config)
            (snapshot / "config.json").write_text(
                json.dumps(tampered_config), encoding="utf-8"
            )
            try:
                validate_snapshot(snapshot, server_rows)
            except InspectionError as error:
                assert "config.json" in str(error), label
            else:
                raise AssertionError(f"tampered {label} config was accepted")
        (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")
        mismatch_rows = {name: dict(row) for name, row in server_rows.items()}
        mismatch_rows["config.json"]["git_blob_sha1"] = "0" * 40
        try:
            validate_snapshot(snapshot, mismatch_rows)
        except InspectionError as error:
            assert "Git blob SHA-1 mismatch" in str(error)
        else:
            raise AssertionError("materialized Git blob mismatch was accepted")
        (snapshot / WEIGHT_FILE).write_bytes(b"forbidden")
        try:
            validate_snapshot(snapshot, server_rows)
        except InspectionError as error:
            assert "weight payload was materialized" in str(error)
        else:
            raise AssertionError("materialized weight payload was accepted")
    print("moss Nano source-contract inspector self-test: PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", default=REPOSITORY)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--server-tree", type=Path)
    parser.add_argument("--vokra-root", type=Path)
    parser.add_argument("--expected-head")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.snapshot, args.output, args.server_tree, args.vokra_root, args.expected_head)):
            parser.error("--self-test accepts no filesystem arguments")
        return args
    if args.repository != REPOSITORY or args.revision != REVISION:
        parser.error("Nano repository and revision are immutable")
    if args.snapshot is None or args.output is None or args.server_tree is None or args.vokra_root is None or args.expected_head is None:
        parser.error("--snapshot, --server-tree, --output, --vokra-root, and --expected-head are required")
    if not HEX40.fullmatch(args.expected_head):
        parser.error("--expected-head requires lowercase 40-hex")
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    print(
        f"moss Nano source-contract inspector: BLOCKED: {SECURITY_ADVISORY}: "
        "the official AutoModel meta route requires VAST revalidation before "
        "source or weight access",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
