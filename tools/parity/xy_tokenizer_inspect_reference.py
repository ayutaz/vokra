#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Safely inventory the official XY-Tokenizer checkpoint.

The upstream artifact is a large PyTorch ``.ckpt``.  This tool is VAST-only:
it accepts only ``torch.load(..., weights_only=True)``, emits a safetensors
replacement for the Rust converter, and records an inspection-only manifest.
It never imports or copies the source implementation and never claims a
runtime or numerical result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import stat
import tempfile
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from safetensors.torch import save_file

UPSTREAM_REPOSITORY = "OpenMOSS-Team/XY_Tokenizer_TTSD_V0"
UPSTREAM_REVISION = "c83433728e698ed0698e88cb5096bc221fb8f8c5"
CHECKPOINT_BYTES = 2_137_328_977
CHECKPOINT_SHA256 = "37c7ac18d0a48f5a1d0687e31af7c0264861232c500206718c98acd8e37d1671"
CONFIG_RELATIVE = Path("config/xy_tokenizer_config.yaml")
CHECKPOINT_RELATIVE = Path("xy_tokenizer.ckpt")
CONFIG_SHA256 = "e7d48677e34f77e5b9fd7dc7a3e0eef7f2d2dd9be9a245d5c1d56489dc748938"
SOURCE_REPOSITORY = "https://github.com/gyt1145028706/XY-Tokenizer"
SOURCE_REVISION = "5df5609c5883e555bd39a2d0b1005ca8f1a8f12e"
FORMAT = "vokra-xy-tokenizer-prepared-v1"

# These are the independently reviewed Git blob identities for the fixed
# implementation checkout.  The source_inventory() checks these against the
# index, HEAD, and streamed working-tree blobs; a source-created manifest is
# never accepted as evidence of implementation identity.
SOURCE_ROLE_BLOBS: dict[str, str] = {
    "config/xy_tokenizer_config.yaml": "83c50a60b3c0db62ce30b9cd65e0b0f5cd290f89",
    "inference.py": "9bb00a176f878d872f8eb7ed7a98501d3abb7e70",
    "inference_for_codec_evaluation.py": "4a98524ac90506a21b6155b31e945163c5d35d5b",
    "requirements.txt": "46b7b2d2aabb074ce87433eba2f55b31eee2363b",
    "utils/helpers.py": "9b144a4ce5ca6fd57b1a2903d940c4b4ffec4d97",
    "xy_tokenizer/model.py": "188f1b607d3e9a5953b3015ea9d262008ef535c0",
    "xy_tokenizer/nn/feature_extractor.py": "4d397b012ffe756fa9dfadc771f81e0afddd3963",
    "xy_tokenizer/nn/modules.py": "cc186d9dadd674172837d527fef0f0de183feb4c",
    "xy_tokenizer/nn/quantizer.py": "a7d28b963e98ea4f62f2a6e06b419cf0da0c2cc4",
}
SELECTED_MODEL_FILES = {".gitattributes", "README.md", CHECKPOINT_RELATIVE.as_posix(), CONFIG_RELATIVE.as_posix()}
SOURCE_LICENSE_ABSENT_BLOCKER = "SOURCE_LICENSE_ABSENT_BLOCKER"
TOPOLOGY_UNVERIFIED_BLOCKER = "TOPOLOGY_CONTRACT_UNVERIFIED_BLOCKER"
EVIDENCE_FILENAME = "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_load_unique(path: Path) -> Any:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)


def regular_files(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*")):
        parts = path.relative_to(root).parts
        if parts == (".cache",) or len(parts) >= 2 and parts[:2] == (".cache", "huggingface"):
            continue
        if any(part in {".cache", ".git"} for part in parts):
            raise RuntimeError(f"unauthenticated metadata path: {path}")
        if path.is_dir() and not path.is_symlink():
            continue
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"payload member is not a regular file: {path}")
        files.append(path)
    return files


def validate_server_packet(snapshot: Path, packet_path: Path) -> dict[str, Any]:
    packet = json_load_unique(packet_path)
    if not isinstance(packet, dict) or set(packet) != {"repository", "requested_revision", "resolved_revision", "files"}:
        raise RuntimeError("HF server packet schema mismatch")
    if packet["repository"] != UPSTREAM_REPOSITORY or packet["requested_revision"] != UPSTREAM_REVISION or packet["resolved_revision"] != UPSTREAM_REVISION:
        raise RuntimeError("HF revision/repository mismatch")
    rows = packet["files"]
    if not isinstance(rows, list) or [row.get("path") for row in rows if isinstance(row, dict)] != sorted(row.get("path") for row in rows if isinstance(row, dict)):
        raise RuntimeError("HF server packet file ordering/schema mismatch")
    by_path: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("type") != "file":
            raise RuntimeError("HF recursive tree contains an invalid entry")
        path = row.get("path")
        if not isinstance(path, str) or not path or Path(path).is_absolute() or "\\" in path or "\x00" in path or ".." in Path(path).parts or path in by_path:
            raise RuntimeError("HF server path is unsafe or duplicated")
        keys = set(row)
        base = {"path", "type", "size", "git_blob_sha1"}
        lfs = base | {"lfs_sha256", "lfs_size", "lfs_pointer_sha1"}
        if keys != base and keys != lfs:
            raise RuntimeError("HF server row has incomplete/extra identity keys")
        if isinstance(row["size"], bool) or not isinstance(row["size"], int) or row["size"] < 0:
            raise RuntimeError("HF server row size is invalid")
        if not isinstance(row["git_blob_sha1"], str) or len(row["git_blob_sha1"]) != 40 or any(c not in "0123456789abcdef" for c in row["git_blob_sha1"]):
            raise RuntimeError("HF server Git identity is invalid")
        if keys == lfs:
            if not isinstance(row["lfs_sha256"], str) or len(row["lfs_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["lfs_sha256"]):
                raise RuntimeError("HF server LFS identity is invalid")
            if isinstance(row["lfs_size"], bool) or not isinstance(row["lfs_size"], int) or row["lfs_size"] != row["size"]:
                raise RuntimeError("HF server LFS size is invalid")
            if not isinstance(row["lfs_pointer_sha1"], str) or len(row["lfs_pointer_sha1"]) != 40:
                raise RuntimeError("HF server LFS pointer identity is invalid")
        by_path[path] = row
    local = {path.relative_to(snapshot).as_posix(): path for path in regular_files(snapshot)}
    for relative in SELECTED_MODEL_FILES:
        if relative not in by_path or relative not in local:
            raise RuntimeError(f"selected HF file missing: {relative}")
        row, path = by_path[relative], local[relative]
        if row["size"] != path.stat().st_size:
            raise RuntimeError(f"HF file size mismatch: {relative}")
        if set(row) == {"path", "type", "size", "git_blob_sha1"}:
            if row["git_blob_sha1"] != git_blob_sha1(path):
                raise RuntimeError(f"HF Git blob mismatch: {relative}")
        else:
            observed = sha256(path)
            pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{observed}\nsize {path.stat().st_size}\n".encode()
            pointer_sha = hashlib.sha1(f"blob {len(pointer)}\0".encode() + pointer).hexdigest()
            if row["lfs_sha256"] != observed or row["lfs_pointer_sha1"] != pointer_sha or row["git_blob_sha1"] != pointer_sha:
                raise RuntimeError(f"HF LFS identity mismatch: {relative}")
    return {"repository": packet["repository"], "requested_revision": packet["requested_revision"], "resolved_revision": packet["resolved_revision"], "server_file_count": len(by_path), "selected_files": sorted(SELECTED_MODEL_FILES)}


class StrictYamlLoader(yaml.SafeLoader):
    pass


def _mapping(loader: StrictYamlLoader, node: yaml.MappingNode, deep: bool = False) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key_node, item_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in value:
            raise ValueError(f"duplicate/invalid YAML key: {key!r}")
        value[key] = loader.construct_object(item_node, deep=deep)
    return value


StrictYamlLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def parse_config(text: str) -> dict[str, Any]:
    if any(marker in text for marker in ("!", "&", "*")):
        raise ValueError("YAML aliases/tags are not accepted")
    try:
        value = yaml.load(text, Loader=StrictYamlLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"strict YAML parse failed: {error}") from error
    if not isinstance(value, dict) or set(value) != {"audio_tokenizer"}:
        raise ValueError("config schema is not canonical")
    # The fixed config bytes are authenticated, but no independently reviewed
    # topology extraction is present in this checkout. Preserve the parsed
    # bytes for evidence while carrying the blocker; never promote guessed
    # fields or the old self-declared TOPOLOGY_CONTRACT.
    return {"raw": value, "topology_status": TOPOLOGY_UNVERIFIED_BLOCKER}


def parse_weight_license(readme: Path) -> dict[str, str]:
    text = readme.read_text(encoding="utf-8")
    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        raise RuntimeError("HF README lacks authenticated top-level frontmatter")
    end = text.find("\n---\n", 4)
    try:
        frontmatter = yaml.load(text[4:end], Loader=StrictYamlLoader)
    except yaml.YAMLError as error:
        raise RuntimeError(f"HF README frontmatter invalid: {error}") from error
    if not isinstance(frontmatter, dict) or set(frontmatter) != {"license"} or frontmatter["license"] != "apache-2.0":
        raise RuntimeError("HF weight license declaration is not exact apache-2.0")
    return {"spdx": "Apache-2.0", "basis": "authenticated HF README top-level frontmatter", "policy": "WEIGHT_ONLY_SOURCE_LICENSE_UNCONFIRMED"}


def require_file(path: Path, expected_sha256: str, label: str, expected_bytes: int | None = None) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    actual_sha256 = sha256(path)
    actual_bytes = path.stat().st_size
    if actual_sha256 != expected_sha256 or (expected_bytes is not None and actual_bytes != expected_bytes):
        raise RuntimeError(f"{label} identity mismatch: bytes={actual_bytes} sha256={actual_sha256}")


def state_dict_from_checkpoint(value: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(value, Mapping) and all(isinstance(key, str) and isinstance(tensor, torch.Tensor) for key, tensor in value.items()):
        return value
    if isinstance(value, Mapping):
        for key in ("state_dict", "model", "module"):
            nested = value.get(key)
            if isinstance(nested, Mapping) and all(isinstance(name, str) and isinstance(tensor, torch.Tensor) for name, tensor in nested.items()):
                if set(value) != {key}:
                    raise RuntimeError("safe checkpoint contains untrusted metadata beside tensor state")
                return nested
    raise RuntimeError("safe checkpoint did not contain a tensor state dict")


def validate_state_dict(state: Mapping[str, torch.Tensor]) -> None:
    if not state:
        raise RuntimeError("safe checkpoint state dict is empty")
    for name, tensor in state.items():
        if (not isinstance(name, str) or not name or "\x00" in name
                or any(part in {"", ".", ".."} for part in name.split("."))):
            raise RuntimeError(f"unsafe state-dict tensor key: {name!r}")
        allowed_dtypes = {torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32, torch.float64}
        if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                or tensor.is_quantized or tensor.dtype not in allowed_dtypes):
            raise RuntimeError(f"unsupported tensor layout at {name}")
        if any(isinstance(axis, bool) or not isinstance(axis, int) or axis < 0 for axis in tensor.shape):
            raise RuntimeError(f"invalid tensor shape at {name}")
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(torch.isfinite(tensor).all().item()):
            raise RuntimeError(f"non-finite tensor at {name}")


def raw_tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Return the exact dense storage bytes, including for scalar tensors."""
    if tensor.layout != torch.strided or tensor.is_quantized:
        raise RuntimeError(f"unsupported tensor layout/dtype: layout={tensor.layout} dtype={tensor.dtype}")
    try:
        dense = tensor.detach().cpu().contiguous()
        return dense.reshape(-1).view(torch.uint8).numpy().tobytes()
    except (RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(f"cannot obtain raw bytes for dtype={tensor.dtype} shape={tuple(tensor.shape)}") from error


def source_inventory(source: Path) -> dict[str, Any]:
    if not (source / ".git").exists():
        raise RuntimeError("source checkout lacks .git metadata")
    head = git(source, "rev-parse", "HEAD")
    origin = git(source, "remote", "get-url", "origin").removesuffix("/").removesuffix(".git")
    if head != SOURCE_REVISION or origin != SOURCE_REPOSITORY:
        raise RuntimeError("source HEAD/origin identity mismatch")
    if git(source, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("source checkout is dirty")
    entries: dict[str, tuple[str, str]] = {}
    for record in git(source, "ls-files", "-s", "-z").split("\0"):
        if not record:
            continue
        metadata, relative = record.split("\t", 1)
        mode, object_id, stage = metadata.split()
        if mode not in {"100644", "100755"} or stage != "0" or relative in entries:
            raise RuntimeError(f"source tracked mode/stage/path mismatch: {relative}")
        path = source / relative
        if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) != {"100644": 0o644, "100755": 0o755}[mode]:
            raise RuntimeError(f"source tracked file/mode mismatch: {relative}")
        head_object = git(source, "rev-parse", f"HEAD:{relative}")
        working_object = git_blob_sha1(path)
        if object_id != head_object or object_id != working_object:
            raise RuntimeError(f"source tracked object mismatch: {relative}")
        entries[relative] = (mode, object_id)
    roles = []
    role_status = "AUTHENTICATED"
    if not SOURCE_ROLE_BLOBS:
        role_status = "SOURCE_ROLE_BLOBS_UNVERIFIED_BLOCKER"
    else:
        for relative, expected in SOURCE_ROLE_BLOBS.items():
            mode_object = entries.get(relative)
            if mode_object is None or mode_object[0] != "100644" or mode_object[1] != expected:
                raise RuntimeError(f"source fixed role mismatch: {relative}")
            roles.append({"path": relative, "mode": mode_object[0], "git_blob_sha1": expected})
    license_path = next((source / name for name in ("LICENSE", "LICENSE.md", "COPYING") if (source / name).is_file()), None)
    license_status = SOURCE_LICENSE_ABSENT_BLOCKER if license_path is None else "PRESENT_UNREVIEWED"
    return {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION, "worktree_status": "CLEAN", "roles": roles, "role_status": role_status, "license_status": license_status}


def inspect(checkpoint: Path, config: Path, source: Path, prepared: Path, output: Path, server_packet: Path) -> None:
    server = validate_server_packet(checkpoint.parent, server_packet)
    require_file(checkpoint, CHECKPOINT_SHA256, "XY-Tokenizer checkpoint", CHECKPOINT_BYTES)
    require_file(config, CONFIG_SHA256, "XY-Tokenizer config")
    weight_license = parse_weight_license(checkpoint.parent / "README.md")
    source_data = source_inventory(source)
    if source_data["role_status"] != "AUTHENTICATED":
        raise RuntimeError(source_data["role_status"])
    config_data = parse_config(config.read_text(encoding="utf-8"))
    known_blockers: list[str] = []
    if source_data["license_status"] == SOURCE_LICENSE_ABSENT_BLOCKER:
        known_blockers.append(SOURCE_LICENSE_ABSENT_BLOCKER)
    if config_data.get("topology_status") == TOPOLOGY_UNVERIFIED_BLOCKER:
        known_blockers.append(TOPOLOGY_UNVERIFIED_BLOCKER)

    # The only checkpoint load permitted by this inspection path. A failure is
    # fatal; there is intentionally no unrestricted pickle fallback.
    checkpoint_value = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    state = state_dict_from_checkpoint(checkpoint_value)
    validate_state_dict(state)
    prepared.parent.mkdir(parents=True, exist_ok=True)
    save_file({name: tensor.detach().cpu().contiguous() for name, tensor in state.items()}, str(prepared))

    tensors: list[dict[str, Any]] = []
    dtype_counts: dict[str, int] = {}
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        raw = raw_tensor_bytes(tensor)
        dtype = str(tensor.dtype).removeprefix("torch.")
        tensors.append({"name": name, "shape": [int(dim) for dim in tensor.shape], "dtype": dtype, "elements": int(tensor.numel()), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    output.mkdir(parents=True, exist_ok=True)
    tensor_manifest = {"tensor_count": len(tensors), "dtype_counts": dtype_counts, "tensors": tensors}
    (output / "tensor-inventory.json").write_text(json.dumps(tensor_manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "config.json").write_text(json.dumps(config_data, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "source-inventory.json").write_text(json.dumps(source_data, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "server-packet.json").write_bytes(server_packet.read_bytes())
    manifest = {
        "format": FORMAT,
        "status": "BLOCKED",
        "evidence_stage": "INSPECTION_ONLY",
        "inspection_status": "AUTHENTICATED_EVIDENCE_COMPLETE",
        "collection_status": "AUTHENTICATED",
        "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
        "cpu_status": "UNSUPPORTED",
        "metal_status": "BLOCKED_BY_CPU",
        "parity_status": "NOT_RUN",
        "publication": "NO_UPLOAD",
        "upstream": {"repository": UPSTREAM_REPOSITORY, "revision": UPSTREAM_REVISION, "checkpoint_sha256": CHECKPOINT_SHA256, "checkpoint_bytes": CHECKPOINT_BYTES},
        "repository": UPSTREAM_REPOSITORY,
        "revision": UPSTREAM_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "official_source": source_data,
        "server_tree": server,
        "source_revision": SOURCE_REVISION,
        "config_sha256": CONFIG_SHA256,
        "prepared": {"path": prepared.name, "bytes": prepared.stat().st_size, "sha256": sha256(prepared)},
        "tensor_count": len(tensors),
        "dtype_counts": dtype_counts,
        "weight_license": weight_license,
        "blockers": known_blockers + ["native/runtime implementation is not available", "numerical parity is NOT_RUN", "publication is NO_UPLOAD"],
        "packets": {
            name: {"bytes": (output / name).stat().st_size, "sha256": sha256(output / name)}
            for name in ("tensor-inventory.json", "config.json", "source-inventory.json", "server-packet.json")
        },
    }
    (output / EVIDENCE_FILENAME).write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def write_error_manifest(output: Path, error: Exception) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in ("tensor-inventory.json", "config.json", "source-inventory.json", "server-packet.json", EVIDENCE_FILENAME):
        path = output / name
        if path.exists() and path.is_file():
            path.unlink()
    manifest = {"format": FORMAT, "status": "BLOCKED", "evidence_stage": "INSPECTION_ONLY", "inspection_status": "INSPECTION_ERROR", "collection_status": "FAILED", "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "cpu_status": "UNSUPPORTED", "metal_status": "BLOCKED_BY_CPU", "parity_status": "NOT_RUN", "publication": "NO_UPLOAD", "error": str(error), "blockers": ["authenticated collection unavailable", SOURCE_LICENSE_ABSENT_BLOCKER, TOPOLOGY_UNVERIFIED_BLOCKER]}
    (output / EVIDENCE_FILENAME).write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def self_test() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    assert hashlib.sha256(b"abc").hexdigest() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert FORMAT == "vokra-xy-tokenizer-prepared-v1"
    assert "weights_only=True" in source
    assert "safetensors.torch" in source
    assert SOURCE_LICENSE_ABSENT_BLOCKER in source and TOPOLOGY_UNVERIFIED_BLOCKER in source
    scalar_int64 = torch.tensor(0x0102030405060708, dtype=torch.int64)
    assert raw_tensor_bytes(scalar_int64) == bytes.fromhex("0807060504030201")
    scalar_float = torch.tensor(-2.5, dtype=torch.float32)
    assert raw_tensor_bytes(scalar_float) == bytes.fromhex("000020c0")
    scalar_bf16 = torch.tensor(-16352, dtype=torch.int16).view(torch.bfloat16)
    assert raw_tensor_bytes(scalar_bf16) == bytes.fromhex("20c0")
    for tensor, expected in (
        (scalar_int64, "380b980886b7a3e726b5c2776160d7842b8f139eaea1e6acf7442cdf890e6287"),
        (scalar_float, "2f90de89a933bd8118953f22550293198c812014b434e29d79bcdea4cb34c56a"),
        (scalar_bf16, "97ee14790cbe8239c38f70bb097ca6f6de1794c846d28376c2de6625c8ad6e53"),
    ):
        assert hashlib.sha256(raw_tensor_bytes(tensor)).hexdigest() == expected
    assert SOURCE_ROLE_BLOBS == {
        "config/xy_tokenizer_config.yaml": "83c50a60b3c0db62ce30b9cd65e0b0f5cd290f89",
        "inference.py": "9bb00a176f878d872f8eb7ed7a98501d3abb7e70",
        "inference_for_codec_evaluation.py": "4a98524ac90506a21b6155b31e945163c5d35d5b",
        "requirements.txt": "46b7b2d2aabb074ce87433eba2f55b31eee2363b",
        "utils/helpers.py": "9b144a4ce5ca6fd57b1a2903d940c4b4ffec4d97",
        "xy_tokenizer/model.py": "188f1b607d3e9a5953b3015ea9d262008ef535c0",
        "xy_tokenizer/nn/feature_extractor.py": "4d397b012ffe756fa9dfadc771f81e0afddd3963",
        "xy_tokenizer/nn/modules.py": "cc186d9dadd674172837d527fef0f0de183feb4c",
        "xy_tokenizer/nn/quantizer.py": "a7d28b963e98ea4f62f2a6e06b419cf0da0c2cc4",
    }
    config_data = parse_config("audio_tokenizer: {}\n")
    assert config_data["topology_status"] == TOPOLOGY_UNVERIFIED_BLOCKER
    try:
        parse_config("audio_tokenizer: {}\naudio_tokenizer: {}\n")
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate YAML key was accepted")
    try:
        parse_config("audio_tokenizer: &alias {}\n")
    except ValueError:
        pass
    else:
        raise AssertionError("YAML alias was accepted")
    bad_state = {"layer..weight": torch.ones(1)}
    try:
        validate_state_dict(bad_state)
    except RuntimeError:
        pass
    else:
        raise AssertionError("unsafe state-dict key was accepted")
    try:
        state_dict_from_checkpoint({"state_dict": {"layer.weight": torch.ones(1)}, "epoch": 1})
    except RuntimeError:
        pass
    else:
        raise AssertionError("checkpoint metadata beside state dict was accepted")
    with tempfile.TemporaryDirectory(prefix="vokra-xy-tokenizer-error-") as directory:
        error_dir = Path(directory)
        write_error_manifest(error_dir, RuntimeError("self-test error"))
        error_manifest = json_load_unique(error_dir / EVIDENCE_FILENAME)
        assert error_manifest["status"] == "BLOCKED"
        assert error_manifest["inspection_status"] == "INSPECTION_ERROR"
        assert error_manifest["collection_status"] == "FAILED"
        normal_contract = {
            "status": "BLOCKED",
            "evidence_stage": "INSPECTION_ONLY",
            "inspection_status": "AUTHENTICATED_EVIDENCE_COMPLETE",
            "collection_status": "AUTHENTICATED",
            "publication": "NO_UPLOAD",
        }
        assert normal_contract["inspection_status"] != error_manifest["inspection_status"]
        assert normal_contract["collection_status"] != error_manifest["collection_status"]
    with tempfile.TemporaryDirectory(prefix="vokra-xy-tokenizer-packet-") as directory:
        model = Path(directory)
        for relative in sorted(SELECTED_MODEL_FILES):
            target = model / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(relative.encode())
        rows = []
        for relative in sorted(SELECTED_MODEL_FILES):
            target = model / relative
            rows.append({"path": relative, "type": "file", "size": target.stat().st_size, "git_blob_sha1": git_blob_sha1(target)})
        rows.append({"path": "extra/source.py", "type": "file", "size": 1, "git_blob_sha1": "0" * 40})
        rows.sort(key=lambda row: row["path"])
        packet = model / "packet.json"
        packet.write_text(json.dumps({"repository": UPSTREAM_REPOSITORY, "requested_revision": UPSTREAM_REVISION, "resolved_revision": UPSTREAM_REVISION, "files": rows}), encoding="utf-8")
        assert validate_server_packet(model, packet)["server_file_count"] == 5
    print("xy_tokenizer_inspect_reference.py self-test: OK (safe-load/source/hash contracts)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--server-packet", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.checkpoint, args.config, args.source, args.prepared, args.output, args.server_packet)):
            parser.error("--self-test accepts no other arguments")
        self_test()
        return 0
    if any(value is None for value in (args.checkpoint, args.config, args.source, args.prepared, args.output, args.server_packet)):
        parser.error("--checkpoint, --config, --source, --prepared, --output, and --server-packet are required")
    try:
        inspect(args.checkpoint, args.config, args.source, args.prepared, args.output, args.server_packet)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        write_error_manifest(args.output, error)
        print(f"XY-Tokenizer inspection: {error}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"XY-Tokenizer inspection: {error}", file=sys.stderr)
        raise SystemExit(2)
