#!/usr/bin/env -S uv run --script
"""Offline-only authenticated AudioLDM2 bundle inspection contract.

This inspector never downloads weights or runs inference.  It verifies the
fixed server-tree object packet and emits an inspection-only status; a missing lock,
source checkout, or model tree is a hard failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Any

BASE_REPOSITORY = "cvssp/audioldm2"
BASE_REVISION = "c8e7e189d324425c05c4c2f81214041ef4107983"
LARGE_REPOSITORY = "cvssp/audioldm2-large"
LARGE_REVISION = "4b0b875a9e0c5305dfc917da808584e50e1c7ed4"
SOURCE_REPOSITORY = "huggingface/diffusers"
SOURCE_ORIGIN = "https://github.com/huggingface/diffusers.git"
SOURCE_COMMIT = "29f15673ed5c14e4843d7c837890910207f72129"
SOURCE_TAG = "v0.21.0"
SOURCE_ROLE_BLOBS = {
    "LICENSE": "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64",
    "setup.py": "6b3492776df311e446e088175c104e269bc1d384",
    "src/diffusers/__init__.py": "5ebab88a2a7442ebd4d581177b382ee67f9e7531",
    "src/diffusers/pipelines/audioldm2/__init__.py": "50330c6774525e713355a79089d78e455ecdb8b9",
    "src/diffusers/pipelines/audioldm2/modeling_audioldm2.py": "d39b2c99ddd035544c99fd9357ec8cd6205e79c8",
    "src/diffusers/pipelines/audioldm2/pipeline_audioldm2.py": "31b9266060b066dbfe2d90dc30f9a686a2c211d0",
    "scripts/convert_original_audioldm2_to_diffusers.py": "f0b22cb4b4c7f93299e43406c5875780fdc8f78f",
}
LOCK = Path(__file__).with_name("audioldm2_reference") / "uv.lock"
REQUIRED_ROLES = {"vae", "unet", "vocoder", "language_model", "projection_model", "text_encoder", "text_encoder_2", "feature_extractor", "tokenizer", "tokenizer_2", "scheduler"}
EXPECTED_COMPONENT_CLASSES = {
    "feature_extractor": ["transformers", "ClapFeatureExtractor"],
    "language_model": ["transformers", "GPT2Model"],
    "projection_model": ["audioldm2", "AudioLDM2ProjectionModel"],
    "scheduler": ["diffusers", "DDIMScheduler"],
    "text_encoder": ["transformers", "ClapModel"],
    "text_encoder_2": ["transformers", "T5EncoderModel"],
    "tokenizer": ["transformers", "RobertaTokenizerFast"],
    "tokenizer_2": ["transformers", "T5TokenizerFast"],
    "unet": ["audioldm2", "AudioLDM2UNet2DConditionModel"],
    "vae": ["diffusers", "AutoencoderKL"],
    "vocoder": ["transformers", "SpeechT5HifiGan"],
}


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _file_digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git_blob_sha1_file(path: Path) -> str:
    h = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _lfs_pointer_sha1(sha256: str, size: int) -> str:
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha256}\nsize {size}\n".encode()
    return _git_blob_sha1(pointer)


def _validate_source_identity(source: Path) -> dict[str, object]:
    if not (source / ".git").exists():
        _fail("official implementation is not a git checkout")
    def git(*args: str) -> str:
        result = subprocess.run(["git", "-C", str(source), *args], check=False, capture_output=True, text=True)
        if result.returncode != 0:
            _fail(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        return result.stdout.strip()
    origin = git("remote", "get-url", "origin")
    head = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--untracked-files=all")
    if origin != SOURCE_ORIGIN or dirty:
        _fail("source origin or clean checkout validation failed")
    if head != SOURCE_COMMIT:
        _fail("source HEAD does not match the fixed official implementation commit")
    tag_object = git("rev-parse", f"refs/tags/{SOURCE_TAG}^{{commit}}")
    if tag_object != SOURCE_COMMIT:
        _fail("source HEAD is not the authenticated Diffusers v0.21.0 tag")
    rows = []
    tracked = set()
    for line in git("ls-files", "--stage", "-z").split("\0"):
        if not line:
            continue
        metadata, path_name = line.split("\t", 1)
        mode, index_object, stage = metadata.split()
        if stage != "0" or mode not in {"100644", "100755"}:
            _fail(f"tracked source entry has unsafe mode/stage: {path_name}")
        if path_name in tracked:
            _fail(f"duplicate tracked source path: {path_name}")
        tracked.add(path_name)
        path = source / path_name
        if path.is_symlink() or not path.is_file() or (path.stat().st_mode & 0o7777) != int(mode[-4:], 8):
            _fail(f"tracked source entry is not regular or mode-drifted: {path_name}")
        head_object = git("rev-parse", f"HEAD:{path_name}")
        working_object = _git_blob_sha1_file(path)
        if index_object != head_object or index_object != working_object:
            _fail(f"tracked source object drift: {path_name}")
        rows.append({"path": path_name, "mode": mode[-4:], "stage": 0, "bytes": path.stat().st_size, "index_object": index_object, "head_object": head_object, "working_git_blob_sha1": working_object})
    if set(SOURCE_ROLE_BLOBS) - tracked:
        _fail("fixed Diffusers role is missing")
    for role, expected in SOURCE_ROLE_BLOBS.items():
        row = next(row for row in rows if row["path"] == role)
        if row["mode"] != "0644" or row["head_object"] != expected:
            _fail(f"fixed Diffusers role identity drift: {role}")
    license_text = (source / "LICENSE").read_text(encoding="utf-8", errors="strict").lower()
    if not all(marker in license_text for marker in ("apache license", "version 2.0, january 2004", "you may obtain a copy", "distributed under the license", "without warranties or conditions")):
        _fail("Diffusers Apache-2.0 license clauses are incomplete")
    return {"repository": SOURCE_REPOSITORY, "origin": origin, "revision": SOURCE_COMMIT, "tag": SOURCE_TAG, "clean": True, "tracked_file_count": len(rows), "tracked_files": rows, "roles": [row for row in rows if row["path"] in SOURCE_ROLE_BLOBS], "license": {"path": "LICENSE", "git_blob_sha1": SOURCE_ROLE_BLOBS["LICENSE"], "spdx": "Apache-2.0"}}


def _fail(message: str) -> None:
    raise RuntimeError(f"audioldm2 inspector BLOCKED: {message}")


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_pairs)


def _safe_relative(value: str) -> None:
    path = Path(value)
    if not value or "\x00" in value or "\\" in value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe relative path: {value!r}")


def _local_files(root: Path) -> set[str]:
    files: set[str] = set()
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"snapshot symlink is forbidden: {rel}")
        if rel == ".cache":
            if not path.is_dir():
                raise ValueError("snapshot cache parent is not a directory")
            continue
        if rel == ".cache/huggingface":
            if not path.is_dir():
                raise ValueError("HF transport cache is not a directory")
            continue
        if rel.startswith(".cache/huggingface/"):
            continue
        if ".cache" in Path(rel).parts:
            raise ValueError(f"unauthenticated cache outside .cache/huggingface: {rel}")
        if path.is_file():
            if rel == ".vokra-server-tree.json" or rel == ".vokra-source-revision":
                continue
            files.add(rel)
        elif not path.is_dir():
            raise ValueError(f"snapshot member is not regular: {rel}")
    return files


def _server_inventory(root: Path, packet_path: Path, repository: str, revision: str, expected_tree: set[str]) -> list[dict[str, Any]]:
    packet = _load_json(packet_path)
    if not isinstance(packet, dict) or set(packet) != {"repository", "requested_revision", "resolved_revision", "walk", "files"}:
        raise ValueError("server packet schema drift")
    if packet["repository"] != repository or packet["requested_revision"] != revision or packet["resolved_revision"] != revision or packet["walk"] != "recursive_file_only":
        raise ValueError("server packet repository/revision drift")
    rows = packet["files"]
    if not isinstance(rows, list) or len(rows) != len(expected_tree):
        raise ValueError("server packet file set is incomplete")
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "type", "size", "git_blob_sha1", "lfs_pointer_git_blob_sha1", "lfs_payload_sha256", "lfs_payload_size"}:
            raise ValueError("server packet row schema drift")
        path, kind, size = row["path"], row["type"], row["size"]
        if not isinstance(path, str) or path in seen or path not in expected_tree or kind != "file":
            raise ValueError(f"server packet path/type drift: {path!r}")
        _safe_relative(path)
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"server packet size drift: {path}")
        git_id = row["git_blob_sha1"]
        pointer_id = row["lfs_pointer_git_blob_sha1"]
        payload_id = row["lfs_payload_sha256"]
        payload_size = row["lfs_payload_size"]
        if payload_id is None:
            if not isinstance(git_id, str) or not re.fullmatch(r"[0-9a-f]{40}", git_id) or pointer_id is not None or payload_size is not None:
                raise ValueError(f"regular server identity drift: {path}")
        else:
            if git_id is not None or not isinstance(pointer_id, str) or not re.fullmatch(r"[0-9a-f]{40}", pointer_id) or not isinstance(payload_id, str) or not re.fullmatch(r"[0-9a-f]{64}", payload_id) or not isinstance(payload_size, int) or isinstance(payload_size, bool) or payload_size != size:
                raise ValueError(f"LFS server identity drift: {path}")
        local = root / path
        if not local.is_file() or local.is_symlink() or local.stat().st_size != size:
            raise ValueError(f"local file/size drift: {path}")
        if payload_id is None:
            if _git_blob_sha1_file(local) != git_id:
                raise ValueError(f"regular Git identity mismatch: {path}")
        else:
            if _file_digest(local) != payload_id or _lfs_pointer_sha1(payload_id, size) != pointer_id:
                raise ValueError(f"LFS payload/pointer mismatch: {path}")
        seen.add(path)
        result.append(dict(row))
    if seen != expected_tree or _local_files(root) != expected_tree:
        raise ValueError("local snapshot differs from authenticated server tree")
    return sorted(result, key=lambda row: row["path"])


def _safetensors_header(path: Path, relative: str | None = None) -> dict[str, Any]:
    display = relative or path.name
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw_size = stream.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"truncated safetensors header: {display}")
        header_size = int.from_bytes(raw_size, "little")
        if header_size <= 0 or header_size > 64 * 1024 * 1024 or header_size > size - 8:
            raise ValueError(f"unsafe safetensors header size: {display}")
        header_raw = stream.read(header_size)
    header = json.loads(header_raw.decode("utf-8"), object_pairs_hook=_strict_pairs)
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not an object")
    metadata = header.pop("__metadata__", None)
    if metadata is not None and (not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items())):
        raise ValueError("safetensors metadata drift")
    payload = size - 8 - header_size
    widths = {"F32": 4, "F16": 2, "BF16": 2}
    intervals: list[tuple[int, int]] = []
    rows = []
    for name, descriptor in header.items():
        _safe_relative(name)
        if not isinstance(descriptor, dict) or set(descriptor) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"safetensors descriptor drift: {name}")
        dtype, shape, offsets = descriptor["dtype"], descriptor["shape"], descriptor["data_offsets"]
        if dtype not in widths or not isinstance(shape, list) or any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in shape) or not isinstance(offsets, list) or len(offsets) != 2 or any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in offsets):
            raise ValueError(f"unsafe safetensors descriptor: {name}")
        start, end = offsets
        elements = 1
        for dim in shape:
            elements *= dim
        if end < start or end > payload or end - start != elements * widths[dtype]:
            raise ValueError(f"safetensors range/shape drift: {name}")
        intervals.append((start, end)); rows.append({"name": name, "dtype": dtype, "shape": shape, "offsets": offsets, "elements": elements})
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError(f"safetensors gap/overlap: {display}")
        cursor = end
    if cursor != payload:
        raise ValueError(f"safetensors trailing payload: {display}")
    return {"path": display, "bytes": size, "header_bytes": header_size, "payload_bytes": payload, "tensor_count": len(rows), "parameters": sum(row["elements"] for row in rows), "tensors": rows}


def _model_index_evidence(snapshot: Path) -> dict[str, Any]:
    model_index = _load_json(snapshot / "model_index.json")
    if not isinstance(model_index, dict):
        raise ValueError("model_index must contain one pipeline object")
    entry = model_index
    if entry.get("_class_name") != "AudioLDM2Pipeline" or entry.get("_diffusers_version") != "0.20.0.dev0":
        raise ValueError("model_index pipeline class drift")
    component_classes = {key: value for key, value in entry.items() if not key.startswith("_")}
    if component_classes != EXPECTED_COMPONENT_CLASSES:
        raise ValueError("model_index component class mapping drift")
    return {"pipeline_class": entry["_class_name"], "diffusers_version": entry["_diffusers_version"], "components": component_classes}


def _model_license_evidence(snapshot: Path) -> dict[str, Any]:
    lines = (snapshot / "README.md").read_text(encoding="utf-8", errors="strict").splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("model-card frontmatter is absent")
    try:
        end = next(index for index in range(1, len(lines)) if lines[index].strip() == "---")
    except StopIteration as error:
        raise ValueError("model-card frontmatter is unterminated") from error
    license_value: str | None = None
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#") or line[0].isspace() or line.startswith("-"):
            continue
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_-]*):[ \t]*(.*)", line)
        if match is None:
            raise ValueError("model-card frontmatter is malformed")
        key, value = match.groups()
        if key == "license":
            if license_value is not None:
                raise ValueError("model-card license is duplicated")
            license_value = value.strip().strip("\"'")
    if license_value != "cc-by-nc-sa-4.0":
        raise ValueError("model-card license is not the fixed cc-by-nc-sa-4.0 declaration")
    return {"path": "README.md", "bytes": (snapshot / "README.md").stat().st_size, "sha256": _file_digest(snapshot / "README.md"), "license": license_value, "publication_block": "NONCOMMERCIAL_SHAREALIKE_RESEARCH_ONLY"}


def _write_error_manifest(output: Path, error: Exception) -> None:
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "vokra.audioldm2.inspection.v1",
        "status": "BLOCKED",
        "inspection_status": "INSPECTION_ERROR",
        "collection_status": "UNVERIFIED",
        "comparison_status": "NOT_RUN_OFFICIAL_ONLY",
        "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
        "cpu_status": "BLOCKED",
        "metal_status": "BLOCKED",
        "parity_status": "NOT_RUN",
        "publication": "NO_UPLOAD",
        "error_type": type(error).__name__,
        "reason": str(error),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def inspect(root: Path, source: Path, output: Path, large: bool = False) -> dict[str, object]:
    if not LOCK.is_file():
        _fail("dedicated transitive uv.lock is absent; fail before downloads")
    revision = LARGE_REVISION if large else BASE_REVISION
    repository = LARGE_REPOSITORY if large else BASE_REPOSITORY
    if large:
        from audioldm2_large_prepare_checkpoint import REQUIRED_TREE
    else:
        from audioldm2_prepare_checkpoint import REQUIRED_TREE
    packet_path = root / ".vokra-server-tree.json"
    if not packet_path.is_file():
        _fail("authoritative server-tree packet is absent")
    try:
        rows = _server_inventory(root, packet_path, repository, revision, set(REQUIRED_TREE))
    except (OSError, KeyError, TypeError, AttributeError, ValueError, json.JSONDecodeError) as exc:
        _fail(f"invalid server-tree packet: {exc}")
    if not root.is_dir() or any(not (root / role).exists() for role in REQUIRED_ROLES):
        _fail("complete component tree is absent")
    if not source.is_dir() or not (source / ".git").exists():
        _fail("official implementation source checkout is absent or unauthenticated")
    source_identity = _validate_source_identity(source)
    model_index = _model_index_evidence(root)
    model_license = _model_license_evidence(root)
    shard_evidence = []
    for relative in sorted(REQUIRED_TREE):
        if relative.endswith(".safetensors"):
            shard_evidence.append(_safetensors_header(root / relative, relative))
    if len(shard_evidence) != len({row["path"] for row in shard_evidence}):
        _fail("safetensors header paths are duplicated")
    packet = {
        "schema": "vokra.audioldm2.inspection.v1",
        "status": "BLOCKED",
        "inspection_status": "AUTHENTICATED_EVIDENCE_COMPLETE",
        "collection_status": "AUTHENTICATED",
        "comparison_status": "NOT_RUN_OFFICIAL_ONLY",
        "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
        "cpu_status": "BLOCKED",
        "metal_status": "BLOCKED",
        "publication": "NO_UPLOAD",
        "repository": repository,
        "resolved_revision": revision,
        "server_tree_sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
        "source_checkout": str(source),
        "source_identity": source_identity,
        "model_index": model_index,
        "model_license": model_license,
        "components": sorted(REQUIRED_ROLES),
        "server_tree": rows,
        "safetensors_headers": shard_evidence,
        "tree_files": [
            {
                "path": relative,
                "bytes": (root / relative).stat().st_size,
                "sha256": _file_digest(root / relative),
            }
            for relative in sorted(REQUIRED_TREE)
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    return packet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--large", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        assert not LOCK.is_file() or LOCK.stat().st_size > 0
        assert SOURCE_TAG == "v0.21.0" and SOURCE_COMMIT == "29f15673ed5c14e4843d7c837890910207f72129"
        assert SOURCE_ROLE_BLOBS["LICENSE"] == "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64"
        assert "scheduler" in REQUIRED_ROLES and len(EXPECTED_COMPONENT_CLASSES) == 11
        sample = b"payload"
        assert _git_blob_sha1(sample) != hashlib.sha1(sample).hexdigest()
        lfs = "0" * 64
        assert _lfs_pointer_sha1(lfs, 7) != _git_blob_sha1(sample)
        with tempfile.TemporaryDirectory() as td:
            fixture = Path(td) / "fixture"
            fixture.mkdir()
            (fixture / "README.md").write_text("---\nlicense: cc-by-nc-sa-4.0\n---\n", encoding="utf-8")
            assert _model_license_evidence(fixture)["license"] == "cc-by-nc-sa-4.0"
            (fixture / "model_index.json").write_text(json.dumps({"_class_name": "AudioLDM2Pipeline", "_diffusers_version": "0.20.0.dev0", **EXPECTED_COMPONENT_CLASSES}), encoding="utf-8")
            assert _model_index_evidence(fixture)["pipeline_class"] == "AudioLDM2Pipeline"
            bad_index = json.loads((fixture / "model_index.json").read_text(encoding="utf-8")); bad_index["_diffusers_version"] = "0.21.0"; (fixture / "model_index.json").write_text(json.dumps(bad_index), encoding="utf-8")
            try:
                _model_index_evidence(fixture)
            except ValueError:
                pass
            else:
                raise AssertionError("model_index Diffusers version drift was accepted")
            (fixture / ".cache" / "huggingface").mkdir(parents=True)
            (fixture / ".cache" / "huggingface" / "transport.json").write_text("{}", encoding="utf-8")
            assert _local_files(fixture) == {"README.md", "model_index.json"}
            (fixture / ".cache" / "other").mkdir()
            try:
                _local_files(fixture)
            except ValueError:
                pass
            else:
                raise AssertionError("cache outside .cache/huggingface was accepted")
            (fixture / ".cache" / "other").rmdir()
            (fixture / "README.md").write_text("---\nlicense: mit\n---\n", encoding="utf-8")
            try:
                _model_license_evidence(fixture)
            except ValueError:
                pass
            else:
                raise AssertionError("wrong AudioLDM2 model-card license was accepted")
            error_output = Path(td) / "error-output"
            _write_error_manifest(error_output, RuntimeError("fixture"))
            error_manifest = _load_json(error_output / "manifest.json")
            assert error_manifest["inspection_status"] == "INSPECTION_ERROR" and error_manifest["collection_status"] == "UNVERIFIED"
            dirty = Path(td) / "dirty"
            dirty.mkdir()
            (dirty / ".git").mkdir()
            (dirty / "source-identity.json").write_text(json.dumps({"repository": SOURCE_REPOSITORY, "origin": SOURCE_ORIGIN, "clean": False, "commit": "a" * 40, "roles": ["pipeline"]}))
            try:
                _validate_source_identity(dirty)
            except RuntimeError:
                pass
            else:
                raise AssertionError("dirty source identity was accepted")
        print("audioldm2_inspect --self-test: OK")
        return 0
    if args.snapshot is None or args.source is None or args.output is None:
        parser.error("--snapshot, --source and --output are required")
    try:
        inspect(args.snapshot, args.source, args.output, args.large)
    except (OSError, RuntimeError, ValueError, KeyError, AttributeError) as exc:
        _write_error_manifest(args.output, exc)
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
