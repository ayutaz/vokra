#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Inspect the fixed Kimi-Audio composite release without converting it.

This inspector authenticates the release envelope, index/shard headers, the
embedded Whisper safetensors, and source/submodule identity. It never reads a
tensor body into a tensor object and never enables an unsafe pickle fallback.
The public result is permanently ``INSPECTION_ONLY`` until a separately
reviewed native/runtime contract exists.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

UPSTREAM_HF = "moonshotai/Kimi-Audio-7B-Instruct"
HF_REVISION = "9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b"
SOURCE_URL = "https://github.com/MoonshotAI/Kimi-Audio.git"
SOURCE_REVISION = "349251e1d8f4f98d58fda59246381faecd7392e0"
GLM4_REVISION = "eb00ce9142e8d98b0ed7c57cd47e0d6d5dce9a1a"
GLM4_URL = "https://github.com/THUDM/GLM-4-Voice.git"
INDEX_NAME = "model.safetensors.index.json"
EXPECTED_TOTAL_SIZE = 19_532_673_280
EXPECTED_WEIGHT_MAP_COUNT = 453
MAX_HEADER_BYTES = 64 * 1024 * 1024
FORMAT = "vokra-kimi-audio-7b-instruct-inspection-v1"
SOURCE_SUBMODULE = "kimia_infer/models/tokenizer/glm4"
EXPECTED_COMPONENTS = {
    "audio_detokenizer/model.pt": (19_008_505_142, "cdeeec41e629565439cd8ef807c8a014ad6ce052cce0c259c7bfe3fe6ada3f51"),
    "vocoder/model.pt": (964_918_850, "a043a75ae865a9f3264500966a2622399e6b29cf362f4e2134adaefd4ba1252c"),
    "vocoder/config.json": (1_402, None),
    "whisper-large-v3/model.safetensors": (3_087_131_376, "d677ab655d1916439c5868c819a0e48cdac574defab83c69b0bbc2b7b31a9f06"),
}
TRANSPORT_CACHE_PATH = PurePosixPath(".cache/huggingface")
SHARD_RE = re.compile(r"^model-(\d+)-of-(\d+)\.safetensors$")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_SHARDS = {f"model-{index}-of-35.safetensors" for index in range(1, 36)} | {"model-36-of-36.safetensors"}
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    """Compute the Git blob object id without invoking Git or reading twice."""
    digest = hashlib.sha1()
    size = path.stat().st_size
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def parse_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"invalid JSON {path}: {error}") from error


def safe_member_name(name: str) -> None:
    path = PurePosixPath(name)
    if "\x00" in name or "\\" in name or not name or path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"unsafe archive member path: {name!r}")


def archive_inventory(path: Path) -> dict[str, Any]:
    """Inventory tar/zip members, rejecting unsafe names and member types."""
    result: dict[str, Any] = {"path": path.name, "container": "unknown", "members": [], "blockers": []}

    def reject_embedded_nul(stream: BinaryIO) -> None:
        position = stream.tell()
        prefix = stream.read(6)
        stream.seek(position)
        if prefix[:2] in {b"\x1f\x8b", b"BZ"} or prefix.startswith(b"\xfd7zXZ"):
            return
        while True:
            header = stream.read(512)
            if not header or header == b"\0" * 512:
                return
            if len(header) != 512:
                raise RuntimeError("truncated tar header")
            for field in (header[:100], header[345:500]):
                nul = field.find(b"\0")
                if nul >= 0 and any(field[nul + 1 :]):
                    raise RuntimeError("embedded NUL in archive member name")
            size_field = header[124:136].rstrip(b"\0 ")
            try:
                size = int(size_field or b"0", 8)
            except ValueError as error:
                raise RuntimeError("invalid tar member size") from error
            stream.seek(((size + 511) // 512) * 512, io.SEEK_CUR)

    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                result["container"] = "zip"
                seen: set[str] = set()
                for item in archive.infolist():
                    name = item.filename
                    safe_member_name(name)
                    if name in seen:
                        raise RuntimeError(f"duplicate archive member: {name!r}")
                    seen.add(name)
                    mode = (item.external_attr >> 16) & 0o170000
                    is_directory = item.is_dir() or name.endswith("/")
                    if mode not in {0, 0o100000, 0o040000} or (is_directory and mode == 0o100000):
                        raise RuntimeError(f"unsafe archive member type: {name!r}")
                    result["members"].append({"name": name, "bytes": item.file_size, "type": "directory" if is_directory else "file"})
        elif tarfile.is_tarfile(path):
            with path.open("rb") as raw:
                reject_embedded_nul(raw)
            with tarfile.open(path, mode="r:*") as archive:
                result["container"] = "tar"
                seen = set()
                for item in archive:
                    name = item.name
                    safe_member_name(name)
                    if name in seen:
                        raise RuntimeError(f"duplicate archive member: {name!r}")
                    seen.add(name)
                    if not (item.isdir() or item.isfile()):
                        raise RuntimeError(f"unsafe archive member type: {name!r}")
                    result["members"].append({"name": name, "bytes": item.size, "type": "directory" if item.isdir() else "file"})
        else:
            result["archive_error"] = "not a recognized archive"
            result["blockers"].append(f"unrecognized archive format: {path}")
    except Exception as error:  # noqa: BLE001 - archive errors are evidence
        result["archive_error"] = f"{type(error).__name__}: {error}"
        result["blockers"].append(str(error))
    return result


def _checked_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{field} must be a non-negative integer")
    return value


def safetensors_header(path: Path) -> dict[str, Any]:
    """Read and validate only the safetensors header and file boundaries."""
    file_size = path.stat().st_size
    if file_size < 8:
        raise RuntimeError(f"safetensors file is shorter than header length: {path}")
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        header_length = int.from_bytes(raw_length, "little")
        if header_length <= 0 or header_length > MAX_HEADER_BYTES:
            raise RuntimeError(f"safetensors header length exceeds explicit bound: {header_length}")
        if 8 + header_length > file_size:
            raise RuntimeError(f"safetensors header exceeds file boundary: {path}")
        raw_header = stream.read(header_length)
    header = json.loads(raw_header.decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
    if not isinstance(header, dict):
        raise RuntimeError("safetensors header root is not an object")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()):
        raise RuntimeError("safetensors metadata must be a string map")
    data_size = file_size - 8 - header_length
    rows: dict[str, dict[str, Any]] = {}
    ranges: list[tuple[int, int, str]] = []
    for name, descriptor in header.items():
        if not isinstance(name, str) or "\x00" in name or "\\" in name:
            raise RuntimeError(f"unsafe tensor name: {name!r}")
        if not isinstance(descriptor, dict):
            raise RuntimeError(f"tensor descriptor is not an object: {name}")
        dtype = descriptor.get("dtype")
        shape = descriptor.get("shape")
        offsets = descriptor.get("data_offsets")
        if not isinstance(dtype, str) or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise RuntimeError(f"invalid tensor descriptor: {name}")
        dtype_bytes = DTYPE_BYTES.get(dtype)
        if dtype_bytes is None:
            raise RuntimeError(f"unsupported safetensors dtype: {dtype}")
        shape_values = [_checked_int(value, f"shape[{name}]") for value in shape]
        offset_values = [_checked_int(value, f"data_offsets[{name}]") for value in offsets]
        begin, end = offset_values
        if begin > end or end > data_size:
            raise RuntimeError(f"tensor offset outside data region: {name}")
        elements = 1
        for dimension in shape_values:
            elements *= dimension
        if end - begin != elements * dtype_bytes:
            raise RuntimeError(f"tensor byte size does not match shape and dtype: {name}")
        ranges.append((begin, end, name))
        rows[name] = {"dtype": dtype, "shape": shape_values, "data_offsets": offset_values, "bytes": end - begin}
    ranges.sort()
    cursor = 0
    for begin, end, name in ranges:
        if begin != cursor:
            raise RuntimeError(f"safetensors gap/overlap before tensor: {name}")
        cursor = end
    if cursor != data_size:
        raise RuntimeError("safetensors data region has trailing gap")
    return {"header_bytes": header_length, "data_bytes": data_size, "metadata": metadata, "tensors": rows}


def validate_index(index: dict[str, Any], root: Path) -> dict[str, Any]:
    metadata = index.get("metadata")
    weight_map = index.get("weight_map")
    if not isinstance(metadata, dict) or metadata.get("total_size") != EXPECTED_TOTAL_SIZE:
        raise RuntimeError(f"index metadata total_size mismatch: {metadata!r}")
    if not isinstance(weight_map, dict) or len(weight_map) != EXPECTED_WEIGHT_MAP_COUNT:
        raise RuntimeError(f"index weight_map count mismatch: {len(weight_map) if isinstance(weight_map, dict) else None}")
    normalized: dict[str, str] = {}
    for tensor_name, shard_name in weight_map.items():
        if not isinstance(tensor_name, str) or not isinstance(shard_name, str):
            raise RuntimeError("index weight_map keys and values must be strings")
        tensor_path = PurePosixPath(tensor_name)
        if "\x00" in tensor_name or "\\" in tensor_name or tensor_path.is_absolute() or ".." in tensor_path.parts:
            raise RuntimeError(f"unsafe tensor key in index: {tensor_name!r}")
        safe_member_name(shard_name)
        if PurePosixPath(shard_name).name != shard_name or not SHARD_RE.fullmatch(shard_name):
            raise RuntimeError(f"index shard is not a direct model shard basename: {shard_name!r}")
        normalized[tensor_name] = shard_name
    indexed = set(normalized.values())
    if indexed != EXPECTED_SHARDS:
        raise RuntimeError(f"root shard basename set mismatch: missing={sorted(EXPECTED_SHARDS - indexed)} extra={sorted(indexed - EXPECTED_SHARDS)}")
    thirty_six = "model-36-of-36.safetensors"
    if thirty_six in indexed:
        expected_36 = {
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
            "mimo_output.weight",
            "model.mimo_norm.weight",
        }
        if {name for name, shard in normalized.items() if shard == thirty_six} != expected_36:
            raise RuntimeError("36-of-36 shard tensor mapping is not the authenticated five-tensor set")
    actual = {path.name for path in root.iterdir() if path.is_file() and SHARD_RE.fullmatch(path.name)}
    if indexed != actual:
        raise RuntimeError(f"indexed shards differ from root shards: missing={sorted(indexed - actual)} orphan={sorted(actual - indexed)}")
    for shard_name in sorted(indexed):
        shard = root / shard_name
        if shard.is_symlink() or not shard.is_file():
            raise RuntimeError(f"indexed shard is not a regular file: {shard}")
    return {"metadata": metadata, "weight_map": normalized, "indexed_shards": sorted(indexed)}


def validate_index_and_shards(root: Path) -> dict[str, Any]:
    index_path = root / INDEX_NAME
    index = parse_json(index_path)
    if not isinstance(index, dict):
        raise RuntimeError("safetensors index root is not an object")
    validated = validate_index(index, root)
    shard_headers: dict[str, Any] = {}
    for shard_name in validated["indexed_shards"]:
        shard_headers[shard_name] = safetensors_header(root / shard_name)
    mapped: dict[str, list[str]] = {name: [] for name in validated["indexed_shards"]}
    for tensor_name, shard_name in validated["weight_map"].items():
        mapped[shard_name].append(tensor_name)
    for shard_name, names in mapped.items():
        actual_names = set(shard_headers[shard_name]["tensors"])
        if actual_names != set(names):
            raise RuntimeError(f"index/header tensor mapping mismatch in {shard_name}")
    return {"index": validated, "shards": shard_headers}


def config_axes(config: dict[str, Any]) -> dict[str, Any]:
    """Authenticate the fixed raw axes without turning them into runtime defaults."""
    expected = {
        "architectures": ["MoonshotKimiaForCausalLM"],
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "max_position_embeddings": 8192,
        "kimia_mimo_layers": 6,
        "kimia_mimo_transformer_from_layer_index": 21,
        "kimia_audio_output_vocab": 16896,
        "kimia_text_output_vocab": 152064,
        "kimia_token_offset": 152064,
        "vocab_size": 168448,
        "kimia_mimo_audiodelaytokens": 6,
        "kimia_adaptor_input_dim": 5120,
        "torch_dtype": "bfloat16",
    }
    actual = {key: config.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(f"Kimi-Audio config axes mismatch: {actual}")
    return actual


def transport_cache_scope(root: Path, blockers: list[str]) -> tuple[Path | None, dict[str, Any]]:
    """Return the exact local_dir transport cache subtree, if valid.

    ``snapshot_download(local_dir=...)`` writes transport metadata below the
    snapshot root's ``.cache/huggingface`` directory.  Only that exact path is
    excluded from model identity accounting; any other cache-like path remains
    a real snapshot member.  The two path components are checked explicitly so
    a symlink or non-directory cannot smuggle files past the tree contract.
    """
    evidence: dict[str, Any] = {
        "path": TRANSPORT_CACHE_PATH.as_posix(),
        "scope": "snapshot_root_exact_transport_subtree",
        "present": False,
        "identity_role": "NON_IDENTITY_TRANSPORT_METADATA",
        "status": "ABSENT",
    }
    cache = root / ".cache"
    if cache.is_symlink() or (cache.exists() and not cache.is_dir()):
        message = f"snapshot root .cache must be a regular directory: {cache}"
        blockers.append(message)
        evidence["status"] = "INVALID"
        evidence["error"] = message
        return None, evidence
    if not cache.exists():
        return None, evidence

    transport = cache / "huggingface"
    if transport.is_symlink() or (transport.exists() and not transport.is_dir()):
        message = f"snapshot root .cache/huggingface must be a regular directory: {transport}"
        blockers.append(message)
        evidence["status"] = "INVALID"
        evidence["error"] = message
        return None, evidence
    if not transport.exists():
        return None, evidence

    evidence["present"] = True
    evidence["status"] = "EXCLUDED"
    try:
        evidence["entry_count"] = sum(1 for _ in transport.rglob("*"))
    except OSError as error:
        message = f"snapshot transport cache inventory failed: {transport}: {error}"
        blockers.append(message)
        evidence["status"] = "INVALID"
        evidence["error"] = message
        return None, evidence
    return transport, evidence


def files_under(root: Path, excluded_transport: Path | None = None) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    blockers: list[str] = []
    if not root.is_dir():
        return [], [f"missing HF snapshot directory: {root}"]
    for path in sorted(root.rglob("*")):
        if excluded_transport is not None:
            try:
                path.relative_to(excluded_transport)
            except ValueError:
                pass
            else:
                continue
        if path.is_symlink():
            try:
                resolved = path.resolve(strict=True)
            except OSError as error:
                blockers.append(f"dangling snapshot symlink: {path}: {error}")
                continue
            try:
                resolved.relative_to(root.resolve())
            except ValueError:
                blockers.append(f"snapshot symlink escapes root: {path}")
                continue
            if not resolved.is_file():
                blockers.append(f"snapshot symlink target is not regular: {path}")
                continue
        elif path.is_dir():
            continue
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            blockers.append(f"dangling snapshot member: {path}: {error}")
            continue
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            blockers.append(f"snapshot member escapes root: {path}")
            continue
        if not resolved.is_file():
            blockers.append(f"snapshot member is not regular: {path}")
            continue
        files.append(path)
    if not files:
        blockers.append(f"HF snapshot has no regular files: {root}")
    return files, blockers


def tree_evidence(
    root: Path,
    server_tree: Path | None,
    blockers: list[str],
    *,
    files: list[Path] | None = None,
    transport_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if files is None:
        excluded_transport, discovered_cache = transport_cache_scope(root, blockers)
        files, local_blockers = files_under(root, excluded_transport)
        blockers.extend(local_blockers)
        if transport_cache is None:
            transport_cache = discovered_cache
    elif transport_cache is None:
        _, transport_cache = transport_cache_scope(root, blockers)
    local = sorted((path.relative_to(root).as_posix(), path.stat().st_size) for path in files)
    if server_tree is None:
        blockers.append("missing HF server tree envelope")
        return {"repository": UPSTREAM_HF, "revision": HF_REVISION, "resolved_revision": None, "files": local, "transport_cache": transport_cache, "status": "MISSING"}
    remote = parse_json(server_tree)
    if not isinstance(remote, dict):
        raise RuntimeError("HF server tree envelope is not an object")
    if remote.get("repository") != UPSTREAM_HF or remote.get("revision") != HF_REVISION or remote.get("resolved_revision") != HF_REVISION:
        raise RuntimeError("HF server tree identity/revision mismatch")
    remote_files = remote.get("files")
    if not isinstance(remote_files, list):
        raise RuntimeError("HF server tree files is not a list")
    server_rows: list[dict[str, Any]] = []
    for item in remote_files:
        if not isinstance(item, dict) or set(item) != {"path", "type", "size", "oid", "lfs_sha256"}:
            raise RuntimeError("HF server tree entry must have exactly path/type/size/oid/lfs_sha256")
        if item["type"] != "file" or not isinstance(item["path"], str) or not isinstance(item["size"], int) or isinstance(item["size"], bool) or item["size"] < 0 or not isinstance(item["oid"], str) or item["lfs_sha256"] is not None and not isinstance(item["lfs_sha256"], str):
            raise RuntimeError(f"HF server tree entry has invalid types: {item!r}")
        safe_member_name(item["path"])
        server_rows.append(item)
    if len({item["path"] for item in server_rows}) != len(server_rows):
        raise RuntimeError("HF server tree contains duplicate file paths")
    server = sorted((item["path"], item["size"]) for item in server_rows)
    if server != local:
        raise RuntimeError(f"HF server/local tree mismatch: server={server!r} local={local!r}")
    records: list[dict[str, Any]] = []
    for item in sorted(server_rows, key=lambda row: row["path"]):
        path = root / item["path"]
        if path.stat().st_size != item["size"]:
            raise RuntimeError(f"HF server/local size mismatch: {item['path']}")
        local_sha256 = sha256(path)
        lfs_sha256 = item["lfs_sha256"]
        oid = item["oid"]
        if lfs_sha256 is not None:
            if not re.fullmatch(r"[0-9a-fA-F]{64}", lfs_sha256) or oid.lower() != lfs_sha256.lower():
                raise RuntimeError(f"HF server LFS identity is malformed: {item['path']}")
            if local_sha256.lower() != lfs_sha256.lower():
                raise RuntimeError(f"HF server/local LFS SHA-256 mismatch: {item['path']}")
        else:
            if not re.fullmatch(r"[0-9a-fA-F]{40}", oid):
                raise RuntimeError(f"HF server Git blob OID is malformed: {item['path']}")
            if git_blob_sha1(path).lower() != oid.lower():
                raise RuntimeError(f"HF server/local Git blob SHA-1 mismatch: {item['path']}")
        records.append({"path": item["path"], "size": item["size"], "oid": oid, "lfs_sha256": lfs_sha256, "local_sha256": local_sha256})
    return {"repository": UPSTREAM_HF, "revision": HF_REVISION, "resolved_revision": remote.get("resolved_revision"), "files": local, "server_files": server, "content_identity": records, "transport_cache": transport_cache, "status": "MATCHED"}


def source_evidence(source: Path, blockers: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"repository": SOURCE_URL, "pinned_revision": SOURCE_REVISION, "resolved_revision": "", "pinned_origin": SOURCE_URL, "resolved_origin": "", "submodule": {"path": SOURCE_SUBMODULE, "pinned_revision": GLM4_REVISION, "pinned_origin": GLM4_URL}}
    try:
        origin = subprocess.run(["git", "-C", str(source), "remote", "get-url", "origin"], check=True, capture_output=True, text=True).stdout.strip()
        result["resolved_origin"] = origin
        if origin != SOURCE_URL:
            blockers.append(f"source origin {origin!r} != pinned {SOURCE_URL!r}")
        head = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        result["resolved_revision"] = head
        if head != SOURCE_REVISION:
            blockers.append(f"source revision {head!r} != pinned {SOURCE_REVISION!r}")
        ls_tree = subprocess.run(["git", "-C", str(source), "ls-tree", "HEAD", "--", SOURCE_SUBMODULE], check=True, capture_output=True, text=True).stdout.strip()
        result["gitlink"] = ls_tree
        if f"commit {GLM4_REVISION}\t{SOURCE_SUBMODULE}" not in ls_tree:
            blockers.append("GLM-4-Voice gitlink does not match pinned revision")
        submodule = source / SOURCE_SUBMODULE
        sub_origin = subprocess.run(["git", "-C", str(submodule), "remote", "get-url", "origin"], check=True, capture_output=True, text=True).stdout.strip()
        sub_head = subprocess.run(["git", "-C", str(submodule), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        result["submodule"].update({"resolved_origin": sub_origin, "resolved_revision": sub_head})
        if sub_origin != GLM4_URL:
            blockers.append(f"GLM-4-Voice submodule origin {sub_origin!r} != pinned {GLM4_URL!r}")
        if sub_head != GLM4_REVISION:
            blockers.append(f"GLM-4-Voice submodule revision {sub_head!r} != pinned {GLM4_REVISION!r}")
        submodule_license = [
            path for path in sorted(submodule.rglob("*"))
            if path.is_file() and not path.is_symlink() and path.name.lower() in {"license", "license.md", "license.txt", "notice"}
        ]
        result["submodule"]["license_records"] = [
            {"path": path.relative_to(source).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in submodule_license
        ]
        if not submodule_license:
            blockers.append("GLM-4-Voice submodule has no tracked license file")
    except (OSError, subprocess.CalledProcessError) as error:
        blockers.append(f"source identity unavailable: {error}")
    tracked = subprocess.run(["git", "-C", str(source), "ls-files"], capture_output=True, text=True).stdout.splitlines() if source.is_dir() else []
    tracked_records: list[dict[str, Any]] = []
    for name in tracked:
        path = source / name
        if not path.is_file() or path.is_symlink():
            continue
        lower_name = name.lower()
        if "license" in lower_name or Path(name).name.lower().startswith(("readme", "notice", "copying")) or "requirements" in lower_name or Path(name).name.lower() in {"pyproject.toml", "setup.py"}:
            record: dict[str, Any] = {"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            text = path.read_text(encoding="utf-8", errors="replace")
            record["declarations"] = sorted(set(re.findall(r"(?i)(?:apache[- ]2\.0|mit|license|qwen|bigvgan|whisper|transformers)[^\n]{0,120}", text)))
            tracked_records.append(record)
    result["tracked_roles"] = {
        "license_files": [name for name in tracked if Path(name).name.lower() in {"license", "license.md", "license.txt", "notice", "copying"}],
        "readme_files": [name for name in tracked if Path(name).name.lower().startswith("readme")],
        "requirements": [name for name in tracked if "requirements" in name.lower() or Path(name).name.lower() in {"pyproject.toml", "setup.py"}],
    }
    if not result["tracked_roles"]["license_files"]:
        blockers.append("Kimi-Audio source root has no tracked LICENSE/NOTICE file")
    result["license_records"] = tracked_records
    if not any("license" in record["path"].lower() or Path(record["path"]).name.lower().startswith(("readme", "notice")) for record in tracked_records):
        blockers.append("Kimi-Audio source license declarations could not be evidenced")
    result["license_status"] = {"source_code": "README_DECLARATIONS_ONLY", "dependencies": "UNREVIEWED_BLOCKER", "submodule": "SEPARATE_LICENSE_REVIEW_REQUIRED", "components": ["BigVGAN", "Whisper", "Qwen", "Transformers"]}
    blockers.extend(["source README declares mixed Apache-2.0/MIT code licenses without a root LICENSE", "BigVGAN/Whisper/Qwen/Transformers/dependency licenses require separate evidence"])
    return result


def component_evidence(root: Path, files: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        suffix = path.suffix.lower()
        name = path.name.lower()
        if suffix in {".json", ".yaml", ".yml", ".py", ".txt", ".model", ".vocab"} or any(token in name for token in ("tokenizer", "config", "license", "readme")):
            record: dict[str, Any] = {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path), "kind": suffix or "named"}
            if suffix == ".json":
                record["json"] = parse_json(path)
            elif suffix == ".py":
                try:
                    ast.parse(path.read_text(encoding="utf-8"))
                    record["python_syntax"] = "VALID"
                except (OSError, SyntaxError) as error:
                    record["python_syntax"] = f"INVALID: {error}"
            records.append(record)
    return records


def checkpoint_evidence(path: Path, blockers: list[str]) -> dict[str, Any]:
    result = {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path), "archive": archive_inventory(path)}
    blockers.extend(result["archive"]["blockers"])
    try:
        import torch
    except ImportError as error:
        blockers.append(f"torch unavailable for safe checkpoint inspection: {error}")
        return result
    try:
        unsafe_globals = torch.serialization.get_unsafe_globals_in_checkpoint(str(path))
        result["unsafe_globals"] = unsafe_globals
        if unsafe_globals:
            blockers.append(f"checkpoint contains unsafe globals; no allowlist fallback is permitted: {path}")
    except Exception as error:  # noqa: BLE001 - failure is a blocker
        blockers.append(f"unsafe-global inventory failed for {path}: {error}")
    try:
        torch.load(str(path), map_location="cpu", weights_only=True)
        result["weights_only_load"] = "OK"
    except Exception as error:  # noqa: BLE001 - no unsafe fallback
        result["weights_only_load"] = f"BLOCKED: {type(error).__name__}: {error}"
        blockers.append(f"weights_only checkpoint load failed: {path}: {error}")
    return result


def _inspect(snapshot: Path, source: Path, evidence: Path, server_tree: Path | None, revision: str) -> int:
    blockers: list[str] = []
    if revision != HF_REVISION:
        blockers.append(f"operator revision {revision!r} differs from pinned revision")
    excluded_transport, transport_cache = transport_cache_scope(snapshot, blockers)
    files, file_blockers = files_under(snapshot, excluded_transport)
    blockers.extend(file_blockers)
    tree = tree_evidence(snapshot, server_tree, blockers, files=files, transport_cache=transport_cache)
    config = parse_json(snapshot / "config.json")
    if not isinstance(config, dict):
        raise RuntimeError("Kimi-Audio config.json root is not an object")
    axes = config_axes(config)
    root_evidence = validate_index_and_shards(snapshot)
    component_identities: dict[str, Any] = {}
    for relative, (expected_bytes, expected_sha256) in EXPECTED_COMPONENTS.items():
        component = snapshot / relative
        if component.is_symlink() or not component.is_file():
            blockers.append(f"missing fixed Kimi-Audio component: {relative}")
            continue
        actual_bytes = component.stat().st_size
        actual_sha256 = sha256(component)
        component_identities[relative] = {"bytes": actual_bytes, "sha256": actual_sha256}
        if actual_bytes != expected_bytes or (expected_sha256 is not None and actual_sha256 != expected_sha256):
            blockers.append(f"fixed component identity mismatch: {relative}")
    whisper = [path for path in files if path.name == "model.safetensors" and path.relative_to(snapshot).as_posix() != "model.safetensors"]
    whisper_headers = {path.relative_to(snapshot).as_posix(): safetensors_header(path) for path in whisper}
    if not whisper_headers:
        blockers.append("embedded Whisper safetensors component is missing")
    checkpoint_records = [checkpoint_evidence(path, blockers) for path in files if path.suffix.lower() in {".pt", ".pth", ".bin"}]
    if not checkpoint_records:
        blockers.append("Kimi-Audio composite has no discovered .pt/.pth/.bin component")
    source_record = source_evidence(source, blockers)
    model_license_records = []
    for path in files:
        if "license" not in path.name.lower() and not path.name.lower().startswith(("readme", "notice")):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        model_license_records.append({"path": path.relative_to(snapshot).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path), "declarations": sorted(set(re.findall(r"(?i)(?:mit|apache[- ]2\.0|license)[^\n]{0,120}", text)))})
    if not any("mit" in declaration.lower() for record in model_license_records for declaration in record["declarations"]):
        blockers.append("Kimi-Audio model-card MIT declaration was not found in the fixed snapshot")
    payload = {
        "format": FORMAT,
        "status": "BLOCKED",
        "evidence_stage": "INSPECTION_ONLY",
        "model": {"repository": UPSTREAM_HF, "revision": HF_REVISION, "resolved_revision": HF_REVISION, "server_tree": tree, "files": [{"path": path.relative_to(snapshot).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files]},
        "root_index": root_evidence,
        "config_axes": axes,
        "fixed_components": component_identities,
        "embedded_whisper": whisper_headers,
        "components": component_evidence(snapshot, files),
        "checkpoints": checkpoint_records,
        "source": source_record,
        "model_license_records": model_license_records,
        "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
        "cpu_status": "UNSUPPORTED",
        "metal_status": "BLOCKED_BY_CPU",
        "parity_status": "NOT_RUN",
        "publication": "NO_UPLOAD",
        "license": {"model_card": "MIT_DECLARED_UNVERIFIED", "source_code": "README_DECLARATIONS_ONLY", "submodule": "SEPARATE_REVIEW_REQUIRED", "BigVGAN_Whisper_Qwen_Transformers": "UNREVIEWED_BLOCKER"},
        "blockers": sorted(set(blockers + ["composite layout/native runtime/parity are not authenticated", "model license and all component/dependency licenses require independent review"])),
    }
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "manifest.json").write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"Kimi-Audio inspection blocked; evidence preserved at {evidence / 'manifest.json'}", file=sys.stderr)
    return 2


def _failure_manifest(evidence: Path, source: Path, revision: str, error: Exception) -> None:
    evidence.mkdir(parents=True, exist_ok=True)
    try:
        origin = subprocess.run(["git", "-C", str(source), "remote", "get-url", "origin"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        origin = ""
    try:
        sub_origin = subprocess.run(["git", "-C", str(source / SOURCE_SUBMODULE), "remote", "get-url", "origin"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sub_origin = ""
    payload = {
        "format": FORMAT,
        "status": "BLOCKED",
        "evidence_stage": "INSPECTION_ONLY",
        "model": {"repository": UPSTREAM_HF, "revision": HF_REVISION, "resolved_revision": revision},
        "source": {"repository": SOURCE_URL, "pinned_revision": SOURCE_REVISION, "pinned_origin": SOURCE_URL, "resolved_origin": origin, "submodule": {"path": SOURCE_SUBMODULE, "pinned_revision": GLM4_REVISION, "pinned_origin": GLM4_URL, "resolved_origin": sub_origin}},
        "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
        "cpu_status": "UNSUPPORTED",
        "metal_status": "BLOCKED_BY_CPU",
        "parity_status": "NOT_RUN",
        "publication": "NO_UPLOAD",
        "error": f"{type(error).__name__}: {error}",
        "blockers": ["inspection_error", f"{type(error).__name__}: {error}"],
    }
    (evidence / "manifest.json").write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def inspect(snapshot: Path, source: Path, evidence: Path, server_tree: Path | None, revision: str) -> int:
    try:
        return _inspect(snapshot, source, evidence, server_tree, revision)
    except Exception as error:  # noqa: BLE001 - preserve all failures as evidence
        _failure_manifest(evidence, source, revision, error)
        print(f"Kimi-Audio inspection blocked; evidence preserved at {evidence / 'manifest.json'}", file=sys.stderr)
        return 2


def _tiny_safetensors(names: list[str]) -> bytes:
    header = {name: {"dtype": "F32", "shape": [1], "data_offsets": [i * 4, (i + 1) * 4]} for i, name in enumerate(names)}
    raw = json.dumps(header, separators=(",", ":")).encode()
    return len(raw).to_bytes(8, "little") + raw + b"\0" * (len(names) * 4)


def self_test() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    for marker in (HF_REVISION, SOURCE_REVISION, GLM4_REVISION, GLM4_URL, "weights_only=True", "NO_UPLOAD", "INSPECTION_ONLY"):
        assert marker in source
    axes_fixture = {
        "architectures": ["MoonshotKimiaForCausalLM"],
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "max_position_embeddings": 8192,
        "kimia_mimo_layers": 6,
        "kimia_mimo_transformer_from_layer_index": 21,
        "kimia_audio_output_vocab": 16896,
        "kimia_text_output_vocab": 152064,
        "kimia_token_offset": 152064,
        "vocab_size": 168448,
        "kimia_mimo_audiodelaytokens": 6,
        "kimia_adaptor_input_dim": 5120,
        "torch_dtype": "bfloat16",
    }
    assert config_axes(axes_fixture) == axes_fixture
    axes_fixture["hidden_size"] = 1
    try:
        config_axes(axes_fixture)
    except RuntimeError:
        pass
    else:
        raise AssertionError("invented config axis accepted")
    with tempfile.TemporaryDirectory(prefix="kimi-inspect-") as temporary:
        tree_root = Path(temporary) / "tree"
        tree_root.mkdir()
        content = tree_root / "config.json"
        content.write_bytes(b"abc")
        transport = tree_root / ".cache" / "huggingface"
        transport.mkdir(parents=True)
        (transport / "download-metadata.json").write_text("transport-only", encoding="utf-8")
        tree_packet = Path(temporary) / "tree.json"
        lfs_digest = sha256(content)
        tree_payload = {"repository": UPSTREAM_HF, "revision": HF_REVISION, "resolved_revision": HF_REVISION, "files": [{"path": "config.json", "type": "file", "size": 3, "oid": lfs_digest, "lfs_sha256": lfs_digest}]}
        tree_packet.write_text(json.dumps(tree_payload), encoding="utf-8")
        matched_tree = tree_evidence(tree_root, tree_packet, [])
        assert matched_tree["status"] == "MATCHED"
        assert matched_tree["transport_cache"]["path"] == ".cache/huggingface"
        assert matched_tree["transport_cache"]["scope"] == "snapshot_root_exact_transport_subtree"
        assert matched_tree["transport_cache"]["present"] is True
        assert matched_tree["transport_cache"]["identity_role"] == "NON_IDENTITY_TRANSPORT_METADATA"
        assert matched_tree["transport_cache"]["status"] == "EXCLUDED"
        assert matched_tree["transport_cache"]["entry_count"] == 1
        nested_cache = tree_root / "nested" / ".cache"
        nested_cache.mkdir(parents=True)
        (nested_cache / "real-extra").write_bytes(b"extra")
        try:
            tree_evidence(tree_root, tree_packet, [])
        except RuntimeError:
            pass
        else:
            raise AssertionError("nested .cache extra file was incorrectly excluded")
        (nested_cache / "real-extra").unlink()
        content.write_bytes(b"abd")
        try:
            tree_evidence(tree_root, tree_packet, [])
        except RuntimeError:
            pass
        else:
            raise AssertionError("same-size wrong content accepted")
        content.write_bytes(b"abc")
        bad_packet = json.loads(tree_packet.read_text(encoding="utf-8"))
        bad_packet["files"][0]["oid"] = "0" * 64
        tree_packet.write_text(json.dumps(bad_packet), encoding="utf-8")
        try:
            tree_evidence(tree_root, tree_packet, [])
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid LFS identity accepted")
        tree_packet.write_text(json.dumps(tree_payload), encoding="utf-8")
        for field in ("repository", "revision"):
            bad_identity = dict(tree_payload)
            bad_identity[field] = "wrong"
            tree_packet.write_text(json.dumps(bad_identity), encoding="utf-8")
            try:
                tree_evidence(tree_root, tree_packet, [])
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"wrong server-tree {field} accepted")
        for files in (tree_payload["files"] + [{"path": "extra", "type": "file", "size": 3, "oid": lfs_digest, "lfs_sha256": lfs_digest}], []):
            bad_tree = dict(tree_payload)
            bad_tree["files"] = files
            tree_packet.write_text(json.dumps(bad_tree), encoding="utf-8")
            try:
                tree_evidence(tree_root, tree_packet, [])
            except RuntimeError:
                pass
            else:
                raise AssertionError("extra/missing server-tree entry accepted")
        blob_packet = dict(tree_payload)
        blob_packet["files"] = [{"path": "config.json", "type": "file", "size": 3, "oid": git_blob_sha1(content), "lfs_sha256": None}]
        tree_packet.write_text(json.dumps(blob_packet), encoding="utf-8")
        tree_evidence(tree_root, tree_packet, [])
        for shape in ("cache-file", "transport-file", "cache-symlink", "transport-symlink"):
            malformed = Path(temporary) / shape
            malformed.mkdir()
            malformed_cache = malformed / ".cache"
            if shape.startswith("cache-"):
                if shape.endswith("file"):
                    malformed_cache.write_bytes(b"not-a-directory")
                else:
                    malformed_cache.symlink_to(Path(temporary) / "outside-cache", target_is_directory=True)
            else:
                malformed_cache.mkdir()
                transport_path = malformed_cache / "huggingface"
                if shape.endswith("file"):
                    transport_path.write_bytes(b"not-a-directory")
                else:
                    transport_path.symlink_to(Path(temporary) / "outside-transport", target_is_directory=True)
            malformed_blockers: list[str] = []
            _, malformed_evidence = transport_cache_scope(malformed, malformed_blockers)
            assert malformed_evidence["status"] == "INVALID"
            assert malformed_blockers
        root = Path(temporary) / "snapshot"
        root.mkdir()
        final_names = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight", "mimo_output.weight", "model.mimo_norm.weight"]
        first_names = [f"tensor.{i}" for i in range(EXPECTED_WEIGHT_MAP_COUNT - 5)]
        weight_map = {name: f"model-{index + 1}-of-35.safetensors" for index, name in enumerate(first_names[:35])}
        weight_map.update({name: "model-1-of-35.safetensors" for name in first_names[35:]})
        weight_map.update({name: "model-36-of-36.safetensors" for name in final_names})
        (root / INDEX_NAME).write_text(json.dumps({"metadata": {"total_size": EXPECTED_TOTAL_SIZE}, "weight_map": weight_map}), encoding="utf-8")
        for index in range(1, 36):
            names = [name for name, shard in weight_map.items() if shard == f"model-{index}-of-35.safetensors"]
            (root / f"model-{index}-of-35.safetensors").write_bytes(_tiny_safetensors(names))
        (root / "model-36-of-36.safetensors").write_bytes(_tiny_safetensors(final_names))
        validated = validate_index_and_shards(root)
        assert validated["index"]["indexed_shards"] == sorted(EXPECTED_SHARDS)
        bad = json.loads((root / INDEX_NAME).read_text(encoding="utf-8"))
        bad["weight_map"]["tensor.0"] = "../escape.safetensors"
        try:
            validate_index(bad, root)
        except RuntimeError:
            pass
        else:
            raise AssertionError("traversal shard path accepted")
        bad = json.loads((root / INDEX_NAME).read_text(encoding="utf-8"))
        bad["weight_map"]["tensor.0"] = "model-2-of-35.safetensors"
        (root / INDEX_NAME).write_text(json.dumps(bad), encoding="utf-8")
        try:
            validate_index_and_shards(root)
        except RuntimeError:
            pass
        else:
            raise AssertionError("orphan/missing shard mapping accepted")
        (root / INDEX_NAME).write_text(json.dumps({"metadata": {"total_size": EXPECTED_TOTAL_SIZE}, "weight_map": weight_map}), encoding="utf-8")
        bad = json.loads((root / INDEX_NAME).read_text(encoding="utf-8"))
        bad["weight_map"]["tensor.0"] = "model-1-of-34.safetensors"
        try:
            validate_index(bad, root)
        except RuntimeError:
            pass
        else:
            raise AssertionError("wrong shard denominator accepted")
        huge = root / "huge.safetensors"
        huge.write_bytes((MAX_HEADER_BYTES + 1).to_bytes(8, "little"))
        try:
            safetensors_header(huge)
        except RuntimeError:
            pass
        else:
            raise AssertionError("huge header accepted")
        bool_shape = root / "bool.safetensors"
        raw = b'{"x":{"dtype":"F32","shape":[true],"data_offsets":[0,0]}}'
        bool_shape.write_bytes(len(raw).to_bytes(8, "little") + raw)
        try:
            safetensors_header(bool_shape)
        except RuntimeError:
            pass
        else:
            raise AssertionError("boolean shape accepted")
        for name in ("../escape", "back\\slash", "nul\x00member"):
            archive = root / f"unsafe-{len(name)}.pt"
            with tarfile.open(archive, "w") as tar:
                member = tarfile.TarInfo(name)
                member.size = 1
                tar.addfile(member, io.BytesIO(b"x"))
            assert archive_inventory(archive)["blockers"]
        failure = Path(temporary) / "failure"
        assert inspect(root / "missing", Path(temporary) / "missing-source", failure, None, HF_REVISION) == 2
        manifest = json.loads((failure / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "BLOCKED" and manifest["evidence_stage"] == "INSPECTION_ONLY"
        assert manifest["source"]["pinned_origin"] == SOURCE_URL
    print("kimi_audio_7b_instruct_inspect.py self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--server-tree", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.snapshot, args.source, args.evidence, args.server_tree, args.revision)):
            parser.error("--self-test accepts no other arguments")
        self_test()
        return 0
    if any(value is None for value in (args.snapshot, args.source, args.evidence, args.server_tree, args.revision)):
        parser.error("--snapshot, --source, --evidence, --server-tree, and --revision are required")
    if not HEX40_RE.fullmatch(args.revision):
        parser.error("--revision must be a complete 40-hex revision")
    return inspect(args.snapshot, args.source, args.evidence, args.server_tree, args.revision)


if __name__ == "__main__":
    raise SystemExit(main())
