#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""VAST-only, fail-closed inspection oracle for Step-Audio-2-mini.

This file inventories a composite checkpoint and its token2wav companions. It
does not run ONNX, instantiate HyperPyYAML, execute custom code, convert, or
claim native/runtime parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import warnings
import zipfile
from pathlib import Path
from typing import Any

from safetensors import safe_open

HF_REPOSITORY = "stepfun-ai/Step-Audio-2-mini"
HF_REVISION = "e36fdd5d71e0ea22f09dd94bbab9bfc544ca1e36"
SOURCE_REPOSITORY = "https://github.com/stepfun-ai/Step-Audio2.git"
SOURCE_REVISION = "76e272b56c3917a8d7188f18bbb5a65dfc8a0845"
TRANSFORMERS_REPOSITORY = "https://github.com/huggingface/transformers"
TRANSFORMERS_TAG = "v4.49.0"
TRANSFORMERS_REVISION = "a22a4378d97d06b7a1d9abad6e0086d30fdea199"
FORMAT = "vokra-step-audio2-mini-inspection-v1"
SHARD_NAMES = {f"model-{i:05d}-of-00004.safetensors" for i in range(1, 5)}
SHARD_BYTES = {
    "model-00001-of-00004.safetensors": (4_925_370_984, "7b88e02b0b8c643412ec68cae009b3952dbd8e27642d61626065a2c420a8b73c"),
    "model-00002-of-00004.safetensors": (4_932_751_008, "3d412c8d2fc17ca3351751f3171d48ff5b139af623aa05749062f132ac2585f1"),
    "model-00003-of-00004.safetensors": (4_988_307_424, "135ae4a891350e8ebf9791ef073d310314e1f75192bece0971bfab7b86c5587c"),
    "model-00004-of-00004.safetensors": (1_784_019_520, "d35bf0ec42ff9ec160dfc6c5cb20a65247f0f8ba1c6edc620398c2ef49a66295"),
}
SHARD_TENSOR_COUNTS = {"model-00001-of-00004.safetensors": 104, "model-00002-of-00004.safetensors": 122, "model-00003-of-00004.safetensors": 366, "model-00004-of-00004.safetensors": 240}
INDEX_BYTES = 64_645
MAX_HEADER_BYTES = 64 * 1024 * 1024
COMPANIONS = {
    "campplus.onnx": 28_303_423,
    "speech_tokenizer_v2_25hz.onnx": 496_082_973,
    "flow.pt": 623_466_603,
    "hift.pt": 83_390_254,
    "flow.yaml": 1_099,
}
DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
MAX_TENSOR_ELEMENTS = 1 << 40
HF_TRANSPORT_CACHE = ".cache/huggingface"
SOURCE_ROLE_BLOBS = {
    "LICENSE": "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64",
    "README.md": "3ec3ddabbdac14f65776a6553b2cbeb1aff69fcc",
    "stepaudio2.py": "1598da2a37e177aa4de093943969460f46359bb7",
    "token2wav.py": "ad4163edc2318778a4ad604d5a30009c5d959e99",
    "cosyvoice2/flow.py": "f252d9b8c90668e92db30d63a404ab7cb3ba905f",
    "cosyvoice2/flow_matching.py": "900c71e265d6c6de7fce84a8c7e3ec4e40162619",
    "cosyvoice2/decoder_dit.py": "cb80edc768fbb42b62824321286bcc21c86bfa3d",
    "flashcosyvoice/cosyvoice2.py": "9a1ede5efd965a12614f15212a01136060763d88",
    "flashcosyvoice/modules/flow.py": "b6ae2816ad5e28e362db6e6c9510bdb8e592cc6d",
    "flashcosyvoice/modules/hifigan.py": "72e49728f0fcd319f1c0a02520954c25d34d2631",
    "flashcosyvoice/modules/qwen2.py": "a19ff49d44e3b82cac0db105c86efd926ac635c0",
    "flashcosyvoice/modules/sampler.py": "329f53af8b4efe798e19505f4ecb103b98fbd6a4",
}
TRANSFORMERS_ROLE_BLOBS = {
    "LICENSE": "68b7d66c97d66c58de883ed0c451af2b3183e6f3",
    "src/transformers/models/qwen2/configuration_qwen2.py": "2e82f1976f3922f3620415f4eace6c6e046243f8",
    "src/transformers/models/qwen2/modeling_qwen2.py": "bf135a46c8d707f5c704cdaf4f766950061b6fc2",
    "src/transformers/models/qwen2_audio/configuration_qwen2_audio.py": "bcfa2ca48e60c7138a0b401f5e44db22bf857c79",
    "src/transformers/models/qwen2_audio/modeling_qwen2_audio.py": "320d2093133fe0cc51c2b76eacb68c2431d4e118",
    "src/transformers/models/whisper/feature_extraction_whisper.py": "1519fb02862364293479bb344a48c5f1e06dc275",
}
FORBIDDEN_MODEL_FILES = {"preprocessor_config.json", "generation_config.json"}


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def blob_sha1(path: Path) -> str:
    h = hashlib.sha1()
    h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_model_card_frontmatter(text: str) -> dict[str, str]:
    """Authenticate only one top-level scalar license; do not claim YAML parsing."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise RuntimeError("model-card frontmatter is missing")
    try:
        end = next(index for index in range(1, len(lines)) if lines[index].strip() == "---")
    except StopIteration as error:
        raise RuntimeError("model-card frontmatter is unterminated") from error
    result: dict[str, str] = {}
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[0].isspace() or line.startswith("-"):
            continue
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_-]*):[ \t]*(.*)", line)
        if match is None:
            raise RuntimeError("model-card frontmatter has malformed top-level YAML")
        key, raw = match.groups()
        if key != "license":
            continue
        if key in result or not raw.strip():
            raise RuntimeError("model-card license is duplicated or non-scalar")
        value = raw.strip().strip('"\'')
        if value != "apache-2.0":
            raise RuntimeError("model-card license is not exactly apache-2.0")
        result[key] = value
    if result.get("license") != "apache-2.0":
        raise RuntimeError("model-card license is not exactly apache-2.0")
    return result


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=strict_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError, RuntimeError) as error:
        raise RuntimeError(f"invalid strict JSON {path}: {error}") from error


def safe_relative(value: str, label: str) -> None:
    path = Path(value)
    if not value or "\x00" in value or "\\" in value or path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"unsafe {label} path: {value!r}")


def license_evidence(root: Path, *, unknown_status: str) -> dict[str, Any]:
    path = root / "LICENSE"
    if not path.is_file():
        return {"license": "UNKNOWN", "status": unknown_status, "path": None, "sha256": None}
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    markers = {
        "title": "apache license, version 2.0" in text,
        "grant": "you may obtain a copy of the license" in text and "you may not use this file except in compliance with the license" in text,
        "warranty": "without warranties or conditions of any kind" in text,
    }
    if not all(markers.values()):
        return {"license": "UNKNOWN", "status": unknown_status, "path": "LICENSE", "sha256": digest(path), "clauses": markers}
    return {"license": "Apache-2.0", "status": "AUTHENTICATED", "path": "LICENSE", "sha256": digest(path), "clauses": markers}


def require_path(document: Any, path: tuple[str, ...], expected: Any) -> None:
    value = document
    for part in path:
        if not isinstance(value, dict) or part not in value:
            raise RuntimeError(f"missing canonical config field: {'.'.join(path)}")
        value = value[part]
    if value != expected:
        raise RuntimeError(f"canonical config mismatch at {'.'.join(path)}: {value!r} != {expected!r}")


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT).strip()


def lfs_pointer_sha1(payload_sha256: str, payload_size: int) -> str:
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{payload_sha256}\nsize {payload_size}\n".encode()
    h = hashlib.sha1()
    h.update(f"blob {len(pointer)}\0".encode())
    h.update(pointer)
    return h.hexdigest()


def inventory_snapshot(root: Path, server_tree: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    envelope = load_json(server_tree)
    if not isinstance(envelope, dict) or set(envelope) != {"repository", "requested_revision", "resolved_revision", "walk", "files"} or envelope.get("repository") != HF_REPOSITORY or envelope.get("requested_revision") != HF_REVISION or envelope.get("resolved_revision") != HF_REVISION or envelope.get("walk") != "recursive_file_only":
        raise RuntimeError("HF server-tree identity mismatch")
    expected = envelope.get("files")
    if not isinstance(expected, list):
        raise RuntimeError("HF server-tree files must be a list")
    actual: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if relative == ".cache":
            if path.is_symlink() or not path.is_dir():
                raise RuntimeError("HF cache parent must be a real directory")
            continue
        if relative == HF_TRANSPORT_CACHE:
            if path.is_symlink() or not path.is_dir():
                raise RuntimeError("HF transport cache must be a real directory")
            continue
        if relative.startswith(HF_TRANSPORT_CACHE + "/"):
            continue
        if ".cache" in path.relative_to(root).parts:
            raise RuntimeError(f"unexpected cache outside {HF_TRANSPORT_CACHE}: {relative}")
        if path.is_symlink():
            raise RuntimeError(f"snapshot payload symlink is not authenticated: {relative}")
        if path.is_file():
            actual.append(relative)
        elif not path.is_dir():
            raise RuntimeError(f"snapshot non-regular member: {path}")
    rows: list[dict[str, Any]] = []
    names: set[str] = set()
    for item in expected:
        required = {"path", "size", "git_blob_sha1", "lfs_pointer_git_blob_sha1", "lfs_payload_sha256", "lfs_payload_size"}
        if not isinstance(item, dict) or set(item) != required or not isinstance(item.get("path"), str) or not isinstance(item.get("size"), int) or isinstance(item["size"], bool) or item["size"] <= 0:
            raise RuntimeError("server-tree item has invalid exact identity fields")
        path_name = item["path"]
        safe_relative(path_name, "server-tree")
        if path_name == HF_TRANSPORT_CACHE or path_name.startswith(HF_TRANSPORT_CACHE + "/"):
            raise RuntimeError("server tree unexpectedly contains the local HF transport cache")
        if path_name in names:
            raise RuntimeError(f"duplicate server-tree path: {path_name}")
        names.add(path_name)
        path = root / path_name
        if not path.is_file() or path.stat().st_size != item["size"]:
            raise RuntimeError(f"server/local path-size mismatch: {path_name}")
        sha = digest(path)
        git_blob = item["git_blob_sha1"]
        pointer_blob = item["lfs_pointer_git_blob_sha1"]
        payload_sha = item["lfs_payload_sha256"]
        payload_size = item["lfs_payload_size"]
        if payload_sha is None:
            if not isinstance(git_blob, str) or len(git_blob) != 40 or any(c not in "0123456789abcdef" for c in git_blob) or pointer_blob is not None or payload_size is not None or blob_sha1(path).lower() != git_blob.lower():
                raise RuntimeError(f"server/local Git blob mismatch: {path_name}")
        else:
            if git_blob is not None or not isinstance(pointer_blob, str) or len(pointer_blob) != 40 or any(c not in "0123456789abcdef" for c in pointer_blob.lower()) or not isinstance(payload_sha, str) or len(payload_sha) != 64 or any(c not in "0123456789abcdef" for c in payload_sha.lower()) or not isinstance(payload_size, int) or isinstance(payload_size, bool) or payload_size != item["size"] or sha.lower() != payload_sha.lower() or lfs_pointer_sha1(payload_sha.lower(), payload_size) != pointer_blob.lower():
                raise RuntimeError(f"server/local LFS pointer/payload mismatch: {path_name}")
        rows.append({"path": path_name, "bytes": path.stat().st_size, "sha256": sha, "git_blob_sha1": git_blob, "lfs_pointer_git_blob_sha1": pointer_blob, "lfs_payload_sha256": payload_sha, "lfs_payload_size": payload_size})
    if set(actual) != names:
        raise RuntimeError("HF server-tree/local snapshot set mismatch")
    return {"repository": envelope["repository"], "requested_revision": envelope["requested_revision"], "resolved_revision": envelope["resolved_revision"], "walk": envelope["walk"]}, rows


def inspect_safetensors(path: Path, snapshot: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise RuntimeError(f"truncated safetensors header: {path}")
        header_bytes = int.from_bytes(prefix, "little")
        if header_bytes <= 0 or header_bytes > MAX_HEADER_BYTES or header_bytes > size - 8:
            raise RuntimeError(f"invalid safetensors header length: {path}")
        raw = stream.read(header_bytes)
    try:
        header = json.loads(raw.decode(), object_pairs_hook=strict_pairs)
    except (UnicodeError, json.JSONDecodeError, RuntimeError) as error:
        raise RuntimeError(f"invalid safetensors header JSON: {path}: {error}") from error
    if not isinstance(header, dict):
        raise RuntimeError("safetensors header is not an object")
    metadata = header.get("__metadata__")
    if metadata is not None and (not isinstance(metadata, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items())):
        raise RuntimeError(f"invalid safetensors metadata: {path}")
    payload = size - 8 - header_bytes
    intervals: list[tuple[int, int, str]] = []
    tensors: list[dict[str, Any]] = []
    for name, record in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(record, dict) or set(record) != {"dtype", "shape", "data_offsets"}:
            raise RuntimeError(f"invalid tensor record: {name}")
        dtype, shape, offsets = record["dtype"], record["shape"], record["data_offsets"]
        if dtype not in DTYPE_BYTES or not isinstance(shape, list) or not shape or len(shape) > 8 or not isinstance(offsets, list) or len(offsets) != 2:
            raise RuntimeError(f"invalid tensor dtype/shape/offsets: {name}")
        elements = 1
        for dimension in shape:
            if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0:
                raise RuntimeError(f"invalid tensor shape: {name}")
            elements *= dimension
            if elements > MAX_TENSOR_ELEMENTS:
                raise RuntimeError(f"tensor shape is unbounded: {name}")
        start, end = offsets
        if any(not isinstance(v, int) or isinstance(v, bool) for v in offsets) or start < 0 or end < start or end > payload or end - start != elements * DTYPE_BYTES[dtype]:
            raise RuntimeError(f"invalid tensor data region: {name}")
        intervals.append((start, end, name))
        tensors.append({"name": name, "dtype": dtype, "shape": shape, "elements": elements, "data_bytes": end - start, "shard": path.relative_to(snapshot).as_posix()})
    cursor = 0
    for start, end, name in sorted(intervals):
        if start != cursor:
            raise RuntimeError(f"safetensors overlap/gap before {name}")
        cursor = end
    if cursor != payload:
        raise RuntimeError("safetensors tail gap")
    with safe_open(str(path), framework="pt") as handle:
        if set(handle.keys()) != {item["name"] for item in tensors}:
            raise RuntimeError(f"safe_open/header key mismatch: {path}")
    return {"path": path.relative_to(snapshot).as_posix(), "bytes": size, "sha256": digest(path), "header_bytes": header_bytes, "data_bytes": payload, "tensor_count": len(tensors)}, tensors


def inventory_weights(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index_path = root / "model.safetensors.index.json"
    if index_path.stat().st_size != INDEX_BYTES:
        raise RuntimeError(f"index byte-size mismatch: {index_path.stat().st_size} != {INDEX_BYTES}")
    index = load_json(index_path)
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise RuntimeError("missing/invalid safetensors weight_map")
    metadata = index.get("metadata")
    if set(index) != {"metadata", "weight_map"} or not isinstance(metadata, dict) or set(metadata) != {"total_size"} or isinstance(metadata.get("total_size"), bool) or metadata.get("total_size") != 16_630_358_528:
        raise RuntimeError("index metadata.total_size mismatch")
    mapping = index["weight_map"]
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in mapping.items()):
        raise RuntimeError("weight_map keys/values must be strings")
    for value in mapping.values():
        safe_relative(value, "weight_map")
    if {Path(v).name for v in mapping.values()} != SHARD_NAMES or any(v != Path(v).name for v in mapping.values()):
        raise RuntimeError("weight_map shard set/path mismatch")
    actual = {p.name for p in root.glob("*.safetensors")}
    if actual != SHARD_NAMES:
        raise RuntimeError(f"expected exact four-shard set, got {sorted(actual)}")
    shards: list[dict[str, Any]] = []
    tensors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(root.glob("*.safetensors")):
        shard, records = inspect_safetensors(path, root)
        expected_bytes, expected_sha = SHARD_BYTES[path.name]
        if shard["bytes"] != expected_bytes or shard["sha256"] != expected_sha:
            raise RuntimeError(f"authenticated shard bytes/SHA mismatch: {path.name}")
        if len(records) != SHARD_TENSOR_COUNTS[path.name]:
            raise RuntimeError(f"authenticated shard tensor-count mismatch: {path.name}")
        for record in records:
            if record["name"] in seen or mapping.get(record["name"]) != path.name:
                raise RuntimeError(f"weight_map tensor coverage mismatch: {record['name']}")
            seen.add(record["name"])
        shards.append(shard)
        tensors.extend(records)
    if seen != set(mapping) or len(tensors) != 832:
        raise RuntimeError("weight_map tensor count/coverage mismatch")
    return shards, tensors


def inspect_onnx(path: Path, root: Path) -> dict[str, Any]:
    import onnx
    model = onnx.load(str(path), load_external_data=False)
    def value_info(value: Any) -> dict[str, Any]:
        tensor = value.type.tensor_type
        return {"name": value.name, "elem_type": tensor.elem_type, "shape": [{"dim_value": d.dim_value, "dim_param": d.dim_param} for d in tensor.shape.dim]}
    initializers = []
    for tensor in model.graph.initializer:
        external = {entry.key: entry.value for entry in tensor.external_data}
        if len(external) != len(tensor.external_data):
            raise RuntimeError(f"duplicate ONNX external-data key: {tensor.name}")
        if "location" in external:
            safe_relative(external["location"], "ONNX external-data")
            snapshot_root = root.resolve()
            external_candidate = path.parent / external["location"]
            if external_candidate.is_symlink():
                raise RuntimeError(f"ONNX external-data symlink is not authenticated: {external['location']}")
            external_path = external_candidate.resolve()
            try:
                external_path.relative_to(snapshot_root)
            except ValueError as error:
                raise RuntimeError(f"ONNX external-data escapes authenticated snapshot: {external['location']}") from error
            if not external_path.is_file() or external_path.is_symlink():
                raise RuntimeError(f"missing ONNX external-data file: {external['location']}")
            for key in ("offset", "length"):
                if key in external and (not external[key].isdigit() or int(external[key]) < 0):
                    raise RuntimeError(f"invalid ONNX external-data {key}: {tensor.name}")
            offset = int(external.get("offset", "0"))
            length = int(external.get("length", str(external_path.stat().st_size - offset)))
            if offset > external_path.stat().st_size or length < 0 or offset + length > external_path.stat().st_size:
                raise RuntimeError(f"ONNX external-data range exceeds file: {tensor.name}")
        initializers.append({"name": tensor.name, "data_type": tensor.data_type, "dims": list(tensor.dims), "raw_data_bytes": len(tensor.raw_data), "external_data": external})
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path), "opsets": [{"domain": op.domain, "version": op.version} for op in model.opset_import], "inputs": [value_info(v) for v in model.graph.input], "outputs": [value_info(v) for v in model.graph.output], "nodes": [{"op_type": n.op_type, "domain": n.domain, "inputs": list(n.input), "outputs": list(n.output)} for n in model.graph.node], "initializers": initializers, "execution": "NOT_PERFORMED"}


def inspect_pt(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError(f"duplicate torch archive member: {path}")
        for info in archive.infolist():
            safe_relative(info.filename, "torch archive")
            mode = (info.external_attr >> 16) & 0xFFFF
            if mode and stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise RuntimeError(f"non-regular torch archive member: {info.filename}")
    import torch
    unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(path)
    if unsafe:
        raise RuntimeError(f"torch checkpoint has unsafe globals: {unsafe}")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise RuntimeError("torch companion is not a string-keyed state dict")
    tensors = []
    for name, tensor in value.items():
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"torch state dict contains non-tensor: {name}")
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"torch state dict contains non-finite tensor: {name}")
        tensors.append({"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape), "elements": tensor.numel()})
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path), "unsafe_globals": [], "resident_scope": "one weights_only state_dict; no model execution", "tensor_count": len(tensors), "tensors": tensors}


def inspect_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if any(line.lstrip().startswith("!") or "!!python" in line.lower() for line in text.splitlines()):
        raise RuntimeError(f"unsafe YAML constructor/tag in {path}")
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path), "parsed": False, "constructor_execution": "NOT_PERFORMED", "class_references": [line.strip() for line in text.splitlines() if "class" in line.lower()]}


def find_companion(root: Path, basename: str) -> Path:
    matches = [path for path in root.rglob(basename) if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"required companion has {len(matches)} matches: {basename}")
    return matches[0]


def source_inventory(
    source: Path,
    transformers: Path,
    *,
    source_revision: str = SOURCE_REVISION,
    transformers_revision: str = TRANSFORMERS_REVISION,
    transformers_tag: str = TRANSFORMERS_TAG,
    source_roles: dict[str, str] | None = None,
    transformer_roles: dict[str, str] | None = None,
) -> dict[str, Any]:
    source_roles = SOURCE_ROLE_BLOBS if source_roles is None else source_roles
    transformer_roles = TRANSFORMERS_ROLE_BLOBS if transformer_roles is None else transformer_roles
    if git(source, "rev-parse", "HEAD") != source_revision or git(transformers, "rev-parse", "HEAD") != transformers_revision:
        raise RuntimeError("source revision mismatch")
    if git(transformers, "describe", "--exact-match", "--tags") != transformers_tag:
        raise RuntimeError("Transformers exact tag mismatch")
    def remote(root: Path, expected: str) -> str:
        value = git(root, "remote", "get-url", "origin").removesuffix(".git").removesuffix("/")
        if value != expected.removesuffix(".git"):
            raise RuntimeError(f"source origin mismatch: {value}")
        return value
    source_origin, transformer_origin = remote(source, SOURCE_REPOSITORY), remote(transformers, TRANSFORMERS_REPOSITORY)
    def tracked_records(root: Path) -> list[dict[str, Any]]:
        entries = git(root, "ls-files", "--stage", "-z").split("\0")
        result = []
        for entry in entries:
            if not entry:
                continue
            metadata, relative = entry.split("\t", 1)
            mode, index_object, stage = metadata.split()
            path = root / relative
            if stage != "0":
                raise RuntimeError(f"tracked source entry is not stage 0: {relative}")
            if mode == "160000" or mode not in {"100644", "100755"}:
                raise RuntimeError(f"unsupported tracked mode/gitlink: {relative} ({mode})")
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"tracked source file is missing/non-regular: {relative}")
            expected_mode = 0o755 if mode == "100755" else 0o644
            if stat.S_IMODE(path.stat().st_mode) != expected_mode:
                raise RuntimeError(f"working-tree mode drift: {relative}")
            head_object = git(root, "rev-parse", f"HEAD:{relative}")
            working_object = blob_sha1(path)
            if index_object != head_object or head_object != working_object:
                raise RuntimeError(f"tracked source identity drift: {relative}")
            result.append({"path": relative, "mode": mode, "stage": stage, "bytes": path.stat().st_size, "sha256": digest(path), "index_object_sha1": index_object, "git_blob_sha1": head_object, "working_blob_sha1": working_object})
        if git(root, "status", "--porcelain", "--untracked-files=all"):
            raise RuntimeError("official source checkout is dirty")
        return sorted(result, key=lambda row: row["path"])
    source_files = tracked_records(source)
    transformer_files = tracked_records(transformers)
    def fixed_roles(root: Path, tracked: list[dict[str, Any]], expected: dict[str, str]) -> list[dict[str, Any]]:
        by_path = {row["path"]: row for row in tracked}
        rows = []
        for relative, expected_blob in expected.items():
            row = by_path.get(relative)
            if row is None or row["index_object_sha1"] != expected_blob or row["git_blob_sha1"] != expected_blob or row["working_blob_sha1"] != expected_blob:
                raise RuntimeError(f"authenticated source role mismatch: {relative}")
            rows.append(row)
        return rows
    source_license = license_evidence(source, unknown_status="SOURCE_LICENSE_UNKNOWN_BLOCKER")
    transformer_license = license_evidence(transformers, unknown_status="TRANSFORMERS_LICENSE_UNKNOWN_BLOCKER")
    if source_license["status"] != "AUTHENTICATED" or transformer_license["status"] != "AUTHENTICATED":
        raise RuntimeError("source Apache-2.0 license clauses are not authenticated")
    source_role_rows = fixed_roles(source, source_files, source_roles)
    transformer_role_rows = fixed_roles(transformers, transformer_files, transformer_roles)
    return {"status": "REFERENCE_SOURCE_SELECTED", "repository": SOURCE_REPOSITORY, "revision": source_revision, "origin": source_origin, "clean": True, "license": source_license, "files": source_files, "role_files": source_role_rows, "transformers": {"repository": TRANSFORMERS_REPOSITORY, "tag": transformers_tag, "revision": transformers_revision, "origin": transformer_origin, "clean": True, "license": transformer_license, "files": transformer_files, "role_files": transformer_role_rows}}


def write_blocked(output: Path, error: Exception, **extra: Any) -> None:
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"format": FORMAT, "status": "BLOCKED", "evidence_stage": "INSPECTION_ONLY", "inspection_status": "INSPECTION_ERROR", "collection_status": "UNVERIFIED", "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "cpu_status": "UNSUPPORTED", "metal_status": "BLOCKED_BY_CPU", "parity_status": "NOT_RUN", "publication": "NO_UPLOAD", "task": "speech-to-speech composite; no native/ONNX runtime claim", "upstream": {"repository": HF_REPOSITORY, "requested_revision": HF_REVISION, "resolved_revision": None, "status": "UNVERIFIED", "walk": "recursive_file_only"}, "official_source": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION, "status": "UNVERIFIED"}, "transformers": {"repository": TRANSFORMERS_REPOSITORY, "tag": TRANSFORMERS_TAG, "revision": TRANSFORMERS_REVISION, "status": "UNVERIFIED"}, "error_type": type(error).__name__, "reason": str(error), "blockers": [str(error)], **extra}
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def inspect(snapshot: Path, source: Path, transformers: Path, server_tree: Path, output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    identity, files = inventory_snapshot(snapshot, server_tree)
    server_by_path = {row["path"]: row for row in files}
    def server_identity(relative: str) -> dict[str, Any]:
        row = server_by_path.get(relative)
        if row is None:
            raise RuntimeError(f"missing authenticated server identity: {relative}")
        return {key: row[key] for key in ("path", "bytes", "sha256", "git_blob_sha1", "lfs_pointer_git_blob_sha1", "lfs_payload_sha256", "lfs_payload_size")}
    readme = snapshot / "README.md"
    if not readme.is_file():
        raise RuntimeError("canonical HF model card README.md is missing")
    model_card = parse_model_card_frontmatter(readme.read_text(encoding="utf-8"))
    model_license = {"path": "README.md", "sha256": digest(readme), "license": "Apache-2.0", "frontmatter": model_card, "server_identity": server_identity("README.md")}
    forbidden = sorted(name for name in FORBIDDEN_MODEL_FILES if (snapshot / name).exists())
    if forbidden:
        raise RuntimeError(f"canonical HF tree unexpectedly contains forbidden JSON companions: {forbidden}")
    required_json = ("config.json", "tokenizer_config.json")
    missing_json = [name for name in required_json if not (snapshot / name).is_file()]
    if missing_json:
        raise RuntimeError(f"required Step-Audio JSON companion missing: {missing_json}")
    parsed = {name: load_json(snapshot / name) for name in required_json}
    json_identities = {name: server_identity(name) for name in required_json}
    if "config.json" not in parsed or parsed["config.json"].get("architectures") != ["StepAudio2ForCausalLM"]:
        raise RuntimeError("StepAudio2ForCausalLM config contract missing")
    config = parsed["config.json"]
    for path, expected in {
        ("text_config", "hidden_size"): 3584,
        ("text_config", "intermediate_size"): 18944,
        ("text_config", "num_hidden_layers"): 28,
        ("text_config", "num_attention_heads"): 28,
        ("text_config", "num_key_value_heads"): 4,
        ("text_config", "max_position_embeddings"): 16384,
        ("text_config", "vocab_size"): 158720,
        ("audio_encoder_config", "n_mels"): 128,
        ("audio_encoder_config", "n_audio_ctx"): 1500,
        ("audio_encoder_config", "n_audio_state"): 1280,
        ("audio_encoder_config", "n_audio_head"): 20,
        ("audio_encoder_config", "n_audio_layer"): 32,
        ("audio_encoder_config", "n_codebook_size"): 4096,
        ("audio_encoder_config", "llm_dim"): 3584,
        ("audio_encoder_config", "kernel_size"): 3,
        ("audio_encoder_config", "adapter_stride"): 2,
    }.items():
        require_path(config, path, expected)
    custom_code = []
    for basename in ("configuration_step_audio_2.py", "modeling_step_audio_2.py"):
        path = find_companion(snapshot, basename)
        relative = path.relative_to(snapshot).as_posix()
        custom_code.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest(path), "server_identity": server_identity(relative)})
    shards, tensors = inventory_weights(snapshot)
    index_identity = server_identity("model.safetensors.index.json")
    for shard in shards:
        shard["server_identity"] = server_identity(shard["path"])
    companions = {}
    for name, expected in COMPANIONS.items():
        path = find_companion(snapshot, name)
        if not path.is_file() or path.stat().st_size != expected:
            raise RuntimeError(f"required token2wav companion missing/size mismatch: {name}")
        if name.endswith(".onnx"):
            companions[name] = inspect_onnx(path, snapshot)
        elif name.endswith(".pt"):
            companions[name] = inspect_pt(path)
        else:
            companions[name] = inspect_yaml(path)
        companions[name]["path"] = path.relative_to(snapshot).as_posix()
        companions[name]["server_identity"] = server_identity(companions[name]["path"])
    sources = source_inventory(source, transformers)
    (output / "snapshot-inventory.json").write_text(json.dumps({"server_tree": identity, "files": files}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "tensor-inventory.json").write_text(json.dumps({"shards": shards, "tensor_count": len(tensors), "tensors": tensors}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "parsed-json.json").write_text(json.dumps(parsed, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "companion-inventory.json").write_text(json.dumps(companions, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "source-inventory.json").write_text(json.dumps({"official": sources, "hf_custom_code": custom_code}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    packets = {path.name: {"bytes": path.stat().st_size, "sha256": digest(path)} for path in output.glob("*-inventory.json")}
    write_blocked(output, RuntimeError("component/license/dataset provenance remains unauthenticated; inspection only"), inspection_status="AUTHENTICATED_EVIDENCE_COMPLETE", collection_status="AUTHENTICATED", upstream={"repository": HF_REPOSITORY, "requested_revision": HF_REVISION, "resolved_revision": HF_REVISION, "status": "AUTHENTICATED", "walk": "recursive_file_only", "license": model_license, "server_tree": identity, "files": files}, parsed_json=parsed, json_identities=json_identities, index_identity=index_identity, hf_custom_code=custom_code, shards=shards, tensor_count=len(tensors), companions=companions, official_source=sources, license_evidence={"model": model_license, "official_source": sources["license"], "transformers": sources["transformers"]["license"]}, dataset_provenance={"status": "BLOCKED_UNAUTHENTICATED"}, packets=packets)
    return 2


def self_test() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    assert "weights_only=True" in source and "load_external_data=False" in source and ("onnxruntime." + "InferenceSession") not in source and ("ort." + "InferenceSession") not in source
    assert FORBIDDEN_MODEL_FILES == {"preprocessor_config.json", "generation_config.json"}
    assert "audio_encoder_config" in source and "configuration_step_audio_2.py" in source and "modeling_step_audio_2.py" in source
    assert len(HF_REVISION) == 40 and len(SOURCE_REVISION) == 40 and len(TRANSFORMERS_REVISION) == 40
    assert SHARD_BYTES["model-00001-of-00004.safetensors"][0] == 4_925_370_984
    assert MAX_HEADER_BYTES == 64 * 1024 * 1024
    try:
        json.loads('{"x":1,"x":2}', object_pairs_hook=strict_pairs)
    except RuntimeError:
        pass
    else:
        raise AssertionError("duplicate JSON key accepted")
    with tempfile.TemporaryDirectory(prefix="vokra-step-audio2-inspect-") as directory:
        root = Path(directory)
        huge = root / "huge.safetensors"
        huge.write_bytes((MAX_HEADER_BYTES + 1).to_bytes(8, "little"))
        try:
            inspect_safetensors(huge, root)
        except RuntimeError:
            pass
        else:
            raise AssertionError("huge header accepted")
        yaml = root / "flow.yaml"
        yaml.write_text("!python/object:evil\n", encoding="utf-8")
        try:
            inspect_yaml(yaml)
        except RuntimeError:
            pass
        else:
            raise AssertionError("unsafe YAML constructor accepted")
        safe = root / "safe.yaml"
        safe.write_text("class: flow\n", encoding="utf-8")
        assert inspect_yaml(safe)["constructor_execution"] == "NOT_PERFORMED"
        canonical_card = "---\nlicense: apache-2.0\ndatasets:\n- speech/example\ntags:\n- audio\nmodel-index:\n- name: Step-Audio\n  results:\n  - task:\n      type: audio\n---\nmodel card"
        assert parse_model_card_frontmatter(canonical_card) == {"license": "apache-2.0"}
        for card in (
            "prose license: apache-2.0",
            "---\nlicense: apache-2.0\nlicense: apache-2.0\n---",
            "---\ndatasets:\n  license: apache-2.0\n---",
            "---\nlicense:\n- apache-2.0\n---",
        ):
            try:
                parse_model_card_frontmatter(card)
            except RuntimeError:
                pass
            else:
                raise AssertionError("invalid/nested model-card license accepted")
        def git_fixture(path: Path, roles: dict[str, str], origin: str, tag: str | None = None) -> tuple[str, dict[str, str]]:
            path.mkdir()
            subprocess.run(["git", "init", "-q", str(path)], check=True)
            subprocess.run(["git", "-C", str(path), "config", "user.email", "step-audio2-selftest@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(path), "config", "user.name", "Step-Audio2 self-test"], check=True)
            apache = "Apache License, Version 2.0\nYou may obtain a copy of the License.\nYou may not use this file except in compliance with the License.\nSoftware distributed under the License is distributed on an AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.\n"
            expected: dict[str, str] = {}
            for relative in roles:
                file_path = path / relative
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(apache if relative == "LICENSE" else f"fixture role {relative}\n", encoding="utf-8")
                expected[relative] = blob_sha1(file_path)
            subprocess.run(["git", "-C", str(path), "add", *roles], check=True)
            subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "fixture"], check=True)
            subprocess.run(["git", "-C", str(path), "remote", "add", "origin", origin], check=True)
            if tag:
                subprocess.run(["git", "-C", str(path), "tag", tag], check=True)
            revision = git(path, "rev-parse", "HEAD")
            return revision, expected
        source_fixture = root / "source-fixture"
        transformers_fixture = root / "transformers-fixture"
        source_fixture_revision, source_fixture_roles = git_fixture(source_fixture, SOURCE_ROLE_BLOBS, SOURCE_REPOSITORY)
        transformers_fixture_revision, transformers_fixture_roles = git_fixture(transformers_fixture, TRANSFORMERS_ROLE_BLOBS, TRANSFORMERS_REPOSITORY, "fixture-transformers")
        selected = source_inventory(source_fixture, transformers_fixture, source_revision=source_fixture_revision, transformers_revision=transformers_fixture_revision, transformers_tag="fixture-transformers", source_roles=source_fixture_roles, transformer_roles=transformers_fixture_roles)
        assert selected["status"] == "REFERENCE_SOURCE_SELECTED"
        assert all(row["stage"] == "0" and row["index_object_sha1"] == row["git_blob_sha1"] == row["working_blob_sha1"] for row in selected["files"] + selected["transformers"]["files"])
        spoof_roles = dict(source_fixture_roles)
        spoof_roles["LICENSE"] = "0" * 40
        try:
            source_inventory(source_fixture, transformers_fixture, source_revision=source_fixture_revision, transformers_revision=transformers_fixture_revision, transformers_tag="fixture-transformers", source_roles=spoof_roles, transformer_roles=transformers_fixture_roles)
        except RuntimeError:
            pass
        else:
            raise AssertionError("source role object spoof accepted")
        subprocess.run(["git", "-C", str(transformers_fixture), "update-index", "--chmod=+x", "LICENSE"], check=True)
        try:
            source_inventory(source_fixture, transformers_fixture, source_revision=source_fixture_revision, transformers_revision=transformers_fixture_revision, transformers_tag="fixture-transformers", source_roles=source_fixture_roles, transformer_roles=transformers_fixture_roles)
        except RuntimeError:
            pass
        else:
            raise AssertionError("tracked mode drift accepted")
        subprocess.run(["git", "-C", str(transformers_fixture), "reset", "-q", "--", "LICENSE"], check=True)
        (source_fixture / "dirty.txt").write_text("dirty\n", encoding="utf-8")
        try:
            source_inventory(source_fixture, transformers_fixture, source_revision=source_fixture_revision, transformers_revision=transformers_fixture_revision, transformers_tag="fixture-transformers", source_roles=source_fixture_roles, transformer_roles=transformers_fixture_roles)
        except RuntimeError:
            pass
        else:
            raise AssertionError("dirty source accepted")
        subprocess.run(["git", "-C", str(source_fixture), "update-index", "--force-remove", "LICENSE"], check=True)
        license_object = git(source_fixture, "rev-parse", "HEAD:LICENSE")
        subprocess.run(["git", "-C", str(source_fixture), "update-index", "--index-info"], input=f"100644 {license_object} 1\tLICENSE\n", text=True, check=True)
        try:
            source_inventory(source_fixture, transformers_fixture, source_revision=source_fixture_revision, transformers_revision=transformers_fixture_revision, transformers_tag="fixture-transformers", source_roles=source_fixture_roles, transformer_roles=transformers_fixture_roles)
        except RuntimeError:
            pass
        else:
            raise AssertionError("non-stage-0 source entry accepted")
        try:
            safe_relative("../escape", "ONNX external-data")
        except RuntimeError:
            pass
        else:
            raise AssertionError("ONNX external path traversal accepted")
        import torch
        state = root / "safe.pt"
        torch.save({"weight": torch.ones((2, 2))}, state)
        assert inspect_pt(state)["unsafe_globals"] == []
        duplicate_pt = root / "duplicate.pt"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(duplicate_pt, "w") as archive:
                archive.writestr("data.pkl", b"x")
                archive.writestr("data.pkl", b"y")
        try:
            inspect_pt(duplicate_pt)
        except RuntimeError:
            pass
        else:
            raise AssertionError("duplicate torch archive member accepted")
        import onnx
        from onnx import TensorProto, helper
        nested = root / "nested"
        nested.mkdir()
        (nested / "weights.bin").write_bytes(b"weights")
        initializer = TensorProto(name="w", data_type=TensorProto.FLOAT, dims=[1], data_location=TensorProto.EXTERNAL)
        initializer.external_data.add(key="location", value="weights.bin")
        graph = helper.make_graph([helper.make_node("Identity", ["input"], ["output"])], "fixture", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])], [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])], initializer=[initializer])
        onnx_path = nested / "fixture.onnx"
        onnx.save(helper.make_model(graph), onnx_path)
        assert inspect_onnx(onnx_path, root)["execution"] == "NOT_PERFORMED"
        graph.initializer[0].external_data[0].value = "../outside.bin"
        onnx.save(helper.make_model(graph), onnx_path)
        try:
            inspect_onnx(onnx_path, root)
        except RuntimeError:
            pass
        else:
            raise AssertionError("ONNX external path escape accepted")
        server_tree = root / "server-tree.json"
        server_tree.write_text(json.dumps({"repository": "wrong", "revision": HF_REVISION, "resolved_revision": HF_REVISION, "files": []}), encoding="utf-8")
        try:
            inventory_snapshot(root, server_tree)
        except RuntimeError:
            pass
        else:
            raise AssertionError("server identity mismatch accepted")
        materialized = root / "materialized"
        materialized.mkdir()
        regular = materialized / "regular.txt"
        regular.write_bytes(b"regular")
        payload = materialized / "payload.bin"
        payload.write_bytes(b"payload")
        packet_rows = [
            {"path": "regular.txt", "size": regular.stat().st_size, "git_blob_sha1": blob_sha1(regular), "lfs_pointer_git_blob_sha1": None, "lfs_payload_sha256": None, "lfs_payload_size": None},
            {"path": "payload.bin", "size": payload.stat().st_size, "git_blob_sha1": None, "lfs_pointer_git_blob_sha1": lfs_pointer_sha1(digest(payload), payload.stat().st_size), "lfs_payload_sha256": digest(payload), "lfs_payload_size": payload.stat().st_size},
        ]
        server_tree.write_text(json.dumps({"repository": HF_REPOSITORY, "requested_revision": HF_REVISION, "resolved_revision": HF_REVISION, "walk": "recursive_file_only", "files": packet_rows}), encoding="utf-8")
        _, verified_rows = inventory_snapshot(materialized, server_tree)
        assert {row["path"] for row in verified_rows} == {"regular.txt", "payload.bin"}
        transport = materialized / HF_TRANSPORT_CACHE
        transport.mkdir(parents=True)
        (transport / "marker").write_bytes(b"ignored")
        inventory_snapshot(materialized, server_tree)
        (transport / "marker").unlink()
        transport.rmdir()
        (materialized / "outside-cache").mkdir()
        transport.symlink_to(materialized / "outside-cache")
        try:
            inventory_snapshot(materialized, server_tree)
        except RuntimeError:
            pass
        else:
            raise AssertionError("transport-cache symlink accepted")
        transport.unlink()
        (materialized / "outside-cache").rmdir()
        nested_cache = materialized / ".cache" / "other"
        nested_cache.mkdir(parents=True)
        try:
            inventory_snapshot(materialized, server_tree)
        except RuntimeError:
            pass
        else:
            raise AssertionError("non-transport cache accepted")
        (nested_cache / "marker").unlink(missing_ok=True)
        nested_cache.rmdir()
        (materialized / ".cache").rmdir()
        failed_output = root / "failed-evidence"
        write_blocked(failed_output, RuntimeError("fixture failure"))
        failed_manifest = load_json(failed_output / "manifest.json")
        assert failed_manifest["status"] == "BLOCKED" and failed_manifest["evidence_stage"] == "INSPECTION_ONLY"
        assert failed_manifest["upstream"]["resolved_revision"] is None and failed_manifest["upstream"]["status"] == "UNVERIFIED"
        packet_rows[1]["lfs_payload_sha256"] = "c" * 64
        server_tree.write_text(json.dumps({"repository": HF_REPOSITORY, "requested_revision": HF_REVISION, "resolved_revision": HF_REVISION, "walk": "recursive_file_only", "files": packet_rows}), encoding="utf-8")
        try:
            inventory_snapshot(materialized, server_tree)
        except RuntimeError:
            pass
        else:
            raise AssertionError("spoofed LFS payload accepted")
        packet_rows[1]["lfs_payload_sha256"] = digest(payload)
        os.symlink(regular, materialized / "payload-link")
        packet_rows.append({"path": "payload-link", "size": regular.stat().st_size, "git_blob_sha1": blob_sha1(regular), "lfs_pointer_git_blob_sha1": None, "lfs_payload_sha256": None, "lfs_payload_size": None})
        server_tree.write_text(json.dumps({"repository": HF_REPOSITORY, "requested_revision": HF_REVISION, "resolved_revision": HF_REVISION, "walk": "recursive_file_only", "files": packet_rows}), encoding="utf-8")
        try:
            inventory_snapshot(materialized, server_tree)
        except RuntimeError:
            pass
        else:
            raise AssertionError("payload symlink accepted")
        normal_output = root / "normal-evidence"
        write_blocked(normal_output, RuntimeError("inspection remains publicly blocked"), inspection_status="AUTHENTICATED_EVIDENCE_COMPLETE", collection_status="AUTHENTICATED")
        normal_manifest = load_json(normal_output / "manifest.json")
        assert normal_manifest["status"] == "BLOCKED" and normal_manifest["inspection_status"] == "AUTHENTICATED_EVIDENCE_COMPLETE" and normal_manifest["collection_status"] == "AUTHENTICATED"
        failed_output = root / "error-evidence"
        write_blocked(failed_output, RuntimeError("collection failed"))
        failed_manifest = load_json(failed_output / "manifest.json")
        assert failed_manifest["inspection_status"] == "INSPECTION_ERROR" and failed_manifest["collection_status"] == "UNVERIFIED" and failed_manifest["upstream"]["resolved_revision"] is None
    print("step_audio2_mini_inspect.py self-test: OK (strict JSON/header/ONNX/PT/YAML fail-closed contracts)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--transformers", type=Path)
    parser.add_argument("--server-tree", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.snapshot, args.source, args.transformers, args.server_tree, args.output)):
            parser.error("--self-test accepts no other arguments")
        self_test()
        return 0
    if any(value is None for value in (args.snapshot, args.source, args.transformers, args.server_tree, args.output)):
        parser.error("all inspection paths are required")
    try:
        return inspect(args.snapshot, args.source, args.transformers, args.server_tree, args.output)
    except Exception as error:
        write_blocked(args.output, error)
        print(f"Step-Audio-2 inspection BLOCKED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
