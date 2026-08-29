#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Fail-closed VAST inspection oracle for Qwen2.5-Omni-7B.

Only headers, metadata, archives, and source/config structure are inspected.
No model code, ONNX/runtime, conversion, or parity claim is made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import stat
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from safetensors import safe_open

HF_REPOSITORY = "Qwen/Qwen2.5-Omni-7B"
HF_REVISION = "ae9e1690543ffd5c0221dc27f79834d0294cba00"
SOURCE_REPOSITORY = "https://github.com/QwenLM/Qwen2.5-Omni.git"
SOURCE_REVISION = "d8a31ca56c0456b6edfcbcbf4bdbb6ae2200ef42"
TRANSFORMERS_REPOSITORY = "https://github.com/huggingface/transformers.git"
TRANSFORMERS_TAG = "v4.52.3"
TRANSFORMERS_REVISION = "f4fc42216cd56ab6b68270bf80d811614d8d59e4"
FORMAT = "vokra-qwen2-5-omni-7b-inspection-v1"
MAX_HEADER_BYTES = 64 * 1024 * 1024
SHARD_NAMES = {f"model-{i:05d}-of-00005.safetensors" for i in range(1, 6)}
SHARD_BYTES = {
    "model-00001-of-00005.safetensors": (4_985_055_504, "5edb02fd7c98803239468375cc9dc1bff492865c2aa086b78f348597021d6cbc"),
    "model-00002-of-00005.safetensors": (4_991_496_800, "7c99b55c6e5bc63fd4b19d4dc23cdc3ddac4b0101bb3c0958cc2b5d05c2bbafe"),
    "model-00003-of-00005.safetensors": (4_991_496_904, "ad00c3ac296300db905934ed213c4077ff49b85d30c0099270c814e2c77ec812"),
    "model-00004-of-00005.safetensors": (4_969_489_824, "152bc7d81441eaba22547d8d96c03d32dd592ce0e0e1d0e449347a4b23a532d3"),
    "model-00005-of-00005.safetensors": (2_425_322_160, "b8b18276481ba8cdf4fe2c98ac4c7a2da6e0d1c8a51850d162a391760cb2b81e"),
}
SHARD_TENSOR_COUNTS = {"model-00001-of-00005.safetensors": 1041, "model-00002-of-00005.safetensors": 131, "model-00003-of-00005.safetensors": 131, "model-00004-of-00005.safetensors": 270, "model-00005-of-00005.safetensors": 875}
INDEX_BYTES = 233_160
TOTAL_SIZE = 22_366_403_936
TENSOR_COUNT = 2_448
DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
TRANSFORMERS_ROLE_FILES = (
    "src/transformers/models/qwen2_5_omni/configuration_qwen2_5_omni.py",
    "src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py",
    "src/transformers/models/qwen2_5_omni/modular_qwen2_5_omni.py",
    "src/transformers/models/qwen2_5_omni/processing_qwen2_5_omni.py",
    "src/transformers/models/llama/modeling_llama.py",
    "src/transformers/models/qwen2/configuration_qwen2.py",
    "src/transformers/models/qwen2/modeling_qwen2.py",
    "src/transformers/models/qwen2_5_vl/configuration_qwen2_5_vl.py",
    "src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py",
    "src/transformers/models/qwen2_audio/configuration_qwen2_audio.py",
    "src/transformers/models/qwen2_audio/modeling_qwen2_audio.py",
    "src/transformers/models/qwen2_vl/configuration_qwen2_vl.py",
    "src/transformers/models/qwen2_vl/modeling_qwen2_vl.py",
    "src/transformers/models/whisper/configuration_whisper.py",
    "src/transformers/models/whisper/modeling_whisper.py",
    "src/transformers/models/whisper/feature_extraction_whisper.py",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
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


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=strict_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError, RuntimeError) as error:
        raise RuntimeError(f"invalid strict JSON {path}: {error}") from error


def require_path(document: Any, path: tuple[str, ...], expected: Any) -> None:
    value = document
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise RuntimeError(f"missing exact config path: {'.'.join(path)}")
        value = value[key]
    if value != expected:
        raise RuntimeError(f"config mismatch at {'.'.join(path)}: {value!r} != {expected!r}")


def validate_config(config: Any) -> dict[str, str]:
    if not isinstance(config, dict) or any(key in config for key in ("audio_config", "video_config", "vision_config")):
        raise RuntimeError("Qwen2.5-Omni config has forbidden/missing root topology")
    exact = {
        ("model_type",): "qwen2_5_omni",
        ("enable_audio_output",): True,
        ("enable_talker",): True,
        ("transformers_version",): "4.50.0.dev0",
        ("thinker_config", "model_type"): "qwen2_5_omni_thinker",
        ("thinker_config", "audio_config", "d_model"): 1280,
        ("thinker_config", "audio_config", "encoder_layers"): 32,
        ("thinker_config", "audio_config", "encoder_attention_heads"): 20,
        ("thinker_config", "audio_config", "num_mel_bins"): 128,
        ("thinker_config", "audio_config", "output_dim"): 3584,
        ("thinker_config", "text_config", "hidden_size"): 3584,
        ("thinker_config", "text_config", "num_hidden_layers"): 28,
        ("thinker_config", "text_config", "num_attention_heads"): 28,
        ("thinker_config", "text_config", "num_key_value_heads"): 4,
        ("thinker_config", "text_config", "vocab_size"): 152064,
        ("thinker_config", "vision_config", "depth"): 32,
        ("thinker_config", "vision_config", "hidden_size"): 1280,
        ("thinker_config", "vision_config", "num_heads"): 16,
        ("thinker_config", "vision_config", "out_hidden_size"): 3584,
        ("thinker_config", "vision_config", "patch_size"): 14,
        ("thinker_config", "vision_config", "temporal_patch_size"): 2,
        ("thinker_config", "vision_config", "window_size"): 112,
        ("talker_config", "model_type"): "qwen2_5_omni_talker",
        ("talker_config", "hidden_size"): 896,
        ("talker_config", "num_hidden_layers"): 24,
        ("talker_config", "num_attention_heads"): 12,
        ("talker_config", "num_key_value_heads"): 4,
        ("talker_config", "vocab_size"): 8448,
        ("token2wav_config", "model_type"): "qwen2_5_omni_token2wav",
        ("token2wav_config", "dit_config", "dim"): 1024,
        ("token2wav_config", "dit_config", "depth"): 22,
        ("token2wav_config", "dit_config", "heads"): 16,
        ("token2wav_config", "dit_config", "num_embeds"): 8193,
        ("token2wav_config", "bigvgan_config", "mel_dim"): 80,
        ("token2wav_config", "bigvgan_config", "upsample_initial_channel"): 1536,
        ("token2wav_config", "bigvgan_config", "upsample_rates"): [5, 3, 2, 2, 2, 2],
    }
    for path, expected in exact.items():
        require_path(config, path, expected)
    return {"model_type": "model_type", "thinker": "thinker_config", "talker": "talker_config", "token2wav": "token2wav_config"}


def safe_relative(value: str, label: str) -> None:
    path = Path(value)
    if not value or "\x00" in value or "\\" in value or path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"unsafe {label} path: {value!r}")


def git(root: Path, *args: str) -> str:
    import subprocess
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT).strip()


def server_inventory(root: Path, packet: Path) -> tuple[dict[str, str], list[dict[str, Any]]]:
    envelope = load_json(packet)
    if not isinstance(envelope, dict) or envelope.get("repository") != HF_REPOSITORY or envelope.get("revision") != HF_REVISION or envelope.get("resolved_revision") != HF_REVISION:
        raise RuntimeError("HF server-tree identity mismatch")
    expected = envelope.get("files")
    if not isinstance(expected, list):
        raise RuntimeError("HF server-tree files are not a list")
    actual: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if ".cache" in relative.parts:
            continue
        if path.is_symlink():
            if not path.exists() or not path.is_file():
                raise RuntimeError(f"snapshot dangling/non-file symlink: {path}")
            try:
                path.resolve().relative_to(root.resolve())
            except ValueError as error:
                raise RuntimeError(f"snapshot symlink escapes root: {path}") from error
            actual.append(relative.as_posix())
        elif path.is_file():
            actual.append(relative.as_posix())
        elif not path.is_dir():
            raise RuntimeError(f"snapshot non-regular member: {path}")
    names: set[str] = set()
    records = []
    for item in expected:
        if not isinstance(item, dict) or set(item) != {"path", "type", "size", "git_blob_sha1", "lfs_sha256"}:
            raise RuntimeError("server-tree item must contain path/type/size/git_blob_sha1/lfs_sha256")
        if item["type"] != "file" or not isinstance(item["path"], str) or not isinstance(item["size"], int) or isinstance(item["size"], bool) or item["size"] < 0:
            raise RuntimeError("server-tree item has invalid path/type/size")
        name = item["path"]
        safe_relative(name, "server-tree")
        if ".cache" in Path(name).parts:
            raise RuntimeError(f"server-tree contains excluded cache path: {name}")
        if name in names:
            raise RuntimeError(f"duplicate server-tree path: {name}")
        names.add(name)
        path = root / name
        if not path.is_file() or path.stat().st_size != item["size"]:
            raise RuntimeError(f"server/local path-size mismatch: {name}")
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError as error:
            raise RuntimeError(f"server/local symlink escapes root: {name}") from error
        local_sha = sha256(path)
        lfs = item["lfs_sha256"]
        git_oid = item["git_blob_sha1"]
        if not isinstance(git_oid, str) or len(git_oid) != 40 or any(c not in "0123456789abcdefABCDEF" for c in git_oid):
            raise RuntimeError(f"server Git blob identity missing/invalid: {name}")
        if lfs is not None:
            if not isinstance(lfs, str) or len(lfs) != 64 or any(c not in "0123456789abcdefABCDEF" for c in lfs) or local_sha.lower() != lfs.lower():
                raise RuntimeError(f"server/local LFS SHA mismatch: {name}")
        elif git_blob_sha1(path).lower() != git_oid.lower():
            raise RuntimeError(f"server/local Git blob mismatch: {name}")
        records.append({"path": name, "bytes": path.stat().st_size, "sha256": local_sha, "server_git_blob_sha1": git_oid, "server_lfs_sha256": lfs})
    if set(actual) != names:
        raise RuntimeError("HF server-tree/local path set mismatch")
    return {"repository": envelope["repository"], "revision": envelope["revision"], "resolved_revision": envelope["resolved_revision"]}, records


def inspect_safetensors(path: Path, root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise RuntimeError(f"truncated safetensors header: {path}")
        header_size = int.from_bytes(prefix, "little")
        if header_size <= 0 or header_size > MAX_HEADER_BYTES or header_size > size - 8:
            raise RuntimeError(f"invalid safetensors header length: {path}")
        raw = stream.read(header_size)
    try:
        header = json.loads(raw.decode(), object_pairs_hook=strict_pairs)
    except (UnicodeError, json.JSONDecodeError, RuntimeError) as error:
        raise RuntimeError(f"invalid safetensors header JSON: {path}: {error}") from error
    if not isinstance(header, dict):
        raise RuntimeError("safetensors header is not an object")
    metadata = header.get("__metadata__")
    if metadata is not None and (not isinstance(metadata, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items())):
        raise RuntimeError("safetensors metadata must be a string map")
    payload = size - 8 - header_size
    intervals: list[tuple[int, int, str]] = []
    tensors = []
    for name, record in header.items():
        if name == "__metadata__":
            continue
        safe_relative(name, "tensor")
        if not isinstance(name, str) or not isinstance(record, dict) or set(record) != {"dtype", "shape", "data_offsets"}:
            raise RuntimeError(f"invalid tensor record: {name}")
        dtype, shape, offsets = record["dtype"], record["shape"], record["data_offsets"]
        if dtype not in DTYPE_BYTES or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise RuntimeError(f"invalid tensor dtype/shape/offsets: {name}")
        elements = 1
        for dim in shape:
            if not isinstance(dim, int) or isinstance(dim, bool) or dim < 0:
                raise RuntimeError(f"invalid tensor shape: {name}")
            elements *= dim
        start, end = offsets
        if any(not isinstance(v, int) or isinstance(v, bool) for v in offsets) or start < 0 or end < start or end > payload or end - start != elements * DTYPE_BYTES[dtype]:
            raise RuntimeError(f"invalid tensor data region: {name}")
        intervals.append((start, end, name))
        tensors.append({"name": name, "dtype": dtype, "shape": shape, "elements": elements, "data_bytes": end - start, "shard": path.relative_to(root).as_posix()})
    cursor = 0
    for start, end, name in sorted(intervals):
        if start != cursor:
            raise RuntimeError(f"safetensors overlap/gap before {name}")
        cursor = end
    if cursor != payload:
        raise RuntimeError("safetensors trailing data gap")
    with safe_open(str(path), framework="pt") as handle:
        if set(handle.keys()) != {item["name"] for item in tensors}:
            raise RuntimeError(f"safe_open/header key mismatch: {path}")
    return {"path": path.relative_to(root).as_posix(), "bytes": size, "sha256": sha256(path), "header_bytes": header_size, "data_bytes": payload, "tensor_count": len(tensors)}, tensors


def inventory_weights(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index_path = root / "model.safetensors.index.json"
    if index_path.stat().st_size != INDEX_BYTES:
        raise RuntimeError("safetensors index byte-size mismatch")
    index = load_json(index_path)
    if not isinstance(index, dict) or not isinstance(index.get("metadata"), dict) or index["metadata"].get("total_size") != TOTAL_SIZE or not isinstance(index.get("weight_map"), dict):
        raise RuntimeError("safetensors index metadata/weight_map mismatch")
    mapping = index["weight_map"]
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in mapping.items()):
        raise RuntimeError("weight_map keys/values must be strings")
    for value in mapping.values():
        safe_relative(value, "weight_map")
    if any(value != Path(value).name for value in mapping.values()) or {Path(value).name for value in mapping.values()} != SHARD_NAMES:
        raise RuntimeError("weight_map physical shard set mismatch")
    if {path.name for path in root.glob("*.safetensors")} != SHARD_NAMES:
        raise RuntimeError("physical five-shard set mismatch")
    shards, tensors, seen = [], [], set()
    for path in sorted(root.glob("*.safetensors")):
        shard, rows = inspect_safetensors(path, root)
        expected_bytes, expected_sha = SHARD_BYTES[path.name]
        if shard["bytes"] != expected_bytes or shard["sha256"] != expected_sha or len(rows) != SHARD_TENSOR_COUNTS[path.name]:
            raise RuntimeError(f"authenticated shard mismatch: {path.name}")
        for row in rows:
            if row["name"] in seen or mapping.get(row["name"]) != path.name:
                raise RuntimeError(f"tensor ownership mismatch: {row['name']}")
            seen.add(row["name"])
        shards.append(shard); tensors.extend(rows)
    if seen != set(mapping) or len(tensors) != TENSOR_COUNT:
        raise RuntimeError("tensor coverage/count mismatch")
    return shards, tensors


def inspect_speaker_archive(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len({member.filename for member in members}) != len(members):
            raise RuntimeError("spk_dict.pt duplicate archive member")
        inventory = []
        for member in members:
            safe_relative(member.filename, "torch archive")
            mode = (member.external_attr >> 16) & 0xFFFF
            if mode and stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise RuntimeError(f"spk_dict.pt non-regular member: {member.filename}")
            inventory.append({"path": member.filename, "bytes": member.file_size, "compress_type": member.compress_type})
    import torch
    unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(path)
    if unsafe:
        raise RuntimeError(f"spk_dict.pt unsafe globals: {unsafe}")
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    def walk(value: Any, name: str = "") -> list[dict[str, Any]]:
        if isinstance(value, torch.Tensor):
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"spk_dict.pt non-finite tensor: {name}")
            return [{"name": name, "dtype": str(value.dtype), "shape": list(value.shape), "elements": value.numel()}]
        if isinstance(value, dict):
            if not all(isinstance(k, (str, int, float, bool, type(None))) for k in value):
                raise RuntimeError(f"spk_dict.pt unsupported dict key: {name}")
            rows = []
            for key, child in value.items(): rows.extend(walk(child, f"{name}.{key}" if name else str(key)))
            return rows
        if isinstance(value, (list, tuple)):
            rows = []
            for i, child in enumerate(value): rows.extend(walk(child, f"{name}[{i}]"))
            return rows
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"spk_dict.pt non-finite primitive: {name}")
        if value is None or isinstance(value, (str, int, float, bool)):
            return []
        raise RuntimeError(f"spk_dict.pt unsupported value: {name}")
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path), "archive": inventory, "unsafe_globals": [], "weights_only": True, "tensors": walk(loaded), "execution": "NOT_PERFORMED"}


def source_inventory(source: Path, transformers: Path) -> dict[str, Any]:
    if git(source, "rev-parse", "HEAD") != SOURCE_REVISION or git(transformers, "rev-parse", "HEAD") != TRANSFORMERS_REVISION or git(transformers, "describe", "--exact-match", "--tags") != TRANSFORMERS_TAG:
        raise RuntimeError("source/Transformers detached identity mismatch")
    def remote(root: Path, expected: str) -> str:
        value = git(root, "remote", "get-url", "origin").removesuffix(".git").removesuffix("/")
        if value != expected.removesuffix(".git"):
            raise RuntimeError(f"origin mismatch: {value}")
        return value
    source_origin, transformer_origin = remote(source, SOURCE_REPOSITORY), remote(transformers, TRANSFORMERS_REPOSITORY)
    tracked = [Path(name) for name in git(source, "ls-files", "-z").split("\0") if name and (source / name).is_file()]
    if not any(path.name == "README.md" for path in tracked) or not any("omni" in path.as_posix().lower() for path in tracked):
        raise RuntimeError("official Qwen2.5-Omni source evidence files missing")
    def records(root: Path, paths: list[Path]) -> list[dict[str, Any]]:
        missing = [path.as_posix() for path in paths if not (root / path).is_file()]
        if missing:
            raise RuntimeError(f"required role files missing: {', '.join(missing)}")
        return [{"path": path.as_posix(), "bytes": (root / path).stat().st_size, "sha256": sha256(root / path)} for path in sorted(set(paths))]
    def license_record(root: Path, unknown: str) -> dict[str, Any]:
        path = root / "LICENSE"
        if not path.is_file(): return {"license": "UNKNOWN", "status": unknown, "path": None, "sha256": None}
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        return {"license": "Apache-2.0" if "apache license" in text and "version 2.0" in text else "UNKNOWN", "status": "AUTHENTICATED" if "apache license" in text and "version 2.0" in text else unknown, "path": "LICENSE", "sha256": sha256(path)}
    transformer_roles = [Path(path) for path in TRANSFORMERS_ROLE_FILES]
    return {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION, "origin": source_origin, "license": license_record(source, "SOURCE_LICENSE_UNKNOWN_BLOCKER"), "files": records(source, tracked), "transformers": {"repository": TRANSFORMERS_REPOSITORY, "tag": TRANSFORMERS_TAG, "revision": TRANSFORMERS_REVISION, "origin": transformer_origin, "license": license_record(transformers, "TRANSFORMERS_LICENSE_UNKNOWN_BLOCKER"), "role_files": records(transformers, transformer_roles)}}


def blocked(output: Path, error: Exception, **extra: Any) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = {"format": FORMAT, "status": "BLOCKED", "evidence_stage": "INSPECTION_ONLY", "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "cpu_status": "UNSUPPORTED", "metal_status": "BLOCKED_BY_CPU", "parity_status": "NOT_RUN", "publication": "NO_UPLOAD", "task": "Thinker/Talker multimodal audio-video-vision-text; no native runtime claim", "upstream": {"repository": HF_REPOSITORY, "revision": HF_REVISION}, "official_source": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION}, "transformers": {"repository": TRANSFORMERS_REPOSITORY, "tag": TRANSFORMERS_TAG, "revision": TRANSFORMERS_REVISION}, "error_type": type(error).__name__, "reason": str(error), "blockers": [str(error)], **extra}
    (output / "manifest.json").write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def inspect(snapshot: Path, source: Path, transformers: Path, server_tree: Path, output: Path) -> int:
    identity, files = server_inventory(snapshot, server_tree)
    model_license = {"license": "UNKNOWN"}
    for path in (snapshot / "LICENSE", snapshot / "README.md"):
        if path.is_file() and "apache-2.0" in path.read_text(encoding="utf-8", errors="replace").lower(): model_license = {"license": "Apache-2.0", "path": path.name, "sha256": sha256(path)}; break
    if model_license["license"] != "Apache-2.0": raise RuntimeError("canonical HF Apache-2.0 model license missing")
    json_paths = sorted(path for path in snapshot.rglob("*.json") if path.is_file())
    if not json_paths: raise RuntimeError("HF JSON companions missing")
    parsed = {path.relative_to(snapshot).as_posix(): {"sha256": sha256(path), "json": load_json(path)} for path in json_paths}
    config = parsed.get("config.json", {}).get("json")
    config_paths = validate_config(config)
    shards, tensors = inventory_weights(snapshot)
    speaker = snapshot / "spk_dict.pt"
    if not speaker.is_file() or speaker.stat().st_size != 259_544 or sha256(speaker) != "6a05609b28f5d42b7b748f0f07592545c8f1f6885b9ae8fff64baf56e86b2a18": raise RuntimeError("spk_dict.pt identity mismatch")
    speaker_evidence = inspect_speaker_archive(speaker)
    sources = source_inventory(source, transformers)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in {"snapshot-inventory.json": {"server_tree": identity, "files": files}, "tensor-inventory.json": {"shards": shards, "tensor_count": len(tensors), "tensors": tensors}, "parsed-json.json": parsed, "speaker-inventory.json": speaker_evidence, "source-inventory.json": sources}.items(): (output / name).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    packets = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in output.glob("*-inventory.json")}
    blocked(output, RuntimeError("component/dependency/dataset provenance remains unauthenticated; inspection only"), upstream={"repository": HF_REPOSITORY, "revision": HF_REVISION, "license": model_license, "server_tree": identity, "files": files}, parsed_json=parsed, config_contract=config_paths, shards=shards, tensor_count=len(tensors), speaker=speaker_evidence, official_source=sources, license_evidence={"model": model_license, "official_source": sources["license"], "transformers": sources["transformers"]["license"]}, dataset_provenance={"status": "BLOCKED_UNAUTHENTICATED"}, packets=packets)
    return 2


def self_test() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    assert "weights_only=True" in source and "safe_open" in source and ("onnx" + "runtime") not in source
    assert len(HF_REVISION) == 40 and len(SOURCE_REVISION) == 40 and len(TRANSFORMERS_REVISION) == 40
    assert TRANSFORMERS_ROLE_FILES[:4] == (
        "src/transformers/models/qwen2_5_omni/configuration_qwen2_5_omni.py",
        "src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py",
        "src/transformers/models/qwen2_5_omni/modular_qwen2_5_omni.py",
        "src/transformers/models/qwen2_5_omni/processing_qwen2_5_omni.py",
    )
    config = {
        "model_type": "qwen2_5_omni", "enable_audio_output": True,
        "enable_talker": True, "transformers_version": "4.50.0.dev0",
        "thinker_config": {
            "model_type": "qwen2_5_omni_thinker",
            "audio_config": {"d_model": 1280, "encoder_layers": 32, "encoder_attention_heads": 20, "num_mel_bins": 128, "output_dim": 3584},
            "text_config": {"hidden_size": 3584, "num_hidden_layers": 28, "num_attention_heads": 28, "num_key_value_heads": 4, "vocab_size": 152064},
            "vision_config": {"depth": 32, "hidden_size": 1280, "num_heads": 16, "out_hidden_size": 3584, "patch_size": 14, "temporal_patch_size": 2, "window_size": 112},
        },
        "talker_config": {"model_type": "qwen2_5_omni_talker", "hidden_size": 896, "num_hidden_layers": 24, "num_attention_heads": 12, "num_key_value_heads": 4, "vocab_size": 8448},
        "token2wav_config": {"model_type": "qwen2_5_omni_token2wav", "dit_config": {"dim": 1024, "depth": 22, "heads": 16, "num_embeds": 8193}, "bigvgan_config": {"mel_dim": 80, "upsample_initial_channel": 1536, "upsample_rates": [5, 3, 2, 2, 2, 2]}},
    }
    assert validate_config(config)["thinker"] == "thinker_config"
    wrong_nested = json.loads(json.dumps(config)); wrong_nested["audio_config"] = wrong_nested["thinker_config"]["audio_config"]
    try: validate_config(wrong_nested)
    except RuntimeError: pass
    else: raise AssertionError("misplaced root audio_config accepted")
    for path, old_key in ((("thinker_config", "audio_config"), "audio_output_dim"), (("token2wav_config", "dit_config"), "num_heads"), (("token2wav_config", "bigvgan_config"), "num_mels")):
        wrong_key = json.loads(json.dumps(config)); section = wrong_key
        for key in path: section = section[key]
        section[old_key] = section.pop({"audio_output_dim": "output_dim", "num_heads": "heads", "num_mels": "mel_dim"}[old_key])
        try: validate_config(wrong_key)
        except RuntimeError: pass
        else: raise AssertionError(f"obsolete config key accepted: {old_key}")
    wrong_video = json.loads(json.dumps(config)); wrong_video["video_config"] = {}
    try: validate_config(wrong_video)
    except RuntimeError: pass
    else: raise AssertionError("forbidden video_config accepted")
    try: json.loads('{"x":1,"x":2}', object_pairs_hook=strict_pairs)
    except RuntimeError: pass
    else: raise AssertionError("duplicate JSON accepted")
    with tempfile.TemporaryDirectory(prefix="vokra-qwen2-omni-") as directory:
        root = Path(directory); huge = root / "huge.safetensors"; huge.write_bytes((MAX_HEADER_BYTES + 1).to_bytes(8, "little"))
        try: inspect_safetensors(huge, root)
        except RuntimeError: pass
        else: raise AssertionError("huge header accepted")
        try: safe_relative("../escape", "weight_map")
        except RuntimeError: pass
        else: raise AssertionError("path traversal accepted")
        unsafe_tensor = root / "unsafe.safetensors"; unsafe_name = json.dumps({"../tensor": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}).encode(); unsafe_tensor.write_bytes(len(unsafe_name).to_bytes(8, "little") + unsafe_name + b"x")
        try: inspect_safetensors(unsafe_tensor, root)
        except RuntimeError: pass
        else: raise AssertionError("unsafe tensor name accepted")
        speaker = root / "spk_dict.pt"
        import torch
        torch.save({"x": torch.ones(2)}, speaker)
        assert inspect_speaker_archive(speaker)["execution"] == "NOT_PERFORMED"
        snapshot = root / "snapshot"; snapshot.mkdir(); content = snapshot / "x"; content.write_bytes(b"abc"); cache_file = snapshot / ".cache" / "internal"; cache_file.parent.mkdir(); cache_file.write_bytes(b"ignored")
        packet = root / "tree.json"; matching = {"repository": HF_REPOSITORY, "revision": HF_REVISION, "resolved_revision": HF_REVISION, "files": [{"path": "x", "type": "file", "size": 3, "git_blob_sha1": git_blob_sha1(content), "lfs_sha256": None}]}; packet.write_text(json.dumps(matching), encoding="utf-8")
        server_inventory(snapshot, packet)
        content.write_bytes(b"abd")
        try: server_inventory(snapshot, packet)
        except RuntimeError: pass
        else: raise AssertionError("same-size content mutation accepted")
        content.write_bytes(b"abc")
        for rows in ([*matching["files"], {"path": "extra", "type": "file", "size": 0, "git_blob_sha1": "0" * 40, "lfs_sha256": None}], []):
            mutated = dict(matching); mutated["files"] = rows; packet.write_text(json.dumps(mutated), encoding="utf-8")
            try: server_inventory(snapshot, packet)
            except RuntimeError: pass
            else: raise AssertionError("extra/missing server path accepted")
        lfs_packet = dict(matching); lfs_packet["files"] = [{"path": "x", "type": "file", "size": 3, "git_blob_sha1": "1" * 40, "lfs_sha256": sha256(content)}]; packet.write_text(json.dumps(lfs_packet), encoding="utf-8"); server_inventory(snapshot, packet)
        packet.write_text(json.dumps({"repository": "wrong", "revision": HF_REVISION, "resolved_revision": HF_REVISION, "files": []}), encoding="utf-8")
        try: server_inventory(root, packet)
        except RuntimeError: pass
        else: raise AssertionError("server identity mismatch accepted")
    print("qwen2_5_omni_7b_inspect.py self-test: OK (strict tree/header/JSON/PT fail-closed contracts)")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--snapshot", type=Path); parser.add_argument("--source", type=Path); parser.add_argument("--transformers", type=Path); parser.add_argument("--server-tree", type=Path); parser.add_argument("--output", type=Path); parser.add_argument("--self-test", action="store_true"); args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.snapshot, args.source, args.transformers, args.server_tree, args.output)): parser.error("--self-test accepts no paths")
        self_test(); return 0
    if any(value is None for value in (args.snapshot, args.source, args.transformers, args.server_tree, args.output)): parser.error("all inspection paths are required")
    try: return inspect(args.snapshot, args.source, args.transformers, args.server_tree, args.output)
    except Exception as error:
        blocked(args.output, error); print(f"Qwen2.5-Omni inspection BLOCKED: {error}", file=sys.stderr); return 2


if __name__ == "__main__": raise SystemExit(main())
