#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Streaming, inspection-only oracle for Microsoft's VibeVoice-ASR 9B.

The HF release is eight large safetensors shards.  This tool never merges or
loads the checkpoint resident: it validates the index, opens one shard at a
time with ``safetensors.safe_open``, and records a canonical tensor manifest.
It does not execute the Transformers model or invent ASR labels/timestamps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import stat
import tempfile
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

UPSTREAM_REPOSITORY = "microsoft/VibeVoice-ASR"
UPSTREAM_REVISION = "d0c9efdb8d614685062c04425d91e01b6f37d944"
SOURCE_REPOSITORY = "https://github.com/microsoft/VibeVoice"
SOURCE_REVISION = "94da20d98b2fa7688e9cbfaf7692ddb4954f7600"
TRANSFORMERS_REPOSITORY = "https://github.com/huggingface/transformers"
TRANSFORMERS_TAG = "v4.51.3"
TRANSFORMERS_REVISION = "5f4ecf2d9f867a1255131d2461d75793c0cf1db2"
FORMAT = "vokra-vibevoice-asr-inspection-v1"
SHARD_COUNT = 8
SOURCE_ROLE_BLOBS: dict[str, str | None] = {
    "LICENSE": "269a8973689dbb250d355f516f8a30c1cc66b8e4",
    "pyproject.toml": "ea0f8370ee781f4552d9ab746cb920033d58bafc",
    "demo/vibevoice_asr_inference_from_file.py": "bf4f75df74c7299c6596bb09d77e8c67400ff1ac",
    "demo/vibevoice_asr_gradio_demo.py": "3e54d12d9249424aca985853417ed769fd5a709f",
    "docs/vibevoice-asr.md": "5e6594486f2d708fafbae13946d85d94d97fb30b",
    "vibevoice/modular/configuration_vibevoice.py": "18451136e5c650345e89f4df052731c7322faab9",
    "vibevoice/modular/modeling_vibevoice_asr.py": "4663d3f494b9d4343d99ce2becf77fdb4b1d6007",
    "vibevoice/modular/modular_vibevoice_tokenizer.py": "454f9c13094ae42b186ed49e22227cea18189ee1",
    "vibevoice/modular/modular_vibevoice_text_tokenizer.py": "9532d9ffe7120eb47b18c52c0a23db9e2d4e3bbf",
    "vibevoice/processor/audio_utils.py": "3f9d112cd4fe7dbd84703776cfbdfbc4ee5cce0d",
    "vibevoice/processor/vibevoice_asr_processor.py": "cacb116dd7f43ee5241c46c164dff3e3740a1f2c",
    "vibevoice/processor/vibevoice_tokenizer_processor.py": "67f61a62f7bd43df46ebc1d8d533fe0dd01adc02",
}
TRANSFORMERS_ROLE_BLOBS: dict[str, str | None] = {
    "LICENSE": "68b7d66c97d66c58de883ed0c451af2b3183e6f3",
    "src/transformers/audio_utils.py": "8420a84e089e03be9a80fb63c237e34203ea28a0",
    "src/transformers/feature_extraction_sequence_utils.py": "c9a26bac9b3dcd9bb14d855f34494a38df3f7f71",
    "src/transformers/generation/utils.py": "95d211cd5e31d79d4c926f78452fb5662f5125cf",
    "src/transformers/models/qwen2/configuration_qwen2.py": "2e82f1976f3922f3620415f4eace6c6e046243f8",
    "src/transformers/models/qwen2/modeling_qwen2.py": "16a7316e2d0e56eafe301a7f2d8693d6cc6c73ec",
    "src/transformers/processing_utils.py": "b1c40e7ff2d7c08e8b8e741a59f933f58c13fb30",
}
EXPECTED_INDEX_METADATA = {"total_parameters": 8_674_021_857, "total_size": 17_348_198_410}
HF_CACHE = Path(".cache/huggingface")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
TRANSPORT_CACHE_PATH = Path(".cache/huggingface")
TRANSPORT_CACHE_SCOPE = "snapshot_root_exact_transport_subtree"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1_bytes(data: bytes) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {len(data)}\0".encode())
    digest.update(data)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    with path.open("rb") as stream:
        data = stream.read()
    return git_blob_sha1_bytes(data)


def lfs_pointer_sha1(payload_sha256: str, payload_bytes: int) -> str:
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{payload_sha256}\nsize {payload_bytes}\n".encode()
    return git_blob_sha1_bytes(pointer)


def strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def source_inventory(
    source: Path,
    transformers_source: Path,
    *,
    source_revision: str = SOURCE_REVISION,
    source_roles: dict[str, str | None] = SOURCE_ROLE_BLOBS,
    transformers_revision: str = TRANSFORMERS_REVISION,
    transformers_roles: dict[str, str | None] = TRANSFORMERS_ROLE_BLOBS,
    transformers_tag: str = TRANSFORMERS_TAG,
) -> dict[str, Any]:
    def checkout(root: Path, repository: str, revision: str, roles: dict[str, str | None], tag: str | None, license_name: str, constraint: str | None) -> dict[str, Any]:
        blockers: list[str] = []
        if git(root, "status", "--porcelain", "--untracked-files=all"):
            blockers.append("checkout is dirty")
        actual = git(root, "rev-parse", "HEAD")
        if actual != revision:
            blockers.append("revision mismatch")
        origin = git(root, "remote", "get-url", "origin").rstrip("/").removesuffix(".git")
        if origin.lower() != repository.lower().removesuffix(".git"):
            blockers.append("origin mismatch")
        if tag is not None:
            try:
                actual_tag = git(root, "describe", "--exact-match", "--tags", "HEAD")
            except subprocess.CalledProcessError:
                actual_tag = ""
            if actual_tag != tag:
                blockers.append("tag mismatch")
        if constraint is not None and constraint not in (root / "pyproject.toml").read_text(encoding="utf-8", errors="replace"):
            blockers.append("pyproject constraint mismatch")
        raw = subprocess.check_output(["git", "-C", str(root), "ls-files", "-s", "-z"])
        entries: dict[str, dict[str, Any]] = {}
        for item in raw.split(b"\0"):
            if not item:
                continue
            header, encoded = item.split(b"\t", 1)
            fields = header.split(); name = encoded.decode()
            if len(fields) != 3 or name in entries:
                blockers.append(f"tracked index schema/duplicate: {name}")
                continue
            mode, index_object, stage = fields[0].decode(), fields[1].decode(), fields[2].decode()
            path = root / name
            if stage != "0" or mode not in {"100644", "100755"} or path.is_symlink() or not path.is_file():
                blockers.append(f"tracked entry is not regular stage-0: {name}")
                continue
            expected_mode = 0o755 if mode == "100755" else 0o644
            if stat.S_IMODE(path.stat().st_mode) != expected_mode:
                blockers.append(f"filesystem mode mismatch: {name}")
            head_object = git(root, "rev-parse", f"HEAD:{name}")
            working_object = git_blob_sha1(path)
            row = {"path": name, "mode": mode, "stage": stage, "index_object_sha1": index_object, "head_object_sha1": head_object, "working_blob_sha1": working_object, "bytes": path.stat().st_size, "sha256": sha256(path)}
            entries[name] = row
            if not (index_object == head_object == working_object):
                blockers.append(f"tracked object mismatch: {name}")
        role_files: list[dict[str, Any]] = []
        for name, expected in roles.items():
            row = entries.get(name)
            if row is None:
                blockers.append(f"missing/untracked fixed role: {name}")
                continue
            role = {**row, "expected_git_blob_sha1": expected}
            role_files.append(role)
            if expected is None:
                blockers.append(f"fixed role blob unavailable for authenticated review: {name}")
            elif role["mode"] != "100644" or not (role["index_object_sha1"] == role["head_object_sha1"] == role["working_blob_sha1"] == expected):
                blockers.append(f"fixed role object/mode mismatch: {name}")
        if license_name == "MIT":
            for demo_name in ("demo/vibevoice_asr_inference_from_file.py", "demo/vibevoice_asr_gradio_demo.py"):
                demo_path = root / demo_name
                demo_text = demo_path.read_text(encoding="utf-8", errors="replace") if demo_path.is_file() and not demo_path.is_symlink() else ""
                if "VibeVoiceASRProcessor" not in demo_text or "Qwen/Qwen2.5-7B" not in demo_text:
                    blockers.append(f"official ASR demo boundary marker missing: {demo_name}")
        connector_path = "vibevoice/modular/modeling_vibevoice_asr.py"
        connector_markers = {
            "acoustic": r"self\.acoustic_connector\s*=\s*SpeechConnector\s*\(\s*config\.acoustic_vae_dim\b",
            "semantic": r"self\.semantic_connector\s*=\s*SpeechConnector\s*\(\s*config\.semantic_vae_dim\b",
        }
        connector_file = root / connector_path
        connector_role = entries.get(connector_path)
        expected_connector = roles.get(connector_path)
        connector_role_authenticated = connector_role is not None and connector_role["mode"] == "100644" and expected_connector is not None and connector_role["index_object_sha1"] == connector_role["head_object_sha1"] == connector_role["working_blob_sha1"] == expected_connector
        connector_text = connector_file.read_text(encoding="utf-8", errors="replace") if connector_file.is_file() and not connector_file.is_symlink() else ""
        connector_evidence = {
            "path": connector_path,
            "role_blob_authenticated": connector_role_authenticated,
            "markers": {name: re.search(pattern, connector_text) is not None for name, pattern in connector_markers.items()},
            "status": "AUTHENTICATED" if connector_role_authenticated and all(re.search(pattern, connector_text) for pattern in connector_markers.values()) else "BLOCKED",
        }
        if license_name == "MIT" and connector_evidence["status"] != "AUTHENTICATED":
            blockers.append("official ASR connector topology marker/blob is not authenticated")
        license_path = root / "LICENSE"
        license_text = license_path.read_text(encoding="utf-8", errors="replace").lower() if license_path.is_file() and not license_path.is_symlink() else ""
        markers = (("permission is hereby granted, free of charge", "the software is provided \"as is\"", "without warranty") if license_name == "MIT" else ("apache license, version 2.0", "you may obtain a copy of the license", "distributed under the license", "without warranties or conditions"))
        license_ok = all(marker in license_text for marker in markers) and "LICENSE" in entries
        if not license_ok:
            blockers.append(f"{license_name} LICENSE clauses/blob unavailable")
        return {"repository": repository, "revision": revision, "resolved_revision": actual, "origin": origin, "tag": tag, "clean": not any("dirty" in item for item in blockers), "tracked_files": sorted(entries.values(), key=lambda row: row["path"]), "role_files": sorted(role_files, key=lambda row: row["path"]), "connector_topology": connector_evidence, "license": {"path": "LICENSE", "license": license_name, "authenticated": license_ok, "bytes": license_path.stat().st_size if license_path.is_file() else None, "sha256": sha256(license_path) if license_path.is_file() else None, "markers": {marker: marker in license_text for marker in markers}}, "constraint": constraint, "blockers": blockers, "status": "AUTHENTICATED" if not blockers else "BLOCKED"}
    return {"source": checkout(source, SOURCE_REPOSITORY, source_revision, source_roles, None, "MIT", "transformers>=4.51.3,<5.0.0"), "transformers": checkout(transformers_source, TRANSFORMERS_REPOSITORY, transformers_revision, transformers_roles, transformers_tag, "Apache-2.0", None)}


def companion_inventory(snapshot: Path) -> list[dict[str, Any]]:
    if not (snapshot / "config.json").is_file():
        raise RuntimeError("HF snapshot lacks config.json")
    shard_names = {f"model-{n:05d}-of-{SHARD_COUNT:05d}.safetensors" for n in range(1, SHARD_COUNT + 1)}
    unexpected = [path.name for path in snapshot.iterdir() if path.name not in shard_names and ("tokenizer" in path.name.lower() or "processor" in path.name.lower() or "generation" in path.name.lower())]
    if unexpected:
        raise RuntimeError(f"unexpected tokenizer/processor/generation companion in canonical model tree: {sorted(unexpected)}")
    return [{"status": "ABSENT_AS_EXPECTED", "files": [], "processor": "NOT_IN_MODEL_SNAPSHOT", "tokenizer": "NOT_IN_MODEL_SNAPSHOT", "generation": "NOT_IN_MODEL_SNAPSHOT"}]


def parsed_json_companions(snapshot: Path, companions: list[dict[str, Any]]) -> dict[str, Any]:
    del companions
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"), object_pairs_hook=strict_pairs)
    if not isinstance(config, dict):
        raise RuntimeError("official config root must be an object")
    sections = {
        "acoustic_tokenizer_config": {
            "model_type": "vibevoice_acoustic_tokenizer",
            "vae_dim": 64,
            "dtype": "bfloat16",
        },
        "semantic_tokenizer_config": {
            "model_type": "vibevoice_semantic_tokenizer",
            "vae_dim": 128,
            "dtype": "bfloat16",
        },
        "decoder_config": {
            "model_type": "qwen2",
            "hidden_size": 3584,
            "num_hidden_layers": 28,
            "dtype": "bfloat16",
        },
        "diffusion_head_config": {
            "model_type": "vibepod_diffusion_head",
            "hidden_size": 3584,
            "latent_size": 64,
            "speech_vae_dim": 64,
        },
    }
    expected = {
        "model_type": "vibevoice",
        "architectures": ["VibeVoiceForASRTraining"],
        "dtype": "float32",
        "acoustic_vae_dim": 64,
        "semantic_vae_dim": 128,
    }
    for key, expected_value in expected.items():
        actual = config.get(key)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise RuntimeError(f"official config exact contract mismatch: {key}")
    for section, required in sections.items():
        if not isinstance(config.get(section), dict):
            raise RuntimeError(f"official config lacks structural section: {section}")
        for key, expected_value in required.items():
            actual = config[section].get(key)
            if type(actual) is not type(expected_value) or actual != expected_value:
                raise RuntimeError(f"official config exact contract mismatch: {section}.{key}")
    return {"config.json": config}


def transport_cache_scope(snapshot: Path) -> tuple[Path | None, dict[str, Any]]:
    """Validate and describe the one non-identity snapshot transport subtree."""

    evidence: dict[str, Any] = {
        "path": TRANSPORT_CACHE_PATH.as_posix(),
        "scope": TRANSPORT_CACHE_SCOPE,
        "present": False,
        "identity_role": "NON_IDENTITY_TRANSPORT_METADATA",
        "status": "ABSENT",
    }
    cache = snapshot / ".cache"
    if cache.is_symlink() or (cache.exists() and not cache.is_dir()):
        raise RuntimeError(f"HF snapshot root .cache must be a real directory: {cache}")
    if not cache.exists():
        return None, evidence
    transport = cache / "huggingface"
    if transport.is_symlink() or (transport.exists() and not transport.is_dir()):
        raise RuntimeError(
            f"HF snapshot root .cache/huggingface must be a real directory: {transport}"
        )
    if not transport.exists():
        return None, evidence
    evidence["present"] = True
    evidence["status"] = "EXCLUDED"
    try:
        evidence["entry_count"] = sum(1 for _ in transport.rglob("*"))
    except OSError as error:
        raise RuntimeError(f"HF transport cache inventory failed: {transport}: {error}") from error
    return transport, evidence


def server_inventory(snapshot: Path, packet: Path) -> dict[str, Any]:
    envelope = json.loads(packet.read_text(encoding="utf-8"), object_pairs_hook=strict_pairs)
    if not isinstance(envelope, dict) or set(envelope) != {"repository", "requested_revision", "resolved_revision", "walk", "files"}:
        raise RuntimeError("HF server packet schema mismatch")
    if envelope["repository"] != UPSTREAM_REPOSITORY or envelope["requested_revision"] != UPSTREAM_REVISION or envelope["resolved_revision"] != UPSTREAM_REVISION or envelope["walk"] != "recursive_file_only":
        raise RuntimeError("HF server packet identity mismatch")
    rows = envelope["files"]
    if not isinstance(rows, list):
        raise RuntimeError("HF server packet files must be a list")
    expected: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "type", "size", "git_blob_sha1", "lfs_pointer_git_blob_sha1", "lfs_sha256"}:
            raise RuntimeError("HF server packet row schema mismatch")
        name, size = row["path"], row["size"]
        if not isinstance(name, str) or not name or "\x00" in name or "\\" in name or name.startswith("/") or any(part in {"", ".", ".."} for part in Path(name).parts) or row["type"] != "file" or not isinstance(size, int) or isinstance(size, bool) or size < 0 or name in expected:
            raise RuntimeError(f"HF server packet unsafe/duplicate row: {name!r}")
        regular, pointer, payload = row["git_blob_sha1"], row["lfs_pointer_git_blob_sha1"], row["lfs_sha256"]
        if regular is not None and (not isinstance(regular, str) or not HEX40.fullmatch(regular)):
            raise RuntimeError(f"HF regular Git identity mismatch: {name}")
        if pointer is not None and (not isinstance(pointer, str) or not HEX40.fullmatch(pointer)):
            raise RuntimeError(f"HF LFS pointer identity mismatch: {name}")
        if payload is not None and (not isinstance(payload, str) or not HEX64.fullmatch(payload)):
            raise RuntimeError(f"HF LFS payload identity mismatch: {name}")
        if (payload is None and (regular is None or pointer is not None)) or (payload is not None and (regular is not None or pointer is None)):
            raise RuntimeError(f"HF regular/LFS fields are inconsistent: {name}")
        if payload is not None and lfs_pointer_sha1(payload, size) != pointer:
            raise RuntimeError(f"HF canonical LFS pointer mismatch: {name}")
        expected[name] = row
    actual: set[str] = set()
    excluded_transport, transport_evidence = transport_cache_scope(snapshot)
    for path in snapshot.rglob("*"):
        relative = path.relative_to(snapshot)
        parts = relative.parts
        if relative == Path(".cache"):
            continue
        if excluded_transport is not None:
            try:
                relative.relative_to(excluded_transport.relative_to(snapshot))
            except ValueError:
                pass
            else:
                continue
        if ".cache" in parts:
            raise RuntimeError(f"unexpected cache path: {path}")
        if path.is_symlink():
            raise RuntimeError(f"HF snapshot payload symlink: {path}")
        if path.is_file():
            actual.add(relative.as_posix())
        elif not path.is_dir():
            raise RuntimeError(f"HF snapshot non-regular member: {path}")
    if actual != set(expected):
        raise RuntimeError(f"HF server/local file set mismatch: missing={sorted(set(expected)-actual)} extra={sorted(actual-set(expected))}")
    records = []
    for name in sorted(expected):
        row, path = expected[name], snapshot / name
        if path.stat().st_size != row["size"]:
            raise RuntimeError(f"HF local size mismatch: {name}")
        digest = sha256(path)
        if row["lfs_sha256"] is None:
            if git_blob_sha1(path) != row["git_blob_sha1"]:
                raise RuntimeError(f"HF local Git blob mismatch: {name}")
        elif digest != row["lfs_sha256"]:
            raise RuntimeError(f"HF local LFS payload mismatch: {name}")
        records.append({**row, "payload_sha256": digest, "payload_bytes": path.stat().st_size})
    return {"repository": envelope["repository"], "requested_revision": envelope["requested_revision"], "resolved_revision": envelope["resolved_revision"], "walk": envelope["walk"], "files": records, "transport_cache": transport_evidence}


def safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise RuntimeError(f"truncated safetensors header: {path.name}")
        header_bytes = int.from_bytes(prefix, "little")
        if header_bytes <= 0 or header_bytes > 64 * 1024 * 1024:
            raise RuntimeError(f"safetensors header exceeds bound: {path.name}")
        raw = stream.read(header_bytes)
        if len(raw) != header_bytes:
            raise RuntimeError(f"truncated safetensors header body: {path.name}")
    header = json.loads(raw.decode("utf-8"), object_pairs_hook=strict_pairs)
    if not isinstance(header, dict):
        raise RuntimeError(f"safetensors header root is not an object: {path.name}")
    payload_bytes = path.stat().st_size - 8 - header_bytes
    if payload_bytes < 0:
        raise RuntimeError(f"safetensors has no payload: {path.name}")
    dtype_bytes = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "I16": 2, "I32": 4, "I64": 8, "U8": 1, "BOOL": 1}
    ranges: list[tuple[int, int, str]] = []
    descriptors: dict[str, Any] = {}
    for name, descriptor in header.items():
        if name == "__metadata__":
            if not isinstance(descriptor, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in descriptor.items()):
                raise RuntimeError(f"invalid safetensors metadata: {path.name}")
            continue
        if not isinstance(name, str) or not name or "\x00" in name or "\\" in name or not isinstance(descriptor, dict) or set(descriptor) != {"dtype", "shape", "data_offsets"}:
            raise RuntimeError(f"invalid safetensors descriptor: {name!r}")
        dtype, shape, offsets = descriptor["dtype"], descriptor["shape"], descriptor["data_offsets"]
        if dtype not in dtype_bytes or not isinstance(shape, list) or not shape or any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in shape) or not isinstance(offsets, list) or len(offsets) != 2 or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in offsets):
            raise RuntimeError(f"invalid safetensors shape/dtype/offsets: {name}")
        begin, end = offsets; elements = math.prod(shape)
        if end < begin or end > payload_bytes or end - begin != elements * dtype_bytes[dtype]:
            raise RuntimeError(f"safetensors offset/byte-count mismatch: {name}")
        ranges.append((begin, end, name)); descriptors[name] = {"dtype": dtype, "shape": shape, "data_offsets": offsets}
    cursor = 0
    for begin, end, name in sorted(ranges):
        if begin != cursor:
            raise RuntimeError(f"safetensors offset gap/overlap: {name}")
        cursor = end
    if cursor != payload_bytes or not descriptors:
        raise RuntimeError(f"safetensors payload coverage mismatch: {path.name}")
    return descriptors


def inventory_shards(snapshot: Path, index: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if set(index) != {"metadata", "weight_map"}:
        raise RuntimeError("safetensors index schema must be exactly metadata+weight_map")
    metadata = index["metadata"]
    if not isinstance(metadata, dict) or set(metadata) != set(EXPECTED_INDEX_METADATA) or metadata != EXPECTED_INDEX_METADATA:
        raise RuntimeError(f"safetensors index metadata mismatch: {metadata!r}")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("safetensors index lacks a non-empty weight_map")
    if any(not isinstance(name, str) or not name or not isinstance(shard, str) for name, shard in weight_map.items()):
        raise RuntimeError("safetensors index weight_map has invalid names")
    expected_shards = {f"model-{n:05d}-of-{SHARD_COUNT:05d}.safetensors" for n in range(1, SHARD_COUNT + 1)}
    indexed_shards = set(weight_map.values())
    if indexed_shards != expected_shards:
        raise RuntimeError(f"index shard coverage mismatch: {sorted(indexed_shards)}")
    actual_shards = {path.name for path in snapshot.glob("*.safetensors")}
    if actual_shards != expected_shards:
        raise RuntimeError(f"snapshot shard set mismatch: {sorted(actual_shards)}")
    shard_records: list[dict[str, Any]] = []
    tensors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for shard_name in sorted(expected_shards):
        shard = snapshot / shard_name
        if not shard.is_file():
            raise RuntimeError(f"missing indexed shard: {shard_name}")
        descriptors = safetensors_header(shard)
        local_names: set[str] = set()
        with safe_open(str(shard), framework="pt") as handle:
            for name in sorted(handle.keys()):
                if not name or "\x00" in name or "\\" in name or any(part in {"", ".", ".."} for part in Path(name).parts):
                    raise RuntimeError(f"unsafe tensor name: {name!r}")
                if name in seen or name in local_names:
                    raise RuntimeError(f"duplicate tensor name across shards: {name}")
                local_names.add(name)
                tensor = handle.get_tensor(name)
                shape = [int(dim) for dim in tensor.shape]
                if any(dim <= 0 for dim in shape) or not bool(torch.isfinite(tensor).all().item()):
                    raise RuntimeError(f"tensor shape/finite contract mismatch: {name}")
                descriptor = descriptors.get(name)
                dtype_names = {"BF16": "bfloat16", "F16": "float16", "F32": "float32", "F64": "float64", "I8": "int8", "I16": "int16", "I32": "int32", "I64": "int64", "U8": "uint8", "BOOL": "bool"}
                if descriptor is None or descriptor["shape"] != shape or dtype_names[descriptor["dtype"]] != str(tensor.dtype).removeprefix("torch.").lower():
                    raise RuntimeError(f"safe_open/header tensor metadata mismatch: {name}")
                tensors.append({"name": name, "shard": shard_name, "shape": shape, "dtype": str(tensor.dtype).removeprefix("torch."), "elements": int(tensor.numel()), "finite": True})
                # Keep the resident scope to one materialized tensor.  The
                # shard itself remains mmap-backed inside safe_open and is
                # never merged into a checkpoint-sized object.
                del tensor
        seen.update(local_names)
        if local_names != set(descriptors):
            raise RuntimeError(f"safe_open/header name coverage mismatch: {shard_name}")
        shard_records.append({"path": shard_name, "bytes": shard.stat().st_size, "sha256": sha256(shard), "tensor_count": len(local_names)})
    if seen != set(weight_map):
        raise RuntimeError("index and shard tensor-name coverage mismatch")
    if any(weight_map[record["name"]] != record["shard"] for record in tensors):
        raise RuntimeError("index tensor-to-shard mapping mismatch")
    return shard_records, tensors


def weight_license(snapshot: Path, companions: list[dict[str, Any]]) -> dict[str, Any]:
    readme = snapshot / "README.md"
    if not readme.is_file() or readme.is_symlink():
        raise RuntimeError("HF weight README is missing")
    lines = readme.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines or lines[0].strip() != "---":
        raise RuntimeError("HF model-card frontmatter is missing")
    try:
        end = next(index for index in range(1, len(lines)) if lines[index].strip() == "---")
    except StopIteration as error:
        raise RuntimeError("HF model-card frontmatter is unterminated") from error
    seen: set[str] = set(); license_value: str | None = None
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#") or line[0].isspace() or line.startswith("-"):
            continue
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_-]*):[ \t]*(.*)", line)
        if match is None:
            raise RuntimeError("HF model-card frontmatter has malformed top-level YAML")
        key, value = match.groups()
        if key in seen:
            raise RuntimeError(f"HF model-card frontmatter key duplicated: {key}")
        seen.add(key)
        if key == "license":
            license_value = value.strip().strip("\"'")
    if license_value != "mit":
        raise RuntimeError("HF weight license must be exactly top-level license: mit")
    return {"license": "MIT", "source": "README.md", "bytes": readme.stat().st_size, "sha256": sha256(readme), "frontmatter": {"license": "mit"}}


def inspect(snapshot: Path, source: Path, transformers_source: Path, server_tree: Path, output: Path) -> int:
    snapshot = snapshot.resolve()
    source = source.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise RuntimeError("inspection output must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    hf_identity = server_inventory(snapshot, server_tree)
    index_path = snapshot / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError("HF snapshot lacks model.safetensors.index.json")
    index = json.loads(index_path.read_text(encoding="utf-8"), object_pairs_hook=strict_pairs)
    shard_records, tensors = inventory_shards(snapshot, index)
    companions = companion_inventory(snapshot)
    parsed_companions = parsed_json_companions(snapshot, companions)
    upstream_license = weight_license(snapshot, companions)
    sources = source_inventory(source, transformers_source)
    if sources["source"]["status"] != "AUTHENTICATED" or sources["transformers"]["status"] != "AUTHENTICATED":
        raise RuntimeError("official source/Transformers authentication is incomplete")
    (output / "tensor-inventory.json").write_text(json.dumps({"tensor_count": len(tensors), "tensors": sorted(tensors, key=lambda value: value["name"])}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "shard-inventory.json").write_text(json.dumps({"shard_count": SHARD_COUNT, "shards": shard_records, "index_sha256": sha256(index_path)}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "companion-inventory.json").write_text(json.dumps({"files": companions}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "source-inventory.json").write_text(json.dumps(sources["source"], sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "parsed-companions.json").write_text(json.dumps(parsed_companions, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "transformers-inventory.json").write_text(json.dumps(sources["transformers"], sort_keys=True, indent=2) + "\n", encoding="utf-8")
    packets = {name: {"bytes": (output / name).stat().st_size, "sha256": sha256(output / name)} for name in ("tensor-inventory.json", "shard-inventory.json", "companion-inventory.json", "source-inventory.json", "parsed-companions.json", "transformers-inventory.json")}
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
        "runtime_parity": "NOT_RUN",
        "numerical_parity": "NOT_RUN",
        "upstream": {"repository": UPSTREAM_REPOSITORY, "revision": UPSTREAM_REVISION, "license": upstream_license, "shard_count": SHARD_COUNT, "index": "model.safetensors.index.json"},
        "hf_server_tree": hf_identity,
        "official_source": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION, "license": sources["source"]["license"], "connector_topology": sources["source"]["connector_topology"], "transformers": sources["transformers"]},
        "tensor_count": len(tensors),
        "shards": shard_records,
        "companions": companions,
        "parsed_companions": parsed_companions,
        "resident_scope": "one tensor at a time via safetensors.safe_open(framework='pt').get_tensor(); shard bytes are hashed streaming",
        "config_topology_contract": "model_type/architectures/dtype and nested tokenizer/connector/decoder fields extracted from config.json; behavioral diarization/timestamp/structured-output markers are validated in official source",
        "config_contract": "acoustic+semantic tokenizers, speech connectors, Qwen decoder, diarization/timestamp/output markers extracted from official config/source",
        "source_license": sources["source"]["license"],
        "external_dependency": {"repository": "Qwen/Qwen2.5-7B", "revision": "UNSELECTED_BLOCKER", "selection_status": "BLOCKED", "files": "NOT_DOWNLOADED", "model_weights": "NOT_DOWNLOADED"},
        "packets": packets,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 2


def write_blocked(output: Path, error: Exception) -> None:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise RuntimeError("blocked evidence output must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"format": FORMAT, "status": "BLOCKED", "evidence_stage": "INSPECTION_ONLY", "inspection_status": "INSPECTION_ERROR", "collection_status": "UNVERIFIED", "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "cpu_status": "UNSUPPORTED", "metal_status": "BLOCKED_BY_CPU", "parity_status": "NOT_RUN", "publication": "NO_UPLOAD", "external_dependency": {"repository": "Qwen/Qwen2.5-7B", "revision": "UNSELECTED_BLOCKER", "selection_status": "BLOCKED", "files": "NOT_DOWNLOADED", "model_weights": "NOT_DOWNLOADED"}, "error_type": type(error).__name__, "reason": str(error), "blockers": [str(error)]}
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "blocker.txt").write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")


def self_test() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    assert hashlib.sha256(b"abc").hexdigest() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert FORMAT == "vokra-vibevoice-asr-inspection-v1"
    assert all(value is not None and HEX40.fullmatch(value) for value in SOURCE_ROLE_BLOBS.values())
    assert all(value is not None and HEX40.fullmatch(value) for value in TRANSFORMERS_ROLE_BLOBS.values())
    assert len(TRANSFORMERS_REVISION) == 40 and all(char in "0123456789abcdef" for char in TRANSFORMERS_REVISION)
    assert "safe_open" in source and 'framework="pt"' in source and ('framework=' + '"' + "numpy" + '"') not in source and "model.safetensors.index.json" in source
    constants = " ".join(str(value) for value in inspect.__code__.co_consts)
    assert "torch.load" not in constants and "pickle" not in constants.lower()
    with tempfile.TemporaryDirectory(prefix="vokra-vibevoice-asr-fixture-") as directory:
        root = Path(directory)
        snapshot = Path(directory) / "snapshot"
        snapshot.mkdir()
        weight_map = {}
        for number in range(1, SHARD_COUNT + 1):
            shard_name = f"model-{number:05d}-of-{SHARD_COUNT:05d}.safetensors"
            tensor_name = f"fixture.tensor.{number}"
            save_file({tensor_name: torch.tensor([float(number)], dtype=torch.bfloat16)}, str(snapshot / shard_name))
            weight_map[tensor_name] = shard_name
        index_path = snapshot / "model.safetensors.index.json"
        index_path.write_text(json.dumps({"metadata": EXPECTED_INDEX_METADATA, "weight_map": weight_map}) + "\n", encoding="utf-8")
        records, tensors = inventory_shards(snapshot, json.loads(index_path.read_text(encoding="utf-8")))
        assert len(records) == SHARD_COUNT and len(tensors) == SHARD_COUNT
        assert all(record["dtype"] == "bfloat16" and record["shape"] == [1] for record in tensors)
        config_fixture = {
            "model_type": "vibevoice",
            "architectures": ["VibeVoiceForASRTraining"],
            "dtype": "float32",
            "acoustic_vae_dim": 64,
            "semantic_vae_dim": 128,
            "acoustic_tokenizer_config": {"model_type": "vibevoice_acoustic_tokenizer", "vae_dim": 64, "dtype": "bfloat16"},
            "semantic_tokenizer_config": {"model_type": "vibevoice_semantic_tokenizer", "vae_dim": 128, "dtype": "bfloat16"},
            "decoder_config": {"model_type": "qwen2", "hidden_size": 3584, "num_hidden_layers": 28, "dtype": "bfloat16"},
            "diffusion_head_config": {"model_type": "vibepod_diffusion_head", "hidden_size": 3584, "latent_size": 64, "speech_vae_dim": 64},
        }
        (snapshot / "config.json").write_text(json.dumps(config_fixture), encoding="utf-8")
        (snapshot / "README.md").write_text("---\nlicense: mit\n---\n", encoding="utf-8")
        lfs_path = snapshot / "external.bin"; lfs_path.write_bytes(b"payload")
        tree_rows = []
        for path in sorted(snapshot.iterdir()):
            if path == lfs_path:
                continue
            payload_sha = sha256(path)
            tree_rows.append({"path": path.name, "type": "file", "size": path.stat().st_size, "git_blob_sha1": git_blob_sha1(path), "lfs_pointer_git_blob_sha1": None, "lfs_sha256": None})
        tree_rows.append({"path": lfs_path.name, "type": "file", "size": lfs_path.stat().st_size, "git_blob_sha1": None, "lfs_pointer_git_blob_sha1": lfs_pointer_sha1(sha256(lfs_path), lfs_path.stat().st_size), "lfs_sha256": sha256(lfs_path)})
        tree_path = Path(directory) / "server-tree.json"
        tree_path.write_text(json.dumps({"repository": UPSTREAM_REPOSITORY, "requested_revision": UPSTREAM_REVISION, "resolved_revision": UPSTREAM_REVISION, "walk": "recursive_file_only", "files": tree_rows}), encoding="utf-8")
        transport = snapshot / ".cache" / "huggingface"
        transport.mkdir(parents=True)
        (transport / "download-metadata.json").write_text("transport-only", encoding="utf-8")
        tree_evidence = server_inventory(snapshot, tree_path)
        assert len(tree_evidence["files"]) == len(tree_rows)
        assert tree_evidence["transport_cache"] == {
            "path": ".cache/huggingface",
            "scope": "snapshot_root_exact_transport_subtree",
            "present": True,
            "identity_role": "NON_IDENTITY_TRANSPORT_METADATA",
            "status": "EXCLUDED",
            "entry_count": 1,
        }
        nested_cache = snapshot / "nested" / ".cache"
        nested_cache.mkdir(parents=True)
        (nested_cache / "extra").write_bytes(b"extra")
        try:
            server_inventory(snapshot, tree_path)
        except RuntimeError as error:
            assert "unexpected cache path" in str(error)
        else:
            raise AssertionError("nested .cache was incorrectly excluded")
        (nested_cache / "extra").unlink()
        nested_cache.rmdir()
        transport_file = snapshot / ".cache" / "huggingface"
        (transport_file / "download-metadata.json").unlink()
        transport_file.rmdir()
        transport_file.write_bytes(b"not-a-directory")
        try:
            server_inventory(snapshot, tree_path)
        except RuntimeError as error:
            assert "real directory" in str(error)
        else:
            raise AssertionError("non-directory transport cache was accepted")
        transport_file.unlink()
        transport_file.symlink_to(root / "outside-transport", target_is_directory=True)
        (root / "outside-transport").mkdir()
        try:
            server_inventory(snapshot, tree_path)
        except RuntimeError as error:
            assert "real directory" in str(error)
        else:
            raise AssertionError("symlinked transport cache was accepted")
        transport_file.unlink()
        (snapshot / ".cache").rmdir()
        (snapshot / ".cache").symlink_to(root / "outside-cache", target_is_directory=True)
        (root / "outside-cache").mkdir()
        try:
            server_inventory(snapshot, tree_path)
        except RuntimeError as error:
            assert "root .cache" in str(error)
        else:
            raise AssertionError("symlinked root cache was accepted")
        (snapshot / ".cache").unlink()
        (snapshot / ".cache").write_bytes(b"not-a-directory")
        try:
            server_inventory(snapshot, tree_path)
        except RuntimeError as error:
            assert "root .cache" in str(error)
        else:
            raise AssertionError("non-directory root cache was accepted")
        (snapshot / ".cache").unlink()
        assert parsed_json_companions(snapshot, [])["config.json"] == config_fixture
        def assert_config_rejected(path: tuple[str, ...], value: Any) -> None:
            candidate = json.loads(json.dumps(config_fixture))
            target = candidate
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            (snapshot / "config.json").write_text(json.dumps(candidate), encoding="utf-8")
            try:
                parsed_json_companions(snapshot, [])
            except RuntimeError:
                return
            raise AssertionError(f"config mismatch was accepted: {'.'.join(path)}")

        for path, value in (
            (("model_type",), "vibevoice_wrong"),
            (("architectures",), ["VibeVoiceWrong"]),
            (("dtype",), "bfloat16"),
            (("acoustic_vae_dim",), 65),
            (("semantic_vae_dim",), 129),
            (("acoustic_tokenizer_config", "model_type"), "wrong_acoustic"),
            (("semantic_tokenizer_config", "model_type"), "wrong_semantic"),
            (("decoder_config", "model_type"), "wrong_decoder"),
            (("diffusion_head_config", "model_type"), "wrong_diffusion"),
            (("acoustic_tokenizer_config", "vae_dim"), 65),
            (("semantic_tokenizer_config", "vae_dim"), 129),
            (("decoder_config", "hidden_size"), 3585),
            (("decoder_config", "num_hidden_layers"), 29),
            (("decoder_config", "dtype"), "float32"),
            (("diffusion_head_config", "hidden_size"), 3585),
            (("diffusion_head_config", "latent_size"), 65),
            (("diffusion_head_config", "speech_vae_dim"), 65),
            (("acoustic_vae_dim",), True),
            (("decoder_config", "num_hidden_layers"), True),
        ):
            assert_config_rejected(path, value)
        missing_section = json.loads(json.dumps(config_fixture))
        missing_section.pop("semantic_tokenizer_config")
        (snapshot / "config.json").write_text(json.dumps(missing_section), encoding="utf-8")
        try:
            parsed_json_companions(snapshot, [])
        except RuntimeError:
            pass
        else:
            raise AssertionError("missing tokenizer structure was accepted")
        (snapshot / "config.json").write_text(json.dumps(config_fixture), encoding="utf-8")
        assert weight_license(snapshot, companion_inventory(snapshot))["license"] == "MIT"
        bad_card = snapshot / "README.md"; bad_card.write_text("prose license: mit\n", encoding="utf-8")
        try:
            weight_license(snapshot, companion_inventory(snapshot))
        except RuntimeError:
            pass
        else:
            raise AssertionError("prose-only model-card license accepted")
        bad_card.write_text("---\nlicense: mit\nlicense: mit\n---\n", encoding="utf-8")
        try:
            weight_license(snapshot, companion_inventory(snapshot))
        except RuntimeError:
            pass
        else:
            raise AssertionError("duplicate model-card license accepted")
        broken = dict(weight_map)
        broken["fixture.orphan"] = "model-00001-of-00008.safetensors"
        try:
            inventory_shards(snapshot, {"metadata": EXPECTED_INDEX_METADATA, "weight_map": broken})
        except RuntimeError:
            pass
        else:
            raise AssertionError("index fixture with missing tensor coverage was accepted")
        def auth_repo(path: Path, roles: tuple[str, ...], license_text: str, *, tag: str | None = None, pyproject: bool = False) -> tuple[str, dict[str, str]]:
            path.mkdir(); subprocess.run(["git", "init", "-q", str(path)], check=True)
            subprocess.run(["git", "-C", str(path), "config", "user.email", "vibevoice-asr-selftest@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(path), "config", "user.name", "VibeVoice ASR self-test"], check=True)
            for name in roles:
                target = path / name; target.parent.mkdir(parents=True, exist_ok=True)
                if name == "LICENSE":
                    content = license_text
                elif pyproject and name == "pyproject.toml":
                    content = "transformers>=4.51.3,<5.0.0\n"
                elif name in {"demo/vibevoice_asr_inference_from_file.py", "demo/vibevoice_asr_gradio_demo.py"}:
                    content = "VibeVoiceASRProcessor.from_pretrained(model_path, language_model_pretrained_name=\"Qwen/Qwen2.5-7B\")\n"
                elif pyproject and name == "vibevoice/modular/modeling_vibevoice_asr.py":
                    content = "self.acoustic_connector = SpeechConnector(config.acoustic_vae_dim, hidden_size)\nself.semantic_connector = SpeechConnector(config.semantic_vae_dim, hidden_size)\n"
                else:
                    content = f"role fixture {name}\n"
                target.write_text(content, encoding="utf-8")
            subprocess.run(["git", "-C", str(path), "add", *roles], check=True); subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "fixture"], check=True)
            if tag is not None: subprocess.run(["git", "-C", str(path), "tag", tag], check=True)
            repository = SOURCE_REPOSITORY if pyproject else TRANSFORMERS_REPOSITORY
            subprocess.run(["git", "-C", str(path), "remote", "add", "origin", repository], check=True)
            revision = git(path, "rev-parse", "HEAD")
            return revision, {name: git_blob_sha1(path / name) for name in roles}
        mit_text = 'MIT License\nPermission is hereby granted, free of charge, to any person obtaining a copy.\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.\n'
        apache_text = 'Apache License, Version 2.0\nYou may obtain a copy of the License.\nDistributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.\n'
        source_fixture = root / "source-auth-fixture"; transformers_fixture = root / "transformers-auth-fixture"
        source_revision, source_roles = auth_repo(source_fixture, tuple(SOURCE_ROLE_BLOBS), mit_text, pyproject=True)
        transformers_revision, transformers_roles = auth_repo(transformers_fixture, tuple(TRANSFORMERS_ROLE_BLOBS), apache_text, tag=TRANSFORMERS_TAG)
        source_evidence = source_inventory(source_fixture, transformers_fixture, source_revision=source_revision, source_roles=source_roles, transformers_revision=transformers_revision, transformers_roles=transformers_roles)
        assert source_evidence["source"]["status"] == "AUTHENTICATED" and source_evidence["transformers"]["status"] == "AUTHENTICATED"
    with tempfile.TemporaryDirectory(prefix="vokra-vibevoice-asr-blocked-") as directory:
        output = Path(directory)
        write_blocked(output, TypeError("fixture type failure"))
        blocked = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        assert blocked["status"] == "BLOCKED"
        assert blocked["error_type"] == "TypeError"
        assert "fixture type failure" in blocked["reason"]
    print("vibevoice_asr_inspect_reference.py self-test: OK (streaming/index/source contracts)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--transformers-source", type=Path)
    parser.add_argument("--server-tree", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.snapshot, args.source, args.transformers_source, args.server_tree, args.output)):
            parser.error("--self-test accepts no other arguments")
        self_test()
        return 0
    if any(value is None for value in (args.snapshot, args.source, args.transformers_source, args.server_tree, args.output)):
        parser.error("--snapshot, --source, --transformers-source, --server-tree, and --output are required")
    try:
        return inspect(args.snapshot, args.source, args.transformers_source, args.server_tree, args.output)
    except Exception as error:
        write_blocked(args.output, error)
        print(f"VibeVoice-ASR inspection BLOCKED: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"VibeVoice-ASR inspection: {error}", file=sys.stderr)
        raise SystemExit(2)
