#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Fail-closed VAST inspection for the gated Sesame CSM-1B release.

Only server identities, safetensors headers, JSON/config structure, and
non-executing archive inventories are inspected. This module never imports
CSM/Mimi code, unpickles a checkpoint, converts weights, or claims parity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import struct
import sys
import stat
import zipfile
from pathlib import Path
from typing import Any

HF_REPOSITORY = "sesame/csm-1b"
HF_REVISION = "c92a71e1c419772e25be7dc14d952c2521a740ab"
SOURCE_REPOSITORY = "https://github.com/SesameAILabs/csm.git"
SOURCE_REVISION = "8f6d947a26f6301deec9696f9bfb28e9e2e0d7d5"
TRANSFORMERS_REPOSITORY = "https://github.com/huggingface/transformers.git"
TRANSFORMERS_TAG = "v4.52.1"
TRANSFORMERS_COMMIT = "945727948c1143a10ac6f7d811aa58bb0d126b5b"
FORMAT = "vokra-csm-1b-inspection-v1"
MAX_HEADER_BYTES = 64 * 1024 * 1024
PUBLIC_REPOSITORY = "vokra/csm-1b"
PUBLIC_REVISION = "81613fc840fa995f4c8f1c48749fd731ed6424b8"
PUBLIC_FILE = "model.gguf"
PUBLIC_BYTES = 6211182304
PUBLIC_SHA256 = "e88619160b0aaaf090328b93aebd84f05734cf0ef3328af06c8067b1e30040dc"
PUBLIC_MANIFEST_SHA256 = "57837031bedcb4eaaecf497d966bb8c7788a3d4f173ba0f9d389e47395604bd1"
# The fixed HF CSM snapshot is now a Transformers composite: codec tensors
# and tokenizer.json are carried in the same authenticated tree. Keep these
# labels explicit so a future converter cannot silently re-introduce the old
# external Mimi/Meta-tokenizer assumption.
MIMI_REPOSITORY = "sesame/csm-1b#codec_model"
TOKENIZER_REPOSITORY = "sesame/csm-1b#tokenizer.json"

TREE = (
    (".gitattributes", 1948, "a70e11f76dd0e85c1f2293fc054c32aa5cae73e8"),
    ("README.md", 12140, "bd0f394147727f843e2abd0c2d3f57ce736bdb91"),
    ("chat_template.jinja", 2002, "d75309f235b5e74612dd78c18862213b610cc325"),
    ("ckpt.pt", 6219618714, "3f30fe9ef91a183ead3e9282c09a710491edc550"),
    ("config.json", 3280, "eb34392016242d345cec269e1623413f9071d910"),
    ("generation_config.json", 264, "e7beaf68e64dcd17fee50190107117d29c357ec0"),
    ("model.safetensors", 6211186784, "67a4748fc437cb9a2fdeb90e6bec9dedb0ad9f86"),
    ("preprocessor_config.json", 271, "42c2fd79770898fbd32fc3e04bd36b8851223700"),
    ("prompts/conversational_a.wav", 2646044, "93c9723750ed0d5168e12d4945035da27bf9541d"),
    ("prompts/conversational_b.wav", 2646044, "c3ccd13faeb0a4ce376d9ce7282805be10cad670"),
    ("prompts/read_speech_a.wav", 831412, "3b2b8f79b197d04765297daf39889961bc695907"),
    ("prompts/read_speech_b.wav", 576052, "367b950ea1eaeaab8eea0a3744af163e3ab2d418"),
    ("prompts/read_speech_c.wav", 385964, "2dd1413d07d5e8351ece2a5517fa10e537cabe66"),
    ("prompts/read_speech_d.wav", 435884, "bc279ba30df1b35742340b1caa71754cceee06e0"),
    ("special_tokens_map.json", 449, "e5b39b6305d89284b04934011c68dbb26bf588ca"),
    ("tokenizer.json", 17209980, "8de5df033b78de76dbe15fdd8b934678b5017aaf"),
    ("tokenizer_config.json", 50563, "9efdba317a3f1ea9acf0a99ad77e2451ecfa220c"),
    ("transformers-00001-of-00002.safetensors", 4944026784, "f6379cd719f180cfe3a0c3bd954903b632195979"),
    ("transformers-00002-of-00002.safetensors", 2189474180, "ca6ac15ccb23215d3813ba049010d5079aa08155"),
    ("transformers.safetensors.index.json", 59730, "6bd497e812938dc53a500a7fc941f4f04c3adecd"),
)
TREE_BY_PATH = {path: (size, oid) for path, size, oid in TREE}
TRANSFORMERS_ROLES = (
    "src/transformers/models/csm/configuration_csm.py",
    "src/transformers/models/csm/modeling_csm.py",
    "src/transformers/models/csm/processing_csm.py",
    "src/transformers/models/csm/generation_csm.py",
    "LICENSE",
)

# These are scoped source facts from the pinned, original implementation. They
# are evidence checks only: the inspector never imports or executes this
# Apache-licensed source. Keeping the values here prevents a generic file
# presence check from silently accepting a topology/config drift.
SOURCE_MARKERS = {
    "models.py": (
        "vocab_size=128_256",
        "num_layers=16",
        "num_heads=32",
        "num_kv_heads=8",
        "embed_dim=2048",
        "max_seq_len=2048",
        "intermediate_dim=8192",
        "rope_base=500_000",
        "scale_factor=32",
        "num_layers=4",
        "num_heads=8",
        "num_kv_heads=2",
        "embed_dim=1024",
        "audio_num_codebooks",
        "codebook0_head",
    ),
    "generator.py": (
        "meta-llama/Llama-3.2-1B",
        "kyutai/moshiko-pytorch-bf16",
        "mimi.set_num_codebooks(32)",
        "self.sample_rate = mimi.sample_rate",
        "CSM_1B_GH_WATERMARK",
        "max_audio_length_ms: float = 90_000",
    ),
}
SOURCE_ROLE_BLOBS = {
    "models.py": "180ca13699594e2818c5d0e906378feb4bff0c7b",
    "generator.py": "4778c81db89abd078a2b45e940a1033be9de1a7d",
    "LICENSE": "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64",
}
TRANSFORMERS_ROLE_BLOBS = {
    "src/transformers/models/csm/configuration_csm.py": "e6d6d2e27c6a7b1d561740010dd3ba75f2255249",
    "src/transformers/models/csm/modeling_csm.py": "58042c64abbe6612f5f5ed0d1776fb45850eaf19",
    "src/transformers/models/csm/processing_csm.py": "486c5eda4c76f7d1a58cf782b11892f1cd9dffeb",
    "src/transformers/models/csm/generation_csm.py": "2fec3ea8919fa0c0e0782b54dcafe79e317ec9f3",
    "LICENSE": "68b7d66c97d66c58de883ed0c451af2b3183e6f3",
}
GENERATION_CSM_ROLE_BLOB = "2fec3ea8919fa0c0e0782b54dcafe79e317ec9f3"


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
    except Exception as error:
        raise RuntimeError(f"strict JSON failure at {path}: {error}") from error


def inspect_generation_config(generation: Any) -> dict[str, Any]:
    """Bind the source config separately from the reference call overrides."""
    if not isinstance(generation, dict):
        raise RuntimeError("CSM generation_config must be an object")
    if "do_sample" in generation and not isinstance(generation["do_sample"], bool):
        raise RuntimeError("CSM generation_config do_sample must be boolean when present")
    return {
        "source_do_sample": generation.get("do_sample"),
        "reference_overrides": {
            "do_sample": False,
            "depth_decoder_do_sample": False,
        },
        "final_generation_semantics": "reference_overrides_are_applied_at_model_generate_boundary",
    }


def safe_path(value: str) -> None:
    path = Path(value)
    if not value or "\0" in value or "\\" in value or path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"unsafe path: {value!r}")


def validate_apache_license(text: str, label: str) -> None:
    """Require the substantive Apache 2.0 title, grant, terms, and warranty."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not re.search(r"\bApache License\s+Version 2\.0,\s*January 2004\b", normalized):
        raise RuntimeError(f"{label} Apache title/version evidence missing")
    if "Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license" not in normalized:
        raise RuntimeError(f"{label} Apache grant evidence missing")
    if "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION" not in normalized:
        raise RuntimeError(f"{label} Apache terms evidence missing")
    if "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND" not in normalized:
        raise RuntimeError(f"{label} Apache warranty evidence missing")


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_blob(path: Path, data: bytes | None = None) -> str:
    if data is not None:
        return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    size = path.stat().st_size
    hasher = hashlib.sha1()
    hasher.update(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def lfs_pointer_blob(size: int, sha: str) -> str:
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha}\nsize {size}\n".encode()
    return hashlib.sha1(f"blob {len(pointer)}\0".encode() + pointer).hexdigest()


def inspect_public_gguf(path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    """Read only the GGUF header and bind it to the reviewed contract."""
    if path.stat().st_size != PUBLIC_BYTES or digest(path) != PUBLIC_SHA256:
        raise RuntimeError("historical public GGUF payload identity mismatch")
    with path.open("rb") as stream:
        header = stream.read(64 * 1024 * 1024)
    cursor = 0

    def take(size: int) -> bytes:
        nonlocal cursor
        if cursor + size > len(header):
            raise RuntimeError("GGUF header is truncated")
        value = header[cursor : cursor + size]
        cursor += size
        return value

    def u32() -> int:
        return struct.unpack("<I", take(4))[0]

    def u64() -> int:
        return struct.unpack("<Q", take(8))[0]

    def string() -> str:
        size = u64()
        if size > 64 * 1024 * 1024:
            raise RuntimeError("GGUF string exceeds bound")
        try:
            return take(size).decode("utf-8")
        except UnicodeDecodeError as error:
            raise RuntimeError("GGUF string is not UTF-8") from error

    def skip(value_type: int) -> None:
        widths = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if value_type in widths:
            take(widths[value_type])
        elif value_type == 8:
            string()
        elif value_type == 9:
            element_type = u32()
            count = u64()
            if count > 1_000_000:
                raise RuntimeError("GGUF metadata array exceeds bound")
            for _ in range(count):
                skip(element_type)
        else:
            raise RuntimeError(f"unknown GGUF metadata type {value_type}")

    if take(4) != b"GGUF":
        raise RuntimeError("historical public artifact is not GGUF")
    version = u32()
    if version not in (2, 3):
        raise RuntimeError("unsupported GGUF version")
    tensor_count = u64()
    metadata_count = u64()
    if tensor_count > 1_000_000 or metadata_count > 1_000_000:
        raise RuntimeError("GGUF header counts exceed bound")
    for _ in range(metadata_count):
        string()
        skip(u32())
    dtype_names = {0: "F32", 1: "F16", 30: "BF16"}
    tensors = []
    for _ in range(tensor_count):
        name = string()
        dimensions = u32()
        if dimensions > 4:
            raise RuntimeError("GGUF tensor rank exceeds runtime contract")
        shape = [u64() for _ in range(dimensions)]
        dtype = dtype_names.get(u32(), "UNKNOWN")
        offset = u64()
        tensors.append({"dtype": dtype, "name": name, "offset": offset, "shape": shape})
    if cursor != contract.get("header_bytes"):
        raise RuntimeError(f"GGUF header length mismatch: {cursor} != {contract.get('header_bytes')}")
    expected = contract.get("tensors")
    if tensors != expected or len(tensors) != 187:
        raise RuntimeError("GGUF tensor descriptors differ from reviewed contract")
    manifest_hash = hashlib.sha256(json.dumps(tensors, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if manifest_hash != contract.get("manifest_sha256"):
        raise RuntimeError("GGUF tensor descriptor manifest hash mismatch")
    return {"bytes": path.stat().st_size, "header_bytes": cursor, "tensor_count": tensor_count, "manifest_sha256": manifest_hash, "descriptor_source": "downloaded GGUF header"}


def inspect_tree(snapshot: Path, packet: Path) -> dict[str, Any]:
    envelope = load_json(packet)
    if not isinstance(envelope, dict) or set(envelope) != {"repository", "requested_revision", "resolved_revision", "files"} or envelope.get("repository") != HF_REPOSITORY or envelope.get("requested_revision") != HF_REVISION or envelope.get("resolved_revision") != HF_REVISION:
        raise RuntimeError("CSM HF server identity mismatch")
    rows = envelope.get("files")
    if not isinstance(rows, list) or len(rows) != len(TREE):
        raise RuntimeError("CSM HF tree must contain exactly 20 files")
    by_path: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "type", "size", "git_blob_sha1", "lfs_sha256", "lfs_size"}:
            raise RuntimeError("malformed server tree row")
        path = row["path"]
        safe_path(path)
        if row["type"] != "file" or path in by_path or not isinstance(row["size"], int) or isinstance(row["size"], bool):
            raise RuntimeError(f"invalid server tree row: {path!r}")
        wanted = TREE_BY_PATH.get(path)
        if wanted is None or row["size"] != wanted[0] or row["git_blob_sha1"].lower() != wanted[1]:
            raise RuntimeError(f"fixed CSM tree identity mismatch: {path}")
        by_path[path] = row
    actual = set()
    for path in snapshot.rglob("*"):
        relative = path.relative_to(snapshot)
        # ``snapshot_download(local_dir=...)`` may leave only this exact
        # Hugging Face transport metadata directory. It is not model content;
        # every other cache path participates in the authenticated set.
        if relative.parts[:2] == (".cache", "huggingface"):
            if path.is_symlink():
                raise RuntimeError(f"transport cache symlink is forbidden: {relative}")
            continue
        if path.is_symlink():
            raise RuntimeError(f"snapshot payload symlink is forbidden: {relative}")
        elif path.is_file():
            actual.add(relative.as_posix())
        elif not path.is_dir():
            raise RuntimeError(f"non-regular snapshot member: {relative}")
    if actual != set(TREE_BY_PATH):
        raise RuntimeError(f"CSM local tree mismatch: missing={sorted(set(TREE_BY_PATH)-actual)} extra={sorted(actual-set(TREE_BY_PATH))}")
    records = []
    for path, row in by_path.items():
        local = snapshot / path
        if local.stat().st_size != row["size"]:
            raise RuntimeError(f"size mismatch: {path}")
        sha = digest(local)
        remote_lfs = row["lfs_sha256"]
        if remote_lfs is not None:
            if not isinstance(remote_lfs, str) or not re.fullmatch(r"[0-9a-f]{64}", remote_lfs) or row["lfs_size"] != row["size"]:
                raise RuntimeError(f"malformed remote LFS identity: {path}")
            if sha != remote_lfs:
                raise RuntimeError(f"remote LFS/local payload mismatch: {path}")
            pointer_oid = lfs_pointer_blob(row["lfs_size"], remote_lfs)
            if pointer_oid != row["git_blob_sha1"]:
                raise RuntimeError(f"LFS pointer Git identity mismatch: {path}")
            identity_kind = "lfs-pointer"
        else:
            if row["lfs_size"] is not None or git_blob(local) != row["git_blob_sha1"]:
                raise RuntimeError(f"Git/LFS identity mismatch: {path}")
            identity_kind = "git-blob"
        records.append({"path": path, "bytes": row["size"], "git_blob_sha1": row["git_blob_sha1"], "lfs_sha256": remote_lfs, "lfs_size": row["lfs_size"], "sha256": sha, "identity_kind": identity_kind})
    if sum(item["bytes"] for item in records) != 19_589_168_489:
        raise RuntimeError("CSM fixed tree total bytes mismatch")
    return {"repository": HF_REPOSITORY, "requested_revision": HF_REVISION, "resolved_revision": HF_REVISION, "files": records}


def inspect_safetensors(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise RuntimeError(f"short safetensors header: {path}")
        header_length = int.from_bytes(raw_length, "little")
        if header_length > MAX_HEADER_BYTES or 8 + header_length > size:
            raise RuntimeError(f"bounded safetensors header violation: {path}")
        header_raw = stream.read(header_length)
    header = json.loads(header_raw.decode("utf-8"), object_pairs_hook=strict_pairs)
    if not isinstance(header, dict):
        raise RuntimeError("safetensors header must be an object")
    widths = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
    data_size = size - 8 - header_length
    ranges = []
    for name, descriptor in header.items():
        if name == "__metadata__":
            if not isinstance(descriptor, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in descriptor.items()):
                raise RuntimeError("safetensors metadata must be a string map")
            continue
        safe_path(name)
        if not isinstance(descriptor, dict) or set(descriptor) != {"dtype", "shape", "data_offsets"}:
            raise RuntimeError(f"invalid tensor descriptor: {name}")
        dtype, shape, offsets = descriptor["dtype"], descriptor["shape"], descriptor["data_offsets"]
        if dtype not in widths or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise RuntimeError(f"invalid tensor descriptor values: {name}")
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in shape + offsets):
            raise RuntimeError(f"invalid tensor shape/offset: {name}")
        start, end = offsets
        elements = 1
        for dim in shape:
            elements *= dim
        if end < start or end > data_size or end - start != elements * widths[dtype]:
            raise RuntimeError(f"tensor span mismatch: {name}")
        ranges.append((start, end, name, dtype, shape, elements))
    ranges.sort()
    cursor = 0
    for start, end, *_ in ranges:
        if start != cursor:
            raise RuntimeError("safetensors gap/overlap")
        cursor = end
    if cursor != data_size or not ranges:
        raise RuntimeError("safetensors data region incomplete")
    tensors = [{"name": x[2], "dtype": x[3], "shape": x[4], "elements": x[5]} for x in ranges]
    manifest_hash = hashlib.sha256(json.dumps(tensors, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"bytes": size, "header_bytes": header_length, "tensor_count": len(ranges), "tensor_manifest_sha256": manifest_hash, "tensors": tensors}


def inspect_index(snapshot: Path) -> dict[str, Any]:
    index = load_json(snapshot / "transformers.safetensors.index.json")
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise RuntimeError("Transformers safetensors index missing weight_map")
    weight_map = index["weight_map"]
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in weight_map.items()):
        raise RuntimeError("weight_map requires string keys and values")
    for tensor_name, shard_name in weight_map.items():
        safe_path(tensor_name)
        safe_path(shard_name)
    shard_names = {"transformers-00001-of-00002.safetensors", "transformers-00002-of-00002.safetensors"}
    if set(weight_map.values()) != shard_names:
        raise RuntimeError("Transformers weight_map shard set mismatch")
    shards = {name: inspect_safetensors(snapshot / name) for name in sorted(shard_names)}
    occurrences: dict[str, list[str]] = {}
    for name, header in shards.items():
        for tensor in header["tensors"]:
            occurrences.setdefault(tensor["name"], []).append(name)
    if any(len(shards_for_tensor) != 1 for shards_for_tensor in occurrences.values()):
        raise RuntimeError("Transformers tensor appears in multiple physical shards")
    owners = {tensor_name: shard_names[0] for tensor_name, shard_names in occurrences.items()}
    if set(owners) != set(weight_map) or any(owners[name] != shard for name, shard in weight_map.items()):
        raise RuntimeError("Transformers index/header ownership mismatch")
    all_tensors = [(shard, tensor) for shard, header in sorted(shards.items()) for tensor in header["tensors"]]
    manifest_hash = hashlib.sha256(json.dumps(all_tensors, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    names = {tensor["name"] for _, tensor in all_tensors}
    # The current fixed snapshot is a CSMForConditionalGeneration composite.
    # Prove that the two transformer shards carry all four model roles before a
    # converter is ever allowed to map them. Prefix counts alone are not a
    # conversion; they are the minimum fail-closed topology check.
    required_exact = {
        "text_embedding": "embed_text_tokens.weight",
        "backbone_audio_embedding": "backbone_model.embed_tokens.embed_audio_tokens.weight",
        "backbone_lm_head": "lm_head.weight",
        "depth_audio_embedding": "depth_decoder.model.embed_tokens.weight",
        "depth_codebook_head": "depth_decoder.codebooks_head.weight",
    }
    missing_exact = [role for role, tensor_name in required_exact.items() if tensor_name not in names]
    if missing_exact:
        raise RuntimeError(f"CSM composite tensor roles missing: {missing_exact}")
    role_prefixes = {
        "backbone": "backbone_model.",
        "depth_decoder": "depth_decoder.",
        "codec_mimi": "codec_model.",
    }
    role_counts = {
        role: sum(name.startswith(prefix) for name in names)
        for role, prefix in role_prefixes.items()
    }
    if any(count == 0 for count in role_counts.values()):
        raise RuntimeError(f"CSM composite role prefix missing: {role_counts}")
    descriptors = {
        tensor["name"]: {"shape": tensor["shape"], "dtype": tensor["dtype"]}
        for _, tensor in all_tensors
    }
    exact_ownership = {
        role: {"tensor": tensor_name, "shard": owners[tensor_name], **descriptors[tensor_name]}
        for role, tensor_name in required_exact.items()
    }
    return {"weight_map_count": len(weight_map), "shards": shards, "tensor_manifest_sha256": manifest_hash, "tensor_count": len(all_tensors), "roles": {"exact": required_exact, "exact_ownership": exact_ownership, "prefix_counts": role_counts, "status": "FULL_CSM_TRANSFORMERS_COMPOSITE_ROLES_PRESENT"}}


def inspect_json_roles(snapshot: Path) -> dict[str, Any]:
    config = load_json(snapshot / "config.json")
    generation = load_json(snapshot / "generation_config.json")
    preprocessor = load_json(snapshot / "preprocessor_config.json")
    tokenizer = load_json(snapshot / "tokenizer_config.json")
    special = load_json(snapshot / "special_tokens_map.json")
    tokenizer_json = load_json(snapshot / "tokenizer.json")
    if not isinstance(config, dict) or not isinstance(generation, dict) or not isinstance(preprocessor, dict) or not isinstance(tokenizer, dict) or not isinstance(special, dict):
        raise RuntimeError("CSM JSON roles must be objects")
    generation_contract = inspect_generation_config(generation)
    if config.get("model_type") != "csm" or config.get("architectures") != ["CsmForConditionalGeneration"]:
        raise RuntimeError("CSM config topology is incomplete")
    if config.get("transformers_weights") != "transformers.safetensors.index.json":
        raise RuntimeError("CSM must select the authenticated transformers.safetensors.index.json composite")
    expected_config = {
        "num_codebooks": 32,
        "vocab_size": 2051,
        "text_vocab_size": 128256,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 2048,
        "codebook_pad_token_id": 2050,
        "codebook_eos_token_id": 0,
        "rope_theta": 500000,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise RuntimeError(f"CSM config {key} mismatch: {config.get(key)!r} != {expected!r}")
    depth = config.get("depth_decoder_config")
    if not isinstance(depth, dict):
        raise RuntimeError("CSM depth_decoder_config is missing")
    expected_depth = {
        "num_codebooks": 32,
        "backbone_hidden_size": 2048,
        "vocab_size": 2051,
        "hidden_size": 1024,
        "intermediate_size": 8192,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "max_position_embeddings": 33,
        "rope_theta": 500000,
    }
    for key, expected in expected_depth.items():
        if depth.get(key) != expected:
            raise RuntimeError(f"CSM depth config {key} mismatch: {depth.get(key)!r} != {expected!r}")
    codec = config.get("codec_config")
    if not isinstance(codec, dict) or codec.get("model_type") != "mimi":
        raise RuntimeError("CSM codec_config is not an embedded Mimi config")
    for key, expected in {"sampling_rate": 24000, "num_codebooks": 32, "codebook_size": 2048, "hidden_size": 512}.items():
        if codec.get(key) != expected:
            raise RuntimeError(f"CSM Mimi config {key} mismatch: {codec.get(key)!r} != {expected!r}")
    if config.get("bos_token_id") != 128000 or config.get("eos_token_id") != 128001 or config.get("pad_token_id") != 128002:
        raise RuntimeError("CSM config special-token IDs differ from the pinned Llama contract")
    if tokenizer.get("tokenizer_class") not in ("PreTrainedTokenizerFast", "LlamaTokenizer", "LlamaTokenizerFast") or not special:
        raise RuntimeError("CSM tokenizer/special-token contract incomplete")
    if any(special.get(key) != expected for key, expected in {"bos_token": "<|begin_of_text|>", "eos_token": "<|end_of_text|>", "pad_token": "<|end_of_text|>"}.items()):
        raise RuntimeError("CSM special-token map mismatch")
    if not isinstance(tokenizer_json, dict) or not isinstance(tokenizer_json.get("model"), dict):
        raise RuntimeError("CSM tokenizer JSON topology incomplete")
    tokenizer_model = tokenizer_json["model"]
    if tokenizer_model.get("type") != "BPE" or not isinstance(tokenizer_model.get("vocab"), dict) or not tokenizer_model["vocab"]:
        raise RuntimeError("CSM tokenizer must be a non-empty BPE vocabulary")
    if not isinstance(tokenizer_model.get("merges"), list) or not tokenizer_model["merges"]:
        raise RuntimeError("CSM tokenizer BPE merges are missing")
    if tokenizer_model.get("byte_fallback") is not True:
        raise RuntimeError(
            "CSM native tokenizer route is staged/unaccepted: raw tokenizer.json "
            "byte_fallback topology is not the reviewed production subset"
        )
    post_processor = tokenizer_json.get("post_processor")
    if not isinstance(post_processor, dict) or post_processor.get("type") != "TemplateProcessing":
        raise RuntimeError("CSM tokenizer post_processor must be TemplateProcessing")
    single = post_processor.get("single")
    if not isinstance(single, list) or len(single) < 3:
        raise RuntimeError("CSM tokenizer post_processor.single is incomplete")
    def processor_atom(value: Any) -> str | None:
        if value == "A":
            return "sequence"
        if value == "<|begin_of_text|>":
            return "bos"
        if value == "<|end_of_text|>":
            return "eos"
        if isinstance(value, dict) and isinstance(value.get("SpecialToken"), dict):
            name = value["SpecialToken"].get("id")
            return {"<|begin_of_text|>": "bos", "<|end_of_text|>": "eos"}.get(name)
        if isinstance(value, dict) and isinstance(value.get("Sequence"), dict) and value["Sequence"].get("id") == "A":
            return "sequence"
        return None
    if processor_atom(single[0]) != "bos" or processor_atom(single[-1]) != "eos" or "sequence" not in {processor_atom(item) for item in single[1:-1]}:
        raise RuntimeError("CSM tokenizer post_processor must be BOS + sequence + EOS")
    special_processor_tokens = post_processor.get("special_tokens")
    if not isinstance(special_processor_tokens, dict) or not {"<|begin_of_text|>", "<|end_of_text|>"}.issubset(special_processor_tokens):
        raise RuntimeError("CSM tokenizer post_processor special-token map is incomplete")
    added_tokens = tokenizer_json.get("added_tokens")
    if not isinstance(added_tokens, list):
        raise RuntimeError("CSM tokenizer added_tokens are missing")
    added = {
        item.get("content"): item.get("id")
        for item in added_tokens
        if isinstance(item, dict)
    }
    if added.get("<|begin_of_text|>") != config.get("bos_token_id", 128000) or added.get("<|end_of_text|>") != config.get("eos_token_id", 128001):
        raise RuntimeError("CSM tokenizer BOS/EOS IDs differ from the pinned config")
    if preprocessor.get("sampling_rate") != 24_000:
        raise RuntimeError("CSM preprocessor sample-rate contract mismatch")
    template = (snapshot / "chat_template.jinja").read_text(encoding="utf-8")
    if "bos_token" not in template and "<|begin_of_text|>" not in template:
        raise RuntimeError("CSM chat template lacks BOS contract")
    if "<|AUDIO|>" not in template or "audio_eos" not in template or "message['role']" not in template:
        raise RuntimeError("CSM chat template audio/speaker contract mismatch")
    return {"config": config, "generation_config": generation, "generation_config_sha256": digest(snapshot / "generation_config.json"), "generation_contract": generation_contract, "preprocessor_config": preprocessor, "tokenizer_config": tokenizer, "tokenizer_json": {"sha256": digest(snapshot / "tokenizer.json"), "model_type": tokenizer_model["type"], "vocab_count": len(tokenizer_model["vocab"]), "merge_count": len(tokenizer_model["merges"]), "byte_fallback": tokenizer_model["byte_fallback"], "added_token_ids": {"bos": added["<|begin_of_text|>"], "eos": added["<|end_of_text|>"]}}, "native_route_status": "STAGED_UNACCEPTED_PENDING_CANONICAL_ID_PARITY", "special_tokens": special, "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(), "composite_config": {"backbone": expected_config, "depth_decoder": expected_depth, "codec": {"sampling_rate": 24000, "num_codebooks": 32, "codebook_size": 2048, "hidden_size": 512}}}


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    # A checkpoint is inspected through PyTorch's restricted loader only. The
    # unrestricted pickle path is intentionally never available here.
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) > 100_000 or archive.infolist() and sum(item.file_size for item in archive.infolist()) > path.stat().st_size:
                raise RuntimeError("bounded ckpt archive limit exceeded")
            for name in names:
                safe_path(name)
                if len(name) > 4096:
                    raise RuntimeError("ckpt archive member name is too long")
            if len(names) != len(set(names)):
                raise RuntimeError("duplicate ckpt archive member")
        import torch

        unsafe_globals = torch.serialization.get_unsafe_globals_in_checkpoint(str(path))
        reject_unsafe_globals(unsafe_globals)
        loaded = torch.load(path, weights_only=True, map_location="cpu")
    except zipfile.BadZipFile as error:
        raise RuntimeError(f"ckpt is not a supported torch archive: {error}") from error
    except Exception as error:
        raise RuntimeError(f"restricted torch.load failed; no unsafe fallback: {error}") from error

    entries: list[dict[str, Any]] = []

    def walk(value: Any, prefix: str, depth: int = 0) -> None:
        if depth > 32 or len(entries) > 100_000:
            raise RuntimeError("ckpt recursive manifest bound exceeded")
        if isinstance(value, torch.Tensor):
            if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
                raise RuntimeError(f"non-finite ckpt tensor: {prefix}")
            entries.append({"name": prefix, "shape": [int(x) for x in value.shape], "dtype": str(value.dtype), "elements": int(value.numel())})
            return
        if isinstance(value, dict):
            for key in sorted(value, key=lambda item: str(item)):
                if not isinstance(key, str):
                    raise RuntimeError("ckpt manifest requires string mapping keys")
                walk(value[key], f"{prefix}.{key}" if prefix else key, depth + 1)
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, f"{prefix}[{index}]", depth + 1)
            return
        if value is None or isinstance(value, (bool, int, float, str)):
            return
        raise RuntimeError(f"unsupported ckpt value at {prefix}: {type(value).__name__}")

    walk(loaded, "")
    if not entries:
        raise RuntimeError("ckpt contains no tensors")
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {"format": "torch-zip", "members": len(names), "execution": "WEIGHTS_ONLY", "unsafe_globals": sorted(unsafe_globals), "tensor_count": len(entries), "tensor_manifest_sha256": hashlib.sha256(encoded).hexdigest(), "tensors": entries, "status": "SAFE_LOAD_COMPLETE"}


def reject_unsafe_globals(values: Any) -> None:
    if values:
        raise RuntimeError(f"ckpt contains unsafe globals: {sorted(values)}")


def _tracked_checkout(root: Path, expected_roles: dict[str, str], label: str) -> dict[str, Any]:
    """Authenticate every tracked object, mode, and working-tree byte."""
    index_raw = subprocess.check_output(["git", "-C", str(root), "ls-files", "-s", "-z"])
    index: dict[str, tuple[str, str]] = {}
    for record in index_raw.split(b"\0"):
        if not record:
            continue
        try:
            header, raw_path = record.split(b"\t", 1)
            mode, oid, stage = header.decode("ascii").split(" ")
            path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise RuntimeError(f"{label} index entry is malformed") from error
        safe_path(path)
        if stage != "0" or path in index:
            raise RuntimeError(f"{label} index has duplicate/non-stage-zero entry: {path}")
        index[path] = (mode, oid)
    head_raw = subprocess.check_output(["git", "-C", str(root), "ls-tree", "-r", "-z", "HEAD"])
    head: dict[str, tuple[str, str]] = {}
    for record in head_raw.split(b"\0"):
        if not record:
            continue
        try:
            header, raw_path = record.split(b"\t", 1)
            mode, kind, oid = header.decode("ascii").split(" ")
            path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise RuntimeError(f"{label} HEAD tree entry is malformed") from error
        safe_path(path)
        if kind != "blob" or path in head:
            raise RuntimeError(f"{label} HEAD contains a non-regular/duplicate entry: {path}")
        head[path] = (mode, oid)
    if set(index) != set(head):
        raise RuntimeError(f"{label} index/HEAD tracked path set mismatch")
    for path, (mode, oid) in index.items():
        if mode not in {"100644", "100755"} or head[path] != (mode, oid):
            raise RuntimeError(f"{label} index/HEAD mode or object mismatch: {path}")
        local = root / path
        if local.is_symlink() or not local.is_file():
            raise RuntimeError(f"{label} tracked path is not a regular file: {path}")
        expected_mode = 0o755 if mode == "100755" else 0o644
        if stat.S_IMODE(local.stat().st_mode) != expected_mode:
            raise RuntimeError(f"{label} working-tree mode mismatch: {path}")
        if git_blob(local) != oid:
            raise RuntimeError(f"{label} index/working-tree object mismatch: {path}")
    for role, expected in expected_roles.items():
        if role not in index or index[role][0] != "100644" or index[role][1] != expected:
            raise RuntimeError(f"{label} fixed role identity mismatch: {role}")
    return {"tracked_count": len(index), "roles": {role: {"mode": index[role][0], "git_blob_sha1": index[role][1]} for role in expected_roles}}


def inspect_source(source: Path, transformers: Path) -> dict[str, Any]:
    if git(source, "rev-parse", "HEAD") != SOURCE_REVISION or git(source, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("CSM source revision/clean checkout mismatch")
    origin = git(source, "remote", "get-url", "origin").rstrip("/").removesuffix(".git")
    if origin != SOURCE_REPOSITORY.removesuffix(".git"):
        raise RuntimeError("CSM source origin mismatch")
    required = ("models.py", "generator.py", "LICENSE")
    source_checkout = _tracked_checkout(source, SOURCE_ROLE_BLOBS, "CSM source")
    if any(not (source / name).is_file() for name in required):
        raise RuntimeError("CSM source role missing")
    source_text = {name: (source / name).read_text(encoding="utf-8", errors="strict") for name in ("models.py", "generator.py", "LICENSE")}
    validate_apache_license(source_text["LICENSE"], "CSM source")
    for role, markers in SOURCE_MARKERS.items():
        missing = [marker for marker in markers if marker not in source_text[role]]
        if missing:
            raise RuntimeError(f"CSM source semantic markers missing in {role}: {missing}")
    transformer_origin = git(transformers, "remote", "get-url", "origin").rstrip("/").removesuffix(".git")
    if transformer_origin != TRANSFORMERS_REPOSITORY.removesuffix(".git"):
        raise RuntimeError("Transformers source origin mismatch")
    if git(transformers, "describe", "--exact-match", "--tags", "HEAD") != TRANSFORMERS_TAG:
        raise RuntimeError("Transformers tag mismatch")
    if git(transformers, "rev-parse", "HEAD") != TRANSFORMERS_COMMIT:
        raise RuntimeError("Transformers resolved commit mismatch")
    if git(transformers, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Transformers checkout is not clean")
    transformer_roles_list = TRANSFORMERS_ROLES
    if any(not (transformers / role).is_file() for role in transformer_roles_list):
        raise RuntimeError("Transformers CSM role missing")
    transformer_commit = git(transformers, "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", transformer_commit):
        raise RuntimeError("Transformers resolved commit is not immutable")
    transformer_expected = {name: TRANSFORMERS_ROLE_BLOBS[name] for name in TRANSFORMERS_ROLES}
    transformer_checkout = _tracked_checkout(transformers, transformer_expected, "Transformers")
    transformer_roles = {name: {"sha256": digest(transformers / name), "git_blob_sha1": git_blob(transformers / name), "mode": "100644"} for name in transformer_roles_list}
    transformers_license = (transformers / "LICENSE").read_text(encoding="utf-8", errors="strict")
    validate_apache_license(transformers_license, "Transformers")
    return {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION, "roles": {name: {"sha256": digest(source / name), "git_blob_sha1": git_blob(source / name), "mode": "100644"} for name in required}, "tracked": source_checkout, "semantic_markers": SOURCE_MARKERS, "transformers": {"repository": TRANSFORMERS_REPOSITORY, "tag": TRANSFORMERS_TAG, "commit": transformer_commit, "tracked": transformer_checkout, "roles": transformer_roles}}


def manifest(output: Path, inspection_status: str, evidence: dict[str, Any] | None = None, error: str | None = None) -> None:
    result: dict[str, Any] = {"format": FORMAT, "status": "BLOCKED", "inspection_status": inspection_status, "collection_status": "AUTHENTICATED" if inspection_status == "AUTHENTICATED_EVIDENCE_COMPLETE" else "FAILED", "evidence_scope": "CSM_TRANSFORMERS_COMPOSITE_AND_FORMATS", "composite_status": "BLOCKED_ROLE_MAPPING_AND_PARITY", "comparison_status": "NOT_RUN_OFFICIAL_ONLY", "evidence_stage": "INSPECTION_ONLY", "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "cpu_status": "UNSUPPORTED", "metal_status": "BLOCKED_BY_CPU", "parity_status": "NOT_RUN", "publication": "NO_UPLOAD", "model": {"repository": HF_REPOSITORY, "revision": HF_REVISION}, "source_identity": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION}, "transformers_identity": {"repository": TRANSFORMERS_REPOSITORY, "tag": TRANSFORMERS_TAG, "commit": TRANSFORMERS_COMMIT}, "required_composite": {"mimi": {"repository": MIMI_REPOSITORY, "status": "IN_SNAPSHOT_CODEC_ROLE_PROOF_REQUIRED"}, "text_tokenizer": {"repository": TOKENIZER_REPOSITORY, "status": "IN_SNAPSHOT_AUTHENTICATED_TOKENIZER"}}, "license_provenance": {"source_code": "Apache-2.0 grant and warranty clauses evidenced by pinned source LICENSE", "transformers_code": "Apache-2.0 grant and warranty clauses evidenced by pinned Transformers LICENSE", "composite_weights": "UNRESOLVED_FAIL_CLOSED — no weight license claim is inferred from the repo card; explicit redistribution/signoff is required"}, "historical_public_artifact": {"repository": PUBLIC_REPOSITORY, "revision": PUBLIC_REVISION, "file": PUBLIC_FILE, "bytes": PUBLIC_BYTES, "sha256": PUBLIC_SHA256, "status": "runtime-incomplete legacy / CSM-core-only"}}
    if evidence:
        result["evidence"] = evidence
    if error:
        result["error"] = error
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT).strip()


def self_test() -> bool:
    standard_apache = """Apache License
Version 2.0, January 2004

Licensed under the Apache License, Version 2.0 (the \"License\");
TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license.
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""
    try:
        validate_apache_license(standard_apache, "self-test")
    except RuntimeError:
        return False
    for missing in (
        standard_apache.replace("Version 2.0, January 2004", "Version 1.0, January 2004"),
        standard_apache.replace("Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license.", "Subject to another grant."),
        standard_apache.replace("WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND", "WITH WARRANTIES"),
    ):
        try:
            validate_apache_license(missing, "self-test")
        except RuntimeError:
            pass
        else:
            return False
    if inspect_generation_config({"do_sample": True}) != {
        "source_do_sample": True,
        "reference_overrides": {"do_sample": False, "depth_decoder_do_sample": False},
        "final_generation_semantics": "reference_overrides_are_applied_at_model_generate_boundary",
    }:
        return False
    try:
        inspect_generation_config({"do_sample": 0})
    except RuntimeError:
        pass
    else:
        return False
    try:
        reject_unsafe_globals(["os.system"])
    except RuntimeError:
        pass
    else:
        return False
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        header = json.dumps({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        good = root / "good.safetensors"
        good.write_bytes(len(header).to_bytes(8, "little") + header + b"\0" * 4)
        if inspect_safetensors(good)["tensor_count"] != 1:
            return False
        huge = root / "huge.safetensors"
        huge.write_bytes((MAX_HEADER_BYTES + 1).to_bytes(8, "little"))
        try:
            inspect_safetensors(huge)
        except RuntimeError:
            pass
        else:
            return False
        duplicate = root / "duplicate.json"
        duplicate.write_text('{"x":1,"x":2}', encoding="utf-8")
        try:
            load_json(duplicate)
        except RuntimeError:
            pass
        else:
            return False
        unsafe = root / "unsafe.safetensors"
        unsafe_header = json.dumps({"../tensor": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        unsafe.write_bytes(len(unsafe_header).to_bytes(8, "little") + unsafe_header + b"\0" * 4)
        try:
            inspect_safetensors(unsafe)
        except RuntimeError:
            pass
        else:
            return False
        if lfs_pointer_blob(4, "0" * 64) != hashlib.sha1(b"blob 126\0version https://git-lfs.github.com/spec/v1\noid sha256:" + b"0" * 64 + b"\nsize 4\n").hexdigest():
            return False
        return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--transformers", type=Path)
    parser.add_argument("--server-tree", type=Path)
    parser.add_argument("--public-contract", type=Path)
    parser.add_argument("--public-gguf", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        passed = self_test()
        print("csm_1b_inspect.py self-test: " + ("OK" if passed else "FAIL"))
        return 0 if passed else 1
    if not all((args.snapshot, args.source, args.transformers, args.server_tree, args.public_contract, args.public_gguf, args.output)):
        parser.error("inspection requires --snapshot --source --transformers --server-tree --public-contract --public-gguf --output")
    try:
        single = inspect_safetensors(args.snapshot / "model.safetensors")
        transformer = inspect_index(args.snapshot)
        public_file = args.public_gguf
        if not public_file.is_file() or public_file.is_symlink() or public_file.stat().st_size != PUBLIC_BYTES or digest(public_file) != PUBLIC_SHA256:
            raise RuntimeError("historical public GGUF payload identity mismatch")
        evidence = {"tree": inspect_tree(args.snapshot, args.server_tree), "single_checkpoint": single, "transformers": transformer, "json_roles": inspect_json_roles(args.snapshot), "legacy_ckpt": inspect_checkpoint(args.snapshot / "ckpt.pt"), "source": inspect_source(args.source, args.transformers), "format_relationship": {"single_manifest_sha256": single["tensor_manifest_sha256"], "transformers_manifest_sha256": transformer["tensor_manifest_sha256"], "single_tensor_names": [item["name"] for item in single["tensors"]], "transformers_tensor_names": sorted({item["name"] for shard in transformer["shards"].values() for item in shard["tensors"]}), "comparison": "recorded independently; no equivalence inferred without reviewed mapping"}, "historical_public_contract": "NOT_PROVIDED"}
        if args.public_contract:
            contract = load_json(args.public_contract)
            if contract.get("repo") != PUBLIC_REPOSITORY or contract.get("revision") != PUBLIC_REVISION or contract.get("filename") != PUBLIC_FILE or contract.get("bytes") not in (None, PUBLIC_BYTES) or contract.get("sha256") not in (None, PUBLIC_SHA256) or contract.get("header_bytes") != 16066 or contract.get("tensor_count") != 187 or contract.get("manifest_sha256") != PUBLIC_MANIFEST_SHA256:
                raise RuntimeError("historical public GGUF contract mismatch")
            metadata = contract.get("metadata")
            if not isinstance(metadata, dict) or metadata.get("vokra.model.arch") != "csm" or metadata.get("vokra.model.name") != "sesame-csm-1b" or metadata.get("vokra.csm.text.vocab_size") != 0 or metadata.get("vokra.csm.audio.vocab_size") != 0 or metadata.get("vokra.csm.audio.n_codebooks") != 32 or metadata.get("vokra.csm.sample_rate") != 24000:
                raise RuntimeError("historical public GGUF metadata is not the known incomplete CSM contract")
            tensors = contract.get("tensors")
            if not isinstance(tensors, list) or len(tensors) != 187 or len({item.get("name") for item in tensors if isinstance(item, dict)}) != 187 or any(not isinstance(item, dict) or set(item) != {"dtype", "name", "offset", "shape"} or not isinstance(item.get("name"), str) or (safe_path(item["name"]) is not None) or not isinstance(item.get("shape"), list) or any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in item["shape"]) or isinstance(item.get("offset"), bool) or not isinstance(item.get("offset"), int) or item["offset"] < 0 or item.get("dtype") != "F32" for item in tensors):
                raise RuntimeError("historical public tensor manifest malformed")
            evidence["historical_public_contract"] = {**inspect_public_gguf(public_file, contract), "repo": PUBLIC_REPOSITORY, "revision": PUBLIC_REVISION, "manifest_sha256": PUBLIC_MANIFEST_SHA256, "tensor_count": len(tensors), "status": "runtime-incomplete legacy / CSM-core-only"}
        manifest(args.output, "AUTHENTICATED_EVIDENCE_COMPLETE", evidence=evidence)
    except Exception as error:
        manifest(args.output, "INSPECTION_ERROR", error=f"{type(error).__name__}: {error}")
        print(f"CSM inspection blocked: {error}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
