#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Inspect the fixed Spark-TTS BiCodec boundary without claiming runtime support.

The inspector authenticates materialized bytes and an independently-produced
Hugging Face server packet before writing evidence. It parses only
safetensors metadata and remains ``INSPECTION_ONLY`` until an audited native
binder and parity run exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import stat
import subprocess
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

UPSTREAM_REPO = "SparkAudio/Spark-TTS-0.5B"
UPSTREAM_REVISION = "642071559bfc6346c2359d19dcb6be3f9dd8a05d"
CHECKPOINT_RELATIVE = Path("BiCodec/model.safetensors")
CONFIG_RELATIVE = Path("BiCodec/config.yaml")
CHECKPOINT_BYTES = 625_518_756
CHECKPOINT_SHA256 = "e9940cd48d4446e4340ced82d234bf5618350dd9f5db900ebe47a4fdb03867ec"
CONFIG_BYTES = 1_164
CONFIG_SHA256 = "744f4093ae2381a2eb44ea8c4a5268a8d1e581498e9bf0808c034d1b076429be"
SOURCE_REPOSITORY = "https://github.com/SparkAudio/Spark-TTS"
SOURCE_REVISION = "2f1ea9082400547242641f5271b6f941c9f439d1"
FORMAT = "vokra-bicodec-inspection-v1"
WEIGHT_LICENSE = "cc-by-nc-sa-4.0"
EVIDENCE_FILES = ("tensor-inventory.json", "config.json", "source-inventory.json", "server-packet.json", "manifest.json")
SOURCE_ROLE_BLOBS = {
    "LICENSE": "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64",
    "README.md": "74f792da85c4bea0ea74b4a186fd4ea8004eedc3",
    "cli/SparkTTS.py": "bc86ce38a7e24704c67d4950914e11278cdd2c55",
    "cli/inference.py": "349f7ea2366d6cb450c11d15e7c80f5f37b7ded8",
    "sparktts/models/audio_tokenizer.py": "d7065eb6c4844ce582cb4afd996ce605cc30129f",
    "sparktts/models/bicodec.py": "8cab2f0247221411f7504e1582df850eb7043409",
    "sparktts/modules/blocks/layers.py": "9de506e9c30bf48efcc00d09da1e210d4058e9a0",
    "sparktts/modules/blocks/samper.py": "e6673bf63e51577fe983d6d52598537eea7c1a95",
    "sparktts/modules/blocks/vocos.py": "31ff7900a8474d48ba794540553b4b7e3e61150c",
    "sparktts/modules/encoder_decoder/feat_decoder.py": "de409789f21691c6a648ce4d0e4cfb2dba1bef33",
    "sparktts/modules/encoder_decoder/feat_encoder.py": "e1f861b10dd26bf6317acc6f60496be4a7c1abc3",
    "sparktts/modules/encoder_decoder/wave_generator.py": "13ca769aa26c2fa8fc7f62c0c0bb27063d34076b",
    "sparktts/modules/fsq/finite_scalar_quantization.py": "da36155078290b5262f9a7b1b7b15b621cc88131",
    "sparktts/modules/fsq/residual_fsq.py": "38a391d8c8b0427e4b21f20d545c7d4f2e367e7c",
    "sparktts/modules/speaker/ecapa_tdnn.py": "18f57ef00bc07cda6e24029f223f2bc11da8adc3",
    "sparktts/modules/speaker/perceiver_encoder.py": "bed1bb128f0bb1591c453480397b6d40d1e19cf3",
    "sparktts/modules/speaker/pooling_layers.py": "fe8289885085c2377069dc57375953a6b87288bd",
    "sparktts/modules/speaker/speaker_encoder.py": "eb5050ad1aa1c13c621cc0501992e3b74f00726b",
    "sparktts/modules/vq/factorized_vector_quantize.py": "f820bc73b559c08e36cbc38de1db0fddd442a727",
    "sparktts/utils/audio.py": "105cd9cd676f3f204bb492b53205eda710d805d0",
    "sparktts/utils/file.py": "bcc7c7ca0713f0e67c6327398bc73990d10413ab",
    "sparktts/utils/token_parser.py": "cc43782b762d6bb51e3c35f3585706750e8d5a22",
}
SELECTED_FILES = {".gitattributes", "README.md", "BiCodec/config.yaml", "BiCodec/model.safetensors"}
MAX_SAFETENSORS_HEADER = 64 * 1024 * 1024
DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "I32": 4, "I64": 8, "F8_E4M3": 1, "F8_E5M2": 1, "F16": 2, "BF16": 2, "F32": 4, "F64": 8, "C64": 8, "C128": 16}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def json_unique(text: str) -> Any:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value
    return json.loads(text, object_pairs_hook=reject)


def regular_files(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*")):
        parts = path.relative_to(root).parts
        if parts == (".cache",) or len(parts) >= 2 and parts[:2] == (".cache", "huggingface"):
            continue
        if any(part in {".cache", ".git"} for part in parts):
            raise RuntimeError(f"unauthenticated cache/metadata path: {path}")
        if path.is_dir() and not path.is_symlink():
            continue
        if path.is_symlink():
            if not path.exists():
                raise RuntimeError(f"dangling payload symlink: {path}")
            raise RuntimeError(f"payload symlink is not accepted: {path}")
        if not path.is_file():
            raise RuntimeError(f"payload member is not regular: {path}")
        files.append(path)
    return files


def validate_transport_cache(root: Path) -> list[str]:
    cache = root / ".cache"
    if not cache.exists():
        return []
    if cache.is_symlink() or not cache.is_dir() or (cache / "huggingface").is_symlink() or not (cache / "huggingface").is_dir():
        raise RuntimeError("only root .cache/huggingface transport metadata is allowed")
    return [".cache/huggingface"] + [p.relative_to(root).as_posix() for p in sorted((cache / "huggingface").rglob("*"))]


def require_exact(path: Path, expected_bytes: int, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    actual_bytes = path.stat().st_size
    actual_sha256 = sha256(path)
    if actual_bytes != expected_bytes or actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{label} identity mismatch: bytes={actual_bytes} sha256={actual_sha256}"
        )


def source_inventory(source: Path) -> dict[str, Any]:
    if not (source / ".git").exists():
        raise RuntimeError("official source checkout lacks .git metadata")
    if git(source, "rev-parse", "HEAD") != SOURCE_REVISION:
        raise RuntimeError("official Spark-TTS source revision mismatch")
    remote = git(source, "remote", "get-url", "origin")
    if remote != SOURCE_REPOSITORY:
        raise RuntimeError(f"official source remote mismatch: {remote}")
    if git(source, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("official source checkout is dirty")
    entries: dict[str, tuple[str, str]] = {}
    for record in git(source, "ls-files", "-s", "-z").split("\0"):
        if not record:
            continue
        metadata, relative = record.split("\t", 1)
        mode, object_id, stage = metadata.split()
        if mode not in {"100644", "100755"} or stage != "0":
            raise RuntimeError(f"source has non-regular/staged tracked entry: {relative}")
        path = source / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"source tracked file missing/non-regular: {relative}")
        expected_filesystem_mode = {"100644": 0o644, "100755": 0o755}[mode]
        if stat.S_IMODE(path.stat().st_mode) != expected_filesystem_mode:
            raise RuntimeError(f"source filesystem mode drift: {relative}")
        if relative in entries:
            raise RuntimeError(f"source index contains duplicate tracked entry: {relative}")
        entries[relative] = (mode, object_id)
    for relative, (_mode, index_object) in entries.items():
        head_object = git(source, "rev-parse", f"HEAD:{relative}")
        working_object = git_blob_sha1(source / relative)
        if index_object != head_object or index_object != working_object:
            raise RuntimeError(f"source tracked object drift: {relative}")
    roles = []
    for relative, expected in SOURCE_ROLE_BLOBS.items():
        mode_object = entries.get(relative)
        if mode_object is None:
            raise RuntimeError(f"source role missing from index: {relative}")
        mode, index_object = mode_object
        if index_object != expected:
            raise RuntimeError(f"source role object mismatch: {relative}")
        if mode != "100644":
            raise RuntimeError(f"fixed source role must be a regular non-executable file: {relative}")
        roles.append({"path": relative, "mode": mode, "git_blob_sha1": expected})
    license_text = (source / "LICENSE").read_text(encoding="utf-8")
    if ("Licensed under the Apache License, Version 2.0" not in license_text
            or "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND" not in license_text):
        raise RuntimeError("source LICENSE lacks complete Apache grant/warranty terms")
    return {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION, "worktree_status": "CLEAN", "roles": roles, "license": {"spdx": "Apache-2.0", "path": "LICENSE", "sha256": sha256(source / "LICENSE")}}


class StrictYamlLoader(yaml.SafeLoader):
    pass


def _mapping(loader: StrictYamlLoader, node: yaml.MappingNode, deep: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise ValueError(f"duplicate/invalid YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


StrictYamlLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def parse_config(text: str) -> dict[str, Any]:
    if re.search(r"(?:^|[\s:])(?:!|&|\*)", text, flags=re.MULTILINE):
        raise ValueError("YAML aliases/tags are not accepted")
    try:
        value = yaml.load(text, Loader=StrictYamlLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"strict YAML parse failed: {error}") from error
    expected: dict[str, Any] = {
        "audio_tokenizer": {
            "mel_params": {"sample_rate": 16000, "n_fft": 1024, "win_length": 640, "hop_length": 320, "mel_fmin": 10, "mel_fmax": None, "num_mels": 128},
            "encoder": {"input_channels": 1024, "vocos_dim": 384, "vocos_intermediate_dim": 2048, "vocos_num_layers": 12, "out_channels": 1024, "sample_ratios": [1, 1]},
            "decoder": {"input_channel": 1024, "channels": 1536, "rates": [8, 5, 4, 2], "kernel_sizes": [16, 11, 8, 4]},
            "quantizer": {"input_dim": 1024, "codebook_size": 8192, "codebook_dim": 8, "commitment": 0.25, "codebook_loss_weight": 2.0, "use_l2_normlize": True, "threshold_ema_dead_code": 0.2},
            "speaker_encoder": {"input_dim": 128, "out_dim": 1024, "latent_dim": 128, "token_num": 32, "fsq_levels": [4, 4, 4, 4, 4, 4], "fsq_num_quantizers": 1},
            "prenet": {"input_channels": 1024, "vocos_dim": 384, "vocos_intermediate_dim": 2048, "vocos_num_layers": 12, "out_channels": 1024, "condition_dim": 1024, "sample_ratios": [1, 1], "use_tanh_at_final": False},
            "postnet": {"input_channels": 1024, "vocos_dim": 384, "vocos_intermediate_dim": 2048, "vocos_num_layers": 6, "out_channels": 1024, "use_tanh_at_final": False},
        }
    }

    def validate_exact(actual: Any, expected_node: Any, path: str) -> None:
        if isinstance(expected_node, dict):
            if not isinstance(actual, dict) or set(actual) != set(expected_node):
                raise ValueError(f"config schema drift at {path}")
            for key, child in expected_node.items():
                validate_exact(actual[key], child, f"{path}.{key}")
            return
        if isinstance(expected_node, list):
            if (not isinstance(actual, list) or len(actual) != len(expected_node)
                    or any(isinstance(item, bool) or not isinstance(item, int) for item in actual)):
                raise ValueError(f"config typed list drift at {path}")
            if actual != expected_node:
                raise ValueError(f"config value drift at {path}: {actual!r}")
            return
        if isinstance(expected_node, bool):
            valid = type(actual) is bool and actual == expected_node
        elif isinstance(expected_node, int):
            valid = type(actual) is int and actual == expected_node
        elif isinstance(expected_node, float):
            valid = type(actual) is float and math.isfinite(actual) and actual == expected_node
        else:
            valid = actual == expected_node
        if not valid:
            raise ValueError(f"config value/type drift at {path}: {actual!r}")

    validate_exact(value, expected, "audio_tokenizer")
    return {"raw": value, "observed_fixed_values": value["audio_tokenizer"]}


def parse_weight_license(path: Path) -> dict[str, str]:
    """Read the model-card declaration from the authenticated README only."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise RuntimeError("HF README lacks a license frontmatter boundary")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise RuntimeError("HF README frontmatter is unterminated")
    try:
        frontmatter = yaml.load(text[4:end], Loader=StrictYamlLoader)
    except yaml.YAMLError as error:
        raise RuntimeError(f"HF README frontmatter is invalid: {error}") from error
    if not isinstance(frontmatter, dict) or frontmatter.get("license") != WEIGHT_LICENSE:
        raise RuntimeError("HF README license declaration is not the fixed CC-BY-NC-SA-4.0 value")
    return {"spdx": WEIGHT_LICENSE, "path": "README.md", "basis": "authenticated HF README frontmatter"}


def safetensors_header(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    import struct
    file_size = path.stat().st_size
    if file_size < 8:
        raise RuntimeError("safetensors file lacks header length")
    with path.open("rb") as stream:
        header_length = struct.unpack("<Q", stream.read(8))[0]
        if header_length <= 0 or header_length > MAX_SAFETENSORS_HEADER or header_length > file_size - 8:
            raise RuntimeError("safetensors header length is invalid or exceeds bound")
        header = json_unique(stream.read(header_length).decode("utf-8"))
    if not isinstance(header, dict) or ("__metadata__" in header and not isinstance(header["__metadata__"], dict)):
        raise RuntimeError("safetensors header schema is invalid")
    payload_size = file_size - 8 - header_length
    ranges: list[tuple[int, int, str, dict[str, Any]]] = []
    for name, item in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(item, dict) or set(item) != {"dtype", "shape", "data_offsets"}:
            raise RuntimeError(f"safetensors tensor header is invalid: {name!r}")
        dtype, shape, offsets = item["dtype"], item["shape"], item["data_offsets"]
        if dtype not in DTYPE_BYTES or not isinstance(shape, list) or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in shape) or not isinstance(offsets, list) or len(offsets) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in offsets):
            raise RuntimeError(f"safetensors tensor metadata is invalid: {name}")
        start, end = offsets
        elements = 1
        for dimension in shape:
            elements *= dimension
        if end < start or elements * DTYPE_BYTES[dtype] != end - start or end > payload_size:
            raise RuntimeError(f"safetensors tensor offsets/shape arithmetic is invalid: {name}")
        ranges.append((start, end, name, {"name": name, "shape": shape, "dtype": dtype, "elements": elements, "bytes": end - start, "data_offsets": [start, end]}))
    ranges.sort(key=lambda row: (row[0], row[1], row[2]))
    cursor = 0
    tensors = []
    counts: dict[str, int] = {}
    for start, end, _name, record in ranges:
        if start != cursor:
            raise RuntimeError("safetensors tensor ranges contain a gap or overlap")
        cursor = end
        tensors.append(record)
        counts[record["dtype"]] = counts.get(record["dtype"], 0) + 1
    if cursor != payload_size or not tensors:
        raise RuntimeError("safetensors tensor ranges do not cover the payload")
    return tensors, counts


def validate_server_packet(model_dir: Path, packet_path: Path) -> dict[str, Any]:
    packet = json_unique(packet_path.read_text(encoding="utf-8"))
    if not isinstance(packet, dict) or set(packet) != {"repository", "requested_revision", "resolved_revision", "files"}:
        raise RuntimeError("server packet schema mismatch")
    if packet["repository"] != UPSTREAM_REPO or packet["requested_revision"] != UPSTREAM_REVISION or packet["resolved_revision"] != UPSTREAM_REVISION:
        raise RuntimeError("server packet revision/repository mismatch")
    rows = packet["files"]
    if not isinstance(rows, list):
        raise RuntimeError("server packet files must be a list")
    if [row.get("path") for row in rows if isinstance(row, dict)] != sorted(
        row.get("path") for row in rows if isinstance(row, dict)
    ):
        raise RuntimeError("server packet file rows are not in canonical path order")
    local_files = regular_files(model_dir)
    local = {p.relative_to(model_dir).as_posix(): p for p in local_files}
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("type") != "file":
            raise RuntimeError("server packet contains non-file row")
        path = row.get("path")
        if (not isinstance(path, str) or not path or Path(path).is_absolute()
                or "\\" in path or any(part in {"", ".", ".."} for part in Path(path).parts)
                or path in seen or path not in local
                or set(row) not in ({"path", "type", "size", "git_blob_sha1"}, {"path", "type", "size", "git_blob_sha1", "lfs_sha256", "lfs_size", "lfs_pointer_sha1"})):
            raise RuntimeError("server packet file row is duplicate, unknown, or incomplete")
        seen.add(path)
        payload = local[path]
        if isinstance(row["size"], bool) or not isinstance(row["size"], int) or row["size"] != payload.stat().st_size:
            raise RuntimeError(f"server packet size mismatch: {path}")
        observed_sha = sha256(payload)
        if set(row) == {"path", "type", "size", "git_blob_sha1"}:
            if (not isinstance(row["git_blob_sha1"], str) or len(row["git_blob_sha1"]) != 40
                    or any(c not in "0123456789abcdef" for c in row["git_blob_sha1"])
                    or row["git_blob_sha1"] != git_blob_sha1(payload)):
                raise RuntimeError(f"server packet Git blob mismatch: {path}")
        else:
            if (not isinstance(row["lfs_sha256"], str) or len(row["lfs_sha256"]) != 64
                    or any(c not in "0123456789abcdef" for c in row["lfs_sha256"])
                    or isinstance(row["lfs_size"], bool) or not isinstance(row["lfs_size"], int)
                    or not isinstance(row["lfs_pointer_sha1"], str) or len(row["lfs_pointer_sha1"]) != 40
                    or any(c not in "0123456789abcdef" for c in row["lfs_pointer_sha1"])
                    or row["lfs_sha256"] != observed_sha or row["lfs_size"] != payload.stat().st_size):
                raise RuntimeError(f"server packet LFS payload mismatch: {path}")
            pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{observed_sha}\nsize {payload.stat().st_size}\n".encode()
            pointer_sha = hashlib.sha1(f"blob {len(pointer)}\0".encode() + pointer).hexdigest()
            if row["lfs_pointer_sha1"] != pointer_sha or row["git_blob_sha1"] != pointer_sha:
                raise RuntimeError(f"server packet LFS pointer mismatch: {path}")
    if seen != set(SELECTED_FILES):
        raise RuntimeError(f"selected server/local tree mismatch: missing={sorted(set(SELECTED_FILES)-seen)!r} extra={sorted(seen-set(SELECTED_FILES))!r}")
    return {"repository": packet["repository"], "requested_revision": packet["requested_revision"], "resolved_revision": packet["resolved_revision"], "file_count": len(seen), "transport_cache": validate_transport_cache(model_dir)}


def inspect(model_dir: Path, source_dir: Path, server_packet: Path, output_dir: Path) -> None:
    model_dir = model_dir.resolve()
    source_dir = source_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / CHECKPOINT_RELATIVE
    config_path = model_dir / CONFIG_RELATIVE
    server = validate_server_packet(model_dir, server_packet)
    if {p.relative_to(model_dir).as_posix() for p in regular_files(model_dir)} != SELECTED_FILES:
        raise RuntimeError("selected model tree contains missing/extra files")
    require_exact(checkpoint, CHECKPOINT_BYTES, CHECKPOINT_SHA256, "BiCodec checkpoint")
    require_exact(config_path, CONFIG_BYTES, CONFIG_SHA256, "BiCodec config")
    source_evidence = source_inventory(source_dir)
    weight_license = parse_weight_license(model_dir / "README.md")
    config = parse_config(config_path.read_text(encoding="utf-8"))
    tensors, dtype_counts = safetensors_header(checkpoint)
    if not tensors:
        raise RuntimeError("checkpoint has no tensors")

    (output_dir / "tensor-inventory.json").write_text(
        json.dumps({"tensor_count": len(tensors), "dtype_counts": dtype_counts, "tensors": tensors}, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(config, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "source-inventory.json").write_text(
        json.dumps(source_evidence, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "server-packet.json").write_bytes(server_packet.read_bytes())
    packets = {
        name: {"bytes": (output_dir / name).stat().st_size, "sha256": sha256(output_dir / name)}
        for name in ("tensor-inventory.json", "config.json", "source-inventory.json", "server-packet.json")
    }
    manifest = {
        "format": FORMAT,
        "status": "BLOCKED",
        "evidence_stage": "INSPECTION_ONLY",
        "collection_status": "AUTHENTICATED",
        "inspection_status": "COMPLETE",
        "runtime_status": "NOT_IMPLEMENTED",
        "cpu_status": "UNSUPPORTED",
        "metal_status": "BLOCKED_BY_CPU",
        "numerical_status": "NOT_RUN",
        "publication": "NO_UPLOAD",
        "reason": "official encode/decode intermediates are not claimed until the native BiCodec binder is audited",
        "upstream": {"repository": UPSTREAM_REPO, "revision": UPSTREAM_REVISION, "checkpoint": CHECKPOINT_RELATIVE.as_posix(), "checkpoint_bytes": CHECKPOINT_BYTES, "checkpoint_sha256": CHECKPOINT_SHA256, "config": CONFIG_RELATIVE.as_posix(), "config_bytes": CONFIG_BYTES, "config_sha256": CONFIG_SHA256},
        "official_source": source_evidence,
        "server_tree": server,
        "tensor_count": len(tensors),
        "dtype_counts": dtype_counts,
        # parse_config returns a wrapper so the raw authenticated YAML and
        # derived observations cannot be confused with one another.
        "config_topology": sorted(config["raw"]["audio_tokenizer"].keys()),
        "packets": packets,
        "weight_license": {**weight_license, "policy": "RESEARCH_ONLY_SHARE_ALIKE_NO_UPLOAD"},
        "blockers": ["native/runtime implementation is not available", "numerical parity is NOT_RUN", "publication is NO_UPLOAD"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def write_error_manifest(output_dir: Path, error: Exception) -> None:
    """Leave an explicit non-authenticated blocker instead of stale evidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in EVIDENCE_FILES:
        path = output_dir / name
        if path.exists() and path.is_file():
            path.unlink()
    manifest = {
        "format": FORMAT,
        "status": "BLOCKED",
        "evidence_stage": "INSPECTION_ONLY",
        "collection_status": "FAILED",
        "inspection_status": "ERROR",
        "runtime_status": "NOT_IMPLEMENTED",
        "cpu_status": "UNSUPPORTED",
        "metal_status": "BLOCKED_BY_CPU",
        "numerical_status": "NOT_RUN",
        "publication": "NO_UPLOAD",
        "error": str(error),
        "reason": "authentication failed; no model or source identity is asserted",
        "blockers": ["authenticated collection unavailable", "native/runtime implementation is not available", "numerical parity is NOT_RUN"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def self_test() -> None:
    global SOURCE_REVISION, SOURCE_ROLE_BLOBS
    assert hashlib.sha256(b"abc").hexdigest() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert FORMAT == "vokra-bicodec-inspection-v1"
    assert "INSPECTION_ONLY" in inspect.__code__.co_consts
    source = Path(__file__).read_text(encoding="utf-8")
    assert "safetensors_header" in source and "StrictYamlLoader" in source
    assert "collection_status" in source and '"status": "BLOCKED"' in source
    try:
        json_unique('{"x": 1, "x": 2}')
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate JSON key accepted")
    try:
        parse_config("audio_tokenizer: &anchor {}\n")
    except ValueError:
        pass
    else:
        raise AssertionError("YAML anchor accepted")
    try:
        parse_config("audio_tokenizer: {}\naudio_tokenizer: {}\n")
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate YAML key accepted")
    try:
        parse_config("audio_tokenizer:\n  mel_params: {}\n  encoder: {}\n  decoder: {}\n  quantizer: {}\n  speaker_encoder: {}\n  prenet: {}\n  postnet: {}\n  extra: {}\n")
    except ValueError:
        pass
    else:
        raise AssertionError("extra config key accepted")
    valid_config = {"audio_tokenizer": {
        "mel_params": {"sample_rate": 16000, "n_fft": 1024, "win_length": 640, "hop_length": 320, "mel_fmin": 10, "mel_fmax": None, "num_mels": 128},
        "encoder": {"input_channels": 1024, "vocos_dim": 384, "vocos_intermediate_dim": 2048, "vocos_num_layers": 12, "out_channels": 1024, "sample_ratios": [1, 1]},
        "decoder": {"input_channel": 1024, "channels": 1536, "rates": [8, 5, 4, 2], "kernel_sizes": [16, 11, 8, 4]},
        "quantizer": {"input_dim": 1024, "codebook_size": 8192, "codebook_dim": 8, "commitment": 0.25, "codebook_loss_weight": 2.0, "use_l2_normlize": True, "threshold_ema_dead_code": 0.2},
        "speaker_encoder": {"input_dim": 128, "out_dim": 1024, "latent_dim": 128, "token_num": 32, "fsq_levels": [4, 4, 4, 4, 4, 4], "fsq_num_quantizers": 1},
        "prenet": {"input_channels": 1024, "vocos_dim": 384, "vocos_intermediate_dim": 2048, "vocos_num_layers": 12, "out_channels": 1024, "condition_dim": 1024, "sample_ratios": [1, 1], "use_tanh_at_final": False},
        "postnet": {"input_channels": 1024, "vocos_dim": 384, "vocos_intermediate_dim": 2048, "vocos_num_layers": 6, "out_channels": 1024, "use_tanh_at_final": False},
    }}
    canonical_yaml = yaml.safe_dump(valid_config, sort_keys=False)
    parsed_config = parse_config(canonical_yaml)
    assert set(parsed_config) == {"raw", "observed_fixed_values"}
    assert parsed_config["raw"]["audio_tokenizer"] == parsed_config["observed_fixed_values"]
    missing = json.loads(json.dumps(valid_config))
    del missing["audio_tokenizer"]["decoder"]
    try:
        parse_config(yaml.safe_dump(missing, sort_keys=False))
    except ValueError:
        pass
    else:
        raise AssertionError("missing canonical component accepted")
    extra = json.loads(json.dumps(valid_config))
    extra["audio_tokenizer"]["decoder"]["extra"] = 1
    try:
        parse_config(yaml.safe_dump(extra, sort_keys=False))
    except ValueError:
        pass
    else:
        raise AssertionError("extra canonical field accepted")
    misnested = json.loads(json.dumps(valid_config))
    misnested["audio_tokenizer"]["encoder"]["decoder"] = misnested["audio_tokenizer"].pop("decoder")
    try:
        parse_config(yaml.safe_dump(misnested, sort_keys=False))
    except ValueError:
        pass
    else:
        raise AssertionError("misnested canonical component accepted")
    for drift in (
        "sample_rate: true",
        "sample_rate: 16001",
        "sample_rate: .nan",
    ):
        broken = yaml.safe_dump(valid_config, sort_keys=False).replace("sample_rate: 16000", drift)
        try:
            parse_config(broken)
        except ValueError:
            pass
        else:
            raise AssertionError(f"config drift accepted: {drift}")

    with tempfile.TemporaryDirectory(prefix="vokra-bicodec-error-") as temporary:
        error_dir = Path(temporary)
        (error_dir / "tensor-inventory.json").write_text("stale", encoding="utf-8")
        write_error_manifest(error_dir, RuntimeError("spoofed boundary"))
        error_manifest = json_unique((error_dir / "manifest.json").read_text(encoding="utf-8"))
        assert error_manifest["status"] == "BLOCKED"
        assert error_manifest["collection_status"] == "FAILED"
        assert error_manifest["publication"] == "NO_UPLOAD"
        assert not (error_dir / "tensor-inventory.json").exists()

    # Exercise source_inventory through its real Git/index/worktree checks,
    # including a tracked executable that is not one of the fixed roles.
    old_source_revision = SOURCE_REVISION
    old_source_roles = SOURCE_ROLE_BLOBS
    with tempfile.TemporaryDirectory(prefix="vokra-bicodec-source-") as temporary:
        source = Path(temporary) / "source"
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "remote", "add", "origin", SOURCE_REPOSITORY], check=True)
        subprocess.run(["git", "-C", str(source), "config", "user.email", "selftest@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(source), "config", "user.name", "BiCodec self-test"], check=True)
        for index, relative in enumerate(old_source_roles):
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            content = b"Licensed under the Apache License, Version 2.0\nWITHOUT WARRANTIES OR CONDITIONS OF ANY KIND\n" if relative == "LICENSE" else f"role-{index}\n".encode()
            target.write_bytes(content)
        executable = source / "selftest-tool.sh"
        executable.write_bytes(b"#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)
        subprocess.run(["git", "-C", str(source), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(source), "commit", "-qm", "fixture"], check=True)
        try:
            SOURCE_REVISION = git(source, "rev-parse", "HEAD")
            SOURCE_ROLE_BLOBS = {relative: git_blob_sha1(source / relative) for relative in old_source_roles}
            inventory = source_inventory(source)
            assert inventory["worktree_status"] == "CLEAN"
            assert all(role["mode"] == "100644" for role in inventory["roles"])
            executable.chmod(0o644)
            subprocess.run(["git", "-C", str(source), "update-index", "--assume-unchanged", "selftest-tool.sh"], check=True)
            try:
                source_inventory(source)
            except RuntimeError:
                pass
            else:
                raise AssertionError("tracked filesystem-mode drift accepted")
        finally:
            SOURCE_REVISION = old_source_revision
            SOURCE_ROLE_BLOBS = old_source_roles

    def write_st(path: Path, header: str, payload: bytes = b"\0\0\0\0") -> None:
        path.write_bytes(struct.pack("<Q", len(header.encode())) + header.encode() + payload)

    with tempfile.TemporaryDirectory(prefix="vokra-bicodec-selftest-") as temporary:
        root = Path(temporary)
        valid_header = json.dumps({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}})
        valid_path = root / "valid.safetensors"
        write_st(valid_path, valid_header)
        tensors, counts = safetensors_header(valid_path)
        assert tensors[0]["name"] == "x" and counts == {"F32": 1}
        malformed = {
            "duplicate": '{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}',
            "gap": '{"x":{"dtype":"F32","shape":[1],"data_offsets":[1,5]}}',
            "bad_dtype": '{"x":{"dtype":"STRING","shape":[1],"data_offsets":[0,4]}}',
            "bad_shape": '{"x":{"dtype":"F32","shape":[true],"data_offsets":[0,4]}}',
            "out_of_bounds": '{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}',
        }
        for name, header in malformed.items():
            path = root / f"{name}.safetensors"
            write_st(path, header)
            try:
                safetensors_header(path)
            except (RuntimeError, ValueError, UnicodeError):
                pass
            else:
                raise AssertionError(f"malformed safetensors accepted: {name}")
        oversized = root / "oversized.safetensors"
        oversized.write_bytes(struct.pack("<Q", MAX_SAFETENSORS_HEADER + 1) + b"{}")
        try:
            safetensors_header(oversized)
        except RuntimeError:
            pass
        else:
            raise AssertionError("oversized safetensors header accepted")

        model = root / "model"
        (model / "BiCodec").mkdir(parents=True)
        for relative, content in (
            (".gitattributes", b"*.safetensors filter=lfs"),
            ("README.md", b"---\nlicense: cc-by-nc-sa-4.0\n---\n"),
            ("BiCodec/config.yaml", b"config"),
            ("BiCodec/model.safetensors", b"weights"),
        ):
            target = model / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        rows = []
        for relative in sorted(SELECTED_FILES):
            payload = model / relative
            rows.append({"path": relative, "type": "file", "size": payload.stat().st_size, "git_blob_sha1": git_blob_sha1(payload)})
        packet = root / "packet.json"
        packet.write_text(json.dumps({"repository": UPSTREAM_REPO, "requested_revision": UPSTREAM_REVISION, "resolved_revision": UPSTREAM_REVISION, "files": rows}), encoding="utf-8")
        validate_server_packet(model, packet)
        packet_value = json.loads(packet.read_text(encoding="utf-8"))
        lfs_row = next(row for row in packet_value["files"] if row["path"] == "BiCodec/model.safetensors")
        lfs_digest = sha256(model / "BiCodec/model.safetensors")
        lfs_size = (model / "BiCodec/model.safetensors").stat().st_size
        pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{lfs_digest}\nsize {lfs_size}\n".encode()
        pointer_sha = hashlib.sha1(f"blob {len(pointer)}\0".encode() + pointer).hexdigest()
        lfs_row.update(lfs_sha256=lfs_digest, lfs_size=lfs_size, lfs_pointer_sha1=pointer_sha, git_blob_sha1=pointer_sha)
        packet.write_text(json.dumps(packet_value), encoding="utf-8")
        validate_server_packet(model, packet)
        lfs_row["lfs_sha256"] = "0" * 64
        packet.write_text(json.dumps(packet_value), encoding="utf-8")
        try:
            validate_server_packet(model, packet)
        except RuntimeError:
            pass
        else:
            raise AssertionError("LFS payload spoof accepted")
        spoof = json.loads(packet.read_text(encoding="utf-8"))
        spoof["resolved_revision"] = "0" * 40
        packet.write_text(json.dumps(spoof), encoding="utf-8")
        try:
            validate_server_packet(model, packet)
        except RuntimeError:
            pass
        else:
            raise AssertionError("server revision spoof accepted")
        assert parse_weight_license(model / "README.md")["spdx"] == WEIGHT_LICENSE
        packet.write_text(json.dumps({"repository": UPSTREAM_REPO, "requested_revision": UPSTREAM_REVISION, "resolved_revision": UPSTREAM_REVISION, "files": rows}), encoding="utf-8")
        baseline_packet = {"repository": UPSTREAM_REPO, "requested_revision": UPSTREAM_REVISION, "resolved_revision": UPSTREAM_REVISION, "files": rows}
        for mutation in (
            lambda value: value["files"].pop(),
            lambda value: value["files"].append({"path": "extra.bin", "type": "file", "size": 0, "git_blob_sha1": "0" * 40}),
            lambda value: value["files"][0].update(path="../README.md"),
        ):
            mutated = json.loads(json.dumps(baseline_packet))
            mutation(mutated)
            packet.write_text(json.dumps(mutated), encoding="utf-8")
            try:
                validate_server_packet(model, packet)
            except RuntimeError:
                pass
            else:
                raise AssertionError("server tree shape/path spoof accepted")
        packet.write_text(json.dumps(baseline_packet), encoding="utf-8")
        payload_link = model / "payload-link"
        payload_link.symlink_to(model / "README.md")
        try:
            validate_server_packet(model, packet)
        except RuntimeError:
            pass
        else:
            raise AssertionError("payload symlink accepted")
        payload_link.unlink()
        (model / ".cache").mkdir()
        cache_spoof = model / ".cache" / "spoof"
        cache_spoof.write_text("not a license", encoding="utf-8")
        try:
            parse_weight_license(cache_spoof)
        except RuntimeError:
            pass
        else:
            raise AssertionError("cache license candidate accepted")
        cache_spoof.unlink()
        (model / ".cache" / "huggingface").mkdir()
        (model / ".cache" / "huggingface" / "metadata.json").write_text("{}", encoding="utf-8")
        validate_server_packet(model, packet)
        nested_cache = model / "BiCodec" / ".cache"
        nested_cache.mkdir()
        (nested_cache / "spoof").write_text("x", encoding="utf-8")
        try:
            validate_server_packet(model, packet)
        except RuntimeError:
            pass
        else:
            raise AssertionError("nested cache accepted")
        (nested_cache / "spoof").unlink()
        nested_cache.rmdir()
    print("bicodec_inspect_reference.py self-test: OK (authenticated packet/header/config contracts)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--server-packet", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.model_dir, args.source_dir, args.server_packet, args.output)):
            parser.error("--self-test accepts no other arguments")
        self_test()
        return 0
    if not args.model_dir or not args.source_dir or not args.server_packet or not args.output:
        parser.error("--model-dir, --source-dir, --server-packet, and --output are required")
    try:
        inspect(args.model_dir, args.source_dir, args.server_packet, args.output)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        write_error_manifest(args.output, error)
        print(f"bicodec inspection: {error}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"bicodec inspection: {error}", file=sys.stderr)
        raise SystemExit(2)
