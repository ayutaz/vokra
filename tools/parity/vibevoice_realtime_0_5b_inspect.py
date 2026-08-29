#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""VAST-only structural inspector for microsoft/VibeVoice-Realtime-0.5B.

This records authenticated snapshot/source evidence only.  It never loads a
model tensor body, runs the upstream pipeline, converts weights, or claims
native/CPU/Metal parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from safetensors import safe_open

HF_REPOSITORY = "microsoft/VibeVoice-Realtime-0.5B"
HF_REVISION = "6bce5f06044837fe6d2c5d7a71a84f0416bd57e4"
SOURCE_REPOSITORY = "https://github.com/microsoft/VibeVoice.git"
SOURCE_REVISION = "94da20d98b2fa7688e9cbfaf7692ddb4954f7600"
TRANSFORMERS_REPOSITORY = "https://github.com/huggingface/transformers.git"
TRANSFORMERS_TAG = "v4.51.3"
TRANSFORMERS_REVISION = "5f4ecf2d9f867a1255131d2461d75793c0cf1db2"
TOKENIZER_REPOSITORY = "Qwen/Qwen2.5-0.5B"
TOKENIZER_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
FORMAT = "vokra-vibevoice-realtime-0.5b-inspection-v1"
MAX_HEADER_BYTES = 64 * 1024 * 1024
TOKENIZER_SELECTED = {"LICENSE", "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"}
MODEL_FILES = {
    ".gitattributes": (1_572, "e685d20cb7927ac8016dadb2514ec1221b1c2a8f", None),
    "README.md": (10_160, "8c2ea6fc74deb70c8d6164d06a12e584498b4379", None),
    "config.json": (2_117, "742245da1dbf39a3d2f9d00899b3a7cd3d3dd692", None),
    "figures/Fig1.png": (123_543, "9227a8b5e4b4c9dd0bbeaf57781f33ce768153d1", "0386a7f577a66324c2b07cf3dff573bc805ce8687c8d6f8b5f3d6d04aed51250"),
    "model.safetensors": (2_035_332_888, "085a1c56d99659990d32b5aa6ad5248530138522", "7758b150b8139deb48ac1ff6f181f745c8fedd5511232fd974b3eb217d83b514"),
    "preprocessor_config.json": (360, "a9e96de2e59454e3896969a8c6d04a52a63c7e17", None),
}
PARAMETERS = 1_017_626_724
AUTHENTICATED_HEADER_BYTES = 79_432
AUTHENTICATED_PAYLOAD_BYTES = 2_035_253_448
QWEN2_ROLE_FILES = (
    "src/transformers/models/qwen2/configuration_qwen2.py",
    "src/transformers/models/qwen2/modeling_qwen2.py",
    "src/transformers/models/qwen2/tokenization_qwen2.py",
)
SOURCE_ROLE_BLOBS = {
    "LICENSE": "269a8973689dbb250d355f516f8a30c1cc66b8e4",
    "pyproject.toml": "ea0f8370ee781f4552d9ab746cb920033d58bafc",
    "vibevoice/modular/configuration_vibevoice.py": "18451136e5c650345e89f4df052731c7322faab9",
    "vibevoice/modular/configuration_vibevoice_streaming.py": "2bd9d6e273c7b79e7483066efa9f2697e335d3fb",
    "vibevoice/modular/modeling_vibevoice.py": "a4ecbab8da413517b25af83dd921d60d3b056bf6",
    "vibevoice/modular/modeling_vibevoice_streaming.py": "c4488c8850dc210cba677bfc61c3c4a654b6c2a5",
    "vibevoice/modular/modeling_vibevoice_streaming_inference.py": "70a489582b88105998281209866d919e738dfc0a",
    "vibevoice/modular/modular_vibevoice_diffusion_head.py": "59de50fb2fe80d6b1ba5a50c9de1ef9cffc4f614",
    "vibevoice/modular/modular_vibevoice_tokenizer.py": "454f9c13094ae42b186ed49e22227cea18189ee1",
    "vibevoice/modular/streamer.py": "5dd7892aed2a416b2eff670c93bc137b3fc216aa",
    "vibevoice/processor/vibevoice_streaming_processor.py": "39c262b1b9859a396b9ea133bf62d782eae1b361",
    "vibevoice/processor/vibevoice_tokenizer_processor.py": "67f61a62f7bd43df46ebc1d8d533fe0dd01adc02",
    "vibevoice/processor/audio_utils.py": "3f9d112cd4fe7dbd84703776cfbdfbc4ee5cce0d",
    "vibevoice/schedule/dpm_solver.py": "b392a480faef86a9e3518fc2c44815ff4dd17171",
    "vibevoice/schedule/timestep_sampler.py": "177b66fcc77da055bbdf7c883be4a38dae699b99",
    "demo/realtime_model_inference_from_file.py": "2a2e711c2d9020790e702993eb5c55ad3e6f6a04",
}
TRANSFORMERS_ROLE_BLOBS = {
    "LICENSE": "68b7d66c97d66c58de883ed0c451af2b3183e6f3",
    "src/transformers/models/qwen2/configuration_qwen2.py": "2e82f1976f3922f3620415f4eace6c6e046243f8",
    "src/transformers/models/qwen2/modeling_qwen2.py": "16a7316e2d0e56eafe301a7f2d8693d6cc6c73ec",
    "src/transformers/models/qwen2/tokenization_qwen2.py": "be2685430f649eab8bde99f217597afd282337c5",
}
SOURCE_ROLE_FILES = tuple(SOURCE_ROLE_BLOBS)
QWEN2_ROLE_FILES = tuple(TRANSFORMERS_ROLE_BLOBS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1_bytes(data: bytes) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {len(data)}\0".encode())
    digest.update(data)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
        raise RuntimeError(f"strict JSON failure at {path}: {error}") from error


def safe_relative(value: str, label: str) -> None:
    path = Path(value)
    if not value or "\x00" in value or "\\" in value or path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"unsafe {label} path: {value!r}")


def validate_symlink_target(name: str, target: bytes) -> None:
    """Validate a Git symlink target lexically without resolving it."""
    if not target or b"\x00" in target or target.startswith(b"/"):
        raise RuntimeError(f"unsafe tracked symlink target: {name}")
    stack = list(Path(name).parts[:-1])
    for component in target.split(b"/"):
        if component in (b"", b"."):
            continue
        if component == b"..":
            if not stack:
                raise RuntimeError(f"tracked symlink target escapes checkout: {name}")
            stack.pop()
        else:
            stack.append(component)


def filesystem_entry(root: Path, name: str) -> tuple[Path, os.stat_result]:
    """lstat a tracked path while refusing symlink traversal in its parents."""
    safe_relative(name, "tracked entry")
    path = root / name
    current = root
    parts = Path(name).parts
    for index, component in enumerate(parts):
        current /= component
        info = os.lstat(current)
        if index < len(parts) - 1 and stat.S_ISLNK(info.st_mode):
            raise RuntimeError(f"tracked path traverses symlink parent: {name}")
    return path, info


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT).strip()


def server_inventory(root: Path, packet: Path, expected_names: set[str] | None = None) -> tuple[dict[str, str], list[dict[str, Any]]]:
    envelope = load_json(packet)
    expected_revision = {HF_REPOSITORY: HF_REVISION, TOKENIZER_REPOSITORY: TOKENIZER_REVISION}.get(envelope.get("repository") if isinstance(envelope, dict) else None)
    if not isinstance(envelope, dict) or set(envelope) != {"repository", "requested_revision", "resolved_revision", "walk", "files"}:
        raise RuntimeError("server-tree envelope schema mismatch")
    if expected_revision is None or envelope["requested_revision"] != expected_revision or envelope["resolved_revision"] != expected_revision or envelope["walk"] != "recursive_file_only":
        raise RuntimeError("server-tree repository/revision/walk envelope mismatch")
    expected = envelope["files"]
    if not isinstance(expected, list):
        raise RuntimeError("server-tree files must be a list")
    actual: set[str] = set()
    transport = Path(".cache/huggingface")
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        parts = relative.parts
        if relative == transport:
            if path.is_symlink() or not path.is_dir():
                raise RuntimeError(f"transport cache must be a real directory: {path}")
            continue
        if transport in relative.parents:
            continue
        if relative == Path(".cache") and (root / transport).is_dir() and not path.is_symlink():
            continue
        if ".cache" in parts:
            raise RuntimeError(f"unexpected cache path outside exact {transport}: {path}")
        if path.is_symlink():
            raise RuntimeError(f"snapshot payload symlink is not authenticated: {path}")
        if path.is_file():
            actual.add(relative.as_posix())
        elif not path.is_dir():
            raise RuntimeError(f"snapshot non-regular member: {path}")
    records: list[dict[str, Any]] = []
    names: set[str] = set()
    for item in expected:
        if not isinstance(item, dict) or set(item) != {"path", "type", "size", "git_blob_sha1", "lfs_pointer_git_blob_sha1", "lfs_sha256"}:
            raise RuntimeError("server-tree item has invalid identity fields")
        name, kind, size = item["path"], item["type"], item["size"]
        git_id, pointer_id, lfs_id = item["git_blob_sha1"], item["lfs_pointer_git_blob_sha1"], item["lfs_sha256"]
        if kind != "file" or not isinstance(name, str) or not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise RuntimeError("server-tree item has invalid type/path/size")
        safe_relative(name, "server-tree")
        if ".cache" in Path(name).parts or name in names:
            raise RuntimeError(f"invalid/duplicate server path: {name}")
        if not isinstance(git_id, str) and git_id is not None:
            raise RuntimeError(f"invalid regular Git blob SHA-1: {name}")
        if not isinstance(pointer_id, str) and pointer_id is not None:
            raise RuntimeError(f"invalid LFS pointer Git blob SHA-1: {name}")
        if not isinstance(lfs_id, str) and lfs_id is not None:
            raise RuntimeError(f"invalid LFS payload SHA-256: {name}")
        for value, length, label in ((git_id, 40, "Git blob SHA-1"), (pointer_id, 40, "LFS pointer Git blob SHA-1"), (lfs_id, 64, "LFS payload SHA-256")):
            if value is not None and (len(value) != length or any(c not in "0123456789abcdef" for c in value)):
                raise RuntimeError(f"invalid {label}: {name}")
        is_lfs = lfs_id is not None
        if is_lfs != (git_id is None and pointer_id is not None):
            raise RuntimeError(f"regular/LFS identity fields are not separated: {name}")
        if not is_lfs and (git_id is None or pointer_id is not None):
            raise RuntimeError(f"regular file identity fields are invalid: {name}")
        file = root / name
        if file.is_symlink() or not file.is_file() or file.stat().st_size != size:
            raise RuntimeError(f"server/local path or size mismatch: {name}")
        local_sha = sha256(file)
        if is_lfs:
            pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{local_sha}\nsize {size}\n".encode()
            if local_sha != lfs_id or git_blob_sha1_bytes(pointer) != pointer_id:
                raise RuntimeError(f"server/local LFS identity mismatch: {name}")
        elif git_blob_sha1(file) != git_id:
            raise RuntimeError(f"server/local Git blob mismatch: {name}")
        names.add(name)
        records.append({"path": name, "bytes": size, "sha256": local_sha, "git_blob_sha1": git_id, "lfs_pointer_git_blob_sha1": pointer_id, "lfs_sha256": lfs_id})
    if actual != names or (expected_names is not None and names != expected_names):
        raise RuntimeError(f"server/local file set mismatch: missing={sorted(names - actual)} extra={sorted(actual - names)}")
    return {"repository": envelope["repository"], "requested_revision": envelope["requested_revision"], "resolved_revision": envelope["resolved_revision"], "walk": envelope["walk"]}, sorted(records, key=lambda row: row["path"])


def require(document: Any, path: tuple[str, ...], value: Any) -> None:
    current = document
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise RuntimeError(f"missing exact config path: {'.'.join(path)}")
        current = current[key]
    if current != value:
        raise RuntimeError(f"config mismatch at {'.'.join(path)}: {current!r} != {value!r}")


def config_fixture() -> dict[str, Any]:
    acoustic = {"causal": True, "channels": 1, "conv_bias": True, "conv_norm": "none", "corpus_normalize": 0.0, "decoder_depths": None, "decoder_n_filters": 32, "decoder_ratios": [8, 5, 5, 4, 2, 2], "disable_last_norm": True, "encoder_depths": "3-3-3-3-3-3-8", "encoder_n_filters": 32, "encoder_ratios": [8, 5, 5, 4, 2, 2], "fix_std": 0.5, "layer_scale_init_value": 1e-6, "layernorm": "RMSNorm", "layernorm_elementwise_affine": True, "layernorm_eps": 1e-5, "mixer_layer": "depthwise_conv", "model_type": "vibevoice_acoustic_tokenizer", "pad_mode": "constant", "std_dist_type": "gaussian", "vae_dim": 64, "weight_init_value": 0.01}
    return {"acoustic_vae_dim": 64, "architectures": ["VibeVoiceStreamingForConditionalGenerationInference"], "model_type": "vibevoice_streaming", "tts_backbone_num_hidden_layers": 20, "transformers_version": "4.51.3", "torch_dtype": "bfloat16", "acoustic_tokenizer_config": acoustic, "decoder_config": {"model_type": "qwen2", "hidden_size": 896, "intermediate_size": 4864, "max_position_embeddings": 8192, "num_hidden_layers": 24, "num_attention_heads": 14, "num_key_value_heads": 2, "vocab_size": 151936, "rope_theta": 1e6, "rms_norm_eps": 1e-6, "tie_word_embeddings": False, "max_window_layers": 24, "sliding_window": None, "rope_scaling": None, "use_cache": True}, "diffusion_head_config": {"model_type": "vibevoice_diffusion_head", "ddpm_beta_schedule": "cosine", "ddpm_num_steps": 1000, "ddpm_num_inference_steps": 20, "prediction_type": "v_prediction", "hidden_size": 896, "latent_size": 64, "speech_vae_dim": 64, "head_layers": 4, "head_ffn_ratio": 3.0, "rms_norm_eps": 1e-5, "ddpm_batch_mul": 4}, "preprocessor_config": {"processor_class": "VibeVoiceStreamingProcessor", "speech_tok_compress_ratio": 3200, "db_normalize": True, "language_model_pretrained_name": "Qwen/Qwen2.5-0.5B", "audio_processor": {"feature_extractor_type": "VibeVoiceTokenizerProcessor", "sampling_rate": 24000, "normalize_audio": True, "target_dB_FS": -25, "eps": 1e-6}}}


def validate_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict) or "semantic_tokenizer_config" in config:
        raise RuntimeError("Realtime config has invalid top-level topology")
    exact = {
        ("model_type",): "vibevoice_streaming", ("architectures",): ["VibeVoiceStreamingForConditionalGenerationInference"], ("acoustic_vae_dim",): 64, ("tts_backbone_num_hidden_layers",): 20, ("transformers_version",): "4.51.3",
        ("torch_dtype",): "bfloat16", ("acoustic_tokenizer_config", "causal"): True, ("acoustic_tokenizer_config", "channels"): 1, ("acoustic_tokenizer_config", "conv_bias"): True, ("acoustic_tokenizer_config", "conv_norm"): "none", ("acoustic_tokenizer_config", "corpus_normalize"): 0.0, ("acoustic_tokenizer_config", "decoder_depths"): None, ("acoustic_tokenizer_config", "decoder_n_filters"): 32, ("acoustic_tokenizer_config", "decoder_ratios"): [8, 5, 5, 4, 2, 2], ("acoustic_tokenizer_config", "disable_last_norm"): True, ("acoustic_tokenizer_config", "encoder_depths"): "3-3-3-3-3-3-8", ("acoustic_tokenizer_config", "encoder_n_filters"): 32, ("acoustic_tokenizer_config", "encoder_ratios"): [8, 5, 5, 4, 2, 2], ("acoustic_tokenizer_config", "fix_std"): 0.5, ("acoustic_tokenizer_config", "layer_scale_init_value"): 1e-6, ("acoustic_tokenizer_config", "layernorm"): "RMSNorm", ("acoustic_tokenizer_config", "layernorm_elementwise_affine"): True, ("acoustic_tokenizer_config", "layernorm_eps"): 1e-5, ("acoustic_tokenizer_config", "mixer_layer"): "depthwise_conv", ("acoustic_tokenizer_config", "model_type"): "vibevoice_acoustic_tokenizer", ("acoustic_tokenizer_config", "pad_mode"): "constant", ("acoustic_tokenizer_config", "std_dist_type"): "gaussian", ("acoustic_tokenizer_config", "vae_dim"): 64, ("acoustic_tokenizer_config", "weight_init_value"): 0.01,
        ("decoder_config", "model_type"): "qwen2", ("decoder_config", "hidden_size"): 896, ("decoder_config", "intermediate_size"): 4864, ("decoder_config", "max_position_embeddings"): 8192, ("decoder_config", "num_hidden_layers"): 24, ("decoder_config", "num_attention_heads"): 14, ("decoder_config", "num_key_value_heads"): 2, ("decoder_config", "vocab_size"): 151936, ("decoder_config", "rope_theta"): 1e6, ("decoder_config", "rms_norm_eps"): 1e-6, ("decoder_config", "tie_word_embeddings"): False, ("decoder_config", "max_window_layers"): 24, ("decoder_config", "sliding_window"): None, ("decoder_config", "rope_scaling"): None, ("decoder_config", "use_cache"): True,
        ("diffusion_head_config", "model_type"): "vibevoice_diffusion_head", ("diffusion_head_config", "ddpm_beta_schedule"): "cosine", ("diffusion_head_config", "ddpm_num_steps"): 1000, ("diffusion_head_config", "ddpm_num_inference_steps"): 20, ("diffusion_head_config", "prediction_type"): "v_prediction", ("diffusion_head_config", "hidden_size"): 896, ("diffusion_head_config", "latent_size"): 64, ("diffusion_head_config", "speech_vae_dim"): 64, ("diffusion_head_config", "head_layers"): 4, ("diffusion_head_config", "head_ffn_ratio"): 3.0, ("diffusion_head_config", "rms_norm_eps"): 1e-5, ("diffusion_head_config", "ddpm_batch_mul"): 4,
    }
    for path, value in exact.items():
        require(config, path, value)
    paths = sorted(exact)
    return {"required_paths": [".".join(path) for path in paths], "values": {".".join(path): exact[path] for path in paths}}


def validate_preprocessor(preprocessor: Any) -> dict[str, Any]:
    exact = {
        ("processor_class",): "VibeVoiceStreamingProcessor",
        ("speech_tok_compress_ratio",): 3200,
        ("db_normalize",): True,
        ("language_model_pretrained_name",): "Qwen/Qwen2.5-0.5B",
        ("audio_processor", "feature_extractor_type"): "VibeVoiceTokenizerProcessor",
        ("audio_processor", "sampling_rate"): 24000,
        ("audio_processor", "normalize_audio"): True,
        ("audio_processor", "target_dB_FS"): -25,
        ("audio_processor", "eps"): 1e-6,
    }
    for path, value in exact.items():
        require(preprocessor, path, value)
    paths = sorted(exact)
    return {"required_paths": [".".join(path) for path in paths], "values": {".".join(path): exact[path] for path in paths}}


def inspect_safetensors(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise RuntimeError("truncated safetensors header")
        header_bytes = int.from_bytes(prefix, "little")
        if header_bytes <= 0 or header_bytes > MAX_HEADER_BYTES or header_bytes > size - 8:
            raise RuntimeError("invalid safetensors header length")
        raw = stream.read(header_bytes)
    try:
        header = json.loads(raw.decode(), object_pairs_hook=strict_pairs)
    except (UnicodeError, json.JSONDecodeError, RuntimeError) as error:
        raise RuntimeError(f"invalid safetensors header: {error}") from error
    if not isinstance(header, dict):
        raise RuntimeError("safetensors header is not an object")
    metadata = header.get("__metadata__")
    if metadata is not None and (not isinstance(metadata, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items())):
        raise RuntimeError("safetensors metadata must be a string map")
    payload = size - 8 - header_bytes
    intervals: list[tuple[int, int, str]] = []
    rows: list[dict[str, Any]] = []
    for name, record in header.items():
        if name == "__metadata__":
            continue
        safe_relative(name, "tensor")
        if not isinstance(record, dict) or set(record) != {"dtype", "shape", "data_offsets"} or record["dtype"] != "BF16":
            raise RuntimeError(f"tensor {name} must be a BF16 descriptor")
        shape, offsets = record["shape"], record["data_offsets"]
        if not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2 or any(not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape) or any(not isinstance(offset, int) or isinstance(offset, bool) for offset in offsets):
            raise RuntimeError(f"invalid tensor shape/offsets: {name}")
        elements = math.prod(shape)
        start, end = offsets
        if start < 0 or end < start or end > payload or end - start != elements * 2:
            raise RuntimeError(f"tensor range gap/overlap/size mismatch: {name}")
        intervals.append((start, end, name))
        rows.append({"name": name, "dtype": "BF16", "shape": shape, "elements": elements, "data_offsets": [start, end]})
    cursor = 0
    for start, end, name in sorted(intervals):
        if start != cursor:
            raise RuntimeError(f"tensor range gap/overlap before {name}")
        cursor = end
    if cursor != payload or sum(row["elements"] for row in rows) != PARAMETERS:
        raise RuntimeError("tensor payload end or authenticated parameter sum mismatch")
    with safe_open(str(path), framework="pt") as handle:
        if set(handle.keys()) != {row["name"] for row in rows}:
            raise RuntimeError("safetensors header/key mismatch")
    return {"path": path.name, "bytes": size, "sha256": sha256(path), "header_bytes": header_bytes, "payload_bytes": payload, "tensor_count": len(rows), "parameter_count": sum(row["elements"] for row in rows), "all_dtype": "BF16"}, rows


def parse_model_card_frontmatter(text: str) -> dict[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise RuntimeError("model-card frontmatter is missing")
    try:
        end = next(index for index in range(1, len(lines)) if lines[index].strip() == "---")
    except StopIteration as error:
        raise RuntimeError("model-card frontmatter is unterminated") from error
    result: dict[str, str] = {}
    seen: set[str] = set()
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#") or line[0].isspace() or line.startswith("-"):
            continue
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_-]*):[ \t]*(.*)", line)
        if match is None:
            raise RuntimeError("model-card frontmatter has malformed top-level YAML")
        key, raw = match.groups()
        if key in seen:
            raise RuntimeError(f"model-card frontmatter key duplicated: {key}")
        seen.add(key)
        if key == "license":
            value = raw.strip().strip("\"'")
            if value != "mit":
                raise RuntimeError("model-card license is not exactly mit")
            result[key] = value
    if result.get("license") != "mit":
        raise RuntimeError("model-card license is not exactly one top-level mit")
    return result


def license_record(root: Path, expected: str, blocker: str) -> dict[str, Any]:
    candidates = [root / "LICENSE", root / "LICENSE.md"]
    markers = {
        "MIT": ("permission is hereby granted, free of charge", "the software is provided \"as is\"", "without warranty"),
        "Apache-2.0": ("apache license, version 2.0", "you may obtain a copy of the license", "distributed under the license", "without warranties or conditions"),
    }[expected]
    for path in candidates:
        if path.is_file() and not path.is_symlink():
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            if all(marker in text for marker in markers):
                return {"license": expected, "status": "AUTHENTICATED", "path": path.name, "sha256": sha256(path), "markers": {marker: marker in text for marker in markers}}
    return {"license": "UNKNOWN", "status": blocker, "path": None, "sha256": None, "markers": {marker: False for marker in markers}}


def model_card_license(root: Path) -> dict[str, Any]:
    path = root / "README.md"
    try:
        parsed = parse_model_card_frontmatter(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, RuntimeError) as error:
        return {"license": "UNKNOWN", "status": "HF_MODEL_LICENSE_UNKNOWN_BLOCKER", "path": "README.md", "sha256": None, "error": str(error)}
    return {"license": "MIT", "status": "AUTHENTICATED", "path": "README.md", "sha256": sha256(path), "frontmatter": parsed}


def select_server_rows(rows: Any, selected: set[str]) -> list[dict[str, Any]]:
    """Validate a full remote walk, then retain only an explicit local allowlist."""
    if not isinstance(rows, list):
        raise RuntimeError("server walk must be a list")
    seen: set[str] = set()
    chosen: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "type", "size", "git_blob_sha1", "lfs_pointer_git_blob_sha1", "lfs_sha256"}:
            raise RuntimeError("server walk row schema mismatch")
        name = row["path"]
        if not isinstance(name, str) or name in seen or row["type"] != "file":
            raise RuntimeError("server walk duplicate/type/path mismatch")
        safe_relative(name, "server walk")
        size = row["size"]
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise RuntimeError(f"server walk size is not a nonnegative integer: {name}")
        regular, pointer, payload = row["git_blob_sha1"], row["lfs_pointer_git_blob_sha1"], row["lfs_sha256"]
        if regular is not None and (not isinstance(regular, str) or not re.fullmatch(r"[0-9a-f]{40}", regular)):
            raise RuntimeError(f"server walk regular Git blob identity is invalid: {name}")
        if pointer is not None and (not isinstance(pointer, str) or not re.fullmatch(r"[0-9a-f]{40}", pointer)):
            raise RuntimeError(f"server walk LFS pointer Git blob identity is invalid: {name}")
        if payload is not None and (not isinstance(payload, str) or not re.fullmatch(r"[0-9a-f]{64}", payload)):
            raise RuntimeError(f"server walk LFS payload identity is invalid: {name}")
        if payload is None:
            if regular is None or pointer is not None:
                raise RuntimeError(f"server walk regular/LFS fields are inconsistent: {name}")
        elif regular is not None or pointer is None:
            raise RuntimeError(f"server walk LFS fields are inconsistent: {name}")
        else:
            pointer_bytes = f"version https://git-lfs.github.com/spec/v1\noid sha256:{payload}\nsize {size}\n".encode()
            if git_blob_sha1_bytes(pointer_bytes) != pointer:
                raise RuntimeError(f"server walk canonical LFS pointer mismatch: {name}")
        seen.add(name)
        if name in selected:
            chosen.append(row)
    if {row["path"] for row in chosen} != selected:
        raise RuntimeError(f"selected server file set mismatch: {sorted(selected - {row['path'] for row in chosen})}")
    return sorted(chosen, key=lambda row: row["path"])


def validate_tokenizer_file_set(names: set[str]) -> None:
    missing = TOKENIZER_SELECTED - names
    if missing:
        raise RuntimeError(f"companion tokenizer files missing: {sorted(missing)}")
    if any(Path(name).suffix in (".bin", ".pt", ".safetensors") for name in names):
        raise RuntimeError("companion tokenizer snapshot unexpectedly contains model weights")


def source_inventory(
    source: Path,
    transformers: Path,
    *,
    source_repository: str = SOURCE_REPOSITORY,
    source_revision: str = SOURCE_REVISION,
    source_roles: dict[str, str] = SOURCE_ROLE_BLOBS,
    transformers_repository: str = TRANSFORMERS_REPOSITORY,
    transformers_revision: str = TRANSFORMERS_REVISION,
    transformers_roles: dict[str, str] = TRANSFORMERS_ROLE_BLOBS,
    transformers_tag: str = TRANSFORMERS_TAG,
) -> dict[str, Any]:
    def checkout(root: Path, repository: str, revision: str, roles: dict[str, str], tag: str | None, license_name: str, blocker: str) -> dict[str, Any]:
        blockers: list[str] = []
        if git(root, "status", "--porcelain", "--untracked-files=all"):
            blockers.append("checkout is dirty")
        actual = git(root, "rev-parse", "HEAD")
        if actual != revision:
            blockers.append("revision mismatch")
        origin = git(root, "remote", "get-url", "origin").removesuffix("/").removesuffix(".git")
        if origin != repository.removesuffix(".git"):
            blockers.append("origin mismatch")
        if tag is not None and git(root, "describe", "--exact-match", "--tags", "HEAD") != tag:
            blockers.append("tag mismatch")
        raw = subprocess.check_output(["git", "-C", str(root), "ls-files", "-s", "-z"])
        entries: dict[str, dict[str, Any]] = {}
        for item in raw.split(b"\0"):
            if not item:
                continue
            header, encoded = item.split(b"\t", 1)
            fields = header.split()
            name = encoded.decode("utf-8")
            if len(fields) != 3 or name in entries:
                blockers.append(f"tracked index schema/duplicate: {name}")
                continue
            mode, index_object, stage = fields[0].decode(), fields[1].decode(), fields[2].decode()
            if stage != "0":
                blockers.append(f"tracked entry is not regular stage-0: {name}")
                continue
            try:
                path, filesystem = filesystem_entry(root, name)
            except (OSError, RuntimeError) as error:
                blockers.append(str(error))
                continue
            if mode in {"100644", "100755"}:
                if stat.S_ISLNK(filesystem.st_mode) or not stat.S_ISREG(filesystem.st_mode):
                    blockers.append(f"tracked entry is not regular stage-0: {name}")
                    continue
                expected_mode = 0o755 if mode == "100755" else 0o644
                if filesystem.st_mode & 0o777 != expected_mode:
                    blockers.append(f"filesystem mode mismatch: {name}")
                head_object = git(root, "rev-parse", f"HEAD:{name}")
                working_object = git_blob_sha1(path)
                entries[name] = {"path": name, "mode": mode, "stage": stage, "index_object_sha1": index_object, "head_object_sha1": head_object, "working_blob_sha1": working_object, "bytes": filesystem.st_size, "sha256": sha256(path)}
                if not (index_object == head_object == working_object):
                    blockers.append(f"tracked object mismatch: {name}")
                continue
            if mode == "120000":
                if not stat.S_ISLNK(filesystem.st_mode):
                    blockers.append(f"tracked entry is not a symlink: {name}")
                    continue
                try:
                    target = os.readlink(os.fsencode(path))
                    if not isinstance(target, bytes):
                        target = os.fsencode(target)
                    validate_symlink_target(name, target)
                except (OSError, RuntimeError) as error:
                    blockers.append(str(error))
                    continue
                head_object = git(root, "rev-parse", f"HEAD:{name}")
                working_object = git_blob_sha1_bytes(target)
                entries[name] = {"path": name, "mode": mode, "stage": stage, "index_object_sha1": index_object, "head_object_sha1": head_object, "working_blob_sha1": working_object, "bytes": len(target), "sha256": sha256_bytes(target), "symlink_target_hex": target.hex()}
                if not (index_object == head_object == working_object):
                    blockers.append(f"tracked symlink object mismatch: {name}")
                continue
            if mode not in {"100644", "100755"}:
                blockers.append(f"tracked entry is not regular stage-0: {name}")
                continue
        role_records: list[dict[str, Any]] = []
        for name, expected in roles.items():
            row = entries.get(name)
            if row is None:
                blockers.append(f"missing/untracked role: {name}")
                continue
            row = {**row, "expected_git_blob_sha1": expected}
            role_records.append(row)
            if row["mode"] != "100644" or not (row["index_object_sha1"] == row["head_object_sha1"] == row["working_blob_sha1"] == expected):
                blockers.append(f"role object/mode mismatch: {name}")
        license_evidence = license_record(root, license_name, blocker)
        if license_evidence["status"] != "AUTHENTICATED":
            blockers.append("license text/blob mismatch")
        return {"repository": repository, "revision": revision, "resolved_revision": actual, "origin": origin, "tag": tag, "clean": not any("dirty" in item for item in blockers), "tracked_files": sorted(entries.values(), key=lambda row: row["path"]), "role_files": sorted(role_records, key=lambda row: row["path"]), "license": license_evidence, "blockers": blockers, "status": "AUTHENTICATED" if not blockers else "BLOCKED"}
    return {"source": checkout(source, source_repository, source_revision, source_roles, None, "MIT", "SOURCE_LICENSE_UNKNOWN_BLOCKER"), "transformers": checkout(transformers, transformers_repository, transformers_revision, transformers_roles, transformers_tag, "Apache-2.0", "TRANSFORMERS_LICENSE_UNKNOWN_BLOCKER")}


def manifest_license_evidence(model_license: dict[str, Any], sources: dict[str, Any]) -> dict[str, Any]:
    source = sources["source"]
    transformers = sources["transformers"]
    return {"model": model_license, "source": source["license"], "transformers": transformers["license"], "base_tokenizer": "SEPARATE_REVIEW_REQUIRED"}


def blocked(output: Path, error: Exception, inspection_status: str = "INSPECTION_ERROR", **extra: Any) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = {"format": FORMAT, "status": "BLOCKED", "inspection_status": inspection_status, "evidence_stage": "INSPECTION_ONLY", "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "cpu_status": "UNSUPPORTED", "metal_status": "BLOCKED_BY_CPU", "parity_status": "NOT_RUN", "publication": "NO_UPLOAD", "task": "Realtime streaming TTS inspection only; no native runtime claim", "upstream": {"repository": HF_REPOSITORY, "revision": HF_REVISION}, "official_source": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION}, "transformers": {"repository": TRANSFORMERS_REPOSITORY, "tag": TRANSFORMERS_TAG, "revision": TRANSFORMERS_REVISION}, "error_type": type(error).__name__, "reason": str(error), "blockers": [str(error)], **extra}
    (output / "manifest.json").write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def inspect(snapshot: Path, companion: Path, source: Path, transformers: Path, model_tree: Path, companion_tree: Path, output: Path) -> int:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise RuntimeError("inspection output must be absent or empty")
    model_identity, model_files = server_inventory(snapshot, model_tree, set(MODEL_FILES))
    for name, (size, git_id, lfs_id) in MODEL_FILES.items():
        row = next(row for row in model_files if row["path"] == name)
        packet_git_id = row["lfs_pointer_git_blob_sha1"] if lfs_id is not None else row["git_blob_sha1"]
        if row["bytes"] != size or packet_git_id is None or packet_git_id.lower() != git_id.lower() or row["lfs_sha256"] != lfs_id:
            raise RuntimeError(f"authenticated model identity mismatch: {name}")
    readme = snapshot / "README.md"
    model_license = model_card_license(snapshot)
    if model_license["status"] != "AUTHENTICATED":
        raise RuntimeError("canonical HF MIT model declaration missing")
    policy_text = readme.read_text(encoding="utf-8", errors="replace").lower()
    policy = {"status": "BLOCKED_POLICY_REVIEW", "research_or_advisory_language": any(token in policy_text for token in ("research", "non-commercial", "advisory"))}
    parsed: dict[str, Any] = {}
    for path in sorted(snapshot.rglob("*.json")):
        if path.is_file() and ".cache" not in path.relative_to(snapshot).parts:
            parsed[path.relative_to(snapshot).as_posix()] = {"sha256": sha256(path), "json": load_json(path)}
    config = parsed.get("config.json", {}).get("json")
    config_evidence = validate_config(config)
    preprocessor = parsed.get("preprocessor_config.json", {}).get("json")
    preprocessor_evidence = validate_preprocessor(preprocessor)
    tensor_evidence, tensors = inspect_safetensors(snapshot / "model.safetensors")
    sources = source_inventory(source, transformers)
    if sources["source"]["status"] != "AUTHENTICATED" or sources["transformers"]["status"] != "AUTHENTICATED":
        raise RuntimeError("official source/Transformers checkout authentication is incomplete")
    tokenizer_identity, tokenizer_files = server_inventory(companion, companion_tree)
    companion_names = {row["path"] for row in tokenizer_files}
    validate_tokenizer_file_set(companion_names)
    companion_json = {row["path"]: {"sha256": sha256(companion / row["path"]), "json": load_json(companion / row["path"])} for row in tokenizer_files if row["path"].endswith(".json")}
    output.mkdir(parents=True, exist_ok=True)
    evidence = {"snapshot-inventory.json": {"server_tree": model_identity, "files": model_files}, "tensor-inventory.json": {"header": tensor_evidence, "tensors": tensors}, "parsed-json.json": parsed, "companion-inventory.json": {"server_tree": tokenizer_identity, "files": tokenizer_files, "json": companion_json}, "source-inventory.json": sources}
    for name, value in evidence.items():
        (output / name).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    packets = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in output.glob("*-inventory.json")}
    blocked(output, RuntimeError("streaming state, diffusion/CFG, acoustic decoder, tokenizer behavior, policy, and dataset provenance remain unauthenticated"), inspection_status="AUTHENTICATED_EVIDENCE_COMPLETE", model_license=model_license, policy=policy, config=config_evidence, preprocessor=preprocessor_evidence, tensors=tensor_evidence, companion_tokenizer={"repository": TOKENIZER_REPOSITORY, "revision": TOKENIZER_REVISION, "model_weights": "NOT_DOWNLOADED", "files": tokenizer_files}, official_source=sources, license_evidence=manifest_license_evidence(model_license, sources), dataset_provenance={"status": "BLOCKED_UNAUTHENTICATED"}, packets=packets)
    return 2


def self_test() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    assert "safe_open" in source and "BF16" in source
    assert len(HF_REVISION) == len(SOURCE_REVISION) == len(TRANSFORMERS_REVISION) == len(TOKENIZER_REVISION) == 40
    assert PARAMETERS == 1_017_626_724
    assert PARAMETERS * 2 == AUTHENTICATED_PAYLOAD_BYTES
    assert 8 + AUTHENTICATED_HEADER_BYTES + AUTHENTICATED_PAYLOAD_BYTES == MODEL_FILES["model.safetensors"][0]
    assert "VibeVoiceStreamingForConditionalGenerationInference" in source
    try:
        json.loads('{"x":1,"x":2}', object_pairs_hook=strict_pairs)
    except RuntimeError:
        pass
    else:
        raise AssertionError("duplicate JSON key accepted")
    assert validate_config(config_fixture())["required_paths"]
    assert validate_preprocessor(config_fixture()["preprocessor_config"])["required_paths"]
    config_evidence = validate_config(config_fixture())
    preprocessor_evidence = validate_preprocessor(config_fixture()["preprocessor_config"])
    json.dumps(config_evidence, sort_keys=True)
    json.dumps(preprocessor_evidence, sort_keys=True)
    assert config_evidence["values"]["decoder_config.hidden_size"] == 896
    assert preprocessor_evidence["values"]["audio_processor.sampling_rate"] == 24000
    assert parse_model_card_frontmatter("---\nlicense: mit\ntags:\n- audio\n---\nprose") == {"license": "mit"}
    for card in ("prose license: mit", "---\nlicense: mit\nlicense: mit\n---", "---\ntags:\n  license: mit\n---", "---\nlicense: apache-2.0\n---"):
        try:
            parse_model_card_frontmatter(card)
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid/prose model-card license accepted")
    selected_rows = [{"path": "LICENSE", "type": "file", "size": 1, "git_blob_sha1": "a" * 40, "lfs_pointer_git_blob_sha1": None, "lfs_sha256": None}, {"path": "README.md", "type": "file", "size": 2, "git_blob_sha1": "b" * 40, "lfs_pointer_git_blob_sha1": None, "lfs_sha256": None}]
    assert [row["path"] for row in select_server_rows(selected_rows + [{"path": "model.safetensors", "type": "file", "size": 3, "git_blob_sha1": "c" * 40, "lfs_pointer_git_blob_sha1": None, "lfs_sha256": None}], {"LICENSE", "README.md"})] == ["LICENSE", "README.md"]
    malformed_unselected = [
        {"path": "bad-hex", "type": "file", "size": 1, "git_blob_sha1": "G" * 40, "lfs_pointer_git_blob_sha1": None, "lfs_sha256": None},
        {"path": "bool-size", "type": "file", "size": True, "git_blob_sha1": "d" * 40, "lfs_pointer_git_blob_sha1": None, "lfs_sha256": None},
        {"path": "bad-lfs", "type": "file", "size": 1, "git_blob_sha1": None, "lfs_pointer_git_blob_sha1": "e" * 40, "lfs_sha256": "f" * 64},
    ]
    for malformed in malformed_unselected:
        try:
            select_server_rows(selected_rows + [malformed], {"LICENSE", "README.md"})
        except RuntimeError:
            pass
        else:
            raise AssertionError("malformed unselected server row accepted")
    try:
        select_server_rows(selected_rows, TOKENIZER_SELECTED)
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing selected tokenizer file accepted")
    try:
        validate_tokenizer_file_set({"LICENSE", "tokenizer.json"})
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing companion tokenizer file accepted")
    bad = config_fixture(); bad["decoder_config"]["hidden_size"] = 897
    try:
        validate_config(bad)
    except RuntimeError:
        pass
    else:
        raise AssertionError("misnested/wrong config value accepted")
    for section, old_key, current_key in (
        ("acoustic_tokenizer_config", "layernorm_affine", "layernorm_elementwise_affine"),
        ("diffusion_head_config", "beta_schedule", "ddpm_beta_schedule"),
        ("diffusion_head_config", "num_inference_timesteps", "ddpm_num_inference_steps"),
        ("diffusion_head_config", "num_train_timesteps", "ddpm_num_steps"),
        ("diffusion_head_config", "ffn_ratio", "head_ffn_ratio"),
        ("diffusion_head_config", "num_layers", "head_layers"),
    ):
        obsolete = config_fixture(); value = obsolete[section].pop(current_key); obsolete[section][old_key] = value
        try:
            validate_config(obsolete)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"obsolete config key accepted: {section}.{old_key}")
    with tempfile.TemporaryDirectory(prefix="vokra-vibevoice-realtime-") as directory:
        root = Path(directory); huge = root / "huge.safetensors"; huge.write_bytes((MAX_HEADER_BYTES + 1).to_bytes(8, "little"))
        try:
            inspect_safetensors(huge)
        except RuntimeError:
            pass
        else:
            raise AssertionError("huge header accepted")
        unsafe = root / "unsafe.safetensors"; raw = json.dumps({"../bad": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}).encode(); unsafe.write_bytes(len(raw).to_bytes(8, "little") + raw + b"\0\0")
        try:
            inspect_safetensors(unsafe)
        except RuntimeError:
            pass
        else:
            raise AssertionError("unsafe tensor name accepted")
        snapshot = root / "snapshot"; snapshot.mkdir(); content = snapshot / "x"; content.write_bytes(b"abc")
        packet = root / "tree.json"; base = {"repository": HF_REPOSITORY, "requested_revision": HF_REVISION, "resolved_revision": HF_REVISION, "walk": "recursive_file_only", "files": [{"path": "x", "type": "file", "size": 3, "git_blob_sha1": git_blob_sha1(content), "lfs_pointer_git_blob_sha1": None, "lfs_sha256": None}]}; packet.write_text(json.dumps(base), encoding="utf-8"); server_inventory(snapshot, packet)
        content.write_bytes(b"abd")
        try:
            server_inventory(snapshot, packet)
        except RuntimeError:
            pass
        else:
            raise AssertionError("same-size mutation accepted")
        content.write_bytes(b"abc")
        transport = snapshot / ".cache" / "huggingface"; transport.mkdir(parents=True); (transport / "metadata").write_bytes(b"cache")
        server_inventory(snapshot, packet)
        (snapshot / ".cache" / "other").mkdir()
        try:
            server_inventory(snapshot, packet)
        except RuntimeError:
            pass
        else:
            raise AssertionError("non-transport cache was accepted")
        (snapshot / ".cache" / "other").rmdir()
        (transport / "metadata").unlink(); transport.rmdir(); (snapshot / ".cache").rmdir()
        payload = snapshot / "lfs"; payload.write_bytes(b"payload")
        lfs_sha = sha256(payload); pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{lfs_sha}\nsize {payload.stat().st_size}\n".encode()
        lfs_packet = {"repository": HF_REPOSITORY, "requested_revision": HF_REVISION, "resolved_revision": HF_REVISION, "walk": "recursive_file_only", "files": [{"path": "x", "type": "file", "size": 3, "git_blob_sha1": git_blob_sha1(content), "lfs_pointer_git_blob_sha1": None, "lfs_sha256": None}, {"path": "lfs", "type": "file", "size": payload.stat().st_size, "git_blob_sha1": None, "lfs_pointer_git_blob_sha1": git_blob_sha1_bytes(pointer), "lfs_sha256": lfs_sha}]}
        lfs_packet_path = root / "lfs-tree.json"; lfs_packet_path.write_text(json.dumps(lfs_packet), encoding="utf-8")
        server_inventory(snapshot, lfs_packet_path)
        lfs_packet["files"][1]["lfs_pointer_git_blob_sha1"] = "0" * 40; lfs_packet_path.write_text(json.dumps(lfs_packet), encoding="utf-8")
        try:
            server_inventory(snapshot, lfs_packet_path)
        except RuntimeError:
            pass
        else:
            raise AssertionError("LFS pointer identity spoof accepted")
        payload.unlink()
        symlink = snapshot / "link"; symlink.symlink_to(content)
        try:
            server_inventory(snapshot, packet)
        except RuntimeError:
            pass
        else:
            raise AssertionError("snapshot payload symlink accepted")
        symlink.unlink()
        def git_fixture(path: Path, repository: str, tag: str | None = None) -> None:
            path.mkdir()
            subprocess.run(["git", "init", "-q", str(path)], check=True)
            subprocess.run(["git", "-C", str(path), "config", "user.email", "vibevoice-selftest@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(path), "config", "user.name", "VibeVoice self-test"], check=True)
            (path / "LICENSE").write_text("MIT License\nPermission is hereby granted, free of charge, to any person obtaining a copy.\nThe software is provided 'as is', without warranty.\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(path), "add", "LICENSE"], check=True)
            subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "fixture"], check=True)
            if tag is not None:
                subprocess.run(["git", "-C", str(path), "tag", tag], check=True)
            subprocess.run(["git", "-C", str(path), "remote", "add", "origin", repository], check=True)
        source_fixture = root / "source-fixture"; transformers_fixture = root / "transformers-fixture"
        git_fixture(source_fixture, SOURCE_REPOSITORY)
        git_fixture(transformers_fixture, TRANSFORMERS_REPOSITORY, TRANSFORMERS_TAG)
        inventory = source_inventory(source_fixture, transformers_fixture)
        assert inventory["source"]["status"] == "BLOCKED" and inventory["transformers"]["status"] == "BLOCKED"
        assert any("missing/untracked role" in item for item in inventory["source"]["blockers"])
        (source_fixture / "dirty.txt").write_text("dirty", encoding="utf-8")
        dirty_inventory = source_inventory(source_fixture, transformers_fixture)
        assert dirty_inventory["source"]["status"] == "BLOCKED" and any("dirty" in item for item in dirty_inventory["source"]["blockers"])
        def authenticated_fixture(path: Path, repository: str, role_paths: tuple[str, ...], license_text: str, tag: str | None = None) -> tuple[str, dict[str, str]]:
            path.mkdir()
            subprocess.run(["git", "init", "-q", str(path)], check=True)
            subprocess.run(["git", "-C", str(path), "config", "user.email", "vibevoice-selftest@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(path), "config", "user.name", "VibeVoice self-test"], check=True)
            for role in role_paths:
                role_path = path / role; role_path.parent.mkdir(parents=True, exist_ok=True)
                role_path.write_text(license_text if role == "LICENSE" else f"fixture role {role}\n", encoding="utf-8")
            (path / "safe-link").symlink_to("LICENSE")
            subprocess.run(["git", "-C", str(path), "add", *role_paths, "safe-link"], check=True)
            subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "authenticated fixture"], check=True)
            if tag is not None:
                subprocess.run(["git", "-C", str(path), "tag", tag], check=True)
            subprocess.run(["git", "-C", str(path), "remote", "add", "origin", repository], check=True)
            revision = git(path, "rev-parse", "HEAD")
            return revision, {role: git_blob_sha1(path / role) for role in role_paths}
        mit_text = 'MIT License\nPermission is hereby granted, free of charge, to any person obtaining a copy.\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.\n'
        apache_text = 'Apache License, Version 2.0\nYou may obtain a copy of the License.\nDistributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.\n'
        positive_source = root / "positive-source"; positive_transformers = root / "positive-transformers"
        positive_source_revision, positive_source_roles = authenticated_fixture(positive_source, SOURCE_REPOSITORY, tuple(SOURCE_ROLE_BLOBS), mit_text)
        positive_transformers_revision, positive_transformers_roles = authenticated_fixture(positive_transformers, TRANSFORMERS_REPOSITORY, tuple(TRANSFORMERS_ROLE_BLOBS), apache_text, TRANSFORMERS_TAG)
        positive_inventory = source_inventory(positive_source, positive_transformers, source_revision=positive_source_revision, source_roles=positive_source_roles, transformers_revision=positive_transformers_revision, transformers_roles=positive_transformers_roles)
        assert positive_inventory["source"]["status"] == "AUTHENTICATED" and positive_inventory["transformers"]["status"] == "AUTHENTICATED"
        safe_links = [row for row in positive_inventory["source"]["tracked_files"] if row["path"] == "safe-link"]
        assert len(safe_links) == 1 and safe_links[0]["mode"] == "120000" and safe_links[0]["symlink_target_hex"] == b"LICENSE".hex()
        (positive_source / "escape-link").symlink_to("../outside")
        subprocess.run(["git", "-C", str(positive_source), "add", "escape-link"], check=True)
        subprocess.run(["git", "-C", str(positive_source), "commit", "-q", "-m", "unsafe symlink fixture"], check=True)
        unsafe_revision = git(positive_source, "rev-parse", "HEAD")
        unsafe_inventory = source_inventory(positive_source, positive_transformers, source_revision=unsafe_revision, source_roles=positive_source_roles, transformers_revision=positive_transformers_revision, transformers_roles=positive_transformers_roles)
        assert unsafe_inventory["source"]["status"] == "BLOCKED" and any("escapes checkout" in item for item in unsafe_inventory["source"]["blockers"])
        for target in (b"/etc/passwd", b"../outside", b"nested/../../outside", b"bad\x00target"):
            try:
                validate_symlink_target("safe-link", target)
            except RuntimeError:
                pass
            else:
                raise AssertionError("unsafe tracked symlink target accepted")
        complete_license = manifest_license_evidence({"license": "MIT", "status": "AUTHENTICATED"}, positive_inventory)
        assert complete_license["source"]["status"] == "AUTHENTICATED" and complete_license["transformers"]["status"] == "AUTHENTICATED"
        for files in ([], base["files"] + [{"path": "extra", "type": "file", "size": 0, "git_blob_sha1": "0" * 40, "lfs_pointer_git_blob_sha1": None, "lfs_sha256": None}]):
            changed = dict(base); changed["files"] = files; packet.write_text(json.dumps(changed), encoding="utf-8")
            try:
                server_inventory(snapshot, packet)
            except RuntimeError:
                pass
            else:
                raise AssertionError("missing/extra server path accepted")
        error_evidence = root / "error-evidence"; blocked(error_evidence, RuntimeError("fixture failure")); error_manifest = load_json(error_evidence / "manifest.json")
        assert error_manifest["inspection_status"] == "INSPECTION_ERROR" and "AUTHENTICATED_EVIDENCE_COMPLETE" not in error_manifest
    print("vibevoice_realtime_0_5b_inspect.py self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--snapshot", type=Path); parser.add_argument("--companion", type=Path); parser.add_argument("--source", type=Path); parser.add_argument("--transformers", type=Path); parser.add_argument("--server-tree", type=Path); parser.add_argument("--companion-server-tree", type=Path); parser.add_argument("--output", type=Path); parser.add_argument("--self-test", action="store_true"); args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.snapshot, args.companion, args.source, args.transformers, args.server_tree, args.companion_server_tree, args.output)): parser.error("--self-test accepts no paths")
        self_test(); return 0
    if any(value is None for value in (args.snapshot, args.companion, args.source, args.transformers, args.server_tree, args.companion_server_tree, args.output)): parser.error("all inspection paths are required")
    try:
        return inspect(args.snapshot, args.companion, args.source, args.transformers, args.server_tree, args.companion_server_tree, args.output)
    except Exception as error:
        blocked(args.output, error); print(f"VibeVoice Realtime inspection BLOCKED: {error}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
