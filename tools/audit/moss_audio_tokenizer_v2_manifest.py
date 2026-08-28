#!/usr/bin/env python3
"""Audit the fixed MOSS Audio Tokenizer v2 tensor manifest.

Without ``--shard-dir`` this audit reads only ``config.json`` and
``model.safetensors.index.json`` at revision
``f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169``. The derived digest remains a
candidate runtime contract.

With ``--shard-dir`` it additionally verifies the exact byte size and SHA-256
of all three 8.49 GB upstream shards, parses their safetensors headers without
mapping tensor payloads, and checks every name, shape, F32 dtype, data range,
and index-to-shard assignment. Only that VAST-only mode emits a confirmed
manifest and clears ``requires_vast_header_confirmation``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path


REVISION = "f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169"
CONFIG_SHA256 = "aeb9a0e9d88c74bf9fbaa81ee54443d463e09b5f335b3306bb798e282a10e564"
INDEX_SHA256 = "912f52f053e04ff7e9abc8f05aa75dfbb40b31c86a0f4ad5c5a36e4aa28a624f"
TENSOR_COUNT = 2_094
PARAMETER_COUNT = 2_123_701_248
SHARD_CONTRACTS = {
    "model-00001-of-00003.safetensors": (
        3_978_639_168,
        "2d9f9182f17b143a23937feb87c63c08221bd28e685e4bc2fa55dcdce17fcde7",
    ),
    "model-00002-of-00003.safetensors": (
        3_992_738_352,
        "d4e48106d0254fe3b00ea0707e88fc6aee076993825e108dd9cef847f9db236e",
    ),
    "model-00003-of-00003.safetensors": (
        523_681_336,
        "d0449fe1b0ef1f6045946867148d8166b9a91a58d0feca4a18b641494d0b22da",
    ),
}
SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}
GGUF_METADATA = {
    "vokra.model.arch": "moss_audio_tokenizer",
    "vokra.model.name": "moss-audio-tokenizer-v2",
    "vokra.model.category": "codec",
    "vokra.moss_audio_tokenizer.variant": "v2",
    "vokra.provenance.weight_license": "permissive",
    "vokra.provenance.license": "apache-2.0",
    "vokra.provenance.model_id": "moss-audio-tokenizer-v2",
    "vokra.provenance.source": (
        "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2 (48 kHz stereo codec, "
        "~2.12B F32 params, 32 residual LFQ codebooks, apache-2.0)"
    ),
    "vokra.provenance.upstream_hf": "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2",
    "vokra.provenance.upstream_revision": REVISION,
    "vokra.moss_audio_tokenizer.config_sha256": CONFIG_SHA256,
    "vokra.moss_audio_tokenizer.configuration_source_sha256": (
        "f87a7a975868ce3f0077f374f46ebd2aab610fd7a26cd7569d16827a14e29529"
    ),
    "vokra.moss_audio_tokenizer.modeling_source_sha256": (
        "7f807e6ee77a60d512e5aa4a8f58a1d5af4e3722f4ab350d70dd538429391cb9"
    ),
    "vokra.moss_audio_tokenizer.index_sha256": INDEX_SHA256,
    "vokra.moss_audio_tokenizer.license_sha256": (
        "50e6751797c50dedd75ef1b8a0d9e42f5f8472e9fbce91f34718e9f97b0c780a"
    ),
}


@dataclass(frozen=True)
class TransformerStage:
    side: str
    module_index: int
    input_dim: int
    output_dim: int
    model_dim: int
    ffn_dim: int
    layers: int


STAGES = (
    TransformerStage("encoder", 1, 240, 384, 768, 3_072, 12),
    TransformerStage("encoder", 3, 768, 384, 768, 3_072, 12),
    TransformerStage("encoder", 5, 768, 384, 768, 3_072, 12),
    TransformerStage("encoder", 7, 768, 384, 768, 3_072, 12),
    TransformerStage("encoder", 9, 768, 640, 768, 3_072, 12),
    TransformerStage("encoder", 11, 1_280, 768, 1_280, 5_120, 32),
    TransformerStage("decoder", 0, 768, 1_280, 1_280, 5_120, 32),
    TransformerStage("decoder", 2, 640, 768, 768, 3_072, 12),
    TransformerStage("decoder", 4, 384, 768, 768, 3_072, 12),
    TransformerStage("decoder", 6, 384, 768, 768, 3_072, 12),
    TransformerStage("decoder", 8, 384, 768, 768, 3_072, 12),
    TransformerStage("decoder", 10, 384, 240, 768, 3_072, 12),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safetensors_header(path: Path) -> dict[str, dict[str, object]]:
    """Read and validate one safetensors header without loading tensor data."""

    file_size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path}: truncated safetensors length prefix")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size == 0 or header_size > file_size - 8:
            raise ValueError(
                f"{path}: invalid safetensors header size {header_size} "
                f"for {file_size}-byte file"
            )
        try:
            raw_header = json.loads(handle.read(header_size).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}: invalid safetensors header JSON: {exc}") from exc
    if not isinstance(raw_header, dict):
        raise ValueError(f"{path}: safetensors header is not an object")

    payload_size = file_size - 8 - header_size
    tensors: dict[str, dict[str, object]] = {}
    ranges: list[tuple[int, int, str]] = []
    for name, descriptor in raw_header.items():
        if name == "__metadata__":
            if not isinstance(descriptor, dict):
                raise ValueError(f"{path}: __metadata__ is not an object")
            continue
        if not isinstance(name, str) or not isinstance(descriptor, dict):
            raise ValueError(f"{path}: invalid tensor descriptor for {name!r}")
        dtype = descriptor.get("dtype")
        shape = descriptor.get("shape")
        offsets = descriptor.get("data_offsets")
        if dtype not in SAFETENSORS_DTYPE_BYTES:
            raise ValueError(f"{path}: {name} has unsupported dtype {dtype!r}")
        if not isinstance(shape, list) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in shape
        ):
            raise ValueError(f"{path}: {name} has invalid shape {shape!r}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in offsets
            )
        ):
            raise ValueError(f"{path}: {name} has invalid data_offsets {offsets!r}")
        start, end = offsets
        expected_bytes = dimension_product(tuple(shape)) * SAFETENSORS_DTYPE_BYTES[dtype]
        if end < start or end - start != expected_bytes:
            raise ValueError(
                f"{path}: {name} byte range [{start}, {end}) does not match "
                f"{dtype} shape {shape} ({expected_bytes} bytes)"
            )
        if end > payload_size:
            raise ValueError(
                f"{path}: {name} ends at {end}, beyond payload size {payload_size}"
            )
        tensors[name] = {
            "dtype": dtype,
            "shape": tuple(shape),
            "data_offsets": (start, end),
        }
        ranges.append((start, end, name))
    if not tensors:
        raise ValueError(f"{path}: safetensors header contains no tensors")

    cursor = 0
    for start, end, name in sorted(ranges):
        if start != cursor:
            raise ValueError(
                f"{path}: non-contiguous payload before {name}: "
                f"expected offset {cursor}, got {start}"
            )
        cursor = end
    if cursor != payload_size:
        raise ValueError(
            f"{path}: tensor payload ends at {cursor}, file payload is {payload_size}"
        )
    return tensors


def validate_real_shards(
    shard_dir: Path,
    weight_map: dict[str, object],
    manifest: dict[str, tuple[int, ...]],
) -> dict[str, dict[str, object]]:
    """Authenticate every fixed shard and compare its real header to the index."""

    unexpected_shards = sorted(
        {
            shard
            for shard in weight_map.values()
            if isinstance(shard, str) and shard not in SHARD_CONTRACTS
        }
    )
    if unexpected_shards:
        raise ValueError(f"index references unexpected shards: {unexpected_shards}")
    invalid_assignments = sorted(
        name for name, shard in weight_map.items() if not isinstance(shard, str)
    )
    if invalid_assignments:
        raise ValueError(
            f"index has non-string shard assignments: {invalid_assignments[:8]}"
        )

    actual: dict[str, tuple[int, ...]] = {}
    shard_results: dict[str, dict[str, object]] = {}
    for filename, (expected_size, expected_sha256) in SHARD_CONTRACTS.items():
        path = shard_dir / filename
        if not path.is_file():
            raise ValueError(f"missing pinned shard: {path}")
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"{filename} size {actual_size} != {expected_size} at {REVISION}"
            )
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"{filename} SHA-256 {actual_sha256} != {expected_sha256} "
                f"at {REVISION}"
            )
        header = safetensors_header(path)
        declared = {name for name, shard in weight_map.items() if shard == filename}
        header_names = set(header)
        if header_names != declared:
            missing = sorted(declared - header_names)
            unexpected = sorted(header_names - declared)
            raise ValueError(
                f"{filename} header/index mismatch: missing={missing[:8]}, "
                f"unexpected={unexpected[:8]}"
            )
        for name, descriptor in header.items():
            if name in actual:
                raise ValueError(f"tensor {name} occurs in multiple shards")
            dtype = descriptor["dtype"]
            shape = descriptor["shape"]
            if dtype != "F32":
                raise ValueError(f"{filename}: {name} dtype {dtype} != F32")
            expected_shape = manifest.get(name)
            if shape != expected_shape:
                raise ValueError(
                    f"{filename}: {name} shape {shape} != {expected_shape}"
                )
            actual[name] = shape  # type: ignore[assignment]
        shard_results[filename] = {
            "bytes": actual_size,
            "sha256": actual_sha256,
            "tensor_count": len(header),
        }

    if actual != manifest:
        missing = sorted(set(manifest) - set(actual))
        unexpected = sorted(set(actual) - set(manifest))
        raise ValueError(
            f"real shard manifest mismatch: missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )
    return shard_results


def validate_gguf(
    path: Path,
    manifest: dict[str, tuple[int, ...]],
) -> dict[str, object]:
    """Authenticate the converted GGUF header without loading tensor payloads."""

    # This sibling module is a zero-dependency, header-only parser. Importing
    # it keeps GGUF decoding logic in one audit implementation.
    from gguf_manifest import read_manifest

    metadata, tensors = read_manifest(path)
    for key, expected in GGUF_METADATA.items():
        actual = metadata.get(key)
        if actual != expected:
            raise ValueError(f"GGUF metadata {key}={actual!r}, expected {expected!r}")
    if len(tensors) != TENSOR_COUNT:
        raise ValueError(
            f"GGUF tensor count {len(tensors)} != pinned {TENSOR_COUNT}"
        )

    actual: dict[str, tuple[int, ...]] = {}
    offsets: set[int] = set()
    for tensor in tensors:
        name = tensor["name"]
        if not isinstance(name, str):
            raise ValueError(f"GGUF tensor has non-string name: {name!r}")
        if name in actual:
            raise ValueError(f"GGUF tensor name is duplicated: {name}")
        dimensions = tensor["dimensions"]
        if not isinstance(dimensions, list) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in dimensions
        ):
            raise ValueError(f"GGUF tensor {name} has invalid dimensions {dimensions!r}")
        shape = tuple(dimensions)
        expected_shape = manifest.get(name)
        if shape != expected_shape:
            raise ValueError(f"GGUF tensor {name} shape {shape} != {expected_shape}")
        if tensor["ggml_type"] != 0:
            raise ValueError(
                f"GGUF tensor {name} type {tensor['ggml_type']} != F32 (0)"
            )
        offset = tensor["offset"]
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError(f"GGUF tensor {name} has invalid offset {offset!r}")
        if offset in offsets:
            raise ValueError(f"GGUF tensor {name} reuses payload offset {offset}")
        offsets.add(offset)
        actual[name] = shape
    if actual != manifest:
        missing = sorted(set(manifest) - set(actual))
        unexpected = sorted(set(actual) - set(manifest))
        raise ValueError(
            f"GGUF manifest mismatch: missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )
    digest = manifest_sha256(actual)
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "tensor_count": len(actual),
        "manifest_sha256": digest,
    }


def add_weight_norm_conv1d(
    manifest: dict[str, tuple[int, ...]],
    prefix: str,
    input_dim: int,
    output_dim: int,
) -> None:
    manifest[f"{prefix}.parametrizations.weight.original0"] = (output_dim, 1, 1)
    manifest[f"{prefix}.parametrizations.weight.original1"] = (
        output_dim,
        input_dim,
        1,
    )
    manifest[f"{prefix}.bias"] = (output_dim,)


def expected_manifest() -> dict[str, tuple[int, ...]]:
    manifest: dict[str, tuple[int, ...]] = {}
    for stage in STAGES:
        prefix = f"{stage.side}.{stage.module_index}"
        manifest[f"{prefix}.input_proj.weight"] = (
            stage.model_dim,
            stage.input_dim,
        )
        manifest[f"{prefix}.output_proj.weight"] = (
            stage.output_dim,
            stage.model_dim,
        )
        for layer in range(stage.layers):
            layer_prefix = f"{prefix}.transformer.layers.{layer}"
            for norm in ("norm1", "norm2"):
                manifest[f"{layer_prefix}.{norm}.weight"] = (stage.model_dim,)
                manifest[f"{layer_prefix}.{norm}.bias"] = (stage.model_dim,)
            manifest[f"{layer_prefix}.self_attn.in_proj.weight"] = (
                stage.model_dim * 3,
                stage.model_dim,
            )
            manifest[f"{layer_prefix}.self_attn.out_proj.weight"] = (
                stage.model_dim,
                stage.model_dim,
            )
            manifest[f"{layer_prefix}.ffn.0.weight"] = (
                stage.ffn_dim,
                stage.model_dim,
            )
            manifest[f"{layer_prefix}.ffn.2.weight"] = (
                stage.model_dim,
                stage.ffn_dim,
            )
            for scale in ("layer_scale_1", "layer_scale_2"):
                manifest[f"{layer_prefix}.{scale}.scale"] = (stage.model_dim,)

    add_weight_norm_conv1d(manifest, "quantizer.input_proj", 768, 512)
    add_weight_norm_conv1d(manifest, "quantizer.output_proj", 512, 768)
    for quantizer in range(32):
        prefix = f"quantizer.quantizers.{quantizer}"
        manifest[f"{prefix}.codebook.weight"] = (1_024, 8)
        add_weight_norm_conv1d(manifest, f"{prefix}.in_proj", 512, 8)
        add_weight_norm_conv1d(manifest, f"{prefix}.out_proj", 8, 512)
    return manifest


def manifest_sha256(manifest: dict[str, tuple[int, ...]]) -> str:
    canonical = bytearray()
    for name in sorted(manifest):
        shape = manifest[name]
        canonical.extend(name.encode("utf-8"))
        canonical.append(0)
        canonical.extend(struct.pack("<Q", len(shape)))
        for dimension in shape:
            canonical.extend(struct.pack("<Q", dimension))
    return hashlib.sha256(canonical).hexdigest()


def validate_config(config: dict[str, object]) -> None:
    expected_scalars = {
        "model_type": "moss-audio-tokenizer",
        "sampling_rate": 48_000,
        "downsample_rate": 3_840,
        "number_channels": 2,
        "enable_channel_interleave": True,
        "code_dim": 768,
        "quantizer_type": "rlfq",
    }
    for key, expected in expected_scalars.items():
        actual = config.get(key)
        if actual != expected:
            raise ValueError(f"config.{key}={actual!r}, expected {expected!r}")
    quantizer = config.get("quantizer_kwargs")
    if not isinstance(quantizer, dict):
        raise ValueError("config.quantizer_kwargs is missing or not an object")
    for key, expected in {
        "input_dim": 768,
        "rvq_dim": 512,
        "output_dim": 768,
        "num_quantizers": 32,
        "codebook_size": 1_024,
        "codebook_dim": 8,
        "quantizer_type": "rlfq",
    }.items():
        actual = quantizer.get(key)
        if actual != expected:
            raise ValueError(
                f"config.quantizer_kwargs[{key!r}]={actual!r}, expected {expected!r}"
            )
    stage_by_key = {(stage.side, stage.module_index): stage for stage in STAGES}
    for side in ("encoder", "decoder"):
        modules = config.get(f"{side}_kwargs")
        if not isinstance(modules, list) or len(modules) != 12:
            raise ValueError(f"config.{side}_kwargs must contain 12 modules")
        for module_index, module in enumerate(modules):
            if not isinstance(module, dict):
                raise ValueError(
                    f"config.{side}_kwargs[{module_index}] is not an object"
                )
            stage = stage_by_key.get((side, module_index))
            if stage is None:
                expected_patch = 240 if module_index in {0, 11} else 2
                if module.get("module_type") != "PatchedPretransform":
                    raise ValueError(
                        f"config.{side}_kwargs[{module_index}] is not a patch module"
                    )
                if module.get("patch_size") != expected_patch:
                    raise ValueError(
                        f"config.{side}_kwargs[{module_index}].patch_size="
                        f"{module.get('patch_size')!r}, expected {expected_patch}"
                    )
                continue
            for key, expected in {
                "module_type": "Transformer",
                "input_dimension": stage.input_dim,
                "output_dimension": stage.output_dim,
                "d_model": stage.model_dim,
                "dim_feedforward": stage.ffn_dim,
                "num_layers": stage.layers,
            }.items():
                actual = module.get(key)
                if actual != expected:
                    raise ValueError(
                        f"config.{side}_kwargs[{module_index}][{key!r}]="
                        f"{actual!r}, expected {expected!r}"
                    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument(
        "--shard-dir",
        type=Path,
        help=(
            "directory containing the three pinned real safetensors shards; "
            "required to confirm the candidate manifest (VAST-only)"
        ),
    )
    parser.add_argument(
        "--gguf",
        type=Path,
        help=(
            "converted v2 GGUF to authenticate against the confirmed real "
            "shards; requires --shard-dir"
        ),
    )
    args = parser.parse_args()
    if args.gguf is not None and args.shard_dir is None:
        parser.error("--gguf requires --shard-dir so conversion is never confirmed from GGUF alone")

    for path, expected, label in (
        (args.config, CONFIG_SHA256, "config"),
        (args.index, INDEX_SHA256, "index"),
    ):
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"{label} SHA-256 {actual} != {expected} at revision {REVISION}"
            )
    config = json.loads(args.config.read_text(encoding="utf-8"))
    index = json.loads(args.index.read_text(encoding="utf-8"))
    validate_config(config)

    manifest = expected_manifest()
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("index.weight_map is missing or not an object")
    expected_names = set(manifest)
    actual_names = set(weight_map)
    if expected_names != actual_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise ValueError(
            f"index name mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    if len(manifest) != TENSOR_COUNT:
        raise ValueError(f"derived {len(manifest)} tensors, expected {TENSOR_COUNT}")
    parameters = sum(
        dimension_product(shape) for shape in manifest.values()
    )
    metadata = index.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("index.metadata is missing or not an object")
    if metadata.get("total_parameters") != PARAMETER_COUNT:
        raise ValueError(
            "index total_parameters does not match the pinned public contract"
        )
    if parameters != PARAMETER_COUNT:
        raise ValueError(
            f"derived {parameters} parameters, expected {PARAMETER_COUNT}"
        )
    total_size = metadata.get("total_size")
    if total_size != PARAMETER_COUNT * 4:
        raise ValueError(
            f"index total_size={total_size!r}, expected {PARAMETER_COUNT * 4}"
        )

    candidate_sha256 = manifest_sha256(manifest)
    result: dict[str, object] = {
        "revision": REVISION,
        "tensor_count": len(manifest),
        "parameter_count": parameters,
        "tensor_bytes_f32": total_size,
        "manifest_sha256_candidate": candidate_sha256,
    }
    if args.shard_dir is None:
        result["requires_vast_header_confirmation"] = True
    else:
        if not args.shard_dir.is_dir():
            raise ValueError(f"--shard-dir is not a directory: {args.shard_dir}")
        shards = validate_real_shards(args.shard_dir, weight_map, manifest)
        confirmed_sha256 = manifest_sha256(manifest)
        if confirmed_sha256 != candidate_sha256:
            raise ValueError(
                f"confirmed manifest {confirmed_sha256} != candidate {candidate_sha256}"
            )
        result.update(
            {
                "manifest_sha256": confirmed_sha256,
                "requires_vast_header_confirmation": False,
                "shards": shards,
            }
        )
        if args.gguf is not None:
            if not args.gguf.is_file():
                raise ValueError(f"--gguf is not a file: {args.gguf}")
            gguf = validate_gguf(args.gguf, manifest)
            if gguf["manifest_sha256"] != confirmed_sha256:
                raise ValueError(
                    f"GGUF manifest {gguf['manifest_sha256']} != confirmed "
                    f"safetensors manifest {confirmed_sha256}"
                )
            result["gguf"] = gguf
    print(json.dumps(result, sort_keys=True))
    return 0


def dimension_product(shape: tuple[int, ...]) -> int:
    product = 1
    for dimension in shape:
        product *= dimension
    return product


if __name__ == "__main__":
    raise SystemExit(main())
