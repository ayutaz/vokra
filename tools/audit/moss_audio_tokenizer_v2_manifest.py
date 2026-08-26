#!/usr/bin/env python3
"""Derive the fixed MOSS Audio Tokenizer v2 tensor manifest from small files.

This audit reads only ``config.json`` and ``model.safetensors.index.json`` at
revision ``f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169``. It does not download or
open any weight shard. The derived digest is a candidate runtime contract and
must still be compared with the real safetensors/GGUF header on VAST before it
is pinned in Rust.
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
    args = parser.parse_args()

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

    result = {
        "revision": REVISION,
        "tensor_count": len(manifest),
        "parameter_count": parameters,
        "tensor_bytes_f32": total_size,
        "manifest_sha256_candidate": manifest_sha256(manifest),
        "requires_vast_header_confirmation": True,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


def dimension_product(shape: tuple[int, ...]) -> int:
    product = 1
    for dimension in shape:
        product *= dimension
    return product


if __name__ == "__main__":
    raise SystemExit(main())
