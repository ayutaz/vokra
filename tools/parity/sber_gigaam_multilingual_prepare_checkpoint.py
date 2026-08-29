#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Prepare the authenticated GigaAM Multilingual CTC checkpoint on VAST only.

The input is accepted only when its SHA-256 and complete tensor manifest match
the fixed evidence. The output is a flat F32 safetensors file plus the exact
sidecar consumed by the Rust converter. This script must not be run on the
maintainer Mac; the worker invokes it after the VAST evidence gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HF_REPOSITORY = "ai-sage/GigaAM-Multilingual"
HF_REVISION = "2f8a57144e6ec3adfd32fe0484d9ea9913305bc8"
INSPECTION_STATUS = "INSPECTION_ONLY"
CHECKPOINT_SHA256 = "e1db43873ec5e296f229572e06e2470fc157ac9f8d4aacabda295630b9b91728"
SOURCE_REVISION = "7447938d791c4f3e643386ee22c33777004293a5"
CONFIG_SHA256 = "c830232c7d51688a630a221517b52585ab5ee57e1d3c21bcbae01759351d2653"
PREPARED_FORMAT = "vokra-gigaam-multilingual-prepared-v1"


def expected_manifest() -> list[tuple[str, list[int]]]:
    rows = [
        ("model.preprocessor.featurizer.0.spectrogram.window", [320]),
        ("model.preprocessor.featurizer.0.mel_scale.fb", [161, 64]),
        ("model.encoder.pre_encode.conv.0.weight", [768, 64, 5]),
        ("model.encoder.pre_encode.conv.0.bias", [768]),
        ("model.encoder.pre_encode.conv.2.weight", [768, 768, 5]),
        ("model.encoder.pre_encode.conv.2.bias", [768]),
    ]
    for layer in range(16):
        prefix = f"model.encoder.layers.{layer}"
        for name in ("norm_feed_forward1", "norm_conv", "norm_self_att", "norm_feed_forward2", "norm_out"):
            rows.extend(((f"{prefix}.{name}.weight", [768]), (f"{prefix}.{name}.bias", [768])))
        for branch in ("feed_forward1", "feed_forward2"):
            rows.extend(((f"{prefix}.{branch}.linear1.weight", [3072, 768]), (f"{prefix}.{branch}.linear1.bias", [3072])))
            rows.extend(((f"{prefix}.{branch}.linear2.weight", [768, 3072]), (f"{prefix}.{branch}.linear2.bias", [768])))
        rows.extend(((f"{prefix}.conv.pointwise_conv1.weight", [1536, 768, 1]), (f"{prefix}.conv.pointwise_conv1.bias", [1536])))
        rows.extend(((f"{prefix}.conv.depthwise_conv.weight", [768, 1, 5]), (f"{prefix}.conv.depthwise_conv.bias", [768])))
        rows.extend(((f"{prefix}.conv.batch_norm.weight", [768]), (f"{prefix}.conv.batch_norm.bias", [768])))
        rows.extend(((f"{prefix}.conv.pointwise_conv2.weight", [768, 768, 1]), (f"{prefix}.conv.pointwise_conv2.bias", [768])))
        for name in ("linear_q", "linear_k", "linear_v", "linear_out"):
            rows.extend(((f"{prefix}.self_attn.{name}.weight", [768, 768]), (f"{prefix}.self_attn.{name}.bias", [768])))
    rows.extend((("model.head.decoder_layers.0.weight", [71, 768, 1]), ("model.head.decoder_layers.0.bias", [71])))
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_symlink_ancestry(path: Path, label: str) -> None:
    current = Path(os.path.abspath(path))
    for ancestor in (current, *current.parents):
        if ancestor.is_symlink():
            raise RuntimeError(f"{label} has symlink ancestry: {ancestor}")


def require_absent_output(path: Path, label: str) -> None:
    reject_symlink_ancestry(path, label)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"{label} must be absent and non-symlink: {path}")


def flatten(value: object, prefix: str = "") -> dict[str, object]:
    # torch tensors are intentionally tested by duck typing only after torch is
    # imported in main; nested checkpoint metadata is never copied.
    import torch

    if isinstance(value, torch.Tensor):
        return {prefix: value}
    if isinstance(value, dict):
        result: dict[str, object] = {}
        for key, child in value.items():
            if not isinstance(key, str) or not key or "/" in key or "\\" in key or ".." in Path(key).parts:
                raise RuntimeError(f"unsafe checkpoint key: {key!r}")
            child_prefix = f"{prefix}.{key}" if prefix else key
            result.update(flatten(child, child_prefix))
        return result
    if isinstance(value, (list, tuple)):
        result = {}
        for index, child in enumerate(value):
            result.update(flatten(child, f"{prefix}[{index}]"))
        return result
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refuse unauthenticated GigaAM Multilingual preparation"
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.self_test:
        if args.input is not None or args.output is not None:
            parser.error("--self-test accepts no input or output")
        assert len(HF_REVISION) == 40 and len(CHECKPOINT_SHA256) == 64
        assert len(CONFIG_SHA256) == 64
        assert len(expected_manifest()) == 552
        print("sber_gigaam_multilingual_prepare_checkpoint self-test: OK")
        return 0
    if not args.input or not args.output:
        parser.error("--input and --output are required outside --self-test")
    source = Path(args.input)
    output = Path(args.output)
    sidecar = output.with_suffix(".manifest.json")
    reject_symlink_ancestry(source, "checkpoint")
    if source.is_symlink() or not source.is_file():
        print(f"missing checkpoint: {source}", file=sys.stderr)
        return 2
    require_absent_output(output, "prepared safetensors")
    require_absent_output(sidecar, "prepared manifest")
    source_real = source.resolve()
    output_real = output.resolve(strict=False)
    sidecar_real = sidecar.resolve(strict=False)
    if source_real in (output_real, sidecar_real) or output_real == sidecar_real:
        print("checkpoint, prepared output, and sidecar must not overlap", file=sys.stderr)
        return 2
    if sha256(source) != CHECKPOINT_SHA256:
        print("checkpoint SHA-256 mismatch; refusing preparation", file=sys.stderr)
        return 2

    import torch
    from safetensors.torch import save_file

    value = torch.load(source, map_location="cpu", weights_only=True)
    tensors = flatten(value)
    expected = expected_manifest()
    expected_names = {name for name, _ in expected}
    if set(tensors) != expected_names:
        missing = sorted(expected_names - set(tensors))[:4]
        extra = sorted(set(tensors) - expected_names)[:4]
        raise RuntimeError(f"checkpoint tensor manifest mismatch: missing={missing}, extra={extra}")
    prepared: dict[str, torch.Tensor] = {}
    sidecar_tensors = []
    for name, shape in expected:
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32 or list(tensor.shape) != shape:
            raise RuntimeError(f"tensor {name} must be F32 {shape}")
        if not bool(torch.isfinite(tensor).all().item()):
            raise RuntimeError(f"tensor {name} contains non-finite values")
        prepared[name] = tensor.contiguous()
        sidecar_tensors.append({"name": name, "shape": shape, "dtype": "F32"})

    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(prepared, str(output))
    prepared_sha256 = sha256(output)
    sidecar.write_text(json.dumps({
        "format": PREPARED_FORMAT,
        "repository": HF_REPOSITORY,
        "revision": HF_REVISION,
        "source_revision": SOURCE_REVISION,
        "config_sha256": CONFIG_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "prepared_sha256": prepared_sha256,
        "tensor_count": len(sidecar_tensors),
        "tensors": sidecar_tensors,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"prepared {len(prepared)} tensors: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
