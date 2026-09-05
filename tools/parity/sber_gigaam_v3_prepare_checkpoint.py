#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Prepare only the authenticated GigaAM v3 checkpoint on VAST.

The source is opened with ``weights_only=True`` and every authenticated
tensor name, shape, and dtype is checked before an absent safetensors output
or sidecar is created.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

HF_REPOSITORY = "ai-sage/GigaAM-v3"
HF_REVISION = "ec1dc1f01d0d627ab2c0d3acc1e235702300d95e"
SOURCE_REVISION = "7447938d791c4f3e643386ee22c33777004293a5"
CONFIG_SHA256 = "02361ba9cafd6c3ec66fcdd73494c3b562a60eb2a2d1b13f3cb04ae440d93e52"
MODELING_SHA256 = "269be43b635b1e510115baa2a843c5cbaa052e8adf0be30dc133a2ba5b5f2d86"
TOKENIZER_SHA256 = "828c12c991019eef952a960661f25a92d6ad279591e2ea466b4aeddf1d20a18a"
CHECKPOINT_BYTES = 448_928_167
CHECKPOINT_SHA256 = "afc6dcbae8320ea56f2cddebc0f13fbf62c9d59b6ddcad899782623c8610826a"
PREPARED_FORMAT = "vokra-gigaam-v3-prepared-v1"


def manifest() -> list[tuple[str, list[int], str]]:
    rows = [("model.preprocessor.featurizer.0.spectrogram.window", [320], "F32"), ("model.preprocessor.featurizer.0.mel_scale.fb", [161, 64], "F32"), ("model.encoder.pre_encode.conv.0.weight", [768, 64, 5], "F16"), ("model.encoder.pre_encode.conv.0.bias", [768], "F16"), ("model.encoder.pre_encode.conv.2.weight", [768, 768, 5], "F16"), ("model.encoder.pre_encode.conv.2.bias", [768], "F16")]
    for layer in range(16):
        prefix = f"model.encoder.layers.{layer}"
        for name in ("norm_feed_forward1", "norm_conv", "norm_self_att", "norm_feed_forward2", "norm_out"):
            rows.extend(((f"{prefix}.{name}.weight", [768], "F16"), (f"{prefix}.{name}.bias", [768], "F16")))
        for branch in ("feed_forward1", "feed_forward2"):
            rows.extend(((f"{prefix}.{branch}.linear1.weight", [3072, 768], "F16"), (f"{prefix}.{branch}.linear1.bias", [3072], "F16"), (f"{prefix}.{branch}.linear2.weight", [768, 3072], "F16"), (f"{prefix}.{branch}.linear2.bias", [768], "F16")))
        rows.extend(((f"{prefix}.conv.pointwise_conv1.weight", [1536, 768, 1], "F16"), (f"{prefix}.conv.pointwise_conv1.bias", [1536], "F16"), (f"{prefix}.conv.depthwise_conv.weight", [768, 1, 5], "F16"), (f"{prefix}.conv.depthwise_conv.bias", [768], "F16"), (f"{prefix}.conv.batch_norm.weight", [768], "F16"), (f"{prefix}.conv.batch_norm.bias", [768], "F16"), (f"{prefix}.conv.pointwise_conv2.weight", [768, 768, 1], "F16"), (f"{prefix}.conv.pointwise_conv2.bias", [768], "F16")))
        for name in ("linear_q", "linear_k", "linear_v", "linear_out"):
            rows.extend(((f"{prefix}.self_attn.{name}.weight", [768, 768], "F16"), (f"{prefix}.self_attn.{name}.bias", [768], "F16")))
    rows.extend((("model.head.decoder.embed.weight", [1025, 320], "F32"), ("model.head.decoder.lstm.weight_ih_l0", [1280, 320], "F32"), ("model.head.decoder.lstm.weight_hh_l0", [1280, 320], "F32"), ("model.head.decoder.lstm.bias_ih_l0", [1280], "F32"), ("model.head.decoder.lstm.bias_hh_l0", [1280], "F32"), ("model.head.joint.pred.weight", [320, 320], "F32"), ("model.head.joint.pred.bias", [320], "F32"), ("model.head.joint.enc.weight", [320, 768], "F32"), ("model.head.joint.enc.bias", [320], "F32"), ("model.head.joint.joint_net.1.weight", [1025, 320], "F32"), ("model.head.joint.joint_net.1.bias", [1025], "F32")))
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject(path: Path, label: str) -> None:
    absolute = Path(os.path.abspath(path))
    if any(ancestor.is_symlink() for ancestor in (absolute, *absolute.parents)):
        raise SystemExit(f"{label} has symlink ancestry")


def flatten(value: object, prefix: str = "") -> dict[str, object]:
    import torch
    if isinstance(value, torch.Tensor):
        return {prefix: value}
    if not isinstance(value, dict):
        return {}
    output: dict[str, object] = {}
    for key, child in value.items():
        if not isinstance(key, str) or not key or "/" in key or "\\" in key or ".." in Path(key).parts:
            raise SystemExit(f"unsafe checkpoint key: {key!r}")
        name = f"{prefix}.{key}" if prefix else key
        nested = flatten(child, name)
        if set(output).intersection(nested):
            raise SystemExit("flattened-name collision")
        output.update(nested)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        assert len(manifest()) == 561 and len(CHECKPOINT_SHA256) == 64
        print("sber_gigaam_v3_prepare_checkpoint self-test: OK")
        return 0
    if not args.input or not args.output:
        parser.error("--input and --output are required")
    source, output = Path(args.input), Path(args.output)
    sidecar = output.with_suffix(".manifest.json")
    for path, label in ((source, "checkpoint"), (output, "prepared output"), (sidecar, "sidecar")):
        reject(path, label)
    if source.name != "pytorch_model.bin" or not source.is_file() or source.is_symlink() or output.exists() or sidecar.exists():
        raise SystemExit("checkpoint must be regular and outputs must be absent")
    if source.stat().st_size != CHECKPOINT_BYTES or sha256(source) != CHECKPOINT_SHA256:
        raise SystemExit("fixed checkpoint size/SHA-256 mismatch")
    import torch
    from safetensors.torch import save_file
    tensors = flatten(torch.load(source, map_location="cpu", weights_only=True))
    expected = manifest()
    expected_names = [name for name, _, _ in expected]
    if set(tensors) != set(expected_names) or len(tensors) != len(expected_names):
        raise SystemExit("checkpoint tensor name set mismatch")
    prepared: dict[str, torch.Tensor] = {}
    rows = []
    for name, shape, dtype in expected:
        tensor = tensors[name]
        wanted = torch.float16 if dtype == "F16" else torch.float32
        if tensor.dtype != wanted or list(tensor.shape) != shape or not bool(torch.isfinite(tensor).all().item()):
            raise SystemExit(f"tensor {name} dtype/shape/finite mismatch")
        prepared[name] = tensor.contiguous()
        rows.append({"name": name, "shape": shape, "dtype": dtype})
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(prepared, str(output))
    sidecar.write_text(json.dumps({"format": PREPARED_FORMAT, "repository": HF_REPOSITORY, "revision": HF_REVISION, "source_revision": SOURCE_REVISION, "config_sha256": CONFIG_SHA256, "modeling_sha256": MODELING_SHA256, "tokenizer_sha256": TOKENIZER_SHA256, "checkpoint_sha256": CHECKPOINT_SHA256, "prepared_sha256": sha256(output), "tensor_count": len(rows), "tensors": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"prepared {len(rows)} tensors: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
