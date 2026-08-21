#!/usr/bin/env python3
"""Prepare the official SpeechBrain LibriTTS HiFi-GAN checkpoint.

The upstream release stores 78 weight-normalized convolution modules as
``bias`` / ``weight_g`` / ``weight_v`` triples in a torch-pickle checkpoint.
Vokra never parses pickle at runtime. This offline sidecar pins the upstream
revision and file hashes, validates all 234 source tensor names and shapes,
folds weight normalization with the PyTorch ``dim=0`` contract, and writes the
exact 156 effective ``bias`` / ``weight`` tensors consumed by the Rust binder.

Run only through the repository's Python 3.12 parity environment::

    uv run --python 3.12 python hifigan_prepare_checkpoint.py \
      --output-dir /tmp/speechbrain-hifigan
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

UPSTREAM_HF = "speechbrain/tts-hifigan-libritts-22050Hz"
UPSTREAM_REVISION = "4188503131602dc234f48d7f22eebea93d788736"
CHECKPOINT_SHA256 = "db0d1249e2c957dca1021749c43334b9c3190664d7c7e386c5c16bef62fd1574"
HYPERPARAMS_SHA256 = "8a7d1fb3eb8f0c979961c7708e4c7182ec3c046cc71c3eb91145996669da9535"

N_MELS = 80
INITIAL_CHANNEL = 512
UPSAMPLE_FACTORS = (8, 8, 2, 2)
UPSAMPLE_KERNELS = (16, 16, 4, 4)
RESBLOCK_KERNELS = (3, 7, 11)
RESBLOCK_DILATIONS = (1, 3, 5)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_shapes() -> dict[str, tuple[int, ...]]:
    modules: dict[str, tuple[int, ...]] = {
        "conv_pre": (INITIAL_CHANNEL, N_MELS, 7),
    }
    for stage, kernel in enumerate(UPSAMPLE_KERNELS):
        in_channels = INITIAL_CHANNEL >> stage
        out_channels = INITIAL_CHANNEL >> (stage + 1)
        modules[f"ups.{stage}"] = (in_channels, out_channels, kernel)
        for branch, res_kernel in enumerate(RESBLOCK_KERNELS):
            block = stage * len(RESBLOCK_KERNELS) + branch
            for layer, _dilation in enumerate(RESBLOCK_DILATIONS):
                modules[f"resblocks.{block}.convs1.{layer}"] = (
                    out_channels,
                    out_channels,
                    res_kernel,
                )
                modules[f"resblocks.{block}.convs2.{layer}"] = (
                    out_channels,
                    out_channels,
                    res_kernel,
                )
    modules["conv_post"] = (1, INITIAL_CHANNEL >> len(UPSAMPLE_FACTORS), 7)
    assert len(modules) == 78
    return modules


def source_name(effective_name: str, suffix: str) -> str:
    return f"{effective_name}.conv.{suffix}"


def require_tensor(
    state: Mapping[str, object], name: str, shape: tuple[int, ...]
) -> torch.Tensor:
    value = state.get(name)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"required tensor {name!r} is missing or not a tensor")
    if tuple(value.shape) != shape:
        raise RuntimeError(
            f"tensor {name!r} shape {tuple(value.shape)} != expected {shape}"
        )
    if not value.is_floating_point():
        raise RuntimeError(f"tensor {name!r} must be floating point, got {value.dtype}")
    return value.detach().to(dtype=torch.float32, device="cpu").contiguous()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--revision", default=UPSTREAM_REVISION)
    parser.add_argument("--out-basename", default="model.safetensors")
    args = parser.parse_args()

    if args.revision != UPSTREAM_REVISION:
        parser.error(
            "--revision must equal the audited commit "
            f"{UPSTREAM_REVISION}; update the pinned hashes in this script first"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=UPSTREAM_HF,
        revision=args.revision,
        local_dir=args.output_dir,
        allow_patterns=["generator.ckpt", "hyperparams.yaml", "README.md"],
    )

    checkpoint = args.output_dir / "generator.ckpt"
    hyperparams = args.output_dir / "hyperparams.yaml"
    for path, expected in (
        (checkpoint, CHECKPOINT_SHA256),
        (hyperparams, HYPERPARAMS_SHA256),
    ):
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"{path.name} sha256 {actual} != audited upstream hash {expected}"
            )

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise RuntimeError(f"checkpoint root must be a mapping, got {type(state)!r}")

    modules = module_shapes()
    expected_source_names: set[str] = set()
    folded: dict[str, torch.Tensor] = {}
    for name, weight_shape in modules.items():
        bias_shape = (weight_shape[1],) if name.startswith("ups.") else (weight_shape[0],)
        g_shape = (weight_shape[0],) + (1,) * (len(weight_shape) - 1)
        bias_name = source_name(name, "bias")
        g_name = source_name(name, "weight_g")
        v_name = source_name(name, "weight_v")
        expected_source_names.update((bias_name, g_name, v_name))

        bias = require_tensor(state, bias_name, bias_shape)
        g = require_tensor(state, g_name, g_shape)
        v = require_tensor(state, v_name, weight_shape)
        norm_dims = tuple(range(1, v.ndim))
        denominator = torch.linalg.vector_norm(v, dim=norm_dims, keepdim=True)
        if not torch.isfinite(denominator).all() or torch.any(denominator == 0):
            raise RuntimeError(f"{v_name!r} has a zero or non-finite dim=0 norm")
        weight = (v * (g / denominator)).contiguous()
        if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
            raise RuntimeError(f"folded module {name!r} contains non-finite values")
        folded[f"{name}.weight"] = weight
        folded[f"{name}.bias"] = bias

    actual_names = set(state)
    if actual_names != expected_source_names:
        missing = sorted(expected_source_names - actual_names)[:8]
        extra = sorted(actual_names - expected_source_names)[:8]
        raise RuntimeError(
            "checkpoint tensor manifest mismatch: "
            f"expected={len(expected_source_names)} actual={len(actual_names)} "
            f"missing={missing} extra={extra}"
        )
    if len(folded) != 156:
        raise RuntimeError(f"internal error: folded {len(folded)} tensors, expected 156")

    output = args.output_dir / args.out_basename
    save_file(
        folded,
        output,
        metadata={
            "source": UPSTREAM_HF,
            "revision": UPSTREAM_REVISION,
            "generator.ckpt.sha256": CHECKPOINT_SHA256,
            "hyperparams.yaml.sha256": HYPERPARAMS_SHA256,
            "weight_norm": "folded dim=0 using float32 vector norm",
        },
    )
    print(
        "hifigan_prepare: "
        f"source_tensors={len(actual_names)} folded_tensors={len(folded)} "
        f"output={output} sha256={sha256(output)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
