#!/usr/bin/env python3
"""Prepare a decoder-only NVIDIA NeMo NanoCodec checkpoint for Vokra.

The upstream ``.nemo`` archive contains a torch-pickle checkpoint.  This
uv-managed, offline sidecar is the sole pickle/NeMo boundary: it restores the
model with the pinned official NeMo package, reads topology from the restored
objects, materializes weight normalization, expands grouped ConvTranspose1d
weights to dense PyTorch layout, and emits:

* canonical F32 decoder tensors in safetensors; and
* JSON containing only checkpoint-derived geometry and immutable provenance.

The Rust converter consumes those two files without Python, torch, pickle,
ONNX, protobuf, or any third-party runtime dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
from pathlib import Path
from typing import NoReturn

import torch
from nemo.collections.tts.models.audio_codec import AudioCodecModel
from safetensors.torch import save_file


NEMO_SPEECH_COMMIT = "4fcff72febec9395fdbd4bfa0747bfda2ecd3cef"
FORMAT_VERSION = 1
AUDITED_REVISIONS = {
    "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps": "5c8e22ed763c14d81337fbe6ca74062f3d10f7e5",
    "nvidia/nemo-nano-codec-22khz-1.78kbps-12.5fps": "c4ab84a92c8d36a8b5a79eaea807cfaf7f03ed86",
    "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps": "fc00890b604aa2de298d2641ffc6c5f6caf8c4d7",
}
AUDITED_CHECKPOINT_SHA256 = {
    "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps": "bd5883099d0c74ceda760b6b7a1600b86da4d8a02531c9c282679951dcb08870",
}
AUDITED_GEOMETRY = {
    "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps": {
        "sample_rate": 22_050,
        "frame_hop": 1764,
        "generator_hop": 1764,
        "n_codebooks": 4,
        "levels_per_group": [9, 8, 8, 7],
        "embed_dim": 16,
        "base_channels": 864,
        "upsample_rates": [7, 7, 6, 3, 2],
    },
    "nvidia/nemo-nano-codec-22khz-1.78kbps-12.5fps": {
        "sample_rate": 22_050,
        "frame_hop": 1764,
        "generator_hop": 1764,
        "n_codebooks": 13,
        "levels_per_group": [8, 7, 6, 6],
        "embed_dim": 52,
        "base_channels": 864,
        "upsample_rates": [7, 7, 6, 3, 2],
    },
    "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps": {
        "sample_rate": 22_050,
        "frame_hop": 1024,
        "generator_hop": 1024,
        "n_codebooks": 8,
        "levels_per_group": [8, 7, 6, 6],
        "embed_dim": 32,
        "base_channels": 864,
        "upsample_rates": [8, 8, 4, 2, 2],
    },
}
FULL_REVISION = re.compile(r"[0-9a-f]{40}")


def fail(message: str) -> NoReturn:
    raise RuntimeError(f"nanocodec prepare: {message}")


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_nemo_source() -> str:
    distribution = importlib.metadata.distribution("nemo_toolkit")
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        fail(
            "nemo_toolkit has no PEP 610 direct_url.json; refusing an unverified source install"
        )
    direct_url = json.loads(direct_url_text)
    source_url = str(direct_url.get("url", ""))
    vcs_info = direct_url.get("vcs_info")
    commit = vcs_info.get("commit_id") if isinstance(vcs_info, dict) else None
    if source_url != "https://github.com/NVIDIA-NeMo/Speech.git":
        fail(f"nemo_toolkit source is not official NVIDIA-NeMo/Speech: {source_url!r}")
    if commit != NEMO_SPEECH_COMMIT:
        fail(f"nemo_toolkit commit {commit!r} != pinned {NEMO_SPEECH_COMMIT}")
    return source_url


def validate_audited_geometry(model_id: str, geometry: dict) -> None:
    expected = AUDITED_GEOMETRY[model_id]
    mismatches = [
        f"{key}={geometry.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if geometry.get(key) != value
    ]
    if mismatches:
        fail(
            f"checkpoint geometry does not match audited profile {model_id}: "
            + "; ".join(mismatches)
        )


def f32(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        fail(f"{name} is not a torch.Tensor")
    result = tensor.detach().cpu().to(torch.float32).contiguous()
    if not torch.isfinite(result).all().item():
        fail(f"{name} contains a non-finite value")
    return result


def add_tensor(
    tensors: dict[str, torch.Tensor], name: str, tensor: torch.Tensor
) -> None:
    if name in tensors:
        fail(f"duplicate canonical tensor {name}")
    tensors[name] = f32(tensor, name=name)


def add_conv(
    tensors: dict[str, torch.Tensor], prefix: str, module: object
) -> None:
    conv = getattr(module, "conv", None)
    if conv is None or conv.__class__.__name__ != "ParametrizedConv1d":
        fail(
            f"{prefix} expected weight-normalized ParametrizedConv1d, got "
            f"{type(conv).__module__}.{type(conv).__name__}"
        )
    if conv.bias is None:
        fail(f"{prefix} convolution has no bias")
    # ``conv.weight`` is the official torch parametrization's effective
    # weight.  Reading it folds weight norm without mirroring its internals.
    add_tensor(tensors, f"{prefix}.weight", conv.weight)
    add_tensor(tensors, f"{prefix}.bias", conv.bias)


def half_snake(module: object, *, context: str) -> tuple[torch.Tensor, torch.Tensor]:
    activation = getattr(module, "activation", None)
    if activation is None or activation.__class__.__name__ != "HalfSnake":
        fail(
            f"{context} expected HalfSnake, got "
            f"{type(activation).__module__}.{type(activation).__name__}"
        )
    snake = getattr(activation, "snake_act", None)
    alpha = getattr(snake, "alpha", None)
    if alpha is None:
        fail(f"{context} HalfSnake has no snake_act.alpha")
    alpha = f32(alpha, name=f"{context}.alpha").reshape(-1)
    alpha_inv = 1.0 / (alpha + 1.0e-9)
    if not torch.isfinite(alpha_inv).all().item():
        fail(f"{context} produces non-finite 1/(alpha+1e-9)")
    return alpha, alpha_inv.contiguous()


def add_half_snake(
    tensors: dict[str, torch.Tensor], prefix: str, module: object
) -> None:
    alpha, alpha_inv = half_snake(module, context=prefix)
    add_tensor(tensors, f"{prefix}.alpha", alpha)
    add_tensor(tensors, f"{prefix}.alpha_inv", alpha_inv)


def dense_grouped_conv_transpose(module: object, *, context: str) -> torch.Tensor:
    conv = getattr(module, "conv", None)
    if conv is None or conv.__class__.__name__ != "ParametrizedConvTranspose1d":
        fail(
            f"{context} expected weight-normalized ParametrizedConvTranspose1d, got "
            f"{type(conv).__module__}.{type(conv).__name__}"
        )
    weight = f32(conv.weight, name=f"{context}.weight")
    in_channels = int(conv.in_channels)
    out_channels = int(conv.out_channels)
    groups = int(conv.groups)
    if groups != out_channels:
        fail(f"{context} groups {groups} != out_channels {out_channels}")
    if in_channels % groups != 0 or tuple(weight.shape[1:2]) != (1,):
        fail(
            f"{context} unexpected grouped shape {tuple(weight.shape)} for "
            f"in={in_channels}, out={out_channels}, groups={groups}"
        )
    kernel = int(weight.shape[2])
    inputs_per_group = in_channels // groups
    dense = torch.zeros((in_channels, out_channels, kernel), dtype=torch.float32)
    for input_channel in range(in_channels):
        output_channel = input_channel // inputs_per_group
        dense[input_channel, output_channel, :] = weight[input_channel, 0, :]
    return dense.contiguous()


def checked_uniform(values: list[list[int]], *, context: str) -> list[int]:
    if not values:
        fail(f"{context} is empty")
    first = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value != first:
            fail(f"{context}[{index}] {value} disagrees with {first}")
    return first


def quantizer_geometry(model: AudioCodecModel) -> tuple[int, list[int], int]:
    quantizer = getattr(model, "vector_quantizer", None)
    if quantizer is None or quantizer.__class__.__name__ != "GroupFiniteScalarQuantizer":
        fail(
            "expected vector_quantizer GroupFiniteScalarQuantizer, got "
            f"{type(quantizer).__module__}.{type(quantizer).__name__}"
        )
    fsqs = list(getattr(quantizer, "fsqs", ()))
    n_codebooks = int(getattr(quantizer, "num_codebooks"))
    if n_codebooks <= 0 or len(fsqs) != n_codebooks:
        fail(
            f"quantizer num_codebooks {n_codebooks} disagrees with "
            f"{len(fsqs)} FSQ modules"
        )
    level_sets = [
        [int(value) for value in fsq.num_levels.detach().cpu().reshape(-1).tolist()]
        for fsq in fsqs
    ]
    levels_per_group = checked_uniform(level_sets, context="FSQ levels")
    if not levels_per_group or any(level < 2 for level in levels_per_group):
        fail(f"invalid FSQ levels {levels_per_group}")
    embed_dim = int(getattr(quantizer, "codebook_dim"))
    expected_embed_dim = n_codebooks * len(levels_per_group)
    if embed_dim != expected_embed_dim:
        fail(
            f"quantizer codebook_dim {embed_dim} != groups {n_codebooks} * "
            f"levels width {len(levels_per_group)}"
        )
    return n_codebooks, levels_per_group, embed_dim


def prepare_decoder(model: AudioCodecModel) -> tuple[dict[str, torch.Tensor], dict]:
    decoder = getattr(model, "audio_decoder", None)
    if decoder is None or decoder.__class__.__name__ != "CausalHiFiGANDecoder":
        fail(
            f"expected CausalHiFiGANDecoder, got "
            f"{type(decoder).__module__}.{type(decoder).__name__}"
        )
    if decoder.out_activation.__class__.__name__ != "ClampActivation":
        fail(
            f"expected ClampActivation, got {decoder.out_activation.__class__.__name__}"
        )

    pre = decoder.pre_conv.conv
    post = decoder.post_conv.conv
    if pre.padding_mode != "zeros" or post.padding_mode != "zeros":
        fail(
            f"expected causal zero padding, got pre={pre.padding_mode}, "
            f"post={post.padding_mode}"
        )
    upsample_rates = [int(rate) for rate in decoder.up_sample_rates]
    if not upsample_rates or any(rate <= 0 for rate in upsample_rates):
        fail(f"invalid upsample rates {upsample_rates}")
    if not (
        len(decoder.activations)
        == len(decoder.up_sample_conv_layers)
        == len(decoder.res_layers)
        == len(upsample_rates)
    ):
        fail("decoder stage lists have inconsistent lengths")

    if not decoder.res_layers:
        fail("decoder has no residual layers")
    first_layer = decoder.res_layers[0]
    if not first_layer.res_blocks:
        fail("decoder has no residual kernel branches")
    resblock_kernel_sizes = [
        int(branch.res_blocks[0].input_conv.conv.kernel_size[0])
        for branch in first_layer.res_blocks
    ]
    resblock_dilations = [
        int(block.input_conv.conv.dilation[0])
        for block in first_layer.res_blocks[0].res_blocks
    ]
    if not resblock_dilations:
        fail("decoder has no residual dilation blocks")

    tensors: dict[str, torch.Tensor] = {}
    add_conv(tensors, "nanocodec.pre_conv", decoder.pre_conv)
    for stage, (activation, upsample, residual_layer) in enumerate(
        zip(
            decoder.activations,
            decoder.up_sample_conv_layers,
            decoder.res_layers,
            strict=True,
        )
    ):
        stage_prefix = f"nanocodec.stage.{stage}"
        add_half_snake(tensors, f"{stage_prefix}.activation", activation)
        dense = dense_grouped_conv_transpose(
            upsample, context=f"{stage_prefix}.upsample"
        )
        add_tensor(tensors, f"{stage_prefix}.upsample.weight", dense)
        if upsample.conv.bias is None:
            fail(f"{stage_prefix}.upsample has no bias")
        add_tensor(
            tensors, f"{stage_prefix}.upsample.bias", upsample.conv.bias
        )
        if len(residual_layer.res_blocks) != len(resblock_kernel_sizes):
            fail(f"stage {stage} residual branch count drift")
        for branch, (branch_module, expected_kernel) in enumerate(
            zip(
                residual_layer.res_blocks,
                resblock_kernel_sizes,
                strict=True,
            )
        ):
            if len(branch_module.res_blocks) != len(resblock_dilations):
                fail(f"stage {stage} branch {branch} dilation count drift")
            for block, (block_module, expected_dilation) in enumerate(
                zip(
                    branch_module.res_blocks,
                    resblock_dilations,
                    strict=True,
                )
            ):
                actual_kernel = int(block_module.input_conv.conv.kernel_size[0])
                actual_dilation = int(block_module.input_conv.conv.dilation[0])
                skip_dilation = int(block_module.skip_conv.conv.dilation[0])
                if actual_kernel != expected_kernel:
                    fail(
                        f"stage {stage} branch {branch} block {block} kernel "
                        f"{actual_kernel} != {expected_kernel}"
                    )
                if actual_dilation != expected_dilation or skip_dilation != 1:
                    fail(
                        f"stage {stage} branch {branch} block {block} dilations "
                        f"input={actual_dilation}, skip={skip_dilation}"
                    )
                prefix = f"{stage_prefix}.branch.{branch}.block.{block}"
                add_half_snake(
                    tensors,
                    f"{prefix}.input_activation",
                    block_module.input_activation,
                )
                add_conv(tensors, f"{prefix}.input_conv", block_module.input_conv)
                add_half_snake(
                    tensors,
                    f"{prefix}.skip_activation",
                    block_module.skip_activation,
                )
                add_conv(tensors, f"{prefix}.skip_conv", block_module.skip_conv)

    add_half_snake(tensors, "nanocodec.post_activation", decoder.post_activation)
    add_conv(tensors, "nanocodec.post_conv", decoder.post_conv)

    n_codebooks, levels_per_group, embed_dim = quantizer_geometry(model)
    input_dim = int(pre.in_channels)
    if input_dim != embed_dim:
        fail(f"decoder input_dim {input_dim} != quantizer embed_dim {embed_dim}")
    frame_hop = int(getattr(model, "samples_per_frame"))
    sample_rate = int(getattr(model, "sample_rate"))
    generator_hop = math.prod(upsample_rates)

    geometry = {
        "format_version": FORMAT_VERSION,
        "target_class": decoder.__class__.__name__,
        "sample_rate": sample_rate,
        "frame_hop": frame_hop,
        "generator_hop": generator_hop,
        "n_codebooks": n_codebooks,
        "levels_per_group": levels_per_group,
        "embed_dim": embed_dim,
        "base_channels": int(pre.out_channels),
        "upsample_rates": upsample_rates,
        "input_kernel_size": int(pre.kernel_size[0]),
        "output_kernel_size": int(post.kernel_size[0]),
        "resblock_kernel_sizes": resblock_kernel_sizes,
        "resblock_dilations": resblock_dilations,
        "activation": "HalfSnake",
        "output_activation": decoder.out_activation.__class__.__name__,
        "pad_mode": pre.padding_mode,
        "grouped_upsample": True,
        "nemo_speech_commit": NEMO_SPEECH_COMMIT,
    }
    return tensors, geometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--revision",
        required=True,
        help="immutable 40-character Hugging Face repository commit",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        fail(f"checkpoint does not exist: {args.checkpoint}")
    expected_revision = AUDITED_REVISIONS.get(args.model_id)
    if expected_revision is None:
        fail(
            "model id is not one of the three audited NVIDIA 22 kHz NanoCodec "
            f"repositories: {args.model_id}"
        )
    if FULL_REVISION.fullmatch(args.revision) is None:
        fail("--revision must be a full 40-character lowercase hexadecimal commit")
    if args.revision != expected_revision:
        fail(
            f"revision {args.revision} does not match audited {expected_revision} "
            f"for {args.model_id}"
        )
    if args.output.resolve() == args.config_output.resolve():
        fail("--output and --config-output must be different files")

    actual_checkpoint_sha256 = checkpoint_sha256(args.checkpoint)
    expected_checkpoint_sha256 = AUDITED_CHECKPOINT_SHA256.get(args.model_id)
    if (
        expected_checkpoint_sha256 is not None
        and actual_checkpoint_sha256 != expected_checkpoint_sha256
    ):
        fail(
            f"checkpoint SHA-256 {actual_checkpoint_sha256} does not match audited "
            f"{expected_checkpoint_sha256} for {args.model_id}"
        )
    nemo_source_url = verify_nemo_source()

    model = AudioCodecModel.restore_from(
        restore_path=str(args.checkpoint), map_location=torch.device("cpu")
    )
    model.eval()
    with torch.inference_mode():
        tensors, config = prepare_decoder(model)
    validate_audited_geometry(args.model_id, config)
    config.update(
        {
            "source_model_id": args.model_id,
            "source_revision": args.revision,
            "checkpoint_sha256": actual_checkpoint_sha256,
            "nemo_source_url": nemo_source_url,
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.config_output.parent.mkdir(parents=True, exist_ok=True)
    tensor_tmp = args.output.with_name(f".{args.output.name}.tmp")
    config_tmp = args.config_output.with_name(f".{args.config_output.name}.tmp")
    try:
        save_file(tensors, str(tensor_tmp))
        config_tmp.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(tensor_tmp, args.output)
        os.replace(config_tmp, args.config_output)
    finally:
        tensor_tmp.unlink(missing_ok=True)
        config_tmp.unlink(missing_ok=True)

    print(
        f"wrote {args.output} ({args.output.stat().st_size} bytes, "
        f"{len(tensors)} decoder tensors) and {args.config_output}; "
        f"model={args.model_id}@{args.revision}, "
        f"groups={config['n_codebooks']}, embed_dim={config['embed_dim']}, "
        f"sample_rate={config['sample_rate']}, frame_hop={config['frame_hop']}, "
        f"generator_hop={config['generator_hop']}"
    )


if __name__ == "__main__":
    main()
