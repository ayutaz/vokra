#!/usr/bin/env python3
"""Dump a real NanoCodec CausalHiFiGANDecoder reference fixture.

The forward and module topology come directly from the pinned official NeMo
package. This script is an offline parity bridge, never a runtime dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import struct
from pathlib import Path
from typing import BinaryIO, Iterable

import torch
from nemo.collections.tts.models.audio_codec import AudioCodecModel


NEMO_SPEECH_COMMIT = "4fcff72febec9395fdbd4bfa0747bfda2ecd3cef"
CHECKPOINT_ID = "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps"
CHECKPOINT_REVISION = "5c8e22ed763c14d81337fbe6ca74062f3d10f7e5"
CHECKPOINT_SHA256 = "bd5883099d0c74ceda760b6b7a1600b86da4d8a02531c9c282679951dcb08870"
MAGIC = b"VKNCHP01"
FORMAT_VERSION = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_reference_environment() -> tuple[str, str, str, str]:
    distribution = importlib.metadata.distribution("nemo_toolkit")
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError(
            "nemo_toolkit has no PEP 610 direct_url.json; refusing an unverified reference install"
        )
    direct_url = json.loads(direct_url_text)
    source_url = str(direct_url.get("url", ""))
    vcs_info = direct_url.get("vcs_info")
    commit = vcs_info.get("commit_id") if isinstance(vcs_info, dict) else None
    if commit != NEMO_SPEECH_COMMIT:
        raise RuntimeError(
            f"imported nemo_toolkit commit {commit!r} != pinned {NEMO_SPEECH_COMMIT}"
        )
    if "github.com/NVIDIA-NeMo/Speech" not in source_url:
        raise RuntimeError(
            f"imported nemo_toolkit source {source_url!r} is not NVIDIA-NeMo/Speech"
        )

    torch_version = str(torch.__version__)
    get_cpu_capability = getattr(torch.backends.cpu, "get_cpu_capability", None)
    cpu_capability = (
        str(get_cpu_capability()) if callable(get_cpu_capability) else "unavailable"
    )
    execution_environment = (
        f"platform={platform.platform()};machine={platform.machine()};"
        f"processor={platform.processor()};cpus={os.cpu_count()};"
        f"torch_threads={torch.get_num_threads()}"
    )
    return source_url, torch_version, cpu_capability, execution_environment


def write_u64(handle: BinaryIO, value: int) -> None:
    if value < 0:
        raise ValueError(f"negative unsigned value: {value}")
    handle.write(struct.pack("<Q", value))


def write_string(handle: BinaryIO, value: str) -> None:
    encoded = value.encode("utf-8")
    write_u64(handle, len(encoded))
    handle.write(encoded)


def write_usizes(handle: BinaryIO, values: Iterable[int]) -> None:
    items = [int(value) for value in values]
    write_u64(handle, len(items))
    for value in items:
        write_u64(handle, value)


def flat_f32(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().to(torch.float32).contiguous().reshape(-1)


def write_f32_tensor(handle: BinaryIO, tensor: torch.Tensor) -> None:
    values = flat_f32(tensor)
    write_u64(handle, values.numel())
    handle.write(values.numpy().tobytes(order="C"))


def write_conv(handle: BinaryIO, module: object) -> None:
    conv = module.conv
    write_f32_tensor(handle, conv.weight)
    if conv.bias is None:
        raise RuntimeError("NanoCodec parity requires convolution bias")
    write_f32_tensor(handle, conv.bias)


def half_snake_tensors(module: object) -> tuple[torch.Tensor, torch.Tensor]:
    activation = module.activation
    if activation.__class__.__name__ != "HalfSnake":
        raise RuntimeError(
            f"expected HalfSnake, got {activation.__class__.__module__}.{activation.__class__.__name__}"
        )
    alpha = flat_f32(activation.snake_act.alpha)
    return alpha, 1.0 / (alpha + 1.0e-9)


def write_half_snake(handle: BinaryIO, module: object) -> None:
    alpha, alpha_inv = half_snake_tensors(module)
    write_f32_tensor(handle, alpha)
    write_f32_tensor(handle, alpha_inv)


def dense_grouped_conv_transpose(module: object) -> torch.Tensor:
    conv = module.conv
    weight = conv.weight.detach().cpu().to(torch.float32).contiguous()
    in_channels = int(conv.in_channels)
    out_channels = int(conv.out_channels)
    groups = int(conv.groups)
    if groups != out_channels:
        raise RuntimeError(
            f"expected NanoCodec groups == out_channels, got {groups} != {out_channels}"
        )
    if in_channels % groups != 0 or weight.shape[1] != 1:
        raise RuntimeError(f"unexpected grouped ConvTranspose1d shape {tuple(weight.shape)}")
    kernel = int(weight.shape[2])
    inputs_per_group = in_channels // groups
    dense = torch.zeros((in_channels, out_channels, kernel), dtype=torch.float32)
    for input_channel in range(in_channels):
        output_channel = input_channel // inputs_per_group
        dense[input_channel, output_channel, :] = weight[input_channel, 0, :]
    return dense


def write_decoder(
    handle: BinaryIO,
    decoder: object,
    frame_hop: int,
    features_frame_major: torch.Tensor,
    expected_pcm: torch.Tensor,
) -> None:
    pre = decoder.pre_conv.conv
    upsample_rates = [int(rate) for rate in decoder.up_sample_rates]
    raw_generator_hop = math.prod(upsample_rates)
    if frame_hop != raw_generator_hop:
        raise RuntimeError(
            f"checkpoint samples_per_frame {frame_hop} != raw generator hop {raw_generator_hop}; "
            "refusing to drop or invent reference waveform samples"
        )
    if len(decoder.res_layers) == 0:
        raise RuntimeError("decoder has no residual layers")
    first_layer = decoder.res_layers[0]
    residual_kernels = [
        int(branch.res_blocks[0].input_conv.conv.kernel_size[0])
        for branch in first_layer.res_blocks
    ]
    residual_dilations = [
        int(block.input_conv.conv.dilation[0]) for block in first_layer.res_blocks[0].res_blocks
    ]

    write_u64(handle, int(pre.in_channels))
    write_u64(handle, int(pre.out_channels))
    write_u64(handle, frame_hop)
    write_usizes(handle, upsample_rates)
    write_u64(handle, int(pre.kernel_size[0]))
    write_u64(handle, int(decoder.post_conv.conv.kernel_size[0]))
    write_usizes(handle, residual_kernels)
    write_usizes(handle, residual_dilations)

    write_conv(handle, decoder.pre_conv)
    for stage_index, (activation, upsample, residual_layer) in enumerate(
        zip(decoder.activations, decoder.up_sample_conv_layers, decoder.res_layers, strict=True)
    ):
        write_half_snake(handle, activation)
        write_f32_tensor(handle, dense_grouped_conv_transpose(upsample))
        if upsample.conv.bias is None:
            raise RuntimeError(f"stage {stage_index} upsample has no bias")
        write_f32_tensor(handle, upsample.conv.bias)
        if len(residual_layer.res_blocks) != len(residual_kernels):
            raise RuntimeError(f"stage {stage_index} residual branch count drift")
        for branch_index, branch in enumerate(residual_layer.res_blocks):
            if len(branch.res_blocks) != len(residual_dilations):
                raise RuntimeError(
                    f"stage {stage_index} branch {branch_index} residual dilation count drift"
                )
            for block_index, block in enumerate(branch.res_blocks):
                dilation = int(block.input_conv.conv.dilation[0])
                if dilation != residual_dilations[block_index]:
                    raise RuntimeError(
                        f"stage {stage_index} branch {branch_index} dilation order drift"
                    )
                write_half_snake(handle, block.input_activation)
                write_conv(handle, block.input_conv)
                write_half_snake(handle, block.skip_activation)
                write_conv(handle, block.skip_conv)
                write_u64(handle, dilation)

    write_half_snake(handle, decoder.post_activation)
    write_conv(handle, decoder.post_conv)
    write_f32_tensor(handle, features_frame_major)
    if expected_pcm.numel() % frame_hop != 0:
        raise RuntimeError("raw NeMo waveform is not aligned to its generator hop")
    write_f32_tensor(handle, expected_pcm)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames <= 0:
        raise ValueError("--frames must be > 0")
    if args.checkpoint_id != CHECKPOINT_ID:
        raise ValueError(
            f"--checkpoint-id must be the audited real-parity profile {CHECKPOINT_ID}"
        )
    if args.checkpoint_revision != CHECKPOINT_REVISION:
        raise ValueError(
            f"--checkpoint-revision {args.checkpoint_revision!r} != audited {CHECKPOINT_REVISION}"
        )
    checkpoint_sha256 = sha256_file(args.checkpoint)
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise RuntimeError(
            f"checkpoint SHA-256 {checkpoint_sha256} != audited {CHECKPOINT_SHA256}"
        )
    source_url, torch_version, cpu_capability, execution_environment = (
        verify_reference_environment()
    )

    model = AudioCodecModel.restore_from(
        restore_path=str(args.checkpoint), map_location=torch.device("cpu")
    )
    model.eval()
    decoder = model.audio_decoder
    if decoder.__class__.__name__ != "CausalHiFiGANDecoder":
        raise RuntimeError(
            f"expected CausalHiFiGANDecoder, got {decoder.__class__.__module__}.{decoder.__class__.__name__}"
        )
    if decoder.out_activation.__class__.__name__ != "ClampActivation":
        raise RuntimeError("NanoCodec parity requires clamp output activation")

    input_dim = int(decoder.pre_conv.conv.in_channels)
    values = torch.arange(input_dim * args.frames, dtype=torch.float32)
    # Deterministic, non-degenerate input; it is stored in the fixture, so the
    # Rust side never independently recreates this formula.
    features = (torch.sin(values * 0.173 + 0.41) * 0.2).reshape(
        1, input_dim, args.frames
    )
    input_len = torch.tensor([args.frames], dtype=torch.long)
    with torch.inference_mode():
        pcm, pcm_len = decoder(inputs=features, input_len=input_len)
    expected_len = int(pcm_len[0].item())
    if pcm.shape != (1, expected_len):
        raise RuntimeError(f"unexpected decoder output shape {tuple(pcm.shape)}")

    # This is a checkpoint/config axis, not an observed-output guess. The
    # forward length separately cross-checks it below.
    frame_hop = int(model.cfg.samples_per_frame)
    raw_generator_hop = expected_len // args.frames
    if raw_generator_hop * args.frames != expected_len:
        raise RuntimeError("decoder output is not aligned to its raw generator hop")
    if frame_hop != raw_generator_hop:
        raise RuntimeError(
            f"checkpoint samples_per_frame {frame_hop} != raw generator hop {raw_generator_hop}; "
            "refusing to drop or invent reference waveform samples"
        )
    features_frame_major = features[0].transpose(0, 1).contiguous()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<I", FORMAT_VERSION))
        write_string(handle, NEMO_SPEECH_COMMIT)
        write_string(handle, source_url)
        write_string(handle, torch_version)
        write_string(handle, cpu_capability)
        write_string(handle, execution_environment)
        write_string(handle, args.checkpoint_id)
        write_string(handle, args.checkpoint_revision)
        write_string(handle, checkpoint_sha256)
        write_decoder(
            handle,
            decoder,
            frame_hop,
            features_frame_major,
            pcm[0, :expected_len],
        )

    print(
        f"wrote {args.output} ({args.output.stat().st_size} bytes): "
        f"frames={args.frames}, input_dim={input_dim}, frame_hop={frame_hop}, "
        f"checkpoint={args.checkpoint_id}@{args.checkpoint_revision}, "
        f"checkpoint_sha256={checkpoint_sha256}, nemo={NEMO_SPEECH_COMMIT}, "
        f"nemo_source={source_url}, torch={torch_version}, "
        f"cpu_capability={cpu_capability}, {execution_environment}"
    )


if __name__ == "__main__":
    main()
