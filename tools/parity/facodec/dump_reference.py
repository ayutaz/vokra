#!/usr/bin/env python3
"""Dump an independent official NaturalSpeech 3 FACodec V2 reference.

The oracle is the unmodified Amphion source checkout supplied by ``--source``.
Both official checkpoint files and the source commit are verified before the
model is imported. Vokra is never imported and no Vokra forward is mirrored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


SOURCE_REPOSITORY = "https://github.com/open-mmlab/Amphion.git"
SOURCE_REVISION = "26f6883110181f1dbfe95c70a7c7dbaf4de5f42a"
UPSTREAM_REPOSITORY = "amphion/naturalspeech3_facodec"
UPSTREAM_REVISION = "314afc3ea1455ba881a0e484ef9408b6cb996736"
ENCODER_SHA256 = "26636b05867f02f8da3690efb8c36f82909f0a8801ccc4bfdc73cdecf5f9c470"
DECODER_SHA256 = "e6a38d81916affae40a72f5517f39ebadeec4fefea67b074f21d4ec3a0156e3a"
SAMPLE_RATE = 16_000
NUM_SAMPLES = 3_200
FRAME_HOP = 200
NUM_CODEBOOKS = 6
CODEBOOK_SIZE = 1_024
EMBEDDING_DIM = 256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(source: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def verify_source(source: Path) -> None:
    if not (source / ".git").is_dir():
        raise RuntimeError(f"official source is not a Git checkout: {source}")
    revision = git_output(source, "rev-parse", "HEAD")
    if revision != SOURCE_REVISION:
        raise RuntimeError(
            f"official source revision {revision!r} != pinned {SOURCE_REVISION}"
        )
    if git_output(source, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("official source checkout is dirty")
    remote = git_output(source, "remote", "get-url", "origin")
    if remote.rstrip("/") not in {
        SOURCE_REPOSITORY.rstrip("/"),
        SOURCE_REPOSITORY.removesuffix(".git").rstrip("/"),
    }:
        raise RuntimeError(f"official source origin {remote!r} is not {SOURCE_REPOSITORY}")


def verify_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{label} SHA-256 {actual} != pinned {expected_sha256}"
        )


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"checkpoint did not contain a non-empty state dict: {path}")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise RuntimeError(f"checkpoint contains non-tensor state entries: {path}")
    return state


def deterministic_pcm() -> torch.Tensor:
    time = torch.arange(NUM_SAMPLES, dtype=torch.float32) / SAMPLE_RATE
    pcm = (
        0.17 * torch.sin(2.0 * math.pi * 173.0 * time)
        + 0.09 * torch.sin(2.0 * math.pi * 997.0 * time + 0.37)
        + 0.035 * torch.cos(2.0 * math.pi * 2_137.0 * time)
        + 0.012 * torch.sin(2.0 * math.pi * 31.0 * time * time)
    )
    return pcm.clamp(-1.0, 1.0).contiguous()


def write_f32(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().to(torch.float32).contiguous().numpy()
    path.write_bytes(np.asarray(array, dtype="<f4").tobytes(order="C"))


def write_u32(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().to(torch.int64).contiguous().numpy()
    if np.any(array < 0) or np.any(array >= CODEBOOK_SIZE):
        raise RuntimeError(f"FACodec code outside 0..{CODEBOOK_SIZE} in {path.name}")
    path.write_bytes(np.asarray(array, dtype="<u4").tobytes(order="C"))


def require_topology(encoder: torch.nn.Module, decoder: torch.nn.Module) -> None:
    if encoder.__class__.__name__ != "FACodecEncoderV2":
        raise RuntimeError(f"unexpected encoder class {encoder.__class__.__name__}")
    if decoder.__class__.__name__ != "FACodecDecoderV2":
        raise RuntimeError(f"unexpected decoder class {decoder.__class__.__name__}")
    if int(encoder.hop_length) != FRAME_HOP or int(decoder.hop_length) != FRAME_HOP:
        raise RuntimeError(
            f"unexpected encoder/decoder hops {encoder.hop_length}/{decoder.hop_length}"
        )
    if int(encoder.block[-1].out_channels) != EMBEDDING_DIM:
        raise RuntimeError(f"encoder output width is not {EMBEDDING_DIM}")
    if int(decoder.model[0].out_channels) != 1_024:
        raise RuntimeError("decoder initial channel width is not 1024")
    group_widths = [len(group.layers) for group in decoder.quantizer]
    if group_widths != [1, 2, 3]:
        raise RuntimeError(f"unexpected FVQ group widths {group_widths}")
    for group in decoder.quantizer:
        for layer in group.layers:
            if int(layer.codebook_size) != CODEBOOK_SIZE or int(layer.codebook_dim) != 8:
                raise RuntimeError(
                    f"unexpected FVQ axes {layer.codebook_size}x{layer.codebook_dim}"
                )


def execution_environment() -> dict[str, object]:
    capability = getattr(torch.backends.cpu, "get_cpu_capability", None)
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_cpu_capability": (
            str(capability()) if callable(capability) else "unavailable"
        ),
        "torch_threads": torch.get_num_threads(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    verify_source(args.source)
    verify_file(args.encoder, ENCODER_SHA256, "official FACodec V2 encoder")
    verify_file(args.decoder, DECODER_SHA256, "official FACodec V2 decoder")

    sys.path.insert(0, str(args.source))
    from models.codec.ns3_codec.facodec import (  # noqa: PLC0415
        FACodecDecoderV2,
        FACodecEncoderV2,
    )

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0x4641434F)

    encoder = FACodecEncoderV2(
        ngf=32,
        up_ratios=[2, 4, 5, 5],
        out_channels=EMBEDDING_DIM,
    )
    decoder = FACodecDecoderV2(
        in_channels=EMBEDDING_DIM,
        upsample_initial_channel=1_024,
        ngf=32,
        up_ratios=[5, 5, 4, 2],
        vq_num_q_c=2,
        vq_num_q_p=1,
        vq_num_q_r=3,
        vq_dim=EMBEDDING_DIM,
        codebook_dim=8,
        codebook_size_prosody=10,
        codebook_size_content=10,
        codebook_size_residual=10,
        use_gr_x_timbre=True,
        use_gr_residual_f0=True,
        use_gr_residual_phone=True,
    )
    encoder.load_state_dict(checkpoint_state(args.encoder), strict=True)
    decoder.load_state_dict(checkpoint_state(args.decoder), strict=True)
    encoder.eval()
    decoder.eval()
    require_topology(encoder, decoder)

    pcm = deterministic_pcm()
    with torch.inference_mode():
        batch = pcm.reshape(1, 1, -1)
        encoder_latent = encoder.inference(batch)
        prosody = encoder.get_prosody_feature(batch)
        quantized, codebook_first, _, group_quantized, speaker = decoder(
            encoder_latent,
            prosody,
            eval_vq=False,
            vq=True,
        )
        decoded = decoder.inference(quantized, speaker)

    frames = NUM_SAMPLES // FRAME_HOP
    expected_shapes = {
        "encoder_latent": [1, EMBEDDING_DIM, frames],
        "prosody": [1, 20, frames],
        "quantized": [1, EMBEDDING_DIM, frames],
        "codes_codebook_first": [NUM_CODEBOOKS, 1, frames],
        "speaker": [1, EMBEDDING_DIM],
        "decoded": [1, 1, frames * FRAME_HOP],
    }
    actual_shapes = {
        "encoder_latent": list(encoder_latent.shape),
        "prosody": list(prosody.shape),
        "quantized": list(quantized.shape),
        "codes_codebook_first": list(codebook_first.shape),
        "speaker": list(speaker.shape),
        "decoded": list(decoded.shape),
    }
    if actual_shapes != expected_shapes:
        raise RuntimeError(f"official FACodec shapes {actual_shapes} != {expected_shapes}")
    if len(group_quantized) != 3:
        raise RuntimeError(f"official decoder returned {len(group_quantized)} FVQ groups")

    tensors = {
        "pcm.f32": pcm,
        "encoder_latent.f32": encoder_latent,
        "prosody.f32": prosody,
        "quantized.f32": quantized,
        "speaker_embedding.f32": speaker.reshape(-1),
        "decoded_pcm.f32": decoded.reshape(-1),
    }
    for name, tensor in tensors.items():
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"official oracle emitted non-finite values in {name}")
    codes_frame_major = codebook_first[:, 0, :].transpose(0, 1).contiguous()

    args.output.mkdir(parents=True, exist_ok=False)
    written: dict[str, Path] = {}
    for name, tensor in tensors.items():
        path = args.output / name
        write_f32(path, tensor)
        written[name] = path
    codes_path = args.output / "codes.u32le"
    write_u32(codes_path, codes_frame_major)
    written[codes_path.name] = codes_path

    manifest = {
        "format": "vokra-naturalspeech3-facodec-v2-reference-v1",
        "oracle": "official FACodecEncoderV2/FACodecDecoderV2 public inference paths",
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "upstream_hf": UPSTREAM_REPOSITORY,
        "upstream_revision": UPSTREAM_REVISION,
        "encoder_sha256": ENCODER_SHA256,
        "decoder_sha256": DECODER_SHA256,
        "sample_rate": SAMPLE_RATE,
        "frame_hop": FRAME_HOP,
        "num_samples": NUM_SAMPLES,
        "frames": frames,
        "num_codebooks": NUM_CODEBOOKS,
        "codebook_size": CODEBOOK_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "shapes": actual_shapes,
        "environment": execution_environment(),
        "files": {name: sha256_file(path) for name, path in sorted(written.items())},
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
