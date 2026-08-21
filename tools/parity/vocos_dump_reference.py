#!/usr/bin/env python3
"""Dump an independent official Vocos feature-decoder reference.

This imports the released ``vocos==0.1.0`` backbone and head classes directly.
It deliberately bypasses the feature extractor: the Rust/CLI contract is
already-computed channel-major mel or Encodec features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from vocos.heads import ISTFTHead
from vocos.models import VocosBackbone

CONFIGS = {
    "mel": dict(
        input_channels=100,
        dim=512,
        intermediate_dim=1536,
        num_layers=8,
        adanorm_num_embeddings=None,
        n_fft=1024,
        hop_length=256,
        padding="center",
    ),
    "encodec": dict(
        input_channels=128,
        dim=384,
        intermediate_dim=1152,
        num_layers=8,
        adanorm_num_embeddings=4,
        n_fft=1280,
        hop_length=320,
        padding="same",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features-out", type=Path, required=True)
    parser.add_argument("--reference-out", type=Path, required=True)
    parser.add_argument("--metadata-out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--bandwidth-id", type=int, default=2)
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")
    if args.variant == "encodec" and not 0 <= args.bandwidth_id < 4:
        parser.error("--bandwidth-id must be in 0..4")

    torch.set_num_threads(1)
    torch.manual_seed(0)
    cfg = CONFIGS[args.variant]
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError("checkpoint must be a direct state dict")

    backbone = VocosBackbone(
        input_channels=cfg["input_channels"],
        dim=cfg["dim"],
        intermediate_dim=cfg["intermediate_dim"],
        num_layers=cfg["num_layers"],
        adanorm_num_embeddings=cfg["adanorm_num_embeddings"],
    )
    head = ISTFTHead(
        dim=cfg["dim"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        padding=cfg["padding"],
    )
    backbone.load_state_dict(
        {key.removeprefix("backbone."): value for key, value in state.items() if key.startswith("backbone.")},
        strict=True,
    )
    head.load_state_dict(
        {key.removeprefix("head."): value for key, value in state.items() if key.startswith("head.")},
        strict=True,
    )
    backbone.eval()
    head.eval()

    channels = cfg["input_channels"]
    grid = torch.arange(channels * args.frames, dtype=torch.float32).reshape(1, channels, args.frames)
    features = 0.35 * torch.sin(grid * 0.017) + 0.15 * torch.cos(grid * 0.031)
    kwargs = {}
    if args.variant == "encodec":
        kwargs["bandwidth_id"] = torch.tensor([args.bandwidth_id], dtype=torch.long)
    with torch.inference_mode():
        reference = head(backbone(features, **kwargs))[0].contiguous()

    args.features_out.parent.mkdir(parents=True, exist_ok=True)
    features[0].contiguous().numpy().astype("<f4", copy=False).tofile(args.features_out)
    reference.numpy().astype("<f4", copy=False).tofile(args.reference_out)
    metadata = {
        "variant": args.variant,
        "frames": args.frames,
        "channels": channels,
        "bandwidth_id": args.bandwidth_id if args.variant == "encodec" else None,
        "samples": reference.numel(),
        "vocos": "0.1.0",
        "torch": torch.__version__,
    }
    args.metadata_out.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
