#!/usr/bin/env python3
"""Dump an independent SpeechT5 HiFi-GAN mel-to-waveform reference.

The forward is executed by the official ``transformers.SpeechT5HifiGan``
implementation loaded from ``microsoft/speecht5_hifigan``.  Vokra code is not
imported, so the committed fixture can detect a shared-loader arithmetic or
layout bug instead of merely comparing the Rust path with a Python mirror.

Run through the parity environment:

    uv run --python 3.12 python speecht5_hifigan_dump_reference.py \
        --checkpoint /path/to/speecht5-hifigan \
        --output /path/to/speecht5_hifigan_reference.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from transformers import SpeechT5HifiGan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=2)
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")

    torch.set_num_threads(1)
    model = SpeechT5HifiGan.from_pretrained(
        args.checkpoint,
        local_files_only=True,
    ).eval()
    n_mels = int(model.config.model_in_dim)

    channel_major = torch.empty((n_mels, args.frames), dtype=torch.float32)
    for channel in range(n_mels):
        for frame in range(args.frames):
            channel_major[channel, frame] = (
                ((channel * 7 + frame * 13) % 31) - 15
            ) / 8.0
    frame_major = channel_major.transpose(0, 1).contiguous()

    with torch.no_grad():
        waveform = model(frame_major).to(dtype=torch.float32, device="cpu")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["input", *[f"{value:.9g}" for value in channel_major.flatten().tolist()]]
        )
        writer.writerow(
            ["output", *[f"{value:.9g}" for value in waveform.flatten().tolist()]]
        )

    print(
        "speecht5_hifigan_reference: "
        f"mels={n_mels} frames={args.frames} samples={waveform.numel()} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
