#!/usr/bin/env python3
"""Dump an independent SpeechBrain HiFi-GAN mel-to-waveform reference.

This imports SpeechBrain 1.0.3's official ``HifiganGenerator``, loads the
audited upstream ``generator.ckpt`` strictly, removes weight normalization via
the upstream method, and calls its public ``inference(padding=True)`` path.
Vokra code and the offline fold implementation are not imported.

    uv run --python 3.12 python hifigan_dump_reference.py \
      --checkpoint /tmp/speechbrain-hifigan/generator.ckpt \
      --output fixtures/hifigan_reference.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torchaudio

# SpeechBrain 1.0.3 still probes this removed torchaudio API while importing
# its package. The generator itself performs no audio I/O, so an empty backend
# list preserves the old import-time contract without affecting the forward.
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: []  # type: ignore[attr-defined]

from speechbrain.lobes.models.HifiGAN import HifiganGenerator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=2)
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")

    torch.set_num_threads(1)
    model = HifiganGenerator(
        in_channels=80,
        out_channels=1,
        resblock_type="1",
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        resblock_kernel_sizes=(3, 7, 11),
        upsample_kernel_sizes=(16, 16, 4, 4),
        upsample_initial_channel=512,
        upsample_factors=(8, 8, 2, 2),
        cond_channels=0,
        conv_post_bias=True,
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.remove_weight_norm()
    model.eval()

    channel_major = torch.empty((80, args.frames), dtype=torch.float32)
    for channel in range(80):
        for frame in range(args.frames):
            channel_major[channel, frame] = (
                ((channel * 7 + frame * 13) % 31) - 15
            ) / 8.0

    with torch.no_grad():
        waveform = model.inference(channel_major.unsqueeze(0), padding=True)
        waveform = waveform.to(dtype=torch.float32, device="cpu").flatten()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["input", *[f"{value:.9g}" for value in channel_major.flatten().tolist()]]
        )
        writer.writerow(["output", *[f"{value:.9g}" for value in waveform.tolist()]])

    expected_samples = (args.frames + 10) * 256
    if waveform.numel() != expected_samples:
        raise RuntimeError(
            f"official inference emitted {waveform.numel()} samples, "
            f"expected ({args.frames} + 2*5) * 256 = {expected_samples}"
        )
    print(
        "hifigan_reference: "
        f"mels=80 frames={args.frames} padding=5 samples={waveform.numel()} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
