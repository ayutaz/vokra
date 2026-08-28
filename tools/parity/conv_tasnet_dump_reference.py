#!/usr/bin/env python3
"""Dump an independent official Asteroid 0.7.0 Conv-TasNet reference.

The oracle loads the pinned upstream ``pytorch_model.bin`` through Asteroid's
own ``ConvTasNet.from_pretrained`` and calls its encoder, masker and decoder.
It never reads a Vokra GGUF and contains no local network-layer mirror.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import asteroid
import numpy as np
import torch
from asteroid.models import ConvTasNet


UPSTREAM_HF = "JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k"
UPSTREAM_REVISION = "bb8a876bc157b5cf3c405994accb798c49146016"
SAMPLE_RATE = 16_000
PCM_SAMPLES = 4_096


def write_f32(path: Path, values: np.ndarray) -> None:
    path.write_bytes(np.asarray(values, dtype="<f4").tobytes(order="C"))


def deterministic_pcm() -> np.ndarray:
    index = np.arange(PCM_SAMPLES, dtype=np.float64)
    time = index / SAMPLE_RATE
    signal = (
        0.18 * np.sin(2.0 * math.pi * 191.0 * time)
        + 0.09 * np.sin(2.0 * math.pi * 503.0 * time + 0.21)
        + 0.035 * np.cos(2.0 * math.pi * 1201.0 * time)
    )
    signal *= np.minimum(1.0, index / 192.0)
    return signal.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)

    model = ConvTasNet.from_pretrained(str(args.checkpoint))
    model.eval()
    pcm = deterministic_pcm()
    waveform = torch.from_numpy(pcm).reshape(1, 1, -1)
    encoded = model.forward_encoder(waveform)
    bottleneck = model.masker.bottleneck(encoded)
    masks = model.forward_masker(encoded)
    masked = model.apply_masks(encoded, masks)
    decoded = model.forward_decoder(masked)
    separated = model(waveform)

    expected = {
        "encoded": (1, 512, 255),
        "bottleneck": (1, 128, 255),
        "masks": (1, 1, 512, 255),
        "decoded": (1, 1, 4096),
        "separated": (1, 1, 4096),
    }
    actual = {
        "encoded": tuple(encoded.shape),
        "bottleneck": tuple(bottleneck.shape),
        "masks": tuple(masks.shape),
        "decoded": tuple(decoded.shape),
        "separated": tuple(separated.shape),
    }
    if actual != expected:
        raise SystemExit(f"unexpected reference shapes: {actual!r}, expected {expected!r}")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_f32(output / "pcm.f32.bin", pcm)
    write_f32(output / "encoder.f32.bin", encoded[0].cpu().numpy())
    write_f32(output / "bottleneck.f32.bin", bottleneck[0].cpu().numpy())
    write_f32(output / "mask.f32.bin", masks[0, 0].cpu().numpy())
    write_f32(output / "separated.f32.bin", separated[0, 0].cpu().numpy())

    manifest = {
        "format": "vokra-conv-tasnet-reference-v1",
        "model_id": UPSTREAM_HF,
        "revision": UPSTREAM_REVISION,
        "sample_rate": SAMPLE_RATE,
        "pcm_samples": PCM_SAMPLES,
        "shapes": {name: list(shape) for name, shape in actual.items()},
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "asteroid": asteroid.__version__,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
