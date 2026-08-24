#!/usr/bin/env python3
"""Dump an independent SpeechBrain SepFormer waveform reference.

The oracle is the pinned ``speechbrain==1.0.3`` implementation and official
three-part checkpoint.  It never reads a Vokra GGUF and has no local layer
mirror fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import numpy as np
import torch
import torchaudio

# SpeechBrain 1.0.3 probes an API removed by newer torchaudio.  The model path
# below consumes an in-memory tensor and never asks torchaudio to decode audio.
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: []  # type: ignore[attr-defined]

import speechbrain  # noqa: E402
from speechbrain.inference.separation import SepformerSeparation  # noqa: E402


DEFAULT_MODEL = "speechbrain/sepformer-wham16k-enhancement"
DEFAULT_REVISION = "90b3c5c3ffe3e04387b566715ab5fff36ec7b9d9"
SAMPLE_RATE = 16_000
PCM_SAMPLES = 4_096


def write_f32(path: Path, values: np.ndarray) -> None:
    path.write_bytes(np.asarray(values, dtype="<f4").tobytes(order="C"))


def deterministic_pcm() -> np.ndarray:
    index = np.arange(PCM_SAMPLES, dtype=np.float64)
    time = index / SAMPLE_RATE
    signal = (
        0.20 * np.sin(2.0 * math.pi * 173.0 * time)
        + 0.11 * np.sin(2.0 * math.pi * 421.0 * time + 0.3)
        + 0.04 * np.cos(2.0 * math.pi * 997.0 * time)
    )
    signal *= np.minimum(1.0, index / 160.0)
    return signal.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--savedir", type=Path, required=True)
    args = parser.parse_args()

    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)

    source_path = Path(args.source)
    revision = None if source_path.exists() else args.revision
    model = SepformerSeparation.from_hparams(
        source=args.source,
        revision=revision,
        savedir=args.savedir,
        run_opts={"device": "cpu"},
    )
    for module in model.mods.values():
        module.eval()

    pcm = deterministic_pcm()
    mixture = torch.from_numpy(pcm).unsqueeze(0)
    encoder = model.mods.encoder(mixture)
    separated = model.separate_batch(mixture)
    if tuple(separated.shape) != (1, PCM_SAMPLES, 1):
        raise SystemExit(f"unexpected separated shape {tuple(separated.shape)}")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_f32(output / "pcm.f32.bin", pcm)
    write_f32(output / "encoder.f32.bin", encoder[0].cpu().numpy())
    write_f32(output / "separated.f32.bin", separated[0, :, 0].cpu().numpy())

    manifest = {
        "format": "vokra-sepformer-reference-v1",
        "model_id": DEFAULT_MODEL,
        "revision": DEFAULT_REVISION,
        "source": args.source,
        "sample_rate": SAMPLE_RATE,
        "pcm_samples": PCM_SAMPLES,
        "encoder_shape": list(encoder.shape),
        "separated_shape": list(separated.shape),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "speechbrain": speechbrain.__version__,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
