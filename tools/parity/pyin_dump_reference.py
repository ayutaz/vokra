#!/usr/bin/env python3
"""Generate independent librosa 0.11.0 fixtures for Vokra PyIN parity.

The oracle is the published ``librosa.pyin`` implementation, not a Python
translation of Vokra.  Keep ``librosa==0.11.0`` pinned in this tree's
``pyproject.toml``/``uv.lock`` and run only through ``uv run``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import librosa
import numpy as np


LIBROSA_REVISION = "af8c839fb15317fa2712ea66e7a22da6a9267b32"
SAMPLE_RATE = 16_000
FRAME_LENGTH = 2_048
HOP_LENGTH = 256
FMIN = 65.0
FMAX = 800.0


def tone(frequency: float, length: int, phase: float = 0.0) -> np.ndarray:
    time = np.arange(length, dtype=np.float64) / SAMPLE_RATE
    return (0.7 * np.sin(2.0 * np.pi * frequency * time + phase)).astype(np.float32)


def make_pcm() -> np.ndarray:
    """Silence boundaries, steady tones, and a one-frame octave spike."""
    pcm = np.zeros(32_768, dtype=np.float32)
    pcm[4_096:12_288] = tone(220.0, 8_192)
    # A deliberately short octave excursion: the HMM should not turn it into
    # a discontinuous one-frame jump.
    pcm[7_936:8_448] = tone(440.0, 512, phase=0.3)
    pcm[16_384:] = tone(330.0, len(pcm) - 16_384, phase=0.1)
    return pcm


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pcm = make_pcm()
    f0, voiced, confidence = librosa.pyin(
        pcm,
        sr=SAMPLE_RATE,
        fmin=FMIN,
        fmax=FMAX,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        center=False,
        fill_na=0.0,
    )
    pcm_path = args.out_dir / "pcm.f32.bin"
    f0_path = args.out_dir / "f0.f32.bin"
    voiced_path = args.out_dir / "voiced.u8.bin"
    confidence_path = args.out_dir / "confidence.f32.bin"
    np.ascontiguousarray(pcm, dtype="<f4").tofile(pcm_path)
    np.ascontiguousarray(f0, dtype="<f4").tofile(f0_path)
    np.ascontiguousarray(voiced, dtype=np.uint8).tofile(voiced_path)
    np.ascontiguousarray(confidence, dtype="<f4").tofile(confidence_path)

    manifest = {
        "oracle": "librosa.pyin",
        "librosa_version": librosa.__version__,
        "librosa_revision": LIBROSA_REVISION,
        "sample_rate": SAMPLE_RATE,
        "frame_length": FRAME_LENGTH,
        "hop_length": HOP_LENGTH,
        "center": False,
        "fmin": FMIN,
        "fmax": FMAX,
        "frames": int(len(f0)),
        "files": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in [pcm_path, f0_path, voiced_path, confidence_path]
        },
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
