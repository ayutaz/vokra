#!/usr/bin/env python3
"""Dump an independent torchfcpe-0.0.4 reference for FCPE parity.

This sidecar imports the official ``torchfcpe`` wheel and runs its bundled
``fcpe_c_v001.pt`` checkpoint.  It does not mirror the Vokra Rust forward.
The generated fixture pins the input PCM plus three useful parity boundaries:

* ``mel.f32`` — official Wav2MelModule output, row-major ``[frames, 128]``;
* ``latent.f32`` — official sigmoid classifier output, ``[frames, 360]``;
* ``f0.f32`` — official local-argmax decoder output, ``[frames]``.

The input is a short deterministic PCM16 WAV.  The official public inference
wrapper's threshold is passed explicitly as ``0.006``; this is the default of
``InferCFNaiveMelPE.forward`` / ``infer`` and the value recommended by the
official PyPI usage example.  The lower-level model decoder's separate
``0.05`` default is not the public waveform-to-F0 API default.

Reproduction (Python 3.12, repository-managed uv environment)::

    uv run --project tools/parity \
      --with /path/to/torchfcpe-0.0.4-py3-none-any.whl \
      --with local-attention==1.11.2 \
      python tools/parity/fcpe_dump_reference.py \
      --wheel /path/to/torchfcpe-0.0.4-py3-none-any.whl \
      --checkpoint /path/to/torchfcpe/assets/fcpe_c_v001.pt \
      --out-dir tests/parity/fcpe

Only the official PyPI wheel with SHA-256
``f042c463d850d76c6f4899a0b84f0b694bb560adf05f4de951097a756d17472d``
and its bundled checkpoint with SHA-256
``b9aeaeb673436eeda50ceafd632aa681aa63417e52eae4207503d180c9b10015``
are accepted.  A different file fails closed instead of producing a fixture
that only appears to describe the published FCPE v001 checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import wave
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import torchfcpe


WHEEL_SHA256 = "f042c463d850d76c6f4899a0b84f0b694bb560adf05f4de951097a756d17472d"
CHECKPOINT_SHA256 = "b9aeaeb673436eeda50ceafd632aa681aa63417e52eae4207503d180c9b10015"
SAMPLE_RATE = 16_000
SAMPLE_COUNT = 5_120
PUBLIC_THRESHOLD = 0.006


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise SystemExit(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual} ({path})"
        )


def deterministic_pcm16() -> np.ndarray:
    """Return 0.32 s of exact PCM16: two harmonics followed by silence."""

    voiced = 3_200
    samples = np.zeros(SAMPLE_COUNT, dtype=np.float64)
    for index in range(voiced):
        time = index / SAMPLE_RATE
        # Stay comfortably below clipping.  A second harmonic exercises the
        # mel front-end without making the pitch target ambiguous.
        samples[index] = 0.35 * math.sin(2.0 * math.pi * 220.0 * time)
        samples[index] += 0.08 * math.sin(2.0 * math.pi * 440.0 * time + 0.3)
    return np.rint(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")


def write_pcm16_wav(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(samples.tobytes(order="C"))


def write_f32(path: Path, tensor: torch.Tensor) -> None:
    values = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    path.write_bytes(values.tobytes(order="C"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    require_sha256(args.wheel, WHEEL_SHA256, "official torchfcpe wheel")
    require_sha256(args.checkpoint, CHECKPOINT_SHA256, "bundled FCPE checkpoint")
    if version("torchfcpe") != "0.0.4":
        raise SystemExit(
            f"expected imported torchfcpe 0.0.4, got {version('torchfcpe')}"
        )

    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pcm_i16 = deterministic_pcm16()
    wav_path = args.out_dir / "input.wav"
    write_pcm16_wav(wav_path, pcm_i16)
    pcm = pcm_i16.astype(np.float32) / np.float32(32768.0)
    wav = torch.from_numpy(pcm).reshape(1, -1, 1)

    reference = torchfcpe.spawn_infer_model_from_pt(
        str(args.checkpoint), device="cpu", bundled_model=True
    )
    with torch.no_grad():
        mel = reference.wav2mel(wav, SAMPLE_RATE)
        latent = reference.model.forward(mel)
        f0 = reference.infer(
            wav,
            sr=SAMPLE_RATE,
            decoder_mode="local_argmax",
            threshold=PUBLIC_THRESHOLD,
            interp_uv=False,
        )

    if tuple(mel.shape[:1]) != (1,) or tuple(latent.shape[:1]) != (1,):
        raise SystemExit(
            f"unexpected batch shapes: mel={tuple(mel.shape)} "
            f"latent={tuple(latent.shape)}"
        )
    if mel.shape[1] != latent.shape[1] or f0.shape[1] != mel.shape[1]:
        raise SystemExit(
            f"frame-count mismatch: mel={tuple(mel.shape)} latent={tuple(latent.shape)} "
            f"f0={tuple(f0.shape)}"
        )

    write_f32(args.out_dir / "mel.f32", mel[0])
    write_f32(args.out_dir / "latent.f32", latent[0])
    write_f32(args.out_dir / "f0.f32", f0[0, :, 0])

    meta = {
        "format": "vokra-fcpe-parity-v1",
        "upstream": "CNChTu/FCPE official PyPI torchfcpe wheel",
        "torchfcpe_version": version("torchfcpe"),
        "torch_version": version("torch"),
        "torchaudio_version": version("torchaudio"),
        "local_attention_version": version("local-attention"),
        "librosa_version": version("librosa"),
        "numpy_version": version("numpy"),
        "wheel_sha256": WHEEL_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "sample_rate": SAMPLE_RATE,
        "sample_count": int(pcm_i16.size),
        "frames": int(mel.shape[1]),
        "n_mels": int(mel.shape[2]),
        "n_pitch_bins": int(latent.shape[2]),
        "decoder": "local_argmax",
        "confidence_threshold": PUBLIC_THRESHOLD,
        "files": {},
    }
    for name in ("input.wav", "mel.f32", "latent.f32", "f0.f32"):
        path = args.out_dir / name
        meta["files"][name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    (args.out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
