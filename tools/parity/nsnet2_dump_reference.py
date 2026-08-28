#!/usr/bin/env python3
"""Dump an independent Microsoft NSNet2 ONNX reference waveform.

This tool deliberately does not mirror the Rust model implementation. It runs
the pinned official ONNX graph through :class:`onnx.reference.ReferenceEvaluator`
and transcribes Microsoft's NumPy frontend/synthesis from the same revision:

* ``NSNet2-baseline/enhance_onnx.py``
* ``NSNet2-baseline/featurelib.py``

Pinned source revision:
``8b87a33b2892f147b5c7ad39ea978453730db269``.

Run only through the repository's uv-managed Python 3.12 environment::

    uv run --project tools/parity --python 3.12 python \
        tools/parity/nsnet2_dump_reference.py \
        --onnx /path/to/nsnet2-20ms-baseline.onnx \
        --input-wav tests/parity/silero_vad/test_16k.wav \
        --output-wav /tmp/nsnet2-reference.wav \
        --dump-npz /tmp/nsnet2-reference.npz

The official frontend is intentionally unusual: symmetric square-root Hann,
``log10(max(abs(STFT)**2, 1e-12))``, one discarded history frame, right-zero
padding to a whole hop, and raw overlap-add without window-sum normalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import soundfile as sf


PINNED_ONNX_SHA256 = "88429b6253600be840ab816f46f466811d20078142fb12bff8cafe2b27bd4ca9"
SAMPLE_RATE = 16_000
N_FFT = 320
WIN_LENGTH = 320
HOP_LENGTH = 160
N_BINS = 161
MIN_GAIN = 10.0 ** (-80.0 / 20.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_stft(signal: np.ndarray) -> np.ndarray:
    """Transcription of pinned ``featurelib.stft(..., nodelay=True)``."""

    signal = np.asarray(signal, dtype=np.float64).reshape(-1, 1)
    n_input = signal.shape[0]
    n_frames_with_history = int(
        np.ceil((n_input + WIN_LENGTH - HOP_LENGTH) / HOP_LENGTH)
    )
    padded_length = n_frames_with_history * HOP_LENGTH
    signal = np.vstack(
        [signal, np.zeros((padded_length - n_input, 1), dtype=np.float64)]
    )

    window = np.sqrt(np.hanning(WIN_LENGTH)).reshape(-1, 1)
    spectrum = np.zeros(
        (N_BINS, n_frames_with_history, 1), dtype=np.complex128
    )
    frame = np.zeros((WIN_LENGTH, 1), dtype=np.float64)
    for frame_index in range(n_frames_with_history):
        offset = frame_index * HOP_LENGTH
        frame = np.vstack(
            [frame[HOP_LENGTH:, :], signal[offset : offset + HOP_LENGTH, :]]
        )
        spectrum[:, frame_index, :] = np.fft.rfft(
            window * frame, n=N_FFT, axis=0
        )

    delay_frames = WIN_LENGTH // HOP_LENGTH - 1
    return np.squeeze(spectrum[:, delay_frames:, :], axis=2)


def official_istft(spectrum: np.ndarray) -> np.ndarray:
    """Transcription of pinned ``featurelib.istft`` (raw overlap-add)."""

    if spectrum.ndim != 2 or spectrum.shape[0] != N_BINS:
        raise ValueError(
            f"expected spectrum [{N_BINS}, frames], got {spectrum.shape}"
        )
    n_frames = spectrum.shape[1]
    output_length = HOP_LENGTH * (n_frames - 1) + WIN_LENGTH
    output = np.zeros(output_length, dtype=np.float64)
    window = np.sqrt(np.hanning(WIN_LENGTH))
    for frame_index in range(n_frames):
        frame = np.fft.irfft(spectrum[:, frame_index], n=N_FFT)
        offset = frame_index * HOP_LENGTH
        output[offset : offset + WIN_LENGTH] += window * frame[:WIN_LENGTH]
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--input-wav", type=Path, required=True)
    parser.add_argument("--output-wav", type=Path, required=True)
    parser.add_argument(
        "--dump-npz",
        type=Path,
        help="optional NumPy archive containing feature, gain, and enhanced taps",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    onnx_sha = sha256(args.onnx)
    if onnx_sha != PINNED_ONNX_SHA256:
        raise SystemExit(
            "refusing unpinned NSNet2 ONNX: "
            f"sha256={onnx_sha}, expected={PINNED_ONNX_SHA256}"
        )

    signal, sample_rate = sf.read(args.input_wav, dtype="float32", always_2d=False)
    if sample_rate != SAMPLE_RATE:
        raise SystemExit(f"input must be {SAMPLE_RATE} Hz, got {sample_rate} Hz")
    if signal.ndim != 1:
        raise SystemExit(f"input must be mono, got shape {signal.shape}")
    if signal.size == 0:
        raise SystemExit("input WAV is empty")

    spectrum = official_stft(signal)
    feature = np.log10(np.maximum(np.abs(spectrum) ** 2, 1e-12))

    model = onnx.load(args.onnx)
    evaluator = ReferenceEvaluator(model)
    input_name = evaluator.input_names[0]
    network_input = np.expand_dims(feature.T.astype(np.float32), axis=0)
    network_output = evaluator.run(None, {input_name: network_input})[0]
    if network_output.shape != network_input.shape:
        raise SystemExit(
            f"ONNX output shape {network_output.shape} != input shape {network_input.shape}"
        )
    gain = np.clip(network_output[0].T, MIN_GAIN, 1.0)
    enhanced = official_istft(spectrum * gain)

    args.output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output_wav, enhanced.astype(np.float32), SAMPLE_RATE, subtype="FLOAT")
    if args.dump_npz is not None:
        args.dump_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.dump_npz,
            feature=feature.astype(np.float32),
            gain=gain.astype(np.float32),
            enhanced=enhanced.astype(np.float32),
        )

    print(
        json.dumps(
            {
                "onnx_sha256": onnx_sha,
                "input_samples": int(signal.size),
                "frames": int(feature.shape[1]),
                "bins": int(feature.shape[0]),
                "output_samples": int(enhanced.size),
                "output_wav": str(args.output_wav),
                "output_wav_sha256": sha256(args.output_wav),
                "gain_min": float(gain.min()),
                "gain_max": float(gain.max()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
