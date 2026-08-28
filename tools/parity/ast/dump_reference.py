#!/usr/bin/env python3
"""Dump an independent official AST AudioSet feature/logit fixture.

The oracle calls the pinned Hugging Face ``ASTFeatureExtractor`` and
``ASTForAudioClassification`` directly. It neither imports Vokra nor mirrors
the AST frontend or forward equations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import wave
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from transformers.models.audio_spectrogram_transformer import (
    feature_extraction_audio_spectrogram_transformer as feature_source,
)
from transformers.models.audio_spectrogram_transformer import (
    modeling_audio_spectrogram_transformer as modeling_source,
)


UPSTREAM_REPO = "MIT/ast-finetuned-audioset-10-10-0.4593"
UPSTREAM_REVISION = "f826b80d28226b62986cc218e5cec390b1096902"
UPSTREAM_MODEL_SHA256 = (
    "ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d"
)
TRANSFORMERS_VERSION = "5.5.0"
FEATURE_SOURCE_SHA256 = (
    "ab4957749b5113067413dcd662dc212952b9a610d297e8b4515e2cab1ff1fce4"
)
MODELING_SOURCE_SHA256 = (
    "7e0e7b1766999fe0dc4e7b730d676b3d4a7bc26d216b20c38e8163516191d5b4"
)
PUBLIC_GGUF_REPO = "vokra/ast-finetuned-audioset"
PUBLIC_GGUF_REVISION = "b23eb8b8fdc5514b911afd18077fe00618932b13"
PUBLIC_GGUF_SHA256 = (
    "f06bf05078d4267193554ec76e143f8541bd3130c3a81ae2a3d6b5424c8b1ac2"
)
INPUT_WAV_SHA256 = (
    "58adb4ea501d955fcd40bfbb69128f8f40428b81d8716b9ed337949773be253f"
)
SAMPLE_RATE = 16_000
MAX_LENGTH = 1_024
NUM_MELS = 128
NUM_LABELS = 527
FRAME_LENGTH = 400
FRAME_SHIFT = 160
FFT_SIZE = 512
PREEMPHASIS = 0.97
LOW_FREQ = 20.0
NORMALIZE_MEAN = -4.2677393
NORMALIZE_STD = 4.5689974


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_source(module: object, expected: str, label: str) -> Path:
    source_path = Path(str(getattr(module, "__file__", "")))
    if not source_path.is_file():
        raise RuntimeError(f"cannot locate installed {label} source")
    actual = sha256_file(source_path)
    if actual != expected:
        raise RuntimeError(
            f"installed {label} source SHA-256 {actual} != pinned {expected}"
        )
    return source_path


def read_pcm16_mono(path: Path) -> np.ndarray:
    if sha256_file(path) != INPUT_WAV_SHA256:
        raise RuntimeError(f"input WAV {path} does not match the pinned fixture")
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1:
            raise RuntimeError("AST parity input must be mono")
        if wav.getsampwidth() != 2:
            raise RuntimeError("AST parity input must be signed PCM16")
        if wav.getframerate() != SAMPLE_RATE:
            raise RuntimeError(f"AST parity input must be {SAMPLE_RATE} Hz")
        frames = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def write_f32(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().to(torch.float32).contiguous().numpy()
    path.write_bytes(np.asarray(array, dtype="<f4").tobytes(order="C"))


def numpy_f64_input_values(pcm: np.ndarray) -> np.ndarray:
    """Secondary, independent float64 Kaldi-equation cross-check.

    The primary oracle remains the official TorchAudio-backed feature
    extractor. This cross-check quantifies the expected float32 FFT/window/log
    drift, especially in near-floor high-frequency mel bins.
    """
    num_fft_bins = FFT_SIZE // 2
    bin_hz = np.arange(num_fft_bins, dtype=np.float64) * SAMPLE_RATE / FFT_SIZE

    def hz_to_mel(freq: np.ndarray | float) -> np.ndarray | float:
        return 1127.0 * np.log(1.0 + np.asarray(freq) / 700.0)

    low_mel = float(hz_to_mel(LOW_FREQ))
    high_mel = float(hz_to_mel(SAMPLE_RATE / 2))
    delta_mel = (high_mel - low_mel) / (NUM_MELS + 1)
    bin_mel = hz_to_mel(bin_hz)
    banks = np.zeros((NUM_MELS, FFT_SIZE // 2 + 1), dtype=np.float64)
    for mel in range(NUM_MELS):
        left = low_mel + mel * delta_mel
        center = left + delta_mel
        right = center + delta_mel
        up = (bin_mel - left) / (center - left)
        down = (right - bin_mel) / (right - center)
        weights = np.maximum(0.0, np.minimum(up, down))
        weights[(bin_mel <= left) | (bin_mel >= right)] = 0.0
        banks[mel, :num_fft_bins] = weights

    pcm64 = pcm.astype(np.float64)
    frames = 1 + (pcm64.size - FRAME_LENGTH) // FRAME_SHIFT
    kept = min(frames, MAX_LENGTH)
    raw = np.zeros((MAX_LENGTH, NUM_MELS), dtype=np.float64)
    window = np.hanning(FRAME_LENGTH)
    floor = float(np.finfo(np.float32).eps)
    for frame in range(kept):
        start = frame * FRAME_SHIFT
        values = pcm64[start : start + FRAME_LENGTH].copy()
        values -= values.mean()
        emphasized = values.copy()
        emphasized[1:] = values[1:] - PREEMPHASIS * values[:-1]
        emphasized[0] = values[0] - PREEMPHASIS * values[0]
        padded = np.zeros(FFT_SIZE, dtype=np.float64)
        padded[:FRAME_LENGTH] = emphasized * window
        spectrum = np.fft.rfft(padded)
        power = spectrum.real**2 + spectrum.imag**2
        raw[frame] = np.log(np.maximum(banks @ power, floor))
    normalized = (raw - NORMALIZE_MEAN) / (NORMALIZE_STD * 2.0)
    return normalized


def error_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64)).reshape(-1)
    index = int(delta.argmax())
    return {
        "max_abs": float(delta[index]),
        "max_index": index,
        "max_frame": index // NUM_MELS,
        "max_mel": index % NUM_MELS,
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "p99": float(np.quantile(delta, 0.99)),
        "p999": float(np.quantile(delta, 0.999)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if importlib.metadata.version("transformers") != TRANSFORMERS_VERSION:
        raise RuntimeError("installed Transformers version is not pinned")
    feature_path = require_source(
        feature_source, FEATURE_SOURCE_SHA256, "AST feature extractor"
    )
    modeling_path = require_source(
        modeling_source, MODELING_SOURCE_SHA256, "AST model"
    )
    checkpoint_path = Path(
        hf_hub_download(
            repo_id=UPSTREAM_REPO,
            filename="model.safetensors",
            revision=UPSTREAM_REVISION,
        )
    )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != UPSTREAM_MODEL_SHA256:
        raise RuntimeError(
            f"upstream model SHA-256 {checkpoint_sha256} != pinned "
            f"{UPSTREAM_MODEL_SHA256}"
        )

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0x415354)

    pcm = read_pcm16_mono(args.audio)
    extractor = ASTFeatureExtractor.from_pretrained(
        UPSTREAM_REPO, revision=UPSTREAM_REVISION
    )
    model = ASTForAudioClassification.from_pretrained(
        UPSTREAM_REPO,
        revision=UPSTREAM_REVISION,
        attn_implementation="eager",
    )
    model.eval()
    if (
        extractor.sampling_rate != SAMPLE_RATE
        or extractor.max_length != MAX_LENGTH
        or extractor.num_mel_bins != NUM_MELS
        or model.config.num_labels != NUM_LABELS
    ):
        raise RuntimeError("pinned official AST frontend/model geometry changed")

    inputs = extractor(pcm, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_values = inputs["input_values"]
    with torch.inference_mode():
        logits = model(input_values=input_values).logits
    if tuple(input_values.shape) != (1, MAX_LENGTH, NUM_MELS):
        raise RuntimeError(f"unexpected input_values shape {tuple(input_values.shape)}")
    if tuple(logits.shape) != (1, NUM_LABELS):
        raise RuntimeError(f"unexpected logits shape {tuple(logits.shape)}")
    if not bool(torch.isfinite(input_values).all() and torch.isfinite(logits).all()):
        raise RuntimeError("official AST oracle emitted non-finite values")
    numpy_input_values = numpy_f64_input_values(pcm)
    frontend_cross_check = error_metrics(
        input_values.detach().cpu().numpy(), numpy_input_values
    )

    args.output.mkdir(parents=True, exist_ok=True)
    input_path = args.output / "input_values.f32le"
    logits_path = args.output / "logits.f32le"
    write_f32(input_path, input_values)
    write_f32(logits_path, logits)
    top_values, top_indices = torch.topk(logits[0], k=10)
    id2label = model.config.id2label
    top10 = [
        {
            "index": int(index),
            "label": str(id2label[int(index)]),
            "logit": float(value),
        }
        for value, index in zip(top_values, top_indices, strict=True)
    ]
    manifest = {
        "schema": "vokra.ast.official-parity.v1",
        "oracle": {
            "library": "transformers",
            "version": TRANSFORMERS_VERSION,
            "feature_source": str(feature_path),
            "feature_source_sha256": FEATURE_SOURCE_SHA256,
            "modeling_source": str(modeling_path),
            "modeling_source_sha256": MODELING_SOURCE_SHA256,
        },
        "upstream": {
            "repo": UPSTREAM_REPO,
            "revision": UPSTREAM_REVISION,
            "model_safetensors_sha256": checkpoint_sha256,
        },
        "vokra_public_gguf": {
            "repo": PUBLIC_GGUF_REPO,
            "revision": PUBLIC_GGUF_REVISION,
            "sha256": PUBLIC_GGUF_SHA256,
        },
        "input": {
            "audio": args.audio.name,
            "audio_sha256": INPUT_WAV_SHA256,
            "sample_rate": SAMPLE_RATE,
            "samples": int(pcm.size),
        },
        "outputs": {
            input_path.name: {
                "dtype": "f32le",
                "shape": [1, MAX_LENGTH, NUM_MELS],
                "sha256": sha256_file(input_path),
            },
            logits_path.name: {
                "dtype": "f32le",
                "shape": [1, NUM_LABELS],
                "sha256": sha256_file(logits_path),
            },
        },
        "secondary_numpy_f64_frontend_cross_check": frontend_cross_check,
        "top10": top10,
        "environment": {
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "torchaudio": importlib.metadata.version("torchaudio"),
            "numpy": str(np.__version__),
            "threads": torch.get_num_threads(),
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"top10": top10, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
