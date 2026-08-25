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
TRANSFORMERS_VERSION = "4.45.2"
FEATURE_SOURCE_SHA256 = (
    "08b31c754524f1e840d8f0bdc04070bd677af2d20c16d66cc0d8556d5c1759cf"
)
MODELING_SOURCE_SHA256 = (
    "9ce5e09eb5e8bedbebd718729f76580cccff047b234bf7bfa05771717e9e5aac"
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
