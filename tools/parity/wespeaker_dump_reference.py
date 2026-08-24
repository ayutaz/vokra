#!/usr/bin/env python3
"""Dump an independent WeSpeaker ResNet34-LM reference fixture.

The oracle imports the pinned upstream WeSpeaker source tree, loads the
official ``avg_model`` checkpoint directly, and uses torchaudio's Kaldi fbank.
It never reads a Vokra GGUF and contains no local copy of the ResNet forward.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio


MODEL_ID = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
MODEL_REVISION = "f0c48c298fd835726c27956a5d617bad7115627e"
SOURCE_REVISION = "45941e7cba2c3ea99e232d02bedf617fc71b0dad"
SAMPLE_RATE = 16_000
PCM_SAMPLES = 32_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_f32(path: Path, values: np.ndarray) -> None:
    path.write_bytes(np.asarray(values, dtype="<f4").tobytes(order="C"))


def deterministic_pcm() -> np.ndarray:
    index = np.arange(PCM_SAMPLES, dtype=np.float64)
    time = index / SAMPLE_RATE
    rng = np.random.default_rng(0x5753504B)
    envelope = np.minimum(1.0, index / 1200.0) * np.minimum(
        1.0, (PCM_SAMPLES - 1 - index) / 1600.0
    )
    chirp_phase = 2.0 * np.pi * (95.0 * time + 0.5 * 185.0 * time * time)
    pcm = envelope * (
        0.31 * np.sin(chirp_phase)
        + 0.17 * np.sin(2.0 * np.pi * 233.0 * time + 0.4)
        + 0.09 * np.sin(2.0 * np.pi * 711.0 * time + 1.1)
    )
    pcm += rng.normal(0.0, 0.003, PCM_SAMPLES)
    return np.asarray(np.clip(pcm, -0.95, 0.95), dtype=np.float32)


def unwrap_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"expected checkpoint dict, got {type(checkpoint).__name__}")
    for key in ("model", "state_dict"):
        nested = checkpoint.get(key)
        if isinstance(nested, dict) and nested and all(
            isinstance(name, str) and isinstance(value, torch.Tensor)
            for name, value in nested.items()
        ):
            checkpoint = nested
            break
    if not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in checkpoint.items()
    ):
        raise SystemExit("checkpoint is not a string-to-tensor state dict")
    return checkpoint  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--wespeaker-source", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.wespeaker_source.resolve()
    if not (source_root / "wespeaker" / "models" / "resnet.py").is_file():
        raise SystemExit(f"not a WeSpeaker source tree: {source_root}")
    sys.path.insert(0, str(source_root))
    resnet = importlib.import_module("wespeaker.models.resnet")

    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    model = resnet.ResNet34(
        feat_dim=80,
        embed_dim=256,
        pooling_func="TSTP",
        two_emb_layer=False,
    ).eval()
    state = unwrap_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    )
    expected_names = set(model.state_dict())
    available_names = set(state)
    missing = sorted(expected_names - available_names)
    extras = sorted(available_names - expected_names)
    if missing:
        raise SystemExit(f"checkpoint misses model tensors: {missing}")
    if extras != ["projection.weight"]:
        raise SystemExit(
            "expected only the unused LM classifier as an extra tensor, got "
            f"{extras}"
        )
    model.load_state_dict({name: state[name] for name in expected_names}, strict=True)

    pcm = deterministic_pcm()
    waveform = torch.from_numpy(pcm.copy()).unsqueeze(0)
    features = torchaudio.compliance.kaldi.fbank(
        waveform * (1 << 15),
        num_mel_bins=80,
        frame_length=25.0,
        frame_shift=10.0,
        round_to_power_of_two=True,
        snip_edges=True,
        dither=0.0,
        sample_frequency=SAMPLE_RATE,
        window_type="hamming",
        use_energy=False,
    )
    features = features - features.mean(dim=0, keepdim=True)
    _, embedding = model(features.unsqueeze(0))

    if tuple(features.shape) != (198, 80):
        raise SystemExit(f"unexpected feature shape {tuple(features.shape)}")
    if tuple(embedding.shape) != (1, 256):
        raise SystemExit(f"unexpected embedding shape {tuple(embedding.shape)}")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_f32(output / "pcm.f32.bin", pcm)
    write_f32(output / "features.f32.bin", features.cpu().numpy())
    write_f32(output / "embedding.f32.bin", embedding[0].cpu().numpy())
    manifest = {
        "format": "vokra-wespeaker-reference-v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "source_revision": SOURCE_REVISION,
        "sample_rate": SAMPLE_RATE,
        "pcm_samples": int(pcm.size),
        "feature_shape": list(features.shape),
        "embedding_shape": list(embedding.shape),
        "checkpoint_sha256": sha256(args.checkpoint),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
