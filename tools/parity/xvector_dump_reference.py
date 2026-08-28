#!/usr/bin/env python3
"""Dump an independent SpeechBrain X-vector reference fixture.

The oracle is the pinned ``speechbrain==1.0.3`` implementation and the
official ``speechbrain/spkrec-xvect-voxceleb`` checkpoint.  It never reads a
Vokra GGUF and contains no local reimplementation of the frontend or TDNN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import torch
import torchaudio
import huggingface_hub

# SpeechBrain 1.0.3 probes an API removed by newer torchaudio.  This parity
# path supplies an in-memory waveform and never asks torchaudio to decode.
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: []  # type: ignore[attr-defined]

# SpeechBrain 1.0.3 calls the pre-0.30 ``use_auth_token`` spelling.  Keep the
# compatibility bridge inside this offline oracle instead of changing either
# the pinned SpeechBrain implementation or Vokra's dependency surface.
_hf_hub_download = huggingface_hub.hf_hub_download


def _hf_hub_download_compat(*args: object, **kwargs: object) -> str:
    use_auth_token = kwargs.pop("use_auth_token", None)
    if use_auth_token is not None and "token" not in kwargs:
        kwargs["token"] = use_auth_token
    try:
        return _hf_hub_download(*args, **kwargs)
    except huggingface_hub.errors.RemoteEntryNotFoundError as error:
        # SpeechBrain treats a missing default custom.py as optional, but its
        # 1.0.3 fetch layer only recognizes the legacy requests HTTPError.
        if kwargs.get("filename") == "custom.py":
            raise ValueError("optional custom.py not present on HF Hub") from error
        raise


huggingface_hub.hf_hub_download = _hf_hub_download_compat  # type: ignore[assignment]

import speechbrain  # noqa: E402
from speechbrain.inference.classifiers import EncoderClassifier  # noqa: E402


DEFAULT_MODEL = "speechbrain/spkrec-xvect-voxceleb"
DEFAULT_REVISION = "56895a2df401be4150a159f3a1c653f00051d477"
SAMPLE_RATE = 16_000
PCM_SAMPLES = 16_000
SEED = 1_234


def write_f32(path: Path, values: np.ndarray) -> None:
    path.write_bytes(np.asarray(values, dtype="<f4").tobytes(order="C"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_pcm() -> np.ndarray:
    index = np.arange(PCM_SAMPLES, dtype=np.float64)
    time = index / SAMPLE_RATE
    signal = (
        0.17 * np.sin(2.0 * math.pi * 173.0 * time)
        + 0.09 * np.sin(2.0 * math.pi * 421.0 * time + 0.3)
        + 0.035 * np.cos(2.0 * math.pi * 997.0 * time)
        + 0.018 * np.sin(2.0 * math.pi * (90.0 * time + 310.0 * time * time))
    )
    signal *= np.minimum(1.0, index / 320.0)
    return signal.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--savedir", type=Path, required=True)
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)

    source_path = Path(args.source)
    revision = None if source_path.exists() else args.revision
    classifier = EncoderClassifier.from_hparams(
        source=args.source,
        revision=revision,
        savedir=args.savedir,
        run_opts={"device": "cpu"},
    )
    for module in classifier.mods.values():
        module.eval()

    pcm = deterministic_pcm()
    waveform = torch.from_numpy(pcm).unsqueeze(0)
    lengths = torch.ones(1)
    raw_features = classifier.mods.compute_features(waveform)
    features = classifier.mods.mean_var_norm(raw_features, lengths)

    # Official StatisticsPooling intentionally injects tiny mean noise even in
    # eval mode.  Pinning the seed makes that upstream behavior reproducible;
    # Vokra's deterministic omission is assessed as a measured bounded delta.
    torch.manual_seed(SEED)
    embedding = classifier.mods.embedding_model(features, lengths)
    encoded = classifier.encode_batch(waveform, lengths)

    if tuple(features.shape) != (1, 101, 24):
        raise SystemExit(f"unexpected normalized feature shape {tuple(features.shape)}")
    if tuple(embedding.shape) != (1, 1, 512):
        raise SystemExit(f"unexpected embedding shape {tuple(embedding.shape)}")
    if tuple(encoded.shape) != (1, 1, 512):
        raise SystemExit(f"unexpected encode_batch shape {tuple(encoded.shape)}")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_f32(output / "pcm.f32.bin", pcm)
    write_f32(output / "features.f32.bin", features[0].cpu().numpy())
    write_f32(output / "embedding.f32.bin", embedding[0, 0].cpu().numpy())

    checkpoint = args.savedir / "embedding_model.ckpt"
    manifest = {
        "format": "vokra-xvector-reference-v1",
        "model_id": DEFAULT_MODEL,
        "revision": DEFAULT_REVISION,
        "source": args.source,
        "sample_rate": SAMPLE_RATE,
        "pcm_samples": PCM_SAMPLES,
        "raw_feature_shape": list(raw_features.shape),
        "feature_shape": list(features.shape),
        "embedding_shape": list(embedding.shape),
        "statistics_pool_noise_seed": SEED,
        "checkpoint_sha256": sha256(checkpoint) if checkpoint.exists() else None,
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
