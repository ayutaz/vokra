#!/usr/bin/env python3
"""Dump an independent SpeechBrain ECAPA-TDNN reference fixture.

The oracle is the pinned ``speechbrain==1.0.3`` implementation, the official
``speechbrain/spkrec-ecapa-voxceleb`` checkpoint, and an official example WAV.
It never reads a Vokra GGUF and contains no local ECAPA/frontend mirror.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import wave
from pathlib import Path

import huggingface_hub
import numpy as np
import torch
import torchaudio

# SpeechBrain 1.0.3 probes an API removed by newer torchaudio. This parity path
# supplies an already decoded in-memory waveform and never asks torchaudio to
# select a decoder backend.
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: []  # type: ignore[attr-defined]

# SpeechBrain 1.0.3 calls the pre-0.30 ``use_auth_token`` spelling. Keep this
# compatibility bridge inside the offline oracle.
_hf_hub_download = huggingface_hub.hf_hub_download


def _hf_hub_download_compat(*args: object, **kwargs: object) -> str:
    use_auth_token = kwargs.pop("use_auth_token", None)
    if use_auth_token is not None and "token" not in kwargs:
        kwargs["token"] = use_auth_token
    try:
        return _hf_hub_download(*args, **kwargs)
    except huggingface_hub.errors.RemoteEntryNotFoundError as error:
        if kwargs.get("filename") == "custom.py":
            raise ValueError("optional custom.py not present on HF Hub") from error
        raise


huggingface_hub.hf_hub_download = _hf_hub_download_compat  # type: ignore[assignment]

import speechbrain  # noqa: E402
from speechbrain.inference.classifiers import EncoderClassifier  # noqa: E402


DEFAULT_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
SAMPLE_RATE = 16_000


def write_f32(path: Path, values: np.ndarray) -> None:
    path.write_bytes(np.asarray(values, dtype="<f4").tobytes(order="C"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pcm16_mono(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as stream:
        channels = stream.getnchannels()
        sample_width = stream.getsampwidth()
        sample_rate = stream.getframerate()
        frames = stream.getnframes()
        payload = stream.readframes(frames)
    if channels != 1 or sample_width != 2 or sample_rate != SAMPLE_RATE:
        raise SystemExit(
            "expected mono PCM16 16 kHz WAV, got "
            f"channels={channels}, width={sample_width}, rate={sample_rate}"
        )
    return np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--source", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--savedir", type=Path, required=True)
    args = parser.parse_args()

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

    pcm = read_pcm16_mono(args.wav)
    waveform = torch.from_numpy(pcm.copy()).unsqueeze(0)
    lengths = torch.ones(1)
    raw_features = classifier.mods.compute_features(waveform)
    features = classifier.mods.mean_var_norm(raw_features, lengths)
    embedding = classifier.mods.embedding_model(features, lengths)
    encoded = classifier.encode_batch(waveform, lengths, normalize=False)

    if tuple(features.shape[0:1]) != (1,) or features.shape[2] != 80:
        raise SystemExit(f"unexpected normalized feature shape {tuple(features.shape)}")
    if tuple(embedding.shape) != (1, 1, 192):
        raise SystemExit(f"unexpected embedding shape {tuple(embedding.shape)}")
    if tuple(encoded.shape) != (1, 1, 192):
        raise SystemExit(f"unexpected encode_batch shape {tuple(encoded.shape)}")
    if not torch.equal(embedding, encoded):
        delta = (embedding - encoded).abs().max().item()
        raise SystemExit(f"embedding_model and encode_batch differ: max_abs={delta}")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_f32(output / "pcm.f32.bin", pcm)
    write_f32(output / "features.f32.bin", features[0].cpu().numpy())
    write_f32(output / "embedding.f32.bin", embedding[0, 0].cpu().numpy())

    checkpoint = args.savedir / "embedding_model.ckpt"
    manifest = {
        "format": "vokra-ecapa-tdnn-reference-v1",
        "model_id": DEFAULT_MODEL,
        "revision": DEFAULT_REVISION,
        "source": args.source,
        "sample_rate": SAMPLE_RATE,
        "pcm_samples": int(pcm.size),
        "raw_feature_shape": list(raw_features.shape),
        "feature_shape": list(features.shape),
        "embedding_shape": list(embedding.shape),
        "wav_sha256": sha256(args.wav),
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
