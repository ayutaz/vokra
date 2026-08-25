#!/usr/bin/env python3
"""Dump an independent official SpeechBrain Lang-ID reference.

The oracle is ``speechbrain.inference.classifiers.EncoderClassifier`` loaded
from one immutable upstream revision. It never reads a Vokra GGUF and contains
no local ECAPA, frontend or classifier mirror. The fixture captures the
official normalized features, embedding, classifier output and ordered label
encoder for the exact same PCM input.
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
from huggingface_hub.errors import RemoteEntryNotFoundError
from requests.exceptions import HTTPError

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: []  # type: ignore[attr-defined]

_hf_hub_download = huggingface_hub.hf_hub_download


def _hf_hub_download_compat(*args: object, **kwargs: object) -> str:
    use_auth_token = kwargs.pop("use_auth_token", None)
    if use_auth_token is not None and "token" not in kwargs:
        kwargs["token"] = use_auth_token
    try:
        return _hf_hub_download(*args, **kwargs)
    except RemoteEntryNotFoundError as error:
        raise HTTPError(f"404 Client Error: {error}") from error


huggingface_hub.hf_hub_download = _hf_hub_download_compat

try:
    import speechbrain
    from speechbrain.inference.classifiers import EncoderClassifier
except Exception as error:  # noqa: BLE001 - loud independent-oracle failure
    raise SystemExit(
        "speechbrain_lang_id_dump_reference: could not import the real "
        f"SpeechBrain implementation ({type(error).__name__}: {error}); a "
        "mirror fallback is forbidden"
    ) from error


DEFAULT_SOURCE = "speechbrain/lang-id-voxlingua107-ecapa"
SAMPLE_RATE = 16_000
EXPECTED = {
    "speechbrain/lang-id-voxlingua107-ecapa": (60, 256, 107),
    "speechbrain/lang-id-commonlanguage_ecapa": (80, 192, 45),
}
PINNED_REVISIONS = {
    "speechbrain/lang-id-voxlingua107-ecapa": (
        "0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9"
    ),
    "speechbrain/lang-id-commonlanguage_ecapa": (
        "70a742bbc513f693efcf73d6d64a5ed14b3a34a4"
    ),
}


def resolve_revision(source: str, revision: str | None) -> str:
    resolved = revision or PINNED_REVISIONS[source]
    if len(resolved) != 40 or any(
        character not in "0123456789abcdefABCDEF" for character in resolved
    ):
        raise SystemExit("--revision must be a full 40-hex commit")
    return resolved.lower()


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


def contiguous_labels(encoder: object) -> list[str]:
    lab2ind = getattr(encoder, "lab2ind", None)
    if not isinstance(lab2ind, dict) or not lab2ind:
        raise RuntimeError("official label encoder has no non-empty lab2ind mapping")
    labels: list[str | None] = [None] * len(lab2ind)
    for label, index in lab2ind.items():
        if not isinstance(label, str) or not isinstance(index, int):
            raise RuntimeError("official label encoder has non string->int entry")
        if index < 0 or index >= len(labels) or labels[index] is not None:
            raise RuntimeError(f"official label encoder has invalid index {index}")
        labels[index] = label
    if any(label is None or not label for label in labels):
        raise RuntimeError("official label encoder indices are not contiguous and non-empty")
    return [label for label in labels if label is not None]


def write_f32(path: Path, values: torch.Tensor | np.ndarray) -> None:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    path.write_bytes(np.asarray(values, dtype="<f4").tobytes(order="C"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--source", choices=sorted(EXPECTED), default=DEFAULT_SOURCE)
    parser.add_argument(
        "--revision",
        help="full upstream commit (defaults to the source-specific audited pin)",
    )
    parser.add_argument("--savedir", type=Path, required=True)
    args = parser.parse_args()
    revision = resolve_revision(args.source, args.revision)

    torch.manual_seed(1234)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    try:
        inference = EncoderClassifier.from_hparams(
            source=args.source,
            revision=revision,
            savedir=args.savedir,
            run_opts={"device": "cpu"},
        )
    except Exception as error:  # noqa: BLE001 - retain official failure detail
        raise SystemExit(
            "speechbrain_lang_id_dump_reference: the real pinned model could "
            f"not be loaded ({type(error).__name__}: {error})"
        ) from error
    for module in inference.mods.values():
        module.eval()

    pcm = read_pcm16_mono(args.wav)
    waveform = torch.from_numpy(pcm.copy()).unsqueeze(0)
    lengths = torch.ones(1)
    raw_features = inference.mods.compute_features(waveform)
    features = inference.mods.mean_var_norm(raw_features, lengths)
    embedding = inference.mods.embedding_model(features, lengths)
    encoded = inference.encode_batch(waveform, lengths, normalize=False)
    classifier_output = inference.mods.classifier(embedding).squeeze(1)
    out_prob, score, index, text_label = inference.classify_batch(waveform, lengths)
    labels = contiguous_labels(inference.hparams.label_encoder)

    n_mels, embedding_dim, class_count = EXPECTED[args.source]
    if tuple(features.shape) != (1, features.shape[1], n_mels):
        raise SystemExit(f"unexpected normalized feature shape {tuple(features.shape)}")
    if tuple(embedding.shape) != (1, 1, embedding_dim):
        raise SystemExit(f"unexpected embedding shape {tuple(embedding.shape)}")
    if tuple(classifier_output.shape) != (1, class_count):
        raise SystemExit(f"unexpected classifier shape {tuple(classifier_output.shape)}")
    if len(labels) != class_count:
        raise SystemExit(f"official label count {len(labels)} != classifier width {class_count}")
    if not torch.equal(embedding, encoded):
        raise SystemExit(
            "manual official embedding and encode_batch differ: "
            f"max_abs={(embedding - encoded).abs().max().item()}"
        )
    if not torch.equal(classifier_output, out_prob):
        raise SystemExit(
            "manual official classifier and classify_batch differ: "
            f"max_abs={(classifier_output - out_prob).abs().max().item()}"
        )
    best_index = int(index.item())
    decoded = text_label[0]
    if decoded != labels[best_index]:
        raise SystemExit(
            f"official decoded label {decoded!r} != ordered label {labels[best_index]!r}"
        )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_f32(output / "pcm.f32.bin", pcm)
    write_f32(output / "features.f32.bin", features[0])
    write_f32(output / "embedding.f32.bin", embedding[0, 0])
    write_f32(output / "scores.f32.bin", out_prob[0])
    (output / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    checkpoint_hashes = {}
    for filename in ["embedding_model.ckpt", "classifier.ckpt", "label_encoder.txt"]:
        path = args.savedir / filename
        checkpoint_hashes[filename] = sha256(path) if path.exists() else None
    manifest = {
        "format": "vokra-speechbrain-lang-id-reference-v1",
        "source": args.source,
        "revision": revision,
        "sample_rate": SAMPLE_RATE,
        "pcm_samples": int(pcm.size),
        "raw_feature_shape": list(raw_features.shape),
        "feature_shape": list(features.shape),
        "embedding_shape": list(embedding.shape),
        "score_shape": list(out_prob.shape),
        "best_index": best_index,
        "best_label": decoded,
        "best_score": float(score.item()),
        "wav_sha256": sha256(args.wav),
        "checkpoint_sha256": checkpoint_hashes,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "speechbrain": speechbrain.__version__,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
