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
import os
import platform
import tempfile
import wave
from pathlib import Path

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


def self_test() -> None:
    assert DEFAULT_SOURCE == "speechbrain/lang-id-voxlingua107-ecapa"
    assert PINNED_REVISIONS[DEFAULT_SOURCE] == "0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9"
    assert EXPECTED[DEFAULT_SOURCE] == (60, 256, 107)
    assert resolve_revision(DEFAULT_SOURCE, None) == PINNED_REVISIONS[DEFAULT_SOURCE]
    try:
        resolve_revision(DEFAULT_SOURCE, "a" * 40)
    except SystemExit:
        pass
    else:
        raise AssertionError("untrusted revision was accepted")
    import tempfile

    with tempfile.TemporaryDirectory(prefix="speechbrain-lang-id-dump-self-test-") as directory:
        root = Path(directory)
        valid = root / "output"
        validate_output_dir(valid)
        valid.mkdir()
        normal_absolute_string = str(root / "normal-output")
        validate_output_dir(normal_absolute_string)
        existing = valid / "keep.bin"
        existing.write_bytes(b"keep")
        try:
            write_no_clobber(existing, b"replace")
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing fixture was overwritten")
        assert existing.read_bytes() == b"keep"
        first = valid / "first.bin"
        second = valid / "second.bin"
        first_payload = b"first"
        second_payload = b"keep-second"
        second.write_bytes(second_payload)
        try:
            publish_fixtures({first: first_payload, second: b"replace-second"})
        except FileExistsError:
            pass
        else:
            raise AssertionError("second fixture collision was accepted")
        assert not first.exists()
        assert second.read_bytes() == second_payload
        assert not list(valid.glob(".*.tmp")), "rollback leaked fixture temporary"
        from unittest.mock import patch

        cleanup = valid / "cleanup.bin"
        with patch.object(Path, "unlink", side_effect=PermissionError("test cleanup failure")):
            publish_fixtures({cleanup: b"complete"})
        assert cleanup.read_bytes() == b"complete"
        for temporary_path in valid.glob(".*.tmp"):
            os.unlink(temporary_path)
        try:
            validate_output_dir(str(root) + "/./dot-output")
        except SystemExit:
            pass
        else:
            raise AssertionError("dot component accepted")
        real = root / "real"
        real.mkdir()
        link = root / "link"
        link.symlink_to(real, target_is_directory=True)
        try:
            validate_output_dir(link / "output")
        except SystemExit:
            pass
        else:
            raise AssertionError("symlink ancestor accepted")
    print("speechbrain_lang_id_dump_reference: stdlib self-test PASS")


def resolve_revision(source: str, revision: str | None) -> str:
    resolved = revision or PINNED_REVISIONS[source]
    if len(resolved) != 40 or any(
        character not in "0123456789abcdefABCDEF" for character in resolved
    ):
        raise SystemExit("--revision must be a full 40-hex commit")
    resolved = resolved.lower()
    if resolved != PINNED_REVISIONS[source]:
        raise SystemExit(
            "--revision must equal the audited immutable revision "
            f"{PINNED_REVISIONS[source]} for {source}"
        )
    return resolved


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
    write_no_clobber(path, np.asarray(values, dtype="<f4").tobytes(order="C"))


def reject_symlink_ancestry(path: Path | str, label: str) -> None:
    raw = os.fspath(path)
    if any(component in {".", ".."} for component in raw.split("/")):
        raise SystemExit(f"{label} must not contain lexical dot components")
    path = Path(raw)
    absolute = path if path.is_absolute() else Path.cwd() / path
    for ancestor in (absolute, *absolute.parents):
        # macOS exposes /var as the system /private/var alias; it is not
        # user-controlled output redirection and is safe to traverse.
        if ancestor.is_symlink() and ancestor != Path("/var"):
            raise SystemExit(f"{label} has symlink ancestry: {ancestor}")


def validate_output_dir(path: Path | str) -> None:
    reject_symlink_ancestry(path, "--output-dir")
    path = Path(path)
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise SystemExit("--output-dir must be a non-symlink directory")
    if path.exists() and any(path.iterdir()):
        raise SystemExit("--output-dir must be empty to prevent fixture clobbering")


def validate_input_path(path: Path | str, label: str) -> None:
    reject_symlink_ancestry(path, label)
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"{label} must be a regular non-symlink file")


def write_no_clobber(path: Path, payload: bytes) -> None:
    publish_fixtures({path: payload})


def cleanup_temp(path: Path | None) -> None:
    if path is not None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _validate_publish_parent(path: Path) -> None:
    reject_symlink_ancestry(path, "fixture output")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise SystemExit(f"fixture output parent is not a regular directory: {path.parent}")


def publish_fixtures(files: dict[Path, bytes]) -> None:
    """Publish all fixture files, rolling back only this call's claims."""
    temporary: list[tuple[Path, Path]] = []
    claimed: list[tuple[Path, Path]] = []
    try:
        for path, payload in files.items():
            _validate_publish_parent(path)
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.append((temporary_path, path))
        for temporary_path, path in temporary:
            # Re-check immediately before every claim to close a parent
            # symlink race after temp creation.
            _validate_publish_parent(path)
            os.link(temporary_path, path)
            claimed.append((temporary_path, path))
    except BaseException:
        for temporary_path, path in reversed(claimed):
            try:
                if os.path.samestat(
                    os.stat(temporary_path, follow_symlinks=False),
                    os.stat(path, follow_symlinks=False),
                ):
                    path.unlink()
            except OSError:
                pass
        raise
    finally:
        for temporary_path, _ in temporary:
            cleanup_temp(temporary_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir")
    parser.add_argument("--wav")
    parser.add_argument("--source", choices=sorted(EXPECTED), default=DEFAULT_SOURCE)
    parser.add_argument(
        "--revision",
        help="full upstream commit (defaults to the source-specific audited pin)",
    )
    parser.add_argument("--savedir")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.output_dir is None or args.wav is None or args.savedir is None:
        parser.error("--output-dir, --wav, and --savedir are required unless --self-test is used")
    validate_output_dir(args.output_dir)
    validate_input_path(args.wav, "--wav")
    reject_symlink_ancestry(args.savedir, "--savedir")
    output_dir = Path(args.output_dir)
    wav_path = Path(args.wav)
    savedir = Path(args.savedir)
    if savedir.is_symlink() or (savedir.exists() and not savedir.is_dir()):
        parser.error("--savedir must be a non-symlink directory")
    revision = resolve_revision(args.source, args.revision)

    global huggingface_hub, np, torch, torchaudio, speechbrain, EncoderClassifier
    import huggingface_hub
    import numpy as np
    import torch
    import torchaudio
    from huggingface_hub.errors import RemoteEntryNotFoundError
    from requests.exceptions import HTTPError
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: []  # type: ignore[attr-defined]
    hf_hub_download = huggingface_hub.hf_hub_download

    def _hf_hub_download_compat(*args: object, **kwargs: object) -> str:
        use_auth_token = kwargs.pop("use_auth_token", None)
        if use_auth_token is not None and "token" not in kwargs:
            kwargs["token"] = use_auth_token
        try:
            return hf_hub_download(*args, **kwargs)
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

    torch.manual_seed(1234)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    try:
        inference = EncoderClassifier.from_hparams(
            source=args.source,
            revision=revision,
            savedir=savedir,
            run_opts={"device": "cpu"},
        )
    except Exception as error:  # noqa: BLE001 - retain official failure detail
        raise SystemExit(
            "speechbrain_lang_id_dump_reference: the real pinned model could "
            f"not be loaded ({type(error).__name__}: {error})"
        ) from error
    for module in inference.mods.values():
        module.eval()

    pcm = read_pcm16_mono(wav_path)
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

    output = output_dir
    output.mkdir(parents=True, exist_ok=True)
    validate_output_dir(output)
    payloads = {
        "pcm.f32.bin": np.asarray(pcm, dtype="<f4").tobytes(order="C"),
        "features.f32.bin": np.asarray(features[0].detach().cpu(), dtype="<f4").tobytes(
            order="C"
        ),
        "embedding.f32.bin": np.asarray(embedding[0, 0].detach().cpu(), dtype="<f4").tobytes(
            order="C"
        ),
        "scores.f32.bin": np.asarray(out_prob[0].detach().cpu(), dtype="<f4").tobytes(
            order="C"
        ),
        "labels.json": (json.dumps(labels, ensure_ascii=False, indent=2) + "\n").encode(
            "utf-8"
        ),
    }
    artifact_hashes = {
        name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()
    }
    artifact_bytes = {name: len(payload) for name, payload in payloads.items()}

    checkpoint_hashes = {}
    for filename in ["embedding_model.ckpt", "classifier.ckpt", "label_encoder.txt"]:
        path = savedir / filename
        if path.is_symlink() or not path.is_file() or not path.stat().st_size:
            raise SystemExit(
                "speechbrain_lang_id_dump_reference: required upstream checkpoint "
                f"is missing, symlinked, or empty: {path}"
            )
        checkpoint_hashes[filename] = sha256(path)
    manifest = {
        "format": "vokra-speechbrain-lang-id-reference-v1",
        "source": args.source,
        "revision": revision,
        "sample_rate": SAMPLE_RATE,
        "device": "cpu",
        "pcm_samples": int(pcm.size),
        "raw_feature_shape": list(raw_features.shape),
        "feature_shape": list(features.shape),
        "embedding_shape": list(embedding.shape),
        "score_shape": list(out_prob.shape),
        "best_index": best_index,
        "best_label": decoded,
        "best_score": float(score.item()),
        "wav_bytes": wav_path.stat().st_size,
        "wav_sha256": sha256(wav_path),
        "checkpoint_sha256": checkpoint_hashes,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "speechbrain": speechbrain.__version__,
        "artifact_sha256": artifact_hashes,
        "artifact_bytes": artifact_bytes,
    }
    payloads["manifest.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    publish_fixtures({output / name: payload for name, payload in payloads.items()})
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
