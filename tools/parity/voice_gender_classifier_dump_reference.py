#!/usr/bin/env python3
"""Dump independent JaesungHuh voice-gender classifier fixtures.

The oracle imports ``model.ECAPA_gender`` from a clean checkout of the exact
upstream Git revision. It loads only the upstream safetensors checkpoint via
``safetensors.torch.load_file`` and records the official frontend, the exact
post-bn6/ReLU vector entering ``fc7``, logits, probabilities, and class
decision. It does not reimplement the model and has no fallback path when the
pinned source cannot be imported.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

UPSTREAM_REPOSITORY = "https://github.com/JaesungHuh/voice-gender-classifier.git"
UPSTREAM_REVISION = "49bcbecfd929ba5a043bde645fdff1a375eb79c7"
UPSTREAM_HF_REVISION = "db1222153bd60337e900be22add7af180452adc0"
UPSTREAM_HF_FILE = "model.safetensors"
CHECKPOINT_BYTES = 61_907_512
CHECKPOINT_SHA256 = "2d8e0be1fdf159d60d5087416e6f6277c5e30ce9e33a61c767a9a409e6c503c5"
UPSTREAM_LICENSE_FILE = "LICENSE"
UPSTREAM_LICENSE_SPDX = "MIT"
UPSTREAM_LICENSE_COPYRIGHT = "Copyright (c) 2024 jaesunghuh"
UPSTREAM_HF_LICENSE = "mit"
SAMPLE_RATE = 16_000
CLASS_LABELS = ["male", "female"]
DUMPER_VERSION = 3
CHECKPOINT_IDENTITY_STATUS = "AUTHENTICATED_FIXED"

np: Any
torch: Any
load_file: Any


def bind_runtime_dependencies() -> None:
    """Import model dependencies only after the dependency-free self-test."""
    global np, torch, load_file
    import numpy as numpy
    import torch as torch_module
    from safetensors.torch import load_file as safetensors_load_file

    np = numpy
    torch = torch_module
    load_file = safetensors_load_file


def dependency_gate() -> int:
    """Check that this dumper is compiled against the fixed identity contract."""
    if (
        not re.fullmatch(r"[0-9a-f]{64}", CHECKPOINT_SHA256)
        or CHECKPOINT_BYTES <= 0
        or UPSTREAM_HF_FILE != "model.safetensors"
        or UPSTREAM_LICENSE_FILE != "LICENSE"
        or UPSTREAM_LICENSE_SPDX != "MIT"
        or UPSTREAM_HF_LICENSE != "mit"
    ):
        print("voice-gender reference BLOCKED: fixed identity contract is invalid", file=sys.stderr)
        return 2
    return 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(checkout: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def validate_checkout(checkout: Path) -> None:
    reject_symlink_ancestry(checkout, "upstream source")
    if checkout.is_symlink() or not checkout.is_dir():
        raise ValueError(f"upstream source must be a regular non-symlink directory: {checkout}")
    if not (checkout / "model.py").is_file():
        raise ValueError(f"not a JaesungHuh voice-gender checkout: {checkout}")
    if git_output(checkout, "rev-parse", "HEAD") != UPSTREAM_REVISION:
        raise ValueError(f"upstream checkout is not pinned to {UPSTREAM_REVISION}")
    if git_output(checkout, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("upstream checkout is dirty")
    license_path = checkout / UPSTREAM_LICENSE_FILE
    if not license_path.is_file() or license_path.is_symlink():
        raise ValueError("upstream checkout is missing its primary MIT license file")
    license_text = license_path.read_text(encoding="utf-8")
    normalized_license = license_text.casefold()
    required_license_evidence = (
        "mit license",
        UPSTREAM_LICENSE_COPYRIGHT.casefold(),
        "permission is hereby granted, free of charge",
    )
    if any(marker not in normalized_license for marker in required_license_evidence):
        raise ValueError("upstream checkout does not contain the pinned standard MIT license evidence")


def validate_checkpoint(checkpoint: Path) -> None:
    if checkpoint.name != UPSTREAM_HF_FILE:
        raise ValueError(f"checkpoint filename must be exactly {UPSTREAM_HF_FILE}")
    actual_bytes = checkpoint.stat().st_size
    if actual_bytes != CHECKPOINT_BYTES:
        raise ValueError(f"checkpoint byte size mismatch: {actual_bytes} != {CHECKPOINT_BYTES}")
    actual_sha256 = sha256_file(checkpoint)
    if actual_sha256 != CHECKPOINT_SHA256:
        raise ValueError(f"checkpoint SHA-256 mismatch: {actual_sha256} != {CHECKPOINT_SHA256}")


def execution_context_allowed(system: str, machine: str, publish_on_vast: str | None) -> bool:
    return system == "Linux" and machine == "x86_64" and publish_on_vast == "1"


def require_vast_context() -> None:
    if not execution_context_allowed(
        platform.system(), platform.machine(), os.environ.get("VOKRA_PUBLISH_ON_VAST")
    ):
        raise RuntimeError("voice-gender checkpoint execution requires VAST Linux x86_64 context")


def canned_pcm() -> np.ndarray:
    count = 2 * SAMPLE_RATE
    time = np.arange(count, dtype=np.float64) / SAMPLE_RATE
    signal = 0.35 * np.sin(2.0 * np.pi * 180.0 * time)
    signal += 0.12 * np.sin(2.0 * np.pi * 360.0 * time + 0.2)
    signal[count // 2 : count // 2 + SAMPLE_RATE // 10] = 0.0
    return np.ascontiguousarray(signal.astype(np.float32))


def read_pcm(path: Path) -> np.ndarray:
    import soundfile as sf

    signal, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"input must be {SAMPLE_RATE} Hz, got {sample_rate}")
    pcm = signal.mean(axis=1, dtype=np.float32)
    if pcm.size == 0 or not np.isfinite(pcm).all():
        raise ValueError("input PCM is empty or non-finite")
    return np.ascontiguousarray(pcm)


def import_model(checkout: Path) -> Any:
    sys.path.insert(0, str(checkout))
    try:
        module = importlib.import_module("model")
        model_type = getattr(module, "ECAPA_gender")
        return model_type(C=1024)
    finally:
        sys.path.pop(0)


def write_raw(path: Path, values: np.ndarray, dtype: str) -> None:
    write_no_clobber(path, np.ascontiguousarray(values, dtype=np.dtype(dtype)).tobytes())


def reject_symlink_ancestry(path: Path | str, label: str) -> None:
    raw = os.fspath(path)
    if any(component in {".", ".."} for component in raw.split("/")):
        raise ValueError(f"{label} must not contain lexical dot components")
    path = Path(raw)
    absolute = path if path.is_absolute() else Path.cwd() / path
    for ancestor in (absolute, *absolute.parents):
        # macOS exposes /var as the system /private/var alias; it is not
        # user-controlled output redirection and is safe to traverse.
        if ancestor.is_symlink() and ancestor != Path("/var"):
            raise ValueError(f"{label} has symlink ancestry: {ancestor}")


def validate_output_dir(path: Path | str) -> None:
    reject_symlink_ancestry(path, "--out-dir")
    path = Path(os.fspath(path))
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise ValueError("--out-dir must be a non-symlink directory")
    if path.exists() and any(path.iterdir()):
        raise ValueError("--out-dir must be empty to prevent fixture clobbering")


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
        raise ValueError(f"fixture output parent is not a regular directory: {path.parent}")


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


def self_test() -> None:
    assert dependency_gate() == 0
    required = [
        "ECAPA_gender",
        "load_file",
        "UPSTREAM_REVISION",
        "UPSTREAM_HF_REVISION",
        "UPSTREAM_HF_FILE",
        "CHECKPOINT_BYTES",
        "CHECKPOINT_SHA256",
        "UPSTREAM_LICENSE_FILE",
        "UPSTREAM_LICENSE_SPDX",
        "UPSTREAM_LICENSE_COPYRIGHT",
        "UPSTREAM_HF_LICENSE",
        "validate_checkpoint",
        "require_vast_context",
        "execution_context_allowed",
        "permission is hereby granted, free of charge",
        "torch.no_grad",
        "register_forward_pre_hook",
        "DUMPER_VERSION",
    ]
    source = Path(__file__).read_text(encoding="utf-8")
    missing = [token for token in required if token not in source]
    if missing:
        raise AssertionError(f"reference contract missing: {missing}")
    assert execution_context_allowed("Linux", "x86_64", "1")
    for context in (("Darwin", "arm64", "1"), ("Linux", "aarch64", "1"), ("Linux", "x86_64", None)):
        assert not execution_context_allowed(*context)
    forbidden_reimplementation = "nn." + "Linear(192, 2)"
    if forbidden_reimplementation in source:
        raise AssertionError("dumper must not contain a model reimplementation")
    with tempfile.TemporaryDirectory(prefix="voice-gender-dump-self-test-") as directory:
        root = Path(directory)
        valid = root / "output"
        validate_output_dir(valid)
        valid.mkdir()
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
        second_payload = b"keep-second"
        second.write_bytes(second_payload)
        try:
            publish_fixtures({first: b"first", second: b"replace-second"})
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
        except ValueError:
            pass
        else:
            raise AssertionError("dot component accepted")
        real = root / "real"
        real.mkdir()
        link = root / "link"
        link.symlink_to(real, target_is_directory=True)
        try:
            validate_output_dir(link / "output")
        except ValueError:
            pass
        else:
            raise AssertionError("symlink ancestor accepted")
    print("voice_gender_classifier_dump_reference.py self-test: PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--upstream-src")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--pcm")
    source.add_argument("--canned", action="store_true")
    parser.add_argument("--out-dir")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        if any(value is not None for value in (args.checkpoint, args.upstream_src, args.pcm, args.out_dir)) or args.canned:
            raise ValueError("--self-test accepts no fixture arguments")
        self_test()
        return 0
    if args.checkpoint is None or args.upstream_src is None or args.out_dir is None:
        raise ValueError("--checkpoint, --upstream-src, and --out-dir are required")
    if (args.pcm is None) == (not args.canned):
        raise ValueError("exactly one of --pcm or --canned is required")
    if dependency_gate() != 0:
        return 2
    require_vast_context()
    bind_runtime_dependencies()
    validate_output_dir(args.out_dir)
    reject_symlink_ancestry(args.checkpoint, "checkpoint")
    reject_symlink_ancestry(args.upstream_src, "upstream source")
    checkpoint = Path(args.checkpoint).expanduser()
    checkout = Path(args.upstream_src).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint}")
    validate_checkpoint(checkpoint)
    validate_checkout(checkout)
    if args.canned:
        pcm = canned_pcm()
    else:
        pcm_path = Path(args.pcm).expanduser()
        reject_symlink_ancestry(pcm_path, "--pcm")
        if pcm_path.is_symlink() or not pcm_path.is_file():
            raise ValueError(f"--pcm must be a regular non-symlink file: {pcm_path}")
        pcm = read_pcm(pcm_path)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = import_model(checkout)
    state = load_file(str(checkpoint), device="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    waveform = torch.from_numpy(pcm).unsqueeze(0)
    with torch.no_grad():
        features = model.logtorchfbank(waveform).squeeze(0).transpose(0, 1).cpu()
        embedding_capture: dict[str, torch.Tensor] = {}

        def capture(_module: Any, _inputs: Any) -> None:
            if not _inputs or _inputs[0].ndim != 2:
                raise RuntimeError("official fc7 pre-hook did not receive [batch, 192] input")
            embedding_capture["fc7_input"] = _inputs[0].detach().cpu()

        handle = model.fc7.register_forward_pre_hook(capture)
        try:
            logits = model(waveform).squeeze(0).cpu()
        finally:
            handle.remove()
    embedding = embedding_capture.get("fc7_input")
    if embedding is None:
        raise RuntimeError("official fc7 pre-hook did not fire")
    probabilities = torch.softmax(logits, dim=0).numpy().astype(np.float32)
    logits_np = logits.numpy().astype(np.float32)
    embedding_np = embedding.squeeze(0).numpy().astype(np.float32)
    features_np = features.numpy().astype(np.float32)
    if logits_np.shape != (2,) or embedding_np.shape != (192,):
        raise RuntimeError(f"unexpected classifier shapes: {logits_np.shape}, {embedding_np.shape}")
    if not np.isfinite(embedding_np).all() or not np.isfinite(logits_np).all():
        raise RuntimeError("official embedding/logits contain non-finite values")
    argmax = np.asarray([int(np.argmax(probabilities))], dtype=np.uint32)
    out_dir.mkdir(parents=True, exist_ok=True)
    validate_output_dir(out_dir)
    payloads = {
        "pcm.f32": np.asarray(pcm, dtype="<f4").tobytes(order="C"),
        "features.f32": np.asarray(features_np, dtype="<f4").tobytes(order="C"),
        "embedding.f32": np.asarray(embedding_np, dtype="<f4").tobytes(order="C"),
        "logits.f32": np.asarray(logits_np, dtype="<f4").tobytes(order="C"),
        "probabilities.f32": np.asarray(probabilities, dtype="<f4").tobytes(order="C"),
        "argmax.u32": np.asarray(argmax, dtype="<u4").tobytes(order="C"),
    }
    metadata = {
        "dumper_version": DUMPER_VERSION,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_hf_revision": UPSTREAM_HF_REVISION,
        "checkpoint_file": UPSTREAM_HF_FILE,
        "checkpoint_bytes": CHECKPOINT_BYTES,
        "upstream_class": "model.ECAPA_gender",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "upstream_license": UPSTREAM_LICENSE_SPDX,
        "upstream_license_file": UPSTREAM_LICENSE_FILE,
        "upstream_license_copyright": UPSTREAM_LICENSE_COPYRIGHT,
        "upstream_hf_license": UPSTREAM_HF_LICENSE,
        "checkpoint_identity_status": CHECKPOINT_IDENTITY_STATUS,
        "sample_rate": SAMPLE_RATE,
        "n_mels": 80,
        "n_fft": 512,
        "win_length": 400,
        "hop_length": 160,
        "feature_frames": int(features_np.shape[0]),
        "feature_dim": int(features_np.shape[1]),
        "embedding_dim": int(embedding_np.shape[0]),
        "class_labels": CLASS_LABELS,
        "outputs": {
            name: {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
            for name, payload in payloads.items()
        },
    }
    payloads["meta.json"] = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    publish_fixtures({out_dir / name: payload for name, payload in payloads.items()})
    print(f"wrote independent voice-gender fixtures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
