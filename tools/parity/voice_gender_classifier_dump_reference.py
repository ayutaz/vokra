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
import subprocess
import sys
from pathlib import Path
from typing import Any

UPSTREAM_REPOSITORY = "https://github.com/JaesungHuh/voice-gender-classifier.git"
UPSTREAM_REVISION = "49bcbecfd929ba5a043bde645fdff1a375eb79c7"
UPSTREAM_HF_REVISION = "db1222153bd60337e900be22add7af180452adc0"
SAMPLE_RATE = 16_000
CLASS_LABELS = ["male", "female"]
DUMPER_VERSION = 2
CHECKPOINT_IDENTITY_STATUS = "UNRESOLVED"

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
    print(
        "voice-gender reference BLOCKED: fixed checkpoint bytes/license evidence is unresolved",
        file=sys.stderr,
    )
    return 2


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
    if not (checkout / "model.py").is_file():
        raise ValueError(f"not a JaesungHuh voice-gender checkout: {checkout}")
    if git_output(checkout, "rev-parse", "HEAD") != UPSTREAM_REVISION:
        raise ValueError(f"upstream checkout is not pinned to {UPSTREAM_REVISION}")
    if git_output(checkout, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("upstream checkout is dirty")


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
    path.write_bytes(np.ascontiguousarray(values, dtype=np.dtype(dtype)).tobytes())


def self_test() -> None:
    assert dependency_gate() == 2
    required = [
        "ECAPA_gender",
        "load_file",
        "UPSTREAM_REVISION",
        "torch.no_grad",
        "register_forward_pre_hook",
        "DUMPER_VERSION",
    ]
    source = Path(__file__).read_text(encoding="utf-8")
    missing = [token for token in required if token not in source]
    if missing:
        raise AssertionError(f"reference contract missing: {missing}")
    forbidden_reimplementation = "nn." + "Linear(192, 2)"
    if forbidden_reimplementation in source:
        raise AssertionError("dumper must not contain a model reimplementation")
    print("voice_gender_classifier_dump_reference.py self-test: PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--upstream-src", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--pcm", type=Path)
    source.add_argument("--canned", action="store_true")
    parser.add_argument("--out-dir", type=Path)
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
    bind_runtime_dependencies()
    checkpoint = args.checkpoint.expanduser().resolve()
    checkout = args.upstream_src.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint}")
    validate_checkout(checkout)
    pcm = canned_pcm() if args.canned else read_pcm(args.pcm.expanduser().resolve())
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
    files = {
        "pcm.f32": (pcm, "<f4"),
        "features.f32": (features_np, "<f4"),
        "embedding.f32": (embedding_np, "<f4"),
        "logits.f32": (logits_np, "<f4"),
        "probabilities.f32": (probabilities, "<f4"),
        "argmax.u32": (argmax, "<u4"),
    }
    for name, (values, dtype) in files.items():
        write_raw(out_dir / name, values, dtype)
    metadata = {
        "dumper_version": DUMPER_VERSION,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_hf_revision": UPSTREAM_HF_REVISION,
        "upstream_class": "model.ECAPA_gender",
        "checkpoint_sha256": sha256_file(checkpoint),
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
            name: {"sha256": sha256_file(out_dir / name), "bytes": (out_dir / name).stat().st_size}
            for name in files
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote independent voice-gender fixtures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
