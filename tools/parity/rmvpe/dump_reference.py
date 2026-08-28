#!/usr/bin/env python3
"""Generate independent RMVPE parity fixtures from the fixed upstream code.

This sidecar imports ``src.inference.RMVPE`` from an exact clean checkout of
``yxlllc/RMVPE`` commit ``0aabafba18289ca938a73af0b0297686abf4922d``. It
does not mirror the network in local Python. A pre-hook on the upstream GRU
captures its 384-feature input and a model hook captures the already-sigmoid
360-class probabilities.

Run only on VAST with Python 3.12 through this directory's uv project. The
upstream ``.pt`` is a pickle and must come from a trusted, digest-recorded
source. No upstream code or checkpoint enters the Vokra runtime.

Outputs are raw little-endian buffers without numpy headers:

* ``pcm.f32``: reference 16-kHz mono input, ``[samples]``;
* ``hidden.f32``: input to ``fc.0.gru``, ``[frames, 384]``;
* ``probabilities.f32``: upstream sigmoid output, ``[frames, 360]``;
* ``argmax.u32``: class argmax, with ``0xffffffff`` for unvoiced frames;
* ``f0.f32``: upstream nine-bin local-average decode, ``[frames]``;
* ``meta.json``: fixed revisions, digests, shapes, and generation settings.
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

import numpy as np
import soundfile as sf
import torch


UPSTREAM_REPOSITORY = "https://github.com/yxlllc/RMVPE"
UPSTREAM_REVISION = "0aabafba18289ca938a73af0b0297686abf4922d"
SAMPLE_RATE = 16_000
HOP_LENGTH = 160
FEATURE_DIM = 384
N_CLASS = 360
DEFAULT_THRESHOLD = 0.03
UNVOICED_CLASS = np.uint32(0xFFFF_FFFF)
DUMPER_VERSION = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git_output(checkout: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def validate_upstream_checkout(checkout: Path) -> None:
    if not (checkout / "src" / "inference.py").is_file():
        raise ValueError(f"{checkout} is not an yxlllc/RMVPE checkout")
    revision = git_output(checkout, "rev-parse", "HEAD")
    if revision != UPSTREAM_REVISION:
        raise ValueError(
            f"upstream checkout is {revision}; expected fixed {UPSTREAM_REVISION}"
        )
    dirty = git_output(checkout, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise ValueError(
            "upstream checkout is dirty; reference provenance requires an exact "
            f"clean {UPSTREAM_REVISION} tree"
        )


def canned_pcm() -> np.ndarray:
    """Return a deterministic two-second multitone/chirp reference clip."""

    count = 2 * SAMPLE_RATE
    time = np.arange(count, dtype=np.float64) / SAMPLE_RATE
    phase = 2.0 * np.pi * (110.0 * time + 0.5 * 330.0 * time * time / 2.0)
    signal = (
        0.42 * np.sin(phase)
        + 0.18 * np.sin(2.0 * np.pi * 220.0 * time + 0.3)
        + 0.08 * np.sin(2.0 * np.pi * 660.0 * time + 0.7)
    )
    # Pin explicit silence so the voiced threshold is exercised too.
    signal[count // 2 : count // 2 + SAMPLE_RATE // 5] = 0.0
    return np.ascontiguousarray(signal.astype(np.float32))


def read_pcm(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"{path} is {sample_rate} Hz; provide exact {SAMPLE_RATE}-Hz PCM so "
            "the reference and Rust paths receive identical samples"
        )
    mono = audio.mean(axis=1, dtype=np.float32)
    if mono.size == 0:
        raise ValueError(f"{path} contains no samples")
    if not np.isfinite(mono).all():
        raise ValueError(f"{path} contains non-finite samples")
    return np.ascontiguousarray(mono, dtype=np.float32)


def import_upstream_predictor(checkout: Path, pt_path: Path) -> Any:
    sys.path.insert(0, str(checkout))
    try:
        module = importlib.import_module("src.inference")
        predictor_type = getattr(module, "RMVPE")
        return predictor_type(str(pt_path), hop_length=HOP_LENGTH)
    finally:
        sys.path.pop(0)


def run_reference(
    predictor: Any, pcm: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    captured: dict[str, torch.Tensor] = {}

    def capture_gru_input(_module: Any, args: tuple[Any, ...]) -> None:
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("upstream fc.0.gru pre-hook did not receive a tensor")
        captured["hidden"] = args[0].detach().cpu()

    def capture_probabilities(
        _module: Any, _args: tuple[Any, ...], output: Any
    ) -> None:
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("upstream E2E0 hook did not return a tensor")
        captured["probabilities"] = output.detach().cpu()

    try:
        gru = predictor.model.fc[0].gru
    except (AttributeError, IndexError, TypeError) as error:
        raise RuntimeError(
            "fixed upstream topology lacks predictor.model.fc[0].gru"
        ) from error

    gru_handle = gru.register_forward_pre_hook(capture_gru_input)
    model_handle = predictor.model.register_forward_hook(capture_probabilities)
    try:
        f0 = predictor.infer_from_audio(
            pcm,
            sample_rate=SAMPLE_RATE,
            device=torch.device("cpu"),
            thred=threshold,
            use_viterbi=False,
        )
    finally:
        gru_handle.remove()
        model_handle.remove()

    frame_count = pcm.size // HOP_LENGTH + 1
    hidden_tensor = captured.get("hidden")
    probabilities_tensor = captured.get("probabilities")
    if hidden_tensor is None or probabilities_tensor is None:
        raise RuntimeError("upstream hooks did not both fire")
    if hidden_tensor.ndim != 3 or tuple(hidden_tensor.shape[:1]) != (1,):
        raise RuntimeError(f"unexpected GRU input shape {tuple(hidden_tensor.shape)}")
    if probabilities_tensor.ndim != 3 or tuple(probabilities_tensor.shape[:1]) != (1,):
        raise RuntimeError(
            f"unexpected probability shape {tuple(probabilities_tensor.shape)}"
        )

    hidden = hidden_tensor[0, :frame_count].numpy().astype(np.float32, copy=False)
    probabilities = (
        probabilities_tensor[0, :frame_count].numpy().astype(np.float32, copy=False)
    )
    f0_array = np.asarray(f0, dtype=np.float32)
    if hidden.shape != (frame_count, FEATURE_DIM):
        raise RuntimeError(
            f"upstream hidden shape {hidden.shape} != {(frame_count, FEATURE_DIM)}"
        )
    if probabilities.shape != (frame_count, N_CLASS):
        raise RuntimeError(
            f"upstream probabilities shape {probabilities.shape} != "
            f"{(frame_count, N_CLASS)}"
        )
    if f0_array.shape != (frame_count,):
        raise RuntimeError(f"upstream f0 shape {f0_array.shape} != {(frame_count,)}")
    for name, values in [
        ("hidden", hidden),
        ("probabilities", probabilities),
        ("f0", f0_array),
    ]:
        if not np.isfinite(values).all():
            raise RuntimeError(f"upstream {name} contains non-finite values")
    return hidden, probabilities, f0_array


def write_raw(path: Path, values: np.ndarray, dtype: str) -> None:
    contiguous = np.ascontiguousarray(values, dtype=np.dtype(dtype))
    path.write_bytes(contiguous.tobytes(order="C"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt-path", type=Path, required=True)
    parser.add_argument("--upstream-src", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pcm", type=Path, help="exact 16-kHz WAV input")
    source.add_argument("--canned", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pt_path = args.pt_path.expanduser().resolve()
    checkout = args.upstream_src.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not pt_path.is_file():
        raise ValueError(f"checkpoint does not exist: {pt_path}")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be within [0, 1]")
    validate_upstream_checkout(checkout)

    pcm = canned_pcm() if args.canned else read_pcm(args.pcm.expanduser().resolve())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    predictor = import_upstream_predictor(checkout, pt_path)
    hidden, probabilities, f0 = run_reference(predictor, pcm, args.threshold)

    voiced = probabilities.max(axis=1) >= args.threshold
    argmax = probabilities.argmax(axis=1).astype(np.uint32)
    argmax = np.where(voiced, argmax, UNVOICED_CLASS).astype(np.uint32)

    out_dir.mkdir(parents=True, exist_ok=True)
    pcm_path = out_dir / "pcm.f32"
    hidden_path = out_dir / "hidden.f32"
    probabilities_path = out_dir / "probabilities.f32"
    argmax_path = out_dir / "argmax.u32"
    f0_path = out_dir / "f0.f32"
    write_raw(pcm_path, pcm, "<f4")
    write_raw(hidden_path, hidden, "<f4")
    write_raw(probabilities_path, probabilities, "<f4")
    write_raw(argmax_path, argmax, "<u4")
    write_raw(f0_path, f0, "<f4")

    meta = {
        "dumper_version": DUMPER_VERSION,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_class": "src.inference.RMVPE / src.model.E2E0",
        "checkpoint_path": str(pt_path),
        "checkpoint_sha256": sha256_file(pt_path),
        "sample_rate": SAMPLE_RATE,
        "hop_length": HOP_LENGTH,
        "sample_count": int(pcm.size),
        "n_frames": int(hidden.shape[0]),
        "feature_dim": int(hidden.shape[1]),
        "n_class": int(probabilities.shape[1]),
        "threshold": float(args.threshold),
        "unvoiced_argmax_sentinel": int(UNVOICED_CLASS),
        "decode": "src.utils.to_local_average_f0(use_viterbi=False)",
        "pcm_sha256": sha256_bytes(pcm.astype("<f4", copy=False).tobytes()),
        "outputs": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in [pcm_path, hidden_path, probabilities_path, argmax_path, f0_path]
        },
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote independent RMVPE reference fixtures to {out_dir}")
    print(f"  upstream revision: {UPSTREAM_REVISION}")
    print(f"  frames/features:   {hidden.shape[0]} x {hidden.shape[1]}")
    print(f'  export VOKRA_RMVPE_REAL_PCM="{pcm_path}"')
    print(f'  export VOKRA_RMVPE_REAL_HIDDEN="{hidden_path}"')
    print(f'  export VOKRA_RMVPE_REAL_HIDDEN_FEATURE_DIM="{hidden.shape[1]}"')
    print(f'  export VOKRA_RMVPE_REAL_ARGMAX="{argmax_path}"')
    print(f'  export VOKRA_RMVPE_REAL_F0="{f0_path}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
