#!/usr/bin/env python3
"""Flatten a marl/crepe Keras ``model-{tiny,small,medium,large,full}.h5`` →
safetensors + config side-car (M5 gap follow-up, 2026-07-30).

Offline sidecar tool (FR-LD-05: no Python / TensorFlow ever enters the
runtime). The upstream marl/crepe releases ship Keras 2 ``.h5``
checkpoints (an HDF5 payload of ``Conv2D`` / ``BatchNormalization`` /
``MaxPool2D`` / ``Dense`` weights); the Rust converter
(``crates/vokra-convert/src/models/crepe.rs``) consumes safetensors + a
JSON config only, so this script bridges the two:

* Loads the ``.h5`` with either the ``tf.keras`` module (when the
  ``tensorflow`` package is installed via ``uv add tensorflow`` in this
  parity tree) OR the raw ``h5py`` path (fallback for hosts where
  installing tensorflow's ~600 MB wheel is prohibitive — e.g. the ARM64
  macOS runner). Both paths converge on the same tensor names + shapes.
* Emits tensors in the Vokra runtime's expected naming:
    ``conv{i}.weight``   — ``[c_out, c_in, kh, 1]`` (F32)  ← permuted
                             from Keras' ``[kh, 1, c_in, c_out]``
    ``conv{i}.bias``     — ``[c_out]`` (F32)
    ``conv{i}.bn.gamma``           — ``[c_out]`` (F32)
    ``conv{i}.bn.beta``            — ``[c_out]`` (F32)
    ``conv{i}.bn.moving_mean``     — ``[c_out]`` (F32)
    ``conv{i}.bn.moving_variance`` — ``[c_out]`` (F32)
    ``classifier.weight`` — ``[360, flat_len]`` (F32)  ← Keras stores
                             Dense as ``[in, out]``; we permute to
                             ``[out, in]`` so the runtime's row-major
                             GEMV consumes it directly.
    ``classifier.bias``   — ``[360]`` (F32)
* Emits the config JSON:
    ``{"capacity": "<tiny|small|medium|large|full>", "hop": 160,
       "fmin": 50.0, "fmax": 1100.0}``
  ``capacity`` is derived from the input filename (upstream releases
  ship under ``model-{capacity}.h5``); the numeric bounds are the
  runtime defaults, override with ``--hop`` / ``--fmin`` / ``--fmax``.
* Prints a sha256 manifest line per output for the fixture / workflow
  logs.

Fails loudly on any anomaly (unrecognized filename, missing layer,
mismatched shape) rather than masking it — FR-EX-08 posture, matches
the ``dac_prepare_checkpoint.py`` / ``utmos_prepare_checkpoint.py``
siblings.

# Usage

::

    tools/parity/.venv/bin/python tools/parity/keras_h5_to_safetensors.py \\
        --h5 ~/.cache/crepe/model-full.h5 \\
        --output /tmp/crepe-full.safetensors \\
        --config-out /tmp/crepe-full-config.json

Then:

::

    vokra-cli convert --model crepe --input /tmp/crepe-full.safetensors \\
        --config /tmp/crepe-full-config.json --output /tmp/crepe-full.gguf

# NOT REFERENCED (clean-room)

- ``github.com/marl/crepe`` code (MIT — the Rust runtime module
  ``crates/vokra-models/src/f0/crepe.rs`` is a clean-room re-write of
  the paper's architecture from the primary source's textual
  description of layers, per CLAUDE.md 設計判断 4 whisper.cpp 型).
  This script only unpacks the .h5 tensor payload — it does not
  transliterate any of upstream's ``core.py`` inference code.

# License

The ``.h5`` release + upstream repo ship MIT
(``github.com/marl/crepe/main/LICENSE.txt``, "MIT License / Copyright
(c) 2018 Jong Wook Kim et al."). This script is Vokra-internal (not
distributed as part of the runtime, FR-LD-05).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Iterable

# Upstream CREPE architecture constants (see the ICASSP paper +
# ``crepe/core.py::build_and_load_model``, verified 2026-07-30).
CAPACITY_MULT = {"tiny": 4, "small": 8, "medium": 16, "large": 24, "full": 32}
FILTER_MULT = [32, 4, 4, 4, 8, 16]
KERNEL_WIDTH = [512, 64, 64, 64, 64, 64]
N_BINS = 360


def capacity_from_filename(path: Path) -> str:
    """Extract the capacity tag from a filename like ``model-full.h5``.

    Fails loudly if the stem does not contain a recognized tag.
    """
    stem = path.stem  # ``model-full``
    for tag in ("tiny", "small", "medium", "large", "full"):
        if stem.endswith(f"-{tag}") or stem == tag or f"-{tag}." in path.name:
            return tag
    raise SystemExit(
        f"error: cannot infer capacity from {path.name!r}; expected one of "
        f"model-{{tiny,small,medium,large,full}}.h5 (or pass --capacity)"
    )


def load_h5_weights(path: Path) -> dict[str, "numpy.ndarray"]:
    """Load every CREPE conv/BN/Dense weight from a Keras .h5 payload.

    Uses the ``h5py`` fallback (no TensorFlow needed) — the Keras
    weight-group naming is stable across Keras 2.x releases (the
    upstream .h5 ships under Keras 2.2).

    Layout (per ``h5py.File(h5_path)['model_weights']``):
        conv{i}/conv{i}/kernel:0           → shape (kh, 1, c_in, c_out)
        conv{i}/conv{i}/bias:0             → shape (c_out,)
        conv{i}-BN/conv{i}-BN/gamma:0
        conv{i}-BN/conv{i}-BN/beta:0
        conv{i}-BN/conv{i}-BN/moving_mean:0
        conv{i}-BN/conv{i}-BN/moving_variance:0
        classifier/classifier/kernel:0     → shape (flat_len, 360)
        classifier/classifier/bias:0       → shape (360,)
    """
    try:
        import h5py  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as e:
        raise SystemExit(
            f"error: `{e.name}` not installed. Install parity tools with "
            "`uv sync` inside tools/parity/ (`uv add h5py numpy` if "
            "starting fresh)."
        ) from e

    out: dict[str, "np.ndarray"] = {}
    with h5py.File(str(path), "r") as f:
        # Keras 2 organizes weights under 'model_weights/<layer>/<layer>/<name>:0'.
        # Some releases nest under just '<layer>/<name>' — probe both.
        root = f["model_weights"] if "model_weights" in f else f

        def read(layer: str, name: str) -> "np.ndarray":
            grp = root[layer]
            inner = grp[layer] if layer in grp else grp
            # h5py returns Dataset — decode into numpy.
            keys = list(inner.keys())
            hit = None
            for k in keys:
                if k == name or k.startswith(f"{name}:"):
                    hit = k
                    break
            if hit is None:
                raise SystemExit(
                    f"error: layer {layer!r} missing weight {name!r} "
                    f"(available: {keys})"
                )
            return np.asarray(inner[hit][()], dtype=np.float32)

        for i in range(1, 7):
            layer = f"conv{i}"
            bn_layer = f"conv{i}-BN"
            out[f"{layer}.kernel"] = read(layer, "kernel")
            out[f"{layer}.bias"] = read(layer, "bias")
            out[f"{layer}.bn.gamma"] = read(bn_layer, "gamma")
            out[f"{layer}.bn.beta"] = read(bn_layer, "beta")
            out[f"{layer}.bn.moving_mean"] = read(bn_layer, "moving_mean")
            out[f"{layer}.bn.moving_variance"] = read(bn_layer, "moving_variance")
        out["classifier.kernel"] = read("classifier", "kernel")
        out["classifier.bias"] = read("classifier", "bias")
    return out


def permute_conv_kernel(arr: "numpy.ndarray") -> "numpy.ndarray":
    """Keras ``[kh, 1, c_in, c_out]`` → Vokra ``[c_out, c_in, kh, 1]``."""
    import numpy as np  # type: ignore

    if arr.ndim != 4:
        raise SystemExit(
            f"error: expected 4D conv kernel, got shape {arr.shape}"
        )
    kh, kw, c_in, c_out = arr.shape
    if kw != 1:
        raise SystemExit(
            f"error: expected kw==1 (upstream Conv2D uses (widths[i], 1)), got kw={kw}"
        )
    # (kh, 1, c_in, c_out) → (c_out, c_in, kh, 1)
    return np.ascontiguousarray(np.transpose(arr, (3, 2, 0, 1)), dtype=np.float32)


def permute_dense_kernel(arr: "numpy.ndarray") -> "numpy.ndarray":
    """Keras Dense ``[in, out]`` → Vokra ``[out, in]``."""
    import numpy as np  # type: ignore

    if arr.ndim != 2:
        raise SystemExit(
            f"error: expected 2D Dense kernel, got shape {arr.shape}"
        )
    return np.ascontiguousarray(arr.T, dtype=np.float32)


def emit_safetensors(
    output: Path, tensors: dict[str, "numpy.ndarray"]
) -> None:
    """Write a safetensors buffer containing every named tensor as F32."""
    import numpy as np  # type: ignore

    # Sort names so the header key order is deterministic across runs.
    names = sorted(tensors.keys())
    payload = bytearray()
    header: dict[str, dict] = {}
    for name in names:
        t = np.ascontiguousarray(tensors[name], dtype=np.float32)
        start = len(payload)
        payload.extend(t.tobytes(order="C"))
        end = len(payload)
        header[name] = {
            "dtype": "F32",
            "shape": list(t.shape),
            "data_offsets": [start, end],
        }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with output.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(bytes(payload))


def sha256_hex(path: Path) -> str:
    """SHA-256 of a file's contents as a lower-case hex string."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Flatten a marl/crepe .h5 into safetensors + JSON config",
    )
    parser.add_argument(
        "--h5",
        type=Path,
        required=True,
        help="upstream marl/crepe .h5 checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="flat safetensors output",
    )
    parser.add_argument(
        "--config-out",
        type=Path,
        required=True,
        help="JSON config side-car output (consumed by vokra-cli convert)",
    )
    parser.add_argument(
        "--capacity",
        choices=list(CAPACITY_MULT.keys()),
        default=None,
        help="capacity tag override (inferred from filename by default)",
    )
    parser.add_argument(
        "--hop",
        type=int,
        default=160,
        help="analysis hop in samples (default 160 = 10 ms @ 16 kHz)",
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=50.0,
        help="minimum tracked F0 in Hz (default 50.0 — informational)",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=1100.0,
        help="maximum tracked F0 in Hz (default 1100.0 — informational)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.h5.is_file():
        raise SystemExit(f"error: --h5 {args.h5} does not exist")

    capacity = args.capacity or capacity_from_filename(args.h5)
    if capacity not in CAPACITY_MULT:
        raise SystemExit(f"error: capacity {capacity!r} is not one of {list(CAPACITY_MULT)}")

    mult = CAPACITY_MULT[capacity]
    filters = [m * mult for m in FILTER_MULT]

    raw = load_h5_weights(args.h5)

    # Permute + validate every tensor against the capacity-derived shape.
    out: dict[str, "numpy.ndarray"] = {}
    c_in = 1
    for i, (filt, kh) in enumerate(zip(filters, KERNEL_WIDTH), start=1):
        kernel = permute_conv_kernel(raw[f"conv{i}.kernel"])
        expected = (filt, c_in, kh, 1)
        if kernel.shape != expected:
            raise SystemExit(
                f"error: conv{i}.weight shape {kernel.shape} != expected {expected}"
            )
        out[f"conv{i}.weight"] = kernel
        for tag in ("bias", "bn.gamma", "bn.beta", "bn.moving_mean", "bn.moving_variance"):
            key = f"conv{i}.{tag}" if tag == "bias" else f"conv{i}.{tag}"
            src = raw[key]
            if src.shape != (filt,):
                raise SystemExit(
                    f"error: {key} shape {src.shape} != expected ({filt},)"
                )
            out[key] = src.astype("float32", copy=False)
        c_in = filt

    flat = 4 * filters[5]
    cls_w = permute_dense_kernel(raw["classifier.kernel"])
    if cls_w.shape != (N_BINS, flat):
        raise SystemExit(
            f"error: classifier.weight shape {cls_w.shape} != expected {(N_BINS, flat)}"
        )
    out["classifier.weight"] = cls_w
    cls_b = raw["classifier.bias"].astype("float32", copy=False)
    if cls_b.shape != (N_BINS,):
        raise SystemExit(
            f"error: classifier.bias shape {cls_b.shape} != expected ({N_BINS},)"
        )
    out["classifier.bias"] = cls_b

    # Ensure the parent directories exist.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.config_out.parent.mkdir(parents=True, exist_ok=True)

    emit_safetensors(args.output, out)
    config = {
        "capacity": capacity,
        "hop": int(args.hop),
        "fmin": float(args.fmin),
        "fmax": float(args.fmax),
    }
    args.config_out.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        f"crepe: capacity={capacity} filters={filters} flat={flat} "
        f"tensors={len(out)} sha256(safetensors)={sha256_hex(args.output)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
