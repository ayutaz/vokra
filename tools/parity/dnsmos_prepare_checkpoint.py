#!/usr/bin/env python3
"""Flatten Microsoft DNSMOS's two ONNX checkpoints into a single merged
safetensors bundle (coverage-audit Wave A ticket ``dnsmos-p808-p835``,
2026-08-03).

Offline sidecar tool (FR-LD-05: no Python / ONNX ever enters the runtime).
DNSMOS ships as two ONNX files inside
``github.com/microsoft/DNS-Challenge/tree/master/DNSMOS/``:

* ``model_v8.onnx``     — the P.808 predictor (single overall MOS scalar,
                           ITU-T P.808 scale).
* ``sig_bak_ovr.onnx``  — the P.835 predictor (three scalars: signal /
                           background / overall, ITU-T P.835 scale).

Vokra's Rust converter (``crates/vokra-convert/src/models/dnsmos.rs``)
consumes safetensors only, so this script bridges the two by walking
each ONNX graph's ``initializer`` list (the constant tensors that carry
the model weights), prefixing every tensor name with the sub-model tag
(``p808.<upstream_name>`` / ``p835.<upstream_name>``), and emitting a
single merged safetensors alongside a sha256 manifest.

The prefixing scheme is what the Rust converter's ``bundle_variants``
detection walks; the future ``vokra_eval::dnsmos::from_gguf`` binder
consumes the prefix to route each tensor to the right sub-model.

# License

MIT (Microsoft DNS-Challenge, ``LICENSE`` in the repo root, verified
2026-08-03). Every ONNX initializer is a plain array of floating-point
weights — no code is executed at parse time (see the ``onnx.load``
posture below).

# Design decisions

* **ONNX parser**: uses the ``onnx`` Python package (Apache-2.0). It
  parses the protobuf schema and never executes model code — the load
  is data-only, matching FR-LD-05's spirit even though this tool is
  offline. We deliberately avoid ``onnxruntime`` (which would drag in a
  full runtime) and hand-write nothing that would risk pickle-style
  code execution.

* **Dtype policy**: F32 / F16 / BF16 pass through verbatim. Any other
  dtype (INT8 quantized weights in a future DNSMOS revision, for
  instance) is a hard error rather than a silent drop (FR-EX-08).
  DNSMOS as of 2026-08-03 ships F32 end-to-end.

* **Name policy**: initializer names are preserved verbatim after the
  ``p808.`` / ``p835.`` prefix. ONNX sometimes suffixes initializer
  names with a serial (``…_0``) or omits the leading module path; we do
  not rewrite these because the Rust binder walks whatever names the
  prep script emits and any mangling would need to travel with the
  binder to stay consistent.

* **Partial bundle**: passing only one of ``--p808`` / ``--p835`` is
  allowed — the Rust converter's ``bundle_variants`` detection
  faithfully advertises the truthful subset in the emitted GGUF. Both
  are recommended for the canonical Vokra publication (the two scores
  make sense together).

# Usage

::

    uv add onnx safetensors numpy
    uv run python tools/parity/dnsmos_prepare_checkpoint.py \\
        --p808 ~/checkpoints/dnsmos/model_v8.onnx \\
        --p835 ~/checkpoints/dnsmos/sig_bak_ovr.onnx \\
        --output ~/checkpoints/dnsmos/model.safetensors

Then::

    vokra-cli convert --model dnsmos-p808-p835 \\
        --input ~/checkpoints/dnsmos/model.safetensors \\
        --output ~/gguf/dnsmos-p808-p835.gguf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

LOG_PREFIX = "dnsmos_prepare_checkpoint:"


def _log(msg: str) -> None:
    print(f"{LOG_PREFIX} {msg}", file=sys.stderr)


# ONNX TensorProto.DataType → safetensors dtype string. See
# https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3 (TensorProto.DataType).
ONNX_DTYPE_TO_ST = {
    1: "F32",   # FLOAT
    10: "F16",  # FLOAT16
    16: "BF16", # BFLOAT16
}


def _initializer_bytes(tensor):
    """Extract an ONNX initializer's raw little-endian bytes.

    ONNX initializers store their payload in one of two ways:

    * ``raw_data``: little-endian byte string (the common path since
      ONNX 1.3+; every F32/F16 initializer in DNSMOS lands here).
    * per-dtype typed lists (``float_data`` / ``int32_data`` / …): a
      fallback the ONNX exporter uses when the raw_data field is empty.

    We prefer raw_data (byte-identical to what the safetensors writer
    below needs) and fall back to numpy conversion only when raw_data
    is empty.
    """
    if tensor.raw_data:
        return tensor.raw_data
    # Fallback: DNSMOS as of 2026-08-03 does not exercise this path
    # (both ONNX files use raw_data end-to-end), but keeping the branch
    # means a future DNSMOS revision that changes emit style will still
    # convert rather than silently produce an empty payload.
    import numpy as np

    dtype = tensor.data_type
    if dtype == 1:  # FLOAT
        arr = np.asarray(list(tensor.float_data), dtype=np.float32)
    elif dtype == 10:  # FLOAT16
        arr = np.asarray(list(tensor.int32_data), dtype=np.uint16).view(np.float16)
    elif dtype == 16:  # BFLOAT16
        # BF16 is packed into int32_data as the raw u16 bit pattern.
        arr = np.asarray(list(tensor.int32_data), dtype=np.uint32).astype(np.uint16)
    else:
        raise SystemExit(
            f"{LOG_PREFIX} initializer {tensor.name!r} has no raw_data and dtype "
            f"{dtype} is not a supported fallback (F32=1 / F16=10 / BF16=16 only)"
        )
    return arr.tobytes()


def _extract_initializers(onnx_path: Path, prefix: str):
    """Walk one ONNX file's initializer list and return an ordered dict
    of ``{prefix + name: (dtype_str, shape, bytes)}``.

    * ``prefix`` — the bundle tag (``"p808."`` / ``"p835."``); every
      initializer name is prefixed so the Rust binder can route each
      tensor to the right sub-model without a graph load.
    * Dtype policy: only F32 / F16 / BF16 are accepted; any other dtype
      raises SystemExit (FR-EX-08 posture — never a silent skip).
    """
    try:
        import onnx  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover — dev env sanity check
        raise SystemExit(
            f"{LOG_PREFIX} `onnx` package not installed. Run "
            f"`uv add onnx` in tools/parity/ first."
        ) from exc

    _log(f"reading {onnx_path}")
    model = onnx.load(str(onnx_path), load_external_data=False)
    initializers = list(model.graph.initializer)
    if not initializers:
        raise SystemExit(
            f"{LOG_PREFIX} {onnx_path} carries no initializers — the ONNX file "
            f"is empty or references external weights (not supported)"
        )

    out = {}
    seen_names = set()
    for t in initializers:
        if t.data_type not in ONNX_DTYPE_TO_ST:
            raise SystemExit(
                f"{LOG_PREFIX} initializer {t.name!r} in {onnx_path.name} has "
                f"dtype {t.data_type} which is not F32 / F16 / BF16 — refusing "
                f"to silently skip (FR-EX-08)"
            )
        dtype_str = ONNX_DTYPE_TO_ST[t.data_type]
        shape = list(t.dims)
        if not shape:
            raise SystemExit(
                f"{LOG_PREFIX} initializer {t.name!r} in {onnx_path.name} has "
                f"an empty shape — refusing to emit a scalar weight (FR-EX-08)"
            )
        data = _initializer_bytes(t)
        # Sanity: byte count must match the shape × elem-size.
        elem_size = {"F32": 4, "F16": 2, "BF16": 2}[dtype_str]
        expected = elem_size
        for d in shape:
            expected *= int(d)
        if len(data) != expected:
            raise SystemExit(
                f"{LOG_PREFIX} initializer {t.name!r} in {onnx_path.name}: "
                f"payload is {len(data)} bytes but shape {shape} × {dtype_str} "
                f"({elem_size} B/elem) expects {expected} bytes"
            )
        prefixed = f"{prefix}{t.name}"
        if prefixed in seen_names:
            raise SystemExit(
                f"{LOG_PREFIX} duplicate initializer name after prefixing: "
                f"{prefixed!r} — refusing to emit an ambiguous bundle"
            )
        seen_names.add(prefixed)
        out[prefixed] = (dtype_str, shape, data)
    _log(f"  extracted {len(out)} initializers (prefix={prefix!r})")
    return out


def _write_safetensors(path: Path, tensors) -> None:
    """Minimal safetensors writer (stdlib only): 8-byte LE header length
    + JSON header + contiguous little-endian tensor data.

    Mirrors the writer in ``dfn3_prepare_checkpoint.py`` — kept inline so
    the prep script has zero non-``onnx`` runtime dependencies (no
    ``safetensors`` Python package required at execute time).
    """
    header = {}
    blobs = []
    offset = 0
    for name, (dtype_str, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype_str,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for b in blobs:
            f.write(b)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Flatten Microsoft DNSMOS's two ONNX checkpoints into a single "
            "merged safetensors bundle for vokra-convert."
        ),
    )
    parser.add_argument(
        "--p808",
        type=Path,
        default=None,
        help="Path to model_v8.onnx (P.808 predictor). Optional — "
        "omitting it produces a P.835-only partial bundle.",
    )
    parser.add_argument(
        "--p835",
        type=Path,
        default=None,
        help="Path to sig_bak_ovr.onnx (P.835 predictor). Optional — "
        "omitting it produces a P.808-only partial bundle.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output safetensors path (merged bundle).",
    )
    args = parser.parse_args()

    if args.p808 is None and args.p835 is None:
        _log("ERROR: at least one of --p808 / --p835 is required")
        return 2

    tensors = {}
    if args.p808 is not None:
        if not args.p808.exists():
            _log(f"ERROR: --p808 not found: {args.p808}")
            return 2
        tensors.update(_extract_initializers(args.p808, prefix="p808."))
    if args.p835 is not None:
        if not args.p835.exists():
            _log(f"ERROR: --p835 not found: {args.p835}")
            return 2
        tensors.update(_extract_initializers(args.p835, prefix="p835."))

    if not tensors:
        _log("ERROR: no initializers were extracted (empty ONNX inputs?)")
        return 2

    _log(f"writing {len(tensors)} tensors to {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_safetensors(args.output, tensors)
    sha = _sha256(args.output)
    _log(f"done: {args.output} sha256={sha}")
    # Manifest line for CI logs / fixture pipelines (matches the
    # dfn3_prepare_checkpoint.py format).
    print(f"{args.output.name} {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
