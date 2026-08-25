#!/usr/bin/env python3
"""Bridge Microsoft NSNet2's ONNX release to a safetensors checkpoint
(Coverage-audit 2026-08-03 Wave A, ticket
``docs/tickets/coverage-audit-2026-08-03/wave-a/nsnet2.md``).

**Offline sidecar tool (FR-LD-05: no Python / ONNX ever enters the runtime).**
The upstream release
``github.com/microsoft/DNS-Challenge/tree/8b87a33b2892f147b5c7ad39ea978453730db269/NSNet2-baseline`` ships
``nsnet2-20ms-baseline.onnx`` (10,752,263 bytes) — a 2-layer GRU + 3-Linear
noise-suppression baseline over the 161-bin STFT log-power of 16 kHz PCM. The
Rust converter (``crates/vokra-convert/src/models/nsnet2.rs``) consumes
safetensors only (zero-dep NFR-DS-02: no ONNX / protobuf in the runtime), so
this script performs the ONNX → safetensors flatten off-band with:

* ``onnx`` (Apache-2.0) to parse the model's ``graph.initializer`` list — every
  weight the runtime forward needs is stored there for NSNet2 (Constant-node
  subgraph walks used by e.g. Silero VAD are not necessary here — NSNet2's
  GRUs / Linear layers keep their weights in the initializer list directly).
* ``numpy`` (BSD-3) to interpret the raw tensor bytes (float32 for every
  NSNet2 initializer, but the writer supports fp16 defensively should a
  future half-precision distillation ship).
* ``safetensors`` (Apache-2.0) via the stdlib-only writer below, so this
  script can run without the ``safetensors`` package if a downstream user
  wants to keep the venv minimal.

Every ONNX initializer is emitted verbatim under its upstream name. The Rust
converter validates the exact 14-entry manifest, assigns semantic tensor
names, transposes MatMul axes and stamps the complete topology/provenance
metadata. No hparam side-car is written.

# Loud-error posture (FR-EX-08)

- Missing / malformed ONNX → propagate the ``onnx`` package's own exception
  (never a silent partial write).
- Unknown initializer dtype (not F32 / F16 / BF16) → hard error listing the
  tensor name and dtype — never silently drop it (the safetensors reader in
  Vokra's converter is float-only by design).
- External-data initializers (``data_location=EXTERNAL``) → hard error —
  NSNet2's release is self-contained; anything else is a corrupt download.
- Empty initializer list → hard error (a real NSNet2 has ~14 initializers).

# Usage (uv, per memory ``feedback-python-uses-uv``)

::

    # If `onnx` is not yet in the tools/parity venv, add it (Python 3.12):
    uv add --project tools/parity onnx

    uv run --project tools/parity python tools/parity/nsnet2_prepare_checkpoint.py \\
        --onnx ~/checkpoints/nsnet2/nsnet2-20ms-baseline.onnx \\
        --output ~/checkpoints/nsnet2/model.safetensors

Then:

::

    ./target/release/vokra-cli convert --model nsnet2 \\
        --input ~/checkpoints/nsnet2/model.safetensors \\
        --output ~/gguf/nsnet2.gguf

# NOT REFERENCED

- No AGPL / GPL / copyleft source is read or referenced. NSNet2's upstream
  ``LICENSE`` is standard MIT (``Copyright (c) Microsoft Corporation``, fetched
  2026-08-03 — CLAUDE.md「ハルシネーション厳禁」).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections import OrderedDict
from pathlib import Path

LOG_PREFIX = "nsnet2_prepare_checkpoint:"

# ONNX TensorProto.DataType numeric codes we accept as float weights (mirrors
# the ``crates/vokra-convert/src/onnx.rs`` allow-list). NSNet2 today ships F32
# only, but the writer is dtype-generic so a future half-precision variant
# lands without a code change.
ONNX_DTYPE_F32 = 1
ONNX_DTYPE_F16 = 10
ONNX_DTYPE_BF16 = 16
# ONNX integer graph-metadata dtypes. NSNet2's upstream ONNX bakes two INT64
# scalar constants (e.g. Reshape target dimensions, Slice axes, GRU seq-length
# anchors) into the ``graph.initializer`` list. They are **not** model weights
# — the compiled Rust forward folds the same integers into the topology and
# never reads them at runtime. The prep script's float-only writer skips them
# with a loud SKIP warning (mirror of `dnsmos_prepare_checkpoint.py`'s TF-export
# scalar-constant skip). Larger INT64 tensors (would be per-token index look-ups
# or quantized codebooks) still hard-fail per FR-EX-08 defensive posture.
ONNX_DTYPE_INT8 = 3
ONNX_DTYPE_INT16 = 5
ONNX_DTYPE_INT32 = 6
ONNX_DTYPE_INT64 = 7
ONNX_INTEGER_DTYPES = {ONNX_DTYPE_INT8, ONNX_DTYPE_INT16, ONNX_DTYPE_INT32, ONNX_DTYPE_INT64}
# Max element count for a graph-metadata integer initializer we will silently
# drop. NSNet2 today ships 2 x INT64[1] (single scalar each). Cap chosen small
# to keep any real weight-like integer tensor loud-failing.
INT_METADATA_MAX_ELEMENTS = 8

SAFETENSORS_DTYPE = {
    ONNX_DTYPE_F32: ("F32", 4),
    ONNX_DTYPE_F16: ("F16", 2),
    ONNX_DTYPE_BF16: ("BF16", 2),
}

# ONNX TensorProto.DataLocation (proto3 default = 0 = DEFAULT; 1 = EXTERNAL).
ONNX_LOCATION_DEFAULT = 0
ONNX_LOCATION_EXTERNAL = 1


def write_safetensors(path: Path, tensors: "OrderedDict[str, tuple[str, list[int], bytes]]") -> None:
    """Minimal safetensors writer (stdlib only): 8-byte LE header length +
    JSON header + contiguous little-endian tensor data. Mirrors the
    ``dfn3_prepare_checkpoint.py`` writer so the pattern is uniform across
    prep scripts."""
    header: "OrderedDict[str, dict]" = OrderedDict()
    offset = 0
    for name, (dtype_str, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype_str,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as fp:
        fp.write(struct.pack("<Q", len(header_bytes)))
        fp.write(header_bytes)
        for _name, (_dtype, _shape, data) in tensors.items():
            fp.write(data)


def _initializer_bytes(init) -> bytes:
    """Extracts the raw little-endian tensor payload from an ONNX
    initializer. ONNX may store tensor data either as ``raw_data`` (already
    little-endian bytes) or via one of the typed repeated fields
    (``float_data`` / ``double_data`` / ``int64_data`` / ``int32_data`` /
    …). NSNet2's F32 initializers use ``raw_data``, but the fallback for
    ``float_data`` keeps the writer robust to older exports.
    """
    if init.HasField("data_location") and init.data_location == ONNX_LOCATION_EXTERNAL:
        raise SystemExit(
            f"{LOG_PREFIX} initializer {init.name!r} uses external data "
            "(data_location=EXTERNAL); NSNet2's release must be self-contained"
        )
    if init.raw_data:
        return bytes(init.raw_data)
    # Fallback for typed repeated fields (rare in NSNet2's release but permitted
    # by the ONNX spec).
    if init.data_type == ONNX_DTYPE_F32 and init.float_data:
        import numpy as np  # deferred: only reach here when the fallback fires

        arr = np.array(init.float_data, dtype=np.float32)
        return arr.tobytes()
    raise SystemExit(
        f"{LOG_PREFIX} initializer {init.name!r} has neither raw_data nor a "
        "matching typed field; cannot recover tensor payload"
    )


def _initializer_shape(init) -> list[int]:
    """ONNX ``TensorProto.dims`` is a repeated int64; safetensors expects a
    plain list of positive ints. Returns them in insertion order."""
    return list(int(d) for d in init.dims)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--onnx",
        required=True,
        type=Path,
        help="path to nsnet2-20ms-baseline.onnx (the NSNet2 upstream ONNX file)",
    )
    ap.add_argument(
        "--output",
        required=True,
        type=Path,
        help="path to write the flattened safetensors checkpoint",
    )
    args = ap.parse_args()

    if not args.onnx.exists():
        raise SystemExit(f"{LOG_PREFIX} input ONNX not found: {args.onnx}")

    # ``onnx`` is a compile-time dep only for the offline prep script (not for
    # the Vokra runtime — FR-LD-05, NFR-DS-02). The import is deferred so an
    # ``--help`` invocation works even in a venv missing the package.
    try:
        import onnx  # Apache-2.0
    except ImportError as e:
        raise SystemExit(
            f"{LOG_PREFIX} the `onnx` package is required to parse the upstream ONNX. "
            f"Install it with `uv add --project tools/parity onnx` (per memory "
            f"`feedback-python-uses-uv`). Original error: {e}"
        )

    model = onnx.load(str(args.onnx))
    initializers = list(model.graph.initializer)
    if not initializers:
        raise SystemExit(
            f"{LOG_PREFIX} the input ONNX has zero initializers — a real NSNet2 "
            "release has ~14 (GRU / Linear weights + biases). Check the download."
        )

    tensors: "OrderedDict[str, tuple[str, list[int], bytes]]" = OrderedDict()
    n_params = 0
    dropped: list[str] = []
    for init in initializers:
        dtype = int(init.data_type)
        # NSNet2 ONNX bakes short INT64 scalars (Reshape / Slice / GRU seq_length
        # anchors) into ``graph.initializer``; these are graph metadata, not
        # weights, and the compiled Rust forward folds them into the topology.
        # Skip them with a loud SKIP note (FR-EX-08 loud-partial), keeping the
        # hard-fail for large / non-graph-metadata integer tensors that would
        # indicate an actual quantized codebook.
        if dtype in ONNX_INTEGER_DTYPES:
            shape_meta = _initializer_shape(init)
            n_elem = 1
            for d in shape_meta:
                n_elem *= max(d, 0)
            if n_elem <= INT_METADATA_MAX_ELEMENTS:
                onnx_name = onnx.TensorProto.DataType.Name(dtype)
                # Inline SKIP log is the audit trail; do **not** re-append to
                # ``dropped`` (which is dedicated to unnamed initializers) to
                # avoid the misleading "unnamed initializer:" print at the tail.
                print(
                    f"{LOG_PREFIX}   SKIP integer graph-metadata initializer "
                    f"{init.name!r} (dtype={onnx_name}, shape={shape_meta}, "
                    f"n_elem={n_elem}) — not a weight, folded into compiled forward"
                )
                continue
            raise SystemExit(
                f"{LOG_PREFIX} initializer {init.name!r} is integer "
                f"(dtype={onnx.TensorProto.DataType.Name(dtype)}, shape={shape_meta}, "
                f"n_elem={n_elem}) but exceeds the {INT_METADATA_MAX_ELEMENTS}-element "
                f"graph-metadata cap; refusing to silently drop what may be a real "
                f"quantized weight (FR-EX-08 loud-fail)"
            )
        if dtype not in SAFETENSORS_DTYPE:
            raise SystemExit(
                f"{LOG_PREFIX} initializer {init.name!r} has unsupported dtype "
                f"(ONNX code {dtype}); NSNet2 must be F32 / F16 / BF16 only"
            )
        dtype_str, element_size = SAFETENSORS_DTYPE[dtype]
        shape = _initializer_shape(init)
        data = _initializer_bytes(init)
        expected = element_size
        for d in shape:
            expected *= d
        if expected != len(data):
            raise SystemExit(
                f"{LOG_PREFIX} initializer {init.name!r} shape / payload mismatch: "
                f"shape={shape} × {element_size}B = {expected}, got {len(data)}"
            )
        if not init.name:
            dropped.append("<unnamed>")
            continue
        tensors[init.name] = (dtype_str, shape, data)
        n_params += expected // element_size

    if not tensors:
        raise SystemExit(
            f"{LOG_PREFIX} no named initializers survived filtering — refusing to "
            "write an empty safetensors"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_safetensors(args.output, tensors)
    for name in dropped:
        print(f"dropped (unnamed initializer): {name}")

    sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"{sha}  {args.output}")
    print(f"tensors={len(tensors)} params={n_params}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
