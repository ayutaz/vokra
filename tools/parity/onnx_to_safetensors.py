#!/usr/bin/env python3
"""ONNX → safetensors bridge (coverage-audit wave-a 2026-08-04).

Offline sidecar tool (FR-LD-05: no Python / PyTorch / ONNX ever enters
the runtime). Extracts every FLOAT / FLOAT16 / BFLOAT16 initializer
from an ONNX model and writes a flat safetensors file the Vokra
converter can consume via its normal `from_safetensors` path.

INT graph-metadata tensors (shape / axes / steps / small integer
constants) are intentionally dropped — they carry ONNX-runtime-only
lowering information that has no runtime meaning in the Vokra graph
and would otherwise fail the safetensors dtype gate downstream. A
sidecar `.stripped-manifest.json` records what was dropped for audit.

Usage:
    uv run python onnx_to_safetensors.py \\
        --input <in.onnx> \\
        --output <out.safetensors> \\
        [--int-threshold 8]

Args:
    --input          ONNX file path
    --output         safetensors output path
    --int-threshold  drop INT tensors with n_elem <= this (default 8,
                     covers Reshape/Slice/Transpose graph-metadata)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--int-threshold", type=int, default=8)
    args = ap.parse_args()

    import numpy as np
    import onnx
    from safetensors.numpy import save_file

    model = onnx.load(str(args.input))

    kept: dict[str, np.ndarray] = {}
    dropped: list[dict] = []

    for init in model.graph.initializer:
        arr = onnx.numpy_helper.to_array(init)
        dtype_name = onnx.TensorProto.DataType.Name(init.data_type)
        # Keep only float weight tensors — the Vokra safetensors reader
        # accepts F32 / F16 / BF16 only, and INT graph-metadata is
        # ONNX-runtime-only lowering hint with no runtime meaning here.
        if arr.dtype.kind == "f":
            # Rename any duplicated names in a deterministic way (ONNX
            # sometimes emits the same weight under two aliases — we
            # keep the first encountered, drop the rest with a note).
            if init.name in kept:
                dropped.append(
                    {"name": init.name, "reason": "duplicate name", "dtype": dtype_name}
                )
                continue
            kept[init.name] = np.ascontiguousarray(arr)
        else:
            # INT tensors: drop if small (graph-metadata) OR always,
            # since Vokra's safetensors reader refuses INT dtypes.
            n_elem = int(np.prod(arr.shape)) if arr.size else 0
            if n_elem <= args.int_threshold:
                dropped.append(
                    {
                        "name": init.name,
                        "reason": f"int graph-metadata (n_elem={n_elem})",
                        "dtype": dtype_name,
                        "shape": list(arr.shape),
                    }
                )
            else:
                dropped.append(
                    {
                        "name": init.name,
                        "reason": f"int large (n_elem={n_elem}) — refuse to fabricate float cast",
                        "dtype": dtype_name,
                        "shape": list(arr.shape),
                    }
                )

    if not kept:
        print(
            f"onnx_to_safetensors: refuse — no float initializers found in {args.input}",
            file=sys.stderr,
        )
        return 1

    save_file(kept, str(args.output))

    manifest_path = args.output.with_suffix(args.output.suffix + ".stripped-manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "source": str(args.input),
                "output": str(args.output),
                "kept": len(kept),
                "dropped": len(dropped),
                "dropped_details": dropped,
            },
            indent=2,
        )
    )

    print(
        f"onnx_to_safetensors: kept {len(kept)} float, dropped {len(dropped)} int; "
        f"manifest -> {manifest_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
