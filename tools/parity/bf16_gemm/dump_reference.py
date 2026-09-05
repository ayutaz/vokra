#!/usr/bin/env -S uv run --script
"""Generate the independent PyTorch BF16 GEMM parity packet.

This file is intentionally a reference-only generator.  It imports PyTorch
and evaluates the real ``torch.matmul`` operator; it does not import Vokra or
reimplement a GEMM kernel.  Run it on the authorized VAST host only.  The
maintainer Mac must not import or execute Torch for this suite.

The contract is the production ``gemm_bf16_on`` input path:

    torch.matmul(
        a.to(torch.bfloat16).to(torch.float32),
        b.to(torch.bfloat16).to(torch.float32),
    )

The row-major B matrix is ``[k, n]``, so the direct matmul result is ``[m, n]``
float32.  Values are deterministic and constructed without a PRNG.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any


GENERATOR = "tools/parity/bf16_gemm/dump_reference.py"
SCHEMA = "vokra-bf16-gemm-parity-v1"
TORCH_VERSION_SERIES = "2.13.0"
ATOL = 1e-3
RTOL = 0.0
ORACLE = (
    "torch.matmul(a.to(torch.bfloat16).to(torch.float32), "
    "b.to(torch.bfloat16).to(torch.float32))"
)
CASES: tuple[tuple[str, tuple[int, int, int], int], ...] = (
    ("full_k32_m8_n64", (8, 64, 96), 11),
    ("tails_m3_n35_k65", (3, 35, 65), 23),
    ("tails_m9_n33_k31", (9, 33, 31), 37),
)


def deterministic_values(count: int, offset: int) -> list[float]:
    """Return finite, non-random f32 inputs with nontrivial BF16 rounding."""

    values: list[float] = []
    for index in range(count):
        # The odd numerator keeps low mantissa bits live; the small periodic
        # term avoids a mostly-zero or power-of-two-only input distribution.
        numerator = ((index + offset) * 37) % 2003 - 1001
        value = numerator / 257.0 + ((index + 3 * offset) % 7) / 4096.0
        values.append(float(value))
    return values


def f32_tensor_bytes(tensor: Any) -> bytes:
    if str(tensor.dtype) != "torch.float32":
        raise RuntimeError(f"expected float32 tensor, got {tensor.dtype}")
    values = tensor.detach().cpu().contiguous().flatten().tolist()
    if not all(math.isfinite(float(value)) for value in values):
        raise RuntimeError("reference tensor contains a non-finite value")
    return struct.pack("<" + "f" * len(values), *values)


def write_tensor(root: Path, filename: str, tensor: Any) -> dict[str, object]:
    data = f32_tensor_bytes(tensor)
    (root / filename).write_bytes(data)
    return {
        "bytes": len(data),
        "dtype": "float32",
        "path": filename,
        "sha256": hashlib.sha256(data).hexdigest(),
        "shape": [int(value) for value in tensor.shape],
    }


def build_case(
    torch: Any, root: Path, name: str, shape: tuple[int, int, int], offset: int
) -> dict[str, object]:
    m, n, k = shape
    a = torch.tensor(deterministic_values(m * k, offset), dtype=torch.float32).reshape(m, k)
    b = torch.tensor(deterministic_values(k * n, offset + 101), dtype=torch.float32).reshape(k, n)
    # This is deliberately the independent PyTorch operation under test.  Do
    # not replace it with a Python loop or a handwritten BF16 emulation.
    a_bf16_f32 = a.to(torch.bfloat16).to(torch.float32)
    b_bf16_f32 = b.to(torch.bfloat16).to(torch.float32)
    output = torch.matmul(a_bf16_f32, b_bf16_f32)
    if str(output.dtype) != "torch.float32" or list(output.shape) != [m, n]:
        raise RuntimeError(f"{name}: unexpected torch.matmul result {output.dtype} {output.shape}")
    tensors = {
        "a": write_tensor(root=root, filename=f"{name}_a.f32", tensor=a),
        "b": write_tensor(root=root, filename=f"{name}_b.f32", tensor=b),
        "output": write_tensor(root=root, filename=f"{name}_output.f32", tensor=output),
    }
    return {"shape": {"m": m, "n": n, "k": k}, "tensors": tensors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    existing = sorted(path.name for path in output.iterdir())
    if existing != ["README.md"]:
        raise SystemExit(
            f"refusing non-empty/stale output directory: {output}; "
            "start with a directory containing only README.md"
        )

    import torch

    if torch.__version__.split("+", 1)[0] != TORCH_VERSION_SERIES:
        raise RuntimeError(
            f"expected Torch {TORCH_VERSION_SERIES} series, got {torch.__version__}"
        )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cases = {
        name: build_case(torch, output, name, shape, offset)
        for name, shape, offset in CASES
    }
    manifest = {
        "cases": cases,
        "comparison": {"atol": ATOL, "rtol": RTOL},
        "provenance": {
            "byte_order": "little-endian",
            "device": "cpu",
            "dtype": "float32",
            "generator": GENERATOR,
            "generator_identity": "deterministic torch.matmul BF16-widened oracle",
            "oracle": ORACLE,
            "randomness": "none",
            "torch_version": torch.__version__,
        },
        "schema": SCHEMA,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (output / "manifest.json").write_bytes(manifest_bytes)
    (output / "manifest.sha256").write_text(
        f"{hashlib.sha256(manifest_bytes).hexdigest()}  manifest.json\n", encoding="ascii"
    )
    print(f"wrote {output / 'manifest.json'}")


if __name__ == "__main__":
    main()
