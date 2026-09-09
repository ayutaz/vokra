#!/usr/bin/env -S uv run --script
"""Dump independent PyTorch Conv2d parity fixtures.

This is a reference-only generator: it imports PyTorch and calls
``torch.nn.functional.conv2d`` / ``conv_transpose2d`` directly.  It never
imports Vokra or reimplements convolution.  The output is raw little-endian
IEEE-754 binary32 plus a manifest containing per-file SHA-256 digests and an
outer ``manifest.sha256`` pin.

Run this on VAST/Scaleway after provisioning the pinned Python environment;
the maintainer machine must not execute this script because it imports and
executes Torch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


def signed_powers(count: int, offset: int) -> list[float]:
    """Return deterministic signed powers of two without a random source."""

    return [
        float((-1 if index & 1 else 1) * (2 ** ((offset + index) % 6)))
        for index in range(count)
    ]


def f32_bytes(tensor: object) -> bytes:
    values = tensor.detach().cpu().contiguous().flatten().tolist()
    # Explicit packing fixes both dtype and byte order independently of the
    # generator host.  Every source tensor is constructed as torch.float32.
    return struct.pack("<" + "f" * len(values), *values)


def write_tensor(root: Path, name: str, tensor: object) -> dict[str, object]:
    if str(tensor.dtype) != "torch.float32":
        raise RuntimeError(f"{name}: expected torch.float32, got {tensor.dtype}")
    data = f32_bytes(tensor)
    path = root / name
    path.write_bytes(data)
    return {
        "path": name,
        "shape": [int(value) for value in tensor.shape],
        "dtype": "float32",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def build_cases(torch: object) -> list[tuple[str, str, dict[str, object], dict[str, object]]]:
    import torch.nn.functional as functional

    # Grouped Conv2d: asymmetric stride/padding, non-unit dilation, and the
    # PyTorch [out, in/groups, kh, kw] weight layout.
    input_1 = torch.tensor(signed_powers(4 * 5 * 6, 0), dtype=torch.float32).reshape(
        1, 4, 5, 6
    )
    weight_1 = torch.tensor(signed_powers(6 * 2 * 3 * 2, 1), dtype=torch.float32).reshape(
        6, 2, 3, 2
    )
    bias_1 = torch.tensor(signed_powers(6, 2), dtype=torch.float32)
    attrs_1 = {
        "in_channels": 4,
        "out_channels": 6,
        "kernel": [3, 2],
        "stride": [2, 1],
        "padding": [1, 2],
        "dilation": [2, 1],
        "output_padding": [0, 0],
        "groups": 2,
    }
    output_1 = functional.conv2d(
        input_1,
        weight_1,
        bias_1,
        stride=tuple(attrs_1["stride"]),
        padding=tuple(attrs_1["padding"]),
        dilation=tuple(attrs_1["dilation"]),
        groups=attrs_1["groups"],
    )

    # Grouped ConvTranspose2d: asymmetric tuples, dilation, and non-zero
    # output padding with the PyTorch [in, out/groups, kh, kw] layout.
    input_2 = torch.tensor(signed_powers(4 * 3 * 4, 3), dtype=torch.float32).reshape(
        1, 4, 3, 4
    )
    weight_2 = torch.tensor(signed_powers(4 * 3 * 2 * 3, 4), dtype=torch.float32).reshape(
        4, 3, 2, 3
    )
    bias_2 = torch.tensor(signed_powers(6, 5), dtype=torch.float32)
    attrs_2 = {
        "in_channels": 4,
        "out_channels": 6,
        "kernel": [2, 3],
        "stride": [2, 3],
        "padding": [1, 2],
        "dilation": [2, 1],
        "output_padding": [1, 2],
        "groups": 2,
    }
    output_2 = functional.conv_transpose2d(
        input_2,
        weight_2,
        bias_2,
        stride=tuple(attrs_2["stride"]),
        padding=tuple(attrs_2["padding"]),
        output_padding=tuple(attrs_2["output_padding"]),
        dilation=tuple(attrs_2["dilation"]),
        groups=attrs_2["groups"],
    )

    # ATen's valid edge case: output_padding == stride on the first axis is
    # accepted because it remains smaller than dilation (1 < 2).
    input_3 = torch.tensor(signed_powers(1, 8), dtype=torch.float32).reshape(1, 1, 1, 1)
    weight_3 = torch.tensor(signed_powers(2, 9), dtype=torch.float32).reshape(1, 1, 2, 1)
    bias_3 = torch.tensor(signed_powers(1, 10), dtype=torch.float32)
    attrs_3 = {
        "in_channels": 1,
        "out_channels": 1,
        "kernel": [2, 1],
        "stride": [1, 2],
        "padding": [0, 0],
        "dilation": [2, 1],
        "output_padding": [1, 0],
        "groups": 1,
    }
    output_3 = functional.conv_transpose2d(
        input_3,
        weight_3,
        bias_3,
        stride=tuple(attrs_3["stride"]),
        padding=tuple(attrs_3["padding"]),
        output_padding=tuple(attrs_3["output_padding"]),
        dilation=tuple(attrs_3["dilation"]),
        groups=attrs_3["groups"],
    )

    return [
        (
            "conv2d_grouped_d2_s21_p12",
            "torch.nn.functional.conv2d",
            attrs_1,
            {"input": input_1, "weight": weight_1, "bias": bias_1, "output": output_1},
        ),
        (
            "conv_transpose2d_grouped_d21_s23_p12_op12",
            "torch.nn.functional.conv_transpose2d",
            attrs_2,
            {"input": input_2, "weight": weight_2, "bias": bias_2, "output": output_2},
        ),
        (
            "conv_transpose2d_op1_lt_dilation",
            "torch.nn.functional.conv_transpose2d",
            attrs_3,
            {"input": input_3, "weight": weight_3, "bias": bias_3, "output": output_3},
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="directory receiving raw f32 tensors and manifest files (run remotely)",
    )
    args = parser.parse_args()
    output: Path = args.output
    # README.md is committed alongside generated files. Permit that one
    # contract file, but reject every other pre-existing entry so regeneration
    # cannot silently leave stale tensors or manifests in the fixture set.
    if output.exists():
        existing = sorted(path.name for path in output.iterdir())
        if existing != ["README.md"]:
            raise SystemExit(
                f"refusing non-empty/stale output directory: {output}; "
                "use a new empty directory containing only README.md"
            )
    output.mkdir(parents=True, exist_ok=True)

    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cases: dict[str, object] = {}
    for case_name, oracle, attrs, tensors in build_cases(torch):
        records = {
            role: write_tensor(output, f"{case_name}_{role}.f32", tensor)
            for role, tensor in tensors.items()
        }
        cases[case_name] = {"oracle": oracle, "attrs": attrs, "tensors": records}

    manifest = {
        "schema": "vokra-conv2d-parity-v1",
        "provenance": {
            "oracle": "PyTorch torch.nn.functional",
            "torch_version": torch.__version__,
            "generator": "tools/parity/conv2d_dump_reference.py",
            "randomness": "none",
            "value_policy": "signed powers of two",
            "dtype": "float32",
            "byte_order": "little-endian",
        },
        "cases": cases,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest_path = output / "manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    (output / "manifest.sha256").write_text(f"{digest}  manifest.json\n", encoding="ascii")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
