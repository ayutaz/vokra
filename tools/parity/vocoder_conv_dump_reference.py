#!/usr/bin/env -S uv run --script
"""Dump independent PyTorch Conv1d/ConvTranspose1d parity fixtures.

This is deliberately a reference-only tool: it calls ``torch.nn.functional``
directly and never imports Vokra.  The fixture values are deterministic signed
powers of two; no RNG, checkpoint, or model artifact is involved.  The output
directory is intentionally required so that running the script cannot create
fixture bytes accidentally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


def signed_powers(count: int, offset: int) -> list[float]:
    """Return deterministic signed powers of two (with no random source)."""

    return [
        float((-1 if index & 1 else 1) * (2 ** ((offset + index) % 6)))
        for index in range(count)
    ]


def f32_bytes(tensor: object) -> bytes:
    values = tensor.detach().cpu().contiguous().flatten().tolist()
    # ``struct.pack`` fixes the on-disk representation independently of host
    # endianness and rounds each value to IEEE-754 binary32.
    return b"".join(struct.pack("<f", value) for value in values)


def write_tensor(root: Path, name: str, tensor: object) -> dict[str, object]:
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

    # Conv1d: [batch, channels, time] and [out, in, kernel].  The effective
    # kernel is 5, while the padded input extent is 9, exercising both edges.
    input_1 = torch.tensor(signed_powers(10, 0), dtype=torch.float32).reshape(1, 2, 5)
    weight_1 = torch.tensor(signed_powers(18, 1), dtype=torch.float32).reshape(3, 2, 3)
    bias_1 = torch.tensor(signed_powers(3, 2), dtype=torch.float32)
    output_1 = functional.conv1d(
        input_1,
        weight_1,
        bias_1,
        stride=2,
        padding=2,
        dilation=2,
    )

    # ConvTranspose1d uses PyTorch's [in, out, kernel] weight layout.  The
    # output-padding value is deliberately non-zero and remains below stride.
    input_2 = torch.tensor(signed_powers(8, 3), dtype=torch.float32).reshape(1, 2, 4)
    weight_2 = torch.tensor(signed_powers(24, 4), dtype=torch.float32).reshape(2, 3, 4)
    bias_2 = torch.tensor(signed_powers(3, 5), dtype=torch.float32)
    output_2 = functional.conv_transpose1d(
        input_2,
        weight_2,
        bias_2,
        stride=3,
        padding=1,
        output_padding=2,
    )

    cases = [
        (
            "conv1d_d2_s2_p2",
            "torch.nn.functional.conv1d",
            {
                "in_channels": 2,
                "out_channels": 3,
                "kernel": 3,
                "stride": 2,
                "dilation": 2,
                "padding": 2,
                "output_padding": 0,
            },
            {"input": input_1, "weight": weight_1, "bias": bias_1, "output": output_1},
        ),
        (
            "conv_transpose1d_s3_p1_op2",
            "torch.nn.functional.conv_transpose1d",
            {
                "in_channels": 2,
                "out_channels": 3,
                "kernel": 4,
                "stride": 3,
                "dilation": 1,
                "padding": 1,
                "output_padding": 2,
            },
            {"input": input_2, "weight": weight_2, "bias": bias_2, "output": output_2},
        ),
    ]
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="directory receiving f32 tensors and manifest.json (run on VAST)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing an existing fixture directory",
    )
    args = parser.parse_args()

    output: Path = args.output
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(f"refusing non-empty output directory: {output} (use --overwrite)")
    output.mkdir(parents=True, exist_ok=True)

    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest_cases: dict[str, object] = {}
    for case_name, oracle, attrs, tensors in build_cases(torch):
        records = {
            role: write_tensor(output, f"{case_name}_{role}.f32", tensor)
            for role, tensor in tensors.items()
        }
        manifest_cases[case_name] = {
            "oracle": oracle,
            "attrs": attrs,
            "tensors": records,
        }

    manifest = {
        "schema": "vokra-vocoder-conv-parity-v1",
        "provenance": {
            "oracle": "PyTorch torch.nn.functional",
            "torch_version": torch.__version__,
            "generator": "tools/parity/vocoder_conv_dump_reference.py",
            "randomness": "none",
            "value_policy": "signed powers of two",
            "dtype": "float32",
            "byte_order": "little-endian",
        },
        "cases": manifest_cases,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {output / 'manifest.json'}")


if __name__ == "__main__":
    main()
