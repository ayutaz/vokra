#!/usr/bin/env python3
"""Dump a tiny BigVGAN alias-free Activation1d reference fixture.

The oracle imports NVIDIA/BigVGAN's own ``activations.py`` and
``alias_free_activation.torch.act.Activation1d`` from a caller-supplied
checkout. It does not mirror the resampling equations locally.

Usage::

    uv run python tools/parity/bigvgan_alias_free_dump_reference.py \
        --upstream-dir /path/to/NVIDIA/BigVGAN \
        --output tools/parity/fixtures/bigvgan_alias_free.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch  # type: ignore[import-not-found]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    upstream = args.upstream_dir.resolve()
    if not (upstream / "activations.py").is_file():
        raise SystemExit(f"missing upstream activations.py under {upstream}")
    sys.path.insert(0, str(upstream))

    from activations import SnakeBeta  # type: ignore[import-not-found]
    from alias_free_activation.torch.act import (  # type: ignore[import-not-found]
        Activation1d,
    )

    torch.set_grad_enabled(False)
    torch.use_deterministic_algorithms(True)
    activation = SnakeBeta(2, alpha_logscale=True)
    activation.alpha.copy_(torch.tensor([-0.2, 0.35], dtype=torch.float32))
    activation.beta.copy_(torch.tensor([0.1, -0.3], dtype=torch.float32))
    wrapped = Activation1d(activation)
    values = torch.tensor(
        [
            [-1.25, -0.75, -0.25, 0.0, 0.25, 0.75, 1.25],
            [0.9, 0.45, 0.1, -0.2, -0.55, -0.85, -1.1],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)
    actual = wrapped(values).squeeze(0)

    up_filter = wrapped.upsample.filter.flatten()
    down_filter = wrapped.downsample.lowpass.filter.flatten()
    if not torch.equal(up_filter, down_filter):
        raise SystemExit("upstream fixture unexpectedly uses different up/down filters")

    lines = ["filter," + ",".join(f"{value:.9g}" for value in up_filter.tolist())]
    for channel in range(2):
        row = [
            "channel",
            str(channel),
            f"{activation.alpha[channel].item():.9g}",
            f"{activation.beta[channel].item():.9g}",
        ]
        row.extend(f"{value:.9g}" for value in values[0, channel].tolist())
        row.extend(f"{value:.9g}" for value in actual[channel].tolist())
        lines.append(",".join(row))
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
