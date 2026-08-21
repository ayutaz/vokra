#!/usr/bin/env python3
"""Dump independent PyTorch references for Chatterbox speaker projections.

Run only through the repository Python environment, for example:

    uv run --with torch --with gguf \
      tools/parity/chatterbox_speaker_projection_dump_reference.py \
      --gguf /path/to/chatterbox-nano-v1.gguf

The three GGUFs are verbatim float pass-through conversions of the official
T3 checkpoints. The projection mapping is the upstream
``cond_enc.spkr_enc`` ``torch.nn.Linear`` call; this script executes
``torch.nn.functional.linear`` rather than mirroring the Rust loop.

Pinned public artifacts used for the 2026-08-21 fixture:

* ``vokra/chatterbox-multilingual-v3`` revision
  ``95c8bf4409c237de930c2eec0274fb2b99a21a09``;
* ``vokra/chatterbox-nano-v1`` revision
  ``49b2f3612ec3e479eb64ce49ab27ae82cbf0b206``;
* ``vokra/chatterbox-turbo-v1`` revision
  ``10fee774c6c5ed890e39cea76d0ae1a320f7a4eb``.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as functional
from gguf import GGUFReader


def tensor(reader: GGUFReader, name: str) -> torch.Tensor:
    value = next((item for item in reader.tensors if item.name == name), None)
    if value is None:
        raise KeyError(f"missing tensor {name!r}")
    # gguf-py's data view reverses the display axes; flattening preserves the
    # pass-through byte order, then the raw GGUF dimensions restore PyTorch's
    # upstream [out_features, in_features] layout.
    array = value.data.copy().reshape(tuple(map(int, value.shape)))
    return torch.from_numpy(array)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--prefix", type=int, default=32)
    args = parser.parse_args()

    reader = GGUFReader(args.gguf)
    weight = tensor(reader, "cond_enc.spkr_enc.weight")
    bias = tensor(reader, "cond_enc.spkr_enc.bias").reshape(-1)
    ids = np.arange(weight.shape[1], dtype=np.float32)
    input_values = (
        np.sin((ids + np.float32(0.25)) * np.float32(0.071))
        * np.float32(0.2)
    )
    output = functional.linear(torch.from_numpy(input_values), weight, bias)
    prefix = output[: args.prefix]
    print(", ".join(format(float(value), ".9g") for value in prefix))
    print(
        f"count={output.numel()} min={float(output.min()):.9g} "
        f"max={float(output.max()):.9g} finite={bool(output.isfinite().all())}"
    )


if __name__ == "__main__":
    main()
