#!/usr/bin/env python3
"""Prepare the official SpeechBrain ECAPA checkpoint for Rust conversion.

Only the 31 integer BatchNorm ``num_batches_tracked`` training counters are
removed. All 200 floating-point inference tensors retain their upstream names,
shapes, dtypes, and values; the Rust converter validates the full manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


EXPECTED_FLOAT_TENSORS = 200
EXPECTED_TRAINING_COUNTERS = 31
UPSTREAM_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    state = torch.load(args.input, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise SystemExit(f"expected state-dict checkpoint, got {type(state).__name__}")

    counters = sorted(name for name in state if name.endswith(".num_batches_tracked"))
    tensors = {
        name: value.detach().cpu().contiguous()
        for name, value in state.items()
        if name not in counters
    }
    if len(counters) != EXPECTED_TRAINING_COUNTERS:
        raise SystemExit(
            f"expected {EXPECTED_TRAINING_COUNTERS} BatchNorm counters, got {len(counters)}"
        )
    if len(tensors) != EXPECTED_FLOAT_TENSORS:
        raise SystemExit(
            f"expected {EXPECTED_FLOAT_TENSORS} inference tensors, got {len(tensors)}"
        )
    non_float = [name for name, value in tensors.items() if not value.is_floating_point()]
    if non_float:
        raise SystemExit(f"non-floating inference tensors remain: {non_float}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        args.output,
        metadata={
            "source": "speechbrain/spkrec-ecapa-voxceleb",
            "revision": UPSTREAM_REVISION,
            "transform": "remove-batchnorm-num-batches-tracked-only",
        },
    )
    print(
        json.dumps(
            {
                "input_entries": len(state),
                "written_tensors": len(tensors),
                "removed_counters": counters,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
