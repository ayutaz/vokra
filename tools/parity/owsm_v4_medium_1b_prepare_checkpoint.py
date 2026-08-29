#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Fail-closed OWSM checkpoint preparation gate.

OWSM v4 medium 1B is a multi-file ESPnet/PyTorch release. This tool no
longer copies a first safetensors file or loads pickle locally; the complete
checkpoint must be inspected by the pinned VAST worker first.
"""
from __future__ import annotations

import argparse
import sys

HF_REPOSITORY = "espnet/owsm_v4_medium_1B"
HF_REVISION = "e10985c8f1d592e905c24d2ac2b2c53e3feb24dc"
INSPECTION_STATUS = "INSPECTION_ONLY"


def main() -> int:
    parser = argparse.ArgumentParser(description="Refuse unauthenticated OWSM preparation")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input-dir")
    parser.add_argument("--input")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.input_dir, args.input, args.output)):
            parser.error("--self-test accepts no input or output")
        assert len(HF_REVISION) == 40
        print("owsm_v4_medium_1b_prepare_checkpoint self-test: OK")
        return 0
    print(
        f"{INSPECTION_STATUS}: refusing arbitrary OWSM input; run the fixed "
        f"VAST inspector for {HF_REPOSITORY}@{HF_REVISION} first. No output was written.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
