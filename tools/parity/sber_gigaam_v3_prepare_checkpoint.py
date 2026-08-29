#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Fail-closed GigaAM v3 RNNT preparation gate.

The upstream checkpoint is an untrusted PyTorch/NeMo artifact. No local
conversion or pickle loading is permitted until the fixed VAST inspector has
authenticated the complete v3 tree, source, config, tokenizer, and manifest.
"""
from __future__ import annotations

import argparse
import sys

HF_REPOSITORY = "ai-sage/GigaAM-v3"
HF_REVISION = "ec1dc1f01d0d627ab2c0d3acc1e235702300d95e"
INSPECTION_STATUS = "INSPECTION_ONLY"


def main() -> int:
    parser = argparse.ArgumentParser(description="Refuse unauthenticated GigaAM v3 preparation")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.self_test:
        if args.input is not None or args.output is not None:
            parser.error("--self-test accepts no input or output")
        assert len(HF_REVISION) == 40
        print("sber_gigaam_v3_prepare_checkpoint self-test: OK")
        return 0
    print(
        f"{INSPECTION_STATUS}: refusing arbitrary GigaAM v3 input; run the fixed "
        f"VAST inspector for {HF_REPOSITORY}@{HF_REVISION}. No output was written.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
