#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Fail-closed GigaAM Multilingual CTC preparation gate.

The upstream checkpoint is an untrusted PyTorch artifact. No local conversion
or pickle loading is permitted until the fixed VAST inspector authenticates
the complete multilingual tree, source, config, and embedded vocabulary.
"""
from __future__ import annotations

import argparse
import sys

HF_REPOSITORY = "ai-sage/GigaAM-Multilingual"
HF_REVISION = "2f8a57144e6ec3adfd32fe0484d9ea9913305bc8"
INSPECTION_STATUS = "INSPECTION_ONLY"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refuse unauthenticated GigaAM Multilingual preparation"
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.self_test:
        if args.input is not None or args.output is not None:
            parser.error("--self-test accepts no input or output")
        assert len(HF_REVISION) == 40
        print("sber_gigaam_multilingual_prepare_checkpoint self-test: OK")
        return 0
    print(
        f"{INSPECTION_STATUS}: refusing arbitrary GigaAM Multilingual input; run the fixed "
        f"VAST inspector for {HF_REPOSITORY}@{HF_REVISION}. No output was written.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
