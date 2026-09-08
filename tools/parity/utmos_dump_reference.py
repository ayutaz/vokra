#!/usr/bin/env python3
"""UTMOS22 reference dump stub with an explicit unsafe-pickle boundary.

The historical UTMOS Lightning checkpoint contains pickled training objects,
and the upstream Lightning loader does not provide an explicit safe
state-dict loading API. This command therefore refuses the reference path
before importing PyTorch, fetching upstream sources, reading a checkpoint, or
creating output. An owner-approved safe state-dict wiring is required before
reference generation can resume.

The original command-line options remain accepted so callers receive a stable
fail-closed result while the safe loader design is pending.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def die(message: str) -> "NoReturn":
    print(f"utmos_dump_reference: {message}", file=sys.stderr)
    raise SystemExit(2)


def self_test() -> None:
    """Check the blocked boundary without importing torch or reading a checkpoint."""
    source = Path(__file__).read_text(encoding="utf-8")
    if "BLOCKED_UNSAFE_PICKLE" not in source:
        raise AssertionError("explicit unsafe-pickle blocker marker is missing")
    for forbidden in (
        "import " + "torch",
        "torch." + "load",
        "load_from_" + "checkpoint",
        "urllib." + "request",
    ):
        if forbidden in source:
            raise AssertionError(f"unsafe or network execution path found: {forbidden}")
    print("utmos_dump_reference self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", help="UTMOS Lightning checkpoint")
    parser.add_argument("--w2v", help="wav2vec_small.pt (architecture source)")
    parser.add_argument("--clip", help="mono 16 kHz WAV")
    parser.add_argument("--outdir")
    parser.add_argument("--repeats", type=int, default=3, help="re-runs for the band")
    parser.add_argument("--record-hashes", action="store_true")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Check the blocked deserialization contract without reading a checkpoint.",
    )
    args = parser.parse_args()

    if args.self_test:
        forbidden_options = (
            "--ckpt",
            "--w2v",
            "--clip",
            "--outdir",
            "--repeats",
            "--record-hashes",
        )
        if any(
            arg in forbidden_options
            or any(arg.startswith(f"{option}=") for option in forbidden_options)
            for arg in sys.argv[1:]
        ):
            parser.error("--self-test accepts no other arguments")
        self_test()
        return 0

    if not args.ckpt or not args.w2v or not args.clip or not args.outdir:
        parser.error("--ckpt, --w2v, --clip, and --outdir are required")
    die(
        "BLOCKED_UNSAFE_PICKLE: the published Lightning checkpoint cannot be "
        "processed without an owner-approved explicit safe state-dict path; "
        "no checkpoint, source, or output was read or created"
    )


if __name__ == "__main__":
    raise SystemExit(main())
