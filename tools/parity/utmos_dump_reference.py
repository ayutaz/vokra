#!/usr/bin/env python3
"""UTMOS22 reference dump stub with an explicit unsafe-pickle boundary.

Current status: 2026-09-12.

The historical UTMOS Lightning checkpoint contains pickled training objects,
and the upstream Lightning loader does not provide an explicit safe
state-dict loading API. This command therefore refuses the reference path
before importing PyTorch, fetching upstream sources, reading a checkpoint, or
creating output. An owner-approved safe state-dict wiring is required before
reference generation can resume.

The dumper accepts a tensor-only ``.safetensors`` state-dict as a preparation
milestone, but it still refuses to run until the upstream model can be built
with a safe wav2vec state-dict as well. It never falls back to the upstream
Lightning checkpoint loader or any unrestricted pickle loader.
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


def validate_state_dict_source(path: str) -> None:
    """Read only a tensor-only source header; never deserialize Python objects."""
    source = Path(path)
    if source.suffix != ".safetensors":
        die(f"safe state-dict must use the `.safetensors` extension: {source.name}")
    try:
        from safetensors import safe_open
    except ImportError:
        die("`safetensors` is not installed; refusing reference execution")
    try:
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
    except Exception as error:  # noqa: BLE001 — malformed input is terminal
        die(f"safetensors state-dict could not be inspected safely ({type(error).__name__}: {error})")
    if not keys:
        die("safetensors state-dict has no tensor keys")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", help="legacy UTMOS Lightning checkpoint (refused)")
    parser.add_argument("--state-dict", help="tensor-only UTMOS state-dict (.safetensors)")
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
            "--state-dict",
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

    if not args.ckpt and not args.state_dict:
        parser.error("one of --state-dict or --ckpt is required")
    if not args.w2v or not args.clip or not args.outdir:
        parser.error("--w2v, --clip, and --outdir are required")
    if args.ckpt:
        die(
            "BLOCKED_UNSAFE_PICKLE: legacy Lightning .ckpt inputs are permanently "
            "refused; provide authenticated tensor-only state-dicts for every "
            "upstream input"
        )
    if args.state_dict:
        validate_state_dict_source(args.state_dict)
    die(
        "BLOCKED_SAFE_REFERENCE: the tensor-only UTMOS state-dict was inspected, "
        "but the independent upstream reference still requires a safe wav2vec "
        "state-dict and owner-approved model-construction path; no model or "
        "audio execution was performed"
    )


if __name__ == "__main__":
    raise SystemExit(main())
