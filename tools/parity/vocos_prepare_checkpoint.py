#!/usr/bin/env python3
"""Prepare an official Vocos pickle checkpoint for the Rust converter.

The runtime never parses pickle.  This offline wrapper pins one of the two
official repositories/revisions and delegates safe ``weights_only=True``
deserialization plus safetensors emission to ``bin_to_safetensors.py``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

RELEASES = {
    "mel": (
        "charactr/vocos-mel-24khz",
        "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21",
    ),
    "encodec": (
        "charactr/vocos-encodec-24khz",
        "4e61d082c08045a4c11e5b148ad93b1d0c591a14",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(RELEASES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--revision",
        help="Override the audited commit (intended only for an explicit re-audit)",
    )
    parser.add_argument("--out-basename", default="model.safetensors")
    args = parser.parse_args()

    repo, audited_revision = RELEASES[args.variant]
    helper = Path(__file__).resolve().with_name("bin_to_safetensors.py")
    if not helper.is_file():
        parser.error(f"missing sibling helper: {helper}")
    command = [
        sys.executable,
        str(helper),
        "--hf-repo",
        repo,
        "--revision",
        args.revision or audited_revision,
        "--output-dir",
        str(args.output_dir),
        "--out-basename",
        args.out_basename,
    ]
    print("vocos_prepare_checkpoint: " + " ".join(command))
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
