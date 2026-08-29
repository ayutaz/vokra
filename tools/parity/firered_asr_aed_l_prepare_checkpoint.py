#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Refuse unsafe FireRedASR-AED-L pickle preparation.

The upstream checkpoint is untrusted and the native AED contract is still
inspection-only. No pickle is loaded and no safetensors output is produced.
Use ``firered_asr_aed_l_inspect.py`` on VAST to collect evidence instead.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(
        "FireRedASR-AED-L preparation is INSPECTION_ONLY: unsafe pickle-to-safetensors conversion is disabled; no output was produced",
        file=__import__("sys").stderr,
    )
    return 2


def self_test() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    unsafe_loader = "torch" + ".load("
    assert unsafe_loader not in source
    assert "INSPECTION_ONLY" in source
    with tempfile.TemporaryDirectory(prefix="firered-preparer-") as directory:
        root = Path(directory)
        ckpt = root / "input.pth.tar"
        output = root / "output.safetensors"
        ckpt.write_bytes(b"untrusted")
        result = subprocess.run([__import__("sys").executable, str(Path(__file__)), "--ckpt", str(ckpt), "--output", str(output)], capture_output=True, text=True)
        assert result.returncode == 2 and not output.exists()
    print("firered preparer self-test PASS")


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--self-test"]:
        self_test()
    else:
        raise SystemExit(main())
