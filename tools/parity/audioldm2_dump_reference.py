#!/usr/bin/env -S uv run --script
"""Strict official-only AudioLDM2 reference gate.

The real reference execution remains intentionally unavailable until the
dedicated frozen environment is generated and audited.  This entry point
fails before importing download/inference packages or touching model data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

LOCK = Path(__file__).with_name("audioldm2_reference") / "uv.lock"
SOURCE_REPOSITORY = "https://github.com/huggingface/diffusers.git"
SOURCE_COMMIT = "29f15673ed5c14e4843d7c837890910207f72129"
SOURCE_VERSION = "0.21.0"

# These are the exact broad declarations from the fixed source setup.py and
# the direct imports in its AudioLDM2 pipeline/model modules. They document
# what a future lock must cover; they are not a version selection. The source
# does not provide a Python-3.12 torch/Transformers pairing, so selecting one
# here would create false parity evidence.
SOURCE_RUNTIME_IMPORTS = (
    "numpy",
    "torch",
    "transformers",
    "huggingface-hub",
    "Pillow",
    "filelock",
    "requests",
    "regex",
    "safetensors",
    "tqdm",
)
OPTIONAL_ROUTE_IMPORTS = (
    "accelerate>=0.17.0 (only enable_model_cpu_offload)",
    "librosa (automatic audio scoring)",
    "scipy (official example output writer)",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        assert LOCK.name == "uv.lock"
        assert SOURCE_REPOSITORY.endswith("diffusers.git")
        assert SOURCE_COMMIT == "29f15673ed5c14e4843d7c837890910207f72129"
        assert SOURCE_VERSION == "0.21.0"
        assert "torch" in SOURCE_RUNTIME_IMPORTS
        assert "transformers" in SOURCE_RUNTIME_IMPORTS
        assert any(item.startswith("librosa") for item in OPTIONAL_ROUTE_IMPORTS)
        print("audioldm2_dump_reference --self-test: OK")
        return 0
    if not LOCK.is_file():
        print("audioldm2 reference BLOCKED: dedicated uv.lock is absent; no download or inference", file=sys.stderr)
        return 2
    print("audioldm2 reference BLOCKED: official inference is not authorized in this work item", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
