#!/usr/bin/env python3
"""Prepare the pinned NVIDIA Parakeet-CTC-1.1B checkpoint for Vokra.

The official file contains 1,652 F32 inference tensors and 42 scalar I64
BatchNorm ``num_batches_tracked`` counters.  Eval BatchNorm never reads the
counters.  This script verifies every official source asset by SHA-256,
removes exactly those counters through the shared auditable helper and copies
the sidecars consumed by the strict converter.

Run with the repository's Python environment only:

    uv run --project tools/parity --python 3.12 python \
      tools/parity/parakeet_ctc_prepare_checkpoint.py --output-dir /path
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

UPSTREAM_HF = "nvidia/parakeet-ctc-1.1b"
UPSTREAM_REVISION = "20e63a0fed6aedba145b74b826dbd41df0941730"
SHA256 = {
    "model.safetensors": "57e0bc26772f3360b7ae0c087f184364179906674d08fc8b71d48a54d4f52145",
    "config.json": "c33a8ddbf447d68d31b2f1d1e4efa061548813b7647913e67560a9b198f06ae1",
    "preprocessor_config.json": "7f26808482a58d8dd187c4b87364810292b91ed7721e099bdbb05ca50da37a98",
    "tokenizer.json": "f3f1dd45c3889ed2b5bf67180caf05f51d7d7e4948c20e5f24d8c24df9cc47aa",
}
COUNTER = re.compile(r"^encoder\.layers\.(\d+)\.conv\.norm\.num_batches_tracked$")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Use an already downloaded official directory (VAST resume path)",
    )
    parser.add_argument(
        "--revision",
        default=UPSTREAM_REVISION,
        help="Override only after an explicit upstream re-audit",
    )
    args = parser.parse_args()
    if args.revision != UPSTREAM_REVISION:
        parser.error(
            f"revision must remain pinned to {UPSTREAM_REVISION}; update source and hashes together"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = args.source_dir or args.output_dir / "upstream"
    source_dir.mkdir(parents=True, exist_ok=True)
    for filename in SHA256:
        path = source_dir / filename
        if not path.is_file():
            path = Path(
                hf_hub_download(
                    repo_id=UPSTREAM_HF,
                    filename=filename,
                    revision=UPSTREAM_REVISION,
                    local_dir=source_dir,
                )
            )
        actual = digest(path)
        if actual != SHA256[filename]:
            sys.exit(
                f"{filename}: SHA-256 {actual} != pinned {SHA256[filename]}"
            )

    prepared = args.output_dir / "model.prepared.safetensors"
    helper = Path(__file__).resolve().with_name("strip_int_tensors.py")
    result = subprocess.run(
        [
            sys.executable,
            str(helper),
            "--input",
            str(source_dir / "model.safetensors"),
            "--output",
            str(prepared),
        ],
        check=False,
    )
    if result.returncode:
        return result.returncode

    manifest_path = prepared.with_suffix(prepared.suffix + ".stripped-manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    layers: list[int] = []
    for tensor in manifest["dropped_tensors"]:
        match = COUNTER.fullmatch(tensor["name"])
        if match is None or tensor["dtype"] != "torch.int64" or tensor["shape"] != []:
            sys.exit(f"unexpected stripped tensor: {tensor}")
        layers.append(int(match.group(1)))
    if (
        manifest["kept_count"] != 1_652
        or manifest["dropped_count"] != 42
        or sorted(layers) != list(range(42))
        or manifest["unknown_stripped"]
    ):
        sys.exit(
            "expected 1652 F32 inference tensors and exactly one scalar I64 "
            f"counter for layers 0..41; got {manifest}"
        )
    for filename in ["config.json", "preprocessor_config.json", "tokenizer.json"]:
        shutil.copyfile(source_dir / filename, args.output_dir / filename)

    audit = {
        "upstream_hf": UPSTREAM_HF,
        "upstream_revision": UPSTREAM_REVISION,
        "source_sha256": SHA256,
        "prepared_sha256": digest(prepared),
        "kept_count": 1_652,
        "dropped_count": 42,
        "dropped_layers": layers,
    }
    (args.output_dir / "prepare-audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
