#!/usr/bin/env python3
"""Prepare the exact sharded aiola/whisper-medusa-v1 release for Vokra.

Run with the repository parity environment (Python 3.12):

    uv run --project tools/parity python \
      tools/parity/whisper_medusa_prepare_checkpoint.py \
      --source-dir /root/whisper-medusa-official \
      --output /root/whisper-medusa-v1-merged.safetensors

The official 6.25 GB artifact must be handled on VAST.  This script never
imports model code; it verifies the pinned Hub files, walks the authoritative
index, and writes one safetensors file for the zero-dependency Rust converter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from safetensors.torch import load_file, save_file

REVISION = "6ea7c2f47658cfc7f9c8d1c158a9fbdb33458462"
SOURCE_REVISION = "19819c37ab15db6e68826e406614a2c86fbb946e"
EXPECTED = {
    "config.json": "16346762b14c116eeda12b48f20e2281b327a11b516f8b004ce065fcb1450186",
    "model.safetensors.index.json": "0b80666c06d5054aa425a07d9f2f4ecabf9e6d7b8333f0dc5d85d4f79c9ff449",
    "model-00001-of-00002.safetensors": "b09e03326f4a9e3cd9bac17a55e17c60a3463e720a1cf0a51b8ba246a2b70b67",
    "model-00002-of-00002.safetensors": "6c496a29e2d131f999bbec815e4bd7a38b2ca436ce0d902237fdbd2971b35b74",
}
EXPECTED_TENSORS = 1_281
EXPECTED_MEDUSA_TENSORS = 22


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(source_dir: Path, output: Path) -> None:
    for name, expected in EXPECTED.items():
        path = source_dir / name
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(f"{name}: SHA-256 {actual}, expected {expected}")

    index = json.loads((source_dir / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    if len(weight_map) != EXPECTED_TENSORS:
        raise SystemExit(
            f"index has {len(weight_map)} tensors, expected {EXPECTED_TENSORS}"
        )
    medusa = sorted(name for name in weight_map if name.startswith("medusa_heads."))
    if len(medusa) != EXPECTED_MEDUSA_TENSORS:
        raise SystemExit(
            f"index has {len(medusa)} Medusa tensors, expected {EXPECTED_MEDUSA_TENSORS}"
        )

    merged = {}
    for shard_name in sorted(set(weight_map.values())):
        for name, tensor in load_file(source_dir / shard_name, device="cpu").items():
            if name in merged:
                raise SystemExit(f"duplicate tensor across shards: {name}")
            if weight_map.get(name) != shard_name:
                raise SystemExit(
                    f"index maps {name!r} to {weight_map.get(name)!r}, found in {shard_name!r}"
                )
            merged[name] = tensor

    missing = sorted(set(weight_map) - set(merged))
    extra = sorted(set(merged) - set(weight_map))
    if missing or extra:
        raise SystemExit(f"shard/index mismatch: missing={missing[:3]} extra={extra[:3]}")

    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        merged,
        output,
        metadata={
            "format": "pt",
            "vokra.source_hf": "aiola/whisper-medusa-v1",
            "vokra.source_revision": REVISION,
            "vokra.source_code_revision": SOURCE_REVISION,
        },
    )
    digest = sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8"
    )
    print(f"wrote {output} ({len(merged)} tensors, sha256={digest})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source_dir, args.output)


if __name__ == "__main__":
    main()
