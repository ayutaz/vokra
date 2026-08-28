#!/usr/bin/env python3
"""Prepare the pinned Microsoft SpeechT5 TTS release for Vokra.

This is a VAST-oriented wrapper around ``bin_to_safetensors.py``. It downloads
only the pinned revision, verifies the original torch-pickle and SentencePiece
content hashes before conversion, and then uses the shared ``weights_only``
pickle bridge to produce ``model.safetensors``. No model forward is run.
The released state dict also contains exactly five scalar integer
``speech_decoder_postnet.layers.{0..4}.batch_norm.num_batches_tracked``
training counters. They are removed by exact name; a missing, non-scalar or
non-integer counter aborts preparation instead of widening the skip rule.

Usage::

    uv run --project tools/parity --python 3.12 python \
        tools/parity/speecht5_tts_prepare_checkpoint.py \
        --output-dir /workspace/speecht5-tts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bin_to_safetensors


UPSTREAM_HF = "microsoft/speecht5_tts"
UPSTREAM_REVISION = "30fcde30f19b87502b8435427b5f5068e401d5f6"
SOURCE_WEIGHT_SHA256 = (
    "d60d28067349ef66b50d8cd643ae56b6d6b8f27def929bc4ef6fcad907954190"
)
TOKENIZER_MODEL_SHA256 = (
    "7fcc48f3e225f627b1641db410ceb0c8649bd2b0c982e150b03f8be3728ab560"
)
LOG_PREFIX = "speecht5_tts_prep:"
EXCLUDED_BATCH_COUNTERS = frozenset(
    f"speech_decoder_postnet.layers.{layer}.batch_norm.num_batches_tracked"
    for layer in range(5)
)


def require_sha256(path: Path, expected: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{LOG_PREFIX} required file is missing: {path}")
    actual = bin_to_safetensors.sha256_of(path)
    if actual != expected:
        raise SystemExit(
            f"{LOG_PREFIX} {path.name} sha256 {actual}, expected {expected}; "
            "refusing an unpinned source"
        )
    print(f"{LOG_PREFIX} verified sha256 {actual}  {path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--out-basename", default="model.safetensors")
    args = parser.parse_args()

    snapshot = bin_to_safetensors.download_checkpoint(
        UPSTREAM_HF, args.output_dir, UPSTREAM_REVISION
    )
    weight = snapshot / "pytorch_model.bin"
    tokenizer = snapshot / "spm_char.model"
    require_sha256(weight, SOURCE_WEIGHT_SHA256)
    require_sha256(tokenizer, TOKENIZER_MODEL_SHA256)

    output = snapshot / args.out_basename
    if output.exists():
        raise SystemExit(
            f"{LOG_PREFIX} refusing to overwrite existing output {output}"
        )
    return bin_to_safetensors.convert_local(
        weight, output, skip_tensor_names=EXCLUDED_BATCH_COUNTERS
    )


if __name__ == "__main__":
    sys.exit(main())
