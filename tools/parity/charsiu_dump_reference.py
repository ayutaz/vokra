#!/usr/bin/env python3
"""Dump an independent Charsiu frame-classification reference.

This script intentionally defines no neural-network layer.  It loads the
canonical ``charsiu/en_w2v2_fc_10ms`` PyTorch state dict into Transformers'
official :class:`Wav2Vec2ForCTC` implementation.  Charsiu's upstream
``Wav2Vec2ForFrameClassification`` uses the same ``wav2vec2`` backbone and
``lm_head`` inference graph; only its training/alignment wrapper differs.

The pinned 400-sample input is the shortest waveform that produces one output
frame through the released seven-layer convolutional frontend.  Keeping the
fixture this small makes the full twelve-layer Rust parity leg practical while
still covering every real checkpoint tensor.

Usage::

    uv run --project tools/parity python tools/parity/charsiu_dump_reference.py \
      --checkpoint-bin /path/to/pytorch_model.bin \
      --config /path/to/config.json \
      --outdir crates/vokra-models/tests/fixtures/charsiu
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REVISION = "e9bf8dd314313fc57f6e4d0b5425bde4bbeac80f"
CHECKPOINT_SHA256 = "6dc8a18422db7c22e951d5f72dc2afc267b942eb0b8459ac6dcc0cf412536de1"
CONFIG_SHA256 = "7406aa4f917267640865688aa62f2337664a3abb9a49a2f204d932b53aeb6cb7"


def die(message: str) -> "NoReturn":
    print(f"charsiu_dump_reference: {message}", file=sys.stderr)
    raise SystemExit(2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        die(f"{label} does not exist: {path}")
    actual = sha256(path)
    if actual != expected:
        die(f"{label} sha256 {actual} != pinned {expected}")
    return actual


def deterministic_pcm() -> np.ndarray:
    # Compute in float64, then pin the exact little-endian float32 bytes.
    i = np.arange(400, dtype=np.float64)
    pcm = (
        0.19 * np.sin(2.0 * np.pi * 173.0 * i / 16_000.0)
        + 0.07 * np.cos(2.0 * np.pi * 811.0 * i / 16_000.0)
        + 0.02 * (i / 399.0 - 0.5)
    )
    return np.asarray(pcm, dtype="<f4")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-bin", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()

    checkpoint_hash = verify_file(
        args.checkpoint_bin, CHECKPOINT_SHA256, "canonical checkpoint"
    )
    config_hash = verify_file(args.config, CONFIG_SHA256, "canonical config")

    import torch
    import transformers
    from transformers import Wav2Vec2Config, Wav2Vec2ForCTC

    with args.config.open("r", encoding="utf-8") as handle:
        raw_config = json.load(handle)
    expected_axes = {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "vocab_size": 42,
        "pad_token_id": 41,
        "num_conv_pos_embeddings": 128,
        "num_conv_pos_embedding_groups": 16,
    }
    for key, expected in expected_axes.items():
        if raw_config.get(key) != expected:
            die(f"config {key}={raw_config.get(key)!r} != pinned {expected!r}")
    if raw_config.get("conv_stride") != [5, 2, 2, 2, 2, 2, 1]:
        die("config conv_stride is not the canonical 10 ms Charsiu stride")
    if raw_config.get("do_stable_layer_norm") is not False:
        die("config must select the released post-norm encoder")

    config = Wav2Vec2Config.from_json_file(str(args.config))
    model = Wav2Vec2ForCTC(config)
    try:
        state = torch.load(args.checkpoint_bin, map_location="cpu", weights_only=True)
    except TypeError:
        # The uv lock currently resolves a modern torch; this branch keeps the
        # offline script usable with older audited torch wheels as well.
        state = torch.load(args.checkpoint_bin, map_location="cpu")
    if not isinstance(state, dict):
        die(f"checkpoint root is {type(state).__name__}, expected a state-dict mapping")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        die(f"state-dict mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()

    pcm = deterministic_pcm()
    with torch.inference_mode():
        logits = model(torch.from_numpy(pcm.copy()).unsqueeze(0)).logits
    logits_np = np.asarray(logits.squeeze(0).cpu(), dtype="<f4")
    if logits_np.shape != (1, 42):
        die(f"canonical 400-sample input produced shape {logits_np.shape}, expected (1, 42)")
    if not np.isfinite(logits_np).all():
        die("reference logits contain NaN or infinity")

    args.outdir.mkdir(parents=True, exist_ok=True)
    pcm_path = args.outdir / "pcm_400.f32.bin"
    logits_path = args.outdir / "logits_1x42.f32.bin"
    pcm.tofile(pcm_path)
    logits_np.tofile(logits_path)
    manifest = {
        "schema": "vokra-charsiu-parity-v1",
        "upstream": "charsiu/en_w2v2_fc_10ms",
        "revision": REVISION,
        "checkpoint_sha256": checkpoint_hash,
        "config_sha256": config_hash,
        "reference_implementation": "transformers.Wav2Vec2ForCTC",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "sample_rate": 16_000,
        "pcm_file": pcm_path.name,
        "pcm_shape": [400],
        "pcm_sha256": sha256(pcm_path),
        "logits_file": logits_path.name,
        "logits_shape": list(logits_np.shape),
        "logits_sha256": sha256(logits_path),
    }
    with (args.outdir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
