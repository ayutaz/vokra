#!/usr/bin/env python3
"""Dump an independent official Parakeet-TDT decoder/head reference.

The full PCM encoder is intentionally outside this Wave-3 fixture.  This calls
the released Transformers ``encoder_projector``, ``decoder`` (embedding +
two-layer LSTM + projector), and combined token/duration ``joint`` modules
directly with real weights.  Rust consumes the same pre-projector encoder row
through ``ParakeetAsr::tdt_head_step``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import ParakeetForTDT

UPSTREAM_REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-hidden-out", type=Path, required=True)
    parser.add_argument("--reference-out", type=Path, required=True)
    parser.add_argument("--metadata-out", type=Path, required=True)
    parser.add_argument("--token-id", type=int, default=8192)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = ParakeetForTDT.from_pretrained(
        args.checkpoint,
        local_files_only=True,
        torch_dtype=torch.float32,
    ).eval()
    if not 0 <= args.token_id < model.config.vocab_size:
        parser.error(f"--token-id must be in 0..{model.config.vocab_size}")

    width = model.config.encoder_config.hidden_size
    grid = torch.arange(width, dtype=torch.float32)
    encoder_hidden = (0.2 * torch.sin(grid * 0.013) + 0.1 * torch.cos(grid * 0.029)).reshape(1, 1, width)
    token = torch.tensor([[args.token_id]], dtype=torch.long)
    with torch.inference_mode():
        encoder_projected = model.encoder_projector(encoder_hidden)
        decoder_projected = model.decoder(token)
        logits = model.joint(
            decoder_hidden_states=decoder_projected,
            encoder_hidden_states=encoder_projected,
        )[0, 0].contiguous()

    args.encoder_hidden_out.parent.mkdir(parents=True, exist_ok=True)
    encoder_hidden[0, 0].numpy().astype("<f4", copy=False).tofile(args.encoder_hidden_out)
    logits.numpy().astype("<f4", copy=False).tofile(args.reference_out)
    metadata = {
        "upstream_revision": UPSTREAM_REVISION,
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
        "token_id": args.token_id,
        "encoder_width": width,
        "joint_width": logits.numel(),
    }
    args.metadata_out.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
