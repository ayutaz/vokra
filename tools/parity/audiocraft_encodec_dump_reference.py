#!/usr/bin/env python3
"""Dump an independent MusicGen-embedded EnCodec decode reference.

The oracle imports Hugging Face Transformers and loads the exact public
facebook/musicgen-small revision whose complete tensor manifest is pinned by
the Vokra MusicGen binder. It calls the official EncodecModel quantizer and
decoder directly; no Rust model math is mirrored here.

This is a VAST-only large-model command. Run through the repository's Python
3.12 policy, for example:

    uv run --no-project --python 3.12 \
      --with 'torch>=2.4,<3' \
      --with 'transformers==4.45.2' \
      --with 'accelerate>=0.34,<2' \
      python tools/parity/audiocraft_encodec_dump_reference.py \
      --output /workspace/audiocraft-encodec-reference.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path


MODEL_ID = "facebook/musicgen-small"
MODEL_REVISION = "257fc170552e35a0db0ffaf7759c14ab18dff9a4"
NUM_CODEBOOKS = 4
CODEBOOK_SIZE = 2048
FRAME_HOP = 640


def deterministic_codes(frames: int) -> list[int]:
    return [
        (frame * 257 + codebook * 503 + 17) % CODEBOOK_SIZE
        for frame in range(frames)
        for codebook in range(NUM_CODEBOOKS)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 1:
        raise ValueError("--frames must be >= 1")
    if args.output.exists():
        raise ValueError(f"refusing to overwrite existing output: {args.output}")

    import torch
    from transformers import MusicgenForConditionalGeneration

    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = MusicgenForConditionalGeneration.from_pretrained(
        args.model_id,
        revision=args.revision,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    codec = model.audio_encoder
    if codec.config.sampling_rate != 32_000:
        raise ValueError(f"unexpected sample rate {codec.config.sampling_rate}")
    if list(codec.config.upsampling_ratios) != [8, 5, 4, 4]:
        raise ValueError(
            f"unexpected ratios {list(codec.config.upsampling_ratios)}"
        )

    frame_major = deterministic_codes(args.frames)
    codes = (
        torch.tensor(frame_major, dtype=torch.long)
        .reshape(args.frames, NUM_CODEBOOKS)
        .transpose(0, 1)
        .contiguous()
        .reshape(1, 1, NUM_CODEBOOKS, args.frames)
    )
    with torch.inference_mode():
        quantized = codec.quantizer.decode(codes[0].transpose(0, 1))
        direct = codec.decoder(quantized)
        public = codec.decode(codes, [None]).audio_values
    if not torch.equal(direct, public):
        max_abs = (direct - public).abs().max().item()
        raise ValueError(f"official direct/public decode mismatch: {max_abs}")

    latent = quantized.detach().cpu().reshape(-1).tolist()
    pcm = public.detach().cpu().reshape(-1).tolist()
    expected_samples = args.frames * FRAME_HOP
    if len(pcm) != expected_samples:
        raise ValueError(f"PCM length {len(pcm)} != {expected_samples}")

    lines = [
        f"source,{args.model_id},{args.revision},transformers-4.45.2",
        f"shape,{args.frames},{NUM_CODEBOOKS},{FRAME_HOP}",
        "codes," + ",".join(str(value) for value in frame_major),
        "latent," + ",".join(f"{value:.9g}" for value in latent),
        "pcm," + ",".join(f"{value:.9g}" for value in pcm),
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
