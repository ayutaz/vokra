#!/usr/bin/env python3
"""Dump an independent raw-PCM Parakeet-TDT reference.

This fixture is produced only by the official Transformers feature extractor,
FastConformer encoder, TDT generation mixin, and tokenizer.  The deterministic
waveform is deliberately synthetic so the reference is redistributable while
still exercising every native frontend/encoder/decoder stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoProcessor, ParakeetForTDT

UPSTREAM_REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
SAMPLE_RATE = 16_000


def write_f32(path: Path, tensor: torch.Tensor) -> None:
    tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tofile(path)


def write_u32(path: Path, values: list[int]) -> None:
    np.asarray(values, dtype="<u4").tofile(path)


def write_wav_f32(path: Path, pcm: torch.Tensor) -> None:
    payload = pcm.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, SAMPLE_RATE, SAMPLE_RATE * 4, 4, 32)
    path.write_bytes(
        b"RIFF"
        + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(payload))
        + b"WAVEfmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(payload))
        + payload
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")

    torch.set_num_threads(1)
    torch.manual_seed(0)
    sample_count = round(SAMPLE_RATE * args.seconds)
    time = torch.arange(sample_count, dtype=torch.float32) / SAMPLE_RATE
    pcm = (
        0.12 * torch.sin(2.0 * torch.pi * 220.0 * time)
        + 0.04 * torch.sin(2.0 * torch.pi * 440.0 * time)
        + 0.01 * torch.cos(2.0 * torch.pi * 37.0 * time)
    )

    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    model = ParakeetForTDT.from_pretrained(
        args.checkpoint,
        local_files_only=True,
        torch_dtype=torch.float32,
    ).eval()
    inputs = processor(
        pcm.numpy(),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )

    with torch.inference_mode():
        subsampled = model.encoder.subsampling(
            inputs.input_features,
            inputs.attention_mask,
        )
        encoded = model.encoder(
            input_features=inputs.input_features,
            attention_mask=inputs.attention_mask,
            output_attention_mask=True,
        )
        generated = model.generate(**inputs)

    sequence = [int(value) for value in generated.sequences[0].tolist()]
    durations = [int(value) for value in generated.durations[0].tolist()]
    blank_id = int(model.config.blank_token_id)
    pad_id = int(model.config.pad_token_id)
    eos_id = int(model.generation_config.eos_token_id)
    emitted = [
        token
        for token in sequence[1:]
        if token not in (blank_id, pad_id, eos_id)
    ]
    text = processor.tokenizer.decode(
        emitted,
        skip_special_tokens=True,
        group_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    valid_feature_frames = int(inputs.attention_mask[0].sum().item())
    valid_encoder_frames = int(encoded.attention_mask[0].sum().item())
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_f32(output_dir / "pcm.f32", pcm)
    write_wav_f32(output_dir / "pcm.wav", pcm)
    write_f32(output_dir / "features.f32", inputs.input_features[0])
    write_f32(output_dir / "subsampled.f32", subsampled[0, :valid_encoder_frames])
    write_f32(output_dir / "encoder.f32", encoded.last_hidden_state[0, :valid_encoder_frames])
    write_u32(output_dir / "tokens.u32", emitted)

    tokenizer_path = args.checkpoint / "tokenizer.json"
    metadata = {
        "upstream_model": "nvidia/parakeet-tdt-0.6b-v3",
        "upstream_revision": UPSTREAM_REVISION,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "sample_rate": SAMPLE_RATE,
        "sample_count": sample_count,
        "input_feature_shape": list(inputs.input_features.shape),
        "valid_feature_frames": valid_feature_frames,
        "subsampled_shape": list(subsampled.shape),
        "encoder_shape": list(encoded.last_hidden_state.shape),
        "valid_encoder_frames": valid_encoder_frames,
        "blank_token_id": blank_id,
        "pad_token_id": pad_id,
        "eos_token_id": eos_id,
        "sequence": sequence,
        "durations": durations,
        "emitted_tokens": emitted,
        "text": text,
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
