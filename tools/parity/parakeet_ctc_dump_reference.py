#!/usr/bin/env python3
"""Dump an independent upstream Transformers Parakeet-CTC reference.

This script imports the pinned upstream package directly. It does not mirror
the model layers. Hooks expose the official feature extractor, subsampler,
encoder and CTC head for a deterministic redistributable three-tone waveform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import wave
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoProcessor, ParakeetForCTC

UPSTREAM_REVISION = "20e63a0fed6aedba145b74b826dbd41df0941730"
TRANSFORMERS_SOURCE_REVISION = "d56c55bf564ddb176759eb6ec199442682564916"
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


def read_pcm16_wav(path: Path) -> torch.Tensor:
    with wave.open(str(path), "rb") as stream:
        if (
            stream.getnchannels() != 1
            or stream.getframerate() != SAMPLE_RATE
            or stream.getsampwidth() != 2
            or stream.getcomptype() != "NONE"
        ):
            raise SystemExit(
                f"{path}: expected 16 kHz mono PCM16 WAV, got "
                f"channels={stream.getnchannels()} rate={stream.getframerate()} "
                f"width={stream.getsampwidth()} compression={stream.getcomptype()}"
            )
        payload = stream.readframes(stream.getnframes())
    pcm = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    return torch.from_numpy(pcm)


def collapse_ctc(sequence: list[int], blank: int) -> list[int]:
    output: list[int] = []
    previous = blank
    for token in sequence:
        if (token != previous or previous == blank) and token != blank:
            output.append(token)
        previous = token
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--audio",
        type=Path,
        help="Use a redistributable 16 kHz mono PCM16 WAV instead of the deterministic tone",
    )
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(0)
    if args.audio is not None:
        pcm = read_pcm16_wav(args.audio)
        input_kind = "redistributable-wav"
        input_sha256 = hashlib.sha256(args.audio.read_bytes()).hexdigest()
    else:
        time = torch.arange(SAMPLE_RATE, dtype=torch.float32) / SAMPLE_RATE
        pcm = (
            0.12 * torch.sin(2.0 * torch.pi * 220.0 * time)
            + 0.04 * torch.sin(2.0 * torch.pi * 440.0 * time)
            + 0.01 * torch.cos(2.0 * torch.pi * 37.0 * time)
        )
        input_kind = "deterministic-three-tone"
        input_sha256 = hashlib.sha256(
            pcm.numpy().astype("<f4", copy=False).tobytes()
        ).hexdigest()
    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    model = ParakeetForCTC.from_pretrained(
        args.checkpoint,
        local_files_only=True,
        dtype=torch.float32,
    ).eval()
    inputs = processor(pcm.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")

    with torch.inference_mode():
        subsampled_unscaled = model.encoder.subsampling(
            inputs.input_features, inputs.attention_mask
        )
        encoded = model.encoder(
            input_features=inputs.input_features,
            attention_mask=inputs.attention_mask,
            output_attention_mask=True,
        )
        logits = model.ctc_head(encoded.last_hidden_state.transpose(1, 2)).transpose(1, 2)
        raw_sequence = logits.argmax(dim=-1)[0].tolist()

    valid_features = int(inputs.attention_mask[0].sum().item())
    valid_encoder = int(encoded.attention_mask[0].sum().item())
    raw_sequence = [int(value) for value in raw_sequence[:valid_encoder]]
    blank = int(model.config.pad_token_id)
    tokens = collapse_ctc(raw_sequence, blank)
    official_text = processor.decode(
        raw_sequence,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_f32(output / "pcm.f32", pcm)
    write_wav_f32(output / "pcm.wav", pcm)
    write_f32(output / "features.f32", inputs.input_features[0, :valid_features])
    write_f32(output / "subsampled_unscaled.f32", subsampled_unscaled[0, :valid_encoder])
    write_f32(output / "encoder.f32", encoded.last_hidden_state[0, :valid_encoder])
    write_f32(output / "logits.f32", logits[0, :valid_encoder])
    write_u32(output / "raw_argmax.u32", raw_sequence)
    write_u32(output / "tokens.u32", tokens)
    (output / "text.txt").write_text(official_text + "\n", encoding="utf-8")

    tokenizer = args.checkpoint / "tokenizer.json"
    metadata = {
        "upstream_model": "nvidia/parakeet-ctc-1.1b",
        "upstream_revision": UPSTREAM_REVISION,
        "transformers_source_revision": TRANSFORMERS_SOURCE_REVISION,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "sample_rate": SAMPLE_RATE,
        "sample_count": len(pcm),
        "input_kind": input_kind,
        "input_sha256": input_sha256,
        "input_feature_shape": list(inputs.input_features.shape),
        "valid_feature_frames": valid_features,
        "subsampled_shape": list(subsampled_unscaled.shape),
        "encoder_shape": list(encoded.last_hidden_state.shape),
        "valid_encoder_frames": valid_encoder,
        "logits_shape": list(logits.shape),
        "blank_token_id": blank,
        "raw_argmax": raw_sequence,
        "tokens": tokens,
        "text": official_text,
        "tokenizer_sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
