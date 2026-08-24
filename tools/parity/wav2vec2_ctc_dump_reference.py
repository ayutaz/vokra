#!/usr/bin/env python3
"""Dump an independent Hugging Face Wav2Vec2 CTC reference fixture.

The oracle is the upstream ``transformers.Wav2Vec2ForCTC`` implementation;
this script does not mirror the Rust layers.  It hooks the official model at
the encoder output and CTC logits, then stores raw little-endian arrays for an
offline Rust parity test.  Missing Transformers/model assets are fatal: there
is deliberately no local reimplementation fallback.
"""

from __future__ import annotations

import argparse
import json
import platform
import wave
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoProcessor, Wav2Vec2ForCTC


DEFAULT_MODEL = "facebook/wav2vec2-base-960h"
DEFAULT_REVISION = "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"


def read_pcm16_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise SystemExit(f"{path}: expected mono PCM16 WAV")
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return samples, sample_rate


def write_f32(path: Path, values: np.ndarray) -> None:
    path.write_bytes(np.asarray(values, dtype="<f4").tobytes(order="C"))


def write_u32(path: Path, values: np.ndarray) -> None:
    path.write_bytes(np.asarray(values, dtype="<u4").tobytes(order="C"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()

    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)

    pcm, sample_rate = read_pcm16_mono(args.audio)
    if sample_rate != 16_000:
        raise SystemExit(f"{args.audio}: expected 16000 Hz, got {sample_rate}")

    processor = AutoProcessor.from_pretrained(args.model_id, revision=args.revision)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_id, revision=args.revision)
    model.eval()
    inputs = processor(pcm, sampling_rate=sample_rate, return_tensors="pt")
    input_values = inputs.input_values
    encoder = model.wav2vec2(input_values).last_hidden_state
    logits = model.lm_head(model.dropout(encoder))
    token_ids = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(token_ids)[0]

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_f32(output / "pcm.f32.bin", pcm)
    write_f32(output / "encoder.f32.bin", encoder[0].cpu().numpy())
    write_f32(output / "logits.f32.bin", logits[0].cpu().numpy())
    # Store the post-CTC-fold ids, not per-frame argmax ids.
    folded: list[int] = []
    previous: int | None = None
    for token in token_ids[0].tolist():
        if token != previous and token != model.config.pad_token_id:
            folded.append(token)
        previous = token
    write_u32(output / "tokens.u32.bin", np.asarray(folded, dtype=np.uint32))
    (output / "text.txt").write_text(text + "\n", encoding="utf-8")

    manifest = {
        "format": "vokra-wav2vec2-ctc-reference-v1",
        "model_id": args.model_id,
        "revision": args.revision,
        "audio": str(args.audio),
        "sample_rate": sample_rate,
        "pcm_samples": int(pcm.size),
        "frames": int(logits.shape[1]),
        "hidden_size": int(encoder.shape[2]),
        "vocab_size": int(logits.shape[2]),
        "pad_blank_id": int(model.config.pad_token_id),
        "text": text,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cpu": platform.processor(),
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
