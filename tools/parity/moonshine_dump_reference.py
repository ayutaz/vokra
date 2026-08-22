"""Dump an independent Moonshine Tiny encoder/decoder/logit reference.

Run only through the parity environment, for example:

    uv run --project tools/parity python tools/parity/moonshine_dump_reference.py \
      --output /workspace/moonshine-tiny-reference.json

The model and tokenizer are fetched from the pinned Hugging Face revision.  The
checkpoint and tokenizer hashes are checked before Transformers sees either
file, so the fixture cannot silently drift to a later upstream revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer

MODEL_ID = "moonshine-ai/moonshine-tiny"
REVISION = "390624ed33d594443aa4aa221f5b9f283b545b5a"
CHECKPOINT_SHA256 = "867cd2215804859c55aa972d740bd5002be149b4e7526328c895d2408848c736"
TOKENIZER_SHA256 = "6579793438bc4fbafffacf699169ff53e3769c5a0a0f5e71cdee8853e8130deb"
DECODER_IDS = [1, 42, 314]


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_pcm() -> np.ndarray:
    index = np.arange(16_000, dtype=np.float32)
    pcm = (
        0.17 * np.sin(2 * np.pi * 223.0 * index / 16_000.0)
        + 0.09 * np.sin(2 * np.pi * 701.0 * index / 16_000.0)
    ).astype(np.float32)
    pcm[::997] += np.float32(0.03)
    return pcm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = hf_hub_download(MODEL_ID, "model.safetensors", revision=REVISION)
    tokenizer = hf_hub_download(MODEL_ID, "tokenizer.json", revision=REVISION)
    actual_checkpoint = sha256(checkpoint)
    actual_tokenizer = sha256(tokenizer)
    if actual_checkpoint != CHECKPOINT_SHA256:
        raise RuntimeError(f"checkpoint SHA-256 drift: {actual_checkpoint}")
    if actual_tokenizer != TOKENIZER_SHA256:
        raise RuntimeError(f"tokenizer SHA-256 drift: {actual_tokenizer}")

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        torch_dtype=torch.float32,
    ).eval()
    tokenizer_impl = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    pcm = torch.from_numpy(deterministic_pcm()).unsqueeze(0)
    ids = torch.tensor([DECODER_IDS], dtype=torch.long)
    encoder = model.model.encoder(pcm).last_hidden_state
    decoder = model.model.decoder(
        input_ids=ids,
        encoder_hidden_states=encoder,
        use_cache=False,
    ).last_hidden_state
    logits = model.proj_out(decoder)
    generated = model.generate(pcm, max_new_tokens=32, do_sample=False)
    generated_ids = generated[0].tolist()

    payload = {
        "schema": "vokra.moonshine.parity.v1",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "checkpoint_sha256": actual_checkpoint,
        "tokenizer_sha256": actual_tokenizer,
        "decoder_ids": DECODER_IDS,
        "pcm": pcm[0].tolist(),
        "encoder_shape": list(encoder.shape[1:]),
        "encoder": encoder[0].contiguous().view(-1).tolist(),
        "decoder_shape": list(decoder.shape[1:]),
        "decoder": decoder[0].contiguous().view(-1).tolist(),
        "last_logits": logits[0, -1].contiguous().tolist(),
        "generated_ids": generated_ids,
        "generated_text": tokenizer_impl.decode(generated_ids, skip_special_tokens=True),
    }
    args.output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "encoder_shape": payload["encoder_shape"],
                "decoder_shape": payload["decoder_shape"],
                "checkpoint_sha256": actual_checkpoint,
                "tokenizer_sha256": actual_tokenizer,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
