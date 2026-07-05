#!/usr/bin/env python3
"""Dump PyTorch (transformers) Whisper-base reference tensors for M0-06 parity.

This is an **offline** tool (FR-LD-05: no Python/PyTorch is ever pulled into the
runtime). It regenerates the fixtures under ``tests/parity/whisper_base/`` that
the Rust parity tests compare against at FP32 ``atol = 0.01`` (NFR-QL-01).

The reference is ``transformers`` ``WhisperForConditionalGeneration`` /
``WhisperProcessor`` for ``openai/whisper-base`` — the same Hugging Face
checkpoint the Vokra GGUF is converted from, so the weights are identical and
parity is meaningful.

What is dumped (kept minimal to avoid repo bloat):

* ``input_pcm.f32``           – deterministic synthetic mono PCM (seeded);
* ``logmel.f32``              – HF log-mel, first ``MEL_FRAMES`` frames ``[80, F]``;
* ``encoder.f32``             – encoder ``last_hidden_state`` first ``ENC_POS`` rows;
* ``logits_last.f32``         – decoder logits at the last prefix position ``[vocab]``;
* ``tokenizer.bin``           – id → bytes vocab (see ``whisper/tokenizer.rs``);
* ``manifest.txt``            – shapes / ids / greedy tokens / atol, ``key = value``;
* ``samples.txt``             – detokenizer ``ids -> text`` cases.

Run (from the repo root)::

    tools/parity/parity-venv/bin/python tools/parity/dump_whisper_reference.py \
        tests/parity/whisper_base

The checkpoint is fetched with ``WhisperForConditionalGeneration.from_pretrained``
(cached by the Hugging Face hub). Only ``torch`` / ``transformers`` / ``numpy``
are required.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

MODEL = "openai/whisper-base"
SEED = 1234
PCM_LEN = 16000  # 1 s at 16 kHz
MEL_FRAMES = 100  # frames of log-mel to dump for parity
ENC_POS = 32  # encoder positions to dump for parity
MAX_NEW = 32  # greedy generation cap
# English-transcribe decode prefix (verified against
# WhisperProcessor.get_decoder_prompt_ids): sot / en / transcribe / notimestamps.
PREFIX = [50258, 50259, 50359, 50363]
EOT = 50257
# Number of model-independent text tokens (ids 0..TEXT_VOCAB_LEN) in the Whisper
# multilingual byte-level BPE vocab. Equals EOT and the special-token floor; the
# first TEXT_VOCAB_LEN records are byte-identical across base..large-v3, so they
# are also written to the vokra-convert resource that the GGUF converter embeds.
TEXT_VOCAB_LEN = 50257
# Where the converter loads the bundled text-vocab resource from (repo-relative).
VOCAB_RESOURCE = (
    Path(__file__).resolve().parents[2]
    / "crates"
    / "vokra-convert"
    / "resources"
    / "whisper_multilingual_text_vocab.bin"
)


def synth_pcm() -> np.ndarray:
    """Deterministic sine + linear chirp + light noise, amplitude < 1."""
    rng = np.random.default_rng(SEED)
    t = np.arange(PCM_LEN, dtype=np.float64) / 16000.0
    tone = 0.3 * np.sin(2 * np.pi * 220.0 * t)
    # Linear chirp 300 -> 2000 Hz.
    f0, f1 = 300.0, 2000.0
    inst = f0 * t + 0.5 * (f1 - f0) / t[-1] * t * t
    chirp = 0.2 * np.sin(2 * np.pi * inst)
    noise = 0.05 * rng.standard_normal(PCM_LEN)
    return (tone + chirp + noise).astype(np.float32)


def write_f32(path: Path, arr) -> None:
    a = np.asarray(arr, dtype="<f4").reshape(-1)
    path.write_bytes(a.tobytes())


def dump_tokenizer(path: Path, tok: WhisperTokenizer, vocab_resource: Path | None = None) -> int:
    """Writes the id -> bytes vocab blob (see whisper/tokenizer.rs).

    When ``vocab_resource`` is given, the first ``TEXT_VOCAB_LEN`` records
    (header-less) are ALSO written there — the model-independent text vocabulary
    the GGUF converter embeds via ``include_bytes!`` (see
    ``vokra-convert/src/models/whisper.rs``). Those records are byte-identical
    across base..large-v3, so the resource can be regenerated from any
    multilingual checkpoint.
    """
    special_ids = set(tok.added_tokens_decoder.keys())
    n = len(tok)
    byte_decoder = tok.byte_decoder
    records = bytearray()
    records += struct.pack("<I", n)
    text_vocab = bytearray()  # first TEXT_VOCAB_LEN records, no count header
    for i in range(n):
        rec = bytearray()
        if i in special_ids:
            rec += struct.pack("<BH", 1, 0)
        else:
            s = tok.convert_ids_to_tokens(i)
            try:
                raw = bytes(byte_decoder[c] for c in s)
            except KeyError:
                # Any token whose chars fall outside the byte alphabet is treated
                # as special (contributes no text) rather than guessed.
                rec += struct.pack("<BH", 1, 0)
            else:
                rec += struct.pack("<BH", 0, len(raw))
                rec += raw
        records += rec
        if i < TEXT_VOCAB_LEN:
            text_vocab += rec
    path.write_bytes(records)

    if vocab_resource is not None:
        # The text floor must be exactly the first special id (== EOT); otherwise
        # the +1-shift / embedding invariants in the converter would be wrong.
        first_special = min(special_ids)
        assert first_special == TEXT_VOCAB_LEN, (
            f"text-vocab floor {TEXT_VOCAB_LEN} != first special id {first_special}"
        )
        vocab_resource.parent.mkdir(parents=True, exist_ok=True)
        vocab_resource.write_bytes(text_vocab)

    return n


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/parity/whisper_base")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    processor = WhisperProcessor.from_pretrained(MODEL)
    tok = WhisperTokenizer.from_pretrained(MODEL)  # slow tokenizer (has byte_decoder)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL)
    model.eval()

    pcm = synth_pcm()
    write_f32(out_dir / "input_pcm.f32", pcm)

    with torch.no_grad():
        feats = processor.feature_extractor(
            pcm, sampling_rate=16000, return_tensors="pt"
        ).input_features  # [1, 80, 3000]
        n_mels = feats.shape[1]
        logmel = feats[0].numpy()  # [80, 3000]
        write_f32(out_dir / "logmel.f32", logmel[:, :MEL_FRAMES])

        enc = model.model.encoder(feats).last_hidden_state  # [1, 1500, 512]
        write_f32(out_dir / "encoder.f32", enc[0, :ENC_POS, :].numpy())

        # Decoder logits over the forced prefix; dump the last-position vector
        # and every position's argmax.
        prefix_t = torch.tensor([PREFIX], dtype=torch.long)
        logits = model(input_features=feats, decoder_input_ids=prefix_t).logits  # [1, P, vocab]
        vocab = logits.shape[-1]
        write_f32(out_dir / "logits_last.f32", logits[0, -1, :].numpy())
        prefix_argmax = logits[0].argmax(dim=-1).tolist()

        # Plain greedy (argmax, no logit suppression) — matches the Rust loop.
        tokens = list(PREFIX)
        greedy = []
        for _ in range(MAX_NEW):
            lg = model(input_features=feats, decoder_input_ids=torch.tensor([tokens])).logits
            nxt = int(lg[0, -1, :].argmax().item())
            greedy.append(nxt)
            if nxt == EOT:
                break
            tokens.append(nxt)

    vocab_n = dump_tokenizer(out_dir / "tokenizer.bin", tok, vocab_resource=VOCAB_RESOURCE)

    # Detokenizer reference samples (ids -> text), including a multibyte case.
    samples = []
    greedy_text = tok.decode(greedy, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    samples.append((greedy, greedy_text))
    for text in ["Hello world", "Vokra 音声 test", "café déjà vu"]:
        ids = tok.encode(text, add_special_tokens=False)
        dec = tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        samples.append((ids, dec))

    with (out_dir / "samples.txt").open("w", encoding="utf-8") as f:
        f.write("# Detokenizer parity: one case per line, `ids | text`.\n")
        f.write("# ids are space-separated; text is the raw HF decode.\n")
        for ids, text in samples:
            f.write(" ".join(str(i) for i in ids))
            f.write(" | ")
            f.write(text.replace("\n", "\\n"))
            f.write("\n")

    manifest = {
        "model": MODEL,
        "seed": SEED,
        "atol": 0.01,
        "pcm_len": PCM_LEN,
        "n_mels": int(n_mels),
        "mel_frames": MEL_FRAMES,
        "enc_pos": ENC_POS,
        "d_model": int(enc.shape[-1]),
        "n_audio_ctx": int(enc.shape[1]),
        "vocab": int(vocab),
        "vocab_tokenizer": int(vocab_n),
        "prefix": PREFIX,
        "eot": EOT,
        "prefix_argmax": prefix_argmax,
        "greedy_tokens": greedy,
        "greedy_text": greedy_text,
    }
    with (out_dir / "manifest.txt").open("w", encoding="utf-8") as f:
        f.write("# Whisper base parity manifest (M0-06). Generated by\n")
        f.write("# tools/parity/dump_whisper_reference.py. `key = value`;\n")
        f.write("# list values are space-separated.\n")
        for k, v in manifest.items():
            if isinstance(v, list):
                f.write(f"{k} = {' '.join(str(x) for x in v)}\n")
            else:
                f.write(f"{k} = {v}\n")

    print(f"wrote fixtures to {out_dir}")
    print(f"  vocab={manifest['vocab']} greedy={len(greedy)} tokens: {greedy}")
    print(f"  greedy_text={greedy_text!r}")
    print(f"  prefix_argmax={prefix_argmax}")


if __name__ == "__main__":
    main()
