#!/usr/bin/env python3
"""Dump PyTorch (transformers) Whisper reference tensors for M0-06 / M2-06 parity.

This is an **offline** tool (FR-LD-05: no Python/PyTorch is ever pulled into the
runtime). It regenerates the fixtures under ``tests/parity/whisper_{size}/``
that the Rust parity tests compare against at FP32 ``atol = 0.01`` (NFR-QL-01).

The reference is ``transformers`` ``WhisperForConditionalGeneration`` /
``WhisperProcessor`` for one of the fixed ``SUPPORTED_MODELS`` checkpoints — the
same Hugging Face checkpoints the Vokra GGUFs are converted from, so the weights
are identical and parity is meaningful.

Input audio is a **real 16 kHz mono WAV** (PCM16 or IEEE_FLOAT32) at
``tests/fixtures/audio/jfk-30s.wav`` by default, truncated / zero-padded to the
first 30 s (the fixed Whisper audio window, ``N_SAMPLES`` in
``crates/vokra-models/src/whisper/mel.rs``). Deterministic synthetic PCM was
used previously (M0-06) but produced ASR-meaningless "parody" transcripts — see
the M2-06 §3 follow-up notes — and was removed to prevent silent drift back to
synthetic input. FR-EX-08 (no silent fallback) is enforced: unsupported audio
(non-mono, wrong sample rate, exotic PCM format) is a hard error, not a
resample-behind-your-back.

What is dumped (kept minimal to avoid repo bloat):

* ``input_pcm.f32``           – first 30 s of the real input WAV as mono f32;
* ``logmel.f32``              – HF log-mel, first ``MEL_FRAMES`` frames ``[n_mels, F]``;
* ``encoder.f32``             – encoder ``last_hidden_state`` first ``ENC_POS`` rows;
* ``logits_last.f32``         – decoder logits at the last prefix position ``[vocab]``;
* ``tokenizer.bin``           – id → bytes vocab (see ``whisper/tokenizer.rs``);
* ``manifest.txt``            – shapes / ids / greedy tokens / atol / provenance,
                                ``key = value``; ``pcm_source`` and ``pcm_sha256``
                                pin which WAV produced this fixture set;
* ``samples.txt``             – detokenizer ``ids -> text`` cases.

Run (from the repo root)::

    tools/parity/parity-venv/bin/python tools/parity/dump_whisper_reference.py \
        --model whisper-base --audio tests/fixtures/audio/jfk-30s.wav

The checkpoint is fetched with ``WhisperForConditionalGeneration.from_pretrained``
(cached by the Hugging Face hub). Only ``torch`` / ``transformers`` / ``numpy``
are required at runtime; WAV parsing uses the stdlib ``struct`` module.
"""

from __future__ import annotations

import argparse
import hashlib
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

# Fixed allowlist of supported sizes; each maps to the HF checkpoint id. No
# silent fallback (FR-EX-08): unknown sizes are rejected by argparse `choices`.
SUPPORTED_MODELS = {
    "whisper-base": "openai/whisper-base",
    "whisper-small": "openai/whisper-small",
    "whisper-medium": "openai/whisper-medium",
    "whisper-large-v3": "openai/whisper-large-v3",
    "whisper-turbo": "openai/whisper-large-v3-turbo",
}
# Determinism knob for PyTorch tie-breaking during decoder greedy sampling. Not
# used for input synthesis — inputs are real audio (see load_pcm).
TORCH_SEED = 1234
# 30 s at 16 kHz — matches N_SAMPLES in crates/vokra-models/src/whisper/mel.rs
# (the fixed Whisper audio window). Real audio shorter than this is zero-padded
# on the right (Whisper does this internally anyway); longer audio is truncated
# to the first 30 s.
PCM_LEN = 30 * 16000
# Repo-relative path to the default input WAV. The file itself is intentionally
# NOT committed alongside this script — see M2-06 §3 honest note. Owner places
# a real 16 kHz mono WAV here before regenerating fixtures.
DEFAULT_AUDIO = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "audio"
    / "jfk-30s.wav"
)
MEL_FRAMES = 100  # frames of log-mel to dump for parity
ENC_POS = 32  # encoder positions to dump for parity
MAX_NEW = 32  # greedy generation cap


def resolve_special_tokens(tok: WhisperTokenizer) -> tuple[list[int], int]:
    """Look up the English-transcribe decode prefix + EOT via the tokenizer.

    The token TEXTS are the same across every multilingual Whisper size but
    the IDs shift when the vocab grows — e.g. large-v3 added `<|yue|>`
    (Cantonese) at 50358, pushing `<|transcribe|>` from 50359 to 50360 and
    `<|notimestamps|>` from 50363 to 50364 (the turbo tokenizer inherits the
    same layout). Hard-coding the base-vocab IDs and reusing them for
    large-v3 / turbo produces a manifest that fights the runtime, which is
    exactly the failure that `weight_load_and_config_smoke` surfaces (its
    assertion compares the manifest prefix against the model's own
    `decoder_start_ids`).

    Returns `(prefix, eot)` where `prefix = [sot, en, transcribe,
    notimestamps]` — the four-token English-transcribe decode prefix used
    for both greedy generation and the decoder-logits fixture.
    """
    # Round-trip validate the special-token names against the tokenizer.
    # `convert_tokens_to_ids` falls back to `unk_token_id` for unknown tokens,
    # and Whisper tokenizers register `<|endoftext|>` AS the unk (both id
    # 50257 on base/small/medium, both 50257 on large-v3/turbo despite the
    # vocab growing — the shift only touches task/language tokens). So the
    # ambiguity is real, and the safest check is: the id must round-trip
    # back through `convert_ids_to_tokens` to the same source string.
    names = (
        "<|startoftranscript|>",
        "<|en|>",
        "<|transcribe|>",
        "<|notimestamps|>",
        "<|endoftext|>",
    )
    ids = [tok.convert_tokens_to_ids(n) for n in names]
    for name, tid in zip(names, ids):
        if tid is None:
            sys.exit(
                f"tokenizer for this model has no {name} special token — "
                "cannot build the parity prefix (got None)"
            )
        rt = tok.convert_ids_to_tokens(tid)
        if rt != name:
            sys.exit(
                f"tokenizer round-trip failed for {name}: id={tid} -> {rt!r}"
            )
    prefix = ids[:4]
    eot = ids[4]
    return prefix, eot
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


def load_pcm(path: Path, n_samples: int) -> np.ndarray:
    """Load ``n_samples`` of mono 16 kHz PCM from a real WAV.

    Mirrors ``crates/vokra-cli/src/wav.rs::parse`` byte-for-byte:

    * PCM16 (``audio_format = 1``, ``bits = 16``) → scaled by ``1 / 32768``;
    * IEEE_FLOAT32 (``audio_format = 3``, ``bits = 32``) → passthrough;
    * everything else is an explicit error (FR-EX-08: no silent normalisation,
      no resample, no channel mixdown — fix your input WAV).

    Shorter clips are zero-padded on the right (Whisper does the same
    internally); longer clips are truncated to ``n_samples``.
    """
    if not path.is_file():
        sys.exit(
            f"audio file not found: {path}\n"
            "  --audio must point at a real 16 kHz mono WAV (PCM16 or "
            "IEEE_FLOAT32).\n"
            "  Recipe: ffmpeg -i input.flac -ac 1 -ar 16000 -c:a pcm_s16le "
            f"{path}\n"
            "  The canonical clip is the openai-whisper `tests/jfk.flac` "
            "excerpt resampled to 16 kHz mono 30 s WAV."
        )
    data = path.read_bytes()
    if len(data) < 12 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        sys.exit(f"{path}: not a RIFF/WAVE file")

    fmt = None
    payload = None
    pos = 12
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4]
        (size,) = struct.unpack_from("<I", data, pos + 4)
        body_start = pos + 8
        body_end = body_start + size
        if body_end > len(data):
            sys.exit(f"{path}: chunk {cid!r} size {size} exceeds file length")
        body = data[body_start:body_end]
        if cid == b"fmt ":
            if size < 16:
                sys.exit(f"{path}: fmt chunk too small ({size} bytes)")
            audio_format, channels, sample_rate = struct.unpack_from(
                "<HHI", body, 0
            )
            (bits,) = struct.unpack_from("<H", body, 14)
            fmt = (audio_format, channels, sample_rate, bits)
        elif cid == b"data":
            payload = body
        # RIFF chunks are word-aligned: skip a pad byte after an odd-sized body.
        pos = body_end + (size & 1)

    if fmt is None:
        sys.exit(f"{path}: no fmt chunk")
    if payload is None:
        sys.exit(f"{path}: no data chunk")

    audio_format, channels, sample_rate, bits = fmt
    if channels != 1:
        sys.exit(
            f"{path}: expected mono, got {channels} channels. Fix with "
            "`ffmpeg -ac 1 ...`; FR-EX-08 forbids silent mixdown."
        )
    if sample_rate != 16000:
        sys.exit(
            f"{path}: expected 16 kHz, got {sample_rate} Hz. Fix with "
            "`ffmpeg -ar 16000 ...`; FR-EX-08 forbids silent resample."
        )

    if audio_format == 3 and bits == 32:
        samples = np.frombuffer(payload, dtype="<f4").astype(np.float32)
    elif audio_format == 1 and bits == 16:
        ints = np.frombuffer(payload, dtype="<i2").astype(np.float32)
        samples = ints / 32768.0
    else:
        sys.exit(
            f"{path}: unsupported PCM format (audio_format={audio_format}, "
            f"bits={bits}); use mono float32 or int16"
        )

    if samples.size >= n_samples:
        return np.ascontiguousarray(samples[:n_samples], dtype=np.float32)
    # Short clip: right-pad with zeros to n_samples (matches Whisper's own
    # internal padding to N_SAMPLES).
    padded = np.zeros(n_samples, dtype=np.float32)
    padded[: samples.size] = samples
    return padded


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
    parser = argparse.ArgumentParser(
        description=(
            "Dump PyTorch (transformers) Whisper reference tensors for parity. "
            "Regenerates fixtures under tests/parity/whisper_{size}/."
        )
    )
    parser.add_argument(
        "--model",
        choices=sorted(SUPPORTED_MODELS.keys()),
        default="whisper-base",
        help="Which Whisper size to dump (fixed allowlist; no silent fallback).",
    )
    parser.add_argument(
        "--audio",
        default=str(DEFAULT_AUDIO),
        help=(
            "Path to a real 16 kHz mono WAV (PCM16 or IEEE_FLOAT32). Truncated "
            "or zero-padded to the first 30 s. Default: %(default)s. FR-EX-08: "
            "unsupported format / rate / channel count is a hard error, not a "
            "silent conversion."
        ),
    )
    parser.add_argument(
        "out_dir",
        nargs="?",
        default=None,
        help=(
            "Output directory. Defaults to tests/parity/whisper_{size}/ derived "
            "from --model."
        ),
    )
    args = parser.parse_args()

    size = args.model
    model_id = SUPPORTED_MODELS[size]
    out_dir = Path(args.out_dir) if args.out_dir is not None else Path(f"tests/parity/{size.replace('-', '_')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_path = Path(args.audio).resolve()
    pcm = load_pcm(audio_path, PCM_LEN)
    pcm_sha256 = hashlib.sha256(audio_path.read_bytes()).hexdigest()
    # Repo-relative source for the manifest; falls back to absolute if the WAV
    # lives outside the repo (auditability > prettiness).
    repo_root = Path(__file__).resolve().parents[2]
    try:
        pcm_source = audio_path.relative_to(repo_root).as_posix()
    except ValueError:
        pcm_source = str(audio_path)

    # The bundled text-vocab resource is regenerated only from the largest tail
    # (whisper-large-v3) — the first 50257 records are byte-identical across all
    # multilingual sizes, so any multilingual checkpoint is technically valid,
    # but we standardise on large-v3 to avoid drift.
    vocab_resource = VOCAB_RESOURCE if size == "whisper-large-v3" else None

    torch.manual_seed(TORCH_SEED)
    processor = WhisperProcessor.from_pretrained(model_id)
    tok = WhisperTokenizer.from_pretrained(model_id)  # slow tokenizer (has byte_decoder)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.eval()

    prefix, eot = resolve_special_tokens(tok)

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
        prefix_t = torch.tensor([prefix], dtype=torch.long)
        logits = model(input_features=feats, decoder_input_ids=prefix_t).logits  # [1, P, vocab]
        vocab = logits.shape[-1]
        write_f32(out_dir / "logits_last.f32", logits[0, -1, :].numpy())
        prefix_argmax = logits[0].argmax(dim=-1).tolist()

        # Plain greedy (argmax, no logit suppression) — matches the Rust loop.
        tokens = list(prefix)
        greedy = []
        for _ in range(MAX_NEW):
            lg = model(input_features=feats, decoder_input_ids=torch.tensor([tokens])).logits
            nxt = int(lg[0, -1, :].argmax().item())
            greedy.append(nxt)
            if nxt == eot:
                break
            tokens.append(nxt)

    vocab_n = dump_tokenizer(out_dir / "tokenizer.bin", tok, vocab_resource=vocab_resource)

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
        "model": model_id,
        "size": size,
        "torch_seed": TORCH_SEED,
        "atol": 0.01,
        "pcm_len": PCM_LEN,
        "pcm_source": pcm_source,
        "pcm_sha256": pcm_sha256,
        "n_mels": int(n_mels),
        "mel_frames": MEL_FRAMES,
        "enc_pos": ENC_POS,
        "d_model": int(enc.shape[-1]),
        "n_audio_ctx": int(enc.shape[1]),
        "vocab": int(vocab),
        "vocab_tokenizer": int(vocab_n),
        "prefix": prefix,
        "eot": eot,
        "prefix_argmax": prefix_argmax,
        "greedy_tokens": greedy,
        "greedy_text": greedy_text,
    }
    with (out_dir / "manifest.txt").open("w", encoding="utf-8") as f:
        f.write("# Whisper parity manifest (M0-06 / M2-06 §3). Generated by\n")
        f.write("# tools/parity/dump_whisper_reference.py. `key = value`;\n")
        f.write("# list values are space-separated. `pcm_source` /\n")
        f.write("# `pcm_sha256` pin the exact WAV that produced this fixture.\n")
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
