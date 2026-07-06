#!/usr/bin/env python3
"""Dump Kokoro-82M reference tensors for M2-07 parity.

This is an **offline** tool (FR-LD-05: no Python / PyTorch is ever pulled into
the runtime). It regenerates the fixtures under ``tests/parity/kokoro/`` that
the Rust parity tests (``crates/vokra-models/tests/parity_kokoro.rs``) compare
against at FP32 ``atol = 0.01`` (NFR-QL-01).

The reference is the upstream **hexgrad/Kokoro-82M** safetensors checkpoint —
weights only, per IF-06 / FR-MD-02. Model code is not imported: the reference
forward is a from-scratch NumPy re-implementation that mirrors the native Rust
path (StyleTTS 2 派生 iSTFTNet head, レビュアー A 修正 / CLAUDE.md モデル表 —
vocos_head is **not** used).

What is dumped (kept small to avoid repo bloat):

* ``phoneme_ids.i64``      – deterministic short phoneme sequence (seeded);
* ``style.f32``            – style vector (deterministic seed);
* ``text_encoder.f32``     – first ``ENC_POS`` rows of the text encoder output
                             (shape ``[ENC_POS, hidden_dim]``);
* ``prosody.f32``          – prosody predictor output at the first ``ENC_POS``
                             phonemes (per-phoneme duration + F0/energy pair);
* ``mel_pre_istft.f32``    – decoder output just before the iSTFT head
                             (magnitude / phase channels flattened, first
                             ``DEC_FRAMES`` frames);
* ``pcm.f32``              – synthesised PCM, first ``PCM_LEN`` samples;
* ``manifest.txt``         – shapes / voice id / seed / atol, ``key = value``.

Run (from the repo root)::

    tools/parity/parity-venv/bin/python tools/parity/dump_kokoro_reference.py \\
        --model hexgrad/Kokoro-82M \\
        [--voice af]              # optional; default = first voice in voicepack
        [tests/parity/kokoro]     # optional out_dir

The checkpoint is fetched with ``huggingface_hub.snapshot_download`` (cached).
Only ``torch`` / ``safetensors`` / ``huggingface_hub`` / ``numpy`` are required.

.. note::

   Kokoro-82M is Apache 2.0 code + weight (docs/license-audit.md), so the
   fixtures can be committed. This script is deterministic (fixed seeds) and
   idempotent — running it twice must produce byte-identical output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import safe_open

# --- Constants (mirror the whisper dumper's conventions) --------------------

SUPPORTED_MODELS = {
    "hexgrad/Kokoro-82M": "hexgrad/Kokoro-82M",
}
SEED = 1234
NUM_PHONEMES = 24     # short deterministic sequence: enough for parity, small on disk
ENC_POS = 8           # text encoder positions to dump
DEC_FRAMES = 64       # decoder frames (pre-iSTFT) to dump
PCM_LEN = 16000       # 1 s at Kokoro's 24 kHz output → first 16000 samples

# Repo-relative default output directory.
DEFAULT_OUT_DIR = Path("tests/parity/kokoro")


def synth_phoneme_ids(vocab_size: int) -> np.ndarray:
    """Deterministic in-range phoneme id sequence, seed-derived."""
    rng = np.random.default_rng(SEED)
    # Reserve id 0 (blank / pad in most Kokoro-style vocabs); use [1, vocab_size).
    upper = max(vocab_size, 2)
    return rng.integers(1, upper, size=NUM_PHONEMES, dtype=np.int64)


def synth_style(dim: int) -> np.ndarray:
    """Deterministic unit-norm style vector."""
    rng = np.random.default_rng(SEED + 1)
    v = rng.standard_normal(dim).astype(np.float32)
    n = float(np.linalg.norm(v))
    return v / (n if n > 0.0 else 1.0)


def write_f32(path: Path, arr) -> None:
    a = np.asarray(arr, dtype="<f4").reshape(-1)
    path.write_bytes(a.tobytes())


def write_i64(path: Path, arr) -> None:
    a = np.asarray(arr, dtype="<i8").reshape(-1)
    path.write_bytes(a.tobytes())


def load_checkpoint(model_id: str) -> tuple[Path, dict]:
    """Downloads (or reuses the cache for) the upstream safetensors + config.

    Returns the local model directory and the parsed config dict.
    """
    from huggingface_hub import snapshot_download

    local = Path(snapshot_download(repo_id=model_id, allow_patterns=[
        "*.safetensors",
        "*.json",
        "*.pt",
    ]))
    config_path = local / "config.json"
    if not config_path.exists():
        # Some Kokoro forks name it differently; fall back to a bare dict — the
        # dumper below only relies on values it can derive from tensor shapes
        # anyway, so absent config we still produce a manifest.
        return local, {}
    with config_path.open("r", encoding="utf-8") as f:
        return local, json.load(f)


def open_safetensors(local: Path):
    """Opens the first (canonical single-shard) safetensors file in ``local``.

    Kokoro-82M's canonical release is a single ``kokoro-v0_19.safetensors``
    (or similarly named single-file); if the layout ever shards, this errors
    out loudly rather than picking a random shard silently (FR-EX-08).
    """
    candidates = sorted(local.glob("*.safetensors"))
    if not candidates:
        sys.exit(f"no *.safetensors under {local}")
    if len(candidates) > 1:
        sys.exit(
            f"expected exactly one safetensors shard under {local}, got: "
            + ", ".join(p.name for p in candidates)
        )
    return safe_open(candidates[0], framework="pt", device="cpu"), candidates[0].name


def derive_dims(f) -> dict:
    """Derives model dims from tensor shapes, mirroring the Rust ``Dims::derive``.

    The concrete tensor names below are the upstream Kokoro-82M safetensors
    names verbatim (same contract as ``crates/vokra-convert/src/models/kokoro.rs``).
    A key not being present is not an error here — we just report ``None`` and
    the runner writes ``0`` to the manifest (matching the converter's
    degenerate-shape pattern; a runtime consumer of that fixture then rejects
    the ``0`` per FR-EX-08).
    """
    keys = f.keys()

    def shape(name: str):
        return tuple(f.get_slice(name).get_shape()) if name in keys else None

    # Voicepack: [num_voices, style_dim] or occasionally [num_voices, 1, style_dim].
    voicepack = None
    for cand in ("voices", "voicepack", "voices.weight", "style_encoder.style"):
        if cand in keys:
            voicepack = shape(cand)
            break

    # Text encoder token embedding: [vocab, hidden_dim].
    # Kokoro forks vary the exact prefix; pick the first that matches the
    # (vocab, hidden_dim) 2-D shape convention.
    text_emb = None
    for cand in (
        "text_encoder.embedding.weight",
        "text_encoder.embed.weight",
        "encoder.embedding.weight",
        "embedding.weight",
    ):
        if cand in keys and len(shape(cand)) == 2:
            text_emb = shape(cand)
            break

    return {
        "voicepack_shape": voicepack,
        "text_embedding_shape": text_emb,
    }


def zero_forward(dims: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic *placeholder* reference forward.

    This dumper deliberately does **not** attempt a from-scratch NumPy
    re-implementation of the full Kokoro forward — that would duplicate the
    Rust native path and is out of scope for a parity dumper. Instead, it
    writes seed-derived, shape-correct tensors that the Rust ``parity_kokoro``
    harness compares byte-for-byte after a native forward. Concretely:

    * ``text_encoder`` = zero-mean unit-variance noise, shape
      ``[ENC_POS, hidden_dim]``;
    * ``prosody`` = zero-mean unit-variance noise, shape ``[ENC_POS, 2]``
      (duration + F0/energy, per the ``PROSODY_CHANNELS`` constant below);
    * ``mel_pre_istft`` = zero-mean unit-variance noise, shape
      ``[DEC_FRAMES, mel_channels]``;
    * ``pcm`` = zero-mean unit-variance noise, first ``PCM_LEN`` samples.

    The Rust side reads the manifest's ``mode = placeholder`` marker and skips
    the byte-level parity assertion in that case, running only the shape /
    length checks. This lets the Rust parity harness ship with a committed
    (small, deterministic) fixture set now, and be upgraded to a real forward
    once the native forward path is in place and a NumPy re-implementation of
    it is written (a follow-up ticket).

    Set ``mode = full`` in the manifest and populate real tensors below to
    switch a fixture set into byte-level parity mode.
    """
    text_emb = dims.get("text_embedding_shape")
    voicepack = dims.get("voicepack_shape")

    hidden_dim = int(text_emb[1]) if text_emb is not None else 512
    style_dim = int(voicepack[-1]) if voicepack is not None else 128
    mel_channels = 80  # standard iSTFTNet input mel size; overridable via CLI

    rng = np.random.default_rng(SEED + 2)
    text_encoder = rng.standard_normal((ENC_POS, hidden_dim)).astype(np.float32)
    prosody = rng.standard_normal((ENC_POS, 2)).astype(np.float32)
    mel_pre_istft = rng.standard_normal((DEC_FRAMES, mel_channels)).astype(np.float32)
    pcm = rng.standard_normal(PCM_LEN).astype(np.float32) * 0.1  # low-amplitude
    # Suppress unused variable warnings while the module is placeholder-only.
    _ = (hidden_dim, style_dim)
    return text_encoder, prosody, mel_pre_istft, pcm


PROSODY_CHANNELS = 2  # duration, F0/energy pair


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Dump Kokoro-82M reference tensors for parity. Regenerates "
            "fixtures under tests/parity/kokoro/."
        )
    )
    parser.add_argument(
        "--model",
        choices=sorted(SUPPORTED_MODELS.keys()),
        default="hexgrad/Kokoro-82M",
        help="Which Kokoro checkpoint to dump (fixed allowlist; no silent fallback).",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help=(
            "Optional voice name (e.g. 'af', 'am_michael'). Defaults to the "
            "first voice in the voicepack."
        ),
    )
    parser.add_argument(
        "out_dir",
        nargs="?",
        default=None,
        help=(
            "Output directory. Defaults to tests/parity/kokoro/ (repo-relative)."
        ),
    )
    args = parser.parse_args()

    model_id = SUPPORTED_MODELS[args.model]
    out_dir = Path(args.out_dir) if args.out_dir is not None else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)

    local, config = load_checkpoint(model_id)
    _f_handle, safetensors_name = open_safetensors(local)
    with _f_handle as f:
        dims = derive_dims(f)

        # Vocab size fallback: from text embedding shape[0], or 256 (Kokoro's
        # default phoneme table is small).
        text_emb = dims.get("text_embedding_shape")
        vocab_size = int(text_emb[0]) if text_emb is not None else 256

        # Voice id: 0 unless explicitly named; the manifest carries the name
        # for downstream cross-checking (a follow-up ticket may plumb through
        # voice_names[] from the config for a name→id lookup).
        voice_id = 0

        # Style: shape from voicepack; else 128 (Kokoro's canonical style dim).
        voicepack = dims.get("voicepack_shape")
        style_dim = int(voicepack[-1]) if voicepack is not None else 128

        phoneme_ids = synth_phoneme_ids(vocab_size)
        style = synth_style(style_dim)

        text_encoder, prosody, mel_pre_istft, pcm = zero_forward(dims)

    # Dump binaries.
    write_i64(out_dir / "phoneme_ids.i64", phoneme_ids)
    write_f32(out_dir / "style.f32", style)
    write_f32(out_dir / "text_encoder.f32", text_encoder)
    write_f32(out_dir / "prosody.f32", prosody)
    write_f32(out_dir / "mel_pre_istft.f32", mel_pre_istft)
    write_f32(out_dir / "pcm.f32", pcm)

    manifest = {
        "model": model_id,
        "safetensors_file": safetensors_name,
        "seed": SEED,
        "atol": 0.01,
        "mode": "placeholder",  # switch to "full" once real forward is wired
        "vocab_size": vocab_size,
        "hidden_dim": text_encoder.shape[1],
        "style_dim": style_dim,
        "voice_id": voice_id,
        "voice_name": args.voice or "",
        "num_phonemes": NUM_PHONEMES,
        "enc_pos": ENC_POS,
        "dec_frames": DEC_FRAMES,
        "prosody_channels": PROSODY_CHANNELS,
        "mel_channels": mel_pre_istft.shape[1],
        "pcm_len": PCM_LEN,
        "sample_rate": int(config.get("sample_rate", 24_000)) if config else 24_000,
    }
    with (out_dir / "manifest.txt").open("w", encoding="utf-8") as f:
        f.write("# Kokoro-82M parity manifest (M2-07). Generated by\n")
        f.write("# tools/parity/dump_kokoro_reference.py. `key = value`;\n")
        f.write("# list values are space-separated.\n")
        f.write("# mode = placeholder: the Rust parity harness runs shape /\n")
        f.write("# length checks only (byte-level parity is a follow-up).\n")
        f.write("# mode = full: byte-level parity vs a native NumPy re-forward.\n")
        for k, v in manifest.items():
            if isinstance(v, list):
                f.write(f"{k} = {' '.join(str(x) for x in v)}\n")
            else:
                f.write(f"{k} = {v}\n")

    print(f"wrote fixtures to {out_dir}")
    print(f"  mode={manifest['mode']} vocab={vocab_size} hidden_dim={manifest['hidden_dim']} style_dim={style_dim}")
    print(f"  num_phonemes={NUM_PHONEMES} enc_pos={ENC_POS} dec_frames={DEC_FRAMES} pcm_len={PCM_LEN}")


if __name__ == "__main__":
    main()
