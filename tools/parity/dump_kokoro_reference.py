#!/usr/bin/env python3
"""Dump Kokoro-82M reference tensors for M2-07 parity.

This is an **offline** tool (FR-LD-05: no Python / PyTorch is ever pulled into
the runtime). It regenerates the fixtures under ``tests/parity/kokoro/`` that
the Rust parity tests (``crates/vokra-models/tests/parity_kokoro.rs``) compare
against at FP32 ``atol = 0.01`` (NFR-QL-01).

The reference is the upstream **hexgrad/Kokoro-82M** checkpoint — weights
only, per IF-06 / FR-MD-02. Model **code** is not imported: the reference
forward is a from-scratch PyTorch re-implementation (used strictly as a
tensor evaluator; f32 ops are the same math the native Rust path implements)
that mirrors the layer-by-layer pipeline in
``crates/vokra-models/src/kokoro/{text_encoder,bert}.rs`` (StyleTTS 2 派生
iSTFTNet head, レビュアー A 修正 / CLAUDE.md モデル表 — vocos_head is **not**
used).

# Modes

* ``--mode placeholder`` (default): seed-derived, shape-correct tensors.
  The Rust parity harness reads ``mode = placeholder`` from the manifest and
  runs shape / length checks only (byte-level parity is intentionally skipped).
* ``--mode full``: byte-level parity mode. Runs a PyTorch re-implementation
  of the Rust forward for every module that has landed
  (T02→T18); dumps its output at the first ``ENC_POS`` positions. Modules
  whose NumPy re-forward is not yet implemented are marked
  ``<module>_mode = placeholder`` in the manifest and the Rust harness
  skips byte-level parity for those.

# Fixture files

* ``phoneme_ids.i64``       – deterministic short phoneme sequence (seeded);
* ``style.f32``             – style vector (deterministic seed; not consumed
                              by text_encoder / bert but needed by prosody /
                              decoder when those land);
* ``text_encoder.f32``      – first ``ENC_POS`` rows of the text encoder output
                              (shape ``[ENC_POS, hidden_dim]``);
* ``bert.f32``              – first ``ENC_POS`` rows of the bert output
                              (shape ``[ENC_POS, 512]``) — only written in
                              ``mode = full`` when the bert branch NumPy
                              re-forward is enabled;
* ``prosody.f32``           – per-phoneme duration + F0/energy pair (shape
                              ``[ENC_POS, 2]``) — placeholder until prosody
                              re-forward lands;
* ``mel_pre_istft.f32``     – decoder pre-iSTFT tensor (shape
                              ``[DEC_FRAMES, mel_channels]``) — placeholder
                              until decoder re-forward lands;
* ``pcm.f32``               – first ``PCM_LEN`` synthesised PCM samples —
                              placeholder until decoder re-forward lands;
* ``manifest.txt``          – shapes / voice id / seed / atol / per-module
                              modes, ``key = value``.

# Determinism

Idempotent: rerunning with the same ``--mode`` and inputs produces byte-
identical output. Fixed seeds (``SEED = 1234``) and ``torch.manual_seed``.

# Usage (from the repo root)

::

    tools/parity/parity-venv/bin/python tools/parity/dump_kokoro_reference.py \\
        --mode {placeholder,full} \\
        --model hexgrad/Kokoro-82M \\
        [--voice af]              # optional; default = first voice in voicepack
        [tests/parity/kokoro]     # optional out_dir

The checkpoint is fetched with ``huggingface_hub.snapshot_download`` (cached).
Only ``torch`` / ``safetensors`` / ``huggingface_hub`` / ``numpy`` are required.

.. note::

   Kokoro-82M is Apache 2.0 code + weight (docs/license-audit.md), so the
   fixtures can be committed.
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

# --- Architectural constants pinned by the upstream manifest ---------------
#
# Every constant here corresponds to a shape axis in
# ``crates/vokra-models/src/kokoro/data/upstream_tensors_v1_0.tsv``. They
# match the Rust-side constants in ``bert.rs`` (N_VOCAB, EMBED_SIZE, HIDDEN,
# FFN_HIDDEN, MAX_POS, N_TOKEN_TYPES, N_LAYERS, OUT_DIM, N_HEADS,
# LAYER_NORM_EPS). If the checkpoint's tensor shape disagrees, we abort
# loudly rather than silently reshape.

BERT_N_VOCAB = 178
BERT_EMBED_SIZE = 128
BERT_HIDDEN = 768
BERT_FFN_HIDDEN = 2048
BERT_MAX_POS = 512
BERT_N_TOKEN_TYPES = 2
BERT_N_LAYERS = 4
BERT_OUT_DIM = 512
BERT_N_HEADS = 12
BERT_LAYER_NORM_EPS = 1e-12

# Text encoder constants (kokoro-v1_0.pth):
TE_KERNEL = 5
TE_PAD = 2
TE_NUM_CNN_BLOCKS = 3
TE_LRELU_SLOPE = 0.1


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
    """Downloads (or reuses the cache for) the upstream checkpoint + config.

    Kokoro-82M's canonical release is ``kokoro-v1_0.pth`` (PyTorch pickle);
    some downstream forks re-export to safetensors. This dumper accepts
    either so it works with the vanilla upstream and with the safetensors
    checkpoint the Vokra converter (``crates/vokra-convert/src/models/kokoro.rs``)
    expects.

    Returns the local model directory and the parsed config dict.
    """
    from huggingface_hub import snapshot_download

    local = Path(snapshot_download(repo_id=model_id, allow_patterns=[
        "*.safetensors",
        "*.json",
        "*.pt",
        "*.pth",
        "*.bin",
    ]))
    config_path = local / "config.json"
    if not config_path.exists():
        # Some Kokoro forks name it differently; fall back to a bare dict — the
        # dumper below only relies on values it can derive from tensor shapes
        # anyway, so absent config we still produce a manifest.
        return local, {}
    with config_path.open("r", encoding="utf-8") as f:
        return local, json.load(f)


class _ShapeMap:
    """Uniform "``name -> shape``" surface over safetensors *or* a torch pickle.

    Kokoro-82M's canonical release ships weights as ``kokoro-v1_0.pth``
    (a torch pickle); the safetensors path is what the Vokra converter
    expects after a downstream re-export. This class hides the difference
    so the shape-derivation code below stays flat.
    """

    def __init__(self, keys, shape_fn, description, tensor_fn=None):
        self._keys = set(keys)
        self._shape_fn = shape_fn
        self.description = description
        self._tensor_fn = tensor_fn  # optional: name -> torch.Tensor (for mode=full)

    def keys(self):
        return self._keys

    def shape(self, name):
        if name not in self._keys:
            return None
        return tuple(self._shape_fn(name))

    def tensor(self, name):
        """Fetch full tensor (torch.Tensor). Only available for the .pth path."""
        if self._tensor_fn is None:
            raise RuntimeError(
                f"tensor({name!r}): shape-only ShapeMap (safetensors path). "
                "mode=full needs the .pth backend."
            )
        if name not in self._keys:
            raise KeyError(f"tensor {name!r} not in checkpoint")
        return self._tensor_fn(name)


def open_checkpoint(local: Path) -> _ShapeMap:
    """Opens the canonical single-shard checkpoint in ``local``.

    Prefers safetensors when present (matches the Vokra converter's input
    shape); falls back to the upstream ``.pth`` / ``.pt`` / ``.bin`` file.
    Fails loudly on multiple candidates in the same family (FR-EX-08).
    """
    st_candidates = sorted(local.glob("*.safetensors"))
    if len(st_candidates) == 1:
        h = safe_open(st_candidates[0], framework="pt", device="cpu")
        h.__enter__()  # keep the file handle open for the caller
        keys = list(h.keys())

        def shape_fn(name):
            return h.get_slice(name).get_shape()

        def tensor_fn(name):
            return h.get_tensor(name)

        return _ShapeMap(
            keys, shape_fn, description=st_candidates[0].name, tensor_fn=tensor_fn
        )
    if len(st_candidates) > 1:
        sys.exit(
            f"expected exactly one safetensors shard under {local}, got: "
            + ", ".join(p.name for p in st_candidates)
        )

    # Fall back to a torch pickle. Kokoro's release is `kokoro-v1_0.pth`; some
    # forks use `.pt` or `.bin`. Only pick top-level model files, not the
    # per-voice `voices/*.pt` style-vector store.
    pt_candidates = sorted(
        p
        for p in [*local.glob("*.pth"), *local.glob("*.pt"), *local.glob("*.bin")]
        if p.parent == local
    )
    if not pt_candidates:
        sys.exit(f"no *.safetensors / *.pth / *.pt / *.bin under {local}")
    if len(pt_candidates) > 1:
        sys.exit(
            f"expected exactly one top-level checkpoint under {local}, got: "
            + ", ".join(p.name for p in pt_candidates)
        )

    # `weights_only=True` refuses arbitrary-code-execution payloads (introduced
    # in PyTorch 2.4). This is the safe way to load an upstream .pth here.
    try:
        state = torch.load(pt_candidates[0], map_location="cpu", weights_only=True)
    except Exception as exc:
        sys.exit(
            f"torch.load({pt_candidates[0].name!r}, weights_only=True) failed: "
            f"{exc}"
        )

    # Kokoro's .pth stores a nested dict: `{'bert': {...}, 'text_encoder': {...},
    # ...}`. Flatten to `submodule.tensor_name -> Tensor`.
    def flatten(prefix: str, obj):
        out = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else k
                out.update(flatten(key, v))
        elif isinstance(obj, torch.Tensor):
            out[prefix] = obj
        return out

    flat = flatten("", state)
    keys = list(flat.keys())

    def shape_fn(name):
        return tuple(flat[name].shape)

    def tensor_fn(name):
        return flat[name]

    return _ShapeMap(
        keys, shape_fn, description=pt_candidates[0].name, tensor_fn=tensor_fn
    )


def derive_dims(store: _ShapeMap) -> dict:
    """Derives model dims from tensor shapes, mirroring the Rust ``Dims::derive``.

    The concrete tensor names below are the upstream Kokoro-82M weight names
    (same contract as ``crates/vokra-convert/src/models/kokoro.rs``). Kokoro's
    canonical .pth ships a nested ``{'bert': ..., 'text_encoder': ...,
    'predictor': ..., 'decoder': ...}`` state dict — after ``open_checkpoint``
    flattens it, the tensor names include the ``.module.`` ``nn.DataParallel``
    prefix (e.g. ``text_encoder.module.embedding.weight``). A key not being
    present is not an error here — we just report ``None`` and the runner writes
    ``0`` to the manifest (matching the converter's degenerate-shape pattern; a
    runtime consumer of that fixture then rejects the ``0`` per FR-EX-08).
    """
    keys = store.keys()

    # Voicepack: [num_voices, style_dim] or occasionally [num_voices, 1,
    # style_dim]. Note: the canonical Kokoro release stores voice style
    # vectors as PER-VOICE FILES under ``voices/*.pt`` (rather than as a
    # single stacked tensor inside the model .pth), so this may be ``None``
    # when consuming the upstream .pth directly.
    voicepack = None
    for cand in ("voices", "voicepack", "voices.weight", "style_encoder.style"):
        if cand in keys:
            voicepack = store.shape(cand)
            break

    # Text encoder token embedding: [vocab, hidden_dim]. Kokoro's real .pth
    # uses the ``text_encoder.module.embedding.weight`` name (nn.DataParallel
    # prefix). We look at that first; a downstream fork may drop the
    # ``.module.`` prefix, so the alternates below are kept for compatibility.
    text_emb = None
    for cand in (
        "text_encoder.module.embedding.weight",
        "text_encoder.embedding.weight",
        "text_encoder.embed.weight",
        "encoder.embedding.weight",
        "embedding.weight",
    ):
        if cand in keys:
            s = store.shape(cand)
            if s is not None and len(s) == 2:
                text_emb = s
                break

    return {
        "voicepack_shape": voicepack,
        "text_embedding_shape": text_emb,
    }


def placeholder_forward(dims: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic *placeholder* reference forward.

    Writes seed-derived, shape-correct tensors that the Rust ``parity_kokoro``
    harness compares byte-for-byte after a native forward (in ``mode = full``)
    or shape-only (in ``mode = placeholder``). Concretely:

    * ``text_encoder`` = zero-mean unit-variance noise, shape
      ``[ENC_POS, hidden_dim]``;
    * ``prosody`` = zero-mean unit-variance noise, shape ``[ENC_POS, 2]``
      (duration + F0/energy, per the ``PROSODY_CHANNELS`` constant below);
    * ``mel_pre_istft`` = zero-mean unit-variance noise, shape
      ``[DEC_FRAMES, mel_channels]``;
    * ``pcm`` = zero-mean unit-variance noise, first ``PCM_LEN`` samples.
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
    # Suppress unused variable warnings.
    _ = (hidden_dim, style_dim)
    return text_encoder, prosody, mel_pre_istft, pcm


# ---------------------------------------------------------------------------
# mode = full — PyTorch re-implementation of the Rust forward path per module
# ---------------------------------------------------------------------------
#
# Each module below mirrors the exact layer-by-layer pipeline in the Rust
# implementation. The re-forwards use torch's F.conv1d / F.layer_norm / etc.
# but at f32 with the same math the native Rust path uses. Byte-level parity
# vs the Rust forward holds at ``atol = 0.01`` (NFR-QL-01).


def _forward_text_encoder(store: _ShapeMap, phoneme_ids: np.ndarray) -> np.ndarray:
    """PyTorch re-implementation of ``kokoro::text_encoder::TextEncoder::forward``.

    Pipeline (see ``crates/vokra-models/src/kokoro/text_encoder.rs``):

    1. Embedding lookup   → [t, hidden]
    2. Transpose          → [1, hidden, t]  (channel-major batch-first)
    3. 3× (WeightNormed Conv1d(k=5, pad=2, stride=1) + bias + γ·x+β + LeakyReLU(0.1))
    4. Transpose          → [1, t, hidden]
    5. BiLSTM(hidden → hidden/2, bidirectional) → [t, hidden]

    WeightNorm reconstruction: ``w = g · v / ||v||_2`` per output channel,
    matching ``kokoro::nn::weight_norm_reconstruct_1d``. Zero-norm rows
    degrade to zero (matches Rust's guard).
    """
    import torch.nn.functional as F
    from torch.nn import LSTM

    device = torch.device("cpu")
    dtype = torch.float32

    # ---- Load weights ----
    emb = store.tensor("text_encoder.module.embedding.weight").to(device=device, dtype=dtype)
    n_vocab, hidden = int(emb.shape[0]), int(emb.shape[1])
    if hidden % 2 != 0:
        raise RuntimeError(f"text_encoder hidden ({hidden}) must be even for BiLSTM")

    # 1. Embedding
    ids = torch.from_numpy(phoneme_ids.astype(np.int64))
    if (ids < 0).any() or (ids >= n_vocab).any():
        raise RuntimeError(
            f"text_encoder: phoneme id out of range 0..{n_vocab}; "
            f"got ids in [{int(ids.min())}, {int(ids.max())}]"
        )
    x = F.embedding(ids, emb)  # [t, hidden]

    # 2. Transpose → [1, hidden, t] for F.conv1d
    x = x.transpose(0, 1).unsqueeze(0).contiguous()

    # 3. Three CNN blocks
    for i in range(TE_NUM_CNN_BLOCKS):
        wg = store.tensor(f"text_encoder.module.cnn.{i}.0.weight_g").to(dtype=dtype)  # [hidden,1,1]
        wv = store.tensor(f"text_encoder.module.cnn.{i}.0.weight_v").to(dtype=dtype)  # [hidden,hidden,K]
        bias = store.tensor(f"text_encoder.module.cnn.{i}.0.bias").to(dtype=dtype)
        gamma = store.tensor(f"text_encoder.module.cnn.{i}.1.gamma").to(dtype=dtype)
        beta = store.tensor(f"text_encoder.module.cnn.{i}.1.beta").to(dtype=dtype)

        # WeightNorm reconstruct: w = g * v / ||v||_2 per output channel.
        # ||v||_2 is L2 over (in_ch, kernel) axes.
        norm = wv.reshape(hidden, -1).norm(dim=1).view(hidden, 1, 1)
        # Zero-norm guard (matches Rust). Use a mask; a well-trained checkpoint
        # never triggers this.
        safe = torch.where(norm > 0, norm, torch.ones_like(norm))
        w = wg * wv / safe
        w = torch.where(norm > 0, w, torch.zeros_like(w))

        # Conv1d k=5, pad=2, stride=1
        x = F.conv1d(x, w, bias=bias, stride=1, padding=TE_PAD)

        # Per-channel affine (γ · x + β)
        x = x * gamma.view(1, -1, 1) + beta.view(1, -1, 1)

        # LeakyReLU(0.1)
        x = F.leaky_relu(x, TE_LRELU_SLOPE)

    # 4. Transpose to [1, t, hidden] for LSTM
    x = x.squeeze(0).transpose(0, 1).unsqueeze(0).contiguous()

    # 5. BiLSTM
    lstm_hidden = hidden // 2
    lstm = LSTM(
        input_size=hidden,
        hidden_size=lstm_hidden,
        num_layers=1,
        bias=True,
        batch_first=True,
        bidirectional=True,
    ).to(device=device, dtype=dtype)
    state_dict = {
        "weight_ih_l0": store.tensor("text_encoder.module.lstm.weight_ih_l0").to(dtype=dtype),
        "weight_hh_l0": store.tensor("text_encoder.module.lstm.weight_hh_l0").to(dtype=dtype),
        "bias_ih_l0": store.tensor("text_encoder.module.lstm.bias_ih_l0").to(dtype=dtype),
        "bias_hh_l0": store.tensor("text_encoder.module.lstm.bias_hh_l0").to(dtype=dtype),
        "weight_ih_l0_reverse": store.tensor(
            "text_encoder.module.lstm.weight_ih_l0_reverse"
        ).to(dtype=dtype),
        "weight_hh_l0_reverse": store.tensor(
            "text_encoder.module.lstm.weight_hh_l0_reverse"
        ).to(dtype=dtype),
        "bias_ih_l0_reverse": store.tensor(
            "text_encoder.module.lstm.bias_ih_l0_reverse"
        ).to(dtype=dtype),
        "bias_hh_l0_reverse": store.tensor(
            "text_encoder.module.lstm.bias_hh_l0_reverse"
        ).to(dtype=dtype),
    }
    lstm.load_state_dict(state_dict)
    lstm.eval()
    with torch.no_grad():
        y, _ = lstm(x)  # [1, t, 2·lstm_hidden] = [1, t, hidden]
    return y.squeeze(0).detach().cpu().numpy().astype(np.float32)


def _forward_bert(store: _ShapeMap, phoneme_ids: np.ndarray) -> np.ndarray:
    """PyTorch re-implementation of ``kokoro::bert::Bert::forward``.

    Pipeline (see ``crates/vokra-models/src/kokoro/bert.rs``):

    1. Embedding sum: word[id] + position[i] + token_type[0]  → [t, 128]
    2. LayerNorm across channels                              → [t, 128]
    3. mapping_in Linear (128 → 768)                          → [t, 768]
    4. 4× ALBERT-shared block:
       * Q/K/V/O linears
       * per-head attention (12 heads, head_dim = 64)
       * scale Q by 1/sqrt(head_dim)
       * residual + LayerNorm (attn_ln)
       * FFN (768 → 2048 → GELU → 768)
       * residual + LayerNorm (full_ln)
    5. Pooler Linear + tanh per-token                         → [t, 768]
    6. Downstream projection (768 → 512)                      → [t, 512]

    LayerNorm eps = 1e-12 (ALBERT convention, distinct from Whisper's 1e-5).
    """
    import torch.nn.functional as F

    device = torch.device("cpu")
    dtype = torch.float32

    def get(name):
        return store.tensor(name).to(device=device, dtype=dtype)

    # ---- Embeddings ----
    word_emb = get("bert.module.embeddings.word_embeddings.weight")
    pos_emb = get("bert.module.embeddings.position_embeddings.weight")
    type_emb = get("bert.module.embeddings.token_type_embeddings.weight")
    emb_ln_w = get("bert.module.embeddings.LayerNorm.weight")
    emb_ln_b = get("bert.module.embeddings.LayerNorm.bias")

    # Shape gates (a mismatch here means the .pth doesn't match the manifest).
    if tuple(word_emb.shape) != (BERT_N_VOCAB, BERT_EMBED_SIZE):
        raise RuntimeError(
            f"bert.word_embeddings shape {tuple(word_emb.shape)} != expected "
            f"({BERT_N_VOCAB}, {BERT_EMBED_SIZE})"
        )
    if tuple(pos_emb.shape) != (BERT_MAX_POS, BERT_EMBED_SIZE):
        raise RuntimeError(
            f"bert.position_embeddings shape {tuple(pos_emb.shape)} != expected "
            f"({BERT_MAX_POS}, {BERT_EMBED_SIZE})"
        )

    ids = torch.from_numpy(phoneme_ids.astype(np.int64))
    if (ids < 0).any() or (ids >= BERT_N_VOCAB).any():
        raise RuntimeError(
            f"bert: phoneme id out of range 0..{BERT_N_VOCAB}; "
            f"got ids in [{int(ids.min())}, {int(ids.max())}]"
        )
    t = ids.shape[0]
    if t > BERT_MAX_POS:
        raise RuntimeError(f"bert: sequence length {t} exceeds MAX_POS {BERT_MAX_POS}")

    # 1. Embedding sum
    word_e = F.embedding(ids, word_emb)  # [t, 128]
    pos_e = pos_emb[:t]  # [t, 128]
    # token_type_id = 0 for every position (single segment)
    type_e = type_emb[0].unsqueeze(0).expand(t, -1)  # [t, 128]
    embeds = word_e + pos_e + type_e

    # 2. LayerNorm across the innermost axis (channels)
    x = F.layer_norm(embeds, (BERT_EMBED_SIZE,), emb_ln_w, emb_ln_b, eps=BERT_LAYER_NORM_EPS)

    # 3. mapping_in Linear (128 → 768)
    mapping_w = get("bert.module.encoder.embedding_hidden_mapping_in.weight")  # [768, 128]
    mapping_b = get("bert.module.encoder.embedding_hidden_mapping_in.bias")    # [768]
    x = F.linear(x, mapping_w, mapping_b)  # [t, 768]

    # ---- Shared ALBERT block (loaded ONCE, applied N_LAYERS times) ----
    prefix = "bert.module.encoder.albert_layer_groups.0.albert_layers.0"
    q_w = get(f"{prefix}.attention.query.weight")
    q_b = get(f"{prefix}.attention.query.bias")
    k_w = get(f"{prefix}.attention.key.weight")
    k_b = get(f"{prefix}.attention.key.bias")
    v_w = get(f"{prefix}.attention.value.weight")
    v_b = get(f"{prefix}.attention.value.bias")
    o_w = get(f"{prefix}.attention.dense.weight")
    o_b = get(f"{prefix}.attention.dense.bias")
    attn_ln_w = get(f"{prefix}.attention.LayerNorm.weight")
    attn_ln_b = get(f"{prefix}.attention.LayerNorm.bias")
    ffn_w = get(f"{prefix}.ffn.weight")
    ffn_b = get(f"{prefix}.ffn.bias")
    ffn_out_w = get(f"{prefix}.ffn_output.weight")
    ffn_out_b = get(f"{prefix}.ffn_output.bias")
    full_ln_w = get(f"{prefix}.full_layer_layer_norm.weight")
    full_ln_b = get(f"{prefix}.full_layer_layer_norm.bias")

    head_dim = BERT_HIDDEN // BERT_N_HEADS  # 64
    scale = head_dim ** -0.5

    for _layer_idx in range(BERT_N_LAYERS):
        # Q/K/V/O — F.linear does out = x @ W^T + b (matches PyTorch nn.Linear)
        q = F.linear(x, q_w, q_b) * scale        # [t, 768]
        k = F.linear(x, k_w, k_b)
        v = F.linear(x, v_w, v_b)
        # Per-head reshape: [t, 768] → [t, N_HEADS, head_dim] → [N_HEADS, t, head_dim]
        q = q.view(t, BERT_N_HEADS, head_dim).transpose(0, 1)  # [12, t, 64]
        k = k.view(t, BERT_N_HEADS, head_dim).transpose(0, 1)
        v = v.view(t, BERT_N_HEADS, head_dim).transpose(0, 1)
        # scores [12, t, t] = q @ k^T (no causal mask; ALBERT is bidirectional)
        scores = q @ k.transpose(-1, -2)
        probs = F.softmax(scores, dim=-1)
        ctx = probs @ v  # [12, t, 64]
        ctx = ctx.transpose(0, 1).contiguous().view(t, BERT_HIDDEN)  # [t, 768]
        # Attention output projection
        attn_out = F.linear(ctx, o_w, o_b)  # [t, 768]
        # Residual + LayerNorm
        x = F.layer_norm(
            attn_out + x, (BERT_HIDDEN,), attn_ln_w, attn_ln_b, eps=BERT_LAYER_NORM_EPS
        )
        # FFN: 768 → 2048 → GELU → 768
        ffn_h = F.linear(x, ffn_w, ffn_b)
        ffn_h = F.gelu(ffn_h)  # PyTorch default is erf-based, matches Rust
        ffn_o = F.linear(ffn_h, ffn_out_w, ffn_out_b)
        x = F.layer_norm(
            ffn_o + x, (BERT_HIDDEN,), full_ln_w, full_ln_b, eps=BERT_LAYER_NORM_EPS
        )

    # 5. Pooler (per-token): Linear + tanh
    pooler_w = get("bert.module.pooler.weight")  # [768, 768]
    pooler_b = get("bert.module.pooler.bias")
    pooled = torch.tanh(F.linear(x, pooler_w, pooler_b))  # [t, 768]

    # 6. Downstream projection 768 → 512
    proj_w = get("bert_encoder.module.weight")   # [512, 768]
    proj_b = get("bert_encoder.module.bias")     # [512]
    out = F.linear(pooled, proj_w, proj_b)       # [t, 512]

    return out.detach().cpu().numpy().astype(np.float32)


PROSODY_CHANNELS = 2  # duration, F0/energy pair

# Prosody predictor constants (kokoro-v1_0.pth):
PROSODY_STYLE_DIM = 128  # matches the .fc.weight shape [1024, 128] on AdaLN
PROSODY_D_MODEL = 512    # BiLSTM output width; matches .lstm.weight_hh_l0 [1024, 256]
PROSODY_HALF = 256       # d_model / 2; F0/N branch shrinks to this in block 1
PROSODY_MAX_DUR = 50     # duration_proj.linear_layer.weight [50, 512]
PROSODY_LSTM_HIDDEN = 256  # d_model / 2
PROSODY_LRELU_SLOPE = 0.1
PROSODY_ADALN_EPS = 1e-5


def _forward_bilstm(store, prefix: str, x: "torch.Tensor", input_dim: int, hidden_dim: int) -> "torch.Tensor":
    """Runs one PyTorch nn.LSTM(bidirectional=True) forward matching the Rust
    [`BiLstm1d`] layout. ``x`` is ``[t, input_dim]`` row-major; returns
    ``[t, 2·hidden_dim]`` row-major.
    """
    from torch.nn import LSTM
    import torch as _torch

    device = _torch.device("cpu")
    dtype = _torch.float32
    lstm = LSTM(
        input_size=input_dim,
        hidden_size=hidden_dim,
        num_layers=1,
        bias=True,
        batch_first=True,
        bidirectional=True,
    ).to(device=device, dtype=dtype)
    sd = {
        "weight_ih_l0": store.tensor(f"{prefix}.weight_ih_l0").to(dtype=dtype),
        "weight_hh_l0": store.tensor(f"{prefix}.weight_hh_l0").to(dtype=dtype),
        "bias_ih_l0": store.tensor(f"{prefix}.bias_ih_l0").to(dtype=dtype),
        "bias_hh_l0": store.tensor(f"{prefix}.bias_hh_l0").to(dtype=dtype),
        "weight_ih_l0_reverse": store.tensor(f"{prefix}.weight_ih_l0_reverse").to(dtype=dtype),
        "weight_hh_l0_reverse": store.tensor(f"{prefix}.weight_hh_l0_reverse").to(dtype=dtype),
        "bias_ih_l0_reverse": store.tensor(f"{prefix}.bias_ih_l0_reverse").to(dtype=dtype),
        "bias_hh_l0_reverse": store.tensor(f"{prefix}.bias_hh_l0_reverse").to(dtype=dtype),
    }
    lstm.load_state_dict(sd)
    lstm.eval()
    with _torch.no_grad():
        y, _ = lstm(x.unsqueeze(0))  # [1, t, 2·hidden]
    return y.squeeze(0)


def _wn_reconstruct(g: "torch.Tensor", v: "torch.Tensor") -> "torch.Tensor":
    """Reconstruct ``w = g · v / ||v||_2`` matching the Rust
    [`weight_norm_reconstruct_1d`]. ``g`` shape ``[out_ch, 1, 1]``, ``v`` shape
    ``[out_ch, in_ch, k]``. Zero-norm rows degrade to zero, matching Rust.
    """
    import torch as _torch
    oc = v.shape[0]
    norm = v.reshape(oc, -1).norm(dim=1).view(oc, 1, 1)
    safe = _torch.where(norm > 0, norm, _torch.ones_like(norm))
    w = g * v / safe
    return _torch.where(norm > 0, w, _torch.zeros_like(w))


def _adaln_layernorm_1d(
    x: "torch.Tensor",
    channels: int,
    fc_w: "torch.Tensor",
    fc_b: "torch.Tensor",
    style: "torch.Tensor",
) -> "torch.Tensor":
    """Row-major ``[t, channels]`` LayerNorm-across-channels + ``(1+γ)·norm(x) + β``.

    Mirrors Rust [`adaln_layernorm_1d`] exactly:
    * γ, β = fc(style)[:channels], fc(style)[channels:] (fc = Linear(style_dim → 2·channels)).
    * eps = 1e-5 (nn.LayerNorm default).
    * variance is biased (division by C, not C-1) — matches Rust's
      ``inv_c = 1/channels`` reduction.
    """
    import torch as _torch
    import torch.nn.functional as F

    # 1. Project style → (γ, β) via row-major Linear.
    gb = F.linear(style.unsqueeze(0), fc_w, fc_b).squeeze(0)  # [2·channels]
    gamma_raw = gb[:channels]
    beta = gb[channels:]
    # 2. LayerNorm across channels per row (biased var, eps=1e-5).
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    norm = (x - mean) / _torch.sqrt(var + PROSODY_ADALN_EPS)
    # 3. (1+γ)·norm(x) + β, broadcasting [channels] across rows.
    return norm * (1.0 + gamma_raw).unsqueeze(0) + beta.unsqueeze(0)


def _adain_channel_major(
    x: "torch.Tensor",
    channels: int,
    fc_w: "torch.Tensor",
    fc_b: "torch.Tensor",
    style: "torch.Tensor",
) -> "torch.Tensor":
    """Channel-major ``[channels, time]`` AdaIN with ``(1+γ)·norm(x) + β``.

    Mirrors Rust ``AdainResBlk::forward``'s norm1/norm2 pattern: the caller
    passes ``gamma_plus_1`` to Rust's [`adain`], so the effective transform is
    ``(1+γ)·norm(x) + β``. Here we fold that shift into this helper so the
    forward chain reads like the ADR.
    """
    import torch as _torch
    import torch.nn.functional as F

    gb = F.linear(style.unsqueeze(0), fc_w, fc_b).squeeze(0)  # [2·channels]
    gamma_raw = gb[:channels]
    beta = gb[channels:]
    # InstanceNorm across time per channel — biased var, eps=1e-5.
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    norm = (x - mean) / _torch.sqrt(var + PROSODY_ADALN_EPS)
    return norm * (1.0 + gamma_raw).unsqueeze(-1) + beta.unsqueeze(-1)


def _adain_res_blk(
    store,
    x: "torch.Tensor",
    prefix: str,
    dim_in: int,
    dim_out: int,
    upsample: bool,
    style: "torch.Tensor",
) -> "torch.Tensor":
    """One StyleTTS 2 AdainResBlk1d — mirrors Rust ``AdainResBlk::forward``.

    ``x`` is channel-major ``[dim_in, t_in]``; returns
    ``[dim_out, t_out]`` with ``t_out = 2·t_in`` when ``upsample`` else ``t_in``.
    ``(residual + shortcut) / sqrt(2)`` at the tail.
    """
    import torch as _torch
    import torch.nn.functional as F

    def get(name):
        return store.tensor(name).to(dtype=_torch.float32)

    learned_sc = dim_in != dim_out

    # --- Shortcut path ---
    if upsample:
        sc = F.interpolate(x.unsqueeze(0), scale_factor=2, mode="nearest").squeeze(0)
    else:
        sc = x.clone()
    if learned_sc:
        wg = get(f"{prefix}.conv1x1.weight_g")
        wv = get(f"{prefix}.conv1x1.weight_v")
        w = _wn_reconstruct(wg, wv)  # [dim_out, dim_in, 1] — NO BIAS in manifest
        sc = F.conv1d(sc.unsqueeze(0), w, bias=None, stride=1, padding=0).squeeze(0)

    # --- Residual path ---
    fc_w = get(f"{prefix}.norm1.fc.weight")
    fc_b = get(f"{prefix}.norm1.fc.bias")
    r = _adain_channel_major(x, dim_in, fc_w, fc_b, style)
    r = F.leaky_relu(r, PROSODY_LRELU_SLOPE)
    if upsample:
        pg = get(f"{prefix}.pool.weight_g")
        pv = get(f"{prefix}.pool.weight_v")  # [dim_in, 1, 3]
        pw = _wn_reconstruct(pg, pv)
        pb = get(f"{prefix}.pool.bias")
        r = F.conv_transpose1d(
            r.unsqueeze(0), pw, pb,
            stride=2, padding=1, output_padding=1, groups=dim_in,
        ).squeeze(0)
    # conv1: [dim_in → dim_out, k=3, pad=1]
    wg = get(f"{prefix}.conv1.weight_g")
    wv = get(f"{prefix}.conv1.weight_v")
    w = _wn_reconstruct(wg, wv)
    b_conv1 = get(f"{prefix}.conv1.bias")
    r = F.conv1d(r.unsqueeze(0), w, bias=b_conv1, stride=1, padding=1).squeeze(0)
    # norm2
    fc_w = get(f"{prefix}.norm2.fc.weight")
    fc_b = get(f"{prefix}.norm2.fc.bias")
    r = _adain_channel_major(r, dim_out, fc_w, fc_b, style)
    r = F.leaky_relu(r, PROSODY_LRELU_SLOPE)
    # conv2: [dim_out → dim_out, k=3, pad=1]
    wg = get(f"{prefix}.conv2.weight_g")
    wv = get(f"{prefix}.conv2.weight_v")
    w = _wn_reconstruct(wg, wv)
    b_conv2 = get(f"{prefix}.conv2.bias")
    r = F.conv1d(r.unsqueeze(0), w, bias=b_conv2, stride=1, padding=1).squeeze(0)

    # (residual + shortcut) / sqrt(2)
    inv_sqrt2 = 1.0 / (2.0 ** 0.5)
    return (r + sc) * inv_sqrt2


def _run_prosody_branch(
    store,
    hidden_ch: "torch.Tensor",
    prefix: str,
    style: "torch.Tensor",
) -> "torch.Tensor":
    """One F0/N branch: 3× AdainResBlk → conv1x1 → squeeze → ``[2·T_frames]``."""
    import torch as _torch
    import torch.nn.functional as F

    d = PROSODY_D_MODEL
    h = PROSODY_HALF
    # Block 0: (d, d, no upsample)
    x = _adain_res_blk(store, hidden_ch, f"{prefix}.0", d, d, False, style)
    # Block 1: (d, half, upsample=True)
    x = _adain_res_blk(store, x, f"{prefix}.1", d, h, True, style)
    # Block 2: (half, half, no upsample)
    x = _adain_res_blk(store, x, f"{prefix}.2", h, h, False, style)
    # Projection Conv1d(half → 1, k=1) — NOT weight-normed.
    proj_w = store.tensor(f"{prefix}_proj.weight").to(dtype=_torch.float32)  # [1, half, 1]
    proj_b = store.tensor(f"{prefix}_proj.bias").to(dtype=_torch.float32)
    out = F.conv1d(x.unsqueeze(0), proj_w, bias=proj_b).squeeze(0).squeeze(0)  # [2·T_frames]
    return out


def _forward_prosody(
    store: _ShapeMap,
    prosody_input: np.ndarray,
    style: np.ndarray,
) -> tuple:
    """PyTorch re-implementation of Rust ``ProsodyPredictor::forward_upstream``.

    Pipeline (mirrors ``crates/vokra-models/src/kokoro/prosody.rs``):

    1. ``prosody_input [T, 512]`` row-major ⊕ style ``[128]`` → ``[T, 640]``.
    2. 3× ( BiLSTM(640 → 512) → AdaLayerNorm(x, style, (1+γ)·norm(x)+β) →
       concat style → ``[T, 640]`` ).
    3. Main LSTM(640 → 512) → ``[T, 512]``.
    4. duration_proj Linear(512 → 50) → ``sigmoid.sum.round.clamp(1, 1024)``
       per phoneme → ``[T]``.
    5. Length regulate ``[T, 640] → [T_frames, 640]`` (repeat each phoneme
       ``durations[j]`` times).
    6. Shared LSTM(640 → 512) → ``[T_frames, 512]`` row-major.
    7. Transpose to ``[512, T_frames]`` channel-major.
    8. F0 branch: 3× AdainResBlk → 1×1 Conv1d → ``[2·T_frames]``.
    9. N branch: same.

    Returns tuple ``(durations, f0, n, hidden, t_frames)``:
    * ``durations`` np.int64 ``[T]``
    * ``f0`` np.float32 ``[2·T_frames]``
    * ``n`` np.float32 ``[2·T_frames]``
    * ``hidden`` np.float32 ``[d_model, T_frames]`` channel-major (matches
      Rust's ``ProsodyOutput.hidden`` layout).
    * ``t_frames`` int
    """
    import torch as _torch
    import torch.nn.functional as F

    if prosody_input.shape[1] != PROSODY_D_MODEL:
        raise RuntimeError(
            f"prosody_input width {prosody_input.shape[1]} != PROSODY_D_MODEL "
            f"({PROSODY_D_MODEL}); Kokoro-82M is fixed at hidden_dim = 512."
        )
    if style.shape[0] != PROSODY_STYLE_DIM:
        raise RuntimeError(
            f"style width {style.shape[0]} != PROSODY_STYLE_DIM ({PROSODY_STYLE_DIM})"
        )
    T = prosody_input.shape[0]
    d = PROSODY_D_MODEL
    sd = PROSODY_STYLE_DIM
    d_te_in = d + sd  # 640
    lstm_h = PROSODY_LSTM_HIDDEN  # 256

    x = _torch.from_numpy(np.ascontiguousarray(prosody_input.astype(np.float32)))  # [T, 512]
    style_t = _torch.from_numpy(style.astype(np.float32))  # [128]

    def concat_style_row(z: "_torch.Tensor") -> "_torch.Tensor":
        s = style_t.unsqueeze(0).expand(z.shape[0], -1)
        return _torch.cat([z, s], dim=-1)

    x_cat = concat_style_row(x)  # [T, 640]

    # --- Duration-encoder stack: 3× BiLSTM + AdaLN + concat style ---
    for i in range(3):
        bilstm_idx = 2 * i
        adaln_idx = 2 * i + 1
        y = _forward_bilstm(
            store,
            f"predictor.module.text_encoder.lstms.{bilstm_idx}",
            x_cat, d_te_in, lstm_h,
        )
        # AdaLN LayerNorm-across-channels + (1+γ)·x + β.
        fc_w = store.tensor(f"predictor.module.text_encoder.lstms.{adaln_idx}.fc.weight").to(dtype=_torch.float32)
        fc_b = store.tensor(f"predictor.module.text_encoder.lstms.{adaln_idx}.fc.bias").to(dtype=_torch.float32)
        norm_out = _adaln_layernorm_1d(y, d, fc_w, fc_b, style_t)
        x_cat = concat_style_row(norm_out)  # [T, 640]

    d_features = x_cat  # [T, 640]

    # --- Main LSTM (640 → 512) ---
    main_out = _forward_bilstm(store, "predictor.module.lstm", d_features, d_te_in, lstm_h)  # [T, 512]

    # --- Duration projection ---
    dur_w = store.tensor("predictor.module.duration_proj.linear_layer.weight").to(dtype=_torch.float32)  # [50, 512]
    dur_b = store.tensor("predictor.module.duration_proj.linear_layer.bias").to(dtype=_torch.float32)    # [50]
    dur_row = F.linear(main_out, dur_w, dur_b)  # [T, 50]
    dur_sigmoid = _torch.sigmoid(dur_row)  # [T, 50]
    sum_dur = dur_sigmoid.sum(dim=-1)  # [T]
    # Round + clamp to [1, 1024]. Non-finite ⇒ 1 (matches Rust's guard).
    finite = _torch.isfinite(sum_dur)
    rounded = _torch.round(sum_dur)
    durations = _torch.where(
        finite,
        _torch.clamp(rounded, min=1, max=1024),
        _torch.ones_like(sum_dur),
    ).to(dtype=_torch.int64)
    t_frames = int(durations.sum().item())

    if t_frames == 0:
        return (
            durations.numpy().astype(np.int64),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros((d, 0), dtype=np.float32),
            0,
        )

    # --- Length regulation: [T, 640] → [T_frames, 640] ---
    d_features_rep = d_features.repeat_interleave(durations, dim=0)  # [T_frames, 640]

    # --- Frame-rate shared BiLSTM (640 → 512) ---
    shared_out = _forward_bilstm(store, "predictor.module.shared", d_features_rep, d_te_in, lstm_h)  # [T_frames, 512]

    # --- Transpose to channel-major [512, T_frames] ---
    hidden_ch = shared_out.transpose(0, 1).contiguous()  # [512, T_frames]

    # --- F0 / N branches ---
    f0 = _run_prosody_branch(store, hidden_ch, "predictor.module.F0", style_t)
    n = _run_prosody_branch(store, hidden_ch, "predictor.module.N", style_t)

    return (
        durations.numpy().astype(np.int64),
        f0.detach().cpu().numpy().astype(np.float32),
        n.detach().cpu().numpy().astype(np.float32),
        hidden_ch.detach().cpu().numpy().astype(np.float32),
        t_frames,
    )


# ---------------------------------------------------------------------------
# T15 decoder re-forward (M2-07 task)
# ---------------------------------------------------------------------------
#
# The Kokoro-82M iSTFTNet decoder pipeline (see
# ``crates/vokra-models/src/kokoro/decoder.rs`` + ``.../decoder/generator.rs``):
#
#   1. F0_conv / N_conv  (1 → 1, k=3, stride=2, pad=1)   f0[T_f] → f0_ds[T_ds]
#   2. Upsample_nearest  f0_ds/n_ds back to T_frames
#   3. Encode input      concat([asr, f0_up, n_up]) → [514, T_frames]
#   4. encode            AdainResBlock1(514 → 1024)
#   5. asr_res           WeightNormedConv1d(512 → 64, k=1) on asr
#   6. decode.0/1/2      AdainResBlock1(1090 → 1024) at T_x = T_frames
#      decode.3          AdainResBlock1(1090 → 512, has_pool=True) → T_x' = 2·T_x − 1
#   7. Generator         2-stage upsample [k=20 s=10, k=12 s=6] w/ MRF (3
#                        kernels [3,7,11]), Snake activation on each resblock;
#                        conv_post → mag/phase at [11, T_gen] where
#                        T_gen = T_x' · 10 · 6.
#   8. iSTFT             Hann/periodic, hop=5, win=20, backward norm, no
#                        center, real_input; length = T_gen · 5.
#
# Assumptions being tested (M2-07 T15 flags):
#
#  * Snake formula = BigVGAN AMP `x + sin²(αx)/(α + eps)` (eps=1e-9)
#  * LeakyReLU slope inside decode blocks = 0.2 (StyleTTS 2 default)
#  * LeakyReLU slope inside generator = 0.1 (HiFi-GAN default)
#  * F0_conv stride=2, pad=1
#  * Generator resblocks = 3 kernels [3, 7, 11] per stage
#  * Noise source = zero-fill (mirrors Rust's T15 simplification — NOT a
#    true SineGen; the parity test therefore validates the ZERO-source math
#    path, and pcm parity does not verify the SineGen assumption).
#
# The reference intentionally mirrors the Rust simplifications so a byte-parity
# pass validates the decoder math (weight_norm reconstruction, AdaIN, Snake,
# LeakyReLU, ConvTranspose, MRF averaging, iSTFT) rather than architectural
# assumptions. The SineGen assumption remains flagged "not-truly-verified" in
# the report — a future WP would replace the zero-source with a real SineGen
# and re-parity.

DECODE_LRELU_SLOPE = 0.2  # StyleTTS 2 default (distinct from generator's 0.1)
GEN_LRELU_SLOPE = 0.1  # HiFi-GAN default
SNAKE_EPS = 1e-9  # Snake α+eps stability (matches Rust nn::snake_activation)


def _maybe_dump_stage(name: str, tensor) -> None:
    """Bisection helper (T17-fixup #1): write `tensor` as little-endian F32 to
    ``$VOKRA_KOKORO_PARITY_DUMP/ref_<name>.f32`` when that env is set.

    Pairs 1:1 with Rust's ``super::maybe_dump_stage("<name>", ...)`` so
    ``tools/parity/kokoro_bisect.py`` can diff each stage in isolation.
    """
    import os

    dump_dir = os.environ.get("VOKRA_KOKORO_PARITY_DUMP")
    if not dump_dir:
        return
    os.makedirs(dump_dir, exist_ok=True)
    try:
        arr = tensor.detach().cpu().numpy().astype(np.float32).reshape(-1)
    except AttributeError:
        arr = np.asarray(tensor, dtype=np.float32).reshape(-1)
    path = os.path.join(dump_dir, f"ref_{name}.f32")
    with open(path, "wb") as f:
        f.write(arr.tobytes())


def _adain_channel_major_no_shift(
    x: "torch.Tensor",
    channels: int,
    fc_w: "torch.Tensor",
    fc_b: "torch.Tensor",
    style: "torch.Tensor",
) -> "torch.Tensor":
    """Channel-major ``[channels, time]`` AdaIN with ``γ·norm(x) + β``
    (NO ``+1`` shift). Mirrors Rust ``nn::adain_conditioned`` used in the
    decoder's ``AdaIN1d::apply`` and the generator's ``AmpResBlock::forward``
    — the decoder / generator convention differs from the prosody predictor's
    (which passes ``gamma_plus_1`` to Rust's ``adain``, hence the shifted
    ``_adain_channel_major`` helper above).
    """
    import torch as _torch
    import torch.nn.functional as F

    gb = F.linear(style.unsqueeze(0), fc_w, fc_b).squeeze(0)  # [2·channels]
    gamma_raw = gb[:channels]
    beta = gb[channels:]
    # InstanceNorm across time per channel — biased var, eps=1e-5.
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    norm = (x - mean) / _torch.sqrt(var + PROSODY_ADALN_EPS)
    return norm * gamma_raw.unsqueeze(-1) + beta.unsqueeze(-1)


def _dec_adain_res_blk(
    store,
    x: "torch.Tensor",
    prefix: str,
    dim_in: int,
    dim_out: int,
    has_pool: bool,
    style_t: "torch.Tensor",
) -> "torch.Tensor":
    """StyleTTS 2 ``AdainResBlk1`` — mirrors Rust ``AdainResBlock1::forward``
    (channel-major layout, ``DECODE_LRELU_SLOPE = 0.2``, pool without
    ``output_padding`` so ``t_out = 2·t_in − 1``).

    Residual: ``norm1(x, s) → LeakyReLU(0.2) → pool(x)? → conv1 → norm2 → LeakyReLU → conv2``
    Shortcut: ``pool(x)? → conv1x1(x)``  (conv1x1 always present since ``dim_in != dim_out``)
    Out    : ``(residual + shortcut) / sqrt(2)``
    """
    import torch as _torch
    import torch.nn.functional as F

    def get(name):
        return store.tensor(name).to(dtype=_torch.float32)

    # --- Residual path -----------------------------------------------------
    # Decoder AdaIN uses ``γ·norm(x) + β`` (no +1 shift) — Rust's
    # ``adain_conditioned`` convention. Prosody's ``+1`` shift lives in
    # ``_adain_channel_major`` above.
    fc_w = get(f"{prefix}.norm1.fc.weight")
    fc_b = get(f"{prefix}.norm1.fc.bias")
    r = _adain_channel_major_no_shift(x, dim_in, fc_w, fc_b, style_t)
    r = F.leaky_relu(r, DECODE_LRELU_SLOPE)
    if has_pool:
        pg = get(f"{prefix}.pool.weight_g")
        pv = get(f"{prefix}.pool.weight_v")  # [dim_in, 1, 3] depthwise
        pb = get(f"{prefix}.pool.bias")
        pw = _wn_reconstruct(pg, pv)
        # Depthwise ConvTranspose1d, output_padding=0 → t_out = 2·t_in − 1
        r = F.conv_transpose1d(
            r.unsqueeze(0), pw, pb,
            stride=2, padding=1, output_padding=0, groups=dim_in,
        ).squeeze(0)
    # conv1: [dim_in → dim_out, k=3, pad=1]
    wg = get(f"{prefix}.conv1.weight_g")
    wv = get(f"{prefix}.conv1.weight_v")
    b_conv1 = get(f"{prefix}.conv1.bias")
    w = _wn_reconstruct(wg, wv)
    r = F.conv1d(r.unsqueeze(0), w, bias=b_conv1, stride=1, padding=1).squeeze(0)
    # norm2 + LeakyReLU (no +1 shift, matching decoder convention)
    fc_w = get(f"{prefix}.norm2.fc.weight")
    fc_b = get(f"{prefix}.norm2.fc.bias")
    r = _adain_channel_major_no_shift(r, dim_out, fc_w, fc_b, style_t)
    r = F.leaky_relu(r, DECODE_LRELU_SLOPE)
    # conv2: [dim_out → dim_out, k=3, pad=1]
    wg = get(f"{prefix}.conv2.weight_g")
    wv = get(f"{prefix}.conv2.weight_v")
    b_conv2 = get(f"{prefix}.conv2.bias")
    w = _wn_reconstruct(wg, wv)
    r = F.conv1d(r.unsqueeze(0), w, bias=b_conv2, stride=1, padding=1).squeeze(0)

    # --- Shortcut path (conv1x1 always present here) ----------------------
    if has_pool:
        pg2 = get(f"{prefix}.pool.weight_g")
        pv2 = get(f"{prefix}.pool.weight_v")
        pb2 = get(f"{prefix}.pool.bias")
        pw2 = _wn_reconstruct(pg2, pv2)
        sc = F.conv_transpose1d(
            x.unsqueeze(0), pw2, pb2,
            stride=2, padding=1, output_padding=0, groups=dim_in,
        ).squeeze(0)
    else:
        sc = x.clone()
    wg = get(f"{prefix}.conv1x1.weight_g")
    wv = get(f"{prefix}.conv1x1.weight_v")
    w = _wn_reconstruct(wg, wv)  # [dim_out, dim_in, 1] — NO BIAS
    sc = F.conv1d(sc.unsqueeze(0), w, bias=None, stride=1, padding=0).squeeze(0)

    # (r + sc) / sqrt(2) — StyleTTS 2 normalisation
    inv_sqrt2 = 1.0 / (2.0 ** 0.5)
    return (r + sc) * inv_sqrt2


def _snake_activation_torch(
    x: "torch.Tensor",
    alpha: "torch.Tensor",
    channels: int,
    time: int,
) -> "torch.Tensor":
    """BigVGAN AMP Snake activation ``x + sin²(αx) / (α + eps)`` — mirrors Rust
    ``nn::snake_activation`` (per-channel α, eps = 1e-9 for numerical stability,
    α = 0 degrades to identity)."""
    import torch as _torch

    # alpha shape [1, channels, 1] → flatten to [channels]
    a = alpha.reshape(-1)
    assert a.shape[0] == channels, f"snake: alpha channels {a.shape[0]} != {channels}"
    assert x.shape == (channels, time), f"snake: x shape {tuple(x.shape)} != ({channels}, {time})"
    inv_a = 1.0 / (a + SNAKE_EPS)
    ax = a.unsqueeze(-1) * x  # [channels, time]
    s = _torch.sin(ax)
    return x + inv_a.unsqueeze(-1) * s * s


def _amp_resblock_forward(
    store,
    x: "torch.Tensor",
    prefix: str,
    channels: int,
    kernel: int,
    style_t: "torch.Tensor",
    dump_prefix: str = "",
) -> "torch.Tensor":
    """HiFi-GAN AMP ResBlock (BigVGAN) — mirrors Rust ``AmpResBlock::forward``.

    3 dilated Conv1d pairs (dilations [1, 3, 5]) with per-sub-block AdaIN + Snake;
    additive residual over 3 sub-blocks.

    When ``dump_prefix`` is non-empty AND ``VOKRA_KOKORO_PARITY_DUMP`` is set,
    emits per-sub-block intermediates via ``_maybe_dump_stage`` under
    ``ref_<dump_prefix>_sub_<j>_{adain1,snake1,c1,adain2,snake2,c2,acc}``.
    """
    import torch as _torch
    import torch.nn.functional as F

    def get(name):
        return store.tensor(name).to(dtype=_torch.float32)

    dilations = [1, 3, 5]
    time = x.shape[1]
    for j in range(3):
        d = dilations[j]
        pad1 = (kernel - 1) * d // 2
        pad2 = (kernel - 1) // 2

        xj = x.clone()
        # adain1 + snake α1 + convs1[j] (dilated) — no-shift AdaIN (decoder convention)
        fc_w = get(f"{prefix}.adain1.{j}.fc.weight")
        fc_b = get(f"{prefix}.adain1.{j}.fc.bias")
        xj = _adain_channel_major_no_shift(xj, channels, fc_w, fc_b, style_t)
        if dump_prefix:
            _maybe_dump_stage(f"{dump_prefix}_sub_{j}_adain1", xj)
        alpha1 = get(f"{prefix}.alpha1.{j}")
        xj = _snake_activation_torch(xj, alpha1, channels, time)
        if dump_prefix:
            _maybe_dump_stage(f"{dump_prefix}_sub_{j}_snake1", xj)
        wg = get(f"{prefix}.convs1.{j}.weight_g")
        wv = get(f"{prefix}.convs1.{j}.weight_v")
        b = get(f"{prefix}.convs1.{j}.bias")
        w = _wn_reconstruct(wg, wv)
        xj = F.conv1d(
            xj.unsqueeze(0), w, bias=b,
            stride=1, padding=pad1, dilation=d,
        ).squeeze(0)
        if dump_prefix:
            _maybe_dump_stage(f"{dump_prefix}_sub_{j}_c1", xj)

        # adain2 + snake α2 + convs2[j] (dilation 1) — no-shift AdaIN
        fc_w = get(f"{prefix}.adain2.{j}.fc.weight")
        fc_b = get(f"{prefix}.adain2.{j}.fc.bias")
        xj = _adain_channel_major_no_shift(xj, channels, fc_w, fc_b, style_t)
        if dump_prefix:
            _maybe_dump_stage(f"{dump_prefix}_sub_{j}_adain2", xj)
        alpha2 = get(f"{prefix}.alpha2.{j}")
        xj = _snake_activation_torch(xj, alpha2, channels, time)
        if dump_prefix:
            _maybe_dump_stage(f"{dump_prefix}_sub_{j}_snake2", xj)
        wg = get(f"{prefix}.convs2.{j}.weight_g")
        wv = get(f"{prefix}.convs2.{j}.weight_v")
        b = get(f"{prefix}.convs2.{j}.bias")
        w = _wn_reconstruct(wg, wv)
        xj = F.conv1d(
            xj.unsqueeze(0), w, bias=b,
            stride=1, padding=pad2, dilation=1,
        ).squeeze(0)
        if dump_prefix:
            _maybe_dump_stage(f"{dump_prefix}_sub_{j}_c2", xj)

        x = x + xj
        if dump_prefix:
            _maybe_dump_stage(f"{dump_prefix}_sub_{j}_acc", x)

    return x


def _nearest_align_torch(x: "torch.Tensor", t_in: int, t_out: int) -> "torch.Tensor":
    """Nearest-neighbor align ``[channels, t_in]`` → ``[channels, t_out]``.
    Mirrors Rust ``upsample_nearest`` / ``nearest_align`` bit-for-bit
    (integer division: ``src = (t · t_in) // t_out``, clamped to ``t_in − 1``).
    """
    import torch as _torch

    if t_in == t_out:
        return x.clone()
    channels = x.shape[0]
    if t_in == 0:
        return _torch.zeros(channels, t_out, dtype=x.dtype)
    out = _torch.zeros(channels, t_out, dtype=x.dtype)
    for t in range(t_out):
        src = min((t * t_in) // t_out, t_in - 1)
        out[:, t] = x[:, src]
    return out


def _forward_generator_torch(
    store,
    x: "torch.Tensor",
    style_t: "torch.Tensor",
) -> tuple:
    """iSTFTNet generator forward — mirrors Rust ``Generator::forward``.

    Args:
        x: ``[in_ch=512, t]`` channel-major torch tensor.
        style_t: ``[style_dim=128]`` torch tensor.

    Returns:
        ``(x_mag, x_phase, t_gen)`` — each tensor is ``[n_half, t_gen]``.

    Uses zero-fill for the source_spec (mirrors Rust's T15 simplification;
    SineGen is not implemented — see module note).
    """
    import torch as _torch
    import torch.nn.functional as F

    g = "decoder.module.generator"

    def get(name):
        return store.tensor(name).to(dtype=_torch.float32)

    # Discover ups stages by probing ups.{i}.weight_v.
    stages = []
    cur_in = x.shape[0]
    for i in range(8):  # safety cap; real Kokoro has 2 stages
        try:
            wv = store.tensor(f"{g}.ups.{i}.weight_v")
        except KeyError:
            break
        wg = get(f"{g}.ups.{i}.weight_g")
        wv_f = wv.to(dtype=_torch.float32)
        b = get(f"{g}.ups.{i}.bias")
        # ConvTranspose1d weight: [in_ch, out_ch/groups, kernel]
        in_ch, out_ch, kernel = int(wv_f.shape[0]), int(wv_f.shape[1]), int(wv_f.shape[2])
        assert in_ch == cur_in, f"ups.{i}: in_ch mismatch {in_ch} vs {cur_in}"
        stride = kernel // 2
        assert 2 * stride == kernel, f"ups.{i}: kernel {kernel} must be 2·stride"
        pad = (kernel - stride) // 2
        stages.append((in_ch, out_ch, kernel, stride, pad, wg, wv_f, b))
        cur_in = out_ch
    assert stages, "generator: no ups stages found"
    n_stages = len(stages)
    gen_final = stages[-1][1]

    # conv_post: split gives (mag, phase) channels — derives n_half.
    cp_wg = get(f"{g}.conv_post.weight_g")
    cp_wv = get(f"{g}.conv_post.weight_v")  # [source_ch=2·n_half, gen_final, k=7]
    cp_b = get(f"{g}.conv_post.bias")
    source_ch = int(cp_wv.shape[0])
    assert source_ch % 2 == 0, f"conv_post out_ch {source_ch} must be even"
    n_half = source_ch // 2
    assert cp_wv.shape[1] == gen_final
    conv_post_kernel = int(cp_wv.shape[2])
    conv_post_pad = (conv_post_kernel - 1) // 2
    cp_w = _wn_reconstruct(cp_wg, cp_wv)

    cur = x.clone()
    for stage in range(n_stages):
        in_ch, out_ch, kernel, stride, pad, wg, wv_f, b = stages[stage]
        # 1. LeakyReLU(0.1) → ups
        cur = F.leaky_relu(cur, GEN_LRELU_SLOPE)
        _maybe_dump_stage(f"gen_stage_{stage}_pre_ups", cur)
        w = _wn_reconstruct(wg, wv_f)  # [in_ch, out_ch, kernel]
        up = F.conv_transpose1d(
            cur.unsqueeze(0), w, b,
            stride=stride, padding=pad, output_padding=0, groups=1,
        ).squeeze(0)  # [out_ch, t_up]
        t_up = up.shape[1]
        _maybe_dump_stage(f"gen_stage_{stage}_ups", up)

        # 2. Noise / source contribution — zero-fill (mirrors Rust T15).
        source_spec = _torch.zeros(source_ch, t_up, dtype=_torch.float32)
        nc_w = get(f"{g}.noise_convs.{stage}.weight")  # [out_ch, source_ch, k]
        nc_b = get(f"{g}.noise_convs.{stage}.bias")
        nc_kernel = int(nc_w.shape[2])
        # Rust's stride/pad from kernel — matches decoder/generator.rs::PlainConv1d
        if nc_kernel > 1:
            nc_stride = (nc_kernel + 1) // 2  # div_ceil(2)
            nc_pad = (nc_stride + 1) // 2  # div_ceil(2)
        else:
            nc_stride = 1
            nc_pad = 0
        noise_x = F.conv1d(
            source_spec.unsqueeze(0), nc_w, bias=nc_b,
            stride=nc_stride, padding=nc_pad,
        ).squeeze(0)
        t_noise = noise_x.shape[1]
        if t_noise != t_up:
            noise_x = _nearest_align_torch(noise_x, t_noise, t_up)
        _maybe_dump_stage(f"gen_stage_{stage}_noise_pre_res", noise_x)

        # noise_res[stage] AmpResBlock — probe kernel from convs1.0.weight_v
        nr_probe = store.tensor(f"{g}.noise_res.{stage}.convs1.0.weight_v")
        nr_kernel = int(nr_probe.shape[2])
        noise_x = _amp_resblock_forward(
            store, noise_x, f"{g}.noise_res.{stage}", out_ch, nr_kernel, style_t,
            dump_prefix=f"gen_stage_{stage}_noise_res",
        )
        _maybe_dump_stage(f"gen_stage_{stage}_noise_post_res", noise_x)

        # 3. Fuse
        fused = up + noise_x
        _maybe_dump_stage(f"gen_stage_{stage}_fused", fused)

        # 4. MRF: average over 3 resblocks per stage.
        n_kernels = 3
        mrf = _torch.zeros_like(fused)
        for k in range(n_kernels):
            rb_idx = stage * n_kernels + k
            rb_probe = store.tensor(f"{g}.resblocks.{rb_idx}.convs1.0.weight_v")
            rb_kernel = int(rb_probe.shape[2])
            branch = _amp_resblock_forward(
                store, fused, f"{g}.resblocks.{rb_idx}", out_ch, rb_kernel, style_t,
            )
            _maybe_dump_stage(f"gen_stage_{stage}_rb_{k}", branch)
            mrf = mrf + branch
        mrf = mrf / float(n_kernels)
        _maybe_dump_stage(f"gen_stage_{stage}_mrf", mrf)
        cur = mrf

    # 5. Head: LeakyReLU → conv_post → split (mag, phase).
    cur = F.leaky_relu(cur, GEN_LRELU_SLOPE)
    _maybe_dump_stage("gen_pre_conv_post", cur)
    post = F.conv1d(
        cur.unsqueeze(0), cp_w, bias=cp_b,
        stride=1, padding=conv_post_pad,
    ).squeeze(0)  # [source_ch, t_gen]
    _maybe_dump_stage("gen_conv_post", post)
    t_gen = int(post.shape[1])
    x_mag = post[:n_half].contiguous()
    x_phase = post[n_half:].contiguous()
    return x_mag, x_phase, t_gen


def _forward_decoder(
    store: _ShapeMap,
    encoder_features_ch: np.ndarray,
    durations: np.ndarray,
    style: np.ndarray,
    istft_n_fft: int,
    istft_hop: int,
    istft_win_length: int,
) -> tuple:
    """PyTorch re-implementation of Rust ``Decoder::forward_full``.

    Args:
        encoder_features_ch: ``[hidden=512, t_in]`` channel-major np.ndarray
            (post-bert / post-text_encoder features that the mainline
            [`crate::kokoro::mod::synthesize_phonemes`] length-regulates before
            calling the decoder).
        durations: ``[t_in]`` np.int64 array — the prosody predictor's
            duration output.
        style: ``[128]`` np.ndarray voice style vector.
        istft_n_fft / istft_hop / istft_win_length: Kokoro-82M iSTFT
            hyper-parameters (20 / 5 / 20 in the upstream release).

    Returns:
        ``(pre_istft_mag, pre_istft_phase, pcm)``:

        * ``pre_istft_mag``: np.ndarray ``[n_half, t_gen]`` channel-major
        * ``pre_istft_phase``: np.ndarray ``[n_half, t_gen]`` channel-major
        * ``pcm``: np.ndarray ``[t_gen · hop]`` audio at 24 kHz

    Note on f0 / n contours: Rust's mainline ``Decoder::forward`` (called by
    ``synthesize_phonemes``) feeds ZERO f0 / n contours to the decoder — the
    prosody predictor's F0 / N branch outputs are currently unused by the
    mainline. This reference matches that behaviour so a parity pass here
    validates the decoder math (weight_norm reconstruction, AdaIN, Snake,
    ConvTranspose, MRF averaging, iSTFT) rather than the yet-unwired F0
    handling. A future WP would wire the prosody F0 / N contours through
    ``Decoder::forward_full`` and re-parity.
    """
    import torch as _torch
    import torch.nn.functional as F

    if encoder_features_ch.ndim != 2:
        raise RuntimeError(
            f"decoder: encoder_features_ch must be 2D [hidden, t_in], got shape {encoder_features_ch.shape}"
        )
    hidden, t_in = encoder_features_ch.shape
    if int(durations.sum()) == 0:
        raise RuntimeError("decoder: sum(durations) = 0 — no frames to synthesise")
    if hidden != 512:
        raise RuntimeError(
            f"decoder: encoder_features_ch hidden {hidden} != 512 (Kokoro-82M is fixed)"
        )
    t_frames = int(durations.sum())

    # 1. length_regulate: [hidden, t_in] → [hidden, t_frames]
    z = np.zeros((hidden, t_frames), dtype=np.float32)
    tf = 0
    for j in range(t_in):
        reps = int(durations[j])
        for _ in range(reps):
            z[:, tf] = encoder_features_ch[:, j]
            tf += 1
    asr = _torch.from_numpy(z).contiguous()  # [512, t_frames]
    style_t = _torch.from_numpy(style.astype(np.float32))

    def get(name):
        return store.tensor(name).to(dtype=_torch.float32)

    # 2. Feed zero f0 / n at t_frames (mirrors Rust mainline path).
    f0 = _torch.zeros(t_frames, dtype=_torch.float32)
    n_energy = _torch.zeros(t_frames, dtype=_torch.float32)

    # 3. F0_conv / N_conv (1 → 1, k=3, stride=2, pad=1) → t_ds = ceil(t_frames/2)
    f0_conv_wg = get("decoder.module.F0_conv.weight_g")
    f0_conv_wv = get("decoder.module.F0_conv.weight_v")
    f0_conv_b = get("decoder.module.F0_conv.bias")
    f0_conv_w = _wn_reconstruct(f0_conv_wg, f0_conv_wv)
    f0_ds = F.conv1d(
        f0.reshape(1, 1, -1), f0_conv_w, bias=f0_conv_b,
        stride=2, padding=1,
    ).reshape(-1)
    t_ds = f0_ds.shape[0]

    n_conv_wg = get("decoder.module.N_conv.weight_g")
    n_conv_wv = get("decoder.module.N_conv.weight_v")
    n_conv_b = get("decoder.module.N_conv.bias")
    n_conv_w = _wn_reconstruct(n_conv_wg, n_conv_wv)
    n_ds = F.conv1d(
        n_energy.reshape(1, 1, -1), n_conv_w, bias=n_conv_b,
        stride=2, padding=1,
    ).reshape(-1)

    # 4. Upsample_nearest f0_ds/n_ds back to t_frames — matches Rust's
    #    ``upsample_nearest``.
    def upsample_nearest_1d(v: "_torch.Tensor", t_in_: int, t_out_: int) -> "_torch.Tensor":
        if t_in_ == t_out_:
            return v.clone()
        if t_in_ == 0:
            return _torch.zeros(t_out_, dtype=v.dtype)
        out_ = _torch.zeros(t_out_, dtype=v.dtype)
        for t in range(t_out_):
            src = min((t * t_in_) // t_out_, t_in_ - 1)
            out_[t] = v[src]
        return out_

    f0_up = upsample_nearest_1d(f0_ds, t_ds, t_frames)  # [t_frames]
    n_up = upsample_nearest_1d(n_ds, t_ds, t_frames)

    # 5. Build encode_input = concat [asr | f0_up | n_up] → [514, t_frames]
    encode_in = _torch.cat([
        asr,
        f0_up.unsqueeze(0),
        n_up.unsqueeze(0),
    ], dim=0)

    # 6. encode block: AdainResBlock1(514 → 1024, no pool)
    encode_in_ch = 512 + 2  # asr_dim + F0 + N = 514
    decode_hidden = 1024
    x = _dec_adain_res_blk(
        store, encode_in, "decoder.module.encode",
        dim_in=encode_in_ch, dim_out=decode_hidden,
        has_pool=False, style_t=style_t,
    )  # [1024, t_frames]
    _maybe_dump_stage("dec_encode", x)

    # 7. asr_res bridge: WeightNormedConv1d(512 → 64, k=1) on the raw asr.
    asr_res_wg = get("decoder.module.asr_res.0.weight_g")
    asr_res_wv = get("decoder.module.asr_res.0.weight_v")
    asr_res_b = get("decoder.module.asr_res.0.bias")
    asr_res_w = _wn_reconstruct(asr_res_wg, asr_res_wv)
    asr_res_out = F.conv1d(
        asr.unsqueeze(0), asr_res_w, bias=asr_res_b,
        stride=1, padding=0,
    ).squeeze(0)  # [64, t_frames]
    _maybe_dump_stage("dec_asr_res", asr_res_out)
    asr_res_out_ch = 64

    # 8. decode.0/1/2/3: concat [x, asr_res, f0_up, n_up] → [1090, t_x]
    concat_in = decode_hidden + asr_res_out_ch + 2  # 1090
    t_x = t_frames

    # Cached per-frame contours as [C, T] channel-major.
    asr_res_ch = asr_res_out  # [64, t_frames]
    f0_up_ch = f0_up.unsqueeze(0)  # [1, t_frames]
    n_up_ch = n_up.unsqueeze(0)

    def interp_ch(v: "_torch.Tensor", t_target: int) -> "_torch.Tensor":
        t_cur = v.shape[1]
        if t_cur == t_target:
            return v.clone()
        return _nearest_align_torch(v, t_cur, t_target)

    for i in range(4):
        has_pool = (i == 3)
        dim_out_i = decode_hidden if i < 3 else 512  # decode.3 → decode_final_out=512

        # Interpolate asr_res / f0 / n to current t_x (identity for i=0..2)
        asr_res_i = interp_ch(asr_res_ch, t_x)
        f0_i = interp_ch(f0_up_ch, t_x)
        n_i = interp_ch(n_up_ch, t_x)

        concat = _torch.cat([x, asr_res_i, f0_i, n_i], dim=0)
        assert concat.shape == (concat_in, t_x), (
            f"decoder decode.{i}: concat shape {tuple(concat.shape)} != ({concat_in}, {t_x})"
        )

        x = _dec_adain_res_blk(
            store, concat, f"decoder.module.decode.{i}",
            dim_in=concat_in, dim_out=dim_out_i,
            has_pool=has_pool, style_t=style_t,
        )
        t_x = x.shape[1]
        _maybe_dump_stage(f"dec_decode_{i}", x)

    # x is now [512, t_x] where t_x = 2·t_frames − 1 (decode.3 pool applied).
    _maybe_dump_stage("dec_pre_generator", x)

    # 9. Generator forward → (x_mag, x_phase, t_gen)
    x_mag, x_phase, t_gen = _forward_generator_torch(store, x, style_t)

    # 10. iSTFT: Hann/periodic window, backward norm, center=False, length = t_gen · hop.
    n_half = istft_n_fft // 2 + 1
    assert x_mag.shape == (n_half, t_gen)
    assert x_phase.shape == (n_half, t_gen)

    # Rust's iSTFT head lowering:
    #   mag   = exp(x_mag)
    #   phase = PhaseActivation::Sin(x_phase) · π   ← Kokoro's dispatched activation
    #   re[f, k]  = mag[k, f] · cos(phase[k, f])
    #   im[f, k]  = mag[k, f] · sin(phase[k, f])
    # (see decoder.rs::run_istft_head).
    mag = _torch.exp(x_mag)
    phase_pi = _torch.sin(x_phase) * float(np.pi)
    re = mag * _torch.cos(phase_pi)  # [n_half, t_gen]
    im = mag * _torch.sin(phase_pi)

    # torch.istft's strict NOLA check fires at the boundaries with hop=5,
    # win=20, center=False (samples n < win_length may have window-sum-of-
    # squares == 0 at position 0). Rust's ``vokra_ops::istft`` leaves those
    # samples as 0 via the NOLA_EPS = 1e-8 guard (see istft.rs). We mirror
    # that behaviour with a manual overlap-add so byte-parity holds at frame
    # boundaries too.
    pcm = _manual_istft(
        re.detach().cpu().numpy().astype(np.float32),
        im.detach().cpu().numpy().astype(np.float32),
        n_fft=istft_n_fft,
        hop=istft_hop,
        win_length=istft_win_length,
        t_gen=t_gen,
        length=t_gen * istft_hop,
    )

    return (
        x_mag.detach().cpu().numpy().astype(np.float32),
        x_phase.detach().cpu().numpy().astype(np.float32),
        pcm,
    )


# --- Manual iSTFT (mirrors Rust ``vokra_ops::istft``) ----------------------


def _manual_istft(
    re: np.ndarray,
    im: np.ndarray,
    n_fft: int,
    hop: int,
    win_length: int,
    t_gen: int,
    length: int,
) -> np.ndarray:
    """NumPy port of Rust ``vokra_ops::istft`` for the Kokoro settings.

    Reproduces the exact overlap-add / NOLA-guard semantics of Rust's
    ``crates/vokra-ops/src/istft.rs`` so byte-parity holds at frame boundaries
    (torch.istft's strict NOLA check would fail here for hop=5 / win=20 /
    center=False, since the length-20 periodic Hann's first sample is 0).

    Args:
        re / im: ``[n_half, t_gen]`` channel-major real / imag bins (numpy).
        n_fft: FFT size (real inverse takes ``n_fft/2+1`` bins).
        hop / win_length: iSTFT parameters. ``win_length ≤ n_fft`` — the
            Rust helper ``build_synth_window`` centers the Hann inside a
            length-``n_fft`` zero-padded window when they differ; here we
            use ``win_length == n_fft`` (Kokoro's setting) so the synth
            window is Hann-of-length-``n_fft`` directly.
        t_gen: number of iSTFT frames (spectrogram time axis width).
        length: target output length (Kokoro passes ``t_gen · hop``).

    Returns:
        ``[length]`` float32 audio (NOLA-violating samples stay 0).
    """
    NOLA_EPS = 1e-8
    # Hann/periodic window matching Rust's ``Window::Hann`` + ``WindowSymmetry::Periodic``:
    # ``w[n] = 0.5 · (1 − cos(2π · n / win_length))`` for n in [0, win_length).
    w = 0.5 - 0.5 * np.cos(
        2.0 * np.pi * np.arange(win_length, dtype=np.float32) / float(win_length)
    )
    if win_length == n_fft:
        synth_window = w.astype(np.float32)
    else:
        # Rust ``build_synth_window`` centers the analysis window inside a
        # length-n_fft zero pad.
        synth_window = np.zeros(n_fft, dtype=np.float32)
        offset = (n_fft - win_length) // 2
        synth_window[offset:offset + win_length] = w

    if t_gen == 0:
        return np.zeros(length, dtype=np.float32)

    total = (t_gen - 1) * hop + n_fft
    acc = np.zeros(total, dtype=np.float32)
    wss = np.zeros(total, dtype=np.float32)

    # `unscale = 1.0` for Backward normalization (see Rust ``frame_unscale``).
    # np.fft.irfft with default norm='backward' includes the 1/n factor.
    for f in range(t_gen):
        spec_bin = re[:, f].astype(np.complex64) + 1j * im[:, f].astype(np.complex64)
        # `norm='backward'` = 1/n on inverse (matches Rust's RealFftPlan::inverse).
        frame_time = np.fft.irfft(spec_bin, n=n_fft).astype(np.float32)
        start = f * hop
        acc[start:start + n_fft] += frame_time * synth_window
        wss[start:start + n_fft] += synth_window * synth_window

    mask = wss > NOLA_EPS
    acc[mask] /= wss[mask]
    # NOLA-violating samples remain 0 (matches Rust).

    # center=False → no trim; honor the target length by truncation / zero-pad.
    if length <= len(acc):
        return acc[:length].astype(np.float32)
    padded = np.zeros(length, dtype=np.float32)
    padded[:len(acc)] = acc
    return padded


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
        "--mode",
        choices=("placeholder", "full"),
        default="placeholder",
        help=(
            "placeholder: seed-derived shape-correct tensors, Rust harness "
            "runs shape/length checks only. full: PyTorch re-implementation of "
            "the Rust forward for every module that has landed; Rust harness "
            "runs byte-level parity vs the reference."
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
    store = open_checkpoint(local)
    checkpoint_name = store.description
    dims = derive_dims(store)

    # Vocab size: derived from the text embedding when present (Kokoro-82M is
    # 178). Falls back to 256 only when the tensor is absent (shape-only
    # ShapeMap without the expected key).
    text_emb = dims.get("text_embedding_shape")
    vocab_size = int(text_emb[0]) if text_emb is not None else 256

    # Voice id: 0 unless explicitly named; the manifest carries the name
    # for downstream cross-checking (a follow-up ticket may plumb through
    # voice_names[] from the config for a name→id lookup).
    voice_id = 0

    # Style: shape from voicepack; else from config.json; else 128 (Kokoro's
    # canonical style dim). Kokoro's upstream .pth stores voice styles as
    # separate voices/*.pt files, so the in-model voicepack is often absent —
    # in that case the config.json is authoritative.
    voicepack = dims.get("voicepack_shape")
    if voicepack is not None:
        style_dim = int(voicepack[-1])
    elif config:
        style_dim = int(config.get("style_dim", 128))
    else:
        style_dim = 128

    hidden_dim = int(text_emb[1]) if text_emb is not None else 512

    phoneme_ids = synth_phoneme_ids(vocab_size)
    style = synth_style(style_dim)

    # ---- Placeholder baseline (always computed; overridden by mode=full) ----
    text_encoder, prosody, mel_pre_istft, pcm = placeholder_forward(dims)
    mel_channels = int(mel_pre_istft.shape[1])

    # ---- Per-module mode markers (updated below by mode=full path) ----
    module_modes = {
        "text_encoder_mode": "placeholder",
        "bert_mode": "placeholder",
        "prosody_mode": "placeholder",
        "decoder_mode": "placeholder",
    }
    bert_out: np.ndarray | None = None
    # Decoder T15 outputs — initialised at outer scope so the manifest block
    # can reference them regardless of whether mode=full computed them.
    decoder_mag: np.ndarray | None = None
    decoder_phase: np.ndarray | None = None
    decoder_pcm: np.ndarray | None = None
    decoder_t_gen = 0
    decoder_n_half = 0
    # Prosody T14 outputs — same rationale.
    prosody_durations = None
    prosody_f0 = None
    prosody_n = None
    prosody_hidden = None
    prosody_t_frames = 0

    if args.mode == "full":
        # We need the .pth backend to run a full forward (the safetensors path
        # only exposes shapes). Fail loudly rather than silently downgrading.
        if store._tensor_fn is None:
            sys.exit(
                "mode=full requires a torch pickle checkpoint (kokoro-v1_0.pth); "
                f"got shape-only backend {checkpoint_name!r}. Re-download the "
                "hexgrad/Kokoro-82M repo (it ships the .pth by default)."
            )

        # --- text_encoder ---
        try:
            te_full = _forward_text_encoder(store, phoneme_ids)  # [T, hidden]
            # Dump the first ENC_POS positions (matches the manifest's enc_pos).
            enc = te_full[:ENC_POS]
            if enc.shape != (ENC_POS, hidden_dim):
                raise RuntimeError(
                    f"text_encoder output shape {enc.shape} != expected "
                    f"({ENC_POS}, {hidden_dim})"
                )
            text_encoder = enc.astype(np.float32)
            module_modes["text_encoder_mode"] = "full"
            print(
                f"  text_encoder: full forward OK, shape {enc.shape} "
                f"(first {ENC_POS} of {te_full.shape[0]} tokens)"
            )
        except Exception as exc:
            print(f"  text_encoder: mode=full FAILED, keeping placeholder ({exc})")

        # --- bert ---
        try:
            bert_full = _forward_bert(store, phoneme_ids)  # [T, 512]
            bert_slice = bert_full[:ENC_POS]
            if bert_slice.shape != (ENC_POS, BERT_OUT_DIM):
                raise RuntimeError(
                    f"bert output shape {bert_slice.shape} != expected "
                    f"({ENC_POS}, {BERT_OUT_DIM})"
                )
            bert_out = bert_slice.astype(np.float32)
            module_modes["bert_mode"] = "full"
            print(
                f"  bert:         full forward OK, shape {bert_slice.shape} "
                f"(first {ENC_POS} of {bert_full.shape[0]} tokens)"
            )
        except Exception as exc:
            print(f"  bert:         mode=full FAILED, keeping placeholder ({exc})")

        # --- prosody ---
        # Upstream prosody predictor consumes the bert (or text-encoder) [T, 512]
        # features + a style vector, runs 3× BiLSTM + AdaLN + main LSTM + duration
        # projection + length regulation to [T_frames, ...] + shared BiLSTM + F0/N
        # AdainResBlk stacks. See ``_forward_prosody`` for the layer-by-layer
        # port of ``crates/vokra-models/src/kokoro/prosody.rs::forward_upstream``.
        # (prosody_durations / prosody_f0 / prosody_n / prosody_hidden /
        # prosody_t_frames initialised at the outer scope so the manifest
        # block can reference them regardless of whether the try below succeeds.)
        # Feed the full (non-truncated) bert output if available; else fall back
        # to the full text_encoder output. Rust's ``synthesize_phonemes`` uses
        # ``bert_out`` when the canary tensor is present (real Kokoro-82M
        # carries it), otherwise the text-encoder features.
        try:
            if bert_out is not None:
                # bert_out is the truncated [ENC_POS, 512] slice; re-run for the
                # full [T, 512] input the prosody consumes.
                prosody_input = _forward_bert(store, phoneme_ids)
            else:
                prosody_input = _forward_text_encoder(store, phoneme_ids)
            if prosody_input.shape[1] != hidden_dim:
                raise RuntimeError(
                    f"prosody_input width {prosody_input.shape[1]} != hidden_dim ({hidden_dim})"
                )
            (
                prosody_durations,
                prosody_f0,
                prosody_n,
                prosody_hidden,
                prosody_t_frames,
            ) = _forward_prosody(store, prosody_input, style)
            module_modes["prosody_mode"] = "full"
            print(
                f"  prosody:      full forward OK, T={prosody_input.shape[0]} phonemes → "
                f"T_frames={prosody_t_frames}, "
                f"durations={list(prosody_durations)}, "
                f"f0 shape=({prosody_f0.shape[0]},), n shape=({prosody_n.shape[0]},), "
                f"hidden shape={prosody_hidden.shape}"
            )
        except Exception as exc:
            print(f"  prosody:      mode=full FAILED, keeping placeholder ({exc})")
            import traceback
            traceback.print_exc()

        # --- decoder ---
        # Upstream iSTFTNet decoder is 375 tensors (concat + AdaLN ResBlocks +
        # HiFi-GAN generator with Snake activation + iSTFT head). This branch
        # runs the M2-07-T15 re-forward that mirrors Rust's
        # ``Decoder::forward_full`` (see ``_forward_decoder`` above).
        # (decoder_mag / decoder_phase / decoder_pcm / decoder_t_gen /
        # decoder_n_half initialised at the outer scope.)
        try:
            if module_modes["prosody_mode"] != "full":
                raise RuntimeError(
                    "prosody_mode != full — decoder re-forward requires "
                    "the prosody predictor's duration output for length regulation"
                )
            assert prosody_durations is not None
            # Encoder features → channel-major [hidden, T_in]. Prefer bert
            # output (matches Rust's ``synthesize_phonemes`` mainline); fall
            # back to text-encoder output otherwise. Both are [T, 512] row-major.
            if bert_out is not None:
                enc_full = _forward_bert(store, phoneme_ids)  # [T, 512]
            else:
                enc_full = _forward_text_encoder(store, phoneme_ids)  # [T, 512]
            enc_ch = np.ascontiguousarray(enc_full.T)  # [hidden=512, T]

            (
                decoder_mag,
                decoder_phase,
                decoder_pcm,
            ) = _forward_decoder(
                store,
                enc_ch,
                prosody_durations,
                style,
                istft_n_fft=20,   # Kokoro-82M constants
                istft_hop=5,
                istft_win_length=20,
            )
            decoder_n_half = int(decoder_mag.shape[0])
            decoder_t_gen = int(decoder_mag.shape[1])
            module_modes["decoder_mode"] = "full"
            print(
                f"  decoder:      full forward OK, t_gen = {decoder_t_gen} frames "
                f"(n_half = {decoder_n_half}), pcm = {decoder_pcm.shape[0]} samples"
            )
        except Exception as exc:  # noqa: BLE001 — dumper-side surface diagnostic
            print(f"  decoder:      mode=full FAILED, keeping placeholder ({exc})")
            import traceback
            traceback.print_exc()

    # ---- Dump binaries ----
    write_i64(out_dir / "phoneme_ids.i64", phoneme_ids)
    write_f32(out_dir / "style.f32", style)
    write_f32(out_dir / "text_encoder.f32", text_encoder)
    write_f32(out_dir / "prosody.f32", prosody)
    write_f32(out_dir / "mel_pre_istft.f32", mel_pre_istft)
    write_f32(out_dir / "pcm.f32", pcm)
    if bert_out is not None:
        write_f32(out_dir / "bert.f32", bert_out)
    else:
        # In placeholder mode the bert.f32 fixture is not written; the Rust
        # parity harness checks bert_mode before reading the file.
        bert_path = out_dir / "bert.f32"
        if bert_path.exists():
            bert_path.unlink()

    # Prosody T14 fixtures (only when mode=full computed them). Rust parity
    # harness gates on ``prosody_mode = full``; when absent the fixtures are
    # cleared so a stale byte-level parity claim isn't accidentally kept.
    prosody_files = (
        "prosody_durations.i64",
        "prosody_f0.f32",
        "prosody_n.f32",
        "prosody_hidden.f32",
    )
    if module_modes["prosody_mode"] == "full":
        assert prosody_durations is not None
        assert prosody_f0 is not None
        assert prosody_n is not None
        assert prosody_hidden is not None
        write_i64(out_dir / "prosody_durations.i64", prosody_durations)
        write_f32(out_dir / "prosody_f0.f32", prosody_f0)
        write_f32(out_dir / "prosody_n.f32", prosody_n)
        # prosody_hidden shape is [d_model, T_frames] channel-major (matches Rust).
        write_f32(out_dir / "prosody_hidden.f32", prosody_hidden)
    else:
        for name in prosody_files:
            p = out_dir / name
            if p.exists():
                p.unlink()

    # T15 decoder fixtures (only when mode=full computed them). Rust parity
    # harness gates on ``decoder_mode = full`` before reading; absent leaves
    # the placeholder `mel_pre_istft.f32` and `pcm.f32` from `placeholder_forward`.
    decoder_files = (
        "decoder_pre_istft_mag.f32",
        "decoder_pre_istft_phase.f32",
        "decoder_pcm.f32",
    )
    if module_modes["decoder_mode"] == "full":
        assert decoder_mag is not None
        assert decoder_phase is not None
        assert decoder_pcm is not None
        write_f32(out_dir / "decoder_pre_istft_mag.f32", decoder_mag)
        write_f32(out_dir / "decoder_pre_istft_phase.f32", decoder_phase)
        write_f32(out_dir / "decoder_pcm.f32", decoder_pcm)
    else:
        for name in decoder_files:
            p = out_dir / name
            if p.exists():
                p.unlink()

    # ---- Manifest ----
    # Global mode is "full" iff at least one module has a full NumPy re-forward;
    # per-module modes tell the Rust harness which byte-check to enable.
    global_mode = "full" if any(v == "full" for v in module_modes.values()) else "placeholder"

    manifest = {
        "model": model_id,
        "checkpoint_file": checkpoint_name,
        "seed": SEED,
        "atol": 0.01,
        "mode": global_mode,
        **module_modes,
        "vocab_size": vocab_size,
        "hidden_dim": text_encoder.shape[1],
        "style_dim": style_dim,
        "voice_id": voice_id,
        "voice_name": args.voice or "",
        "num_phonemes": NUM_PHONEMES,
        "enc_pos": ENC_POS,
        "dec_frames": DEC_FRAMES,
        "prosody_channels": PROSODY_CHANNELS,
        "mel_channels": mel_channels,
        "bert_out_dim": BERT_OUT_DIM if bert_out is not None else 0,
        "pcm_len": PCM_LEN,
        "sample_rate": int(config.get("sample_rate", 24_000)) if config else 24_000,
        # Prosody T14 fixture dimensions. Zero when prosody_mode = placeholder;
        # the Rust harness gates on ``prosody_mode == "full"`` before reading.
        "prosody_t_frames": prosody_t_frames,
        "prosody_d_model": PROSODY_D_MODEL if module_modes["prosody_mode"] == "full" else 0,
        "prosody_f0_len": (
            int(prosody_f0.shape[0])
            if module_modes["prosody_mode"] == "full" and prosody_f0 is not None
            else 0
        ),
        "prosody_n_len": (
            int(prosody_n.shape[0])
            if module_modes["prosody_mode"] == "full" and prosody_n is not None
            else 0
        ),
        # Decoder T15 fixture dimensions. Zero when decoder_mode = placeholder;
        # the Rust harness gates on ``decoder_mode == "full"`` before reading.
        # `decoder_pcm_len` may differ from `pcm_len` (the placeholder key) because
        # the real decoder's pcm length depends on prosody durations, generator
        # upsampling factors, and the iSTFT hop.
        "decoder_n_half": decoder_n_half if module_modes["decoder_mode"] == "full" else 0,
        "decoder_t_gen": decoder_t_gen if module_modes["decoder_mode"] == "full" else 0,
        "decoder_pcm_len": (
            int(decoder_pcm.shape[0])
            if module_modes["decoder_mode"] == "full" and decoder_pcm is not None
            else 0
        ),
        # Kokoro-82M iSTFT hyper-parameters — dumped so the Rust harness can
        # cross-check against the loaded GGUF (converter M2-07-T06 writes
        # `vokra.kokoro.istft.n_fft` = 20 / `.hop` = 5 / `.win_length` = 20).
        "istft_n_fft": 20 if module_modes["decoder_mode"] == "full" else 0,
        "istft_hop": 5 if module_modes["decoder_mode"] == "full" else 0,
        "istft_win_length": 20 if module_modes["decoder_mode"] == "full" else 0,
    }
    with (out_dir / "manifest.txt").open("w", encoding="utf-8") as f:
        f.write("# Kokoro-82M parity manifest (M2-07). Generated by\n")
        f.write("# tools/parity/dump_kokoro_reference.py. `key = value`;\n")
        f.write("# list values are space-separated.\n")
        f.write("# mode = placeholder: the Rust parity harness runs shape /\n")
        f.write("# length checks only (byte-level parity is a follow-up).\n")
        f.write("# mode = full: at least one module has a real NumPy re-forward.\n")
        f.write("# Per-module gates (`<module>_mode`) select byte-parity per module.\n")
        for k, v in manifest.items():
            if isinstance(v, list):
                f.write(f"{k} = {' '.join(str(x) for x in v)}\n")
            else:
                f.write(f"{k} = {v}\n")

    print(f"wrote fixtures to {out_dir}")
    print(
        f"  mode={manifest['mode']} "
        f"text_encoder_mode={manifest['text_encoder_mode']} "
        f"bert_mode={manifest['bert_mode']} "
        f"prosody_mode={manifest['prosody_mode']} "
        f"decoder_mode={manifest['decoder_mode']}"
    )
    print(
        f"  vocab={vocab_size} hidden_dim={manifest['hidden_dim']} "
        f"style_dim={style_dim}"
    )
    print(
        f"  num_phonemes={NUM_PHONEMES} enc_pos={ENC_POS} "
        f"dec_frames={DEC_FRAMES} pcm_len={PCM_LEN}"
    )
    if module_modes["decoder_mode"] == "full":
        print(
            f"  decoder: n_half={manifest['decoder_n_half']} "
            f"t_gen={manifest['decoder_t_gen']} "
            f"decoder_pcm_len={manifest['decoder_pcm_len']} "
            f"istft={manifest['istft_n_fft']}/{manifest['istft_hop']}/{manifest['istft_win_length']}"
        )


if __name__ == "__main__":
    main()
