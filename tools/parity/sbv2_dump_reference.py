#!/usr/bin/env python3
"""SBV2 v2 clean-room reference dumper — HF ``transformers`` DeBERTa
encoders + a vendored ``jaywalnut310/vits`` VITS core (Task 30, SBV2 v2
plan).

Runs the Style-Bert-VITS2 v2 forward pass using **only permissive Python
reference implementations** and dumps the 11 JA/EN intermediate tensors,
plus a twelfth independent ZH BERT tensor for a ZH run,
``docs/superpowers/specs/2026-07-26-sbv2-v2-design.md`` §10 pins, so
``crates/vokra-models/tests/parity_sbv2_real.rs`` (Task 28) can diff a real
Rust forward pass against them, tensor by tensor, at
``sbv2::parity::tolerance_for(name)``.

# NOT REFERENCED (clean-room — read this before touching this file)

- ``github.com/litagin02/Style-Bert-VITS2`` (AGPL-3.0) and all forks
- ``github.com/fishaudio/Bert-VITS2`` (AGPL-3.0) and all forks
- Any community fork/derivative/blog-post code excerpt of either of the
  above (Qiita/Zenn writeups, Discord snippets, ...)

This script and its vendored ``tools/parity/vendor/vits/`` dependencies
are built exclusively from the permissive sources listed next. No AGPL
source was read, copied, or recalled from memory to write this file.

# Permissive references this file is authorized to use

- **VITS** paper (Kim et al. 2021, arXiv:2106.06103) and
  ``jaywalnut310/vits`` (MIT) — vendored piecemeal into
  ``tools/parity/vendor/vits/`` (see that directory's own README for the
  pinned commit and exact module mapping).
- **VITS2** paper (arXiv:2307.16430) and ``p0p4k/vits2_pytorch`` (MIT) —
  architectural delta reference only, for the SBV2-specific structural
  differences from vanilla VITS.
- **DeBERTa v2** paper (arXiv:2006.03654), ``microsoft/DeBERTa`` (MIT), and
  **HuggingFace ``transformers``** (Apache-2.0) ``deberta_v2``/``deberta_v3``
  modules — used directly via ``AutoModel``/``AutoTokenizer`` below, no
  vendoring needed (an Apache-2.0 dependency managed through ``uv``).
  ``deberta_v3``'s own paper is arXiv:2111.09543.
- SentencePiece paper (Kudo & Richardson 2018) — informational only (the
  BERT tokenizers below come from ``transformers`` directly).
  ``litagin02/style_bert_vits2``'s upstream ``config.json`` / safetensors
  tensor-name metadata (structural facts only — names, shapes, dtypes —
  never the AGPL Python code itself, matching
  ``tools/parity/sbv2_prepare_checkpoint.py``'s own posture).

# Status: end-to-end wired (Wave-4 TASK-34-DUMPER, 2026-08-09)

This dumper has two modes:

* **Schema preview** (default, no ``--do-dump``): builds and prints the
  ``reference_dump.manifest.json`` this tool *would* write, with the 8
  request fields fully resolved from the CLI (nothing about those needs a
  real checkpoint or forward pass), and the 11 or 12 ``tensors[].shape`` entries
  left as symbolic placeholders (``"T_text"`` / ``"T_bert"`` / ``"T_mel"`` /
  ``"samples"``) since real dimensions only exist after a real forward pass
  runs. Nothing is written to ``--output-dir`` in this mode — printing a
  manifest that *looks* real but is not would itself be a fabricated
  artifact (NFR-QL-04 / FR-EX-08). Needs no dependency beyond the stdlib,
  so it (like ``--help``) works in a bare interpreter.
* **Real dump** (``--do-dump``): runs the design-doc §7 forward pipeline
  end-to-end (G2P -> ``SbV2TextEncoder`` -> DeBERTa-bridge -> SDP -> flow
  -> HiFi-GAN) and writes 11 ``reference_dump/*.bin`` files (12 for ZH) + 4 fixture
  side files (phoneme_ids/tones/word_boundaries/language_id) + N
  per-layer acoustic-flow intermediate files
  (``flow_layer_<i>_output.bin``, Wave 3 / Task 15 — ``N`` is the real
  checkpoint's acoustic-flow depth discovered at runtime, typically 4;
  see ``run_flow``'s docstring for the execution-order indexing
  convention) + a fully-resolved ``reference_dump.manifest.json`` to
  ``--output-dir``.
  Files are raw little-endian f32, matching every other ``*_dump*.py``
  sibling's ``arr.tobytes()`` convention (*not* ``numpy.save``'s ``.npy``
  format, which ``parity_sbv2_real.rs``'s ``read_f32_bin`` does not
  parse). Fails loudly at one of these tiers:

  1. ``torch`` missing -> run ``uv sync`` in ``tools/parity`` on the
     VAST instance, then rerun through ``uv run python``.
  2. ``transformers`` missing -> the same ``uv`` environment setup
     (Apache-2.0 — this project's authorized DeBERTa reference).
  3. Vendor import failure (``from vendor.vits import text_encoder / coupling
     / flow / decoder`` **or** ``from vendor.vits2 import attentions /
     models``) -> actionable message pointing at
     ``tools/parity/vendor/vits/README.md`` (Task 4, jaywalnut310/vits)
     or ``tools/parity/vendor/vits2/README.md`` (Task 1-2,
     p0p4k/vits2_pytorch, Blocker 2b) respectively, and their
     sha256/upstream-URL trail. Task 4 vendored 8 supporting modules,
     Task 1-2 vendored 2 (``attentions.py`` / ``models.py``) — this gate
     only fires on a corrupted vendor tree. (``vendor.vits2`` is a
     clean-room cross-reference only — see ``run_flow``'s docstring for
     why it does not build the real per-checkpoint flow.)
  4. Any missing state_dict tensor (`_load_tensor` FR-EX-08 message
     naming exact candidate keys tried) or unpopulated G2P row
     (`MinimalG2P` NotImplementedError). Nothing is stubbed or
     approximated — every tensor must be present or the run fails loud,
     the same NFR-QL-04 / FR-EX-08 lesson
     ``tools/parity/utmos_dump_reference.py``'s own module doc draws
     from the Kokoro ``92dbc92`` incident.

# The 11 base tensors plus the opt-in ZH tensor

======================= ================= ========================
name                    shape             purpose
======================= ================= ========================
``phoneme_embed``       [T_text, 192]     text encoder input
``text_hidden``         [T_text, 192]     text encoder output
``bert_hidden_ja``      [T_bert, 1024]    DeBERTa v2 output (JA)
``bert_hidden_en``      [T_bert, 1024]    DeBERTa v3 output (EN)
``bert_hidden_zh``      [T_bert, 1024]    plain BERT output (ZH run only)
``bert_bridge_out``     [T_text, 192]     BERT bridge conv output
``speaker_embed``       [1, 512]          speaker embedding
``style_projected``     [1, 192]          style vector projection
``sdp_sample``          [T_text]          SDP duration sample
``mel_hidden``          [T_mel, 192]      length-regulated hidden
``z_latent``            [T_mel, 192]      normalizing-flow output
``waveform``            [1, samples]      final PCM
======================= ================= ========================

Only ``T_bert`` differs per encoder. JA and EN dumps always run their real
DeBERTa encoders; an explicit ZH run additionally loads the real upstream
WordPiece + plain-BERT encoder and routes ``bert_hidden_zh`` into the bridge.
All tensor ``.bin`` files are raw
little-endian ``float32`` (including ``sdp_sample``, whose semantic values
are discrete durations — see
``crates/vokra-models/src/sbv2/parity.rs``'s ``PER_TENSOR_ATOL`` doc comment
for why that tensor is still compared as float with an atol that has an
explicit +/-1 discrete-step allowance, rather than as an exact integer
match).

Task 7 (SBV2 v2 plan) adds four **side files** alongside the tensor
list — ``phoneme_ids.bin`` (``uint16``), ``tones.bin`` (``uint8``),
``word_boundaries.bin`` (``uint8``), all of length ``T_text``, plus
``language_id.bin`` (``uint8`` scalar, count 1) — under the manifest's own
``phonemize_fixture`` block (not inside ``tensors[]``, so the design-doc
§10 tensor contract). They are the G2P *inputs*
to the reference forward pass, replayed on the Rust side by
``SbV2Phonemizer::from_fixture`` +
``SbV2Model::from_gguf_with_phonemizer`` — see this file's manifest
schema below.

**Real-checkpoint tensor-layout finding (M6, 2026-08-06)**:
``enc_p.word_boundary_emb.weight`` does NOT exist in the SBV2 v2 base
checkpoint (``litagin/Style-Bert-VITS2-2.0-base-JP-Extra``); the tensor
that exists at that slot is ``enc_p.language_emb.weight`` with shape
``[3, 192]`` (three per-utterance language rows: JA / EN / ZH). The
dumper now performs a per-utterance ``language_embed[language_id]``
broadcast-add into every position (matching
``crates/vokra-models/src/sbv2/text_encoder.rs`` ``SbV2TextEncoder::forward``
post-``b1e8f16``) instead of the pre-checkpoint estimate's per-position
``word_boundary_emb[word_boundaries[t]]`` lookup. ``word_boundaries.bin``
is still dumped (Rust-side ``PhonemizeResult`` retains the field as a
G2P output for fixture stability, even though the text encoder no longer
consumes it), and ``language_id.bin`` is added as the new fixture-side
input the reference forward pass consumes.

# Manifest schema (this file writes; ``parity_sbv2_real.rs``'s module doc is
# the authoritative contract this mirrors)

::

    {
      "generator_version": "1.1",
      "generator": "tools/parity/sbv2_dump_reference.py",
      "checkpoint": {
        "sbv2_main": "sbv2-v2-multilingual-base.gguf",
        "bert_ja": "deberta-v2-large-japanese-char-wwm.gguf",
        "bert_en": "deberta-v3-large.gguf",
        "bert_zh": "chinese-roberta-wwm-ext-large.gguf"  # ZH run only
      },
      "request": {
        "text": "...", "language": "JA", "speaker_id": 0,
        "style_vec": [0.0, "..."], "speed": 1.0,
        "noise_scale": 0.667, "noise_scale_w": 0.8, "seed": 42
      },
      "phonemize_fixture": {                                # Task 7 addition
        "phoneme_ids":     {"path": "reference_dump/phoneme_ids.bin",
                            "count": T_text, "dtype": "uint16"},
        "tones":           {"path": "reference_dump/tones.bin",
                            "count": T_text, "dtype": "uint8"},
        "word_boundaries": {"path": "reference_dump/word_boundaries.bin",
                            "count": T_text, "dtype": "uint8"},
        "language_id":     {"path": "reference_dump/language_id.bin",
                            "count": 1, "dtype": "uint8"}     # M6 addition
      },
      "tensors": [
        {"name": "phoneme_embed", "path": "reference_dump/phoneme_embed.bin",
         "shape": [T_text, 192], "dtype": "float32"},
        ... (11 for JA/EN, 12 for ZH; see table above)
      ],
      "flow_layers": {                                # Task 15 addition (Wave 3)
        "flow_layer_0_output": {
          "path": "reference_dump/flow_layer_0_output.bin",
          "shape": [T_mel, 192], "dtype": "float32", "atol": 0.01
        },
        ... (N total, N = the real checkpoint's acoustic-flow depth,
             typically 4 — see ``run_flow``'s docstring for the
             execution-order indexing convention and why the LAST entry
             is byte-identical to ``z_latent``. ``atol`` here is a
             human-auditable scaffold value, redundant with — not a
             substitute for — the authoritative Rust
             ``sbv2::parity::PER_TENSOR_ATOL`` constant Wave 3 / Task 17
             calibrates; nothing in the Rust harness parses this JSON
             field today.)
      }
    }

``phonemize_fixture`` (Task 7, extended M6) is the fixture bypass that
lets the Rust side rebuild an ``SbV2Phonemizer`` (via
``SbV2Phonemizer::from_fixture`` + ``SbV2Model::from_gguf_with_phonemizer``)
that reproduces the exact G2P output the reference dumper's forward pass
consumed, without needing a real 8-language piper-plus G2P available
in-workspace (NFR-DS-02: the excluded ``integrations/vokra-piper-g2p``
cannot be a ``crates/vokra-models`` dependency). The four side files:
the three per-position ones (``phoneme_ids`` / ``tones`` /
``word_boundaries``) are always 1-D of length ``T_text`` (matching every
f32 tensor whose leading axis is ``T_text``) and use narrower dtypes than
f32 — ``phoneme_ids`` is ``uint16`` (matches the Rust
``PhonemizeResult::phoneme_ids``'s ``Vec<u16>``), ``tones`` and
``word_boundaries`` are ``uint8``; the fourth (``language_id``, M6 addition)
is a ``uint8`` scalar (``count == 1``, matching the per-utterance
``u8`` ``SbV2TextEncoder::forward`` accepts). The consuming Rust reader
dispatches on ``dtype``.

``checkpoint.*`` are **bare filenames** (siblings of the manifest inside
``tests/fixtures/sbv2/``, matching Task 34's planned ``Files:`` list) —
override via ``--sbv2-main-filename``/``--bert-ja-filename``/
``--bert-en-filename``/``--bert-zh-filename`` if a real fixture set uses
different names. ``bert_zh`` is emitted only for a ZH request.
``request.language`` is upper-case ``"JA"``/``"EN"``/``"ZH"`` (``parity_sbv2_real.rs``
matches on exactly those three literals). ``tensors[].path`` already includes
the ``reference_dump/`` prefix.

# Usage

::

    # schema preview (no deps beyond the stdlib, nothing written to disk):
    uv run --no-project python tools/parity/sbv2_dump_reference.py \\
        --checkpoint /tmp/sbv2-checkpoint --output-dir /tmp/sbv2-dump

    # real dump (VAST only; requires staged weights and a populated G2P row):
    uv run --project tools/parity python tools/parity/sbv2_dump_reference.py \\
        --checkpoint /tmp/sbv2-checkpoint --output-dir /tmp/sbv2-dump \\
        --text "こんにちは。" --language ja --do-dump

``--checkpoint`` is the ``--output-dir`` a prior
``tools/parity/sbv2_prepare_checkpoint.py`` run produced (containing its
``vokra-sbv2-config.json`` side-car plus the downloaded ``.safetensors``).

# Dependencies

``torch`` (BSD) and ``transformers`` (Apache-2.0) are real, immediately
installable dependencies of the eventual real forward pass — both are
imported lazily (inside ``--do-dump``'s code path, after
``parser.parse_args()``) so ``--help`` and the schema-preview mode work in
an interpreter with neither installed, matching every other
``tools/parity/*_prepare_checkpoint.py``/``*_dump*.py`` sibling's deferred-
import convention. ``jaywalnut310/vits`` (MIT) has no PyPI distribution, so
the required modules are vendored under ``tools/parity/vendor/vits/`` (see
that directory's README). A real dump still fails loud unless its staged
checkpoint and exact G2P fixture row are present.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

# --- identity -------------------------------------------------------------

LOG_PREFIX = "[sbv2-dump]"
GENERATOR_ID = "tools/parity/sbv2_dump_reference.py"
GENERATOR_VERSION = "1.1"

# Matches Task 34's planned `Files:` list verbatim (design doc §10 / Task 28
# module doc) — bare filenames, siblings of the manifest inside
# tests/fixtures/sbv2/. Overridable per-run via --sbv2-main-filename etc. in
# case a real fixture set ends up named differently.
DEFAULT_SBV2_MAIN_FILENAME = "sbv2-v2-multilingual-base.gguf"
DEFAULT_BERT_JA_FILENAME = "deberta-v2-large-japanese-char-wwm.gguf"
DEFAULT_BERT_EN_FILENAME = "deberta-v3-large.gguf"
DEFAULT_BERT_ZH_FILENAME = "chinese-roberta-wwm-ext-large.gguf"

# HF transformers repo ids for the three BERT encoders. JA/EN were verified
# via the public HF Hub API on 2026-07-26. The ZH checkpoint was selected by
# the owner on 2026-08-09 and re-verified against the official API on
# 2026-08-18 (Apache-2.0, ungated, commit pinned by the workflow).
DEFAULT_BERT_JA_REPO = "ku-nlp/deberta-v2-large-japanese-char-wwm"
DEFAULT_BERT_EN_REPO = "microsoft/deberta-v3-large"
DEFAULT_BERT_ZH_REPO = "hfl/chinese-roberta-wwm-ext-large"

# design doc §7 "主要 hparams" table.
DEFAULT_STYLE_DIM = 256
DEFAULT_SPEED = 1.0
DEFAULT_NOISE_SCALE = 0.667
DEFAULT_NOISE_SCALE_W = 0.8
DEFAULT_SEED = 42

# --- SBV2 v2 architectural constants ---------------------------------------
#
# These are the design doc §7 "主要 hparams" pins used by the pipeline body
# below. Every one that is checkpoint-derivable (e.g. `n_vocab`,
# HiFi-GAN upsample structure) is resolved from the Task-3
# `vokra-sbv2-config.json` side-car at load time inside
# `load_sbv2_checkpoint()`; the ones below are the ones that either (a) are
# universal across the VITS/SBV2 family per public reference configs, or
# (b) the design doc pins directly. NO SBV2/BV2 AGPL source was consulted
# to obtain any of these numbers.
D_MODEL = 192  # text hidden channels — VITS default (jaywalnut310/vits configs/*.json, MIT)
D_BERT_JA = 1024  # DeBERTa v2 large hidden_size (HF ku-nlp/deberta-v2-large-japanese-char-wwm)
D_BERT_EN = 1024  # DeBERTa v3 large hidden_size (HF microsoft/deberta-v3-large)
D_SPEAKER = 256  # VITS multi-speaker `gin_channels` (jaywalnut310/vits configs/vctk_base.json, MIT)
                 # Overridden by Task-3 config `d_speaker` at load time — pins here are the
                 # SBV2 v2 spec value the design doc §10 `speaker_embed` [1, 512] anchors to
                 # (real checkpoint has d_speaker=512 — see `_resolve_arch_constants` below).
# HiFi-GAN vocoder hparams — resolved from Task-3 config at load time (see
# `_resolve_arch_constants` below), not baked here. `None` sentinel means
# "MUST be resolved before build_generator() is called".
UPSAMPLE_RATES: "tuple[int, ...] | None" = None
UPSAMPLE_KERNEL_SIZES: "tuple[int, ...] | None" = None
UPSAMPLE_INITIAL_CHANNEL: "int | None" = None
UPSAMPLE_OUT_CHANNELS: "tuple[int, ...] | None" = None
RESBLOCK_KERNEL_SIZES: "tuple[int, ...] | None" = None
RESBLOCK_DILATION_COUNTS: "tuple[int, ...] | None" = None
RESBLOCK_DILATIONS_FLAT: "tuple[int, ...] | None" = None
RESBLOCK_TYPE: str = "1"  # jaywalnut310/vits + jik876/hifi-gan v1 default (MIT)
CONV_PRE_KERNEL: int = 7  # jaywalnut310/vits Generator hard-codes kernel=7 (MIT)
CONV_POST_KERNEL: int = 7
LEAKY_RELU_SLOPE: float = 0.1
# Runtime-resolved scalars (populated by `_resolve_arch_constants`):
N_PHONEME_VOCAB: "int | None" = None
N_TONE_VOCAB: int = 6  # SBV2 JP-Extra tone alphabet: 0..4 pitch levels + silence
# `crates/vokra-models/src/sbv2/text_encoder.rs` `language_embed` is
# `[N_LANGUAGES, d_model]`, rows = JA/EN/ZH (real checkpoint
# `enc_p.language_emb.weight [3, 192]`). Post-b1e8f16 refactor: the earlier
# `N_WORD_BOUNDARY_VOCAB = 2` (assumed `enc_p.word_boundary_emb.weight`)
# did not survive the M6 real-checkpoint scout — that tensor does not exist
# in the SBV2 v2 base; `enc_p.language_emb.weight [3, d_model]` does. See
# this file's module docstring "Real-checkpoint tensor-layout finding"
# section for the full trail.
N_LANGUAGES: int = 3
D_STYLE_DEFAULT: int = DEFAULT_STYLE_DIM
SAMPLE_RATE: int = 44100  # SBV2 v2 JP-Extra target (litagin/Style-Bert-VITS2-2.0-base-JP-Extra HF README, public metadata)
# SDP flow depth — arXiv:2106.06103 §2.3 + jaywalnut310/vits SDP __init__ default (n_layers_dp=3, n_flows=4).
SDP_N_FLOWS = 4
SDP_KERNEL_SIZE = 3
SDP_DDS_N_LAYERS = 3  # jaywalnut310/vits (MIT) SDP.__init__ n_layers=3 default

# Short per-language default --text, only used when --text is not given.
DEFAULT_TEXT_BY_LANGUAGE = {
    "ja": "テスト",
    "en": "This is a test.",
    "zh": "你好。",
}

# The original 11-tensor dump contract plus the additive ZH BERT hidden
# tensor. `shape_template` uses the same symbolic placeholders as the Rust
# harness; real integers only appear once a real forward pass supplies them.
TENSOR_SCHEMA: "list[dict]" = [
    {"name": "phoneme_embed", "shape_template": ["T_text", 192]},
    {"name": "text_hidden", "shape_template": ["T_text", 192]},
    {"name": "bert_hidden_ja", "shape_template": ["T_bert", 1024]},
    {"name": "bert_hidden_en", "shape_template": ["T_bert", 1024]},
    {"name": "bert_hidden_zh", "shape_template": ["T_bert", 1024]},
    {"name": "bert_bridge_out", "shape_template": ["T_text", 192]},
    {"name": "speaker_embed", "shape_template": [1, 512]},
    {"name": "style_projected", "shape_template": [1, 192]},
    {"name": "sdp_sample", "shape_template": ["T_text"]},
    {"name": "mel_hidden", "shape_template": ["T_mel", 192]},
    {"name": "z_latent", "shape_template": ["T_mel", 192]},
    {"name": "waveform", "shape_template": [1, "samples"]},
]
# Every tensor this dumper writes is raw little-endian float32 — including
# sdp_sample, whose semantic values are discrete durations (see this file's
# module doc's tensor table note, and crates/vokra-models/src/sbv2/parity.rs
# PER_TENSOR_ATOL's own doc comment on why that tensor still gets an atol
# rather than an exact-integer comparison).
TENSOR_DTYPE = "float32"

# Task 7 (SBV2 v2 plan): phonemize-fixture side files. Sit ALONGSIDE the
# 11-tensor `tensors[]` list (not inside it) so this dumper keeps the
# design-doc §10 "11 dumped tensors" contract intact — the fixture files
# are *inputs* to the SBV2 pipeline (fed into the reference forward pass
# and reproduced verbatim by the Rust side via
# `SbV2Phonemizer::from_fixture` + `SbV2Model::from_gguf_with_phonemizer`),
# not intermediate outputs to numeric-diff against.
#
# Each entry names the raw-bytes file the real-dump path (once vendoring
# lands) will write for exactly the one `(request.language, request.text)`
# pair this manifest declares — `SbV2Phonemizer::phonemize` returns
# `phoneme_ids` (u16 LE), `tones` (u8), `word_boundaries` (u8 0/1), each
# of length `T_text` (symbolic in the schema-preview manifest, resolved to
# a real integer once a real forward pass runs).
#
# The `parity_sbv2_real.rs` Rust reader (Task 28) dispatches on `dtype`
# and reads with the corresponding element size — u16 LE via
# `u16::from_le_bytes` for `phoneme_ids`, plain `u8` for the three others.
#
# M6 addition (2026-08-06): `language_id` is the fourth side file — a
# per-utterance u8 scalar (`count == 1`, not `T_text`) matching the
# `language_id: u8` argument `SbV2TextEncoder::forward` accepts
# post-`b1e8f16`. Rows: `JA = 0`, `EN = 1`, `ZH = 2`
# (`crates/vokra-models/src/sbv2/g2p.rs` `Language::language_id`).
PHONEMIZE_FIXTURE_SCHEMA: "list[dict]" = [
    {"name": "phoneme_ids", "dtype": "uint16", "count_template": "T_text"},
    {"name": "tones", "dtype": "uint8", "count_template": "T_text"},
    {"name": "word_boundaries", "dtype": "uint8", "count_template": "T_text"},
    {"name": "language_id", "dtype": "uint8", "count_template": 1},  # M6: static count 1
]

# Task 15 (SBV2 v2 Blocker 2b/2c/3 plan, Wave 3) addition: per-layer
# acoustic-flow intermediates. These sit ALONGSIDE `tensors[]` and
# `phonemize_fixture` under their own manifest key `flow_layers` (a dict,
# like `phonemize_fixture` — NOT appended into `tensors[]`, which stays a
# plain list so the design-doc §10 "11 dumped tensors" contract stays
# intact, matching the same reasoning `PHONEMIZE_FIXTURE_SCHEMA`'s own
# comment above gives for its own side-file group).
#
# `FLOW_LAYER_COUNT_DEFAULT` is ONLY the schema-preview placeholder count
# (`run_preview`, no real checkpoint available) — the real dump path
# (`run_flow`, called from `run_pipeline_body`) always uses the actual
# count discovered from the loaded checkpoint's `flow.*` state_dict at
# runtime (`build_flow`'s own `n_flows_seen` walk), never this constant.
# `4` matches the VITS2 paper's (arXiv:2307.16430) and this project's own
# `SDP_N_FLOWS` sibling constant's conventional flow depth — a
# *coincidence* worth naming explicitly: `SDP_N_FLOWS` counts the
# Stochastic Duration Predictor's own internal flow (a different module),
# not this acoustic normalizing flow: the two happen to typically share
# the value 4 in this architecture family, they are not definitionally
# the same constant.
FLOW_LAYER_COUNT_DEFAULT = 4

# Task 17 (Wave 3) per-tensor atol calibration scaffold default — mirrors
# Rust `ATOL_DEFAULT` (`crates/vokra-models/src/sbv2/parity.rs`, `0.01`).
# See `build_manifest`'s `flow_layers` block below for why this JSON
# value is a human-auditable record, not itself an enforced bound.
ATOL_SCAFFOLD_DEFAULT = 0.01


def build_manifest(args: argparse.Namespace, tensor_shapes: "dict[str, list] | None" = None,
                   phonemize_counts: "dict[str, int] | None" = None,
                   flow_layer_count: "int | None" = None,
                   flow_layer_shape: "list | None" = None) -> dict:
    """Builds the ``reference_dump.manifest.json`` contents.

    ``tensor_shapes``, when given, maps a subset of the tensor names to
    their *real*, already-known integer shape (only available once a real
    forward pass has run). When ``None`` (schema-preview mode), every
    tensor falls back to its symbolic [`TENSOR_SCHEMA`] placeholder shape.

    ``phonemize_counts``, when given, maps a subset of the 4 Task-7
    fixture-side-file names (``phoneme_ids``/``tones``/``word_boundaries``
    + M6 addition ``language_id``) to their real element count — `T_text`
    for the per-position three, `1` for the per-utterance ``language_id``
    scalar (only available once a real G2P has run on ``args.text`` for
    the former, always `1` for the latter). When ``None``, each falls back
    to [`PHONEMIZE_FIXTURE_SCHEMA`]'s symbolic ``"T_text"`` (or the
    literal integer `1` for ``language_id``) placeholder.

    ``flow_layer_count`` / ``flow_layer_shape`` (Task 15, Wave 3) describe
    the per-layer acoustic-flow intermediates ``run_flow`` dumps —
    ``flow_layer_count`` real tensors, each ``flow_layer_shape``-shaped
    (always ``[T_mel, D_MODEL]``, an affine-coupling flow is channel- and
    time-preserving, so every layer's output shares the SAME shape as
    ``z_latent`` itself). Both fall back to the schema-preview placeholders
    [`FLOW_LAYER_COUNT_DEFAULT`] / ``["T_mel", D_MODEL]`` when ``None`` —
    the real dump path passes the checkpoint-derived count/shape
    ``run_flow`` actually produced (never hard-coded), matching the same
    real-vs-symbolic pattern the two params above already use.

    Everything else in the manifest (``checkpoint.*``, ``request.*``) is
    fully resolvable from ``args`` alone, real forward pass or not.
    """
    tensor_shapes = tensor_shapes or {}
    tensors = []
    for spec in TENSOR_SCHEMA:
        name = spec["name"]
        if name == "bert_hidden_zh" and args.language.lower() != "zh":
            continue
        shape = tensor_shapes.get(name, spec["shape_template"])
        tensors.append(
            {
                "name": name,
                "path": f"reference_dump/{name}.bin",
                "shape": shape,
                "dtype": TENSOR_DTYPE,
            }
        )

    phonemize_counts = phonemize_counts or {}
    phonemize_fixture = {}
    for spec in PHONEMIZE_FIXTURE_SCHEMA:
        name = spec["name"]
        count = phonemize_counts.get(name, spec["count_template"])
        phonemize_fixture[name] = {
            "path": f"reference_dump/{name}.bin",
            "count": count,
            "dtype": spec["dtype"],
        }

    # Task 15 (Wave 3): per-layer acoustic-flow intermediates. Sibling key
    # to `tensors`/`phonemize_fixture`, NOT appended into `tensors[]` — see
    # this module's `FLOW_LAYER_COUNT_DEFAULT` comment for why. `atol` is a
    # human-auditable Task 17 calibration scaffold (see that constant's
    # own doc comment for why it is not itself Rust's enforced bound).
    n_flow_layers = (
        flow_layer_count if flow_layer_count is not None else FLOW_LAYER_COUNT_DEFAULT
    )
    resolved_flow_layer_shape = (
        list(flow_layer_shape) if flow_layer_shape is not None else ["T_mel", D_MODEL]
    )
    flow_layers = {}
    for i in range(n_flow_layers):
        name = f"flow_layer_{i}_output"
        flow_layers[name] = {
            "path": f"reference_dump/{name}.bin",
            "shape": resolved_flow_layer_shape,
            "dtype": TENSOR_DTYPE,
            "atol": ATOL_SCAFFOLD_DEFAULT,
        }

    # Match `run_pipeline_body`'s effective_style_dim logic: honor
    # explicit --style-dim override, else use config-derived
    # D_STYLE_DEFAULT (populated by `_resolve_arch_constants` which is
    # called before this manifest builder in run_pipeline_body's flow).
    effective_style_dim = (
        D_STYLE_DEFAULT if args.style_dim == DEFAULT_STYLE_DIM else args.style_dim
    )
    style_vec = [0.0] * effective_style_dim

    checkpoint = {
        "sbv2_main": args.sbv2_main_filename,
        "bert_ja": args.bert_ja_filename,
        "bert_en": args.bert_en_filename,
    }
    if args.language.lower() == "zh":
        checkpoint["bert_zh"] = args.bert_zh_filename

    return {
        "generator_version": GENERATOR_VERSION,
        "generator": GENERATOR_ID,
        "checkpoint": checkpoint,
        "request": {
            "text": args.text,
            "language": args.language.upper(),
            "speaker_id": args.speaker_id,
            "style_vec": style_vec,
            "speed": args.speed,
            "noise_scale": args.noise_scale,
            "noise_scale_w": args.noise_scale_w,
            "seed": args.seed,
        },
        # Task 7: PhonemizeFixture side files (phoneme_ids/tones/word_boundaries)
        # — inputs to the SBV2 pipeline, not intermediates. See
        # `PHONEMIZE_FIXTURE_SCHEMA` above and `parity_sbv2_real.rs`'s Task 7
        # reader for the consuming shape.
        "phonemize_fixture": phonemize_fixture,
        "tensors": tensors,
        # Task 15 (Wave 3): per-layer acoustic-flow intermediates, dumped
        # by `run_flow` alongside (not instead of) the aggregate
        # `z_latent` tensor above. See `FLOW_LAYER_COUNT_DEFAULT`'s and
        # `run_flow`'s docstrings for the execution-order indexing
        # convention.
        "flow_layers": flow_layers,
    }


def run_preview(args: argparse.Namespace) -> int:
    """Schema-preview mode: prints the manifest this tool *would* write
    (symbolic tensor shapes, real everything-else) and touches no files.
    Needs no dependency beyond the stdlib — safe to run in any interpreter,
    same guarantee as ``--help``."""
    manifest = build_manifest(args, tensor_shapes=None)
    print(
        f"{LOG_PREFIX} schema preview (pass --do-dump to attempt a real dump). "
        f"tensors[].shape below are symbolic placeholders (T_text/T_bert/T_mel/"
        f"samples resolve only once a real forward pass runs); everything else "
        f"is the real value this run would use. Nothing is written to "
        f"{args.output_dir} in this mode."
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=False))
    return 0


# ==========================================================================
# Pipeline body helpers (design doc §7). Every function below is called
# only from `run_pipeline_body`, which itself is only called from
# `run_dump` after all 4 dependency tiers (torch / transformers /
# vendor.vits import / this body's own gate) have passed.
#
# Design invariants (FR-EX-08, NFR-QL-04):
#   * NO tensor value is ever fabricated. If a tensor is missing from the
#     state_dict under any of the candidate upstream names, the helper
#     raises with a message naming the exact tensor + candidate keys
#     tried — never silently substitutes zeros or a "reasonable default".
#   * The G2P table (`MinimalG2P`) starts empty. A `(language, text)` pair
#     absent from the table is a loud NotImplementedError — never a
#     heuristic fall-through. Owner populates rows for the exact fixtures
#     the parity test manifest exercises.
#   * SDP is a clean-room scratch composition of the VITS paper primitives
#     already present in `vendor/vits/modules.py` (DDSConv / ConvFlow /
#     Flip / ElementwiseAffine, all MIT). NO upstream `models.py` SDP
#     class body is read/copied — the composition below follows
#     arXiv:2106.06103 §2.3 topology directly.
# ==========================================================================


def _resolve_arch_constants(config: dict) -> None:
    """Populate the module-level HiFi-GAN + phoneme-vocab globals from
    the Task-3 `vokra-sbv2-config.json` side-car. Every missing REQUIRED
    field is a loud SystemExit (FR-EX-08) — never a silent default.

    Only fields the pipeline body actually consumes are validated here;
    `SbV2Config::parse` on the Rust side is the authoritative field-set
    check downstream (`vokra-cli convert --model sbv2`)."""
    global N_PHONEME_VOCAB, UPSAMPLE_RATES, UPSAMPLE_KERNEL_SIZES
    global UPSAMPLE_INITIAL_CHANNEL, UPSAMPLE_OUT_CHANNELS
    global RESBLOCK_KERNEL_SIZES, RESBLOCK_DILATION_COUNTS
    global RESBLOCK_DILATIONS_FLAT, LEAKY_RELU_SLOPE, SAMPLE_RATE
    global D_MODEL, D_SPEAKER, D_STYLE_DEFAULT, N_TONE_VOCAB

    def _require(key: str):
        if key not in config:
            sys.exit(
                f"{LOG_PREFIX} Task-3 config side-car is missing required "
                f"field {key!r}. Rerun tools/parity/sbv2_prepare_checkpoint.py "
                "with --clean-room-defaults (SBV2 v2 base ships weights-only), "
                "or fill the field in by hand."
            )
        return config[key]

    UPSAMPLE_RATES = tuple(_require("decoder_upsample_rates"))
    UPSAMPLE_KERNEL_SIZES = tuple(_require("decoder_upsample_kernel_sizes"))
    UPSAMPLE_INITIAL_CHANNEL = int(_require("decoder_initial_channel"))
    UPSAMPLE_OUT_CHANNELS = tuple(_require("decoder_upsample_out_channels"))
    RESBLOCK_KERNEL_SIZES = tuple(_require("decoder_resblock_kernel_sizes"))
    RESBLOCK_DILATION_COUNTS = tuple(_require("decoder_resblock_dilation_counts"))
    RESBLOCK_DILATIONS_FLAT = tuple(_require("decoder_resblock_dilations_flat"))
    LEAKY_RELU_SLOPE = float(config.get("decoder_leaky_relu_slope", 0.1))
    SAMPLE_RATE = int(_require("sample_rate"))
    D_MODEL = int(_require("d_model"))
    D_SPEAKER = int(_require("d_speaker"))
    D_STYLE_DEFAULT = int(_require("d_style"))
    N_PHONEME_VOCAB = int(_require("n_vocab"))
    N_TONE_VOCAB = int(_require("n_tones"))


class MinimalG2P:
    """Per-(language, text) hand-crafted phonemization for the fixture-only
    G2P bypass (`crates/vokra-models/tests/parity_sbv2_real.rs`'s Task 7
    `PhonemizeFixture` reader).

    Rationale (design doc §7 + `parity_sbv2_real.rs` module doc "The G2P
    bypass"): the Rust parity test does NOT wire a real 8-language piper-
    plus G2P (that crate lives OUT of the zero-dep root workspace per
    NFR-DS-02). Instead it replays whatever `phoneme_ids` / `tones` /
    `word_boundaries` this dumper wrote via
    `SbV2Phonemizer::from_fixture`. So the ONLY property this class needs
    is: given the same `(language, text)` pair, always produce the same
    output (byte-stable across runs).

    Populating the tables is an OWNER task — read the SBV2 v2 config's
    `phoneme_id_map` from a real checkpoint, then transcribe the target
    `--text` through it once, commit the row here. NEVER add a heuristic
    fall-through (FR-EX-08) — a cache miss is a loud NotImplementedError.
    """

    # Owner-populated rows go here. Each value MUST be a dict with the
    # three keys below, each a list of length T_text. `phoneme_ids` values
    # must be in `[0, N_PHONEME_VOCAB)`, `tones` in `[0, N_TONE_VOCAB)`.
    # `word_boundaries` is retained as a G2P output (Rust
    # `PhonemizeResult::word_boundaries` still carries it post-M6 for
    # fixture stability) even though the text encoder no longer consumes
    # it — post-M6 real-checkpoint scout, `enc_p.word_boundary_emb.weight`
    # does not exist and the per-utterance `language_embed` broadcast-add
    # replaced the per-position word-boundary lookup. Values should be in
    # `{0, 1}` (word-boundary flag) for schema stability with the pre-M6
    # fixture format.
    # Populated 2026-08-06 for the first `--do-dump` end-to-end run on
    # `litagin/Style-Bert-VITS2-2.0-base-JP-Extra` (`enc_p.emb.weight`
    # `[112, 192]`, `enc_p.tone_emb.weight` `[12, 192]`,
    # `enc_p.language_emb.weight` `[3, 192]`; observed with
    # `safetensors.safe_open(...).keys()`, no AGPL upstream referenced —
    # only tensor shapes, matching this file's own "Clean-room reminder").
    #
    # The IDs below are chosen from the observed valid ranges
    # (phoneme_ids ∈ [0, 112), tones ∈ [0, 12)), **not** transcribed from
    # a real SBV2 phoneme_id_map (which would require reading the AGPL
    # upstream to obtain). The parity contract this dumper participates
    # in is *comparing the same G2P output through Python and Rust*
    # (`parity_sbv2_real.rs` module doc "The G2P bypass" — Rust reads
    # `phoneme_ids.bin` verbatim, Python fed `encoder.emb(ids)` the same
    # sequence), so semantically-neutral placeholder IDs still test the
    # numerical equivalence of the two forward passes; they do NOT test
    # that the produced audio is intelligible for the input text. A real
    # SBV2 phoneme_id_map (once available to the owner) is dropped in
    # here without any Rust-side change: the fixture format is
    # unchanged.
    #
    # T_text = 8 for "テスト" (a rounded typical mora expansion: BOS +
    # t + e + s + u + t + o + EOS, matching the vanilla-VITS
    # BOS/PAD/EOS framing convention every SBV2 fork inherits;
    # word_boundaries[0]=1 for the first phoneme's word start, rest 0).
    _JA_TABLE: "dict[str, dict[str, list[int]]]" = {
        "テスト": {
            "phoneme_ids":     [1, 10, 11, 12, 13, 14, 15, 2],
            "tones":           [0,  1,  0,  1,  0,  1,  0, 0],
            "word_boundaries": [1,  0,  0,  0,  0,  0,  0, 0],
        },
    }
    # T_text = 16 for "This is a test." (English default): BOS + T + h + i
    # + s + space-elided-boundary + i + s + a + t + e + s + t + . + EOS
    # — the vanilla-VITS EN framing convention (`phonemize_en_char_mapping`
    # in `crates/vokra-models/src/sbv2/g2p.rs` also skips spaces this way).
    # Tones are all 0 (`Language::EN` documented convention).
    _EN_TABLE: "dict[str, dict[str, list[int]]]" = {
        "This is a test.": {
            "phoneme_ids":     [1, 20, 21, 22, 23, 22, 23, 24, 25, 26, 27, 25, 26, 28, 29, 2],
            "tones":           [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0],
            "word_boundaries": [1,  0,  0,  0,  0,  1,  0,  1,  1,  0,  0,  0,  0,  0,  0, 0],
        },
    }
    # Fixture-only ZH row. As with the JA/EN rows above, these are valid
    # checkpoint vocabulary/tone indices replayed byte-for-byte by Rust; this
    # row deliberately does not claim production Mandarin G2P correctness.
    # The independent part of this leg is the real HF WordPiece+BERT forward
    # and every downstream SBV2 tensor, not this hand-authored input fixture.
    _ZH_TABLE: "dict[str, dict[str, list[int]]]" = {
        "你好。": {
            "phoneme_ids":     [1, 30, 31, 32, 2],
            "tones":           [0,  3,  3,  0, 0],
            "word_boundaries": [1,  0,  1,  0, 0],
        },
    }

    def phonemize(self, text: str, language: str) -> dict:
        tables = {
            "JA": self._JA_TABLE,
            "EN": self._EN_TABLE,
            "ZH": self._ZH_TABLE,
        }
        table = tables[language.upper()]
        if text not in table:
            raise NotImplementedError(
                f"{LOG_PREFIX} MinimalG2P has no entry for "
                f"(language={language!r}, text={text!r}). Add one row to "
                f"MinimalG2P._{language.upper()}_TABLE (fields: phoneme_ids, "
                "tones, word_boundaries; each a list of ints of length "
                "T_text). NEVER add a heuristic fall-through (FR-EX-08)."
            )
        row = table[text]
        for k in ("phoneme_ids", "tones", "word_boundaries"):
            if k not in row:
                raise NotImplementedError(
                    f"{LOG_PREFIX} MinimalG2P row for "
                    f"(language={language!r}, text={text!r}) is missing key "
                    f"{k!r}."
                )
        if not (len(row["phoneme_ids"]) == len(row["tones"]) == len(row["word_boundaries"])):
            raise NotImplementedError(
                f"{LOG_PREFIX} MinimalG2P row for "
                f"(language={language!r}, text={text!r}) has inconsistent "
                "lengths across phoneme_ids/tones/word_boundaries."
            )
        return row


def prepare_torch(args: argparse.Namespace, torch):
    """Step 0. Deterministic seeding for the SDP's Gaussian draws (design
    doc §10: `sdp_sample` is compared with an atol that assumes both
    sides used identical seeds, otherwise it degenerates to
    noise-vs-noise) + CPU-only device (a `.bin` dump should not vary by
    CUDA driver/cuDNN benchmarking heuristic).

    Wave 3 (Task 15, per-tensor atol calibration prep) strengthens this
    beyond the pre-existing `torch.manual_seed` alone. `sdp_sample` and
    `run_flow`'s prior-noise draw already re-seed locally right before
    their own `torch.randn`/`.normal_()` calls (see those call sites'
    own RNG-parity comments) — this function's job is everything
    *upstream* of that: `nn.Module.__init__`'s Kaiming inits and any op
    that could otherwise pick a non-deterministic reduction order on
    this machine. That matters for Wave 3's "is a stage's drift a real
    bug or just noise?" triage — a reference dump that is not
    reproducible against ITSELF makes any atol calibrated against it
    meaningless. `torch.use_deterministic_algorithms(True, warn_only=False)`
    is intentionally strict (not `warn_only=True`): a silent fallback to
    a non-deterministic op would defeat the entire point of a byte-
    reproducible reference fixture (the same FR-EX-08 / NFR-QL-04
    "never silently approximate" posture this file's module doc already
    applies elsewhere), so a genuinely non-deterministic CPU op should
    surface as a loud `RuntimeError` at the call site that hits it,
    not as silent divergence discovered only after atol calibration.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
    except AttributeError:
        # Pre-dates this API (torch < 1.8) — not expected for any torch
        # version this project targets (see tools/parity/pyproject.toml's
        # `torch` pin), but kept defensive rather than assuming a floor.
        # manual_seed below still applies; only the strict-enforcement
        # half of Wave 3's reproducibility story is unavailable here.
        print(
            f"{LOG_PREFIX} torch.use_deterministic_algorithms unavailable "
            f"on this torch build ({getattr(torch, '__version__', '?')}) "
            "— continuing with manual_seed-only determinism."
        )
    torch.manual_seed(args.seed)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)

    # numpy is a real (non-stdlib) dependency — imported locally here,
    # matching write_f32_bin/write_u16_bin/write_u8_bin's own established
    # per-function `import numpy as np` convention rather than a
    # module-level import (schema-preview / --help must still work with
    # neither numpy nor torch installed).
    import numpy as _np

    _np.random.seed(args.seed)
    random.seed(args.seed)

    return torch.device("cpu")


def load_sbv2_checkpoint(args: argparse.Namespace):
    """Step 1. Reads Task-3 output: `--checkpoint` is a directory
    containing `vokra-sbv2-config.json` (+ optional siblings) plus one
    or more `.safetensors` weight file(s). Returns `(config, state_dict)`.

    Multi-shard checkpoints (`model-*-of-*.safetensors` +
    `*.safetensors.index.json`) are refused loudly — that is a Task-3
    limitation (see `sbv2_prepare_checkpoint.py` "Known limitations")
    plus `memory [[project-vokra-cli-sharded-safetensors]]`, they must
    be merged externally before running this dumper.
    """
    from safetensors.torch import load_file

    config_path = args.checkpoint / "vokra-sbv2-config.json"
    if not config_path.exists():
        sys.exit(
            f"{LOG_PREFIX} missing {config_path} — rerun "
            f"tools/parity/sbv2_prepare_checkpoint.py --output-dir "
            f"{args.checkpoint} first (Task 3 gate)."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Collect all sibling .safetensors files (SBV2 base ships as
    # e.g. G_0.safetensors alongside D_0/WD_0 — we merge every `.safetensors`
    # file we find into one state_dict, and downstream Rust `convert --model
    # sbv2` picks its own subset). Refuse shard indices loudly.
    st_files = sorted(args.checkpoint.rglob("*.safetensors"))
    if not st_files:
        sys.exit(
            f"{LOG_PREFIX} no .safetensors files found under {args.checkpoint} "
            "— did Task-3 prep actually complete?"
        )
    for st in st_files:
        if st.name.endswith(".safetensors.index.json") or "of-" in st.stem:
            sys.exit(
                f"{LOG_PREFIX} multi-shard checkpoint detected at {st}. Merge "
                "shards externally before running this dumper (memory "
                "[[project-vokra-cli-sharded-safetensors]])."
            )
    state_dict: dict = {}
    for st in st_files:
        piece = load_file(str(st), device="cpu")
        # `enc_p.emb.weight` from G_0 must never silently collide with a
        # same-named tensor in D_0 / WD_0 — loud on first duplicate.
        for name, tensor in piece.items():
            if name in state_dict:
                sys.exit(
                    f"{LOG_PREFIX} tensor name {name!r} appears in more than "
                    f"one .safetensors file under {args.checkpoint} — refusing "
                    "to silently pick one (FR-EX-08). Move the extra files "
                    "out of --checkpoint or merge them by hand."
                )
            state_dict[name] = tensor

    _resolve_arch_constants(config)
    return config, state_dict


def load_bert_encoders(args: argparse.Namespace, transformers):
    """Step 2. Load JA/EN and, for a ZH run, the third real encoder.

    DeBERTa v2 (JA), DeBERTa v3 (EN), and plain BERT (ZH) all load through
    the actual upstream `transformers.AutoModel` / `AutoTokenizer` classes.
    There is no local mirror fallback: import/load failure aborts the dump.
    Keeping ZH conditional preserves the existing JA/EN workflow's download
    and memory footprint until the explicit four-file leg is requested.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    # transformers 5 preserves fp16 source weights by default. All Vokra BERT
    # GGUFs and the independent CPU oracle compare in float32, so pin the
    # reference dtype rather than inheriting a version-dependent default.
    tok_ja = AutoTokenizer.from_pretrained(args.bert_ja_repo)
    model_ja = AutoModel.from_pretrained(args.bert_ja_repo, dtype=torch.float32).eval()
    tok_en = AutoTokenizer.from_pretrained(args.bert_en_repo)
    model_en = AutoModel.from_pretrained(args.bert_en_repo, dtype=torch.float32).eval()
    zh = None
    if args.language.lower() == "zh":
        tok_zh = AutoTokenizer.from_pretrained(args.bert_zh_repo)
        model_zh = AutoModel.from_pretrained(args.bert_zh_repo, dtype=torch.float32).eval()
        zh = (tok_zh, model_zh)
    return (tok_ja, model_ja), (tok_en, model_en), zh


def _load_tensor(state_dict: dict, candidates: "list[str]", role: str, torch):
    """Loud lookup helper: tries each candidate upstream tensor name in
    order (Task-3 normalization is best-effort; the raw safetensors may
    use `enc_p.*` / `dp.*` / `flow.*` / `dec.*` / `emb_g.*` / etc.).
    Fails with FR-EX-08 verbose message on cache miss — never falls back
    to a fabricated zero."""
    for name in candidates:
        if name in state_dict:
            return state_dict[name].to(dtype=torch.float32)
    sys.exit(
        f"{LOG_PREFIX} missing tensor for {role}: none of "
        f"{candidates!r} present in the checkpoint. If your checkpoint uses "
        "a different name, add it to the candidate list here (do not fabricate)."
    )


def build_text_encoder(state_dict: dict, torch):
    """Step 4a. Instantiate vendored `vendor.vits.text_encoder.TextEncoder`
    and load its weights from the state_dict (upstream naming:
    `enc_p.emb.*`, `enc_p.encoder.*`, `enc_p.proj.*`).

    The SBV2 additions (tone_emb + language_emb, post-M6 real-checkpoint
    correction from the design doc's original tone+word_boundary_emb
    estimate) live outside this class — see `run_text_encoder` for the
    additive sum before the transformer stack.

    Rationale for filter_channels=768, n_heads=2, n_layers=6,
    kernel_size=3, p_dropout=0.1: VITS/SBV2 base convention across
    published permissive-reference configs (jaywalnut310/vits
    configs/*.json — MIT). Real `filter_channels` etc. would resolve
    from `d_ff` / `n_text_layers` in the Task-3 config side-car; a full
    real port would call `_require("d_ff")` etc. here. This vendoring
    uses the design-doc §7 pins.
    """
    from vendor.vits.text_encoder import TextEncoder as VitsTextEncoder

    encoder = VitsTextEncoder(
        n_vocab=N_PHONEME_VOCAB,
        out_channels=D_MODEL,
        hidden_channels=D_MODEL,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.1,
    ).eval()

    # Upstream naming: enc_p.emb.weight, enc_p.encoder.*, enc_p.proj.weight/bias.
    # A missing tensor is loud (FR-EX-08); we DO NOT silently keep the
    # random-init from VitsTextEncoder.__init__.
    emb_w = _load_tensor(state_dict, ["enc_p.emb.weight"], "text_encoder.phoneme_embed", torch)
    with torch.no_grad():
        encoder.emb.weight.copy_(emb_w)
        # `enc_p.encoder.*` reflects the vendored `attentions.Encoder` layout
        # 1:1 (both are `models.py`'s TextEncoder.encoder = Encoder(...) plus
        # the exact same MultiHeadAttention/FFN internals we just imported).
        # A prefix-scoped state_dict load handles all inner keys uniformly.
        encoder_prefix = "enc_p.encoder."
        encoder_state = {
            k[len(encoder_prefix):]: v.to(dtype=torch.float32)
            for k, v in state_dict.items()
            if k.startswith(encoder_prefix)
        }
        if not encoder_state:
            sys.exit(
                f"{LOG_PREFIX} no tensors under `enc_p.encoder.*` in the "
                "checkpoint — text encoder inner weights cannot be loaded."
            )
        missing_keys, unexpected_keys = encoder.encoder.load_state_dict(
            encoder_state, strict=False
        )
        # Missing keys = weights the vendored VitsTextEncoder.encoder expects
        # but the checkpoint does not supply. Loud (FR-EX-08).
        if missing_keys:
            sys.exit(
                f"{LOG_PREFIX} encoder.encoder is missing "
                f"{len(missing_keys)} tensor(s) after loading: "
                f"{missing_keys[:8]}{'...' if len(missing_keys) > 8 else ''}"
            )
        # `enc_p.proj` (stats projection, `[D_MODEL*2, D_MODEL, 1]` + bias):
        encoder.proj.weight.copy_(
            _load_tensor(state_dict, ["enc_p.proj.weight"], "text_encoder.proj.weight", torch)
        )
        encoder.proj.bias.copy_(
            _load_tensor(state_dict, ["enc_p.proj.bias"], "text_encoder.proj.bias", torch)
        )
    return encoder


def build_sbv2_extras(state_dict: dict, torch):
    """Step 4b. SBV2's tone + language embedding tables (additive
    contributions to the phoneme embedding, applied BEFORE the
    transformer stack). These do NOT live in vanilla VITS — they are
    SBV2 additions per design doc §7 "既存 piper-plus VITS text encoder
    拡張 — tone + language embed 追加" (post-M6 correction; the design
    doc's original "tone + word_boundary" phrasing pre-dates the M6
    real-checkpoint scout — see this file's module docstring
    "Real-checkpoint tensor-layout finding" section for the trail).

    Both use `torch.nn.Embedding` (a `[V, D]` weight lookup — no
    architectural novelty), so no clean-room scratch is needed for the
    layer itself; only the weight-loading path is scratch.
    """
    from torch import nn as _nn

    tone_emb = _nn.Embedding(N_TONE_VOCAB, D_MODEL)
    lang_emb = _nn.Embedding(N_LANGUAGES, D_MODEL)
    with torch.no_grad():
        # Candidate naming: upstream may spell this `enc_p.tone_emb.weight`,
        # `enc_p.emb_tone.weight`, or drop it entirely (base ships without
        # tones on some SKUs). Loud FR-EX-08 error if none present.
        tone_emb.weight.copy_(
            _load_tensor(
                state_dict,
                ["enc_p.tone_emb.weight", "enc_p.emb_tone.weight"],
                "tone_embed",
                torch,
            )
        )
        # `enc_p.language_emb.weight` is SBV2 v2-specific and observed
        # `[N_LANGUAGES=3, D_MODEL=192]` on the real base checkpoint
        # (`litagin/Style-Bert-VITS2-2.0-base-JP-Extra`). Matches
        # `crates/vokra-models/src/sbv2/text_encoder.rs`
        # `SbV2TextEncoder::language_embed` (`[N_LANGUAGES, D_MODEL]`,
        # post-`b1e8f16`). Row ordering: `JA = 0`, `EN = 1`, `ZH = 2`
        # (`crates/vokra-models/src/sbv2/g2p.rs` `Language::language_id`);
        # pending real-checkpoint config verification per Rust
        # `SbV2TextEncoder::forward`'s `language_id` doc note.
        lang_emb.weight.copy_(
            _load_tensor(
                state_dict,
                [
                    "enc_p.language_emb.weight",
                    "enc_p.lang_emb.weight",
                    "enc_p.emb_language.weight",
                ],
                "language_embed",
                torch,
            )
        )
    return tone_emb, lang_emb


def run_text_encoder(encoder, tone_emb, lang_emb, phoneme_ids, tones,
                     language_id, torch):
    """Step 4c. Runs the SBV2 text encoder forward, returning
    (phoneme_embed [T_text, D_MODEL], text_hidden [T_text, D_MODEL],
    x_mask [1, 1, T_text]).

    SBV2's extension of vanilla VITS (per design doc §7, post-M6
    real-checkpoint correction):

        x = (emb_phoneme[t] + emb_tone[t] + emb_language[lang]) * sqrt(d_model)

    where `emb_language[lang]` is a single per-utterance row (`[D_MODEL]`)
    broadcast-added identically to every position `t`, NOT a per-position
    lookup — corresponds to `SbV2TextEncoder::forward`'s per-utterance
    `language_id: u8` argument on the Rust side (post-`b1e8f16`;
    `crates/vokra-models/src/sbv2/text_encoder.rs`).

    `language_id` is a plain `int` (0/1/2 for JA/EN/ZH, matching
    `Language::language_id`), not a list — caller resolves it from
    `--language` once per invocation.

    The dumper writes phoneme_embed as [T_text, 192] and text_hidden as
    [T_text, 192], matching design doc §10. Internally VITS shapes are
    [B, D, T]; we transpose+squeeze on write.
    """
    import math as _math
    from vendor.vits import commons

    ids = torch.tensor([phoneme_ids], dtype=torch.long)  # [1, T]
    ton = torch.tensor([tones], dtype=torch.long)        # [1, T]
    # `language_id` is a per-utterance scalar (NOT a per-position vector) —
    # `SbV2TextEncoder::forward` on the Rust side takes `language_id: u8`.
    # Wrap in shape `[1]` for the embedding lookup, get `[1, D_MODEL]`, then
    # broadcast-add into every position of `[1, T, D_MODEL]` via numpy /
    # torch broadcasting (trailing-dim alignment expands `[1, D_MODEL]` →
    # `[1, 1, D_MODEL]` → `[1, T, D_MODEL]`). Doing the broadcast this way
    # matches the Rust `let lang_row = &self.language_embed[lang_start..]`
    # hoisted-slice pattern exactly (see `text_encoder.rs::forward` L214-235).
    lang_id_tensor = torch.tensor([language_id], dtype=torch.long)  # [1] scalar
    x_lengths = torch.tensor([len(phoneme_ids)], dtype=torch.long)

    # Additive SBV2 embed sum BEFORE sqrt scaling. Corresponds to
    # `SbV2TextEncoder::forward`'s phoneme+tone+language sum on the Rust side.
    x_phon = encoder.emb(ids)                        # [1, T, D]
    x_tone = tone_emb(ton)                           # [1, T, D]
    lang_row = lang_emb(lang_id_tensor).unsqueeze(1)  # [1, 1, D] (broadcast target)
    phoneme_embed = x_phon + x_tone + lang_row       # [1, T, D] via broadcast over T
    x = phoneme_embed * _math.sqrt(D_MODEL)

    # Rest of vendored VitsTextEncoder.forward (inlined so we can capture
    # `text_hidden` — the vendored `.forward` returns `x, m, logs, x_mask`,
    # we need `x` here for design doc §10's `text_hidden` slot).
    x = torch.transpose(x, 1, -1)  # [1, D, T]
    x_mask = torch.unsqueeze(
        commons.sequence_mask(x_lengths, x.size(2)), 1
    ).to(x.dtype)
    text_hidden = encoder.encoder(x * x_mask, x_mask)  # [1, D, T]

    return (
        phoneme_embed.squeeze(0),                        # [T, D]
        text_hidden.squeeze(0).transpose(0, 1),          # [T, D]
        x_mask,                                          # [1, 1, T]
    )


def run_bert(tok, model, text: str, torch):
    """Steps 5 & 6. Returns `hidden_state [T_bert, hidden_size]`.
    Standard HF transformers tokenize+forward; the tokenizer's own
    T_bert (subword seq length for `text`) defines the dump's `T_bert`
    dimension.
    """
    inputs = tok(text, return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.squeeze(0)  # [T_bert, H]


class BertBridge:
    """Step 7. SBV2's language projection — a single Conv1d(D_BERT,
    D_MODEL, kernel=1) applied to the ACTIVE language's DeBERTa hidden
    state, aligned from `T_bert` to `T_text` and added to `text_hidden`.

    This is scratch (`docs/superpowers/specs/2026-07-26-sbv2-v2-design.md`
    §7 "新規, conv1d 1 層 + additive residual") — not present in vanilla
    jaywalnut310/vits. No SBV2/BV2 AGPL source was consulted.

    NOTE on length alignment: `T_bert` (BERT subword seq length) does NOT
    equal `T_text` (phoneme seq length). Design doc §7 leaves the exact
    alignment strategy checkpoint-specific; this implementation uses
    linear interpolation on the T axis as a placeholder, honestly
    reported by the manifest (see comment near `bert_bridge_out` write).
    A real checkpoint may carry attention-pool alignment weights
    (`bert_align_*`) — inspect and rewrite this class if so.
    """

    def __init__(self, state_dict: dict, language: str, torch):
        language_upper = language.upper()
        prefix = f"enc_p.bert_proj_{language_upper.lower()}"
        # The real JP-Extra fixture collapses the language projections into a single
        # `enc_p.bert_proj` — try that as a fallback (design doc §7 does
        # not pin the exact SKU-vs-name choice, so we try both honestly).
        self.weight = _load_tensor(
            state_dict,
            [f"{prefix}.weight", "enc_p.bert_proj.weight"],
            f"bert_bridge.{language_upper}.weight",
            torch,
        )  # [D_MODEL, D_BERT, 1]
        self.bias = _load_tensor(
            state_dict,
            [f"{prefix}.bias", "enc_p.bert_proj.bias"],
            f"bert_bridge.{language_upper}.bias",
            torch,
        )  # [D_MODEL]

    def forward(self, bert_hidden, text_hidden_transposed, t_text: int, torch):
        """`bert_hidden`: [T_bert, D_BERT]. `text_hidden_transposed`:
        [D_MODEL, T_text] (VITS-side [D, T] convention). Returns
        [T_text, D_MODEL]."""
        import torch.nn.functional as _F

        # HF DeBERTa `AutoModel.forward` sometimes returns fp16 hidden states
        # even when the checkpoint is fp32 (transformers ≥5 default) — the
        # bridge Conv1d's weights are fp32 (loaded from the SBV2 safetensors
        # verbatim), so cast the BERT side up to the weight dtype to avoid
        # `RuntimeError: Input type (c10::Half) and bias type (float) should
        # be the same`. Byte-exact on fp32-in, and honest upcast on fp16-in.
        bert_hidden = bert_hidden.to(dtype=self.weight.dtype)
        bert_bt = bert_hidden.transpose(0, 1).unsqueeze(0)  # [1, D_BERT, T_bert]
        # Placeholder: linear interpolation on T axis. See class docstring.
        bert_aligned = _F.interpolate(
            bert_bt, size=t_text, mode="linear", align_corners=False
        )  # [1, D_BERT, T_text]
        projected = _F.conv1d(bert_aligned, weight=self.weight, bias=self.bias)
        # projected: [1, D_MODEL, T_text]
        bridge_out = projected.squeeze(0) + text_hidden_transposed  # [D_MODEL, T_text]
        return bridge_out.transpose(0, 1)  # [T_text, D_MODEL]


def run_speaker_embedding(state_dict: dict, speaker_id: int, torch):
    """Step 8. Speaker embedding lookup — returns [1, D_SPEAKER] (float32).

    Two code paths:
      (a) fine-tuned SKU: `emb_g.weight [n_speakers, D_SPEAKER]` table
          lookup by `--speaker-id` (design doc §7 canonical form).
      (b) base ckpt (Blocker 3, 2026-08-06): base SBV2 v2 ships an
          `enc_p.encoder.spk_emb_linear.weight [D_MODEL, D_SPEAKER]`
          projection but NO `emb_g` table — the runtime expects an
          external caller-supplied embedding. For the dumper's
          deterministic clean-room reference, we substitute an all-zero
          `[1, D_SPEAKER]` vector (matches the Rust Blocker-3 zero-shot
          default for `SbV2SynthRequest::speaker_embedding = None +
          ExternalSpeakerProjection = Some`, so both sides agree). The
          D_SPEAKER dim is recovered from `spk_emb_linear.weight`'s
          trailing shape.
    """
    # Path (a): emb_g table lookup.
    table_candidates = ["emb_g.weight", "emb_g", "sbv2.speaker.table"]
    for name in table_candidates:
        if name in state_dict:
            table = state_dict[name]
            if speaker_id < 0 or speaker_id >= table.shape[0]:
                sys.exit(
                    f"{LOG_PREFIX} --speaker-id {speaker_id} out of range "
                    f"[0, {table.shape[0]}) for this checkpoint's emb_g table."
                )
            return table[speaker_id : speaker_id + 1].to(dtype=torch.float32)

    # Path (b): base ckpt zero-shot default (Blocker 3).
    # Recover D_SPEAKER from `enc_p.encoder.spk_emb_linear.weight`'s
    # trailing dim = [D_MODEL, D_SPEAKER].
    spk_linear = state_dict.get("enc_p.encoder.spk_emb_linear.weight")
    if spk_linear is None:
        sys.exit(
            f"{LOG_PREFIX} missing tensor for speaker.table: neither "
            f"{table_candidates!r} nor `enc_p.encoder.spk_emb_linear.weight` "
            "present in the checkpoint. If your checkpoint uses a different "
            "name, add it to the candidate list here (do not fabricate)."
        )
    d_speaker = spk_linear.shape[1]
    print(
        f"{LOG_PREFIX} speaker: base ckpt (no emb_g table) — using all-zero "
        f"[1, {d_speaker}] default (Blocker 3 zero-shot, matches Rust "
        f"SbV2SynthRequest::speaker_embedding = None)"
    )
    return torch.zeros((1, d_speaker), dtype=torch.float32)


class StyleVectorInjector:
    """Step 9. SBV2 v2's `emb_g_style` path — an AdaIN-flavored
    Linear(D_STYLE → D_MODEL) + bias projecting the caller-supplied
    style vector into the model space.

    Scratch (design doc §7 "新規, AdaIN 系 scale+bias") — not present in
    vanilla jaywalnut310/vits. No SBV2/BV2 AGPL source was consulted.
    """

    def __init__(self, state_dict: dict, torch, d_model: int, d_style: int):
        # Candidate names: SBV2 SKUs may spell this `emb_g_style.weight`
        # (most common) or `style_proj.weight` (some forks).
        #
        # Blocker 6 (2026-08-06): SBV2 v2 base ckpt ships **no** style
        # projection tensors (style is trained per-speaker during fine-
        # tune). Base inference is an identity injector equivalent to
        # `style_vec = 0`. Falls back to all-zero weights so
        # `forward(style_vec) = style_vec @ 0 + 0 = zeros[D_MODEL]`.
        # Matches Rust Blocker-6 `StyleVectorInjector` zero-weight
        # identity fallback (crates/vokra-models/src/sbv2/mod.rs).
        weight_candidates = [
            "emb_g_style.weight",
            "style_proj.weight",
            "sbv2.style_injector.proj_scale",
        ]
        bias_candidates = [
            "emb_g_style.bias",
            "style_proj.bias",
            "sbv2.style_injector.proj_bias",
        ]
        has_weight = any(k in state_dict for k in weight_candidates)
        has_bias = any(k in state_dict for k in bias_candidates)
        if has_weight and has_bias:
            self.weight = _load_tensor(
                state_dict, weight_candidates, "style_injector.weight", torch,
            )
            self.bias = _load_tensor(
                state_dict, bias_candidates, "style_injector.bias", torch,
            )
        elif not has_weight and not has_bias:
            print(
                f"{LOG_PREFIX} style: base ckpt (no style projection tensors) "
                f"— using all-zero [{d_model}, {d_style}] weight + [{d_model}] "
                "bias identity fallback (Blocker 6, matches Rust "
                "StyleVectorInjector zero-weight default)"
            )
            self.weight = torch.zeros((d_model, d_style), dtype=torch.float32)
            self.bias = torch.zeros((d_model,), dtype=torch.float32)
        else:
            sys.exit(
                f"{LOG_PREFIX} style: only one of weight/bias present in "
                "checkpoint — a converter must emit both or neither "
                "(FR-EX-08: partial style projection is undefined)"
            )

    def forward(self, style_vec, torch):
        """`style_vec`: [1, D_STYLE]. Returns [1, D_MODEL]."""
        return style_vec @ self.weight.T + self.bias


class SDPReference:
    """Step 10. Clean-room scratch composition of the
    StochasticDurationPredictor described in arXiv:2106.06103 §2.3
    (Kim et al. 2021, "Conditional Variational Autoencoder with
    Adversarial Learning for End-to-End Text-to-Speech", VITS).

    Rationale for scratch (per Task 30 owner ruling — Option B of the
    scout report's Options A/B): the vendored
    `tools/parity/vendor/vits/` deliberately excluded upstream
    `models.py`'s `StochasticDurationPredictor` class as "training-side"
    (README §"What is NOT vendored"). Rather than re-open the vendor to
    add it (Option A, requires touching a licensed vendor tree in the
    same commit as pipeline body work), this class composes the SDP
    directly from the paper's structural description using ONLY the
    primitive building blocks already present in `vendor/vits/modules.py`
    (`DDSConv`, `ConvFlow`, `Flip`, `ElementwiseAffine` — all MIT via
    upstream `modules.py`).

    Composition (paper §2.3, matching the primitives' native usage in
    upstream `modules.py` docstrings):

        conditioning:   Conv1d(D_MODEL, filter_channels, 1)   # `.pre`
                        DDSConv(filter_channels, K=3, L=3)    # `.convs`
                        Conv1d(filter_channels, filter_channels, 1)  # `.proj`
        post-conditioning: same structure as above, gated on `g`
                        (speaker + style, projected)
        flows:          [ElementwiseAffine(2)] +
                        [ConvFlow(2, filter_channels, K=3, L=3), Flip()] * n_flows
        inference:      z ~ N(0, noise_scale_w * I) shape=[B, 2, T]
                        for flow in reversed(flows):
                            z = flow(z, x_mask, g=h_cond, reverse=True)
                        logw = z[:, 0, :]                # first channel is log-duration
                        w = exp(logw) * x_mask
                        durations = ceil(w).long()

    Tensor loading uses upstream `dp.*` naming (Task-3 normalized).
    Weights that a real checkpoint provides but the paper's "training-
    side" post- / posterior branches consume are LOADED and IGNORED
    silently at inference (they only affect the loss during training) —
    this is honest per §2.3's own explicit separation.

    NB: SBV2 v2 adds a "JP tone conditioning" branch (design doc §7
    "既存 piper-plus SDP 拡張（JP tone conditioning）"). The exact wiring
    is checkpoint-specific — owner MUST inspect the real checkpoint's
    `dp.tone_conditioning.*` weights and rewrite this class's `.forward`
    to add that path once a real fixture lands. As-shipped, the tone
    conditioning is NOT applied — this is a loud KNOWN LIMITATION
    documented here + at the manifest write site.
    """

    def __init__(self, state_dict: dict, filter_channels: int, torch):
        from torch import nn as _nn
        from vendor.vits import modules as _vits_modules

        # SDP is a flow module — its `forward` inverts the flow to sample
        # durations from a Gaussian prior. Build it as a proper nn.Module
        # so state_dict loading works uniformly.
        #
        # Blocker 7 (2026-08-06): the real SBV2 v2 base ckpt uses `sdp.*`
        # prefix (not `dp.*` — the old paper-default was VITS1's `dp` and
        # `sdp` sibling; SBV2 v2 keeps only the stochastic `sdp` and
        # renames it). The topology also adds a speaker-conditioning
        # `cond: Conv1d(D_SPEAKER, filter_channels, 1)` layer that
        # conditions the SDP on `g` (speaker vector). Recover both from
        # the real tensor shapes rather than assume paper defaults.
        sdp_prefix = "sdp."
        sdp_state = {
            k[len(sdp_prefix):]: v.to(dtype=torch.float32)
            for k, v in state_dict.items()
            if k.startswith(sdp_prefix) and not k.startswith("sdp.post_")
        }
        if not sdp_state:
            # Fallback: try `dp.*` prefix for VITS1-style checkpoints.
            dp_prefix = "dp."
            sdp_state = {
                k[len(dp_prefix):]: v.to(dtype=torch.float32)
                for k, v in state_dict.items()
                if k.startswith(dp_prefix)
            }
        if not sdp_state:
            sys.exit(
                f"{LOG_PREFIX} no `sdp.*` or `dp.*` tensors found in "
                "checkpoint — SDP cannot be initialized."
            )

        # Recover d_speaker from `cond.weight` shape if the layer exists;
        # else use the paper's non-conditioned SDP topology.
        cond_weight = sdp_state.get("cond.weight")
        d_speaker = int(cond_weight.shape[1]) if cond_weight is not None else 0

        class _Sdp(_nn.Module):
            def __init__(self, in_channels: int, filter_channels: int,
                         kernel_size: int, n_layers: int, n_flows: int,
                         d_speaker_cond: int):
                super().__init__()
                self.pre = _nn.Conv1d(in_channels, filter_channels, 1)
                self.convs = _vits_modules.DDSConv(
                    filter_channels, kernel_size, n_layers=n_layers, p_dropout=0.0
                )
                self.proj = _nn.Conv1d(filter_channels, filter_channels, 1)
                # SBV2 v2 addition: speaker conditioning (present when
                # `sdp.cond.*` exists in the checkpoint).
                if d_speaker_cond > 0:
                    self.cond = _nn.Conv1d(d_speaker_cond, filter_channels, 1)
                self.flows = _nn.ModuleList()
                self.flows.append(_vits_modules.ElementwiseAffine(2))
                for _ in range(n_flows):
                    self.flows.append(_vits_modules.ConvFlow(
                        2, filter_channels, kernel_size, n_layers=n_layers
                    ))
                    self.flows.append(_vits_modules.Flip())

        self._m = _Sdp(
            in_channels=D_MODEL,
            filter_channels=filter_channels,
            kernel_size=SDP_KERNEL_SIZE,
            n_layers=SDP_DDS_N_LAYERS,
            n_flows=SDP_N_FLOWS,
            d_speaker_cond=d_speaker,
        ).eval()

        # Load strictly on the reduced key set (post-*/posterior training-
        # side already excluded above). ANY known key missing triggers a
        # loud FR-EX-08 error.
        missing_keys, unexpected_keys = self._m.load_state_dict(
            sdp_state, strict=False
        )
        if missing_keys:
            sys.exit(
                f"{LOG_PREFIX} SDP is missing {len(missing_keys)} tensor(s): "
                f"{missing_keys[:8]}{'...' if len(missing_keys) > 8 else ''}. "
                "Real ckpt topology diverges from the built model."
            )
        if unexpected_keys:
            print(
                f"{LOG_PREFIX} SDP: {len(unexpected_keys)} unexpected keys "
                f"ignored (training-side or unknown): "
                f"{unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}"
            )

    def sample(self, x, x_mask, g, noise_scale_w: float, torch):
        """`x`: [B, D_MODEL, T_text] text-hidden features.
        `x_mask`: [B, 1, T_text]. `g`: [B, D_SPEAKER, 1] speaker/style
        combined conditioning. Returns durations `[T_text]` (float32,
        semantic values are discrete counts; see design doc §10 note on
        why the .bin file is still f32).

        Faithful mirror of upstream
        ``StochasticDurationPredictor.forward(reverse=True)`` in
        ``tools/parity/vendor/vits/sdp.py`` (jaywalnut310/vits MIT):

            x = self.pre(x)
            if g is not None:
              x = x + self.cond(g)             # speaker/style conditioning
            x = self.convs(x, x_mask)          # DDS with x_mask (no g)
            x = self.proj(x) * x_mask
            ...
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]   # drop the useless vflow
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=True)
            logw = z[:, 0, :]

        Two prior Task-30 shortcuts that this now un-cuts:

        1. The old Python body skipped ``+ self.cond(g)`` entirely — the
           `sdp.cond.*` weights were LOADED (see ``__init__``) but never
           applied. Upstream unconditionally applies them when ``g`` is
           provided, and the real SBV2 v2 base ckpt does carry ``sdp.cond.*``
           tensors. Post-fix we now match upstream: apply ``.cond(g)`` iff
           the ``cond`` submodule was constructed from a non-empty
           ``d_speaker_cond``. Otherwise the branch is skipped like upstream
           does when ``gin_channels == 0``.
        2. The old Python walked ALL 9 items of ``reversed(self.flows)``,
           including the "useless vflow" at forward index 1 that upstream
           explicitly drops via ``flows[:-2] + [flows[-1]]``. Applying it
           at inference produces log-durations diverging from upstream by
           the full accumulated log-abs-det of an un-trained ConvFlow — a
           silent parity break the fixture happened to mask when the ckpt's
           un-trained weights ran near identity. Post-fix we drop the same
           layer upstream drops.
        """
        import torch.nn.functional as _F

        # Conditioning branch — upstream `StochasticDurationPredictor.forward`
        # lines 96-101 verbatim (no `* x_mask` on `pre`, add `cond(g)` before
        # `convs`, only `proj * x_mask` after `proj`).
        x = self._m.pre(x)
        if g is not None and hasattr(self._m, "cond"):
            x = x + self._m.cond(g)
        x = self._m.convs(x, x_mask)
        x = self._m.proj(x) * x_mask

        # Inference: sample from Gaussian prior, invert the flow.
        b = x.shape[0]
        t = x.shape[2]
        # RNG parity note (2026-08-08, workflow wf_eadb75fc-2eb-2 fix): the
        # naïve `torch.randn(b, 2, t)` produces a contiguous tensor, so
        # torch's `normal_kernel` dispatches to the SIMD `normal_fill_AVX2`
        # fast path on x86_64 CI hosts (see
        # ATen/native/cpu/DistributionTemplates.h:230-255). That path uses
        # `avx_mathfun`'s `log256_ps` / `sincos256_ps` — vectorized
        # approximations that differ from libm's scalar `logf`/`cosf`/`sinf`
        # by ~1 ULP for a non-trivial fraction of inputs. Vokra's Rust port
        # (`vokra_core::rng::TorchRandnStream`) is bit-exact against
        # `at::normal_distribution<double>` (the scalar streaming path);
        # matching the AVX2 approximations would require porting
        # `avx_mathfun`, which is out of scope and would break on ARM64
        # (M1 dev machines, no AVX2). Instead, force torch to take the
        # `else` branch of `normal_kernel` (line 246) by giving it a
        # non-contiguous tensor — `torch.empty(...)[...]` with a stride
        # mismatch does exactly this, without changing the sample count
        # (still b*2*t) or the seed contract.
        z_big = torch.empty(b, 2, t + 1, dtype=x.dtype, device=x.device)
        z = z_big[..., :t]
        assert not z.is_contiguous(), (
            "SBV2 SDP noise buffer must be non-contiguous so torch's "
            "normal_kernel takes the scalar `at::normal_distribution<double>` "
            "path (which matches Vokra Rust bit-exactly). If this assert "
            "fires, `torch.empty(...)[..., :t]` no longer produces a stride-"
            "mismatched view on this torch version — pick another view op."
        )
        z.normal_(0, 1)
        z = z * noise_scale_w
        # Upstream: `flows = list(reversed(self.flows))[:-2] + [flows[-1]]`.
        # `self._m.flows` is `[EA, CF, Flip, CF, Flip, CF, Flip, CF, Flip]`
        # (9 items for n_flows=4). Reversed then `[:-2] + [-1]` drops the
        # second-to-last item of the reversed list — which corresponds to
        # the FIRST ConvFlow in forward order (upstream `sdp.flows.1`) —
        # and keeps the trailing `EA`. This mirrors Rust
        # `SbV2SDP::sample`'s `self.flows[1..].iter().rev()` slice (Rust
        # stores only the 4 ConvFlows post-conversion, so the "skip first"
        # translates directly there).
        flows = list(reversed(self._m.flows))
        flows = flows[:-2] + [flows[-1]]
        for flow in flows:
            z = flow(z, x_mask, g=x, reverse=True)
        logw = z[:, 0, :]                    # [B, T_text]
        w = torch.exp(logw) * x_mask.squeeze(1)  # [B, T_text]
        # Round-up-to-integer durations (upstream convention), still f32
        # for the .bin dump (design doc §10 explicitly keeps sdp_sample
        # as float32 with a discrete-step atol).
        w_ceil = torch.ceil(w)
        return w_ceil.squeeze(0)             # [T_text]


def length_regulate(text_features, durations, torch):
    """Step 11. `text_features`: [T_text, D_MODEL], `durations`:
    [T_text] (float, integer counts). Returns `mel_hidden`:
    [T_mel, D_MODEL] where T_mel = int(round(durations.sum())).

    Plain `repeat_interleave` — matches upstream VITS's own length
    regulator behavior (no learnable regulator, no separate module
    weights).
    """
    counts = durations.round().clamp(min=1).long()  # [T_text]
    mel_hidden = torch.repeat_interleave(text_features, counts, dim=0)
    return mel_hidden  # [T_mel, D_MODEL]


def build_flow(state_dict: dict, torch):
    """Step 12a. Instantiate the vendored
    `vendor.vits.sbv2_flow.Sbv2TransformerCouplingBlock` (Blocker 8,
    2026-08-06) and load `flow.*` weights.

    Blocker 8 rationale (see `tools/parity/vendor/vits/sbv2_flow.py`
    header + `tools/parity/vendor/vits/README.md` target-files table):
    the sibling `vendor.vits.flow.ResidualCouplingBlock` uses a WN-based
    coupling (jaywalnut310/vits `enc.in_layers.*` / `enc.res_skip_layers.*`),
    which does NOT load the SBV2 v2 base checkpoint's `flow.*` state_dict
    (108 missing tensors under `strict=False`: SBV2 carries transformer-
    encoder weights `enc.attn_layers.*` / `enc.norm_layers_1.*` /
    `enc.ffn_layers.*` / `enc.spk_emb_linear.*` per coupling). The
    clean-room `Sbv2TransformerCouplingBlock` in `sbv2_flow.py` composes
    the SBV2 v2 layout from MIT primitives already vendored under
    `vendor/vits/`.

    Every architectural parameter passed to the constructor below is
    RECOVERED FROM THE REAL TENSOR SHAPES in `state_dict` — no
    invention, no "reasonable default" (FR-EX-08 / NFR-QL-04). A
    checkpoint whose `flow.*` sub-tree does not decompose cleanly
    against this build path fails loudly with a message naming exactly
    which tensor was expected + what the ckpt actually carries.
    """
    from vendor.vits.sbv2_flow import Sbv2TransformerCouplingBlock

    flow_prefix = "flow."
    flow_state = {
        k[len(flow_prefix):]: v.to(dtype=torch.float32)
        for k, v in state_dict.items()
        if k.startswith(flow_prefix)
    }
    if not flow_state:
        sys.exit(
            f"{LOG_PREFIX} no `flow.*` tensors found in checkpoint — "
            "normalizing flow cannot be initialized."
        )

    # Recover every hparam from real tensor shapes (Blocker 8).
    #
    # ---- pre.weight [hidden_channels, half_channels, 1] ----
    if "flows.0.pre.weight" not in flow_state:
        sys.exit(
            f"{LOG_PREFIX} flow: missing `flow.flows.0.pre.weight` — checkpoint "
            "does not look like a VITS/SBV2 normalizing flow. Inspect "
            "state_dict keys."
        )
    pre_w = flow_state["flows.0.pre.weight"]
    if pre_w.dim() != 3 or pre_w.shape[2] != 1:
        sys.exit(
            f"{LOG_PREFIX} flow: `flows.0.pre.weight` shape {tuple(pre_w.shape)} "
            "unexpected (expected [hidden_channels, half_channels, 1] Conv1d "
            "kernel=1)."
        )
    hidden_channels = int(pre_w.shape[0])
    half_channels = int(pre_w.shape[1])
    channels = 2 * half_channels

    # ---- enc.spk_emb_linear.weight [hidden_channels, gin_channels] ----
    if "flows.0.enc.spk_emb_linear.weight" not in flow_state:
        sys.exit(
            f"{LOG_PREFIX} flow: missing `flow.flows.0.enc.spk_emb_linear.weight` — "
            "SBV2 v2 base ckpt is expected to carry per-coupling speaker "
            "conditioning (Blocker 8). If this is a non-SBV2 VITS1 checkpoint, "
            "swap back to `vendor.vits.flow.ResidualCouplingBlock` and adjust."
        )
    spk_w = flow_state["flows.0.enc.spk_emb_linear.weight"]
    if spk_w.dim() != 2 or spk_w.shape[0] != hidden_channels:
        sys.exit(
            f"{LOG_PREFIX} flow: `flows.0.enc.spk_emb_linear.weight` shape "
            f"{tuple(spk_w.shape)} disagrees with hidden_channels={hidden_channels} "
            "(expected [hidden_channels, gin_channels] Linear)."
        )
    gin_channels = int(spk_w.shape[1])

    # ---- n_layers: count attn_layers.<i>.conv_q.weight ----
    n_layers = 0
    while f"flows.0.enc.attn_layers.{n_layers}.conv_q.weight" in flow_state:
        n_layers += 1
    if n_layers == 0:
        sys.exit(
            f"{LOG_PREFIX} flow: no `flows.0.enc.attn_layers.<i>.conv_q.weight` — "
            "SBV2 v2 base ckpt is expected to carry >= 1 transformer layer."
        )

    # ---- n_heads / k_channels / window_size:
    # emb_rel_k [n_heads_rel, 2*window_size + 1, k_channels], heads_share=True → n_heads_rel=1
    emb_rel_k = flow_state["flows.0.enc.attn_layers.0.emb_rel_k"]
    if emb_rel_k.dim() != 3:
        sys.exit(
            f"{LOG_PREFIX} flow: `flows.0.enc.attn_layers.0.emb_rel_k` shape "
            f"{tuple(emb_rel_k.shape)} unexpected (expected 3D [n_heads_rel, "
            "2*window_size+1, k_channels])."
        )
    k_channels = int(emb_rel_k.shape[2])
    if hidden_channels % k_channels != 0:
        sys.exit(
            f"{LOG_PREFIX} flow: hidden_channels={hidden_channels} not divisible "
            f"by k_channels={k_channels} (from emb_rel_k trailing dim)."
        )
    n_heads = hidden_channels // k_channels
    window_size_span = int(emb_rel_k.shape[1])
    if window_size_span % 2 != 1:
        sys.exit(
            f"{LOG_PREFIX} flow: emb_rel_k middle dim {window_size_span} even; "
            "expected odd (2*window_size + 1)."
        )
    window_size = (window_size_span - 1) // 2

    # ---- filter_channels / kernel_size: ffn_layers.0.conv_1.weight [filter, hidden, kernel]
    ffn_c1 = flow_state["flows.0.enc.ffn_layers.0.conv_1.weight"]
    if ffn_c1.dim() != 3 or ffn_c1.shape[1] != hidden_channels:
        sys.exit(
            f"{LOG_PREFIX} flow: `flows.0.enc.ffn_layers.0.conv_1.weight` shape "
            f"{tuple(ffn_c1.shape)} disagrees with hidden_channels="
            f"{hidden_channels} (expected [filter_channels, hidden_channels, "
            "kernel_size])."
        )
    filter_channels = int(ffn_c1.shape[0])
    kernel_size = int(ffn_c1.shape[2])

    # ---- n_flows: count non-Flip coupling layers (flows.<i>.pre.weight) ----
    n_flows_seen = 0
    while f"flows.{2 * n_flows_seen}.pre.weight" in flow_state:
        n_flows_seen += 1
    if n_flows_seen == 0:
        sys.exit(
            f"{LOG_PREFIX} flow: no coupling layers found (expected "
            "flows.0/2/4/6.pre.weight for interleaved [coupling, Flip] stack)."
        )

    # ---- mean_only: post.weight output dim ----
    post_w = flow_state["flows.0.post.weight"]
    if post_w.dim() != 3 or post_w.shape[1] != hidden_channels or post_w.shape[2] != 1:
        sys.exit(
            f"{LOG_PREFIX} flow: `flows.0.post.weight` shape {tuple(post_w.shape)} "
            f"disagrees with hidden_channels={hidden_channels} (expected "
            "[out, hidden_channels, 1])."
        )
    out_channels = int(post_w.shape[0])
    if out_channels == half_channels:
        mean_only = True
    elif out_channels == 2 * half_channels:
        mean_only = False
    else:
        sys.exit(
            f"{LOG_PREFIX} flow: `flows.0.post.weight` out={out_channels} matches "
            f"neither half_channels={half_channels} (mean_only=True) nor "
            f"2*half_channels={2 * half_channels} (mean_only=False)."
        )

    print(
        f"{LOG_PREFIX} flow (Blocker 8): "
        f"channels={channels}, hidden={hidden_channels}, gin={gin_channels}, "
        f"n_layers={n_layers}, n_heads={n_heads}, k_channels={k_channels}, "
        f"window_size={window_size}, filter={filter_channels}, "
        f"kernel_size={kernel_size}, n_flows={n_flows_seen}, mean_only={mean_only}"
    )

    flow = Sbv2TransformerCouplingBlock(
        channels=channels,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        n_heads=n_heads,
        n_layers=n_layers,
        filter_channels=filter_channels,
        p_dropout=0.0,
        window_size=window_size,
        n_flows=n_flows_seen,
        gin_channels=gin_channels,
        mean_only=mean_only,
    ).eval()

    with torch.no_grad():
        missing_keys, unexpected_keys = flow.load_state_dict(
            flow_state, strict=False
        )
    # Loud FR-EX-08 on ANY discrepancy — this is a real fixture load, no
    # silent tolerance for architectural drift.
    if missing_keys:
        sys.exit(
            f"{LOG_PREFIX} flow is missing {len(missing_keys)} tensor(s) after "
            f"loading `flow.*` into Sbv2TransformerCouplingBlock: "
            f"{missing_keys[:8]}{'...' if len(missing_keys) > 8 else ''}. "
            "Inspect the checkpoint — an SBV2 v2 base ckpt should load 0 "
            "missing / 0 unexpected keys against this build path."
        )
    if unexpected_keys:
        sys.exit(
            f"{LOG_PREFIX} flow has {len(unexpected_keys)} unexpected tensor(s) "
            f"after loading `flow.*`: "
            f"{unexpected_keys[:8]}{'...' if len(unexpected_keys) > 8 else ''}. "
            "Inspect the checkpoint — an SBV2 v2 base ckpt should load 0 "
            "missing / 0 unexpected keys against this build path."
        )
    return flow


def run_flow(flow, mel_hidden, x_mask_mel, g, noise_scale: float, torch, seed: int,
             dump_dir: "Path | None" = None):
    """Step 12b. `mel_hidden`: [T_mel, D_MODEL] (prior mean). Returns
    `(z_latent, n_flow_layers_dumped)` — `z_latent`: [T_mel, D_MODEL],
    `n_flow_layers_dumped`: the real per-layer count discovered below.

    VITS inference: sample `z_p ~ N(mel_hidden, noise_scale * I)`, then
    pass through `flow.reverse` to invert to `z`.

    RNG PARITY (2026-08-09): the noise draw uses a **fresh** `torch.manual_seed(seed)`
    stream fed through a NON-CONTIGUOUS view (`torch.empty(B, D, T+1)[..., :T]`)
    so it byte-matches Rust's `draw_flow_prior_noise`, which uses
    `TorchRandnStream::new(seed)` — a fresh stream at the same seed. Rationale
    is identical to the SDP.sample fix above (see the block comment there);
    the non-contig view forces torch's `normal_kernel` off the AVX2 fast path
    onto the scalar `at::normal_distribution<double>` path that Rust's port
    bit-matches.

    Wave 3 (Task 15) per-layer instrumentation: the pre-Task-15 body ended
    with a single `flow(z_p, x_mask_mel, g=g, reverse=True)` call. Below,
    that call is replaced by a manual unrolling of
    `Sbv2TransformerCouplingBlock.forward`'s own `reverse=True` body
    (`for layer in reversed(self.flows): x = layer(x, x_mask, g=g,
    reverse=True)`, `tools/parity/vendor/vits/sbv2_flow.py`) — calling
    each sub-module's `.forward` directly through `nn.Module.__call__` is
    numerically IDENTICAL to calling the block itself (neither
    `Sbv2TransformerCouplingLayer` nor `Flip` registers forward/backward
    hooks, so there is no dispatch difference); this is the same call
    sequence with a dump inserted after every
    `Sbv2TransformerCouplingLayer` step, not a reimplementation, so
    `z_latent` is bit-identical to what the single-shot call used to
    produce (verified by inspection, not re-run — Task 15 does not
    execute this script; Task 16 regenerates fixtures against it).

    When `dump_dir` is given, each `Sbv2TransformerCouplingLayer` step's
    output is written to `<dump_dir>/flow_layer_<i>_output.bin` (raw
    little-endian f32, same `[T_mel, D_MODEL]` on-disk convention as
    `z_latent.bin` — `write_f32_bin` applies the identical
    `.squeeze(0).transpose(0, 1)` reshape). `<i>` counts EXECUTION order
    during this reverse traversal: `i=0` is the FIRST coupling layer
    applied (the LAST-*constructed* one, since `reversed(self.flows)`
    walks construction order backwards), `i=n_flows-1` the last. For this
    block's even `[layer, Flip] * n_flows` interleaving, the very last op
    in the whole reverse traversal is always a coupling layer
    (construction index 0) — never a trailing `Flip` — so
    `flow_layer_<n_flows-1>_output.bin` ends up BYTE-IDENTICAL to
    `z_latent.bin`. That is an architectural fact of this interleaving,
    not a bug: kept as an honest per-layer trace point rather than
    special-cased away, so a future diff tool doesn't need to know this
    quirk to be correct — it can just diff every file uniformly.
    `n_flow_layers_dumped` (this function's 2nd return value) is the real
    count discovered from `flow.flows` at runtime — never hard-coded —
    so `run_pipeline_body`'s `build_manifest` call can't silently
    under/over-count against a checkpoint whose acoustic-flow depth
    differs from the common `4`.

    `tools/parity/vendor/vits2` (`p0p4k/vits2_pytorch`, Blocker 2b
    vendoring, Task 1-2) is DELIBERATELY NOT used to build or replay this
    flow, despite being the plan's namesake vendor tree: its
    `TransformerCouplingLayer.enc` is `modules.WN` (WaveNet), which does
    NOT match the real SBV2 v2 base checkpoint's attention-based
    `flow.flows.<i>.enc.attn_layers.*` tensor names — the same layout
    mismatch `vendor/vits/sbv2_flow.py`'s own Blocker 8 header already
    documents for jaywalnut310/vits's `ResidualCouplingBlock` (verified
    2026-08-06 by safetensors introspection of the real checkpoint, per
    that file's header). `Sbv2TransformerCouplingLayer` below (whose
    `enc` sub-tree IS the checkpoint's attention-based encoder) remains
    the sole real-checkpoint-loading path; `vendor.vits2` stays available
    as a clean-room cross-reference (imported and sanity-gated in
    `run_dump`, already read by Task 12's SBV2-SPK-EMB-LINEAR-DECISION
    ADR investigation) for future numerical comparison work, not a
    drop-in replacement here.
    """
    from vendor.vits.sbv2_flow import Sbv2TransformerCouplingLayer

    # Rearrange to VITS-convention [B, D, T].
    z_p = mel_hidden.transpose(0, 1).unsqueeze(0)  # [1, D, T_mel]
    if noise_scale != 0.0:
        # Fresh seed → matches Rust `TorchRandnStream::new(seed)`.
        torch.manual_seed(seed)
        b, d, t = z_p.shape
        # Non-contig view to force scalar path (see SDP.sample comment above).
        noise_big = torch.empty(b, d, t + 1, dtype=z_p.dtype, device=z_p.device)
        noise = noise_big[..., :t]
        assert not noise.is_contiguous(), (
            "flow noise buffer must be non-contiguous so torch takes the "
            "scalar at::normal_distribution<double> path that matches Rust"
        )
        noise.normal_(0, 1)
        z_p = z_p + noise * noise_scale

    # Manual unrolling of `Sbv2TransformerCouplingBlock.forward(reverse=True)`
    # — see docstring above for why this is numerically identical to the
    # single `flow(z_p, x_mask_mel, g=g, reverse=True)` call it replaces.
    x = z_p
    n_flow_layers_dumped = 0
    for sub_layer in reversed(flow.flows):
        x = sub_layer(x, x_mask_mel, g=g, reverse=True)
        if isinstance(sub_layer, Sbv2TransformerCouplingLayer):
            if dump_dir is not None:
                write_f32_bin(
                    dump_dir / f"flow_layer_{n_flow_layers_dumped}_output.bin",
                    x.squeeze(0).transpose(0, 1),  # [T_mel, D_MODEL]
                    torch,
                )
            n_flow_layers_dumped += 1
    z = x

    return z.squeeze(0).transpose(0, 1), n_flow_layers_dumped  # [T_mel, D_MODEL], int


def build_generator(state_dict: dict, torch):
    """Step 13a. Instantiate vendored `vendor.vits.decoder.Generator`
    and load `dec.*` weights. Removes weight_norm for inference
    stability (matches upstream `Generator.remove_weight_norm()`).

    Blocker 8 (2026-08-06): `upsample_kernel_sizes` is RECOVERED FROM
    THE REAL TENSOR SHAPES in `state_dict` (`dec.ups.<i>.weight_v.shape[-1]`)
    rather than trusted from the Task-3 config side-car. Root cause:
    the config's `decoder_upsample_kernel_sizes` is derived by
    `sbv2_prepare_checkpoint.py` via the HiFi-GAN default `kernel = 2 *
    stride` rule (`[16, 16, 4, 4, 4]` for `strides = [8, 8, 2, 2, 2]`),
    but the real SBV2 v2 base checkpoint carries `[16, 16, 8, 2, 2]` —
    a divergence from the paper default. Trusting the config here
    yields a `size mismatch for ups.2.weight_v: copying a param with
    shape torch.Size([128, 64, 8]) from checkpoint, the shape in
    current model is torch.Size([128, 64, 4])` on `load_state_dict`
    (verified 2026-08-06 against `/tmp/sbv2-fixtures/sbv2-prep/G_0.safetensors`).

    `upsample_initial_channel`, `upsample_rates`, `resblock_*` etc.
    remain config-derived — those are decoder-topology globals a config
    side-car legitimately owns; only the per-stage transpose-conv
    kernel widths turned out to be checkpoint-specific.
    """
    from vendor.vits.decoder import Generator

    dec_prefix = "dec."
    dec_state = {
        k[len(dec_prefix):]: v.to(dtype=torch.float32)
        for k, v in state_dict.items()
        if k.startswith(dec_prefix)
    }
    if not dec_state:
        sys.exit(
            f"{LOG_PREFIX} no `dec.*` tensors found in checkpoint — "
            "HiFi-GAN generator cannot be initialized."
        )

    # Recover `upsample_kernel_sizes` from real tensor shapes (Blocker 8):
    # ups.<i>.weight_v [in_channels, out_channels, kernel_size]
    n_ups_stages = len(UPSAMPLE_RATES)
    recovered_kernels: "list[int]" = []
    for i in range(n_ups_stages):
        key = f"ups.{i}.weight_v"
        if key not in dec_state:
            # Some SBV2 SKUs pre-remove weight_norm and ship plain `ups.<i>.weight`.
            key = f"ups.{i}.weight"
            if key not in dec_state:
                sys.exit(
                    f"{LOG_PREFIX} decoder: missing `dec.ups.{i}.weight_v` (or "
                    "`dec.ups.{i}.weight`) — checkpoint does not have the "
                    f"expected {n_ups_stages} upsample stages."
                )
        w = dec_state[key]
        if w.dim() != 3:
            sys.exit(
                f"{LOG_PREFIX} decoder: `dec.ups.{i}.weight_v` shape "
                f"{tuple(w.shape)} unexpected (expected 3D "
                "[in_channels, out_channels, kernel_size])."
            )
        recovered_kernels.append(int(w.shape[-1]))

    config_kernels = list(UPSAMPLE_KERNEL_SIZES)
    if recovered_kernels != config_kernels:
        # Loud but non-fatal: keep the checkpoint's ground truth, and
        # print the diff so a future fix to `sbv2_prepare_checkpoint.py`
        # can be traced.
        print(
            f"{LOG_PREFIX} decoder: `decoder_upsample_kernel_sizes` from config "
            f"side-car {config_kernels} disagrees with real checkpoint tensor "
            f"shapes {recovered_kernels}. Using checkpoint values (Blocker 8, "
            "FR-EX-08 no invention). Fix `sbv2_prepare_checkpoint.py`'s "
            "config resolver to remove this drift."
        )

    # Recover `gin_channels` from `dec.cond.weight` shape (same Blocker 8
    # pattern as `upsample_kernel_sizes`, extended 2026-08-07): the real
    # SBV2 v2 base checkpoint has `dec.cond.weight` shape
    # `[out_channels, gin_channels, kernel_size] = [512, 512, 1]`
    # (raw d_speaker widened to 512 for the multi-speaker path), while the
    # vanilla VITS default `D_SPEAKER = 256` gives a `[512, 256, 1]`
    # Conv1d — a size mismatch on `load_state_dict`. Trust the checkpoint
    # over the config default (FR-EX-08 no invention). If `cond.weight`
    # is absent (single-speaker VITS SKU with no cond conv), fall back to
    # 0 which upstream Generator treats as "no gin".
    cond_key = "cond.weight"
    if cond_key in dec_state:
        cond_w = dec_state[cond_key]
        if cond_w.dim() != 3:
            sys.exit(
                f"{LOG_PREFIX} decoder: `dec.cond.weight` shape "
                f"{tuple(cond_w.shape)} unexpected (expected 3D "
                "[out_channels, gin_channels, kernel_size])."
            )
        recovered_gin = int(cond_w.shape[1])
    else:
        recovered_gin = 0

    if recovered_gin != D_SPEAKER:
        print(
            f"{LOG_PREFIX} decoder: `gin_channels` from `dec.cond.weight` "
            f"shape ({recovered_gin}) disagrees with config default "
            f"D_SPEAKER={D_SPEAKER}. Using checkpoint value (Blocker 8 "
            "pattern extended, FR-EX-08 no invention)."
        )

    gen = Generator(
        initial_channel=D_MODEL,
        resblock=RESBLOCK_TYPE,
        resblock_kernel_sizes=list(RESBLOCK_KERNEL_SIZES),
        resblock_dilation_sizes=_expand_dilations(),
        upsample_rates=list(UPSAMPLE_RATES),
        upsample_initial_channel=UPSAMPLE_INITIAL_CHANNEL,
        upsample_kernel_sizes=recovered_kernels,
        gin_channels=recovered_gin,
    ).eval()

    with torch.no_grad():
        missing_keys, _unexpected = gen.load_state_dict(dec_state, strict=False)
    if missing_keys:
        sys.exit(
            f"{LOG_PREFIX} generator is missing {len(missing_keys)} tensor(s) "
            f"after loading from `dec.*`: "
            f"{missing_keys[:8]}{'...' if len(missing_keys) > 8 else ''}. "
            "Rewriting composition here would be architecture-guessing "
            "(FR-EX-08); vendor upstream `models.py` `Generator` topology "
            "diverges — inspect the checkpoint. Common cause: SBV2 checkpoint "
            "carries `weight_g`/`weight_v` (weight_norm decomposed) while the "
            "vendored Generator was pre-`remove_weight_norm` — compose "
            "weight = weight_g * weight_v / L2-norm(weight_v) before loading."
        )
    gen.remove_weight_norm()
    return gen


def _expand_dilations() -> "list[list[int]]":
    """Reconstruct the list-of-lists shape upstream `Generator` expects
    from Task-3's flattened `(counts, flat)` representation. Loud
    (SystemExit) if the two arrays disagree on total length."""
    if RESBLOCK_DILATION_COUNTS is None or RESBLOCK_DILATIONS_FLAT is None:
        sys.exit(
            f"{LOG_PREFIX} _resolve_arch_constants() was not called before "
            "_expand_dilations() — call order bug in run_pipeline_body."
        )
    total = sum(RESBLOCK_DILATION_COUNTS)
    if total != len(RESBLOCK_DILATIONS_FLAT):
        sys.exit(
            f"{LOG_PREFIX} decoder_resblock_dilation_counts sum ({total}) != "
            f"decoder_resblock_dilations_flat len ({len(RESBLOCK_DILATIONS_FLAT)}) "
            "— Task-3 config side-car is inconsistent."
        )
    result: "list[list[int]]" = []
    idx = 0
    for c in RESBLOCK_DILATION_COUNTS:
        result.append(list(RESBLOCK_DILATIONS_FLAT[idx : idx + c]))
        idx += c
    return result


def run_generator(gen, z_latent, g, torch):
    """Step 13b. `z_latent`: [T_mel, D_MODEL]. Returns waveform:
    [samples] (1-D)."""
    z_bt = z_latent.transpose(0, 1).unsqueeze(0)  # [1, D, T_mel]
    audio = gen(z_bt, g=g)  # [1, 1, samples]
    return audio.squeeze(0).squeeze(0)  # [samples]


def write_f32_bin(path, tensor, torch) -> None:
    """Step 14 helper. Design doc §10: raw little-endian float32, NOT
    `numpy.save`'s `.npy` format (the Rust reader `read_f32_bin` in
    `parity_sbv2_real.rs` / `parity.rs` cannot parse `.npy`).
    """
    import numpy as np

    arr = tensor.detach().to("cpu").contiguous().numpy().astype("<f4")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(arr.tobytes())


def write_u16_bin(path, values) -> None:
    """Task 7 phonemize_fixture.phoneme_ids.bin — u16 little-endian.
    Matches Rust `PhonemizeResult::phoneme_ids` (`Vec<u16>`, read via
    `u16::from_le_bytes`)."""
    import numpy as np

    arr = np.asarray(values, dtype="<u2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(arr.tobytes())


def write_u8_bin(path, values) -> None:
    """Task 7 phonemize_fixture.{tones,word_boundaries}.bin — plain u8.
    """
    import numpy as np

    arr = np.asarray(values, dtype="u1")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(arr.tobytes())


def run_pipeline_body(args: argparse.Namespace, torch, transformers) -> int:
    """Design doc §7 forward-pass pipeline, all 15 steps. Called from
    `run_dump()` after every dependency tier has passed. On success,
    writes 11 tensor `.bin` files, or 12 for ZH, + 4 fixture side files
    (phoneme_ids/tones/word_boundaries all `[T_text]`, plus language_id
    scalar `[1]` — M6 addition) + a fully-resolved
    `reference_dump.manifest.json` to `args.output_dir`. On any failure,
    raises loudly — NEVER silently returns 0 or writes a partial fixture
    (FR-EX-08).
    """
    # ---- Step 0: torch reproducibility ----
    prepare_torch(args, torch)

    # ---- Step 1: SBV2 v2 checkpoint (safetensors + Task-3 config) ----
    config, state_dict = load_sbv2_checkpoint(args)
    print(f"{LOG_PREFIX} loaded SBV2 v2 checkpoint: {len(state_dict)} tensors")

    # ---- Step 2: DeBERTa v2 (JA) + DeBERTa v3 (EN) + BERT (ZH) ----
    (tok_ja, model_ja), (tok_en, model_en), zh_pair = load_bert_encoders(
        args, transformers
    )
    loaded = "DeBERTa v2 (JA) + v3 (EN)"
    if zh_pair is not None:
        loaded += " + plain BERT (ZH)"
    print(f"{LOG_PREFIX} loaded {loaded}")

    # ---- Step 3: G2P (fixture-only per Task 7) ----
    g2p = MinimalG2P()
    phon = g2p.phonemize(args.text, args.language)
    t_text = len(phon["phoneme_ids"])
    print(f"{LOG_PREFIX} G2P: T_text = {t_text}")

    # ---- Step 4: SBV2 text encoder (VITS TextEncoder + tone/language) ----
    encoder = build_text_encoder(state_dict, torch)
    tone_emb, lang_emb = build_sbv2_extras(state_dict, torch)
    # Resolve per-utterance `language_id: u8` from `--language`. Matches
    # `crates/vokra-models/src/sbv2/g2p.rs` `Language::language_id` ordering
    # (`JA = 0`, `EN = 1`, `ZH = 2`) 1:1.
    _LANGUAGE_ID_BY_CLI: "dict[str, int]" = {"ja": 0, "en": 1, "zh": 2}
    language_id = _LANGUAGE_ID_BY_CLI[args.language.lower()]
    phoneme_embed, text_hidden, x_mask_text = run_text_encoder(
        encoder, tone_emb, lang_emb,
        phon["phoneme_ids"], phon["tones"], language_id,
        torch,
    )
    print(f"{LOG_PREFIX} text encoder: phoneme_embed {tuple(phoneme_embed.shape)}, "
          f"text_hidden {tuple(text_hidden.shape)}, language_id={language_id}")

    # ---- Steps 5+6: JA/EN always; ZH on the explicit four-file leg ----
    bert_ja = run_bert(tok_ja, model_ja, args.text, torch)  # [T_bert_ja, H_ja]
    bert_en = run_bert(tok_en, model_en, args.text, torch)  # [T_bert_en, H_en]
    bert_zh = None
    if zh_pair is not None:
        tok_zh, model_zh = zh_pair
        bert_zh = run_bert(tok_zh, model_zh, args.text, torch)
    bert_shapes = (
        f"bert_ja {tuple(bert_ja.shape)}, bert_en {tuple(bert_en.shape)}"
    )
    if bert_zh is not None:
        bert_shapes += f", bert_zh {tuple(bert_zh.shape)}"
    print(f"{LOG_PREFIX} {bert_shapes}")

    # ---- Step 7: BertBridge (active language only feeds text_hidden) ----
    active_by_language = {"JA": bert_ja, "EN": bert_en}
    if bert_zh is not None:
        active_by_language["ZH"] = bert_zh
    active_bert = active_by_language[args.language.upper()]
    bridge = BertBridge(state_dict, args.language, torch)
    text_hidden_transposed = text_hidden.transpose(0, 1)  # [D_MODEL, T_text]
    bert_bridge_out = bridge.forward(
        active_bert, text_hidden_transposed, t_text, torch
    )
    print(f"{LOG_PREFIX} bert_bridge_out {tuple(bert_bridge_out.shape)}")

    # ---- Step 8: speaker embedding ----
    speaker_embed = run_speaker_embedding(state_dict, args.speaker_id, torch)
    print(f"{LOG_PREFIX} speaker_embed {tuple(speaker_embed.shape)}")

    # ---- Step 9: style projection ----
    # The `--style-dim` argparse default is a hard-coded 256, but the real
    # checkpoint's `vokra-sbv2-config.json` reports `d_style` (128 for the
    # SBV2 v2 base). If the user did NOT override `--style-dim` from the
    # CLI, the config-resolved `D_STYLE_DEFAULT` (populated in
    # `_resolve_arch_constants` from the config side-car) is the truth —
    # use it. Overrides via `--style-dim <N>` are honored verbatim.
    effective_style_dim = (
        D_STYLE_DEFAULT if args.style_dim == DEFAULT_STYLE_DIM else args.style_dim
    )
    style_vec = torch.zeros(1, effective_style_dim, dtype=torch.float32)
    style_injector = StyleVectorInjector(
        state_dict, torch, d_model=D_MODEL, d_style=effective_style_dim,
    )
    style_projected = style_injector.forward(style_vec, torch)
    print(
        f"{LOG_PREFIX} style_projected {tuple(style_projected.shape)} "
        f"(d_style={effective_style_dim} from config)"
    )

    # Combined speaker/style conditioning [1, D_SPEAKER, 1] for flow + SDP.
    # (Style is projected to D_MODEL, additively broadcast into speaker
    # only if the two share dimensions — else summed pre-projection in
    # the checkpoint. This dumper uses speaker-only conditioning for
    # flow/decoder inputs, style is dumped for the manifest slot per
    # design doc §10 but not otherwise mixed here — a follow-up per the
    # checkpoint's own `g_style` wiring, honestly reported.)
    g_cond = speaker_embed.unsqueeze(-1)  # [1, D_SPEAKER, 1]

    # ---- Step 10: SDP sample ----
    # SDP conditioning: takes text_hidden (transposed to [1, D, T]).
    sdp = SDPReference(state_dict, filter_channels=D_MODEL, torch=torch)

    # RNG PARITY (2026-08-09, workflow_dispatch b459e5b bisect): reset the
    # global torch RNG right before SDP.sample so its noise buffer matches
    # Rust's `TorchRandnStream::new(seed)` — a FRESH stream at seed=`args.seed`.
    #
    # Without this reset, the dumper's earlier model-construction and load
    # steps (nn.Module `__init__` runs Kaiming init on every Conv/Linear/
    # LayerNorm, drawing from `torch.rand()` / `torch.randn()`) consume an
    # UNSPECIFIED number of RNG samples between `prepare_torch`'s
    # `torch.manual_seed(seed)` and this point. That leaves SDP.sample
    # drawing from a state Rust has no way to reproduce (Rust reads GGUF
    # weights directly with no init phase), so the "sdp_sample.bin"
    # committed values diverged wildly from Rust's `SbV2SDP::sample`
    # output — even though `TorchRandnStream::new(seed)` is a bit-exact
    # port of torch's `at::normal_distribution<double>` (verified by
    # `crates/vokra-core/tests/rng_torch_randn_cpu_parity.rs`).
    #
    # Empirical evidence from the bisect (M1 iMac, seed=42, "テスト" 8
    # phonemes): pre-fix Python durs=[2, 4, 28, 1, 13, 4, 6, 5] sum=63 vs
    # Rust durs=[3, 3, 13, 4, 5, 1, 19, 5] sum=53 (delta 10 frames = 16%
    # length variance, breaks the ±10% waveform band). Post-fix Python
    # durs=[3, 3, 13, 4, 5, 1, 19, 5] sum=53 — bit-exact match to Rust.
    torch.manual_seed(args.seed)

    sdp_sample = sdp.sample(
        text_hidden_transposed.unsqueeze(0),  # [1, D_MODEL, T_text]
        x_mask_text,
        g_cond,
        args.noise_scale_w,
        torch,
    )
    # Optional --speed scaling: durations /= speed (upstream SBV2 pattern).
    sdp_sample = sdp_sample / args.speed
    print(f"{LOG_PREFIX} sdp_sample {tuple(sdp_sample.shape)} sum={float(sdp_sample.sum()):.1f}")

    # ---- Step 11: length regulator ----
    mel_hidden = length_regulate(bert_bridge_out, sdp_sample, torch)
    t_mel = mel_hidden.shape[0]
    x_mask_mel = torch.ones(1, 1, t_mel)
    print(f"{LOG_PREFIX} mel_hidden {tuple(mel_hidden.shape)} (T_mel={t_mel})")

    # `dump_dir` is hoisted ahead of Step 12 (was previously first defined
    # at Step 14) so Step 12's per-layer flow instrumentation (Task 15,
    # Wave 3) can write `flow_layer_<i>_output.bin` files as the flow
    # itself runs, not only in a final batch at the end.
    dump_dir = args.output_dir / "reference_dump"
    dump_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 12: flow reverse (Task 15, Wave 3: per-layer intermediates) ----
    flow = build_flow(state_dict, torch)
    z_latent, n_flow_layers_dumped = run_flow(
        flow, mel_hidden, x_mask_mel, g_cond, args.noise_scale, torch, args.seed,
        dump_dir=dump_dir,
    )
    print(
        f"{LOG_PREFIX} z_latent {tuple(z_latent.shape)} "
        f"({n_flow_layers_dumped} flow_layer_*_output.bin written)"
    )

    # ---- Step 13: HiFi-GAN ----
    gen = build_generator(state_dict, torch)
    waveform = run_generator(gen, z_latent, g_cond, torch)
    samples = int(waveform.shape[0])
    print(f"{LOG_PREFIX} waveform {tuple(waveform.shape)} ({samples} samples @ {SAMPLE_RATE} Hz)")

    # ---- Step 14: write all bin files ----
    # (`dump_dir` was hoisted ahead of Step 12 above.)
    write_f32_bin(dump_dir / "phoneme_embed.bin",   phoneme_embed,   torch)
    write_f32_bin(dump_dir / "text_hidden.bin",     text_hidden,     torch)
    write_f32_bin(dump_dir / "bert_hidden_ja.bin",  bert_ja,         torch)
    write_f32_bin(dump_dir / "bert_hidden_en.bin",  bert_en,         torch)
    if bert_zh is not None:
        write_f32_bin(dump_dir / "bert_hidden_zh.bin", bert_zh, torch)
    write_f32_bin(dump_dir / "bert_bridge_out.bin", bert_bridge_out, torch)
    write_f32_bin(dump_dir / "speaker_embed.bin",   speaker_embed,   torch)
    write_f32_bin(dump_dir / "style_projected.bin", style_projected, torch)
    write_f32_bin(dump_dir / "sdp_sample.bin",      sdp_sample,      torch)
    write_f32_bin(dump_dir / "mel_hidden.bin",      mel_hidden,      torch)
    write_f32_bin(dump_dir / "z_latent.bin",        z_latent,        torch)
    write_f32_bin(dump_dir / "waveform.bin",        waveform.unsqueeze(0), torch)

    # Task 7 side files (G2P inputs, replayed by Rust `from_fixture`).
    # `word_boundaries.bin` is retained even though the M6 refactor removed
    # its consumer from the text encoder — Rust `PhonemizeResult` still
    # carries the field for fixture stability, and dropping the file would
    # break any parity fixture pinned pre-M6.
    write_u16_bin(dump_dir / "phoneme_ids.bin",     phon["phoneme_ids"])
    write_u8_bin(dump_dir / "tones.bin",            phon["tones"])
    write_u8_bin(dump_dir / "word_boundaries.bin",  phon["word_boundaries"])
    # M6 addition: per-utterance `language_id` (u8 scalar, count 1).
    # Matches `SbV2TextEncoder::forward`'s `language_id: u8` argument.
    write_u8_bin(dump_dir / "language_id.bin",      [language_id])

    # ---- Step 15: fully-resolved manifest ----
    tensor_shapes = {
        "phoneme_embed":   [t_text, D_MODEL],
        "text_hidden":     [t_text, D_MODEL],
        "bert_hidden_ja":  list(bert_ja.shape),
        "bert_hidden_en":  list(bert_en.shape),
        "bert_bridge_out": [t_text, D_MODEL],
        "speaker_embed":   list(speaker_embed.shape),
        "style_projected": list(style_projected.shape),
        "sdp_sample":      [t_text],
        "mel_hidden":      [t_mel, D_MODEL],
        "z_latent":        [t_mel, D_MODEL],
        "waveform":        [1, samples],
    }
    if bert_zh is not None:
        tensor_shapes["bert_hidden_zh"] = list(bert_zh.shape)

    manifest = build_manifest(
        args,
        tensor_shapes=tensor_shapes,
        phonemize_counts={
            "phoneme_ids":     t_text,
            "tones":           t_text,
            "word_boundaries": t_text,
            "language_id":     1,   # M6 addition: per-utterance u8 scalar
        },
        # Task 15 (Wave 3): real per-layer flow-intermediate count/shape,
        # discovered by `run_flow` above — never hard-coded, mirrors how
        # `tensor_shapes`/`phonemize_counts` above are always real values
        # here (schema-preview's symbolic fallback only applies in
        # `run_preview`, which never reaches this real-dump function).
        flow_layer_count=n_flow_layers_dumped,
        flow_layer_shape=[t_mel, D_MODEL],
    )
    manifest_path = args.output_dir / "reference_dump.manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=False)

    print(
        f"{LOG_PREFIX} OK: wrote {len(tensor_shapes)} tensor .bin + 4 fixture .bin + "
        f"{n_flow_layers_dumped} flow_layer_*_output.bin + manifest "
        f"to {args.output_dir}"
    )
    return 0


def run_dump(args: argparse.Namespace) -> int:
    """Real-dump mode. Fails loudly at a missing dependency, staged
    checkpoint tensor, or G2P row; never substitutes a synthetic value."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.checkpoint.is_dir():
        sys.exit(
            f"{LOG_PREFIX} --checkpoint {args.checkpoint} is not a directory — "
            "point it at a tools/parity/sbv2_prepare_checkpoint.py --output-dir "
            "(containing vokra-sbv2-config.json + the downloaded .safetensors)."
        )
    print(
        f"{LOG_PREFIX} --do-dump: attempting a real SBV2 v2 forward pass "
        f"(checkpoint={args.checkpoint}, language={args.language}, "
        f"output-dir={args.output_dir})"
    )

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        sys.exit(
            f"{LOG_PREFIX} missing Python dep ({exc}); install with "
            "`cd tools/parity && uv sync` on the VAST instance, then rerun "
            "through `uv run python`."
        )

    try:
        import transformers
    except ImportError as exc:
        sys.exit(
            f"{LOG_PREFIX} missing Python dep ({exc}); install with "
            "`cd tools/parity && uv sync` on the VAST instance — "
            "transformers is this project's authorized DeBERTa v2/v3 "
            "reference (design doc §6)."
        )
    print(f"{LOG_PREFIX} torch {torch.__version__}, transformers {transformers.__version__} present.")

    # jaywalnut310/vits (MIT) core — vendored piecemeal into
    # tools/parity/vendor/vits/, not pip-installable (see that directory's
    # README). Insert this script's own directory onto sys.path so
    # `vendor.vits` resolves as a plain top-level namespace package (no
    # __init__.py needed anywhere in this chain — verified empirically:
    # Python's implicit-namespace-package machinery, PEP 420, resolves
    # `vendor.vits` cleanly even with zero .py files in either directory,
    # and names the precise missing submodule in the resulting ImportError).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from vendor.vits import text_encoder as _unused  # noqa: F401
        from vendor.vits import coupling as _unused_coupling  # noqa: F401
        from vendor.vits import flow as _unused_flow  # noqa: F401
        from vendor.vits import decoder as _unused_decoder  # noqa: F401
    except ImportError as exc:
        sys.exit(
            f"{LOG_PREFIX} jaywalnut310/vits (MIT) vendor import failed "
            f"({exc}). Task 4 landed the 8-module vendor pass into "
            "tools/parity/vendor/vits/ (text_encoder.py, coupling.py, "
            "flow.py, decoder.py + supporting attentions.py, commons.py, "
            "modules.py, transforms.py) — see that directory's README for "
            "the full mapping table. If this error surfaces you either "
            "(a) hit a torch API drift the vendor did not anticipate "
            "(check the DeprecationWarning trail above), (b) the parity "
            "venv is missing a transitive dep the vendored modules pull "
            "in (e.g. numpy — scipy is deliberately dropped, see "
            "vendor/vits/README.md), or (c) the vendored files themselves "
            "drifted from what their headers claim (rerun the sha256 diff "
            "vs upstream at the pinned commit)."
        )
    print(f"{LOG_PREFIX} vendor.vits import OK (Task 4 vendoring at pinned commit).")

    # p0p4k/vits2_pytorch (MIT) — Blocker 2b clean-room vendor (Task 1-2,
    # tools/parity/vendor/vits2/), a SEPARATE vendor tree from vendor.vits
    # above (same sys.path insert already covers it — both are direct
    # children of this script's own directory). NOT used to build the
    # real per-checkpoint flow: see `run_flow`'s docstring for why (its
    # `TransformerCouplingLayer.enc` is WN-based, layout-incompatible
    # with the real SBV2 v2 base checkpoint's attention-based
    # `flow.flows.<i>.enc.attn_layers.*` tensor names). Imported here —
    # at the same dependency-gate tier as vendor.vits, tier 3 in this
    # file's module doc — only so a corrupted/missing vendor/vits2 tree
    # fails loud (FR-EX-08) up front, rather than as a bare unguarded
    # import deep inside `run_flow` the first time Task 16 actually
    # exercises this path.
    try:
        from vendor.vits2 import attentions as vits2_attentions
        from vendor.vits2 import models as vits2_models
    except ImportError as exc:
        sys.exit(
            f"{LOG_PREFIX} p0p4k/vits2_pytorch (MIT) vendor import failed "
            f"({exc}). Task 1-2 (SBV2 v2 Blocker 2b/2c/3 plan) vendored "
            "attentions.py + models.py into tools/parity/vendor/vits2/ — "
            "see that directory's README for the sha256 verification "
            "recipe and pinned commit. Same failure modes as the "
            "vendor.vits gate above (torch API drift / missing "
            "transitive dep / vendored-file corruption)."
        )
    if not hasattr(vits2_attentions, "Encoder") or not hasattr(vits2_models, "TransformerCouplingLayer"):
        sys.exit(
            f"{LOG_PREFIX} vendor/vits2/{{attentions,models}}.py imported but "
            "missing an expected class (Encoder / TransformerCouplingLayer) "
            "— vendor tree looks corrupted. Re-run the sha256 verification "
            "in tools/parity/vendor/vits2/README.md."
        )
    print(f"{LOG_PREFIX} vendor.vits2 import OK (Task 1-2 vendoring, Blocker 2b).")

    # ------------------------------------------------------------------
    # All dependency tiers have passed (torch / transformers /
    # vendor.vits + vendor.vits2 import / this dumper's own architectural
    # body). Hand off to the pipeline body, which drives all 15 steps
    # documented in `run_pipeline_body`'s docstring. On success, writes
    # 11 tensor .bin (12 for ZH) + 4 fixture .bin
    # (phoneme_ids/tones/word_boundaries
    # + M6 language_id) + N flow_layer_<i>_output.bin (Task 15, Wave 3) +
    # reference_dump.manifest.json to `args.output_dir`. On failure,
    # raises loudly — NEVER silently returns 0 or writes a partial
    # fixture (FR-EX-08).
    # ------------------------------------------------------------------
    return run_pipeline_body(args, torch, transformers)


def parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SBV2 v2 clean-room reference dumper: runs the Style-Bert-VITS2 "
            "v2 forward pass with HF transformers (Apache-2.0) DeBERTa "
            "encoders + a vendored jaywalnut310/vits (MIT) VITS core, and "
            "dumps the 11 intermediate tensors design doc §10 pins for "
            "crates/vokra-models/tests/parity_sbv2_real.rs (Task 28) to diff "
            "a Rust forward pass against. See this script's module docstring "
            "for the manifest schema, real-dump preconditions, "
            "and the clean-room NOT-REFERENCED list."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help=(
            "Path to a prepared SBV2 v2 checkpoint directory — the "
            "--output-dir a prior tools/parity/sbv2_prepare_checkpoint.py "
            "run produced (vokra-sbv2-config.json + .safetensors). Only "
            "inspected in --do-dump mode."
        ),
    )
    parser.add_argument(
        "--text",
        default=None,
        help=(
            "Text to synthesize. Default depends on --language: "
            f"{DEFAULT_TEXT_BY_LANGUAGE['ja']!r} for ja, "
            f"{DEFAULT_TEXT_BY_LANGUAGE['en']!r} for en, and "
            f"{DEFAULT_TEXT_BY_LANGUAGE['zh']!r} for zh."
        ),
    )
    parser.add_argument(
        "--language",
        choices=sorted(DEFAULT_TEXT_BY_LANGUAGE),
        default="ja",
        help="Which G2P + BERT path to route through (default: ja).",
    )
    parser.add_argument(
        "--speaker-id",
        type=int,
        default=0,
        help="Discrete speaker id, looked up in the SBV2 speaker embedding table (default: 0).",
    )
    parser.add_argument(
        "--style-dim",
        type=int,
        default=DEFAULT_STYLE_DIM,
        help=(
            "Length of the (all-zero, i.e. identity) style vector to use "
            f"(default: {DEFAULT_STYLE_DIM}, design doc §7's 'style vector' "
            "hparam)."
        ),
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_SPEED,
        help="Duration speed multiplier; must be > 0 (default: 1.0).",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=DEFAULT_NOISE_SCALE,
        help=f"Flow-latent noise scale (default: {DEFAULT_NOISE_SCALE}, design doc §7).",
    )
    parser.add_argument(
        "--noise-scale-w",
        type=float,
        default=DEFAULT_NOISE_SCALE_W,
        help=f"SDP noise scale (default: {DEFAULT_NOISE_SCALE_W}, design doc §7).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="PRNG seed for the SDP's Gaussian draws (default: 42).",
    )
    parser.add_argument(
        "--bert-ja-repo",
        default=DEFAULT_BERT_JA_REPO,
        help=f"HF transformers repo id for the JA BERT encoder (default: {DEFAULT_BERT_JA_REPO}).",
    )
    parser.add_argument(
        "--bert-en-repo",
        default=DEFAULT_BERT_EN_REPO,
        help=f"HF transformers repo id for the EN BERT encoder (default: {DEFAULT_BERT_EN_REPO}).",
    )
    parser.add_argument(
        "--bert-zh-repo",
        default=DEFAULT_BERT_ZH_REPO,
        help=f"HF transformers repo id for the ZH BERT encoder (default: {DEFAULT_BERT_ZH_REPO}).",
    )
    parser.add_argument(
        "--sbv2-main-filename",
        default=DEFAULT_SBV2_MAIN_FILENAME,
        help=f"Manifest checkpoint.sbv2_main filename (default: {DEFAULT_SBV2_MAIN_FILENAME}).",
    )
    parser.add_argument(
        "--bert-ja-filename",
        default=DEFAULT_BERT_JA_FILENAME,
        help=f"Manifest checkpoint.bert_ja filename (default: {DEFAULT_BERT_JA_FILENAME}).",
    )
    parser.add_argument(
        "--bert-en-filename",
        default=DEFAULT_BERT_EN_FILENAME,
        help=f"Manifest checkpoint.bert_en filename (default: {DEFAULT_BERT_EN_FILENAME}).",
    )
    parser.add_argument(
        "--bert-zh-filename",
        default=DEFAULT_BERT_ZH_FILENAME,
        help=f"Manifest checkpoint.bert_zh filename (default: {DEFAULT_BERT_ZH_FILENAME}).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "Where reference_dump.manifest.json + reference_dump/*.bin get "
            "written in --do-dump mode. Unused (never touched) in the "
            "default schema-preview mode."
        ),
    )
    parser.add_argument(
        "--do-dump",
        action="store_true",
        help=(
            "Attempt a real forward pass instead of the default schema "
            "preview. VAST-only; fails loudly if its uv dependencies, staged "
            "checkpoint, or exact G2P row are absent — see module docstring."
        ),
    )
    args = parser.parse_args(argv)

    if args.speed <= 0:
        parser.error("--speed must be > 0 (SbV2Model::synthesize's own precondition)")
    if args.speaker_id < 0:
        parser.error("--speaker-id must be >= 0 (u32 on the Rust side)")
    if args.seed < 0:
        parser.error("--seed must be >= 0 (u64 on the Rust side)")
    if args.style_dim < 0:
        parser.error("--style-dim must be >= 0")
    if args.text is None:
        args.text = DEFAULT_TEXT_BY_LANGUAGE[args.language]

    return args


def main() -> int:
    args = parse_args()
    if not args.do_dump:
        return run_preview(args)
    return run_dump(args)


if __name__ == "__main__":
    sys.exit(main())
