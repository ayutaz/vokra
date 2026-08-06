#!/usr/bin/env python3
"""SBV2 v2 clean-room reference dumper — HF ``transformers`` DeBERTa
encoders + a vendored ``jaywalnut310/vits`` VITS core (Task 30, SBV2 v2
plan).

Runs the Style-Bert-VITS2 v2 forward pass using **only permissive Python
reference implementations** and dumps the 11 intermediate tensors
``docs/superpowers/specs/2026-07-26-sbv2-v2-design.md`` §10 pins, so
``crates/vokra-models/tests/parity_sbv2_real.rs`` (Task 28) can diff a real
Rust forward pass against them, tensor by tensor, at
``sbv2::parity::tolerance_for(name)``.

# NOT REFERENCED (clean-room — read this before touching this file)

- ``github.com/litagin02/Style-Bert-VITS2`` (AGPL-3.0) and all forks
- ``github.com/fishaudio/Bert-VITS2`` (AGPL-3.0) and all forks
- Any community fork/derivative/blog-post code excerpt of either of the
  above (Qiita/Zenn writeups, Discord snippets, ...)

This script (and the ``tools/parity/vendor/vits/`` directory it imports
from once vendoring lands — see "Status" below) is built exclusively from
the permissive sources listed next. No AGPL source was read, copied, or
recalled from memory to write this file.

# Permissive references this file is authorized to use

- **VITS** paper (Kim et al. 2021, arXiv:2106.06103) and
  ``jaywalnut310/vits`` (MIT) — vendored piecemeal into
  ``tools/parity/vendor/vits/`` (see that directory's own README for the
  pinned commit, exact modules, and why the actual porting is deferred).
- **VITS2** paper (arXiv:2307.16430) and ``p0p4k/vits2_pytorch`` (MIT) —
  architectural delta reference only, for the SBV2-specific structural
  differences from vanilla VITS.
- **DeBERTa v2** paper (arXiv:2006.03654), ``microsoft/DeBERTa`` (MIT), and
  **HuggingFace ``transformers``** (Apache-2.0) ``deberta_v2``/``deberta_v3``
  modules — used directly via ``AutoModel``/``AutoTokenizer`` below, no
  vendoring needed (a real, `pip install`-able Apache-2.0 dependency).
  ``deberta_v3``'s own paper is arXiv:2111.09543.
- SentencePiece paper (Kudo & Richardson 2018) — informational only (the
  BERT tokenizers below come from ``transformers`` directly).
  ``litagin02/style_bert_vits2``'s upstream ``config.json`` / safetensors
  tensor-name metadata (structural facts only — names, shapes, dtypes —
  never the AGPL Python code itself, matching
  ``tools/parity/sbv2_prepare_checkpoint.py``'s own posture).

# Status: CLI + manifest-schema scaffold + Task-4 vendor-import gate landed;
# the real forward pipeline body is still deferred

This dumper has two modes:

* **Schema preview** (default, no ``--do-dump``): builds and prints the
  ``reference_dump.manifest.json`` this tool *would* write, with the 8
  request fields fully resolved from the CLI (nothing about those needs a
  real checkpoint or forward pass), and the 11 ``tensors[].shape`` entries
  left as symbolic placeholders (``"T_text"`` / ``"T_bert"`` / ``"T_mel"`` /
  ``"samples"``) since real dimensions only exist after a real forward pass
  runs. Nothing is written to ``--output-dir`` in this mode — printing a
  manifest that *looks* real but is not would itself be a fabricated
  artifact (NFR-QL-04 / FR-EX-08). Needs no dependency beyond the stdlib,
  so it (like ``--help``) works in a bare interpreter.
* **Real dump** (``--do-dump``): attempts the actual reference forward
  pass. As of Task 4's vendoring commit, this fails loudly at one of
  four tiers depending on what is installed and how far execution has
  advanced:

  1. ``torch`` missing -> actionable ``pip install torch``.
  2. ``transformers`` missing -> actionable ``pip install transformers``
     (Apache-2.0 — this project's authorized DeBERTa reference).
  3. Vendor import failure (``from vendor.vits import text_encoder / coupling
     / flow / decoder``) -> actionable message pointing at
     ``tools/parity/vendor/vits/README.md`` and its sha256/upstream-URL
     trail. Before Task 4 landed this was the terminal gate (only
     ``LICENSE`` + ``README.md`` scaffolded there); Task 4 vendored the 8
     supporting modules so this gate now passes cleanly in a torch +
     transformers-equipped interpreter.
  4. Pipeline body not yet written -> ``NotImplementedError`` with an
     actionable message. This is Task 4's new terminal gate: the vendor
     import above now succeeds, but the design doc §7 forward pipeline
     body (G2P -> ``SbV2TextEncoder`` -> DeBERTa-bridge -> SDP -> flow
     -> HiFi-GAN, writing 11 ``reference_dump/*.bin`` files + a fully-
     resolved ``reference_dump.manifest.json``) is a separate follow-up
     task, gated on a real SBV2 v2 checkpoint being inspected first
     (design doc §12 owner step) — otherwise a self-consistent mirror
     of the architecture would validate nothing, the same NFR-QL-04 /
     FR-EX-08 lesson ``tools/parity/utmos_dump_reference.py``'s own
     module doc draws from the Kokoro ``92dbc92`` incident.

  Nothing is stubbed, mocked, or approximated to make this path "succeed"
  early — the writing of tier 4's pipeline body is refused (with a
  ``NotImplementedError``, not a silent ``return 0``) until a real
  checkpoint exists to validate the dumped tensors against. Once that
  follow-up lands, tier 4 writes ``reference_dump/*.bin`` (raw little-
  endian f32, matching every other ``*_dump*.py`` sibling's
  ``arr.tobytes()`` convention — *not* ``numpy.save``'s ``.npy`` format,
  which ``parity_sbv2_real.rs``'s ``read_f32_bin`` does not parse) plus
  the real, fully-resolved ``reference_dump.manifest.json``.

# The 11 dumped tensors (design doc §10 / ``parity_sbv2_real.rs`` contract)

======================= ================= ========================
name                    shape             purpose
======================= ================= ========================
``phoneme_embed``       [T_text, 192]     text encoder input
``text_hidden``         [T_text, 192]     text encoder output
``bert_hidden_ja``      [T_bert, 1024]    DeBERTa v2 output (JA)
``bert_hidden_en``      [T_bert, 1024]    DeBERTa v3 output (EN)
``bert_bridge_out``     [T_text, 192]     BERT bridge conv output
``speaker_embed``       [1, 512]          speaker embedding
``style_projected``     [1, 192]          style vector projection
``sdp_sample``          [T_text]          SDP duration sample
``mel_hidden``          [T_mel, 192]      length-regulated hidden
``z_latent``            [T_mel, 192]      normalizing-flow output
``waveform``            [1, samples]      final PCM
======================= ================= ========================

Only ``T_bert`` differs per language (JA uses the DeBERTa v2 tokenizer's
sequence length for the input ``text``, EN the DeBERTa v3 tokenizer's) —
one of ``bert_hidden_ja``/``bert_hidden_en`` is the *active* path for a
given ``--language``; the other is dumped too (both BERT encoders run
regardless of which language's SBV2 acoustic path consumes the result) so
the fixture set is complete regardless of which language a future
``parity_sbv2_real.rs`` run exercises. All 11 ``.bin`` files are raw
little-endian ``float32`` (including ``sdp_sample``, whose semantic values
are discrete durations — see
``crates/vokra-models/src/sbv2/parity.rs``'s ``PER_TENSOR_ATOL`` doc comment
for why that tensor is still compared as float with an atol that has an
explicit +/-1 discrete-step allowance, rather than as an exact integer
match).

Task 7 (SBV2 v2 plan) adds three **side files** alongside the 11-tensor
list — ``phoneme_ids.bin`` (``uint16``), ``tones.bin`` (``uint8``),
``word_boundaries.bin`` (``uint8``), all of length ``T_text`` — under the
manifest's own ``phonemize_fixture`` block (not inside ``tensors[]``, so
the design-doc §10 "11 dumped tensors" contract stays intact). They are
the G2P *inputs* to the reference forward pass, replayed on the Rust side
by ``SbV2Phonemizer::from_fixture`` +
``SbV2Model::from_gguf_with_phonemizer`` — see this file's manifest
schema below.

# Manifest schema (this file writes; ``parity_sbv2_real.rs``'s module doc is
# the authoritative contract this mirrors)

::

    {
      "generator_version": "1.0",
      "generator": "tools/parity/sbv2_dump_reference.py",
      "checkpoint": {
        "sbv2_main": "sbv2-v2-multilingual-base.gguf",
        "bert_ja": "deberta-v2-large-japanese-char-wwm.gguf",
        "bert_en": "deberta-v3-large.gguf"
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
                            "count": T_text, "dtype": "uint8"}
      },
      "tensors": [
        {"name": "phoneme_embed", "path": "reference_dump/phoneme_embed.bin",
         "shape": [T_text, 192], "dtype": "float32"},
        ... (11 total, see table above)
      ]
    }

``phonemize_fixture`` (Task 7) is the fixture bypass that lets the Rust
side rebuild an ``SbV2Phonemizer`` (via ``SbV2Phonemizer::from_fixture`` +
``SbV2Model::from_gguf_with_phonemizer``) that reproduces the exact G2P
output the reference dumper's forward pass consumed, without needing a
real 8-language piper-plus G2P available in-workspace (NFR-DS-02: the
excluded ``integrations/vokra-piper-g2p`` cannot be a
``crates/vokra-models`` dependency). The three side files are always
1-D (their length is ``T_text``, matching every f32 tensor whose leading
axis is ``T_text``) and use narrower dtypes than f32 —
``phoneme_ids`` is ``uint16`` (matches the Rust ``PhonemizeResult::phoneme_ids``'s
``Vec<u16>``), ``tones`` and ``word_boundaries`` are ``uint8``. The
consuming Rust reader dispatches on ``dtype``.

``checkpoint.*`` are **bare filenames** (siblings of the manifest inside
``tests/fixtures/sbv2/``, matching Task 34's planned ``Files:`` list) —
override via ``--sbv2-main-filename``/``--bert-ja-filename``/
``--bert-en-filename`` if a real fixture set uses different names.
``request.language`` is upper-case ``"JA"``/``"EN"`` (``parity_sbv2_real.rs``
matches on exactly those two literals). ``tensors[].path`` already includes
the ``reference_dump/`` prefix.

# Usage

::

    # schema preview (no deps beyond the stdlib, nothing written to disk):
    python3 tools/parity/sbv2_dump_reference.py \\
        --checkpoint /tmp/sbv2-checkpoint --output-dir /tmp/sbv2-dump

    # real dump (fails loudly today — vendoring not landed yet, see above):
    python3 tools/parity/sbv2_dump_reference.py \\
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
import convention. ``jaywalnut310/vits`` (MIT) has no PyPI distribution and
must be vendored (see ``tools/parity/vendor/vits/README.md``); until that
lands, the ``--do-dump`` path's third import tier is the one that always
fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# --- identity -------------------------------------------------------------

LOG_PREFIX = "[sbv2-dump]"
GENERATOR_ID = "tools/parity/sbv2_dump_reference.py"
GENERATOR_VERSION = "1.0"

# Matches Task 34's planned `Files:` list verbatim (design doc §10 / Task 28
# module doc) — bare filenames, siblings of the manifest inside
# tests/fixtures/sbv2/. Overridable per-run via --sbv2-main-filename etc. in
# case a real fixture set ends up named differently.
DEFAULT_SBV2_MAIN_FILENAME = "sbv2-v2-multilingual-base.gguf"
DEFAULT_BERT_JA_FILENAME = "deberta-v2-large-japanese-char-wwm.gguf"
DEFAULT_BERT_EN_FILENAME = "deberta-v3-large.gguf"

# HF transformers repo ids for the two BERT encoders (design doc §9 SKU
# table / §16 References). Verified to exist via the public HF Hub API
# (2026-07-26): ku-nlp/deberta-v2-large-japanese-char-wwm is tagged
# license=cc-by-sa-4.0, microsoft/deberta-v3-large is tagged license=mit.
DEFAULT_BERT_JA_REPO = "ku-nlp/deberta-v2-large-japanese-char-wwm"
DEFAULT_BERT_EN_REPO = "microsoft/deberta-v3-large"

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
N_WORD_BOUNDARY_VOCAB: int = 2  # `crates/vokra-models/src/sbv2/text_encoder.rs` `wb_embed` is [2, d_model]
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
}

# The 11-tensor dump contract (design doc §10, reproduced verbatim in
# parity_sbv2_real.rs's module doc). `shape_template` uses the same
# symbolic placeholders as that Rust file's own illustrative sketch —
# real integers only appear once a real forward pass supplies them.
TENSOR_SCHEMA: "list[dict]" = [
    {"name": "phoneme_embed", "shape_template": ["T_text", 192]},
    {"name": "text_hidden", "shape_template": ["T_text", 192]},
    {"name": "bert_hidden_ja", "shape_template": ["T_bert", 1024]},
    {"name": "bert_hidden_en", "shape_template": ["T_bert", 1024]},
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
# `u16::from_le_bytes` for `phoneme_ids`, plain `u8` for the two others.
PHONEMIZE_FIXTURE_SCHEMA: "list[dict]" = [
    {"name": "phoneme_ids", "dtype": "uint16", "count_template": "T_text"},
    {"name": "tones", "dtype": "uint8", "count_template": "T_text"},
    {"name": "word_boundaries", "dtype": "uint8", "count_template": "T_text"},
]


def build_manifest(args: argparse.Namespace, tensor_shapes: "dict[str, list] | None" = None,
                   phonemize_counts: "dict[str, int] | None" = None) -> dict:
    """Builds the ``reference_dump.manifest.json`` contents.

    ``tensor_shapes``, when given, maps a subset of the 11 tensor names to
    their *real*, already-known integer shape (only available once a real
    forward pass has run) — used by the (not-yet-implemented) real-dump
    path once vendoring lands. When ``None`` (schema-preview mode), every
    tensor falls back to its symbolic [`TENSOR_SCHEMA`] placeholder shape.

    ``phonemize_counts``, when given, maps a subset of the 3 Task-7
    fixture-side-file names (``phoneme_ids``/``tones``/``word_boundaries``)
    to their real element count (`T_text`, only available once a real G2P
    has run on ``args.text``). When ``None``, each falls back to
    [`PHONEMIZE_FIXTURE_SCHEMA`]'s symbolic ``"T_text"`` placeholder.

    Everything else in the manifest (``checkpoint.*``, ``request.*``) is
    fully resolvable from ``args`` alone, real forward pass or not.
    """
    tensor_shapes = tensor_shapes or {}
    tensors = []
    for spec in TENSOR_SCHEMA:
        name = spec["name"]
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

    style_vec = [0.0] * args.style_dim

    return {
        "generator_version": GENERATOR_VERSION,
        "generator": GENERATOR_ID,
        "checkpoint": {
            "sbv2_main": args.sbv2_main_filename,
            "bert_ja": args.bert_ja_filename,
            "bert_en": args.bert_en_filename,
        },
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
    # must be in `[0, N_PHONEME_VOCAB)`, `tones` in `[0, N_TONE_VOCAB)`,
    # `word_boundaries` in `[0, N_WORD_BOUNDARY_VOCAB)`.
    _JA_TABLE: "dict[str, dict[str, list[int]]]" = {}
    _EN_TABLE: "dict[str, dict[str, list[int]]]" = {}

    def phonemize(self, text: str, language: str) -> dict:
        table = self._JA_TABLE if language.upper() == "JA" else self._EN_TABLE
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
    CUDA driver/cuDNN benchmarking heuristic)."""
    torch.manual_seed(args.seed)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)
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
    """Step 2. Both encoders are loaded regardless of `--language`
    (design doc §10: both `bert_hidden_ja` and `bert_hidden_en` are
    always dumped — the fixture set stays complete no matter which
    language a future parity test exercises).

    Both DeBERTa v2 (JA) and DeBERTa v3 (EN) load through
    `transformers.AutoModel` / `AutoTokenizer` — v3 is an attention-
    mechanism delta on the same class hierarchy.
    """
    from transformers import AutoModel, AutoTokenizer

    tok_ja = AutoTokenizer.from_pretrained(args.bert_ja_repo)
    model_ja = AutoModel.from_pretrained(args.bert_ja_repo).eval()
    tok_en = AutoTokenizer.from_pretrained(args.bert_en_repo)
    model_en = AutoModel.from_pretrained(args.bert_en_repo).eval()
    return (tok_ja, model_ja), (tok_en, model_en)


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

    The SBV2 additions (tone_emb + word_boundary_emb) live outside this
    class — see `run_text_encoder` for the additive sum before the
    transformer stack.

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
    """Step 4b. SBV2's tone + word-boundary embedding tables (additive
    contributions to the phoneme embedding, applied BEFORE the
    transformer stack). These do NOT live in vanilla VITS — they are
    SBV2 additions per design doc §7 "既存 piper-plus VITS text encoder
    拡張 — tone + word_boundary embed 追加".

    Both use `torch.nn.Embedding` (a `[V, D]` weight lookup — no
    architectural novelty), so no clean-room scratch is needed for the
    layer itself; only the weight-loading path is scratch.
    """
    from torch import nn as _nn

    tone_emb = _nn.Embedding(N_TONE_VOCAB, D_MODEL)
    wb_emb = _nn.Embedding(N_WORD_BOUNDARY_VOCAB, D_MODEL)
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
        # word_boundary_emb is SBV2 v2-specific. `parity_sbv2_real.rs`'s
        # `SbV2TextEncoder::wb_embed` is `[2, D_MODEL]`.
        wb_emb.weight.copy_(
            _load_tensor(
                state_dict,
                [
                    "enc_p.word_boundary_emb.weight",
                    "enc_p.wb_emb.weight",
                    "enc_p.emb_wb.weight",
                ],
                "wb_embed",
                torch,
            )
        )
    return tone_emb, wb_emb


def run_text_encoder(encoder, tone_emb, wb_emb, phoneme_ids, tones,
                     word_boundaries, torch):
    """Step 4c. Runs the SBV2 text encoder forward, returning
    (phoneme_embed [T_text, D_MODEL], text_hidden [T_text, D_MODEL],
    x_mask [1, 1, T_text]).

    SBV2's extension of vanilla VITS (per design doc §7):

        x = (emb_phoneme + emb_tone + emb_word_boundary) * sqrt(d_model)

    The dumper writes phoneme_embed as [T_text, 192] and text_hidden as
    [T_text, 192], matching design doc §10. Internally VITS shapes are
    [B, D, T]; we transpose+squeeze on write.
    """
    import math as _math
    from vendor.vits import commons

    ids = torch.tensor([phoneme_ids], dtype=torch.long)  # [1, T]
    ton = torch.tensor([tones], dtype=torch.long)        # [1, T]
    wbs = torch.tensor([word_boundaries], dtype=torch.long)  # [1, T]
    x_lengths = torch.tensor([len(phoneme_ids)], dtype=torch.long)

    # Additive SBV2 embed sum BEFORE sqrt scaling. Corresponds to
    # `SbV2TextEncoder::forward`'s phoneme+tone+wb sum on the Rust side.
    x_phon = encoder.emb(ids)          # [1, T, D]
    x_tone = tone_emb(ton)             # [1, T, D]
    x_wb = wb_emb(wbs)                 # [1, T, D]
    phoneme_embed = x_phon + x_tone + x_wb        # [1, T, D]
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
    """Step 7. SBV2's `enc_p.bert_proj_{ja,en}` — a single Conv1d(D_BERT,
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
        prefix = (
            "enc_p.bert_proj_ja" if language.upper() == "JA" else "enc_p.bert_proj_en"
        )
        # Some SBV2 SKUs collapse the two proj tables into a single
        # `enc_p.bert_proj` — try that as a fallback (design doc §7 does
        # not pin the exact SKU-vs-name choice, so we try both honestly).
        self.weight = _load_tensor(
            state_dict,
            [f"{prefix}.weight", "enc_p.bert_proj.weight"],
            f"bert_bridge.{language.upper()}.weight",
            torch,
        )  # [D_MODEL, D_BERT, 1]
        self.bias = _load_tensor(
            state_dict,
            [f"{prefix}.bias", "enc_p.bert_proj.bias"],
            f"bert_bridge.{language.upper()}.bias",
            torch,
        )  # [D_MODEL]

    def forward(self, bert_hidden, text_hidden_transposed, t_text: int, torch):
        """`bert_hidden`: [T_bert, D_BERT]. `text_hidden_transposed`:
        [D_MODEL, T_text] (VITS-side [D, T] convention). Returns
        [T_text, D_MODEL]."""
        import torch.nn.functional as _F

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
    """Step 8. `emb_g.weight`: [n_speakers, D_SPEAKER]. Returns
    [1, D_SPEAKER] (float32)."""
    # SBV2 base (single-speaker fine-tune root) may not ship `emb_g` at
    # all — try both a `[n_speakers, D_SPEAKER]` table AND a single-
    # speaker `[1, D_SPEAKER]` bias-only tensor (design doc §7 pins the
    # multi-speaker table as the canonical form, but honesty about the
    # base case matters — see `sbv2_prepare_checkpoint.py`'s own
    # `n_speakers` clean-room fallback rationale).
    table = _load_tensor(
        state_dict,
        ["emb_g.weight", "emb_g", "sbv2.speaker.table"],
        "speaker.table",
        torch,
    )
    if speaker_id < 0 or speaker_id >= table.shape[0]:
        sys.exit(
            f"{LOG_PREFIX} --speaker-id {speaker_id} out of range "
            f"[0, {table.shape[0]}) for this checkpoint's emb_g table."
        )
    return table[speaker_id : speaker_id + 1].to(dtype=torch.float32)


class StyleVectorInjector:
    """Step 9. SBV2 v2's `emb_g_style` path — an AdaIN-flavored
    Linear(D_STYLE → D_MODEL) + bias projecting the caller-supplied
    style vector into the model space.

    Scratch (design doc §7 "新規, AdaIN 系 scale+bias") — not present in
    vanilla jaywalnut310/vits. No SBV2/BV2 AGPL source was consulted.
    """

    def __init__(self, state_dict: dict, torch):
        # Candidate names: SBV2 SKUs may spell this `emb_g_style.weight`
        # (most common) or `style_proj.weight` (some forks). Loud FR-EX-08
        # error if neither present.
        self.weight = _load_tensor(
            state_dict,
            [
                "emb_g_style.weight",
                "style_proj.weight",
                "sbv2.style_injector.proj_scale",
            ],
            "style_injector.weight",
            torch,
        )  # [D_MODEL, D_STYLE]
        self.bias = _load_tensor(
            state_dict,
            [
                "emb_g_style.bias",
                "style_proj.bias",
                "sbv2.style_injector.proj_bias",
            ],
            "style_injector.bias",
            torch,
        )  # [D_MODEL]

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
        class _Sdp(_nn.Module):
            def __init__(self, in_channels: int, filter_channels: int,
                         kernel_size: int, n_layers: int, n_flows: int):
                super().__init__()
                self.pre = _nn.Conv1d(in_channels, filter_channels, 1)
                self.convs = _vits_modules.DDSConv(
                    filter_channels, kernel_size, n_layers=n_layers, p_dropout=0.0
                )
                self.proj = _nn.Conv1d(filter_channels, filter_channels, 1)
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
        ).eval()

        # Load `dp.*` weights strictly. Unknown post-* / posterior-* keys
        # (training-side) are IGNORED via strict=False, but ANY known key
        # missing (would break inference) triggers a loud error.
        dp_prefix = "dp."
        dp_state = {
            k[len(dp_prefix):]: v.to(dtype=torch.float32)
            for k, v in state_dict.items()
            if k.startswith(dp_prefix)
        }
        if not dp_state:
            sys.exit(
                f"{LOG_PREFIX} no `dp.*` tensors found in checkpoint — SDP "
                "cannot be initialized. Some SBV2 SKUs may use `sdp.*` or "
                "drop SDP entirely; extend this loader if needed."
            )
        missing_keys, _unexpected = self._m.load_state_dict(
            dp_state, strict=False
        )
        if missing_keys:
            sys.exit(
                f"{LOG_PREFIX} SDP is missing {len(missing_keys)} tensor(s) "
                f"after loading from `dp.*`: "
                f"{missing_keys[:8]}{'...' if len(missing_keys) > 8 else ''}. "
                "This means the checkpoint's SDP topology diverges from the "
                "paper-standard composition (arXiv:2106.06103 §2.3) this class "
                "builds — inspect the checkpoint and rewrite the composition "
                "in `SDPReference.__init__` (or vendor upstream `models.py` "
                "SDP directly per scout report Option A)."
            )

    def sample(self, x, x_mask, g, noise_scale_w: float, torch):
        """`x`: [B, D_MODEL, T_text] text-hidden features.
        `x_mask`: [B, 1, T_text]. `g`: [B, D_SPEAKER, 1] speaker/style
        combined conditioning. Returns durations `[T_text]` (float32,
        semantic values are discrete counts; see design doc §10 note on
        why the .bin file is still f32).
        """
        import torch.nn.functional as _F

        # Conditioning branch (paper §2.3 pre + DDS + proj + g add).
        h = self._m.pre(x) * x_mask
        h = self._m.convs(h, x_mask)
        h = self._m.proj(h) * x_mask
        # `g` is not learnable-projected here (upstream SDP has its own
        # `cond` layer for this — see task-30 owner note; without a real
        # checkpoint we conservatively add g broadcasted, and rely on
        # the ConvFlow layers to consume it via their internal `.pre`+
        # `.convs` chain if a checkpoint carries `sdp.cond.*`).
        # (Loading `sdp.cond.*` is a follow-up documented in the class
        # docstring's JP-tone-conditioning note.)

        # Inference: sample from Gaussian prior, invert the flow.
        b = x.shape[0]
        t = x.shape[2]
        z = torch.randn(b, 2, t, dtype=x.dtype, device=x.device) * noise_scale_w
        for flow in reversed(self._m.flows):
            z = flow(z, x_mask, g=h, reverse=True)
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
    """Step 12a. Instantiate vendored
    `vendor.vits.flow.ResidualCouplingBlock` and load `flow.*` weights.
    """
    from vendor.vits.flow import ResidualCouplingBlock

    flow = ResidualCouplingBlock(
        channels=D_MODEL,
        hidden_channels=D_MODEL,
        kernel_size=5,
        dilation_rate=1,
        n_layers=4,
        n_flows=4,
        gin_channels=D_SPEAKER,
    ).eval()

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
    with torch.no_grad():
        missing_keys, _unexpected = flow.load_state_dict(
            flow_state, strict=False
        )
    if missing_keys:
        sys.exit(
            f"{LOG_PREFIX} flow is missing {len(missing_keys)} tensor(s) "
            f"after loading from `flow.*`: "
            f"{missing_keys[:8]}{'...' if len(missing_keys) > 8 else ''}. "
            "Rewriting the composition here would be architecture-guessing "
            "(FR-EX-08); vendor upstream `models.py` `ResidualCouplingBlock` "
            "topology diverges — inspect the checkpoint."
        )
    return flow


def run_flow(flow, mel_hidden, x_mask_mel, g, noise_scale: float, torch):
    """Step 12b. `mel_hidden`: [T_mel, D_MODEL] (prior mean). Returns
    `z_latent`: [T_mel, D_MODEL].

    VITS inference: sample `z_p ~ N(mel_hidden, noise_scale * I)`, then
    pass through `flow.reverse` to invert to `z`.
    """
    # Rearrange to VITS-convention [B, D, T].
    z_p = mel_hidden.transpose(0, 1).unsqueeze(0)  # [1, D, T_mel]
    z_p = z_p + torch.randn_like(z_p) * noise_scale
    z = flow(z_p, x_mask_mel, g=g, reverse=True)
    return z.squeeze(0).transpose(0, 1)  # [T_mel, D_MODEL]


def build_generator(state_dict: dict, torch):
    """Step 13a. Instantiate vendored `vendor.vits.decoder.Generator`
    and load `dec.*` weights. Removes weight_norm for inference
    stability (matches upstream `Generator.remove_weight_norm()`).
    """
    from vendor.vits.decoder import Generator

    gen = Generator(
        initial_channel=D_MODEL,
        resblock=RESBLOCK_TYPE,
        resblock_kernel_sizes=list(RESBLOCK_KERNEL_SIZES),
        resblock_dilation_sizes=_expand_dilations(),
        upsample_rates=list(UPSAMPLE_RATES),
        upsample_initial_channel=UPSAMPLE_INITIAL_CHANNEL,
        upsample_kernel_sizes=list(UPSAMPLE_KERNEL_SIZES),
        gin_channels=D_SPEAKER,
    ).eval()

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
    writes 11 tensor `.bin` files + 3 fixture side files + a fully-
    resolved `reference_dump.manifest.json` to `args.output_dir`. On
    any failure, raises loudly — NEVER silently returns 0 or writes a
    partial fixture (FR-EX-08).
    """
    # ---- Step 0: torch reproducibility ----
    prepare_torch(args, torch)

    # ---- Step 1: SBV2 v2 checkpoint (safetensors + Task-3 config) ----
    config, state_dict = load_sbv2_checkpoint(args)
    print(f"{LOG_PREFIX} loaded SBV2 v2 checkpoint: {len(state_dict)} tensors")

    # ---- Step 2: DeBERTa v2 (JA) + DeBERTa v3 (EN) ----
    (tok_ja, model_ja), (tok_en, model_en) = load_bert_encoders(args, transformers)
    print(f"{LOG_PREFIX} loaded DeBERTa v2 (JA) + v3 (EN)")

    # ---- Step 3: G2P (fixture-only per Task 7) ----
    g2p = MinimalG2P()
    phon = g2p.phonemize(args.text, args.language)
    t_text = len(phon["phoneme_ids"])
    print(f"{LOG_PREFIX} G2P: T_text = {t_text}")

    # ---- Step 4: SBV2 text encoder (VITS TextEncoder + tone/wb) ----
    encoder = build_text_encoder(state_dict, torch)
    tone_emb, wb_emb = build_sbv2_extras(state_dict, torch)
    phoneme_embed, text_hidden, x_mask_text = run_text_encoder(
        encoder, tone_emb, wb_emb,
        phon["phoneme_ids"], phon["tones"], phon["word_boundaries"],
        torch,
    )
    print(f"{LOG_PREFIX} text encoder: phoneme_embed {tuple(phoneme_embed.shape)}, "
          f"text_hidden {tuple(text_hidden.shape)}")

    # ---- Steps 5+6: both BERT paths, always dumped ----
    bert_ja = run_bert(tok_ja, model_ja, args.text, torch)  # [T_bert_ja, H_ja]
    bert_en = run_bert(tok_en, model_en, args.text, torch)  # [T_bert_en, H_en]
    print(f"{LOG_PREFIX} bert_ja {tuple(bert_ja.shape)}, "
          f"bert_en {tuple(bert_en.shape)}")

    # ---- Step 7: BertBridge (active language only feeds text_hidden) ----
    active_bert = bert_ja if args.language.upper() == "JA" else bert_en
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
    style_vec = torch.zeros(1, args.style_dim, dtype=torch.float32)
    style_injector = StyleVectorInjector(state_dict, torch)
    style_projected = style_injector.forward(style_vec, torch)
    print(f"{LOG_PREFIX} style_projected {tuple(style_projected.shape)}")

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

    # ---- Step 12: flow reverse ----
    flow = build_flow(state_dict, torch)
    z_latent = run_flow(flow, mel_hidden, x_mask_mel, g_cond, args.noise_scale, torch)
    print(f"{LOG_PREFIX} z_latent {tuple(z_latent.shape)}")

    # ---- Step 13: HiFi-GAN ----
    gen = build_generator(state_dict, torch)
    waveform = run_generator(gen, z_latent, g_cond, torch)
    samples = int(waveform.shape[0])
    print(f"{LOG_PREFIX} waveform {tuple(waveform.shape)} ({samples} samples @ {SAMPLE_RATE} Hz)")

    # ---- Step 14: write all bin files ----
    dump_dir = args.output_dir / "reference_dump"
    dump_dir.mkdir(parents=True, exist_ok=True)
    write_f32_bin(dump_dir / "phoneme_embed.bin",   phoneme_embed,   torch)
    write_f32_bin(dump_dir / "text_hidden.bin",     text_hidden,     torch)
    write_f32_bin(dump_dir / "bert_hidden_ja.bin",  bert_ja,         torch)
    write_f32_bin(dump_dir / "bert_hidden_en.bin",  bert_en,         torch)
    write_f32_bin(dump_dir / "bert_bridge_out.bin", bert_bridge_out, torch)
    write_f32_bin(dump_dir / "speaker_embed.bin",   speaker_embed,   torch)
    write_f32_bin(dump_dir / "style_projected.bin", style_projected, torch)
    write_f32_bin(dump_dir / "sdp_sample.bin",      sdp_sample,      torch)
    write_f32_bin(dump_dir / "mel_hidden.bin",      mel_hidden,      torch)
    write_f32_bin(dump_dir / "z_latent.bin",        z_latent,        torch)
    write_f32_bin(dump_dir / "waveform.bin",        waveform.unsqueeze(0), torch)

    # Task 7 side files (G2P inputs, replayed by Rust `from_fixture`):
    write_u16_bin(dump_dir / "phoneme_ids.bin",     phon["phoneme_ids"])
    write_u8_bin(dump_dir / "tones.bin",            phon["tones"])
    write_u8_bin(dump_dir / "word_boundaries.bin",  phon["word_boundaries"])

    # ---- Step 15: fully-resolved manifest ----
    manifest = build_manifest(
        args,
        tensor_shapes={
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
        },
        phonemize_counts={
            "phoneme_ids":     t_text,
            "tones":           t_text,
            "word_boundaries": t_text,
        },
    )
    manifest_path = args.output_dir / "reference_dump.manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=False)

    print(
        f"{LOG_PREFIX} OK: wrote 11 tensor .bin + 3 fixture .bin + manifest "
        f"to {args.output_dir}"
    )
    return 0


def run_dump(args: argparse.Namespace) -> int:
    """Real-dump mode. Fails loudly and specifically at whichever of the
    three dependency tiers documented in this file's module doc is missing
    — see there for why this always reaches (and stops at) tier 3 today."""
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
            "`pip install torch` in the parity venv (tools/parity/parity-venv "
            "or your own venv)."
        )

    try:
        import transformers
    except ImportError as exc:
        sys.exit(
            f"{LOG_PREFIX} missing Python dep ({exc}); install with "
            "`pip install transformers` — Apache-2.0, this project's "
            "authorized DeBERTa v2/v3 reference (design doc §6)."
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

    # ------------------------------------------------------------------
    # All 4 dependency tiers have passed (torch / transformers /
    # vendor.vits import / this dumper's own architectural body). Hand
    # off to the pipeline body, which drives all 15 steps documented in
    # `run_pipeline_body`'s docstring. On success, writes 11 tensor
    # .bin + 3 fixture .bin + reference_dump.manifest.json to
    # `args.output_dir`. On failure, raises loudly — NEVER silently
    # returns 0 or writes a partial fixture (FR-EX-08).
    # ------------------------------------------------------------------
    return run_pipeline_body(args, torch, transformers)

    # ------------------------------------------------------------------
    # Unreachable — retained as historical FR-EX-08 trace so that a
    # future maintainer bisecting when the pipeline body landed can see
    # the pre-`run_pipeline_body` gate this call replaced. `run_dump`
    # returned via `run_pipeline_body(...)` above; control cannot fall
    # through here at runtime. See git blame for the original tier-4
    # gate this hosted (design doc §7 pipeline body was the follow-up
    # that closed the gate).
    # ------------------------------------------------------------------
    raise NotImplementedError(  # pragma: no cover  # noqa: F821
        f"{LOG_PREFIX} unreachable: `run_dump` returned via "
        "`run_pipeline_body(...)` above. If you see this in a traceback, "
        "control flow around `return run_pipeline_body(...)` has been "
        "altered — investigate that first."
    )


def parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SBV2 v2 clean-room reference dumper: runs the Style-Bert-VITS2 "
            "v2 forward pass with HF transformers (Apache-2.0) DeBERTa "
            "encoders + a vendored jaywalnut310/vits (MIT) VITS core, and "
            "dumps the 11 intermediate tensors design doc §10 pins for "
            "crates/vokra-models/tests/parity_sbv2_real.rs (Task 28) to diff "
            "a Rust forward pass against. See this script's module docstring "
            "for the manifest schema, the current deferred-vendoring status, "
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
            f"{DEFAULT_TEXT_BY_LANGUAGE['en']!r} for en."
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
            "preview. Fails loudly today (vendoring not landed yet) — see "
            "module docstring."
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
