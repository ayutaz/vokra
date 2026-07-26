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

# Status: CLI + manifest-schema scaffold; the real forward pass is deferred

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
  pass. As of this commit that **always** fails loudly, in one of three
  tiers depending on what is installed:

  1. ``torch`` missing -> actionable ``pip install torch``.
  2. ``transformers`` missing -> actionable ``pip install transformers``
     (Apache-2.0 — this project's authorized DeBERTa reference).
  3. Both present -> fails at ``from vendor.vits import text_encoder``,
     because ``tools/parity/vendor/vits/`` currently ships only a
     ``LICENSE`` + ``README.md`` scaffold (see that directory), no vendored
     module. This is **the** real gate: the VITS core (text encoder /
     normalizing flow / HiFi-GAN decoder reference) has not been ported
     yet. See that README for exactly what a follow-up needs to add.

  Nothing is stubbed, mocked, or approximated to make this path "succeed"
  early — an SBV2 forward pass assembled from a self-consistent mirror of
  the architecture (rather than the real permissive reference
  implementations) would validate nothing, the same lesson
  ``tools/parity/utmos_dump_reference.py``'s own module doc draws from the
  Kokoro ``92dbc92`` incident. Once vendoring lands, ``--do-dump`` gains the
  G2P -> ``SbV2TextEncoder`` -> DeBERTa-bridge -> SDP -> flow -> HiFi-GAN
  pipeline from design doc §7, writing ``reference_dump/*.bin`` (raw
  little-endian f32, matching every other ``*_dump*.py`` sibling's
  ``arr.tobytes()`` convention — *not* ``numpy.save``'s ``.npy`` format,
  which ``parity_sbv2_real.rs``'s ``read_f32_bin`` does not parse) plus the
  real, fully-resolved ``reference_dump.manifest.json``.

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
    except ImportError as exc:
        sys.exit(
            f"{LOG_PREFIX} jaywalnut310/vits (MIT) has not been vendored yet "
            f"({exc}). tools/parity/vendor/vits/ currently ships only "
            "LICENSE + README.md (scaffold, Task 30) — see that README's "
            "'What a follow-up vendoring pass should add here' table for the "
            "minimal modules (text_encoder.py / coupling.py / flow.py / "
            "decoder.py) a follow-up must port before --do-dump can run a "
            "real forward pass. This is deliberate scaffolding, not a bug: "
            "fabricating a tensor dump without the real permissive reference "
            "implementation would validate nothing (see "
            "tools/parity/utmos_dump_reference.py's own module doc, and "
            "memory `feedback-honest-parity-atol`) — refused rather than "
            "attempted."
        )

    # Unreachable until the vendoring above lands. When it does, this is
    # where the real pipeline goes (design doc §7): G2P -> SbV2TextEncoder
    # (dump phoneme_embed/text_hidden) -> DeBERTa v2/v3 via `transformers`
    # (dump bert_hidden_ja/bert_hidden_en) -> BertBridge (bert_bridge_out)
    # -> speaker/style conditioning (speaker_embed/style_projected) -> SDP
    # (sdp_sample) -> length regulator (mel_hidden) -> vendored VITS flow
    # (z_latent) -> vendored VITS/HiFi-GAN decoder (waveform), writing each
    # as raw little-endian float32 to <output-dir>/reference_dump/<name>.bin
    # plus the fully-resolved reference_dump.manifest.json alongside it.
    #
    # Task 7: the same G2P call whose output feeds SbV2TextEncoder above
    # also dumps the three PhonemizeFixture side files (paths per
    # PHONEMIZE_FIXTURE_SCHEMA):
    #   with open(<output-dir>/reference_dump/phoneme_ids.bin, "wb") as f:
    #       np.asarray(phon.phoneme_ids, dtype="<u2").tofile(f)   # uint16 LE
    #   with open(<output-dir>/reference_dump/tones.bin, "wb") as f:
    #       np.asarray(phon.tones, dtype="u1").tofile(f)          # uint8
    #   with open(<output-dir>/reference_dump/word_boundaries.bin, "wb") as f:
    #       np.asarray(phon.word_boundaries, dtype="u1").tofile(f)
    # Pass their real element count (`len(phon.phoneme_ids)`) as the
    # `phonemize_counts` dict to `build_manifest` so the emitted
    # `phonemize_fixture.*.count` fields carry the real integer instead of
    # the "T_text" placeholder.
    return 0  # pragma: no cover - unreachable today, see above


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
