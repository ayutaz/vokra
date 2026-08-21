#!/usr/bin/env python3
"""DeBERTa v3 (EN) standalone clean-room reference dumper (Task 31, SBV2 v2
plan) — HF ``transformers`` ``AutoModel``/``AutoTokenizer`` only.

Sibling of ``deberta_v2_dump_reference.py`` (same task) and
``sbv2_dump_reference.py`` (Task 30), scoped to *just* the EN BERT encoder
``SbV2Model`` embeds (``DebertaV3Encoder`` on the Rust side,
``crates/vokra-bert/src/deberta_v3.rs``). Useful for isolating BERT-only
numerical parity independently of the rest of the SBV2 v2 pipeline (text
encoder / SDP / flow / HiFi-GAN decoder), which — unlike this dumper — is
still blocked on the ``jaywalnut310/vits`` vendoring ``sbv2_dump_reference.py``
defers (see that file's own module doc). This dumper has **no such gate**:
DeBERTa v3 is loaded directly from the HF Hub via ``AutoModel``, a real,
uv-locked Apache-2.0 dependency — no vendoring needed.

# NOT REFERENCED (clean-room — read this before touching this file)

- ``github.com/litagin02/Style-Bert-VITS2`` (AGPL-3.0) and all forks
- ``github.com/fishaudio/Bert-VITS2`` (AGPL-3.0) and all forks
- Any community fork/derivative/blog-post code excerpt of either of the
  above (Qiita/Zenn writeups, Discord snippets, ...)

This file is built exclusively from the permissive sources listed next.

# Permissive references this file is authorized to use

- **DeBERTa v3** paper (He, Gao, Chen 2021, arXiv:2111.09543).
- **HuggingFace ``transformers``** (Apache-2.0) ``deberta_v2``/``deberta_v3``
  modules — used directly via ``AutoModel``/``AutoTokenizer`` below, the
  actual reference implementation this dumper runs (no re-implementation,
  no vendoring: see ``tools/parity/utmos_dump_reference.py``'s own module
  doc and memory ``feedback-honest-parity-atol`` for why a self-consistent
  mirror of the architecture would validate nothing — this script
  sidesteps that trap entirely by calling the real upstream module).
- The default ``--hf-repo`` below (structural facts only: config, license —
  never a third party's *code*).

# A DeBERTa v3 quirk worth recording

Verified live via the HF Hub (2026-07-26, same pass as the config facts
below): ``microsoft/deberta-v3-large``'s own ``config.json`` declares
``"model_type": "deberta-v2"`` — v3 has no separate model class in
``transformers``; only the pretraining objective (Replaced Token
Detection, ELECTRA-style generator/discriminator) differs from v2, and
that is training-only. The one inference-relevant delta
(``crates/vokra-bert/src/deberta_v3.rs``'s own module doc: a single
position-embedding table shared across all layers, vs. v2's one-table-
per-layer) lives entirely inside ``transformers.DebertaV2Model`` already —
``AutoModel.from_pretrained("microsoft/deberta-v3-large")`` transparently
loads a ``DebertaV2Model`` instance and handles it correctly. Net effect
for this script: its forward-pass code is **identical** to
``deberta_v2_dump_reference.py``'s — the two files differ only in their
default constants (repo id / text / log prefix) — kept as two separate
files per this task's own brief (mirrors the Rust side, where
``DebertaV2Encoder``/``DebertaV3Encoder`` are likewise two distinct types
despite ``DebertaV3Encoder`` reusing v2's structs almost verbatim).

# Default checkpoint

``microsoft/deberta-v3-large`` — the EN BERT encoder design doc §9's SKU
table pins (``docs/superpowers/specs/2026-07-26-sbv2-v2-design.md`` §9,
§16). License ``mit`` (verified live via the HF Hub API by Task 30's own
review, recorded in that task's report). Architecture facts below were
verified live against this repo's raw ``config.json`` on the HF Hub
(``huggingface.co/microsoft/deberta-v3-large/raw/main/config.json``),
fetched 2026-07-26 — not assumed from any "large BERT" convention:

======================= =====================================
field                   value
======================= =====================================
``model_type``          ``deberta-v2`` (see quirk above)
``hidden_size``         1024
``num_hidden_layers``   24
``num_attention_heads`` 16
``vocab_size``          128100
``position_buckets``    256
======================= =====================================

These four numeric facts (1024/24/16/128100) are used **only** as the
schema-preview mode's illustrative placeholders when ``--hf-repo`` is left
at this default (see ``build_preview_manifest`` below) — never assumed for
an overridden ``--hf-repo``, and never assumed as ground truth for the real
dump path either (``run_dump`` reads every shape from the actual loaded
``model.config`` / real tensors, live, regardless of what this docstring
says).

# What this dumper does (``--do-dump``)

1. Tokenizes ``--text`` with ``AutoTokenizer.from_pretrained(--hf-repo)``.
2. Runs ``AutoModel.from_pretrained(--hf-repo, attn_implementation="eager",
   dtype=torch.float32)``
   (eager attention — required for ``output_attentions=True`` to return
   real per-layer tensors; the sdpa/flash-attention-2 backends
   ``transformers`` may pick by default cannot materialize attention
   weights and silently return ``None`` instead, a documented
   ``transformers`` limitation, not a bug introduced here) with
   ``output_hidden_states=True, output_attentions=True``.
3. Dumps, as raw little-endian bytes (see "Byte format" below):
   - ``input_ids`` — the tokenized input, ``[1, T]``, ``int64``.
   - ``embedding`` — ``hidden_states[0]`` (post-embedding-LayerNorm, before
     any transformer layer), ``[1, T, 1024]``, ``float32``.
   - ``layer_NN_output`` (``NN`` = ``00``..``n_layers-1``, zero-padded) —
     ``hidden_states[NN + 1]`` (output of transformer layer ``NN``),
     ``[1, T, 1024]``, ``float32``.
   - ``layer_NN_attention`` — ``attentions[NN]`` (post-softmax attention
     probabilities of layer ``NN``), ``[1, num_heads, T, T]``, ``float32``.
   - ``final_hidden`` — ``outputs.last_hidden_state`` (the model's official
     output tensor), ``[1, T, 1024]``, ``float32``. Dumped *separately*
     from ``layer_{n_layers-1:02d}_output`` even though the two are
     expected to be the same tensor for a plain encoder stack with no
     post-encoder transform — not asserted equal here (no real checkpoint
     has been run against this script yet to confirm), so both are written
     and a future consumer can compare them itself.
4. Writes ``reference_dump.manifest.json`` (schema below) plus one
   ``reference_dump/<name>.bin`` per tensor above.

# Byte format

Every ``.bin`` file is a **raw, headerless, little-endian** byte dump of
its tensor (``tensor.numpy().tobytes()``), **not** ``numpy.save``'s
``.npy`` format. This is a deliberate deviation from the task brief's
literal wording ("argparse + transformers + numpy.save") to match every
sibling ``tools/parity/*_dump*.py``'s actual convention
(``sbv2_dump_reference.py``, ``utmos_dump_reference.py``, ``moshi_dump.py``,
...) — a ``.npy`` file's header would silently corrupt any future Rust-side
flat-byte reader in this codebase (every one of them, e.g.
``parity_sbv2_real.rs``'s ``read_f32_bin``, parses a headerless ``f32``
stream). ``input_ids`` is the one ``int64`` exception (token indices have
no floating-point semantics at all, unlike e.g. ``sbv2_dump_reference.py``'s
``sdp_sample``, which is dumped as ``float32`` despite holding discrete
values because it is a genuine float-space computation output — see that
file's own module doc); every other tensor here is ``float32``.

Endianness: relies on ``numpy``'s native byte order, matching every
sibling dumper's convention — safe because this project's entire ISA scope
is little-endian only (CLAUDE.md "明示的スコープ外: POWER/PPC (VSX)、
LoongArch..." — no big-endian target exists), so no explicit ``<f4``/``<i8``
dtype cast is applied.

# Manifest schema (``reference_dump.manifest.json``)

::

    {
      "generator_version": "1.0",
      "generator": "tools/parity/deberta_v3_dump_reference.py",
      "hf_repo": "microsoft/deberta-v3-large",
      "revision": null,
      "text": "This is a test.",
      "seed": 42,
      "n_layers": 24,
      "tensors": [
        {"name": "input_ids", "path": "reference_dump/input_ids.bin",
         "shape": [1, 8], "dtype": "int64"},
        {"name": "embedding", "path": "reference_dump/embedding.bin",
         "shape": [1, 8, 1024], "dtype": "float32"},
        {"name": "layer_00_output", "path": "reference_dump/layer_00_output.bin",
         "shape": [1, 8, 1024], "dtype": "float32"},
        {"name": "layer_00_attention", "path": "reference_dump/layer_00_attention.bin",
         "shape": [1, 16, 8, 8], "dtype": "float32"},
        "... (one layer_NN_output + layer_NN_attention pair per layer) ...",
        {"name": "final_hidden", "path": "reference_dump/final_hidden.bin",
         "shape": [1, 8, 1024], "dtype": "float32"}
      ]
    }

``n_layers`` and every ``tensors[].shape``'s sequence-length entry are only
resolvable once a real forward pass runs — see "Modes" below for how the
default (schema-preview) mode represents that honestly rather than
guessing.

# Modes

* **Schema preview** (default, no ``--do-dump``): prints the manifest this
  tool *would* write, with ``n_layers`` left as the literal string
  ``"TBD real load"`` and every tensor shape's sequence-length dimension
  fixed at a placeholder ``8`` (neither is knowable without tokenizing
  ``--text`` against the real tokenizer and running the real model — doing
  so would require the network + real dependencies this mode deliberately
  avoids, matching ``--help``'s own no-dependency guarantee). Only the
  first ``layer_00_output``/``layer_00_attention`` pair is shown (each
  carries a ``template_note`` explaining it stands in for all
  ``n_layers`` pairs). Nothing is written to ``--output-dir`` in this
  mode — printing a manifest that *looks* real but is not would itself be
  a fabricated artifact (NFR-QL-04 / FR-EX-08).
* **Real dump** (``--do-dump``): attempts the actual forward pass described
  above. Fails loudly at whichever of two tiers is missing in the current
  environment: (1) ``torch`` missing, (2) ``transformers`` missing, or (3)
  ``--hf-repo``/network problem (bad repo id, no network access, gated repo
  without ``huggingface-cli login``, ...). Nothing is stubbed, mocked, or
  approximated to make any of these "succeed" early.

# Usage

::

    # schema preview (no deps beyond the stdlib, nothing written to disk):
    uv run --project tools/parity/deberta_v3 --frozen python \\
        tools/parity/deberta_v3_dump_reference.py --output-dir /tmp/dbv3-dump

    # real dump (needs torch + transformers + network access to the HF Hub):
    uv run --project tools/parity/deberta_v3 --frozen python \\
        tools/parity/deberta_v3_dump_reference.py \\
        --text "This is a test." --output-dir /tmp/dbv3-dump --do-dump

# Dependencies

``torch`` (BSD) and ``transformers`` (Apache-2.0) are imported lazily
(inside ``--do-dump``'s code path, after ``parser.parse_args()``) so
``--help`` and the schema-preview mode work in an interpreter with neither
installed — matching every other ``tools/parity/*_dump*.py`` sibling's
deferred-import convention.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# --- identity ---------------------------------------------------------------

LOG_PREFIX = "[deberta-v3-dump]"
GENERATOR_ID = "tools/parity/deberta_v3_dump_reference.py"
GENERATOR_VERSION = "1.0"

# design doc §9 SKU table / §16 References. Verified to exist, with license
# mit, via the public HF Hub API (2026-07-26, Task 30's own review — see
# sbv2_dump_reference.py's identical constant + citation).
DEFAULT_HF_REPO = "microsoft/deberta-v3-large"

# Verified live against this repo's raw config.json on the HF Hub
# (2026-07-26 — see module doc table above). Used only as schema-preview
# placeholders when --hf-repo == DEFAULT_HF_REPO; never assumed for an
# overridden --hf-repo, and never assumed (only measured) in --do-dump.
DEFAULT_HIDDEN_SIZE = 1024
DEFAULT_NUM_HEADS = 16
DEFAULT_NUM_LAYERS = 24  # informational only — NOT used as a preview placeholder (see build_preview_manifest: n_layers is always "TBD real load" there, regardless of this known default-repo fact, because --hf-repo is user-overridable)

DEFAULT_TEXT = "This is a test."
DEFAULT_SEED = 42

# Schema-preview-only placeholder sequence length (ambiguity resolution #3
# of this task's brief: "T = 8"). Never used in --do-dump, where the real
# tokenized length is used instead.
PREVIEW_SEQ_LEN = 8


def build_preview_manifest(args: argparse.Namespace) -> dict:
    """Builds the schema-preview ``reference_dump.manifest.json`` contents.

    Every field resolvable from ``args`` alone (no network, no real model)
    is real; ``n_layers`` and the layer-indexed tensor entries are left as
    explicit placeholders (see module doc "Modes") rather than guessed.
    """
    is_default_repo = args.hf_repo == DEFAULT_HF_REPO
    # A user-overridden --hf-repo may have a different hidden_size/
    # num_heads than the default repo's verified values above — only
    # knowable once --do-dump actually loads that repo's real config.
    # Matches this project's established placeholder-for-unknowable-
    # until-load convention (sbv2_dump_reference.py's T_text/T_mel/samples
    # symbolic shapes).
    hidden_size = DEFAULT_HIDDEN_SIZE if is_default_repo else "D"
    num_heads = DEFAULT_NUM_HEADS if is_default_repo else "H"
    seq_len = PREVIEW_SEQ_LEN

    tensors = [
        {
            "name": "input_ids",
            "path": "reference_dump/input_ids.bin",
            "shape": [1, seq_len],
            "dtype": "int64",
        },
        {
            "name": "embedding",
            "path": "reference_dump/embedding.bin",
            "shape": [1, seq_len, hidden_size],
            "dtype": "float32",
        },
        {
            "name": "layer_00_output",
            "path": "reference_dump/layer_00_output.bin",
            "shape": [1, seq_len, hidden_size],
            "dtype": "float32",
            "template_note": (
                "one of n_layers (\"TBD real load\") layer_NN_output "
                "entries, NN = 00..n_layers-1 zero-padded; only the first "
                "is previewed here"
            ),
        },
        {
            "name": "layer_00_attention",
            "path": "reference_dump/layer_00_attention.bin",
            "shape": [1, num_heads, seq_len, seq_len],
            "dtype": "float32",
            "template_note": (
                "one of n_layers (\"TBD real load\") layer_NN_attention "
                "entries, NN = 00..n_layers-1 zero-padded; only the first "
                "is previewed here"
            ),
        },
        {
            "name": "final_hidden",
            "path": "reference_dump/final_hidden.bin",
            "shape": [1, seq_len, hidden_size],
            "dtype": "float32",
        },
    ]

    return {
        "generator_version": GENERATOR_VERSION,
        "generator": GENERATOR_ID,
        "hf_repo": args.hf_repo,
        "revision": args.revision,
        "text": args.text,
        "seed": args.seed,
        "n_layers": "TBD real load",
        "tensors": tensors,
    }


def run_preview(args: argparse.Namespace) -> int:
    """Schema-preview mode: prints the manifest this tool *would* write and
    touches no files. Needs no dependency beyond the stdlib — safe to run
    in any interpreter, same guarantee as ``--help``."""
    manifest = build_preview_manifest(args)
    print(
        f"{LOG_PREFIX} schema preview (pass --do-dump to attempt a real "
        f"dump). tensors[].shape use a seq_len={PREVIEW_SEQ_LEN} placeholder "
        "and n_layers=\"TBD real load\" (real values only exist once a real "
        "HF Hub forward pass runs); only 1 of n_layers layer_NN_output/"
        "layer_NN_attention pairs is shown (see each entry's "
        f"template_note). Nothing is written to {args.output_dir} in this "
        "mode."
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=False))
    return 0


def run_dump(args: argparse.Namespace) -> int:
    """Real-dump mode: loads ``--hf-repo`` via HF ``transformers``, runs a
    real forward pass over ``--text``, and dumps every tensor this file's
    module doc lists. Fails loudly and specifically at whichever of the two
    dependency tiers (or the network/repo tier) is missing — see module
    doc "Modes"."""
    dump_dir = args.output_dir / "reference_dump"
    dump_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
    except ImportError as exc:
        sys.exit(
            f"{LOG_PREFIX} missing Python dep ({exc}); install with "
            "`uv sync --project tools/parity/deberta_v3 --frozen`."
        )

    try:
        import transformers
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        sys.exit(
            f"{LOG_PREFIX} missing Python dep ({exc}); install with "
            "`uv sync --project tools/parity/deberta_v3 --frozen` — the "
            "locked Apache-2.0 reference environment."
        )
    print(f"{LOG_PREFIX} torch {torch.__version__}, transformers {transformers.__version__} present.")

    # Defensive determinism only — an eval()-mode BERT forward pass has no
    # dropout/sampling of its own, so this seed does not change the output,
    # but it is recorded in the manifest for reproducibility bookkeeping
    # (matches sbv2_dump_reference.py's --seed, which the SDP's Gaussian
    # draws there do actually consume).
    torch.manual_seed(args.seed)

    print(
        f"{LOG_PREFIX} loading tokenizer + model from {args.hf_repo!r} "
        f"(revision={args.revision or 'default'}) ..."
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.hf_repo, revision=args.revision)
        # attn_implementation="eager" is required for output_attentions=True
        # to return real per-layer attention-probability tensors — the
        # sdpa/flash-attention-2 backends transformers may pick by default
        # cannot materialize attention weights and silently return None
        # instead (a documented transformers limitation, not a bug here);
        # this dumper's whole point is to capture those tensors, so it must
        # not let a faster backend silently degrade the dump (FR-EX-08).
        # transformers 5 loads this fp16 checkpoint at its source dtype by
        # default, whereas the locked 4.x oracle loaded it as float32. Vokra's
        # GGUF and CPU reference path are float32, so make that numerical
        # contract explicit instead of inheriting a library-version default.
        model = AutoModel.from_pretrained(
            args.hf_repo,
            revision=args.revision,
            attn_implementation="eager",
            dtype=torch.float32,
        )
    except Exception as exc:  # noqa: BLE001 - from_pretrained raises many distinct types across huggingface_hub/transformers/requests versions; the message is what matters
        sys.exit(
            f"{LOG_PREFIX} could not load {args.hf_repo!r} "
            f"(revision={args.revision or 'default'}): {type(exc).__name__}: {exc}\n"
            f"{LOG_PREFIX} check network access, --hf-repo spelling, and "
            "--revision; gated/private repos need `huggingface-cli login` first."
        )
    model.eval()

    encoded = tokenizer(args.text, return_tensors="pt")
    seq_len = encoded["input_ids"].shape[1]
    print(f"{LOG_PREFIX} tokenized {args.text!r} -> {seq_len} tokens")

    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True, output_attentions=True)

    hidden_states = outputs.hidden_states
    attentions = outputs.attentions
    if hidden_states is None or attentions is None:
        sys.exit(
            f"{LOG_PREFIX} model output is missing hidden_states/attentions "
            "despite output_hidden_states=True, output_attentions=True "
            f"(hidden_states present={hidden_states is not None}, "
            f"attentions present={attentions is not None}). Refusing to "
            "write a partial dump."
        )
    n_layers = len(attentions)
    if len(hidden_states) != n_layers + 1:
        sys.exit(
            f"{LOG_PREFIX} unexpected shapes: len(hidden_states)="
            f"{len(hidden_states)} != len(attentions)+1={n_layers + 1}. "
            f"{args.hf_repo} may have a non-standard encoder architecture "
            "this dumper's fixed hidden_states[0]=embedding / "
            "hidden_states[1:]=per-layer convention does not hold for. "
            "Refusing to write a partial/misaligned dump."
        )
    if any(a is None for a in attentions):
        sys.exit(
            f"{LOG_PREFIX} one or more per-layer attention tensors is None "
            "even with attn_implementation='eager' — refusing to write a "
            "partial dump."
        )

    tensors_meta: "list[dict]" = []

    def dump(name: str, tensor, dtype: str) -> None:
        path = dump_dir / f"{name}.bin"
        if dtype == "int64":
            arr = tensor.detach().to(torch.int64).contiguous().numpy()
        else:
            arr = tensor.detach().to(torch.float32).contiguous().numpy()
        path.write_bytes(arr.tobytes())
        tensors_meta.append(
            {
                "name": name,
                "path": f"reference_dump/{name}.bin",
                "shape": list(tensor.shape),
                "dtype": dtype,
            }
        )
        print(f"{LOG_PREFIX}   {name:24s} shape={list(tensor.shape)} dtype={dtype}")

    dump("input_ids", encoded["input_ids"], "int64")
    dump("embedding", hidden_states[0], "float32")
    for i in range(n_layers):
        dump(f"layer_{i:02d}_output", hidden_states[i + 1], "float32")
        dump(f"layer_{i:02d}_attention", attentions[i], "float32")
    dump("final_hidden", outputs.last_hidden_state, "float32")

    manifest = {
        "generator_version": GENERATOR_VERSION,
        "generator": GENERATOR_ID,
        "hf_repo": args.hf_repo,
        "revision": args.revision,
        "text": args.text,
        "seed": args.seed,
        "n_layers": n_layers,
        "tensors": tensors_meta,
    }
    manifest_path = args.output_dir / "reference_dump.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(f"{LOG_PREFIX} wrote {manifest_path}")
    return 0


def parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DeBERTa v3 (EN) clean-room reference dumper: runs the real HF "
            "transformers (Apache-2.0) AutoModel/AutoTokenizer forward pass "
            "and dumps per-layer hidden_states + attention tensors, for "
            "isolating BERT-only numerical parity ahead of the full SBV2 v2 "
            "pipeline (Task 30's sbv2_dump_reference.py). See this script's "
            "module docstring for the manifest schema, byte format, and the "
            "clean-room NOT-REFERENCED list."
        )
    )
    parser.add_argument(
        "--hf-repo",
        default=DEFAULT_HF_REPO,
        help=f"HF Hub repo id for the DeBERTa v3 encoder (default: {DEFAULT_HF_REPO}).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional HF revision (branch / tag / commit sha) to pin (default: repo's default branch).",
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help=f"Text to tokenize and encode (default: {DEFAULT_TEXT!r}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "PRNG seed, recorded in the manifest for reproducibility "
            f"bookkeeping (default: {DEFAULT_SEED}). A plain BERT forward "
            "pass in eval() mode has no dropout/sampling of its own, so "
            "this does not change the dumped tensors."
        ),
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
            "preview. Fails loudly if torch/transformers are missing or "
            "--hf-repo cannot be loaded — see module docstring."
        ),
    )
    args = parser.parse_args(argv)

    if args.seed < 0:
        parser.error("--seed must be >= 0")

    return args


def main() -> int:
    args = parse_args()
    if not args.do_dump:
        return run_preview(args)
    return run_dump(args)


if __name__ == "__main__":
    sys.exit(main())
