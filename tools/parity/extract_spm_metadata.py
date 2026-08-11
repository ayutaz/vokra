#!/usr/bin/env python3
"""SentencePiece ``spm.model`` → JSON side-car for the DeBERTa v3 GGUF
converter (Blocker 5, 2026-08-06).

Purpose
=======

The Rust ``crates/vokra-convert/src/models/deberta_v3.rs`` converter accepts
a ``--tokenizer`` side-car so downstream ``SbertTokenizer::from_gguf``
succeeds on real fixtures. To keep a SentencePiece protobuf parser out of
Rust (NFR-DS-02 zero-dep: root ``Cargo.lock`` stays ``vokra-*`` only), the
converter reads a small JSON blob instead. This script produces that blob
from an upstream ``spm.model`` by calling the official
``sentencepiece.SentencePieceProcessor`` (Apache-2.0, permissive) — the
same library any owner running the SBV2 v2 parity dumpers already has
installed via ``tools/parity/pyproject.toml``.

Output schema
=============

The emitted JSON is a single flat object exactly matching the shape
``write_tokenizer_spm_json`` expects:

.. code-block:: json

   {
     "pieces":  ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "<0x00>", ..., "▁the", ...],
     "scores":  [0.0, 0.0, 0.0, 0.0, 0.0, ..., -3.14, ...],
     "unk_id":  3,
     "bos_id":  1,
     "eos_id":  2,
     "pad_id":  0
   }

``pieces`` and ``scores`` are parallel arrays (id = index) and always have
the same length. ``unk_id`` / ``bos_id`` / ``eos_id`` are required by
``SbertTokenizer::from_gguf``. ``pad_id`` is optional; the converter
stamps it only if present in the JSON.

Usage
=====

.. code-block:: bash

   uv run tools/parity/extract_spm_metadata.py \\
       --input /tmp/sbv2-fixtures/deberta-v3-en/spm.model \\
       --output /tmp/sbv2-fixtures/deberta-v3-en/spm.json

Then convert with the vokra-cli side:

.. code-block:: bash

   vokra-cli convert --model deberta-v3 \\
       --input /tmp/sbv2-fixtures/deberta-v3-en/model.safetensors \\
       --tokenizer /tmp/sbv2-fixtures/deberta-v3-en/spm.json \\
       --output /tmp/sbv2-fixtures/deberta-v3-blocker5.gguf

Design notes
============

Ancillary control / user / byte-fallback pieces are emitted verbatim
(including SentencePiece byte-fallback tokens like ``<0x00>`` through
``<0xFF>`` if present in the model) — the converter treats every piece
the same and ``SbertTokenizer::encode`` at runtime performs viterbi over
the full inventory. There is no separate ``kind`` field per piece; the
Rust side treats the whole vocab as a Unigram inventory.

# NOT REFERENCED (clean-room per the SBV2 v2 plan)
- github.com/litagin02/Style-Bert-VITS2 (AGPL-3.0)
- github.com/fishaudio/Bert-VITS2 (AGPL-3.0)
- Any community fork/blog snippet of either of the above.

# Permissive references
- google/sentencepiece (Apache-2.0) — the SentencePieceProcessor Python
  API and its ``.model``/``.piece_to_id`` docstrings this script relies on.
- Kudo & Richardson 2018 (arXiv:1808.06226) — the Unigram algorithm.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _extract(spm_path: Path) -> dict:
    """Load ``spm.model`` and dump ``pieces`` / ``scores`` / specials.

    Returns a JSON-serializable ``dict`` matching the schema in this
    module's own docstring. Raises ``FileNotFoundError`` /
    ``RuntimeError`` (from sentencepiece) on failure — the CLI wrapper
    turns those into a non-zero exit.
    """
    try:
        import sentencepiece as spm
    except ImportError as e:  # pragma: no cover — imported in tests
        raise SystemExit(
            "extract_spm_metadata.py: `sentencepiece` is not installed. "
            "Run `uv sync` from tools/parity/ (Apache-2.0, permissive) or "
            "`pip install sentencepiece>=0.2.2` in your active venv."
        ) from e

    sp = spm.SentencePieceProcessor(model_file=str(spm_path))
    n = sp.get_piece_size()
    pieces: list[str] = [sp.id_to_piece(i) for i in range(n)]
    scores: list[float] = [float(sp.get_score(i)) for i in range(n)]

    payload: dict = {
        "pieces": pieces,
        "scores": scores,
        "unk_id": int(sp.unk_id()),
        "bos_id": int(sp.bos_id()),
        "eos_id": int(sp.eos_id()),
    }

    # SentencePiece's `pad_id()` returns -1 when the model has no
    # explicit PAD control piece. The Rust converter omits `pad_id`
    # from the GGUF when the JSON does not carry it, so we mirror that
    # by simply not writing the key.
    pad = int(sp.pad_id())
    if pad >= 0:
        payload["pad_id"] = pad

    return payload


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="extract_spm_metadata",
        description=(
            "SentencePiece spm.model → JSON side-car for the vokra-cli "
            "deberta-v3 --tokenizer path. Keeps the SentencePiece protobuf "
            "parser out of Rust (NFR-DS-02)."
        ),
    )
    ap.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the upstream spm.model (SentencePiece Unigram protobuf).",
    )
    ap.add_argument(
        "--output",
        required=True,
        type=Path,
        help=(
            "Path to write the JSON side-car. Overwritten if it exists. "
            "The Rust converter accepts it verbatim as --tokenizer."
        ),
    )
    ap.add_argument(
        "--indent",
        type=int,
        default=None,
        help=(
            "Indent width for pretty-printing (e.g. 2). Default: compact "
            "one-line JSON to keep the side-car small."
        ),
    )
    args = ap.parse_args(argv)

    if not args.input.is_file():
        raise SystemExit(f"extract_spm_metadata.py: input does not exist: {args.input}")

    payload = _extract(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=args.indent)
    args.output.write_text(text, encoding="utf-8")

    n = len(payload["pieces"])
    specials = ", ".join(
        f"{k}={payload[k]}"
        for k in ("unk_id", "bos_id", "eos_id", "pad_id")
        if k in payload
    )
    print(
        f"extract_spm_metadata: wrote {args.output} "
        f"({n} pieces, {specials})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
