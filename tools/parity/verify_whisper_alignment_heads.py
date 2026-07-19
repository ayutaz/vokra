#!/usr/bin/env python3
"""Cross-check the converter's builtin Whisper alignment-heads table against
upstream openai-whisper (M4-residual cc-08 — alignment-heads verification).

The converter hardcodes the DTW cross-attention head selection per size in
``builtin_alignment_heads`` (crates/vokra-convert/src/models/whisper.rs). The
authoritative upstream source is the ``_ALIGNMENT_HEADS`` base85+gzip boolean
masks in the installed ``whisper`` package's ``__init__.py`` — exactly what
``whisper.load_model`` feeds ``Whisper.set_alignment_heads``. This script:

1. decodes every supported size's upstream mask with the documented method::

       mask  = np.frombuffer(gzip.decompress(base64.b85decode(DUMP)), dtype=bool)
                   .reshape(n_text_layer, n_text_head)
       pairs = np.argwhere(mask)   # (layer, head), row-major

2. **parses the actual Rust table out of the source file** (no second copy
   that could drift in lockstep) and compares flat ``[layer, head, …]`` lists
   exactly;

3. optionally cross-checks the decoded pairs against the real
   ``model.alignment_heads`` buffer for any ``~/.cache/whisper/*.pt``
   checkpoint already present (``--buffers``; never downloads).

Exit 0 = every size matches; any mismatch, parse failure or grid-size
inconsistency is a hard non-zero error (FR-EX-08: never report a pass it did
not compute).

Run (from the repo root, any venv with ``openai-whisper`` + ``numpy``)::

    python tools/parity/verify_whisper_alignment_heads.py [--buffers]

Verified 2026-07-19 with openai-whisper 20250625: all 5 sizes match, and the
base / large-v3-turbo real-checkpoint buffers match their base85 decodes.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import re
import sys
from pathlib import Path

import numpy as np
import whisper

# (rust size label, upstream _ALIGNMENT_HEADS key, n_text_layer, n_text_head).
# Grids are the openai/whisper model registry values (base 6x8 / small 12x12 /
# medium 24x16 / large-v3 32x20 / turbo 4x20); the mask length assertion below
# re-validates each grid against the decoded blob, so a wrong row here cannot
# silently pass.
SIZES = [
    ("whisper-base", "base", 6, 8),
    ("whisper-small", "small", 12, 12),
    ("whisper-medium", "medium", 24, 16),
    ("whisper-large-v3", "large-v3", 32, 20),
    ("whisper-turbo", "large-v3-turbo", 4, 20),
]

RUST_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "crates"
    / "vokra-convert"
    / "src"
    / "models"
    / "whisper.rs"
)


def upstream_flat(key: str, n_layer: int, n_head: int) -> list[int]:
    """Decode one upstream base85 dump to a flat [layer, head, …] list."""
    dump = whisper._ALIGNMENT_HEADS[key]
    mask = np.frombuffer(gzip.decompress(base64.b85decode(dump)), dtype=bool)
    if mask.size != n_layer * n_head:
        sys.exit(
            f"{key}: decoded mask has {mask.size} entries, grid says "
            f"{n_layer}x{n_head}={n_layer * n_head} — registry drift, refusing"
        )
    pairs = np.argwhere(mask.reshape(n_layer, n_head))
    return [int(x) for pair in pairs.tolist() for x in pair]


def rust_builtin_table(source: Path) -> dict[str, list[int]]:
    """Extract `builtin_alignment_heads`'s match arms from the Rust source.

    Parses `"whisper-<size>" => Some(&[ … ])` literal arms inside the
    function body. A missing function or zero parsed arms is a hard error —
    if the Rust shape changes, this script must be updated, not skipped.
    """
    text = source.read_text(encoding="utf-8")
    m = re.search(
        r"fn builtin_alignment_heads\([^)]*\)[^{]*\{(?P<body>.*?)\n\}",
        text,
        re.DOTALL,
    )
    if not m:
        sys.exit(f"could not locate builtin_alignment_heads() in {source}")
    body = m.group("body")
    table: dict[str, list[int]] = {}
    for arm in re.finditer(
        r'"(?P<name>whisper-[a-z0-9.-]+)"\s*=>\s*Some\(&\[(?P<vals>[^\]]*)\]\)',
        body,
        re.DOTALL,
    ):
        vals = [int(v) for v in re.findall(r"\d+", arm.group("vals"))]
        table[arm.group("name")] = vals
    if not table:
        sys.exit(f"parsed zero match arms from builtin_alignment_heads in {source}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the converter's builtin alignment-heads table "
        "against upstream openai-whisper."
    )
    parser.add_argument(
        "--buffers",
        action="store_true",
        help="Also cross-check real model.alignment_heads buffers for any "
        "size whose .pt checkpoint is already in ~/.cache/whisper "
        "(never downloads).",
    )
    args = parser.parse_args()

    rust = rust_builtin_table(RUST_SOURCE)
    failures = 0
    for rust_name, upstream_key, n_layer, n_head in SIZES:
        expect = upstream_flat(upstream_key, n_layer, n_head)
        got = rust.get(rust_name)
        if got is None:
            print(f"FAIL {rust_name}: no arm in the Rust builtin table")
            failures += 1
            continue
        status = "OK" if got == expect else "FAIL"
        if got != expect:
            failures += 1
        pairs = [tuple(expect[i : i + 2]) for i in range(0, len(expect), 2)]
        print(f"{status} {rust_name}: upstream({upstream_key}) {pairs}")
        if got != expect:
            print(f"     rust table has {got}")

    extra = set(rust) - {name for name, *_ in SIZES}
    if extra:
        print(f"FAIL: Rust table has unverified extra arms: {sorted(extra)}")
        failures += 1

    if args.buffers:
        cache = Path.home() / ".cache" / "whisper"
        for rust_name, upstream_key, n_layer, n_head in SIZES:
            if not (cache / f"{upstream_key}.pt").is_file():
                continue
            model = whisper.load_model(upstream_key, device="cpu")
            heads = model.alignment_heads
            dense = heads.to_dense() if heads.is_sparse else heads
            buf = [int(x) for p in dense.nonzero().tolist() for x in p]
            expect = upstream_flat(upstream_key, n_layer, n_head)
            status = "OK" if buf == expect else "FAIL"
            if buf != expect:
                failures += 1
            print(f"{status} {rust_name}: real checkpoint buffer vs base85 decode")

    if failures:
        sys.exit(f"{failures} alignment-heads mismatch(es) — see FAIL lines above")
    print("all alignment-heads tables verified against upstream")


if __name__ == "__main__":
    main()
