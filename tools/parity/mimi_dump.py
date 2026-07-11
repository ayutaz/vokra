#!/usr/bin/env python3
"""Dump Kyutai (Moshi) Mimi RVQ codec reference tensors for M3-06 parity.

This is an **offline** tool (FR-LD-05: no Python / PyTorch is ever pulled into
the runtime). It regenerates the fixtures under
``tests/parity/mimi/`` that a future ``parity_mimi`` Rust test can compare
against at FP32 ``atol = 0.01`` (NFR-QL-01).

Status (2026-07-09)
-------------------

M3-06 lands the Vokra-side op + a self-consistency parity harness against an
**internal** ramp / identity oracle (see
``crates/vokra-ops/src/mimi_rvq.rs`` unit tests). A **Kyutai reference dump**
is a `TBD` that runs once with an appropriate Mimi checkpoint on hand — the
`moshi` (Kyutai) PyPI package is Apache 2.0 code, but the Mimi weights are
CC-BY 4.0 and require the M2-13 compliance gate to accept the
``AttributionRequired`` licence class before the runtime can load them
(``registry_lookup("mimi") == AttributionRequired``). See
``docs/adr/M3-06-mimi-rvq.md`` §D5 for the rationale — the parity fixture
lands with M3-09 (CosyVoice2), which pulls in the same Mimi checkpoint and
lets the two tasks share one download step.

Intended usage (once a Mimi checkpoint is on hand)
--------------------------------------------------

* Install the Kyutai reference implementation offline
  (``pip install moshi``; only used inside this script — the Vokra runtime
  never depends on it).
* Load Mimi's decode codebooks (``model.decode_only()`` or the reference
  ``mimi.MimiModel`` — check the current Kyutai API in the version pinned
  in ``manifest.txt``).
* Dump the following to ``tests/parity/mimi/``:

  - ``codebook_tables.f32``  – ``[n_codebooks, codebook_size, d_model]``
                                row-major (matches the Vokra
                                ``CodebookTable`` layout);
  - ``codes.u32``            – ``[time, n_codebooks]`` fixed-seed codes;
  - ``decoded_features.f32`` – reference decode output
                                ``[time, d_model]`` row-major;
  - ``manifest.txt``         – shapes / seed / sha256 / ``moshi`` version.

* The sibling Rust test (``crates/vokra-ops/tests/parity_mimi_reference.rs``,
  to be added with M3-09) loads the fixture, calls
  ``vokra_ops::mimi_rvq_decode`` on the same codes / tables, and asserts
  ``max|diff| <= 0.01`` (FP32).

Non-goals
---------

* Fetching / bundling the Mimi weights: keep the fixture repo-friendly
  (``.gitignore`` skips large ``codebook_tables.f32``) or upload to a
  side-cache — the M2-13 gate + the CC-BY 4.0 attribution requirement mean
  a full model dump is not committed to git.
* Any Metal / CUDA parity: those seams land in Wave 5 (M3-06 T14 / T15)
  and reuse the same fixture on-device.

Running (placeholder)
---------------------

Since the Kyutai reference is not on this development machine, this script
exits with a non-zero status and prints the intended flow. Once a machine
has ``moshi`` + a Mimi checkpoint in cache, the flow below is what a
follow-up commit will implement:

    tools/parity/parity-venv/bin/python tools/parity/mimi_dump.py \\
        --out tests/parity/mimi --seed 0 --time 100 --n-codebooks 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("tests/parity/mimi"),
        help="Fixture output directory (relative to repo root).",
    )
    ap.add_argument("--seed", type=int, default=0, help="Deterministic RNG seed.")
    ap.add_argument(
        "--time",
        type=int,
        default=100,
        help="Number of timesteps in the codes fixture.",
    )
    ap.add_argument(
        "--n-codebooks", type=int, default=8, help="RVQ codebook count (Mimi = 8)."
    )
    args = ap.parse_args()

    # Sanity: verify the reference is on-machine before writing anything.
    try:
        import moshi  # noqa: F401  # type: ignore[import-untyped]
    except ImportError:
        print(
            "mimi_dump.py: `moshi` (Kyutai) package is not installed on this "
            "machine. This is expected on the CC dev machine — the M3-06 "
            "parity harness is internal-oracle-only; a real Mimi dump is "
            "part of the M3-09 (CosyVoice2) work. See "
            "docs/adr/M3-06-mimi-rvq.md §D5.",
            file=sys.stderr,
        )
        return 2

    print(
        f"mimi_dump.py: would dump n_codebooks={args.n_codebooks} time={args.time} "
        f"seed={args.seed} into {args.out}",
        file=sys.stderr,
    )
    print(
        "mimi_dump.py: real Mimi codec dumping is a TBD — see the module "
        "docstring for the intended flow once a Mimi checkpoint is loaded.",
        file=sys.stderr,
    )
    return 2  # explicit non-zero: no fixture was written


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
