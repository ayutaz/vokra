#!/usr/bin/env python3
"""Placeholder-gate for model-level RTF regression baselines (X-06-T07/T09/T11/T12).

A model-level bench baseline JSON (the `rtf`-carrying report
`vokra-cli bench --format json` emits) MUST be seeded from a real
ubuntu-latest CI measurement — an M1/NEON rig false-fails the 5% gate (see
docs/bench-baselines/README.md, the mel-frontend NEON-skew example). Until the
owner seeds it, the committed file is a PLACEHOLDER and the gate must
CLEAN-SKIP: report "not yet measured", never a green pass on a number that
does not exist (fabricated pass 禁止, FR-EX-08).

This helper is the single classifier the CI step and its oracle share, so
"what counts as seeded" cannot drift between the gate and its test.

Contract:
  * classify(path) -> "placeholder" | "seeded"
      - "$placeholder": true            -> placeholder (skip)
      - a numeric top-level "rtf" field -> seeded (gate)
      - neither                         -> raise (malformed; the CI step
        must FAIL, not silently pass — a broken baseline is a defect, not a
        skip)
  * missing file / unparseable JSON     -> raise

CLI: `python3 tools/bench/baseline_gate.py <path>` prints the classification
to stdout and exits 0; a missing/malformed file prints to stderr and exits 2
(so a `set -e` shell step fails loudly rather than treating it as a skip).

stdlib-only (zero-dep NFR-DS-02).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


class BaselineError(Exception):
    """A baseline file that is neither a placeholder nor a real seeded report."""


def classify(path) -> str:
    """Return "placeholder" or "seeded"; raise BaselineError otherwise."""
    p = Path(path)
    if not p.is_file():
        raise BaselineError(f"baseline file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BaselineError(f"baseline is not valid JSON: {p} ({exc})") from exc
    if not isinstance(data, dict):
        raise BaselineError(f"baseline is not a JSON object: {p}")
    if data.get("$placeholder") is True:
        # A placeholder must NOT also carry a numeric rtf — that would be an
        # ambiguous half-seeded file that could read as a pass. Reject it.
        rtf = data.get("rtf")
        if isinstance(rtf, (int, float)):
            raise BaselineError(
                f"baseline marks itself a placeholder but also carries a numeric "
                f'rtf={rtf}: {p} — remove one (a placeholder has "rtf": null)'
            )
        return "placeholder"
    rtf = data.get("rtf")
    if isinstance(rtf, bool) or not isinstance(rtf, (int, float)):
        raise BaselineError(
            f'baseline has no numeric top-level "rtf" and is not marked '
            f'"$placeholder": true — {p} is malformed, refusing to treat it as '
            f"either a skip or a pass"
        )
    return "seeded"


def main(argv) -> int:
    if len(argv) != 2:
        print("usage: baseline_gate.py <baseline.json>", file=sys.stderr)
        return 2
    try:
        print(classify(argv[1]))
        return 0
    except BaselineError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
