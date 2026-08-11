#!/usr/bin/env python3
"""Parses ``crates/vokra-models/tests/parity_sbv2_real.rs``'s stderr
output (as captured by ``.github/workflows/parity-sbv2-real.yml``'s
``$RUNNER_TEMP/parity.log``) into a machine-readable atol-summary
JSON. Consumed by ``sbv2_atol_updater.py`` (WP-02, 2026-08-09) and
uploaded as a workflow artifact for owner review.

# Contract

The parity test emits per-tensor rows in the format (see
`parity_sbv2_real.rs`'s WP-02 refactor of
`diff_intermediates_against_manifest`):

    [parity_sbv2_real] <name>: max |Δ| = <max_diff:.6e> atol <atol> verdict <PASS|FAIL> [<Marker>]

where `<Marker>` is one of:
  - `[Measured]`
  - `[EstimatedPreFixture]`
  - `[UnmeasuredDefault(ATOL_DEFAULT)]`
  - `[UNPINNED]`

Every recognized row becomes one entry in the emitted JSON's
`entries` array; a line that starts with `[parity_sbv2_real]` but
does not match the regex is warned about on stderr and skipped (so a
future test-side format change is loud, not silent — FR-EX-08).

# Zero-dependency invariant

Pure-stdlib. Runs in the parity workflow's uv-managed venv but does
NOT need any of its dependencies (`torch`, `transformers`, etc.);
this script is intentionally light so a fresh-venv failure of the
heavy DL doesn't block artifact production.

# Usage

    uv run python tools/parity/sbv2_atol_summary_from_log.py \\
        --log <path/to/parity.log> \\
        --output <path/to/atol-summary.json> \\
        [--ci-run-id <id>] [--ci-run-url <url>]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Matches (verbatim) the stderr row `parity_sbv2_real.rs` emits per
# intermediate tensor. Groups:
#   1 name          e.g. "bert_hidden_ja"
#   2 max_diff      e.g. "5.483e-06"
#   3 atol          e.g. "0.02"
#   4 verdict       "PASS" or "FAIL"
#   5 marker_raw    e.g. "[EstimatedPreFixture]"
ROW_RE = re.compile(
    r"^\[parity_sbv2_real\]\s+"
    r"(?P<name>\S+):\s+"
    r"max\s+\|Δ\|\s+=\s+"
    r"(?P<max_diff>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+"
    r"atol\s+(?P<atol>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+"
    r"verdict\s+(?P<verdict>PASS|FAIL)\s+"
    r"(?P<marker>\[[^\]]+\])"
)

# Maps the parity test's marker syntax to the shorter, machine-friendly
# calibration status names `sbv2_atol_updater.py` expects.
MARKER_TO_STATUS = {
    "[Measured]": "Measured",
    "[EstimatedPreFixture]": "EstimatedPreFixture",
    "[UnmeasuredDefault(ATOL_DEFAULT)]": "UnmeasuredDefault",
    "[UNPINNED]": "Unpinned",
}


def parse_log(log_text: str) -> list[dict[str, Any]]:
    """Extracts per-tensor rows from parity.log text. Ignores lines
    that do not start with `[parity_sbv2_real]`; loudly warns (via
    stderr) on a `[parity_sbv2_real]` prefix that does not match the
    row schema (the pre-emission "waveform parity OK" summary line
    intentionally lives in a different format and is skipped)."""
    entries: list[dict[str, Any]] = []
    for line in log_text.splitlines():
        if not line.startswith("[parity_sbv2_real]"):
            continue
        m = ROW_RE.match(line)
        if not m:
            # Skip the informational summary line the waveform block
            # emits ("waveform parity OK: rust=... samples ..."). It
            # uses a different format on purpose (multi-metric).
            if "waveform parity OK" in line:
                continue
            print(
                f"[sbv2_atol_summary_from_log] warning: unrecognized "
                f"[parity_sbv2_real] row (schema drift?): {line!r}",
                file=sys.stderr,
            )
            continue
        marker = m.group("marker")
        status = MARKER_TO_STATUS.get(marker, marker)
        entries.append({
            "name": m.group("name"),
            "max_diff": float(m.group("max_diff")),
            "current_atol": float(m.group("atol")),
            "calibration_status": status,
            "verdict": m.group("verdict"),
        })
    return entries


def build_summary(
    entries: list[dict[str, Any]],
    *,
    ci_run_id: str | None,
    ci_run_url: str | None,
) -> dict[str, Any]:
    """Wraps ``entries`` in the top-level schema
    ``sbv2_atol_updater.propose_from_summary`` expects."""
    return {
        "schema_version": 1,
        "ci": {
            "run_id": ci_run_id or "",
            "run_url": ci_run_url or "",
        },
        "entries": entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Parses parity_sbv2_real.rs stderr rows from parity.log into "
            "an atol-summary.json artifact for `sbv2_atol_updater.py` and "
            "owner review."
        )
    )
    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="Path to the tee'd parity.log from `parity-sbv2-real.yml`.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the atol-summary.json artifact.",
    )
    parser.add_argument(
        "--ci-run-id",
        default=None,
        help="Optional CI run id (e.g. `${{ github.run_id }}`) for traceability.",
    )
    parser.add_argument(
        "--ci-run-url",
        default=None,
        help="Optional CI run URL (e.g. `${{ github.run_url }}`) for traceability.",
    )
    args = parser.parse_args(argv)

    if not args.log.is_file():
        # Do NOT fail — emit an empty-entries summary. Rationale:
        # the parity job might have hard-failed at an earlier step
        # (missing GGUF, torch install failure, etc.) so no parity
        # rows exist. An empty summary is HONEST — the updater will
        # emit "no proposals" and the owner sees the upstream failure
        # in the other step outputs. Fabricating rows here would be
        # exactly the FR-EX-08 anti-pattern the workflow guards
        # against.
        print(
            f"[sbv2_atol_summary_from_log] warning: log file {args.log} "
            "does not exist; emitting empty summary. See other CI step "
            "logs for the upstream failure that prevented parity from "
            "running.",
            file=sys.stderr,
        )
        entries: list[dict[str, Any]] = []
    else:
        log_text = args.log.read_text(encoding="utf-8", errors="replace")
        entries = parse_log(log_text)

    summary = build_summary(
        entries,
        ci_run_id=args.ci_run_id,
        ci_run_url=args.ci_run_url,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(
        f"[sbv2_atol_summary_from_log] wrote {len(entries)} entries to "
        f"{args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
