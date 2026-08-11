#!/usr/bin/env python3
"""SBV2 parity atol updater — proposes edits to
``crates/vokra-models/src/sbv2/parity.rs::PER_TENSOR_ATOL`` and
``tests/fixtures/sbv2/atol-measurements.json`` from a workflow-artifact
summary JSON emitted by ``.github/workflows/parity-sbv2-real.yml`` (WP-02,
2026-08-09).

# Contract: proposal-only, never auto-commits

Per memory ``feedback-honest-parity-atol`` (Kokoro T17-fixup #5/#6
REVERT precedent), CC does not silently loosen parity bounds. This
script therefore:

- **Reads** a machine-readable summary the workflow just produced
  (per-tensor `max_diff` + `current_atol` + `calibration_status` +
  `verdict`).
- **Computes** what the atol table + baseline file COULD look like
  after an empirical measurement (measured × margin factor, default
  1.6× — the honest-atol memory's 1.5-2× midpoint).
- **Prints** a human-reviewable proposal.
- **Does NOT** mutate any tracked file. The owner reviews the
  proposal, decides which changes to accept (both, one, neither, or
  a modified value), and manually edits BOTH the code and the baseline
  in ONE commit per the redundant-recording rule.

# Usage

    # Standalone:
    uv run python tools/parity/sbv2_atol_updater.py --summary <path.json>

    # In-workflow: consumes the artifact the parity workflow uploaded.
    # The step following the parity run reads the same JSON and pipes
    # it through this script for the step summary.

# Non-goals

- No auto-commit / no in-place mutation of `PER_TENSOR_ATOL` or the
  baseline file (see contract above).
- No fabricated proposals — a summary entry missing `max_diff` is a
  loud error (FR-EX-08), not a silently-skipped tensor.
- No dispatch on tensor NAME (e.g. "waveform gets a special path"):
  the proposal logic is uniform across all tensors, gated only by the
  input `calibration_status` and `verdict` values. Per-tensor rustdoc
  derivations live in `PER_TENSOR_ATOL` and stay there.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import IO, Any

# The margin factor applied when converting a measured max|Δ| into a
# proposed atol. Kokoro precedent: `PROSODY_F0_ATOL = 0.05 =
# 3.27e-2 × ~1.5-1.85×`. We take the middle of that 1.5-2× range so
# the proposal is neither the tightest possible bound (a single
# hardware/libm variation flip red) nor a lazy 2× wraparound.
DEFAULT_MARGIN_FACTOR: float = 1.6

# When a `Measured` tensor's proposed atol is within ±10% of its
# current atol, we emit NO proposal — spurious tightening/loosening
# is exactly the "CI green so tighten" pattern the honest-atol memory
# warns against. Owners who genuinely want to promote a tiny change
# can edit both files by hand; the drift detector's ±50% band still
# allows that.
NO_CHANGE_BAND: float = 0.10


def propose_from_summary(
    summary: dict[str, Any],
    *,
    margin_factor: float = DEFAULT_MARGIN_FACTOR,
    no_change_band: float = NO_CHANGE_BAND,
) -> list[dict[str, Any]]:
    """Walks the summary's ``entries`` list and returns a list of
    proposed atol updates. Pure: does not touch the filesystem.

    Each proposal dict has keys:
      - ``name``: tensor name.
      - ``action``: one of ``"PROMOTE_TO_ESTIMATED"``,
        ``"PROMOTE_TO_MEASURED"``, ``"TIGHTEN"``, or ``"LOOSEN"``.
      - ``current_atol``: the atol the summary observed.
      - ``current_status``: the calibration status the summary observed.
      - ``measured_max_diff``: the ``max_diff`` from the summary.
      - ``margin_factor``: the multiplier used for ``suggested_atol``.
      - ``suggested_atol``: ``measured_max_diff × margin_factor``.
      - ``verdict``: ``"PASS"`` or ``"FAIL"`` from the summary.

    Raises ``ValueError`` on any malformed input — per FR-EX-08 (no
    silent skipping).
    """
    if not isinstance(summary, dict):
        raise ValueError(
            f"summary must be a JSON object (dict), got {type(summary).__name__}"
        )
    entries = summary.get("entries")
    if entries is None:
        raise ValueError(
            "summary is missing required key `entries` — the workflow-artifact "
            "schema is `{\"entries\": [{\"name\": ..., \"max_diff\": ..., "
            "\"current_atol\": ..., \"calibration_status\": ..., "
            "\"verdict\": ...}, ...]}`"
        )
    if not isinstance(entries, list):
        raise ValueError(
            f"summary.entries must be a JSON array, got {type(entries).__name__}"
        )

    proposals: list[dict[str, Any]] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(
                f"summary.entries[{i}] must be a JSON object, got "
                f"{type(entry).__name__}"
            )
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"summary.entries[{i}] is missing a non-empty `name` string"
            )
        # max_diff / current_atol are the only NUMERIC fields we validate
        # strictly — status + verdict pass through untouched.
        if "max_diff" not in entry:
            raise ValueError(
                f"summary.entries[{i}] (name={name!r}) is missing required "
                "numeric field `max_diff`"
            )
        max_diff = entry["max_diff"]
        if not isinstance(max_diff, (int, float)):
            raise ValueError(
                f"summary.entries[{i}] (name={name!r}) has non-numeric "
                f"`max_diff` = {max_diff!r}"
            )
        if max_diff < 0:
            raise ValueError(
                f"summary.entries[{i}] (name={name!r}) has negative "
                f"`max_diff` = {max_diff} — absolute differences are "
                "always non-negative"
            )
        if "current_atol" not in entry:
            raise ValueError(
                f"summary.entries[{i}] (name={name!r}) is missing required "
                "numeric field `current_atol`"
            )
        current_atol = entry["current_atol"]
        if not isinstance(current_atol, (int, float)) or current_atol <= 0:
            raise ValueError(
                f"summary.entries[{i}] (name={name!r}) has non-positive "
                f"`current_atol` = {current_atol!r}"
            )

        # Zero measurement = bit-identical, no meaningful proposal.
        # `every_atol_value_is_positive` on the Rust side would reject
        # a zero atol; a proposal of "0 × 1.6 = 0" would be nonsense.
        if max_diff == 0:
            continue

        status = entry.get("calibration_status", "")
        verdict = entry.get("verdict", "")
        suggested = float(max_diff) * margin_factor
        # Ratio of change: same shape as the Rust drift-detector's
        # calculation, so proposals crossing the ±50% baseline drift
        # gate are obvious in the output.
        drift_ratio = abs(suggested - current_atol) / current_atol

        # Rule 1: FAIL always proposes LOOSEN (the current bound was
        # exceeded — the measured value is what the new proposed atol
        # would need to bracket).
        if verdict == "FAIL":
            action = "LOOSEN"
        # Rule 2: pass-through defaults with a real measurement graduate
        # to `EstimatedPreFixture` (the calibration hole closes).
        elif status == "UnmeasuredDefault":
            action = "PROMOTE_TO_ESTIMATED"
        # Rule 3: EstimatedPreFixture with a passing measurement
        # graduates to `Measured` — that IS the empirical-measurement
        # cycle's happy path.
        elif status == "EstimatedPreFixture":
            action = "PROMOTE_TO_MEASURED"
        # Rule 4: already-`Measured` with a significant drift (outside
        # the no-change band) proposes TIGHTEN (or LOOSEN, but this arm
        # is only reached when verdict == PASS so LOOSEN is impossible
        # here — a passing measurement is strictly ≤ current_atol).
        elif status == "Measured":
            if drift_ratio <= no_change_band:
                continue  # no proposal — current bound still valid
            action = "TIGHTEN" if suggested < current_atol else "LOOSEN"
        # Rule 5: unknown status — refuse to guess. Owner can still add
        # a bound by hand; the updater's silence is honest.
        else:
            continue

        proposals.append({
            "name": name,
            "action": action,
            "current_atol": current_atol,
            "current_status": status,
            "measured_max_diff": max_diff,
            "margin_factor": margin_factor,
            "suggested_atol": suggested,
            "verdict": verdict,
        })
    return proposals


def load_and_propose(
    summary_path: Path,
    *,
    margin_factor: float = DEFAULT_MARGIN_FACTOR,
) -> list[dict[str, Any]]:
    """Reads a summary JSON file and returns the proposals.
    Convenience wrapper for :func:`propose_from_summary`."""
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    return propose_from_summary(summary, margin_factor=margin_factor)


def render_proposals(
    proposals: list[dict[str, Any]],
    *,
    out: IO[str] = sys.stdout,
    ci_run_url: str | None = None,
) -> None:
    """Writes a human-reviewable summary of ``proposals`` to ``out``.

    Every proposal names BOTH files the owner needs to edit
    (``crates/vokra-models/src/sbv2/parity.rs`` and
    ``tests/fixtures/sbv2/atol-measurements.json``) — the redundant-
    recording rule (memory ``feedback-honest-parity-atol``) requires
    both moves in the same commit.
    """
    header = "=" * 72
    out.write(f"\n{header}\n")
    out.write("SBV2 atol updater — PROPOSED changes (owner review required)\n")
    out.write(f"{header}\n\n")
    if ci_run_url:
        out.write(f"Source measurement: {ci_run_url}\n\n")
    if not proposals:
        out.write("No changes proposed — every measured tensor is within the\n")
        out.write("no-change band of its current atol, or every summary entry\n")
        out.write("was zero-measured (bit-identical).\n")
        out.write(f"\n{header}\n")
        return

    for i, p in enumerate(proposals, start=1):
        out.write(f"[{i}/{len(proposals)}] {p['name']} ({p['action']})\n")
        out.write(f"    current atol:       {p['current_atol']}\n")
        out.write(f"    current status:     {p['current_status']}\n")
        out.write(f"    measured max|Δ|:    {p['measured_max_diff']}\n")
        out.write(f"    margin factor:      {p['margin_factor']}×\n")
        out.write(f"    suggested atol:     {p['suggested_atol']}\n")
        out.write(f"    verdict:            {p['verdict']}\n")
        out.write("    files to update TOGETHER (owner review required, no auto-commit):\n")
        out.write("      - crates/vokra-models/src/sbv2/parity.rs\n")
        out.write("            (PER_TENSOR_ATOL entry + atol_calibration_for arm + rustdoc\n")
        out.write("             derivation block; do NOT lose the derivation on revision)\n")
        out.write("      - tests/fixtures/sbv2/atol-measurements.json\n")
        out.write("            (per_tensor_atol value + provenance block)\n")
        out.write("      - docs/adr/sbv2-parity-atol.md §5 (owner narrative + rationale)\n")
        out.write("\n")

    out.write(f"{header}\n")
    out.write("REMEMBER: this is a PROPOSAL. CC never auto-commits atol changes\n")
    out.write("(memory `feedback-honest-parity-atol`, Kokoro T17-fixup #5/#6\n")
    out.write("REVERT precedent). Owner reviews each proposal, decides which to\n")
    out.write("accept, then edits BOTH files (code + baseline) in ONE commit.\n")
    out.write(f"{header}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "SBV2 parity atol updater. Reads the JSON summary produced by "
            "`.github/workflows/parity-sbv2-real.yml` and PROPOSES atol table "
            "updates — never auto-commits (honest-atol discipline)."
        )
    )
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to the atol-summary.json artifact from the parity workflow.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN_FACTOR,
        help=(
            f"Multiplier applied to `measured_max_diff` to derive the "
            f"suggested atol (default: {DEFAULT_MARGIN_FACTOR}, midpoint of "
            "the honest-atol memory's 1.5-2× range)."
        ),
    )
    parser.add_argument(
        "--ci-run-url",
        default=None,
        help="Optional CI run URL to include in the header for traceability.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit the proposals as a JSON array on stdout instead of the "
            "human-readable renderer. Useful for piping into another tool."
        ),
    )
    args = parser.parse_args(argv)

    proposals = load_and_propose(args.summary, margin_factor=args.margin)

    if args.json:
        json.dump(proposals, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        render_proposals(proposals, ci_run_url=args.ci_run_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
