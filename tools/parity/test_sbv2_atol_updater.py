#!/usr/bin/env python3
"""Unit tests for ``sbv2_atol_updater.py`` (WP-02, 2026-08-09).

Pure-stdlib, network-free, filesystem-free (aside from ``tempfile``): the
tests hand-craft workflow-artifact JSON blobs and assert what the
updater proposes. This exercises every code path that runs inside
`.github/workflows/parity-sbv2-real.yml`'s owner-review step without
requiring a live parity run.

The updater is deliberately proposal-only per the honest-atol discipline
(memory ``feedback-honest-parity-atol``): CC never auto-commits atol
changes, so every code path here either **prints** a proposal to stdout
or **returns** a structured proposal dict — never mutates the tracked
baseline file itself.

Run: ``uv run python -m unittest tools.parity.test_sbv2_atol_updater``
     (or ``uv run python -m unittest discover tools/parity -p
     'test_sbv2_atol_*.py'`` for the ``discover`` variant).
"""
from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path

# Import the module under test (sibling script, not an installed package).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sbv2_atol_updater as updater  # noqa: E402


def make_summary(
    entries: list[dict],
    *,
    ci_run_id: str = "TEST-RUN-000",
    ci_run_url: str = "https://example.invalid/ci/000",
) -> dict:
    """Builds a well-formed atol-summary artifact matching the
    ``parity-sbv2-real.yml`` post-parity step's output schema."""
    return {
        "schema_version": 1,
        "ci": {"run_id": ci_run_id, "run_url": ci_run_url},
        "entries": entries,
    }


class ProposalTests(unittest.TestCase):
    def test_measured_tensor_within_10pct_of_current_atol_proposes_no_change(self):
        """A tensor whose measured max|Δ| is very close to what the atol
        would derive as `measured × 1.6` (~= current atol) MUST NOT
        propose a change — spurious tightening/loosening is the exact
        Kokoro T17-fixup #5/#6 REVERT pattern the honest-atol rule
        exists to prevent."""
        # Current waveform atol = 1.5 (Measured, 0.9248 × 1.6). If we
        # remeasure and get 0.94 (very close to 0.9248), the proposed
        # atol is 0.94 × 1.6 = 1.504 — well within a 10% no-change band.
        summary = make_summary([
            {
                "name": "waveform",
                "max_diff": 0.94,
                "current_atol": 1.5,
                "calibration_status": "Measured",
                "verdict": "PASS",
            }
        ])
        proposals = updater.propose_from_summary(summary)
        # PASS + within no-change band = no proposal emitted (the tensor
        # is simply "current bound still valid").
        self.assertEqual(proposals, [], f"expected no-change but got: {proposals}")

    def test_estimated_tensor_with_measurement_proposes_tightening(self):
        """A pre-fixture `EstimatedPreFixture` bound (e.g. bert_hidden_ja
        = 0.02) with a real max|Δ| an order of magnitude tighter (say
        1e-3) MUST propose a `PROMOTE_TO_MEASURED` with the new value
        (1e-3 × 1.6 = 1.6e-3) — this is exactly the owner-cycle
        tightening the whole WP-02 infrastructure exists to enable."""
        summary = make_summary([
            {
                "name": "bert_hidden_ja",
                "max_diff": 1e-3,
                "current_atol": 0.02,
                "calibration_status": "EstimatedPreFixture",
                "verdict": "PASS",
            }
        ])
        proposals = updater.propose_from_summary(summary)
        self.assertEqual(len(proposals), 1)
        p = proposals[0]
        self.assertEqual(p["name"], "bert_hidden_ja")
        self.assertEqual(p["action"], "PROMOTE_TO_MEASURED")
        # Suggested = measured × 1.6 (the updater's default margin;
        # documented alongside the honest-atol memory's 1.5-2× range).
        self.assertAlmostEqual(p["suggested_atol"], 1.6e-3, places=6)
        self.assertEqual(p["current_atol"], 0.02)

    def test_failed_tensor_proposes_loosening_with_measured_status(self):
        """A tensor whose max|Δ| EXCEEDED the current atol (FAIL verdict)
        needs a bigger bound. Since CC never auto-commits, this is a
        proposed loosening the owner reviews — but the proposal itself
        must include the suggested new atol so a comparison is easy."""
        # Waveform=1.5 but measurement was 2.5 (way over) → propose
        # 2.5 × 1.6 = 4.0. Owner reviews and decides whether to accept
        # (real cross-platform libm drift got worse) or reject (regression
        # to fix in code).
        summary = make_summary([
            {
                "name": "waveform",
                "max_diff": 2.5,
                "current_atol": 1.5,
                "calibration_status": "Measured",
                "verdict": "FAIL",
            }
        ])
        proposals = updater.propose_from_summary(summary)
        self.assertEqual(len(proposals), 1)
        p = proposals[0]
        self.assertEqual(p["name"], "waveform")
        self.assertEqual(p["action"], "LOOSEN")
        self.assertAlmostEqual(p["suggested_atol"], 4.0, places=6)

    def test_unmeasured_default_tensor_with_measurement_proposes_derivation(self):
        """An `UnmeasuredDefault` tensor (e.g. `phoneme_embed` falling
        back to ATOL_DEFAULT=0.01) whose actual max|Δ| turns out to be
        much tighter (say 5e-5) SHOULD propose promoting it to an
        `EstimatedPreFixture` entry with the measured × 1.6 value —
        this closes one of WP-01's 6 pass-through calibration holes."""
        summary = make_summary([
            {
                "name": "phoneme_embed",
                "max_diff": 5e-5,
                "current_atol": 0.01,
                "calibration_status": "UnmeasuredDefault",
                "verdict": "PASS",
            }
        ])
        proposals = updater.propose_from_summary(summary)
        self.assertEqual(len(proposals), 1)
        p = proposals[0]
        self.assertEqual(p["name"], "phoneme_embed")
        self.assertEqual(p["action"], "PROMOTE_TO_ESTIMATED")
        self.assertAlmostEqual(p["suggested_atol"], 8e-5, places=8)

    def test_measured_tensor_with_significant_drift_proposes_update(self):
        """Waveform=1.5 (Measured, from 0.9248 × 1.6) but a new run gets
        0.4 (much tighter). Ratio 1.5 / (0.4×1.6) = 2.34× drift — well
        outside the no-change band. Propose TIGHTEN with suggested new
        atol 0.4 × 1.6 = 0.64."""
        summary = make_summary([
            {
                "name": "waveform",
                "max_diff": 0.4,
                "current_atol": 1.5,
                "calibration_status": "Measured",
                "verdict": "PASS",
            }
        ])
        proposals = updater.propose_from_summary(summary)
        self.assertEqual(len(proposals), 1)
        p = proposals[0]
        self.assertEqual(p["name"], "waveform")
        self.assertEqual(p["action"], "TIGHTEN")
        self.assertAlmostEqual(p["suggested_atol"], 0.64, places=6)

    def test_never_auto_commits_baseline_file(self):
        """`propose_from_summary` is pure: input is a JSON dict, output
        is a list of dicts. Nothing on the filesystem MAY be touched
        (memory `feedback-honest-parity-atol`: CC does not silently
        loosen; owner reviews)."""
        # We do not need to feed a real baseline path — propose_from_
        # summary is pure by construction. This test is a signature-
        # level contract check: the function is called without any
        # writable filesystem argument.
        summary = make_summary([
            {
                "name": "z_latent",
                "max_diff": 5e-3,
                "current_atol": 0.03,
                "calibration_status": "EstimatedPreFixture",
                "verdict": "PASS",
            }
        ])
        # No `baseline_path` in the signature: the ONLY inputs are the
        # in-memory summary dict + optional kwargs (margin factor etc.).
        proposals = updater.propose_from_summary(summary)
        self.assertIsInstance(proposals, list)
        # Sanity: the pure function did produce output.
        self.assertEqual(len(proposals), 1)

    def test_render_proposal_output_is_human_reviewable(self):
        """The stdout renderer prints a full narrative for each proposal,
        naming the tensor, the current value, the proposed value, the
        derivation (measured × margin), and the exact file paths the
        owner needs to edit — never a bare `bert_hidden_ja: 0.02 ->
        0.0016` diff. Kokoro precedent: PROSODY_F0_ATOL = 0.05 has a
        multi-paragraph rustdoc block that would silently rot away if
        the updater proposed edits without pointing at it."""
        proposals = [
            {
                "name": "bert_hidden_ja",
                "action": "PROMOTE_TO_MEASURED",
                "current_atol": 0.02,
                "current_status": "EstimatedPreFixture",
                "measured_max_diff": 1e-3,
                "margin_factor": 1.6,
                "suggested_atol": 1.6e-3,
                "verdict": "PASS",
            }
        ]
        buf = io.StringIO()
        updater.render_proposals(proposals, out=buf, ci_run_url="https://ci.example/999")
        text = buf.getvalue()
        # Bare tensor name + both file paths + measurement provenance.
        self.assertIn("bert_hidden_ja", text)
        self.assertIn("PROMOTE_TO_MEASURED", text)
        self.assertIn("0.02", text)  # current
        # Suggested atol printed either as literal or scientific;
        # accept either format the renderer chooses.
        self.assertTrue(
            "0.0016" in text or "1.6e-03" in text or "1.6e-3" in text,
            f"suggested atol not visible in output: {text!r}",
        )
        # Both files the owner will need to edit.
        self.assertIn("crates/vokra-models/src/sbv2/parity.rs", text)
        self.assertIn("tests/fixtures/sbv2/atol-measurements.json", text)
        # Explicit "no auto-commit" reminder.
        self.assertIn("owner review", text.lower())

    def test_missing_required_summary_key_is_actionable_error(self):
        """A summary JSON missing the top-level `entries` key MUST loud-
        fail with an actionable message naming the missing key — never
        a silent empty proposal (FR-EX-08)."""
        with self.assertRaises(ValueError) as ctx:
            updater.propose_from_summary({"schema_version": 1})
        self.assertIn("entries", str(ctx.exception))

    def test_entry_missing_max_diff_is_actionable_error(self):
        """A per-entry dict missing `max_diff` MUST loud-fail — the
        proposal cannot be computed without it."""
        summary = make_summary([{"name": "bert_hidden_ja"}])
        with self.assertRaises(ValueError) as ctx:
            updater.propose_from_summary(summary)
        self.assertIn("max_diff", str(ctx.exception))

    def test_zero_measured_max_diff_produces_no_proposal(self):
        """A zero measurement means "bit-identical" — no positive atol
        can wrap that meaningfully via `× margin`, so the updater emits
        no proposal (the owner already has a good bound; a zero-based
        atol would be silently unmeetable per
        `every_atol_value_is_positive` on the Rust side)."""
        summary = make_summary([
            {
                "name": "text_hidden",
                "max_diff": 0.0,
                "current_atol": 0.01,
                "calibration_status": "UnmeasuredDefault",
                "verdict": "PASS",
            }
        ])
        proposals = updater.propose_from_summary(summary)
        self.assertEqual(proposals, [])

    def test_negative_max_diff_is_actionable_error(self):
        """A negative max|Δ| is nonsensical (absolute difference is
        always non-negative). Reject with a clear message."""
        summary = make_summary([
            {
                "name": "text_hidden",
                "max_diff": -0.001,
                "current_atol": 0.01,
                "calibration_status": "UnmeasuredDefault",
                "verdict": "PASS",
            }
        ])
        with self.assertRaises(ValueError) as ctx:
            updater.propose_from_summary(summary)
        self.assertIn("max_diff", str(ctx.exception))


class SummaryParsingTests(unittest.TestCase):
    """End-to-end: read a summary JSON file, walk it, propose."""

    def test_parses_summary_json_file_end_to_end(self):
        """The main entry point reads a summary JSON path and produces
        the same proposals list as `propose_from_summary` would with the
        same dict."""
        import tempfile

        summary_dict = make_summary([
            {
                "name": "bert_hidden_ja",
                "max_diff": 5e-4,
                "current_atol": 0.02,
                "calibration_status": "EstimatedPreFixture",
                "verdict": "PASS",
            }
        ])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(summary_dict, f)
            path = f.name

        try:
            proposals = updater.load_and_propose(Path(path))
        finally:
            Path(path).unlink()

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["name"], "bert_hidden_ja")
        self.assertEqual(proposals[0]["action"], "PROMOTE_TO_MEASURED")


if __name__ == "__main__":
    unittest.main()
