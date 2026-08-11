#!/usr/bin/env python3
"""Unit tests for ``sbv2_atol_summary_from_log.py`` (WP-02, 2026-08-09).

Pure-stdlib: fabricates parity.log snippets in-memory and asserts what
:func:`sbv2_atol_summary_from_log.parse_log` produces. No shelling out
to `cargo test`, no filesystem writes.

Run: ``uv run python -m unittest test_sbv2_atol_summary_from_log``
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sbv2_atol_summary_from_log as sfl  # noqa: E402


class ParseLogTests(unittest.TestCase):
    def test_passing_row_is_parsed(self):
        log = (
            "some noise before\n"
            "[parity_sbv2_real] bert_hidden_ja: max |Δ| = 5.483e-06 atol 0.02 verdict PASS [EstimatedPreFixture]\n"
            "some noise after\n"
        )
        entries = sfl.parse_log(log)
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e["name"], "bert_hidden_ja")
        self.assertAlmostEqual(e["max_diff"], 5.483e-06, places=10)
        self.assertAlmostEqual(e["current_atol"], 0.02, places=6)
        self.assertEqual(e["calibration_status"], "EstimatedPreFixture")
        self.assertEqual(e["verdict"], "PASS")

    def test_failing_row_is_parsed_with_fail_verdict(self):
        log = (
            "[parity_sbv2_real] waveform: max |Δ| = 2.5e+00 atol 1.5 verdict FAIL [Measured]\n"
        )
        entries = sfl.parse_log(log)
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e["name"], "waveform")
        self.assertAlmostEqual(e["max_diff"], 2.5, places=6)
        self.assertEqual(e["verdict"], "FAIL")
        self.assertEqual(e["calibration_status"], "Measured")

    def test_unmeasured_default_marker_maps_to_short_status(self):
        """The parity test emits `[UnmeasuredDefault(ATOL_DEFAULT)]` for
        the 6 pass-through tensors; the JSON summary uses the shorter
        `UnmeasuredDefault` key so `sbv2_atol_updater.py`'s dispatch
        logic doesn't have to strip parenthesised suffixes."""
        log = (
            "[parity_sbv2_real] phoneme_embed: max |Δ| = 5.0e-05 atol 0.01 verdict PASS "
            "[UnmeasuredDefault(ATOL_DEFAULT)]\n"
        )
        entries = sfl.parse_log(log)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["calibration_status"], "UnmeasuredDefault")

    def test_all_ten_intermediate_rows_parsed(self):
        """A full parity run emits one row per intermediate — the JSON
        summary must carry all of them (order preserved)."""
        log = "\n".join([
            "[parity_sbv2_real] phoneme_embed: max |Δ| = 1.0e-07 atol 0.01 verdict PASS [UnmeasuredDefault(ATOL_DEFAULT)]",
            "[parity_sbv2_real] text_hidden: max |Δ| = 2.0e-07 atol 0.01 verdict PASS [UnmeasuredDefault(ATOL_DEFAULT)]",
            "[parity_sbv2_real] bert_hidden_ja: max |Δ| = 3.0e-06 atol 0.02 verdict PASS [EstimatedPreFixture]",
            "[parity_sbv2_real] bert_hidden_en: max |Δ| = 4.0e-06 atol 0.02 verdict PASS [EstimatedPreFixture]",
            "[parity_sbv2_real] bert_bridge_out: max |Δ| = 5.0e-07 atol 0.01 verdict PASS [UnmeasuredDefault(ATOL_DEFAULT)]",
            "[parity_sbv2_real] speaker_embed: max |Δ| = 6.0e-07 atol 0.01 verdict PASS [UnmeasuredDefault(ATOL_DEFAULT)]",
            "[parity_sbv2_real] style_projected: max |Δ| = 7.0e-07 atol 0.01 verdict PASS [UnmeasuredDefault(ATOL_DEFAULT)]",
            "[parity_sbv2_real] sdp_sample: max |Δ| = 8.0e-06 atol 0.05 verdict PASS [EstimatedPreFixture]",
            "[parity_sbv2_real] mel_hidden: max |Δ| = 9.0e-07 atol 0.01 verdict PASS [UnmeasuredDefault(ATOL_DEFAULT)]",
            "[parity_sbv2_real] z_latent: max |Δ| = 1.0e-05 atol 0.03 verdict PASS [EstimatedPreFixture]",
        ])
        entries = sfl.parse_log(log)
        self.assertEqual(len(entries), 10)
        names = [e["name"] for e in entries]
        self.assertEqual(names, [
            "phoneme_embed", "text_hidden", "bert_hidden_ja", "bert_hidden_en",
            "bert_bridge_out", "speaker_embed", "style_projected", "sdp_sample",
            "mel_hidden", "z_latent",
        ])

    def test_lines_without_parity_prefix_are_ignored(self):
        log = (
            "hello world\n"
            "cargo test output\n"
            "[not_parity_sbv2_real] noise\n"
        )
        entries = sfl.parse_log(log)
        self.assertEqual(entries, [])

    def test_waveform_summary_line_is_intentionally_skipped(self):
        """The waveform block emits its OWN informational line
        (`[parity_sbv2_real] waveform parity OK: rust=... samples ...`)
        that does not use the per-tensor `max |Δ| = ...` format. It
        must be skipped without emitting an "unrecognized row" warning."""
        log = (
            "[parity_sbv2_real] waveform parity OK: rust=27136 samples "
            "ref=27136 samples (ratio 1.0000, band ±10%), overlap 27136 "
            "samples: max |Δ| = 5.000e-01, RMS |Δ| = 1.000e-02 <= atol 1.5\n"
        )
        entries = sfl.parse_log(log)
        # Zero entries because this line intentionally does not match
        # ROW_RE (multi-metric format). If it DID match, we'd double-
        # count waveform (the intermediate loop also emits a row for
        # every tensor except waveform, which uses the dedicated block).
        self.assertEqual(entries, [])

    def test_empty_input_produces_empty_summary(self):
        entries = sfl.parse_log("")
        self.assertEqual(entries, [])

    def test_row_with_integer_atol_still_parses(self):
        """`atol` in the log format is `.6e` for max_diff but a bare
        float for atol (`0.02`, `1.5`, ...). A future-safe integer
        (`1`) should still parse."""
        log = (
            "[parity_sbv2_real] some_tensor: max |Δ| = 1.0e-03 atol 1 verdict PASS [Measured]\n"
        )
        entries = sfl.parse_log(log)
        self.assertEqual(len(entries), 1)
        self.assertAlmostEqual(entries[0]["current_atol"], 1.0, places=6)


class BuildSummaryTests(unittest.TestCase):
    def test_top_level_schema_matches_updater_expectations(self):
        summary = sfl.build_summary(
            entries=[{"name": "foo", "max_diff": 0.1, "current_atol": 1.0,
                      "calibration_status": "Measured", "verdict": "PASS"}],
            ci_run_id="123",
            ci_run_url="https://ci.example/123",
        )
        self.assertEqual(summary["schema_version"], 1)
        self.assertEqual(summary["ci"]["run_id"], "123")
        self.assertEqual(summary["ci"]["run_url"], "https://ci.example/123")
        self.assertEqual(len(summary["entries"]), 1)

    def test_none_ci_fields_become_empty_string(self):
        """The updater's `render_proposals` accepts `None` for
        `ci_run_url`; the build helper normalises to an empty string
        so downstream consumers do not have to distinguish the two."""
        summary = sfl.build_summary(entries=[], ci_run_id=None, ci_run_url=None)
        self.assertEqual(summary["ci"]["run_id"], "")
        self.assertEqual(summary["ci"]["run_url"], "")


class EndToEndTests(unittest.TestCase):
    def test_summary_from_parse_log_feeds_updater_cleanly(self):
        """Cross-check: the entries `parse_log` emits are directly
        consumable by `sbv2_atol_updater.propose_from_summary` — no
        rename/adapt step in between. This locks the two scripts'
        contract."""
        import sbv2_atol_updater as updater

        log = (
            "[parity_sbv2_real] bert_hidden_ja: max |Δ| = 5.0e-04 atol 0.02 "
            "verdict PASS [EstimatedPreFixture]\n"
        )
        entries = sfl.parse_log(log)
        summary = sfl.build_summary(entries, ci_run_id="TEST", ci_run_url=None)
        # Must succeed (no ValueError on any missing key).
        proposals = updater.propose_from_summary(summary)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["name"], "bert_hidden_ja")
        self.assertEqual(proposals[0]["action"], "PROMOTE_TO_MEASURED")


if __name__ == "__main__":
    unittest.main()
