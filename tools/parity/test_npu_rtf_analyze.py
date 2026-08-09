#!/usr/bin/env python3
"""test_npu_rtf_analyze.py — unit tests for ``npu_rtf_analyze.py``.

Stdlib only (`unittest`) to preserve the NFR-DS-02 zero-dep constraint on
the parity tooling — no pytest, no numpy, no pandas.

The tests exercise the three scenarios called out in the WP-15 P2 spec:

1. **canned N=10 clean run** — every iteration succeeds, placement is
   above the 90 % floor, CV is low. Analyzer should report OK on both
   the CV and placement axes.
2. **flaky run (30 % CPU fallback)** — 3 / 10 iterations report
   ``ane_frac ≈ 0.55`` (below the 90 % floor). Analyzer must WARN on
   placement, per the FR-EX-08 disqualifies-the-run contract.
3. **noisy run (CV > 0.5)** — RTF samples span an order of magnitude
   with stable placement. Analyzer must WARN on CV.

The tests also cover a few adversarial edge cases (missing probe, all
failed iterations, malformed placement dict, QNN ``htp_frac`` vs the
legacy ``dsp_frac`` alias) so future refactors do not silently regress
these boundaries.

Run: ``python3 tools/parity/test_npu_rtf_analyze.py``. Exit code 0 = all
tests passed; non-zero = at least one failure with the standard
``unittest`` diagnostic block.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest


# Make the sibling analyzer importable without installing anything.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import npu_rtf_analyze as analyzer  # noqa: E402  (import-after-sys.path.insert)


# ---------------------------------------------------------------------------
# JSONL fixtures
# ---------------------------------------------------------------------------

def _iter_line(
    i: int,
    rtf: float,
    backend: str = "coreml",
    ane_frac: float = 0.98,
    gpu_frac: float = 0.01,
    cpu_frac: float = 0.01,
    include_placement: bool = True,
    htp_frac: float | None = None,
    include_dsp_alias: bool = False,
) -> str:
    """Produce one successful iteration JSON line.

    The keys mirror what ``npu_rtf_variance.sh`` actually emits so the
    tests break when the shell script contract drifts. Two backend-
    specific twists are supported:

    - ``htp_frac`` — when set, emits the QNN placement fraction instead
      of the CoreML one.
    - ``include_dsp_alias`` — when True, uses the legacy ``dsp_frac``
      key rather than ``htp_frac``. This exercises the QNN back-compat
      path in ``_extract_npu_frac``.
    """
    obj: dict = {
        "iter": i,
        "timestamp": f"2026-08-10T10:00:{i:02d}Z",
        "status": "ok",
        "rtf": rtf,
        "latency_ms": rtf * 30_000.0,
        "backend": backend,
        "gguf": "/root/whisper.gguf",
        "audio": "/root/jfk-30s.wav",
        "host": "test-host",
        "device_name": "test-device",
        "device_os": "test-os",
        "device_soc": "test-soc",
        "label": "test",
    }
    if include_placement:
        if htp_frac is not None:
            key = "dsp_frac" if include_dsp_alias else "htp_frac"
            obj["placement"] = {key: htp_frac, "cpu_frac": 1.0 - htp_frac}
        else:
            obj["placement"] = {
                "ane_frac": ane_frac,
                "gpu_frac": gpu_frac,
                "cpu_frac": cpu_frac,
            }
    else:
        obj["placement"] = None
    obj["bench"] = {"rtf": rtf, "latency_ms": {"mean": rtf * 30_000.0}}
    return json.dumps(obj)


def _failure_line(i: int, backend: str = "coreml") -> str:
    return json.dumps({
        "iter": i,
        "timestamp": f"2026-08-10T10:00:{i:02d}Z",
        "status": "error",
        "exit_code": 3,
        "error": "backend unavailable",
        "backend": backend,
        "label": "test",
    })


def _summary_line(
    iters_requested: int,
    iters_failed: int,
    backend: str = "coreml",
    placement_probe: str = "/opt/probes/ane.sh",
) -> str:
    return json.dumps({
        "type": "summary",
        "iters_requested": iters_requested,
        "iters_failed": iters_failed,
        "started_at": "2026-08-10T10:00:00Z",
        "ended_at": "2026-08-10T10:03:00Z",
        "backend": backend,
        "label": "test",
        "host": "test-host",
        "device_name": "test-device",
        "device_os": "test-os",
        "device_soc": "test-soc",
        "gguf": "/root/whisper.gguf",
        "audio": "/root/jfk-30s.wav",
        "placement_probe": placement_probe,
    })


# ---------------------------------------------------------------------------
# Scenario 1: canned N=10 clean run
# ---------------------------------------------------------------------------

class CleanRunTests(unittest.TestCase):
    """N=10 successful iterations, placement >= 0.98, RTF ~ 0.15."""

    def setUp(self) -> None:
        base_rtf = 0.15
        # A little jitter — enough to be realistic without tripping CV.
        rtfs = [base_rtf + (0.001 * (i - 5)) for i in range(1, 11)]
        lines = [_iter_line(i, r) for i, r in enumerate(rtfs, start=1)]
        lines.append(_summary_line(10, 0))
        self.samples, self.failures, self.summary = analyzer.parse_jsonl(lines)

    def test_parses_ten_samples(self) -> None:
        self.assertEqual(len(self.samples), 10)
        self.assertEqual(len(self.failures), 0)
        self.assertIsNotNone(self.summary)
        assert self.summary is not None
        self.assertEqual(self.summary.iters_requested, 10)
        self.assertEqual(self.summary.iters_failed, 0)

    def test_cv_below_threshold(self) -> None:
        stats = analyzer.summarize([s.rtf for s in self.samples])
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertLess(stats.cv, analyzer.CV_WARN_THRESHOLD)

    def test_all_npu_fractions_valid(self) -> None:
        placement = analyzer.placement_report(self.samples)
        self.assertEqual(placement.total, 10)
        self.assertEqual(placement.with_probe, 10)
        self.assertEqual(placement.with_npu_frac, 10)
        self.assertEqual(placement.below_threshold, 0)
        self.assertIsNotNone(placement.mean_npu_frac)
        assert placement.mean_npu_frac is not None
        self.assertGreater(placement.mean_npu_frac, analyzer.PLACEMENT_WARN_THRESHOLD)

    def test_markdown_says_ok_on_both_axes(self) -> None:
        report = analyzer.format_markdown(self.samples, self.failures, self.summary)
        self.assertIn("OK: CV", report)
        # Placement OK banner mentions the kept-on-NPU count.
        self.assertIn("kept ≥ 90% of hot ops", report)
        # Silent-fallback WARN must NOT appear on a clean run.
        self.assertNotIn("silent-CPU-fallback pattern", report)

    def test_json_flag_matches_markdown_verdict(self) -> None:
        # Exercise the JSON output path via the CLI so future contract
        # drift in the report shape is caught.
        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, "clean.jsonl")
            with open(in_path, "w", encoding="utf-8") as fh:
                base_rtf = 0.15
                rtfs = [base_rtf + (0.001 * (i - 5)) for i in range(1, 11)]
                for i, r in enumerate(rtfs, start=1):
                    fh.write(_iter_line(i, r) + "\n")
                fh.write(_summary_line(10, 0) + "\n")

            out_path = os.path.join(td, "clean.json")
            rc = analyzer.main([in_path, "--output", out_path, "--format", "json"])
            self.assertEqual(rc, 0)

            with open(out_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

        self.assertFalse(data["stats"]["cv_warn"])
        self.assertFalse(data["placement"]["placement_warn"])
        self.assertEqual(data["placement"]["below_threshold"], 0)


# ---------------------------------------------------------------------------
# Scenario 2: flaky (30 % CPU fallback) → placement WARN
# ---------------------------------------------------------------------------

class FlakyRunTests(unittest.TestCase):
    """7 iters at ane_frac=0.98, 3 iters at ane_frac=0.55 (below floor)."""

    def setUp(self) -> None:
        lines: list[str] = []
        for i in range(1, 8):
            lines.append(_iter_line(i, 0.14, ane_frac=0.98))
        for i in range(8, 11):
            lines.append(_iter_line(i, 0.42, ane_frac=0.55, cpu_frac=0.44))
        lines.append(_summary_line(10, 0))
        self.samples, self.failures, self.summary = analyzer.parse_jsonl(lines)

    def test_flaky_iters_recorded(self) -> None:
        self.assertEqual(len(self.samples), 10)
        placement = analyzer.placement_report(self.samples)
        self.assertEqual(placement.below_threshold, 3)

    def test_placement_warn_fires(self) -> None:
        report = analyzer.format_markdown(self.samples, self.failures, self.summary)
        self.assertIn("silent-CPU-fallback pattern FR-EX-08 forbids", report)
        self.assertIn("3 / 10", report)
        # RTF WARN must NOT swallow the placement WARN — both are
        # independent axes and both should be reported.
        self.assertIn("Coefficient-of-variation warning", report)
        self.assertIn("Placement", report)

    def test_json_flag_marks_placement_warn(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, "flaky.jsonl")
            with open(in_path, "w", encoding="utf-8") as fh:
                for i in range(1, 8):
                    fh.write(_iter_line(i, 0.14, ane_frac=0.98) + "\n")
                for i in range(8, 11):
                    fh.write(
                        _iter_line(i, 0.42, ane_frac=0.55, cpu_frac=0.44) + "\n"
                    )
                fh.write(_summary_line(10, 0) + "\n")

            out_path = os.path.join(td, "flaky.json")
            rc = analyzer.main([in_path, "--output", out_path, "--format", "json"])
            self.assertEqual(rc, 0)

            with open(out_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

        self.assertTrue(data["placement"]["placement_warn"])
        self.assertEqual(data["placement"]["below_threshold"], 3)


# ---------------------------------------------------------------------------
# Scenario 3: noisy (CV > 0.5) → CV WARN
# ---------------------------------------------------------------------------

class NoisyRunTests(unittest.TestCase):
    """Wide RTF spread with stable placement — CV WARN, placement OK."""

    def setUp(self) -> None:
        # Two clusters that produce a very high CV without exceeding
        # a single-order-of-magnitude spread — 5 slow + 5 fast.
        rtfs = [0.05, 0.06, 0.07, 0.08, 0.09,
                0.90, 1.00, 1.10, 1.20, 1.30]
        lines = [_iter_line(i, r) for i, r in enumerate(rtfs, start=1)]
        lines.append(_summary_line(10, 0))
        self.samples, self.failures, self.summary = analyzer.parse_jsonl(lines)

    def test_cv_over_threshold(self) -> None:
        stats = analyzer.summarize([s.rtf for s in self.samples])
        self.assertIsNotNone(stats)
        assert stats is not None
        # 5 × ~0.07 and 5 × ~1.10 => stddev roughly matches mean.
        self.assertGreater(stats.cv, analyzer.CV_WARN_THRESHOLD)

    def test_cv_warn_fires(self) -> None:
        report = analyzer.format_markdown(self.samples, self.failures, self.summary)
        self.assertIn("**WARNING**: CV", report)
        # Placement remains OK because the sample defaults are healthy.
        self.assertIn("kept ≥ 90% of hot ops", report)


# ---------------------------------------------------------------------------
# Edge cases (adversarial coverage)
# ---------------------------------------------------------------------------

class EdgeCaseTests(unittest.TestCase):

    def test_all_iterations_failed_returns_1(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, "allfail.jsonl")
            with open(in_path, "w", encoding="utf-8") as fh:
                for i in range(1, 11):
                    fh.write(_failure_line(i) + "\n")
                fh.write(_summary_line(10, 10) + "\n")
            out_path = os.path.join(td, "allfail.md")
            rc = analyzer.main([in_path, "--output", out_path, "--format", "markdown"])
        self.assertEqual(rc, 1)

    def test_missing_placement_probe_triggers_warn(self) -> None:
        lines: list[str] = [
            _iter_line(i, 0.14, include_placement=False) for i in range(1, 11)
        ]
        lines.append(_summary_line(10, 0, placement_probe=""))
        samples, failures, summary = analyzer.parse_jsonl(lines)
        report = analyzer.format_markdown(samples, failures, summary)
        self.assertIn("no iteration produced a valid NPU fraction", report)

    def test_qnn_htp_frac_recognised(self) -> None:
        lines: list[str] = [
            _iter_line(i, 0.20, backend="qnn", htp_frac=0.97) for i in range(1, 6)
        ]
        lines.append(_summary_line(5, 0, backend="qnn"))
        samples, _, _ = analyzer.parse_jsonl(lines)
        placement = analyzer.placement_report(samples)
        self.assertEqual(placement.with_npu_frac, 5)
        self.assertEqual(placement.below_threshold, 0)

    def test_qnn_dsp_frac_alias_recognised(self) -> None:
        # Older QNN profiler dumps use ``dsp_frac`` — the analyzer must
        # accept both to avoid a spurious "placement=unknown" WARN.
        lines: list[str] = [
            _iter_line(
                i, 0.22, backend="qnn", htp_frac=0.95, include_dsp_alias=True
            )
            for i in range(1, 4)
        ]
        lines.append(_summary_line(3, 0, backend="qnn"))
        samples, _, _ = analyzer.parse_jsonl(lines)
        placement = analyzer.placement_report(samples)
        self.assertEqual(placement.with_npu_frac, 3)

    def test_non_json_line_captured_as_failure(self) -> None:
        raw_lines = [
            "this is not json at all",
            _iter_line(1, 0.15),
            _summary_line(1, 0),
        ]
        samples, failures, summary = analyzer.parse_jsonl(raw_lines)
        self.assertEqual(len(samples), 1)
        self.assertEqual(len(failures), 1)
        self.assertIn("non-JSON line", failures[0].error)
        self.assertEqual(failures[0].iter, -1)

    def test_stdin_input(self) -> None:
        # Round-trip through ``sys.stdin`` to exercise the `-` code path.
        lines = [_iter_line(i, 0.15) for i in range(1, 4)]
        lines.append(_summary_line(3, 0))
        buf = io.StringIO("\n".join(lines) + "\n")
        old_stdin = sys.stdin
        old_stdout = sys.stdout
        sys.stdin = buf
        sys.stdout = io.StringIO()
        try:
            rc = analyzer.main(["-", "--format", "markdown"])
            out = sys.stdout.getvalue()
        finally:
            sys.stdin = old_stdin
            sys.stdout = old_stdout
        self.assertEqual(rc, 0)
        self.assertIn("NPU delegate RTF variance report", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
