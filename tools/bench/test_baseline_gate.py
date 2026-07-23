#!/usr/bin/env python3
"""Oracle for the placeholder-gate (X-06-T10).

Pins the fabricated-pass guard mechanically: an un-seeded baseline must
CLEAN-SKIP (classified "placeholder"), a seeded one must GATE (classified
"seeded"), and anything malformed must FAIL LOUDLY — never silently read as a
pass. Also proves the classifier is not a no-op by feeding it inputs that must
raise.

stdlib-only. Run: python3 tools/bench/test_baseline_gate.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_gate as bg  # noqa: E402


def write(tmp: Path, obj) -> Path:
    p = tmp / "b.json"
    if isinstance(obj, str):
        p.write_text(obj, encoding="utf-8")
    else:
        p.write_text(json.dumps(obj), encoding="utf-8")
    return p


class TestClassify(unittest.TestCase):
    def test_placeholder_is_a_skip(self):
        with tempfile.TemporaryDirectory() as td:
            p = write(Path(td), {"$placeholder": True, "task": "vad", "rtf": None})
            self.assertEqual(bg.classify(p), "placeholder")

    def test_a_real_report_with_numeric_rtf_is_seeded(self):
        with tempfile.TemporaryDirectory() as td:
            p = write(Path(td), {"task": "vad", "iters": 30, "rtf": 0.0034})
            self.assertEqual(bg.classify(p), "seeded")

    def test_integer_rtf_is_seeded_too(self):
        with tempfile.TemporaryDirectory() as td:
            p = write(Path(td), {"rtf": 0})
            self.assertEqual(bg.classify(p), "seeded")

    def test_missing_file_raises_not_skips(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(bg.BaselineError):
                bg.classify(Path(td) / "nope.json")

    def test_malformed_neither_placeholder_nor_rtf_raises(self):
        """A file that is neither is a DEFECT — it must fail, not be treated as
        a skip (which would hide a broken baseline) or a pass."""
        with tempfile.TemporaryDirectory() as td:
            p = write(Path(td), {"task": "vad", "note": "someone deleted rtf"})
            with self.assertRaises(bg.BaselineError):
                bg.classify(p)

    def test_rtf_null_without_placeholder_flag_raises(self):
        """`rtf: null` alone is not a valid seeded baseline and is not marked a
        placeholder — it must fail loudly rather than pass."""
        with tempfile.TemporaryDirectory() as td:
            p = write(Path(td), {"task": "vad", "rtf": None})
            with self.assertRaises(bg.BaselineError):
                bg.classify(p)

    def test_rtf_as_bool_is_not_a_number(self):
        with tempfile.TemporaryDirectory() as td:
            p = write(Path(td), {"rtf": True})
            with self.assertRaises(bg.BaselineError):
                bg.classify(p)

    def test_ambiguous_placeholder_with_numeric_rtf_raises(self):
        """A half-seeded file (placeholder flag AND a number) is ambiguous and
        could read as a pass — reject it."""
        with tempfile.TemporaryDirectory() as td:
            p = write(Path(td), {"$placeholder": True, "rtf": 0.003})
            with self.assertRaises(bg.BaselineError):
                bg.classify(p)

    def test_non_object_json_raises(self):
        with tempfile.TemporaryDirectory() as td:
            p = write(Path(td), "[1, 2, 3]")
            with self.assertRaises(bg.BaselineError):
                bg.classify(p)

    def test_unparsable_json_raises(self):
        with tempfile.TemporaryDirectory() as td:
            p = write(Path(td), "{not json")
            with self.assertRaises(bg.BaselineError):
                bg.classify(p)


class TestShippedPlaceholders(unittest.TestCase):
    """The baselines committed by this WP must be genuine placeholders until
    the owner seeds them from ubuntu-latest CI — a committed numeric baseline
    from any other rig is the exact false-regression the README warns about."""

    def test_silero_baseline_ships_as_a_placeholder(self):
        p = REPO / "docs" / "bench-baselines" / "silero_vad_baseline.json"
        self.assertTrue(p.is_file(), f"missing {p}")
        self.assertEqual(bg.classify(p), "placeholder")

    def test_whisper_base_nightly_baseline_ships_as_a_placeholder(self):
        p = REPO / "docs" / "bench-baselines" / "whisper_base_asr_nightly_baseline.json"
        self.assertTrue(p.is_file(), f"missing {p}")
        self.assertEqual(bg.classify(p), "placeholder")


class TestCli(unittest.TestCase):
    def _run(self, arg):
        return subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent / "baseline_gate.py"), str(arg)],
            capture_output=True,
            text=True,
        )

    def test_cli_prints_placeholder_and_exits_zero(self):
        with tempfile.TemporaryDirectory() as td:
            p = write(Path(td), {"$placeholder": True, "rtf": None})
            res = self._run(p)
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertEqual(res.stdout.strip(), "placeholder")

    def test_cli_prints_seeded_and_exits_zero(self):
        with tempfile.TemporaryDirectory() as td:
            p = write(Path(td), {"rtf": 0.0034})
            res = self._run(p)
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertEqual(res.stdout.strip(), "seeded")

    def test_cli_on_malformed_exits_nonzero_so_a_set_e_step_fails(self):
        """This is the mechanical proof the gate is not a no-op: a broken
        baseline exits 2, so `STATE=$(baseline_gate.py ...)` under `set -e`
        aborts the CI step rather than silently skipping."""
        with tempfile.TemporaryDirectory() as td:
            p = write(Path(td), {"note": "no rtf"})
            res = self._run(p)
            self.assertEqual(res.returncode, 2)
            self.assertIn("::error::", res.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
