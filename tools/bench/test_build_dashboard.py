#!/usr/bin/env python3
"""Oracle for the benchmark dashboard generator (X-06-T15).

Pins the three properties the deliverable must have and the two red lines it
must not cross:

  * renders structured rows from a synthetic corpus (real data flows through);
  * a MISSING input becomes a "no data" cell, never a fabricated number
    (FR-EX-08) — proved by feeding an empty tree;
  * the output is self-contained — self_check() finds no external reference —
    AND self_check is NOT a no-op: it flags a doctored page that loads an
    external script / stylesheet / image / font.

stdlib-only. Run: python3 tools/bench/test_build_dashboard.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_dashboard as bd  # noqa: E402


def make_corpus(root: Path):
    """A minimal repo tree the loaders can read."""
    (root / "docs" / "bench-baselines" / "m5-14-final-2026-07-18").mkdir(parents=True)
    (root / "docs" / "perf").mkdir(parents=True)
    (root / "docs" / "benchmarks").mkdir(parents=True)
    # A CI-gate baseline, an M1-rig baseline, and a placeholder.
    bb = root / "docs" / "bench-baselines"
    (bb / "mel_frontend_baseline.json").write_text(
        json.dumps({"task": "mel-frontend", "rtf": 0.003115}), encoding="utf-8"
    )
    (bb / "m5-14-final-2026-07-18" / "silero-vad.m1.baseline.json").write_text(
        json.dumps({"provenance": "M1-RIG-SCOPED ...", "task": "vad", "rtf": 0.003343}),
        encoding="utf-8",
    )
    (bb / "silero_vad_baseline.json").write_text(
        json.dumps({"$placeholder": True, "task": "vad", "rtf": None}), encoding="utf-8"
    )
    # A GPU perf file.
    (root / "docs" / "perf" / "cuda.json").write_text(
        json.dumps({"model": "whisper-large-v3", "backend": "cuda", "hardware": "RTX 4090",
                    "median_rtf": 0.1133, "measured_at": "2026-07-07", "gate_status": "preparation"}),
        encoding="utf-8",
    )
    # An rtf jsonl (two iterations, same label).
    (bb / "m5-14-final-2026-07-18" / "rtf-decomposed.jsonl").write_text(
        "\n".join(json.dumps({"rtf": r, "label": "decomposed"}) for r in (0.78, 0.79)) + "\n",
        encoding="utf-8",
    )
    # An M5-14 report with the vs-ORT table.
    (bb / "m5-14-final-2026-07-18" / "report.md").write_text(
        "# report\n\n"
        "| model (leg) | Wave-0 | now | speedup | vs-ORT | target | met? |\n"
        "|---|---|---|---|---|---|---|\n"
        "| whisper base | 1.9 s | **0.81 s** | 2.40x | **0.41x** | none | MET |\n",
        encoding="utf-8",
    )
    # A device scaffold doc.
    (root / "docs" / "benchmarks" / "v0.9-device-benchmarks.md").write_text(
        "# v0.9 Device Benchmarks\n\n> scaffold — 実測値は依頼者引き渡し後に追記。\n", encoding="utf-8"
    )


class TestRendering(unittest.TestCase):
    def test_renders_structured_rows_from_a_synthetic_corpus(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_corpus(root)
            page = bd.build_html(root, "test")
            # Real values flow through.
            self.assertIn("0.003115", page)      # mel-frontend gate baseline
            self.assertIn("whisper-large-v3", page)  # gpu perf
            self.assertIn("0.1133", page)        # gpu median rtf
            self.assertIn("whisper base", page)  # m5-14 table row
            self.assertIn("0.41x", page)         # m5-14 vs-ORT cell (bold stripped)
            # Provenance classes are distinguished.
            self.assertIn("CI gate (ubuntu-latest, PR-blocking)", page)
            self.assertIn("M1-rig reference (NOT a CI gate)", page)
            self.assertIn("placeholder (unseeded — no verdict)", page)

    def test_placeholder_rtf_is_no_data_not_a_fabricated_number(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_corpus(root)
            page = bd.build_html(root, "test")
            # The placeholder row must show the no-data span for rtf, not 0 or null.
            self.assertIn("no data", page)
            self.assertNotIn(">null<", page)
            self.assertNotIn(">None<", page)

    def test_empty_tree_renders_no_data_never_invents_values(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "docs").mkdir()
            page = bd.build_html(root, "test")
            # Every section is present but empty → no-data cells.
            for title in ("CPU RTF vs ONNX Runtime", "GPU performance", "Committed regression baselines"):
                self.assertIn(title, page)
            self.assertIn("no data", page)
            # No stray numeric fabrication: the only digits are in the CSS / meta.
            self.assertNotIn("0.1133", page)


class TestSelfContained(unittest.TestCase):
    def test_generated_page_has_no_external_references(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_corpus(root)
            page = bd.build_html(root, "test")
            self.assertEqual(bd.self_check(page), [], "generated page must be self-contained")

    def test_real_repo_corpus_is_self_contained(self):
        page = bd.build_html(REPO, "test")
        self.assertEqual(bd.self_check(page), [])

    def test_self_check_flags_an_external_script(self):
        bad = '<html><head><script src="https://cdn.example.com/x.js"></script></head></html>'
        self.assertNotEqual(bd.self_check(bad), [], "self_check must catch an external script")

    def test_self_check_flags_an_external_stylesheet(self):
        bad = '<html><head><link rel="stylesheet" href="https://fonts.googleapis.com/x"></head></html>'
        self.assertNotEqual(bd.self_check(bad), [])

    def test_self_check_flags_an_external_image(self):
        bad = '<html><body><img src="//cdn.example.com/logo.png"></body></html>'
        self.assertNotEqual(bd.self_check(bad), [])

    def test_self_check_flags_a_css_import(self):
        bad = "<html><head><style>@import url(https://x/y.css);</style></head></html>"
        self.assertNotEqual(bd.self_check(bad), [])

    def test_self_check_allows_a_plain_text_url_in_prose(self):
        """A URL that is escaped TEXT (not an attribute) must not trip the
        check — provenance strings legitimately name openslr.org etc."""
        ok = "<html><body><p>corpus from https://www.openslr.org/12 (CC BY 4.0)</p></body></html>"
        self.assertEqual(bd.self_check(ok), [])


class TestCli(unittest.TestCase):
    def _run(self, *a):
        return subprocess.run(
            [sys.executable, str(HERE / "build_dashboard.py"), *a],
            capture_output=True, text=True,
        )

    def test_self_check_subcommand_passes_on_the_real_build(self):
        res = self._run("--self-check")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("self-check OK", res.stdout)

    def test_self_check_subcommand_fails_on_a_doctored_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.html"
            p.write_text('<script src="https://x/y.js"></script>', encoding="utf-8")
            res = self._run("--self-check", str(p))
            self.assertEqual(res.returncode, 1)
            self.assertIn("SELF-CHECK FAILED", res.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
