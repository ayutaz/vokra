#!/usr/bin/env python3
"""TDD oracle for the X-06 nightly ASR-WER leg.

Covers both halves of the leg:
  * `tools/eval/librispeech_wer.py` — corpus parsing, utterance pinning,
    the missing-input hard errors, and the verdict boundary.
  * `.github/workflows/nightly-asr-wer.yml` — the surface that cannot be
    unit-tested from Python (cron slot, advisory posture, scope note) is
    asserted against the YAML as text, and the workflow's inline bash
    subset-extraction block is EXTRACTED AND EXECUTED against a synthetic
    corpus tree so the test pins behaviour rather than formatting.

stdlib-only (zero-dep NFR-DS-02), matching
tools/parity/test_parity_whisper_workflow.py: the scoring stack (jiwer +
transformers) is NOT imported here, so this runs on any checkout. The
functions that need it are exercised by the workflow itself and by the
local end-to-end run recorded in the leg's authoring notes.

Run: python3 tools/eval/test_librispeech_wer.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "nightly-asr-wer.yml"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import librispeech_wer as lw  # noqa: E402

TRANS_BODY = "\n".join(
    f"1272-128104-{i:04d} REFERENCE TEXT NUMBER {i} WITH SEVERAL WORDS" for i in range(15)
)


def make_corpus(root: Path) -> Path:
    """Build a minimal dev-clean/1272/128104 tree with a 15-utterance chapter."""
    chapter = root / "dev-clean" / "1272" / "128104"
    chapter.mkdir(parents=True)
    (chapter / "1272-128104.trans.txt").write_text(TRANS_BODY + "\n", encoding="utf-8")
    return chapter


def make_transcripts(root: Path, ids, text="REFERENCE TEXT NUMBER 0 WITH SEVERAL WORDS") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for uid in ids:
        (root / f"{uid}.txt").write_text(text + "\n", encoding="utf-8")
    return root


class TestCalibrationConstants(unittest.TestCase):
    """A threshold is only defensible if it sits above what was measured."""

    def test_campaign_baseline_is_the_librispeech_only_aggregate(self):
        # 9 word edits over 206 reference words. Pinned so a careless edit to
        # the constant has to confront the arithmetic it claims.
        self.assertAlmostEqual(lw.CAMPAIGN_CALIBRATION.baseline_wer, 9 / 206, places=12)
        self.assertAlmostEqual(lw.CAMPAIGN_BASELINE_WER, 9 / 206, places=12)

    def test_chapter_baseline_matches_its_measurement(self):
        # 21 word edits over 340 reference words, measured 2026-07-19.
        self.assertAlmostEqual(lw.CHAPTER_CALIBRATION.baseline_wer, 21 / 340, places=12)

    def test_every_threshold_clears_its_own_baseline_by_the_headroom_rule(self):
        for cal in lw.CALIBRATIONS:
            with self.subTest(cal.name):
                self.assertGreaterEqual(
                    cal.threshold,
                    cal.baseline_wer + lw.HEADROOM_PP / 100,
                    "rounding must never pull a threshold under baseline+headroom",
                )
                # Rounding up to 0.2pp can add at most that much on top.
                self.assertLess(cal.threshold, cal.baseline_wer + lw.HEADROOM_PP / 100 + 0.002)

    def test_headroom_buys_a_few_word_errors_not_a_free_pass(self):
        """Headroom must scale with corpus size and stay small in word terms."""
        for cal, ref_words in ((lw.CAMPAIGN_CALIBRATION, 206), (lw.CHAPTER_CALIBRATION, 340)):
            with self.subTest(cal.name):
                extra = (cal.threshold - cal.baseline_wer) * ref_words
                self.assertGreaterEqual(extra, 3.0)
                self.assertLess(extra, 6.5, "headroom grew past a few word errors")

    def test_the_two_calibrations_really_do_differ(self):
        """If they were close, per-slice calibration would be over-engineering.

        They are not: the 7 utterances the campaign excluded are harder, so
        the full chapter scores ~1.8pp worse. Sharing one threshold would
        make --all-in-chapter fail for reasons that are not regressions.
        """
        gap = lw.CHAPTER_CALIBRATION.baseline_wer - lw.CAMPAIGN_CALIBRATION.baseline_wer
        self.assertGreater(gap, 0.015)
        self.assertGreater(
            lw.CHAPTER_CALIBRATION.baseline_wer,
            lw.CAMPAIGN_CALIBRATION.threshold,
            "the chapter baseline exceeds the campaign threshold — exactly the "
            "false red that per-slice calibration prevents",
        )

    def test_derive_threshold_rounds_up_to_the_documented_grid(self):
        self.assertAlmostEqual(lw.derive_threshold(9 / 206), 0.06, places=10)
        self.assertAlmostEqual(lw.derive_threshold(21 / 340), 0.078, places=10)
        # Rounding is UP, never to nearest: a value a hair over a grid line
        # must not round back under baseline+headroom.
        self.assertGreaterEqual(lw.derive_threshold(0.0401), 0.0401 + lw.HEADROOM_PP / 100)

    def test_default_threshold_is_the_campaign_slice_threshold(self):
        self.assertAlmostEqual(lw.DEFAULT_THRESHOLD, lw.CAMPAIGN_CALIBRATION.threshold, places=12)

    def test_pinned_subset_is_the_eight_campaign_utterances(self):
        self.assertEqual(len(lw.CAMPAIGN_UTTERANCES), 8)
        self.assertEqual(len(set(lw.CAMPAIGN_UTTERANCES)), 8, "duplicate utterance id")
        for uid in lw.CAMPAIGN_UTTERANCES:
            self.assertRegex(uid, r"^1272-128104-\d{4}$")

    def test_chapter_set_is_the_15_utterance_superset(self):
        self.assertEqual(len(lw.CHAPTER_UTTERANCES), 15)
        self.assertTrue(set(lw.CAMPAIGN_UTTERANCES) < set(lw.CHAPTER_UTTERANCES))

    def test_corpus_sha256_is_a_full_length_digest(self):
        self.assertRegex(lw.DEV_CLEAN_SHA256, r"^[0-9a-f]{64}$")


class TestCorpusParsing(unittest.TestCase):
    def test_parse_trans_file_reads_id_and_text(self):
        with tempfile.TemporaryDirectory() as td:
            chapter = make_corpus(Path(td))
            parsed = lw.parse_trans_file(chapter / "1272-128104.trans.txt")
            self.assertEqual(len(parsed), 15)
            self.assertEqual(
                parsed["1272-128104-0000"], "REFERENCE TEXT NUMBER 0 WITH SEVERAL WORDS"
            )

    def test_parse_trans_file_rejects_a_line_without_a_transcript(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.trans.txt"
            p.write_text("1272-128104-0000\n", encoding="utf-8")
            with self.assertRaises(lw.ScoringError):
                lw.parse_trans_file(p)

    def test_utterance_flac_path_maps_into_speaker_chapter_dirs(self):
        got = lw.utterance_flac_path(Path("/c/dev-clean"), "1272-128104-0007")
        self.assertEqual(got, Path("/c/dev-clean/1272/128104/1272-128104-0007.flac"))

    def test_collect_references_returns_every_pinned_utterance(self):
        with tempfile.TemporaryDirectory() as td:
            make_corpus(Path(td))
            refs = lw.collect_references(
                Path(td) / "dev-clean", list(lw.CAMPAIGN_UTTERANCES)
            )
            self.assertEqual(sorted(refs), sorted(lw.CAMPAIGN_UTTERANCES))

    def test_collect_references_hard_fails_on_a_missing_utterance(self):
        """A short corpus must not silently become a smaller measurement."""
        with tempfile.TemporaryDirectory() as td:
            chapter = make_corpus(Path(td))
            # Drop one pinned utterance from the chapter's ground truth.
            kept = [ln for ln in TRANS_BODY.splitlines() if not ln.startswith("1272-128104-0005")]
            (chapter / "1272-128104.trans.txt").write_text("\n".join(kept) + "\n", encoding="utf-8")
            with self.assertRaises(lw.ScoringError) as ctx:
                lw.collect_references(Path(td) / "dev-clean", list(lw.CAMPAIGN_UTTERANCES))
            self.assertIn("1272-128104-0005", str(ctx.exception))

    def test_collect_references_hard_fails_when_the_chapter_is_absent(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(lw.ScoringError):
                lw.collect_references(Path(td) / "dev-clean", ["1272-128104-0000"])


class TestHypothesisLoading(unittest.TestCase):
    def test_reads_every_transcript(self):
        with tempfile.TemporaryDirectory() as td:
            d = make_transcripts(Path(td) / "tr", lw.CAMPAIGN_UTTERANCES)
            hyps = lw.read_hypotheses(d, list(lw.CAMPAIGN_UTTERANCES))
            self.assertEqual(len(hyps), 8)

    def test_missing_transcript_is_an_error_not_an_empty_hypothesis(self):
        """An empty hypothesis scores 100% deletions — that would read as a
        model collapse when the real fault is a failed transcription step."""
        with tempfile.TemporaryDirectory() as td:
            d = make_transcripts(Path(td) / "tr", lw.CAMPAIGN_UTTERANCES[:-1])
            with self.assertRaises(lw.ScoringError) as ctx:
                lw.read_hypotheses(d, list(lw.CAMPAIGN_UTTERANCES))
            self.assertIn(lw.CAMPAIGN_UTTERANCES[-1], str(ctx.exception))

    def test_whitespace_only_transcript_is_an_error(self):
        with tempfile.TemporaryDirectory() as td:
            d = make_transcripts(Path(td) / "tr", lw.CAMPAIGN_UTTERANCES)
            (d / f"{lw.CAMPAIGN_UTTERANCES[0]}.txt").write_text("   \n", encoding="utf-8")
            with self.assertRaises(lw.ScoringError):
                lw.read_hypotheses(d, list(lw.CAMPAIGN_UTTERANCES))


class TestVerdict(unittest.TestCase):
    def test_below_threshold_passes(self):
        ok, line = lw.verdict(0.0437, 0.06)
        self.assertTrue(ok)
        self.assertIn("PASS", line)

    def test_exactly_at_threshold_passes(self):
        ok, _ = lw.verdict(0.06, 0.06)
        self.assertTrue(ok, "the gate is <=, so the boundary is inside the budget")

    def test_above_threshold_fails(self):
        ok, line = lw.verdict(0.0601, 0.06)
        self.assertFalse(ok)
        self.assertIn("FAIL", line)

    def test_a_collapse_grade_regression_fails_loudly(self):
        # The defect class this leg exists to catch (campaign found WER 0.3-1.0).
        ok, _ = lw.verdict(1.0, lw.DEFAULT_THRESHOLD)
        self.assertFalse(ok)


class TestUtteranceSelection(unittest.TestCase):
    def _args(self, **kw):
        base = dict(utterances="", all_in_chapter=False, librispeech_root="")
        base.update(kw)
        return type("A", (), base)()

    def test_default_is_the_campaign_calibrated_set(self):
        ids, cal = lw.resolve_utterances(self._args())
        self.assertEqual(tuple(ids), lw.CAMPAIGN_UTTERANCES)
        self.assertEqual(cal, lw.CAMPAIGN_CALIBRATION)

    def test_custom_selection_has_no_calibration(self):
        """A different corpus slice must not inherit someone else's baseline."""
        ids, cal = lw.resolve_utterances(
            self._args(utterances="1272-128104-0001,1272-128104-0004")
        )
        self.assertEqual(ids, ["1272-128104-0001", "1272-128104-0004"])
        self.assertIsNone(cal)

    def test_restating_the_pinned_set_explicitly_still_matches_its_calibration(self):
        ids, cal = lw.resolve_utterances(self._args(utterances=",".join(lw.CAMPAIGN_UTTERANCES)))
        self.assertEqual(tuple(ids), lw.CAMPAIGN_UTTERANCES)
        self.assertEqual(cal, lw.CAMPAIGN_CALIBRATION)

    def test_all_in_chapter_selects_the_chapter_calibration(self):
        with tempfile.TemporaryDirectory() as td:
            make_corpus(Path(td))
            ids, cal = lw.resolve_utterances(self._args(all_in_chapter=True, librispeech_root=td))
            self.assertEqual(len(ids), 15)
            self.assertEqual(
                cal,
                lw.CHAPTER_CALIBRATION,
                "the full chapter has its own measured baseline, so it is gated too",
            )

    def test_a_reordered_set_still_matches_its_calibration(self):
        """Corpus WER is total-edits/total-words, so order cannot change it."""
        reordered = list(reversed(lw.CAMPAIGN_UTTERANCES))
        self.assertEqual(lw.calibration_for(reordered), lw.CAMPAIGN_CALIBRATION)

    def test_a_duplicated_id_does_not_match_a_calibration(self):
        dupey = list(lw.CAMPAIGN_UTTERANCES[:-1]) + [lw.CAMPAIGN_UTTERANCES[0]]
        self.assertEqual(len(dupey), len(lw.CAMPAIGN_UTTERANCES))
        self.assertIsNone(lw.calibration_for(dupey))

    def test_duplicate_utterances_are_rejected_not_silently_collapsed(self):
        with self.assertRaises(lw.ScoringError) as ctx:
            lw.resolve_utterances(self._args(utterances="1272-128104-0000,1272-128104-0000"))
        self.assertIn("duplicate", str(ctx.exception))


class TestReportRendering(unittest.TestCase):
    def _report(self, baseline=lw.CAMPAIGN_BASELINE_WER, threshold=0.06):
        return {
            "model": "whisper-base",
            "calibration": "campaign-8" if baseline is not None else None,
            "utterance_set": "campaign-8" if baseline is not None else "custom (2, uncalibrated)",
            "threshold": threshold,
            "baseline_wer": baseline,
            "baseline_provenance": "test",
            "verdict": "PASS: WER 4.3689% <= threshold 6.0000%",
            "passed": True,
            "measurement": {
                "wer": 0.043689320388349516,
                "cer": 0.018,
                "substitutions": 7,
                "deletions": 1,
                "insertions": 1,
                "edits": 9,
                "ref_words": 206,
                "n_utterances": 8,
                "per_utterance": [
                    {"utterance": "1272-128104-0000", "wer": 0.0, "cer": 0.0, "ref_words": 17}
                ],
            },
        }

    def test_summary_carries_verdict_threshold_and_baseline_delta(self):
        md = lw.render_summary(self._report())
        self.assertIn("PASS", md)
        self.assertIn("6.0000%", md)
        self.assertIn("baseline (`campaign-8`)", md)
        self.assertIn("delta vs baseline", md)
        self.assertIn("Advisory leg", md)

    def test_summary_says_no_calibration_for_an_uncalibrated_set(self):
        md = lw.render_summary(self._report(baseline=None, threshold=None))
        self.assertIn("no calibration for this utterance set", md)
        self.assertIn("not applied (uncalibrated slice)", md)
        self.assertNotIn("delta vs baseline", md)


class TestWorkflowSurface(unittest.TestCase):
    """Assertions on the YAML that Python-level tests cannot reach."""

    @classmethod
    def setUpClass(cls):
        cls.text = WORKFLOW.read_text(encoding="utf-8")

    def test_workflow_exists_and_is_named(self):
        self.assertIn("name: nightly-asr-wer", self.text)

    def test_cron_uses_the_documented_free_nightly_slot(self):
        """04:17 (il2cpp) and 04:47 (webgl) are taken; this leg claims 05:17.

        The weekly Monday ladder (04:00/05:00/05:30/06:00/06:30/07:00/07:30/
        08:00/08:30) is a separate series; 05:17 collides with none of it.
        """
        crons = re.findall(r'-\s+cron:\s*"([^"]+)"', self.text)
        self.assertEqual(crons, ["17 5 * * *"], f"unexpected cron set: {crons}")

    def test_cron_slot_is_free_across_the_whole_workflow_tree(self):
        """Guard the stagger for real rather than trusting the header comment."""
        ours = None
        taken = []
        for wf in sorted((REPO / ".github" / "workflows").glob("*.yml")):
            for cron in re.findall(r'-\s+cron:\s*"([^"]+)"', wf.read_text(encoding="utf-8")):
                minute, hour = cron.split()[0], cron.split()[1]
                if wf.name == WORKFLOW.name:
                    ours = (minute, hour, cron)
                else:
                    taken.append((wf.name, minute, hour, cron))
        self.assertIsNotNone(ours, "this workflow declares no cron")
        for name, minute, hour, cron in taken:
            self.assertFalse(
                minute == ours[0] and hour == ours[1],
                f"cron slot {ours[2]} collides with {name} ({cron})",
            )

    def _trigger_block(self) -> str:
        """The `on:` mapping only — the header prose discusses triggers by
        name, so a whole-file substring search would match documentation
        rather than configuration."""
        m = re.search(r"^on:\n(.*?)^\w", self.text, re.S | re.M)
        self.assertIsNotNone(m, "workflow has no `on:` block")
        return m.group(1)

    def test_is_advisory_not_a_required_check(self):
        # No pull_request trigger: the leg must never gate a merge.
        triggers = self._trigger_block()
        self.assertNotIn("pull_request", triggers)
        self.assertIn("workflow_dispatch:", triggers)
        self.assertIn("schedule:", triggers)
        self.assertIn("advisory", self.text.lower())

    def test_header_scopes_out_the_utmos_and_device_matrix_legs(self):
        head = self.text.split("name: nightly-asr-wer", 1)[0]
        self.assertIn("M5-15", head, "header must hand the LJSpeech/UTMOS leg to M5-15")
        self.assertRegex(head, r"(?i)ljspeech")
        self.assertRegex(head, r"(?i)tier[ -]?2")
        self.assertRegex(head, r"(?i)owner")

    def test_threshold_default_matches_the_scorer(self):
        m = re.search(r"WER_THRESHOLD:\s*\"?([0-9.]+)", self.text)
        self.assertIsNotNone(m, "workflow must declare WER_THRESHOLD")
        self.assertAlmostEqual(float(m.group(1)), lw.DEFAULT_THRESHOLD, places=6)

    def test_pins_the_corpus_checksum_the_scorer_recorded(self):
        self.assertIn(lw.DEV_CLEAN_SHA256, self.text)

    def test_workflow_url_uses_the_vars_fallback_seam(self):
        """WP X-10-T04/T05 graceful-fallback seam.

        The workflow env's DEV_CLEAN_URL must be a `${{ vars.<KEY> ||
        '<literal>' }}` expression so that a Vokra-owned mirror can be
        introduced by setting an org variable, without editing this
        workflow. The literal fallback must remain the OpenSLR canonical
        URL so behaviour is UNCHANGED when the org variable is unset.
        """
        # Anchor to the `env:` mapping, tolerate any quoting/whitespace.
        # env values are top-level under `env:`, so the anchor is
        # `\n  DEV_CLEAN_URL:` (two-space indent for job-level env).
        m = re.search(
            r"\n  DEV_CLEAN_URL:\s*"
            r"\$\{\{\s*vars\.(VOKRA_CORPUS_LIBRISPEECH_MIRROR_URL)\s*\|\|\s*"
            r"'([^']+)'\s*\}\}",
            self.text,
        )
        self.assertIsNotNone(
            m,
            "DEV_CLEAN_URL must be `${{ vars.VOKRA_CORPUS_LIBRISPEECH_MIRROR_URL || "
            "'<openslr-url>' }}` — the WP X-10-T04/T05 graceful-fallback seam",
        )
        self.assertEqual(
            m.group(2),
            "https://www.openslr.org/resources/12/dev-clean.tar.gz",
            "fallback URL must remain OpenSLR canonical so unset-var behaviour "
            "matches pre-seam behaviour bit-for-bit",
        )

    def test_workflow_fallback_pin_matches_calibration(self):
        """The literal SHA in the fallback must equal `lw.DEV_CLEAN_SHA256`.

        Prevents silent drift where someone edits the workflow env literal
        in isolation and forgets the scorer side (or vice versa). Same
        posture as `test_threshold_default_matches_the_scorer` for the
        WER_THRESHOLD constant.
        """
        m = re.search(
            r"\n  DEV_CLEAN_SHA256:\s*"
            r"\$\{\{\s*vars\.(VOKRA_CORPUS_LIBRISPEECH_MIRROR_SHA256)\s*\|\|\s*"
            r"'([0-9a-f]{64})'\s*\}\}",
            self.text,
        )
        self.assertIsNotNone(
            m,
            "DEV_CLEAN_SHA256 must be `${{ vars.VOKRA_CORPUS_LIBRISPEECH_MIRROR_SHA256 || "
            "'<64-hex-sha>' }}` — the WP X-10-T04/T05 graceful-fallback seam",
        )
        self.assertEqual(
            m.group(2),
            lw.DEV_CLEAN_SHA256,
            "workflow fallback SHA must equal lw.DEV_CLEAN_SHA256; both sides "
            "describe the same OpenSLR file and must not drift silently",
        )

    def test_cache_key_still_binds_to_the_effective_sha(self):
        """SHA-key change is the actual re-fetch mechanism (see P2 docstring).

        If someone accidentally rewrote the cache key to a static string,
        a corpus-swap tripwire (bump DEV_CLEAN_SHA256 to re-fetch on drift)
        would silently stop working. Anchor the key to `env.DEV_CLEAN_SHA256`
        so any refactor has to touch this test.
        """
        self.assertIn(
            "key: librispeech-dev-clean-${{ env.DEV_CLEAN_SHA256 }}",
            self.text,
            "corpus cache key must interpolate env.DEV_CLEAN_SHA256 so a pin bump "
            "actually forces a cache miss and re-fetch",
        )

    def test_guards_the_root_cargo_lock(self):
        """Zero-dep NFR-DS-02: same tripwire the other asset workflows carry."""
        self.assertIn("git diff --exit-code", self.text)
        self.assertIn("Cargo.lock", self.text)

    def test_no_dispatch_input_is_interpolated_into_a_run_body(self):
        """Script-injection guard.

        `threshold` is a free-text dispatch input. Interpolating `${{ ... }}`
        into a `run:` body splices its text into the shell; passing it through
        `env:` keeps it a value. Asserted structurally so a later edit cannot
        reintroduce the pattern.
        """
        # Body of each `run: |` block, up to the next same-indent YAML key.
        bodies = re.findall(r"^(\s+)run: \|\n((?:\1  .*\n|\n)*)", self.text, re.M)
        self.assertTrue(bodies, "no run: blocks found — did the workflow shape change?")
        offenders = [b for _indent, b in bodies if "${{" in b]
        self.assertEqual(
            offenders, [], "GitHub expressions must reach run: bodies via env:, not interpolation"
        )

    def test_extracted_subset_step_selects_only_the_pinned_utterances(self):
        """Execute the workflow's own extraction block against a fake tarball.

        This is the step that keeps the nightly download bounded; if it ever
        extracted the whole 337 MB corpus, or silently extracted nothing, the
        WER would be measured on the wrong thing.
        """
        block = self._extract_bash_block("subset-extract")
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            # Build a tar that mimics dev-clean's layout, with extra chapters
            # that the block must NOT pull out.
            stage = td_p / "stage" / "LibriSpeech" / "dev-clean"
            for spk, chap in (("1272", "128104"), ("9999", "999999")):
                d = stage / spk / chap
                d.mkdir(parents=True)
                (d / f"{spk}-{chap}.trans.txt").write_text("x\n", encoding="utf-8")
                for i in range(15):
                    (d / f"{spk}-{chap}-{i:04d}.flac").write_text("f\n", encoding="utf-8")
            tarball = td_p / "dev-clean.tar.gz"
            subprocess.run(
                ["tar", "czf", str(tarball), "-C", str(td_p / "stage"), "LibriSpeech"],
                check=True,
            )
            out = td_p / "corpus"
            out.mkdir()
            res = subprocess.run(
                ["bash", "-c", block],
                cwd=td_p,
                env={
                    "PATH": "/usr/bin:/bin:/usr/local/bin",
                    "TARBALL": str(tarball),
                    "CORPUS_DIR": str(out),
                    "SPEAKER": lw.CAMPAIGN_SPEAKER,
                    "CHAPTER": lw.CAMPAIGN_CHAPTER,
                },
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 0, f"subset step failed:\n{res.stderr}")
            got = out / "dev-clean" / lw.CAMPAIGN_SPEAKER / lw.CAMPAIGN_CHAPTER
            self.assertTrue(got.is_dir(), f"pinned chapter not extracted:\n{res.stdout}")
            self.assertTrue((got / f"1272-128104.trans.txt").is_file())
            self.assertEqual(len(list(got.glob("*.flac"))), 15)
            self.assertFalse(
                (out / "dev-clean" / "9999").exists(),
                "extraction must stay scoped to the pinned chapter",
            )

    def _extract_bash_block(self, marker: str) -> str:
        """Pull a `# <<<marker` … `# marker>>>` fenced run: block out of the YAML."""
        m = re.search(
            rf"#\s*<<<{re.escape(marker)}\n(.*?)\n\s*#\s*{re.escape(marker)}>>>",
            self.text,
            re.S,
        )
        self.assertIsNotNone(m, f"no bash block fenced as {marker} in {WORKFLOW.name}")
        body = m.group(1)
        # Strip the YAML block-scalar indentation.
        lines = [ln for ln in body.splitlines() if ln.strip()]
        indent = min(len(ln) - len(ln.lstrip()) for ln in lines)
        return "\n".join(ln[indent:] if len(ln) >= indent else ln for ln in body.splitlines())


if __name__ == "__main__":
    unittest.main(verbosity=2)
