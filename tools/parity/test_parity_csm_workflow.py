#!/usr/bin/env python3
"""TDD / pre-verify oracle for the parity-csm-real workflow (X-06-T03).

`.github/workflows/parity-csm-real.yml` landed in main via M4 PR #8 (ff12104)
but has never run: it is `workflow_dispatch` + weekly `schedule` only, no
`pull_request:` trigger. This oracle pins the surface so the first weekly
cron fire cannot red on a structure defect:

  * dispatch + weekly Monday cron 06:30 UTC, matching the parity ladder;
  * the committed SYNTHETIC leg always runs (format/manifest pin), while the
    REAL leg is gated on VOKRA_CSM_PARITY_DIR and, when that is empty, the
    step prints an explicit "clean skip, not a pass" notice rather than a
    fabricated green — the gated repos (sesame/csm-1b + meta-llama tokenizer)
    cannot be fetched anonymously, so the real leg waits for the T29 owner
    dump;
  * advisory posture: no `pull_request:` trigger, never a merge gate.

stdlib-only (zero-dep NFR-DS-02). The always-on synthetic leg itself is a
Rust test (`cargo test -p vokra-models --test parity_csm`) — this oracle
asserts the YAML wires it, and the leg's authoring notes record the local
green run; this file does not shell out to cargo.

Run: python3 tools/parity/test_parity_csm_workflow.py
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "parity-csm-real.yml"
SYNTHETIC_TEST = REPO / "crates" / "vokra-models" / "tests" / "parity_csm.rs"


class TestParityCsmSurface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = WORKFLOW.read_text(encoding="utf-8")

    def test_workflow_exists_and_is_named(self):
        self.assertIn("name: parity-csm-real", self.text)

    def test_weekly_cron_is_the_documented_monday_slot(self):
        """Weekly Monday ladder: Kokoro 04:00, Whisper 05:00, Vulkan 05:30,
        CUDA-RTF 06:00, CSM 06:30, ... This leg claims 06:30 and nothing else."""
        crons = re.findall(r'-\s+cron:\s*"?\'?([0-9*/ ]+?)"?\'?$', self.text, re.M)
        crons = [c.strip() for c in crons if c.strip()]
        self.assertEqual(crons, ["30 6 * * 1"], f"unexpected cron set: {crons}")

    def test_cron_slot_is_free_tree_wide_at_that_minute_hour_day(self):
        """CSM shares the weekly runner window with CUDA-RTF (06:00) and Moshi
        (07:30); assert no OTHER workflow lands on the same minute+hour+dow.
        (The exhaustive daily sweep is the single source in the asr-wer
        oracle; here we only guard this leg's own slot.)"""
        ours = ("30", "6", "1")  # minute, hour, day-of-week
        collisions = []
        for wf in sorted((REPO / ".github" / "workflows").glob("*.yml")):
            if wf.name == WORKFLOW.name:
                continue
            for cron in re.findall(r'-\s+cron:\s*"?\'?([0-9*/ ]+?)"?\'?$', wf.read_text(encoding="utf-8"), re.M):
                parts = cron.split()
                if len(parts) == 5 and (parts[0], parts[1], parts[4]) == ours:
                    collisions.append((wf.name, cron.strip()))
        self.assertEqual(collisions, [], f"06:30 Mon slot collides: {collisions}")

    def _trigger_block(self) -> str:
        m = re.search(r"^on:\n(.*?)^\w", self.text, re.S | re.M)
        self.assertIsNotNone(m, "workflow has no `on:` block")
        return m.group(1)

    def test_is_advisory_not_a_required_check(self):
        triggers = self._trigger_block()
        self.assertNotIn(
            "pull_request",
            triggers,
            "a pull_request trigger would make the gated CSM leg a merge gate — "
            "HF flakiness / gated downloads must never block PRs",
        )
        self.assertIn("workflow_dispatch:", triggers)
        self.assertIn("schedule:", triggers)
        # Header states the not-required posture.
        self.assertRegex(self.text, r"(?i)not\s+a\s+required")

    def test_synthetic_leg_always_runs(self):
        """The committed synthetic fixture leg is unconditional — it is the
        format/manifest pin that stays green regardless of the gated repos."""
        self.assertIn("cargo test -p vokra-models --test parity_csm", self.text)
        self.assertTrue(SYNTHETIC_TEST.is_file(), f"missing {SYNTHETIC_TEST}")
        rs = SYNTHETIC_TEST.read_text(encoding="utf-8")
        self.assertIn(
            "fn synthetic_fixture_manifest_roundtrip",
            rs,
            "the always-on synthetic leg the YAML runs must exist in parity_csm.rs",
        )

    def test_real_leg_is_env_gated_on_the_parity_dir(self):
        self.assertIn("VOKRA_CSM_PARITY_DIR", self.text)
        # The gate: when the dir is empty the env var is UNSET so the Rust
        # test's `env::var_os` returns None and the real leg cleanly skips.
        self.assertRegex(
            self.text,
            r"if\s*\[\s*-z\s*\"?\$\{?VOKRA_CSM_PARITY_DIR",
            "the workflow must unset/skip the real leg when the parity dir is empty",
        )

    def test_absent_reference_is_a_clean_skip_not_a_pass(self):
        """Fabricated-pass guard. The notice printed when the real reference is
        absent must SAY it is a clean skip, not silently read as a pass."""
        self.assertRegex(
            self.text,
            r"(?i)clean skip,?\s*not a pass",
            "absent CSM reference must print 'clean skip, not a pass'",
        )
        # And the Rust test itself fails loudly (never silent-passes) once the
        # dir IS set but the weight binding is missing — the honest-failure end
        # of the same contract.
        rs = SYNTHETIC_TEST.read_text(encoding="utf-8")
        self.assertIn("VOKRA_CSM_PARITY_DIR", rs)

    def test_no_dispatch_input_is_interpolated_into_a_run_body(self):
        """Script-injection guard. `parity_dir` is a free-text dispatch input;
        it must reach the shell via `env:`, not `${{ }}` interpolation.

        The Summary step is exempt: it interpolates `inputs.parity_dir` only
        inside a `[ -z ... ]` test and an echo, and the input is a runner path
        the owner controls (not third-party). The gate is on the STEP that
        runs cargo — that one takes the value through env: VOKRA_CSM_PARITY_DIR."""
        # The cargo-running step must not interpolate the raw input.
        m = re.search(
            r"name: Synthetic fixture leg.*?run: \|\n(.*?)(?=\n      - name:|\Z)",
            self.text,
            re.S,
        )
        self.assertIsNotNone(m, "no synthetic-leg run block found")
        self.assertNotIn(
            "${{",
            m.group(1),
            "the cargo step must take the parity dir via env:, not interpolation",
        )
        # And it does take it via env:.
        self.assertRegex(self.text, r"VOKRA_CSM_PARITY_DIR:\s*\$\{\{\s*inputs\.parity_dir")


if __name__ == "__main__":
    unittest.main(verbosity=2)
