#!/usr/bin/env python3
"""Oracle for the Tier 2 device nightly workflow (X-06-T22).

milestones.md §10 X-06 splits Tier 1/2 device RTF into "CI job definition =
CC / self-hosted device lab = owner". This is the CC job definition:
`.github/workflows/nightly-tier2-device.yml`. The mechanical guarantee this
oracle pins is the one the WP hinges on — that with NO device runner in the
fleet, the workflow does not record a threshold as "met":

  * a runner-absent run CLEAN-SKIPS with a `## Skipped: no self-hosted
    runner` step summary — it does NOT fabricate a pass;
  * the measure leg is `if: false`-guarded so a weekly cron cannot red with
    "no matching runner" before a runner exists (the gpu-cuda-rtf pattern);
  * the NFR-PF-09/10 thresholds are encoded verbatim (Whisper base < 0.5 on
    Pi 5, < 1.0 on Pi 4, Whisper tiny < 2.0, Silero VAD + Kokoro real-time),
    so a careless edit that loosens a threshold is caught;
  * advisory posture (no `pull_request:` trigger) and the Cargo.lock tripwire.

stdlib-only (zero-dep NFR-DS-02). Run:
python3 tools/parity/test_nightly_tier2_workflow.py
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "nightly-tier2-device.yml"


class TestNightlyTier2Surface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = WORKFLOW.read_text(encoding="utf-8")

    def test_workflow_exists_and_is_named(self):
        self.assertTrue(WORKFLOW.is_file(), f"missing {WORKFLOW}")
        self.assertIn("name: nightly-tier2-device", self.text)

    def test_cron_slot_is_free_tree_wide(self):
        crons = re.findall(r'-\s+cron:\s*"([^"]+)"', self.text)
        self.assertEqual(len(crons), 1, f"expected one cron, got {crons}")
        m, h = crons[0].split()[0], crons[0].split()[1]
        collisions = []
        for wf in sorted((REPO / ".github" / "workflows").glob("*.yml")):
            if wf.name == WORKFLOW.name:
                continue
            for cron in re.findall(r'-\s+cron:\s*"([^"]+)"', wf.read_text(encoding="utf-8")):
                if (cron.split()[0], cron.split()[1]) == (m, h):
                    collisions.append((wf.name, cron))
        self.assertEqual(collisions, [], f"slot {crons[0]} collides: {collisions}")

    def _trigger_block(self) -> str:
        mm = re.search(r"^on:\n(.*?)^\w", self.text, re.S | re.M)
        self.assertIsNotNone(mm, "workflow has no `on:` block")
        return mm.group(1)

    def test_is_advisory_not_a_required_check(self):
        triggers = self._trigger_block()
        self.assertNotIn(
            "pull_request",
            triggers,
            "device RTF must never gate a PR — no device runner is present at PR time",
        )
        self.assertIn("workflow_dispatch:", triggers)
        self.assertIn("schedule:", triggers)
        self.assertRegex(self.text, r"(?i)not\s+(a\s+)?required")

    def test_has_a_label_presence_gate_job(self):
        """A gate job that runs on ubuntu-latest and decides skip-vs-proceed
        from a repo variable, so the workflow always resolves even with no
        device in the fleet."""
        self.assertRegex(self.text, r"vars\.VOKRA_TIER2_SELF_HOSTED")
        self.assertRegex(
            self.text,
            r"runs-on:\s*ubuntu-latest",
            "the gate job must run on a GitHub-hosted runner so it always executes",
        )

    def test_runner_absent_is_a_clean_skip_not_a_fabricated_pass(self):
        """THE guard: no runner → a step summary that SAYS it skipped, never a
        silent green that reads as 'threshold met'."""
        self.assertIn("## Skipped: no self-hosted runner", self.text)
        # And the skip path must not print any PASS/met verdict.
        self.assertNotRegex(
            self.text,
            r"(?i)threshold\s+met.*no self-hosted|no self-hosted.*threshold\s+met",
            "the skip path must not claim a threshold was met",
        )

    def test_measure_leg_is_if_false_guarded_until_a_runner_exists(self):
        """Without the `if: false`, a weekly cron reds with 'no matching
        runner' the moment a self-hosted label is referenced but absent —
        exactly the gpu-cuda-rtf.yml red line."""
        self.assertRegex(
            self.text,
            r"if:\s*false\b",
            "the measure leg must be `if: false`-guarded (mirrors gpu-cuda-rtf.yml)",
        )

    def test_measure_leg_targets_a_self_hosted_device_runner(self):
        self.assertRegex(self.text, r"runs-on:\s*\[\s*self-hosted")

    def test_nfr_pf_09_10_thresholds_are_encoded_verbatim(self):
        """A loosened threshold is a silent quality regression. Pin the exact
        NFR-PF-09/10 numbers + the two real-time rows."""
        # Whisper base < 0.5 (Pi 5, NFR-PF-09) and < 1.0 (Pi 4, NFR-PF-10).
        self.assertRegex(self.text, r"0\.5\b", "missing Whisper base < 0.5 (Pi 5)")
        self.assertRegex(self.text, r"1\.0\b", "missing Whisper base < 1.0 (Pi 4)")
        # Whisper tiny < 2.0 (Pi 3B, NFR-PF-10).
        self.assertRegex(self.text, r"2\.0\b", "missing Whisper tiny < 2.0 (Pi 3B)")
        # The two real-time rows.
        self.assertRegex(self.text, r"(?i)silero.*real|real.*silero")
        self.assertRegex(self.text, r"(?i)kokoro.*real|real.*kokoro")
        # The requirement IDs are named so the mapping is auditable.
        self.assertIn("NFR-PF-09", self.text)
        self.assertIn("NFR-PF-10", self.text)

    def test_each_device_row_names_its_pi_model(self):
        for pi in ("Pi 5", "Pi 4", "Pi 3B"):
            self.assertIn(pi, self.text, f"threshold table must name {pi}")

    def test_guards_the_root_cargo_lock(self):
        self.assertIn("git diff --exit-code", self.text)
        self.assertIn("Cargo.lock", self.text)

    def test_no_dispatch_input_is_interpolated_into_a_run_body(self):
        bodies = re.findall(r"^(\s+)run: \|\n((?:\1  .*\n|\n)*)", self.text, re.M)
        offenders = [b for _indent, b in bodies if "${{" in b]
        self.assertEqual(
            offenders, [], "GitHub expressions must reach run: bodies via env:, not interpolation"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
