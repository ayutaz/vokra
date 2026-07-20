#!/usr/bin/env python3
"""TDD / pre-verify oracle for the nightly-webgl workflow (X-06-T02).

`.github/workflows/nightly-webgl.yml` landed in main via M4 PR #8 (ff12104)
but has never run: it is `schedule` + `workflow_dispatch` only, with no
`pull_request:` trigger, so PR CI never exercised it. This oracle pins the
surface a Python-level test can reach so the first cron fire cannot red on a
formatting/structure defect that a text-assert would have caught:

  * the cron slot (04:47 UTC) is the one the ladder documents, and it is
    free across the whole workflow tree;
  * the two legs exist and have the right coupling — a license-free
    wasm-harness leg that always runs, and a license-GATED Unity WebGL build
    leg that SKIPS (never fabricates a pass) when secrets.UNITY_LICENSE is
    absent;
  * advisory posture: no `pull_request:` trigger, so the leg can never block
    a merge;
  * the zero-dep Cargo.lock tripwire is present in the build script the
    workflow invokes (the workflow header delegates the guard there).

stdlib-only (zero-dep NFR-DS-02), matching
tools/eval/test_librispeech_wer.py and tools/parity/test_parity_whisper_workflow.py:
this runs on any checkout with no third-party imports. It does NOT run the
Unity build (license-gated, owner) or the wasm-harness (emsdk, owner
pre-verify) — those legs are exercised by the workflow itself once the owner
dispatches it.

Run: python3 tools/parity/test_nightly_webgl_workflow.py
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "nightly-webgl.yml"
BUILD_SCRIPT = REPO / "scripts" / "build-unity-webgl-lib.sh"


class TestNightlyWebglSurface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = WORKFLOW.read_text(encoding="utf-8")

    def test_workflow_exists_and_is_named(self):
        self.assertIn("name: nightly-webgl", self.text)

    def test_cron_uses_the_documented_free_nightly_slot(self):
        """04:17 (il2cpp), 04:47 (this leg), 05:17 (asr-wer) is the nightly
        ladder. This leg claims 04:47 and nothing else."""
        crons = re.findall(r'-\s+cron:\s*"([^"]+)"', self.text)
        self.assertEqual(crons, ["47 4 * * *"], f"unexpected cron set: {crons}")

    def test_cron_slot_is_the_only_one_at_that_minute_hour_tree_wide(self):
        """Assert THIS leg's slot is free tree-wide, without re-walking the
        whole tree three times (the exhaustive uniqueness sweep is the single
        source in tools/eval/test_librispeech_wer.py). We only confirm no
        OTHER workflow shares 47/4."""
        ours = ("47", "4")
        collisions = []
        for wf in sorted((REPO / ".github" / "workflows").glob("*.yml")):
            if wf.name == WORKFLOW.name:
                continue
            for cron in re.findall(r'-\s+cron:\s*"([^"]+)"', wf.read_text(encoding="utf-8")):
                parts = cron.split()
                if (parts[0], parts[1]) == ours:
                    collisions.append((wf.name, cron))
        self.assertEqual(collisions, [], f"04:47 slot collides: {collisions}")

    def _trigger_block(self) -> str:
        m = re.search(r"^on:\n(.*?)^\w", self.text, re.S | re.M)
        self.assertIsNotNone(m, "workflow has no `on:` block")
        return m.group(1)

    def test_is_advisory_not_a_required_check(self):
        triggers = self._trigger_block()
        self.assertNotIn(
            "pull_request",
            triggers,
            "a pull_request trigger would make this WebGL leg a merge gate — "
            "Unity license / game-ci / browser flakiness must never block PRs",
        )
        self.assertIn("workflow_dispatch:", triggers)
        self.assertIn("schedule:", triggers)
        # The header states the not-required / promotion posture explicitly.
        self.assertRegex(self.text, r"(?i)not\s+a\s+required")

    def test_two_legs_exist(self):
        """The dual-leg structure (ADR M4-02 §8): a license-free harness leg
        and a license-gated Unity build leg, fed by a preflight license check."""
        self.assertIn("wasm-harness:", self.text)
        self.assertIn("unity-webgl-build:", self.text)
        self.assertIn("preflight:", self.text)

    def test_wasm_harness_leg_needs_no_license(self):
        """The harness leg must run unconditionally — it is the coverage that
        keeps the workflow useful while secrets.UNITY_LICENSE is unprovisioned.
        It carries no `needs: preflight` / license `if:` gate."""
        block = self._job_block("wasm-harness")
        self.assertNotIn("has_license", block, "the harness leg must not be license-gated")
        self.assertIn(
            "build-unity-webgl-lib.sh",
            block,
            "the harness leg must invoke the pinned-emcc build/verify script",
        )

    def test_unity_build_leg_is_license_gated_and_skips_not_passes(self):
        """The fabricated-pass guard. When no license is present the Unity
        build leg must be SKIPPED (its `if:` is false), never run-and-green.
        A skipped job does not count as a passing build."""
        # The job carries an `if:` predicated on the preflight license output.
        job = self._job_block("unity-webgl-build")
        self.assertIn("needs: preflight", job)
        self.assertRegex(
            job,
            r"if:\s*needs\.preflight\.outputs\.has_license\s*==\s*'true'",
            "the Unity build leg must be gated on the license being present, so it "
            "SKIPS (never fabricates a pass) when the secret is absent",
        )

    def test_missing_license_is_a_warning_annotation_not_a_silent_pass(self):
        """FR-EX-08 / fabricated-pass: absent license emits a ::warning:: that
        names the skip, not a green ✓ with no signal."""
        block = self._job_block("preflight")
        self.assertIn("has_license=false", block)
        self.assertRegex(
            block,
            r"::warning[^\n]*UNITY_LICENSE",
            "an absent license must surface as a ::warning:: annotation",
        )

    def test_cargo_lock_tripwire_lives_in_the_invoked_build_script(self):
        """Zero-dep NFR-DS-02. This workflow delegates the guard to the build
        script (header: 'the build script has an internal tripwire'); assert
        the script it invokes actually carries it, rather than trusting the
        comment."""
        self.assertIn(
            "build-unity-webgl-lib.sh",
            self.text,
            "workflow must invoke the build script that owns the tripwire",
        )
        self.assertTrue(BUILD_SCRIPT.is_file(), f"missing {BUILD_SCRIPT}")
        script = BUILD_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("Cargo.lock", script)
        self.assertRegex(
            script,
            r"(?i)cargo\.lock\s+changed|NFR-DS-02 tripwire",
            "build script must fail if the root Cargo.lock changed",
        )

    def test_no_dispatch_input_is_interpolated_into_a_run_body(self):
        """Script-injection guard, mirroring the asr-wer oracle. `dry_run` is a
        free-text dispatch input; it must reach a `run:` body via `env:` or a
        job-level `if:`, never spliced in as `${{ }}` inside the shell."""
        bodies = re.findall(r"^(\s+)run: \|\n((?:\1  .*\n|\n)*)", self.text, re.M)
        self.assertTrue(bodies, "no run: blocks found — did the workflow shape change?")
        offenders = [b for _indent, b in bodies if "${{" in b]
        self.assertEqual(
            offenders,
            [],
            "GitHub expressions must reach run: bodies via env:, not interpolation",
        )

    def _job_block(self, name: str) -> str:
        """Return the text of a single top-level job (`  <name>:` up to the
        next top-level job key)."""
        m = re.search(rf"^  {re.escape(name)}:\n(.*?)(?=^  \w|\Z)", self.text, re.S | re.M)
        self.assertIsNotNone(m, f"no job named {name}")
        return m.group(1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
