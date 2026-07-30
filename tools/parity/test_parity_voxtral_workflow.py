#!/usr/bin/env python3
"""TDD / pre-verify oracle for `.github/workflows/parity-voxtral-real.yml`
(FQ-07, 2026-07-31).

The parity-voxtral-real workflow is the fifth in the "published-model parity"
family (kokoro / whisper / csm / moshi / voxtral). Voxtral ships in two BF16
variants at `mistralai/Voxtral-{Mini-3B,Small-24B}-2507` (both Apache-2.0):

  * `mini-3b-2507`  — 8.7 GB, already published as vokra/voxtral-mini-3b-2507
                      on 2026-07-23. Runs on ubuntu-latest via the MappedLazy
                      port that landed the same day.
  * `small-24b-2507` — 48 GB (11 shards). REQUIRES vast.ai (see
                      docs/handoff/vast-ai-large-model-publish.md §2).
                      Deferred with `if: false` posture: the matrix carries
                      the entry with `enabled: false` so the deferral is
                      pinned and a future flip is one-line.

This oracle pins the workflow's on-disk shape so a YAML / matrix / gate
drift is caught before a schedule-only or gate-variable-locked leg silently
regresses. stdlib-only (zero-dep NFR-DS-02) — same discipline as
`test_parity_whisper_workflow.py` / `test_parity_csm_workflow.py`.

What is checked:

  * Trigger surface: workflow_dispatch (with `run_dumper` + `include_small_24b`
    inputs), schedule (cron '15 5 * * 1'), pull_request (paths filter).
  * Cron slot: Monday 05:15 UTC is free tree-wide (does not collide with
    any other workflow at the same minute+hour+day-of-week).
  * Matrix: mini-3b-2507 entry has `enabled: true`; small-24b-2507 entry
    has `enabled: false` (the `if: false` posture the FQ-07 gap spec pins).
  * Gate variables: `VOKRA_VOXTRAL_ENABLE` for conversion,
    `VOKRA_VOXTRAL_HARNESS_READY` for the dumper leg.
  * Env vars: `VOKRA_VOXTRAL_GGUF`, `VOKRA_VOXTRAL_REF_DIR`,
    `VOKRA_VOXTRAL_BOS` are all set in the parity step (they gate
    parity_voxtral.rs / voxtral_real_gguf.rs / voxtral_transcription_prompt.rs).
  * PR paths filter covers the voxtral-adjacent code (crates + tests + tools).
  * Fabricated-pass guards: `include_small_24b=true` on ubuntu-latest
    hard-errors (never silently succeeds); missing gate variables print
    ::notice:: with "clean skip, not a pass".
  * Advisory posture: NOT a required check (HF flakiness must not block
    PRs); pull_request trigger present but only for narrow paths.
  * Dumper delegation: voxtral_dump_reference.py --self-test succeeds
    (offline, network-free).
  * Parity test file (`crates/vokra-models/tests/parity_voxtral.rs`) exists
    and uses the env vars this workflow sets.

Run: python3 tools/parity/test_parity_voxtral_workflow.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "parity-voxtral-real.yml"
SHIM = REPO / "tools" / "parity" / "voxtral_dump_reference.py"
DELEGATED_DUMPER = REPO / "tools" / "parity" / "dump_voxtral_reference.py"
PARITY_TEST = REPO / "crates" / "vokra-models" / "tests" / "parity_voxtral.rs"
REAL_GGUF_TEST = REPO / "crates" / "vokra-models" / "tests" / "voxtral_real_gguf.rs"
PROMPT_TEST = REPO / "crates" / "vokra-models" / "tests" / "voxtral_transcription_prompt.rs"


def _crons_in(text: str) -> list[str]:
    """All cron expressions in a workflow, regardless of quoting style."""
    out = []
    for m in re.finditer(
        r"^\s*-\s+cron:\s*(?:\"([^\"]+)\"|'([^']+)'|([^\s#]+))\s*(?:#.*)?$",
        text,
        re.MULTILINE,
    ):
        expr = m.group(1) or m.group(2) or m.group(3)
        if expr:
            out.append(expr.strip())
    return out


class WorkflowExists(unittest.TestCase):
    def test_workflow_file_exists(self):
        self.assertTrue(WORKFLOW.is_file(), f"missing {WORKFLOW}")

    def test_workflow_is_valid_yaml(self):
        # The `python-parity-oracles` job runs stdlib-only, so we do not
        # import PyYAML here. Use a subprocess call to `python -c` with
        # `yaml.safe_load` guarded by ImportError so this test cleanly
        # skips when PyYAML is absent (the fix_verification_command has
        # its own explicit YAML load that runs when PyYAML is present).
        try:
            import yaml  # type: ignore  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML not installed — the verify command runs "
                          "its own yaml.safe_load; oracle stays stdlib-only")
        import yaml  # type: ignore
        yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


class WorkflowName(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = WORKFLOW.read_text(encoding="utf-8")

    def test_workflow_is_named(self):
        self.assertIn("name: parity-voxtral-real", self.text)


class Triggers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = WORKFLOW.read_text(encoding="utf-8")

    def _trigger_block(self) -> str:
        m = re.search(r"^on:\n(.*?)^[A-Za-z]", self.text, re.S | re.M)
        self.assertIsNotNone(m, "workflow has no `on:` block")
        return m.group(1)

    def test_workflow_dispatch_present(self):
        self.assertIn("workflow_dispatch:", self._trigger_block())

    def test_workflow_dispatch_has_run_dumper_input(self):
        # A dispatch-only opt-in for the Phase B leg. Default MUST be
        # "false" — running the dumper without owner consent would blow
        # the multi-GB HF envelope on every dispatch.
        self.assertRegex(self.text, r"run_dumper:\s*\n(?:.*\n)*?\s+default:\s+\"false\"")

    def test_workflow_dispatch_has_include_small_24b_input(self):
        # The gap spec pins this input by name so a large-runner
        # enablement is a clean one-line change (input + matrix flip).
        self.assertRegex(self.text, r"include_small_24b:\s*\n(?:.*\n)*?\s+default:\s+\"false\"")

    def test_schedule_present_with_monday_0515_utc_cron(self):
        # Monday 05:15 UTC — the 30-min pocket between whisper (05:00) and
        # vulkan (05:30) in the weekly parity ladder.
        crons = _crons_in(self.text)
        self.assertEqual(
            crons,
            ["15 5 * * 1"],
            f"unexpected cron set: {crons} (expected exactly Monday 05:15 UTC)",
        )

    def test_pull_request_paths_filter_present(self):
        # PR runs only when voxtral-adjacent code changes.
        self.assertIn("pull_request:", self._trigger_block())
        self.assertRegex(self.text, r"paths:")

    def test_pr_paths_include_voxtral_crates(self):
        # Every voxtral-touching file must reach this workflow.
        for required in (
            ".github/workflows/parity-voxtral-real.yml",
            "crates/vokra-convert/src/models/voxtral.rs",
            "crates/vokra-models/src/voxtral/**",
            "crates/vokra-models/tests/parity_voxtral.rs",
            "tools/parity/voxtral_dump_reference.py",
        ):
            self.assertIn(required, self.text, f"PR paths filter missing {required}")


class CronSlotIsFreeTreewide(unittest.TestCase):
    """Assert nothing else runs on the same minute+hour+day-of-week."""

    def test_no_other_workflow_shares_the_0515_monday_slot(self):
        ours = ("15", "5", "1")  # minute, hour, day-of-week
        collisions = []
        for wf in sorted((REPO / ".github" / "workflows").glob("*.yml")):
            if wf.name == WORKFLOW.name:
                continue
            for cron in _crons_in(wf.read_text(encoding="utf-8")):
                parts = cron.split()
                if len(parts) == 5 and (parts[0], parts[1], parts[4]) == ours:
                    collisions.append((wf.name, cron))
        self.assertEqual(
            collisions,
            [],
            f"Monday 05:15 UTC slot collides with: {collisions}",
        )


class MatrixShape(unittest.TestCase):
    """Pin the mini-3b (enabled) + small-24b (deferred) matrix layout."""

    @classmethod
    def setUpClass(cls):
        cls.text = WORKFLOW.read_text(encoding="utf-8")
        # Extract the matrix JSON that the setup step writes.
        # Format (single-line to satisfy GITHUB_OUTPUT rules):
        #   matrix+='{"variant":"mini-3b-2507","enabled":true,...}'
        # We collect every include entry and parse each.
        entries = re.findall(
            r'matrix\+=\'({[^\']+})[,\']', cls.text
        )
        if not entries:
            # Fallback: try single JSON block if it was inlined that way.
            m = re.search(r"matrix='(\{[^']+\})'", cls.text)
            if m:
                blob = json.loads(m.group(1))
                cls.include = blob.get("include", [])
                return
        cls.include = []
        for raw in entries:
            try:
                cls.include.append(json.loads(raw))
            except json.JSONDecodeError as e:
                raise AssertionError(
                    f"matrix entry failed to parse as JSON: {raw!r} ({e})"
                )

    def test_matrix_carries_exactly_two_variants(self):
        variants = sorted(e.get("variant", "") for e in self.include)
        self.assertEqual(
            variants,
            ["mini-3b-2507", "small-24b-2507"],
            f"unexpected variant set: {variants}",
        )

    def test_mini_3b_entry_is_enabled(self):
        mini = [e for e in self.include if e.get("variant") == "mini-3b-2507"]
        self.assertEqual(len(mini), 1, "mini-3b-2507 entry missing or duplicated")
        entry = mini[0]
        self.assertEqual(
            entry.get("enabled"),
            True,
            "mini-3b-2507 must be enabled (runs on ubuntu-latest)",
        )
        self.assertEqual(entry.get("hf_repo"), "mistralai/Voxtral-Mini-3B-2507")
        self.assertEqual(entry.get("cli_model"), "voxtral")
        self.assertEqual(entry.get("license"), "apache-2.0")
        self.assertEqual(entry.get("env_gguf"), "VOKRA_VOXTRAL_GGUF")

    def test_small_24b_entry_is_deferred_with_if_false_posture(self):
        # The `if: false` posture the FQ-07 gap spec pins: the entry is
        # in the matrix (so the oracle can pin the deferral) but
        # `enabled: false` gates every step from running on ubuntu-latest.
        small = [e for e in self.include if e.get("variant") == "small-24b-2507"]
        self.assertEqual(len(small), 1, "small-24b-2507 entry missing or duplicated")
        entry = small[0]
        self.assertEqual(
            entry.get("enabled"),
            False,
            "small-24b-2507 must be `enabled: false` (~48 GB requires vast.ai)",
        )
        self.assertEqual(entry.get("hf_repo"), "mistralai/Voxtral-Small-24B-2507")

    def test_workflow_header_points_at_vast_ai_runbook(self):
        # The gap spec pins the runbook cross-reference. Owners reading
        # the workflow header must see where to go when they want to run
        # the deferred variant.
        self.assertIn(
            "docs/handoff/vast-ai-large-model-publish.md",
            self.text,
            "workflow must cross-reference the vast.ai runbook",
        )

    def test_steps_gate_on_matrix_enabled(self):
        # Every substantial step must be guarded by `if: matrix.enabled`
        # so the small-24b-2507 leg is a true `if: false` skip, not a
        # partial run that spuriously succeeds.
        # We check that the phrase appears many times (each real step).
        gates = self.text.count("if: matrix.enabled")
        self.assertGreaterEqual(
            gates,
            5,
            f"expected many `if: matrix.enabled` gates, saw only {gates} — "
            f"a partial gate leaves fabricated-pass windows",
        )


class GateVariables(unittest.TestCase):
    """Fabricated-pass 禁止: gate vars gate the appropriate legs."""

    @classmethod
    def setUpClass(cls):
        cls.text = WORKFLOW.read_text(encoding="utf-8")

    def test_conversion_gated_on_vokra_voxtral_enable(self):
        # cron/schedule path requires VOKRA_VOXTRAL_ENABLE=1.
        self.assertRegex(
            self.text,
            r"VOKRA_VOXTRAL_ENABLE",
            "cron leg must be gated on VOKRA_VOXTRAL_ENABLE (see DFN3/DeBERTa "
            "precedents)",
        )

    def test_dumper_gated_on_harness_ready(self):
        self.assertRegex(
            self.text,
            r"VOKRA_VOXTRAL_HARNESS_READY",
            "dumper leg must be gated on VOKRA_VOXTRAL_HARNESS_READY",
        )

    def test_clean_skip_not_a_pass_notices(self):
        # When gates are unset, the setup job emits ::notice::s that say
        # so — never a silent pass. Assert the phrase appears.
        self.assertIn(
            "clean skip, not a pass",
            self.text.lower().replace("not a pass", "not a pass"),
        )

    def test_include_small_24b_true_hard_errors(self):
        # The oracle-pinned fabricated-pass guard: forcing small-24b on
        # ubuntu-latest is a hard error, never a partial success.
        self.assertRegex(
            self.text,
            r"::error[^\n]*small.24b[^\n]*vast\.ai",
            "include_small_24b=true must hard-error and point at vast.ai",
        )


class ParityEnvVars(unittest.TestCase):
    """The three env vars parity_voxtral.rs / voxtral_real_gguf.rs /
    voxtral_transcription_prompt.rs test files consume.

    The gap spec explicitly names all three (VOKRA_VOXTRAL_GGUF /
    VOKRA_VOXTRAL_REF_DIR / VOKRA_VOXTRAL_BOS) — pin the workflow sets
    each so the harness sees them.
    """

    @classmethod
    def setUpClass(cls):
        cls.text = WORKFLOW.read_text(encoding="utf-8")

    def test_vokra_voxtral_gguf_set(self):
        self.assertIn("VOKRA_VOXTRAL_GGUF", self.text)

    def test_vokra_voxtral_ref_dir_set(self):
        self.assertIn("VOKRA_VOXTRAL_REF_DIR", self.text)

    def test_vokra_voxtral_bos_set(self):
        self.assertIn("VOKRA_VOXTRAL_BOS", self.text)

    def test_env_vars_are_scoped_to_the_parity_step(self):
        # The parity harness step MUST pass all three via `env:` (never
        # inline into the shell body — same script-injection guard the
        # CSM oracle applies to parity_dir).
        m = re.search(
            r"name: Run parity harness.*?env:(.*?)run:",
            self.text,
            re.S,
        )
        self.assertIsNotNone(m, "no parity harness step with env: block")
        env_block = m.group(1)
        for var in ("VOKRA_VOXTRAL_GGUF", "VOKRA_VOXTRAL_REF_DIR",
                    "VOKRA_VOXTRAL_BOS"):
            self.assertIn(var, env_block, f"parity step must set {var}")


class AdvisoryPosture(unittest.TestCase):
    """The workflow must not become a required check silently."""

    @classmethod
    def setUpClass(cls):
        cls.text = WORKFLOW.read_text(encoding="utf-8")

    def test_workflow_header_declares_not_required(self):
        # Same posture every parity-*-real workflow declares.
        self.assertRegex(
            self.text,
            r"(?i)not\s+(a\s+)?required\s+check",
            "header must state the not-required posture",
        )

    def test_zero_dep_tripwire_present(self):
        # Root Cargo.lock must be re-checked at the end of the parity leg.
        self.assertIn("git diff --exit-code Cargo.lock", self.text)


class ShimDelegation(unittest.TestCase):
    """Voxtral_dump_reference.py is a shim that delegates to the existing
    dump_voxtral_reference.py. Verify the shim exists, has --self-test,
    and its self-test passes offline."""

    def test_shim_exists(self):
        self.assertTrue(SHIM.is_file(), f"missing shim {SHIM}")

    def test_delegated_dumper_exists(self):
        self.assertTrue(
            DELEGATED_DUMPER.is_file(),
            f"missing delegated dumper {DELEGATED_DUMPER} — shim would fail",
        )

    def test_shim_self_test_succeeds_offline(self):
        # Runs the shim's own --self-test. Zero-dep, network-free.
        proc = subprocess.run(
            [sys.executable, str(SHIM), "--self-test"],
            capture_output=True,
            text=True,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"shim --self-test failed:\nstdout={proc.stdout}\nstderr={proc.stderr}",
        )
        self.assertIn("self-test OK", proc.stdout)

    def test_shim_rejects_unknown_flag(self):
        proc = subprocess.run(
            [sys.executable, str(SHIM), "--not-a-real-flag"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(
            proc.returncode,
            0,
            "shim must reject unknown flags (fabricated-pass guard)",
        )

    def test_shim_requires_checkpoint_dir_with_do_dump(self):
        proc = subprocess.run(
            [sys.executable, str(SHIM), "--do-dump"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(
            proc.returncode,
            0,
            "shim --do-dump without --checkpoint-dir must be a hard error "
            "(FR-EX-08: missing input never silently skips)",
        )

    def test_shim_rejects_mutually_exclusive_flags(self):
        proc = subprocess.run(
            [sys.executable, str(SHIM), "--self-test", "--do-dump"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(
            proc.returncode,
            0,
            "--self-test and --do-dump must be mutually exclusive",
        )


class RustHarnessAlignment(unittest.TestCase):
    """The Rust test files this workflow feeds must actually consume
    the env vars this oracle asserts are set."""

    def test_parity_voxtral_reads_vokra_voxtral_gguf(self):
        self.assertTrue(PARITY_TEST.is_file())
        src = PARITY_TEST.read_text(encoding="utf-8")
        self.assertIn("VOKRA_VOXTRAL_GGUF", src)

    def test_voxtral_real_gguf_reads_vokra_voxtral_gguf(self):
        self.assertTrue(REAL_GGUF_TEST.is_file())
        src = REAL_GGUF_TEST.read_text(encoding="utf-8")
        self.assertIn("VOKRA_VOXTRAL_GGUF", src)

    def test_voxtral_transcription_prompt_reads_ref_dir_and_gguf(self):
        # The transcription-prompt test consumes VOKRA_VOXTRAL_REF_DIR
        # (offline prompt.json) + VOKRA_VOXTRAL_GGUF (compact vocab).
        self.assertTrue(PROMPT_TEST.is_file())
        src = PROMPT_TEST.read_text(encoding="utf-8")
        self.assertIn("VOKRA_VOXTRAL_REF_DIR", src)
        self.assertIn("VOKRA_VOXTRAL_GGUF", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
