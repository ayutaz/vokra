#!/usr/bin/env python3
"""Static regression tests for the hosted SBV2 real-parity contract.

These checks are intentionally offline and stdlib-only.  They do not invoke
Cargo, download checkpoints, or execute a model; they prevent a workflow edit
from turning the authenticated ignored test into an exit-0 no-op again.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "parity-sbv2-real.yml"
REAL_TEST = ROOT / "crates" / "vokra-models" / "tests" / "parity_sbv2_real.rs"


def workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def parity_step(text: str) -> str:
    start = text.index("- name: Run SBV2 real-checkpoint full-manifest parity test")
    end = text.index("- name: Emit atol-summary.json artifact", start)
    return text[start:end]


class RealParityWorkflowContractTests(unittest.TestCase):
    def test_rust_guard_remains_ignored_and_requires_fixture_replay(self):
        source = REAL_TEST.read_text(encoding="utf-8")
        self.assertRegex(
            source,
            r'(?m)^#\[test\]\s*\n#\[ignore\s*=\s*"[^"]+"\]\s*\n'
            r"fn parity_sbv2_real_waveform_matches_reference_dump\s*\(",
        )
        self.assertIn('"fixture-replay"', source)
        self.assertIn("is_absolute()", source)

    def test_workflow_selects_only_the_authorized_ignored_test(self):
        step = parity_step(workflow_text())
        self.assertIn(
            "parity_sbv2_real_waveform_matches_reference_dump -- --ignored --exact --nocapture --color never",
            step,
        )
        self.assertEqual(
            step.count("parity_sbv2_real_waveform_matches_reference_dump"),
            1,
        )
        self.assertIn('fixture_dir="$GITHUB_WORKSPACE/$FIXTURE_DIR"', step)
        self.assertIn('export VOKRA_SBV2_FIXTURE_DIR="$fixture_dir"', step)
        self.assertIn("export VOKRA_SBV2_G2P_MODE=fixture-replay", step)
        self.assertIn('reference_dump.manifest.json', step)
        text = workflow_text()
        self.assertLess(
            text.index("name: Verify regenerated GGUFs against committed sidecars"),
            text.index("name: Run SBV2 real-checkpoint full-manifest parity test"),
        )
        self.assertLess(
            text.index("name: Check independent reference fixture precondition"),
            text.index("name: Run SBV2 real-checkpoint full-manifest parity test"),
        )

    def test_workflow_requires_exact_non_ignored_execution_verdict(self):
        text = workflow_text()
        step = parity_step(text)
        self.assertIn("PARITY_VERDICT_EXIT", step)
        self.assertIn(
            "1 passed; 0 failed; 0 ignored; 0 measured; [0-9]+ filtered out",
            step,
        )
        self.assertIn(
            "steps.reference_fixture.outputs.available == 'true' && env.PARITY_EXIT == '0' && env.PARITY_VERDICT_EXIT == '0'",
            text[text.index("id: atol_summary"):],
        )
        final = text[text.index("- name: Fail if the dumper"):]
        self.assertIn("env.PARITY_VERDICT_EXIT != '0'", final)

    def test_verdict_parser_accepts_only_one_real_test(self):
        step = parity_step(workflow_text())
        match = re.search(r"verdict_line=\$\(grep -E '([^']+)'", step)
        self.assertIsNotNone(match)
        verdict_re = re.compile(match.group(1))
        self.assertTrue(
            verdict_re.fullmatch(
                "test result: ok. 1 passed; 0 failed; 0 ignored; "
                "0 measured; 11 filtered out; finished in 12.34s"
            )
        )
        self.assertFalse(
            verdict_re.fullmatch(
                "test result: ok. 0 passed; 0 failed; 1 ignored; "
                "0 measured; 11 filtered out; finished in 0.01s"
            )
        )
        self.assertFalse(
            verdict_re.fullmatch(
                "test result: ok. 0 passed; 0 failed; 0 ignored; "
                "0 measured; 11 filtered out; finished in 0.01s"
            )
        )
        self.assertFalse(
            verdict_re.fullmatch(
                "test result: FAILED. 1 passed; 1 failed; 0 ignored; "
                "0 measured; 11 filtered out; finished in 12.34s"
            )
        )
        self.assertFalse(
            verdict_re.fullmatch(
                "test result: ok. 2 passed; 0 failed; 0 ignored; "
                "0 measured; 10 filtered out; finished in 12.34s"
            )
        )

    def test_cpu_environment_is_recorded_before_real_parity_step(self):
        text = workflow_text()
        self.assertLess(
            text.index("name: Record hosted CPU environment"),
            text.index("name: Regenerate reference dump (sbv2_dump_reference.py --do-dump)"),
        )
        self.assertLess(
            text.index("name: Record hosted CPU environment"),
            text.index("name: Run SBV2 real-checkpoint full-manifest parity test"),
        )
        self.assertIn("model name", text)
        self.assertIn("isa_flags", text)
        self.assertIn('nproc 2>/dev/null', text)
        self.assertIn("torch.backends.cpu.get_cpu_capability()", text)


if __name__ == "__main__":
    unittest.main()
