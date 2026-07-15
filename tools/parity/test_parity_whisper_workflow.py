#!/usr/bin/env python3
"""TDD oracle for `.github/workflows/parity-whisper-real.yml`'s M4-14 surface.

Covers:
  * T05 — setup-job size matrix (5-size family, workflow_dispatch opt-in
    gating, cron/PR stay base-only) + dumper case mapping (5 sizes, unknown
    SIZE hard-fails per FR-EX-08).
  * T06 — per-size atol verdict step (PER_SIZE_ATOL mirror of the Rust
    `atol_for()`, drift guard, fabricated-pass guard). Added in the T06
    stage of the WP.

stdlib-only (zero-dep NFR-DS-02 — same discipline as
test_dump_whisper_vocab_gate.py): the workflow YAML is treated as text.
The bash blocks (matrix computation, dumper case mapping) and the verdict
step's inline Python are EXTRACTED from the YAML and EXECUTED against
controlled inputs, so these tests pin behaviour, not formatting. GitHub's
`${{ … }}` template expressions are substituted the way Actions would
(inputs render as empty strings on non-dispatch triggers).

Run: python3 tools/parity/test_parity_whisper_workflow.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "parity-whisper-real.yml"

ALL_SIZES = ["base", "small", "medium", "large-v3", "turbo"]

# Expected per-size matrix fields. env_var must equal the Rust harness's
# size_env_var() (parity_whisper.rs), fixture_dir its fixtures_dir_for(),
# hf_id the dumper's SUPPORTED_MODELS (dump_whisper_reference.py).
EXPECT_FIELDS = {
    "base": (
        "openai/whisper-base",
        "VOKRA_WHISPER_BASE_GGUF",
        "tests/parity/whisper_base",
        "whisper-base.gguf",
    ),
    "small": (
        "openai/whisper-small",
        "VOKRA_WHISPER_SMALL_GGUF",
        "tests/parity/whisper_small",
        "whisper-small.gguf",
    ),
    "medium": (
        "openai/whisper-medium",
        "VOKRA_WHISPER_MEDIUM_GGUF",
        "tests/parity/whisper_medium",
        "whisper-medium.gguf",
    ),
    "large-v3": (
        "openai/whisper-large-v3",
        "VOKRA_WHISPER_LARGE_V3_GGUF",
        "tests/parity/whisper_large_v3",
        "whisper-large-v3.gguf",
    ),
    "turbo": (
        "openai/whisper-large-v3-turbo",
        "VOKRA_WHISPER_TURBO_GGUF",
        "tests/parity/whisper_turbo",
        "whisper-turbo.gguf",
    ),
}

DISPATCH_INPUTS = ("include_large_v3", "include_small", "include_medium", "include_turbo")


def workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _extract_step_run(step_name: str) -> str:
    """Extract (dedented) the `run: |` body of the named step."""
    text = workflow_text()
    # From the step's `- name:` line to the next sibling `- name:` (6-space
    # indent) or end-of-file; then the `run: |` block inside it.
    step = re.search(
        rf"^      - name: {re.escape(step_name)}\n(.*?)(?=^      - name: |\Z)",
        text,
        re.DOTALL | re.MULTILINE,
    )
    if step is None:
        raise AssertionError(f"step not found in workflow: {step_name!r}")
    run = re.search(r"^        run: \|\n(.*)\Z", step.group(1), re.DOTALL | re.MULTILINE)
    if run is None:
        raise AssertionError(f"step {step_name!r} has no `run: |` block")
    return textwrap.dedent(run.group(1))


def _substitute_github_exprs(body: str, event_name: str, inputs: dict[str, str]) -> str:
    """Substitute `${{ … }}` the way Actions renders them.

    On non-dispatch triggers, `github.event.inputs.*` render as EMPTY
    strings — mirroring that here keeps the test honest about the cron/PR
    path.
    """
    body = body.replace("${{ github.event_name }}", event_name)

    def sub_input(m: re.Match[str]) -> str:
        name = m.group(1)
        if event_name != "workflow_dispatch":
            return ""
        return inputs.get(name, "")

    body = re.sub(r"\$\{\{ github\.event\.inputs\.([A-Za-z0-9_]+) \}\}", sub_input, body)
    leftovers = re.findall(r"\$\{\{[^}]*\}\}", body)
    if leftovers:
        raise AssertionError(f"unsubstituted GitHub expressions: {leftovers}")
    return body


def run_matrix_step(event_name: str, inputs: dict[str, str] | None = None) -> dict:
    """Execute the setup job's matrix computation; return the parsed matrix."""
    body = _substitute_github_exprs(
        _extract_step_run("Compute size matrix"), event_name, inputs or {}
    )
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "gh_output"
        summary = Path(td) / "gh_summary"
        out.touch()
        summary.touch()
        env = dict(os.environ, GITHUB_OUTPUT=str(out), GITHUB_STEP_SUMMARY=str(summary))
        proc = subprocess.run(
            ["bash", "-c", body], env=env, capture_output=True, text=True, cwd=REPO
        )
        if proc.returncode != 0:
            raise AssertionError(
                f"matrix bash failed rc={proc.returncode}\nstdout:{proc.stdout}\nstderr:{proc.stderr}"
            )
        for line in out.read_text().splitlines():
            if line.startswith("matrix="):
                return json.loads(line[len("matrix=") :])
    raise AssertionError("matrix step did not write matrix= to GITHUB_OUTPUT")


def matrix_sizes(matrix: dict) -> list[str]:
    return [row["size"] for row in matrix["include"]]


class MatrixComputation(unittest.TestCase):
    """T05 — 5-size family with workflow_dispatch opt-in gating (ADR M4-14 §D3)."""

    def test_pr_event_matrix_is_base_only(self):
        # Runner-minutes red-line: PR triggers must never pull multi-GB
        # checkpoints. Only base is allowed on pull_request.
        m = run_matrix_step("pull_request")
        self.assertEqual(matrix_sizes(m), ["base"])

    def test_schedule_event_matrix_is_base_only(self):
        # Weekly cron stays base-only (M2-06 design carried into M4-14).
        m = run_matrix_step("schedule")
        self.assertEqual(matrix_sizes(m), ["base"])

    def test_dispatch_with_defaults_runs_all_five_sizes(self):
        # workflow_dispatch defaults are all "true": dispatching is itself
        # the opt-in act (mirrors the pre-existing include_large_v3 gate).
        m = run_matrix_step(
            "workflow_dispatch", {k: "true" for k in DISPATCH_INPUTS}
        )
        self.assertEqual(sorted(matrix_sizes(m)), sorted(ALL_SIZES))

    def test_dispatch_matrix_rows_carry_correct_fields(self):
        m = run_matrix_step(
            "workflow_dispatch", {k: "true" for k in DISPATCH_INPUTS}
        )
        by_size = {row["size"]: row for row in m["include"]}
        for size, (hf_id, env_var, fixture_dir, gguf) in EXPECT_FIELDS.items():
            row = by_size[size]
            self.assertEqual(row["hf_id"], hf_id, size)
            self.assertEqual(row["env_var"], env_var, size)
            self.assertEqual(row["fixture_dir"], fixture_dir, size)
            self.assertEqual(row["gguf"], gguf, size)

    def test_dispatch_each_carryover_size_can_be_opted_out(self):
        # Each include_* input independently drops exactly its own leg;
        # base is unconditional.
        for opted_out, size in [
            ("include_small", "small"),
            ("include_medium", "medium"),
            ("include_turbo", "turbo"),
            ("include_large_v3", "large-v3"),
        ]:
            inputs = {k: "true" for k in DISPATCH_INPUTS}
            inputs[opted_out] = "false"
            m = run_matrix_step("workflow_dispatch", inputs)
            expect = sorted(s for s in ALL_SIZES if s != size)
            self.assertEqual(sorted(matrix_sizes(m)), expect, f"opting out {size}")

    def test_dispatch_all_carryovers_opted_out_leaves_base(self):
        m = run_matrix_step(
            "workflow_dispatch", {k: "false" for k in DISPATCH_INPUTS}
        )
        self.assertEqual(matrix_sizes(m), ["base"])


class DumperCaseMapping(unittest.TestCase):
    """T05 — dumper case must map all 5 sizes; unknown SIZE exits 2 (FR-EX-08)."""

    @staticmethod
    def _run_case(size: str) -> subprocess.CompletedProcess:
        body = _extract_step_run("Regenerate parity fixture (dump_whisper_reference.py)")
        case = re.search(r'(case "\$SIZE" in\n.*?esac)', body, re.DOTALL)
        if case is None:
            raise AssertionError("dumper step has no case \"$SIZE\" mapping")
        script = case.group(1) + '\necho "DUMPER_MODEL=${DUMPER_MODEL}"\n'
        return subprocess.run(
            ["bash", "-c", script],
            env=dict(os.environ, SIZE=size),
            capture_output=True,
            text=True,
        )

    def test_all_five_sizes_map_to_dumper_models(self):
        # The mapped value must be a SUPPORTED_MODELS key of the dumper
        # (whisper-{size}; turbo -> whisper-turbo which the dumper maps to
        # openai/whisper-large-v3-turbo).
        for size in ALL_SIZES:
            proc = self._run_case(size)
            self.assertEqual(proc.returncode, 0, f"{size}: {proc.stderr}")
            self.assertIn(f"DUMPER_MODEL=whisper-{size}", proc.stdout, size)

    def test_unknown_size_exits_2(self):
        proc = self._run_case("tiny")
        self.assertEqual(proc.returncode, 2, "unknown size must hard-fail (FR-EX-08)")
        self.assertIn("unknown SIZE", proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
