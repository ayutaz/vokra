"""Offline stdlib-only contract checks for the dormant cache-observer workflow."""

from __future__ import annotations

from pathlib import Path
import re
import unittest


LEAF = Path(__file__).resolve().parent
REPO = LEAF.parents[2]
WORKFLOW = REPO / ".github/workflows/kyutai-candle-cache-observer-compatibility.yml"
OLD_WORKFLOW = REPO / ".github/workflows/kyutai-candle-cpu-compatibility.yml"
CHECKER = LEAF / "check_cache_observer_compatibility.py"
LEGACY = LEAF / "check_cpu_compatibility.py"

APPLICATION_FILES = (
    "application_receipt.json",
    "baseline_cpu_metadata.json",
    "baseline_whole_metadata.json",
    "candidate.lock",
    "final.lock",
    "overlay_cpu_metadata.json",
    "overlay_whole_metadata.json",
    "source_receipt.json",
)

OLD_INPUTS = (
    ".github/workflows/kyutai-candle-cpu-compatibility.yml",
    ".github/workflows/kyutai-candle-paste-application.yml",
    "tools/parity/kyutai_candle_reference/check_cpu_compatibility.py",
    "tools/parity/kyutai_candle_reference/test_cpu_compatibility.py",
    "tools/parity/kyutai_candle_reference/apply_paste_overlay.py",
    "tools/parity/kyutai_candle_reference/collect_metadata.py",
    "tools/parity/kyutai_candle_reference/pin_remediation_diagnostic.py",
    "tools/parity/kyutai_candle_reference/prepare_paste_overlay.py",
    "docs/handoff/kyutai-candle-paste-parents-source-2026-10-15.json",
    "docs/handoff/kyutai-candle-pastey-source-2026-10-15.json",
    "docs/handoff/kyutai-candle-candidate-2026-10-15.lock",
    "docs/handoff/kyutai-candle-current-cpu-evidence-2026-10-16.json",
)

NEW_INPUTS = (
    ".gitattributes",
    "tools/parity/kyutai_candle_reference/asr_observer_patch.py",
    "tools/parity/kyutai_candle_reference/cache_observer_patch.py",
    "tools/parity/kyutai_candle_reference/check_cache_observer_compatibility.py",
    "tools/parity/kyutai_candle_reference/test_asr_observer_patch.py",
    "tools/parity/kyutai_candle_reference/test_cache_observer_patch.py",
    "tools/parity/kyutai_candle_reference/test_cache_observer_compatibility.py",
    "tools/parity/kyutai_candle_reference/test_cache_observer_workflow.py",
    "tools/parity/kyutai_candle_reference/pinned-asr.rs",
    "tools/parity/kyutai_candle_reference/pinned-LICENSE",
    "tools/parity/kyutai_candle_reference/pinned-transformer.rs",
    "tools/parity/kyutai_candle_reference/pinned-lm.rs",
)

WORKFLOW_PATH = ".github/workflows/kyutai-candle-cache-observer-compatibility.yml"
EVIDENCE_INPUTS = tuple(
    f"tools/parity/kyutai_candle_reference/application-evidence-89/{name}"
    for name in APPLICATION_FILES
)


class CacheObserverWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")
        cls.old_workflow = OLD_WORKFLOW.read_text(encoding="utf-8")
        cls.checker = CHECKER.read_text(encoding="utf-8")
        cls.legacy = LEGACY.read_text(encoding="utf-8")

    def test_one_shot_identity_and_fail_closed_trigger(self) -> None:
        self.assertIn("github.event_name == 'pull_request'", self.workflow)
        self.assertIn("github.event.action == 'labeled'", self.workflow)
        self.assertIn("types: [labeled]", self.workflow)
        self.assertIn("github.event.label.name == 'candle-cache-observer-compatibility-once-20261017'", self.workflow)
        self.assertIn("github.event.pull_request.number == 259", self.workflow)
        self.assertIn("github.run_attempt == 1", self.workflow)
        self.assertIn("github.repository == 'ayutaz/vokra'", self.workflow)
        self.assertIn("github.event.pull_request.head.repo.full_name == github.repository", self.workflow)
        self.assertNotRegex(self.workflow, r"(?m)^\s+push:")
        self.assertNotIn("synchronize", self.workflow)
        self.assertNotIn("workflow_dispatch", self.workflow)
        self.assertNotIn("pull_request_target", self.workflow)
        self.assertNotIn("candle-cpu-compatibility-confirm-once-20261016", self.workflow)
        self.assertNotIn("candle-cpu-compatibility-once-20261016", self.workflow)
        self.assertNotIn("secrets.", self.workflow)

    def test_permissions_concurrency_runner_and_timeout(self) -> None:
        self.assertRegex(self.workflow, r"permissions:\n  contents: read\n\nconcurrency:")
        self.assertNotIn("actions: write", self.workflow)
        self.assertIn("cancel-in-progress: false", self.workflow)
        self.assertIn("runs-on: ubuntu-22.04", self.workflow)
        self.assertIn("timeout-minutes: 20", self.workflow)
        self.assertNotIn("continue-on-error", self.workflow)
        self.assertNotIn("set +e", self.workflow)

    def test_exact_checkout_and_pinned_actions(self) -> None:
        self.assertIn("actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1", self.workflow)
        self.assertIn("ref: ${{ github.event.pull_request.head.sha }}", self.workflow)
        self.assertIn("persist-credentials: false", self.workflow)
        self.assertIn("sparse-checkout-cone-mode: false", self.workflow)
        self.assertIn("actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97", self.workflow)
        self.assertIn("astral-sh/setup-uv@c18668ad3cf93ea998bef934396af7bb5c839dc7", self.workflow)
        self.assertIn("version: '0.8.17'", self.workflow)
        self.assertIn("dtolnay/rust-toolchain@4360b52568e2003a75bf9bc1d59f33a8e3fc893c", self.workflow)
        self.assertIn("toolchain: 1.99.0", self.workflow)
        self.assertIn("actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a", self.workflow)

    def test_sparse_surface_reuses_legacy_metadata_interface_without_old_artifact(self) -> None:
        sparse = self.workflow.split("sparse-checkout: |", 1)[1].split("\n\n", 1)[0]
        sparse_paths = {line.strip() for line in sparse.splitlines() if line.strip()}
        expected_sparse = set(OLD_INPUTS) | set(NEW_INPUTS) | {WORKFLOW_PATH} | set(EVIDENCE_INPUTS)
        self.assertEqual(sparse_paths, expected_sparse)
        for path in (*OLD_INPUTS, *NEW_INPUTS, WORKFLOW_PATH, *EVIDENCE_INPUTS):
            self.assertIn(path, sparse_paths)
        self.assertNotIn("README", sparse)
        self.assertNotIn("            .git\n", sparse)
        self.assertNotIn("11985400172", self.workflow)
        self.assertNotIn("actions/download-artifact", self.workflow)
        self.assertIn("test ! -e tools/parity/cosyvoice2_llm_reference", self.workflow)
        self.assertIn("test ! -e tests/parity/silero_vad", self.workflow)
        self.assertIn('test "$(find "$evidence_root" -mindepth 1 -maxdepth 1 -type f | wc -l)" -eq 8', self.workflow)
        for name in APPLICATION_FILES:
            self.assertIn(f'test -f "$evidence_root/{name}"', self.workflow)

    def test_source_only_tests_run_before_actual_checker(self) -> None:
        source_pos = self.workflow.index("name: Run source-only tests")
        selftest_pos = self.workflow.index("name: Run cache observer self-test")
        compatibility_pos = self.workflow.index("name: Run compatibility runner offline tests")
        contract_pos = self.workflow.index("name: Run workflow contract tests")
        checker_pos = self.workflow.index("name: Build and probe actual cache observer library once")
        self.assertLess(source_pos, checker_pos)
        self.assertLess(source_pos, selftest_pos)
        self.assertLess(selftest_pos, checker_pos)
        self.assertLess(compatibility_pos, checker_pos)
        self.assertLess(contract_pos, checker_pos)
        for pattern in (
            "-p 'test_cache_observer_patch.py' -v",
            "-p 'test_asr_observer_patch.py' -v",
            "-p 'test_cache_observer_compatibility.py' -v",
            "-p 'test_cache_observer_workflow.py' -v",
        ):
            self.assertIn(pattern, self.workflow)
        self.assertNotIn("test_cpu_compatibility.py' -v", self.workflow)
        self.assertIn("UV_OFFLINE: '1'", self.workflow)
        self.assertIn("--no-project --offline --python 3.12", self.workflow)

    def test_actual_checker_contract_is_explicit_and_not_activated_by_tests(self) -> None:
        self.assertIn("VOKRA_KYUTAI_CANDLE_OBSERVER_REMOTE_LINUX: '1'", self.workflow)
        self.assertIn("VOKRA_CANDLE_WORKFLOW_HEAD: ${{ github.event.pull_request.head.sha }}", self.workflow)
        self.assertIn('--application-root "$GITHUB_WORKSPACE/tools/parity/kyutai_candle_reference/application-evidence-89"', self.workflow)
        self.assertNotIn('--application-root "$RUNNER_TEMP/application-evidence-89"', self.workflow)
        self.assertIn('--deadline-seconds 900', self.workflow)
        self.assertIn("--output \"$RUNNER_TEMP/candle-cache-observer-compatibility-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}/compatibility.json\"", self.workflow)
        self.assertIn("APPLICATION_FILES = legacy.APPLICATION_FILES", self.checker)
        for name in APPLICATION_FILES:
            self.assertIn(f'"{name}"', self.legacy)
        self.assertNotIn("VOKRA_CANDLE_REMOTE_LINUX", self.workflow)
        self.assertNotIn("cargo build", self.workflow)
        self.assertNotIn("rustc", self.workflow)

    def test_upload_is_bounded_receipt_evidence_only_and_retained(self) -> None:
        upload = self.workflow.split("- name: Upload bounded compatibility receipt and evidence", 1)[1]
        self.assertIn("if: always()", upload)
        self.assertIn("if-no-files-found: error", upload)
        self.assertIn("retention-days: 7", upload)
        self.assertIn("/compatibility.json", upload)
        self.assertIn("/evidence", upload)
        for forbidden in ("workspace", "models", "target", "application-evidence-89"):
            self.assertNotIn(forbidden, upload)

    def test_new_checker_is_source_only_and_does_not_invoke_upstream_in_contract_suite(self) -> None:
        self.assertIn("SOURCE_COMPATIBILITY_REVIEW_PENDING", self.checker)
        self.assertIn('"model_execution": "NOT_AUTHORIZED"', self.checker)
        self.assertIn('"upload": "NO_UPLOAD"', self.checker)
        self.assertNotIn("huggingface_hub", self.checker.lower())
        self.assertNotIn("cargo run", self.checker)


if __name__ == "__main__":
    unittest.main()
