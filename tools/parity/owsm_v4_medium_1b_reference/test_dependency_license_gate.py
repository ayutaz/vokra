"""Offline causal tests for the OWSM dependency admission gate."""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "owsm_dependency_license_gate", ROOT / "dependency_license_gate.py"
)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GATE
SPEC.loader.exec_module(GATE)


def import_roots(source: str) -> set[str]:
    tree = ast.parse(source)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


class DependencyLicenseGateTests(unittest.TestCase):
    def test_checked_in_43_row_fixture_stays_pending(self) -> None:
        result = GATE.audit(ROOT)
        self.assertEqual(result["status"], GATE.BLOCKED_STATUS)
        self.assertEqual(result["lock_package_count"], 43)
        self.assertEqual(result["execution"], "NO_LOCAL_EXECUTION")

    def test_scalar_allow_flip_is_unbound_and_cli_blocks(self) -> None:
        with self.fixture_copy() as project:
            pyproject_path = project / "pyproject.toml"
            pyproject = pyproject_path.read_text(encoding="utf-8").replace(
                'dependency_license_audit = "PENDING_PRIMARY_SOURCE_AUDIT_FAIL_CLOSED"',
                'dependency_license_audit = "AUDITED_ALLOW"',
            )
            pyproject_path.write_text(pyproject, encoding="utf-8")
            result = GATE.audit(project)
            self.assertEqual(result["status"], GATE.UNBOUND_APPROVAL_STATUS)
            self.assertTrue(any("hash-bound" in error for error in result["errors"]))
            self.assertNotEqual(result["status"], "AUDITED_ALLOW")
            with self.argv("--project", str(project)), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(GATE.main(), 2)

    def test_transitive_lock_mutation_never_becomes_allow(self) -> None:
        with self.fixture_copy() as project:
            path = project / "uv.lock"
            lock = path.read_text(encoding="utf-8")
            old = 'name = "certifi"\nversion = "2026.7.22"'
            self.assertEqual(lock.count(old), 1)
            path.write_text(lock.replace(old, 'name = "certifi"\nversion = "2026.7.23"'), encoding="utf-8")
            result = GATE.audit(project)
            self.assertNotEqual(result["status"], "AUDITED_ALLOW")
            self.assertEqual(result["status"], GATE.BLOCKED_STATUS)

    def test_cuda_torch_row_is_rejected(self) -> None:
        with self.fixture_copy() as project:
            path = project / "uv.lock"
            lock = path.read_text(encoding="utf-8")
            old = 'name = "torch"\nversion = "2.6.0+cpu"'
            self.assertEqual(lock.count(old), 1)
            path.write_text(lock.replace(old, 'name = "torch"\nversion = "2.6.0+cuda"'), encoding="utf-8")
            result = GATE.audit(project)
            self.assertEqual(result["status"], "BLOCKED_LOCK_CONTRACT")

    def test_non_cpu_torch_index_is_rejected(self) -> None:
        with self.fixture_copy() as project:
            path = project / "uv.lock"
            lock = path.read_text(encoding="utf-8")
            self.assertGreater(lock.count("download.pytorch.org/whl/cpu"), 0)
            path.write_text(lock.replace("download.pytorch.org/whl/cpu", "example.invalid/other"), encoding="utf-8")
            result = GATE.audit(project)
            self.assertEqual(result["status"], "BLOCKED_LOCK_CONTRACT")

    def test_direct_pin_mutation_is_rejected(self) -> None:
        with self.fixture_copy() as project:
            path = project / "pyproject.toml"
            pyproject = path.read_text(encoding="utf-8")
            self.assertIn('"humanfriendly==10.0"', pyproject)
            path.write_text(pyproject.replace('"humanfriendly==10.0"', '"humanfriendly==10.1"'), encoding="utf-8")
            result = GATE.audit(project)
            self.assertEqual(result["status"], "BLOCKED_LOCK_CONTRACT")

    def test_unbound_allow_with_bad_lock_keeps_contract_status(self) -> None:
        with self.fixture_copy() as project:
            pyproject_path = project / "pyproject.toml"
            pyproject = pyproject_path.read_text(encoding="utf-8").replace(
                'dependency_license_audit = "PENDING_PRIMARY_SOURCE_AUDIT_FAIL_CLOSED"',
                'dependency_license_audit = "AUDITED_ALLOW"',
            )
            pyproject_path.write_text(pyproject, encoding="utf-8")
            lock_path = project / "uv.lock"
            lock = lock_path.read_text(encoding="utf-8")
            old = 'name = "torch"\nversion = "2.6.0+cpu"'
            self.assertEqual(lock.count(old), 1)
            lock = lock.replace(old, 'name = "torch"\nversion = "2.6.0+cuda"')
            lock_path.write_text(lock, encoding="utf-8")
            result = GATE.audit(project)
            self.assertEqual(result["status"], "BLOCKED_LOCK_CONTRACT")
            self.assertTrue(any("hash-bound" in error for error in result["errors"]))

    def test_gate_has_no_runtime_model_imports(self) -> None:
        source = (ROOT / "dependency_license_gate.py").read_text(encoding="utf-8")
        self.assertTrue(import_roots(source).isdisjoint({"torch", "numpy", "model"}))
        synthetic = "from torch import Tensor\nfrom numpy import array\nimport model\n"
        self.assertEqual(import_roots(synthetic), {"torch", "numpy", "model"})
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("numpy", sys.modules)

    @staticmethod
    @contextlib.contextmanager
    def fixture_copy():
        with tempfile.TemporaryDirectory(prefix="owsm-gate-test-") as directory:
            project = Path(directory)
            shutil.copy2(ROOT / "pyproject.toml", project / "pyproject.toml")
            shutil.copy2(ROOT / "uv.lock", project / "uv.lock")
            yield project

    @staticmethod
    @contextlib.contextmanager
    def argv(*arguments: str):
        previous = sys.argv
        sys.argv = ["dependency_license_gate.py", *arguments]
        try:
            yield
        finally:
            sys.argv = previous


if __name__ == "__main__":
    unittest.main()
