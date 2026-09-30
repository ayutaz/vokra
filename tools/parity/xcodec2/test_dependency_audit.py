#!/usr/bin/env -S uv run --no-project --python 3.12 python
"""Stdlib-only tests for the XCodec2 model-free evidence contract."""

from __future__ import annotations

import ast
import base64
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest
import zipfile

import collect_dependency_evidence as collector
import dependency_audit as audit


class DependencyAuditTests(unittest.TestCase):
    def test_linux_closure_and_fail_closed_documents(self) -> None:
        lock = tomllib.loads(audit.LOCK.read_text(encoding="utf-8"))
        closure = audit.reachable_lock_identities(lock)
        self.assertEqual(len(closure), 63)
        self.assertEqual(len(closure) - 1, 62)
        manifest = audit.strict_json(audit.MANIFEST)
        rows = audit.strict_json(audit.ROWS)
        self.assertEqual(manifest["publication"], "NO_UPLOAD")
        self.assertEqual(rows["status"], "BLOCKED_PENDING_PRIMARY_BYTES")

    def test_marker_target_is_linux_not_macos(self) -> None:
        self.assertTrue(audit.marker_reaches("sys_platform != 'darwin' and platform_machine == 'x86_64'"))
        self.assertFalse(audit.marker_reaches("sys_platform == 'darwin'"))
        for version, expected in (("3.8", False), ("3.9", True), ("3.10", True), ("3.12", True), ("3.13", True)):
            environment = dict(audit.TARGET_ENV)
            environment["python_version"] = version
            environment["python_full_version"] = version + ".0"
            self.assertEqual(audit.marker_reaches("python_version >= '3.9'", environment), expected)
            self.assertEqual(audit.marker_reaches("python_full_version < '3.10.0'", environment), version == "3.8" or version == "3.9")
        self.assertTrue(audit.marker_reaches("python_full_version == '3.12'", {**audit.TARGET_ENV, "python_full_version": "3.12.0"}))
        self.assertTrue(audit.marker_reaches("python_version >= '3.12.0'", {**audit.TARGET_ENV, "python_version": "3.12"}))

    def test_activated_extra_reaches_aiohttp_subgraph(self) -> None:
        lock = tomllib.loads(audit.LOCK.read_text(encoding="utf-8"))
        closure = set(audit.reachable_lock_identities(lock))
        self.assertIn(("aiohttp", "3.14.3"), closure)
        self.assertIn(("yarl", "1.24.5"), closure)
        self.assertNotIn(("torch", "2.13.0"), closure)
        first = audit.reachable_lock_identities(lock)
        second = audit.reachable_lock_identities(lock)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))

    def test_manifest_policy_and_rows_hash_tampering_is_rejected(self) -> None:
        lock = tomllib.loads(audit.LOCK.read_text(encoding="utf-8"))
        closure = audit.reachable_lock_identities(lock)
        manifest = audit.strict_json(audit.MANIFEST)
        rows = audit.strict_json(audit.ROWS)
        actual = {
            "pyproject_sha256": audit.sha256_file(audit.PYPROJECT),
            "uv_lock_sha256": audit.sha256_file(audit.LOCK),
            "dependency_audit_sha256": audit.sha256_file(audit.ROWS),
        }
        policy_tampered = copy.deepcopy(manifest)
        policy_tampered["policy"]["automatic_exceptions"] = True
        with self.assertRaises(audit.AuditError):
            audit.validate_gate_documents(policy_tampered, rows, actual, closure)
        hash_tampered = dict(actual)
        hash_tampered["dependency_audit_sha256"] = "0" * 64
        with self.assertRaises(audit.AuditError):
            audit.validate_gate_documents(manifest, rows, hash_tampered, closure)

    def test_archive_license_bytes_are_literal_and_traversal_safe(self) -> None:
        with tempfile.TemporaryDirectory(prefix="xcodec2-audit-test-") as directory:
            path = Path(directory) / "demo.whl"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("demo/LICENSE", b"MIT\n")
                archive.writestr("demo/large-native.bin", b"x" * (collector.MAX_MEMBER_BYTES + 1))
            evidence = collector.archive_license_files(path)
            self.assertEqual(evidence[0]["bytes"], 4)
            self.assertEqual(evidence[0]["sha256"], collector.sha256_bytes(b"MIT\n"))

    def test_wheel_installed_license_and_native_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="xcodec2-binding-test-") as directory:
            path = Path(directory) / "demo.whl"
            payloads = {"demo/LICENSE": b"MIT\n", "demo/libdemo.so": b"\x7fELFnative", "demo/omitted.so": b"not-in-publisher-record", "demo/RECORD": b"source-side record", "demo-1.dist-info/METADATA": b"Name: demo\n", "demo-1.dist-info/WHEEL": b"Wheel-Version: 1.0\nTag: py3-none-any\n"}
            record_rows = []
            for name, value in payloads.items():
                if name == "demo/omitted.so":
                    continue
                digest = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).decode().rstrip("=")
                record_rows.append(f"{name},sha256={digest},{len(value)}")
            payloads["demo-1.dist-info/RECORD"] = ("\n".join(record_rows) + "\ndemo-1.dist-info/RECORD,,\n").encode()
            clean_path = Path(directory) / "clean.whl"
            with zipfile.ZipFile(clean_path, "w") as archive:
                for name, value in payloads.items():
                    if name != "demo/omitted.so":
                        archive.writestr(name, value)
            with zipfile.ZipFile(path, "w") as archive:
                for name, value in payloads.items():
                    archive.writestr(name, value)
            # Installer-generated RECORD/WHEEL bytes are intentionally excluded;
            # this simulates a legitimate RECORD rewrite after installation.
            installed = {name: {"path": name, "bytes": len(value), "sha256": collector.sha256_bytes(value)} for name, value in payloads.items() if not collector.is_generated_installer_path(name)}
            binding, failures = collector.compare_wheel_payloads(clean_path, installed)
            self.assertEqual(binding["status"], "PAYLOAD_SCOPED_VERIFIED")
            self.assertFalse(failures)
            binding, failures = collector.compare_wheel_payloads(path, installed)
            self.assertEqual(binding["status"], "BLOCKED")
            self.assertTrue(any("omitted.so" in failure for failure in failures))
            self.assertFalse(collector.is_generated_installer_path("demo/RECORD"))
            removed = dict(installed)
            removed.pop("demo/LICENSE")
            binding, failures = collector.compare_wheel_payloads(clean_path, removed)
            self.assertEqual(binding["status"], "BLOCKED")
            self.assertTrue(any("missing publisher RECORD path" in failure for failure in failures))
            removed_native = dict(installed)
            removed_native.pop("demo/libdemo.so")
            _, native_failures = collector.compare_wheel_payloads(clean_path, removed_native)
            self.assertTrue(any("demo/libdemo.so" in failure for failure in native_failures))
            tampered = copy.deepcopy(installed)
            tampered["demo/LICENSE"]["sha256"] = "0" * 64
            tampered["demo/libdemo.so"]["bytes"] += 1
            binding, failures = collector.compare_wheel_payloads(clean_path, tampered)
            self.assertEqual(binding["status"], "BLOCKED")
            self.assertEqual(len(failures), 2)

    def test_collector_has_no_model_imports(self) -> None:
        for filename in ("dependency_audit.py", "collect_dependency_evidence.py"):
            tree = ast.parse((Path(__file__).parent / filename).read_text(encoding="utf-8"))
            imported = {
                alias.name.split(".", 1)[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            }
            imported.update(
                node.module.split(".", 1)[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module
            )
            self.assertNotIn("torch", imported)
            self.assertNotIn("xcodec2", imported)

    def test_collector_cli_rejects_symlink_project_before_collection(self) -> None:
        with tempfile.TemporaryDirectory(prefix="xcodec2-cli-path-test-") as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            linked = root / "linked"
            linked.symlink_to(real, target_is_directory=True)
            old_argv = sys.argv
            old_collect = collector.collect
            old_write = collector.write_atomic
            try:
                sys.argv = ["collect_dependency_evidence.py", "--project", str(linked / "xcodec2"), "--output", str(root / "report.json"), "--expected-head", "0" * 40]
                collector.collect = lambda _head: (_ for _ in ()).throw(AssertionError("collection must not run"))
                collector.write_atomic = lambda _path, _report: (_ for _ in ()).throw(AssertionError("write must not run"))
                self.assertEqual(collector.main(), 2)
            finally:
                sys.argv = old_argv
                collector.collect = old_collect
                collector.write_atomic = old_write


if __name__ == "__main__":
    unittest.main()
