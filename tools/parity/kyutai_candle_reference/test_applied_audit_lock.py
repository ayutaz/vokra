#!/usr/bin/env -S uv run --no-project --python 3.12
"""Focused tests for the fixed application-89 advisory profile."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import audit_lock
import current_artifact


ACTUAL_APPLIED = Path(os.environ.get(
    "VOKRA_CANDLE_APPLIED_ARTIFACT_ROOT",
    "/private/tmp/vokra-candle-application-result-89-20261016.rzuwzN",
))
CURRENT_ENVELOPE = Path(os.environ["VOKRA_CANDLE_CURRENT_ENVELOPE_ROOT"]) / "envelope.json" if os.environ.get("VOKRA_CANDLE_CURRENT_ENVELOPE_ROOT") else None


def _report() -> dict:
    return {
        "database": {"advisory-count": 42, "last-commit": None},
        "lockfile": {"dependency-count": 154},
        "settings": {"ignore": [], "target_arch": [], "target_os": [], "severity": None, "informational_warnings": ["unmaintained", "unsound", "notice"]},
        "vulnerabilities": {"found": False, "count": 0, "list": []},
        "warnings": {"unmaintained": [], "unsound": [], "notice": []},
    }


def _provenance() -> dict[str, str]:
    return {"commit": audit_lock.ADVISORY_DB_COMMIT, "tree": audit_lock.ADVISORY_DB_TREE, "tracked_untracked_status": "clean"}


def _actual_files() -> dict[str, Path]:
    return {name: ACTUAL_APPLIED / name for name, _size, _digest in audit_lock.APPLIED_MEMBERS}


class AppliedAuditProfileTests(unittest.TestCase):
    def test_profile_is_registered_and_closed(self) -> None:
        self.assertIs(audit_lock.resolve_profile("applied"), audit_lock.APPLIED_PROFILE)
        self.assertEqual(audit_lock.APPLIED_PROFILE.producer_run_id, 38758745134)
        self.assertEqual(audit_lock.APPLIED_PROFILE.artifact_id, 11985400172)
        self.assertEqual(audit_lock.APPLIED_PROFILE.lock_sha256, "21bb31459ac8c2f337cba3e9f79d0e64f48259dc4ed0df9c99ccb0480578a7d6")
        with self.assertRaises(audit_lock.AuditError):
            audit_lock.resolve_profile(type(audit_lock.APPLIED_PROFILE)(**audit_lock.APPLIED_PROFILE.__dict__))

    def test_fixed_db_absence_requires_real_crates_directory(self) -> None:
        if not ACTUAL_APPLIED.is_dir():
            self.skipTest("uses the authenticated application-89 lock fixture")
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "db"
            (db / "crates").mkdir(parents=True)
            facts = audit_lock._validate_applied_source_absence(db, ACTUAL_APPLIED / "final.lock")
            self.assertEqual(facts["source_absent_package_count"], 17)
            self.assertEqual(len(facts["source_absent_packages"]), 17)

    def test_missing_or_symlinked_db_crates_directory_rejects_coverage(self) -> None:
        if not ACTUAL_APPLIED.is_dir():
            self.skipTest("uses the authenticated application-89 lock fixture")
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "db"
            db.mkdir()
            with self.assertRaises(audit_lock.AuditError):
                audit_lock._validate_applied_source_absence(db, ACTUAL_APPLIED / "final.lock")
            (db / "crates").symlink_to(Path(temp) / "missing-crates")
            with self.assertRaises(audit_lock.AuditError):
                audit_lock._validate_applied_source_absence(db, ACTUAL_APPLIED / "final.lock")

    def test_any_fixed_db_path_crate_rejects_coverage(self) -> None:
        if not ACTUAL_APPLIED.is_dir():
            self.skipTest("uses the authenticated application-89 lock fixture")
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "db"
            (db / "crates").mkdir(parents=True)
            (db / "crates" / "gemm").write_bytes(b"unexpected")
            with self.assertRaises(audit_lock.AuditError):
                audit_lock._validate_applied_source_absence(db, ACTUAL_APPLIED / "final.lock")

    def test_dangling_fixed_db_path_symlink_rejects_coverage(self) -> None:
        if not ACTUAL_APPLIED.is_dir():
            self.skipTest("uses the authenticated application-89 lock fixture")
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "db"
            (db / "crates").mkdir(parents=True)
            (db / "crates" / "moshi").symlink_to(Path(temp) / "missing")
            with self.assertRaises(audit_lock.AuditError):
                audit_lock._validate_applied_source_absence(db, ACTUAL_APPLIED / "final.lock")

    def test_wrong_source_identity_rejects_coverage(self) -> None:
        if not ACTUAL_APPLIED.is_dir():
            self.skipTest("uses the authenticated application-89 lock fixture")
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "db"
            (db / "crates").mkdir(parents=True)
            lock = Path(temp) / "final.lock"
            text = (ACTUAL_APPLIED / "final.lock").read_text()
            lock.write_text(text.replace('name = "moshi"\nversion = "0.6.4"', 'name = "moshi"\nversion = "0.6.4"\nsource = "registry+https://github.com/rust-lang/crates.io-index"', 1))
            with self.assertRaises(audit_lock.AuditError):
                audit_lock._validate_applied_source_absence(db, lock)

    def test_source_absent_version_and_checksum_mutations_reject(self) -> None:
        if not ACTUAL_APPLIED.is_dir():
            self.skipTest("uses the authenticated application-89 lock fixture")
        for mutation in (
            ('name = "moshi"\nversion = "0.6.4"', 'name = "moshi"\nversion = "9.9.9"'),
            ('name = "moshi"\nversion = "0.6.4"', 'name = "moshi"\nversion = "0.6.4"\nchecksum = "deadbeef"'),
        ):
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as temp:
                    db = Path(temp) / "db"
                    (db / "crates").mkdir(parents=True)
                    lock = Path(temp) / "final.lock"
                    lock.write_text((ACTUAL_APPLIED / "final.lock").read_text().replace(*mutation, 1))
                    with self.assertRaises(audit_lock.AuditError):
                        audit_lock._validate_applied_source_absence(db, lock)

    def test_applied_producer_binding_is_exact(self) -> None:
        run = {"id": audit_lock.APPLIED_PROFILE.producer_run_id, "status": "completed", "conclusion": "success", "head_sha": audit_lock.APPLIED_PROFILE.producer_head_sha, "run_attempt": 1}
        artifact = {"id": audit_lock.APPLIED_PROFILE.artifact_id, "name": audit_lock.APPLIED_PROFILE.artifact_name, "size_in_bytes": audit_lock.APPLIED_PROFILE.artifact_size, "expired": False, "expires_at": "2099-01-01T00:00:00Z", "workflow_run": {"id": run["id"], "head_sha": run["head_sha"]}}
        def api(url: str, _token: str, _deadline: float) -> dict:
            return run if "/artifacts?" not in url else {"artifacts": [artifact]}
        facts = audit_lock.validate_producer(repository="ayutaz/vokra", token="test", api=api, deadline=audit_lock._now() + 5, profile="applied")
        self.assertEqual(facts["artifact_id"], audit_lock.APPLIED_PROFILE.artifact_id)
        with self.assertRaises(audit_lock.AuditError):
            audit_lock.validate_producer(repository="ayutaz/vokra", token="test", api=api, deadline=audit_lock._now() + 5, profile="applied", expected_head="wrong")

    def test_authenticated_application_artifact_has_eight_members(self) -> None:
        if not ACTUAL_APPLIED.is_dir() or (ACTUAL_APPLIED / ".git").exists():
            self.skipTest("hosted application-89 artifact is not present")
        if CURRENT_ENVELOPE is not None:
            envelope = current_artifact.validate_envelope(json.loads(CURRENT_ENVELOPE.read_text()))
            receipt = audit_lock.validate_artifact(ACTUAL_APPLIED, profile="current", envelope=envelope)
        else:
            receipt = audit_lock.validate_artifact(ACTUAL_APPLIED, profile="applied")
        self.assertEqual(len(receipt["members"]), 8)
        if CURRENT_ENVELOPE is None:
            self.assertEqual(receipt["total_bytes"], audit_lock.APPLIED_PROFILE.extracted_bytes)

    def test_current_envelope_total_allows_registered_extraction_over_4mib(self) -> None:
        total = 4_333_048
        sizes = [500_000] * 7 + [833_048]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for index, (name, size) in enumerate(zip(current_artifact.MEMBER_NAMES, sizes, strict=True)):
                data = bytes((index + 1,)) * size
                (root / name).write_bytes(data)
                rows.append({"path": name, "bytes": size, "sha256": audit_lock._sha256(data)})
            envelope = {
                "schema": current_artifact.CURRENT_SCHEMA,
                "repository": current_artifact.PRODUCTION_REPOSITORY,
                "caller": "applied-advisory",
                "workflow_name": current_artifact.CALLERS["applied-advisory"],
                "caller_workflow_path": current_artifact.CALLER_PATHS["applied-advisory"],
                "producer_workflow_path": current_artifact.PRODUCER_WORKFLOW_PATH,
                "producer_workflow_name": current_artifact.PRODUCER_WORKFLOW_NAME,
                "producer_job_name": current_artifact.PRODUCER_JOB_NAME,
                "run_id": 123,
                "run_attempt": 1,
                "source_head": "a" * 40,
                "workflow_sha": "b" * 40,
                "run_ref": "refs/pull/123/merge",
                "artifact_id": 456,
                "artifact_name": "kyutai-candle-paste-application-evidence-123-1",
                "artifact_digest": "c" * 64,
                "artifact_size": 259_361,
                "source_commit": current_artifact.SOURCE_COMMIT,
                "final_lock_sha256": current_artifact.FINAL_LOCK_SHA256,
                "members": rows,
                "application_receipt_sha256": rows[0]["sha256"],
            }
            with mock.patch.object(audit_lock, "_validate_applied_metadata"):
                receipt = audit_lock.validate_artifact(root, profile="current", envelope=envelope)
            self.assertEqual(receipt["total_bytes"], total)
            self.assertGreater(receipt["total_bytes"], audit_lock.MAX_ARTIFACT_BYTES)

            (root / "unexpected").write_bytes(b"x")
            with self.assertRaises(audit_lock.AuditError):
                audit_lock.validate_artifact(root, profile="current", envelope=envelope)
            (root / "unexpected").unlink()

            mutated = dict(envelope)
            mutated["members"] = [dict(row) for row in envelope["members"]]
            mutated["members"][1]["sha256"] = "0" * 64
            with self.assertRaisesRegex(audit_lock.AuditError, "artifact member hash/size mismatch"):
                audit_lock.validate_artifact(root, profile="current", envelope=mutated)

            member_over_cap = dict(envelope)
            member_over_cap["members"] = [dict(row) for row in envelope["members"]]
            member_over_cap["members"][0]["bytes"] = current_artifact.MAX_MEMBER_BYTES + 1
            with self.assertRaises(current_artifact.CurrentArtifactError):
                current_artifact.validate_envelope(member_over_cap)

    def test_authenticated_application_metadata_binds_cpu_whole_and_lock(self) -> None:
        if not ACTUAL_APPLIED.is_dir():
            self.skipTest("hosted application-89 artifact is not present")
        audit_lock._validate_applied_metadata(_actual_files())
        final = json.loads((ACTUAL_APPLIED / "application_receipt.json").read_text())
        self.assertEqual(final["status"], "SOURCE_APPLICATION_REVIEW_PENDING")

    def test_coverage_failure_cannot_become_pass_with_rc0_and_count154(self) -> None:
        if not ACTUAL_APPLIED.is_dir():
            self.skipTest("uses the authenticated application-89 artifact")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cwd, cargo_home, db = (root / name for name in ("cwd", "cargo-home", "db"))
            cwd.mkdir(); cargo_home.mkdir(); (db / "crates").mkdir(parents=True)
            (db / "crates" / "gemm").mkdir()
            report_path = root / "report.json"
            with mock.patch.dict(os.environ, {"HOME": str(root), "GITHUB_ACTIONS": "true", "RUNNER_OS": "Linux", "RUNNER_ARCH": "X64", "GITHUB_TOKEN": "test"}, clear=False):
                rc = audit_lock.run_audit(
                    artifact_root=ACTUAL_APPLIED, db=db, cwd=cwd, cargo_home=cargo_home,
                    auditor="cargo-audit", repository="ayutaz/vokra", output=report_path,
                    deadline_seconds=10, profile="applied",
                    verifier=lambda *args, **kwargs: {"cargo_audit_version": "0.22.2", "binary_path": "/trusted/cargo-audit", "release_commit": audit_lock.CARGO_AUDIT_RELEASE_COMMIT},
                    producer_validator=lambda **kwargs: {"run_id": audit_lock.APPLIED_PROFILE.producer_run_id},
                    artifact_validator=lambda *args, **kwargs: {"members": {}, "total_bytes": audit_lock.APPLIED_PROFILE.extracted_bytes},
                    provenance=lambda *args, **kwargs: _provenance(),
                    command_runner=lambda *args, **kwargs: audit_lock.CommandResult(tuple(args[0]), 0, json.dumps(_report()).encode(), b"", False, False),
                    host_guard=lambda _env: None,
                )
            self.assertEqual(rc, 2)
            self.assertNotEqual(json.loads(report_path.read_text()).get("status"), "PASS")

    def test_complete_fixed_coverage_allows_clean_native_report(self) -> None:
        if not ACTUAL_APPLIED.is_dir():
            self.skipTest("uses the authenticated application-89 artifact")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cwd, cargo_home, db = (root / name for name in ("cwd", "cargo-home", "db"))
            cwd.mkdir(); cargo_home.mkdir(); (db / "crates").mkdir(parents=True)
            report_path = root / "report.json"
            calls: list[tuple[str, ...]] = []
            def command(*args, **kwargs):
                calls.append(tuple(args[0]))
                return audit_lock.CommandResult(tuple(args[0]), 0, json.dumps(_report()).encode(), b"", False, False)
            with mock.patch.dict(os.environ, {"HOME": str(root), "GITHUB_ACTIONS": "true", "RUNNER_OS": "Linux", "RUNNER_ARCH": "X64", "GITHUB_TOKEN": "test"}, clear=False):
                rc = audit_lock.run_audit(
                    artifact_root=ACTUAL_APPLIED, db=db, cwd=cwd, cargo_home=cargo_home,
                    auditor="cargo-audit", repository="ayutaz/vokra", output=report_path,
                    deadline_seconds=10, profile="applied",
                    verifier=lambda *args, **kwargs: {"cargo_audit_version": "0.22.2", "binary_path": "/trusted/cargo-audit", "release_commit": audit_lock.CARGO_AUDIT_RELEASE_COMMIT},
                    producer_validator=lambda **kwargs: {"run_id": audit_lock.APPLIED_PROFILE.producer_run_id},
                    artifact_validator=lambda *args, **kwargs: {"members": {}, "total_bytes": audit_lock.APPLIED_PROFILE.extracted_bytes},
                    provenance=lambda *args, **kwargs: _provenance(), command_runner=command,
                    host_guard=lambda _env: None,
                )
            self.assertEqual(rc, 0)
            self.assertTrue(calls)
            self.assertEqual(json.loads(report_path.read_text())["status"], "PASS")

    def test_post_command_created_path_cannot_become_pass(self) -> None:
        if not ACTUAL_APPLIED.is_dir():
            self.skipTest("uses the authenticated application-89 artifact")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cwd, cargo_home, db = (root / name for name in ("cwd", "cargo-home", "db"))
            cwd.mkdir(); cargo_home.mkdir(); (db / "crates").mkdir(parents=True)
            report_path = root / "report.json"
            def command(*args, **kwargs):
                (db / "crates" / "gemm").mkdir()
                return audit_lock.CommandResult(tuple(args[0]), 0, json.dumps(_report()).encode(), b"", False, False)
            with mock.patch.dict(os.environ, {"HOME": str(root), "GITHUB_ACTIONS": "true", "RUNNER_OS": "Linux", "RUNNER_ARCH": "X64", "GITHUB_TOKEN": "test"}, clear=False):
                rc = audit_lock.run_audit(
                    artifact_root=ACTUAL_APPLIED, db=db, cwd=cwd, cargo_home=cargo_home,
                    auditor="cargo-audit", repository="ayutaz/vokra", output=report_path,
                    deadline_seconds=10, profile="applied",
                    verifier=lambda *args, **kwargs: {"cargo_audit_version": "0.22.2", "binary_path": "/trusted/cargo-audit", "release_commit": audit_lock.CARGO_AUDIT_RELEASE_COMMIT},
                    producer_validator=lambda **kwargs: {"run_id": audit_lock.APPLIED_PROFILE.producer_run_id},
                    artifact_validator=lambda *args, **kwargs: {"members": {}, "total_bytes": audit_lock.APPLIED_PROFILE.extracted_bytes},
                    provenance=lambda *args, **kwargs: _provenance(), command_runner=command,
                    host_guard=lambda _env: None,
                )
            self.assertEqual(rc, 2)
            self.assertNotEqual(json.loads(report_path.read_text())["status"], "PASS")

    def test_native_vulnerability_return1_remains_failure(self) -> None:
        if not ACTUAL_APPLIED.is_dir():
            self.skipTest("uses the authenticated application-89 artifact")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cwd, cargo_home, db = (root / name for name in ("cwd", "cargo-home", "db"))
            cwd.mkdir(); cargo_home.mkdir(); (db / "crates").mkdir(parents=True)
            report_path = root / "report.json"
            report = _report()
            report["vulnerabilities"] = {"found": True, "count": 1, "list": [{}]}
            with mock.patch.dict(os.environ, {"HOME": str(root), "GITHUB_ACTIONS": "true", "RUNNER_OS": "Linux", "RUNNER_ARCH": "X64", "GITHUB_TOKEN": "test"}, clear=False):
                rc = audit_lock.run_audit(
                    artifact_root=ACTUAL_APPLIED, db=db, cwd=cwd, cargo_home=cargo_home,
                    auditor="cargo-audit", repository="ayutaz/vokra", output=report_path,
                    deadline_seconds=10, profile="applied",
                    verifier=lambda *args, **kwargs: {"cargo_audit_version": "0.22.2", "binary_path": "/trusted/cargo-audit", "release_commit": audit_lock.CARGO_AUDIT_RELEASE_COMMIT},
                    producer_validator=lambda **kwargs: {"run_id": audit_lock.APPLIED_PROFILE.producer_run_id},
                    artifact_validator=lambda *args, **kwargs: {"members": {}, "total_bytes": audit_lock.APPLIED_PROFILE.extracted_bytes},
                    provenance=lambda *args, **kwargs: _provenance(),
                    command_runner=lambda *args, **kwargs: audit_lock.CommandResult(tuple(args[0]), 1, json.dumps(report).encode(), b"finding", False, False),
                    host_guard=lambda _env: None,
                )
            facts = json.loads(report_path.read_text())
            self.assertEqual(rc, 1)
            self.assertEqual(facts["status"], "FAIL")
            self.assertEqual(facts["audit"]["classification"], "FAIL_FINDINGS")

    def test_lock_mutation_during_command_and_after_provenance_rejects(self) -> None:
        if not ACTUAL_APPLIED.is_dir():
            self.skipTest("uses the authenticated application-89 artifact")
        for mutate_after_provenance in (False, True):
            with self.subTest(mutate_after_provenance=mutate_after_provenance), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                artifact = root / "artifact"
                artifact.mkdir()
                (artifact / "final.lock").write_bytes((ACTUAL_APPLIED / "final.lock").read_bytes())
                cwd, cargo_home, db = (root / name for name in ("cwd", "cargo-home", "db"))
                cwd.mkdir(); cargo_home.mkdir(); (db / "crates").mkdir(parents=True)
                report_path = root / "report.json"
                calls = {"provenance": 0}
                def provenance(*args, **kwargs):
                    calls["provenance"] += 1
                    if mutate_after_provenance and calls["provenance"] == 2:
                        (artifact / "final.lock").write_bytes((artifact / "final.lock").read_bytes() + b"\n")
                    return _provenance()
                def command(*args, **kwargs):
                    if not mutate_after_provenance:
                        (artifact / "final.lock").write_bytes((artifact / "final.lock").read_bytes() + b"\n")
                    return audit_lock.CommandResult(tuple(args[0]), 0, json.dumps(_report()).encode(), b"", False, False)
                with mock.patch.dict(os.environ, {"HOME": str(root), "GITHUB_ACTIONS": "true", "RUNNER_OS": "Linux", "RUNNER_ARCH": "X64", "GITHUB_TOKEN": "test"}, clear=False):
                    rc = audit_lock.run_audit(
                        artifact_root=artifact, db=db, cwd=cwd, cargo_home=cargo_home,
                        auditor="cargo-audit", repository="ayutaz/vokra", output=report_path,
                        deadline_seconds=10, profile="applied",
                        verifier=lambda *args, **kwargs: {"cargo_audit_version": "0.22.2", "binary_path": "/trusted/cargo-audit", "release_commit": audit_lock.CARGO_AUDIT_RELEASE_COMMIT},
                        producer_validator=lambda **kwargs: {"run_id": audit_lock.APPLIED_PROFILE.producer_run_id},
                        artifact_validator=lambda *args, **kwargs: {"members": {}, "total_bytes": audit_lock.APPLIED_PROFILE.extracted_bytes},
                        provenance=provenance, command_runner=command,
                        host_guard=lambda _env: None,
                    )
                facts = json.loads(report_path.read_text())
                self.assertEqual(rc, 2)
                self.assertNotEqual(facts["status"], "PASS")
                self.assertIn(facts["audit"]["classification"], {"ERROR_LOCK_MUTATED", "ERROR_DATABASE_CHANGED"})

    def test_supplemental_deadline_is_fail_closed(self) -> None:
        if not ACTUAL_APPLIED.is_dir():
            self.skipTest("uses the authenticated application-89 lock fixture")
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "db"
            (db / "crates").mkdir(parents=True)
            with self.assertRaises(audit_lock.AuditError):
                audit_lock._validate_applied_source_absence(db, ACTUAL_APPLIED / "final.lock", deadline=audit_lock._now() - 1)

    def test_applied_report_keeps_native_filters_and154_count(self) -> None:
        facts = audit_lock.validate_report(_report(), profile="applied")
        self.assertEqual(facts["lock_dependency_count"], 154)
        bad = _report()
        bad["settings"]["ignore"] = ["RUSTSEC-2026-0001"]
        with self.assertRaises(audit_lock.AuditError):
            audit_lock.validate_report(bad, profile="applied")


if __name__ == "__main__":
    unittest.main()
