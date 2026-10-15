#!/usr/bin/env -S uv run --no-project --python 3.12
"""Offline contract tests for the authenticated candidate advisory profile."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import audit_lock


ACTUAL_CANDIDATE = Path(os.environ.get(
    "VOKRA_CANDLE_CANDIDATE_ARTIFACT_ROOT",
    "/private/tmp/vokra-candle-pin-result-27a-20261015.FnvBej",
))


def candidate_report() -> dict:
    return {
        "database": {"advisory-count": 42, "last-commit": None},
        "lockfile": {"dependency-count": 154},
        "settings": {
            "ignore": [], "target_arch": [], "target_os": [], "severity": None,
            "informational_warnings": ["unmaintained", "unsound", "notice"],
        },
        "vulnerabilities": {"found": False, "count": 0, "list": []},
        "warnings": {"unmaintained": [], "unsound": [], "notice": []},
    }


def synthetic_candidate_metadata_files(root: Path, mutate=None) -> dict[str, Path]:
    registry = "registry+https://example.invalid/index"
    package_rows = []
    lock_rows = []
    for index in range(154):
        name = "moshi" if index == 0 else f"crate-{index}"
        version = "0.1.0"
        source = None if index == 0 else registry
        package_id = f"path+file:///workspace#{name}@{version}" if source is None else f"{source}#{name}@{version}"
        package_rows.append({"id": package_id, "name": name, "version": version, "source": source})
        lock_rows.append(f'[[package]]\nname = "{name}"\nversion = "{version}"' + (f'\nsource = "{source}"' if source else ""))
    node_rows = [{"id": row["id"], "deps": []} for row in package_rows]
    node_rows[0]["deps"] = [{"name": "crate-1", "pkg": package_rows[1]["id"], "dep_kinds": [{"kind": None, "target": None}]}]
    whole = {
        "approval": None, "cpu_execution_profile": "NOT_CPU_EXECUTION_PROFILE", "gpu_execution": "NOT_AUTHORIZED",
        "model_execution": "NOT_AUTHORIZED", "platform_filter": None, "source_commit": "e6a55d2722a65870ef52a6c9f6ecfc0e90f38362",
        "status": "PIN_REMEDIATION_WHOLE_METADATA_UNREVIEWED", "target": None, "upload": "NO_UPLOAD",
        "metadata": {"packages": package_rows, "resolve": {"nodes": node_rows, "root": None}, "workspace_members": [package_rows[0]["id"]]},
    }
    cpu = {"approval": None, "gpu_execution": "NOT_AUTHORIZED", "model_execution": "NOT_AUTHORIZED", "platform_filter": "x86_64-unknown-linux-gnu", "target": "x86_64-unknown-linux-gnu", "upload": "NO_UPLOAD", "status": "PIN_REMEDIATION_CPU_METADATA_UNREVIEWED"}
    diagnostic = {"approval": None, "model_execution": "NOT_AUTHORIZED", "status": "PIN_REMEDIATION_SOURCE_ONLY_REVIEW_PENDING", "upload": "NO_UPLOAD"}
    if mutate is not None:
        mutate(whole, cpu)
    files = {
        audit_lock.CANDIDATE_RESULT_PREFIX + "candidate.lock": root / "candidate.lock",
        audit_lock.CANDIDATE_RESULT_PREFIX + "candidate_whole_metadata.json": root / "whole.json",
        audit_lock.CANDIDATE_RESULT_PREFIX + "candidate_cpu_metadata.json": root / "cpu.json",
        audit_lock.CANDIDATE_RESULT_PREFIX + "pin_diagnostic.json": root / "diagnostic.json",
    }
    files[audit_lock.CANDIDATE_RESULT_PREFIX + "candidate.lock"].write_text("version = 4\n\n" + "\n\n".join(lock_rows) + "\n")
    files[audit_lock.CANDIDATE_RESULT_PREFIX + "candidate_whole_metadata.json"].write_text(json.dumps(whole))
    files[audit_lock.CANDIDATE_RESULT_PREFIX + "candidate_cpu_metadata.json"].write_text(json.dumps(cpu))
    files[audit_lock.CANDIDATE_RESULT_PREFIX + "pin_diagnostic.json"].write_text(json.dumps(diagnostic))
    return files


class CandidateAuditProfileTests(unittest.TestCase):
    def test_only_registered_profiles_are_resolvable(self) -> None:
        self.assertIs(audit_lock.resolve_profile("baseline"), audit_lock.BASELINE_PROFILE)
        self.assertIs(audit_lock.resolve_profile("candidate"), audit_lock.CANDIDATE_PROFILE)
        with self.assertRaises(audit_lock.AuditError):
            audit_lock.resolve_profile("154")
        with self.assertRaises(audit_lock.AuditError):
            audit_lock.resolve_profile(type(audit_lock.CANDIDATE_PROFILE)(**audit_lock.CANDIDATE_PROFILE.__dict__))

    def test_candidate_report_requires154_and_preserves_audit_filters(self) -> None:
        facts = audit_lock.validate_report(candidate_report(), profile="candidate")
        self.assertEqual(facts["lock_dependency_count"], 154)
        for mutation in (154, 161):
            report = candidate_report()
            report["lockfile"]["dependency-count"] = mutation
            if mutation == 154:
                continue
            with self.assertRaises(audit_lock.AuditError):
                audit_lock.validate_report(report, profile="candidate")
        report = candidate_report()
        report["settings"]["ignore"] = ["RUSTSEC-2026-0001"]
        with self.assertRaises(audit_lock.AuditError):
            audit_lock.validate_report(report, profile="candidate")

    def test_portable_whole_graph_validator_rejects_identity_and_boundary_mutations(self) -> None:
        mutations = (
            lambda whole, _cpu: whole["metadata"]["packages"][1].update(version="9.9.9"),
            lambda whole, _cpu: whole["metadata"]["packages"][1].update(source="registry+https://other.invalid/index"),
            lambda whole, _cpu: whole["metadata"]["resolve"]["nodes"][0]["deps"][0].update(dep_kinds=[]),
            lambda whole, _cpu: whole["metadata"].update(packages=whole["metadata"]["packages"][:105], resolve={"nodes": whole["metadata"]["resolve"]["nodes"][:105], "root": None}),
            lambda whole, _cpu: whole["metadata"]["resolve"].update(root=whole["metadata"]["packages"][1]["id"]),
            lambda whole, _cpu: whole.update(source_commit="wrong-source-commit"),
            lambda whole, _cpu: whole.update(model_execution="AUTHORIZED"),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as temp:
                    files = synthetic_candidate_metadata_files(Path(temp), mutation)
                    with self.assertRaises(audit_lock.AuditError):
                        audit_lock._validate_candidate_metadata(files)

    def test_candidate_orchestration_matrix_is_profiled_without_real_artifact(self) -> None:
        provenance = {"commit": audit_lock.ADVISORY_DB_COMMIT, "tree": audit_lock.ADVISORY_DB_TREE, "tracked_untracked_status": "clean"}
        for tool_rc, findings, expected_rc, expected_class in (
            (0, False, 0, "PASS"), (1, True, 1, "FAIL_FINDINGS"), (0, True, 2, "ERROR_RC0_FINDINGS"),
        ):
            with self.subTest(tool_rc=tool_rc, findings=findings):
                with tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    cwd, home, db, artifact = (root / name for name in ("cwd", "home", "db", "artifact"))
                    cwd.mkdir(); home.mkdir(); db.mkdir(); artifact.mkdir()
                    report = candidate_report()
                    if findings:
                        report["vulnerabilities"] = {"found": True, "count": 1, "list": [{}]}
                    seen: dict[str, object] = {}
                    def artifact_validator(received_root, *, profile):
                        seen["root"] = received_root
                        seen["profile"] = profile
                        return {"members": {}, "total_bytes": 0}
                    def command(argv, **kwargs):
                        seen["argv"] = tuple(argv)
                        return audit_lock.CommandResult(tuple(argv), tool_rc, json.dumps(report).encode(), b"finding" if findings else b"", False, False)
                    with mock.patch.dict(os.environ, {"HOME": str(root), "GITHUB_TOKEN": "secret", "GITHUB_ACTIONS": "true", "RUNNER_OS": "Linux", "RUNNER_ARCH": "X64"}, clear=False):
                        rc = audit_lock.run_audit(
                            artifact_root=artifact, db=db, cwd=cwd, cargo_home=home,
                            auditor="cargo-audit", repository="ayutaz/vokra", output=root / "report.json",
                            deadline_seconds=10, profile="candidate",
                            verifier=lambda *args, **kwargs: {"cargo_audit_version": "0.22.2", "binary_path": "/trusted/cargo-audit", "binary_sha256": "a" * 64, "release_commit": audit_lock.CARGO_AUDIT_RELEASE_COMMIT},
                            producer_validator=lambda **kwargs: {"run_id": audit_lock.CANDIDATE_PROFILE.producer_run_id},
                            artifact_validator=artifact_validator,
                            provenance=lambda *args, **kwargs: provenance,
                            command_runner=command,
                            host_guard=lambda _env: None,
                        )
                    self.assertEqual((rc, json.loads((root / "report.json").read_text())["audit"]["classification"]), (expected_rc, expected_class))
                    self.assertIs(seen["profile"], audit_lock.CANDIDATE_PROFILE)
                    self.assertTrue(any(str(argument).endswith(audit_lock.CANDIDATE_PROFILE.lock_relative_path) for argument in seen["argv"]))

    def test_candidate_producer_binding_rejects_old_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calls: list[str] = []
            run = {
                "id": audit_lock.CANDIDATE_PROFILE.producer_run_id,
                "status": "completed", "conclusion": "success",
                "head_sha": audit_lock.CANDIDATE_PROFILE.producer_head_sha, "run_attempt": 1,
            }
            artifact = {
                "id": audit_lock.CANDIDATE_PROFILE.artifact_id,
                "name": audit_lock.CANDIDATE_PROFILE.artifact_name,
                "size_in_bytes": audit_lock.CANDIDATE_PROFILE.artifact_size,
                "expired": False, "expires_at": "2099-01-01T00:00:00Z",
                "workflow_run": {"id": run["id"], "head_sha": run["head_sha"]},
            }
            def api(url: str, _token: str, _deadline: float) -> dict:
                calls.append(url)
                return run if "/artifacts?" not in url else {"artifacts": [artifact]}
            binding = audit_lock.validate_producer(
                repository="ayutaz/vokra", token="test", api=api,
                deadline=audit_lock._now() + 5, profile="candidate",
            )
            self.assertEqual(binding["artifact_id"], audit_lock.CANDIDATE_PROFILE.artifact_id)
            with self.assertRaises(audit_lock.AuditError):
                audit_lock.validate_producer(
                    repository="ayutaz/vokra", token="test", api=api,
                    deadline=audit_lock._now() + 5, profile="candidate",
                    run_id=audit_lock.PRODUCER_RUN_ID,
                )

    def test_candidate_workflow_uses_only_registered_contract(self) -> None:
        workflow = Path(__file__).parents[3] / ".github/workflows/kyutai-candle-candidate-advisory.yml"
        text = workflow.read_text()
        for needle in (
            "--profile candidate", "artifact-ids: \"11961100426\"", "run-id: 38711006458",
            audit_lock.ADVISORY_DB_COMMIT, audit_lock.DOWNLOAD_ACTION_COMMIT,
            audit_lock.INSTALL_ACTION_COMMIT, "fallback: none", "digest-mismatch: error", "retention-days: 1",
            "persist-credentials: false", "sparse-checkout-cone-mode: false",
            ".github/workflows/kyutai-candle-source-advisory.yml",
        ):
            self.assertIn(needle, text)
        self.assertNotIn("--ignore", text)
        self.assertNotIn("--target-os", text)
        self.assertNotIn("cargo build", text)
        self.assertNotIn("cargo test", text)
        self.assertNotIn("cargo run", text)

    @unittest.skipUnless(ACTUAL_CANDIDATE.is_dir(), "hosted candidate artifact is not present")
    def test_candidate_orchestration_uses_whole_lock_and154_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cwd, home, db = (root / name for name in ("cwd", "home", "db"))
            cwd.mkdir(); home.mkdir(); db.mkdir()
            provenance = {"commit": audit_lock.ADVISORY_DB_COMMIT, "tree": audit_lock.ADVISORY_DB_TREE, "tracked_untracked_status": "clean"}
            calls: list[tuple[str, ...]] = []
            def command(argv, **kwargs):
                calls.append(tuple(argv))
                return audit_lock.CommandResult(tuple(argv), 1, json.dumps(candidate_report()).encode(), b"finding", False, False)
            with mock.patch.dict(os.environ, {"HOME": str(root), "GITHUB_TOKEN": "secret", "GITHUB_ACTIONS": "true", "RUNNER_OS": "Linux", "RUNNER_ARCH": "X64"}, clear=False):
                rc = audit_lock.run_audit(
                    artifact_root=ACTUAL_CANDIDATE, db=db, cwd=cwd, cargo_home=home,
                    auditor="cargo-audit", repository="ayutaz/vokra", output=root / "report.json",
                    deadline_seconds=10, profile="candidate",
                    verifier=lambda *args, **kwargs: {"cargo_audit_version": "0.22.2", "binary_path": "/trusted/cargo-audit", "binary_sha256": "a" * 64, "release_commit": audit_lock.CARGO_AUDIT_RELEASE_COMMIT},
                    producer_validator=lambda **kwargs: {"run_id": audit_lock.CANDIDATE_PROFILE.producer_run_id},
                    provenance=lambda *args, **kwargs: provenance,
                    command_runner=command,
                    host_guard=lambda _env: None,
                )
            self.assertEqual(rc, 1)
            self.assertEqual(json.loads((root / "report.json").read_text())["audit"]["report_facts"]["lock_dependency_count"], 154)
            self.assertTrue(any(str(argument).endswith(audit_lock.CANDIDATE_PROFILE.lock_relative_path) for argument in calls[0]))

    @unittest.skipUnless(ACTUAL_CANDIDATE.is_dir(), "hosted candidate artifact is not present")
    def test_recovered_candidate_artifact_matches_all13_members_and_graph(self) -> None:
        receipt = audit_lock.validate_artifact(ACTUAL_CANDIDATE, profile="candidate")
        self.assertEqual(len(receipt["members"]), 13)
        self.assertEqual(receipt["total_bytes"], audit_lock.CANDIDATE_PROFILE.extracted_bytes)

    @unittest.skipUnless(ACTUAL_CANDIDATE.is_dir(), "hosted candidate artifact is not present")
    def test_recovered_candidate_hash_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            copied = Path(temp) / "candidate"
            shutil.copytree(ACTUAL_CANDIDATE, copied)
            lock = next(copied.glob("*/candidate.lock"))
            lock.write_bytes(lock.read_bytes() + b"\n")
            with self.assertRaises(audit_lock.AuditError):
                audit_lock.validate_artifact(copied, profile="candidate")


if __name__ == "__main__":
    unittest.main(verbosity=2)
