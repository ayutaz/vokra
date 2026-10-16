#!/usr/bin/env -S uv run --no-project --python 3.12
"""Stdlib-only tests for the bounded Candle advisory wrapper."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest import mock

import audit_lock


def good_report() -> dict:
    return {
        "database": {"advisory-count": 42, "last-commit": None, "last-updated": None},
        "lockfile": {"dependency-count": 161},
        "settings": {
            "ignore": [],
            "target_arch": [],
            "target_os": [],
            "severity": None,
            "informational_warnings": ["unmaintained", "unsound", "notice"],
        },
        "vulnerabilities": {"found": False, "count": 0, "list": []},
        "warnings": {"unmaintained": [], "unsound": [], "yanked": []},
    }


class AuditLockTests(unittest.TestCase):
    def _run_orchestrated(self, returncode: int, report: dict | None = None, *, stdout: bytes | None = None, timed_out: bool = False, stdout_truncated: bool = False, stderr_truncated: bool = False) -> tuple[int, dict]:
        report = report or good_report()
        result = audit_lock.CommandResult(
            ("cargo-audit", "audit"),
            returncode,
            stdout if stdout is not None else json.dumps(report).encode(),
            b"diagnostic",
            stdout_truncated,
            stderr_truncated,
            timed_out,
        )
        producer = {"run_id": audit_lock.PRODUCER_RUN_ID, "artifact_id": audit_lock.ARTIFACT_ID}
        receipt = {"members": {}, "total_bytes": 0}
        provenance = {"commit": audit_lock.ADVISORY_DB_COMMIT, "tree": audit_lock.ADVISORY_DB_TREE, "tracked_untracked_status": "clean"}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cwd, home, db, artifact = (root / name for name in ("cwd", "home", "db", "artifact"))
            cwd.mkdir(); home.mkdir(); db.mkdir(); artifact.mkdir()
            output = root / "report.json"
            with mock.patch.dict(os.environ, {"HOME": str(root / "runner-home"), "GITHUB_TOKEN": "secret", "GITHUB_ACTIONS": "true", "RUNNER_OS": "Linux", "RUNNER_ARCH": "X64"}, clear=False):
                rc = audit_lock.run_audit(
                    artifact_root=artifact, db=db, cwd=cwd, cargo_home=home, auditor="cargo-audit", repository="ayutaz/vokra", output=output,
                    deadline_seconds=10,
                    verifier=lambda *args, **kwargs: {"cargo_audit_version": "0.22.2", "binary_path": "/trusted/cargo-audit", "binary_sha256": "a" * 64, "release_commit": audit_lock.CARGO_AUDIT_RELEASE_COMMIT},
                    producer_validator=lambda **kwargs: producer,
                    artifact_validator=lambda _root: receipt,
                    provenance=lambda *args, **kwargs: provenance,
                    command_runner=lambda *args, **kwargs: audit_lock.CommandResult(tuple(args[0]), result.returncode, result.stdout, result.stderr, result.stdout_truncated, result.stderr_truncated, result.timed_out),
                    host_guard=lambda _env: None,
                )
            self.assertTrue(output.exists())
            return rc, json.loads(output.read_text())

    def test_pins_scope_command_and_native_null_commit_contract(self) -> None:
        argv = audit_lock.build_audit_argv("cargo-audit", Path("patched/Cargo.lock"), Path("db"))
        self.assertEqual(argv[0:2], ("cargo-audit", "audit"))
        for flag in ("--file", "--db", "--no-fetch", "--no-yanked", "--quiet", "--json", "--deny", "warnings"):
            self.assertIn(flag, argv)
        self.assertNotIn("--ignore", argv)
        self.assertNotIn("--target-os", argv)
        self.assertEqual(audit_lock.validate_report(good_report())["database_last_commit"], None)
        self.assertEqual(audit_lock.ADVISORY_DB_COMMIT, "edb58f5974b83eb8302332821cb6d389777aa351")
        self.assertEqual(audit_lock.ADVISORY_DB_TREE, "9cadd9fc13d75527fa5fd3f14376ed3a1a5e3b63")
        self.assertEqual(audit_lock.PRODUCER_HEAD_SHA, "b111f377bfbfc2f8d079f10791d61befb2d3ff5d")
        self.assertEqual(audit_lock.PRODUCER_RUN_ID, 38690241603)
        self.assertEqual(audit_lock.ARTIFACT_ID, 11951567227)

    def test_report_filters_counters_and_warning_classes_are_fail_closed(self) -> None:
        for key, value in (("ignore", ["RUSTSEC-2026-0001"]), ("target_os", ["linux"]), ("severity", "low")):
            report = good_report()
            report["settings"][key] = value
            with self.assertRaises(audit_lock.AuditError):
                audit_lock.validate_report(report)
        report = good_report()
        report["settings"]["informational_warnings"] = ["unmaintained"]
        with self.assertRaises(audit_lock.AuditError):
            audit_lock.validate_report(report)
        report = good_report()
        report["database"]["last-commit"] = audit_lock.ADVISORY_DB_COMMIT
        with self.assertRaises(audit_lock.AuditError):
            audit_lock.validate_report(report)
        report = good_report()
        report["lockfile"]["dependency-count"] = 105
        with self.assertRaises(audit_lock.AuditError):
            audit_lock.validate_report(report)
        report = good_report()
        report["vulnerabilities"] = {"found": True, "count": 1, "list": [{}]}
        self.assertEqual(audit_lock.validate_report(report)["vulnerability_count"], 1)

    def test_artifact_exact_members_and_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for relative, (size, digest) in audit_lock.EXPECTED_MEMBERS.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                # Test fixture is replaced with expected bytes by monkeypatching
                # the inventory for a small synthetic tree below.
                path.write_bytes(b"x" * min(size, 64))
            original = audit_lock.EXPECTED_MEMBERS.copy()
            try:
                audit_lock.EXPECTED_MEMBERS.clear()
                audit_lock.EXPECTED_MEMBERS.update(
                    {relative: (len(b"x" * min(size, 64)), audit_lock._sha256(b"x" * min(size, 64))) for relative, (size, _digest) in original.items()}
                )
                receipt = audit_lock.validate_artifact(root)
                self.assertEqual(set(receipt["members"]), set(original))
                (root / "patched/Cargo.lock").write_bytes(b"bad")
                with self.assertRaises(audit_lock.AuditError):
                    audit_lock.validate_artifact(root)
            finally:
                audit_lock.EXPECTED_MEMBERS.clear()
                audit_lock.EXPECTED_MEMBERS.update(original)
            (root / "source_receipt.json").unlink()
            (root / "source_receipt.json").symlink_to(root / "metadata.json")
            with self.assertRaises(audit_lock.AuditError):
                audit_lock.validate_artifact(root)

    def test_config_isolation_and_producer_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cwd = root / "cwd"
            home = root / "cargo-home"
            cwd.mkdir()
            home.mkdir()
            audit_lock._assert_clean_config(cwd, home)
            (cwd / ".cargo").mkdir()
            (cwd / ".cargo" / "audit.toml").write_text("[advisories]\nignore=['RUSTSEC-1']\n")
            with self.assertRaises(audit_lock.AuditError):
                audit_lock._assert_clean_config(cwd, home)
            (cwd / ".cargo" / "audit.toml").unlink()
            (cwd / ".cargo").rmdir()
            (home / "audit.toml").write_text("[target]\nos=['linux']\n")
            with self.assertRaises(audit_lock.AuditError):
                audit_lock._assert_clean_config(cwd, home)

        run = {"id": audit_lock.PRODUCER_RUN_ID, "status": "completed", "conclusion": "success", "head_sha": audit_lock.PRODUCER_HEAD_SHA, "run_attempt": 1}
        artifact = {"id": audit_lock.ARTIFACT_ID, "name": audit_lock.ARTIFACT_NAME, "size_in_bytes": audit_lock.ARTIFACT_SIZE, "expired": False, "expires_at": "2099-01-01T00:00:00Z", "workflow_run": {"id": audit_lock.PRODUCER_RUN_ID, "head_sha": audit_lock.PRODUCER_HEAD_SHA}}
        responses = {"run": run, "artifacts": {"artifacts": [artifact]}}
        def fake_api(url: str, token: str, deadline: float) -> dict:
            return responses["run"] if "/actions/runs/38690241603" in url and not url.endswith("artifacts?per_page=100") else responses["artifacts"]
        binding = audit_lock.validate_producer(repository="ayutaz/vokra", token="test", api=fake_api, deadline=audit_lock._now() + 10)
        self.assertEqual(binding["artifact_id"], audit_lock.ARTIFACT_ID)
        bad = dict(run, head_sha="wrong")
        responses["run"] = bad
        with self.assertRaises(audit_lock.AuditError):
            audit_lock.validate_producer(repository="ayutaz/vokra", token="test", api=fake_api, deadline=audit_lock._now() + 10)

    def test_external_git_provenance_is_distinct_from_native_report_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "db"
            cwd = Path(temp) / "cwd"
            db.mkdir()
            cwd.mkdir()
            def fake_git(argv, **_kwargs):
                if argv[-1] == "HEAD":
                    output = audit_lock.ADVISORY_DB_COMMIT.encode()
                elif argv[-1] == "HEAD^{tree}":
                    output = audit_lock.ADVISORY_DB_TREE.encode()
                else:
                    output = b""
                return audit_lock.CommandResult(tuple(argv), 0, output, b"", False, False)
            receipt = audit_lock.git_provenance(
                db,
                cwd=cwd,
                env=dict(os.environ),
                deadline=audit_lock._now() + 10,
                runner=fake_git,
            )
            self.assertEqual(receipt["commit"], audit_lock.ADVISORY_DB_COMMIT)
            self.assertEqual(receipt["tree"], audit_lock.ADVISORY_DB_TREE)
            def dirty_git(argv, **kwargs):
                result = fake_git(argv, **kwargs)
                if argv[-1] == "--untracked-files=all":
                    return audit_lock.CommandResult(tuple(argv), 0, b"?? dirty", b"", False, False)
                return result
            with self.assertRaises(audit_lock.AuditError):
                audit_lock.git_provenance(
                    db,
                    cwd=cwd,
                    env=dict(os.environ),
                    deadline=audit_lock._now() + 10,
                    runner=dirty_git,
                )
            def truncated_git(argv, **kwargs):
                result = fake_git(argv, **kwargs)
                return audit_lock.CommandResult(result.argv, result.returncode, result.stdout, result.stderr, True, False)
            with self.assertRaises(audit_lock.AuditError):
                audit_lock.git_provenance(db, cwd=cwd, env=dict(os.environ), deadline=audit_lock._now() + 10, runner=truncated_git)

    def test_return_codes_json_retention_caps_and_timeout_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            script = root / "fake-auditor"
            script.write_text(textwrap.dedent(f"""
                #!{sys.executable}
                import json, os, sys, time
                mode = sys.argv[1]
                if mode == 'timeout': time.sleep(2)
                if mode == 'stall': os.write(1, b'x'); time.sleep(2)
                if mode == 'noise': sys.stdout.write('x' * (2 * 1024 * 1024))
                if mode == 'bad': sys.stdout.write('{{bad')
                else: sys.stdout.write(json.dumps({good_report()!r}))
                sys.stderr.write('diagnostic')
                raise SystemExit(int(mode) if mode.isdigit() else 0)
            """).lstrip())
            script.chmod(script.stat().st_mode | stat.S_IXUSR)
            env = dict(os.environ)
            for mode, expected_rc in (("0", 0), ("1", 1), ("2", 2), ("bad", 0)):
                result = audit_lock.run_command((str(script), mode), cwd=root, env=env, deadline=audit_lock._now() + 5, stdout_cap=128, stderr_cap=32)
                self.assertEqual(result.returncode, expected_rc)
            noisy = audit_lock.run_command((str(script), "noise"), cwd=root, env=env, deadline=audit_lock._now() + 5, stdout_cap=128, stderr_cap=32)
            self.assertEqual(len(noisy.stdout), 128)
            self.assertTrue(noisy.stdout_truncated)
            timed = audit_lock.run_command((str(script), "timeout"), cwd=root, env=env, deadline=audit_lock._now() + 0.2, stdout_cap=128, stderr_cap=32)
            self.assertTrue(timed.timed_out)
            self.assertEqual(timed.returncode, 124)
            stalled = audit_lock.run_command((str(script), "stall"), cwd=root, env=env, deadline=audit_lock._now() + 0.2, stdout_cap=128, stderr_cap=32)
            self.assertTrue(stalled.timed_out)

            marker = root / "descendant.pid"
            heartbeat = root / "descendant.heartbeat"
            descendant = root / "descendant"
            descendant.write_text(textwrap.dedent(f"""
                #!{sys.executable}
                import os, signal, sys, time
                marker = sys.argv[1]
                heartbeat = sys.argv[2]
                child = os.fork()
                if child == 0:
                    signal.signal(signal.SIGTERM, signal.SIG_IGN)
                    while True:
                        open(heartbeat, 'w').write(str(time.monotonic()))
                        time.sleep(0.01)
                while not os.path.exists(heartbeat): time.sleep(0.01)
                open(marker, 'w').write(str(child))
                raise SystemExit(0)
            """).lstrip())
            descendant.chmod(descendant.stat().st_mode | stat.S_IXUSR)
            # Give process startup ample time. Once the fixture has actually
            # forked and published both readiness markers, advance the
            # injected clock past the fixed outer deadline. This exercises the
            # real timeout/owned-process-group cleanup path without a startup
            # race against a sub-second wall-clock cutoff.
            real_now = audit_lock._now
            outer_deadline = real_now() + 30.0
            cutoff_consumed = False

            def deterministic_now() -> float:
                nonlocal cutoff_consumed
                if not cutoff_consumed and marker.exists() and heartbeat.exists():
                    cutoff_consumed = True
                    return outer_deadline + 1.0
                return real_now()

            with mock.patch.object(audit_lock, "_now", side_effect=deterministic_now):
                held = audit_lock.run_command(
                    (str(descendant), str(marker), str(heartbeat)),
                    cwd=root,
                    env=env,
                    deadline=outer_deadline,
                    stdout_cap=128,
                    stderr_cap=32,
                )
            self.assertTrue(held.timed_out)
            self.assertTrue(cutoff_consumed)
            for _ in range(100):
                if marker.exists() and heartbeat.exists():
                    break
                time.sleep(0.01)
            self.assertTrue(marker.exists() and heartbeat.exists())
            first = heartbeat.read_text()
            time.sleep(0.2)
            self.assertEqual(heartbeat.read_text(), first, "exited parent left owned descendant running")

            injected_marker = root / "injected-descendant.pid"
            injected = root / "injected-descendant"
            injected.write_text(textwrap.dedent(f"""
                #!{sys.executable}
                import os, signal, sys, time
                marker = sys.argv[1]
                heartbeat = marker + '.heartbeat'
                child = os.fork()
                if child == 0:
                    signal.signal(signal.SIGTERM, signal.SIG_IGN)
                    while True:
                        open(heartbeat, 'w').write(str(time.monotonic()))
                        time.sleep(0.01)
                while not os.path.exists(heartbeat): time.sleep(0.01)
                open(marker, 'w').write(str(child))
                os.write(1, b'x')
                time.sleep(10)
            """).lstrip())
            injected.chmod(injected.stat().st_mode | stat.S_IXUSR)
            original_read = os.read
            original_popen = audit_lock.subprocess.Popen
            target_fds: set[int] = set()
            def capture_popen(*args, **kwargs):
                process = original_popen(*args, **kwargs)
                target_fds.update((process.stdout.fileno(), process.stderr.fileno()))
                return process
            def fail_read(fd, size):
                if fd not in target_fds:
                    return original_read(fd, size)
                for _ in range(200):
                    if injected_marker.exists():
                        break
                    time.sleep(0.01)
                raise OSError("injected pipe failure")
            with mock.patch.object(audit_lock.subprocess, "Popen", side_effect=capture_popen):
                with mock.patch.object(audit_lock.os, "read", side_effect=fail_read):
                    with self.assertRaises(OSError):
                        audit_lock.run_command((str(injected), str(injected_marker)), cwd=root, env=env, deadline=audit_lock._now() + 5, stdout_cap=128, stderr_cap=32)
            self.assertTrue(injected_marker.exists())
            heartbeat = injected_marker.with_name(injected_marker.name + ".heartbeat")
            self.assertTrue(heartbeat.exists())
            first = heartbeat.read_text()
            time.sleep(0.2)
            self.assertEqual(heartbeat.read_text(), first, "unexpected pipe failure left owned descendant running")

    def test_orchestration_retains_report_and_classifies_all_return_codes(self) -> None:
        for tool_rc, expected_rc, expected_status in ((0, 0, "PASS"), (1, 1, "FAIL"), (2, 2, "ERROR")):
            rc, evidence = self._run_orchestrated(tool_rc)
            self.assertEqual((rc, evidence["status"]), (expected_rc, expected_status))
            self.assertEqual(evidence["audit"]["report"]["lockfile"]["dependency-count"], 161)
            self.assertEqual(evidence["audit"]["command"]["argv"][0], "/trusted/cargo-audit")
        finding = good_report()
        finding["vulnerabilities"] = {"found": True, "count": 1, "list": [{}]}
        rc, evidence = self._run_orchestrated(0, finding)
        self.assertEqual((rc, evidence["status"], evidence["audit"]["classification"]), (2, "ERROR", "ERROR_RC0_FINDINGS"))
        rc, evidence = self._run_orchestrated(1)
        self.assertEqual((rc, evidence["audit"]["classification"]), (1, "FAIL_NO_FINDINGS_RETURN1"))
        rc, evidence = self._run_orchestrated(0, stdout=b"not-json")
        self.assertEqual((rc, evidence["audit"]["classification"]), (2, "ERROR_BAD_JSON"))
        rc, evidence = self._run_orchestrated(2, timed_out=True)
        self.assertEqual((rc, evidence["audit"]["classification"]), (2, "ERROR_TIMEOUT"))
        for output_flag, expected_field in (("stdout_truncated", "stdout_truncated"), ("stderr_truncated", "stderr_truncated")):
            kwargs = {output_flag: True}
            rc, evidence = self._run_orchestrated(0, **kwargs)
            self.assertEqual((rc, evidence["audit"]["classification"], evidence["audit"]["command"][expected_field]), (2, "ERROR_OUTPUT_LIMIT", True))

    def test_guard_and_child_environment_are_fail_closed(self) -> None:
        calls: list[str] = []
        with self.assertRaises(audit_lock.AuditError):
            audit_lock._host_guard({"GITHUB_ACTIONS": "true", "RUNNER_OS": "macOS", "RUNNER_ARCH": "ARM64"})
        child = audit_lock._child_env(Path("/tmp/cargo-home"), {"HOME": "/runner", "PATH": "/bin", "GITHUB_TOKEN": "secret", "GIT_CONFIG_GLOBAL": "evil"})
        self.assertEqual(child["HOME"], "/runner")
        self.assertNotIn("GITHUB_TOKEN", child)
        self.assertEqual(child["GIT_CONFIG_GLOBAL"], os.devnull)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "evidence.json"
            audit_lock._write_json(output, {"ok": True})
            with self.assertRaises(audit_lock.AuditError):
                audit_lock._write_json(output, {"overwrite": True})
            with self.assertRaises(audit_lock.AuditError):
                audit_lock._write_json(Path("relative.json"), {})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cwd, home, db, artifact = (root / name for name in ("cwd", "home", "db", "artifact"))
            cwd.mkdir(); home.mkdir(); db.mkdir(); artifact.mkdir()
            called = []
            def denied(_env):
                raise audit_lock.AuditError("host must be Linux x86_64")
            rc = audit_lock.run_audit(
                artifact_root=artifact, db=db, cwd=cwd, cargo_home=home, auditor="cargo-audit", repository="ayutaz/vokra", output=root / "guard.json",
                verifier=lambda *args, **kwargs: called.append("auditor") or {}, host_guard=denied,
            )
            self.assertEqual(rc, 2)
            self.assertEqual(called, [])

    def test_api_body_reader_uses_deadline_socket_timeout(self) -> None:
        class FakeSocket:
            timeout: float | None = None
            def settimeout(self, value: float) -> None:
                self.timeout = value

        fake_socket = FakeSocket()
        class FakeResponse:
            fp = type("FP", (), {"raw": type("Raw", (), {"_sock": fake_socket})()})()
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self, _size: int) -> bytes:
                time.sleep(2)
                return b"{}"
        class FakeOpener:
            def open(self, _request, timeout: float):
                self.timeout = timeout
                return FakeResponse()
        opener = FakeOpener()
        with mock.patch("audit_lock.urllib.request.build_opener", return_value=opener):
            started = time.monotonic()
            with self.assertRaises(audit_lock.AuditError):
                audit_lock._api_json("https://api.github.com/repos/ayutaz/vokra/actions/runs/1", "secret", audit_lock._now() + 0.12)
            self.assertLess(time.monotonic() - started, 0.5)
        self.assertIsNotNone(fake_socket.timeout)
        self.assertLessEqual(fake_socket.timeout or 1.0, 30.0)
        with mock.patch("audit_lock.signal.getitimer", return_value=(1.0, 0.0)):
            with self.assertRaises(audit_lock.AuditError):
                with audit_lock._absolute_alarm(audit_lock._now() + 1):
                    pass

    def test_auditor_version_is_exact_and_stream_limits_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            binary = root / "cargo-audit"
            binary.write_bytes(b"trusted binary")
            binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
            def runner_for(text: bytes, *, rc: int = 0, truncated: bool = False):
                return lambda *args, **kwargs: audit_lock.CommandResult(tuple(args[0]), rc, text, b"", truncated, False)
            env = {"PATH": str(root), "HOME": str(root / "home")}
            good = audit_lock.verify_auditor("cargo-audit", cwd=root, env=env, deadline=audit_lock._now() + 5, runner=runner_for(b"cargo-audit-audit 0.22.2\n"))
            self.assertEqual(good["cargo_audit_version"], "0.22.2")
            self.assertEqual(good["binary_path"], str(binary))
            for text, rc, truncated in ((b"cargo-audit-audit 0.22.20\n", 0, False), (b"cargo-audit 0.22.2\n", 0, False), (b"cargo-audit-audit 0.22.2-alpha\n", 0, False), (b"cargo-audit-audit 0.22.2+build\n", 0, False), (b"cargo-audit-audit 0.22.2\nextra\n", 0, False), (b"cargo-audit-audit 0.22.2\n", 1, False), (b"cargo-audit-audit 0.22.2\n", 0, True)):
                with self.assertRaises(audit_lock.AuditError) as caught:
                    audit_lock.verify_auditor("cargo-audit", cwd=root, env=env, deadline=audit_lock._now() + 5, runner=runner_for(text, rc=rc, truncated=truncated))
                self.assertEqual(caught.exception.version_evidence["stdout"], text.decode())
                self.assertEqual(caught.exception.version_evidence["stdout_truncated"], truncated)

    def test_version_mismatch_evidence_is_retained_by_orchestration(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cwd, home, db, artifact = (root / name for name in ("cwd", "home", "db", "artifact"))
            cwd.mkdir(); home.mkdir(); db.mkdir(); artifact.mkdir()
            raw = audit_lock.CommandResult(("/trusted/cargo-audit", "audit", "--version"), 1, b"cargo-audit-audit 0.22.20\n", b"bad", False, False)
            error = audit_lock.AuditError("cargo-audit version mismatch")
            setattr(error, "version_evidence", audit_lock._command_evidence(raw))
            def fail_verifier(*args, **kwargs):
                raise error
            with mock.patch.dict(os.environ, {"HOME": str(root), "GITHUB_ACTIONS": "true", "RUNNER_OS": "Linux", "RUNNER_ARCH": "X64"}, clear=False):
                rc = audit_lock.run_audit(artifact_root=artifact, db=db, cwd=cwd, cargo_home=home, auditor="cargo-audit", repository="ayutaz/vokra", output=root / "report.json", verifier=fail_verifier, host_guard=lambda _env: None)
            evidence = json.loads((root / "report.json").read_text())
            self.assertEqual(rc, 2)
            self.assertEqual(evidence["tool"]["version_check"]["stdout"], "cargo-audit-audit 0.22.20\n")

    def test_workflow_contract_has_reviewed_triggers_and_pins(self) -> None:
        root = Path(__file__).parents[3]
        workflow_dir = root / ".github/workflows"
        workflows = {
            name: (workflow_dir / name).read_text()
            for name in (
                "kyutai-candle-source-advisory.yml",
                "kyutai-candle-candidate-advisory.yml",
                "kyutai-candle-applied-advisory.yml",
            )
        }
        source = workflows["kyutai-candle-source-advisory.yml"]
        candidate = workflows["kyutai-candle-candidate-advisory.yml"]
        applied = workflows["kyutai-candle-applied-advisory.yml"]
        self.assertNotIn("pull_request:", source)
        self.assertNotIn("pull_request:", candidate)
        self.assertIn("pull_request:", applied)
        for workflow_name in workflows:
            self.assertIn(f'      - ".github/workflows/{workflow_name}"', applied)
        for text in workflows.values():
            self.assertIn("workflow_dispatch:", text)
            self.assertIn(audit_lock.DOWNLOAD_ACTION_COMMIT, text)
            self.assertIn(audit_lock.INSTALL_ACTION_COMMIT, text)
            self.assertIn(audit_lock.ADVISORY_DB_COMMIT, text)
            self.assertIn("fallback: none", text)
            self.assertIn("set -euo pipefail", text)
            self.assertIn("set +e", text)
            self.assertIn("rc=$?", text)
            self.assertIn('exit "$rc"', text)
            self.assertNotIn("continue-on-error:", text)
            self.assertNotIn("--ignore", text)
            for workflow_name in workflows:
                self.assertIn(f".github/workflows/{workflow_name}", text)
        self.assertIn('artifact-ids: "11951567227"', source)
        self.assertIn('artifact-ids: "11961100426"', candidate)
        self.assertIn('artifact-ids: "11985400172"', applied)
        self.assertIn("run-id: 38758745134", applied)
        self.assertIn("--profile applied", applied)
        self.assertNotIn("--profile applied", source)
        self.assertNotIn("--profile applied", candidate)
        self.assertNotIn("--token", audit_lock.main.__code__.co_names)
        for text in workflows.values():
            self.assertIn("permissions:\n  contents: read\n  actions: read", text)
            self.assertIn("sparse-checkout-cone-mode: false", text)
            self.assertIn("persist-credentials: false", text)
            self.assertIn("set -e\n          test -f", text)
            self.assertNotIn("cargo test", text)
            self.assertNotIn("cargo build", text)
            self.assertNotIn("cargo run", text)
            self.assertNotIn("cargo metadata", text)
            self.assertNotIn("uv sync", text)
            self.assertNotIn("huggingface", text.lower())
        for path in (
            ".github/workflows/kyutai-candle-source-advisory.yml",
            "tools/parity/kyutai_candle_reference/audit_lock.py",
            "tools/parity/kyutai_candle_reference/test_audit_lock.py",
        ):
            self.assertIn(path, source)


if __name__ == "__main__":
    # The repository's local-memory guard permits explicit stdlib self-tests;
    # this suite only exercises fake auditors and synthetic evidence.
    unittest.main(argv=[sys.argv[0], "-v"] if "--self-test" in sys.argv else None)
