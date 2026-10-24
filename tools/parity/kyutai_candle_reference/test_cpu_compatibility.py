from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import tempfile
import time
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SPEC = importlib.util.spec_from_file_location("cpu_compatibility", HERE / "check_cpu_compatibility.py")
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class CpuCompatibilityContractTests(unittest.TestCase):
    def test_application_contract_is_fixed_and_review_pending(self) -> None:
        self.assertEqual(len(CHECKER.APPLICATION_FILES), 8)
        self.assertEqual(CHECKER.APPLICATION_FILES["final.lock"], CHECKER.FINAL_LOCK_SHA256)
        self.assertEqual(CHECKER.APPLICATION_FILES["source_receipt.json"], CHECKER.SOURCE_RECEIPT_SHA256)
        self.assertEqual(CHECKER.APPLIED_RUN, "38758745134")
        self.assertEqual(CHECKER.APPLIED_ATTEMPT, "1")
        self.assertEqual(CHECKER.APPLIED_ARTIFACT, "kyutai-candle-paste-application-evidence-38758745134-1")

    def test_cpu_scope_contract_is_not_gpu_or_model_execution(self) -> None:
        self.assertEqual(CHECKER.TARGET, "x86_64-unknown-linux-gnu")
        self.assertEqual(CHECKER.ROOT_FEATURES["moshi"], [])
        source = (HERE / "check_cpu_compatibility.py").read_text(encoding="utf-8")
        self.assertIn('"model_execution": "NOT_AUTHORIZED"', source)
        self.assertIn('"status": "CPU_COMPATIBILITY_REVIEW_PENDING"', source)
        self.assertIn('VOKRA_CANDLE_REMOTE_LINUX', source)
        self.assertIn('sys.platform.startswith("linux")', source)
        self.assertNotIn("cargo run", source)
        self.assertNotIn("huggingface_hub", source.lower())
        self.assertTrue(all("@" in item for item in CHECKER.REQUIRED_BUILD_ARTIFACTS))
        self.assertIn("gemm-c32@0.17.1", CHECKER.REQUIRED_BUILD_ARTIFACTS)
        self.assertIn("gemm-c64@0.18.2", CHECKER.REQUIRED_BUILD_ARTIFACTS)
        self.assertIn("pastey@0.2.3", CHECKER.REQUIRED_BUILD_ARTIFACTS)

    def test_deadline_and_environment_are_fail_closed(self) -> None:
        with self.assertRaises(CHECKER.CompatibilityError):
            CHECKER._remaining(0.0)
        env = CHECKER._sanitized_env(cargo_home=Path("/tmp/cargo"), target_dir=Path("/tmp/target"), offline=True)
        self.assertEqual(env["CARGO_NET_OFFLINE"], "true")
        for key in ("RUSTFLAGS", "RUSTC_WRAPPER", "HF_TOKEN", "HUGGINGFACE_TOKEN", "VAST_API_KEY"):
            self.assertNotIn(key, env)

    def test_toolchain_binding_is_explicit_and_not_caller_selected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vokra-rustup-") as temporary:
            rustup_home = Path(temporary) / "rustup"
            rustup_home.mkdir()
            env = CHECKER._sanitized_env(cargo_home=Path(temporary) / "cargo", target_dir=Path(temporary) / "target", offline=True)
            env["RUSTUP_TOOLCHAIN"] = "poisoned"
            CHECKER._bind_trusted_toolchain(env, rustup_home=rustup_home)
            self.assertEqual(env["RUSTUP_HOME"], str(rustup_home))
            self.assertEqual(env["RUSTUP_TOOLCHAIN"], "1.99.0")

    def test_compiler_artifact_scope_is_complete(self) -> None:
        complete = {"stdout": "\n".join(json.dumps({"reason": "compiler-artifact", "package_id": item, "target": {"kind": CHECKER.REQUIRED_BUILD_KINDS[item]}}) for item in CHECKER.REQUIRED_BUILD_ARTIFACTS)}
        self.assertEqual(len(CHECKER._artifact_ids(complete)), len(CHECKER.REQUIRED_BUILD_ARTIFACTS))
        with self.assertRaises(CHECKER.CompatibilityError):
            CHECKER._artifact_ids({"stdout": json.dumps({"reason": "compiler-artifact", "package_id": "pastey@0.2.3", "target": {"kind": ["lib"]}})})

    def test_bounded_commands_fail_on_rc_truncation_and_timeout(self) -> None:
        env = CHECKER._sanitized_env(cargo_home=Path("/tmp/cargo"), target_dir=Path("/tmp/target"), offline=True)
        with self.assertRaises(CHECKER.BoundedCommandError) as failed:
            CHECKER._run_bounded(["/bin/sh", "-c", "exit 7"], cwd=HERE, env=env, deadline=time.monotonic() + 5)
        self.assertEqual(failed.exception.result["returncode"], 7)
        with self.assertRaises(CHECKER.BoundedCommandError) as truncated:
            CHECKER._run_bounded(["/usr/bin/yes", "compatibility"], cwd=HERE, env=env, deadline=time.monotonic() + 5, max_output=1024)
        self.assertTrue(truncated.exception.result["truncated"])
        with self.assertRaises(CHECKER.BoundedCommandError) as timed_out:
            CHECKER._run_bounded(["/bin/sh", "-c", "sleep 5"], cwd=HERE, env=env, deadline=time.monotonic() + 0.2)
        self.assertTrue(timed_out.exception.result["timed_out"])

    def test_cleanup_kills_term_ignoring_descendant_after_leader_exit(self) -> None:
        if not __import__("sys").platform.startswith("linux"):
            self.skipTest("process-group orphan semantics are validated on hosted Linux")
        env = CHECKER._sanitized_env(cargo_home=Path("/tmp/cargo"), target_dir=Path("/tmp/target"), offline=True)
        with tempfile.TemporaryDirectory(prefix="vokra-orphan-") as temporary:
            pid_file = Path(temporary) / "child.pid"
            child_pid = None
            try:
                child_script = f"trap '' TERM; echo $$ > {shlex.quote(str(pid_file))}; sleep 30"
                command = f"/bin/sh -c {shlex.quote(child_script)} & exit 0"
                with self.assertRaises(CHECKER.BoundedCommandError) as failed:
                    CHECKER._run_bounded(["/bin/sh", "-c", command], cwd=HERE, env=env, deadline=time.monotonic() + 0.25)
                self.assertTrue(failed.exception.result["timed_out"])
                child_pid = int(pid_file.read_text(encoding="utf-8"))
                for _ in range(20):
                    try:
                        os.kill(child_pid, 0)
                    except ProcessLookupError:
                        break
                    state_path = Path(f"/proc/{child_pid}/stat")
                    if state_path.is_file():
                        state = state_path.read_text(encoding="utf-8").split()[2]
                        if state == "Z":
                            break
                    time.sleep(0.05)
                else:
                    self.fail("TERM-ignoring descendant survived process-group cleanup")
            finally:
                if child_pid is not None:
                    try:
                        os.kill(child_pid, CHECKER.signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_selected_source_scope_ignores_operational_cache_but_binds_source(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vokra-source-scope-") as temporary:
            root = Path(temporary)
            cargo_home = root / "cargo-home"
            source = cargo_home / "registry" / "src" / "index-hash" / "tiny-1.0.0"
            source.mkdir(parents=True)
            (source / "Cargo.toml").write_text("[package]\nname='tiny'\nversion='1.0.0'\n", encoding="utf-8")
            (source / "src.rs").write_text("pub fn stable() {}\n", encoding="utf-8")
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({"metadata": {"packages": [{"name": "tiny", "version": "1.0.0", "source": "registry+https://example.invalid/index"}]}}), encoding="utf-8")
            before = CHECKER._selected_source_scope(metadata, cargo_home, deadline=time.monotonic() + 5)
            cache = cargo_home / "registry" / "cache"
            cache.mkdir(parents=True)
            (cache / "operational.lock").write_text("mutable", encoding="utf-8")
            self.assertEqual(before, CHECKER._selected_source_scope(metadata, cargo_home, deadline=time.monotonic() + 5))
            (source / "src.rs").write_text("pub fn changed() {}\n", encoding="utf-8")
            self.assertNotEqual(before, CHECKER._selected_source_scope(metadata, cargo_home, deadline=time.monotonic() + 5))

    def test_api_probe_binds_target_and_host_dependencies_and_propagates_failure(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vokra-api-probe-") as temporary:
            root = Path(temporary)
            target_dir = root / "target"
            target_deps = target_dir / CHECKER.TARGET / "debug" / "deps"
            host_deps = target_dir / "debug" / "deps"
            target_deps.mkdir(parents=True)
            host_deps.mkdir(parents=True)
            (target_deps / "libmoshi-test.rlib").write_bytes(b"moshi")
            (host_deps / "libbytemuck_derive-test.so").write_bytes(b"host")
            evidence_dir = root / "evidence"
            receipt = {"commands": []}
            captured: list[list[str]] = []

            def fake_stage(stage, *, receipt, evidence_dir, argv, cwd, env, deadline, max_output=2 * 1024 * 1024):
                self.assertEqual(stage, "api_probe")
                captured.append(argv)
                Path(argv[-1]).write_bytes(b"probe")
                return {"stdout": ""}

            with mock.patch.object(CHECKER, "_run_stage", side_effect=fake_stage):
                CHECKER._run_api_probe(root=root, target_dir=target_dir, deadline=time.monotonic() + 5, env={}, receipt=receipt, evidence_dir=evidence_dir)
            command = captured[0]
            self.assertIn(f"dependency={target_deps}", command)
            self.assertIn(f"dependency={host_deps}", command)
            self.assertIn("--target", command)
            self.assertIn(CHECKER.TARGET, command)

            def fail_stage(stage, *, receipt, evidence_dir, argv, cwd, env, deadline, max_output=2 * 1024 * 1024):
                raise CHECKER.BoundedCommandError("probe failure", {"argv": argv, "returncode": 1, "stdout": "", "stderr": "missing host dependency", "stdout_bytes": b"", "stderr_bytes": b"missing host dependency", "stdout_sha256": "", "stderr_sha256": "", "timed_out": False, "truncated": False})

            with mock.patch.object(CHECKER, "_run_stage", side_effect=fail_stage):
                with self.assertRaises(CHECKER.BoundedCommandError):
                    CHECKER._run_api_probe(root=root, target_dir=target_dir, deadline=time.monotonic() + 5, env={}, receipt=receipt, evidence_dir=evidence_dir)

    def test_mocked_orchestration_reaches_build_and_probe_and_fails_closed(self) -> None:
        proof = {
            "source_proof_pre": {"files_checked": 23, "unchanged_files": 21, "entries": [{}] * 23},
            "source_proof_post": {"files_checked": 23, "unchanged_files": 21, "entries": [{}] * 23},
            "member_premetadata_checks": [{"members_checked": 308, "non_manifest_members": 292}],
            "member_postchecks": [{"members_checked": 308, "non_manifest_members": 292}],
        }

        def run_fake(stage, *, receipt, evidence_dir, argv, cwd, env, deadline, max_output=2 * 1024 * 1024):
            stages.append(stage)
            if stage == "source_application":
                applied = Path(argv[3])
                workspace = Path(argv[4])
                applied.mkdir(parents=True)
                (workspace / "projection" / "moshi-core").mkdir(parents=True)
                (workspace / "projection" / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
                (workspace / "projection" / "moshi-core" / "Cargo.toml").write_text("[package]\nname='moshi'\nversion='0'\n", encoding="utf-8")
                (workspace / "vendor").mkdir()
                (workspace / "cargo-home" / "registry" / "src").mkdir(parents=True)
                (applied / "source_receipt.json").write_text("{}", encoding="utf-8")
                (applied / "application_receipt.json").write_text("{}", encoding="utf-8")
            if stage == "checkout_head":
                return {"stdout": os.environ["VOKRA_CANDLE_WORKFLOW_HEAD"]}
            if stage == "namespace_preflight":
                return {"stdout": "rustc 1.99.0\ncargo 1.99.0\n"}
            if stage == "cargo_build":
                return {"stdout": ""}
            if stage == "post_build_cpu_metadata":
                return {"stdout": "{}"}
            return {"stdout": ""}

        def fail_build(stage, *, receipt, evidence_dir, argv, cwd, env, deadline, max_output=2 * 1024 * 1024):
            if stage == "cargo_build":
                stages.append(stage)
                result = {
                    "argv": argv, "returncode": 1, "elapsed_seconds": 0.01,
                    "timed_out": False, "truncated": False,
                    "stdout_sha256": CHECKER.hashlib.sha256(b"").hexdigest(),
                    "stderr_sha256": CHECKER.hashlib.sha256(b"build failed").hexdigest(),
                    "stdout_bytes": b"", "stderr_bytes": b"build failed",
                    "stdout": "", "stderr": "build failed",
                }
                raise CHECKER.BoundedCommandError("mock build failure", result)
            return run_fake(stage=stage, receipt=receipt, evidence_dir=evidence_dir, argv=argv, cwd=cwd, env=env, deadline=deadline, max_output=max_output)

        def fake_json(path: Path):
            if path.name == "source_receipt.json":
                return {"files": {str(index): {} for index in range(23)}}
            return {}

        def fake_hash(path: Path) -> str:
            if path.name == "source_receipt.json":
                return CHECKER.SOURCE_RECEIPT_SHA256
            if path.name == "Cargo.toml":
                return CHECKER.ROOT_MANIFEST_SHA256
            return CHECKER.FINAL_LOCK_SHA256

        with tempfile.TemporaryDirectory(prefix="vokra-orchestration-") as temporary:
            output = Path(temporary) / "receipt.json"
            stages: list[str] = []
            patches = [
                mock.patch.object(CHECKER, "_verify_application", return_value=proof),
                mock.patch.object(CHECKER, "_verify_application_receipt", return_value=proof),
                mock.patch.object(CHECKER, "_verify_fixed_files"),
                mock.patch.object(CHECKER, "_bind_trusted_toolchain"),
                mock.patch.object(CHECKER, "_run_stage", side_effect=run_fake),
                mock.patch.object(CHECKER, "_json", side_effect=fake_json),
                mock.patch.object(CHECKER, "_bounded_read", return_value=b"{}"),
                mock.patch.object(CHECKER, "SOURCE_RECEIPT_SHA256", hashlib.sha256(b"{}").hexdigest()),
                mock.patch.object(CHECKER, "FINAL_LOCK_SHA256", hashlib.sha256(b"{}").hexdigest()),
                mock.patch.object(CHECKER, "_sha256", side_effect=fake_hash),
                mock.patch.object(CHECKER, "_validate_cpu_metadata", return_value={"packages": 105, "nodes": 105, "fingerprint": CHECKER.CPU_FINGERPRINT}),
                mock.patch.object(CHECKER, "_tree_fingerprint", return_value={"files": 1, "bytes": 1, "sha256": "tree"}),
                mock.patch.object(CHECKER, "_selected_source_scope", return_value={"registry_sources": {}}),
                mock.patch.object(CHECKER, "_metadata_package_identities", return_value={}),
                mock.patch.object(CHECKER, "_artifact_ids", return_value=list(CHECKER.REQUIRED_BUILD_ARTIFACTS)),
                mock.patch.object(CHECKER, "_run_api_probe", return_value={"stdout": ""}),
                mock.patch.object(CHECKER, "_evidence_total", return_value=0),
                mock.patch.object(CHECKER.sys, "platform", "linux"),
                mock.patch.object(CHECKER.os, "readlink", return_value="net:[host]"),
                mock.patch.dict(os.environ, {"VOKRA_CANDLE_REMOTE_LINUX": "1", "VOKRA_CANDLE_WORKFLOW_HEAD": "a" * 40}, clear=False),
            ]
            for patcher in patches:
                patcher.start()
            try:
                result = CHECKER.check(Path(temporary) / "application", output, 30)
                self.assertEqual(result, 0, output.read_text(encoding="utf-8") if output.exists() else "no receipt")
                self.assertEqual(stages, ["checkout_head", "source_application", "namespace_preflight", "cargo_build", "post_build_cpu_metadata"])
            finally:
                for patcher in reversed(patches):
                    patcher.stop()

            stages.clear()
            patches[4] = mock.patch.object(CHECKER, "_run_stage", side_effect=fail_build)
            for patcher in patches:
                patcher.start()
            try:
                self.assertEqual(CHECKER.check(Path(temporary) / "application", output, 30), 2)
                self.assertIn("cargo_build", stages, output.read_text(encoding="utf-8") if output.exists() else "no receipt")
                self.assertNotIn("api_probe", stages)
            finally:
                for patcher in reversed(patches):
                    patcher.stop()

    def test_non_linux_or_missing_marker_writes_failure_evidence(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vokra-cpu-gate-") as temporary:
            with mock.patch.object(CHECKER.sys, "platform", "linux"), mock.patch.dict(os.environ, {"VOKRA_CANDLE_REMOTE_LINUX": "0"}, clear=False):
                output = Path(temporary) / "linux-marker-failure.json"
                self.assertEqual(CHECKER.check(Path(temporary) / "missing", output, 1), 2)
                self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], "CPU_COMPATIBILITY_FAILED")
            with mock.patch.object(CHECKER.sys, "platform", "darwin"), mock.patch.dict(os.environ, {"VOKRA_CANDLE_REMOTE_LINUX": "1"}, clear=False):
                output = Path(temporary) / "non-linux-failure.json"
                self.assertEqual(CHECKER.check(Path(temporary) / "missing", output, 1), 2)
                self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], "CPU_COMPATIBILITY_FAILED")

    def test_public_check_rejects_invalid_deadline(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vokra-cpu-deadline-") as temporary:
            output = Path(temporary) / "failure.json"
            self.assertEqual(CHECKER.check(Path(temporary) / "missing", output, CHECKER.MAX_DEADLINE + 1), 2)
            self.assertIn("deadline_seconds", output.read_text(encoding="utf-8"))

    def test_workflow_is_one_shot_and_builds_not_checks(self) -> None:
        workflow = (REPO / ".github/workflows/kyutai-candle-cpu-compatibility.yml").read_text(encoding="utf-8")
        self.assertIn("types: [labeled]", workflow)
        self.assertIn("candle-cpu-compatibility-confirm-once-20261016", workflow)
        self.assertNotIn("candle-cpu-compatibility-once-20261016", workflow)
        self.assertNotIn("workflow_dispatch", workflow)
        self.assertIn("timeout-minutes: 20", workflow)
        self.assertIn("runs-on: ubuntu-22.04", workflow)
        self.assertIn("actions: read", workflow)
        self.assertIn("ref: ${{ github.event.pull_request.head.sha }}", workflow)
        self.assertIn("actions/download-artifact@9000827ccba6bdab643e8b6fd33ac0654aef8333", workflow)
        self.assertIn("source-application:", workflow)
        self.assertIn("caller: cpu-compatibility", workflow)
        self.assertIn("artifact-ids: ${{ needs.source-application.outputs.primary_artifact_id }}", workflow)
        self.assertIn("artifact-ids: ${{ needs.source-application.outputs.envelope_artifact_id }}", workflow)
        self.assertIn("digest-mismatch: error", workflow)
        self.assertIn("--envelope", workflow)
        self.assertNotIn("artifact-ids: 11985400172", workflow)
        self.assertNotIn("run-id: 38758745134", workflow)
        self.assertIn("--deadline-seconds 900", workflow)
        self.assertIn("if-no-files-found: error", workflow)
        self.assertIn("cargo build", (HERE / "check_cpu_compatibility.py").read_text(encoding="utf-8"))
        self.assertIn("--lib", (HERE / "check_cpu_compatibility.py").read_text(encoding="utf-8"))
        self.assertIn("--offline", (HERE / "check_cpu_compatibility.py").read_text(encoding="utf-8"))
        self.assertIn("--no-default-features", (HERE / "check_cpu_compatibility.py").read_text(encoding="utf-8"))

    def test_frozen_helpers_are_explicitly_bound(self) -> None:
        self.assertEqual(len(CHECKER.FROZEN_FILES), 4)
        for relative in CHECKER.FROZEN_FILES:
            self.assertTrue(relative.endswith(".py"))
        source = (HERE / "check_cpu_compatibility.py").read_text(encoding="utf-8")
        self.assertIn("apply_paste_overlay.py", source)
        self.assertIn("prepare_paste_overlay.py", source)

    def test_metadata_shape_rejects_forbidden_gpu_name(self) -> None:
        metadata = {
            "metadata": {
                "packages": [{"name": "cudarc", "version": "1", "source": None}],
                "resolve": {"nodes": []},
            }
        }
        path = HERE / ".cpu-compatibility-negative.json"
        try:
            path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(CHECKER.CompatibilityError):
                CHECKER._validate_cpu_metadata(path)
        finally:
            path.unlink(missing_ok=True)

    def test_authenticated_cpu_fixture_positive_and_mutation_negative(self) -> None:
        configured = os.environ.get("VOKRA_CPU_APPLICATION_ROOT")
        if not configured:
            self.skipTest("hosted authenticated application fixture not provided")
        original = Path(configured)
        with tempfile.TemporaryDirectory(prefix="vokra-cpu-fixture-") as temporary:
            copy = Path(temporary) / "application"
            copy.mkdir()
            for name in CHECKER.APPLICATION_FILES:
                shutil.copy2(original / name, copy / name)
            CHECKER._verify_application(copy)
            baseline = CHECKER._validate_cpu_metadata(copy / "baseline_cpu_metadata.json", expected_fingerprint=None, role_counts=(104, 0, 1))
            self.assertEqual(baseline["packages"], 105)
            summary = CHECKER._validate_cpu_metadata(copy / "overlay_cpu_metadata.json")
            self.assertEqual(summary["packages"], 105)
            (copy / "extra.txt").write_text("unexpected", encoding="utf-8")
            with self.assertRaises(CHECKER.CompatibilityError):
                CHECKER._verify_application(copy)
            (copy / "extra.txt").unlink()
            mutated = json.loads((copy / "overlay_cpu_metadata.json").read_text(encoding="utf-8"))
            mutated["metadata"]["resolve"]["nodes"][0]["features"].append("mutated-feature")
            (copy / "overlay_cpu_metadata.json").write_text(json.dumps(mutated), encoding="utf-8")
            with self.assertRaises(CHECKER.CompatibilityError):
                CHECKER._verify_application(copy)
            with self.assertRaises(CHECKER.CompatibilityError):
                CHECKER._validate_cpu_metadata(copy / "overlay_cpu_metadata.json")

            original_metadata = json.loads((original / "overlay_cpu_metadata.json").read_text(encoding="utf-8"))
            for mutation in ("identity", "source_path", "edge"):
                candidate = json.loads(json.dumps(original_metadata))
                if mutation == "identity":
                    candidate["metadata"]["packages"][0]["name"] = "forged-package"
                elif mutation == "source_path":
                    candidate["metadata"]["packages"][0]["manifest_path"] = "/forged/aho-corasick-1.1.3/Cargo.toml"
                else:
                    candidate["metadata"]["resolve"]["nodes"][0]["deps"][0]["name"] = "forged-edge"
                path = copy / f"{mutation}.json"
                path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.assertRaises(CHECKER.CompatibilityError):
                    CHECKER._validate_cpu_metadata(path)

    def test_namespace_and_cleanup_are_explicit(self) -> None:
        source = (HERE / "check_cpu_compatibility.py").read_text(encoding="utf-8")
        self.assertIn("NAMESPACE_PREFIX", source)
        self.assertIn("--map-root-user", source)
        self.assertIn("--kill-child=KILL", source)
        self.assertIn("setpriv", source)
        self.assertIn("signal.SIGINT", source)
        self.assertIn("signal.SIGTERM", source)
        self.assertIn("_reap_process", source)
        self.assertIn("start_new_session=True", source)
        self.assertIn("os.killpg", source)


if __name__ == "__main__":
    if "--self-test" in __import__("sys").argv:
        unittest.main(argv=[__import__("sys").argv[0]])
    else:
        unittest.main()
