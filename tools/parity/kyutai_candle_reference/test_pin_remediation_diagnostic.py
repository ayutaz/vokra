from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("pin_remediation_diagnostic", HERE / "pin_remediation_diagnostic.py")
assert SPEC is not None and SPEC.loader is not None
DIAG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAG)
COLLECTOR_SPEC = importlib.util.spec_from_file_location("collect_metadata_for_pin_tests", HERE / "collect_metadata.py")
assert COLLECTOR_SPEC is not None and COLLECTOR_SPEC.loader is not None
LEAF = importlib.util.module_from_spec(COLLECTOR_SPEC)
COLLECTOR_SPEC.loader.exec_module(LEAF)


REGISTRY = "registry+https://github.com/rust-lang/crates.io-index"
SOURCE_FILES = {
    "Cargo.toml": b"[workspace]\nmembers = [\"moshi-core\"]\n",
    "Cargo.lock": b"version = 4\n",
    "LICENSE": b"MIT\n",
    "README.md": b"source\n",
    "moshi-core/Cargo.toml": b"[package]\nname = \"moshi\"\nversion = \"0.6.4\"\n",
    "moshi-core/src/lib.rs": b"pub fn synthetic() {}\n",
}
OLD_VERSIONS = {name: old for (name, _new), old in (
    (("crossbeam-epoch", "0.9.20"), "0.9.18"),
    (("anyhow", "1.0.103"), "1.0.98"),
    (("rand", "0.9.3"), "0.9.0"),
    (("backtrace", "0.3.76"), "0.3.75"),
)}
MEMMAP_OLD = "0.9.5"
MEMMAP_PATCHED = "0.9.11"


def _lock(versions: dict[str, str]) -> bytes:
    rows = ["version = 4", "", "[[package]]", 'name = "moshi"', 'version = "0.6.4"', ""]
    for name, version in versions.items():
        rows.extend([
            "[[package]]",
            f'name = "{name}"',
            f'version = "{version}"',
            f'source = "{REGISTRY}"',
            f'checksum = "{"a" * 64}"',
            "",
        ])
    return ("\n".join(rows)).encode()


def _metadata(versions: dict[str, str], *, cpu: bool) -> dict[str, object]:
    moshi_id = "path+file:///projection/moshi-core#moshi@0.6.4"
    packages: list[dict[str, object]] = [{
        "id": moshi_id, "name": "moshi", "version": "0.6.4", "source": None,
        "manifest_path": "/projection/moshi-core/Cargo.toml", "features": {}, "dependencies": [],
    }]
    if not cpu:
        for name, version in versions.items():
            packages.append({
                "id": f"{REGISTRY}#{name}@{version}", "name": name, "version": version,
                "source": REGISTRY, "manifest_path": f"/{name}/Cargo.toml", "features": {}, "dependencies": [],
            })
    nodes = [{"id": moshi_id, "features": [], "deps": []}]
    if not cpu:
        nodes.extend({"id": package["id"], "features": [], "deps": []} for package in packages[1:])
    return {"packages": packages, "workspace_members": [moshi_id], "resolve": {"root": None, "nodes": nodes}}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class PinRemediationTests(unittest.TestCase):
    def test_workflow_is_sparse_bounded_and_has_exact_evidence_paths(self) -> None:
        workflow = (HERE.parents[2] / ".github/workflows/kyutai-candle-pin-remediation.yml").read_text(encoding="utf-8")
        self.assertIn("runs-on: ubuntu-latest", workflow)
        self.assertIn("timeout-minutes: 15", workflow)
        self.assertIn("persist-credentials: false", workflow)
        self.assertIn("sparse-checkout-cone-mode: false", workflow)
        self.assertNotIn("filter:", workflow)
        self.assertNotIn("license_gate_manifest.json", workflow)
        self.assertNotIn("silero-vad-v5.gguf", workflow)
        self.assertIn("cancel-in-progress: true", workflow)
        self.assertIn("--deadline-seconds 480", workflow)
        self.assertIn("--capture-source", workflow)
        self.assertEqual(workflow.count("Capture and run the bounded four-pin source-only diagnostic"), 1)
        self.assertNotIn("continue-on-error", workflow)
        self.assertNotIn("cache:", workflow)
        check = workflow[workflow.index("Check the exact bounded evidence set"):workflow.index("Upload exact bounded evidence")]
        upload = workflow[workflow.index("Upload exact bounded evidence"):]
        expected = [
            "source_receipt.json", "source/rust/Cargo.lock", "base/Cargo.lock", "patched/Cargo.lock",
            "initial_metadata.json", "whole_lock_metadata.json", "metadata.json",
            "candidate.lock", "candidate_cpu_metadata.json", "candidate_whole_metadata.json", "pin_diagnostic.json",
            "baseline_cpu_metadata.json", "baseline_whole_metadata.json",
        ]
        for relative in expected:
            self.assertIn(relative, check)
            self.assertIn(relative, upload)
        self.assertNotIn("*", check + upload)

    def test_allowlist_requires_exact_precise_updates_and_metadata(self) -> None:
        for name, version in DIAG.PINS:
            DIAG._validate_argv(["cargo", "update", "-p", name, "--precise", version], "cargo")
        DIAG._validate_argv(["cargo", *DIAG.CPU_METADATA_ARGS], "cargo")
        DIAG._validate_argv(["cargo", *DIAG.WHOLE_METADATA_ARGS], "cargo")
        for argv in (
            ["cargo", "update"],
            ["cargo", "update", "-p", "rand"],
            ["cargo", "update", "-p", "rand", "--precise", "0.9.3", "--offline"],
            ["rustc", "update", "-p", "rand", "--precise", "0.9.3"],
        ):
            with self.subTest(argv=argv):
                    with self.assertRaises(DIAG.DiagnosticError):
                        DIAG._validate_argv(argv, "cargo")

    def test_unlocked_whole_capture_preserves_two_mib_bound_and_exact_argv(self) -> None:
        class FakeCollector:
            def run_bounded_subprocess(self, argv: list[str], **kwargs: object) -> dict[str, object]:
                del kwargs
                self.argv = argv
                return {"argv": argv, "returncode": 0, "stdout": "x" * 728260, "stderr": ""}

        collector = FakeCollector()
        result = DIAG._run_candidate(
            collector,
            ["cargo", *DIAG.WHOLE_METADATA_ARGS],
            cwd=Path("."),
            env={},
            deadline=1.0,
            cargo_name="cargo",
        )
        self.assertEqual(tuple(collector.argv[1:]), DIAG.WHOLE_METADATA_ARGS)
        self.assertEqual(len(result["stdout"]), 728260)
        self.assertLessEqual(len(result["stdout"].encode()), 2 * 1024 * 1024)
        self.assertEqual(DIAG.WHOLE_METADATA_ARGS, LEAF.WHOLE_LOCK_METADATA_ARGS)
        self.assertGreaterEqual(LEAF.MAX_WHOLE_LOCK_CAPTURE_BYTES, 728260)

    def test_capture_error_does_not_fabricate_empty_stream_evidence(self) -> None:
        argv = ["cargo", *DIAG.WHOLE_METADATA_ARGS]
        evidence = DIAG._command_evidence(
            argv,
            {"returncode": None, "capture_error": "Cargo command exceeded shared deadline"},
            3,
        )
        self.assertIsNone(evidence["stdout"])
        self.assertIsNone(evidence["stderr"])
        self.assertEqual(evidence["stream_status"], "UNAVAILABLE_CAPTURE_ERROR")

    def test_lock_delta_keeps_full_added_removed_records_and_root_fields(self) -> None:
        before_key = ("base", "1.0.0", REGISTRY)
        after_key = ("added", "2.0.0", REGISTRY)
        before = {before_key: {"name": "base", "version": "1.0.0", "source": REGISTRY, "checksum": "a" * 64, "dependencies": []}}
        after = {after_key: {"name": "added", "version": "2.0.0", "source": REGISTRY, "checksum": "b" * 64, "dependencies": ["base 1.0.0"]}}
        delta = DIAG._ordered_lock_delta(
            LEAF,
            before,
            after,
            before_document={"version": 4, "metadata": {"profile": "base"}, "package": list(before.values())},
            after_document={"version": 5, "metadata": {"profile": "candidate"}, "package": list(after.values())},
        )
        self.assertEqual(delta["added_records"][0]["record"]["version"], "2.0.0")
        self.assertEqual(delta["removed_records"][0]["record"]["checksum"], "a" * 64)
        self.assertTrue(delta["root_fields"]["changed"])
        self.assertEqual(delta["root_fields"]["after"]["version"], 5)

    def test_lock_delta_reports_unrelated_identity_and_order(self) -> None:
        before = {("moshi", "0.6.4", None): {"name": "moshi", "version": "0.6.4", "dependencies": []}}
        after = dict(before)
        after[("unrelated", "1.0.0", REGISTRY)] = {"name": "unrelated", "version": "1.0.0", "source": REGISTRY, "dependencies": []}
        delta = DIAG._ordered_lock_delta(LEAF, before, after)
        self.assertEqual(len(delta["added"]), 1)
        self.assertFalse(delta["retained_order_changed"])

    def test_lock_delta_reports_target_scalar_mutation_without_identity_change(self) -> None:
        key = ("backtrace", "0.3.76", REGISTRY)
        before = {key: {"name": key[0], "version": key[1], "source": key[2], "checksum": "a" * 64, "dependencies": [], "unknown_root": {"flag": False}}}
        after = {key: {**before[key], "checksum": "b" * 64, "unknown_root": {"flag": True}}}
        delta = DIAG._ordered_lock_delta(LEAF, before, after)
        self.assertEqual(len(delta["retained_record_changes"]), 1)
        self.assertEqual(delta["retained_record_changes"][0]["package"].split("@", 1)[0], "backtrace")
        self.assertEqual(delta["retained_record_changes"][0]["after"]["unknown_root"]["flag"], True)

    def test_graph_delta_records_edges_features_and_order(self) -> None:
        before = {"package_order": ["a", "b"], "features": {"a": ["default"]}, "declarations": {"a": ["b"]}, "edges": [{"source": "a", "name": "b", "target": "b", "dep_kinds": [{"kind": None, "target": None}]}]}
        after = {"package_order": ["b", "a"], "features": {"a": ["default", "std"]}, "declarations": {"a": ["b", "c"]}, "edges": []}
        delta = DIAG._graph_delta(before, after)
        self.assertTrue(delta["package_order_changed"])
        self.assertEqual(delta["feature_changes"][0]["after"], ["default", "std"])
        self.assertEqual(delta["edges_removed"][0]["source"], "a")

    def test_metadata_snapshot_preserves_full_declaration_and_package_record(self) -> None:
        metadata = _metadata({"memmap2": MEMMAP_PATCHED}, cpu=False)
        package = metadata["packages"][1]
        package["dependencies"] = [{
            "name": "serde",
            "req": "^1.0.0",
            "kind": "normal",
            "rename": None,
            "optional": True,
            "uses_default_features": False,
            "features": ["derive"],
            "target": "cfg(unix)",
            "registry": REGISTRY,
        }]
        snapshot = DIAG._metadata_snapshot(LEAF, metadata, whole=True)
        declaration = snapshot["declarations"][DIAG._lock_label(LEAF, ("memmap2", MEMMAP_PATCHED, REGISTRY))][0]
        self.assertEqual(declaration["req"], "^1.0.0")
        self.assertEqual(declaration["target"], "cfg(unix)")
        self.assertEqual(declaration["features"], ["derive"])
        self.assertTrue(any(key.startswith("memmap2@") for key in snapshot["package_records"]))

    def test_recovered_metadata_writer_keeps_full_native_json_under_cap(self) -> None:
        fixture_root = Path("/private/tmp/vokra-candle-hosted-semantic-metadata-20261015.1tKZ0u")
        if not fixture_root.is_dir():
            self.skipTest("hosted semantic metadata fixture is not present")
        with tempfile.TemporaryDirectory(prefix="vokra-pin-size-test-", dir=str(Path(tempfile.gettempdir()).resolve())) as temp:
            for filename, whole in (("initial_metadata.json", False), ("whole_lock_metadata.json", True)):
                wrapper = json.loads((fixture_root / filename).read_text(encoding="utf-8"))
                metadata = wrapper["metadata"]
                snapshot = DIAG._metadata_snapshot(LEAF, metadata, whole=whole)
                output = Path(temp) / filename
                DIAG._write_graph_evidence(
                    LEAF,
                    output,
                    status="SIZE_REGRESSION",
                    source_commit=LEAF.SOURCE_COMMIT,
                    target=None if whole else DIAG.TARGET,
                    metadata=metadata,
                    snapshot=snapshot,
                )
                self.assertLessEqual(output.stat().st_size, DIAG.MAX_JSON_BYTES)
                self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["metadata"], metadata)

    def test_portable_large_package_field_proves_no_graph_duplication(self) -> None:
        metadata = _metadata({}, cpu=False)
        metadata["packages"][0]["description"] = "x" * 1_700_000
        snapshot = DIAG._metadata_snapshot(LEAF, metadata, whole=True)
        with tempfile.TemporaryDirectory(prefix="vokra-pin-portable-size-", dir=str(Path(tempfile.gettempdir()).resolve())) as temp:
            output = Path(temp) / "large.json"
            DIAG._write_graph_evidence(
                LEAF,
                output,
                status="SIZE_REGRESSION",
                source_commit=LEAF.SOURCE_COMMIT,
                target=None,
                metadata=metadata,
                snapshot=snapshot,
            )
            self.assertLessEqual(output.stat().st_size, DIAG.MAX_JSON_BYTES)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["metadata"], metadata)

    def test_authentication_rejects_mutated_source_with_rewritten_receipt(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vokra-pin-auth-test-", dir=str(Path(tempfile.gettempdir()).resolve())) as temp:
            root = Path(temp)
            old_files, old_urls = LEAF.SOURCE_FILES, LEAF.SOURCE_URLS
            LEAF.SOURCE_FILES = {name: (LEAF.git_blob_sha1(original if name == "Cargo.lock" else data), len(original if name == "Cargo.lock" else data)) for name, data in SOURCE_FILES.items() for original in [_lock({**OLD_VERSIONS, "memmap2": MEMMAP_OLD})]}
            LEAF.SOURCE_URLS = {name: f"https://raw.invalid/{name}" for name in SOURCE_FILES}
            try:
                packet, *_ = self._make_packet(root / "packet", OLD_VERSIONS)
                source = packet / "source" / "rust" / "README.md"
                source.write_bytes(b"modify\n")
                receipt_path = packet / "source_receipt.json"
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                entry = receipt["files"]["README.md"]
                entry["bytes"] = len(source.read_bytes())
                entry["sha256"] = _sha(source.read_bytes())
                entry["git_blob_sha1"] = LEAF.git_blob_sha1(source.read_bytes())
                receipt["total_bytes"] = sum(item["bytes"] for item in receipt["files"].values())
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                with self.assertRaisesRegex(DIAG.DiagnosticError, "frozen Git blob mismatch"):
                    DIAG._authenticate_packet(packet, LEAF)
            finally:
                LEAF.SOURCE_FILES, LEAF.SOURCE_URLS = old_files, old_urls

    def test_rehashed_forged_base_is_rejected_by_frozen_base_gate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vokra-pin-base-test-", dir=str(Path(tempfile.gettempdir()).resolve())) as temp:
            root = Path(temp)
            old_files, old_urls = LEAF.SOURCE_FILES, LEAF.SOURCE_URLS
            LEAF.SOURCE_FILES = {name: (LEAF.git_blob_sha1(original if name == "Cargo.lock" else data), len(original if name == "Cargo.lock" else data)) for name, data in SOURCE_FILES.items() for original in [_lock({**OLD_VERSIONS, "memmap2": MEMMAP_OLD})]}
            LEAF.SOURCE_URLS = {name: f"https://raw.invalid/{name}" for name in SOURCE_FILES}
            try:
                packet, _original, _patched, initial, whole = self._make_packet(root / "packet", OLD_VERSIONS)
                forged = _lock({**OLD_VERSIONS, "memmap2": MEMMAP_OLD, "unrelated": "1.0.0"})
                base_path = packet / "base" / "Cargo.lock"
                base_path.write_bytes(forged)
                metadata_path = packet / "metadata.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata["lock_receipts"]["projected_base"].update(bytes=len(forged), sha256=_sha(forged))
                metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                authenticated = DIAG._authenticate_packet(packet, LEAF)
                self.assertEqual(authenticated["lock_bytes"]["projected_base"], forged)
                with self.assertRaises(LEAF.CaptureError):
                    LEAF.validate_base_lock(
                        packet / "source" / "rust" / "Cargo.lock",
                        base_path,
                        cpu_metadata=initial["metadata"],
                        whole_metadata=whole["metadata"],
                    )
            finally:
                LEAF.SOURCE_FILES, LEAF.SOURCE_URLS = old_files, old_urls

    def test_capture_and_run_keeps_one_absolute_deadline_after_capture(self) -> None:
        class FakeCollector:
            class CaptureError(Exception):
                pass

            def capture(self, root: Path, *, deadline_seconds: float, platform_name: str) -> None:
                self.root = root
                self.capture_deadline_seconds = deadline_seconds

        fake = FakeCollector()
        observed: dict[str, object] = {}
        with mock.patch.object(DIAG, "_load_collector", return_value=fake), mock.patch.object(DIAG, "time") as clock, mock.patch.dict(os.environ, {"VOKRA_CANDLE_REMOTE_LINUX": "1"}, clear=False):
            clock.monotonic.side_effect = [100.0, 100.0, 101.0]
            with mock.patch.object(DIAG, "run", side_effect=lambda *args, **kwargs: observed.update(kwargs) or {"status": "test"}):
                result = DIAG.capture_and_run(Path("packet"), Path("output"), Path("workspace"), deadline_seconds=120, platform_name="Linux")
        self.assertEqual(result["status"], "test")
        self.assertEqual(observed["deadline_mono"], 220.0)
        self.assertEqual(fake.capture_deadline_seconds, 118.0)

    def _make_packet(self, root: Path, versions: dict[str, str]) -> tuple[Path, bytes, bytes, dict[str, object], dict[str, object]]:
        source_root = root / "source" / "rust"
        source_root.mkdir(parents=True)
        original_versions = {**versions, "memmap2": MEMMAP_OLD}
        patched_versions = {**versions, "memmap2": MEMMAP_PATCHED}
        original_lock = _lock(original_versions)
        patched_lock = _lock(patched_versions)
        for relative, data in SOURCE_FILES.items():
            path = source_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(original_lock if relative == "Cargo.lock" else data)
        leaf_files = {name: (LEAF.git_blob_sha1(data), len(data)) for name, data in SOURCE_FILES.items()}
        receipt_files = {
            name: {"url": f"https://raw.invalid/{name}", "bytes": len(data), "git_blob_sha1": LEAF.git_blob_sha1(data), "sha256": _sha(data)}
            for name, data in ((name, original_lock if name == "Cargo.lock" else data) for name, data in SOURCE_FILES.items())
        }
        receipt = {"source_commit": LEAF.SOURCE_COMMIT, "base_url": LEAF.RAW_BASE, "provenance": "official-raw-fixed-git-blob-authenticated", "files": receipt_files, "total_bytes": sum(entry["bytes"] for entry in receipt_files.values())}
        (root / "source_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
        base = root / "base" / "Cargo.lock"
        patched = root / "patched" / "Cargo.lock"
        projection = root / "projection" / "Cargo.lock"
        base.parent.mkdir(parents=True, exist_ok=True)
        base.write_bytes(original_lock)
        for path in (patched, projection):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(patched_lock)
        initial = {"status": "INITIAL_METADATA_UNREVIEWED", "source_commit": LEAF.SOURCE_COMMIT, "metadata": _metadata(versions, cpu=True)}
        whole = {"status": "WHOLE_LOCK_METADATA_UNREVIEWED", "source_commit": LEAF.SOURCE_COMMIT, "target": None, "platform_filter": None, "metadata": _metadata(original_versions, cpu=False)}
        (root / "initial_metadata.json").write_text(json.dumps(initial), encoding="utf-8")
        (root / "whole_lock_metadata.json").write_text(json.dumps(whole), encoding="utf-8")
        lock_receipts = {}
        for name, path in (("original", source_root / "Cargo.lock"), ("projected_base", base), ("patched", patched), ("final_projection", projection)):
            data = path.read_bytes()
            lock_receipts[name] = {"path": str(path), "bytes": len(data), "sha256": _sha(data)}
        packet = {"source_commit": LEAF.SOURCE_COMMIT, "lock_receipts": lock_receipts}
        (root / "metadata.json").write_text(json.dumps(packet), encoding="utf-8")
        return root, original_lock, patched_lock, initial, whole

    def test_run_records_four_pin_order_and_keeps_paste_unresolved(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vokra-pin-test-", dir=str(Path(tempfile.gettempdir()).resolve())) as temp:
            root = Path(temp)
            old_files, old_urls = LEAF.SOURCE_FILES, LEAF.SOURCE_URLS
            LEAF.SOURCE_FILES = {name: (LEAF.git_blob_sha1(original if name == "Cargo.lock" else data), len(original if name == "Cargo.lock" else data)) for name, data in SOURCE_FILES.items() for original in [_lock({**OLD_VERSIONS, "memmap2": MEMMAP_OLD})]}
            LEAF.SOURCE_URLS = {name: f"https://raw.invalid/{name}" for name in SOURCE_FILES}
            try:
                packet, base_lock, candidate_lock, initial, whole = self._make_packet(root / "packet", OLD_VERSIONS)
                output, workspace = root / "output", root / "workspace"
                candidate_cpu = json.dumps(initial["metadata"])
                baseline_whole_value = _metadata({**OLD_VERSIONS, "memmap2": MEMMAP_PATCHED}, cpu=False)
                candidate_whole_value = _metadata({name: desired for name, desired in DIAG.PINS} | {"memmap2": MEMMAP_PATCHED}, cpu=False)
                calls: list[list[str]] = []

                def fake_run(argv: list[str], *, cwd: Path, env: dict[str, str], deadline: float, validate: bool) -> dict[str, object]:
                    del env, deadline, validate
                    calls.append(argv)
                    if argv[1] == "update":
                        update_index = len([call for call in calls if len(call) > 1 and call[1] == "update"]) - 1
                        current = dict(OLD_VERSIONS)
                        current["memmap2"] = MEMMAP_PATCHED
                        for prior_name, prior_version in DIAG.PINS[: update_index + 1]:
                            current[prior_name] = prior_version
                        (cwd / "Cargo.lock").write_bytes(_lock(current))
                    if tuple(argv[1:]) == DIAG.CPU_METADATA_ARGS:
                        return {"argv": argv, "returncode": 0, "stdout": candidate_cpu, "stderr": ""}
                    if tuple(argv[1:]) == DIAG.WHOLE_METADATA_ARGS:
                        update_count = len([call for call in calls if len(call) > 1 and call[1] == "update"])
                        value = candidate_whole_value if update_count == len(DIAG.PINS) else baseline_whole_value
                        return {"argv": argv, "returncode": 0, "stdout": json.dumps(value), "stderr": ""}
                    return {"argv": argv, "returncode": 0, "stdout": "", "stderr": ""}

                with mock.patch.object(DIAG, "_load_collector", return_value=LEAF), mock.patch.object(LEAF, "validate_base_lock", return_value={"status": "synthetic"}), mock.patch.object(LEAF, "validate_patch_lock"), mock.patch.dict(os.environ, {"VOKRA_CANDLE_REMOTE_LINUX": "1"}, clear=False), mock.patch.object(DIAG.platform, "system", return_value="Linux"), mock.patch.object(DIAG.shutil, "which", return_value="/synthetic/cargo"), mock.patch.object(LEAF, "_toolchain_probe", return_value={"rustc_version": "1.99.0", "host": DIAG.TARGET}), mock.patch.object(LEAF, "run_bounded_subprocess", side_effect=fake_run):
                    result = DIAG.run(packet, output, workspace, platform_name="Linux")
                self.assertEqual(result["status"], "PIN_REMEDIATION_SOURCE_ONLY_REVIEW_PENDING")
                update_calls = [call for call in calls if len(call) > 1 and call[1] == "update"]
                self.assertEqual([call[3] for call in update_calls], [name for name, _version in DIAG.PINS])
                self.assertEqual(result["unresolved"], {"package": "paste", "status": "EXPLICITLY_UNRESOLVED"})
                self.assertTrue((output / "candidate_cpu_metadata.json").is_file())
                self.assertTrue((output / "candidate_whole_metadata.json").is_file())
            finally:
                LEAF.SOURCE_FILES, LEAF.SOURCE_URLS = old_files, old_urls

    def test_whole_capture_lock_mutation_blocks_before_any_update_for_both_rcs(self) -> None:
        for whole_rc in (0, 1):
            with self.subTest(whole_rc=whole_rc), tempfile.TemporaryDirectory(prefix="vokra-pin-whole-mutation-", dir=str(Path(tempfile.gettempdir()).resolve())) as temp:
                root = Path(temp)
                old_files, old_urls = LEAF.SOURCE_FILES, LEAF.SOURCE_URLS
                LEAF.SOURCE_FILES = {name: (LEAF.git_blob_sha1(original if name == "Cargo.lock" else data), len(original if name == "Cargo.lock" else data)) for name, data in SOURCE_FILES.items() for original in [_lock({**OLD_VERSIONS, "memmap2": MEMMAP_OLD})]}
                LEAF.SOURCE_URLS = {name: f"https://raw.invalid/{name}" for name in SOURCE_FILES}
                try:
                    packet, _base, _candidate, _initial, _whole = self._make_packet(root / "packet", OLD_VERSIONS)
                    baseline_cpu = json.dumps(_metadata(OLD_VERSIONS, cpu=True))
                    baseline_whole = json.dumps(_metadata({**OLD_VERSIONS, "memmap2": MEMMAP_PATCHED}, cpu=False))
                    calls: list[list[str]] = []

                    def fake_run(argv: list[str], *, cwd: Path, env: dict[str, str], deadline: float, validate: bool) -> dict[str, object]:
                        del env, deadline, validate
                        calls.append(argv)
                        if tuple(argv[1:]) == DIAG.CPU_METADATA_ARGS:
                            return {"argv": argv, "returncode": 0, "stdout": baseline_cpu, "stderr": ""}
                        if tuple(argv[1:]) == DIAG.WHOLE_METADATA_ARGS:
                            (cwd / "Cargo.lock").write_bytes((cwd / "Cargo.lock").read_bytes() + b"\n")
                            return {"argv": argv, "returncode": whole_rc, "stdout": baseline_whole, "stderr": "blocked" if whole_rc else ""}
                        return {"argv": argv, "returncode": 0, "stdout": "", "stderr": ""}

                    with mock.patch.object(DIAG, "_load_collector", return_value=LEAF), mock.patch.object(LEAF, "validate_base_lock", return_value={"status": "synthetic"}), mock.patch.object(LEAF, "validate_patch_lock"), mock.patch.dict(os.environ, {"VOKRA_CANDLE_REMOTE_LINUX": "1"}, clear=False), mock.patch.object(DIAG.platform, "system", return_value="Linux"), mock.patch.object(DIAG.shutil, "which", return_value="/synthetic/cargo"), mock.patch.object(LEAF, "_toolchain_probe", return_value={"rustc_version": "1.99.0", "host": DIAG.TARGET}), mock.patch.object(LEAF, "run_bounded_subprocess", side_effect=fake_run):
                        with self.assertRaises(DIAG.DiagnosticError):
                            DIAG.run(packet, root / "output", root / "workspace", platform_name="Linux")
                    failure = json.loads((root / "output" / "pin_diagnostic.json").read_text(encoding="utf-8"))
                    self.assertEqual(len(calls), 2)
                    self.assertEqual(failure["commands"][-1]["returncode"], whole_rc)
                    self.assertIn("Cargo.lock changed", failure["error"])
                finally:
                    LEAF.SOURCE_FILES, LEAF.SOURCE_URLS = old_files, old_urls

    def test_failure_retains_bounded_blocked_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vokra-pin-fail-test-", dir=str(Path(tempfile.gettempdir()).resolve())) as temp:
            root = Path(temp)
            old_files, old_urls = LEAF.SOURCE_FILES, LEAF.SOURCE_URLS
            LEAF.SOURCE_FILES = {name: (LEAF.git_blob_sha1(original if name == "Cargo.lock" else data), len(original if name == "Cargo.lock" else data)) for name, data in SOURCE_FILES.items() for original in [_lock({**OLD_VERSIONS, "memmap2": MEMMAP_OLD})]}
            LEAF.SOURCE_URLS = {name: f"https://raw.invalid/{name}" for name in SOURCE_FILES}
            try:
                packet, _base, _candidate, _initial, _whole = self._make_packet(root / "packet", OLD_VERSIONS)
                baseline_cpu = json.dumps(_metadata(OLD_VERSIONS, cpu=True))
                baseline_whole = json.dumps(_metadata({**OLD_VERSIONS, "memmap2": MEMMAP_PATCHED}, cpu=False))
                def fake_run(argv: list[str], **kwargs: object) -> dict[str, object]:
                    del kwargs
                    if argv[1] == "update" and argv[2] == "-p" and argv[3] == "anyhow":
                        return {"argv": argv, "returncode": 1, "stdout": "", "stderr": "blocked"}
                    if tuple(argv[1:]) == DIAG.CPU_METADATA_ARGS:
                        return {"argv": argv, "returncode": 0, "stdout": baseline_cpu, "stderr": ""}
                    if tuple(argv[1:]) == DIAG.WHOLE_METADATA_ARGS:
                        return {"argv": argv, "returncode": 0, "stdout": baseline_whole, "stderr": ""}
                    return {"argv": argv, "returncode": 0, "stdout": "", "stderr": ""}
                with mock.patch.object(DIAG, "_load_collector", return_value=LEAF), mock.patch.object(LEAF, "validate_base_lock", return_value={"status": "synthetic"}), mock.patch.object(LEAF, "validate_patch_lock"), mock.patch.dict(os.environ, {"VOKRA_CANDLE_REMOTE_LINUX": "1"}, clear=False), mock.patch.object(DIAG.platform, "system", return_value="Linux"), mock.patch.object(DIAG.shutil, "which", return_value="/synthetic/cargo"), mock.patch.object(LEAF, "_toolchain_probe", return_value={"rustc_version": "1.99.0", "host": DIAG.TARGET}), mock.patch.object(LEAF, "run_bounded_subprocess", side_effect=fake_run):
                    with self.assertRaises(DIAG.DiagnosticError):
                        DIAG.run(packet, root / "output", root / "workspace", platform_name="Linux")
                failure = json.loads((root / "output" / "pin_diagnostic.json").read_text(encoding="utf-8"))
                self.assertEqual(failure["status"], "PIN_REMEDIATION_BLOCKED")
                self.assertEqual(failure["completed_commands"], 4)
                self.assertIsNone(failure["approval"])
                self.assertEqual(failure["commands"][-1]["returncode"], 1)
                self.assertEqual(failure["commands"][-1]["stderr"]["text"], "blocked")
            finally:
                LEAF.SOURCE_FILES, LEAF.SOURCE_URLS = old_files, old_urls


if __name__ == "__main__":
    unittest.main()
