#!/usr/bin/env python3
"""Model-free contract tests for the VAST Moshi metadata binder.

The production-path tests mock only the authenticated source auditor, tool
identity and Cargo process. They never import Moshi, resolve a package, or
touch a checkpoint.
"""

from __future__ import annotations

import importlib.util
import contextlib
import io
import json
import copy
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("mimi_rust_core_metadata_binding", HERE / "mimi_rust_core_metadata_binding.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def valid_metadata() -> tuple[dict, dict]:
    root = "path+file:///moshi-core#moshi@0.6.4"
    candle = "registry+https://github.com/rust-lang/crates.io-index#candle-core@0.9.1"
    metadata = {
        "packages": [
            {"id": root, "name": "moshi", "version": "0.6.4", "source": None, "manifest_path": "/tmp/moshi-core/Cargo.toml"},
            {"id": candle, "name": "candle-core", "version": "0.9.1", "source": "registry+https://github.com/rust-lang/crates.io-index", "manifest_path": "/tmp/candle-core/Cargo.toml"},
        ],
        "workspace_members": [root],
        "resolve": {"root": root, "nodes": [
            {"id": root, "features": [], "deps": [{"pkg": candle, "dep_kinds": [{"kind": None, "target": None}]}]},
            {"id": candle, "features": [], "deps": []},
        ]},
    }
    lock = {"package": [
        {"name": "moshi", "version": "0.6.4", "source": None},
        {"name": "candle-core", "version": "0.9.1", "source": "registry+https://github.com/rust-lang/crates.io-index", "checksum": "a" * 64},
    ]}
    return metadata, lock


def tool_fixture(root: Path) -> dict:
    cargo = root / "cargo-real"; rustc = root / "rustc-real"
    cargo.write_bytes(b"cargo fixture"); rustc.write_bytes(b"rustc fixture")
    row = lambda path, version: {"launcher": str(path), "launcher_resolved": str(path), "rustup": str(path), "rustup_which": str(path), "resolved": str(path), "sha256": "a" * 64, "bytes": path.stat().st_size, "version": version}
    return {"cargo": row(cargo, "cargo 1 fixture"), "rustc": row(rustc, "rustc 1 fixture")}


class MetadataBindingTests(unittest.TestCase):
    def test_metadata_requires_exact_root_target_and_preserves_dep_kinds(self) -> None:
        metadata, lock = valid_metadata()
        result = MODULE.inspect_metadata(metadata, lock, authenticated_target=MODULE.TARGET)
        self.assertEqual(result["root"], metadata["workspace_members"][0])
        self.assertEqual(result["activation"], "OPEN_REACHABILITY_ONLY_TARGET_DEP_KINDS_NOT_BOUND")
        self.assertEqual(result["authenticated_target"], MODULE.TARGET)
        self.assertEqual(result["active_reachable"][0]["dependencies"][0]["dep_kinds"][0]["kind"], None)
        metadata["target_filter"] = "aarch64-unknown-linux-gnu"
        with self.assertRaisesRegex(MODULE.BindingError, "authenticated target"):
            MODULE.inspect_metadata(metadata, lock, authenticated_target="aarch64-unknown-linux-gnu")

    def test_metadata_rejects_wrong_root_lock_checksum_duplicates_and_dangling(self) -> None:
        metadata, lock = valid_metadata(); metadata["resolve"]["root"] = "wrong"
        with self.assertRaisesRegex(MODULE.BindingError, "resolve.root"):
            MODULE.inspect_metadata(metadata, lock, authenticated_target=MODULE.TARGET)
        metadata, lock = valid_metadata(); metadata["resolve"]["nodes"][0]["deps"][0]["pkg"] = "missing"
        with self.assertRaisesRegex(MODULE.BindingError, "dangling dependency edge"):
            MODULE.inspect_metadata(metadata, lock, authenticated_target=MODULE.TARGET)
        metadata, lock = valid_metadata(); lock["package"][1]["checksum"] = "bad"
        with self.assertRaisesRegex(MODULE.BindingError, "checksum"):
            MODULE.inspect_metadata(metadata, lock, authenticated_target=MODULE.TARGET)
        metadata, lock = valid_metadata(); lock["package"].append(dict(lock["package"][1]))
        with self.assertRaisesRegex(MODULE.BindingError, "duplicate Cargo.lock"):
            MODULE.inspect_metadata(metadata, lock)
        metadata, lock = valid_metadata(); lock["package"] = [lock["package"][1]]
        with self.assertRaisesRegex(MODULE.BindingError, "root package"):
            MODULE.inspect_metadata(metadata, lock)

    def test_native_marker_remains_blocked(self) -> None:
        metadata, lock = valid_metadata(); metadata["resolve"]["nodes"][1]["features"] = ["metal"]
        result = MODULE.inspect_metadata(metadata, lock, authenticated_target=MODULE.TARGET)
        self.assertEqual(result["status"], "BLOCKED_NATIVE_MARKER")

    def test_bounded_command_closes_streams_on_overflow_and_timeout(self) -> None:
        real_popen = MODULE.subprocess.Popen; captured = []
        def capture(*args, **kwargs):
            process = real_popen(*args, **kwargs); captured.append(process); return process
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            with self.assertRaises(MODULE.BindingError):
                with mock.patch.object(MODULE.subprocess, "Popen", capture): MODULE.command_bytes([sys.executable, "-S", "-c", "import sys; sys.stdout.write('x' * 4096)"], limit=64)
            with self.assertRaises(MODULE.BindingError):
                with mock.patch.object(MODULE.subprocess, "Popen", capture): MODULE.command_bytes([sys.executable, "-S", "-c", "import time; time.sleep(1)"], limit=1024, timeout=0.05)
        self.assertEqual(len(captured), 2)
        for process in captured:
            self.assertIsNotNone(process.poll()); self.assertTrue(process.stdout.closed); self.assertTrue(process.stderr.closed)
        self.assertFalse(any(issubclass(w.category, ResourceWarning) for w in caught))

    def _source_stub(self, source_root: Path, audit_dir: Path) -> None:
        derived = audit_dir / "derived-workspace"; derived.mkdir(parents=True)
        (derived / "Cargo.lock").write_text("""version = 3

[[package]]
name = "moshi"
version = "0.6.4"

[[package]]
name = "candle-core"
version = "0.9.1"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
""", encoding="utf-8")
        (derived / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
        manifest_sha = MODULE.sha256(b"[workspace]\n")
        lock_sha = MODULE.sha256((derived / "Cargo.lock").read_bytes())
        inventory = {
            "status": "STATIC_SOURCE_INVENTORY",
            "source": {"before": {"revision": MODULE.PINNED_REV, "origin": "official", "clean": True, "status_bytes": 0}, "after": {"revision": MODULE.PINNED_REV, "origin": "official", "clean": True, "status_bytes": 0}, "revision": MODULE.PINNED_REV, "origin": "official", "clean": True, "rows": []},
            "lock": {"sha256": MODULE.EXPECTED_LOCK_SHA256},
            "workspace": {"path": str(derived), "derived_manifest_sha256": manifest_sha, "copied_lock": {"sha256": lock_sha}, "copied_source_paths": [], "copied_source_sha256": {}},
        }
        inventory["inventory_payload_sha256"] = MODULE._inventory_payload_sha(inventory)
        (audit_dir / "source-inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    def test_production_bind_uses_authenticated_absolute_cargo_and_retains_raw_packet(self) -> None:
        metadata, _lock = valid_metadata(); raw = (json.dumps(metadata) + "\n").encode()
        with tempfile.TemporaryDirectory(prefix="mimi-binding-") as temp:
            root = Path(temp); source = root / "source"; source.mkdir(); (source / "rust").mkdir(); (source / "rust/Cargo.lock").write_text("lock", encoding="utf-8")
            output = root / "packet"; tools = tool_fixture(root); calls = []
            def command(argv, **kwargs):
                calls.append((argv, kwargs)); return 0, raw, b""
            with mock.patch.object(MODULE, "_source_readback", side_effect=[{"facts": {"revision": MODULE.PINNED_REV, "origin": "official", "clean": True, "status_bytes": 0}, "lock_sha256": MODULE.EXPECTED_LOCK_SHA256, "lock_bytes": 4, "selected_sha256": {}}] * 2):
                result = MODULE.bind(source, output, source_audit=self._source_stub, tools_fn=lambda: tools, command_fn=command)
            expected_argv = [tools["cargo"]["resolved"], *MODULE.METADATA_ARGS, "--manifest-path", str(output / "source-audit/derived-workspace/Cargo.toml")]
            self.assertEqual(result, 0); self.assertEqual(calls[0][0], expected_argv); self.assertEqual(calls[0][1]["env"]["CARGO_NET_OFFLINE"], "true"); self.assertEqual(calls[0][1]["env"]["CARGO_TERM_COLOR"], "never"); self.assertTrue(calls[0][1]["env"]["PATH"].startswith(str(Path(tools["rustc"]["resolved"]).parent)))
            self.assertEqual((output / "metadata-command.stdout").read_bytes(), raw); self.assertTrue((output / "metadata.json").exists()); self.assertTrue((output / "metadata-binding.json").exists())

    def test_unbound_tools_prevent_metadata_and_source_failure_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mimi-binding-") as temp:
            root = Path(temp); source = root / "source"; source.mkdir(); (source / "rust").mkdir(); (source / "rust/Cargo.lock").write_text("lock", encoding="utf-8"); output = root / "packet"; called = []
            def no_command(*args, **kwargs): called.append(True); raise AssertionError("Cargo must not run")
            def no_tools(): raise MODULE.BindingError("unbound toolchain")
            readback = {"facts": {"revision": MODULE.PINNED_REV, "origin": "official", "clean": True, "status_bytes": 0}, "lock_sha256": MODULE.EXPECTED_LOCK_SHA256, "lock_bytes": 4, "selected_sha256": {}}
            with mock.patch.object(MODULE, "_source_readback", return_value=readback):
                result = MODULE.bind(source, output, source_audit=self._source_stub, tools_fn=no_tools, command_fn=no_command)
            self.assertEqual(result, 2); self.assertFalse(called); self.assertEqual((output / "metadata-command.exit").read_text(), "NOT_RUN\n")
            output2 = root / "packet2"
            (source / "rust/Cargo.lock").write_text("lock", encoding="utf-8")
            result = MODULE.bind(source, output2, source_audit=lambda *_: (_ for _ in ()).throw(MODULE.BindingError("source identity")), tools_fn=no_tools, command_fn=no_command)
            self.assertEqual(result, 2); self.assertFalse(called)

    def test_source_or_lock_mutation_blocks_after_command(self) -> None:
        metadata, _lock = valid_metadata(); raw = (json.dumps(metadata) + "\n").encode()
        with tempfile.TemporaryDirectory(prefix="mimi-binding-") as temp:
            root = Path(temp); source = root / "source"; source.mkdir(); (source / "rust").mkdir(); (source / "rust/Cargo.lock").write_text("lock", encoding="utf-8"); output = root / "packet"; tools = tool_fixture(root)
            before = {"facts": {"revision": MODULE.PINNED_REV, "origin": "official", "clean": True, "status_bytes": 0}, "lock_sha256": MODULE.EXPECTED_LOCK_SHA256, "lock_bytes": 1, "selected_sha256": {}}; after = dict(before); after["lock_sha256"] = "b"
            with mock.patch.object(MODULE, "_source_readback", side_effect=[before, after]):
                result = MODULE.bind(source, output, source_audit=self._source_stub, tools_fn=lambda: tools, command_fn=lambda *a, **k: (0, raw, b""))
            self.assertEqual(result, 2); packet = json.loads((output / "metadata-binding.json").read_text()); self.assertEqual(packet["status"], "BLOCKED"); self.assertIn("source lock", packet["error"])

    def test_production_nonzero_retains_raw_command_and_execution_state(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mimi-binding-") as temp:
            root = Path(temp); source = root / "source"; source.mkdir(); (source / "rust").mkdir(); (source / "rust/Cargo.lock").write_text("lock", encoding="utf-8"); output = root / "packet"; tools = tool_fixture(root)
            readback = {"facts": {"revision": MODULE.PINNED_REV, "origin": "official", "clean": True, "status_bytes": 0}, "lock_sha256": MODULE.EXPECTED_LOCK_SHA256, "lock_bytes": 4, "selected_sha256": {}}
            with mock.patch.object(MODULE, "_source_readback", return_value=readback):
                result = MODULE.bind(source, output, source_audit=self._source_stub, tools_fn=lambda: tools, command_fn=lambda *a, **k: (7, b"partial-json", b"metadata failed"))
            self.assertEqual(result, 2)
            packet = json.loads((output / "metadata-binding.json").read_text())
            self.assertEqual(packet["execution"]["cargo"], {"attempted": True, "completed": True, "exit": 7})
            self.assertEqual((output / "metadata-command.stdout").read_bytes(), b"partial-json")
            self.assertEqual((output / "metadata-command.stderr").read_bytes(), b"metadata failed")
            self.assertEqual((output / "metadata-command.exit").read_text(), "7\n")

    def test_inventory_digest_tamper_prevents_cargo(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mimi-binding-") as temp:
            root = Path(temp); source = root / "source"; source.mkdir(); output = root / "packet"; called = []
            def tampered_audit(source_root, audit_dir):
                self._source_stub(source_root, audit_dir)
                inventory_path = audit_dir / "source-inventory.json"; inventory = json.loads(inventory_path.read_text()); inventory["inventory_payload_sha256"] = "0" * 64; inventory_path.write_text(json.dumps(inventory, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            result = MODULE.bind(source, output, source_audit=tampered_audit, tools_fn=lambda: (_ for _ in ()).throw(AssertionError("tool lookup after digest tamper")), command_fn=lambda *a, **k: called.append(True))
            packet = json.loads((output / "metadata-binding.json").read_text())
            self.assertEqual(result, 2); self.assertFalse(called); self.assertIn("payload digest", packet["error"])

    def test_tool_identity_mutation_blocks_after_real_command_boundary(self) -> None:
        metadata, _lock = valid_metadata(); raw = (json.dumps(metadata) + "\n").encode()
        with tempfile.TemporaryDirectory(prefix="mimi-binding-") as temp:
            root = Path(temp); source = root / "source"; source.mkdir(); (source / "rust").mkdir(); (source / "rust/Cargo.lock").write_text("lock", encoding="utf-8"); output = root / "packet"; tools = tool_fixture(root); changed = copy.deepcopy(tools); changed["cargo"]["sha256"] = "changed"
            readback = {"facts": {"revision": MODULE.PINNED_REV, "origin": "official", "clean": True, "status_bytes": 0}, "lock_sha256": MODULE.EXPECTED_LOCK_SHA256, "lock_bytes": 4, "selected_sha256": {}}
            with mock.patch.object(MODULE, "_source_readback", return_value=readback):
                result = MODULE.bind(source, output, source_audit=self._source_stub, tools_fn=mock.Mock(side_effect=[tools, changed]), command_fn=lambda *a, **k: (0, raw, b""))
            packet = json.loads((output / "metadata-binding.json").read_text())
            self.assertEqual(result, 2); self.assertIn("toolchain changed", packet["error"]); self.assertTrue(packet["execution"]["cargo"]["attempted"])

    def test_symlink_parent_cli_is_blocked_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mimi-binding-") as temp:
            root = Path(temp); source = root / "source"; source.mkdir(); real = root / "real"; real.mkdir(); link = root / "link"; link.symlink_to(real, target_is_directory=True)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                result = MODULE.main(["--source-root", str(source), "--output", str(link / "packet")])
            self.assertEqual(result, 2); self.assertIn("BLOCKED:", stderr.getvalue()); self.assertIn("output parent is symlink", stderr.getvalue()); self.assertNotIn("Traceback", stderr.getvalue())

    def test_source_auditor_exception_and_packet_symlink_are_blocked(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mimi-binding-") as temp:
            root = Path(temp); source = root / "source"; source.mkdir(); output = root / "packet"
            result = MODULE.bind(source, output, source_audit=lambda *_: (_ for _ in ()).throw(MODULE.SOURCE_AUDITOR.AuditError("source audit failed")))
            self.assertEqual(result, 2)
            packet = json.loads((output / "metadata-binding.json").read_text()); self.assertEqual(packet["status"], "BLOCKED")
            packet_root = root / "packet-files"; packet_root.mkdir(); (packet_root / "real").write_text("x", encoding="utf-8"); (packet_root / "link").symlink_to(packet_root / "real")
            with self.assertRaises(MODULE.BindingError): MODULE._packet_files(packet_root)

    def test_derived_workspace_mutation_blocks_after_command(self) -> None:
        metadata, _lock = valid_metadata(); raw = (json.dumps(metadata) + "\n").encode()
        with tempfile.TemporaryDirectory(prefix="mimi-binding-") as temp:
            root = Path(temp); source = root / "source"; source.mkdir(); (source / "rust").mkdir(); (source / "rust/Cargo.lock").write_text("lock", encoding="utf-8"); output = root / "packet"; tools = tool_fixture(root)
            source_readback = {"facts": {"revision": MODULE.PINNED_REV, "origin": "official", "clean": True, "status_bytes": 0}, "lock_sha256": MODULE.EXPECTED_LOCK_SHA256, "lock_bytes": 4, "selected_sha256": {}}
            calls = []
            def derived_readback(path, workspace=None):
                value = {"manifest_sha256": MODULE.sha256((path / "Cargo.toml").read_bytes()), "lock_sha256": MODULE.sha256((path / "Cargo.lock").read_bytes()), "lock_bytes": (path / "Cargo.lock").stat().st_size, "copied_source_sha256": {}}
                if calls: value["manifest_sha256"] = "changed"
                calls.append(True)
                return value
            with mock.patch.object(MODULE, "_source_readback", return_value=source_readback), mock.patch.object(MODULE, "_derived_readback", side_effect=derived_readback):
                result = MODULE.bind(source, output, source_audit=self._source_stub, tools_fn=lambda: tools, command_fn=lambda *a, **k: (0, raw, b""))
            packet = json.loads((output / "metadata-binding.json").read_text())
            self.assertEqual(result, 2); self.assertIn("derived Cargo.toml", packet["error"])

    def test_native_block_and_source_before_drift_are_not_promoted(self) -> None:
        metadata, _lock = valid_metadata(); metadata["resolve"]["nodes"][1]["features"] = ["metal"]; raw = (json.dumps(metadata) + "\n").encode()
        with tempfile.TemporaryDirectory(prefix="mimi-binding-") as temp:
            root = Path(temp); source = root / "source"; source.mkdir(); (source / "rust").mkdir(); (source / "rust/Cargo.lock").write_text("lock", encoding="utf-8"); tools = tool_fixture(root)
            output = root / "native-packet"; readback = {"facts": {"revision": MODULE.PINNED_REV, "origin": "official", "clean": True, "status_bytes": 0}, "lock_sha256": MODULE.EXPECTED_LOCK_SHA256, "lock_bytes": 4, "selected_sha256": {}}
            with mock.patch.object(MODULE, "_source_readback", return_value=readback):
                result = MODULE.bind(source, output, source_audit=self._source_stub, tools_fn=lambda: tools, command_fn=lambda *a, **k: (0, raw, b""))
            packet = json.loads((output / "metadata-binding.json").read_text())
            self.assertEqual(result, 2); self.assertEqual(packet["metadata"]["status"], "BLOCKED_NATIVE_MARKER"); self.assertEqual(packet["native"], "BLOCKED_NATIVE_MARKER")
            drift = root / "drift-packet"; called = []
            bad = dict(readback); bad["facts"] = dict(readback["facts"]); bad["facts"]["revision"] = "wrong"
            with mock.patch.object(MODULE, "_source_readback", return_value=bad):
                result = MODULE.bind(source, drift, source_audit=self._source_stub, tools_fn=lambda: (_ for _ in ()).throw(AssertionError("tool lookup after source drift")), command_fn=lambda *a, **k: called.append(True))
            drift_packet = json.loads((drift / "metadata-binding.json").read_text())
            self.assertEqual(result, 2); self.assertFalse(called); self.assertIn("source facts", drift_packet["error"])

    def test_copied_member_mutation_blocks_production_path(self) -> None:
        metadata, _lock = valid_metadata(); raw = (json.dumps(metadata) + "\n").encode()
        with tempfile.TemporaryDirectory(prefix="mimi-binding-") as temp:
            root = Path(temp); source = root / "source"; source.mkdir(); (source / "rust").mkdir(); (source / "rust/Cargo.lock").write_text("lock", encoding="utf-8"); output = root / "packet"; tools = tool_fixture(root)
            def audit_with_member(source_root, audit_dir):
                self._source_stub(source_root, audit_dir); derived = audit_dir / "derived-workspace"; (derived / "moshi-core").mkdir(); (derived / "moshi-core/Cargo.toml").write_text("[package]\n", encoding="utf-8")
                inventory_path = audit_dir / "source-inventory.json"; inventory = json.loads(inventory_path.read_text()); inventory["workspace"]["copied_source_paths"] = ["rust/moshi-core/Cargo.toml"]; inventory["workspace"]["copied_source_sha256"] = {"rust/moshi-core/Cargo.toml": MODULE.sha256(b"[package]\n")}; inventory["inventory_payload_sha256"] = MODULE._inventory_payload_sha(inventory); inventory_path.write_text(json.dumps(inventory, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            source_readback = {"facts": {"revision": MODULE.PINNED_REV, "origin": "official", "clean": True, "status_bytes": 0}, "lock_sha256": MODULE.EXPECTED_LOCK_SHA256, "lock_bytes": 4, "selected_sha256": {}}
            original_derived = MODULE._derived_readback; calls = []
            def derived_read(path, workspace=None):
                value = original_derived(path, workspace)
                if calls: value["copied_source_sha256"] = {"rust/moshi-core/Cargo.toml": "changed"}
                calls.append(True); return value
            with mock.patch.object(MODULE, "_source_readback", return_value=source_readback), mock.patch.object(MODULE, "_derived_readback", side_effect=derived_read):
                result = MODULE.bind(source, output, source_audit=audit_with_member, tools_fn=lambda: tools, command_fn=lambda *a, **k: (0, raw, b""))
            packet = json.loads((output / "metadata-binding.json").read_text())
            self.assertEqual(result, 2); self.assertIn("derived copied source", packet["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
