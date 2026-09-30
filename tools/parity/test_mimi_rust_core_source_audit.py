#!/usr/bin/env python3
"""Stdlib-only synthetic tests for the source/closure audit contract."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("mimi_rust_core_source_audit", HERE / "mimi_rust_core_source_audit.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RustCoreAuditTests(unittest.TestCase):
    def test_raw_path_rejects_traversal_and_noncanonical_forms(self) -> None:
        for value in ("../Cargo.toml", "rust/../Cargo.toml", "./rust/Cargo.toml", "rust//Cargo.toml", "rust\\Cargo.toml"):
            with self.subTest(value=value):
                with self.assertRaises(MODULE.AuditError):
                    MODULE._safe_rel(value)
        self.assertEqual(MODULE._safe_rel("rust/moshi-core/src/mimi.rs"), "rust/moshi-core/src/mimi.rs")

    def test_nofollow_regular_read_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mimi-rust-audit-") as temp:
            root = Path(temp)
            target = root / "target"
            target.write_bytes(b"source")
            link = root / "link"
            link.symlink_to(target)
            with self.assertRaises(MODULE.AuditError):
                MODULE._bounded_bytes(link)

    def test_bounded_read_rejects_oversized_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mimi-rust-audit-") as temp:
            path = Path(temp) / "large"
            path.write_bytes(b"x" * (MODULE.MAX_READ + 1))
            with self.assertRaises(MODULE.AuditError):
                MODULE._bounded_bytes(path)

    def test_git_blob_id_is_content_bound(self) -> None:
        self.assertNotEqual(MODULE._git_blob_id(b"a"), MODULE._git_blob_id(b"b"))

    def test_noclobber_output_preserves_existing_bytes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mimi-rust-audit-") as temp:
            path = Path(temp) / "manifest.json"
            MODULE._write_noclobber(path, b"first")
            with self.assertRaises(MODULE.AuditError):
                MODULE._write_noclobber(path, b"second")
            self.assertEqual(path.read_bytes(), b"first")

    def test_metadata_resolve_checks_lock_and_blocks_native_features(self) -> None:
        lock = {"package": [
            {"name": "moshi", "version": "0.6.4", "source": None},
            {"name": "candle-core", "version": "0.9.1", "source": "registry+https://github.com/rust-lang/crates.io-index", "checksum": "a" * 64},
        ]}
        metadata = {
            "packages": [
                {"id": "path+file:///moshi-core#moshi@0.6.4", "name": "moshi", "version": "0.6.4", "source": None, "manifest_path": "/tmp/moshi-core/Cargo.toml"},
                {"id": "registry+https://github.com/rust-lang/crates.io-index#candle-core@0.9.1", "name": "candle-core", "version": "0.9.1", "source": "registry+https://github.com/rust-lang/crates.io-index", "manifest_path": "/tmp/candle-core/Cargo.toml"},
            ],
            "resolve": {"root": "path+file:///moshi-core#moshi@0.6.4", "nodes": [
                {"id": "path+file:///moshi-core#moshi@0.6.4", "features": [], "deps": [{"pkg": "registry+https://github.com/rust-lang/crates.io-index#candle-core@0.9.1", "dep_kinds": []}]},
                {"id": "registry+https://github.com/rust-lang/crates.io-index#candle-core@0.9.1", "features": ["metal"], "deps": []},
            ]},
            "workspace_members": ["path+file:///moshi-core#moshi@0.6.4"],
            "workspace_default_members": ["path+file:///moshi-core#moshi@0.6.4"],
        }
        result = MODULE._metadata_inventory(metadata, lock)
        self.assertEqual(result["status"], "BLOCKED_FEATURE_OR_NATIVE_MARKER")
        self.assertIn("candle-core:metal", result["blocked_markers"])

    def test_metadata_missing_active_lock_row_fails_closed(self) -> None:
        root = "path+file:///moshi-core#moshi@0.6.4"
        metadata = {
            "packages": [
                {"id": root, "name": "moshi", "version": "0.6.4", "source": None, "manifest_path": "/tmp/moshi-core/Cargo.toml"},
                {"id": "registry+https://github.com/rust-lang/crates.io-index#missing@1", "name": "missing", "version": "1", "source": "registry+https://github.com/rust-lang/crates.io-index", "manifest_path": "/tmp/missing/Cargo.toml"},
            ],
            "workspace_members": [root],
            "resolve": {"root": root, "nodes": [{"id": root, "features": [], "deps": [{"pkg": "registry+https://github.com/rust-lang/crates.io-index#missing@1", "dep_kinds": []}]}, {"id": "registry+https://github.com/rust-lang/crates.io-index#missing@1", "features": [], "deps": []}]},
        }
        with self.assertRaisesRegex(MODULE.AuditError, "sourced package absent from lock"):
            MODULE._metadata_inventory(metadata, {"package": [{"name": "moshi", "version": "0.6.4", "source": None}]})

    def test_metadata_rejects_unknown_workspace_root(self) -> None:
        root = "path+file:///moshi-core#moshi@0.6.4"
        metadata = {
            "packages": [{"id": root, "name": "moshi", "version": "0.6.4", "source": None, "manifest_path": "/tmp/moshi-core/Cargo.toml"}],
            "workspace_members": ["missing-root"],
            "resolve": {"nodes": [{"id": root, "deps": [], "features": []}]},
        }
        with self.assertRaisesRegex(MODULE.AuditError, "metadata root package"):
            MODULE._metadata_inventory(metadata, {"package": [{"name": "moshi", "version": "0.6.4", "source": None}]})

    def test_metadata_rejects_malformed_root_source_before_lock_lookup(self) -> None:
        root = "path+file:///moshi-core#moshi@0.6.4"
        metadata = {
            "packages": [{"id": root, "name": "moshi", "version": "0.6.4", "source": {}, "manifest_path": "/tmp/moshi-core/Cargo.toml"}],
            "workspace_members": [root],
            "resolve": {"root": root, "nodes": [{"id": root, "deps": [], "features": []}]},
        }
        with self.assertRaisesRegex(MODULE.AuditError, "malformed package source"):
            MODULE._metadata_inventory(metadata, {"package": []})

    def test_bounded_subprocess_output_and_timeout_close_children(self) -> None:
        captured = []
        real_popen = MODULE.subprocess.Popen

        def capture(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            captured.append(process)
            return process

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            with self.assertRaises(MODULE.AuditError):
                with mock.patch.object(MODULE.subprocess, "Popen", capture):
                    MODULE._command_bytes([sys.executable, "-S", "-c", "import sys; sys.stdout.write('x' * 4096)"], 64)
            with self.assertRaises(MODULE.AuditError):
                with mock.patch.object(MODULE.subprocess, "Popen", capture):
                    MODULE._command_bytes([sys.executable, "-S", "-c", "import time; time.sleep(1)"], 1024, timeout=0.05)
        self.assertEqual(len(captured), 2)
        for process in captured:
            self.assertIsNotNone(process.poll())
            self.assertTrue(process.stdout.closed)
            self.assertTrue(process.stderr.closed)
        self.assertFalse(any(issubclass(item.category, ResourceWarning) for item in caught))

    def test_metadata_rejects_dangling_dependency_edge(self) -> None:
        root = "path+file:///moshi-core#moshi@0.6.4"
        metadata = {
            "packages": [{"id": root, "name": "moshi", "version": "0.6.4", "source": None, "manifest_path": "/tmp/moshi-core/Cargo.toml"}],
            "workspace_members": [root],
            "resolve": {"root": root, "nodes": [{"id": root, "deps": [{"pkg": "missing", "dep_kinds": []}], "features": []}]},
        }
        with self.assertRaisesRegex(MODULE.AuditError, "dangling dependency edge"):
            MODULE._metadata_inventory(metadata, {"package": [{"name": "moshi", "version": "0.6.4", "source": None}]})

    def test_metadata_distinguishes_inactive_optional_native_package(self) -> None:
        root = "path+file:///moshi-core#moshi@0.6.4"
        optional = "registry+https://github.com/rust-lang/crates.io-index#candle-core@0.9.1"
        metadata = {
            "packages": [
                {"id": root, "name": "moshi", "version": "0.6.4", "source": None, "manifest_path": "/tmp/moshi-core/Cargo.toml"},
                {"id": optional, "name": "candle-core", "version": "0.9.1", "source": "registry+https://github.com/rust-lang/crates.io-index", "manifest_path": "/tmp/candle-core/Cargo.toml"},
            ],
            "workspace_members": [root],
            "resolve": {"root": root, "nodes": [{"id": root, "deps": [], "features": []}]},
        }
        result = MODULE._metadata_inventory(metadata, {"package": [{"name": "moshi", "version": "0.6.4", "source": None}, {"name": "candle-core", "version": "0.9.1", "source": "registry+https://github.com/rust-lang/crates.io-index", "checksum": "a" * 64}]})
        self.assertEqual(result["active_package_count"], 1)
        self.assertEqual(result["all_package_count"], 2)
        self.assertEqual(result["blocked_markers"], [])

    def test_registry_inventory_without_external_source_is_open(self) -> None:
        result = MODULE._registry_inventory({"active_packages": []}, None)
        self.assertEqual(result["status"], "OPEN_REGISTRY_SOURCE_NOT_PROVIDED")

    def test_registry_archive_is_inspected_without_license_approval(self) -> None:
        lock_checksum = "b" * 64
        with tempfile.TemporaryDirectory(prefix="mimi-rust-audit-") as temp:
            root = Path(temp)
            archive = root / "candle-core-0.9.1.crate"
            payload = io.BytesIO()
            with tarfile.open(fileobj=payload, mode="w:gz") as tar:
                for name, data in (
                    ("candle-core-0.9.1/Cargo.toml", b"[package]\nname='candle-core'\n"),
                    ("candle-core-0.9.1/LICENSE", b"Apache-2.0\n"),
                ):
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
            archive.write_bytes(payload.getvalue())
            metadata = {"active_packages": [{"name": "candle-core", "version": "0.9.1", "lock_checksum": lock_checksum}]}
            result = MODULE._registry_inventory(metadata, root)
            self.assertEqual(result["status"], "OPEN_MISSING_SOURCE_ROWS")
            self.assertEqual(result["packages"][0]["status"], "BLOCKED_ARCHIVE_SHA_MISMATCH")
            self.assertNotIn("license_notice_open", result["packages"][0])

    def test_registry_archive_matching_checksum_is_inspected_open(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mimi-rust-audit-") as temp:
            root = Path(temp)
            archive = root / "candle-core-0.9.1.crate"
            payload = io.BytesIO()
            with tarfile.open(fileobj=payload, mode="w:gz") as tar:
                for name, data in (
                    ("candle-core-0.9.1/Cargo.toml", b"[package]\nname='candle-core'\n"),
                    ("candle-core-0.9.1/LICENSE", b"Apache-2.0\n"),
                ):
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
            archive.write_bytes(payload.getvalue())
            checksum = MODULE._sha(archive.read_bytes())
            metadata = {"active_packages": [{"name": "candle-core", "version": "0.9.1", "lock_checksum": checksum}]}
            result = MODULE._registry_inventory(metadata, root)
            self.assertEqual(result["status"], "SOURCE_BYTES_INSPECTED_OPEN")
            self.assertEqual(result["packages"][0]["status"], "AUTHENTICATED_ARCHIVE_INSPECTED_OPEN")
            self.assertTrue(result["packages"][0]["license_notice_open"])

    def test_registry_archive_rejects_foreign_prefix_after_checksum(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mimi-rust-audit-") as temp:
            root = Path(temp)
            archive = root / "candle-core-0.9.1.crate"
            payload = io.BytesIO()
            with tarfile.open(fileobj=payload, mode="w:gz") as tar:
                info = tarfile.TarInfo("other-0.9.1/Cargo.toml")
                info.size = 3
                tar.addfile(info, io.BytesIO(b"bad"))
            archive.write_bytes(payload.getvalue())
            metadata = {"active_packages": [{"name": "candle-core", "version": "0.9.1", "lock_checksum": MODULE._sha(archive.read_bytes())}]}
            with self.assertRaises(MODULE.AuditError):
                MODULE._registry_inventory(metadata, root)

    def test_registry_archive_rejects_links_and_duplicate_members(self) -> None:
        for mode in ("link", "duplicate"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(prefix="mimi-rust-audit-") as temp:
                root = Path(temp)
                archive = root / "candle-core-0.9.1.crate"
                payload = io.BytesIO()
                with tarfile.open(fileobj=payload, mode="w:gz") as tar:
                    if mode == "link":
                        info = tarfile.TarInfo("candle-core-0.9.1/LICENSE")
                        info.type = tarfile.SYMTYPE
                        info.linkname = "outside"
                        tar.addfile(info)
                    else:
                        for _ in range(2):
                            info = tarfile.TarInfo("candle-core-0.9.1/LICENSE")
                            info.size = 2
                            tar.addfile(info, io.BytesIO(b"ok"))
                archive.write_bytes(payload.getvalue())
                metadata = {"active_packages": [{"name": "candle-core", "version": "0.9.1", "lock_checksum": MODULE._sha(archive.read_bytes())}]}
                with self.assertRaises(MODULE.AuditError):
                    MODULE._registry_inventory(metadata, root)

    def test_production_audit_builds_bounded_derived_workspace(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mimi-rust-audit-") as temp:
            root = Path(temp) / "source"
            output = Path(temp) / "inventory"
            (root / "rust/moshi-core/src").mkdir(parents=True)
            cargo_root = """[workspace]\nmembers=[\"moshi-core\"]\nresolver=\"2\"\n[workspace.package]\nversion=\"0.1.0\"\nedition=\"2021\"\nlicense=\"MIT\"\ndescription=\"fixture\"\nrepository=\"https://example.invalid\"\nkeywords=[]\ncategories=[]\n[workspace.dependencies]\ncandle=\"0.1\"\ncandle-nn=\"0.1\"\ncandle-transformers=\"0.1\"\nrayon=\"1\"\nserde=\"1\"\ntracing=\"0.1\"\ncandle-flash-attn={version=\"0.1\",optional=true}\n"""
            cargo_core = """[package]\nname=\"moshi\"\nversion=\"0.1.0\"\n[dependencies]\ncandle={workspace=true}\ncandle-nn={workspace=true}\ncandle-transformers={workspace=true}\nrayon={workspace=true}\nserde={workspace=true}\ntracing={workspace=true}\ncandle-flash-attn={workspace=true}\n"""
            contents = {
                "rust/Cargo.toml": cargo_root.encode(),
                "rust/moshi-core/Cargo.toml": cargo_core.encode(),
                "rust/Cargo.lock": b"fixture-lock",
                "rust/README.md": b"fixture\n",
                "LICENSE-APACHE": b"Apache\n",
                "LICENSE-MIT": b"MIT\n",
                "rust/moshi-core/src/mimi.rs": b"pub fn fixture() {}\n",
            }
            for rel, data in contents.items():
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            old_declarations = MODULE.SOURCE_DECLARATIONS
            old_expected = MODULE.EXPECTED_LOCK_SHA256
            old_git = MODULE._git
            old_git_bytes = MODULE._git_bytes
            MODULE.SOURCE_DECLARATIONS = tuple(rel for rel in contents if not rel.endswith("/src/mimi.rs"))  # type: ignore[assignment]
            fixture_lock_sha = MODULE._sha(contents["rust/Cargo.lock"])
            MODULE.EXPECTED_LOCK_SHA256 = fixture_lock_sha
            def fake_git(_root: Path, args: list[str]) -> str:
                if args[:2] == ["rev-parse", "HEAD"]:
                    return MODULE.PINNED_REV + "\n"
                if args[:2] == ["remote", "get-url"]:
                    return "https://github.com/kyutai-labs/moshi.git\n"
                if args[:1] == ["status"]:
                    return ""
                if args[:1] == ["ls-files"]:
                    return "rust/moshi-core/src/mimi.rs\0"
                if args[:1] == ["rev-parse"]:
                    return MODULE._git_blob_id(contents[args[1].split(":", 1)[1]]) + "\n"
                raise AssertionError(args)
            def fake_git_bytes(_root: Path, args: list[str], _limit: int) -> bytes:
                return contents[args[-1].split(":", 1)[1]]
            try:
                MODULE._git = fake_git
                MODULE._git_bytes = fake_git_bytes
                self.assertEqual(MODULE.audit(root, output), 0)
            finally:
                MODULE.SOURCE_DECLARATIONS = old_declarations
                MODULE.EXPECTED_LOCK_SHA256 = old_expected
                MODULE._git = old_git
                MODULE._git_bytes = old_git_bytes
            inventory = json.loads((output / "source-inventory.json").read_text(encoding="utf-8"))
            self.assertEqual(inventory["overall"], "OPEN")
            self.assertEqual(inventory["status"], "STATIC_SOURCE_INVENTORY")
            self.assertTrue(all(value is False for value in inventory["execution"].values()))
            self.assertEqual(inventory["workspace"]["copied_lock"]["sha256"], fixture_lock_sha)
            self.assertIn("Cargo.lock", inventory["workspace"]["files"])
            self.assertEqual((output / "derived-workspace/moshi-core/src/mimi.rs").read_bytes(), contents["rust/moshi-core/src/mimi.rs"])
            recorded_digest = inventory.pop("inventory_payload_sha256")
            self.assertEqual(recorded_digest, MODULE._sha((json.dumps(inventory, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()))

    def test_output_parent_validator_rejects_symlink_before_creation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mimi-rust-audit-") as temp:
            root = Path(temp)
            real = root / "real"
            real.mkdir()
            link = root / "link"
            link.symlink_to(real, target_is_directory=True)
            with self.assertRaises(MODULE.AuditError):
                MODULE._validate_output_parent(link / "new-output")

    def test_source_root_ancestor_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mimi-rust-audit-") as temp:
            root = Path(temp)
            real = root / "real"
            real.mkdir()
            link = root / "link"
            link.symlink_to(real, target_is_directory=True)
            with self.assertRaises(MODULE.AuditError):
                MODULE._validate_path_ancestors(link / "source")

    def test_output_parent_symlink_is_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mimi-rust-audit-") as temp:
            root = Path(temp)
            real = root / "real"
            real.mkdir()
            link = root / "link"
            link.symlink_to(real, target_is_directory=True)
            self.assertTrue(link.is_symlink())
            with self.assertRaises(MODULE.AuditError):
                MODULE._ensure_parent_chain(link, "nested/file")

    def test_json_fixture_is_bounded_and_stdlib_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mimi-rust-audit-") as temp:
            path = Path(temp) / "metadata.json"
            path.write_text(json.dumps({"packages": [], "resolve": {"nodes": []}}), encoding="utf-8")
            self.assertEqual(MODULE._read_json(path)["packages"], [])
            self.assertEqual(os.path.basename(__file__), "test_mimi_rust_core_source_audit.py")


if __name__ == "__main__":
    unittest.main(verbosity=2)
