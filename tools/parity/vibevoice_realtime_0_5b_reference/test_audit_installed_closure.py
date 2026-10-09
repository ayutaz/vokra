"""Stdlib-only regression tests for the bounded closure collector.

These tests exercise synthetic wheel metadata and bytes only.  They are not
evidence that a third-party distribution was installed, imported, or approved.
"""

from __future__ import annotations

import contextlib
import io
import importlib.util
import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import zipfile
from pathlib import Path
from unittest import mock


_MODULE_PATH = Path(__file__).with_name("audit_installed_closure.py")
_SPEC = importlib.util.spec_from_file_location("vokra_vibevoice_audit", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(AUDIT)


class CollectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="vokra-vibevoice-test-")
        self.root = Path(self.temp.name).resolve()
        self.archive, self.files, self.site = AUDIT._fake_wheel(self.root)
        archive_sha = AUDIT.hashlib.sha256(self.archive.read_bytes()).hexdigest()
        url = f"https://files.pythonhosted.org/packages/{self.archive.name}"
        self.manifest = {
            "format": "vokra-vibevoice-selected-wheel-manifest-v1",
            "platform": {"system": "Linux", "machine": "x86_64", "python": "3.12"},
            "project_sha256": "p" * 64,
            "uv_lock_sha256": "l" * 64,
            "artifacts": [{
                "name": "demo",
                "version": "1.0",
                "url": url,
                "filename": self.archive.name,
                "sha256": archive_sha,
                "bytes": self.archive.stat().st_size,
                "path": str(self.archive),
            }],
        }
        self.locked = {
            "demo": {
                "name": "demo",
                "version": "1.0",
                "source": {"registry": "https://pypi.org/simple"},
                "wheels": [{"url": url, "hash": f"sha256:{archive_sha}", "size": self.archive.stat().st_size}],
            }
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_runtime_requires_no_site_and_no_bytecode_flags(self) -> None:
        AUDIT.validate_runtime_flags(SimpleNamespace(no_site=True, dont_write_bytecode=True))
        with self.assertRaisesRegex(AUDIT.AuditError, "-S"):
            AUDIT.validate_runtime_flags(SimpleNamespace(no_site=False, dont_write_bytecode=True))
        with self.assertRaisesRegex(AUDIT.AuditError, "-B"):
            AUDIT.validate_runtime_flags(SimpleNamespace(no_site=True, dont_write_bytecode=False))

    def test_audit_rejects_missing_runtime_flags_before_input_reads(self) -> None:
        args = argparse.Namespace(
            project=self.root / "missing-project.toml",
            lock=self.root / "missing-lock.toml",
            site_packages=self.root / "missing-site",
            venv_root=self.root / "missing-venv",
            scripts_root=self.root / "missing-bin",
            selected_wheel_manifest=self.root / "missing-manifest.json",
            source_root=self.root / "missing-source",
            output=self.root / "missing-output.json",
        )
        for flags, expected in (
            (SimpleNamespace(no_site=False, dont_write_bytecode=True), "-S"),
            (SimpleNamespace(no_site=True, dont_write_bytecode=False), "-B"),
        ):
            with self.subTest(expected=expected), mock.patch.object(AUDIT.sys, "flags", flags), mock.patch.object(
                AUDIT, "read_with_identity", side_effect=AssertionError("input read occurred before runtime gate")
            ), self.assertRaisesRegex(AUDIT.AuditError, expected):
                AUDIT.audit(args)

    def test_positive_record_license_and_archive_binding(self) -> None:
        selected = AUDIT.verify_selected_artifacts(
            self.manifest, self.root / "manifest.json", "p" * 64, "l" * 64, self.locked
        )
        row = AUDIT.inspect_installed_package(self.site, self.root / "scripts", "demo", self.locked["demo"], selected["demo"])
        self.assertEqual(row["license_file_status"], "PRESENT")
        self.assertEqual(row["license_files"][0]["path"], "demo-1.0.dist-info/LICENCE")
        self.assertEqual(row["archive_binding"], "METADATA_WHEEL_RECORD_LOGICAL_LICENSE_NATIVE_BYTES_BOUND")

    def test_declared_python_source_allows_bounded_installer_pyc(self) -> None:
        pyc = self.site / "demo" / "__pycache__" / "__init__.cpython-312.pyc"
        pyc.parent.mkdir()
        pyc.write_bytes(b"synthetic-pyc\n")
        record = self.site / "demo-1.0.dist-info" / "RECORD"
        record.write_text(record.read_text(encoding="utf-8") + f"demo/__pycache__/__init__.cpython-312.pyc,,{pyc.stat().st_size}\n", encoding="utf-8")
        selected = AUDIT.verify_selected_artifacts(self.manifest, self.root / "manifest.json", "p" * 64, "l" * 64, self.locked)
        row = AUDIT.inspect_installed_package(self.site, self.root / "scripts", "demo", self.locked["demo"], selected["demo"])
        self.assertEqual(row["relocation"]["generated_pyc"][0]["path"], "demo/__pycache__/__init__.cpython-312.pyc")
        self.assertEqual(row["relocation"]["generated_pyc"][0]["binding"], "UNPROVEN_INSTALLER_SOURCE")

    def test_installed_name_comes_from_authenticated_metadata(self) -> None:
        original = self.site / "demo-1.0.dist-info"
        renamed = self.site / "unrelated-directory-name.dist-info"
        original.rename(renamed)
        self.assertEqual(AUDIT.installed_distribution_names(self.site), {"demo"})

    def test_missing_archive_record_member_is_blocked(self) -> None:
        tampered = self.root / "missing-member.whl"
        with zipfile.ZipFile(self.archive) as source, zipfile.ZipFile(tampered, "w") as target:
            for info in source.infolist():
                if info.filename != "demo/__init__.py":
                    target.writestr(info, source.read(info))
        members = AUDIT.archive_members(tampered)
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.validate_archive_record(tampered, members, "demo-1.0.dist-info")

    def test_declared_large_archive_member_is_rejected_before_read(self) -> None:
        with zipfile.ZipFile(self.archive) as archive:
            info = archive.getinfo("demo/__init__.py")
        info.file_size = AUDIT.MAX_ARCHIVE_MEMBER_BYTES + 1
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.archive_member_bytes(self.archive, {info.filename: info}, info.filename)

    def test_symlink_wheel_member_is_blocked(self) -> None:
        symlink = self.root / "symlink.whl"
        info = zipfile.ZipInfo("demo-1.0.dist-info/link")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(symlink, "w") as archive:
            archive.writestr(info, "target")
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.archive_members(symlink)

    def test_sdist_substitution_is_blocked(self) -> None:
        row = self.manifest["artifacts"][0]
        row["filename"] = "demo-1.0.tar.gz"
        row["url"] = "https://files.pythonhosted.org/packages/demo-1.0.tar.gz"
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.verify_selected_artifacts(
                self.manifest, self.root / "manifest.json", "p" * 64, "l" * 64, self.locked
            )

    def test_archive_digest_mismatch_is_blocked(self) -> None:
        row = self.manifest["artifacts"][0]
        row["sha256"] = "0" * 64
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.verify_selected_artifacts(
                self.manifest, self.root / "manifest.json", "p" * 64, "l" * 64, self.locked
            )

    def test_selected_wheel_missing_lock_size_is_blocked(self) -> None:
        self.locked["demo"]["wheels"][0].pop("size")
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.verify_selected_artifacts(
                self.manifest, self.root / "manifest.json", "p" * 64, "l" * 64, self.locked
            )

    def test_record_extra_noninstaller_path_is_blocked(self) -> None:
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.bind_installed_record_to_wheel(
                self.site,
                self.root / "scripts",
                {"demo-1.0.dist-info/RECORD": ("", -1), "unreviewed.bin": ("sha256=" + "a" * 43, 1)},
                {"demo-1.0.dist-info/RECORD": ("", -1)},
                set(),
                self.archive,
            )

    def test_console_wrapper_relocation_is_bound_to_declared_entry_point(self) -> None:
        root = self.root / "relocated"
        root.mkdir()
        _, site, scripts, selected, locked = AUDIT._fake_relocated_wheel(root, purelib_native=True)
        row = AUDIT.inspect_installed_package(site, scripts, "demo", locked, selected)
        self.assertEqual(row["relocation"]["relocated_scripts"][0]["installed"], "../../../bin/demo")
        self.assertEqual(row["relocation"]["relocated_scripts"][0]["script"], "demo")
        self.assertEqual(row["relocation"]["relocated_scripts"][0]["bytes"], len(b"#!/private/venv/bin/python\n# rewritten by installer\n"))
        self.assertEqual(row["relocation"]["relocated_scripts"][0]["binding"], "UNPROVEN_INSTALLER_SOURCE")
        self.assertEqual(row["native_files"][0]["archive_path"], "demo-1.0.data/purelib/demo_native.so")

    def test_generated_console_wrapper_is_separately_unproven(self) -> None:
        root = self.root / "generated"
        root.mkdir()
        _, site, scripts, selected, locked = AUDIT._fake_relocated_wheel(root, archive_script=False)
        row = AUDIT.inspect_installed_package(site, scripts, "demo", locked, selected)
        self.assertEqual(row["relocation"]["relocated_scripts"], [])
        self.assertEqual(row["relocation"]["generated_wrappers"][0]["binding"], "UNPROVEN_INSTALLER_SOURCE")
        self.assertEqual(row["relocation"]["generated_wrappers"][0]["entry_point_source"], "demo-1.0.dist-info/entry_points.txt")
        self.assertEqual(row["relocation"]["generated_wrappers"][0]["entry_point_source_bytes"], len(b"[console_scripts]\ndemo = demo:main\n"))

    def test_metadata_24_license_file_uses_dist_info_licenses_path(self) -> None:
        root = self.root / "pep639"
        root.mkdir()
        archive, _, site = AUDIT._fake_wheel(root, metadata_version="2.4", license_path="LICENSE", archive_license_path="licenses/LICENSE")
        digest = AUDIT.hashlib.sha256(archive.read_bytes()).hexdigest()
        url = f"https://files.pythonhosted.org/packages/{archive.name}"
        selected = {"name": "demo", "version": "1.0", "url": url, "filename": archive.name, "bytes": archive.stat().st_size, "sha256": digest, "path": str(archive)}
        locked = {"name": "demo", "version": "1.0", "source": {"registry": "https://pypi.org/simple"}, "wheels": [{"url": url, "hash": f"sha256:{digest}", "size": archive.stat().st_size}]}
        row = AUDIT.inspect_installed_package(site, root / "scripts", "demo", locked, selected)
        self.assertEqual(row["license_files"][0]["path"], "demo-1.0.dist-info/licenses/LICENSE")
        nested = self.root / "pep639-nested"
        nested.mkdir()
        archive, _, site = AUDIT._fake_wheel(nested, metadata_version="2.4", license_path="licenses/LICENSE.MIT", archive_license_path="licenses/licenses/LICENSE.MIT")
        digest = AUDIT.hashlib.sha256(archive.read_bytes()).hexdigest()
        url = f"https://files.pythonhosted.org/packages/{archive.name}"
        selected = {"name": "demo", "version": "1.0", "url": url, "filename": archive.name, "bytes": archive.stat().st_size, "sha256": digest, "path": str(archive)}
        locked = {"name": "demo", "version": "1.0", "source": {"registry": "https://pypi.org/simple"}, "wheels": [{"url": url, "hash": f"sha256:{digest}", "size": archive.stat().st_size}]}
        row = AUDIT.inspect_installed_package(site, nested / "scripts", "demo", locked, selected)
        self.assertEqual(row["license_files"][0]["path"], "demo-1.0.dist-info/licenses/licenses/LICENSE.MIT")

    def test_metadata_24_license_header_does_not_bind_legacy_archive_path(self) -> None:
        root = self.root / "pep639-invalid"
        root.mkdir()
        archive, _, site = AUDIT._fake_wheel(root, metadata_version="2.4", license_path="LICENSE", archive_license_path="LICENSE")
        digest = AUDIT.hashlib.sha256(archive.read_bytes()).hexdigest()
        selected = {"name": "demo", "version": "1.0", "url": f"https://files.pythonhosted.org/packages/{archive.name}", "filename": archive.name, "bytes": archive.stat().st_size, "sha256": digest, "path": str(archive)}
        locked = {"demo": {"name": "demo", "version": "1.0", "source": {"registry": "https://pypi.org/simple"}, "wheels": [{"url": selected["url"], "hash": f"sha256:{digest}", "size": archive.stat().st_size}]}}
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.inspect_installed_package(site, root / "scripts", "demo", locked["demo"], selected)

    def test_duplicate_required_headers_are_blocked(self) -> None:
        metadata = self.site / "demo-1.0.dist-info" / "METADATA"
        metadata.write_bytes(metadata.read_bytes().replace(b"Version: 1.0\n", b"Version: 1.0\nVersion: 1.0\n"))
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.inspect_installed_package(self.site, self.root / "scripts", "demo", self.locked["demo"], self.manifest["artifacts"][0])

    def test_read_budget_and_record_row_bounds_fail_closed(self) -> None:
        old_read_budget = AUDIT.MAX_TOTAL_READ_BYTES
        old_record_rows = AUDIT.MAX_RECORD_ROWS
        try:
            AUDIT.MAX_TOTAL_READ_BYTES = 1
            with self.assertRaises(AUDIT.AuditError):
                AUDIT.read_bounded(self.archive, 1024, "budget")
            AUDIT.MAX_TOTAL_READ_BYTES = old_read_budget
            AUDIT.MAX_RECORD_ROWS = 2
            body = b"a,,1\nb,,1\nc,,1\n"
            with self.assertRaises(AUDIT.AuditError):
                AUDIT.parse_record_rows(body, "rows")
        finally:
            AUDIT.MAX_TOTAL_READ_BYTES = old_read_budget
            AUDIT.MAX_RECORD_ROWS = old_record_rows

    def test_inventory_revalidation_catches_same_size_wrapper_and_new_file(self) -> None:
        root = self.root / "inventory"
        root.mkdir()
        _, site, scripts, _, _ = AUDIT._fake_relocated_wheel(root)
        initial = AUDIT.validate_install_inventory(site, scripts)
        wrapper = scripts / "demo"
        original = wrapper.read_bytes()
        wrapper.write_bytes(b"X" * len(original))
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.revalidate_install_inventory(site, scripts, initial)
        wrapper.write_bytes(original)
        (site / "unregistered.py").write_bytes(b"unexpected\n")
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.revalidate_install_inventory(site, scripts, initial)

    def test_multiple_installer_and_pyc_rows_are_sorted_and_hash_bound(self) -> None:
        root = self.root / "multiple-generated"
        root.mkdir()
        archive, _, site = AUDIT._fake_wheel(root)
        scripts = root / "scripts"
        scripts.mkdir()
        dist = "demo-1.0.dist-info"
        def encoded(data: bytes) -> str:
            return "sha256=" + AUDIT.base64.urlsafe_b64encode(AUDIT.hashlib.sha256(data).digest()).decode().rstrip("=")
        installed: dict[str, tuple[str, int]] = {f"{dist}/RECORD": ("", -1), "demo/__init__.py": (encoded(b"demo\n"), 5)}
        archive_records: dict[str, tuple[str, int]] = {f"{dist}/RECORD": ("", -1), "demo/__init__.py": (encoded(b"demo\n"), 5), "demo/one.py": (encoded(b"one\n"), 4), "demo/two.py": (encoded(b"two\n"), 4)}
        for relative, data in {
            "demo/one.py": b"one\n", "demo/two.py": b"two\n",
            "demo/__pycache__/one.cpython-312.pyc": b"pyc-one", "demo/__pycache__/two.cpython-312.pyc": b"pyc-two",
            f"{dist}/INSTALLER": b"uv\n", f"{dist}/REQUESTED": b"\n",
        }.items():
            target = site / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            installed[relative] = (encoded(data), len(data))
        result = AUDIT.bind_installed_record_to_wheel(site, scripts, installed, archive_records, set(), archive)
        self.assertEqual([row["path"] for row in result["installer_generated"]], [f"{dist}/INSTALLER", f"{dist}/REQUESTED"])
        self.assertEqual([row["path"] for row in result["generated_pyc"]], ["demo/__pycache__/one.cpython-312.pyc", "demo/__pycache__/two.cpython-312.pyc"])
        self.assertTrue(all(row["binding"] == "UNPROVEN_INSTALLER_SOURCE" for row in result["generated_pyc"]))

    def test_site_virtualenv_bootstrap_symlinks_are_blocked(self) -> None:
        for name in sorted(AUDIT.VENV_SITE_BOOTSTRAP):
            root = self.root / f"bootstrap-{name.replace('.', '-')}"
            root.mkdir()
            _, site, scripts, _, _ = AUDIT._fake_relocated_wheel(root)
            target = root / f"external-{name}"
            target.write_bytes(b"external bootstrap\n")
            (site / name).symlink_to(target)
            with self.assertRaises(AUDIT.AuditError):
                AUDIT.validate_install_inventory(site, scripts)

    def test_site_virtualenv_bootstrap_swap_is_caught_after_inventory(self) -> None:
        root = self.root / "bootstrap-swap"
        root.mkdir()
        _, site, scripts, _, _ = AUDIT._fake_relocated_wheel(root)
        bootstrap = site / "_virtualenv.py"
        bootstrap.write_bytes(b"trusted-looking bootstrap\n")
        initial = AUDIT.validate_install_inventory(site, scripts)
        external = root / "external-bootstrap.py"
        external.write_bytes(b"external bootstrap\n")
        bootstrap.unlink()
        bootstrap.symlink_to(external)
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.revalidate_install_inventory(site, scripts, initial)

    def test_file_identity_catches_same_hash_inode_replacement(self) -> None:
        target = self.root / "identity.bin"
        target.write_bytes(b"stable\n")
        initial = AUDIT.file_identity(target, "identity")
        replacement = self.root / "replacement.bin"
        replacement.write_bytes(b"stable\n")
        os.replace(replacement, target)
        self.assertNotEqual(initial["ino"], AUDIT.file_identity(target, "identity")["ino"])

    def test_read_with_identity_rejects_replacement_between_snapshot_and_read(self) -> None:
        target = self.root / "snapshot-race.bin"
        target.write_bytes(b"before\n")
        replacement = self.root / "snapshot-race-replacement.bin"
        original_snapshot = AUDIT.snapshot
        calls = 0
        def race_snapshot(path: Path, label: str) -> tuple[int, int, int, int, int, int]:
            nonlocal calls
            if calls == 1:
                replacement.write_bytes(b"after!\n")
                os.replace(replacement, target)
            info = original_snapshot(path, label)
            calls += 1
            return info
        with mock.patch.object(AUDIT, "snapshot", side_effect=race_snapshot), self.assertRaises(AUDIT.AuditError):
            AUDIT.read_with_identity(target, 1024, "snapshot race")

    def test_source_identity_rejects_dirty_checkout(self) -> None:
        source = self.root / "source-dirty"
        source.mkdir()
        subprocess.run(["git", "-C", str(source), "init", "-q"], check=True)
        subprocess.run(["git", "-C", str(source), "config", "user.email", "audit@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(source), "config", "user.name", "audit"], check=True)
        (source / "README").write_text("clean\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(source), "add", "README"], check=True)
        subprocess.run(["git", "-C", str(source), "commit", "-qm", "source"], check=True)
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        subprocess.run(["git", "-C", str(source), "remote", "add", "origin", AUDIT.OFFICIAL_SOURCE_REPOSITORY], check=True)
        (source / "README").write_text("dirty\n", encoding="utf-8")
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.source_identity(source, revision)

    def test_project_torch_declaration_is_exact(self) -> None:
        project = {
            "project": {
                "name": AUDIT.PROJECT_NAME,
                "requires-python": "==3.12.*",
                "dependencies": ["torch==2.13.0evil ; platform_machine == 'x86_64' and sys_platform == 'linux'"],
            },
            "tool": {
                "uv": {
                    "package": False,
                    "environments": ["sys_platform == 'linux' and platform_machine == 'x86_64'"],
                    "sources": {"torch": {"index": "pytorch-cpu"}},
                    "index": [{"name": "pytorch-cpu", "url": "https://download.pytorch.org/whl/cpu", "explicit": True}],
                },
                "vokra": {"reference": {"torch_distribution": AUDIT.CPU_TORCH_VERSION, "torch_index": "https://download.pytorch.org/whl/cpu"}},
            },
        }
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.validate_project(project)

    def test_wheel_tag_mismatch_is_blocked(self) -> None:
        AUDIT.wheel_tags_compatible("demo-1.0-py2.py3-none-any.whl")
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.wheel_tags_compatible("demo-1.0-py2-none-any.whl")
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.wheel_tags_compatible("demo-1.0-cp313-cp313-manylinux_2_28_x86_64.whl")
        oversized_platform = ".".join(f"p{index}" for index in range(129))
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.wheel_tags_compatible(f"demo-1.0-py3-none-{oversized_platform}.whl")
        archive = self.root / "tag-mismatch.whl"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("demo-1.0.dist-info/WHEEL", "Wheel-Version: 1.0\nTag: py3-none-any\n\n")
        members = AUDIT.archive_members(archive)
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.validate_wheel_metadata_tags(archive, members, "demo-1.0.dist-info", "demo-1.0-cp312-cp312-manylinux_2_28_x86_64.whl")
        duplicate = self.root / "tag-duplicate.whl"
        with zipfile.ZipFile(duplicate, "w") as output:
            output.writestr("demo-1.0.dist-info/WHEEL", "Wheel-Version: 1.0\nTag: py3-none-any\nTag: py3-none-any\n\n")
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.validate_wheel_metadata_tags(duplicate, AUDIT.archive_members(duplicate), "demo-1.0.dist-info", "demo-1.0-py3-none-any.whl")
        expanded = self.root / "tag-expanded.whl"
        with zipfile.ZipFile(expanded, "w") as output:
            output.writestr("demo-1.0.dist-info/WHEEL", "Wheel-Version: 1.0\nTag: cp312-cp312-manylinux_2_28_x86_64\nTag: cp312-cp312-manylinux_2_17_x86_64\n\n")
        AUDIT.validate_wheel_metadata_tags(expanded, AUDIT.archive_members(expanded), "demo-1.0.dist-info", "demo-1.0-cp312-cp312-manylinux_2_28_x86_64.manylinux_2_17_x86_64.whl")
        missing = self.root / "tag-missing.whl"
        with zipfile.ZipFile(missing, "w") as output:
            output.writestr("demo-1.0.dist-info/WHEEL", "Wheel-Version: 1.0\nTag: cp312-cp312-manylinux_2_28_x86_64\n\n")
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.validate_wheel_metadata_tags(missing, AUDIT.archive_members(missing), "demo-1.0.dist-info", "demo-1.0-cp312-cp312-manylinux_2_28_x86_64.manylinux_2_17_x86_64.whl")
        extra = self.root / "tag-extra.whl"
        with zipfile.ZipFile(extra, "w") as output:
            output.writestr("demo-1.0.dist-info/WHEEL", "Wheel-Version: 1.0\nTag: cp312-cp312-manylinux_2_28_x86_64\nTag: cp312-cp312-manylinux_2_17_x86_64\nTag: cp312-cp312-manylinux_2_12_x86_64\n\n")
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.validate_wheel_metadata_tags(extra, AUDIT.archive_members(extra), "demo-1.0.dist-info", "demo-1.0-cp312-cp312-manylinux_2_28_x86_64.manylinux_2_17_x86_64.whl")

    def test_public_main_rejects_malformed_input_without_output(self) -> None:
        output = self.root / "must-not-be-created.json"
        argv = [
            "audit_installed_closure.py",
            "--project", str(self.root / "missing-project.toml"),
            "--lock", str(self.root / "missing-lock.toml"),
            "--site-packages", str(self.root / "missing-site"),
            "--venv-root", str(self.root / "missing-venv"),
            "--scripts-root", str(self.root / "missing-bin"),
            "--selected-wheel-manifest", str(self.root / "malformed.json"),
            "--source-root", str(self.root / "missing-source"),
            "--output", str(output),
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(AUDIT.platform, "system", return_value="Linux"), mock.patch.object(AUDIT.platform, "machine", return_value="x86_64"), mock.patch.dict(AUDIT.os.environ, {"VOKRA_PUBLISH_ON_VAST": "1"}, clear=False):
            self.assertEqual(AUDIT.main(), 2)
        self.assertFalse(output.exists())

    def test_selected_manifest_rejects_duplicate_keys_at_every_object_level(self) -> None:
        valid = json.dumps(self.manifest, separators=(",", ":"))
        def inject_duplicate(raw: str, key: str, replacement: object) -> str:
            encoded_key = json.dumps(key, separators=(",", ":"))
            marker = encoded_key + ":"
            key_offset = raw.index(marker)
            value_offset = key_offset + len(marker)
            _, value_end = json.JSONDecoder().raw_decode(raw, value_offset)
            encoded_value = json.dumps(replacement, separators=(",", ":"))
            return raw[:value_end] + "," + encoded_key + ":" + encoded_value + raw[value_end:]

        def replacement_for(value: object) -> object:
            if isinstance(value, str):
                return value + "-different"
            if isinstance(value, int):
                return value + 1
            if isinstance(value, dict):
                return {**value, "unexpected": True}
            if isinstance(value, list):
                return value + [{"unexpected": True}]
            raise AssertionError(f"unhandled synthetic manifest value: {value!r}")

        cases = (
            [("top-level", key, self.manifest[key]) for key in ("format", "platform", "project_sha256", "uv_lock_sha256", "artifacts")]
            + [("platform", key, self.manifest["platform"][key]) for key in ("system", "machine", "python")]
            + [("artifact", key, self.manifest["artifacts"][0][key]) for key in ("name", "version", "url", "filename", "sha256", "bytes", "path")]
        )
        for scope, key, original in cases:
            for variant, replacement in (("equal", original), ("different", replacement_for(original))):
                with self.subTest(scope=scope, key=key, variant=variant):
                    body = inject_duplicate(valid, key, replacement)
                    self.assertNotEqual(body, valid)
                    ordinary = json.loads(body)
                    if variant == "equal":
                        self.assertEqual(ordinary, self.manifest)
                    if scope == "top-level":
                        self.assertEqual(ordinary[key], replacement)
                    elif scope == "platform":
                        self.assertEqual(ordinary["platform"][key], replacement)
                    else:
                        self.assertEqual(ordinary["artifacts"][0][key], replacement)
                    with self.assertRaisesRegex(AUDIT.AuditError, "duplicate JSON object key"):
                        AUDIT.parse_json_bytes(body.encode(), f"{scope} manifest")

        duplicate_path = self.root / "duplicate-read-json.json"
        duplicate_path.write_bytes(inject_duplicate(valid, "path", self.manifest["artifacts"][0]["path"]).encode())
        with self.assertRaisesRegex(AUDIT.AuditError, "duplicate JSON object key"):
            AUDIT.read_json(duplicate_path, AUDIT.MAX_MANIFEST_BYTES, "duplicate read manifest")

    def test_duplicate_manifest_is_rejected_before_report_creation(self) -> None:
        project = self.root / "project.toml"
        lock = self.root / "uv.lock"
        project.write_text("synthetic project\n", encoding="utf-8")
        lock.write_text("synthetic lock\n", encoding="utf-8")
        manifest_path = self.root / "duplicate-manifest.json"
        manifest_path.write_text(
            json.dumps(self.manifest, separators=(",", ":")).replace(
                '"name":"demo","version":',
                '"name":"demo","name":"demo","version":',
                1,
            ),
            encoding="utf-8",
        )
        output = self.root / "duplicate-manifest-report.json"
        argv = [
            "audit_installed_closure.py",
            "--project", str(project),
            "--lock", str(lock),
            "--site-packages", str(self.site),
            "--venv-root", str(self.root / "synthetic-venv"),
            "--scripts-root", str(self.root / "synthetic-venv" / "bin"),
            "--selected-wheel-manifest", str(manifest_path),
            "--source-root", str(self.root / "missing-source"),
            "--output", str(output),
        ]
        stderr = io.StringIO()
        with mock.patch.object(AUDIT.platform, "system", return_value="Linux"), mock.patch.object(AUDIT.platform, "machine", return_value="x86_64"), mock.patch.object(AUDIT, "validate_install_roots", return_value=(self.site, self.root / "synthetic-venv", self.root / "synthetic-venv" / "bin")), mock.patch.object(AUDIT, "validate_venv_layout", return_value={}), mock.patch.object(AUDIT, "parse_toml_bytes", side_effect=[{}, {}]), mock.patch.object(AUDIT, "validate_project"), mock.patch.object(AUDIT, "validate_lock", return_value={}), mock.patch.object(AUDIT, "verify_selected_artifacts") as verify, mock.patch.object(sys, "argv", argv), mock.patch.dict(AUDIT.os.environ, {"VOKRA_PUBLISH_ON_VAST": "1"}, clear=False), contextlib.redirect_stderr(stderr):
            self.assertEqual(AUDIT.main(), 2)
        self.assertIn("duplicate JSON object key", stderr.getvalue())
        verify.assert_not_called()
        self.assertFalse(output.exists())
        self.assertFalse(Path(str(output) + ".sha256").exists())

        existing_output = self.root / "duplicate-manifest-existing.json"
        existing_sidecar = Path(str(existing_output) + ".sha256")
        existing_output.write_bytes(b"existing report\n")
        existing_sidecar.write_bytes(b"existing digest\n")
        argv[-1] = str(existing_output)
        before_output = existing_output.read_bytes()
        before_sidecar = existing_sidecar.read_bytes()
        with mock.patch.object(AUDIT.platform, "system", return_value="Linux"), mock.patch.object(AUDIT.platform, "machine", return_value="x86_64"), mock.patch.object(AUDIT, "validate_install_roots", return_value=(self.site, self.root / "synthetic-venv", self.root / "synthetic-venv" / "bin")), mock.patch.object(AUDIT, "validate_venv_layout", return_value={}), mock.patch.object(AUDIT, "parse_toml_bytes", side_effect=[{}, {}]), mock.patch.object(AUDIT, "validate_project"), mock.patch.object(AUDIT, "validate_lock", return_value={}), mock.patch.object(AUDIT, "verify_selected_artifacts") as verify, mock.patch.object(sys, "argv", argv), mock.patch.dict(AUDIT.os.environ, {"VOKRA_PUBLISH_ON_VAST": "1"}, clear=False), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(AUDIT.main(), 2)
        verify.assert_not_called()
        self.assertEqual(existing_output.read_bytes(), before_output)
        self.assertEqual(existing_sidecar.read_bytes(), before_sidecar)

    def test_undeclared_console_wrapper_and_out_of_root_path_are_blocked(self) -> None:
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.wheel_install_location("demo-1.0.data/scripts/not-declared", "demo-1.0.dist-info", {"demo"})
        (self.root / "scripts").mkdir()
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.record_location(self.site, self.root / "scripts", "../../outside/not-a-script", "record")
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.bind_installed_record_to_wheel(
                self.site,
                self.root / "scripts",
                {},
                {"demo-1.0.data/purelib/bin/demo": ("", 1), "bin/demo": ("", 1)},
                set(),
                self.archive,
            )

    def test_full_audit_orchestrator_is_fail_closed_without_real_checkout(self) -> None:
        demo_root = self.root / "demo-wheel"
        demo_root.mkdir()
        demo_archive, _, demo_site = AUDIT._fake_wheel(demo_root)
        torch_root = self.root / "torch-wheel"
        torch_root.mkdir()
        torch_archive, _, torch_site = AUDIT._fake_wheel(
            torch_root,
            package_name="torch",
            package_version="2.13.0+cpu",
            archive_name="torch-2.13.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl",
        )
        venv = self.root / "venv"
        site = venv / "lib" / "python3.12" / "site-packages"
        scripts = venv / "bin"
        site.mkdir(parents=True)
        scripts.mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text(
            "home = /usr/bin\ninclude-system-site-packages = false\nversion = 3.12.8\n",
            encoding="utf-8",
        )
        interpreter = scripts / "python"
        interpreter.write_bytes(b"synthetic CPython 3.12 interpreter\n")
        interpreter.chmod(0o755)
        (scripts / "python3").symlink_to("python")
        (scripts / "python3.12").symlink_to("python")
        for source in (demo_site, torch_site):
            for child in source.iterdir():
                destination = site / child.name
                if child.is_dir():
                    shutil.copytree(child, destination)
                else:
                    shutil.copy2(child, destination)
        def artifact_line(archive: Path, url: str) -> str:
            digest = AUDIT.hashlib.sha256(archive.read_bytes()).hexdigest()
            return f"{{ hash = 'sha256:{digest}', url = '{url}', size = {archive.stat().st_size} }}"
        demo_url = f"https://files.pythonhosted.org/packages/{demo_archive.name}"
        torch_url = f"https://download-r2.pytorch.org/whl/cpu/{torch_archive.name.replace('+', '%2B')}"
        source = self.root / "source"
        source.mkdir()
        subprocess.run(["git", "-C", str(source), "init", "-q"], check=True)
        subprocess.run(["git", "-C", str(source), "config", "user.email", "audit@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(source), "config", "user.name", "audit"], check=True)
        (source / "README").write_text("synthetic source\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(source), "add", "README"], check=True)
        subprocess.run(["git", "-C", str(source), "commit", "-qm", "source"], check=True)
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        subprocess.run(["git", "-C", str(source), "remote", "add", "origin", AUDIT.OFFICIAL_SOURCE_REPOSITORY], check=True)
        project = self.root / "pyproject.toml"
        lock = self.root / "uv.lock"
        project.write_text(
            "[project]\nname = 'vokra-vibevoice-realtime-0-5b-reference'\n"
            "requires-python = '==3.12.*'\ndependencies = ['demo==1.0', \"torch==2.13.0 ; platform_machine == 'x86_64' and sys_platform == 'linux'\"]\n"
            "[tool.uv]\npackage = false\nenvironments = [\"sys_platform == 'linux' and platform_machine == 'x86_64'\"]\n"
            "[tool.uv.sources]\ntorch = { index = 'pytorch-cpu' }\n"
            "[[tool.uv.index]]\nname = 'pytorch-cpu'\nurl = 'https://download.pytorch.org/whl/cpu'\nexplicit = true\n"
            "[tool.vokra.reference]\nofficial_source_revision = '" + revision + "'\n"
            "torch_distribution = '2.13.0+cpu'\ntorch_index = 'https://download.pytorch.org/whl/cpu'\n",
            encoding="utf-8",
        )
        lock.write_text(
            "version = 1\nrevision = 3\nrequires-python = '==3.12.*'\n\n"
            "[[package]]\nname = 'vokra-vibevoice-realtime-0-5b-reference'\nversion = '0.0'\nsource = { virtual = '.' }\n\n"
            "[[package]]\nname = 'demo'\nversion = '1.0'\nsource = { registry = 'https://pypi.org/simple' }\n"
            f"sdist = []\nwheels = [{artifact_line(demo_archive, demo_url)}]\n\n"
            "[[package]]\nname = 'torch'\nversion = '2.13.0+cpu'\nsource = { registry = 'https://download.pytorch.org/whl/cpu' }\n"
            f"sdist = []\nwheels = [{artifact_line(torch_archive, torch_url)}]\n",
            encoding="utf-8",
        )
        project_sha = AUDIT.hashlib.sha256(project.read_bytes()).hexdigest()
        lock_sha = AUDIT.hashlib.sha256(lock.read_bytes()).hexdigest()
        def selected_row(name: str, version: str, archive: Path, url: str) -> dict[str, object]:
            return {"name": name, "version": version, "url": url, "filename": archive.name, "sha256": AUDIT.hashlib.sha256(archive.read_bytes()).hexdigest(), "bytes": archive.stat().st_size, "path": str(archive)}
        manifest = {"format": "vokra-vibevoice-selected-wheel-manifest-v1", "platform": {"system": "Linux", "machine": "x86_64", "python": "3.12"}, "project_sha256": project_sha, "uv_lock_sha256": lock_sha, "artifacts": [selected_row("demo", "1.0", demo_archive, demo_url), selected_row("torch", "2.13.0+cpu", torch_archive, torch_url)]}
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        args = argparse.Namespace(
            project=project,
            lock=lock,
            site_packages=site,
            venv_root=venv,
            scripts_root=scripts,
            selected_wheel_manifest=manifest_path,
            source_root=self.root / "source",
            output=self.root / "report.json",
        )
        with mock.patch.object(AUDIT.platform, "system", return_value="Linux"), mock.patch.object(AUDIT.platform, "machine", return_value="x86_64"), mock.patch.object(AUDIT.sys, "executable", str(interpreter)), mock.patch.dict(AUDIT.os.environ, {"VOKRA_PUBLISH_ON_VAST": "1"}, clear=False):
            report = AUDIT.audit(args)
        self.assertEqual(report["status"], "OWNER_REVIEW_REQUIRED_NO_UPLOAD")
        self.assertEqual(report["execution"]["publication"], "NO_UPLOAD")
        self.assertEqual(report["runtime_flags"], {
            "no_site": True,
            "dont_write_bytecode": True,
            "environment_scrub": "REQUIRED_EXTERNAL_CLEAN_BOOTSTRAP",
        })
        self.assertEqual(report["packages"][0]["lock"]["name"], "demo")
        original_inspect = AUDIT.inspect_installed_package
        def run_mutation(mutator: object) -> None:
            def mutate(*call_args: object, **call_kwargs: object) -> dict[str, object]:
                result = original_inspect(*call_args, **call_kwargs)
                if call_args[2] == "demo":
                    mutator()
                return result
            with mock.patch.object(AUDIT, "inspect_installed_package", side_effect=mutate), mock.patch.object(AUDIT.platform, "system", return_value="Linux"), mock.patch.object(AUDIT.platform, "machine", return_value="x86_64"), mock.patch.object(AUDIT.sys, "executable", str(interpreter)), mock.patch.dict(AUDIT.os.environ, {"VOKRA_PUBLISH_ON_VAST": "1"}, clear=False):
                with self.assertRaises(AUDIT.AuditError):
                    AUDIT.audit(args)

        run_mutation(lambda: (site / "demo" / "__init__.py").write_bytes(b"XXXXX"))
        (site / "demo" / "__init__.py").write_bytes(b"demo\n")
        run_mutation(lambda: (site / "late-unregistered.py").write_bytes(b"late\n"))
        (site / "late-unregistered.py").unlink()
        record = site / "demo-1.0.dist-info" / "RECORD"
        original_record = record.read_bytes()
        def mutate_source_and_coherent_record() -> None:
            mutated = b"XXXXX"
            source_path = site / "demo" / "__init__.py"
            source_path.write_bytes(mutated)
            encoded = AUDIT.base64.urlsafe_b64encode(AUDIT.hashlib.sha256(mutated).digest()).decode().rstrip("=")
            marker = b"demo/__init__.py,sha256="
            offset = original_record.index(marker) + len(marker)
            record.write_bytes(original_record[:offset] + encoded.encode() + original_record[offset + 43:])
        run_mutation(mutate_source_and_coherent_record)
        (site / "demo" / "__init__.py").write_bytes(b"demo\n")
        record.write_bytes(original_record)
        public_mutation_output = self.root / "late-mutation-report.json"
        public_argv = [
            "audit_installed_closure.py", "--project", str(project), "--lock", str(lock),
            "--site-packages", str(site), "--venv-root", str(venv), "--scripts-root", str(scripts),
            "--selected-wheel-manifest", str(manifest_path), "--source-root", str(source), "--output", str(public_mutation_output),
        ]
        def mutate_public(*call_args: object, **call_kwargs: object) -> dict[str, object]:
            result = original_inspect(*call_args, **call_kwargs)
            if call_args[2] == "demo":
                mutated = b"XXXXX"
                (site / "demo" / "__init__.py").write_bytes(mutated)
                encoded = AUDIT.base64.urlsafe_b64encode(AUDIT.hashlib.sha256(mutated).digest()).decode().rstrip("=")
                marker = b"demo/__init__.py,sha256="
                offset = original_record.index(marker) + len(marker)
                record.write_bytes(original_record[:offset] + encoded.encode() + original_record[offset + 43:])
            return result
        with mock.patch.object(AUDIT, "inspect_installed_package", side_effect=mutate_public), mock.patch.object(AUDIT.platform, "system", return_value="Linux"), mock.patch.object(AUDIT.platform, "machine", return_value="x86_64"), mock.patch.object(AUDIT.sys, "executable", str(interpreter)), mock.patch.object(sys, "argv", public_argv), mock.patch.dict(AUDIT.os.environ, {"VOKRA_PUBLISH_ON_VAST": "1"}, clear=False):
            self.assertEqual(AUDIT.main(), 2)
        self.assertFalse(public_mutation_output.exists())
        self.assertFalse(Path(str(public_mutation_output) + ".sha256").exists())
        (site / "demo" / "__init__.py").write_bytes(b"demo\n")
        record.write_bytes(original_record)

        def mutate_source() -> None:
            (source / "late-dirty.py").write_bytes(b"late\n")
        run_mutation(mutate_source)
        (source / "late-dirty.py").unlink()
        original_project = project.read_bytes()
        original_manifest = manifest_path.read_bytes()
        for label, target, body in (
            ("manifest-list", manifest_path, b"[]\n"),
            ("project-list", project, b"[]\n"),
        ):
            target.write_bytes(body)
            malformed_output = self.root / f"{label}.json"
            argv = [
                "audit_installed_closure.py", "--project", str(project), "--lock", str(lock),
                "--site-packages", str(site), "--venv-root", str(venv), "--scripts-root", str(scripts),
                "--selected-wheel-manifest", str(manifest_path), "--source-root", str(source), "--output", str(malformed_output),
            ]
            with mock.patch.object(AUDIT.platform, "system", return_value="Linux"), mock.patch.object(AUDIT.platform, "machine", return_value="x86_64"), mock.patch.object(AUDIT.sys, "executable", str(interpreter)), mock.patch.object(sys, "argv", argv), mock.patch.dict(AUDIT.os.environ, {"VOKRA_PUBLISH_ON_VAST": "1"}, clear=False):
                self.assertEqual(AUDIT.main(), 2)
            self.assertFalse(malformed_output.exists())
            project.write_bytes(original_project)
            manifest_path.write_bytes(original_manifest)

    def test_venv_invalid_utf8_and_system_site_packages_are_blocked(self) -> None:
        root = self.root / "venv-layout"
        root.mkdir()
        _, site, scripts, _, _ = AUDIT._fake_relocated_wheel(root)
        venv = root / "venv"
        (venv / "pyvenv.cfg").write_text("home = /usr/bin\nversion = 3.12.8\ninclude-system-site-packages = true\n", encoding="utf-8")
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.validate_venv_layout(venv, scripts)
        (venv / "pyvenv.cfg").write_bytes(b"home = /usr/bin\nversion = 3.12.8\ninclude-system-site-packages = false\n\xff")
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.validate_venv_layout(venv, scripts)

    def test_lock_scope_requires_cpu_torch_and_rejects_cuda_rows(self) -> None:
        wheel = {
            "url": "https://files.pythonhosted.org/packages/demo-1.0-py3-none-any.whl",
            "hash": "sha256:" + "a" * 64,
            "size": 10,
        }
        torch_wheel = {
            "url": "https://download-r2.pytorch.org/whl/cpu/torch-2.13.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl",
            "hash": "sha256:" + "b" * 64,
            "size": 10,
        }
        document = {
            "version": 1,
            "revision": 3,
            "requires-python": "==3.12.*",
            "package": [
                {"name": "vokra-vibevoice-realtime-0-5b-reference", "version": "0.0", "source": {"virtual": "."}},
                {"name": "demo", "version": "1.0", "source": {"registry": "https://pypi.org/simple"}, "sdist": [], "wheels": [wheel]},
                {"name": "torch", "version": "2.13.0+cpu", "source": {"registry": "https://download.pytorch.org/whl/cpu"}, "sdist": [], "wheels": [torch_wheel]},
            ],
        }
        self.assertEqual(set(AUDIT.validate_lock(document)), {"demo", "torch"})
        document["package"].append({"name": "nvidia-cuda-runtime", "version": "1", "source": {"registry": "https://pypi.org/simple"}, "sdist": [], "wheels": [wheel]})
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.validate_lock(document)

    def test_current_family_project_and_lock_scope_validate_without_install(self) -> None:
        family = Path(__file__).parent
        project = AUDIT.read_toml(family / "pyproject.toml", "current family project")
        lock = AUDIT.read_toml(family / "uv.lock", "current family lock")
        AUDIT.validate_project(project)
        rows = lock["package"]
        registry_names = {AUDIT.normalize_name(row["name"]) for row in rows if "registry" in row.get("source", {})}
        self.assertEqual(len(registry_names), 41)
        self.assertIn("torch", registry_names)
        self.assertFalse("triton" in registry_names or any(name.startswith("nvidia-") for name in registry_names))
        validated = AUDIT.validate_lock(lock)
        self.assertEqual(set(validated), registry_names)
        for row in rows:
            if "registry" not in row.get("source", {}):
                continue
            for wheel in row.get("wheels", []):
                self.assertRegex(wheel.get("hash", ""), r"^sha256:[0-9a-f]{64}$")
                self.assertIsInstance(wheel.get("size"), int)
                self.assertGreater(wheel["size"], 0)


if __name__ == "__main__":
    unittest.main()
