#!/usr/bin/env -S uv run --no-project --python 3.12 python
"""Stdlib-only regression tests for the locked-sdist source inspector."""

from __future__ import annotations

import hashlib
import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))
import inspect_locked_sdist_sources as inspector  # noqa: E402


def make_archive(path: Path, members: dict[str, bytes]) -> tuple[int, str]:
    with tarfile.open(path, "w:gz") as archive:
        for name, body in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            archive.addfile(info, io.BytesIO(body))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path.stat().st_size, digest


def identity(path: Path, digest: str, size: int) -> dict[str, object]:
    return {
        "url": "https://files.pythonhosted.org/demo.tar.gz",
        "sha256": digest,
        "bytes": size,
    }


class LockedSdistSourceInspectorTests(unittest.TestCase):
    def test_active_and_optional_classification(self) -> None:
        with tempfile.TemporaryDirectory(prefix="xcodec2-inspector-test-") as directory:
            archive = Path(directory) / "demo.tar.gz"
            size, digest = make_archive(
                archive,
                {"demo-1.0/setup.py": b"from setuptools import setup\nsetup(name='demo')\n"},
            )
            lock = {
                "package": [
                    {
                        "name": "demo",
                        "version": "1.0",
                        "sdist": {"url": "https://files.pythonhosted.org/demo.tar.gz", "hash": f"sha256:{digest}", "size": size},
                        "wheels": [],
                    },
                    {
                        "name": "wheel-only",
                        "version": "1.0",
                        "sdist": {"url": "https://files.pythonhosted.org/wheel.tar.gz", "hash": f"sha256:{digest}", "size": size},
                        "wheels": [{"url": "https://files.pythonhosted.org/wheel-only-1.0-py312-none-any.whl"}],
                    },
                ]
            }
            report = inspector.inspect_lock_data_for_test(lock, {"demo==1.0": archive})
            self.assertEqual([row["identity"] for row in report["active_sdists"]], ["demo==1.0"])
            self.assertEqual([row["identity"] for row in report["sdists_with_target_wheel_not_selected"]], ["wheel-only==1.0"])
            self.assertTrue(report["active_sdists"])

    def test_archive_hash_and_source_policy_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="xcodec2-inspector-test-") as directory:
            archive = Path(directory) / "demo.tar.gz"
            size, digest = make_archive(archive, {"demo-1.0/setup.py": b"import urllib.request\n"})
            with self.assertRaises(inspector.InspectionError):
                inspector.inspect_source("demo==1.0", identity(archive, "0" * 64, size), archive)
            with self.assertRaises(inspector.InspectionError):
                inspector.inspect_source("demo==1.0", identity(archive, digest, size), archive)

    def test_unknown_backend_and_archive_traversal_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="xcodec2-inspector-test-") as directory:
            root = Path(directory)
            unknown = root / "unknown.tar.gz"
            unknown_size, unknown_digest = make_archive(
                unknown,
                {"demo-1.0/pyproject.toml": b"[build-system]\nbuild-backend='unknown.backend'\n"},
            )
            with self.assertRaises(inspector.InspectionError):
                inspector.inspect_source("demo==1.0", identity(unknown, unknown_digest, unknown_size), unknown)
            traversal = root / "traversal.tar.gz"
            traversal_size, traversal_digest = make_archive(
                traversal,
                {"demo-1.0/setup.py": b"from setuptools import setup\n", "demo-1.0/../escape": b"x"},
            )
            with self.assertRaises(inspector.InspectionError):
                inspector.inspect_source("demo==1.0", identity(traversal, traversal_digest, traversal_size), traversal)

    def test_duplicate_members_and_unsafe_setup_code_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="xcodec2-inspector-test-") as directory:
            root = Path(directory)
            duplicate = root / "duplicate.tar.gz"
            with tarfile.open(duplicate, "w:gz") as archive:
                for body in (b"from setuptools import setup\n", b"from setuptools import setup\n"):
                    info = tarfile.TarInfo("demo-1.0/setup.py")
                    info.size = len(body)
                    archive.addfile(info, io.BytesIO(body))
            duplicate_size, duplicate_digest = inspector.sha256_file(duplicate)
            with self.assertRaises(inspector.InspectionError):
                inspector.inspect_source("demo==1.0", identity(duplicate, duplicate_digest, duplicate_size), duplicate)
            unsafe = root / "unsafe.tar.gz"
            unsafe_size, unsafe_digest = make_archive(
                unsafe,
                {"demo-1.0/setup.py": b"import os\nos.system('echo bad')\nsetup(name='demo')\n"},
            )
            with self.assertRaises(inspector.InspectionError):
                inspector.inspect_source("demo==1.0", identity(unsafe, unsafe_digest, unsafe_size), unsafe)

    def test_wheel_compatibility_is_conservative(self) -> None:
        compatible = "https://files.pythonhosted.org/pkg-1.0-cp311-abi3-manylinux_2_17_x86_64.whl"
        self.assertTrue(inspector.wheel_matches_target(compatible))
        self.assertTrue(inspector.wheel_matches_target("https://files.pythonhosted.org/pkg-1.0-cp312-cp312-manylinux_2_35_x86_64.whl"))
        self.assertTrue(inspector.wheel_matches_target("https://files.pythonhosted.org/pkg-1.0-py2.py3-none-any.whl"))
        self.assertTrue(inspector.wheel_matches_target("https://files.pythonhosted.org/pkg-1.0-cp32-abi3-manylinux_2_17_x86_64.whl"))
        self.assertTrue(inspector.wheel_matches_target("https://files.pythonhosted.org/pkg-1.0-cp311-abi3-manylinux_2_17_x86_64.whl"))
        self.assertTrue(inspector.wheel_matches_target("https://files.pythonhosted.org/pkg-1.0-cp312-cp312-manylinux_2_17_x86_64.whl"))
        self.assertFalse(inspector.wheel_matches_target("https://files.pythonhosted.org/pkg-1.0-py2-none-any.whl"))
        self.assertFalse(inspector.wheel_matches_target("https://files.pythonhosted.org/pkg-1.0-cp27-abi3-manylinux_2_17_x86_64.whl"))
        for invalid_tag in ("cp3", "cp90", "cp200", "cp313"):
            self.assertFalse(inspector.wheel_matches_target(f"https://files.pythonhosted.org/pkg-1.0-{invalid_tag}-abi3-manylinux_2_17_x86_64.whl"))
        self.assertFalse(inspector.wheel_matches_target("https://files.pythonhosted.org/pkg-1.0-cp311-cp311-manylinux_2_17_x86_64.whl"))
        self.assertFalse(inspector.wheel_matches_target("https://files.pythonhosted.org/pkg-1.0-cp312-cp312-manylinux_2_36_x86_64.whl"))
        self.assertFalse(inspector.wheel_matches_target("https://files.pythonhosted.org/pkg-1.0-cp312-cp312-manylinux_2_17_aarch64.whl"))
        with self.assertRaises(inspector.InspectionError):
            inspector.wheel_matches_target("https://untrusted.example/pkg-1.0-py3-none-any.whl")

    def test_dynamic_build_configuration_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="xcodec2-inspector-test-") as directory:
            root = Path(directory)
            dynamic = root / "dynamic.toml.tar.gz"
            dynamic_size, dynamic_digest = make_archive(
                dynamic,
                {"demo-1.0/pyproject.toml": b"[build-system]\nbuild-backend='setuptools.build_meta'\n[tool.setuptools.dynamic]\nversion={attr='demo.__version__'}\n"},
            )
            with self.assertRaises(inspector.InspectionError):
                inspector.inspect_source("demo==1.0", identity(dynamic, dynamic_digest, dynamic_size), dynamic)
            cfg_dynamic = root / "cfg-dynamic.tar.gz"
            cfg_size, cfg_digest = make_archive(
                cfg_dynamic,
                {
                    "demo-1.0/setup.py": b"from setuptools import setup\nsetup(name='demo')\n",
                    "demo-1.0/setup.cfg": b"[options]\ncmdclass = bad\n",
                },
            )
            with self.assertRaises(inspector.InspectionError):
                inspector.inspect_source("demo==1.0", identity(cfg_dynamic, cfg_digest, cfg_size), cfg_dynamic)

    def test_symlinked_lock_and_existing_output_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="xcodec2-inspector-test-") as directory:
            root = Path(directory)
            lock = root / "uv.lock"
            lock.write_text("package = []\n", encoding="utf-8")
            link = root / "uv-link.lock"
            link.symlink_to(lock)
            with self.assertRaises(inspector.InspectionError):
                inspector.inspect_lock(link, {})
            output = root / "report.json"
            output.write_text("old\n", encoding="utf-8")
            old_argv = sys.argv
            try:
                sys.argv = [str(inspector.__file__), "--lock", str(lock), "--output", str(output)]
                self.assertEqual(inspector.main(), 2)
            finally:
                sys.argv = old_argv
            self.assertEqual(output.read_text(encoding="utf-8"), "old\n")


if __name__ == "__main__":
    unittest.main()
