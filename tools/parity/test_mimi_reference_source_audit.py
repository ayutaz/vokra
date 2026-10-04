#!/usr/bin/env -S uv run --no-project --python 3.12 python
"""Stdlib-only fail-closed tests for the static Moshi source inventory."""

from __future__ import annotations

import gc
import json
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import mimi_reference_source_audit as audit


def _fake_checkout(root: Path) -> None:
    paths = {
        "moshi/pyproject.toml": b"[project]\nname='moshi'\nrequires-python='>=3.10,<3.15'\ndependencies=['torch>=2.2,<2.10','numpy>=1.26,<2.3']\n",
        "moshi/LICENSE": b"MIT\n",
        "moshi/LICENSE.audiocraft": b"Audiocraft attribution\n",
        "moshi/moshi/__init__.py": b"from .models import loaders\n",
        "moshi/moshi/models/loaders.py": b"import importlib as il\nfrom importlib import import_module as im\nfrom .compression import MimiModel\n__import__('literal_mod')\nil.import_module('literal_alias')\nim(name)\n",
    }
    for relative, raw in paths.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)


def _git_fixture(root: Path, *, origin: str = audit.SOURCE_URL, revision: str = audit.SOURCE_REVISION, status: str = ""):
    tracked = "\0".join(
        [
            "moshi/pyproject.toml",
            "moshi/LICENSE",
            "moshi/LICENSE.audiocraft",
            "moshi/moshi/__init__.py",
            "moshi/moshi/models/loaders.py",
        ]
    ) + "\0"

    def fake(*args: str) -> str:
        if args == ("remote", "get-url", "origin"):
            return origin
        if args == ("rev-parse", "HEAD"):
            return revision
        if args == ("status", "--porcelain=v1", "--untracked-files=all"):
            return status
        if args == ("ls-files", "-z"):
            return tracked
        raise AssertionError(args)

    return fake


def _tempdir() -> tempfile.TemporaryDirectory[str]:
    # Resolve platform temp aliases (for example macOS /tmp) without baking
    # a host-specific directory into the Linux/VAST stdlib test suite.
    temp_root = Path(tempfile.gettempdir()).resolve()
    return tempfile.TemporaryDirectory(prefix="vokra-mimi-source-audit-", dir=str(temp_root))


class StaticSourceAuditTests(unittest.TestCase):
    def test_inventory_records_declared_constraints_and_dynamic_unknown(self) -> None:
        with _tempdir() as directory:
            root = Path(directory)
            _fake_checkout(root)
            inventory = audit.collect_inventory(root, git_runner=_git_fixture(root))
            self.assertEqual(inventory["kind"], "STATIC_SOURCE_INVENTORY")
            self.assertEqual(inventory["status"], "PENDING_REVIEW_NOT_OWNER_SIGNABLE")
            self.assertIn("torch>=2.2,<2.10", inventory["declared_dependencies"]["dependencies"])
            self.assertEqual(len(inventory["literal_dynamic_imports"]), 2)
            self.assertEqual(len(inventory["unresolved_dynamic_imports"]), 1)
            self.assertNotIn("approval", inventory)

    def test_wrong_origin_rejected(self) -> None:
        with _tempdir() as directory:
            root = Path(directory)
            _fake_checkout(root)
            with self.assertRaises(ValueError):
                audit.collect_inventory(root, git_runner=_git_fixture(root, origin="https://example.invalid/moshi.git"))

    def test_wrong_revision_rejected(self) -> None:
        with _tempdir() as directory:
            root = Path(directory)
            _fake_checkout(root)
            with self.assertRaises(ValueError):
                audit.collect_inventory(root, git_runner=_git_fixture(root, revision="0" * 40))

    def test_dirty_checkout_rejected(self) -> None:
        with _tempdir() as directory:
            root = Path(directory)
            _fake_checkout(root)
            with self.assertRaises(ValueError):
                audit.collect_inventory(root, git_runner=_git_fixture(root, status=" M moshi/moshi/__init__.py"))

    def test_missing_primary_license_rejected(self) -> None:
        with _tempdir() as directory:
            root = Path(directory)
            _fake_checkout(root)
            (root / "moshi/LICENSE.audiocraft").unlink()
            with self.assertRaises(ValueError):
                audit.collect_inventory(root, git_runner=_git_fixture(root))

    def test_symlink_and_path_traversal_rejected(self) -> None:
        with _tempdir() as directory:
            root = Path(directory)
            _fake_checkout(root)
            (root / "moshi/LICENSE.audiocraft").unlink()
            (root / "moshi/LICENSE.audiocraft").symlink_to(root / "moshi/LICENSE")
            with self.assertRaises(ValueError):
                audit.collect_inventory(root, git_runner=_git_fixture(root))
            with self.assertRaises(ValueError):
                audit._split_tracked("moshi/../outside\0")

    def test_duplicate_tracked_path_rejected(self) -> None:
        with self.assertRaises(ValueError):
            audit._split_tracked("moshi/LICENSE\0moshi/LICENSE\0")
        with self.assertRaises(ValueError):
            audit._split_tracked("moshi//LICENSE\0")
        with self.assertRaises(ValueError):
            audit._split_tracked("moshi/./LICENSE\0")

    def test_symlink_parent_rejected_before_read(self) -> None:
        with _tempdir() as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaises(ValueError):
                audit._safe_absolute(alias / "checkout", "source_root")

    def test_dirty_during_read_rejected_by_final_git_readback(self) -> None:
        with _tempdir() as directory:
            root = Path(directory)
            _fake_checkout(root)
            state = {"status_calls": 0}
            fixed = _git_fixture(root)

            def changing(*args: str) -> str:
                if args == ("status", "--porcelain=v1", "--untracked-files=all"):
                    state["status_calls"] += 1
                    return "" if state["status_calls"] == 1 else " M moshi/moshi/__init__.py"
                return fixed(*args)

            with self.assertRaises(ValueError):
                audit.collect_inventory(root, git_runner=changing)
            self.assertEqual(state["status_calls"], 2)

    def test_growing_file_rejected_by_nofollow_snapshot(self) -> None:
        with _tempdir() as directory:
            path = Path(directory) / "source.py"
            path.write_bytes(b"x")
            original_fstat = audit.os.fstat
            calls = {"count": 0}

            def growing(fd: int):
                calls["count"] += 1
                result = original_fstat(fd)
                if calls["count"] == 2:
                    path.write_bytes(b"xx")
                    result = original_fstat(fd)
                return result

            with patch.object(audit.os, "fstat", side_effect=growing):
                with self.assertRaises(ValueError):
                    audit._read_regular_bytes(path, "growing source")

    def test_same_size_path_replacement_rejected(self) -> None:
        with _tempdir() as directory:
            path = Path(directory) / "source.py"
            path.write_bytes(b"x")
            original_fstat = audit.os.fstat
            calls = {"count": 0}

            def replacing(fd: int):
                calls["count"] += 1
                result = original_fstat(fd)
                if calls["count"] == 2:
                    replacement = path.with_suffix(".replacement")
                    replacement.write_bytes(b"y")
                    path.unlink()
                    replacement.rename(path)
                return result

            with patch.object(audit.os, "fstat", side_effect=replacing):
                with self.assertRaises(ValueError):
                    audit._read_regular_bytes(path, "replaced source")

    def test_git_output_limit_rejected_with_fake_subprocess(self) -> None:
        with _tempdir() as directory:
            root = Path(directory)
            fake_git = root / "git"
            fake_git.write_text("#!/bin/sh\nprintf 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'\n", encoding="utf-8")
            fake_git.chmod(0o755)
            with patch.dict(os.environ, {"PATH": f"{root}:{os.environ.get('PATH', '')}"}):
                with patch.object(audit, "MAX_GIT_OUTPUT_BYTES", 32):
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ResourceWarning)
                        with self.assertRaises(ValueError):
                            audit._git(root, "status")
                        gc.collect()
                    self.assertEqual(
                        [warning for warning in caught if warning.category is ResourceWarning],
                        [],
                    )

    def test_git_timeout_rejected_with_fake_subprocess(self) -> None:
        with _tempdir() as directory:
            root = Path(directory)
            fake_git = root / "git"
            fake_git.write_text("#!/bin/sh\nexec sleep 2\n", encoding="utf-8")
            fake_git.chmod(0o755)
            with patch.dict(os.environ, {"PATH": f"{root}:{os.environ.get('PATH', '')}"}):
                with patch.object(audit, "GIT_TIMEOUT_SECONDS", 0.05):
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ResourceWarning)
                        with self.assertRaises(ValueError):
                            audit._git(root, "status")
                        gc.collect()
                    self.assertEqual(
                        [warning for warning in caught if warning.category is ResourceWarning],
                        [],
                    )

    def test_git_pipe_wrappers_close_under_resource_warning_guard(self) -> None:
        with _tempdir() as directory:
            root = Path(directory)
            fake_git = root / "git"
            fake_git.write_text("#!/bin/sh\nprintf 'ok'\n", encoding="utf-8")
            fake_git.chmod(0o755)
            with patch.dict(os.environ, {"PATH": f"{root}:{os.environ.get('PATH', '')}"}):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ResourceWarning)
                    self.assertEqual(audit._git(root, "status"), "ok")
                    gc.collect()
                self.assertEqual(
                    [warning for warning in caught if warning.category is ResourceWarning],
                    [],
                )

    def test_temporary_replacement_is_not_cleaned_as_owned(self) -> None:
        with _tempdir() as directory:
            root = Path(directory)
            output = root / "inventory.json"
            original_link = audit.os.link
            intruder: dict[str, Path] = {}

            def replace_then_link(source: Path, destination: Path, *args: object, **kwargs: object) -> None:
                intruder["path"] = source
                source.unlink()
                source.write_bytes(b"intruder")
                original_link(source, destination, *args, **kwargs)

            with patch.object(audit.os, "link", side_effect=replace_then_link):
                with self.assertRaises(ValueError):
                    audit.publish_no_clobber(output, {"kind": "STATIC_SOURCE_INVENTORY"})
            self.assertEqual(output.read_bytes(), b"intruder")
            self.assertEqual(intruder["path"].read_bytes(), b"intruder")

    def test_oversized_file_and_output_clobber_rejected(self) -> None:
        with _tempdir() as directory:
            root = Path(directory)
            _fake_checkout(root)
            target = root / "moshi/moshi/models/loaders.py"
            target.write_bytes(b"x" * 33)
            with patch.object(audit, "MAX_FILE_BYTES", 32):
                with self.assertRaises(ValueError):
                    audit.collect_inventory(root, git_runner=_git_fixture(root))
            output = root / "inventory.json"
            value = {"kind": "STATIC_SOURCE_INVENTORY"}
            audit.publish_no_clobber(output, value)
            with self.assertRaises(FileExistsError):
                audit.publish_no_clobber(output, value)

    def test_new_output_is_not_overwritten_by_existing_symlink(self) -> None:
        with _tempdir() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_bytes(b"third-party")
            output = root / "inventory.json"
            output.symlink_to(target)
            with self.assertRaises((ValueError, FileExistsError)):
                audit.publish_no_clobber(output, {"kind": "STATIC_SOURCE_INVENTORY"})
            self.assertEqual(target.read_bytes(), b"third-party")


if __name__ == "__main__":
    unittest.main()
