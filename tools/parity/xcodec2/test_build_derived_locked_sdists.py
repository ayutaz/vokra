#!/usr/bin/env -S uv run --no-project --python 3.12 python
"""Stdlib-only tests for the bounded derived-wheel writer."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
import zipfile


sys.path.insert(0, str(Path(__file__).parent))
import build_derived_locked_sdists as builder  # noqa: E402
import inspect_locked_sdist_sources as inspector  # noqa: E402


def _archive(path: Path, files: dict[str, bytes], *, symlink: tuple[str, str] | None = None) -> tuple[int, str]:
    names: set[str] = set()
    with tarfile.open(path, "w:gz") as archive:
        paths: set[str] = set()
        for name in files:
            parts = Path(name).parts
            for index in range(1, len(parts)):
                paths.add("/".join(parts[:index]) + "/")
        for directory in sorted(paths):
            info = tarfile.TarInfo(directory)
            info.type = tarfile.DIRTYPE
            archive.addfile(info)
            names.add(directory.rstrip("/"))
        for name, body in files.items():
            if name in names:
                raise AssertionError("test fixture file/directory collision")
            info = tarfile.TarInfo(name)
            info.size = len(body)
            archive.addfile(info, io.BytesIO(body))
            names.add(name)
        if symlink is not None:
            name, target = symlink
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            archive.addfile(info)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path.stat().st_size, digest


def _antlr_files(setup: bytes | None = None) -> dict[str, bytes]:
    return {
        "antlr4-python3-runtime-4.9.3/PKG-INFO": b"Metadata-Version: 1.0\nName: antlr4-python3-runtime\nVersion: 4.9.3\n\n",
        "antlr4-python3-runtime-4.9.3/setup.py": setup or b"from setuptools import setup\nsetup(name='antlr4-python3-runtime', version='4.9.3', packages=['antlr4'], package_dir={'': 'src'}, scripts=['bin/pygrun'])\n",
        "antlr4-python3-runtime-4.9.3/setup.cfg": b"[egg_info]\ntag_build = \ntag_date = 0\n",
        "antlr4-python3-runtime-4.9.3/src/antlr4/__init__.py": b"VALUE = 1\n",
        "antlr4-python3-runtime-4.9.3/bin/pygrun": b"#!python\nprint('fixture')\n",
    }


def _xcodec_files(setup: bytes | None = None) -> dict[str, bytes]:
    return {
        "xcodec2-0.1.5/PKG-INFO": b"Metadata-Version: 2.2\nName: xcodec2\nVersion: 0.1.5\n\n",
        "xcodec2-0.1.5/LICENSE": b"fixture license bytes; copied only from authenticated source\n",
        "xcodec2-0.1.5/setup.py": setup or b"from setuptools import setup, find_packages\nsetup(name='xcodec2', version='0.1.5', packages=find_packages(exclude=['tests*', 'docs*']))\n",
        "xcodec2-0.1.5/setup.cfg": b"[egg_info]\ntag_build = \ntag_date = 0\n",
        "xcodec2-0.1.5/xcodec2/__init__.py": b"VALUE = 2\n",
        "xcodec2-0.1.5/xcodec2/vq/__init__.py": b"VALUE = 3\n",
        "xcodec2-0.1.5/xcodec2/vq/decoder.py": b"VALUE = 4\n",
        "xcodec2-0.1.5/xcodec2.egg-info/PKG-INFO": b"generated metadata must not be packaged\n",
    }


def _lock(archives: dict[str, Path]) -> Path:
    rows = []
    for identity, path in sorted(archives.items()):
        name, version = identity.split("==", 1)
        size, digest = inspector.sha256_file(path)
        rows.append({"name": name, "version": version, "sdist": {"url": f"https://files.pythonhosted.org/{name}.tar.gz", "hash": f"sha256:{digest}", "size": size}, "wheels": []})
    lock = path.parent / "uv.lock"
    # This fixture is consumed through tomllib by the production path, so use
    # the smallest valid TOML representation without invoking a TOML writer.
    lines = []
    for row in rows:
        lines.extend(["[[package]]", f"name = \"{row['name']}\"", f"version = \"{row['version']}\"", f"sdist = {{ url = \"{row['sdist']['url']}\", hash = \"{row['sdist']['hash']}\", size = {row['sdist']['size']} }}", "wheels = []", ""])
    lock.write_text("\n".join(lines), encoding="utf-8")
    return lock


def _fixture_rows(archives: dict[str, Path]) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for identity, path in archives.items():
        size, digest = inspector.sha256_file(path)
        rows[identity] = {
            "identity": identity,
            "artifact": {"url": f"https://files.pythonhosted.org/{identity}.tar.gz", "sha256": digest, "bytes": size},
        }
    return rows


def _assert_record(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        record = next(name for name in names if name.endswith(".dist-info/RECORD"))
        rows = list(csv.reader(io.StringIO(archive.read(record).decode("utf-8"))))
        by_name = {row[0]: row for row in rows}
        for name in names:
            row = by_name[name]
            if name == record:
                if row != [name, "", ""]:
                    raise AssertionError("RECORD self row is not empty")
                continue
            encoded = row[1].removeprefix("sha256=")
            observed = base64.urlsafe_b64encode(hashlib.sha256(archive.read(name)).digest()).rstrip(b"=").decode()
            if encoded != observed or row[2] != str(len(archive.read(name))):
                raise AssertionError(f"RECORD mismatch for {name}")


def run_smoke_test() -> None:
    with tempfile.TemporaryDirectory(prefix="xcodec2-derived-wheel-test-") as directory:
        root = Path(directory)
        antlr = root / "antlr4.tar.gz"
        xcodec = root / "xcodec2.tar.gz"
        _archive(antlr, _antlr_files())
        _archive(xcodec, _xcodec_files())
        archives = {"antlr4-python3-runtime==4.9.3": antlr, "xcodec2==0.1.5": xcodec}
        lock = _lock(archives)
        output_one = root / "out-one"
        output_two = root / "out-two"
        output_one.mkdir()
        output_two.mkdir()
        rows = _fixture_rows(archives)
        first = builder._build_rows_for_test(rows, archives, output_one)
        second = builder._build_rows_for_test(rows, archives, output_two)
        if [p.read_bytes() for p, _ in first] != [p.read_bytes() for p, _ in second]:
            raise AssertionError("derived wheels are not reproducible")
        for wheel, provenance in first:
            _assert_record(wheel)
            data = json.loads(provenance.read_text(encoding="utf-8"))
            if data["status"] != "DERIVED_NOT_OFFICIAL" or data["execution_policy"] != "UNAPPROVED_NO_EXECUTION":
                raise AssertionError("derived artifact status was weakened")
            with zipfile.ZipFile(wheel) as archive:
                if wheel.name.startswith("xcodec2-") and "xcodec2-0.1.5.dist-info/licenses/LICENSE" not in archive.namelist():
                    raise AssertionError("authenticated source license was not copied")
                for row in data["source_files"]:
                    if row["target"] in archive.namelist() and hashlib.sha256(archive.read(row["target"])).hexdigest() != row["sha256"]:
                        raise AssertionError("source-to-wheel hash mismatch")

        unsafe = root / "unsafe.tar.gz"
        _archive(unsafe, _xcodec_files(setup=b"import os\nos.system('bad')\n"))
        bad_archives = dict(archives)
        bad_archives["xcodec2==0.1.5"] = unsafe
        bad_lock = _lock(bad_archives)
        (root / "unsafe-out").mkdir()
        try:
            builder._build_rows_for_test(_fixture_rows(bad_archives), bad_archives, root / "unsafe-out")
        except (builder.BuildError, FileNotFoundError):
            pass
        else:
            raise AssertionError("unsafe setup source accepted")

        dynamic_cfg = root / "dynamic-cfg.tar.gz"
        dynamic_files = _xcodec_files()
        dynamic_files["xcodec2-0.1.5/setup.cfg"] = b"[options]\ncmdclass = unsafe\n"
        _archive(dynamic_cfg, dynamic_files)
        dynamic_archives = dict(archives)
        dynamic_archives["xcodec2==0.1.5"] = dynamic_cfg
        dynamic_lock = _lock(dynamic_archives)
        (root / "dynamic-out").mkdir()
        try:
            builder._build_rows_for_test(_fixture_rows(dynamic_archives), dynamic_archives, root / "dynamic-out")
        except builder.BuildError:
            pass
        else:
            raise AssertionError("dynamic setup.cfg accepted")

        collision = root / "collision.tar.gz"
        with tarfile.open(collision, "w:gz") as archive:
            info = tarfile.TarInfo("xcodec2-0.1.5/xcodec2")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
            info = tarfile.TarInfo("xcodec2-0.1.5/xcodec2/")
            info.type = tarfile.DIRTYPE
            archive.addfile(info)
        collision_digest = hashlib.sha256(collision.read_bytes()).hexdigest()
        try:
            builder._read_archive(collision, {"bytes": collision.stat().st_size, "sha256": collision_digest})
        except builder.BuildError:
            pass
        else:
            raise AssertionError("file/directory collision accepted")

        symlink = root / "symlink.tar.gz"
        _archive(symlink, _xcodec_files(), symlink=("xcodec2-0.1.5/xcodec2/link.py", "../escape"))
        symlink_digest = hashlib.sha256(symlink.read_bytes()).hexdigest()
        try:
            builder._read_archive(symlink, {"bytes": symlink.stat().st_size, "sha256": symlink_digest})
        except builder.BuildError:
            pass
        else:
            raise AssertionError("archive symlink accepted")

        multiroot = root / "multiroot.tar.gz"
        with tarfile.open(multiroot, "w:gz") as archive:
            for name in ("one/", "two/"):
                info = tarfile.TarInfo(name)
                info.type = tarfile.DIRTYPE
                archive.addfile(info)
        multiroot_digest = hashlib.sha256(multiroot.read_bytes()).hexdigest()
        try:
            builder._read_archive(multiroot, {"bytes": multiroot.stat().st_size, "sha256": multiroot_digest})
        except builder.BuildError:
            pass
        else:
            raise AssertionError("multiple archive roots accepted")

        oversized = root / "oversized.tar.gz"
        with tarfile.open(oversized, "w:gz") as archive:
            info = tarfile.TarInfo("xcodec2-0.1.5/")
            info.type = tarfile.DIRTYPE
            archive.addfile(info)
            info = tarfile.TarInfo("xcodec2-0.1.5/payload.bin")
            info.size = inspector.MAX_MEMBER_BYTES + 1
            archive.addfile(info, io.BytesIO(b"x" * info.size))
        oversized_digest = hashlib.sha256(oversized.read_bytes()).hexdigest()
        try:
            builder._read_archive(oversized, {"bytes": oversized.stat().st_size, "sha256": oversized_digest})
        except builder.BuildError:
            pass
        else:
            raise AssertionError("oversized archive member accepted")

        partial = root / "partial-out"
        partial.mkdir()
        existing = partial / "xcodec2-0.1.5-py3-none-any.whl"
        existing.write_bytes(b"do not overwrite")
        try:
            builder._build_rows_for_test(rows, archives, partial)
        except builder.BuildError:
            pass
        else:
            raise AssertionError("existing output accepted")
        if existing.read_bytes() != b"do not overwrite":
            raise AssertionError("existing output was overwritten")


class DerivedLockedSdistTests(unittest.TestCase):
    def test_reproducibility_record_and_source_equality(self) -> None:
        run_smoke_test()

    def test_fixed_pin_rejection_and_bounded_lock(self) -> None:
        with tempfile.TemporaryDirectory(prefix="xcodec2-derived-pin-test-") as directory:
            root = Path(directory)
            antlr = root / "antlr4.tar.gz"
            xcodec = root / "xcodec2.tar.gz"
            _archive(antlr, _antlr_files())
            _archive(xcodec, _xcodec_files())
            archives = {"antlr4-python3-runtime==4.9.3": antlr, "xcodec2==0.1.5": xcodec}
            output = root / "out"
            output.mkdir()
            with self.assertRaises(builder.BuildError):
                builder.build(_lock(archives), archives, output)
            oversized = root / "oversized.lock"
            oversized.write_bytes(b"x" * (inspector.MAX_LOCK_BYTES + 1))
            with self.assertRaises(builder.BuildError):
                builder.build(oversized, archives, output)

    def test_dependency_hash_binding_and_csv_record_escaping(self) -> None:
        with tempfile.TemporaryDirectory(prefix="xcodec2-derived-record-test-") as directory:
            root = Path(directory)
            antlr = root / "antlr4.tar.gz"
            xcodec = root / "xcodec2.tar.gz"
            _archive(antlr, _antlr_files())
            _archive(xcodec, _xcodec_files())
            archives = {"antlr4-python3-runtime==4.9.3": antlr, "xcodec2==0.1.5": xcodec}
            output = root / "out"
            output.mkdir()
            outputs = builder._build_rows_for_test(_fixture_rows(archives), archives, output)
            data = json.loads(outputs[0][1].read_text(encoding="utf-8"))
            dependency = data["generator_dependencies"][0]
            size, digest = builder._hash_file(Path(dependency["path"]))
            self.assertEqual(dependency["bytes"], size)
            self.assertEqual(dependency["sha256"], digest)
            generator = data["generator"]
            generator_size, generator_digest = builder._hash_file(Path(generator["path"]))
            self.assertEqual(generator["bytes"], generator_size)
            self.assertEqual(generator["sha256"], generator_digest)
            wheel, _ = builder._zip_bytes(
                {"pkg/a,b.py": b"a", 'pkg/quote".py': b"b", "pkg/new\nline.py": b"c"},
                b"Metadata-Version: 2.2\nName: demo\nVersion: 1\n\n",
                "demo-1.dist-info",
            )
            with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
                record = next(name for name in archive.namelist() if name.endswith("/RECORD"))
                rows = list(csv.reader(io.StringIO(archive.read(record).decode("utf-8"))))
                self.assertEqual({row[0] for row in rows}, set(archive.namelist()))

    def test_metadata_identity_and_ambiguous_mapping_rejection(self) -> None:
        with tempfile.TemporaryDirectory(prefix="xcodec2-derived-metadata-test-") as directory:
            root = Path(directory)
            output = root / "out"
            output.mkdir()
            antlr = root / "antlr4.tar.gz"
            xcodec = root / "xcodec2.tar.gz"
            _archive(xcodec, _xcodec_files())
            _archive(antlr, _antlr_files(setup=b"from setuptools import setup\nsetup(name='antlr4-python3-runtime', name='duplicate', version='4.9.3', packages=['antlr4'], scripts=['bin/pygrun'])\n"))
            archives = {"antlr4-python3-runtime==4.9.3": antlr, "xcodec2==0.1.5": xcodec}
            with self.assertRaisesRegex(builder.BuildError, "duplicate setup keyword"):
                builder._build_rows_for_test(_fixture_rows(archives), archives, output)

            # Restore the valid ANTLR input so the following XCodec2 cases fail
            # at the mapping under test rather than at the earlier fixture.
            _archive(antlr, _antlr_files())
            bad_antlr = root / "bad-package-dir.tar.gz"
            _archive(bad_antlr, _antlr_files(setup=b"from setuptools import setup\nsetup(name='antlr4-python3-runtime', version='4.9.3', packages=['antlr4'], package_dir={'': 'wrong'}, scripts=['bin/pygrun'])\n"))
            with self.assertRaisesRegex(builder.BuildError, "package_dir mapping"):
                builder._build_rows_for_test(
                    {"antlr4-python3-runtime==4.9.3": {"identity": "antlr4-python3-runtime==4.9.3", "artifact": {"url": "fixture", "bytes": bad_antlr.stat().st_size, "sha256": hashlib.sha256(bad_antlr.read_bytes()).hexdigest()}}, "xcodec2==0.1.5": _fixture_rows(archives)["xcodec2==0.1.5"]},
                    {"antlr4-python3-runtime==4.9.3": bad_antlr, "xcodec2==0.1.5": xcodec},
                    output,
                )
            missing = root / "missing-name.tar.gz"
            missing_files = _xcodec_files()
            missing_files["xcodec2-0.1.5/PKG-INFO"] = b"Metadata-Version: 2.2\nVersion: 0.1.5\n\n"
            _archive(missing, missing_files)
            missing_archives = {"antlr4-python3-runtime==4.9.3": antlr, "xcodec2==0.1.5": missing}
            with self.assertRaisesRegex(builder.BuildError, "requires exactly one Name header"):
                builder._build_rows_for_test(_fixture_rows(missing_archives), missing_archives, output)
            missing_root = root / "missing-root-init.tar.gz"
            missing_root_files = _xcodec_files()
            del missing_root_files["xcodec2-0.1.5/xcodec2/__init__.py"]
            _archive(missing_root, missing_root_files)
            missing_root_archives = {"antlr4-python3-runtime==4.9.3": antlr, "xcodec2==0.1.5": missing_root}
            with self.assertRaisesRegex(builder.BuildError, "root package lacks __init__"):
                builder._build_rows_for_test(_fixture_rows(missing_root_archives), missing_root_archives, output)
            parent_missing = root / "missing-parent-init.tar.gz"
            parent_files = _xcodec_files()
            del parent_files["xcodec2-0.1.5/xcodec2/vq/__init__.py"]
            _archive(parent_missing, parent_files)
            parent_archives = {"antlr4-python3-runtime==4.9.3": antlr, "xcodec2==0.1.5": parent_missing}
            with self.assertRaisesRegex(builder.BuildError, "outside discovered package"):
                builder._build_rows_for_test(_fixture_rows(parent_archives), parent_archives, output)

    def test_archive_structure_failures_are_named(self) -> None:
        with tempfile.TemporaryDirectory(prefix="xcodec2-derived-archive-test-") as directory:
            root = Path(directory)
            def reject(path: Path, pattern: str) -> None:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.assertRaisesRegex(builder.BuildError, pattern):
                    builder._read_archive(path, {"bytes": path.stat().st_size, "sha256": digest})

            symlink = root / "symlink.tar.gz"
            _archive(symlink, _xcodec_files(), symlink=("xcodec2-0.1.5/xcodec2/link.py", "../escape"))
            reject(symlink, "link or special")

            duplicate = root / "duplicate.tar.gz"
            with tarfile.open(duplicate, "w:gz") as archive:
                directory = tarfile.TarInfo("xcodec2-0.1.5/")
                directory.type = tarfile.DIRTYPE
                archive.addfile(directory)
                for body in (b"one", b"two"):
                    info = tarfile.TarInfo("xcodec2-0.1.5/data.txt")
                    info.size = len(body)
                    archive.addfile(info, io.BytesIO(body))
            reject(duplicate, "duplicate archive member")

            traversal = root / "traversal.tar.gz"
            with tarfile.open(traversal, "w:gz") as archive:
                info = tarfile.TarInfo("../escape")
                info.size = 1
                archive.addfile(info, io.BytesIO(b"x"))
            reject(traversal, "archive traversal")

            special = root / "special.tar.gz"
            with tarfile.open(special, "w:gz") as archive:
                directory = tarfile.TarInfo("xcodec2-0.1.5/")
                directory.type = tarfile.DIRTYPE
                archive.addfile(directory)
                fifo = tarfile.TarInfo("xcodec2-0.1.5/pipe")
                fifo.type = tarfile.FIFOTYPE
                archive.addfile(fifo)
            reject(special, "link or special")

            multiroot = root / "multiroot.tar.gz"
            with tarfile.open(multiroot, "w:gz") as archive:
                for name in ("one/", "two/"):
                    info = tarfile.TarInfo(name)
                    info.type = tarfile.DIRTYPE
                    archive.addfile(info)
            reject(multiroot, "multiple top-level roots")

            collision = root / "collision.tar.gz"
            with tarfile.open(collision, "w:gz") as archive:
                directory = tarfile.TarInfo("xcodec2-0.1.5/")
                directory.type = tarfile.DIRTYPE
                archive.addfile(directory)
                info = tarfile.TarInfo("xcodec2-0.1.5/pkg")
                info.size = 1
                archive.addfile(info, io.BytesIO(b"x"))
                info = tarfile.TarInfo("xcodec2-0.1.5/pkg/child.py")
                info.size = 1
                archive.addfile(info, io.BytesIO(b"x"))
            reject(collision, "file-directory collision")

    def test_partial_rollback_preserves_existing_output(self) -> None:
        with tempfile.TemporaryDirectory(prefix="xcodec2-derived-rollback-test-") as directory:
            root = Path(directory)
            antlr = root / "antlr4.tar.gz"
            xcodec = root / "xcodec2.tar.gz"
            _archive(antlr, _antlr_files())
            _archive(xcodec, _xcodec_files())
            archives = {"antlr4-python3-runtime==4.9.3": antlr, "xcodec2==0.1.5": xcodec}
            output = root / "out"
            output.mkdir()
            existing = output / "xcodec2-0.1.5-py3-none-any.whl"
            existing.write_bytes(b"preserve")
            with self.assertRaises(builder.BuildError):
                builder._build_rows_for_test(_fixture_rows(archives), archives, output)
            self.assertEqual(existing.read_bytes(), b"preserve")
            self.assertFalse((output / "antlr4_python3_runtime-4.9.3-py3-none-any.whl").exists())


if __name__ == "__main__":
    unittest.main()
