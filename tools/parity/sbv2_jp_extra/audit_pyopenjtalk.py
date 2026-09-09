#!/usr/bin/env -S uv run --no-project --offline --python 3.12
"""Fail-closed audit for the SBV2 pyopenjtalk reference dependency.

The audit deliberately uses importlib.metadata and git plumbing only: it does
not import pyopenjtalk or execute native code.  The VAST wrapper runs it before
the official Style-Bert-VITS2 module is imported.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import io
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
import urllib.parse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping


PYOPENJTALK_COMMIT = "0f0fc44e782a8134cd9a51d80b57b48a7c95bb80"
PYOPENJTALK_TAG = "v0.4.1"
PYOPENJTALK_URL = "https://github.com/r9y9/pyopenjtalk.git"
PYOPENJTALK_SDIST_SHA256 = "d5ada46f7fc2b52c1c79c273eb9668ff6ad7ab276a8db9d8be119ef93440f0dc"
OPEN_JTALK_DICT_URL = "https://github.com/r9y9/open_jtalk/releases/download/v1.11.1/open_jtalk_dic_utf_8-1.11.tar.gz"
OPEN_JTALK_DICT_BASE_URL = "https://github.com/r9y9/open_jtalk/releases/download/v1.11.1"
OPEN_JTALK_DICT_ARCHIVE_NAME = "open_jtalk_dic_utf_8-1.11.tar.gz"
OPEN_JTALK_DICT_ROOT = "open_jtalk_dic_utf_8-1.11/"
OPEN_JTALK_DICT_ARCHIVE_SIZE = 23646843
OPEN_JTALK_DICT_ARCHIVE_SHA256 = "fe6ba0e43542cef98339abdffd903e062008ea170b04e7e2a35da805902f382a"
OPEN_JTALK_DICT_FILES = {
    "COPYING": (5865, "f4eca42ebd930e2c6e57fca58319d989bebcd1510cb7714b149c50f5425135ea"),
    "char.bin": (262496, "888ee94c5a8a7a26d24ab3f1b7155441351954fd51ea06b4a2f78bd742492b2f"),
    "left-id.def": (77672, "db1adac8a7f9e5854cd82ea044c85115249206c8181b9d88cf92ae2ee5e87b84"),
    "matrix.bin": (3792262, "62fd16b4f64c851d5dc352ef0d5740c5fc83ddc7c203b2b0b1fc5271969a14ce"),
    "pos-id.def": (1923, "3460aa742053085af47cdfc889a1e0e6f557e89b406e501ba81c9ccc286de0c7"),
    "rewrite.def": (7457, "7f7c8dfbfe24092e8a149a9b6e0a3a7f1c2cf37d6c3dc29d1cccc6c004da9c1c"),
    "right-id.def": (77672, "db1adac8a7f9e5854cd82ea044c85115249206c8181b9d88cf92ae2ee5e87b84"),
    "sys.dic": (103073776, "ca57d9029691a70a5dfb99afc2844180256161d7130da65b1a867510e129b9a6"),
    "unk.dic": (5690, "ce97851ecda075914fa3ffe7294a1ab34ee4f6d56ba6bf9197d74143b5dffbfe"),
}
LOGURU_SOURCE_URL = "https://github.com/Delgan/loguru.git"
LOGURU_TAG = "0.7.3"
LOGURU_TAG_OBJECT = "eb27ef8546577adbb88ad36b62b4eca9e9dae217"
LOGURU_COMMIT = "ae3bfd1b85b6b4a3db535f69b975687c79498be4"
LOGURU_SOURCE_BLOBS = {
    "LICENSE": "5285f420ff222526f9afa7acf507362367132f9c",
    "pyproject.toml": "4ea6eb8e860bee2582875b19ceac328ac17dc7af",
}
LOGURU_LICENSE_SHA256 = "b35d026cc7aca9d5859a02eb87ddf7a386a24c986838651bd1f283f94e003327"
LOGURU_PYPROJECT_SHA256 = "d49514866c6bb998295f2c64c561eace7c6233082d8ac19ee2ecc9c99f68754c"
LOGURU_SDIST_SHA256 = "19480589e77d47b8d85b2c827ad95d49bf31b0dcde16593892eb51dd18706eb6"
LOGURU_SDIST_SIZE = 63559
LOGURU_WHEEL_SHA256 = "31a33c10c8e1e10422bfd431aeb5d351c7cf7fa671e3c4df004162264b28220c"
LOGURU_WHEEL_SIZE = 61595
SOURCE_BLOBS = {
    "pyopenjtalk/__init__.py": "656c5089f529150828b5b6fe512b0ca942d9a3a8",
    "pyproject.toml": "9de1588afb8603b1ca9f13c3faca1f658057ba33",
    "LICENSE.md": "d66bbcca2d9f4d1f9244ea80ec5acda93dbb469b",
    ".gitmodules": "e70e7ee15d28fe99ec08bc42b19399e6ba291827",
    "pyopenjtalk/htsvoice/LICENSE_mei_normal.htsvoice": "753611721aea5b6ab7f713229c04cdbf8e63dff5",
    "pyopenjtalk/htsvoice/README.md": "6dff758c2495c69157ea84c05aff9a114eeb3f2c",
}
UPSTREAM_BUILD_REQUIREMENTS = {
    "setuptools>=64",
    "setuptools_scm>=8",
    "cython>=0.29.16",
    "cmake",
    "numpy>=1.25.0; python_version>='3.9'",
    "oldest-supported-numpy; python_version<'3.9'",
}
SUBMODULES = {
    "lib/open_jtalk": {
        "url": "https://github.com/r9y9/open_jtalk.git",
        "commit": "462fc38e7520aa89e4d32b2611749208528c901e",
        "blobs": {
            "src/COPYING": "495268369d51f7794083769e3305ef108593ab94",
            "src/mecab-naist-jdic/COPYING": "05d9789fde8883f09b0b9a814a53e6000346e964",
            "src/mecab/COPYING": "8c50c6c47472d3b190177ce754c5227091040856",
        },
    },
    "lib/hts_engine_API": {
        "url": "https://github.com/r9y9/hts_engine_API.git",
        "commit": "214e26dfb7f728ff9db39c14a59db709abcc121d",
        "blobs": {"src/COPYING": "55081f59b6f2e3ec7be3e32e72cca9ebea099671"},
    },
}
BUILD_CONSTRAINTS = {
    "setuptools==80.9.0",
    "setuptools-scm==9.2.0",
    "cython==3.1.4",
    "cmake==4.1.0",
    "numpy==2.5.2",
}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MIT_CLASSIFIER = "License :: OSI Approved :: MIT License"


class AuditError(RuntimeError):
    pass


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise AuditError(f"git command failed in {root}: {args!r}: {error}") from error
    return result.stdout.strip()


def _git_blob(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise AuditError(f"cannot read source file {path}: {error}") from error
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as error:
        raise AuditError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _tree(root: Path) -> dict[str, tuple[str, str]]:
    rows: dict[str, tuple[str, str]] = {}
    for line in _git(root, "ls-tree", "-r", "HEAD").splitlines():
        fields = line.split(None, 3)
        if len(fields) == 4:
            _mode, kind, blob, path = fields
            rows[path] = (kind, blob)
    return rows


def _require_clean_regular_checkout(root: Path, label: str, *, allow_gitlinks: set[str]) -> None:
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise AuditError(f"{label} checkout is dirty or contains untracked files")
    for line in _git(root, "ls-files", "-s").splitlines():
        fields = line.split(None, 3)
        if len(fields) != 4:
            raise AuditError(f"{label} has malformed index entry")
        mode, _object, _stage, relative = fields
        if mode == "160000":
            if relative not in allow_gitlinks:
                raise AuditError(f"{label} has an unexpected gitlink: {relative}")
            continue
        if mode not in {"100644", "100755"}:
            raise AuditError(f"{label} has a symlink or non-regular tracked path: {relative}")
        _regular(root / relative, f"{label}/{relative}")


def _regular(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise AuditError(f"{label} must be a regular file: {path}")


def _license_text(path: Path, label: str) -> str:
    _regular(path, label)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise AuditError(f"{label} is not readable UTF-8: {path}: {error}") from error
    upper = text.upper()
    if "GPL" in upper or "LGPL" in upper:
        raise AuditError(f"forbidden GPL/LGPL marker in {label}: {path}")
    return text


def _require_modified_bsd(text: str, label: str) -> None:
    upper = text.upper()
    # These clauses are the authenticated modified-BSD form used by both
    # pinned native projects; their text does not literally say "BSD".
    required = (
        "ALL RIGHTS RESERVED",
        "REDISTRIBUTION AND USE IN SOURCE AND BINARY FORMS",
        "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS",
    )
    if any(marker not in upper for marker in required):
        raise AuditError(f"{label} is not the expected modified BSD text")


def _verify_license_bytes(data: bytes, label: str) -> None:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AuditError(f"{label} is not readable UTF-8") from error
    upper = text.upper()
    if "GPL" in upper or "LGPL" in upper:
        raise AuditError(f"forbidden GPL/LGPL marker in {label}")
    _require_modified_bsd(text, label)


def verify_dictionary_archive(
    archive_path: Path,
    *,
    expected_size: int = OPEN_JTALK_DICT_ARCHIVE_SIZE,
    expected_sha256: str = OPEN_JTALK_DICT_ARCHIVE_SHA256,
    expected_root: str = OPEN_JTALK_DICT_ROOT,
    expected_files: Mapping[str, tuple[int, str]] = OPEN_JTALK_DICT_FILES,
) -> None:
    """Authenticate the fixed Open JTalk dictionary archive before extraction."""

    if archive_path.name != OPEN_JTALK_DICT_ARCHIVE_NAME:
        raise AuditError(f"dictionary archive name drifted: {archive_path.name}")
    _regular(archive_path, "Open JTalk dictionary archive")
    if archive_path.stat().st_size != expected_size:
        raise AuditError("Open JTalk dictionary archive size drifted")
    if _sha256(archive_path) != expected_sha256:
        raise AuditError("Open JTalk dictionary archive SHA-256 drifted")
    # tarfile normalizes a directory member's trailing slash, while the
    # authenticated source semantics name the root with a slash.
    root_name = expected_root.rstrip("/")
    expected_names = {root_name, *(root_name + "/" + name for name in expected_files)}
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)) or set(names) != expected_names:
                raise AuditError("Open JTalk dictionary archive member set drifted")
            for member in members:
                parts = member.name.split("/")
                if member.name.startswith("/") or "\\" in member.name or any(
                    part in {".", ".."} for part in parts
                ):
                    raise AuditError(f"unsafe Open JTalk dictionary archive path: {member.name}")
                if member.name == root_name:
                    if not member.isdir() or member.linkname:
                        raise AuditError("Open JTalk dictionary archive root is not a regular directory")
                    continue
                if not member.isreg() or member.linkname:
                    raise AuditError(f"Open JTalk dictionary archive member is not regular: {member.name}")
                name = member.name.removeprefix(root_name + "/")
                expected_file_size, expected_file_sha256 = expected_files[name]
                if member.size != expected_file_size:
                    raise AuditError(f"Open JTalk dictionary archive size drifted: {name}")
                stream = archive.extractfile(member)
                if stream is None:
                    raise AuditError(f"cannot read Open JTalk dictionary archive member: {name}")
                data = stream.read()
                if len(data) != expected_file_size or hashlib.sha256(data).hexdigest() != expected_file_sha256:
                    raise AuditError(f"Open JTalk dictionary archive payload drifted: {name}")
                if name == "COPYING":
                    _verify_license_bytes(data, "Open JTalk dictionary archive COPYING")
    except (OSError, tarfile.TarError) as error:
        raise AuditError(f"cannot read Open JTalk dictionary archive: {error}") from error


def verify_dictionary_directory(
    dictionary_dir: Path,
    *,
    expected_files: Mapping[str, tuple[int, str]] = OPEN_JTALK_DICT_FILES,
) -> None:
    """Authenticate the exact direct-file payload produced from the archive."""

    if dictionary_dir.is_symlink() or not dictionary_dir.is_dir():
        raise AuditError(f"Open JTalk dictionary payload must be a regular directory: {dictionary_dir}")
    try:
        entries = list(dictionary_dir.iterdir())
    except OSError as error:
        raise AuditError(f"cannot inspect Open JTalk dictionary payload: {error}") from error
    names = {entry.name for entry in entries}
    if len(entries) != len(names) or names != set(expected_files):
        raise AuditError("Open JTalk dictionary payload member set drifted")
    for name, (expected_size, expected_sha256) in expected_files.items():
        path = dictionary_dir / name
        _regular(path, f"Open JTalk dictionary payload/{name}")
        if path.stat().st_size != expected_size or _sha256(path) != expected_sha256:
            raise AuditError(f"Open JTalk dictionary payload drifted: {name}")
    copying = _license_text(dictionary_dir / "COPYING", "Open JTalk dictionary COPYING")
    _require_modified_bsd(copying, "Open JTalk dictionary COPYING")


def _verify_repo(
    root: Path,
    expected_commit: str,
    expected_url: str,
    label: str,
    *,
    allow_gitlinks: set[str] | None = None,
) -> dict[str, tuple[str, str]]:
    if not root.is_dir() or not (root / ".git").exists():
        raise AuditError(f"{label} is not a git checkout: {root}")
    if _git(root, "rev-parse", "HEAD") != expected_commit:
        raise AuditError(f"{label} commit is not authenticated: {_git(root, 'rev-parse', 'HEAD')}")
    actual_url = _git(root, "config", "--get", "remote.origin.url").removesuffix(".git") + ".git"
    if actual_url != expected_url:
        raise AuditError(f"{label} origin drifted: {actual_url} != {expected_url}")
    _require_clean_regular_checkout(root, label, allow_gitlinks=allow_gitlinks or set())
    return _tree(root)


def verify_source_tree(source_dir: Path) -> None:
    tree = _verify_repo(
        source_dir,
        PYOPENJTALK_COMMIT,
        PYOPENJTALK_URL,
        "pyopenjtalk source",
        allow_gitlinks=set(SUBMODULES),
    )
    try:
        tag_commit = _git(source_dir, "rev-parse", f"refs/tags/{PYOPENJTALK_TAG}^{{}}")
    except AuditError as error:
        raise AuditError(f"pyopenjtalk release tag is missing: {PYOPENJTALK_TAG}") from error
    if tag_commit != PYOPENJTALK_COMMIT:
        raise AuditError(f"pyopenjtalk release tag drifted: {tag_commit} != {PYOPENJTALK_COMMIT}")
    for path, expected in SOURCE_BLOBS.items():
        if tree.get(path) != ("blob", expected):
            raise AuditError(f"pyopenjtalk source blob drifted: {path}")
        file_path = source_dir / path
        _regular(file_path, f"pyopenjtalk {path}")
        if _git_blob(file_path) != expected:
            raise AuditError(f"pyopenjtalk worktree blob drifted: {path}")
    try:
        init_text = (source_dir / "pyopenjtalk/__init__.py").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise AuditError(f"cannot read authenticated pyopenjtalk dictionary source semantics: {error}") from error
    if OPEN_JTALK_DICT_BASE_URL not in init_text or OPEN_JTALK_DICT_ARCHIVE_NAME not in init_text:
        raise AuditError("pyopenjtalk dictionary URL/name semantics drifted")
    try:
        build_metadata = tomllib.loads((source_dir / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise AuditError(f"cannot parse authenticated pyopenjtalk build metadata: {error}") from error
    build_requires = set(build_metadata.get("build-system", {}).get("requires", []))
    if build_requires != UPSTREAM_BUILD_REQUIREMENTS:
        raise AuditError(f"pyopenjtalk build-system requirements drifted: {sorted(build_requires)}")
    selected_names = {entry.split("==", 1)[0].replace("-", "_") for entry in BUILD_CONSTRAINTS}
    for requirement in build_requires:
        name = re.match(r"^([A-Za-z0-9_-]+)", requirement)
        if name is None or name.group(1).replace("-", "_") not in selected_names:
            continue
        minimum = re.search(r">=([0-9.]+)", requirement)
        if minimum is None:
            continue
        selected = next(
            entry.split("==", 1)[1]
            for entry in BUILD_CONSTRAINTS
            if entry.split("==", 1)[0].replace("-", "_") == name.group(1).replace("-", "_")
        )
        if tuple(int(part) for part in selected.split(".")) < tuple(int(part) for part in minimum.group(1).split(".")):
            raise AuditError(f"selected build constraint does not satisfy {requirement}: {selected}")
    mit = _license_text(source_dir / "LICENSE.md", "pyopenjtalk MIT license")
    if "MIT" not in mit.upper() or "Permission is hereby granted" not in mit:
        raise AuditError("pyopenjtalk LICENSE.md is not the expected MIT text")
    voice = _license_text(
        source_dir / "pyopenjtalk/htsvoice/LICENSE_mei_normal.htsvoice",
        "mei_normal.htsvoice license",
    )
    if (
        "CREATIVE COMMONS ATTRIBUTION 3.0" not in voice.upper()
        and "CC BY 3.0" not in voice.upper()
        and "CC-BY-3.0" not in voice.upper()
    ):
        raise AuditError("mei_normal.htsvoice license is not authenticated CC-BY-3.0")
    readme = _license_text(source_dir / "pyopenjtalk/htsvoice/README.md", "voice license README")
    if "LICENSE_MEI" not in readme.upper():
        raise AuditError("voice README does not bind the bundled voice license")

    for relative, expected in SUBMODULES.items():
        submodule = source_dir / relative
        sub_tree = _verify_repo(submodule, expected["commit"], expected["url"], relative)
        parent_row = tree.get(relative)
        if parent_row != ("commit", expected["commit"]):
            raise AuditError(f"parent submodule pointer drifted: {relative}")
        for path, blob in expected["blobs"].items():
            if sub_tree.get(path) != ("blob", blob):
                raise AuditError(f"{relative} license blob drifted: {path}")
            text = _license_text(submodule / path, f"{relative}/{path}")
            _require_modified_bsd(text, f"{relative}/{path}")


def verify_loguru_source(source_dir: Path) -> None:
    """Authenticate the upstream loguru license source used by the audit."""
    tree = _verify_repo(source_dir, LOGURU_COMMIT, LOGURU_SOURCE_URL, "loguru source")
    tag_object = _git(source_dir, "rev-parse", f"refs/tags/{LOGURU_TAG}")
    if tag_object != LOGURU_TAG_OBJECT:
        raise AuditError(f"loguru release tag object drifted: {tag_object} != {LOGURU_TAG_OBJECT}")
    tag_commit = _git(source_dir, "rev-parse", f"refs/tags/{LOGURU_TAG}^{{}}")
    if tag_commit != LOGURU_COMMIT:
        raise AuditError(f"loguru release tag commit drifted: {tag_commit} != {LOGURU_COMMIT}")
    for relative, expected in LOGURU_SOURCE_BLOBS.items():
        if tree.get(relative) != ("blob", expected):
            raise AuditError(f"loguru source blob drifted: {relative}")
        path = source_dir / relative
        _regular(path, f"loguru {relative}")
        if _git_blob(path) != expected:
            raise AuditError(f"loguru worktree blob drifted: {relative}")
    try:
        metadata = tomllib.loads((source_dir / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise AuditError(f"cannot parse authenticated loguru build metadata: {error}") from error
    project = metadata.get("project", {})
    if project.get("name") != "loguru":
        raise AuditError(f"loguru source project name drifted: {project.get('name')!r}")
    license_metadata = project.get("license")
    if not isinstance(license_metadata, dict) or license_metadata.get("text") != "MIT":
        raise AuditError("loguru source pyproject license must be {text = \"MIT\"}")
    if MIT_CLASSIFIER not in set(project.get("classifiers", [])):
        raise AuditError("loguru source pyproject lacks the authenticated MIT classifier")
    license_path = source_dir / "LICENSE"
    license_text = _license_text(license_path, "loguru source MIT license")
    if _sha256(license_path) != LOGURU_LICENSE_SHA256:
        raise AuditError("loguru source LICENSE SHA-256 drifted")
    if "MIT" not in license_text.upper() or "Permission is hereby granted" not in license_text:
        raise AuditError("loguru source LICENSE is not the expected MIT text")
    if _sha256(source_dir / "pyproject.toml") != LOGURU_PYPROJECT_SHA256:
        raise AuditError("loguru source pyproject.toml SHA-256 drifted")


def verify_lock(project_dir: Path) -> None:
    try:
        project = tomllib.loads((project_dir / "pyproject.toml").read_text(encoding="utf-8"))
        lock = tomllib.loads((project_dir / "uv.lock").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise AuditError(f"cannot parse SBV2 uv metadata: {error}") from error
    constraints = set(project.get("tool", {}).get("uv", {}).get("build-constraint-dependencies", []))
    if constraints != BUILD_CONSTRAINTS:
        raise AuditError(f"build constraints are not the authenticated exact set: {sorted(constraints)}")
    dependencies = project.get("project", {}).get("dependencies", [])
    if "pyopenjtalk==0.4.1" not in dependencies or "loguru==0.7.3" not in dependencies:
        raise AuditError("SBV2 project does not pin pyopenjtalk 0.4.1 and loguru 0.7.3 directly")
    rows = [row for row in lock.get("package", []) if row.get("name") == "pyopenjtalk"]
    if len(rows) != 1:
        raise AuditError("uv.lock must contain exactly one pyopenjtalk package row")
    row = rows[0]
    if row.get("version") != "0.4.1" or row.get("source", {}).get("registry") != "https://pypi.org/simple":
        raise AuditError("uv.lock pyopenjtalk version/source drifted")
    sdist = row.get("sdist", {})
    if sdist.get("hash") != f"sha256:{PYOPENJTALK_SDIST_SHA256}" or sdist.get("size") != 1397999:
        raise AuditError("uv.lock pyopenjtalk sdist identity drifted")
    if row.get("wheels"):
        raise AuditError("pyopenjtalk unexpectedly has a wheel in the pinned lock")
    loguru_rows = [row for row in lock.get("package", []) if row.get("name") == "loguru"]
    if len(loguru_rows) != 1:
        raise AuditError("uv.lock must contain exactly one loguru package row")
    loguru = loguru_rows[0]
    if loguru.get("version") != "0.7.3" or loguru.get("source", {}).get("registry") != "https://pypi.org/simple":
        raise AuditError("uv.lock loguru version/source drifted")
    loguru_sdist = loguru.get("sdist", {})
    if loguru_sdist.get("hash") != f"sha256:{LOGURU_SDIST_SHA256}" or loguru_sdist.get("size") != LOGURU_SDIST_SIZE:
        raise AuditError("uv.lock loguru sdist identity drifted")
    wheels = loguru.get("wheels", [])
    if len(wheels) != 1 or wheels[0].get("hash") != f"sha256:{LOGURU_WHEEL_SHA256}" or wheels[0].get("size") != LOGURU_WHEEL_SIZE:
        raise AuditError("uv.lock loguru wheel identity drifted")
    manifest = lock.get("manifest", {})
    manifest_constraints = {
        row.get("name"): row.get("specifier") for row in manifest.get("build-constraints", [])
    }
    expected_manifest = {entry.split("==", 1)[0]: "==" + entry.split("==", 1)[1] for entry in BUILD_CONSTRAINTS}
    if manifest_constraints != expected_manifest:
        raise AuditError(f"uv.lock build-constraint manifest drifted: {manifest_constraints}")


def _metadata_files(dist: importlib.metadata.Distribution) -> list[str]:
    return [str(path).replace("\\", "/") for path in (dist.files or [])]


def _verify_record(dist: importlib.metadata.Distribution, label: str) -> list[str]:
    files = _metadata_files(dist)
    record_candidates = [Path(name) for name in files if Path(name).name == "RECORD"]
    if len(record_candidates) != 1:
        raise AuditError(f"installed {label} must have exactly one RECORD")
    record_path = Path(dist.locate_file(record_candidates[0]))
    try:
        record_rows = [line.split(",") for line in record_path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeDecodeError) as error:
        raise AuditError(f"cannot read {label} RECORD: {error}") from error
    record_names: set[str] = set()
    for row in record_rows:
        if len(row) != 3:
            raise AuditError(f"{label} RECORD has a malformed row")
        name = urllib.parse.unquote(row[0])
        if not name or name in record_names or name.startswith("/") or ".." in Path(name).parts:
            raise AuditError(f"{label} RECORD has an invalid/duplicate path: {name}")
        record_names.add(name)
        installed_path = Path(dist.locate_file(name))
        _regular(installed_path, f"installed {label}/{name}")
        is_record = name == str(record_candidates[0])
        if is_record:
            if row[1] or row[2]:
                raise AuditError(f"{label} RECORD row must omit its own hash and size: {name}")
            continue
        if not row[2] or not row[2].isdigit() or int(row[2]) != installed_path.stat().st_size:
            raise AuditError(f"{label} RECORD size mismatch: {name}")
        if not row[1].startswith("sha256=") or not row[1].removeprefix("sha256="):
            raise AuditError(f"{label} RECORD requires a non-empty sha256 hash: {name}")
        else:
            encoded = row[1].removeprefix("sha256=")
            actual = base64.urlsafe_b64encode(hashlib.sha256(installed_path.read_bytes()).digest()).rstrip(b"=").decode()
            if actual != encoded:
                raise AuditError(f"{label} RECORD hash mismatch: {name}")
    if record_names != set(files):
        raise AuditError(f"{label} installed file list does not match RECORD")
    return files


def _require_mit_classifier(metadata: Any, label: str) -> None:
    """Require the authenticated classifier; wheel metadata may omit License."""
    classifiers = set(metadata.get_all("Classifier", []))
    if MIT_CLASSIFIER not in classifiers:
        raise AuditError(f"{label} does not declare the authenticated MIT classifier")


def verify_installed_metadata() -> None:
    try:
        pyopenjtalk = importlib.metadata.distribution("pyopenjtalk")
        loguru = importlib.metadata.distribution("loguru")
    except importlib.metadata.PackageNotFoundError as error:
        raise AuditError(f"required distribution is not installed: {error}") from error
    if pyopenjtalk.version != "0.4.1":
        raise AuditError(f"installed pyopenjtalk version drifted: {pyopenjtalk.version}")
    if loguru.version != "0.7.3":
        raise AuditError(f"installed loguru version drifted: {loguru.version}")
    for name in ("g2p_en", "distance", "num2words"):
        try:
            importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        raise AuditError(f"forbidden GPL/LGPL distribution is installed: {name}")
    for dist, label in ((pyopenjtalk, "pyopenjtalk"), (loguru, "loguru")):
        metadata = dist.metadata
        _require_mit_classifier(metadata, label)
    files = _verify_record(pyopenjtalk, "pyopenjtalk")
    _verify_record(loguru, "loguru")
    required = {
        "pyopenjtalk/htsvoice/mei_normal.htsvoice",
        "pyopenjtalk/htsvoice/LICENSE_mei_normal.htsvoice",
    }
    if not required.issubset(files):
        raise AuditError(f"installed pyopenjtalk is missing bundled voice/license: {sorted(required - set(files))}")
    license_payloads = [
        Path(name)
        for name in files
        if Path(name).name.upper().startswith(("LICENSE", "COPYING", "NOTICE"))
    ]
    expected_license_paths = {
        Path(name)
        for name in files
        if Path(name).name.upper() == "LICENSE.MD"
    } | {Path("pyopenjtalk/htsvoice/LICENSE_mei_normal.htsvoice")}
    unexpected = [path for path in license_payloads if path not in expected_license_paths]
    if unexpected:
        raise AuditError(f"unexpected pyopenjtalk license payloads: {unexpected}")
    license_candidates = [Path(name) for name in files if Path(name).name.upper() == "LICENSE.MD"]
    if not license_candidates:
        raise AuditError("installed pyopenjtalk has no LICENSE.md payload")
    for relative in license_candidates:
        text = _license_text(Path(pyopenjtalk.locate_file(relative)), "installed pyopenjtalk MIT license")
        if "MIT" in text.upper() and "Permission is hereby granted" in text:
            break
    else:
        raise AuditError("installed pyopenjtalk LICENSE.md is not the expected MIT text")
    voice_path = Path(pyopenjtalk.locate_file("pyopenjtalk/htsvoice/LICENSE_mei_normal.htsvoice"))
    voice = _license_text(voice_path, "installed mei_normal.htsvoice license")
    if (
        "CREATIVE COMMONS ATTRIBUTION 3.0" not in voice.upper()
        and "CC BY 3.0" not in voice.upper()
        and "CC-BY-3.0" not in voice.upper()
    ):
        raise AuditError("installed mei_normal.htsvoice license is not CC-BY-3.0")

    native = [
        Path(name)
        for name in files
        if name.startswith("pyopenjtalk/") and name.endswith((".so", ".dylib", ".pyd"))
    ]
    expected_native = {"openjtalk", "htsengine"}
    native_stems = {path.name.split(".", 1)[0] for path in native}
    if native_stems != expected_native:
        raise AuditError(f"unexpected pyopenjtalk native binary inventory: {sorted(native)}")


def _write_dictionary_fixture(
    archive_path: Path,
    members: Mapping[str, tuple[str, bytes]],
) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="w:gz") as archive:
        root_info = tarfile.TarInfo(OPEN_JTALK_DICT_ROOT)
        root_info.type = tarfile.DIRTYPE
        root_info.mtime = 0
        archive.addfile(root_info)
        for name, (kind, data) in members.items():
            info = tarfile.TarInfo(OPEN_JTALK_DICT_ROOT + name)
            info.mtime = 0
            if kind == "regular":
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            elif kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = "char.bin"
                archive.addfile(info)
            else:
                raise AssertionError(f"unknown self-test archive member kind: {kind}")


def _expect_failure(label: str, operation: Any) -> None:
    try:
        operation()
    except AuditError:
        return
    raise AssertionError(f"self-test accepted forbidden case: {label}")


def self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="sbv2-pyopenjtalk-audit-") as raw:
        root = Path(raw)
        good = root / "good.txt"
        good.write_text("MIT License\nPermission is hereby granted", encoding="utf-8")
        assert "MIT" in _license_text(good, "self-test").upper()
        loguru_metadata = SimpleNamespace(
            get_all=lambda *_args: [MIT_CLASSIFIER],
            get=lambda _key: None,
        )
        # loguru 0.7.3 omits the optional License field; the authenticated
        # classifier plus the authenticated source LICENSE and installed
        # RECORD checks are enough.
        _require_mit_classifier(loguru_metadata, "self-test loguru")
        assert "MIT" in _license_text(good, "self-test loguru payload").upper()
        bad = root / "bad.txt"
        bad.write_text("GNU GPL", encoding="utf-8")
        try:
            _license_text(bad, "self-test forbidden")
        except AuditError:
            pass
        else:
            raise AssertionError("GPL self-test was accepted")
        repo = root / "checkout"
        repo.mkdir()
        (repo / "tracked.txt").write_text("clean\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.email", "audit@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.name", "audit-self-test"], check=True)
        subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", "clean"], check=True)
        _require_clean_regular_checkout(repo, "self-test", allow_gitlinks=set())
        (repo / "untracked.txt").write_text("unexpected\n", encoding="utf-8")
        try:
            _require_clean_regular_checkout(repo, "self-test dirty", allow_gitlinks=set())
        except AuditError:
            pass
        else:
            raise AssertionError("untracked source file was accepted")
        (repo / "untracked.txt").unlink()
        (repo / "tracked.txt").unlink()
        (repo / "tracked.txt").symlink_to("missing-target")
        subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", "symlink"], check=True)
        try:
            _require_clean_regular_checkout(repo, "self-test symlink", allow_gitlinks=set())
        except AuditError:
            pass
        else:
            raise AssertionError("tracked symlink was accepted")
        record_root = root / "record-fixture"
        (record_root / "pkg.dist-info").mkdir(parents=True)
        payload_path = record_root / "payload.bin"
        payload_path.write_bytes(b"payload")
        record_path = record_root / "pkg.dist-info/RECORD"
        payload_hash = base64.urlsafe_b64encode(hashlib.sha256(payload_path.read_bytes()).digest()).rstrip(b"=").decode()
        record_path.write_text(
            f"payload.bin,sha256={payload_hash},{payload_path.stat().st_size}\n"
            "pkg.dist-info/RECORD,,\n",
            encoding="utf-8",
        )
        distribution = SimpleNamespace(
            files=[Path("payload.bin"), Path("pkg.dist-info/RECORD")],
            locate_file=lambda name: record_root / str(name),
        )
        _verify_record(distribution, "self-test")
        record_path.write_text(
            "payload.bin,md5=not-allowed,7\n"
            "pkg.dist-info/RECORD,,\n",
            encoding="utf-8",
        )
        try:
            _verify_record(distribution, "self-test bad hash")
        except AuditError:
            pass
        else:
            raise AssertionError("non-sha256 RECORD hash was accepted")
        record_path.write_text(
            "payload.bin,sha256=,\n"
            "pkg.dist-info/RECORD,,\n",
            encoding="utf-8",
        )
        try:
            _verify_record(distribution, "self-test bad size")
        except AuditError:
            pass
        else:
            raise AssertionError("empty non-RECORD hash/size was accepted")
        bsd = (
            b"Copyright 2024\n"
            b"Redistribution and use in source and binary forms are permitted.\n"
            b"This software is provided by the copyright holders and contributors.\n"
            b"All rights reserved.\n"
        )
        payloads = {"COPYING": bsd}
        payloads.update({name: f"{name}\n".encode() for name in OPEN_JTALK_DICT_FILES if name != "COPYING"})
        expected_files = {
            name: (len(data), hashlib.sha256(data).hexdigest()) for name, data in payloads.items()
        }
        good_archive = root / "dictionary-good" / OPEN_JTALK_DICT_ARCHIVE_NAME
        _write_dictionary_fixture(good_archive, {name: ("regular", data) for name, data in payloads.items()})
        verify_dictionary_archive(
            good_archive,
            expected_size=good_archive.stat().st_size,
            expected_sha256=_sha256(good_archive),
            expected_files=expected_files,
        )
        good_dir = root / "dictionary-good" / OPEN_JTALK_DICT_ROOT.rstrip("/")
        good_dir.mkdir()
        for name, data in payloads.items():
            (good_dir / name).write_bytes(data)
        verify_dictionary_directory(good_dir, expected_files=expected_files)
        bad_expected = dict(expected_files)
        bad_expected["char.bin"] = (expected_files["char.bin"][0], "0" * 64)
        _expect_failure(
            "dictionary payload hash drift",
            lambda: verify_dictionary_archive(
                good_archive,
                expected_size=good_archive.stat().st_size,
                expected_sha256=_sha256(good_archive),
                expected_files=bad_expected,
            ),
        )
        traversal = dict((name, ("regular", data)) for name, data in payloads.items())
        traversal["../escape"] = ("regular", b"escape")
        traversal_archive = root / "dictionary-traversal" / OPEN_JTALK_DICT_ARCHIVE_NAME
        _write_dictionary_fixture(traversal_archive, traversal)
        _expect_failure(
            "dictionary archive traversal",
            lambda: verify_dictionary_archive(
                traversal_archive,
                expected_size=traversal_archive.stat().st_size,
                expected_sha256=_sha256(traversal_archive),
                expected_files=expected_files,
            ),
        )
        symlink_members = dict((name, ("regular", data)) for name, data in payloads.items())
        symlink_members["char.bin"] = ("symlink", b"")
        symlink_archive = root / "dictionary-symlink" / OPEN_JTALK_DICT_ARCHIVE_NAME
        _write_dictionary_fixture(symlink_archive, symlink_members)
        _expect_failure(
            "dictionary archive symlink",
            lambda: verify_dictionary_archive(
                symlink_archive,
                expected_size=symlink_archive.stat().st_size,
                expected_sha256=_sha256(symlink_archive),
                expected_files=expected_files,
            ),
        )
        extra_members = dict((name, ("regular", data)) for name, data in payloads.items())
        extra_members["extra"] = ("regular", b"extra")
        extra_archive = root / "dictionary-extra" / OPEN_JTALK_DICT_ARCHIVE_NAME
        _write_dictionary_fixture(extra_archive, extra_members)
        _expect_failure(
            "dictionary archive extra member",
            lambda: verify_dictionary_archive(
                extra_archive,
                expected_size=extra_archive.stat().st_size,
                expected_sha256=_sha256(extra_archive),
                expected_files=expected_files,
            ),
        )
        extra_dir = root / "dictionary-extra-dir"
        shutil.copytree(good_dir, extra_dir)
        (extra_dir / "extra").write_bytes(b"extra")
        _expect_failure("dictionary payload extra member", lambda: verify_dictionary_directory(extra_dir, expected_files=expected_files))
        symlink_dir = root / "dictionary-symlink-dir"
        shutil.copytree(good_dir, symlink_dir)
        (symlink_dir / "char.bin").unlink()
        (symlink_dir / "char.bin").symlink_to("missing")
        _expect_failure("dictionary payload symlink", lambda: verify_dictionary_directory(symlink_dir, expected_files=expected_files))
        bad_license_dir = root / "dictionary-license"
        shutil.copytree(good_dir, bad_license_dir)
        bad_license = b"not a license\n"
        (bad_license_dir / "COPYING").write_bytes(bad_license)
        bad_license_expected = dict(expected_files)
        bad_license_expected["COPYING"] = (len(bad_license), hashlib.sha256(bad_license).hexdigest())
        bad_license_archive = root / "dictionary-license-archive" / OPEN_JTALK_DICT_ARCHIVE_NAME
        bad_license_payloads = dict(payloads)
        bad_license_payloads["COPYING"] = bad_license
        _write_dictionary_fixture(
            bad_license_archive,
            {name: ("regular", data) for name, data in bad_license_payloads.items()},
        )
        _expect_failure(
            "dictionary archive COPYING license",
            lambda: verify_dictionary_archive(
                bad_license_archive,
                expected_size=bad_license_archive.stat().st_size,
                expected_sha256=_sha256(bad_license_archive),
                expected_files=bad_license_expected,
            ),
        )
        _expect_failure(
            "dictionary COPYING license",
            lambda: verify_dictionary_directory(bad_license_dir, expected_files=bad_license_expected),
        )
    assert HEX40.fullmatch(PYOPENJTALK_COMMIT)
    assert HEX40.fullmatch(LOGURU_TAG_OBJECT)
    assert HEX40.fullmatch(LOGURU_COMMIT)
    assert HEX64.fullmatch(LOGURU_LICENSE_SHA256)
    assert HEX64.fullmatch(LOGURU_PYPROJECT_SHA256)
    assert HEX64.fullmatch(PYOPENJTALK_SDIST_SHA256)
    assert HEX64.fullmatch(OPEN_JTALK_DICT_ARCHIVE_SHA256)
    assert HEX64.fullmatch(LOGURU_SDIST_SHA256)
    assert HEX64.fullmatch(LOGURU_WHEEL_SHA256)
    assert LOGURU_SDIST_SIZE > 0 and LOGURU_WHEEL_SIZE > 0 and OPEN_JTALK_DICT_ARCHIVE_SIZE > 0
    print("sbv2 pyopenjtalk license audit self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--project-dir", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--loguru-source-dir", type=Path)
    parser.add_argument("--dictionary-archive", type=Path)
    parser.add_argument("--dictionary-dir", type=Path)
    parser.add_argument("--phase", choices=("static", "post"), default="post")
    args = parser.parse_args()
    if args.self_test:
        if (
            args.project_dir is not None
            or args.source_dir is not None
            or args.loguru_source_dir is not None
            or args.dictionary_archive is not None
            or args.dictionary_dir is not None
            or args.phase != "post"
        ):
            parser.error("--self-test accepts no execution options")
        self_test()
        return 0
    if args.project_dir is None or args.source_dir is None or args.loguru_source_dir is None or args.dictionary_archive is None:
        parser.error(
            "execution requires --project-dir, --source-dir, --loguru-source-dir, and --dictionary-archive"
        )
    if args.phase == "static" and args.dictionary_dir is not None:
        parser.error("--dictionary-dir is only valid for --phase post")
    if args.phase == "post" and args.dictionary_dir is None:
        parser.error("--phase post requires --dictionary-dir")
    try:
        verify_lock(args.project_dir)
        verify_source_tree(args.source_dir)
        verify_loguru_source(args.loguru_source_dir)
        verify_dictionary_archive(args.dictionary_archive)
        if args.phase == "post":
            verify_dictionary_directory(args.dictionary_dir)
            verify_installed_metadata()
    except (AuditError, OSError, ValueError) as error:
        print(f"sbv2 pyopenjtalk license audit BLOCKED: {error}", file=sys.stderr)
        return 2
    if args.phase == "static":
        print(
            "STATIC_SOURCE_LOCK_LICENSE_PASS; DICTIONARY_ARCHIVE_PASS; residual=build-only archive hashes "
            "are not lock-authenticated and require VAST archive/license/native audit"
        )
    else:
        print(
            "POST_INSTALL_PAYLOAD_PASS; DICTIONARY_ARCHIVE_PASS; DICTIONARY_PAYLOAD_PASS; residual=build-only archive hashes "
            "are not lock-authenticated and require VAST archive/license/native audit"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
