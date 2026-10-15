#!/usr/bin/env -S uv run --no-project --python 3.12 python
"""Statically inspect active locked sdists before any environment sync.

This tool never imports a package, runs a build backend, installs anything, or
downloads an archive.  The caller must provide each active locked sdist as a
local file and the tool verifies its lock-bound size and SHA-256 first.  A
successful result is source inspection only, not build or model-execution
provenance.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import configparser
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
import tempfile
import tomllib
from typing import Any
from urllib.parse import unquote, urlsplit


MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 10_000
MAX_MEMBER_BYTES = 8 * 1024 * 1024
MAX_TOTAL_MEMBER_BYTES = 64 * 1024 * 1024
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_LOCK_BYTES = 8 * 1024 * 1024
TARGET = "linux-x86_64-cp312"
ALLOWED_BACKENDS = {"setuptools.build_meta", "setuptools.build_meta:__legacy__"}
ALLOWED_ARTIFACT_HOSTS = {"files.pythonhosted.org"}
BUILD_FILES = {"pyproject.toml", "setup.py", "setup.cfg", "MANIFEST.in"}
ALLOWED_SETUP_KEYWORDS = {
    "author",
    "author_email",
    "classifiers",
    "description",
    "install_requires",
    "license",
    "name",
    "package_dir",
    "packages",
    "python_requires",
    "scripts",
    "url",
    "version",
}
BAD_CALLS = {"__import__", "compile", "eval", "exec", "load", "run", "urlopen"}
BAD_TEXT = re.compile(r"(?i)(snapshot_download|hf_hub_download|from_pretrained|torch\.hub|subprocess|urlopen|requests\.)")
ALLOWED_CFG_SECTIONS = {"egg_info"}
ALLOWED_CFG_KEYS = {"tag_build", "tag_date"}
ALLOWED_MANIFEST_DIRECTIVES = {"exclude", "graft", "global-exclude", "global-include", "include", "prune", "recursive-exclude", "recursive-include"}


class InspectionError(ValueError):
    """A fail-closed source inspection error."""


def canonical_path(path: Path) -> None:
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise InspectionError(f"path is not absolute and canonical: {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            if current == Path("/var") and current.resolve() == Path("/private/var"):
                continue
            raise InspectionError(f"symlink ancestry is not allowed: {current}")


def regular_archive(path: Path) -> None:
    canonical_path(path)
    if path.is_symlink() or not path.is_file():
        raise InspectionError(f"archive is not a regular file: {path}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise InspectionError(f"archive stat failed: {type(exc).__name__}") from exc
    if size <= 0 or size > MAX_ARCHIVE_BYTES:
        raise InspectionError("archive exceeds bounded compressed size")


def regular_lock(path: Path) -> None:
    canonical_path(path)
    if path.is_symlink() or not path.is_file():
        raise InspectionError(f"lock is not a regular file: {path}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise InspectionError(f"lock stat failed: {type(exc).__name__}") from exc
    if size <= 0 or size > MAX_LOCK_BYTES:
        raise InspectionError("uv.lock exceeds bounded size")


def sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            size += len(chunk)
            if size > MAX_ARCHIVE_BYTES:
                raise InspectionError("archive exceeds bounded size")
            digest.update(chunk)
    return size, digest.hexdigest()


def safe_member(name: str) -> str:
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name or name.startswith("/"):
        raise InspectionError(f"unsafe archive member: {name!r}")
    clean = name.rstrip("/")
    parts = clean.split("/")
    if not clean or any(part in {"", ".", ".."} for part in parts):
        raise InspectionError(f"archive traversal: {name!r}")
    return clean


def wheel_matches_target(url: str) -> bool:
    parsed = urlsplit(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise InspectionError(f"wheel URL port is malformed: {url}") from exc
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.query or parsed.fragment or port not in {None, 443}:
        raise InspectionError(f"wheel URL is unsafe: {url}")
    if parsed.hostname not in ALLOWED_ARTIFACT_HOSTS or not parsed.path:
        raise InspectionError(f"wheel URL host is not an allowed PyPI artifact host: {url}")
    name = os.path.basename(unquote(parsed.path)).casefold()
    if not name.endswith(".whl"):
        raise InspectionError(f"locked wheel URL is not a wheel: {url}")
    parts = name[:-4].split("-")
    if len(parts) < 5:
        raise InspectionError(f"locked wheel filename is malformed: {url}")
    python_tags, abi_tags, platform_tags = (parts[-3].split("."), parts[-2].split("."), parts[-1].split("."))

    def compatible_platform(tag: str) -> bool:
        if tag == "any":
            return True
        if tag in {"manylinux1_x86_64", "manylinux2010_x86_64", "manylinux2014_x86_64"}:
            return True
        match = re.fullmatch(r"manylinux_2_(\d+)_x86_64", tag)
        return match is not None and int(match.group(1)) <= 35

    for python_tag in python_tags:
        for abi_tag in abi_tags:
            if python_tag == "py3" and abi_tag == "none":
                python_compatible = True
            elif python_tag == "py312" and abi_tag == "none":
                python_compatible = True
            elif python_tag.startswith("cp") and python_tag[2:].isdigit():
                version_text = python_tag[2:]
                version = int(version_text)
                python_compatible = version == 312 and abi_tag in {"cp312", "abi3"}
                if abi_tag == "abi3" and re.fullmatch(r"3(?:[2-9]|1[0-2])", version_text):
                    python_compatible = True
            else:
                python_compatible = False
            if python_compatible and any(compatible_platform(tag) for tag in platform_tags):
                return True
    return False


def _artifact(package: dict[str, Any], key: str) -> dict[str, Any]:
    value = package.get(key)
    if not isinstance(value, dict):
        raise InspectionError(f"locked {key} identity is missing for {package.get('name')}")
    url, digest, size = value.get("url"), value.get("hash"), value.get("size")
    if not isinstance(url, str) or not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise InspectionError(f"locked {key} identity is malformed for {package.get('name')}")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0 or size > MAX_ARCHIVE_BYTES:
        raise InspectionError(f"locked {key} size is malformed for {package.get('name')}")
    parsed = urlsplit(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise InspectionError(f"locked {key} URL port is malformed for {package.get('name')}") from exc
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or port not in {None, 443}
        or parsed.hostname not in ALLOWED_ARTIFACT_HOSTS
        or not parsed.path
    ):
        raise InspectionError(f"locked {key} URL is unsafe for {package.get('name')}")
    return {"url": url, "sha256": digest.removeprefix("sha256:"), "bytes": size}


def active_sdists(lock: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise InspectionError("uv.lock package table is missing")
    active: list[dict[str, Any]] = []
    optional: list[dict[str, Any]] = []
    seen_identities: set[str] = set()
    for package in packages:
        if not isinstance(package, dict) or not isinstance(package.get("name"), str) or not isinstance(package.get("version"), str):
            raise InspectionError("uv.lock package identity is malformed")
        identity = f"{package['name']}=={package['version']}"
        if identity in seen_identities:
            raise InspectionError(f"duplicate uv.lock package identity: {identity}")
        seen_identities.add(identity)
        if not isinstance(package.get("sdist"), dict):
            continue
        artifact = _artifact(package, "sdist")
        row = {"identity": identity, "artifact": artifact}
        wheels = package.get("wheels", [])
        if not isinstance(wheels, list):
            raise InspectionError(f"wheel list is malformed for {row['identity']}")
        target_wheel = False
        for item in wheels:
            if not isinstance(item, dict) or not isinstance(item.get("url"), str):
                raise InspectionError(f"wheel identity is malformed for {row['identity']}")
            target_wheel = wheel_matches_target(item["url"]) or target_wheel
        if target_wheel:
            optional.append(row)
        else:
            active.append(row)
    return sorted(active, key=lambda row: row["identity"]), sorted(optional, key=lambda row: row["identity"])


def _relative_members(path: Path, expected: dict[str, Any]) -> dict[str, bytes]:
    regular_archive(path)
    size, observed = sha256_file(path)
    if size != expected["bytes"] or observed != expected["sha256"]:
        raise InspectionError(f"archive bytes do not match lock for {expected['url']}")
    members: dict[str, bytes] = {}
    seen_names: set[str] = set()
    root: str | None = None
    total = 0
    with tarfile.open(path, mode="r:*") as archive:
        for index, info in enumerate(archive, start=1):
            if index > MAX_ARCHIVE_MEMBERS:
                raise InspectionError("archive member count exceeds bound")
            name = safe_member(info.name)
            if name in seen_names:
                raise InspectionError(f"duplicate archive member: {name}")
            seen_names.add(name)
            parts = PurePosixPath(name).parts
            if len(parts) == 1:
                if root is None:
                    root = parts[0]
                elif parts[0] != root:
                    raise InspectionError("sdist has multiple top-level directories")
                if not info.isdir():
                    raise InspectionError("sdist root member is not a directory")
                continue
            if len(parts) < 2:
                raise InspectionError("sdist member has no top-level directory")
            if root is None:
                root = parts[0]
            elif parts[0] != root:
                raise InspectionError("sdist has multiple top-level directories")
            if info.issym() or info.islnk() or info.isdev() or info.isfifo() or (not info.isdir() and not info.isfile()):
                raise InspectionError(f"archive link or special member: {name}")
            if info.size < 0 or info.size > MAX_MEMBER_BYTES:
                raise InspectionError(f"archive member exceeds bound: {name}")
            total += info.size
            if total > MAX_TOTAL_MEMBER_BYTES:
                raise InspectionError("archive uncompressed member bound exceeded")
            relative = "/".join(parts[1:])
            if info.isfile() and "/" not in relative and relative.endswith(".py") and relative not in {"setup.py"} and not relative.startswith("build"):
                raise InspectionError(f"unknown root build script: {relative}")
            if info.isfile() and (relative in BUILD_FILES or relative.startswith("build") and relative.endswith(".py")):
                if info.size > MAX_SOURCE_BYTES:
                    raise InspectionError(f"build source exceeds bound: {relative}")
                stream = archive.extractfile(info)
                if stream is None:
                    raise InspectionError(f"build source cannot be read: {relative}")
                body = stream.read(MAX_SOURCE_BYTES + 1)
                stream.close()
                if len(body) != info.size or len(body) > MAX_SOURCE_BYTES:
                    raise InspectionError(f"build source size changed: {relative}")
                members[relative] = body
    return members


def _ast_findings(relative: str, body: bytes) -> list[str]:
    try:
        text = body.decode("utf-8")
        tree = ast.parse(text, filename=relative)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise InspectionError(f"build source is not valid UTF-8 Python: {relative}") from exc
    findings: list[str] = []

    def literal(node: ast.AST) -> bool:
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return all(literal(item) for item in node.elts)
        if isinstance(node, ast.Dict):
            return all(key is not None and literal(key) and literal(value) for key, value in zip(node.keys, node.values))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "find_packages":
            return not node.args and all(keyword.arg == "exclude" and literal(keyword.value) for keyword in node.keywords)
        return False

    setup_calls = 0
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.module != "setuptools" or node.level or any(alias.asname for alias in node.names) or any(alias.name not in {"setup", "find_packages"} for alias in node.names):
                findings.append(f"{relative}: unsupported import")
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "setup":
            setup_calls += 1
            if node.value.args:
                findings.append(f"{relative}: setup positional arguments are not allowed")
            for keyword in node.value.keywords:
                if keyword.arg not in ALLOWED_SETUP_KEYWORDS or not literal(keyword.value):
                    findings.append(f"{relative}: unsupported setup keyword or value: {keyword.arg}")
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        findings.append(f"{relative}: unsupported top-level build code: {type(node).__name__}")
    if setup_calls != 1:
        findings.append(f"{relative}: expected exactly one setuptools.setup call")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
            if name in BAD_CALLS or (isinstance(node.func, ast.Attribute) and node.func.attr in BAD_CALLS):
                findings.append(f"{relative}: forbidden call {name}")
    if BAD_TEXT.search(text):
        findings.append(f"{relative}: forbidden network/model build reference")
    return findings


def _cfg_findings(relative: str, text: str) -> list[str]:
    parser = configparser.ConfigParser(strict=True, interpolation=None)
    try:
        parser.read_string(text)
    except configparser.Error as exc:
        raise InspectionError(f"{relative} is malformed") from exc
    findings: list[str] = []
    for section in parser.sections():
        if section not in ALLOWED_CFG_SECTIONS:
            findings.append(f"{relative}: unrecognized executable configuration section: {section}")
            continue
        for key in parser[section]:
            if key not in ALLOWED_CFG_KEYS:
                findings.append(f"{relative}: unrecognized executable configuration key: {section}.{key}")
    return findings


def _manifest_findings(relative: str, text: str) -> list[str]:
    findings: list[str] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        directive, *patterns = line.split()
        if directive not in ALLOWED_MANIFEST_DIRECTIVES or not patterns:
            findings.append(f"{relative}:{line_number}: unrecognized manifest directive")
    return findings


def inspect_source(identity: str, expected: dict[str, Any], archive: Path) -> dict[str, Any]:
    files = _relative_members(archive, expected)
    if "setup.py" not in files and "pyproject.toml" not in files:
        raise InspectionError(f"no recognized build entrypoint in {identity}")
    findings: list[str] = []
    source_rows: list[dict[str, Any]] = []
    build_system_requires: list[str] = []
    for relative in sorted(files):
        body = files[relative]
        if relative.endswith(".py"):
            findings.extend(_ast_findings(relative, body))
        elif relative == "pyproject.toml":
            text = body.decode("utf-8")
            if BAD_TEXT.search(text):
                findings.append(f"{relative}: forbidden network/model build reference")
        elif relative == "setup.cfg":
            text = body.decode("utf-8")
            findings.extend(_cfg_findings(relative, text))
            if BAD_TEXT.search(text):
                findings.append(f"{relative}: forbidden network/model build reference")
        elif relative == "MANIFEST.in":
            text = body.decode("utf-8")
            findings.extend(_manifest_findings(relative, text))
            if BAD_TEXT.search(text):
                findings.append(f"{relative}: forbidden network/model build reference")
        source_rows.append({"path": relative, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()})
    if "pyproject.toml" in files:
        try:
            pyproject = tomllib.loads(files["pyproject.toml"].decode("utf-8"))
        except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise InspectionError(f"pyproject.toml is malformed for {identity}") from exc
        if set(pyproject) != {"build-system"}:
            raise InspectionError(f"unrecognized dynamic pyproject configuration for {identity}")
        build = pyproject.get("build-system")
        if not isinstance(build, dict) or build.get("build-backend") not in ALLOWED_BACKENDS:
            raise InspectionError(f"unknown build backend for {identity}")
        if set(build) - {"build-backend", "backend-path", "requires"}:
            raise InspectionError(f"unrecognized dynamic build-system configuration for {identity}")
        if build.get("backend-path"):
            raise InspectionError(f"custom backend path is not statically trusted for {identity}")
        requires = build.get("requires", [])
        if not isinstance(requires, list) or any(not isinstance(item, str) for item in requires):
            raise InspectionError(f"build requirements are not statically inspectable for {identity}")
        build_system_requires = list(requires)
    if findings:
        raise InspectionError("; ".join(sorted(set(findings))))
    return {
        "identity": identity,
        "archive": expected,
        "build_sources": source_rows,
        "build_system_requires": build_system_requires,
        "build_system_requires_status": "UNEXECUTED_UNAUDITED",
        "inspection": "STATIC_ONLY_NOT_BUILD_PROVENANCE",
    }


def inspect_lock(lock_path: Path, archives: dict[str, Path]) -> dict[str, Any]:
    regular_lock(lock_path)
    try:
        lock_bytes = lock_path.read_bytes()
        if len(lock_bytes) > MAX_LOCK_BYTES:
            raise InspectionError("uv.lock exceeds bounded size")
        lock = tomllib.loads(lock_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise InspectionError(f"uv.lock cannot be read: {type(exc).__name__}") from exc
    active, optional = active_sdists(lock)
    active_ids = {row["identity"] for row in active}
    if set(archives) != active_ids:
        missing = sorted(active_ids - set(archives))
        extra = sorted(set(archives) - active_ids)
        raise InspectionError(f"archive mapping mismatch: missing={missing}, extra={extra}")
    audited = [inspect_source(row["identity"], row["artifact"], archives[row["identity"]]) for row in active]
    return {
        "schema": "vokra-xcodec2-locked-sdist-source-audit-v1",
        "target": TARGET,
        "status": "STATIC_ONLY_NOT_BUILD_PROVENANCE",
        "audit_status": "PASS",
        "publication": "NO_UPLOAD",
        "build_executed": False,
        "model_activity": {"imports": False, "weights_acquired": False, "weights_executed": False, "network": False},
        "resolution_basis": "LOCK_ARTIFACT_TAGS_ONLY_NOT_INSTALLED",
        "active_sdist_set_is_prediction": True,
        "uv_lock_markers_evaluated": False,
        "uv_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "active_sdists": audited,
        "sdists_with_target_wheel_not_selected": optional,
    }


def parse_archive_arg(value: str) -> tuple[str, Path]:
    match = re.fullmatch(r"(.+==[^=]+)=(/.*)", value)
    if match is None:
        raise InspectionError("--archive must be IDENTITY=/absolute/archive.tar.gz")
    identity, raw_path = match.groups()
    path = Path(raw_path)
    canonical_path(path)
    return identity, path


def make_test_archive(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, body in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            archive.addfile(info, io.BytesIO(body))


def self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="xcodec2-sdist-source-test-") as directory:
        root = Path(directory)
        body = b"from setuptools import setup\nsetup(name='demo')\n"
        archive = root / "demo.tar.gz"
        make_test_archive(archive, {"demo-1.0/setup.py": body, "demo-1.0/setup.cfg": b"[egg_info]\n"})
        size, digest = sha256_file(archive)
        lock = {"package": [{"name": "demo", "version": "1.0", "sdist": {"url": "https://files.pythonhosted.org/demo.tar.gz", "hash": "sha256:" + digest, "size": size}, "wheels": []}, {"name": "wheel-only", "version": "1.0", "sdist": {"url": "https://files.pythonhosted.org/wheel.tar.gz", "hash": "sha256:" + digest, "size": size}, "wheels": [{"url": "https://files.pythonhosted.org/wheel-only-1.0-py312-none-any.whl"}]}]}
        lock_path = root / "uv.lock"
        lock_path.write_text("", encoding="utf-8")
        report = inspect_lock_data_for_test(lock, {"demo==1.0": archive})
        if report["active_sdists"][0]["inspection"] != "STATIC_ONLY_NOT_BUILD_PROVENANCE" or len(report["sdists_with_target_wheel_not_selected"]) != 1:
            raise AssertionError("active/optional sdist classification failed")
        bad = root / "bad.tar.gz"
        make_test_archive(bad, {"demo-1.0/setup.py": b"import urllib.request\n"})
        bad_size, bad_digest = sha256_file(bad)
        try:
            inspect_source("demo==1.0", {"url": "https://files.pythonhosted.org/demo.tar.gz", "sha256": bad_digest, "bytes": bad_size}, bad)
        except InspectionError:
            pass
        else:
            raise AssertionError("network build reference accepted")
        unknown = root / "unknown.tar.gz"
        make_test_archive(unknown, {"demo-1.0/pyproject.toml": b"[build-system]\nbuild-backend='unknown.backend'\n"})
        unknown_size, unknown_digest = sha256_file(unknown)
        try:
            inspect_source("demo==1.0", {"url": "https://files.pythonhosted.org/demo.tar.gz", "sha256": unknown_digest, "bytes": unknown_size}, unknown)
        except InspectionError:
            pass
        else:
            raise AssertionError("unknown build backend accepted")
        traversal = root / "traversal.tar.gz"
        make_test_archive(traversal, {"demo-1.0/setup.py": body, "demo-1.0/../escape": b"x"})
        traversal_size, traversal_digest = sha256_file(traversal)
        try:
            inspect_source("demo==1.0", {"url": "https://files.pythonhosted.org/demo.tar.gz", "sha256": traversal_digest, "bytes": traversal_size}, traversal)
        except InspectionError:
            pass
        else:
            raise AssertionError("archive traversal accepted")
        duplicate = root / "duplicate.tar.gz"
        with tarfile.open(duplicate, "w:gz") as duplicate_archive:
            for duplicate_body in (b"from setuptools import setup\n", b"from setuptools import setup\n"):
                duplicate_info = tarfile.TarInfo("demo-1.0/setup.py")
                duplicate_info.size = len(duplicate_body)
                duplicate_archive.addfile(duplicate_info, io.BytesIO(duplicate_body))
        duplicate_size, duplicate_digest = sha256_file(duplicate)
        try:
            inspect_source("demo==1.0", {"url": "https://files.pythonhosted.org/demo.tar.gz", "sha256": duplicate_digest, "bytes": duplicate_size}, duplicate)
        except InspectionError:
            pass
        else:
            raise AssertionError("duplicate archive member accepted")
        unsafe = root / "unsafe.tar.gz"
        make_test_archive(unsafe, {"demo-1.0/setup.py": b"import os\nos.system('bad')\nsetup(name='demo')\n"})
        unsafe_size, unsafe_digest = sha256_file(unsafe)
        try:
            inspect_source("demo==1.0", {"url": "https://files.pythonhosted.org/demo.tar.gz", "sha256": unsafe_digest, "bytes": unsafe_size}, unsafe)
        except InspectionError:
            pass
        else:
            raise AssertionError("unsafe setup code accepted")
        if not wheel_matches_target("https://files.pythonhosted.org/pkg-1.0-cp311-abi3-manylinux_2_17_x86_64.whl"):
            raise AssertionError("compatible abi3 wheel rejected")
        if wheel_matches_target("https://files.pythonhosted.org/pkg-1.0-cp311-cp311-manylinux_2_17_x86_64.whl"):
            raise AssertionError("incompatible CPython wheel accepted")
        if wheel_matches_target("https://files.pythonhosted.org/pkg-1.0-cp312-cp312-manylinux_2_36_x86_64.whl"):
            raise AssertionError("unsupported glibc wheel accepted")
        if not wheel_matches_target("https://files.pythonhosted.org/pkg-1.0-py2.py3-none-any.whl"):
            raise AssertionError("compressed Python 2/3 wheel rejected")
        if wheel_matches_target("https://files.pythonhosted.org/pkg-1.0-py2-none-any.whl"):
            raise AssertionError("standalone Python 2 wheel accepted")
        if wheel_matches_target("https://files.pythonhosted.org/pkg-1.0-cp27-abi3-manylinux_2_17_x86_64.whl"):
            raise AssertionError("pre-ABI3 CPython wheel accepted")
        for invalid_tag in ("cp3", "cp90", "cp200", "cp313"):
            if wheel_matches_target(f"https://files.pythonhosted.org/pkg-1.0-{invalid_tag}-abi3-manylinux_2_17_x86_64.whl"):
                raise AssertionError(f"invalid CPython ABI3 wheel accepted: {invalid_tag}")
        if not wheel_matches_target("https://files.pythonhosted.org/pkg-1.0-cp32-abi3-manylinux_2_17_x86_64.whl"):
            raise AssertionError("CPython 3.2 ABI3 wheel rejected")
        if not wheel_matches_target("https://files.pythonhosted.org/pkg-1.0-cp311-abi3-manylinux_2_17_x86_64.whl"):
            raise AssertionError("CPython 3.11 ABI3 wheel rejected")
        if not wheel_matches_target("https://files.pythonhosted.org/pkg-1.0-cp312-cp312-manylinux_2_17_x86_64.whl"):
            raise AssertionError("CPython 3.12 exact wheel rejected")
        try:
            wheel_matches_target("https://untrusted.example/pkg-1.0-py3-none-any.whl")
        except InspectionError:
            pass
        else:
            raise AssertionError("untrusted wheel host accepted")
        dynamic = root / "dynamic.toml.tar.gz"
        make_test_archive(dynamic, {"demo-1.0/pyproject.toml": b"[build-system]\nbuild-backend='setuptools.build_meta'\n[tool.setuptools.dynamic]\nversion={attr='demo.__version__'}\n"})
        dynamic_size, dynamic_digest = sha256_file(dynamic)
        try:
            inspect_source("demo==1.0", {"url": "https://files.pythonhosted.org/demo.tar.gz", "sha256": dynamic_digest, "bytes": dynamic_size}, dynamic)
        except InspectionError:
            pass
        else:
            raise AssertionError("dynamic pyproject configuration accepted")
        cfg_dynamic = root / "cfg-dynamic.tar.gz"
        make_test_archive(cfg_dynamic, {"demo-1.0/setup.py": body, "demo-1.0/setup.cfg": b"[options]\ncmdclass = bad\n"})
        cfg_dynamic_size, cfg_dynamic_digest = sha256_file(cfg_dynamic)
        try:
            inspect_source("demo==1.0", {"url": "https://files.pythonhosted.org/demo.tar.gz", "sha256": cfg_dynamic_digest, "bytes": cfg_dynamic_size}, cfg_dynamic)
        except InspectionError:
            pass
        else:
            raise AssertionError("dynamic setup.cfg configuration accepted")
        oversized_lock = root / "oversized.lock"
        oversized_lock.write_bytes(b"x" * (MAX_LOCK_BYTES + 1))
        try:
            regular_lock(oversized_lock)
        except InspectionError:
            pass
        else:
            raise AssertionError("oversized uv.lock accepted")
        empty_lock = root / "empty.lock"
        empty_lock.write_text("package = []\n", encoding="utf-8")
        old_argv = sys.argv
        try:
            sys.argv = [
                str(__file__),
                "--lock",
                str(empty_lock),
                "--archive",
                f"demo==1={archive}",
                "--archive",
                f"demo==1={archive}",
            ]
            with contextlib.redirect_stderr(io.StringIO()):
                if main() != 2:
                    raise AssertionError("duplicate --archive identity accepted")
        finally:
            sys.argv = old_argv
    print("locked sdist source inspector: PASS (stdlib static self-test)")
    return 0


def inspect_lock_data_for_test(lock: dict[str, Any], archives: dict[str, Path]) -> dict[str, Any]:
    active, optional = active_sdists(lock)
    if set(archives) != {row["identity"] for row in active}:
        raise InspectionError("test archive mapping mismatch")
    return {"active_sdists": [inspect_source(row["identity"], row["artifact"], archives[row["identity"]]) for row in active], "sdists_with_target_wheel_not_selected": optional}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=Path(__file__).with_name("uv.lock"))
    parser.add_argument("--archive", action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.archive or args.output:
            parser.error("--self-test accepts no archive/output")
        return self_test()
    try:
        archives: dict[str, Path] = {}
        for value in args.archive:
            identity, archive = parse_archive_arg(value)
            if identity in archives:
                raise InspectionError(f"duplicate --archive identity: {identity}")
            archives[identity] = archive
        report = inspect_lock(args.lock, archives)
        payload = json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
        if args.output:
            canonical_path(args.output)
            if args.output.exists() or args.output.is_symlink():
                raise InspectionError("output must be a new regular path")
            if not args.output.parent.is_dir():
                raise InspectionError("output parent must be an existing directory")
            args.output.write_text(payload, encoding="utf-8")
        else:
            print(payload, end="")
        return 0
    except (InspectionError, OSError, UnicodeError, ValueError) as exc:
        print(f"locked sdist source inspector: BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
