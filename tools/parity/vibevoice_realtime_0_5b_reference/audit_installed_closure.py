#!/usr/bin/env -S uv run --no-project --python 3.12 python
"""Bounded, fail-closed audit of the locked VibeVoice reference closure.

This collector is deliberately an evidence collector, not an approval gate.
It reads an already prepared Linux/CPython 3.12 environment and an explicit
selected-wheel manifest.  It never downloads, installs, imports, or executes a
third-party package or model.  The selected manifest is untrusted input: every
wheel is rebound to the current lock and to the archive bytes on disk before
any license/native fact is reported.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
import hashlib
import importlib.util
import io
import json
import os
import platform
import re
import stat
import subprocess
import sys
import tempfile
import tomllib
import unittest
import urllib.parse
import zipfile
from pathlib import Path
from typing import Any
from unittest import mock


FORMAT = "vokra-vibevoice-realtime-installed-closure-evidence-v2"
STATUS = "OWNER_REVIEW_REQUIRED_NO_UPLOAD"
PROJECT_NAME = "vokra-vibevoice-realtime-0-5b-reference"
PYTHON_REQUIREMENT = "==3.12.*"
OFFICIAL_SOURCE_REPOSITORY = "https://github.com/microsoft/VibeVoice.git"
REGISTRY_PREFIXES = (
    "https://pypi.org/simple",
    "https://download.pytorch.org/whl/cpu",
    "https://download.pytorch.org/whl",
)
ARTIFACT_PREFIXES = (
    ("files.pythonhosted.org", "/"),
    ("download-r2.pytorch.org", "/"),
    ("download.pytorch.org", "/whl/"),
)
CPU_TORCH_VERSION = "2.13.0+cpu"
LICENSE_BASENAMES = frozenset(
    {
        "license",
        "license.txt",
        "license.md",
        "licence",
        "licence.txt",
        "licence.md",
        "copying",
        "copying.txt",
        "notice",
        "notice.txt",
    }
)
NATIVE_SUFFIXES = (".so", ".dylib", ".pyd", ".dll")
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_METADATA_BYTES = 2 * 1024 * 1024
MAX_RECORD_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
MAX_ARCHIVE_MEMBER_BYTES = 1024 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 100_000
MAX_ARCHIVE_UNPACKED_BYTES = 8 * 1024 * 1024 * 1024
MAX_HASH_BYTES_PER_FILE = 1024 * 1024 * 1024
MAX_TOTAL_HASH_BYTES = 16 * 1024 * 1024 * 1024
MAX_TOTAL_READ_BYTES = 16 * 1024 * 1024 * 1024
MAX_DIRECTORY_ENTRIES = 100_000
MAX_RECORD_ROWS = 100_000
MAX_WHEEL_FILENAME_BYTES = 512
MAX_EXPANDED_WHEEL_TAGS = 128
_HASH_BYTES_USED = 0
_READ_BYTES_USED = 0
VENV_SCRIPT_BOOTSTRAP = frozenset({"activate", "activate.csh", "activate.fish", "activate.nu", "activate.ps1", "activate_this.py", "python", "python3", "python3.12"})
VENV_SITE_BOOTSTRAP = frozenset({"_virtualenv.py", "_virtualenv.pth"})


class AuditError(ValueError):
    """The supplied closure cannot be authenticated fail-closed."""


def fail(message: str) -> None:
    raise AuditError(message)


def normalize_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def require_absolute(path: Path, label: str) -> Path:
    if not path.is_absolute():
        fail(f"{label} must be absolute")
    return path


def require_directory_chain(path: Path, label: str) -> None:
    require_absolute(path, label)
    current = path
    while True:
        try:
            info = current.lstat()
        except OSError as error:
            fail(f"{label} is inaccessible: {current}: {error}")
        if stat.S_ISLNK(info.st_mode):
            fail(f"{label} traverses a symlink: {current}")
        if not stat.S_ISDIR(info.st_mode):
            fail(f"{label} is not a directory: {current}")
        if current == current.parent:
            return
        current = current.parent


def snapshot(path: Path, label: str) -> tuple[int, int, int, int, int, int]:
    try:
        info = path.lstat()
    except OSError as error:
        fail(f"{label} is inaccessible: {error}")
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        fail(f"{label} is not a regular non-symlink file")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_nlink)


def read_bounded(path: Path, limit: int, label: str) -> bytes:
    global _READ_BYTES_USED
    before = snapshot(path, label)
    if before[2] > limit:
        fail(f"{label} exceeds bounded read ({before[2]} > {limit})")
    if _READ_BYTES_USED + before[2] > MAX_TOTAL_READ_BYTES:
        fail(f"read budget exceeded before reading {label}")
    chunks: list[bytes] = []
    total = 0
    try:
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(min(1 << 20, limit + 1 - total))
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    fail(f"{label} exceeded bounded read")
                chunks.append(chunk)
    except OSError as error:
        fail(f"{label} could not be read: {error}")
    after = snapshot(path, label)
    if after != before:
        fail(f"{label} changed while being read")
    _READ_BYTES_USED += total
    return b"".join(chunks)


def read_with_identity(path: Path, limit: int, label: str) -> tuple[bytes, dict[str, Any]]:
    """Read and hash one stable byte snapshot; parsing must use these bytes."""
    global _READ_BYTES_USED
    before = snapshot(path, label)
    if before[2] > limit:
        fail(f"{label} exceeds bounded read ({before[2]} > {limit})")
    if _READ_BYTES_USED + before[2] > MAX_TOTAL_READ_BYTES:
        fail(f"read budget exceeded before reading {label}")
    chunks: list[bytes] = []
    total = 0
    try:
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(min(1 << 20, limit + 1 - total))
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    fail(f"{label} exceeded bounded read")
                chunks.append(chunk)
    except OSError as error:
        fail(f"{label} could not be read: {error}")
    after = snapshot(path, label)
    if after != before or total != before[2]:
        fail(f"{label} changed during bounded snapshot")
    _READ_BYTES_USED += total
    body = b"".join(chunks)
    digest = hashlib.sha256(body).hexdigest()
    if len(body) != before[2]:
        fail(f"{label} size changed after bounded read")
    return body, {"path": str(path), "bytes": len(body), "sha256": digest, "dev": before[0], "ino": before[1], "mtime_ns": before[3], "ctime_ns": before[4], "nlink": before[5]}


def hash_bounded(path: Path, limit: int, label: str) -> tuple[int, str]:
    global _HASH_BYTES_USED, _READ_BYTES_USED
    before = snapshot(path, label)
    if before[2] > limit:
        fail(f"{label} exceeds bounded hash ({before[2]} > {limit})")
    if _HASH_BYTES_USED + before[2] > MAX_TOTAL_HASH_BYTES:
        fail(f"hash budget exceeded before reading {label}")
    if _READ_BYTES_USED + before[2] > MAX_TOTAL_READ_BYTES:
        fail(f"read budget exceeded before hashing {label}")
    digest = hashlib.sha256()
    total = 0
    try:
        with path.open("rb") as stream:
            while True:
                block = stream.read(1 << 20)
                if not block:
                    break
                total += len(block)
                if total > limit:
                    fail(f"{label} exceeded bounded hash")
                digest.update(block)
    except OSError as error:
        fail(f"{label} could not be hashed: {error}")
    after = snapshot(path, label)
    if after != before or total != before[2]:
        fail(f"{label} changed while being hashed")
    _HASH_BYTES_USED += total
    _READ_BYTES_USED += total
    return total, digest.hexdigest()


def file_identity(path: Path, label: str, limit: int = MAX_METADATA_BYTES) -> dict[str, Any]:
    before = snapshot(path, label)
    size, digest = hash_bounded(path, limit, label)
    after = snapshot(path, label)
    if after != before:
        fail(f"{label} changed during identity capture")
    return {"path": str(path), "bytes": size, "sha256": digest, "dev": before[0], "ino": before[1], "mtime_ns": before[3], "ctime_ns": before[4], "nlink": before[5]}


def read_json(path: Path, limit: int, label: str) -> dict[str, Any]:
    try:
        value = json.loads(read_bounded(path, limit, label))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"{label} is not bounded UTF-8 JSON: {error}")
    if not isinstance(value, dict):
        fail(f"{label} must be a JSON object")
    return value


def read_toml(path: Path, label: str) -> dict[str, Any]:
    try:
        value = tomllib.loads(read_bounded(path, MAX_METADATA_BYTES, label).decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        fail(f"{label} is invalid TOML: {error}")
    return value


def parse_json_bytes(body: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"{label} is not bounded UTF-8 JSON: {error}")
    if not isinstance(value, dict):
        fail(f"{label} must be a JSON object")
    return value


def parse_toml_bytes(body: bytes, label: str) -> dict[str, Any]:
    try:
        value = tomllib.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        fail(f"{label} is invalid TOML: {error}")
    if not isinstance(value, dict):
        fail(f"{label} must be a TOML table")
    return value


def validate_project(document: dict[str, Any]) -> None:
    project = document.get("project")
    if not isinstance(project, dict) or project.get("name") != PROJECT_NAME:
        fail("reference project name mismatch")
    if project.get("requires-python") != PYTHON_REQUIREMENT:
        fail("reference project is not pinned to Python 3.12")
    dependencies = project.get("dependencies")
    if not isinstance(dependencies, list) or any(not isinstance(item, str) for item in dependencies):
        fail("reference project dependencies are malformed")
    expected_torch_dependency = "torch==2.13.0 ; platform_machine == 'x86_64' and sys_platform == 'linux'"
    torch_dependencies = [item for item in dependencies if re.match(r"^torch(?:[<=> ;]|$)", item)]
    if torch_dependencies != [expected_torch_dependency]:
        fail("project Torch dependency is not the exact Linux CPU declaration")
    tool = document.get("tool")
    if not isinstance(tool, dict):
        fail("project tool table is malformed")
    tool_uv = tool.get("uv")
    if not isinstance(tool_uv, dict) or tool_uv.get("package") is not False:
        fail("reference project must remain non-package")
    environments = tool_uv.get("environments")
    if not isinstance(environments, list) or "sys_platform == 'linux' and platform_machine == 'x86_64'" not in environments:
        fail("reference project must retain Linux x86_64 scope")
    sources = tool_uv.get("sources")
    if not isinstance(sources, dict) or sources.get("torch") != {"index": "pytorch-cpu"}:
        fail("project Torch source is not the reviewed explicit index")
    indexes = tool_uv.get("index")
    if indexes != [{"name": "pytorch-cpu", "url": "https://download.pytorch.org/whl/cpu", "explicit": True}]:
        fail("project lacks the exact explicit PyTorch CPU index")
    reference = tool.get("vokra", {}).get("reference") if isinstance(tool.get("vokra"), dict) else None
    if not isinstance(reference, dict) or reference.get("torch_distribution") != CPU_TORCH_VERSION or reference.get("torch_index") != "https://download.pytorch.org/whl/cpu":
        fail("project reference Torch distribution/index facts are not exact")


def as_artifact_list(row: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = row.get(key, [])
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        fail(f"lock {key} artifacts are malformed for {row.get('name')}")
    return value


def lock_artifact_filename(url: str) -> str:
    parsed = strict_https_url(url, "artifact URL")
    filename = urllib.parse.unquote(parsed.path.rsplit("/", 1)[-1])
    if not filename:
        fail("lock artifact URL has no filename")
    return filename


def strict_https_url(url: str, label: str) -> urllib.parse.SplitResult:
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError as error:
        fail(f"{label} is malformed: {error}")
    if parsed.scheme != "https" or parsed.username or parsed.password or port is not None or parsed.query or parsed.fragment or not parsed.hostname:
        fail(f"{label} is not strict HTTPS: {url!r}")
    return parsed


def validate_lock(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if document.get("version") != 1 or document.get("revision") != 3:
        fail("uv.lock schema/version drifted")
    if document.get("requires-python") != PYTHON_REQUIREMENT:
        fail("uv.lock Python requirement drifted")
    rows = document.get("package")
    if not isinstance(rows, list) or not rows:
        fail("uv.lock has no package rows")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not isinstance(row.get("version"), str):
            fail("malformed uv.lock package row")
        name = row["name"]
        normalized = normalize_name(name)
        if normalized in result:
            fail(f"duplicate uv.lock package row: {name}")
        if normalized == "triton" or normalized.startswith("nvidia-"):
            fail(f"CUDA/NVIDIA package is outside the CPU audit: {name}")
        source = row.get("source")
        if not isinstance(source, dict) or len(source) != 1 or set(source) not in ({"registry"}, {"virtual"}):
            fail(f"malformed uv.lock source for {name}")
        if "registry" in source:
            registry = source["registry"]
            if not isinstance(registry, str) or not any(registry == candidate for candidate in REGISTRY_PREFIXES):
                fail(f"unapproved registry source for {name}: {registry!r}")
            for key in ("sdist", "wheels"):
                for artifact in as_artifact_list(row, key):
                    digest = str(artifact.get("hash", ""))
                    url = artifact.get("url")
                    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
                        fail(f"lock artifact hash missing for {name}")
                    if not isinstance(url, str):
                        fail(f"unapproved artifact URL for {name}: {url!r}")
                    parsed = strict_https_url(url, "artifact URL")
                    if not any(parsed.hostname == host and parsed.path.startswith(prefix) for host, prefix in ARTIFACT_PREFIXES):
                        fail(f"unapproved artifact URL for {name}: {url!r}")
                    lock_artifact_filename(url)
                    if "size" in artifact and (isinstance(artifact["size"], bool) or not isinstance(artifact["size"], int) or artifact["size"] <= 0):
                        fail(f"invalid lock artifact size for {name}")
        result[normalized] = row
    virtual_rows = [row for row in result.values() if "virtual" in row["source"]]
    if len(virtual_rows) != 1 or virtual_rows[0]["source"].get("virtual") != ".":
        fail("uv.lock must contain exactly one repository-root virtual row")
    torch = result.get("torch")
    if torch is None or torch["version"] != CPU_TORCH_VERSION:
        fail("the selected Torch row is not the reviewed CPU wheel")
    if str(torch["source"].get("registry")) != "https://download.pytorch.org/whl/cpu":
        fail("Torch source is not the reviewed CPU index")
    return {key: row for key, row in result.items() if "registry" in row["source"]}


def safe_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        fail(f"{label} must be a non-empty POSIX relative path")
    path = Path(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        fail(f"{label} contains an unsafe path")
    return path.as_posix()


def safe_child(root: Path, relative: str, label: str) -> Path:
    current = root
    for part in Path(relative).parts:
        current = current / part
        try:
            info = current.lstat()
        except OSError as error:
            fail(f"{label} is missing: {current}: {error}")
        if stat.S_ISLNK(info.st_mode):
            fail(f"{label} traverses a symlink: {relative}")
    return current


def validate_install_roots(site: Path, venv_root: Path, scripts_root: Path) -> tuple[Path, Path, Path]:
    site = require_absolute(site, "site-packages")
    venv_root = require_absolute(venv_root, "venv root")
    scripts_root = require_absolute(scripts_root, "scripts root")
    require_directory_chain(venv_root, "venv root")
    require_directory_chain(scripts_root, "scripts root")
    require_directory_chain(site, "site-packages")
    venv_real = venv_root.resolve(strict=True)
    site_real = site.resolve(strict=True)
    scripts_real = scripts_root.resolve(strict=True)
    try:
        site_real.relative_to(venv_real)
        scripts_real.relative_to(venv_real)
    except ValueError:
        fail("site-packages and scripts root must be inside the supplied venv root")
    if site_real == scripts_real:
        fail("site-packages and scripts root must be distinct")
    if scripts_real != venv_real / "bin":
        fail("scripts root must be the supplied venv's bin directory")
    if site_real.name != "site-packages" or not re.fullmatch(r"python3\.12", site_real.parent.name):
        fail("site-packages must be the CPython 3.12 venv layout")
    return site_real, venv_real, scripts_real


def validate_venv_layout(venv_root: Path, scripts_root: Path) -> dict[str, Any]:
    cfg = venv_root / "pyvenv.cfg"
    try:
        cfg_body = read_bounded(cfg, MAX_METADATA_BYTES, "pyvenv.cfg").decode("utf-8")
    except UnicodeDecodeError as error:
        fail(f"pyvenv.cfg is not UTF-8: {error}")
    values: dict[str, str] = {}
    for line in cfg_body.splitlines():
        if not line.strip():
            continue
        key, separator, value = line.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z0-9_.-]+", key.strip()):
            fail("pyvenv.cfg has malformed fields")
        normalized_key = key.strip()
        if normalized_key in values:
            fail("pyvenv.cfg has duplicate fields")
        values[normalized_key] = value.strip()
    if not re.fullmatch(r"3\.12(?:\.\d+)?", values.get("version", "")):
        fail("pyvenv.cfg is not a CPython 3.12 environment")
    if values.get("include-system-site-packages") != "false":
        fail("venv must not include system site-packages")
    if not values.get("home"):
        fail("pyvenv.cfg lacks a Python home")
    interpreter = scripts_root / "python"
    try:
        interpreter_info = interpreter.lstat()
        resolved = interpreter.resolve(strict=True)
        resolved_info = resolved.stat()
    except OSError as error:
        fail(f"venv Python interpreter is inaccessible: {error}")
    if not stat.S_ISREG(resolved_info.st_mode) or not (resolved_info.st_mode & stat.S_IXUSR):
        fail("venv Python interpreter is not an executable regular file")
    aliases: dict[str, dict[str, Any]] = {}
    for name in ("python", "python3", "python3.12"):
        alias = scripts_root / name
        try:
            alias_info = alias.lstat()
            alias_resolved = alias.resolve(strict=True)
            alias_target = alias_resolved.stat()
        except OSError as error:
            fail(f"venv Python alias is inaccessible: {error}")
        if not stat.S_ISREG(alias_target.st_mode) or not (alias_target.st_mode & stat.S_IXUSR):
            fail(f"venv Python alias is not executable: {alias}")
        if alias_resolved != resolved:
            fail("venv Python aliases do not share one interpreter target")
        aliases[name] = {"path": str(alias), "resolved_path": str(alias_resolved), "symlink": stat.S_ISLNK(alias_info.st_mode), "bytes": alias_target.st_size, "sha256": hash_bounded(alias_resolved, MAX_HASH_BYTES_PER_FILE, f"venv interpreter {name}")[1]}
    try:
        runtime = Path(sys.executable).resolve(strict=True)
    except OSError as error:
        fail(f"running Python executable is inaccessible: {error}")
    if runtime != resolved:
        fail("running Python executable is not the supplied venv interpreter")
    return {
        "pyvenv_cfg": file_identity(cfg, "pyvenv.cfg"),
        "python": {"path": str(interpreter), "resolved_path": str(resolved), "symlink": stat.S_ISLNK(interpreter_info.st_mode), "bytes": resolved_info.st_size, "mode": resolved_info.st_mode & 0o777, "aliases": aliases, "runtime_executable": str(runtime)},
        "version": values["version"],
    }


def record_location(site: Path, scripts_root: Path, relative: str, label: str) -> tuple[Path, str, str]:
    if not isinstance(relative, str) or not relative or "\\" in relative or Path(relative).is_absolute():
        fail(f"{label} has an unsafe RECORD path")
    path = Path(relative)
    if any(part == "" or part == "." for part in path.parts):
        fail(f"{label} has an unsafe RECORD path")
    if ".." not in path.parts:
        actual = safe_child(site, relative, label)
        return actual, "site", path.as_posix()
    candidate = (site / path).resolve(strict=False)
    scripts_real = scripts_root.resolve(strict=True)
    try:
        scripts_relative = candidate.relative_to(scripts_real).as_posix()
    except ValueError:
        fail(f"{label} leaves the trusted site/scripts roots")
    if not scripts_relative or scripts_relative == ".":
        fail(f"{label} resolves to the scripts root")
    actual = safe_child(scripts_real, scripts_relative, label)
    return actual, "scripts", scripts_relative


def wheel_tags_compatible(filename: str) -> None:
    if len(filename.encode("utf-8")) > MAX_WHEEL_FILENAME_BYTES:
        fail(f"wheel filename exceeds bounded tag length: {filename}")
    if not filename.endswith(".whl"):
        fail(f"selected artifact is not a wheel: {filename}")
    fields = filename[:-4].split("-")
    if len(fields) < 5:
        fail(f"wheel filename has no complete tag triplet: {filename}")
    python_tag, abi_tag, platform_tag = fields[-3:]
    python_tags = python_tag.split(".")
    abi_tags = abi_tag.split(".")
    platform_tags = platform_tag.split(".")
    if any(not value for value in (*python_tags, *abi_tags, *platform_tags)):
        fail(f"wheel filename has an empty expanded tag: {filename}")
    if any(len(value) > 64 or not re.fullmatch(r"[A-Za-z0-9_]+", value) for value in (*python_tags, *abi_tags, *platform_tags)) or any(len(set(values)) != len(values) for values in (python_tags, abi_tags, platform_tags)):
        fail(f"wheel filename has malformed expanded tags: {filename}")
    if len(python_tags) * len(abi_tags) * len(platform_tags) > MAX_EXPANDED_WHEEL_TAGS:
        fail(f"wheel filename expands to too many tags: {filename}")
    supported = False
    for candidate_python in python_tags:
        if candidate_python.startswith("cp"):
            match = re.fullmatch(r"cp(\d{2,3})", candidate_python)
            if match is None or int(match.group(1)) > 312:
                continue
            if int(match.group(1)) != 312 and "abi3" not in abi_tags:
                continue
        elif candidate_python not in {"py3", "py312"}:
            continue
        for candidate_platform in platform_tags:
            if candidate_platform == "any" or ("x86_64" in candidate_platform and not any(bad in candidate_platform for bad in ("win", "macos", "aarch64", "arm64"))):
                supported = True
    if not supported:
        fail(f"wheel tags are outside CPython 3.12 Linux x86_64 scope: {filename}")


def validate_wheel_metadata_tags(archive_path: Path, members: dict[str, zipfile.ZipInfo], dist_info_prefix: str, filename: str) -> None:
    wheel_name = f"{dist_info_prefix}/WHEEL"
    body = archive_member_bytes(archive_path, members, wheel_name, MAX_METADATA_BYTES)
    headers = parse_headers(body, f"wheel {wheel_name}")
    if len(headers.get("wheel-version", [])) != 1:
        fail(f"wheel metadata must have one Wheel-Version header: {wheel_name}")
    tags = headers.get("tag", [])
    if not tags or len(set(tags)) != len(tags) or any(not re.fullmatch(r"[^\s-]+-[^\s-]+-[^\s-]+", tag) for tag in tags):
        fail(f"wheel WHEEL Tag headers are malformed: {wheel_name}")
    fields = filename[:-4].split("-")
    expected = tuple(fields[-3:])
    if len(fields) < 5 or any(not part for part in expected):
        fail(f"wheel filename tags are malformed: {filename}")
    python_tags = expected[0].split(".")
    abi_tags = expected[1].split(".")
    platform_tags = expected[2].split(".")
    if len(filename.encode("utf-8")) > MAX_WHEEL_FILENAME_BYTES or any(len(value) > 64 or not re.fullmatch(r"[A-Za-z0-9_]+", value) for value in (*python_tags, *abi_tags, *platform_tags)) or any(len(set(values)) != len(values) for values in (python_tags, abi_tags, platform_tags)):
        fail(f"wheel filename has malformed expanded tags: {filename}")
    if len(python_tags) * len(abi_tags) * len(platform_tags) > MAX_EXPANDED_WHEEL_TAGS:
        fail(f"wheel filename expands to too many tags: {filename}")
    expected_tags = {f"{python}-{abi}-{platform}" for python in python_tags for abi in abi_tags for platform in platform_tags}
    actual_tags = set(tags)
    if actual_tags != expected_tags:
        fail(f"wheel filename tags disagree with WHEEL Tag: {filename}")


def manifest_artifact_rows(manifest: dict[str, Any], project_sha: str, lock_sha: str) -> list[dict[str, Any]]:
    required = {"format", "platform", "project_sha256", "uv_lock_sha256", "artifacts"}
    if set(manifest) != required or manifest.get("format") != "vokra-vibevoice-selected-wheel-manifest-v1":
        fail("selected wheel manifest schema is not exact")
    platform_row = manifest["platform"]
    if platform_row != {"system": "Linux", "machine": "x86_64", "python": "3.12"}:
        fail("selected wheel manifest platform is not Linux x86_64 CPython 3.12")
    if manifest["project_sha256"] != project_sha or manifest["uv_lock_sha256"] != lock_sha:
        fail("selected wheel manifest is bound to a different project or lock")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, list) or any(not isinstance(row, dict) for row in artifacts):
        fail("selected wheel manifest artifacts must be a list")
    return artifacts


def verify_selected_artifacts(
    manifest: dict[str, Any],
    manifest_path: Path,
    project_sha: str,
    lock_sha: str,
    locked: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    rows = manifest_artifact_rows(manifest, project_sha, lock_sha)
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            fail("selected wheel artifact row must be an object")
        required = {"name", "version", "url", "filename", "sha256", "bytes", "path"}
        if set(row) != required:
            fail("selected wheel artifact row schema is not exact")
        name = row["name"]
        normalized = normalize_name(name) if isinstance(name, str) else ""
        if normalized in selected or normalized not in locked:
            fail(f"selected wheel package is duplicate or not locked: {name!r}")
        if row["version"] != locked[normalized]["version"]:
            fail(f"selected wheel version mismatch for {name}")
        url = row["url"]
        filename = row["filename"]
        if not isinstance(url, str) or not isinstance(filename, str) or lock_artifact_filename(url) != filename:
            fail(f"selected wheel URL/filename mismatch for {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(row["sha256"])) or isinstance(row["bytes"], bool) or not isinstance(row["bytes"], int) or row["bytes"] <= 0:
            fail(f"selected wheel digest/size malformed for {name}")
        wheel_tags_compatible(filename)
        lock_wheels = as_artifact_list(locked[normalized], "wheels")
        matching_lock = [item for item in lock_wheels if item.get("url") == url and item.get("hash") == f"sha256:{row['sha256']}" and lock_artifact_filename(url) == filename]
        if len(matching_lock) != 1:
            fail(f"selected wheel is not one exact lock-listed wheel for {name}")
        if "size" not in matching_lock[0] or isinstance(matching_lock[0]["size"], bool) or not isinstance(matching_lock[0]["size"], int) or matching_lock[0]["size"] <= 0:
            fail(f"selected lock wheel size is missing or malformed for {name}")
        if matching_lock[0]["size"] != row["bytes"]:
            fail(f"selected wheel size differs from lock for {name}")
        if not isinstance(row["path"], str):
            fail(f"selected {name} archive path must be a string")
        path = require_absolute(Path(row["path"]), f"selected {name} archive")
        if path == manifest_path:
            fail(f"selected {name} archive overlaps the manifest")
        size, digest = hash_bounded(path, MAX_ARCHIVE_BYTES, f"selected {name} archive")
        if size != row["bytes"] or digest != row["sha256"]:
            fail(f"selected {name} archive bytes do not match manifest")
        selected[normalized] = {
            "name": name,
            "version": row["version"],
            "url": url,
            "filename": filename,
            "bytes": size,
            "sha256": digest,
            "path": str(path),
        }
    if set(selected) != set(locked):
        fail(f"selected wheel set is incomplete: missing={sorted(set(locked) - set(selected))}")
    return selected


def parse_headers(body: bytes, label: str) -> dict[str, list[str]]:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as error:
        fail(f"{label} is not UTF-8 metadata: {error}")
    result: dict[str, list[str]] = {}
    last_key: str | None = None
    for line in text.splitlines():
        if not line:
            break
        if line[:1].isspace() and last_key is not None:
            result[last_key][-1] += " " + line.strip()
            continue
        key, separator, value = line.partition(":")
        if not separator:
            fail(f"{label} has malformed metadata header")
        last_key = key.casefold()
        result.setdefault(last_key, []).append(value.strip())
    return result


def record_hash(value: str, label: str) -> str:
    if not value.startswith("sha256="):
        fail(f"{label} has unsupported RECORD hash")
    encoded = value[7:]
    if not re.fullmatch(r"[A-Za-z0-9_-]{43}", encoded):
        fail(f"{label} has malformed RECORD hash encoding")
    try:
        raw = base64.urlsafe_b64decode(encoded + "===")
    except (ValueError, binascii.Error) as error:  # type: ignore[name-defined]
        fail(f"{label} has malformed RECORD hash: {error}")
    if len(raw) != 32:
        fail(f"{label} has non-SHA256 RECORD hash")
    return raw.hex()


def is_license_path(path: str, headers: list[str]) -> bool:
    lower = path.casefold()
    base = Path(path).name.casefold()
    header_names = {Path(item.replace("\\", "/")).name.casefold() for item in headers}
    return base in LICENSE_BASENAMES or base in header_names or "/licenses/" in f"/{lower}/"


def is_native_path(path: str) -> bool:
    lower = path.casefold()
    return lower.endswith(NATIVE_SUFFIXES) or ".so." in lower


def archive_members(path: Path) -> dict[str, zipfile.ZipInfo]:
    before = snapshot(path, f"wheel archive {path}")
    if before[2] > MAX_ARCHIVE_BYTES:
        fail(f"wheel archive exceeds bounded size: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_ARCHIVE_MEMBERS or len(infos) > MAX_DIRECTORY_ENTRIES:
                fail(f"wheel has too many archive members: {path}")
            total = 0
            result: dict[str, zipfile.ZipInfo] = {}
            for info in infos:
                name = info.filename
                safe = safe_relative(name, "wheel member")
                if safe in result:
                    fail(f"wheel contains duplicate member: {safe}")
                if info.flag_bits & 0x1:
                    fail(f"encrypted wheel member: {safe}")
                mode = (info.external_attr >> 16) & 0o170000
                if mode == stat.S_IFLNK:
                    fail(f"wheel contains symlink member: {safe}")
                if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                    fail(f"wheel member exceeds bound: {safe}")
                total += info.file_size
                if total > MAX_ARCHIVE_UNPACKED_BYTES:
                    fail("wheel unpacked size exceeds bound")
                result[safe] = info
            after = snapshot(path, f"wheel archive {path}")
            if after != before:
                fail(f"wheel archive changed while listing members: {path}")
            return result
    except zipfile.BadZipFile as error:
        fail(f"selected wheel is not a valid ZIP archive: {error}")


def hash_archive_member(path: Path, info: zipfile.ZipInfo, label: str) -> tuple[int, str]:
    global _HASH_BYTES_USED, _READ_BYTES_USED
    if info.file_size > MAX_HASH_BYTES_PER_FILE:
        fail(f"{label} exceeds bounded hash")
    if _HASH_BYTES_USED + info.file_size > MAX_TOTAL_HASH_BYTES:
        fail(f"hash budget exceeded before reading {label}")
    if _READ_BYTES_USED + info.file_size > MAX_TOTAL_READ_BYTES:
        fail(f"read budget exceeded before reading {label}")
    digest = hashlib.sha256()
    total = 0
    try:
        with zipfile.ZipFile(path) as archive, archive.open(info, "r") as stream:
            while True:
                block = stream.read(min(1 << 20, MAX_HASH_BYTES_PER_FILE + 1 - total))
                if not block:
                    break
                total += len(block)
                if total > MAX_HASH_BYTES_PER_FILE:
                    fail(f"{label} exceeds bounded hash")
                digest.update(block)
    except (OSError, zipfile.BadZipFile, KeyError, RuntimeError) as error:
        fail(f"{label} could not be read: {error}")
    _HASH_BYTES_USED += total
    _READ_BYTES_USED += total
    return total, digest.hexdigest()


def archive_member_bytes(
    path: Path,
    members: dict[str, zipfile.ZipInfo],
    member: str,
    limit: int = MAX_METADATA_BYTES,
) -> bytes:
    global _READ_BYTES_USED
    info = members.get(member)
    if info is None:
        fail(f"selected wheel lacks required member: {member}")
    if info.file_size > limit:
        fail(f"selected wheel member exceeds bounded read: {member}")
    if _READ_BYTES_USED + info.file_size > MAX_TOTAL_READ_BYTES:
        fail(f"read budget exceeded before reading wheel member {member}")
    try:
        with zipfile.ZipFile(path) as archive, archive.open(info, "r") as stream:
            chunks: list[bytes] = []
            total = 0
            while True:
                block = stream.read(min(1 << 20, limit + 1 - total))
                if not block:
                    break
                total += len(block)
                if total > limit:
                    fail(f"selected wheel member exceeded bounded read: {member}")
                if _READ_BYTES_USED + total > MAX_TOTAL_READ_BYTES:
                    fail(f"read budget exceeded while reading wheel member {member}")
                chunks.append(block)
            body = b"".join(chunks)
    except (OSError, zipfile.BadZipFile, KeyError, RuntimeError) as error:
        fail(f"selected wheel member could not be read: {member}: {error}")
    if len(body) != info.file_size:
        fail(f"selected wheel member size changed: {member}")
    _READ_BYTES_USED += len(body)
    return body


def bind_file_to_archive_member(
    installed: Path,
    archive_path: Path,
    info: zipfile.ZipInfo,
    label: str,
    archive_label: str | None = None,
) -> dict[str, Any]:
    installed_size, installed_digest = hash_bounded(installed, MAX_HASH_BYTES_PER_FILE, f"installed member {label}")
    archive_size, archive_digest = hash_archive_member(archive_path, info, f"wheel member {label}")
    if installed_size != archive_size or installed_digest != archive_digest:
        fail(f"installed member differs from selected wheel: {label}")
    result = {"path": label, "bytes": installed_size, "sha256": installed_digest}
    if archive_label is not None and archive_label != label:
        result["archive_path"] = archive_label
    return result


def reverse_archive_locations(
    archive_records: dict[str, tuple[str, int]],
    dist_info_prefix: str,
    script_names: set[str],
) -> dict[tuple[str, str], str]:
    locations: dict[tuple[str, str], str] = {}
    for relative in archive_records:
        location = wheel_install_location(relative, dist_info_prefix, script_names)
        if location in locations:
            fail(f"wheel relocation target collides: {relative} and {locations[location]}")
        locations[location] = relative
    return locations


def parse_record_rows(body: bytes, label: str, allow_parent_paths: bool = False) -> dict[str, tuple[str, int]]:
    try:
        reader = csv.reader(io.StringIO(body.decode("utf-8")))
        rows = []
        for row in reader:
            rows.append(row)
            if len(rows) > MAX_RECORD_ROWS:
                fail(f"RECORD has too many rows: {label}")
    except (UnicodeDecodeError, csv.Error) as error:
        fail(f"malformed RECORD {label}: {error}")
    records: dict[str, tuple[str, int]] = {}
    for row in rows:
        if len(row) != 3:
            fail(f"malformed RECORD row in {label}")
        if allow_parent_paths:
            value = row[0]
            path = Path(value)
            if not value or value.startswith("/") or "\\" in value or any(part in ("", ".") for part in path.parts):
                fail(f"RECORD path in {label} is unsafe")
            relative = path.as_posix()
        else:
            relative = safe_relative(row[0], f"RECORD path in {label}")
        if relative in records:
            fail(f"duplicate RECORD path: {relative}")
        try:
            size = -1 if row[2] == "" else int(row[2])
        except ValueError:
            fail(f"malformed RECORD size: {relative}")
        if size < -1:
            fail(f"negative RECORD size: {relative}")
        records[relative] = (row[1], size)
    return records


def parse_record(
    site: Path,
    scripts_root: Path,
    dist_info: Path,
) -> tuple[dict[str, tuple[str, int]], dict[str, Any]]:
    record_path = dist_info / "RECORD"
    body = read_bounded(record_path, MAX_RECORD_BYTES, f"{record_path}")
    records = parse_record_rows(body, str(record_path), allow_parent_paths=True)
    record_relative = dist_info.relative_to(site).as_posix() + "/RECORD"
    if record_relative not in records:
        fail("RECORD does not contain its own path")
    for relative, (encoded, expected_size) in records.items():
        actual, _, _ = record_location(site, scripts_root, relative, f"RECORD member {relative}")
        actual_size, actual_digest = hash_bounded(actual, MAX_HASH_BYTES_PER_FILE, f"RECORD member {relative}")
        if expected_size != -1 and expected_size != actual_size:
            fail(f"RECORD size mismatch: {relative}")
        if relative != record_relative and not encoded and pyc_source_path(relative) is None:
            fail(f"RECORD hash missing: {relative}")
        if encoded and record_hash(encoded, f"RECORD {relative}") != actual_digest:
            fail(f"RECORD hash mismatch: {relative}")
    return records, {"path": record_relative, "rows": len(records), "external_paths": sum(".." in Path(path).parts for path in records)}


def installer_generated_path(relative: str, dist_info_prefix: str) -> bool:
    path = Path(relative)
    return path.parent.as_posix() == dist_info_prefix and path.name in {"INSTALLER", "REQUESTED", "direct_url.json"}


def pyc_source_path(relative: str) -> str | None:
    path = Path(relative)
    if "__pycache__" not in path.parts or not re.fullmatch(r"[^/]+\.cpython-312(?:\.opt-[0-2])?\.pyc", path.name):
        return None
    index = path.parts.index("__pycache__")
    if index + 1 != len(path.parts) - 1:
        return None
    stem = path.name.split(".cpython-312", 1)[0]
    return Path(*path.parts[:index], f"{stem}.py").as_posix()


def declared_script_names(archive_path: Path, members: dict[str, zipfile.ZipInfo], dist_info_prefix: str) -> set[str]:
    member = f"{dist_info_prefix}/entry_points.txt"
    if member not in members:
        return set()
    body = archive_member_bytes(archive_path, members, member, MAX_METADATA_BYTES)
    names: set[str] = set()
    section = ""
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as error:
        fail(f"wheel entry_points.txt is not UTF-8: {error}")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            continue
        if section not in {"console_scripts", "gui_scripts"} or "=" not in line:
            continue
        name = line.split("=", 1)[0].strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            fail(f"wheel entry point has unsafe script name: {name!r}")
        names.add(name)
    return names


def wheel_install_location(relative: str, dist_info_prefix: str, script_names: set[str]) -> tuple[str, str]:
    parts = Path(relative).parts
    if len(parts) >= 3 and parts[0].endswith(".data"):
        if parts[1] in {"purelib", "platlib"}:
            return "site", Path(*parts[2:]).as_posix()
        if parts[1] == "scripts" and len(parts) == 3:
            script = parts[2]
            if script not in script_names:
                fail(f"wheel script is not declared by entry_points.txt: {script}")
            return "scripts", script
        fail(f"wheel .data member has unsupported relocation: {relative}")
    return "site", relative


def bind_installed_record_to_wheel(
    site: Path,
    scripts_root: Path,
    records: dict[str, tuple[str, int]],
    archive_records: dict[str, tuple[str, int]],
    script_names: set[str],
    archive_path: Path,
) -> dict[str, Any]:
    dist_info_prefix = next((path.removesuffix("/RECORD") for path in archive_records if path.endswith(".dist-info/RECORD")), None)
    if dist_info_prefix is None:
        fail("wheel RECORD lacks a dist-info RECORD member")
    mapped: dict[tuple[str, str], tuple[str | None, int, str]] = {}
    archive_script_names: set[str] = set()
    for relative, (archive_hash, archive_size) in archive_records.items():
        scope, target = wheel_install_location(relative, dist_info_prefix, script_names)
        if scope == "scripts":
            archive_script_names.add(target)
        if (scope, target) in mapped:
            fail(f"wheel relocation target collides: {relative} and {mapped[(scope, target)][0]}")
        mapped[(scope, target)] = (relative, archive_size, archive_hash)
    # Most wheels carry entry_points.txt but no .data/scripts member: pip/uv
    # generate those wrappers during installation.  Bind their installed
    # RECORD/hash/path separately from archive bytes, never as archive equality.
    for script in sorted(script_names - archive_script_names):
        key = ("scripts", script)
        if key in mapped:
            fail(f"wheel script relocation collides with generated wrapper: {script}")
        mapped[key] = (None, -1, "")
    seen: set[tuple[str, str]] = set()
    relocated: list[dict[str, Any]] = []
    installer_generated: list[dict[str, Any]] = []
    generated_pycs: list[dict[str, Any]] = []
    generated_wrappers: list[dict[str, Any]] = []
    for relative, (installed_hash, installed_size) in records.items():
        actual, scope, target = record_location(site, scripts_root, relative, f"installed RECORD member {relative}")
        key = (scope, target)
        expected = mapped.get(key)
        if expected is None:
            pyc_source = pyc_source_path(relative)
            if pyc_source is not None:
                source_locations = {wheel_install_location(path, dist_info_prefix, script_names) for path in archive_records}
                if ("site", pyc_source) not in source_locations:
                    fail(f"installed pyc has no wheel Python source: {relative}")
                generated_size, generated_digest = hash_bounded(actual, MAX_HASH_BYTES_PER_FILE, f"generated pyc {relative}")
                generated_row = {"path": relative, "bytes": generated_size, "sha256": generated_digest, "source_path": pyc_source, "binding": "UNPROVEN_INSTALLER_SOURCE"}
                generated_pycs.append(generated_row)
                continue
            if not installer_generated_path(relative, dist_info_prefix):
                fail(f"installed RECORD has unbound non-installer path: {relative}")
            generated_size, generated_digest = hash_bounded(actual, MAX_HASH_BYTES_PER_FILE, f"installer-generated {relative}")
            installer_generated.append({"path": relative, "bytes": generated_size, "sha256": generated_digest, "binding": "UNPROVEN_INSTALLER_SOURCE"})
            continue
        seen.add(key)
        archive_relative, archive_size, archive_hash = expected
        if scope == "scripts":
            if not installed_hash or installed_size < 0:
                fail(f"installed wrapper lacks a bounded RECORD identity: {relative}")
            installed_digest = record_hash(installed_hash, f"RECORD {relative}")
            if archive_relative is None:
                entry_source = f"{dist_info_prefix}/entry_points.txt"
                entry_hash, entry_size = archive_records.get(entry_source, ("", -1))
                if not entry_hash or entry_size == -1:
                    fail("generated wrapper lacks authenticated entry_points.txt source")
                generated_wrappers.append({"entry_point": target, "installed": relative, "bytes": installed_size, "sha256": installed_digest, "entry_point_source": entry_source, "entry_point_source_bytes": entry_size, "entry_point_source_sha256": record_hash(entry_hash, entry_source), "binding": "UNPROVEN_INSTALLER_SOURCE"})
            else:
                relocated.append({"archive": archive_relative, "installed": relative, "script": target, "bytes": installed_size, "sha256": installed_digest, "binding": "UNPROVEN_INSTALLER_SOURCE"})
            continue
        if archive_hash and installed_hash != archive_hash:
            fail(f"installed RECORD changes wheel hash: {relative}")
        if archive_size != -1 and installed_size != archive_size:
            fail(f"installed RECORD changes wheel size: {relative}")
    missing = sorted(set(mapped) - seen)
    if missing:
        fail(f"installed RECORD omits wheel install targets: {missing[:8]}")
    return {"relocated_scripts": relocated, "generated_wrappers": generated_wrappers, "installer_generated": sorted(installer_generated, key=lambda row: row["path"]), "generated_pyc": sorted(generated_pycs, key=lambda row: row["path"])}


def validate_archive_record(
    archive_path: Path,
    members: dict[str, zipfile.ZipInfo],
    dist_info_prefix: str,
) -> dict[str, tuple[str, int]]:
    record_name = f"{dist_info_prefix}/RECORD"
    body = archive_member_bytes(archive_path, members, record_name, MAX_RECORD_BYTES)
    records = parse_record_rows(body, f"wheel {record_name}")
    if set(records) != set(members):
        fail("wheel RECORD does not enumerate exactly the wheel members")
    for relative, (encoded, expected_size) in records.items():
        info = members[relative]
        if expected_size != -1 and expected_size != info.file_size:
            fail(f"wheel RECORD size mismatch: {relative}")
        if relative != record_name and not encoded:
            fail(f"wheel RECORD hash missing: {relative}")
        if encoded:
            actual_size, actual_digest = hash_archive_member(archive_path, info, f"wheel member {relative}")
            if actual_size != info.file_size or record_hash(encoded, f"wheel RECORD {relative}") != actual_digest:
                fail(f"wheel RECORD hash mismatch: {relative}")
    return records


def inspect_installed_package(
    site: Path,
    scripts_root: Path,
    normalized: str,
    lock_row: dict[str, Any],
    selected: dict[str, Any],
) -> dict[str, Any]:
    candidates = []
    try:
        site_children = list(site.iterdir())
    except OSError as error:
        fail(f"site-packages cannot be listed: {error}")
    if len(site_children) > MAX_DIRECTORY_ENTRIES:
        fail("site-packages has too many top-level entries")
    for child in site_children:
        if child.name.casefold().endswith(".dist-info"):
            if child.is_symlink() or not child.is_dir():
                fail(f"invalid dist-info entry: {child}")
            metadata_path = child / "METADATA"
            headers = parse_headers(read_bounded(metadata_path, MAX_METADATA_BYTES, f"{metadata_path}"), str(metadata_path))
            name = headers.get("name", [""])[0]
            if normalize_name(name) == normalized:
                candidates.append((child, headers))
    if len(candidates) != 1:
        fail(f"expected exactly one installed dist-info for {normalized}, found {len(candidates)}")
    dist_info, headers = candidates[0]
    for required_header in ("name", "version", "metadata-version"):
        if len(headers.get(required_header, [])) != 1 or not headers[required_header][0]:
            fail(f"installed METADATA must have one {required_header}: {dist_info}")
    version = headers.get("version", [""])[0]
    if version != lock_row["version"]:
        fail(f"installed version mismatch for {normalized}")
    records, record_summary = parse_record(site, scripts_root, dist_info)
    required_metadata = [dist_info / "METADATA", dist_info / "WHEEL", dist_info / "RECORD"]
    for item in required_metadata:
        relative = item.relative_to(site).as_posix()
        if relative not in records:
            fail(f"RECORD lacks required metadata member: {relative}")
    license_headers = headers.get("license-file", [])
    declared_license_paths = []
    metadata_version = headers.get("metadata-version", [""])[0]
    try:
        metadata_version_tuple = tuple(int(part) for part in metadata_version.split(".")[:2])
    except ValueError:
        fail(f"invalid Metadata-Version in {dist_info / 'METADATA'}")
    for header in license_headers:
        declared = safe_relative(header, f"License-File in {dist_info / 'METADATA'}")
        dist_info_prefix = dist_info.relative_to(site).as_posix()
        if metadata_version_tuple >= (2, 4):
            pep639_path = f"{dist_info_prefix}/licenses/{declared}"
            candidates = [pep639_path] if pep639_path in records else []
        else:
            candidates = [candidate for candidate in (declared, f"{dist_info_prefix}/{declared}") if candidate in records]
        if len(candidates) != 1:
            fail(f"License-File does not resolve uniquely to RECORD member: {declared}")
        declared_license_paths.append(candidates[0])
    license_paths = sorted(set(declared_license_paths) | {path for path in records if is_license_path(path, [])})
    native_paths = sorted(path for path in records if is_native_path(path))
    archive_path = Path(selected["path"])
    members = archive_members(archive_path)
    dist_info_prefix = dist_info.relative_to(site).as_posix()
    validate_wheel_metadata_tags(archive_path, members, dist_info_prefix, selected["filename"])
    archive_records = validate_archive_record(archive_path, members, dist_info_prefix)
    script_names = declared_script_names(archive_path, members, dist_info_prefix)
    relocation = bind_installed_record_to_wheel(site, scripts_root, records, archive_records, script_names, archive_path)
    archive_locations = reverse_archive_locations(archive_records, dist_info_prefix, script_names)
    # RECORD itself is installer-mutated: the archive RECORD is authenticated
    # separately above, while the installed RECORD is checked against it with
    # the explicit installer-generated-entry policy.
    bind_metadata = [item.relative_to(site).as_posix() for item in required_metadata if item.name != "RECORD"]
    interesting = sorted(set(license_paths + native_paths + bind_metadata))
    bound: list[dict[str, Any]] = []
    for relative in interesting:
        installed, scope, target = record_location(site, scripts_root, relative, f"installed member {relative}")
        archive_relative = archive_locations.get((scope, target))
        if archive_relative is None:
            fail(f"installed member has no wheel relocation binding: {relative}")
        info = members.get(archive_relative)
        if info is None:
            fail(f"selected wheel lacks bound installed member: {archive_relative}")
        bound.append(bind_file_to_archive_member(installed, archive_path, info, relative, archive_relative))
    return {
        "name": headers["name"][0],
        "version": version,
        "record": record_summary,
        "relocation": relocation,
        "license_files": [item for item in bound if item["path"] in license_paths],
        "license_file_status": "PRESENT" if license_paths else "MISSING_BUNDLED_LICENSE_UNRESOLVED",
        "native_files": [item for item in bound if item["path"] in native_paths],
        "native_payload_status": "ARCHIVE_AND_RECORD_BYTES_BOUND" if native_paths else "NONE_OBSERVED",
        "archive_binding": "METADATA_WHEEL_RECORD_LOGICAL_LICENSE_NATIVE_BYTES_BOUND",
    }


def installed_distribution_names(site: Path) -> set[str]:
    names: set[str] = set()
    try:
        children = list(site.iterdir())
    except OSError as error:
        fail(f"site-packages cannot be listed: {error}")
    if len(children) > MAX_DIRECTORY_ENTRIES:
        fail("site-packages has too many top-level entries")
    for child in children:
        if not child.name.casefold().endswith(".dist-info"):
            continue
        if child.is_symlink() or not child.is_dir():
            fail(f"invalid dist-info entry: {child}")
        metadata_path = child / "METADATA"
        headers = parse_headers(read_bounded(metadata_path, MAX_METADATA_BYTES, f"{metadata_path}"), str(metadata_path))
        values = headers.get("name", [])
        if len(values) != 1 or not values[0]:
            fail(f"installed dist-info has no unique Name: {child}")
        normalized = normalize_name(values[0])
        if normalized in names:
            fail(f"duplicate installed distribution Name: {values[0]}")
        names.add(normalized)
    return names


def bounded_regular_files(root: Path, label: str, bootstrap_names: frozenset[str] = frozenset(), allow_bootstrap_symlinks: bool = False) -> set[Path]:
    """List an install root without following symlinks or unbounded trees."""
    result: set[Path] = set()
    pending = [root]
    entries = 0
    while pending:
        current = pending.pop()
        try:
            children = list(current.iterdir())
        except OSError as error:
            fail(f"{label} cannot be listed: {error}")
        entries += len(children)
        if entries > MAX_DIRECTORY_ENTRIES:
            fail(f"{label} has too many directory entries")
        for child in children:
            try:
                info = child.lstat()
            except OSError as error:
                fail(f"{label} entry is inaccessible: {error}")
            if stat.S_ISLNK(info.st_mode):
                # The interpreter bootstrap is authenticated separately by
                # validate_venv_layout; all package/source symlinks fail closed.
                if child.parent == root and child.name in bootstrap_names:
                    if allow_bootstrap_symlinks:
                        continue
                    fail(f"{label} bootstrap must be a regular non-symlink file: {child}")
                fail(f"{label} contains an unreviewed symlink: {child}")
            elif stat.S_ISDIR(info.st_mode):
                pending.append(child)
            elif stat.S_ISREG(info.st_mode):
                if child.parent == root and child.name in bootstrap_names:
                    continue
                result.add(child)
            else:
                fail(f"{label} contains an unsupported file type: {child}")
    return result


def validate_install_inventory(site: Path, scripts_root: Path) -> dict[str, Any]:
    expected_site: set[Path] = set()
    expected_scripts: set[Path] = set()
    for dist_info in sorted(site.glob("*.dist-info")):
        records, _ = parse_record(site, scripts_root, dist_info)
        for relative in records:
            actual, scope, _ = record_location(site, scripts_root, relative, f"inventory RECORD member {relative}")
            (expected_scripts if scope == "scripts" else expected_site).add(actual)
    actual_site = bounded_regular_files(site, "site-packages", VENV_SITE_BOOTSTRAP)
    actual_scripts = bounded_regular_files(scripts_root, "venv scripts", VENV_SCRIPT_BOOTSTRAP, allow_bootstrap_symlinks=True)
    # python/python3/python3.12 are venv bootstrap interpreters, not package
    # RECORD targets; they are retained as a separately reported boundary.
    bootstrap = {scripts_root / name for name in ("python", "python3", "python3.12")}
    unregistered_site = sorted(str(path) for path in actual_site - expected_site)
    unregistered_scripts = sorted(str(path) for path in actual_scripts - expected_scripts)
    missing_site = sorted(str(path) for path in expected_site - actual_site)
    missing_scripts = sorted(str(path) for path in expected_scripts - actual_scripts)
    if unregistered_site or unregistered_scripts or missing_site or missing_scripts:
        fail(
            "installed inventory differs from RECORD: "
            f"unregistered_site={unregistered_site[:4]}, unregistered_scripts={unregistered_scripts[:4]}, "
            f"missing_site={missing_site[:4]}, missing_scripts={missing_scripts[:4]}"
        )
    identities: dict[str, dict[str, Any]] = {}
    for path in sorted(actual_site | actual_scripts):
        identities[str(path)] = file_identity(path, f"installed inventory {path}", MAX_HASH_BYTES_PER_FILE)
    for name in sorted(VENV_SITE_BOOTSTRAP):
        path = site / name
        if path.exists() and not path.is_symlink():
            identities[str(path)] = file_identity(path, f"venv site bootstrap {path}", MAX_HASH_BYTES_PER_FILE)
    bootstrap_identities: list[dict[str, Any]] = []
    for name in sorted(VENV_SCRIPT_BOOTSTRAP):
        path = scripts_root / name
        if not path.exists() and not path.is_symlink():
            continue
        try:
            info = path.lstat()
            target = path.resolve(strict=True)
            target_info = target.stat()
        except OSError as error:
            fail(f"venv bootstrap is inaccessible: {path}: {error}")
        if not stat.S_ISREG(target_info.st_mode):
            fail(f"venv bootstrap is not a regular file: {path}")
        target_size, target_digest = hash_bounded(target, MAX_HASH_BYTES_PER_FILE, f"venv bootstrap {path}")
        identities[str(path)] = {"path": str(path), "target": str(target), "bytes": target_size, "sha256": target_digest, "dev": target_info.st_dev, "ino": target_info.st_ino, "mtime_ns": target_info.st_mtime_ns, "ctime_ns": target_info.st_ctime_ns, "nlink": target_info.st_nlink, "symlink": stat.S_ISLNK(info.st_mode)}
        bootstrap_identities.append({"path": str(path), "target": str(target), "symlink": stat.S_ISLNK(info.st_mode), "bytes": target_size, "sha256": target_digest, "status": "UNPROVEN_INSTALLER_BOOTSTRAP"})
    return {
        "site_files": len(actual_site),
        "script_files": len(actual_scripts),
        "bootstrap_interpreters": bootstrap_identities,
        "bootstrap_files": sorted(str(path) for path in identities if Path(path).parent == site and Path(path).name in VENV_SITE_BOOTSTRAP),
        "record_bound_files": len(expected_site) + len(expected_scripts),
        "_identities": identities,
    }


def source_identity(source_root: Path, expected_revision: str) -> dict[str, Any]:
    require_directory_chain(source_root, "official source checkout")
    def run(*args: str) -> str:
        try:
            result = subprocess.run(["git", "-C", str(source_root), *args], check=True, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            fail(f"source git command failed: {error}")
        return result.stdout.strip()
    def state() -> tuple[str, str, str]:
        return (
            run("rev-parse", "--verify", "HEAD"),
            run("status", "--porcelain", "--untracked-files=all"),
            run("remote", "get-url", "origin"),
        )
    start_revision, start_status, start_remote = state()
    revision = start_revision
    if revision != expected_revision:
        fail("official source revision mismatch")
    if start_status:
        fail("official source checkout is dirty")
    remote = start_remote
    accepted = {OFFICIAL_SOURCE_REPOSITORY, OFFICIAL_SOURCE_REPOSITORY.removesuffix(".git")}
    if remote not in accepted:
        fail("official source origin mismatch")
    end_revision, end_status, end_remote = state()
    if (end_revision, end_status, end_remote) != (start_revision, start_status, start_remote):
        fail("official source checkout changed during collection")
    return {"repository": OFFICIAL_SOURCE_REPOSITORY, "origin": remote, "revision": revision, "working_tree": "CLEAN"}


def revalidate_install_inventory(site: Path, scripts_root: Path, initial: dict[str, Any]) -> None:
    current = validate_install_inventory(site, scripts_root)
    if current.get("_identities") != initial.get("_identities"):
        fail("installed package files or bootstrap files changed during collection")
    for key in ("site_files", "script_files", "record_bound_files"):
        if current.get(key) != initial.get(key):
            fail(f"installed inventory count changed during collection: {key}")


def audit(args: argparse.Namespace) -> dict[str, Any]:
    global _HASH_BYTES_USED, _READ_BYTES_USED
    _HASH_BYTES_USED = 0
    _READ_BYTES_USED = 0
    if platform.system() != "Linux" or platform.machine() != "x86_64" or sys.version_info[:2] != (3, 12):
        fail("installed closure audit requires Linux x86_64 CPython 3.12")
    if os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1":
        fail("VOKRA_PUBLISH_ON_VAST=1 is required")
    collector_path = Path(__file__).absolute()
    require_directory_chain(collector_path.parent, "collector parent")
    collector = file_identity(collector_path, "collector")
    if collector_path.is_symlink():
        fail("collector must not be a symlink")
    project = require_absolute(Path(args.project), "project")
    lock = require_absolute(Path(args.lock), "uv.lock")
    site = require_absolute(Path(args.site_packages), "site-packages")
    venv_root = require_absolute(Path(args.venv_root), "venv root")
    scripts_root = require_absolute(Path(args.scripts_root), "scripts root")
    manifest_path = require_absolute(Path(args.selected_wheel_manifest), "selected wheel manifest")
    output = require_absolute(Path(args.output), "output")
    site, venv_root, scripts_root = validate_install_roots(site, venv_root, scripts_root)
    venv_layout = validate_venv_layout(venv_root, scripts_root)
    project_body, project_identity = read_with_identity(project, MAX_METADATA_BYTES, "project")
    project_doc = parse_toml_bytes(project_body, "project")
    validate_project(project_doc)
    lock_body, lock_identity = read_with_identity(lock, MAX_METADATA_BYTES, "uv.lock")
    lock_doc = parse_toml_bytes(lock_body, "uv.lock")
    locked = validate_lock(lock_doc)
    project_sha = project_identity["sha256"]
    lock_sha = lock_identity["sha256"]
    manifest_body, manifest_identity = read_with_identity(manifest_path, MAX_MANIFEST_BYTES, "selected wheel manifest")
    manifest = parse_json_bytes(manifest_body, "selected wheel manifest")
    manifest_sha = manifest_identity["sha256"]
    selected = verify_selected_artifacts(manifest, manifest_path, project_sha, lock_sha, locked)
    try:
        expected_revision = project_doc["tool"]["vokra"]["reference"]["official_source_revision"]
    except (KeyError, TypeError):
        fail("project does not declare an official source revision")
    if not isinstance(expected_revision, str) or not re.fullmatch(r"[0-9a-f]{40,64}", expected_revision):
        fail("official source revision is not a lower-hex Git revision")
    source = source_identity(Path(args.source_root), expected_revision)
    # Capture every installed file before any package metadata/RECORD binding;
    # later package inspection must not become the first observation of a file.
    install_inventory = validate_install_inventory(site, scripts_root)
    installed_names = installed_distribution_names(site)
    if installed_names != set(locked):
        fail(f"installed distribution set differs from lock: extra={sorted(installed_names - set(locked))}, missing={sorted(set(locked) - installed_names)}")
    packages = []
    for normalized in sorted(locked):
        packages.append({
            "lock": {"name": locked[normalized]["name"], "version": locked[normalized]["version"], "source": locked[normalized]["source"]},
            "selected_artifact": selected[normalized],
            "installed": inspect_installed_package(site, scripts_root, normalized, locked[normalized], selected[normalized]),
            "license_status": "PENDING_PRIMARY_SOURCE_AND_OWNER_REVIEW",
        })
    for normalized, artifact in selected.items():
        size, digest = hash_bounded(Path(artifact["path"]), MAX_ARCHIVE_BYTES, f"selected {normalized} archive revalidation")
        if size != artifact["bytes"] or digest != artifact["sha256"]:
            fail(f"selected {normalized} archive changed during inspection")
    if installed_distribution_names(site) != installed_names:
        fail("installed distribution set changed during inspection")
    revalidate_install_inventory(site, scripts_root, install_inventory)
    if source_identity(Path(args.source_root), expected_revision) != source:
        fail("official source checkout changed during package collection")
    if validate_venv_layout(venv_root, scripts_root) != venv_layout:
        fail("venv layout changed during collection")
    if file_identity(project, "project final revalidation") != project_identity:
        fail("project changed during collection")
    if file_identity(lock, "uv.lock final revalidation") != lock_identity:
        fail("uv.lock changed during collection")
    if file_identity(manifest_path, "selected wheel manifest final revalidation") != manifest_identity:
        fail("selected wheel manifest changed during collection")
    if file_identity(collector_path, "collector revalidation") != collector:
        fail("collector changed during audit")
    missing = sorted(item["lock"]["name"] for item in packages if item["installed"]["license_file_status"] != "PRESENT")
    return {
        "format": FORMAT,
        "status": STATUS,
        "scope": {"system": "Linux", "machine": "x86_64", "python": "3.12", "cpu_only": True},
        "venv_layout": venv_layout,
        "install_layout": {"venv_root": str(venv_root), "site_packages": str(site), "scripts_root": str(scripts_root), "external_record_paths": "TRUSTED_VENV_SCRIPTS_ONLY", "inventory": {key: value for key, value in install_inventory.items() if key != "_identities"}},
        "collector": collector,
        "project": project_identity,
        "uv_lock": lock_identity,
        "selected_wheel_manifest": {"path": str(manifest_path), "bytes": manifest_identity["bytes"], "sha256": manifest_sha, "status": "UNTRUSTED_INPUT_BOUND_TO_LOCK_AND_ARCHIVE_BYTES"},
        "official_source": source,
        "packages": packages,
        "license_evidence_summary": {"missing_license_distribution_count": len(missing), "missing_license_distributions": missing},
        "execution": {"model_download": "NO_MODEL_DOWNLOAD", "model_execution": "NO_MODEL_EXECUTION", "publication": "NO_UPLOAD"},
        "approval": {"owner": "UNAPPROVED", "license": "UNRESOLVED", "native": "UNRESOLVED_FOR_RELEASE", "runtime_compatibility": "UNPROVEN"},
        "blockers": [
            "Selected wheel and installed RECORD bytes are bound, but package license/native review and owner approval remain unresolved.",
            *([f"Bundled license files remain unresolved for: {', '.join(missing)}."] if missing else []),
        ],
    }


def write_exclusive(path: Path, body: bytes) -> dict[str, Any]:
    require_directory_chain(path.parent, "output parent")
    if path.exists() or path.is_symlink():
        fail(f"output already exists: {path}")
    created: tuple[int, int] | None = None
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            created_info = os.fstat(fd)
            created = (created_info.st_dev, created_info.st_ino)
            offset = 0
            while offset < len(body):
                offset += os.write(fd, body[offset:])
            os.fsync(fd)
        finally:
            os.close(fd)
    except FileExistsError:
        fail(f"output already exists: {path}")
    except Exception:
        if created is not None:
            try:
                info = path.lstat()
                if (info.st_dev, info.st_ino) == created:
                    path.unlink()
            except OSError:
                pass
        raise
    return {"path": str(path), "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def write_report(output: Path, document: dict[str, Any]) -> None:
    body = (json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    sidecar = Path(str(output) + ".sha256")
    if output.exists() or output.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        fail(f"report output or sidecar already exists: {output}")
    identity = write_exclusive(output, body)
    created = snapshot(output, "report output")
    try:
        write_exclusive(sidecar, (identity["sha256"] + "\n").encode())
    except Exception:
        try:
            if snapshot(output, "report output") == created:
                output.unlink()
        except OSError:
            pass
        raise


def _fake_wheel(
    root: Path,
    metadata_version: str = "2.1",
    license_path: str = "LICENCE",
    package_name: str = "demo",
    package_version: str = "1.0",
    archive_name: str | None = None,
    archive_license_path: str | None = None,
) -> tuple[Path, dict[str, Any], Path]:
    normalized_dist = f"{package_name}-{package_version}.dist-info"
    archive = root / (archive_name or f"{package_name}-{package_version}-py3-none-any.whl")
    wheel_tag = "-".join(archive.name[:-4].split("-")[-3:])
    files = {
        f"{package_name}/__init__.py": f"{package_name}\n".encode(),
        f"{normalized_dist}/METADATA": f"Metadata-Version: {metadata_version}\nName: {package_name}\nVersion: {package_version}\nLicense-File: {license_path}\n\n".encode(),
        f"{normalized_dist}/WHEEL": f"Wheel-Version: 1.0\nTag: {wheel_tag}\n\n".encode(),
        f"{normalized_dist}/{archive_license_path or license_path}": b"MPL-2.0 AND MIT\n",
    }
    def digest(value: bytes) -> str:
        return base64.urlsafe_b64encode(hashlib.sha256(value).digest()).decode().rstrip("=")
    rows = [f"{name},sha256={digest(value)},{len(value)}" for name, value in files.items()]
    rows.append(f"{normalized_dist}/RECORD,,")
    files[f"{normalized_dist}/RECORD"] = ("\n".join(rows) + "\n").encode()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as output:
        for name, value in files.items():
            output.writestr(name, value)
    site = root / "site"
    (site / normalized_dist).mkdir(parents=True)
    (site / package_name).mkdir()
    for name, value in files.items():
        target = site / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
    return archive, files, site


def _fake_relocated_wheel(root: Path, archive_script: bool = True, purelib_native: bool = False) -> tuple[Path, Path, Path, dict[str, Any], dict[str, Any]]:
    """Build one wheel plus a venv layout with a rewritten console wrapper."""
    archive = root / "demo-1.0-py3-none-any.whl"
    dist = "demo-1.0.dist-info"
    archive_files = {
        "demo/__init__.py": b"demo\n",
        f"{dist}/METADATA": b"Metadata-Version: 2.1\nName: demo\nVersion: 1.0\nLicense-File: LICENCE\n\n",
        f"{dist}/WHEEL": b"Wheel-Version: 1.0\nTag: py3-none-any\n\n",
        f"{dist}/entry_points.txt": b"[console_scripts]\ndemo = demo:main\n",
        f"{dist}/LICENCE": b"MIT\n",
    }
    if archive_script:
        archive_files["demo-1.0.data/scripts/demo"] = b"#!/usr/bin/python\n# generated from wheel\n"
    if purelib_native:
        archive_files["demo-1.0.data/purelib/demo_native.so"] = b"synthetic native bytes\n"

    def digest(value: bytes) -> str:
        return base64.urlsafe_b64encode(hashlib.sha256(value).digest()).decode().rstrip("=")

    rows = [f"{name},sha256={digest(value)},{len(value)}" for name, value in archive_files.items()]
    rows.append(f"{dist}/RECORD,,")
    archive_files[f"{dist}/RECORD"] = ("\n".join(rows) + "\n").encode()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as output:
        for name, value in archive_files.items():
            output.writestr(name, value)

    venv = root / "venv"
    site = venv / "lib" / "python3.12" / "site-packages"
    scripts = venv / "bin"
    (site / dist).mkdir(parents=True)
    (site / "demo").mkdir()
    scripts.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(
        "home = /usr/bin\ninclude-system-site-packages = false\nversion = 3.12.8\n",
        encoding="utf-8",
    )
    interpreter = scripts / "python"
    interpreter.write_bytes(b"synthetic CPython 3.12 interpreter\n")
    interpreter.chmod(0o755)
    for alias in ("python3", "python3.12"):
        (scripts / alias).symlink_to("python")
    installed = {
        "demo/__init__.py": archive_files["demo/__init__.py"],
        f"{dist}/METADATA": archive_files[f"{dist}/METADATA"],
        f"{dist}/WHEEL": archive_files[f"{dist}/WHEEL"],
        f"{dist}/entry_points.txt": archive_files[f"{dist}/entry_points.txt"],
        f"{dist}/LICENCE": archive_files[f"{dist}/LICENCE"],
    }
    if purelib_native:
        installed["demo_native.so"] = archive_files["demo-1.0.data/purelib/demo_native.so"]
    for name, value in installed.items():
        target = site / name
        target.write_bytes(value)
    wrapper = b"#!/private/venv/bin/python\n# rewritten by installer\n"
    (scripts / "demo").write_bytes(wrapper)
    installed_rows = [f"{name},sha256={digest(value)},{len(value)}" for name, value in installed.items()]
    installed_rows.append(f"../../../bin/demo,sha256={digest(wrapper)},{len(wrapper)}")
    installed_rows.append(f"{dist}/RECORD,,")
    (site / dist / "RECORD").write_text("\n".join(installed_rows) + "\n", encoding="utf-8")
    url = f"https://files.pythonhosted.org/packages/{archive.name}"
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    selected = {"name": "demo", "version": "1.0", "url": url, "filename": archive.name, "bytes": archive.stat().st_size, "sha256": archive_sha, "path": str(archive)}
    locked = {"name": "demo", "version": "1.0", "source": {"registry": "https://pypi.org/simple"}, "wheels": [{"url": url, "hash": f"sha256:{archive_sha}", "size": archive.stat().st_size}]}
    return archive, site, scripts, selected, locked


def run_full_stdlib_regression_suite() -> None:
    test_path = Path(__file__).with_name("test_audit_installed_closure.py")
    if not test_path.is_file() or test_path.is_symlink():
        fail("stdlib regression suite is missing or symlinked")
    spec = importlib.util.spec_from_file_location("vokra_vibevoice_audit_regressions", test_path)
    if spec is None or spec.loader is None:
        fail("could not load stdlib regression suite")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
    if not result.wasSuccessful():
        fail(f"stdlib regression suite failed: {stream.getvalue()[-4000:]}")
    print(f"audit_installed_closure unittest: PASS ({result.testsRun} cases)")


def self_test() -> None:
    global _HASH_BYTES_USED, _READ_BYTES_USED
    _HASH_BYTES_USED = 0
    _READ_BYTES_USED = 0
    with tempfile.TemporaryDirectory(prefix="vokra-vibevoice-audit-") as temp:
        root = Path(temp).resolve()
        archive, files, site = _fake_wheel(root)
        archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
        manifest = {"format": "vokra-vibevoice-selected-wheel-manifest-v1", "platform": {"system": "Linux", "machine": "x86_64", "python": "3.12"}, "project_sha256": "p" * 64, "uv_lock_sha256": "l" * 64, "artifacts": [{"name": "demo", "version": "1.0", "url": "https://files.pythonhosted.org/packages/demo-1.0-py3-none-any.whl", "filename": archive.name, "sha256": archive_sha, "bytes": archive.stat().st_size, "path": str(archive)}]}
        locked = {"demo": {"name": "demo", "version": "1.0", "source": {"registry": "https://pypi.org/simple"}, "wheels": [{"url": manifest["artifacts"][0]["url"], "hash": f"sha256:{archive_sha}", "size": archive.stat().st_size}]}}
        selected = verify_selected_artifacts(manifest, root / "manifest.json", "p" * 64, "l" * 64, locked)
        row = inspect_installed_package(site, root / "scripts", "demo", locked["demo"], selected["demo"])
        assert row["license_file_status"] == "PRESENT"
        assert row["license_files"][0]["path"].endswith("/LICENCE")
        pyc = site / "demo" / "__pycache__" / "__init__.cpython-312.pyc"
        pyc.parent.mkdir()
        pyc.write_bytes(b"synthetic-pyc\n")
        broken_record = site / "demo-1.0.dist-info" / "RECORD"
        broken_record.write_text(broken_record.read_text(encoding="utf-8") + f"demo/__pycache__/__init__.cpython-312.pyc,,{pyc.stat().st_size}\n", encoding="utf-8")
        pyc_row = inspect_installed_package(site, root / "scripts", "demo", locked["demo"], selected["demo"])
        assert pyc_row["relocation"]["generated_pyc"][0]["path"] == "demo/__pycache__/__init__.cpython-312.pyc"
        broken = site / "demo-1.0.dist-info" / "RECORD"
        broken.write_text("demo/__init__.py,,\n", encoding="utf-8")
        try:
            inspect_installed_package(site, root / "scripts", "demo", locked["demo"], selected["demo"])
        except AuditError:
            pass
        else:
            raise AssertionError("incomplete RECORD was accepted")
        assert write_exclusive(root / "out.json", b"{}\n")["bytes"] == 3
        try:
            write_exclusive(root / "out.json", b"tamper\n")
        except AuditError:
            pass
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing output was overwritten")
        relocated_root = root / "relocated"
        relocated_root.mkdir()
        _, relocated_site, relocated_scripts, selected_row, locked_row = _fake_relocated_wheel(relocated_root, purelib_native=True)
        relocated = inspect_installed_package(relocated_site, relocated_scripts, "demo", locked_row, selected_row)
        assert relocated["relocation"]["relocated_scripts"] == [{
            "archive": "demo-1.0.data/scripts/demo",
            "installed": "../../../bin/demo",
            "script": "demo",
            "bytes": len(b"#!/private/venv/bin/python\n# rewritten by installer\n"),
            "sha256": hashlib.sha256(b"#!/private/venv/bin/python\n# rewritten by installer\n").hexdigest(),
            "binding": "UNPROVEN_INSTALLER_SOURCE",
        }]
        assert relocated["relocation"]["installer_generated"] == []
        assert relocated["native_payload_status"] == "ARCHIVE_AND_RECORD_BYTES_BOUND"
        assert relocated["native_files"][0]["archive_path"] == "demo-1.0.data/purelib/demo_native.so"
        generated_root = root / "generated"
        generated_root.mkdir()
        _, generated_site, generated_scripts, generated_selected, generated_locked = _fake_relocated_wheel(generated_root, archive_script=False)
        generated = inspect_installed_package(generated_site, generated_scripts, "demo", generated_locked, generated_selected)
        assert generated["relocation"]["relocated_scripts"] == []
        assert generated["relocation"]["generated_wrappers"][0]["entry_point"] == "demo"
        assert generated["relocation"]["generated_wrappers"][0]["binding"] == "UNPROVEN_INSTALLER_SOURCE"
        assert generated["relocation"]["generated_wrappers"][0]["bytes"] == len(b"#!/private/venv/bin/python\n# rewritten by installer\n")
        try:
            wheel_install_location("demo-1.0.data/scripts/unknown", "demo-1.0.dist-info", {"demo"})
        except AuditError:
            pass
        else:
            raise AssertionError("undeclared wheel script was accepted")
        try:
            record_location(relocated_site, relocated_scripts, "../../outside/not-a-script", "synthetic RECORD")
        except AuditError:
            pass
        else:
            raise AssertionError("out-of-root RECORD path was accepted")
        extra = relocated_site / "unreviewed-license.txt"
        extra.write_bytes(b"unreviewed\n")
        extra_digest = base64.urlsafe_b64encode(hashlib.sha256(extra.read_bytes()).digest()).decode().rstrip("=")
        relocated_record = relocated_site / "demo-1.0.dist-info" / "RECORD"
        relocated_record.write_text(relocated_record.read_text(encoding="utf-8") + f"unreviewed-license.txt,sha256={extra_digest},{extra.stat().st_size}\n", encoding="utf-8")
        try:
            inspect_installed_package(relocated_site, relocated_scripts, "demo", locked_row, selected_row)
        except AuditError:
            pass
        else:
            raise AssertionError("unbound installed license file was accepted")
        project = root / "synthetic-pyproject.toml"
        lock = root / "synthetic-uv.lock"
        project.write_text(
            "[project]\nname = 'vokra-vibevoice-realtime-0-5b-reference'\n"
            "requires-python = '==3.12.*'\ndependencies = ['torch==2.13.0']\n"
            "[tool.vokra.reference]\nofficial_source_revision = '" + "a" * 40 + "'\n",
            encoding="utf-8",
        )
        lock.write_text("version = 1\nrevision = 3\nrequires-python = '==3.12.*'\n", encoding="utf-8")
        source = root / "source"
        source.mkdir()
        orchestrator_root = root / "orchestrator"
        orchestrator_root.mkdir()
        orchestrator_archive, orchestrator_site, orchestrator_scripts, _, orchestrator_locked = _fake_relocated_wheel(orchestrator_root)
        project_sha = hashlib.sha256(project.read_bytes()).hexdigest()
        lock_sha = hashlib.sha256(lock.read_bytes()).hexdigest()
        archive_sha = hashlib.sha256(orchestrator_archive.read_bytes()).hexdigest()
        url = f"https://files.pythonhosted.org/packages/{orchestrator_archive.name}"
        manifest = {
            "format": "vokra-vibevoice-selected-wheel-manifest-v1",
            "platform": {"system": "Linux", "machine": "x86_64", "python": "3.12"},
            "project_sha256": project_sha,
            "uv_lock_sha256": lock_sha,
            "artifacts": [{"name": "demo", "version": "1.0", "url": url, "filename": orchestrator_archive.name, "sha256": archive_sha, "bytes": orchestrator_archive.stat().st_size, "path": str(orchestrator_archive)}],
        }
        manifest_path = root / "synthetic-manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        args = argparse.Namespace(project=project, lock=lock, site_packages=orchestrator_site, venv_root=orchestrator_root / "venv", scripts_root=orchestrator_scripts, selected_wheel_manifest=manifest_path, source_root=source, output=root / "synthetic-report.json")
        locked_row = {"demo": orchestrator_locked}
        with mock.patch.object(sys.modules[__name__], "validate_project"), mock.patch.object(sys.modules[__name__], "validate_lock", return_value=locked_row), mock.patch.object(sys.modules[__name__], "source_identity", return_value={"revision": "a" * 40}), mock.patch.object(platform, "system", return_value="Linux"), mock.patch.object(platform, "machine", return_value="x86_64"), mock.patch.object(sys, "executable", str(orchestrator_scripts / "python")), mock.patch.dict(os.environ, {"VOKRA_PUBLISH_ON_VAST": "1"}, clear=False):
            report = audit(args)
        assert report["status"] == STATUS and report["execution"]["publication"] == "NO_UPLOAD"
        run_full_stdlib_regression_suite()
    print("audit_installed_closure self-test: PASS (synthetic wheel/RECORD/relocation/security/orchestrator cases)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--site-packages", type=Path)
    parser.add_argument("--venv-root", type=Path)
    parser.add_argument("--scripts-root", type=Path)
    parser.add_argument("--selected-wheel-manifest", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    required = (args.project, args.lock, args.site_packages, args.venv_root, args.scripts_root, args.selected_wheel_manifest, args.source_root, args.output)
    if any(value is None for value in required):
        parser.error("audit mode requires --project --lock --site-packages --venv-root --scripts-root --selected-wheel-manifest --source-root --output")
    try:
        report = audit(args)
        write_report(Path(args.output), report)
    except AuditError as error:
        print(f"audit: BLOCKED: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "publication": report["execution"]["publication"], "packages": len(report["packages"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
