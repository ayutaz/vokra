#!/usr/bin/env -S uv run --frozen --project tools/parity/dia_1_6b_reference python
"""Collect factual dependency, publisher-license, and native-payload evidence.

This auditor is deliberately independent of the Dia adapter.  It reads only
the dedicated ``pyproject.toml``/``uv.lock`` and installed distribution
metadata/files.  It never imports torch, Dia, DAC, Vokra, or any model
checkpoint.  The Linux x86_64 Python 3.12 closure is selected from the lock's
resolution markers and every selected distribution must match exactly.

The report is factual evidence, not a license classification or owner
sign-off.  The execution gate remains ``BLOCKED_UNREVIEWED_TRANSITIVE`` until
the collected native/bundled license facts have received an independent
review.  A missing publisher file may be recovered only from that package's
exact locked PyPI sdist; no other network path is permitted.
"""

from __future__ import annotations

import argparse
import ast
import base64
from collections import Counter
import hashlib
import importlib.metadata as metadata
import io
import json
import os
import platform
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import zipfile


SCHEMA = "vokra-dia-dependency-audit-v1"
COMPACT_SCHEMA = "vokra-dia-dependency-audit-compact-v1"
PROJECT_NAME = "vokra-dia-1-6b-reference"
PROJECT_VERSION = "0.1.0"
LOCK_SCHEMA = "uv-lock-v1-python312"
LOCK_SHA256 = "ccdfaf4cfedd7780f8c1032a42341f28ac56bec7353f4563f9a1b44b764cf29c"
PYPROJECT_SHA256 = "56430b6f50620df9ce3383f535dec1755843a4a9bab9758e34cf69e9913b6fc2"
GATE_STATUS = "BLOCKED_UNREVIEWED_TRANSITIVE"
PUBLICATION = "NO_UPLOAD"
ALLOWED_REGISTRIES = {
    "https://pypi.org/simple": "files.pythonhosted.org",
    "https://download.pytorch.org/whl/cpu": "download-r2.pytorch.org",
}
EXPECTED_DIRECT_DEPENDENCIES = {
    "einops": "0.8.2",
    "gguf": "0.19.0",
    "huggingface-hub": "0.30.2",
    "numpy": "2.2.5",
    "pydantic": "2.11.3",
    "soundfile": "0.13.1",
    "torch": "2.6.0",
    "torchaudio": "2.6.0",
}
EXPECTED_LINUX_DIRECT_IDS = {
    "einops==0.8.2",
    "gguf==0.19.0",
    "huggingface-hub==0.30.2",
    "numpy==2.2.5",
    "pydantic==2.11.3",
    "soundfile==0.13.1",
    "torch==2.6.0+cpu",
    "torchaudio==2.6.0+cpu",
}
LICENSE_NAMES = {"license", "licence", "copying", "notice", "copyright"}
NATIVE_SUFFIXES = {".so", ".dylib", ".dll", ".pyd", ".a"}
ELF_MAGIC = b"\x7fELF"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
PYPI_HOST = "files.pythonhosted.org"
MAX_LICENSE_BYTES = 2 * 1024 * 1024
MAX_SDIST_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_MEMBER_BYTES = 8 * 1024 * 1024
MAX_ARCHIVE_TOTAL_BYTES = 128 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 10_000
MAX_REDIRECTS = 3


class AuditError(RuntimeError):
    """A malformed or unreviewed closure fact."""


class SdistError(AuditError):
    """An exact locked sdist fallback could not be verified."""

    def __init__(self, message: str, requests: int = 1) -> None:
        super().__init__(message)
        self.requests = requests


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value.strip()).casefold()


def normalized_version(value: str) -> str:
    return re.sub(r"\s+", "", value.strip()).casefold()


def identity(name: str, version: str) -> str:
    return f"{normalized_name(name)}=={normalized_version(version)}"


def regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise AuditError(f"{label} is missing, non-regular, or symlinked: {path}")


def safe_project(path: Path) -> None:
    if not path.is_absolute() or any(part in {".", ".."} for part in path.parts[1:]):
        raise AuditError("--project must be an absolute path without dot components")
    regular_file(path / "pyproject.toml", "pyproject.toml")
    regular_file(path / "uv.lock", "uv.lock")


def safe_output(path: Path, project: Path) -> None:
    if not path.is_absolute() or any(part in {".", ".."} for part in path.parts[1:]):
        raise AuditError("--output must be an absolute path without dot components")
    if path.exists() or path.is_symlink():
        raise AuditError(f"--output must not exist: {path}")
    for cursor in (path.parent, *path.parent.parents):
        if cursor.is_symlink():
            raise AuditError(f"--output has symlinked ancestry: {cursor}")
        if cursor == Path(cursor.anchor):
            break
    resolved = path.resolve(strict=False)
    for protected, label in ((project, "Dia project"), (project.parents[2], "Vokra checkout")):
        try:
            resolved.relative_to(protected.resolve(strict=True))
        except ValueError:
            continue
        raise AuditError(f"--output overlaps the {label}")


def artifact_url(value: Any, registry: str, label: str) -> None:
    if not isinstance(value, dict):
        raise AuditError(f"{label} artifact is malformed")
    required = {"url", "hash", "upload-time"}
    if "size" in value:
        required.add("size")
    if set(value) != required:
        raise AuditError(f"{label} artifact keys drifted")
    url = value.get("url")
    parsed = urlsplit(url) if isinstance(url, str) else None
    expected_prefix = "/packages/" if registry == "https://pypi.org/simple" else "/whl/cpu/"
    if (
        parsed is None
        or parsed.scheme != "https"
        or parsed.hostname != ALLOWED_REGISTRIES[registry]
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.port not in (None, 443)
        or not parsed.path.startswith(expected_prefix)
        or any(part in {"", ".", ".."} for part in parsed.path.split("/")[1:])
    ):
        raise AuditError(f"{label} artifact URL is outside its reviewed registry")
    if not isinstance(value.get("hash"), str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value["hash"]):
        raise AuditError(f"{label} artifact hash is malformed")
    if not isinstance(value.get("upload-time"), str) or not value["upload-time"].strip():
        raise AuditError(f"{label} artifact upload-time is malformed")
    if "size" in value and (isinstance(value["size"], bool) or not isinstance(value["size"], int) or value["size"] <= 0):
        raise AuditError(f"{label} artifact size is malformed")


def lock_rows(lock: dict[str, Any], project: dict[str, Any]) -> list[dict[str, Any]]:
    if set(lock) != {"version", "revision", "requires-python", "resolution-markers", "package"}:
        raise AuditError("uv.lock top-level schema drifted")
    if lock["version"] != 1 or lock["revision"] != 3 or lock["requires-python"] != "==3.12.*":
        raise AuditError("uv.lock resolver identity drifted")
    if not isinstance(lock["resolution-markers"], list) or not isinstance(lock["package"], list):
        raise AuditError("uv.lock resolver markers/package table malformed")
    root = project.get("project")
    if not isinstance(root, dict) or root.get("name") != PROJECT_NAME or root.get("version") != PROJECT_VERSION:
        raise AuditError("Dia reference project identity drifted")
    seen: set[tuple[str, str, str]] = set()
    virtual = 0
    result: list[dict[str, Any]] = []
    for row in lock["package"]:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not isinstance(row.get("version"), str):
            raise AuditError("uv.lock package identity is malformed")
        source = row.get("source")
        if not isinstance(source, dict) or set(source) not in ({"virtual"}, {"registry"}):
            raise AuditError(f"uv.lock source is malformed: {row['name']}")
        key = (row["name"], row["version"], canonical(source))
        if key in seen:
            raise AuditError(f"uv.lock duplicate package identity: {key!r}")
        seen.add(key)
        if "virtual" in source:
            virtual += 1
            if source != {"virtual": "."} or row["name"] != PROJECT_NAME or row["version"] != PROJECT_VERSION:
                raise AuditError("uv.lock virtual root is not bound to pyproject")
        else:
            registry = source.get("registry")
            if registry not in ALLOWED_REGISTRIES:
                raise AuditError(f"uv.lock registry is not reviewed: {registry!r}")
            artifacts = []
            if "sdist" in row:
                artifact_url(row["sdist"], registry, f"{row['name']} sdist")
                artifacts.append("sdist")
            wheels = row.get("wheels", [])
            if not isinstance(wheels, list):
                raise AuditError(f"{row['name']} wheels are malformed")
            for index, wheel in enumerate(wheels):
                artifact_url(wheel, registry, f"{row['name']} wheel {index}")
                artifacts.append("wheel")
            if not artifacts:
                raise AuditError(f"{row['name']} has no locked artifacts")
        markers = row.get("resolution-markers", [])
        if not isinstance(markers, list) or any(not isinstance(marker, str) or not marker.strip() for marker in markers):
            raise AuditError(f"{row['name']} resolution markers are malformed")
        dependencies = row.get("dependencies", [])
        if not isinstance(dependencies, list) or any(not isinstance(dep, dict) or not isinstance(dep.get("name"), str) for dep in dependencies):
            raise AuditError(f"{row['name']} dependencies are malformed")
        result.append({
            "name": row["name"],
            "version": row["version"],
            "source": source,
            "resolution_markers": sorted(markers),
            "dependencies": dependencies,
        })
    if virtual != 1:
        raise AuditError("uv.lock must contain exactly one virtual root")
    return sorted(result, key=lambda row: (normalized_name(row["name"]), normalized_version(row["version"]), canonical(row["source"])))


def marker_active(marker: str) -> bool:
    """Evaluate only the lock marker grammar against this VAST target."""
    environment = {
        "implementation_name": "cpython",
        "implementation_version": platform.python_version(),
        "os_name": "posix",
        "platform_machine": "x86_64",
        "platform_python_implementation": "CPython",
        "platform_release": platform.release(),
        "platform_system": "Linux",
        "platform_version": platform.version(),
        "python_full_version": platform.python_version(),
        "python_version": platform.python_version()[:3],
        "sys_platform": "linux",
    }
    try:
        expression = ast.parse(marker, mode="eval").body

        def evaluate(node: ast.AST) -> Any:
            if isinstance(node, ast.Name) and node.id in environment:
                return environment[node.id]
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return node.value
            if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
                values = [bool(evaluate(value)) for value in node.values]
                return all(values) if isinstance(node.op, ast.And) else any(values)
            if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
                left, right = evaluate(node.left), evaluate(node.comparators[0])
                if isinstance(node.ops[0], ast.Eq):
                    return left == right
                if isinstance(node.ops[0], ast.NotEq):
                    return left != right
            raise AuditError("resolution marker uses an unsupported expression")

        return bool(evaluate(expression))
    except (SyntaxError, ValueError, TypeError, AuditError) as error:
        raise AuditError(f"resolution marker cannot be evaluated: {marker!r}") from error


def active_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve the Linux closure from the virtual root and dependency markers.

    A lock row without ``resolution-markers`` is not necessarily installed on
    this target (for example, colorama is reachable only through tqdm's
    Windows-marked dependency).  Traversing the root is therefore required;
    filtering rows by their own marker list alone overstates the installed
    closure.
    """
    by_name: dict[str, list[dict[str, Any]]] = {}
    virtual: dict[str, Any] | None = None
    for row in rows:
        if row["source"] == {"virtual": "."}:
            virtual = row
            continue
        if row["resolution_markers"] and not any(marker_active(marker) for marker in row["resolution_markers"]):
            continue
        by_name.setdefault(normalized_name(row["name"]), []).append(row)
    if virtual is None:
        raise AuditError("uv.lock virtual root is missing")
    selected: dict[tuple[str, str, str], dict[str, Any]] = {}
    visiting: set[tuple[str, str, str]] = set()

    def visit(row: dict[str, Any]) -> None:
        key = (row["name"], row["version"], canonical(row["source"]))
        if key in selected:
            return
        if key in visiting:
            raise AuditError(f"dependency cycle cannot be resolved: {row['name']}")
        visiting.add(key)
        for dependency in row["dependencies"]:
            marker = dependency.get("marker")
            if marker is not None and (not isinstance(marker, str) or not marker_active(marker)):
                continue
            name = dependency.get("name")
            if not isinstance(name, str):
                raise AuditError(f"dependency name is malformed: {row['name']}")
            candidates = by_name.get(normalized_name(name), [])
            if isinstance(dependency.get("version"), str):
                candidates = [candidate for candidate in candidates if normalized_version(candidate["version"]) == normalized_version(dependency["version"])]
            if isinstance(dependency.get("source"), dict):
                candidates = [candidate for candidate in candidates if candidate["source"] == dependency["source"]]
            if len(candidates) != 1:
                raise AuditError(f"dependency does not resolve to one active lock row: {name}")
            visit(candidates[0])
        visiting.remove(key)
        selected[key] = row

    visit(virtual)
    return sorted((row for row in selected.values() if row["source"] != {"virtual": "."}), key=lambda row: (normalized_name(row["name"]), normalized_version(row["version"]), canonical(row["source"])))


def project_contract(project: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    safe_project(project)
    pyproject_bytes = (project / "pyproject.toml").read_bytes()
    lock_bytes = (project / "uv.lock").read_bytes()
    if sha256_bytes(pyproject_bytes) != PYPROJECT_SHA256 or sha256_bytes(lock_bytes) != LOCK_SHA256:
        raise AuditError("Dia pyproject.toml or uv.lock bytes differ from the reviewed contract")
    try:
        pyproject = tomllib.loads(pyproject_bytes.decode("utf-8"))
        lock = tomllib.loads(lock_bytes.decode("utf-8"))
    except (UnicodeError, tomllib.TOMLDecodeError) as error:
        raise AuditError(f"Dia closure TOML is unreadable: {error}") from error
    reference = pyproject.get("tool", {}).get("vokra", {}).get("reference", {})
    if reference.get("dependency_license_audit") != GATE_STATUS:
        raise AuditError("Dia dependency license gate is not fail-closed")
    project_dependencies = pyproject.get("project", {}).get("dependencies")
    if not isinstance(project_dependencies, list):
        raise AuditError("Dia direct dependency list is missing")
    expected_specs = {f"{name}=={version}" for name, version in EXPECTED_DIRECT_DEPENDENCIES.items()}
    if set(project_dependencies) != expected_specs:
        raise AuditError("Dia direct dependency versions differ from the reviewed contract")
    rows = lock_rows(lock, pyproject)
    active = active_rows(rows)
    active_ids = {identity(row["name"], row["version"]) for row in active}
    if not EXPECTED_LINUX_DIRECT_IDS.issubset(active_ids):
        raise AuditError("Linux x86_64 active closure is missing a direct dependency")
    contract = {
        "project": PROJECT_NAME,
        "project_version": PROJECT_VERSION,
        "lock_schema": LOCK_SCHEMA,
        "pyproject_bytes": len(pyproject_bytes),
        "pyproject_sha256": PYPROJECT_SHA256,
        "uv_lock_bytes": len(lock_bytes),
        "uv_lock_sha256": LOCK_SHA256,
        "gate_status": GATE_STATUS,
        "allowed_registries": sorted(ALLOWED_REGISTRIES),
        "direct_dependency_versions": dict(EXPECTED_DIRECT_DEPENDENCIES),
        "all_lock_rows_sha256": sha256_bytes(canonical(rows).encode("utf-8")),
        "active_linux_x86_64_rows_sha256": sha256_bytes(canonical(active).encode("utf-8")),
    }
    return pyproject, lock, rows, contract


def installed_distributions() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if isinstance(name, str) and name.strip() and isinstance(dist.version, str) and dist.version.strip():
            records.append({"distribution": dist, "name": name, "version": dist.version, "identity": identity(name, dist.version), "location": str(Path(dist.locate_file("")))})
    return sorted(records, key=lambda row: (row["identity"], row["location"]))


def metadata_facts(dist: metadata.Distribution) -> dict[str, Any]:
    def clean(value: Any) -> str | None:
        return value.strip() if isinstance(value, str) and value.strip() else None

    classifiers = sorted(value.removeprefix("License :: ") for value in (dist.metadata.get_all("Classifier") or []) if value.startswith("License :: "))
    return {
        "license_declared": clean(dist.metadata.get("License")),
        "license_expression_declared": clean(dist.metadata.get("License-Expression")),
        "license_classifiers_declared": classifiers,
        "requires_python": clean(dist.metadata.get("Requires-Python")),
    }


def entry_path(dist: metadata.Distribution, entry: Any) -> Path | None:
    root = Path(dist.locate_file(""))
    lexical = Path(dist.locate_file(entry))
    try:
        relative = lexical.relative_to(root)
    except ValueError:
        return None
    if root.is_symlink() or any((root / part).is_symlink() for part in relative.parts):
        return None
    path = lexical.resolve(strict=False)
    try:
        path.relative_to(root.resolve(strict=False))
    except ValueError:
        return None
    return path


def is_license_path(value: str) -> bool:
    name = Path(value).name.casefold()
    return name in LICENSE_NAMES or any(name.startswith(token + suffix) for token in LICENSE_NAMES for suffix in (".", "-", "_"))


def publisher_files(dist: metadata.Distribution) -> tuple[list[dict[str, Any]], list[str]]:
    found: list[dict[str, Any]] = []
    unsafe: list[str] = []
    for entry in sorted(dist.files or [], key=str):
        relative = str(entry)
        if not is_license_path(relative):
            continue
        path = entry_path(dist, entry)
        if path is None or path.is_symlink() or not path.is_file():
            unsafe.append(relative)
            continue
        size = path.stat().st_size
        if size > MAX_LICENSE_BYTES:
            unsafe.append(f"{relative}:oversized")
            continue
        content = path.read_bytes()
        found.append({"path": relative, "bytes": len(content), "sha256": sha256_bytes(content), "content_base64": base64.b64encode(content).decode("ascii"), "source": "installed-distribution"})
    return found, unsafe


def _archive_member_name(name: str) -> str:
    if isinstance(name, str) and name.endswith("/"):
        name = name[:-1]
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name or name.startswith("/") or any(part in {"", ".", ".."} for part in name.split("/")):
        raise AuditError("locked sdist contains an unsafe member path")
    return name


def archive_license_files(body: bytes, url: str) -> list[dict[str, Any]]:
    if len(body) > MAX_SDIST_BYTES:
        raise AuditError("locked sdist is oversized")
    suffix = urlsplit(url).path.casefold()
    if suffix.endswith(".zip"):
        kind = "zip"
    elif suffix.endswith((".tar.gz", ".tgz")):
        kind = "tar.gz"
    elif suffix.endswith((".tar.bz2", ".tbz2")):
        kind = "tar.bz2"
    elif suffix.endswith((".tar.xz", ".txz")):
        kind = "tar.xz"
    else:
        raise AuditError("locked sdist archive format is unsupported")
    found: list[dict[str, Any]] = []
    names: set[str] = set()
    total = 0

    def add(name: str, payload: bytes) -> None:
        if not is_license_path(name):
            return
        if len(payload) > MAX_ARCHIVE_MEMBER_BYTES:
            raise AuditError("locked sdist license member is oversized")
        found.append({"path": name, "bytes": len(payload), "sha256": sha256_bytes(payload), "content_base64": base64.b64encode(payload).decode("ascii"), "source": "locked-sdist"})

    try:
        if kind == "zip":
            with zipfile.ZipFile(io.BytesIO(body)) as archive:
                members = archive.infolist()
                if len(members) > MAX_ARCHIVE_MEMBERS:
                    raise AuditError("locked sdist has too many members")
                for info in members:
                    name = _archive_member_name(info.filename)
                    if name in names:
                        raise AuditError("locked sdist has duplicate members")
                    names.add(name)
                    mode = (info.external_attr >> 16) & 0o170000
                    if mode not in (0, stat.S_IFREG, stat.S_IFDIR):
                        raise AuditError("locked sdist has a special member")
                    if info.is_dir() or mode == stat.S_IFDIR:
                        continue
                    if info.file_size < 0 or info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                        raise AuditError("locked sdist member is oversized")
                    total += info.file_size
                    if total > MAX_ARCHIVE_TOTAL_BYTES:
                        raise AuditError("locked sdist archive is oversized")
                    payload = archive.read(info)
                    if len(payload) != info.file_size:
                        raise AuditError("locked sdist member is truncated")
                    add(name, payload)
        else:
            mode = {"tar.gz": "r:gz", "tar.bz2": "r:bz2", "tar.xz": "r:xz"}[kind]
            with tarfile.open(fileobj=io.BytesIO(body), mode=mode) as archive:
                for index, member in enumerate(archive, 1):
                    if index > MAX_ARCHIVE_MEMBERS:
                        raise AuditError("locked sdist has too many members")
                    name = _archive_member_name(member.name)
                    if name in names:
                        raise AuditError("locked sdist has duplicate members")
                    names.add(name)
                    if member.isdir():
                        continue
                    if not member.isfile() or member.size < 0 or member.size > MAX_ARCHIVE_MEMBER_BYTES:
                        raise AuditError("locked sdist has an unsafe member")
                    total += member.size
                    if total > MAX_ARCHIVE_TOTAL_BYTES:
                        raise AuditError("locked sdist archive is oversized")
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise AuditError("locked sdist member cannot be read")
                    payload = stream.read(MAX_ARCHIVE_MEMBER_BYTES + 1)
                    if len(payload) != member.size:
                        raise AuditError("locked sdist member is truncated")
                    add(name, payload)
    except (OSError, EOFError, tarfile.TarError, zipfile.BadZipFile) as error:
        raise AuditError(f"locked sdist archive is unreadable: {type(error).__name__}") from error
    if not found:
        raise AuditError("locked sdist contains no LICENSE/NOTICE/COPYING member")
    return found


def fetch_locked_sdist(row: dict[str, Any], raw: dict[str, Any], fetcher: Callable[[str], tuple[str, bytes]] | None = None) -> dict[str, Any]:
    source = row["source"]
    if source != {"registry": "https://pypi.org/simple"} or not isinstance(raw.get("sdist"), dict):
        raise SdistError(f"{row['name']} has no reviewed PyPI sdist fallback")
    artifact = raw["sdist"]
    artifact_url(artifact, source["registry"], f"{row['name']} sdist")
    requested = artifact["url"]
    trace = [requested]
    try:
        if fetcher is not None:
            final, body = fetcher(requested)
        else:
            class Redirects(HTTPRedirectHandler):
                def redirect_request(self, request: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Request:
                    if len(trace) > MAX_REDIRECTS:
                        raise AuditError("locked sdist redirect limit exceeded")
                    parsed = urlsplit(newurl)
                    if parsed.scheme != "https" or parsed.hostname != PYPI_HOST or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.port not in (None, 443) or not parsed.path.startswith("/packages/"):
                        raise AuditError("locked sdist redirect left reviewed host/path")
                    trace.append(newurl)
                    return super().redirect_request(request, fp, code, msg, headers, newurl)

            with build_opener(Redirects()).open(Request(requested, headers={"User-Agent": "vokra-dia-dependency-audit/1"}), timeout=30) as response:
                final = response.geturl()
                body = response.read(MAX_SDIST_BYTES + 1)
        if not isinstance(body, bytes) or len(body) != artifact["size"] or sha256_bytes(body) != artifact["hash"].removeprefix("sha256:"):
            raise AuditError("locked sdist bytes do not match uv.lock")
        final_parsed = urlsplit(final)
        if final_parsed.scheme != "https" or final_parsed.hostname != PYPI_HOST or final_parsed.query or final_parsed.fragment or final_parsed.path != urlsplit(requested).path:
            raise AuditError("locked sdist final URL is not the exact locked path")
        files = archive_license_files(body, requested)
    except (AuditError, OSError, ValueError, HTTPError, URLError, tarfile.TarError, zipfile.BadZipFile) as error:
        raise SdistError(f"locked sdist fallback failed: {type(error).__name__}") from error
    return {"status": "ACQUIRED_LOCKED_SDIST_LICENSE_BYTES", "requested_url": requested, "final_url": final, "redirect_trace": trace, "bytes": len(body), "sha256": sha256_bytes(body), "license_files": files, "auditor_network_requests": 1}


def native_facts(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            magic = stream.read(4)
    except OSError as error:
        return {"format": "unknown", "inspection": "error", "error_type": type(error).__name__}
    if magic != ELF_MAGIC:
        return {"format": "non-elf", "inspection": "not-applicable", "needed": []}
    try:
        result = subprocess.run(["readelf", "-d", str(path)], capture_output=True, text=True, check=False, timeout=60)
    except (OSError, subprocess.SubprocessError) as error:
        return {"format": "elf", "inspection": "error", "needed": [], "error_type": type(error).__name__}
    needed = sorted(set(match.group(1) for match in re.finditer(r"\(NEEDED\).*?\[([^]]+)\]", result.stdout)))
    return {"format": "elf", "inspection": "ok" if result.returncode == 0 else "error", "readelf_returncode": result.returncode, "needed": needed, "stderr": result.stderr.strip()[-500:] if result.returncode else None}


def native_files(dist: metadata.Distribution) -> tuple[list[dict[str, Any]], list[str]]:
    found: list[dict[str, Any]] = []
    unsafe: list[str] = []
    for entry in sorted(dist.files or [], key=str):
        relative = str(entry)
        basename = Path(relative).name.casefold()
        candidate = Path(basename).suffix in NATIVE_SUFFIXES or ".so." in basename
        path = entry_path(dist, entry)
        if path is None:
            if candidate:
                unsafe.append(f"{relative}:outside-distribution-root")
            continue
        if path.is_symlink() or not path.is_file():
            if candidate:
                unsafe.append(f"{relative}:non-regular-or-symlink")
            continue
        try:
            with path.open("rb") as stream:
                magic = stream.read(4)
            if not candidate and magic != ELF_MAGIC:
                continue
            facts = native_facts(path)
            found.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path), "native": facts})
            if facts.get("inspection") == "error":
                unsafe.append(f"{relative}:readelf-failed")
        except OSError as error:
            unsafe.append(f"{relative}:read-failed:{type(error).__name__}")
    return found, unsafe


def raw_rows_by_identity(lock: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows = lock.get("package")
    if not isinstance(rows, list):
        raise AuditError("uv.lock package table is missing")
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("name"), str) and isinstance(row.get("version"), str) and isinstance(row.get("source"), dict):
            result[(row["name"], row["version"], canonical(row["source"]))] = row
    return result


def inspect_package(row: dict[str, Any], raw: dict[str, Any], candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str], int]:
    key = identity(row["name"], row["version"])
    result: dict[str, Any] = {"lock": {"name": row["name"], "version": row["version"], "source": row["source"]}, "installed": None}
    failures: list[str] = []
    if len(candidates) != 1:
        return result, [f"installed closure missing or duplicated: {key}"], 0
    dist = candidates[0]["distribution"]
    licenses, unsafe_licenses = publisher_files(dist)
    native, unsafe_native = native_files(dist)
    fallback: dict[str, Any] | None = None
    requests = 0
    if not licenses:
        try:
            fallback = fetch_locked_sdist(row, raw)
            requests = int(fallback["auditor_network_requests"])
        except SdistError as error:
            requests = error.requests
            failures.append(f"locked sdist license fallback blocked: {key}:{error}")
    result["installed"] = {"name": dist.metadata.get("Name"), "version": dist.version, "identity": key, "location": candidates[0]["location"], **metadata_facts(dist), "publisher_license_files": licenses, "locked_sdist_license_fallback": fallback, "native_files": native, "bundled_libraries": native}
    if not licenses and not (fallback and fallback.get("license_files")):
        failures.append(f"publisher license bytes missing: {key}")
    failures.extend(f"unsafe publisher license path: {key}:{value}" for value in unsafe_licenses)
    failures.extend(f"unsafe native path: {key}:{value}" for value in unsafe_native)
    return result, failures, requests


def repository_facts(project: Path) -> tuple[dict[str, Any], list[str]]:
    root = project.parents[2]
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise AuditError(f"git checkout identity unavailable: {type(error).__name__}") from error
    failures = []
    if not HEX40.fullmatch(head):
        failures.append("git HEAD is not a 40-hex commit")
    if status:
        failures.append("git checkout is dirty")
    return {"root": str(root), "head": head, "clean": not status, "audit_script_sha256": sha256_file(Path(__file__).resolve())}, failures


def top_level_native_facts(packages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten native rows while retaining their owning distribution identity."""
    return [
        {
            "package_identity": package["installed"]["identity"],
            "package_name": normalized_name(package["installed"]["name"]),
            "package_version": normalized_version(package["installed"]["version"]),
            **item,
        }
        for package in packages
        if package["installed"]
        for item in package["installed"]["native_files"]
    ]


def audit(project: Path) -> dict[str, Any]:
    pyproject, lock, rows, contract = project_contract(project)
    active = active_rows(rows)
    records = installed_distributions()
    by_id: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_id.setdefault(record["identity"], []).append(record)
    expected = [identity(row["name"], row["version"]) for row in active]
    actual = [record["identity"] for record in records]
    expected_counts, actual_counts = Counter(expected), Counter(actual)
    closure = {"expected": sorted(expected), "installed": sorted(actual), "missing": sorted((expected_counts - actual_counts).elements()), "unexpected": sorted((actual_counts - expected_counts).elements()), "duplicate_identities": sorted(key for key, count in actual_counts.items() if count > 1), "exact": not (expected_counts - actual_counts or actual_counts - expected_counts)}
    raw = raw_rows_by_identity(lock)
    packages: list[dict[str, Any]] = []
    failures: list[str] = []
    repository, repository_failures = repository_facts(project)
    failures.extend(repository_failures)
    requests = 0
    for row in active:
        raw_row = raw.get((row["name"], row["version"], canonical(row["source"])))
        if raw_row is None:
            raise AuditError(f"active lock row cannot bind to raw uv.lock: {row['name']}")
        package, package_failures, package_requests = inspect_package(row, raw_row, by_id.get(identity(row["name"], row["version"]), []))
        packages.append(package)
        failures.extend(package_failures)
        requests += package_requests
    if not closure["exact"]:
        failures.append("installed Linux x86_64 closure does not exactly match active uv.lock rows")
    if sys.platform != "linux" or platform.machine().casefold() not in {"x86_64", "amd64"}:
        failures.append(f"audit host is not Linux x86_64: {sys.platform}/{platform.machine()}")
    if sys.version_info[:2] != (3, 12):
        failures.append(f"audit Python is not 3.12: {platform.python_version()}")
    native = top_level_native_facts(packages)
    missing = sorted(package["lock"]["name"] for package in packages if package["installed"] and not package["installed"]["publisher_license_files"] and not (package["installed"].get("locked_sdist_license_fallback") or {}).get("license_files"))
    report = {
        "schema": SCHEMA,
        "status": "BLOCKED" if failures else "FACTS_COLLECTED_GATE_BLOCKED",
        "dependency_license_audit": GATE_STATUS,
        "publication": PUBLICATION,
        "audit_scope": "Linux x86_64 Python 3.12 installed dependency metadata, publisher LICENSE/NOTICE bytes, and native payload facts; model/source/checkpoint-free",
        "policy": {"license_classification": "NOT_PERFORMED", "owner_signoff": "NOT_PERFORMED", "model_acquisition": "NONE", "source_acquisition": "NONE", "torch_imported": False, "dia_imported": False, "vokra_imported": False, "cargo_invoked": False, "upload_performed": False},
        "repository": repository,
        "environment": {"python": platform.python_version(), "platform": sys.platform, "machine": platform.machine(), "readelf_required": True, "auditor_network_requests": requests, "auditor_network_scope": "exact locked PyPI sdist license fallback only"},
        "contract": contract,
        "lock_rows": {"all_rows": rows, "active_linux_x86_64_rows": active, "all_rows_accounted": True},
        "closure": closure,
        "packages": packages,
        "license_facts": {"packages": len(packages), "publisher_license_evidence_missing": missing, "publisher_bytes_recorded": sum(len(package["installed"]["publisher_license_files"]) for package in packages if package["installed"]), "classification": "raw installed metadata and publisher bytes only; no classification"},
        "native_facts": {"bundled_library_count": len(native), "files": native},
        "model_source_facts": {"requested_files": [], "non_license_requests": [], "proof": "collector has no model/source/checkpoint acquisition or import path"},
        "failures": sorted(set(failures)),
    }
    return report


def write_no_clobber(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor != -1:
            os.close(descriptor)


def blocked_report(error: BaseException) -> dict[str, Any]:
    """Build the minimal schema-preserving report for an aborted audit."""
    return {
        "schema": SCHEMA,
        "status": "BLOCKED",
        "dependency_license_audit": GATE_STATUS,
        "publication": PUBLICATION,
        "policy": {
            "license_classification": "NOT_PERFORMED",
            "owner_signoff": "NOT_PERFORMED",
            "model_acquisition": "NONE",
            "source_acquisition": "NONE",
            "torch_imported": False,
            "dia_imported": False,
            "vokra_imported": False,
            "cargo_invoked": False,
            "upload_performed": False,
        },
        "failures": [f"audit aborted: {type(error).__name__}: {error}"],
    }


def self_test() -> int:
    project = Path(__file__).resolve().parent
    _pyproject, lock, rows, contract = project_contract(project)
    active = active_rows(rows)
    assert contract["gate_status"] == GATE_STATUS
    assert blocked_report(AuditError("synthetic"))["publication"] == PUBLICATION
    original_installed_distributions = globals()["installed_distributions"]
    try:
        globals()["installed_distributions"] = lambda: []
        synthetic_report = audit(project)
    finally:
        globals()["installed_distributions"] = original_installed_distributions
    assert synthetic_report["publication"] == PUBLICATION
    assert synthetic_report["dependency_license_audit"] == GATE_STATUS
    assert isinstance(synthetic_report["native_facts"]["files"], list)
    assert set(EXPECTED_LINUX_DIRECT_IDS).issubset({identity(row["name"], row["version"]) for row in active})
    assert len(rows) == 34
    assert len(active) == 30
    assert "colorama==0.4.6" not in {identity(row["name"], row["version"]) for row in active}
    assert all(row["source"].get("registry") in ALLOWED_REGISTRIES for row in active)
    assert identity("Torch", "2.6.0+CPU") == "torch==2.6.0+cpu"
    synthetic_packages = [{"installed": {"identity": "numpy==2.2.5", "name": "NumPy", "version": "2.2.5", "native_files": [{"path": "numpy.libs/libx.so", "bytes": 1, "sha256": "a" * 64, "native": {}}]}}]
    flattened = top_level_native_facts(synthetic_packages)
    assert flattened[0]["package_identity"] == "numpy==2.2.5"
    assert flattened[0]["package_name"] == "numpy" and flattened[0]["package_version"] == "2.2.5"
    assert is_license_path("pkg/LICENSE.txt") and is_license_path("pkg/NOTICE") and is_license_path("pkg/COPYING")
    assert not is_license_path("pkg/README.md")
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        payload = b"license bytes"
        member = tarfile.TarInfo("demo-1/LICENSE")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    body = archive_buffer.getvalue()
    artifact = {"url": "https://files.pythonhosted.org/packages/aa/bb/demo-1.tar.gz", "hash": "sha256:" + sha256_bytes(body), "size": len(body), "upload-time": "2026-01-01T00:00:00Z"}
    row = {"name": "demo", "version": "1", "source": {"registry": "https://pypi.org/simple"}}
    fetched = fetch_locked_sdist(row, {**row, "sdist": artifact}, lambda url: (url, body))
    assert fetched["license_files"][0]["source"] == "locked-sdist"
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w") as archive:
        archive.writestr("demo-1/", "")
        archive.writestr("demo-1/NOTICE", "notice bytes")
    zip_files = archive_license_files(zip_buffer.getvalue(), "https://files.pythonhosted.org/packages/aa/bb/demo-1.zip")
    assert zip_files[0]["path"] == "demo-1/NOTICE"
    try:
        archive_license_files(body, "https://files.pythonhosted.org/packages/aa/bb/demo-1.whl")
    except AuditError:
        pass
    else:
        raise AssertionError("unsupported archive format accepted")
    temp_parent = "/private/tmp" if Path("/private/tmp").is_dir() and not Path("/private/tmp").is_symlink() else None
    with tempfile.TemporaryDirectory(prefix="dia-dependency-audit-selftest-", dir=temp_parent) as directory:
        root = Path(directory)
        output = root / "nested" / "audit.json"
        safe_output(output, project)
        write_no_clobber(output, b"{}\n")
        try:
            write_no_clobber(output, b"clobber\n")
        except FileExistsError:
            pass
        else:
            raise AssertionError("output clobber accepted")
    assert not any(token in canonical(lock) for token in ("librosa", "soxr", "gradio", "triton", "nvidia-"))
    print("dia dependency audit: self-test PASS (offline, model-free, no network)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="collect Dia locked dependency/license/native facts")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.project is not None or args.output is not None:
            parser.error("--self-test accepts no project/output")
        return self_test()
    if args.project is None or args.output is None:
        parser.error("--project and --output are required")
    safe_project(args.project)
    safe_output(args.output, args.project)
    try:
        report = audit(args.project)
    except (AuditError, OSError, UnicodeError, TypeError, ValueError, KeyError, AttributeError, subprocess.SubprocessError) as error:
        report = blocked_report(error)
    write_no_clobber(args.output, (canonical(report) + "\n").encode("utf-8"))
    if report.get("failures") or report.get("dependency_license_audit") != GATE_STATUS:
        print("dia dependency audit: BLOCKED", file=sys.stderr)
        return 2
    print(f"DIA_DEPENDENCY_AUDIT status={report['status']} gate={GATE_STATUS} output={args.output}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
