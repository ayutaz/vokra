#!/usr/bin/env python3
"""Model-free factual audit for the pinned NeuTTS Air reference closure.

This audit runs after an exact frozen environment sync.  It imports only
standard-library modules, never imports a model package, and never downloads
weights.  Network access is limited to exact locked PyPI sdists used only for
missing publisher files and the four fixed primary-source LICENSE paths.
"""

from __future__ import annotations

import argparse
import base64
import copy
from collections import Counter
import hashlib
import importlib.metadata as metadata
import io
import json
import platform
import posixpath
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import tomllib

try:
    import preflight_gate
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from tools.parity.neutts_air import preflight_gate


SCHEMA = "vokra-neutts-air-dependency-audit-v1"
PYPI_FILE_HOST = "files.pythonhosted.org"
HF_LICENSE_HOSTS = {"cdn-lfs.huggingface.co", "cdn-lfs-us-1.hf.co"}
LICENSE_FILE_NAMES = {"license", "copying", "notice", "copyright"}
ELF_MAGIC = b"\x7fELF"
NATIVE_SUFFIXES = {".so", ".dylib", ".dll", ".pyd"}
MAX_LICENSE_BYTES = 2 * 1024 * 1024
MAX_SDIST_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_MEMBER_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_TOTAL_BYTES = 128 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 4096
MAX_SDIST_LICENSE_TOTAL_BYTES = 4 * 1024 * 1024
MAX_SDIST_REDIRECTS = 3
MARKER_VARIABLES = {"sys_platform", "platform_machine", "implementation_name"}
MARKER_TOKEN = re.compile(r"\s*(?:(and|or|==|!=|\(|\))|([A-Za-z_][A-Za-z0-9_]*)|('(?:[^'\\]|\\.)*'))")


class AuditError(ValueError):
    """A fail-closed contract, transport, or factual evidence error."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def identity(name: str, version: str) -> str:
    normalized = re.sub(r"[-_.]+", "-", name.strip()).casefold()
    return f"{normalized}=={version.strip().casefold()}"


def load_json(path: Path) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise AuditError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)
    except (OSError, UnicodeError, json.JSONDecodeError, AuditError) as exc:
        raise AuditError(f"cannot read JSON {path}: {exc}") from exc


def _marker_matches(marker: Any, environment: dict[str, str]) -> bool:
    if marker is None:
        return True
    if not isinstance(marker, str) or not marker.strip():
        raise AuditError("lock marker is malformed")
    tokens: list[tuple[str, str]] = []
    position = 0
    while position < len(marker):
        match = MARKER_TOKEN.match(marker, position)
        if match is None:
            if marker[position:].strip():
                raise AuditError(f"unsupported lock marker grammar: {marker}")
            break
        if match.group(1) is not None:
            tokens.append(("operator", match.group(1)))
        elif match.group(2) is not None:
            tokens.append(("identifier", match.group(2)))
        else:
            tokens.append(("string", match.group(3)[1:-1]))
        position = match.end()
    cursor = 0

    def peek(value: str | None = None) -> tuple[str, str] | None:
        if cursor >= len(tokens):
            return None
        token = tokens[cursor]
        return token if value is None or token[1] == value else None

    def take(value: str | None = None) -> tuple[str, str]:
        nonlocal cursor
        token = peek(value)
        if token is None:
            raise AuditError(f"unsupported lock marker grammar: {marker}")
        cursor += 1
        return token

    def atom() -> bool:
        if peek("(") is not None:
            take("(")
            result = disjunction()
            take(")")
            return result
        variable = take()[1]
        if variable not in MARKER_VARIABLES:
            raise AuditError(f"unsupported lock marker variable: {variable}")
        operator = take()[1]
        if operator not in {"==", "!="}:
            raise AuditError(f"unsupported lock marker operator: {operator}")
        literal = take()
        if literal[0] != "string" or "\\" in literal[1]:
            raise AuditError(f"unsupported lock marker literal: {marker}")
        result = environment[variable] == literal[1]
        return result if operator == "==" else not result

    def conjunction() -> bool:
        result = atom()
        while peek("and") is not None:
            take("and")
            result = atom() and result
        return result

    def disjunction() -> bool:
        result = conjunction()
        while peek("or") is not None:
            take("or")
            result = conjunction() or result
        return result

    result = disjunction()
    if cursor != len(tokens):
        raise AuditError(f"unsupported lock marker grammar: {marker}")
    return result


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    return (identity(row["name"], row["version"]), canonical_json(row["source"]))


def _active_lock_graph(lock: dict[str, Any]) -> tuple[set[tuple[str, str]], dict[tuple[str, str], str]]:
    rows = lock.get("package")
    if not isinstance(rows, list) or not rows:
        raise AuditError("lock package rows are malformed")
    row_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    by_name: dict[str, list[dict[str, Any]]] = {}
    resolved: dict[tuple[str, str], bool] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not isinstance(row.get("version"), str) or not isinstance(row.get("source"), dict):
            raise AuditError("lock package identity is malformed")
        key = _row_key(row)
        if key in row_by_key:
            raise AuditError(f"duplicate lock package identity: {row['name']}=={row['version']}")
        row_by_key[key] = row
        by_name.setdefault(identity(row["name"], "").split("==", 1)[0], []).append(row)
        markers = row.get("resolution-markers")
        if markers is None:
            resolved[key] = True
        elif isinstance(markers, list) and markers:
            matches = [_marker_matches(marker, {"sys_platform": "linux", "platform_machine": "x86_64", "implementation_name": "cpython"}) for marker in markers]
            if sum(matches) > 1:
                raise AuditError(f"ambiguous package resolution-markers: {row['name']}")
            resolved[key] = any(matches)
        else:
            raise AuditError(f"package resolution-markers are malformed: {row['name']}")
        dependencies = row.get("dependencies", [])
        if not isinstance(dependencies, list):
            raise AuditError(f"lock dependencies are malformed: {row['name']}")
        for dependency in dependencies:
            if not isinstance(dependency, dict) or not isinstance(dependency.get("name"), str) or not dependency["name"].strip():
                raise AuditError(f"lock dependency is malformed: {row['name']}")
            _marker_matches(dependency.get("marker"), {"sys_platform": "linux", "platform_machine": "x86_64", "implementation_name": "cpython"})
    roots = [row for row in rows if row.get("source") == {"virtual": "."}]
    if len(roots) != 1:
        raise AuditError("lock must contain exactly one virtual project row")
    active: set[tuple[str, str]] = set()
    visiting: set[tuple[str, str]] = set()

    def visit(row: dict[str, Any]) -> None:
        key = _row_key(row)
        if key in visiting:
            raise AuditError(f"dependency cycle encountered at {row['name']}")
        if key in active:
            return
        visiting.add(key)
        for dependency in row.get("dependencies", []):
            if not _marker_matches(dependency.get("marker"), {"sys_platform": "linux", "platform_machine": "x86_64", "implementation_name": "cpython"}):
                continue
            dependency_name = identity(dependency["name"], "").split("==", 1)[0]
            candidates = [candidate for candidate in by_name.get(dependency_name, []) if resolved[_row_key(candidate)]]
            if "version" in dependency:
                if not isinstance(dependency["version"], str):
                    raise AuditError(f"dependency version is malformed: {row['name']} -> {dependency_name}")
                candidates = [candidate for candidate in candidates if candidate["version"] == dependency["version"]]
            if "source" in dependency:
                if not isinstance(dependency["source"], dict):
                    raise AuditError(f"dependency source is malformed: {row['name']} -> {dependency_name}")
                source = canonical_json(dependency["source"])
                candidates = [candidate for candidate in candidates if canonical_json(candidate["source"]) == source]
            if len(candidates) != 1:
                raise AuditError(f"missing or ambiguous lock dependency: {row['name']} -> {dependency_name}")
            visit(candidates[0])
        visiting.remove(key)
        active.add(key)

    visit(roots[0])
    inactive_reasons: dict[tuple[str, str], str] = {}
    for row in rows:
        key = _row_key(row)
        if row.get("source") == {"virtual": "."}:
            inactive_reasons[key] = "virtual project row; no installed distribution is expected"
        elif key not in active and not resolved[key]:
            inactive_reasons[key] = "package resolution-marker is false for Linux x86_64"
        elif key not in active:
            inactive_reasons[key] = "not reachable from the virtual project dependency graph for Linux x86_64"
    return active, inactive_reasons


def classify_lock_rows(lock: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    active_keys, inactive_reasons = _active_lock_graph(lock)
    active: list[dict[str, Any]] = []
    inactive: list[dict[str, Any]] = []
    for row in sorted(lock["package"], key=lambda item: (_row_key(item), canonical_json(item))):
        item = {"name": row["name"], "version": row["version"], "source": row["source"]}
        key = _row_key(row)
        if row.get("source") == {"virtual": "."}:
            inactive.append({
                **item,
                "identity": identity(row["name"], row["version"]),
                "status": "INACTIVE_VIRTUAL_PROJECT",
                "reason": "virtual project row; no installed distribution is expected",
            })
        elif key not in active_keys:
            reason = inactive_reasons[key]
            status = "INACTIVE_RESOLUTION_MARKER" if "resolution-marker" in reason else "INACTIVE_UNREACHABLE_DEPENDENCY"
            inactive.append({
                **item,
                "identity": identity(row["name"], row["version"]),
                "status": status,
                "reason": reason,
            })
        else:
            active.append({
                **item,
                "identity": identity(row["name"], row["version"]),
                "status": "ACTIVE_LINUX_INSTALLED",
                "reason": "registry row is active under the locked Linux x86_64 environment",
            })
    return active, inactive


def _is_license_path(relative: str) -> bool:
    basename = Path(relative).name.casefold()
    if basename in LICENSE_FILE_NAMES:
        return True
    return any(
        basename.startswith(f"{marker}{separator}")
        for marker in LICENSE_FILE_NAMES
        for separator in (".", "-", "_")
    )


def _entry_path(dist: metadata.Distribution, entry: Any) -> Path | None:
    path = Path(dist.locate_file(entry))
    root = Path(dist.locate_file(""))
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        return None
    return path


def _publisher_files(dist: metadata.Distribution) -> tuple[list[dict[str, Any]], list[str]]:
    result: list[dict[str, Any]] = []
    unsafe: list[str] = []
    for entry in sorted(dist.files or [], key=str):
        relative = str(entry)
        if not _is_license_path(relative):
            continue
        path = _entry_path(dist, entry)
        if path is None or path.is_symlink() or not path.is_file():
            unsafe.append(relative)
            continue
        result.append({"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)})
    return result, unsafe


def _first_four(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read(4)


def _elf_needed(path: Path) -> dict[str, Any]:
    if _first_four(path) != ELF_MAGIC:
        return {"format": "non-elf", "needed": [], "inspection": "not-applicable"}
    try:
        completed = subprocess.run(
            ["readelf", "-d", str(path)], check=False, capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"format": "elf", "needed": [], "inspection": "error", "error": str(exc)}
    needed = sorted(
        match.group(1)
        for line in completed.stdout.splitlines()
        if (match := re.search(r"\(NEEDED\).*\[([^]]+)\]", line))
    )
    result = {
        "format": "elf",
        "needed": needed,
        "inspection": "ok" if completed.returncode == 0 else "error",
        "readelf_returncode": completed.returncode,
    }
    if completed.returncode != 0:
        result["error"] = (completed.stdout + completed.stderr)[-2000:]
    return result


def _native_files(dist: metadata.Distribution) -> tuple[list[dict[str, Any]], list[str]]:
    result: list[dict[str, Any]] = []
    unsafe: list[str] = []
    for entry in sorted(dist.files or [], key=str):
        relative = str(entry)
        basename = Path(relative).name.casefold()
        native_name = (
            Path(basename).suffix in NATIVE_SUFFIXES
            or ".so." in basename
            or basename.endswith(".dll")
        )
        path = _entry_path(dist, entry)
        if path is None or path.is_symlink() or not path.is_file():
            if native_name:
                unsafe.append(f"{relative}:not-regular")
            continue
        try:
            magic = _first_four(path)
            if not native_name and magic != ELF_MAGIC:
                continue
            inspection = _elf_needed(path)
            result.append({
                "distribution_shipped": True,
                "bundled": True,
                "origin": "installed-distribution",
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "candidate": "native-suffix" if native_name else "elf-magic",
                "needed": inspection,
            })
        except OSError as exc:
            unsafe.append(f"{relative}:read-failed:{exc}")
    return result, unsafe


def _validate_sdist_url(artifact: dict[str, Any], url: str, *, initial: bool = False) -> None:
    expected = artifact.get("url")
    if not isinstance(expected, str) or not isinstance(url, str):
        raise AuditError("locked sdist URL is missing")
    try:
        parsed = urlsplit(url)
        expected_parts = urlsplit(expected)
        port = parsed.port
        expected_port = expected_parts.port
    except ValueError as exc:
        raise AuditError("locked sdist URL has an invalid port") from exc
    for candidate, candidate_port in ((parsed, port), (expected_parts, expected_port)):
        if (
            candidate.scheme != "https"
            or candidate.hostname != PYPI_FILE_HOST
            or candidate_port not in (None, 443)
            or candidate.username is not None
            or candidate.password is not None
            or candidate.query
            or candidate.fragment
            or not candidate.path
        ):
            raise AuditError("locked sdist URL is not the exact official PyPI path")
    if initial and url != expected:
        raise AuditError("initial locked sdist URL differs from the lock")
    if parsed.path != expected_parts.path:
        raise AuditError("locked sdist redirect changed the exact artifact path")


class _SdistRedirects(HTTPRedirectHandler):
    def __init__(self, artifact: dict[str, Any], trace: list[str]) -> None:
        super().__init__()
        self.artifact = artifact
        self.trace = trace

    def redirect_request(self, request: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Request | None:
        if len(self.trace) - 1 >= MAX_SDIST_REDIRECTS:
            raise AuditError("locked sdist redirect chain exceeds the bounded limit")
        resolved = urljoin(request.full_url, newurl)
        _validate_sdist_url(self.artifact, resolved)
        self.trace.append(resolved)
        return super().redirect_request(request, fp, code, msg, headers, resolved)


def _archive_member_path(name: str) -> str:
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise AuditError("sdist archive contains an invalid member path")
    raw_parts = name[:-1].split("/") if name.endswith("/") else name.split("/")
    if not raw_parts or any(part in {"", ".", ".."} for part in raw_parts):
        raise AuditError("sdist archive contains an unsafe member path")
    if name.startswith("/") or re.match(r"^[A-Za-z]:/", name):
        raise AuditError("sdist archive contains an absolute member path")
    normalized = posixpath.normpath(name)
    if normalized in ("", ".") or normalized.startswith("../"):
        raise AuditError("sdist archive contains an unsafe member path")
    return normalized


def _archive_format(url: str) -> str:
    path = urlsplit(url).path.casefold()
    for suffix, value in (
        (".tar.gz", "tar.gz"), (".tgz", "tar.gz"), (".tar.bz2", "tar.bz2"),
        (".tbz2", "tar.bz2"), (".tar.xz", "tar.xz"), (".txz", "tar.xz"), (".zip", "zip"),
    ):
        if path.endswith(suffix):
            return value
    raise AuditError("unsupported locked sdist archive format")


def _archive_license_files(body: bytes, archive_format: str, archive_identity: dict[str, Any]) -> list[dict[str, Any]]:
    if len(body) > MAX_SDIST_BYTES:
        raise AuditError("locked sdist archive exceeds the bounded size")
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    total = 0
    total_uncompressed = 0

    def add(path: str, payload: bytes, declared: int) -> None:
        nonlocal total
        if len(payload) != declared:
            raise AuditError(f"license member size differs from its declaration: {path}")
        if not _is_license_path(path):
            return
        if len(payload) > MAX_LICENSE_BYTES:
            raise AuditError(f"license member is oversized: {path}")
        total += len(payload)
        if total > MAX_SDIST_LICENSE_TOTAL_BYTES:
            raise AuditError("license members exceed the bounded aggregate")
        candidates.append({
            "path": path,
            "size": len(payload),
            "sha256": sha256_bytes(payload),
            "content_base64": base64.b64encode(payload).decode("ascii"),
            "archive_identity": archive_identity,
        })

    if archive_format == "zip":
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            members = archive.infolist()
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise AuditError("sdist archive contains too many members")
            for info in members:
                path = _archive_member_path(info.filename)
                if path in seen:
                    raise AuditError(f"sdist archive contains duplicate member path: {path}")
                seen.add(path)
                file_type = (info.external_attr >> 16) & 0o170000
                name_is_dir = info.filename.endswith("/")
                if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise AuditError(f"sdist archive contains a special member: {path}")
                if (file_type == stat.S_IFREG and name_is_dir) or (file_type == stat.S_IFDIR and not name_is_dir):
                    raise AuditError(f"ZIP member type contradicts its name: {path}")
                if info.file_size < 0 or info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                    raise AuditError(f"sdist archive member is oversized: {path}")
                total_uncompressed += info.file_size
                if total_uncompressed > MAX_ARCHIVE_TOTAL_BYTES:
                    raise AuditError("sdist archive aggregate is oversized")
                if name_is_dir or file_type == stat.S_IFDIR:
                    continue
                if _is_license_path(path):
                    if info.file_size > MAX_LICENSE_BYTES:
                        raise AuditError(f"license member is oversized: {path}")
                    with archive.open(info, "r") as stream:
                        payload = stream.read(MAX_LICENSE_BYTES + 1)
                    add(path, payload, info.file_size)
    else:
        mode = {"tar.gz": "r:gz", "tar.bz2": "r:bz2", "tar.xz": "r:xz"}[archive_format]
        with tarfile.open(fileobj=io.BytesIO(body), mode=mode) as archive:
            for number, member in enumerate(archive, start=1):
                if number > MAX_ARCHIVE_MEMBERS:
                    raise AuditError("sdist archive contains too many members")
                path = _archive_member_path(member.name)
                if path in seen:
                    raise AuditError(f"sdist archive contains duplicate member path: {path}")
                seen.add(path)
                if member.issym() or member.islnk() or not (member.isdir() or member.isreg()):
                    raise AuditError(f"sdist archive contains a non-file/non-directory member: {path}")
                if member.size < 0 or member.size > MAX_ARCHIVE_MEMBER_BYTES:
                    raise AuditError(f"sdist archive member is oversized: {path}")
                total_uncompressed += member.size
                if total_uncompressed > MAX_ARCHIVE_TOTAL_BYTES:
                    raise AuditError("sdist archive aggregate is oversized")
                if member.isdir() or not _is_license_path(path):
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    raise AuditError(f"license member is unreadable: {path}")
                with stream:
                    payload = stream.read(MAX_LICENSE_BYTES + 1)
                add(path, payload, member.size)
    if not candidates:
        raise AuditError("locked sdist contains no bounded license candidate")
    return sorted(candidates, key=lambda item: item["path"])


def _fetch_locked_sdist(row: dict[str, Any], fetcher: Callable[[str], tuple[str, bytes]] | None = None) -> dict[str, Any]:
    artifact = row.get("sdist")
    if not isinstance(artifact, dict) or set(artifact) != {"url", "hash", "size", "upload-time"}:
        raise AuditError(f"locked sdist artifact schema is not exact for {identity(row['name'], row['version'])}")
    expected_hash = artifact["hash"]
    expected_size = artifact["size"]
    if (
        not isinstance(expected_hash, str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_hash)
        or isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size <= 0
        or expected_size > MAX_SDIST_BYTES
        or not isinstance(artifact["upload-time"], str)
        or not artifact["upload-time"].strip()
    ):
        raise AuditError("locked sdist artifact values are malformed")
    _validate_sdist_url(artifact, artifact["url"], initial=True)
    trace = [artifact["url"]]
    if fetcher is None:
        opener = build_opener(_SdistRedirects(artifact, trace))
        request = Request(artifact["url"], headers={"Accept": "application/octet-stream", "User-Agent": "vokra-neutts-air-audit/1"})
        try:
            with opener.open(request, timeout=30) as response:  # noqa: S310 - exact host/path is checked
                final_url = response.geturl()
                _validate_sdist_url(artifact, final_url)
                if final_url not in trace:
                    trace.append(final_url)
                length = response.headers.get("Content-Length")
                if length and length.isdigit() and int(length) > MAX_SDIST_BYTES:
                    raise AuditError("locked sdist Content-Length exceeds the bounded size")
                body = response.read(MAX_SDIST_BYTES + 1)
        except AuditError:
            raise
        except (OSError, UnicodeError) as exc:
            raise AuditError(f"locked sdist fetch failed: {exc}") from exc
    else:
        try:
            final_url, body = fetcher(artifact["url"])
        except Exception as exc:  # noqa: BLE001 - injected transport is bounded by the caller
            raise AuditError(f"locked sdist fetch failed: {exc}") from exc
        _validate_sdist_url(artifact, final_url)
        if final_url != trace[-1]:
            trace.append(final_url)
    if not isinstance(body, bytes) or len(body) > MAX_SDIST_BYTES:
        raise AuditError("locked sdist response is not bounded bytes")
    actual_hash = "sha256:" + sha256_bytes(body)
    if len(body) != expected_size:
        raise AuditError("locked sdist response size does not match the lock")
    if actual_hash != expected_hash:
        raise AuditError("locked sdist response hash does not match the lock")
    archive_format = _archive_format(artifact["url"])
    archive_identity = {
        "requested_url": artifact["url"], "final_url": final_url, "url_trace": trace,
        "size": len(body), "sha256": actual_hash, "upload-time": artifact["upload-time"], "format": archive_format,
    }
    return {
        "status": "PASS",
        "archive_identity": archive_identity,
        "publisher_files": _archive_license_files(body, archive_format, archive_identity),
    }


def _metadata_fields(dist: metadata.Distribution) -> dict[str, Any]:
    classifiers = sorted(
        value.removeprefix("License :: ")
        for value in (dist.metadata.get_all("Classifier") or [])
        if value.startswith("License :: ")
    )
    declared = dist.metadata.get("License")
    expression = dist.metadata.get("License-Expression")
    return {
        "license": declared.strip() if isinstance(declared, str) and declared.strip() else None,
        "license_expression": expression.strip() if isinstance(expression, str) and expression.strip() else None,
        "license_classifiers": classifiers,
    }


def _distribution_records() -> list[dict[str, Any]]:
    records = []
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name and dist.version:
            records.append({
                "distribution": dist,
                "identity": identity(name, dist.version),
                "name": name,
                "version": dist.version,
                "location": str(Path(dist.locate_file(""))),
            })
    return sorted(records, key=lambda item: (item["identity"], item["location"]))


def compare_multiset(expected: list[str], actual: list[str]) -> dict[str, Any]:
    expected_counts = Counter(expected)
    actual_counts = Counter(actual)
    missing = sorted((expected_counts - actual_counts).elements())
    unexpected = sorted((actual_counts - expected_counts).elements())
    return {
        "expected": sorted(expected), "installed": sorted(actual), "missing": missing,
        "unexpected": unexpected,
        "duplicate_identities": sorted(key for key, count in actual_counts.items() if count > 1),
        "exact": not missing and not unexpected,
    }


def _inspect_package(row: dict[str, Any], record: dict[str, Any] | None, duplicate: bool, sdist_fetcher: Callable[[str], tuple[str, bytes]] | None) -> tuple[dict[str, Any], list[str]]:
    lock_data = {
        "name": row["name"], "version": row["version"], "source": row["source"],
        "artifacts": {"sdist": row.get("sdist"), "wheels": row.get("wheels", [])},
    }
    if record is None:
        return {"lock": lock_data, "installed": None}, [f"installed closure missing: {identity(row['name'], row['version'])}"]
    dist = record["distribution"]
    publisher, unsafe_publisher = _publisher_files(dist)
    native, unsafe_native = _native_files(dist)
    locked_sdist = None
    if not publisher:
        try:
            locked_sdist = _fetch_locked_sdist(row, sdist_fetcher)
        except AuditError as exc:
            locked_sdist = {
                "status": "BLOCKED",
                "archive_identity": {"requested_url": row.get("sdist", {}).get("url") if isinstance(row.get("sdist"), dict) else None},
                "publisher_files": [], "error": str(exc),
            }
    metadata_data = _metadata_fields(dist)
    installed = {
        "name": dist.metadata.get("Name"), "version": dist.version,
        "normalized_identity": record["identity"], "location": record["location"],
        **metadata_data, "publisher_files": publisher,
        "locked_sdist_license_audit": locked_sdist, "native_files": native,
        "bundled_libraries": [
            {"distribution": dist.metadata.get("Name"), "path": item["path"], "size": item["size"], "sha256": item["sha256"], "needed": item["needed"]}
            for item in native
        ],
    }
    failures: list[str] = []
    sdist_valid = bool(locked_sdist and locked_sdist.get("status") == "PASS" and locked_sdist.get("publisher_files"))
    metadata_valid = bool(metadata_data["license"] or metadata_data["license_expression"] or metadata_data["license_classifiers"] or sdist_valid)
    if not metadata_valid:
        failures.append(f"missing package license metadata: {record['identity']}")
    if not publisher and not sdist_valid:
        failures.append(f"missing publisher LICENSE/NOTICE evidence: {record['identity']}")
    if locked_sdist and locked_sdist.get("status") == "BLOCKED":
        failures.append(f"locked sdist publisher evidence blocked: {record['identity']}: {locked_sdist['error']}")
    failures.extend(f"unsafe publisher path: {record['identity']}:{item}" for item in unsafe_publisher)
    failures.extend(f"unsafe native path: {record['identity']}:{item}" for item in unsafe_native)
    if any(item["needed"].get("inspection") == "error" for item in native):
        failures.append(f"ELF NEEDED inspection failed: {record['identity']}")
    if duplicate:
        failures.append(f"duplicate installed distribution: {record['identity']}")
    return {"lock": lock_data, "installed": installed}, failures


def _fixed_license_items(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    source = manifest["source_identity"]
    source_repo = source["repo"].removeprefix("https://github.com/").removesuffix(".git")
    return [
        {"id": f"public-gguf:{manifest['public_identity']['repo']}@{manifest['public_identity']['revision']}", "kind": "huggingface", "repo": manifest["public_identity"]["repo"], "revision": manifest["public_identity"]["revision"], "claimed_license": manifest["public_identity"].get("license")},
        {"id": f"companion-gguf:{manifest['companion_identity']['repo']}@{manifest['companion_identity']['revision']}", "kind": "huggingface", "repo": manifest["companion_identity"]["repo"], "revision": manifest["companion_identity"]["revision"], "claimed_license": manifest["companion_identity"].get("license")},
        {"id": f"gated-upstream:{manifest['upstream_identity']['repo']}@{manifest['upstream_identity']['revision']}", "kind": "huggingface", "repo": manifest["upstream_identity"]["repo"], "revision": manifest["upstream_identity"]["revision"], "claimed_license": manifest["upstream_identity"].get("license")},
        {"id": f"official-source:{source['repo']}@{source['revision']}#{source['path']}", "kind": "github", "repo": source_repo, "revision": source["revision"], "claimed_license": source.get("license")},
    ]


def _license_url(item: dict[str, Any]) -> str:
    if item["kind"] == "github":
        return f"https://raw.githubusercontent.com/{item['repo']}/{item['revision']}/LICENSE"
    return f"https://huggingface.co/{item['repo']}/raw/{item['revision']}/LICENSE"


def _validate_license_url(item: dict[str, Any], url: str, *, initial: bool = False) -> None:
    expected = _license_url(item)
    parsed = urlsplit(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise AuditError("LICENSE URL has an invalid port") from exc
    if parsed.scheme != "https" or parsed.username or parsed.password or port not in (None, 443) or parsed.query or parsed.fragment:
        raise AuditError("LICENSE URL has unsafe URL components")
    if initial and url != expected:
        raise AuditError("initial LICENSE URL differs from the fixed identity")
    if url == expected:
        return
    if item["kind"] == "huggingface" and parsed.hostname in HF_LICENSE_HOSTS and parsed.path == f"/{item['repo']}/resolve/{item['revision']}/LICENSE":
        return
    raise AuditError("LICENSE URL is not the exact fixed path")


class _LicenseRedirects(HTTPRedirectHandler):
    def __init__(self, item: dict[str, Any], trace: list[str]) -> None:
        super().__init__()
        self.item = item
        self.trace = trace

    def redirect_request(self, request: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Request | None:
        if len(self.trace) - 1 >= MAX_SDIST_REDIRECTS:
            raise AuditError("LICENSE redirect chain exceeds the bounded limit")
        resolved = urljoin(request.full_url, newurl)
        _validate_license_url(self.item, resolved)
        self.trace.append(resolved)
        return super().redirect_request(request, fp, code, msg, headers, resolved)


def _fetch_license(item: dict[str, Any], fetcher: Callable[[str], tuple[str, bytes]] | None = None) -> dict[str, Any]:
    requested_url = _license_url(item)
    _validate_license_url(item, requested_url, initial=True)
    trace = [requested_url]
    if fetcher is None:
        opener = build_opener(_LicenseRedirects(item, trace))
        request = Request(requested_url, headers={"Accept": "text/plain", "User-Agent": "vokra-neutts-air-audit/1"})
        try:
            with opener.open(request, timeout=30) as response:  # noqa: S310 - allowlist checked
                final_url = response.geturl()
                _validate_license_url(item, final_url)
                if final_url not in trace:
                    trace.append(final_url)
                body = response.read(MAX_LICENSE_BYTES + 1)
        except AuditError:
            raise
        except (OSError, UnicodeError) as exc:
            raise AuditError(f"fixed LICENSE fetch failed: {exc}") from exc
    else:
        try:
            final_url, body = fetcher(requested_url)
        except Exception as exc:  # noqa: BLE001 - injected transport is bounded
            raise AuditError(f"fixed LICENSE fetch failed: {exc}") from exc
        _validate_license_url(item, final_url)
        if final_url != trace[-1]:
            trace.append(final_url)
    if not isinstance(body, bytes) or len(body) > MAX_LICENSE_BYTES:
        raise AuditError("fixed LICENSE response exceeds the bounded size")
    return {
        "id": item["id"], "kind": item["kind"], "repo": item["repo"], "revision": item["revision"],
        "claimed_license": item["claimed_license"], "requested_url": requested_url, "final_url": final_url,
        "url_trace": trace, "acquired_file": "LICENSE", "size": len(body), "sha256": sha256_bytes(body),
        "content_base64": base64.b64encode(body).decode("ascii"),
        "license_classification": "UNCLASSIFIED_PRIMARY_SOURCE_BYTES_ONLY",
    }


def audit_model_licenses(manifest: dict[str, Any], fetcher: Callable[[str], tuple[str, bytes]] | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for item in _fixed_license_items(manifest):
        try:
            records.append(_fetch_license(item, fetcher))
        except AuditError as exc:
            records.append({
                "id": item["id"], "kind": item["kind"], "repo": item["repo"], "revision": item["revision"],
                "claimed_license": item["claimed_license"], "requested_url": _license_url(item), "final_url": None,
                "acquired_file": None, "status": "BLOCKED_FACTUAL_LICENSE_PATH", "error": str(exc),
            })
            failures.append(f"BLOCKED_FACTUAL_LICENSE_PATH: {item['id']}: {exc}")
    return records, failures


def _contract(project: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bytes, bytes]:
    project_path, lock_path, manifest_path = project / "pyproject.toml", project / "uv.lock", project / "license_gate_manifest.json"
    if any(path.is_symlink() or not path.is_file() for path in (project_path, lock_path, manifest_path)):
        raise AuditError("NeuTTS Air project, lock, or manifest is missing/symlinked")
    project_bytes, lock_bytes = project_path.read_bytes(), lock_path.read_bytes()
    project_data = tomllib.loads(project_bytes.decode("utf-8"))
    lock_data = tomllib.loads(lock_bytes.decode("utf-8"))
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("gate_version") != preflight_gate.GATE_VERSION:
        raise AuditError("NeuTTS Air manifest version is unsupported")
    if sha256_bytes(project_bytes) != preflight_gate.PYPROJECT_SHA256 or sha256_bytes(lock_bytes) != preflight_gate.LOCK_SHA256:
        raise AuditError("project or lock bytes differ from the reviewed contract")
    preflight_gate.validate_project_schema(project_data)
    rows = preflight_gate.canonical_package_rows(lock_data)
    if preflight_gate.canonical_digest(rows) != manifest.get("package_rows_sha256"):
        raise AuditError("canonical lock rows differ from the manifest")
    if manifest.get("public_identity") != preflight_gate.PUBLIC_IDENTITY or manifest.get("companion_identity") != preflight_gate.COMPANION_IDENTITY or manifest.get("upstream_identity") != preflight_gate.UPSTREAM_IDENTITY or manifest.get("source_identity") != preflight_gate.SOURCE_IDENTITY:
        raise AuditError("fixed model/source identity drifted from the manifest")
    return project_data, lock_data, manifest, project_bytes, lock_bytes


def _repository_identity(project: Path) -> dict[str, Any]:
    repository = project.parents[2]
    try:
        commit = subprocess.run(["git", "-C", str(repository), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AuditError(f"git commit identity unavailable: {exc}") from exc
    return {"root": str(repository), "commit": commit, "audit_script_sha256": sha256_file(Path(__file__).resolve())}


def _dependency_acquisition(packages: list[dict[str, Any]]) -> dict[str, Any]:
    requests = []
    not_needed = []
    for package in packages:
        installed = package.get("installed")
        lock = package.get("lock") if isinstance(package.get("lock"), dict) else {}
        audit = installed.get("locked_sdist_license_audit") if isinstance(installed, dict) else None
        if not isinstance(audit, dict):
            continue
        archive = audit.get("archive_identity") if isinstance(audit.get("archive_identity"), dict) else {}
        requests.append({"package": lock.get("name"), "version": lock.get("version"), "url": archive.get("requested_url"), "status": audit.get("status")})
    return {
        "policy": "exact locked PyPI sdist only when installed publisher evidence is missing",
        "scope": "active installed rows lacking publisher LICENSE/NOTICE files",
        "attempted_requests": sorted(requests, key=lambda item: (item["package"] or "", item["version"] or "")),
        "not_needed": sorted(not_needed, key=lambda item: (item["package"] or "", item["version"] or "")),
        "out_of_scope_requests": [], "model_files": [], "in_memory_archive_inspection": True,
    }


def _approval_state(manifest: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    expected_review_ids = sorted(f"{row['name']}@{row['version']}" for row in lock["package"])
    review_rows = manifest.get("review_rows")
    if not isinstance(review_rows, list):
        raise AuditError("dependency review rows are missing")
    review_ids = [row.get("id") for row in review_rows if isinstance(row, dict)]
    if review_ids != expected_review_ids or len(set(review_ids)) != len(expected_review_ids):
        raise AuditError("dependency review rows do not provide exact one-to-one lock coverage")
    if any(not isinstance(row, dict) or set(row) != {"id", "status", "license", "native_review", "bundled_review", "evidence"} for row in review_rows):
        raise AuditError("dependency review row schema is malformed")
    model_rows = manifest.get("model_reviews")
    if not isinstance(model_rows, list) or [row.get("id") for row in model_rows if isinstance(row, dict)] != preflight_gate.MODEL_REVIEW_IDS or len(set(row.get("id") for row in model_rows if isinstance(row, dict))) != len(preflight_gate.MODEL_REVIEW_IDS):
        raise AuditError("model/source review rows do not provide exact one-to-one coverage")
    if any(not isinstance(row, dict) or set(row) != {"id", "status", "license", "native_review", "bundled_review", "evidence"} for row in model_rows):
        raise AuditError("model/source review row schema is malformed")
    operator = manifest.get("operator_approval")
    if not isinstance(operator, dict) or set(operator) != {"schema", "decision", "signer", "digest"}:
        raise AuditError("operator approval schema is malformed")
    publication = manifest.get("publication")
    if publication != "NO_UPLOAD":
        raise AuditError("publication policy drifted from NO_UPLOAD")
    dependency_pending = [row["id"] for row in review_rows if row["status"] != "REVIEWED"]
    model_pending = [row["id"] for row in model_rows if row["status"] != "REVIEWED"]
    blockers = []
    if dependency_pending:
        blockers.append({"kind": "dependency_reviews", "pending_ids": dependency_pending})
    if model_pending:
        blockers.append({"kind": "model_source_reviews", "pending_ids": model_pending})
    if operator.get("decision") != "APPROVED":
        blockers.append({"kind": "operator_approval", "decision": operator.get("decision")})
    return {
        "dependency_reviews": {"count": len(review_rows), "rows": review_rows},
        "model_reviews": {"count": len(model_rows), "rows": model_rows},
        "operator_approval": operator,
        "publication": publication,
        "approval_blockers": blockers,
        "publication_permitted": False,
    }


def audit_environment(project: Path, fetch_model_licenses: bool = True, sdist_fetcher: Callable[[str], tuple[str, bytes]] | None = None) -> dict[str, Any]:
    project_data, lock, manifest, project_bytes, lock_bytes = _contract(project)
    approval_state = _approval_state(manifest, lock)
    active_rows, inactive_rows = classify_lock_rows(lock)
    registry_rows = [row for row in lock["package"] if row.get("source") != {"virtual": "."}]
    active_registry_keys = {(item["name"], item["version"]) for item in active_rows}
    expected_rows = [row for row in registry_rows if (row["name"], row["version"]) in active_registry_keys]
    records = _distribution_records()
    expected_ids = [identity(row["name"], row["version"]) for row in expected_rows]
    actual_ids = [record["identity"] for record in records]
    closure = compare_multiset(expected_ids, actual_ids)
    by_identity: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_identity.setdefault(record["identity"], []).append(record)
    packages: list[dict[str, Any]] = []
    failures: list[str] = []
    for row in expected_rows:
        key = identity(row["name"], row["version"])
        candidates = by_identity.get(key, [])
        package, package_failures = _inspect_package(row, candidates[0] if len(candidates) == 1 else None, len(candidates) > 1, sdist_fetcher)
        packages.append(package)
        failures.extend(package_failures)
    if not closure["exact"]:
        failures.append("installed normalized name+version multiset does not exactly match uv.lock")
    if sys.version_info[:2] != (3, 12):
        failures.append(f"Python runtime is not 3.12: {platform.python_version()}")
    if sys.platform != "linux" or platform.machine().casefold() not in {"x86_64", "amd64"}:
        failures.append(f"audit host is not Linux x86_64: {sys.platform}/{platform.machine()}")
    model_license_files, model_failures = audit_model_licenses(manifest) if fetch_model_licenses else ([], [])
    failures.extend(model_failures)
    all_lock_rows = []
    package_by_key = {_row_key(item["lock"]): item for item in packages}
    inactive_by_identity = {item["identity"]: item for item in inactive_rows}
    for row in sorted(lock["package"], key=lambda item: (_row_key(item), canonical_json(item))):
        key = _row_key(row)
        if key in package_by_key:
            all_lock_rows.append(package_by_key[key])
        else:
            inactive = inactive_by_identity[identity(row["name"], row["version"])]
            all_lock_rows.append({
                "lock": {"name": row["name"], "version": row["version"], "source": row["source"], "artifacts": {"sdist": row.get("sdist"), "wheels": row.get("wheels", [])}},
                "audit_status": inactive["status"], "inactive_reason": inactive["reason"], "installed": None,
            })
    review_by_id = {row["id"]: row for row in approval_state["dependency_reviews"]["rows"]}
    for package in all_lock_rows:
        lock_item = package["lock"]
        package["review"] = review_by_id[f"{lock_item['name']}@{lock_item['version']}"]
    return {
        "schema": SCHEMA, "status": "BLOCKED" if failures else "PASS", "repository": _repository_identity(project),
        "environment": {"python": platform.python_version(), "platform": sys.platform, "machine": platform.machine(), "readelf_required": True, "model_code_imported": False, "cargo_invoked": False},
        "project": {"name": project_data["project"]["name"], "version": project_data["project"]["version"], "pyproject_bytes": len(project_bytes), "pyproject_sha256": sha256_bytes(project_bytes), "uv_lock_bytes": len(lock_bytes), "uv_lock_sha256": sha256_bytes(lock_bytes)},
        "lock_rows": {"accounted_rows": len(all_lock_rows), "active_linux_installed": active_rows, "inactive_or_virtual": inactive_rows, "all_rows_accounted": len(all_lock_rows) == len(lock["package"])},
        "closure": closure, "packages": all_lock_rows, "dependency_acquisition": _dependency_acquisition(all_lock_rows),
        "approval_state": approval_state,
        "fixed_source_model_companion_identities": _fixed_license_items(manifest), "model_license_files": model_license_files,
        "model_acquisition": {"scope": "fixed source/model/companion LICENSE paths only", "policy": "allow-listed exact primary-source LICENSE-only fetch", "requested_files": [item["requested_url"] for item in model_license_files], "non_license_requests": [], "non_license_files": [], "model_files": []},
        "failures": sorted(set(failures)),
    }


def _minimal_blocked_report(project: Path, exc: Exception) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "BLOCKED",
        "project": {"name": project.name, "version": None, "pyproject_sha256": None, "uv_lock_sha256": None},
        "environment": {"model_code_imported": False, "cargo_invoked": False},
        "lock_rows": {"accounted_rows": 0, "active_linux_installed": [], "inactive_or_virtual": [], "all_rows_accounted": False},
        "closure": {"expected": [], "installed": [], "missing": [], "unexpected": []},
        "packages": [],
        "dependency_acquisition": {"policy": "exact locked PyPI sdist only when installed publisher evidence is missing", "scope": "active installed rows lacking publisher LICENSE/NOTICE files", "attempted_requests": [], "not_needed": [], "out_of_scope_requests": [], "model_files": []},
        "approval_state": {"dependency_reviews": {"count": 0, "rows": []}, "model_reviews": {"count": 0, "rows": []}, "operator_approval": None, "publication": None, "approval_blockers": [{"kind": "environment_audit", "error": str(exc)}], "publication_permitted": False},
        "model_license_files": [],
        "model_acquisition": {"requested_files": [], "non_license_requests": [], "non_license_files": [], "model_files": []},
        "failures": [f"ENVIRONMENT_AUDIT_BLOCKED: {type(exc).__name__}: {exc}"],
    }


def run(project: Path, output: Path, fetch_model_licenses: bool) -> int:
    try:
        report = audit_environment(project, fetch_model_licenses)
    except Exception as exc:  # noqa: BLE001 - blocked reports are mandatory
        report = _minimal_blocked_report(project, exc)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"neutts-air dependency audit: BLOCKED: report write failed: {exc}", file=sys.stderr)
        return 2
    if report["status"] == "BLOCKED":
        print("neutts-air dependency audit: BLOCKED: " + "; ".join(report["failures"]), file=sys.stderr)
        return 2
    print(f"neutts-air dependency audit: PASS ({output})")
    return 0


def self_test() -> int:
    project = Path(__file__).resolve().parent
    lock = tomllib.loads((project / "uv.lock").read_text(encoding="utf-8"))
    lock_rows = preflight_gate.canonical_package_rows(lock)
    custom_torch_rows = [
        row for row in lock_rows
        if row["name"] == "torch" and row["source"] == {"registry": preflight_gate.PYTORCH_CPU_REGISTRY}
    ]
    custom_wheels = {
        (wheel["url"], wheel["hash"])
        for row in custom_torch_rows
        for wheel in row["wheels"]
    }
    if (
        len(custom_torch_rows) != 2
        or custom_wheels != preflight_gate.PYTORCH_CPU_ARTIFACTS_WITHOUT_SIZE
        or any(set(wheel) != preflight_gate.ARTIFACT_KEYS - {"size"} for row in custom_torch_rows for wheel in row["wheels"])
    ):
        raise AssertionError("tracked custom-index torch wheel schema drifted")
    custom_url, custom_hash = next(iter(custom_wheels))
    custom_artifact = {"url": custom_url, "hash": custom_hash, "upload-time": "2026-01-01T00:00:00Z"}
    preflight_gate.validate_artifact(
        custom_artifact,
        "self-test custom-index torch wheel",
        preflight_gate.PYTORCH_CPU_REGISTRY,
        package_name="torch",
        artifact_kind="wheels",
    )
    for label, package_name, artifact_kind, registry, mutate in (
        ("wrong-package", "not-torch", "wheels", preflight_gate.PYTORCH_CPU_REGISTRY, lambda value: None),
        ("wrong-kind", "torch", "sdist", preflight_gate.PYTORCH_CPU_REGISTRY, lambda value: None),
        ("wrong-registry", "torch", "wheels", "https://pypi.org/simple", lambda value: None),
        ("wrong-hash", "torch", "wheels", preflight_gate.PYTORCH_CPU_REGISTRY, lambda value: value.update(hash="sha256:" + "0" * 64)),
        ("extra-key", "torch", "wheels", preflight_gate.PYTORCH_CPU_REGISTRY, lambda value: value.update(extra="reject")),
        ("bool-size", "torch", "wheels", preflight_gate.PYTORCH_CPU_REGISTRY, lambda value: value.update(size=True)),
        ("missing-upload-time", "torch", "wheels", preflight_gate.PYTORCH_CPU_REGISTRY, lambda value: value.pop("upload-time")),
    ):
        candidate = dict(custom_artifact)
        mutate(candidate)
        try:
            preflight_gate.validate_artifact(
                candidate,
                f"self-test custom-index {label}",
                registry,
                package_name=package_name,
                artifact_kind=artifact_kind,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted malformed custom-index wheel: {label}")
    active, inactive = classify_lock_rows(lock)
    assert len(lock["package"]) == len(active) + len(inactive)
    assert len(inactive) == 3
    assert any(item["status"] == "INACTIVE_VIRTUAL_PROJECT" for item in inactive)
    assert len(active) == 36
    colorama = next(item for item in inactive if item["name"] == "colorama")
    assert colorama["status"] == "INACTIVE_UNREACHABLE_DEPENDENCY"
    tampered = copy.deepcopy(lock)
    root = next(item for item in tampered["package"] if item["source"] == {"virtual": "."})
    root["dependencies"][0]["marker"] = "sys_platform === 'linux'"
    try:
        classify_lock_rows(tampered)
    except AuditError:
        pass
    else:
        raise AssertionError("unsupported dependency marker grammar was accepted")
    cycle = copy.deepcopy(lock)
    numpy_row = next(item for item in cycle["package"] if item["name"] == "numpy")
    numpy_row["dependencies"] = [{"name": root["name"]}]
    try:
        classify_lock_rows(cycle)
    except AuditError:
        pass
    else:
        raise AssertionError("reachable dependency cycle was accepted")
    assert compare_multiset(["foo-bar==1.0"], ["foo-bar==1.0", "foo-bar==1.0"])["unexpected"] == ["foo-bar==1.0"]
    assert _is_license_path("LICENSE") and _is_license_path("license.txt") and _is_license_path("NOTICE-extra")
    assert not _is_license_path("unlicensed-file") and not _is_license_path("project-license")
    body = b"license\n"
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo("demo-1/LICENSE")
        member.size = len(body)
        archive.addfile(member, io.BytesIO(body))
    artifact = {"url": "https://files.pythonhosted.org/packages/demo/demo-1.tar.gz", "hash": "sha256:" + sha256_bytes(archive_buffer.getvalue()), "size": len(archive_buffer.getvalue()), "upload-time": "2026-01-01T00:00:00Z"}
    row = {"name": "demo", "version": "1", "source": {"registry": "https://pypi.org/simple"}, "sdist": artifact}
    fetched = _fetch_locked_sdist(row, lambda url: (url, archive_buffer.getvalue()))
    assert fetched["status"] == "PASS" and fetched["publisher_files"][0]["path"] == "demo-1/LICENSE"
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("demo-1/LICENSE", body)
    zip_payload = zip_buffer.getvalue()
    zip_artifact = {**artifact, "url": artifact["url"].removesuffix(".tar.gz") + ".zip", "hash": "sha256:" + sha256_bytes(zip_payload), "size": len(zip_payload)}
    zip_row = {**row, "sdist": zip_artifact}
    assert _fetch_locked_sdist(zip_row, lambda url: (url, zip_payload))["publisher_files"][0]["path"] == "demo-1/LICENSE"
    def synthetic_tar(entries: list[tuple[str, bytes, str]]) -> bytes:
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w:gz") as archive:
            for name, payload, kind in entries:
                member = tarfile.TarInfo(name)
                if kind == "symlink":
                    member.type = tarfile.SYMTYPE
                    member.linkname = "target"
                    archive.addfile(member)
                elif kind == "fifo":
                    member.type = tarfile.FIFOTYPE
                    archive.addfile(member)
                elif kind == "dir":
                    member.type = tarfile.DIRTYPE
                    archive.addfile(member)
                else:
                    member.size = len(payload)
                    archive.addfile(member, io.BytesIO(payload))
        return output.getvalue()
    def synthetic_zip_special(name: str, file_type: int) -> bytes:
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = (file_type | 0o644) << 16
            archive.writestr(info, b"special")
        return output.getvalue()
    for bad_body in (
        synthetic_tar([("../LICENSE", b"bad", "file")]),
        synthetic_tar([("demo-1//LICENSE", b"bad", "file")]),
        synthetic_tar([("demo-1/./LICENSE", b"bad", "file")]),
        synthetic_tar([("demo-1/LICENSE", b"", "symlink")]),
        synthetic_tar([("demo-1/device", b"", "fifo")]),
        synthetic_tar([("demo-1/LICENSE", b"one", "file"), ("demo-1/LICENSE", b"two", "file")]),
    ):
        bad_row = {**row, "sdist": {**artifact, "size": len(bad_body), "hash": "sha256:" + sha256_bytes(bad_body)}}
        try:
            _fetch_locked_sdist(bad_row, lambda url, payload=bad_body: (url, payload))
        except AuditError:
            pass
        else:
            raise AssertionError("accepted unsafe sdist archive")
    for bad_zip in (
        synthetic_zip_special("demo-1/socket", stat.S_IFSOCK),
        synthetic_zip_special("demo-1/LICENSE/", stat.S_IFREG),
        synthetic_zip_special("demo-1/NOTICE", stat.S_IFDIR),
    ):
        bad_row = {**zip_row, "sdist": {**zip_artifact, "size": len(bad_zip), "hash": "sha256:" + sha256_bytes(bad_zip)}}
        try:
            _fetch_locked_sdist(bad_row, lambda url, payload=bad_zip: (url, payload))
        except AuditError:
            pass
        else:
            raise AssertionError("accepted unsafe ZIP archive")
    many = synthetic_tar([(f"demo-1/file-{index}", b"x", "file") for index in range(MAX_ARCHIVE_MEMBERS + 1)])
    many_row = {**row, "sdist": {**artifact, "size": len(many), "hash": "sha256:" + sha256_bytes(many)}}
    try:
        _fetch_locked_sdist(many_row, lambda url: (url, many))
    except AuditError:
        pass
    else:
        raise AssertionError("accepted archive with too many members")
    redirect_trace = [artifact["url"]]
    redirect_guard = _SdistRedirects(artifact, redirect_trace)
    redirect_guard.redirect_request(Request(artifact["url"]), None, 302, "found", {}, artifact["url"])
    assert len(redirect_trace) == 2
    try:
        redirect_guard.redirect_request(Request(artifact["url"]), None, 302, "found", {}, artifact["url"] + "?download=1")
    except AuditError:
        pass
    else:
        raise AssertionError("accepted unsafe sdist redirect")
    for mutation in ("missing", "extra", "bool-size", "empty-upload-time"):
        tampered = dict(artifact)
        if mutation == "missing":
            tampered.pop("upload-time")
        elif mutation == "extra":
            tampered["extra"] = "reject"
        elif mutation == "bool-size":
            tampered["size"] = True
        else:
            tampered["upload-time"] = " "
        try:
            _fetch_locked_sdist({**row, "sdist": tampered}, lambda url: (url, archive_buffer.getvalue()))
        except AuditError:
            pass
        else:
            raise AssertionError(f"accepted malformed artifact: {mutation}")
    for bad in (artifact["url"] + "?download=1", artifact["url"] + "#fragment", artifact["url"].replace("https://", "https://user@", 1), artifact["url"].replace("files.pythonhosted.org", "files.pythonhosted.org:8443"), artifact["url"].replace(".tar.gz", ".whl")):
        try:
            _fetch_locked_sdist({**row, "sdist": {**artifact, "url": bad}}, lambda url: (url, archive_buffer.getvalue()))
        except AuditError:
            pass
        else:
            raise AssertionError(f"accepted unsafe sdist URL: {bad}")
    model_manifest = load_json(project / "license_gate_manifest.json")
    approval_state = _approval_state(model_manifest, lock)
    assert approval_state["dependency_reviews"]["count"] == 39
    assert approval_state["model_reviews"]["count"] == 4
    assert approval_state["operator_approval"]["decision"] == "PENDING_REVIEW"
    assert approval_state["publication"] == "NO_UPLOAD"
    assert not approval_state["publication_permitted"]
    assert {item["kind"] for item in approval_state["approval_blockers"]} == {
        "dependency_reviews", "model_source_reviews", "operator_approval"
    }
    items = _fixed_license_items(model_manifest)
    assert len(items) == 4 and all(_license_url(item).endswith("/LICENSE") for item in items)
    license_result = _fetch_license(items[0], lambda url: (url, b"primary source"))
    assert license_result["license_classification"] == "UNCLASSIFIED_PRIMARY_SOURCE_BYTES_ONLY"
    for bad in (_license_url(items[0]) + "?x=1", _license_url(items[0]) + "#x", _license_url(items[0]).replace("https://", "https://user@", 1)):
        try:
            _fetch_license(items[0], lambda url, bad=bad: (bad, b"x"))
        except AuditError:
            pass
        else:
            raise AssertionError("accepted unsafe LICENSE redirect")
    with tempfile.TemporaryDirectory(prefix="neutts-air-audit-") as directory:
        output = Path(directory) / "blocked.json"
        assert run(Path(directory) / "missing", output, False) == 2
        assert json.loads(output.read_text(encoding="utf-8"))["status"] == "BLOCKED"
    print("neutts-air dependency audit: self-test PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fetch-model-licenses", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.project is None or args.output is None:
        parser.error("--project and --output are required")
    return run(args.project, args.output, args.fetch_model_licenses)


if __name__ == "__main__":
    raise SystemExit(main())
