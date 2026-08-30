#!/usr/bin/env python3
"""Model-free audit of the pinned Qwen3-ASR reference environment.

The audit is intentionally separate from :mod:`preflight_gate`: the gate
authenticates the inputs before synchronization, while this script inspects
the already synchronized, exact environment.  It never imports qwen-asr,
torch, or any model code.  It only inspects exact locked PyPI sdists for
missing publisher files and the two allow-listed model LICENSE URLs below.
"""

from __future__ import annotations

import argparse
import base64
import copy
from collections import Counter
import hashlib
import importlib.metadata as metadata
from io import BytesIO
import json
import platform
import re
from pathlib import PurePosixPath
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse
from urllib.parse import urljoin
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

import tomllib

try:
    import preflight_gate
except ModuleNotFoundError:  # pragma: no cover - direct script execution uses the first branch
    from tools.parity.qwen3_asr import preflight_gate


SCHEMA = "vokra-qwen3-asr-dependency-audit-v1"
MODEL_LICENSES = (
    {
        "repo": "Qwen/Qwen3-ASR-0.6B",
        "revision": "5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
    },
    {
        "repo": "Qwen/Qwen3-ASR-1.7B",
        "revision": "7278e1e70fe206f11671096ffdd38061171dd6e5",
    },
)
LICENSE_FILE_NAMES = {"license", "copying", "notice", "copyright"}
NATIVE_SUFFIXES = {".so", ".dylib", ".dll", ".pyd"}
ELF_MAGIC = b"\x7fELF"
MAX_LICENSE_BYTES = 2 * 1024 * 1024
MAX_SDIST_BYTES = 64 * 1024 * 1024
MAX_SDIST_MEMBERS = 4096
MAX_SDIST_MEMBER_BYTES = 32 * 1024 * 1024
MAX_SDIST_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
MAX_SDIST_LICENSE_TOTAL_BYTES = 4 * 1024 * 1024
MISSING_PUBLISHER_LICENSE_ROWS = frozenset(
    {
        ("cython", "3.3.0"),
        ("dynet38", "2.2"),
        ("gradio-client", "2.5.0"),
        ("qwen-omni-utils", "0.0.9"),
        ("soynlp", "0.0.493"),
        ("tokenizers", "0.22.2"),
        ("tqdm", "4.70.0"),
    }
)
PYPI_SDIST_HOST = "files.pythonhosted.org"
HF_LICENSE_REDIRECT_HOSTS = {
    "cdn-lfs.huggingface.co",
    "cdn-lfs-us-1.hf.co",
}


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


def model_license_url(repo: str, revision: str) -> str:
    return f"https://huggingface.co/{repo}/raw/{revision}/LICENSE"


def _validate_license_url(url: str, repo: str, revision: str, *, initial: bool) -> None:
    """Accept only the fixed raw URL or its bounded CDN redirect shape."""
    expected = model_license_url(repo, revision)
    if initial and url != expected:
        raise ValueError(f"unexpected initial model LICENSE URL: {url}")
    try:
        parsed = urlparse(url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"invalid model LICENSE URL: {url}") from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"unsafe model LICENSE URL: {url}")
    raw_path = urlparse(expected).path
    redirect_path = f"/{repo}/resolve/{revision}/LICENSE"
    if not initial and not (
        url == expected
        or (parsed.hostname in HF_LICENSE_REDIRECT_HOSTS and parsed.path == redirect_path)
    ):
        raise ValueError(f"non-license model URL was returned: {url}")
    if initial and (parsed.hostname != "huggingface.co" or parsed.path != raw_path):
        raise ValueError(f"unexpected initial model LICENSE URL: {url}")


def _license_file(path: Path, relative: str) -> dict[str, Any]:
    return {
        "path": relative,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _publisher_files(dist: metadata.Distribution) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for entry in sorted(dist.files or [], key=lambda value: str(value)):
        relative = str(entry)
        if not _is_license_candidate(relative):
            continue
        path = dist.locate_file(entry)
        if path.is_file() and not path.is_symlink():
            result.append(_license_file(path, relative))
    return result


def _needed_libraries(path: Path) -> dict[str, Any]:
    if not path.is_file() or not _has_elf_magic(path):
        return {"format": "non-elf", "needed": []}
    try:
        completed = subprocess.run(
            ["readelf", "-d", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"format": "elf", "needed": [], "error": str(exc)}
    needed = sorted(
        match.group(1)
        for line in completed.stdout.splitlines()
        if (match := re.search(r"\(NEEDED\).*\[([^]]+)\]", line))
    )
    return {"format": "elf", "needed": needed}


def _has_elf_magic(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(4) == ELF_MAGIC


def _native_files(dist: metadata.Distribution) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for entry in sorted(dist.files or [], key=lambda value: str(value)):
        relative = str(entry)
        path = dist.locate_file(entry)
        lower_name = path.name.casefold()
        if not (
            path.suffix.casefold() in NATIVE_SUFFIXES
            or any(token in lower_name for token in (".so.", ".dylib."))
            or lower_name.endswith(".dll")
        ) or not path.is_file() or path.is_symlink():
            continue
        libraries = _needed_libraries(path)
        result.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "bundled": True,
                "needed": libraries,
            }
        )
    return result


def _license_fields(dist: metadata.Distribution) -> dict[str, Any]:
    value = dist.metadata.get("License-Expression") or dist.metadata.get("License")
    classifiers = sorted(
        entry.removeprefix("License :: ")
        for entry in dist.metadata.get_all("Classifier", [])
        if entry.startswith("License :: ")
    )
    return {
        "license": value.strip() if isinstance(value, str) and value.strip() else None,
        "license_expression": dist.metadata.get("License-Expression"),
        "license_classifiers": classifiers,
    }


def _license_metadata_satisfied(
    license_data: dict[str, Any],
    sdist_license_evidence: dict[str, Any] | None,
) -> bool:
    return bool(
        license_data["license"]
        or license_data["license_classifiers"]
        or (
            sdist_license_evidence is not None
            and sdist_license_evidence.get("status") == "ACQUIRED_LOCKED_SDIST_LICENSE_BYTES"
            and bool(sdist_license_evidence.get("license_files"))
        )
    )


_MARKER_VARIABLES = {"implementation_name", "platform_machine", "sys_platform"}
_MARKER_TOKEN = re.compile(r"\s*(?:(and|or|==|!=|\(|\))|([A-Za-z_][A-Za-z0-9_]*)|('(?:[^'\\]|\\.)*'))")


def _marker_environment() -> dict[str, str]:
    return {
        "implementation_name": sys.implementation.name,
        "platform_machine": platform.machine().casefold(),
        "sys_platform": sys.platform,
    }


def _marker_matches(marker: Any, environment: dict[str, str]) -> bool:
    if marker is None:
        return True
    if not isinstance(marker, str) or not marker.strip():
        raise ValueError("lock marker must be a non-empty string")
    tokens: list[tuple[str, str]] = []
    position = 0
    while position < len(marker):
        match = _MARKER_TOKEN.match(marker, position)
        if match is None:
            if marker[position:].strip():
                raise ValueError(f"unsupported lock marker grammar: {marker}")
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
            raise ValueError(f"unsupported lock marker grammar: {marker}")
        cursor += 1
        return token

    def parse_atom() -> bool:
        if peek("(") is not None:
            take("(")
            result = parse_or()
            take(")")
            return result
        variable = take()[1]
        if variable not in _MARKER_VARIABLES:
            raise ValueError(f"unsupported lock marker variable: {variable}")
        operator = take()[1]
        if operator not in {"==", "!="}:
            raise ValueError(f"unsupported lock marker operator: {operator}")
        literal = take()[1]
        if tokens[cursor - 1][0] != "string" or "\\" in literal:
            raise ValueError(f"unsupported lock marker literal: {marker}")
        result = environment[variable] == literal
        return result if operator == "==" else not result

    def parse_and() -> bool:
        result = parse_atom()
        while peek("and") is not None:
            take("and")
            result = parse_atom() and result
        return result

    def parse_or() -> bool:
        result = parse_and()
        while peek("or") is not None:
            take("or")
            result = parse_and() or result
        return result

    result = parse_or()
    if cursor != len(tokens):
        raise ValueError(f"unsupported lock marker grammar: {marker}")
    return result


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        row["name"].casefold().replace("-", "_"),
        row["version"],
        canonical_json(row["source"]),
    )


def _resolved_for_environment(row: dict[str, Any], environment: dict[str, str]) -> bool:
    markers = row.get("resolution-markers")
    if markers is None:
        return True
    if not isinstance(markers, list) or not markers:
        raise ValueError(f"package resolution-markers are malformed: {row.get('name')}")
    matches = [_marker_matches(marker, environment) for marker in markers]
    if sum(matches) > 1:
        raise ValueError(f"ambiguous package resolution-markers: {row.get('name')}")
    return any(matches)


def _active_lock_graph(
    lock: dict[str, Any],
    environment: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], str]]:
    environment = _marker_environment() if environment is None else environment
    rows = lock.get("package")
    if not isinstance(rows, list):
        raise ValueError("lock package rows are malformed")
    top_markers = lock.get("resolution-markers")
    if top_markers is not None:
        if not isinstance(top_markers, list) or not top_markers:
            raise ValueError("lock resolution-markers are malformed")
        if sum(_marker_matches(marker, environment) for marker in top_markers) != 1:
            raise ValueError("lock resolution-markers are ambiguous for the current environment")
    row_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not isinstance(row.get("version"), str):
            raise ValueError("lock package identity is malformed")
        if not isinstance(row.get("source"), dict):
            raise ValueError(f"lock package source is malformed: {row.get('name')}")
        key = _row_key(row)
        if key in row_by_key:
            raise ValueError(f"duplicate lock package identity: {row['name']}=={row['version']}")
        row_by_key[key] = row
        _resolved_for_environment(row, environment)
        for dependency in row.get("dependencies", []):
            if not isinstance(dependency, dict) or not isinstance(dependency.get("name"), str):
                raise ValueError(f"lock dependency is malformed: {row['name']}")
            _marker_matches(dependency.get("marker"), environment)
        by_name.setdefault(row["name"].casefold().replace("-", "_"), []).append(row)
    roots = [row for row in rows if row.get("source") == {"virtual": "."}]
    if len(roots) != 1:
        raise ValueError("lock must contain exactly one virtual project row")
    root = roots[0]
    active: set[tuple[str, str, str]] = set()
    visiting: set[tuple[str, str, str]] = set()

    def visit(row: dict[str, Any]) -> None:
        key = _row_key(row)
        if key in visiting:
            raise ValueError(f"dependency cycle encountered at {row['name']}")
        if key in active:
            return
        visiting.add(key)
        for dependency in row.get("dependencies", []):
            if not _marker_matches(dependency.get("marker"), environment):
                continue
            name = dependency["name"].casefold().replace("-", "_")
            candidates = [candidate for candidate in by_name.get(name, []) if _resolved_for_environment(candidate, environment)]
            if "version" in dependency:
                if not isinstance(dependency["version"], str):
                    raise ValueError(f"lock dependency version is malformed: {row['name']} -> {name}")
                candidates = [candidate for candidate in candidates if candidate["version"] == dependency["version"]]
            if "source" in dependency:
                if not isinstance(dependency["source"], dict):
                    raise ValueError(f"lock dependency source is malformed: {row['name']} -> {name}")
                source = canonical_json(dependency["source"])
                candidates = [candidate for candidate in candidates if canonical_json(candidate["source"]) == source]
            if len(candidates) != 1:
                raise ValueError(f"missing or ambiguous lock dependency: {row['name']} -> {name}")
            visit(candidates[0])
        visiting.remove(key)
        active.add(key)

    visit(root)
    inactive_reasons: dict[tuple[str, str, str], str] = {}
    for row in rows:
        key = _row_key(row)
        if key in active:
            continue
        if row.get("source") == {"virtual": "."}:
            inactive_reasons[key] = "virtual project row; no installed distribution is expected"
        elif "resolution-markers" in row and not _resolved_for_environment(row, environment):
            inactive_reasons[key] = "package resolution-marker is false for the current environment"
        else:
            inactive_reasons[key] = "not reachable from the virtual project dependency graph for the current environment"
    selected = sorted(
        (row_by_key[key] for key in active if row_by_key[key].get("source") != {"virtual": "."}),
        key=lambda row: (row["name"], row["version"], canonical_json(row["source"])),
    )
    return selected, inactive_reasons


def _active_lock_packages(
    lock: dict[str, Any],
    environment: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    return _active_lock_graph(lock, environment)[0]


class _SdistAuditError(ValueError):
    def __init__(self, message: str, *, stage: str, observation: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.stage = stage
        self.observation = observation or {
            "stage": stage,
            "bytes_acquired": False,
            "observed_size": None,
            "observed_sha256": None,
            "size_verified": False,
            "sha256_verified": False,
            "final_url": None,
            "redirect_trace": [],
        }


MAX_SDIST_REDIRECTS = 3


class _BoundedSdistRedirectHandler(HTTPRedirectHandler):
    def __init__(self, expected: str) -> None:
        super().__init__()
        self.expected = expected
        self.trace = [expected]

    def _validated_target(self, current: str, target: str) -> str:
        if len(self.trace) - 1 >= MAX_SDIST_REDIRECTS:
            raise _SdistAuditError(
                "locked sdist redirect limit exceeded",
                stage="redirect_validation",
            )
        resolved = urljoin(current, target)
        _validate_sdist_url(resolved, self.expected, initial=False)
        self.trace.append(resolved)
        return resolved

    def redirect_request(self, req: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Request:
        resolved = self._validated_target(req.full_url, newurl)
        return super().redirect_request(req, fp, code, msg, headers, resolved)


def _validate_sdist_url(url: str, expected: str, *, initial: bool) -> None:
    if initial and url != expected:
        raise ValueError(f"unexpected initial locked sdist URL: {url}")
    try:
        parsed = urlparse(url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"invalid locked sdist URL: {url}") from exc
    expected_path = urlparse(expected).path
    if (
        parsed.scheme != "https"
        or parsed.hostname != PYPI_SDIST_HOST
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or parsed.query
        or parsed.fragment
        or parsed.path != expected_path
    ):
        raise ValueError(f"unsafe locked sdist URL: {url}")


def _archive_format(url: str) -> tuple[str, str]:
    lower = url.casefold()
    for suffix, mode in (
        ((".tar.gz", ".tgz"), ("tar.gz", "r:gz")),
        ((".tar.bz2", ".tbz2"), ("tar.bz2", "r:bz2")),
        ((".tar.xz", ".txz"), ("tar.xz", "r:xz")),
        ((".zip",), ("zip", "zip")),
    ):
        if lower.endswith(suffix):
            return mode
    raise ValueError("unsupported locked sdist archive format")


def _safe_archive_path(name: str) -> str:
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise ValueError("unsafe archive member path")
    raw_parts = name[:-1].split("/") if name.endswith("/") else name.split("/")
    if not raw_parts or any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError("unsafe archive member path")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("unsafe archive member path")
    if path.parts[0].endswith(":"):
        raise ValueError("unsafe archive member path")
    return str(path)


def _is_license_candidate(path: str) -> bool:
    basename = PurePosixPath(path).name.casefold()
    if basename in LICENSE_FILE_NAMES:
        return True
    return any(
        basename.startswith(f"{marker}{separator}")
        for marker in LICENSE_FILE_NAMES
        for separator in (".", "-", "_")
    )


def _bounded_license_bytes(data: bytes, total: int) -> tuple[dict[str, Any], int]:
    if len(data) > MAX_LICENSE_BYTES:
        raise ValueError("locked sdist license candidate exceeds the bounded size")
    total += len(data)
    if total > MAX_SDIST_LICENSE_TOTAL_BYTES:
        raise ValueError("locked sdist license candidates exceed the bounded total size")
    return {
        "size": len(data),
        "sha256": sha256_bytes(data),
        "content_base64": base64.b64encode(data).decode("ascii"),
    }, total


def _archive_license_files(body: bytes, url: str) -> tuple[str, list[dict[str, Any]]]:
    archive_format, mode = _archive_format(url)
    if len(body) > MAX_SDIST_BYTES:
        raise ValueError("locked sdist archive exceeds the bounded size")
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    total = 0
    member_count = 0
    uncompressed_total = 0

    if archive_format == "zip":
        with zipfile.ZipFile(BytesIO(body)) as archive:
            members = archive.infolist()
            if len(members) > MAX_SDIST_MEMBERS:
                raise ValueError("locked sdist has too many archive members")
            for member in members:
                member_count += 1
                if member_count > MAX_SDIST_MEMBERS:
                    raise ValueError("locked sdist has too many archive members")
                safe_name = _safe_archive_path(member.filename)
                if safe_name in seen:
                    raise ValueError("duplicate locked sdist archive member")
                seen.add(safe_name)
                mode_bits = (member.external_attr >> 16) & 0o170000
                if mode_bits == stat.S_IFLNK or (mode_bits and mode_bits not in {stat.S_IFREG, stat.S_IFDIR}):
                    raise ValueError("links or special files are not allowed in locked sdists")
                name_is_dir = member.filename.endswith("/")
                if (mode_bits == stat.S_IFREG and name_is_dir) or (mode_bits == stat.S_IFDIR and not name_is_dir):
                    raise ValueError("ZIP member type contradicts its directory name")
                if member.file_size < 0 or member.file_size > MAX_SDIST_MEMBER_BYTES:
                    raise ValueError("locked sdist member exceeds the bounded size")
                uncompressed_total += member.file_size
                if uncompressed_total > MAX_SDIST_UNCOMPRESSED_BYTES:
                    raise ValueError("locked sdist members exceed the bounded total size")
                if member.is_dir() or mode_bits == stat.S_IFDIR:
                    continue
                if _is_license_candidate(safe_name):
                    with archive.open(member, "r") as handle:
                        data = handle.read(MAX_LICENSE_BYTES + 1)
                    if len(data) != member.file_size:
                        raise ValueError("ZIP license member size does not match its declaration")
                    evidence, total = _bounded_license_bytes(data, total)
                    candidates.append({"path": safe_name, **evidence})
    else:
        with tarfile.open(fileobj=BytesIO(body), mode=mode) as archive:
            for member in archive:
                member_count += 1
                if member_count > MAX_SDIST_MEMBERS:
                    raise ValueError("locked sdist has too many archive members")
                safe_name = _safe_archive_path(member.name)
                if safe_name in seen:
                    raise ValueError("duplicate locked sdist archive member")
                seen.add(safe_name)
                if member.issym() or member.islnk() or not (member.isdir() or member.isreg()):
                    raise ValueError("links or special files are not allowed in locked sdists")
                if member.size < 0 or member.size > MAX_SDIST_MEMBER_BYTES:
                    raise ValueError("locked sdist member exceeds the bounded size")
                uncompressed_total += member.size
                if uncompressed_total > MAX_SDIST_UNCOMPRESSED_BYTES:
                    raise ValueError("locked sdist members exceed the bounded total size")
                if member.isdir() or not _is_license_candidate(safe_name):
                    continue
                if member.size > MAX_LICENSE_BYTES:
                    raise ValueError("locked sdist license candidate exceeds the bounded size")
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError("locked sdist license member cannot be read")
                with handle:
                    data = handle.read(MAX_LICENSE_BYTES + 1)
                if len(data) != member.size:
                    raise ValueError("tar license member size does not match its declaration")
                evidence, total = _bounded_license_bytes(data, total)
                candidates.append({"path": safe_name, **evidence})
    if not candidates:
        raise ValueError("locked sdist contains no bounded license/notice candidate")
    return archive_format, sorted(candidates, key=lambda item: item["path"])


def _controlled_sdist_error(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, _SdistAuditError):
        error = (
            _controlled_sdist_error(exc.__cause__)
            if exc.__cause__ is not None
            else {"kind": "VALIDATION_ERROR"}
        )
        error["stage"] = exc.stage
        return error
    if isinstance(exc, HTTPError):
        return {"kind": "HTTP_ERROR", "status": exc.code}
    if isinstance(exc, URLError):
        return {"kind": "URL_ERROR"}
    if isinstance(exc, ValueError):
        return {"kind": "VALIDATION_ERROR"}
    if isinstance(exc, OSError):
        return {"kind": "OS_ERROR"}
    return {"kind": "UNEXPECTED_ERROR"}


def _fetch_sdist_license(
    row: dict[str, Any],
    fetcher: Callable[[str], tuple[str, bytes]] | None = None,
) -> dict[str, Any]:
    artifact = row.get("sdist")
    if not isinstance(artifact, dict):
        raise _SdistAuditError(
            f"locked sdist is missing: {row.get('name')}=={row.get('version')}",
            stage="identity",
        )
    if set(artifact) != {"url", "hash", "size", "upload-time"}:
        raise _SdistAuditError("locked sdist artifact schema is not exact", stage="identity")
    url = artifact.get("url")
    expected_hash = artifact.get("hash")
    expected_size = artifact.get("size")
    upload_time = artifact.get("upload-time")
    if (
        not isinstance(url, str)
        or not isinstance(expected_hash, str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_hash)
        or not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or not isinstance(upload_time, str)
        or not upload_time.strip()
        or expected_size <= 0
        or expected_size > MAX_SDIST_BYTES
        or row.get("source") != {"registry": "https://pypi.org/simple"}
    ):
        raise _SdistAuditError("locked sdist identity is malformed", stage="identity")
    try:
        _validate_sdist_url(url, url, initial=True)
    except ValueError as exc:
        raise _SdistAuditError(str(exc), stage="initial_url_validation") from exc
    redirect_trace = [url]
    if fetcher is None:
        request = Request(url, headers={"Accept": "application/octet-stream", "User-Agent": "vokra-qwen3-asr-audit/1"})
        redirect_handler = _BoundedSdistRedirectHandler(url)
        opener = build_opener(redirect_handler)
        try:
            with opener.open(request, timeout=30) as response:
                final_url = response.geturl()
                redirect_trace = list(redirect_handler.trace)
                try:
                    _validate_sdist_url(final_url, url, initial=False)
                except ValueError as exc:
                    raise _SdistAuditError(
                        str(exc),
                        stage="response_url_validation",
                        observation={
                            "stage": "response_url_validation",
                            "bytes_acquired": False,
                            "observed_size": None,
                            "observed_sha256": None,
                            "size_verified": False,
                            "sha256_verified": False,
                            "final_url": final_url,
                            "redirect_trace": redirect_trace,
                        },
                    ) from exc
                body = response.read(MAX_SDIST_BYTES + 1)
        except _SdistAuditError as exc:
            # Redirect validation can fail inside the handler before a response
            # exists. Preserve the exact trace accumulated up to that point.
            if isinstance(exc.observation, dict) and not exc.observation.get("redirect_trace"):
                exc.observation["redirect_trace"] = list(redirect_handler.trace)
            raise
        except (OSError, UnicodeError) as exc:
            raise _SdistAuditError(
                "locked sdist fetch failed",
                stage="fetch",
                observation={
                    "stage": "fetch",
                    "bytes_acquired": False,
                    "observed_size": None,
                    "observed_sha256": None,
                    "size_verified": False,
                    "sha256_verified": False,
                    "final_url": None,
                    "redirect_trace": list(redirect_handler.trace),
                },
            ) from exc
    else:
        body: Any = None
        final_url: str | None = None
        try:
            final_url, body = fetcher(url)
            _validate_sdist_url(final_url, url, initial=False)
            redirect_trace = [url] if final_url == url else [url, final_url]
        except ValueError as exc:
            acquired = isinstance(body, bytes)
            observed_size = len(body) if acquired else None
            observed_sha256 = sha256_bytes(body) if acquired and observed_size <= MAX_SDIST_BYTES else None
            raise _SdistAuditError(
                str(exc),
                stage="response_url_validation",
                observation={
                    "stage": "response_url_validation",
                    "bytes_acquired": acquired,
                    "observed_size": observed_size,
                    "observed_sha256": observed_sha256,
                    "size_verified": False,
                    "sha256_verified": False,
                    "final_url": final_url,
                    "redirect_trace": [url, final_url] if final_url is not None else [url],
                },
            ) from exc
    if not isinstance(body, bytes):
        raise _SdistAuditError("locked sdist response is not bytes", stage="response_received", observation={
            "stage": "response_received",
            "bytes_acquired": False,
            "observed_size": None,
            "observed_sha256": None,
            "size_verified": False,
            "sha256_verified": False,
            "final_url": final_url,
            "redirect_trace": redirect_trace,
        })
    observed_size = len(body)
    observed_sha256 = sha256_bytes(body) if observed_size <= MAX_SDIST_BYTES else None
    observation = {
        "stage": "response_received",
        "bytes_acquired": True,
        "observed_size": observed_size,
        "observed_sha256": observed_sha256,
        "size_verified": observed_size == expected_size,
        "sha256_verified": observed_sha256 == expected_hash.removeprefix("sha256:"),
        "final_url": final_url,
        "redirect_trace": redirect_trace,
    }
    if observed_size != expected_size:
        raise _SdistAuditError("locked sdist response size does not match the lock", stage="size_verification", observation=observation)
    actual_hash = sha256_bytes(body)
    if actual_hash != expected_hash.removeprefix("sha256:"):
        raise _SdistAuditError("locked sdist response hash does not match the lock", stage="hash_verification", observation=observation)
    observation["stage"] = "archive_inspection"
    try:
        archive_format, license_files = _archive_license_files(body, url)
    except Exception as exc:
        raise _SdistAuditError(str(exc), stage="archive_inspection", observation=observation) from exc
    return {
        "status": "ACQUIRED_LOCKED_SDIST_LICENSE_BYTES",
        "package": row["name"],
        "version": row["version"],
        "archive": {
            "url": url,
            "final_url": final_url,
            "redirect_trace": redirect_trace,
            "size": expected_size,
            "hash": expected_hash,
            "upload-time": upload_time,
            "format": archive_format,
        },
        "license_files": license_files,
    }


def _blocked_sdist_record(row: dict[str, Any], exc: Exception) -> dict[str, Any]:
    artifact = row.get("sdist") if isinstance(row.get("sdist"), dict) else {}
    observation = getattr(exc, "observation", None)
    if not isinstance(observation, dict):
        observation = {
            "stage": "fetch",
            "bytes_acquired": False,
            "observed_size": None,
            "observed_sha256": None,
            "size_verified": False,
            "sha256_verified": False,
            "final_url": None,
            "redirect_trace": [artifact.get("url")],
        }
    else:
        observation = {
            "stage": "fetch",
            "bytes_acquired": False,
            "observed_size": None,
            "observed_sha256": None,
            "size_verified": False,
            "sha256_verified": False,
            "final_url": None,
            "redirect_trace": [artifact.get("url")],
            **observation,
        }
    return {
        "status": "BLOCKED_FACTUAL_SDIST_LICENSE_PATH",
        "package": row.get("name"),
        "version": row.get("version"),
        "requested_url": artifact.get("url"),
        "archive": {
            "url": artifact.get("url"),
            "final_url": observation.get("final_url"),
            "redirect_trace": observation.get("redirect_trace", [artifact.get("url")]),
            "size": artifact.get("size"),
            "hash": artifact.get("hash"),
            "upload-time": artifact.get("upload-time"),
        },
        "acquired_archive_bytes": observation.get("bytes_acquired", False),
        "archive_observation": observation,
        "license_files": [],
        "error": _controlled_sdist_error(exc),
    }


def _distribution_map() -> dict[str, list[metadata.Distribution]]:
    result: dict[str, list[metadata.Distribution]] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            result.setdefault(name.casefold().replace("-", "_"), []).append(dist)
    return result


def _closure_differences(expected: list[str], actual: list[str]) -> tuple[list[str], list[str]]:
    expected_counts = Counter(expected)
    actual_counts = Counter(actual)
    missing = sorted((expected_counts - actual_counts).elements())
    unexpected = sorted((actual_counts - expected_counts).elements())
    return missing, unexpected


def _lock_identity(row: dict[str, Any]) -> str:
    return f"{row['name']}=={row['version']} ({canonical_json(row['source'])})"


def _repository_identity(project: Path) -> dict[str, Any]:
    repository = project.parents[2]
    try:
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"git commit identity unavailable: {exc}") from exc
    return {
        "commit": commit,
        "audit_script_sha256": sha256_file(Path(__file__).resolve()),
    }


def audit_environment(
    project: Path,
    sdist_fetcher: Callable[[str], tuple[str, bytes]] | None = None,
) -> dict[str, Any]:
    lock_path = project / "uv.lock"
    pyproject_path = project / "pyproject.toml"
    lock_bytes = lock_path.read_bytes()
    pyproject_bytes = pyproject_path.read_bytes()
    lock = tomllib.loads(lock_bytes.decode("utf-8"))
    pyproject = tomllib.loads(pyproject_bytes.decode("utf-8"))
    preflight_gate._validate_lock_shape(lock, pyproject)
    if sha256_file(lock_path) != preflight_gate.LOCK_SHA256:
        raise ValueError("uv.lock bytes do not match the reviewed preflight digest")
    if sha256_file(pyproject_path) != preflight_gate.PYPROJECT_SHA256:
        raise ValueError("pyproject.toml bytes do not match the reviewed preflight digest")

    all_rows = sorted(lock["package"], key=lambda row: (row["name"], row["version"], canonical_json(row["source"])))
    expected, inactive_reasons = _active_lock_graph(lock)
    active_keys = {_row_key(row) for row in expected}
    distributions = _distribution_map()
    expected_identities = [
        f"{row['name'].casefold().replace('-', '_')}=={row['version']}" for row in expected
    ]
    actual_identities = sorted(
        f"{name}=={dist.version}"
        for name, values in distributions.items()
        for dist in values
    )
    missing, unexpected = _closure_differences(expected_identities, actual_identities)
    packages: list[dict[str, Any]] = []
    failures: list[str] = []
    dependency_acquisition_rows: list[dict[str, Any]] = []
    dependency_acquisition_not_needed: list[dict[str, Any]] = []

    def acquisition_row(row: dict[str, Any]) -> dict[str, Any]:
        artifact = row.get("sdist") if isinstance(row.get("sdist"), dict) else {}
        return {
            "package": row["name"],
            "version": row["version"],
            "url": artifact.get("url"),
            "status": "REQUESTED",
        }

    for row in expected:
        key = row["name"].casefold().replace("-", "_")
        candidates = [dist for dist in distributions.get(key, []) if dist.version == row["version"]]
        if len(candidates) != 1:
            failures.append(f"installed closure mismatch: {row['name']}=={row['version']}")
            packages.append({"lock": row, "installed": None})
            continue
        dist = candidates[0]
        license_data = _license_fields(dist)
        publisher_files = _publisher_files(dist)
        native_files = _native_files(dist)
        sdist_license_evidence: dict[str, Any] | None = None
        metadata_missing = not license_data["license"] and not license_data["license_classifiers"]
        acquisition_key = (row["name"], row["version"])
        if acquisition_key in MISSING_PUBLISHER_LICENSE_ROWS and publisher_files:
            dependency_acquisition_not_needed = [
                item for item in dependency_acquisition_not_needed
                if (item["package"], item["version"]) != acquisition_key
            ]
            dependency_acquisition_not_needed.append(
                {
                    "package": row["name"],
                    "version": row["version"],
                    "url": row.get("sdist", {}).get("url") if isinstance(row.get("sdist"), dict) else None,
                    "status": "NOT_NEEDED",
                }
            )
        if not publisher_files:
            if acquisition_key in MISSING_PUBLISHER_LICENSE_ROWS:
                request = acquisition_row(row)
                dependency_acquisition_rows.append(request)
                try:
                    sdist_license_evidence = _fetch_sdist_license(row, sdist_fetcher)
                    request["status"] = sdist_license_evidence["status"]
                except Exception as exc:
                    sdist_license_evidence = _blocked_sdist_record(row, exc)
                    request["status"] = sdist_license_evidence["status"]
                    failures.append(
                        "BLOCKED_FACTUAL_SDIST_LICENSE_PATH: "
                        f"{row['name']}=={row['version']}: "
                        f"{canonical_json(_controlled_sdist_error(exc))}"
                    )
            else:
                failures.append(f"missing publisher license/notice files: {row['name']}=={row['version']}")
        if metadata_missing and not _license_metadata_satisfied(license_data, sdist_license_evidence):
            failures.append(f"missing publisher license metadata: {row['name']}=={row['version']}")
        if any("error" in item["needed"] for item in native_files):
            failures.append(f"ELF NEEDED inspection failed: {row['name']}=={row['version']}")
        packages.append(
            {
                "lock": {
                    "name": row["name"],
                    "version": row["version"],
                    "source": row["source"],
                    "artifacts": {"sdist": row.get("sdist"), "wheels": row.get("wheels", [])},
                },
                "installed": {
                    "name": dist.metadata.get("Name"),
                    "version": dist.version,
                    **license_data,
                    "publisher_files": publisher_files,
                    "sdist_license_evidence": sdist_license_evidence,
                    "native_files": native_files,
                    "bundled_libraries": [item for item in native_files if item["bundled"]],
                },
            }
        )
    inactive_rows: list[dict[str, str]] = []
    active_packages = packages
    packages = []
    active_by_key = {
        _row_key(item["lock"]): item
        for item in active_packages
    }
    for row in all_rows:
        key = _row_key(row)
        if key in active_keys:
            packages.append(active_by_key[key])
            continue
        reason = inactive_reasons[key]
        inactive_rows.append({"identity": _lock_identity(row), "reason": reason})
        packages.append(
            {
                "lock": {
                    "name": row["name"],
                    "version": row["version"],
                    "source": row["source"],
                    "artifacts": {"sdist": row.get("sdist"), "wheels": row.get("wheels", [])},
                },
                "audit_status": "INACTIVE_LOCK_ROW",
                "inactive_reason": reason,
                "installed": None,
            }
        )
    if missing or unexpected:
        failures.append(
            "installed closure mismatch: "
            + canonical_json({"missing": missing, "unexpected": unexpected})
        )
    return {
        "schema": SCHEMA,
        "repository": _repository_identity(project),
        "project": {
            "name": pyproject["project"]["name"],
            "version": pyproject["project"]["version"],
            "pyproject_sha256": sha256_bytes(pyproject_bytes),
            "lock_sha256": sha256_bytes(lock_bytes),
        },
        "closure": {
            "platform": sys.platform,
            "python": platform.python_version(),
            "expected": expected_identities,
            "installed": actual_identities,
            "missing": missing,
            "unexpected": unexpected,
        },
        "locked_rows": [_lock_identity(row) for row in all_rows],
        "active_installed_rows": expected_identities,
        "inactive_rows": inactive_rows,
        "dependency_acquisition": {
            "policy": "exact locked PyPI sdist URLs only for missing publisher files",
            "scope": "active installed rows whose wheel lacks publisher license files",
            "rows": sorted(dependency_acquisition_rows, key=lambda item: (item["package"], item["version"])),
            "attempted_requests": sorted(
                dependency_acquisition_rows,
                key=lambda item: (item["package"], item["version"]),
            ),
            "not_needed": sorted(
                dependency_acquisition_not_needed,
                key=lambda item: (item["package"], item["version"]),
            ),
            "out_of_scope_requests": [],
            "model_files": [],
        },
        "packages": packages,
        "failures": sorted(failures),
    }


def _controlled_license_error(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, HTTPError):
        return {"kind": "HTTP_ERROR", "status": exc.code}
    if isinstance(exc, URLError):
        return {"kind": "URL_ERROR"}
    if isinstance(exc, ValueError):
        return {"kind": "VALIDATION_ERROR"}
    if isinstance(exc, OSError):
        return {"kind": "OS_ERROR"}
    return {"kind": "UNEXPECTED_ERROR"}


def _blocked_license_record(repo: str, revision: str, exc: Exception) -> dict[str, Any]:
    url = model_license_url(repo, revision)
    return {
        "status": "BLOCKED_FACTUAL_LICENSE_PATH",
        "repo": repo,
        "revision": revision,
        "requested_url": url,
        "url": url,
        "resolved_host": None,
        "resolved_path": None,
        "acquired_bytes": False,
        "size": None,
        "sha256": None,
        "content_base64": None,
        "error": _controlled_license_error(exc),
    }


def _fetch_license(
    repo: str,
    revision: str,
    fetcher: Callable[[str], tuple[str, bytes]] | None = None,
) -> dict[str, Any]:
    if not any(item["repo"] == repo and item["revision"] == revision for item in MODEL_LICENSES):
        raise ValueError(f"model LICENSE identity is not fixed in the audit allowlist: {repo}@{revision}")
    url = model_license_url(repo, revision)
    _validate_license_url(url, repo, revision, initial=True)
    if fetcher is None:
        request = Request(url, headers={"Accept": "text/plain", "User-Agent": "vokra-qwen3-asr-audit/1"})
        with urlopen(request, timeout=30) as response:  # noqa: S310 - exact URL is allow-listed below
            final_url = response.geturl()
            _validate_license_url(final_url, repo, revision, initial=False)
            parsed = urlparse(final_url)
            body = response.read(MAX_LICENSE_BYTES + 1)
    else:
        final_url, body = fetcher(url)
        _validate_license_url(final_url, repo, revision, initial=False)
        parsed = urlparse(final_url)
    if len(body) > MAX_LICENSE_BYTES:
        raise ValueError("model LICENSE response exceeds the bounded audit size")
    return {
        "repo": repo,
        "revision": revision,
        "url": url,
        "resolved_host": parsed.hostname,
        "resolved_path": parsed.path,
        "size": len(body),
        "sha256": sha256_bytes(body),
        "content_base64": base64.b64encode(body).decode("ascii"),
        "license_classification": "UNCLASSIFIED_PRIMARY_SOURCE_BYTES_ONLY",
    }


def audit_model_licenses(
    fetcher: Callable[[str], tuple[str, bytes]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for item in MODEL_LICENSES:
        try:
            records.append(_fetch_license(item["repo"], item["revision"], fetcher))
        except (OSError, UnicodeError, ValueError) as exc:
            records.append(_blocked_license_record(item["repo"], item["revision"], exc))
            error = _controlled_license_error(exc)
            failures.append(
                "BLOCKED_FACTUAL_LICENSE_PATH: "
                f"{item['repo']}@{item['revision']}: {canonical_json(error)}"
            )
    return records, failures


def _minimal_blocked_report(project: Path, exc: Exception) -> dict[str, Any]:
    project_data: dict[str, Any] = {
        "name": project.name,
        "version": None,
        "pyproject_sha256": None,
        "lock_sha256": None,
    }
    for filename, field in (("pyproject.toml", "pyproject_sha256"), ("uv.lock", "lock_sha256")):
        path = project / filename
        try:
            project_data[field] = sha256_file(path)
        except OSError:
            pass
    try:
        project_data["name"] = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))["project"]["name"]
    except (OSError, UnicodeError, ValueError, KeyError, TypeError):
        pass
    try:
        repository = _repository_identity(project)
    except Exception:
        repository = {"commit": None, "audit_script_sha256": sha256_file(Path(__file__).resolve())}
    failure = f"ENVIRONMENT_AUDIT_BLOCKED: {_controlled_license_error(exc)['kind']}"
    return {
        "schema": SCHEMA,
        "status": "BLOCKED",
        "repository": repository,
        "project": project_data,
        "closure": {
            "platform": sys.platform,
            "python": platform.python_version(),
            "expected": [],
            "installed": [],
            "missing": [],
            "unexpected": [],
        },
        "locked_rows": [],
        "active_installed_rows": [],
        "inactive_rows": [],
        "dependency_acquisition": {
            "policy": "exact locked PyPI sdist URLs only for missing publisher files",
            "scope": "active installed rows whose wheel lacks publisher license files",
            "rows": [],
            "attempted_requests": [],
            "not_needed": [],
            "out_of_scope_requests": [],
            "model_files": [],
        },
        "packages": [],
        "failures": [failure],
    }


def run(project: Path, output: Path, fetch_model_licenses: bool) -> int:
    environment_blocked = False
    try:
        report = audit_environment(project)
    except Exception as exc:  # fail closed while preserving a reviewable blocked report
        report = _minimal_blocked_report(project, exc)
        environment_blocked = True
    if fetch_model_licenses and not environment_blocked:
        records, license_failures = audit_model_licenses()
    else:
        records, license_failures = [], []
    report["model_license_files"] = records
    report["model_acquisition"] = {
        "policy": "allowlist-only LICENSE URLs",
        "requested_files": [item["requested_url"] if "requested_url" in item else item["url"] for item in records],
        "non_license_files": [],
        "model_files": [],
    }
    report["failures"] = sorted([*report.get("failures", []), *license_failures])
    report["status"] = "BLOCKED" if report["failures"] else "PASS"
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"qwen3-asr dependency audit: BLOCKED: report write failed ({type(exc).__name__})", file=sys.stderr)
        return 2
    if report["status"] == "BLOCKED":
        print("qwen3-asr dependency audit: BLOCKED: " + "; ".join(report["failures"]), file=sys.stderr)
        return 2
    print(f"qwen3-asr dependency audit: PASS ({output})")
    return 0


def self_test() -> int:
    project = Path(__file__).resolve().parent
    lock = tomllib.loads((project / "uv.lock").read_text(encoding="utf-8"))
    assert len(_active_lock_packages(lock)) == 91
    expected_torch = {"2.13.0"} if sys.platform == "darwin" else {"2.13.0+cpu"}
    assert {row["version"] for row in _active_lock_packages(lock) if row["name"] == "torch"} == expected_torch
    linux_environment = {
        "implementation_name": "cpython",
        "platform_machine": "x86_64",
        "sys_platform": "linux",
    }
    linux_active, linux_inactive = _active_lock_graph(lock, linux_environment)
    assert len(linux_active) == 91
    assert {row["version"] for row in linux_active if row["name"] == "torch"} == {"2.13.0+cpu"}
    assert linux_inactive[next(_row_key(row) for row in lock["package"] if row["name"] == "colorama")] == (
        "not reachable from the virtual project dependency graph for the current environment"
    )
    assert linux_inactive[next(_row_key(row) for row in lock["package"] if row["name"] == "tzdata")] == (
        "not reachable from the virtual project dependency graph for the current environment"
    )
    assert "resolution-marker" in linux_inactive[next(_row_key(row) for row in lock["package"] if row["name"] == "torch" and row["version"] == "2.13.0")]
    darwin_environment = {
        "implementation_name": "cpython",
        "platform_machine": "arm64",
        "sys_platform": "darwin",
    }
    darwin_active, darwin_inactive = _active_lock_graph(lock, darwin_environment)
    assert len(darwin_active) == 91
    assert {row["version"] for row in darwin_active if row["name"] == "torch"} == {"2.13.0"}
    assert "resolution-marker" in darwin_inactive[next(_row_key(row) for row in lock["package"] if row["name"] == "torch" and row["version"] == "2.13.0+cpu")]
    tampered_marker = copy.deepcopy(lock)
    virtual = next(row for row in tampered_marker["package"] if row["source"] == {"virtual": "."})
    virtual["dependencies"][0]["marker"] = "sys_platform === 'linux'"
    try:
        _active_lock_packages(tampered_marker, linux_environment)
    except ValueError:
        pass
    else:
        print("qwen3-asr dependency audit: unsupported marker grammar accepted", file=sys.stderr)
        return 1
    tampered_variable = copy.deepcopy(lock)
    virtual = next(row for row in tampered_variable["package"] if row["source"] == {"virtual": "."})
    virtual["dependencies"][0]["marker"] = "python_version == '3.12'"
    try:
        _active_lock_packages(tampered_variable, linux_environment)
    except ValueError:
        pass
    else:
        print("qwen3-asr dependency audit: unsupported marker variable accepted", file=sys.stderr)
        return 1
    tampered_cycle = copy.deepcopy(lock)
    numpy_row = next(row for row in tampered_cycle["package"] if row["name"] == "numpy")
    numpy_row["dependencies"] = [{"name": "qwen-asr"}]
    try:
        _active_lock_packages(tampered_cycle, linux_environment)
    except ValueError:
        pass
    else:
        print("qwen3-asr dependency audit: reachable dependency cycle accepted", file=sys.stderr)
        return 1
    assert _closure_differences(["torch==2.13.0+cpu"], ["torch==2.13.0"])[0] == ["torch==2.13.0+cpu"]
    assert _closure_differences(["anyio==4.14.2"], ["anyio==4.14.2", "anyio==4.14.2"])[1] == ["anyio==4.14.2"]
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    assert model_license_url(MODEL_LICENSES[0]["repo"], MODEL_LICENSES[0]["revision"]).endswith("/LICENSE")
    good_url = model_license_url(MODEL_LICENSES[0]["repo"], MODEL_LICENSES[0]["revision"])
    good_body = b"Apache License\n"
    requested_urls: list[str] = []

    def fetch_good(url: str) -> tuple[str, bytes]:
        requested_urls.append(url)
        return url, good_body

    good_records, good_failures = audit_model_licenses(fetch_good)
    assert good_records[0]["sha256"] == sha256_bytes(good_body)
    assert not good_failures
    assert requested_urls == [
        model_license_url(item["repo"], item["revision"]) for item in MODEL_LICENSES
    ]
    valid_cdn = (
        "https://cdn-lfs.huggingface.co/"
        f"{MODEL_LICENSES[0]['repo']}/resolve/{MODEL_LICENSES[0]['revision']}/LICENSE"
    )
    assert _fetch_license(
        MODEL_LICENSES[0]["repo"],
        MODEL_LICENSES[0]["revision"],
        lambda url: (valid_cdn, good_body),
    )["size"] == len(good_body)
    second_url = model_license_url(MODEL_LICENSES[1]["repo"], MODEL_LICENSES[1]["revision"])

    def blocked_record_test(
        fetcher: Callable[[str], tuple[str, bytes]],
        blocked_count: int,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        records, failures = audit_model_licenses(fetcher)
        assert len(records) == len(MODEL_LICENSES)
        assert len(failures) == blocked_count
        assert sum(item.get("status") == "BLOCKED_FACTUAL_LICENSE_PATH" for item in records) == blocked_count
        for item in records:
            if item.get("status") == "BLOCKED_FACTUAL_LICENSE_PATH":
                assert item["acquired_bytes"] is False
                assert item["content_base64"] is None
        return records, failures

    def fail_first(url: str) -> tuple[str, bytes]:
        if url == good_url:
            raise HTTPError(url, 404, "not found", {}, None)
        return url, good_body

    first_failed, first_failures = blocked_record_test(fail_first, 1)
    assert first_failed[0]["status"] == "BLOCKED_FACTUAL_LICENSE_PATH"
    assert first_failed[1]["sha256"] == sha256_bytes(good_body)
    assert first_failures[0].startswith("BLOCKED_FACTUAL_LICENSE_PATH:")

    def fail_second(url: str) -> tuple[str, bytes]:
        if url == second_url:
            raise URLError("offline")
        return url, good_body

    second_failed, second_failures = blocked_record_test(fail_second, 1)
    assert second_failed[0]["sha256"] == sha256_bytes(good_body)
    assert second_failed[1]["status"] == "BLOCKED_FACTUAL_LICENSE_PATH"
    assert second_failures[0].startswith("BLOCKED_FACTUAL_LICENSE_PATH:")

    both_failed, both_failures = blocked_record_test(
        lambda url: (_ for _ in ()).throw(ValueError("blocked path")), 2
    )
    assert all(item["status"] == "BLOCKED_FACTUAL_LICENSE_PATH" for item in both_failed)
    assert len(both_failures) == 2

    partial_records, partial_failures = blocked_record_test(
        lambda url: (valid_cdn, good_body) if url == good_url else (_ for _ in ()).throw(URLError("offline")),
        1,
    )
    assert partial_records[0]["resolved_host"] == "cdn-lfs.huggingface.co"
    assert partial_records[1]["status"] == "BLOCKED_FACTUAL_LICENSE_PATH"
    assert len(partial_failures) == 1

    blocked_responses = (
        lambda url: (url.replace("LICENSE", "model.safetensors"), b"weights"),
        lambda url: ("https://example.invalid/LICENSE", good_body),
    )
    for blocked_response in blocked_responses:
        records, failures = blocked_record_test(blocked_response, 2)
        assert all(item["status"] == "BLOCKED_FACTUAL_LICENSE_PATH" for item in records)
        assert len(failures) == 2
    for redirected in (
        "https://huggingface.co/Qwen/Qwen3-ASR-0.6B/raw/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/LICENSE.txt",
        "https://cdn-lfs.huggingface.co/Qwen/Qwen3-ASR-0.6B/model.safetensors",
        "https://cdn-lfs.huggingface.co/Qwen/Qwen3-ASR-0.6B/resolve/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/LICENSE?download=1",
        "https://cdn-lfs.huggingface.co/Qwen/Qwen3-ASR-0.6B/resolve/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/LICENSE#fragment",
        "https://user:password@cdn-lfs.huggingface.co/Qwen/Qwen3-ASR-0.6B/resolve/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/LICENSE",
        "https://cdn-lfs.huggingface.co:8443/Qwen/Qwen3-ASR-0.6B/resolve/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/LICENSE",
        "https://cdn-lfs.huggingface.co/Qwen/Qwen3-ASR-0.6B/resolve/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/LICENSE.txt",
        "https://cdn-lfs.huggingface.co/Qwen/Qwen3-ASR-0.6B/resolve/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/model.safetensors",
    ):
        records, failures = blocked_record_test(lambda url, redirected=redirected: (redirected, good_body), 2)
        assert all(item["status"] == "BLOCKED_FACTUAL_LICENSE_PATH" for item in records)
        assert len(failures) == 2
    records, failures = blocked_record_test(lambda url: (url, b"x" * (MAX_LICENSE_BYTES + 1)), 2)
    assert all(item["status"] == "BLOCKED_FACTUAL_LICENSE_PATH" for item in records)
    assert len(failures) == 2

    sdist_source_row = copy.deepcopy(next(row for row in lock["package"] if row["name"] == "cython"))

    def synthetic_tar(
        member_names: list[str],
        payload: bytes,
        *,
        symlink: bool = False,
        special: bool = False,
    ) -> bytes:
        archive_bytes = BytesIO()
        with tarfile.open(fileobj=archive_bytes, mode="w:gz") as archive:
            for member_name in member_names:
                member = tarfile.TarInfo(member_name)
                if special:
                    member.type = tarfile.FIFOTYPE
                elif symlink:
                    member.type = tarfile.SYMTYPE
                    member.linkname = "target"
                else:
                    member.size = len(payload)
                archive.addfile(member, None if (symlink or special) else BytesIO(payload))
        return archive_bytes.getvalue()

    def synthetic_row(body: bytes, *, suffix: str = ".tar.gz") -> dict[str, Any]:
        row = copy.deepcopy(sdist_source_row)
        row["sdist"]["url"] = row["sdist"]["url"].removesuffix(".tar.gz") + suffix
        row["sdist"]["size"] = len(body)
        row["sdist"]["hash"] = "sha256:" + sha256_bytes(body)
        return row

    archive_body = synthetic_tar(["cython-3.3.0/LICENSE"], b"Apache License\n")
    archive_row = synthetic_row(archive_body)
    archive_evidence = _fetch_sdist_license(archive_row, lambda url: (url, archive_body))
    assert archive_evidence["status"] == "ACQUIRED_LOCKED_SDIST_LICENSE_BYTES"
    assert archive_evidence["archive"]["format"] == "tar.gz"
    assert archive_evidence["archive"]["final_url"] == archive_row["sdist"]["url"]
    assert archive_evidence["archive"]["redirect_trace"] == [archive_row["sdist"]["url"]]
    assert archive_evidence["archive"]["upload-time"] == archive_row["sdist"]["upload-time"]
    assert archive_evidence["license_files"][0]["path"] == "cython-3.3.0/LICENSE"
    assert _is_license_candidate("LICENSE")
    assert _is_license_candidate("license.txt")
    assert _is_license_candidate("LICENSE-notice")
    assert not _is_license_candidate("unlicensed-file")
    assert not _is_license_candidate("project-license")
    for artifact_mutation in ("missing", "extra", "bool-size", "empty-upload-time"):
        malformed_row = copy.deepcopy(archive_row)
        if artifact_mutation == "missing":
            del malformed_row["sdist"]["upload-time"]
        elif artifact_mutation == "extra":
            malformed_row["sdist"]["unexpected"] = "reject"
        elif artifact_mutation == "bool-size":
            malformed_row["sdist"]["size"] = True
        else:
            malformed_row["sdist"]["upload-time"] = "  "
        try:
            _fetch_sdist_license(malformed_row, lambda url: (url, archive_body))
        except ValueError:
            pass
        else:
            print(f"qwen3-asr dependency audit: malformed sdist {artifact_mutation} accepted", file=sys.stderr)
            return 1
    dynet_row = next(row for row in lock["package"] if row["name"] == "dynet38")
    assert "sdist" not in dynet_row
    dynet_blocked = _blocked_sdist_record(
        dynet_row,
        _SdistAuditError("locked sdist is missing", stage="identity"),
    )
    assert dynet_blocked["status"] == "BLOCKED_FACTUAL_SDIST_LICENSE_PATH"
    assert dynet_blocked["requested_url"] is None
    assert dynet_blocked["acquired_archive_bytes"] is False
    assert dynet_blocked["archive_observation"]["stage"] == "identity"
    assert dynet_blocked["error"]["stage"] == "identity"
    post_fetch_blocked = _blocked_sdist_record(
        archive_row,
        _SdistAuditError(
            "hash mismatch",
            stage="hash_verification",
            observation={
                "stage": "hash_verification",
                "bytes_acquired": True,
                "observed_size": len(archive_body),
                "observed_sha256": sha256_bytes(archive_body),
                "size_verified": True,
                "sha256_verified": False,
                "final_url": archive_row["sdist"]["url"],
                "redirect_trace": [archive_row["sdist"]["url"]],
            },
        ),
    )
    assert post_fetch_blocked["acquired_archive_bytes"] is True
    assert post_fetch_blocked["archive_observation"]["sha256_verified"] is False
    null_license_metadata = {"license": None, "license_classifiers": []}
    assert _license_metadata_satisfied(null_license_metadata, archive_evidence)
    assert not _license_metadata_satisfied(
        null_license_metadata,
        {"status": "ACQUIRED_LOCKED_SDIST_LICENSE_BYTES", "license_files": []},
    )
    assert not _license_metadata_satisfied(null_license_metadata, dynet_blocked)
    zip_bytes = BytesIO()
    with zipfile.ZipFile(zip_bytes, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("cython-3.3.0/LICENSE", b"Apache License\n")
    zip_body = zip_bytes.getvalue()
    zip_row = synthetic_row(zip_body, suffix=".zip")
    assert _fetch_sdist_license(zip_row, lambda url: (url, zip_body))["archive"]["format"] == "zip"
    zip_many_bytes = BytesIO()
    with zipfile.ZipFile(zip_many_bytes, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("cython-3.3.0/LICENSE", b"license")
        archive.writestr("cython-3.3.0/NOTICE", b"notice")
    zip_many_body = zip_many_bytes.getvalue()
    zip_special_bytes = BytesIO()
    with zipfile.ZipFile(zip_special_bytes, "w") as archive:
        special_member = zipfile.ZipInfo("cython-3.3.0/LICENSE")
        special_member.external_attr = stat.S_IFIFO << 16
        archive.writestr(special_member, b"license")
    zip_special_body = zip_special_bytes.getvalue()
    zip_contradictory_bytes = BytesIO()
    with zipfile.ZipFile(zip_contradictory_bytes, "w") as archive:
        regular_directory = zipfile.ZipInfo("cython-3.3.0/LICENSE/")
        regular_directory.external_attr = stat.S_IFREG << 16
        archive.writestr(regular_directory, b"")
        directory_file = zipfile.ZipInfo("cython-3.3.0/NOTICE")
        directory_file.external_attr = stat.S_IFDIR << 16
        archive.writestr(directory_file, b"")
    zip_contradictory_body = zip_contradictory_bytes.getvalue()

    for mutation in ("url", "hash", "size"):
        tampered_row = copy.deepcopy(archive_row)
        if mutation == "url":
            tampered_row["sdist"]["url"] += "?download=1"
        elif mutation == "hash":
            tampered_row["sdist"]["hash"] = "sha256:" + "0" * 64
        else:
            tampered_row["sdist"]["size"] += 1
        try:
            _fetch_sdist_license(tampered_row, lambda url: (url, archive_body))
        except ValueError:
            pass
        else:
            print(f"qwen3-asr dependency audit: tampered sdist {mutation} accepted", file=sys.stderr)
            return 1
    try:
        _fetch_sdist_license(archive_row, lambda url: (url + "?download=1", archive_body))
    except ValueError:
        pass
    else:
        print("qwen3-asr dependency audit: redirected sdist query accepted", file=sys.stderr)
        return 1
    for unsafe_redirect in (
        archive_row["sdist"]["url"].replace("https://", "https://user:password@", 1),
        archive_row["sdist"]["url"].replace("files.pythonhosted.org", "files.pythonhosted.org:8443", 1),
        archive_row["sdist"]["url"].removesuffix(".tar.gz") + "/LICENSE.txt",
    ):
        try:
            _fetch_sdist_license(archive_row, lambda url, unsafe=unsafe_redirect: (unsafe, archive_body))
        except ValueError:
            pass
        else:
            print("qwen3-asr dependency audit: unsafe sdist redirect accepted", file=sys.stderr)
            return 1
    redirect_handler = _BoundedSdistRedirectHandler(archive_row["sdist"]["url"])
    assert redirect_handler._validated_target(archive_row["sdist"]["url"], archive_row["sdist"]["url"]) == archive_row["sdist"]["url"]
    try:
        redirect_handler._validated_target(
            archive_row["sdist"]["url"],
            archive_row["sdist"]["url"] + "?download=1",
        )
    except ValueError:
        pass
    else:
        print("qwen3-asr dependency audit: redirect handler accepted query", file=sys.stderr)
        return 1
    redirect_handler = _BoundedSdistRedirectHandler(archive_row["sdist"]["url"])
    for _ in range(MAX_SDIST_REDIRECTS):
        redirect_handler._validated_target(archive_row["sdist"]["url"], archive_row["sdist"]["url"])
    try:
        redirect_handler._validated_target(archive_row["sdist"]["url"], archive_row["sdist"]["url"])
    except ValueError:
        pass
    else:
        print("qwen3-asr dependency audit: redirect limit not enforced", file=sys.stderr)
        return 1

    for invalid_body in (
        synthetic_tar(["../LICENSE"], b"license"),
        synthetic_tar(["/LICENSE"], b"license"),
        synthetic_tar(["cython-3.3.0/a//b"], b"payload"),
        synthetic_tar(["cython-3.3.0/a/./b"], b"payload"),
        synthetic_tar(["cython-3.3.0/README.md"], b"readme"),
        synthetic_tar(["cython-3.3.0/LICENSE", "cython-3.3.0/LICENSE"], b"license"),
        synthetic_tar(["cython-3.3.0/LICENSE"], b"license", symlink=True),
        synthetic_tar(["cython-3.3.0/LICENSE"], b"license", special=True),
        synthetic_tar(["cython-3.3.0/LICENSE"], b"x" * (MAX_LICENSE_BYTES + 1)),
    ):
        try:
            _archive_license_files(invalid_body, archive_row["sdist"]["url"])
        except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile):
            pass
        else:
            print("qwen3-asr dependency audit: unsafe or unlicensed sdist accepted", file=sys.stderr)
            return 1
    old_member_limit = globals()["MAX_SDIST_MEMBERS"]
    old_total_limit = globals()["MAX_SDIST_UNCOMPRESSED_BYTES"]
    try:
        globals()["MAX_SDIST_MEMBERS"] = 1
        for limited_body, limited_url in (
            (synthetic_tar(["cython-3.3.0/LICENSE", "cython-3.3.0/README.md"], b"license"), archive_row["sdist"]["url"]),
            (zip_many_body, zip_row["sdist"]["url"]),
        ):
            try:
                _archive_license_files(limited_body, limited_url)
            except ValueError:
                continue
            print("qwen3-asr dependency audit: archive member limit not enforced", file=sys.stderr)
            return 1
    finally:
        globals()["MAX_SDIST_MEMBERS"] = old_member_limit
    try:
        globals()["MAX_SDIST_UNCOMPRESSED_BYTES"] = 1
        for limited_body, limited_url in (
            (synthetic_tar(["cython-3.3.0/LICENSE", "cython-3.3.0/NOTICE"], b"license"), archive_row["sdist"]["url"]),
            (zip_many_body, zip_row["sdist"]["url"]),
        ):
            try:
                _archive_license_files(limited_body, limited_url)
            except ValueError:
                continue
            print("qwen3-asr dependency audit: archive aggregate limit not enforced", file=sys.stderr)
            return 1
    finally:
        globals()["MAX_SDIST_UNCOMPRESSED_BYTES"] = old_total_limit
    try:
        _archive_license_files(zip_special_body, zip_row["sdist"]["url"])
    except ValueError:
        pass
    else:
        print("qwen3-asr dependency audit: ZIP special member accepted", file=sys.stderr)
        return 1
    try:
        _archive_license_files(zip_contradictory_body, zip_row["sdist"]["url"])
    except ValueError:
        pass
    else:
        print("qwen3-asr dependency audit: ZIP type/name contradiction accepted", file=sys.stderr)
        return 1
    try:
        _archive_license_files(archive_body, archive_row["sdist"]["url"].removesuffix(".tar.gz") + ".whl")
    except ValueError:
        pass
    else:
        print("qwen3-asr dependency audit: unsupported sdist archive accepted", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="qwen3-asr-audit-blocked-") as directory:
        output = Path(directory) / "blocked.json"
        original_audit_environment = globals()["audit_environment"]
        original_audit_model_licenses = globals()["audit_model_licenses"]
        globals()["audit_environment"] = lambda project: {
            "schema": SCHEMA,
            "closure": {"expected": ["closure==facts"], "installed": ["closure==facts"]},
            "dependency_acquisition": {
                "policy": "exact locked PyPI sdist URLs only for missing publisher files",
                "scope": "active installed rows whose wheel lacks publisher license files",
                "rows": [],
                "attempted_requests": [],
                "not_needed": [],
                "out_of_scope_requests": [],
                "model_files": [],
            },
            "packages": [
                {
                    "installed": {
                        "sdist_license_evidence": {
                            "status": "BLOCKED_FACTUAL_SDIST_LICENSE_PATH",
                            "acquired_archive_bytes": False,
                        }
                    }
                }
            ],
            "failures": [],
        }
        blocked = {
            "status": "BLOCKED_FACTUAL_LICENSE_PATH",
            "repo": MODEL_LICENSES[0]["repo"],
            "revision": MODEL_LICENSES[0]["revision"],
            "requested_url": good_url,
            "acquired_bytes": False,
            "content_base64": None,
        }
        globals()["audit_model_licenses"] = lambda: ([blocked], ["BLOCKED_FACTUAL_LICENSE_PATH: test"])
        try:
            assert run(Path(directory) / "project", output, True) == 2
        finally:
            globals()["audit_environment"] = original_audit_environment
            globals()["audit_model_licenses"] = original_audit_model_licenses
        written = json.loads(output.read_text(encoding="utf-8"))
        assert written["status"] == "BLOCKED"
        assert written["closure"]["expected"] == ["closure==facts"]
        assert written["dependency_acquisition"]["out_of_scope_requests"] == []
        assert written["dependency_acquisition"]["model_files"] == []
        assert written["packages"][0]["installed"]["sdist_license_evidence"]["status"] == "BLOCKED_FACTUAL_SDIST_LICENSE_PATH"
        assert written["model_license_files"][0]["status"] == "BLOCKED_FACTUAL_LICENSE_PATH"

    with tempfile.TemporaryDirectory(prefix="qwen3-asr-audit-invalid-") as directory:
        output = Path(directory) / "blocked.json"
        assert run(Path(directory) / "missing-project", output, False) == 2
        written = json.loads(output.read_text(encoding="utf-8"))
        assert written["status"] == "BLOCKED"
        assert written["failures"]
        assert written["dependency_acquisition"] == {
            "policy": "exact locked PyPI sdist URLs only for missing publisher files",
            "scope": "active installed rows whose wheel lacks publisher license files",
            "rows": [],
            "attempted_requests": [],
            "not_needed": [],
            "out_of_scope_requests": [],
            "model_files": [],
        }
    with tempfile.TemporaryDirectory(prefix="qwen3-asr-audit-") as directory:
        output = Path(directory) / "audit.json"
        short = output.with_name("short")
        short.write_bytes(b"not an ELF payload")
        assert _needed_libraries(short) == {"format": "non-elf", "needed": []}
        report = {
            "schema": SCHEMA,
            "model_license_files": [],
            "model_acquisition": {"policy": "allowlist-only LICENSE URLs", "requested_files": [], "non_license_files": []},
        }
        output.write_text(canonical_json(report) + "\n", encoding="utf-8")
        assert json.loads(output.read_text(encoding="utf-8"))["schema"] == SCHEMA
    print("qwen3-asr dependency audit: self-test PASS")
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
