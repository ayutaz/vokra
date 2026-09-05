#!/usr/bin/env python3
"""Model-free factual audit for the pinned Parler-TTS reference closure.

The preflight gate authenticates the files and approval before synchronization.
This module is deliberately a second, post-sync operation: it inspects only
``importlib.metadata`` records and installed files.  It does not import
Parler-TTS, Transformers, Torch, DAC, or any other model implementation.  Its
only network operations are allow-listed requests for exact locked PyPI sdists
needed when an installed distribution lacks publisher evidence, plus the four
exact primary-source ``LICENSE`` files in the existing Parler contract.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from email.message import Message
import hashlib
import importlib.metadata as metadata
import io
import json
import platform
import posixpath
import re
import stat
import subprocess
import tarfile
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qsl, urlsplit
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

import tomllib

try:
    import preflight_gate
except ModuleNotFoundError:  # pragma: no cover - direct script execution uses this path
    from tools.parity.parler_tts import preflight_gate


SCHEMA = "vokra-parler-tts-dependency-audit-v1"
MAX_LICENSE_BYTES = 2 * 1024 * 1024
MAX_SDIST_BYTES = 16 * 1024 * 1024
MAX_ARCHIVE_MEMBER_BYTES = 8 * 1024 * 1024
MAX_ARCHIVE_TOTAL_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 10000
MAX_SDIST_REDIRECTS = 4
ELF_MAGIC = b"\x7fELF"
NATIVE_SUFFIXES = {".so", ".dylib", ".dll", ".pyd"}
LICENSE_FILE_NAMES = {"license", "licence", "copying", "notice", "copyright"}
HF_LICENSE_HOSTS = {
    "huggingface.co",
    "hf.co",
    "cdn-lfs.huggingface.co",
    "cdn-lfs-us-1.hf.co",
}
HF_MODEL_INFO_HOSTS = {"huggingface.co", "hf.co"}
MAX_MODEL_INFO_BYTES = 2 * 1024 * 1024
MAX_MODEL_INFO_REDIRECTS = 4
MODEL_METADATA_SCHEMA = "vokra-parler-tts-hf-model-metadata-v1"
MODEL_INFO_REQUIRED_KEYS = {"id", "sha", "private", "gated", "disabled", "cardData", "siblings"}
PYPI_FILE_HOST = "files.pythonhosted.org"


class AuditError(ValueError):
    """A fail-closed contract or factual environment error."""


class LicensePathError(AuditError):
    """A fixed LICENSE path could not be fetched, with HTTP status if known."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash in bounded chunks so a large native library is never read eagerly."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def normalized_name(name: str) -> str:
    """PEP 503 normalization used for the exact installed closure multiset."""
    return re.sub(r"[-_.]+", "-", name.strip()).casefold()


def normalized_version(version: str) -> str:
    # uv records the exact PEP 440 spelling.  Case folding is only presentation
    # normalization; it does not discard a local version such as +cpu.
    return re.sub(r"\s+", "", version.strip()).casefold()


def identity(name: str, version: str) -> str:
    return f"{normalized_name(name)}=={normalized_version(version)}"


def reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuditError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_json_keys)
    except (OSError, UnicodeError, json.JSONDecodeError, AuditError) as exc:
        raise AuditError(f"cannot read JSON {path}: {exc}") from exc


def regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _contract(project: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bytes, bytes]:
    project_path = project / "pyproject.toml"
    lock_path = project / "uv.lock"
    manifest_path = project / "license_gate_manifest.json"
    if not all(regular_file(path) for path in (project_path, lock_path, manifest_path)):
        raise AuditError("Parler pyproject.toml, uv.lock, or gate manifest is missing/symlinked")
    try:
        project_bytes = project_path.read_bytes()
        lock_bytes = lock_path.read_bytes()
        project_data = tomllib.loads(project_bytes.decode("utf-8"))
        lock_data = tomllib.loads(lock_bytes.decode("utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise AuditError(f"Parler closure bytes are unreadable: {exc}") from exc
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("gate_version") != preflight_gate.GATE_VERSION:
        raise AuditError("Parler gate manifest version is unsupported")
    if sha256_bytes(project_bytes) != preflight_gate.PYPROJECT_SHA256:
        raise AuditError("pyproject.toml bytes differ from the reviewed Parler contract")
    if sha256_bytes(lock_bytes) != preflight_gate.LOCK_SHA256:
        raise AuditError("uv.lock bytes differ from the reviewed Parler contract")
    try:
        preflight_gate.validate_project_schema(project_data)
        rows = preflight_gate.canonical_package_rows(lock_data, project_data["project"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AuditError(f"Parler pyproject/uv.lock schema is invalid: {exc}") from exc
    if preflight_gate.canonical_digest(rows) != manifest.get("package_rows_sha256"):
        raise AuditError("canonical Parler lock rows differ from the manifest")
    if manifest.get("source_identity") != {
        "repo": preflight_gate.SOURCE_REPO,
        "revision": preflight_gate.SOURCE_REVISION,
        "license": "Apache-2.0",
    }:
        raise AuditError("Parler source identity drifted from the reviewed manifest")
    if manifest.get("variants") != preflight_gate.VARIANTS:
        raise AuditError("Parler model variant identities drifted from the reviewed manifest")
    if manifest.get("dac_identity") != preflight_gate.DAC_IDENTITY:
        raise AuditError("Parler DAC identity drifted from the reviewed manifest")
    if manifest.get("model_metadata_fallback") != preflight_gate.MODEL_METADATA_FALLBACK:
        raise AuditError("Parler HF model-metadata fallback contract drifted from the reviewed manifest")
    return project_data, lock_data, manifest, project_bytes, lock_bytes


def _expected_packages(lock: dict[str, Any]) -> list[dict[str, Any]]:
    rows = lock.get("package")
    if not isinstance(rows, list) or not rows:
        raise AuditError("uv.lock has no package rows")
    registry = [row for row in rows if row.get("source") != {"virtual": "."}]
    if not registry:
        raise AuditError("uv.lock has no registry package rows")
    for row in registry:
        source = row.get("source")
        if not isinstance(source, dict) or source.get("registry") not in {
            "https://pypi.org/simple",
            "https://download.pytorch.org/whl/cpu",
        }:
            raise AuditError(f"uv.lock has an unapproved package index: {row.get('name')}")
    return sorted(registry, key=lambda row: (normalized_name(row["name"]), normalized_version(row["version"])))


def classify_lock_rows(lock: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Account for every row, including the virtual project row.

    This exact lock is constrained to Linux x86_64 by pyproject.toml.  Registry
    rows are therefore active-installed candidates; the virtual project row is
    explicitly inactive and must not be mistaken for an installed wheel.
    """
    active: list[dict[str, Any]] = []
    inactive: list[dict[str, Any]] = []
    for row in sorted(lock["package"], key=lambda item: (normalized_name(item["name"]), normalized_version(item["version"]))):
        item = {"name": row["name"], "version": row["version"], "source": row["source"]}
        if row.get("source") == {"virtual": "."}:
            inactive.append({
                "identity": identity(row["name"], row["version"]),
                "source": row["source"],
                "status": "INACTIVE_VIRTUAL_PROJECT",
                "reason": "virtual project row; no installed distribution is expected",
            })
        else:
            active.append({
                **item,
                "identity": identity(row["name"], row["version"]),
                "status": "ACTIVE_LINUX_INSTALLED",
                "reason": "registry row is active under the locked Linux x86_64 environment",
            })
    return active, inactive


def _distribution_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        version = dist.version
        if not name or not version:
            continue
        records.append({
            "distribution": dist,
            "identity": identity(name, version),
            "name": name,
            "version": version,
            "location": str(Path(dist.locate_file(""))),
        })
    return sorted(records, key=lambda item: (item["identity"], item["location"]))


def compare_multiset(expected: list[str], actual: list[str]) -> dict[str, Any]:
    expected_counts = Counter(expected)
    actual_counts = Counter(actual)
    missing = sorted((expected_counts - actual_counts).elements())
    unexpected = sorted((actual_counts - expected_counts).elements())
    duplicate_identities = sorted(item for item, count in actual_counts.items() if count > 1)
    return {
        "expected": sorted(expected),
        "installed": sorted(actual),
        "missing": missing,
        "unexpected": unexpected,
        "duplicate_identities": duplicate_identities,
        "exact": not missing and not unexpected,
    }


def _metadata_fields(dist: metadata.Distribution) -> dict[str, Any]:
    classifiers = sorted(
        value.removeprefix("License :: ")
        for value in (dist.metadata.get_all("Classifier") or [])
        if value.startswith("License :: ")
    )
    expression = dist.metadata.get("License-Expression")
    declared = dist.metadata.get("License")
    return {
        "license": declared.strip() if isinstance(declared, str) and declared.strip() else None,
        "license_expression": expression.strip() if isinstance(expression, str) and expression.strip() else None,
        "license_classifiers": classifiers,
    }


def _entry_path(dist: metadata.Distribution, entry: Any) -> Path | None:
    path = Path(dist.locate_file(entry))
    root = Path(dist.locate_file(""))
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        return None
    return path


def _is_license_path(relative: str) -> bool:
    basename = Path(relative).name.casefold()
    stem = Path(basename).stem
    return stem in LICENSE_FILE_NAMES or any(
        token in basename for token in ("license", "licence", "copying", "notice", "copyright")
    )


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


def _validate_sdist_url(artifact: dict[str, Any], url: str, *, initial: bool = False) -> None:
    """Validate a locked PyPI sdist URL before any bytes are accepted."""
    expected = artifact.get("url")
    if not isinstance(expected, str) or not isinstance(url, str):
        raise AuditError("locked sdist has no exact URL")
    parsed = urlsplit(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise AuditError(f"locked sdist URL has an invalid port: {url}") from exc
    if parsed.scheme != "https" or parsed.hostname != PYPI_FILE_HOST or port not in (None, 443):
        raise AuditError(f"locked sdist URL is not the official PyPI file host: {url}")
    if parsed.username is not None or parsed.password is not None:
        raise AuditError(f"locked sdist URL contains userinfo: {url}")
    if parsed.query or parsed.fragment or not parsed.path:
        raise AuditError(f"locked sdist URL contains query, fragment, or empty path: {url}")
    expected_parts = urlsplit(expected)
    try:
        expected_port = expected_parts.port
    except ValueError as exc:
        raise AuditError("locked sdist URL in uv.lock has an invalid port") from exc
    if (
        expected_parts.scheme != "https"
        or expected_parts.hostname != PYPI_FILE_HOST
        or expected_parts.username is not None
        or expected_parts.password is not None
        or expected_parts.query
        or expected_parts.fragment
        or expected_port not in (None, 443)
    ):
        raise AuditError("locked sdist URL in uv.lock is not an exact official PyPI URL")
    if initial and url != expected:
        raise AuditError(f"initial locked sdist URL differs from uv.lock: {url}")
    if parsed.path != expected_parts.path:
        raise AuditError(f"locked sdist redirect changed the exact path: {url}")


class _SdistRedirects(HTTPRedirectHandler):
    def __init__(self, artifact: dict[str, Any], trace: list[str]) -> None:
        super().__init__()
        self.artifact = artifact
        self.trace = trace

    def redirect_request(self, request: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Request | None:
        if len(self.trace) - 1 >= MAX_SDIST_REDIRECTS:
            raise AuditError("locked sdist redirect chain exceeds the bounded limit")
        _validate_sdist_url(self.artifact, newurl)
        self.trace.append(newurl)
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def _archive_member_path(name: str) -> str:
    """Return a safe POSIX archive path, rejecting traversal and absolutes."""
    if not isinstance(name, str) or not name or "\x00" in name:
        raise AuditError("sdist archive contains an invalid member path")
    if "\\" in name:
        raise AuditError(f"sdist archive contains a backslash member path: {name}")
    portable = name
    if portable.startswith("/") or re.match(r"^[A-Za-z]:/", portable):
        raise AuditError(f"sdist archive contains an absolute member path: {name}")
    parts = portable.split("/")
    if any(part in ("..", ".") for part in parts):
        raise AuditError(f"sdist archive contains a traversal member path: {name}")
    if any(part == "" for part in parts[:-1]):
        raise AuditError(f"sdist archive contains an empty member path component: {name}")
    normalized = posixpath.normpath(portable)
    if normalized in ("", ".") or normalized.startswith("../"):
        raise AuditError(f"sdist archive contains an unsafe member path: {name}")
    return normalized


def _archive_format(url: str) -> str:
    path = urlsplit(url).path.casefold()
    for suffix, archive_format in (
        (".tar.gz", "tar.gz"),
        (".tgz", "tar.gz"),
        (".tar.bz2", "tar.bz2"),
        (".tbz2", "tar.bz2"),
        (".tar.xz", "tar.xz"),
        (".txz", "tar.xz"),
        (".zip", "zip"),
    ):
        if path.endswith(suffix):
            return archive_format
    raise AuditError(f"unsupported locked sdist archive format: {path}")


def _license_candidates_from_archive_impl(body: bytes, archive_format: str, archive_identity: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    total_uncompressed = 0

    def add_candidate(path: str, payload: bytes) -> None:
        if not _is_license_path(path):
            return
        if len(payload) > MAX_LICENSE_BYTES:
            raise AuditError(f"sdist publisher license member is oversized: {path}")
        if sum(item["size"] for item in candidates) + len(payload) > MAX_LICENSE_BYTES:
            raise AuditError("sdist publisher license members exceed the bounded aggregate")
        candidates.append({
            "path": path,
            "size": len(payload),
            "sha256": sha256_bytes(payload),
            "content_base64": base64.b64encode(payload).decode("ascii"),
            "archive_identity": archive_identity,
        })

    if archive_format == "zip":
        try:
            archive = zipfile.ZipFile(io.BytesIO(body))
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise AuditError(f"locked sdist ZIP is unreadable: {exc}") from exc
        with archive:
            for member_number, info in enumerate(archive.infolist(), start=1):
                if member_number > MAX_ARCHIVE_MEMBERS:
                    raise AuditError("sdist archive contains too many members")
                path = _archive_member_path(info.filename)
                if path in seen_paths:
                    raise AuditError(f"sdist archive contains duplicate member path: {path}")
                seen_paths.add(path)
                file_type = (info.external_attr >> 16) & 0o170000
                if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise AuditError(f"sdist archive contains a special member: {path}")
                if file_type == stat.S_IFDIR and not info.is_dir():
                    raise AuditError(f"sdist archive contains a malformed directory member: {path}")
                if info.is_dir():
                    continue
                if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                    raise AuditError(f"sdist archive member is oversized: {path}")
                total_uncompressed += info.file_size
                if total_uncompressed > MAX_ARCHIVE_TOTAL_BYTES:
                    raise AuditError("sdist archive aggregate is oversized")
                if _is_license_path(path):
                    try:
                        payload = archive.read(info)
                    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                        raise AuditError(f"sdist publisher license member is unreadable: {path}: {exc}") from exc
                    if len(payload) != info.file_size:
                        raise AuditError(f"sdist publisher license member size changed: {path}")
                    add_candidate(path, payload)
    else:
        try:
            archive = tarfile.open(
                fileobj=io.BytesIO(body),
                mode={"tar.gz": "r:gz", "tar.bz2": "r:bz2", "tar.xz": "r:xz"}[archive_format],
            )
        except (OSError, tarfile.TarError) as exc:
            raise AuditError(f"locked sdist tar archive is unreadable: {exc}") from exc
        with archive:
            for member_number, member in enumerate(archive, start=1):
                if member_number > MAX_ARCHIVE_MEMBERS:
                    raise AuditError("sdist archive contains too many members")
                path = _archive_member_path(member.name)
                if path in seen_paths:
                    raise AuditError(f"sdist archive contains duplicate member path: {path}")
                seen_paths.add(path)
                if member.isdir():
                    continue
                if not member.isfile():
                    raise AuditError(f"sdist archive contains a non-file/non-directory member: {path}")
                if member.size < 0 or member.size > MAX_ARCHIVE_MEMBER_BYTES:
                    raise AuditError(f"sdist archive member is oversized: {path}")
                total_uncompressed += member.size
                if total_uncompressed > MAX_ARCHIVE_TOTAL_BYTES:
                    raise AuditError("sdist archive aggregate is oversized")
                if _is_license_path(path):
                    if not member.isfile():
                        raise AuditError(f"sdist publisher license candidate is not a regular file: {path}")
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise AuditError(f"sdist publisher license member is unreadable: {path}")
                    payload = stream.read(MAX_LICENSE_BYTES + 1)
                    if len(payload) != member.size:
                        raise AuditError(f"sdist publisher license member size changed: {path}")
                    add_candidate(path, payload)
    if not candidates:
        raise AuditError("locked sdist contains no LICENSE/LICENCE/COPYING/NOTICE/COPYRIGHT candidate")
    return candidates


def _license_candidates_from_archive(body: bytes, archive_format: str, archive_identity: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        return _license_candidates_from_archive_impl(body, archive_format, archive_identity)
    except AuditError:
        raise
    except Exception as exc:  # noqa: BLE001 - malformed archives become factual blockers
        raise AuditError(f"locked sdist archive inspection failed: {exc}") from exc


def _fetch_locked_sdist(
    row: dict[str, Any], fetcher: Callable[[str], tuple[str, bytes]] | None = None
) -> dict[str, Any]:
    artifact = row.get("sdist")
    if not isinstance(artifact, dict):
        raise AuditError(f"no locked sdist is available for {identity(row['name'], row['version'])}")
    if set(artifact) != {"url", "hash", "size", "upload-time"}:
        raise AuditError(f"locked sdist schema is incomplete for {identity(row['name'], row['version'])}")
    expected_url = artifact["url"]
    expected_hash = artifact["hash"]
    expected_size = artifact["size"]
    if not isinstance(expected_hash, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_hash):
        raise AuditError("locked sdist hash is not a SHA-256")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size <= 0 or expected_size > MAX_SDIST_BYTES:
        raise AuditError("locked sdist size is missing, invalid, or above the audit bound")
    _validate_sdist_url(artifact, expected_url, initial=True)
    trace = [expected_url]
    if fetcher is None:
        opener = build_opener(_SdistRedirects(artifact, trace))
        request = Request(expected_url, headers={"Accept": "application/octet-stream", "User-Agent": "vokra-parler-tts-audit/1"})
        try:
            with opener.open(request, timeout=30) as response:  # noqa: S310 - exact host/path is validated
                final_url = response.geturl()
                _validate_sdist_url(artifact, final_url)
                if final_url not in trace:
                    trace.append(final_url)
                content_length = response.headers.get("Content-Length")
                if content_length and content_length.isdigit() and int(content_length) > MAX_SDIST_BYTES:
                    raise AuditError("locked sdist Content-Length exceeds the bounded audit size")
                body = response.read(MAX_SDIST_BYTES + 1)
        except AuditError:
            raise
        except Exception as exc:  # noqa: BLE001 - network failure is a factual blocker
            raise AuditError(f"locked sdist is not retrievable: {exc}") from exc
    else:
        try:
            final_url, body = fetcher(expected_url)
        except Exception as exc:  # noqa: BLE001 - synthetic/transport failure is a factual blocker
            raise AuditError(f"locked sdist is not retrievable: {exc}") from exc
        _validate_sdist_url(artifact, final_url)
        if not trace or trace[-1] != final_url:
            trace.append(final_url)
    if not isinstance(body, bytes):
        raise AuditError("locked sdist response was not bytes")
    if len(body) > MAX_SDIST_BYTES:
        raise AuditError("locked sdist response exceeds the bounded audit size")
    actual_hash = "sha256:" + sha256_bytes(body)
    if len(body) != expected_size:
        raise AuditError(f"locked sdist size mismatch: expected {expected_size}, got {len(body)}")
    if actual_hash != expected_hash:
        raise AuditError(f"locked sdist hash mismatch: expected {expected_hash}, got {actual_hash}")
    archive_format = _archive_format(expected_url)
    archive_identity = {
        "requested_url": expected_url,
        "final_url": final_url,
        "url_trace": trace,
        "size": len(body),
        "sha256": actual_hash,
        "format": archive_format,
    }
    return {
        "status": "PASS",
        "archive_identity": archive_identity,
        "publisher_files": _license_candidates_from_archive(body, archive_format, archive_identity),
    }


def _first_four(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read(4)


def _elf_needed(path: Path) -> dict[str, Any]:
    magic = _first_four(path)
    if magic != ELF_MAGIC:
        return {"format": "non-elf", "needed": [], "inspection": "not-applicable"}
    try:
        completed = subprocess.run(
            ["readelf", "-d", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"format": "elf", "needed": [], "inspection": "error", "error": str(exc)}
    output = f"{completed.stdout}\n{completed.stderr}"
    needed = sorted(
        match.group(1)
        for line in completed.stdout.splitlines()
        if (match := re.search(r"\(NEEDED\).*\[([^]]+)\]", line))
    )
    if completed.returncode == 0:
        inspection = "ok"
    elif "no dynamic section" in output.casefold():
        inspection = "no-dynamic-section"
    else:
        inspection = "error"
    result: dict[str, Any] = {
        "format": "elf",
        "needed": needed,
        "inspection": inspection,
        "readelf_returncode": completed.returncode,
    }
    if inspection == "error":
        result["error"] = output.strip()[-2000:]
    return result


def _native_files(dist: metadata.Distribution) -> tuple[list[dict[str, Any]], list[str]]:
    result: list[dict[str, Any]] = []
    unsafe: list[str] = []
    for entry in sorted(dist.files or [], key=str):
        relative = str(entry)
        entry_name = Path(relative).name.casefold()
        suffix_candidate = (
            Path(entry_name).suffix.casefold() in NATIVE_SUFFIXES
            or ".so." in entry_name
            or entry_name.endswith(".dll")
        )
        try:
            path = _entry_path(dist, entry)
        except (OSError, ValueError) as exc:
            if suffix_candidate:
                unsafe.append(f"{relative}:locate-failed:{exc}")
            continue
        if path is None:
            if suffix_candidate:
                unsafe.append(f"{relative}:out-of-distribution-root")
            continue
        if path.is_symlink():
            if suffix_candidate:
                unsafe.append(f"{relative}:symlink")
            continue
        if not path.is_file():
            if suffix_candidate:
                unsafe.append(f"{relative}:missing-or-not-regular")
            continue
        try:
            magic = _first_four(path)
        except OSError as exc:
            if suffix_candidate:
                unsafe.append(f"{relative}:magic-read-failed:{exc}")
            continue
        if not suffix_candidate and magic != ELF_MAGIC:
            continue
        try:
            inspection = _elf_needed(path)
            size = path.stat().st_size
            digest = sha256_file(path)
        except OSError as exc:
            unsafe.append(f"{relative}:native-read-failed:{exc}")
            continue
        result.append({
            "distribution_shipped": True,
            "bundled": True,
            "origin": "installed-distribution",
            "path": relative,
            "size": size,
            "sha256": digest,
            "candidate": "elf-magic" if magic == ELF_MAGIC and not suffix_candidate else "native-suffix",
            "needed": inspection,
        })
    return result, unsafe


def _inspect_package(
    row: dict[str, Any],
    record: dict[str, Any] | None,
    duplicate: bool,
    sdist_fetcher: Callable[[str], tuple[str, bytes]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    lock_data = {
        "name": row["name"],
        "version": row["version"],
        "source": row["source"],
        "artifacts": {"sdist": row.get("sdist"), "wheels": row.get("wheels", [])},
    }
    if record is None:
        return {"lock": lock_data, "installed": None}, [f"installed closure missing: {identity(row['name'], row['version'])}"]
    dist = record["distribution"]
    publisher, unsafe_publisher = _publisher_files(dist)
    native, unsafe_native = _native_files(dist)
    locked_sdist: dict[str, Any] | None = None
    if not publisher:
        try:
            locked_sdist = _fetch_locked_sdist(row, sdist_fetcher)
        except AuditError as exc:
            locked_sdist = {
                "status": "BLOCKED",
                "archive_identity": {
                    "requested_url": row.get("sdist", {}).get("url") if isinstance(row.get("sdist"), dict) else None,
                },
                "publisher_files": [],
                "error": str(exc),
            }
    installed = {
        "name": dist.metadata.get("Name"),
        "version": dist.version,
        "normalized_identity": record["identity"],
        "location": record["location"],
        **_metadata_fields(dist),
        "publisher_files": publisher,
        "locked_sdist_license_audit": locked_sdist,
        "native_files": native,
        "bundled_libraries": [
            {
                "distribution": dist.metadata.get("Name"),
                "path": item["path"],
                "size": item["size"],
                "sha256": item["sha256"],
                "needed": item["needed"],
            }
            for item in native
        ],
    }
    failures: list[str] = []
    sdist_license_valid = bool(
        locked_sdist
        and locked_sdist.get("status") == "PASS"
        and locked_sdist.get("publisher_files")
    )
    if (
        not installed["license"]
        and not installed["license_expression"]
        and not installed["license_classifiers"]
        and not sdist_license_valid
    ):
        failures.append(f"missing package license metadata: {record['identity']}")
    if not publisher and not (locked_sdist and locked_sdist.get("status") == "PASS" and locked_sdist.get("publisher_files")):
        failures.append(f"missing publisher LICENSE/NOTICE evidence: {record['identity']}")
    if locked_sdist and locked_sdist.get("status") == "BLOCKED":
        failures.append(f"locked sdist publisher evidence blocked: {record['identity']}: {locked_sdist['error']}")
    failures.extend(f"unsafe publisher path: {record['identity']}:{path}" for path in unsafe_publisher)
    failures.extend(f"unsafe native path: {record['identity']}:{path}" for path in unsafe_native)
    if any(item["needed"]["inspection"] == "error" for item in native):
        failures.append(f"ELF NEEDED inspection failed: {record['identity']}")
    if duplicate:
        failures.append(f"duplicate installed distribution: {record['identity']}")
    return {"lock": lock_data, "installed": installed}, failures


def _dependency_acquisition(packages: list[dict[str, Any]]) -> dict[str, Any]:
    requests: list[dict[str, Any]] = []
    for package in packages:
        installed = package.get("installed")
        audit = installed.get("locked_sdist_license_audit") if isinstance(installed, dict) else None
        if not isinstance(audit, dict):
            continue
        lock = package.get("lock") if isinstance(package.get("lock"), dict) else {}
        archive = audit.get("archive_identity") if isinstance(audit.get("archive_identity"), dict) else {}
        requests.append({
            "identity": identity(lock.get("name", ""), lock.get("version", "")),
            "package": lock.get("name"),
            "requested_url": archive.get("requested_url"),
            "status": audit.get("status"),
            "purpose": "publisher-license-evidence-only",
        })
    return {
        "policy": "exact locked PyPI sdist only when installed publisher evidence is missing",
        "in_memory_archive_inspection": True,
        "requests": sorted(requests, key=lambda item: (item["identity"], item["requested_url"] or "")),
        "out_of_scope_requests": [],
        "model_files": [],
    }


def _license_url(item: dict[str, Any]) -> str:
    if item["kind"] == "github":
        return f"https://raw.githubusercontent.com/{item['repo']}/{item['revision']}/LICENSE"
    return f"https://huggingface.co/{item['repo']}/raw/{item['revision']}/LICENSE"


def _allowed_license_urls(item: dict[str, Any]) -> set[tuple[str, str, str]]:
    """Return exact (host, path, scheme) tuples bound to one fixed identity."""
    raw = urlsplit(_license_url(item))
    allowed = {(raw.hostname or "", raw.path, raw.scheme)}
    if item["kind"] == "huggingface":
        path = f"/{item['repo']}/resolve/{item['revision']}/LICENSE"
        allowed.update(
            (host, path, "https")
            for host in HF_LICENSE_HOSTS
            if host.startswith("cdn-lfs")
        )
    return allowed


def _validate_license_url(item: dict[str, Any], url: str, *, initial: bool = False) -> None:
    parsed = urlsplit(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise AuditError(f"LICENSE URL has an invalid port: {url}") from exc
    if parsed.scheme != "https" or port not in (None, 443):
        raise AuditError(f"non-allow-listed LICENSE host or scheme: {url}")
    if parsed.username is not None or parsed.password is not None:
        raise AuditError(f"LICENSE URL contains userinfo: {url}")
    if parsed.query or parsed.fragment:
        raise AuditError(f"LICENSE URL contains query or fragment: {url}")
    expected = _license_url(item)
    if initial and url != expected:
        raise AuditError(f"initial LICENSE URL is not the generated fixed URL: {url}")
    if (parsed.hostname, parsed.path, parsed.scheme) not in _allowed_license_urls(item):
        raise AuditError(f"LICENSE URL is not the exact pinned identity path: {url}")


class _LicenseRedirects(HTTPRedirectHandler):
    def __init__(self, item: dict[str, Any], trace: list[str]) -> None:
        super().__init__()
        self.item = item
        self.trace = trace

    def redirect_request(self, request: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Request | None:
        _validate_license_url(self.item, newurl)
        self.trace.append(newurl)
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def _fetch_license(
    item: dict[str, Any], fetcher: Callable[[str], tuple[str, bytes]] | None = None
) -> dict[str, Any]:
    requested_url = _license_url(item)
    _validate_license_url(item, requested_url, initial=True)
    trace = [requested_url]
    if fetcher is None:
        opener = build_opener(_LicenseRedirects(item, trace))
        request = Request(requested_url, headers={"Accept": "text/plain", "User-Agent": "vokra-parler-tts-audit/1"})
        try:
            with opener.open(request, timeout=30) as response:  # noqa: S310 - exact host/path is validated
                final_url = response.geturl()
                _validate_license_url(item, final_url)
                if final_url not in trace:
                    trace.append(final_url)
                content_length = response.headers.get("Content-Length")
                if content_length and content_length.isdigit() and int(content_length) > MAX_LICENSE_BYTES:
                    raise AuditError("upstream LICENSE Content-Length exceeds the bounded audit size")
                body = response.read(MAX_LICENSE_BYTES + 1)
        except AuditError:
            raise
        except HTTPError as exc:
            raise LicensePathError(
                f"exact LICENSE path is not retrievable: HTTP {exc.code}",
                status_code=exc.code,
            ) from exc
        except Exception as exc:  # noqa: BLE001 - network failures become factual blockers
            raise LicensePathError(f"exact LICENSE path is not retrievable: {exc}") from exc
    else:
        final_url, body = fetcher(requested_url)
        _validate_license_url(item, final_url)
        trace.append(final_url)
    if not isinstance(body, bytes):
        raise AuditError("upstream LICENSE response was not bytes")
    if len(body) > MAX_LICENSE_BYTES:
        raise AuditError("upstream LICENSE response exceeds the bounded audit size")
    return {
        "id": item["id"],
        "kind": item["kind"],
        "repo": item["repo"],
        "revision": item["revision"],
        "claimed_license": item["claimed_license"],
        "requested_url": requested_url,
        "final_url": final_url,
        "url_trace": trace,
        "acquired_file": "LICENSE",
        "size": len(body),
        "sha256": sha256_bytes(body),
        "content_base64": base64.b64encode(body).decode("ascii"),
        "license_classification": "UNCLASSIFIED_PRIMARY_SOURCE_BYTES_ONLY",
    }


def _model_info_url(item: dict[str, Any]) -> str:
    return f"https://huggingface.co/api/models/{item['repo']}?revision={item['revision']}"


def _model_info_entry(item: dict[str, Any]) -> dict[str, Any] | None:
    """Return the narrow metadata fallback contract for one fixed model.

    The fallback is intentionally unavailable for source and DAC identities:
    their license evidence must remain an authenticated LICENSE/provenance
    path, rather than treating an unrelated API card as a weight license.
    """
    if item.get("kind") != "huggingface":
        return None
    fallback = preflight_gate.MODEL_METADATA_FALLBACK
    for entry in fallback["entries"]:
        if entry["repo"] == item.get("repo") and entry["revision"] == item.get("revision"):
            return entry
    return None


def _validate_model_info_url(item: dict[str, Any], url: str, *, initial: bool = False) -> None:
    expected = _model_info_url(item)
    parsed = urlsplit(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise AuditError(f"HF model-info URL has an invalid port: {url}") from exc
    if parsed.scheme != "https" or parsed.hostname not in HF_MODEL_INFO_HOSTS or port not in (None, 443):
        raise AuditError(f"non-allow-listed HF model-info host or scheme: {url}")
    if parsed.username is not None or parsed.password is not None:
        raise AuditError(f"HF model-info URL contains userinfo: {url}")
    if parsed.fragment:
        raise AuditError(f"HF model-info URL contains a fragment: {url}")
    expected_parts = urlsplit(expected)
    if initial and url != expected:
        raise AuditError(f"initial HF model-info URL is not the generated fixed URL: {url}")
    if parsed.path != expected_parts.path:
        raise AuditError(f"HF model-info URL changed the exact repository path: {url}")
    if parse_qsl(parsed.query, keep_blank_values=True) != [("revision", item["revision"])] or parsed.query != expected_parts.query:
        raise AuditError(f"HF model-info URL does not carry the exact revision query: {url}")


class _ModelInfoRedirects(HTTPRedirectHandler):
    def __init__(self, item: dict[str, Any], trace: list[str]) -> None:
        super().__init__()
        self.item = item
        self.trace = trace

    def redirect_request(self, request: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Request | None:
        if len(self.trace) - 1 >= MAX_MODEL_INFO_REDIRECTS:
            raise AuditError("HF model-info redirect chain exceeds the bounded limit")
        _validate_model_info_url(self.item, newurl)
        self.trace.append(newurl)
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def _validate_model_info_payload(payload: Any, expected: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    if not isinstance(payload, dict) or not MODEL_INFO_REQUIRED_KEYS.issubset(payload):
        raise AuditError("HF model-info response is missing required schema fields")
    if payload.get("id") != expected["repo"] or payload.get("sha") != expected["sha"]:
        raise AuditError("HF model-info response identity does not match the pinned repo/revision")
    if payload.get("private") is not False or payload.get("gated") is not False or payload.get("disabled") is not False:
        raise AuditError("HF model-info response is not public, non-gated, and enabled")
    card_data = payload.get("cardData")
    if not isinstance(card_data, dict) or card_data.get("license") != expected["card_data_license"]:
        raise AuditError("HF model-info cardData.license does not match the pinned contract")
    siblings = payload.get("siblings")
    if not isinstance(siblings, list) or not siblings:
        raise AuditError("HF model-info response has no exact repository tree")
    names: list[str] = []
    for sibling in siblings:
        if not isinstance(sibling, dict) or set(sibling) != {"rfilename"} or not isinstance(sibling["rfilename"], str):
            raise AuditError("HF model-info sibling schema is not the unexpanded exact tree shape")
        name = sibling["rfilename"]
        if not name or "\x00" in name or name.startswith("/") or "\\" in name or any(part in {"", ".", ".."} for part in name.split("/")):
            raise AuditError(f"HF model-info tree contains an unsafe filename: {name!r}")
        names.append(name)
    if len(set(names)) != len(names):
        raise AuditError("HF model-info tree contains duplicate filenames")
    license_files = sorted(
        name for name in names
        if Path(name).name.casefold() in LICENSE_FILE_NAMES
        or any(token in Path(name).name.casefold() for token in ("license", "licence", "copying", "notice", "copyright"))
    )
    if license_files:
        raise AuditError("HF model-info exact tree contains license-like files; metadata fallback is not applicable")
    return sorted(names), card_data


def _fetch_model_info(
    item: dict[str, Any], fetcher: Callable[[str], tuple[str, bytes]] | None = None
) -> dict[str, Any]:
    expected = _model_info_entry(item)
    if expected is None:
        raise AuditError("HF model-info fallback is not allow-listed for this identity")
    requested_url = _model_info_url(item)
    _validate_model_info_url(item, requested_url, initial=True)
    trace = [requested_url]
    if fetcher is None:
        opener = build_opener(_ModelInfoRedirects(item, trace))
        request = Request(requested_url, headers={"Accept": "application/json", "User-Agent": "vokra-parler-tts-audit/1"})
        try:
            with opener.open(request, timeout=30) as response:  # noqa: S310 - exact host/path/query are validated
                final_url = response.geturl()
                _validate_model_info_url(item, final_url)
                if final_url not in trace:
                    trace.append(final_url)
                content_length = response.headers.get("Content-Length")
                if content_length and content_length.isdigit() and int(content_length) > MAX_MODEL_INFO_BYTES:
                    raise AuditError("HF model-info Content-Length exceeds the bounded response size")
                body = response.read(MAX_MODEL_INFO_BYTES + 1)
        except AuditError:
            raise
        except HTTPError as exc:
            raise AuditError(f"HF model-info endpoint is not retrievable: HTTP {exc.code}") from exc
        except Exception as exc:  # noqa: BLE001 - network failures become factual blockers
            raise AuditError(f"HF model-info endpoint is not retrievable: {exc}") from exc
    else:
        final_url, body = fetcher(requested_url)
        _validate_model_info_url(item, final_url)
        if final_url not in trace:
            trace.append(final_url)
    if not isinstance(body, bytes) or len(body) > MAX_MODEL_INFO_BYTES:
        raise AuditError("HF model-info response is not bounded bytes")
    try:
        payload = json.loads(body.decode("utf-8"), object_pairs_hook=reject_duplicate_json_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AuditError(f"HF model-info response is not strict JSON: {exc}") from exc
    names, card_data = _validate_model_info_payload(payload, expected)
    return {
        "schema": MODEL_METADATA_SCHEMA,
        "repo": expected["repo"],
        "revision": expected["revision"],
        "sha": payload["sha"],
        "requested_url": requested_url,
        "final_url": final_url,
        "url_trace": trace,
        "response_bytes": len(body),
        "response_sha256": sha256_bytes(body),
        "id": payload["id"],
        "private": payload["private"],
        "gated": payload["gated"],
        "disabled": payload["disabled"],
        "card_data_license": card_data["license"],
        "tree": {
            "file_count": len(names),
            "files": names,
            "license_files": [],
            "files_sha256": sha256_bytes(canonical_json(names).encode()),
        },
        "license_classification": "AUTHENTICATED_HF_CARD_DATA_ONLY",
    }


def _fixed_license_items(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    source = manifest["source_identity"]
    variants = manifest["variants"]
    dac = manifest["dac_identity"]
    source_repo = source["repo"].removeprefix("https://github.com/").removesuffix(".git")
    return [
        {
            "id": f"parler-tts-source@{source['revision']}",
            "kind": "github",
            "repo": source_repo,
            "revision": source["revision"],
            "claimed_license": source["license"],
        },
        *[
            {
                "id": f"{variant['upstream_repo']}@{variant['upstream_revision']}",
                "kind": "huggingface",
                "repo": variant["upstream_repo"],
                "revision": variant["upstream_revision"],
                "claimed_license": variant["upstream_license"],
            }
            for variant in variants
        ],
        {
            "id": f"{dac['repo']}@{dac['revision']}",
            "kind": "huggingface",
            "repo": dac["repo"],
            "revision": dac["revision"],
            "claimed_license": dac["license"],
        },
    ]


def audit_model_licenses(
    manifest: dict[str, Any], fetcher: Callable[[str], tuple[str, bytes]] | None = None,
    metadata_fetcher: Callable[[str], tuple[str, bytes]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    files: list[dict[str, Any]] = []
    failures: list[str] = []
    for item in _fixed_license_items(manifest):
        try:
            files.append(_fetch_license(item, fetcher))
        except LicensePathError as exc:
            fallback = _model_info_entry(item)
            if exc.status_code == 404 and fallback is not None:
                try:
                    metadata = _fetch_model_info(item, metadata_fetcher)
                except AuditError as metadata_exc:
                    files.append({
                        "id": item["id"],
                        "kind": item["kind"],
                        "repo": item["repo"],
                        "revision": item["revision"],
                        "claimed_license": item["claimed_license"],
                        "requested_url": _license_url(item),
                        "final_url": None,
                        "acquired_file": None,
                        "status": "BLOCKED_FACTUAL_LICENSE_PATH",
                        "error": str(exc),
                        "metadata_fallback": {"status": "BLOCKED", "error": str(metadata_exc)},
                    })
                    failures.append(f"{item['id']}: {exc}; HF model-info fallback: {metadata_exc}")
                else:
                    files.append({
                        "id": item["id"],
                        "kind": item["kind"],
                        "repo": item["repo"],
                        "revision": item["revision"],
                        "claimed_license": item["claimed_license"],
                        "requested_url": _license_url(item),
                        "final_url": None,
                        "acquired_file": None,
                        "status": "PASS_METADATA_FALLBACK",
                        "error": str(exc),
                        "metadata_fallback": metadata,
                    })
                continue
            files.append({
                "id": item["id"],
                "kind": item["kind"],
                "repo": item["repo"],
                "revision": item["revision"],
                "claimed_license": item["claimed_license"],
                "requested_url": _license_url(item),
                "final_url": None,
                "acquired_file": None,
                "status": "BLOCKED_FACTUAL_LICENSE_PATH",
                "error": str(exc),
            })
            failures.append(f"{item['id']}: {exc}")
        except AuditError as exc:
            files.append({
                "id": item["id"],
                "kind": item["kind"],
                "repo": item["repo"],
                "revision": item["revision"],
                "claimed_license": item["claimed_license"],
                "requested_url": _license_url(item),
                "final_url": None,
                "acquired_file": None,
                "status": "BLOCKED_FACTUAL_LICENSE_PATH",
                "error": str(exc),
            })
            failures.append(f"{item['id']}: {exc}")
    return files, failures


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
        raise AuditError(f"git commit identity unavailable: {exc}") from exc
    return {"root": str(repository), "commit": commit, "audit_script_sha256": sha256_file(Path(__file__).resolve())}


def audit_environment(
    project: Path,
    fetch_model_licenses: bool = True,
    sdist_fetcher: Callable[[str], tuple[str, bytes]] | None = None,
) -> dict[str, Any]:
    project_data, lock, manifest, project_bytes, lock_bytes = _contract(project)
    expected_rows = _expected_packages(lock)
    active_rows, inactive_rows = classify_lock_rows(lock)
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
        package, package_failures = _inspect_package(
            row, candidates[0] if len(candidates) == 1 else None, len(candidates) > 1, sdist_fetcher
        )
        packages.append(package)
        failures.extend(package_failures)
    if not closure["exact"]:
        failures.append("installed normalized name+version multiset does not exactly match uv.lock")
    if sys.version_info[:2] != (3, 12):
        failures.append(f"Python runtime is not 3.12: {platform.python_version()}")
    if sys.platform != "linux" or platform.machine().casefold() not in {"x86_64", "amd64"}:
        failures.append(f"audit host is not Linux x86_64: {sys.platform}/{platform.machine()}")
    model_license_files, license_failures = audit_model_licenses(manifest) if fetch_model_licenses else ([], [])
    failures.extend(license_failures)
    return {
        "schema": SCHEMA,
        "status": "BLOCKED" if failures else "PASS",
        "repository": _repository_identity(project),
        "environment": {
            "python": platform.python_version(),
            "platform": sys.platform,
            "machine": platform.machine(),
            "readelf_required": True,
            "model_code_imported": False,
            "cargo_invoked": False,
        },
        "project": {
            "name": project_data["project"]["name"],
            "version": project_data["project"]["version"],
            "pyproject_bytes": len(project_bytes),
            "pyproject_sha256": sha256_bytes(project_bytes),
            "uv_lock_bytes": len(lock_bytes),
            "uv_lock_sha256": sha256_bytes(lock_bytes),
        },
        "lock_rows": {
            "accounted_rows": len(lock["package"]),
            "active_linux_installed": active_rows,
            "inactive_or_virtual": inactive_rows,
            "all_rows_accounted": len(active_rows) + len(inactive_rows) == len(lock["package"]),
        },
        "closure": closure,
        "packages": packages,
        "dependency_acquisition": _dependency_acquisition(packages),
        "fixed_source_model_dac_identities": _fixed_license_items(manifest),
        "model_license_files": model_license_files,
        "model_acquisition": {
            "scope": "fixed source/model/DAC LICENSE paths plus exact HF model-info metadata fallback",
            "policy": "allow-listed exact primary-source LICENSE fetch, with model-only 404 fallback to authenticated HF API metadata",
            "requested_files": [item["requested_url"] for item in model_license_files],
            "metadata_requests": [
                item["metadata_fallback"]["requested_url"]
                for item in model_license_files
                if isinstance(item.get("metadata_fallback"), dict)
                and isinstance(item["metadata_fallback"].get("requested_url"), str)
            ],
            "non_license_requests": [],
            "non_license_files": [],
            "proof": "audit code has no model-weight acquisition path and imports no model/Torch code; HF metadata responses are bounded JSON only",
        },
        "failures": sorted(set(failures)),
    }


def run(project: Path, output: Path, fetch_model_licenses: bool) -> int:
    try:
        report = audit_environment(project, fetch_model_licenses)
    except (AuditError, OSError, UnicodeError, ValueError) as exc:
        report = {
            "schema": SCHEMA,
            "status": "BLOCKED",
            "environment": {"model_code_imported": False, "cargo_invoked": False},
            "model_acquisition": {"requested_files": [], "metadata_requests": [], "non_license_requests": [], "non_license_files": []},
            "failures": [str(exc)],
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    if report["failures"]:
        print("parler dependency audit: BLOCKED: " + "; ".join(report["failures"]), file=sys.stderr)
        return 2
    print(f"parler dependency audit: PASS ({output})")
    return 0


def self_test() -> int:
    expected = ["foo-bar==1.0", "torch==2.11.0+cpu"]
    exact = compare_multiset(expected, ["torch==2.11.0+cpu", "foo-bar==1.0"])
    assert exact["exact"] and not exact["missing"] and not exact["unexpected"]
    assert identity("foo_bar", "1.0") == "foo-bar==1.0"
    assert identity("torch", "2.11.0+CPU") == "torch==2.11.0+cpu"
    mismatch = compare_multiset(expected, ["foo-bar==1.0", "torch==2.10.0+cpu"])
    assert mismatch["missing"] == ["torch==2.11.0+cpu"]
    assert mismatch["unexpected"] == ["torch==2.10.0+cpu"]
    duplicate = compare_multiset(["foo-bar==1.0"], ["foo-bar==1.0", "foo-bar==1.0"])
    assert duplicate["duplicate_identities"] == ["foo-bar==1.0"]
    assert duplicate["unexpected"] == ["foo-bar==1.0"]
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    assert all(_is_license_path(path) for path in ("LICENCE", "licence.txt", "pkg/License.md"))
    assert not _is_license_path("pkg/README.md")

    manifest = load_json(Path(__file__).resolve().parent / "license_gate_manifest.json")
    items = _fixed_license_items(manifest)
    assert len(items) == 4
    assert all(_license_url(item).endswith("/LICENSE") for item in items)
    body = b"primary source bytes\n"
    result = _fetch_license(items[0], lambda url: (url, body))
    assert result["requested_url"] == result["final_url"]
    assert result["size"] == len(body) and result["sha256"] == sha256_bytes(body)
    assert result["license_classification"] == "UNCLASSIFIED_PRIMARY_SOURCE_BYTES_ONLY"
    hf_raw = _license_url(items[1])
    hf_cdn = f"https://cdn-lfs.huggingface.co/{items[1]['repo']}/resolve/{items[1]['revision']}/LICENSE"
    assert _fetch_license(items[1], lambda _url: (hf_cdn, body))["final_url"] == hf_cdn
    hf_port_443 = hf_raw.replace("https://huggingface.co/", "https://huggingface.co:443/")
    assert _fetch_license(items[1], lambda _url: (hf_port_443, body))["final_url"] == hf_port_443
    unsafe_urls = (
        hf_raw.replace("/raw/", "/raw/other-repo/", 1),
        hf_raw.replace(items[1]["revision"], "0" * 40, 1),
        hf_raw.replace("https://", "https://audit-user@", 1),
        hf_raw.replace("https://huggingface.co/", "https://huggingface.co:8443/", 1),
        hf_raw + "?download=true",
        hf_raw + "#fragment",
        hf_raw.replace("/LICENSE", "/LICENSE.txt", 1),
        hf_raw.replace("/LICENSE", "/model.safetensors", 1),
    )
    for bad in unsafe_urls:
        try:
            _fetch_license(items[1], lambda _url, bad=bad: (bad, body))
        except AuditError:
            pass
        else:
            raise AssertionError(f"accepted unsafe LICENSE redirect: {bad}")
    try:
        _validate_license_url(items[1], hf_raw.replace(items[1]["repo"], "nearby/repo", 1), initial=True)
    except AuditError:
        pass
    else:
        raise AssertionError("accepted a non-fixed initial LICENSE URL")
    source_raw = _license_url(items[0])
    for bad in (
        source_raw.replace("huggingface/parler-tts", "nearby/repo", 1),
        source_raw.replace(items[0]["revision"], "0" * 40, 1),
        source_raw.replace("raw.githubusercontent.com", "github.com", 1),
    ):
        try:
            _fetch_license(items[0], lambda _url, bad=bad: (bad, body))
        except AuditError:
            pass
        else:
            raise AssertionError(f"accepted unsafe source LICENSE redirect: {bad}")
    try:
        _fetch_license(items[0], lambda url: (url, b"x" * (MAX_LICENSE_BYTES + 1)))
    except AuditError:
        pass
    else:
        raise AssertionError("accepted an over-sized LICENSE response")

    # A model LICENSE 404 may use only the exact, bounded HF model-info
    # response.  Source and DAC identities are deliberately not eligible for
    # this fallback, so a missing DAC LICENSE remains a factual blocker.
    model_item = items[1]
    expected_model = _model_info_entry(model_item)
    assert expected_model is not None
    model_payload = {
        "id": expected_model["repo"],
        "sha": expected_model["sha"],
        "private": False,
        "gated": False,
        "disabled": False,
        "cardData": {"license": "apache-2.0"},
        "siblings": [{"rfilename": "config.json"}, {"rfilename": "model.safetensors"}],
    }
    model_body = canonical_json(model_payload).encode()
    model_result = _fetch_model_info(model_item, lambda url: (url, model_body))
    assert model_result["schema"] == MODEL_METADATA_SCHEMA
    assert model_result["repo"] == model_item["repo"] and model_result["sha"] == model_item["revision"]
    assert model_result["tree"]["license_files"] == []
    assert model_result["license_classification"] == "AUTHENTICATED_HF_CARD_DATA_ONLY"
    short_host = _model_info_url(model_item).replace("https://huggingface.co", "https://hf.co", 1)
    assert _fetch_model_info(model_item, lambda _url: (short_host, model_body))["final_url"] == short_host
    for unsafe in (
        _model_info_url(model_item).replace("huggingface.co", "evil.example", 1),
        _model_info_url(model_item).replace(model_item["revision"], "0" * 40, 1),
        _model_info_url(model_item) + "&extra=1",
        _model_info_url(model_item).replace("https://", "https://audit-user@", 1),
        _model_info_url(model_item) + "#fragment",
    ):
        try:
            _validate_model_info_url(model_item, unsafe)
        except AuditError:
            pass
        else:
            raise AssertionError(f"accepted unsafe HF model-info URL: {unsafe}")
    try:
        _fetch_model_info(model_item, lambda url: (url, b"{}"))
    except AuditError:
        pass
    else:
        raise AssertionError("accepted an incomplete HF model-info schema")
    disabled_model = dict(model_payload)
    disabled_model["disabled"] = True
    try:
        _fetch_model_info(model_item, lambda url: (url, canonical_json(disabled_model).encode()))
    except AuditError:
        pass
    else:
        raise AssertionError("accepted a disabled HF model-info response")
    licensed_tree = dict(model_payload)
    licensed_tree["siblings"] = [{"rfilename": "LICENSE"}]
    try:
        _fetch_model_info(model_item, lambda url: (url, canonical_json(licensed_tree).encode()))
    except AuditError:
        pass
    else:
        raise AssertionError("accepted an HF model-info tree containing LICENSE")
    try:
        _fetch_model_info(model_item, lambda url: (url, b"x" * (MAX_MODEL_INFO_BYTES + 1)))
    except AuditError:
        pass
    else:
        raise AssertionError("accepted an over-sized HF model-info response")

    def missing_model_license(url: str) -> tuple[str, bytes]:
        if "huggingface.co/" in url:
            raise LicensePathError("exact LICENSE path is not retrievable: HTTP 404", status_code=404)
        return url, body

    def model_info_fetcher(url: str) -> tuple[str, bytes]:
        matching = next(entry for entry in preflight_gate.MODEL_METADATA_FALLBACK["entries"] if entry["repo"] in url)
        payload = {
            "id": matching["repo"], "sha": matching["sha"], "private": False, "gated": False, "disabled": False,
            "cardData": {"license": matching["card_data_license"]},
            "siblings": [{"rfilename": "config.json"}, {"rfilename": "model.safetensors"}],
        }
        return url, canonical_json(payload).encode()

    fallback_files, fallback_failures = audit_model_licenses(manifest, missing_model_license, model_info_fetcher)
    assert [item.get("status", "PASS") for item in fallback_files] == [
        "PASS", "PASS_METADATA_FALLBACK", "PASS_METADATA_FALLBACK", "BLOCKED_FACTUAL_LICENSE_PATH"
    ]
    assert len(fallback_failures) == 1 and "dac_44khZ_8kbps" in fallback_failures[0]

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

    def synthetic_zip(entries: list[tuple[str, bytes]]) -> bytes:
        output = io.BytesIO()
        with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, payload in entries:
                archive.writestr(name, payload)
        return output.getvalue()

    def synthetic_zip_special(name: str, file_type: int) -> bytes:
        output = io.BytesIO()
        with zipfile.ZipFile(output, mode="w") as archive:
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = (file_type | 0o644) << 16
            archive.writestr(info, b"special")
        return output.getvalue()

    def synthetic_row(body: bytes, suffix: str = ".tar.gz") -> tuple[dict[str, Any], dict[str, Any]]:
        url = f"https://files.pythonhosted.org/packages/self-test/demo-1{suffix}"
        artifact = {
            "url": url,
            "hash": "sha256:" + sha256_bytes(body),
            "size": len(body),
            "upload-time": "2026-01-01T00:00:00Z",
        }
        return {"name": "demo", "version": "1", "source": {"registry": "https://pypi.org/simple"}, "sdist": artifact}, artifact

    good_archive = synthetic_tar([
        ("demo-1/", b"", "dir"),
        ("demo-1/LICENSE", b"demo license\n", "file"),
    ])
    good_row, good_artifact = synthetic_row(good_archive)
    good_sdist = _fetch_locked_sdist(good_row, lambda url: (url, good_archive))
    assert good_sdist["status"] == "PASS"
    assert good_sdist["publisher_files"][0]["path"] == "demo-1/LICENSE"
    assert good_sdist["publisher_files"][0]["content_base64"] == base64.b64encode(b"demo license\n").decode("ascii")
    assert good_sdist["archive_identity"]["sha256"] == good_artifact["hash"]
    assert good_sdist["archive_identity"]["url_trace"] == [good_artifact["url"]]
    british_archive = synthetic_tar([
        ("demo-1/", b"", "dir"),
        ("demo-1/LICENCE", b"british spelling\n", "file"),
    ])
    british_row, _ = synthetic_row(british_archive)
    british_sdist = _fetch_locked_sdist(british_row, lambda url: (url, british_archive))
    assert british_sdist["status"] == "PASS"
    assert british_sdist["publisher_files"][0]["path"] == "demo-1/LICENCE"
    for label, mutate in (
        ("hash", lambda artifact: artifact.update(hash="sha256:" + "0" * 64)),
        ("size", lambda artifact: artifact.update(size=artifact["size"] + 1)),
        ("host", lambda artifact: artifact.update(url=artifact["url"].replace("files.pythonhosted.org", "example.invalid"))),
    ):
        candidate = {**good_row, "sdist": dict(good_artifact)}
        mutate(candidate["sdist"])
        try:
            _fetch_locked_sdist(candidate, lambda url: (good_artifact["url"], good_archive))
        except AuditError:
            pass
        else:
            raise AssertionError(f"accepted tampered locked sdist {label}")
    try:
        _fetch_locked_sdist(good_row, lambda _url: (good_artifact["url"] + "?download=1", good_archive))
    except AuditError:
        pass
    else:
        raise AssertionError("accepted a non-exact locked sdist redirect")
    redirect_at_bound_trace = [good_artifact["url"]] * MAX_SDIST_REDIRECTS
    redirect_at_bound = _SdistRedirects(good_artifact, redirect_at_bound_trace)
    try:
        redirect_at_bound.redirect_request(Request(good_artifact["url"]), None, 302, "found", {}, good_artifact["url"])
    except AuditError:
        raise AssertionError("rejected the fourth locked sdist redirect")
    assert len(redirect_at_bound_trace) == MAX_SDIST_REDIRECTS + 1
    redirect_guard = _SdistRedirects(good_artifact, [good_artifact["url"]] * (MAX_SDIST_REDIRECTS + 1))
    try:
        redirect_guard.redirect_request(Request(good_artifact["url"]), None, 302, "found", {}, good_artifact["url"])
    except AuditError:
        pass
    else:
        raise AssertionError("accepted an overlong locked sdist redirect chain")
    for bad_url in (
        good_artifact["url"].replace("https://", "https://audit-user@", 1),
        good_artifact["url"].replace("https://files.pythonhosted.org/", "https://files.pythonhosted.org:8443/", 1),
        good_artifact["url"] + "#fragment",
        good_artifact["url"] + "?download=1",
    ):
        try:
            _fetch_locked_sdist({**good_row, "sdist": {**good_artifact, "url": bad_url}}, lambda url: (url, good_archive))
        except AuditError:
            pass
        else:
            raise AssertionError(f"accepted unsafe locked sdist URL: {bad_url}")
    for label, archive_body in (
        ("traversal", synthetic_tar([("../LICENSE", b"bad", "file")])),
        ("absolute", synthetic_tar([("/LICENSE", b"bad", "file")])),
        ("backslash", synthetic_tar([("demo-1\\LICENSE", b"bad", "file")])),
        ("empty-component", synthetic_tar([("demo-1//LICENSE", b"bad", "file")])),
        ("dot-component", synthetic_tar([("demo-1/./LICENSE", b"bad", "file")])),
        ("link", synthetic_tar([("demo-1/LICENSE", b"", "symlink")])),
        ("special", synthetic_tar([("demo-1/device", b"", "fifo")])),
        ("duplicate", synthetic_tar([("demo-1/LICENSE", b"one", "file"), ("demo-1/LICENSE", b"two", "file")])),
        ("no-license", synthetic_tar([("demo-1/README", b"not a license", "file")])),
    ):
        row, _ = synthetic_row(archive_body)
        try:
            _fetch_locked_sdist(row, lambda url, payload=archive_body: (url, payload))
        except AuditError:
            pass
        else:
            raise AssertionError(f"accepted unsafe/no-license sdist archive: {label}")
    for special_name in ("demo-1/socket", "demo-1/special/"):
        special_zip = synthetic_zip_special(special_name, stat.S_IFSOCK)
        row, _ = synthetic_row(special_zip, ".zip")
        try:
            _fetch_locked_sdist(row, lambda url, payload=special_zip: (url, payload))
        except AuditError:
            pass
        else:
            raise AssertionError("accepted a ZIP special member")
    many_members = synthetic_tar([(f"demo-1/file-{index}", b"", "file") for index in range(MAX_ARCHIVE_MEMBERS + 1)])
    row, _ = synthetic_row(many_members)
    try:
        _fetch_locked_sdist(row, lambda url: (url, many_members))
    except AuditError:
        pass
    else:
        raise AssertionError("accepted an archive with too many members")
    huge_member = synthetic_tar([("demo-1/LICENSE", b"x" * (MAX_ARCHIVE_MEMBER_BYTES + 1), "file")])
    row, _ = synthetic_row(huge_member)
    try:
        _fetch_locked_sdist(row, lambda url: (url, huge_member))
    except AuditError:
        pass
    else:
        raise AssertionError("accepted an oversized sdist member")
    aggregate = synthetic_zip([(f"demo-1/file-{index}", b"x" * MAX_ARCHIVE_MEMBER_BYTES) for index in range(9)])
    row, _ = synthetic_row(aggregate, ".zip")
    try:
        _fetch_locked_sdist(row, lambda url: (url, aggregate))
    except AuditError:
        pass
    else:
        raise AssertionError("accepted an oversized sdist aggregate")
    unsupported, _ = synthetic_row(good_archive, ".rar")
    try:
        _fetch_locked_sdist(unsupported, lambda url: (url, good_archive))
    except AuditError:
        pass
    else:
        raise AssertionError("accepted an unsupported sdist archive format")

    class EmptyPublisherDistribution:
        files: list[str] = []
        metadata = Message()
        metadata["Name"] = "demo"
        metadata["Version"] = "1"
        version = "1"

        def locate_file(self, entry: Any) -> Path:
            return Path(entry)

    package, package_failures = _inspect_package(
        good_row,
        {"distribution": EmptyPublisherDistribution(), "identity": "demo==1", "location": "self-test"},
        False,
        lambda url: (url, good_archive),
    )
    assert package["installed"]["locked_sdist_license_audit"]["status"] == "PASS"
    assert not any("missing publisher LICENSE/NOTICE evidence" in failure for failure in package_failures)
    assert not any("missing package license metadata" in failure for failure in package_failures)
    blocked_row, _ = synthetic_row(good_archive)
    blocked_row["sdist"]["hash"] = "sha256:" + "0" * 64
    blocked_package, blocked_failures = _inspect_package(
        blocked_row,
        {"distribution": EmptyPublisherDistribution(), "identity": "demo==1", "location": "self-test"},
        False,
        lambda url: (url, good_archive),
    )
    assert blocked_package["installed"]["locked_sdist_license_audit"]["status"] == "BLOCKED"
    assert any("locked sdist publisher evidence blocked" in failure for failure in blocked_failures)
    assert any("missing package license metadata" in failure for failure in blocked_failures)
    blocked_package["lock"]["name"] = "demo-blocked"
    dependency_acquisition = _dependency_acquisition([package, blocked_package])
    assert dependency_acquisition["policy"] == "exact locked PyPI sdist only when installed publisher evidence is missing"
    assert dependency_acquisition["in_memory_archive_inspection"] is True
    assert [item["status"] for item in dependency_acquisition["requests"]] == ["BLOCKED", "PASS"]
    assert all(item["requested_url"] == good_artifact["url"] for item in dependency_acquisition["requests"])
    assert dependency_acquisition["out_of_scope_requests"] == []
    assert dependency_acquisition["model_files"] == []
    with tempfile.TemporaryDirectory(prefix="parler-audit-report-self-test-") as directory:
        report_path = Path(directory) / "blocked.json"
        assert run(Path(directory) / "not-a-project", report_path, False) == 2
        persisted = load_json(report_path)
        assert persisted["status"] == "BLOCKED" and persisted["failures"]

    with tempfile.TemporaryDirectory(prefix="parler-audit-self-test-") as directory:
        root = Path(directory)
        path = root / "short"
        path.write_bytes(b"not an ELF")
        assert _elf_needed(path)["format"] == "non-elf"
        native = root / "native.so"
        native.write_bytes(b"native")

        class FakeDistribution:
            files = ["native.so", "../escaped.so"]

            def locate_file(self, entry: Any) -> Path:
                return root if str(entry) == "" else root / str(entry)

        original_first_four = _first_four

        def fail_magic_read(_path: Path) -> bytes:
            raise OSError("self-test magic read failure")

        try:
            globals()["_first_four"] = fail_magic_read
            _, unsafe = _native_files(FakeDistribution())
        finally:
            globals()["_first_four"] = original_first_four
        assert any(item.startswith("../escaped.so:out-of-distribution-root") for item in unsafe)
        assert any(item.startswith("native.so:magic-read-failed:") for item in unsafe)
        assert canonical_json(json.loads(canonical_json({"z": 1, "a": 2}))) == '{"a":2,"z":1}'
    print("parler dependency audit: self-test PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fetch-model-licenses", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.project is not None or args.output is not None or args.fetch_model_licenses:
            parser.error("--self-test accepts no project/output/fetch arguments")
        return self_test()
    if args.project is None or args.output is None:
        parser.error("--project and --output are required")
    return run(args.project, args.output, args.fetch_model_licenses)


if __name__ == "__main__":
    raise SystemExit(main())
