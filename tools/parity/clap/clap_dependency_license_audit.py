#!/usr/bin/env python3
"""Inventory the locked CLAP Python environment without loading a model.

This is an owner-independent evidence collector.  It reads the frozen uv lock,
inspects installed distribution metadata, and hashes bundled license and native
payload files.  It never imports Transformers/Torch and never acquires or
opens a checkpoint.  License disposition remains pending even when the
inventory is structurally complete.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import sys
import tarfile
import tempfile
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "vokra-clap-htsat-fused-dependency-license-inventory-v1"
PENDING_STATUS = "PENDING_OWNER_REVIEW"
PENDING_DEPENDENCY_STATUS = "PENDING_VAST_AUDIT"
NO_WEIGHTS = "NOT_ACQUIRED"
NO_UPLOAD = "NO_UPLOAD"
HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
NATIVE_SUFFIXES = (".dylib", ".dll", ".pyd", ".so")
LICENSE_PREFIXES = ("license", "copying", "notice")
PYPI_ARTIFACT_HOST = "files.pythonhosted.org"
MAX_SDIST_BYTES = 32 * 1024 * 1024
MAX_LICENSE_EVIDENCE_BYTES = 4 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 100_000
SDIST_TIMEOUT_SECONDS = 30


def normalize_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cli_path(raw: str, label: str) -> Path:
    """Preserve and validate raw CLI spelling before Path normalization."""

    if not isinstance(raw, str) or not raw.startswith("/") or raw == "/" or "\x00" in raw:
        raise RuntimeError(f"{label} path must be absolute, non-root, and NUL-free")
    parts = raw.split("/")
    if any(part in {"", ".", ".."} for part in parts[1:]):
        raise RuntimeError(f"{label} path contains unsafe lexical components")
    current = Path("/")
    for part in parts[1:]:
        current /= part
        if current.is_symlink() and current != Path("/var"):
            raise RuntimeError(f"{label} path has symlink ancestry: {current}")
    return Path(raw)


def write_atomic_no_replace(path: Path, text: str) -> None:
    require_output_parent(path)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"output already exists: {path}")
    temporary: Path | None = None
    temporary_identity: tuple[int, int] | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            temporary_stat = os.stat(temporary, follow_symlinks=False)
            if not stat.S_ISREG(temporary_stat.st_mode):
                raise RuntimeError(f"temporary output is not regular: {temporary}")
            temporary_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        verify_temporary_identity(temporary, temporary_identity)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RuntimeError(f"output already exists: {path}") from exc
        verify_file_identity(path, temporary_identity, "published output")
    finally:
        cleanup_temporary(temporary, temporary_identity)


def verify_temporary_identity(path: Path, expected: tuple[int, int] | None) -> None:
    verify_file_identity(path, expected, "temporary output")


def verify_file_identity(path: Path, expected: tuple[int, int] | None, label: str) -> None:
    if expected is None or not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError(f"{label} identity cannot be verified safely")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        current = os.fstat(fd)
        if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != expected:
            raise RuntimeError(f"{label} was replaced or is not regular: {path}")
        os.fsync(fd)
    finally:
        os.close(fd)


def cleanup_temporary(path: Path | None, expected: tuple[int, int] | None) -> None:
    if path is None or expected is None:
        return
    try:
        current = os.stat(path, follow_symlinks=False)
        if stat.S_ISREG(current.st_mode) and (current.st_dev, current.st_ino) == expected:
            path.unlink()
    except OSError:
        pass


def require_output_parent(path: Path) -> None:
    """Require an existing, regular, symlink-free output ancestry."""

    if not path.is_absolute() or path.parent == Path("/") or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise RuntimeError(f"output path must be absolute and dot-free: {path}")
    current = path.parent
    while True:
        if current.is_symlink() and current != Path("/var"):
            raise RuntimeError(f"output path has symlink ancestry: {current}")
        parent = current.parent
        if parent == current:
            break
        current = parent
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise RuntimeError(f"output parent must be an existing regular directory: {path.parent}")


def parse_hash(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not HASH_RE.fullmatch(value):
        raise RuntimeError(f"{label} is missing or is not a sha256 hash: {value!r}")
    return value


def validate_pypi_artifact_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != PYPI_ARTIFACT_HOST
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or not parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError(f"locked sdist URL is not an exact PyPI artifact URL: {url!r}")


class _RestrictedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        validate_pypi_artifact_url(newurl)
        old_host = urllib.parse.urlparse(req.full_url).hostname
        if old_host != urllib.parse.urlparse(newurl).hostname:
            raise RuntimeError("locked sdist redirect changed artifact host")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_exact_sdist(
    artifact: dict[str, Any], *, open_url: Any | None = None
) -> bytes:
    """Fetch one lock-pinned sdist, retaining only a bounded byte buffer.

    ``open_url`` exists solely for network-free tests.  Production callers use
    an opener with a redirect handler that permits only files.pythonhosted.org.
    The bytes are verified before any archive parser sees them.
    """

    url = artifact.get("url")
    validate_pypi_artifact_url(url)
    expected_hash = parse_hash(artifact.get("hash"), label="locked sdist hash")
    expected_size = artifact.get("size")
    if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 1:
        raise RuntimeError("locked sdist size is missing or invalid")
    if expected_size > MAX_SDIST_BYTES:
        raise RuntimeError(f"locked sdist exceeds size bound: {expected_size}")
    if open_url is None:
        opener = urllib.request.build_opener(_RestrictedRedirectHandler())
        open_url = opener.open
    request = urllib.request.Request(url, headers={"Accept": "application/octet-stream"})
    try:
        response = open_url(request, timeout=SDIST_TIMEOUT_SECONDS)
        with response:
            response_headers = getattr(response, "headers", {}) or {}
            content_length = response_headers.get("Content-Length")
            if content_length is not None:
                try:
                    if int(content_length) != expected_size:
                        raise RuntimeError("locked sdist Content-Length differs from uv.lock")
                except ValueError as exc:
                    raise RuntimeError("locked sdist Content-Length is invalid") from exc
            digest = hashlib.sha256()
            payload = bytearray()
            while True:
                chunk = response.read(min(1024 * 1024, MAX_SDIST_BYTES + 1 - len(payload)))
                if not chunk:
                    break
                payload.extend(chunk)
                digest.update(chunk)
                if len(payload) > MAX_SDIST_BYTES:
                    raise RuntimeError("locked sdist exceeds in-memory size bound")
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(f"locked sdist fetch failed: {exc}") from exc
    if len(payload) != expected_size:
        raise RuntimeError(
            f"locked sdist size mismatch: got {len(payload)}, expected {expected_size}"
        )
    if f"sha256:{digest.hexdigest()}" != expected_hash:
        raise RuntimeError("locked sdist SHA-256 differs from uv.lock")
    return bytes(payload)


def artifact_record(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a table")
    url = value.get("url")
    if not isinstance(url, str) or not url:
        raise RuntimeError(f"{label} has no URL")
    record: dict[str, Any] = {
        "url": url,
        "hash": parse_hash(value.get("hash"), label=f"{label}.hash"),
    }
    if "size" in value:
        size = value["size"]
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise RuntimeError(f"{label}.size is invalid: {size!r}")
        record["size"] = size
    return record


def parse_lock(lock_path: Path, project_path: Path) -> dict[str, Any]:
    if lock_path.is_symlink() or not lock_path.is_file():
        raise RuntimeError(f"uv.lock is missing or symlinked: {lock_path}")
    if project_path.is_symlink() or not project_path.is_file():
        raise RuntimeError(f"pyproject.toml is missing or symlinked: {project_path}")
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    project = tomllib.loads(project_path.read_text(encoding="utf-8"))
    packages = lock.get("package")
    if not isinstance(packages, list) or not packages:
        raise RuntimeError("uv.lock has no package entries")
    project_name = normalize_name(project["project"]["name"])
    seen: set[str] = set()
    inventory: list[dict[str, Any]] = []
    virtual_packages: list[dict[str, str]] = []
    for index, package in enumerate(packages):
        if not isinstance(package, dict):
            raise RuntimeError(f"uv.lock package {index} is not a table")
        name = package.get("name")
        version = package.get("version")
        source = package.get("source")
        if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
            raise RuntimeError(f"uv.lock package {index} has incomplete name/version")
        normalized = normalize_name(name)
        if normalized in seen:
            raise RuntimeError(f"uv.lock has duplicate package name: {name}")
        seen.add(normalized)
        if not isinstance(source, dict) or not source:
            raise RuntimeError(f"uv.lock {name} has no source table")
        if source.get("virtual") == "." or normalized == project_name:
            virtual_packages.append({"name": name, "version": version})
            continue
        artifacts: dict[str, Any] = {}
        if "sdist" in package:
            artifacts["sdist"] = artifact_record(package["sdist"], label=f"{name}.sdist")
        wheels = package.get("wheels", [])
        if not isinstance(wheels, list):
            raise RuntimeError(f"uv.lock {name}.wheels is not a list")
        artifacts["wheels"] = [
            artifact_record(item, label=f"{name}.wheels[{wheel_index}]")
            for wheel_index, item in enumerate(wheels)
        ]
        if not artifacts.get("sdist") and not artifacts["wheels"]:
            raise RuntimeError(f"uv.lock {name} has no hashed sdist or wheel")
        inventory.append(
            {
                "name": name,
                "normalized_name": normalized,
                "version": version,
                "source": source,
                "artifacts": artifacts,
            }
        )
    if not inventory:
        raise RuntimeError("uv.lock has no non-virtual active packages")
    return {
        "project_name": project["project"]["name"],
        "project_dependencies": sorted(project["project"].get("dependencies", [])),
        "selection": "all non-virtual package entries under the single Linux x86_64 locked resolution",
        "virtual_packages": virtual_packages,
        "requires_python": lock.get("requires-python"),
        "resolution_markers": lock.get("resolution-markers", []),
        "packages": sorted(inventory, key=lambda item: item["normalized_name"]),
    }


def metadata_values(metadata: Any, key: str) -> list[str]:
    values = metadata.get_all(key) if hasattr(metadata, "get_all") else None
    if not values:
        return []
    return [text for value in values if (text := str(value).strip())]


def regular_file_record(path: Path, *, relative: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"bundled file is missing or symlinked: {relative}")
    return {
        "path": relative,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def safe_dist_path(dist: Any, file: Any, *, prefix: Path | None = None) -> Path:
    raw = Path(dist.locate_file(file))
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    prefix_path = Path(sys.prefix) if prefix is None else prefix
    prefix_absolute = prefix_path.absolute()
    prefix_real = prefix_path.resolve()
    # Normalize RECORD's lexical ``..`` entries without resolving symlinks;
    # symlink ancestry is checked separately below.
    raw_absolute = Path(os.path.normpath(str(raw.absolute())))
    try:
        raw_absolute.relative_to(prefix_absolute)
    except ValueError as exc:
        raise RuntimeError(f"distribution file escapes sys.prefix: {raw}") from exc
    current = raw_absolute
    while current != prefix_absolute:
        if current.is_symlink():
            raise RuntimeError(f"distribution file has symlinked ancestry: {raw}")
        if current == current.parent:
            raise RuntimeError(f"distribution file has unsafe ancestry: {raw}")
        current = current.parent
    try:
        resolved = raw_absolute.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as exc:
        raise RuntimeError(f"distribution file cannot be strictly resolved: {raw}") from exc
    try:
        resolved.relative_to(prefix_real)
    except ValueError as exc:
        raise RuntimeError(f"distribution file resolves outside sys.prefix: {raw}") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise RuntimeError(f"distribution file is missing or not a regular file: {raw}")
    return resolved


def is_license_file(path: Path) -> bool:
    name = path.name.lower()
    return name.startswith(LICENSE_PREFIXES)


def _safe_archive_member_name(name: str) -> str:
    """Return a canonical archive path or reject traversal/absolute names."""

    if not name or "\x00" in name:
        raise RuntimeError("sdist archive contains an invalid member path")
    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:/", normalized):
        raise RuntimeError(f"sdist archive member has absolute path: {name!r}")
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if ".." in parts:
        raise RuntimeError(f"sdist archive member escapes its root: {name!r}")
    if not parts:
        raise RuntimeError(f"sdist archive member has an empty canonical path: {name!r}")
    return "/".join(parts)


def _hash_archive_stream(stream: Any, *, size: int) -> dict[str, Any]:
    if size < 0 or size > MAX_LICENSE_EVIDENCE_BYTES:
        raise RuntimeError(f"sdist license evidence exceeds size bound: {size}")
    digest = hashlib.sha256()
    remaining = size
    while remaining:
        chunk = stream.read(min(1024 * 1024, remaining))
        if not chunk:
            raise RuntimeError("sdist license evidence ended before its declared size")
        digest.update(chunk)
        remaining -= len(chunk)
    if stream.read(1):
        raise RuntimeError("sdist license evidence exceeded its declared size")
    return {"size": size, "sha256": digest.hexdigest()}


def _archive_candidate(path: str, identity: dict[str, Any], *, archive: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "locked_sdist",
        "path": path,
        "size": identity["size"],
        "sha256": identity["sha256"],
        "archive": {
            "url": archive["url"],
            "size": archive["size"],
            "sha256": archive["hash"].split(":", 1)[1],
        },
        "disposition": PENDING_STATUS,
        "publication": NO_UPLOAD,
    }


def inspect_sdist_license_evidence(data: bytes, artifact: dict[str, Any]) -> list[dict[str, Any]]:
    """Hash only LICENSE/COPYING/NOTICE regular files in a verified sdist.

    No archive member is extracted to disk.  Unsafe paths, malformed archives,
    links, and oversized candidate members are factual evidence failures.
    """

    if len(data) != artifact.get("size"):
        raise RuntimeError("sdist archive bytes do not match the locked size")
    archive_hash = parse_hash(artifact.get("hash"), label="locked sdist hash")
    if hashlib.sha256(data).hexdigest() != archive_hash.split(":", 1)[1]:
        raise RuntimeError("sdist archive bytes do not match the locked hash")
    candidates: list[dict[str, Any]] = []
    try:
        with tarfile.open(fileobj=BytesIO(data), mode="r:*") as archive_file:
            members = archive_file.getmembers()
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise RuntimeError("sdist archive has too many members")
            seen_paths: set[str] = set()
            for member in members:
                canonical = _safe_archive_member_name(member.name)
                if canonical in seen_paths:
                    raise RuntimeError(f"sdist archive has duplicate canonical member: {canonical!r}")
                seen_paths.add(canonical)
                if not member.isfile():
                    if is_license_file(Path(canonical)):
                        raise RuntimeError(f"sdist license candidate is not a regular file: {member.name!r}")
                    continue
                if not is_license_file(Path(canonical)):
                    continue
                stream = archive_file.extractfile(member)
                if stream is None:
                    raise RuntimeError(f"sdist license candidate cannot be read: {member.name!r}")
                with stream:
                    identity = _hash_archive_stream(stream, size=member.size)
                candidates.append(_archive_candidate(canonical, identity, archive=artifact))
    except (tarfile.ReadError, EOFError) as tar_error:
        try:
            with zipfile.ZipFile(BytesIO(data)) as archive_file:
                members = archive_file.infolist()
                if len(members) > MAX_ARCHIVE_MEMBERS:
                    raise RuntimeError("sdist archive has too many members")
                seen_paths = set()
                for member in members:
                    canonical = _safe_archive_member_name(member.filename)
                    if canonical in seen_paths:
                        raise RuntimeError(f"sdist archive has duplicate canonical member: {canonical!r}")
                    seen_paths.add(canonical)
                    is_directory = member.is_dir()
                    is_symlink = (member.external_attr >> 16) & 0o170000 == 0o120000
                    if is_symlink:
                        if is_license_file(Path(canonical)):
                            raise RuntimeError(f"sdist license candidate is a symlink: {member.filename!r}")
                        continue
                    zip_type = (member.external_attr >> 16) & 0o170000
                    is_license_candidate = is_license_file(Path(canonical))
                    if zip_type not in {0, 0o100000}:
                        if is_license_candidate:
                            raise RuntimeError(f"sdist license candidate is a special file: {member.filename!r}")
                        continue
                    if is_directory:
                        if is_license_candidate:
                            raise RuntimeError(f"sdist license candidate is not a regular file: {member.filename!r}")
                        continue
                    if not is_license_candidate:
                        continue
                    with archive_file.open(member, "r") as stream:
                        identity = _hash_archive_stream(stream, size=member.file_size)
                    candidates.append(_archive_candidate(canonical, identity, archive=artifact))
        except (zipfile.BadZipFile, EOFError) as zip_error:
            raise RuntimeError("sdist is neither a readable tar nor zip archive") from zip_error
    return sorted(candidates, key=lambda item: item["path"])


def is_native_file(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(NATIVE_SUFFIXES) or ".so." in name


def is_inventory_candidate(file: Any) -> bool:
    path = Path(str(file))
    return is_license_file(path) or is_native_file(path)


def distribution_files(dist: Any) -> tuple[list[Any] | None, list[str]]:
    files = dist.files
    if files is None:
        return None, []
    result: list[Any] = []
    for file in files:
        if is_inventory_candidate(file):
            result.append(file)
    return result, []


def collect_distribution(
    dist: Any,
    expected: dict[str, Any],
    *,
    prefix: Path | None = None,
    sdist_fetcher: Any | None = None,
) -> dict[str, Any]:
    metadata = dist.metadata
    observed_name = metadata.get("Name")
    observed_version = metadata.get("Version")
    raw_license_expression = metadata.get("License-Expression")
    raw_legacy_license = metadata.get("License")
    license_expression = str(raw_license_expression).strip() if raw_license_expression else None
    legacy_license = str(raw_legacy_license).strip() if raw_legacy_license else None
    license_values = metadata_values(metadata, "License")
    classifiers = [value for value in metadata_values(metadata, "Classifier") if value.startswith("License ::")]
    files, unknown_files = distribution_files(dist)
    license_files: list[dict[str, Any]] = []
    native_files: list[dict[str, Any]] = []
    file_errors: list[str] = list(unknown_files)
    if files is not None:
        for file in files:
            relative = str(file)
            try:
                path = safe_dist_path(dist, file, prefix=prefix)
                if is_license_file(path):
                    license_files.append(regular_file_record(path, relative=relative))
                if is_native_file(path):
                    native_files.append(regular_file_record(path, relative=relative))
            except RuntimeError as exc:
                file_errors.append(str(exc))
    if files is None:
        license_status = "UNKNOWN"
        native_status = "UNKNOWN"
    else:
        license_status = "UNKNOWN" if file_errors and not license_files else "MISSING" if not license_files else "PRESENT" if len(license_files) == 1 else "MULTIPLE"
        native_status = "UNKNOWN" if file_errors else "NONE" if not native_files else "PRESENT" if len(native_files) == 1 else "MULTIPLE"
    findings: list[str] = []
    row_review_flags: list[str] = []
    sdist_candidates: list[dict[str, Any]] = []
    sdist_inspection_status = "NOT_ATTEMPTED"
    sdist_inspection_succeeded = False
    if observed_name is None or observed_version is None:
        findings.append("distribution metadata name/version missing")
    if normalize_name(str(observed_name)) != expected["normalized_name"]:
        findings.append("distribution name does not match uv.lock")
    if str(observed_version) != expected["version"]:
        findings.append("distribution version does not match uv.lock")
    alternative_publisher_evidence = bool(
        legacy_license or license_values or classifiers or license_files
    )
    if license_expression is None:
        if alternative_publisher_evidence:
            row_review_flags.append("SPDX License-Expression missing; alternative publisher evidence present")
    if license_status in {"MISSING", "UNKNOWN"}:
        if license_status == "UNKNOWN":
            findings.append("bundled license files unknown")
        elif sdist_fetcher is None:
            findings.append("bundled license files missing and no locked sdist fetcher")
        else:
            sdist = expected.get("artifacts", {}).get("sdist")
            if not isinstance(sdist, dict):
                findings.append("bundled license files missing and uv.lock has no sdist")
            else:
                try:
                    sdist_data = sdist_fetcher(sdist)
                    sdist_candidates = inspect_sdist_license_evidence(sdist_data, sdist)
                    sdist_inspection_succeeded = True
                    sdist_inspection_status = "SUCCESS"
                except (RuntimeError, OSError) as exc:
                    findings.append(f"locked sdist license evidence unavailable: {exc}")
                    sdist_candidates = []
                    sdist_inspection_status = "FAILED"
                if sdist_candidates:
                    alternative_publisher_evidence = True
                    row_review_flags.append("locked sdist license evidence requires owner review")
                elif sdist_inspection_succeeded:
                    if alternative_publisher_evidence:
                        row_review_flags.append(
                            "exact locked sdist has no bundled candidate; publisher metadata requires owner review"
                        )
                    else:
                        findings.append("locked sdist contains no LICENSE/COPYING/NOTICE evidence")
    installed_candidates = [
        {
            "source": "installed_distribution",
            "path": item["path"],
            "size": item["size"],
            "sha256": item["sha256"],
            "disposition": PENDING_STATUS,
            "publication": NO_UPLOAD,
        }
        for item in license_files
    ]
    candidate_evidence = sorted(installed_candidates + sdist_candidates, key=lambda item: (item["source"], item["path"]))
    candidate_evidence_digest = hashlib.sha256(
        json.dumps(candidate_evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if candidate_evidence:
        alternative_publisher_evidence = True
    if license_expression is None:
        if alternative_publisher_evidence and not row_review_flags:
            row_review_flags.append("SPDX License-Expression missing; alternative license evidence present")
        elif not alternative_publisher_evidence:
            findings.append("publisher license metadata and bundled license evidence missing")
    if file_errors:
        findings.append("distribution file inventory contains unknown entries")
    return {
        "name": observed_name,
        "version": observed_version,
        "expected": {"name": expected["name"], "version": expected["version"]},
        "license": {
            "spdx_expression": license_expression,
            "legacy_license": legacy_license,
            "metadata_license_values": license_values,
            "classifiers": classifiers,
            "bundled_files_status": license_status,
            "bundled_files": license_files,
            "sdist_files_status": "PRESENT" if sdist_candidates else "NONE",
            "sdist_inspection_status": sdist_inspection_status,
            "candidate_evidence": candidate_evidence,
            "candidate_evidence_sha256": candidate_evidence_digest,
            "disposition": PENDING_STATUS,
            "publication": NO_UPLOAD,
        },
        "native_payload": {
            "status": native_status,
            "files": native_files,
            "disposition": PENDING_STATUS,
            "publication": NO_UPLOAD,
        },
        "file_inventory_errors": file_errors,
        "findings": findings,
        "review_flags": row_review_flags,
    }


def collect_inventory(
    lock_data: dict[str, Any],
    distributions: Iterable[Any],
    *,
    prefix: Path | None = None,
    sdist_fetcher: Any | None = None,
) -> dict[str, Any]:
    expected = {item["normalized_name"]: item for item in lock_data["packages"]}
    installed: dict[str, list[Any]] = {}
    for dist in distributions:
        name = dist.metadata.get("Name")
        if name is not None and normalize_name(str(name)) in expected:
            installed.setdefault(normalize_name(str(name)), []).append(dist)
    rows: list[dict[str, Any]] = []
    findings: list[str] = []
    review_flags: list[str] = []
    for normalized_name, package in sorted(expected.items()):
        candidates = installed.get(normalized_name, [])
        if not candidates:
            rows.append({
                "name": package["name"],
                "version": None,
                "expected": {"name": package["name"], "version": package["version"]},
                "distribution_status": "MISSING",
                "findings": ["installed distribution missing"],
            })
            findings.append(f"missing distribution: {package['name']}")
            continue
        if len(candidates) > 1:
            rows.append({
                "name": package["name"],
                "version": None,
                "expected": {"name": package["name"], "version": package["version"]},
                "distribution_status": "MULTIPLE",
                "candidate_count": len(candidates),
                "findings": ["multiple installed distributions match normalized name"],
            })
            findings.append(f"multiple distributions: {package['name']}")
            continue
        row = collect_distribution(
            candidates[0], package, prefix=prefix, sdist_fetcher=sdist_fetcher
        )
        row["distribution_status"] = "PRESENT"
        rows.append(row)
        findings.extend(f"{package['name']}: {finding}" for finding in row["findings"])
        review_flags.extend(f"{package['name']}: {flag}" for flag in row["review_flags"])
        if row["license"]["bundled_files_status"] == "MULTIPLE":
            review_flags.append(f"multiple bundled license files: {package['name']}")
        if row["native_payload"]["status"] == "MULTIPLE":
            review_flags.append(f"multiple native payload files: {package['name']}")
        if row["native_payload"]["status"] == "UNKNOWN":
            findings.append(f"native payload inventory unknown: {package['name']}")
    return {
        "distributions": rows,
        "findings": sorted(set(findings)),
        "review_flags": sorted(set(review_flags)),
    }


def audit(
    project_path: Path,
    lock_path: Path,
    distributions: Iterable[Any] | None = None,
    *,
    prefix: Path | None = None,
    sdist_fetcher: Any | None = None,
) -> dict[str, Any]:
    lock_data = parse_lock(lock_path, project_path)
    installed = collect_inventory(
        lock_data,
        importlib.metadata.distributions() if distributions is None else distributions,
        prefix=prefix,
        sdist_fetcher=fetch_exact_sdist if sdist_fetcher is None else sdist_fetcher,
    )
    findings = installed["findings"]
    return {
        "schema": SCHEMA,
        "status": "BLOCKED" if findings else PENDING_STATUS,
        "dependency_audit_status": PENDING_DEPENDENCY_STATUS,
        "owner_approval": "PENDING_OWNER_APPROVAL",
        "weights": NO_WEIGHTS,
        "model_load": "NOT_PERFORMED",
        "model_forward": "NOT_PERFORMED",
        "publication": NO_UPLOAD,
        "disposition": {
            "license": PENDING_STATUS,
            "spdx": PENDING_STATUS,
            "native_payload": PENDING_STATUS,
            "publication": NO_UPLOAD,
        },
        "inputs": {
            "project": {"path": project_path.name, "sha256": sha256_file(project_path)},
            "lock": {"path": lock_path.name, "sha256": sha256_file(lock_path)},
            "python": platform.python_version(),
            "prefix": sys.prefix,
        },
        "locked_environment": lock_data,
        "installed_environment": installed,
        "findings": findings,
    }


class _SyntheticDistribution:
    def __init__(
        self,
        root: Path,
        name: str,
        version: str,
        *,
        license_text: str | None,
        native: bool = False,
        license_expression: str | None = "MIT",
        extra_files: dict[str, bytes] | None = None,
        classifier: bool = True,
        legacy_license: str | None = None,
    ):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.metadata = {
            "Name": name,
            "Version": version,
            "License-Expression": license_expression,
            "License": legacy_license if legacy_license is not None else ("MIT" if license_text is not None else None),
            "Classifier": "License :: OSI Approved :: MIT License" if classifier else None,
        }
        files: list[str] = []
        if license_text is not None:
            license_path = root / "LICENSE.txt"
            license_path.write_text(license_text, encoding="utf-8")
            files.append("LICENSE.txt")
        if native:
            native_path = root / "module.so"
            native_path.write_bytes(b"native")
            files.append("module.so")
        for relative, payload in (extra_files or {}).items():
            target = Path(os.path.normpath(str(root / relative)))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            files.append(relative)
        self._files = files

    @property
    def files(self) -> list[str]:  # type: ignore[override]
        return self._files

    def locate_file(self, file: str) -> Path:
        return self.root / str(file)


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="vokra-clap-license-") as temporary:
        root = Path(temporary)
        project = root / "pyproject.toml"
        lock = root / "uv.lock"
        project.write_text('[project]\nname = "synthetic-clap"\nversion = "0.0.0"\n', encoding="utf-8")
        lock.write_text(
            """version = 1\nrequires-python = \"==3.12.*\"\n[[package]]\nname = \"synthetic-clap\"\nversion = \"0.0.0\"\nsource = { virtual = \".\" }\n[[package]]\nname = \"demo-package\"\nversion = \"1.0.0\"\nsource = { registry = \"https://pypi.org/simple\" }\nsdist = { url = \"https://example.invalid/demo.tar.gz\", hash = \"sha256:0000000000000000000000000000000000000000000000000000000000000000\" }\n""",
            encoding="utf-8",
        )
        malformed_lock = root / "malformed.lock"
        malformed_lock.write_text(
            lock.read_text(encoding="utf-8").replace("sha256:" + "0" * 64, "sha256:not-a-hash"),
            encoding="utf-8",
        )
        try:
            parse_lock(malformed_lock, project)
        except RuntimeError as exc:
            assert "sha256" in str(exc)
        else:
            raise AssertionError("malformed lock artifact hash was accepted")
        dist = _SyntheticDistribution(root / "dist", "demo-package", "1.0.0", license_text="MIT\n", native=True)
        result = audit(project, lock, [dist], prefix=root)
        assert result["status"] == PENDING_STATUS
        assert result["dependency_audit_status"] != "COMPLETE"
        assert result["locked_environment"]["virtual_packages"][0]["name"] == "synthetic-clap"
        license_record = result["installed_environment"]["distributions"][0]["license"]
        assert license_record["spdx_expression"] == "MIT"
        assert license_record["legacy_license"] == "MIT"
        assert result["installed_environment"]["distributions"][0]["native_payload"]["status"] == "PRESENT"
        class _NoRecordDistribution(_SyntheticDistribution):
            @property
            def files(self) -> None:  # type: ignore[override]
                return None

        no_record = _NoRecordDistribution(
            root / "no-record",
            "demo-package",
            "1.0.0",
            license_text=None,
        )
        fetch_calls: list[dict[str, Any]] = []

        def unexpected_fetch(artifact: dict[str, Any]) -> bytes:
            fetch_calls.append(artifact)
            raise AssertionError("sdist fetch was attempted for an unknown RECORD")

        no_record_result = audit(
            project,
            lock,
            [no_record],
            prefix=root,
            sdist_fetcher=unexpected_fetch,
        )
        no_record_row = no_record_result["installed_environment"]["distributions"][0]
        assert no_record_result["status"] == "BLOCKED"
        assert no_record_row["license"]["bundled_files_status"] == "UNKNOWN"
        assert no_record_row["native_payload"]["status"] == "UNKNOWN"
        assert any("bundled license files unknown" in item for item in no_record_result["findings"])
        assert fetch_calls == []
        missing_pep639 = _SyntheticDistribution(
            root / "missing-pep639",
            "demo-package",
            "1.0.0",
            license_text="MIT\n",
            license_expression=None,
        )
        missing_pep639_result = audit(project, lock, [missing_pep639], prefix=root)
        assert missing_pep639_result["status"] == PENDING_STATUS
        assert "SPDX License-Expression missing" in " ".join(
            missing_pep639_result["installed_environment"]["review_flags"]
        )
        record_dist = _SyntheticDistribution(
            root / "lib/python/site-packages/demo.dist-info",
            "demo-package",
            "1.0.0",
            license_text=None,
            extra_files={
                "../../../bin/tool": b"console-script",
                "../../../share/licenses/demo/LICENSE": b"MIT\n",
            },
        )
        record_result = audit(project, lock, [record_dist], prefix=root)
        assert record_result["status"] == PENDING_STATUS
        record_row = record_result["installed_environment"]["distributions"][0]
        assert record_row["license"]["bundled_files_status"] == "PRESENT"
        assert record_row["file_inventory_errors"] == []
        assert record_row["license"]["bundled_files"][0]["path"] == "../../../share/licenses/demo/LICENSE"
        assert record_row["license"]["candidate_evidence_sha256"]
        duplicate = audit(project, lock, [dist, dist], prefix=root)
        assert duplicate["status"] == "BLOCKED"
        assert duplicate["installed_environment"]["distributions"][0]["distribution_status"] == "MULTIPLE"
        missing_license = _SyntheticDistribution(
            root / "missing",
            "demo-package",
            "1.0.0",
            license_text=None,
            license_expression=None,
            classifier=False,
        )
        empty_sdist = BytesIO()
        with tarfile.open(fileobj=empty_sdist, mode="w:gz"):
            pass
        empty_bytes = empty_sdist.getvalue()
        empty_artifact = {
            "url": "https://files.pythonhosted.org/packages/demo-package-1.0.0.tar.gz",
            "hash": f"sha256:{hashlib.sha256(empty_bytes).hexdigest()}",
            "size": len(empty_bytes),
        }
        empty_lock = root / "empty.lock"
        empty_lock.write_text(
            lock.read_text(encoding="utf-8").replace(
                'hash = "sha256:' + "0" * 64 + '"',
                f'hash = "{empty_artifact["hash"]}", size = {empty_artifact["size"]}',
            ),
            encoding="utf-8",
        )
        empty_fetch = lambda artifact: empty_bytes
        missing = audit(
            project, empty_lock, [missing_license], prefix=root, sdist_fetcher=empty_fetch
        )
        assert missing["status"] == "BLOCKED"
        assert "contains no LICENSE/COPYING/NOTICE" in " ".join(missing["findings"])
        assert missing["installed_environment"]["distributions"][0]["license"]["sdist_inspection_status"] == "SUCCESS"
        tqdm_shape = _SyntheticDistribution(
            root / "tqdm-shape",
            "demo-package",
            "1.0.0",
            license_text=None,
            license_expression=None,
            classifier=False,
            legacy_license="MPL-2.0 AND MIT",
        )
        tqdm_result = audit(
            project, empty_lock, [tqdm_shape], prefix=root, sdist_fetcher=empty_fetch
        )
        tqdm_row = tqdm_result["installed_environment"]["distributions"][0]
        assert tqdm_result["status"] == PENDING_STATUS
        assert tqdm_row["license"]["sdist_files_status"] == "NONE"
        assert tqdm_row["license"]["sdist_inspection_status"] == "SUCCESS"
        assert any("no bundled candidate" in item for item in tqdm_row["review_flags"])
        assert not any("contains no LICENSE/COPYING/NOTICE" in item for item in tqdm_result["findings"])
        def failing_fetch(artifact: dict[str, Any]) -> bytes:
            raise RuntimeError("synthetic network failure")

        unavailable = audit(
            project, empty_lock, [missing_license], prefix=root, sdist_fetcher=failing_fetch
        )
        unavailable_findings = unavailable["findings"]
        assert sum("locked sdist license evidence unavailable" in item for item in unavailable_findings) == 1
        assert not any("contains no LICENSE/COPYING/NOTICE" in item for item in unavailable_findings)
        assert missing["installed_environment"]["distributions"][0]["license"]["legacy_license"] is None
        evidence_tar = BytesIO()
        with tarfile.open(fileobj=evidence_tar, mode="w:gz") as archive_file:
            info = tarfile.TarInfo("demo-package-1.0.0/LICENSE")
            payload = b"MIT\n"
            info.size = len(payload)
            archive_file.addfile(info, BytesIO(payload))
        evidence_bytes = evidence_tar.getvalue()
        artifact = {
            "url": "https://files.pythonhosted.org/packages/demo-package-1.0.0.tar.gz",
            "hash": f"sha256:{hashlib.sha256(evidence_bytes).hexdigest()}",
            "size": len(evidence_bytes),
        }
        evidence_lock = root / "evidence.lock"
        evidence_lock.write_text(
            lock.read_text(encoding="utf-8").replace(
                'hash = "sha256:' + "0" * 64 + '"',
                f'hash = "{artifact["hash"]}", size = {artifact["size"]}',
            ),
            encoding="utf-8",
        )

        class _StaticResponse:
            def __init__(self, payload: bytes):
                self.payload = payload
                self.headers = {"Content-Length": str(len(payload))}

            def __enter__(self) -> "_StaticResponse":
                return self

            def __exit__(self, *args: Any) -> None:
                return None

            def read(self, amount: int = -1) -> bytes:
                payload, self.payload = self.payload, b""
                return payload if amount < 0 else payload[:amount]

        fetched = fetch_exact_sdist(
            artifact, open_url=lambda request, timeout: _StaticResponse(evidence_bytes)
        )
        assert fetched == evidence_bytes
        assert inspect_sdist_license_evidence(fetched, artifact)[0]["path"].endswith("/LICENSE")
        wrong_hash = dict(artifact, hash="sha256:" + "0" * 64)
        try:
            fetch_exact_sdist(
                wrong_hash,
                open_url=lambda request, timeout: _StaticResponse(evidence_bytes),
            )
        except RuntimeError as exc:
            assert "SHA-256" in str(exc)
        else:
            raise AssertionError("tampered sdist hash was accepted")
        wrong_size = dict(artifact, size=len(evidence_bytes) + 1)
        try:
            fetch_exact_sdist(
                wrong_size,
                open_url=lambda request, timeout: _StaticResponse(evidence_bytes),
            )
        except RuntimeError as exc:
            assert "Content-Length" in str(exc)
        else:
            raise AssertionError("tampered sdist size was accepted")
        handler = _RestrictedRedirectHandler()
        try:
            handler.redirect_request(
                urllib.request.Request(artifact["url"]),
                None,
                302,
                "found",
                {},
                "https://evil.invalid/demo.tar.gz",
            )
        except RuntimeError as exc:
            assert "PyPI artifact URL" in str(exc)
        else:
            raise AssertionError("cross-host sdist redirect was accepted")
        unsafe_tar = BytesIO()
        with tarfile.open(fileobj=unsafe_tar, mode="w") as archive_file:
            info = tarfile.TarInfo("../../LICENSE")
            payload = b"MIT\n"
            info.size = len(payload)
            archive_file.addfile(info, BytesIO(payload))
        unsafe_artifact = dict(
            artifact,
            hash=f"sha256:{hashlib.sha256(unsafe_tar.getvalue()).hexdigest()}",
            size=len(unsafe_tar.getvalue()),
        )
        try:
            inspect_sdist_license_evidence(unsafe_tar.getvalue(), unsafe_artifact)
        except RuntimeError as exc:
            assert "escapes" in str(exc)
        else:
            raise AssertionError("unsafe sdist path was accepted")
        duplicate_tar = BytesIO()
        with tarfile.open(fileobj=duplicate_tar, mode="w") as archive_file:
            for name in ("LICENSE", "./LICENSE"):
                info = tarfile.TarInfo(name)
                payload = b"MIT\n"
                info.size = len(payload)
                archive_file.addfile(info, BytesIO(payload))
        duplicate_tar_bytes = duplicate_tar.getvalue()
        duplicate_tar_artifact = dict(
            artifact,
            hash=f"sha256:{hashlib.sha256(duplicate_tar_bytes).hexdigest()}",
            size=len(duplicate_tar_bytes),
        )
        try:
            inspect_sdist_license_evidence(duplicate_tar_bytes, duplicate_tar_artifact)
        except RuntimeError as exc:
            assert "duplicate canonical" in str(exc)
        else:
            raise AssertionError("duplicate tar canonical path was accepted")
        duplicate_zip = BytesIO()
        with zipfile.ZipFile(duplicate_zip, mode="w") as archive_file:
            archive_file.writestr("LICENSE", b"MIT\n")
            archive_file.writestr("./LICENSE", b"MIT\n")
        duplicate_zip_bytes = duplicate_zip.getvalue()
        duplicate_zip_artifact = dict(
            artifact,
            hash=f"sha256:{hashlib.sha256(duplicate_zip_bytes).hexdigest()}",
            size=len(duplicate_zip_bytes),
        )
        try:
            inspect_sdist_license_evidence(duplicate_zip_bytes, duplicate_zip_artifact)
        except RuntimeError as exc:
            assert "duplicate canonical" in str(exc)
        else:
            raise AssertionError("duplicate zip canonical path was accepted")
        special_zip = BytesIO()
        with zipfile.ZipFile(special_zip, mode="w") as archive_file:
            special_info = zipfile.ZipInfo("LICENSE")
            special_info.external_attr = 0o010644 << 16
            archive_file.writestr(special_info, b"MIT\n")
        special_zip_bytes = special_zip.getvalue()
        special_zip_artifact = dict(
            artifact,
            hash=f"sha256:{hashlib.sha256(special_zip_bytes).hexdigest()}",
            size=len(special_zip_bytes),
        )
        try:
            inspect_sdist_license_evidence(special_zip_bytes, special_zip_artifact)
        except RuntimeError as exc:
            assert "special file" in str(exc)
        else:
            raise AssertionError("special zip license file was accepted")
        try:
            _safe_archive_member_name("./")
        except RuntimeError as exc:
            assert "empty canonical" in str(exc)
        else:
            raise AssertionError("empty canonical archive path was accepted")
        pending = audit(
            project,
            evidence_lock,
            [missing_license],
            prefix=root,
            sdist_fetcher=lambda artifact: evidence_bytes,
        )
        pending_row = pending["installed_environment"]["distributions"][0]
        assert pending["status"] == PENDING_STATUS
        assert pending_row["license"]["sdist_files_status"] == "PRESENT"
        assert pending_row["review_flags"]
        with tempfile.TemporaryDirectory(
            prefix="vokra-clap-license-escape-", dir=root.parent
        ) as escaped_root:
            escaped = _SyntheticDistribution(
                Path(escaped_root),
                "demo-package",
                "1.0.0",
                license_text="MIT\n",
                native=True,
            )
            escaped_result = audit(project, lock, [escaped], prefix=root)
            assert escaped_result["status"] == "BLOCKED"
            escaped_row = escaped_result["installed_environment"]["distributions"][0]
            assert "escapes sys.prefix" in " ".join(escaped_row["file_inventory_errors"])
        output = root / "inventory.json"
        write_atomic_no_replace(output, "first\n")
        try:
            write_atomic_no_replace(output, "second\n")
        except RuntimeError as exc:
            assert "already exists" in str(exc)
        else:
            raise AssertionError("inventory output replacement was accepted")
        for raw in ("relative", "/", "/tmp/./audit.json", "/tmp/../audit.json", "/tmp/audit\x00.json"):
            try:
                cli_path(raw, "output")
            except RuntimeError:
                pass
            else:
                raise AssertionError("unsafe raw output path was accepted")
        owned_temp = root / "owned.tmp"
        owned_temp.write_text("owner", encoding="utf-8")
        owned_stat = os.stat(owned_temp, follow_symlinks=False)
        owned_temp.unlink()
        owned_temp.write_text("replacement", encoding="utf-8")
        try:
            verify_temporary_identity(owned_temp, (owned_stat.st_dev, owned_stat.st_ino))
        except RuntimeError:
            pass
        else:
            raise AssertionError("replacement temporary was accepted")
        cleanup_temporary(owned_temp, (owned_stat.st_dev, owned_stat.st_ino))
        assert owned_temp.exists(), "cleanup removed a replacement temporary"
        real_output_dir = root / "real-output"
        real_output_dir.mkdir()
        linked_output_dir = root / "linked-output"
        linked_output_dir.symlink_to(real_output_dir, target_is_directory=True)
        try:
            write_atomic_no_replace(linked_output_dir / "unsafe.json", "{}\n")
        except RuntimeError as exc:
            assert "symlink ancestry" in str(exc)
        else:
            raise AssertionError("symlinked output ancestry was accepted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--project")
    parser.add_argument("--lock")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.project, args.lock, args.output)):
            parser.error("--self-test accepts no audit paths")
        self_test()
        print("clap dependency/license audit self-test: OK")
        return 0
    if args.project is None or args.lock is None or args.output is None:
        parser.error("normal runs require --project, --lock, and --output")
    project = cli_path(args.project, "project")
    lock = cli_path(args.lock, "lock")
    output = cli_path(args.output, "output")
    evidence = audit(project, lock)
    try:
        write_atomic_no_replace(output, json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    except RuntimeError as exc:
        parser.error(str(exc))
    print(f"CLAP_DEPENDENCY_LICENSE_AUDIT {evidence['status']}: {output}")
    return 0 if evidence["status"] != "BLOCKED" else 2


if __name__ == "__main__":
    sys.exit(main())
