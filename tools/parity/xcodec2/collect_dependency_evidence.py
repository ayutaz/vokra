#!/usr/bin/env -S uv run --no-project --python 3.12 python
"""Collect model-free XCodec2 Linux dependency evidence.

The collector is intentionally an evidence producer, never an approval tool.
It reads installed distribution metadata and locked package archives, records
literal license/notice bytes and native ELF dependencies, and always reports
``NO_UPLOAD`` plus owner review.  It does not import Torch/XCodec2 or acquire
models, checkpoints, or audio.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import csv
import hashlib
import importlib.metadata as metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import zipfile

import dependency_audit as audit


MAX_ARTIFACT_BYTES = 4 * 1024 * 1024 * 1024
MAX_LICENSE_BYTES = 2 * 1024 * 1024
MAX_LICENSE_TOTAL_BYTES = 4 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 10000
MAX_MEMBER_BYTES = 8 * 1024 * 1024
MAX_TOTAL_MEMBER_BYTES = 64 * 1024 * 1024
MAX_REDIRECTS = 3
LICENSE_NAMES = {"license", "licence", "copying", "notice", "copyright", "eula", "end_user_license", "end-user-license"}
NATIVE_SUFFIXES = {".so", ".pyd", ".dylib", ".dll"}
ELF_MAGIC = b"\x7fELF"


class EvidenceError(ValueError):
    """A fail-closed evidence error."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def identity(name: str, version: str) -> str:
    return f"{re.sub(r'[-_.]+', '-', name).casefold()}=={version.casefold()}"


def assert_safe_path(path: Path) -> None:
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise EvidenceError(f"path must be absolute and canonical: {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise EvidenceError(f"symlink ancestry is not allowed: {current}")


def safe_member(name: str) -> str:
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name or name.startswith("/"):
        raise EvidenceError(f"unsafe archive member: {name!r}")
    clean = name.rstrip("/")
    if not clean or any(part in {"", ".", ".."} for part in clean.split("/")):
        raise EvidenceError(f"archive traversal: {name!r}")
    return clean


def is_license_path(name: str) -> bool:
    base = PurePosixPath(name).name.casefold()
    return base in LICENSE_NAMES or any(base.startswith(prefix + sep) for prefix in LICENSE_NAMES for sep in (".", "-", "_"))


def archive_license_files(path: Path) -> list[dict[str, Any]]:
    names: set[str] = set()
    total = 0
    license_total = 0
    found: list[dict[str, Any]] = []

    def record(name: str, size: int, regular: bool, reader: Callable[[], bytes]) -> None:
        nonlocal total, license_total
        clean = safe_member(name)
        if clean in names:
            raise EvidenceError(f"duplicate archive member: {clean}")
        names.add(clean)
        if len(names) > MAX_ARCHIVE_MEMBERS or size < 0:
            raise EvidenceError("archive member/count bound exceeded")
        if not regular:
            return
        if not is_license_path(clean):
            return
        if size > MAX_MEMBER_BYTES:
            raise EvidenceError("license member size exceeded")
        total += size
        if total > MAX_TOTAL_MEMBER_BYTES:
            raise EvidenceError("license aggregate bound exceeded")
        license_total += size
        if license_total > MAX_LICENSE_TOTAL_BYTES:
            raise EvidenceError("license aggregate bound exceeded")
        data = reader()
        if len(data) != size or size > MAX_LICENSE_BYTES:
            raise EvidenceError("license member size changed or exceeded bound")
        found.append({"path": clean, "bytes": size, "sha256": sha256_bytes(data), "content_base64": base64.b64encode(data).decode("ascii")})

    suffix = path.name.casefold()
    try:
        if suffix.endswith((".whl", ".zip")):
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    kind = (info.external_attr >> 16) & 0o170000
                    directory = info.is_dir() or info.filename.endswith("/") or kind == stat.S_IFDIR
                    if kind and kind not in {stat.S_IFREG, stat.S_IFDIR}:
                        raise EvidenceError(f"zip link/special member: {info.filename}")
                    record(info.filename, 0 if directory else info.file_size, not directory, lambda info=info: archive.read(info))
        else:
            if not suffix.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".tar")):
                raise EvidenceError("unsupported locked archive type")
            mode = "r:gz" if suffix.endswith((".tar.gz", ".tgz")) else "r:bz2" if suffix.endswith((".tar.bz2", ".tbz2")) else "r:xz" if suffix.endswith((".tar.xz", ".txz")) else "r:"
            with tarfile.open(path, mode=mode) as archive:
                for info in archive:
                    if info.issym() or info.islnk() or info.isdev() or info.isfifo() or (not info.isdir() and not info.isfile()):
                        raise EvidenceError(f"tar link/special member: {info.name}")
                    stream = archive.extractfile(info) if info.isfile() else None
                    record(info.name, info.size if info.isfile() else 0, info.isfile(), lambda stream=stream: stream.read(MAX_LICENSE_BYTES + 1) if stream else b"")
                    if stream:
                        stream.close()
    except (OSError, EOFError, RuntimeError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise EvidenceError(f"archive inspection failed: {type(exc).__name__}") from exc
    return sorted(found, key=lambda item: item["path"])


def validate_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in audit.PYPI_HOSTS | audit.TORCH_HOSTS or parsed.username or parsed.password or parsed.port not in (None, 443) or parsed.query or parsed.fragment or not parsed.path:
        raise EvidenceError("artifact URL is outside the audited HTTPS hosts")


class SafeRedirects(HTTPRedirectHandler):
    def __init__(self, trace: list[str], initial: str) -> None:
        self.trace, self.initial = trace, initial

    def redirect_request(self, request: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Request | None:
        if len(self.trace) - 1 >= MAX_REDIRECTS:
            raise EvidenceError("redirect limit exceeded")
        resolved = urljoin(request.full_url, newurl)
        validate_url(resolved)
        if urlsplit(resolved).path != urlsplit(self.initial).path:
            raise EvidenceError("redirect changed locked artifact path")
        self.trace.append(resolved)
        return super().redirect_request(request, fp, code, msg, headers, resolved)


def fetch_artifact(artifact: dict[str, Any], temporary: Path, fetcher: Callable[[str], tuple[str, bytes]] | None = None) -> dict[str, Any]:
    url, expected_hash = artifact.get("url"), artifact.get("hash")
    expected_size = artifact.get("size")
    if not isinstance(url, str) or not isinstance(expected_hash, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_hash):
        raise EvidenceError("locked artifact identity is malformed")
    validate_url(url)
    if expected_size is not None and (not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size <= 0 or expected_size > MAX_ARTIFACT_BYTES):
        raise EvidenceError("locked artifact size is malformed")
    trace = [url]
    basename = PurePosixPath(urlsplit(url).path).name
    if not basename or basename in {".", ".."} or "/" in basename or "\\" in basename or ".." in basename:
        raise EvidenceError("locked artifact basename is unsafe")
    output = temporary / (sha256_bytes(url.encode()) + "-" + basename)
    if fetcher is not None:
        final, body = fetcher(url)
        validate_url(final)
        if urlsplit(final).path != urlsplit(url).path or not isinstance(body, bytes):
            raise EvidenceError("test fetcher changed artifact identity")
        if len(body) > MAX_ARTIFACT_BYTES or expected_size is not None and len(body) != expected_size or "sha256:" + sha256_bytes(body) != expected_hash:
            raise EvidenceError("test artifact bytes do not match lock")
        output.write_bytes(body)
        return {"url": url, "final_url": final, "bytes": len(body), "sha256": sha256_bytes(body), "temporary": str(output)}
    digest = hashlib.sha256()
    count = 0
    try:
        opener = build_opener(SafeRedirects(trace, url))
        request = Request(url, headers={"Accept": "application/octet-stream", "User-Agent": "vokra-xcodec2-evidence/1"})
        with opener.open(request, timeout=60) as response, output.open("xb") as stream:
            final = urljoin(url, response.geturl())
            validate_url(final)
            if urlsplit(final).path != urlsplit(url).path:
                raise EvidenceError("response redirect changed locked artifact path")
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                count += len(chunk)
                if count > MAX_ARTIFACT_BYTES:
                    raise EvidenceError("artifact exceeds bounded size")
                digest.update(chunk)
                stream.write(chunk)
    except (HTTPError, URLError, OSError, UnicodeError, ValueError) as exc:
        raise EvidenceError(f"artifact acquisition failed: {type(exc).__name__}") from exc
    observed = digest.hexdigest()
    if expected_size is not None and count != expected_size or "sha256:" + observed != expected_hash:
        raise EvidenceError("locked artifact bytes do not match lock")
    return {"url": url, "final_url": trace[-1], "bytes": count, "sha256": observed, "temporary": str(output)}


def safe_dist_path(dist: metadata.Distribution, entry: Any) -> Path | None:
    root = Path(dist.locate_file(""))
    path = Path(dist.locate_file(entry))
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    current = root
    if current.is_symlink():
        return None
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return None
    resolved_root, resolved = root.resolve(strict=False), path.resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        return None
    return resolved


def installed_license_files(dist: metadata.Distribution) -> tuple[list[dict[str, Any]], list[str]]:
    result: list[dict[str, Any]] = []
    unsafe: list[str] = []
    total = 0
    for entry in sorted(dist.files or [], key=str):
        relative = str(entry)
        if not is_license_path(relative):
            continue
        path = safe_dist_path(dist, entry)
        if path is None or path.is_symlink() or not path.is_file():
            unsafe.append(relative)
            continue
        size = path.stat().st_size
        if size > MAX_LICENSE_BYTES or total + size > MAX_LICENSE_TOTAL_BYTES:
            raise EvidenceError(f"publisher license bounds exceeded: {relative}")
        data = path.read_bytes()
        result.append({"path": relative, "bytes": size, "sha256": sha256_bytes(data), "content_base64": base64.b64encode(data).decode("ascii")})
        total += size
    return result, unsafe


def native_evidence(dist: metadata.Distribution) -> tuple[list[dict[str, Any]], list[str]]:
    native: list[dict[str, Any]] = []
    failures: list[str] = []
    for entry in sorted(dist.files or [], key=str):
        relative = str(entry)
        path = safe_dist_path(dist, entry)
        name = Path(relative).name.casefold()
        if path is None or path.is_symlink() or not path.is_file():
            if Path(relative).suffix.casefold() in NATIVE_SUFFIXES or ".so." in name:
                failures.append(f"unsafe native path: {relative}")
            continue
        try:
            with path.open("rb") as stream:
                magic = stream.read(4)
            if not (Path(relative).suffix.casefold() in NATIVE_SUFFIXES or ".so." in name or magic == ELF_MAGIC):
                continue
            item: dict[str, Any] = {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path), "elf": {"format": "non-elf", "needed": [], "inspection": "not-applicable"}}
            if magic == ELF_MAGIC:
                try:
                    result = subprocess.run(["readelf", "-d", str(path)], capture_output=True, text=True, timeout=60, check=False)
                    needed = sorted(match.group(1) for line in result.stdout.splitlines() if (match := re.search(r"\(NEEDED\).*\[([^]]+)\]", line)))
                    item["elf"] = {"format": "elf", "needed": needed, "inspection": "ok" if result.returncode == 0 else "error", "readelf_returncode": result.returncode}
                    if result.returncode != 0:
                        failures.append(f"readelf failed: {relative}")
                except (OSError, subprocess.TimeoutExpired) as exc:
                    failures.append(f"readelf error {type(exc).__name__}: {relative}")
            native.append(item)
        except OSError as exc:
            failures.append(f"native read failed {type(exc).__name__}: {relative}")
    return native, failures


def installed_payload_hashes(dist: metadata.Distribution, targets: set[str]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Hash selected installed payload files without reading them into RAM."""
    result: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    by_name = {str(entry): entry for entry in (dist.files or [])}
    for relative in sorted(targets):
        if relative not in by_name:
            failures.append(f"installed binding path is not listed by RECORD: {relative}")
            continue
        path = safe_dist_path(dist, by_name[relative])
        if path is None or path.is_symlink() or not path.is_file():
            failures.append(f"unsafe installed binding path: {relative}")
            continue
        try:
            result[relative] = {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        except OSError as exc:
            failures.append(f"installed binding read failed {type(exc).__name__}: {relative}")
    return result, failures


def wheel_record_entries(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Read and validate the publisher RECORD inventory from a wheel."""
    if not path.name.casefold().endswith(".whl"):
        raise EvidenceError("publisher RECORD requires a locked wheel")
    record_members: list[str] = []
    record_bytes: bytes | None = None
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            clean = safe_member(info.filename)
            if clean.endswith(".dist-info/RECORD"):
                record_members.append(clean)
                if info.file_size > MAX_LICENSE_BYTES:
                    raise EvidenceError("publisher RECORD exceeds bound")
                with archive.open(info, "r") as stream:
                    record_bytes = stream.read(MAX_LICENSE_BYTES + 1)
    if len(record_members) != 1 or record_bytes is None or len(record_bytes) > MAX_LICENSE_BYTES:
        raise EvidenceError("wheel must contain exactly one bounded RECORD")
    try:
        rows = list(csv.reader(record_bytes.decode("utf-8").splitlines()))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise EvidenceError("publisher RECORD is not valid UTF-8 CSV") from exc
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if len(row) != 3:
            raise EvidenceError("publisher RECORD row is malformed")
        relative, encoded_hash, encoded_size = row
        relative = safe_member(relative)
        if relative in seen:
            raise EvidenceError(f"duplicate publisher RECORD path: {relative}")
        seen.add(relative)
        if encoded_hash and not encoded_hash.startswith("sha256="):
            raise EvidenceError(f"publisher RECORD hash algorithm is unsupported: {relative}")
        expected_hash = None
        if encoded_hash:
            try:
                digest = base64.urlsafe_b64decode(encoded_hash.removeprefix("sha256=") + "===")
            except (ValueError, UnicodeError) as exc:
                raise EvidenceError(f"publisher RECORD hash is malformed: {relative}") from exc
            if len(digest) != hashlib.sha256().digest_size:
                raise EvidenceError(f"publisher RECORD hash length is invalid: {relative}")
            expected_hash = digest.hex()
        if encoded_size and (not encoded_size.isdigit() or int(encoded_size) < 0):
            raise EvidenceError(f"publisher RECORD size is malformed: {relative}")
        entries.append({"path": relative, "sha256": expected_hash, "bytes": int(encoded_size) if encoded_size else None})
    return entries, record_members[0]


def is_generated_installer_path(relative: str) -> bool:
    base = PurePosixPath(relative).name
    parts = PurePosixPath(relative).parts
    return (len(parts) == 2 and parts[0].endswith(".dist-info") and base in {"RECORD", "WHEEL", "INSTALLER", "REQUESTED"}) or relative.casefold().endswith(".pyc")


def is_wheel_data_relocation(relative: str) -> bool:
    return any(part.endswith(".data") for part in PurePosixPath(relative).parts)


def archive_required_inventory(path: Path) -> dict[str, dict[str, Any]]:
    """Independently inventory archive LICENSE/native/METADATA members."""
    if not path.name.casefold().endswith(".whl"):
        raise EvidenceError("archive inventory requires a locked wheel")
    result: dict[str, dict[str, Any]] = {}
    license_total = 0
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            clean = safe_member(info.filename)
            kind = (info.external_attr >> 16) & 0o170000
            if kind and kind not in {stat.S_IFREG, stat.S_IFDIR}:
                raise EvidenceError(f"wheel link/special member: {clean}")
            if info.is_dir() or kind == stat.S_IFDIR:
                continue
            parts = PurePosixPath(clean).parts
            metadata_member = len(parts) == 2 and parts[0].endswith(".dist-info") and parts[1] == "METADATA"
            license_member = is_license_path(clean)
            native_member = Path(clean).suffix.casefold() in NATIVE_SUFFIXES or ".so." in Path(clean).name.casefold()
            with archive.open(info, "r") as stream:
                first = stream.read(4)
                if not (metadata_member or license_member or native_member or first == ELF_MAGIC):
                    continue
                digest = hashlib.sha256()
                digest.update(first)
                count = len(first)
                while True:
                    chunk = stream.read(1 << 20)
                    if not chunk:
                        break
                    count += len(chunk)
                    if license_member and (count > MAX_MEMBER_BYTES or license_total + count > MAX_LICENSE_TOTAL_BYTES):
                        raise EvidenceError(f"archive license bounds exceeded: {clean}")
                    if count > MAX_ARTIFACT_BYTES:
                        raise EvidenceError(f"required archive member exceeds bound: {clean}")
                    digest.update(chunk)
            if license_member:
                if count > MAX_MEMBER_BYTES or license_total + count > MAX_LICENSE_TOTAL_BYTES:
                    raise EvidenceError(f"archive license bounds exceeded: {clean}")
                license_total += count
            roles = []
            if metadata_member:
                roles.append("METADATA")
            if license_member:
                roles.append("LICENSE_OR_NOTICE")
            if native_member or first == ELF_MAGIC:
                roles.append("NATIVE")
            if clean in result:
                raise EvidenceError(f"duplicate required archive member: {clean}")
            result[clean] = {"path": clean, "bytes": count, "sha256": digest.hexdigest(), "roles": roles}
    return result


def archive_member_hashes(path: Path, targets: set[str]) -> dict[str, dict[str, Any]]:
    """Stream-hash selected wheel members without loading native payloads in RAM."""
    if not path.name.casefold().endswith(".whl"):
        raise EvidenceError("installed-build binding requires a locked wheel")
    result: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            clean = safe_member(info.filename)
            kind = (info.external_attr >> 16) & 0o170000
            if kind and kind not in {stat.S_IFREG, stat.S_IFDIR}:
                raise EvidenceError(f"wheel link/special member: {clean}")
            if clean not in targets:
                continue
            if clean in result or info.is_dir() or kind == stat.S_IFDIR:
                raise EvidenceError(f"wheel binding member is duplicate or non-regular: {clean}")
            digest = hashlib.sha256()
            count = 0
            with archive.open(info, "r") as stream:
                while True:
                    chunk = stream.read(1 << 20)
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > MAX_ARTIFACT_BYTES:
                        raise EvidenceError(f"wheel binding member exceeds bound: {clean}")
                    digest.update(chunk)
            result[clean] = {"path": clean, "bytes": count, "sha256": digest.hexdigest()}
    return result


def compare_wheel_payloads(path: Path, installed: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    records, record_path = wheel_record_entries(path)
    required = {entry["path"] for entry in records if not is_generated_installer_path(entry["path"])}
    archive_inventory = archive_required_inventory(path)
    required.update(archive_inventory)
    if any(is_wheel_data_relocation(relative) for relative in required):
        raise EvidenceError("wheel .data relocation is unsupported; an authenticated mapping is required")
    archive = archive_member_hashes(path, required)
    failures: list[str] = []
    comparisons: list[dict[str, Any]] = []
    record_by_path = {entry["path"]: entry for entry in records}
    for relative in sorted(required):
        observed = archive.get(relative)
        if observed is None:
            failures.append(f"selected wheel is missing publisher RECORD payload: {relative}")
            continue
        expected_record = record_by_path.get(relative)
        if expected_record is None:
            failures.append(f"required archive member is absent from publisher RECORD: {relative}")
        record_match = expected_record is not None and ((expected_record["sha256"] is None or expected_record["sha256"] == observed["sha256"]) and (expected_record["bytes"] is None or expected_record["bytes"] == observed["bytes"]))
        expected = installed.get(relative)
        if expected is None:
            failures.append(f"installed payload is missing publisher RECORD path: {relative}")
            comparisons.append({"path": relative, "publisher_record": expected_record, "archive": observed, "installed": None, "match": False})
            continue
        match = record_match and observed["bytes"] == expected["bytes"] and observed["sha256"] == expected["sha256"]
        comparisons.append({"path": relative, "publisher_record": expected_record, "installed": expected, "archive": observed, "match": match})
        if not record_match:
            failures.append(f"publisher RECORD does not match archive: {relative}")
        if not match:
            failures.append(f"installed/archive payload mismatch: {relative}")
    return {"status": "PAYLOAD_SCOPED_VERIFIED" if not failures else "BLOCKED", "scope": "publisher RECORD payload excluding generated installer metadata", "full_installed_build_identity": "UNVERIFIED", "record_path": record_path, "files": comparisons, "excluded_generated_metadata": ["*.dist-info/RECORD", "*.dist-info/WHEEL", "*.dist-info/INSTALLER", "*.dist-info/REQUESTED", "*.pyc"]}, failures


def wheel_tags(url: str) -> set[tuple[str, str, str]]:
    name = PurePosixPath(urlsplit(url).path).name
    if not name.endswith(".whl"):
        raise EvidenceError("locked wheel URL does not name a wheel")
    fields = name[:-4].split("-")
    if len(fields) < 5:
        raise EvidenceError("malformed wheel filename")
    python_tag, abi_tag, platform_tag = fields[-3:]
    return {(python, abi, platform_name) for python in python_tag.split(".") for abi in abi_tag.split(".") for platform_name in platform_tag.split(".")}


def installed_wheel_info(dist: metadata.Distribution) -> dict[str, Any]:
    candidates = []
    for entry in sorted(dist.files or [], key=str):
        relative = str(entry)
        parts = PurePosixPath(relative).parts
        if len(parts) == 2 and parts[0].endswith(".dist-info") and parts[1] == "WHEEL":
            path = safe_dist_path(dist, entry)
            if path is None or path.is_symlink() or not path.is_file():
                raise EvidenceError("installed WHEEL path is unsafe")
            candidates.append((relative, path))
    if len(candidates) != 1:
        raise EvidenceError("installed distribution must contain exactly one WHEEL file")
    relative, path = candidates[0]
    data = path.read_bytes()
    tags: set[tuple[str, str, str]] = set()
    for line in data.decode("utf-8").splitlines():
        if line.startswith("Tag:"):
            fields = line[5:].strip().split("-")
            if len(fields) != 3:
                raise EvidenceError("installed WHEEL tag is malformed")
            tags.update((python, abi, platform_name) for python in fields[0].split(".") for abi in fields[1].split(".") for platform_name in fields[2].split("."))
    if not tags:
        raise EvidenceError("installed WHEEL has no tags")
    return {"path": relative, "bytes": len(data), "sha256": sha256_bytes(data), "tags": ["-".join(tag) for tag in sorted(tags)]}


def choose_artifact(row: dict[str, Any], installed_tags: set[tuple[str, str, str]]) -> tuple[str, dict[str, Any], set[tuple[str, str, str]]]:
    matches = []
    for artifact in row.get("wheels", []):
        if not isinstance(artifact, dict) or not isinstance(artifact.get("url"), str):
            continue
        overlap = wheel_tags(artifact["url"]) & installed_tags
        if overlap:
            matches.append((artifact, overlap))
    if len(matches) != 1:
        if not row.get("wheels") and isinstance(row.get("sdist"), dict):
            return "locked_sdist", row["sdist"], set()
        raise EvidenceError(f"installed WHEEL binds {len(matches)} locked wheels for {row['name']}; expected one")
    return "locked_wheel", matches[0][0], matches[0][1]


def metadata_fields(dist: metadata.Distribution) -> dict[str, Any]:
    return {
        "license": (dist.metadata.get("License") or "").strip() or None,
        "license_expression": (dist.metadata.get("License-Expression") or "").strip() or None,
        "license_classifiers": sorted(value.removeprefix("License :: ") for value in (dist.metadata.get_all("Classifier") or []) if value.startswith("License :: ")),
    }


def inspect_distribution(row: dict[str, Any], record: dict[str, Any], temporary: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    dist: metadata.Distribution = record["distribution"]
    wheel = installed_wheel_info(dist)
    artifact_kind, locked_artifact, matched_tags = choose_artifact(row, set(tuple(tag.split("-")) for tag in wheel["tags"]))
    artifact = fetch_artifact(locked_artifact, temporary)
    publisher, unsafe = installed_license_files(dist)
    native, native_failures = native_evidence(dist)
    if artifact_kind == "locked_wheel":
        wheel_records, _ = wheel_record_entries(Path(artifact["temporary"]))
        wheel_record_paths = {entry["path"] for entry in wheel_records}
        archive_inventory = archive_required_inventory(Path(artifact["temporary"]))
        binding_targets = {entry["path"] for entry in wheel_records if not is_generated_installer_path(entry["path"])} | set(archive_inventory)
    else:
        wheel_record_paths = set()
        archive_inventory = {}
        binding_targets = {item["path"] for item in native} | {item["path"] for item in publisher}
    installed_payload, binding_failures = installed_payload_hashes(dist, binding_targets)
    installed_build_binding: dict[str, Any]
    wheel_binding: dict[str, Any] | None = None
    if artifact_kind == "locked_wheel":
        wheel_binding, wheel_failures = compare_wheel_payloads(Path(artifact["temporary"]), installed_payload)
        binding_failures.extend(wheel_failures)
        installed_build_binding = wheel_binding
    else:
        installed_build_binding = {"status": "UNVERIFIED", "reason": "locked sdist bytes do not identify the installed build payload; an authenticated build/RECORD proof is required"}
        binding_failures.append("installed build identity is unverified for locked sdist")
    sdist_evidence: list[dict[str, Any]] = []
    wheel_license_evidence = archive_license_files(Path(artifact["temporary"])) if artifact_kind == "locked_wheel" else []
    failures = list(native_failures) + binding_failures + [f"unsafe publisher license path: {path}" for path in unsafe]
    if artifact_kind == "locked_wheel":
        for relative, item in archive_inventory.items():
            if relative not in wheel_record_paths:
                failures.append(f"required archive {','.join(item['roles'])} member is absent from publisher RECORD: {relative}")
    for item in wheel_license_evidence:
        if artifact_kind == "locked_wheel" and item["path"] not in wheel_record_paths:
            failures.append(f"selected wheel license is absent from publisher RECORD: {item['path']}")
        if item["path"] not in installed_payload:
            failures.append(f"selected wheel license is not present in installed payload: {item['path']}")
    if artifact_kind == "locked_wheel":
        for item in native:
            if item["path"] not in wheel_record_paths:
                failures.append(f"selected wheel native payload is absent from publisher RECORD: {item['path']}")
    if not publisher:
        sdist = row.get("sdist")
        if wheel_license_evidence:
            sdist_evidence = wheel_license_evidence
        elif not isinstance(sdist, dict):
            failures.append("missing primary LICENSE/NOTICE bytes and locked sdist fallback")
        else:
            fallback = fetch_artifact(sdist, temporary)
            sdist_evidence = archive_license_files(Path(fallback["temporary"]))
            if not sdist_evidence:
                failures.append("locked sdist has no primary LICENSE/NOTICE bytes")
    chosen = publisher or sdist_evidence
    reasons = ["metadata/classifiers are descriptive only; owner/legal must review literal primary bytes"]
    if not chosen:
        reasons.append("primary LICENSE/NOTICE bytes are missing or not yet collected")
    if native:
        reasons.append("native payload and ELF NEEDED rows require owner review")
    if row["name"].casefold() == "numpy":
        reasons.append("NumPy GPL/GCC/LGPL runtime components remain unresolved; do not infer approval from metadata")
    detail = {
        "name": dist.metadata.get("Name"),
        "version": dist.version,
        "identity": record["identity"],
        **metadata_fields(dist),
        "selected_artifact": {"kind": artifact_kind, "url": artifact["url"], "sha256": artifact["sha256"], "bytes": artifact["bytes"], "wheel_binding": {"wheel_path": wheel["path"], "wheel_sha256": wheel["sha256"], "installed_tags": wheel["tags"], "matched_tags": ["-".join(tag) for tag in sorted(matched_tags)]}},
        "installed_build_binding": installed_build_binding,
        "installed_payload_hashes": installed_payload,
        "wheel_license_notice_files": wheel_license_evidence,
        "archive_required_inventory": archive_inventory,
        "publisher_license_notice_files": publisher,
        "locked_sdist_license_fallback": sdist_evidence,
        "native_payload": {"files": native, "errors": native_failures},
        "policy_reasons": reasons,
        "factual_failures": failures,
    }
    license_row = {"name": row["name"], "version": row["version"], "license": "UNRESOLVED", "status": "CANDIDATE_OWNER_REVIEW", "reasons": reasons, "primary_bytes": [{"path": item["path"], "bytes": item["bytes"], "sha256": item["sha256"]} for item in chosen]}
    package_row = {"name": row["name"], "version": row["version"], "registry": row["source"]["registry"], "artifact": {"kind": artifact_kind, "url": artifact["url"], "sha256": artifact["sha256"], "bytes": artifact["bytes"]}, "license": "UNRESOLVED", "native_file_count": len(native)}
    return package_row, license_row, detail, failures


def collect(expected_head: str) -> dict[str, Any]:
    contract = audit.load_contract(expected_head)
    closure = contract["closure"]
    rows = {(row["name"].casefold(), row["version"]): row for row in contract["lock"]["package"]}
    real_keys = [key for key in closure if rows[key].get("source") != {"virtual": "."}]
    records: dict[str, list[dict[str, Any]]] = {}
    for dist in metadata.distributions():
        name, version = dist.metadata.get("Name"), dist.version
        if name and version:
            records.setdefault(identity(name, version), []).append({"distribution": dist, "identity": identity(name, version), "location": str(Path(dist.locate_file("")))})
    expected = {identity(*key) for key in real_keys}
    actual = Counter(key for key in records for _ in records[key])
    closure_facts = {"expected": sorted(expected), "installed": sorted(actual.elements()), "missing": sorted(expected - set(actual)), "unexpected": sorted(set(actual) - expected), "duplicates": sorted(key for key, count in actual.items() if count != 1), "exact": set(actual) == expected and all(count == 1 for count in actual.values())}
    package_rows: list[dict[str, Any]] = []
    license_rows: list[dict[str, Any]] = []
    package_evidence: list[dict[str, Any]] = []
    failures = [] if closure_facts["exact"] else ["installed distributions do not exactly match reachable Linux uv.lock closure"]
    with tempfile.TemporaryDirectory(prefix="xcodec2-dependency-evidence-") as temporary_name:
        temporary = Path(temporary_name)
        for key in real_keys:
            found = records.get(identity(*key), [])
            if len(found) != 1:
                failures.append(f"installed closure is not one-to-one: {identity(*key)}")
                continue
            try:
                package, license_row, detail, row_failures = inspect_distribution(rows[key], found[0], temporary)
                package_rows.append(package)
                license_rows.append(license_row)
                package_evidence.append(detail)
                failures.extend(f"{identity(*key)}: {failure}" for failure in row_failures)
            except (EvidenceError, OSError, ValueError) as exc:
                failures.append(f"{identity(*key)}: {exc}")
    return {
        "schema": "vokra-xcodec2-dependency-evidence-v1",
        "status": "BLOCKED_FACTUAL_COLLECTION" if failures else "BLOCKED_OWNER_REVIEW",
        "publication": "NO_UPLOAD",
        "git": contract["git"],
        "project": {**contract["hashes"], "closure_rows": len(closure), "external_rows": len(real_keys)},
        "lock": {"python": "==3.12.*", "platform": "linux-x86_64", "reachable_closure": [f"{name}=={version}" for name, version in closure], "closure": closure_facts},
        "package_rows": package_rows,
        "license_rows": license_rows,
        "package_rows_sha256": sha256_bytes(canonical(package_rows).encode()),
        "license_rows_sha256": sha256_bytes(canonical(license_rows).encode()),
        "package_evidence": package_evidence,
        "owner_review_blockers": sorted({"all license rows remain UNRESOLVED until owner/legal review", "NO_UPLOAD is mandatory", *[f"factual collection: {failure}" for failure in failures]}),
        "factual_failures": sorted(set(failures)),
        "model_activity": {"model_code_imported": False, "weights_acquired": False, "weights_imported": False, "weights_executed": False, "audio_acquired": False, "audio_imported": False, "audio_executed": False, "source_repo_downloaded": False, "cargo_invoked": False, "uploaded": False},
    }


def write_atomic(output: Path, report: dict[str, Any]) -> None:
    assert_safe_path(output)
    if output == audit.REPO_ROOT or audit.REPO_ROOT in output.parents or output == audit.PROJECT or audit.PROJECT in output.parents:
        raise EvidenceError("output must be outside the checkout")
    sidecar = Path(str(output) + ".sha256")
    assert_safe_path(sidecar)
    if output.exists() or output.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        raise EvidenceError("output and sidecar must be absent")
    output.parent.mkdir(parents=True, exist_ok=True)
    assert_safe_path(output.parent)
    payload = (canonical(report) + "\n").encode("utf-8")
    temporary = output.parent / f".{output.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        with sidecar.open("xb") as stream:
            stream.write(f"{sha256_bytes(payload)}  {output.name}\n".encode())
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="xcodec2-evidence-self-test-") as directory:
        root = Path(directory)
        archive = root / "demo.whl"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("demo/LICENSE", b"MIT\n")
        files = archive_license_files(archive)
        if files[0]["sha256"] != sha256_bytes(b"MIT\n"):
            raise AssertionError("license archive evidence mismatch")
        malicious = root / "bad.whl"
        with zipfile.ZipFile(malicious, "w") as handle:
            handle.writestr("../LICENSE", b"bad")
        try:
            archive_license_files(malicious)
        except EvidenceError:
            pass
        else:
            raise AssertionError("archive traversal accepted")
        body = b"payload"
        artifact = {"url": "https://files.pythonhosted.org/packages/demo-1-py3-none-any.whl", "hash": "sha256:" + sha256_bytes(body), "size": len(body)}
        result = fetch_artifact(artifact, root, lambda url: (url, body))
        if result["sha256"] != sha256_bytes(body):
            raise AssertionError("artifact evidence mismatch")
        try:
            fetch_artifact({**artifact, "hash": "sha256:" + "0" * 64}, root, lambda url: (url, body))
        except EvidenceError:
            pass
        else:
            raise AssertionError("artifact hash tamper accepted")
        link_target = root / "real-output"
        link_target.mkdir()
        link_parent = root / "linked-output"
        link_parent.symlink_to(link_target, target_is_directory=True)
        try:
            write_atomic(link_parent / "report.json", {"status": "BLOCKED_OWNER_REVIEW"})
        except EvidenceError:
            pass
        else:
            raise AssertionError("symlink output ancestry accepted")
    if "torch" in sys.modules or "xcodec2" in sys.modules:
        raise AssertionError("collector self-test imported a model package")
    print("xcodec2 dependency evidence collector: PASS (model-free self-test)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-head")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.project or args.output or args.expected_head:
            parser.error("--self-test accepts no project/output/head")
        return self_test()
    if args.project is None or args.output is None or args.expected_head is None:
        parser.error("--project, --output, and --expected-head are required")
    try:
        assert_safe_path(args.project)
    except EvidenceError:
        print("collector: BLOCKED: project path has unsafe or symlink ancestry", file=sys.stderr)
        return 2
    if not args.project.is_absolute() or args.project != audit.PROJECT:
        print("collector: BLOCKED: project must be the canonical xcodec2 project", file=sys.stderr)
        return 2
    try:
        report = collect(args.expected_head)
        write_atomic(args.output, report)
    except (EvidenceError, audit.AuditError, OSError, UnicodeError, ValueError) as exc:
        print(f"collector: BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(f"collector: {report['status']} ({args.output})", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
