#!/usr/bin/env python3
"""Model-free factual audit of the frozen MOSS Nano Python closure.

The audit reads the checked-in lock/manifest and the installed distribution
metadata only.  It never imports torch, Transformers, custom model code, or
weights.  Locked artifacts are reported by their resolver URL/hash and
resolver-supplied size; the official PyTorch CPU wheel index omits the Torch
wheel size, which remains unresolved rather than being invented. When a wheel
does not carry publisher license files, the exact locked PyPI sdist
is the only permitted fallback.  Native ELF payloads (including CUDA/NVIDIA
and Triton payloads) are hashed and inspected with ``readelf`` where present.
The report deliberately remains BLOCKED while the owner review rows are
unresolved.
"""
from __future__ import annotations

import argparse
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
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import tomllib

try:
    import license_gate
except ModuleNotFoundError:  # direct execution from repository root
    from tools.parity.moss_audio_tokenizer_nano import license_gate


SCHEMA = "vokra-moss-audio-tokenizer-nano-dependency-audit-v1"
PYPI_HOST = "files.pythonhosted.org"
ELF_MAGIC = b"\x7fELF"
NATIVE_SUFFIXES = {".so", ".dylib", ".dll", ".pyd"}
LICENSE_NAMES = {
    "license", "licence", "copying", "notice", "copyright", "eula",
    "nvidia_sla", "nvidia-sla", "end_user_license", "end-user-license",
}
NATIVE_FAMILIES = ("nvidia-", "torch", "triton")
CPU_TORCH_SOURCE = {"registry": "https://download.pytorch.org/whl/cpu"}
CPU_TORCH_VERSION = "2.7.1+cpu"
MAX_LICENSE_BYTES = 2 * 1024 * 1024
MAX_SDIST_BYTES = 64 * 1024 * 1024
MAX_MEMBER_BYTES = 8 * 1024 * 1024
MAX_TOTAL_MEMBER_BYTES = 64 * 1024 * 1024
MAX_LICENSE_TOTAL_BYTES = 4 * 1024 * 1024
MAX_MEMBERS = 10_000
MAX_REDIRECTS = 3


class AuditError(ValueError):
    """A factual or structural audit blocker."""


class SdistError(AuditError):
    def __init__(self, message: str, *, acquired: bool | None = None,
                 verified: bool = False, size: int | None = None,
                 digest: str | None = None) -> None:
        super().__init__(message)
        self.acquired = acquired
        self.verified = verified
        self.observed_size = size
        self.observed_sha256 = digest


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: Any) -> str:
    return sha256_bytes(canonical(value).encode())


def norm_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value.strip()).casefold()


def identity(name: str, version: str) -> str:
    return f"{norm_name(name)}=={re.sub(r'\\s+', '', version.strip()).casefold()}"


def strict_json(path: Path) -> Any:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AuditError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)
    except (OSError, UnicodeError, json.JSONDecodeError, AuditError) as exc:
        raise AuditError(f"cannot read JSON {path}: {exc}") from exc


def regular(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def assert_no_symlink_ancestors(path: Path) -> None:
    """Reject lexical path ancestry that can redirect the audit input/output."""
    if not path.is_absolute():
        raise AuditError(f"path must be absolute: {path}")
    current = Path("/")
    parts = path.parts[1:]
    for index, part in enumerate(parts):
        if not part or part in {".", ".."}:
            raise AuditError(f"path contains unsafe lexical component: {path}")
        current /= part
        if current.is_symlink():
            raise AuditError(f"path contains symlink ancestor: {current}")


def validate_project_path(project: Path) -> None:
    assert_no_symlink_ancestors(project)
    if not project.is_dir() or project.is_symlink():
        raise AuditError(f"project is not a real directory: {project}")


def package_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return row["name"], row["version"], canonical(row["source"])


def forbidden_accelerator_row(row: dict[str, Any]) -> str | None:
    """Return a blocker for any CUDA/NVIDIA/Triton lock identity."""
    name = norm_name(row.get("name", ""))
    if name.startswith("nvidia-"):
        return f"CUDA/NVIDIA distribution is forbidden in the CPU closure: {row.get('name')}=={row.get('version')}"
    if name == "triton":
        return f"Triton distribution is forbidden in the CPU closure: {row.get('name')}=={row.get('version')}"
    if name == "torch" and (row.get("version") != CPU_TORCH_VERSION or row.get("source") != CPU_TORCH_SOURCE):
        return f"torch is not the exact CPU wheel identity: {row.get('version')} from {row.get('source')}"
    return None


def validate_cpu_closure(rows: list[dict[str, Any]]) -> None:
    blockers = [reason for row in rows if (reason := forbidden_accelerator_row(row))]
    torch_rows = [row for row in rows if norm_name(row.get("name", "")) == "torch"]
    if len(torch_rows) != 1:
        blockers.append(f"CPU closure must contain exactly one torch row, found {len(torch_rows)}")
    if blockers:
        raise AuditError("; ".join(blockers))


def contract(project: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], bytes, bytes]:
    validate_project_path(project)
    project_path, lock_path, manifest_path = (project / name for name in ("pyproject.toml", "uv.lock", "license_gate_manifest.json"))
    for input_path in (project_path, lock_path, manifest_path):
        assert_no_symlink_ancestors(input_path)
    if not all(regular(path) for path in (project_path, lock_path, manifest_path)):
        raise AuditError("Nano project/lock/manifest is missing or symlinked")
    try:
        project_bytes, lock_bytes = project_path.read_bytes(), lock_path.read_bytes()
        project_data, lock_data = tomllib.loads(project_bytes.decode()), tomllib.loads(lock_bytes.decode())
        manifest = strict_json(manifest_path)
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise AuditError(f"closure is unreadable: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("gate_version") != 1:
        raise AuditError("gate manifest schema/version is unsupported")
    if sha256_bytes(project_bytes) != license_gate.PROJECT_SHA256 or sha256_bytes(lock_bytes) != license_gate.LOCK_SHA256:
        raise AuditError("project/lock bytes differ from the code-bound contract")
    try:
        rows = license_gate.lock_rows(lock_data)
        if license_gate.artifact_error(lock_data):
            raise AuditError(license_gate.artifact_error(lock_data) or "malformed resolver artifact")
        validate_cpu_closure(rows)
        license_gate.project_identity(project_bytes)
    except (SystemExit, ValueError) as exc:
        raise AuditError(f"closure schema is invalid: {exc}") from exc
    if manifest.get("lock_sha256") != license_gate.LOCK_SHA256 or manifest.get("project_sha256") != license_gate.PROJECT_SHA256:
        raise AuditError("manifest does not bind exact closure bytes")
    if manifest.get("package_rows") != rows or manifest.get("package_rows_sha256") != digest(rows):
        raise AuditError("manifest package rows do not bind exact lock rows")
    reviews = manifest.get("package_review_rows")
    if not isinstance(reviews, list) or len(reviews) != len(rows):
        raise AuditError("manifest package review rows do not cover every locked row")
    expected = {package_key(row) for row in rows}
    actual: set[tuple[str, str, str]] = set()
    for review in reviews:
        if not isinstance(review, dict) or set(review) != license_gate.PACKAGE_REVIEW_SCHEMA:
            raise AuditError("manifest package review row schema drifted")
        key = package_key(review)
        if key in actual or key not in expected:
            raise AuditError("manifest package review rows are not one-to-one with lock rows")
        actual.add(key)
    if actual != expected or manifest.get("package_review_rows_sha256") != digest(reviews):
        raise AuditError("manifest package review rows do not bind exact lock rows")
    return project_data, lock_data, manifest, rows, project_bytes, lock_bytes


def audit_rows(lock: dict[str, Any], canonical_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rebind resolver artifacts to the manifest-authenticated lock rows.

    ``license_gate.lock_rows`` intentionally strips ``sdist``/``wheels`` so
    its result is stable for the package-row manifest digest.  The dependency
    audit needs those resolver identities to fetch a locked sdist, however.
    ``artifact_error`` has already authenticated the raw package tables before
    this helper is called; this second exact identity check prevents an
    artifact from being attached to a different marker/dependency row.
    """
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise AuditError("lock package table is not a list for audit artifact binding")
    raw_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for package in packages:
        if not isinstance(package, dict):
            raise AuditError("lock package row is not an object for audit artifact binding")
        try:
            key = package_key(package)
        except (KeyError, TypeError):
            raise AuditError("lock package row has no exact audit identity") from None
        if key in raw_by_key:
            raise AuditError(f"duplicate raw lock identity for audit artifact binding: {key[:2]}")
        raw_by_key[key] = package

    canonical_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in canonical_rows:
        try:
            key = package_key(row)
        except (KeyError, TypeError):
            raise AuditError("canonical lock row has no exact audit identity") from None
        if key in canonical_by_key:
            raise AuditError(f"duplicate canonical lock identity for audit artifact binding: {key[:2]}")
        canonical_by_key[key] = row
    if set(raw_by_key) != set(canonical_by_key):
        raise AuditError("raw and canonical lock identities differ for audit artifact binding")

    result: list[dict[str, Any]] = []
    for key, canonical_row in canonical_by_key.items():
        raw = raw_by_key[key]
        for field in ("name", "version", "source", "resolution-markers", "dependencies"):
            raw_value = raw.get(field, []) if field in {"resolution-markers", "dependencies"} else raw.get(field)
            if raw_value != canonical_row.get(field):
                raise AuditError(f"raw/canonical lock {field} differs for audit identity: {key[:2]}")
        merged = dict(canonical_row)
        if raw.get("source") != {"virtual": "."}:
            if "sdist" not in raw and "wheels" not in raw:
                raise AuditError(f"raw lock artifacts are incomplete for audit identity: {key[:2]}")
            for artifact_key in ("sdist", "wheels"):
                if artifact_key in raw:
                    merged[artifact_key] = raw[artifact_key]
        result.append(merged)
    return sorted(result, key=lambda row: (row["name"], row["version"]))


def compare_multiset(expected: list[str], actual: list[str]) -> dict[str, Any]:
    want, got = Counter(expected), Counter(actual)
    return {
        "expected": sorted(expected), "installed": sorted(actual),
        "missing": sorted((want - got).elements()),
        "unexpected": sorted((got - want).elements()),
        "duplicate_identities": sorted(key for key, count in got.items() if count > 1),
        "exact": not (want - got or got - want),
    }


def active_marker(row: dict[str, Any]) -> bool:
    markers = row.get("resolution-markers", [])
    return not markers or "platform_machine == 'x86_64' and sys_platform == 'linux'" in markers


def installed_distributions() -> list[dict[str, Any]]:
    result = []
    for dist in metadata.distributions():
        name, version = dist.metadata.get("Name"), dist.version
        if name and version:
            result.append({"distribution": dist, "name": name, "version": version,
                           "identity": identity(name, version),
                           "location": str(Path(dist.locate_file("")))})
    return sorted(result, key=lambda item: (item["identity"], item["location"]))


def metadata_fields(dist: metadata.Distribution) -> dict[str, Any]:
    classifiers = sorted(
        value.removeprefix("License :: ")
        for value in (dist.metadata.get_all("Classifier") or [])
        if value.startswith("License :: ")
    )
    clean = lambda value: value.strip() if isinstance(value, str) and value.strip() else None
    return {
        "license": clean(dist.metadata.get("License")),
        "license_expression": clean(dist.metadata.get("License-Expression")),
        "license_classifiers": classifiers,
    }


def safe_installed_path(dist: metadata.Distribution, entry: Any) -> Path | None:
    root = Path(dist.locate_file(""))
    lexical = Path(dist.locate_file(entry))
    try:
        relative = lexical.relative_to(root)
    except ValueError:
        return None
    if root.is_symlink() or any((root / part).is_symlink() for part in relative.parts):
        return None
    resolved_root, resolved = root.resolve(strict=False), lexical.resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        return None
    return resolved


def license_candidate(value: str) -> bool:
    name = PurePosixPath(value).name.casefold()
    return name in LICENSE_NAMES or any(name.startswith(prefix + separator) for prefix in LICENSE_NAMES for separator in (".", "-", "_"))


def publisher_files(dist: metadata.Distribution) -> tuple[list[dict[str, Any]], list[str]]:
    files, unsafe = [], []
    for entry in sorted(dist.files or [], key=str):
        relative = str(entry)
        if not license_candidate(relative):
            continue
        path = safe_installed_path(dist, entry)
        if path is None or path.is_symlink() or not path.is_file():
            unsafe.append(relative)
            continue
        files.append({"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)})
    return files, unsafe


def elf_details(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            if handle.read(4) != ELF_MAGIC:
                return {"format": "non-elf", "needed": [], "inspection": "not-applicable"}
    except OSError as exc:
        return {"format": "unknown", "needed": [], "inspection": "error", "error": str(exc)}
    try:
        result = subprocess.run(["readelf", "-d", str(path)], capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"format": "elf", "needed": [], "inspection": "error", "error": type(exc).__name__}
    needed = sorted(match.group(1) for line in result.stdout.splitlines() if (match := re.search(r"\(NEEDED\).*\[([^]]+)\]", line)))
    status = "ok" if result.returncode == 0 else "error"
    return {"format": "elf", "needed": needed, "inspection": status, "readelf_returncode": result.returncode}


def native_files(dist: metadata.Distribution) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    files, errors = [], []
    family = norm_name(dist.metadata.get("Name", ""))
    family_kind = "nvidia-cuda" if family.startswith("nvidia-") else "torch" if family == "torch" else "triton" if family == "triton" else "other"
    for entry in sorted(dist.files or [], key=str):
        relative = str(entry)
        name = Path(relative).name.casefold()
        candidate = Path(relative).suffix.casefold() in NATIVE_SUFFIXES or ".so." in name
        path = safe_installed_path(dist, entry)
        if path is None or path.is_symlink() or not path.is_file():
            if candidate:
                errors.append({"path": relative, "stage": "path", "error": "native candidate is not a regular file"})
            continue
        try:
            with path.open("rb") as handle:
                magic = handle.read(4)
            if not candidate and magic != ELF_MAGIC:
                continue
            files.append({"path": relative, "size": path.stat().st_size,
                          "sha256": sha256_file(path), "bundled": True,
                          "family": family_kind, "candidate": "native-suffix" if candidate else "elf-magic",
                          "elf": elf_details(path)})
        except OSError as exc:
            errors.append({"path": relative, "stage": "read", "error": str(exc)})
    return files, errors


def safe_member(name: str) -> str:
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name or name.startswith("/"):
        raise AuditError(f"unsafe archive path: {name!r}")
    clean = name.rstrip("/")
    if not clean or any(part in {"", ".", ".."} for part in clean.split("/")):
        raise AuditError(f"archive path traversal: {name!r}")
    return clean


def _read_tar(handle: tarfile.TarFile, info: tarfile.TarInfo) -> bytes:
    stream = handle.extractfile(info)
    if stream is None:
        raise AuditError(f"cannot read archive license member: {info.name}")
    with stream:
        return stream.read(MAX_LICENSE_BYTES + 1)


def archive_license_files(url: str, body: bytes) -> tuple[str, list[dict[str, Any]]]:
    if len(body) > MAX_SDIST_BYTES:
        raise AuditError("sdist archive exceeds bounded size")
    lower = urlsplit(url).path.casefold()
    if lower.endswith((".tar.gz", ".tgz")):
        mode = ("tar.gz", "r:gz")
    elif lower.endswith((".tar.bz2", ".tbz2")):
        mode = ("tar.bz2", "r:bz2")
    elif lower.endswith((".tar.xz", ".txz")):
        mode = ("tar.xz", "r:xz")
    elif lower.endswith(".tar"):
        mode = ("tar", "r:")
    elif lower.endswith(".zip"):
        mode = ("zip", "zip")
    else:
        raise AuditError("unsupported locked sdist archive type")
    names: set[str] = set(); total = 0; license_total = 0; result = []
    def member(name: str, size: int, regular_file: bool, reader: Callable[[], bytes]) -> None:
        nonlocal total, license_total
        clean = safe_member(name)
        if clean in names:
            raise AuditError(f"duplicate archive member: {clean}")
        names.add(clean)
        if len(names) > MAX_MEMBERS or size < 0 or size > MAX_MEMBER_BYTES:
            raise AuditError("archive member/count bound exceeded")
        if not regular_file:
            return
        total += size
        if total > MAX_TOTAL_MEMBER_BYTES:
            raise AuditError("archive aggregate bound exceeded")
        if not license_candidate(clean):
            return
        if size > MAX_LICENSE_BYTES:
            raise AuditError("license member bound exceeded")
        data = reader()
        if len(data) != size:
            raise AuditError("archive member size changed")
        license_total += size
        if license_total > MAX_LICENSE_TOTAL_BYTES:
            raise AuditError("license aggregate bound exceeded")
        result.append({"path": clean, "size": size, "sha256": sha256_bytes(data), "content_base64": base64.b64encode(data).decode("ascii")})
    try:
        if mode[0] == "zip":
            with zipfile.ZipFile(io.BytesIO(body)) as archive:
                for info in archive.infolist():
                    kind = (info.external_attr >> 16) & 0o170000
                    directory = info.is_dir() or info.filename.endswith("/") or kind == stat.S_IFDIR
                    if kind and kind not in {stat.S_IFREG, stat.S_IFDIR}:
                        raise AuditError(f"zip link/special member: {info.filename!r}")
                    member(info.filename, 0 if directory else info.file_size, not directory, lambda info=info: archive.read(info))
        else:
            with tarfile.open(fileobj=io.BytesIO(body), mode=mode[1]) as archive:
                for info in archive:
                    if info.issym() or info.islnk() or info.isdev() or info.isfifo() or (not info.isdir() and not info.isfile()):
                        raise AuditError(f"tar link/special member: {info.name!r}")
                    member(info.name, info.size if info.isfile() else 0, info.isfile(), lambda info=info: _read_tar(archive, info))
    except (OSError, EOFError, RuntimeError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise AuditError(f"archive inspection failed: {exc}") from exc
    if not result:
        raise AuditError("locked sdist contains no license/EULA candidate")
    return mode[0], sorted(result, key=lambda item: item["path"])


def validate_pypi_url(value: str, expected: str, *, initial: bool) -> None:
    parsed = urlsplit(value)
    if (parsed.scheme != "https" or parsed.hostname != PYPI_HOST or parsed.username or parsed.password
            or parsed.port not in (None, 443) or parsed.query or parsed.fragment or not parsed.path):
        raise AuditError("sdist URL escaped files.pythonhosted.org")
    if initial and value != expected:
        raise AuditError("initial sdist URL differs from lock")
    if not initial and parsed.path != urlsplit(expected).path:
        raise AuditError("redirect changed locked sdist path")


class SdistRedirects(HTTPRedirectHandler):
    def __init__(self, trace: list[str], expected: str) -> None:
        self.trace, self.expected = trace, expected
    def redirect_request(self, request: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Request | None:
        if len(self.trace) - 1 >= MAX_REDIRECTS:
            raise AuditError("sdist redirect limit exceeded")
        resolved = urljoin(request.full_url, newurl)
        validate_pypi_url(resolved, self.expected, initial=False)
        self.trace.append(resolved)
        return super().redirect_request(request, fp, code, msg, headers, resolved)


def fetch_locked_sdist(row: dict[str, Any], fetcher: Callable[[str], tuple[str, bytes]] | None = None) -> dict[str, Any]:
    if row.get("source") != {"registry": "https://pypi.org/simple"}:
        raise SdistError("sdist fallback is restricted to PyPI", acquired=None)
    artifact = row.get("sdist")
    if not isinstance(artifact, dict) or set(artifact) != license_gate.ARTIFACT_KEYS:
        raise SdistError("exact locked sdist artifact is unavailable", acquired=None)
    url, expected_hash, expected_size = artifact.get("url"), artifact.get("hash"), artifact.get("size")
    if (not isinstance(url, str) or not isinstance(expected_hash, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_hash)
            or isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size <= 0 or expected_size > MAX_SDIST_BYTES):
        raise SdistError("malformed or unbounded exact sdist identity", acquired=None)
    validate_pypi_url(url, url, initial=True)
    trace = [url]
    try:
        if fetcher is None:
            opener = build_opener(SdistRedirects(trace, url))
            request = Request(url, headers={"Accept": "application/octet-stream", "User-Agent": "vokra-moss-nano-audit/1"})
            with opener.open(request, timeout=30) as response:
                final = urljoin(url, response.geturl()); validate_pypi_url(final, url, initial=False)
                if final != trace[-1]: trace.append(final)
                length = response.headers.get("Content-Length")
                if length and length.isdigit() and int(length) > MAX_SDIST_BYTES:
                    raise SdistError("Content-Length exceeds bound", acquired=None)
                body = response.read(MAX_SDIST_BYTES + 1)
        else:
            final, body = fetcher(url); final = urljoin(url, final); validate_pypi_url(final, url, initial=False)
            if final != trace[-1]: trace.append(final)
    except SdistError:
        raise
    except (HTTPError, URLError, OSError, UnicodeError, ValueError) as exc:
        raise SdistError(type(exc).__name__, acquired=None) from exc
    if not isinstance(body, bytes):
        raise SdistError("fetcher did not return bytes", acquired=None)
    observed = sha256_bytes(body)
    if len(body) != expected_size:
        raise SdistError("sdist size mismatch", acquired=True, size=len(body), digest="sha256:" + observed)
    if "sha256:" + observed != expected_hash:
        raise SdistError("sdist SHA-256 mismatch", acquired=True, size=len(body), digest="sha256:" + observed)
    try:
        archive_format, files = archive_license_files(final, body)
    except AuditError as exc:
        raise SdistError(str(exc), acquired=True, verified=True, size=len(body), digest="sha256:" + observed) from exc
    return {"status": "ACQUIRED_LOCKED_SDIST_LICENSE_BYTES", "package": row["name"], "version": row["version"],
            "archive": {"url": url, "final_url": final, "url_trace": trace, "size": expected_size, "hash": expected_hash, "format": archive_format},
            "license_files": files}


def blocked_sdist(row: dict[str, Any], exc: Exception) -> dict[str, Any]:
    artifact = row.get("sdist") if isinstance(row.get("sdist"), dict) else {}
    return {"status": "BLOCKED_FACTUAL_SDIST_LICENSE_PATH", "package": row.get("name"), "version": row.get("version"),
            "requested_url": artifact.get("url"), "archive": {"url": artifact.get("url"), "size": artifact.get("size"), "hash": artifact.get("hash"),
            "observed_size": getattr(exc, "observed_size", None), "observed_sha256": getattr(exc, "observed_sha256", None)},
            "acquired_archive_bytes": getattr(exc, "acquired", None), "verified_archive": bool(getattr(exc, "verified", False)),
            "license_files": [], "error": type(exc).__name__}


def inspect_package(row: dict[str, Any], record: dict[str, Any] | None, review: dict[str, Any] | None,
                    sdist_fetcher: Callable[[str], tuple[str, bytes]] | None) -> tuple[dict[str, Any], list[str]]:
    base = {"lock": {key: row[key] for key in ("name", "version", "source")}, "review": review}
    if record is None:
        return {**base, "installed": None}, [f"installed closure missing: {identity(row['name'], row['version'])}"]
    dist = record["distribution"]
    publisher, unsafe = publisher_files(dist)
    native, native_errors = native_files(dist)
    fields = metadata_fields(dist)
    sdist = None
    if not publisher:
        try:
            sdist = fetch_locked_sdist(row, sdist_fetcher)
        except (HTTPError, URLError, OSError, UnicodeError, ValueError) as exc:
            sdist = blocked_sdist(row, exc)
    native_family = norm_name(dist.metadata.get("Name", ""))
    native_scope = "cuda/nvidia" if native_family.startswith("nvidia-") else "torch" if native_family == "torch" else "triton" if native_family == "triton" else "other"
    native_status = "NATIVE_PAYLOAD_SCANNED" if native else "NO_NATIVE_PAYLOAD"
    eula_files = [item for item in publisher if license_candidate(item["path"])]
    installed = {
        "name": dist.metadata.get("Name"), "version": dist.version, "normalized_identity": record["identity"],
        **fields,
        "publisher_license_eula_files": publisher,
        "unsafe_publisher_paths": unsafe,
        "sdist_license_evidence": sdist,
        "native_payload": {"scope": native_scope, "status": native_status, "files": native, "errors": native_errors,
                           "eula_status": "PRESENT_IN_PACKAGE" if eula_files or (sdist and sdist.get("license_files")) else "ABSENT_PUBLISHER_AND_SDIST"},
    }
    failures: list[str] = []
    has_license = bool(fields["license"] or fields["license_expression"] or fields["license_classifiers"])
    verified_sdist = bool(sdist and sdist.get("status") == "ACQUIRED_LOCKED_SDIST_LICENSE_BYTES" and sdist.get("license_files"))
    if not has_license and not verified_sdist:
        failures.append(f"missing package license metadata: {record['identity']}")
    if not publisher and not verified_sdist:
        failures.append(f"missing publisher LICENSE/EULA evidence: {record['identity']}")
        if sdist:
            failures.append(f"{sdist['status']}: {record['identity']}")
    failures.extend(f"unsafe publisher path: {record['identity']}:{path}" for path in unsafe)
    failures.extend(f"native candidate inspection failed: {record['identity']}:{error['path']}" for error in native_errors)
    failures.extend(f"ELF NEEDED inspection failed: {record['identity']}" for item in native if item["elf"].get("inspection") == "error")
    # Native CUDA/NVIDIA/Triton payloads must expose a license/EULA fact.  A
    # blocked package row is intentional until an owner reviews this evidence.
    if native and not eula_files and not verified_sdist:
        failures.append(f"native payload license/EULA evidence missing: {record['identity']}")
    return {**base, "lock": {**base["lock"], "artifacts": {"sdist": row.get("sdist"), "wheels": row.get("wheels", [])}}, "installed": installed}, failures


def approval_state(manifest: dict[str, Any]) -> dict[str, Any]:
    package_blockers = []
    for row in manifest.get("package_review_rows", []) if isinstance(manifest.get("package_review_rows"), list) else []:
        if isinstance(row, dict) and (row.get("status") != "REVIEWED" or row.get("license") in {None, "UNRESOLVED"} or row.get("native_bundled_review") in {None, "OWNER_REVIEW_REQUIRED"}):
            package_blockers.append({key: row.get(key) for key in ("name", "version", "status", "license", "native_bundled_review")})
    license_blockers = []
    for row in manifest.get("license_rows", []) if isinstance(manifest.get("license_rows"), list) else []:
        if isinstance(row, dict) and (row.get("status") != "REVIEWED" or row.get("license") in {None, "UNRESOLVED"} or row.get("conclusion") in {None, "UNRESOLVED"}):
            license_blockers.append({key: row.get(key) for key in ("id", "status", "license", "conclusion")})
    approval = manifest.get("approval")
    approval_blockers = [] if isinstance(approval, dict) and approval.get("status") == "OWNER_SIGNOFF_APPROVED" else [{"required": "OWNER_SIGNOFF_APPROVED"}]
    return {"package_review_blockers": package_blockers, "license_review_blockers": license_blockers,
            "approval": approval, "approval_blockers": approval_blockers,
            "publication_decision": manifest.get("publication_decision"), "publication_permitted": False}


def blocked_approval_state(project: Path) -> dict[str, Any]:
    try:
        value = strict_json(project / "license_gate_manifest.json")
        if isinstance(value, dict):
            return approval_state(value)
    except (AuditError, OSError, UnicodeError, ValueError):
        pass
    return {"package_review_blockers": [], "license_review_blockers": [], "approval": None,
            "approval_blockers": [{"reason": "manifest unavailable"}], "publication_decision": None,
            "publication_permitted": False}


def git_identity(project: Path, expected_head: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", expected_head):
        raise AuditError("--expected-head must be lowercase 40-hex")
    root = project.resolve().parents[2]
    try:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10)
        clean = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise AuditError(f"cannot bind git commit: {type(exc).__name__}") from exc
    commit = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise AuditError("invalid git commit identity")
    status = clean.stdout
    if commit != expected_head:
        raise AuditError(f"VAST checkout HEAD differs from --expected-head: {commit} != {expected_head}")
    if status:
        raise AuditError("VAST checkout is not clean")
    return {"expected_head": expected_head, "head": commit, "clean": True,
            "status_porcelain": status, "audit_script_sha256": sha256_file(Path(__file__))}


def audit_environment(project: Path, expected_head: str,
                      sdist_fetcher: Callable[[str], tuple[str, bytes]] | None = None) -> dict[str, Any]:
    project_data, lock, manifest, canonical_rows, project_bytes, lock_bytes = contract(project)
    bound_rows = audit_rows(lock, canonical_rows)
    real_rows = [row for row in bound_rows if row.get("source") != {"virtual": "."}]
    active_rows = [row for row in real_rows if active_marker(row)]
    inactive_rows = [row for row in real_rows if not active_marker(row)]
    records = installed_distributions()
    by_id: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_id.setdefault(record["identity"], []).append(record)
    closure = compare_multiset([identity(row["name"], row["version"]) for row in active_rows], [record["identity"] for record in records])
    closure.update({"lock_rows": len(canonical_rows), "active_distributions": len(active_rows),
                    "virtual_project_rows": sum(row.get("source") == {"virtual": "."} for row in lock.get("package", []))})
    reviews = {(row.get("name"), row.get("version")): row for row in manifest.get("package_review_rows", []) if isinstance(row, dict)}
    packages, failures = [], []
    for record in records:
        name = norm_name(record["name"])
        if name.startswith("nvidia-"):
            failures.append(f"CUDA/NVIDIA distribution is installed in the CPU closure: {record['identity']}")
        elif name == "triton":
            failures.append(f"Triton distribution is installed in the CPU closure: {record['identity']}")
        elif name == "torch" and record["version"] != CPU_TORCH_VERSION:
            failures.append(f"installed torch is not the exact CPU wheel identity: {record['identity']}")
    for review in manifest.get("package_review_rows", []):
        if isinstance(review, dict) and (review.get("status") != "REVIEWED" or review.get("license") in {None, "UNRESOLVED"} or review.get("native_bundled_review") in {None, "OWNER_REVIEW_REQUIRED"}):
            failures.append(f"manifest package review remains unresolved: {review.get('name')}=={review.get('version')}")
    for row in active_rows:
        key = identity(row["name"], row["version"]); candidates = by_id.get(key, [])
        item, errors = inspect_package(row, candidates[0] if len(candidates) == 1 else None, reviews.get((row["name"], row["version"])), sdist_fetcher)
        if len(candidates) > 1:
            errors.append(f"duplicate installed distribution: {key}")
        packages.append(item); failures.extend(errors)
    for row in inactive_rows:
        packages.append({"lock": row, "installed": None, "activity": {"status": "INACTIVE_LOCK_ROW", "reason": "resolution marker is not Linux x86_64"}})
    virtual = next((row for row in lock.get("package", []) if row.get("source") == {"virtual": "."}), None)
    packages.append({"lock": virtual, "installed": None, "activity": {"status": "INACTIVE_VIRTUAL_PROJECT", "reason": "virtual project row is not an installed distribution"}})
    if not closure["exact"]:
        failures.append("installed distribution multiset does not exactly match uv.lock")
    if sys.platform != "linux" or platform.machine().casefold() not in {"x86_64", "amd64"}:
        failures.append("audit host is not Linux x86_64")
    if sys.version_info[:2] != (3, 12):
        failures.append(f"Python runtime is not 3.12: {platform.python_version()}")
    return {
        "schema": SCHEMA, "status": "BLOCKED" if failures else "PASS",
        "environment": {"python": platform.python_version(), "platform": sys.platform, "machine": platform.machine(),
                        "readelf_required": True, "model_code_imported": False, "weights_acquired": False,
                        "weights_imported": False, "weights_executed": False, "cargo_invoked": False},
        "project": {"name": project_data["project"]["name"], "version": project_data["project"]["version"],
                    "pyproject_bytes": len(project_bytes), "pyproject_sha256": sha256_bytes(project_bytes),
                    "uv_lock_bytes": len(lock_bytes), "uv_lock_sha256": sha256_bytes(lock_bytes)},
        "git": git_identity(project, expected_head), "locked_rows": sorted(lock["package"], key=lambda row: (norm_name(row["name"]), row["version"])),
        "active_lock_rows": active_rows, "inactive_rows": inactive_rows,
        "closure": closure, "packages": packages, "approval_state": approval_state(manifest),
        "source_license_contract": {"status": "AUTHENTICATED_MODEL_CARD_LICENSE_NO_REPO_LICENSE_FILE", "repo": license_gate.REPO, "revision": license_gate.REVISION, "file": None},
        "model_acquisition": {"policy": "no model files requested", "requested_files": [], "non_license_files": [], "proof": "dependency audit does not import or acquire model weights"},
        "dependency_acquisition": {"scope": "exact lock artifact metadata and publisher/locked-sdist license-EULA evidence", "model_files": [], "non_license_files": [],
                                   "native_payload_scope": ["CUDA/NVIDIA", "Torch", "Triton"]},
        "failures": sorted(set(failures)),
    }


def write_report_no_replace(output: Path, payload: bytes) -> None:
    """Create a report exactly once, preserving a concurrent creator's file."""
    if not output.is_absolute():
        raise AuditError("report output must be absolute")
    assert_no_symlink_ancestors(output.parent)
    if output.parent.exists() and (output.parent.is_symlink() or not output.parent.is_dir()):
        raise AuditError("report output parent is not a real directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    assert_no_symlink_ancestors(output.parent)
    try:
        descriptor = os.open(str(output), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError as exc:
        raise AuditError("report output was created concurrently or already exists") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        # Keep a partial report rather than unlinking a path whose ownership
        # could be ambiguous after a race; callers can inspect/remove it.
        raise


def run(project: Path, output: Path, expected_head: str) -> int:
    if output.exists() or output.is_symlink():
        print("moss Nano dependency audit: BLOCKED: output must be absent", file=sys.stderr)
        return 2
    try:
        report = audit_environment(project, expected_head)
    except (AuditError, OSError, UnicodeError, ValueError) as exc:
        report = {"schema": SCHEMA, "status": "BLOCKED", "environment": {"model_code_imported": False, "weights_acquired": False, "weights_executed": False, "cargo_invoked": False},
                  "approval_state": blocked_approval_state(project), "locked_rows": [], "packages": [], "failures": [str(exc)]}
    try:
        write_report_no_replace(output, (canonical(report) + "\n").encode())
    except (AuditError, OSError) as exc:
        print(f"moss Nano dependency audit: BLOCKED: {exc}", file=sys.stderr)
        return 2
    if report["failures"]:
        print("moss Nano dependency audit: BLOCKED: " + "; ".join(report["failures"]), file=sys.stderr)
        return 2
    print(f"moss Nano dependency audit: PASS ({output})")
    return 0


def self_test() -> int:
    project = Path(__file__).resolve().parent
    _, lock, manifest, rows, _, _ = contract(project)
    if len(rows) != 35 or len(manifest["package_review_rows"]) != 35 or sum(row.get("source") == {"virtual": "."} for row in lock["package"]) != 1:
        raise SystemExit("self-test expected 34 active distributions plus one virtual project row")
    torch_row = next((row for row in rows if norm_name(row["name"]) == "torch"), None)
    if torch_row is None or torch_row["version"] != CPU_TORCH_VERSION or torch_row["source"] != CPU_TORCH_SOURCE:
        raise SystemExit("self-test lost the locked torch 2.7.1+cpu identity")
    if any(forbidden_accelerator_row(row) for row in rows):
        raise SystemExit("self-test found a forbidden CUDA/NVIDIA/Triton lock row")
    for forbidden in ("nvidia-cublas-cu12", "triton"):
        if forbidden_accelerator_row({"name": forbidden, "version": "1", "source": {"registry": "https://pypi.org/simple"}}) is None:
            raise SystemExit(f"self-test accepted forbidden distribution: {forbidden}")
    if forbidden_accelerator_row({"name": "torch", "version": "2.7.1+cu126", "source": {"registry": "https://download.pytorch.org/whl/cu126"}}) is None:
        raise SystemExit("self-test accepted CUDA torch identity")
    audit_source = {"registry": "https://pypi.org/simple"}
    audit_canonical = [{"name": "tokenizers", "version": "0.22.2", "source": audit_source,
                        "resolution-markers": [], "dependencies": []}]
    audit_artifact = {"url": f"https://{PYPI_HOST}/packages/tokenizers-0.22.2.tar.gz",
                      "hash": "sha256:" + "a" * 64, "size": 1,
                      "upload-time": "2026-01-01T00:00:00Z"}
    audit_raw = {"name": audit_canonical[0]["name"], "version": audit_canonical[0]["version"],
                 "source": audit_source, "sdist": audit_artifact, "wheels": []}
    if "sdist" in audit_canonical[0] or "wheels" in audit_canonical[0]:
        raise SystemExit("self-test canonical lock rows unexpectedly carry resolver artifacts")
    bound = audit_rows({"package": [audit_raw]}, audit_canonical)
    if bound != [{**audit_canonical[0], "sdist": audit_artifact, "wheels": []}]:
        raise SystemExit("self-test failed to restore raw resolver artifacts")
    for altered in (
        {**audit_raw, "resolution-markers": ["tampered"]},
        {**audit_raw, "dependencies": [{"name": "wrong", "marker": "always"}]},
    ):
        try:
            audit_rows({"package": [altered]}, audit_canonical)
        except AuditError:
            pass
        else:
            raise SystemExit("self-test accepted a raw/canonical lock identity mismatch")
    try:
        audit_rows({"package": [audit_raw, dict(audit_raw)]}, audit_canonical)
    except AuditError:
        pass
    else:
        raise SystemExit("self-test accepted a duplicate raw lock identity")
    try:
        audit_rows({"package": []}, audit_canonical)
    except AuditError:
        pass
    else:
        raise SystemExit("self-test accepted a missing raw lock identity")
    state = approval_state(manifest)
    if not state["package_review_blockers"] or not state["license_review_blockers"] or state["publication_permitted"]:
        raise SystemExit("self-test did not preserve fail-closed owner boundaries")
    if not compare_multiset(["a==1"], ["a==1"])["exact"]:
        raise SystemExit("self-test multiset equality failed")
    mismatch = compare_multiset(["a==1"], ["a==2"])
    if mismatch["missing"] != ["a==1"] or mismatch["unexpected"] != ["a==2"]:
        raise SystemExit("self-test multiset mismatch failed")
    for bad in ("../LICENSE", "/LICENSE", "a\\LICENSE", "a/./LICENSE"):
        try:
            safe_member(bad)
        except AuditError:
            pass
        else:
            raise SystemExit(f"self-test accepted unsafe archive path: {bad}")
    body = io.BytesIO()
    with tarfile.open(fileobj=body, mode="w:gz") as archive:
        data = b"Apache-2.0\n"; info = tarfile.TarInfo("demo/LICENSE"); info.size = len(data); archive.addfile(info, io.BytesIO(data))
    raw = body.getvalue()
    fake = {"name": "tokenizers", "version": "0.22.2", "source": {"registry": "https://pypi.org/simple"},
            "sdist": {"url": f"https://{PYPI_HOST}/packages/tokenizers-0.22.2.tar.gz", "hash": "sha256:" + sha256_bytes(raw), "size": len(raw), "upload-time": "2026-01-01T00:00:00Z"}}
    evidence = fetch_locked_sdist(fake, lambda url: (url, raw))
    if evidence["license_files"][0]["sha256"] != sha256_bytes(b"Apache-2.0\n"):
        raise SystemExit("self-test exact sdist license bytes failed")
    try:
        fetch_locked_sdist({**fake, "sdist": {**fake["sdist"], "hash": "sha256:" + "0" * 64}}, lambda url: (url, raw))
    except SdistError:
        pass
    else:
        raise SystemExit("self-test accepted sdist hash tamper")
    try:
        fetch_locked_sdist(fake, lambda url: ("https://evil.invalid/tokenizers-0.22.2.tar.gz", raw))
    except AuditError:
        pass
    else:
        raise SystemExit("self-test accepted unsafe sdist redirect")
    temporary = Path(tempfile.mkdtemp(prefix="moss-nano-audit-self-test-"))
    try:
        existing = temporary / "existing.json"
        existing.write_bytes(b"owner-created")
        try:
            write_report_no_replace(existing, b"audit-created")
        except AuditError:
            pass
        else:
            raise SystemExit("self-test accepted a concurrent/no-replace output race")
        if existing.read_bytes() != b"owner-created":
            raise SystemExit("self-test clobbered the pre-existing output")
        linked_parent = temporary / "linked-parent"
        linked_parent.symlink_to(temporary, target_is_directory=True)
        try:
            write_report_no_replace(linked_parent / "report.json", b"blocked")
        except AuditError:
            pass
        else:
            raise SystemExit("self-test accepted a symlink output parent")
        project_link = temporary / "project-link"
        project_link.symlink_to(project, target_is_directory=True)
        try:
            contract(project_link)
        except AuditError:
            pass
        else:
            raise SystemExit("self-test accepted a symlink project path")
    finally:
        shutil.rmtree(temporary)
    print("moss Nano dependency_audit self-test: OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-head", type=str)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.project or args.output or args.expected_head:
            parser.error("--self-test accepts no project/output")
        return self_test()
    if args.project is None or args.output is None or args.expected_head is None:
        parser.error("--project, --output, and --expected-head are required")
    return run(args.project, args.output, args.expected_head)


if __name__ == "__main__":
    raise SystemExit(main())
