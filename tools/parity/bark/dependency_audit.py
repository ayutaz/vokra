#!/usr/bin/env python3
"""Model-free factual audit for the pinned Bark reference environment.

This module is deliberately independent of Bark, Transformers and Vokra
runtime imports.  It is run only after an authorised VAST frozen sync and
inspects ``importlib.metadata`` records, not model code or model weights.
The only network operations are allow-listed requests for the two upstream
``LICENSE`` files and exact locked PyPI sdists needed for missing wheel
publisher files. Their bytes are measured and retained in the report as
primary-source hashes; no license class is inferred from a package expectation.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import importlib.metadata as metadata
import io
import json
import platform
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
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import tomllib

try:
    import license_gate
except ModuleNotFoundError:  # pragma: no cover - package invocation fallback
    from tools.parity.bark import license_gate


SCHEMA = "vokra-bark-dependency-audit-v1"
MAX_LICENSE_BYTES = 2 * 1024 * 1024
MAX_SDIST_BYTES = 64 * 1024 * 1024
MAX_SDIST_MEMBER_BYTES = 8 * 1024 * 1024
MAX_SDIST_TOTAL_MEMBER_BYTES = 64 * 1024 * 1024
MAX_SDIST_LICENSE_TOTAL_BYTES = 4 * 1024 * 1024
MAX_SDIST_MEMBERS = 10_000
MAX_SDIST_REDIRECTS = 3
ELF_MAGIC = b"\x7fELF"
NATIVE_SUFFIXES = {".so", ".dylib", ".dll", ".pyd"}
LICENSE_FILE_NAMES = {"license", "copying", "notice", "copyright"}
SDIST_ARTIFACT_KEYS = {"url", "hash", "size", "upload-time"}
PYPI_FILE_HOST = "files.pythonhosted.org"
LICENSE_CDN_HOSTS = {
    "cdn-lfs.huggingface.co",
    "cdn-lfs-us-1.hf.co",
}

# These are the fixed upstream identities already present in the Bark worker
# and gate manifest.  No source-code revision is invented here: the source
# contract is inspected separately and remains blocked when absent.
MODEL_LICENSES = (
    {
        "id": "bark-small-weight-mit",
        "repo": "suno/bark-small",
        "revision": "1dbd7a128513b8ae4a4e2130fed57b7ac9da5bcd",
    },
    {
        "id": "bark-full-weight-mit",
        "repo": "suno/bark",
        "revision": "70a8a7d34168586dc5d028fa9666aceade177992",
    },
)
MODEL_REPO_FIELDS = {
    "small_upstream_repo": "suno/bark-small",
    "small_upstream_revision": MODEL_LICENSES[0]["revision"],
    "full_upstream_repo": "suno/bark",
    "full_upstream_revision": MODEL_LICENSES[1]["revision"],
}
REQUIRED_IDENTITY_FIELDS = (
    "small_public_repo",
    "small_public_revision",
    "small_public_bytes",
    "small_public_sha256",
    "small_public_file",
    "small_upstream_repo",
    "small_upstream_revision",
    "small_checkpoint_bytes",
    "small_checkpoint_sha256",
    "small_config_bytes",
    "small_config_sha256",
    "full_public_repo",
    "full_public_revision",
    "full_public_bytes",
    "full_public_sha256",
    "full_public_file",
    "full_upstream_repo",
    "full_upstream_revision",
    "full_checkpoint_bytes",
    "full_checkpoint_sha256",
    "full_config_bytes",
    "full_config_sha256",
    "generation_config_bytes",
    "generation_config_sha256",
    "transformers_version",
    "transformers_source_revision",
    "transformers_sdist_sha256",
    "transformers_wheel_sha256",
)


class AuditError(ValueError):
    """A fail-closed factual audit input or environment error."""


class SdistAuditError(AuditError):
    """An sdist error carrying truthful acquisition/verification stage facts."""

    def __init__(
        self,
        message: str,
        *,
        acquired_archive_bytes: bool | None = None,
        verified_archive: bool = False,
        observed_size: int | None = None,
        observed_sha256: str | None = None,
    ) -> None:
        super().__init__(message)
        self.acquired_archive_bytes = acquired_archive_bytes
        self.verified_archive = verified_archive
        self.observed_size = observed_size
        self.observed_sha256 = observed_sha256


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


def normalized_name(name: str) -> str:
    """Return the PEP 503 normalized distribution name."""
    return re.sub(r"[-_.]+", "-", name.strip()).casefold()


def normalized_version(version: str) -> str:
    # The lock stores the exact resolved PEP 440 spelling.  Case-folding is
    # safe for the local-version labels and catches presentation-only drift.
    return re.sub(r"\s+", "", version.strip()).casefold()


def identity(name: str, version: str) -> str:
    return f"{normalized_name(name)}=={normalized_version(version)}"


def load_json(path: Path) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AuditError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError, AuditError) as exc:
        raise AuditError(f"cannot read JSON {path}: {exc}") from exc


def _regular(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _contract(project: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], bytes, bytes]:
    project_path = project / "pyproject.toml"
    lock_path = project / "uv.lock"
    manifest_path = project / "license_gate_manifest.json"
    if not all(_regular(path) for path in (project_path, lock_path, manifest_path)):
        raise AuditError("Bark pyproject.toml, uv.lock, or gate manifest is missing/symlinked")
    try:
        project_bytes = project_path.read_bytes()
        lock_bytes = lock_path.read_bytes()
        project_data = tomllib.loads(project_bytes.decode("utf-8"))
        lock_data = tomllib.loads(lock_bytes.decode("utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise AuditError(f"Bark closure bytes are unreadable: {exc}") from exc
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("gate_version") != 1:
        raise AuditError("Bark gate manifest version is unsupported")
    if sha256_bytes(project_bytes) != manifest.get("project_sha256"):
        raise AuditError("pyproject.toml bytes differ from the reviewed Bark contract")
    if sha256_bytes(lock_bytes) != manifest.get("lock_sha256"):
        raise AuditError("uv.lock bytes differ from the reviewed Bark contract")
    try:
        locked_rows = license_gate.rows(lock_data)
        license_gate.validate_project_schema(project_data)
    except SystemExit as exc:
        raise AuditError("Bark pyproject/uv.lock structural schema is invalid") from exc
    identities = manifest.get("identities")
    if not isinstance(identities, dict):
        raise AuditError("Bark fixed identity table is missing")
    for field in REQUIRED_IDENTITY_FIELDS:
        if field not in identities:
            raise AuditError(f"Bark fixed identity field is missing: {field}")
    for field, expected in MODEL_REPO_FIELDS.items():
        if identities.get(field) != expected:
            raise AuditError(f"Bark model identity drifted: {field}")
    return project_data, lock_data, manifest, locked_rows, project_bytes, lock_bytes


def _expected_packages(lock: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    locked_rows = sorted(
        lock.get("package", []),
        key=lambda row: (normalized_name(row["name"]), normalized_version(row["version"])),
    )
    virtual_rows = [row for row in locked_rows if row.get("source") == {"virtual": "."}]
    packages = [row for row in locked_rows if row.get("source") != {"virtual": "."}]
    if not packages:
        raise AuditError("Bark lock has no registry packages")
    if len(virtual_rows) != 1:
        raise AuditError("Bark lock must contain exactly one virtual project row")
    if any(row.get("source", {}).get("registry") not in {
        "https://pypi.org/simple",
        "https://download.pytorch.org/whl/cpu",
    } for row in packages):
        raise AuditError("Bark lock contains an unapproved package index")
    return locked_rows, packages, virtual_rows[0]


def _distribution_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        version = dist.version
        if not name or not version:
            continue
        location = Path(dist.locate_file(""))
        records.append(
            {
                "distribution": dist,
                "identity": identity(name, version),
                "name": name,
                "version": version,
                "location": str(location),
            }
        )
    return sorted(records, key=lambda item: (item["identity"], item["location"]))


def _git_identity(project: Path) -> dict[str, Any]:
    """Bind evidence to the checkout commit and this audit script's bytes."""
    root = project.resolve()
    for _ in range(3):
        root = root.parent
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise AuditError(f"cannot bind audit to a git commit: {exc}") from exc
    commit = completed.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise AuditError("git HEAD is not a full commit identity")
    script = Path(__file__).resolve()
    return {"commit": commit, "audit_script_sha256": sha256_file(script)}


def compare_multiset(expected: list[str], actual: list[str]) -> dict[str, Any]:
    expected_counts = Counter(expected)
    actual_counts = Counter(actual)
    missing = sorted((expected_counts - actual_counts).elements())
    unexpected = sorted((actual_counts - expected_counts).elements())
    duplicates = sorted(
        item for item, count in actual_counts.items() if count > 1
    )
    return {
        "expected": sorted(expected),
        "installed": sorted(actual),
        "missing": missing,
        "unexpected": unexpected,
        "duplicate_identities": duplicates,
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
    try:
        path.relative_to(Path(dist.locate_file("")))
    except ValueError:
        return None
    return path


def _is_license_path(relative: str) -> bool:
    basename = Path(relative).name.casefold()
    stem = Path(basename).stem
    return stem in LICENSE_FILE_NAMES or any(
        token in basename for token in ("license", "copying", "notice", "copyright")
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
        result.append({
            "path": relative,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return result, unsafe


def _first_four(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read(4)


def _elf_needed(path: Path, magic: bytes | None = None) -> dict[str, Any]:
    if (magic if magic is not None else _first_four(path)) != ELF_MAGIC:
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
        status = "ok"
    elif "no dynamic section" in output.casefold():
        status = "no-dynamic-section"
    else:
        status = "error"
    result: dict[str, Any] = {
        "format": "elf",
        "needed": needed,
        "inspection": status,
        "readelf_returncode": completed.returncode,
    }
    if status == "error":
        result["error"] = output.strip()[-2000:]
    return result


def _native_files(dist: metadata.Distribution) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for entry in sorted(dist.files or [], key=str):
        relative = str(entry)
        path = _entry_path(dist, entry)
        suffix_candidate = (
            Path(relative).suffix.casefold() in NATIVE_SUFFIXES
            or ".so." in Path(relative).name.casefold()
            or Path(relative).name.casefold().endswith(".dll")
        )
        if path is None:
            if suffix_candidate:
                errors.append({"path": relative, "stage": "locate", "error": "path escapes distribution root"})
            continue
        if path.is_symlink():
            if suffix_candidate:
                errors.append({"path": relative, "stage": "symlink", "error": "native candidate is symlinked"})
            continue
        if not path.is_file():
            if suffix_candidate:
                errors.append({"path": relative, "stage": "regular-file", "error": "native candidate is not a regular file"})
            continue
        try:
            magic = _first_four(path)
        except OSError as exc:
            errors.append({"path": relative, "stage": "magic", "error": str(exc)})
            continue
        if not suffix_candidate and magic != ELF_MAGIC:
            continue
        try:
            size = path.stat().st_size
            file_hash = sha256_file(path)
            inspection = _elf_needed(path, magic)
        except OSError as exc:
            errors.append({"path": relative, "stage": "hash-or-inspect", "error": str(exc)})
            continue
        result.append({
            "path": relative,
            "size": size,
            "sha256": file_hash,
            "bundled": True,
            "origin": "installed-distribution",
            "candidate": "elf-magic" if magic == ELF_MAGIC and not suffix_candidate else "native-suffix",
            "needed": inspection,
        })
    return result, errors


def _archive_member_path(name: str) -> str:
    if not isinstance(name, str) or not name or "\x00" in name:
        raise AuditError("sdist archive member has an invalid path")
    if "\\" in name or name.startswith("/"):
        raise AuditError(f"sdist archive member path is absolute or non-POSIX: {name!r}")
    candidate = name.rstrip("/")
    parts = candidate.split("/")
    if not candidate or any(part in {"", ".", ".."} for part in parts):
        raise AuditError(f"sdist archive member path contains traversal: {name!r}")
    return candidate


def _archive_format(archive_url: str) -> tuple[str, str]:
    path = urlsplit(archive_url).path.casefold()
    for suffixes, format_name, mode in (
        ((".tar.gz", ".tgz"), "tar.gz", "r:gz"),
        ((".tar.bz2", ".tbz2"), "tar.bz2", "r:bz2"),
        ((".tar.xz", ".txz"), "tar.xz", "r:xz"),
        ((".tar",), "tar", "r:"),
        ((".zip",), "zip", "zip"),
    ):
        if path.endswith(suffixes):
            return format_name, mode
    raise AuditError("locked sdist URL has an unsupported archive type")


def _archive_license_files(body: bytes, archive_url: str) -> tuple[str, list[dict[str, Any]]]:
    archive_format, archive_mode = _archive_format(archive_url)
    if len(body) > MAX_SDIST_BYTES:
        raise AuditError("locked sdist archive exceeds the bounded size")
    candidates: list[dict[str, Any]] = []
    names: set[str] = set()
    total_member_size = 0
    total_license_size = 0
    member_count = 0

    def inspect_member(name: str, size: int, is_regular: bool, read_member: Callable[[], bytes]) -> None:
        nonlocal total_member_size, total_license_size, member_count
        normalized = _archive_member_path(name)
        if normalized in names:
            raise AuditError(f"sdist archive contains duplicate member: {normalized}")
        names.add(normalized)
        member_count += 1
        if member_count > MAX_SDIST_MEMBERS:
            raise AuditError("sdist archive contains too many members")
        if size < 0 or size > MAX_SDIST_MEMBER_BYTES:
            raise AuditError(f"sdist archive member exceeds the bounded size: {normalized}")
        if not is_regular:
            return
        total_member_size += size
        if total_member_size > MAX_SDIST_TOTAL_MEMBER_BYTES:
            raise AuditError("sdist archive aggregate member size exceeds the bounded size")
        if not _is_license_path(normalized):
            return
        if size > MAX_LICENSE_BYTES:
            raise AuditError(f"sdist license candidate exceeds the bounded size: {normalized}")
        candidate_body = read_member()
        if len(candidate_body) != size:
            raise AuditError(f"sdist member size changed while reading: {normalized}")
        total_license_size += size
        if total_license_size > MAX_SDIST_LICENSE_TOTAL_BYTES:
            raise AuditError("sdist license candidates exceed the bounded total size")
        candidates.append({
            "path": normalized,
            "size": size,
            "sha256": sha256_bytes(candidate_body),
            "content_base64": base64.b64encode(candidate_body).decode("ascii"),
        })

    try:
        if archive_format == "zip":
            with zipfile.ZipFile(io.BytesIO(body), "r") as archive:
                for info in archive.infolist():
                    mode = (info.external_attr >> 16) & 0o170000
                    if mode and mode not in {stat.S_IFREG, stat.S_IFDIR}:
                        raise AuditError(f"sdist zip member is a link or special file: {info.filename!r}")
                    directory_marker = info.is_dir() or info.filename.endswith("/")
                    if (mode == stat.S_IFDIR and not directory_marker) or (mode == stat.S_IFREG and directory_marker):
                        raise AuditError(f"sdist zip member type disagrees with its directory name: {info.filename!r}")
                    is_directory = directory_marker or mode == stat.S_IFDIR
                    inspect_member(
                        info.filename,
                        0 if is_directory else info.file_size,
                        not is_directory,
                        lambda info=info: archive.read(info),
                    )
        else:
            with tarfile.open(fileobj=io.BytesIO(body), mode=archive_mode) as archive:
                for member in archive:
                    if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                        raise AuditError(f"sdist tar member is a link or special file: {member.name!r}")
                    if not member.isdir() and not member.isfile():
                        raise AuditError(f"sdist tar member has an unsupported type: {member.name!r}")
                    regular = member.isfile()
                    if regular:
                        def read_tar_member(member: tarfile.TarInfo = member) -> bytes:
                            handle = archive.extractfile(member)
                            if handle is None:
                                raise AuditError(f"sdist license member cannot be read: {member.name!r}")
                            with handle:
                                return handle.read(MAX_LICENSE_BYTES + 1)
                    else:
                        read_tar_member = lambda: b""
                    inspect_member(member.name, member.size if regular else 0, regular, read_tar_member)
    except (OSError, EOFError, RuntimeError, tarfile.TarError, zipfile.BadZipFile, NotImplementedError) as exc:
        raise AuditError(f"sdist archive inspection failed: {exc}") from exc
    if not candidates:
        raise AuditError("sdist archive contains no bounded LICENSE/COPYING/NOTICE/COPYRIGHT file")
    return archive_format, sorted(candidates, key=lambda item: item["path"])


def _validate_sdist_url(url: str, expected: str, *, initial: bool) -> None:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise AuditError(f"invalid locked sdist URL: {url}") from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname != PYPI_FILE_HOST
        or port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise AuditError(f"locked sdist URL is not an authenticated PyPI file URL: {url}")
    expected_path = urlsplit(expected).path
    if initial and url != expected:
        raise AuditError(f"initial sdist URL differs from the exact lock URL: {url}")
    if not initial and parsed.path != expected_path:
        raise AuditError(f"sdist redirect changed the exact locked path: {url}")


class _SdistRedirects(HTTPRedirectHandler):
    def __init__(self, trace: list[str], expected: str) -> None:
        super().__init__()
        self.trace = trace
        self.expected = expected

    def redirect_request(self, request: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Request | None:
        if len(self.trace) - 1 >= MAX_SDIST_REDIRECTS:
            raise AuditError("locked sdist redirect limit exceeded")
        _validate_sdist_url(newurl, self.expected, initial=False)
        self.trace.append(newurl)
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def _fetch_locked_sdist_license(
    row: dict[str, Any], fetcher: Callable[[str], tuple[str, bytes]] | None = None
) -> dict[str, Any]:
    if row.get("source") != {"registry": "https://pypi.org/simple"}:
        raise AuditError("locked sdist fallback is restricted to the PyPI registry")
    sdist = row.get("sdist")
    if not isinstance(sdist, dict) or set(sdist) != SDIST_ARTIFACT_KEYS:
        raise AuditError(f"locked sdist evidence is missing: {identity(row['name'], row['version'])}")
    requested_url = sdist.get("url")
    expected_hash = sdist.get("hash")
    expected_size = sdist.get("size")
    if (
        not isinstance(requested_url, str)
        or not isinstance(expected_hash, str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_hash)
        or isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size <= 0
        or expected_size > MAX_SDIST_BYTES
        or not isinstance(sdist["upload-time"], str)
        or not sdist["upload-time"].strip()
    ):
        raise AuditError("locked sdist identity is malformed or exceeds the bounded size")
    _validate_sdist_url(requested_url, requested_url, initial=True)
    trace = [requested_url]
    if fetcher is None:
        opener = build_opener(_SdistRedirects(trace, requested_url))
        request = Request(requested_url, headers={"Accept": "application/octet-stream", "User-Agent": "vokra-bark-audit/1"})
        with opener.open(request, timeout=30) as response:  # noqa: S310 - exact PyPI file URL is validated above
            final_url = response.geturl()
            _validate_sdist_url(final_url, requested_url, initial=False)
            if trace[-1] != final_url:
                trace.append(final_url)
            content_length = response.headers.get("Content-Length")
            if content_length and content_length.isdigit() and int(content_length) > MAX_SDIST_BYTES:
                raise AuditError("locked sdist Content-Length exceeds the bounded audit size")
            body = response.read(min(expected_size + 1, MAX_SDIST_BYTES + 1))
    else:
        final_url, body = fetcher(requested_url)
        _validate_sdist_url(final_url, requested_url, initial=False)
        if trace[-1] != final_url:
            trace.append(final_url)
    if not isinstance(body, bytes):
        raise AuditError("locked sdist fetch did not return bytes")
    if len(body) != expected_size:
        observed_hash = "sha256:" + sha256_bytes(body) if len(body) <= MAX_SDIST_BYTES else None
        raise SdistAuditError(
            f"locked sdist size mismatch: expected {expected_size}, got {len(body)}",
            acquired_archive_bytes=True,
            observed_size=len(body),
            observed_sha256=observed_hash,
        )
    observed_hash = "sha256:" + sha256_bytes(body)
    if observed_hash != expected_hash:
        raise SdistAuditError(
            "locked sdist SHA-256 mismatch",
            acquired_archive_bytes=True,
            observed_size=len(body),
            observed_sha256=observed_hash,
        )
    try:
        archive_format, license_files = _archive_license_files(body, final_url)
    except AuditError as exc:
        raise SdistAuditError(
            str(exc), acquired_archive_bytes=True, verified_archive=True,
            observed_size=len(body), observed_sha256=observed_hash,
        ) from exc
    return {
        "status": "ACQUIRED_LOCKED_SDIST_LICENSE_BYTES",
        "package": row["name"],
        "version": row["version"],
        "archive": {
            "url": requested_url,
            "final_url": final_url,
            "url_trace": trace,
            "size": expected_size,
            "hash": expected_hash,
            "format": archive_format,
        },
        "license_files": license_files,
    }


def _blocked_sdist_record(row: dict[str, Any], exc: Exception) -> dict[str, Any]:
    sdist = row.get("sdist") if isinstance(row.get("sdist"), dict) else {}
    requested_url = sdist.get("url")
    acquired = getattr(exc, "acquired_archive_bytes", None)
    return {
        "status": "BLOCKED_FACTUAL_SDIST_LICENSE_PATH",
        "package": row.get("name"),
        "version": row.get("version"),
        "requested_url": requested_url,
        "archive": {
            "url": requested_url,
            "size": sdist.get("size"),
            "hash": sdist.get("hash"),
            "observed_size": getattr(exc, "observed_size", None),
            "observed_sha256": getattr(exc, "observed_sha256", None),
        },
        "acquired_archive_bytes": acquired,
        "verified_archive": bool(getattr(exc, "verified_archive", False)),
        "license_files": [],
        "error": _controlled_license_error(exc),
    }


def _inspect_package(
    row: dict[str, Any], record: dict[str, Any] | None, review: dict[str, Any] | None,
    sdist_fetcher: Callable[[str], tuple[str, bytes]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    if record is None:
        return {
            "lock": {"name": row["name"], "version": row["version"], "source": row["source"]},
            "installed": None,
            "review": review,
        }, [f"installed closure missing: {identity(row['name'], row['version'])}"]
    dist = record["distribution"]
    publisher, unsafe_publisher = _publisher_files(dist)
    native, native_errors = _native_files(dist)
    metadata_fields = _metadata_fields(dist)
    sdist_license_evidence: dict[str, Any] | None = None
    if not publisher:
        try:
            sdist_license_evidence = _fetch_locked_sdist_license(row, sdist_fetcher)
        except (HTTPError, OSError, UnicodeError, ValueError) as exc:
            sdist_license_evidence = _blocked_sdist_record(row, exc)
    installed = {
        "name": dist.metadata.get("Name"),
        "version": dist.version,
        "normalized_identity": record["identity"],
        **metadata_fields,
        "publisher_files": publisher,
        "sdist_license_evidence": sdist_license_evidence,
        "native_files": native,
        "native_errors": native_errors,
        "bundled_libraries": [
            {
                "distribution": dist.metadata.get("Name"),
                "path": item["path"],
                "sha256": item["sha256"],
                "needed": item["needed"],
            }
            for item in native
        ],
    }
    failures: list[str] = []
    verified_sdist_license = (
        sdist_license_evidence is not None
        and sdist_license_evidence.get("status") == "ACQUIRED_LOCKED_SDIST_LICENSE_BYTES"
        and bool(sdist_license_evidence.get("license_files"))
    )
    installed_license_metadata = (
        bool(metadata_fields["license"])
        or bool(metadata_fields["license_expression"])
        or bool(metadata_fields["license_classifiers"])
    )
    if not installed_license_metadata and not verified_sdist_license:
        failures.append(f"missing package license metadata: {record['identity']}")
    if not publisher and sdist_license_evidence is None:
        failures.append(f"missing publisher LICENSE/NOTICE evidence: {record['identity']}")
    elif not publisher and (
        sdist_license_evidence.get("status") != "ACQUIRED_LOCKED_SDIST_LICENSE_BYTES"
        or not sdist_license_evidence.get("license_files")
    ):
        failures.append(f"missing publisher LICENSE/NOTICE evidence: {record['identity']}")
        failures.append(
            f"{sdist_license_evidence['status']}: {record['identity']}: "
            f"{canonical_json(sdist_license_evidence['error'])}"
        )
    failures.extend(f"unsafe publisher path: {record['identity']}:{path}" for path in unsafe_publisher)
    failures.extend(
        f"native candidate inspection failed: {record['identity']}:{item['path']}:{item['stage']}"
        for item in native_errors
    )
    if any(item["needed"]["inspection"] == "error" for item in native):
        failures.append(f"ELF NEEDED inspection failed: {record['identity']}")
    return {
        "lock": {
            "name": row["name"],
            "version": row["version"],
            "source": row["source"],
            "artifacts": {"sdist": row.get("sdist"), "wheels": row.get("wheels", [])},
        },
        "installed": installed,
        "review": review,
    }, failures


def _license_url(repo: str, revision: str) -> str:
    return f"https://huggingface.co/{repo}/raw/{revision}/LICENSE"


def _validate_license_url(url: str, item: dict[str, str]) -> None:
    if not any(item == allowed for allowed in MODEL_LICENSES):
        raise AuditError("LICENSE item is not one of the fixed Bark model identities")
    parsed = urlsplit(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise AuditError(f"invalid LICENSE URL port: {url}") from exc
    raw_url = _license_url(item["repo"], item["revision"])
    cdn_path = f"/{item['repo']}/resolve/{item['revision']}/LICENSE"
    raw_match = parsed.hostname == "huggingface.co" and url == raw_url
    cdn_match = parsed.hostname in LICENSE_CDN_HOSTS and parsed.path == cdn_path
    if (
        parsed.scheme != "https"
        or port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not (raw_match or cdn_match)
    ):
        raise AuditError(f"non-allow-listed LICENSE host or scheme: {url}")


class _LicenseRedirects(HTTPRedirectHandler):
    def __init__(self, trace: list[str], item: dict[str, str]) -> None:
        super().__init__()
        self.trace = trace
        self.item = item

    def redirect_request(self, request: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Request | None:
        _validate_license_url(newurl, self.item)
        self.trace.append(newurl)
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def _fetch_license(
    item: dict[str, str],
    fetcher: Callable[[str], tuple[str, bytes]] | None = None,
    claimed_license: str | None = None,
) -> dict[str, Any]:
    requested_url = _license_url(item["repo"], item["revision"])
    _validate_license_url(requested_url, item)
    trace = [requested_url]
    if fetcher is None:
        opener = build_opener(_LicenseRedirects(trace, item))
        request = Request(requested_url, headers={"Accept": "text/plain", "User-Agent": "vokra-bark-audit/1"})
        with opener.open(request, timeout=30) as response:  # noqa: S310 - host/path validated above
            final_url = response.geturl()
            _validate_license_url(final_url, item)
            if trace[-1] != final_url:
                trace.append(final_url)
            content_length = response.headers.get("Content-Length")
            if content_length and content_length.isdigit() and int(content_length) > MAX_LICENSE_BYTES:
                raise AuditError("upstream LICENSE Content-Length exceeds the bounded audit size")
            body = response.read(MAX_LICENSE_BYTES + 1)
    else:
        final_url, body = fetcher(requested_url)
        _validate_license_url(final_url, item)
        if trace[-1] != final_url:
            trace.append(final_url)
    if len(body) > MAX_LICENSE_BYTES:
        raise AuditError("upstream LICENSE response exceeds the bounded audit size")
    body_hash = sha256_bytes(body)
    return {
        "id": item["id"],
        "repo": item["repo"],
        "revision": item["revision"],
        "requested_url": requested_url,
        "final_url": final_url,
        "url_trace": trace,
        "size": len(body),
        "sha256": body_hash,
        "primary_source_bytes": len(body),
        "primary_source_sha256": body_hash,
        "claimed_license": claimed_license,
        "claimed_license_source": "existing manifest only; unresolved claims are recorded as null",
        "license_classification": "UNCLASSIFIED_PRIMARY_SOURCE_BYTES_ONLY",
        "acquired_file": "LICENSE",
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


def _blocked_license_record(
    item: dict[str, str], claimed_license: str | None, exc: Exception
) -> dict[str, Any]:
    requested_url = _license_url(item["repo"], item["revision"])
    return {
        "status": "BLOCKED_FACTUAL_LICENSE_PATH",
        "id": item["id"],
        "repo": item["repo"],
        "revision": item["revision"],
        "requested_url": requested_url,
        "final_url": None,
        "url_trace": [requested_url],
        "acquired_bytes": False,
        "size": None,
        "bytes": None,
        "sha256": None,
        "primary_source_bytes": None,
        "primary_source_sha256": None,
        "content_base64": None,
        "claimed_license": claimed_license,
        "claimed_license_source": "existing manifest only; unresolved claims are recorded as null",
        "license_classification": "UNAVAILABLE_FACTUAL_LICENSE_PATH",
        "acquired_file": None,
        "error": _controlled_license_error(exc),
    }


def audit_model_licenses(
    manifest: dict[str, Any], fetcher: Callable[[str], tuple[str, bytes]] | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
    claims = {
        row.get("id"): row.get("license")
        for row in manifest.get("license_rows", [])
        if isinstance(row, dict) and row.get("id") in {item["id"] for item in MODEL_LICENSES}
    }
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for item in MODEL_LICENSES:
        claimed = claims.get(item["id"])
        try:
            records.append(
                _fetch_license(
                    item,
                    fetcher,
                    claimed if claimed not in {None, "UNRESOLVED"} else None,
                )
            )
        except (HTTPError, OSError, UnicodeError, ValueError) as exc:
            controlled_error = _controlled_license_error(exc)
            records.append(_blocked_license_record(
                item, claimed if claimed not in {None, "UNRESOLVED"} else None, exc
            ))
            failures.append(
                f"BLOCKED_FACTUAL_LICENSE_PATH: {item['repo']}@{item['revision']}: "
                f"{canonical_json(controlled_error)}"
            )
    return records, failures


def audit_environment(
    project: Path,
    fetch_model_licenses: bool = True,
    sdist_fetcher: Callable[[str], tuple[str, bytes]] | None = None,
) -> dict[str, Any]:
    project_data, lock, manifest, locked_rows_from_contract, project_bytes, lock_bytes = _contract(project)
    locked_rows, expected_rows, virtual_row = _expected_packages(lock)
    if locked_rows != locked_rows_from_contract:
        raise AuditError("Bark lock row ordering or content is not canonical")
    records = _distribution_records()
    expected_ids = [identity(row["name"], row["version"]) for row in expected_rows]
    actual_ids = [record["identity"] for record in records]
    closure = compare_multiset(expected_ids, actual_ids)
    by_identity: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_identity.setdefault(record["identity"], []).append(record)
    packages: list[dict[str, Any]] = []
    failures: list[str] = []
    review_map = {
        (row.get("name"), row.get("version")): row
        for row in manifest.get("package_review_rows", [])
        if isinstance(row, dict)
    }
    for row in expected_rows:
        key = identity(row["name"], row["version"])
        candidates = by_identity.get(key, [])
        review = review_map.get((row["name"], row["version"]))
        package, package_failures = _inspect_package(
            row, candidates[0] if len(candidates) == 1 else None, review, sdist_fetcher
        )
        if review is None:
            package_failures.append(f"package review evidence missing: {key}")
        if len(candidates) > 1:
            package_failures.append(f"duplicate installed distribution: {key}")
        packages.append(package)
        failures.extend(package_failures)
    virtual_review = next(
        (
            row for row in manifest.get("package_review_rows", [])
            if isinstance(row, dict)
            and (row.get("name"), row.get("version")) == (virtual_row["name"], virtual_row["version"])
        ),
        None,
    )
    packages.append({
        "lock": virtual_row,
        "installed": None,
        "review": virtual_review,
        "activity": {
            "status": "INACTIVE_VIRTUAL_PROJECT",
            "reason": "uv.lock virtual project row (package=false) is not an installed distribution",
            "evidence": "source={virtual='.'}; installed distribution multiset intentionally excludes this row",
        },
    })
    if virtual_review is None:
        failures.append("virtual project row has no matching package review evidence")
    if not closure["exact"]:
        failures.append("installed distribution multiset does not exactly match uv.lock")
    if sys.version_info[:2] != (3, 12):
        failures.append(f"Python runtime is not 3.12: {platform.python_version()}")
    if sys.platform != "linux" or platform.machine().casefold() not in {"x86_64", "amd64"}:
        failures.append(f"audit host is not Linux x86_64: {sys.platform}/{platform.machine()}")

    identities = manifest["identities"]
    source_fields = sorted(key for key in identities if "bark_source" in key or key in {"source_repo", "source_revision"})
    source_contract = {
        "status": "BLOCKED_MISSING_PINNED_SOURCE_REVISION",
        "required_fields": ["bark_source_repo", "bark_source_revision"],
        "present_related_fields": source_fields,
        "license_file": "LICENSE",
    }
    failures.append("Bark source-code LICENSE revision is absent from the existing identity contract")
    if fetch_model_licenses:
        model_license_files, model_license_failures = audit_model_licenses(manifest)
        failures.extend(model_license_failures)
    else:
        model_license_files = []
    git = _git_identity(project)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "BLOCKED" if failures else "PASS",
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
        "git": git,
        "fixed_bark_identities": {key: identities[key] for key in REQUIRED_IDENTITY_FIELDS},
        "locked_rows": locked_rows,
        "virtual_project_row": {
            "row": virtual_row,
            "status": "INACTIVE_VIRTUAL_PROJECT",
            "reason": "uv.lock virtual project row (package=false) is not an installed distribution",
        },
        "active_installed_rows": [
            {
                "name": record["name"],
                "version": record["version"],
                "normalized_identity": record["identity"],
                "location": record["location"],
            }
            for record in records
        ],
        "closure": closure,
        "packages": packages,
        "source_license_contract": source_contract,
        "model_license_files": model_license_files,
        "model_acquisition": {
            "policy": "allow-listed model LICENSE-only in-memory fetch",
            "requested_files": [item["requested_url"] for item in model_license_files],
            "non_license_files": [],
            "non_license_requests": [],
            "proof": "no model-weight request path exists in this audit; this section fetches only exact HF LICENSE URLs",
        },
        "dependency_acquisition": {
            "policy": "exact locked PyPI sdist only when installed publisher LICENSE/NOTICE evidence is absent",
            "requests": [
                {
                    "package": item["lock"]["name"],
                    "version": item["lock"]["version"],
                    "url": evidence.get("requested_url", evidence.get("archive", {}).get("url")),
                    "status": evidence.get("status"),
                }
                for item in packages
                if isinstance(item.get("installed"), dict)
                and (evidence := item["installed"].get("sdist_license_evidence")) is not None
            ],
            "out_of_scope_requests": [],
            "model_files": [],
            "non_license_files": [],
        },
        "failures": sorted(set(failures)),
    }
    return report


def run(project: Path, output: Path, fetch_model_licenses: bool) -> int:
    try:
        report = audit_environment(project, fetch_model_licenses)
    except (AuditError, OSError, UnicodeError, ValueError) as exc:
        report = {
            "schema": SCHEMA,
            "status": "BLOCKED",
            "environment": {"model_code_imported": False, "cargo_invoked": False},
            "model_acquisition": {"requested_files": [], "non_license_files": [], "non_license_requests": []},
            "failures": [str(exc)],
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    if report["failures"]:
        print("bark dependency audit: BLOCKED: " + "; ".join(report["failures"]), file=sys.stderr)
        return 2
    print(f"bark dependency audit: PASS ({output})")
    return 0


def self_test() -> int:
    expected = [identity("foo-bar", "1.0"), identity("torch", "2.13.0+cpu")]
    exact = compare_multiset(expected, [identity("torch", "2.13.0+cpu"), identity("foo_bar", "1.0")])
    assert exact["exact"] and not exact["missing"] and not exact["unexpected"]
    mismatch = compare_multiset(expected, [identity("foo-bar", "1.0"), identity("torch", "2.12.0+cpu")])
    assert mismatch["missing"] == [identity("torch", "2.13.0+cpu")]
    assert mismatch["unexpected"] == [identity("torch", "2.12.0+cpu")]
    duplicate = compare_multiset([identity("foo-bar", "1.0")], [identity("foo_bar", "1.0"), identity("foo-bar", "1.0")])
    assert duplicate["duplicate_identities"] == [identity("foo-bar", "1.0")]
    assert duplicate["unexpected"] == [identity("foo-bar", "1.0")]

    good = _license_url(MODEL_LICENSES[0]["repo"], MODEL_LICENSES[0]["revision"])
    cdn_good = (
        "https://cdn-lfs.huggingface.co/"
        f"{MODEL_LICENSES[0]['repo']}/resolve/{MODEL_LICENSES[0]['revision']}/LICENSE"
    )
    body = b"primary source bytes\n"
    result = _fetch_license(MODEL_LICENSES[0], lambda url: (url, body))
    assert result["requested_url"] == good
    assert result["final_url"] == good
    assert result["size"] == len(body) and result["sha256"] == sha256_bytes(body)
    assert result["license_classification"] == "UNCLASSIFIED_PRIMARY_SOURCE_BYTES_ONLY"
    cdn_result = _fetch_license(MODEL_LICENSES[0], lambda _url: (cdn_good, body))
    assert cdn_result["final_url"] == cdn_good and cdn_result["url_trace"] == [good, cdn_good]
    for bad in (
        good.replace("/LICENSE", "/model.safetensors"),
        "https://example.invalid/LICENSE",
        "http://huggingface.co/suno/bark/LICENSE",
        "https://huggingface.co:444/suno/bark/LICENSE",
        "https://user:hunter2@huggingface.co/suno/bark/LICENSE",
        f"{good}?download=true",
        f"{good}#fragment",
        f"https://huggingface.co/suno/bark/resolve/{MODEL_LICENSES[0]['revision']}/LICENSE",
        f"https://cdn-lfs.huggingface.co/suno/bark-small/resolve/{'0' * 40}/LICENSE",
        f"https://cdn-lfs.huggingface.co/suno/bark-small/resolve/{MODEL_LICENSES[0]['revision']}/LICENSE.txt",
        f"https://cdn-lfs.huggingface.co/suno/bark-small/resolve/{MODEL_LICENSES[0]['revision']}/model.safetensors",
        f"https://cdn-lfs.huggingface.co/suno/bark-small/resolve/{MODEL_LICENSES[0]['revision']}/LICENSE?download=true",
        f"https://cdn-lfs.huggingface.co/suno/bark-small/resolve/{MODEL_LICENSES[0]['revision']}/LICENSE#fragment",
        f"https://cdn-lfs.huggingface.co:444/suno/bark-small/resolve/{MODEL_LICENSES[0]['revision']}/LICENSE",
        f"https://user:hunter2@cdn-lfs.huggingface.co/suno/bark-small/resolve/{MODEL_LICENSES[0]['revision']}/LICENSE",
    ):
        try:
            _fetch_license(MODEL_LICENSES[0], lambda _url, bad=bad: (bad, body))
        except AuditError:
            pass
        else:
            raise AssertionError(f"accepted unsafe LICENSE redirect: {bad}")
    try:
        _fetch_license(MODEL_LICENSES[0], lambda url: (url, b"x" * (MAX_LICENSE_BYTES + 1)))
    except AuditError:
        pass
    else:
        raise AssertionError("accepted an over-sized LICENSE response")

    def synthetic_sdist_row(archive_body: bytes, suffix: str = ".tar.gz") -> dict[str, Any]:
        url = f"https://{PYPI_FILE_HOST}/packages/demo/demo-1.0{suffix}"
        return {
            "name": "demo",
            "version": "1.0",
            "source": {"registry": "https://pypi.org/simple"},
            "sdist": {
                "url": url,
                "hash": "sha256:" + sha256_bytes(archive_body),
                "size": len(archive_body),
                "upload-time": "2026-01-01T00:00:00.000Z",
            },
        }

    def make_tar(members: list[tuple[str, bytes]]) -> bytes:
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w:gz") as archive:
            for name, member_body in members:
                info = tarfile.TarInfo(name)
                info.size = len(member_body)
                archive.addfile(info, io.BytesIO(member_body))
        return output.getvalue()

    tar_body = make_tar([
        ("demo-1.0/LICENSE", b"MIT\n"),
        ("demo-1.0/README.md", b"not a license\n"),
    ])
    tar_row = synthetic_sdist_row(tar_body)
    tar_result = _fetch_locked_sdist_license(tar_row, lambda url: (url, tar_body))
    assert tar_result["status"] == "ACQUIRED_LOCKED_SDIST_LICENSE_BYTES"
    assert tar_result["archive"]["hash"] == tar_row["sdist"]["hash"]
    assert tar_result["archive"]["format"] == "tar.gz"
    assert tar_result["license_files"] == [{
        "path": "demo-1.0/LICENSE",
        "size": 4,
        "sha256": sha256_bytes(b"MIT\n"),
        "content_base64": base64.b64encode(b"MIT\n").decode("ascii"),
    }]

    zip_output = io.BytesIO()
    with zipfile.ZipFile(zip_output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("demo-1.0/COPYING", b"COPYING\n")
    zip_body = zip_output.getvalue()
    zip_result = _fetch_locked_sdist_license(
        synthetic_sdist_row(zip_body, ".zip"), lambda url: (url, zip_body)
    )
    assert zip_result["archive"]["format"] == "zip"
    assert zip_result["license_files"][0]["path"] == "demo-1.0/COPYING"

    def expect_sdist_blocked(
        row: dict[str, Any], archive_body: bytes, final_url: str | None = None,
        acquired_archive_bytes: bool | None = None, verified_archive: bool | None = None,
    ) -> None:
        try:
            _fetch_locked_sdist_license(
                row,
                lambda url: (final_url or url, archive_body),
            )
        except AuditError as exc:
            blocked_record = _blocked_sdist_record(row, exc)
            if acquired_archive_bytes is not None:
                assert getattr(exc, "acquired_archive_bytes", None) == acquired_archive_bytes
                assert blocked_record["acquired_archive_bytes"] == acquired_archive_bytes
            if verified_archive is not None:
                assert getattr(exc, "verified_archive", False) == verified_archive
                assert blocked_record["verified_archive"] == verified_archive
            return
        raise AssertionError("accepted unsafe or unverifiable locked sdist")

    tampered_hash = dict(tar_row)
    tampered_hash["sdist"] = dict(tar_row["sdist"], hash="sha256:" + "0" * 64)
    expect_sdist_blocked(tampered_hash, tar_body, acquired_archive_bytes=True, verified_archive=False)
    tampered_size = dict(tar_row)
    tampered_size["sdist"] = dict(tar_row["sdist"], size=len(tar_body) + 1)
    expect_sdist_blocked(tampered_size, tar_body, acquired_archive_bytes=True, verified_archive=False)
    missing_upload_time = dict(tar_row)
    missing_upload_time["sdist"] = dict(tar_row["sdist"])
    missing_upload_time["sdist"].pop("upload-time")
    expect_sdist_blocked(missing_upload_time, tar_body, acquired_archive_bytes=None, verified_archive=False)
    extra_sdist_key = dict(tar_row)
    extra_sdist_key["sdist"] = dict(tar_row["sdist"], unexpected="x")
    expect_sdist_blocked(extra_sdist_key, tar_body, acquired_archive_bytes=None, verified_archive=False)
    bad_host = dict(tar_row)
    bad_host["sdist"] = dict(tar_row["sdist"], url=tar_row["sdist"]["url"].replace(PYPI_FILE_HOST, "example.invalid"))
    expect_sdist_blocked(bad_host, tar_body)
    expect_sdist_blocked(tar_row, tar_body, "https://example.invalid/packages/demo/demo-1.0.tar.gz", acquired_archive_bytes=None)
    expect_sdist_blocked(tar_row, tar_body, tar_row["sdist"]["url"] + "?download=1", acquired_archive_bytes=None)

    traversal_body = make_tar([("../LICENSE", b"bad\n")])
    expect_sdist_blocked(synthetic_sdist_row(traversal_body), traversal_body, acquired_archive_bytes=True, verified_archive=True)
    symlink_output = io.BytesIO()
    with tarfile.open(fileobj=symlink_output, mode="w:gz") as archive:
        link = tarfile.TarInfo("demo-1.0/LICENSE")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        archive.addfile(link)
    symlink_body = symlink_output.getvalue()
    expect_sdist_blocked(synthetic_sdist_row(symlink_body), symlink_body, acquired_archive_bytes=True, verified_archive=True)
    zip_link_output = io.BytesIO()
    with zipfile.ZipFile(zip_link_output, "w") as archive:
        link = zipfile.ZipInfo("demo-1.0/LICENSE")
        link.external_attr = stat.S_IFLNK << 16
        archive.writestr(link, b"/etc/passwd")
    zip_link_body = zip_link_output.getvalue()
    expect_sdist_blocked(synthetic_sdist_row(zip_link_body, ".zip"), zip_link_body, acquired_archive_bytes=True, verified_archive=True)
    zip_regular_directory_output = io.BytesIO()
    with zipfile.ZipFile(zip_regular_directory_output, "w") as archive:
        regular_directory = zipfile.ZipInfo("demo-1.0/LICENSE/")
        regular_directory.external_attr = stat.S_IFREG << 16
        archive.writestr(regular_directory, b"not a directory")
    zip_regular_directory_body = zip_regular_directory_output.getvalue()
    expect_sdist_blocked(
        synthetic_sdist_row(zip_regular_directory_body, ".zip"),
        zip_regular_directory_body,
        acquired_archive_bytes=True,
        verified_archive=True,
    )
    zip_directory_file_output = io.BytesIO()
    with zipfile.ZipFile(zip_directory_file_output, "w") as archive:
        directory_file = zipfile.ZipInfo("demo-1.0/LICENSE")
        directory_file.external_attr = stat.S_IFDIR << 16
        archive.writestr(directory_file, b"")
    zip_directory_file_body = zip_directory_file_output.getvalue()
    expect_sdist_blocked(
        synthetic_sdist_row(zip_directory_file_body, ".zip"),
        zip_directory_file_body,
        acquired_archive_bytes=True,
        verified_archive=True,
    )
    zip_special_directory_output = io.BytesIO()
    with zipfile.ZipFile(zip_special_directory_output, "w") as archive:
        special_directory = zipfile.ZipInfo("demo-1.0/LICENSE/")
        special_directory.external_attr = stat.S_IFLNK << 16
        archive.writestr(special_directory, b"target")
    zip_special_directory_body = zip_special_directory_output.getvalue()
    expect_sdist_blocked(
        synthetic_sdist_row(zip_special_directory_body, ".zip"),
        zip_special_directory_body,
        acquired_archive_bytes=True,
        verified_archive=True,
    )
    duplicate_body = make_tar([
        ("demo-1.0/LICENSE", b"one\n"),
        ("demo-1.0/LICENSE", b"two\n"),
    ])
    expect_sdist_blocked(synthetic_sdist_row(duplicate_body), duplicate_body, acquired_archive_bytes=True, verified_archive=True)
    no_license_body = make_tar([("demo-1.0/README.md", b"readme\n")])
    expect_sdist_blocked(synthetic_sdist_row(no_license_body), no_license_body, acquired_archive_bytes=True, verified_archive=True)
    bomb_body = make_tar([("demo-1.0/LICENSE", b"x" * (MAX_SDIST_MEMBER_BYTES + 1))])
    expect_sdist_blocked(synthetic_sdist_row(bomb_body), bomb_body, acquired_archive_bytes=True, verified_archive=True)
    unsupported = synthetic_sdist_row(tar_body, ".whl")
    expect_sdist_blocked(unsupported, tar_body, acquired_archive_bytes=True, verified_archive=True)

    def failing_fetcher(
        failures_by_index: dict[int, Callable[[str], BaseException]]
    ) -> tuple[Callable[[str], tuple[str, bytes]], list[str]]:
        calls: list[str] = []

        def fetch(url: str) -> tuple[str, bytes]:
            index = len(calls)
            calls.append(url)
            if index in failures_by_index:
                raise failures_by_index[index](url)
            return url, body

        return fetch, calls

    failure_cases = {
        "first": {0: lambda _url: ValueError("first LICENSE path is unavailable")},
        "second": {1: lambda _url: HTTPError(good, 404, "second LICENSE path is unavailable", {}, None)},
        "both": {
            0: lambda _url: OSError("first LICENSE transport failed"),
            1: lambda _url: ValueError("second LICENSE path is unavailable"),
        },
        "partial": {0: lambda _url: HTTPError(good, 503, "partial LICENSE outage", {}, None)},
        "none": {},
    }
    for case, failure_factories in failure_cases.items():
        fetcher, calls = failing_fetcher(failure_factories)
        license_records, license_failures = audit_model_licenses(
            {"license_rows": []}, fetcher
        )
        assert len(calls) == len(MODEL_LICENSES) == len(license_records)
        assert len(license_failures) == len(failure_factories)
        assert [record["repo"] for record in license_records] == [
            item["repo"] for item in MODEL_LICENSES
        ]
        for index, record in enumerate(license_records):
            if index in failure_factories:
                assert record["status"] == "BLOCKED_FACTUAL_LICENSE_PATH"
                assert record["requested_url"] == calls[index]
                assert record["acquired_bytes"] is False
                assert record["bytes"] is None and record["sha256"] is None
                assert record["primary_source_bytes"] is None
                assert record["primary_source_sha256"] is None
                assert record["error"]["kind"] in {"HTTP_ERROR", "URL_ERROR", "OS_ERROR", "VALIDATION_ERROR"}
            else:
                assert "status" not in record
        if case == "none":
            assert all(record.get("status") is None for record in license_records)

    # Exercise the actual report writer with both fixed paths failing.  Patch
    # only this self-test's call sites so it cannot contact the network or scan
    # the host's installed distributions, while the real contract parser still
    # contributes the complete closure facts to the report.
    original_fetch_license = globals()["_fetch_license"]
    original_fetch_sdist_license = globals()["_fetch_locked_sdist_license"]
    original_distribution_records = globals()["_distribution_records"]

    class EmptyMetadata(dict[str, str]):
        def get_all(self, _key: str) -> list[str]:
            return []

    class EmptyDistribution:
        metadata = EmptyMetadata(Name="safetensors")
        version = "0.4.5"
        files: list[str] = []

        def locate_file(self, _entry: Any) -> Path:
            return Path(__file__).resolve().parent

    fallback_row = dict(tar_row, name="safetensors", version="0.4.5")
    fallback_package, fallback_failures = _inspect_package(
        fallback_row,
        {
            "distribution": EmptyDistribution(),
            "identity": identity("safetensors", "0.4.5"),
            "name": "safetensors",
            "version": "0.4.5",
            "location": str(Path(__file__).resolve().parent),
        },
        None,
        lambda url: (url, tar_body),
    )
    assert fallback_package["installed"]["sdist_license_evidence"]["status"] == "ACQUIRED_LOCKED_SDIST_LICENSE_BYTES"
    assert fallback_package["installed"]["sdist_license_evidence"]["license_files"][0]["path"].endswith("/LICENSE")
    assert fallback_package["installed"]["license"] is None
    assert fallback_package["installed"]["license_expression"] is None
    assert fallback_package["installed"]["license_classifiers"] == []
    assert not any("missing package license metadata" in item for item in fallback_failures)
    assert not any("missing publisher LICENSE/NOTICE evidence" in item for item in fallback_failures)

    def blocked_fetch(
        item: dict[str, str],
        _fetcher: Callable[[str], tuple[str, bytes]] | None = None,
        claimed_license: str | None = None,
    ) -> dict[str, Any]:
        del claimed_license
        raise ValueError(f"self-test unavailable LICENSE path: {item['repo']}")

    def blocked_sdist_fetch(
        _row: dict[str, Any],
        _fetcher: Callable[[str], tuple[str, bytes]] | None = None,
    ) -> dict[str, Any]:
        raise HTTPError("https://files.pythonhosted.org/packages/demo/demo-1.0.tar.gz", 404, "self-test sdist unavailable", {}, None)

    globals()["_fetch_license"] = blocked_fetch
    globals()["_fetch_locked_sdist_license"] = blocked_sdist_fetch
    globals()["_distribution_records"] = lambda: [{
        "distribution": EmptyDistribution(),
        "identity": identity("safetensors", "0.4.5"),
        "name": "safetensors",
        "version": "0.4.5",
        "location": str(Path(__file__).resolve().parent),
    }]
    try:
        with tempfile.TemporaryDirectory(prefix="bark-audit-report-self-test-") as directory:
            output = Path(directory) / "blocked-report.json"
            result_code = run(Path(__file__).resolve().parent, output, True)
            blocked_report = json.loads(output.read_text(encoding="utf-8"))
            assert result_code == 2
            assert blocked_report["status"] == "BLOCKED"
            assert blocked_report["locked_rows"]
            assert len(blocked_report["packages"]) == len(blocked_report["locked_rows"])
            assert blocked_report["source_license_contract"]["status"] == "BLOCKED_MISSING_PINNED_SOURCE_REVISION"
            safetensors_report = next(
                item for item in blocked_report["packages"]
                if item["lock"].get("name") == "safetensors"
            )
            blocked_sdist = safetensors_report["installed"]["sdist_license_evidence"]
            assert blocked_sdist["status"] == "BLOCKED_FACTUAL_SDIST_LICENSE_PATH"
            assert blocked_sdist["acquired_archive_bytes"] is None
            assert blocked_sdist["verified_archive"] is False
            assert blocked_sdist["license_files"] == []
            assert blocked_sdist["error"]["kind"] == "HTTP_ERROR"
            dependency_requests = blocked_report["dependency_acquisition"]["requests"]
            safetensors_request = next(
                item for item in dependency_requests if item["package"] == "safetensors"
            )
            assert safetensors_request["url"] == blocked_sdist["requested_url"]
            assert safetensors_request["status"] == "BLOCKED_FACTUAL_SDIST_LICENSE_PATH"
            assert len(blocked_report["model_license_files"]) == len(MODEL_LICENSES)
            assert all(
                item.startswith("https://huggingface.co/")
                for item in blocked_report["model_acquisition"]["requested_files"]
            )
            assert blocked_report["model_acquisition"]["non_license_requests"] == []
            assert all(
                record["status"] == "BLOCKED_FACTUAL_LICENSE_PATH"
                and record["acquired_bytes"] is False
                and record["size"] is None
                and record["bytes"] is None
                and record["sha256"] is None
                and record["content_base64"] is None
                for record in blocked_report["model_license_files"]
            )
            assert len(blocked_report["failures"]) >= len(MODEL_LICENSES)
    finally:
        globals()["_fetch_license"] = original_fetch_license
        globals()["_fetch_locked_sdist_license"] = original_fetch_sdist_license
        globals()["_distribution_records"] = original_distribution_records

    with tempfile.TemporaryDirectory(prefix="bark-audit-self-test-") as directory:
        path = Path(directory) / "not-native"
        path.write_bytes(b"not an ELF file")
        assert _elf_needed(path)["format"] == "non-elf"
        first = canonical_json({"b": 2, "a": 1})
        second = canonical_json(json.loads(first))
        assert first == second == '{"a":1,"b":2}'
        output = Path(directory) / "report.json"
        output.write_text(first + "\n", encoding="utf-8")
        assert output.read_text(encoding="utf-8") == first + "\n"
    print("bark dependency audit: self-test PASS")
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
