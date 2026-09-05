#!/usr/bin/env python3
"""Model-free factual audit for the pinned SpeechT5 Python environment.

The wrapper performs the authorized VAST synchronization first.  This module
then inspects only the frozen lock, ``importlib.metadata`` records, publisher
license files, and installed native payloads.  It never imports Torch,
Transformers, a model implementation, or Vokra code.  When an installed wheel
has no publisher file, it may request only that package's exact locked PyPI
sdist, verify its URL/redirect/size/hash, and retain bounded archive license
bytes/hashes; it has no model/source acquisition path.  The preceding
dependency-only ``uv sync`` may use the locked package indexes, while the
``auditor_network_requests`` evidence field counts attempted license fallback
requests, including failed fetches.
License metadata is recorded as observed; this tool does not classify
licenses or create owner sign-off.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import importlib.metadata as metadata
import importlib.util
import io
import json
import platform
import re
import stat
import subprocess
import sys
import sysconfig
import tarfile
import tomllib
import zipfile
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse, unquote
from urllib.request import HTTPRedirectHandler, Request, build_opener


SCHEMA = "vokra-speecht5-dependency-audit-v1"
COMPACT_SCHEMA = "vokra-speecht5-dependency-audit-compact-v1"
ELF_MAGIC = b"\x7fELF"
NATIVE_SUFFIXES = {".so", ".dylib", ".dll", ".pyd", ".a"}
LICENSE_TOKENS = ("license", "licence", "copying", "notice", "copyright")
REGISTRIES = {"https://pypi.org/simple", "https://download.pytorch.org/whl/cpu"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
PYPI_HOST = "files.pythonhosted.org"
MAX_SDIST_BYTES = 64 * 1024 * 1024
MAX_MEMBER_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 4096
MAX_SDIST_REDIRECTS = 3


class AuditError(RuntimeError):
    """A factual or contract failure that must remain fail-closed."""


class LockedSdistFetchError(AuditError):
    """A locked-sdist request was attempted but did not yield valid evidence."""

    def __init__(self, message: str, network_requests: int = 1) -> None:
        super().__init__(message)
        self.network_requests = network_requests


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AuditError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError, AuditError) as error:
        raise AuditError(f"cannot read strict JSON {path}: {error}") from error


def regular(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise AuditError(f"{label} is missing, non-regular, or symlinked: {path}")


def safe_absolute_path(path: Path, label: str) -> None:
    if not path.is_absolute() or any(component in {".", ".."} for component in path.parts[1:]):
        raise AuditError(f"{label} must be absolute without . or .. components: {path}")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise AuditError(f"{label} has a symlinked ancestor: {current}")


def paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def normalized_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip()).casefold()


def normalized_version(version: str) -> str:
    return re.sub(r"\s+", "", version.strip()).casefold()


def identity(name: str, version: str) -> str:
    return f"{normalized_name(name)}=={normalized_version(version)}"


def load_gate(project: Path) -> Any:
    gate_path = project / "preflight_gate.py"
    regular(gate_path, "preflight gate")
    spec = importlib.util.spec_from_file_location("vokra_speecht5_audit_preflight_gate", gate_path)
    if spec is None or spec.loader is None:
        raise AuditError("preflight gate cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise AuditError(f"preflight gate constants cannot be loaded: {type(error).__name__}") from error
    return module


def lock_rows(lock: dict[str, Any]) -> list[dict[str, Any]]:
    rows = lock.get("package")
    if not isinstance(rows, list) or not rows:
        raise AuditError("uv.lock package rows are missing")
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not isinstance(row.get("version"), str):
            raise AuditError("uv.lock package row identity is malformed")
        source = row.get("source")
        if not isinstance(source, dict):
            raise AuditError(f"uv.lock source is malformed: {row['name']}")
        source_key = json.dumps(source, sort_keys=True)
        key = (row["name"], row["version"], source_key)
        if key in seen:
            raise AuditError(f"uv.lock duplicate package identity: {row['name']}")
        seen.add(key)
        if source != {"virtual": "."} and source.get("registry") not in REGISTRIES:
            raise AuditError(f"uv.lock package index is not reviewed: {row['name']}")
        result.append({
            "name": row["name"],
            "version": row["version"],
            "source": source,
            "resolution-markers": row.get("resolution-markers", []),
            "dependencies": row.get("dependencies", []),
        })
    return sorted(result, key=lambda row: (normalized_name(row["name"]), normalized_version(row["version"]), json.dumps(row["source"], sort_keys=True)))


def contract(project: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    safe_absolute_path(project, "--project")
    if project.is_symlink() or not project.is_dir():
        raise AuditError(f"--project is not a real directory: {project}")
    project_path = project / "pyproject.toml"
    lock_path = project / "uv.lock"
    manifest_path = project / "license_gate_manifest.json"
    for path, label in ((project_path, "pyproject.toml"), (lock_path, "uv.lock"), (manifest_path, "license gate manifest")):
        regular(path, label)
    try:
        project_bytes = project_path.read_bytes()
        lock_bytes = lock_path.read_bytes()
        project_data = tomllib.loads(project_bytes.decode("utf-8"))
        lock_data = tomllib.loads(lock_bytes.decode("utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise AuditError(f"SpeechT5 closure bytes are unreadable: {error}") from error
    manifest = strict_json(manifest_path)
    if not isinstance(project_data, dict) or not isinstance(lock_data, dict) or not isinstance(manifest, dict):
        raise AuditError("SpeechT5 contract roots are not objects")
    gate = load_gate(project)
    project_sha = sha256_bytes(project_bytes)
    lock_sha = sha256_bytes(lock_bytes)
    if project_sha != gate.PYPROJECT_SHA256 or lock_sha != gate.LOCK_SHA256:
        raise AuditError("active pyproject.toml or uv.lock bytes differ from the reviewed contract")
    rows = lock_rows(lock_data)
    rows_sha = sha256_bytes(canonical_json(rows).encode("utf-8"))
    if manifest.get("pyproject_sha256") != project_sha or manifest.get("lock_sha256") != lock_sha or manifest.get("package_rows_sha256") != rows_sha:
        raise AuditError("license gate manifest is not bound to the active project/lock rows")
    contract_data = {
        "project_name": project_data.get("project", {}).get("name"),
        "project_version": project_data.get("project", {}).get("version"),
        "pyproject_bytes": len(project_bytes),
        "pyproject_sha256": project_sha,
        "uv_lock_bytes": len(lock_bytes),
        "uv_lock_sha256": lock_sha,
        "package_rows_sha256": rows_sha,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_gate_version": manifest.get("gate_version"),
        "manifest_dependency_audit_reference": manifest.get("dependency_audit_evidence"),
        "manifest_operator_approval_record": manifest.get("operator_approval"),
    }
    return project_data, lock_data, manifest, {"rows": rows, "raw_lock": lock_data, "contract": contract_data, "gate_sha256": sha256_file(project / "preflight_gate.py")}


def compare_multiset(expected: list[str], actual: list[str]) -> dict[str, Any]:
    expected_counts = Counter(expected)
    actual_counts = Counter(actual)
    missing = sorted((expected_counts - actual_counts).elements())
    unexpected = sorted((actual_counts - expected_counts).elements())
    duplicates = sorted(identity for identity, count in actual_counts.items() if count > 1)
    return {"expected": sorted(expected), "installed": sorted(actual), "missing": missing, "unexpected": unexpected, "duplicate_identities": duplicates, "exact": not missing and not unexpected}


def distribution_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if not name or not dist.version:
            continue
        records.append({"distribution": dist, "identity": identity(name, dist.version), "name": name, "version": dist.version, "location": str(Path(dist.locate_file("")))})
    return sorted(records, key=lambda item: (item["identity"], item["location"]))


def entry_path(dist: metadata.Distribution, entry: Any) -> Path | None:
    path = Path(dist.locate_file(entry))
    root = Path(dist.locate_file(""))
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        return None
    return path


def is_license_path(relative: str) -> bool:
    lower = Path(relative).name.casefold()
    return any(token in lower for token in LICENSE_TOKENS)


def metadata_facts(dist: metadata.Distribution) -> dict[str, Any]:
    classifiers = sorted(value for value in (dist.metadata.get_all("Classifier") or []) if value.startswith("License :: "))
    declared = dist.metadata.get("License")
    expression = dist.metadata.get("License-Expression")
    return {
        "license_declared": declared.strip() if isinstance(declared, str) and declared.strip() else None,
        "license_expression_declared": expression.strip() if isinstance(expression, str) and expression.strip() else None,
        "license_classifiers_declared": classifiers,
    }


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
        found.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return found, unsafe


def locked_sdist_artifact(row: dict[str, Any], raw_row: dict[str, Any] | None) -> dict[str, Any]:
    if raw_row is None or raw_row.get("source") != {"registry": "https://pypi.org/simple"}:
        raise AuditError(f"{row['name']} has no reviewed PyPI sdist fallback")
    artifact = raw_row.get("sdist")
    required = {"url", "hash", "size", "upload-time"}
    if (
        not isinstance(artifact, dict)
        or set(artifact) != required
        or not isinstance(artifact.get("url"), str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(artifact.get("hash")))
        or not isinstance(artifact.get("size"), int)
        or artifact["size"] <= 0
        or artifact["size"] > MAX_SDIST_BYTES
        or not isinstance(artifact.get("upload-time"), str)
    ):
        raise AuditError(f"{row['name']} locked sdist artifact is malformed")
    _validate_locked_sdist_url(artifact["url"], artifact["url"], initial=True)
    return artifact


def _validate_locked_sdist_url(value: str, expected: str, *, initial: bool) -> None:
    """Validate the exact files.pythonhosted.org URL before opening it."""
    if initial and value != expected:
        raise AuditError("initial locked sdist URL drifted")
    parsed = urlparse(value)
    expected_parsed = urlparse(expected)
    try:
        port = parsed.port
        expected_port = expected_parsed.port
    except ValueError as error:
        raise AuditError("locked sdist URL has an invalid port") from error
    for candidate in (parsed, expected_parsed):
        decoded_path = unquote(candidate.path)
        if (
            candidate.scheme != "https"
            or candidate.hostname != PYPI_HOST
            or candidate.username is not None
            or candidate.password is not None
            or candidate.query
            or candidate.fragment
            or candidate.params
            or not candidate.path.startswith("/packages/")
            or any(part in {"", ".", ".."} for part in decoded_path.split("/")[1:])
        ):
            raise AuditError("locked sdist URL is not an exact files.pythonhosted.org path")
    if port not in (None, 443) or expected_port not in (None, 443):
        raise AuditError("locked sdist URL has an unsafe port")
    if parsed.path != expected_parsed.path:
        raise AuditError("locked sdist redirect changed exact path")


def _safe_archive_member(name: str) -> str:
    if not isinstance(name, str):
        raise AuditError("archive member name is not text")
    if name.endswith("/"):
        name = name[:-1]
    if (
        not name
        or "\x00" in name
        or "\\" in name
        or name.startswith("/")
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise AuditError("unsafe locked sdist archive member path")
    return name


def archive_license_files(body: bytes, url: str) -> list[dict[str, Any]]:
    suffix = urlparse(url).path.casefold()
    if suffix.endswith(".zip"):
        kind = "zip"
    elif suffix.endswith((".tar.gz", ".tgz")):
        kind = "tar.gz"
    elif suffix.endswith((".tar.bz2", ".tbz2")):
        kind = "tar.bz2"
    elif suffix.endswith((".tar.xz", ".txz")):
        kind = "tar.xz"
    else:
        raise AuditError("unsupported locked sdist archive format")
    if len(body) > MAX_SDIST_BYTES:
        raise AuditError("locked sdist exceeds bounded download size")
    found: list[dict[str, Any]] = []
    names: set[str] = set()
    total = 0

    def add(name: str, payload: bytes) -> None:
        if not is_license_path(name):
            return
        if len(payload) > MAX_MEMBER_BYTES:
            raise AuditError("license archive member is oversized")
        found.append({"path": name, "bytes": len(payload), "sha256": sha256_bytes(payload), "content_base64": base64.b64encode(payload).decode("ascii"), "source": "locked-sdist"})

    try:
        if kind == "zip":
            with zipfile.ZipFile(io.BytesIO(body)) as archive:
                for index, info in enumerate(archive.infolist(), 1):
                    if index > MAX_ARCHIVE_MEMBERS:
                        raise AuditError("locked sdist archive has too many members")
                    name = _safe_archive_member(info.filename)
                    if name in names:
                        raise AuditError("locked sdist archive contains duplicate members")
                    names.add(name)
                    file_type = (info.external_attr >> 16) & 0o170000
                    if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                        raise AuditError("locked sdist archive contains a special member")
                    if info.is_dir() or file_type == stat.S_IFDIR:
                        continue
                    if info.file_size < 0 or info.file_size > MAX_MEMBER_BYTES:
                        raise AuditError("locked sdist archive member is oversized")
                    total += info.file_size
                    if total > MAX_ARCHIVE_BYTES:
                        raise AuditError("locked sdist archive is oversized")
                    payload = archive.read(info)
                    if len(payload) != info.file_size:
                        raise AuditError("locked sdist archive member is truncated")
                    add(name, payload)
        else:
            mode = {"tar.gz": "r:gz", "tar.bz2": "r:bz2", "tar.xz": "r:xz"}[kind]
            with tarfile.open(fileobj=io.BytesIO(body), mode=mode) as archive:
                for index, member in enumerate(archive, 1):
                    if index > MAX_ARCHIVE_MEMBERS:
                        raise AuditError("locked sdist archive has too many members")
                    name = _safe_archive_member(member.name)
                    if name in names:
                        raise AuditError("locked sdist archive contains duplicate members")
                    names.add(name)
                    if member.isdir():
                        continue
                    if not member.isfile() or member.size < 0 or member.size > MAX_MEMBER_BYTES:
                        raise AuditError("locked sdist archive contains an unsafe member type")
                    total += member.size
                    if total > MAX_ARCHIVE_BYTES:
                        raise AuditError("locked sdist archive is oversized")
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise AuditError("locked sdist archive member cannot be read")
                    payload = stream.read(MAX_MEMBER_BYTES + 1)
                    if len(payload) != member.size:
                        raise AuditError("locked sdist archive member is truncated")
                    add(name, payload)
    except AuditError:
        raise
    except (OSError, EOFError, tarfile.TarError, zipfile.BadZipFile) as error:
        raise AuditError(f"locked sdist archive is unreadable: {type(error).__name__}") from error
    if not found:
        raise AuditError("locked sdist has no LICENSE/LICENCE/COPYING/NOTICE/COPYRIGHT member")
    return found


def fetch_locked_sdist(row: dict[str, Any], raw_row: dict[str, Any] | None, fetcher: Any = None) -> dict[str, Any]:
    artifact = locked_sdist_artifact(row, raw_row)
    url = artifact["url"]
    trace = [url]
    final = url
    try:
        if fetcher is not None:
            final, body = fetcher(url)
        else:
            class LockedSdistRedirects(HTTPRedirectHandler):
                def redirect_request(self, request: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Request:
                    if len(trace) - 1 >= MAX_SDIST_REDIRECTS:
                        raise AuditError("locked sdist redirect limit exceeded")
                    resolved = urljoin(request.full_url, newurl)
                    _validate_locked_sdist_url(resolved, url, initial=False)
                    trace.append(resolved)
                    return super().redirect_request(request, fp, code, msg, headers, resolved)

            with build_opener(LockedSdistRedirects()).open(Request(url, headers={"User-Agent": "vokra-speecht5-dependency-audit/1"}), timeout=30) as response:
                final = response.geturl()
                body = response.read(MAX_SDIST_BYTES + 1)
        _validate_locked_sdist_url(final, url, initial=False)
        if not isinstance(body, bytes) or len(body) != artifact["size"] or sha256_bytes(body) != artifact["hash"].removeprefix("sha256:"):
            raise AuditError("locked sdist size or sha256 does not match uv.lock")
        license_files = archive_license_files(body, url)
    except LockedSdistFetchError:
        raise
    except (AuditError, OSError, UnicodeError, ValueError, TypeError, HTTPError, URLError) as error:
        raise LockedSdistFetchError(f"locked sdist fetch or archive validation failed: {type(error).__name__}") from error
    return {
        "status": "ACQUIRED_LOCKED_SDIST_LICENSE_BYTES",
        "requested_url": url,
        "final_url": final,
        "redirect_trace": trace,
        "size": len(body),
        "sha256": sha256_bytes(body),
        "license_files": license_files,
        "auditor_network_requests": 1,
    }


def elf_facts(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        magic = stream.read(4)
    if magic != ELF_MAGIC:
        return {"format": "non-elf", "needed": [], "inspection": "not-applicable"}
    try:
        result = subprocess.run(["readelf", "-d", str(path)], capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        return {"format": "elf", "needed": [], "inspection": "error", "error_type": type(error).__name__}
    needed = sorted(set(match.group(1) for match in re.finditer(r"\(NEEDED\).*?\[([^]]+)\]", result.stdout)))
    inspection = "ok" if result.returncode == 0 else "error"
    facts: dict[str, Any] = {"format": "elf", "needed": needed, "inspection": inspection, "readelf_returncode": result.returncode}
    if inspection == "error":
        facts["error"] = result.stderr.strip()[-500:]
    return facts


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
            facts = elf_facts(path)
            found.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path), "bundled": True, "native": facts})
        except OSError as error:
            unsafe.append(f"{relative}:read-failed:{type(error).__name__}")
    return found, unsafe


def inspect_package(row: dict[str, Any], candidates: list[dict[str, Any]], raw_row: dict[str, Any] | None) -> tuple[dict[str, Any], list[str], int]:
    key = identity(row["name"], row["version"])
    record = candidates[0] if len(candidates) == 1 else None
    package: dict[str, Any] = {"lock": {"name": row["name"], "version": row["version"], "source": row["source"], "sdist": raw_row.get("sdist") if raw_row else None}, "installed": None}
    failures: list[str] = []
    if record is None:
        failures.append(f"installed closure missing or duplicated: {key}")
        return package, failures, 0
    dist = record["distribution"]
    licenses, unsafe_licenses = publisher_files(dist)
    native, unsafe_native = native_files(dist)
    facts = metadata_facts(dist)
    sdist_license: dict[str, Any] | None = None
    network_requests = 0
    if not licenses:
        try:
            sdist_license = fetch_locked_sdist(row, raw_row)
            network_requests = int(sdist_license["auditor_network_requests"])
        except LockedSdistFetchError as error:
            network_requests = error.network_requests
            failures.append(f"locked sdist license fallback blocked: {key}:{type(error).__name__}")
        except (AuditError, OSError, UnicodeError, ValueError, TypeError, HTTPError, URLError) as error:
            failures.append(f"locked sdist license fallback blocked: {key}:{type(error).__name__}")
    package["installed"] = {"name": dist.metadata.get("Name"), "version": dist.version, "identity": key, "location": record["location"], **facts, "publisher_license_files": licenses, "locked_sdist_license_fallback": sdist_license, "native_files": native, "bundled_libraries": native}
    if not licenses and not (sdist_license and sdist_license.get("license_files")):
        failures.append(f"publisher license file evidence missing: {key}")
    failures.extend(f"unsafe publisher license path: {key}:{item}" for item in unsafe_licenses)
    failures.extend(f"unsafe native path: {key}:{item}" for item in unsafe_native)
    failures.extend(f"ELF NEEDED inspection failed: {key}:{item['path']}" for item in native if item["native"].get("inspection") == "error")
    return package, failures, network_requests


def build_facts(project_data: dict[str, Any], rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    values = project_data.get("tool", {}).get("uv", {}).get("build-constraint-dependencies", [])
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise AuditError("uv build constraints are malformed")
    constraints = []
    for value in values:
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^=]+)", value)
        if match is None:
            raise AuditError(f"uv build constraint is not exact: {value}")
        constraints.append({"name": match.group(1), "specifier": value, "normalized_name": normalized_name(match.group(1))})
    active_names = {normalized_name(row["name"]) for row in rows if row["source"] != {"virtual": "."}}
    build_only = sorted(item["normalized_name"] for item in constraints if item["normalized_name"] not in active_names)
    overlap = sorted(item["normalized_name"] for item in constraints if item["normalized_name"] in active_names)
    installed_ids = {item["identity"] for item in records}
    observed_build_only = sorted(item for item in installed_ids if normalized_name(item.split("==", 1)[0]) in set(build_only))
    failures = [f"isolated build-only dependency leaked into final environment: {item}" for item in observed_build_only]
    return {"constraints": constraints, "build_only_by_lock_boundary": build_only, "runtime_overlap": overlap, "observed_build_only_installed": observed_build_only, "build_only_absent": not observed_build_only}, failures


def repository_facts(project: Path) -> tuple[dict[str, Any], list[str]]:
    repository = project.parents[2]
    try:
        head = subprocess.run(["git", "-C", str(repository), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=all"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise AuditError(f"git checkout identity unavailable: {type(error).__name__}") from error
    failures = []
    if not HEX40.fullmatch(head):
        failures.append("git HEAD is not an exact commit")
    if status:
        failures.append("git checkout is dirty")
    return {"root": str(repository), "head": head, "clean": not status, "audit_script_sha256": sha256_file(Path(__file__).resolve())}, failures


def audit_environment(project: Path) -> dict[str, Any]:
    project_data, lock_data, manifest, binding = contract(project)
    rows = binding["rows"]
    repository, repository_failures = repository_facts(project)
    records = distribution_records()
    expected = [identity(row["name"], row["version"]) for row in rows if row["source"] != {"virtual": "."}]
    actual = [record["identity"] for record in records]
    closure = compare_multiset(expected, actual)
    by_identity: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_identity.setdefault(record["identity"], []).append(record)
    raw_rows = lock_data.get("package")
    if not isinstance(raw_rows, list):
        raise AuditError("uv.lock package rows are missing")
    packages: list[dict[str, Any]] = []
    failures = list(repository_failures)
    network_requests = 0
    for row in rows:
        if row["source"] == {"virtual": "."}:
            continue
        matches = [candidate for candidate in raw_rows if candidate.get("name") == row["name"] and candidate.get("version") == row["version"] and candidate.get("source") == row["source"]]
        if len(matches) != 1:
            raise AuditError(f"uv.lock raw package row cannot bind: {identity(row['name'], row['version'])}")
        package, package_failures, package_requests = inspect_package(row, by_identity.get(identity(row["name"], row["version"]), []), matches[0])
        packages.append(package)
        failures.extend(package_failures)
        network_requests += package_requests
    if not closure["exact"]:
        failures.append("installed normalized name/version closure does not exactly match uv.lock")
    if sys.platform != "linux" or platform.machine().casefold() not in {"x86_64", "amd64"}:
        failures.append(f"audit host is not Linux x86_64: {sys.platform}/{platform.machine()}")
    if sys.version_info[:2] != (3, 12):
        failures.append(f"audit Python is not 3.12: {platform.python_version()}")
    build, build_failures = build_facts(project_data, rows, records)
    failures.extend(build_failures)
    native = [item for package in packages if package["installed"] for item in package["installed"]["native_files"]]
    missing_publisher_files = sorted(package["lock"]["name"] for package in packages if package["installed"] and not package["installed"]["publisher_license_files"])
    missing_license_evidence = sorted(package["lock"]["name"] for package in packages if package["installed"] and not package["installed"]["publisher_license_files"] and not (package["installed"].get("locked_sdist_license_fallback") or {}).get("license_files"))
    report = {
        "schema": SCHEMA,
        "status": "PASS" if not failures else "BLOCKED",
        "audit_scope": "fresh synchronized Linux x86_64 Python 3.12 environment; model-free dependency/license/native/bundled facts",
        "policy": {"license_classification": "NOT_PERFORMED", "owner_signoff": "NOT_PERFORMED", "model_acquisition": "NONE", "source_model_acquisition": "NONE", "torch_imported": False, "model_code_imported": False, "cargo_invoked": False, "upload_performed": False},
        "repository": repository,
        "environment": {"python": platform.python_version(), "platform": sys.platform, "machine": platform.machine(), "readelf_required": True, "auditor_network_requests": network_requests, "auditor_network_scope": "exact locked PyPI sdist license fallback only; no model/source requests"},
        "contract": binding["contract"],
        "lock_rows": {"all_rows": rows, "active_registry_rows": [row for row in rows if row["source"] != {"virtual": "."}], "virtual_rows": [row for row in rows if row["source"] == {"virtual": "."}], "all_rows_accounted": True},
        "closure": closure,
        "packages": packages,
        "license_facts": {"packages": len(packages), "publisher_license_files_missing": missing_publisher_files, "publisher_license_evidence_missing": missing_license_evidence, "locked_sdist_fallback_packages": sorted(package["lock"]["name"] for package in packages if package["installed"] and package["installed"].get("locked_sdist_license_fallback")), "classification": "raw publisher metadata/files only; locked sdist license bytes/hashes"},
        "native_facts": {"bundled_library_count": len(native), "files": native},
        "build_only_facts": build,
        "model_source_facts": {"requested_files": [], "non_license_requests": [], "proof": "auditor has no model/source acquisition or model import path"},
        "failures": sorted(set(failures)),
    }
    return report


def compact_report(full: dict[str, Any], full_sha: str) -> dict[str, Any]:
    contract_data = full.get("contract", {})
    return {
        "schema": COMPACT_SCHEMA,
        "full_audit_sha256": full_sha,
        "audit_scope": full.get("audit_scope"),
        "repository": full.get("repository"),
        "environment": full.get("environment"),
        "inputs": {"pyproject_sha256": contract_data.get("pyproject_sha256"), "uv_lock_sha256": contract_data.get("uv_lock_sha256")},
        "status": full.get("status"),
        "closure": full.get("closure"),
        "license_facts": full.get("license_facts"),
        "native_facts": full.get("native_facts"),
        "build_only_facts": full.get("build_only_facts"),
        "numpy_source_build": {"setup_args": ["-Dblas=none", "-Dlapack=none"], "no_binary_package": ["numpy"]},
        "operator_approval": "PENDING_REVIEW",
        "manifest_operator_approval_record": contract_data.get("manifest_operator_approval_record"),
        "model_acquisition": {"requested_files": [], "non_license_requests": []},
    }


def output_paths(full_output: Path, compact_output: Path) -> None:
    for path, label in ((full_output, "full output"), (compact_output, "compact output")):
        safe_absolute_path(path, label)
        if path.exists() or path.is_symlink():
            raise AuditError(f"{label} must be absent: {path}")
    if full_output == compact_output or full_output in compact_output.parents or compact_output in full_output.parents:
        raise AuditError("full and compact outputs overlap")


def run(project: Path, full_output: Path, compact_output: Path) -> int:
    output_paths(full_output, compact_output)
    safe_absolute_path(project, "--project")
    repository = project.parents[2]
    if paths_overlap(full_output, repository) or paths_overlap(compact_output, repository) or paths_overlap(full_output, project) or paths_overlap(compact_output, project):
        raise AuditError("audit outputs overlap the checkout or SpeechT5 project")
    try:
        report = audit_environment(project)
    except (AuditError, OSError, UnicodeError, ValueError, TypeError, KeyError, AttributeError, subprocess.SubprocessError) as error:
        report = {"schema": SCHEMA, "status": "BLOCKED", "policy": {"license_classification": "NOT_PERFORMED", "owner_signoff": "NOT_PERFORMED", "model_acquisition": "NONE", "source_model_acquisition": "NONE", "torch_imported": False, "model_code_imported": False, "cargo_invoked": False, "upload_performed": False}, "failures": [f"audit aborted: {type(error).__name__}: {error}"]}
    full_output.parent.mkdir(parents=True, exist_ok=True)
    full_output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    compact_output.parent.mkdir(parents=True, exist_ok=True)
    compact_output.write_text(canonical_json(compact_report(report, sha256_file(full_output))) + "\n", encoding="utf-8")
    if report.get("failures"):
        print("speecht5 dependency audit: BLOCKED: " + "; ".join(report["failures"]), file=sys.stderr)
        return 2
    print(f"SPEECHT5_DEPENDENCY_AUDIT status=PASS full_sha256={sha256_file(full_output)} compact={compact_output}")
    return 0


def self_test() -> int:
    assert normalized_name("Foo_bar") == "foo-bar"
    assert identity("torch", "2.4.1+CPU") == "torch==2.4.1+cpu"
    assert compare_multiset(["a==1"], ["a==1"])["exact"]
    mismatch = compare_multiset(["a==1"], ["a==2"])
    assert mismatch["missing"] == ["a==1"] and mismatch["unexpected"] == ["a==2"]
    assert is_license_path("pkg/LICENSE.txt") and is_license_path("pkg/LICENCE.txt") and is_license_path("pkg/COPYING")
    assert not is_license_path("pkg/README.md")
    with __import__("tempfile").TemporaryDirectory(prefix="speecht5-dependency-audit-selftest-") as directory:
        root = Path(directory).resolve()
        archive_buffer = io.BytesIO()
        with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
            payload = b"demo licence bytes"
            member = tarfile.TarInfo("demo-1/LICENCE")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        archive_body = archive_buffer.getvalue()
        artifact = {
            "url": "https://files.pythonhosted.org/packages/aa/bb/demo-1.tar.gz",
            "hash": "sha256:" + sha256_bytes(archive_body),
            "size": len(archive_body),
            "upload-time": "2026-01-01T00:00:00Z",
        }
        sdist_row = {"name": "demo", "version": "1", "source": {"registry": "https://pypi.org/simple"}, "sdist": artifact}
        fetched = fetch_locked_sdist(sdist_row, sdist_row, lambda url: (url, archive_body))
        assert fetched["auditor_network_requests"] == 1
        assert fetched["license_files"] == [{"path": "demo-1/LICENCE", "bytes": len(payload), "sha256": sha256_bytes(payload), "content_base64": base64.b64encode(payload).decode("ascii"), "source": "locked-sdist"}]
        try:
            fetch_locked_sdist(sdist_row, sdist_row, lambda url: (_ for _ in ()).throw(URLError("offline self-test failure")))
        except LockedSdistFetchError as error:
            assert error.network_requests == 1
        else:
            raise AssertionError("failed locked sdist fetch was not reported")
        for bad_url in (
            "http://files.pythonhosted.org/packages/aa/bb/demo-1.tar.gz",
            "https://user:pass@files.pythonhosted.org/packages/aa/bb/demo-1.tar.gz",
            "https://files.pythonhosted.org:444/packages/aa/bb/demo-1.tar.gz",
            "https://files.pythonhosted.org/packages/aa/bb/demo-1.tar.gz?x=1",
            "https://files.pythonhosted.org/packages/aa/bb/../demo-1.tar.gz",
            "https://evil.example/packages/aa/bb/demo-1.tar.gz",
        ):
            called = False
            try:
                bad_row = {**sdist_row, "sdist": {**artifact, "url": bad_url}}
                fetch_locked_sdist(bad_row, bad_row, lambda url: (_ for _ in ()).throw(AssertionError("unsafe URL reached fetcher")))
            except AuditError:
                called = True
            assert called
        try:
            fetch_locked_sdist(sdist_row, sdist_row, lambda url: (url + "?redirected=1", archive_body))
        except AuditError:
            pass
        else:
            raise AssertionError("unsafe locked sdist redirect accepted")
        unsafe_buffer = io.BytesIO()
        with tarfile.open(fileobj=unsafe_buffer, mode="w:gz") as archive:
            payload = b"unsafe"
            member = tarfile.TarInfo("../LICENCE")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        try:
            archive_license_files(unsafe_buffer.getvalue(), artifact["url"])
        except AuditError:
            pass
        else:
            raise AssertionError("unsafe archive path accepted")
        symlink_buffer = io.BytesIO()
        with tarfile.open(fileobj=symlink_buffer, mode="w:gz") as archive:
            member = tarfile.TarInfo("demo-1/LICENCE")
            member.type = tarfile.SYMTYPE
            member.linkname = "other"
            archive.addfile(member)
        try:
            archive_license_files(symlink_buffer.getvalue(), artifact["url"])
        except AuditError:
            pass
        else:
            raise AssertionError("unsafe archive member type accepted")
        duplicate = root / "duplicate.json"
        duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
        try:
            strict_json(duplicate)
        except AuditError:
            pass
        else:
            raise AssertionError("duplicate JSON key accepted")
        full = root / "full.json"
        compact = root / "compact.json"
        output_paths(full, compact)
        full.write_text("x", encoding="utf-8")
        try:
            output_paths(full, compact)
        except AuditError:
            pass
        else:
            raise AssertionError("existing output accepted")
        compact = compact_report({"repository": {"root": str(root), "head": "a" * 40, "clean": True, "audit_script_sha256": "b" * 64}, "environment": {"auditor_network_requests": 1, "auditor_network_scope": "exact locked PyPI sdist license fallback only; no model/source requests"}, "contract": {"pyproject_sha256": "c" * 64, "uv_lock_sha256": "d" * 64}, "closure": {"exact": True}, "license_facts": {}, "native_facts": {}, "build_only_facts": {}}, "e" * 64)
        assert compact["repository"]["root"] == str(root)
        assert HEX40.fullmatch(compact["repository"]["head"])
        assert compact["repository"]["clean"] is True
        assert HEX64.fullmatch(compact["repository"]["audit_script_sha256"])
        assert compact["environment"]["auditor_network_requests"] == 1
        assert "license fallback only" in compact["environment"]["auditor_network_scope"]
    print("speecht5 dependency audit: self-test PASS (offline, model-free, no network)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path)
    parser.add_argument("--full-output", type=Path)
    parser.add_argument("--compact-output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.project, args.full_output, args.compact_output)):
            parser.error("--self-test accepts no project/output arguments")
        return self_test()
    if args.project is None or args.full_output is None or args.compact_output is None:
        parser.error("--project, --full-output, and --compact-output are required")
    return run(args.project, args.full_output, args.compact_output)


if __name__ == "__main__":
    raise SystemExit(main())
