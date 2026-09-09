#!/usr/bin/env -S uv run --frozen --project tools/parity/irodori_text_block_reference --python 3.12 python
"""Model-free dependency/license audit for the isolated Irodori TextBlock route.

The audit is intentionally independent of the Irodori checkout.  It validates
the committed project and lock, then (on the disposable Linux worker) inspects
only the installed distributions, publisher license files, locked source
archives, and ELF payloads.  It never imports torch or Irodori code and never
obtains a checkpoint.  A clean factual audit still produces a blocked report:
dependency approval belongs to the owner and is not inferred by this tool.
"""

from __future__ import annotations

import argparse
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
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse


SCHEMA = "vokra-irodori-text-block-dependency-audit-v1"
PYPI_REGISTRY = "https://pypi.org/simple"
PYPI_HOST = "files.pythonhosted.org"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
MAX_SDIST_BYTES = 64 * 1024 * 1024
MAX_LICENSE_BYTES = 2 * 1024 * 1024
MAX_LICENSE_TOTAL = 4 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 20_000
ELF_MAGIC = b"\x7fELF"
LICENSE_BASENAMES = {
    "license", "licence", "copying", "notice", "copyright", "eula",
    "nvidia_sla", "nvidia-sla", "end_user_license", "end-user-license",
}
FORBIDDEN_TOKENS = ("cuda", "nvidia", "triton")
DIRECT = {"safetensors": "0.8.0", "torch": "2.13.0"}
EXPECTED_PACKAGES = {
    "filelock": {"3.32.5"},
    "fsspec": {"2026.7.0"},
    "jinja2": {"3.1.6"},
    "markupsafe": {"3.0.3"},
    "mpmath": {"1.3.0"},
    "networkx": {"3.6.1"},
    "safetensors": {"0.8.0"},
    "sympy": {"1.14.0"},
    "torch": {"2.13.0", "2.13.0+cpu"},
    "typing-extensions": {"4.16.0"},
    "vokra-irodori-text-block-reference": {"0.1.0"},
}
REFERENCE_PROJECT = Path(__file__).resolve().parent
EXPECTED_HEAD_PATTERN = re.compile(r"[0-9a-f]{40}")
REPORT_KEYS = frozenset(
    {
        "schema",
        "status",
        "review",
        "project",
        "closure",
        "packages",
        "failures",
        "environment",
        "git",
        "model_acquisition",
        "publication",
    }
)
PACKAGE_KEYS = frozenset({"lock", "installed"})
INSTALLED_KEYS = frozenset(
    {
        "name",
        "version",
        "license",
        "license_expression",
        "license_classifiers",
        "publisher_license_files",
        "unsafe_license_paths",
        "locked_sdist_license",
        "native_payloads",
        "native_payload_errors",
    }
)
ENVIRONMENT_KEYS = frozenset(
    {
        "python",
        "platform",
        "machine",
        "readelf_required",
        "irodori_source_fetched",
        "irodori_source_imported",
        "model_code_imported",
        "weights_acquired",
        "weights_imported",
        "weights_executed",
        "cargo_invoked",
    }
)


class AuditError(ValueError):
    """A structural, provenance, or safety failure."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hash a file incrementally so a large native wheel never enters RAM."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_bounded(path: Path, maximum: int) -> bytes:
    """Read a publisher file with a hard upper bound (including TOCTOU growth)."""

    with path.open("rb") as handle:
        value = handle.read(maximum + 1)
    if len(value) > maximum:
        raise AuditError(f"publisher license file exceeds {maximum} bytes: {path}")
    return value


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def strict_json(data: bytes | str) -> Any:
    """Decode JSON while rejecting duplicate keys instead of silently merging."""

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AuditError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(data, object_pairs_hook=unique)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise AuditError(f"invalid JSON evidence: {exc}") from exc


def _canonical_string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise AuditError(f"{label} must be a list of strings")
    if value != sorted(set(value)):
        raise AuditError(f"{label} must be sorted and duplicate-free")
    return value


def norm_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value.strip()).casefold()


def identity(name: str, version: str) -> str:
    return f"{norm_name(name)}=={re.sub(r'\s+', '', version).casefold()}"


def contains_forbidden(value: str) -> bool:
    lowered = value.casefold()
    # Native NEEDED entries are commonly named ``libcudart.so`` or
    # ``libnvidia-*.so``; a separator-only test would miss those payloads.
    return any(token in lowered for token in FORBIDDEN_TOKENS)


def license_candidate(path: str) -> bool:
    name = PurePosixPath(path).name.casefold()
    return name in LICENSE_BASENAMES or any(
        name.startswith(prefix + separator)
        for prefix in LICENSE_BASENAMES
        for separator in (".", "-", "_")
    )


def safe_relative(path: str) -> str:
    if not path or "\x00" in path or "\\" in path or path.startswith("/"):
        raise AuditError(f"unsafe archive path: {path!r}")
    clean = path.rstrip("/")
    if not clean or any(part in {"", ".", ".."} for part in clean.split("/")):
        raise AuditError(f"archive path traversal: {path!r}")
    return clean


def validate_expected_head(value: str | None) -> str:
    if value is None or EXPECTED_HEAD_PATTERN.fullmatch(value) is None:
        raise AuditError("--expected-head requires lowercase 40-hex")
    return value


def validate_project(project: Path = REFERENCE_PROJECT) -> tuple[dict[str, Any], list[dict[str, Any]], bytes, bytes]:
    project_path, lock_path = project / "pyproject.toml", project / "uv.lock"
    if not project.is_dir() or project.is_symlink() or not project_path.is_file() or project_path.is_symlink() or not lock_path.is_file() or lock_path.is_symlink():
        raise AuditError("Irodori TextBlock reference project or lock is missing/symlinked")
    project_bytes, lock_bytes = project_path.read_bytes(), lock_path.read_bytes()
    project_data = tomllib.loads(project_bytes.decode("utf-8"))
    lock_data = tomllib.loads(lock_bytes.decode("utf-8"))
    table = project_data.get("project")
    if not isinstance(table, dict) or table.get("name") != "vokra-irodori-text-block-reference" or table.get("version") != "0.1.0" or table.get("requires-python") != "==3.12.*":
        raise AuditError("Irodori TextBlock project identity drifted")
    deps = table.get("dependencies")
    if deps != ["safetensors==0.8.0", "torch==2.13.0"]:
        raise AuditError("Irodori TextBlock direct dependency pins drifted")
    uv = project_data.get("tool", {}).get("uv", {})
    if uv.get("package") is not False or uv.get("override-dependencies") != ["setuptools ; python_version < '0'"]:
        raise AuditError("Irodori TextBlock uv isolation policy drifted")
    contract = project_data.get("tool", {}).get("vokra", {}).get("irodori_text_block_reference")
    expected_contract = {
        "status": "PENDING_DEPENDENCY_LICENSE_AUDIT",
        "dependency_license_status": "PENDING_OWNER_REVIEW",
        "execution": "NOT_AUTHORIZED",
        "publication": "NO_UPLOAD",
    }
    if contract != expected_contract:
        raise AuditError("Irodori TextBlock owner-review contract drifted")
    if project_data.get("tool", {}).get("uv", {}).get("sources", {}).get("torch") != {
        "index": "pytorch-cpu", "marker": "sys_platform == 'linux' or sys_platform == 'win32'",
    }:
        raise AuditError("Irodori TextBlock CPU source policy drifted")
    indexes = uv.get("index")
    if indexes != [{"name": "pytorch-cpu", "url": TORCH_CPU_INDEX, "explicit": True}]:
        raise AuditError("Irodori TextBlock CPU index drifted")
    if (lock_data.get("version"), lock_data.get("revision"), lock_data.get("requires-python")) != (1, 3, "==3.12.*"):
        raise AuditError("Irodori TextBlock lock format or Python pin drifted")
    rows = lock_data.get("package")
    expected_row_count = sum(len(versions) for versions in EXPECTED_PACKAGES.values())
    if not isinstance(rows, list) or len(rows) != expected_row_count:
        raise AuditError("Irodori TextBlock lock package count drifted")
    observed: dict[str, set[str]] = {}
    seen: set[tuple[str, str, str]] = set()
    for raw in rows:
        if not isinstance(raw, dict) or not isinstance(raw.get("name"), str) or not isinstance(raw.get("version"), str):
            raise AuditError("Irodori TextBlock lock has malformed package row")
        name, version = raw["name"], raw["version"]
        source = raw.get("source")
        if not isinstance(source, dict) or len(source) != 1 or source not in ({"registry": PYPI_REGISTRY}, {"registry": TORCH_CPU_INDEX}, {"virtual": "."}):
            raise AuditError(f"unapproved lock source for {name}")
        key = (norm_name(name), version, canonical(source))
        if key in seen:
            raise AuditError(f"duplicate lock identity: {name}=={version}")
        seen.add(key)
        observed.setdefault(norm_name(name), set()).add(version)
        if source == {"virtual": "."}:
            if norm_name(name) != "vokra-irodori-text-block-reference":
                raise AuditError("unexpected virtual lock project")
            continue
        if source == {"registry": PYPI_REGISTRY}:
            sdist = raw.get("sdist")
            # The inactive macOS torch row is wheel-only in the lock.  It is
            # retained to prove the platform split, but is not part of the
            # Linux worker's audited installation.
            inactive_macos_torch = norm_name(name) == "torch" and raw.get("resolution-markers") == ["sys_platform != 'linux' and sys_platform != 'win32'"]
            if inactive_macos_torch and sdist is None:
                continue
            if not isinstance(sdist, dict) or set(sdist) != {"url", "hash", "size", "upload-time"}:
                raise AuditError(f"locked PyPI sdist identity missing for {name}")
            if not isinstance(sdist["url"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(sdist["hash"])) or not isinstance(sdist["size"], int) or not 0 < sdist["size"] <= MAX_SDIST_BYTES:
                raise AuditError(f"locked PyPI sdist identity malformed for {name}")
        elif raw.get("sdist") is not None:
            raise AuditError(f"CPU-index row unexpectedly has sdist: {name}")
        if contains_forbidden(name):
            raise AuditError(f"forbidden CUDA/NVIDIA/Triton package: {name}")
        for dependency in raw.get("dependencies", []):
            if isinstance(dependency, dict) and contains_forbidden(str(dependency.get("name", ""))):
                raise AuditError(f"forbidden transitive package: {dependency}")
    if observed != EXPECTED_PACKAGES:
        raise AuditError(f"Irodori TextBlock lock package set drifted: {observed!r}")
    return project_data, rows, project_bytes, lock_bytes


def active_linux(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if row["source"] == {"virtual": "."}:
            continue
        markers = row.get("resolution-markers", [])
        if not isinstance(markers, list) or any(not isinstance(item, str) for item in markers):
            raise AuditError(f"invalid resolution marker for {row['name']}")
        if not markers or "sys_platform == 'linux' or sys_platform == 'win32'" in markers:
            result.append(row)
        elif markers != ["sys_platform != 'linux' and sys_platform != 'win32'"]:
            raise AuditError(f"unsupported Linux resolution marker for {row['name']}: {markers!r}")
    return sorted(result, key=lambda row: (norm_name(row["name"]), row["version"]))


def installed_distributions() -> dict[str, list[metadata.Distribution]]:
    result: dict[str, list[metadata.Distribution]] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            result.setdefault(identity(name, dist.version), []).append(dist)
    return result


def safe_dist_file(dist: metadata.Distribution, entry: Any) -> Path | None:
    root, path = Path(dist.locate_file("")), Path(dist.locate_file(entry))
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    if root.is_symlink() or any((root / part).is_symlink() for part in relative.parts):
        return None
    resolved_root, resolved = root.resolve(), path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        return None
    return resolved


def publisher_files(dist: metadata.Distribution) -> tuple[list[dict[str, Any]], list[str]]:
    files, unsafe = [], []
    for entry in sorted(dist.files or [], key=str):
        relative = str(entry)
        if not license_candidate(relative):
            continue
        path = safe_dist_file(dist, entry)
        if path is None or path.is_symlink() or not path.is_file():
            unsafe.append(relative)
            continue
        size = path.stat().st_size
        if size > MAX_LICENSE_BYTES:
            unsafe.append(f"{relative}:oversized")
            continue
        files.append({"path": relative, "size": size, "sha256": sha256_bytes(read_bounded(path, MAX_LICENSE_BYTES))})
    return files, unsafe


def native_payloads(dist: metadata.Distribution) -> tuple[list[dict[str, Any]], list[str]]:
    payloads, errors = [], []
    for entry in sorted(dist.files or [], key=str):
        path = safe_dist_file(dist, entry)
        if path is None or path.is_symlink() or not path.is_file():
            continue
        try:
            with path.open("rb") as handle:
                magic = handle.read(4)
            if magic != ELF_MAGIC:
                continue
            try:
                result = subprocess.run(["readelf", "-d", str(path)], capture_output=True, text=True, timeout=60, check=False)
                needed = sorted(match.group(1) for line in result.stdout.splitlines() if (match := re.search(r"\(NEEDED\).*\[([^]]+)\]", line)))
                inspection = "ok" if result.returncode == 0 else "error"
                error = None if inspection == "ok" else f"readelf exit {result.returncode}"
            except (OSError, subprocess.TimeoutExpired) as exc:
                needed, inspection, error = [], type(exc).__name__, str(exc)
            payloads.append({"path": str(entry), "size": path.stat().st_size, "sha256": sha256_file(path), "needed": needed, "inspection": inspection, "error": error})
        except OSError as exc:
            errors.append(f"{entry}: {exc}")
    return payloads, errors


def archive_license_files(url: str, body: bytes) -> list[dict[str, Any]]:
    if len(body) > MAX_SDIST_BYTES:
        raise AuditError("locked sdist exceeds size bound")
    if url.endswith((".tar.gz", ".tgz")):
        opener = lambda: tarfile.open(fileobj=io.BytesIO(body), mode="r:gz")
    elif url.endswith(".tar.bz2"):
        opener = lambda: tarfile.open(fileobj=io.BytesIO(body), mode="r:bz2")
    elif url.endswith(".tar.xz"):
        opener = lambda: tarfile.open(fileobj=io.BytesIO(body), mode="r:xz")
    elif url.endswith(".zip"):
        opener = lambda: zipfile.ZipFile(io.BytesIO(body))
    else:
        raise AuditError("unsupported locked sdist format")
    result, total = [], 0
    with opener() as archive:
        members = archive.infolist() if isinstance(archive, zipfile.ZipFile) else archive.getmembers()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise AuditError("locked sdist has too many members")
        for info in members:
            name = info.filename if isinstance(archive, zipfile.ZipFile) else info.name
            clean = safe_relative(name)
            is_file = not (info.is_dir() if isinstance(archive, zipfile.ZipFile) else not info.isfile())
            if not is_file or not license_candidate(clean):
                continue
            size = info.file_size if isinstance(archive, zipfile.ZipFile) else info.size
            if size > MAX_LICENSE_BYTES or total + size > MAX_LICENSE_TOTAL:
                raise AuditError("locked sdist license bytes exceed bound")
            handle = archive.open(info) if isinstance(archive, zipfile.ZipFile) else archive.extractfile(info)
            if handle is None:
                raise AuditError("locked sdist license member cannot be read")
            with handle:
                data = handle.read(MAX_LICENSE_BYTES + 1)
            if len(data) != size:
                raise AuditError("locked sdist license member size changed")
            total += size
            result.append({"path": clean, "size": size, "sha256": sha256_bytes(data)})
    if not result:
        raise AuditError("locked sdist has no license candidate")
    return sorted(result, key=lambda item: item["path"])


def locked_sdist_license(row: dict[str, Any]) -> dict[str, Any]:
    if row["source"] != {"registry": PYPI_REGISTRY}:
        return {"status": "NOT_APPLICABLE_NON_PYPI_SOURCE", "license_files": []}
    artifact = row["sdist"]
    url = artifact["url"]
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != PYPI_HOST or parsed.query or parsed.fragment:
        raise AuditError(f"locked sdist URL escaped PyPI host: {url}")
    request = urllib.request.Request(url, headers={"Accept": "application/octet-stream", "User-Agent": "vokra-irodori-dependency-audit/1"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            final = response.geturl()
            body = response.read(MAX_SDIST_BYTES + 1)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
        return {"status": "BLOCKED_LOCKED_SDIST_FETCH", "error": type(exc).__name__, "license_files": []}
    final_parts = urlparse(final)
    if final_parts.scheme != "https" or final_parts.hostname != PYPI_HOST or final_parts.path != parsed.path:
        raise AuditError(f"locked sdist redirect escaped or changed path: {final}")
    digest = sha256_bytes(body)
    if len(body) != artifact["size"] or "sha256:" + digest != artifact["hash"]:
        return {"status": "BLOCKED_LOCKED_SDIST_IDENTITY", "observed_size": len(body), "observed_sha256": "sha256:" + digest, "license_files": []}
    try:
        files = archive_license_files(final, body)
    except AuditError as exc:
        return {"status": "BLOCKED_LOCKED_SDIST_LICENSE", "error": str(exc), "license_files": []}
    return {"status": "ACQUIRED_LOCKED_SDIST_LICENSE_BYTES", "url": url, "final_url": final, "size": len(body), "sha256": "sha256:" + digest, "license_files": files}


def audit(project: Path, expected_head: str) -> dict[str, Any]:
    expected_head = validate_expected_head(expected_head)
    project_data, rows, project_bytes, lock_bytes = validate_project(project)
    active = active_linux(rows)
    installed = installed_distributions()
    expected_ids = [identity(row["name"], row["version"]) for row in active]
    observed_ids = sorted(installed)
    failures: list[str] = []
    if sys.platform != "linux" or platform.machine().casefold() not in {"x86_64", "amd64"}:
        failures.append("audit host must be Linux x86_64")
    if sys.version_info[:2] != (3, 12):
        failures.append(f"audit Python must be 3.12, got {platform.python_version()}")
    if expected_ids != observed_ids:
        failures.append(f"installed closure differs from Linux lock: missing={sorted(set(expected_ids)-set(observed_ids))!r} extra={sorted(set(observed_ids)-set(expected_ids))!r}")
    package_reports = []
    for row in active:
        key = identity(row["name"], row["version"])
        candidates = installed.get(key, [])
        if len(candidates) != 1:
            package_reports.append({"lock": row, "installed": None, "status": "BLOCKED_INSTALLED_IDENTITY"})
            failures.append(f"installed distribution count is not one: {key}")
            continue
        dist = candidates[0]
        publisher, unsafe = publisher_files(dist)
        native, native_errors = native_payloads(dist)
        metadata_fields = {"name": dist.metadata.get("Name"), "version": dist.version, "license": dist.metadata.get("License"), "license_expression": dist.metadata.get("License-Expression"), "license_classifiers": sorted(value.removeprefix("License :: ") for value in (dist.metadata.get_all("Classifier") or []) if value.startswith("License :: "))}
        source = None if publisher else locked_sdist_license(row)
        if not publisher and not (source and source.get("license_files")):
            failures.append(f"primary publisher/source license bytes unavailable: {key}")
        if unsafe:
            failures.append(f"unsafe/oversized publisher license path: {key}")
        if native_errors:
            failures.extend(f"native payload inspection failed: {key}: {item}" for item in native_errors)
        for item in native:
            if item.get("inspection") != "ok":
                failures.append(f"native payload inspection incomplete: {key}: {item.get('path', '<unknown>')}")
            if contains_forbidden(item.get("path", "")) or any(contains_forbidden(value) for value in item.get("needed", [])):
                failures.append(f"forbidden CUDA/NVIDIA/Triton native payload: {key}: {item.get('path', '<unknown>')}")
        package_reports.append({"lock": row, "installed": {**metadata_fields, "publisher_license_files": publisher, "unsafe_license_paths": unsafe, "locked_sdist_license": source, "native_payloads": native, "native_payload_errors": native_errors}})
    status = "BLOCKED_FACTUAL_AUDIT" if failures else "BLOCKED_OWNER_REVIEW"
    return {
        "schema": SCHEMA,
        "status": status,
        "review": "PENDING_OWNER_APPROVAL",
        "project": {"name": project_data["project"]["name"], "version": project_data["project"]["version"], "pyproject_sha256": sha256_bytes(project_bytes), "uv_lock_sha256": sha256_bytes(lock_bytes)},
        "closure": {"lock_rows": len(rows), "active_linux_rows": len(active), "expected": sorted(expected_ids), "observed": observed_ids, "exact": expected_ids == observed_ids},
        "packages": package_reports,
        "failures": sorted(set(failures)),
        "environment": {"python": platform.python_version(), "platform": sys.platform, "machine": platform.machine(), "readelf_required": True, "irodori_source_fetched": False, "irodori_source_imported": False, "model_code_imported": False, "weights_acquired": False, "weights_imported": False, "weights_executed": False, "cargo_invoked": False},
        "git": {"expected_head": expected_head},
        "model_acquisition": {"requested_files": [], "policy": "NO_IRODORI_SOURCE_OR_MODEL_OR_CHECKPOINT_REQUESTS"},
        "publication": "NO_UPLOAD",
    }


def _validate_license_entries(value: Any, label: str) -> None:
    if not isinstance(value, list):
        raise AuditError(f"{label} must be a list")
    for item in value:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "size", "sha256"}
            or not isinstance(item["path"], str)
            or not item["path"]
            or not isinstance(item["size"], int)
            or isinstance(item["size"], bool)
            or item["size"] <= 0
            or item["size"] > MAX_LICENSE_BYTES
            or not isinstance(item["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
        ):
            raise AuditError(f"{label} contains malformed evidence")
        safe_relative(item["path"])


def _validate_native_entries(value: Any) -> None:
    if not isinstance(value, list):
        raise AuditError("native payload evidence must be a list")
    for item in value:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "size", "sha256", "needed", "inspection", "error"}
            or not isinstance(item["path"], str)
            or not item["path"]
            or not isinstance(item["size"], int)
            or isinstance(item["size"], bool)
            or item["size"] <= 0
            or not isinstance(item["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
            or not isinstance(item["needed"], list)
            or any(not isinstance(value, str) for value in item["needed"])
            or item["inspection"] not in {"ok", "error"}
            or (item["inspection"] == "ok" and item["error"] is not None)
            or (item["inspection"] == "error" and (not isinstance(item["error"], str) or not item["error"]))
        ):
            raise AuditError("native payload evidence is malformed or incomplete")
        safe_relative(item["path"])


def _validate_sdist_evidence(value: Any, row: dict[str, Any]) -> None:
    if not isinstance(value, dict) or not isinstance(value.get("status"), str):
        raise AuditError("dependency report locked sdist evidence is malformed")
    status = value["status"]
    if status == "ACQUIRED_LOCKED_SDIST_LICENSE_BYTES":
        artifact = row.get("sdist")
        if (
            not isinstance(artifact, dict)
            or set(value) != {"status", "url", "final_url", "size", "sha256", "license_files"}
            or value["url"] != artifact["url"]
            or value["final_url"] != artifact["url"]
            or value["size"] != artifact["size"]
            or value["sha256"] != artifact["hash"]
        ):
            raise AuditError("dependency report locked sdist identity drift")
        _validate_license_entries(value["license_files"], "locked sdist license files")
        if not value["license_files"]:
            raise AuditError("dependency report locked sdist has no license files")
        return
    if status == "NOT_APPLICABLE_NON_PYPI_SOURCE":
        if row.get("source") != {"registry": TORCH_CPU_INDEX} or set(value) != {"status", "license_files"} or value["license_files"] != []:
            raise AuditError("dependency report non-PyPI sdist disposition is malformed")
        return
    if row.get("source") != {"registry": PYPI_REGISTRY} or not isinstance(row.get("sdist"), dict):
        raise AuditError("dependency report blocked sdist lacks a PyPI lock artifact")
    if status == "BLOCKED_LOCKED_SDIST_FETCH":
        expected = {"status", "error", "license_files"}
    elif status == "BLOCKED_LOCKED_SDIST_IDENTITY":
        expected = {"status", "observed_size", "observed_sha256", "license_files"}
    elif status == "BLOCKED_LOCKED_SDIST_LICENSE":
        expected = {"status", "error", "license_files"}
    else:
        raise AuditError("dependency report locked sdist disposition is unknown")
    if set(value) != expected or value["license_files"] != []:
        raise AuditError("dependency report blocked sdist evidence is malformed")
    if "error" in value and not isinstance(value["error"], str):
        raise AuditError("dependency report sdist error is malformed")
    if "observed_size" in value and (not isinstance(value["observed_size"], int) or value["observed_size"] < 0):
        raise AuditError("dependency report observed sdist size is malformed")
    if "observed_sha256" in value and (not isinstance(value["observed_sha256"], str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value["observed_sha256"]) is None):
        raise AuditError("dependency report observed sdist hash is malformed")


def validate_report(path: Path, project: Path, expected_head: str) -> dict[str, Any]:
    """Validate the complete factual report before a worker accepts it.

    The report is evidence, not an approval.  Exact keys and lock identities
    prevent a partial or hand-edited report from being mistaken for a clean
    audit.  This function never imports torch, resolves packages, or touches
    an Irodori checkout or model.
    """

    expected_head = validate_expected_head(expected_head)
    if not path.is_file() or path.is_symlink():
        raise AuditError("dependency report must be a regular file")
    value = strict_json(path.read_bytes())
    project_data, rows, project_bytes, lock_bytes = validate_project(project)
    active = active_linux(rows)
    expected_ids = [identity(row["name"], row["version"]) for row in active]
    if not isinstance(value, dict) or set(value) != REPORT_KEYS:
        raise AuditError("dependency report root schema drift")
    if value["schema"] != SCHEMA or value["status"] not in {"BLOCKED_FACTUAL_AUDIT", "BLOCKED_OWNER_REVIEW"}:
        raise AuditError("dependency report disposition drift")
    if value["review"] != "PENDING_OWNER_APPROVAL" or value["publication"] != "NO_UPLOAD":
        raise AuditError("dependency report approval/publication gate drift")
    project_value = value["project"]
    expected_project = {
        "name": project_data["project"]["name"],
        "version": project_data["project"]["version"],
        "pyproject_sha256": sha256_bytes(project_bytes),
        "uv_lock_sha256": sha256_bytes(lock_bytes),
    }
    if project_value != expected_project:
        raise AuditError("dependency report project identity/hash drift")
    closure = value["closure"]
    expected_observed = closure.get("observed") if isinstance(closure, dict) else None
    expected_expected = closure.get("expected") if isinstance(closure, dict) else None
    if (
        not isinstance(closure, dict)
        or set(closure) != {"lock_rows", "active_linux_rows", "expected", "observed", "exact"}
        or closure["lock_rows"] != len(rows)
        or closure["active_linux_rows"] != len(active)
        or _canonical_string_list(expected_expected, "dependency report expected closure") != expected_ids
        or _canonical_string_list(expected_observed, "dependency report observed closure") is None
        or not isinstance(closure["exact"], bool)
        or closure["exact"] != (expected_observed == expected_ids)
    ):
        raise AuditError("dependency report closure drift")
    failures = _canonical_string_list(value["failures"], "dependency report failures")
    environment = value["environment"]
    if not isinstance(environment, dict) or set(environment) != ENVIRONMENT_KEYS:
        raise AuditError("dependency report environment schema drift")
    if not isinstance(environment["platform"], str) or not isinstance(environment["machine"], str):
        raise AuditError("dependency report host identity is malformed")
    if environment["readelf_required"] is not True or any(
        environment[key] is not False
        for key in (
            "irodori_source_fetched",
            "irodori_source_imported",
            "model_code_imported",
            "weights_acquired",
            "weights_imported",
            "weights_executed",
            "cargo_invoked",
        )
    ):
        raise AuditError("dependency report claims unsafe source/model activity")
    if not isinstance(environment["python"], str):
        raise AuditError("dependency report host identity is malformed")
    if value["git"] != {"expected_head": expected_head}:
        raise AuditError("dependency report HEAD binding drift")
    if value["model_acquisition"] != {
        "requested_files": [],
        "policy": "NO_IRODORI_SOURCE_OR_MODEL_OR_CHECKPOINT_REQUESTS",
    }:
        raise AuditError("dependency report model acquisition drift")
    packages = value["packages"]
    if not isinstance(packages, list) or len(packages) != len(active):
        raise AuditError("dependency report package count drift")
    expected_failures: set[str] = set()
    if environment["platform"] != "linux" or environment["machine"].casefold() not in {"x86_64", "amd64"}:
        expected_failures.add("audit host must be Linux x86_64")
    if re.fullmatch(r"3\.12\.\d+", environment["python"]) is None:
        expected_failures.add(f"audit Python must be 3.12, got {environment['python']}")
    observed = closure["observed"]
    if observed != expected_ids:
        expected_failures.add(
            f"installed closure differs from Linux lock: missing={sorted(set(expected_ids)-set(observed))!r} extra={sorted(set(observed)-set(expected_ids))!r}"
        )
    for item, row in zip(packages, active, strict=True):
        if not isinstance(item, dict) or item.get("lock") != row:
            raise AuditError("dependency report locked package identity drift")
        allowed_item_keys = {PACKAGE_KEYS, PACKAGE_KEYS | {"status"}}
        if set(item) not in allowed_item_keys:
            raise AuditError("dependency report package schema drift")
        installed = item.get("installed")
        if installed is None:
            if set(item) != PACKAGE_KEYS | {"status"} or item.get("status") != "BLOCKED_INSTALLED_IDENTITY":
                raise AuditError("dependency report incomplete package disposition drift")
            expected_failures.add(f"installed distribution count is not one: {identity(row['name'], row['version'])}")
            continue
        if set(item) != PACKAGE_KEYS:
            raise AuditError("dependency report package schema drift")
        if not isinstance(installed, dict) or set(installed) != INSTALLED_KEYS:
            raise AuditError("dependency report installed package schema drift")
        if not isinstance(installed["name"], str) or not isinstance(installed["version"], str):
            raise AuditError("dependency report installed package identity is malformed")
        if identity(str(installed["name"]), str(installed["version"])) != identity(row["name"], row["version"]):
            raise AuditError("dependency report installed package identity drift")
        for key in ("license", "license_expression"):
            if installed[key] is not None and not isinstance(installed[key], str):
                raise AuditError("dependency report license metadata type drift")
        if not isinstance(installed["license_classifiers"], list) or any(not isinstance(item, str) for item in installed["license_classifiers"]):
            raise AuditError("dependency report license classifier metadata drift")
        _validate_license_entries(installed["publisher_license_files"], "publisher license files")
        if not isinstance(installed["unsafe_license_paths"], list) or any(not isinstance(item, str) or not item for item in installed["unsafe_license_paths"]):
            raise AuditError("dependency report unsafe publisher license paths are malformed")
        if installed["unsafe_license_paths"]:
            expected_failures.add(
                f"unsafe/oversized publisher license path: {identity(row['name'], row['version'])}"
            )
        _validate_native_entries(installed["native_payloads"])
        if not isinstance(installed["native_payload_errors"], list) or any(not isinstance(item, str) or not item for item in installed["native_payload_errors"]):
            raise AuditError("dependency report native payload errors are malformed")
        key = identity(row["name"], row["version"])
        for error in installed["native_payload_errors"]:
            expected_failures.add(f"native payload inspection failed: {key}: {error}")
        for native in installed["native_payloads"]:
            if native["inspection"] != "ok":
                expected_failures.add(f"native payload inspection incomplete: {key}: {native['path']}")
            if native["inspection"] == "ok" and native["error"] is not None:
                raise AuditError("dependency report successful native inspection carries an error")
            if contains_forbidden(native["path"]) or any(contains_forbidden(value) for value in native["needed"]):
                expected_failures.add(f"forbidden CUDA/NVIDIA/Triton native payload: {key}: {native['path']}")
        publisher = installed["publisher_license_files"]
        source = installed["locked_sdist_license"]
        if publisher and source is not None:
            raise AuditError("dependency report contains both publisher and sdist license evidence")
        if not publisher and not (isinstance(source, dict) and source.get("license_files")):
            expected_failures.add(f"primary publisher/source license bytes unavailable: {key}")
        if source is not None:
            _validate_sdist_evidence(source, row)
    expected_failures = set(expected_failures)
    actual_failures = set(failures)
    if actual_failures != expected_failures:
        raise AuditError(
            f"dependency report failures do not match evidence: expected={sorted(expected_failures)!r} actual={sorted(actual_failures)!r}"
        )
    expected_status = "BLOCKED_FACTUAL_AUDIT" if expected_failures else "BLOCKED_OWNER_REVIEW"
    if value["status"] != expected_status:
        raise AuditError("dependency report status does not match failures")
    if closure["exact"] is not True and value["status"] != "BLOCKED_FACTUAL_AUDIT":
        raise AuditError("inexact dependency closure cannot remain owner-review")
    return value


def validate_output_path(path: Path) -> None:
    if not path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise AuditError("output must be absolute and contain no dot components")
    if path.exists() or path.is_symlink():
        raise AuditError("output must be absent")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise AuditError("output parent must be an existing real directory")
    current = parent
    while True:
        if current.is_symlink():
            raise AuditError(f"output ancestor is a symlink: {current}")
        if current == current.parent:
            break
        current = current.parent
    try:
        resolved = parent.resolve(strict=True)
    except OSError as exc:
        raise AuditError(f"output parent cannot be resolved: {exc}") from exc
    if resolved != parent:
        raise AuditError("output parent resolves through a symlink")


def write_no_replace(path: Path, payload: bytes) -> None:
    validate_output_path(path)
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def self_test() -> int:
    global installed_distributions, publisher_files, native_payloads
    _, rows, _, _ = validate_project()
    if len(active_linux(rows)) != 10:
        raise SystemExit("self-test expected 10 Linux active distributions")
    if any(contains_forbidden(row["name"]) for row in rows):
        raise SystemExit("self-test found accelerator package")
    body = io.BytesIO()
    with tarfile.open(fileobj=body, mode="w:gz") as archive:
        value = b"MIT License\n"
        info = tarfile.TarInfo("pkg-1.0/LICENSE")
        info.size = len(value)
        archive.addfile(info, io.BytesIO(value))
    files = archive_license_files("https://files.pythonhosted.org/pkg-1.0.tar.gz", body.getvalue())
    if files != [{"path": "pkg-1.0/LICENSE", "size": len(b"MIT License\n"), "sha256": sha256_bytes(b"MIT License\n")}]:
        raise SystemExit("self-test lost exact source license bytes")
    if not contains_forbidden("nvidia-cuda-runtime") or contains_forbidden("safetensors"):
        raise SystemExit("self-test accelerator rejection is broken")
    if any("cuda" in value.casefold() for value in ["libtorch_cpu.so"]):
        raise SystemExit("self-test falsely rejected CPU payload")
    try:
        strict_json('{"status":"BLOCKED","status":"OWNER"}')
    except AuditError:
        pass
    else:
        raise SystemExit("self-test accepted duplicate report key")
    for malformed in (b"{", b"\xff", None):
        try:
            strict_json(malformed)
        except AuditError:
            pass
        else:
            raise SystemExit("self-test leaked malformed JSON/UTF-8/type error")
    sdist_row = {"source": {"registry": PYPI_REGISTRY}, "sdist": {"url": "https://files.pythonhosted.org/pkg-1.0.tar.gz", "hash": "sha256:" + "0" * 64, "size": 1}}
    _validate_sdist_evidence({"status": "BLOCKED_LOCKED_SDIST_FETCH", "error": "URLError", "license_files": []}, sdist_row)
    _validate_sdist_evidence({"status": "BLOCKED_LOCKED_SDIST_IDENTITY", "observed_size": 1, "observed_sha256": "sha256:" + "1" * 64, "license_files": []}, sdist_row)
    _validate_sdist_evidence({"status": "BLOCKED_LOCKED_SDIST_LICENSE", "error": "missing license", "license_files": []}, sdist_row)
    try:
        _validate_sdist_evidence({"status": "BLOCKED_LOCKED_SDIST_FETCH", "error": "URLError", "license_files": [{"path": "LICENSE", "size": 1, "sha256": "0" * 64}]}, sdist_row)
    except AuditError:
        pass
    else:
        raise SystemExit("self-test accepted blocked sdist with license bytes")
    # Exercise report-level semantics with a synthetic package-only report;
    # no torch import or dependency synchronization is involved.
    saved_installed, saved_publisher, saved_native = installed_distributions, publisher_files, native_payloads
    class _Meta(dict):
        def get_all(self, key):
            return []
    class _Dist:
        def __init__(self, name, version):
            self.version = version
            self.metadata = _Meta({"Name": name, "License": "BSD", "License-Expression": "BSD-3-Clause"})
    active = active_linux(rows)
    installed_distributions = lambda: {identity(row["name"], row["version"]): [_Dist(row["name"], row["version"])] for row in active}
    publisher_files = lambda dist: ([{"path": "LICENSE", "size": 3, "sha256": "0" * 64}], [])
    native_payloads = lambda dist: ([], [])
    head = "0123456789abcdef" * 2 + "01234567"
    try:
        with tempfile.TemporaryDirectory(prefix="irodori-report-validator-", dir="/private/tmp") as directory:
            report_path = Path(directory) / "report.json"
            report = audit(REFERENCE_PROJECT, head)
            write_no_replace(report_path, (canonical(report) + "\n").encode())
            validate_report(report_path, REFERENCE_PROJECT, head)

            def expect_rejected(mutated: dict[str, Any], label: str) -> None:
                candidate = Path(directory) / f"{label}.json"
                write_no_replace(candidate, (canonical(mutated) + "\n").encode())
                try:
                    validate_report(candidate, REFERENCE_PROJECT, head)
                except AuditError:
                    return
                raise SystemExit(f"self-test accepted invalid report: {label}")

            def expect_payload_rejected(payload: bytes, label: str) -> None:
                candidate = Path(directory) / f"{label}.json"
                candidate.write_bytes(payload)
                try:
                    validate_report(candidate, REFERENCE_PROJECT, head)
                except AuditError:
                    return
                raise SystemExit(f"self-test accepted invalid report payload: {label}")

            expect_payload_rejected(b"{", "malformed-json")
            expect_payload_rejected(b"\xff", "malformed-utf8")
            expect_payload_rejected(b"[]", "malformed-root-type")

            bad_status = json.loads(canonical(report))
            bad_status["status"] = "BLOCKED_OWNER_REVIEW"
            expect_rejected(bad_status, "bad-status")
            bad_exact = json.loads(canonical(report))
            bad_exact["closure"]["exact"] = False
            expect_rejected(bad_exact, "bad-exact")
            for observed, label in ((None, "null"), (["z", "a"], "unsorted"), (["a", "a"], "duplicate")):
                bad_observed = json.loads(canonical(report))
                bad_observed["closure"]["observed"] = observed
                expect_rejected(bad_observed, f"bad-observed-{label}")
            bad_failures = json.loads(canonical(report))
            bad_failures["failures"] = ["synthetic failure", "synthetic failure"]
            expect_rejected(bad_failures, "duplicate-failures")
            missing_failure = json.loads(canonical(report))
            missing_failure["packages"][0]["installed"]["publisher_license_files"] = []
            missing_failure["packages"][0]["installed"]["locked_sdist_license"] = None
            expect_rejected(missing_failure, "missing-critical-failure")
            honest_incomplete = json.loads(canonical(report))
            first = honest_incomplete["packages"][0]
            first["installed"] = None
            first["status"] = "BLOCKED_INSTALLED_IDENTITY"
            key = identity(active[0]["name"], active[0]["version"])
            honest_incomplete["failures"] = sorted(set(honest_incomplete["failures"]) | {f"installed distribution count is not one: {key}"})
            honest_incomplete["status"] = "BLOCKED_FACTUAL_AUDIT"
            candidate = Path(directory) / "honest-incomplete.json"
            write_no_replace(candidate, (canonical(honest_incomplete) + "\n").encode())
            validate_report(candidate, REFERENCE_PROJECT, head)
            malformed_incomplete = json.loads(canonical(honest_incomplete))
            del malformed_incomplete["packages"][0]["status"]
            expect_rejected(malformed_incomplete, "malformed-incomplete")
            missing_package_key = json.loads(canonical(report))
            del missing_package_key["packages"][0]["installed"]
            expect_rejected(missing_package_key, "missing-package-key")
    finally:
        installed_distributions, publisher_files, native_payloads = saved_installed, saved_publisher, saved_native
    with tempfile.TemporaryDirectory(prefix="irodori-dependency-hash-test-", dir="/private/tmp") as directory:
        native = Path(directory) / "large-native.so"
        native.write_bytes((b"native-payload-" * 200_000) + b"\n")
        if sha256_file(native) != sha256_bytes(native.read_bytes()):
            raise SystemExit("self-test streaming native hash mismatch")
    with tempfile.TemporaryDirectory(prefix="irodori-dependency-audit-test-", dir="/private/tmp") as directory:
        output = Path(directory) / "report.json"
        write_no_replace(output, b"owner")
        try:
            write_no_replace(output, b"clobber")
        except AuditError:
            pass
        else:
            raise SystemExit("self-test allowed report clobber")
        for bad in (Path("relative.json"), Path(directory) / "missing" / "report.json"):
            try:
                validate_output_path(bad)
            except AuditError:
                pass
            else:
                raise SystemExit(f"self-test accepted unsafe output path: {bad}")
        symlink_parent = Path(directory) / "link"
        symlink_parent.symlink_to(Path(directory), target_is_directory=True)
        try:
            validate_output_path(symlink_parent / "report.json")
        except AuditError:
            pass
        else:
            raise SystemExit("self-test accepted symlink output ancestor")
    for bad_head in (None, "A" * 40, "0" * 39, "0" * 41, "0" * 39 + "G"):
        try:
            validate_expected_head(bad_head)
        except AuditError:
            pass
        else:
            raise SystemExit(f"self-test accepted invalid/missing expected HEAD: {bad_head!r}")
    if validate_expected_head("0123456789abcdef" * 2 + "01234567") != "0123456789abcdef" * 2 + "01234567":
        raise SystemExit("self-test rejected valid expected HEAD")
    for bad in ("../LICENSE", "/LICENSE", "pkg\\LICENSE", "pkg/./LICENSE"):
        try:
            safe_relative(bad)
        except AuditError:
            pass
        else:
            raise SystemExit(f"self-test accepted unsafe archive path: {bad}")
    print("irodori_text_block dependency_audit self-test: OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate-contract", action="store_true")
    parser.add_argument("--validate-output", action="store_true")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-head")
    args = parser.parse_args()
    if args.self_test:
        if args.validate_contract or args.validate_output or args.project or args.output or args.expected_head:
            parser.error("--self-test accepts no other arguments")
        return self_test()
    if args.validate_contract:
        if args.validate_output or args.project or args.output or args.expected_head:
            parser.error("--validate-contract accepts no other arguments")
        validate_project()
        print("irodori_text_block dependency contract: OK")
        return 0
    if args.validate_output:
        if args.project is None or args.output is None or args.expected_head is None:
            parser.error("--validate-output requires --project, --output, and --expected-head")
        try:
            validate_report(args.output, args.project, args.expected_head)
        except (AuditError, OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
            print(f"Irodori dependency report rejected: {exc}", file=sys.stderr)
            return 2
        print("irodori_text_block dependency report: OK")
        return 0
    if args.project is None or args.output is None or args.expected_head is None:
        parser.error("--project, --output, and --expected-head are required")
    try:
        report = audit(args.project, validate_expected_head(args.expected_head))
        write_no_replace(args.output, (canonical(report) + "\n").encode())
    except (AuditError, OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        report = {"schema": SCHEMA, "status": "BLOCKED_FACTUAL_AUDIT", "review": "PENDING_OWNER_APPROVAL", "failures": [str(exc)], "model_acquisition": {"requested_files": []}, "publication": "NO_UPLOAD"}
        try:
            write_no_replace(args.output, (canonical(report) + "\n").encode())
        except (AuditError, OSError) as write_error:
            print(f"Irodori dependency audit blocked: {write_error}", file=sys.stderr)
            return 2
    print(f"Irodori dependency audit: {report['status']} ({args.output})", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
