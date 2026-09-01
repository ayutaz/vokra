#!/usr/bin/env -S uv run --no-project --python 3.12 python
"""Collect factual evidence for the locked LiteRT reference closure.

The collector is intentionally stdlib-only.  It never imports a third-party
distribution, executes a model, downloads an artifact, or decides whether a
license is acceptable.  The VAST worker runs it after a frozen sync and writes
an evidence packet whose status remains owner-review-required.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import tomllib
from email.parser import BytesParser
from email.policy import compat32
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

CHUNK_SIZE = 1 << 20
MAX_METADATA_BYTES = 4 << 20
MAX_LICENSE_BYTES = 4 << 20
MAX_RECORD_BYTES = 16 << 20
MAX_NATIVE_BYTES = 2 << 30
MAX_RECORD_ENTRIES = 1_000_000

EXPECTED_PROJECT_SHA256 = "2b114885d54470c8397528b37572e3632202ca0b9d65ac349ec7e7da4e331f03"
EXPECTED_LOCK_SHA256 = "da75839f6195c27c32a15f097a40450c18b317ad78e9036ec2a1618472b85555"
# Keep this list in lock-step with the independent reference project's reviewed
# lock.  The project itself is virtual (package=false), so it is not expected
# to appear in site-packages.
EXPECTED_VERSIONS = {
    "ai-edge-litert": "2.2.0",
    "backports-strenum": "1.3.1",
    "flatbuffers": "25.12.19",
    "ml-dtypes": "0.6.0",
    "numpy": "2.5.2",
    "protobuf": "7.36.1",
    "tqdm": "4.70.0",
    "typing-extensions": "4.16.0",
    "vokra-microwakeword-reference": "0.1.0",
}
PROJECT_NAME = "vokra-microwakeword-reference"
PLATFORM_MARKER = "platform_machine == 'x86_64' and sys_platform == 'linux'"
LICENSE_BASENAMES = ("license", "licence", "copying", "notice", "copyright")
NATIVE_SUFFIXES = (".dylib", ".dll", ".pyd")


def normalize_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, limit: int | None = None) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        while True:
            block = stream.read(CHUNK_SIZE)
            if not block:
                break
            total += len(block)
            if limit is not None and total > limit:
                raise ValueError(f"oversize file: {path}")
            digest.update(block)
    return digest.hexdigest(), total


def _decode_record_sha256(value: str) -> bytes:
    """Decode RECORD's URL-safe, optionally unpadded base64 digest."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+={0,2}", value):
        raise ValueError("invalid base64 characters or padding")
    unpadded = value.rstrip("=")
    if len(unpadded) % 4 == 1:
        raise ValueError("invalid base64 length")
    padded = unpadded + "=" * ((-len(unpadded)) % 4)
    return base64.b64decode(padded, altchars=b"-_", validate=True)


def _safe_record_parts(value: str) -> list[str]:
    if not value or "\\" in value or value.startswith("/"):
        raise ValueError(f"unsafe RECORD path: {value!r}")
    parts = value.split("/")
    if any(part == "" for part in parts):
        raise ValueError(f"unsafe RECORD path: {value!r}")
    return parts


def _is_license_candidate(path: PurePosixPath) -> bool:
    basename = path.name.casefold()
    return any(
        basename == prefix
        or basename.startswith(prefix + ".")
        or basename.startswith(prefix + "-")
        or basename.startswith(prefix + "_")
        for prefix in LICENSE_BASENAMES
    )


def _is_native_payload(path: PurePosixPath) -> bool:
    basename = path.name.casefold()
    return (
        ".so" in basename
        and (basename.endswith(".so") or ".so." in basename)
    ) or basename.endswith(NATIVE_SUFFIXES)


def _canonical_under(environment_root: Path, site_packages: Path, value: str) -> tuple[Path, str]:
    """Resolve a RECORD path within the venv, rejecting symlink/escape."""
    try:
        site_relative = site_packages.relative_to(environment_root)
    except ValueError as error:
        raise ValueError("site-packages is outside environment root") from error
    stack: list[str] = []
    for part in (*site_relative.parts, *_safe_record_parts(value)):
        if part == ".":
            continue
        if part == "..":
            if not stack:
                raise ValueError(f"RECORD path escapes environment root: {value!r}")
            stack.pop()
        else:
            stack.append(part)
            current = environment_root.joinpath(*stack)
            try:
                info = current.lstat()
            except FileNotFoundError as error:
                raise ValueError(f"missing RECORD path component: {value!r}") from error
            if stat.S_ISLNK(info.st_mode):
                raise ValueError(f"symlink in installed distribution: {value!r}")
    current = environment_root.joinpath(*stack)
    try:
        info = current.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"missing RECORD file: {value!r}") from error
    if stat.S_ISLNK(info.st_mode):
        raise ValueError(f"symlink in installed distribution: {value!r}")
    if not current.is_file():
        raise ValueError(f"RECORD path is not a regular file: {value!r}")
    return current, "/".join(stack)


def _bounded_bytes(path: Path, limit: int, label: str) -> tuple[bytes, str, int]:
    digest = hashlib.sha256()
    total = 0
    chunks: list[bytes] = []
    with path.open("rb") as stream:
        while True:
            block = stream.read(CHUNK_SIZE)
            if not block:
                break
            total += len(block)
            if total > limit:
                raise ValueError(f"oversize {label}: {path}")
            digest.update(block)
            chunks.append(block)
    return b"".join(chunks), digest.hexdigest(), total


def _metadata_evidence(path: Path) -> dict[str, Any]:
    data, digest, size = _bounded_bytes(path, MAX_METADATA_BYTES, "METADATA")
    message = BytesParser(policy=compat32).parsebytes(data)
    classifiers = [value.strip() for value in message.get_all("Classifier", []) if value.strip()]
    return {
        "path": path.name,
        "bytes": size,
        "sha256": digest,
        "name": [value.strip() for value in message.get_all("Name", [])],
        "version": [value.strip() for value in message.get_all("Version", [])],
        "license": [value.strip() for value in message.get_all("License", [])],
        "license_expression": [value.strip() for value in message.get_all("License-Expression", [])],
        "license_file": [value.strip() for value in message.get_all("License-File", [])],
        "classifiers": classifiers,
        "license_classifiers": [value for value in classifiers if value.casefold().startswith("license ::")],
    }


def _record_evidence(
    path: Path, environment_root: Path, site_packages: Path
) -> tuple[dict[str, Any], list[tuple[str, Path]], list[str]]:
    data, digest, size = _bounded_bytes(path, MAX_RECORD_BYTES, "RECORD")
    entries: list[dict[str, Any]] = []
    owned: list[tuple[str, Path]] = []
    failures: list[str] = []
    seen_paths: set[str] = set()
    try:
        decoded = data.decode("utf-8")
    except UnicodeDecodeError as error:
        return (
            {"path": path.name, "bytes": size, "sha256": digest, "entries": []},
            [],
            [f"RECORD is not UTF-8: {error}"],
        )
    try:
        reader = csv.reader(decoded.splitlines())
        for row_number, row in enumerate(reader, start=1):
            entry: dict[str, Any] = {
                "row": row_number,
                "declared": {"path": row[0] if row else None, "hash": None, "size": None},
                "actual": None,
                "validation": "MALFORMED_ROW",
                "errors": [],
            }
            if len(row) != 3:
                entry["declared"]["fields"] = [str(item) for item in row]
                entry["errors"].append(f"RECORD row has {len(row)} fields")
                entries.append(entry)
                failures.extend(entry["errors"])
                continue
            declared_path, declared_hash, declared_size = (str(item) for item in row)
            entry["declared"] = {"path": declared_path, "hash": None, "size": None}
            if len(entries) >= MAX_RECORD_ENTRIES:
                entry["errors"].append("RECORD has too many entries")
                entry["validation"] = "OVERSIZE"
                entries.append(entry)
                failures.extend(entry["errors"])
                break
            if declared_path in seen_paths:
                entry["errors"].append(f"duplicate RECORD path: {declared_path}")
            seen_paths.add(declared_path)

            if declared_hash == "":
                entry["declared"]["hash"] = {"algorithm": None, "value": None, "status": "EMPTY"}
            elif "=" not in declared_hash:
                entry["declared"]["hash"] = {"algorithm": None, "value": declared_hash, "status": "MALFORMED"}
                entry["errors"].append(f"malformed RECORD hash: {declared_path}")
            else:
                algorithm, value = declared_hash.split("=", 1)
                hash_evidence = {"algorithm": algorithm, "value": value, "status": "DECLARED"}
                if algorithm != "sha256":
                    hash_evidence["status"] = "UNSUPPORTED"
                    entry["errors"].append(f"unsupported RECORD hash algorithm: {declared_path}")
                else:
                    try:
                        decoded_hash = _decode_record_sha256(value)
                    except (ValueError, binascii.Error):
                        hash_evidence["status"] = "MALFORMED"
                        entry["errors"].append(f"malformed sha256 RECORD hash: {declared_path}")
                    else:
                        if len(decoded_hash) != hashlib.sha256().digest_size:
                            hash_evidence["status"] = "MALFORMED"
                            entry["errors"].append(f"invalid sha256 RECORD hash length: {declared_path}")
                        else:
                            hash_evidence["value"] = decoded_hash.hex()
                            hash_evidence["status"] = "VALID"
                entry["declared"]["hash"] = hash_evidence

            if declared_size == "":
                entry["declared"]["size"] = {"value": None, "status": "EMPTY"}
            elif re.fullmatch(r"(?:0|[1-9][0-9]*)", declared_size):
                entry["declared"]["size"] = {"value": int(declared_size), "status": "VALID"}
            else:
                entry["declared"]["size"] = {"value": declared_size, "status": "MALFORMED"}
                entry["errors"].append(f"malformed RECORD size: {declared_path}")

            try:
                target, environment_relative = _canonical_under(environment_root, site_packages, declared_path)
            except ValueError as error:
                entry["errors"].append(str(error))
            else:
                actual_sha256, actual_size = sha256_file(target, MAX_NATIVE_BYTES)
                entry["resolved_path"] = environment_relative
                entry["actual"] = {"sha256": actual_sha256, "bytes": actual_size}
                declared_hash_data = entry["declared"]["hash"]
                if declared_hash_data["status"] == "VALID" and declared_hash_data["value"] != actual_sha256:
                    entry["errors"].append(f"RECORD hash mismatch: {declared_path}")
                declared_size_data = entry["declared"]["size"]
                if declared_size_data["status"] == "VALID" and declared_size_data["value"] != actual_size:
                    entry["errors"].append(f"RECORD size mismatch: {declared_path}")
                owned.append((declared_path, target))
            if entry["errors"]:
                entry["validation"] = "FAIL"
                failures.extend(entry["errors"])
            elif entry["declared"]["hash"]["status"] == "EMPTY" or entry["declared"]["size"]["status"] == "EMPTY":
                entry["validation"] = "EMPTY_DECLARATION"
            else:
                entry["validation"] = "MATCH"
            entries.append(entry)
    except csv.Error as error:
        failures.append(f"invalid RECORD CSV: {error}")
    canonical_entries = json.dumps(entries, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return (
        {
            "path": path.name,
            "bytes": size,
            "sha256": digest,
            "entries": entries,
            "entries_count": len(entries),
            "entries_sha256": sha256_bytes(canonical_entries),
        },
        owned,
        failures,
    )


def _readelf_needed(path: Path) -> dict[str, Any]:
    readelf = shutil.which("readelf")
    if readelf is None:
        return {"status": "UNAVAILABLE"}
    try:
        result = subprocess.run(
            [readelf, "-d", "--", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "ERROR", "error": str(error)[:256]}
    if result.returncode != 0:
        return {"status": "NOT_ELF_OR_UNREADABLE", "returncode": result.returncode}
    needed = sorted(set(re.findall(r"Shared library: \[([^]]+)\]", result.stdout)))
    return {"status": "COLLECTED", "needed": needed}


def _environment_relative(environment_root: Path, path: Path) -> str:
    return "/".join(path.relative_to(environment_root).parts)


def _installed_inventory(
    environment_root: Path, site_packages: Path, expected_names: set[str] | None = None
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str], str]:
    """Enumerate every dist-info once and bind later collection to that view."""
    expected_names = expected_names or {name for name in EXPECTED_VERSIONS if name != PROJECT_NAME}
    expected_normalized = {normalize_name(name) for name in expected_names}
    entries: list[dict[str, Any]] = []
    failures: list[str] = []
    try:
        walked = os.walk(site_packages, topdown=True, followlinks=False)
        for directory, dirnames, filenames in walked:
            directory_path = Path(directory)
            for name in sorted(list(dirnames), key=str.casefold):
                candidate = directory_path / name
                if candidate.is_symlink():
                    failures.append(f"symlink in installed site-packages: {_environment_relative(environment_root, candidate)}")
                    dirnames.remove(name)
            for name in sorted(filenames, key=str.casefold):
                candidate = directory_path / name
                if candidate.is_symlink():
                    failures.append(f"symlink in installed site-packages: {_environment_relative(environment_root, candidate)}")
            for name in sorted(list(dirnames), key=str.casefold):
                dist_info = directory_path / name
                if not name.casefold().endswith(".dist-info"):
                    continue
                entry: dict[str, Any] = {
                    "path": _environment_relative(environment_root, dist_info),
                    "metadata": None,
                    "normalized_name": None,
                    "name": None,
                    "version": None,
                    "status": "MALFORMED",
                    "failures": [],
                }
                metadata_path = dist_info / "METADATA"
                if not metadata_path.is_file() or metadata_path.is_symlink():
                    entry["failures"].append("missing or symlinked METADATA")
                else:
                    try:
                        metadata = _metadata_evidence(metadata_path)
                    except (OSError, ValueError) as error:
                        entry["failures"].append(f"METADATA failure: {error}")
                    else:
                        metadata["path"] = _environment_relative(environment_root, metadata_path)
                        entry["metadata"] = metadata
                        names = metadata["name"]
                        versions = metadata["version"]
                        if len(names) == 1:
                            entry["name"] = names[0]
                            entry["normalized_name"] = normalize_name(names[0])
                        else:
                            entry["failures"].append("METADATA Name must occur exactly once")
                        if len(versions) == 1:
                            entry["version"] = versions[0]
                        else:
                            entry["failures"].append("METADATA Version must occur exactly once")
                if entry["normalized_name"] not in expected_normalized:
                    entry["failures"].append("unknown installed distribution")
                elif entry["version"] != EXPECTED_VERSIONS.get(entry["normalized_name"], ""):
                    entry["failures"].append("installed distribution version drift")
                if entry["failures"]:
                    failures.extend(f"{entry['path']}: {failure}" for failure in entry["failures"])
                else:
                    entry["status"] = "VALID"
                entries.append(entry)
    except OSError as error:
        failures.append(f"cannot enumerate site-packages: {error}")
    entries.sort(key=lambda item: item["path"].casefold())
    by_name: dict[str, dict[str, Any]] = {}
    duplicate_names: set[str] = set()
    for entry in entries:
        name = entry["normalized_name"]
        if name is None or entry["status"] != "VALID":
            continue
        if name in by_name or name in duplicate_names:
            failures.append(f"duplicate installed distribution name: {name}")
            by_name.pop(name, None)
            duplicate_names.add(name)
        else:
            by_name[name] = entry
    for name in sorted(expected_normalized - set(by_name)):
        failures.append(f"missing installed distribution: {name}")
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return entries, by_name, failures, sha256_bytes(canonical)


def collect_distribution(
    environment_root: Path,
    site_packages: Path,
    expected_name: str,
    expected_version: str,
    inventory_entry: dict[str, Any] | None,
    inventory_sha256: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "expected_name": expected_name,
        "expected_version": expected_version,
        "status": "EVIDENCE_COLLECTED_OWNER_REVIEW_REQUIRED",
        "metadata": None,
        "record": None,
        "license_candidates": [],
        "native_payloads": [],
        "failures": [],
        "inventory_sha256": inventory_sha256,
    }
    if inventory_entry is None:
        result["failures"].append("missing installed distribution")
        result["status"] = "COLLECTION_FAILED_FAIL_CLOSED"
        return result
    result["dist_info_path"] = inventory_entry["path"]
    dist_info = environment_root / Path(inventory_entry["path"])
    if inventory_entry["status"] != "VALID":
        result["failures"].extend(inventory_entry["failures"])
        result["status"] = "COLLECTION_FAILED_FAIL_CLOSED"
        return result
    metadata_path = dist_info / "METADATA"
    metadata = inventory_entry["metadata"]
    result["metadata"] = metadata
    result["dist_info"] = dist_info.name
    names = metadata["name"]
    versions = metadata["version"]
    if len(names) != 1 or normalize_name(names[0]) != normalize_name(expected_name):
        result["failures"].append("installed metadata name mismatch")
    if len(versions) != 1 or versions[0] != expected_version:
        result["failures"].append("installed metadata version mismatch")
    record_path = dist_info / "RECORD"
    if not record_path.is_file() or record_path.is_symlink():
        result["failures"].append("missing installed RECORD")
        result["status"] = "COLLECTION_FAILED_FAIL_CLOSED"
        return result
    try:
        record, owned, record_failures = _record_evidence(record_path, environment_root, site_packages)
    except (OSError, ValueError) as error:
        result["failures"].append(f"RECORD failure: {error}")
        result["status"] = "COLLECTION_FAILED_FAIL_CLOSED"
        return result
    result["record"] = record
    result["record"]["path"] = f"{result['dist_info_path']}/RECORD"
    result["failures"].extend(record_failures)
    if not owned:
        result["failures"].append("RECORD owns no files")
    for relative_name, path in sorted(owned, key=lambda item: item[0].casefold()):
        relative = PurePosixPath(relative_name)
        if _is_license_candidate(relative):
            try:
                digest, size = sha256_file(path, MAX_LICENSE_BYTES)
            except (OSError, ValueError) as error:
                result["failures"].append(str(error))
            else:
                result["license_candidates"].append({"path": relative_name, "bytes": size, "sha256": digest})
        if _is_native_payload(relative):
            try:
                digest, size = sha256_file(path, MAX_NATIVE_BYTES)
            except (OSError, ValueError) as error:
                result["failures"].append(str(error))
            else:
                result["native_payloads"].append(
                    {"path": relative_name, "bytes": size, "sha256": digest, "readelf": _readelf_needed(path)}
                )
    if not result["license_candidates"]:
        result["failures"].append("no bounded license candidate")
    if result["failures"]:
        result["status"] = "COLLECTION_FAILED_FAIL_CLOSED"
    return result


def _project_dependencies(project: dict[str, Any]) -> dict[str, str]:
    dependencies = project.get("project", {}).get("dependencies", [])
    parsed: dict[str, str] = {}
    for item in dependencies:
        if not isinstance(item, str) or "==" not in item:
            raise ValueError(f"unsupported project dependency declaration: {item!r}")
        name, version = item.split("==", 1)
        parsed[normalize_name(name)] = version
    return parsed


def _lock_rows(lock: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    rows = lock.get("package", [])
    failures: list[str] = []
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            failures.append("malformed lock package row")
            continue
        name = normalize_name(row["name"])
        if name in by_name:
            failures.append(f"duplicate lock package row: {name}")
        by_name[name] = row
    expected = {normalize_name(name): version for name, version in EXPECTED_VERSIONS.items()}
    if set(by_name) != set(expected):
        failures.append(f"unknown/missing lock packages: {sorted(set(by_name) ^ set(expected))}")
    for name, version in expected.items():
        if by_name.get(name, {}).get("version") != version:
            failures.append(f"lock version drift: {name}")
    return by_name, failures


def _selected_closure(rows: dict[str, dict[str, Any]], project_name: str) -> tuple[list[str], list[str]]:
    selected: set[str] = set()
    failures: list[str] = []
    pending = [normalize_name(project_name)]
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        selected.add(name)
        row = rows.get(name)
        if row is None:
            failures.append(f"selected closure references unknown package: {name}")
            continue
        for dependency in row.get("dependencies", []):
            if not isinstance(dependency, dict) or not isinstance(dependency.get("name"), str):
                failures.append(f"malformed dependency edge in {name}")
                continue
            marker = dependency.get("marker")
            if marker is not None and marker != PLATFORM_MARKER:
                continue
            pending.append(normalize_name(dependency["name"]))
    return sorted(selected), failures


def _lock_evidence(lock: dict[str, Any], project: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    rows, failures = _lock_rows(lock)
    selected, closure_failures = _selected_closure(rows, PROJECT_NAME)
    failures.extend(closure_failures)
    direct = _project_dependencies(project)
    if direct != {"ai-edge-litert": "2.2.0", "numpy": "2.5.2"}:
        failures.append("direct dependency set/version drift")
    if project.get("project", {}).get("requires-python") != "==3.12.*":
        failures.append("Python 3.12 contract drift")
    expected_external = sorted(normalize_name(name) for name in EXPECTED_VERSIONS if name != PROJECT_NAME)
    if selected != sorted([normalize_name(PROJECT_NAME), *expected_external]):
        failures.append("selected Linux x86_64 closure drift")
    row_evidence: list[dict[str, Any]] = []
    for name in sorted(rows):
        row = rows[name]
        row_evidence.append(
            {
                "name": name,
                "version": row.get("version"),
                "source": row.get("source"),
                "dependencies": row.get("dependencies", []),
                "sdist": row.get("sdist"),
                "wheels": row.get("wheels", []),
                "selected": name in selected,
            }
        )
    return {
        "selected_platform": "Linux x86_64 CPython 3.12",
        "resolution_markers": lock.get("resolution-markers", []),
        "selected_closure": selected,
        "rows": row_evidence,
        "rows_sha256": sha256_bytes(json.dumps(row_evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")),
    }, failures


def collect(project_path: Path, lock_path: Path, environment_root: Path, site_packages: Path) -> dict[str, Any]:
    failures: list[str] = []
    try:
        project_bytes = project_path.read_bytes()
        lock_bytes = lock_path.read_bytes()
        project = tomllib.loads(project_bytes.decode("utf-8"))
        lock = tomllib.loads(lock_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        return {
            "schema": "microwakeword-reference-dependency-evidence-v1",
            "status": "COLLECTION_FAILED_FAIL_CLOSED",
            "publication_permitted": False,
            "fixture_generation_permitted": False,
            "failures": [f"project/lock parse failure: {error}"],
        }
    if sha256_bytes(project_bytes) != EXPECTED_PROJECT_SHA256:
        failures.append("pyproject digest drift")
    if sha256_bytes(lock_bytes) != EXPECTED_LOCK_SHA256:
        failures.append("uv.lock digest drift")
    lock_evidence, lock_failures = _lock_evidence(lock, project)
    failures.extend(lock_failures)
    installed_entries, installed_by_name, inventory_failures, inventory_sha256 = _installed_inventory(
        environment_root, site_packages
    )
    failures.extend(inventory_failures)
    distributions = []
    for name, version in sorted(EXPECTED_VERSIONS.items()):
        if name == PROJECT_NAME:
            continue
        normalized = normalize_name(name)
        distributions.append(
            collect_distribution(
                environment_root,
                site_packages,
                name,
                version,
                installed_by_name.get(normalized),
                inventory_sha256,
            )
        )
    if any(item["status"] != "EVIDENCE_COLLECTED_OWNER_REVIEW_REQUIRED" for item in distributions):
        failures.append("installed closure collection failed")
    return {
        "schema": "microwakeword-reference-dependency-evidence-v1",
        "status": "EVIDENCE_COLLECTED_OWNER_REVIEW_REQUIRED" if not failures else "COLLECTION_FAILED_FAIL_CLOSED",
        "publication_permitted": False,
        "fixture_generation_permitted": False,
        "owner_review_required": True,
        "platform": {"system": "Linux", "machine": "x86_64", "python": "3.12"},
        "project": {"path": project_path.name, "bytes": len(project_bytes), "sha256": sha256_bytes(project_bytes)},
        "uv_lock": {"path": lock_path.name, "bytes": len(lock_bytes), "sha256": sha256_bytes(lock_bytes)},
        "lock": lock_evidence,
        "installed_inventory": {
            "status": "PASS" if not inventory_failures else "FAIL_CLOSED",
            "sha256": inventory_sha256,
            "entries": installed_entries,
            "failures": inventory_failures,
        },
        "installed_distributions": distributions,
        "failures": failures,
    }


def _reject_symlink_ancestors(path: Path, label: str) -> None:
    if not path.is_absolute():
        raise SystemExit(f"{label} must be an absolute path: {path}")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise SystemExit(f"{label} has a symlink ancestor: {current}")


def require_regular_file(path: Path, label: str) -> None:
    _reject_symlink_ancestors(path, label)
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"{label} must be an existing regular non-symlink file: {path}")


def require_directory(path: Path, label: str) -> None:
    _reject_symlink_ancestors(path, label)
    if path.is_symlink() or not path.is_dir():
        raise SystemExit(f"{label} must be an existing non-symlink directory: {path}")


def write_exclusive(path: Path, payload: str) -> None:
    _reject_symlink_ancestors(path.parent, "output parent")
    if path.is_symlink() or path.exists() or not path.parent.is_dir() or path.parent.is_symlink():
        raise SystemExit(f"output must be an absent file with a real parent: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(payload)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _write_record(path: Path, rows: Iterable[tuple[str, str, str]]) -> None:
    path.write_text("".join(",".join(row) + "\n" for row in rows), encoding="utf-8")


def self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="microwakeword-reference-audit-") as temporary:
        root = Path(temporary) / "venv"
        site = root / "lib" / "python3.12" / "site-packages"
        site.mkdir(parents=True)
        dist = site / "tqdm-4.70.0.dist-info"
        dist.mkdir()
        package = site / "tqdm"
        package.mkdir()
        executable = root / "bin" / "runner"
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"#!/bin/sh\n")
        license_path = dist / "LICENCE.txt"
        license_path.write_bytes(b"Copyright owner\n")
        native = package / "engine.so.1"
        native.write_bytes(b"not an ELF\n")
        metadata = b"Name: tqdm\nVersion: 4.70.0\nLicense: BSD-3-Clause\nLicense-File: LICENCE.txt\nClassifier: License :: OSI Approved :: BSD License\n\n"
        (dist / "METADATA").write_bytes(metadata)

        def record_hash(data: bytes) -> str:
            return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode("ascii").rstrip("=")

        rows = [
            ("tqdm-4.70.0.dist-info/METADATA", record_hash(metadata), str(len(metadata))),
            ("tqdm-4.70.0.dist-info/LICENCE.txt", record_hash(license_path.read_bytes()), str(license_path.stat().st_size)),
            ("tqdm/engine.so.1", record_hash(native.read_bytes()), str(native.stat().st_size)),
            ("../../../bin/runner", record_hash(executable.read_bytes()), str(executable.stat().st_size)),
            ("tqdm-4.70.0.dist-info/RECORD", "", ""),
        ]
        _write_record(dist / "RECORD", rows)
        _, inventory, inventory_failures, inventory_sha256 = _installed_inventory(root, site, {"tqdm"})
        assert not inventory_failures, inventory_failures
        result = collect_distribution(root, site, "tqdm", "4.70.0", inventory["tqdm"], inventory_sha256)
        assert result["status"] == "EVIDENCE_COLLECTED_OWNER_REVIEW_REQUIRED", result
        assert result["license_candidates"][0]["path"].endswith("LICENCE.txt")
        assert result["native_payloads"][0]["path"].endswith("engine.so.1")
        assert len(result["record"]["entries"]) == 5
        runner_entry = next(item for item in result["record"]["entries"] if item["declared"]["path"] == "../../../bin/runner")
        assert runner_entry["validation"] == "MATCH"
        assert runner_entry["resolved_path"] == "bin/runner"

        mismatch_rows = list(rows)
        mismatch_rows[0] = (mismatch_rows[0][0], mismatch_rows[0][1], str(len(metadata) + 1))
        _write_record(dist / "RECORD", mismatch_rows)
        mismatch = collect_distribution(root, site, "tqdm", "4.70.0", inventory["tqdm"], inventory_sha256)
        assert mismatch["status"] == "COLLECTION_FAILED_FAIL_CLOSED"
        mismatch_entry = mismatch["record"]["entries"][0]
        assert mismatch_entry["validation"] == "FAIL"
        assert any("size mismatch" in failure for failure in mismatch["failures"])

        hash_mismatch_rows = list(rows)
        hash_mismatch_rows[0] = (hash_mismatch_rows[0][0], record_hash(b"tampered"), str(len(metadata)))
        _write_record(dist / "RECORD", hash_mismatch_rows)
        hash_mismatch = collect_distribution(root, site, "tqdm", "4.70.0", inventory["tqdm"], inventory_sha256)
        assert hash_mismatch["record"]["entries"][0]["validation"] == "FAIL"
        assert any("hash mismatch" in failure for failure in hash_mismatch["failures"])

        outside = root / "outside"
        outside.write_bytes(b"outside")
        (dist / "RECORD").write_text("../../../../outside,,7\n", encoding="utf-8")
        escaped = collect_distribution(root, site, "tqdm", "4.70.0", inventory["tqdm"], inventory_sha256)
        assert any("escapes environment root" in failure for failure in escaped["failures"])
        link = root / "bin" / "untrusted-link"
        link.symlink_to(outside)
        (dist / "RECORD").write_text("../../../bin/untrusted-link,,7\n", encoding="utf-8")
        symlinked = collect_distribution(root, site, "tqdm", "4.70.0", inventory["tqdm"], inventory_sha256)
        assert any("symlink in installed distribution" in failure for failure in symlinked["failures"])
        missing = collect_distribution(root, site, "missing", "1.0", None, inventory_sha256)
        assert missing["status"] == "COLLECTION_FAILED_FAIL_CLOSED"

        unknown_dist = site / "unknown-1.0.dist-info"
        unknown_dist.mkdir()
        (unknown_dist / "METADATA").write_text("Name: unknown\nVersion: 1.0\n\n", encoding="utf-8")
        broken_dist = site / "broken-1.0.dist-info"
        broken_dist.mkdir()
        _, _, inventory_failures, _ = _installed_inventory(root, site, {"tqdm"})
        assert any("unknown installed distribution" in failure for failure in inventory_failures)
        assert any("missing or symlinked METADATA" in failure for failure in inventory_failures)

        duplicate_dist = site / "tqdm_4.70.0.dist-info"
        duplicate_dist.mkdir()
        (duplicate_dist / "METADATA").write_text("Name: TQDM\nVersion: 4.70.0\n\n", encoding="utf-8")
        _, _, inventory_failures, _ = _installed_inventory(root, site, {"tqdm"})
        assert any("duplicate installed distribution name: tqdm" in failure for failure in inventory_failures)
    print("microWakeWord closure collector self-test: PASS (fake dist-info only)", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--environment-root", type=Path)
    parser.add_argument("--site-packages", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.project, args.lock, args.environment_root, args.site_packages, args.output)):
            parser.error("--self-test cannot be combined with paths/output")
        return self_test()
    if not all(value is not None for value in (args.project, args.lock, args.environment_root, args.site_packages, args.output)):
        parser.error("--project, --lock, --environment-root, --site-packages, and --output are required")
    require_regular_file(args.project, "project")
    require_regular_file(args.lock, "lock")
    require_directory(args.environment_root, "environment root")
    require_directory(args.site_packages, "site-packages")
    try:
        args.site_packages.relative_to(args.environment_root)
    except ValueError as error:
        raise SystemExit("site-packages must be inside environment root") from error
    report = collect(args.project, args.lock, args.environment_root, args.site_packages)
    write_exclusive(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "EVIDENCE_COLLECTED_OWNER_REVIEW_REQUIRED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
