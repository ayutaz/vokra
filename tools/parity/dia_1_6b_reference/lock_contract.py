"""Shared factual contract for the current Dia lock and Linux closure.

This module deliberately delegates lock parsing, marker evaluation, source
validation, and canonical digests to ``dependency_audit.py``.  It contains no
installed-distribution, Torch, model, or network access.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any


SCHEMA = "vokra-dia-uv-lock-license-audit-v2"
GATE_STATUS = "BLOCKED_UNREVIEWED_TRANSITIVE"
LICENSE_CLASSIFICATION = "NOT_PERFORMED"
LICENSE_REVIEW_STATUS = "PENDING_OWNER_REVIEW"
EXPECTED_ALL_ROWS = 31
EXPECTED_ACTIVE_ROWS = 27
EXPECTED_LOCK_SHA256 = "8356b43556bb3c94a75556b57b839cce15d4b4e899d8df033404c6b2ccf1c5de"
EXPECTED_PYPROJECT_SHA256 = "4dcc396ff3f7387b4b00b32db00ad79fa38f3cf1ef7ad22e3e7f3f8f563be4eb"
HEX64 = re.compile(r"[0-9a-f]{64}")
ALLOWED_REGISTRIES = {"https://pypi.org/simple", "https://download.pytorch.org/whl/cpu"}


class LockContractError(ValueError):
    """A malformed, stale, or tampered current-lock evidence packet."""


def _dependency_audit_module():
    module_name = "_vokra_dia_dependency_audit_contract"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    path = Path(__file__).with_name("dependency_audit.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise LockContractError("dependency_audit.py cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _digest(audit: Any, rows: list[dict[str, Any]]) -> str:
    return audit.sha256_bytes(audit.canonical(rows).encode("utf-8"))


def _row_shape(row: Any) -> bool:
    return (
        isinstance(row, dict)
        and set(row) == {"name", "version", "source", "resolution_markers", "dependencies"}
        and isinstance(row["name"], str)
        and isinstance(row["version"], str)
        and isinstance(row["source"], dict)
        and isinstance(row["resolution_markers"], list)
        and all(isinstance(marker, str) and marker.strip() for marker in row["resolution_markers"])
        and isinstance(row["dependencies"], list)
        and all(isinstance(dep, dict) and isinstance(dep.get("name"), str) for dep in row["dependencies"])
    )


def _validate_rows(rows: Any, expected_count: int, expected_digest: str, label: str) -> None:
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise LockContractError(f"{label} row count is not {expected_count}")
    identities: set[tuple[str, str, str]] = set()
    for row in rows:
        if not _row_shape(row):
            raise LockContractError(f"{label} row schema is malformed")
        source = row["source"]
        if set(source) not in ({"virtual"}, {"registry"}):
            raise LockContractError(f"{label} row source is malformed")
        if set(source) == {"virtual"} and (not isinstance(source.get("virtual"), str) or source.get("virtual") != "."):
            raise LockContractError(f"{label} virtual source is malformed")
        if set(source) == {"registry"} and (not isinstance(source.get("registry"), str) or source.get("registry") not in ALLOWED_REGISTRIES):
            raise LockContractError(f"{label} registry source is not reviewed")
        identity = (row["name"], row["version"], json.dumps(source, sort_keys=True, separators=(",", ":")))
        if identity in identities:
            raise LockContractError(f"{label} contains duplicate package identity")
        identities.add(identity)
    if rows != sorted(rows, key=lambda row: (
        row["name"].casefold().replace("_", "-").replace(".", "-"),
        row["version"].casefold(),
        json.dumps(row["source"], sort_keys=True),
    )):
        raise LockContractError(f"{label} rows are not canonically sorted")
    if not isinstance(expected_digest, str) or not HEX64.fullmatch(expected_digest):
        raise LockContractError(f"{label} digest is malformed")


def build_lock_audit(project: Path) -> dict[str, Any]:
    """Build the exact current all-row and Linux active-row evidence packet."""
    audit = _dependency_audit_module()
    try:
        _pyproject, _lock, rows, contract = audit.project_contract(project)
        active = audit.active_rows(rows)
    except Exception as error:
        raise LockContractError(f"current dependency contract rejected: {error}") from error
    if len(rows) != EXPECTED_ALL_ROWS or len(active) != EXPECTED_ACTIVE_ROWS:
        raise LockContractError("current lock cardinality changed")
    all_digest = _digest(audit, rows)
    active_digest = _digest(audit, active)
    if contract["all_lock_rows_sha256"] != all_digest or contract["active_linux_x86_64_rows_sha256"] != active_digest:
        raise LockContractError("dependency_audit digest semantics drifted")
    packet = {
        "schema": SCHEMA,
        "status": GATE_STATUS,
        "license_classification": LICENSE_CLASSIFICATION,
        "license_review_status": LICENSE_REVIEW_STATUS,
        "project": audit.PROJECT_NAME,
        "project_version": audit.PROJECT_VERSION,
        "lock_schema": audit.LOCK_SCHEMA,
        "pyproject_sha256": contract["pyproject_sha256"],
        "uv_lock_sha256": contract["uv_lock_sha256"],
        "direct_dependency_versions": dict(audit.EXPECTED_DIRECT_DEPENDENCIES),
        "all_lock_rows": {
            "package_count": len(rows),
            "rows_sha256": all_digest,
            "rows": rows,
        },
        "active_linux_x86_64_rows": {
            "package_count": len(active),
            "rows_sha256": active_digest,
            "rows": active,
        },
    }
    validate_lock_audit_packet(packet, project, expected=packet)
    return packet


def validate_lock_audit_packet(packet: Any, project: Path, *, expected: dict[str, Any] | None = None) -> None:
    """Validate schema, current project bytes, rows, markers, and digests."""
    if not isinstance(packet, dict):
        raise LockContractError("dependency audit packet is not an object")
    required = {
        "schema", "status", "license_classification", "license_review_status",
        "project", "project_version", "lock_schema", "pyproject_sha256",
        "uv_lock_sha256", "direct_dependency_versions", "all_lock_rows",
        "active_linux_x86_64_rows",
    }
    if set(packet) != required:
        raise LockContractError("dependency audit packet schema drifted")
    if packet["schema"] != SCHEMA or packet["status"] != GATE_STATUS or packet["license_classification"] != LICENSE_CLASSIFICATION or packet["license_review_status"] != LICENSE_REVIEW_STATUS:
        raise LockContractError("dependency audit status is not explicit pending/unreviewed factual evidence")
    if packet["project"] != "vokra-dia-1-6b-reference" or packet["project_version"] != "0.1.0" or packet["lock_schema"] != "uv-lock-v1-python312":
        raise LockContractError("dependency audit project identity drifted")
    if packet["pyproject_sha256"] != EXPECTED_PYPROJECT_SHA256 or packet["uv_lock_sha256"] != EXPECTED_LOCK_SHA256:
        raise LockContractError("dependency audit project bytes are stale")
    if not isinstance(packet["direct_dependency_versions"], dict):
        raise LockContractError("direct dependency map is malformed")
    audit = _dependency_audit_module()
    if packet["direct_dependency_versions"] != audit.EXPECTED_DIRECT_DEPENDENCIES:
        raise LockContractError("direct dependency map is stale or tampered")
    try:
        _pyproject, _lock, rows, contract = audit.project_contract(project)
        active = audit.active_rows(rows)
    except Exception as error:
        raise LockContractError(f"current dependency contract rejected: {error}") from error
    sections = (("all_lock_rows", rows, EXPECTED_ALL_ROWS, contract["all_lock_rows_sha256"]), ("active_linux_x86_64_rows", active, EXPECTED_ACTIVE_ROWS, contract["active_linux_x86_64_rows_sha256"]))
    for name, expected_rows, expected_count, expected_digest in sections:
        section = packet[name]
        if not isinstance(section, dict) or set(section) != {"package_count", "rows_sha256", "rows"}:
            raise LockContractError(f"{name} section schema drifted")
        if isinstance(section["package_count"], bool) or not isinstance(section["package_count"], int):
            raise LockContractError(f"{name} package count is malformed")
        _validate_rows(section["rows"], expected_count, section["rows_sha256"], name)
        actual_digest = _digest(audit, section["rows"])
        if section["package_count"] != expected_count or section["rows_sha256"] != expected_digest or actual_digest != expected_digest or section["rows"] != expected_rows:
            raise LockContractError(f"{name} membership or digest does not match current lock")
    if expected is not None and packet != expected:
        raise LockContractError("dependency audit packet differs from canonical current contract")
