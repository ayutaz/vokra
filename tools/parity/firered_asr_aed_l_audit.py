#!/usr/bin/env -S uv run --frozen --project tools/parity/firered_asr_aed_l --python 3.12 python
"""Audit the dedicated FireRed Python closure without acquiring a model.

The lock is the authority for the active Linux/x86_64 package graph.  This
tool records installed distribution metadata, publisher/license candidates,
and native ELF payload hashes.  It deliberately remains BLOCKED until every
transitive row has an explicit owner review; an audit is not a license grant.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import tempfile
import tomllib
from pathlib import Path
from typing import Any

REPOSITORY = "FireRedTeam/FireRedASR-AED-L"
MODEL_REVISION = "e57f5960d03cff1071ff7acbb409314d1e70ed3d"
AUDIT_FORMAT = "vokra-firered-asr-aed-l-dependency-audit-v1"
OWNER_APPROVAL_FORMAT = "vokra-firered-asr-aed-l-owner-approval-v1"
OWNER_APPROVAL_DECISION = "APPROVE"
OWNER_HANDLE = "yousan"
EXPECTED_ACTIVE_CLOSURE_ROWS = 27
NATIVE_SOURCE_URL = "https://github.com/csukuangfj/kaldi-native-fbank.git"
NATIVE_SOURCE_REVISION = "f68c6b43f739697d7ab02ff6debacee130e1d541"
NATIVE_SOURCE_LICENSE_PATH = "LICENSE"
NATIVE_SOURCE_LICENSE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"


class DuplicateJsonKey(ValueError):
    """Raised when an approval artifact attempts last-key-wins parsing."""


def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKey(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result

UNLOCK_REQUIREMENTS = (
    {
        "id": "owner_per_row_license_review",
        "status": "OWNER_REVIEW_REQUIRED",
        "evidence": "explicit owner approval for every active closure row, publisher, and native payload",
    },
    {
        "id": "native_frontend_dependency_review",
        "status": "OWNER_REVIEW_REQUIRED",
        "evidence": "license/source review for kaldi-native-fbank and its transitive native payloads",
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_regular_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"refusing to hash non-regular file: {path}")
    return sha256_file(path)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _publisher_urls(metadata: importlib.metadata.PackageMetadata) -> list[dict[str, str]]:
    """Capture declared project/publisher URLs without treating them as approval."""
    candidates: list[dict[str, str]] = []
    for field in ("Home-page", "Project-URL", "Download-URL", "Source", "Source-URL"):
        for value in metadata.get_all(field) or []:
            value = value.strip()
            if not value:
                continue
            url = value.rsplit(",", 1)[-1].strip() if field == "Project-URL" else value
            if "://" in url:
                candidates.append({"field": field, "value": value, "url": url})
    return sorted(candidates, key=lambda item: (item["field"], item["url"], item["value"]))


def _metadata_evidence(distribution: importlib.metadata.Distribution) -> dict[str, Any] | None:
    """Hash the installed METADATA file when importlib exposes it."""
    for file in distribution.files or ():
        if str(file).upper().endswith(".DIST-INFO/METADATA"):
            path = distribution.locate_file(file)
            if path.is_file() and not path.is_symlink():
                return {"path": str(file), "bytes": path.stat().st_size, "sha256": sha256_regular_file(path)}
    return None


def _pinned_native_source_license(project_path: Path) -> dict[str, Any]:
    """Verify the checked-out native frontend source and record its license."""
    license_path = project_path / NATIVE_SOURCE_LICENSE_PATH
    evidence: dict[str, Any] = {
        "kind": "pinned_source_license",
        "source_url": NATIVE_SOURCE_URL,
        "source_revision": NATIVE_SOURCE_REVISION,
        "source_revision_observed": None,
        "source_url_observed": None,
        "source_revision_verified": False,
        "source_url_verified": False,
        "license_path": NATIVE_SOURCE_LICENSE_PATH,
        "license_bytes": None,
        "license_sha256": None,
        "license_sha256_verified": False,
    }
    try:
        revision = subprocess.run(
            ["git", "-C", str(project_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        evidence["source_revision_observed"] = revision
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        remote = subprocess.run(
            ["git", "-C", str(project_path), "remote", "get-url", "origin"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        evidence["source_url_observed"] = remote
    except (OSError, subprocess.CalledProcessError):
        pass
    if license_path.is_file() and not license_path.is_symlink():
        evidence["license_bytes"] = license_path.stat().st_size
        evidence["license_sha256"] = sha256_regular_file(license_path)
    evidence["source_revision_verified"] = evidence["source_revision_observed"] == NATIVE_SOURCE_REVISION
    observed_url = evidence["source_url_observed"]
    if isinstance(observed_url, str):
        evidence["source_url_verified"] = observed_url.removesuffix("/").removesuffix(".git") == NATIVE_SOURCE_URL.removesuffix(".git")
    evidence["license_sha256_verified"] = evidence["license_sha256"] == NATIVE_SOURCE_LICENSE_SHA256
    return evidence


def _project_record(name: str, project_path: Path) -> dict[str, Any] | None:
    """Represent the lock's virtual local project without inventing a wheel."""
    pyproject = project_path / "pyproject.toml"
    if not pyproject.is_file() or pyproject.is_symlink():
        return None
    with pyproject.open("rb") as stream:
        project = tomllib.load(stream).get("project", {})
    if not isinstance(project, dict) or project.get("name") != name or not isinstance(project.get("version"), str):
        return None
    urls = []
    for label, url in (project.get("urls") or {}).items():
        if isinstance(label, str) and isinstance(url, str) and "://" in url:
            urls.append({"field": f"Project-URL: {label}", "value": url, "url": url})
    license_value = project.get("license")
    candidates = []
    if isinstance(license_value, str) and license_value:
        candidates.append({"field": "License", "value": license_value})
    elif isinstance(license_value, dict):
        for key in ("text", "file"):
            if isinstance(license_value.get(key), str) and license_value[key]:
                candidates.append({"field": f"License-{key.title()}", "value": license_value[key]})
    return {
        "name": name,
        "installed": True,
        "installation_scope": "local_project",
        "metadata_kind": "pyproject.toml",
        "version": project["version"],
        "metadata": {"kind": "pyproject.toml", "path": str(pyproject), "bytes": pyproject.stat().st_size, "sha256": sha256_regular_file(pyproject)},
        "publisher_urls": sorted(urls, key=lambda item: (item["field"], item["url"])),
        "license_candidates": candidates,
        "native_payloads": [],
    }


def _version_matches(installed: Any, locked: Any) -> bool:
    return installed == locked or (
        isinstance(installed, str)
        and isinstance(locked, str)
        and installed.split("+", 1)[0] == locked
    )


def publish_json_no_clobber(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        raise ValueError(f"refusing to overwrite dependency audit: {path}")
    if path.parent.is_symlink():
        raise ValueError(f"dependency audit path ancestor is a symlink: {path.parent}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False, mode="w", encoding="utf-8") as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _matches_linux_x86_64(row: dict[str, Any]) -> bool:
    marker = str(row.get("resolution-markers", row.get("marker", ""))).lower()
    return not marker or not any(token in marker for token in ("sys_platform == 'win32'", "sys_platform == 'darwin'", "platform_machine == 'aarch64'"))


def active_rows(lock_path: Path) -> list[dict[str, Any]]:
    with lock_path.open("rb") as stream:
        lock = tomllib.load(stream)
    packages = lock.get("package")
    if not isinstance(packages, list) or not packages:
        raise ValueError("uv lock has no package rows")
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in packages:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ValueError("uv lock package row is malformed")
        if _matches_linux_x86_64(row):
            by_name.setdefault(row["name"], []).append(row)
    roots = [row for row in packages if isinstance(row.get("name"), str) and row["name"].startswith("vokra-firered-asr-aed-l")]
    if not roots:
        # uv may name the local project after its directory; retaining all
        # rows is safer than silently omitting a dependency from an audit.
        selected = list(by_name.values())
    else:
        selected_names = {row["name"] for row in roots}
        pending = list(selected_names)
        while pending:
            name = pending.pop()
            for row in by_name.get(name, []):
                for dependency in row.get("dependencies", []):
                    dep_name = dependency.get("name") if isinstance(dependency, dict) else None
                    if isinstance(dep_name, str) and dep_name not in selected_names:
                        selected_names.add(dep_name)
                        pending.append(dep_name)
        selected = [by_name[name] for name in sorted(selected_names) if name in by_name]
    rows: list[dict[str, Any]] = []
    for variants in selected:
        for row in variants:
            rows.append({
                "name": row["name"],
                "version": row.get("version"),
                "source": row.get("source"),
                "dependencies": row.get("dependencies", []),
                "resolution_markers": row.get("resolution-markers", row.get("marker")),
            })
    if not rows:
        raise ValueError("active Linux/x86_64 closure is empty")
    return sorted(rows, key=lambda row: (row["name"], str(row.get("version")), json.dumps(row.get("source"), sort_keys=True)))


def distribution_record(
    name: str,
    local_project_path: Path | None = None,
    native_source_path: Path | None = None,
) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        project_record = _project_record(name, local_project_path) if local_project_path is not None else None
        if project_record is not None:
            return project_record
        record = {
            "name": name,
            "installed": False,
            "installation_scope": "distribution",
            "metadata_kind": "METADATA",
            "metadata": None,
            "publisher_urls": [],
            "license_candidates": [],
            "native_payloads": [],
        }
        if name == "kaldi-native-fbank" and native_source_path is not None:
            source_license = _pinned_native_source_license(native_source_path)
            record["license_candidates"].append(source_license)
            record["native_source_license"] = source_license
        return record
    metadata = distribution.metadata
    licenses = []
    for key in ("License", "License-Expression", "License-File"):
        values = metadata.get_all(key) or []
        licenses.extend({"field": key, "value": value} for value in values if value)
    licenses.extend({"field": "Classifier", "value": value} for value in (metadata.get_all("Classifier") or []) if "License" in value)
    native: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = list(licenses)
    for file in distribution.files or ():
        path = distribution.locate_file(file)
        lower = str(file).lower()
        if any(token in Path(lower).name for token in ("license", "copying", "notice")):
            regular = path.is_file() and not path.is_symlink()
            candidate = {"path": str(file), "exists": path.exists(), "regular": regular, "symlink": path.is_symlink()}
            if regular:
                candidate.update({"bytes": path.stat().st_size, "sha256": sha256_regular_file(path)})
            candidates.append(candidate)
        regular = path.is_file() and not path.is_symlink()
        elf = False
        if regular:
            with path.open("rb") as stream:
                elf = stream.read(4) == b"\x7fELF"
        if lower.endswith((".so", ".so.1", ".pyd", ".dylib")) or elf:
            item = {"path": str(file), "exists": path.exists(), "regular": regular, "symlink": path.is_symlink(), "elf": elf}
            if regular:
                item.update({"bytes": path.stat().st_size, "sha256": sha256_regular_file(path)})
            native.append(item)
    record = {
        "name": name,
        "installed": True,
        "installation_scope": "distribution",
        "metadata_kind": "METADATA",
        "version": distribution.version,
        "metadata": _metadata_evidence(distribution),
        "publisher_urls": _publisher_urls(metadata),
        "license_candidates": candidates,
        "native_payloads": native,
    }
    if name == "kaldi-native-fbank" and native_source_path is not None:
        source_license = _pinned_native_source_license(native_source_path)
        record["license_candidates"].append(source_license)
        record["native_source_license"] = source_license
    return record


def _source_identity(row: dict[str, Any]) -> dict[str, Any]:
    source = row.get("source")
    if not isinstance(source, dict):
        return {"kind": "unresolved", "value": source}
    if "git" in source:
        return {"kind": "git", "value": source["git"]}
    if "registry" in source:
        return {"kind": "registry", "value": source["registry"]}
    if "virtual" in source:
        return {"kind": "virtual", "value": source["virtual"]}
    return {"kind": "unknown", "value": source}


def _review_ledger(rows: list[dict[str, Any]], evidence: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ledger = []
    for row in rows:
        record = evidence[row["name"]]
        ledger.append(
            {
                "name": row["name"],
                "version": row.get("version"),
                "source_identity": _source_identity(row),
                "row_sha256": row["row_sha256"],
                "publisher_urls_sha256": canonical_sha256(record["publisher_urls"]),
                "license_candidates_sha256": canonical_sha256(record["license_candidates"]),
                "native_payloads_sha256": canonical_sha256(record["native_payloads"]),
                "review_status": "OWNER_REVIEW_REQUIRED",
                "owner_decision": None,
            }
        )
    return ledger


def _approval_failures(approval: dict[str, Any], scope: dict[str, Any], ledger: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    if set(approval) != {"format", "owner", "approved_at_utc", "decision", "scope", "scope_sha256", "rows"}:
        failures.append("owner approval top-level schema has missing or extra fields")
    if approval.get("format") != OWNER_APPROVAL_FORMAT:
        failures.append("owner approval format mismatch")
    if approval.get("owner") != OWNER_HANDLE:
        failures.append(f"owner approval owner must be {OWNER_HANDLE}")
    timestamp = approval.get("approved_at_utc")
    if not isinstance(timestamp, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z", timestamp):
        failures.append("owner approval timestamp is missing or malformed")
    else:
        try:
            approved_at = datetime.fromisoformat(timestamp[:-1] + "+00:00")
        except ValueError:
            failures.append("owner approval timestamp is not a valid UTC calendar time")
        else:
            if approved_at > datetime.now(timezone.utc) + timedelta(minutes=5):
                failures.append("owner approval timestamp is too far in the future")
    if approval.get("decision") != OWNER_APPROVAL_DECISION:
        failures.append("owner approval decision is not APPROVE")
    if set(scope) != {"lock_sha256", "active_closure_sha256", "source_identity_aggregate_sha256", "distribution_evidence_sha256", "license_candidate_aggregate_sha256", "native_payload_aggregate_sha256", "publisher_url_aggregate_sha256", "review_ledger_sha256"}:
        failures.append("audit digest scope schema drifted")
    if approval.get("scope") != scope or approval.get("scope_sha256") != canonical_sha256(scope):
        failures.append("owner approval exact digest scope mismatch")
    approved_rows = approval.get("rows")
    if len(ledger) != EXPECTED_ACTIVE_CLOSURE_ROWS:
        failures.append(f"active dependency closure must contain exactly {EXPECTED_ACTIVE_CLOSURE_ROWS} rows")
    if not isinstance(approved_rows, list) or len(approved_rows) != EXPECTED_ACTIVE_CLOSURE_ROWS:
        failures.append(f"owner approval must contain exactly {EXPECTED_ACTIVE_CLOSURE_ROWS} active closure rows")
        return failures
    expected = {(row["name"], row["version"], row["row_sha256"]): row for row in ledger}
    seen: set[tuple[Any, ...]] = set()
    required = {
        "name",
        "version",
        "source_identity",
        "row_sha256",
        "publisher_urls_sha256",
        "license_candidates_sha256",
        "native_payloads_sha256",
        "license_review",
        "publisher_review",
        "native_payload_review",
    }
    for row in approved_rows:
        if not isinstance(row, dict) or set(row) != required:
            failures.append("owner approval row is incomplete")
            continue
        key = (row.get("name"), row.get("version"), row.get("row_sha256"))
        expected_row = expected.get(key)
        if expected_row is None or key in seen:
            failures.append(f"owner approval row identity mismatch: {key!r}")
            continue
        seen.add(key)
        for field in ("publisher_urls_sha256", "license_candidates_sha256", "native_payloads_sha256"):
            if row[field] != expected_row[field]:
                failures.append(f"owner approval {field} mismatch: {row['name']}")
        if row["source_identity"] != expected_row["source_identity"]:
            failures.append(f"owner approval source identity mismatch: {row['name']}")
        for field in ("license_review", "publisher_review", "native_payload_review"):
            if row[field] != OWNER_APPROVAL_DECISION:
                failures.append(f"owner approval {field} is not APPROVE: {row['name']}")
    if len(seen) != EXPECTED_ACTIVE_CLOSURE_ROWS:
        failures.append("owner approval row set is incomplete")
    return failures


def build_manifest(
    lock_path: Path,
    project: Path | None,
    approval: dict[str, Any] | None = None,
    approval_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = active_rows(lock_path)
    row_records = []
    for row in rows:
        row_records.append({**row, "row_sha256": canonical_sha256(row)})
    evidence_records = {
        row["name"]: distribution_record(row["name"], lock_path.parent, project)
        for row in rows
    }
    collection_failures = []
    for row in row_records:
        evidence = evidence_records[row["name"]]
        installed_version = evidence.get("version")
        locked_version = row.get("version")
        # The CPU torch wheel carries a local +cpu build suffix; that suffix is
        # part of the wheel identity but does not change the locked release.
        version_match = _version_matches(installed_version, locked_version)
        evidence["locked_version"] = locked_version
        evidence["version_match"] = version_match
        if not evidence.get("installed"):
            collection_failures.append(f"missing installed distribution: {row['name']}")
        if not version_match:
            collection_failures.append(f"installed version mismatch: {row['name']}")
        if evidence.get("metadata") is None:
            collection_failures.append(f"missing METADATA evidence: {row['name']}")
        if row["name"] == "kaldi-native-fbank":
            source_license = evidence.get("native_source_license")
            if not isinstance(source_license, dict) or not all(
                source_license.get(key) is True
                for key in ("source_revision_verified", "source_url_verified", "license_sha256_verified")
            ):
                collection_failures.append("kaldi-native-fbank pinned source/LICENSE evidence is not verified")
    # Include the virtual local project row as a first-class evidence record;
    # it is matched from this lock's pyproject rather than silently omitted.
    packages = [evidence_records[row["name"]] for row in rows]
    native_source = evidence_records.get("kaldi-native-fbank", {}).get("native_source_license")
    ledger = _review_ledger(row_records, evidence_records)
    license_aggregate = [
        {"name": row["name"], "version": row.get("version"), "license_candidates": evidence_records[row["name"]]["license_candidates"]}
        for row in row_records
    ]
    native_aggregate = [
        {"name": row["name"], "version": row.get("version"), "native_payloads": evidence_records[row["name"]]["native_payloads"]}
        for row in row_records
    ]
    publisher_aggregate = [
        {"name": row["name"], "version": row.get("version"), "publisher_urls": evidence_records[row["name"]]["publisher_urls"]}
        for row in row_records
    ]
    distribution_aggregate = [evidence_records[row["name"]] for row in row_records]
    source_aggregate = [
        {"name": row["name"], "version": row.get("version"), "source_identity": _source_identity(row), "row_sha256": row["row_sha256"]}
        for row in row_records
    ]
    exact_scope = {
        "lock_sha256": sha256_regular_file(lock_path),
        "active_closure_sha256": canonical_sha256(row_records),
        "source_identity_aggregate_sha256": canonical_sha256(source_aggregate),
        "distribution_evidence_sha256": canonical_sha256(distribution_aggregate),
        "license_candidate_aggregate_sha256": canonical_sha256(license_aggregate),
        "native_payload_aggregate_sha256": canonical_sha256(native_aggregate),
        "publisher_url_aggregate_sha256": canonical_sha256(publisher_aggregate),
        "review_ledger_sha256": canonical_sha256(ledger),
    }
    approval_failures = _approval_failures(approval, exact_scope, ledger) if approval is not None else ["owner approval artifact was not supplied"]
    approval_valid = approval is not None and not approval_failures
    approved = approval_valid and not collection_failures
    status = "OWNER_APPROVED" if approved else "BLOCKED_UNREVIEWED_TRANSITIVE"
    owner_approval = {
        "status": "VALIDATED" if approval_valid else ("INVALID" if approval is not None else "MISSING"),
        "required": "explicit owner approval for every active closure row, publisher URLs, license candidates, and native payloads",
        "minimum_fields": ["format", "owner", "approved_at_utc", "decision", "scope", "scope_sha256", "rows"],
        "scope_sha256": canonical_sha256(exact_scope),
    }
    if approval_failures:
        owner_approval["failures"] = approval_failures
    if collection_failures:
        owner_approval.setdefault("failures", []).extend(collection_failures)
    if approval_artifact is not None:
        owner_approval["artifact"] = approval_artifact
        owner_approval["summary"] = {
            "owner": approval.get("owner") if approval else None,
            "approved_at_utc": approval.get("approved_at_utc") if approval else None,
            "decision": approval.get("decision") if approval else None,
        }
    protocol_status = (
        "OWNER_APPROVED"
        if approved
        else "BLOCKED_INVALID_OWNER_APPROVAL"
        if approval is not None and approval_failures
        else "BLOCKED_INCOMPLETE_EVIDENCE"
        if approval is not None
        else "BLOCKED_NO_OWNER_APPROVAL"
    )
    protocol_reason = (
        "exact owner approval scope and complete distribution evidence validated"
        if approved
        else "invalid owner approval artifact"
        if approval is not None and approval_failures
        else "installed distribution evidence is incomplete"
        if approval is not None
        else "owner approval artifact was not supplied"
    )
    return {
        "format": AUDIT_FORMAT,
        "status": status,
        "publication": "NO_UPLOAD",
        "platform": {"system": platform.system(), "machine": platform.machine(), "required": "Linux/x86_64"},
        "lock": {"path": str(lock_path), "sha256": sha256_regular_file(lock_path)},
        "lock_artifact": {"format": "uv.lock", "path": str(lock_path), "sha256": sha256_regular_file(lock_path), "active_platform": "Linux/x86_64"},
        "model": {"repository": REPOSITORY, "revision": MODEL_REVISION},
        "active_closure": {"rows": row_records, "row_count": len(row_records), "row_digest": canonical_sha256(row_records)},
        "installed_distributions": packages,
        "distribution_evidence": [evidence_records[row["name"]] for row in row_records],
        "distribution_evidence_sha256": exact_scope["distribution_evidence_sha256"],
        "source_identity_aggregate": {"sha256": exact_scope["source_identity_aggregate_sha256"], "rows": source_aggregate},
        "publisher_url_aggregate": {"sha256": exact_scope["publisher_url_aggregate_sha256"], "rows": publisher_aggregate},
        "license_candidate_aggregate": {"sha256": exact_scope["license_candidate_aggregate_sha256"], "rows": license_aggregate},
        "native_payload_aggregate": {"sha256": exact_scope["native_payload_aggregate_sha256"], "rows": native_aggregate},
        "review_ledger": {"row_count": len(ledger), "sha256": exact_scope["review_ledger_sha256"], "rows": ledger},
        "exact_digest_gate": {"algorithm": "sha256", "scope": exact_scope, "scope_sha256": canonical_sha256(exact_scope)},
        "native_source": native_source,
        "owner_approval": owner_approval,
        "collection_failures": collection_failures,
        "gate": {"status": status, "reason": "all active closure rows require explicit owner approval before model snapshot" if not approved else "exact owner approval scope matches immutable audit evidence"},
        "unlock_requirements": list(UNLOCK_REQUIREMENTS),
        "collection_protocol": {
            "status": protocol_status,
            "reason": protocol_reason,
            "vast_command": "VOKRA_PUBLISH_ON_VAST=1 bash scripts/publish/vast-ai/run-firered-asr-aed-l-inspection.sh --work-dir /dev/shm/vokra-firered-asr-aed-l-inspection",
            "expected_artifacts": (
                ["dependency-audit.json", "validation.log", "manifest.json", "server_tree.json"]
                if approved
                else ["dependency-audit.json", "validation.log"]
            ),
            "approved_route_expected_artifacts": [
                "dependency-audit.json",
                "validation.log",
                "manifest.json",
                "server_tree.json",
            ],
        },
    }


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="firered-audit-") as directory:
        lock = Path(directory) / "uv.lock"
        extra_names = [f"synthetic-{index:02d}" for index in range(23)]
        dependencies = ', '.join(
            [
                '{ name = "synthetic" }',
                '{ name = "setuptools" }',
                '{ name = "kaldi-native-fbank" }',
                *(f'{{ name = "{name}" }}' for name in extra_names),
            ]
        )
        extra_packages = "\n".join(
            f'[[package]]\nname = "{name}"\nversion = "1.2.3"\nsource = {{ registry = "https://example.invalid" }}'
            for name in extra_names
        )
        lock.write_text(
            f'''version = 1
[[package]]
name = "firered-asr-aed-l"
version = "0.0.0"
dependencies = [{dependencies}]
[[package]]
name = "setuptools"
version = "80.9.0"
source = {{ registry = "https://pypi.org/simple" }}
[[package]]
name = "synthetic"
version = "1.2.3"
source = {{ registry = "https://example.invalid" }}
[[package]]
name = "kaldi-native-fbank"
version = "1.15"
source = {{ registry = "https://pypi.org/simple" }}
{extra_packages}
''',
            encoding="utf-8",
        )
        (Path(directory) / "pyproject.toml").write_text('[project]\nname = "firered-asr-aed-l"\nversion = "0.0.0"\n', encoding="utf-8")
        rows = active_rows(lock)
        assert len(rows) == 27
        assert len({row["name"] for row in rows}) == 27
        assert {"firered-asr-aed-l", "setuptools", "synthetic"}.issubset({row["name"] for row in rows})
        assert next(row for row in rows if row["name"] == "setuptools")["version"] == "80.9.0"
        manifest = build_manifest(lock, None)
        assert manifest["status"] == "BLOCKED_UNREVIEWED_TRANSITIVE"
        assert manifest["publication"] == "NO_UPLOAD"
        assert {item["id"] for item in manifest["unlock_requirements"]} == {
            "owner_per_row_license_review",
            "native_frontend_dependency_review",
        }
        assert len(manifest["lock"]["sha256"]) == 64
        assert len(manifest["active_closure"]["row_digest"]) == 64
        assert all(len(row["row_sha256"]) == 64 for row in manifest["active_closure"]["rows"])
        assert manifest["review_ledger"]["row_count"] == 27
        assert len(manifest["installed_distributions"]) == 27
        assert len(manifest["review_ledger"]["sha256"]) == 64
        assert len(manifest["exact_digest_gate"]["scope_sha256"]) == 64
        assert manifest["lock_artifact"]["format"] == "uv.lock"
        assert len(manifest["source_identity_aggregate"]["sha256"]) == 64
        assert manifest["owner_approval"]["status"] == "MISSING"
        assert manifest["collection_protocol"]["status"] == "BLOCKED_NO_OWNER_APPROVAL"
        assert manifest["collection_protocol"]["expected_artifacts"] == ["dependency-audit.json", "validation.log"]
        assert manifest["collection_protocol"]["approved_route_expected_artifacts"] == [
            "dependency-audit.json",
            "validation.log",
            "manifest.json",
            "server_tree.json",
        ]
        root_evidence = next(item for item in manifest["distribution_evidence"] if item["name"] == "firered-asr-aed-l")
        assert root_evidence["installed"] is True
        assert root_evidence["installation_scope"] == "local_project"
        assert root_evidence["metadata_kind"] == "pyproject.toml"
        assert "kaldi-native-fbank pinned source/LICENSE evidence is not verified" in manifest["collection_failures"]
        approval = {
            "format": OWNER_APPROVAL_FORMAT,
            "owner": OWNER_HANDLE,
            "approved_at_utc": "2026-01-01T00:00:00Z",
            "decision": OWNER_APPROVAL_DECISION,
            "scope": manifest["exact_digest_gate"]["scope"],
            "scope_sha256": manifest["exact_digest_gate"]["scope_sha256"],
            "rows": [
                {
                    "name": row["name"],
                    "version": row["version"],
                    "source_identity": row["source_identity"],
                    "row_sha256": row["row_sha256"],
                    "publisher_urls_sha256": row["publisher_urls_sha256"],
                    "license_candidates_sha256": row["license_candidates_sha256"],
                    "native_payloads_sha256": row["native_payloads_sha256"],
                    "license_review": OWNER_APPROVAL_DECISION,
                    "publisher_review": OWNER_APPROVAL_DECISION,
                    "native_payload_review": OWNER_APPROVAL_DECISION,
                }
                for row in manifest["review_ledger"]["rows"]
            ],
        }
        assert _approval_failures(approval, manifest["exact_digest_gate"]["scope"], manifest["review_ledger"]["rows"]) == []
        assert _approval_failures({**approval, "extra": True}, manifest["exact_digest_gate"]["scope"], manifest["review_ledger"]["rows"])
        assert _approval_failures({**approval, "owner": "other"}, manifest["exact_digest_gate"]["scope"], manifest["review_ledger"]["rows"])
        assert _approval_failures({**approval, "approved_at_utc": "2099-01-01T00:00:00Z"}, manifest["exact_digest_gate"]["scope"], manifest["review_ledger"]["rows"])
        assert _approval_failures({**approval, "rows": [*approval["rows"], approval["rows"][0]]}, manifest["exact_digest_gate"]["scope"], manifest["review_ledger"]["rows"])
        invalid = build_manifest(lock, None, {"_invalid_json": True}, {"path": "approval.json", "bytes": 1, "sha256": "0" * 64})
        assert invalid["status"] == "BLOCKED_UNREVIEWED_TRANSITIVE"
        assert invalid["owner_approval"]["status"] == "INVALID"
        assert invalid["collection_protocol"]["status"] == "BLOCKED_INVALID_OWNER_APPROVAL"
        for duplicate in (
            '{"format": 1, "format": 2}',
            '{"scope": {"lock_sha256": "a", "lock_sha256": "b"}}',
            '{"rows": [{"source_identity": {"kind": "git", "kind": "registry"}}]}',
        ):
            try:
                json.loads(duplicate, object_pairs_hook=reject_duplicate_pairs)
            except DuplicateJsonKey:
                pass
            else:
                raise AssertionError("duplicate approval JSON key was accepted")
        original_record = globals()["distribution_record"]
        versions = {row["name"]: str(row["version"]) for row in rows}

        native_source = Path(directory) / "kaldi-source"
        routed_calls: list[tuple[str, Path | None, Path | None]] = []

        def complete_record(
            name: str,
            local_project_path: Path | None = None,
            native_source_path: Path | None = None,
        ) -> dict[str, Any]:
            routed_calls.append((name, local_project_path, native_source_path))
            record = {
                "name": name,
                "installed": True,
                "installation_scope": "local_project" if name == "firered-asr-aed-l" else "distribution",
                "metadata_kind": "pyproject.toml" if name == "firered-asr-aed-l" else "METADATA",
                "version": versions[name],
                "metadata": {"kind": "fixture", "path": f"{name}/METADATA", "bytes": 1, "sha256": "0" * 64},
                "publisher_urls": [],
                "license_candidates": [],
                "native_payloads": [],
            }
            if name == "kaldi-native-fbank" and native_source_path is not None:
                source_license = {
                    "kind": "pinned_source_license",
                    "source_url": NATIVE_SOURCE_URL,
                    "source_revision": NATIVE_SOURCE_REVISION,
                    "source_revision_observed": NATIVE_SOURCE_REVISION,
                    "source_url_observed": NATIVE_SOURCE_URL,
                    "source_revision_verified": True,
                    "source_url_verified": True,
                    "license_path": NATIVE_SOURCE_LICENSE_PATH,
                    "license_bytes": 123,
                    "license_sha256": NATIVE_SOURCE_LICENSE_SHA256,
                    "license_sha256_verified": True,
                }
                record["license_candidates"] = [source_license]
                record["native_source_license"] = source_license
            return record

        globals()["distribution_record"] = complete_record
        try:
            complete = build_manifest(lock, native_source)
            assert routed_calls
            assert all(local == lock.parent and native == native_source for _, local, native in routed_calls)
            approval["scope"] = complete["exact_digest_gate"]["scope"]
            approval["scope_sha256"] = complete["exact_digest_gate"]["scope_sha256"]
            approval["rows"] = [
                {
                    "name": row["name"],
                    "version": row["version"],
                    "source_identity": row["source_identity"],
                    "row_sha256": row["row_sha256"],
                    "publisher_urls_sha256": row["publisher_urls_sha256"],
                    "license_candidates_sha256": row["license_candidates_sha256"],
                    "native_payloads_sha256": row["native_payloads_sha256"],
                    "license_review": OWNER_APPROVAL_DECISION,
                    "publisher_review": OWNER_APPROVAL_DECISION,
                    "native_payload_review": OWNER_APPROVAL_DECISION,
                }
                for row in complete["review_ledger"]["rows"]
            ]
            approved = build_manifest(lock, native_source, approval, {"path": "approval.json", "bytes": 1, "sha256": "0" * 64})
            assert approved["status"] == "OWNER_APPROVED"
            assert approved["owner_approval"]["status"] == "VALIDATED"
            assert approved["collection_protocol"]["status"] == "OWNER_APPROVED"
            assert approved["collection_protocol"]["expected_artifacts"] == approved["collection_protocol"]["approved_route_expected_artifacts"]
            kaldi = next(item for item in approved["distribution_evidence"] if item["name"] == "kaldi-native-fbank")
            assert kaldi["native_source_license"]["license_sha256"] == NATIVE_SOURCE_LICENSE_SHA256
            assert kaldi["license_candidates"][0]["source_revision"] == NATIVE_SOURCE_REVISION
            assert approved["exact_digest_gate"]["scope"]["license_candidate_aggregate_sha256"] == approved["license_candidate_aggregate"]["sha256"]
        finally:
            globals()["distribution_record"] = original_record
        # The synthetic environment has no installed transitive distributions;
        # exact owner approval must not override that collection blocker.
        assert build_manifest(lock, None, approval)["status"] == "BLOCKED_UNREVIEWED_TRANSITIVE"
        approval["scope"] = {**approval["scope"], "lock_sha256": "0" * 64}
        assert _approval_failures(approval, manifest["exact_digest_gate"]["scope"], manifest["review_ledger"]["rows"])
        output = Path(directory) / "audit.json"
        publish_json_no_clobber(output, manifest)
        try:
            publish_json_no_clobber(output, manifest)
        except ValueError:
            pass
        else:
            raise AssertionError("dependency audit clobber accepted")
        assert not list(Path(directory).glob("*.tmp"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--owner-approval", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("firered dependency audit self-test PASS")
        return 0
    if not args.lock or not args.output:
        parser.error("--lock and --output are required")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SystemExit("Linux/x86_64 audit is required")
    approval = None
    approval_artifact = None
    if args.owner_approval:
        if args.owner_approval.is_symlink() or not args.owner_approval.is_file():
            raise SystemExit("owner approval must be an existing regular JSON file")
        try:
            with args.owner_approval.open("rb") as stream:
                approval = json.load(stream, object_pairs_hook=reject_duplicate_pairs)
        except (OSError, json.JSONDecodeError, DuplicateJsonKey):
            approval = {"_invalid_json": True}
        if not isinstance(approval, dict):
            approval = {"_invalid_schema": True}
        approval_artifact = {"path": str(args.owner_approval), "bytes": args.owner_approval.stat().st_size, "sha256": sha256_regular_file(args.owner_approval)}
    manifest = build_manifest(args.lock, args.project, approval, approval_artifact)
    publish_json_no_clobber(args.output, manifest)
    print(f"firered dependency audit: {manifest['status']}")
    return 0 if manifest["status"] == "OWNER_APPROVED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
