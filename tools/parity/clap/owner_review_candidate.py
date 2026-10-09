#!/usr/bin/env python3
"""Validate the tracked, non-approving CLAP owner-review candidate.

This module intentionally has no third-party imports.  It binds the exact
model-free VAST evidence already collected for CLAP, but never turns that
evidence into an owner decision, model permission, or publication permission.
The candidate's payload digest makes the tracked record immutable: changing
any bound fact is rejected even when the replacement still looks plausible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parent
CANDIDATE = PROJECT / "owner_review_candidate.json"
LICENSE_AUDIT_DOCUMENT = PROJECT.parents[2] / "docs" / "license-audit.md"
LICENSE_ROW_PREFIX = b"| **CLAP HTSAT-fused** (`laion/clap-htsat-fused`)"
SCHEMA = "vokra-clap-htsat-fused-owner-review-candidate-v2"
CANONICALIZATION = "json-sort-keys-utf8-no-whitespace-v1"
REPOSITORY = "laion/clap-htsat-fused"
SOURCE_REPOSITORY = "https://huggingface.co/laion/clap-htsat-fused"
REVISION = "365dea6ef167def6676140ed93bbc43f84dabb28"
LICENSE_SIGNOFF = {
    "document": "docs/license-audit.md",
    "record": "CLAP HTSAT-fused (laion/clap-htsat-fused)",
    "date": "2026-07-30",
    "handle": "yousan",
    "decision": "COMMERCIAL",
    "license": "apache-2.0",
    "row_sha256": "3698402b9541de8e1836d60cbc52fc52de8305149c87172d0548113bc29ad833",
}
MODEL_FREE_AUDIT_SHA256 = (
    "3671fccdda418ef85bccc21d11888527ac9fb6394d725715e8327524474435d0"
)
DEPENDENCY_INVENTORY_SHA256 = (
    "9f28e0261ff1958481ff44c3b9e66367dbc0fe1182616bb6cb0110be7f258019"
)
SUMMARY_SHA256 = (
    "8610ce0d5f6cf3caf82cd9206f7850cefb4c067b4590fbe570d4beda06861670"
)
DEPENDENCY_FACTUAL_FINDINGS = 0
DEPENDENCY_REVIEW_FLAGS = 28
DEPENDENCY_DISTRIBUTIONS = 34
PREVIOUS_LOCK_SHA256 = "84c8f2fb375dd1532570d4de2dcd6a416e19847cdc511816f8db86387ec1974c"
CURRENT_LOCK_SHA256 = "a42cc43f4b3ba12cb40755e41ed8f73360866c4fc60716b9d19f20d45c91e3ec"


def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_document(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"candidate is missing, symlinked, or empty: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_pairs,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"candidate is invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("candidate root is not a JSON object")
    return value


def canonical_payload(document: dict[str, Any]) -> bytes:
    payload = dict(document)
    payload.pop("integrity", None)
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def payload_sha256(document: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_payload(document)).hexdigest()


def require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys drifted: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    return value


def require_hex(value: Any, length: int, label: str) -> None:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{label} is not {length}-hex")
    if any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} is not lowercase hexadecimal")


def read_license_audit_row(path: Path = LICENSE_AUDIT_DOCUMENT) -> bytes:
    """Return the unique, exact CLAP row from a regular audit document."""

    try:
        mode = os.lstat(path).st_mode
    except OSError as exc:
        raise ValueError(f"license audit document cannot be inspected: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise ValueError(f"license audit document is not a regular file: {path}")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"license audit document cannot be read: {path}") from exc
    rows = [line for line in payload.splitlines(keepends=True) if line.startswith(LICENSE_ROW_PREFIX)]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one CLAP audit row, found {len(rows)}")
    row = rows[0]
    row_sha256 = hashlib.sha256(row).hexdigest()
    if row_sha256 != LICENSE_SIGNOFF["row_sha256"]:
        raise ValueError("CLAP audit-row SHA-256 does not match the sign-off record")
    required_tokens = (
        LICENSE_ROW_PREFIX,
        b"apache-2.0",
        b"2026-07-30 yousan",
        "☑ Commercial".encode("utf-8"),
    )
    for token in required_tokens:
        if token not in row:
            raise ValueError(f"CLAP audit row is missing required sign-off token: {token!r}")
    return row


def validate_document(document: dict[str, Any]) -> dict[str, Any]:
    root = require_exact_keys(
        document,
        {
            "schema",
            "candidate_status",
            "upstream",
            "evidence",
            "refresh",
            "license_signoff",
            "disposition",
            "approval",
            "integrity",
        },
        "candidate",
    )
    if root["schema"] != SCHEMA:
        raise ValueError("candidate schema drifted")
    if root["candidate_status"] != "PENDING_OWNER_REVIEW":
        raise ValueError("candidate status is not PENDING_OWNER_REVIEW")

    upstream = require_exact_keys(
        root["upstream"], {"repository", "source_repository", "revision"}, "upstream"
    )
    if upstream != {
        "repository": REPOSITORY,
        "source_repository": SOURCE_REPOSITORY,
        "revision": REVISION,
    }:
        raise ValueError("upstream identity drifted")

    evidence = require_exact_keys(
        root["evidence"],
        {
            "model_free_audit_sha256",
            "dependency_inventory_sha256",
            "summary_sha256",
            "status",
            "factual_findings",
            "review_flags",
            "distributions",
        },
        "evidence",
    )
    if evidence["status"] != "CURRENT_VAST_MODEL_FREE":
        raise ValueError("candidate evidence is not marked as current VAST model-free evidence")
    if evidence["factual_findings"] != DEPENDENCY_FACTUAL_FINDINGS:
        raise ValueError("dependency factual finding count drifted")
    if evidence["review_flags"] != DEPENDENCY_REVIEW_FLAGS:
        raise ValueError("dependency review flag count drifted")
    if evidence["distributions"] != DEPENDENCY_DISTRIBUTIONS:
        raise ValueError("dependency distribution count drifted")
    expected_evidence = {
        "model_free_audit_sha256": MODEL_FREE_AUDIT_SHA256,
        "dependency_inventory_sha256": DEPENDENCY_INVENTORY_SHA256,
        "summary_sha256": SUMMARY_SHA256,
    }
    for key, expected in expected_evidence.items():
        require_hex(evidence[key], 64, f"evidence.{key}")
        if evidence[key] != expected:
            raise ValueError(f"evidence.{key} does not match the VAST record")

    refresh = require_exact_keys(
        root["refresh"],
        {
            "reason",
            "previous_torch_version",
            "current_torch_version",
            "previous_lock_sha256",
            "current_lock_sha256",
            "status",
        },
        "refresh",
    )
    expected_refresh = {
        "reason": "TORCH_SECURITY_REFRESH",
        "previous_torch_version": "2.7.1+cpu",
        "current_torch_version": "2.13.0+cpu",
        "previous_lock_sha256": PREVIOUS_LOCK_SHA256,
        "current_lock_sha256": CURRENT_LOCK_SHA256,
        "status": "VAST_EVIDENCE_REGENERATED",
    }
    if refresh != expected_refresh:
        raise ValueError("candidate dependency refresh disposition drifted")
    current_lock = hashlib.sha256((PROJECT / "uv.lock").read_bytes()).hexdigest()
    if current_lock != CURRENT_LOCK_SHA256:
        raise ValueError("current CLAP uv.lock does not match refresh binding")

    if root["license_signoff"] != LICENSE_SIGNOFF:
        raise ValueError("model license sign-off citation drifted")
    read_license_audit_row()

    disposition = require_exact_keys(
        root["disposition"],
        {
            "model_license_status",
            "dependency_status",
            "runtime_status",
            "publication",
            "weights",
            "model_load",
            "model_forward",
            "owner_approval",
        },
        "disposition",
    )
    expected_disposition = {
        "model_license_status": "SIGNED_COMMERCIAL",
        "dependency_status": "PENDING_OWNER_REVIEW",
        "runtime_status": "BLOCKED",
        "publication": "NO_UPLOAD",
        "weights": "NOT_ACQUIRED",
        "model_load": "NOT_PERFORMED",
        "model_forward": "NOT_PERFORMED",
        "owner_approval": "PENDING_OWNER_APPROVAL",
    }
    if disposition != expected_disposition:
        raise ValueError("fail-closed disposition drifted")

    approval = require_exact_keys(root["approval"], {"status", "digest"}, "approval")
    if approval != {"status": "NOT_PROVIDED", "digest": None}:
        raise ValueError("candidate contains an approval or approval-like digest")

    integrity = require_exact_keys(
        root["integrity"], {"canonicalization", "payload_sha256"}, "integrity"
    )
    if integrity["canonicalization"] != CANONICALIZATION:
        raise ValueError("candidate canonicalization drifted")
    require_hex(integrity["payload_sha256"], 64, "integrity.payload_sha256")
    computed = payload_sha256(document)
    if integrity["payload_sha256"] != computed:
        raise ValueError("candidate payload SHA-256 does not match its content")
    return document


def self_test() -> None:
    assert len(REVISION) == 40
    assert all(char in "0123456789abcdef" for char in REVISION)
    candidate = read_document(CANDIDATE)
    validate_document(candidate)

    tampered = json.loads(json.dumps(candidate))
    tampered["evidence"]["summary_sha256"] = "0" * 64
    try:
        validate_document(tampered)
    except ValueError as exc:
        assert "summary_sha256" in str(exc)
    else:
        raise AssertionError("evidence hash tampering was accepted")

    tampered = json.loads(json.dumps(candidate))
    tampered["refresh"]["current_lock_sha256"] = "0" * 64
    try:
        validate_document(tampered)
    except ValueError as exc:
        assert "refresh" in str(exc) or "lock" in str(exc)
    else:
        raise AssertionError("stale lock refresh tampering was accepted")

    tampered = json.loads(json.dumps(candidate))
    tampered["schema"] = "vokra-clap-htsat-fused-owner-review-candidate-v0"
    try:
        validate_document(tampered)
    except ValueError as exc:
        assert "schema" in str(exc)
    else:
        raise AssertionError("schema tampering was accepted")

    tampered = json.loads(json.dumps(candidate))
    tampered["candidate_status"] = "APPROVED"
    try:
        validate_document(tampered)
    except ValueError as exc:
        assert "candidate status" in str(exc)
    else:
        raise AssertionError("status tampering was accepted")

    tampered = json.loads(json.dumps(candidate))
    tampered["license_signoff"]["decision"] = "REJECTED"
    try:
        validate_document(tampered)
    except ValueError as exc:
        assert "sign-off" in str(exc)
    else:
        raise AssertionError("license sign-off tampering was accepted")

    source = LICENSE_AUDIT_DOCUMENT.read_bytes()
    with tempfile.TemporaryDirectory(prefix="vokra-clap-candidate-") as temporary:
        root = Path(temporary)
        tampered_row = root / "tampered-license-audit.md"
        commercial_token = "☑ Commercial".encode("utf-8")
        source_row = next(
            line for line in source.splitlines(keepends=True)
            if line.startswith(LICENSE_ROW_PREFIX)
        )
        changed_row = source_row.replace(
            commercial_token, "☐ Commercial".encode("utf-8"), 1
        )
        tampered_row.write_bytes(source.replace(source_row, changed_row, 1))
        try:
            read_license_audit_row(tampered_row)
        except ValueError as exc:
            assert "SHA-256" in str(exc)
        else:
            raise AssertionError("license audit-row tampering was accepted")

        missing_row = root / "missing-license-audit.md"
        missing_row.write_bytes(b"\n".join(line for line in source.splitlines() if not line.startswith(LICENSE_ROW_PREFIX)) + b"\n")
        try:
            read_license_audit_row(missing_row)
        except ValueError as exc:
            assert "exactly one" in str(exc)
        else:
            raise AssertionError("missing license audit row was accepted")

        duplicate_row = root / "duplicate-license-audit.md"
        duplicate_row.write_bytes(source + next(line for line in source.splitlines(keepends=True) if line.startswith(LICENSE_ROW_PREFIX)))
        try:
            read_license_audit_row(duplicate_row)
        except ValueError as exc:
            assert "exactly one" in str(exc)
        else:
            raise AssertionError("duplicate license audit rows were accepted")

    tampered = json.loads(json.dumps(candidate))
    tampered["approval"] = {"status": "APPROVED", "digest": "0" * 64}
    try:
        validate_document(tampered)
    except ValueError as exc:
        assert "approval" in str(exc)
    else:
        raise AssertionError("approval digest was accepted")

    with tempfile.TemporaryDirectory(prefix="vokra-clap-candidate-") as temporary:
        duplicate = Path(temporary) / "duplicate.json"
        duplicate.write_text('{"schema": "one", "schema": "two"}\n', encoding="utf-8")
        try:
            read_document(duplicate)
        except ValueError as exc:
            assert "duplicate JSON key" in str(exc)
        else:
            raise AssertionError("duplicate JSON keys were accepted")

    normal = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--candidate", str(CANDIDATE)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert normal.returncode == 2, normal
    assert "PENDING_OWNER_REVIEW" in normal.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    args = parser.parse_args()
    if args.self_test:
        if args.candidate != CANDIDATE:
            parser.error("--self-test accepts no candidate path")
        self_test()
        print("clap owner-review candidate self-test: OK")
        return 0
    try:
        validate_document(read_document(args.candidate))
    except ValueError as exc:
        print(f"CLAP_OWNER_REVIEW_CANDIDATE BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(f"CLAP_OWNER_REVIEW_CANDIDATE PENDING_OWNER_REVIEW: {args.candidate}")
    # A valid candidate is still a blocked disposition. Exit 2 prevents
    # callers from treating owner-review evidence as execution approval.
    return 2


if __name__ == "__main__":
    sys.exit(main())
