#!/usr/bin/env -S uv run --frozen --project tools/parity/dia_1_6b_reference python
"""Validate an external, hash-bound Dia dependency approval.

The dependency audit emits an unsigned ``owner-review-scope.json``.  This
module only validates a separately supplied approval against that exact scope;
it never creates, signs, or upgrades an approval and never touches model
inputs.  The scope remains a factual, fail-closed audit with ``NO_UPLOAD``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

SCOPE_SCHEMA = "vokra-dia-dependency-owner-scope-v1"
APPROVAL_SCHEMA = "vokra-dia-dependency-owner-approval-v1"
AUDIT_SCHEMA = "vokra-dia-dependency-audit-v1"
GATE = "BLOCKED_UNREVIEWED_TRANSITIVE"
PUBLICATION = "NO_UPLOAD"
SCOPE_STATUS = "PENDING_OWNER_REVIEW"
APPROVAL_STATUS = "APPROVED"
APPROVAL_DECISION = "APPROVE"
HEAD_RE = re.compile(r"[0-9a-f]{40}")
SHA_RE = re.compile(r"[0-9a-f]{64}")
PLACEHOLDER_RE = re.compile(r"(?:^|[\s_-])(todo|pending|owner|example|tbd|unknown|test)(?:$|[\s_-])")


class ApprovalError(RuntimeError):
    """An approval, scope, or evidence binding is invalid."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest_bytes(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                h.update(block)
    except (OSError, UnicodeError) as error:
        raise ApprovalError(f"cannot read evidence file: {path}") from error
    return h.hexdigest()


def parse_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError, ApprovalError) as error:
        raise ApprovalError(f"invalid {label} JSON: {path}") from error
    if not isinstance(value, dict):
        raise ApprovalError(f"{label} JSON root must be an object")
    return value


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ApprovalError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def require_external_regular(path: Path, label: str) -> None:
    if not path.is_absolute() or any(part in {".", ".."} for part in path.parts[1:]):
        raise ApprovalError(f"{label} path must be absolute and free of dot components")
    cursor = Path(path.anchor)
    for part in path.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ApprovalError(f"{label} path has symlink ancestry")
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ApprovalError(f"{label} must be a non-empty regular file")


def _hex(value: Any, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ApprovalError(f"{label} must be lowercase hexadecimal")
    return value


def _scope_evidence(scope: dict[str, Any]) -> dict[str, Any]:
    audit = scope.get("audit")
    if not isinstance(audit, dict):
        raise ApprovalError("scope audit evidence is missing")
    return {
        "package_evidence_sha256": audit.get("package_evidence_sha256"),
        "native_evidence_sha256": audit.get("native_evidence_sha256"),
        "publisher_evidence_sha256": audit.get("publisher_evidence_sha256"),
    }


def validate_scope(scope_path: Path, expected_head: str) -> tuple[dict[str, Any], str]:
    require_external_regular(scope_path, "dependency scope")
    scope = parse_json(scope_path, "dependency scope")
    if set(scope) != {"schema", "status", "dependency_license_audit", "publication", "owner_review", "audit", "native_inventory", "preparation", "scope_sha256"}:
        raise ApprovalError("dependency scope key set is not exact")
    if scope.get("schema") != SCOPE_SCHEMA or scope.get("status") != SCOPE_STATUS or scope.get("dependency_license_audit") != GATE or scope.get("publication") != PUBLICATION:
        raise ApprovalError("dependency scope is not the unsigned fail-closed scope")
    scope_sha = _hex(scope.get("scope_sha256"), SHA_RE, "scope_sha256")
    unsigned = dict(scope)
    del unsigned["scope_sha256"]
    if hashlib.sha256(canonical(unsigned)).hexdigest() != scope_sha:
        raise ApprovalError("dependency scope digest mismatch")
    audit = scope["audit"]
    if not isinstance(audit, dict) or audit.get("expected_head") != expected_head:
        raise ApprovalError("dependency scope HEAD mismatch")
    _hex(expected_head, HEAD_RE, "expected_head")
    if audit.get("package_count") != 26 or not isinstance(audit.get("package_identities"), list) or len(audit["package_identities"]) != 26 or len(set(audit["package_identities"])) != 26 or audit["package_identities"] != sorted(audit["package_identities"]):
        raise ApprovalError("dependency scope package evidence is not the exact 26-row closure")
    native = scope.get("native_inventory")
    if not isinstance(native, list) or audit.get("native_file_count") != len(native):
        raise ApprovalError("dependency scope native evidence is incomplete")
    native_keys: list[tuple[str, str]] = []
    for row in native:
        path = row.get("path") if isinstance(row, dict) else None
        if not isinstance(row, dict) or not isinstance(row.get("package_identity"), str) or row["package_identity"] not in set(audit["package_identities"]) or not isinstance(path, str) or not path or path.startswith("/") or "\\" in path or any(part in {"", ".", ".."} for part in path.split("/")) or not SHA_RE.fullmatch(str(row.get("sha256", ""))):
            raise ApprovalError("dependency scope native evidence row is malformed")
        native_keys.append((row["package_identity"], path))
    if len(native_keys) != len(set(native_keys)) or len({path for _, path in native_keys}) != len(native_keys):
        raise ApprovalError("dependency scope native evidence contains duplicate paths")
    evidence = _scope_evidence(scope)
    if any(not isinstance(value, str) or not SHA_RE.fullmatch(value) for value in evidence.values()):
        raise ApprovalError("dependency scope evidence digest is missing")
    if audit.get("failures") != [] or audit.get("publisher_bytes_recorded") != 50:
        raise ApprovalError("dependency scope publisher/failure evidence is incomplete")
    owner_review = scope["owner_review"]
    if not isinstance(owner_review, dict) or owner_review.get("decision") is not None or owner_review.get("approval_schema") != APPROVAL_SCHEMA:
        raise ApprovalError("dependency scope contains an owner decision or wrong approval schema")
    return scope, scope_sha


def validate_approval(scope_path: Path, approval_path: Path, approval_sha256: str, expected_head: str) -> dict[str, Any]:
    _hex(approval_sha256, SHA_RE, "approval_sha256")
    _hex(expected_head, HEAD_RE, "expected_head")
    scope, scope_sha = validate_scope(scope_path, expected_head)
    require_external_regular(approval_path, "dependency approval")
    if digest_bytes(approval_path) != approval_sha256:
        raise ApprovalError("dependency approval SHA-256 mismatch")
    approval = parse_json(approval_path, "dependency approval")
    required = {
        "schema", "status", "decision", "signer", "scope_sha256", "expected_head",
        "dependency_license_audit", "publication", "package_evidence_sha256",
        "native_evidence_sha256", "publisher_evidence_sha256",
    }
    if set(approval) != required:
        raise ApprovalError("dependency approval key set is not exact")
    if approval.get("schema") != APPROVAL_SCHEMA or approval.get("status") != APPROVAL_STATUS or approval.get("decision") != APPROVAL_DECISION:
        raise ApprovalError("dependency approval is not an explicit approval record")
    signer = approval.get("signer")
    if not isinstance(signer, str) or not signer.strip() or PLACEHOLDER_RE.search(signer.strip().lower()):
        raise ApprovalError("dependency approval signer is empty or a placeholder")
    if approval.get("scope_sha256") != scope_sha or approval.get("expected_head") != expected_head or approval.get("dependency_license_audit") != GATE or approval.get("publication") != PUBLICATION:
        raise ApprovalError("dependency approval identity is stale or not NO_UPLOAD")
    expected_evidence = _scope_evidence(scope)
    for key, value in expected_evidence.items():
        if approval.get(key) != value:
            raise ApprovalError(f"dependency approval {key} does not match generated scope")
    return approval


def _fixture_scope(head: str) -> dict[str, Any]:
    packages = sorted(f"p{i}==1" for i in range(26))
    native = [{"package_identity": packages[0], "path": "x.so", "sha256": "a" * 64}]
    publisher = {"packages": 26, "publisher_license_evidence_missing": [], "publisher_bytes_recorded": 50}
    audit = {
        "expected_head": head,
        "package_count": 26,
        "package_identities": packages,
        "native_file_count": len(native),
        "publisher_bytes_recorded": 50,
        "package_evidence_sha256": hashlib.sha256(canonical(packages)).hexdigest(),
        "native_evidence_sha256": hashlib.sha256(canonical(native)).hexdigest(),
        "publisher_evidence_sha256": hashlib.sha256(canonical(publisher)).hexdigest(),
        "failures": [],
    }
    value: dict[str, Any] = {
        "schema": SCOPE_SCHEMA,
        "status": SCOPE_STATUS,
        "dependency_license_audit": GATE,
        "publication": PUBLICATION,
        "owner_review": {"decision": None, "approval_schema": APPROVAL_SCHEMA},
        "audit": audit,
        "native_inventory": native,
        "preparation": {"evidence_sha256": "b" * 64},
    }
    value["scope_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return value


def self_test() -> int:
    import tempfile

    head = "a" * 40
    temp_parent = "/private/tmp" if Path("/private/tmp").is_dir() and not Path("/private/tmp").is_symlink() else None
    with tempfile.TemporaryDirectory(prefix="dia-dependency-approval-", dir=temp_parent) as directory:
        root = Path(directory)
        scope_path = root / "scope.json"
        scope = _fixture_scope(head)
        scope_path.write_bytes(canonical(scope) + b"\n")
        evidence = _scope_evidence(scope)
        approval = {
            "schema": APPROVAL_SCHEMA,
            "status": APPROVAL_STATUS,
            "decision": APPROVAL_DECISION,
            "signer": "external-signer",
            "scope_sha256": scope["scope_sha256"],
            "expected_head": head,
            "dependency_license_audit": GATE,
            "publication": PUBLICATION,
            **evidence,
        }
        approval_path = root / "approval.json"
        approval_path.write_bytes(canonical(approval) + b"\n")
        approval_sha = digest_bytes(approval_path)
        validate_approval(scope_path, approval_path, approval_sha, head)

        def expect_reject(label: str, mutate: Any) -> None:
            candidate = json.loads(approval_path.read_text(encoding="utf-8"))
            mutate(candidate)
            path = root / f"{label}.json"
            path.write_bytes(canonical(candidate) + b"\n")
            try:
                validate_approval(scope_path, path, digest_bytes(path), head)
            except ApprovalError:
                return
            raise AssertionError(f"tampered approval accepted: {label}")

        expect_reject("wrong-scope", lambda value: value.update(scope_sha256="0" * 64))
        expect_reject("wrong-head", lambda value: value.update(expected_head="b" * 40))
        expect_reject("wrong-native", lambda value: value.update(native_evidence_sha256="0" * 64))
        expect_reject("wrong-publication", lambda value: value.update(publication="UPLOAD"))
        expect_reject("placeholder", lambda value: value.update(signer="TODO owner"))
        duplicate = root / "duplicate.json"
        duplicate.write_text('{"schema":"x","schema":"y"}\n', encoding="utf-8")
        try:
            validate_approval(scope_path, duplicate, digest_bytes(duplicate), head)
        except ApprovalError:
            pass
        else:
            raise AssertionError("duplicate approval key accepted")
        scope_path.write_text(scope_path.read_text(encoding="utf-8").replace(scope["scope_sha256"], "0" * 64), encoding="utf-8")
        try:
            validate_approval(scope_path, approval_path, approval_sha, head)
        except ApprovalError:
            pass
        else:
            raise AssertionError("tampered scope accepted")
    print("dia dependency approval: self-test PASS (model-free success/tamper fail-closed)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--scope", type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--approval-sha256")
    parser.add_argument("--expected-head")
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.scope, args.approval, args.approval_sha256, args.expected_head)):
            parser.error("--self-test accepts no other arguments")
        return self_test()
    if None in (args.scope, args.approval, args.approval_sha256, args.expected_head):
        parser.error("--scope, --approval, --approval-sha256, and --expected-head are required")
    try:
        validate_approval(args.scope, args.approval, args.approval_sha256, args.expected_head)
    except ApprovalError as error:
        parser.error(str(error))
    print("Dia dependency approval validation: OK (scope-bound, NO_UPLOAD)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
