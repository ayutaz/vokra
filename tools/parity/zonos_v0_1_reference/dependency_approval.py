#!/usr/bin/env -S uv run --no-project --offline --python 3.12
"""Validate the external owner approval for the Zonos dependency scope.

The dependency approval is deliberately separate from the source/model
approval consumed by the Zonos worker.  This module only authorizes the
transition from the model-free dependency audit to the next pre-acquisition
stage.  It never downloads, imports, or executes a model or checkpoint.

The record is external to the checkout and is bound twice: the caller passes
the expected raw-file SHA-256, and the record contains a canonical
``sha256-canonical-approval-v1`` attestation over its unsigned fields.  The
audit report remains ``BLOCKED_UNREVIEWED_TRANSITIVE`` and ``NO_UPLOAD`` even
when this transition validates successfully.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any


SCHEMA = "vokra-zonos-dependency-approval-v1"
APPROVAL_KIND = "DEPENDENCY_SCOPE_ONLY"
APPROVAL_STATUS = "APPROVED_FOR_PRE_ACQUISITION"
DECISION = "ALLOW_SOURCE_CHECKPOINT_ACQUISITION"
AUDIT_SCHEMA = "vokra-zonos-dependency-audit-v1"
AUDIT_STATUS = "BLOCKED_UNREVIEWED_TRANSITIVE"
PUBLICATION = "NO_UPLOAD"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
DATE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")

APPROVAL_KEYS = {
    "schema",
    "approval_kind",
    "status",
    "publication",
    "decision",
    "owner",
    "signature",
    "audit",
}
OWNER_KEYS = {"handle", "role", "attested", "signed_at"}
SIGNATURE_KEYS = {"algorithm", "value"}
AUDIT_KEYS = {
    "report_sha256",
    "candidate_scope_sha256",
    "expected_head",
    "scope_digests",
}
SCOPE_DIGEST_KEYS = {
    "lock_rows_sha256",
    "preparation_sha256",
    "constraints_sha256",
    "installed_closure_sha256",
    "native_files_sha256",
    "publisher_files_sha256",
    "numpy_record_sha256",
    "numpy_native_policy_sha256",
    "numpy_runtime_config_sha256",
    "publisher_archive_manifest_sha256",
}
SCOPE_KEYS = {
    "schema",
    "lock_rows_sha256",
    "preparation_path",
    "preparation_sha256",
    "constraints_sha256",
    "sdist_identity",
    "wheel_identity",
    "installed_closure_sha256",
    "native_files_sha256",
    "publisher_files_sha256",
    "numpy_record_sha256",
    "numpy_native_policy_sha256",
    "numpy_runtime_config_sha256",
    "publisher_archive_manifest_sha256",
    "failures",
    "model_access",
    "source_access",
    "checkpoint_access",
    "publication",
    "execution_identity",
}
PLACEHOLDER_HANDLES = {
    "",
    "owner",
    "signer",
    "placeholder",
    "todo",
    "tbd",
    "unknown",
    "unresolved",
    "pending",
    "example",
    "example.invalid",
}


class ApprovalError(ValueError):
    """Raised for any fail-closed approval or audit mismatch."""


def canonical(value: Any) -> bytes:
    """Match the audit's canonical JSON digest contract exactly."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ApprovalError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ApprovalError(f"invalid JSON file: {path}") from error
    if not isinstance(value, dict):
        raise ApprovalError(f"JSON root is not an object: {path}")
    return value, raw


def _ancestor_has_symlink(path: Path) -> bool:
    current = path.absolute().parent
    while True:
        if current.is_symlink():
            return True
        if current == current.parent:
            return False
        current = current.parent


def _repository_root() -> Path:
    # dependency_approval.py -> zonos_v0_1_reference -> parity -> tools -> repo
    return Path(__file__).resolve().parents[3]


def require_external_file(path: Path, label: str, *, must_exist: bool = True) -> None:
    if not path.is_absolute() or path == Path(path.anchor):
        raise ApprovalError(f"{label} must be an absolute file path")
    if _ancestor_has_symlink(path) or path.is_symlink():
        raise ApprovalError(f"{label} has symlinked path components")
    if must_exist and (not path.is_file()):
        raise ApprovalError(f"{label} must be a regular file")
    if not must_exist and path.exists():
        raise ApprovalError(f"{label} must not already exist")
    root = _repository_root()
    try:
        path.relative_to(root)
    except ValueError:
        pass
    else:
        raise ApprovalError(f"{label} must be outside the checkout")


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.absolute()
    right = right.absolute()
    return left == right or left in right.parents or right in left.parents


def _require_hex(value: Any, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ApprovalError(f"{label} must be lowercase hexadecimal")
    return value


def _require_exact_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ApprovalError(f"{label} schema is not exact")
    return value


def _require_scope(report: dict[str, Any]) -> dict[str, Any]:
    if (
        report.get("schema") != AUDIT_SCHEMA
        or report.get("status") != AUDIT_STATUS
        or report.get("publication") != PUBLICATION
    ):
        raise ApprovalError("audit status/publication is not fail-closed")
    scope = _require_exact_keys(report.get("candidate_scope"), SCOPE_KEYS, "candidate scope")
    if scope.get("schema") != "vokra-zonos-dependency-approval-scope-v1":
        raise ApprovalError("candidate scope schema is not exact")
    if scope.get("publication") != PUBLICATION:
        raise ApprovalError("candidate scope publication is not NO_UPLOAD")
    if scope.get("failures") != []:
        raise ApprovalError("dependency audit contains collection failures")
    for key in ("model_access", "source_access", "checkpoint_access"):
        if scope.get(key) is not False:
            raise ApprovalError(f"candidate scope crosses acquisition boundary: {key}")
    execution = scope.get("execution_identity")
    report_execution = report.get("execution_identity")
    if not isinstance(execution, dict) or execution != report_execution:
        raise ApprovalError("candidate scope execution identity is not report-bound")
    if not HEX40.fullmatch(str(execution.get("expected_head", ""))):
        raise ApprovalError("candidate scope expected HEAD is malformed")
    if execution.get("actual_head") != execution.get("expected_head"):
        raise ApprovalError("candidate scope HEAD is not an exact clean identity")
    if not isinstance(report.get("installed"), dict) or report["installed"].get("status") != "COLLECTED":
        raise ApprovalError("installed dependency closure is not collected")
    for key in SCOPE_DIGEST_KEYS:
        _require_hex(scope.get(key), HEX64, f"candidate scope {key}")
    if report.get("candidate_scope_sha256") != hashlib.sha256(canonical(scope)).hexdigest():
        raise ApprovalError("candidate scope canonical digest mismatch")
    installed = report["installed"]
    digests = installed.get("digests")
    if not isinstance(digests, dict):
        raise ApprovalError("installed dependency digests are missing")
    for key in (
        "installed_closure_sha256",
        "native_files_sha256",
        "publisher_files_sha256",
        "numpy_record_sha256",
        "numpy_runtime_config_sha256",
    ):
        if digests.get(key) != scope[key]:
            raise ApprovalError(f"installed digest is not scope-bound: {key}")
    list_digests = {
        "installed_closure_sha256": installed.get("installed_distributions"),
        "native_files_sha256": installed.get("native_files"),
        "publisher_files_sha256": installed.get("publisher_license_notice_files"),
    }
    for key, rows in list_digests.items():
        if not isinstance(rows, list) or hashlib.sha256(canonical(rows)).hexdigest() != scope[key]:
            raise ApprovalError(f"installed list digest is not scope-bound: {key}")
    native_policy = installed.get("numpy_native_policy")
    if (
        not isinstance(native_policy, dict)
        or native_policy.get("schema") != "vokra-zonos-numpy-native-policy-v1"
        or native_policy.get("status") != "PASS_NO_FORBIDDEN_BLAS"
        or native_policy.get("forbidden_boundaries") != []
        or not isinstance(native_policy.get("native_file_count"), int)
        or native_policy["native_file_count"] < 1
    ):
        raise ApprovalError("NumPy no-forbidden-BLAS evidence is incomplete")
    if hashlib.sha256(canonical(native_policy)).hexdigest() != scope["numpy_native_policy_sha256"]:
        raise ApprovalError("NumPy native policy digest is not scope-bound")
    runtime_config = installed.get("numpy_runtime_config")
    if (
        not isinstance(runtime_config, dict)
        or runtime_config.get("schema") != "vokra-zonos-numpy-runtime-config-v1"
        or runtime_config.get("status") != "PASS_NO_FORBIDDEN_BLAS"
        or runtime_config.get("forbidden_boundaries") != []
    ):
        raise ApprovalError("NumPy runtime no-forbidden-BLAS evidence is incomplete")
    if hashlib.sha256(canonical(runtime_config)).hexdigest() != scope["numpy_runtime_config_sha256"]:
        raise ApprovalError("NumPy runtime config digest is not scope-bound")
    numpy_record = installed.get("numpy_record")
    if (
        not isinstance(numpy_record, dict)
        or numpy_record.get("schema") != "vokra-zonos-numpy-record-v2"
        or numpy_record.get("status") != "PASS"
        or not isinstance(numpy_record.get("rows"), list)
        or not isinstance(numpy_record.get("generated"), list)
    ):
        raise ApprovalError("prepared NumPy RECORD evidence is incomplete")
    if hashlib.sha256(canonical(numpy_record)).hexdigest() != scope["numpy_record_sha256"]:
        raise ApprovalError("prepared NumPy RECORD digest is not scope-bound")
    archive = installed.get("publisher_archive")
    if not isinstance(archive, dict) or not isinstance(archive.get("manifest_sha256"), str):
        raise ApprovalError("publisher archive manifest evidence is missing")
    if archive["manifest_sha256"] != scope["publisher_archive_manifest_sha256"]:
        raise ApprovalError("publisher archive digest is not scope-bound")
    publisher = installed.get("publisher_license_notice_files")
    native = installed.get("native_files")
    closure = installed.get("installed_distributions")
    if not all(isinstance(value, list) and value for value in (publisher, native, closure)):
        raise ApprovalError("installed/native/publisher facts are empty")
    return scope


def _validate_approval_record(
    approval: dict[str, Any],
    report: dict[str, Any],
    report_path: Path,
    expected_head: str,
) -> None:
    _require_exact_keys(approval, APPROVAL_KEYS, "dependency approval")
    if approval.get("schema") != SCHEMA:
        raise ApprovalError("approval schema is not dependency approval")
    if approval.get("approval_kind") != APPROVAL_KIND:
        raise ApprovalError("approval kind is not dependency-scope-only")
    if approval.get("status") != APPROVAL_STATUS or approval.get("publication") != PUBLICATION:
        raise ApprovalError("approval status/publication is not exact")
    if approval.get("decision") != DECISION:
        raise ApprovalError("approval decision is not pre-acquisition dependency approval")
    owner = _require_exact_keys(approval.get("owner"), OWNER_KEYS, "owner attestation")
    handle = owner.get("handle")
    if not isinstance(handle, str) or not handle.strip() or handle.strip().casefold() in PLACEHOLDER_HANDLES:
        raise ApprovalError("owner signer is missing or a placeholder")
    if owner.get("role") != "owner/legal" or owner.get("attested") is not True:
        raise ApprovalError("owner attestation is not explicit")
    if not isinstance(owner.get("signed_at"), str) or not DATE.fullmatch(owner["signed_at"]):
        raise ApprovalError("owner signed_at must be an ISO date")
    signature = _require_exact_keys(approval.get("signature"), SIGNATURE_KEYS, "approval signature")
    if signature.get("algorithm") != "sha256-canonical-approval-v1":
        raise ApprovalError("approval signature algorithm is not exact")
    signature_value = _require_hex(signature.get("value"), HEX64, "approval signature")
    unsigned = dict(approval)
    unsigned.pop("signature")
    if hashlib.sha256(canonical(unsigned)).hexdigest() != signature_value:
        raise ApprovalError("approval signature does not authenticate the record")
    audit = _require_exact_keys(approval.get("audit"), AUDIT_KEYS, "approval audit binding")
    report_digest = _require_hex(audit.get("report_sha256"), HEX64, "approval report digest")
    candidate_digest = _require_hex(audit.get("candidate_scope_sha256"), HEX64, "approval candidate scope digest")
    approval_head = _require_hex(audit.get("expected_head"), HEX40, "approval expected HEAD")
    scope_digests = _require_exact_keys(audit.get("scope_digests"), SCOPE_DIGEST_KEYS, "approval scope digests")
    for key, value in scope_digests.items():
        _require_hex(value, HEX64, f"approval scope digest {key}")
    scope = _require_scope(report)
    if report_digest != digest_bytes(report_path):
        raise ApprovalError("approval report SHA-256 does not match the current audit")
    if candidate_digest != report.get("candidate_scope_sha256"):
        raise ApprovalError("approval candidate scope digest is stale")
    if approval_head != expected_head or expected_head != scope["execution_identity"]["expected_head"]:
        raise ApprovalError("approval expected HEAD does not match the audit")
    if scope_digests != {key: scope[key] for key in sorted(SCOPE_DIGEST_KEYS)}:
        raise ApprovalError("approval dependency/native/NumPy/archive digests are stale")


def validate(
    approval_path: Path,
    audit_path: Path,
    expected_head: str,
    approval_sha256: str,
) -> str:
    """Validate one external approval and return its raw-file digest."""
    if not HEX40.fullmatch(expected_head):
        raise ApprovalError("expected HEAD must be lowercase 40-hex")
    _require_hex(approval_sha256, HEX64, "caller approval SHA-256")
    require_external_file(approval_path, "dependency approval")
    require_external_file(audit_path, "dependency audit")
    if _paths_overlap(approval_path, audit_path):
        raise ApprovalError("approval and audit paths overlap")
    approval, raw = read_json(approval_path)
    raw_digest = hashlib.sha256(raw).hexdigest()
    if raw_digest != approval_sha256:
        raise ApprovalError("caller approval SHA-256 differs from the record")
    report, _ = read_json(audit_path)
    _validate_approval_record(approval, report, audit_path, expected_head)
    return raw_digest


def _signed_record(report: dict[str, Any], report_path: Path, handle: str = "yousan") -> dict[str, Any]:
    """Build only an in-memory self-test record; never write an owner decision."""
    scope = report["candidate_scope"]
    scope_digests = {key: scope[key] for key in sorted(SCOPE_DIGEST_KEYS)}
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "approval_kind": APPROVAL_KIND,
        "status": APPROVAL_STATUS,
        "publication": PUBLICATION,
        "decision": DECISION,
        "owner": {"handle": handle, "role": "owner/legal", "attested": True, "signed_at": "2026-09-13"},
        "signature": {"algorithm": "sha256-canonical-approval-v1", "value": ""},
        "audit": {
            "report_sha256": digest_bytes(report_path),
            "candidate_scope_sha256": report["candidate_scope_sha256"],
            "expected_head": scope["execution_identity"]["expected_head"],
            "scope_digests": scope_digests,
        },
    }
    unsigned = dict(record)
    unsigned.pop("signature")
    record["signature"] = {
        "algorithm": "sha256-canonical-approval-v1",
        "value": hashlib.sha256(canonical(unsigned)).hexdigest(),
    }
    return record


def self_test_temp_root() -> str:
    """Resolve a platform-native temporary root without a macOS-only path."""
    return str(Path(os.environ.get("TMPDIR") or tempfile.gettempdir()).resolve())


def _self_test_report(root: Path) -> tuple[dict[str, Any], Path]:
    """Create a tiny synthetic report with all protected scope boundaries."""
    head = "a" * 40
    empty_digest = hashlib.sha256(b"[]").hexdigest()
    native_policy = {
        "schema": "vokra-zonos-numpy-native-policy-v1",
        "status": "PASS_NO_FORBIDDEN_BLAS",
        "native_file_count": 1,
        "forbidden_boundaries": [],
    }
    runtime_config = {
        "schema": "vokra-zonos-numpy-runtime-config-v1",
        "status": "PASS_NO_FORBIDDEN_BLAS",
        "config": {},
        "forbidden_boundaries": [],
    }
    numpy_record = {
        "schema": "vokra-zonos-numpy-record-v2",
        "status": "PASS",
        "rows": [{"path": "numpy/core.py", "bytes": 1, "sha256": "b" * 64, "generated": False}],
        "generated": [{"path": "numpy-2.2.2.dist-info/REQUESTED", "bytes": 1, "sha256": "c" * 64, "generated": True}],
    }
    execution = {
        "expected_head": head,
        "actual_head": head,
        "dependency_audit_sha256": "d" * 64,
        "wrapper_sha256": "e" * 64,
        "preparer_sha256": "f" * 64,
        "pyproject_sha256": "1" * 64,
        "uv_lock_sha256": "2" * 64,
        "constraints_sha256": "3" * 64,
        "platform": {"system": "Linux", "machine": "x86_64", "sys_platform": "linux", "python": "3.12.0"},
    }
    installed_rows = [{"name": "numpy"}]
    native_rows = [{"sha256": "b" * 64}]
    publisher_rows = [{"sha256": "a" * 64}]
    scope: dict[str, Any] = {
        "schema": "vokra-zonos-dependency-approval-scope-v1",
        "lock_rows_sha256": "4" * 64,
        "preparation_path": str(root / "preparation.json"),
        "preparation_sha256": "5" * 64,
        "constraints_sha256": "6" * 64,
        "sdist_identity": {"sha256": "7" * 64, "bytes": 1},
        "wheel_identity": {"sha256": "8" * 64, "bytes": 1},
        "installed_closure_sha256": hashlib.sha256(canonical(installed_rows)).hexdigest(),
        "native_files_sha256": hashlib.sha256(canonical(native_rows)).hexdigest(),
        "publisher_files_sha256": hashlib.sha256(canonical(publisher_rows)).hexdigest(),
        "numpy_record_sha256": hashlib.sha256(canonical(numpy_record)).hexdigest(),
        "numpy_native_policy_sha256": hashlib.sha256(canonical(native_policy)).hexdigest(),
        "numpy_runtime_config_sha256": hashlib.sha256(canonical(runtime_config)).hexdigest(),
        "publisher_archive_manifest_sha256": "9" * 64,
        "failures": [],
        "model_access": False,
        "source_access": False,
        "checkpoint_access": False,
        "publication": PUBLICATION,
        "execution_identity": execution,
    }
    report = {
        "schema": AUDIT_SCHEMA,
        "status": AUDIT_STATUS,
        "publication": PUBLICATION,
        "execution_identity": execution,
        "candidate_scope": scope,
        "candidate_scope_sha256": hashlib.sha256(canonical(scope)).hexdigest(),
        "installed": {
            "status": "COLLECTED",
            "digests": {
                "installed_closure_sha256": hashlib.sha256(canonical(installed_rows)).hexdigest(),
                "native_files_sha256": hashlib.sha256(canonical(native_rows)).hexdigest(),
                "publisher_files_sha256": hashlib.sha256(canonical(publisher_rows)).hexdigest(),
                "numpy_record_sha256": hashlib.sha256(canonical(numpy_record)).hexdigest(),
                "numpy_runtime_config_sha256": hashlib.sha256(canonical(runtime_config)).hexdigest(),
            },
            "numpy_native_policy": native_policy,
            "numpy_runtime_config": runtime_config,
            "numpy_record": numpy_record,
            "publisher_archive": {"manifest_sha256": "9" * 64},
            "publisher_license_notice_files": publisher_rows,
            "native_files": native_rows,
            "installed_distributions": installed_rows,
        },
    }
    report_path = root / "audit.json"
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    return report, report_path


def self_test() -> int:
    saved_tmpdir = os.environ.pop("TMPDIR", None)
    try:
        default_temp_root = self_test_temp_root()
        if default_temp_root != str(Path(tempfile.gettempdir()).resolve()):
            raise AssertionError("default temporary root resolution drifted")
    finally:
        if saved_tmpdir is not None:
            os.environ["TMPDIR"] = saved_tmpdir
    with tempfile.TemporaryDirectory(prefix="vokra-zonos-temp-root-") as configured_root:
        saved_tmpdir = os.environ.get("TMPDIR")
        os.environ["TMPDIR"] = configured_root
        try:
            configured_temp_root = self_test_temp_root()
            if configured_temp_root != str(Path(configured_root).resolve()):
                raise AssertionError("TMPDIR temporary root resolution drifted")
        finally:
            if saved_tmpdir is None:
                os.environ.pop("TMPDIR", None)
            else:
                os.environ["TMPDIR"] = saved_tmpdir
    temporary_root = self_test_temp_root()
    with tempfile.TemporaryDirectory(
        prefix="vokra-zonos-dependency-approval-", dir=temporary_root
    ) as directory:
        root = Path(directory)
        report, report_path = _self_test_report(root)
        approval_path = root / "approval.json"
        approval = _signed_record(report, report_path)
        approval_path.write_text(json.dumps(approval, sort_keys=True) + "\n", encoding="utf-8")
        approval_sha = digest_bytes(approval_path)
        validate(approval_path, report_path, "a" * 40, approval_sha)
        # Re-signing a changed scope cannot defeat the caller's raw-file hash.
        tampered = json.loads(approval_path.read_text(encoding="utf-8"))
        tampered["audit"]["scope_digests"]["preparation_sha256"] = "0" * 64
        tampered_path = root / "tampered-scope.json"
        tampered_path.write_text(json.dumps(tampered, sort_keys=True) + "\n", encoding="utf-8")
        try:
            validate(tampered_path, report_path, "a" * 40, approval_sha)
        except ApprovalError:
            pass
        else:
            raise AssertionError("tampered scope approval accepted")
        # The validator must reject a stale candidate scope even if the record
        # is freshly re-signed.
        stale = _signed_record(report, report_path)
        stale["audit"]["candidate_scope_sha256"] = "0" * 64
        unsigned = dict(stale)
        unsigned.pop("signature")
        stale["signature"] = {
            "algorithm": "sha256-canonical-approval-v1",
            "value": hashlib.sha256(canonical(unsigned)).hexdigest(),
        }
        stale_path = root / "stale.json"
        stale_path.write_text(json.dumps(stale, sort_keys=True) + "\n", encoding="utf-8")
        try:
            validate(stale_path, report_path, "a" * 40, digest_bytes(stale_path))
        except ApprovalError:
            pass
        else:
            raise AssertionError("stale candidate scope accepted")
        for label, mutation in (
            ("duplicate", '{"schema":"x","schema":"y"}'),
            ("placeholder", None),
            ("wrong-head", None),
            ("no-upload", None),
        ):
            if label == "duplicate":
                duplicate = root / "duplicate.json"
                duplicate.write_text(mutation, encoding="utf-8")
                try:
                    read_json(duplicate)
                except ApprovalError:
                    pass
                else:
                    raise AssertionError("duplicate approval key accepted")
                continue
            candidate = json.loads(approval_path.read_text(encoding="utf-8"))
            if label == "placeholder":
                candidate["owner"]["handle"] = "TODO"
            elif label == "wrong-head":
                candidate["audit"]["expected_head"] = "b" * 40
            else:
                candidate["publication"] = "UPLOAD"
            path = root / f"{label}.json"
            path.write_text(json.dumps(candidate, sort_keys=True) + "\n", encoding="utf-8")
            try:
                validate(path, report_path, "a" * 40, digest_bytes(path))
            except ApprovalError:
                pass
            else:
                raise AssertionError(f"{label} approval accepted")
        symlink = root / "symlink.json"
        symlink.symlink_to(approval_path)
        try:
            validate(symlink, report_path, "a" * 40, approval_sha)
        except ApprovalError:
            pass
        else:
            raise AssertionError("symlink approval accepted")
        symlink_dir = root / "symlink-dir"
        symlink_dir.symlink_to(root, target_is_directory=True)
        try:
            validate(symlink_dir / "approval.json", report_path, "a" * 40, approval_sha)
        except ApprovalError:
            pass
        else:
            raise AssertionError("symlinked approval parent accepted")
        try:
            validate(report_path, report_path, "a" * 40, digest_bytes(report_path))
        except ApprovalError:
            pass
        else:
            raise AssertionError("overlapping approval/audit paths accepted")
    print("zonos dependency approval: self-test PASS (model-free, hash-bound, NO_UPLOAD)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--approval-sha256")
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--expected-head")
    args = parser.parse_args(argv)
    if args.self_test:
        if any(value is not None for value in (args.approval, args.approval_sha256, args.audit, args.expected_head)):
            parser.error("--self-test accepts no other options")
        return self_test()
    if None in (args.approval, args.approval_sha256, args.audit, args.expected_head):
        parser.error("--approval, --approval-sha256, --audit, and --expected-head are required")
    try:
        digest = validate(args.approval, args.audit, args.expected_head, args.approval_sha256)
    except (OSError, ApprovalError, ValueError) as error:
        print(f"zonos dependency approval: BLOCKED: {error}", file=sys.stderr)
        return 2
    print(f"zonos dependency approval: PASS (raw_sha256={digest}; audit remains {AUDIT_STATUS}; publication={PUBLICATION})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
