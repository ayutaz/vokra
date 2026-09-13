#!/usr/bin/env -S uv run --no-project --offline --python 3.12 python
"""Cryptographically verified, fail-closed gate for Canary dependencies.

The dependency audit is factual evidence and remains blocked.  An owner must
provide a detached Ed25519 signature over an exact approval record and a
separately supplied public-key trust anchor.  This module only uses the Python
standard library plus the host ``ssh-keygen`` verifier; it never imports a model,
reads a checkpoint, or writes audit evidence.
"""

from __future__ import annotations

import argparse
import base64
from datetime import date
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any


SCHEMA = "vokra-canary-1b-dependency-approval-v1"
AUDIT_SCHEMA = "vokra-canary-1b-reference-dependency-audit-v1"
AUDIT_STATUS = "BLOCKED_UNREVIEWED_TRANSITIVE"
PUBLICATION = "NO_UPLOAD"
DECISION = "APPROVED"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
DATE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
SIGNER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@+-]{0,63}$")
PLACEHOLDERS = {
    "",
    "anonymous",
    "example",
    "none",
    "null",
    "owner-example",
    "owner_review_required",
    "pending",
    "pending-review",
    "pending_review",
    "review-required",
    "review_required",
    "self-test",
    "self_test",
    "test",
    "todo",
    "unresolved",
}
VARIANTS = {"canary-1b-flash": "canary-1b-flash", "canary-1b-v2": "canary-1b-v2"}
APPROVAL_KEYS = {
    "schema",
    "variant",
    "model",
    "expected_head",
    "candidate_owner_scope_sha256",
    "dependency_audit_report_sha256",
    "dependency_audit_schema",
    "dependency_audit_status",
    "publication",
    "no_upload",
    "decision",
    "signer",
    "signed_at",
    "signature_algorithm",
    "signer_key_sha256",
    "signature_base64",
}
MODEL_FREE_KEYS = (
    "weights_acquired",
    "source_acquired",
    "model_imported",
    "model_executed",
    "cargo_invoked",
    "upload",
)
COMPATIBILITY_BLOCKED = "BLOCKED_SECURITY_INCOMPATIBLE_CANARY_CLOSURE"
MIN_SAFE_LIGHTNING = (2, 6, 6)
KNOWN_ONELOGGER = (2, 3, 1)


class ApprovalError(ValueError):
    """Malformed, stale, unsigned, or unsafe approval input."""


class CompatibilityError(ValueError):
    """The locked Canary dependency closure is not an explicitly reviewed safe pair."""


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def version_tuple(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, str):
        raise CompatibilityError(f"{label} version is missing")
    match = re.fullmatch(r"(\d+(?:\.\d+)*)", value)
    if match is None:
        raise CompatibilityError(f"{label} version is malformed: {value!r}")
    return tuple(int(part) for part in match.group(1).split("."))


def compatibility_check(lock_path: Path) -> int:
    """Block model workers until an upstream-compatible secure closure exists."""
    try:
        with lock_path.open("rb") as stream:
            lock = tomllib.load(stream)
        rows = lock.get("package")
        if not isinstance(rows, list):
            raise CompatibilityError("uv lock package table is missing")
        packages = {
            row.get("name"): row.get("version")
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("name"), str)
        }
        lightning = version_tuple(packages.get("lightning"), "lightning")
        one_logger = version_tuple(
            packages.get("nv-one-logger-pytorch-lightning-integration"),
            "nv-one-logger-pytorch-lightning-integration",
        )
    except (OSError, tomllib.TOMLDecodeError, CompatibilityError) as error:
        print(f"{COMPATIBILITY_BLOCKED}: cannot establish locked closure: {error}", file=sys.stderr)
        return 2

    if lightning < MIN_SAFE_LIGHTNING:
        print(
            f"{COMPATIBILITY_BLOCKED}: lightning {'.'.join(map(str, lightning))} "
            f"is below security floor {'.'.join(map(str, MIN_SAFE_LIGHTNING))}",
            file=sys.stderr,
        )
        return 2
    if lightning == MIN_SAFE_LIGHTNING and one_logger == KNOWN_ONELOGGER:
        print(
            f"{COMPATIBILITY_BLOCKED}: lightning 2.6.6 is security-fixed but "
            "the released OneLogger 2.3.1 trainer override is incompatible at import time",
            file=sys.stderr,
        )
        return 2
    print(
        f"{COMPATIBILITY_BLOCKED}: no VAST-reviewed safe Lightning/OneLogger pair "
        f"is recorded (lightning={'.'.join(map(str, lightning))}, "
        f"one_logger={'.'.join(map(str, one_logger))})",
        file=sys.stderr,
    )
    return 2


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ApprovalError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_symlink_ancestry(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor or "/")
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ApprovalError(f"symlinked approval/audit path: {path}")


def regular_file(path: Path, label: str) -> Path:
    reject_symlink_ancestry(path)
    if not path.is_file() or path.is_symlink():
        raise ApprovalError(f"{label} is missing or not a regular file: {path}")
    return path


def regular_directory(path: Path, label: str) -> Path:
    reject_symlink_ancestry(path)
    if not path.is_dir() or path.is_symlink():
        raise ApprovalError(f"{label} is missing or symlinked: {path}")
    return path


def load_json(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    source = regular_file(path, label)
    try:
        payload = source.read_bytes()
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ApprovalError, TypeError) as error:
        raise ApprovalError(f"unreadable {label}: {error}") from error
    if not isinstance(value, dict):
        raise ApprovalError(f"{label} must be a JSON object")
    return payload, value


def reject_path_overlap(report_path: Path, approval_path: Path, key_path: Path, repo_root: Path) -> None:
    report = report_path.resolve(strict=True)
    approval = approval_path.resolve(strict=True)
    key = key_path.resolve(strict=True)
    repository = regular_directory(repo_root, "repository root").resolve(strict=True)
    paths = (report, approval, key)
    if len(set(paths)) != len(paths):
        raise ApprovalError("approval, audit report, and signer key paths overlap")
    for left in paths:
        for right in paths:
            if left != right and (left in right.parents or right in left.parents):
                raise ApprovalError("approval, audit report, and signer key paths overlap")
    if approval in repository.parents or approval == repository or repository in approval.parents:
        raise ApprovalError("approval must be external to the repository")
    if key in repository.parents or key == repository or repository in key.parents:
        raise ApprovalError("signer key must be external to the repository")
    if approval in report.parent.parents or approval == report.parent or report.parent in approval.parents:
        raise ApprovalError("approval must be outside the audit evidence directory")


def reviewed_signer(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if not SIGNER.fullmatch(value):
        return False
    normalized = re.sub(r"\s+", "_", value.strip().casefold())
    if not normalized or normalized in PLACEHOLDERS:
        return False
    if "example" in normalized or "self-test" in normalized or "self_test" in normalized:
        return False
    if re.search(r"(?:^|[-_])test(?:$|[-_])", normalized):
        return False
    return True


def valid_signed_at(value: Any) -> bool:
    if not isinstance(value, str) or not DATE.fullmatch(value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def model_free_environment(report: dict[str, Any]) -> dict[str, Any]:
    environment = report.get("environment")
    if not isinstance(environment, dict):
        raise ApprovalError("audit report environment is missing")
    result: dict[str, Any] = {}
    for key in MODEL_FREE_KEYS:
        if key not in environment:
            raise ApprovalError(f"audit report model-free field is missing: {key}")
        result[key] = environment[key]
    if any(result[key] is not False for key in MODEL_FREE_KEYS[:-1]) or result["upload"] != PUBLICATION:
        raise ApprovalError("audit report is not model-free/no-upload evidence")
    return result


def candidate_scope(report: dict[str, Any]) -> dict[str, Any]:
    required = (
        "package_rows",
        "package_facts",
        "dependency_paths",
        "inactive_lock_rows",
        "collector_failures",
        "project_sha256",
        "lock_sha256",
        "audit_script_sha256",
        "wrapper_sha256",
    )
    for key in required:
        if key not in report:
            raise ApprovalError(f"audit report scope field is missing: {key}")
    return {
        "active_rows": report["package_rows"],
        "active_facts": report["package_facts"],
        "dependency_paths": report["dependency_paths"],
        "inactive_rows": report["inactive_lock_rows"],
        "collector_failures": report["collector_failures"],
        "project_sha256": report["project_sha256"],
        "lock_sha256": report["lock_sha256"],
        "audit_script_sha256": report["audit_script_sha256"],
        "wrapper_sha256": report["wrapper_sha256"],
        "model_free": model_free_environment(report),
    }


def validate_report(report: dict[str, Any], expected_head: str, variant: str) -> str:
    if not HEX40.fullmatch(expected_head):
        raise ApprovalError("expected HEAD must be lowercase 40-hex")
    if report.get("schema") != AUDIT_SCHEMA:
        raise ApprovalError("wrong dependency audit schema")
    if report.get("status") != AUDIT_STATUS:
        raise ApprovalError("factual audit must remain BLOCKED_UNREVIEWED_TRANSITIVE")
    if report.get("publication") != PUBLICATION:
        raise ApprovalError("dependency audit publication is not NO_UPLOAD")
    if report.get("clean") is not True or report.get("head") != expected_head or report.get("expected_head") != expected_head:
        raise ApprovalError("dependency audit HEAD is stale or not clean")
    if not isinstance(report.get("collector_failures"), list) or report["collector_failures"]:
        raise ApprovalError("dependency audit has collector failures")
    inventory = report.get("distribution_inventory")
    if not isinstance(inventory, dict) or inventory.get("exact") is not True:
        raise ApprovalError("dependency audit distribution inventory is not exact")
    for key in ("project_sha256", "lock_sha256", "audit_script_sha256", "wrapper_sha256"):
        if not isinstance(report.get(key), str) or not HEX64.fullmatch(report[key]):
            raise ApprovalError(f"dependency audit digest is invalid: {key}")
    scope_digest = report.get("candidate_owner_scope_sha256")
    if not isinstance(scope_digest, str) or not HEX64.fullmatch(scope_digest):
        raise ApprovalError("candidate_owner_scope_sha256 is invalid")
    if digest(candidate_scope(report)) != scope_digest:
        raise ApprovalError("candidate owner scope digest does not match audit facts")
    if variant not in VARIANTS:
        raise ApprovalError(f"unknown Canary variant: {variant}")
    return scope_digest


def verify_signature(approval: dict[str, Any], key_path: Path) -> None:
    if approval.get("signature_algorithm") != "ssh-ed25519-v1":
        raise ApprovalError("dependency approval signature algorithm is not ssh-ed25519-v1")
    key = regular_file(key_path, "trusted signer key")
    if approval.get("signer_key_sha256") != sha256_file(key):
        raise ApprovalError("trusted signer key SHA-256 mismatch")
    encoded = approval.get("signature_base64")
    if not isinstance(encoded, str) or not encoded or len(encoded) > 8192:
        raise ApprovalError("dependency approval signature is missing")
    try:
        signature = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as error:
        raise ApprovalError("dependency approval signature is not valid base64") from error
    payload = canonical({key: value for key, value in approval.items() if key != "signature_base64"}).encode("utf-8")
    try:
        with tempfile.TemporaryDirectory(prefix="vokra-canary-signature-", dir="/private/tmp") as directory:
            root = Path(directory)
            payload_path = root / "payload"
            signature_path = root / "signature"
            allowed_signers_path = root / "allowed-signers"
            payload_path.write_bytes(payload)
            signature_path.write_bytes(signature)
            allowed_signers_path.write_text(f"{approval['signer']} {key.read_text(encoding='utf-8').strip()}\n", encoding="utf-8")
            result = subprocess.run(
                ["ssh-keygen", "-Y", "verify", "-f", str(allowed_signers_path), "-I", approval["signer"], "-n", "vokra-canary-dependency", "-s", str(signature_path)],
                capture_output=True,
                text=True,
                input=payload.decode("utf-8"),
                check=False,
                timeout=10,
            )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ApprovalError(f"Ed25519 signature verifier unavailable: {type(error).__name__}") from error
    if result.returncode != 0:
        raise ApprovalError("dependency approval signature verification failed")


def validate(*, report_path: Path, approval_path: Path, key_path: Path, repo_root: Path, expected_head: str, variant: str, approval_sha256: str) -> tuple[bool, str]:
    try:
        report_bytes, report = load_json(report_path, "dependency audit report")
        approval_bytes, approval = load_json(approval_path, "external dependency approval")
        if not HEX64.fullmatch(approval_sha256) or hashlib.sha256(approval_bytes).hexdigest() != approval_sha256:
            raise ApprovalError("external dependency approval SHA-256 mismatch")
        reject_path_overlap(report_path, approval_path, key_path, repo_root)
        scope_digest = validate_report(report, expected_head, variant)
        if set(approval) != APPROVAL_KEYS:
            raise ApprovalError("dependency approval schema is not exact")
        expected = {
            "schema": SCHEMA,
            "variant": variant,
            "model": VARIANTS[variant],
            "expected_head": expected_head,
            "candidate_owner_scope_sha256": scope_digest,
            "dependency_audit_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "dependency_audit_schema": AUDIT_SCHEMA,
            "dependency_audit_status": AUDIT_STATUS,
            "publication": PUBLICATION,
            "no_upload": True,
            "decision": DECISION,
        }
        for key, value in expected.items():
            if approval.get(key) != value:
                raise ApprovalError(f"dependency approval identity mismatch: {key}")
        if not reviewed_signer(approval.get("signer")):
            raise ApprovalError("dependency approval signer is missing or a placeholder")
        if not valid_signed_at(approval.get("signed_at")):
            raise ApprovalError("dependency approval signed_at is missing or invalid")
        verify_signature(approval, key_path)
        return True, "PASS"
    except (OSError, TypeError, ValueError) as error:
        return False, str(error)


def synthetic_report(head: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "status": AUDIT_STATUS,
        "publication": PUBLICATION,
        "clean": True,
        "head": head,
        "expected_head": head,
        "distribution_inventory": {"exact": True},
        "package_rows": [{"name": "demo", "version": "1", "source": {"registry": "https://pypi.org/simple"}}],
        "package_facts": [{"name": "demo", "version": "1", "fact_sha256": "a" * 64}],
        "dependency_paths": [{"name": "demo", "version": "1", "paths": [["project", "demo"]]}],
        "inactive_lock_rows": [],
        "collector_failures": [],
        "project_sha256": "b" * 64,
        "lock_sha256": "c" * 64,
        "audit_script_sha256": "d" * 64,
        "wrapper_sha256": "e" * 64,
        "environment": {"weights_acquired": False, "source_acquired": False, "model_imported": False, "model_executed": False, "cargo_invoked": False, "upload": PUBLICATION},
    }
    report["candidate_owner_scope_sha256"] = digest(candidate_scope(report))
    return report


def sign_for_self_test(approval: dict[str, Any], private_key: Path, payload_path: Path, signature_path: Path) -> None:
    payload_path.write_bytes(canonical({key: value for key, value in approval.items() if key != "signature_base64"}).encode("utf-8"))
    result = subprocess.run(["ssh-keygen", "-Y", "sign", "-f", str(private_key), "-n", "vokra-canary-dependency", str(payload_path)], capture_output=True, check=False, timeout=10)
    generated = Path(f"{payload_path}.sig")
    if result.returncode != 0 or not generated.is_file():
        raise RuntimeError("ssh-keygen self-test signing failed")
    generated.replace(signature_path)
    approval["signature_base64"] = base64.b64encode(signature_path.read_bytes()).decode("ascii")


def self_test() -> int:
    head = "a" * 40
    with tempfile.TemporaryDirectory(prefix="vokra-canary-dependency-approval-", dir="/private/tmp") as directory:
        root = Path(directory)
        repository, evidence, external = root / "repo", root / "audit", root / "owner"
        repository.mkdir(); evidence.mkdir(); external.mkdir()
        report_path, approval_path = evidence / "report.json", external / "approval.json"
        key_path, private_path = external / "owner.pub", external / "owner.key"
        result = subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(private_path)], capture_output=True, check=False, timeout=10)
        if result.returncode != 0:
            print("dependency approval self-test: ssh-keygen Ed25519 unavailable", file=sys.stderr)
            return 1
        result = subprocess.run(["ssh-keygen", "-y", "-f", str(private_path)], capture_output=True, text=True, check=False, timeout=10)
        if result.returncode != 0:
            print("dependency approval self-test: public-key export failed", file=sys.stderr)
            return 1
        key_path.write_text(result.stdout, encoding="utf-8")
        report = synthetic_report(head)
        report_bytes = (canonical(report) + "\n").encode("utf-8")
        report_path.write_bytes(report_bytes)
        approval = {
            "schema": SCHEMA, "variant": "canary-1b-flash", "model": VARIANTS["canary-1b-flash"], "expected_head": head,
            "candidate_owner_scope_sha256": report["candidate_owner_scope_sha256"], "dependency_audit_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "dependency_audit_schema": AUDIT_SCHEMA, "dependency_audit_status": AUDIT_STATUS, "publication": PUBLICATION, "no_upload": True, "decision": DECISION,
            "signer": "fixture-owner", "signed_at": "2026-09-13", "signature_algorithm": "ssh-ed25519-v1", "signer_key_sha256": sha256_file(key_path), "signature_base64": "",
        }
        payload_path, signature_path = root / "payload", root / "signature"
        sign_for_self_test(approval, private_path, payload_path, signature_path)
        approval_path.write_text(canonical(approval) + "\n", encoding="utf-8")
        kwargs = dict(report_path=report_path, approval_path=approval_path, key_path=key_path, repo_root=repository, expected_head=head, variant="canary-1b-flash", approval_sha256=sha256_file(approval_path))
        if not validate(**kwargs)[0]:
            print("dependency approval self-test: valid signature rejected", file=sys.stderr)
            return 1

        def rejected(label: str, mutate: Any) -> bool:
            candidate = json.loads(approval_path.read_text(encoding="utf-8")); mutate(candidate)
            candidate_path = external / f"{label}.json"; candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            return not validate(**{**kwargs, "approval_path": candidate_path, "approval_sha256": sha256_file(candidate_path)})[0]

        checks = {
            "scope": lambda value: value.update(candidate_owner_scope_sha256="0" * 64),
            "report-hash": lambda value: value.update(dependency_audit_report_sha256="0" * 64),
            "variant": lambda value: value.update(variant="canary-1b-v2"),
            "head": lambda value: value.update(expected_head="b" * 40),
            "upload": lambda value: value.update(no_upload=False),
            "publication": lambda value: value.update(publication="UPLOAD"),
            "signer": lambda value: value.update(signer="PENDING_REVIEW"),
            "signed-at": lambda value: value.update(signed_at=""),
            "signature": lambda value: value.update(signature_base64=base64.b64encode(b"bad").decode("ascii")),
            "extra-key": lambda value: value.update(extra="not-allowed"),
        }
        for label, mutate in checks.items():
            if not rejected(label, mutate):
                print(f"dependency approval accepted {label} tamper", file=sys.stderr); return 1
        duplicate = external / "duplicate.json"; duplicate.write_text('{"schema":"x","schema":"y"}', encoding="utf-8")
        if not validate(**{**kwargs, "approval_path": duplicate, "approval_sha256": sha256_file(duplicate)})[1].startswith("unreadable"):
            print("dependency approval duplicate-key rejection lost", file=sys.stderr); return 1
        link = external / "approval-link.json"; link.symlink_to(approval_path)
        if validate(**{**kwargs, "approval_path": link, "approval_sha256": sha256_file(approval_path)})[0]:
            print("dependency approval accepted symlink", file=sys.stderr); return 1
        if validate(**{**kwargs, "approval_path": report_path, "approval_sha256": sha256_file(report_path)})[0]:
            print("dependency approval accepted overlapping paths", file=sys.stderr); return 1
    parser = build_parser()
    invalid_args = parser.parse_args(["--self-test", "--lock", "unused.lock"])
    try:
        validate_mode_args(parser, invalid_args)
    except SystemExit as error:
        if error.code != 2:
            print("dependency approval CLI accepted an unexpected parser exit", file=sys.stderr)
            return 1
    else:
        print("dependency approval CLI accepted --self-test with --lock", file=sys.stderr)
        return 1
    print("canary_1b dependency approval gate self-test: PASS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--compatibility-check", action="store_true")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--audit-report", type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--trusted-signer-key", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--expected-head")
    parser.add_argument("--variant", choices=sorted(VARIANTS))
    parser.add_argument("--approval-sha256")
    return parser


def validate_mode_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.compatibility_check:
        normal_values = (
            args.audit_report,
            args.approval,
            args.trusted_signer_key,
            args.repo_root,
            args.expected_head,
            args.variant,
            args.approval_sha256,
        )
        if args.self_test or args.lock is None or any(value is not None for value in normal_values):
            parser.error("--compatibility-check requires only --lock")
    elif args.lock is not None:
        parser.error("--lock requires --compatibility-check")
    values = (args.audit_report, args.approval, args.trusted_signer_key, args.repo_root, args.expected_head, args.variant, args.approval_sha256)
    if args.self_test:
        if args.lock is not None or any(value is not None for value in values):
            parser.error("--self-test accepts no other arguments")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_mode_args(parser, args)
    if args.compatibility_check:
        return compatibility_check(args.lock)
    values = (args.audit_report, args.approval, args.trusted_signer_key, args.repo_root, args.expected_head, args.variant, args.approval_sha256)
    if args.self_test:
        return self_test()
    if any(value is None for value in values):
        parser.error("normal runs require audit-report, approval, trusted-signer-key, repo-root, expected-head, variant, and approval-sha256")
    ok, reason = validate(report_path=args.audit_report, approval_path=args.approval, key_path=args.trusted_signer_key, repo_root=args.repo_root, expected_head=args.expected_head, variant=args.variant, approval_sha256=args.approval_sha256)
    if ok:
        print("canary_1b dependency approval gate: PASS"); return 0
    print(f"canary_1b dependency approval gate: BLOCKED: {reason}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
