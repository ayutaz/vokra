#!/usr/bin/env python3
"""Offline, fail-closed approval gate for ReazonSpeech NeMo v2.

This gate intentionally imports no model, NeMo, torch, or network package.
It authenticates the exact source/archive identity, the locked parity
environment, and an external no-upload approval before a worker is allowed to
probe a host, create scratch space, download a checkpoint, or invoke Cargo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any


SCHEMA = "vokra-reazonspeech-nemo-v2-preflight-v1"
APPROVAL_SCHEMA = "vokra-reazonspeech-nemo-v2-approval-v1"
MODEL = "reazonspeech-nemo-v2"
UPSTREAM_REPO = "reazon-research/reazonspeech-nemo-v2"
UPSTREAM_REVISION = "33693408be76b7cba9fd4a7546a0a8772430211b"
ARCHIVE_BYTES = 2_477_946_880
ARCHIVE_SHA256 = "d196d43ad03466ca88beeda4bf5fafb07bab7202d4b663b8e4f12cb0a4381fae"
LICENSE_SPDX = "Apache-2.0"
REFERENCE_IMPLEMENTATION = "nemo.collections.asr.models.EncDecRNNTBPEModel.restore_from"
VAST_ROUTE = "VAST-only conversion and official-NeMo reference; no upload"
APPLE_ROUTE = "Darwin arm64 CPU and Metal parity; staged inputs only"
PROJECT_RELATIVE = "tools/parity/pyproject.toml"
LOCK_RELATIVE = "tools/parity/uv.lock"
PLACEHOLDERS = {
    "",
    "none",
    "null",
    "pending",
    "pending_external",
    "pending_review",
    "owner_signoff_required",
    "owner_review_required",
    "review_required",
    "todo",
    "unresolved",
    "self-test",
    "self_test",
    "test",
    "example",
    "approval_required",
    "operator_approval_required",
}
MANIFEST_KEYS = {
    "schema",
    "model",
    "upstream_repo",
    "upstream_revision",
    "archive_bytes",
    "archive_sha256",
    "license_spdx",
    "no_upload",
    "dependency_project",
    "dependency_lock",
    "reference_implementation",
    "vast_route",
    "apple_route",
    "operator_approval",
    "approval_scope_sha256",
}
APPROVAL_KEYS = {
    "schema",
    "model",
    "upstream_repo",
    "upstream_revision",
    "archive_bytes",
    "archive_sha256",
    "license_spdx",
    "project_sha256",
    "lock_sha256",
    "manifest_sha256",
    "no_upload",
    "decision",
    "signer",
    "scope_sha256",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> str:
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)


def regular_file(path: Path) -> bool:
    """Require a regular file and reject symlinked ancestors as well."""
    try:
        absolute = Path(os.path.abspath(path))
        # macOS exposes the normal temporary directory through the system
        # symlinks /var (and, on some hosts, /tmp).  Those fixed OS aliases do
        # not make an approval path attacker-controlled; every other
        # symlinked ancestor remains forbidden.
        fixed_os_aliases = {Path("/var"), Path("/tmp")}
        return path.is_file() and not path.is_symlink() and all(
            not parent.is_symlink() or parent in fixed_os_aliases
            for parent in absolute.parents
        )
    except OSError:
        return False


def blocked(reason: str) -> tuple[bool, str]:
    return False, reason


def approval_scope(manifest: dict[str, Any], project_sha: str, lock_sha: str) -> dict[str, Any]:
    """Return the canonical, approval-bound facts (never the approval itself)."""
    return {
        "schema": manifest["schema"],
        "model": manifest["model"],
        "upstream_repo": manifest["upstream_repo"],
        "upstream_revision": manifest["upstream_revision"],
        "archive_bytes": manifest["archive_bytes"],
        "archive_sha256": manifest["archive_sha256"],
        "license_spdx": manifest["license_spdx"],
        "no_upload": manifest["no_upload"],
        "dependency_project": manifest["dependency_project"],
        "dependency_lock": manifest["dependency_lock"],
        "project_sha256": project_sha,
        "lock_sha256": lock_sha,
        "reference_implementation": manifest["reference_implementation"],
        "vast_route": manifest["vast_route"],
        "apple_route": manifest["apple_route"],
    }


def validate_manifest(manifest: Any) -> None:
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
        raise ValueError("preflight manifest schema is not exact")
    exact = {
        "schema": SCHEMA,
        "model": MODEL,
        "upstream_repo": UPSTREAM_REPO,
        "upstream_revision": UPSTREAM_REVISION,
        "archive_bytes": ARCHIVE_BYTES,
        "archive_sha256": ARCHIVE_SHA256,
        "license_spdx": LICENSE_SPDX,
        "no_upload": True,
        "dependency_project": PROJECT_RELATIVE,
        "dependency_lock": LOCK_RELATIVE,
        "reference_implementation": REFERENCE_IMPLEMENTATION,
        "vast_route": VAST_ROUTE,
        "apple_route": APPLE_ROUTE,
        "operator_approval": "PENDING_EXTERNAL",
    }
    for key, expected in exact.items():
        value = manifest.get(key)
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"manifest identity/status mismatch: {key}")
    if not isinstance(manifest["approval_scope_sha256"], str) or len(manifest["approval_scope_sha256"]) != 64:
        raise ValueError("manifest approval scope is malformed")


def validate_approval(
    approval: Any,
    manifest: dict[str, Any],
    manifest_sha: str,
    project_sha: str,
    lock_sha: str,
    *,
    allow_self_test: bool = False,
) -> None:
    if not isinstance(approval, dict) or set(approval) != APPROVAL_KEYS:
        raise ValueError("external approval schema is not exact")
    scope = approval_scope(manifest, project_sha, lock_sha)
    scope_sha = canonical(scope)
    if manifest["approval_scope_sha256"] != scope_sha:
        raise ValueError("manifest approval scope is not bound to exact inputs")
    expected = {
        "schema": APPROVAL_SCHEMA,
        "model": MODEL,
        "upstream_repo": UPSTREAM_REPO,
        "upstream_revision": UPSTREAM_REVISION,
        "archive_bytes": ARCHIVE_BYTES,
        "archive_sha256": ARCHIVE_SHA256,
        "license_spdx": LICENSE_SPDX,
        "project_sha256": project_sha,
        "lock_sha256": lock_sha,
        "manifest_sha256": manifest_sha,
        "no_upload": True,
        "decision": "APPROVED",
        "scope_sha256": scope_sha,
    }
    for key, expected_value in expected.items():
        value = approval.get(key)
        if type(value) is not type(expected_value) or value != expected_value:
            raise ValueError(f"external approval fact mismatch: {key}")
    signer = approval.get("signer")
    if not isinstance(signer, str) or not signer.strip():
        raise ValueError("external approval signer is missing")
    normalized = re.sub(r"[^a-z0-9]+", "_", signer.strip().casefold()).strip("_")
    if normalized in PLACEHOLDERS and not (allow_self_test and normalized == "self_test"):
        raise ValueError("external approval signer is a placeholder")


def validate(
    manifest_path: Path,
    approval_path: Path | None,
    project_path: Path,
    lock_path: Path,
    *,
    allow_self_test: bool = False,
) -> tuple[bool, str]:
    for path, label in (
        (manifest_path, "preflight manifest"),
        (project_path, "parity project"),
        (lock_path, "parity lock"),
    ):
        if not regular_file(path):
            return blocked(f"{label} is missing, symlinked, or not regular")
    if approval_path is None or not regular_file(approval_path):
        return blocked("external approval is missing, symlinked, or not regular")
    try:
        manifest_bytes = manifest_path.read_bytes()
        project_bytes = project_path.read_bytes()
        lock_bytes = lock_path.read_bytes()
        manifest = load_json(manifest_path)
        approval = load_json(approval_path)
        validate_manifest(manifest)
        validate_approval(
            approval,
            manifest,
            digest(manifest_bytes),
            digest(project_bytes),
            digest(lock_bytes),
            allow_self_test=allow_self_test,
        )
    except (OSError, UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError) as error:
        return blocked(str(error))
    return True, "PASS"


def self_test() -> int:
    """Exercise approval, duplicate, tamper, and missing-input fences only."""
    root = Path(__file__).resolve().parents[3]
    manifest_path = Path(__file__).resolve().with_name("license_gate_manifest.json")
    project = root / PROJECT_RELATIVE
    lock = root / LOCK_RELATIVE
    ok, reason = validate(manifest_path, None, project, lock)
    if ok or "external approval is missing" not in reason:
        print(f"reazonspeech-nemo-v2 preflight gate: expected pending approval, got {reason}")
        return 1
    with tempfile.TemporaryDirectory(prefix="reazonspeech-nemo-v2-preflight-") as directory:
        temp = Path(directory)
        temp_project = temp / "pyproject.toml"
        temp_lock = temp / "uv.lock"
        shutil.copy2(project, temp_project)
        shutil.copy2(lock, temp_lock)
        temp_manifest = temp / "manifest.json"
        shutil.copy2(manifest_path, temp_manifest)
        manifest = load_json(temp_manifest)
        scope_sha = canonical(
            approval_scope(manifest, digest(temp_project.read_bytes()), digest(temp_lock.read_bytes()))
        )
        if manifest["approval_scope_sha256"] != scope_sha:
            print("reazonspeech-nemo-v2 preflight gate: committed scope digest is stale")
            return 1
        approval = {
            "schema": APPROVAL_SCHEMA,
            "model": MODEL,
            "upstream_repo": UPSTREAM_REPO,
            "upstream_revision": UPSTREAM_REVISION,
            "archive_bytes": ARCHIVE_BYTES,
            "archive_sha256": ARCHIVE_SHA256,
            "license_spdx": LICENSE_SPDX,
            "project_sha256": digest(temp_project.read_bytes()),
            "lock_sha256": digest(temp_lock.read_bytes()),
            "manifest_sha256": digest(temp_manifest.read_bytes()),
            "no_upload": True,
            "decision": "APPROVED",
            "signer": "self-test",
            "scope_sha256": scope_sha,
        }
        approval_path = temp / "approval.json"
        approval_path.write_text(json.dumps(approval), encoding="utf-8")
        ok, reason = validate(temp_manifest, approval_path, temp_project, temp_lock, allow_self_test=True)
        if not ok:
            print(f"reazonspeech-nemo-v2 preflight gate: approved self-test failed: {reason}")
            return 1
        if validate(temp_manifest, approval_path, temp_project, temp_lock)[0]:
            print("reazonspeech-nemo-v2 preflight gate: self-test signer accepted in production mode")
            return 1
        duplicate = temp / "duplicate.json"
        duplicate.write_text('{"schema":"x","schema":"y"}', encoding="utf-8")
        if validate(temp_manifest, duplicate, temp_project, temp_lock)[0]:
            print("reazonspeech-nemo-v2 preflight gate: duplicate approval accepted")
            return 1
        tampered = dict(approval)
        tampered["archive_sha256"] = "0" * 64
        tampered_path = temp / "tampered.json"
        tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
        if validate(temp_manifest, tampered_path, temp_project, temp_lock, allow_self_test=True)[0]:
            print("reazonspeech-nemo-v2 preflight gate: tampered approval accepted")
            return 1
        missing = temp / "missing.json"
        if validate(temp_manifest, missing, temp_project, temp_lock)[0]:
            print("reazonspeech-nemo-v2 preflight gate: missing approval accepted")
            return 1
        symlink = temp / "symlink.json"
        symlink.symlink_to(approval_path)
        if validate(temp_manifest, symlink, temp_project, temp_lock, allow_self_test=True)[0]:
            print("reazonspeech-nemo-v2 preflight gate: symlink approval accepted")
            return 1
        for placeholder in ("TODO", "OWNER_SIGNOFF_REQUIRED", "pending_review"):
            placeholder_approval = dict(approval)
            placeholder_approval["signer"] = placeholder
            placeholder_path = temp / "placeholder.json"
            placeholder_path.write_text(json.dumps(placeholder_approval), encoding="utf-8")
            if validate(temp_manifest, placeholder_path, temp_project, temp_lock, allow_self_test=True)[0]:
                print(f"reazonspeech-nemo-v2 preflight gate: placeholder signer accepted: {placeholder}")
                return 1
    print("reazonspeech-nemo-v2 preflight gate: self-test PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--approval-evidence", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.manifest, args.approval_evidence, args.project, args.lock)):
            parser.error("--self-test accepts no other arguments")
        return self_test()
    if any(value is None for value in (args.manifest, args.approval_evidence, args.project, args.lock)):
        parser.error("normal runs require --manifest, --approval-evidence, --project, and --lock")
    passed, reason = validate(args.manifest, args.approval_evidence, args.project, args.lock)
    if not passed:
        print(f"reazonspeech-nemo-v2 preflight gate: BLOCKED: {reason}")
        return 2
    print("reazonspeech-nemo-v2 preflight gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
