#!/usr/bin/env python3
"""Model-free MOSS-TTS Local closure audit.

This audit reads only the checked-in project, lock, and pending gate manifest.
It does not import a model package, fetch a checkpoint, execute inference, or
make a license/owner decision.  The report records the exact pending approval
scope, including the 438-tensor Local identity and its required v2 companion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

import preflight_gate as gate


SCHEMA = "vokra-moss-tts-local-model-free-audit-v1"
BLOCKERS = [
    "BLOCKED_PACKAGE_LICENSE_REVIEW: locked Python package license and native/bundled review rows remain unresolved",
    "BLOCKED_COMPOSITE_PCM: official Local-plus-v2 end-to-end PCM comparison remains COMPOSITE_PCM_NOT_RUN",
    "BLOCKED_OWNER_SIGNOFF: approval.status is OWNER_SIGNOFF_REQUIRED and no owner evidence is accepted",
    "BLOCKED_REAL_WEIGHT_PARITY: checkpoint conversion, official reference, CPU parity, and Metal parity were not run in model-free mode",
]


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: Any) -> str:
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def load_json(path: Path) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)


def regular(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")


def safe_output(path: Path) -> None:
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise ValueError("audit output must be an absolute lexical-safe path")
    current = Path(path.anchor)
    for part in path.parts[1:-1]:
        current /= part
        # macOS exposes /var and /tmp as stable system aliases.  Resolve only
        # those two aliases implicitly; all task-controlled ancestors must be
        # real directories so an attacker cannot redirect the report.
        if current.is_symlink() and current not in {Path("/var"), Path("/tmp")}:
            raise ValueError("audit output has a symlinked ancestor")
        if not current.is_dir():
            raise ValueError("audit output has an unsafe or non-directory ancestor")
    if os.path.lexists(path):
        raise ValueError("audit output must not already exist")


def publish_no_replace(path: Path, payload: bytes) -> None:
    """Publish a small report atomically without clobbering an existing path."""

    safe_output(path)
    fd = -1
    temporary: Path | None = None
    try:
        for attempt in range(32):
            candidate = path.parent / f".{path.name}.tmp-{os.getpid()}-{attempt}"
            try:
                fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
            except FileExistsError:
                continue
            temporary = candidate
            break
        if fd < 0 or temporary is None:
            raise OSError("could not reserve unique audit temp file")
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise OSError("zero-byte report write")
            offset += written
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.link(temporary, path)
        os.unlink(temporary)
        temporary = None
    except Exception:
        if fd >= 0:
            os.close(fd)
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        raise


def unresolved(value: Any) -> bool:
    if not isinstance(value, str):
        return True
    normalized = "_".join(value.strip().casefold().split())
    return normalized in gate.PLACEHOLDERS


def validate_inputs(lock_path: Path, project_path: Path, manifest_path: Path) -> tuple[bytes, bytes, bytes, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    for path, label in ((lock_path, "uv.lock"), (project_path, "pyproject.toml"), (manifest_path, "license gate manifest")):
        regular(path, label)
    lock_bytes = lock_path.read_bytes()
    project_bytes = project_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    if digest(lock_bytes) != gate.LOCK_SHA256 or digest(project_bytes) != gate.PROJECT_SHA256:
        raise ValueError("lock/project bytes differ from the code-bound closure")
    try:
        lock_data = tomllib.loads(lock_bytes.decode())
        project_data = tomllib.loads(project_bytes.decode())
        manifest = load_json(manifest_path)
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"closure input is unreadable: {error}") from error
    if not isinstance(project_data, dict) or project_data.get("project", {}).get("name") != "vokra-moss-tts-local-parity":
        raise ValueError("pyproject identity is not the fixed MOSS-TTS Local project")
    if gate.artifact_error(lock_data) is not None:
        raise ValueError(f"resolver artifact metadata is incomplete: {gate.artifact_error(lock_data)}")
    rows = gate.lock_rows(lock_data)
    expected_keys = {
        "gate_version", "lock_sha256", "project_sha256", "package_rows_sha256",
        "package_review_rows_sha256", "package_review_rows", "local_identity",
        "companion_identity", "prompt_contract", "publication", "numeric_state",
        "composite_pcm", "approval_scope_sha256", "approval",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_keys:
        raise ValueError("MOSS-TTS Local manifest schema has missing or extra keys")
    if manifest["gate_version"] != 1 or manifest["lock_sha256"] != gate.LOCK_SHA256 or manifest["project_sha256"] != gate.PROJECT_SHA256:
        raise ValueError("MOSS-TTS Local gate/version digest is not fixed")
    if manifest["package_rows_sha256"] != canonical(rows):
        raise ValueError("canonical lock package graph is not bound")
    reviews = manifest["package_review_rows"]
    expected = [(row["name"], row["version"], row["source"]) for row in rows]
    if not isinstance(reviews, list) or manifest["package_review_rows_sha256"] != canonical(reviews):
        raise ValueError("package review rows are missing or tampered")
    actual = []
    seen: set[tuple[str, str]] = set()
    for row in reviews:
        if not isinstance(row, dict) or set(row) != {"name", "version", "source", "license", "status", "native_bundled_review"}:
            raise ValueError("package review row schema is not exact")
        key = (row["name"], row["version"])
        if key in seen:
            raise ValueError(f"duplicate package review identity: {key}")
        seen.add(key)
        actual.append((row["name"], row["version"], row["source"]))
    if sorted(actual) != sorted(expected) or len(seen) != len(expected):
        raise ValueError("package review rows do not cover the exact lock set")
    if manifest["local_identity"] != gate.LOCAL_IDENTITY:
        raise ValueError("438-tensor Local identity is not the fixed contract")
    if manifest["companion_identity"] != gate.COMPANION_IDENTITY:
        raise ValueError("required tokenizer-v2 companion identity is not the fixed contract")
    if manifest["prompt_contract"] != {"shape": ["rows", 13], "dtype": "u32le", "nonempty": True}:
        raise ValueError("prompt contract drifted")
    if manifest["publication"] != "NO_UPLOAD" or manifest["numeric_state"] != "MEASURED_NOT_GATED" or manifest["composite_pcm"] != "COMPOSITE_PCM_NOT_RUN":
        raise ValueError("publication or measurement posture drifted")
    if manifest["approval_scope_sha256"] != gate.approval_scope(manifest):
        raise ValueError("pending approval scope is not canonical")
    approval = manifest["approval"]
    if not isinstance(approval, dict) or set(approval) != {"status", "signer", "scope_sha256", "digest", "evidence_sha256"} or approval["status"] != "OWNER_SIGNOFF_REQUIRED" or any(approval[key] is not None for key in ("signer", "scope_sha256", "digest", "evidence_sha256")):
        raise ValueError("approval state is not the pending owner-signoff contract")
    return lock_bytes, project_bytes, manifest_bytes, lock_data, manifest, rows


def audit(lock_path: Path, project_path: Path, manifest_path: Path, output: Path | None) -> int:
    lock_bytes, project_bytes, manifest_bytes, _lock_data, manifest, rows = validate_inputs(lock_path, project_path, manifest_path)
    scope_sha256 = gate.approval_scope(manifest)
    unresolved_rows = sum(
        1 for row in manifest["package_review_rows"]
        if unresolved(row["license"]) or unresolved(row["status"]) or unresolved(row["native_bundled_review"])
    )
    report = {
        "schema": SCHEMA,
        "decision": "BLOCKED",
        "status": "BLOCKED_OWNER_REVIEW",
        "evidence_stage": "MODEL_FREE_DEPENDENCY_IDENTITY_SCOPE_AUDIT",
        "no_upload": True,
        "scope_sha256": scope_sha256,
        "scope_status": "PENDING_REVIEW_NOT_OWNER_SIGNABLE",
        "scope_basis": "LOCKED_PACKAGE_REVIEWS_PLUS_438_TENSOR_LOCAL_IDENTITY_AND_REQUIRED_V2_COMPANION",
        "owner_signoff_eligible": False,
        "checkout_inputs": {
            "uv_lock": {"bytes": len(lock_bytes), "sha256": digest(lock_bytes), "status": "VERIFIED"},
            "pyproject": {"bytes": len(project_bytes), "sha256": digest(project_bytes), "status": "VERIFIED"},
            "license_manifest": {"bytes": len(manifest_bytes), "sha256": digest(manifest_bytes), "status": "VERIFIED"},
        },
        "package_review": {
            "locked_rows": len(rows),
            "review_rows": len(manifest["package_review_rows"]),
            "unresolved_rows": unresolved_rows,
            "status": "PENDING_OWNER_REVIEW",
            "license_classification": "NOT_INFERRED",
            "native_bundled_classification": "NOT_INFERRED",
        },
        "resolver_artifacts": {
            "status": "VERIFIED_LOCK_METADATA",
            "registry_rows": sum(row.get("source") != {"virtual": "."} for row in rows),
            "artifact_urls_hashes_sizes": "LOCK_BOUND",
        },
        "local_identity": {
            "repository": gate.LOCAL_IDENTITY["repository"],
            "revision": gate.LOCAL_IDENTITY["revision"],
            "model_path": gate.LOCAL_IDENTITY["model_path"],
            "model_bytes": gate.LOCAL_IDENTITY["model_bytes"],
            "model_sha256": gate.LOCAL_IDENTITY["model_sha256"],
            "tensor_count": gate.LOCAL_IDENTITY["tensor_count"],
            "status": "VERIFIED_IDENTITY_ONLY_NO_WEIGHT_MATERIALIZATION",
        },
        "companion_identity": {**gate.COMPANION_IDENTITY, "status": "VERIFIED_IDENTITY_ONLY_NO_WEIGHT_MATERIALIZATION"},
        "composite_pcm": {
            "status": "COMPOSITE_PCM_NOT_RUN",
            "reason": "Official v2 PCM sidecar and end-to-end Local-plus-v2 execution require owner-approved real-weight VAST work",
        },
        "publication": "NO_UPLOAD",
        "blockers": BLOCKERS,
        "next_step": "After package/license review and owner approval, run run-moss-tts-local-composite-validation.sh on disposable VAST with both approval evidence files.",
    }
    if output is not None:
        try:
            publish_no_replace(output, (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())
        except OSError as error:
            raise ValueError(f"cannot publish model-free audit report: {error}") from error
        print(f"audit_report={output}")
    print(f"MOSS-TTS Local model-free audit: BLOCKED_OWNER_REVIEW scope_sha256={scope_sha256} scope_status=PENDING_REVIEW_NOT_OWNER_SIGNABLE")
    for blocker in BLOCKERS:
        print(blocker)
    return 2


def self_test() -> None:
    project = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="moss-local-audit-") as raw:
        root = Path(raw)
        lock = root / "uv.lock"
        pyproject = root / "pyproject.toml"
        manifest = root / "manifest.json"
        output = root / "report.json"
        lock.write_bytes((project / "uv.lock").read_bytes())
        pyproject.write_bytes((project / "pyproject.toml").read_bytes())
        manifest.write_bytes((project / "license_gate_manifest.json").read_bytes())
        if audit(lock, pyproject, manifest, output) != 2:
            raise SystemExit("self-test did not preserve the pending audit decision")
        report = load_json(output)
        if not isinstance(report, dict) or report.get("scope_status") != "PENDING_REVIEW_NOT_OWNER_SIGNABLE" or report.get("no_upload") is not True or report.get("local_identity", {}).get("tensor_count") != 438:
            raise SystemExit("self-test report contract drifted")
        try:
            audit(lock, pyproject, manifest, output)
        except ValueError:
            pass
        else:
            raise SystemExit("self-test accepted output clobber")
        manifest.write_text('{"approval":{"status":"OWNER_SIGNOFF_REQUIRED","approval":{"status":"OWNER_SIGNOFF_REQUIRED"}}}', encoding="utf-8")
        try:
            audit(lock, pyproject, manifest, None)
        except ValueError:
            pass
        else:
            raise SystemExit("self-test accepted malformed manifest")
    print("model_free_audit.py self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.lock, args.project, args.manifest, args.output)):
            parser.error("--self-test is exclusive with audit input/output arguments")
        self_test()
        return 0
    if not all((args.lock, args.project, args.manifest)):
        parser.error("--lock, --project, and --manifest are required")
    try:
        return audit(args.lock, args.project, args.manifest, args.output)
    except (OSError, ValueError, tomllib.TOMLDecodeError, json.JSONDecodeError) as error:
        print(f"MOSS-TTS Local model-free audit: BLOCKED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
