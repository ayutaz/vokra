#!/usr/bin/env -S uv run --frozen --project tools/parity/dia_1_6b_reference python
"""Bind a Dia dependency audit to a deterministic, unsigned owner-review scope.

This tool only consumes the model-free VAST audit report and preparation
evidence.  It never classifies a license and cannot produce VAST_READY without
an exact future owner approval record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

SCHEMA = "vokra-dia-dependency-owner-scope-v1"
APPROVAL_SCHEMA = "vokra-dia-dependency-owner-approval-v1"
GATE = "BLOCKED_UNREVIEWED_TRANSITIVE"
PUBLICATION = "NO_UPLOAD"
PROJECT = Path(__file__).parent
LOCK_SHA256 = "58218102471c94979b1e9147759abf50fa3784793c193ff30cdde908400650dc"
PYPROJECT_SHA256 = "fa675f2c7542bd9eebedcc6ba29963f49093305c7a518542d71fad424449e77b"
HEAD_RE = re.compile(r"[0-9a-f]{40}")
SHA_RE = re.compile(r"[0-9a-f]{64}")


class ScopeError(RuntimeError):
    pass


def digest_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ScopeError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ScopeError(f"JSON root is not an object: {path}")
    return value


def require_output(path: Path) -> None:
    if not path.is_absolute() or any(part in {".", ".."} for part in path.parts[1:]) or path == Path(path.anchor):
        raise ScopeError("owner scope output must be absolute and free of dot components")
    if path.exists() or path.is_symlink():
        raise ScopeError("owner scope output must be absent")
    cursor = path.parent
    while True:
        if cursor.is_symlink() or not cursor.is_dir():
            raise ScopeError("owner scope output parent must be a regular directory")
        if cursor == Path(cursor.anchor):
            break
        cursor = cursor.parent


def require_audit(report: dict[str, Any]) -> None:
    if report.get("schema") != "vokra-dia-dependency-audit-v1" or report.get("status") != "FACTS_COLLECTED_GATE_BLOCKED" or report.get("dependency_license_audit") != GATE or report.get("publication") != PUBLICATION:
        raise ScopeError("audit schema/gate/publication is not fail-closed")
    repository = report.get("repository")
    if not isinstance(repository, dict) or not HEAD_RE.fullmatch(repository.get("head", "")) or repository.get("clean") is not True:
        raise ScopeError("audit repository identity is missing or dirty")
    contract = report.get("contract")
    if not isinstance(contract, dict) or contract.get("gate_status") != GATE or contract.get("uv_lock_sha256") != LOCK_SHA256 or contract.get("pyproject_sha256") != PYPROJECT_SHA256:
        raise ScopeError("audit contract identity is stale or not fail-closed")
    closure = report.get("closure")
    if not isinstance(closure, dict) or closure.get("exact") is not True or closure.get("missing") != [] or closure.get("unexpected") != [] or closure.get("duplicate_identities") != [] or len(closure.get("expected", [])) != 26 or closure.get("expected") != closure.get("installed"):
        raise ScopeError("audit is not the exact 26-package Linux closure")
    packages = report.get("packages")
    if not isinstance(packages, list) or len(packages) != 26:
        raise ScopeError("audit package inventory is not exactly 26 rows")
    identities = []
    for item in packages:
        lock = item.get("lock") if isinstance(item, dict) else None
        installed = item.get("installed") if isinstance(item, dict) else None
        if not isinstance(lock, dict) or not isinstance(lock.get("name"), str) or not isinstance(lock.get("version"), str) or not isinstance(installed, dict) or not isinstance(installed.get("identity"), str):
            raise ScopeError("audit package row is malformed")
        identities.append(lock["name"] + "==" + lock["version"])
    if len(identities) != 26 or len(set(identities)) != 26:
        raise ScopeError("audit package identities are incomplete or duplicated")
    license_facts = report.get("license_facts")
    if not isinstance(license_facts, dict) or license_facts.get("packages") != 26 or license_facts.get("publisher_license_evidence_missing") != [] or license_facts.get("publisher_bytes_recorded") != 50:
        raise ScopeError("publisher license evidence does not match the factual d9 report")
    native = report.get("native_facts")
    if not isinstance(native, dict) or not isinstance(native.get("files"), list):
        raise ScopeError("native/bundled inventory is missing")
    if report.get("failures") != []:
        raise ScopeError("audit contains collection failures")


def require_preparation(preparation: dict[str, Any]) -> None:
    if preparation.get("schema") != "vokra-dia-reference-preparation-v1" or preparation.get("status") != "PREPARED_NO_BLAS" or preparation.get("publication") != PUBLICATION:
        raise ScopeError("preparation evidence is not the exact no-BLAS preparation")
    build = preparation.get("build")
    if not isinstance(build, dict) or build.get("isolation") != "no-build-isolation; builder venv preinstalled from hash-pinned constraints":
        raise ScopeError("preparation isolation fact drifted")
    if preparation.get("runtime", {}).get("soundfile_installed") is not False or preparation.get("runtime", {}).get("torchaudio_installed") is not False:
        raise ScopeError("optional audio distributions were not proven absent")


def build_scope(report_path: Path, preparation_path: Path, expected_head: str) -> dict[str, Any]:
    if not HEAD_RE.fullmatch(expected_head):
        raise ScopeError("expected HEAD must be lowercase 40-hex")
    report = read_json(report_path)
    preparation = read_json(preparation_path)
    require_audit(report)
    require_preparation(preparation)
    if report["repository"]["head"] != expected_head:
        raise ScopeError("audit report HEAD does not match the requested exact HEAD")
    for name, expected in (("uv.lock", LOCK_SHA256), ("pyproject.toml", PYPROJECT_SHA256)):
        path = PROJECT / name
        if path.is_symlink() or not path.is_file() or digest_bytes(path) != expected:
            raise ScopeError(f"current Dia project input drifted: {name}")
    packages = report["packages"]
    package_identities = sorted(item["installed"]["identity"] for item in packages)
    native_inventory = report["native_facts"]["files"]
    native_rows = []
    package_identity_set = set(package_identities)
    for item in native_inventory:
        if not isinstance(item, dict) or not isinstance(item.get("package_identity"), str) or item["package_identity"] not in package_identity_set or not SHA_RE.fullmatch(item.get("sha256", "")):
            raise ScopeError("native inventory row lacks package identity or digest")
        native_rows.append(item)
    scope = {
        "schema": SCHEMA,
        "status": "PENDING_OWNER_REVIEW",
        "dependency_license_audit": GATE,
        "publication": PUBLICATION,
        "owner_review": {"decision": None, "required": "owner/legal must classify every package and native/bundled payload", "approval_schema": APPROVAL_SCHEMA},
        "audit": {"report_sha256": digest_bytes(report_path), "report_bytes": report_path.stat().st_size, "expected_head": expected_head, "project_lock_sha256": LOCK_SHA256, "project_pyproject_sha256": PYPROJECT_SHA256, "package_count": 26, "package_identities": package_identities, "publisher_bytes_recorded": 50, "native_file_count": len(native_rows), "native_inventory_sha256": hashlib.sha256(canonical(native_rows)).hexdigest(), "failures": []},
        "native_inventory": native_rows,
        "preparation": {"evidence_sha256": digest_bytes(preparation_path), "schema": preparation["schema"], "status": preparation["status"], "publication": preparation["publication"]},
    }
    scope["scope_sha256"] = hashlib.sha256(canonical(scope)).hexdigest()
    return scope


def validate_scope(scope: dict[str, Any], report_path: Path, preparation_path: Path, expected_head: str) -> None:
    if scope.get("schema") != SCHEMA or scope.get("status") != "PENDING_OWNER_REVIEW" or scope.get("dependency_license_audit") != GATE or scope.get("publication") != PUBLICATION:
        raise ScopeError("owner scope is not unsigned fail-closed candidate")
    recorded = scope.get("scope_sha256")
    if not isinstance(recorded, str) or not SHA_RE.fullmatch(recorded):
        raise ScopeError("owner scope digest is missing")
    unsigned = dict(scope)
    del unsigned["scope_sha256"]
    if hashlib.sha256(canonical(unsigned)).hexdigest() != recorded:
        raise ScopeError("owner scope digest mismatch")
    expected = build_scope(report_path, preparation_path, expected_head)
    if expected != scope:
        raise ScopeError("owner scope no longer matches exact report/head/preparation facts")


def self_test() -> int:
    assert SCHEMA.endswith("v1") and APPROVAL_SCHEMA.endswith("v1")
    for unsafe in (Path("relative.json"), Path("/tmp/../owner.json"), Path("/")):
        try: require_output(unsafe)
        except ScopeError: pass
        else: raise AssertionError("unsafe owner scope output accepted")
    with __import__("tempfile").TemporaryDirectory(prefix="dia-owner-scope-") as directory:
        root = Path(directory)
        link = root / "link"; link.mkdir()
        link_target = root / "link-alias"; link_target.symlink_to(link, target_is_directory=True)
        try: require_output(root / "link-alias" / "scope.json")
        except ScopeError: pass
        else: raise AssertionError("symlinked owner scope output accepted")
        report = {"schema": "vokra-dia-dependency-audit-v1", "status": "FACTS_COLLECTED_GATE_BLOCKED", "dependency_license_audit": GATE, "publication": PUBLICATION, "repository": {"head": "0" * 40, "clean": True}, "contract": {"gate_status": GATE, "uv_lock_sha256": LOCK_SHA256, "pyproject_sha256": PYPROJECT_SHA256}, "closure": {"exact": True, "missing": [], "unexpected": [], "duplicate_identities": [], "expected": [f"p{i}==1" for i in range(26)], "installed": [f"p{i}==1" for i in range(26)]}, "packages": [{"lock": {"name": f"p{i}", "version": "1"}, "installed": {"identity": f"p{i}==1"}} for i in range(26)], "license_facts": {"packages": 26, "publisher_license_evidence_missing": [], "publisher_bytes_recorded": 50}, "native_facts": {"files": [{"package_identity": "p0==1", "sha256": "a" * 64, "path": "x.so"}]}, "failures": []}
        preparation = {"schema": "vokra-dia-reference-preparation-v1", "status": "PREPARED_NO_BLAS", "publication": PUBLICATION, "build": {"isolation": "no-build-isolation; builder venv preinstalled from hash-pinned constraints"}, "runtime": {"soundfile_installed": False, "torchaudio_installed": False}}
        report_path, preparation_path = root / "report.json", root / "preparation.json"
        report_path.write_text(json.dumps(report), encoding="utf-8"); preparation_path.write_text(json.dumps(preparation), encoding="utf-8")
        scope = build_scope(report_path, preparation_path, "0" * 40)
        validate_scope(scope, report_path, preparation_path, "0" * 40)
        for key, value in (("head", "1" * 40), ("clean", False)):
            drifted = json.loads(json.dumps(report)); drifted["repository"][key] = value; report_path.write_text(json.dumps(drifted), encoding="utf-8")
            try: build_scope(report_path, preparation_path, "0" * 40)
            except ScopeError: pass
            else: raise AssertionError(f"report repository drift accepted: {key}")
        report_path.write_text(json.dumps(report), encoding="utf-8")
        for key, value in (("status", "BLOCKED"), ("dependency_license_audit", "AUDITED_ALLOW")):
            drifted = json.loads(json.dumps(report)); drifted[key] = value; report_path.write_text(json.dumps(drifted), encoding="utf-8")
            try: build_scope(report_path, preparation_path, "0" * 40)
            except ScopeError: pass
            else: raise AssertionError(f"report status drift accepted: {key}")
        report_path.write_text(json.dumps(report), encoding="utf-8")
        for key, value in (("uv_lock_sha256", "1" * 64), ("pyproject_sha256", "1" * 64)):
            drifted = json.loads(json.dumps(report)); drifted["contract"][key] = value; report_path.write_text(json.dumps(drifted), encoding="utf-8")
            try: build_scope(report_path, preparation_path, "0" * 40)
            except ScopeError: pass
            else: raise AssertionError(f"report contract drift accepted: {key}")
        report_path.write_text(json.dumps(report), encoding="utf-8")
        broken = dict(scope); broken["status"] = "VAST_READY"
        try: validate_scope(broken, report_path, preparation_path, "0" * 40)
        except ScopeError: pass
        else: raise AssertionError("unsigned VAST_READY scope accepted")
    print("dia owner-review scope: self-test PASS (model-free, unsigned, NO_UPLOAD)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--preparation", type=Path)
    parser.add_argument("--expected-head")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if None in (args.report, args.preparation, args.expected_head):
        parser.error("--report, --preparation, and --expected-head are required")
    if args.validate:
        if args.output is None: parser.error("--validate requires --output as scope JSON")
        validate_scope(read_json(args.output), args.report, args.preparation, args.expected_head)
        print("dia owner-review scope: validation PASS (PENDING_OWNER_REVIEW)")
        return 0
    if args.output is None: parser.error("scope generation requires --output")
    scope = build_scope(args.report, args.preparation, args.expected_head)
    require_output(args.output)
    args.output.write_bytes(canonical(scope) + b"\n")
    print("dia owner-review scope: PENDING_OWNER_REVIEW (NO_UPLOAD)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
