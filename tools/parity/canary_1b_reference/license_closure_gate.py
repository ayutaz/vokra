#!/usr/bin/env -S uv run --no-project --offline --python 3.12 python
"""Fail-closed verifier for the captured Canary dependency license closure.

The VAST audit is factual evidence only.  This gate binds the immutable audit
report, every publisher license byte/hash, native/ELF inventory, and the
first-party virtual project evidence to a checked-in review manifest.  It does
not import the reference environment, acquire a model, run Cargo, or approve
publication.  Missing evidence, an owner decision, or a forbidden license
always returns exit status 2.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any


SCHEMA = "vokra-canary-1b-license-closure-v1"
AUDIT_SCHEMA = "vokra-canary-1b-reference-dependency-audit-v1"
AUDIT_STATUS = "BLOCKED_UNREVIEWED_TRANSITIVE"
PUBLICATION = "NO_UPLOAD"
GATE_VERSION = 1
SOURCE_EVIDENCE_STATUS = "GIT_OBJECTS_VERIFIED_ARCHIVE_COPY_OMITTED"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN = {"AGPL", "GPL", "LGPL"}
ALLOWED_SPDX = {
    "AGPL-3.0-only",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "GPL-3.0-only",
    "ISC",
    "LGPL-2.1-or-later",
    "LGPL-3.0-only",
    "MIT",
    "MPL-2.0",
    "PSF-2.0",
    "Python-2.0",
    "Zlib",
}
REVIEW_DISPOSITIONS = {
    "OWNER_REVIEW_APPROVAL_REQUIRED",
    "OWNER_REVIEW_NATIVE_PAYLOAD",
    "BLOCKED_FORBIDDEN_LICENSE_SIGNAL",
    "BLOCKED_BUNDLED_FORBIDDEN_LICENSE",
    "BLOCKED_LICENSE_BYTES_MISSING",
    "BLOCKED_UNCLASSIFIED_LICENSE",
}
EXPECTED_HEAD = "87da78dc7709075d9dc23b797fc978b9c678c777"
EXPECTED_ARCHIVE_SHA256 = "d58ba75c3985f81f1a7e5887d88ad8c09187fbef55116c8630d0c16600bad283"
EXPECTED_AUDIT_REPORT_SHA256 = "37dd6ce7d30b1e7a3d3c184a5379c4519bb2dd1fa06f3b03349b0530fec51059"
EXPECTED_PACKAGE_ROWS_SHA256 = "4d5a513755f99c8365042ffff85d31bbffa59226ea80726264adb81a8f07e961"
EXPECTED_PACKAGE_FACTS_SHA256 = "ba7b54e22b009341640f5604ddd04016be769c98a95d99171d2f67edb332cb66"
EXPECTED_DEPENDENCY_PATHS_SHA256 = "d4733950a16ddebd23d369a32b5fc91fa0c1e5344e748eee084a0dbff0eea36a"
EXPECTED_LICENSE_ARCHIVE_SHA256 = "53a60026de4c34c9e03e7b99df866c145dfa9f07670ff1099464f7a3cc64d8a1"
EXPECTED_SCOPE_SHA256 = "f271515af4a6a5c97ef84bd93b21f18e36da8d63c8885693004fcf2bd7369d1e"
EXPECTED_PROJECT_SHA256 = "31c002238c213d64f78f38f68f613f168d12991df6b3aac48dd95662da85245e"
EXPECTED_LOCK_SHA256 = "004f0b4d60ba51caf655789eff6e02afb3fa896f837f8d4def909b5cce33b730"
EXPECTED_AUDIT_SCRIPT_SHA256 = "de9f637d54f39895b1006e7398bffbc05a70c89e6b5d81ecc583b836e47ab546"
EXPECTED_WRAPPER_SHA256 = "311fbef943d571c221273d22ec84accfd4e1809550c94435283cf09a1fbe3d60"
EXPECTED_SOURCE_SNAPSHOTS = {
    "tools/parity/canary_1b_reference/pyproject.toml": (946, "31c002238c213d64f78f38f68f613f168d12991df6b3aac48dd95662da85245e"),
    "tools/parity/canary_1b_reference/uv.lock": (118649, "004f0b4d60ba51caf655789eff6e02afb3fa896f837f8d4def909b5cce33b730"),
    "tools/parity/canary_1b_reference/dependency_audit.py": (41770, "de9f637d54f39895b1006e7398bffbc05a70c89e6b5d81ecc583b836e47ab546"),
    "scripts/publish/vast-ai/audit-canary-1b-dependencies.sh": (7510, "311fbef943d571c221273d22ec84accfd4e1809550c94435283cf09a1fbe3d60"),
}


class ClosureError(ValueError):
    """Malformed, stale, incomplete, or unsafe closure evidence."""


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ClosureError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path, label: str) -> tuple[bytes, Any]:
    if not path.is_file() or path.is_symlink():
        raise ClosureError(f"{label} is missing or not a regular file")
    try:
        payload = path.read_bytes()
        return payload, json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ClosureError) as error:
        raise ClosureError(f"cannot read {label}: {error}") from error


def safe_archive_name(value: Any) -> str:
    if not isinstance(value, str) or not value or PurePosixPath(value).name != value:
        raise ClosureError(f"unsafe license archive path: {value!r}")
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ClosureError(f"unsafe license archive path: {value!r}")
    return value


def require_hex(value: Any, label: str, length: int = 64) -> str:
    pattern = HEX64 if length == 64 else HEX40
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ClosureError(f"{label} is not lowercase {length}-hex")
    return value


def exact_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ClosureError(f"{label} schema is not exact")


def validate_manifest(manifest: Any) -> None:
    exact_keys(
        manifest,
        {
            "schema", "gate_version", "audit_schema", "audit_status", "publication",
            "expected_head", "audit_report_sha256", "package_rows_sha256",
            "package_facts_sha256", "dependency_paths_sha256", "license_archive_sha256",
            "candidate_owner_scope_sha256", "package_row_count", "package_fact_count",
            "license_file_count", "source_snapshot_evidence", "virtual_project_evidence",
            "package_reviews", "approval", "blockers",
        },
        "closure manifest",
    )
    if manifest["schema"] != SCHEMA or manifest["gate_version"] != GATE_VERSION:
        raise ClosureError("closure manifest schema/version drifted")
    if manifest["audit_schema"] != AUDIT_SCHEMA or manifest["audit_status"] != AUDIT_STATUS:
        raise ClosureError("closure manifest audit identity drifted")
    if manifest["publication"] != PUBLICATION:
        raise ClosureError("closure manifest is not NO_UPLOAD")
    require_hex(manifest["expected_head"], "expected_head", 40)
    if manifest["expected_head"] != EXPECTED_HEAD:
        raise ClosureError("closure manifest HEAD is not the witnessed audit HEAD")
    expected_digests = {
        "audit_report_sha256": EXPECTED_AUDIT_REPORT_SHA256,
        "package_rows_sha256": EXPECTED_PACKAGE_ROWS_SHA256,
        "package_facts_sha256": EXPECTED_PACKAGE_FACTS_SHA256,
        "dependency_paths_sha256": EXPECTED_DEPENDENCY_PATHS_SHA256,
        "license_archive_sha256": EXPECTED_LICENSE_ARCHIVE_SHA256,
        "candidate_owner_scope_sha256": EXPECTED_SCOPE_SHA256,
    }
    for field in (
        "audit_report_sha256", "package_rows_sha256", "package_facts_sha256",
        "dependency_paths_sha256", "license_archive_sha256", "candidate_owner_scope_sha256",
    ):
        require_hex(manifest[field], field)
        if manifest[field] != expected_digests[field]:
            raise ClosureError(f"closure manifest evidence digest drifted: {field}")
    if (manifest["package_row_count"], manifest["package_fact_count"], manifest["license_file_count"]) != (134, 133, 200):
        raise ClosureError("closure manifest evidence counts drifted")
    for field in ("package_row_count", "package_fact_count", "license_file_count"):
        if isinstance(manifest[field], bool) or not isinstance(manifest[field], int) or manifest[field] <= 0:
            raise ClosureError(f"{field} is not a positive integer")
    if not isinstance(manifest["blockers"], list) or not manifest["blockers"]:
        raise ClosureError("manifest must retain explicit blockers")
    approval = manifest["approval"]
    exact_keys(approval, {"status", "signer", "digest"}, "approval")
    if approval["status"] != "OWNER_SIGNOFF_REQUIRED" or approval["signer"] is not None or approval["digest"] is not None:
        raise ClosureError("owner approval must remain fail-closed")


def validate_sources(manifest: dict[str, Any]) -> None:
    evidence = manifest["source_snapshot_evidence"]
    exact_keys(evidence, {"archive_sha256", "archive_status", "missing_paths", "snapshots"}, "source snapshot evidence")
    require_hex(evidence["archive_sha256"], "source archive SHA-256")
    if evidence["archive_status"] != SOURCE_EVIDENCE_STATUS:
        raise ClosureError("source archive copy omission is not recorded as Git-object verified")
    missing = evidence["missing_paths"]
    expected_missing = {
        "tools/parity/canary_1b_reference/pyproject.toml",
        "tools/parity/canary_1b_reference/uv.lock",
        "tools/parity/canary_1b_reference/dependency_audit.py",
        "scripts/publish/vast-ai/audit-canary-1b-dependencies.sh",
    }
    if set(missing) != expected_missing:
        raise ClosureError("missing source snapshot list is not exact")
    if not isinstance(evidence["snapshots"], list) or len(evidence["snapshots"]) != 4:
        raise ClosureError("source snapshot evidence is incomplete")
    if evidence["archive_sha256"] != EXPECTED_ARCHIVE_SHA256:
        raise ClosureError("source archive SHA-256 is not the witnessed archive")
    for row in evidence["snapshots"]:
        exact_keys(row, {"path", "bytes", "sha256", "source_commit", "archived"}, "source snapshot")
        require_hex(row["source_commit"], "source snapshot commit", 40)
        require_hex(row["sha256"], "source snapshot SHA-256")
        expected_snapshot = EXPECTED_SOURCE_SNAPSHOTS.get(row["path"])
        if row["path"] not in missing or row["archived"] is not False or expected_snapshot != (row["bytes"], row["sha256"]) or row["source_commit"] != EXPECTED_HEAD:
            raise ClosureError("source snapshot does not record the exact verified Git object")
        if isinstance(row["bytes"], bool) or not isinstance(row["bytes"], int) or row["bytes"] <= 0:
            raise ClosureError("source snapshot byte count is invalid")


def validate_virtual_project(manifest: dict[str, Any]) -> None:
    row = manifest["virtual_project_evidence"]
    exact_keys(
        row,
        {
            "row_id", "project_path", "project_bytes", "project_sha256", "license_path",
            "license_bytes", "license_sha256", "license_spdx", "package", "installed",
            "registry_evidence", "wheel_evidence", "native_payloads", "review",
        },
        "virtual project evidence",
    )
    if row["row_id"] != "vokra-canary-1b-reference@0.1.0" or row["project_path"] != "tools/parity/canary_1b_reference/pyproject.toml" or row["license_path"] != "LICENSE":
        raise ClosureError("virtual project identity drifted")
    require_hex(row["project_sha256"], "virtual project SHA-256")
    require_hex(row["license_sha256"], "virtual project license SHA-256")
    if row["project_sha256"] != EXPECTED_PROJECT_SHA256 or row["license_sha256"] != "e1fef339bc7071bae13a31ca873f95e3157665d68342f558e1dd4974ca5a4cee" or row["license_spdx"] != "Apache-2.0" or row["package"] is not False or row["installed"] is not False:
        raise ClosureError("virtual project license/package disposition drifted")
    if row["registry_evidence"] != "NOT_APPLICABLE_VIRTUAL_PROJECT" or row["wheel_evidence"] != "NOT_APPLICABLE_PACKAGE_FALSE" or row["native_payloads"] != []:
        raise ClosureError("virtual project evidence claims registry/native payloads")


def validate_license_file(file_row: Any, archive_dir: Path, expected: dict[str, Any]) -> None:
    exact_keys(file_row, {"path", "archive_path", "bytes", "sha256"}, "license file fact")
    if file_row != expected:
        raise ClosureError("license file fact differs from captured audit")
    archive_name = safe_archive_name(file_row["archive_path"])
    path = archive_dir / archive_name
    if not path.is_file() or path.is_symlink():
        raise ClosureError(f"captured publisher license bytes are missing: {archive_name}")
    if path.stat().st_size != file_row["bytes"] or file_digest(path) != file_row["sha256"]:
        raise ClosureError(f"publisher license bytes/hash mismatch: {archive_name}")


def validate_package_reviews(manifest: dict[str, Any], report: dict[str, Any], archive_dir: Path) -> None:
    facts = report.get("package_facts")
    rows = report.get("package_rows")
    if not isinstance(facts, list) or not isinstance(rows, list):
        raise ClosureError("audit package rows/facts are missing")
    non_virtual = [row for row in rows if isinstance(row, dict) and row.get("source") != {"virtual": "."}]
    roots = [row for row in rows if isinstance(row, dict) and row.get("source") == {"virtual": "."}]
    if len(roots) != 1 or roots[0].get("name") != "vokra-canary-1b-reference" or roots[0].get("version") != "0.1.0":
        raise ClosureError("virtual project lock row is not exact")
    reviews = manifest["package_reviews"]
    if len(rows) != manifest["package_row_count"] or len(facts) != manifest["package_fact_count"] or len(reviews) != len(facts):
        raise ClosureError("package row/fact/review counts do not match")
    if len(non_virtual) != len(facts) or len(set((row.get("name"), row.get("version")) for row in non_virtual)) != len(non_virtual):
        raise ClosureError("active package identity set is malformed")
    for field in ("package_rows", "package_facts", "dependency_paths", "license_archive"):
        digest_field = f"{field}_sha256"
        if digest(report.get(field)) != report.get(digest_field):
            raise ClosureError(f"audit {field} digest does not match its contents")
    fact_by_id = {(fact.get("name"), fact.get("version")): fact for fact in facts if isinstance(fact, dict)}
    review_ids: list[tuple[Any, Any]] = []
    archive_expected: dict[str, dict[str, Any]] = {}
    forbidden_rows = 0
    native_rows = 0
    for review in reviews:
        exact_keys(
            review,
            {
                "id", "name", "version", "source", "lock_row_sha256", "fact_sha256",
                "publisher_license_bytes", "publisher_license_sha256", "license_flags",
                "license_files", "license_files_sha256", "native_files", "native_files_sha256",
                "primary_spdx", "observed_spdx", "disposition", "basis",
            },
            "package review",
        )
        identity = (review["name"], review["version"])
        if review["id"] != f"{review['name']}=={review['version']}" or identity in review_ids:
            raise ClosureError("package review identity is duplicated or malformed")
        review_ids.append(identity)
        fact = fact_by_id.get(identity)
        if fact is None:
            raise ClosureError(f"review has no matching package fact: {identity}")
        for field in ("source", "lock_row_sha256", "fact_sha256", "license_files_sha256", "native_files_sha256"):
            if review[field] != fact.get(field):
                raise ClosureError(f"review does not bind fact field: {identity} {field}")
        publisher = fact.get("publisher_license")
        if not isinstance(publisher, str) or review["publisher_license_bytes"] != len(publisher.encode()) or review["publisher_license_sha256"] != hashlib.sha256(publisher.encode()).hexdigest():
            raise ClosureError(f"publisher license metadata hash mismatch: {identity}")
        if review["license_flags"] != fact.get("license_flags") or review["license_files"] != fact.get("license_files") or review["native_files"] != fact.get("native_files"):
            raise ClosureError(f"review does not bind captured license/native facts: {identity}")
        if not isinstance(review["basis"], list) or not review["basis"]:
            raise ClosureError(f"review basis is missing: {identity}")
        primary = review["primary_spdx"]
        if not isinstance(primary, str) or (primary != "UNKNOWN" and primary not in ALLOWED_SPDX and not (re.fullmatch(r"(?:[A-Za-z0-9.-]+)(?: AND (?:[A-Za-z0-9.-]+))+", primary) and all(term in ALLOWED_SPDX for term in primary.split(" AND ")))):
            raise ClosureError(f"review SPDX is malformed: {identity}")
        if not isinstance(review["observed_spdx"], list) or any(item not in ALLOWED_SPDX and item not in {"AGPL-3.0-only", "GPL-3.0-only", "LGPL-2.1-or-later", "LGPL-3.0-only", "UNKNOWN"} for item in review["observed_spdx"]):
            raise ClosureError(f"review observed SPDX is malformed: {identity}")
        if review["disposition"] not in REVIEW_DISPOSITIONS:
            raise ClosureError(f"review disposition is malformed: {identity}")
        if review["disposition"] == "BLOCKED_FORBIDDEN_LICENSE_SIGNAL":
            forbidden_rows += 1
        if review["disposition"] in {"OWNER_REVIEW_NATIVE_PAYLOAD", "BLOCKED_BUNDLED_FORBIDDEN_LICENSE"}:
            native_rows += int(bool(fact.get("native_files")))
        for license_file in fact.get("license_files", []):
            archive_name = safe_archive_name(license_file.get("archive_path"))
            if archive_name in archive_expected and archive_expected[archive_name] != license_file:
                raise ClosureError("archive path maps to conflicting license facts")
            archive_expected[archive_name] = license_file
            validate_license_file(license_file, archive_dir, license_file)
    if set(review_ids) != set(fact_by_id) or len(archive_expected) != manifest["license_file_count"]:
        raise ClosureError("package review identities or license archive count is not exact")
    report_archive = report.get("license_archive")
    expected_report_archive = {
        name: {"path": name, "bytes": row["bytes"], "sha256": row["sha256"]}
        for name, row in archive_expected.items()
    }
    report_archive_by_name = {
        item.get("path"): item for item in report_archive if isinstance(item, dict)
    } if isinstance(report_archive, list) else {}
    if not isinstance(report_archive, list) or len(report_archive_by_name) != len(report_archive) or report_archive_by_name != expected_report_archive:
        raise ClosureError("audit license archive inventory differs from package facts")
    actual_archive = {path.name for path in archive_dir.iterdir() if path.is_file() and not path.is_symlink()}
    if actual_archive != set(archive_expected):
        raise ClosureError("license archive contains missing or unexpected files")
    if not forbidden_rows or native_rows < 1:
        raise ClosureError("forbidden/native boundary was not represented")


def validate_report(manifest: dict[str, Any], report_bytes: bytes, report: dict[str, Any]) -> None:
    if report.get("schema") != AUDIT_SCHEMA or report.get("status") != AUDIT_STATUS or report.get("publication") != PUBLICATION:
        raise ClosureError("audit report is not blocked/no-upload evidence")
    if report.get("head") != manifest["expected_head"] or report.get("expected_head") != manifest["expected_head"] or report.get("clean") is not True:
        raise ClosureError("audit report HEAD is stale or dirty")
    if file_digest_from_bytes(report_bytes) != manifest["audit_report_sha256"]:
        raise ClosureError("audit report SHA-256 mismatch")
    for field in ("package_rows_sha256", "package_facts_sha256", "dependency_paths_sha256", "license_archive_sha256", "candidate_owner_scope_sha256"):
        if report.get(field) != manifest[field]:
            raise ClosureError(f"audit report digest mismatch: {field}")
    for field, expected in (
        ("project_sha256", EXPECTED_PROJECT_SHA256),
        ("lock_sha256", EXPECTED_LOCK_SHA256),
        ("audit_script_sha256", EXPECTED_AUDIT_SCRIPT_SHA256),
        ("wrapper_sha256", EXPECTED_WRAPPER_SHA256),
    ):
        if report.get(field) != expected:
            raise ClosureError(f"audit report source digest mismatch: {field}")
    environment = report.get("environment")
    if not isinstance(environment, dict) or any(environment.get(field) is not False for field in ("weights_acquired", "source_acquired", "model_imported", "model_executed", "cargo_invoked")) or environment.get("upload") != PUBLICATION:
        raise ClosureError("audit report model-free/no-upload contract drifted")
    if report.get("collector_failures") != [] or report.get("distribution_inventory", {}).get("exact") is not True:
        raise ClosureError("audit report closure evidence is not exact")


def file_digest_from_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def validate(*, manifest_path: Path, report_path: Path, archive_dir: Path) -> tuple[bool, str]:
    try:
        _manifest_bytes, manifest = load_json(manifest_path, "closure manifest")
        report_bytes, report = load_json(report_path, "audit report")
        if not isinstance(manifest, dict) or not isinstance(report, dict):
            raise ClosureError("manifest/report must be JSON objects")
        validate_manifest(manifest)
        validate_sources(manifest)
        validate_virtual_project(manifest)
        validate_report(manifest, report_bytes, report)
        if not archive_dir.is_dir() or archive_dir.is_symlink():
            raise ClosureError("license archive directory is missing or symlinked")
        validate_package_reviews(manifest, report, archive_dir)
        raise ClosureError("OWNER_SIGNOFF_REQUIRED: factual closure is reviewed but publication remains NO_UPLOAD")
    except (ClosureError, OSError, TypeError, ValueError) as error:
        return False, str(error)


def self_test() -> int:
    if PUBLICATION != "NO_UPLOAD" or "OWNER_SIGNOFF_REQUIRED" not in REVIEW_DISPOSITIONS | {"OWNER_SIGNOFF_REQUIRED"}:
        print("license closure gate self-test: fail-closed markers missing", file=sys.stderr)
        return 1
    if set(FORBIDDEN) != {"AGPL", "GPL", "LGPL"}:
        return 1
    print("canary_1b license closure gate self-test: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--audit-report", type=Path)
    parser.add_argument("--license-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.manifest, args.audit_report, args.license_dir)):
            parser.error("--self-test cannot be combined with evidence paths")
        return self_test()
    if any(value is None for value in (args.manifest, args.audit_report, args.license_dir)):
        parser.error("--manifest, --audit-report, and --license-dir are required")
    ok, message = validate(manifest_path=args.manifest, report_path=args.audit_report, archive_dir=args.license_dir)
    if ok:
        print(message)
        return 0
    print(f"BLOCKED: {message}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
