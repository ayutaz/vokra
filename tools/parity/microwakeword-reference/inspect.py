"""Fail-closed dependency/license preflight for the LiteRT oracle.

This is deliberately reference-only. It audits the locked dependency graph and
wheel identities but does not import LiteRT or inspect model bytes. The current
closure remains BLOCKED_PENDING_VAST_EVIDENCE until an owner reviews the exact
installed Linux x86_64 closure collected on VAST.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

EXPECTED_DEPENDENCIES = {"ai-edge-litert": "==2.1.5", "numpy": "==2.5.2"}
EXPECTED_VERSIONS = {
    "ai-edge-litert": "2.1.5",
    "backports-strenum": "1.3.1",
    "flatbuffers": "25.12.19",
    "numpy": "2.5.2",
    "protobuf": "7.36.1",
    "tqdm": "4.70.0",
    "typing-extensions": "4.16.0",
    "vokra-microwakeword-reference": "0.1.0",
}
EXPECTED_AI_EDGE_WHEEL_SHA256 = "sha256:f1c6d8db4382890881baeb8ed13c0802ada022e0b104b0db8fccf31353899ee0"
# This is the selected CPython 3.12 manylinux x86_64 wheel digest in uv.lock.
EXPECTED_NUMPY_WHEEL_SHA256 = "3cdec01fa790a186d430433fdd4d4ffb70eed6f0eeb4bf05c8dbe2dce0a9bcb8"
EXPECTED_ROWS = frozenset(EXPECTED_VERSIONS)
EXPECTED_PROJECT_SHA256 = "2438d719428e497cc7f101429ba31fb5016e72737659d55aa0269d0824b1183d"
EXPECTED_LOCK_SHA256 = "736fca6145c24984531ef11258cd64aebbb188fa8830300b09232cac0fe567f3"

PRIMARY_SOURCE_PROVENANCE = {
    "selected": {
        "package": "ai-edge-litert",
        "version": "2.1.5",
        "wheel_sha256": "f1c6d8db4382890881baeb8ed13c0802ada022e0b104b0db8fccf31353899ee0",
        "source_tag": "v2.1.5",
        "source_commit": "9d26e89d88ef8785b6a1e54ec41ac8add215a125",
        "license_blob": "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64",
        "restrictive_terms_path_present": False,
    },
    "rejected": {
        "package": "ai-edge-litert",
        "version": "2.2.0",
        "reason": "restrictive TERMS_OF_USE/use-redistribution-benchmark restrictions",
        "publication_permitted": False,
    },
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _wheel_hashes(row: dict[str, Any]) -> list[str]:
    return [str(wheel.get("hash", "")) for wheel in row.get("wheels", [])]


def _strict_json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not valid strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _metadata_license_declarations_present(metadata: dict[str, Any]) -> bool:
    """Recognize substantive bounded declarations, excluding pointers/placeholders."""
    fields = ("license", "license_expression", "license_classifiers")
    return all(isinstance(metadata.get(field), list) for field in fields) and any(
        isinstance(value, str)
        and value.strip()
        and value.strip().casefold() != "unknown"
        for field in fields
        for value in metadata[field]
    )


def _validate_dependency_evidence(value: dict[str, Any], project_sha256: str, lock_sha256: str) -> dict[str, Any]:
    required = {
        "schema", "status", "publication_permitted", "fixture_generation_permitted",
        "owner_review_required", "platform", "project", "uv_lock", "lock",
        "installed_inventory", "installed_distributions", "failures",
    }
    if set(value) != required:
        raise ValueError(f"dependency evidence keys drift: {sorted(value)}")
    if value["schema"] != "microwakeword-reference-dependency-evidence-v1":
        raise ValueError("dependency evidence schema drift")
    if value["status"] != "EVIDENCE_COLLECTED_OWNER_REVIEW_REQUIRED":
        raise ValueError("dependency evidence collection status is not owner-review-required")
    if value["publication_permitted"] is not False or value["fixture_generation_permitted"] is not False:
        raise ValueError("dependency evidence permission flags drift")
    if value["owner_review_required"] is not True or value["failures"] != []:
        raise ValueError("dependency evidence is not a complete collection")
    if value["platform"] != {"system": "Linux", "machine": "x86_64", "python": "3.12"}:
        raise ValueError("dependency evidence platform drift")
    for key, expected_path, expected_sha in (
        ("project", "pyproject.toml", project_sha256),
        ("uv_lock", "uv.lock", lock_sha256),
    ):
        item = value[key]
        if not isinstance(item, dict) or item.get("path") != expected_path or item.get("sha256") != expected_sha:
            raise ValueError(f"dependency evidence {key} identity drift")
    expected = set(EXPECTED_VERSIONS) - {"vokra-microwakeword-reference"}
    inventory = value["installed_inventory"]
    if not isinstance(inventory, dict) or inventory.get("status") != "PASS" or inventory.get("failures") != []:
        raise ValueError("dependency evidence installed inventory is not complete")
    entries = inventory.get("entries")
    if not isinstance(entries, list) or len(entries) != len(expected):
        raise ValueError("dependency evidence installed inventory count drift")
    names: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("dependency evidence inventory entry is malformed")
        name = entry.get("normalized_name")
        if name not in expected or name in names or entry.get("status") != "VALID" or entry.get("failures") != []:
            raise ValueError(f"dependency evidence inventory row drift: {name!r}")
        if entry.get("version") != EXPECTED_VERSIONS[name]:
            raise ValueError(f"dependency evidence inventory version drift: {name}")
        metadata = entry.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("name") != [entry.get("name")] or metadata.get("version") != [entry.get("version")]:
            raise ValueError(f"dependency evidence inventory metadata drift: {name}")
        names.add(name)
    if names != expected:
        raise ValueError("dependency evidence installed inventory set drift")
    installed = value["installed_distributions"]
    if not isinstance(installed, list) or len(installed) != len(expected):
        raise ValueError("dependency evidence installed distribution count drift")
    installed_names = set()
    inventory_sha = inventory.get("sha256")
    if not isinstance(inventory_sha, str) or len(inventory_sha) != 64:
        raise ValueError("dependency evidence inventory digest malformed")
    for row in installed:
        if not isinstance(row, dict):
            raise ValueError("dependency evidence installed row is malformed")
        name = row.get("expected_name")
        if name not in expected or name in installed_names or row.get("expected_version") != EXPECTED_VERSIONS[name]:
            raise ValueError(f"dependency evidence installed row drift: {name!r}")
        if row.get("status") != "EVIDENCE_COLLECTED_OWNER_REVIEW_REQUIRED" or row.get("failures") != []:
            raise ValueError(f"dependency evidence installed row is not complete: {name}")
        if row.get("inventory_sha256") != inventory_sha:
            raise ValueError(f"dependency evidence inventory binding drift: {name}")
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"dependency evidence metadata missing: {name}")
        declarations = [metadata.get(key) for key in ("license", "license_expression", "license_file", "license_classifiers")]
        if not all(isinstance(item, list) for item in declarations):
            raise ValueError(f"dependency evidence metadata license fields malformed: {name}")
        record = row.get("record")
        if not isinstance(record, dict) or not isinstance(record.get("entries"), list) or not record["entries"]:
            raise ValueError(f"dependency evidence RECORD missing: {name}")
        for record_entry in record["entries"]:
            if not isinstance(record_entry, dict) or record_entry.get("validation") not in {"MATCH", "EMPTY_DECLARATION", "FAIL", "MALFORMED_ROW", "OVERSIZE"}:
                raise ValueError(f"dependency evidence RECORD row malformed: {name}")
        candidates = row.get("license_candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"dependency evidence license candidates malformed: {name}")
        if not candidates and not _metadata_license_declarations_present(metadata):
            raise ValueError(f"dependency evidence has neither license candidate nor metadata declarations: {name}")
        installed_names.add(name)
    if installed_names != expected:
        raise ValueError("dependency evidence installed distribution set drift")
    return {"status": "VALIDATED_PENDING_OWNER_REVIEW", "versions": {name: EXPECTED_VERSIONS[name] for name in sorted(expected)}}


def inspect_documents(
    project_bytes: bytes,
    lock_bytes: bytes,
    *,
    dependency_evidence: bytes | dict[str, Any] | None = None,
    enforce_file_digests: bool = True,
) -> dict[str, Any]:
    project = tomllib.loads(project_bytes.decode("utf-8"))
    lock = tomllib.loads(lock_bytes.decode("utf-8"))
    project_sha256 = sha256_bytes(project_bytes)
    lock_sha256 = sha256_bytes(lock_bytes)
    project_data = project.get("project", {})
    dependencies = {}
    failures: list[str] = []
    for item in project_data.get("dependencies", []):
        try:
            name, specifier = item.split("==", 1)
        except (AttributeError, ValueError):
            failures.append("malformed direct dependency declaration")
            continue
        dependencies[name] = f"=={specifier}"
    if dependencies != EXPECTED_DEPENDENCIES:
        failures.append("direct dependency set/version drift")
    if project_data.get("requires-python") != "==3.12.*":
        failures.append("Python 3.12 contract drift")
    packages = lock.get("package", [])
    names = [row.get("name") for row in packages if isinstance(row, dict)]
    duplicate_rows = sorted({name for name in names if names.count(name) > 1 and isinstance(name, str)})
    if duplicate_rows:
        failures.append(f"duplicate lock package rows: {duplicate_rows}")
    rows = {row.get("name"): row for row in packages if isinstance(row, dict) and isinstance(row.get("name"), str)}
    if set(rows) != EXPECTED_ROWS:
        failures.append(f"unknown/missing lock rows: {sorted(set(rows) ^ EXPECTED_ROWS)}")
    for name, version in EXPECTED_VERSIONS.items():
        if rows.get(name, {}).get("version") != version:
            failures.append(f"{name} version drift")
    if enforce_file_digests and project_sha256 != EXPECTED_PROJECT_SHA256:
        failures.append("pyproject digest drift")
    if enforce_file_digests and lock_sha256 != EXPECTED_LOCK_SHA256:
        failures.append("uv.lock digest drift")
    ai_edge = rows.get("ai-edge-litert", {})
    if len(ai_edge.get("wheels", [])) != 1 or _wheel_hashes(ai_edge) != [EXPECTED_AI_EDGE_WHEEL_SHA256]:
        failures.append("ai-edge-litert wheel identity drift")
    if any(row.get("version") == "2.2.0" for row in packages if isinstance(row, dict) and row.get("name") == "ai-edge-litert"):
        failures.append("REJECTED_AI_EDGE_LITERT_2.2.0_RESTRICTIVE_TERMS")
    numpy_wheels = rows.get("numpy", {}).get("wheels", [])
    if not any(wheel.get("hash") == f"sha256:{EXPECTED_NUMPY_WHEEL_SHA256}" for wheel in numpy_wheels):
        failures.append("numpy selected wheel identity drift")

    evidence_report: dict[str, Any] | None = None
    if dependency_evidence is None:
        failures.append("dependency evidence input is required")
        evidence_status = "MISSING"
    else:
        try:
            evidence_report = dependency_evidence if isinstance(dependency_evidence, dict) else _strict_json(dependency_evidence, "dependency evidence")
            evidence_contract = _validate_dependency_evidence(evidence_report, project_sha256, lock_sha256)
            evidence_status = evidence_contract["status"]
        except ValueError as error:
            failures.append(str(error))
            evidence_status = "INVALID"
    # No installed 2.1.5 evidence fingerprint is stamped in this phase. Even
    # a structurally valid report therefore remains pending owner review.
    failures.append("BLOCKED_PENDING_VAST_EVIDENCE: exact installed 2.1.5 fingerprints are not stamped")
    unreviewed = sorted(EXPECTED_ROWS - {"vokra-microwakeword-reference"})
    return {
        "schema": "microwakeword-reference-dependency-audit-v2",
        "status": "BLOCKED_PENDING_VAST_EVIDENCE",
        "publication_permitted": False,
        "fixture_generation_permitted": False,
        "project_sha256": project_sha256,
        "uv_lock_sha256": lock_sha256,
        "locked_rows": sorted(name for name in names if isinstance(name, str)),
        "expected_rows": sorted(EXPECTED_ROWS),
        "locked_row_count": len(packages),
        "duplicate_rows": duplicate_rows,
        "unreviewed_rows": unreviewed,
        "dependency_evidence_status": evidence_status,
        "primary_source_provenance": PRIMARY_SOURCE_PROVENANCE,
        "license_review": {
            "status": "BLOCKED_PENDING_VAST_EVIDENCE",
            "bounded_primary_source_evidence_recorded": evidence_status == "VALIDATED_PENDING_OWNER_REVIEW",
            "rows_requiring_owner_evidence": unreviewed,
        },
        "failures": failures,
    }


def self_test() -> int:
    root = Path(__file__).parent
    project_bytes = (root / "pyproject.toml").read_bytes()
    lock_bytes = (root / "uv.lock").read_bytes()
    report = inspect_documents(project_bytes, lock_bytes)
    assert report["status"] == "BLOCKED_PENDING_VAST_EVIDENCE"
    assert not report["fixture_generation_permitted"]
    assert report["dependency_evidence_status"] == "MISSING"
    assert "dependency evidence input is required" in report["failures"]
    assert _metadata_license_declarations_present(
        {"license": ["Apache-2.0"], "license_expression": [], "license_classifiers": []}
    )
    assert not _metadata_license_declarations_present(
        {"license": [], "license_expression": [], "license_classifiers": [], "license_file": ["LICENSE"]}
    )
    assert not _metadata_license_declarations_present(
        {"license": ["  UNKNOWN  "], "license_expression": [], "license_classifiers": []}
    )
    tampered = lock_bytes.replace(b'version = "2.1.5"', b'version = "9.9.9"', 1)
    tampered_report = inspect_documents(project_bytes, tampered, enforce_file_digests=False)
    assert tampered_report["status"] == "BLOCKED_PENDING_VAST_EVIDENCE"
    assert "ai-edge-litert version drift" in tampered_report["failures"]
    unknown = lock_bytes.replace(b'[[package]]\nname = "tqdm"', b'[[package]]\nname = "unknown-row"', 1)
    assert any("unknown/missing lock rows" in failure for failure in inspect_documents(project_bytes, unknown, enforce_file_digests=False)["failures"])
    duplicate = lock_bytes.replace(b'[[package]]\nname = "tqdm"', b'[[package]]\nname = "numpy"', 1)
    duplicate_report = inspect_documents(project_bytes, duplicate, enforce_file_digests=False)
    assert any("duplicate lock package rows" in failure for failure in duplicate_report["failures"])
    rejected = lock_bytes.replace(b'version = "2.1.5"', b'version = "2.2.0"', 1)
    rejected_report = inspect_documents(project_bytes, rejected, enforce_file_digests=False)
    assert "REJECTED_AI_EDGE_LITERT_2.2.0_RESTRICTIVE_TERMS" in rejected_report["failures"]
    assert rejected_report["status"] == "BLOCKED_PENDING_VAST_EVIDENCE"
    with tempfile.TemporaryDirectory(prefix="microwakeword-inspector-") as temporary:
        output = Path(temporary) / "audit.json"
        write_exclusive(output, "{}\n")
        try:
            write_exclusive(output, "{}\n")
        except SystemExit:
            pass
        else:
            raise AssertionError("inspector output clobber was accepted")
        target = Path(temporary) / "target.json"
        target.write_text("existing\n", encoding="utf-8")
        link = Path(temporary) / "output-link.json"
        try:
            link.symlink_to(target)
        except OSError:
            pass
        else:
            try:
                write_exclusive(link, "{}\n")
            except SystemExit:
                pass
            else:
                raise AssertionError("inspector symlink output was accepted")
    print("microWakeWord dependency inspector self-test: PASS (blocked fail-closed)", file=sys.stderr)
    return 0


def require_regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"input must be an existing regular non-symlink file: {path}")


def write_exclusive(path: Path, payload: str) -> None:
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise SystemExit(f"output parent must be an existing real directory: {parent}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise SystemExit(f"refusing to overwrite existing output: {path}") from error
    created = os.fstat(fd)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            fd = -1
            output.write(payload)
    except BaseException:
        # Never unlink a path that was replaced after our exclusive create.
        # lstat identity, not pathname equality, is the ownership proof.
        try:
            current = path.lstat()
        except FileNotFoundError:
            current = None
        if current is not None and (current.st_dev, current.st_ino) == (created.st_dev, created.st_ino):
            path.unlink()
        raise
    finally:
        if fd >= 0:
            os.close(fd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--project", type=Path, default=Path(__file__).parent / "pyproject.toml")
    parser.add_argument("--lock", type=Path, default=Path(__file__).parent / "uv.lock")
    parser.add_argument("--dependency-evidence", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if args.project != Path(__file__).parent / "pyproject.toml" or args.lock != Path(__file__).parent / "uv.lock" or args.dependency_evidence is not None or args.output is not None:
            parser.error("--self-test cannot be combined with paths/output")
        return self_test()
    require_regular_file(args.project)
    require_regular_file(args.lock)
    evidence_bytes = None
    if args.dependency_evidence is not None:
        require_regular_file(args.dependency_evidence)
        evidence_bytes = args.dependency_evidence.read_bytes()
    if args.output is not None and args.output.is_symlink():
        raise SystemExit(f"refusing symlink output: {args.output}")
    report = inspect_documents(args.project.read_bytes(), args.lock.read_bytes(), dependency_evidence=evidence_bytes)
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        write_exclusive(args.output, payload)
    print(payload, end="")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
