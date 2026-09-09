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
import re
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
EXPECTED_EVIDENCE_SHA256 = "2b24695d106665b5cbc17357b1a43ff03ab75235d35e7d3ed03e5c7c7a68069d"
EXPECTED_INVENTORY_SHA256 = "eeda7a48d6effc9e6b94a4bf806c319d8933f53bdd7e58c1c5b22afacecf4f21"
EXPECTED_LOCK_ROWS_SHA256 = "1f962ffccf851985838f2566831686399a7268001b62568eb38afeea39fa78e0"
VALIDATED_EVIDENCE_STATUS = "VALIDATED_EXACT_OWNER_REVIEWED"
EXPECTED_DISTRIBUTION_FINGERPRINTS = {
    "ai-edge-litert": {
        "metadata_sha256": "2ff89104841d22614e9db27451eaa5c877df0428183f02e64491db87a8dc112a",
        "record_sha256": "f09594d7fbd6a753e125185faf9a02e47c051b15c14ac5d0620472dc329954a9",
        "entries_sha256": "15887120bb4eaed216398b9cddc579b1cb2153f505c40a6c34967158d069dc59",
        "entries_count": 111,
        "license_sha256": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
        "native_count": 21,
        "native_sha256": "5922d4cc56d71193247216e1d8c4ec40bdc9738f5e8068fb63c945ebacfec244",
    },
    "backports-strenum": {
        "metadata_sha256": "00c9808e6b16b66f0800c99a7764ec880cc3e4213aefcc969549aa638dc2ceb1",
        "record_sha256": "361b5576025d5457163e8681298856d9cf9322c5223e06262a8cb360948173a5",
        "entries_sha256": "850121e98fdd524aad3770ede9b513a1aec08478e1f8b4be40ee3726da256a5a",
        "entries_count": 10,
        "license_sha256": "1456366f70983ea77b4a53983840e07a8a92ce8d42519a36c5233e1c5165957b",
        "native_count": 0,
        "native_sha256": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    },
    "flatbuffers": {
        "metadata_sha256": "9f38239f96faa21522a6aadab6912e1e560708bbe9c9f696a0679c9403fa089e",
        "record_sha256": "731215f6ccac6f388f03039973bdfcad16911e24175e0b8c42fd9f63c51d8e31",
        "entries_sha256": "a8bff3fe1dbd45dd79cca828ad0e0f90be3a3dd7c0728b5a200dac5f7353425a",
        "entries_count": 16,
        "license_sha256": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
        "native_count": 0,
        "native_sha256": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    },
    "numpy": {
        "metadata_sha256": "9a895d801184d2176f926af84eb47f2db1af2756afac99d64a2bf482d5d5e024",
        "record_sha256": "9481bebcff878734ecbd8aa6d9657ccd1cbc9954fdf33135b1c8bd4630a271bb",
        "entries_sha256": "618706ef4fa917e4bc9d21f9f79574d83ba082f1d7a4ee41af5a9bb815e3adfd",
        "entries_count": 930,
        "license_sha256": "b3b294667c6b6f0449db39bc4332791c9c255f6f7862a68703e91d48944e9602",
        "native_count": 22,
        "native_sha256": "ca57a96ef5b34726bb545370f58ddff03c0c80d30d4da3db63554f41a0cd2751",
    },
    "protobuf": {
        "metadata_sha256": "947ed3f132e70411555388d7fcd36712fdd60c8db6345001fb23bb56a2c6cab7",
        "record_sha256": "4216c5ad085514f7067e76acd5381281de47bab20e420b64b58b832e4718ce90",
        "entries_sha256": "39730d6b8d55dedd77700e952436f280922f31cdff6e36b5160c4da5c871f24f",
        "entries_count": 65,
        "license_sha256": "d3d3033edd4e8d71417e469dc7d152e16693d91ffe592d13f4901e4c3daabe6d",
        "native_count": 1,
        "native_sha256": "026517929474fae4094397e6d0d3691fb0dc7b3ec8342f929d26c4f2d532511f",
    },
    "tqdm": {
        "metadata_sha256": "0d95b85b90428f8776afc4a92b17c5094f78ee98d150b3255acbcdf4e9d57941",
        "record_sha256": "66e1f7728ce97510e82cb941b72db047488890892e1ab72fc3c0babf50a8fbe8",
        "entries_sha256": "1d71097811799bdab59f1deab773a921f3f54682abdffa234cfb3fb37c1dd83f",
        "entries_count": 42,
        "license_sha256": "7410801b25a32167dd9ac0ba9c56dc38d30be51246eaae50d0fef854cf66593c",
        "native_count": 0,
        "native_sha256": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    },
    "typing-extensions": {
        "metadata_sha256": "b05084ca1d50879865178d9fff9fabeab61bdfb1f361bfbde95421ffc8f9be46",
        "record_sha256": "9e07fea845cac76293d927afa7157a14ecba39b318bc8af236b8de530b7656c1",
        "entries_sha256": "af7eec5d405dd20cb258ea92fd0d5a855a4a2354102d25c499845fbd6c371f4f",
        "entries_count": 7,
        "license_sha256": "985806ae5d1e0989a43086db24bb63859ff3cc4381e878f004bb9815703cf433",
        "native_count": 0,
        "native_sha256": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    },
}

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
    "accepted": {
        "ai-edge-litert": {
            "source_tag": "v2.1.5",
            "source_commit": "9d26e89d88ef8785b6a1e54ec41ac8add215a125",
            "license_blob": "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64",
            "declared_license": "Apache-2.0",
            "caveat": "reference-only; google_tensor compiler source has Apache header; no TERMS path in the selected wheel",
        },
        "backports-strenum": {
            "source_commit": "9cb063cdc2d2e94229e7fc66c9a989379e1bff68",
            "license_blob": "f42f8adbed845d6a8c3cb07b12ffd186f6c23bc4",
            "declared_license": "MIT",
            "caveat": "MIT pyproject/metadata with bundled PSF license; reference-only",
        },
        "flatbuffers": {
            "source_commit": "7e163021e59cca4f8e1e35a7c828b5c6b7915953",
            "license_blob": "d645695673349e3947e8e5ae42332d0ac3164cd7",
            "declared_license": "Apache-2.0",
            "caveat": "reference-only",
        },
        "numpy": {
            "source_commit": "48fecee5453aa1d31e6b79dcb3969dc1a6d1a891",
            "license_blob": "f37a12cc4cccf83af4517809791777e71c1df2a9",
            "declared_license": "BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0",
            "caveat": "native OpenBLAS/GCC exception/libquadmath GPL/LGPL payloads are accepted only as unredistributed developer/reference dependency",
        },
        "protobuf": {
            "source_tag": "v36.1",
            "source_commit": "f377bfefc5e2cfab68b816903c25b23e091c439d",
            "license_blob": "19b305b00060a774a9180fb916c14b49edb2008f",
            "declared_license": "BSD-3-Clause",
            "caveat": "reference-only; not a runtime Cargo dependency",
        },
        "tqdm": {
            "source_commit": "96f2e60e4584cdab57a23602e27043d0465254ad",
            "license_blob": "194caf554f8f10ba4cac8a81b631a61d0d81f60d",
            "declared_license": "MPL-2.0 AND MIT",
            "caveat": "retain dual expression; do not simplify to MIT; reference-only",
        },
        "typing-extensions": {
            "source_commit": "f29cd28d8ed7642cafb1d18daf5aa41be6a5c0aa",
            "license_blob": "f26bcf4d2de6eb136e31006ca3ab447d5e488adf",
            "declared_license": "PSF-2.0",
            "caveat": "reference-only",
        },
    },
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha256_bytes(payload)


def _has_restrictive_terms_path(paths: list[str]) -> bool:
    markers = ("terms_of_use", "terms_and_conditions", "restrictive_terms", "restricted_terms")
    return any(
        any(marker in re.sub(r"[^a-z0-9]+", "_", path.casefold()) for marker in markers)
        for path in paths
    )


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
        expected_bytes = 364 if key == "project" else 6343
        if (
            not isinstance(item, dict)
            or item.get("path") != expected_path
            or item.get("bytes") != expected_bytes
            or item.get("sha256") != expected_sha
        ):
            raise ValueError(f"dependency evidence {key} identity drift")
    lock = value["lock"]
    if (
        not isinstance(lock, dict)
        or lock.get("selected_platform") != "Linux x86_64 CPython 3.12"
        or lock.get("resolution_markers") != ["platform_machine == 'x86_64' and sys_platform == 'linux'"]
        or lock.get("rows_sha256") != EXPECTED_LOCK_ROWS_SHA256
        or _canonical_json_sha256(lock.get("rows")) != EXPECTED_LOCK_ROWS_SHA256
    ):
        raise ValueError("dependency evidence lock fingerprint drift")
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
    if inventory_sha != EXPECTED_INVENTORY_SHA256:
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
        fingerprint = EXPECTED_DISTRIBUTION_FINGERPRINTS[name]
        if metadata.get("sha256") != fingerprint["metadata_sha256"]:
            raise ValueError(f"dependency evidence metadata fingerprint drift: {name}")
        declarations = [metadata.get(key) for key in ("license", "license_expression", "license_file", "license_classifiers")]
        if not all(isinstance(item, list) for item in declarations):
            raise ValueError(f"dependency evidence metadata license fields malformed: {name}")
        record = row.get("record")
        if not isinstance(record, dict) or not isinstance(record.get("entries"), list) or not record["entries"]:
            raise ValueError(f"dependency evidence RECORD missing: {name}")
        if (
            record.get("sha256") != fingerprint["record_sha256"]
            or record.get("entries_sha256") != fingerprint["entries_sha256"]
            or record.get("entries_count") != fingerprint["entries_count"]
            or _canonical_json_sha256(record["entries"]) != fingerprint["entries_sha256"]
        ):
            raise ValueError(f"dependency evidence RECORD fingerprint drift: {name}")
        empty_rows = []
        record_paths = []
        for record_entry in record["entries"]:
            if not isinstance(record_entry, dict):
                raise ValueError(f"dependency evidence RECORD row malformed: {name}")
            validation = record_entry.get("validation")
            if validation not in {"MATCH", "EMPTY_DECLARATION"}:
                raise ValueError(f"dependency evidence RECORD contains non-success row: {name}")
            declared = record_entry.get("declared")
            if not isinstance(declared, dict) or not isinstance(declared.get("path"), str):
                raise ValueError(f"dependency evidence RECORD path missing: {name}")
            record_paths.append(declared["path"])
            if validation == "EMPTY_DECLARATION":
                empty_rows.append(record_entry)
        expected_record_path = f"{row.get('dist_info')}/RECORD"
        if len(empty_rows) != 1 or empty_rows[0]["declared"]["path"] != expected_record_path:
            raise ValueError(f"dependency evidence RECORD self-row drift: {name}")
        self_declared = empty_rows[0]["declared"]
        if (
            self_declared.get("hash", {}).get("status") != "EMPTY"
            or self_declared.get("size", {}).get("status") != "EMPTY"
        ):
            raise ValueError(f"dependency evidence RECORD self-row declaration drift: {name}")
        candidates = row.get("license_candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"dependency evidence license candidates malformed: {name}")
        if _canonical_json_sha256(candidates) != fingerprint["license_sha256"]:
            raise ValueError(f"dependency evidence license fingerprint drift: {name}")
        candidate_paths = []
        for candidate in candidates:
            if not isinstance(candidate, dict) or not isinstance(candidate.get("path"), str):
                raise ValueError(f"dependency evidence license candidate malformed: {name}")
            candidate_paths.append(candidate["path"])
        if not candidates and not _metadata_license_declarations_present(metadata):
            raise ValueError(f"dependency evidence has neither license candidate nor metadata declarations: {name}")
        native = row.get("native_payloads")
        if not isinstance(native, list) or len(native) != fingerprint["native_count"]:
            raise ValueError(f"dependency evidence native payload count drift: {name}")
        if _canonical_json_sha256(native) != fingerprint["native_sha256"]:
            raise ValueError(f"dependency evidence native payload fingerprint drift: {name}")
        native_paths = []
        for item in native:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise ValueError(f"dependency evidence native payload malformed: {name}")
            native_paths.append(item["path"])
        if _has_restrictive_terms_path(record_paths + candidate_paths + native_paths):
            raise ValueError(f"dependency evidence contains restrictive TERMS_OF_USE path: {name}")
        installed_names.add(name)
    if installed_names != expected:
        raise ValueError("dependency evidence installed distribution set drift")
    return {"status": VALIDATED_EVIDENCE_STATUS, "versions": {name: EXPECTED_VERSIONS[name] for name in sorted(expected)}}


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
            if not isinstance(dependency_evidence, bytes):
                raise ValueError("dependency evidence raw bytes are required for exact fingerprint validation")
            if sha256_bytes(dependency_evidence) != EXPECTED_EVIDENCE_SHA256:
                raise ValueError("dependency evidence raw fingerprint drift")
            evidence_report = _strict_json(dependency_evidence, "dependency evidence")
            evidence_contract = _validate_dependency_evidence(evidence_report, project_sha256, lock_sha256)
            evidence_status = evidence_contract["status"]
        except ValueError as error:
            failures.append(str(error))
            evidence_status = "INVALID"
    if evidence_status != VALIDATED_EVIDENCE_STATUS:
        failures.append("BLOCKED_PENDING_VAST_EVIDENCE: exact installed 2.1.5 evidence is required")
    valid = not failures and evidence_status == VALIDATED_EVIDENCE_STATUS
    status = "PASS" if valid else "BLOCKED_PENDING_VAST_EVIDENCE"
    unreviewed = [] if valid else sorted(EXPECTED_ROWS - {"vokra-microwakeword-reference"})
    return {
        "schema": "microwakeword-reference-dependency-audit-v2",
        "status": status,
        "publication_permitted": False,
        "fixture_generation_permitted": valid,
        "project_sha256": project_sha256,
        "uv_lock_sha256": lock_sha256,
        "locked_rows": sorted(name for name in names if isinstance(name, str)),
        "expected_rows": sorted(EXPECTED_ROWS),
        "locked_row_count": len(packages),
        "duplicate_rows": duplicate_rows,
        "unreviewed_rows": unreviewed,
        "dependency_evidence_status": evidence_status,
        "dependency_evidence_sha256": (
            sha256_bytes(dependency_evidence) if isinstance(dependency_evidence, bytes) else None
        ),
        "primary_source_provenance": PRIMARY_SOURCE_PROVENANCE,
        "license_review": {
            "status": status,
            "bounded_primary_source_evidence_recorded": evidence_status == VALIDATED_EVIDENCE_STATUS,
            "rows_requiring_owner_evidence": unreviewed,
        },
        "failures": failures,
    }


def _authoritative_negative_self_tests(evidence_bytes: bytes) -> None:
    """Exercise phase-2 evidence fingerprints without importing dependencies."""
    evidence = _strict_json(evidence_bytes, "authoritative dependency evidence")
    _validate_dependency_evidence(evidence, EXPECTED_PROJECT_SHA256, EXPECTED_LOCK_SHA256)
    mutations: list[tuple[str, Any]] = []
    metadata_tamper = json.loads(json.dumps(evidence))
    metadata_tamper["installed_distributions"][0]["metadata"]["sha256"] = "0" * 64
    mutations.append(("metadata", metadata_tamper))
    record_tamper = json.loads(json.dumps(evidence))
    record_tamper["installed_distributions"][0]["record"]["entries"][0]["validation"] = "FAIL"
    mutations.append(("RECORD", record_tamper))
    license_tamper = json.loads(json.dumps(evidence))
    license_tamper["installed_distributions"][0]["license_candidates"] = [
        {"path": "LICENSE", "bytes": 1, "sha256": "0" * 64}
    ]
    mutations.append(("license candidate", license_tamper))
    native_tamper = json.loads(json.dumps(evidence))
    native_tamper["installed_distributions"][0]["native_payloads"][0]["path"] = "tampered.so"
    mutations.append(("native payload", native_tamper))
    terms_tamper = json.loads(json.dumps(evidence))
    terms_tamper["installed_distributions"][0]["record"]["entries"][0]["declared"]["path"] = "TERMS_OF_USE"
    mutations.append(("restrictive terms", terms_tamper))
    for label, mutated in mutations:
        try:
            _validate_dependency_evidence(mutated, EXPECTED_PROJECT_SHA256, EXPECTED_LOCK_SHA256)
        except ValueError:
            continue
        raise AssertionError(f"tampered {label} evidence was accepted")
    assert _has_restrictive_terms_path(["package/TERMS_OF_USE"])


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
    evidence_path_value = os.environ.get("VOKRA_MWW_REFERENCE_EVIDENCE")
    if evidence_path_value:
        evidence_path = Path(evidence_path_value)
        if evidence_path.is_symlink() or not evidence_path.is_file():
            raise AssertionError(f"configured evidence self-test path is not a regular file: {evidence_path}")
        evidence_bytes = evidence_path.read_bytes()
        exact_report = inspect_documents(project_bytes, lock_bytes, dependency_evidence=evidence_bytes)
        assert exact_report["status"] == "PASS"
        assert exact_report["fixture_generation_permitted"] is True
        assert exact_report["publication_permitted"] is False
        assert exact_report["failures"] == []
        assert exact_report["dependency_evidence_status"] == VALIDATED_EVIDENCE_STATUS
        _authoritative_negative_self_tests(evidence_bytes)
    for spelling in ("TERMS-OF-USE", "terms_and_conditions", "restrictive_terms", "restricted-terms"):
        assert _has_restrictive_terms_path([f"package/{spelling}"])
    assert not _has_restrictive_terms_path(["package/terms.py", "ordinary_dependency"])
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
