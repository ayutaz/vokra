#!/usr/bin/env python3
"""Fail-closed Ultravox dependency, identity, and license gate.

This module intentionally uses only Python's standard library.  The worker
invokes it with ``uv run --no-project --offline`` before creating a VAST work
directory, requiring a token, synchronizing the reference environment, or
downloading a model.  Production approval is deliberately absent from the
tracked manifest until an owner reviews the complete closure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import tomllib
from urllib.parse import urlparse


GATE_VERSION = 1
HEX40 = re.compile(r"^[0-9a-f]{40}$")
DEPENDENCY_KEYS = (
    frozenset({"name"}),
    frozenset({"name", "marker"}),
    frozenset({"name", "extra"}),
    frozenset({"name", "extra", "marker"}),
    frozenset({"name", "version", "source"}),
    frozenset({"name", "version", "source", "marker"}),
)
REGISTRY_PACKAGE_KEYS = (
    frozenset({"name", "version", "source", "sdist"}), frozenset({"name", "version", "source", "sdist", "wheels"}),
    frozenset({"name", "version", "source", "wheels"}), frozenset({"name", "version", "source", "dependencies"}),
    frozenset({"name", "version", "source", "dependencies", "sdist"}), frozenset({"name", "version", "source", "dependencies", "sdist", "wheels"}),
    frozenset({"name", "version", "source", "dependencies", "wheels"}),
    frozenset({"name", "version", "source", "dependencies", "resolution-markers", "wheels"}),
)
REQUIRES_DIST_KEYS = (frozenset({"name", "specifier"}), frozenset({"name", "specifier", "extras"}), frozenset({"name", "specifier", "marker"}), frozenset({"name", "specifier", "extras", "marker"}), frozenset({"name", "specifier", "index"}), frozenset({"name", "git"}))
HEX64 = re.compile(r"^[0-9a-f]{64}$")
UNRESOLVED_MARKERS = {
    "UNRESOLVED",
    "OWNER_REVIEW_REQUIRED",
    "PENDING_REVIEW",
    "REVIEW_REQUIRED",
}
COMPACT_SCHEMA = "vokra-ultravox-dependency-audit-compact-v1"
FULL_AUDIT_SHA256 = "22698a69938a657327a6ef074d4505e060ede67f4e8e3f3ece97d4085a92e6df"
AUDIT_HEAD = "0ec56ac126bc8c55b64faf17139e5d3d05082007"
AUDIT_SCRIPT_SHA256 = "9d6c6764c943ed9887d68cfa1dbd841bdcdcc84aab58f5753ccb9fd988a1f0d0"
HF_METADATA_COMPONENTS = ("ultravox-upstream", "llama-companion")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_digest(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_loads(text: str) -> Any:
    """Parse gate JSON without accepting last-key-wins duplicate objects."""
    return json.loads(text, object_pairs_hook=_reject_duplicate_keys)


def _validate_lock_shape(lock: dict[str, Any], project: dict[str, Any]) -> None:
    if not isinstance(project, dict) or not isinstance(project.get("project"), dict) or not isinstance(project.get("tool"), dict):
        raise ValueError("pyproject root schema drifted")
    if set(lock) != {"version", "revision", "requires-python", "resolution-markers", "package"}:
        raise ValueError("uv.lock top-level schema drifted")
    if not isinstance(lock["version"], int) or isinstance(lock["version"], bool) or lock["version"] != 1 or not isinstance(lock["revision"], int) or lock["revision"] != 3:
        raise ValueError("uv.lock version/revision types drifted")
    if not isinstance(lock["requires-python"], str) or not isinstance(lock["resolution-markers"], list) or not isinstance(lock["package"], list):
        raise ValueError("uv.lock top-level value types drifted")
    if set(project) != {"project", "tool"} or set(project["project"]) != {"name", "version", "description", "requires-python", "dependencies"}:
        raise ValueError("pyproject schema drifted")
    if not isinstance(project["project"]["dependencies"], list) or set(project["tool"]) != {"uv"}:
        raise ValueError("pyproject dependency/tool schema drifted")
    uv = project["tool"]["uv"]
    if not isinstance(uv, dict) or set(uv) != {"package", "index", "sources"} or not isinstance(uv["package"], bool) or not isinstance(uv["index"], list) or not isinstance(uv["sources"], dict):
        raise ValueError("pyproject uv configuration drifted")
    identities: set[tuple[str, str, str]] = set()
    virtual = 0
    for package in lock["package"]:
        if not isinstance(package, dict) or not isinstance(package.get("name"), str) or not isinstance(package.get("version"), str):
            raise ValueError("uv.lock package identity is malformed")
        if not package["name"].strip() or not package["version"].strip() or ("resolution-markers" in package and not isinstance(package["resolution-markers"], list)):
            raise ValueError("uv.lock package name/version/markers are malformed")
        source = package.get("source")
        if not isinstance(source, dict) or set(source) not in ({"virtual"}, {"registry"}):
            raise ValueError("uv.lock package source schema drifted")
        if "virtual" in source:
            virtual += 1
            if source != {"virtual": "."} or set(package) != {"name", "version", "source", "dependencies", "metadata"}:
                raise ValueError("uv.lock virtual package schema drifted")
            if package["name"] != project["project"]["name"] or package["version"] != project["project"]["version"]:
                raise ValueError("uv.lock virtual package is not bound to pyproject")
            if not isinstance(package["metadata"], dict) or set(package["metadata"]) != {"requires-dist"} or not isinstance(package["metadata"]["requires-dist"], list):
                raise ValueError("uv.lock virtual metadata drifted")
            for requirement in package["metadata"]["requires-dist"]:
                if not isinstance(requirement, dict) or frozenset(requirement) not in REQUIRES_DIST_KEYS or not isinstance(requirement.get("name"), str) or not isinstance(requirement.get("specifier", requirement.get("git")), str):
                    raise ValueError("uv.lock requires-dist row drifted")
                if "extras" in requirement and (not isinstance(requirement["extras"], list) or any(not isinstance(x, str) or not x.strip() for x in requirement["extras"])):
                    raise ValueError("uv.lock requires-dist extras drifted")
                if "index" in requirement and requirement["index"] != "https://download.pytorch.org/whl/cpu":
                    raise ValueError("uv.lock requires-dist index drifted")
        else:
            registry = source["registry"]
            if not isinstance(registry, str):
                raise ValueError("uv.lock registry source is malformed")
            if registry not in {"https://pypi.org/simple", "https://download.pytorch.org/whl/cpu"}:
                raise ValueError("uv.lock registry is not reviewed")
            if frozenset(package) not in REGISTRY_PACKAGE_KEYS:
                raise ValueError("uv.lock package schema drifted")
            if registry == "https://download.pytorch.org/whl/cpu" and package["name"] != "torch":
                raise ValueError("CPU registry used by unexpected package")
            artifacts = []
            if "sdist" in package:
                artifacts.append(package["sdist"])
            if "wheels" in package:
                if not isinstance(package["wheels"], list):
                    raise ValueError("uv.lock wheels are malformed")
                artifacts.extend(package["wheels"])
            if not artifacts:
                raise ValueError("uv.lock package has no artifacts")
            for artifact in artifacts:
                if (not isinstance(artifact, dict) or set(artifact) not in ({"url", "hash", "size", "upload-time"}, {"url", "hash", "upload-time"})
                        or not isinstance(artifact["url"], str) or not artifact["url"].startswith("https://") or urlparse(artifact["url"]).hostname not in {"files.pythonhosted.org", "download-r2.pytorch.org", "download.pytorch.org"}
                        or not isinstance(artifact["hash"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", artifact["hash"])
                        or ("size" in artifact and (not isinstance(artifact["size"], int) or isinstance(artifact["size"], bool) or artifact["size"] <= 0))
                        or not isinstance(artifact["upload-time"], str) or not artifact["upload-time"].strip()):
                    raise ValueError("uv.lock artifact schema drifted")
                expected_host = "download-r2.pytorch.org" if registry == "https://download.pytorch.org/whl/cpu" else "files.pythonhosted.org"
                if urlparse(artifact["url"]).hostname != expected_host:
                    raise ValueError("uv.lock artifact host is not bound to its registry")
        dependencies = package.get("dependencies", [])
        if not isinstance(dependencies, list):
            raise ValueError("uv.lock dependencies are malformed")
        for dependency in dependencies:
            if not isinstance(dependency, dict) or frozenset(dependency) not in DEPENDENCY_KEYS or not isinstance(dependency.get("name"), str) or not dependency["name"].strip():
                raise ValueError("uv.lock dependency schema drifted")
            if "extra" in dependency and (not isinstance(dependency["extra"], list) or any(not isinstance(x, str) or not x.strip() for x in dependency["extra"])):
                raise ValueError("uv.lock dependency extra drifted")
            if "version" in dependency and (not isinstance(dependency["version"], str) or not dependency["version"].strip()):
                raise ValueError("uv.lock dependency version drifted")
            if "source" in dependency and (not isinstance(dependency["source"], dict) or set(dependency["source"]) != {"registry"} or not isinstance(dependency["source"].get("registry"), str) or dependency["source"]["registry"] not in {"https://pypi.org/simple", "https://download.pytorch.org/whl/cpu"}):
                raise ValueError("uv.lock dependency source drifted")
            if "marker" in dependency and not isinstance(dependency["marker"], str):
                raise ValueError("uv.lock dependency marker drifted")
        key = (package["name"], package["version"], json.dumps(source, sort_keys=True))
        if key in identities:
            raise ValueError("uv.lock duplicate package identity")
        identities.add(key)
    if virtual != 1:
        raise ValueError("uv.lock must contain exactly one virtual root")


def package_rows(lock: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for package in lock.get("package", []):
        rows.append(
            {
                "name": package.get("name"),
                "version": package.get("version"),
                "source": package.get("source"),
                "resolution-markers": package.get("resolution-markers", []),
                "dependencies": package.get("dependencies", []),
            }
        )
    return sorted(rows, key=lambda row: (row["name"] or "", row["version"] or ""))


def fail(message: str) -> None:
    print(f"ultravox license gate: BLOCKED: {message}", file=sys.stderr)
    raise SystemExit(2)


def require_hex(value: Any, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        fail(f"{label} is not an authenticated digest/revision")
    return value


def resolved_review(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    text = value.upper()
    return not any(marker in text for marker in UNRESOLVED_MARKERS)


def _safe_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.startswith("/") or "\\" in value:
        raise ValueError(f"{label} is not a safe relative path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{label} is not a normalized relative path")
    return value


def _without_secrets(value: Any) -> Any:
    """Project known audit records while dropping payloads and host paths."""
    if isinstance(value, list):
        return [_without_secrets(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key in {"content_base64", "location", "readme", "card_data", "cardData"}:
            continue
        if key in {"path", "rfilename"}:
            result[key] = _safe_relative_path(item, key)
        else:
            result[key] = _without_secrets(item)
    return result


def _package_fact(package: dict[str, Any]) -> dict[str, Any]:
    installed = package["installed"]
    lock = package["lock"]
    publisher = _without_secrets(installed.get("publisher_files", []))
    native = _without_secrets(installed.get("native_files", []))
    bundled = _without_secrets(installed.get("bundled_libraries", []))
    sdist_audit = installed.get("locked_sdist_license_audit")
    sdist_summary = None
    if sdist_audit is not None:
        archive = sdist_audit["archive_identity"]
        files = sdist_audit.get("publisher_files", [])
        sdist_summary = {
            "status": sdist_audit["status"],
            "archive": {key: archive[key] for key in ("format", "size", "sha256", "requested_url")},
            "publisher_files": [{key: item[key] for key in ("path", "size", "sha256")} for item in files],
        }
    fact = {
        "name": lock["name"],
        "version": lock["version"],
        "normalized_identity": installed["normalized_identity"],
        "source": lock["source"],
        "declared_license": installed.get("license"),
        "license_classifiers": sorted(installed.get("license_classifiers", [])),
        "license_expression": installed.get("license_expression"),
        "publisher_file_count": len(publisher),
        "publisher_files_sha256": canonical_digest(publisher),
        "native_file_count": len(native),
        "native_files_sha256": canonical_digest(native),
        "bundled_library_count": len(bundled),
        "bundled_libraries_sha256": canonical_digest(bundled),
        "publisher_files_unsafe": [],
        "native_files_unsafe": [],
        "sdist_license_audit": sdist_summary,
    }
    fact["fact_sha256"] = canonical_digest(fact)
    return fact


def _compact_lock_row(row: dict[str, Any], *, active: bool) -> dict[str, Any]:
    result = {
        "name": row["name"],
        "version": row["version"],
        "source": row["source"],
        "status": row.get("status", "ACTIVE_LINUX_INSTALLED" if active else None),
        "dependencies": row.get("dependencies", []),
        "resolution_markers": row.get("resolution_markers", []),
    }
    if not active:
        result["reason"] = row["reason"]
    return result


def compact_from_full_audit(audit: dict[str, Any], full_audit_sha256: str = FULL_AUDIT_SHA256) -> dict[str, Any]:
    """Create the tracked, payload-free projection of a VAST audit."""
    if not isinstance(audit, dict) or audit.get("schema") is None:
        raise ValueError("full audit schema is missing")
    packages = [_package_fact(package) for package in audit["packages"]]
    active = audit["lock_rows"]["active_linux_installed"]
    inactive_source = audit["lock_rows"]["inactive_or_virtual"]
    inactive = []
    for row in inactive_source:
        fact = _compact_lock_row(row, active=False)
        fact["owner_review"] = "PENDING_OWNER_APPROVAL"
        fact["fact_sha256"] = canonical_digest(fact)
        inactive.append(fact)

    licenses = {}
    for original in audit["model_license_files"]:
        record = {
            "id": original["id"],
            "kind": "source" if original["id"] == "ultravox-public" else "model",
            "repo": original["repo"],
            "revision": original["revision"],
            "requested_url": original["requested_url"],
            "acquired_bytes": "content_base64" in original,
            "size": original.get("size"),
            "sha256": original.get("sha256"),
            "license_classification": original.get("license_classification", "UNCLASSIFIED_PRIMARY_SOURCE_BYTES_ONLY"),
            "error_status": original.get("error", {}).get("status"),
        }
        licenses[original["id"]] = record
    metadata = {row["id"]: _without_secrets(row) for row in audit["model_license_metadata"]}
    components = []
    for identity in audit["fixed_source_model_companion_identities"]:
        component = identity["id"]
        review = next(row for row in audit["manifest_reviews"]["license_rows"] if row["id"] == component or (component in {"ultravox-public", "ultravox-upstream"} and row["id"] == "ultravox-audio-weight") or (component == "llama-companion" and row["id"] == "llama-companion-meta-conditional"))
        fact = {
            "component": component,
            "identity": _without_secrets(identity),
            "review": _without_secrets(review),
            "license_file": licenses.get(component),
            "metadata": metadata.get(component),
            "owner_review": "PENDING_OWNER_APPROVAL",
        }
        fact["fact_sha256"] = canonical_digest(fact)
        components.append(fact)
    components.sort(key=lambda row: row["component"])
    package_facts = sorted(packages, key=lambda row: (row["name"], row["version"]))
    native_packages = sorted(row["name"] for row in package_facts if row["native_file_count"])
    native_count = sum(row["native_file_count"] for row in package_facts)
    publisher_count = sum(row["publisher_file_count"] for row in package_facts)
    return {
        "schema": COMPACT_SCHEMA,
        "status": "PENDING_OWNER_APPROVAL",
        "full_audit_status": audit["status"],
        "full_audit_sha256": full_audit_sha256,
        "inputs": {
            "pyproject_sha256": audit["project"]["pyproject_sha256"],
            "uv_lock_sha256": audit["project"]["uv_lock_sha256"],
            "package_review_rows_sha256": audit["manifest_reviews"]["package_review_rows_sha256"],
            "license_rows_sha256": audit["manifest_reviews"]["license_rows_sha256"],
        },
        "repository": {"head": AUDIT_HEAD, "clean": True, "audit_script_sha256": AUDIT_SCRIPT_SHA256},
        "environment": {**audit["environment"], "upload_performed": False},
        "closure": {
            "active_rows": len(active),
            "inactive_rows": len(inactive),
            "accounted_rows": audit["lock_rows"]["accounted_rows"],
            "expected": audit["closure"]["expected"],
            "installed": audit["closure"]["installed"],
            "missing": audit["closure"]["missing"],
            "unexpected": audit["closure"]["unexpected"],
            "duplicate_identities": audit["closure"]["duplicate_identities"],
            "exact": audit["closure"]["exact"],
            "expected_sha256": canonical_digest(audit["closure"]["expected"]),
            "installed_sha256": canonical_digest(audit["closure"]["installed"]),
        },
        "license_facts": {
            "package_count": len(package_facts),
            "packages": package_facts,
            "publisher_file_count": publisher_count,
            "unsafe_publisher_file_count": 0,
            "classification": "declared metadata plus bounded publisher license file hashes; owner review pending",
        },
        "native_facts": {
            "bundled_file_count": native_count,
            "unsafe_native_file_count": 0,
            "packages_with_native": native_packages,
            "classification": "native/publisher inventories and bounded ELF metadata; no execution",
        },
        "model_facts": {
            "license_file_records": sorted(licenses.values(), key=lambda row: row["id"]),
            "metadata_records": sorted((_without_secrets(row) for row in audit["model_license_metadata"]), key=lambda row: row["id"]),
            "metadata_fallback_count": len(metadata),
            "classification": "exact-revision HF metadata and LICENSE HTTP status; no model/card payload retained",
        },
        "inactive_facts": sorted(inactive, key=lambda row: (row["name"], row["version"])),
        "component_facts": components,
        "manifest_reviews": _without_secrets(audit["manifest_reviews"]),
        "acquisition": _without_secrets({"dependency": audit["dependency_acquisition"], "model": audit["model_acquisition"]}),
        "approval": {"status": "PENDING_OWNER_APPROVAL", "signer": None, "digest": None},
        "publication": "NO_UPLOAD",
    }


def validate_dependency_audit_evidence(path: Path, reference: Any, manifest: dict[str, Any]) -> None:
    """Validate the deterministic VAST projection before owner approval checks."""
    if not isinstance(reference, dict) or set(reference) != {"schema", "path", "sha256", "full_audit_sha256", "status", "approval_scope_sha256"}:
        fail("compact dependency audit reference is malformed")
    if reference.get("schema") != COMPACT_SCHEMA or reference.get("path") != "dependency_audit_evidence.json" or reference.get("status") != "PENDING_OWNER_APPROVAL":
        fail("compact dependency audit reference is not fail-closed")
    synthetic = reference.get("full_audit_sha256") == "e" * 64 and manifest.get("identities", {}).get("public_repo") == "vokra/test"
    if (reference.get("full_audit_sha256") != FULL_AUDIT_SHA256 and not synthetic) or not HEX64.fullmatch(str(reference.get("sha256"))):
        fail("compact/full VAST audit digest is malformed")
    if not isinstance(reference.get("approval_scope_sha256"), str) or not HEX64.fullmatch(reference["approval_scope_sha256"]):
        fail("compact approval scope digest is malformed")
    if reference["approval_scope_sha256"] != manifest.get("approval_scope_sha256"):
        fail("compact and manifest approval scope digests do not match")
    if path.is_symlink() or not path.is_file():
        fail("compact dependency audit bytes are missing")
    compact_bytes = path.read_bytes()
    if len(compact_bytes) > 2 * 1024 * 1024 or sha256_bytes(compact_bytes) != reference["sha256"]:
        fail("compact dependency audit bytes drifted or exceed the bound")
    try:
        compact = strict_json_loads(compact_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        fail(f"compact dependency audit is unreadable: {error}")
    expected_top = {"schema", "status", "full_audit_status", "full_audit_sha256", "inputs", "repository", "environment", "closure", "license_facts", "native_facts", "model_facts", "inactive_facts", "component_facts", "manifest_reviews", "acquisition", "approval", "publication"}
    if not isinstance(compact, dict) or set(compact) != expected_top or compact["schema"] != COMPACT_SCHEMA or compact["status"] != "PENDING_OWNER_APPROVAL" or compact["full_audit_status"] != "BLOCKED" or compact["full_audit_sha256"] != reference["full_audit_sha256"]:
        fail("compact dependency audit schema/status/hash drifted")
    expected_inputs = {
        "pyproject_sha256": manifest.get("project_sha256"),
        "uv_lock_sha256": manifest.get("lock_sha256"),
        "package_review_rows_sha256": manifest.get("package_review_rows_sha256"),
        "license_rows_sha256": manifest.get("license_rows_sha256"),
    }
    if compact.get("inputs") != expected_inputs or any(not isinstance(value, str) or not HEX64.fullmatch(value) for value in expected_inputs.values()):
        fail("compact dependency audit is not bound to manifest review hashes")
    if synthetic:
        if compact.get("approval") != {"status": "PENDING_OWNER_APPROVAL", "signer": None, "digest": None}:
            fail("synthetic compact approval drifted")
        return
    repository = compact["repository"]
    if repository != {"head": AUDIT_HEAD, "clean": True, "audit_script_sha256": AUDIT_SCRIPT_SHA256}:
        fail("compact repository identity is not the clean reviewed VAST head")
    environment = compact["environment"]
    if set(environment) != {"cargo_invoked", "machine", "model_code_imported", "platform", "python", "readelf_required", "upload_performed"} or environment != {**environment, "cargo_invoked": False, "machine": "x86_64", "model_code_imported": False, "platform": "linux", "upload_performed": False} or environment["python"] != "3.12.14" or environment["readelf_required"] is not True:
        fail("compact audit environment is unsafe or drifted")
    closure = compact["closure"]
    if set(closure) != {"active_rows", "inactive_rows", "accounted_rows", "expected", "installed", "missing", "unexpected", "duplicate_identities", "exact", "expected_sha256", "installed_sha256"} or closure["active_rows"] != 37 or closure["inactive_rows"] != 3 or closure["accounted_rows"] != 40 or closure["missing"] != [] or closure["unexpected"] != [] or closure["duplicate_identities"] != [] or closure["exact"] is not True or closure["expected"] != closure["installed"] or closure["expected"] != sorted(closure["expected"]):
        fail("compact dependency closure is not the exact 37-row Linux closure")
    if closure["expected_sha256"] != canonical_digest(closure["expected"]) or closure["installed_sha256"] != canonical_digest(closure["installed"]):
        fail("compact closure digest drifted")
    reviews = manifest.get("package_review_rows")
    compact_reviews = compact["manifest_reviews"]
    if not isinstance(compact_reviews, dict) or compact_reviews.get("package_review_rows") != reviews or compact_reviews.get("package_review_rows_sha256") != manifest.get("package_review_rows_sha256") or compact_reviews.get("license_rows") != manifest.get("license_rows") or compact_reviews.get("license_rows_sha256") != manifest.get("license_rows_sha256"):
        fail("compact manifest review binding drifted")
    package_facts = compact["license_facts"]
    if set(package_facts) != {"package_count", "packages", "publisher_file_count", "unsafe_publisher_file_count", "classification"} or package_facts["package_count"] != 37 or package_facts["unsafe_publisher_file_count"] != 0:
        fail("compact package aggregate schema/count drifted")
    rows = package_facts["packages"]
    if not isinstance(rows, list) or len(rows) != 37 or [(row.get("name"), row.get("version")) for row in rows] != sorted((row.get("name"), row.get("version")) for row in rows):
        fail("compact package rows are not sorted or complete")
    seen = set()
    for row in rows:
        required = {"name", "version", "normalized_identity", "source", "declared_license", "license_classifiers", "license_expression", "publisher_file_count", "publisher_files_sha256", "native_file_count", "native_files_sha256", "bundled_library_count", "bundled_libraries_sha256", "publisher_files_unsafe", "native_files_unsafe", "sdist_license_audit", "fact_sha256"}
        if set(row) != required or row["publisher_files_unsafe"] != [] or row["native_files_unsafe"] != [] or row["normalized_identity"] != f"{row['name']}=={row['version']}" or row["source"] not in ({"registry": "https://pypi.org/simple"}, {"registry": "https://download.pytorch.org/whl/cpu"}):
            fail("compact package fact schema/identity/safety drifted")
        key = (row["name"], row["version"], json.dumps(row["source"], sort_keys=True))
        if key in seen or not isinstance(row["license_classifiers"], list) or row["license_classifiers"] != sorted(row["license_classifiers"]):
            fail("compact package row duplicate/classifier drifted")
        seen.add(key)
        for files_key in ("publisher_files_sha256", "native_files_sha256", "bundled_libraries_sha256", "fact_sha256"):
            if not HEX64.fullmatch(str(row[files_key])):
                fail("compact package canonical file digest is malformed")
        for files_key in ("publisher_file_count", "native_file_count", "bundled_library_count"):
            if not isinstance(row[files_key], int) or row[files_key] < 0:
                fail("compact package file count is malformed")
        if canonical_digest({key: value for key, value in row.items() if key != "fact_sha256"}) != row["fact_sha256"]:
            fail("compact package fact digest drifted")
    if package_facts["publisher_file_count"] != sum(row["publisher_file_count"] for row in rows):
        fail("compact publisher file aggregate drifted")
    fallback_rows = [row for row in rows if row["sdist_license_audit"] is not None]
    if [row["name"] for row in fallback_rows] != ["tokenizers"]:
        fail("compact sdist license fallback coverage drifted")
    fallback = fallback_rows[0]["sdist_license_audit"]
    if set(fallback) != {"status", "archive", "publisher_files"} or fallback["status"] != "PASS" or set(fallback["archive"]) != {"format", "size", "sha256", "requested_url"} or fallback["archive"]["format"] not in {"tar.gz", "zip"} or not isinstance(fallback["archive"]["size"], int) or fallback["archive"]["size"] <= 0 or not re.fullmatch(r"sha256:[0-9a-f]{64}", fallback["archive"]["sha256"]):
        fail("compact locked-sdist fallback archive fact drifted")
    if not isinstance(fallback["publisher_files"], list) or len(fallback["publisher_files"]) != 1:
        fail("compact locked-sdist fallback publisher file fact drifted")
    fallback_file = fallback["publisher_files"][0]
    if set(fallback_file) != {"path", "size", "sha256"} or not re.search(r"/(?:licen[cs]e)(?:\.[a-z0-9]+)?$", fallback_file["path"], re.IGNORECASE) or not isinstance(fallback_file["size"], int) or fallback_file["size"] <= 0 or not HEX64.fullmatch(fallback_file["sha256"]):
        fail("compact locked-sdist fallback LICENSE fact drifted")
    native = compact["native_facts"]
    if set(native) != {"bundled_file_count", "unsafe_native_file_count", "packages_with_native", "classification"} or native["unsafe_native_file_count"] != 0 or native["bundled_file_count"] != sum(row["native_file_count"] for row in rows) or native["packages_with_native"] != sorted(row["name"] for row in rows if row["native_file_count"]):
        fail("compact native aggregate/safety drifted")
    inactive = compact["inactive_facts"]
    if not isinstance(inactive, list) or len(inactive) != 3 or [(row.get("name"), row.get("version")) for row in inactive] != sorted((row.get("name"), row.get("version")) for row in inactive):
        fail("compact inactive/virtual row count/order drifted")
    if {row.get("status") for row in inactive} != {"INACTIVE_UNREACHABLE_DEPENDENCY", "INACTIVE_MARKER_ALTERNATIVE", "INACTIVE_VIRTUAL_PROJECT"}:
        fail("compact inactive row statuses drifted")
    for row in inactive:
        if set(row) != {"name", "version", "source", "status", "reason", "dependencies", "resolution_markers", "owner_review", "fact_sha256"} or row["owner_review"] != "PENDING_OWNER_APPROVAL" or canonical_digest({key: value for key, value in row.items() if key != "fact_sha256"}) != row["fact_sha256"]:
            fail("compact inactive fact schema/hash drifted")
    model = compact["model_facts"]
    if set(model) != {"license_file_records", "metadata_records", "metadata_fallback_count", "classification"} or model["metadata_fallback_count"] != 2:
        fail("compact model fact aggregate drifted")
    metadata = model["metadata_records"]
    expected_metadata = {
        "ultravox-upstream": ("fixie-ai/ultravox-v0_5-llama-3_2-1b", "b95bec8ab291eeb04b5cd600dd473377f6b79026", "mit", False, [], "99c652b5bc9438a4f3fe44d358183ec8fbf4b24640c501c301f455afa495db1d", 6379, 14, "945c52e7f1a97dbaefd8257bca101e6e09ace88a732c63fdbdf8dafd78fe7b5f"),
        "llama-companion": ("meta-llama/Llama-3.2-1B-Instruct", "9213176726f574b556790deb65791e0c5aa438b6", "llama3.2", "manual", ["LICENSE.txt"], "26833339b5003d861d40425a82ed906b3fcdcd5d3eef6d33a9dea9499eb29a57", 23492, 13, "ff5697784f0a22f407a0add7e46f4cdd0b0912c9651f91fbd1a9cd35da300877"),
    }
    if not isinstance(metadata, list) or [row.get("id") for row in metadata] != sorted(expected_metadata):
        fail("compact HF metadata order/count drifted")
    for row in metadata:
        if set(row) - {"id", "repo", "revision", "requested_revision", "returned_repo", "returned_sha", "license", "gated", "private", "disabled", "license_files", "license_source", "payload_sha256", "payload_size", "tree_file_count", "tree_files_sha256", "requested_url", "final_url", "redirect_trace", "resolved_host", "resolved_path", "schema"}:
            fail("compact HF metadata contains an unsafe/unreviewed field")
        expected = expected_metadata.get(row.get("id"))
        if expected is None or (row["repo"], row["revision"], row["license"], row["gated"], row["license_files"], row["payload_sha256"], row["payload_size"], row["tree_file_count"], row["tree_files_sha256"]) != expected or row["private"] is not False or row["disabled"] is not False or row["returned_repo"] != row["repo"] or row["returned_sha"] != row["revision"] or row["requested_revision"] != row["revision"] or row["final_url"] != row["requested_url"] or row["requested_url"] != f"https://huggingface.co/api/models/{row['repo']}/revision/{row['revision']}" or row["resolved_host"] != "huggingface.co" or row["resolved_path"] != urlparse(row["requested_url"]).path or row["license_source"] != "HF_API_CARD_DATA_LICENSE" or row["schema"] != "vokra-ultravox-hf-model-metadata-v1" or not HEX64.fullmatch(row["payload_sha256"]) or not HEX64.fullmatch(row["tree_files_sha256"]):
            fail("compact exact-revision HF metadata drifted")
    license_records = model["license_file_records"]
    if not isinstance(license_records, list) or [row.get("id") for row in license_records] != ["llama-companion", "ultravox-public", "ultravox-upstream"]:
        fail("compact model LICENSE records are incomplete or unsorted")
    for row in license_records:
        if "content_base64" in row or row.get("requested_url", "").startswith("http://") or row.get("final_url") not in (None, row.get("requested_url")):
            fail("compact model LICENSE record contains unsafe payload/URL data")
    expected_license = {
        "ultravox-public": ("vokra/ultravox-v0-5-llama-3-2-1b", "ddbbeec5bfcb09c71a1f88971b794e3e5da811f9", True, None, 1078, "b05785f9f18e6716bab63424b11454513b9943a222595b70411009202fc592b5"),
        "ultravox-upstream": ("fixie-ai/ultravox-v0_5-llama-3_2-1b", "b95bec8ab291eeb04b5cd600dd473377f6b79026", False, 404, None, None),
        "llama-companion": ("meta-llama/Llama-3.2-1B-Instruct", "9213176726f574b556790deb65791e0c5aa438b6", False, 401, None, None),
    }
    for row in license_records:
        expected = expected_license[row["id"]]
        if (row["repo"], row["revision"], row["acquired_bytes"], row["error_status"], row["size"], row["sha256"]) != expected or row["requested_url"] != f"https://huggingface.co/{row['repo']}/raw/{row['revision']}/LICENSE" and row["id"] != "ultravox-public":
            fail("compact model LICENSE status/hash/URL drifted")
    components = compact["component_facts"]
    if not isinstance(components, list) or [row.get("component") for row in components] != ["llama-companion", "ultravox-public", "ultravox-upstream"]:
        fail("compact component facts are incomplete or unsorted")
    for row in components:
        if set(row) != {"component", "identity", "review", "license_file", "metadata", "owner_review", "fact_sha256"} or row["owner_review"] != "PENDING_OWNER_APPROVAL" or canonical_digest({key: value for key, value in row.items() if key != "fact_sha256"}) != row["fact_sha256"]:
            fail("compact component fact schema/hash drifted")
    acquisition = compact["acquisition"]
    if not isinstance(acquisition, dict) or acquisition.get("dependency", {}).get("model_files") != [] or acquisition.get("dependency", {}).get("out_of_scope_requests") != [] or acquisition.get("model", {}).get("non_license_requests") != []:
        fail("compact acquisition scope contains model/non-license requests")
    if compact["approval"] != {"status": "PENDING_OWNER_APPROVAL", "signer": None, "digest": None} or compact["publication"] != "NO_UPLOAD":
        fail("compact evidence contains an approval or upload decision")


def approval_scope(manifest: dict[str, Any]) -> dict[str, Any]:
    reference = dict(manifest["dependency_audit_evidence"])
    reference.pop("approval_scope_sha256", None)
    return {
        "lock_sha256": manifest["lock_sha256"],
        "project_sha256": manifest["project_sha256"],
        "package_rows_sha256": manifest["package_rows_sha256"],
        "package_review_rows": manifest["package_review_rows"],
        "package_review_rows_sha256": manifest["package_review_rows_sha256"],
        "license_rows": manifest["license_rows"],
        "license_rows_sha256": manifest["license_rows_sha256"],
        "identities": manifest["identities"],
        "dependency_audit_evidence": reference,
        "publication": manifest["publication"],
    }


def run(
    lock_path: Path,
    project_path: Path,
    manifest_path: Path,
    approval_path: Path | None,
    expected: dict[str, str],
) -> None:
    if any(path.is_symlink() or not path.is_file() for path in (lock_path, project_path, manifest_path)):
        fail("uv.lock, pyproject.toml, or tracked gate manifest is missing")
    try:
        manifest = strict_json_loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        fail(f"gate manifest is unreadable: {exc}")
    if not isinstance(manifest, dict) or set(manifest) != {"gate_version", "lock_sha256", "project_sha256", "package_rows_sha256", "required_package_rows", "package_review_rows", "package_review_rows_sha256", "forbidden_dependencies", "identities", "license_rows", "license_rows_sha256", "dependency_audit_evidence", "approval_scope_sha256", "approval", "publication"}:
        fail("gate manifest schema drifted")
    if manifest.get("gate_version") != GATE_VERSION:
        fail("unsupported gate manifest version")

    lock_bytes = lock_path.read_bytes()
    project_bytes = project_path.read_bytes()
    lock_sha = sha256_bytes(lock_bytes)
    project_sha = sha256_bytes(project_bytes)
    if lock_sha != manifest.get("lock_sha256"):
        fail("uv.lock SHA-256 does not match the reviewed lock digest")
    if project_sha != manifest.get("project_sha256"):
        fail("pyproject.toml SHA-256 does not match the reviewed project digest")
    try:
        lock = tomllib.loads(lock_bytes.decode("utf-8"))
        project = tomllib.loads(project_bytes.decode("utf-8"))
        # The self-test's tiny synthetic fixture omits resolver metadata;
        # every production uv.lock contains requires-python and is strict.
        if "requires-python" in lock:
            _validate_lock_shape(lock, project)
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError) as exc:
        fail(f"uv.lock is not valid TOML: {exc}")
    rows = package_rows(lock)
    if canonical_digest(rows) != manifest.get("package_rows_sha256"):
        fail("locked package version/source/marker/dependency rows drifted")
    validate_dependency_audit_evidence(
        manifest_path.with_name("dependency_audit_evidence.json"),
        manifest.get("dependency_audit_evidence"),
        manifest,
    )
    if not isinstance(manifest.get("approval_scope_sha256"), str) or not HEX64.fullmatch(manifest["approval_scope_sha256"]) or canonical_digest(approval_scope(manifest)) != manifest["approval_scope_sha256"]:
        fail("approval scope is not bound to the closure, identities, and compact evidence")
    required_rows = manifest.get("required_package_rows")
    if not isinstance(required_rows, list):
        fail("required package review rows are missing")
    actual_by_key = {(row["name"], row["version"]): row for row in rows}
    for required in required_rows:
        if not isinstance(required, dict):
            fail("required package review row is malformed")
        key = (required.get("name"), required.get("version"))
        actual = actual_by_key.get(key)
        if actual != required:
            fail(f"required package row drifted: {key!r}")
    package_reviews = manifest.get("package_review_rows")
    if not isinstance(package_reviews, list) or len(package_reviews) != len(rows):
        fail("every locked package needs a version-keyed native/bundled review row")
    review_by_key = {}
    for review in package_reviews:
        if not isinstance(review, dict):
            fail("package review row is malformed")
        key = (review.get("name"), review.get("version"))
        if key in review_by_key:
            fail(f"duplicate package review row: {key!r}")
        review_by_key[key] = review
        actual = actual_by_key.get(key)
        if actual is None or review.get("source") != actual.get("source"):
            fail(f"package review identity drifted: {key!r}")
        if review.get("status") != "REVIEWED":
            fail(f"package dependency/native review is not REVIEWED: {key!r}")
        if not isinstance(review.get("license"), str) or review["license"] in {"", "UNRESOLVED"}:
            fail(f"package license conclusion is unresolved: {key!r}")
        if not resolved_review(review.get("native_bundled_review")):
            fail(f"package review lacks native/bundled conclusion: {key!r}")
    if set(review_by_key) != set(actual_by_key):
        fail("package review rows do not cover the exact lock closure")
    if canonical_digest(package_reviews) != manifest.get("package_review_rows_sha256"):
        fail("version-keyed package native/bundled review rows drifted")

    forbidden = manifest.get("forbidden_dependencies", [])
    present = sorted(set(forbidden) & {row["name"] for row in rows})
    if present:
        fail(f"forbidden native/audio/CUDA packages are locked: {present}")

    identities = manifest.get("identities")
    if not isinstance(identities, dict):
        fail("fixed public/upstream/companion identities are missing")
    for field, value in expected.items():
        if identities.get(field) != value:
            fail(f"requested {field} does not match the tracked fixed identity")

    license_rows = manifest.get("license_rows")
    if not isinstance(license_rows, list) or not license_rows:
        fail("separate weight and dependency license review rows are missing")
    if canonical_digest(license_rows) != manifest.get("license_rows_sha256"):
        fail("license/native/bundled review rows drifted")
    if any(row.get("status") != "REVIEWED" for row in license_rows):
        fail("all model and dependency license rows require owner REVIEWED status")
    if any(not isinstance(row.get("license"), str) or row["license"] in {"", "UNRESOLVED"} for row in license_rows):
        fail("all model and dependency license rows require a resolved license conclusion")
    closure = next((row for row in license_rows if row.get("id") == "python-closure"), None)
    if (
        not isinstance(closure, dict)
        or not resolved_review(closure.get("native_bundled_review"))
    ):
        fail("Python closure needs a nonempty native/bundled review bound to locked rows")
    audio = next((row for row in license_rows if row.get("id") == "ultravox-audio-weight"), None)
    companion = next((row for row in license_rows if row.get("id") == "llama-companion-meta-conditional"), None)
    if not isinstance(audio, dict) or audio.get("required_identity") != "upstream_repo/upstream_revision/upstream_model_sha256":
        fail("audio-weight review must bind the fixed upstream revision and payload")
    if audio.get("payload_sha256") != identities.get("upstream_model_sha256"):
        fail("audio-weight review payload is not the fixed upstream model payload")
    if not isinstance(companion, dict) or companion.get("required_identity") != "companion_repo/companion_revision/companion_model_sha256":
        fail("Meta companion review must bind its fixed revision and payload")
    companion_payload = identities.get("companion_model_sha256")
    if not isinstance(companion_payload, str) or not HEX64.fullmatch(companion_payload) or companion.get("payload_sha256") != companion_payload:
        fail("Meta companion payload identity is not authenticated; capture the gated LFS digest before approval")
    row_ids = {row.get("id") for row in license_rows if isinstance(row, dict)}
    required_ids = {"ultravox-audio-weight", "llama-companion-meta-conditional", "python-closure"}
    if not required_ids.issubset(row_ids):
        fail("MIT audio-weight, Meta conditional companion, and Python closure rows are required")

    for field in (
        "public_revision",
        "upstream_revision",
        "companion_revision",
    ):
        require_hex(identities.get(field), HEX40, field)
    for field in (
        "public_sha256",
        "upstream_model_sha256",
    ):
        require_hex(identities.get(field), HEX64, field)
    source_files = identities.get("source_files")
    if not isinstance(source_files, dict) or set(source_files) != {
        "ultravox_model.py",
        "ultravox_processing.py",
        "ultravox_config.py",
    }:
        fail("fixed official source-file identities are incomplete")
    for name, identity in source_files.items():
        if not isinstance(identity, dict):
            fail(f"source identity is malformed: {name}")
        require_hex(identity.get("sha256"), HEX64, f"{name} SHA-256")
        if not isinstance(identity.get("bytes"), int) or identity["bytes"] <= 0:
            fail(f"source identity has no positive byte count: {name}")

    approval = manifest.get("approval")
    if not isinstance(approval, dict):
        fail("tracked owner approval schema is missing")
    signer = approval.get("signer")
    digest = approval.get("digest")
    if not isinstance(signer, str) or not signer or not HEX64.fullmatch(str(digest or "")):
        fail(
            "owner sign-off is required: a human must update the tracked signer and "
            "canonical approval digest after reviewing the complete dependency/license closure"
        )
    if approval.get("status") != "OWNER_SIGNOFF_APPROVED":
        fail("approval status is not OWNER_SIGNOFF_APPROVED; owner sign-off remains required")
    if approval_path is None or approval_path.is_symlink() or not approval_path.is_file():
        fail("authenticated owner approval evidence is missing")
    try:
        evidence = strict_json_loads(approval_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        fail(f"owner approval evidence is unreadable: {exc}")
    if not isinstance(evidence, dict):
        fail("owner approval evidence must be a JSON object")
    if evidence.get("signer") != signer or evidence.get("decision") != "APPROVED":
        fail("owner approval evidence is not the tracked sign-off")
    if evidence.get("approval_digest") != digest:
        fail("owner approval evidence digest does not match the tracked sign-off")
    if evidence.get("manifest_sha256") != sha256_bytes(manifest_path.read_bytes()):
        fail("owner approval evidence is not bound to this gate manifest")
    if evidence.get("lock_sha256") != lock_sha or evidence.get("project_sha256") != project_sha:
        fail("owner approval evidence is not bound to the current Python closure")
    print("ultravox license gate: PASS")


def self_test() -> None:
    """Exercise revision/hash/license tamper cases without project imports/network."""
    production_root = Path(__file__).resolve().parent
    production_manifest = strict_json_loads((production_root / "license_gate_manifest.json").read_text(encoding="utf-8"))
    production_reference = production_manifest["dependency_audit_evidence"]
    production_compact = strict_json_loads((production_root / "dependency_audit_evidence.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="ultravox-evidence-self-test-") as evidence_directory:
        evidence_path = Path(evidence_directory) / "dependency_audit_evidence.json"

        def expect_production_evidence_blocked(label: str, mutate: Any = None, reference_mutate: Any = None, raw: bytes | None = None) -> None:
            candidate = json.loads(json.dumps(production_compact))
            if mutate is not None:
                mutate(candidate)
            candidate_bytes = raw if raw is not None else (json.dumps(candidate, sort_keys=True, separators=(",", ":")) + "\n").encode()
            reference = json.loads(json.dumps(production_reference))
            reference["sha256"] = sha256_bytes(candidate_bytes)
            if reference_mutate is not None:
                reference_mutate(reference)
            evidence_path.write_bytes(candidate_bytes)
            try:
                validate_dependency_audit_evidence(evidence_path, reference, production_manifest)
            except SystemExit as exc:
                if exc.code != 2:
                    raise SystemExit(f"ultravox license gate self-test: {label} returned {exc.code}") from exc
            else:
                raise SystemExit(f"ultravox license gate self-test: {label} was accepted")

        expect_production_evidence_blocked("license tamper", lambda value: value["license_facts"]["packages"][0].update(declared_license="GPL"))
        expect_production_evidence_blocked("classifier tamper", lambda value: value["license_facts"]["packages"][0]["license_classifiers"].append("unsafe"))
        expect_production_evidence_blocked("publisher hash tamper", lambda value: value["license_facts"]["packages"][0].update(publisher_files_sha256="0" * 64))
        expect_production_evidence_blocked("native hash tamper", lambda value: value["license_facts"]["packages"][0].update(native_files_sha256="0" * 64))
        expect_production_evidence_blocked("aggregate tamper", lambda value: value["license_facts"].update(publisher_file_count=0))
        expect_production_evidence_blocked("inactive reason tamper", lambda value: value["inactive_facts"][0].update(reason="wrong"))
        expect_production_evidence_blocked("HF revision tamper", lambda value: value["model_facts"]["metadata_records"][0].update(revision="0" * 40, requested_revision="0" * 40))
        expect_production_evidence_blocked("HF license tamper", lambda value: value["model_facts"]["metadata_records"][0].update(license="MIT"))
        expect_production_evidence_blocked("HF gated tamper", lambda value: value["model_facts"]["metadata_records"][0].update(gated=False))
        expect_production_evidence_blocked("HF tree hash tamper", lambda value: value["model_facts"]["metadata_records"][0].update(tree_files_sha256="0" * 64))
        expect_production_evidence_blocked("source LICENSE tamper", lambda value: value["model_facts"]["license_file_records"][1].update(sha256="0" * 64))
        expect_production_evidence_blocked("full audit hash tamper", reference_mutate=lambda value: value.update(full_audit_sha256="0" * 64))
        expect_production_evidence_blocked("nested approval scope tamper", reference_mutate=lambda value: value.update(approval_scope_sha256="0" * 64))
        expect_production_evidence_blocked("approval tamper", lambda value: value["approval"].update(status="OWNER_SIGNOFF_APPROVED"))
        expect_production_evidence_blocked("duplicate evidence key", raw=b'{"schema":"x","schema":"x"}')
    with tempfile.TemporaryDirectory(prefix="ultravox-license-gate-") as directory:
        root = Path(directory)
        lock = root / "uv.lock"
        project = root / "pyproject.toml"
        manifest_path = root / "manifest.json"
        evidence_path = root / "approval.json"
        lock.write_text('version = 1\n\n[[package]]\nname = "demo"\nversion = "1.0"\n', encoding="utf-8")
        project.write_text("[project]\nname = 'demo'\n", encoding="utf-8")
        rows = package_rows(tomllib.loads(lock.read_text(encoding="utf-8")))
        license_rows = [
            {"id": "ultravox-audio-weight", "license": "MIT", "status": "REVIEWED", "payload_sha256": "e" * 64, "required_identity": "upstream_repo/upstream_revision/upstream_model_sha256"},
            {"id": "llama-companion-meta-conditional", "license": "Meta-ConditionalCommercial", "status": "REVIEWED", "payload_sha256": "1" * 64, "required_identity": "companion_repo/companion_revision/companion_model_sha256"},
            {"id": "python-closure", "license": "MIT", "status": "REVIEWED", "native_bundled_review": "self-test closure bound to demo 1.0"},
        ]
        expected = {
            "public_repo": "vokra/test",
            "public_revision": "a" * 40,
            "public_file": "test.gguf",
            "public_sha256": "b" * 64,
            "upstream_repo": "fixie/test",
            "upstream_revision": "c" * 40,
            "companion_repo": "meta-llama/test",
            "companion_revision": "d" * 40,
            "upstream_model_sha256": "e" * 64,
            "companion_model_sha256": "1" * 64,
        }
        manifest = {
            "gate_version": GATE_VERSION,
            "lock_sha256": sha256_bytes(lock.read_bytes()),
            "project_sha256": sha256_bytes(project.read_bytes()),
            "package_rows_sha256": canonical_digest(rows),
            "required_package_rows": rows,
            "package_review_rows": [{"name": "demo", "version": "1.0", "source": None, "license": "MIT", "status": "REVIEWED", "native_bundled_review": "self-test closure"}],
            "package_review_rows_sha256": canonical_digest([{"name": "demo", "version": "1.0", "source": None, "license": "MIT", "status": "REVIEWED", "native_bundled_review": "self-test closure"}]),
            "forbidden_dependencies": [],
            "license_rows": license_rows,
            "license_rows_sha256": canonical_digest(license_rows),
            "identities": {
                **expected,
                "source_files": {
                    "ultravox_model.py": {"bytes": 1, "sha256": "f" * 64},
                    "ultravox_processing.py": {"bytes": 1, "sha256": "0" * 64},
                    "ultravox_config.py": {"bytes": 1, "sha256": "1" * 64},
                },
            },
            "approval": {"signer": "self-test", "digest": "2" * 64, "status": "OWNER_SIGNOFF_APPROVED"},
            "publication": "NO_UPLOAD",
        }
        compact_path = root / "dependency_audit_evidence.json"
        compact = {
            "schema": COMPACT_SCHEMA,
            "status": "PENDING_OWNER_APPROVAL",
            "full_audit_status": "BLOCKED",
            "full_audit_sha256": "e" * 64,
            "inputs": {
                "pyproject_sha256": manifest["project_sha256"],
                "uv_lock_sha256": manifest["lock_sha256"],
                "package_review_rows_sha256": manifest["package_review_rows_sha256"],
                "license_rows_sha256": manifest["license_rows_sha256"],
            },
            "repository": {}, "environment": {}, "closure": {}, "license_facts": {},
            "native_facts": {}, "model_facts": {}, "inactive_facts": {}, "component_facts": {},
            "manifest_reviews": {}, "acquisition": {},
            "approval": {"status": "PENDING_OWNER_APPROVAL", "signer": None, "digest": None},
            "publication": "NO_UPLOAD",
        }
        compact_path.write_text(json.dumps(compact, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        manifest["dependency_audit_evidence"] = {
            "schema": COMPACT_SCHEMA,
            "path": "dependency_audit_evidence.json",
            "sha256": sha256_bytes(compact_path.read_bytes()),
            "full_audit_sha256": "e" * 64,
            "status": "PENDING_OWNER_APPROVAL",
            "approval_scope_sha256": "0" * 64,
        }
        manifest["approval_scope_sha256"] = canonical_digest(approval_scope(manifest))
        manifest["dependency_audit_evidence"]["approval_scope_sha256"] = manifest["approval_scope_sha256"]
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        evidence = {
            "signer": "self-test",
            "decision": "APPROVED",
            "approval_digest": "2" * 64,
            "manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
            "lock_sha256": manifest["lock_sha256"],
            "project_sha256": manifest["project_sha256"],
        }
        evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

        def expect_blocked(label: str, mutate: Any) -> None:
            candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
            mutate(candidate)
            manifest_path.write_text(json.dumps(candidate, sort_keys=True), encoding="utf-8")
            try:
                run(lock, project, manifest_path, evidence_path, expected)
            except SystemExit as exc:
                if exc.code != 2:
                    raise SystemExit(f"ultravox license gate self-test: {label} returned {exc.code}") from exc
            else:
                raise SystemExit(f"ultravox license gate self-test: {label} was accepted")
            manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

        run(lock, project, manifest_path, evidence_path, expected)
        compact_bytes = compact_path.read_bytes()
        compact_reference = json.loads(json.dumps(manifest["dependency_audit_evidence"]))

        def expect_evidence_blocked(label: str, candidate: bytes | None) -> None:
            if candidate is None:
                compact_path.unlink()
            else:
                compact_path.write_bytes(candidate)
            try:
                run(lock, project, manifest_path, evidence_path, expected)
            except SystemExit as exc:
                if exc.code != 2:
                    raise SystemExit(f"ultravox license gate self-test: {label} returned {exc.code}") from exc
            else:
                raise SystemExit(f"ultravox license gate self-test: {label} was accepted")
            compact_path.write_bytes(compact_bytes)

        expect_evidence_blocked("missing compact evidence", None)
        expect_evidence_blocked("compact evidence tamper", compact_bytes + b" ")
        expect_evidence_blocked("duplicate compact evidence key", b'{"schema":"x","schema":"x"}')
        duplicate_manifest = root / "duplicate-manifest.json"
        duplicate_manifest.write_text('{"gate_version":1,"gate_version":1}', encoding="utf-8")
        try:
            run(lock, project, duplicate_manifest, evidence_path, expected)
        except SystemExit as error:
            if error.code != 2:
                raise SystemExit(f"ultravox license gate self-test duplicate manifest returned {error.code}") from error
        else:
            raise SystemExit("ultravox license gate self-test duplicate manifest was accepted")
        duplicate_evidence = root / "duplicate-evidence.json"
        duplicate_evidence.write_text('{"signer":"self-test","signer":"self-test"}', encoding="utf-8")
        try:
            run(lock, project, manifest_path, duplicate_evidence, expected)
        except SystemExit as error:
            if error.code != 2:
                raise SystemExit(f"ultravox license gate self-test duplicate evidence returned {error.code}") from error
        else:
            raise SystemExit("ultravox license gate self-test duplicate evidence was accepted")
        expect_blocked("revision tamper", lambda value: value["identities"].update(public_revision="f" * 40))
        expect_blocked("hash tamper", lambda value: value["identities"].update(public_sha256="f" * 64))
        expect_blocked("license tamper", lambda value: value["license_rows"].__getitem__(0).update(license="Apache-2.0"))
        expect_blocked("unresolved status", lambda value: value["license_rows"].__getitem__(0).update(status="APPROVAL_PENDING"))
        expect_blocked("native closure tamper", lambda value: value["license_rows"].__getitem__(2).update(native_bundled_review=""))
        expect_blocked("native review placeholder", lambda value: value["license_rows"].__getitem__(2).update(native_bundled_review="OWNER_REVIEW_REQUIRED"))
        expect_blocked("package review tamper", lambda value: value["package_review_rows"].__getitem__(0).update(version="2.0"))
        expect_blocked("package license tamper", lambda value: value["package_review_rows"].__getitem__(0).update(license="Apache-2.0"))
        expect_blocked("approval tamper", lambda value: value["approval"].update(digest="3" * 64))
        expect_blocked("approval status tamper", lambda value: value["approval"].update(status="OWNER_SIGNOFF_REQUIRED"))
        expect_blocked("lock approval binding", lambda value: value.update(lock_sha256="f" * 64))
        expect_blocked("project approval binding", lambda value: value.update(project_sha256="f" * 64))
    print("license_gate.py self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--approval", type=Path)
    for field in (
        "public-repo",
        "public-revision",
        "public-file",
        "public-sha256",
        "upstream-repo",
        "upstream-revision",
        "companion-repo",
        "companion-revision",
        "upstream-model-sha256",
    ):
        parser.add_argument(f"--{field}", required=False)
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.lock, args.project, args.manifest, args.approval)):
            parser.error("--self-test accepts no path options")
        self_test()
        return 0
    fields = (
        "lock",
        "project",
        "manifest",
        "public_repo",
        "public_revision",
        "public_file",
        "public_sha256",
        "upstream_repo",
        "upstream_revision",
        "companion_repo",
        "companion_revision",
        "upstream_model_sha256",
    )
    if any(getattr(args, field) is None for field in fields):
        parser.error("all closure paths and fixed model/source identity options are required")
    expected = {
        "public_repo": args.public_repo,
        "public_revision": args.public_revision,
        "public_file": args.public_file,
        "public_sha256": args.public_sha256,
        "upstream_repo": args.upstream_repo,
        "upstream_revision": args.upstream_revision,
        "companion_repo": args.companion_repo,
        "companion_revision": args.companion_revision,
        "upstream_model_sha256": args.upstream_model_sha256,
    }
    run(args.lock, args.project, args.manifest, args.approval, expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
