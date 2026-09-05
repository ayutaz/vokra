#!/usr/bin/env python3
"""Fail-closed, dependency-free BigVGAN lock/license gate.

This is deliberately run with ``uv run --no-project`` before the dedicated
reference environment is synchronized. It only reads the lock, gate manifest,
and optional operator evidence. Missing approval/evidence exits 2; no package
metadata is fetched and no model/source acquisition is attempted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from urllib.parse import urlsplit
from pathlib import Path
from typing import Any

import tomllib


GATE_VERSION = 1
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
MANIFEST_KEYS = {
    "gate_version", "lock_sha256", "project_sha256", "package_rows_sha256", "package_review_rows",
    "package_review_rows_sha256", "identities", "required_package_rows", "forbidden_dependencies",
    "license_rows", "license_rows_sha256", "publication", "approval",
}
LOCK_KEYS = {"version", "revision", "requires-python", "resolution-markers", "supported-markers", "package"}
PACKAGE_KEYS = {"name", "version", "source", "resolution-markers", "dependencies", "sdist", "wheels", "metadata"}
ARTIFACT_KEYS = {"url", "hash", "size", "upload-time"}
REGISTRY_PACKAGE_SCHEMAS = {
    frozenset({"name", "version", "source", "sdist", "wheels"}),
    frozenset({"name", "version", "source", "dependencies", "sdist", "wheels"}),
    frozenset({"name", "version", "source", "resolution-markers", "dependencies", "wheels"}),
}
DEPENDENCY_SCHEMAS = {
    frozenset({"name", "marker"}), frozenset({"name", "marker", "source", "version"}),
}
METADATA_REQUIREMENT_SCHEMAS = {frozenset({"name", "specifier", "index"})}
LICENSE_ROW_KEYS = {"id", "status", "component", "source", "license", "payload_sha256", "required_evidence_fields", "approval_schema", "approval_signer", "approval_digest", "review"}
REVIEW_PLACEHOLDERS = {"", "unresolved", "pending", "pending_review", "owner_review_required", "review_required", "todo", "null", "none"}


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_digest(value: Any) -> str:
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def reviewed(value: Any) -> bool:
    normalized = re.sub(r"\s+", "_", value.strip()).casefold() if isinstance(value, str) else ""
    return normalized not in REVIEW_PLACEHOLDERS


def load_json(path: Path) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)


def validate_artifact(value: Any, label: str, registry: str) -> None:
    if not isinstance(value, dict) or set(value) != ARTIFACT_KEYS:
        fail(f"{label} artifact schema is malformed")
    expected_host = "download-r2.pytorch.org" if registry == "https://download.pytorch.org/whl/cpu" else "files.pythonhosted.org"
    parsed = urlsplit(value["url"]) if isinstance(value["url"], str) else None
    if parsed is None or parsed.scheme != "https" or parsed.netloc != expected_host or not parsed.path:
        fail(f"{label} artifact URL is not the authenticated {expected_host} host")
    if not isinstance(value["hash"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value["hash"]):
        fail(f"{label} artifact hash is not a SHA-256")
    if isinstance(value["size"], bool) or not isinstance(value["size"], int) or value["size"] <= 0:
        fail(f"{label} artifact size is not positive")
    if not isinstance(value["upload-time"], str) or not value["upload-time"].strip():
        fail(f"{label} artifact upload-time is missing")


def validate_metadata(value: Any, label: str) -> None:
    if not isinstance(value, dict) or set(value) != {"requires-dist"} or not isinstance(value["requires-dist"], list):
        fail(f"{label} metadata schema is malformed")
    for requirement in value["requires-dist"]:
        if not isinstance(requirement, dict) or frozenset(requirement) not in METADATA_REQUIREMENT_SCHEMAS or not isinstance(requirement.get("name"), str) or not requirement["name"].strip() or not isinstance(requirement.get("specifier"), str) or not requirement["specifier"].strip():
            fail(f"{label} metadata requirement is malformed")
        if "index" in requirement and requirement["index"] not in ("https://pypi.org/simple", "https://download.pytorch.org/whl/cpu"):
            fail(f"{label} metadata index is not approved")


def package_rows(lock: dict[str, Any]) -> list[dict[str, Any]]:
    packages = lock.get("package")
    if not isinstance(packages, list) or not packages or set(lock) != LOCK_KEYS or lock.get("version") != 1 or lock.get("revision") != 3 or lock.get("requires-python") != "==3.12.*" or not isinstance(lock.get("resolution-markers"), list) or any(not isinstance(item, str) or not item.strip() for item in lock["resolution-markers"]) or not isinstance(lock.get("supported-markers"), list) or any(not isinstance(item, str) or not item.strip() for item in lock["supported-markers"]):
        fail("uv.lock package table/top-level schema is missing or malformed")
    rows = []
    identities: set[tuple[str, str]] = set()
    virtual = []
    for package in packages:
        if not isinstance(package, dict) or set(package) - PACKAGE_KEYS:
            fail("uv.lock package row is malformed")
        if not isinstance(package.get("name"), str) or not package["name"].strip() or not isinstance(package.get("version"), str) or not package["version"].strip():
            fail("uv.lock package name/version is not an exact nonempty string")
        identity = (package["name"], package["version"])
        if identity in identities:
            fail(f"uv.lock duplicate package identity: {identity!r}")
        identities.add(identity)
        source = package.get("source")
        if not isinstance(source, dict) or len(source) != 1 or set(source) not in ({"registry"}, {"virtual"}):
            fail(f"uv.lock source schema is malformed: {identity!r}")
        if ("virtual" in source and frozenset(package) != frozenset({"name", "version", "source", "dependencies", "metadata"})) or ("registry" in source and frozenset(package) not in REGISTRY_PACKAGE_SCHEMAS):
            fail(f"uv.lock package row schema drifted: {identity!r}")
        registry = source.get("registry")
        if "registry" in source and registry not in ("https://pypi.org/simple", "https://download.pytorch.org/whl/cpu"):
            fail(f"uv.lock registry source is not an approved index: {identity!r}")
        if "virtual" in source:
            virtual.append(package)
            if source["virtual"] != ".":
                fail("uv.lock virtual source is not '.'")
        markers = package.get("resolution-markers", [])
        if not isinstance(markers, list) or any(not isinstance(item, str) or not item.strip() for item in markers):
            fail(f"uv.lock package markers are malformed: {identity!r}")
        dependencies = package.get("dependencies", [])
        if not isinstance(dependencies, list):
            fail(f"uv.lock dependencies are malformed: {identity!r}")
        for dependency in dependencies:
            if not isinstance(dependency, dict) or frozenset(dependency) not in DEPENDENCY_SCHEMAS or not isinstance(dependency.get("name"), str):
                fail(f"uv.lock dependency row is malformed: {identity!r}")
            if not isinstance(dependency.get("marker"), str) or not dependency["marker"].strip():
                    fail(f"uv.lock dependency field is malformed: {identity!r}")
            if "source" in dependency:
                if not isinstance(dependency.get("version"), str) or not dependency["version"].strip():
                    fail(f"uv.lock dependency version is malformed: {identity!r}")
                dependency_source = dependency["source"]
                if not isinstance(dependency_source, dict) or len(dependency_source) != 1 or set(dependency_source) not in ({"registry"}, {"virtual"}):
                    fail(f"uv.lock dependency source is malformed: {identity!r}")
                if "registry" in dependency_source and dependency_source["registry"] not in ("https://pypi.org/simple", "https://download.pytorch.org/whl/cpu"):
                    fail(f"uv.lock dependency registry source is not an approved index: {identity!r}")
                if "virtual" in dependency_source and dependency_source["virtual"] != ".":
                    fail(f"uv.lock dependency virtual source is not '.': {identity!r}")
        if "virtual" in source:
            if set(package) != {"name", "version", "source", "dependencies", "metadata"}:
                fail(f"{identity!r} virtual package schema is malformed")
            validate_metadata(package["metadata"], f"{identity!r} virtual")
            if "sdist" in package or "wheels" in package:
                fail(f"{identity!r} virtual package must not contain artifacts")
        else:
            if "metadata" in package:
                validate_metadata(package["metadata"], f"{identity!r}")
            if "sdist" in package:
                validate_artifact(package["sdist"], f"{identity!r} sdist", registry)
            if "wheels" in package:
                if not isinstance(package["wheels"], list) or not package["wheels"]:
                    fail(f"{identity!r} wheels table is malformed")
                for artifact in package["wheels"]:
                    validate_artifact(artifact, f"{identity!r} wheel", registry)
            if "sdist" not in package and not package.get("wheels"):
                fail(f"{identity!r} registry package has no authenticated artifacts")
        rows.append(package)
    if len(virtual) != 1:
        fail("uv.lock must contain exactly one virtual project package")
    return sorted(rows, key=lambda row: (row["name"], row["version"]))


def expected_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": row["name"],
        "version": row["version"],
        "source": row["source"],
        "resolution-markers": row["resolution-markers"],
        "dependencies": row["dependencies"],
    }


def fail(message: str) -> None:
    print(f"bigvgan license gate: BLOCKED: {message}", file=sys.stderr)
    raise SystemExit(2)


def validate_project_schema(project: dict[str, Any]) -> None:
    if set(project) != {"project", "tool"} or not isinstance(project.get("project"), dict) or set(project["project"]) != {"name", "version", "description", "requires-python", "dependencies"}:
        fail("pyproject.toml structural schema drifted")
    fields = project["project"]
    if fields["requires-python"] != "==3.12.*" or not isinstance(fields["dependencies"], list) or any(not isinstance(item, str) or not item.strip() for item in fields["dependencies"]):
        fail("pyproject.toml project contract drifted")
    tool = project["tool"]
    if set(tool) != {"uv", "vokra"} or not isinstance(tool.get("uv"), dict) or set(tool["uv"]) != {"package", "environments", "sources", "index"} or not isinstance(tool.get("vokra"), dict) or set(tool["vokra"]) != {"bigvgan_reference"}:
        fail("pyproject.toml tool schema drifted")
    uv = tool["uv"]
    if uv["package"] is not False or uv["environments"] != ["sys_platform == 'linux' and platform_machine == 'x86_64'", "sys_platform == 'darwin' and platform_machine == 'arm64'"] or uv["sources"] != {"torch": {"index": "pytorch-cpu"}} or uv["index"] != [{"name": "pytorch-cpu", "url": "https://download.pytorch.org/whl/cpu", "explicit": True}]:
        fail("pyproject.toml uv contract drifted")
    reference = tool["vokra"]["bigvgan_reference"]
    if set(reference) != {"status", "source_repository", "checkpoint_identity", "license_gate", "forbidden_dependencies", "publication"} or any(not isinstance(reference[key], str) or not reference[key].strip() for key in ("status", "source_repository", "checkpoint_identity", "license_gate", "publication")) or not isinstance(reference["forbidden_dependencies"], list) or any(not isinstance(item, str) or not item.strip() for item in reference["forbidden_dependencies"]):
        fail("pyproject.toml reference policy schema drifted")


def approval_scope(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    approval = manifest.get("approval")
    return canonical_digest({
        "schema": "bigvgan-approval-scope-v1",
        "lock_sha256": manifest.get("lock_sha256"),
        "project_sha256": manifest.get("project_sha256"),
        "package_rows": rows,
        "package_rows_sha256": manifest.get("package_rows_sha256"),
        "package_review_rows": manifest.get("package_review_rows"),
        "package_review_rows_sha256": manifest.get("package_review_rows_sha256"),
        "identities": manifest.get("identities"),
        "license_rows": manifest.get("license_rows"),
        "license_rows_sha256": manifest.get("license_rows_sha256"),
        "publication": manifest.get("publication"),
        "expected_decision": "APPROVED",
        "expected_status": "OWNER_SIGNOFF_APPROVED",
        "signer": approval.get("signer") if isinstance(approval, dict) else None,
    })


def run(
    lock_path: Path,
    project_path: Path,
    manifest_path: Path,
    evidence_path: Path | None,
    source_revision: str | None = None,
    model_revision: str | None = None,
    checkpoint_sha256: str | None = None,
    config_sha256: str | None = None,
) -> None:
    if not regular_file(lock_path) or not regular_file(project_path) or not regular_file(manifest_path):
        fail("lock, pyproject, or gate manifest is missing")
    try:
        manifest = load_json(manifest_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        fail(f"gate manifest is unreadable: {exc}")
    if not isinstance(manifest, dict) or manifest.get("gate_version") != GATE_VERSION:
        fail("unsupported gate manifest version")
    if set(manifest) != MANIFEST_KEYS:
        fail("gate manifest top-level schema drifted")
    lock_bytes = lock_path.read_bytes()
    project_bytes = project_path.read_bytes()
    if digest_bytes(lock_bytes) != manifest.get("lock_sha256"):
        fail("uv.lock SHA-256 does not match the reviewed lock digest")
    if digest_bytes(project_bytes) != manifest.get("project_sha256"):
        fail("pyproject.toml SHA-256 does not match the reviewed project digest")
    try:
        lock = tomllib.loads(lock_bytes.decode("utf-8"))
        project = tomllib.loads(project_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        fail(f"uv.lock is not valid TOML: {exc}")
    validate_project_schema(project)
    rows = package_rows(lock)
    project_identity = project.get("project")
    virtual = [row for row in rows if row.get("source") == {"virtual": "."}]
    if not isinstance(project_identity, dict) or not isinstance(project_identity.get("name"), str) or not isinstance(project_identity.get("version"), str) or len(virtual) != 1 or (virtual[0]["name"], virtual[0]["version"]) != (project_identity["name"], project_identity["version"]):
        fail("uv.lock virtual project is not bound to pyproject.toml")
    if canonical_digest(rows) != manifest.get("package_rows_sha256"):
        fail("package version/source/marker/dependency rows drifted")
    required_rows = manifest.get("required_package_rows")
    if not isinstance(required_rows, list) or len(required_rows) != len(rows):
        fail("required package rows do not cover the complete lock closure")
    expected_package_ids = [(row["name"], row["version"]) for row in rows]
    if any(not isinstance(expected, dict) or set(expected) != {"name", "version"} or not isinstance(expected.get("name"), str) or not isinstance(expected.get("version"), str) for expected in required_rows):
        fail("required package rows have a malformed schema")
    required_ids = [(expected["name"], expected["version"]) for expected in required_rows]
    if required_ids != expected_package_ids or len(set(required_ids)) != len(required_ids):
        fail("required package rows contain missing, extra, reordered, or duplicate identities")
    actual_by_key = {(row["name"], row["version"]): row for row in rows}
    for expected in required_rows:
        key = (expected.get("name"), expected.get("version"))
        actual = actual_by_key.get(key)
        if actual is None:
            fail(f"required package row drifted: {key!r}")
    if len(actual_by_key) != len(required_rows):
        fail("required package rows contain duplicate identities")

    package_reviews = manifest.get("package_review_rows")
    if not isinstance(package_reviews, list) or len(package_reviews) != len(rows) or canonical_digest(package_reviews) != manifest.get("package_review_rows_sha256"):
        fail("package review rows do not bind the complete lock closure")
    package_ids = [f"{row['name']}@{row['version']}" for row in rows]
    if [item.get("id") for item in package_reviews if isinstance(item, dict)] != package_ids:
        fail("package review row identities drifted")
    for item in package_reviews:
        if not isinstance(item, dict) or set(item) != {"id", "status", "license", "native_bundled_review"} or item["status"] != "REVIEWED" or not reviewed(item["license"]) or not reviewed(item["native_bundled_review"]):
            fail("package license/native/bundled review is unresolved")

    forbidden = manifest.get("forbidden_dependencies", [])
    names = {row["name"] for row in rows}
    present = sorted(set(forbidden) & names)
    if present:
        fail(f"forbidden native/audio/CUDA packages are locked: {present}")

    license_rows = manifest.get("license_rows", [])
    expected_license_ids = ["bigvgan-source-license", "bigvgan-model-license", "python-cpu-closure-native-bundled"]
    if not isinstance(license_rows, list) or len(license_rows) != len(expected_license_ids) or any(not isinstance(row, dict) or set(row) != LICENSE_ROW_KEYS for row in license_rows):
        fail("license rows are not the exact schema/set")
    license_ids = [row["id"] for row in license_rows]
    if license_ids != expected_license_ids or len(set(license_ids)) != len(expected_license_ids):
        fail("license rows contain missing, extra, reordered, or duplicate identities")
    if canonical_digest(license_rows) != manifest.get("license_rows_sha256"):
        fail("license/native/bundled review rows drifted")
    reviewed_identities = manifest.get("identities")
    if not isinstance(reviewed_identities, dict):
        fail("reviewed model/source identities are missing from the manifest")
    expected_identities = {
        "source_commit": source_revision,
        "hf_revision": model_revision,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "lock_sha256": manifest.get("lock_sha256"),
        "source_license_blob_sha256": reviewed_identities.get("source_license_blob_sha256"),
        "model_license_blob_sha256": reviewed_identities.get("model_license_blob_sha256"),
    }
    for field, value in expected_identities.items():
        if field in {field_name for row in license_rows for field_name in row.get("required_evidence_fields", [])}:
            if not isinstance(value, str) or not value:
                fail(f"expected {field} identity is required")
            if field in {"source_commit", "hf_revision"} and not HEX40.fullmatch(value):
                fail(f"expected {field} is not a revision")
            if field.endswith("sha256") and not HEX64.fullmatch(value):
                fail(f"expected {field} is not a SHA-256")
            reviewed_value = {
                "source_commit": reviewed_identities.get("source_revision"),
                "hf_revision": reviewed_identities.get("model_revision"),
                "checkpoint_sha256": reviewed_identities.get("checkpoint_sha256"),
                "config_sha256": reviewed_identities.get("config_sha256"),
            }.get(field)
            if reviewed_value is not None and value != reviewed_value:
                fail(f"requested {field} does not match the reviewed manifest identity")
    approval = manifest.get("approval")
    if not isinstance(approval, dict) or set(approval) != {"status", "signer", "digest"} or approval.get("status") != "OWNER_SIGNOFF_APPROVED" or not isinstance(approval.get("signer"), str) or not approval["signer"] or not HEX64.fullmatch(str(approval.get("digest", ""))):
        fail("owner approval remains pending or has an invalid schema")
    scope = approval_scope(manifest, rows)
    if approval["digest"] != scope or manifest.get("publication") != "NO_UPLOAD":
        fail("owner approval does not cover the exact closure and NO_UPLOAD decision")
    if evidence_path is None or not regular_file(evidence_path):
        fail("operator license evidence JSON is required; unresolved review rows remain")
    try:
        evidence = load_json(evidence_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        fail(f"operator license evidence is unreadable: {exc}")
    evidence_keys = {"schema", "decision", "scope_sha256", "manifest_sha256", "lock_sha256", "project_sha256", "signer", "digest", "license_rows_sha256", "package_review_rows_sha256", "rows"}
    if not isinstance(evidence, dict) or set(evidence) != evidence_keys or evidence.get("schema") != "bigvgan-approval-evidence-v1" or evidence.get("decision") != "APPROVED" or evidence.get("scope_sha256") != scope or evidence.get("digest") != scope or evidence.get("signer") != approval["signer"] or evidence.get("manifest_sha256") != digest_bytes(manifest_path.read_bytes()) or evidence.get("lock_sha256") != manifest.get("lock_sha256") or evidence.get("project_sha256") != manifest.get("project_sha256") or evidence.get("license_rows_sha256") != manifest.get("license_rows_sha256") or evidence.get("package_review_rows_sha256") != manifest.get("package_review_rows_sha256"):
        fail("operator approval evidence is not bound to the exact canonical scope")
    evidence_rows = evidence.get("rows") if isinstance(evidence, dict) else None
    if not isinstance(evidence_rows, list):
        fail("operator license evidence must contain a rows array")
    if evidence.get("license_rows_sha256") != manifest.get("license_rows_sha256"):
        fail("operator evidence is not bound to this review-row digest")
    expected_evidence_ids = expected_license_ids
    if len(evidence_rows) != len(expected_evidence_ids) or any(not isinstance(row, dict) for row in evidence_rows):
        fail("operator evidence rows are not the exact count")
    evidence_ids = [row.get("id") for row in evidence_rows]
    if evidence_ids != expected_evidence_ids or len(set(evidence_ids)) != len(expected_evidence_ids):
        fail("operator evidence rows contain missing, extra, reordered, or duplicate identities")
    by_id = {row["id"]: row for row in evidence_rows}
    for row in license_rows:
        if row["status"] != "REVIEWED" or not reviewed(row["component"]) or not reviewed(row["source"]) or not reviewed(row["license"]) or not reviewed(row["review"]):
            fail(f"license/native/bundled review is unresolved: {row.get('id')}")
        if not isinstance(row["required_evidence_fields"], list) or not row["required_evidence_fields"] or any(not isinstance(field, str) or not field for field in row["required_evidence_fields"]):
            fail(f"license evidence field schema is malformed: {row.get('id')}")
        if row["approval_schema"] != "v1" or not isinstance(row["approval_signer"], str) or not reviewed(row["approval_signer"]) or not HEX64.fullmatch(str(row["approval_digest"] or "")):
            fail(f"license row approval identity is unresolved: {row.get('id')}")
        evidence_row = by_id.get(row.get("id"))
        if not isinstance(evidence_row, dict) or evidence_row.get("status") != "REVIEWED":
            fail(f"operator evidence is not reviewed: {row.get('id')}")
        expected_evidence_keys = {"id", "status", "license", "payload_sha256", "approval", *row["required_evidence_fields"]}
        if set(evidence_row) != expected_evidence_keys:
            fail(f"operator evidence row schema drifted: {row.get('id')}")
        if not HEX64.fullmatch(str(evidence_row.get("payload_sha256", ""))):
            fail(f"operator evidence lacks payload SHA-256: {row.get('id')}")
        if row.get("payload_sha256") is not None and evidence_row.get("payload_sha256") != row.get("payload_sha256"):
            fail(f"operator evidence payload digest does not match the reviewed payload: {row.get('id')}")
        if not reviewed(evidence_row.get("license")):
            fail(f"operator evidence lacks a license class: {row.get('id')}")
        if row.get("license") not in (None, "UNRESOLVED") and evidence_row.get("license") != row.get("license"):
            fail(f"operator evidence license does not match the reviewed license: {row.get('id')}")
        for field in row.get("required_evidence_fields", []):
            value = evidence_row.get(field)
            if not reviewed(value):
                fail(f"operator evidence lacks authenticated field {field!r}: {row.get('id')}")
            if field.endswith(("_sha256", "_digest")) and not HEX64.fullmatch(value):
                fail(f"operator evidence field {field!r} is not a SHA-256: {row.get('id')}")
            if field.endswith(("_revision", "_commit")) and not HEX40.fullmatch(value):
                fail(f"operator evidence field {field!r} is not a revision: {row.get('id')}")
            expected = expected_identities.get(field)
            if expected is not None and value != expected:
                fail(f"operator evidence field {field!r} does not match the requested identity: {row.get('id')}")
        approval_schema = row.get("approval_schema")
        approval_signer = row.get("approval_signer")
        approval_digest = row.get("approval_digest")
        approval = evidence_row.get("approval")
        if (
            approval_schema != "v1"
            or not isinstance(approval_signer, str)
            or not approval_signer
            or not HEX64.fullmatch(str(approval_digest or ""))
        ):
            fail(f"review row lacks a fixed sign-off identity: {row.get('id')}")
        if not isinstance(approval, dict) or set(approval) != {"schema", "decision", "signer", "signature_sha256"} or approval.get("schema") != approval_schema:
            fail(f"operator evidence lacks the required approval schema: {row.get('id')}")
        if approval.get("decision") != "APPROVED" or approval.get("signer") != approval_signer:
            fail(f"operator evidence approval is not the fixed sign-off: {row.get('id')}")
        if approval.get("signature_sha256") != approval_digest:
            fail(f"operator evidence approval digest is not the fixed sign-off: {row.get('id')}")
        if not HEX64.fullmatch(str(approval.get("signature_sha256", ""))):
            fail(f"operator evidence approval signature is not a SHA-256: {row.get('id')}")
    print("bigvgan license gate: PASS")


def self_test() -> None:
    """Prove missing/tampered review evidence blocks without a project."""
    with tempfile.TemporaryDirectory(prefix="bigvgan-license-gate-") as directory:
        root = Path(directory)
        lock = root / "uv.lock"
        lock.write_text("version = 1\nrevision = 3\nrequires-python = '==3.12.*'\nresolution-markers = []\nsupported-markers = []\n\n[[package]]\nname = 'bigvgan-self-test'\nversion = '0.0.0'\nsource = { virtual = '.' }\ndependencies = []\nmetadata = { requires-dist = [] }\n", encoding="utf-8")
        project = root / "pyproject.toml"
        project.write_text('''[project]
name = "bigvgan-self-test"
version = "0.0.0"
description = "self-test"
requires-python = "==3.12.*"
dependencies = []

[tool.uv]
package = false
environments = ["sys_platform == 'linux' and platform_machine == 'x86_64'", "sys_platform == 'darwin' and platform_machine == 'arm64'"]

[tool.uv.sources]
torch = { index = "pytorch-cpu" }

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.vokra.bigvgan_reference]
status = "self-test"
source_repository = "self-test"
checkpoint_identity = "self-test"
license_gate = "self-test"
forbidden_dependencies = []
publication = "NO_UPLOAD"
''', encoding="utf-8")
        rows = package_rows(tomllib.loads(lock.read_text(encoding="utf-8")))
        valid_artifact = {"url": "https://files.pythonhosted.org/packages/demo.whl", "hash": "sha256:" + "0" * 64, "size": 1, "upload-time": "2024-01-01T00:00:00Z"}
        for label, mutate in {"missing-size": lambda value: value.pop("size"), "missing-upload-time": lambda value: value.pop("upload-time"), "extra-key": lambda value: value.update(extra="x"), "bool-size": lambda value: value.update(size=True), "wrong-host": lambda value: value.update(url="https://example.invalid/demo.whl")}.items():
            candidate = dict(valid_artifact)
            mutate(candidate)
            try:
                validate_artifact(candidate, f"self-test {label}", "https://pypi.org/simple")
            except SystemExit as exc:
                if exc.code != 2:
                    raise
            else:
                raise SystemExit(f"bigvgan license gate self-test accepted {label} artifact")
        malformed_lock = tomllib.loads("""version = 1
revision = 3
requires-python = '==3.12.*'
resolution-markers = []
supported-markers = []

[[package]]
name = 'bigvgan-self-test'
version = '0.0.0'
source = { virtual = '.' }
dependencies = []
metadata = { requires-dist = [] }

[[package]]
name = 'registry-demo'
version = '1.0.0'
source = { registry = 'https://pypi.org/simple' }
""")
        try:
            package_rows(malformed_lock)
        except SystemExit as exc:
            if exc.code != 2:
                raise
        else:
            raise SystemExit("bigvgan license gate self-test accepted a registry package without artifacts")
        license_rows = [
            {
                "id": "bigvgan-source-license",
                "status": "REVIEWED",
                "component": "self-test source",
                "source": "https://example.invalid/source",
                "license": "MIT",
                "payload_sha256": "0" * 64,
                "required_evidence_fields": ["source_commit", "source_license_blob_sha256"],
                "approval_schema": "v1",
                "approval_signer": "self-test-signer",
                "approval_digest": "2" * 64,
                "review": "self-test review",
            },
            {
                "id": "bigvgan-model-license",
                "status": "REVIEWED",
                "component": "self-test model",
                "source": "https://example.invalid/model",
                "license": "MIT",
                "payload_sha256": "1" * 64,
                "required_evidence_fields": ["hf_revision", "model_license_blob_sha256", "hf_card_data_object_sha256", "checkpoint_sha256", "config_sha256"],
                "approval_schema": "v1",
                "approval_signer": "self-test-signer",
                "approval_digest": "2" * 64,
                "review": "self-test review",
            },
            {
                "id": "python-cpu-closure-native-bundled",
                "status": "REVIEWED",
                "component": "self-test closure",
                "source": "https://example.invalid/closure",
                "license": "MIT",
                "payload_sha256": "3" * 64,
                "required_evidence_fields": ["lock_sha256", "native_bundled_review"],
                "approval_schema": "v1",
                "approval_signer": "self-test-signer",
                "approval_digest": "2" * 64,
                "review": "self-test review",
            }
        ]
        manifest = {
            "gate_version": GATE_VERSION,
            "lock_sha256": digest_bytes(lock.read_bytes()),
            "project_sha256": digest_bytes(project.read_bytes()),
            "package_rows_sha256": canonical_digest(rows),
            "required_package_rows": [{"name": "bigvgan-self-test", "version": "0.0.0"}],
            "package_review_rows": [{"id": "bigvgan-self-test@0.0.0", "status": "REVIEWED", "license": "MIT", "native_bundled_review": "self-test closure review"}],
            "package_review_rows_sha256": canonical_digest([{"id": "bigvgan-self-test@0.0.0", "status": "REVIEWED", "license": "MIT", "native_bundled_review": "self-test closure review"}]),
            "forbidden_dependencies": [],
            "license_rows": license_rows,
            "license_rows_sha256": canonical_digest(license_rows),
            "publication": "NO_UPLOAD",
            "approval": {"status": "OWNER_SIGNOFF_APPROVED", "signer": "self-test-signer", "digest": ""},
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        expected_source = "a" * 40
        expected_lock = manifest["lock_sha256"]
        manifest["identities"] = {
            "source_revision": expected_source,
            "model_revision": "b" * 40,
            "checkpoint_sha256": "c" * 64,
            "config_sha256": "d" * 64,
            "source_license_blob_sha256": "0" * 64,
            "model_license_blob_sha256": "1" * 64,
        }
        manifest["approval"]["digest"] = approval_scope(manifest, rows)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        expected_evidence = {
            "schema": "bigvgan-approval-evidence-v1",
            "decision": "APPROVED",
            "scope_sha256": manifest["approval"]["digest"],
            "manifest_sha256": "",
            "lock_sha256": manifest["lock_sha256"],
            "project_sha256": manifest["project_sha256"],
            "signer": "self-test-signer",
            "digest": manifest["approval"]["digest"],
            "license_rows_sha256": manifest["license_rows_sha256"],
            "package_review_rows_sha256": manifest["package_review_rows_sha256"],
            "rows": [
                {
                    "id": "bigvgan-source-license",
                    "status": "REVIEWED",
                    "license": "MIT",
                    "payload_sha256": "0" * 64,
                    "source_commit": expected_source,
                    "source_license_blob_sha256": "0" * 64,
                    "approval": {
                        "schema": "v1",
                        "decision": "APPROVED",
                        "signer": "self-test-signer",
                        "signature_sha256": "2" * 64,
                    },
                },
                {
                    "id": "bigvgan-model-license",
                    "status": "REVIEWED",
                    "license": "MIT",
                    "payload_sha256": "1" * 64,
                    "hf_revision": "b" * 40,
                    "model_license_blob_sha256": "1" * 64,
                    "hf_card_data_object_sha256": "1" * 64,
                    "checkpoint_sha256": "c" * 64,
                    "config_sha256": "d" * 64,
                    "approval": {
                        "schema": "v1",
                        "decision": "APPROVED",
                        "signer": "self-test-signer",
                        "signature_sha256": "2" * 64,
                    },
                },
                {
                    "id": "python-cpu-closure-native-bundled",
                    "status": "REVIEWED",
                    "license": "MIT",
                    "payload_sha256": "3" * 64,
                    "lock_sha256": expected_lock,
                    "native_bundled_review": "self-test closure review",
                    "approval": {
                        "schema": "v1",
                        "decision": "APPROVED",
                        "signer": "self-test-signer",
                        "signature_sha256": "2" * 64,
                    },
                }
            ],
        }
        expected_evidence["manifest_sha256"] = digest_bytes(manifest_path.read_bytes())
        evidence_path = root / "evidence.json"

        def expect_manifest_blocked(label: str, mutate: Any) -> None:
            candidate = json.loads(json.dumps(manifest))
            mutate(candidate)
            candidate_path = root / f"{label}.manifest.json"
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            try:
                run(lock, project, candidate_path, evidence_path, expected_source, "b" * 40, "c" * 64, "d" * 64)
            except SystemExit as exc:
                if exc.code != 2:
                    raise SystemExit(f"bigvgan license gate self-test: {label} returned {exc.code}") from exc
            except Exception as exc:
                raise SystemExit(f"bigvgan license gate self-test: {label} raised unexpectedly: {exc}") from exc
            else:
                raise SystemExit(f"bigvgan license gate self-test: {label} was accepted")

        def expect_blocked(label: str, mutate: Any) -> None:
            candidate = json.loads(json.dumps(expected_evidence))
            mutate(candidate)
            evidence_path.write_text(json.dumps(candidate), encoding="utf-8")
            try:
                run(lock, project, manifest_path, evidence_path, expected_source, "b" * 40, "c" * 64, "d" * 64)
            except SystemExit as exc:
                if exc.code != 2:
                    raise SystemExit(f"bigvgan license gate self-test: {label} returned {exc.code}") from exc
            else:
                raise SystemExit(f"bigvgan license gate self-test: {label} was accepted")

        try:
            run(lock, project, manifest_path, None, expected_source, "b" * 40, "c" * 64, "d" * 64)
        except SystemExit as exc:
            if exc.code != 2:
                raise SystemExit("bigvgan license gate self-test: missing evidence did not exit 2") from exc
        else:
            raise SystemExit("bigvgan license gate self-test: missing evidence was accepted")

        evidence_path.write_text(json.dumps(expected_evidence), encoding="utf-8")
        run(
            lock,
            project,
            manifest_path,
            evidence_path,
            expected_source,
            "b" * 40,
            "c" * 64,
            "d" * 64,
        )
        duplicate_manifest = root / "duplicate-manifest.json"
        duplicate_manifest.write_text('{"approval": 1, "approval": 2}', encoding="utf-8")
        try:
            run(lock, project, duplicate_manifest, evidence_path, expected_source, "b" * 40, "c" * 64, "d" * 64)
        except SystemExit as exc:
            if exc.code != 2:
                raise SystemExit("bigvgan license gate self-test: duplicate manifest did not exit 2") from exc
        except Exception as exc:
            raise SystemExit(f"bigvgan license gate self-test: duplicate manifest raised unexpectedly: {exc}") from exc
        else:
            raise SystemExit("bigvgan license gate self-test: duplicate manifest was accepted")
        duplicate_evidence = root / "duplicate-evidence.json"
        duplicate_evidence.write_text('{"signer": 1, "signer": 2}', encoding="utf-8")
        try:
            run(lock, project, manifest_path, duplicate_evidence, expected_source, "b" * 40, "c" * 64, "d" * 64)
        except SystemExit as exc:
            if exc.code != 2:
                raise SystemExit("bigvgan license gate self-test: duplicate evidence did not exit 2") from exc
        except Exception as exc:
            raise SystemExit(f"bigvgan license gate self-test: duplicate evidence raised unexpectedly: {exc}") from exc
        else:
            raise SystemExit("bigvgan license gate self-test: duplicate evidence was accepted")
        expect_manifest_blocked("required-package-missing", lambda value: value["required_package_rows"].pop())
        expect_manifest_blocked("required-package-extra", lambda value: value["required_package_rows"].append({"name": "extra", "version": "1"}))
        expect_manifest_blocked("required-package-duplicate", lambda value: value["required_package_rows"].append(dict(value["required_package_rows"][0])))
        expect_manifest_blocked("license-row-nondict", lambda value: value["license_rows"].__setitem__(0, "malformed"))
        expect_manifest_blocked("license-row-missing", lambda value: value["license_rows"][0].pop("review"))
        expect_manifest_blocked("license-row-extra", lambda value: value["license_rows"][0].update(extra="unexpected"))
        expect_manifest_blocked("license-row-reordered", lambda value: value.update(license_rows=list(reversed(value["license_rows"]))))
        expect_manifest_blocked("license-row-duplicate", lambda value: value["license_rows"].__setitem__(1, dict(value["license_rows"][0])))
        expect_blocked("evidence-row-missing", lambda value: value["rows"].pop())
        expect_blocked("evidence-row-extra", lambda value: value["rows"].append(dict(value["rows"][0])))
        expect_blocked("evidence-row-duplicate", lambda value: value["rows"].__setitem__(1, dict(value["rows"][0])))
        expect_blocked("approval tamper", lambda value: value["rows"][0]["approval"].update(signature_sha256="4" * 64))
        expect_blocked("source revision tamper", lambda value: value["rows"][0].update(source_commit="e" * 40))
        expect_blocked("source payload tamper", lambda value: value["rows"][0].update(payload_sha256="4" * 64))
        expect_blocked("license tamper", lambda value: value["rows"][0].update(license="Apache-2.0"))
        expect_blocked("model revision tamper", lambda value: value["rows"][1].update(hf_revision="e" * 40))
        expect_blocked("checkpoint tamper", lambda value: value["rows"][1].update(checkpoint_sha256="e" * 64))
        expect_blocked("config tamper", lambda value: value["rows"][1].update(config_sha256="e" * 64))
        expect_blocked("lock tamper", lambda value: value["rows"][2].update(lock_sha256="e" * 64))
        expect_blocked("native closure tamper", lambda value: value["rows"][2].update(native_bundled_review=""))
        duplicate_json = root / "duplicate.json"
        duplicate_json.write_text('{"approval": 1, "approval": 2}', encoding="utf-8")
        try:
            load_json(duplicate_json)
        except ValueError:
            pass
        else:
            raise SystemExit("bigvgan license gate self-test accepted duplicate JSON keys")
    print("license_gate.py self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--license-evidence", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument("--model-revision")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--config-sha256")
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.lock, args.project, args.manifest, args.license_evidence, args.source_revision, args.model_revision, args.checkpoint_sha256, args.config_sha256)):
            parser.error("--self-test accepts no other arguments")
        self_test()
        return 0
    if args.lock is None or args.project is None or args.manifest is None or any(value is None for value in (args.source_revision, args.model_revision, args.checkpoint_sha256, args.config_sha256)):
        parser.error("--lock, --project, --manifest, --source-revision, --model-revision, --checkpoint-sha256, and --config-sha256 are required")
    run(args.lock, args.project, args.manifest, args.license_evidence, args.source_revision, args.model_revision, args.checkpoint_sha256, args.config_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
