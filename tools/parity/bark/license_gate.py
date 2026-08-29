#!/usr/bin/env python3
"""Offline, fail-closed Bark dependency and provenance approval gate."""

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
APPROVAL_SCOPE_SCHEMA = "bark-approval-scope-v1"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
UNRESOLVED = ("UNRESOLVED", "OWNER_REVIEW_REQUIRED", "PENDING_REVIEW", "REVIEW_REQUIRED")
PACKAGE_REVIEW_KEYS = {"name", "version", "source", "license", "status", "native_bundled_review"}
MANIFEST_KEYS = {
    "gate_version", "lock_sha256", "project_sha256", "package_rows_sha256",
    "package_review_rows_sha256", "package_review_rows", "identities", "license_rows",
    "license_rows_sha256", "approval", "publication_decision",
}
LOCK_KEYS = {"version", "revision", "requires-python", "resolution-markers", "supported-markers", "package"}
PACKAGE_KEYS = {"name", "version", "source", "resolution-markers", "dependencies", "sdist", "wheels", "metadata"}
ARTIFACT_KEYS = {"url", "hash", "size", "upload-time"}
REGISTRY_PACKAGE_SCHEMAS = {
    frozenset({"name", "version", "source", "sdist", "wheels"}),
    frozenset({"name", "version", "source", "dependencies", "sdist", "wheels"}),
    frozenset({"name", "version", "source", "dependencies", "wheels"}),
}
DEPENDENCY_SCHEMAS = {frozenset({"name", "marker"})}
METADATA_REQUIREMENT_SCHEMAS = {
    frozenset({"name", "specifier"}), frozenset({"name", "specifier", "index"}),
}


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: Any) -> str:
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


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
        block(f"{label} artifact schema is malformed")
    expected_host = "download-r2.pytorch.org" if registry == "https://download.pytorch.org/whl/cpu" else "files.pythonhosted.org"
    parsed = urlsplit(value["url"]) if isinstance(value["url"], str) else None
    if parsed is None or parsed.scheme != "https" or parsed.netloc != expected_host or not parsed.path:
        block(f"{label} artifact URL is not the authenticated {expected_host} host")
    if not isinstance(value["hash"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value["hash"]):
        block(f"{label} artifact hash is not a SHA-256")
    if isinstance(value["size"], bool) or not isinstance(value["size"], int) or value["size"] <= 0:
        block(f"{label} artifact size is not positive")
    if not isinstance(value["upload-time"], str) or not value["upload-time"].strip():
        block(f"{label} artifact upload-time is missing")


def validate_metadata(value: Any, label: str) -> None:
    if not isinstance(value, dict) or set(value) != {"requires-dist"} or not isinstance(value["requires-dist"], list):
        block(f"{label} metadata schema is malformed")
    for requirement in value["requires-dist"]:
        if not isinstance(requirement, dict) or frozenset(requirement) not in METADATA_REQUIREMENT_SCHEMAS or not isinstance(requirement.get("name"), str) or not requirement["name"].strip() or not isinstance(requirement.get("specifier"), str) or not requirement["specifier"].strip():
            block(f"{label} metadata requirement is malformed")
        if "index" in requirement and requirement["index"] not in ("https://pypi.org/simple", "https://download.pytorch.org/whl/cpu"):
            block(f"{label} metadata index is not approved")


def rows(lock: dict[str, Any]) -> list[dict[str, Any]]:
    packages = lock.get("package")
    if not isinstance(packages, list) or not packages:
        block("uv.lock package table is missing, empty, or malformed")
    if set(lock) != LOCK_KEYS or lock.get("version") != 1 or lock.get("revision") != 3:
        block("uv.lock top-level schema drifted")
    if lock.get("requires-python") != "==3.12.*" or not isinstance(lock.get("resolution-markers"), list) or any(not isinstance(item, str) or not item.strip() for item in lock["resolution-markers"]) or not isinstance(lock.get("supported-markers"), list) or any(not isinstance(item, str) or not item.strip() for item in lock["supported-markers"]):
        block("uv.lock marker schema is malformed")
    virtual_count = 0
    identities: set[tuple[str, str]] = set()
    canonical_rows: list[dict[str, Any]] = []
    for package in packages:
        if not isinstance(package, dict):
            block("uv.lock contains a malformed package row")
        if set(package) - PACKAGE_KEYS:
            block("uv.lock package row schema drifted")
        if not isinstance(package.get("name"), str) or not package["name"].strip():
            block("uv.lock package name is not a nonempty string")
        if not isinstance(package.get("version"), str) or not package["version"].strip():
            block("uv.lock package version is not a nonempty string")
        identity = (package["name"], package["version"])
        if identity in identities:
            block(f"uv.lock contains duplicate package identity: {identity!r}")
        identities.add(identity)
        source = package.get("source")
        if not isinstance(source, dict) or len(source) != 1 or set(source) not in ({"registry"}, {"virtual"}):
            block(f"uv.lock source schema is malformed: {identity!r}")
        if ("virtual" in source and frozenset(package) != frozenset({"name", "version", "source", "dependencies", "metadata"})) or ("registry" in source and frozenset(package) not in REGISTRY_PACKAGE_SCHEMAS):
            block(f"uv.lock package row schema drifted: {identity!r}")
        registry = source.get("registry")
        if "registry" in source and registry not in ("https://pypi.org/simple", "https://download.pytorch.org/whl/cpu"):
            block(f"uv.lock registry source is not an approved index: {identity!r}")
        if "virtual" in source:
            virtual_count += 1
            if source["virtual"] != ".":
                block("uv.lock virtual project source is not '.'")
        markers = package.get("resolution-markers", [])
        if not isinstance(markers, list) or any(not isinstance(item, str) or not item.strip() for item in markers):
            block(f"uv.lock package markers are malformed: {identity!r}")
        dependencies = package.get("dependencies", [])
        if not isinstance(dependencies, list):
            block(f"uv.lock dependencies are malformed: {identity!r}")
        for dependency in dependencies:
            if not isinstance(dependency, dict) or frozenset(dependency) not in DEPENDENCY_SCHEMAS or not isinstance(dependency.get("name"), str):
                block(f"uv.lock dependency row is malformed: {identity!r}")
            if not isinstance(dependency.get("marker"), str) or not dependency["marker"].strip():
                    block(f"uv.lock dependency field is malformed: {identity!r}")
        if "virtual" in source:
            if set(package) != {"name", "version", "source", "dependencies", "metadata"}:
                block(f"{identity!r} virtual package schema is malformed")
            validate_metadata(package["metadata"], f"{identity!r} virtual")
            if "sdist" in package or "wheels" in package:
                block(f"{identity!r} virtual package must not contain artifacts")
        else:
            if "metadata" in package:
                validate_metadata(package["metadata"], f"{identity!r}")
            if "sdist" in package:
                validate_artifact(package["sdist"], f"{identity!r} sdist", registry)
            if "wheels" in package:
                if not isinstance(package["wheels"], list) or not package["wheels"]:
                    block(f"{identity!r} wheels table is malformed")
                for artifact in package["wheels"]:
                    validate_artifact(artifact, f"{identity!r} wheel", registry)
            if "sdist" not in package and not package.get("wheels"):
                block(f"{identity!r} registry package has no authenticated artifacts")
        canonical_rows.append(package)
    if virtual_count != 1:
        block("uv.lock must contain exactly one virtual project package")
    return sorted(canonical_rows, key=lambda row: (row["name"], row["version"]))


def block(message: str) -> None:
    print(f"bark license gate: BLOCKED: {message}", file=sys.stderr)
    raise SystemExit(2)


def validate_project_schema(project: dict[str, Any]) -> None:
    if set(project) != {"project", "tool"} or not isinstance(project["project"], dict) or set(project["project"]) != {"name", "version", "description", "requires-python", "dependencies"}:
        block("pyproject.toml structural schema drifted")
    fields = project["project"]
    if any(not isinstance(fields[key], str) or not fields[key].strip() for key in ("name", "version", "description", "requires-python")):
        block("pyproject.toml project field types drifted")
    if fields["requires-python"] != "==3.12.*" or not isinstance(fields["dependencies"], list) or any(not isinstance(item, str) or not item.strip() for item in fields["dependencies"]):
        block("pyproject.toml Python/dependency contract drifted")
    tool = project["tool"]
    if set(tool) != {"uv"} or not isinstance(tool["uv"], dict) or set(tool["uv"]) != {"package", "environments", "sources", "index"}:
        block("pyproject.toml uv schema drifted")
    uv = tool["uv"]
    if uv["package"] is not False or uv["environments"] != ["sys_platform == 'linux' and platform_machine == 'x86_64'"] or uv["sources"] != {"torch": {"index": "pytorch-cpu"}} or uv["index"] != [{"name": "pytorch-cpu", "url": "https://download.pytorch.org/whl/cpu", "explicit": True}]:
        block("pyproject.toml uv environment contract drifted")


def resolved(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    normalized = re.sub(r"\s+", "_", value.strip()).casefold()
    return normalized not in {marker.casefold() for marker in UNRESOLVED} and normalized not in {"", "null", "none", "todo", "pending"}


def approval_scope(manifest: dict[str, Any], package_rows: list[dict[str, Any]] | None = None) -> str:
    """Digest every reviewed input and the explicit no-publication decision."""
    payload = {
        "schema": APPROVAL_SCOPE_SCHEMA,
        "lock_sha256": manifest.get("lock_sha256"),
        "project_sha256": manifest.get("project_sha256"),
        "package_rows_sha256": manifest.get("package_rows_sha256"),
        "package_rows": package_rows,
        "package_review_rows": manifest.get("package_review_rows"),
        "package_review_rows_sha256": manifest.get("package_review_rows_sha256"),
        "license_rows": manifest.get("license_rows"),
        "license_rows_sha256": manifest.get("license_rows_sha256"),
        "identities": manifest.get("identities"),
        "publication_decision": manifest.get("publication_decision"),
        "expected_approval_decision": "APPROVED",
        "expected_approval_status": "OWNER_SIGNOFF_APPROVED",
    }
    return canonical(payload)


def run(lock_path: Path, project_path: Path, manifest_path: Path, approval_path: Path | None, expected: dict[str, Any]) -> None:
    if not all(regular_file(path) for path in (lock_path, project_path, manifest_path)):
        block("uv.lock, pyproject.toml, or tracked gate manifest is missing")
    try:
        manifest = load_json(manifest_path)
        lock_bytes = lock_path.read_bytes()
        project_bytes = project_path.read_bytes()
        lock = tomllib.loads(lock_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, json.JSONDecodeError, ValueError) as error:
        block(f"closure input is unreadable: {error}")
    if not isinstance(manifest, dict) or manifest.get("gate_version") != GATE_VERSION:
        block("unsupported gate manifest version")
    if set(manifest) != MANIFEST_KEYS:
        block("gate manifest top-level schema drifted")
    lock_sha = digest(lock_bytes)
    project_sha = digest(project_bytes)
    if lock_sha != manifest.get("lock_sha256"):
        block("uv.lock bytes differ from reviewed closure")
    if project_sha != manifest.get("project_sha256"):
        block("pyproject.toml bytes differ from reviewed closure")
    package_rows = rows(lock)
    try:
        project = tomllib.loads(project_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        block(f"pyproject.toml is not valid TOML: {error}")
    validate_project_schema(project)
    project_identity = project.get("project")
    virtual = [row for row in package_rows if row.get("source") == {"virtual": "."}]
    if not isinstance(project_identity, dict) or not isinstance(project_identity.get("name"), str) or not isinstance(project_identity.get("version"), str) or len(virtual) != 1:
        block("uv.lock virtual project is not bound to pyproject.toml")
    if (virtual[0]["name"], virtual[0]["version"]) != (project_identity["name"], project_identity["version"]):
        block("uv.lock virtual project identity differs from pyproject.toml")
    if canonical(package_rows) != manifest.get("package_rows_sha256"):
        block("package version/source/marker/dependency rows drifted")
    review_rows = manifest.get("package_review_rows")
    if not isinstance(review_rows, list) or len(review_rows) != len(package_rows):
        block("every locked package needs a version-keyed review row")
    actual = {(row["name"], row["version"]): row for row in package_rows}
    seen: set[tuple[str, str]] = set()
    for review in review_rows:
        if not isinstance(review, dict) or set(review) != PACKAGE_REVIEW_KEYS:
            block("package review row is malformed")
        key = (review.get("name"), review.get("version"))
        if key in seen or key not in actual:
            block(f"package review identity is not a unique lock row: {key!r}")
        seen.add(key)
        if review.get("source") != actual[key].get("source"):
            block(f"package review source drifted: {key!r}")
        if review.get("status") != "REVIEWED" or not resolved(review.get("license")) or not resolved(review.get("native_bundled_review")):
            block(f"package license/native review is unresolved: {key!r}")
    if seen != set(actual):
        block("package review rows do not cover the exact lock closure")
    if canonical(review_rows) != manifest.get("package_review_rows_sha256"):
        block("version-keyed package review rows drifted")

    identities = manifest.get("identities")
    if not isinstance(identities, dict) or any(identities.get(key) != value for key, value in expected.items()):
        block("fixed public/upstream/Transformers identities drifted")
    for key in ("small_public_revision", "small_upstream_revision", "full_public_revision", "full_upstream_revision", "transformers_source_revision"):
        if not HEX40.fullmatch(str(identities.get(key, ""))):
            block(f"invalid fixed revision: {key}")
    for key in (
        "small_public_sha256", "full_public_sha256", "small_checkpoint_sha256", "full_checkpoint_sha256",
        "small_config_sha256", "full_config_sha256", "generation_config_sha256", "transformers_sdist_sha256",
        "transformers_wheel_sha256",
    ):
        if not HEX64.fullmatch(str(identities.get(key, ""))):
            block(f"invalid fixed SHA-256 identity: {key}")
    for role in ("small", "full"):
        if identities.get(f"{role}_public_file") != "model.gguf" or identities.get(f"{role}_upstream_file") != "pytorch_model.bin":
            block(f"{role} file role identity drifted")
    license_rows = manifest.get("license_rows")
    expected_license_ids = ["bark-source-mit", "bark-small-weight-mit", "bark-full-weight-mit", "python-closure"]
    if not isinstance(license_rows, list) or len(license_rows) != len(expected_license_ids) or any(not isinstance(row, dict) for row in license_rows):
        block("separate MIT source/weight and Python closure rows are required")
    license_ids = [row.get("id") for row in license_rows]
    if license_ids != expected_license_ids or len(set(license_ids)) != len(expected_license_ids):
        block("separate MIT source/weight and Python closure rows are required")
    if canonical(license_rows) != manifest.get("license_rows_sha256"):
        block("license review rows drifted")
    for row in license_rows:
        expected_keys = {"id", "license", "status", "conclusion", "evidence"} if row.get("id") != "python-closure" else {"id", "license", "status", "native_bundled_review", "evidence"}
        if set(row) != expected_keys:
            block(f"license review row schema is malformed: {row.get('id') if isinstance(row, dict) else None}")
        if row.get("status") != "REVIEWED" or not resolved(row.get("license")) or not resolved(row.get("evidence")):
            block(f"license review is unresolved: {row.get('id')}")
        if row.get("id") == "python-closure":
            if not resolved(row.get("native_bundled_review")):
                block("Python native/bundled closure review is unresolved")
        elif not resolved(row.get("conclusion")):
            block(f"license conclusion is unresolved: {row.get('id')}")
    closure = next(row for row in license_rows if row.get("id") == "python-closure")
    if not resolved(closure.get("native_bundled_review")):
        block("Python native/bundled closure review is unresolved")
    approval = manifest.get("approval")
    if not isinstance(approval, dict) or set(approval) != {"status", "signer", "digest"}:
        block("owner approval schema drifted")
    if not isinstance(approval, dict) or approval.get("status") != "OWNER_SIGNOFF_APPROVED":
        block("owner sign-off remains required; tracked approval is fail-closed")
    signer, approval_digest = approval.get("signer"), approval.get("digest")
    if not isinstance(signer, str) or not signer or not HEX64.fullmatch(str(approval_digest or "")):
        block("tracked owner signer and approval digest are missing")
    if approval_path is None or not regular_file(approval_path):
        block("authenticated owner approval evidence is missing")
    try:
        evidence = load_json(approval_path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        block(f"owner approval evidence is unreadable: {error}")
    if not isinstance(evidence, dict) or set(evidence) != {"signer", "decision", "scope_schema", "scope_sha256", "approval_digest", "manifest_sha256", "lock_sha256", "project_sha256"} or evidence.get("signer") != signer or evidence.get("decision") != "APPROVED" or evidence.get("approval_digest") != approval_digest:
        block("owner approval evidence does not match tracked sign-off")
    if evidence.get("manifest_sha256") != digest(manifest_path.read_bytes()) or evidence.get("lock_sha256") != lock_sha or evidence.get("project_sha256") != project_sha:
        block("owner approval evidence is not bound to this exact closure")
    scope = approval_scope(manifest, package_rows)
    if manifest.get("publication_decision") != "NO_UPLOAD":
        block("publication decision is not the explicit NO_UPLOAD policy")
    if approval.get("digest") != scope:
        block("owner approval digest does not cover the canonical closure scope")
    if evidence.get("scope_schema") != APPROVAL_SCOPE_SCHEMA or evidence.get("scope_sha256") != scope or evidence.get("approval_digest") != scope:
        block("owner approval evidence is not bound to the canonical closure scope")
    print("bark license gate: PASS")


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="bark-license-gate-") as raw:
        root = Path(raw)
        lock = root / "uv.lock"
        project = root / "pyproject.toml"
        manifest_path = root / "manifest.json"
        evidence_path = root / "approval.json"
        lock.write_text("version = 1\nrevision = 3\nrequires-python = '==3.12.*'\nresolution-markers = []\nsupported-markers = []\n\n[[package]]\nname = 'demo'\nversion = '0.0.0'\nsource = { virtual = '.' }\ndependencies = []\nmetadata = { requires-dist = [] }\n", encoding="utf-8")
        project.write_text("""[project]
name = 'demo'
version = '0.0.0'
description = 'self-test'
requires-python = '==3.12.*'
dependencies = []

[tool.uv]
package = false
environments = ["sys_platform == 'linux' and platform_machine == 'x86_64'"]

[tool.uv.sources]
torch = { index = 'pytorch-cpu' }

[[tool.uv.index]]
name = 'pytorch-cpu'
url = 'https://download.pytorch.org/whl/cpu'
explicit = true
""", encoding="utf-8")
        expected = {
            "small_public_revision": "a" * 40, "small_upstream_revision": "b" * 40,
            "full_public_revision": "c" * 40, "full_upstream_revision": "d" * 40,
            "transformers_source_revision": "e" * 40,
            "small_public_sha256": "1" * 64, "full_public_sha256": "2" * 64,
            "small_checkpoint_sha256": "3" * 64, "full_checkpoint_sha256": "4" * 64,
            "small_config_sha256": "5" * 64, "full_config_sha256": "6" * 64,
            "generation_config_sha256": "7" * 64, "transformers_sdist_sha256": "8" * 64,
            "transformers_wheel_sha256": "9" * 64,
            "small_public_file": "model.gguf", "small_upstream_file": "pytorch_model.bin",
            "full_public_file": "model.gguf", "full_upstream_file": "pytorch_model.bin",
        }
        package_rows = rows(tomllib.loads(lock.read_text(encoding="utf-8")))
        valid_artifact = {
            "url": "https://files.pythonhosted.org/packages/demo.whl",
            "hash": "sha256:" + "0" * 64,
            "size": 1,
            "upload-time": "2024-01-01T00:00:00Z",
        }
        artifact_tamper_cases = {
            "missing-size": lambda value: value.pop("size"),
            "missing-upload-time": lambda value: value.pop("upload-time"),
            "extra-key": lambda value: value.update(extra="x"),
            "bool-size": lambda value: value.update(size=True),
            "wrong-host": lambda value: value.update(url="https://example.invalid/demo.whl"),
        }
        for label, mutate in artifact_tamper_cases.items():
            candidate = dict(valid_artifact)
            mutate(candidate)
            try:
                validate_artifact(candidate, f"self-test {label}", "https://pypi.org/simple")
            except SystemExit as exc:
                if exc.code != 2:
                    raise
            else:
                raise SystemExit(f"bark license gate self-test accepted {label} artifact")
        registry_without_artifacts = tomllib.loads("""version = 1
revision = 3
requires-python = '==3.12.*'
resolution-markers = []
supported-markers = []

[[package]]
name = 'demo'
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
            rows(registry_without_artifacts)
        except SystemExit as exc:
            if exc.code != 2:
                raise
        else:
            raise SystemExit("bark license gate self-test accepted a registry package without artifacts")
        package_review = [{"name": "demo", "version": "0.0.0", "source": {"virtual": "."}, "license": "MIT", "status": "REVIEWED", "native_bundled_review": "demo review"}]
        license_rows = [
            {"id": "bark-source-mit", "license": "MIT", "status": "REVIEWED", "conclusion": "self-test source conclusion", "evidence": "self-test evidence"},
            {"id": "bark-small-weight-mit", "license": "MIT", "status": "REVIEWED", "conclusion": "self-test weight conclusion", "evidence": "self-test evidence"},
            {"id": "bark-full-weight-mit", "license": "MIT", "status": "REVIEWED", "conclusion": "self-test weight conclusion", "evidence": "self-test evidence"},
            {"id": "python-closure", "license": "MIT", "status": "REVIEWED", "native_bundled_review": "demo review", "evidence": "self-test evidence"},
        ]
        manifest = {
            "gate_version": GATE_VERSION, "lock_sha256": digest(lock.read_bytes()), "project_sha256": digest(project.read_bytes()),
            "package_rows_sha256": canonical(package_rows), "package_review_rows": package_review,
            "package_review_rows_sha256": canonical(package_review), "license_rows": license_rows,
            "license_rows_sha256": canonical(license_rows), "identities": expected,
            "publication_decision": "NO_UPLOAD",
            "approval": {"status": "OWNER_SIGNOFF_APPROVED", "signer": "self-test", "digest": None},
        }
        manifest["approval"]["digest"] = approval_scope(manifest, package_rows)
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        evidence = {"signer": "self-test", "decision": "APPROVED", "scope_schema": APPROVAL_SCOPE_SCHEMA, "scope_sha256": manifest["approval"]["digest"], "approval_digest": manifest["approval"]["digest"], "manifest_sha256": digest(manifest_path.read_bytes()), "lock_sha256": manifest["lock_sha256"], "project_sha256": manifest["project_sha256"]}
        evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        run(lock, project, manifest_path, evidence_path, expected)

        def unreadable(label: str, path: Path, candidate_manifest: Path) -> None:
            try:
                run(lock, project, candidate_manifest, path, expected)
            except SystemExit as error:
                if error.code != 2:
                    raise
            except Exception as error:
                raise SystemExit(f"bark license gate self-test {label} raised unexpectedly: {error}") from error
            else:
                raise SystemExit(f"bark license gate self-test accepted {label}")

        duplicate_manifest = root / "duplicate-manifest.json"
        duplicate_manifest.write_text('{"approval": 1, "approval": 2}', encoding="utf-8")
        unreadable("duplicate manifest", evidence_path, duplicate_manifest)
        duplicate_evidence = root / "duplicate-evidence.json"
        duplicate_evidence.write_text('{"signer": 1, "signer": 2}', encoding="utf-8")
        unreadable("duplicate evidence", duplicate_evidence, manifest_path)

        def blocked(label: str, mutate: Any) -> None:
            candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
            mutate(candidate)
            manifest_path.write_text(json.dumps(candidate, sort_keys=True), encoding="utf-8")
            try:
                run(lock, project, manifest_path, evidence_path, expected)
            except SystemExit as error:
                if error.code != 2:
                    raise
            else:
                raise SystemExit(f"bark license gate self-test accepted {label}")
            manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

        blocked("revision tamper", lambda value: value["identities"].update(small_public_revision="f" * 40))
        blocked("hash tamper", lambda value: value["identities"].update(full_public_sha256="f" * 64))
        blocked("license tamper", lambda value: value["license_rows"].__getitem__(0).update(license="Apache-2.0"))
        blocked("closure tamper", lambda value: value["package_review_rows"].__getitem__(0).update(native_bundled_review="OWNER_REVIEW_REQUIRED"))
        blocked("approval tamper", lambda value: value["approval"].update(status="OWNER_SIGNOFF_REQUIRED"))
        blocked("scope tamper", lambda value: value["approval"].update(digest="a" * 64))
        blocked("publication decision tamper", lambda value: value.update(publication_decision="UPLOAD"))
        candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
        candidate["approval"]["digest"] = "a" * 64
        manifest_path.write_text(json.dumps(candidate, sort_keys=True), encoding="utf-8")
        evidence["manifest_sha256"] = digest(manifest_path.read_bytes())
        evidence["scope_sha256"] = "a" * 64
        evidence["approval_digest"] = "a" * 64
        evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        try:
            run(lock, project, manifest_path, evidence_path, expected)
        except SystemExit as error:
            if error.code != 2:
                raise
        else:
            raise SystemExit("bark license gate self-test accepted an arbitrary approval digest")
        try:
            rows({"package": []})
        except SystemExit as error:
            if error.code != 2:
                raise
        else:
            raise SystemExit("bark license gate self-test accepted an empty lock")
        duplicate_json = root / "duplicate.json"
        duplicate_json.write_text('{"approval": 1, "approval": 2}', encoding="utf-8")
        try:
            load_json(duplicate_json)
        except ValueError:
            pass
        else:
            raise SystemExit("bark license gate self-test accepted duplicate JSON keys")
    print("license_gate.py self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--approval", type=Path)
    for field in ("small-public-repo", "small-upstream-repo", "full-public-repo", "full-upstream-repo", "small-public-revision", "small-upstream-revision", "full-public-revision", "full-upstream-revision", "transformers-version", "transformers-source-revision", "small-public-sha256", "full-public-sha256", "small-checkpoint-sha256", "full-checkpoint-sha256", "small-config-sha256", "full-config-sha256", "generation-config-sha256", "transformers-sdist-sha256", "transformers-wheel-sha256"):
        parser.add_argument(f"--{field}")
    for field in ("small-public-bytes", "full-public-bytes", "small-checkpoint-bytes", "full-checkpoint-bytes", "small-config-bytes", "full-config-bytes", "generation-config-bytes"):
        parser.add_argument(f"--{field}", type=int)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    paths = (args.lock, args.project, args.manifest)
    if any(path is None for path in paths):
        parser.error("--lock, --project, and --manifest are required")
    expected = {field.replace("-", "_"): getattr(args, field.replace("-", "_")) for field in ("small_public_repo", "small_upstream_repo", "full_public_repo", "full_upstream_repo", "transformers_version", "small_public_revision", "small_upstream_revision", "full_public_revision", "full_upstream_revision", "transformers_source_revision", "small_public_sha256", "full_public_sha256", "small_checkpoint_sha256", "full_checkpoint_sha256", "small_config_sha256", "full_config_sha256", "generation_config_sha256", "transformers_sdist_sha256", "transformers_wheel_sha256", "small_public_bytes", "full_public_bytes", "small_checkpoint_bytes", "full_checkpoint_bytes", "small_config_bytes", "full_config_bytes", "generation_config_bytes")}
    expected.update({"small_public_file": "model.gguf", "small_upstream_file": "pytorch_model.bin", "full_public_file": "model.gguf", "full_upstream_file": "pytorch_model.bin"})
    run(args.lock, args.project, args.manifest, args.approval, expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
