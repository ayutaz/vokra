#!/usr/bin/env python3
"""Dependency-free fail-closed gate for the WeSpeaker real-weight run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any
import tomllib
from urllib.parse import urlparse

GATE_VERSION = 1
# Filled from the committed bytes after the dedicated lock is finalized.
LOCK_SHA256 = "996f10762498f29a8f6c24d3403ebac4734118f8150137b716ddf5d54e512b6e"
PYPROJECT_SHA256 = "4d5a2bae9fdd3dff3d1224235c6e125995f32e491e3f42bb2063281d2a9d1850"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
DEPENDENCY_KEYS = (
    frozenset({"name"}),
    frozenset({"name", "marker"}),
    frozenset({"name", "extra"}),
    frozenset({"name", "extra", "marker"}),
)
REGISTRY_PACKAGE_KEYS = (
    frozenset({"name", "version", "source", "sdist", "wheels"}),
    frozenset({"name", "version", "source", "dependencies", "sdist", "wheels"}),
    frozenset({"name", "version", "source", "dependencies", "wheels"}),
)
VIRTUAL_PACKAGE_KEYS = frozenset({"name", "version", "source", "dependencies", "metadata"})
ARTIFACT_KEYS = {"url", "hash", "size", "upload-time"}
REGISTRY_URLS = {
    "https://pypi.org/simple": "files.pythonhosted.org",
    "https://download.pytorch.org/whl/cpu": "download-r2.pytorch.org",
}
PLACEHOLDERS = {"", "none", "null", "unresolved", "pending", "pending_review", "owner_review_required", "review_required", "todo"}
MANIFEST_FIELDS = {"approval_scope_sha256", "component_reviews", "dependency_reviews", "dependency_reviews_sha256", "fixed_identities", "gate_version", "lock_sha256", "no_upload", "operator_approval", "package_rows_sha256", "pyproject_sha256"}

SOURCE_REVISION = "45941e7cba2c3ea99e232d02bedf617fc71b0dad"
CHECKPOINT_REVISION = "f0c48c298fd835726c27956a5d617bad7115627e"
CHECKPOINT_SHA256 = "9872b375f2c6a3851ca471cbbf59e06efd23a627d78bf5872e1f0269fd298449"
PUBLIC_REVISION = "8e27acd8a875088f1a7321f40610397bf964a446"
PUBLIC_SHA256 = "6dccbc026e9c32a8f99f3441e64f1ff52e36afb055442595c86cda8021c78c39"
PUBLIC_BYTES = 26_584_064
JFK_SHA256 = "58adb4ea501d955fcd40bfbb69128f8f40428b81d8716b9ed337949773be253f"
JFK_BYTES = 352_078

SOURCE_IDENTITY = {
    "repo": "https://github.com/wenet-e2e/wespeaker.git",
    "revision": SOURCE_REVISION,
    "license_spdx": "apache-2.0",
    "license": {"path": "LICENSE", "bytes": 11357, "sha256": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4", "git_blob_sha1": "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64"},
    "files": [
        {"path": "wespeaker/models/resnet.py", "role": "official_loader", "bytes": 9564, "sha256": "6f3c8219be2c9a8b9eabed8169c1abaec3e48670be7aaf1e792138b2b20e68c4", "git_blob_sha1": "17607e6d2c72627e15db4214cacfa9d7b89ca945"},
        {"path": "wespeaker/models/pooling_layers.py", "role": "official_pooling", "bytes": 10255, "sha256": "768910f8e88cb47e742274563339d7e780cb9d56c629c4d4124605296686f0f9", "git_blob_sha1": "47120eead47a511939267470496539804c17b7d3"},
    ],
}

FIXED_IDENTITIES = {
    "source": SOURCE_IDENTITY,
    "checkpoint": {"repo": "Wespeaker/wespeaker-voxceleb-resnet34-LM", "revision": CHECKPOINT_REVISION, "path": "avg_model.pt", "bytes": 45053131, "sha256": CHECKPOINT_SHA256, "git_oid": "7f92ddd059d244c7d2653650d3be85de9f136c41", "config": {"path": "config.yaml", "bytes": 1673, "sha256": "3cf7d3243464cd939083e29d2be65c2abcdd954c1a64559bad73b74ffdb0db3e", "git_oid": "1941982501edc3909a56c9bca025fecf10cf28d2"}, "license_spdx": "cc-by-4.0", "owner_signoff_date": "2026-07-28", "role": "official_checkpoint"},
    "corrected_replacement": {"repo": "vokra/pyannote-wespeaker-voxceleb-resnet34-lm", "revision": PUBLIC_REVISION, "path": "pyannote-wespeaker.restamped.gguf", "bytes": PUBLIC_BYTES, "sha256": PUBLIC_SHA256, "license_spdx": "cc-by-4.0", "role": "authorized_corrected_replacement"},
    "legacy_mislicensed": {"repo": "vokra/wespeaker", "revision": "a20ec15a61be1b5c5cb0f4805dbf72bb341e946f", "path": "wespeaker.gguf", "bytes": None, "sha256": "d2dd9114179e28d14bd7c6ec372807823f1064c4f6cdc2349a83aa652635553d", "stamped_license_spdx": "apache-2.0", "actual_license_spdx": "cc-by-4.0", "accepted": False, "role": "rejected_mislicensed_legacy"},
    "fixture": {"path": "tests/fixtures/audio/jfk-30s.wav", "bytes": JFK_BYTES, "sha256": JFK_SHA256, "license_spdx": "public-domain", "role": "fixed_reference_input"},
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> str:
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def load_json(path: Path) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)


def unresolved(value: Any) -> bool:
    if value is None or not isinstance(value, str):
        return value is None
    return re.sub(r"\s+", "_", value.strip().casefold()) in PLACEHOLDERS


def artifact_valid(value: Any, expected_host: str) -> bool:
    """Validate one locked artifact row and bind its URL to the source host."""

    if (
        not isinstance(value, dict)
        or set(value) != ARTIFACT_KEYS
        or not isinstance(value.get("url"), str)
        or not isinstance(value.get("hash"), str)
        or not isinstance(value.get("size"), int)
        or isinstance(value.get("size"), bool)
        or value["size"] <= 0
        or not isinstance(value.get("upload-time"), str)
        or not value["upload-time"].strip()
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", value["hash"])
    ):
        return False
    try:
        parsed = urlparse(value["url"])
    except ValueError:
        return False
    return (
        parsed.scheme == "https"
        and parsed.hostname == expected_host
        and parsed.netloc == expected_host
        and parsed.path.startswith("/")
        and not parsed.query
        and not parsed.fragment
    )


def source_valid(value: Any) -> bool:
    if not isinstance(value, dict) or len(value) != 1 or set(value) not in ({"registry"}, {"virtual"}):
        return False
    if "registry" in value:
        return isinstance(value["registry"], str) and value["registry"] in REGISTRY_URLS
    return value["virtual"] == "."


def package_rows(lock: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(lock, dict) or set(lock) != {"version", "revision", "requires-python", "resolution-markers", "supported-markers", "package"}:
        raise ValueError("uv.lock top-level schema drifted")
    if (lock["version"] != 1 or lock["revision"] != 3
            or lock["requires-python"] != "==3.12.*"
            or lock["resolution-markers"] != ["platform_machine == 'x86_64' and sys_platform == 'linux'"]
            or lock["supported-markers"] != ["platform_machine == 'x86_64' and sys_platform == 'linux'"]):
        raise ValueError("uv.lock top-level types drifted")
    packages = lock.get("package")
    if not isinstance(packages, list) or not packages:
        raise ValueError("uv.lock package table is missing or empty")
    result = []
    identities: set[tuple[str, str, str]] = set()
    for package in packages:
        if (not isinstance(package, dict) or not isinstance(package.get("name"), str)
                or not package["name"].strip() or not isinstance(package.get("version"), str)
                or not package["version"].strip()):
            raise ValueError("uv.lock package row is malformed")
        source = package.get("source")
        if not isinstance(source, dict) or set(source) not in ({"virtual"}, {"registry"}):
            raise ValueError("uv.lock package source is malformed")
        markers = package.get("resolution-markers", [])
        dependencies = package.get("dependencies", [])
        sdist = package.get("sdist")
        wheels = package.get("wheels", [])
        if (not isinstance(markers, list) or any(not isinstance(marker, str) for marker in markers)
                or not isinstance(dependencies, list)
                or any(
                    not isinstance(dep, dict)
                    or frozenset(dep) not in DEPENDENCY_KEYS
                    or not isinstance(dep.get("name"), str)
                    or not dep["name"].strip()
                    or ("marker" in dep and (not isinstance(dep["marker"], str) or not dep["marker"].strip()))
                    or ("extra" in dep and (not isinstance(dep["extra"], str) or not dep["extra"].strip()))
                    for dep in dependencies
                )
                or not isinstance(wheels, list)
                or sdist is not None and not isinstance(sdist, dict)):
            raise ValueError("uv.lock package dependency rows are malformed")
        if source == {"virtual": "."}:
            if frozenset(package) != VIRTUAL_PACKAGE_KEYS:
                raise ValueError("uv.lock virtual package schema drifted")
            metadata = package.get("metadata")
            if (not isinstance(metadata, dict) or set(metadata) != {"requires-dist"}
                    or not isinstance(metadata["requires-dist"], list)
                    or not metadata["requires-dist"]):
                raise ValueError("uv.lock virtual metadata drifted")
            for requirement in metadata["requires-dist"]:
                if (not isinstance(requirement, dict)
                        or frozenset(requirement) not in (frozenset({"name", "specifier"}), frozenset({"name", "specifier", "index"}))
                        or not isinstance(requirement.get("name"), str) or not requirement["name"].strip()
                        or not isinstance(requirement.get("specifier"), str) or not requirement["specifier"].strip()
                        or ("index" in requirement and requirement["index"] != "https://download.pytorch.org/whl/cpu")):
                    raise ValueError("uv.lock requires-dist metadata drifted")
        else:
            if frozenset(package) not in REGISTRY_PACKAGE_KEYS:
                raise ValueError("uv.lock registry package schema drifted")
            registry = source.get("registry")
            if not isinstance(registry, str) or registry not in REGISTRY_URLS:
                raise ValueError("uv.lock registry is not reviewed")
            expected_host = REGISTRY_URLS[registry]
            artifacts = ([sdist] if sdist is not None else []) + wheels
            if not artifacts:
                raise ValueError("uv.lock registry package has no artifacts")
            if any(not artifact_valid(artifact, expected_host) for artifact in artifacts):
                raise ValueError("uv.lock artifact schema drifted")
        identity = (package["name"], package["version"], json.dumps(source, sort_keys=True))
        if identity in identities:
            raise ValueError("uv.lock package identities are duplicated")
        identities.add(identity)
        result.append({"name": package["name"], "version": package["version"], "source": source, "resolution-markers": markers, "dependencies": dependencies, "sdist": sdist, "wheels": wheels})
    return sorted(result, key=lambda row: (row["name"], row["version"], json.dumps(row["source"], sort_keys=True)))


def artifact_blocker(rows: list[dict[str, Any]]) -> str | None:
    virtual = 0
    for row in rows:
        if not isinstance(row, dict):
            return "malformed normalized package row"
        source = row.get("source")
        if source == {"virtual": "."}:
            virtual += 1
            continue
        if not isinstance(source, dict) or set(source) != {"registry"}:
            return f"malformed package source: {row.get('name')}"
        registry = source.get("registry")
        if not isinstance(registry, str) or registry not in REGISTRY_URLS:
            return f"malformed package source: {row.get('name')}"
        host = REGISTRY_URLS[registry]
        wheels = row.get("wheels")
        sdist = row.get("sdist")
        if not isinstance(wheels, list) or (sdist is not None and not isinstance(sdist, dict)):
            return f"resolver artifact rows are malformed: {row.get('name')}"
        artifacts = ([sdist] if sdist is not None else []) + wheels
        if not artifacts:
            return f"resolver artifact URL/hash/size is missing: {row.get('name')}"
        for artifact in artifacts:
            if not artifact_valid(artifact, host):
                return f"resolver artifact URL/hash/size is incomplete: {row.get('name')}"
    return None if virtual == 1 else "lock must contain exactly one virtual project row"


def expected_components() -> list[dict[str, Any]]:
    return [
        {"id": "source", "identity": SOURCE_IDENTITY, "license": "apache-2.0", "role": "official_tooling"},
        {"id": "checkpoint", "identity": FIXED_IDENTITIES["checkpoint"], "license": "cc-by-4.0", "role": "official_weight"},
        {"id": "corrected_replacement", "identity": FIXED_IDENTITIES["corrected_replacement"], "license": "cc-by-4.0", "role": "replacement_public_artifact"},
        {"id": "legacy_mislicensed", "identity": FIXED_IDENTITIES["legacy_mislicensed"], "license": "cc-by-4.0", "role": "rejected_legacy_artifact"},
        {"id": "fixture", "identity": FIXED_IDENTITIES["fixture"], "license": "public-domain", "role": "reference_input"},
    ]


def expected_dependency_reviews(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"id": f"{row['name']}@{row['version']}", "name": row["name"], "version": row["version"], "source": row["source"], "status": "PENDING_REVIEW", "license": None, "native_review": None, "bundled_review": None, "payload_sha256": None} for row in rows]


def fixed_blocker() -> str | None:
    if FIXED_IDENTITIES["legacy_mislicensed"]["accepted"] is not False:
        return "legacy mislicensed artifact rejection identity drifted"
    if FIXED_IDENTITIES["legacy_mislicensed"]["stamped_license_spdx"] == FIXED_IDENTITIES["legacy_mislicensed"]["actual_license_spdx"]:
        return "legacy mislicensed artifact no longer records the license mismatch"
    source_license = SOURCE_IDENTITY["license"]
    if not isinstance(source_license.get("bytes"), int) or not isinstance(source_license.get("sha256"), str) or not HEX64.fullmatch(source_license["sha256"]) or not isinstance(source_license.get("git_blob_sha1"), str) or not HEX40.fullmatch(source_license["git_blob_sha1"]):
        return "official source LICENSE identity is incomplete"
    for row in SOURCE_IDENTITY["files"]:
        if not isinstance(row.get("bytes"), int) or not HEX64.fullmatch(str(row.get("sha256"))) or not HEX40.fullmatch(str(row.get("git_blob_sha1"))):
            return f"official source file identity is incomplete: {row['path']}"
    if not isinstance(FIXED_IDENTITIES["checkpoint"]["bytes"], int):
        return "official checkpoint byte identity is incomplete"
    return None


def validate(project: Path, manifest_path: Path, evidence_path: Path | None = None, *, _self_test: bool = False) -> tuple[bool, str]:
    lock_path, pyproject_path = project / "uv.lock", project / "pyproject.toml"
    if any(path.is_symlink() or not path.is_file() for path in (lock_path, pyproject_path, manifest_path)):
        return False, "gate inputs must be regular non-symlink files"
    try:
        lock_bytes = lock_path.read_bytes(); pyproject_bytes = pyproject_path.read_bytes(); manifest = load_json(manifest_path)
        project_toml = tomllib.loads(pyproject_bytes.decode())
        project_metadata = project_toml.get("project") if isinstance(project_toml, dict) else None
        project_tool = project_toml.get("tool") if isinstance(project_toml, dict) else None
        if (not isinstance(project_toml, dict) or set(project_toml) != {"project", "tool"}
                or not isinstance(project_metadata, dict)
                or set(project_metadata) != {"name", "version", "description", "requires-python", "dependencies"}
                or not isinstance(project_tool, dict)):
            raise ValueError("pyproject root/project schema drifted")
        expected_project = {
            "name": "vokra-wespeaker-parity", "version": "0.1.0",
            "description": "Pinned independent WeSpeaker ResNet34-LM parity oracle",
            "requires-python": ">=3.12,<3.13",
            "dependencies": ["numpy==2.3.5", "safetensors==0.7.0", "torch==2.9.1", "torchaudio==2.9.1"],
        }
        if project_metadata != expected_project:
            raise ValueError("pyproject project metadata drifted")
        uv = project_tool.get("uv")
        if set(project_tool) != {"uv"} or not isinstance(uv, dict) or set(uv) != {"package", "environments", "sources", "index"}:
            raise ValueError("pyproject uv schema drifted")
        if uv != {
            "package": False,
            "environments": ["python_full_version == '3.12.*' and platform_machine == 'x86_64' and sys_platform == 'linux'"],
            "sources": {"torch": {"index": "pytorch-cpu"}, "torchaudio": {"index": "pytorch-cpu"}},
            "index": [{"name": "pytorch-cpu", "url": "https://download.pytorch.org/whl/cpu", "explicit": True}],
        }:
            raise ValueError("pyproject uv index/source configuration drifted")
        lock = tomllib.loads(lock_bytes.decode())
        rows = package_rows(lock)
    except (OSError, UnicodeError, json.JSONDecodeError, tomllib.TOMLDecodeError, TypeError, ValueError) as exc:
        return False, f"gate input malformed: {exc}"
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_FIELDS:
        return False, "manifest schema is not exact"
    if not isinstance(project_metadata, dict) or not isinstance(project_metadata.get("name"), str) or not isinstance(project_metadata.get("version"), str):
        return False, "pyproject project identity is malformed"
    virtual = [package for package in lock.get("package", []) if isinstance(package, dict) and package.get("source") == {"virtual": "."}]
    if len(virtual) != 1 or virtual[0].get("name") != project_metadata["name"] or virtual[0].get("version") != project_metadata["version"]:
        return False, "virtual project row does not bind to pyproject identity"
    if digest(lock_bytes) != LOCK_SHA256 or manifest.get("lock_sha256") != LOCK_SHA256:
        return False, "uv.lock bytes are not the fixed reviewed lock"
    if digest(pyproject_bytes) != PYPROJECT_SHA256 or manifest.get("pyproject_sha256") != PYPROJECT_SHA256:
        return False, "pyproject bytes are not the fixed reviewed project"
    if canonical(rows) != manifest.get("package_rows_sha256"):
        return False, "canonical package graph drifted"
    if not _self_test and artifact_blocker(rows):
        return False, artifact_blocker(rows) or "resolver artifact identity is unresolved"
    reviews = manifest.get("dependency_reviews")
    fields = {"id", "name", "version", "source", "status", "license", "native_review", "bundled_review", "payload_sha256"}
    expected_keys = {(r["name"], r["version"], json.dumps(r["source"], sort_keys=True)) for r in rows}
    seen = set()
    if not isinstance(reviews, list) or len(reviews) != len(rows) or canonical(reviews) != manifest.get("dependency_reviews_sha256"):
        return False, "dependency review closure is missing or unbound"
    for row in reviews:
        if not isinstance(row, dict) or set(row) != fields:
            return False, "dependency review schema is not exact"
        if (not isinstance(row.get("id"), str) or not isinstance(row.get("name"), str)
                or not row["name"].strip() or not isinstance(row.get("version"), str)
                or not row["version"].strip() or not source_valid(row.get("source"))
                or not isinstance(row.get("status"), str)
                or (row.get("license") is not None and not isinstance(row.get("license"), str))
                or (row.get("native_review") is not None and not isinstance(row.get("native_review"), str))
                or (row.get("bundled_review") is not None and not isinstance(row.get("bundled_review"), str))
                or (row.get("payload_sha256") is not None and not isinstance(row.get("payload_sha256"), str))):
            return False, "dependency review schema is not exact"
        key = (row["name"], row["version"], json.dumps(row["source"], sort_keys=True))
        if key in seen or key not in expected_keys or row.get("id") != f"{row.get('name')}@{row.get('version')}":
            return False, "dependency review identities are missing, extra, or duplicated"
        seen.add(key)
        if row.get("status") != "REVIEWED" or unresolved(row.get("license")) or not isinstance(row.get("license"), str) or not isinstance(row.get("native_review"), str) or unresolved(row.get("native_review")) or not isinstance(row.get("bundled_review"), str) or unresolved(row.get("bundled_review")) or not isinstance(row.get("payload_sha256"), str) or not HEX64.fullmatch(row["payload_sha256"]):
            return False, f"dependency review is unresolved: {row.get('id')}"
    if seen != expected_keys:
        return False, "dependency review coverage drifted"
    components = manifest.get("component_reviews"); expected = expected_components()
    if not isinstance(components, list) or len(components) != len(expected):
        return False, "source/checkpoint/replacement component rows are incomplete"
    component_fields = {"id", "identity", "license", "role", "status", "payload_sha256", "signer", "approval_digest"}
    for actual, fixed in zip(components, expected, strict=True):
        if not isinstance(actual, dict) or set(actual) != component_fields or actual.get("id") != fixed["id"] or actual.get("identity") != fixed["identity"]:
            return False, "fixed component identity drifted"
        if actual.get("status") != "REVIEWED" or not isinstance(actual.get("license"), str) or unresolved(actual.get("license")) or (fixed["license"] is not None and actual.get("license") != fixed["license"]) or not isinstance(actual.get("payload_sha256"), str) or not HEX64.fullmatch(actual["payload_sha256"]) or not isinstance(actual.get("signer"), str) or unresolved(actual["signer"]):
            return False, f"component review is unresolved: {actual.get('id')}"
        if actual.get("approval_digest") != canonical({k: actual[k] for k in ("id", "identity", "license", "status", "payload_sha256")}):
            return False, f"component approval is not bound: {actual.get('id')}"
    fixed = {"source": SOURCE_IDENTITY, **FIXED_IDENTITIES}
    if manifest.get("fixed_identities") != fixed or manifest.get("no_upload") != "NO_UPLOAD":
        return False, "fixed identity or NO_UPLOAD policy drifted"
    if not _self_test:
        reason = fixed_blocker()
        if reason:
            return False, reason
    scope = {"lock_sha256": LOCK_SHA256, "pyproject_sha256": PYPROJECT_SHA256, "package_rows_sha256": manifest["package_rows_sha256"], "dependency_reviews": reviews, "component_reviews": components, "fixed_identities": fixed, "no_upload": "NO_UPLOAD"}
    scope_sha = canonical(scope)
    if manifest.get("approval_scope_sha256") != scope_sha:
        return False, "approval scope is not bound to exact closure"
    approval = manifest.get("operator_approval")
    if not isinstance(approval, dict) or set(approval) != {"decision", "signer", "digest"} or approval.get("decision") != "APPROVED" or not isinstance(approval.get("signer"), str) or unresolved(approval.get("signer")) or approval.get("digest") != scope_sha:
        return False, "operator approval is pending or invalid"
    if any(row.get("signer") != approval["signer"] for row in components):
        return False, "component and operator signers differ"
    if evidence_path is None:
        return False, "external approval evidence is required"
    if evidence_path.is_symlink() or not evidence_path.is_file():
        return False, "external approval evidence must be a regular non-symlink file"
    try:
        evidence = load_json(evidence_path)
    except (OSError, UnicodeError, ValueError) as exc:
        return False, f"external approval evidence is malformed: {exc}"
    evidence_fields = {"decision", "digest", "evidence_sha256", "manifest_sha256", "scope_sha256", "signer"}
    if not isinstance(evidence, dict) or set(evidence) != evidence_fields:
        return False, "external approval evidence schema is not exact"
    unsigned_evidence = {key: evidence[key] for key in evidence_fields if key != "evidence_sha256"}
    if evidence.get("evidence_sha256") != canonical(unsigned_evidence):
        return False, "external approval evidence hash is not bound"
    if evidence.get("decision") != "APPROVED" or evidence.get("digest") != scope_sha or evidence.get("scope_sha256") != scope_sha or evidence.get("manifest_sha256") != digest(manifest_path.read_bytes()) or not isinstance(evidence.get("signer"), str) or unresolved(evidence.get("signer")) or evidence.get("signer") != approval["signer"]:
        return False, "external approval evidence does not bind the manifest/scope"
    return True, "PASS"


def self_test() -> int:
    project = Path(__file__).resolve().parent
    manifest = project / "license_gate_manifest.json"
    lock = tomllib.loads((project / "uv.lock").read_text(encoding="utf-8"))
    duplicate_lock = json.loads(json.dumps(lock))
    duplicate_lock["package"].append(duplicate_lock["package"][0])
    try:
        package_rows(duplicate_lock)
    except ValueError:
        pass
    else:
        print("wespeaker preflight gate: duplicate package identity accepted", file=sys.stderr); return 1
    def reject_lock_tamper(label: str, mutate: Any) -> bool:
        candidate = json.loads(json.dumps(lock))
        mutate(candidate)
        try:
            package_rows(candidate)
        except (TypeError, ValueError):
            return True
        print(f"wespeaker preflight gate: {label} lock tamper accepted", file=sys.stderr)
        return False

    structural_cases = (
        ("top-level extra", lambda value: value.update({"unexpected": True})),
        ("package extra", lambda value: value["package"][0].update({"unexpected": True})),
        ("source non-dict", lambda value: value["package"][0].update({"source": "not-a-source"})),
        ("source extra", lambda value: value["package"][0]["source"].update({"unexpected": True})),
        ("unknown registry", lambda value: value["package"][0]["source"].update({"registry": "https://pypi.example.invalid/simple"})),
        ("unknown virtual", lambda value: value["package"][-1].update({"source": {"virtual": "other"}})),
        ("dependency missing", lambda value: next(package for package in value["package"] if package.get("dependencies"))["dependencies"][0].pop("name")),
        ("dependency extra", lambda value: next(package for package in value["package"] if package.get("dependencies"))["dependencies"][0].update({"unexpected": True})),
        ("dependency type", lambda value: next(package for package in value["package"] if package.get("dependencies"))["dependencies"].__setitem__(0, "not-a-row")),
        ("virtual metadata extra", lambda value: next(package for package in value["package"] if package.get("source") == {"virtual": "."})["metadata"].update({"unexpected": True})),
        ("virtual metadata missing", lambda value: next(package for package in value["package"] if package.get("source") == {"virtual": "."})["metadata"]["requires-dist"][0].pop("name")),
        ("virtual metadata extra row field", lambda value: next(package for package in value["package"] if package.get("source") == {"virtual": "."})["metadata"]["requires-dist"][0].update({"unexpected": True})),
        ("virtual metadata row type", lambda value: next(package for package in value["package"] if package.get("source") == {"virtual": "."})["metadata"]["requires-dist"].__setitem__(0, "not-a-row")),
    )
    for label, mutate in structural_cases:
        if not reject_lock_tamper(label, mutate):
            return 1
    for artifact_field in ("sdist", "wheels"):
        source_package = next(package for package in lock["package"] if package.get("source", {}).get("registry") and package.get(artifact_field))
        source_name, source_version = source_package["name"], source_package["version"]
        for label, mutate in (
            ("missing-url", lambda value: value.pop("url")),
            ("extra-field", lambda value: value.update({"unexpected": True})),
            ("non-positive-size", lambda value: value.update({"size": 0})),
            ("boolean-size", lambda value: value.update({"size": True})),
            ("blank-upload-time", lambda value: value.update({"upload-time": " "})),
            ("wrong-host", lambda value: value.update({"url": "https://evil.example/artifact.whl"})),
        ):
            def mutate_artifact(candidate: dict[str, Any], mutate: Any = mutate, artifact_field: str = artifact_field, source_name: str = source_name, source_version: str = source_version) -> None:
                package = next(item for item in candidate["package"] if item.get("name") == source_name and item.get("version") == source_version)
                artifacts = package[artifact_field]
                mutate(artifacts[0] if isinstance(artifacts, list) else artifacts)
            if not reject_lock_tamper(f"{artifact_field} {label}", mutate_artifact):
                return 1
    malformed_artifact = [{"name": "fixture", "version": "1", "source": {"registry": "https://pypi.org/simple"}, "sdist": None, "wheels": [{"url": "https://example.invalid/a.whl", "hash": "sha256:" + "a" * 64, "size": True}]}]
    if artifact_blocker(malformed_artifact) is None:
        print("wespeaker preflight gate: boolean artifact size accepted", file=sys.stderr); return 1
    valid_artifact = {"url": "https://files.pythonhosted.org/packages/a.whl", "hash": "sha256:" + "a" * 64, "size": 1, "upload-time": "2026-01-01T00:00:00Z"}
    artifact_rows = [{"name": "fixture", "version": "1", "source": {"registry": "https://pypi.org/simple"}, "sdist": None, "wheels": [valid_artifact]}, {"name": "root", "version": "1", "source": {"virtual": "."}, "sdist": None, "wheels": []}]
    for label, mutate in (("missing-time", lambda value: value.pop("upload-time")), ("blank-time", lambda value: value.update({"upload-time": "  "})), ("extra-field", lambda value: value.update({"extra": 1})), ("wrong-host", lambda value: value.update({"url": "https://evil.example/a.whl"}))):
        candidate = json.loads(json.dumps(artifact_rows)); mutate(candidate[0]["wheels"][0])
        if artifact_blocker(candidate) is None:
            print(f"wespeaker preflight gate: {label} artifact accepted", file=sys.stderr); return 1
    ok, reason = validate(project, manifest)
    if ok or not ("unresolved" in reason or "pending" in reason or "artifact" in reason):
        print(f"wespeaker preflight gate: expected pending review, got {reason}", file=sys.stderr); return 1
    with tempfile.TemporaryDirectory(prefix="wespeaker-gate-") as directory:
        root = Path(directory)
        test_project = root / "project"
        test_project.mkdir()
        shutil.copy2(project / "uv.lock", test_project / "uv.lock")
        shutil.copy2(project / "pyproject.toml", test_project / "pyproject.toml")
        base = json.loads(manifest.read_text(encoding="utf-8"))
        for row in base["dependency_reviews"]:
            row.update(status="REVIEWED", license="SELF_TEST", native_review="SELF_TEST",
                       bundled_review="SELF_TEST", payload_sha256="a" * 64)
        for row, fixed in zip(base["component_reviews"], expected_components(), strict=True):
            row.update(status="REVIEWED", license=fixed["license"] or "SELF_TEST", payload_sha256="b" * 64,
                       signer="self-test")
            row["approval_digest"] = canonical({key: row[key] for key in
                ("id", "identity", "license", "status", "payload_sha256",)})
        base["dependency_reviews_sha256"] = canonical(base["dependency_reviews"])
        scope = {"lock_sha256": LOCK_SHA256, "pyproject_sha256": PYPROJECT_SHA256,
                 "package_rows_sha256": base["package_rows_sha256"],
                 "dependency_reviews": base["dependency_reviews"],
                 "component_reviews": base["component_reviews"],
                 "fixed_identities": base["fixed_identities"], "no_upload": "NO_UPLOAD"}
        base["approval_scope_sha256"] = canonical(scope)
        base["operator_approval"] = {"decision": "APPROVED", "signer": "self-test",
                                     "digest": base["approval_scope_sha256"]}
        approved = root / "manifest.json"
        approved.write_text(json.dumps(base), encoding="utf-8")
        evidence = root / "evidence.json"
        evidence.write_text(json.dumps({
            "scope_sha256": base["approval_scope_sha256"],
            "manifest_sha256": digest(approved.read_bytes()), "signer": "self-test",
            "digest": base["approval_scope_sha256"], "decision": "APPROVED",
        }), encoding="utf-8")
        evidence_data = {"scope_sha256": base["approval_scope_sha256"], "manifest_sha256": digest(approved.read_bytes()), "signer": "self-test", "digest": base["approval_scope_sha256"], "decision": "APPROVED"}
        evidence_data["evidence_sha256"] = canonical(evidence_data)
        evidence.write_text(json.dumps(evidence_data), encoding="utf-8")
        baseline = validate(test_project, approved, evidence, _self_test=True)
        if baseline != (True, "PASS"):
            print(f"wespeaker preflight gate: approved self-test baseline failed: {baseline[1]}", file=sys.stderr)
            return 1
        duplicate_manifest = root / "duplicate-manifest.json"
        duplicate_manifest.write_text('{"gate_version":1,"gate_version":1}', encoding="utf-8")
        if validate(test_project, duplicate_manifest, evidence, _self_test=True)[0]:
            print("wespeaker preflight gate: duplicate manifest key accepted", file=sys.stderr)
            return 1
        duplicate_evidence = root / "duplicate-evidence.json"
        duplicate_evidence.write_text('{"decision":"APPROVED","decision":"APPROVED"}', encoding="utf-8")
        if validate(test_project, approved, duplicate_evidence, _self_test=True)[0]:
            print("wespeaker preflight gate: duplicate evidence key accepted", file=sys.stderr)
            return 1
        for label, mutate in (
            ("missing dependency review", lambda value: value["dependency_reviews"].pop()),
            ("extra dependency review", lambda value: value["dependency_reviews"].append(value["dependency_reviews"][0].copy())),
        ):
            candidate = json.loads(approved.read_text(encoding="utf-8"))
            mutate(candidate)
            candidate["dependency_reviews_sha256"] = canonical(candidate["dependency_reviews"])
            candidate_path = root / f"{label.replace(' ', '-')}.json"
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            if validate(test_project, candidate_path, evidence, _self_test=True)[0]:
                print(f"wespeaker preflight gate: {label} accepted", file=sys.stderr)
                return 1
        duplicate_rows = json.loads(approved.read_text(encoding="utf-8"))
        duplicate_rows["dependency_reviews"][1] = duplicate_rows["dependency_reviews"][0].copy()
        duplicate_rows["dependency_reviews_sha256"] = canonical(duplicate_rows["dependency_reviews"])
        duplicate_path = root / "duplicate-dependency-review.json"
        duplicate_path.write_text(json.dumps(duplicate_rows), encoding="utf-8")
        if validate(test_project, duplicate_path, evidence, _self_test=True)[0]:
            print("wespeaker preflight gate: duplicate dependency review accepted", file=sys.stderr)
            return 1
        for label, mutate in (
            ("dependency review source non-dict", lambda row: row.update({"source": "not-a-source"})),
            ("dependency review source extra", lambda row: row.update({"source": {"registry": "https://pypi.org/simple", "unexpected": True}})),
            ("dependency review source unknown", lambda row: row.update({"source": {"registry": "https://pypi.example.invalid/simple"}})),
            ("dependency review name type", lambda row: row.update({"name": None})),
            ("dependency review version type", lambda row: row.update({"version": 7})),
            ("dependency review status type", lambda row: row.update({"status": False})),
            ("dependency review license type", lambda row: row.update({"license": False})),
            ("dependency review bundled type", lambda row: row.update({"bundled_review": 1})),
            ("dependency review payload type", lambda row: row.update({"payload_sha256": True})),
        ):
            candidate = json.loads(approved.read_text(encoding="utf-8"))
            mutate(candidate["dependency_reviews"][0])
            candidate["dependency_reviews_sha256"] = canonical(candidate["dependency_reviews"])
            candidate_path = root / f"{label.replace(' ', '-')}.json"
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            if validate(test_project, candidate_path, evidence, _self_test=True)[0]:
                print(f"wespeaker preflight gate: {label} accepted", file=sys.stderr)
                return 1
        for field, value in (("scope_sha256", "0" * 64), ("digest", "0" * 64), ("signer", "tampered"), ("decision", "REJECTED"), ("evidence_sha256", "0" * 64), ("manifest_sha256", "0" * 64)):
            tampered_evidence = json.loads(evidence.read_text(encoding="utf-8"))
            tampered_evidence[field] = value
            evidence.write_text(json.dumps(tampered_evidence), encoding="utf-8")
            if validate(test_project, approved, evidence, _self_test=True)[0]:
                print(f"wespeaker preflight gate: evidence {field} tamper accepted", file=sys.stderr)
                return 1
            evidence.write_text(json.dumps(evidence_data), encoding="utf-8")
        for field in ("lock_sha256", "pyproject_sha256", "approval_scope_sha256"):
            candidate = json.loads(approved.read_text(encoding="utf-8"))
            candidate[field] = "0" * 64
            approved.write_text(json.dumps(candidate), encoding="utf-8")
            if validate(test_project, approved, evidence, _self_test=True)[0]:
                print(f"wespeaker preflight gate: {field} tamper accepted", file=sys.stderr)
                return 1
            approved.write_text(json.dumps(base), encoding="utf-8")
        candidate = json.loads(approved.read_text(encoding="utf-8"))
        candidate["dependency_reviews"][0]["native_review"] = False
        candidate["dependency_reviews_sha256"] = canonical(candidate["dependency_reviews"])
        approved.write_text(json.dumps(candidate), encoding="utf-8")
        if validate(test_project, approved, evidence, _self_test=True)[0]:
            print("wespeaker preflight gate: boolean native review accepted", file=sys.stderr)
            return 1
        approved.write_text(json.dumps(base), encoding="utf-8")
        candidate = json.loads(approved.read_text(encoding="utf-8"))
        candidate["component_reviews"][0]["signer"] = False
        approved.write_text(json.dumps(candidate), encoding="utf-8")
        if validate(test_project, approved, evidence, _self_test=True)[0]:
            print("wespeaker preflight gate: boolean signer accepted", file=sys.stderr)
            return 1
        approved.write_text(json.dumps(base), encoding="utf-8")
        for identity_path in (("source", "revision"), ("checkpoint", "revision"), ("corrected_replacement", "sha256"), ("legacy_mislicensed", "sha256"), ("fixture", "sha256")):
            candidate = json.loads(approved.read_text(encoding="utf-8"))
            candidate["fixed_identities"][identity_path[0]][identity_path[1]] = "0" * (40 if identity_path[1] == "revision" else 64)
            approved.write_text(json.dumps(candidate), encoding="utf-8")
            if validate(test_project, approved, evidence, _self_test=True)[0]:
                print(f"wespeaker preflight gate: {identity_path[0]} identity tamper accepted", file=sys.stderr)
                return 1
            approved.write_text(json.dumps(base), encoding="utf-8")
    print("wespeaker preflight gate: self-test PASS")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--approval-evidence", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        raise SystemExit(self_test())
    if args.project is None or args.manifest is None or args.approval_evidence is None:
        parser.error("--project, --manifest and --approval-evidence are required")
    passed, reason = validate(args.project, args.manifest, args.approval_evidence)
    if not passed:
        print(f"wespeaker preflight gate: BLOCKED: {reason}", file=sys.stderr); raise SystemExit(2)
    print("wespeaker preflight gate: PASS")
