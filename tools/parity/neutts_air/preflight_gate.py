#!/usr/bin/env python3
"""Offline, fail-closed gate for the NeuTTS Air reference closure.

Only the Python standard library is used here.  The VAST worker invokes this
gate before checking the host, reading a token, creating scratch space, or
contacting uv/Hugging Face.  Production remains blocked until an owner records
the exact dependency, artifact, source, and native/bundled-code reviews.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from urllib.parse import urlsplit
from pathlib import Path
from typing import Any

import tomllib

GATE_VERSION = 1
LOCK_SHA256 = "e72e864edbf75e85dd08fd377c6bee60f516aade2167731e5a0890631d2e0e35"
PYPROJECT_SHA256 = "fbb6be95757eef47bbb20e3fd0aa251ed7b373d66d098cce71cf386230183901"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
REVIEW_PLACEHOLDERS = {
    "", "unresolved", "pending", "pending_review", "owner_review_required",
    "review_required", "todo", "null", "none",
}
MANIFEST_KEYS = {"approval_scope_sha256", "companion_identity", "gate_version", "lock_sha256", "model_reviews", "operator_approval", "package_rows_sha256", "public_identity", "publication", "pyproject_sha256", "reference_route", "review_rows", "review_rows_sha256", "source_identity", "upstream_identity"}
APPROVAL_KEYS = {"schema", "decision", "signer", "digest"}
EVIDENCE_KEYS = {"schema", "decision", "scope_sha256", "manifest_sha256", "lock_sha256", "pyproject_sha256", "signer", "digest"}

PUBLIC_IDENTITY = {
    "repo": "vokra/neutts-air",
    "revision": "df2b47ec81862f0e3a19eb2638a6a2bcd2f13b8c",
    "file": "neutts-air.gguf",
    "bytes": 1495883328,
    "sha256": "f6caf559e919b16d77ac28177e59ee5427a5de92bdeedd719ecab00b4afbb754",
    "license": "Apache-2.0",
}
COMPANION_IDENTITY = {
    "repo": "vokra/distill-neucodec",
    "revision": "1471e4d9b82bfb98ae201f02e746fca346c3eb56",
    "file": "model.gguf",
    "bytes": 1025417504,
    "sha256": "15e60e7e5f7242255b18e1386b26c2a8f872c77a56ca241ee82c8aa5d8b6327f",
    "license": "Apache-2.0",
}
UPSTREAM_IDENTITY = {
    "repo": "neuphonic/neutts-air",
    "revision": "3b58b776406b62fdc137e31ea53d728f5c22a4ed",
    "files": [
        {"name": name, "bytes": None, "sha256": None}
        for name in (
            "config.json", "generation_config.json", "model.safetensors",
            "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
            "vocab.json",
        )
    ],
    "license": None,
}
SOURCE_IDENTITY = {
    "repo": "https://github.com/neuphonic/neutts.git",
    "revision": "3e9415df12633f8a74ac6f92418c7cd5c8c4bf0e",
    "path": "neuttsair/neutts.py",
    "bytes": 9035,
    "sha256": "e68b87dae6718903337a08eff56afbd58ba261d829624ea5a00a343c8cefb7c1",
    "license": None,
}
MODEL_REVIEW_IDS = [
    "public-gguf:vokra/neutts-air@df2b47ec81862f0e3a19eb2638a6a2bcd2f13b8c",
    "companion-gguf:vokra/distill-neucodec@1471e4d9b82bfb98ae201f02e746fca346c3eb56",
    "gated-upstream:neuphonic/neutts-air@3b58b776406b62fdc137e31ea53d728f5c22a4ed",
    "official-source:https://github.com/neuphonic/neutts.git@3e9415df12633f8a74ac6f92418c7cd5c8c4bf0e#neuttsair/neutts.py",
]
REFERENCE_ROUTE = {
    "entrypoint": "Qwen2ForCausalLM",
    "transformers": "5.5.0",
    "torch": "2.13.0+cpu",
    "device": "cpu",
    "dtype": "float32",
    "vocab_size": 217652,
    "fp32_atol": 0.01,
    "greedy_ids": "exact",
    "upload": "NO_UPLOAD",
}
LOCK_KEYS = {"version", "revision", "requires-python", "resolution-markers", "manifest", "package"}
PACKAGE_KEYS = {"name", "version", "source", "resolution-markers", "dependencies", "sdist", "wheels", "metadata"}
ARTIFACT_KEYS = {"url", "hash", "size", "upload-time"}
PYTORCH_CPU_REGISTRY = "https://download.pytorch.org/whl/cpu"
# uv's tracked lock shape omits ``size`` for these exact custom-index wheels.
# This is an authenticated, finite exception: a filename or package name alone
# must never make an otherwise incomplete artifact acceptable.
PYTORCH_CPU_ARTIFACTS_WITHOUT_SIZE = {
    (
        "https://download-r2.pytorch.org/whl/cpu/torch-2.13.0-cp312-cp312-macosx_14_0_arm64.whl",
        "sha256:2fe228aba290d14b9f31b049be550dbd469c3fd3013d7a19705b30454da97027",
    ),
    (
        "https://download-r2.pytorch.org/whl/cpu/torch-2.13.0%2Bcpu-cp312-cp312-linux_s390x.whl",
        "sha256:ffadde149901c8afa138daa38d898264003cfcf1a3336ca5cd964b5af227d867",
    ),
    (
        "https://download-r2.pytorch.org/whl/cpu/torch-2.13.0%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl",
        "sha256:6f307c2c32d764ffc6ff6893b801fad6d4752f3e67966cb8abf1843427c02604",
    ),
    (
        "https://download-r2.pytorch.org/whl/cpu/torch-2.13.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl",
        "sha256:4ca4a9394b0c771238a4f73590fdbbc4debad85ed0fa63d026ae1b085da7d6e2",
    ),
    (
        "https://download-r2.pytorch.org/whl/cpu/torch-2.13.0%2Bcpu-cp312-cp312-win_amd64.whl",
        "sha256:a8b450c1e58e5800e5b4691dac412f8d2d65a1dc3298166f91596603a3531e6f",
    ),
    (
        "https://download-r2.pytorch.org/whl/cpu/torch-2.13.0%2Bcpu-cp312-cp312-win_arm64.whl",
        "sha256:fa0762705b933624d59f6823db9ce7ec2e35b3e1e9c319c9db51fbeecfc3e319",
    ),
}
REGISTRY_PACKAGE_SCHEMAS = {
    frozenset({"name", "version", "source", "sdist", "wheels"}),
    frozenset({"name", "version", "source", "dependencies", "sdist", "wheels"}),
    frozenset({"name", "version", "source", "resolution-markers", "dependencies", "wheels"}),
}
DEPENDENCY_SCHEMAS = {
    frozenset({"name"}), frozenset({"name", "marker"}),
    frozenset({"name", "marker", "source", "version"}),
}
METADATA_REQUIREMENT_SCHEMAS = {
    frozenset({"name", "specifier"}), frozenset({"name", "specifier", "index"}),
}


def validate_project_schema(project: dict[str, Any]) -> None:
    if set(project) != {"project", "tool"} or not isinstance(project.get("project"), dict) or set(project["project"]) != {"name", "version", "description", "requires-python", "dependencies"}:
        raise ValueError("pyproject.toml structural schema drifted")
    p = project["project"]
    if p["requires-python"] != ">=3.12,<3.13" or not isinstance(p["dependencies"], list) or any(not isinstance(x, str) or not x.strip() for x in p["dependencies"]):
        raise ValueError("pyproject.toml project contract drifted")
    tool = project["tool"]
    if set(tool) != {"uv"} or not isinstance(tool["uv"], dict) or set(tool["uv"]) != {"package", "constraint-dependencies", "index", "sources"}:
        raise ValueError("pyproject.toml uv schema drifted")
    uv = tool["uv"]
    if uv["package"] is not False or uv["constraint-dependencies"] != ["setuptools>=83.0.0"] or uv["index"] != [{"name": "pytorch-cpu", "url": "https://download.pytorch.org/whl/cpu", "explicit": True}] or uv["sources"] != {"torch": {"index": "pytorch-cpu"}}:
        raise ValueError("pyproject.toml uv contract drifted")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_digest(value: Any) -> str:
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def load_json(path: Path) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in items:
            if key in out:
                raise ValueError(f"duplicate JSON key: {key}")
            out[key] = value
        return out
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)


def validate_artifact(value: Any, label: str, registry: str, *, package_name: str | None = None, artifact_kind: str | None = None) -> None:
    if not isinstance(value, dict) or set(value) not in (ARTIFACT_KEYS, ARTIFACT_KEYS - {"size"}):
        raise ValueError(f"{label} artifact schema is not exact")
    if "size" not in value and (
        registry != PYTORCH_CPU_REGISTRY
        or package_name != "torch"
        or artifact_kind != "wheels"
        or (value.get("url"), value.get("hash")) not in PYTORCH_CPU_ARTIFACTS_WITHOUT_SIZE
    ):
        raise ValueError(f"{label} artifact size is missing outside the exact PyTorch CPU exception")
    url = value["url"]
    expected_host = "download-r2.pytorch.org" if registry == PYTORCH_CPU_REGISTRY else "files.pythonhosted.org"
    parsed = urlsplit(url) if isinstance(url, str) else None
    if parsed is None or parsed.scheme != "https" or parsed.netloc != expected_host or not parsed.path:
        raise ValueError(f"{label} artifact URL is not the authenticated {expected_host} host")
    if not isinstance(value["hash"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value["hash"]):
        raise ValueError(f"{label} artifact hash is malformed")
    if "size" in value and (isinstance(value["size"], bool) or not isinstance(value["size"], int) or value["size"] <= 0):
        raise ValueError(f"{label} artifact size is not positive")
    if not isinstance(value["upload-time"], str) or not value["upload-time"].strip():
        raise ValueError(f"{label} artifact upload-time is missing")


def validate_metadata(value: Any, label: str) -> None:
    if not isinstance(value, dict) or set(value) != {"requires-dist"} or not isinstance(value["requires-dist"], list):
        raise ValueError(f"{label} metadata schema is malformed")
    for requirement in value["requires-dist"]:
        if not isinstance(requirement, dict) or frozenset(requirement) not in METADATA_REQUIREMENT_SCHEMAS or not isinstance(requirement.get("name"), str) or not requirement["name"].strip() or not isinstance(requirement.get("specifier"), str) or not requirement["specifier"].strip():
            raise ValueError(f"{label} metadata requirement is malformed")
        if "index" in requirement and requirement["index"] not in ("https://pypi.org/simple", "https://download.pytorch.org/whl/cpu"):
            raise ValueError(f"{label} metadata index is not approved")


def canonical_package_rows(lock: dict[str, Any]) -> list[dict[str, Any]]:
    if set(lock) != LOCK_KEYS or lock.get("version") != 1 or lock.get("revision") != 3 or lock.get("requires-python") != "==3.12.*" or not isinstance(lock.get("resolution-markers"), list) or any(not isinstance(marker, str) or not marker for marker in lock["resolution-markers"]) or not isinstance(lock.get("manifest"), dict) or set(lock["manifest"]) != {"constraints"} or not isinstance(lock["manifest"]["constraints"], list):
        raise ValueError("uv.lock top-level schema is malformed")
    if lock["manifest"]["constraints"] != [{"name": "setuptools", "specifier": ">=83.0.0"}]:
        raise ValueError("uv.lock constraint manifest is not the exact reviewed variant")
    packages = lock.get("package")
    if not isinstance(packages, list) or not packages:
        raise ValueError("uv.lock package table is missing or empty")
    rows: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for package in packages:
        if not isinstance(package, dict):
            raise ValueError("uv.lock contains a malformed package row")
        if set(package) - PACKAGE_KEYS:
            raise ValueError("uv.lock package row has unknown fields")
        dependencies = package.get("dependencies", [])
        if not isinstance(dependencies, list) or not isinstance(package.get("resolution-markers", []), list):
            raise ValueError("uv.lock contains malformed dependencies")
        name, version = package.get("name"), package.get("version")
        if not isinstance(name, str) or not name.strip() or not isinstance(version, str) or not version.strip() or (name, version) in identities:
            raise ValueError("uv.lock package identity is missing or duplicated")
        identities.add((name, version))
        source = package.get("source")
        if not isinstance(source, dict) or len(source) != 1 or set(source) not in ({"registry"}, {"virtual"}):
            raise ValueError("uv.lock package source is malformed")
        if "registry" in source and source["registry"] not in ("https://pypi.org/simple", "https://download.pytorch.org/whl/cpu"):
            raise ValueError("uv.lock package registry is not an approved index")
        virtual = "virtual" in source
        if virtual and source["virtual"] != ".":
            raise ValueError("uv.lock virtual source is not '.'")
        if not virtual and frozenset(package) not in REGISTRY_PACKAGE_SCHEMAS:
            raise ValueError("uv.lock registry package schema is not an exact committed variant")
        for dependency in dependencies:
            if not isinstance(dependency, dict) or frozenset(dependency) not in DEPENDENCY_SCHEMAS or not isinstance(dependency.get("name"), str) or not dependency["name"].strip():
                raise ValueError("uv.lock dependency row is malformed")
            for field in ("marker", "version"):
                if field in dependency and (not isinstance(dependency[field], str) or not dependency[field].strip()):
                    raise ValueError("uv.lock dependency field is malformed")
            if "source" in dependency:
                dependency_source = dependency["source"]
                if not isinstance(dependency_source, dict) or len(dependency_source) != 1 or set(dependency_source) not in ({"registry"}, {"virtual"}):
                    raise ValueError("uv.lock dependency source is malformed")
                if "registry" in dependency_source and dependency_source["registry"] not in ("https://pypi.org/simple", "https://download.pytorch.org/whl/cpu"):
                    raise ValueError("uv.lock dependency registry is not an approved index")
                if "virtual" in dependency_source and dependency_source["virtual"] != ".":
                    raise ValueError("uv.lock dependency virtual source is not '.'")
        metadata = package.get("metadata")
        if virtual and set(package) != {"name", "version", "source", "dependencies", "metadata"}:
            raise ValueError("uv.lock virtual project schema is not exact")
        if "metadata" in package or virtual:
            validate_metadata(metadata, f"{(name, version)!r}")
        for artifact_name in ("sdist", "wheels"):
            artifacts = package.get(artifact_name, [] if artifact_name == "wheels" else None)
            if artifact_name == "sdist" and "sdist" in package and not isinstance(artifacts, dict):
                raise ValueError("uv.lock sdist artifact is malformed")
            if artifact_name == "wheels" and "wheels" in package and not isinstance(artifacts, list):
                raise ValueError("uv.lock wheels artifact table is malformed")
            for artifact in ([] if artifacts is None else (artifacts if isinstance(artifacts, list) else [artifacts])):
                validate_artifact(
                    artifact,
                    f"{(name, version)!r} {artifact_name}",
                    source.get("registry", ""),
                    package_name=name,
                    artifact_kind=artifact_name,
                )
        if virtual and ("sdist" in package or "wheels" in package):
            raise ValueError("uv.lock virtual project must not contain artifacts")
        if not virtual and "sdist" not in package and not package.get("wheels"):
            raise ValueError("uv.lock registry package has no authenticated artifacts")
        rows.append({
            "name": name,
            "version": version,
            "source": source,
            "resolution-markers": package.get("resolution-markers", []),
            "dependencies": dependencies,
            "sdist": package.get("sdist"),
            "wheels": package.get("wheels", []),
            "metadata": metadata,
        })
    if any(not isinstance(row["name"], str) or not isinstance(row["version"], str) for row in rows):
        raise ValueError("uv.lock has a package row without an exact name/version")
    return sorted(rows, key=lambda row: (row["name"], row["version"]))


def reviewed_value(value: Any) -> bool:
    normalized = re.sub(r"\s+", "_", value.strip()).casefold() if isinstance(value, str) else ""
    return normalized not in REVIEW_PLACEHOLDERS


def resolved_upstream_identity(identity: Any) -> bool:
    if not isinstance(identity, dict) or not reviewed_value(identity.get("license")):
        return False
    files = identity.get("files")
    if not isinstance(files, list) or len(files) != 7:
        return False
    names = [item.get("name") for item in files if isinstance(item, dict)]
    expected = [item["name"] for item in UPSTREAM_IDENTITY["files"]]
    if names != expected or len(set(names)) != len(expected):
        return False
    return all(
        isinstance(item, dict)
        and isinstance(item.get("bytes"), int)
        and item["bytes"] > 0
        and isinstance(item.get("sha256"), str)
        and HEX64.fullmatch(item["sha256"]) is not None
        for item in files
    )


def resolved_source_identity(identity: Any) -> bool:
    return (
        isinstance(identity, dict)
        and reviewed_value(identity.get("license"))
        and isinstance(identity.get("bytes"), int)
        and identity["bytes"] > 0
        and isinstance(identity.get("sha256"), str)
        and HEX64.fullmatch(identity["sha256"]) is not None
    )


def blocked(reason: str) -> tuple[bool, str]:
    return False, reason


def validate(project: Path, manifest_path: Path, evidence_path: Path | None = None) -> tuple[bool, str]:
    lock_path = project / "uv.lock"
    pyproject_path = project / "pyproject.toml"
    if any(path.is_symlink() or not path.is_file() for path in (lock_path, pyproject_path, manifest_path)):
        return blocked("lock, project, or gate manifest is missing")
    try:
        manifest = load_json(manifest_path)
        lock_bytes = lock_path.read_bytes()
        pyproject_bytes = pyproject_path.read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return blocked(f"gate inputs are unreadable: {exc}")
    try:
        validate_project_schema(tomllib.loads(pyproject_bytes.decode("utf-8")))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError) as exc:
        return blocked(f"pyproject.toml schema is invalid: {exc}")
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS or manifest.get("gate_version") != GATE_VERSION:
        return blocked("unsupported gate manifest version")
    if digest_bytes(lock_bytes) != LOCK_SHA256 or manifest.get("lock_sha256") != LOCK_SHA256:
        return blocked("uv.lock bytes are not the reviewed exact lock")
    if digest_bytes(pyproject_bytes) != PYPROJECT_SHA256 or manifest.get("pyproject_sha256") != PYPROJECT_SHA256:
        return blocked("pyproject.toml bytes are not the reviewed exact project")
    try:
        rows = canonical_package_rows(tomllib.loads(lock_bytes.decode("utf-8")))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError) as exc:
        return blocked(f"uv.lock canonicalization failed: {exc}")
    if canonical_digest(rows) != manifest.get("package_rows_sha256"):
        return blocked("canonical version/source/marker/dependency rows drifted")
    identities = [f'{row["name"]}@{row["version"]}' for row in rows]
    review_rows = manifest.get("review_rows")
    if not isinstance(review_rows, list):
        return blocked("dependency review rows are missing")
    review_ids = [row.get("id") for row in review_rows if isinstance(row, dict)]
    if review_ids != sorted(identities) or len(review_ids) != len(identities) or len(set(review_ids)) != len(identities):
        return blocked("dependency review rows do not cover the exact lock identities")
    if any(
        not isinstance(row, dict)
        or set(row) != {"id", "status", "license", "native_review", "bundled_review", "evidence"}
        for row in review_rows
    ):
        return blocked("dependency review row schema drifted")
    if canonical_digest(review_rows) != manifest.get("review_rows_sha256"):
        return blocked("dependency review row digest drifted")
    if manifest.get("public_identity") != PUBLIC_IDENTITY or manifest.get("companion_identity") != COMPANION_IDENTITY:
        return blocked("public GGUF identity drifted")
    if manifest.get("upstream_identity") != UPSTREAM_IDENTITY:
        return blocked("gated upstream seven-file identity contract drifted")
    if manifest.get("source_identity") != SOURCE_IDENTITY:
        return blocked("official source identity drifted")
    if not resolved_upstream_identity(UPSTREAM_IDENTITY):
        return blocked("gated upstream payload/license identity is unresolved")
    if not resolved_source_identity(SOURCE_IDENTITY):
        return blocked("official source license identity is unresolved")
    if manifest.get("reference_route") != REFERENCE_ROUTE or manifest.get("publication") != "NO_UPLOAD":
        return blocked("reference route or publication policy drifted")
    model_reviews = manifest.get("model_reviews")
    if not isinstance(model_reviews, list):
        return blocked("model/source/component review rows are missing")
    model_ids = [row.get("id") for row in model_reviews if isinstance(row, dict)]
    if model_ids != MODEL_REVIEW_IDS or len(set(model_ids)) != len(MODEL_REVIEW_IDS):
        return blocked("model/source/component reviews are not the exact identity set")
    if any(
        not isinstance(row, dict)
        or set(row) != {"id", "status", "license", "native_review", "bundled_review", "evidence"}
        for row in model_reviews
    ):
        return blocked("model/source/component review row schema drifted")
    for row in model_reviews:
        if (
            not isinstance(row, dict) or row.get("status") != "REVIEWED"
            or not reviewed_value(row.get("license"))
            or not reviewed_value(row.get("native_review"))
            or not reviewed_value(row.get("bundled_review"))
            or not reviewed_value(row.get("evidence"))
        ):
            return blocked(f"model/source/component review is unresolved: {row.get('id')}")
    for row in review_rows:
        if (
            row.get("status") != "REVIEWED"
            or not reviewed_value(row.get("license"))
            or not reviewed_value(row.get("native_review"))
            or not reviewed_value(row.get("bundled_review"))
            or not reviewed_value(row.get("evidence"))
        ):
            return blocked(f"dependency review is unresolved: {row.get('id')}")
    scope = {
        "lock_sha256": LOCK_SHA256,
        "pyproject_sha256": PYPROJECT_SHA256,
        "package_rows_sha256": manifest["package_rows_sha256"],
        "review_rows": review_rows,
        "public_identity": PUBLIC_IDENTITY,
        "companion_identity": COMPANION_IDENTITY,
        "upstream_identity": UPSTREAM_IDENTITY,
        "source_identity": SOURCE_IDENTITY,
        "reference_route": REFERENCE_ROUTE,
        "model_reviews": model_reviews,
        "publication": "NO_UPLOAD",
    }
    scope_sha256 = canonical_digest(scope)
    if manifest.get("approval_scope_sha256") != scope_sha256:
        return blocked("operator approval scope is not bound to exact inputs")
    approval = manifest.get("operator_approval")
    if (
        not isinstance(approval, dict) or set(approval) != APPROVAL_KEYS or approval.get("schema") != "v1"
        or approval.get("decision") != "APPROVED"
        or not isinstance(approval.get("signer"), str) or not approval["signer"]
        or not isinstance(approval.get("digest"), str) or not HEX64.fullmatch(approval["digest"])
        or approval["digest"] != scope_sha256
    ):
        return blocked("exact operator approval is pending or invalid")
    evidence_path = evidence_path or manifest_path.with_name("license_gate_evidence.json")
    if evidence_path.is_symlink() or not evidence_path.is_file():
        return blocked("authenticated operator approval evidence is missing")
    try:
        evidence = load_json(evidence_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return blocked(f"operator approval evidence is unreadable: {exc}")
    if (
        not isinstance(evidence, dict) or set(evidence) != EVIDENCE_KEYS or evidence.get("schema") != "v1"
        or evidence.get("decision") != "APPROVED"
        or evidence.get("scope_sha256") != scope_sha256
        or evidence.get("manifest_sha256") != digest_bytes(manifest_path.read_bytes())
        or evidence.get("lock_sha256") != LOCK_SHA256
        or evidence.get("pyproject_sha256") != PYPROJECT_SHA256
        or evidence.get("signer") != approval["signer"]
        or evidence.get("digest") != approval["digest"]
    ):
        return blocked("authenticated operator approval evidence is not bound to this scope")
    return True, "PASS"


def self_test() -> int:
    project = Path(__file__).resolve().parent
    manifest_path = project / "license_gate_manifest.json"
    lock = tomllib.loads((project / "uv.lock").read_text(encoding="utf-8"))
    rows = canonical_package_rows(lock)
    torch_rows = [
        row for row in rows
        if row["name"] == "torch" and row["source"] == {"registry": PYTORCH_CPU_REGISTRY}
    ]
    if len(torch_rows) != 2 or any(
        not wheels or any(set(wheel) != ARTIFACT_KEYS - {"size"} for wheel in wheels)
        for wheels in (row["wheels"] for row in torch_rows)
    ):
        print("neutts-air custom-index torch wheel schema is not the tracked size-less variant", file=sys.stderr)
        return 1
    custom_url, custom_hash = next(iter(PYTORCH_CPU_ARTIFACTS_WITHOUT_SIZE))
    custom = {
        "url": custom_url,
        "hash": custom_hash,
        "upload-time": "2026-01-01T00:00:00Z",
    }
    try:
        validate_artifact(custom, "self-test custom-index torch wheel", PYTORCH_CPU_REGISTRY, package_name="torch", artifact_kind="wheels")
    except ValueError as exc:
        print(f"neutts-air tracked custom-index wheel was rejected: {exc}", file=sys.stderr)
        return 1
    malformed_custom = {
        "wrong-pair": lambda value: value.update(hash="sha256:" + "0" * 64),
        "wrong-package": lambda value: None,
        "wrong-kind": lambda value: None,
        "wrong-registry": lambda value: None,
        "extra-key": lambda value: value.update(extra="reject"),
        "bool-size": lambda value: value.update(size=True),
        "missing-upload-time": lambda value: value.pop("upload-time"),
    }
    for label, mutate in malformed_custom.items():
        candidate = dict(custom)
        package_name, artifact_kind, registry = "torch", "wheels", PYTORCH_CPU_REGISTRY
        if label == "wrong-package":
            package_name = "not-torch"
        elif label == "wrong-kind":
            artifact_kind = "sdist"
        elif label == "wrong-registry":
            registry = "https://pypi.org/simple"
        mutate(candidate)
        try:
            validate_artifact(candidate, f"self-test custom-index {label}", registry, package_name=package_name, artifact_kind=artifact_kind)
        except ValueError:
            pass
        else:
            print(f"neutts-air malformed custom-index wheel was accepted: {label}", file=sys.stderr)
            return 1
    ok, reason = validate(project, manifest_path)
    if ok or ("unresolved" not in reason and "artifact" not in reason):
        print(f"neutts-air gate: expected pending production gate, got {reason}", file=sys.stderr)
        return 1
    if "artifact" in reason:
        valid = {"url": "https://files.pythonhosted.org/packages/demo.whl", "hash": "sha256:" + "0" * 64, "size": 1, "upload-time": "2024-01-01T00:00:00Z"}
        cases = {"missing-size": lambda value: value.pop("size"), "missing-upload-time": lambda value: value.pop("upload-time"), "extra-key": lambda value: value.update(extra="x"), "bool-size": lambda value: value.update(size=True), "wrong-host": lambda value: value.update(url="https://example.invalid/demo.whl")}
        for label, mutate in cases.items():
            candidate = dict(valid); mutate(candidate)
            try:
                validate_artifact(candidate, f"self-test {label}", "https://pypi.org/simple")
            except ValueError:
                pass
            else:
                print(f"neutts-air artifact tamper accepted: {label}", file=sys.stderr); return 1
        try:
            canonical_package_rows({"version": 1, "revision": 3, "requires-python": "==3.12.*", "resolution-markers": [], "manifest": {"constraints": [{"name": "setuptools", "specifier": ">=83.0.0"}]}, "package": [
                {"name": "demo", "version": "0", "source": {"virtual": "."}, "dependencies": [], "metadata": {"requires-dist": []}},
                {"name": "registry-demo", "version": "1", "source": {"registry": "https://pypi.org/simple"}, "dependencies": []},
            ]})
        except ValueError:
            pass
        else:
            print("neutts-air registry package without artifacts accepted", file=sys.stderr); return 1
        print("neutts-air gate: self-test PASS (production artifact schema blocker)")
        return 0
    for value in ("", " null ", " PENDING_REVIEW ", "owner review required", "TODO"):
        if reviewed_value(value):
            print("neutts-air gate: placeholder normalization self-test failed", file=sys.stderr)
            return 1
    if not reviewed_value("official evidence: TODO was resolved"):  # longer citation is valid
        print("neutts-air gate: legitimate citation self-test failed", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory(prefix="neutts-air-gate-") as directory:
        root = Path(directory)
        test_project = root / "project"
        test_project.mkdir()
        shutil.copy2(project / "uv.lock", test_project / "uv.lock")
        shutil.copy2(project / "pyproject.toml", test_project / "pyproject.toml")
        duplicate_manifest = root / "duplicate-manifest.json"
        duplicate_manifest.write_text('{"gate_version":1,"gate_version":1}')
        try:
            load_json(duplicate_manifest)
        except ValueError as exc:
            if "duplicate JSON key" not in str(exc):
                print("neutts-air duplicate manifest key reported the wrong error", file=sys.stderr); return 1
        else:
            print("neutts-air duplicate manifest key was accepted", file=sys.stderr); return 1
        duplicate_evidence = root / "duplicate-evidence.json"
        duplicate_evidence.write_text('{"schema":"v1","schema":"v1"}')
        try:
            load_json(duplicate_evidence)
        except ValueError as exc:
            if "duplicate JSON key" not in str(exc):
                print("neutts-air duplicate evidence key reported the wrong error", file=sys.stderr); return 1
        else:
            print("neutts-air duplicate evidence key was accepted", file=sys.stderr); return 1
        approved = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in approved["review_rows"]:
            row.update({"status": "REVIEWED", "license": "SELF_TEST", "native_review": "SELF_TEST", "bundled_review": "SELF_TEST", "evidence": "self-test evidence"})
        for row in approved["model_reviews"]:
            row.update({"status": "REVIEWED", "license": "SELF_TEST", "native_review": "SELF_TEST", "bundled_review": "SELF_TEST", "evidence": "self-test evidence"})
        approved["review_rows_sha256"] = canonical_digest(approved["review_rows"])
        scope = {"lock_sha256": LOCK_SHA256, "pyproject_sha256": PYPROJECT_SHA256, "package_rows_sha256": approved["package_rows_sha256"], "review_rows": approved["review_rows"], "public_identity": PUBLIC_IDENTITY, "companion_identity": COMPANION_IDENTITY, "upstream_identity": UPSTREAM_IDENTITY, "source_identity": SOURCE_IDENTITY, "reference_route": REFERENCE_ROUTE, "model_reviews": approved["model_reviews"], "publication": "NO_UPLOAD"}
        approved["approval_scope_sha256"] = canonical_digest(scope)
        approved["operator_approval"] = {"schema": "v1", "decision": "APPROVED", "signer": "self-test", "digest": approved["approval_scope_sha256"]}
        approved_path = root / "manifest.json"
        approved_path.write_text(json.dumps(approved), encoding="utf-8")
        evidence_path = root / "license_gate_evidence.json"
        evidence_path.write_text(json.dumps({"schema": "v1", "decision": "APPROVED", "scope_sha256": approved["approval_scope_sha256"], "manifest_sha256": digest_bytes(approved_path.read_bytes()), "lock_sha256": LOCK_SHA256, "pyproject_sha256": PYPROJECT_SHA256, "signer": "self-test", "digest": approved["approval_scope_sha256"]}), encoding="utf-8")
        approved_ok, approved_reason = validate(test_project, approved_path, evidence_path)
        if approved_ok or "payload/license identity is unresolved" not in approved_reason:
            print("neutts-air gate: unresolved identity bypass self-test failed", file=sys.stderr)
            return 1

        def rejected(label: str, mutate: Any) -> bool:
            candidate = json.loads(approved_path.read_text(encoding="utf-8"))
            mutate(candidate)
            path = root / f"{label}.json"
            path.write_text(json.dumps(candidate), encoding="utf-8")
            return not validate(test_project, path, evidence_path)[0]

        checks = {
            "lock": lambda value: value.update(lock_sha256="0" * 64),
            "public": lambda value: value["public_identity"].update(sha256="0" * 64),
            "companion": lambda value: value["companion_identity"].update(sha256="0" * 64),
            "upstream": lambda value: value["upstream_identity"]["files"].pop(),
            "source": lambda value: value["source_identity"].update(sha256="0" * 64),
            "scope": lambda value: value.update(approval_scope_sha256="0" * 64),
            "signer": lambda value: value["operator_approval"].update(signer="other"),
        }
        for label, mutate in checks.items():
            if not rejected(label, mutate):
                print(f"neutts-air gate: {label} tamper was accepted", file=sys.stderr)
                return 1
        for label in ("missing", "extra", "duplicate"):
            def mutate(value: dict[str, Any], label: str = label) -> None:
                if label == "missing":
                    value["model_reviews"].pop()
                elif label == "extra":
                    value["model_reviews"].append(dict(value["model_reviews"][0]))
                else:
                    value["model_reviews"][1]["id"] = value["model_reviews"][0]["id"]
            if not rejected(f"model-{label}", mutate):
                print(f"neutts-air gate: model {label} identity tamper was accepted", file=sys.stderr)
                return 1
        for field in ("license", "native_review", "bundled_review", "evidence"):
            if not rejected(f"placeholder-{field}", lambda value, field=field: value["model_reviews"][0].update(**{field: "  pEnDiNg_ReViEw  "})):
                print(f"neutts-air gate: {field} placeholder was accepted", file=sys.stderr)
                return 1
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence["scope_sha256"] = "0" * 64
        evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        if validate(test_project, approved_path, evidence_path)[0]:
            print("neutts-air gate: evidence tamper was accepted", file=sys.stderr)
            return 1
    print("neutts-air gate: self-test PASS (pending production + approved baseline/tamper cases)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    manifest = args.manifest or args.project / "license_gate_manifest.json"
    ok, reason = validate(args.project, manifest, args.evidence)
    if not ok:
        print(f"neutts-air preflight gate: BLOCKED: {reason}", file=sys.stderr)
        return 2
    print("neutts-air preflight gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
