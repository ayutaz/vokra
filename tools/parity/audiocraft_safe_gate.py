#!/usr/bin/env python3
"""Dependency-free, fail-closed preflight for the AudioCraft sidecars."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

TARGET_MARKER = "platform_machine == 'x86_64' and sys_platform == 'linux'"
FORBIDDEN = frozenset({"soxr", "rubberband", "triton", "nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12"})
REQUIRED_LICENSE_KEYS = frozenset({"name", "version", "license", "primary_source"})
ARTIFACT_KEYS = frozenset({"url", "hash", "size", "upload-time"})
APPROVAL_KEYS = frozenset({"manifest_sha256", "scope_sha256", "signer", "digest", "decision", "evidence_sha256"})
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
LOCK_KEYS = frozenset({"version", "revision", "requires-python", "resolution-markers", "supported-markers", "package"})
PACKAGE_KEYS = frozenset({"name", "version", "source", "dependencies", "resolution-markers", "sdist", "wheels", "metadata"})
DEPENDENCY_KEYS = frozenset({"name", "source", "version", "marker"})
METADATA_KEYS = frozenset({"requires-dist"})

EXPECTED_IDENTITIES: dict[str, dict[str, Any]] = {
    "magnet-medium-30secs": {
        "model": {"repo": "facebook/magnet-medium-30secs", "revision": "2559c5978450f62782cf1d17826d384fb93fb64b", "license_spdx": "CC-BY-NC-4.0", "readme_bytes": 10695, "readme_sha256": "85f191c1d886dc8e907986a100f0415751cb86479d18268d2cfacd614b5fd6db", "compression_bytes": 236003715, "compression_sha256": "91598c7da3d183eb8e0cc19cbbdc4f64f2d0c53069f9c8aa84185d0e33873c67", "state_bytes": 3677670163, "state_sha256": "9bc89122b640f11394f51e6e77b4194d57da328ae81b2301ee316de916dbf4c5"},
        "source": {"repo": "facebookresearch/audiocraft", "revision": "905371a779f608169353fe6ad42bb5fc10c5c9a8", "license_spdx": "MIT", "license_path": "LICENSE", "license_bytes": 1088, "license_sha256": "da6d3703ed11cbe42bd212c725957c98da23cbff1998c05fa4b3d976d1a58e93", "roles": [
            {"path": "models/__init__.py", "bytes": 669, "sha256": "75c8f32bd306a2df4203eb0cc4bf7edeaf831ea83100c176a481d31e14c5a531"}, {"path": "models/magnet.py", "bytes": 4400, "sha256": "d0307171a21fd96cb898fefff1190df124d5ce245823a71e1a62b8b8ffe8cdcf"}, {"path": "models/loaders.py", "bytes": 6699, "sha256": "91a79fcb028b2f1fafdac53b1060598bf0e4a6927467131f20055f6b8a33e4c2"}, {"path": "models/lm_magnet.py", "bytes": 24724, "sha256": "0c9349d1238d0a0aa276c921c3629de72bbadc9f19fa81b9429fa2e82678e06b"},
        ]},
    },
    "magnet-small-10secs": {
        "model": {"repo": "facebook/magnet-small-10secs", "revision": "2c9084771bd2e83c5c7e36303e24550da30da8e0", "license_spdx": "CC-BY-NC-4.0", "readme_bytes": 10693, "readme_sha256": "e7fba2ce044a85fdcff253fa250e661044d9071d6f5033e5eab3f2ca42ce16e4", "compression_bytes": 236003715, "compression_sha256": "91598c7da3d183eb8e0cc19cbbdc4f64f2d0c53069f9c8aa84185d0e33873c67", "state_bytes": 840844851, "state_sha256": "0594e551ed9c40464b5918f5ddcce348e491e912e61d69f4d5d64d4ddd1a6ade"},
        "source": {"repo": "facebookresearch/audiocraft", "revision": "905371a779f608169353fe6ad42bb5fc10c5c9a8", "license_spdx": "MIT", "license_path": "LICENSE", "license_bytes": 1088, "license_sha256": "da6d3703ed11cbe42bd212c725957c98da23cbff1998c05fa4b3d976d1a58e93", "roles": [
            {"path": "models/__init__.py", "bytes": 669, "sha256": "75c8f32bd306a2df4203eb0cc4bf7edeaf831ea83100c176a481d31e14c5a531"}, {"path": "models/magnet.py", "bytes": 4400, "sha256": "d0307171a21fd96cb898fefff1190df124d5ce245823a71e1a62b8b8ffe8cdcf"}, {"path": "models/loaders.py", "bytes": 6699, "sha256": "91a79fcb028b2f1fafdac53b1060598bf0e4a6927467131f20055f6b8a33e4c2"}, {"path": "models/lm_magnet.py", "bytes": 24724, "sha256": "0c9349d1238d0a0aa276c921c3629de72bbadc9f19fa81b9429fa2e82678e06b"},
        ]},
    },
    "melodyflow-t24-30secs": {
        "model": {"repo": "facebook/melodyflow-t24-30secs", "revision": "77bcfce24371bf29a06152c72169162c6f2791de", "license_spdx": "CC-BY-NC-4.0", "readme_bytes": 6560, "readme_sha256": "ab790ac275d6035184dabfa467be8ec8aa08a762ee3610cf43a061db45a8f0a1", "compression_bytes": 238776630, "compression_sha256": "c075ee7c5b13d50937d1e4f197f3e940c3f3b74207857cb0e1e17891010fdc6d", "state_bytes": 3849817990, "state_sha256": "e9f95857aa1e0906fb44017ca2e4e8205395599693d6e80e5c3b8b7fc16498ef"},
        "source": {"repo": "facebook/MelodyFlow", "revision": "9d0d223e9a63bbb8c20b9f57c5afcb4de297e6da", "license_spdx": "MIT", "license_path": "LICENSE", "license_bytes": 1088, "license_sha256": "da6d3703ed11cbe42bd212c725957c98da23cbff1998c05fa4b3d976d1a58e93", "license_weights_path": "LICENSE_weights", "license_weights_bytes": 19333, "license_weights_sha256": "336255dc30193e8e15d689d9481bb05673d89055718f3a96923a7ffb99adbbaf", "roles": [
            {"path": "models/__init__.py", "bytes": 735, "sha256": "79a3de6fa1f606bb058150aa0a8959b0c0d2fef84d7fc5b4b72bcd80becfa866"}, {"path": "flow.py", "bytes": 21274, "sha256": "06d148fd8e40ada00a034c6d5189a35b5cb3b01f7ee140b6b6a94e02905a9892"}, {"path": "loaders.py", "bytes": 9509, "sha256": "d4cdc731e145fb2c5257f8691b2330034d99b604bfdcb4da49f9fc7cc718f31b"}, {"path": "melodyflow.py", "bytes": 13083, "sha256": "655c9d697b698b8c2dcdc560062dcbc32423b4468912bab3e5f4094b08071a43"}, {"path": "requirements.txt", "bytes": 375, "sha256": "6766a3bb39e304094e1ca651b3acbeccf53130221afbfe933b31f1b25e2aa35"},
        ]},
    },
}

# Replaced with the final policy byte hashes below.  A mutable policy cannot
# authenticate itself; this code-bound value is intentionally fail-closed.
EXPECTED_PROJECT_SHA256: dict[str, str] = {
    "magnet-medium-30secs": "c2d5d0a5202599b73ce8c69f351533690887683a6c42aa8aa99b50b0d0fee51d",
    "magnet-small-10secs": "f281f2d024b6853d43906d3b59ddb2556d34904cc69df30762a417758f2bfdf7",
    "melodyflow-t24-30secs": "d1d77d6aae82dd886c09bf27ee75598cc93cf577f761d1dab73fdcf29f5b407b",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)


def _package_rows(lock: dict[str, Any]) -> list[dict[str, Any]]:
    if set(lock) != LOCK_KEYS or lock.get("version") != 1 or type(lock.get("version")) is not int or lock.get("revision") != 3 or type(lock.get("revision")) is not int:
        raise ValueError("uv.lock top-level schema drifted")
    if lock.get("requires-python") != "==3.12.*" or lock.get("resolution-markers") != [TARGET_MARKER] or lock.get("supported-markers") != [TARGET_MARKER]:
        raise ValueError("uv.lock marker schema malformed")
    packages = lock.get("package")
    if not isinstance(packages, list) or not packages:
        raise ValueError("uv.lock package table is missing or empty")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for package in packages:
        if not isinstance(package, dict):
            raise ValueError("uv.lock package row is not a table")
        if set(package) - PACKAGE_KEYS:
            raise ValueError("uv.lock package row has unknown fields")
        name, version, source = package.get("name"), package.get("version"), package.get("source")
        if not isinstance(name, str) or not name.strip() or not isinstance(version, str) or not version.strip():
            raise ValueError("uv.lock package row lacks nonempty name/version")
        if not isinstance(source, dict) or len(source) != 1 or set(source) not in ({"registry"}, {"virtual"}) or not all(isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in source.items()):
            raise ValueError(f"uv.lock source schema is invalid for {name}")
        if set(source) == {"registry"}:
            # These are the only registry-row variants emitted by all three
            # committed locks.  Keeping the variants explicit prevents a
            # future resolver from silently adding an unreviewed field.
            if name == "torch" and version == "2.7.1+cpu":
                expected_keys = {"name", "version", "source", "dependencies", "wheels"}
            elif "dependencies" in package:
                expected_keys = {"name", "version", "source", "dependencies", "sdist", "wheels"}
            else:
                expected_keys = {"name", "version", "source", "sdist", "wheels"}
            if set(package) != expected_keys:
                raise ValueError(f"registry package schema is not exact for {name}")
        else:
            if set(package) != {"name", "version", "source", "dependencies", "metadata"}:
                raise ValueError("virtual package schema is not exact")
        if set(source) == {"virtual"} and source["virtual"] != ".":
            raise ValueError("virtual package source must be '.'")
        if set(source) == {"registry"} and source["registry"] not in {"https://pypi.org/simple", "https://download.pytorch.org/whl/cpu"}:
            raise ValueError(f"unapproved package registry for {name}")
        identity = (name.lower(), version)
        if identity in seen:
            raise ValueError(f"duplicate uv.lock package identity: {name} {version}")
        seen.add(identity)
        markers = []
        deps = package.get("dependencies", [])
        if not isinstance(deps, list):
            raise ValueError(f"dependencies malformed for {name}")
        dep_rows: list[dict[str, Any]] = []
        dep_seen: set[str] = set()
        for dep in deps:
            if not isinstance(dep, dict) or set(dep) != {"name", "marker"}:
                raise ValueError(f"dependency schema malformed for {name}")
            dep_name = dep.get("name")
            if not isinstance(dep_name, str) or not dep_name.strip() or dep_name.lower() in dep_seen:
                raise ValueError(f"dependency name malformed or duplicated for {name}")
            dep_seen.add(dep_name.lower())
            if dep.get("marker") != TARGET_MARKER:
                raise ValueError(f"dependency marker malformed for {name}")
            # Keep the canonical representation stable across all three
            # existing locks; source/version are absent in the TOML row and
            # therefore represented explicitly as null here.
            dep_rows.append({"name": dep_name, "source": None, "version": None, "marker": dep.get("marker")})
        metadata = package.get("metadata")
        if set(source) == {"registry"} and "metadata" in package:
            raise ValueError(f"registry metadata is not allowed for {name}")
        if metadata is not None and (not isinstance(metadata, dict) or set(metadata) != METADATA_KEYS or not isinstance(metadata.get("requires-dist"), list)):
            raise ValueError(f"package metadata malformed for {name}")
        if source == {"virtual": "."}:
            if "sdist" in package or "wheels" in package or metadata is None:
                raise ValueError("virtual package metadata/artifacts malformed")
            expected_requirements = [
                {"name": "huggingface-hub", "specifier": "==1.27.0"},
                {"name": "safetensors", "specifier": "==0.7.0"},
                {"name": "torch", "specifier": "==2.7.1", "index": "https://download.pytorch.org/whl/cpu"},
            ]
            if metadata["requires-dist"] != expected_requirements:
                    raise ValueError("virtual requires-dist metadata malformed")
        rows.append({"name": name, "version": version, "source": source, "resolution_markers": markers, "dependencies": sorted(dep_rows, key=_canonical), "metadata": metadata, "sdist": package.get("sdist"), "wheels": package.get("wheels")})
    virtual = [row for row in rows if row["source"] == {"virtual": "."}]
    if len(virtual) != 1:
        raise ValueError("uv.lock must contain exactly one virtual package")
    return sorted(rows, key=lambda row: (row["name"].lower(), row["version"], _canonical(row)))


def _lock_artifacts_complete(lock: dict[str, Any]) -> bool:
    try:
        rows = _package_rows(lock)
    except ValueError:
        return False
    for package in rows:
        if package["source"] == {"virtual": "."}:
            continue
        registry = package["source"]["registry"]
        sdist, wheels = package.get("sdist"), package.get("wheels")
        if package["name"] != "torch" and not isinstance(sdist, dict):
            return False
        if package["name"] == "torch" and sdist is not None:
            return False
        if not isinstance(wheels, list) or not wheels or not all(isinstance(item, dict) for item in wheels):
            return False
        expected_host = "download-r2.pytorch.org" if registry == "https://download.pytorch.org/whl/cpu" else "files.pythonhosted.org"
        for artifact in ([sdist] if sdist is not None else []) + wheels:
            if set(artifact) != ARTIFACT_KEYS:
                return False
            url, digest, size = artifact.get("url"), artifact.get("hash"), artifact.get("size")
            if not isinstance(url, str) or not url.startswith("https://") or urlparse(url).netloc != expected_host:
                return False
            if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
                return False
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                return False
            if not isinstance(artifact["upload-time"], str) or not artifact["upload-time"].strip():
                return False
    return True


def _approval_scope(policy: dict[str, Any], project_sha: str, lock_sha: str, rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> str:
    # The project bytes are separately code-bound below.  Excluding only the
    # scope field here avoids a self-referential hash while still covering all
    # replay-affecting policy, lock and identity rows.
    # ``project_sha`` is bound by the external evidence's manifest_sha256.  It
    # is intentionally not inside this digest because the scope is stored in
    # the project policy itself (which would otherwise be self-referential).
    return _sha256(_canonical({"lock_sha256": lock_sha, "package_rows": rows, "license_records": sorted(records, key=_canonical), "identities": {"model": policy.get("model_identity"), "source": policy.get("source_identity")}, "dependencies": policy.get("dependencies"), "forbidden_packages": policy.get("forbidden_packages"), "owner_clearance": policy.get("owner_clearance"), "publication": policy.get("publication"), "status": policy.get("status"), "runtime_status": policy.get("runtime_status"), "expected_operator_decision": "APPROVED", "expected_operator_status": "APPROVED", "expected_owner_clearance": "APPROVED_OWNER_SIGNOFF", "decision": "APPROVE_OWNER_CLEARANCE_FOR_VAST_ONLY"}))


def _read_approval(path: Path, manifest_sha: str, scope_sha: str, expected_signer: str) -> tuple[bool, str]:
    if not path.is_file() or path.is_symlink():
        return False, "external approval evidence must be a regular file"
    try:
        evidence = _load_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return False, f"external approval evidence is invalid: {exc}"
    if not isinstance(evidence, dict) or set(evidence) != APPROVAL_KEYS:
        return False, "external approval evidence schema is not exact"
    if evidence.get("manifest_sha256") != manifest_sha or evidence.get("scope_sha256") != scope_sha:
        return False, "external approval is not bound to manifest/scope"
    signer, digest, decision = evidence.get("signer"), evidence.get("digest"), evidence.get("decision")
    if not isinstance(signer, str) or not signer.strip() or signer != expected_signer or not isinstance(digest, str) or not HEX64.fullmatch(digest):
        return False, "external approval signer/digest is malformed"
    if decision != "APPROVE_OWNER_CLEARANCE_FOR_VAST_ONLY":
        return False, "external approval decision is not the fixed approval"
    approval_core = {"manifest_sha256": manifest_sha, "scope_sha256": scope_sha, "signer": signer, "decision": decision}
    if digest != _sha256(_canonical(approval_core)):
        return False, "external approval digest is not bound to its decision scope"
    unsigned = dict(evidence); unsigned.pop("evidence_sha256")
    if evidence.get("evidence_sha256") != _sha256(_canonical(unsigned)):
        return False, "external approval evidence digest is not canonical"
    return True, signer


def _validate_pyproject_schema(pyproject: dict[str, Any]) -> None:
    if set(pyproject) != {"project", "tool"} or not isinstance(pyproject.get("project"), dict) or not isinstance(pyproject.get("tool"), dict):
        raise ValueError("pyproject top-level schema is not exact")
    project = pyproject["project"]
    if set(project) != {"name", "version", "description", "requires-python", "dependencies"}:
        raise ValueError("pyproject project schema is not exact")
    if any(not isinstance(project.get(key), str) or not project[key].strip() for key in ("name", "version", "description", "requires-python")) or not isinstance(project["dependencies"], list) or not all(isinstance(item, str) and item.strip() for item in project["dependencies"]):
        raise ValueError("pyproject project fields are malformed")
    tool = pyproject["tool"]
    if set(tool) != {"uv", "vokra"} or not isinstance(tool.get("uv"), dict) or not isinstance(tool.get("vokra"), dict):
        raise ValueError("pyproject tool schema is not exact")
    uv = tool["uv"]
    if set(uv) != {"package", "environments", "sources", "index"} or uv.get("package") is not False or uv.get("environments") != ["python_full_version == '3.12.*' and platform_machine == 'x86_64' and sys_platform == 'linux'"]:
        raise ValueError("pyproject uv schema is not exact")
    if uv.get("sources") != {"torch": [{"index": "pytorch-cpu"}]} or uv.get("index") != [{"name": "pytorch-cpu", "url": "https://download.pytorch.org/whl/cpu", "explicit": True}]:
        raise ValueError("pyproject uv source/index policy drifted")
    vokra = tool["vokra"]
    if set(vokra) != {"audiocraft_reference"} or not isinstance(vokra["audiocraft_reference"], dict):
        raise ValueError("pyproject vokra schema is not exact")
    policy = vokra["audiocraft_reference"]
    expected_policy = {"operator_approval", "model", "source_license", "weight_license", "source_repository", "source_revision", "source_license_evidence", "weight_license_evidence", "status", "publication", "runtime_status", "lock_policy", "forbidden_packages", "license_blocker", "owner_clearance", "dependencies", "model_identity", "source_identity", "license_records", "lock_sha256", "package_rows_sha256", "approval_scope_sha256", "license_rows_sha256"}
    if set(policy) != expected_policy:
        raise ValueError("AudioCraft policy schema is not exact")
    if not isinstance(policy.get("operator_approval"), dict) or set(policy["operator_approval"]) != {"decision", "signer", "digest"}:
        raise ValueError("operator approval schema is not exact")
    for key in ("model_identity", "source_identity"):
        if not isinstance(policy.get(key), dict):
            raise ValueError(f"{key} schema is malformed")
    if set(policy["model_identity"]) != {"repo", "revision", "license_spdx", "readme_bytes", "readme_sha256", "compression_bytes", "compression_sha256", "state_bytes", "state_sha256"}:
        raise ValueError("model identity schema is not exact")
    source_keys = set(policy["source_identity"])
    if source_keys not in ({"repo", "revision", "license_spdx", "license_path", "license_bytes", "license_sha256", "roles"}, {"repo", "revision", "license_spdx", "license_path", "license_bytes", "license_sha256", "license_weights_path", "license_weights_bytes", "license_weights_sha256", "roles"}):
        raise ValueError("source identity schema is not exact")


def inspect_project(project: Path, approval_evidence: Path | None = None) -> tuple[int, str]:
    pyproject_path, lock_path = project / "pyproject.toml", project / "uv.lock"
    if not pyproject_path.is_file() or pyproject_path.is_symlink() or not lock_path.is_file() or lock_path.is_symlink():
        return 2, "dedicated pyproject.toml + uv.lock are required before acquisition"
    try:
        pyproject_bytes, lock_bytes = pyproject_path.read_bytes(), lock_path.read_bytes()
        pyproject = tomllib.loads(pyproject_bytes.decode("utf-8")); lock = tomllib.loads(lock_bytes.decode("utf-8")); _validate_pyproject_schema(pyproject); rows = _package_rows(lock)
    except (OSError, UnicodeDecodeError, ValueError, tomllib.TOMLDecodeError) as exc:
        return 2, f"lock metadata is invalid: {exc}"
    policy = pyproject.get("tool", {}).get("vokra", {}).get("audiocraft_reference")
    if not isinstance(policy, dict):
        return 2, "AudioCraft reference policy metadata is missing"
    project_name = pyproject.get("project", {}).get("name")
    model_key = next((key for key, identity in EXPECTED_IDENTITIES.items() if identity["model"]["repo"] == policy.get("model")), None)
    if model_key is None or project_name != f"vokra-{model_key}-reference":
        return 2, "model/project identity is not one of the fixed AudioCraft targets"
    if policy.get("publication") != "NO_UPLOAD" or policy.get("status") != "BLOCKED_LICENSE_REVIEW_VAST_ONLY":
        return 2, "publication or owner/license status is not fail-closed"
    operator_approval = policy.get("operator_approval")
    if not isinstance(operator_approval, dict) or set(operator_approval) != {"decision", "signer", "digest"} or not all(isinstance(operator_approval[key], str) for key in operator_approval):
        return 2, "operator approval schema is not exact"
    if operator_approval["decision"] == "PENDING_OWNER_SIGNOFF":
        if operator_approval["signer"] or operator_approval["digest"]:
            return 2, "pending operator approval must have empty signer/digest"
        return 2, "operator approval remains explicitly pending"
    if operator_approval["decision"] != "APPROVED" or not operator_approval["signer"].strip() or not HEX64.fullmatch(operator_approval["digest"]):
        return 2, "operator approval must be APPROVED with canonical signer/digest"
    if lock.get("requires-python") != "==3.12.*" or lock.get("resolution-markers") != [TARGET_MARKER] or lock.get("supported-markers") != [TARGET_MARKER]:
        return 2, "uv.lock does not cover exactly Linux x86_64 Python 3.12"
    expected_dependencies = ["huggingface-hub==1.27.0", "safetensors==0.7.0", "torch==2.7.1"]
    if pyproject.get("project", {}).get("dependencies") != expected_dependencies or policy.get("dependencies") != expected_dependencies:
        return 2, "dedicated project dependency closure drifted"
    virtual = [row for row in rows if row["source"] == {"virtual": "."}]
    if len(virtual) != 1 or (virtual[0]["name"], virtual[0]["version"]) != (project_name, pyproject.get("project", {}).get("version")):
        return 2, "uv.lock virtual package does not match pyproject"
    if virtual[0]["metadata"] != {"requires-dist": [{"name": "huggingface-hub", "specifier": "==1.27.0"}, {"name": "safetensors", "specifier": "==0.7.0"}, {"name": "torch", "specifier": "==2.7.1", "index": "https://download.pytorch.org/whl/cpu"}]}:
        return 2, "uv.lock virtual requires-dist metadata drifted"
    names = {row["name"].lower() for row in rows}
    if names & FORBIDDEN:
        return 2, f"forbidden package is present in uv.lock: {', '.join(sorted(names & FORBIDDEN))}"
    if any(name.startswith("nvidia-") or "cuda" in name for name in names):
        return 2, "CUDA/NVIDIA packages are forbidden in uv.lock"
    if "https://download.pytorch.org/whl/cpu" not in lock_bytes.decode("utf-8"):
        return 2, "torch is not bound to the official CPU index"
    if not _lock_artifacts_complete(lock):
        return 2, "uv.lock artifact URL/hash/size/source rows are incomplete or malformed"
    expected = EXPECTED_IDENTITIES[model_key]
    if policy.get("model_identity") != expected["model"] or policy.get("source_identity") != expected["source"]:
        return 2, "authenticated model/source identity table does not match fixed primary-source evidence"
    if policy.get("source_license") != expected["source"]["license_spdx"] or policy.get("weight_license") != expected["model"]["license_spdx"] or policy.get("source_repository") != expected["source"]["repo"] or policy.get("source_revision") != expected["source"]["revision"]:
        return 2, "legacy model/source identity fields are not bound to the fixed tables"
    expected_license_evidence = f"LICENSE:{expected['source']['license_bytes']}:{expected['source']['license_sha256']}"
    expected_weight_evidence = f"{expected['model']['repo']}@{expected['model']['revision']}"
    if policy.get("source_license_evidence") != expected_license_evidence or policy.get("weight_license_evidence") != expected_weight_evidence:
        return 2, "primary-source license evidence fields are not bound"
    records = policy.get("license_records")
    if not isinstance(records, list):
        return 2, "version-keyed license records are missing"
    nonvirtual = {(row["name"], row["version"]) for row in rows if row["source"] != {"virtual": "."}}
    record_ids = {(r.get("name"), r.get("version")) for r in records if isinstance(r, dict)}
    if record_ids != nonvirtual or len(record_ids) != len(records) or any(not isinstance(r, dict) or set(r) != REQUIRED_LICENSE_KEYS for r in records):
        return 2, "license records do not exactly cover the locked packages"
    project_sha, lock_sha = _sha256(pyproject_bytes), _sha256(lock_bytes)
    if EXPECTED_PROJECT_SHA256[model_key] != project_sha:
        return 2, "pyproject SHA-256 is not the code-bound project identity"
    if policy.get("lock_sha256") != lock_sha or policy.get("package_rows_sha256") != _sha256(_canonical(rows)):
        return 2, "uv.lock or canonical package-row digest is not bound"
    license_sha = _sha256(_canonical(sorted(records, key=_canonical)))
    if policy.get("license_rows_sha256") != license_sha:
        return 2, "canonical license-row digest is not bound"
    scope_sha = _approval_scope(policy, project_sha, lock_sha, rows, records)
    if policy.get("approval_scope_sha256") != scope_sha:
        return 2, "canonical approval scope is not bound"
    if operator_approval["digest"] != scope_sha:
        return 2, "operator approval digest is not bound to canonical scope"
    if approval_evidence is None:
        return 2, "BLOCKED: external owner approval evidence is required before acquisition"
    ok, detail = _read_approval(approval_evidence, project_sha, scope_sha, operator_approval["signer"])
    if not ok:
        return 2, f"BLOCKED: {detail}"
    if policy.get("owner_clearance") != "APPROVED_OWNER_SIGNOFF":
        return 2, "BLOCKED: CC-BY-NC-4.0 owner clearance is unresolved"
    return 0, f"APPROVED VAST-only AudioCraft closure signer={detail}"


def self_test() -> int:
    artifact = {"url": "https://download-r2.pytorch.org/whl/cpu/t.whl", "hash": "sha256:" + "a" * 64, "size": 1, "upload-time": "2025-01-01T00:00:00Z"}
    lock = {"version": 1, "revision": 3, "requires-python": "==3.12.*", "resolution-markers": [TARGET_MARKER], "supported-markers": [TARGET_MARKER], "package": [{"name": "torch", "version": "2.7.1+cpu", "source": {"registry": "https://download.pytorch.org/whl/cpu"}, "dependencies": [{"name": "demo", "marker": TARGET_MARKER}], "wheels": [dict(artifact)]}, {"name": "demo", "version": "1.0", "source": {"virtual": "."}, "dependencies": [], "metadata": {"requires-dist": [{"name": "huggingface-hub", "specifier": "==1.27.0"}, {"name": "safetensors", "specifier": "==0.7.0"}, {"name": "torch", "specifier": "==2.7.1", "index": "https://download.pytorch.org/whl/cpu"}]}}]}
    assert _lock_artifacts_complete(lock)
    tampered = json.loads(json.dumps(lock)); tampered["package"][0]["wheels"][0]["size"] = True; assert not _lock_artifacts_complete(tampered)
    tampered = json.loads(json.dumps(lock)); tampered["package"][0].pop("wheels"); assert not _lock_artifacts_complete(tampered)
    tampered = json.loads(json.dumps(lock)); tampered["package"][0]["unknown"] = 1; assert not _lock_artifacts_complete(tampered)
    tampered = json.loads(json.dumps(lock)); tampered["package"][0]["wheels"][0]["upload-time"] = ""; assert not _lock_artifacts_complete(tampered)
    tampered = json.loads(json.dumps(lock)); tampered["package"][0]["dependencies"] = [{"name": "torch"}]; assert not _lock_artifacts_complete(tampered)
    tampered = json.loads(json.dumps(lock)); tampered["package"].append(dict(tampered["package"][1])); assert not _lock_artifacts_complete(tampered)
    tampered = json.loads(json.dumps(lock)); tampered["package"][0]["wheels"][0]["url"] = "https://evil.example/t.whl"; assert not _lock_artifacts_complete(tampered)
    tampered = json.loads(json.dumps(lock)); tampered["resolution-markers"] = ["platform_machine == 'aarch64'"]; assert not _lock_artifacts_complete(tampered)
    tampered = json.loads(json.dumps(lock)); tampered["package"][0]["source"] = {"registry": "https://evil.example/simple"}; assert not _lock_artifacts_complete(tampered)
    # Exercise copies of every committed lock, including each actual row
    # variant.  These are metadata-only mutations; no resolver or network is
    # involved and every malformed copy must fail closed rather than raising.
    for project_dir in ("magnet_medium_30secs", "magnet_small_10secs", "melodyflow_t24_30secs"):
        committed = tomllib.loads((Path(__file__).resolve().parent / project_dir / "uv.lock").read_bytes().decode())
        assert _lock_artifacts_complete(committed), project_dir
        registry_indexes = [i for i, row in enumerate(committed["package"]) if row.get("source", {}).get("registry")]
        assert registry_indexes
        for index in registry_indexes:
            row = committed["package"][index]
            for field in ("name", "version", "source", "wheels"):
                changed = copy.deepcopy(committed); changed["package"][index].pop(field); assert not _lock_artifacts_complete(changed)
            changed = copy.deepcopy(committed); changed["package"][index]["unexpected"] = True; assert not _lock_artifacts_complete(changed)
            changed = copy.deepcopy(committed); changed["package"][index]["source"] = {"registry": "https://evil.example/simple"}; assert not _lock_artifacts_complete(changed)
            wheel = changed["package"][index]["wheels"][0]; wheel["url"] = "https://evil.example/file.whl"; assert not _lock_artifacts_complete(changed)
            changed = copy.deepcopy(committed); wheel = changed["package"][index]["wheels"][0]; wheel["size"] = False; assert not _lock_artifacts_complete(changed)
            changed = copy.deepcopy(committed); wheel = changed["package"][index]["wheels"][0]; wheel["upload-time"] = ""; assert not _lock_artifacts_complete(changed)
            if "dependencies" in row:
                changed = copy.deepcopy(committed); changed["package"][index]["dependencies"][0].pop("marker", None); assert not _lock_artifacts_complete(changed)
            if "sdist" in row:
                changed = copy.deepcopy(committed); changed["package"][index]["sdist"].pop("hash"); assert not _lock_artifacts_complete(changed)
        virtual_index = next(i for i, row in enumerate(committed["package"]) if row.get("source") == {"virtual": "."})
        changed = copy.deepcopy(committed); changed["package"][virtual_index]["metadata"]["requires-dist"][0].pop("specifier"); assert not _lock_artifacts_complete(changed)
        changed = copy.deepcopy(committed); changed["package"][virtual_index]["metadata"]["unexpected"] = True; assert not _lock_artifacts_complete(changed)
    schema_project = tomllib.loads((Path(__file__).resolve().parent / "magnet_medium_30secs" / "pyproject.toml").read_text(encoding="utf-8"))
    for section, key in (("project", "dependencies"), ("tool", "uv"), ("tool", "vokra")):
        changed = copy.deepcopy(schema_project)
        if section == "project":
            changed[section].pop(key)
        elif key == "uv":
            changed[section].pop(key)
        else:
            changed[section][key].pop("audiocraft_reference")
        try:
            _validate_pyproject_schema(changed)
        except ValueError:
            pass
        else:
            raise AssertionError(f"missing pyproject field accepted: {section}.{key}")
    changed = copy.deepcopy(schema_project); changed["unknown"] = True
    try:
        _validate_pyproject_schema(changed)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown pyproject field accepted")
    with tempfile.TemporaryDirectory() as directory:
        evidence = Path(directory) / "approval.json"
        base = {"manifest_sha256": "a" * 64, "scope_sha256": "b" * 64, "signer": "owner@example.invalid", "decision": "APPROVE_OWNER_CLEARANCE_FOR_VAST_ONLY"}
        base["digest"] = _sha256(_canonical({"manifest_sha256": base["manifest_sha256"], "scope_sha256": base["scope_sha256"], "signer": base["signer"], "decision": base["decision"]}))
        record = dict(base); record["evidence_sha256"] = _sha256(_canonical(record)); evidence.write_text(json.dumps(record), encoding="utf-8"); assert _read_approval(evidence, "a" * 64, "b" * 64, "owner@example.invalid")[0]
        for key, value in (("scope_sha256", "d" * 64), ("signer", "other@example.invalid"), ("decision", "REJECT")):
            changed = dict(record); changed[key] = value; evidence.write_text(json.dumps(changed), encoding="utf-8"); assert not _read_approval(evidence, "a" * 64, "b" * 64, "owner@example.invalid")[0]
        evidence.write_text('{"manifest_sha256":"' + "a" * 64 + '","manifest_sha256":"' + "a" * 64 + '"}', encoding="utf-8")
        assert not _read_approval(evidence, "a" * 64, "b" * 64, "owner@example.invalid")[0]
    with tempfile.TemporaryDirectory(prefix="audiocraft-approved-") as directory:
        root = Path(directory); project_path = root / "pyproject.toml"; lock_path = root / "uv.lock"; evidence = root / "approval.json"
        source_project = Path(__file__).resolve().parent / "magnet_medium_30secs" / "pyproject.toml"
        lock_source = source_project.with_name("uv.lock")
        project_text = source_project.read_text(encoding="utf-8")
        project_text = project_text.replace('operator_approval = { decision = "PENDING_OWNER_SIGNOFF", signer = "", digest = "" }', 'operator_approval = { decision = "APPROVED", signer = "owner@example.invalid", digest = "' + "0" * 64 + '" }').replace('owner_clearance = "UNRESOLVED_OWNER_SIGNOFF"', 'owner_clearance = "APPROVED_OWNER_SIGNOFF"')
        project_path.write_text(project_text, encoding="utf-8"); lock_path.write_bytes(lock_source.read_bytes())
        parsed_project = tomllib.loads(project_path.read_bytes().decode()); parsed_lock = tomllib.loads(lock_path.read_bytes().decode()); policy = parsed_project["tool"]["vokra"]["audiocraft_reference"]; approved_rows = _package_rows(parsed_lock); records = policy["license_records"]; project_sha = _sha256(project_path.read_bytes()); lock_sha = _sha256(lock_path.read_bytes()); scope_sha = _approval_scope(policy, project_sha, lock_sha, approved_rows, records)
        project_text = re.sub(r'approval_scope_sha256 = "[0-9a-f]{64}"', f'approval_scope_sha256 = "{scope_sha}"', project_text).replace('digest = "' + "0" * 64 + '"', f'digest = "{scope_sha}"')
        project_path.write_text(project_text, encoding="utf-8")
        parsed_project = tomllib.loads(project_path.read_bytes().decode()); policy = parsed_project["tool"]["vokra"]["audiocraft_reference"]; scope_sha = _approval_scope(policy, _sha256(project_path.read_bytes()), lock_sha, approved_rows, records)
        # The scope is independent of the operator digest; the preceding value is final.
        core = {"manifest_sha256": _sha256(project_path.read_bytes()), "scope_sha256": scope_sha, "signer": "owner@example.invalid", "decision": "APPROVE_OWNER_CLEARANCE_FOR_VAST_ONLY"}; approval_digest = _sha256(_canonical(core)); record = {**core, "digest": approval_digest}; record["evidence_sha256"] = _sha256(_canonical(record)); evidence.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        previous = EXPECTED_PROJECT_SHA256["magnet-medium-30secs"]; EXPECTED_PROJECT_SHA256["magnet-medium-30secs"] = _sha256(project_path.read_bytes())
        try:
            rc, reason = inspect_project(root, evidence)
            if rc != 0:
                print(f"approved temporary baseline was rejected: {reason}", file=sys.stderr); return 1
            for label, mutate in (("owner-clearance", lambda text: text.replace("APPROVED_OWNER_SIGNOFF", "UNRESOLVED_OWNER_SIGNOFF")), ("model-identity", lambda text: text.replace("readme_sha256 = \"85f191c1d886dc8e907986a100f0415751cb86479d18268d2cfacd614b5fd6db\"", "readme_sha256 = \"" + "0" * 64 + "\""))):
                original = project_path.read_text(encoding="utf-8"); project_path.write_text(mutate(original), encoding="utf-8"); changed_rc, _ = inspect_project(root, evidence)
                if changed_rc == 0:
                    print(f"approved baseline accepted {label} tamper", file=sys.stderr); return 1
                project_path.write_text(original, encoding="utf-8")
            original_lock = lock_path.read_bytes(); lock_path.write_bytes(original_lock + b"\n# tampered\n"); changed_rc, _ = inspect_project(root, evidence)
            if changed_rc == 0:
                print("approved baseline accepted lock tamper", file=sys.stderr); return 1
        finally:
            EXPECTED_PROJECT_SHA256["magnet-medium-30secs"] = previous
    print("audiocraft_safe_gate self-test: PASS")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--project", type=Path); parser.add_argument("--approval-evidence", type=Path); parser.add_argument("--self-test", action="store_true"); args = parser.parse_args(argv)
    if args.self_test:
        if args.project is not None or args.approval_evidence is not None: parser.error("--self-test accepts no other options")
        return self_test()
    if args.project is None or args.approval_evidence is None: parser.error("--project and --approval-evidence are required unless --self-test is given")
    rc, message = inspect_project(args.project, args.approval_evidence); print(message, file=sys.stderr); return rc


if __name__ == "__main__":
    raise SystemExit(main())
