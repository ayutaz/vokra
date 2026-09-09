#!/usr/bin/env python3
"""Offline, fail-closed gate for the Qwen3-ASR reference environment.

This module intentionally uses only the Python standard library.  It must run
before ``uv sync`` or any model/source download, so an unresolved dependency,
native/bundled-code review, or operator approval cannot be hidden by a fresh
environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from urllib.parse import urlparse
from pathlib import Path
from typing import Any

import tomllib

try:
    from wheel_audit import (
        BACKEND_MEMBERS,
        LICENSE_BYTES,
        LICENSE_MEMBER,
        LICENSE_SHA256,
        RECORD_BYTES,
        RECORD_MEMBER,
        RECORD_SHA256,
        WHEEL_BYTES,
        WHEEL_SHA256,
        WHEEL_URL,
    )
except ModuleNotFoundError:  # pragma: no cover - package import path
    from tools.parity.qwen3_asr.wheel_audit import (
        BACKEND_MEMBERS,
        LICENSE_BYTES,
        LICENSE_MEMBER,
        LICENSE_SHA256,
        RECORD_BYTES,
        RECORD_MEMBER,
        RECORD_SHA256,
        WHEEL_BYTES,
        WHEEL_SHA256,
        WHEEL_URL,
    )

GATE_VERSION = 1
LOCK_SHA256 = "807bf3ad2cbc236c7a83b6a5fcd2d5d36fe1a5bccd2eef4d1b659aee49b2e6d1"
PYPROJECT_SHA256 = "aa12415124f42b414e5fee217aafab3240c0dc2290e8705a257b1482bc96bba2"
OFFICIAL_WHEEL_SCHEMA = "vokra-qwen3-asr-official-transformers-wheel-v1"
REFERENCE_AUDIO_SHA256 = "241c0d93cc7ed8792c85c525d1e02b8c33850b791902a5e75b79c2d500e71a1a"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
GRADIO_CLIENT_SOURCE_EVIDENCE_FILENAME = "gradio_client_2_5_0_source_evidence.json"
GRADIO_CLIENT_SOURCE_EVIDENCE_SCHEMA = "vokra-qwen3-asr-gradio-client-source-evidence-v1"
GRADIO_CLIENT_SOURCE_EVIDENCE_SHA256 = "3e582bc2dc7651e3f7196786d3b222d1cf1d00c932f136e516cdb8c3696e3c12"
GRADIO_CLIENT_SOURCE_COMMIT = "43f5de68579919b0632ceb6107a99c629483ea2f"
GRADIO_CLIENT_SOURCE_REPOSITORY = "gradio-app/gradio"
GRADIO_CLIENT_PACKAGE = "gradio-client"
GRADIO_CLIENT_VERSION = "2.5.0"
PYTORCH_CPU_REGISTRY = "https://download.pytorch.org/whl/cpu"
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
DEPENDENCY_KEYS = (
    frozenset({"name"}),
    frozenset({"name", "marker"}),
    frozenset({"name", "extra"}),
    frozenset({"name", "extra", "marker"}),
    frozenset({"name", "version", "source"}),
    frozenset({"name", "version", "source", "marker"}),
)
REGISTRY_PACKAGE_KEYS = (
    frozenset({"name", "version", "source", "sdist"}),
    frozenset({"name", "version", "source", "sdist", "wheels"}),
    frozenset({"name", "version", "source", "wheels"}),
    frozenset({"name", "version", "source", "dependencies"}),
    frozenset({"name", "version", "source", "dependencies", "sdist"}),
    frozenset({"name", "version", "source", "dependencies", "sdist", "wheels"}),
    frozenset({"name", "version", "source", "dependencies", "wheels"}),
    frozenset({"name", "version", "source", "dependencies", "resolution-markers", "wheels"}),
)
REQUIRES_DIST_KEYS = (
    frozenset({"name", "specifier"}),
    frozenset({"name", "specifier", "extras"}),
    frozenset({"name", "specifier", "marker"}),
    frozenset({"name", "specifier", "extras", "marker"}),
    frozenset({"name", "specifier", "index"}),
    frozenset({"name", "git"}),
)
REVIEW_PLACEHOLDERS = {
    "", "unresolved", "pending", "pending_review", "owner_review_required",
    "review_required", "todo", "null", "none",
}

VARIANTS = [
    {
        "slug": "0.6b",
        "repo": "Qwen/Qwen3-ASR-0.6B",
        "revision": "5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
        "model_kind": "qwen3-asr-0.6b",
        "cpu_test": "qwen3_asr_0_6b_cpu_matches_official_reference",
    },
    {
        "slug": "1.7b",
        "repo": "Qwen/Qwen3-ASR-1.7B",
        "revision": "7278e1e70fe206f11671096ffdd38061171dd6e5",
        "model_kind": "qwen3-asr-1.7b",
        "cpu_test": "qwen3_asr_1_7b_cpu_matches_official_reference",
    },
]
MODEL_IDENTITIES = [
    {"repo": "Qwen/Qwen3-ASR-0.6B", "revision": "5eb144179a02acc5e5ba31e748d22b0cf3e303b0", "license_status": "PENDING_REVIEW", "license_digest": None},
    {"repo": "Qwen/Qwen3-ASR-1.7B", "revision": "7278e1e70fe206f11671096ffdd38061171dd6e5", "license_status": "PENDING_REVIEW", "license_digest": None},
]


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_digest(value: Any) -> str:
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


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


def validate_gradio_client_source_evidence(value: Any) -> None:
    """Validate the fixed primary-source mapping without inferring approval."""
    expected_artifacts = [
        {
            "kind": "sdist",
            "filename": "gradio_client-2.5.0.tar.gz",
            "url": "https://files.pythonhosted.org/packages/e8/e6/6b6029f5fe2ad7f1211105d530e34d991014c2cae463f9223033031cfc4f/gradio_client-2.5.0.tar.gz",
            "bytes": 59013,
            "sha256": "4cde99bad62149595c30c90876ca2e405e3a13687ecf895474f3412cb476673d",
            "upload_time": "2026-04-20T23:16:21.518Z",
            "sigstore_entry": 1343702326,
            "integrity_url": "https://pypi.org/integrity/gradio-client/2.5.0/gradio_client-2.5.0.tar.gz/provenance",
        },
        {
            "kind": "wheel",
            "filename": "gradio_client-2.5.0-py3-none-any.whl",
            "url": "https://files.pythonhosted.org/packages/78/81/0a861b8e1ff42960139c6cd4c7dd591292fa09ea1ae2d87677441cba4c00/gradio_client-2.5.0-py3-none-any.whl",
            "bytes": 59952,
            "sha256": "d43e2179c29076292a76485ad7ed2e6eaa19d14ac58283bd7f5beabfe4ca958c",
            "upload_time": "2026-04-20T23:16:20.186Z",
            "sigstore_entry": 1343702371,
            "integrity_url": "https://pypi.org/integrity/gradio-client/2.5.0/gradio_client-2.5.0-py3-none-any.whl/provenance",
        },
    ]
    expected_sources = [
        {
            "path": "client/python/gradio_client/package.json",
            "git_blob": "aeaddd5b21f53e9b5d121f8a9cadb2a3e0e8700d",
            "bytes": 132,
            "sha256": "c5b4ca18417503a049c351fd5af142436f2464d4cd85d14cec4fe66239e07b90",
            "url": f"https://github.com/{GRADIO_CLIENT_SOURCE_REPOSITORY}/blob/{GRADIO_CLIENT_SOURCE_COMMIT}/client/python/gradio_client/package.json",
            "declarations": {"version": "2.5.0"},
        },
        {
            "path": "client/python/pyproject.toml",
            "git_blob": "c7dcf0846f28d69ae5ef19d35027eed15f10ec34",
            "bytes": 2141,
            "sha256": "6ae27b2aa511b1f84305d601e948dfb22021c8488d3025a66e01612879745787",
            "url": f"https://github.com/{GRADIO_CLIENT_SOURCE_REPOSITORY}/blob/{GRADIO_CLIENT_SOURCE_COMMIT}/client/python/pyproject.toml",
            "declarations": {
                "project": "gradio_client",
                "dynamic_version_path": "gradio_client/package.json",
                "license": "Apache-2.0",
            },
        },
        {
            "path": "LICENSE",
            "git_blob": "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64",
            "bytes": 11357,
            "sha256": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
            "url": f"https://github.com/{GRADIO_CLIENT_SOURCE_REPOSITORY}/blob/{GRADIO_CLIENT_SOURCE_COMMIT}/LICENSE",
            "declarations": {"license": "Apache-2.0"},
        },
        {
            "path": ".github/workflows/publish.yml",
            "git_blob": "90ce46c10e1d22311344aebc4def23049ab6a758",
            "bytes": 3272,
            "sha256": "c880b48223142ac679714df18e4f7ddb127e9cbc567125e11d1008231897e9a6",
            "url": f"https://github.com/{GRADIO_CLIENT_SOURCE_REPOSITORY}/blob/{GRADIO_CLIENT_SOURCE_COMMIT}/.github/workflows/publish.yml",
            "declarations": {"workflow": ".github/workflows/publish.yml"},
        },
    ]
    expected = {
        "schema": GRADIO_CLIENT_SOURCE_EVIDENCE_SCHEMA,
        "package": {
            "name": GRADIO_CLIENT_PACKAGE,
            "version": GRADIO_CLIENT_VERSION,
            "pypi_project_url": "https://pypi.org/project/gradio-client/2.5.0/",
            "artifacts": expected_artifacts,
        },
        "trusted_publishing": {
            "repository": GRADIO_CLIENT_SOURCE_REPOSITORY,
            "workflow": ".github/workflows/publish.yml",
            "ref": "refs/heads/main",
            "commit": GRADIO_CLIENT_SOURCE_COMMIT,
            "event": "push",
        },
        "source_commit": GRADIO_CLIENT_SOURCE_COMMIT,
        "source_files": expected_sources,
        "license": "Apache-2.0",
        "factual_status": "SOURCE_MAPPING_EVIDENCE_COMPLETE",
        "package_review_status": "PENDING_REVIEW",
        "operator_approval": "PENDING",
        "publication": "NO_UPLOAD",
    }
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("gradio-client source evidence schema drifted")
    if value != expected:
        raise ValueError("gradio-client source evidence identity drifted")


def load_gradio_client_source_evidence(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("gradio-client source evidence is missing")
    raw = path.read_bytes()
    evidence = strict_json_loads(raw.decode("utf-8"))
    validate_gradio_client_source_evidence(evidence)
    return evidence, digest_bytes(raw)


def _artifact(value: Any, *, allow_missing_size: bool = False) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) not in ({"url", "hash", "size", "upload-time"}, {"url", "hash", "upload-time"}):
        raise ValueError("uv.lock artifact must have exactly url/hash/size/upload-time")
    if "size" not in value and (not allow_missing_size or (value.get("url"), value.get("hash")) not in PYTORCH_CPU_ARTIFACTS_WITHOUT_SIZE):
        raise ValueError("uv.lock artifact size is missing outside the reviewed PyTorch CPU exception")
    if (not isinstance(value["url"], str) or not value["url"].startswith("https://")
            or urlparse(value["url"]).hostname not in {"files.pythonhosted.org", "download-r2.pytorch.org", "download.pytorch.org"}
            or not isinstance(value["hash"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value["hash"])
            or not isinstance(value["upload-time"], str) or not value["upload-time"].strip()):
        raise ValueError("uv.lock artifact has invalid URL/hash/size/upload-time")
    if "size" in value and (not isinstance(value["size"], int) or isinstance(value["size"], bool) or value["size"] <= 0):
        raise ValueError("uv.lock artifact has invalid URL/hash/size/upload-time")
    return value


def _validate_lock_shape(lock: dict[str, Any], project: dict[str, Any]) -> None:
    if not isinstance(project, dict) or not isinstance(project.get("project"), dict) or not isinstance(project.get("tool"), dict):
        raise ValueError("pyproject root schema drifted")
    if set(lock) != {"version", "revision", "requires-python", "resolution-markers", "manifest", "package"}:
        raise ValueError("uv.lock top-level schema drifted")
    if not isinstance(lock["version"], int) or isinstance(lock["version"], bool) or lock["version"] != 1 or not isinstance(lock["revision"], int) or lock["revision"] != 3:
        raise ValueError("uv.lock version/revision types drifted")
    if not isinstance(lock["requires-python"], str) or not isinstance(lock["resolution-markers"], list) or not isinstance(lock["manifest"], dict) or set(lock["manifest"]) != {"constraints"} or not isinstance(lock["manifest"]["constraints"], list) or any(not isinstance(row, dict) or set(row) != {"name", "specifier"} or not isinstance(row["name"], str) or not isinstance(row["specifier"], str) for row in lock["manifest"]["constraints"]):
        raise ValueError("uv.lock top-level value types drifted")
    if set(project) != {"project", "tool"} or set(project["project"]) != {"name", "version", "description", "requires-python", "dependencies"}:
        raise ValueError("pyproject schema drifted")
    if not isinstance(project["project"]["dependencies"], list) or set(project["tool"]) != {"uv"}:
        raise ValueError("pyproject dependency/tool schema drifted")
    uv = project["tool"]["uv"]
    if not isinstance(uv, dict) or set(uv) != {"package", "constraint-dependencies", "index", "sources"} or not isinstance(uv["constraint-dependencies"], list) or not isinstance(uv["index"], list) or not isinstance(uv["sources"], dict):
        raise ValueError("pyproject uv configuration drifted")
    packages = lock["package"]
    if not isinstance(packages, list) or not packages:
        raise ValueError("uv.lock package table is missing")
    identities: set[tuple[str, str, str]] = set()
    virtual = 0
    for package in packages:
        if not isinstance(package, dict) or not isinstance(package.get("name"), str) or not isinstance(package.get("version"), str):
            raise ValueError("uv.lock contains a malformed package row")
        if not package["name"].strip() or not package["version"].strip() or ("resolution-markers" in package and not isinstance(package["resolution-markers"], list)):
            raise ValueError("uv.lock package name/version/markers are malformed")
        source = package.get("source")
        if not isinstance(source, dict) or set(source) not in ({"virtual"}, {"registry"}):
            raise ValueError("uv.lock package source schema drifted")
        if "virtual" in source:
            virtual += 1
            if source != {"virtual": "."} or set(package) != {"name", "version", "source", "dependencies", "metadata"}:
                raise ValueError("uv.lock virtual root schema drifted")
            if package["name"] != project["project"]["name"] or package["version"] != project["project"]["version"]:
                raise ValueError("uv.lock virtual root is not bound to pyproject")
            if not isinstance(package["metadata"], dict) or set(package["metadata"]) != {"requires-dist"} or not isinstance(package["metadata"]["requires-dist"], list):
                raise ValueError("uv.lock virtual metadata drifted")
            for requirement in package["metadata"]["requires-dist"]:
                if not isinstance(requirement, dict) or frozenset(requirement) not in REQUIRES_DIST_KEYS or not isinstance(requirement.get("name"), str) or not isinstance(requirement.get("specifier", requirement.get("git")), str):
                    raise ValueError("uv.lock requires-dist row drifted")
                if "index" in requirement and requirement["index"] != "https://download.pytorch.org/whl/cpu":
                    raise ValueError("uv.lock requires-dist index drifted")
                if "extras" in requirement and (not isinstance(requirement["extras"], list) or any(not isinstance(x, str) for x in requirement["extras"])):
                    raise ValueError("uv.lock requires-dist extras drifted")
                if "marker" in requirement and not isinstance(requirement["marker"], str):
                    raise ValueError("uv.lock requires-dist marker drifted")
        else:
            registry = source["registry"]
            if not isinstance(registry, str):
                raise ValueError("uv.lock registry source is malformed")
            if registry not in {"https://pypi.org/simple", "https://download.pytorch.org/whl/cpu"}:
                raise ValueError("uv.lock contains an unreviewed registry")
            if frozenset(package) not in REGISTRY_PACKAGE_KEYS:
                raise ValueError("uv.lock package schema drifted")
            if registry == "https://download.pytorch.org/whl/cpu" and package["name"] not in {"torch", "torchaudio"}:
                raise ValueError("CPU registry used by an unexpected package")
            if registry == "https://pypi.org/simple" and package["name"] in {"torch", "torchaudio"}:
                raise ValueError("torch package is not bound to CPU registry")
            if "optional-dependencies" in package and (not isinstance(package["optional-dependencies"], dict) or set(package["optional-dependencies"]) != set()):
                raise ValueError("uv.lock optional dependency metadata drifted")
            if "sdist" in package and not isinstance(package["sdist"], dict):
                raise ValueError("uv.lock sdist is malformed")
            if "sdist" in package:
                _artifact(
                    package["sdist"],
                    allow_missing_size=registry == PYTORCH_CPU_REGISTRY and package["name"] == "torch",
                )
            if "wheels" in package:
                if not isinstance(package["wheels"], list):
                    raise ValueError("uv.lock wheels are malformed")
                for wheel in package["wheels"]:
                    _artifact(
                        wheel,
                        allow_missing_size=registry == PYTORCH_CPU_REGISTRY and package["name"] == "torch",
                    )
            expected_host = "download-r2.pytorch.org" if registry == "https://download.pytorch.org/whl/cpu" else "files.pythonhosted.org"
            for artifact in ([package["sdist"]] if "sdist" in package else []) + package.get("wheels", []):
                if urlparse(artifact["url"]).hostname != expected_host:
                    raise ValueError("uv.lock artifact host is not bound to its registry")
        dependencies = package.get("dependencies", [])
        if not isinstance(dependencies, list):
            raise ValueError("uv.lock dependencies are not a list")
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
        identity = (package["name"], package["version"], json.dumps(source, sort_keys=True))
        if identity in identities:
            raise ValueError("uv.lock contains duplicate package identity")
        identities.add(identity)
    if virtual != 1:
        raise ValueError("uv.lock must contain exactly one virtual root")


def canonical_package_rows(lock: dict[str, Any]) -> list[dict[str, Any]]:
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise ValueError("uv.lock package table is missing")
    rows: list[dict[str, Any]] = []
    for package in packages:
        if not isinstance(package, dict):
            raise ValueError("uv.lock contains a malformed package row")
        dependencies = package.get("dependencies", [])
        if not isinstance(dependencies, list):
            raise ValueError("uv.lock contains malformed dependencies")
        source = package.get("source")
        if not isinstance(source, dict):
            raise ValueError("uv.lock package source is malformed")
        rows.append(
            {
                "name": package.get("name"),
                "version": package.get("version"),
                "source": package.get("source"),
                "resolution-markers": package.get("resolution-markers", []),
                "dependencies": dependencies,
                "artifacts": {
                    "sdist": package.get("sdist"),
                    "wheels": package.get("wheels", []),
                },
            }
        )
    if any(not isinstance(row["name"], str) or not isinstance(row["version"], str) for row in rows):
        raise ValueError("uv.lock has a package row without an exact name/version")
    return sorted(rows, key=lambda row: (row["name"], row["version"]))


def expected_official_wheel() -> dict[str, Any]:
    return {
        "schema": OFFICIAL_WHEEL_SCHEMA,
        "name": "qwen-asr",
        "version": "0.0.6",
        "url": WHEEL_URL,
        "bytes": WHEEL_BYTES,
        "sha256": WHEEL_SHA256,
        "license": {
            "member": LICENSE_MEMBER,
            "bytes": LICENSE_BYTES,
            "sha256": LICENSE_SHA256,
            "spdx": "Apache-2.0",
        },
        "record": {
            "member": RECORD_MEMBER,
            "bytes": RECORD_BYTES,
            "sha256": RECORD_SHA256,
        },
        "official_backend_members": {
            name: {"bytes": size, "sha256": digest}
            for name, (size, digest) in sorted(BACKEND_MEMBERS.items())
        },
        "execution": "OFFICIAL_TRANSFORMERS_BACKEND_ONLY",
        "publication": "NO_UPLOAD",
    }


def blocked(reason: str) -> tuple[bool, str]:
    return False, reason


def reviewed_value(value: Any) -> bool:
    normalized = re.sub(r"\s+", "_", value.strip()).casefold() if isinstance(value, str) else ""
    return normalized not in REVIEW_PLACEHOLDERS


def validate(project: Path, manifest_path: Path, evidence_path: Path | None = None) -> tuple[bool, str]:
    lock_path = project / "uv.lock"
    pyproject_path = project / "pyproject.toml"
    if any(path.is_symlink() or not path.is_file() for path in (lock_path, pyproject_path, manifest_path)):
        return blocked("project lock/pyproject or gate manifest is missing")
    try:
        manifest = strict_json_loads(manifest_path.read_text(encoding="utf-8"))
        lock_bytes = lock_path.read_bytes()
        pyproject_bytes = pyproject_path.read_bytes()
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return blocked(f"gate inputs are unreadable: {exc}")
    if not isinstance(manifest, dict):
        return blocked("gate manifest root must be an object")
    if set(manifest) != {
        "gate_version", "lock_sha256", "pyproject_sha256", "package_rows_sha256",
        "review_rows", "review_rows_sha256", "variants", "model_identities",
        "reference_audio", "official_wheel",
        "approval_scope_sha256", "operator_approval",
    }:
        return blocked("gate manifest schema drifted")
    if manifest.get("gate_version") != GATE_VERSION:
        return blocked("unsupported gate manifest version")
    try:
        lock = tomllib.loads(lock_bytes.decode("utf-8"))
        project_data = tomllib.loads(pyproject_bytes.decode("utf-8"))
        _validate_lock_shape(lock, project_data)
        rows = canonical_package_rows(lock)
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError) as exc:
        return blocked(f"uv.lock canonicalization failed: {exc}")
    if digest_bytes(lock_bytes) != LOCK_SHA256 or manifest.get("lock_sha256") != LOCK_SHA256:
        return blocked("uv.lock bytes are not the reviewed exact lock")
    if digest_bytes(pyproject_bytes) != PYPROJECT_SHA256 or manifest.get("pyproject_sha256") != PYPROJECT_SHA256:
        return blocked("pyproject.toml bytes are not the reviewed exact project")
    if canonical_digest(rows) != manifest.get("package_rows_sha256"):
        return blocked("canonical version/source/marker/dependency rows drifted")
    try:
        official_wheel = manifest["official_wheel"]
        expected_wheel = expected_official_wheel()
        if official_wheel != expected_wheel:
            raise ValueError("official qwen-asr wheel identity drifted")
    except (KeyError, TypeError, ValueError) as exc:
        return blocked(f"official qwen-asr wheel evidence is invalid: {exc}")
    identities = [f'{row["name"]}@{row["version"]}' for row in rows]
    review_rows = manifest.get("review_rows")
    if not isinstance(review_rows, list):
        return blocked("version-keyed dependency review rows are missing")
    review_ids = [row.get("id") for row in review_rows if isinstance(row, dict)]
    if review_ids != sorted(identities) or len(review_ids) != len(identities) or len(set(review_ids)) != len(identities):
        return blocked("dependency review rows do not cover the exact lock identities")
    if canonical_digest(review_rows) != manifest.get("review_rows_sha256"):
        return blocked("dependency review row digest drifted")
    if manifest.get("variants") != VARIANTS:
        return blocked("fixed Qwen3-ASR repositories/revisions/tests drifted")
    model_identities = manifest.get("model_identities")
    if not isinstance(model_identities, list) or any(not isinstance(item, dict) for item in model_identities) or [
        {key: item.get(key) for key in ("repo", "revision")} for item in model_identities
    ] != [{key: item[key] for key in ("repo", "revision")} for item in MODEL_IDENTITIES]:
        return blocked("model weight/license identities drifted")
    for identity in model_identities:
        if (
            identity["license_status"] != "REVIEWED"
            or not isinstance(identity["license_digest"], str)
            or not HEX64.fullmatch(identity["license_digest"])
            or not reviewed_value(identity.get("native_review"))
            or not reviewed_value(identity.get("bundled_review"))
            or not reviewed_value(identity.get("evidence"))
        ):
            return blocked(f"model license review is unresolved: {identity['repo']}@{identity['revision']}")
    audio = manifest.get("reference_audio")
    if not isinstance(audio, dict) or audio.get("path") != "tests/parity/utmos/ref-clip.wav" or audio.get("sha256") != REFERENCE_AUDIO_SHA256:
        return blocked("fixed reference-audio identity drifted")
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
        "variants": VARIANTS,
        "model_identities": model_identities,
        "reference_audio": audio,
        "official_wheel": official_wheel,
        "review_rows": review_rows,
    }
    scope_sha256 = canonical_digest(scope)
    if manifest.get("approval_scope_sha256") != scope_sha256:
        return blocked("operator approval scope is not bound to the exact inputs")
    approval = manifest.get("operator_approval")
    if (
        not isinstance(approval, dict)
        or approval.get("schema") != "v1"
        or approval.get("decision") != "APPROVED"
        or not isinstance(approval.get("signer"), str)
        or not approval["signer"]
        or not isinstance(approval.get("digest"), str)
        or not HEX64.fullmatch(approval["digest"])
        or approval["digest"] != scope_sha256
    ):
        return blocked("exact operator approval is pending or invalid")
    if evidence_path is None:
        evidence_path = manifest_path.with_name("license_gate_evidence.json")
    if evidence_path.is_symlink() or not evidence_path.is_file():
        return blocked("authenticated operator approval evidence is missing")
    try:
        evidence = strict_json_loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return blocked(f"operator approval evidence is unreadable: {exc}")
    if (
        not isinstance(evidence, dict)
        or evidence.get("schema") != "v1"
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


def main(project: Path, manifest: Path, evidence: Path | None) -> int:
    ok, reason = validate(project, manifest, evidence)
    if not ok:
        print(f"qwen3-asr preflight gate: BLOCKED: {reason}", file=sys.stderr)
        return 2
    print("qwen3-asr preflight gate: PASS")
    return 0


def self_test() -> int:
    project = Path(__file__).resolve().parent
    manifest = project / "license_gate_manifest.json"
    if reviewed_value("  PENDING_REVIEW  ") or not reviewed_value("reviewed citation: TODO was resolved"):
        print("qwen3-asr preflight gate: placeholder normalization self-test failed", file=sys.stderr)
        return 1
    reviewed_url, reviewed_hash = next(iter(PYTORCH_CPU_ARTIFACTS_WITHOUT_SIZE))
    _artifact(
        {"url": reviewed_url, "hash": reviewed_hash, "upload-time": "2026-07-08T12:26:18Z"},
        allow_missing_size=True,
    )
    try:
        _artifact(
            {"url": reviewed_url + ".tampered", "hash": reviewed_hash, "upload-time": "2026-07-08T12:26:18Z"},
            allow_missing_size=True,
        )
    except ValueError:
        pass
    else:
        print("qwen3-asr preflight gate: unreviewed missing-size artifact accepted", file=sys.stderr)
        return 1
    if expected_official_wheel()["schema"] != OFFICIAL_WHEEL_SCHEMA:
        print("qwen3-asr preflight gate: official wheel schema self-test failed", file=sys.stderr)
        return 1
    ok, reason = validate(project, manifest)
    if ok or ("unresolved" not in reason and "approval" not in reason):
        print("qwen3-asr preflight gate: self-test expected pending approval", file=sys.stderr)
        return 1
    # The checked-in manifest intentionally has no owner approval yet.  That
    # is the expected production disposition, not a self-test failure: the
    # self-test proves that the gate rejects an unresolved approval and keeps
    # the execution path fail-closed.
    print("qwen3-asr preflight gate: self-test PASS (approval remains pending)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        raise SystemExit(self_test())
    if args.project is None or args.manifest is None:
        parser.error("--project and --manifest are required")
    raise SystemExit(main(args.project, args.manifest, args.evidence))
