#!/usr/bin/env python3
"""Offline fail-closed gate for the MOSS-Audio VAST parity closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit
from pathlib import Path
from typing import Any

import tomllib

GATE_VERSION = 1
LOCK_SHA256 = "f26e7504e980c5a62fdcb1bd2ed1d9726da09c839cb9f251412b4d4145fbd59f"
PYPROJECT_SHA256 = "d321bfae5af886eb9ef0fc2fd3696c425c77c5c247353957e05317ab1efb43d0"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
REVIEW_SENTINELS = {
    "", "none", "null", "unresolved", "pending", "pending_review",
    "owner_review_required", "review_required", "todo",
}
MANIFEST_KEYS = {
    "approval_scope_sha256", "component_reviews", "dependency_reviews",
    "dependency_reviews_sha256", "fixed_identities", "gate_version",
    "lock_sha256", "no_upload", "operator_approval", "package_rows_sha256",
    "pyproject_sha256",
}
APPROVAL_KEYS = {"schema", "decision", "signer", "digest"}
EVIDENCE_KEYS = {"schema", "decision", "scope_sha256", "manifest_sha256", "signer", "digest"}
LOCK_KEYS = {"version", "revision", "requires-python", "resolution-markers", "package"}
PACKAGE_KEYS = {"name", "version", "source", "resolution-markers", "dependencies", "sdist", "wheels", "metadata"}
ARTIFACT_KEYS = {"url", "hash", "size", "upload-time"}
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

SOURCE_IDENTITY = {
    "repo": "OpenMOSS/MOSS-Audio",
    "revision": "5cbb1d823937cd5b5de3d8fa4d3a7253ebd3b883",
    "license_spdx": None,
    "license": {"path": "LICENSE", "blob_sha256": None, "payload_sha256": None},
    "files": {
        "src/configuration_moss_audio.py": "e597dca441ff7fb58a5ec43186fafdfce19f31dada4955b4910059baa5d52ebd",
        "src/modeling_moss_audio.py": "a52513e518c68a0ba7c636a1ab0e12f7755ceebd0ae033235dc5e2551bfcbf9c",
        "src/processing_moss_audio.py": "05fb788cbdc6482eded8d70f7d2f524bc0cdca47d001acab5661c11f02cc6fe6",
    },
    "source_file_contract": [
        {"path": "src/configuration_moss_audio.py", "role": "configuration", "bytes": None, "sha256": "e597dca441ff7fb58a5ec43186fafdfce19f31dada4955b4910059baa5d52ebd", "git_blob_sha1": None, "status": "UNRESOLVED"},
        {"path": "src/modeling_moss_audio.py", "role": "modeling", "bytes": None, "sha256": "a52513e518c68a0ba7c636a1ab0e12f7755ceebd0ae033235dc5e2551bfcbf9c", "git_blob_sha1": None, "status": "UNRESOLVED"},
        {"path": "src/processing_moss_audio.py", "role": "processing", "bytes": None, "sha256": "05fb788cbdc6482eded8d70f7d2f524bc0cdca47d001acab5661c11f02cc6fe6", "git_blob_sha1": None, "status": "UNRESOLVED"},
        {"path": "LICENSE", "role": "license", "bytes": None, "sha256": None, "git_blob_sha1": None, "status": "UNRESOLVED"},
    ],
}
VARIANTS = {
    "4b": {
        "repo": "OpenMOSS-Team/MOSS-Audio-4B-Instruct",
        "revision": "6907a499dc0e87cc77c8ae0fe23fd0eb5476a02d",
        "model_name": "moss-audio-4b-instruct",
        "license_spdx": None,
        "config_sha256": "e528a941446f4443f1b9fede12ea484e58a79d494c28d21ef1e73b5148abfbfa",
        "tokenizer_config_sha256": "443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0",
        "processor_config_sha256": "0749d81701d2a2a2e83ca4d549fbebb1a205acac1ac7bdccea7965c1913b2cbf",
        "license": {"path": "LICENSE", "payload_sha256": None},
        "snapshot_files": [
            {"path": "config.json", "role": "config", "bytes": None, "sha256": "e528a941446f4443f1b9fede12ea484e58a79d494c28d21ef1e73b5148abfbfa", "status": "UNRESOLVED"},
            {"path": "tokenizer_config.json", "role": "tokenizer", "bytes": 5404, "sha256": "443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0", "status": "REVIEWED"},
            {"path": "processor_config.json", "role": "processor", "bytes": 426, "sha256": "0749d81701d2a2a2e83ca4d549fbebb1a205acac1ac7bdccea7965c1913b2cbf", "status": "REVIEWED"},
            {"path": "vocab.json", "role": "common_asset", "bytes": 3383407, "sha256": "87a257b04b17642a0688c98cd1df89c398bda4fee532d6f88b38a659ecb4ac8d", "status": "REVIEWED"},
            {"path": "merges.txt", "role": "common_asset", "bytes": 1671853, "sha256": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5", "status": "REVIEWED"},
            {"path": "chat_template.jinja", "role": "common_asset", "bytes": 4116, "sha256": "87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5", "status": "REVIEWED"},
            {"path": "generation_config.json", "role": "common_asset", "bytes": 121, "sha256": "bb52bfdd308deaea4ec800bf0165e75770b0a4e5c105963bee1b0398f4043d3e", "status": "REVIEWED"},
            {"path": "model.safetensors.index.json", "role": "checkpoint_index", "bytes": None, "sha256": None, "status": "UNRESOLVED"},
            {"path": "__UNRESOLVED_CHECKPOINT_SHARD__", "role": "checkpoint_shard", "bytes": None, "sha256": None, "status": "UNRESOLVED"},
            {"path": "LICENSE", "role": "license", "bytes": None, "sha256": None, "status": "UNRESOLVED"},
        ],
    },
    "8b": {
        "repo": "OpenMOSS-Team/MOSS-Audio-8B-Instruct",
        "revision": "6521a39181b47a18f2d9f4b3acfb5bca7b76b57f",
        "model_name": "moss-audio-8b-instruct",
        "license_spdx": None,
        "config_sha256": "535154c2a5bcbd0e18e2f92bcf370ac74b530eec97ad4fd9317993ba0a316536",
        "tokenizer_config_sha256": "0869e41f5d123ff144a811f0d83c5d18871dcd4b4064f46bf9def194bfbc6f41",
        "processor_config_sha256": "6a5c462858acb299db0d2d967b63d520b72d178f44d1619c33fc860f25fdccbf",
        "license": {"path": "LICENSE", "payload_sha256": None},
        "snapshot_files": [
            {"path": "config.json", "role": "config", "bytes": None, "sha256": "535154c2a5bcbd0e18e2f92bcf370ac74b530eec97ad4fd9317993ba0a316536", "status": "UNRESOLVED"},
            {"path": "tokenizer_config.json", "role": "tokenizer", "bytes": 6114, "sha256": "0869e41f5d123ff144a811f0d83c5d18871dcd4b4064f46bf9def194bfbc6f41", "status": "REVIEWED"},
            {"path": "processor_config.json", "role": "processor", "bytes": 427, "sha256": "6a5c462858acb299db0d2d967b63d520b72d178f44d1619c33fc860f25fdccbf", "status": "REVIEWED"},
            {"path": "vocab.json", "role": "common_asset", "bytes": 3383407, "sha256": "87a257b04b17642a0688c98cd1df89c398bda4fee532d6f88b38a659ecb4ac8d", "status": "REVIEWED"},
            {"path": "merges.txt", "role": "common_asset", "bytes": 1671853, "sha256": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5", "status": "REVIEWED"},
            {"path": "chat_template.jinja", "role": "common_asset", "bytes": 4116, "sha256": "87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5", "status": "REVIEWED"},
            {"path": "generation_config.json", "role": "common_asset", "bytes": 121, "sha256": "bb52bfdd308deaea4ec800bf0165e75770b0a4e5c105963bee1b0398f4043d3e", "status": "REVIEWED"},
            {"path": "model.safetensors.index.json", "role": "checkpoint_index", "bytes": None, "sha256": None, "status": "UNRESOLVED"},
            {"path": "__UNRESOLVED_CHECKPOINT_SHARD__", "role": "checkpoint_shard", "bytes": None, "sha256": None, "status": "UNRESOLVED"},
            {"path": "LICENSE", "role": "license", "bytes": None, "sha256": None, "status": "UNRESOLVED"},
        ],
    },
}
COMMON_ASSETS = {
    "vocab.json": "87a257b04b17642a0688c98cd1df89c398bda4fee532d6f88b38a659ecb4ac8d",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "chat_template.jinja": "87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5",
    "generation_config.json": "bb52bfdd308deaea4ec800bf0165e75770b0a4e5c105963bee1b0398f4043d3e",
}


def validate_project_schema(project: dict[str, Any]) -> None:
    if set(project) != {"project", "tool"} or not isinstance(project.get("project"), dict) or set(project["project"]) != {"name", "version", "description", "requires-python", "dependencies"}:
        raise ValueError("pyproject.toml structural schema drifted")
    p = project["project"]
    if p["requires-python"] != ">=3.12,<3.13" or not isinstance(p["dependencies"], list) or any(not isinstance(x, str) or not x.strip() for x in p["dependencies"]):
        raise ValueError("pyproject.toml project contract drifted")
    tool = project["tool"]
    if set(tool) != {"uv"} or not isinstance(tool["uv"], dict) or set(tool["uv"]) != {"package", "index", "sources"}:
        raise ValueError("pyproject.toml uv schema drifted")
    uv = tool["uv"]
    if uv["package"] is not False or uv["index"] != [{"name": "pytorch-cpu", "url": "https://download.pytorch.org/whl/cpu", "explicit": True}] or uv["sources"] != {"torch": {"index": "pytorch-cpu"}, "torchaudio": {"index": "pytorch-cpu"}}:
        raise ValueError("pyproject.toml uv contract drifted")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    """Load JSON while rejecting duplicate object keys."""
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)


def canonical(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def unresolved(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    normalized = re.sub(r"\s+", "_", value.strip().casefold())
    return normalized in REVIEW_SENTINELS


def resolved_string(value: Any) -> bool:
    return isinstance(value, str) and not unresolved(value)


def resolved_hex64(value: Any) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def validate_artifact(value: Any, label: str, registry: str) -> None:
    if not isinstance(value, dict) or set(value) != ARTIFACT_KEYS:
        raise ValueError(f"{label} artifact schema is not exact")
    url = value["url"]
    expected_host = "download-r2.pytorch.org" if registry == "https://download.pytorch.org/whl/cpu" else "files.pythonhosted.org"
    parsed = urlsplit(url) if isinstance(url, str) else None
    if parsed is None or parsed.scheme != "https" or parsed.netloc != expected_host or not parsed.path:
        raise ValueError(f"{label} artifact URL is not the authenticated {expected_host} host")
    if not isinstance(value["hash"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value["hash"]):
        raise ValueError(f"{label} artifact hash is malformed")
    if isinstance(value["size"], bool) or not isinstance(value["size"], int) or value["size"] <= 0:
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


def package_rows(lock: dict[str, Any]) -> list[dict[str, Any]]:
    if set(lock) != LOCK_KEYS or lock.get("version") != 1 or lock.get("revision") != 3 or lock.get("requires-python") != "==3.12.*" or not isinstance(lock.get("resolution-markers"), list) or any(not isinstance(marker, str) or not marker for marker in lock["resolution-markers"]):
        raise ValueError("uv.lock top-level schema is malformed")
    packages = lock.get("package")
    if not isinstance(packages, list) or not packages:
        raise ValueError("uv.lock package table is missing or empty")
    rows: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for package in packages:
        if not isinstance(package, dict):
            raise ValueError("uv.lock package row is not an object")
        if set(package) - PACKAGE_KEYS:
            raise ValueError("uv.lock package row has unknown fields")
        name, version, source = package.get("name"), package.get("version"), package.get("source")
        if not isinstance(name, str) or not name.strip() or not isinstance(version, str) or not version.strip():
            raise ValueError("uv.lock package name/version is malformed")
        if not isinstance(source, dict) or len(source) != 1 or set(source) not in ({"registry"}, {"virtual"}):
            raise ValueError("uv.lock package source is malformed")
        if "registry" in source and source["registry"] not in ("https://pypi.org/simple", "https://download.pytorch.org/whl/cpu"):
            raise ValueError("uv.lock package registry is not an approved index")
        virtual = "virtual" in source
        if virtual and source["virtual"] != ".":
            raise ValueError("uv.lock virtual source is not '.'")
        if not virtual and frozenset(package) not in REGISTRY_PACKAGE_SCHEMAS:
            raise ValueError("uv.lock registry package schema is not an exact committed variant")
        markers = package.get("resolution-markers", [])
        dependencies = package.get("dependencies", [])
        if not isinstance(markers, list) or not isinstance(dependencies, list):
            raise ValueError("uv.lock package markers/dependencies are malformed")
        identity = (name, version)
        if identity in identities:
            raise ValueError("uv.lock contains a duplicate package identity")
        identities.add(identity)
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
        if virtual and set(package) != {"name", "version", "source", "dependencies", "metadata"}:
            raise ValueError("uv.lock virtual project schema is not exact")
        metadata = package.get("metadata")
        if "metadata" in package or virtual:
            validate_metadata(metadata, f"{identity!r}")
        for artifact_name in ("sdist", "wheels"):
            artifacts = package.get(artifact_name, [] if artifact_name == "wheels" else None)
            if artifact_name == "sdist" and "sdist" in package and not isinstance(artifacts, dict):
                raise ValueError("uv.lock sdist artifact is malformed")
            if artifact_name == "wheels" and "wheels" in package and not isinstance(artifacts, list):
                raise ValueError("uv.lock wheels artifact table is malformed")
            candidates = [] if artifacts is None else (artifacts if isinstance(artifacts, list) else [artifacts])
            for artifact in candidates:
                validate_artifact(artifact, f"{identity!r} {artifact_name}", source.get("registry", ""))
        if virtual and ("sdist" in package or "wheels" in package):
            raise ValueError("uv.lock virtual project must not contain artifacts")
        if not virtual and "sdist" not in package and not package.get("wheels"):
            raise ValueError("uv.lock registry package has no authenticated artifacts")
        rows.append({"name": name, "version": version, "source": source,
                     "resolution-markers": markers, "dependencies": dependencies,
                     "sdist": package.get("sdist"), "wheels": package.get("wheels", []),
                     "metadata": metadata})
    return sorted(rows, key=lambda row: (row["name"], row["version"], json.dumps(row["source"], sort_keys=True)))


def verify_snapshot(snapshot: Path, variant: str) -> tuple[bool, str]:
    return verify_snapshot_contract(snapshot, VARIANTS[variant]["snapshot_files"])


def verify_snapshot_contract(snapshot: Path, contract: list[dict[str, Any]]) -> tuple[bool, str]:
    if snapshot.is_symlink() or not snapshot.is_dir():
        return block("snapshot root is not a regular directory")
    if not isinstance(contract, list) or not contract or any(
        not isinstance(row, dict)
        or set(row) != {"path", "role", "bytes", "sha256", "status"}
        or not isinstance(row["path"], str)
        or not row["path"]
        or not isinstance(row["role"], str)
        or row["status"] != "REVIEWED"
        or not isinstance(row["bytes"], int)
        or row["bytes"] < 0
        or not isinstance(row["sha256"], str)
        or not HEX64.fullmatch(row["sha256"])
        for row in contract
    ):
        return block("snapshot file identity contract is incomplete; exact shard/index/license evidence is required")
    expected = {row["path"] for row in contract}
    if len(expected) != len(contract) or any(path in (".cache",) or "/" in path for path in expected):
        return block("snapshot file identity contract has duplicate or unsafe paths")
    try:
        entries = list(snapshot.iterdir())
    except OSError as exc:
        return block(f"snapshot is unreadable: {exc}")
    transport_cache = snapshot / ".cache"
    if transport_cache in entries and (transport_cache.is_symlink() or not transport_cache.is_dir()):
        return block("snapshot transport .cache is not a directory")
    visible = [path for path in entries if path.name != ".cache"]
    actual = {path.name for path in visible}
    if actual != expected or any(path.is_symlink() or not path.is_file() for path in visible):
        return block("snapshot has missing, extra, symlink, or non-regular entries")
    for row in contract:
        path = snapshot / row["path"]
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            return block(f"snapshot identity mismatch: {row['path']}")
    return True, "PASS"


def verify_source(
    source: Path,
    *,
    contract: list[dict[str, Any]] | None = None,
    expected_revision: str | None = None,
) -> tuple[bool, str]:
    if source.is_symlink() or not source.is_dir():
        return block("official source root is not a real directory")
    try:
        status = subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
            check=True, capture_output=True, text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        return block(f"official source worktree is unreadable: {exc}")
    if status:
        return block("official source worktree is dirty")
    try:
        revision = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return block(f"official source revision is unreadable: {exc}")
    if revision != (expected_revision or SOURCE_IDENTITY["revision"]):
        return block("official source revision mismatch")
    contract = contract or SOURCE_IDENTITY["source_file_contract"]
    if any(
        row["status"] != "REVIEWED"
        or not isinstance(row["bytes"], int)
        or not isinstance(row["sha256"], str)
        or not HEX64.fullmatch(row["sha256"])
        or not isinstance(row["git_blob_sha1"], str)
        or not HEX40.fullmatch(row["git_blob_sha1"])
        for row in contract
    ):
        return block("official source file/license identity contract is incomplete")
    for row in contract:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            return block(f"official source path escapes checkout: {row['path']}")
        path = source / relative
        try:
            contained = path.resolve().is_relative_to(source.resolve())
        except OSError:
            contained = False
        if not contained:
            return block(f"official source path escapes checkout: {row['path']}")
        if path.is_symlink() or not path.is_file() or path.stat().st_size != row["bytes"]:
            return block(f"official source file identity mismatch: {row['path']}")
        if sha256_file(path) != row["sha256"]:
            return block(f"official source payload hash mismatch: {row['path']}")
        blob = subprocess.run(
            ["git", "-C", str(source), "hash-object", row["path"]], check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        if blob != row["git_blob_sha1"]:
            return block(f"official source git blob identity mismatch: {row['path']}")
    return True, "PASS"


def fixed_identity_blocker() -> str | None:
    if not resolved_string(SOURCE_IDENTITY.get("license_spdx")):
        return "official source license SPDX identity is incomplete"
    source_license = SOURCE_IDENTITY["license"]
    if not resolved_hex64(source_license.get("blob_sha256")) or not resolved_hex64(source_license.get("payload_sha256")):
        return "official source LICENSE blob/payload identity is incomplete"
    for row in SOURCE_IDENTITY["source_file_contract"]:
        if row["status"] != "REVIEWED" or not isinstance(row["bytes"], int) or not HEX64.fullmatch(str(row["sha256"])) or not HEX40.fullmatch(str(row["git_blob_sha1"])):
            return f"official source file identity is incomplete: {row['path']}"
    for variant, identity in VARIANTS.items():
        if not resolved_string(identity.get("license_spdx")):
            return f"{variant} model license SPDX identity is incomplete"
        if not resolved_hex64(identity["license"].get("payload_sha256")):
            return f"{variant} model LICENSE identity is incomplete"
        for row in identity["snapshot_files"]:
            if row["status"] != "REVIEWED" or not isinstance(row["bytes"], int) or not HEX64.fullmatch(str(row["sha256"])):
                return f"{variant} snapshot identity is incomplete: {row['path']}"
    for component in expected_components():
        if component["kind"] == "public_artifact" and not resolved_string(component["identity"].get("license_spdx")):
            return f"{component['id']} license identity is incomplete"
        if component["kind"] == "public_artifact" and not resolved_hex64(component["identity"].get("license_payload_sha256")):
            return f"{component['id']} payload identity is incomplete"
    return None


def expected_components() -> list[dict[str, Any]]:
    components = [
        {"id": "source:OpenMOSS/MOSS-Audio@5cbb1d823937cd5b5de3d8fa4d3a7253ebd3b883", "kind": "source", "identity": SOURCE_IDENTITY},
        {"id": "model:OpenMOSS-Team/MOSS-Audio-4B-Instruct@6907a499dc0e87cc77c8ae0fe23fd0eb5476a02d", "kind": "model", "identity": VARIANTS["4b"]},
        {"id": "model:OpenMOSS-Team/MOSS-Audio-8B-Instruct@6521a39181b47a18f2d9f4b3acfb5bca7b76b57f", "kind": "model", "identity": VARIANTS["8b"]},
        {"id": "public-license:moss-audio-4b-instruct", "kind": "public_artifact", "identity": {"artifact": "moss-audio-4b-instruct", "license_spdx": None, "license_payload_sha256": None}},
        {"id": "public-license:moss-audio-8b-instruct", "kind": "public_artifact", "identity": {"artifact": "moss-audio-8b-instruct", "license_spdx": None, "license_payload_sha256": None}},
    ]
    return [dict(row, status="PENDING_REVIEW", license=None, payload_sha256=None, signer=None, approval_digest=None) for row in components]


def block(reason: str) -> tuple[bool, str]:
    return False, reason


def validate(project: Path, manifest_path: Path, evidence_path: Path | None = None, *, _self_test: bool = False) -> tuple[bool, str]:
    lock_path, pyproject_path = project / "uv.lock", project / "pyproject.toml"
    if any(path.is_symlink() or not path.is_file() for path in (lock_path, pyproject_path, manifest_path)):
        return block("lock, pyproject, or MOSS-Audio gate manifest is missing")
    try:
        manifest = load_json(manifest_path)
        lock_bytes, project_bytes = lock_path.read_bytes(), pyproject_path.read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return block(f"gate input is unreadable: {exc}")
    try:
        validate_project_schema(tomllib.loads(project_bytes.decode("utf-8")))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError) as exc:
        return block(f"pyproject.toml schema is invalid: {exc}")
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS or manifest.get("gate_version") != GATE_VERSION:
        return block("unsupported or malformed gate manifest")
    if sha256(lock_bytes) != LOCK_SHA256 or manifest.get("lock_sha256") != LOCK_SHA256:
        return block("uv.lock bytes are not the fixed reviewed lock")
    if sha256(project_bytes) != PYPROJECT_SHA256 or manifest.get("pyproject_sha256") != PYPROJECT_SHA256:
        return block("pyproject bytes are not the fixed reviewed project")
    try:
        rows = package_rows(tomllib.loads(lock_bytes.decode("utf-8")))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError) as exc:
        return block(f"uv.lock canonicalization failed: {exc}")
    if canonical(rows) != manifest.get("package_rows_sha256"):
        return block("canonical package rows drifted")
    reviews = manifest.get("dependency_reviews")
    review_fields = {"id", "name", "version", "source", "status", "license", "native_review", "bundled_review", "payload_sha256"}
    if not isinstance(reviews, list) or len(reviews) != len(rows):
        return block("dependency review rows are missing or incomplete")
    expected_keys = {(row["name"], row["version"], json.dumps(row["source"], sort_keys=True)) for row in rows}
    seen: set[tuple[str, str, str]] = set()
    for row in reviews:
        if not isinstance(row, dict) or set(row) != review_fields:
            return block("dependency review row schema is not exact")
        key = (row.get("name"), row.get("version"), json.dumps(row.get("source"), sort_keys=True))
        if key in seen or key not in expected_keys or row.get("id") != f"{row.get('name')}@{row.get('version')}":
            return block("dependency review identities are missing, extra, or duplicated")
        seen.add(key)
    if seen != expected_keys or canonical(reviews) != manifest.get("dependency_reviews_sha256"):
        return block("dependency reviews do not cover the exact lock")
    for row in reviews:
        if unresolved(row.get("status")) or row.get("status") != "REVIEWED":
            return block(f"dependency review is unresolved: {row.get('id')}")
        if not isinstance(row.get("license"), str) or unresolved(row.get("license")):
            return block(f"dependency license review is unresolved: {row.get('id')}")
        if not resolved_string(row.get("native_review")) or not resolved_string(row.get("bundled_review")):
            return block(f"dependency native/bundled review is unresolved: {row.get('id')}")
        if not isinstance(row.get("payload_sha256"), str) or not HEX64.fullmatch(row["payload_sha256"]):
            return block(f"dependency payload identity is unresolved: {row.get('id')}")
    fixed_identities = {"source": SOURCE_IDENTITY, "variants": VARIANTS, "common_assets": COMMON_ASSETS}
    components = manifest.get("component_reviews")
    expected = expected_components()
    component_fields = {"id", "kind", "identity", "status", "license", "payload_sha256", "signer", "approval_digest"}
    if not isinstance(components, list) or len(components) != len(expected):
        return block("model/source/license component rows are missing")
    for actual, fixed_component in zip(components, expected, strict=True):
        if not isinstance(actual, dict) or set(actual) != component_fields:
            return block("component review row schema is not exact")
        if {key: actual.get(key) for key in ("id", "kind", "identity")} != {key: fixed_component[key] for key in ("id", "kind", "identity")}:
            return block("fixed model/source/public identity drifted")
        if unresolved(actual.get("status")) or actual.get("status") != "REVIEWED":
            return block(f"component review is unresolved: {actual.get('id')}")
        if not isinstance(actual.get("license"), str) or unresolved(actual.get("license")):
            return block(f"component license is unresolved: {actual.get('id')}")
        fixed_spdx = fixed_component["identity"].get("license_spdx") if isinstance(fixed_component.get("identity"), dict) else None
        if resolved_string(fixed_spdx) and actual.get("license") != fixed_spdx:
            return block(f"component license does not match fixed SPDX: {actual.get('id')}")
        if not isinstance(actual.get("payload_sha256"), str) or not HEX64.fullmatch(actual["payload_sha256"]):
            return block(f"component payload identity is unresolved: {actual.get('id')}")
        if not resolved_string(actual.get("signer")):
            return block(f"component approval is unresolved: {actual.get('id')}")
        component_scope = {key: actual[key] for key in ("id", "kind", "identity", "status", "license", "payload_sha256")}
        if actual.get("approval_digest") != canonical(component_scope):
            return block(f"component approval is not bound: {actual.get('id')}")
    if manifest.get("fixed_identities") != fixed_identities or manifest.get("no_upload") != "NO_UPLOAD":
        return block("fixed MOSS-Audio identities or NO_UPLOAD policy drifted")
    if not _self_test:
        fixed_blocker = fixed_identity_blocker()
        if fixed_blocker:
            return block(fixed_blocker)
    scope = {"lock_sha256": LOCK_SHA256, "pyproject_sha256": PYPROJECT_SHA256,
             "package_rows_sha256": manifest["package_rows_sha256"],
             "dependency_reviews": reviews, "component_reviews": components,
             "fixed_identities": fixed_identities, "no_upload": "NO_UPLOAD"}
    scope_sha = canonical(scope)
    if manifest.get("approval_scope_sha256") != scope_sha:
        return block("approval scope is not bound to exact closure")
    approval = manifest.get("operator_approval")
    if (not isinstance(approval, dict) or set(approval) != APPROVAL_KEYS or approval.get("schema") != "v1"
            or approval.get("decision") != "APPROVED" or not resolved_string(approval.get("signer"))
            or approval.get("digest") != scope_sha):
        return block("operator approval is pending or invalid")
    evidence_path = evidence_path or manifest_path.with_name("license_gate_evidence.json")
    if evidence_path.is_symlink() or not evidence_path.is_file():
        return block("authenticated operator evidence is missing")
    try:
        evidence = load_json(evidence_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return block(f"operator evidence is unreadable: {exc}")
    if (not isinstance(evidence, dict) or set(evidence) != EVIDENCE_KEYS or evidence.get("schema") != "v1"
            or evidence.get("scope_sha256") != scope_sha
            or evidence.get("manifest_sha256") != sha256(manifest_path.read_bytes())
            or evidence.get("signer") != approval["signer"] or evidence.get("digest") != scope_sha
            or evidence.get("decision") != "APPROVED"):
        return block("operator evidence is not bound to this exact manifest and scope")
    return True, "PASS"


def self_test() -> int:
    project = Path(__file__).resolve().parent
    manifest_path = project / "license_gate_manifest.json"
    ok, reason = validate(project, manifest_path)
    if ok or ("unresolved" not in reason and "artifact" not in reason):
        print(f"moss_audio preflight gate: expected pending review, got {reason}", file=sys.stderr)
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
                print(f"moss_audio artifact tamper accepted: {label}", file=sys.stderr); return 1
        try:
            package_rows({"version": 1, "revision": 3, "requires-python": "==3.12.*", "resolution-markers": [], "package": [
                {"name": "demo", "version": "0", "source": {"virtual": "."}, "dependencies": [], "metadata": {"requires-dist": []}},
                {"name": "registry-demo", "version": "1", "source": {"registry": "https://pypi.org/simple"}, "dependencies": []},
            ]})
        except ValueError:
            pass
        else:
            print("moss_audio registry package without artifacts accepted", file=sys.stderr); return 1
        print("moss_audio preflight gate: self-test PASS (production artifact schema blocker)")
        return 0
    with tempfile.TemporaryDirectory(prefix="moss-audio-gate-") as directory:
        root, test_project = Path(directory), Path(directory) / "project"
        test_project.mkdir()
        mini = root / "mini-snapshot"
        mini.mkdir()
        mini_contract = [
            {"path": "a.bin", "role": "test", "bytes": 3, "sha256": sha256(b"abc"), "status": "REVIEWED"},
            {"path": "b.bin", "role": "test", "bytes": 0, "sha256": sha256(b""), "status": "REVIEWED"},
        ]
        (mini / "a.bin").write_bytes(b"abc")
        (mini / "b.bin").write_bytes(b"")
        (mini / ".cache").mkdir()
        if verify_snapshot_contract(mini, mini_contract) != (True, "PASS"):
            print("moss_audio snapshot resolved mini-contract baseline failed", file=sys.stderr)
            return 1
        (mini / "b.bin").unlink()
        if verify_snapshot_contract(mini, mini_contract)[0]:
            print("moss_audio snapshot missing-file tamper accepted", file=sys.stderr); return 1
        (mini / "b.bin").write_bytes(b"")
        (mini / "extra").write_bytes(b"x")
        if verify_snapshot_contract(mini, mini_contract)[0]:
            print("moss_audio snapshot extra-file tamper accepted", file=sys.stderr); return 1
        (mini / "extra").unlink()
        (mini / "b.bin").unlink()
        (mini / "b.bin").symlink_to("a.bin")
        if verify_snapshot_contract(mini, mini_contract)[0]:
            print("moss_audio snapshot symlink tamper accepted", file=sys.stderr); return 1
        (mini / "b.bin").unlink()
        (mini / "b.bin").write_bytes(b"x")
        if verify_snapshot_contract(mini, mini_contract)[0]:
            print("moss_audio snapshot hash tamper accepted", file=sys.stderr); return 1
        mini_source = root / "source"
        (mini_source / "src").mkdir(parents=True)
        source_payloads = {"src/a.py": b"alpha", "src/b.py": b"beta", "LICENSE": b"license"}
        for relative, payload in source_payloads.items():
            (mini_source / relative).write_bytes(payload)
        subprocess.run(["git", "-C", str(mini_source), "init", "-q"], check=True)
        subprocess.run(["git", "-C", str(mini_source), "config", "user.email", "self-test@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(mini_source), "config", "user.name", "self-test"], check=True)
        subprocess.run(["git", "-C", str(mini_source), "add", "."], check=True)
        subprocess.run(["git", "-C", str(mini_source), "commit", "-qm", "baseline"], check=True)
        mini_revision = subprocess.run(["git", "-C", str(mini_source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        mini_source_contract = []
        for relative in ("src/a.py", "src/b.py", "LICENSE"):
            payload = source_payloads[relative]
            blob = subprocess.run(["git", "-C", str(mini_source), "hash-object", relative], check=True, capture_output=True, text=True).stdout.strip()
            mini_source_contract.append({"path": relative, "role": "test", "bytes": len(payload), "sha256": sha256(payload), "git_blob_sha1": blob, "status": "REVIEWED"})
        if verify_source(mini_source, contract=mini_source_contract, expected_revision=mini_revision) != (True, "PASS"):
            print("moss_audio source resolved mini-contract baseline failed", file=sys.stderr); return 1
        (mini_source / "dirty.txt").write_text("dirty", encoding="utf-8")
        if verify_source(mini_source, contract=mini_source_contract, expected_revision=mini_revision)[0]:
            print("moss_audio source dirty tamper accepted", file=sys.stderr); return 1
        (mini_source / "dirty.txt").unlink()
        (mini_source / "untracked.txt").write_text("untracked", encoding="utf-8")
        if verify_source(mini_source, contract=mini_source_contract, expected_revision=mini_revision)[0]:
            print("moss_audio source untracked tamper accepted", file=sys.stderr); return 1
        (mini_source / "untracked.txt").unlink()
        source_link = root / "source-link"
        source_link.symlink_to(mini_source, target_is_directory=True)
        if verify_source(source_link, contract=mini_source_contract, expected_revision=mini_revision)[0]:
            print("moss_audio source root symlink accepted", file=sys.stderr); return 1
        source_link.unlink()
        original = (mini_source / "src/a.py").read_bytes()
        (mini_source / "src/a.py").unlink()
        (mini_source / "src/a.py").symlink_to("b.py")
        if verify_source(mini_source, contract=mini_source_contract, expected_revision=mini_revision)[0]:
            print("moss_audio source file symlink accepted", file=sys.stderr); return 1
        (mini_source / "src/a.py").unlink()
        (mini_source / "src/a.py").write_bytes(original)
        subprocess.run(["git", "-C", str(mini_source), "update-index", "--assume-unchanged", "src/a.py"], check=True)
        (mini_source / "src/a.py").write_bytes(b"tampered")
        if verify_source(mini_source, contract=mini_source_contract, expected_revision=mini_revision)[0]:
            print("moss_audio source hash tamper accepted", file=sys.stderr); return 1
        subprocess.run(["git", "-C", str(mini_source), "update-index", "--no-assume-unchanged", "src/a.py"], check=True)
        (mini_source / "src/a.py").write_bytes(original)
        shutil.copy2(project / "uv.lock", test_project / "uv.lock")
        shutil.copy2(project / "pyproject.toml", test_project / "pyproject.toml")
        base = json.loads(manifest_path.read_text(encoding="utf-8"))
        base["component_reviews"] = expected_components()
        for row in base["dependency_reviews"]:
            row.update(status="REVIEWED", license="SELF_TEST", native_review="SELF_TEST", bundled_review="SELF_TEST", payload_sha256="a" * 64)
        for row in base["component_reviews"]:
            row.update(status="REVIEWED", license="SELF_TEST", payload_sha256="b" * 64, signer="self-test")
            row["approval_digest"] = canonical({key: row[key] for key in ("id", "kind", "identity", "status", "license", "payload_sha256")})
        base["dependency_reviews_sha256"] = canonical(base["dependency_reviews"])
        scope = {"lock_sha256": LOCK_SHA256, "pyproject_sha256": PYPROJECT_SHA256, "package_rows_sha256": base["package_rows_sha256"], "dependency_reviews": base["dependency_reviews"], "component_reviews": base["component_reviews"], "fixed_identities": base["fixed_identities"], "no_upload": "NO_UPLOAD"}
        base["approval_scope_sha256"] = canonical(scope)
        base["operator_approval"] = {"schema": "v1", "decision": "APPROVED", "signer": "self-test", "digest": base["approval_scope_sha256"]}
        approved = root / "manifest.json"
        approved.write_text(json.dumps(base), encoding="utf-8")
        evidence = root / "evidence.json"
        evidence.write_text(json.dumps({"schema": "v1", "scope_sha256": base["approval_scope_sha256"], "manifest_sha256": sha256(approved.read_bytes()), "signer": "self-test", "digest": base["approval_scope_sha256"], "decision": "APPROVED"}), encoding="utf-8")
        good, reason = validate(test_project, approved, evidence, _self_test=True)
        if not good:
            print(f"moss_audio preflight gate: approved baseline failed: {reason}", file=sys.stderr)
            return 1
        duplicate_manifest = root / "duplicate-manifest.json"
        duplicate_manifest.write_text('{"gate_version":1,"gate_version":1}', encoding="utf-8")
        duplicate_ok, duplicate_reason = validate(test_project, duplicate_manifest, evidence, _self_test=True)
        if duplicate_ok or "duplicate JSON key" not in duplicate_reason:
            print("moss_audio duplicate manifest key was not rejected cleanly", file=sys.stderr)
            return 1
        duplicate_evidence = root / "duplicate-evidence.json"
        duplicate_evidence.write_text('{"schema":"v1","schema":"v1"}', encoding="utf-8")
        duplicate_ok, duplicate_reason = validate(test_project, approved, duplicate_evidence, _self_test=True)
        if duplicate_ok or "duplicate JSON key" not in duplicate_reason:
            print("moss_audio duplicate evidence key was not rejected cleanly", file=sys.stderr)
            return 1
        for identity_path in (
            ("source", "license", "payload_sha256"),
            ("variants", "4b", "license", "payload_sha256"),
            ("variants", "8b", "license", "payload_sha256"),
        ):
            candidate = json.loads(approved.read_text(encoding="utf-8"))
            target = candidate["fixed_identities"]
            for key in identity_path[:-1]:
                target = target[key]
            target[identity_path[-1]] = "x"
            approved.write_text(json.dumps(candidate), encoding="utf-8")
            if validate(test_project, approved, evidence, _self_test=True)[0]:
                print("moss_audio fixed license hash tamper accepted", file=sys.stderr); return 1
            approved.write_text(json.dumps(base), encoding="utf-8")
        candidate = json.loads(approved.read_text(encoding="utf-8"))
        candidate["fixed_identities"]["source"]["license"]["blob_sha256"] = "x"
        approved.write_text(json.dumps(candidate), encoding="utf-8")
        if validate(test_project, approved, evidence, _self_test=True)[0]:
            print("moss_audio fixed source license blob tamper accepted", file=sys.stderr); return 1
        approved.write_text(json.dumps(base), encoding="utf-8")
        candidate = json.loads(approved.read_text(encoding="utf-8"))
        candidate["fixed_identities"]["source"]["license"]["payload_sha256"] = "x"
        approved.write_text(json.dumps(candidate), encoding="utf-8")
        if validate(test_project, approved, evidence, _self_test=True)[0]:
            print("moss_audio fixed source license payload tamper accepted", file=sys.stderr); return 1
        approved.write_text(json.dumps(base), encoding="utf-8")
        candidate = json.loads(approved.read_text(encoding="utf-8"))
        candidate["dependency_reviews"][0]["native_review"] = False
        candidate["dependency_reviews_sha256"] = canonical(candidate["dependency_reviews"])
        approved.write_text(json.dumps(candidate), encoding="utf-8")
        if validate(test_project, approved, evidence, _self_test=True)[0]:
            print("moss_audio boolean native review accepted", file=sys.stderr); return 1
        approved.write_text(json.dumps(base), encoding="utf-8")
        candidate = json.loads(approved.read_text(encoding="utf-8"))
        candidate["component_reviews"][0]["signer"] = False
        approved.write_text(json.dumps(candidate), encoding="utf-8")
        if validate(test_project, approved, evidence, _self_test=True)[0]:
            print("moss_audio boolean component signer accepted", file=sys.stderr); return 1
        approved.write_text(json.dumps(base), encoding="utf-8")
        good, reason = validate(test_project, approved, evidence)
        if good or "identity is incomplete" not in reason:
            print("moss-audio incomplete fixed identity was accepted", file=sys.stderr)
            return 1
        for tamper in ("lock_sha256", "pyproject_sha256"):
            candidate = json.loads(approved.read_text(encoding="utf-8")); candidate[tamper] = "0" * 64
            approved.write_text(json.dumps(candidate), encoding="utf-8")
            if validate(test_project, approved, evidence, _self_test=True)[0]:
                print(f"moss_audio preflight gate: {tamper} tamper accepted", file=sys.stderr); return 1
            approved.write_text(json.dumps(base), encoding="utf-8")
        for mutate in (("component_reviews", 1, "identity", "revision", "0" * 40), ("dependency_reviews", 0, "license", None, "TODO")):
            candidate = json.loads(approved.read_text(encoding="utf-8"))
            if mutate[3] is None:
                candidate[mutate[0]][mutate[1]][mutate[2]] = mutate[4]
            else:
                candidate[mutate[0]][mutate[1]][mutate[2]][mutate[3]] = mutate[4]
            approved.write_text(json.dumps(candidate), encoding="utf-8")
            if validate(test_project, approved, evidence, _self_test=True)[0]:
                print("moss_audio preflight gate: identity/review tamper accepted", file=sys.stderr); return 1
            approved.write_text(json.dumps(base), encoding="utf-8")
    print("moss_audio preflight gate: self-test PASS")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--verify-snapshot", action="store_true")
    parser.add_argument("--verify-source", action="store_true")
    parser.add_argument("--license-spdx", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--variant", choices=sorted(VARIANTS))
    args = parser.parse_args()
    if args.self_test:
        raise SystemExit(self_test())
    if args.verify_snapshot:
        if args.snapshot is None or args.variant is None:
            parser.error("--verify-snapshot requires --snapshot and --variant")
        passed, reason = verify_snapshot(args.snapshot, args.variant)
        if not passed:
            print(f"moss_audio snapshot gate: BLOCKED: {reason}", file=sys.stderr)
            raise SystemExit(2)
        print("moss_audio snapshot gate: PASS")
        raise SystemExit(0)
    if args.verify_source:
        if args.source is None:
            parser.error("--verify-source requires --source")
        passed, reason = verify_source(args.source)
        if not passed:
            print(f"moss_audio source gate: BLOCKED: {reason}", file=sys.stderr)
            raise SystemExit(2)
        print("moss_audio source gate: PASS")
        raise SystemExit(0)
    if args.license_spdx:
        if args.variant not in VARIANTS:
            parser.error("--license-spdx requires --variant")
        value = VARIANTS[args.variant].get("license_spdx")
        if not resolved_string(value):
            print("moss_audio license gate: BLOCKED: fixed model license SPDX is incomplete", file=sys.stderr)
            raise SystemExit(2)
        print(value)
        raise SystemExit(0)
    if args.project is None or args.manifest is None:
        parser.error("--project and --manifest are required")
    passed, reason = validate(args.project, args.manifest, args.evidence)
    if not passed:
        print(f"moss_audio preflight gate: BLOCKED: {reason}", file=sys.stderr)
        raise SystemExit(2)
    print("moss_audio preflight gate: PASS")
