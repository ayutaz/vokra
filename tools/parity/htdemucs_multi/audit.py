#!/usr/bin/env -S uv run --no-project --python 3.12 python
"""Fail-closed HT-Demucs source, release, and dependency audit.

This module is stdlib-only by design.  It can run before the dedicated uv
environment is resolved.  A resolved lock is evidence of reproducibility;
the primary package license rows and owner approval are separate gates.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import tomllib
import re
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).parent
REPO_ROOT = PROJECT.parents[2]
GATE = PROJECT / "license_gate_manifest.json"
LOCK = PROJECT / "uv.lock"
UPSTREAM_URL = "https://github.com/facebookresearch/demucs"
UPSTREAM_REVISION = "e976d93ecc3865e5757426930257e200846a520a"
WEIGHT_IDS = ("f7e0c4bc", "d12395a8", "92cfc3b6", "04573f0d", "5c90dfd2")
WEIGHT_ROOT = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/"
SOURCE_ROLES = {"LICENSE", "demucs/apply.py", "demucs/audio.py", "demucs/hdemucs.py", "demucs/htdemucs.py", "demucs/pretrained.py", "demucs/repo.py", "demucs/states.py", "demucs/remote/htdemucs_ft.yaml", "demucs/remote/htdemucs_6s.yaml"}
TARGET_ENV = {
    "python_full_version": "3.12.0", "python_version": "3.12", "sys_platform": "linux",
    "platform_machine": "x86_64", "platform_system": "Linux", "implementation_name": "cpython",
}
TARGET_RESOLUTION_MARKERS = {
    "platform_machine == 'x86_64' and sys_platform == 'linux'",
    "python_full_version >= '3.12' and python_full_version < '3.13' and platform_machine == 'x86_64' and sys_platform == 'linux'",
}
UPSTREAM_REQUIREMENTS_FILE = "upstream_requirements_minimal.snapshot"
ACTIVE_IMPORT_PACKAGES = {
    "dora-search", "einops", "julius", "numpy", "openunmix", "pyyaml",
    "torch", "torchaudio", "tqdm",
}
UPSTREAM_REQUIREMENTS_PACKAGES = ACTIVE_IMPORT_PACKAGES - {"numpy"} | {"lameenc"}


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def load_gate() -> dict[str, Any]:
    if not GATE.is_file() or GATE.is_symlink():
        raise ValueError("HT-Demucs license gate manifest is missing or symlinked")
    value = json.loads(GATE.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)
    if not isinstance(value, dict):
        raise ValueError("license gate manifest must be an object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    """Hash the canonical JSON bytes used by the primary audit rows."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def marker_value(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in TARGET_ENV:
        return TARGET_ENV[node.id]
    raise ValueError("unsupported or unknown environment marker")


def marker_eval(node: ast.AST) -> bool:
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        values = [marker_eval(item) for item in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
        left, right = marker_value(node.left), marker_value(node.comparators[0])
        op = node.ops[0]
        if isinstance(op, ast.Eq): return left == right
        if isinstance(op, ast.NotEq): return left != right
        if isinstance(op, ast.Gt): return left > right
        if isinstance(op, ast.GtE): return left >= right
        if isinstance(op, ast.Lt): return left < right
        if isinstance(op, ast.LtE): return left <= right
    raise ValueError("unsupported dependency marker expression")


def marker_reaches(marker: str) -> bool:
    try:
        return marker_eval(ast.parse(marker, mode="eval").body)
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"dependency marker cannot be evaluated: {marker}") from error


def marker_reaches_any(markers: list[str]) -> bool:
    return not markers or any(marker_reaches(marker) for marker in markers)


def validate_inactive_rows(rows: Any) -> None:
    if not isinstance(rows, list):
        raise ValueError("inactive dependency rows must be arrays")
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {"name", "version", "reason", "markers"}
                or not all(isinstance(row[key], str) and row[key] for key in ("name", "version", "reason", "markers"))):
            raise ValueError("inactive dependency row schema drifted")


def parse_lock_data(lock: dict[str, Any], project_data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Parse the lock document header and package identities for production."""
    if set(lock) != {"version", "revision", "requires-python", "resolution-markers", "supported-markers", "package"}:
        raise ValueError("uv.lock top-level schema/version drifted")
    if lock["version"] != 1 or lock["revision"] != 3 or lock["requires-python"] != "==3.12.*":
        raise ValueError("uv.lock top-level identity drifted")
    markers = lock["resolution-markers"]
    if not isinstance(markers, list) or len(markers) != 1 or markers[0] not in TARGET_RESOLUTION_MARKERS or lock["supported-markers"] != markers:
        raise ValueError("uv.lock marker contract drifted")
    seen: set[tuple[str, str]] = set()
    packages = lock["package"]
    if not isinstance(packages, list) or not packages:
        raise ValueError("uv.lock has no package records")
    for package in packages:
        if not isinstance(package, dict) or not isinstance(package.get("name"), str) or not isinstance(package.get("version"), str):
            raise ValueError("uv.lock package identity drifted")
        identity = (package["name"].lower(), package["version"])
        if identity in seen:
            raise ValueError("uv.lock contains duplicate (name, version)")
        seen.add(identity)
        source = package.get("source")
        if source == {"registry": "https://registry.invalid"} or not isinstance(source, dict):
            raise ValueError("uv.lock package source drifted")
        cpu = package["name"].lower() in {"torch", "torchaudio"} and package.get("source", {}).get("registry", "").rstrip("/") == "https://download.pytorch.org/whl/cpu"
        if package.get("source", {}).get("virtual") == "." and "metadata" not in package:
            raise ValueError("uv.lock virtual project metadata missing")
        if package.get("source", {}).get("virtual") != "." and "metadata" in package:
            raise ValueError("uv.lock metadata is only valid on the virtual project")
        for artifact in [package.get("sdist")] + list(package.get("wheels", [])):
            if artifact is None:
                continue
            required = {"url", "hash", "upload-time"} | (set() if cpu else {"size"})
            allowed_keys = {frozenset(required), frozenset(required | {"size"})} if cpu else {frozenset(required)}
            if set(artifact) not in allowed_keys:
                raise ValueError("uv.lock artifact size/schema drifted")
            allowed_origin = ((artifact["url"].startswith("https://download.pytorch.org/whl/cpu/") or artifact["url"].startswith("https://download-r2.pytorch.org/whl/cpu/")) if cpu else artifact["url"].startswith("https://files.pythonhosted.org/"))
            if not allowed_origin:
                raise ValueError("uv.lock artifact origin drifted")
            if not isinstance(artifact["upload-time"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", artifact["hash"]):
                raise ValueError("uv.lock artifact digest metadata drifted")
            if "size" in artifact and (not isinstance(artifact["size"], int) or artifact["size"] <= 0):
                raise ValueError("uv.lock artifact size drifted")
    return {"packages": packages, "resolution_markers": markers, "supported_markers": lock["supported-markers"]}


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def verify_gate_contract(gate: dict[str, Any]) -> None:
    if set(gate) != {"schema", "status", "publication", "upstream", "weights", "dataset", "dependency_audit", "blockers"}:
        raise ValueError("license gate top-level schema drifted")
    if not isinstance(gate["status"], str) or gate["publication"] != "NO_UPLOAD":
        raise ValueError("license gate status/publication schema drifted")
    if gate.get("schema") != "vokra-htdemucs-multi-license-gate-v1":
        raise ValueError("license gate schema drifted")
    upstream = gate.get("upstream")
    if not isinstance(upstream, dict) or upstream.get("repository") != UPSTREAM_URL:
        raise ValueError("upstream repository identity drifted")
    if upstream.get("revision") != UPSTREAM_REVISION:
        raise ValueError("upstream revision drifted")
    license_row = upstream.get("license")
    if (not isinstance(license_row, dict) or set(license_row) != {"spdx", "path", "bytes", "sha256", "git_blob_sha1"}
            or license_row.get("spdx") != "MIT" or not isinstance(license_row.get("path"), str)
            or not isinstance(license_row.get("bytes"), int) or isinstance(license_row.get("bytes"), bool)
            or license_row["bytes"] <= 0 or not re.fullmatch(r"[0-9a-f]{64}", license_row.get("sha256", ""))
            or not re.fullmatch(r"[0-9a-f]{40}", license_row.get("git_blob_sha1", ""))):
        raise ValueError("upstream license row drifted")
    configs = upstream.get("config_sha256")
    if not isinstance(configs, dict) or set(configs) != {"htdemucs_ft.yaml", "htdemucs_6s.yaml"}:
        raise ValueError("config identity set drifted")
    for value in configs.values():
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("config SHA-256 is not a lowercase 64-hex value")
    roles = upstream.get("roles")
    if not isinstance(roles, dict) or set(roles) != SOURCE_ROLES:
        raise ValueError("source role schema drifted")
    for role, blob in roles.items():
        if Path(role).is_absolute() or ".." in Path(role).parts or not isinstance(blob, str) or len(blob) != 40 or any(c not in "0123456789abcdef" for c in blob):
            raise ValueError("source role identity is malformed")
    weights = gate.get("weights")
    if not isinstance(weights, list) or len(weights) != len(WEIGHT_IDS):
        raise ValueError("weight identity count drifted")
    seen: set[str] = set()
    for row in weights:
        if not isinstance(row, dict) or set(row) != {"model_id", "filename", "url", "sha256", "tensor_count", "parameter_count"}:
            raise ValueError("weight identity schema drifted")
        model_id = row["model_id"]
        if model_id not in WEIGHT_IDS or model_id in seen:
            raise ValueError("weight member set/order drifted")
        seen.add(model_id)
        digest = row["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"invalid full weight SHA-256 for {model_id}")
        filename = row["filename"]
        if not isinstance(filename, str) or Path(filename).name != filename or not filename.startswith(model_id + "-"):
            raise ValueError(f"weight filename schema drifted for {model_id}")
        if filename.removesuffix(".th").split("-")[-1] != digest[:8]:
            raise ValueError(f"weight filename digest prefix drifted for {model_id}")
        if row["url"] != WEIGHT_ROOT + filename:
            raise ValueError(f"weight URL schema drifted for {model_id}")
        if (not isinstance(row["tensor_count"], int) or isinstance(row["tensor_count"], bool)
                or not isinstance(row["parameter_count"], int) or isinstance(row["parameter_count"], bool)
                or row["tensor_count"] <= 0 or row["parameter_count"] <= 0):
            raise ValueError(f"invalid inventory counts for {model_id}")
    if tuple(row["model_id"] for row in weights) != WEIGHT_IDS:
        raise ValueError("weight member order is not the authenticated order")
    dataset = gate["dataset"]
    if (set(dataset) != {"name", "provenance_status", "license_status", "source"}
            or dataset.get("name") != "MUSDB18"
            or dataset.get("provenance_status") not in {"AUTHENTICATED", "UNAUTHENTICATED"}
            or dataset.get("license_status") not in {"APPROVED", "UNREVIEWED"}
            or not isinstance(dataset["source"], str) or not dataset["source"].startswith("https://")):
        raise ValueError("dataset provenance schema drifted")
    if not isinstance(gate["blockers"], list) or not gate["blockers"] or not all(isinstance(item, str) and item for item in gate["blockers"]):
        raise ValueError("license gate blockers must be explicit strings")
    dependency = gate.get("dependency_audit")
    if (not isinstance(dependency, dict)
            or dependency.get("source_file") != UPSTREAM_REQUIREMENTS_FILE
            or dependency.get("rows_file") != "dependency_audit.json"):
        raise ValueError("dependency audit contract drifted")
    for key in ("source_file_sha256", "pyproject_sha256", "rows_file_sha256"):
        value = dependency.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"dependency {key} is not a fixed lowercase 64-hex digest")
    active = dependency.get("active_import_closure")
    if (not isinstance(active, dict)
            or active.get("path") != "pyproject.toml"
            or active.get("packages") != sorted(ACTIVE_IMPORT_PACKAGES)
            or active.get("excluded_upstream_packages") != ["lameenc"]):
        raise ValueError("active import closure contract drifted")
    if (not isinstance(active.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", active["sha256"])):
        raise ValueError("active import closure digest is not fixed")


def audit_source(source: Path, gate: dict[str, Any]) -> dict[str, Any]:
    if not source.is_absolute() or source.resolve(strict=False) != source or source.is_symlink() or not source.is_dir():
        raise ValueError("source checkout must be an absolute canonical directory")
    if not (source / ".git").exists():
        raise ValueError("source checkout lacks .git metadata")
    if git(source, "rev-parse", "HEAD") != UPSTREAM_REVISION:
        raise ValueError("source HEAD does not match the authenticated revision")
    origin = git(source, "remote", "get-url", "origin").removesuffix("/").removesuffix(".git")
    if origin != UPSTREAM_URL:
        raise ValueError("source origin does not match the authenticated repository")
    if git(source, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("source checkout is dirty")
    upstream = gate["upstream"]
    roles = upstream["roles"]
    for role, expected_blob in roles.items():
        path = source / role
        if not path.is_file() or path.is_symlink() or path.resolve(strict=True) != path:
            raise ValueError(f"source role missing or symlinked: {role}")
        actual_blob = git(source, "rev-parse", f"HEAD:{role}")
        if actual_blob != expected_blob:
            raise ValueError(f"source role blob drifted: {role}")
    license_path = source / upstream["license"]["path"]
    if license_path.stat().st_size != upstream["license"]["bytes"]:
        raise ValueError("source LICENSE byte count drifted")
    if sha256(license_path) != upstream["license"]["sha256"]:
        raise ValueError("source LICENSE SHA-256 drifted")
    for config_name, expected in upstream["config_sha256"].items():
        path = source / "demucs" / "remote" / config_name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"source config identity drifted: {config_name}")
    return {"repository": UPSTREAM_URL, "revision": UPSTREAM_REVISION, "origin": origin, "dirty": False, "role_blobs": roles, "status": "CLEAN"}


def audit_dependency_rows(gate: dict[str, Any]) -> dict[str, Any]:
    dependency = gate["dependency_audit"]
    requirements = PROJECT / dependency["source_file"]
    if not requirements.is_file() or requirements.is_symlink() or sha256(requirements) != dependency["source_file_sha256"]:
        raise ValueError("upstream requirements_minimal snapshot identity drifted")
    direct_lines = [line.strip() for line in requirements.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    direct_names = {re.split(r"[<>=!~;\s]", line, maxsplit=1)[0].lower() for line in direct_lines}
    if direct_names != UPSTREAM_REQUIREMENTS_PACKAGES or len(direct_names) != len(direct_lines):
        raise ValueError("upstream direct requirement set drifted")
    pyproject = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject.get("project", {}).get("dependencies")
    if not isinstance(dependencies, list):
        raise ValueError("dedicated pyproject dependency schema drifted")
    pyproject_names = {re.split(r"[<>=!~;\s]", item.strip(), maxsplit=1)[0].lower() for item in dependencies if isinstance(item, str)}
    if pyproject_names != ACTIVE_IMPORT_PACKAGES or len(pyproject_names) != len(dependencies):
        raise ValueError("pyproject direct/import requirement distinction drifted")
    active = dependency["active_import_closure"]
    if active["packages"] != sorted(pyproject_names) or active["excluded_upstream_packages"] != ["lameenc"]:
        raise ValueError("active import closure package set drifted")
    if sha256(PROJECT / "pyproject.toml") != dependency["pyproject_sha256"] or sha256(PROJECT / "pyproject.toml") != active["sha256"]:
        raise ValueError("dedicated pyproject.toml identity drifted")
    rows_path = PROJECT / dependency["rows_file"]
    if not rows_path.is_file() or rows_path.is_symlink() or sha256(rows_path) != dependency["rows_file_sha256"]:
        raise ValueError("dependency audit row file identity drifted")
    rows = json.loads(rows_path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)
    expected_keys = {"schema", "status", "python", "platform", "direct_requirements_source", "active_import_closure", "compatibility", "package_rows", "license_rows", "inactive_package_rows", "inactive_license_rows", "package_rows_sha256", "license_rows_sha256", "forbidden_license_policy", "blockers"}
    if set(rows) != expected_keys or rows["schema"] != "vokra-htdemucs-multi-dependency-audit-v1":
        raise ValueError("dependency audit row schema drifted")
    direct = rows["direct_requirements_source"]
    if direct != {"path": dependency["source_file"], "sha256": dependency["source_file_sha256"]}:
        raise ValueError("dependency direct-requirements identity drifted")
    active_row = rows["active_import_closure"]
    if active_row != {"path": "pyproject.toml", "sha256": dependency["active_import_closure"]["sha256"], "packages": sorted(ACTIVE_IMPORT_PACKAGES), "excluded_upstream_packages": ["lameenc"]}:
        raise ValueError("dependency active-import identity drifted")
    if not isinstance(rows["package_rows"], list) or not isinstance(rows["license_rows"], list):
        raise ValueError("dependency audit rows must be arrays")
    validate_inactive_rows(rows["inactive_package_rows"])
    validate_inactive_rows(rows["inactive_license_rows"])
    for key, values in (("package_rows_sha256", rows["package_rows"]), ("license_rows_sha256", rows["license_rows"])):
        digest = rows[key]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"dependency {key} is not a fixed lowercase 64-hex digest")
        if digest != json_sha256(values):
            raise ValueError(f"dependency {key} does not match canonical row content")
    if not rows["package_rows"] or not rows["license_rows"]:
        raise ValueError("dependency package/license rows are incomplete")
    policy = rows["forbidden_license_policy"]
    if not isinstance(policy, dict) or policy.get("status") != "FAIL_CLOSED":
        raise ValueError("forbidden-license policy is not fail-closed")
    license_versions: list[tuple[str, str]] = []
    for row in rows["license_rows"]:
        if not isinstance(row, dict) or not isinstance(row.get("license"), str) or row.get("status") != "APPROVED":
            raise ValueError("dependency license row is not approved")
        if any(token in row["license"].upper() for token in ("GPL", "LGPL", "UNKNOWN")):
            raise ValueError(f"forbidden dependency license: {row.get('name')}")
    package_names: list[str] = []
    for row in rows["package_rows"]:
        if not isinstance(row, dict) or set(row) != {"name", "version", "artifact", "license"}:
            raise ValueError("dependency package row schema drifted")
        if not all(isinstance(row[key], str) and row[key] for key in ("name", "version", "license")):
            raise ValueError("dependency package row has invalid text")
        artifact = row["artifact"]
        if not isinstance(artifact, dict) or set(artifact) != {"kind", "url", "sha256", "bytes"}:
            raise ValueError("dependency package artifact schema drifted")
        if artifact["kind"] not in {"locked_sdist", "locked_wheel", "virtual-local"} or not isinstance(artifact["url"], str) or not artifact["url"]:
            raise ValueError("dependency package artifact identity is invalid")
        digest = artifact["sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("dependency package artifact SHA-256 is not fixed")
        if not isinstance(artifact["bytes"], int) or isinstance(artifact["bytes"], bool) or artifact["bytes"] <= 0:
            raise ValueError("dependency package artifact byte count is invalid")
        package_names.append(row["name"].lower())
    license_names: list[str] = []
    for row in rows["license_rows"]:
        if set(row) != {"name", "version", "license", "status", "source", "sha256", "evidence"}:
            raise ValueError("dependency license row schema drifted")
        if not all(isinstance(row[key], str) and row[key] for key in ("name", "version", "license", "status", "source")):
            raise ValueError("dependency license row has invalid text")
        digest = row["sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("dependency license SHA-256 is not fixed")
        evidence = row["evidence"]
        if not isinstance(evidence, dict) or set(evidence) != {"kind", "artifact_kind", "artifact_url", "artifact_sha256", "artifact_bytes", "license_path", "license_bytes", "license_sha256"}:
            raise ValueError("dependency license evidence schema drifted")
        if evidence["kind"] not in {"locked_sdist", "publisher_bytes", "local_file"}:
            raise ValueError("approved license requires exact locked artifact evidence")
        if (evidence["artifact_kind"] not in {"locked_sdist", "locked_wheel", "virtual-local"}
                or not isinstance(evidence["artifact_url"], str) or not evidence["artifact_url"]
                or not isinstance(evidence["artifact_sha256"], str) or len(evidence["artifact_sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in evidence["artifact_sha256"])
                or not isinstance(evidence["artifact_bytes"], int) or isinstance(evidence["artifact_bytes"], bool) or evidence["artifact_bytes"] <= 0
                or not isinstance(evidence["license_path"], str) or not evidence["license_path"]
                or not isinstance(evidence["license_bytes"], int) or isinstance(evidence["license_bytes"], bool)
                or evidence["license_bytes"] <= 0 or not isinstance(evidence["license_sha256"], str)
                or len(evidence["license_sha256"]) != 64):
            raise ValueError("dependency license evidence is not primary-byte bound")
        if row["sha256"] != evidence["license_sha256"]:
            raise ValueError("dependency license digest is not bound to evidence bytes")
        license_names.append(row["name"].lower())
        license_versions.append((row["name"].lower(), row["version"]))
    package_versions = [(row["name"].lower(), row["version"]) for row in rows["package_rows"]]
    if len(set(package_versions)) != len(package_versions) or len(set(license_versions)) != len(license_versions):
        raise ValueError("dependency package/license rows contain duplicate names")
    if set(package_versions) != set(license_versions):
        raise ValueError("dependency package/license row names do not match")
    return {"package_rows": len(rows["package_rows"]), "license_rows": len(rows["license_rows"]), "status": "DEPENDENCY_ROWS_OK"}


def audit_lock(gate: dict[str, Any]) -> dict[str, Any]:
    dependency = gate["dependency_audit"]
    if not LOCK.is_file() or LOCK.is_symlink():
        raise ValueError("dedicated uv.lock is absent; generate it on VAST from this pyproject")
    try:
        lock = tomllib.loads(LOCK.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"uv.lock is not valid TOML: {error}") from error
    parsed = parse_lock_data(lock)
    packages = parsed["packages"]
    package_keys = {"name", "version", "source", "resolution-markers", "dependencies", "sdist", "wheels", "metadata"}
    lock_rows: dict[tuple[str, str], dict[str, Any]] = {}
    artifact_hashes: dict[tuple[str, str], set[str]] = {}
    artifact_records: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for package in packages:
        if not isinstance(package, dict) or not set(package).issubset(package_keys):
            raise ValueError("uv.lock package schema drifted")
        if not isinstance(package.get("name"), str) or not isinstance(package.get("version"), str):
            raise ValueError("uv.lock package name/version is invalid")
        identity = (package["name"].lower(), package["version"])
        if identity in lock_rows:
            raise ValueError("uv.lock contains duplicate (name, version)")
        source = package.get("source")
        if not isinstance(source, dict) or len(source) != 1 or not next(iter(source)) in {"virtual", "registry"}:
            raise ValueError("uv.lock package source schema is unsupported")
        if "registry" in source:
            expected_registry = "https://download.pytorch.org/whl/cpu" if package["name"].lower() in {"torch", "torchaudio"} else "https://pypi.org/simple"
            if source["registry"].rstrip("/") != expected_registry.rstrip("/"):
                raise ValueError("uv.lock registry does not match the audited index")
        if any(token in json.dumps(package, sort_keys=True).lower() for token in ("cuda", "nvidia", "triton")):
            raise ValueError("CUDA/NVIDIA/Triton package or source is forbidden")
        markers = package.get("resolution-markers", [])
        if not isinstance(markers, list) or not all(isinstance(marker, str) for marker in markers):
            raise ValueError("uv.lock resolution markers are invalid")
        dependencies = package.get("dependencies", [])
        if not isinstance(dependencies, list) or not all(isinstance(dep, dict) and isinstance(dep.get("name"), str) for dep in dependencies):
            raise ValueError("uv.lock dependency rows are invalid")
        if source == {"virtual": "."}:
            metadata = package.get("metadata")
            if not isinstance(metadata, dict) or set(metadata) != {"requires-dist"} or not isinstance(metadata["requires-dist"], list):
                raise ValueError("uv.lock virtual project metadata schema drifted")
            for requirement in metadata["requires-dist"]:
                if (not isinstance(requirement, dict) or not set(requirement).issubset({"name", "specifier", "index"})
                        or not isinstance(requirement.get("name"), str)
                        or ("specifier" in requirement and not isinstance(requirement["specifier"], str))):
                    raise ValueError("uv.lock virtual requires-dist schema drifted")
        elif "metadata" in package:
            raise ValueError("uv.lock metadata is only valid on the virtual project")
        hashes: set[str] = set()
        records: list[dict[str, Any]] = []
        for artifact_key in ("sdist", "wheels"):
            artifacts = package.get(artifact_key, [] if artifact_key == "wheels" else None)
            if artifacts is None:
                continue
            if artifact_key == "sdist":
                artifacts = [artifacts]
            if not isinstance(artifacts, list):
                raise ValueError("uv.lock artifact schema is invalid")
            for artifact in artifacts:
                match = re.fullmatch(r"sha256:([0-9a-f]{64})", artifact["hash"])
                if match is None:
                    raise ValueError("uv.lock artifact hash is not a fixed SHA-256")
                hashes.add(match.group(1))
                records.append({"kind": "locked_sdist" if artifact_key == "sdist" else "locked_wheel", "url": artifact["url"], "sha256": match.group(1), "bytes": artifact.get("size")})
        lock_rows[identity] = package
        artifact_hashes[identity] = hashes
        artifact_records[identity] = records
    project = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
    project_name = project.get("project", {}).get("name")
    project_rows = [identity for identity in lock_rows if identity[0] == str(project_name).lower()]
    if len(project_rows) != 1:
        raise ValueError("uv.lock virtual project identity is missing or duplicated")
    virtual_metadata = lock_rows[project_rows[0]].get("metadata", {}).get("requires-dist", [])
    expected_metadata = {}
    for item in project.get("project", {}).get("dependencies", []):
        if isinstance(item, str):
            match = re.match(r"^([A-Za-z0-9_.-]+)(.*)$", item.strip())
            if match:
                expected_metadata[match.group(1).lower()] = match.group(2)
    expected_metadata_names = set(expected_metadata)
    if {item["name"].lower() for item in virtual_metadata} != expected_metadata_names:
        raise ValueError("uv.lock virtual requires-dist dependency contract drifted")
    for item in virtual_metadata:
        specifier = item.get("specifier", "")
        if specifier != expected_metadata[item["name"].lower()]:
            raise ValueError("uv.lock virtual requires-dist specifier drifted")
        if item["name"].lower() in {"torch", "torchaudio"}:
            if item.get("index", "").rstrip("/") != "https://download.pytorch.org/whl/cpu":
                raise ValueError("uv.lock virtual Torch index drifted")
        elif "index" in item:
            raise ValueError("unexpected custom index in virtual requires-dist")
    active = {project_rows[0]}
    queue = [project_rows[0]]
    while queue:
        identity = queue.pop()
        for dep in lock_rows[identity].get("dependencies", []):
            marker = dep.get("marker")
            if marker is not None and not marker_reaches(marker):
                continue
            candidates = [item for item in lock_rows if item[0] == dep["name"].lower()]
            reachable = [item for item in candidates if marker_reaches_any(lock_rows[item].get("resolution-markers", []))]
            if len(reachable) != 1:
                raise ValueError(f"dependency closure is ambiguous or missing: {dep['name']}")
            if reachable[0] not in active:
                active.add(reachable[0]); queue.append(reachable[0])
    rows = audit_dependency_rows(gate)
    audit = json.loads((PROJECT / gate["dependency_audit"]["rows_file"]).read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)
    if {(row["name"].lower(), row["version"]) for row in audit["package_rows"]} != active:
        raise ValueError("package audit rows do not exactly match reachable lock closure")
    inactive = {(row["name"].lower(), row["version"]) for row in audit["inactive_package_rows"] + audit["inactive_license_rows"]}
    if inactive & active:
        raise ValueError("inactive dependency row is reachable on the target")
    inactive_package = {(row["name"].lower(), row["version"]) for row in audit["inactive_package_rows"]}
    inactive_license = {(row["name"].lower(), row["version"]) for row in audit["inactive_license_rows"]}
    if inactive_package != inactive_license or inactive_package != set(lock_rows) - active:
        raise ValueError("inactive dependency rows do not exactly cover the lock")
    license_by_identity = {(row["name"].lower(), row["version"]): row for row in audit["license_rows"]}
    for row in audit["package_rows"]:
        identity = (row["name"].lower(), row["version"])
        artifact = row["artifact"]
        if artifact["kind"] == "virtual-local":
            if lock_rows[identity]["source"] != {"virtual": "."} or artifact["url"] != "pyproject.toml" or artifact["sha256"] != sha256(PROJECT / "pyproject.toml") or artifact["bytes"] != (PROJECT / "pyproject.toml").stat().st_size:
                raise ValueError("virtual package artifact is not bound to pyproject bytes")
        if artifact["kind"] != "virtual-local" and artifact["sha256"] not in artifact_hashes[identity]:
            raise ValueError("package artifact digest is not bound to uv.lock")
        if artifact["kind"] != "virtual-local":
            matching_lock_artifacts = [item for item in artifact_records[identity] if item["kind"] == artifact["kind"] and item["url"] == artifact["url"] and item["sha256"] == artifact["sha256"]]
            if len(matching_lock_artifacts) != 1 or (matching_lock_artifacts[0]["bytes"] is not None and matching_lock_artifacts[0]["bytes"] != artifact["bytes"]):
                raise ValueError("selected package artifact URL/hash/bytes is not bound to uv.lock")
        evidence = license_by_identity[identity]["evidence"]
        if evidence["artifact_kind"] != artifact["kind"] or evidence["artifact_url"] != artifact["url"] or evidence["artifact_sha256"] != artifact["sha256"] or evidence["artifact_bytes"] != artifact["bytes"]:
            raise ValueError("license evidence artifact is not the selected lock artifact")
        if artifact["kind"] == "virtual-local" and evidence["kind"] != "local_file":
            raise ValueError("virtual project license must bind local primary bytes")
        if artifact["kind"] == "virtual-local":
            if evidence["license_path"] != "LICENSE" or license_by_identity[identity]["license"] != "Apache-2.0":
                raise ValueError("virtual project requires repository-root Apache-2.0 LICENSE evidence")
            license_bytes = REPO_ROOT / "LICENSE"
            if (not license_bytes.is_file() or license_bytes.is_symlink()
                    or license_bytes.resolve(strict=True) != license_bytes
                    or evidence["license_bytes"] != license_bytes.stat().st_size
                    or evidence["license_sha256"] != sha256(license_bytes)):
                raise ValueError("virtual project requires an actual repository license file")
    expected = dependency.get("lock_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("uv.lock SHA-256 has not been fixed by the VAST dependency audit")
    actual = sha256(LOCK)
    if actual != expected:
        raise ValueError("dedicated uv.lock SHA-256 drifted")
    for key in ("package_rows_sha256", "license_rows_sha256"):
        value = dependency.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"dependency {key} has not been fixed by primary-byte review")
    return {"lock_sha256": actual, "status": "LOCK_IDENTITY_OK"}


def self_test() -> None:
    gate = load_gate()
    verify_gate_contract(gate)
    assert gate["status"].startswith("BLOCKED_")
    assert gate["publication"] == "NO_UPLOAD"
    assert gate["dependency_audit"]["lock_sha256"] is None
    assert len(gate["weights"]) == 5
    dependency = json.loads((PROJECT / "dependency_audit.json").read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)
    assert dependency["compatibility"]["status"] == "BLOCKED_UNSATISFIABLE_PY312_TORCHAUDIO"
    assert "scanner" in dependency["compatibility"]["reason"]
    assert marker_reaches("sys_platform == 'linux' and platform_machine == 'x86_64'")
    assert not marker_reaches("sys_platform == 'darwin'")
    try:
        marker_reaches("nvidia == '1'")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown marker was accepted")
    try:
        validate_inactive_rows([{"name": "x", "version": "1", "reason": "inactive"}])
    except ValueError:
        pass
    else:
        raise AssertionError("malformed inactive row was accepted")
    base = {"version": 1, "revision": 3, "requires-python": "==3.12.*", "resolution-markers": ["platform_machine == 'x86_64' and sys_platform == 'linux'"], "supported-markers": ["platform_machine == 'x86_64' and sys_platform == 'linux'"], "package": [{"name": "torch", "version": "2", "source": {"registry": "https://download.pytorch.org/whl/cpu"}, "wheels": [{"url": "https://download-r2.pytorch.org/whl/cpu/torch.whl", "hash": "sha256:" + "0" * 64, "upload-time": "now"}]}]}
    parse_lock_data(base)
    with_size = {**base, "package": [{**base["package"][0], "wheels": [{**base["package"][0]["wheels"][0], "size": 1}]}]}
    parse_lock_data(with_size)
    for broken in (
        {**base, "supported-markers": ["platform_machine == 'aarch64'"]},
        {**base, "package": base["package"] * 2},
        {**base, "package": [{**base["package"][0], "name": "numpy", "source": {"registry": "https://pypi.org/simple"}}]},
        {**base, "package": [{**base["package"][0], "source": {"registry": "https://registry.invalid"}}]},
    ):
        try:
            parse_lock_data(broken)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed fake lock was accepted")
    source = Path(__file__).read_text(encoding="utf-8")
    snapshot = (PROJECT / UPSTREAM_REQUIREMENTS_FILE).read_text(encoding="utf-8")
    project = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
    project_names = {re.split(r"[<>=!~;\s]", item.strip(), maxsplit=1)[0].lower() for item in project["project"]["dependencies"]}
    assert "lameenc" in snapshot
    assert "lameenc" not in project_names and "numpy" in project_names
    dumper_source = (PROJECT / "dump_reference.py").read_text(encoding="utf-8")
    dumper_tree = ast.parse(dumper_source)
    dumper_imports = {
        alias.name
        for node in ast.walk(dumper_tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "demucs.audio" not in dumper_imports and "lameenc" not in dumper_imports
    assert sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "import_module"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "demucs.audio"
        for node in ast.walk(dumper_tree)
    ) == 1
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "import_module"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "lameenc"
        for node in ast.walk(dumper_tree)
    )
    for token in ("tomllib", "duplicate (name, version)", "CUDA/NVIDIA/Triton", "artifact_sha256", "locked_sdist"):
        assert token in source, f"lock contract missing: {token}"
    try:
        verify_gate_contract({**gate, "weights": gate["weights"][:-1]})
    except ValueError:
        pass
    else:
        raise AssertionError("truncated weight set was accepted")
    print("htdemucs multi audit self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dependency-gate", action="store_true")
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args()
    try:
        if args.self_test:
            if args.dependency_gate or args.source_dir is not None:
                raise ValueError("--self-test accepts no other options")
            self_test()
            return 0
        gate = load_gate()
        verify_gate_contract(gate)
        result: dict[str, Any] = {"status": "BLOCKED", "publication": "NO_UPLOAD"}
        if args.source_dir is not None:
            result["source"] = audit_source(args.source_dir, gate)
        if args.dependency_gate:
            result["dependency"] = audit_lock(gate)
        if not args.dependency_gate and args.source_dir is None:
            raise ValueError("use --dependency-gate or --source-dir")
        if gate.get("status") != "APPROVED_FOR_VAST_REFERENCE" or gate.get("blockers"):
            print(json.dumps(result, sort_keys=True))
            return 2
        result["status"] = "AUDIT_PASS"
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, subprocess.CalledProcessError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "BLOCKED", "publication": "NO_UPLOAD", "error": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
