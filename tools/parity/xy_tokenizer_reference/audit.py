#!/usr/bin/env -S uv run --no-project --python 3.12 python
"""Fail-closed audit for the XY-Tokenizer reference environment.

The generator is intentionally unable to manufacture an ``AUDITED_ALLOW``.
It checks an exact project/lock, the Linux x86_64 active closure, structured
license-byte evidence, and native bundled payload evidence, then emits a
BLOCKED report requiring an independent owner sign-off.  Self-tests use only
small fake TOML/JSON fixtures; they never import Torch or the upstream model.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import tempfile
import tomllib
from pathlib import Path
from typing import Any

SOURCE_REVISION = "5df5609c5883e555bd39a2d0b1005ca8f1a8f12e"
SOURCE_REPOSITORY = "https://github.com/gyt1145028706/XY-Tokenizer"
SOURCE_ROLE_BLOBS = {
    "config/xy_tokenizer_config.yaml": "83c50a60b3c0db62ce30b9cd65e0b0f5cd290f89",
    "inference.py": "9bb00a176f878d872f8eb7ed7a98501d3abb7e70",
    "inference_for_codec_evaluation.py": "4a98524ac90506a21b6155b31e945163c5d35d5b",
    "requirements.txt": "46b7b2d2aabb074ce87433eba2f55b31eee2363b",
    "utils/helpers.py": "9b144a4ce5ca6fd57b1a2903d940c4b4ffec4d97",
    "xy_tokenizer/model.py": "188f1b607d3e9a5953b3015ea9d262008ef535c0",
    "xy_tokenizer/nn/feature_extractor.py": "4d397b012ffe756fa9dfadc771f81e0afddd3963",
    "xy_tokenizer/nn/modules.py": "cc186d9dadd674172837d527fef0f0de183feb4c",
    "xy_tokenizer/nn/quantizer.py": "a7d28b963e98ea4f62f2a6e06b419cf0da0c2cc4",
}
EXPECTED_IMPORTS = {"einops", "librosa", "numpy", "scipy", "torch", "torchaudio", "transformers", "yaml"}
NATIVE_PACKAGES = {"cffi", "llvmlite", "numba", "numpy", "scipy", "soundfile", "torch", "torchaudio"}
STDLIB_IMPORTS = {"copy", "dataclasses", "logging", "math", "typing"}
MARKER = "sys_platform == 'linux' and platform_machine == 'x86_64'"
MARKER_ALIASES = {MARKER, "platform_machine == 'x86_64' and sys_platform == 'linux'"}
PYPI_INDEX = "https://pypi.org/simple"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
CPU_ARTIFACT_PREFIXES = (f"{CPU_INDEX}/", "https://download-r2.pytorch.org/whl/cpu/")
SCHEMA = "vokra-xy-tokenizer-dependency-audit-v1"
BLOCKER = "DEPENDENCY_CLOSURE_LICENSE_UNVERIFIED_BLOCKER"
DIRECT_DEPENDENCIES = {
    "numpy": "numpy",
    "torch": "torch>=2.0",
    "torchaudio": "torchaudio",
    "einops": "einops",
    "librosa": "librosa",
    "pyyaml": "pyyaml",
    "transformers": "transformers",
    "scipy": "scipy",
}
LOCK_TOP_LEVEL_KEYS = {"version", "revision", "requires-python", "resolution-markers", "supported-markers", "package"}
VIRTUAL_PROJECT_NAME = "vokra-xy-tokenizer-reference"
TARGET_MARKER_VALUES = {
    "implementation_name": "cpython",
    "platform_machine": "x86_64",
    "platform_python_implementation": "CPython",
    "python_version": "3.12",
    "sys_platform": "linux",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_unique(path: Path) -> Any:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)


def evaluate_marker(marker: str) -> bool:
    """Evaluate the small PEP 508 marker subset for the target runtime."""
    if not isinstance(marker, str) or not marker:
        raise ValueError("dependency marker must be a non-empty string")
    try:
        tree = ast.parse(marker, mode="eval")
    except SyntaxError as error:
        raise ValueError(f"dependency marker syntax is invalid: {marker}") from error

    def operand(node: ast.AST) -> str | tuple[str, ...]:
        if isinstance(node, ast.Name) and node.id in TARGET_MARKER_VALUES:
            return TARGET_MARKER_VALUES[node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, (ast.Tuple, ast.List)):
            values = tuple(operand(item) for item in node.elts)
            if not all(isinstance(item, str) for item in values):
                raise ValueError("dependency marker collection contains an invalid operand")
            return values
        raise ValueError("dependency marker contains an unknown name or operand")

    def evaluate(node: ast.AST) -> bool:
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)) and node.values:
            values = [evaluate(value) for value in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
            left = operand(node.left)
            right = operand(node.comparators[0])
            operation = node.ops[0]
            if isinstance(operation, ast.Eq):
                return left == right
            if isinstance(operation, ast.NotEq):
                return left != right
            if isinstance(operation, ast.In):
                return isinstance(right, tuple) and left in right
            if isinstance(operation, ast.NotIn):
                return isinstance(right, tuple) and left not in right
            if isinstance(operation, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) and isinstance(left, str) and isinstance(right, str):
                return {ast.Lt: left < right, ast.LtE: left <= right, ast.Gt: left > right, ast.GtE: left >= right}[type(operation)]
            raise ValueError("dependency marker uses an unknown operator")
        raise ValueError("dependency marker expression is unsupported")

    return evaluate(tree.body)


def regular(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")


def canonical_existing(path: Path, label: str, directory: bool | None = None) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.exists():
        raise ValueError(f"{label} must be an absolute existing non-symlink path")
    if directory is True and not path.is_dir():
        raise ValueError(f"{label} must be a directory")
    if directory is False and not path.is_file():
        raise ValueError(f"{label} must be a file")
    if any(parent.is_symlink() for parent in path.parents):
        raise ValueError(f"{label} parents must not contain symlinks")
    return path.resolve(strict=True)


def canonical_absent_output(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink() or path.exists():
        raise ValueError("audit output must be an absent absolute path")
    if any(parent.is_symlink() for parent in path.parents) or not path.parent.is_dir():
        raise ValueError("audit output parent must be an existing non-symlink directory")
    return path.parent.resolve(strict=True) / path.name


def disjoint(*paths: Path) -> None:
    for index, path in enumerate(paths):
        for other in paths[index + 1 :]:
            if path == other or path.is_relative_to(other) or other.is_relative_to(path):
                raise ValueError("project/source/output paths must be canonical and disjoint")


def validate_pyproject(data: dict[str, Any]) -> None:
    if set(data) != {"project", "tool"}:
        raise ValueError("pyproject top-level schema is not exact")
    project = data.get("project")
    if not isinstance(project, dict) or set(project) != {"name", "version", "description", "requires-python", "dependencies"}:
        raise ValueError("pyproject project schema is not exact")
    if project["name"] != "vokra-xy-tokenizer-reference" or project["version"] != "0.1.0" or project["requires-python"] != "==3.12.*":
        raise ValueError("pyproject identity/python policy mismatch")
    dependencies = project.get("dependencies")
    if not isinstance(dependencies, list) or not all(isinstance(dependency, str) for dependency in dependencies) or len(dependencies) != len(set(dependencies)) or set(dependencies) != set(DIRECT_DEPENDENCIES.values()):
        raise ValueError("pyproject direct dependency set mismatch")
    tool = data.get("tool")
    uv = tool.get("uv") if isinstance(tool, dict) else None
    if not isinstance(tool, dict) or set(tool) != {"uv"} or not isinstance(uv, dict) or set(uv) != {"package", "environments", "sources", "index"}:
        raise ValueError("pyproject uv schema is not exact")
    if uv.get("package") is not False or uv.get("environments") != [MARKER]:
        raise ValueError("pyproject Linux x86_64 policy mismatch")
    if uv.get("sources") != {"torch": {"index": "pytorch-cpu"}, "torchaudio": {"index": "pytorch-cpu"}}:
        raise ValueError("pyproject Torch source mapping mismatch")
    indexes = uv.get("index")
    if not isinstance(indexes, list) or len(indexes) != 1 or set(indexes[0]) != {"name", "url", "explicit"} or indexes[0] != {"name": "pytorch-cpu", "url": CPU_INDEX, "explicit": True}:
        raise ValueError("pyproject CPU index schema mismatch")


def supported_wheel(url: str) -> bool:
    filename = url.rsplit("/", 1)[-1].lower()
    if not filename.endswith(".whl"):
        return False
    if any(token in filename for token in ("win", "windows", "macosx", "aarch64", "arm64")):
        return False
    return filename.endswith("-any.whl") or ("manylinux" in filename or "musllinux" in filename) and "x86_64" in filename or "linux_x86_64" in filename


def official_cpu_artifact(url: str) -> bool:
    return any(url.startswith(prefix) for prefix in CPU_ARTIFACT_PREFIXES)


def validate_requires_dist(row: dict[str, Any]) -> None:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict) or set(metadata) != {"requires-dist"} or not isinstance(metadata["requires-dist"], list):
        raise ValueError("uv.lock virtual project metadata schema drifted")
    requirements = metadata["requires-dist"]
    if len(requirements) != len(DIRECT_DEPENDENCIES):
        raise ValueError("uv.lock virtual requires-dist set drifted")
    seen: set[str] = set()
    for requirement in requirements:
        if not isinstance(requirement, dict) or not isinstance(requirement.get("name"), str) or requirement["name"] not in DIRECT_DEPENDENCIES or requirement["name"] in seen:
            raise ValueError("uv.lock virtual requires-dist identity drifted")
        name = requirement["name"]
        expected_keys = {"name"}
        if name == "torch":
            expected_keys |= {"specifier", "index"}
        elif name == "torchaudio":
            expected_keys.add("index")
        if set(requirement) != expected_keys:
            raise ValueError("uv.lock virtual requires-dist schema drifted")
        if name == "torch" and (requirement.get("specifier") != ">=2.0" or requirement.get("index") != CPU_INDEX):
            raise ValueError("uv.lock torch requires-dist must preserve torch>=2.0 on the CPU index")
        if name == "torchaudio" and requirement.get("index") != CPU_INDEX:
            raise ValueError("uv.lock Torch requires-dist must use the CPU index")
        seen.add(name)


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def source_evidence(source: Path) -> dict[str, Any]:
    if source.is_symlink() or not source.is_dir():
        raise ValueError("source must be a real directory")
    if git(source, "rev-parse", "HEAD") != SOURCE_REVISION or git(source, "remote", "get-url", "origin").removesuffix("/").removesuffix(".git") != SOURCE_REPOSITORY:
        raise ValueError("source revision/origin mismatch")
    if git(source, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("source checkout is dirty")
    files = {}
    for relative, expected in SOURCE_ROLE_BLOBS.items():
        path = source / relative
        if path.is_symlink() or not path.is_file() or blob_sha1(path) != expected:
            raise ValueError(f"source role mismatch: {relative}")
        files[relative] = {"git_blob_sha1": expected, "sha256": sha256(path)}
    imports: set[str] = set()
    for relative in ("xy_tokenizer/model.py", "xy_tokenizer/nn/feature_extractor.py", "xy_tokenizer/nn/modules.py", "xy_tokenizer/nn/quantizer.py"):
        tree = ast.parse((source / relative).read_text(encoding="utf-8"), filename=relative)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imports.add(node.module.split(".")[0])
    unexpected = imports - EXPECTED_IMPORTS - STDLIB_IMPORTS
    if unexpected:
        raise ValueError(f"unexpected third-party runtime imports: {sorted(unexpected)}")
    actual = imports & EXPECTED_IMPORTS
    if actual != EXPECTED_IMPORTS:
        raise ValueError(f"runtime import closure mismatch: {sorted(actual)}")
    helper_text = (source / "utils/helpers.py").read_text(encoding="utf-8")
    if "debugpy" not in helper_text or "debugpy" in imports:
        raise ValueError("debugpy route exclusion proof failed")
    return {
        "repository": SOURCE_REPOSITORY,
        "revision": SOURCE_REVISION,
        "files": files,
        "runtime_imports": sorted(actual),
        "debugpy": {
            "present_in_historical_helpers_role": True,
            "route_imports_contain": False,
            "excluded": True,
        },
    }


def lock_rows(lock: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    regular(lock, "uv.lock")
    data = tomllib.loads(lock.read_text(encoding="utf-8"))
    if set(data) != LOCK_TOP_LEVEL_KEYS:
        raise ValueError("uv.lock top-level schema is not exact")
    if data.get("version") != 1 or data.get("revision") != 3 or data.get("requires-python") != "==3.12.*":
        raise ValueError("uv.lock schema version is unsupported")
    rows = data.get("package")
    if not isinstance(rows, list) or not rows:
        raise ValueError("uv.lock package rows are absent")
    seen_name_versions: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not row["name"] or not isinstance(row.get("version"), str) or not row["version"]:
            raise ValueError("uv.lock package identity is invalid")
        if set(row) - {"name", "version", "source", "dependencies", "sdist", "wheels", "resolution-markers", "optional-dependencies", "metadata"}:
            raise ValueError("uv.lock package row has unknown fields")
        source = row.get("source")
        if not isinstance(source, dict) or set(source) not in ({"registry"}, {"virtual"}):
            raise ValueError("uv.lock package source is invalid")
        source_value = source[next(iter(source))]
        if not isinstance(source_value, str) or not source_value:
            raise ValueError("uv.lock package source value is invalid")
        if source == {"virtual": "."} and (row["name"], row["version"]) != (VIRTUAL_PROJECT_NAME, "0.1.0"):
            raise ValueError("only the virtual project identity may use source={virtual='.'}")
        if source == {"virtual": "."}:
            validate_requires_dist(row)
        elif "metadata" in row:
            raise ValueError("uv.lock metadata is allowed only on the virtual project row")
        markers = row.get("resolution-markers", [])
        if not isinstance(markers, list) or not all(isinstance(marker, str) and marker in MARKER_ALIASES for marker in markers) or len(set(markers)) != len(markers):
            raise ValueError("uv.lock package resolution markers are invalid")
        dependencies = row.get("dependencies", [])
        if not isinstance(dependencies, list):
            raise ValueError("uv.lock dependency list is invalid")
        for dependency in dependencies:
            if not isinstance(dependency, dict) or set(dependency) - {"name", "marker"} or not isinstance(dependency.get("name"), str) or not dependency["name"]:
                raise ValueError("uv.lock dependency row is invalid")
            if "marker" in dependency:
                if not isinstance(dependency["marker"], str):
                    raise ValueError("uv.lock dependency marker is invalid")
                evaluate_marker(dependency["marker"])
        optional = row.get("optional-dependencies", {})
        if not isinstance(optional, dict) or any(not isinstance(group, str) or not group or not isinstance(group_dependencies, list) for group, group_dependencies in optional.items()):
            raise ValueError("uv.lock optional dependency rows are invalid")
        for group_dependencies in optional.values():
            for dependency in group_dependencies:
                if not isinstance(dependency, dict) or set(dependency) - {"name", "marker"} or not isinstance(dependency.get("name"), str) or not dependency["name"]:
                    raise ValueError("uv.lock optional dependency row is invalid")
                if "marker" in dependency:
                    if not isinstance(dependency["marker"], str):
                        raise ValueError("uv.lock optional dependency marker is invalid")
                    evaluate_marker(dependency["marker"])
        for artifact_key in ("sdist", "wheels"):
            artifacts = row.get(artifact_key, []) if artifact_key == "wheels" else ([row[artifact_key]] if artifact_key in row else [])
            if not isinstance(artifacts, list):
                raise ValueError("uv.lock artifact list is invalid")
            for artifact in artifacts:
                registry = source.get("registry")
                required_artifact_keys = {"url", "hash", "upload-time"}
                allowed_artifact_keys = required_artifact_keys | {"size"}
                size_required = registry != CPU_INDEX
                if not isinstance(artifact, dict) or set(artifact) - allowed_artifact_keys or not required_artifact_keys <= set(artifact) or (size_required and "size" not in artifact) or not isinstance(artifact["url"], str) or not artifact["url"].startswith("https://") or (registry == CPU_INDEX and not official_cpu_artifact(artifact["url"])) or not isinstance(artifact["hash"], str) or not artifact["hash"].startswith("sha256:") or len(artifact["hash"]) != 71 or any(character not in "0123456789abcdef" for character in artifact["hash"][7:]) or ("size" in artifact and (isinstance(artifact["size"], bool) or not isinstance(artifact["size"], int) or artifact["size"] <= 0)) or not isinstance(artifact["upload-time"], str) or not artifact["upload-time"]:
                    raise ValueError("uv.lock artifact identity is invalid")
        identity = (row["name"], row["version"])
        if "registry" in source:
            registry = source["registry"]
            expected_registry = CPU_INDEX if row["name"] in {"torch", "torchaudio"} else PYPI_INDEX
            if registry != expected_registry:
                raise ValueError(f"package registry is not approved for {row['name']}")
        if identity in seen_name_versions:
            raise ValueError("duplicate uv.lock package identity")
        seen_name_versions.add(identity)
    return data, rows


def active_closure(data: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    resolution_markers = data.get("resolution-markers")
    supported_markers = data.get("supported-markers")
    if (
        not isinstance(resolution_markers, list)
        or not all(isinstance(marker, str) for marker in resolution_markers)
        or set(resolution_markers) - MARKER_ALIASES
        or not isinstance(supported_markers, list)
        or not all(isinstance(marker, str) for marker in supported_markers)
        or set(supported_markers) - MARKER_ALIASES
        or len(resolution_markers) != 1
        or len(supported_markers) != 1
    ):
        raise ValueError("uv.lock marker policy is not Linux x86_64 only")
    virtual = [row for row in rows if isinstance(row.get("source"), dict) and row["source"].get("virtual") == "."]
    if len(virtual) != 1:
        raise ValueError("uv.lock must contain exactly one virtual project row")
    if virtual[0]["name"] != VIRTUAL_PROJECT_NAME or virtual[0]["version"] != "0.1.0" or virtual[0]["source"] != {"virtual": "."}:
        raise ValueError("virtual row is not the exact local project identity")
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_name.setdefault(row["name"], []).append(row)
    pending = [virtual[0]["name"]]
    selected: dict[str, dict[str, Any]] = {}
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        candidates = by_name.get(name, [])
        active_candidates = [row for row in candidates if not row.get("resolution-markers") or set(row["resolution-markers"]) & MARKER_ALIASES]
        if len(active_candidates) != 1:
            raise ValueError(f"active dependency row missing: {name}")
        row = active_candidates[0]
        if name in {"cuda-bindings", "triton"} or name.startswith("nvidia-"):
            raise ValueError(f"forbidden active CUDA package: {name}")
        selected[name] = row
        for dependency in row.get("dependencies", []):
            if not isinstance(dependency, dict) or not isinstance(dependency.get("name"), str) or not dependency["name"]:
                raise ValueError("lock dependency row is invalid")
            marker = dependency.get("marker")
            if marker is not None and not evaluate_marker(marker):
                continue
            pending.append(dependency["name"])
    return [selected[name] for name in sorted(selected)]


def validate_license_rows(rows: list[dict[str, Any]], evidence: dict[str, Any], project_pyproject: Path | None = None) -> list[dict[str, Any]]:
    if not isinstance(evidence, dict) or set(evidence) != {"packages"} or not isinstance(evidence["packages"], list):
        raise ValueError("license evidence must contain exactly structured package rows")
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for row in evidence["packages"]:
        if not isinstance(row, dict) or set(row) != {"name", "version", "license", "artifact", "native_payloads"}:
            raise ValueError("license evidence row schema mismatch")
        identity = (row.get("name"), row.get("version"))
        if not all(isinstance(item, str) and item for item in identity) or identity in seen:
            raise ValueError("duplicate/invalid license evidence identity")
        seen.add(identity)
        license_row = row["license"]
        if not isinstance(license_row, dict) or set(license_row) != {"kind", "source", "revision", "sha256", "bytes", "spdx"}:
            raise ValueError("license evidence must be publisher or locked-sdist byte evidence")
        if license_row["kind"] not in {"publisher", "locked-sdist"} or not isinstance(license_row["source"], str) or not license_row["source"].startswith("https://") or not isinstance(license_row["revision"], str) or not license_row["revision"] or not isinstance(license_row["sha256"], str) or len(license_row["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in license_row["sha256"]) or isinstance(license_row["bytes"], bool) or not isinstance(license_row["bytes"], int) or license_row["bytes"] <= 0 or not isinstance(license_row["spdx"], str) or not license_row["spdx"]:
            raise ValueError("license byte evidence is incomplete")
        if any(token in license_row["spdx"].upper() for token in ("GPL", "LGPL", "UNKNOWN", "UNLICENSED")):
            raise ValueError("GPL/LGPL/unknown license is not allowed")
        artifact = row["artifact"]
        if not isinstance(artifact, dict) or set(artifact) != {"kind", "url", "sha256", "bytes"}:
            raise ValueError("selected artifact identity is incomplete")
        if artifact.get("kind") == "virtual-local":
            if project_pyproject is None or artifact.get("url") != "pyproject.toml" or artifact.get("sha256") != sha256(project_pyproject) or artifact.get("bytes") != project_pyproject.stat().st_size:
                raise ValueError("virtual project artifact is not bound to pyproject bytes")
        elif artifact.get("kind") in {"wheel", "sdist"}:
            if not isinstance(artifact.get("url"), str) or not artifact["url"].startswith("https://") or not isinstance(artifact.get("sha256"), str) or not artifact["sha256"].startswith("sha256:") or len(artifact["sha256"]) != 71 or any(character not in "0123456789abcdef" for character in artifact["sha256"][7:]) or isinstance(artifact.get("bytes"), bool) or not isinstance(artifact.get("bytes"), int) or artifact["bytes"] <= 0:
                raise ValueError("selected artifact identity is incomplete")
            if artifact["kind"] == "wheel" and not supported_wheel(artifact["url"]):
                raise ValueError("selected wheel is not Linux x86_64 or universal")
        else:
            raise ValueError("selected artifact kind is invalid")
        lock_row = next((candidate for candidate in rows if (candidate["name"], candidate["version"]) == identity), None)
        if lock_row is None:
            raise ValueError("selected artifact package is not locked")
        if artifact["kind"] == "virtual-local":
            if (lock_row["name"], lock_row["version"]) != (VIRTUAL_PROJECT_NAME, "0.1.0") or lock_row["source"] != {"virtual": "."}:
                raise ValueError("virtual-local artifact requires the exact virtual project row")
        else:
            lock_artifacts = lock_row.get("wheels", []) if artifact["kind"] == "wheel" else ([lock_row["sdist"]] if "sdist" in lock_row else [])
            if not any(candidate.get("url") == artifact["url"] and candidate.get("hash") == artifact["sha256"] and ("size" not in candidate or candidate["size"] == artifact["bytes"]) for candidate in lock_artifacts):
                raise ValueError("selected artifact is not bound to uv.lock")
        payloads = row["native_payloads"]
        if not isinstance(payloads, list):
            raise ValueError("native bundled payload evidence is missing")
        if identity[0] in NATIVE_PACKAGES and not payloads:
            raise ValueError("native bundled payload evidence is incomplete")
        for payload in payloads:
            if not isinstance(payload, dict) or set(payload) != {"name", "sha256", "bytes", "license_source", "license_revision", "artifact_sha256"} or not isinstance(payload.get("name"), str) or not isinstance(payload.get("sha256"), str) or len(payload["sha256"]) != 64 or any(character not in "0123456789abcdef" for character in payload["sha256"]) or not isinstance(payload.get("bytes"), int) or payload["bytes"] <= 0 or not isinstance(payload.get("license_source"), str) or not payload["license_source"].startswith("https://") or not isinstance(payload.get("license_revision"), str) or not payload["license_revision"] or payload.get("artifact_sha256") != artifact["sha256"]:
                raise ValueError("native bundled payload evidence is incomplete")
        out.append({"name": identity[0], "version": identity[1], "license": license_row, "artifact": artifact, "native_payloads": payloads})
    lock_identity = {(row["name"], row["version"]) for row in rows}
    if seen != lock_identity:
        raise ValueError("license evidence does not cover the active closure exactly")
    return out


def compatible_projection(package_evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project rich evidence into the dumper's exact gate schema."""
    return [
        {
            "name": row["name"],
            "version": row["version"],
            "license": row["license"]["spdx"],
            "evidence": {
                "source": row["license"]["source"],
                "revision": row["license"]["revision"],
                "sha256": row["license"]["sha256"],
            },
        }
        for row in package_evidence
    ]


def audit(project: Path, source: Path | None, output: Path) -> dict[str, Any]:
    project = canonical_existing(project, "project", directory=True)
    if source is not None:
        source = canonical_existing(source, "source", directory=True)
    output = canonical_absent_output(output)
    if source is not None:
        disjoint(project, source, output)
    else:
        disjoint(project, output)
    pyproject, lock, evidence_path = (project / name for name in ("pyproject.toml", "uv.lock", "license_evidence.json"))
    for path in (pyproject, lock, evidence_path):
        regular(path, path.name)
    blockers: list[str] = []
    pyproject_data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    try:
        validate_pyproject(pyproject_data)
    except ValueError as error:
        blockers.append(f"REFERENCE_PROJECT_POLICY_MISMATCH:{error}")
    active: list[dict[str, Any]] = []
    package_evidence: list[dict[str, Any]] = []
    try:
        lock_data, all_rows = lock_rows(lock)
        active = active_closure(lock_data, all_rows)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        blockers.append(f"{BLOCKER}:{error}")
    else:
        try:
            package_evidence = validate_license_rows(active, json_unique(evidence_path), pyproject)
        except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
            blockers.append(f"{BLOCKER}:{error}")
    source_packet: dict[str, Any] | None = None
    if source is not None:
        try:
            source_packet = source_evidence(source)
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            blockers.append(f"SOURCE_AUTHENTICATION_BLOCKER:{error}")
    blockers.append("OWNER_SIGNOFF_REQUIRED")
    # Keep the dumper-facing projection deliberately small and compatible;
    # richer artifact/native evidence stays in the human review manifest.
    projection = compatible_projection(package_evidence)
    manifest = {"schema": SCHEMA, "status": "BLOCKED", "lock_sha256": sha256(lock), "pyproject_sha256": sha256(pyproject), "packages": projection}
    output.mkdir()
    (output / "dependency_audit.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "license_gate_manifest.json").write_text(json.dumps({"schema": "vokra-xy-tokenizer-license-gate-v1", "status": "BLOCKED", "publication": "NO_UPLOAD", "owner_signoff": "REQUIRED", "blockers": blockers, "project": {"pyproject_sha256": manifest["pyproject_sha256"], "uv_lock_sha256": manifest["lock_sha256"], "active_package_count": len(active)}, "source": source_packet, "packages": package_evidence}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def self_test() -> None:
    assert MARKER == "sys_platform == 'linux' and platform_machine == 'x86_64'"
    assert evaluate_marker("implementation_name != 'PyPy'")
    assert not evaluate_marker("implementation_name == 'PyPy'")
    for unsupported in ("unknown_marker == 'x'", "sys_platform ~= 'linux'"):
        try:
            evaluate_marker(unsupported)
        except ValueError:
            pass
        else:
            raise AssertionError("unsupported dependency marker accepted")
    project_data = tomllib.loads((Path(__file__).parent / "pyproject.toml").read_text(encoding="utf-8"))
    validate_pyproject(project_data)
    tracked_lock = Path(__file__).parent / "uv.lock"
    if tracked_lock.is_file():
        tracked_data, tracked_rows = lock_rows(tracked_lock)
        assert len(tracked_rows) == 57
        assert len(active_closure(tracked_data, tracked_rows)) == 57
    unknown_project_data = {**project_data, "project": {**project_data["project"], "unexpected": True}}
    try:
        validate_pyproject(unknown_project_data)
    except ValueError as error:
        assert "schema" in str(error)
    else:
        raise AssertionError("unknown pyproject field accepted")
    header = "version = 1\nrevision = 3\nrequires-python = \"==3.12.*\"\nresolution-markers = [\"sys_platform == 'linux' and platform_machine == 'x86_64'\"]\nsupported-markers = [\"sys_platform == 'linux' and platform_machine == 'x86_64'\"]\n"
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        metadata_inline = "metadata={requires-dist=[{name='numpy'},{name='torch',specifier='>=2.0',index='https://download.pytorch.org/whl/cpu'},{name='torchaudio',index='https://download.pytorch.org/whl/cpu'},{name='einops'},{name='librosa'},{name='pyyaml'},{name='transformers'},{name='scipy'}]}\n"

        def write_fixture(path: Path, text: str) -> None:
            path.write_text(text.replace("source={virtual='.'}\n", "source={virtual='.'}\n" + metadata_inline), encoding="utf-8")

        lock = root / "uv.lock"
        write_fixture(lock, header + "[[package]]\nname='vokra-xy-tokenizer-reference'\nversion='0.1.0'\nsource={virtual='.'}\n")
        data, rows = lock_rows(lock)
        assert active_closure(data, rows)[0]["name"] == "vokra-xy-tokenizer-reference"
        package_text = lock.read_text(encoding="utf-8").split("[[package]]", 1)[1]
        missing_metadata = root / "missing-metadata.lock"
        missing_metadata.write_text(header + "[[package]]\nname='vokra-xy-tokenizer-reference'\nversion='0.1.0'\nsource={virtual='.'}\n", encoding="utf-8")
        try:
            lock_rows(missing_metadata)
        except ValueError as error:
            assert "metadata" in str(error)
        else:
            raise AssertionError("virtual project without requires-dist accepted")
        metadata_text = """
[package.metadata]
requires-dist = [
    { name = 'numpy' },
    { name = 'torch', specifier = '>=2.0', index = 'https://download.pytorch.org/whl/cpu' },
    { name = 'torchaudio', index = 'https://download.pytorch.org/whl/cpu' },
    { name = 'einops' },
    { name = 'librosa' },
    { name = 'pyyaml' },
    { name = 'transformers' },
    { name = 'scipy' },
]
"""
        metadata_lock = root / "virtual-metadata.lock"
        metadata_lock.write_text(header + "[[package]]\nname='vokra-xy-tokenizer-reference'\nversion='0.1.0'\nsource={virtual='.'}\n" + metadata_text, encoding="utf-8")
        metadata_data, metadata_rows = lock_rows(metadata_lock)
        assert metadata_data["package"][0]["metadata"]["requires-dist"][1]["index"] == CPU_INDEX
        drifted_metadata = root / "drifted-metadata.lock"
        drifted_metadata.write_text(metadata_lock.read_text(encoding="utf-8").replace("{ name = 'numpy' }", "{ name = 'numpy', specifier = '>=1' }", 1), encoding="utf-8")
        try:
            lock_rows(drifted_metadata)
        except ValueError as error:
            assert "schema" in str(error)
        else:
            raise AssertionError("plain dependency specifier drift accepted")
        missing_torch_index = root / "missing-torch-index.lock"
        missing_torch_index.write_text(metadata_lock.read_text(encoding="utf-8").replace(", index = 'https://download.pytorch.org/whl/cpu'", "", 1), encoding="utf-8")
        try:
            lock_rows(missing_torch_index)
        except ValueError as error:
            assert "schema" in str(error) or "CPU index" in str(error)
        else:
            raise AssertionError("missing Torch index accepted")
        wrong_torch_specifier = root / "wrong-torch-specifier.lock"
        wrong_torch_specifier.write_text(metadata_lock.read_text(encoding="utf-8").replace("specifier = '>=2.0'", "specifier = '==2.0'", 1), encoding="utf-8")
        try:
            lock_rows(wrong_torch_specifier)
        except ValueError as error:
            assert "torch>=2.0" in str(error)
        else:
            raise AssertionError("resolved Torch pin accepted in requires-dist")
        registry_metadata = root / "registry-metadata.lock"
        registry_metadata.write_text(header + "[[package]]\nname='vokra-xy-tokenizer-reference'\nversion='0.1.0'\nsource={virtual='.'}\n[[package]]\nname='numpy'\nversion='1'\nsource={registry='https://pypi.org/simple'}\nmetadata={requires-dist=[]}\n", encoding="utf-8")
        try:
            lock_rows(registry_metadata)
        except ValueError as error:
            assert "metadata" in str(error)
        else:
            raise AssertionError("registry package metadata accepted")
        missing_requires_python = root / "missing-requires-python.lock"
        missing_requires_python.write_text(header.replace('requires-python = "==3.12.*"\n', "") + "[[package]]" + package_text, encoding="utf-8")
        try:
            lock_rows(missing_requires_python)
        except ValueError as error:
            assert "top-level" in str(error) or "schema" in str(error)
        else:
            raise AssertionError("lock without exact requires-python accepted")
        unknown_top_level = root / "unknown-top-level.lock"
        unknown_top_level.write_text(header + "unexpected = true\n[[package]]" + package_text, encoding="utf-8")
        try:
            lock_rows(unknown_top_level)
        except ValueError as error:
            assert "top-level" in str(error)
        else:
            raise AssertionError("unknown lock top-level field accepted")
        duplicate_lock = root / "duplicate.lock"
        duplicate_lock.write_text(lock.read_text(encoding="utf-8") + "[[package]]\nname='vokra-xy-tokenizer-reference'\nversion='0.1.0'\nsource={virtual='.'}\n" + metadata_inline, encoding="utf-8")
        try:
            lock_rows(duplicate_lock)
        except ValueError as error:
            assert "duplicate" in str(error)
        else:
            raise AssertionError("duplicate lock identity accepted")
        duplicate_source = root / "duplicate-source.lock"
        write_fixture(duplicate_source, header + "[[package]]\nname='vokra-xy-tokenizer-reference'\nversion='0.1.0'\nsource={virtual='.'}\n[[package]]\nname='vokra-xy-tokenizer-reference'\nversion='0.1.0'\nsource={registry='https://pypi.org/simple'}\n")
        try:
            lock_rows(duplicate_source)
        except ValueError as error:
            assert "duplicate" in str(error)
        else:
            raise AssertionError("same name/version from different sources accepted")
        forbidden = root / "forbidden.lock"
        write_fixture(forbidden, header + "[[package]]\nname='vokra-xy-tokenizer-reference'\nversion='0.1.0'\nsource={virtual='.'}\ndependencies=[{name='cuda-bindings'}]\n[[package]]\nname='cuda-bindings'\nversion='1'\nsource={registry='https://pypi.org/simple'}\n")
        forbidden_data, forbidden_rows = lock_rows(forbidden)
        try:
            active_closure(forbidden_data, forbidden_rows)
        except ValueError as error:
            assert "forbidden active CUDA" in str(error)
        else:
            raise AssertionError("active CUDA package accepted")
        wrong_registry = root / "wrong-registry.lock"
        write_fixture(wrong_registry, header + "[[package]]\nname='vokra-xy-tokenizer-reference'\nversion='0.1.0'\nsource={virtual='.'}\n[[package]]\nname='torch'\nversion='2.0'\nsource={registry='https://pypi.org/simple'}\n")
        try:
            lock_rows(wrong_registry)
        except ValueError as error:
            assert "registry" in str(error)
        else:
            raise AssertionError("wrong Torch registry accepted")
        artifact_lock = root / "artifact.lock"
        write_fixture(artifact_lock, header + "[[package]]\nname='vokra-xy-tokenizer-reference'\nversion='0.1.0'\nsource={virtual='.'}\ndependencies=[{name='numpy'}]\n[[package]]\nname='numpy'\nversion='1'\nsource={registry='https://pypi.org/simple'}\nwheels=[{url='https://example.invalid/numpy-1-py3-none-manylinux_2_17_x86_64.whl',hash='sha256:" + "a" * 64 + "',size=1,upload-time='2026-01-01T00:00:00Z'}]\n")
        artifact_data, artifact_rows = lock_rows(artifact_lock)
        artifact_active = active_closure(artifact_data, artifact_rows)
        artifact_package = [row for row in artifact_active if row["name"] == "numpy"]
        pypi_missing_size = root / "pypi-missing-size.lock"
        write_fixture(pypi_missing_size, header + "[[package]]\nname='vokra-xy-tokenizer-reference'\nversion='0.1.0'\nsource={virtual='.'}\ndependencies=[{name='numpy'}]\n[[package]]\nname='numpy'\nversion='1'\nsource={registry='https://pypi.org/simple'}\nwheels=[{url='https://example.invalid/numpy-1-py3-none-manylinux_2_17_x86_64.whl',hash='sha256:" + "a" * 64 + "',upload-time='2026-01-01T00:00:00Z'}]\n")
        try:
            lock_rows(pypi_missing_size)
        except ValueError as error:
            assert "artifact" in str(error)
        else:
            raise AssertionError("PyPI size-less artifact accepted")
        cpu_lock = root / "cpu-size-less.lock"
        write_fixture(cpu_lock, header + "[[package]]\nname='vokra-xy-tokenizer-reference'\nversion='0.1.0'\nsource={virtual='.'}\ndependencies=[{name='torch'}]\n[[package]]\nname='torch'\nversion='2.0'\nsource={registry='https://download.pytorch.org/whl/cpu'}\nwheels=[{url='https://download.pytorch.org/whl/cpu/torch-2.0-cp312-cp312-manylinux_2_17_x86_64.whl',hash='sha256:" + "d" * 64 + "',upload-time='2026-01-01T00:00:00Z'}]\n")
        cpu_data, cpu_rows = lock_rows(cpu_lock)
        cpu_active = active_closure(cpu_data, cpu_rows)
        cpu_r2_lock = root / "cpu-r2-size-less.lock"
        write_fixture(cpu_r2_lock, header + "[[package]]\nname='vokra-xy-tokenizer-reference'\nversion='0.1.0'\nsource={virtual='.'}\ndependencies=[{name='torch'}]\n[[package]]\nname='torch'\nversion='2.0'\nsource={registry='https://download.pytorch.org/whl/cpu'}\nwheels=[{url='https://download-r2.pytorch.org/whl/cpu/torch-2.0-cp312-cp312-manylinux_2_17_x86_64.whl',hash='sha256:" + "d" * 64 + "',upload-time='2026-01-01T00:00:00Z'}]\n")
        r2_data, r2_rows = lock_rows(cpu_r2_lock)
        r2_active = active_closure(r2_data, r2_rows)
        assert r2_active[0]["name"] == "torch"
        cpu_evil_lock = root / "cpu-evil-size-less.lock"
        write_fixture(cpu_evil_lock, header + "[[package]]\nname='vokra-xy-tokenizer-reference'\nversion='0.1.0'\nsource={virtual='.'}\ndependencies=[{name='torch'}]\n[[package]]\nname='torch'\nversion='2.0'\nsource={registry='https://download.pytorch.org/whl/cpu'}\nwheels=[{url='https://evil.example/download/torch-2.0-cp312-cp312-manylinux_2_17_x86_64.whl',hash='sha256:" + "d" * 64 + "',upload-time='2026-01-01T00:00:00Z'}]\n")
        try:
            lock_rows(cpu_evil_lock)
        except ValueError as error:
            assert "artifact" in str(error)
        else:
            raise AssertionError("non-PyTorch size-less CPU artifact accepted")
        project_file = root / "pyproject.toml"
        project_file.write_text("[project]\nname='vokra-xy-tokenizer-reference'\n", encoding="utf-8")
        license_base = {"kind": "publisher", "source": "https://example.invalid/license", "revision": "r1", "sha256": "b" * 64, "bytes": 1, "spdx": "BSD-3-Clause"}
        artifact_base = {"kind": "wheel", "url": "https://example.invalid/numpy-1-py3-none-manylinux_2_17_x86_64.whl", "sha256": "sha256:" + "a" * 64, "bytes": 1}
        rich = {"name": "numpy", "version": "1", "license": license_base, "artifact": artifact_base, "native_payloads": [{"name": "numpy.core", "sha256": "c" * 64, "bytes": 1, "license_source": "https://example.invalid/native", "license_revision": "r1", "artifact_sha256": artifact_base["sha256"]}]}
        virtual_rich = {"name": "vokra-xy-tokenizer-reference", "version": "0.1.0", "license": license_base, "artifact": {"kind": "virtual-local", "url": "pyproject.toml", "sha256": sha256(project_file), "bytes": project_file.stat().st_size}, "native_payloads": []}
        assert len(validate_license_rows(artifact_active, {"packages": [virtual_rich, rich]}, project_file)) == 2
        cpu_artifact = {"kind": "wheel", "url": "https://download-r2.pytorch.org/whl/cpu/torch-2.0-cp312-cp312-manylinux_2_17_x86_64.whl", "sha256": "sha256:" + "d" * 64, "bytes": 1}
        cpu_rich = {"name": "torch", "version": "2.0", "license": license_base, "artifact": cpu_artifact, "native_payloads": [{"name": "torch._C", "sha256": "e" * 64, "bytes": 1, "license_source": "https://example.invalid/native", "license_revision": "r1", "artifact_sha256": cpu_artifact["sha256"]}]}
        assert len(validate_license_rows(r2_active, {"packages": [virtual_rich, cpu_rich]}, project_file)) == 2
        registry_virtual_bypass = dict(rich, name="numpy", artifact=virtual_rich["artifact"])
        try:
            validate_license_rows(artifact_package, {"packages": [registry_virtual_bypass]}, project_file)
        except ValueError as error:
            assert "virtual project row" in str(error)
        else:
            raise AssertionError("registry package accepted virtual-local artifact evidence")
        projection = compatible_projection([rich])[0]
        assert set(projection) == {"name", "version", "license", "evidence"}
        assert set(projection["evidence"]) == {"source", "revision", "sha256"}
        try:
            validate_license_rows(artifact_package, {"packages": [rich, dict(rich)]})
        except ValueError as error:
            assert "duplicate" in str(error)
        else:
            raise AssertionError("duplicate audit identity accepted")
        wrong_artifact = dict(rich, artifact=dict(artifact_base, sha256="sha256:" + "d" * 64))
        try:
            validate_license_rows(artifact_package, {"packages": [wrong_artifact]})
        except ValueError as error:
            assert "artifact" in str(error)
        else:
            raise AssertionError("wrong artifact digest accepted")
        gpl = dict(rich, license=dict(license_base, spdx="GPL-3.0"))
        try:
            validate_license_rows(artifact_package, {"packages": [gpl]})
        except ValueError as error:
            assert "GPL" in str(error)
        else:
            raise AssertionError("GPL license accepted")
        no_native = dict(rich, native_payloads=[])
        try:
            validate_license_rows(artifact_package, {"packages": [no_native]})
        except ValueError as error:
            assert "native" in str(error)
        else:
            raise AssertionError("missing native payload evidence accepted")
        duplicate = root / "duplicate.json"
        duplicate.write_text('{"x":1,"x":2}\n', encoding="utf-8")
        try:
            json_unique(duplicate)
        except ValueError as error:
            assert "duplicate JSON key" in str(error)
        else:
            raise AssertionError("duplicate JSON key accepted")
        existing = root / "existing"
        existing.mkdir()
        try:
            canonical_absent_output(existing)
        except ValueError as error:
            assert "absent" in str(error)
        else:
            raise AssertionError("existing output accepted")
        output_link = root / "output-link"
        output_link.symlink_to(existing)
        try:
            canonical_absent_output(output_link)
        except ValueError as error:
            assert "absent" in str(error)
        else:
            raise AssertionError("symlink output accepted")
        bad = [{"name": "x", "version": "1", "license": {}, "native_payloads": []}]
        try:
            validate_license_rows(rows, {"packages": bad})
        except ValueError:
            pass
        else:
            raise AssertionError("malformed license row accepted")
    print("xy_tokenizer_reference/audit.py self-test: OK (fake fixtures only)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.project, args.source, args.output)):
            parser.error("--self-test accepts no paths")
        self_test()
        return 0
    if args.project is None or args.output is None:
        parser.error("--project and --output are required")
    audit(args.project, args.source, args.output)
    print(args.output / "dependency_audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
