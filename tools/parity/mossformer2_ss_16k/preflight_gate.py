#!/usr/bin/env python3
"""Offline fail-closed gate for MossFormer2-SS-16K validation."""

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
LOCK_SHA256 = "d82e385ad4658a5c28620ee9b8cb9f0a9c9ff69ccbdc97e492773fe8aaf85b96"
PYPROJECT_SHA256 = "97944f41609032f29287983609c4f9d096441f0faa91c39d8c8118039d6727f3"
PACKAGE_ROWS_SHA256 = "957b75892964fbe7f1517b6f8c67ec23b288e89d769494212ab310bef1ad5c85"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
REVIEW_PLACEHOLDERS = {"", "unresolved", "pending", "pending_review", "owner_review_required", "review_required", "todo", "null", "none"}
MANIFEST_KEYS = {"approval_scope_sha256", "audit_identity", "gate_version", "lock_sha256", "model_reviews", "numeric_bounds", "numeric_state", "operator_approval", "package_rows_sha256", "public_identity", "publication", "pyproject_sha256", "review_rows", "review_rows_sha256", "source_identity", "upstream_identity"}
APPROVAL_KEYS = {"schema", "decision", "signer", "digest"}
EVIDENCE_KEYS = {"schema", "decision", "scope_sha256", "manifest_sha256", "signer", "digest"}

PUBLIC_IDENTITY = {"repo": "vokra/mossformer2-ss-16k", "revision": "0e9ba9258cead4252f8e5279598af296ada08bf7", "file": "mossformer2-ss-16k.gguf", "bytes": 223058240, "sha256": "822516b75873dbeb814dac72f7ca0b5fb75254dd051dfdfdda54987347330f0c", "license": "Apache-2.0"}
UPSTREAM_IDENTITY = {"repo": "alibabasglab/MossFormer2_SS_16K", "revision": "407cb030cd66340918ebb6c8cc63b18f8592cdbe", "file": "last_best_checkpoint.pt", "bytes": 670353271, "sha256": "00a3a48bda492db1e829b85dd443f8f43a43039a3e90f1a24962ea9caf14a11a", "license": None}
SOURCE_IDENTITY = {"repo": "https://github.com/modelscope/ClearerVoice-Studio", "revision": "6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61", "license": None, "files": [{"name": name, "bytes": None, "sha256": None} for name in ("LICENSE", "clearvoice/clearvoice/models/mossformer2_ss/mossformer2.py", "clearvoice/clearvoice/models/mossformer2_ss/mossformer2_block.py", "clearvoice/clearvoice/models/mossformer2_ss/fsmn.py", "clearvoice/clearvoice/models/mossformer2_ss/conv_module.py", "clearvoice/clearvoice/models/mossformer2_ss/layer_norm.py")]}
AUDIT_IDENTITY = {"manifest_sha256": "eb4b366872789b95228a172846259f6aa205a75c678f90941d5e8a3e9a47fb8b", "tensor_count": 1076, "parameter_count": 55735666}
MODEL_REVIEW_IDS = [
    "public-gguf:vokra/mossformer2-ss-16k@0e9ba9258cead4252f8e5279598af296ada08bf7",
    "upstream-checkpoint:alibabasglab/MossFormer2_SS_16K@407cb030cd66340918ebb6c8cc63b18f8592cdbe",
    "official-source:https://github.com/modelscope/ClearerVoice-Studio@6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61",
]
LOCK_KEYS = {"version", "revision", "requires-python", "package"}
PACKAGE_KEYS = {"name", "version", "source", "resolution-markers", "dependencies", "optional-dependencies", "sdist", "wheels", "metadata"}
ARTIFACT_KEYS = {"url", "hash", "size", "upload-time"}
REGISTRY_PACKAGE_SCHEMAS = {
    frozenset({"name", "version", "source", "sdist", "wheels"}),
    frozenset({"name", "version", "source", "dependencies", "sdist", "wheels"}),
    frozenset({"name", "version", "source", "dependencies", "wheels"}),
    frozenset({"name", "version", "source", "wheels"}),
    frozenset({"name", "version", "source", "optional-dependencies", "wheels"}),
}
DEPENDENCY_SCHEMAS = {
    frozenset({"name"}), frozenset({"name", "marker"}), frozenset({"name", "marker", "extra"}),
}
METADATA_REQUIREMENT_SCHEMAS = {frozenset({"name", "specifier"})}


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: Any) -> str:
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def validate_project_schema(project: dict[str, Any]) -> None:
    if set(project) != {"project"} or not isinstance(project.get("project"), dict) or set(project["project"]) != {"name", "version", "description", "requires-python", "dependencies"}:
        raise ValueError("pyproject.toml structural schema drifted")
    p = project["project"]
    if p["requires-python"] != ">=3.12,<3.13" or not isinstance(p["dependencies"], list) or any(not isinstance(x, str) or not x.strip() for x in p["dependencies"]):
        raise ValueError("pyproject.toml project contract drifted")


def load_json(path: Path) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in items:
            if key in out:
                raise ValueError(f"duplicate JSON key: {key}")
            out[key] = value
        return out
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)


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
    if set(lock) != LOCK_KEYS or lock.get("version") != 1 or lock.get("revision") != 3 or lock.get("requires-python") != "==3.12.*":
        raise ValueError("uv.lock top-level schema is malformed")
    packages = lock.get("package")
    if not isinstance(packages, list) or not packages:
        raise ValueError("uv.lock package table is missing or empty")
    rows = []
    identities = set()
    for package in packages:
        if not isinstance(package, dict) or set(package) - PACKAGE_KEYS or not isinstance(package.get("dependencies", []), list) or not isinstance(package.get("resolution-markers", []), list):
            raise ValueError("uv.lock contains malformed package row")
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
        for dependency in package.get("dependencies", []):
            if not isinstance(dependency, dict) or frozenset(dependency) not in DEPENDENCY_SCHEMAS or not isinstance(dependency.get("name"), str) or not dependency["name"].strip():
                raise ValueError("uv.lock dependency row is malformed")
            if "extra" in dependency and (not isinstance(dependency["extra"], list) or any(not isinstance(extra, str) or not extra.strip() for extra in dependency["extra"])):
                raise ValueError("uv.lock dependency extra selector is malformed")
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
        optional_dependencies = package.get("optional-dependencies", {})
        if not isinstance(optional_dependencies, dict):
            raise ValueError("uv.lock optional-dependencies schema is malformed")
        for group, items in optional_dependencies.items():
            if not isinstance(group, str) or not group.strip() or not isinstance(items, list):
                raise ValueError("uv.lock optional-dependencies schema is malformed")
            for item in items:
                if not isinstance(item, dict) or frozenset(item) != frozenset({"name", "marker"}) or not isinstance(item.get("name"), str) or not item["name"].strip():
                    raise ValueError("uv.lock optional dependency row is malformed")
                if not isinstance(item["marker"], str) or not item["marker"].strip():
                    raise ValueError("uv.lock optional dependency marker is malformed")
                for field in ("marker", "version"):
                    if field in item and (not isinstance(item[field], str) or not item[field].strip()):
                        raise ValueError("uv.lock optional dependency field is malformed")
                if "source" in item:
                    item_source = item["source"]
                    if not isinstance(item_source, dict) or len(item_source) != 1 or set(item_source) not in ({"registry"}, {"virtual"}):
                        raise ValueError("uv.lock optional dependency source is malformed")
                    if "registry" in item_source and item_source["registry"] not in ("https://pypi.org/simple", "https://download.pytorch.org/whl/cpu"):
                        raise ValueError("uv.lock optional dependency registry is not approved")
                    if "virtual" in item_source and item_source["virtual"] != ".":
                        raise ValueError("uv.lock optional dependency virtual source is not '.'")
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
                validate_artifact(artifact, f"{(name, version)!r} {artifact_name}", source.get("registry", ""))
        if virtual and ("sdist" in package or "wheels" in package):
            raise ValueError("uv.lock virtual project must not contain artifacts")
        if not virtual and "sdist" not in package and not package.get("wheels"):
            raise ValueError("uv.lock registry package has no authenticated artifacts")
        rows.append({"name": name, "version": version, "source": source, "resolution-markers": package.get("resolution-markers", []), "dependencies": package.get("dependencies", []), "optional-dependencies": optional_dependencies, "sdist": package.get("sdist"), "wheels": package.get("wheels", []), "metadata": metadata})
    if any(not isinstance(row["name"], str) or not isinstance(row["version"], str) for row in rows):
        raise ValueError("uv.lock package row lacks exact name/version")
    return sorted(rows, key=lambda row: (row["name"], row["version"]))


def reviewed(value: Any) -> bool:
    normalized = re.sub(r"\s+", "_", value.strip()).casefold() if isinstance(value, str) else ""
    return normalized not in REVIEW_PLACEHOLDERS


def blocked(reason: str) -> tuple[bool, str]:
    return False, reason


def identities_resolved() -> tuple[bool, str]:
    if not reviewed(UPSTREAM_IDENTITY.get("license")):
        return blocked("upstream checkpoint license/source-role identity is unresolved")
    files = SOURCE_IDENTITY.get("files")
    if not reviewed(SOURCE_IDENTITY.get("license")) or not isinstance(files, list) or not files:
        return blocked("official source license/file identity is unresolved")
    expected = [row["name"] for row in SOURCE_IDENTITY["files"]]
    if [row.get("name") for row in files if isinstance(row, dict)] != expected or len(set(expected)) != len(expected):
        return blocked("official source file identity set is unresolved")
    if any(not isinstance(row, dict) or not isinstance(row.get("bytes"), int) or row["bytes"] <= 0 or not isinstance(row.get("sha256"), str) or not HEX64.fullmatch(row["sha256"]) for row in files):
        return blocked("official source file bytes/SHA identities are unresolved")
    return True, "PASS"


def verify_source(checkout: Path) -> tuple[bool, str]:
    """Verify the pinned source checkout using only stdlib and streaming I/O.

    The fixed source identity is deliberately unresolved today.  This mode is
    still part of the production chain so authenticated byte/SHA rows become
    enforceable before any source import or checkpoint preparation.
    """
    if checkout.is_symlink():
        return blocked("source checkout must not be a symlink")
    try:
        root = checkout.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        return blocked(f"source checkout is unreadable: {exc}")
    if not root.is_dir() or root.is_symlink():
        return blocked("source checkout is not a regular directory")
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        if revision != SOURCE_IDENTITY["revision"]:
            return blocked("official source checkout revision drifted")
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        if status:
            return blocked("official source checkout is not exactly clean")
    except (OSError, subprocess.CalledProcessError) as exc:
        return blocked(f"official source revision is unavailable: {exc}")
    rows = SOURCE_IDENTITY.get("files")
    if not reviewed(SOURCE_IDENTITY.get("license")) or not isinstance(rows, list) or len(rows) != 6:
        return blocked("official source license/file identity is unresolved")
    names = [row.get("name") if isinstance(row, dict) else None for row in rows]
    if len(set(names)) != 6 or any(not isinstance(name, str) or not name for name in names):
        return blocked("official source file identity set is unresolved")
    for row in rows:
        name = row["name"]
        candidate = root / name
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, RuntimeError, ValueError):
            return blocked(f"source file escapes checkout or is missing: {name}")
        if candidate.is_symlink() or not candidate.is_file():
            return blocked(f"source file is not a regular non-symlink file: {name}")
        expected_bytes, expected_sha = row.get("bytes"), row.get("sha256")
        if not isinstance(expected_bytes, int) or expected_bytes <= 0 or not isinstance(expected_sha, str) or not HEX64.fullmatch(expected_sha):
            return blocked(f"source file identity is unresolved: {name}")
        actual_bytes = 0
        hasher = hashlib.sha256()
        try:
            with candidate.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    actual_bytes += len(block)
                    hasher.update(block)
        except OSError as exc:
            return blocked(f"source file cannot be read: {name}: {exc}")
        if actual_bytes != expected_bytes or hasher.hexdigest() != expected_sha:
            return blocked(f"source file bytes/SHA mismatch: {name}")
    return True, "PASS"


def validate(project: Path, manifest_path: Path, evidence_path: Path | None = None) -> tuple[bool, str]:
    lock_path, project_path = project / "uv.lock", project / "pyproject.toml"
    if any(path.is_symlink() or not path.is_file() for path in (lock_path, project_path, manifest_path)):
        return blocked("lock, project, or gate manifest is missing")
    try:
        manifest = load_json(manifest_path)
        lock_bytes, project_bytes = lock_path.read_bytes(), project_path.read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return blocked(f"gate inputs are unreadable: {exc}")
    try:
        validate_project_schema(tomllib.loads(project_bytes.decode("utf-8")))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError) as exc:
        return blocked(f"pyproject.toml schema is invalid: {exc}")
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS or manifest.get("gate_version") != GATE_VERSION:
        return blocked("unsupported gate manifest version")
    if digest(lock_bytes) != LOCK_SHA256 or manifest.get("lock_sha256") != LOCK_SHA256:
        return blocked("uv.lock bytes are not the reviewed exact lock")
    if digest(project_bytes) != PYPROJECT_SHA256 or manifest.get("pyproject_sha256") != PYPROJECT_SHA256:
        return blocked("pyproject bytes are not the reviewed exact project")
    try:
        rows = package_rows(tomllib.loads(lock_bytes.decode()))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError) as exc:
        return blocked(f"uv.lock canonicalization failed: {exc}")
    if canonical(rows) != PACKAGE_ROWS_SHA256 or manifest.get("package_rows_sha256") != PACKAGE_ROWS_SHA256:
        return blocked("canonical package graph drifted")
    reviews = manifest.get("review_rows")
    expected_ids = [f'{row["name"]}@{row["version"]}' for row in rows]
    if not isinstance(reviews, list) or [row.get("id") for row in reviews if isinstance(row, dict)] != expected_ids or len(set(expected_ids)) != len(expected_ids) or any(not isinstance(row, dict) or set(row) != {"id", "status", "license", "native_review", "bundled_review", "evidence"} for row in reviews):
        return blocked("dependency review rows are not the exact schema/set")
    if canonical(reviews) != manifest.get("review_rows_sha256"):
        return blocked("dependency review rows digest drifted")
    if manifest.get("public_identity") != PUBLIC_IDENTITY or manifest.get("upstream_identity") != UPSTREAM_IDENTITY or manifest.get("source_identity") != SOURCE_IDENTITY or manifest.get("audit_identity") != AUDIT_IDENTITY:
        return blocked("fixed model/source/audit identity drifted")
    resolved, reason = identities_resolved()
    if not resolved:
        return False, reason
    if manifest.get("publication") != "NO_UPLOAD" or manifest.get("numeric_state") != "MEASURED_NOT_GATED" or manifest.get("numeric_bounds") != "UNSET":
        return blocked("publication or numeric state drifted")
    model_reviews = manifest.get("model_reviews")
    if not isinstance(model_reviews, list) or [row.get("id") for row in model_reviews if isinstance(row, dict)] != MODEL_REVIEW_IDS or len(set(MODEL_REVIEW_IDS)) != len(MODEL_REVIEW_IDS) or any(not isinstance(row, dict) or set(row) != {"id", "status", "license", "native_review", "bundled_review", "evidence"} for row in model_reviews):
        return blocked("model/source review rows are not the exact schema/set")
    fixed_licenses = {
        MODEL_REVIEW_IDS[0]: PUBLIC_IDENTITY["license"],
        MODEL_REVIEW_IDS[1]: UPSTREAM_IDENTITY.get("license"),
        MODEL_REVIEW_IDS[2]: SOURCE_IDENTITY.get("license"),
    }
    for row in model_reviews:
        fixed_license = fixed_licenses[row["id"]]
        if not reviewed(fixed_license) or row["license"] != fixed_license:
            return blocked(f"model/source review license is not bound to fixed identity: {row['id']}")
    for row in model_reviews + reviews:
        if row["status"] != "REVIEWED" or not all(reviewed(row.get(key)) for key in ("license", "native_review", "bundled_review", "evidence")):
            return blocked(f"review is unresolved: {row['id']}")
    scope = {"lock_sha256": LOCK_SHA256, "pyproject_sha256": PYPROJECT_SHA256, "package_rows_sha256": PACKAGE_ROWS_SHA256, "review_rows": reviews, "model_reviews": model_reviews, "public_identity": PUBLIC_IDENTITY, "upstream_identity": UPSTREAM_IDENTITY, "source_identity": SOURCE_IDENTITY, "audit_identity": AUDIT_IDENTITY, "publication": "NO_UPLOAD", "numeric_state": "MEASURED_NOT_GATED", "numeric_bounds": "UNSET"}
    scope_sha = canonical(scope)
    if manifest.get("approval_scope_sha256") != scope_sha:
        return blocked("operator approval scope is not bound to exact inputs")
    approval = manifest.get("operator_approval")
    if not isinstance(approval, dict) or set(approval) != APPROVAL_KEYS or approval.get("schema") != "v1" or approval.get("decision") != "APPROVED" or not isinstance(approval.get("signer"), str) or not approval["signer"] or approval.get("digest") != scope_sha or not isinstance(approval.get("digest"), str) or not HEX64.fullmatch(approval["digest"]):
        return blocked("operator approval is pending or invalid")
    evidence_path = evidence_path or manifest_path.with_name("license_gate_evidence.json")
    if evidence_path.is_symlink() or not evidence_path.is_file():
        return blocked("authenticated operator evidence is missing")
    try:
        evidence = load_json(evidence_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return blocked(f"operator evidence unreadable: {exc}")
    if not isinstance(evidence, dict) or set(evidence) != EVIDENCE_KEYS or evidence.get("schema") != "v1" or evidence.get("decision") != "APPROVED" or evidence.get("scope_sha256") != scope_sha or evidence.get("manifest_sha256") != digest(manifest_path.read_bytes()) or evidence.get("signer") != approval["signer"] or evidence.get("digest") != approval["digest"]:
        return blocked("authenticated operator evidence is not bound to scope")
    return True, "PASS"


def self_test() -> int:
    project = Path(__file__).resolve().parent
    manifest = project / "license_gate_manifest.json"
    ok, reason = validate(project, manifest)
    if ok or ("unresolved" not in reason and "artifact" not in reason and "canonical package graph drifted" not in reason):
        print(f"mossformer2 gate: expected unresolved production blocker, got {reason}", file=sys.stderr)
        return 1
    if "artifact" in reason or "canonical package graph drifted" in reason:
        valid = {"url": "https://files.pythonhosted.org/packages/demo.whl", "hash": "sha256:" + "0" * 64, "size": 1, "upload-time": "2024-01-01T00:00:00Z"}
        cases = {"missing-size": lambda value: value.pop("size"), "missing-upload-time": lambda value: value.pop("upload-time"), "extra-key": lambda value: value.update(extra="x"), "bool-size": lambda value: value.update(size=True), "wrong-host": lambda value: value.update(url="https://example.invalid/demo.whl")}
        for label, mutate in cases.items():
            candidate = dict(valid); mutate(candidate)
            try:
                validate_artifact(candidate, f"self-test {label}", "https://pypi.org/simple")
            except ValueError:
                pass
            else:
                print(f"mossformer2 artifact tamper accepted: {label}", file=sys.stderr); return 1
        try:
            package_rows({"version": 1, "revision": 3, "requires-python": "==3.12.*", "package": [
                {"name": "demo", "version": "0", "source": {"virtual": "."}, "dependencies": [], "metadata": {"requires-dist": []}},
                {"name": "registry-demo", "version": "1", "source": {"registry": "https://pypi.org/simple"}, "dependencies": []},
            ]})
        except ValueError:
            pass
        else:
            print("mossformer2 registry package without artifacts accepted", file=sys.stderr); return 1
        print("mossformer2 gate: self-test PASS (production artifact schema blocker)")
        return 0
    if reviewed(" PENDING_REVIEW ") or not reviewed("authenticated citation: TODO was resolved"):
        print("mossformer2 gate: placeholder normalization failed", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory(prefix="mossformer2-gate-") as directory:
        root = Path(directory); test_project = root / "project"; test_project.mkdir()
        shutil.copy2(project / "uv.lock", test_project / "uv.lock"); shutil.copy2(project / "pyproject.toml", test_project / "pyproject.toml")
        duplicate_manifest = root / "duplicate-manifest.json"
        duplicate_manifest.write_text('{"gate_version":1,"gate_version":1}')
        try:
            load_json(duplicate_manifest)
        except ValueError as exc:
            if "duplicate JSON key" not in str(exc):
                print("mossformer2 duplicate manifest key reported the wrong error", file=sys.stderr); return 1
        else:
            print("mossformer2 duplicate manifest key was accepted", file=sys.stderr); return 1
        duplicate_evidence = root / "duplicate-evidence.json"
        duplicate_evidence.write_text('{"schema":"v1","schema":"v1"}')
        try:
            load_json(duplicate_evidence)
        except ValueError as exc:
            if "duplicate JSON key" not in str(exc):
                print("mossformer2 duplicate evidence key reported the wrong error", file=sys.stderr); return 1
        else:
            print("mossformer2 duplicate evidence key was accepted", file=sys.stderr); return 1
        candidate = json.loads(manifest.read_text())
        candidate["upstream_identity"]["license"] = "Apache-2.0"
        candidate["source_identity"]["license"] = "Apache-2.0"
        for row in candidate["source_identity"]["files"]:
            row["bytes"], row["sha256"] = 1, "0" * 64
        for row in candidate["review_rows"] + candidate["model_reviews"]:
            row.update({"status": "REVIEWED", "license": "SELF_TEST", "native_review": "SELF_TEST", "bundled_review": "SELF_TEST", "evidence": "SELF_TEST"})
        candidate["review_rows_sha256"] = canonical(candidate["review_rows"])
        candidate_path = root / "manifest.json"; candidate_path.write_text(json.dumps(candidate))
        if validate(test_project, candidate_path)[0]:
            print("mossformer2 gate: unresolved identity bypass accepted", file=sys.stderr)
            return 1
        source_probe = root / "source"
        source_probe.mkdir()
        source_ok, source_reason = verify_source(source_probe)
        if source_ok or not any(token in source_reason for token in ("unresolved", "unavailable")):
            print("mossformer2 gate: unresolved source verification bypass accepted", file=sys.stderr)
            return 1
    print("mossformer2 gate: self-test PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent); parser.add_argument("--manifest", type=Path); parser.add_argument("--evidence", type=Path); parser.add_argument("--verify-source", type=Path); parser.add_argument("--self-test", action="store_true"); args = parser.parse_args()
    if args.self_test: return self_test()
    if args.verify_source:
        ok, reason = verify_source(args.verify_source)
        if not ok:
            print(f"mossformer2 source verification: BLOCKED: {reason}", file=sys.stderr)
            return 2
        print("mossformer2 source verification: PASS")
        return 0
    ok, reason = validate(args.project, args.manifest or args.project / "license_gate_manifest.json", args.evidence)
    if not ok: print(f"mossformer2 preflight gate: BLOCKED: {reason}", file=sys.stderr); return 2
    print("mossformer2 preflight gate: PASS"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
