#!/usr/bin/env python3
"""Fail-closed MMS-1B-All staging closure gate.

This module deliberately contains no model/runtime imports.  The committed
tree currently has no authenticated MMS-specific resolver lock, so a normal
run must stop before a cache, host probe, or network operation.  Once an
owner supplies a dedicated lock and manifest, this gate verifies their exact
schemas, artifact provenance, package/native review rows, and approval scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

GATE_VERSION = 1
REPOSITORY = "facebook/mms-1b-all"
REVISION = "3d33597edbdaaba14a8e858e2c8caa76e3cec0cd"
MODEL = "mms-1b-all"
LICENSE = "cc-by-nc-4.0"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
LANGUAGE = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")
PLACEHOLDERS = {"", "null", "none", "pending", "pending_review", "unresolved", "todo", "owner_signoff_required", "owner_review_required", "review_required"}
REGISTRIES = {"https://pypi.org/simple", "https://download.pytorch.org/whl/cpu"}
MANIFEST_KEYS = {
    "gate_version", "lock_sha256", "project_sha256", "package_rows",
    "package_rows_sha256", "package_review_rows", "package_review_rows_sha256",
    "identities", "license_rows", "license_rows_sha256", "publication_decision",
}
REVIEW_KEYS = {"name", "version", "source", "status", "license", "native_bundled_review"}
LICENSE_ROW_KEYS = {"id", "status", "license", "conclusion", "evidence"}
APPROVAL_KEYS = {
    "schema", "model", "upstream_repo", "upstream_revision", "language",
    "license_spdx", "project_sha256", "lock_sha256", "manifest_sha256",
    "no_upload", "decision", "signer", "scope_sha256",
}
REFERENCE_KEYS = {
    "contract", "repository", "revision", "resolved_snapshot", "language",
    "composition", "selected_vocabulary", "source_files", "transformers_source",
    "runtime", "state_dict_tensor_manifest", "logits_shape", "logits_finite",
    "logits_nonzero", "logits_dtype", "greedy_token_ids_sha256", "decoded_text", "license",
    "runtime_status", "parity_status", "tolerance",
}
PREPARED_KEYS = {"contract", "repository", "revision", "language", "source_files", "composition", "license", "runtime_status", "parity_status"}


def blocked(message: str) -> None:
    print(f"mms-1b-all license gate: BLOCKED: {message}", file=sys.stderr)
    raise SystemExit(2)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canon(value: Any) -> str:
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def regular_file(path: Path) -> bool:
    absolute = Path(os.path.abspath(path))
    try:
        return path.is_file() and not path.is_symlink() and all(not p.is_symlink() for p in absolute.parents)
    except OSError:
        return False


def load_json(path: Path) -> Any:
    def reject(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)


def resolved(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().casefold() not in PLACEHOLDERS


def artifact(value: Any, label: str, registry: str) -> None:
    if not isinstance(value, dict) or set(value) != {"url", "hash", "size", "upload-time"}:
        raise ValueError(f"{label} artifact schema is not exact")
    url = value["url"] if isinstance(value["url"], str) else ""
    parsed = urlsplit(url)
    expected_host = "download.pytorch.org" if registry == "https://download.pytorch.org/whl/cpu" else "files.pythonhosted.org"
    if parsed.scheme != "https" or parsed.netloc != expected_host or not parsed.path or parsed.query or parsed.fragment:
        raise ValueError(f"{label} registry host is not authenticated")
    if not isinstance(value["hash"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value["hash"]):
        raise ValueError(f"{label} artifact hash is malformed")
    if isinstance(value["size"], bool) or not isinstance(value["size"], int) or value["size"] <= 0:
        raise ValueError(f"{label} artifact size is malformed")
    if not isinstance(value["upload-time"], str) or not value["upload-time"].strip():
        raise ValueError(f"{label} artifact upload-time is missing")


def lock_rows(lock: dict[str, Any]) -> list[dict[str, Any]]:
    if set(lock) != {"version", "revision", "requires-python", "resolution-markers", "supported-markers", "package"}:
        raise ValueError("lock top-level schema is not exact")
    if type(lock["version"]) is not int or type(lock["revision"]) is not int or lock["version"] != 1 or lock["revision"] != 3 or lock["requires-python"] != "==3.12.*":
        raise ValueError("lock version/python contract is not exact")
    if any(not isinstance(lock[key], list) or any(not isinstance(x, str) or not x.strip() for x in lock[key]) for key in ("resolution-markers", "supported-markers")):
        raise ValueError("lock marker schema is malformed")
    packages = lock["package"]
    if not isinstance(packages, list) or not packages:
        raise ValueError("dedicated lock package table is missing or empty")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    virtual = 0
    allowed = {"name", "version", "source", "resolution-markers", "dependencies", "optional-dependencies", "sdist", "wheels", "metadata"}
    for package in packages:
        if not isinstance(package, dict) or set(package) - allowed:
            raise ValueError("lock package row contains unknown fields")
        name, version, source = package.get("name"), package.get("version"), package.get("source")
        if not isinstance(name, str) or not name.strip() or not isinstance(version, str) or not version.strip():
            raise ValueError("lock package identity is malformed")
        if re.search(r"cuda|nvidia|triton", name, re.I):
            raise ValueError("CUDA/NVIDIA/Triton package is forbidden")
        key = (name, version)
        if key in seen:
            raise ValueError(f"duplicate lock package identity: {key!r}")
        seen.add(key)
        if not isinstance(source, dict) or len(source) != 1 or set(source) not in ({"registry"}, {"virtual"}):
            raise ValueError("lock package source is malformed")
        if "registry" in source and source["registry"] not in REGISTRIES:
            raise ValueError("lock registry is not approved")
        if "virtual" in source:
            virtual += 1
            if source["virtual"] != "." or set(package) != {"name", "version", "source", "dependencies", "metadata"}:
                raise ValueError("virtual project package schema is not exact")
            if not isinstance(package["metadata"], dict) or set(package["metadata"]) != {"requires-dist"} or not isinstance(package["metadata"]["requires-dist"], list):
                raise ValueError("virtual project metadata schema is not exact")
            for requirement in package["metadata"]["requires-dist"]:
                if not isinstance(requirement, dict) or set(requirement) not in ({"name", "specifier"}, {"name", "specifier", "index"}) or not isinstance(requirement.get("name"), str) or not requirement["name"].strip() or not isinstance(requirement.get("specifier"), str) or not requirement["specifier"].strip():
                    raise ValueError("virtual project requirement schema is not exact")
                if "index" in requirement and requirement["index"] not in REGISTRIES:
                    raise ValueError("virtual requirement index is not an approved CPU registry")
        markers = package.get("resolution-markers", [])
        if not isinstance(markers, list) or any(not isinstance(x, str) or not x.strip() for x in markers):
            raise ValueError("package resolution-markers are malformed")
        dependencies = package.get("dependencies", [])
        if not isinstance(dependencies, list):
            raise ValueError("package dependencies are malformed")
        for dep in dependencies:
            if not isinstance(dep, dict) or set(dep) - {"name", "marker", "extra"} or not isinstance(dep.get("name"), str) or not dep["name"].strip() or ("marker" in dep and (not isinstance(dep["marker"], str) or not dep["marker"].strip())) or ("extra" in dep and (not isinstance(dep["extra"], list) or any(not isinstance(x, str) or not x.strip() for x in dep["extra"]))):
                raise ValueError("package dependency row is malformed")
            if re.search(r"cuda|nvidia|triton", dep["name"], re.I):
                raise ValueError("CUDA/NVIDIA/Triton dependency is forbidden")
        for kind in ("sdist", "wheels"):
            if kind not in package:
                continue
            values = package[kind] if kind == "wheels" else [package[kind]]
            if kind == "wheels" and (not isinstance(values, list) or not values):
                raise ValueError("wheel artifact table is malformed")
            if kind == "sdist" and not isinstance(package[kind], dict):
                raise ValueError("sdist artifact is malformed")
            for item in values:
                artifact(item, f"{name} {kind}", source.get("registry", ""))
        if "optional-dependencies" in package:
            optional = package["optional-dependencies"]
            if not isinstance(optional, dict) or any(not isinstance(group, str) or not group.strip() or not isinstance(values, list) or any(not isinstance(dep, dict) or set(dep) != {"name", "marker"} or not isinstance(dep["name"], str) or not dep["name"].strip() or not isinstance(dep["marker"], str) or not dep["marker"].strip() for dep in values) for group, values in optional.items()):
                raise ValueError("optional dependency schema is malformed")
        if "virtual" not in source and not package.get("sdist") and not package.get("wheels"):
            raise ValueError(f"package {key!r} has no authenticated artifact")
        rows.append(package)
    if virtual != 1:
        raise ValueError("dedicated lock must contain exactly one virtual project")
    return sorted(rows, key=lambda row: (row["name"], row["version"]))


def project_schema(project: dict[str, Any]) -> None:
    if set(project) != {"project", "tool"} or not isinstance(project["project"], dict) or not isinstance(project["tool"], dict):
        raise ValueError("dedicated pyproject schema is not exact")
    metadata = project["project"]
    expected = {"name", "version", "description", "requires-python", "dependencies"}
    if set(metadata) != expected or metadata["name"] != "vokra-mms-1b-all-parity" or metadata["version"] != "0.1.0" or metadata["requires-python"] != "==3.12.*" or not isinstance(metadata.get("description"), str) or not metadata["description"].strip() or not isinstance(metadata["dependencies"], list) or not metadata["dependencies"] or any(not isinstance(item, str) or not item.strip() for item in metadata["dependencies"]):
        raise ValueError("dedicated pyproject identity/dependencies are unresolved")
    uv = project["tool"].get("uv")
    if not isinstance(uv, dict) or set(uv) != {"package", "environments", "sources", "index"} or uv.get("package") is not False or uv.get("environments") != ["sys_platform == 'linux' and platform_machine == 'x86_64'"] or not isinstance(uv.get("sources"), dict) or not isinstance(uv.get("index"), list) or not uv["index"]:
        raise ValueError("dedicated project is not Linux x86_64 CPU-only")
    index_names = set()
    for index in uv["index"]:
        if not isinstance(index, dict) or set(index) != {"name", "url", "explicit"} or index["url"] not in REGISTRIES or index["explicit"] is not True:
            raise ValueError("dedicated project registry schema is not exact")
        if not isinstance(index["name"], str) or not index["name"].strip() or index["name"] in index_names:
            raise ValueError("dedicated project registry name is malformed")
        index_names.add(index["name"])
    if any(not isinstance(name, str) or not name.strip() or not isinstance(source, dict) or set(source) != {"index"} or source["index"] not in index_names for name, source in uv["sources"].items()):
        raise ValueError("dedicated project source mapping is not exact")
    if re.search(r"cuda|nvidia|triton", json.dumps(project, sort_keys=True), re.I):
        raise ValueError("CUDA/NVIDIA/Triton dependency is forbidden")


def validate_tensor_manifest(value: Any, label: str) -> None:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} tensor manifest is missing")
    for name, row in value.items():
        if not isinstance(name, str) or not name or not isinstance(row, dict) or set(row) != {"shape", "dtype"} or not isinstance(row["shape"], list) or any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in row["shape"]) or not isinstance(row["dtype"], str) or not re.fullmatch(r"torch\.(?:float16|float32|bfloat16|int8|int16|int32|int64|bool)", row["dtype"]):
            raise ValueError(f"{label} tensor row is malformed: {name!r}")


def validate_reference(path: Path) -> None:
    if not regular_file(path):
        raise ValueError("reference manifest is missing or symlinked")
    value = load_json(path)
    if not isinstance(value, dict) or set(value) != REFERENCE_KEYS:
        raise ValueError("reference manifest schema is not exact")
    if value["contract"] != "vokra-mms-1b-all-backbone-adapter-v1" or value["repository"] != REPOSITORY or value["revision"] != REVISION or value["composition"] != "AutoProcessor.from_pretrained(target_lang=language) + Wav2Vec2ForCTC.from_pretrained(target_lang=language)":
        raise ValueError("reference identity drifted")
    language = value["language"]
    if not isinstance(language, str) or not LANGUAGE.fullmatch(language):
        raise ValueError("reference language is malformed")
    if not isinstance(value["resolved_snapshot"], str):
        raise ValueError("reference snapshot is not a string")
    snapshot = Path(value["resolved_snapshot"])
    if not snapshot.is_absolute() or snapshot.name != REVISION or any(part in {".", ".."} for part in snapshot.parts) or not snapshot.is_dir() or any(part.is_symlink() for part in (snapshot, *snapshot.parents)):
        raise ValueError("reference snapshot is not an absolute pinned symlink-free directory")
    if value["license"] != LICENSE or value["runtime_status"] != "INSPECTION_ONLY" or value["parity_status"] != "INSPECTION_ONLY" or value["tolerance"] is not None:
        raise ValueError("reference is not inspection-only")
    selected = value["selected_vocabulary"]
    if not isinstance(selected, dict) or set(selected) != {"path", "sha256", "sidecar_path", "sidecar_sha256", "labels"} or selected["path"] != f"vocab.json[{language}]" or selected["sidecar_path"] != f"vocabs/{language}.txt" or not HEX64.fullmatch(str(selected["sha256"])) or not HEX64.fullmatch(str(selected["sidecar_sha256"])) or isinstance(selected["labels"], bool) or not isinstance(selected["labels"], int) or selected["labels"] <= 0:
        raise ValueError("reference vocabulary identity is malformed")
    if not isinstance(value["source_files"], dict) or set(value["source_files"]) != {"config.json", "preprocessor_config.json", "model.safetensors", "vocab.json", f"adapter.{language}.safetensors", f"vocabs/{language}.txt"}:
        raise ValueError("reference source file set is not exact")
    for label, row in value["source_files"].items():
        if not isinstance(row, dict) or set(row) != {"sha256", "bytes"} or not HEX64.fullmatch(str(row["sha256"])) or isinstance(row["bytes"], bool) or not isinstance(row["bytes"], int) or row["bytes"] <= 0:
            raise ValueError(f"reference source file row is malformed: {label}")
    source = value["transformers_source"]
    source_path = Path(source["path"]) if isinstance(source, dict) and isinstance(source.get("path"), str) else Path(".")
    if not isinstance(source, dict) or set(source) != {"path", "sha256"} or not source_path.is_absolute() or not source_path.is_file() or source_path.is_symlink() or any(parent.is_symlink() for parent in source_path.parents) or not isinstance(source["sha256"], str) or not HEX64.fullmatch(source["sha256"]):
        raise ValueError("reference Transformers source identity is malformed")
    runtime = value["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != {"python", "platform", "torch", "transformers"} or any(not isinstance(runtime[key], str) or not runtime[key].strip() for key in runtime):
        raise ValueError("reference runtime schema is malformed")
    validate_tensor_manifest(value["state_dict_tensor_manifest"], "reference")
    if not isinstance(value["logits_shape"], list) or len(value["logits_shape"]) != 3 or any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in value["logits_shape"]) or not isinstance(value["logits_finite"], bool) or not isinstance(value["logits_nonzero"], bool) or not value["logits_finite"] or not value["logits_nonzero"] or value["logits_dtype"] != "torch.float32" or not HEX64.fullmatch(str(value["greedy_token_ids_sha256"])) or not isinstance(value["decoded_text"], str):
        raise ValueError("reference output schema is malformed")


def validate_prepared(path: Path) -> None:
    if not regular_file(path):
        raise ValueError("prepared manifest is missing or symlinked")
    value = load_json(path)
    if not isinstance(value, dict) or set(value) != PREPARED_KEYS:
        raise ValueError("prepared manifest schema is not exact")
    if value["contract"] != "vokra-mms-1b-all-backbone-adapter-v1" or value["repository"] != REPOSITORY or value["revision"] != REVISION or value["composition"] != "UNAUTHENTICATED; compare official Transformers composed state_dict before conversion" or value["license"] != LICENSE or value["runtime_status"] != "INSPECTION_ONLY" or value["parity_status"] != "INSPECTION_ONLY":
        raise ValueError("prepared manifest identity/status drifted")
    language = value["language"]
    if not isinstance(language, str) or not LANGUAGE.fullmatch(language):
        raise ValueError("prepared language is malformed")
    expected = {"model.safetensors", f"adapter.{language}.safetensors", f"vocabs/{language}.txt", "vocab.json"}
    files = value["source_files"]
    if not isinstance(files, dict) or set(files) != expected:
        raise ValueError("prepared source file set is not exact")
    for label, row in files.items():
        expected_keys = {"sha256", "bytes", "tensor_manifest"} if label in {"model.safetensors", f"adapter.{language}.safetensors"} else {"sha256", "bytes"} if label == f"vocabs/{language}.txt" else {"sha256", "selected_labels"}
        if not isinstance(row, dict) or set(row) != expected_keys or not isinstance(row.get("sha256"), str) or not HEX64.fullmatch(row["sha256"]):
            raise ValueError(f"prepared source hash is malformed: {label}")
        if label in {"model.safetensors", f"adapter.{language}.safetensors"}:
            if isinstance(row["bytes"], bool) or not isinstance(row["bytes"], int) or row["bytes"] <= 0:
                raise ValueError(f"prepared source size is malformed: {label}")
            validate_tensor_manifest(row["tensor_manifest"], f"prepared {label}")
        else:
            if label == f"vocabs/{language}.txt" and (isinstance(row.get("bytes"), bool) or not isinstance(row.get("bytes"), int) or row["bytes"] <= 0):
                raise ValueError(f"prepared source size is malformed: {label}")
            if label == "vocab.json" and (isinstance(row.get("selected_labels"), bool) or not isinstance(row.get("selected_labels"), int) or row["selected_labels"] <= 0):
                raise ValueError(f"prepared vocabulary label count is malformed: {label}")


def approval_scope(value: dict[str, Any]) -> str:
    return canon({key: value[key] for key in ("schema", "model", "upstream_repo", "upstream_revision", "language", "license_spdx", "project_sha256", "lock_sha256", "manifest_sha256", "no_upload", "decision")})


def validate_evidence_bindings(identities: dict[str, Any], prepared: dict[str, Any], reference: dict[str, Any], explicit_language: str) -> None:
    if prepared["language"] != explicit_language or reference["language"] != explicit_language:
        raise ValueError("evidence language does not match explicit language")
    for key, source_path in (("backbone", "model.safetensors"), ("adapter", f"adapter.{explicit_language}.safetensors"), ("vocabulary", f"vocabs/{explicit_language}.txt")):
        identity = identities[key]
        prepared_row = prepared["source_files"][source_path]
        reference_row = reference["source_files"][source_path]
        if any(row["sha256"] != identity["sha256"] or row["bytes"] != identity["bytes"] for row in (prepared_row, reference_row)):
            raise ValueError(f"{source_path} identity differs across closure evidence")


def run(lock_path: Path, project_path: Path, manifest_path: Path, approval_path: Path, reference_path: Path | None, prepared_path: Path | None, explicit_language: str) -> None:
    for path, label in ((lock_path, "dedicated uv.lock"), (project_path, "dedicated pyproject"), (manifest_path, "closure manifest"), (approval_path, "approval evidence")):
        if not regular_file(path):
            blocked(f"{label} is missing; authenticated MMS closure is not committed")
    try:
        lock_bytes = lock_path.read_bytes()
        project_bytes = project_path.read_bytes()
        lock = tomllib.loads(lock_bytes.decode("utf-8"))
        project = tomllib.loads(project_bytes.decode("utf-8"))
        manifest = load_json(manifest_path)
        approval = load_json(approval_path)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, json.JSONDecodeError, ValueError) as error:
        blocked(f"closure input is unreadable: {error}")
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS or manifest.get("gate_version") != GATE_VERSION:
        blocked("closure manifest schema is not exact")
    if not isinstance(approval, dict) or set(approval) != APPROVAL_KEYS:
        blocked("approval evidence schema is not exact")
    if not LANGUAGE.fullmatch(explicit_language):
        blocked("explicit language adapter is malformed")
    try:
        project_schema(project)
        rows = lock_rows(lock)
    except (KeyError, TypeError, ValueError) as error:
        blocked(str(error))
    lock_digest, project_digest = sha(lock_bytes), sha(project_bytes)
    if manifest.get("lock_sha256") != lock_digest or manifest.get("project_sha256") != project_digest or not HEX64.fullmatch(lock_digest) or not HEX64.fullmatch(project_digest):
        blocked("manifest does not bind exact project/lock bytes")
    if manifest.get("package_rows") != rows or manifest.get("package_rows_sha256") != canon(rows):
        blocked("canonical package rows drifted")
    reviews = manifest.get("package_review_rows")
    if not isinstance(reviews, list) or len(reviews) != len(rows) or manifest.get("package_review_rows_sha256") != canon(reviews):
        blocked("every locked package needs an exact review row")
    actual = {(row["name"], row["version"]): row for row in rows}
    seen: set[tuple[str, str]] = set()
    for review in reviews:
        if not isinstance(review, dict) or set(review) != REVIEW_KEYS:
            blocked("package review row schema is not exact")
        key = (review.get("name"), review.get("version"))
        if key in seen or key not in actual or review.get("source") != actual[key]["source"] or review.get("status") != "REVIEWED" or not resolved(review.get("license")) or not resolved(review.get("native_bundled_review")):
            blocked(f"package/native review is unresolved or unbound: {key!r}")
        seen.add(key)
    if seen != set(actual):
        blocked("package review rows do not cover exact closure")
    identities = manifest.get("identities")
    if not isinstance(identities, dict) or set(identities) != {"repository", "revision", "language", "backbone", "adapter", "vocabulary"} or identities["repository"] != REPOSITORY or identities["revision"] != REVISION or identities["language"] != explicit_language or identities["language"] != approval.get("language"):
        blocked("backbone/one-adapter identity schema is not exact")
    for key in ("backbone", "adapter", "vocabulary"):
        row = identities[key]
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"} or not isinstance(row["path"], str) or not row["path"] or isinstance(row["bytes"], bool) or not isinstance(row["bytes"], int) or row["bytes"] <= 0 or not HEX64.fullmatch(str(row["sha256"])):
            blocked(f"{key} identity is malformed")
    if identities["backbone"]["path"] != "model.safetensors" or identities["adapter"]["path"] != f"adapter.{explicit_language}.safetensors" or identities["vocabulary"]["path"] != f"vocabs/{explicit_language}.txt":
        blocked("backbone/adapter/vocabulary paths are not separated")
    licenses = manifest.get("license_rows")
    if not isinstance(licenses, list) or len(licenses) != 3 or [row.get("id") for row in licenses if isinstance(row, dict)] != ["source-license", "weights-license", "python-closure"]:
        blocked("license review rows are missing, duplicated, reordered, or extra")
    if any(not isinstance(row, dict) or set(row) != LICENSE_ROW_KEYS or row.get("status") != "REVIEWED" or not resolved(row.get("license")) or not resolved(row.get("conclusion")) or not resolved(row.get("evidence")) for row in licenses) or manifest.get("license_rows_sha256") != canon(licenses):
        blocked("license/native bundled review is unresolved")
    if manifest.get("publication_decision") != "NO_UPLOAD":
        blocked("publication decision is not NO_UPLOAD")
    if approval.get("schema") != "vokra-mms-1b-all-approval-v1" or approval.get("model") != MODEL or approval.get("upstream_repo") != REPOSITORY or approval.get("upstream_revision") != REVISION or approval.get("license_spdx") != LICENSE or approval.get("project_sha256") != project_digest or approval.get("lock_sha256") != lock_digest or approval.get("no_upload") is not True or approval.get("decision") != "APPROVED" or not isinstance(approval.get("language"), str) or not LANGUAGE.fullmatch(approval["language"]) or not isinstance(approval.get("manifest_sha256"), str) or not HEX64.fullmatch(approval["manifest_sha256"]) or approval["manifest_sha256"] != sha(manifest_path.read_bytes()) or not isinstance(approval.get("signer"), str) or not approval["signer"].strip() or approval["signer"].strip().casefold() in PLACEHOLDERS or approval.get("scope_sha256") != approval_scope(approval):
        blocked("owner approval does not bind exact noncommercial NO_UPLOAD scope")
    prepared = None
    reference = None
    if prepared_path is not None:
        try:
            validate_prepared(prepared_path)
            prepared = load_json(prepared_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            blocked(f"prepared manifest is invalid: {error}")
        if prepared["language"] != explicit_language:
            blocked("prepared language does not match explicit language")
    if reference_path is not None:
        try:
            validate_reference(reference_path)
            reference = load_json(reference_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            blocked(f"reference manifest is invalid: {error}")
        if reference["language"] != explicit_language:
            blocked("reference language does not match explicit language")
    if prepared is not None and reference is not None:
        try:
            validate_evidence_bindings(identities, prepared, reference, explicit_language)
        except (KeyError, TypeError, ValueError) as error:
            blocked(str(error))
    print("mms-1b-all license gate: PASS")


def self_test() -> None:
    assert load_json.__name__ == "load_json"
    for value in (None, "", "TODO", "OWNER_SIGNOFF_REQUIRED", "pending_review"):
        assert not resolved(value)
    assert resolved("owner signoff recorded in external evidence")
    assert LANGUAGE.fullmatch("eng") and LANGUAGE.fullmatch("azj-script_cyrillic")
    assert not LANGUAGE.fullmatch("eng/../x")
    valid_artifact = {"url": "https://files.pythonhosted.org/packages/x.whl", "hash": "sha256:" + "a" * 64, "size": 1, "upload-time": "2026-01-01T00:00:00Z"}
    artifact(valid_artifact, "self-test", "https://pypi.org/simple")
    for suffix in ("?query=1", "#fragment"):
        candidate = dict(valid_artifact); candidate["url"] += suffix
        try:
            artifact(candidate, "self-test", "https://pypi.org/simple")
        except ValueError:
            pass
        else:
            raise SystemExit("self-test accepted artifact URL query/fragment")
    virtual = {"name": "demo", "version": "0.1.0", "source": {"virtual": "."}, "dependencies": [], "metadata": {"requires-dist": []}}
    lock_shape = {"version": 1, "revision": 3, "requires-python": "==3.12.*", "resolution-markers": [], "supported-markers": [], "package": [virtual]}
    lock_rows(lock_shape)
    for field, bad in (("version", True), ("resolution-markers", "not-a-list")):
        candidate = dict(lock_shape); candidate[field] = bad
        try:
            lock_rows(candidate)
        except (TypeError, ValueError):
            pass
        else:
            raise SystemExit(f"self-test accepted malformed lock field: {field}")
    with __import__("tempfile").TemporaryDirectory(dir="/private/tmp") as directory:
        root = Path(directory)
        duplicate = root / "duplicate.json"
        duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
        try:
            load_json(duplicate)
        except ValueError:
            pass
        else:
            raise SystemExit("self-test accepted duplicate JSON key")
        target = root / "manifest.json"
        target.write_text("[]", encoding="utf-8")
        try:
            value = load_json(target)
        except Exception as error:  # pragma: no cover - parser must not crash gate
            raise SystemExit(f"self-test parser crashed: {error}") from error
        if isinstance(value, dict):
            raise SystemExit("self-test accepted non-object as a generic manifest")
        link_parent = root / "link-parent"
        link_parent.mkdir()
        link = root / "link"
        link.symlink_to(link_parent, target_is_directory=True)
        if regular_file(link / "missing.json"):
            raise SystemExit("self-test accepted symlinked ancestry")
        snapshot = root / REVISION
        snapshot.mkdir()
        digest = "a" * 64
        source_files = {name: {"sha256": digest, "bytes": 1} for name in ("config.json", "preprocessor_config.json", "model.safetensors", "vocab.json", "adapter.eng.safetensors", "vocabs/eng.txt")}
        source_file = root / "Wav2Vec2ForCTC.py"; source_file.write_text("source", encoding="utf-8")
        reference = {"contract": "vokra-mms-1b-all-backbone-adapter-v1", "repository": REPOSITORY, "revision": REVISION, "resolved_snapshot": str(snapshot), "language": "eng", "composition": "AutoProcessor.from_pretrained(target_lang=language) + Wav2Vec2ForCTC.from_pretrained(target_lang=language)", "selected_vocabulary": {"path": "vocab.json[eng]", "sha256": digest, "sidecar_path": "vocabs/eng.txt", "sidecar_sha256": digest, "labels": 1}, "source_files": source_files, "transformers_source": {"path": str(source_file), "sha256": digest}, "runtime": {"python": "3.12.0", "platform": "Linux", "torch": "2.0", "transformers": "5.0"}, "state_dict_tensor_manifest": {"weight": {"shape": [1], "dtype": "torch.float32"}}, "logits_shape": [1, 2, 3], "logits_finite": True, "logits_nonzero": True, "logits_dtype": "torch.float32", "greedy_token_ids_sha256": digest, "decoded_text": "", "license": LICENSE, "runtime_status": "INSPECTION_ONLY", "parity_status": "INSPECTION_ONLY", "tolerance": None}
        reference_path = root / "reference.json"
        reference_path.write_text(json.dumps(reference), encoding="utf-8")
        validate_reference(reference_path)
        for field, bad in (("composition", "unverified"), ("transformers_source", {"path": "x", "sha256": "bad"}), ("logits_shape", [1, 2, 3, 4])):
            candidate = json.loads(json.dumps(reference)); candidate[field] = bad; reference_path.write_text(json.dumps(candidate), encoding="utf-8")
            try:
                validate_reference(reference_path)
            except ValueError:
                pass
            else:
                raise SystemExit(f"self-test accepted reference tamper: {field}")
        reference_path.write_text(json.dumps(reference), encoding="utf-8")
        prepared = {"contract": "vokra-mms-1b-all-backbone-adapter-v1", "repository": REPOSITORY, "revision": REVISION, "language": "eng", "source_files": {"model.safetensors": {"sha256": digest, "bytes": 1, "tensor_manifest": {"weight": {"shape": [1], "dtype": "torch.float32"}}}, "adapter.eng.safetensors": {"sha256": digest, "bytes": 1, "tensor_manifest": {"weight": {"shape": [1], "dtype": "torch.float32"}}}, "vocabs/eng.txt": {"sha256": digest, "bytes": 1}, "vocab.json": {"sha256": digest, "selected_labels": 1}}, "composition": "UNAUTHENTICATED; compare official Transformers composed state_dict before conversion", "license": LICENSE, "runtime_status": "INSPECTION_ONLY", "parity_status": "INSPECTION_ONLY"}
        prepared_path = root / "prepared.json"; prepared_path.write_text(json.dumps(prepared), encoding="utf-8"); validate_prepared(prepared_path)
        identities = {"backbone": {"path": "model.safetensors", "bytes": 1, "sha256": digest}, "adapter": {"path": "adapter.eng.safetensors", "bytes": 1, "sha256": digest}, "vocabulary": {"path": "vocabs/eng.txt", "bytes": 1, "sha256": digest}}
        validate_evidence_bindings(identities, prepared, reference, "eng")
        mismatched = json.loads(json.dumps(reference)); mismatched["source_files"]["model.safetensors"]["sha256"] = "b" * 64
        try:
            validate_evidence_bindings(identities, prepared, mismatched, "eng")
        except ValueError:
            pass
        else:
            raise SystemExit("self-test accepted cross-evidence hash replacement")
        mismatched = json.loads(json.dumps(prepared)); mismatched["language"] = "spa"
        try:
            validate_evidence_bindings(identities, mismatched, reference, "eng")
        except ValueError:
            pass
        else:
            raise SystemExit("self-test accepted cross-evidence language replacement")
        candidate = json.loads(json.dumps(prepared)); candidate["source_files"]["adapter.eng.safetensors"]["tensor_manifest"]["weight"] = {"shape": [1]}; prepared_path.write_text(json.dumps(candidate), encoding="utf-8")
        try:
            validate_prepared(prepared_path)
        except ValueError:
            pass
        else:
            raise SystemExit("self-test accepted prepared tensor-row tamper")
    print("mms_1b_all license gate self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--approval-evidence", type=Path)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--prepared-manifest", type=Path)
    parser.add_argument("--language")
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.lock, args.project, args.manifest, args.approval_evidence, args.reference_manifest, args.prepared_manifest, args.language)):
            parser.error("--self-test accepts no other arguments")
        self_test()
        return 0
    if any(value is None for value in (args.lock, args.project, args.manifest, args.approval_evidence)):
        parser.error("normal runs require --lock, --project, --manifest, and --approval-evidence")
    if args.language is None:
        parser.error("normal runs require --language")
    run(args.lock, args.project, args.manifest, args.approval_evidence, args.reference_manifest, args.prepared_manifest, args.language)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
