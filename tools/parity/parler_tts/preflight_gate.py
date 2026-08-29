#!/usr/bin/env python3
"""Offline, fail-closed gate for the Parler-TTS reference closure.

This file deliberately uses only the Python standard library.  The worker runs
it before creating a scratch directory, syncing uv, or touching a model/source
endpoint.  Production rows remain pending until an owner records review and
authenticated approval evidence.
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
LOCK_SHA256 = "8d9946116a096f66daef0a1323a0d915045c812ae7d49b120e5c98b4bdb13df9"
PYPROJECT_SHA256 = "6514e3b3ed6e1878ce19bf5ffb1f45f19d096604d78ba2728a3094e935f569b3"
SOURCE_REPO = "https://github.com/huggingface/parler-tts.git"
SOURCE_REVISION = "d108732cd57788ec86bc857d99a6cabd66663d68"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
REVIEW_PLACEHOLDERS = {
    "", "unresolved", "pending", "pending_review", "owner_review_required",
    "review_required", "todo", "null", "none",
}
MANIFEST_KEYS = {
    "gate_version", "lock_sha256", "pyproject_sha256", "package_rows_sha256", "review_rows",
    "review_rows_sha256", "source_identity", "variants", "dac_identity", "reference_route",
    "model_reviews", "approval_scope_sha256", "operator_approval",
}
LOCK_KEYS = {"version", "revision", "requires-python", "resolution-markers", "supported-markers", "package"}
PACKAGE_KEYS = {"name", "version", "source", "resolution-markers", "dependencies", "sdist", "wheels", "metadata"}
ARTIFACT_KEYS = {"url", "hash", "size", "upload-time"}
REGISTRY_PACKAGE_SCHEMAS = {
    frozenset({"name", "version", "source", "sdist", "wheels"}),
    frozenset({"name", "version", "source", "dependencies", "sdist", "wheels"}),
    frozenset({"name", "version", "source", "dependencies", "wheels"}),
    frozenset({"name", "source", "version", "wheels"}),
}
REGISTRY_PACKAGE_SCHEMAS = {
    frozenset({"name", "version", "source", "sdist", "wheels"}),
    frozenset({"name", "version", "source", "dependencies", "sdist", "wheels"}),
    frozenset({"name", "version", "source", "dependencies", "wheels"}),
    frozenset({"name", "version", "source", "wheels"}),
}
DEPENDENCY_SCHEMAS = {frozenset({"name", "marker"})}
METADATA_REQUIREMENT_SCHEMAS = {frozenset({"name", "specifier"}), frozenset({"name", "specifier", "index"})}
REVIEW_ROW_KEYS = {"id", "status", "license", "native_review", "bundled_review", "evidence"}

VARIANTS = [
    {
        "slug": "english",
        "upstream_repo": "parler-tts/parler-tts-mini-v1",
        "upstream_revision": "0392b9451a601e528fd863bbb0598431fee810d9",
        "upstream_license": "Apache-2.0",
        "checkpoint_bytes": 3511490560,
        "checkpoint_sha256": "bc430eb6752b96ffb3f67036d1a6e207fbd031575a775716ffa64ef1eeb03692",
        "config_bytes": 6930,
        "config_sha256": "d8d2afa72bf3b098263a073c4d4df18627b76e1eb454c48f60bc5f787b2433b1",
        "generation_bytes": 265,
        "generation_sha256": "77831b39a5e0c4dba09b4dcbe37ce082e10f94c646920b20678c9c5289e52440",
        "public_repo": "vokra/parler-tts-mini-v1",
        "public_license": "Apache-2.0",
        "public_revision": "cb02a124c8d125231b396a293608f2488ae2e4d2",
        "public_file": "parler-tts-mini-v1.gguf",
        "public_bytes": 3511459168,
        "public_sha256": "7f69b811edae6cbe82fdfa8e72e6181945d4466748349aa74d994fb566785ddc",
    },
    {
        "slug": "multilingual",
        "upstream_repo": "parler-tts/parler-tts-mini-multilingual-v1.1",
        "upstream_revision": "11b27d57855dec1ce0914ba1f12363bf2ea75ba3",
        "upstream_license": "Apache-2.0",
        "checkpoint_bytes": 3751321772,
        "checkpoint_sha256": "79c64e3705e0ccce122988c7817f0d65efa3fd37625906d90765858bdab38412",
        "config_bytes": 7467,
        "config_sha256": "06d4cb727521542cab6b26d3ad1c8517d51fd1f551600ec67a59575364e221c6",
        "generation_bytes": 218,
        "generation_sha256": "3bb518e78ea5f32fbbcfc7f0aaed388e7aefede474d2bf4b8cf4502fd6b27a92",
        "public_repo": "vokra/parler-tts-mini-multilingual",
        "public_license": "Apache-2.0",
        "public_revision": "6f0f56788f06e6d514e0fab8530663b8af8b1fe2",
        "public_file": "parler-tts-mini.gguf",
        "public_bytes": 3751292736,
        "public_sha256": "d1edf792305a486192be73dfb279891febb6e81735abf06b2ae90b29da94134d",
    },
]

DAC_IDENTITY = {
    "repo": "parler-tts/dac_44khZ_8kbps",
    "revision": "5cf6b8ad50fbb17e52c341410a1d00083201b6a9",
    "license": "MIT",
    "files": [
        {"name": "config.json", "bytes": 227, "git_blob_sha1": "eee649ead33aad8fab39ccdf7b3f1fb708d02caa"},
        {"name": "model.safetensors", "bytes": 306642416, "sha256": "f65197de6142f9e0d186f78fb3aa12d47fde62f4c650e7ee5a254157618230f7"},
    ],
}
MODEL_REVIEW_IDENTITIES = [
    "parler-tts-source@d108732cd57788ec86bc857d99a6cabd66663d68",
    "parler-tts-mini-v1@0392b9451a601e528fd863bbb0598431fee810d9",
    "parler-tts-mini-multilingual-v1.1@11b27d57855dec1ce0914ba1f12363bf2ea75ba3",
    "dac_44khZ_8kbps@5cf6b8ad50fbb17e52c341410a1d00083201b6a9",
]


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_digest(value: Any) -> str:
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def validate_project_schema(project: dict[str, Any]) -> None:
    if set(project) != {"project", "tool"} or not isinstance(project.get("project"), dict) or set(project["project"]) != {"name", "version", "description", "requires-python", "dependencies"}:
        raise ValueError("pyproject.toml structural schema drifted")
    p = project["project"]
    if p["requires-python"] != "==3.12.*" or not isinstance(p["dependencies"], list) or any(not isinstance(x, str) or not x.strip() for x in p["dependencies"]):
        raise ValueError("pyproject.toml project contract drifted")
    uv = project["tool"].get("uv") if isinstance(project["tool"], dict) else None
    if set(project["tool"]) != {"uv"} or not isinstance(uv, dict) or set(uv) != {"package", "environments", "sources", "index"} or uv["package"] is not False or uv["environments"] != ["sys_platform == 'linux' and platform_machine == 'x86_64'"] or uv["sources"] != {"torch": {"index": "pytorch-cpu"}, "torchaudio": {"index": "pytorch-cpu"}} or uv["index"] != [{"name": "pytorch-cpu", "url": "https://download.pytorch.org/whl/cpu", "explicit": True}]:
        raise ValueError("pyproject.toml uv contract drifted")


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
        raise ValueError(f"{label} artifact schema is malformed")
    expected_host = "download-r2.pytorch.org" if registry == "https://download.pytorch.org/whl/cpu" else "files.pythonhosted.org"
    parsed = urlsplit(value["url"]) if isinstance(value["url"], str) else None
    if parsed is None or parsed.scheme != "https" or parsed.netloc != expected_host or not parsed.path:
        raise ValueError(f"{label} artifact URL is not the authenticated {expected_host} host")
    if not isinstance(value["hash"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value["hash"]):
        raise ValueError(f"{label} artifact hash is not a SHA-256")
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


def canonical_package_rows(lock: dict[str, Any], project: dict[str, Any]) -> list[dict[str, Any]]:
    packages = lock.get("package")
    if not isinstance(packages, list) or not packages or set(lock) != LOCK_KEYS or lock.get("version") != 1 or lock.get("revision") != 3 or lock.get("requires-python") != "==3.12.*" or not isinstance(lock.get("resolution-markers"), list) or any(not isinstance(item, str) or not item.strip() for item in lock["resolution-markers"]) or not isinstance(lock.get("supported-markers"), list) or any(not isinstance(item, str) or not item.strip() for item in lock["supported-markers"]):
        raise ValueError("uv.lock package table is missing")
    rows = []
    identities: set[tuple[str, str]] = set()
    virtual = []
    for package in packages:
        if not isinstance(package, dict):
            raise ValueError("uv.lock contains a malformed package row")
        if set(package) - PACKAGE_KEYS:
            raise ValueError("uv.lock package row contains an unknown field")
        dependencies = package.get("dependencies", [])
        if not isinstance(dependencies, list):
            raise ValueError("uv.lock contains malformed dependencies")
        if not isinstance(package.get("name"), str) or not package["name"].strip() or not isinstance(package.get("version"), str) or not package["version"].strip():
            raise ValueError("uv.lock has a package row without an exact name/version")
        identity = (package["name"], package["version"])
        if identity in identities:
            raise ValueError(f"uv.lock has duplicate package identity: {identity!r}")
        identities.add(identity)
        source = package.get("source")
        if not isinstance(source, dict) or len(source) != 1 or set(source) not in ({"registry"}, {"virtual"}):
            raise ValueError(f"uv.lock source schema is malformed: {identity!r}")
        registry = source.get("registry")
        if "registry" in source and registry not in ("https://pypi.org/simple", "https://download.pytorch.org/whl/cpu"):
            raise ValueError(f"uv.lock registry source is not an approved index: {identity!r}")
        if "virtual" in source:
            virtual.append(package)
            if source["virtual"] != ".":
                raise ValueError("uv.lock virtual source is not '.'")
        elif frozenset(package) not in REGISTRY_PACKAGE_SCHEMAS:
            raise ValueError(f"uv.lock registry package schema is not an exact committed variant: {identity!r}")
        markers = package.get("resolution-markers", [])
        if not isinstance(markers, list) or any(not isinstance(item, str) for item in markers):
            raise ValueError(f"uv.lock markers are malformed: {identity!r}")
        for dependency in dependencies:
            if not isinstance(dependency, dict) or frozenset(dependency) not in DEPENDENCY_SCHEMAS or not isinstance(dependency.get("name"), str):
                raise ValueError(f"uv.lock dependency row is malformed: {identity!r}")
            if not dependency["name"].strip() or not isinstance(dependency.get("marker"), str) or not dependency["marker"].strip():
                raise ValueError(f"uv.lock dependency fields are malformed: {identity!r}")
            for field in ("marker", "version"):
                if field in dependency and not isinstance(dependency[field], str):
                    raise ValueError(f"uv.lock dependency field is malformed: {identity!r}")
            if "source" in dependency:
                dependency_source = dependency["source"]
                if not isinstance(dependency_source, dict) or len(dependency_source) != 1 or set(dependency_source) not in ({"registry"}, {"virtual"}):
                    raise ValueError(f"uv.lock dependency source is malformed: {identity!r}")
                if "registry" in dependency_source and dependency_source["registry"] not in ("https://pypi.org/simple", "https://download.pytorch.org/whl/cpu"):
                    raise ValueError(f"uv.lock dependency registry source is not an approved index: {identity!r}")
                if "virtual" in dependency_source and dependency_source["virtual"] != ".":
                    raise ValueError(f"uv.lock dependency virtual source is not '.': {identity!r}")
        if "virtual" in source:
            if set(package) != {"name", "version", "source", "dependencies", "metadata"}:
                raise ValueError(f"{identity!r} virtual package schema is malformed")
            validate_metadata(package["metadata"], f"{identity!r} virtual")
            if "sdist" in package or "wheels" in package:
                raise ValueError(f"{identity!r} virtual package must not contain artifacts")
        else:
            if "metadata" in package:
                validate_metadata(package["metadata"], f"{identity!r}")
            if "sdist" in package:
                validate_artifact(package["sdist"], f"{identity!r} sdist", registry)
            if "wheels" in package:
                if not isinstance(package["wheels"], list) or not package["wheels"]:
                    raise ValueError(f"{identity!r} wheels table is malformed")
                for artifact in package["wheels"]:
                    validate_artifact(artifact, f"{identity!r} wheel", registry)
            if "sdist" not in package and not package.get("wheels"):
                raise ValueError(f"{identity!r} registry package has no authenticated artifacts")
        rows.append(package)
    if len(virtual) != 1 or (virtual[0]["name"], virtual[0]["version"]) != (project.get("name"), project.get("version")):
        raise ValueError("uv.lock virtual project is not bound to pyproject.toml")
    if any(not isinstance(row["name"], str) or not isinstance(row["version"], str) for row in rows):
        raise ValueError("uv.lock has a package row without an exact name/version")
    return sorted(rows, key=lambda row: (row["name"], row["version"]))


def blocked(reason: str) -> tuple[bool, str]:
    return False, reason


def reviewed_value(value: Any) -> bool:
    """Accept a real citation even when it contains a placeholder word."""
    normalized = re.sub(r"\s+", "_", value.strip()).casefold() if isinstance(value, str) else ""
    return normalized not in REVIEW_PLACEHOLDERS


def validate(project: Path, manifest_path: Path, evidence_path: Path | None = None) -> tuple[bool, str]:
    lock_path = project / "uv.lock"
    pyproject_path = project / "pyproject.toml"
    if not regular_file(lock_path) or not regular_file(pyproject_path) or not regular_file(manifest_path):
        return blocked("project lock/pyproject or gate manifest is missing")
    try:
        manifest = load_json(manifest_path)
        lock_bytes = lock_path.read_bytes()
        pyproject_bytes = pyproject_path.read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return blocked(f"gate inputs are unreadable: {exc}")
    if not isinstance(manifest, dict) or manifest.get("gate_version") != GATE_VERSION:
        return blocked("unsupported gate manifest version")
    if set(manifest) != MANIFEST_KEYS:
        return blocked("gate manifest top-level schema drifted")
    if digest_bytes(lock_bytes) != LOCK_SHA256 or manifest.get("lock_sha256") != LOCK_SHA256:
        return blocked("uv.lock bytes are not the reviewed exact lock")
    if digest_bytes(pyproject_bytes) != PYPROJECT_SHA256 or manifest.get("pyproject_sha256") != PYPROJECT_SHA256:
        return blocked("pyproject.toml bytes are not the reviewed exact project")
    try:
        project_data = tomllib.loads(pyproject_bytes.decode("utf-8")).get("project", {})
        full_project = tomllib.loads(pyproject_bytes.decode("utf-8"))
        validate_project_schema(full_project)
        rows = canonical_package_rows(tomllib.loads(lock_bytes.decode("utf-8")), project_data)
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError) as exc:
        return blocked(f"uv.lock canonicalization failed: {exc}")
    if canonical_digest(rows) != manifest.get("package_rows_sha256"):
        return blocked("canonical version/source/marker/dependency rows drifted")
    identities = [f'{row["name"]}@{row["version"]}' for row in rows]
    review_rows = manifest.get("review_rows")
    if not isinstance(review_rows, list):
        return blocked("version-keyed dependency review rows are missing")
    review_ids = [row.get("id") for row in review_rows if isinstance(row, dict)]
    if review_ids != sorted(identities) or len(review_ids) != len(identities) or len(set(review_ids)) != len(identities):
        return blocked("dependency review rows do not cover the exact lock identities")
    if canonical_digest(review_rows) != manifest.get("review_rows_sha256"):
        return blocked("dependency review row digest drifted")
    if manifest.get("source_identity") != {
        "repo": SOURCE_REPO, "revision": SOURCE_REVISION, "license": "Apache-2.0"
    }:
        return blocked("official source identity drifted")
    if manifest.get("variants") != VARIANTS or manifest.get("dac_identity") != DAC_IDENTITY:
        return blocked("fixed model or DAC identities drifted")
    route = manifest.get("reference_route")
    if route != {
        "entrypoint": "ParlerTTSForConditionalGeneration",
        "transformers": "4.46.1",
        "torch": "2.11.0+cpu",
        "torchaudio": "2.11.0+cpu",
        "torch_index": "https://download.pytorch.org/whl/cpu",
        "excluded": ["descript-audio-codec", "descript-audiotools", "librosa", "soxr", "soundfile", "protobuf"],
    }:
        return blocked("reference dependency route drifted")
    for row in review_rows:
        if not isinstance(row, dict) or set(row) != REVIEW_ROW_KEYS:
            return blocked(f"dependency review row schema is malformed: {row.get('id') if isinstance(row, dict) else None}")
        if (
            row.get("status") != "REVIEWED"
            or not reviewed_value(row.get("license"))
            or not reviewed_value(row.get("native_review"))
            or not reviewed_value(row.get("bundled_review"))
            or not reviewed_value(row.get("evidence"))
        ):
            return blocked(f"dependency review is unresolved: {row.get('id')}")
    model_reviews = manifest.get("model_reviews")
    if not isinstance(model_reviews, list):
        return blocked("model/source/DAC review rows are missing")
    model_review_ids = [item.get("id") for item in model_reviews if isinstance(item, dict)]
    if model_review_ids != MODEL_REVIEW_IDENTITIES or len(set(model_review_ids)) != len(MODEL_REVIEW_IDENTITIES):
        return blocked("model/source/DAC review rows are not the exact identity set")
    for item in model_reviews:
        if (
            not isinstance(item, dict) or set(item) != REVIEW_ROW_KEYS or item.get("status") != "REVIEWED"
            or not reviewed_value(item.get("license"))
            or not reviewed_value(item.get("native_review"))
            or not reviewed_value(item.get("bundled_review"))
            or not reviewed_value(item.get("evidence"))
        ):
            return blocked("model/DAC license or native/bundled review is unresolved")
    scope = {
        "lock_sha256": LOCK_SHA256, "pyproject_sha256": PYPROJECT_SHA256,
        "package_rows_sha256": manifest["package_rows_sha256"], "package_rows": rows, "review_rows": review_rows,
        "source_identity": manifest["source_identity"], "variants": VARIANTS,
        "dac_identity": DAC_IDENTITY, "reference_route": route,
        "model_reviews": model_reviews,
    }
    scope_sha256 = canonical_digest(scope)
    if manifest.get("approval_scope_sha256") != scope_sha256:
        return blocked("operator approval scope is not bound to exact inputs")
    approval = manifest.get("operator_approval")
    if (
        not isinstance(approval, dict) or approval.get("schema") != "v1" or approval.get("decision") != "APPROVED"
        or not isinstance(approval.get("signer"), str) or not approval["signer"]
        or not isinstance(approval.get("digest"), str) or not HEX64.fullmatch(approval["digest"])
        or approval["digest"] != scope_sha256
    ):
        return blocked("exact operator approval is pending or invalid")
    evidence_path = evidence_path or manifest_path.with_name("license_gate_evidence.json")
    if not regular_file(evidence_path):
        return blocked("authenticated operator approval evidence is missing")
    try:
        evidence = load_json(evidence_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return blocked(f"operator approval evidence is unreadable: {exc}")
    if isinstance(evidence, dict) and set(evidence) != {"schema", "decision", "scope_sha256", "manifest_sha256", "lock_sha256", "pyproject_sha256", "signer", "digest"}:
        return blocked("operator approval evidence schema drifted")
    if (
        not isinstance(evidence, dict) or evidence.get("schema") != "v1" or evidence.get("decision") != "APPROVED"
        or evidence.get("scope_sha256") != scope_sha256
        or evidence.get("manifest_sha256") != digest_bytes(manifest_path.read_bytes())
        or evidence.get("lock_sha256") != LOCK_SHA256 or evidence.get("pyproject_sha256") != PYPROJECT_SHA256
        or evidence.get("signer") != approval["signer"] or evidence.get("digest") != approval["digest"]
    ):
        return blocked("authenticated operator approval evidence is not bound to this scope")
    return True, "PASS"


def self_test() -> int:
    project = Path(__file__).resolve().parent
    manifest_path = project / "license_gate_manifest.json"
    ok, reason = validate(project, manifest_path)
    if ok or ("unresolved" not in reason and "artifact" not in reason):
        print(f"parler gate: expected pending production gate, got {reason}", file=sys.stderr)
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
                print(f"parler artifact tamper accepted: {label}", file=sys.stderr); return 1
        print("parler gate: self-test PASS (production artifact schema blocker)")
        return 0
    with tempfile.TemporaryDirectory(prefix="parler-gate-") as directory:
        root = Path(directory)
        test_project = root / "project"
        test_project.mkdir()
        shutil.copy2(project / "uv.lock", test_project / "uv.lock")
        shutil.copy2(project / "pyproject.toml", test_project / "pyproject.toml")
        test_rows = canonical_package_rows(
            tomllib.loads((test_project / "uv.lock").read_text(encoding="utf-8")),
            tomllib.loads((test_project / "pyproject.toml").read_text(encoding="utf-8"))["project"],
        )
        approved = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in approved["review_rows"]:
            row.update({"status": "REVIEWED", "license": "SELF_TEST", "native_review": "SELF_TEST", "bundled_review": "SELF_TEST", "evidence": "self-test-evidence"})
        for row in approved["model_reviews"]:
            row.update({"status": "REVIEWED", "license": "SELF_TEST", "native_review": "SELF_TEST", "bundled_review": "SELF_TEST", "evidence": "self-test-evidence"})
        approved["review_rows_sha256"] = canonical_digest(approved["review_rows"])
        scope = {"lock_sha256": LOCK_SHA256, "pyproject_sha256": PYPROJECT_SHA256,
                 "package_rows_sha256": approved["package_rows_sha256"], "package_rows": test_rows, "review_rows": approved["review_rows"],
                 "source_identity": approved["source_identity"], "variants": VARIANTS,
                 "dac_identity": DAC_IDENTITY, "reference_route": approved["reference_route"],
                 "model_reviews": approved["model_reviews"]}
        approved["approval_scope_sha256"] = canonical_digest(scope)
        approved["operator_approval"] = {"schema": "v1", "decision": "APPROVED", "signer": "self-test", "digest": approved["approval_scope_sha256"]}
        approved_path = root / "manifest.json"
        approved_path.write_text(json.dumps(approved), encoding="utf-8")
        evidence_path = root / "license_gate_evidence.json"
        evidence_path.write_text(json.dumps({"schema": "v1", "decision": "APPROVED", "scope_sha256": approved["approval_scope_sha256"], "manifest_sha256": digest_bytes(approved_path.read_bytes()), "lock_sha256": LOCK_SHA256, "pyproject_sha256": PYPROJECT_SHA256, "signer": "self-test", "digest": approved["approval_scope_sha256"]}), encoding="utf-8")
        good, why = validate(test_project, approved_path, evidence_path)
        if not good:
            print(f"parler gate: approved baseline failed: {why}", file=sys.stderr)
            return 1
        duplicate_manifest = root / "duplicate-manifest.json"
        duplicate_manifest.write_text('{"gate_version":1,"gate_version":1}', encoding="utf-8")
        duplicate_ok, duplicate_why = validate(test_project, duplicate_manifest, evidence_path)
        if duplicate_ok or "duplicate JSON key" not in duplicate_why:
            print("parler gate: duplicate manifest key was not rejected cleanly", file=sys.stderr)
            return 1
        duplicate_evidence = root / "duplicate-evidence.json"
        duplicate_evidence.write_text('{"schema":"v1","schema":"v1"}', encoding="utf-8")
        duplicate_ok, duplicate_why = validate(test_project, approved_path, duplicate_evidence)
        if duplicate_ok or "duplicate JSON key" not in duplicate_why:
            print("parler gate: duplicate evidence key was not rejected cleanly", file=sys.stderr)
            return 1
        mutations = {
            "lock": lambda value: value.update(lock_sha256="0" * 64),
            "package": lambda value: value["review_rows"][0].update(license="tampered"),
            "unresolved": lambda value: value["review_rows"][0].update(status="UNRESOLVED", license="approved", native_review="approved", bundled_review="approved"),
            "model": lambda value: value["variants"][0].update(upstream_revision="0" * 40),
            "source": lambda value: value["source_identity"].update(revision="0" * 40),
            "dac": lambda value: value["dac_identity"].update(revision="0" * 40),
            "route": lambda value: value["reference_route"].update(transformers="0.0.0"),
            "scope": lambda value: value.update(approval_scope_sha256="0" * 64),
            "signer": lambda value: value["operator_approval"].update(signer="other"),
        }
        for label, mutate in mutations.items():
            candidate = json.loads(approved_path.read_text(encoding="utf-8"))
            mutate(candidate)
            candidate_path = root / f"{label}.json"
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            if validate(test_project, candidate_path, evidence_path)[0]:
                print(f"parler gate: {label} tamper was accepted", file=sys.stderr)
                return 1
        for field in ("license", "native_review", "bundled_review", "evidence"):
            candidate = json.loads(approved_path.read_text(encoding="utf-8"))
            candidate["review_rows"][0][field] = "  PeNdInG_ReViEw  "
            candidate_path = root / f"placeholder-{field}.json"
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            if validate(test_project, candidate_path, evidence_path)[0]:
                print(f"parler gate: {field} placeholder was accepted", file=sys.stderr)
                return 1
        for label in ("missing", "extra", "duplicate"):
            candidate = json.loads(approved_path.read_text(encoding="utf-8"))
            if label == "missing":
                candidate["model_reviews"].pop()
            elif label == "extra":
                candidate["model_reviews"].append(dict(candidate["model_reviews"][0]))
            else:
                candidate["model_reviews"][1]["id"] = candidate["model_reviews"][0]["id"]
            candidate_path = root / f"model-{label}.json"
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            if validate(test_project, candidate_path, evidence_path)[0]:
                print(f"parler gate: model {label} identity tamper was accepted", file=sys.stderr)
                return 1
        candidate = json.loads(approved_path.read_text(encoding="utf-8"))
        candidate["model_reviews"][0]["license"] = "\tOWNER_REVIEW_REQUIRED\n"
        candidate_path = root / "model-placeholder.json"
        candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
        if validate(test_project, candidate_path, evidence_path)[0]:
            print("parler gate: model placeholder was accepted", file=sys.stderr)
            return 1
        evidence_tampered = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence_tampered["scope_sha256"] = "0" * 64
        evidence_path.write_text(json.dumps(evidence_tampered), encoding="utf-8")
        if validate(test_project, approved_path, evidence_path)[0]:
            print("parler gate: evidence tamper was accepted", file=sys.stderr)
            return 1
        duplicate_json = root / "duplicate.json"
        duplicate_json.write_text('{"schema": 1, "schema": 2}', encoding="utf-8")
        try:
            load_json(duplicate_json)
        except ValueError:
            pass
        else:
            print("parler gate: duplicate JSON keys were accepted", file=sys.stderr)
            return 1
    print("parler gate: self-test PASS (pending production + approved baseline/tamper cases)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--evidence", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    manifest = args.manifest or args.project / "license_gate_manifest.json"
    ok, reason = validate(args.project, manifest, args.evidence)
    if not ok:
        print(f"parler preflight gate: BLOCKED: {reason}", file=sys.stderr)
        return 2
    print("parler preflight gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
