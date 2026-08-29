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

GATE_VERSION = 1
LOCK_SHA256 = "3a7809a06bcaa9e18d89c8fab77860054098726a8cfcd51a658cf461c5c89d42"
PYPROJECT_SHA256 = "adf757e1349d365dcda13c4944dbdd435470e9db4c201049e8f49bfba60bfecb"
REFERENCE_AUDIO_SHA256 = "241c0d93cc7ed8792c85c525d1e02b8c33850b791902a5e75b79c2d500e71a1a"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
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


def _artifact(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"url", "hash", "size", "upload-time"}:
        raise ValueError("uv.lock artifact must have exactly url/hash/size/upload-time")
    if (not isinstance(value["url"], str) or not value["url"].startswith("https://")
            or urlparse(value["url"]).hostname not in {"files.pythonhosted.org", "download-r2.pytorch.org", "download.pytorch.org"}
            or not isinstance(value["hash"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value["hash"])
            or not isinstance(value["size"], int) or isinstance(value["size"], bool) or value["size"] <= 0
            or not isinstance(value["upload-time"], str) or not value["upload-time"].strip()):
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
                _artifact(package["sdist"])
            if "wheels" in package:
                if not isinstance(package["wheels"], list):
                    raise ValueError("uv.lock wheels are malformed")
                for wheel in package["wheels"]:
                    _artifact(wheel)
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
    if set(manifest) != {"gate_version", "lock_sha256", "pyproject_sha256", "package_rows_sha256", "review_rows", "review_rows_sha256", "variants", "model_identities", "reference_audio", "approval_scope_sha256", "operator_approval"}:
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
    global LOCK_SHA256
    project = Path(__file__).resolve().parent
    manifest = project / "license_gate_manifest.json"
    if reviewed_value("  PENDING_REVIEW  ") or not reviewed_value("reviewed citation: TODO was resolved"):
        print("qwen3-asr preflight gate: placeholder normalization self-test failed", file=sys.stderr)
        return 1
    ok, reason = validate(project, manifest)
    if ok or not ("operator approval" in reason or "unresolved" in reason or "artifact" in reason):
        print("qwen3-asr preflight gate: self-test expected pending approval", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory(prefix="qwen3-asr-gate-") as directory:
        root = Path(directory)
        test_project = root / "project"
        test_project.mkdir()
        shutil.copy2(project / "uv.lock", test_project / "uv.lock")
        shutil.copy2(project / "pyproject.toml", test_project / "pyproject.toml")
        # The checked-in lock intentionally remains fail-closed while older
        # resolver records lack wheel sizes.  Complete only the disposable
        # baseline with positive fixture sizes; production still requires the
        # reviewed lock bytes and rejects those records.
        complete_lock = re.sub(
            r'(hash = "sha256:[0-9a-f]{64}")(, upload-time =)',
            r'\1, size = 1\2',
            (test_project / "uv.lock").read_text(encoding="utf-8"),
        )
        (test_project / "uv.lock").write_text(complete_lock, encoding="utf-8")
        test_lock_sha = digest_bytes((test_project / "uv.lock").read_bytes())
        test_lock = tomllib.loads(complete_lock)
        test_rows = canonical_package_rows(test_lock)
        old_lock_sha = LOCK_SHA256
        LOCK_SHA256 = test_lock_sha
        shutil.copy2(manifest, root / "manifest.json")
        approved = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        approved["lock_sha256"] = test_lock_sha
        approved["package_rows_sha256"] = canonical_digest(test_rows)
        for row in approved["review_rows"]:
            row.update({"status": "REVIEWED", "license": "SELF_TEST", "native_review": "SELF_TEST", "bundled_review": "SELF_TEST", "evidence": "self-test-evidence"})
        for identity in approved["model_identities"]:
            identity.update({"license_status": "REVIEWED", "license_digest": "0" * 64, "native_review": "SELF_TEST", "bundled_review": "SELF_TEST", "evidence": "self-test-evidence"})
        approved["review_rows_sha256"] = canonical_digest(approved["review_rows"])
        scope = {
            "lock_sha256": LOCK_SHA256,
            "pyproject_sha256": PYPROJECT_SHA256,
            "package_rows_sha256": approved["package_rows_sha256"],
            "variants": VARIANTS,
            "model_identities": approved["model_identities"],
            "reference_audio": approved["reference_audio"],
            "review_rows": approved["review_rows"],
        }
        approved["approval_scope_sha256"] = canonical_digest(scope)
        approved["operator_approval"] = {
            "schema": "v1", "decision": "APPROVED", "signer": "self-test-signer",
            "digest": approved["approval_scope_sha256"],
        }
        approved_manifest = root / "approved-manifest.json"
        approved_manifest.write_text(json.dumps(approved), encoding="utf-8")
        evidence = root / "license_gate_evidence.json"
        evidence.write_text(json.dumps({
            "schema": "v1", "decision": "APPROVED",
            "scope_sha256": approved["approval_scope_sha256"],
            "manifest_sha256": digest_bytes(approved_manifest.read_bytes()),
            "lock_sha256": LOCK_SHA256, "pyproject_sha256": PYPROJECT_SHA256,
            "signer": "self-test-signer", "digest": approved["approval_scope_sha256"],
        }), encoding="utf-8")
        ok, reason = validate(test_project, approved_manifest, evidence)
        if not ok:
            print(f"qwen3-asr preflight gate: approved baseline self-test failed: {reason}", file=sys.stderr)
            return 1
        duplicate_manifest = root / "duplicate-manifest.json"
        duplicate_manifest.write_text('{"gate_version":1,"gate_version":1}', encoding="utf-8")
        ok, _ = validate(test_project, duplicate_manifest, evidence)
        if ok:
            print("qwen3-asr preflight gate: duplicate manifest key accepted", file=sys.stderr)
            return 1
        duplicate_evidence = root / "duplicate-evidence.json"
        duplicate_evidence.write_text('{"schema":"v1","schema":"v1"}', encoding="utf-8")
        ok, _ = validate(test_project, approved_manifest, duplicate_evidence)
        if ok:
            print("qwen3-asr preflight gate: duplicate evidence key accepted", file=sys.stderr)
            return 1

        def assert_blocked(label: str, mutate: Any) -> bool:
            candidate = json.loads(approved_manifest.read_text(encoding="utf-8"))
            mutate(candidate)
            path = root / f"{label}.json"
            path.write_text(json.dumps(candidate), encoding="utf-8")
            blocked_ok, _ = validate(test_project, path, evidence)
            return not blocked_ok

        lock = test_project / "uv.lock"
        lock.write_bytes(lock.read_bytes() + b"\n")
        ok, reason = validate(test_project, approved_manifest, evidence)
        if ok or "lock" not in reason:
            print("qwen3-asr preflight gate: lock tamper self-test failed", file=sys.stderr)
            return 1
        shutil.copy2(project / "uv.lock", lock)
        LOCK_SHA256 = old_lock_sha
        if not assert_blocked("variant-tamper", lambda value: value["variants"][0].update(revision="0" * 40)):
            print("qwen3-asr preflight gate: variant tamper self-test failed", file=sys.stderr)
            return 1
        if not assert_blocked("model-tamper", lambda value: value["model_identities"][0].update(repo="Qwen/tampered")):
            print("qwen3-asr preflight gate: model identity tamper self-test failed", file=sys.stderr)
            return 1
        if not assert_blocked("model-license-tamper", lambda value: value["model_identities"][0].update(license_digest="1" * 64)):
            print("qwen3-asr preflight gate: model license tamper self-test failed", file=sys.stderr)
            return 1
        for field in ("license_status", "native_review", "bundled_review", "evidence"):
            if not assert_blocked(f"model-{field}-placeholder", lambda value, field=field: value["model_identities"][0].update(**{field: "  pEnDiNg_ReViEw  "})):
                print(f"qwen3-asr preflight gate: model {field} placeholder self-test failed", file=sys.stderr)
                return 1
        if not assert_blocked("audio-tamper", lambda value: value["reference_audio"].update(sha256="2" * 64)):
            print("qwen3-asr preflight gate: audio tamper self-test failed", file=sys.stderr)
            return 1
        def unresolved_row(value: dict[str, Any]) -> None:
            value["review_rows"][0]["native_review"] = "UNRESOLVED"
            value["review_rows_sha256"] = canonical_digest(value["review_rows"])
        if not assert_blocked("review-tamper", unresolved_row):
            print("qwen3-asr preflight gate: unresolved-row self-test failed", file=sys.stderr)
            return 1
        if not assert_blocked("scope-tamper", lambda value: value.update(approval_scope_sha256="3" * 64)):
            print("qwen3-asr preflight gate: scope tamper self-test failed", file=sys.stderr)
            return 1
        if not assert_blocked("signer-tamper", lambda value: value["operator_approval"].update(signer="other-signer")):
            print("qwen3-asr preflight gate: signer tamper self-test failed", file=sys.stderr)
            return 1
        evidence_tampered = json.loads(evidence.read_text(encoding="utf-8"))
        evidence_tampered["scope_sha256"] = "4" * 64
        evidence.write_text(json.dumps(evidence_tampered), encoding="utf-8")
        ok, _ = validate(test_project, approved_manifest, evidence)
        if ok:
            print("qwen3-asr preflight gate: evidence tamper self-test failed", file=sys.stderr)
            return 1
    print("qwen3-asr preflight gate: self-test PASS")
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
