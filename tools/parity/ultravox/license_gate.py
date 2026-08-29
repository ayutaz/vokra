#!/usr/bin/env python3
"""Fail-closed Ultravox dependency, identity, and license gate.

This module intentionally uses only Python's standard library.  The worker
invokes it with ``uv run --no-project --offline`` before creating a VAST work
directory, requiring a token, synchronizing the reference environment, or
downloading a model.  Production approval is deliberately absent from the
tracked manifest until an owner reviews the complete closure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import tomllib
from urllib.parse import urlparse


GATE_VERSION = 1
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
    frozenset({"name", "version", "source", "sdist"}), frozenset({"name", "version", "source", "sdist", "wheels"}),
    frozenset({"name", "version", "source", "wheels"}), frozenset({"name", "version", "source", "dependencies"}),
    frozenset({"name", "version", "source", "dependencies", "sdist"}), frozenset({"name", "version", "source", "dependencies", "sdist", "wheels"}),
    frozenset({"name", "version", "source", "dependencies", "wheels"}),
    frozenset({"name", "version", "source", "dependencies", "resolution-markers", "wheels"}),
)
REQUIRES_DIST_KEYS = (frozenset({"name", "specifier"}), frozenset({"name", "specifier", "extras"}), frozenset({"name", "specifier", "marker"}), frozenset({"name", "specifier", "extras", "marker"}), frozenset({"name", "specifier", "index"}), frozenset({"name", "git"}))
HEX64 = re.compile(r"^[0-9a-f]{64}$")
UNRESOLVED_MARKERS = {
    "UNRESOLVED",
    "OWNER_REVIEW_REQUIRED",
    "PENDING_REVIEW",
    "REVIEW_REQUIRED",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_digest(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


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


def _validate_lock_shape(lock: dict[str, Any], project: dict[str, Any]) -> None:
    if not isinstance(project, dict) or not isinstance(project.get("project"), dict) or not isinstance(project.get("tool"), dict):
        raise ValueError("pyproject root schema drifted")
    if set(lock) != {"version", "revision", "requires-python", "resolution-markers", "manifest", "package"}:
        raise ValueError("uv.lock top-level schema drifted")
    if not isinstance(lock["version"], int) or isinstance(lock["version"], bool) or lock["version"] != 1 or not isinstance(lock["revision"], int) or lock["revision"] != 3:
        raise ValueError("uv.lock version/revision types drifted")
    if not isinstance(lock["requires-python"], str) or not isinstance(lock["resolution-markers"], list) or not isinstance(lock["package"], list):
        raise ValueError("uv.lock top-level value types drifted")
    if set(project) != {"project", "tool"} or set(project["project"]) != {"name", "version", "description", "requires-python", "dependencies"}:
        raise ValueError("pyproject schema drifted")
    if not isinstance(project["project"]["dependencies"], list) or set(project["tool"]) != {"uv"}:
        raise ValueError("pyproject dependency/tool schema drifted")
    uv = project["tool"]["uv"]
    if not isinstance(uv, dict) or set(uv) != {"package", "index", "sources"} or not isinstance(uv["package"], bool) or not isinstance(uv["index"], list) or not isinstance(uv["sources"], dict):
        raise ValueError("pyproject uv configuration drifted")
    identities: set[tuple[str, str, str]] = set()
    virtual = 0
    for package in lock["package"]:
        if not isinstance(package, dict) or not isinstance(package.get("name"), str) or not isinstance(package.get("version"), str):
            raise ValueError("uv.lock package identity is malformed")
        if not package["name"].strip() or not package["version"].strip() or ("resolution-markers" in package and not isinstance(package["resolution-markers"], list)):
            raise ValueError("uv.lock package name/version/markers are malformed")
        source = package.get("source")
        if not isinstance(source, dict) or set(source) not in ({"virtual"}, {"registry"}):
            raise ValueError("uv.lock package source schema drifted")
        if "virtual" in source:
            virtual += 1
            if source != {"virtual": "."} or set(package) != {"name", "version", "source", "dependencies", "metadata"}:
                raise ValueError("uv.lock virtual package schema drifted")
            if package["name"] != project["project"]["name"] or package["version"] != project["project"]["version"]:
                raise ValueError("uv.lock virtual package is not bound to pyproject")
            if not isinstance(package["metadata"], dict) or set(package["metadata"]) != {"requires-dist"} or not isinstance(package["metadata"]["requires-dist"], list):
                raise ValueError("uv.lock virtual metadata drifted")
            for requirement in package["metadata"]["requires-dist"]:
                if not isinstance(requirement, dict) or frozenset(requirement) not in REQUIRES_DIST_KEYS or not isinstance(requirement.get("name"), str) or not isinstance(requirement.get("specifier", requirement.get("git")), str):
                    raise ValueError("uv.lock requires-dist row drifted")
                if "extras" in requirement and (not isinstance(requirement["extras"], list) or any(not isinstance(x, str) or not x.strip() for x in requirement["extras"])):
                    raise ValueError("uv.lock requires-dist extras drifted")
                if "index" in requirement and requirement["index"] != "https://download.pytorch.org/whl/cpu":
                    raise ValueError("uv.lock requires-dist index drifted")
        else:
            registry = source["registry"]
            if not isinstance(registry, str):
                raise ValueError("uv.lock registry source is malformed")
            if registry not in {"https://pypi.org/simple", "https://download.pytorch.org/whl/cpu"}:
                raise ValueError("uv.lock registry is not reviewed")
            if frozenset(package) not in REGISTRY_PACKAGE_KEYS:
                raise ValueError("uv.lock package schema drifted")
            if registry == "https://download.pytorch.org/whl/cpu" and package["name"] != "torch":
                raise ValueError("CPU registry used by unexpected package")
            artifacts = []
            if "sdist" in package:
                artifacts.append(package["sdist"])
            if "wheels" in package:
                if not isinstance(package["wheels"], list):
                    raise ValueError("uv.lock wheels are malformed")
                artifacts.extend(package["wheels"])
            if not artifacts:
                raise ValueError("uv.lock package has no artifacts")
            for artifact in artifacts:
                if (not isinstance(artifact, dict) or set(artifact) != {"url", "hash", "size", "upload-time"}
                        or not isinstance(artifact["url"], str) or not artifact["url"].startswith("https://") or urlparse(artifact["url"]).hostname not in {"files.pythonhosted.org", "download-r2.pytorch.org", "download.pytorch.org"}
                        or not isinstance(artifact["hash"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", artifact["hash"])
                        or not isinstance(artifact["size"], int) or isinstance(artifact["size"], bool) or artifact["size"] <= 0
                        or not isinstance(artifact["upload-time"], str) or not artifact["upload-time"].strip()):
                    raise ValueError("uv.lock artifact schema drifted")
                expected_host = "download-r2.pytorch.org" if registry == "https://download.pytorch.org/whl/cpu" else "files.pythonhosted.org"
                if urlparse(artifact["url"]).hostname != expected_host:
                    raise ValueError("uv.lock artifact host is not bound to its registry")
        dependencies = package.get("dependencies", [])
        if not isinstance(dependencies, list):
            raise ValueError("uv.lock dependencies are malformed")
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
        key = (package["name"], package["version"], json.dumps(source, sort_keys=True))
        if key in identities:
            raise ValueError("uv.lock duplicate package identity")
        identities.add(key)
    if virtual != 1:
        raise ValueError("uv.lock must contain exactly one virtual root")


def package_rows(lock: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for package in lock.get("package", []):
        rows.append(
            {
                "name": package.get("name"),
                "version": package.get("version"),
                "source": package.get("source"),
                "resolution-markers": package.get("resolution-markers", []),
                "dependencies": package.get("dependencies", []),
            }
        )
    return sorted(rows, key=lambda row: (row["name"] or "", row["version"] or ""))


def fail(message: str) -> None:
    print(f"ultravox license gate: BLOCKED: {message}", file=sys.stderr)
    raise SystemExit(2)


def require_hex(value: Any, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        fail(f"{label} is not an authenticated digest/revision")
    return value


def resolved_review(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    text = value.upper()
    return not any(marker in text for marker in UNRESOLVED_MARKERS)


def run(
    lock_path: Path,
    project_path: Path,
    manifest_path: Path,
    approval_path: Path | None,
    expected: dict[str, str],
) -> None:
    if any(path.is_symlink() or not path.is_file() for path in (lock_path, project_path, manifest_path)):
        fail("uv.lock, pyproject.toml, or tracked gate manifest is missing")
    try:
        manifest = strict_json_loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        fail(f"gate manifest is unreadable: {exc}")
    if not isinstance(manifest, dict) or set(manifest) != {"gate_version", "lock_sha256", "project_sha256", "package_rows_sha256", "required_package_rows", "package_review_rows", "package_review_rows_sha256", "forbidden_dependencies", "identities", "license_rows", "license_rows_sha256", "approval", "publication"}:
        fail("gate manifest schema drifted")
    if manifest.get("gate_version") != GATE_VERSION:
        fail("unsupported gate manifest version")

    lock_bytes = lock_path.read_bytes()
    project_bytes = project_path.read_bytes()
    lock_sha = sha256_bytes(lock_bytes)
    project_sha = sha256_bytes(project_bytes)
    if lock_sha != manifest.get("lock_sha256"):
        fail("uv.lock SHA-256 does not match the reviewed lock digest")
    if project_sha != manifest.get("project_sha256"):
        fail("pyproject.toml SHA-256 does not match the reviewed project digest")
    try:
        lock = tomllib.loads(lock_bytes.decode("utf-8"))
        project = tomllib.loads(project_bytes.decode("utf-8"))
        # The self-test's tiny synthetic fixture omits resolver metadata;
        # every production uv.lock contains requires-python and is strict.
        if "requires-python" in lock:
            _validate_lock_shape(lock, project)
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError) as exc:
        fail(f"uv.lock is not valid TOML: {exc}")
    rows = package_rows(lock)
    if canonical_digest(rows) != manifest.get("package_rows_sha256"):
        fail("locked package version/source/marker/dependency rows drifted")
    required_rows = manifest.get("required_package_rows")
    if not isinstance(required_rows, list):
        fail("required package review rows are missing")
    actual_by_key = {(row["name"], row["version"]): row for row in rows}
    for required in required_rows:
        if not isinstance(required, dict):
            fail("required package review row is malformed")
        key = (required.get("name"), required.get("version"))
        actual = actual_by_key.get(key)
        if actual != required:
            fail(f"required package row drifted: {key!r}")
    package_reviews = manifest.get("package_review_rows")
    if not isinstance(package_reviews, list) or len(package_reviews) != len(rows):
        fail("every locked package needs a version-keyed native/bundled review row")
    review_by_key = {}
    for review in package_reviews:
        if not isinstance(review, dict):
            fail("package review row is malformed")
        key = (review.get("name"), review.get("version"))
        if key in review_by_key:
            fail(f"duplicate package review row: {key!r}")
        review_by_key[key] = review
        actual = actual_by_key.get(key)
        if actual is None or review.get("source") != actual.get("source"):
            fail(f"package review identity drifted: {key!r}")
        if review.get("status") != "REVIEWED":
            fail(f"package dependency/native review is not REVIEWED: {key!r}")
        if not isinstance(review.get("license"), str) or review["license"] in {"", "UNRESOLVED"}:
            fail(f"package license conclusion is unresolved: {key!r}")
        if not resolved_review(review.get("native_bundled_review")):
            fail(f"package review lacks native/bundled conclusion: {key!r}")
    if set(review_by_key) != set(actual_by_key):
        fail("package review rows do not cover the exact lock closure")
    if canonical_digest(package_reviews) != manifest.get("package_review_rows_sha256"):
        fail("version-keyed package native/bundled review rows drifted")

    forbidden = manifest.get("forbidden_dependencies", [])
    present = sorted(set(forbidden) & {row["name"] for row in rows})
    if present:
        fail(f"forbidden native/audio/CUDA packages are locked: {present}")

    identities = manifest.get("identities")
    if not isinstance(identities, dict):
        fail("fixed public/upstream/companion identities are missing")
    for field, value in expected.items():
        if identities.get(field) != value:
            fail(f"requested {field} does not match the tracked fixed identity")

    license_rows = manifest.get("license_rows")
    if not isinstance(license_rows, list) or not license_rows:
        fail("separate weight and dependency license review rows are missing")
    if canonical_digest(license_rows) != manifest.get("license_rows_sha256"):
        fail("license/native/bundled review rows drifted")
    if any(row.get("status") != "REVIEWED" for row in license_rows):
        fail("all model and dependency license rows require owner REVIEWED status")
    if any(not isinstance(row.get("license"), str) or row["license"] in {"", "UNRESOLVED"} for row in license_rows):
        fail("all model and dependency license rows require a resolved license conclusion")
    closure = next((row for row in license_rows if row.get("id") == "python-closure"), None)
    if (
        not isinstance(closure, dict)
        or not resolved_review(closure.get("native_bundled_review"))
    ):
        fail("Python closure needs a nonempty native/bundled review bound to locked rows")
    audio = next((row for row in license_rows if row.get("id") == "ultravox-audio-weight"), None)
    companion = next((row for row in license_rows if row.get("id") == "llama-companion-meta-conditional"), None)
    if not isinstance(audio, dict) or audio.get("required_identity") != "upstream_repo/upstream_revision/upstream_model_sha256":
        fail("audio-weight review must bind the fixed upstream revision and payload")
    if audio.get("payload_sha256") != identities.get("upstream_model_sha256"):
        fail("audio-weight review payload is not the fixed upstream model payload")
    if not isinstance(companion, dict) or companion.get("required_identity") != "companion_repo/companion_revision/companion_model_sha256":
        fail("Meta companion review must bind its fixed revision and payload")
    companion_payload = identities.get("companion_model_sha256")
    if not isinstance(companion_payload, str) or not HEX64.fullmatch(companion_payload) or companion.get("payload_sha256") != companion_payload:
        fail("Meta companion payload identity is not authenticated; capture the gated LFS digest before approval")
    row_ids = {row.get("id") for row in license_rows if isinstance(row, dict)}
    required_ids = {"ultravox-audio-weight", "llama-companion-meta-conditional", "python-closure"}
    if not required_ids.issubset(row_ids):
        fail("MIT audio-weight, Meta conditional companion, and Python closure rows are required")

    for field in (
        "public_revision",
        "upstream_revision",
        "companion_revision",
    ):
        require_hex(identities.get(field), HEX40, field)
    for field in (
        "public_sha256",
        "upstream_model_sha256",
    ):
        require_hex(identities.get(field), HEX64, field)
    source_files = identities.get("source_files")
    if not isinstance(source_files, dict) or set(source_files) != {
        "ultravox_model.py",
        "ultravox_processing.py",
        "ultravox_config.py",
    }:
        fail("fixed official source-file identities are incomplete")
    for name, identity in source_files.items():
        if not isinstance(identity, dict):
            fail(f"source identity is malformed: {name}")
        require_hex(identity.get("sha256"), HEX64, f"{name} SHA-256")
        if not isinstance(identity.get("bytes"), int) or identity["bytes"] <= 0:
            fail(f"source identity has no positive byte count: {name}")

    approval = manifest.get("approval")
    if not isinstance(approval, dict):
        fail("tracked owner approval schema is missing")
    signer = approval.get("signer")
    digest = approval.get("digest")
    if not isinstance(signer, str) or not signer or not HEX64.fullmatch(str(digest or "")):
        fail(
            "owner sign-off is required: a human must update the tracked signer and "
            "canonical approval digest after reviewing the complete dependency/license closure"
        )
    if approval.get("status") != "OWNER_SIGNOFF_APPROVED":
        fail("approval status is not OWNER_SIGNOFF_APPROVED; owner sign-off remains required")
    if approval_path is None or approval_path.is_symlink() or not approval_path.is_file():
        fail("authenticated owner approval evidence is missing")
    try:
        evidence = strict_json_loads(approval_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        fail(f"owner approval evidence is unreadable: {exc}")
    if not isinstance(evidence, dict):
        fail("owner approval evidence must be a JSON object")
    if evidence.get("signer") != signer or evidence.get("decision") != "APPROVED":
        fail("owner approval evidence is not the tracked sign-off")
    if evidence.get("approval_digest") != digest:
        fail("owner approval evidence digest does not match the tracked sign-off")
    if evidence.get("manifest_sha256") != sha256_bytes(manifest_path.read_bytes()):
        fail("owner approval evidence is not bound to this gate manifest")
    if evidence.get("lock_sha256") != lock_sha or evidence.get("project_sha256") != project_sha:
        fail("owner approval evidence is not bound to the current Python closure")
    print("ultravox license gate: PASS")


def self_test() -> None:
    """Exercise revision/hash/license tamper cases without project imports/network."""
    with tempfile.TemporaryDirectory(prefix="ultravox-license-gate-") as directory:
        root = Path(directory)
        lock = root / "uv.lock"
        project = root / "pyproject.toml"
        manifest_path = root / "manifest.json"
        evidence_path = root / "approval.json"
        lock.write_text('version = 1\n\n[[package]]\nname = "demo"\nversion = "1.0"\n', encoding="utf-8")
        project.write_text("[project]\nname = 'demo'\n", encoding="utf-8")
        rows = package_rows(tomllib.loads(lock.read_text(encoding="utf-8")))
        license_rows = [
            {"id": "ultravox-audio-weight", "license": "MIT", "status": "REVIEWED", "payload_sha256": "e" * 64, "required_identity": "upstream_repo/upstream_revision/upstream_model_sha256"},
            {"id": "llama-companion-meta-conditional", "license": "Meta-ConditionalCommercial", "status": "REVIEWED", "payload_sha256": "1" * 64, "required_identity": "companion_repo/companion_revision/companion_model_sha256"},
            {"id": "python-closure", "license": "MIT", "status": "REVIEWED", "native_bundled_review": "self-test closure bound to demo 1.0"},
        ]
        expected = {
            "public_repo": "vokra/test",
            "public_revision": "a" * 40,
            "public_file": "test.gguf",
            "public_sha256": "b" * 64,
            "upstream_repo": "fixie/test",
            "upstream_revision": "c" * 40,
            "companion_repo": "meta-llama/test",
            "companion_revision": "d" * 40,
            "upstream_model_sha256": "e" * 64,
            "companion_model_sha256": "1" * 64,
        }
        manifest = {
            "gate_version": GATE_VERSION,
            "lock_sha256": sha256_bytes(lock.read_bytes()),
            "project_sha256": sha256_bytes(project.read_bytes()),
            "package_rows_sha256": canonical_digest(rows),
            "required_package_rows": rows,
            "package_review_rows": [{"name": "demo", "version": "1.0", "source": None, "license": "MIT", "status": "REVIEWED", "native_bundled_review": "self-test closure"}],
            "package_review_rows_sha256": canonical_digest([{"name": "demo", "version": "1.0", "source": None, "license": "MIT", "status": "REVIEWED", "native_bundled_review": "self-test closure"}]),
            "forbidden_dependencies": [],
            "license_rows": license_rows,
            "license_rows_sha256": canonical_digest(license_rows),
            "identities": {
                **expected,
                "source_files": {
                    "ultravox_model.py": {"bytes": 1, "sha256": "f" * 64},
                    "ultravox_processing.py": {"bytes": 1, "sha256": "0" * 64},
                    "ultravox_config.py": {"bytes": 1, "sha256": "1" * 64},
                },
            },
            "approval": {"signer": "self-test", "digest": "2" * 64, "status": "OWNER_SIGNOFF_APPROVED"},
            "publication": "NO_UPLOAD",
        }
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        evidence = {
            "signer": "self-test",
            "decision": "APPROVED",
            "approval_digest": "2" * 64,
            "manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
            "lock_sha256": manifest["lock_sha256"],
            "project_sha256": manifest["project_sha256"],
        }
        evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

        def expect_blocked(label: str, mutate: Any) -> None:
            candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
            mutate(candidate)
            manifest_path.write_text(json.dumps(candidate, sort_keys=True), encoding="utf-8")
            try:
                run(lock, project, manifest_path, evidence_path, expected)
            except SystemExit as exc:
                if exc.code != 2:
                    raise SystemExit(f"ultravox license gate self-test: {label} returned {exc.code}") from exc
            else:
                raise SystemExit(f"ultravox license gate self-test: {label} was accepted")
            manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

        run(lock, project, manifest_path, evidence_path, expected)
        duplicate_manifest = root / "duplicate-manifest.json"
        duplicate_manifest.write_text('{"gate_version":1,"gate_version":1}', encoding="utf-8")
        try:
            run(lock, project, duplicate_manifest, evidence_path, expected)
        except SystemExit as error:
            if error.code != 2:
                raise SystemExit(f"ultravox license gate self-test duplicate manifest returned {error.code}") from error
        else:
            raise SystemExit("ultravox license gate self-test duplicate manifest was accepted")
        duplicate_evidence = root / "duplicate-evidence.json"
        duplicate_evidence.write_text('{"signer":"self-test","signer":"self-test"}', encoding="utf-8")
        try:
            run(lock, project, manifest_path, duplicate_evidence, expected)
        except SystemExit as error:
            if error.code != 2:
                raise SystemExit(f"ultravox license gate self-test duplicate evidence returned {error.code}") from error
        else:
            raise SystemExit("ultravox license gate self-test duplicate evidence was accepted")
        expect_blocked("revision tamper", lambda value: value["identities"].update(public_revision="f" * 40))
        expect_blocked("hash tamper", lambda value: value["identities"].update(public_sha256="f" * 64))
        expect_blocked("license tamper", lambda value: value["license_rows"].__getitem__(0).update(license="Apache-2.0"))
        expect_blocked("unresolved status", lambda value: value["license_rows"].__getitem__(0).update(status="APPROVAL_PENDING"))
        expect_blocked("native closure tamper", lambda value: value["license_rows"].__getitem__(2).update(native_bundled_review=""))
        expect_blocked("native review placeholder", lambda value: value["license_rows"].__getitem__(2).update(native_bundled_review="OWNER_REVIEW_REQUIRED"))
        expect_blocked("package review tamper", lambda value: value["package_review_rows"].__getitem__(0).update(version="2.0"))
        expect_blocked("package license tamper", lambda value: value["package_review_rows"].__getitem__(0).update(license="Apache-2.0"))
        expect_blocked("approval tamper", lambda value: value["approval"].update(digest="3" * 64))
        expect_blocked("approval status tamper", lambda value: value["approval"].update(status="OWNER_SIGNOFF_REQUIRED"))
        expect_blocked("lock approval binding", lambda value: value.update(lock_sha256="f" * 64))
        expect_blocked("project approval binding", lambda value: value.update(project_sha256="f" * 64))
    print("license_gate.py self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--approval", type=Path)
    for field in (
        "public-repo",
        "public-revision",
        "public-file",
        "public-sha256",
        "upstream-repo",
        "upstream-revision",
        "companion-repo",
        "companion-revision",
        "upstream-model-sha256",
    ):
        parser.add_argument(f"--{field}", required=False)
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.lock, args.project, args.manifest, args.approval)):
            parser.error("--self-test accepts no path options")
        self_test()
        return 0
    fields = (
        "lock",
        "project",
        "manifest",
        "public_repo",
        "public_revision",
        "public_file",
        "public_sha256",
        "upstream_repo",
        "upstream_revision",
        "companion_repo",
        "companion_revision",
        "upstream_model_sha256",
    )
    if any(getattr(args, field) is None for field in fields):
        parser.error("all closure paths and fixed model/source identity options are required")
    expected = {
        "public_repo": args.public_repo,
        "public_revision": args.public_revision,
        "public_file": args.public_file,
        "public_sha256": args.public_sha256,
        "upstream_repo": args.upstream_repo,
        "upstream_revision": args.upstream_revision,
        "companion_repo": args.companion_repo,
        "companion_revision": args.companion_revision,
        "upstream_model_sha256": args.upstream_model_sha256,
    }
    run(args.lock, args.project, args.manifest, args.approval, expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
