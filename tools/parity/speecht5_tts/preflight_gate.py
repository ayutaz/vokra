#!/usr/bin/env python3
"""Offline, fail-closed gate for the SpeechT5 TTS parity closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import tomllib
from urllib.parse import urlparse

GATE_VERSION = 1
LOCK_SHA256 = "32d372461b63302e58e7ed3f5b7da1495f472a496717e5b85f6767911ad703cb"
PYPROJECT_SHA256 = "814d6445596fe0f778cc4efd4ce42cf54db0d90d040b1c8661ccec6cbd0ba6a1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
DEPENDENCY_KEYS = (
    frozenset({"name"}),
    frozenset({"name", "marker"}),
    frozenset({"name", "extra"}),
    frozenset({"name", "extra", "marker"}),
)
REGISTRY_PACKAGE_KEYS = (
    frozenset({"name", "version", "source", "sdist", "wheels"}),
    frozenset({"name", "version", "source", "wheels"}),
    frozenset({"name", "version", "source", "dependencies", "sdist", "wheels"}),
    frozenset({"name", "version", "source", "dependencies", "wheels"}),
)
REQUIRES_DIST_KEYS = (frozenset({"name", "specifier"}), frozenset({"name", "specifier", "extras"}), frozenset({"name", "specifier", "marker"}), frozenset({"name", "specifier", "extras", "marker"}), frozenset({"name", "specifier", "index"}), frozenset({"name", "git"}))
REVIEW_SENTINELS = {"", "none", "null", "unresolved", "pending", "pending_review", "owner_review_required", "review_required", "todo"}

TTS_FILES = {
    "pytorch_model.bin": ("585476837", "d60d28067349ef66b50d8cd643ae56b6d6b8f27def929bc4ef6fcad907954190"),
    "spm_char.model": ("238473", "7fcc48f3e225f627b1641db410ceb0c8649bd2b0c982e150b03f8be3728ab560"),
    "config.json": ("2062", "2caf62dde93699a90cfc35ff2a8de27b02b479a0c98881cbc55f9682cc43e258"),
    "tokenizer_config.json": ("232", "d589430c619db2d95ff0fa757a187b55ef5ea44eff7fb08a6fbf0e78e32a6247"),
    "added_tokens.json": ("40", "74be21ecff0a1fb1f304fe7c72ab21e4f0c046f8359fdf2852eb1b80967069ad"),
    "special_tokens_map.json": ("234", "2a098b61fe8ec4cfd7674832ca00b4268c07569743a4ad15c8164e8f60ebf981"),
}
MODEL_ROWS = [
    {"id": "tts:microsoft/speecht5_tts@30fcde30f19b87502b8435427b5f5068e401d5f6", "kind": "model", "status": "PENDING_REVIEW", "license": None, "native_review": None, "bundled_review": None},
    {"id": "tts-public:vokra/speecht5-tts@43cf6592038616d116a98fde4764d827ece59033", "kind": "model", "status": "PENDING_REVIEW", "license": None, "native_review": None, "bundled_review": None},
    {"id": "vocoder:microsoft/speecht5_hifigan@bb6f429406e86a9992357a972c0698b22043307d", "kind": "model", "status": "PENDING_REVIEW", "license": None, "native_review": None, "bundled_review": None},
    {"id": "source:transformers@5.5.0", "kind": "source", "status": "PENDING_REVIEW", "license": None, "native_review": None, "bundled_review": None},
    {"id": "source:bin_to_safetensors", "kind": "source", "status": "PENDING_REVIEW", "license": None, "native_review": None, "bundled_review": None},
]


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: Any) -> str:
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


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
    if set(lock) != {"version", "revision", "requires-python", "resolution-markers", "supported-markers", "package"}:
        raise ValueError("uv.lock top-level schema drifted")
    if not isinstance(lock["version"], int) or isinstance(lock["version"], bool) or lock["version"] != 1 or not isinstance(lock["revision"], int) or lock["revision"] != 3:
        raise ValueError("uv.lock version/revision types drifted")
    if not isinstance(lock["requires-python"], str) or not isinstance(lock["resolution-markers"], list) or not isinstance(lock["supported-markers"], list) or not isinstance(lock["package"], list):
        raise ValueError("uv.lock top-level value types drifted")
    if set(project) != {"project", "tool"} or set(project["project"]) != {"name", "version", "description", "requires-python", "dependencies"}:
        raise ValueError("pyproject schema drifted")
    if not isinstance(project["project"]["dependencies"], list) or set(project["tool"]) != {"uv"}:
        raise ValueError("pyproject dependency/tool schema drifted")
    uv = project["tool"]["uv"]
    if not isinstance(uv, dict) or set(uv) != {"package", "environments", "sources", "index"} or not isinstance(uv["package"], bool) or not isinstance(uv["environments"], list) or not isinstance(uv["sources"], dict) or not isinstance(uv["index"], list):
        raise ValueError("pyproject uv configuration drifted")
    seen: set[tuple[str, str, str]] = set()
    virtual = 0
    for package in lock["package"]:
        if not isinstance(package, dict) or not isinstance(package.get("name"), str) or not isinstance(package.get("version"), str):
            raise ValueError("uv.lock package identity is malformed")
        if not package["name"].strip() or not package["version"].strip():
            raise ValueError("uv.lock package name/version are malformed")
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
            if "sdist" in package:
                artifacts = [package["sdist"]]
                if not isinstance(package["sdist"], dict):
                    raise ValueError("uv.lock sdist is malformed")
            else:
                artifacts = []
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
            if "marker" in dependency and not isinstance(dependency["marker"], str):
                raise ValueError("uv.lock dependency marker drifted")
        key = (package["name"], package["version"], json.dumps(source, sort_keys=True))
        if key in seen:
            raise ValueError("uv.lock duplicate package identity")
        seen.add(key)
    if virtual != 1:
        raise ValueError("uv.lock must contain exactly one virtual root")


def is_unresolved(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    normalized = re.sub(r"\s+", "_", value.strip().casefold())
    return normalized in REVIEW_SENTINELS


def package_rows(lock: dict[str, Any]) -> list[dict[str, Any]]:
    packages = lock.get("package")
    if not isinstance(packages, list) or not packages:
        raise ValueError("uv.lock package rows are missing or empty")
    rows = []
    for package in packages:
        if (
            not isinstance(package, dict)
            or not isinstance(package.get("name"), str)
            or not package["name"].strip()
            or not isinstance(package.get("version"), str)
            or not package["version"].strip()
            or not isinstance(package.get("source"), (dict, type(None)))
            or not isinstance(package.get("resolution-markers", []), list)
            or not isinstance(package.get("dependencies", []), list)
        ):
            raise ValueError("uv.lock package row is malformed")
        rows.append({
            "name": package["name"], "version": package["version"],
            "source": package.get("source"),
            "resolution-markers": package.get("resolution-markers", []),
            "dependencies": package.get("dependencies", []),
        })
    return sorted(rows, key=lambda row: (row["name"], row["version"], json.dumps(row["source"], sort_keys=True)))


def blocked(reason: str) -> tuple[bool, str]:
    return False, reason


def validate(project: Path, manifest_path: Path, evidence_path: Path | None = None) -> tuple[bool, str]:
    lock_path, pyproject_path = project / "uv.lock", project / "pyproject.toml"
    if any(path.is_symlink() or not path.is_file() for path in (lock_path, pyproject_path, manifest_path)):
        return blocked("lock, project, or gate manifest is missing")
    try:
        manifest = strict_json_loads(manifest_path.read_text(encoding="utf-8"))
        lock_bytes, project_bytes = lock_path.read_bytes(), pyproject_path.read_bytes()
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return blocked(f"gate input is unreadable: {exc}")
    if not isinstance(manifest, dict):
        return blocked("gate manifest root is not an object")
    if set(manifest) != {"gate_version", "lock_sha256", "pyproject_sha256", "package_rows_sha256", "dependency_reviews", "dependency_reviews_sha256", "model_reviews", "tts_identity", "vocoder_identity", "public_tts_identity", "transformers_route", "approval_scope_sha256", "operator_approval"}:
        return blocked("gate manifest schema drifted")
    if manifest.get("gate_version") != GATE_VERSION:
        return blocked("unsupported gate version")
    if digest(lock_bytes) != LOCK_SHA256 or manifest.get("lock_sha256") != LOCK_SHA256:
        return blocked("uv.lock bytes are not the reviewed exact lock")
    if digest(project_bytes) != PYPROJECT_SHA256 or manifest.get("pyproject_sha256") != PYPROJECT_SHA256:
        return blocked("pyproject bytes are not the reviewed exact project")
    try:
        lock = tomllib.loads(lock_bytes.decode("utf-8"))
        project = tomllib.loads(project_bytes.decode("utf-8"))
        _validate_lock_shape(lock, project)
        rows = package_rows(lock)
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError) as exc:
        return blocked(f"uv.lock canonicalization failed: {exc}")
    if canonical(rows) != manifest.get("package_rows_sha256"):
        return blocked("canonical dependency version/source/marker/row digest drifted")
    reviews = manifest.get("dependency_reviews")
    review_fields = {"id", "name", "version", "source", "status", "license", "native_review", "bundled_review"}
    if not isinstance(reviews, list) or len(reviews) != len(rows):
        return blocked("version-keyed dependency license/native/bundled rows are missing")
    expected_reviews = {
        (row["name"], row["version"], json.dumps(row["source"], sort_keys=True)) for row in rows
    }
    review_keys = set()
    for row in reviews:
        if not isinstance(row, dict) or set(row) != review_fields:
            return blocked("dependency review row schema is not exact")
        key = (row.get("name"), row.get("version"), json.dumps(row.get("source"), sort_keys=True))
        if (
            key in review_keys
            or not isinstance(row.get("id"), str)
            or row.get("id") != f"{row.get('name')}@{row.get('version')}"
            or not isinstance(row.get("name"), str)
            or not isinstance(row.get("version"), str)
        ):
            return blocked("dependency review rows contain duplicate or malformed identities")
        review_keys.add(key)
    if review_keys != expected_reviews or canonical(reviews) != manifest.get("dependency_reviews_sha256"):
        return blocked("dependency review rows do not cover the exact lock")
    for row in reviews:
        if (
            is_unresolved(row.get("status"))
            or row.get("status") != "REVIEWED"
            or not isinstance(row.get("license"), str)
            or is_unresolved(row.get("license"))
            or any(is_unresolved(row.get(field)) for field in ("native_review", "bundled_review"))
        ):
            return blocked(f"dependency review is unresolved: {row.get('id')}")
    model_reviews = manifest.get("model_reviews")
    model_fields = {"id", "kind", "status", "license", "native_review", "bundled_review"}
    if not isinstance(model_reviews, list) or len(model_reviews) != len(MODEL_ROWS) or any(
        not isinstance(row, dict) or set(row) != model_fields for row in model_reviews
    ) or [
        {key: row.get(key) for key in ("id", "kind")} for row in model_reviews
    ] != [{key: row[key] for key in ("id", "kind")} for row in MODEL_ROWS]:
        return blocked("model/source/license identity rows drifted")
    for row in model_reviews:
        if (
            is_unresolved(row["status"])
            or row["status"] != "REVIEWED"
            or not isinstance(row.get("license"), str)
            or is_unresolved(row["license"])
            or any(is_unresolved(row[field]) for field in ("native_review", "bundled_review"))
        ):
            return blocked(f"model/source license review is unresolved: {row['id']}")
    expected_tts = {
        "repo": "microsoft/speecht5_tts", "revision": "30fcde30f19b87502b8435427b5f5068e401d5f6",
        "files": {name: list(value) for name, value in TTS_FILES.items()},
    }
    if manifest.get("tts_identity") != expected_tts:
        return blocked("fixed SpeechT5 TTS revision/tokenizer identity drifted")
    expected_vocoder = {
        "repo": "microsoft/speecht5_hifigan", "revision": "bb6f429406e86a9992357a972c0698b22043307d",
        "pytorch_model_sha256": "b171e9bcd8a2b50dc9780040478dfa26783a9ee4be012cf5776914f091d6887b",
    }
    if manifest.get("vocoder_identity") != expected_vocoder:
        return blocked("fixed HiFi-GAN revision/artifact identity drifted")
    if manifest.get("public_tts_identity") != {
        "repo": "vokra/speecht5-tts", "revision": "43cf6592038616d116a98fde4764d827ece59033",
        "bytes": 585382432, "sha256": "f26019f5e2f7106d834b0b1fd4f66286839e000350caad169388467452c8dde0",
    }:
        return blocked("public SpeechT5 GGUF identity drifted")
    if manifest.get("transformers_route") != "transformers==5.5.0 / SpeechT5ForTextToSpeech.generate_speech":
        return blocked("Transformers lock/dumper route is not bound")
    scope = {"lock_sha256": LOCK_SHA256, "pyproject_sha256": PYPROJECT_SHA256, "package_rows_sha256": manifest["package_rows_sha256"], "dependency_reviews": reviews, "model_reviews": model_reviews, "tts_identity": expected_tts, "vocoder_identity": expected_vocoder, "public_tts_identity": manifest["public_tts_identity"], "transformers_route": manifest["transformers_route"]}
    scope_sha = canonical(scope)
    if manifest.get("approval_scope_sha256") != scope_sha:
        return blocked("approval scope is not bound to exact closure")
    approval = manifest.get("operator_approval")
    if not isinstance(approval, dict) or approval.get("decision") != "APPROVED" or not isinstance(approval.get("signer"), str) or not approval["signer"] or approval.get("digest") != scope_sha or not HEX64.fullmatch(str(approval.get("digest"))):
        return blocked("operator approval is pending or invalid")
    evidence_path = evidence_path or manifest_path.with_name("license_gate_evidence.json")
    if evidence_path.is_symlink() or not evidence_path.is_file():
        return blocked("authenticated approval evidence is missing")
    try:
        evidence = strict_json_loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return blocked(f"approval evidence is unreadable: {exc}")
    if not isinstance(evidence, dict) or evidence.get("scope_sha256") != scope_sha or evidence.get("manifest_sha256") != digest(manifest_path.read_bytes()) or evidence.get("signer") != approval["signer"] or evidence.get("digest") != approval["digest"] or evidence.get("decision") != "APPROVED":
        return blocked("approval evidence is not authenticated to this manifest and scope")
    return True, "PASS"


def self_test() -> int:
    global LOCK_SHA256
    project = Path(__file__).resolve().parent
    manifest = project / "license_gate_manifest.json"
    ok, reason = validate(project, manifest)
    if ok or not ("unresolved" in reason or "canonicalization" in reason):
        print(f"speecht5 preflight gate: expected pending review, got {reason}", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory(prefix="speecht5-gate-") as directory:
        root = Path(directory); test_project = root / "project"; test_project.mkdir()
        shutil.copy2(project / "uv.lock", test_project / "uv.lock"); shutil.copy2(project / "pyproject.toml", test_project / "pyproject.toml")
        complete_lock = re.sub(
            r'(hash = "sha256:[0-9a-f]{64}")(, upload-time =)',
            r'\1, size = 1\2',
            (test_project / "uv.lock").read_text(encoding="utf-8"),
        )
        (test_project / "uv.lock").write_text(complete_lock, encoding="utf-8")
        LOCK_SHA256 = digest((test_project / "uv.lock").read_bytes())
        complete_rows = package_rows(tomllib.loads(complete_lock))
        base = json.loads(manifest.read_text(encoding="utf-8"))
        base["lock_sha256"] = LOCK_SHA256
        base["package_rows_sha256"] = canonical(complete_rows)
        for row in base["dependency_reviews"]:
            row.update(status="REVIEWED", license="SELF_TEST", native_review="SELF_TEST", bundled_review="SELF_TEST")
        for row in base["model_reviews"]:
            row.update(status="REVIEWED", license="SELF_TEST", native_review="SELF_TEST", bundled_review="SELF_TEST")
        # A boolean False is a reviewed negative conclusion, not a pending sentinel.
        base["dependency_reviews"][0]["native_review"] = False
        base["model_reviews"][0]["bundled_review"] = False
        base["dependency_reviews_sha256"] = canonical(base["dependency_reviews"])
        scope = {"lock_sha256": LOCK_SHA256, "pyproject_sha256": PYPROJECT_SHA256, "package_rows_sha256": base["package_rows_sha256"], "dependency_reviews": base["dependency_reviews"], "model_reviews": base["model_reviews"], "tts_identity": base["tts_identity"], "vocoder_identity": base["vocoder_identity"], "public_tts_identity": base["public_tts_identity"], "transformers_route": base["transformers_route"]}
        base["approval_scope_sha256"] = canonical(scope); base["operator_approval"] = {"decision":"APPROVED", "signer":"self-test", "digest":base["approval_scope_sha256"]}
        approved = root / "manifest.json"; approved.write_text(json.dumps(base), encoding="utf-8")
        evidence = root / "license_gate_evidence.json"; evidence.write_text(json.dumps({"scope_sha256":base["approval_scope_sha256"], "manifest_sha256":digest(approved.read_bytes()), "signer":"self-test", "digest":base["approval_scope_sha256"], "decision":"APPROVED"}), encoding="utf-8")
        ok, reason = validate(test_project, approved, evidence)
        if not ok:
            print(f"speecht5 preflight gate: approved baseline failed: {reason}", file=sys.stderr); return 1
        duplicate_manifest = root / "duplicate-manifest.json"
        duplicate_manifest.write_text('{"gate_version":1,"gate_version":1}', encoding="utf-8")
        ok, _ = validate(test_project, duplicate_manifest, evidence)
        if ok:
            print("speecht5 preflight gate: duplicate manifest key accepted", file=sys.stderr); return 1
        duplicate_evidence = root / "duplicate-evidence.json"
        duplicate_evidence.write_text('{"schema":"v1","schema":"v1"}', encoding="utf-8")
        ok, _ = validate(test_project, approved, duplicate_evidence)
        if ok:
            print("speecht5 preflight gate: duplicate evidence key accepted", file=sys.stderr); return 1
        tampered = json.loads(approved.read_text(encoding="utf-8")); tampered["vocoder_identity"]["revision"] = "0" * 40; bad = root / "tampered.json"; bad.write_text(json.dumps(tampered), encoding="utf-8")
        ok, _ = validate(test_project, bad, evidence)
        if ok:
            print("speecht5 preflight gate: tamper self-test failed", file=sys.stderr); return 1
        for field in ("lock_sha256", "pyproject_sha256"):
            candidate = json.loads(approved.read_text(encoding="utf-8"))
            candidate[field] = "0" * 64
            bad.write_text(json.dumps(candidate), encoding="utf-8")
            ok, _ = validate(test_project, bad, evidence)
            if ok:
                print(f"speecht5 preflight gate: {field} tamper accepted", file=sys.stderr); return 1
        for placeholder in ("OWNER_REVIEW_REQUIRED", " pending_review ", "REVIEW_REQUIRED", "TODO", "none", "null"):
            for field in ("status", "license", "native_review", "bundled_review"):
                candidate = json.loads(approved.read_text(encoding="utf-8"))
                candidate["dependency_reviews"][0][field] = placeholder
                candidate["dependency_reviews_sha256"] = canonical(candidate["dependency_reviews"])
                bad.write_text(json.dumps(candidate), encoding="utf-8")
                ok, _ = validate(test_project, bad, evidence)
                if ok:
                    print(f"speecht5 preflight gate: dependency placeholder accepted: {field}={placeholder!r}", file=sys.stderr); return 1
            for field in ("status", "license", "native_review", "bundled_review"):
                candidate = json.loads(approved.read_text(encoding="utf-8"))
                candidate["model_reviews"][0][field] = placeholder
                scope = {"lock_sha256": LOCK_SHA256, "pyproject_sha256": PYPROJECT_SHA256, "package_rows_sha256": candidate["package_rows_sha256"], "dependency_reviews": candidate["dependency_reviews"], "model_reviews": candidate["model_reviews"], "tts_identity": candidate["tts_identity"], "vocoder_identity": candidate["vocoder_identity"], "public_tts_identity": candidate["public_tts_identity"], "transformers_route": candidate["transformers_route"]}
                candidate["approval_scope_sha256"] = canonical(scope)
                bad.write_text(json.dumps(candidate), encoding="utf-8")
                ok, _ = validate(test_project, bad, evidence)
                if ok:
                    print(f"speecht5 preflight gate: model placeholder accepted: {field}={placeholder!r}", file=sys.stderr); return 1
        missing = json.loads(approved.read_text(encoding="utf-8")); missing["dependency_reviews"].pop(); missing["dependency_reviews_sha256"] = canonical(missing["dependency_reviews"]); bad.write_text(json.dumps(missing), encoding="utf-8")
        ok, _ = validate(test_project, bad, evidence)
        if ok:
            print("speecht5 preflight gate: missing dependency review accepted", file=sys.stderr); return 1
    print("speecht5 preflight gate: self-test PASS")
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
    passed, reason = validate(args.project, args.manifest, args.evidence)
    if not passed:
        print(f"speecht5 preflight gate: BLOCKED: {reason}", file=sys.stderr)
        raise SystemExit(2)
    print("speecht5 preflight gate: PASS")
