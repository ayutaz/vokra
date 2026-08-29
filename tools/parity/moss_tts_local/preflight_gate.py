#!/usr/bin/env python3
"""Offline, stdlib-only fail-closed gate for the MOSS-TTS Local composite."""
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

LOCK_SHA256 = "f4d225fad7bb7fbc5fa2342855a91aa74b929612d87c36f2c97294e4df49cc29"
PROJECT_SHA256 = "18cc19e890ca0b762985af3bd48216177a6a487133da97b2b172ab28087ee0b3"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PLACEHOLDERS = {"", "null", "none", "unresolved", "pending", "pending_review", "owner_review_required", "review_required", "todo"}
REPOSITORY = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
REVISION = "be7766a6735b98bd793f7c79fb720b4d0f5d13b8"
SOURCE_DIGESTS = {
    "configuration": "826f81f163b1b557ad13f83c4f35008f4fee5a6cb6311b4316ff3dbb25149411",
    "configuration_source": "ab6debcb92032cb9dc91ae80aed77dbadd2e59848208baef2b062bd6def3f3be",
    "modeling_source": "b0a66211943ae580b087f3e71495fea2f455701a4f6c29b6d3562218f7668c5f",
    "processing_source": "3fc5616b1ec3408162b7d859a7696725a40525313b20f9b31a06ee55c93bd7ad",
    "gpt2_source": "f2e877104669f1e6c7cd34680f0da1a8a159e032123ee56b660b63929b6c8989",
    "qwen3_source": "100163bd7ecf31a59bafacc0b032ace9339edc992a3eb4cc80662502e04e46f0",
    "processor_config": "db574bfebad009e05193196a63a4eeecd353eeca177ccfff28b9379d595d88b7",
}
LOCAL_IDENTITY = {
    "repository": REPOSITORY,
    "revision": REVISION,
    "tree": "full_hf_tree",
    "model_path": "model.safetensors",
    "model_bytes": 9100859544,
    "model_sha256": "608f1ff64bc6caa9be836060fc7c78a15c4658c4a07b8d73c78d6f70d1b39c23",
    "tensor_count": 438,
    "license": "apache-2.0",
    "source_roles": [
        {"role": "configuration", "path": "config.json", "sha256": SOURCE_DIGESTS["configuration"], "bytes": 10045, "git_blob_sha1": "c9e6b86a7d151bf800bb030831bba929163ad43f"},
        {"role": "configuration_source", "path": "configuration_moss_tts.py", "sha256": SOURCE_DIGESTS["configuration_source"], "bytes": 7160, "git_blob_sha1": "04b453edd7077170ebbd6a3ffb8167d9af8ac458"},
        {"role": "modeling_source", "path": "modeling_moss_tts.py", "sha256": SOURCE_DIGESTS["modeling_source"], "bytes": 26379, "git_blob_sha1": "4891b570440663b44f0304fe44e6df3d9baf55f8"},
        {"role": "processing_source", "path": "processing_moss_tts.py", "sha256": SOURCE_DIGESTS["processing_source"], "bytes": 37496, "git_blob_sha1": "2b15449bb1504a564207eb4c513430416864fce3"},
        {"role": "gpt2_source", "path": "gpt2_decoder.py", "sha256": SOURCE_DIGESTS["gpt2_source"], "bytes": 30896, "git_blob_sha1": "84c597cf1bcca240562dec1417a53fb19f47bfca"},
        {"role": "qwen3_source", "path": "qwen3_decoder.py", "sha256": SOURCE_DIGESTS["qwen3_source"], "bytes": 25473, "git_blob_sha1": "3756bd7033d88388d51a9e64d2fa0efd71959e84"},
        {"role": "processor_config", "path": "processor_config.json", "sha256": SOURCE_DIGESTS["processor_config"], "bytes": 210, "git_blob_sha1": "9bb021c7cc13f3e886f4be9ca326b68f6b42c461"},
    ],
}


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: Any) -> str:
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)


SCOPE_KEYS = ("lock_sha256", "project_sha256", "package_rows_sha256", "package_review_rows", "package_review_rows_sha256", "local_identity", "prompt_contract", "publication", "numeric_state", "composite_pcm")


def approval_scope(manifest: dict[str, Any]) -> str:
    return canonical({"schema": "moss-tts-local-approval-v1", **{key: manifest.get(key) for key in SCOPE_KEYS}})


def approval_manifest_digest(manifest: dict[str, Any]) -> str:
    """Bind the manifest while excluding the evidence digest's self-reference."""
    bound = json.loads(json.dumps(manifest))
    bound["approval"]["evidence_sha256"] = None
    return digest(json.dumps(bound, sort_keys=True, separators=(",", ":")).encode())


def reviewed(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return re.sub(r"\s+", "_", value.strip()).casefold() not in PLACEHOLDERS


LOCK_KEYS = {"version", "revision", "requires-python", "resolution-markers", "supported-markers", "package"}
PACKAGE_KEYS = (
    frozenset({"name", "version", "source", "sdist", "wheels"}),
    frozenset({"name", "version", "source", "wheels"}),
    frozenset({"name", "version", "source", "dependencies", "sdist", "wheels"}),
    frozenset({"name", "version", "source", "dependencies", "wheels"}),
    frozenset({"name", "version", "source", "dependencies", "metadata"}),
)
ARTIFACT_KEYS = {"url", "hash", "size", "upload-time"}


def _artifact_valid(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == ARTIFACT_KEYS
        and isinstance(value["url"], str)
        and value["url"].startswith("https://")
        and value["url"].split("/", 3)[2] in {"files.pythonhosted.org", "download-r2.pytorch.org"}
        and isinstance(value["hash"], str)
        and bool(re.fullmatch(r"sha256:[0-9a-f]{64}", value["hash"]))
        and isinstance(value["size"], int)
        and not isinstance(value["size"], bool)
        and value["size"] > 0
        and isinstance(value["upload-time"], str)
        and bool(value["upload-time"].strip())
    )


def lock_rows(lock: dict[str, Any]) -> list[dict[str, Any]]:
    if set(lock) != LOCK_KEYS or lock.get("version") != 1 or type(lock.get("version")) is not int or lock.get("revision") != 3 or type(lock.get("revision")) is not int:
        raise ValueError("lock top-level schema drifted")
    if not isinstance(lock.get("requires-python"), str) or not isinstance(lock.get("resolution-markers"), list) or any(not isinstance(item, str) for item in lock["resolution-markers"]) or not isinstance(lock.get("supported-markers"), list) or any(not isinstance(item, str) for item in lock["supported-markers"]):
        raise ValueError("lock marker schema malformed")
    packages = lock.get("package")
    if not isinstance(packages, list) or not packages:
        raise ValueError("lock package table missing")
    rows = []
    identities: set[tuple[str, str]] = set()
    virtual_count = 0
    for package in packages:
        if not isinstance(package, dict) or not isinstance(package.get("name"), str) or not isinstance(package.get("version"), str):
            raise ValueError("malformed lock package row")
        if frozenset(package) not in PACKAGE_KEYS or not package["name"].strip() or not package["version"].strip():
            raise ValueError("malformed lock package schema")
        identity = (package["name"], package["version"])
        if identity in identities:
            raise ValueError(f"duplicate lock package identity: {identity!r}")
        identities.add(identity)
        source = package.get("source")
        if not isinstance(source, dict) or len(source) != 1 or set(source) not in ({"registry"}, {"virtual"}):
            raise ValueError("malformed lock package source")
        if "registry" in source and (not isinstance(source["registry"], str) or source["registry"] not in {"https://pypi.org/simple", "https://download.pytorch.org/whl/cu126"}):
            raise ValueError("lock registry source must be HTTPS")
        if "virtual" in source:
            virtual_count += 1
            if source["virtual"] != ".":
                raise ValueError("lock virtual source must be '.'")
        markers = package.get("resolution-markers", [])
        if not isinstance(markers, list) or any(not isinstance(item, str) for item in markers):
            raise ValueError("malformed lock package markers")
        dependencies = package.get("dependencies", [])
        if not isinstance(dependencies, list):
            raise ValueError("malformed lock dependencies")
        for dependency in dependencies:
            if not isinstance(dependency, dict) or set(dependency) != {"name", "marker"} or not isinstance(dependency["name"], str) or not dependency["name"].strip() or not isinstance(dependency["marker"], str):
                raise ValueError("malformed lock dependency row")
        if "sdist" in package and not _artifact_valid(package["sdist"]):
            raise ValueError("malformed lock sdist artifact")
        if "registry" in source:
            expected_host = "download-r2.pytorch.org" if source["registry"] == "https://download.pytorch.org/whl/cu126" else "files.pythonhosted.org"
            artifacts = ([package["sdist"]] if "sdist" in package else []) + package.get("wheels", [])
            if any(not isinstance(item, dict) or item["url"].split("/", 3)[2] != expected_host for item in artifacts):
                raise ValueError("lock artifact host is not bound to its registry")
        if "wheels" in package and (not isinstance(package["wheels"], list) or not package["wheels"] or any(not _artifact_valid(item) for item in package["wheels"])):
            raise ValueError("malformed lock wheel artifacts")
        if "virtual" in source:
            metadata = package.get("metadata")
            if "sdist" in package or "wheels" in package or not isinstance(metadata, dict) or set(metadata) != {"requires-dist"} or not isinstance(metadata["requires-dist"], list):
                raise ValueError("virtual project metadata/artifacts malformed")
            for requirement in metadata["requires-dist"]:
                if not isinstance(requirement, dict) or set(requirement) not in ({"name", "specifier"}, {"name", "specifier", "index"}) or not isinstance(requirement.get("name"), str) or not requirement["name"].strip() or not isinstance(requirement.get("specifier"), str) or ("index" in requirement and requirement["index"] != "https://download.pytorch.org/whl/cu126"):
                    raise ValueError("malformed virtual requires-dist metadata")
        elif "metadata" in package:
            raise ValueError("non-virtual package metadata is not allowed")
        rows.append({"name": package["name"], "version": package["version"], "source": package.get("source"), "resolution-markers": package.get("resolution-markers", []), "dependencies": package.get("dependencies", [])})
    if virtual_count != 1:
        raise ValueError("lock must contain exactly one virtual project package")
    return sorted(rows, key=lambda row: (row["name"], row["version"]))


def artifact_error(lock: dict[str, Any]) -> str | None:
    for package in lock.get("package", []):
        source = package.get("source", {})
        if isinstance(source, dict) and source.get("virtual") is not None:
            continue
        artifacts = []
        for field in ("sdist", "wheels"):
            value = package.get(field)
            if field == "wheels":
                if value is not None and not isinstance(value, list):
                    return f"{package.get('name')}: wheels must be an array"
                artifacts.extend(value or [])
            elif value is not None:
                artifacts.append(value)
        if not artifacts:
            return f"{package.get('name')}: no resolver artifacts"
        for artifact in artifacts:
            if (not isinstance(artifact, dict) or set(artifact) != ARTIFACT_KEYS or not isinstance(artifact.get("url"), str)
                    or not artifact["url"].startswith("https://") or not isinstance(artifact.get("hash"), str)
                    or not re.fullmatch(r"sha256:[0-9a-f]{64}", artifact["hash"]) or not isinstance(artifact.get("size"), int)
                    or isinstance(artifact["size"], bool) or artifact["size"] <= 0 or not isinstance(artifact.get("upload-time"), str)
                    or not artifact["upload-time"].strip()):
                return f"{package.get('name')}: malformed resolver artifact"
    return None


def blocked(reason: str) -> tuple[bool, str]:
    return False, reason


def validate(project: Path, manifest_path: Path, approval_evidence: Path | None = None) -> tuple[bool, str]:
    lock_path, project_path = project / "uv.lock", project / "pyproject.toml"
    if any(path.is_symlink() or not path.is_file() for path in (lock_path, project_path, manifest_path)):
        return blocked("dedicated Local lock/project/manifest is missing")
    try:
        lock_bytes, project_bytes = lock_path.read_bytes(), project_path.read_bytes()
        manifest = load_json(manifest_path)
        lock_data = tomllib.loads(lock_bytes.decode())
        rows = lock_rows(lock_data)
        project_data = tomllib.loads(project_bytes.decode())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, tomllib.TOMLDecodeError, ValueError) as exc:
        return blocked(f"gate inputs are unreadable: {exc}")
    expected_project = {
        "name": "vokra-moss-tts-local-parity",
        "version": "0.1.0",
        "description": "Pinned official MOSS-TTS Local reference environment; VAST only",
        "requires-python": "==3.12.*",
        "dependencies": ["huggingface-hub==1.27.0", "numpy==2.3.5", "safetensors==0.8.0", "torch==2.7.1", "transformers==5.5.0"],
    }
    if (set(project_data) != {"project", "tool"} or project_data.get("project") != expected_project
            or set(project_data.get("tool", {})) != {"uv"}
            or project_data["tool"]["uv"] != {
                "package": False,
                "environments": ["python_full_version == '3.12.*' and platform_machine == 'x86_64' and sys_platform == 'linux'"],
                "index": [{"name": "pytorch-cuda", "url": "https://download.pytorch.org/whl/cu126", "explicit": True}],
                "sources": {"torch": {"index": "pytorch-cuda"}},
            }):
        return blocked("Local pyproject schema or resolver source drifted")
    if artifact_error(lock_data) is not None:
        return blocked(artifact_error(lock_data) or "resolver artifact metadata is missing")
    if digest(lock_bytes) != LOCK_SHA256:
        return blocked("Local uv.lock bytes are not the reviewed exact closure")
    if digest(project_bytes) != PROJECT_SHA256:
        return blocked("Local pyproject bytes are not the reviewed exact project")
    project_identity = project_data.get("project")
    virtual = [package for package in lock_data["package"] if package.get("source") == {"virtual": "."}]
    if not isinstance(project_identity, dict) or not isinstance(project_identity.get("name"), str) or not isinstance(project_identity.get("version"), str) or len(virtual) != 1 or (virtual[0]["name"], virtual[0]["version"]) != (project_identity["name"], project_identity["version"]):
        return blocked("Local virtual project row is not bound to pyproject identity")
    expected_manifest_keys = {"gate_version", "lock_sha256", "project_sha256", "package_rows_sha256", "package_review_rows_sha256", "package_review_rows", "local_identity", "prompt_contract", "publication", "numeric_state", "composite_pcm", "approval"}
    if not isinstance(manifest, dict) or set(manifest) != expected_manifest_keys or manifest.get("gate_version") != 1 or manifest.get("lock_sha256") != LOCK_SHA256 or manifest.get("project_sha256") != PROJECT_SHA256:
        return blocked("Local gate version or closure digest drifted")
    if manifest.get("package_rows_sha256") != canonical(rows):
        return blocked("Local canonical package graph drifted")
    reviews = manifest.get("package_review_rows")
    expected = [(row["name"], row["version"], row["source"]) for row in rows]
    if not isinstance(reviews, list) or any(not isinstance(row, dict) or set(row) != {"name", "version", "source", "license", "status", "native_bundled_review"} for row in reviews):
        return blocked("Local package review rows are not the exact lock set/schema")
    actual = [(row["name"], row["version"], row["source"]) for row in reviews]
    if sorted(actual) != sorted(expected) or len({(item[0], item[1]) for item in actual}) != len(expected):
        return blocked("Local package review rows are not the exact lock set/schema")
    if manifest.get("package_review_rows_sha256") != canonical(reviews):
        return blocked("Local package review rows digest drifted")
    if manifest.get("local_identity") != LOCAL_IDENTITY:
        return blocked("fixed Local full-tree identity contract drifted")
    if manifest.get("prompt_contract") != {"shape": ["rows", 13], "dtype": "u32le", "nonempty": True}:
        return blocked("prompt contract drifted")
    if manifest.get("publication") != "NO_UPLOAD" or manifest.get("numeric_state") != "MEASURED_NOT_GATED" or manifest.get("composite_pcm") != "COMPOSITE_PCM_NOT_RUN":
        return blocked("publication or measurement posture drifted")
    if not reviewed(LOCAL_IDENTITY.get("license")) or LOCAL_IDENTITY.get("model_path") != "model.safetensors" or not isinstance(LOCAL_IDENTITY.get("model_bytes"), int) or not HEX64.fullmatch(LOCAL_IDENTITY.get("model_sha256", "") or "") or any(not isinstance(row.get("bytes"), int) or not isinstance(row.get("path"), str) or not re.fullmatch(r"[0-9a-f]{40}", row.get("git_blob_sha1", "")) for row in LOCAL_IDENTITY["source_roles"]):
        return blocked("Local model/source identity is unresolved")
    for row in reviews:
        if row["status"] != "REVIEWED" or not all(reviewed(row.get(key)) for key in ("license", "native_bundled_review")):
            return blocked(f"Local package review unresolved: {row['name']}@{row['version']}")
    approval = manifest.get("approval")
    if not isinstance(approval, dict) or set(approval) != {"status", "signer", "scope_sha256", "digest", "evidence_sha256"} or approval.get("status") != "OWNER_SIGNOFF_APPROVED":
        return blocked("Local owner signoff remains required")
    expected_scope = approval_scope(manifest)
    if not reviewed(approval.get("signer")) or approval.get("scope_sha256") != expected_scope or approval.get("digest") != expected_scope or not HEX64.fullmatch(str(approval.get("scope_sha256"))) or not HEX64.fullmatch(str(approval.get("digest"))):
        return blocked("Local approval scope/signer is not canonical")
    if approval_evidence is None or not approval_evidence.is_file() or approval_evidence.is_symlink():
        return blocked("Local external approval evidence is missing")
    try:
        evidence = load_json(approval_evidence)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return blocked(f"Local external approval evidence is unreadable: {exc}")
    expected_evidence_keys = {"schema", "manifest_sha256", "scope_sha256", "signer", "approval_digest", "decision"}
    if not isinstance(evidence, dict) or set(evidence) != expected_evidence_keys or evidence.get("schema") != "moss-tts-local-approval-v1" or evidence.get("manifest_sha256") != approval_manifest_digest(manifest) or evidence.get("scope_sha256") != expected_scope or evidence.get("approval_digest") != expected_scope or evidence.get("signer") != approval.get("signer") or evidence.get("decision") != "APPROVED" or approval.get("evidence_sha256") != digest(approval_evidence.read_bytes()):
        return blocked("Local external approval evidence does not bind manifest/scope/signer/decision")
    return True, "PASS"


def self_test() -> int:
    project = Path(__file__).resolve().parent
    ok, reason = validate(project, project / "license_gate_manifest.json")
    if ok or "unresolved" not in reason:
        print(f"moss Local gate: expected unresolved blocker, got {reason}", file=sys.stderr)
        return 1
    for value in (None, "", " PENDING_REVIEW ", "owner review required"):
        if reviewed(value):
            print(f"placeholder normalization failed: {value!r}", file=sys.stderr)
            return 1
    if not reviewed("citation: TODO was resolved in authenticated evidence"):
        return 1
    lock_data = tomllib.loads((project / "uv.lock").read_bytes().decode())
    if artifact_error(lock_data) is not None:
        print(f"genuine resolver artifact metadata rejected: {artifact_error(lock_data)}", file=sys.stderr)
        return 1
    broken_lock = json.loads(json.dumps(lock_data))
    for package in broken_lock["package"]:
        if package.get("source", {}).get("virtual") is None:
            if "wheels" in package:
                package["wheels"][0].pop("size", None)
            else:
                package["sdist"].pop("size", None)
            break
    if artifact_error(broken_lock) is None:
        print("missing resolver artifact metadata was accepted", file=sys.stderr)
        return 1
    strict_cases: list[tuple[str, Any]] = []
    strict_cases.append(("top-level", {**json.loads(json.dumps(lock_data)), "unexpected": True}))
    malformed_package = json.loads(json.dumps(lock_data)); malformed_package["package"][0]["unexpected"] = True; strict_cases.append(("package-schema", malformed_package))
    malformed_source = json.loads(json.dumps(lock_data)); malformed_source["package"][0]["source"] = {"registry": True}; strict_cases.append(("source-schema", malformed_source))
    malformed_dependency = json.loads(json.dumps(lock_data)); dependency_package = next(item for item in malformed_dependency["package"] if item.get("dependencies")); dependency_package["dependencies"][0]["marker"] = True; strict_cases.append(("dependency-schema", malformed_dependency))
    malformed_virtual = json.loads(json.dumps(lock_data)); virtual_package = next(item for item in malformed_virtual["package"] if item.get("source") == {"virtual": "."}); virtual_package["source"] = {"virtual": "other"}; strict_cases.append(("virtual-source", malformed_virtual))
    for label, malformed in strict_cases:
        try:
            lock_rows(malformed)
        except ValueError:
            continue
        print(f"strict lock parser accepted malformed row: {label}", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory(prefix="moss-local-gate-") as directory:
        root = Path(directory); candidate = root / "manifest.json"
        shutil.copy2(project / "uv.lock", root / "uv.lock"); shutil.copy2(project / "pyproject.toml", root / "pyproject.toml")
        manifest = load_json(project / "license_gate_manifest.json")
        forged = json.loads(json.dumps(manifest)); forged["approval"] = {"status": "OWNER_SIGNOFF_APPROVED", "signer": "self-authored", "scope_sha256": approval_scope(forged), "digest": approval_scope(forged), "evidence_sha256": None}
        candidate.write_text(json.dumps(forged), encoding="utf-8")
        ok, reason = validate(root, candidate)
        if ok or "unresolved" not in reason:
            print(f"self-authored approval bypass accepted: {reason}", file=sys.stderr); return 1
        for label, mutate in (("missing", lambda rows: rows.pop()), ("extra", lambda rows: rows.append(dict(rows[-1]))), ("duplicate", lambda rows: rows.insert(1, dict(rows[0])))):
            altered = json.loads(json.dumps(manifest)); mutate(altered["package_review_rows"])
            candidate.write_text(json.dumps(altered), encoding="utf-8")
            ok, reason = validate(root, candidate)
            if ok or "package review rows" not in reason:
                print(f"package review {label} bypass accepted: {reason}", file=sys.stderr); return 1
        for label, value in (("not-list", {"name": "wrong"}), ("malformed-row", ["wrong-row"])):
            altered = json.loads(json.dumps(manifest)); altered["package_review_rows"] = value
            candidate.write_text(json.dumps(altered), encoding="utf-8")
            ok, reason = validate(root, candidate)
            if ok or "package review rows" not in reason:
                print(f"malformed package review {label} bypass accepted: {reason}", file=sys.stderr); return 1
        candidate.write_text('{"gate_version":1,"gate_version":1}', encoding="utf-8")
        ok, reason = validate(root, candidate)
        if ok or "duplicate JSON key" not in reason:
            print(f"duplicate tracked manifest key bypass accepted: {reason}", file=sys.stderr); return 1
        approved = json.loads(json.dumps(manifest))
        for row in approved["package_review_rows"]:
            row.update({"license": "authenticated package citation", "status": "REVIEWED", "native_bundled_review": "authenticated no native/bundled payload"})
        approved["package_review_rows_sha256"] = canonical(approved["package_review_rows"])
        signer = "owner@example.invalid"
        scope = approval_scope(approved)
        approved["approval"] = {"status": "OWNER_SIGNOFF_APPROVED", "signer": signer, "scope_sha256": scope, "digest": scope, "evidence_sha256": None}
        evidence = root / "approval.json"
        evidence_data = {"schema": "moss-tts-local-approval-v1", "manifest_sha256": approval_manifest_digest(approved), "scope_sha256": scope, "signer": signer, "approval_digest": scope, "decision": "APPROVED"}
        evidence.write_text(json.dumps(evidence_data, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        approved["approval"]["evidence_sha256"] = digest(evidence.read_bytes())
        candidate.write_text(json.dumps(approved, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        ok, reason = validate(root, candidate, evidence)
        if not ok:
            print(f"approved baseline was rejected: {reason}", file=sys.stderr); return 1
        evidence.write_text('{"schema":"moss-tts-local-approval-v1","schema":"moss-tts-local-approval-v1"}', encoding="utf-8")
        ok, reason = validate(root, candidate, evidence)
        if ok or "duplicate JSON key" not in reason:
            print(f"duplicate external evidence key bypass accepted: {reason}", file=sys.stderr); return 1
        evidence.write_text(json.dumps(evidence_data, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        for label, mutate in (("manifest", lambda value: value["prompt_contract"].update({"dtype": "wrong"})), ("scope", lambda value: value["approval"].update({"scope_sha256": "0" * 64})), ("signer", lambda value: value["approval"].update({"signer": "other@example.invalid"})), ("digest", lambda value: value["approval"].update({"digest": "1" * 64})), ("decision", lambda value: value), ("evidence-hash", lambda value: evidence.write_text(evidence.read_text() + "x", encoding="utf-8"))):
            altered = json.loads(json.dumps(approved)); mutate(altered)
            if label == "decision":
                evidence.write_text(json.dumps({**evidence_data, "decision": "REJECTED"}, sort_keys=True, separators=(",", ":")), encoding="utf-8")
            candidate.write_text(json.dumps(altered, sort_keys=True, separators=(",", ":")), encoding="utf-8")
            if validate(root, candidate, evidence)[0]:
                print(f"approval tamper bypass accepted: {label}", file=sys.stderr); return 1
            evidence.write_text(json.dumps(evidence_data, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        extra = json.loads(json.dumps(approved)); extra["unexpected"] = True; candidate.write_text(json.dumps(extra), encoding="utf-8")
        if validate(root, candidate, evidence)[0]:
            print("manifest extra key bypass accepted", file=sys.stderr); return 1
        evidence.unlink()
        evidence.symlink_to(candidate)
        candidate.write_text(json.dumps(approved), encoding="utf-8")
        if validate(root, candidate, evidence)[0]:
            print("symlink evidence bypass accepted", file=sys.stderr); return 1
    print("moss Local gate: self-test PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent); parser.add_argument("--manifest", type=Path); parser.add_argument("--approval-evidence", type=Path); parser.add_argument("--self-test", action="store_true"); args = parser.parse_args()
    if args.self_test: return self_test()
    ok, reason = validate(args.project, args.manifest or args.project / "license_gate_manifest.json", args.approval_evidence)
    if not ok: print(f"moss Local gate: BLOCKED: {reason}", file=sys.stderr); return 2
    print("moss Local gate: PASS"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
