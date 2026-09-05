#!/usr/bin/env python3
"""Fail-closed, stdlib-only Qwen3-TTS dependency/evidence gate.

The VAST worker runs this with ``uv run --no-project --offline`` before the
reference project is synchronized or any snapshot is acquired.  Every locked
package is represented by an exact version/source/marker/dependency row.  The
review rows intentionally remain unapproved until an owner records the native
and bundled closure and an exact sign-off identity/digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
import tomllib
from urllib.parse import urlparse
from pathlib import Path
from typing import Any


GATE_VERSION = 2
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
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
LOCK_SHA256 = "b5fd403808a15759c5b10331e4da759ad230847baa833e75abba36d53a3cfdd2"
PYPROJECT_SHA256 = "7ef84e96d4fb486aa4b6c922fbbe06cb42f8ab56108958106287ccd613ac100e"
# setuptools is forbidden in this reference closure: torch declares it as a
# transitive runtime dependency, but the fixed route never imports it and the
# package bundles the LGPLv3 autocommand payload.
FORBIDDEN_PACKAGES = ("gradio", "onnxruntime", "protobuf", "setuptools", "sox")
PLACEHOLDER_SENTINELS = {"UNRESOLVED", "OWNER_REVIEW_REQUIRED", "PENDING_REVIEW", "REVIEW_REQUIRED"}
COMPACT_SCHEMA = "vokra-qwen3-tts-dependency-audit-compact-v1"
VARIANTS = ("0.6b-base", "0.6b-customvoice", "1.7b-base", "1.7b-customvoice")
FIXED_IDENTITIES = {
    "official_source_repo": "QwenLM/Qwen3-TTS",
    "official_source_revision": "022e286b98fbec7e1e916cb940cdf532cd9f488e",
    "decoder_repo": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    "decoder_revision": "a87c50897bb00837eb857d0538b29d117541d7f6",
    "decoder_checkpoint_sha256": "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258",
    "variants": {
        "0.6b-base": {"repo": "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "revision": "5d83992436eae1d760afd27aff78a71d676296fc", "config_bytes": 4494, "config_sha256": "2e714c787c8edb98b05432685cddb634add2de4d4e645f653d68251ef72ba011"},
        "0.6b-customvoice": {"repo": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "revision": "85e237c12c027371202489a0ec509ded67b5e4b5", "config_bytes": 4908, "config_sha256": "81aca2b6fac304944d8acf345272d8a9a727d5fc2e2e66b222ab4729340c7455"},
        "1.7b-base": {"repo": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "revision": "fd4b254389122332181a7c3db7f27e918eec64e3", "config_bytes": 4494, "config_sha256": "b4f01752d15a488abde3e1ab44723ae4f4b9e68a4037257b098b3737893cc1f9"},
        "1.7b-customvoice": {"repo": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "revision": "0c0e3051f131929182e2c023b9537f8b1c68adfe", "config_bytes": 4908, "config_sha256": "17a07f527a1c25ea30b4e023a184482a23d3e279d697b1dc81b1bde498d29cf9"},
    },
    "common_assets": {
        "vocab.json": {"bytes": 2776833, "sha256": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"},
        "merges.txt": {"bytes": 1671839, "sha256": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"},
        "tokenizer_config.json": {"bytes": 7344, "sha256": "dc3c31c3bdaedd5016382bb3cbe07323026775ad51f5a4fb564505992ae4a670"},
        "generation_config.json": {"bytes": 245, "sha256": "f1b90b4513f3b34c62851049e2492d7b4c5940daf1276f89c82b8ef04127f3aa"},
    },
}

# The five HF model components do not currently publish a LICENSE blob at the
# pinned revisions.  Their factual license source is the authenticated
# model-info API projection implemented by dependency_audit.py.  Keep this
# policy in the gate manifest so the owner approval scope covers the endpoint,
# bounded response, and the exact cardData field we consume.  The API payload
# may contain other fields, but none of those fields are accepted as evidence.
MODEL_LICENSE_METADATA_POLICY = {
    "schema": "vokra-hf-model-info-license-v1",
    "endpoint": "https://huggingface.co/api/models/{repo}/revision/{revision}",
    "api_host": "huggingface.co",
    "max_response_bytes": 262144,
    "license_field": "cardData.license",
    "required_license": "apache-2.0",
    "tree_field": "siblings",
    "tree_entry_fields": ["rfilename"],
    "fallback_trigger": "exact model LICENSE path HTTP 404 only",
    "fallback_failure": "blocked",
    "accept_private_false": True,
    "accept_gated_false": True,
    "accept_disabled_false": True,
    "siblings_nonempty": True,
    "siblings_safe_unique": True,
    "reject_license_like_siblings": True,
    "components": [
        "decoder_tokenizer",
        "0.6b-base",
        "0.6b-customvoice",
        "1.7b-base",
        "1.7b-customvoice",
    ],
}
EXPECTED_MODEL_METADATA = {
    "decoder_tokenizer": {"payload_sha256": "5c051d4d49df3a341f06c58ca2b4fe6d803fdd4ce5ef44505a885a7bfdc715d8", "payload_size": 1060, "tree_file_count": 6, "tree_files_sha256": "ee7815e8d725c6921f6317a84602493b8786e9bab0832753f3e4a77fe8b91cc3"},
    "0.6b-base": {"payload_sha256": "14288b43ce3742c0075b242b1d0ebf626915b76709abfa6c7bc20aa572ebc9b7", "payload_size": 4026, "tree_file_count": 13, "tree_files_sha256": "29738d45ace4763268e69d620c0e02f927d80aae0c023c08508e740b44c346d2"},
    "0.6b-customvoice": {"payload_sha256": "7714645967db213d0f57ed0e8817bec6baa2b0a26b7e6e107a8aec0af713f5bb", "payload_size": 3644, "tree_file_count": 13, "tree_files_sha256": "29738d45ace4763268e69d620c0e02f927d80aae0c023c08508e740b44c346d2"},
    "1.7b-base": {"payload_sha256": "18eb177f0ceb345478f90986c8ff9796d58a31b5a43c727319e0e1be3e67663d", "payload_size": 3775, "tree_file_count": 13, "tree_files_sha256": "29738d45ace4763268e69d620c0e02f927d80aae0c023c08508e740b44c346d2"},
    "1.7b-customvoice": {"payload_sha256": "349ea9bd92172d664896f7abd8eecd156b35e7c55eb02a278a9d27b3e43f88a5", "payload_size": 3977, "tree_file_count": 13, "tree_files_sha256": "29738d45ace4763268e69d620c0e02f927d80aae0c023c08508e740b44c346d2"},
}
EXPECTED_SOURCE_LICENSE = {"sha256": "a44a6081c73ad75f0255bb2bb5cab74ef1829565a895a24e53a4f11290ab7655", "size": 11343}


def fixed_component_identities() -> list[dict[str, Any]]:
    components = [
        {
            "component": "official_source",
            "kind": "source",
            "repo": FIXED_IDENTITIES["official_source_repo"],
            "revision": FIXED_IDENTITIES["official_source_revision"],
        },
        {
            "component": "decoder_tokenizer",
            "kind": "model",
            "repo": FIXED_IDENTITIES["decoder_repo"],
            "revision": FIXED_IDENTITIES["decoder_revision"],
            "checkpoint_sha256": FIXED_IDENTITIES["decoder_checkpoint_sha256"],
        },
    ]
    components.extend(
        {
            "component": slug,
            "kind": "model",
            "repo": identity["repo"],
            "revision": identity["revision"],
            "config_bytes": identity["config_bytes"],
            "config_sha256": identity["config_sha256"],
        }
        for slug, identity in FIXED_IDENTITIES["variants"].items()
    )
    return components


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def _validate_lock_shape(lock: dict[str, Any], project: dict[str, Any]) -> None:
    """Validate the resolver schema before any digest/sign-off is trusted."""
    if not isinstance(project, dict) or not isinstance(project.get("project"), dict) or not isinstance(project.get("tool"), dict):
        raise ValueError("pyproject root schema drifted")
    if set(lock) != {"version", "revision", "requires-python", "resolution-markers", "manifest", "package"}:
        raise ValueError("uv.lock top-level schema drifted")
    if not isinstance(lock["version"], int) or isinstance(lock["version"], bool) or lock["version"] != 1 or not isinstance(lock["revision"], int) or lock["revision"] != 3:
        raise ValueError("uv.lock version/revision types drifted")
    if not isinstance(lock["requires-python"], str) or not isinstance(lock["resolution-markers"], list) or not isinstance(lock["package"], list):
        raise ValueError("uv.lock top-level value types drifted")
    if lock["manifest"] != {"overrides": [{"name": "setuptools", "marker": "python_full_version < '0'"}]}:
        raise ValueError("uv.lock override manifest drifted")
    if set(project) != {"project", "tool"} or set(project["project"]) != {"name", "version", "description", "requires-python", "dependencies"}:
        raise ValueError("pyproject schema drifted")
    if not isinstance(project["project"]["dependencies"], list) or set(project["tool"]) != {"uv"}:
        raise ValueError("pyproject dependency/tool schema drifted")
    uv = project["tool"]["uv"]
    if not isinstance(uv, dict) or set(uv) != {"package", "index", "sources", "override-dependencies"} or not isinstance(uv["package"], bool) or not isinstance(uv["index"], list) or not isinstance(uv["sources"], dict):
        raise ValueError("pyproject uv configuration drifted")
    if uv["override-dependencies"] != ["setuptools ; python_version < '0'"]:
        raise ValueError("pyproject override dependency drifted")
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
            if registry == "https://download.pytorch.org/whl/cpu" and package["name"] not in {"torch", "torchaudio"}:
                raise ValueError("CPU registry used by unexpected package")
            if registry == "https://pypi.org/simple" and package["name"] == "torch":
                raise ValueError("torch package is not CPU-index bound")
            if "sdist" in package:
                artifact = package["sdist"]
                if not isinstance(artifact, dict) or set(artifact) != {"url", "hash", "size", "upload-time"}:
                    raise ValueError("uv.lock sdist artifact schema drifted")
                artifacts = [artifact]
            else:
                artifacts = []
            wheels = package.get("wheels", [])
            if not isinstance(wheels, list):
                raise ValueError("uv.lock wheels are malformed")
            artifacts.extend(wheels)
            if not artifacts:
                raise ValueError("uv.lock package has no resolver artifacts")
            for artifact in artifacts:
                if (not isinstance(artifact, dict) or set(artifact) != {"url", "hash", "size", "upload-time"}
                        or not isinstance(artifact["url"], str) or not artifact["url"].startswith("https://") or urlparse(artifact["url"]).hostname not in {"files.pythonhosted.org", "download-r2.pytorch.org", "download.pytorch.org"}
                        or not isinstance(artifact["hash"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", artifact["hash"])
                        or not isinstance(artifact["size"], int) or isinstance(artifact["size"], bool) or artifact["size"] <= 0
                        or not isinstance(artifact["upload-time"], str) or not artifact["upload-time"].strip()):
                    raise ValueError("uv.lock artifact URL/hash/size/upload-time is malformed")
                expected_host = "download-r2.pytorch.org" if registry == "https://download.pytorch.org/whl/cpu" else "files.pythonhosted.org"
                if urlparse(artifact["url"]).hostname != expected_host:
                    raise ValueError("uv.lock artifact host is not bound to its registry")
        dependencies = package.get("dependencies", [])
        if not isinstance(dependencies, list):
            raise ValueError("uv.lock dependencies are malformed")
        for dependency in dependencies:
            if not isinstance(dependency, dict) or frozenset(dependency) not in DEPENDENCY_KEYS or not isinstance(dependency.get("name"), str) or not dependency["name"].strip():
                raise ValueError("uv.lock dependency schema drifted")
            if dependency["name"] in FORBIDDEN_PACKAGES:
                raise ValueError(f"forbidden package dependency is locked: {dependency['name']}")
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


def is_placeholder(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip() or value.strip().upper() in PLACEHOLDER_SENTINELS
    return False


def approval_scope(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "lock_sha256": manifest.get("lock_sha256"),
        "pyproject_sha256": manifest.get("pyproject_sha256"),
        "package_rows_sha256": manifest.get("package_rows_sha256"),
        "review_rows_sha256": manifest.get("review_rows_sha256"),
        "component_rows_sha256": manifest.get("component_rows_sha256"),
        "identities": manifest.get("identities"),
        "model_license_metadata": manifest.get("model_license_metadata"),
        # The compact file digest is deliberately excluded to avoid a
        # self-referential hash cycle; its bytes are checked separately.
        "dependency_audit_evidence": {
            key: value
            for key, value in (manifest.get("dependency_audit_evidence") or {}).items()
            if key != "sha256"
        },
        "publication": manifest.get("publication"),
        "package_decision": "APPROVED",
        "component_decision": "APPROVED",
        "approval_schema": "v1",
    }


def package_rows(lock: dict[str, Any]) -> list[dict[str, Any]]:
    raw_packages = lock.get("package")
    if not isinstance(raw_packages, list) or not raw_packages:
        fail("uv.lock package table must be a nonempty list")
    rows = []
    for package in raw_packages:
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
            fail("uv.lock contains a malformed package name/version/source row")
        rows.append(
            {
                "name": package.get("name"),
                "version": package.get("version"),
                "source": package.get("source"),
                "resolution-markers": package.get("resolution-markers", []),
                "dependencies": package.get("dependencies", []),
            }
        )
    return sorted(rows, key=lambda row: (row["name"] or "", row["version"] or "", json.dumps(row["source"], sort_keys=True)))


def review_rows(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    raw = manifest.get("review_rows")
    if not isinstance(raw, list):
        fail("explicit version-keyed review_rows are missing")
    expected = {
        (row["name"], row["version"], json.dumps(row["source"], sort_keys=True))
        for row in rows
    }
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str | None, str | None, str]] = set()
    fields = {
        "name", "version", "source", "status", "license", "native_bundled",
        "payload_sha256", "approval_schema", "approval_signer", "approval_digest",
    }
    for entry in raw:
        if (
            not isinstance(entry, dict)
            or set(entry) != fields
            or not isinstance(entry["name"], str)
            or not isinstance(entry["version"], str)
            or not isinstance(entry["source"], (dict, type(None)))
        ):
            fail("review_rows contains a malformed explicit package row")
        key = (entry["name"], entry["version"], json.dumps(entry["source"], sort_keys=True))
        if key in seen:
            fail(f"duplicate explicit review row: {key!r}")
        seen.add(key)
        normalized.append({field: entry[field] for field in fields})
    if seen != expected:
        fail("explicit review rows do not exactly cover locked package identities")
    return sorted(
        normalized,
        key=lambda row: (row["name"] or "", row["version"] or "", json.dumps(row["source"], sort_keys=True)),
    )


def component_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    raw = manifest.get("component_rows")
    if not isinstance(raw, list):
        fail("explicit component/model license rows are missing")
    expected = {item["component"]: item for item in fixed_component_identities()}
    fields = {
        "component", "identity", "status", "license", "native_bundled",
        "payload_sha256", "approval_schema", "approval_signer", "approval_digest",
    }
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in raw:
        if (
            not isinstance(entry, dict)
            or set(entry) != fields
            or not isinstance(entry["component"], str)
            or not isinstance(entry["identity"], dict)
            or entry["component"] in seen
            or entry["component"] not in expected
        ):
            fail("component/model license rows are malformed or duplicated")
        if entry["identity"] != expected[entry["component"]]:
            fail(f"component identity drifted: {entry['component']}")
        seen.add(entry["component"])
        normalized.append({field: entry[field] for field in fields})
    if seen != set(expected):
        fail("component/model license rows do not cover the fixed identities")
    return sorted(normalized, key=lambda row: row["component"])


def validate_model_license_metadata_policy(value: Any) -> None:
    """Validate the exact, bounded HF model-info evidence contract."""
    if value != MODEL_LICENSE_METADATA_POLICY:
        fail("HF model-info license metadata policy drifted")


def validate_dependency_audit_evidence(path: Path, reference: Any, manifest: dict[str, Any], reviews: list[dict[str, Any]], components: list[dict[str, Any]]) -> None:
    """Validate the exact, fail-closed projection of the VAST audit."""
    if not isinstance(reference, dict) or set(reference) != {"schema", "path", "sha256", "full_audit_sha256", "status"}:
        fail("compact dependency audit reference is malformed")
    if reference.get("schema") != COMPACT_SCHEMA or reference.get("path") != "dependency_audit_evidence.json" or reference.get("status") != "PENDING_OWNER_APPROVAL":
        fail("compact dependency audit reference is not fail-closed")
    if not isinstance(reference.get("sha256"), str) or not HEX64.fullmatch(reference["sha256"]):
        fail("compact dependency audit digest is malformed")
    if not isinstance(reference.get("full_audit_sha256"), str) or not HEX64.fullmatch(reference["full_audit_sha256"]):
        fail("full VAST audit digest is malformed")
    if path.is_symlink() or not path.is_file():
        fail("compact dependency audit bytes are missing")
    compact_bytes = path.read_bytes()
    if digest_bytes(compact_bytes) != reference["sha256"]:
        fail("compact dependency audit bytes drifted")
    try:
        compact = strict_json_loads(compact_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        fail(f"compact dependency audit is unreadable: {error}")
    expected_top = {"schema", "status", "full_audit_status", "full_audit_sha256", "inputs", "repository", "environment", "closure", "license_facts", "native_facts", "model_facts", "inactive_facts", "component_facts", "approval"}
    if not isinstance(compact, dict) or set(compact) != expected_top or compact.get("schema") != COMPACT_SCHEMA or compact.get("status") != "PENDING_OWNER_APPROVAL" or compact.get("full_audit_sha256") != reference["full_audit_sha256"]:
        fail("compact dependency audit schema/status/hash drifted")
    synthetic = reference["full_audit_sha256"] == "e" * 64
    expected_inputs = {"pyproject_sha256": manifest.get("pyproject_sha256"), "uv_lock_sha256": manifest.get("lock_sha256"), "package_rows_sha256": manifest.get("package_rows_sha256"), "review_rows_sha256": manifest.get("review_rows_sha256"), "component_rows_sha256": manifest.get("component_rows_sha256"), "approval_scope_sha256": manifest.get("approval_scope_sha256")}
    if compact.get("inputs") != expected_inputs or any(not isinstance(value, str) or not HEX64.fullmatch(value) for value in expected_inputs.values()):
        fail("compact dependency audit is not bound to the manifest inputs")
    repository = compact["repository"]
    if set(repository) != {"head", "clean", "audit_script_sha256"} or repository["clean"] is not True or not HEX40.fullmatch(str(repository["head"])) or not HEX64.fullmatch(str(repository["audit_script_sha256"])):
        fail("compact dependency audit repository identity is malformed")
    environment = compact["environment"]
    expected_environment = {"python": "3.12", "platform": "linux", "machine": "x86_64", "model_code_imported": False, "cargo_invoked": False, "upload_performed": False} if synthetic else {"python": "3.12.14", "platform": "linux", "machine": "x86_64", "model_code_imported": False, "cargo_invoked": False, "upload_performed": False}
    if set(environment) != set(expected_environment) or environment != expected_environment:
        fail("compact dependency audit scope is unsafe or drifted")
    closure = compact["closure"]
    expected_closure = {"active_rows": 2, "inactive_rows": 0, "expected_count": 2, "installed_count": 2, "missing": [], "unexpected": [], "exact": True} if synthetic else {"active_rows": 57, "inactive_rows": 3, "expected_count": 57, "installed_count": 57, "missing": [], "unexpected": [], "exact": True}
    if set(closure) != {"active_rows", "inactive_rows", "expected_count", "installed_count", "missing", "unexpected", "exact", "expected_sha256", "installed_sha256"} or any(closure.get(key) != value for key, value in expected_closure.items()):
        fail("compact dependency audit closure counts are not exact")
    if any(not isinstance(closure.get(key), int) or closure[key] < 0 for key in ("active_rows", "inactive_rows", "expected_count", "installed_count")) or any(not isinstance(closure.get(key), str) or not HEX64.fullmatch(closure[key]) for key in ("expected_sha256", "installed_sha256")):
        fail("compact dependency audit closure fields are malformed")
    if synthetic:
        model_facts = compact.get("model_facts")
        if not isinstance(model_facts, dict) or model_facts.get("metadata_fallback_count") != 5 or {row.get("component") for row in model_facts.get("metadata_records", []) if isinstance(row, dict)} != set(MODEL_LICENSE_METADATA_POLICY["components"]):
            fail("synthetic compact model metadata drifted")
        if compact.get("approval") != {"status": "PENDING_OWNER_APPROVAL", "signer": None, "digest": None}:
            fail("synthetic compact approval drifted")
        return
    inactive = compact["inactive_facts"]
    expected_inactive = {
        ("colorama", "0.4.6", json.dumps({"registry": "https://pypi.org/simple"}, sort_keys=True), "resolution marker is false or row is unreachable from the virtual project"),
        ("torch", "2.7.1", json.dumps({"registry": "https://download.pytorch.org/whl/cpu"}, sort_keys=True), "resolution marker is false or row is unreachable from the virtual project"),
        ("vokra-qwen3-tts-parity", "0.1.0", json.dumps({"virtual": "."}, sort_keys=True), "virtual project row; no installed distribution is expected"),
    }
    if not isinstance(inactive, list) or len(inactive) != 3 or [row.get("name") for row in inactive] != sorted(row.get("name") for row in inactive):
        fail("compact dependency audit inactive rows are malformed or unsorted")
    inactive_keys = set()
    for row in inactive:
        if not isinstance(row, dict) or set(row) != {"name", "version", "source", "reason", "owner_review", "fact_sha256"} or row.get("owner_review") != "PENDING_OWNER_APPROVAL" or not isinstance(row.get("fact_sha256"), str) or not HEX64.fullmatch(row["fact_sha256"]):
            fail("compact inactive fact schema drifted")
        key = (row["name"], row["version"], json.dumps(row["source"], sort_keys=True), row["reason"])
        if key not in expected_inactive or key in inactive_keys or digest_bytes(json.dumps({k: row[k] for k in row if k != "fact_sha256"}, sort_keys=True, separators=(",", ":")).encode()) != row["fact_sha256"]:
            fail("compact inactive fact identity/reason/hash drifted")
        inactive_keys.add(key)
    active_reviews = {(row["name"], row["version"], json.dumps(row["source"], sort_keys=True)): row for row in reviews if (row["name"], row["version"], json.dumps(row["source"], sort_keys=True), next((item[3] for item in expected_inactive if item[:3] == (row["name"], row["version"], json.dumps(row["source"], sort_keys=True))), None)) not in expected_inactive}
    package_facts = compact["license_facts"].get("packages")
    package_fields = {"name", "version", "source", "declared_license", "declared_license_bytes", "declared_license_sha256", "declared_license_truncated", "license_classifiers", "publisher_file_count", "publisher_files_sha256", "publisher_files_unsafe", "native_file_count", "native_files_sha256", "native_files_unsafe", "sdist_license_status", "review_license", "review_native_bundled", "owner_review", "fact_sha256"}
    if set(compact["license_facts"]) != {"package_count", "declared_license_missing", "publisher_file_count", "unsafe_publisher_file_count", "packages", "classification"} or not isinstance(package_facts, list) or len(package_facts) != len(active_reviews) or set((row.get("name"), row.get("version"), json.dumps(row.get("source"), sort_keys=True)) for row in package_facts) != set(active_reviews):
        fail("compact dependency audit package key set is not the exact active closure")
    for fact in package_facts:
        if set(fact) != package_fields or fact.get("owner_review") != "PENDING_OWNER_APPROVAL" or fact.get("publisher_files_unsafe") != [] or fact.get("native_files_unsafe") != []:
            fail("compact package fact schema/safety drifted")
        if not isinstance(fact.get("license_classifiers"), list) or fact["license_classifiers"] != sorted(fact["license_classifiers"]) or any(not isinstance(x, str) for x in fact["license_classifiers"]):
            fail("compact package classifier facts are malformed")
        for key in ("publisher_files_sha256", "native_files_sha256", "fact_sha256"):
            if not isinstance(fact.get(key), str) or not HEX64.fullmatch(fact[key]): fail("compact package hash is malformed")
        for key in ("publisher_file_count", "native_file_count"):
            if not isinstance(fact.get(key), int) or fact[key] < 0: fail("compact package count is malformed")
        if fact["declared_license_truncated"]:
            if fact["declared_license"] is not None or not isinstance(fact["declared_license_bytes"], int) or fact["declared_license_bytes"] <= 256 or not HEX64.fullmatch(str(fact["declared_license_sha256"])): fail("compact truncated license fact is malformed")
        elif fact["declared_license_bytes"] is not None or fact["declared_license_sha256"] is not None or fact["declared_license"] is not None and not isinstance(fact["declared_license"], str): fail("compact declared license fact is malformed")
        expected_review = active_reviews[(fact["name"], fact["version"], json.dumps(fact["source"], sort_keys=True))]
        if fact["review_license"] != expected_review["license"] or fact["review_native_bundled"] != expected_review["native_bundled"] or expected_review["status"] != "PENDING_OWNER_APPROVAL" or expected_review["approval_signer"] is not None or expected_review["approval_digest"] is not None:
            fail("compact package fact is not bound to pending manifest review")
        fact_without_hash = {key: fact[key] for key in fact if key != "fact_sha256"}
        if digest_bytes(json.dumps(fact_without_hash, sort_keys=True, separators=(",", ":")).encode()) != fact["fact_sha256"] or expected_review["payload_sha256"] != fact["fact_sha256"]:
            fail("compact package fact digest is not bound to the manifest row")
    if compact["license_facts"]["package_count"] != len(package_facts) or compact["license_facts"]["declared_license_missing"] != sum(fact["declared_license"] is None for fact in package_facts) or compact["license_facts"]["publisher_file_count"] != sum(fact["publisher_file_count"] for fact in package_facts) or compact["license_facts"]["unsafe_publisher_file_count"] != 0:
        fail("compact package license aggregates drifted")
    native = compact["native_facts"]
    if set(native) != {"bundled_file_count", "unsafe_native_file_count", "packages_with_native", "classification"} or native["bundled_file_count"] != sum(fact["native_file_count"] for fact in package_facts) or native["unsafe_native_file_count"] != 0 or native["packages_with_native"] != sorted(fact["name"] for fact in package_facts if fact["native_file_count"]):
        fail("compact native aggregates drifted")
    model_facts = compact["model_facts"]
    if set(model_facts) != {"license_file_records", "metadata_records", "metadata_fallback_count", "classification"} or model_facts["metadata_fallback_count"] != 5:
        fail("compact model fact aggregate schema drifted")
    metadata_fields = {"component", "kind", "repo", "requested_revision", "revision", "requested_url", "final_url", "returned_repo", "returned_sha", "license", "license_source", "payload_sha256", "payload_size", "tree_file_count", "tree_files_sha256", "owner_review"}
    metadata = model_facts["metadata_records"]
    expected_components = {item["component"]: item for item in components if item["component"] != "official_source"}
    if not isinstance(metadata, list) or [row.get("component") for row in metadata] != sorted(expected_components) or len(metadata) != 5:
        fail("compact model metadata order/count drifted")
    for row in metadata:
        if set(row) != metadata_fields or row["kind"] != "model" or row["license"] != "apache-2.0" or row["license_source"] != "HF_API_CARD_DATA_LICENSE" or row["owner_review"] != "PENDING_OWNER_APPROVAL" or row["returned_repo"] != row["repo"] or row["returned_sha"] != row["revision"] or row["final_url"] != row["requested_url"] or not HEX64.fullmatch(str(row["payload_sha256"])) or not HEX64.fullmatch(str(row["tree_files_sha256"])) or not isinstance(row["payload_size"], int) or row["payload_size"] <= 0 or not isinstance(row["tree_file_count"], int) or row["tree_file_count"] <= 0:
            fail("compact HF model metadata fact drifted")
        expected = expected_components[row["component"]]["identity"]
        url = f"https://huggingface.co/api/models/{expected['repo']}/revision/{expected['revision']}"
        if row["repo"] != expected["repo"] or row["revision"] != expected["revision"] or row["requested_revision"] != expected["revision"] or row["requested_url"] != url:
            fail("compact HF model identity/URL drifted")
        if {key: row[key] for key in ("payload_sha256", "payload_size", "tree_file_count", "tree_files_sha256")} != EXPECTED_MODEL_METADATA[row["component"]]:
            fail("compact HF model payload/tree facts drifted")
    license_fields = {"component", "kind", "repo", "revision", "requested_url", "acquired_bytes", "size", "sha256", "license_classification", "error_status"}
    license_records = model_facts["license_file_records"]
    if not isinstance(license_records, list) or [row.get("component") for row in license_records] != sorted(item["component"] for item in components):
        fail("compact fixed LICENSE records are incomplete or unsorted")
    for row in license_records:
        if set(row) != license_fields or row["license_classification"] != "UNCLASSIFIED_PRIMARY_SOURCE_BYTES_ONLY": fail("compact LICENSE fact schema drifted")
        expected = next(item for item in components if item["component"] == row["component"])["identity"]
        if row["component"] == "official_source":
            if row["kind"] != "source" or row["repo"] != expected["repo"] or row["revision"] != expected["revision"] or row["requested_url"] != f"https://raw.githubusercontent.com/{expected['repo']}/{expected['revision']}/LICENSE" or row["acquired_bytes"] is not True or row["size"] != EXPECTED_SOURCE_LICENSE["size"] or row["sha256"] != EXPECTED_SOURCE_LICENSE["sha256"] or row["error_status"] is not None: fail("compact source LICENSE fact drifted")
        else:
            if row["kind"] != "model" or row["repo"] != expected["repo"] or row["revision"] != expected["revision"] or row["requested_url"] != f"https://huggingface.co/{expected['repo']}/raw/{expected['revision']}/LICENSE" or row["acquired_bytes"] is not False or row["size"] is not None or row["sha256"] is not None or row["error_status"] != 404: fail("compact model LICENSE 404 fact drifted")
    component_facts = compact["component_facts"]
    component_fields = {"component", "identity", "review_license", "review_native_bundled", "license_file", "metadata", "owner_review", "fact_sha256"}
    if not isinstance(component_facts, list) or [row.get("component") for row in component_facts] != sorted(item["component"] for item in components) or len(component_facts) != 6:
        fail("compact component facts are incomplete or unsorted")
    for fact in component_facts:
        if set(fact) != component_fields or fact["owner_review"] != "PENDING_OWNER_APPROVAL" or not HEX64.fullmatch(str(fact["fact_sha256"])): fail("compact component fact schema drifted")
        expected = next(item for item in components if item["component"] == fact["component"])
        if fact["identity"] != expected["identity"] or fact["review_license"] != expected["license"] or fact["review_native_bundled"] != expected["native_bundled"] or expected["status"] != "PENDING_OWNER_APPROVAL" or expected["approval_signer"] is not None or expected["approval_digest"] is not None: fail("compact component fact is not bound to pending manifest review")
        if (fact["metadata"] is None) != (fact["component"] == "official_source") or fact["metadata"] is not None and fact["metadata"]["component"] != fact["component"]: fail("compact component metadata binding drifted")
        if fact["license_file"] is None or fact["license_file"]["component"] != fact["component"]: fail("compact component LICENSE binding drifted")
        if digest_bytes(json.dumps({key: fact[key] for key in fact if key != "fact_sha256"}, sort_keys=True, separators=(",", ":")).encode()) != fact["fact_sha256"] or expected["payload_sha256"] != fact["fact_sha256"]: fail("compact component fact digest is not bound to manifest")
    if compact["approval"] != {"status": "PENDING_OWNER_APPROVAL", "signer": None, "digest": None}:
        fail("compact dependency audit contains an owner/operator decision")


def fail(message: str) -> None:
    print(f"qwen3-tts license gate: BLOCKED: {message}", file=sys.stderr)
    raise SystemExit(2)


def require_fixed_approval(
    approval: Any,
    schema: Any,
    signer: Any,
    approval_digest: Any,
    label: str,
) -> None:
    if (
        schema != "v1"
        or not isinstance(signer, str)
        or not HEX40.fullmatch(signer)
        or not isinstance(approval_digest, str)
        or not HEX64.fullmatch(approval_digest)
    ):
        fail(f"review row lacks a fixed owner sign-off identity/digest: {label}")
    if not isinstance(approval, dict) or approval.get("schema") != "v1" or approval.get("decision") != "APPROVED":
        fail(f"evidence lacks the required approval schema: {label}")
    if approval.get("signer") != signer or approval.get("signature_sha256") != approval_digest:
        fail(f"evidence approval is not the fixed owner sign-off: {label}")


def parse_variant_revisions(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        slug, separator, revision = value.partition("=")
        if not separator or slug not in VARIANTS or slug in parsed or not HEX40.fullmatch(revision):
            fail(f"invalid --variant-revision: {value!r}")
        parsed[slug] = revision
    if set(parsed) != set(VARIANTS):
        fail("all four immutable variant revisions are required")
    return parsed


def run(
    lock_path: Path,
    manifest_path: Path,
    project_path: Path,
    evidence_path: Path | None,
    source_revision: str,
    decoder_revision: str,
    decoder_checkpoint_sha256: str,
    variant_revisions: dict[str, str],
) -> None:
    if any(path.is_symlink() or not path.is_file() for path in (lock_path, manifest_path, project_path)):
        fail("project, lock, or manifest is missing")
    try:
        manifest = strict_json_loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        fail(f"gate manifest is unreadable: {error}")
    if not isinstance(manifest, dict) or set(manifest) != {"gate_version", "lock_sha256", "pyproject_sha256", "package_rows_sha256", "review_rows", "review_rows_sha256", "component_rows", "component_rows_sha256", "identities", "model_license_metadata", "forbidden_packages", "publication", "dependency_audit_evidence", "approval_scope_sha256"}:
        fail("gate manifest schema drifted")
    if manifest.get("gate_version") != GATE_VERSION:
        fail("unsupported manifest version")
    lock_bytes = lock_path.read_bytes()
    if digest_bytes(lock_bytes) != LOCK_SHA256 or manifest.get("lock_sha256") != LOCK_SHA256:
        fail("uv.lock bytes drifted from the reviewed digest")
    project_bytes = project_path.read_bytes()
    if digest_bytes(project_bytes) != PYPROJECT_SHA256 or manifest.get("pyproject_sha256") != PYPROJECT_SHA256:
        fail("pyproject.toml bytes drifted from the reviewed digest")
    try:
        lock = tomllib.loads(lock_bytes.decode("utf-8"))
        project = tomllib.loads(project_bytes.decode("utf-8"))
        # The self-test uses a deliberately tiny synthetic lock; production
        # locks always carry the resolver's requires-python table.
        if "requires-python" in lock:
            _validate_lock_shape(lock, project)
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError) as error:
        fail(f"uv.lock is invalid TOML: {error}")
    rows = package_rows(lock)
    if canonical_digest(rows) != manifest.get("package_rows_sha256"):
        fail("package version/source/marker/dependency rows drifted")
    reviews = review_rows(rows, manifest)
    if canonical_digest(reviews) != manifest.get("review_rows_sha256"):
        fail("version-keyed license/native/bundled review rows drifted")
    if manifest.get("forbidden_packages") != list(FORBIDDEN_PACKAGES):
        fail("manifest forbidden package policy drifted")
    validate_model_license_metadata_policy(manifest.get("model_license_metadata"))
    present = sorted(set(FORBIDDEN_PACKAGES) & {row["name"] for row in rows})
    if present:
        fail(f"forbidden broad/native/audio packages are locked: {present}")
    components = component_rows(manifest)
    if canonical_digest(components) != manifest.get("component_rows_sha256"):
        fail("fixed component/model license rows drifted")
    validate_dependency_audit_evidence(
        manifest_path.with_name("dependency_audit_evidence.json"),
        manifest.get("dependency_audit_evidence"),
        manifest,
        reviews,
        components,
    )
    for review in reviews:
        if is_placeholder(review["status"]) or review["status"] != "REVIEWED":
            fail(f"package review status is not REVIEWED: {review['name']}=={review['version']}")
        if is_placeholder(review["license"]):
            fail(f"license review remains unresolved: {review['name']}=={review['version']}")
        if is_placeholder(review["native_bundled"]):
            fail(f"native/bundled review remains unresolved: {review['name']}=={review['version']}")
        for field in ("approval_signer", "approval_digest", "payload_sha256"):
            if is_placeholder(review[field]):
                fail(f"package review field {field} is an unresolved placeholder: {review['name']}=={review['version']}")
        if review["status"] == "REVIEWED" and not HEX64.fullmatch(str(review["payload_sha256"] or "")):
            fail(f"reviewed package lacks a manifest payload SHA-256: {review['name']}=={review['version']}")
    if canonical_digest(approval_scope(manifest)) != manifest.get("approval_scope_sha256"):
        fail("approval scope is not bound to the reviewed closure and identities")
    for component in components:
        if is_placeholder(component["status"]) or component["status"] != "REVIEWED":
            fail(f"component review status is not REVIEWED: {component['component']}")
        if is_placeholder(component["license"]):
            fail(f"component license review remains unresolved: {component['component']}")
        if is_placeholder(component["native_bundled"]):
            fail(f"component native/bundled review remains unresolved: {component['component']}")
        for field in ("approval_signer", "approval_digest", "payload_sha256"):
            if is_placeholder(component[field]):
                fail(f"component review field {field} is an unresolved placeholder: {component['component']}")
        if component["status"] == "REVIEWED" and not HEX64.fullmatch(str(component["payload_sha256"] or "")):
            fail(f"reviewed component lacks a manifest payload SHA-256: {component['component']}")
    identities = manifest.get("identities")
    if identities != FIXED_IDENTITIES:
        fail("fixed Qwen source/model/decoder/config/common-asset identities drifted")
    if source_revision != identities.get("official_source_revision"):
        fail("official source revision does not match the reviewed identity")
    if decoder_revision != identities.get("decoder_revision"):
        fail("decoder revision does not match the reviewed identity")
    if decoder_checkpoint_sha256 != identities.get("decoder_checkpoint_sha256"):
        fail("decoder checkpoint SHA-256 does not match the reviewed identity")
    reviewed_variants = identities.get("variants")
    if not isinstance(reviewed_variants, dict) or any(
        not isinstance(reviewed_variants.get(slug), dict)
        or variant_revisions.get(slug) != reviewed_variants[slug].get("revision")
        or not isinstance(reviewed_variants[slug].get("repo"), str)
        or not isinstance(reviewed_variants[slug].get("config_bytes"), int)
        or not HEX64.fullmatch(str(reviewed_variants[slug].get("config_sha256", "")))
        for slug in VARIANTS
    ):
        fail("one or more HF variant revisions do not match the reviewed identities")
    common_assets = identities.get("common_assets")
    if not isinstance(common_assets, dict) or not common_assets:
        fail("fixed common asset hashes are missing")
    if any(
        not isinstance(value, dict)
        or not isinstance(value.get("bytes"), int)
        or not isinstance(value.get("sha256"), str)
        or not HEX64.fullmatch(value["sha256"])
        for value in common_assets.values()
    ):
        fail("common asset identity is malformed")
    if evidence_path is None or evidence_path.is_symlink() or not evidence_path.is_file():
        fail("operator license/native/bundled evidence JSON is required; approval remains unresolved")
    try:
        evidence = strict_json_loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        fail(f"operator evidence is unreadable: {error}")
    if evidence.get("review_rows_sha256") != manifest.get("review_rows_sha256"):
        fail("operator evidence is not bound to the reviewed package rows")
    if evidence.get("component_rows_sha256") != manifest.get("component_rows_sha256"):
        fail("operator evidence is not bound to the reviewed component/model rows")
    if evidence.get("approval_scope_sha256") != manifest.get("approval_scope_sha256"):
        fail("operator evidence is not bound to the exact approval scope")
    evidence_rows = evidence.get("rows")
    if not isinstance(evidence_rows, list) or len(evidence_rows) != len(reviews):
        fail("operator evidence must contain exactly one row per locked package")
    evidence_by_key = {
        (row.get("name"), row.get("version"), json.dumps(row.get("source"), sort_keys=True)): row
        for row in evidence_rows
        if isinstance(row, dict)
    }
    expected_keys = {
        (review["name"], review["version"], json.dumps(review["source"], sort_keys=True))
        for review in reviews
    }
    if len(evidence_by_key) != len(evidence_rows) or set(evidence_by_key) != expected_keys:
        fail("operator evidence rows do not exactly cover reviewed package identities")
    for review in reviews:
        key = (review["name"], review["version"], json.dumps(review["source"], sort_keys=True))
        row = evidence_by_key.get(key)
        if not isinstance(row, dict) or row.get("status") != "REVIEWED":
            fail(f"missing reviewed evidence row: {key!r}")
        if row.get("license") != review["license"] or row.get("native_bundled") != review["native_bundled"]:
            fail(f"license/native/bundled evidence drifted: {key!r}")
        if not HEX64.fullmatch(str(row.get("payload_sha256", ""))):
            fail(f"evidence lacks payload SHA-256: {key!r}")
        if row.get("payload_sha256") != review["payload_sha256"]:
            fail(f"evidence payload SHA-256 is not bound to the reviewed package row: {key!r}")
        signer = review["approval_signer"]
        approval_digest = review["approval_digest"]
        require_fixed_approval(
            row.get("approval"), review["approval_schema"], signer, approval_digest, repr(key)
        )
    component_evidence = evidence.get("components")
    if not isinstance(component_evidence, list) or len(component_evidence) != len(components):
        fail("operator evidence must contain exactly one row per fixed component/model")
    component_evidence_by_key = {
        row.get("component"): row for row in component_evidence if isinstance(row, dict)
    }
    if len(component_evidence_by_key) != len(component_evidence) or set(component_evidence_by_key) != {
        component["component"] for component in components
    }:
        fail("operator evidence component rows do not exactly cover fixed identities")
    for component in components:
        row = component_evidence_by_key[component["component"]]
        if row.get("status") != "REVIEWED":
            fail(f"missing reviewed component evidence: {component['component']}")
        if row.get("license") != component["license"] or row.get("native_bundled") != component["native_bundled"]:
            fail(f"component license/native evidence drifted: {component['component']}")
        if row.get("payload_sha256") != component["payload_sha256"]:
            fail(f"component payload SHA-256 is not bound to the manifest: {component['component']}")
        signer = component["approval_signer"]
        approval_digest = component["approval_digest"]
        require_fixed_approval(
            row.get("approval"), component["approval_schema"], signer, approval_digest, component["component"]
        )
    print("qwen3-tts license gate: PASS")


def self_test() -> None:
    global LOCK_SHA256, PYPROJECT_SHA256
    # Exercise semantic tamper checks against the committed production
    # projection, updating only the outer file digest in each candidate.
    production_root = Path(__file__).resolve().parent
    production_manifest = strict_json_loads((production_root / "license_gate_manifest.json").read_text(encoding="utf-8"))
    production_lock = tomllib.loads((production_root / "uv.lock").read_text(encoding="utf-8"))
    production_rows = package_rows(production_lock)
    production_reviews = review_rows(production_rows, production_manifest)
    production_components = component_rows(production_manifest)
    production_compact = strict_json_loads((production_root / "dependency_audit_evidence.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="qwen3-tts-compact-tamper-") as directory:
        compact_path = Path(directory) / "dependency_audit_evidence.json"
        compact_base = json.loads(json.dumps(production_compact))
        reference_base = json.loads(json.dumps(production_manifest["dependency_audit_evidence"]))
        for label, mutate in (
            ("declared license", lambda value: value["license_facts"]["packages"][0].update(declared_license="GPL-3.0-only")),
            ("native count", lambda value: value["license_facts"]["packages"][0].update(native_file_count=99)),
            ("model repo", lambda value: value["model_facts"]["metadata_records"][0].update(repo="tampered/model")),
            ("model revision", lambda value: value["model_facts"]["metadata_records"][0].update(revision="0" * 40)),
            ("model license", lambda value: value["model_facts"]["metadata_records"][0].update(license="mit")),
            ("model tree hash", lambda value: value["model_facts"]["metadata_records"][0].update(tree_files_sha256="0" * 64)),
            ("source license", lambda value: value["model_facts"]["license_file_records"][-1].update(sha256="0" * 64)),
            ("aggregate count", lambda value: value["native_facts"].update(bundled_file_count=0)),
        ):
            candidate = json.loads(json.dumps(compact_base))
            mutate(candidate)
            compact_path.write_text(json.dumps(candidate), encoding="utf-8")
            candidate_reference = json.loads(json.dumps(reference_base))
            candidate_reference["sha256"] = digest_bytes(compact_path.read_bytes())
            try:
                validate_dependency_audit_evidence(compact_path, candidate_reference, production_manifest, production_reviews, production_components)
            except SystemExit as error:
                if error.code != 2:
                    raise SystemExit(f"qwen3-tts gate self-test production {label}: exit {error.code}") from error
            else:
                raise SystemExit(f"qwen3-tts gate self-test production {label}: tamper accepted")
    with tempfile.TemporaryDirectory(prefix="qwen3-tts-license-gate-") as directory:
        root = Path(directory)
        lock = root / "uv.lock"
        lock.write_text(
            'version = 1\n\n[[package]]\nname = "apache-package"\nversion = "2.0"\nsource = { registry = "https://pypi.org/simple" }\n\n[[package]]\nname = "mit-package"\nversion = "1.0"\nsource = { registry = "https://pypi.org/simple" }\n',
            encoding="utf-8",
        )
        rows = package_rows(tomllib.loads(lock.read_text(encoding="utf-8")))
        review_values = {
            "apache-package": ("Apache-2.0", "no native or bundled code"),
            "mit-package": ("MIT", "no native or bundled code"),
        }
        explicit_reviews = [
            {
                "name": row["name"],
                "version": row["version"],
                "source": row["source"],
                "status": "REVIEWED",
                "license": review_values[row["name"]][0],
                "native_bundled": review_values[row["name"]][1],
                "payload_sha256": "0" * 64,
                "approval_schema": "v1",
                "approval_signer": "a" * 40,
                "approval_digest": "b" * 64,
            }
            for row in rows
        ]
        manifest = {
            "gate_version": GATE_VERSION,
            "lock_sha256": digest_bytes(lock.read_bytes()),
            "package_rows_sha256": canonical_digest(rows),
            "review_rows": explicit_reviews,
            "component_rows": [
                {
                    "component": identity["component"],
                    "identity": identity,
                    "status": "REVIEWED",
                    "license": "Apache-2.0" if identity["component"] == "official_source" else "MIT",
                    "native_bundled": "no native or bundled code",
                    "payload_sha256": "0" * 64,
                    "approval_schema": "v1",
                    "approval_signer": "a" * 40,
                    "approval_digest": "b" * 64,
                }
                for identity in fixed_component_identities()
            ],
            "identities": json.loads(json.dumps(FIXED_IDENTITIES)),
            "model_license_metadata": json.loads(json.dumps(MODEL_LICENSE_METADATA_POLICY)),
            "forbidden_packages": list(FORBIDDEN_PACKAGES),
            "publication": "NO_UPLOAD",
            "dependency_audit_evidence": {
                "schema": COMPACT_SCHEMA,
                "path": "dependency_audit_evidence.json",
                "sha256": "0" * 64,
                "full_audit_sha256": "e" * 64,
                "status": "PENDING_OWNER_APPROVAL",
            },
        }
        reviews = review_rows(rows, manifest)
        manifest["review_rows_sha256"] = canonical_digest(reviews)
        components = component_rows(manifest)
        manifest["component_rows_sha256"] = canonical_digest(components)
        project = root / "pyproject.toml"
        project.write_text('[project]\nname = "qwen3-tts-self-test"\n', encoding="utf-8")
        LOCK_SHA256 = digest_bytes(lock.read_bytes())
        PYPROJECT_SHA256 = digest_bytes(project.read_bytes())
        manifest["pyproject_sha256"] = digest_bytes(project.read_bytes())
        manifest_path = root / "manifest.json"
        compact_path = root / "dependency_audit_evidence.json"
        compact = {
            "schema": COMPACT_SCHEMA,
            "status": "PENDING_OWNER_APPROVAL",
            "full_audit_status": "BLOCKED",
            "full_audit_sha256": "e" * 64,
            "inputs": {
                "pyproject_sha256": manifest["pyproject_sha256"],
                "uv_lock_sha256": manifest["lock_sha256"],
                "package_rows_sha256": manifest["package_rows_sha256"],
                "review_rows_sha256": manifest["review_rows_sha256"],
                "component_rows_sha256": manifest["component_rows_sha256"],
                "approval_scope_sha256": "0" * 64,
            },
            "repository": {"head": "a" * 40, "clean": True, "audit_script_sha256": "b" * 64},
            "environment": {"python": "3.12", "platform": "linux", "machine": "x86_64", "model_code_imported": False, "cargo_invoked": False, "upload_performed": False},
            "closure": {"active_rows": 2, "inactive_rows": 0, "expected_count": 2, "installed_count": 2, "missing": [], "unexpected": [], "exact": True, "expected_sha256": "c" * 64, "installed_sha256": "d" * 64},
            "license_facts": {"package_count": 2, "declared_license_missing": 0, "publisher_file_count": 0, "unsafe_publisher_file_count": 0, "packages": [{"name": row["name"], "version": row["version"], "source": row["source"]} for row in rows], "classification": "self-test"},
            "native_facts": {"bundled_file_count": 0, "unsafe_native_file_count": 0, "packages_with_native": [], "classification": "self-test"},
            "model_facts": {"license_file_records": [], "metadata_records": [{"component": component, "revision": "a" * 40} for component in MODEL_LICENSE_METADATA_POLICY["components"]], "metadata_fallback_count": 5, "classification": "self-test"},
            "inactive_facts": [],
            "component_facts": [],
            "approval": {"status": "PENDING_OWNER_APPROVAL", "signer": None, "digest": None},
        }
        manifest["approval_scope_sha256"] = canonical_digest(approval_scope(manifest))
        compact["inputs"]["approval_scope_sha256"] = manifest["approval_scope_sha256"]
        compact_path.write_text(json.dumps(compact), encoding="utf-8")
        manifest["dependency_audit_evidence"]["sha256"] = digest_bytes(compact_path.read_bytes())
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        validate_dependency_audit_evidence(
            compact_path,
            manifest["dependency_audit_evidence"],
            manifest,
            reviews,
            components,
        )
        compact_base = json.loads(json.dumps(compact))
        compact_reference = json.loads(json.dumps(manifest["dependency_audit_evidence"]))
        for label, mutate in (
            ("compact closure tamper", lambda value: value["closure"].update(exact=False)),
            ("compact model metadata tamper", lambda value: value["model_facts"]["metadata_records"][0].update(component="tampered")),
            ("compact approval tamper", lambda value: value["approval"].update(signer="a" * 40)),
        ):
            candidate = json.loads(json.dumps(compact_base))
            mutate(candidate)
            compact_path.write_text(json.dumps(candidate), encoding="utf-8")
            candidate_reference = json.loads(json.dumps(compact_reference))
            candidate_reference["sha256"] = digest_bytes(compact_path.read_bytes())
            try:
                validate_dependency_audit_evidence(compact_path, candidate_reference, manifest, reviews, components)
            except SystemExit as error:
                if error.code != 2:
                    raise SystemExit(f"qwen3-tts gate self-test {label}: exit {error.code}") from error
            else:
                raise SystemExit(f"qwen3-tts gate self-test {label}: tamper accepted")
        compact_path.write_text(json.dumps(compact_base), encoding="utf-8")
        evidence_path = root / "evidence.json"
        key_rows = [
            {
                "name": row["name"],
                "version": row["version"],
                "source": row["source"],
                "status": "REVIEWED",
                "license": review_values[row["name"]][0],
                "native_bundled": review_values[row["name"]][1],
                "payload_sha256": "0" * 64,
                "approval": {"schema": "v1", "decision": "APPROVED", "signer": "a" * 40, "signature_sha256": "b" * 64},
            }
            for row in reviews
        ]
        evidence = {"review_rows_sha256": manifest["review_rows_sha256"], "rows": key_rows}
        evidence["component_rows_sha256"] = manifest["component_rows_sha256"]
        evidence["approval_scope_sha256"] = manifest["approval_scope_sha256"]
        evidence["components"] = [
            {
                "component": component["component"],
                "status": "REVIEWED",
                "license": component["license"],
                "native_bundled": component["native_bundled"],
                "payload_sha256": component["payload_sha256"],
                "approval": {"schema": "v1", "decision": "APPROVED", "signer": "a" * 40, "signature_sha256": "b" * 64},
            }
            for component in components
        ]
        evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        revisions = {slug: FIXED_IDENTITIES["variants"][slug]["revision"] for slug in VARIANTS}
        run(
            lock,
            manifest_path,
            project,
            evidence_path,
            FIXED_IDENTITIES["official_source_revision"],
            FIXED_IDENTITIES["decoder_revision"],
            FIXED_IDENTITIES["decoder_checkpoint_sha256"],
            revisions,
        )
        duplicate_manifest = root / "duplicate-manifest.json"
        duplicate_manifest.write_text('{"gate_version":1,"gate_version":1}', encoding="utf-8")
        try:
            run(lock, duplicate_manifest, project, evidence_path, FIXED_IDENTITIES["official_source_revision"], FIXED_IDENTITIES["decoder_revision"], FIXED_IDENTITIES["decoder_checkpoint_sha256"], revisions)
        except SystemExit as error:
            if error.code != 2:
                raise SystemExit(f"qwen3-tts gate self-test duplicate manifest returned {error.code}") from error
        else:
            raise SystemExit("qwen3-tts gate self-test duplicate manifest was accepted")
        duplicate_evidence = root / "duplicate-evidence.json"
        duplicate_evidence.write_text('{"review_rows_sha256":"x","review_rows_sha256":"x"}', encoding="utf-8")
        try:
            run(lock, manifest_path, project, duplicate_evidence, FIXED_IDENTITIES["official_source_revision"], FIXED_IDENTITIES["decoder_revision"], FIXED_IDENTITIES["decoder_checkpoint_sha256"], revisions)
        except SystemExit as error:
            if error.code != 2:
                raise SystemExit(f"qwen3-tts gate self-test duplicate evidence returned {error.code}") from error
        else:
            raise SystemExit("qwen3-tts gate self-test duplicate evidence was accepted")
        manifest_base = json.loads(json.dumps(manifest))
        evidence_base = json.loads(json.dumps(evidence))

        def blocked(label: str, mutate_manifest: Any = None, mutate_evidence: Any = None, tamper_lock: bool = False, lock_contents: str | None = None, tamper_project: bool = False, refresh_review_digest: bool = False, refresh_component_digest: bool = False) -> None:
            candidate_manifest = json.loads(json.dumps(manifest_base))
            candidate_evidence = json.loads(json.dumps(evidence_base))
            original_lock = lock.read_bytes()
            original_project = project.read_bytes()
            if mutate_manifest is not None:
                mutate_manifest(candidate_manifest)
            if mutate_evidence is not None:
                mutate_evidence(candidate_evidence)
            try:
                if refresh_review_digest:
                    candidate_manifest["review_rows_sha256"] = canonical_digest(
                        review_rows(rows, candidate_manifest)
                    )
                if refresh_component_digest:
                    candidate_manifest["component_rows_sha256"] = canonical_digest(
                        component_rows(candidate_manifest)
                    )
                manifest_path.write_text(json.dumps(candidate_manifest), encoding="utf-8")
                evidence_path.write_text(json.dumps(candidate_evidence), encoding="utf-8")
                if tamper_lock:
                    lock.write_text("version = 1\n# tampered\n", encoding="utf-8")
                if lock_contents is not None:
                    lock.write_text(lock_contents, encoding="utf-8")
                if tamper_project:
                    project.write_text('[project]\nname = "tampered"\n', encoding="utf-8")
                run(lock, manifest_path, project, evidence_path, FIXED_IDENTITIES["official_source_revision"], FIXED_IDENTITIES["decoder_revision"], FIXED_IDENTITIES["decoder_checkpoint_sha256"], revisions)
            except SystemExit as error:
                if error.code != 2:
                    raise SystemExit(f"qwen3-tts gate self-test {label}: exit {error.code}") from error
            else:
                raise SystemExit(f"qwen3-tts gate self-test {label}: tamper accepted")
            finally:
                lock.write_bytes(original_lock)
                project.write_bytes(original_project)

        blocked("source revision tamper", lambda value: value["identities"].update(official_source_revision="0" * 40))
        blocked("decoder revision tamper", lambda value: value["identities"].update(decoder_revision="0" * 40))
        blocked("decoder checkpoint tamper", lambda value: value["identities"].update(decoder_checkpoint_sha256="0" * 64))
        for slug in VARIANTS:
            blocked(f"{slug} revision tamper", lambda value, slug=slug: value["identities"]["variants"][slug].update(revision="0" * 40))
            blocked(f"{slug} config bytes tamper", lambda value, slug=slug: value["identities"]["variants"][slug].update(config_bytes=1))
            blocked(f"{slug} config hash tamper", lambda value, slug=slug: value["identities"]["variants"][slug].update(config_sha256="0" * 64))
        for asset in FIXED_IDENTITIES["common_assets"]:
            blocked(f"{asset} bytes tamper", lambda value, asset=asset: value["identities"]["common_assets"][asset].update(bytes=1))
            blocked(f"{asset} hash tamper", lambda value, asset=asset: value["identities"]["common_assets"][asset].update(sha256="0" * 64))
        blocked(
            "review row source tamper",
            mutate_manifest=lambda value: value["review_rows"][0].update(source={"registry": "https://tampered.invalid/simple"}),
            refresh_review_digest=True,
        )
        blocked(
            "component identity tamper",
            mutate_manifest=lambda value: value["component_rows"][0]["identity"].update(revision="0" * 40),
            refresh_component_digest=True,
        )
        blocked(
            "component payload tamper",
            mutate_manifest=lambda value: value["component_rows"][0].update(payload_sha256="1" * 64),
            refresh_component_digest=True,
        )
        blocked(
            "component approval tamper",
            mutate_evidence=lambda value: value["components"][0]["approval"].update(signature_sha256="1" * 64),
        )
        blocked(
            "approval scope tamper",
            mutate_manifest=lambda value: value.update(approval_scope_sha256="0" * 64),
        )
        blocked(
            "approval evidence scope tamper",
            mutate_evidence=lambda value: value.update(approval_scope_sha256="0" * 64),
        )
        blocked("lock bytes tamper", tamper_lock=True)
        blocked("manifest lock digest tamper", mutate_manifest=lambda value: value.update(lock_sha256="0" * 64))
        blocked("empty package table", lock_contents="version = 1\npackage = []\n")
        blocked("malformed package row", lock_contents='version = 1\n[[package]]\nname = 1\nversion = "1"\n')
        blocked("pyproject bytes tamper", tamper_project=True)
        blocked("manifest pyproject digest tamper", mutate_manifest=lambda value: value.update(pyproject_sha256="0" * 64))
        blocked("forbidden package policy tamper", mutate_manifest=lambda value: value.update(forbidden_packages=["gradio"]))
        blocked("license tamper", mutate_evidence=lambda value: value["rows"][0].update(license="GPL-3.0-only"))
        blocked("native closure tamper", mutate_evidence=lambda value: value["rows"][0].update(native_bundled="changed"))
        blocked("approval tamper", mutate_evidence=lambda value: value["rows"][0]["approval"].update(signature_sha256="1" * 64))
        blocked(
            "unresolved license row",
            mutate_manifest=lambda value: value["review_rows"][0].update(license="UNRESOLVED"),
            refresh_review_digest=True,
        )
        blocked(
            "unresolved native row",
            mutate_manifest=lambda value: value["review_rows"][0].update(native_bundled="UNRESOLVED"),
            refresh_review_digest=True,
        )
        blocked(
            "unreviewed row status",
            mutate_manifest=lambda value: value["review_rows"][0].update(status="UNRESOLVED"),
            refresh_review_digest=True,
        )
        for placeholder in ("OWNER_REVIEW_REQUIRED", "pending_review", " REVIEW_REQUIRED ", " "):
            blocked(
                f"package license placeholder {placeholder!r}",
                mutate_manifest=lambda value, placeholder=placeholder: value["review_rows"][0].update(license=placeholder),
                refresh_review_digest=True,
            )
            blocked(
                f"package native placeholder {placeholder!r}",
                mutate_manifest=lambda value, placeholder=placeholder: value["review_rows"][0].update(native_bundled=placeholder),
                refresh_review_digest=True,
            )
            for field in ("approval_signer", "approval_digest", "payload_sha256"):
                blocked(
                    f"package {field} placeholder {placeholder!r}",
                    mutate_manifest=lambda value, field=field, placeholder=placeholder: value["review_rows"][0].update(**{field: placeholder}),
                    refresh_review_digest=True,
                )
            blocked(
                f"component license placeholder {placeholder!r}",
                mutate_manifest=lambda value, placeholder=placeholder: value["component_rows"][0].update(license=placeholder),
                refresh_component_digest=True,
            )
            blocked(
                f"component native placeholder {placeholder!r}",
                mutate_manifest=lambda value, placeholder=placeholder: value["component_rows"][0].update(native_bundled=placeholder),
                refresh_component_digest=True,
            )
            for field in ("approval_signer", "approval_digest", "payload_sha256"):
                blocked(
                    f"component {field} placeholder {placeholder!r}",
                    mutate_manifest=lambda value, field=field, placeholder=placeholder: value["component_rows"][0].update(**{field: placeholder}),
                    refresh_component_digest=True,
                )
        for field in ("approval_signer", "approval_digest", "payload_sha256"):
            blocked(
                f"package {field} null placeholder",
                mutate_manifest=lambda value, field=field: value["review_rows"][0].update(**{field: None}),
                refresh_review_digest=True,
            )
            blocked(
                f"component {field} null placeholder",
                mutate_manifest=lambda value, field=field: value["component_rows"][0].update(**{field: None}),
                refresh_component_digest=True,
            )
    print("license_gate.py self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--license-evidence", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument("--decoder-revision")
    parser.add_argument("--decoder-checkpoint-sha256")
    parser.add_argument("--variant-revision", action="append", default=[])
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.lock, args.manifest, args.project, args.license_evidence, args.source_revision, args.decoder_revision, args.decoder_checkpoint_sha256)) or args.variant_revision:
            parser.error("--self-test accepts no other arguments")
        self_test()
        return 0
    if args.lock is None or args.manifest is None or args.project is None or args.source_revision is None or args.decoder_revision is None or args.decoder_checkpoint_sha256 is None:
        parser.error("--project, --lock, --manifest, --source-revision, --decoder-revision, --decoder-checkpoint-sha256, and four --variant-revision values are required")
    run(args.lock, args.manifest, args.project, args.license_evidence, args.source_revision, args.decoder_revision, args.decoder_checkpoint_sha256, parse_variant_revisions(args.variant_revision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
