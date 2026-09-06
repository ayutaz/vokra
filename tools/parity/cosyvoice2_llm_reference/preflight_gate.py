#!/usr/bin/env -S uv run --no-project --python 3.12 python
"""Model-free, fail-closed gate for the CosyVoice2 LLM reference project."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import tomllib


PROJECT_NAME = "cosyvoice2-llm-reference"
PYTHON_REQUIREMENT = "==3.12.*"
ENVIRONMENT = "sys_platform == 'linux' and platform_machine == 'x86_64'"
TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
PYPI_INDEX = "https://pypi.org/simple"
DIRECT_PINS = {"numpy": "2.3.5", "torch": "2.7.1", "transformers": "5.10.4"}
FORBIDDEN = ("cosyvoice", "librosa", "soxr", "soundfile", "onnx", "torchaudio", "triton", "nvidia")
LICENSE_FORMAT = "vokra-cosyvoice2-llm-license-gate-v1"
MODEL_REPOSITORY = "FunAudioLLM/CosyVoice2-0.5B"
MODEL_REVISION = "eec1ae6c79877dbd9379285cf8789c9e0879293d"
MODEL_PATH = "llm.pt"
MODEL_BYTES = 2_023_316_821
MODEL_SHA256 = "b144ef55b51ce8cfb79a73c90dbba0bdaba4e451c0ebcfab20f769264f84a608"
QWEN_CONFIG_PATH = "CosyVoice-BlankEN/config.json"
QWEN_CONFIG_BYTES = 659
QWEN_CONFIG_SHA256 = "168aa1bd401abc3bc262ba15ba4e499627a8b4e006e9d050b47c22de20660185"
QWEN_CONFIG_BLOB_SHA1 = "463b055262b6c66c4629a74a4b300bfe2ed31d3c"
SOURCE_REPOSITORY = "https://github.com/FunAudioLLM/CosyVoice.git"
SOURCE_REVISION = "8555549e882236e6541748b1042d95693caa82ba"
LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
LICENSE_BLOB_SHA1 = "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64"
LICENSE_BYTES = 11_357
TENSOR_COUNT = 295
TENSOR_MANIFEST_SHA256 = "07cf10ae088c27a7c88e1c08fb231d00b01bba0c13f312a74d2fd4b35403bda2"
SOURCE_ROLES = {
    "cosyvoice/cli/cosyvoice.py": {"blob_sha1": "cc443bed44c651a47492fc7e2142e3a88fb47627", "sha256": "8e44f0f0144378561a00ebc065fdb15a843bc4650e68683bebb6624827731859", "marker": "CosyVoice2"},
    "cosyvoice/llm/llm.py": {"blob_sha1": "59ebd48fde1f1b69240391fdac6e2afc1035e123", "sha256": "6439d57fcf78bcdcad6d31812f3f4b02bd34f513333711ee317d71d1fd14d2de", "marker": "Qwen2LM"},
    "cosyvoice/tokenizer/tokenizer.py": {"blob_sha1": "43fb39a2b543cc7ba4ec95fca9327596c34dcff0", "sha256": "94340fc7cdf270c69a3aeb63290c5241044e20714e01fea736f361f9e5a56df2", "marker": "Qwen"},
}


class GateError(ValueError):
    """The dedicated reference project is not authorized to execute."""


def fail(message: str) -> None:
    raise GateError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_digest(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def file_identity(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        fail(f"missing or symlinked scope file: {path}")
    try:
        content = path.read_bytes()
    except OSError as error:
        fail(f"scope file unreadable: {path}: {error}")
    return {"bytes": len(content), "sha256": sha256_bytes(content)}


def read_toml(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        fail(f"missing or symlinked TOML: {path}")
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        fail(f"invalid TOML {path}: {error}")


def validate_project(document: dict[str, Any]) -> None:
    project = document.get("project")
    if not isinstance(project, dict) or project.get("name") != PROJECT_NAME:
        fail("project name mismatch")
    if project.get("requires-python") != PYTHON_REQUIREMENT:
        fail("Python requirement is not exact 3.12")
    dependencies = project.get("dependencies")
    expected = {f"{name}=={version}" for name, version in DIRECT_PINS.items()}
    if dependencies != sorted(expected):
        fail("direct dependency pins drifted")
    tool_uv = document.get("tool", {}).get("uv")
    if not isinstance(tool_uv, dict) or tool_uv.get("package") is not False:
        fail("reference project must remain non-package")
    if tool_uv.get("environments") != [ENVIRONMENT]:
        fail("reference project is not Linux x86_64-only")
    if document.get("tool", {}).get("uv", {}).get("sources") != {"torch": {"index": "pytorch-cpu"}}:
        fail("torch CPU source is not explicit")
    if document.get("tool", {}).get("uv", {}).get("index") != [{"name": "pytorch-cpu", "url": TORCH_INDEX, "explicit": True}]:
        fail("PyTorch CPU index is not canonical")


def validate_lock(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if document.get("version") != 1 or document.get("revision") != 3:
        fail("uv.lock schema/version drifted")
    if document.get("requires-python") != PYTHON_REQUIREMENT:
        fail("uv.lock Python requirement drifted")
    rows = document.get("package")
    if not isinstance(rows, list) or not rows:
        fail("dedicated uv.lock is missing package rows")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not isinstance(row.get("version"), str):
            fail("malformed uv.lock package row")
        name = row["name"]
        if name in result:
            fail(f"duplicate uv.lock package row: {name}")
        source = row.get("source", {})
        if not isinstance(source, dict) or len(source) != 1 or set(source) not in ({"registry"}, {"virtual"}):
            fail(f"malformed source for {name}")
        if "registry" in source and source["registry"] not in {PYPI_INDEX, TORCH_INDEX}:
            fail(f"unapproved registry for {name}")
        # The non-package project row is intentionally named
        # ``cosyvoice2-llm-reference``.  Its identity is authenticated by the
        # exact virtual source and project validation below; forbidden-closure
        # scanning applies only to registry rows and their artifacts.
        if "registry" in source and any(token in json.dumps(row, sort_keys=True).lower() for token in FORBIDDEN):
            fail(f"forbidden closure marker in {name}")
        for key in ("sdist", "wheels"):
            artifacts = row.get(key, [])
            if key == "sdist" and isinstance(artifacts, dict):
                artifacts = [artifacts]
            if not isinstance(artifacts, list):
                fail(f"malformed {key} artifacts for {name}")
            for artifact in artifacts:
                if not isinstance(artifact, dict) or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(artifact.get("hash", ""))):
                    fail(f"artifact hash missing for {name}")
                url = artifact.get("url")
                if not isinstance(url, str) or not (url.startswith("https://files.pythonhosted.org/") or url.startswith("https://download-r2.pytorch.org/")):
                    fail(f"artifact URL host is not approved for {name}")
                if "size" not in artifact:
                    allowed_torch_size_omission = (
                        name == "torch"
                        and key == "wheels"
                        and source == {"registry": TORCH_INDEX}
                        and url.startswith("https://download-r2.pytorch.org/")
                    )
                    if not allowed_torch_size_omission:
                        fail(f"artifact size missing for {name}")
                elif type(artifact["size"]) is not int or artifact["size"] <= 0:
                    fail(f"artifact size missing for {name}")
                if not isinstance(artifact.get("upload-time"), str) or not artifact["upload-time"].strip():
                    fail(f"artifact upload-time is missing for {name}")
        result[name] = row
    virtual_rows = [row for row in result.values() if "virtual" in row["source"]]
    if len(virtual_rows) != 1:
        fail("uv.lock must contain exactly one virtual project row")
    virtual = virtual_rows[0]
    if (
        virtual["name"] != PROJECT_NAME
        or virtual["version"] != "0.1.0"
        or virtual["source"] != {"virtual": "."}
        or "sdist" in virtual
        or "wheels" in virtual
    ):
        fail("uv.lock virtual project identity or artifact schema drifted")
    for name, version in DIRECT_PINS.items():
        row = result.get(name)
        if row is None or row.get("version") not in {version, f"{version}+cpu"}:
            fail(f"locked {name} pin drifted")
        if name == "torch" and row.get("source", {}).get("registry") != TORCH_INDEX:
            fail("torch is not locked to CPU index")
    return result


def validate_license_manifest(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        fail("license gate manifest is missing or symlinked")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(f"license manifest is invalid: {error}")
    required = {"format", "status", "owner_signoff", "publication", "blockers", "package_review", "native_payload_review", "weight_review", "source_review", "approval"}
    if not isinstance(data, dict) or set(data) != required or data.get("format") != LICENSE_FORMAT:
        fail("license manifest schema drifted")
    if data.get("publication") != "NO_UPLOAD":
        fail("publication must remain NO_UPLOAD")
    if (data.get("status"), data.get("owner_signoff")) not in {
        ("PENDING_REVIEW", "OWNER_SIGNOFF_REQUIRED"),
        ("APPROVED", "OWNER_SIGNED_OFF"),
    }:
        fail("license approval state is unknown")
    if not isinstance(data.get("blockers"), list):
        fail("license blockers must be an explicit list")
    if data["status"] == "PENDING_REVIEW" and not data["blockers"]:
        fail("pending license gate must retain explicit blockers")
    if data["status"] == "APPROVED" and data["blockers"]:
        fail("approved license gate must have no blockers")
    rows = data.get("package_review")
    if not isinstance(rows, list):
        fail("package review is malformed")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"name", "version", "status"} or row["status"] not in {"PENDING_PRIMARY_SOURCE_REVIEW", "APPROVED"}:
            fail("package license review row is malformed")
    for field in ("native_payload_review", "weight_review", "source_review"):
        if data[field] not in {"PENDING_PRIMARY_SOURCE_REVIEW", "APPROVED"}:
            fail(f"{field} status is unknown")
    approval = data.get("approval")
    if not isinstance(approval, dict) or set(approval) != {"signer", "scope_sha256"}:
        fail("approval schema drifted")
    if data["status"] == "APPROVED":
        if not isinstance(approval["signer"], str) or not approval["signer"].strip() or not re.fullmatch(r"[0-9a-f]{64}", str(approval["scope_sha256"])):
            fail("approved license gate lacks complete signoff")
    elif approval["signer"] is not None:
        fail("pending license gate signer must be null")
    return data


def package_review_rows(manifest: dict[str, Any]) -> list[dict[str, str]]:
    reviewed: list[dict[str, str]] = []
    names: set[str] = set()
    for row in manifest["package_review"]:
        if not isinstance(row["name"], str) or not isinstance(row["version"], str):
            fail("package/native license evidence has a malformed package identity")
        if row["name"] in names:
            fail("package/native license evidence contains a duplicate package review")
        names.add(row["name"])
        reviewed.append({"name": row["name"], "version": row["version"], "status": row["status"]})
    return sorted(reviewed, key=lambda row: (row["name"], row["version"], row["status"]))


def approval_scope(project: Path, lock: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": "vokra-cosyvoice2-llm-approval-scope-v1",
        "status": manifest["status"],
        "owner_signoff": manifest["owner_signoff"],
        "blockers": manifest["blockers"],
        "approval": {"signer": manifest["approval"]["signer"]},
        "project": {"name": PROJECT_NAME, **file_identity(project)},
        "uv_lock": file_identity(lock),
        "package_review": package_review_rows(manifest),
        "review_status": {
            "native_payload": manifest["native_payload_review"],
            "weight": manifest["weight_review"],
            "source": manifest["source_review"],
        },
        "publication": "NO_UPLOAD",
        "identities": {
            "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION, "path": MODEL_PATH, "bytes": MODEL_BYTES, "sha256": MODEL_SHA256},
            "qwen_config": {"path": QWEN_CONFIG_PATH, "bytes": QWEN_CONFIG_BYTES, "sha256": QWEN_CONFIG_SHA256, "git_blob_sha1": QWEN_CONFIG_BLOB_SHA1},
            "source": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION, "roles": SOURCE_ROLES},
            "license": {"path": "LICENSE", "spdx": "Apache-2.0", "bytes": LICENSE_BYTES, "sha256": LICENSE_SHA256, "git_blob_sha1": LICENSE_BLOB_SHA1},
            "tensor_manifest": {"count": TENSOR_COUNT, "sha256": TENSOR_MANIFEST_SHA256},
        },
    }


def gate(project: Path, lock: Path, license_manifest: Path) -> dict[str, Any]:
    project_doc = read_toml(project)
    validate_project(project_doc)
    rows = validate_lock(read_toml(lock))
    manifest = validate_license_manifest(license_manifest)
    registry_pairs = {(name, row["version"]) for name, row in rows.items() if "registry" in row["source"]}
    reviewed = package_review_rows(manifest)
    reviewed_pairs = [(row["name"], row["version"]) for row in reviewed]
    if set(reviewed_pairs) != registry_pairs or len(reviewed_pairs) != len(registry_pairs):
        fail("package/native license evidence does not cover the exact lock closure")
    scope = approval_scope(project, lock, manifest)
    expected_scope_sha256 = canonical_digest(scope)
    actual_scope_sha256 = manifest["approval"]["scope_sha256"]
    if actual_scope_sha256 is not None:
        if not isinstance(actual_scope_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", actual_scope_sha256):
            fail("approval scope digest is malformed")
        if actual_scope_sha256 != expected_scope_sha256:
            fail("approval scope digest does not match authenticated inputs")
    elif manifest["status"] == "APPROVED" or manifest["owner_signoff"] == "OWNER_SIGNED_OFF":
        fail("approved license gate lacks approval scope digest")
    if manifest["status"] == "APPROVED" and (
        any(row["status"] != "APPROVED" for row in manifest["package_review"])
        or manifest["native_payload_review"] != "APPROVED"
        or manifest["weight_review"] != "APPROVED"
        or manifest["source_review"] != "APPROVED"
    ):
        fail("approved gate contains unresolved package/native/source review")
    if manifest["status"] != "APPROVED" or manifest["owner_signoff"] != "OWNER_SIGNED_OFF":
        fail("execution blocked until package/native license evidence and owner signoff")
    return {"status": "PASS", "package_count": len(registry_pairs), "publication": "NO_UPLOAD"}


def self_test() -> None:
    here = Path(__file__).resolve().parent
    validate_project(read_toml(here / "pyproject.toml"))
    manifest = validate_license_manifest(here / "license_gate_manifest.json")
    assert manifest["status"] == "PENDING_REVIEW"
    for status, owner_signoff in (("APPROVED", "OWNER_SIGNOFF_REQUIRED"), ("PENDING_REVIEW", "OWNER_SIGNED_OFF")):
        mixed = copy.deepcopy(manifest)
        mixed["status"] = status
        mixed["owner_signoff"] = owner_signoff
        mixed_path = here / ".cosyvoice2-llm-preflight-mixed-self-test.json"
        try:
            mixed_path.write_text(json.dumps(mixed), encoding="utf-8")
            validate_license_manifest(mixed_path)
        except GateError:
            pass
        finally:
            mixed_path.unlink(missing_ok=True)
        if mixed_path.exists():
            raise AssertionError("mixed approval state self-test artifact remains")
    approved_with_blockers = copy.deepcopy(manifest)
    approved_with_blockers["status"] = "APPROVED"
    approved_with_blockers["owner_signoff"] = "OWNER_SIGNED_OFF"
    approved_with_blockers["approval"] = {"signer": "self-test-owner", "scope_sha256": "0" * 64}
    approved_with_blockers_path = here / ".cosyvoice2-llm-preflight-approved-blocker-self-test.json"
    try:
        approved_with_blockers_path.write_text(json.dumps(approved_with_blockers), encoding="utf-8")
        validate_license_manifest(approved_with_blockers_path)
    except GateError:
        pass
    finally:
        approved_with_blockers_path.unlink(missing_ok=True)
    pending_without_blockers = copy.deepcopy(manifest)
    pending_without_blockers["blockers"] = []
    pending_without_blockers_path = here / ".cosyvoice2-llm-preflight-pending-no-blocker-self-test.json"
    try:
        pending_without_blockers_path.write_text(json.dumps(pending_without_blockers), encoding="utf-8")
        validate_license_manifest(pending_without_blockers_path)
    except GateError:
        pass
    finally:
        pending_without_blockers_path.unlink(missing_ok=True)
    valid_lock = {
        "version": 1,
        "revision": 3,
        "requires-python": PYTHON_REQUIREMENT,
        "package": [
            {"name": PROJECT_NAME, "version": "0.1.0", "source": {"virtual": "."}},
            *[
                {
                    "name": name,
                    "version": f"{version}+cpu" if name == "torch" else version,
                    "source": {"registry": TORCH_INDEX if name == "torch" else PYPI_INDEX},
                }
                for name, version in DIRECT_PINS.items()
            ],
        ],
    }
    validate_lock(valid_lock)
    torch_artifact = {
        "url": "https://download-r2.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl",
        "hash": "sha256:" + "0" * 64,
        "upload-time": "2025-06-03T18:27:57Z",
    }
    allowed_missing_size = copy.deepcopy(valid_lock)
    next(row for row in allowed_missing_size["package"] if row["name"] == "torch")["wheels"] = [torch_artifact]
    validate_lock(allowed_missing_size)
    missing_pypi_size = copy.deepcopy(valid_lock)
    next(row for row in missing_pypi_size["package"] if row["name"] == "numpy")["wheels"] = [
        {"url": "https://files.pythonhosted.org/packages/numpy.whl", "hash": "sha256:" + "0" * 64, "upload-time": "2026-01-01T00:00:00Z"}
    ]
    try:
        validate_lock(missing_pypi_size)
    except GateError as error:
        assert "artifact size missing" in str(error)
    else:
        raise AssertionError("missing PyPI artifact size accepted")
    missing_non_torch_size = copy.deepcopy(valid_lock)
    missing_non_torch_size["package"].append(
        {
            "name": "not-torch",
            "version": "1.0",
            "source": {"registry": TORCH_INDEX},
            "wheels": [{"url": "https://download-r2.pytorch.org/whl/cpu/not-torch.whl", "hash": "sha256:" + "0" * 64, "upload-time": "2026-01-01T00:00:00Z"}],
        }
    )
    try:
        validate_lock(missing_non_torch_size)
    except GateError as error:
        assert "artifact size missing" in str(error)
    else:
        raise AssertionError("missing non-torch artifact size accepted")
    for invalid_size in (0, -1):
        invalid_torch_size = copy.deepcopy(allowed_missing_size)
        next(row for row in invalid_torch_size["package"] if row["name"] == "torch")["wheels"][0]["size"] = invalid_size
        try:
            validate_lock(invalid_torch_size)
        except GateError as error:
            assert "artifact size missing" in str(error)
        else:
            raise AssertionError("invalid torch artifact size accepted")
    forbidden_lock = copy.deepcopy(valid_lock)
    forbidden_lock["package"].append(
        {
            "name": "forbidden-package",
            "version": "1.0",
            "source": {"registry": PYPI_INDEX},
            "wheels": [{"url": "https://files.pythonhosted.org/packages/librosa.whl", "hash": "sha256:" + "0" * 64, "size": 1, "upload-time": "2026-01-01T00:00:00Z"}],
        }
    )
    try:
        validate_lock(forbidden_lock)
    except GateError as error:
        assert "forbidden" in str(error)
    else:
        raise AssertionError("forbidden registry closure marker accepted")
    for invalid_lock in (
        {**valid_lock, "package": valid_lock["package"][1:]},
        {**valid_lock, "package": [*valid_lock["package"], {"name": "other-project", "version": "0.1.0", "source": {"virtual": "."}}]},
        {**valid_lock, "package": [{**row, "name": "wrong-project"} if row.get("source") == {"virtual": "."} else row for row in valid_lock["package"]]},
        {**valid_lock, "package": [{**row, "source": {"registry": PYPI_INDEX}} if row.get("source") == {"virtual": "."} else row for row in valid_lock["package"]]},
    ):
        try:
            validate_lock(invalid_lock)
        except GateError:
            pass
        else:
            raise AssertionError("invalid virtual project lock accepted")
    with tempfile.TemporaryDirectory(prefix="cosyvoice2-llm-coverage-") as temp:
        temp_root = Path(temp)
        lock_path = temp_root / "uv.lock"
        lock_lines = ["version = 1", "revision = 3", f"requires-python = '{PYTHON_REQUIREMENT}'", "", ""]
        lock_lines.extend(
            [
                "[[package]]",
                f"name = '{PROJECT_NAME}'",
                "version = '0.1.0'",
                "source = { virtual = '.' }",
                "",
            ]
        )
        for name, version in DIRECT_PINS.items():
            registry = TORCH_INDEX if name == "torch" else PYPI_INDEX
            lock_version = f"{version}+cpu" if name == "torch" else version
            lock_lines.extend(["[[package]]", f"name = '{name}'", f"version = '{lock_version}'", f"source = {{ registry = '{registry}' }}", ""])
        lock_path.write_text("\n".join(lock_lines), encoding="utf-8")
        covered_manifest = copy.deepcopy(manifest)
        covered_rows = validate_lock(valid_lock)
        covered_manifest["package_review"] = [
            {"name": name, "version": covered_rows[name]["version"], "status": "PENDING_PRIMARY_SOURCE_REVIEW"}
            for name in DIRECT_PINS
        ]
        manifest_path = temp_root / "license.json"
        manifest_path.write_text(json.dumps(covered_manifest), encoding="utf-8")
        try:
            gate(here / "pyproject.toml", lock_path, manifest_path)
        except GateError as error:
            assert "execution blocked" in str(error)
        else:
            raise AssertionError("pending covered registry closure unexpectedly passed")
        wrong_version = copy.deepcopy(covered_manifest)
        wrong_version["package_review"][0]["version"] = "not-the-locked-version"
        wrong_version_path = temp_root / "wrong-version.json"
        wrong_version_path.write_text(json.dumps(wrong_version), encoding="utf-8")
        try:
            gate(here / "pyproject.toml", lock_path, wrong_version_path)
        except GateError as error:
            assert "exact lock closure" in str(error)
        else:
            raise AssertionError("wrong package review version accepted")
        duplicate_review = copy.deepcopy(covered_manifest)
        duplicate_review["package_review"].append(copy.deepcopy(duplicate_review["package_review"][0]))
        duplicate_path = temp_root / "duplicate-review.json"
        duplicate_path.write_text(json.dumps(duplicate_review), encoding="utf-8")
        try:
            gate(here / "pyproject.toml", lock_path, duplicate_path)
        except GateError as error:
            assert "duplicate package review" in str(error)
        else:
            raise AssertionError("duplicate package review accepted")
        approved_manifest = copy.deepcopy(covered_manifest)
        approved_manifest["status"] = "APPROVED"
        approved_manifest["owner_signoff"] = "OWNER_SIGNED_OFF"
        approved_manifest["native_payload_review"] = "APPROVED"
        approved_manifest["weight_review"] = "APPROVED"
        approved_manifest["source_review"] = "APPROVED"
        approved_manifest["blockers"] = []
        for row in approved_manifest["package_review"]:
            row["status"] = "APPROVED"
        approved_manifest["approval"] = {"signer": "self-test-owner", "scope_sha256": None}
        approved_manifest["approval"]["scope_sha256"] = canonical_digest(approval_scope(here / "pyproject.toml", lock_path, approved_manifest))
        approved_path = temp_root / "approved.json"
        approved_path.write_text(json.dumps(approved_manifest), encoding="utf-8")
        assert gate(here / "pyproject.toml", lock_path, approved_path)["status"] == "PASS"
        lock_bytes = lock_path.read_bytes()
        lock_path.write_bytes(lock_bytes + b"\n")
        try:
            gate(here / "pyproject.toml", lock_path, approved_path)
        except GateError as error:
            assert "scope digest" in str(error)
        else:
            raise AssertionError("lock scope drift accepted")
        lock_path.write_bytes(lock_bytes)
        for field in ("native_payload_review", "weight_review", "source_review"):
            drifted = copy.deepcopy(approved_manifest)
            drifted[field] = "PENDING_PRIMARY_SOURCE_REVIEW"
            drifted_path = temp_root / f"{field}-drift.json"
            drifted_path.write_text(json.dumps(drifted), encoding="utf-8")
            try:
                gate(here / "pyproject.toml", lock_path, drifted_path)
            except GateError as error:
                assert "scope digest" in str(error)
            else:
                raise AssertionError(f"{field} scope drift accepted")
        wrong_version_approved = copy.deepcopy(approved_manifest)
        wrong_version_approved["package_review"][0]["version"] = "wrong-version"
        wrong_version_path = temp_root / "approved-wrong-version.json"
        wrong_version_path.write_text(json.dumps(wrong_version_approved), encoding="utf-8")
        try:
            gate(here / "pyproject.toml", lock_path, wrong_version_path)
        except GateError as error:
            assert "exact lock closure" in str(error)
        else:
            raise AssertionError("approved package version drift accepted")
        signer_drift = copy.deepcopy(approved_manifest)
        signer_drift["approval"]["signer"] = "different-owner"
        signer_path = temp_root / "signer-drift.json"
        signer_path.write_text(json.dumps(signer_drift), encoding="utf-8")
        try:
            gate(here / "pyproject.toml", lock_path, signer_path)
        except GateError as error:
            assert "scope digest" in str(error)
        else:
            raise AssertionError("approval signer scope drift accepted")
        blocker_drift = copy.deepcopy(approved_manifest)
        blocker_drift["blockers"] = ["unexpected blocker"]
        blocker_path = temp_root / "blocker-drift.json"
        blocker_path.write_text(json.dumps(blocker_drift), encoding="utf-8")
        try:
            gate(here / "pyproject.toml", lock_path, blocker_path)
        except GateError as error:
            assert "no blockers" in str(error)
        else:
            raise AssertionError("approved blocker drift accepted")
    try:
        gate(here / "pyproject.toml", here / "uv.lock", here / "license_gate_manifest.json")
    except GateError as error:
        assert any(marker in str(error) for marker in ("blocked", "missing", "exact lock closure"))
    else:
        raise AssertionError("pending/missing-lock gate unexpectedly passed")
    with tempfile.TemporaryDirectory(prefix="cosyvoice2-llm-gate-") as temp:
        path = Path(temp) / "manifest.json"
        bad = copy.deepcopy(manifest)
        bad["publication"] = "UPLOAD"
        path.write_text(json.dumps(bad), encoding="utf-8")
        try:
            validate_license_manifest(path)
        except GateError:
            pass
        else:
            raise AssertionError("publication drift accepted")
    print("cosyvoice2_llm preflight self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--show-approval-scope", action="store_true")
    parser.add_argument("--project", type=Path, default=Path(__file__).with_name("pyproject.toml"))
    parser.add_argument("--lock", type=Path, default=Path(__file__).with_name("uv.lock"))
    parser.add_argument("--license-manifest", type=Path, default=Path(__file__).with_name("license_gate_manifest.json"))
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
        elif args.show_approval_scope:
            project = args.project
            lock = args.lock
            manifest = validate_license_manifest(args.license_manifest)
            validate_project(read_toml(project))
            validate_lock(read_toml(lock))
            scope = approval_scope(project, lock, manifest)
            print(json.dumps({"scope": scope, "scope_sha256": canonical_digest(scope)}, sort_keys=True))
        else:
            print(json.dumps(gate(args.project, args.lock, args.license_manifest), sort_keys=True))
    except (GateError, OSError, UnicodeError) as error:
        print(f"cosyvoice2_llm preflight: BLOCKED: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
