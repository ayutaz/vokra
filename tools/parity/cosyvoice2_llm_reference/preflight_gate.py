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
                if not isinstance(artifact.get("size"), int) or artifact["size"] <= 0:
                    fail(f"artifact size missing for {name}")
                url = artifact.get("url")
                if not isinstance(url, str) or not (url.startswith("https://files.pythonhosted.org/") or url.startswith("https://download-r2.pytorch.org/")):
                    fail(f"artifact URL host is not approved for {name}")
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
    if data.get("status") not in {"PENDING_REVIEW", "APPROVED"} or data.get("owner_signoff") not in {"OWNER_SIGNOFF_REQUIRED", "OWNER_SIGNED_OFF"}:
        fail("license approval state is unknown")
    if not isinstance(data.get("blockers"), list) or not data["blockers"]:
        fail("license blockers must be explicit")
    rows = data.get("package_review")
    if not isinstance(rows, list):
        fail("package review is malformed")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"name", "version", "status"} or row["status"] not in {"PENDING_PRIMARY_SOURCE_REVIEW", "APPROVED"}:
            fail("package license review row is malformed")
    approval = data.get("approval")
    if not isinstance(approval, dict) or set(approval) != {"signer", "scope_sha256"}:
        fail("approval schema drifted")
    if data["status"] == "APPROVED" or data["owner_signoff"] == "OWNER_SIGNED_OFF":
        if data["status"] != "APPROVED" or data["owner_signoff"] != "OWNER_SIGNED_OFF" or not isinstance(approval["signer"], str) or not approval["signer"].strip() or not re.fullmatch(r"[0-9a-f]{64}", str(approval["scope_sha256"])):
            fail("approved license gate lacks complete signoff")
    return data


def gate(project: Path, lock: Path, license_manifest: Path) -> dict[str, Any]:
    project_doc = read_toml(project)
    validate_project(project_doc)
    rows = validate_lock(read_toml(lock))
    manifest = validate_license_manifest(license_manifest)
    registry_rows = {name for name, row in rows.items() if "registry" in row["source"]}
    reviewed = {row["name"] for row in manifest["package_review"]}
    if reviewed != registry_rows:
        fail("package/native license evidence does not cover the exact lock closure")
    if manifest["status"] == "APPROVED" and (
        any(row["status"] != "APPROVED" for row in manifest["package_review"])
        or manifest["native_payload_review"] != "APPROVED"
        or manifest["weight_review"] != "APPROVED"
        or manifest["source_review"] != "APPROVED"
    ):
        fail("approved gate contains unresolved package/native/source review")
    if manifest["status"] != "APPROVED" or manifest["owner_signoff"] != "OWNER_SIGNED_OFF":
        fail("execution blocked until package/native license evidence and owner signoff")
    return {"status": "PASS", "package_count": len(registry_rows), "publication": "NO_UPLOAD"}


def self_test() -> None:
    here = Path(__file__).resolve().parent
    validate_project(read_toml(here / "pyproject.toml"))
    manifest = validate_license_manifest(here / "license_gate_manifest.json")
    assert manifest["status"] == "PENDING_REVIEW"
    valid_lock = {
        "version": 1,
        "revision": 3,
        "requires-python": PYTHON_REQUIREMENT,
        "package": [
            {"name": PROJECT_NAME, "version": "0.1.0", "source": {"virtual": "."}},
            *[
                {
                    "name": name,
                    "version": version,
                    "source": {"registry": TORCH_INDEX if name == "torch" else PYPI_INDEX},
                }
                for name, version in DIRECT_PINS.items()
            ],
        ],
    }
    validate_lock(valid_lock)
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
            lock_lines.extend(["[[package]]", f"name = '{name}'", f"version = '{version}'", f"source = {{ registry = '{registry}' }}", ""])
        lock_path.write_text("\n".join(lock_lines), encoding="utf-8")
        covered_manifest = copy.deepcopy(manifest)
        covered_manifest["package_review"] = [
            {"name": name, "version": version, "status": "PENDING_PRIMARY_SOURCE_REVIEW"}
            for name, version in DIRECT_PINS.items()
        ]
        manifest_path = temp_root / "license.json"
        manifest_path.write_text(json.dumps(covered_manifest), encoding="utf-8")
        try:
            gate(here / "pyproject.toml", lock_path, manifest_path)
        except GateError as error:
            assert "execution blocked" in str(error)
        else:
            raise AssertionError("pending covered registry closure unexpectedly passed")
    try:
        gate(here / "pyproject.toml", here / "uv.lock", here / "license_gate_manifest.json")
    except GateError as error:
        assert "blocked" in str(error) or "missing" in str(error)
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
    parser.add_argument("--project", type=Path, default=Path(__file__).with_name("pyproject.toml"))
    parser.add_argument("--lock", type=Path, default=Path(__file__).with_name("uv.lock"))
    parser.add_argument("--license-manifest", type=Path, default=Path(__file__).with_name("license_gate_manifest.json"))
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
        else:
            print(json.dumps(gate(args.project, args.lock, args.license_manifest), sort_keys=True))
    except (GateError, OSError, UnicodeError) as error:
        print(f"cosyvoice2_llm preflight: BLOCKED: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
