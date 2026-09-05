#!/usr/bin/env -S uv run --frozen --project tools/parity/cosyvoice2_hift_reference python
"""Fail-closed dependency preflight for the official CosyVoice2 HiFT oracle.

This gate is model-free.  It authenticates the dedicated Python project and
lockfile before a VAST worker is allowed to import the upstream checkout.  The
only intended runtime packages are the pinned CPU torch, numpy, and scipy
closure on Linux x86_64; source/model license review remains pending.
"""

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

EXPECTED_PROJECT_NAME = "cosyvoice2-hift-reference"
EXPECTED_REQUIRES_PYTHON = "==3.12.*"
EXPECTED_ENVIRONMENT = "sys_platform == 'linux' and platform_machine == 'x86_64'"
EXPECTED_LOCK_MARKER = "platform_machine == 'x86_64' and sys_platform == 'linux'"
EXPECTED_PINS = {"torch": "2.7.1", "numpy": "2.3.5", "scipy": "1.16.3"}
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
FORBIDDEN = ("triton", "nvidia", "librosa", "soxr", "soundfile")
MANIFEST_NAME = "license_gate_manifest.json"
AUTHORIZED_STATUS = "APPROVED"
AUTHORIZED_SIGNOFF = "OWNER_SIGNED_OFF"
APPROVAL_SCHEMA = "cosyvoice2-hift-approval-scope-v1"
APPROVED_COMPONENT_STATUSES = {"APPROVED", "REVIEWED"}
LOCK_KEYS = {"version", "revision", "requires-python", "resolution-markers", "supported-markers", "package"}
ARTIFACT_KEYS = {"url", "hash", "size", "upload-time"}
PYPI_REGISTRY = "https://pypi.org/simple"
REGISTRY_HOSTS = {PYPI_REGISTRY: "files.pythonhosted.org", PYTORCH_CPU_INDEX: "download-r2.pytorch.org"}
PACKAGE_BASE_KEYS = {"name", "version", "source", "dependencies", "resolution-markers", "sdist", "wheels", "metadata"}
EXPECTED_SOURCE = {
    "repository": "https://github.com/FunAudioLLM/CosyVoice.git",
    "revision": "8555549e882236e6541748b1042d95693caa82ba",
    "license_status": "PENDING_PRIMARY_SOURCE_REVIEW",
}
EXPECTED_MODEL = {
    "repository": "FunAudioLLM/CosyVoice2-0.5B",
    "revision": "eec1ae6c79877dbd9379285cf8789c9e0879293d",
    "file": "hift.pt",
    "bytes": 83_390_254,
    "sha256": "3386cc880324d4e98e05987b99107f49e40ed925b8ecc87c1f4939432d429879",
    "weight_license_status": "PENDING_PRIMARY_SOURCE_REVIEW",
}
EXPECTED_CONFIG = {
    "file": "cosyvoice2.yaml",
    "bytes": 7_330,
    "sha256": "0af2c0d010c477187c39f3e8fd5f1ae2e4e6f90ad03ba37c10ed6c6a87b05959",
    "git_blob_sha1": "bc19267bbfd373c9a760b7667a74349ddd487db1",
}
EXPECTED_CLOSURE_STATUS = "PENDING_PACKAGE_AND_NATIVE_PAYLOAD_REVIEW"
EXPECTED_EVIDENCE = {
    "tensor_count": 328,
    "tensor_dtype": "F32",
    "tensor_manifest_sha256": "cecbb2d68f91337f263db0f0333c75573516e7087b6e75d6ea647b3f86afec7c",
    "source_roles": {
        "cosyvoice/hifigan/generator.py": "326a1a70ae7707662939c20493b3a8e4b0906216",
        "cosyvoice/hifigan/f0_predictor.py": "5797c31aada757ac7ef65a70ff8ee21867a25df8",
        "cosyvoice/transformer/activation.py": "8cea54816385d3b6585ccc2417bc71630d578177",
        "cosyvoice/utils/common.py": "6f5a3dd8b7ae99601783c3a4ed91b3b64270fab3",
    },
}
EXPECTED_DECISION = "DO_NOT_DISTRIBUTE_UNTIL_OWNER_SIGNOFF"


class GateError(ValueError):
    """The dedicated reference project is not safe to execute."""


def fail(message: str) -> None:
    raise GateError(message)


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_toml(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        fail(f"missing or symlinked TOML file: {path}")
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        fail(f"unreadable TOML file {path}: {error}")


def package_rows(lock: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if set(lock) != LOCK_KEYS or lock.get("version") != 1 or lock.get("revision") != 3:
        fail("uv.lock top-level schema drifted")
    rows = lock.get("package")
    if not isinstance(rows, list) or not rows:
        fail("uv.lock has no package rows")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not isinstance(row.get("version"), str) or not row["version"]:
            fail("uv.lock contains a malformed package row")
        name = row["name"]
        if set(row) - PACKAGE_BASE_KEYS:
            fail(f"uv.lock package row has unknown fields: {name}")
        source = row.get("source")
        if not isinstance(source, dict) or len(source) != 1 or set(source) not in ({"registry"}, {"virtual"}):
            fail(f"uv.lock package source schema is malformed: {name}")
        if "virtual" in source:
            if source["virtual"] != "." or set(row) != {"name", "version", "source", "dependencies", "metadata"}:
                fail(f"uv.lock virtual package schema drifted: {name}")
        else:
            registry = source.get("registry")
            if registry not in REGISTRY_HOSTS:
                fail(f"uv.lock registry is not approved: {name}")
            allowed = (
                {"name", "version", "source", "sdist", "wheels"},
                {"name", "version", "source", "dependencies", "sdist", "wheels"},
                {"name", "version", "source", "dependencies", "wheels"},
            )
            if set(row) not in allowed:
                fail(f"uv.lock registry package schema drifted: {name}")
            for artifact_key in ("sdist", "wheels"):
                artifacts = row.get(artifact_key)
                if artifact_key == "sdist":
                    artifacts = [artifacts] if artifacts is not None else []
                if not isinstance(artifacts, list) or not artifacts:
                    if artifact_key == "wheels" and "wheels" in row:
                        fail(f"uv.lock wheels schema is malformed: {name}")
                    continue
                for artifact in artifacts:
                    if not isinstance(artifact, dict) or set(artifact) != ARTIFACT_KEYS:
                        fail(f"uv.lock artifact schema drifted: {name}")
                    url = artifact.get("url", "")
                    host = REGISTRY_HOSTS[registry]
                    if not isinstance(url, str) or not url.startswith(f"https://{host}/"):
                        fail(f"uv.lock artifact URL host is not approved: {name}")
                    if not isinstance(artifact.get("hash"), str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", artifact["hash"]):
                        fail(f"uv.lock artifact hash is not SHA-256: {name}")
                    if isinstance(artifact.get("size"), bool) or not isinstance(artifact.get("size"), int) or artifact["size"] <= 0:
                        fail(f"uv.lock artifact size is not positive: {name}")
                    if not isinstance(artifact.get("upload-time"), str) or not artifact["upload-time"].strip():
                        fail(f"uv.lock artifact upload-time is missing: {name}")
        dependencies = row.get("dependencies", [])
        if not isinstance(dependencies, list):
            fail(f"uv.lock dependency list is malformed: {name}")
        for dependency in dependencies:
            if not isinstance(dependency, dict) or set(dependency) != {"name", "marker"}:
                fail(f"uv.lock dependency row schema drifted: {name}")
            if not isinstance(dependency["name"], str) or not dependency["name"].strip() or not isinstance(dependency["marker"], str) or not dependency["marker"].strip():
                fail(f"uv.lock dependency row is malformed: {name}")
        if "metadata" in row:
            metadata = row["metadata"]
            if not isinstance(metadata, dict) or set(metadata) != {"requires-dist"} or not isinstance(metadata["requires-dist"], list):
                fail(f"uv.lock metadata schema drifted: {name}")
            for requirement in metadata["requires-dist"]:
                if not isinstance(requirement, dict) or set(requirement) not in ({"name", "specifier"}, {"name", "specifier", "index"}):
                    fail(f"uv.lock requires-dist schema drifted: {name}")
                if not isinstance(requirement["name"], str) or not requirement["name"].strip() or not isinstance(requirement["specifier"], str) or not requirement["specifier"].strip():
                    fail(f"uv.lock requires-dist entry is malformed: {name}")
                if "index" in requirement and requirement["index"] not in REGISTRY_HOSTS:
                    fail(f"uv.lock requires-dist index is not approved: {name}")
        if name in result:
            fail(f"uv.lock contains duplicate package row: {name}")
        result[name] = row
    return result


def wheel_urls(row: dict[str, Any]) -> list[str]:
    wheels = row.get("wheels", [])
    if not isinstance(wheels, list):
        fail(f"uv.lock wheels field is malformed for {row.get('name')}")
    urls = []
    for wheel in wheels:
        if not isinstance(wheel, dict) or not isinstance(wheel.get("url"), str):
            fail(f"uv.lock wheel row is malformed for {row.get('name')}")
        urls.append(wheel["url"])
    return urls


def validate_project(document: dict[str, Any]) -> None:
    project = document.get("project")
    if not isinstance(project, dict):
        fail("pyproject has no [project] table")
    if project.get("name") != EXPECTED_PROJECT_NAME:
        fail(f"project name drifted: {project.get('name')!r}")
    if project.get("requires-python") != EXPECTED_REQUIRES_PYTHON:
        fail("project is not pinned to Python 3.12")
    dependencies = project.get("dependencies")
    expected_dependencies = {
        f"{name}=={version}" for name, version in EXPECTED_PINS.items()
    }
    if not isinstance(dependencies, list) or set(dependencies) != expected_dependencies or len(dependencies) != len(expected_dependencies):
        fail(f"direct dependency pins drifted: {dependencies!r}")
    tool_uv = document.get("tool", {}).get("uv")
    if not isinstance(tool_uv, dict):
        fail("pyproject has no [tool.uv] table")
    if tool_uv.get("package") is not False:
        fail("reference project must remain non-package")
    if tool_uv.get("environments") != [EXPECTED_ENVIRONMENT]:
        fail("reference project is not Linux x86_64-only")
    if tool_uv.get("sources") != {"torch": {"index": "pytorch-cpu"}}:
        fail("torch is not explicitly sourced from the PyTorch CPU index")
    indexes = tool_uv.get("index")
    if indexes != [
        {
            "name": "pytorch-cpu",
            "url": PYTORCH_CPU_INDEX,
            "explicit": True,
        }
    ]:
        fail("PyTorch CPU index is not explicit and canonical")


def validate_lock(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if document.get("requires-python") != EXPECTED_REQUIRES_PYTHON:
        fail("uv.lock Python requirement drifted")
    markers = document.get("resolution-markers")
    if markers != [EXPECTED_LOCK_MARKER]:
        fail(f"uv.lock resolution markers drifted: {markers!r}")
    rows = package_rows(document)
    expected_names = {EXPECTED_PROJECT_NAME, *EXPECTED_PINS, "filelock", "fsspec", "jinja2", "markupsafe", "mpmath", "networkx", "setuptools", "sympy", "typing-extensions"}
    if set(rows) != expected_names:
        fail(f"dependency closure drifted: {sorted(set(rows) ^ expected_names)}")
    for name, version in EXPECTED_PINS.items():
        row = rows[name]
        locked = str(row.get("version", ""))
        if name == "torch":
            if locked not in {"2.7.1+cpu", "2.7.1"}:
                fail(f"torch lock version drifted: {locked}")
            if row.get("source", {}).get("registry") != PYTORCH_CPU_INDEX:
                fail("torch is not sourced from the PyTorch CPU index")
            urls = wheel_urls(row)
            if len(urls) != 1 or not re.search(
                r"torch-2\.7\.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64\.whl$", urls[0]
            ):
                fail(f"torch is not pinned to one Linux x86_64 CPU wheel: {urls!r}")
        elif locked != version:
            fail(f"{name} lock version drifted: {locked!r}")
        elif row.get("source", {}).get("registry") != "https://pypi.org/simple":
            fail(f"{name} must resolve from PyPI")
        if any(token in json.dumps(row, sort_keys=True).lower() for token in FORBIDDEN):
            fail(f"forbidden package/native payload marker found in {name} row")
    for name, row in rows.items():
        if any(token in json.dumps(row, sort_keys=True).lower() for token in FORBIDDEN):
            fail(f"forbidden package/native payload marker found in {name} row")
    return rows


def approval_scope(manifest: dict[str, Any], project_sha256: str, lock_sha256: str) -> str:
    approval = manifest.get("approval")
    signer = approval.get("signer") if isinstance(approval, dict) else None
    return canonical_digest(
        {
            "schema": APPROVAL_SCHEMA,
            "project_sha256": project_sha256,
            "uv_lock_sha256": lock_sha256,
            "component_rows": {
                "source": manifest.get("source"),
                "model": manifest.get("model"),
                "config": manifest.get("config"),
                "python_closure": manifest.get("python_closure"),
            },
            "publication": manifest.get("publication"),
            "signer": signer,
            "expected_status": AUTHORIZED_STATUS,
            "expected_owner_signoff": AUTHORIZED_SIGNOFF,
        }
    )


def validate_license_manifest(path: Path, project_sha256: str, lock_sha256: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        fail(f"missing or symlinked license gate manifest: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(f"unreadable license gate manifest: {error}")
    if not isinstance(data, dict):
        fail("license gate manifest root must be an object")
    expected_keys = {"gate_version", "status", "owner_signoff", "publication", "source", "model", "config", "python_closure", "evidence", "decision", "approval"}
    if set(data) != expected_keys or data.get("gate_version") != 1:
        fail("license gate manifest schema drifted")
    if data.get("status") not in {"PENDING_REVIEW", AUTHORIZED_STATUS}:
        fail("license gate has an unknown status")
    if data.get("owner_signoff") not in {"OWNER_SIGNOFF_REQUIRED", AUTHORIZED_SIGNOFF}:
        fail("license gate has an unknown owner signoff state")
    if data.get("publication") != "NO_UPLOAD":
        fail("license gate publication must remain NO_UPLOAD")
    source = data.get("source")
    model = data.get("model")
    closure = data.get("python_closure")
    if not isinstance(source, dict) or set(source) != set(EXPECTED_SOURCE) or source.get("repository") != EXPECTED_SOURCE["repository"] or source.get("revision") != EXPECTED_SOURCE["revision"] or source.get("license_status") not in {EXPECTED_SOURCE["license_status"], *APPROVED_COMPONENT_STATUSES}:
        fail("license source row schema drifted")
    if not isinstance(model, dict) or set(model) != set(EXPECTED_MODEL) or any(model.get(key) != value for key, value in EXPECTED_MODEL.items() if key != "weight_license_status") or model.get("weight_license_status") not in {EXPECTED_MODEL["weight_license_status"], *APPROVED_COMPONENT_STATUSES}:
        fail("license model row schema drifted")
    if not isinstance(data.get("config"), dict) or data["config"] != EXPECTED_CONFIG:
        fail("license config row drifted")
    if not isinstance(closure, dict) or set(closure) != {"runtime_pins", "platform", "license_status", "forbidden_packages"}:
        fail("license Python closure row schema drifted")
    approval = data.get("approval")
    if not isinstance(approval, dict) or set(approval) != {"schema", "signer", "scope_sha256"} or approval.get("schema") != APPROVAL_SCHEMA:
        fail("license approval schema drifted")
    if closure.get("platform") != "linux-x86_64-cpu":
        fail("license manifest platform drifted")
    if closure.get("runtime_pins") != EXPECTED_PINS:
        fail("license manifest dependency pins drifted")
    if tuple(closure.get("forbidden_packages", [])) != FORBIDDEN:
        fail("license manifest forbidden package list drifted")
    if closure.get("license_status") not in {EXPECTED_CLOSURE_STATUS, *APPROVED_COMPONENT_STATUSES}:
        fail("license Python closure status is unknown")
    if data.get("evidence") != EXPECTED_EVIDENCE:
        fail("license evidence rows drifted")
    if data.get("decision") != EXPECTED_DECISION:
        fail("license decision drifted")
    if data.get("status") == AUTHORIZED_STATUS or data.get("owner_signoff") == AUTHORIZED_SIGNOFF:
        if data.get("status") != AUTHORIZED_STATUS or data.get("owner_signoff") != AUTHORIZED_SIGNOFF:
            fail("license approval requires both approved status and owner signoff")
        if any(row.get(status_key) not in APPROVED_COMPONENT_STATUSES for row, status_key in ((source, "license_status"), (model, "weight_license_status"), (closure, "license_status"))):
            fail("license approval has unresolved component status")
        signer = approval.get("signer")
        if not isinstance(signer, str) or not signer.strip():
            fail("license approval signer is missing")
        if approval.get("scope_sha256") != approval_scope(data, project_sha256, lock_sha256):
            fail("license approval scope digest is stale or incorrect")
    return data


def gate(project_path: Path, lock_path: Path, license_path: Path) -> dict[str, Any]:
    project = read_toml(project_path)
    lock = read_toml(lock_path)
    validate_project(project)
    rows = validate_lock(lock)
    project_sha256 = sha256_bytes(project_path.read_bytes())
    lock_sha256 = sha256_bytes(lock_path.read_bytes())
    license_manifest = validate_license_manifest(license_path, project_sha256, lock_sha256)
    if (
        license_manifest.get("status") != AUTHORIZED_STATUS
        or license_manifest.get("owner_signoff") != AUTHORIZED_SIGNOFF
    ):
        fail("production execution is blocked until explicit owner license signoff")
    return {
        "status": "PASS",
        "project": project_path.as_posix(),
        "lock": lock_path.as_posix(),
        "package_count": len(rows),
        "platform": "linux-x86_64-cpu",
        "license_status": AUTHORIZED_STATUS,
        "owner_signoff": AUTHORIZED_SIGNOFF,
        "publication": "NO_UPLOAD",
        "project_sha256": project_sha256,
        "lock_sha256": lock_sha256,
        "license_manifest_sha256": sha256_bytes(license_path.read_bytes()),
        "approval_scope_sha256": license_manifest["approval"]["scope_sha256"],
    }


def self_test() -> None:
    here = Path(__file__).resolve().parent
    project = read_toml(here / "pyproject.toml")
    lock = read_toml(here / "uv.lock")
    validate_project(project)
    rows = validate_lock(lock)
    assert len(rows) == 13
    project_sha256 = sha256_bytes((here / "pyproject.toml").read_bytes())
    lock_sha256 = sha256_bytes((here / "uv.lock").read_bytes())
    manifest = validate_license_manifest(here / MANIFEST_NAME, project_sha256, lock_sha256)
    assert manifest["status"] == "PENDING_REVIEW"
    try:
        gate(here / "pyproject.toml", here / "uv.lock", here / MANIFEST_NAME)
    except GateError as error:
        assert "owner license signoff" in str(error)
    else:
        raise AssertionError("pending license gate was authorized")
    with tempfile.TemporaryDirectory(prefix="cosyvoice2-hift-preflight-") as temp:
        root = Path(temp)
        bad_project = root / "pyproject.toml"
        bad_project.write_text(
            (here / "pyproject.toml").read_text(encoding="utf-8").replace(
                "numpy==2.3.5", "numpy==2.3.4"
            ),
            encoding="utf-8",
        )
        try:
            gate(bad_project, here / "uv.lock", here / MANIFEST_NAME)
        except GateError:
            pass
        else:
            raise AssertionError("dependency drift was accepted")
        approved = root / MANIFEST_NAME
        approved_manifest = dict(manifest)
        approved_manifest["status"] = AUTHORIZED_STATUS
        approved_manifest["owner_signoff"] = AUTHORIZED_SIGNOFF
        approved_manifest["source"] = dict(approved_manifest["source"], license_status="APPROVED")
        approved_manifest["model"] = dict(approved_manifest["model"], weight_license_status="APPROVED")
        approved_manifest["python_closure"] = dict(approved_manifest["python_closure"], license_status="APPROVED")
        approved_manifest["approval"] = dict(approved_manifest["approval"], signer="owner@example.invalid")
        approved_manifest["approval"]["scope_sha256"] = approval_scope(approved_manifest, project_sha256, lock_sha256)
        approved.write_text(json.dumps(approved_manifest), encoding="utf-8")
        result = gate(here / "pyproject.toml", here / "uv.lock", approved)
        assert result["status"] == "PASS"
        drift_cases = {
            "source identity": ("source", "repository", "https://example.invalid/CosyVoice.git"),
            "model identity": ("model", "revision", "0" * 40),
            "config identity": ("config", "sha256", "0" * 64),
            "evidence facts": ("evidence", "tensor_count", 327),
            "source role blob": ("evidence", "source_roles", {"cosyvoice/hifigan/generator.py": "0" * 40}),
            "decision": ("decision", None, "ALLOW_UPLOAD"),
        }
        for label, (row_name, field, value) in drift_cases.items():
            candidate = copy.deepcopy(approved_manifest)
            if row_name == "evidence" and field == "source_roles":
                candidate[row_name][field]["cosyvoice/hifigan/generator.py"] = value["cosyvoice/hifigan/generator.py"]
            elif row_name == "decision":
                candidate[row_name] = value
            else:
                candidate[row_name][field] = value
            candidate["approval"]["scope_sha256"] = approval_scope(candidate, project_sha256, lock_sha256)
            drift_path = root / f"{label.replace(' ', '-')}.json"
            drift_path.write_text(json.dumps(candidate), encoding="utf-8")
            try:
                gate(here / "pyproject.toml", here / "uv.lock", drift_path)
            except GateError:
                pass
            else:
                raise AssertionError(f"{label} drift was accepted")
        stale = dict(approved_manifest)
        stale["approval"] = dict(stale["approval"], scope_sha256="0" * 64)
        stale_path = root / "stale.json"
        stale_path.write_text(json.dumps(stale), encoding="utf-8")
        try:
            gate(here / "pyproject.toml", here / "uv.lock", stale_path)
        except GateError as error:
            assert "scope digest" in str(error)
        else:
            raise AssertionError("stale approval scope was accepted")
        unresolved = dict(approved_manifest)
        unresolved["source"] = dict(unresolved["source"], license_status="PENDING_PRIMARY_SOURCE_REVIEW")
        unresolved["approval"] = dict(unresolved["approval"], scope_sha256=approval_scope(unresolved, project_sha256, lock_sha256))
        unresolved_path = root / "unresolved.json"
        unresolved_path.write_text(json.dumps(unresolved), encoding="utf-8")
        try:
            gate(here / "pyproject.toml", here / "uv.lock", unresolved_path)
        except GateError as error:
            assert "unresolved component status" in str(error)
        else:
            raise AssertionError("unresolved component approval was accepted")
    print("cosyvoice2_hift preflight self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--project", type=Path, default=Path(__file__).with_name("pyproject.toml"))
    parser.add_argument("--lock", type=Path, default=Path(__file__).with_name("uv.lock"))
    parser.add_argument("--license-manifest", type=Path, default=Path(__file__).with_name(MANIFEST_NAME))
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
        else:
            print(json.dumps(gate(args.project, args.lock, args.license_manifest), sort_keys=True))
    except (GateError, OSError, UnicodeError) as error:
        print(f"cosyvoice2_hift preflight: BLOCKED: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
