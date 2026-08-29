#!/usr/bin/env python3
"""Offline, fail-closed Conv-TasNet closure and approval gate.

This module is stdlib-only and is invoked before any VAST host probe, scratch
directory, uv cache, network access, conversion, or Cargo command.  The
upstream checkpoint has contradictory license notices, so production remains
blocked until an owner records a complete, externally authenticated review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

GATE_VERSION = 1
LOCK_SHA256 = "0124b1179f2795d324b94d975bd2dd9c1f7943e2951358c7cc427704935899f8"
PROJECT_SHA256 = "0ab5226a359c7d5268761d1f4685a1723d69c2430bba0735ed79be6f943b59f8"
UPSTREAM_REPO = "JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k"
UPSTREAM_REVISION = "bb8a876bc157b5cf3c405994accb798c49146016"
CHECKPOINT = {"path": "pytorch_model.bin", "bytes": 20130704, "sha256": "dd8ddefe95a35761f8a48643a618eba908572d04d33208a8ed5451fb5a4378d0"}
ASTEROID_ARTIFACTS = {
    "wheel_sha256": "ea97a24901d9d9851b4a594171bd7c6dd900fee2c132b9ce045aa09926d489c7",
    "sdist_sha256": "0326f28c5342495cb08ba0520efd0e21e39435dfd78854837fdd5a6c9c9ca410",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
LOCK_KEYS = {"version", "revision", "requires-python", "resolution-markers", "supported-markers", "package"}
PACKAGE_SCHEMAS = {
    frozenset({"name", "source", "version", "sdist", "wheels"}),
    frozenset({"name", "source", "version", "dependencies", "sdist", "wheels"}),
    frozenset({"name", "source", "version", "dependencies", "wheels"}),
    frozenset({"name", "source", "version", "optional-dependencies", "sdist", "wheels"}),
    frozenset({"name", "source", "version", "sdist"}),
    frozenset({"name", "source", "version", "dependencies", "metadata"}),
}
DEPENDENCY_SCHEMAS = {frozenset({"name"}), frozenset({"name", "marker"}), frozenset({"name", "marker", "extra"})}
ARTIFACT_KEYS = {"url", "hash", "size", "upload-time"}
PYPROJECT_KEYS = {"project", "tool"}
MANIFEST_KEYS = {"gate_version", "lock_sha256", "project_sha256", "package_rows_sha256", "package_review_rows", "package_review_rows_sha256", "identities", "license_rows", "license_rows_sha256", "reference_contract", "publication", "approval"}
REVIEW_KEYS = {"name", "version", "source", "status", "license", "native_bundled_review"}
LICENSE_KEYS = {"id", "status", "license", "conclusion", "evidence"}
PLACEHOLDERS = {"", "null", "none", "unresolved", "pending", "pending_review", "owner_review_required", "review_required", "todo"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canon(value: Any) -> str:
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def load_json(path: Path) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)


def resolved(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return re.sub(r"\s+", "_", value.strip().casefold()) not in PLACEHOLDERS


def artifact(value: Any, label: str, registry: str) -> None:
    if not isinstance(value, dict) or set(value) != ARTIFACT_KEYS:
        raise ValueError(f"{label} artifact schema is not exact")
    host = "files.pythonhosted.org" if registry == "https://pypi.org/simple" else "download-r2.pytorch.org"
    parsed = urlsplit(value["url"]) if isinstance(value["url"], str) else None
    if parsed is None or parsed.scheme != "https" or parsed.netloc != host or not parsed.path:
        raise ValueError(f"{label} artifact host is not authenticated")
    if not isinstance(value["hash"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value["hash"]):
        raise ValueError(f"{label} artifact hash is malformed")
    if isinstance(value["size"], bool) or not isinstance(value["size"], int) or value["size"] <= 0:
        raise ValueError(f"{label} artifact size is malformed")
    if not isinstance(value["upload-time"], str) or not value["upload-time"].strip():
        raise ValueError(f"{label} artifact upload-time is missing")


def metadata(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {"requires-dist"} or not isinstance(value["requires-dist"], list):
        raise ValueError("lock metadata schema is malformed")
    for row in value["requires-dist"]:
        if not isinstance(row, dict) or set(row) not in ({"name", "specifier"}, {"name", "specifier", "index"}) or not isinstance(row.get("name"), str) or not row["name"].strip() or not isinstance(row.get("specifier"), str) or not row["specifier"].strip():
            raise ValueError("lock metadata requirement is malformed")
        if "index" in row and row["index"] not in ("https://pypi.org/simple", "https://download.pytorch.org/whl/cpu"):
            raise ValueError("lock metadata index is not approved")


def lock_rows(lock: dict[str, Any]) -> list[dict[str, Any]]:
    if set(lock) != LOCK_KEYS or lock.get("version") != 1 or lock.get("revision") != 3 or lock.get("requires-python") != "==3.12.*":
        raise ValueError("lock top-level schema is not exact")
    for key in ("resolution-markers", "supported-markers"):
        if not isinstance(lock[key], list) or any(not isinstance(item, str) or not item.strip() for item in lock[key]):
            raise ValueError("lock marker schema is malformed")
    packages = lock.get("package")
    if not isinstance(packages, list) or not packages:
        raise ValueError("lock package table is missing or empty")
    seen: set[tuple[str, str]] = set(); virtual = 0; rows = []
    for package in packages:
        if not isinstance(package, dict) or set(package) - {"name", "version", "source", "resolution-markers", "dependencies", "optional-dependencies", "sdist", "wheels", "metadata"}:
            raise ValueError("lock package row contains unknown fields")
        name, version, source = package.get("name"), package.get("version"), package.get("source")
        if not isinstance(name, str) or not name.strip() or not isinstance(version, str) or not version.strip():
            raise ValueError("lock package name/version is malformed")
        key = (name, version)
        if key in seen:
            raise ValueError("lock package identity is duplicated")
        seen.add(key)
        if not isinstance(source, dict) or len(source) != 1 or set(source) not in ({"registry"}, {"virtual"}):
            raise ValueError("lock package source is malformed")
        if "registry" in source and source["registry"] not in ("https://pypi.org/simple", "https://download.pytorch.org/whl/cpu"):
            raise ValueError("lock registry is not approved")
        if "virtual" in source:
            virtual += 1
            if source["virtual"] != "." or frozenset(package) != frozenset({"name", "version", "source", "dependencies", "metadata"}):
                raise ValueError("virtual project package schema is not exact")
            metadata(package["metadata"])
        elif frozenset(package) not in PACKAGE_SCHEMAS:
            raise ValueError(f"package schema is not an exact committed variant: {key!r}")
        markers = package.get("resolution-markers", [])
        if not isinstance(markers, list) or any(not isinstance(item, str) or not item.strip() for item in markers):
            raise ValueError("package resolution-markers are malformed")
        dependencies = package.get("dependencies", [])
        if not isinstance(dependencies, list):
            raise ValueError("package dependencies are malformed")
        for dep in dependencies:
            if not isinstance(dep, dict) or frozenset(dep) not in DEPENDENCY_SCHEMAS or not isinstance(dep.get("name"), str) or not dep["name"].strip():
                raise ValueError("dependency row schema is malformed")
            if "marker" in dep and (not isinstance(dep["marker"], str) or not dep["marker"].strip()):
                raise ValueError("dependency marker is blank")
            if "extra" in dep and (not isinstance(dep["extra"], list) or any(not isinstance(item, str) or not item.strip() for item in dep["extra"])):
                raise ValueError("dependency extra is malformed")
        if "optional-dependencies" in package:
            optional = package["optional-dependencies"]
            if not isinstance(optional, dict):
                raise ValueError("optional dependency table is malformed")
            for group, values in optional.items():
                if not isinstance(group, str) or not group.strip() or not isinstance(values, list):
                    raise ValueError("optional dependency group is malformed")
                for dep in values:
                    if not isinstance(dep, dict) or set(dep) != {"name", "marker"} or not isinstance(dep.get("name"), str) or not dep["name"].strip() or not isinstance(dep["marker"], str) or not dep["marker"].strip():
                        raise ValueError("optional dependency row is malformed")
        registry = source.get("registry", "")
        if "metadata" in package and "virtual" not in source:
            metadata(package["metadata"])
        for kind in ("sdist", "wheels"):
            if kind in package:
                values = package[kind] if kind == "wheels" else [package[kind]]
                if kind == "wheels" and (not isinstance(values, list) or not values):
                    raise ValueError("wheel artifact table is malformed")
                if kind == "sdist" and not isinstance(package[kind], dict):
                    raise ValueError("sdist artifact is malformed")
                for item in values:
                    artifact(item, f"{key} {kind}", registry)
        if "virtual" not in source and "sdist" not in package and not package.get("wheels"):
            raise ValueError("registry package has no authenticated artifact")
        rows.append(package)
    if virtual != 1:
        raise ValueError("lock must contain exactly one virtual project")
    return sorted(rows, key=lambda row: (row["name"], row["version"]))


def project_schema(project: dict[str, Any]) -> None:
    if set(project) != {"project", "tool"} or not isinstance(project.get("project"), dict) or set(project["project"]) != {"name", "version", "description", "requires-python", "dependencies"}:
        raise ValueError("pyproject structural schema is not exact")
    p = project["project"]
    if any(not isinstance(p.get(key), str) or not p[key].strip() for key in ("name", "version", "description", "requires-python")) or p["name"] != "vokra-conv-tasnet-parity" or p["version"] != "0.1.0" or p["description"] != "Pinned CPU-only Asteroid 0.7.0 oracle for Conv-TasNet parity; VAST only." or p["requires-python"] != "==3.12.*" or p["dependencies"] != ["asteroid==0.7.0", "huggingface-hub==1.28.0", "numpy==2.3.5", "safetensors==0.7.0", "torch==2.9.1"]:
        raise ValueError("pyproject project fields are malformed")
    tool = project["tool"]
    if set(tool) != {"uv"} or not isinstance(tool["uv"], dict) or set(tool["uv"]) != {"package", "environments", "sources", "index"}:
        raise ValueError("pyproject uv schema is not exact")
    uv = tool["uv"]
    if uv["package"] is not False or uv["environments"] != ["sys_platform == 'linux' and platform_machine == 'x86_64'"] or uv["sources"] != {"torch": {"index": "pytorch-cpu"}} or uv["index"] != [{"name": "pytorch-cpu", "url": "https://download.pytorch.org/whl/cpu", "explicit": True}]:
        raise ValueError("pyproject CPU-only uv contract drifted")


def run(lock_path: Path, project_path: Path, manifest_path: Path, evidence_path: Path | None) -> None:
    if any(path.is_symlink() or not path.is_file() for path in (lock_path, project_path, manifest_path)):
        blocked("lock, project, or manifest is not a regular file")
    try:
        lock_bytes = lock_path.read_bytes(); project_bytes = project_path.read_bytes(); manifest = load_json(manifest_path)
        lock = tomllib.loads(lock_bytes.decode()); project = tomllib.loads(project_bytes.decode())
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, json.JSONDecodeError, ValueError) as error:
        blocked(f"closure input is unreadable: {error}")
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS or manifest.get("gate_version") != GATE_VERSION:
        blocked("manifest schema or gate version is invalid")
    if sha(lock_bytes) != LOCK_SHA256 or manifest.get("lock_sha256") != LOCK_SHA256 or sha(project_bytes) != PROJECT_SHA256 or manifest.get("project_sha256") != PROJECT_SHA256:
        blocked("project or lock bytes are not the fixed reviewed closure")
    try:
        project_schema(project); rows = lock_rows(lock)
    except (ValueError, KeyError) as error:
        blocked(str(error))
    asteroid_rows = [row for row in rows if row["name"] == "asteroid" and row["version"] == "0.7.0"]
    if len(asteroid_rows) != 1:
        blocked("locked Asteroid identity is missing or duplicated")
    asteroid_row = asteroid_rows[0]
    if asteroid_row.get("sdist", {}).get("hash") != f"sha256:{ASTEROID_ARTIFACTS['sdist_sha256']}" or not any(item.get("hash") == f"sha256:{ASTEROID_ARTIFACTS['wheel_sha256']}" for item in asteroid_row.get("wheels", [])):
        blocked("locked Asteroid artifact hashes do not match the authenticated identity")
    virtual = [row for row in rows if row["source"] == {"virtual": "."}]
    if len(virtual) != 1 or (virtual[0]["name"], virtual[0]["version"]) != (project["project"]["name"], project["project"]["version"]):
        blocked("virtual project identity is not bound to pyproject")
    if not isinstance(manifest.get("package_rows_sha256"), str) or not HEX64.fullmatch(manifest["package_rows_sha256"]) or canon(rows) != manifest["package_rows_sha256"]:
        blocked("canonical package rows drifted")
    reviews = manifest.get("package_review_rows")
    if not isinstance(reviews, list) or len(reviews) != len(rows) or canon(reviews) != manifest.get("package_review_rows_sha256"):
        blocked("every locked package needs an exact version-keyed review row")
    actual = {(row["name"], row["version"]): row for row in rows}; seen: set[tuple[str, str]] = set()
    for review in reviews:
        if not isinstance(review, dict) or set(review) != REVIEW_KEYS:
            blocked("package review row schema is not exact")
        key = (review.get("name"), review.get("version"))
        if key in seen or key not in actual or review.get("source") != actual[key]["source"] or review.get("status") != "REVIEWED" or not resolved(review.get("license")) or not resolved(review.get("native_bundled_review")):
            blocked(f"package review is unresolved or not bound: {key!r}")
        seen.add(key)
    if seen != set(actual):
        blocked("package reviews do not cover exact lock closure")
    identities = manifest.get("identities")
    expected = {"upstream_repo": UPSTREAM_REPO, "upstream_revision": UPSTREAM_REVISION, "checkpoint": CHECKPOINT, "asteroid_artifacts": ASTEROID_ARTIFACTS}
    if identities != expected:
        blocked("fixed upstream/checkpoint/Asteroid identities drifted")
    contracts = manifest.get("reference_contract")
    if not isinstance(contracts, dict) or set(contracts) != {"topology", "sample_rate", "license_context"} or contracts.get("topology") != {"n_filters": 512, "kernel_size": 32, "stride": 16, "bn_chan": 128, "hid_chan": 512, "n_blocks": 8, "n_repeats": 3, "n_src": 1} or contracts.get("sample_rate") != 16000 or contracts.get("license_context") != {"upstream_yaml": "CC-BY-SA-4.0", "license_body": "CC-BY-SA-3.0", "wham": "CC-BY-NC-4.0 Research-only"}:
        blocked("topology or license contradiction contract drifted")
    license_rows = manifest.get("license_rows")
    if not isinstance(license_rows, list) or [row.get("id") for row in license_rows if isinstance(row, dict)] != ["upstream-license-contradiction", "wham-research-restriction", "asteroid-python-closure"] or any(not isinstance(row, dict) or set(row) != LICENSE_KEYS or row.get("status") != "UNRESOLVED" for row in license_rows) or canon(license_rows) != manifest.get("license_rows_sha256"):
        blocked("license review rows are not the exact unresolved contract")
    if manifest.get("publication") != "NO_UPLOAD":
        blocked("publication must remain NO_UPLOAD")
    approval = manifest.get("approval")
    if not isinstance(approval, dict) or set(approval) != {"status", "signer", "digest"} or approval.get("status") != "OWNER_SIGNOFF_APPROVED" or not resolved(approval.get("signer")) or approval.get("digest") != canon({"lock_sha256": LOCK_SHA256, "project_sha256": PROJECT_SHA256, "package_rows": rows, "package_review_rows": reviews, "license_rows": license_rows, "identities": identities, "reference_contract": contracts, "publication": "NO_UPLOAD"}):
        blocked("external owner approval is missing or not bound")
    if evidence_path is None or evidence_path.is_symlink() or not evidence_path.is_file():
        blocked("external approval evidence is missing or not regular")
    try: evidence = load_json(evidence_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error: blocked(f"approval evidence is unreadable: {error}")
    if not isinstance(evidence, dict) or set(evidence) != {"schema", "decision", "signer", "scope_sha256", "manifest_sha256", "lock_sha256", "project_sha256"} or evidence.get("schema") != "conv-tasnet-approval-v1" or evidence.get("decision") != "APPROVED" or evidence.get("signer") != approval["signer"] or evidence.get("scope_sha256") != approval["digest"] or evidence.get("manifest_sha256") != sha(manifest_path.read_bytes()) or evidence.get("lock_sha256") != LOCK_SHA256 or evidence.get("project_sha256") != PROJECT_SHA256:
        blocked("approval evidence is not bound to exact scope")
    print("conv-tasnet license gate: PASS")


def blocked(message: str) -> None:
    print(f"conv-tasnet license gate: BLOCKED: {message}", file=sys.stderr)
    raise SystemExit(2)


def self_test() -> int:
    project = Path(__file__).parent
    ok = False
    try: run(project / "uv.lock", project / "pyproject.toml", project / "license_gate_manifest.json", None)
    except SystemExit as error: ok = error.code == 2
    if not ok:
        print("conv-tasnet gate self-test: production closure unexpectedly passed", file=sys.stderr); return 1
    with __import__("tempfile").TemporaryDirectory(prefix="conv-tasnet-gate-") as raw:
        root = Path(raw); duplicate = root / "duplicate.json"; duplicate.write_text('{"gate_version":1,"gate_version":1}', encoding="utf-8")
        try: load_json(duplicate)
        except ValueError: pass
        else: return 1
    valid_artifact = {"url": "https://files.pythonhosted.org/pkg.whl", "hash": "sha256:" + "a" * 64, "size": 1, "upload-time": "2026-01-01T00:00:00Z"}
    for tampered in (
        {key: value for key, value in valid_artifact.items() if key != "size"},
        {**valid_artifact, "size": True},
        {**valid_artifact, "upload-time": ""},
        {**valid_artifact, "extra": "rejected"},
        {**valid_artifact, "url": "https://evil.example/pkg.whl"},
    ):
        try: artifact(tampered, "self-test", "https://pypi.org/simple")
        except ValueError: pass
        else: return 1
    print("conv-tasnet gate self-test: PASS")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--lock", type=Path); parser.add_argument("--project", type=Path); parser.add_argument("--manifest", type=Path); parser.add_argument("--evidence", type=Path); parser.add_argument("--self-test", action="store_true"); args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.lock, args.project, args.manifest, args.evidence)): parser.error("--self-test accepts no other arguments")
        raise SystemExit(self_test())
    if not all(value is not None for value in (args.lock, args.project, args.manifest, args.evidence)): parser.error("lock/project/manifest/evidence are required")
    run(args.lock, args.project, args.manifest, args.evidence)
