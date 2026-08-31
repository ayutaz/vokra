#!/usr/bin/env -S uv run --no-project --python 3.12 python
"""Fail-closed dependency and provenance gate for microWakeWord.

This inspector intentionally does not import ai-edge-litert, numpy, or gguf.
It authenticates the dedicated lock and its complete resolved dependency
graph before a VAST worker may sync an environment or materialize a TFLite
artifact.  The model conversion/reference path remains VAST-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).with_name("microwakeword")
LOCK = PROJECT / "uv.lock"
MODEL_REPOSITORY = "esphome/micro-wake-word-models"
MODEL_REVISION = "05b65922cc433c9df13e98e32a7fe520758c837e"
SOURCE_REPOSITORY = "https://github.com/kahrendt/microWakeWord"
SOURCE_REVISION = "4665173cd35f1cff9a61e06fc427f124766c488e"
MODEL_ARTIFACT_BYTES_SHA256: str | None = None
LOCK_SHA256 = "05e8317758e7c884e8e86e110af5b39cdd23eff63b6a66705225e6baa3ab5e13"
PACKAGE_ROWS_SHA256 = "d5c8aaca80e340be13e719de14d1486df193977f31ea80dea3bf954030057343"
RESOLUTION_MARKERS_SHA256 = "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
LICENSE_ROWS_SHA256 = "eae9f062f7ceb787fe36e09290fbc04b8f2f842df9de612b79f95f7fd615c58f"
PACKAGE_COUNT = 10

# These are Git object IDs (not file SHA-256 values).  The model bytes digest
# remains intentionally unset until the VAST-only acquisition records it.
SOURCE_MANIFEST = {
    "repository": SOURCE_REPOSITORY,
    "revision": SOURCE_REVISION,
    "license": {"path": "LICENSE", "git_blob": "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64", "size": 11357},
    "roles": {
        "inference.py": "ec0634376accb8e7832205c117149f4acb3e6cf0",
        "mixednet.py": "75cbb9fa950fa4135a0e3a4171b9fba84c4b989c",
        "layers/stream.py": "37b77702c8ee8038c4e6e91979560e264e7555c1",
        "audio/spectrograms.py": "5adb585ab3a650dfd17728a0e200a143d41c23f7",
        "pyproject.toml": "e2156f94b8a2bc4821cccd72492889016e40b532",
    },
}
MODEL_MANIFEST = {
    "repository": MODEL_REPOSITORY,
    "revision": MODEL_REVISION,
    "license": {"path": "LICENSE", "git_blob": "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64", "size": 11357},
    "target": {
        "path": "models/v2/hey_jarvis.tflite",
        "git_blob": "0075302434cc72a460ced0b8f6c09c69214e5cf0",
        "size": 52272,
        "bytes_sha256": MODEL_ARTIFACT_BYTES_SHA256,
    },
    "companion": {
        "path": "models/v2/hey_jarvis.json",
        "git_blob": "e6733fe13852f04a5a3ae83e0d39b5726aee62cc",
        "size": 388,
    },
}

# Version-keyed evidence is deliberately conservative.  For each exact PyPI
# JSON record, evidence is selected in this order: license_expression, then
# license, then wheel METADATA/Classifiers. A resolved lock is not license
# approval: policy-sensitive rows keep the worker at exit 2.
LICENSE_ROWS = [
    {"name": "ai-edge-litert", "version": "2.2.0", "license": "Apache-2.0_PRECOMPILED_WHEEL_NOTICES_REVIEW_REQUIRED", "evidence_field": "license", "primary_source": "https://pypi.org/pypi/ai-edge-litert/2.2.0/json"},
    {"name": "backports-strenum", "version": "1.3.1", "license": "MIT", "evidence_field": "license", "primary_source": "https://pypi.org/pypi/backports-strenum/1.3.1/json"},
    {"name": "colorama", "version": "0.4.6", "license": "BSD-3-Clause", "evidence_field": "classifiers", "primary_source": "https://pypi.org/pypi/colorama/0.4.6/json"},
    {"name": "flatbuffers", "version": "25.12.19", "license": "Apache-2.0", "evidence_field": "license", "primary_source": "https://pypi.org/pypi/flatbuffers/25.12.19/json"},
    {"name": "microwakeword-prep", "version": "0.1.0", "license": "FIRST_PARTY", "evidence_field": "repository", "primary_source": "repository"},
    {"name": "ml-dtypes", "version": "0.6.0", "license": "Apache-2.0_EIGEN_MPL-2.0_WHEEL_NOTICE_REVIEW_REQUIRED", "evidence_field": "description/license section", "primary_source": "https://pypi.org/pypi/ml-dtypes/0.6.0/json"},
    {"name": "numpy", "version": "2.5.2", "license": "BSD-3-Clause_AND_0BSD_AND_MIT_AND_Zlib_AND_CC0-1.0_BUNDLED_NOTICES_REVIEW_REQUIRED", "evidence_field": "license_expression", "primary_source": "https://pypi.org/pypi/numpy/2.5.2/json"},
    {"name": "protobuf", "version": "7.36.0", "license": "BSD-3-Clause_METADATA_REVIEW_REQUIRED", "evidence_field": "license", "primary_source": "https://pypi.org/pypi/protobuf/7.36.0/json"},
    {"name": "tqdm", "version": "4.70.0", "license": "MPL-2.0_AND_MIT_BLOCKED_BY_POLICY", "evidence_field": "license", "primary_source": "https://pypi.org/pypi/tqdm/4.70.0/json"},
    {"name": "typing-extensions", "version": "4.16.0", "license": "PSF-2.0_BLOCKED_BY_POLICY", "evidence_field": "license_expression", "primary_source": "https://pypi.org/pypi/typing-extensions/4.16.0/json"},
]
BLOCKERS = [
    "ai-edge-litert==2.2.0: precompiled TFLite runtime wheel notices require review",
    "tqdm==4.70.0: MPL-2.0/MIT requires owner policy clearance",
    "typing-extensions==4.16.0: PSF-2.0 requires owner policy clearance",
    "numpy==2.5.2: PyPI license_expression includes bundled BSD/0BSD/MIT/Zlib/CC0 notices requiring review",
    "ml-dtypes==0.6.0: exact PyPI description/license section declares an Eigen/MPL-2.0 notice for precompiled wheels",
    "protobuf==7.36.0: metadata/precompiled-wheel notice review is required",
    "models/v2/hey_jarvis.tflite: artifact byte SHA-256 is pending VAST-only acquisition",
    "hey_jarvis tensor manifest: authenticated VAST constant-buffer inspection is required",
]


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def canonical_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    packages = document.get("package")
    if not isinstance(packages, list):
        raise RuntimeError("uv.lock package table is missing")
    names = [p.get("name") for p in packages if isinstance(p, dict)]
    if len(names) != len(set(names)):
        raise RuntimeError("uv.lock has ambiguous duplicate package names")
    by_name = {p.get("name"): p for p in packages if isinstance(p, dict)}
    rows: list[dict[str, Any]] = []
    for package in packages:
        if not isinstance(package, dict) or not isinstance(package.get("name"), str) or not isinstance(package.get("version"), str):
            raise RuntimeError("malformed uv.lock package row")
        source = package.get("source")
        if not isinstance(source, dict) or len(source) != 1 or not all(isinstance(v, str) for v in source.values()):
            raise RuntimeError(f"malformed source for {package.get('name')}")
        markers = package.get("resolution-markers", [])
        if not isinstance(markers, list) or any(not isinstance(marker, str) for marker in markers):
            raise RuntimeError(f"malformed resolution markers for {package.get('name')}")
        deps = []
        for dep in package.get("dependencies", []):
            if not isinstance(dep, dict) or not isinstance(dep.get("name"), str):
                raise RuntimeError(f"malformed dependency in {package['name']}")
            target = by_name.get(dep["name"])
            if not isinstance(target, dict):
                raise RuntimeError(f"unresolved dependency {dep['name']}")
            # Keep every dependency qualifier from uv.lock and bind it to the
            # exact target version/source; names alone are insufficient.
            row = {k: dep[k] for k in sorted(dep)}
            row["resolved_version"] = target["version"]
            row["resolved_source"] = {k: target["source"][k] for k in sorted(target["source"])}
            deps.append(row)
        rows.append({
            "name": package["name"],
            "version": package["version"],
            "source": {k: source[k] for k in sorted(source)},
            "resolution_markers": sorted(markers),
            "dependencies": sorted(deps, key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":"))),
        })
    return sorted(rows, key=lambda row: (row["name"], row["version"], json.dumps(row["source"], sort_keys=True)))


def audit_lock() -> dict[str, Any]:
    if not LOCK.is_file() or hashlib.sha256(LOCK.read_bytes()).hexdigest() != LOCK_SHA256:
        raise RuntimeError("dedicated microWakeWord uv.lock is absent or identity drifted")
    project = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
    if project.get("project", {}).get("dependencies") != [
        "numpy==2.5.2", "ai-edge-litert==2.2.0"
    ]:
        raise RuntimeError("direct dependency pins drifted from the reviewed lock")
    document = tomllib.loads(LOCK.read_text(encoding="utf-8"))
    if document.get("requires-python") != "==3.12.*":
        raise RuntimeError("microWakeWord lock is not Python 3.12-only")
    markers = document.get("resolution-markers", [])
    if digest(sorted(markers)) != RESOLUTION_MARKERS_SHA256:
        raise RuntimeError("resolution marker identity drifted")
    rows = canonical_rows(document)
    if len(rows) != PACKAGE_COUNT or digest(rows) != PACKAGE_ROWS_SHA256:
        raise RuntimeError("package/dependency resolution identity drifted")
    declared = sorted(LICENSE_ROWS, key=lambda row: (row["name"], row["version"]))
    if len(declared) != PACKAGE_COUNT or digest(declared) != LICENSE_ROWS_SHA256:
        raise RuntimeError("version-keyed license evidence digest is invalid")
    locked = {(row["name"], row["version"]) for row in rows}
    evidenced = {(row["name"], row["version"]) for row in declared}
    if locked != evidenced:
        raise RuntimeError("license evidence does not cover every locked package")
    return {"package_count": len(rows), "package_rows_sha256": PACKAGE_ROWS_SHA256, "resolution_markers_sha256": RESOLUTION_MARKERS_SHA256, "license_rows_sha256": LICENSE_ROWS_SHA256, "model_repository": MODEL_REPOSITORY, "model_revision": MODEL_REVISION, "model_manifest": MODEL_MANIFEST, "source_repository": SOURCE_REPOSITORY, "source_revision": SOURCE_REVISION, "source_manifest": SOURCE_MANIFEST, "blockers": BLOCKERS}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dependency-gate", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        assert "--project" not in Path(__file__).read_text(encoding="utf-8").splitlines()[0]
        assert digest(sorted(LICENSE_ROWS, key=lambda row: (row["name"], row["version"]))) == LICENSE_ROWS_SHA256
        assert SOURCE_REVISION == "4665173cd35f1cff9a61e06fc427f124766c488e"
        assert MODEL_REVISION == "05b65922cc433c9df13e98e32a7fe520758c837e"
        assert MODEL_MANIFEST == {
            "repository": "esphome/micro-wake-word-models",
            "revision": "05b65922cc433c9df13e98e32a7fe520758c837e",
            "license": {"path": "LICENSE", "git_blob": "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64", "size": 11357},
            "target": {"path": "models/v2/hey_jarvis.tflite", "git_blob": "0075302434cc72a460ced0b8f6c09c69214e5cf0", "size": 52272, "bytes_sha256": None},
            "companion": {"path": "models/v2/hey_jarvis.json", "git_blob": "e6733fe13852f04a5a3ae83e0d39b5726aee62cc", "size": 388},
        }
        assert SOURCE_MANIFEST["license"] == {"path": "LICENSE", "git_blob": "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64", "size": 11357}
        assert MODEL_MANIFEST["target"]["bytes_sha256"] is None
        assert SOURCE_MANIFEST["roles"]["inference.py"] == "ec0634376accb8e7832205c117149f4acb3e6cf0"
        assert audit_lock()["package_count"] == PACKAGE_COUNT
        print("microwakeword inspector self-test: PASS")
        return 0
    if not args.dependency_gate:
        parser.error("use --dependency-gate or --self-test")
    audit = audit_lock()
    print(json.dumps({"status": "BLOCKED_UNREVIEWED_TRANSITIVE", "publication": "NO_UPLOAD", **audit}, sort_keys=True))
    print("microWakeWord dependency/license gate is blocked: " + "; ".join(BLOCKERS), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
