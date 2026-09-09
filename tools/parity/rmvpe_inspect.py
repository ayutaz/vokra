#!/usr/bin/env -S uv run --no-project --python 3.12 python
"""Fail-closed RMVPE identity, lock, and dependency-license gate.

This inspector is intentionally stdlib-only.  It runs before any VAST worker
action which could resolve a package, clone source, or acquire a checkpoint.
The lock digest binds every package row, dependency qualifier, and resolution
marker; a resolved lock is evidence of reproducibility, not license approval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).with_name("rmvpe")
LOCK = PROJECT / "uv.lock"
PYPI_REGISTRY = "https://pypi.org/simple"
PYPI_EVIDENCE_PREFIX = "https://pypi.org/pypi/"
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
LOCK_SHA256 = "747057f4e8596d801d5d0450e6e10a33fc467ab9e9a6cf2063460d1ea019919d"
PACKAGE_ROWS_SHA256 = "ecc622c63e8a487c4440cdc838f22af7b31fae783cca41f693b0f870dd9a1819"
RESOLUTION_MARKERS_SHA256 = "70a0c0d228b605430c8219bfc8e4ed66652a5f06d64cab841fee543266f3bffa"
LICENSE_ROWS_SHA256 = "2afebac3c079863d28415885412c11fd2acf7e3f3b9a686e2c855455da8eedec"
PACKAGE_COUNT = 40

UPSTREAM_REPOSITORY = "https://github.com/yxlllc/RMVPE"
UPSTREAM_REVISION = "0aabafba18289ca938a73af0b0297686abf4922d"
SOURCE_MANIFEST = {
    "repository": UPSTREAM_REPOSITORY,
    "revision": UPSTREAM_REVISION,
    "license": {"path": None, "git_blob": None, "status": "ABSENT_AT_FIXED_REVISION"},
    "roles": {
        "README.md": "c8faab898598c878b4694d5f16846e694d5e9646",
        "src/inference.py": "6c004cf87abf73e457a9dbc153d8a83f680c8b4d",
        "src/model.py": "214788a381fe83ebe2cc7c5e531cae37aef47e0b",
        "src/deepunet.py": "d0d5d777b760ad7a9c9cd9c23014065401948248",
        "src/spec.py": "cc3e7916f6e993925c840089ce3b3be5e1a907a4",
        "src/constants.py": "6ddee8211193ef41fd95705a80c9139ac81ffe74",
    },
}
RELEASE_ARCHIVE = {
    "repository": UPSTREAM_REPOSITORY,
    "tag": "230917",
    "asset": "rmvpe.zip",
    "url": "https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip",
    "size": 340638958,
    # GitHub did not publish a digest for this asset.  Do not hash bytes on
    # the Mac; VAST acquisition must fill this field before execution.
    "bytes_sha256": None,
    "member": "model.pt",
}
MODEL_ARTIFACT = {
    "repository": "yxlllc/RMVPE release 230917",
    "path": "model.pt",
    "bytes_sha256": None,
    "status": "UNSET_UNTIL_VAST_ACQUISITION",
}
PUBLIC_TARGET = {
    "repository": "vokra/rmvpe",
    "revision": "3eb5fa8946f1074ba3959074c5cde95ec22b8c91",
    "path": "rmvpe.gguf",
    "size": 181010688,
    "bytes_sha256": "208fc73819586b4546f2cba7a829033c5900c44af1ad48fe9d3e727cc1a932fb",
    "provenance": "REJECTED_MISSTAMPS_MIT_PERMISSIVE; NOT_A_PARITY_INPUT",
}

# Evidence is keyed by exact lock name/version/source.  The URLs point to the
# primary release metadata or official PyTorch CPU wheel index.  Native and
# bundled notices deliberately remain blockers until owner review.
LICENSE_ROWS = [
    {"name": "audioread", "version": "3.1.0", "license": "MIT", "source": "https://pypi.org/pypi/audioread/3.1.0/json", "status": "UNREVIEWED"},
    {"name": "certifi", "version": "2026.7.22", "license": "MPL-2.0", "source": "https://pypi.org/pypi/certifi/2026.7.22/json", "status": "BLOCKED_POLICY"},
    {"name": "cffi", "version": "2.1.1", "license": "MIT_NATIVE_EXTENSION", "source": "https://pypi.org/pypi/cffi/2.1.1/json", "status": "BLOCKED_NATIVE_REVIEW"},
    {"name": "charset-normalizer", "version": "3.5.1", "license": "MIT", "source": "https://pypi.org/pypi/charset-normalizer/3.5.1/json", "status": "UNREVIEWED"},
    {"name": "decorator", "version": "5.3.1", "license": "BSD-2-Clause", "source": "https://pypi.org/pypi/decorator/5.3.1/json", "status": "UNREVIEWED"},
    {"name": "filelock", "version": "3.32.4", "license": "Unlicense", "source": "https://pypi.org/pypi/filelock/3.32.4/json", "status": "BLOCKED_POLICY"},
    {"name": "fsspec", "version": "2026.7.0", "license": "BSD-3-Clause", "source": "https://pypi.org/pypi/fsspec/2026.7.0/json", "status": "UNREVIEWED"},
    {"name": "idna", "version": "3.19", "license": "BSD-3-Clause", "source": "https://pypi.org/pypi/idna/3.19/json", "status": "UNREVIEWED"},
    {"name": "jinja2", "version": "3.1.6", "license": "BSD-3-Clause", "source": "https://pypi.org/pypi/Jinja2/3.1.6/json", "status": "UNREVIEWED"},
    {"name": "joblib", "version": "1.5.3", "license": "BSD-3-Clause", "source": "https://pypi.org/pypi/joblib/1.5.3/json", "status": "UNREVIEWED"},
    {"name": "lazy-loader", "version": "0.5", "license": "BSD-3-Clause", "source": "https://pypi.org/pypi/lazy-loader/0.5/json", "status": "UNREVIEWED"},
    {"name": "librosa", "version": "0.11.0", "license": "ISC_NATIVE_AUDIO_CLOSURE", "source": "https://pypi.org/pypi/librosa/0.11.0/json", "status": "BLOCKED_CLOSURE"},
    {"name": "llvmlite", "version": "0.49.0", "license": "BSD-2-Clause_NATIVE_WHEEL", "source": "https://pypi.org/pypi/llvmlite/0.49.0/json", "status": "BLOCKED_NATIVE_REVIEW"},
    {"name": "markupsafe", "version": "3.0.3", "license": "BSD-3-Clause_NATIVE_EXTENSION", "source": "https://pypi.org/pypi/MarkupSafe/3.0.3/json", "status": "BLOCKED_NATIVE_REVIEW"},
    {"name": "mpmath", "version": "1.3.0", "license": "BSD-3-Clause", "source": "https://pypi.org/pypi/mpmath/1.3.0/json", "status": "UNREVIEWED"},
    {"name": "msgpack", "version": "1.2.2", "license": "Apache-2.0", "source": "https://pypi.org/pypi/msgpack/1.2.2/json", "status": "UNREVIEWED"},
    {"name": "narwhals", "version": "2.25.0", "license": "MIT", "source": "https://pypi.org/pypi/narwhals/2.25.0/json", "status": "UNREVIEWED"},
    {"name": "networkx", "version": "3.6.1", "license": "BSD-3-Clause", "source": "https://pypi.org/pypi/networkx/3.6.1/json", "status": "UNREVIEWED"},
    {"name": "numba", "version": "0.67.0", "license": "BSD-2-Clause_NATIVE_WHEEL", "source": "https://pypi.org/pypi/numba/0.67.0/json", "status": "BLOCKED_NATIVE_REVIEW"},
    {"name": "numpy", "version": "2.3.5", "license": "BSD-3-Clause_AND_BUNDLED_NOTICES", "source": "https://pypi.org/pypi/numpy/2.3.5/json", "status": "BLOCKED_BUNDLED_NOTICES"},
    {"name": "packaging", "version": "26.3", "license": "Apache-2.0", "source": "https://pypi.org/pypi/packaging/26.3/json", "status": "UNREVIEWED"},
    {"name": "platformdirs", "version": "4.11.5", "license": "MIT", "source": "https://pypi.org/pypi/platformdirs/4.11.5/json", "status": "UNREVIEWED"},
    {"name": "pooch", "version": "1.9.0", "license": "BSD-3-Clause", "source": "https://pypi.org/pypi/pooch/1.9.0/json", "status": "UNREVIEWED"},
    {"name": "pycparser", "version": "3.0", "license": "BSD-3-Clause", "source": "https://pypi.org/pypi/pycparser/3.0/json", "status": "UNREVIEWED"},
    {"name": "requests", "version": "2.34.2", "license": "Apache-2.0", "source": "https://pypi.org/pypi/requests/2.34.2/json", "status": "UNREVIEWED"},
    {"name": "rmvpe-parity", "version": "0.2.0", "license": "FIRST_PARTY", "source": "repository", "status": "UNREVIEWED"},
    {"name": "safetensors", "version": "0.8.0", "license": "Apache-2.0_NATIVE_EXTENSION", "source": "https://pypi.org/pypi/safetensors/0.8.0/json", "status": "BLOCKED_NATIVE_REVIEW"},
    {"name": "scikit-learn", "version": "1.9.0", "license": "BSD-3-Clause_NATIVE_WHEEL", "source": "https://pypi.org/pypi/scikit-learn/1.9.0/json", "status": "BLOCKED_NATIVE_REVIEW"},
    {"name": "scipy", "version": "1.18.1", "license": "BSD-3-Clause_AND_BUNDLED_NOTICES", "source": "https://pypi.org/pypi/scipy/1.18.1/json", "status": "BLOCKED_BUNDLED_NOTICES"},
    {"name": "setuptools", "version": "84.0.0", "license": "MIT", "source": "https://pypi.org/pypi/setuptools/84.0.0/json", "status": "UNREVIEWED"},
    {"name": "soundfile", "version": "0.14.0", "license": "BSD-3-Clause_PLUS_LIBSNDFILE_LGPL", "source": "https://pypi.org/pypi/soundfile/0.14.0/json", "status": "BLOCKED_NATIVE_REVIEW"},
    {"name": "soxr", "version": "1.1.0", "license": "LGPL-2.1_NATIVE", "source": "https://pypi.org/pypi/soxr/1.1.0/json", "status": "BLOCKED_POLICY"},
    {"name": "sympy", "version": "1.14.0", "license": "BSD-3-Clause", "source": "https://pypi.org/pypi/sympy/1.14.0/json", "status": "UNREVIEWED"},
    {"name": "threadpoolctl", "version": "3.6.0", "license": "BSD-3-Clause", "source": "https://pypi.org/pypi/threadpoolctl/3.6.0/json", "status": "UNREVIEWED"},
    {"name": "torch", "version": "2.7.1", "license": "BSD-3-Clause_PLUS_BUNDLED_NOTICES", "source": "https://download.pytorch.org/whl/cpu/torch-2.7.1-cp312-none-macosx_11_0_arm64.whl", "status": "BLOCKED_BUNDLED_NOTICES"},
    {"name": "torch", "version": "2.7.1+cpu", "license": "BSD-3-Clause_PLUS_BUNDLED_NOTICES", "source": "https://download.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl", "status": "BLOCKED_BUNDLED_NOTICES"},
    {"name": "torchaudio", "version": "2.7.1", "license": "BSD-2-Clause_PLUS_BUNDLED_NOTICES", "source": "https://download.pytorch.org/whl/cpu/torchaudio-2.7.1-cp312-cp312-manylinux_2_28_x86_64.whl", "status": "BLOCKED_BUNDLED_NOTICES"},
    {"name": "torchaudio", "version": "2.7.1+cpu", "license": "BSD-2-Clause_PLUS_BUNDLED_NOTICES", "source": "https://download.pytorch.org/whl/cpu/torchaudio-2.7.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl", "status": "BLOCKED_BUNDLED_NOTICES"},
    {"name": "typing-extensions", "version": "4.16.0", "license": "PSF-2.0", "source": "https://pypi.org/pypi/typing-extensions/4.16.0/json", "status": "BLOCKED_POLICY"},
    {"name": "urllib3", "version": "2.7.0", "license": "MIT", "source": "https://pypi.org/pypi/urllib3/2.7.0/json", "status": "UNREVIEWED"},
]

BLOCKERS = [
    "yxlllc/RMVPE has no LICENSE at the fixed source revision; checkpoint terms remain unknown",
    "librosa@0.11.0 resolves soxr@1.1.0 (LGPL/native) and soundfile@0.14.0 (bundled libsndfile/LGPL via cffi)",
    "numba@0.67.0 and llvmlite@0.49.0 are native wheels requiring separate notice review",
    "safetensors@0.8.0 is a native extension wheel requiring separate notice review",
    "numpy@2.3.5 and scipy@1.18.1 carry bundled third-party notices",
    "official CPU torch/torchaudio wheels carry bundled notices not yet audited",
    "MPL/Unlicense/PSF rows require owner policy clearance; no sign-off is present",
    "checkpoint and release archive byte SHA-256 remain unset until VAST-only acquisition",
]


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def canonical_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    packages = document.get("package")
    if not isinstance(packages, list):
        raise ValueError("uv.lock package table is missing")
    by_identity: dict[tuple[str, str, str], dict[str, Any]] = {}
    for package in packages:
        if not isinstance(package, dict) or not isinstance(package.get("name"), str) or not isinstance(package.get("version"), str):
            raise ValueError("malformed uv.lock package row")
        source = package.get("source")
        if not isinstance(source, dict) or len(source) != 1 or not all(isinstance(value, str) for value in source.values()):
            raise ValueError(f"malformed source for {package['name']}")
        key = (package["name"], package["version"], json.dumps(source, sort_keys=True, separators=(",", ":")))
        if key in by_identity:
            raise ValueError(f"duplicate package identity: {key}")
        by_identity[key] = package
    rows: list[dict[str, Any]] = []
    for package in packages:
        source = package["source"]
        markers = package.get("resolution-markers", [])
        if not isinstance(markers, list) or any(not isinstance(marker, str) for marker in markers):
            raise ValueError(f"malformed resolution markers for {package['name']}")
        dependencies = []
        for dependency in package.get("dependencies", []):
            if not isinstance(dependency, dict) or not isinstance(dependency.get("name"), str):
                raise ValueError(f"malformed dependency in {package['name']}")
            qualifier = {key: dependency[key] for key in sorted(dependency)}
            candidates = [
                row for row in packages
                if isinstance(row, dict) and row.get("name") == dependency["name"]
                and ("version" not in dependency or row.get("version") == dependency["version"])
                and ("source" not in dependency or row.get("source") == dependency["source"])
            ]
            if len(candidates) != 1:
                raise ValueError(f"dependency qualifier does not resolve uniquely: {package['name']} -> {dependency['name']}")
            qualifier["resolved_version"] = candidates[0]["version"]
            qualifier["resolved_source"] = candidates[0]["source"]
            dependencies.append(qualifier)
        rows.append({
            "name": package["name"],
            "version": package["version"],
            "source": source,
            "resolution_markers": sorted(markers),
            "dependencies": sorted(dependencies, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))),
        })
    return sorted(rows, key=lambda row: (row["name"], row["version"], json.dumps(row["source"], sort_keys=True)))


def audit_lock() -> dict[str, Any]:
    if not LOCK.is_file() or hashlib.sha256(LOCK.read_bytes()).hexdigest() != LOCK_SHA256:
        raise ValueError("dedicated RMVPE uv.lock is absent or identity drifted")
    document = tomllib.loads(LOCK.read_text(encoding="utf-8"))
    if document.get("requires-python") != "==3.12.*":
        raise ValueError("lock is not Python 3.12-only")
    markers = document.get("resolution-markers")
    if digest(sorted(markers)) != RESOLUTION_MARKERS_SHA256:
        raise ValueError("resolution marker digest drifted")
    rows = canonical_rows(document)
    if len(rows) != PACKAGE_COUNT or digest(rows) != PACKAGE_ROWS_SHA256:
        raise ValueError("package/dependency row digest drifted")
    evidence = sorted(LICENSE_ROWS, key=lambda row: (row["name"], row["version"], row["source"]))
    if len(evidence) != PACKAGE_COUNT or digest(evidence) != LICENSE_ROWS_SHA256:
        raise ValueError("license evidence digest or row count drifted")
    for row in rows:
        name = row["name"]
        expected_source = {"virtual": "."} if name == "rmvpe-parity" else {
            "registry": PYTORCH_CPU_INDEX if name in {"torch", "torchaudio"} else PYPI_REGISTRY
        }
        if row["source"] != expected_source:
            raise ValueError(f"unexpected package source for {name}@{row['version']}: {row['source']}")
    for row in evidence:
        source = row["source"]
        if row["name"] == "rmvpe-parity":
            valid_source = source == "repository"
        elif row["name"] in {"torch", "torchaudio"}:
            valid_source = source.startswith(PYTORCH_CPU_INDEX + "/")
        else:
            valid_source = source.startswith(PYPI_EVIDENCE_PREFIX)
        if not valid_source:
            raise ValueError(f"license evidence is not a primary source for {row['name']}@{row['version']}")
    if {(row["name"], row["version"]) for row in rows} != {(row["name"], row["version"]) for row in evidence}:
        raise ValueError("license evidence does not cover every lock name/version")
    return {
        "package_count": len(rows),
        "lock_sha256": LOCK_SHA256,
        "package_rows_sha256": PACKAGE_ROWS_SHA256,
        "resolution_markers_sha256": RESOLUTION_MARKERS_SHA256,
        "license_rows_sha256": LICENSE_ROWS_SHA256,
        "source_manifest": SOURCE_MANIFEST,
        "release_archive": RELEASE_ARCHIVE,
        "model_artifact": MODEL_ARTIFACT,
        "public_target": PUBLIC_TARGET,
        "blockers": BLOCKERS,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dependency-gate", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        audit_lock()
        assert all(len(value) == 40 for value in (UPSTREAM_REVISION,))
        assert RELEASE_ARCHIVE["bytes_sha256"] is None
        assert MODEL_ARTIFACT["bytes_sha256"] is None
        print("rmvpe inspector self-test: PASS")
        return 0
    if not args.dependency_gate:
        parser.error("use --dependency-gate or --self-test")
    result = audit_lock()
    print(json.dumps({"status": "BLOCKED_UNREVIEWED_NATIVE_AND_MODEL_LICENSE", "publication": "NO_UPLOAD", **result}, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
