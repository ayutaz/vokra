#!/usr/bin/env -S uv run --no-sync --frozen --project tools/parity/baichuan_audio_instruct --python 3.12 python
"""Inspection-only evidence collector for Baichuan-Audio-Instruct.

The five-shard, ~21 GB release is never merged or loaded into resident
memory. Safetensors headers are parsed directly and only metadata is retained;
all composition, source, and license claims remain fail-closed evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

HF_REPOSITORY = "baichuan-inc/Baichuan-Audio-Instruct"
HF_REVISION = "1c86512d863376f9ea0c32bb77451b9f428283c8"
SOURCE_REPOSITORY = "https://github.com/baichuan-inc/Baichuan-Audio.git"
SOURCE_REVISION = "805d456433dbf3e0edb2bdd302f733a4bd38ea84"
SOURCE_ROLE_PATHS = (
    "web_demo/generation.py", "baichuan_audio/modeling_baichuan_audio.py",
    "baichuan_audio/configuration_baichuan_audio.py", "baichuan_audio/processing_baichuan_audio.py",
    "baichuan_audio/generation_baichuan_audio.py", "baichuan_audio/flow_matching.py",
    "baichuan_audio/matcha_components.py", "baichuan_audio/vector_quantize.py",
    "third_party/Matcha-TTS/matcha/models/components/flow_matching.py",
    "third_party/Matcha-TTS/matcha/models/components/decoder.py",
    "third_party/cosy24k_vocoder/LICENSE", "NOTICE", "LICENSE",
)
# Values are deliberately empty until the exact upstream source checkout is
# available for an offline authenticated audit.  Empty rows remain blocked;
# they are not self-asserted hashes and can never produce MATCHED evidence.
SOURCE_ROLE_BLOBS: dict[str, str] = {}
# The Matcha gitlink is deliberately not inferred from a checkout or from
# .gitmodules.  It must be filled from the authenticated upstream tree before
# this inspector can ever report a usable source identity.
MATCHA_REVISION: str | None = None
COMPONENT_COUNT = 5
FORMAT = "vokra-baichuan-audio-instruct-inspection-v1"
REFERENCE_PROJECT = Path(__file__).with_name("baichuan_audio_instruct")
REFERENCE_LOCK = REFERENCE_PROJECT / "uv.lock"
REFERENCE_LOCK_SHA256 = "0e8ca64e2f81060732c317fd6d10e01df7c3a5eb122426ef5d695e9813df7625"
REFERENCE_PACKAGE_ROWS_SHA256 = "a276c50b73fcbc7f0ac22667d6d56516bf7dea7e9e420456562f128b4fa36b2b"
REFERENCE_RESOLUTION_MARKERS_SHA256 = "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
REFERENCE_PACKAGE_COUNT = 14
DEPENDENCY_LICENSE_AUDIT_STATUS = "BLOCKED_UNREVIEWED_TRANSITIVE"
# These are version-specific conclusions from the corresponding PyPI JSON
# primary metadata.  Conservative suffixes keep MPL, PSF, and native cases
# blocked until the owner records an explicit exception and upstream notices
# are reviewed; no row grants installation or execution permission.
DEPENDENCY_LICENSE_CONCLUSIONS = {
    "certifi==2026.7.22": ("MPL-2.0_BLOCKED_BY_POLICY", "https://pypi.org/pypi/certifi/2026.7.22/json"),
    "charset-normalizer==3.5.1": ("MIT_REVIEWED", "https://pypi.org/pypi/charset-normalizer/3.5.1/json"),
    "colorama==0.4.6": ("BSD-3-Clause_REVIEWED", "https://pypi.org/pypi/colorama/0.4.6/json"),
    "filelock==3.32.4": ("MIT_REVIEWED", "https://pypi.org/pypi/filelock/3.32.4/json"),
    "fsspec==2026.7.0": ("BSD-3-Clause_REVIEWED", "https://pypi.org/pypi/fsspec/2026.7.0/json"),
    "huggingface-hub==0.24.7": ("Apache-2.0_REVIEWED", "https://pypi.org/pypi/huggingface-hub/0.24.7/json"),
    "idna==3.19": ("BSD-3-Clause_REVIEWED", "https://pypi.org/pypi/idna/3.19/json"),
    "packaging==26.3": ("Apache-2.0_REVIEWED", "https://pypi.org/pypi/packaging/26.3/json"),
    "pyyaml==6.0.3": ("MIT_NATIVE_EXTENSION_REVIEW_REQUIRED", "https://pypi.org/pypi/PyYAML/6.0.3/json"),
    "requests==2.34.2": ("Apache-2.0_REVIEWED", "https://pypi.org/pypi/requests/2.34.2/json"),
    "tqdm==4.70.0": ("MPL-2.0_AND_MIT_BLOCKED_BY_POLICY", "https://pypi.org/pypi/tqdm/4.70.0/json"),
    "typing-extensions==4.16.0": ("PSF-2.0_BLOCKED_BY_POLICY", "https://pypi.org/pypi/typing-extensions/4.16.0/json"),
    "urllib3==2.7.0": ("MIT_REVIEWED", "https://pypi.org/pypi/urllib3/2.7.0/json"),
    "vokra-baichuan-audio-instruct-inspection==0.1.0": ("FIRST_PARTY_NOT_INDEPENDENT_DEPENDENCY_SCOPE", "repository"),
}
LICENSE_BLOCKERS = (
    "certifi==2026.7.22 declares MPL-2.0; owner policy clearance is required",
    "tqdm==4.70.0 declares MPL-2.0 AND MIT; file-level MPL review and owner policy clearance are required",
    "typing-extensions==4.16.0 is PSF-2.0; owner policy clearance is required",
    "PyYAML==6.0.3 has a native extension; native distribution notice review is required",
)
MAX_HEADER_BYTES = 64 * 1024 * 1024
CUSTOM_ROLE_FILES = (
    "audio_modeling_omni.py", "modeling_omni.py", "configuration_omni.py",
    "flow_matching.py", "matcha_components.py", "matcha_feat.py",
    "matcha_transformer.py", "processor_omni.py", "generation_utils.py",
    "vector_quantize.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_lock_rows(packages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a duplicate-safe identity for every uv.lock package row.

    Package-level resolution markers and dependency markers are retained even
    though this minimal environment has only one platform-independent
    resolution.  This prevents a future lock rewrite from silently changing a
    conditional dependency while preserving only name/version/source.
    """
    rows: list[dict[str, Any]] = []
    for package in packages:
        if not isinstance(package, dict) or not isinstance(package.get("name"), str) or not isinstance(package.get("version"), str):
            raise RuntimeError("uv.lock contains a malformed package row")
        source = package.get("source")
        if not isinstance(source, dict) or set(source) not in ({"registry"}, {"virtual"}) or not isinstance(next(iter(source.values()), None), str):
            raise RuntimeError(f"uv.lock package source is malformed: {package.get('name')}")
        resolution_markers = package.get("resolution-markers", [])
        if not isinstance(resolution_markers, list) or any(not isinstance(marker, str) for marker in resolution_markers):
            raise RuntimeError(f"uv.lock package resolution markers are malformed: {package['name']}")
        dependency_markers: list[dict[str, str | None]] = []
        dependencies = package.get("dependencies", [])
        if not isinstance(dependencies, list):
            raise RuntimeError(f"uv.lock package dependencies are malformed: {package['name']}")
        for dependency in dependencies:
            if not isinstance(dependency, dict) or not isinstance(dependency.get("name"), str):
                raise RuntimeError(f"uv.lock dependency row is malformed: {package['name']}")
            marker = dependency.get("marker")
            if marker is not None and not isinstance(marker, str):
                raise RuntimeError(f"uv.lock dependency marker is malformed: {package['name']}")
            dependency_markers.append({"name": dependency["name"], "marker": marker})
        identity = {
            "name": package["name"],
            "version": package["version"],
            "source": {key: source[key] for key in sorted(source)},
            "resolution_markers": sorted(resolution_markers),
            "dependency_markers": sorted(dependency_markers, key=lambda item: (item["name"], item["marker"] or "")),
        }
        rows.append({
            **identity,
            "row_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        })
    return sorted(rows, key=lambda row: (row["name"], row["version"], json.dumps(row["source"], sort_keys=True), row["resolution_markers"], row["dependency_markers"]))


def lock_rows_sha256(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def resolution_markers_sha256(markers: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(markers), separators=(",", ":")).encode("utf-8")).hexdigest()


def lock_dependency_audit(lock_path: Path = REFERENCE_LOCK) -> dict[str, Any]:
    """Authenticate the reviewed lock and all versioned license rows."""
    if not lock_path.is_file():
        raise RuntimeError("dedicated Baichuan inspection uv.lock is absent")
    lock_digest = sha256(lock_path)
    if lock_digest != REFERENCE_LOCK_SHA256:
        raise RuntimeError("dedicated Baichuan inspection uv.lock SHA-256 is not the reviewed identity")
    try:
        document = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError(f"dedicated uv.lock is not valid TOML: {error}") from error
    if document.get("requires-python") != "==3.12.*":
        raise RuntimeError("dedicated uv.lock is not restricted to Python 3.12")
    resolution_markers = document.get("resolution-markers", [])
    if not isinstance(resolution_markers, list) or any(not isinstance(marker, str) for marker in resolution_markers):
        raise RuntimeError("dedicated uv.lock resolution markers are malformed")
    resolution_markers = sorted(resolution_markers)
    if resolution_markers_sha256(resolution_markers) != REFERENCE_RESOLUTION_MARKERS_SHA256:
        raise RuntimeError("dedicated uv.lock resolution marker identity drifted")
    packages = document.get("package")
    if not isinstance(packages, list):
        raise RuntimeError("dedicated uv.lock package table is malformed")
    rows = canonical_lock_rows(packages)
    if len(rows) != REFERENCE_PACKAGE_COUNT or lock_rows_sha256(rows) != REFERENCE_PACKAGE_ROWS_SHA256:
        raise RuntimeError("dedicated uv.lock package row identity or marker digest drifted")
    names = {row["name"] for row in rows}
    forbidden = sorted(name for name in names if name == "soxr" or name == "triton" or name.startswith("nvidia-"))
    if forbidden:
        raise RuntimeError(f"forbidden dependency is present in dedicated lock: {forbidden}")
    conclusions = {
        f"{row['name']}=={row['version']}": {
            "license": DEPENDENCY_LICENSE_CONCLUSIONS[f"{row['name']}=={row['version']}"][0],
            "primary_source": DEPENDENCY_LICENSE_CONCLUSIONS[f"{row['name']}=={row['version']}"][1],
            "source": next(iter(row["source"].values())),
        }
        for row in rows
        if f"{row['name']}=={row['version']}" in DEPENDENCY_LICENSE_CONCLUSIONS
    }
    if len(conclusions) != len(rows) or set(conclusions) != set(DEPENDENCY_LICENSE_CONCLUSIONS):
        raise RuntimeError("version-specific primary-source license conclusion inventory is incomplete")
    try:
        project = tomllib.loads((REFERENCE_PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError(f"dedicated pyproject license metadata is invalid: {error}") from error
    metadata = project.get("tool", {}).get("vokra", {}).get("baichuan_audio_instruct_inspection", {})
    if not isinstance(metadata, dict):
        raise RuntimeError("dedicated dependency license metadata is missing")
    expected = {
        "lock_sha256": REFERENCE_LOCK_SHA256,
        "lock_package_rows_sha256": REFERENCE_PACKAGE_ROWS_SHA256,
        "lock_resolution_markers_sha256": REFERENCE_RESOLUTION_MARKERS_SHA256,
        "lock_package_count": REFERENCE_PACKAGE_COUNT,
        "marker_digest_schema": "package-resolution-and-dependency-markers-v2",
        "dependency_license_audit": DEPENDENCY_LICENSE_AUDIT_STATUS,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RuntimeError("dedicated dependency gate metadata is not bound to the reviewed lock")
    blocker_rows = metadata.get("license_blockers")
    if tuple(blocker_rows or ()) != LICENSE_BLOCKERS:
        raise RuntimeError("dedicated license blocker evidence drifted")
    declared_rows = metadata.get("license_conclusions")
    if not isinstance(declared_rows, list) or len(declared_rows) != len(rows):
        raise RuntimeError("dedicated license conclusion rows are missing")
    declared = {
        f"{row.get('name')}=={row.get('version')}": {
            "license": row.get("license"),
            "primary_source": row.get("primary_source"),
        }
        for row in declared_rows
        if isinstance(row, dict)
    }
    if len(declared) != len(declared_rows) or declared != {key: {"license": value["license"], "primary_source": value["primary_source"]} for key, value in conclusions.items()}:
        raise RuntimeError("dedicated license conclusion rows do not match locked versions")
    return {
        "status": metadata["dependency_license_audit"],
        "lock_sha256": lock_digest,
        "package_rows_sha256": REFERENCE_PACKAGE_ROWS_SHA256,
        "resolution_markers_sha256": REFERENCE_RESOLUTION_MARKERS_SHA256,
        "package_count": len(rows),
        "package_rows": rows,
        "license_conclusions": conclusions,
        "blockers": list(LICENSE_BLOCKERS),
    }


def require_dependency_gate() -> dict[str, Any]:
    audit = lock_dependency_audit()
    if audit["status"] != "AUDITED_ALLOW":
        blockers = "; ".join(audit["blockers"])
        raise RuntimeError(f"Baichuan dependency/license gate is not affirmatively approved; blocked before sync/acquisition: {blockers}")
    return audit


def git_blob_sha1_bytes(data: bytes) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {len(data)}\0".encode())
    digest.update(data)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    """Hash a regular Git blob without allocating a whole model shard."""
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def lfs_pointer_bytes(payload_sha256: str, payload_bytes: int) -> bytes:
    return f"version https://git-lfs.github.com/spec/v1\noid sha256:{payload_sha256}\nsize {payload_bytes}\n".encode()


def strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def regular_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise RuntimeError(f"missing directory: {root}")
    result = []
    for path in sorted(root.rglob("*")):
        if any(part in {".cache", ".git"} for part in path.relative_to(root).parts):
            continue
        if path.is_dir() and not path.is_symlink():
            continue
        if path.is_symlink():
            if not path.exists() or not path.is_file():
                raise RuntimeError(f"dangling/non-regular symlink: {path}")
            raise RuntimeError(f"symlink is not an authenticated regular file: {path}")
        if not path.is_file():
            raise RuntimeError(f"non-regular file: {path}")
        result.append(path)
    if not result:
        raise RuntimeError(f"empty directory: {root}")
    return result


def identity(path: Path, root: Path) -> dict[str, Any]:
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}


def safe_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError("path must be a non-empty UTF-8 relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.name in {"", "."}:
        raise ValueError(f"path escapes the snapshot: {value!r}")
    return path.as_posix()


def strict_shard_name(value: Any) -> str:
    if not isinstance(value, str) or "\x00" in value or "\\" in value:
        raise ValueError("shard name must be a plain string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.name != value:
        raise ValueError(f"shard path is not root-direct: {value!r}")
    if not re.fullmatch(r"model-0000[1-5]-of-00005\.safetensors", value):
        raise ValueError(f"unexpected canonical shard name: {value!r}")
    return value


def license_declaration(path: Path, root: Path, expected: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"(?im)^\s*license\s*:\s*([^#\n]+)", text)
    declared = match.group(1).strip().strip('"\'') if match else ""
    status = "AUTHENTICATED_DECLARATION" if declared.lower() == expected else "UNKNOWN_MISMATCH"
    return {**identity(path, root), "declared": declared, "expected": expected, "status": status}


def server_tree(snapshot: Path, packet: Path, blockers: list[str]) -> dict[str, Any]:
    try:
        remote = json.loads(packet.read_text(encoding="utf-8"), object_pairs_hook=strict_pairs)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        blockers.append(f"server tree parse failed: {error}")
        return {"status": "BLOCKED_PACKET_PARSE"}
    if not isinstance(remote, dict) or remote.get("repository") != HF_REPOSITORY or remote.get("revision") != HF_REVISION or remote.get("resolved_revision") != HF_REVISION:
        blockers.append("server tree repository/revision/resolved_revision mismatch")
    rows = remote.get("files", []) if isinstance(remote, dict) else []
    if not isinstance(rows, list) or not rows:
        blockers.append("server tree has no complete file rows")
        rows = []
    remote_by_path: dict[str, dict[str, Any]] = {}
    for item in rows:
        if not isinstance(item, dict) or set(item) != {"path", "type", "size", "git_blob_sha1", "lfs_pointer_git_blob_sha1", "lfs_sha256"}:
            blockers.append("server tree row is malformed, duplicated, or weakly identified")
            continue
        if item.get("type") != "file" or not isinstance(item.get("path"), str) or not isinstance(item.get("size"), int) or item["size"] < 0 or (item.get("git_blob_sha1") is not None and not re.fullmatch(r"[0-9a-f]{40}", str(item["git_blob_sha1"]))) or (item.get("lfs_pointer_git_blob_sha1") is not None and not re.fullmatch(r"[0-9a-f]{40}", str(item["lfs_pointer_git_blob_sha1"]))) or (item.get("lfs_sha256") is not None and not re.fullmatch(r"[0-9a-f]{64}", str(item["lfs_sha256"]))):
            blockers.append("server tree row is malformed, duplicated, or weakly identified")
            continue
        if (item["lfs_sha256"] is None and (item["git_blob_sha1"] is None or item["lfs_pointer_git_blob_sha1"] is not None)) or (item["lfs_sha256"] is not None and (item["git_blob_sha1"] is not None or item["lfs_pointer_git_blob_sha1"] is None)):
            blockers.append("server tree row must distinguish regular Git blobs from LFS pointer blobs")
            continue
        try:
            item["path"] = safe_relative_path(item["path"])
        except ValueError as error:
            blockers.append(f"server tree path is unsafe: {error}")
            continue
        if item["path"] in remote_by_path:
            blockers.append("server tree contains duplicate normalized paths")
            continue
        remote_by_path[item["path"]] = item
    remote_set = {(path, row["size"]) for path, row in remote_by_path.items()}
    local = regular_files(snapshot)
    local_set = {(path.relative_to(snapshot).as_posix(), path.stat().st_size) for path in local}
    missing, extra = sorted(remote_set - local_set), sorted(local_set - remote_set)
    if missing or extra:
        blockers.append(f"server/local tree mismatch: missing={missing!r} extra={extra!r}")
    verified_files = []
    for path in local:
        relative = path.relative_to(snapshot).as_posix()
        row = remote_by_path.get(relative)
        if row is None:
            continue
        payload_bytes = path.stat().st_size
        if payload_bytes != row["size"]:
            blockers.append(f"materialized payload size mismatch: {relative}")
        payload_sha = sha256(path)
        if row["lfs_sha256"] is None:
            if git_blob_sha1(path) != row["git_blob_sha1"]:
                blockers.append(f"canonical Git blob SHA-1 mismatch: {relative}")
        else:
            if payload_sha != row["lfs_sha256"]:
                blockers.append(f"LFS payload SHA-256 mismatch: {relative}")
            pointer_sha = git_blob_sha1_bytes(lfs_pointer_bytes(payload_sha, payload_bytes))
            if pointer_sha != row["lfs_pointer_git_blob_sha1"]:
                blockers.append(f"canonical Git-LFS pointer Git blob SHA-1 mismatch: {relative}")
        verified_files.append({"path": relative, "bytes": payload_bytes, "payload_bytes": payload_bytes, "git_blob_sha1": row["git_blob_sha1"], "lfs_pointer_git_blob_sha1": row["lfs_pointer_git_blob_sha1"], "lfs_sha256": row["lfs_sha256"], "payload_sha256": payload_sha})
    return {"status": "MATCHED" if not blockers else "MISMATCH", "repository": remote.get("repository"), "revision": remote.get("revision"), "resolved_revision": remote.get("resolved_revision"), "packet_sha256": sha256(packet), "files": sorted(verified_files, key=lambda item: item["path"]), "missing": missing, "extra": extra}


DTYPE_BYTES = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1, "I16": 2, "U16": 2, "I32": 4, "U32": 4, "I64": 8, "U64": 8, "BOOL": 1}


def safetensors_header(path: Path, root: Path, blockers: list[str]) -> dict[str, Any]:
    item = identity(path, root)
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError("short header length")
            header_size = int.from_bytes(prefix, "little")
            file_size = path.stat().st_size
            if header_size > file_size - 8 or header_size > MAX_HEADER_BYTES:
                raise ValueError(f"header length {header_size} exceeds safe limit")
            header_bytes = handle.read(header_size)
            if len(header_bytes) != header_size:
                raise ValueError("truncated header")
        header = json.loads(header_bytes, object_pairs_hook=strict_pairs)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        blockers.append(f"safetensors header parse failed for {path}: {error}")
        return {**item, "status": "BLOCKED_HEADER_PARSE", "error": str(error)}
    if not isinstance(header, dict):
        blockers.append(f"safetensors header is not an object: {path}")
        return {**item, "status": "BLOCKED_HEADER_PARSE"}
    if header_size > path.stat().st_size - 8:
        blockers.append(f"safetensors header exceeds file: {path}")
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items()):
        blockers.append(f"safetensors metadata is not a string map: {path}")
    data_start = 8 + header_size
    file_size = path.stat().st_size
    ranges: list[tuple[int, int, str]] = []
    tensors: list[dict[str, Any]] = []
    for name, spec in sorted(header.items()):
        if name == "__metadata__":
            continue
        if not isinstance(spec, dict) or not isinstance(spec.get("shape"), list) or not isinstance(spec.get("data_offsets"), list) or len(spec["data_offsets"]) != 2:
            blockers.append(f"malformed tensor entry: {path}:{name}")
            continue
        dtype = str(spec.get("dtype"))
        if any(isinstance(axis, bool) or not isinstance(axis, int) or axis < 0 for axis in spec["shape"]):
            blockers.append(f"tensor shape axes must be non-negative integers: {path}:{name}")
            continue
        offsets = spec["data_offsets"]
        if any(isinstance(offset, bool) or not isinstance(offset, int) or offset < 0 for offset in offsets):
            blockers.append(f"tensor offsets must be non-negative integers: {path}:{name}")
            continue
        shape = list(spec["shape"])
        start, end = offsets
        elements = 1
        for axis in shape:
            if axis < 0:
                blockers.append(f"negative tensor shape: {path}:{name}")
            elements *= axis
        expected = elements * DTYPE_BYTES.get(dtype, 0)
        if dtype not in DTYPE_BYTES or end < start or start + expected != end or data_start + end > file_size:
            blockers.append(f"invalid tensor byte range/shape/dtype: {path}:{name}")
        ranges.append((start, end, name))
        tensors.append({"name": name, "shape": shape, "dtype": dtype, "elements": elements, "data_offsets": [start, end], "finite": "NOT_CHECKED_HEADER_ONLY"})
    ranges.sort()
    cursor = 0
    for start, end, name in ranges:
        if start < cursor:
            blockers.append(f"overlapping tensor ranges: {path}:{name}")
        if start > cursor:
            blockers.append(f"gap in tensor data region: {path}:{start}-{cursor}")
        cursor = max(cursor, end)
    if cursor != file_size - data_start:
        blockers.append(f"tensor data region does not end at file boundary: {path}")
    item.update({"status": "HEADER_ONLY", "header_bytes": header_size, "metadata": header.get("__metadata__", {}), "tensor_count": len(tensors), "tensors": tensors, "resident_scope": "header-only; tensor body never read", "finite": "NOT_CHECKED_HEADER_ONLY"})
    blockers.append(f"tensor body finiteness is unverified in header-only inspection: {path}")
    return item


def json_evidence(path: Path, root: Path, blockers: list[str]) -> dict[str, Any]:
    item = identity(path, root)
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=strict_pairs)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        blockers.append(f"JSON parse failed: {path}: {error}")
        return {**item, "status": "BLOCKED_JSON_PARSE", "error": str(error)}
    item.update({"status": "PARSED_CANONICAL_JSON", "type": type(value).__name__, "top_level_keys": sorted(value) if isinstance(value, dict) else None, "object_count": len(value) if isinstance(value, (dict, list)) else 1, "raw": value if path.name.endswith("index.json") else None})
    return item


def validate_json_semantics(snapshot: Path, packets: list[dict[str, Any]], blockers: list[str]) -> dict[str, Any]:
    """Require typed component topology; never infer missing hyperparameters."""
    values: dict[str, Any] = {}
    for packet in packets:
        path = snapshot / packet["path"]
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=strict_pairs)
        except Exception:
            continue
        values[packet["path"]] = value
    config = values.get("config.json")
    if not isinstance(config, dict):
        blockers.append("canonical config.json object is missing")
    else:
        required = ("model_type", "architectures", "auto_map", "hidden_size", "num_hidden_layers", "num_attention_heads", "vocab_size")
        if any(key not in config for key in required):
            blockers.append("config.json lacks required model topology fields")
        if not isinstance(config.get("model_type"), str) or not isinstance(config.get("architectures"), list) or not config.get("architectures") or any(not isinstance(value, str) for value in config["architectures"]):
            blockers.append("config.json architecture semantics are malformed")
        if not isinstance(config.get("auto_map"), dict) or not config["auto_map"] or any(not isinstance(key, str) or not isinstance(value, str) for key, value in config["auto_map"].items()):
            blockers.append("config.json custom auto_map semantics are missing or malformed")
        for key in ("hidden_size", "num_hidden_layers", "num_attention_heads", "vocab_size"):
            if isinstance(config.get(key), bool) or not isinstance(config.get(key), int) or config[key] <= 0:
                blockers.append(f"config.json {key} is not a positive integer")
        if all(isinstance(config.get(key), int) and config[key] > 0 for key in ("hidden_size", "num_attention_heads")) and config["hidden_size"] % config["num_attention_heads"]:
            blockers.append("config hidden/head dimensions are not divisible")
    for name in ("tokenizer_config.json", "processor_config.json", "preprocessor_config.json", "generation_config.json"):
        matches = [value for path, value in values.items() if Path(path).name == name]
        if len(matches) != 1 or not isinstance(matches[0], dict):
            blockers.append(f"canonical {name} semantic object is missing or ambiguous")
    tokenizer = next((value for path, value in values.items() if Path(path).name == "tokenizer_config.json"), None)
    if isinstance(tokenizer, dict) and not isinstance(tokenizer.get("tokenizer_class"), str):
        blockers.append("tokenizer_config.json lacks a typed tokenizer_class")
    processor = next((value for path, value in values.items() if Path(path).name == "processor_config.json"), None)
    if isinstance(processor, dict) and not isinstance(processor.get("processor_class"), str):
        blockers.append("processor_config.json lacks a typed processor_class")
    generation = next((value for path, value in values.items() if Path(path).name == "generation_config.json"), None)
    if isinstance(generation, dict) and ("eos_token_id" not in generation or "pad_token_id" not in generation):
        blockers.append("generation_config.json lacks explicit special-token IDs")
    if isinstance(generation, dict):
        vocab_size = config.get("vocab_size") if isinstance(config, dict) else None
        for key in ("eos_token_id", "pad_token_id"):
            token_id = generation.get(key)
            if isinstance(token_id, list):
                valid_ids = all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in token_id)
            else:
                valid_ids = isinstance(token_id, int) and not isinstance(token_id, bool) and token_id >= 0
            if not valid_ids:
                blockers.append(f"generation_config.json {key} is not a non-negative integer ID")
            elif isinstance(vocab_size, int) and vocab_size > 0:
                ids = token_id if isinstance(token_id, list) else [token_id]
                if any(value >= vocab_size for value in ids):
                    blockers.append(f"generation_config.json {key} exceeds vocab_size")
    return {"status": "PARSED_TYPED_COMPONENT_SEMANTICS", "config": config if isinstance(config, dict) else None, "json_paths": sorted(values)}


def source_inventory(source: Path, blockers: list[str]) -> dict[str, Any]:
    try:
        actual = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        names = subprocess.run(["git", "-C", str(source), "ls-files", "-z"], check=True, capture_output=True).stdout.split(b"\0")
    except (OSError, subprocess.CalledProcessError) as error:
        blockers.append(f"source git inventory failed: {error}")
        return {"repository": SOURCE_REPOSITORY, "pinned_revision": SOURCE_REVISION, "resolved_revision": "", "files": []}
    if actual != SOURCE_REVISION:
        blockers.append(f"source revision {actual!r} != pinned {SOURCE_REVISION!r}")
    try:
        origin = subprocess.run(["git", "-C", str(source), "remote", "get-url", "origin"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        origin = ""
        blockers.append(f"source origin unavailable: {error}")
    expected_origin = SOURCE_REPOSITORY
    if origin.removesuffix("/") != expected_origin.removesuffix("/"):
        blockers.append(f"source origin {origin!r} != pinned {expected_origin!r}")
    files = [source / os.fsdecode(name) for name in names if name and (source / os.fsdecode(name)).is_file() and not (source / os.fsdecode(name)).is_symlink()]
    if git_status := subprocess.run(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"], check=True, capture_output=True, text=True).stdout.strip():
        blockers.append(f"official source checkout is dirty: {git_status}")
    roles: dict[str, dict[str, Any]] = {}
    if set(SOURCE_ROLE_BLOBS) != set(SOURCE_ROLE_PATHS):
        blockers.append("fixed source role Git blob table is incomplete or has extra roles")
    for relative in SOURCE_ROLE_PATHS:
        path = source / relative
        if not path.is_file() or path.is_symlink():
            blockers.append(f"official source required role missing/non-regular: {relative}")
            continue
        actual_blob = git_blob_sha1(path)
        expected_blob = SOURCE_ROLE_BLOBS.get(relative)
        if expected_blob is None:
            blockers.append(f"fixed Git blob table is unavailable for source role: {relative}")
        elif actual_blob != expected_blob:
            blockers.append(f"source role Git blob mismatch: {relative}")
        roles[relative] = {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path), "git_blob_sha1": actual_blob, "expected_git_blob_sha1": expected_blob}
    if set(roles) != set(SOURCE_ROLE_PATHS):
        blockers.append("official source role set is incomplete")
    gitlink = subprocess.run(["git", "-C", str(source), "ls-files", "-s", "third_party/Matcha-TTS"], check=True, capture_output=True, text=True).stdout.strip().split()
    if len(gitlink) < 4 or gitlink[0] != "160000" or gitlink[2] != "0" or gitlink[3] != "third_party/Matcha-TTS":
        blockers.append("Matcha gitlink index entry is missing or malformed")
    matcha = source / "third_party/Matcha-TTS"
    if not matcha.is_dir():
        blockers.append("Matcha submodule checkout is missing")
    else:
        matcha_head = subprocess.run(["git", "-C", str(matcha), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        matcha_origin = subprocess.run(["git", "-C", str(matcha), "remote", "get-url", "origin"], check=False, capture_output=True, text=True).stdout.strip()
        if MATCHA_REVISION is None:
            blockers.append("fixed Matcha gitlink revision table is unavailable")
        elif len(gitlink) < 2 or matcha_head != MATCHA_REVISION or gitlink[1] != MATCHA_REVISION:
            blockers.append("Matcha gitlink/checkout revision is not authenticated")
        if not matcha_origin:
            blockers.append("Matcha submodule origin is unavailable")
        if subprocess.run(["git", "-C", str(matcha), "status", "--porcelain", "--untracked-files=all"], check=False, capture_output=True, text=True).stdout.strip():
            blockers.append("Matcha submodule checkout is dirty")
    source_license_path = source / "LICENSE"
    if not source_license_path.is_file():
        blockers.append("official source LICENSE is missing")
        source_license = {"status": "UNKNOWN"}
    else:
        license_text = source_license_path.read_text(encoding="utf-8", errors="replace")
        source_license = {**identity(source_license_path, source), "status": "DECLARATION_REQUIRES_SEPARATE_AUDIT", "declared_components": {"custom_code": "UNAUTHENTICATED", "matcha": "UNAUTHENTICATED", "cosy24k_vocoder": "UNAUTHENTICATED", "qwen": "UNAUTHENTICATED", "whisper": "UNAUTHENTICATED", "dataset": "UNAUTHENTICATED"}}
        blockers.append("component license inheritance is not inferred from the root LICENSE")
    return {"repository": SOURCE_REPOSITORY, "pinned_revision": SOURCE_REVISION, "resolved_revision": actual, "origin": origin, "clean": not bool(git_status), "files": [identity(path, source) for path in sorted(files)], "role_files": roles, "role_blob_table": SOURCE_ROLE_BLOBS, "role_blob_table_status": "AUTHENTICATED" if set(SOURCE_ROLE_BLOBS) == set(SOURCE_ROLE_PATHS) else "BLOCKED_UNAVAILABLE", "gitlinks": [identity(path, source) for path in files if path.name == ".gitmodules"], "matcha_revision": MATCHA_REVISION, "license": source_license}


def validate_tensor_shard_map(weight_map: dict[str, str], packets: list[dict[str, Any]], blockers: list[str]) -> set[str]:
    seen: set[str] = set()
    for packet in packets:
        actual_shard = Path(packet["path"]).name
        for tensor in packet.get("tensors", []):
            if tensor["name"] in seen:
                blockers.append(f"duplicate tensor name across shards: {tensor['name']}")
            indexed_shard = weight_map.get(tensor["name"])
            if indexed_shard != actual_shard:
                blockers.append(f"weight_map shard mismatch for tensor {tensor['name']}: index={indexed_shard!r} actual={actual_shard!r}")
            seen.add(tensor["name"])
    return seen


def inspect(snapshot: Path, source: Path, output: Path, tree: Path) -> int:
    blockers: list[str] = []
    files = regular_files(snapshot)
    tree_packet = server_tree(snapshot, tree, blockers)
    indexes = [path for path in files if path.name == "model.safetensors.index.json"]
    if len(indexes) != 1:
        blockers.append("exactly one model.safetensors.index.json is required")
        index_value: dict[str, Any] = {}
    else:
        index_value = json.loads(indexes[0].read_text(encoding="utf-8"), object_pairs_hook=strict_pairs)
    weight_map = index_value.get("weight_map", {}) if isinstance(index_value, dict) else {}
    if not isinstance(weight_map, dict) or not weight_map:
        blockers.append("index weight_map must be a non-empty object")
        weight_map = {}
    valid_weight_map: dict[str, str] = {}
    for key, value in weight_map.items():
        if not isinstance(key, str) or not isinstance(value, str):
            blockers.append("index weight_map keys and values must be strings")
            continue
        try:
            valid_weight_map[key] = strict_shard_name(value)
        except ValueError as error:
            blockers.append(f"invalid index shard path: {error}")
    weight_map = valid_weight_map
    shard_names = sorted(set(weight_map.values()))
    if len(shard_names) != COMPONENT_COUNT:
        blockers.append(f"expected exactly {COMPONENT_COUNT} indexed safetensors shards, found {len(shard_names)}")
    shard_paths = [snapshot / name for name in shard_names]
    if any(not path.is_file() for path in shard_paths):
        blockers.append("weight_map references missing shard")
    actual_shards = sorted(path.relative_to(snapshot).as_posix() for path in files if path.suffix == ".safetensors" and path.name != "model.safetensors.index.json")
    if sorted(shard_names) != actual_shards:
        blockers.append(f"index shard orphan/missing mismatch: indexed={shard_names!r} actual={actual_shards!r}")
    tensor_packets = [safetensors_header(path, snapshot, blockers) for path in shard_paths if path.is_file()]
    seen = validate_tensor_shard_map(weight_map, tensor_packets, blockers)
    if set(weight_map) != seen:
        blockers.append("weight_map keys do not exactly match tensor names")
    json_packets = [json_evidence(path, snapshot, blockers) for path in files if path.suffix == ".json"]
    json_semantics = validate_json_semantics(snapshot, json_packets, blockers)
    custom_roles = {name: [identity(path, snapshot) for path in files if path.relative_to(snapshot).as_posix() == name] for name in CUSTOM_ROLE_FILES}
    model_identity_by_path = {row["path"]: row for row in tree_packet.get("files", [])}
    model_files = []
    for path in files:
        row = model_identity_by_path.get(path.relative_to(snapshot).as_posix())
        if row is None:
            model_files.append(identity(path, snapshot))
        else:
            model_files.append({**identity(path, snapshot), "git_blob_sha1": row["git_blob_sha1"], "lfs_pointer_git_blob_sha1": row["lfs_pointer_git_blob_sha1"], "lfs_sha256": row["lfs_sha256"], "payload_sha256": row["payload_sha256"], "payload_bytes": row["payload_bytes"]})
    for name, matches in custom_roles.items():
        if len(matches) != 1:
            blockers.append(f"HF custom role file cardinality mismatch: {name} ({len(matches)})")
    readme = snapshot / "README.md"
    if not readme.is_file():
        blockers.append("HF README weight card is missing")
        weight_license: dict[str, Any] = {"status": "UNKNOWN"}
    else:
        weight_license = license_declaration(readme, snapshot, "apache-2.0")
        if weight_license["status"] != "AUTHENTICATED_DECLARATION":
            blockers.append("HF weight license declaration is not canonical apache-2.0")
    license_files = [identity(path, snapshot) for path in files if "license" in path.name.lower() or path.name.lower() in {"notice", "copying", "readme.md"}]
    source = source_inventory(source, blockers)
    source_license = source.get("license", {"status": "UNKNOWN"})
    if source_license.get("declared_components"):
        blockers.append("source LICENSE declaration to HF Matcha bundle files is unauthenticated")
    blockers.extend(["native Baichuan composition/runtime is not implemented", "dependency licenses are unreviewed", "dataset/training provenance is unauthenticated", "HF custom/CosyVoice/Matcha/Whisper/Qwen/vocoder licenses require separate audit"])
    payload = {"format": FORMAT, "status": "BLOCKED", "inspection_status": "INSPECTION_ONLY", "evidence_stage": "INSPECTION_ONLY", "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "native_status": "BLOCKED", "cpu_status": "UNSUPPORTED", "metal_status": "BLOCKED_BY_CPU", "parity_status": "NOT_RUN", "publication": "NO_UPLOAD", "model": {"repository": HF_REPOSITORY, "revision": HF_REVISION, "resolved_revision": tree_packet.get("resolved_revision"), "server_tree": tree_packet, "files": model_files, "index": json_evidence(indexes[0], snapshot, blockers) if indexes else None, "shards": tensor_packets, "json": json_packets, "json_semantics": json_semantics, "custom_role_files": custom_roles, "license_evidence": {"weight_card": weight_license, "files": license_files}}, "official_source": source, "license_evidence": {"weight_declaration": weight_license, "source": source_license, "custom_code": "UNREVIEWED_BLOCKER", "components": "UNREVIEWED_BLOCKER", "dependencies": "UNREVIEWED_BLOCKER", "datasets": "UNAUTHENTICATED_BLOCKER"}, "blockers": sorted(set(blockers))}
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 2


def write_blocked(output: Path, error: Exception, tree: Path | None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    packet = None
    if tree is not None and tree.is_file():
        packet = {"path": str(tree), "bytes": tree.stat().st_size, "sha256": sha256(tree)}
    payload = {"format": FORMAT, "status": "BLOCKED", "inspection_status": "INSPECTION_ONLY", "evidence_stage": "INSPECTION_ONLY", "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "native_status": "BLOCKED", "cpu_status": "UNSUPPORTED", "metal_status": "BLOCKED_BY_CPU", "parity_status": "NOT_RUN", "publication": "NO_UPLOAD", "model": {"repository": HF_REPOSITORY, "revision": HF_REVISION}, "server_tree_packet": packet, "error_type": type(error).__name__, "error": str(error), "blockers": [str(error), "native Baichuan composition/runtime is not implemented", "dependency licenses are unreviewed", "dataset/training provenance is unauthenticated"]}
    (output / "manifest.json").write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def self_test() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    assert len(HF_REVISION) == len(SOURCE_REVISION) == 40
    assert "safetensors_header" in source
    audit = lock_dependency_audit()
    assert audit["status"] == DEPENDENCY_LICENSE_AUDIT_STATUS == "BLOCKED_UNREVIEWED_TRANSITIVE"
    assert audit["package_count"] == REFERENCE_PACKAGE_COUNT == len(DEPENDENCY_LICENSE_CONCLUSIONS)
    assert audit["package_rows_sha256"] == REFERENCE_PACKAGE_ROWS_SHA256
    assert audit["license_conclusions"]["certifi==2026.7.22"]["license"].startswith("MPL-2.0")
    assert audit["license_conclusions"]["typing-extensions==4.16.0"]["license"].startswith("PSF-2.0")
    with REFERENCE_LOCK.open("rb") as stream:
        lock_document = tomllib.load(stream)
    lock_packages = lock_document["package"]
    assert resolution_markers_sha256(sorted(lock_document.get("resolution-markers", []))) == REFERENCE_RESOLUTION_MARKERS_SHA256
    assert resolution_markers_sha256(["sys_platform == 'win32'"]) != REFERENCE_RESOLUTION_MARKERS_SHA256
    for label, altered in (
        ("deleted", lock_packages[:-1]),
        ("unknown", lock_packages + [{"name": "unknown-package", "version": "0.0.0", "source": {"registry": "https://example.invalid/simple"}}]),
        ("tampered-source", [{**package, "source": {"registry": "https://example.invalid/simple"}} if package.get("name") == "huggingface-hub" else package for package in lock_packages]),
        ("tampered-marker", [{**package, "dependencies": [{**dependency, "marker": "sys_platform == 'win32'"} if dependency.get("name") == "tqdm" else dependency for dependency in package.get("dependencies", [])]} if package.get("name") == "huggingface-hub" else package for package in lock_packages]),
        ("tampered-resolution-marker", [{**package, "resolution-markers": ["sys_platform == 'win32'"]} if package.get("name") == "huggingface-hub" else package for package in lock_packages]),
    ):
        candidate = canonical_lock_rows(copy.deepcopy(altered))
        assert len(candidate) != REFERENCE_PACKAGE_COUNT or lock_rows_sha256(candidate) != REFERENCE_PACKAGE_ROWS_SHA256, label
    with tempfile.TemporaryDirectory(prefix="baichuan-inspect-") as directory:
        root = Path(directory)
        shard = root / "a.safetensors"
        header = json.dumps({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        shard.write_bytes(len(header).to_bytes(8, "little") + header + b"\0\0\0\0")
        blockers: list[str] = []
        packet = safetensors_header(shard, root, blockers)
        assert packet["tensor_count"] == 1 and any("finiteness" in blocker for blocker in blockers)
        assert strict_shard_name("model-00001-of-00005.safetensors")
        for unsafe_name in ("../model-00001-of-00005.safetensors", "/tmp/model-00001-of-00005.safetensors", "model-00001-of-00005\\x.safetensors"):
            try:
                strict_shard_name(unsafe_name)
            except ValueError:
                pass
            else:
                raise AssertionError("unsafe shard path was accepted")
        bad_shard = root / "bad.safetensors"
        bad_header = json.dumps({"x": {"dtype": "F32", "shape": [True], "data_offsets": [0, 4]}}).encode()
        bad_shard.write_bytes(len(bad_header).to_bytes(8, "little") + bad_header + b"\0\0\0\0")
        bad_blockers: list[str] = []
        safetensors_header(bad_shard, root, bad_blockers)
        assert any("shape axes" in blocker for blocker in bad_blockers)
        huge_header = root / "huge.safetensors"
        huge_header.write_bytes((MAX_HEADER_BYTES + 1).to_bytes(8, "little") + b"{}")
        huge_blockers: list[str] = []
        safetensors_header(huge_header, root, huge_blockers)
        assert any("header length" in blocker for blocker in huge_blockers)
        snapshot = root / "snapshot"
        snapshot.mkdir()
        payload = snapshot / "payload.bin"
        payload.write_bytes(b"authenticated payload")
        valid_tree = root / "tree.json"
        payload_digest = sha256(payload)
        pointer_digest = git_blob_sha1_bytes(lfs_pointer_bytes(payload_digest, payload.stat().st_size))
        assert pointer_digest != git_blob_sha1(payload)
        valid_row = {"path": "payload.bin", "type": "file", "size": payload.stat().st_size, "git_blob_sha1": None, "lfs_pointer_git_blob_sha1": pointer_digest, "lfs_sha256": payload_digest}
        valid_tree.write_text(json.dumps({"repository": HF_REPOSITORY, "revision": HF_REVISION, "resolved_revision": HF_REVISION, "files": [valid_row]}), encoding="utf-8")
        tree_blockers: list[str] = []
        assert server_tree(snapshot, valid_tree, tree_blockers)["status"] == "MATCHED" and not tree_blockers
        spoof = dict(valid_row)
        spoof["lfs_sha256"] = "0" * 64
        spoof_tree = root / "spoof-tree.json"
        spoof_tree.write_text(json.dumps({"repository": HF_REPOSITORY, "revision": HF_REVISION, "resolved_revision": "0" * 40, "files": [spoof]}), encoding="utf-8")
        spoof_blockers: list[str] = []
        assert server_tree(snapshot, spoof_tree, spoof_blockers)["status"] != "MATCHED"
        assert any("payload SHA-256 mismatch" in blocker for blocker in spoof_blockers)
        pointer_spoof = dict(valid_row)
        pointer_spoof["lfs_pointer_git_blob_sha1"] = "0" * 40
        pointer_tree = root / "pointer-spoof-tree.json"
        pointer_tree.write_text(json.dumps({"repository": HF_REPOSITORY, "revision": HF_REVISION, "resolved_revision": HF_REVISION, "files": [pointer_spoof]}), encoding="utf-8")
        pointer_blockers: list[str] = []
        assert server_tree(snapshot, pointer_tree, pointer_blockers)["status"] != "MATCHED"
        assert any("pointer Git blob SHA-1 mismatch" in blocker for blocker in pointer_blockers)
        size_spoof = dict(valid_row)
        size_spoof["size"] += 1
        size_tree = root / "size-spoof-tree.json"
        size_tree.write_text(json.dumps({"repository": HF_REPOSITORY, "revision": HF_REVISION, "resolved_revision": HF_REVISION, "files": [size_spoof]}), encoding="utf-8")
        size_blockers: list[str] = []
        assert server_tree(snapshot, size_tree, size_blockers)["status"] != "MATCHED"
        assert any("payload size mismatch" in blocker for blocker in size_blockers)
        duplicate_tree = root / "duplicate-tree.json"
        duplicate_tree.write_text('{"repository":"%s","repository":"spoof","revision":"%s","resolved_revision":"%s","files":[]}' % (HF_REPOSITORY, HF_REVISION, HF_REVISION), encoding="utf-8")
        duplicate_blockers: list[str] = []
        assert server_tree(snapshot, duplicate_tree, duplicate_blockers)["status"] == "BLOCKED_PACKET_PARSE"
        traversal_tree = root / "traversal-tree.json"
        traversal_tree.write_text(json.dumps({"repository": HF_REPOSITORY, "revision": HF_REVISION, "resolved_revision": HF_REVISION, "files": [{**valid_row, "path": "../payload.bin"}]}), encoding="utf-8")
        traversal_blockers: list[str] = []
        assert server_tree(snapshot, traversal_tree, traversal_blockers)["status"] != "MATCHED"
        semantic_blockers: list[str] = []
        (root / "config.json").write_text(json.dumps({"model_type": "baichuan", "architectures": ["Baichuan"], "hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2, "vocab_size": 16}), encoding="utf-8")
        assert validate_json_semantics(root, [{"path": "config.json"}], semantic_blockers)["status"] == "PARSED_TYPED_COMPONENT_SEMANTICS"
        assert any("tokenizer_config.json" in blocker for blocker in semantic_blockers)
        drift_blockers: list[str] = []
        (root / "config.json").write_text(json.dumps({"model_type": "baichuan", "architectures": [], "hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 3, "vocab_size": 16}), encoding="utf-8")
        validate_json_semantics(root, [{"path": "config.json"}], drift_blockers)
        assert any("architecture semantics" in blocker or "divisible" in blocker for blocker in drift_blockers)
        mapping_blockers: list[str] = []
        seen = validate_tensor_shard_map(
            {"tensor": "model-00002-of-00005.safetensors"},
            [{"path": "model-00001-of-00005.safetensors", "tensors": [{"name": "tensor"}]}],
            mapping_blockers,
        )
        assert seen == {"tensor"} and any("shard mismatch" in blocker for blocker in mapping_blockers)
        bad = root / "bad.json"
        bad.write_text("{", encoding="utf-8")
        json_evidence(bad, root, blockers)
        assert any("JSON parse failed" in blocker for blocker in blockers)
    print("baichuan_audio_instruct_inspect self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dependency-gate", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--server-tree", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if args.dependency_gate or any(value is not None for value in (args.snapshot, args.source, args.server_tree, args.output)):
            parser.error("--self-test accepts no other arguments")
        self_test()
        return 0
    if args.dependency_gate:
        if any(value is not None for value in (args.snapshot, args.source, args.server_tree, args.output)):
            parser.error("--dependency-gate accepts no inspection arguments")
        try:
            require_dependency_gate()
        except RuntimeError as error:
            print(f"Baichuan dependency/license gate BLOCKED: {error}", file=sys.stderr)
            return 2
        print("Baichuan dependency/license gate: AUDITED_ALLOW")
        return 0
    if any(value is None for value in (args.snapshot, args.source, args.server_tree, args.output)):
        parser.error("normal runs require --snapshot --source --server-tree --output")
    try:
        return inspect(args.snapshot, args.source, args.output, args.server_tree)
    except Exception as error:  # noqa: BLE001 - preserve fixed identity/status blocker manifest
        write_blocked(args.output, error, args.server_tree)
        print(f"Baichuan inspection BLOCKED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
