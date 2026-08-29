#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Inspect the pinned VieNeu-TTS bundle without producing a runtime artifact.

This is intentionally an evidence collector, not a VieNeu implementation.  It
does not assume tensor names, dimensions, ONNX graph axes, or a tokenizer
layout.  All source and model files are identified by SHA-256, and the only
tensor/graph readers used here are safetensors.safe_open and offline ONNX
parsing.  No ONNX Runtime execution, pickle loading, or conversion is done.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


MODEL_REPOSITORY = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
MODEL_REVISION = "2da0efab622a1722125991736524f080b751ef5b"
SOURCE_REPOSITORY = "pnnbao97/VieNeu-TTS"
SOURCE_URL = "https://github.com/pnnbao97/VieNeu-TTS.git"
SOURCE_TAG_OBJECT = "1bc18895b8c6c6f8c927272d36c9b0befc127029"
SOURCE_TAG_NAME = "v3.0.0"
SOURCE_PEELED_COMMIT = "28392eee571db0da31632882ac7226faa2d09d5d"
# Retain the established name for the immutable tag-object identity.
SOURCE_REVISION = SOURCE_TAG_OBJECT
MOSS_REPOSITORY = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
MOSS_REVISION = "6aa02b01e445cc585582cf0ba480bc3ea6c8dd68"
FORMAT = "vokra-vieneu-v3-turbo-inspection-v2"

# Fixed source roles selected from the pinned engine import boundary.  Values
# are GitHub API tree object IDs for SOURCE_REVISION, not hashes computed from
# a local self-authored packet.
SOURCE_ROLE_BLOBS: dict[str, str] = {
    "LICENSE": "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64",
    "pyproject.toml": "1fe2997dc8ef8c9aa2a379d2a9ba65295afcea4",
    "uv.lock": "276a29b03a351367657225f0119c97a10c98b160",
    "src/vieneu/_v3_turbo_engine/__init__.py": "4a9cfd1e899bd334630ab49098df533b3b8ba5f9",
    "src/vieneu/_v3_turbo_engine/configuration_v3_turbo.py": "6ce05ecb686426ae73bab18e2b5bc2a0ac46eaaf",
    "src/vieneu/_v3_turbo_engine/hub_load_v3_turbo.py": "2c2e33d2c9cbe8acf9609ebefc8cb036e01bc924",
    "src/vieneu/_v3_turbo_engine/inference_v3_turbo.py": "10ab7a744b04c11eab6d04a6f70a958896864d63",
    "src/vieneu/_v3_turbo_engine/modeling_v3_turbo.py": "220fd8a7d4acb2b444250e19c2086cdb30ed8085",
    "src/vieneu/_v3_turbo_engine/onnx_runtime_lite.py": "913274b10c4c0952cb8cda23521351e59b18db63",
    "src/vieneu/_v3_turbo_engine/prompt_v3_turbo.py": "ae8b28f6076ab7517addaea7f17592427e859ea9",
}
SOURCE_ROLE_PATHS = {role: role for role in SOURCE_ROLE_BLOBS}
OPTIONAL_SOURCE_ROLES = {
    "speaker": "src/vieneu/_v3_turbo_engine/speaker.py",
    "onnx_denoiser": "src/vieneu/_v3_turbo_engine/onnx_denoiser.py",
}
SOURCE_ROLE_BLOBS_UNREVIEWED_BLOCKER = "SOURCE_ROLE_BLOBS_UNREVIEWED_BLOCKER"
SOURCE_LICENSE_UNREVIEWED_BLOCKER = "SOURCE_LICENSE_UNREVIEWED_BLOCKER"
MODEL_LICENSE_DECLARATION_UNREVIEWED_BLOCKER = "MODEL_LICENSE_DECLARATION_UNREVIEWED_BLOCKER"
MODEL_LICENSE_ABSENT_BLOCKER = "MODEL_LICENSE_ABSENT_BLOCKER"
DEPENDENCY_LICENSE_UNREVIEWED_BLOCKER = "DEPENDENCY_LICENSE_UNREVIEWED_BLOCKER"
TOPOLOGY_UNVERIFIED_BLOCKER = "TOPOLOGY_CONTRACT_UNVERIFIED_BLOCKER"
RUNTIME_BLOCKER = "NATIVE_RUNTIME_AND_NUMERICAL_PARITY_NOT_IMPLEMENTED"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_load_unique(path: Path) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)


def git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(path: Path, root: Path) -> str:
    """Return a normalized relative path, rejecting traversal and symlinks."""
    if path.is_symlink():
        raise ValueError(f"symlink is not an authenticated bundle member: {path}")
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"path escapes bundle root: {path}") from error
    if not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
        raise ValueError(f"unsafe bundle path: {path}")
    return relative.as_posix()


def files_under(root: Path) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    blockers: list[str] = []
    if not root.is_dir():
        return files, [f"missing bundle directory: {root}"]
    for path in sorted(root.rglob("*")):
        try:
            relative = path.relative_to(root)
        except ValueError:
            blockers.append(f"bundle member is outside root: {path}")
            continue
        # snapshot_download(local_dir=...) creates exactly this root transport
        # metadata subtree.  It is excluded from model identity, while every
        # other cache (including nested .cache) is an authentication failure.
        if relative.parts[:2] == (".cache", "huggingface"):
            continue
        if any(part in {".git", ".cache"} for part in relative.parts):
            blockers.append(f"unauthenticated metadata path: {path}")
            continue
        if path.is_dir():
            continue
        try:
            safe_relative(path, root)
        except ValueError as error:
            blockers.append(str(error))
            continue
        if not path.is_file():
            blockers.append(f"bundle member is not a regular file: {path}")
            continue
        files.append(path)
    if not files:
        blockers.append(f"bundle contains no regular files: {root}")
    return files, blockers


def file_identity(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": safe_relative(path, root),
        "size": path.stat().st_size,
        "sha256": sha256(path),
    }


def license_evidence(root: Path, files: Iterable[Path]) -> list[dict[str, Any]]:
    """Parse declarations; a filename alone is never license evidence."""
    evidence: list[dict[str, Any]] = []
    for path in files:
        lowered = path.name.lower()
        if not ("license" in lowered or lowered in {"notice", "copying", "readme.md"}):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            evidence.append({**file_identity(path, root), "status": "READ_ERROR", "error": str(error)})
            continue
        front_matter = ""
        if text.startswith("---"):
            parts = text.split("---", 2)
            front_matter = parts[1] if len(parts) == 3 else ""
        declaration_text = front_matter
        if lowered != "readme.md":
            declaration_text += "\n" + text
        declarations = re.findall(
            r"(?im)(?:spdx-license-identifier\s*:\s*|^\s*license\s*:\s*|^\s*license\s*=\s*)([^\n#]+)",
            declaration_text,
        )
        if lowered != "readme.md":
            declarations.extend(
                match.strip()
                for match in re.findall(
                    r"(?i)\b(?:Apache License(?:,? Version)?\s*[0-9.]*|MIT License|BSD(?:-\d-Clause)?|CC-BY[^\s]*)\b",
                    text,
                )
            )
        declarations = [item.strip().strip('"\'') for item in declarations if item.strip()]
        evidence.append(
            {
                **file_identity(path, root),
                "status": "DECLARED_UNVERIFIED" if declarations else "UNKNOWN",
                "spdx_identifiers": re.findall(
                    r"(?i)spdx-license-identifier\s*:\s*([a-z0-9.+-]+)", text
                ),
                "declared_license": declarations,
            }
        )
    return evidence


def has_license_declaration(evidence: Iterable[dict[str, Any]]) -> bool:
    return any(item.get("status") == "DECLARED_UNVERIFIED" for item in evidence)


def validate_external_location(location: str) -> Path:
    candidate = Path(location)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"external data path traversal: {location}")
    return candidate


def valid_external_range(offset: int, length: int | None, file_size: int) -> bool:
    """Validate an ONNX external-data byte range without unchecked addition."""
    if offset < 0 or offset > file_size:
        return False
    return length is None or length >= 0 and length <= file_size - offset


def tree_evidence(
    root: Path, tree_path: Path | None, repository: str, revision: str, blockers: list[str]
) -> dict[str, Any]:
    """Compare local bytes with a complete, API-generated HF server packet."""
    if tree_path is None:
        blockers.append(f"missing server tree evidence for {repository}")
        return {"status": "BLOCKED_MISSING_SERVER_TREE", "files": []}
    try:
        remote = json_load_unique(tree_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        blockers.append(f"server tree parse failed for {repository}: {error}")
        return {"status": "BLOCKED_SERVER_TREE_PARSE", "files": []}
    if not isinstance(remote, dict) or set(remote) != {
        "repository", "requested_revision", "resolved_revision", "files"
    }:
        blockers.append(f"server tree schema mismatch for {repository}")
        return {"status": "BLOCKED_SERVER_TREE_SCHEMA", "files": []}
    if (
        remote["repository"] != repository
        or remote["requested_revision"] != revision
        or remote["resolved_revision"] != revision
    ):
        blockers.append(f"server tree identity mismatch for {repository}")
    rows = remote["files"]
    if not isinstance(rows, list):
        blockers.append(f"server tree files are not a list for {repository}")
        return {"status": "BLOCKED_SERVER_TREE_SCHEMA", "files": []}
    row_paths = [row.get("path") if isinstance(row, dict) else None for row in rows]
    if not rows or any(not isinstance(path, str) for path in row_paths) or row_paths != sorted(row_paths):
        blockers.append(f"server tree file ordering/emptiness mismatch for {repository}")
    remote_files: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("type") != "file":
            blockers.append(f"server tree has unknown/non-file entry for {repository}")
            continue
        path = row.get("path")
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or "\\" in path
            or "\x00" in path
            or ".." in Path(path).parts
            or path in remote_files
        ):
            blockers.append(f"server tree has unsafe/duplicate path for {repository}")
            continue
        keys = set(row)
        base = {"path", "type", "size", "git_blob_sha1"}
        lfs = base | {"lfs_sha256", "lfs_size", "lfs_pointer_sha1"}
        if keys not in (base, lfs):
            blockers.append(f"server tree row schema mismatch: {repository}:{path}")
            continue
        if (
            isinstance(row["size"], bool)
            or not isinstance(row["size"], int)
            or row["size"] < 0
            or not isinstance(row["git_blob_sha1"], str)
            or len(row["git_blob_sha1"]) != 40
            or any(char not in "0123456789abcdef" for char in row["git_blob_sha1"])
        ):
            blockers.append(f"server tree identity malformed: {repository}:{path}")
            continue
        if keys == lfs and (
            not isinstance(row["lfs_sha256"], str)
            or len(row["lfs_sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in row["lfs_sha256"])
            or isinstance(row["lfs_size"], bool)
            or not isinstance(row["lfs_size"], int)
            or row["lfs_size"] != row["size"]
            or not isinstance(row["lfs_pointer_sha1"], str)
            or len(row["lfs_pointer_sha1"]) != 40
            or any(char not in "0123456789abcdef" for char in row["lfs_pointer_sha1"])
        ):
            blockers.append(f"server tree LFS identity malformed: {repository}:{path}")
            continue
        remote_files[path] = row
    local_files, local_blockers = files_under(root)
    blockers.extend(local_blockers)
    local_by_path = {safe_relative(path, root): path for path in local_files}
    missing = sorted(set(remote_files) - set(local_by_path))
    extra = sorted(set(local_by_path) - set(remote_files))
    if missing or extra:
        blockers.append(
            f"HF server/local tree mismatch for {repository}: missing={missing!r} extra={extra!r}"
        )
    identity_mismatches: list[str] = []
    for path, row in remote_files.items():
        local = local_by_path.get(path)
        if local is None or local.stat().st_size != row["size"]:
            identity_mismatches.append(path)
            continue
        if set(row) == {"path", "type", "size", "git_blob_sha1"}:
            if git_blob_sha1(local) != row["git_blob_sha1"]:
                identity_mismatches.append(path)
            continue
        payload_sha = sha256(local)
        pointer = (
            "version https://git-lfs.github.com/spec/v1\n"
            f"oid sha256:{row['lfs_sha256']}\nsize {row['lfs_size']}\n"
        ).encode("ascii")
        pointer_sha = hashlib.sha1(f"blob {len(pointer)}\0".encode("ascii") + pointer).hexdigest()
        if (
            payload_sha != row["lfs_sha256"]
            or pointer_sha != row["lfs_pointer_sha1"]
            or pointer_sha != row["git_blob_sha1"]
        ):
            identity_mismatches.append(path)
    if identity_mismatches:
        blockers.append(f"HF server/local identity mismatch for {repository}: {identity_mismatches!r}")
    return {
        "status": "MATCHED" if not missing and not extra and not identity_mismatches else "MISMATCH",
        "repository": repository,
        "requested_revision": revision,
        "resolved_revision": revision,
        "server_tree_sha256": sha256(tree_path),
        "server_files": sorted(remote_files),
        "local_files": sorted(local_by_path),
        "missing_local": missing,
        "unexpected_local": extra,
        "identity_mismatches": identity_mismatches,
    }


def token_id_fields(value: Any, prefix: str = "") -> dict[str, Any]:
    fields: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if "token" in str(key).lower() or str(key).lower().endswith("_id"):
                if isinstance(child, (str, int, float, bool)) or child is None:
                    fields[child_prefix] = child
                elif isinstance(child, list) and all(
                    isinstance(item, (str, int, float, bool)) or item is None for item in child
                ):
                    fields[child_prefix] = child
            fields.update(token_id_fields(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            fields.update(token_id_fields(child, f"{prefix}[{index}]"))
    return fields


def topology_fields(value: Any, prefix: str = "") -> dict[str, Any]:
    """Extract topology-shaped config fields while retaining their raw values."""
    fields: dict[str, Any] = {}
    topology_markers = (
        "architect",
        "hidden",
        "layer",
        "head",
        "vocab",
        "position",
        "embed",
        "channel",
        "kernel",
        "stride",
        "sample_rate",
        "quant",
        "codebook",
    )
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if any(marker in str(key).lower() for marker in topology_markers):
                fields[child_prefix] = child
            fields.update(topology_fields(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            fields.update(topology_fields(child, f"{prefix}[{index}]"))
    return fields


def config_evidence(root: Path, all_files: list[Path], blockers: list[str]) -> dict[str, Any]:
    paths = sorted(path for path in all_files if path.name == "config.json")
    if not paths:
        blockers.append("model bundle has no config.json")
        return {"files": []}
    result: dict[str, Any] = {"files": []}
    for path in paths:
        try:
            value = json_load_unique(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            blockers.append(f"config parse failed for {path}: {error}")
            continue
        if not isinstance(value, dict):
            blockers.append(f"config is not a JSON object: {path}")
            continue
        result["files"].append(
            {
                **file_identity(path, root),
                "status": "UNVERIFIED_TOPOLOGY",
                "model_type": value.get("model_type"),
                "architectures": value.get("architectures"),
                "token_id_fields": token_id_fields(value),
                "topology_fields": topology_fields(value),
                "config": value,
            }
        )
    return result


def tokenizer_evidence(root: Path, all_files: list[Path], blockers: list[str]) -> dict[str, Any]:
    candidates = [
        path
        for path in all_files
        if any(marker in path.name.lower() for marker in ("tokenizer", "vocab", "special_tokens"))
    ]
    evidence: list[dict[str, Any]] = []
    for path in candidates:
        item = file_identity(path, root)
        if path.suffix.lower() == ".json":
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                blockers.append(f"tokenizer JSON parse failed for {path}: {error}")
                item["status"] = "BLOCKED_JSON_PARSE"
            else:
                item["status"] = "PARSED_UNVERIFIED_JSON"
                item["token_id_fields"] = token_id_fields(value)
                item["json"] = value
        elif path.suffix.lower() in {".txt", ".vocab"}:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError) as error:
                blockers.append(f"tokenizer text parse failed for {path}: {error}")
                item["status"] = "BLOCKED_TEXT_PARSE"
            else:
                item["status"] = "PARSED_UNVERIFIED_LINES"
                item["nonempty_line_count"] = sum(bool(line.strip()) for line in lines)
        else:
            item["status"] = "HASH_ONLY_BINARY_TOKENIZER_ASSET"
        evidence.append(item)
    if not evidence:
        blockers.append("model bundle has no identifiable tokenizer/vocabulary sidecar")
    return {"files": evidence}


def safetensors_evidence(
    root: Path, all_files: list[Path], blockers: list[str], *, required: bool = True
) -> dict[str, Any]:
    paths = [path for path in all_files if path.suffix == ".safetensors"]
    if not paths:
        if required:
            blockers.append("model bundle has no safetensors checkpoint")
        return {"files": []}
    try:
        from safetensors import safe_open
    except ImportError as error:
        blockers.append(f"safetensors safe_open unavailable: {error}")
        return {"files": []}
    evidence: list[dict[str, Any]] = []
    for path in paths:
        item: dict[str, Any] = file_identity(path, root)
        tensors: dict[str, Any] = {}
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for key in sorted(handle.keys()):
                    tensor = handle.get_tensor(key)
                    finite = bool(tensor.isfinite().all().item())
                    tensors[key] = {
                        "shape": [int(axis) for axis in tensor.shape],
                        "dtype": str(tensor.dtype),
                        "count": int(tensor.numel()),
                        "finite": finite,
                    }
                    if not finite:
                        blockers.append(f"non-finite safetensors tensor: {path}:{key}")
                    del tensor
        except Exception as error:  # noqa: BLE001 - preserve a loud evidence blocker
            item["status"] = "BLOCKED_SAFETENSORS_PARSE"
            item["error"] = f"{type(error).__name__}: {error}"
            blockers.append(f"safetensors parse failed for {path}: {error}")
            evidence.append(item)
            continue
        item["status"] = "SAFE_OPENED"
        item["resident_scope"] = "one_tensor_at_a_time; tensor released before next key"
        item["tensor_count"] = len(tensors)
        item["parameter_count"] = sum(entry["count"] for entry in tensors.values())
        item["tensor_manifest"] = tensors
        if not tensors:
            blockers.append(f"empty safetensors manifest: {path}")
        evidence.append(item)
    return {"files": evidence}


def onnx_shape(value_info: Any) -> list[Any]:
    shape = []
    tensor_type = value_info.type.tensor_type
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            shape.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            shape.append(str(dimension.dim_param))
        else:
            shape.append(None)
    return shape


def onnx_evidence(root: Path, all_files: list[Path], blockers: list[str]) -> dict[str, Any]:
    paths = [path for path in all_files if path.suffix == ".onnx"]
    if not paths:
        blockers.append("model bundle has no ONNX subgraph")
        return {"files": []}
    try:
        import onnx
    except ImportError as error:
        blockers.append(f"offline ONNX parser unavailable: {error}")
        return {"files": []}
    evidence: list[dict[str, Any]] = []
    for path in paths:
        item: dict[str, Any] = file_identity(path, root)
        try:
            graph = onnx.load(str(path), load_external_data=False)
            external_data_files: list[dict[str, Any]] = []
            for initializer in graph.graph.initializer:
                entries = {str(entry.key): str(entry.value) for entry in initializer.external_data}
                location = entries.get("location")
                if location is None:
                    if entries:
                        blockers.append(f"ONNX external data has no location: {path}:{initializer.name}")
                    continue
                try:
                    location_path = validate_external_location(location)
                except ValueError:
                    blockers.append(f"ONNX external data path traversal: {path}:{location}")
                    continue
                data_path = path.parent / location_path
                try:
                    relative = safe_relative(data_path, root)
                except ValueError as error:
                    blockers.append(f"ONNX external data is unsafe: {path}:{error}")
                    continue
                if not data_path.is_file():
                    blockers.append(f"ONNX external data file is missing: {data_path}")
                    continue
                try:
                    offset = int(entries.get("offset", "0"))
                    length = int(entries["length"]) if "length" in entries else None
                except ValueError as error:
                    blockers.append(f"ONNX external data range is malformed: {path}:{error}")
                    continue
                file_size = data_path.stat().st_size
                if not valid_external_range(offset, length, file_size):
                    blockers.append(f"ONNX external data range is outside file: {path}:{location}")
                    continue
                external_data_files.append(
                    {
                        "initializer": initializer.name,
                        "location": relative,
                        "offset": offset,
                        "length": length,
                        "file": file_identity(data_path, root),
                    }
                )
            model = {
                "ir_version": int(graph.ir_version),
                "opsets": [
                    {"domain": str(opset.domain), "version": int(opset.version)}
                    for opset in graph.opset_import
                ],
                "inputs": [
                    {"name": value.name, "shape": onnx_shape(value)}
                    for value in graph.graph.input
                ],
                "outputs": [
                    {"name": value.name, "shape": onnx_shape(value)}
                    for value in graph.graph.output
                ],
                "initializers": [
                    {
                        "name": initializer.name,
                        "shape": [int(axis) for axis in initializer.dims],
                        "dtype": onnx.TensorProto.DataType.Name(initializer.data_type),
                        "external_data": [
                            {"key": str(entry.key), "value": str(entry.value)}
                            for entry in initializer.external_data
                        ],
                    }
                    for initializer in graph.graph.initializer
                ],
                "external_data_files": external_data_files,
                "nodes": [
                    {"op_type": node.op_type, "domain": node.domain, "name": node.name}
                    for node in graph.graph.node
                ],
            }
            item["status"] = "PARSED_OFFLINE"
            item["graph"] = model
        except Exception as error:  # noqa: BLE001 - preserve a loud evidence blocker
            item["status"] = "BLOCKED_ONNX_PARSE"
            item["error"] = f"{type(error).__name__}: {error}"
            blockers.append(f"ONNX parse failed for {path}: {error}")
        evidence.append(item)
    return {"files": evidence}


def git_revision(root: Path, expected: str, label: str, blockers: list[str]) -> str:
    try:
        actual = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        blockers.append(f"{label} revision unavailable: {error}")
        return ""
    if actual != expected:
        blockers.append(f"{label} revision {actual!r} != pinned {expected!r}")
    return actual


def validate_source_tag_identity(
    tag_object: str, object_type: str, tag_content: str, head: str
) -> list[str]:
    """Validate both an annotated tag object and its checked-out commit."""

    blockers: list[str] = []
    if tag_object != SOURCE_TAG_OBJECT:
        blockers.append(f"VieNeu tag object {tag_object!r} != pinned {SOURCE_TAG_OBJECT!r}")
    if object_type != "tag":
        blockers.append(f"VieNeu source object type {object_type!r} != 'tag'")
    headers: dict[str, str] = {}
    for line in tag_content.splitlines():
        if not line:
            break
        key, separator, value = line.partition(" ")
        if separator and key in {"object", "type", "tag"}:
            if key in headers:
                blockers.append(f"VieNeu annotated tag has duplicate {key} header")
            headers[key] = value
    if headers.get("object") != SOURCE_PEELED_COMMIT:
        blockers.append(
            f"VieNeu tag target {headers.get('object')!r} != pinned {SOURCE_PEELED_COMMIT!r}"
        )
    if headers.get("type") != "commit":
        blockers.append(f"VieNeu tag target type {headers.get('type')!r} != 'commit'")
    if headers.get("tag") != SOURCE_TAG_NAME:
        blockers.append(f"VieNeu tag name {headers.get('tag')!r} != pinned {SOURCE_TAG_NAME!r}")
    if head != SOURCE_PEELED_COMMIT:
        blockers.append(f"VieNeu source HEAD {head!r} != peeled {SOURCE_PEELED_COMMIT!r}")
    return blockers


def git_tag_identity(root: Path, blockers: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "pinned_tag_object": SOURCE_TAG_OBJECT,
        "pinned_tag_name": SOURCE_TAG_NAME,
        "pinned_peeled_commit": SOURCE_PEELED_COMMIT,
        "resolved_tag_object": None,
        "resolved_tag_type": None,
        "resolved_peeled_commit": None,
        "resolved_revision": None,
        "tag_content": None,
    }
    try:
        tag_object = git_output(root, "rev-parse", f"{SOURCE_TAG_NAME}^{{tag}}").strip()
        tag_type = git_output(root, "cat-file", "-t", tag_object).strip()
        tag_content = git_output(root, "cat-file", "-p", tag_object)
        peeled_commit = git_output(root, "rev-parse", f"{SOURCE_TAG_NAME}^{{commit}}").strip()
        head = git_output(root, "rev-parse", "HEAD").strip()
    except (OSError, subprocess.CalledProcessError) as error:
        blockers.append(f"VieNeu annotated tag identity unavailable: {error}")
        return result
    result.update(
        {
            "resolved_tag_object": tag_object,
            "resolved_tag_type": tag_type,
            "resolved_peeled_commit": peeled_commit,
            "resolved_revision": head,
            "tag_content": tag_content,
        }
    )
    blockers.extend(validate_source_tag_identity(tag_object, tag_type, tag_content, head))
    if peeled_commit != SOURCE_PEELED_COMMIT:
        blockers.append(
            f"VieNeu annotated tag peeling {peeled_commit!r} != pinned {SOURCE_PEELED_COMMIT!r}"
        )
    return result


def git_output(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def tracked_files(root: Path, blockers: list[str]) -> list[Path]:
    try:
        output = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        blockers.append(f"source tracked-file inventory unavailable: {error}")
        return []
    result: list[Path] = []
    for encoded in output.split(b"\0"):
        if not encoded:
            continue
        path = root / os.fsdecode(encoded)
        try:
            safe_relative(path, root)
        except ValueError as error:
            blockers.append(str(error))
            continue
        if path.is_symlink() or not path.is_file():
            blockers.append(f"tracked source file is missing or non-regular: {path}")
            continue
        result.append(path)
    if not result:
        blockers.append(f"source checkout has no tracked regular files: {root}")
    return sorted(result)


def validate_fixed_roles(entries: Mapping[str, Mapping[str, str]]) -> None:
    for relative, expected in SOURCE_ROLE_BLOBS.items():
        actual = entries.get(relative)
        if actual is None or actual.get("mode") != "100644" or actual.get("git_blob_sha1") != expected:
            raise RuntimeError(f"VieNeu source fixed role blob/mode mismatch: {relative}")


def source_evidence(root: Path, blockers: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "repository": SOURCE_REPOSITORY,
        "url": SOURCE_URL,
        "pinned_revision": SOURCE_TAG_OBJECT,
        "pinned_tag_name": SOURCE_TAG_NAME,
        "pinned_peeled_commit": SOURCE_PEELED_COMMIT,
        "resolved_revision": None,
        "origin": None,
        "worktree_status": "UNKNOWN",
        "tracked_files": [],
        "license_files": [],
        "implementation_files": [],
        "dependency_boundary_files": [],
        "identity_status": "UNVERIFIED",
        "role_status": "UNREVIEWED",
        "license_status": "UNVERIFIED",
    }
    result.update(git_tag_identity(root, blockers))
    if not (root / ".git").exists():
        blockers.append("VieNeu source checkout lacks .git metadata")
    try:
        origin = git_output(root, "remote", "get-url", "origin").strip().removesuffix("/").removesuffix(".git")
    except (OSError, subprocess.CalledProcessError) as error:
        blockers.append(f"VieNeu source origin unavailable: {error}")
    else:
        result["origin"] = origin
        if origin != SOURCE_REPOSITORY:
            blockers.append(f"VieNeu source origin {origin!r} != pinned {SOURCE_REPOSITORY!r}")
    try:
        dirty = git_output(root, "status", "--porcelain", "--untracked-files=all")
    except (OSError, subprocess.CalledProcessError) as error:
        blockers.append(f"VieNeu source cleanliness unavailable: {error}")
    else:
        result["worktree_status"] = "DIRTY" if dirty else "CLEAN"
        if dirty:
            blockers.append("VieNeu source checkout is dirty")
    files = tracked_files(root, blockers)
    tracked: dict[str, dict[str, str]] = {}
    try:
        records = git_output(root, "ls-files", "-s", "-z").split("\0")
    except (OSError, subprocess.CalledProcessError) as error:
        blockers.append(f"VieNeu source index inventory unavailable: {error}")
        records = []
    for record in records:
        if not record:
            continue
        try:
            metadata, relative = record.split("\t", 1)
            mode, object_id, stage = metadata.split()
        except ValueError:
            blockers.append("VieNeu source index record is malformed")
            continue
        if relative in tracked or mode not in {"100644", "100755"} or stage != "0":
            blockers.append(f"VieNeu source tracked mode/stage mismatch: {relative}")
            continue
        path = root / relative
        if path.is_symlink() or not path.is_file():
            blockers.append(f"VieNeu source tracked path is not a regular file: {relative}")
            continue
        expected_mode = {"100644": 0o644, "100755": 0o755}[mode]
        if stat.S_IMODE(path.stat().st_mode) != expected_mode:
            blockers.append(f"VieNeu source filesystem mode mismatch: {relative}")
            continue
        try:
            head_object = git_output(root, "rev-parse", f"HEAD:{relative}").strip()
            working_object = git_blob_sha1(path)
        except (OSError, subprocess.CalledProcessError) as error:
            blockers.append(f"VieNeu source object unavailable for {relative}: {error}")
            continue
        if object_id != head_object or object_id != working_object:
            blockers.append(f"VieNeu source index/HEAD/working object mismatch: {relative}")
            continue
        tracked[relative] = {"mode": mode, "git_blob_sha1": object_id}
    if len(tracked) != len(files):
        blockers.append("VieNeu source tracked inventory does not match index")
    identity_ok = (
        result["resolved_tag_object"] == SOURCE_TAG_OBJECT
        and result["resolved_peeled_commit"] == SOURCE_PEELED_COMMIT
        and result["resolved_revision"] == SOURCE_PEELED_COMMIT
        and result["origin"] == SOURCE_REPOSITORY
        and result["worktree_status"] == "CLEAN"
        and len(tracked) == len(files)
    )
    result["identity_status"] = "AUTHENTICATED" if identity_ok else "UNVERIFIED"
    result["tracked_files"] = [file_identity(path, root) for path in files]
    result["required_source_roles"] = {}
    for role, relative in SOURCE_ROLE_PATHS.items():
        path = root / relative
        if relative not in tracked:
            blockers.append(f"VieNeu source fixed role is missing: {relative}")
            result["identity_status"] = "UNVERIFIED"
            result["required_source_roles"][role] = None
        else:
            result["required_source_roles"][role] = {
                "path": relative,
                **tracked[relative],
                "reviewed": relative in SOURCE_ROLE_BLOBS,
            }
    result["optional_source_roles"] = {
        role: (
            {"path": relative, **tracked[relative]}
            if relative in tracked
            else {"path": relative, "present": False}
        )
        for role, relative in OPTIONAL_SOURCE_ROLES.items()
    }
    if not SOURCE_ROLE_BLOBS:
        blockers.append(SOURCE_ROLE_BLOBS_UNREVIEWED_BLOCKER)
        result["role_status"] = "UNREVIEWED"
    else:
        result["role_status"] = "AUTHENTICATED"
        try:
            validate_fixed_roles(tracked)
        except RuntimeError as error:
            blockers.append(str(error))
            result["identity_status"] = "UNVERIFIED"
    for path in files:
        relative = safe_relative(path, root)
        result["license_files"].extend(license_evidence(root, [path]))
        if path.suffix in {".py", ".json", ".yaml", ".yml", ".toml"} or Path(relative).name.startswith(
            ("requirements", "setup", "pyproject")
        ):
            result["implementation_files"].append(file_identity(path, root))
            # Dependency roles require a fixed, reviewed path/blob table. Do
            # not promote arbitrary marker matches in tracked source text.
    license_path = root / "LICENSE"
    license_ok = False
    if "LICENSE" not in tracked:
        blockers.append("VieNeu source fixed LICENSE role is missing")
    elif tracked["LICENSE"]["git_blob_sha1"] != SOURCE_ROLE_BLOBS["LICENSE"]:
        blockers.append("VieNeu source LICENSE blob mismatch")
    elif not license_path.is_file() or license_path.is_symlink():
        blockers.append("VieNeu source LICENSE is not a regular file")
    else:
        license_text = license_path.read_text(encoding="utf-8", errors="replace").lower()
        license_ok = all(
            marker in license_text
            for marker in (
                "apache license",
                "terms and conditions for use",
                "licensed under the apache license, version 2.0",
                "as is",
                "without warranties or conditions of any kind",
            )
        )
        if not license_ok:
            blockers.append("VieNeu source LICENSE clauses are not the authenticated Apache-2.0 text")
    pyproject = root / "pyproject.toml"
    pyproject_ok = False
    if pyproject.is_file() and not pyproject.is_symlink():
        normalized = " ".join(pyproject.read_text(encoding="utf-8", errors="replace").lower().split())
        pyproject_ok = 'license = { file = "license" }' in normalized
    if not pyproject_ok:
        blockers.append("VieNeu source pyproject license metadata does not bind LICENSE")
    result["license_status"] = "AUTHENTICATED_APACHE_2" if license_ok and pyproject_ok else "UNVERIFIED"
    if result["license_status"] != "AUTHENTICATED_APACHE_2":
        blockers.append(SOURCE_LICENSE_UNREVIEWED_BLOCKER)
    if not result["implementation_files"]:
        blockers.append("VieNeu source has no implementation/config files")
    if not result["dependency_boundary_files"]:
        blockers.append("VieNeu source has no sea-g2p/MOSS/ONNX/SDK dependency evidence")
    result["dependency_license_status"] = DEPENDENCY_LICENSE_UNREVIEWED_BLOCKER
    blockers.append(DEPENDENCY_LICENSE_UNREVIEWED_BLOCKER)
    return result


def companion_evidence(root: Path, tree_path: Path | None, blockers: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "repository": MOSS_REPOSITORY,
        "pinned_revision": MOSS_REVISION,
        "local_snapshot": str(root),
    }
    files, file_blockers = files_under(root)
    blockers.extend(f"MOSS companion: {message}" for message in file_blockers)
    result["files"] = [file_identity(path, root) for path in files]
    result["license_files"] = license_evidence(root, files)
    result["safetensors"] = safetensors_evidence(root, files, blockers, required=False)
    result["server_tree"] = tree_evidence(root, tree_path, MOSS_REPOSITORY, MOSS_REVISION, blockers)
    if not files:
        blockers.append("MOSS companion has no authenticated files")
    if not has_license_declaration(result["license_files"]):
        blockers.append("MOSS companion has no parsed license declaration")
    result["license_status"] = (
        "DECLARED_UNVERIFIED" if has_license_declaration(result["license_files"]) else "ABSENT"
    )
    if result["license_status"] == "DECLARED_UNVERIFIED":
        blockers.append(
            f"{MODEL_LICENSE_DECLARATION_UNREVIEWED_BLOCKER}: MOSS companion terms require primary review"
        )
    else:
        blockers.append(f"{MODEL_LICENSE_ABSENT_BLOCKER}: MOSS companion license is absent")
    return result


def inspect(
    model_dir: Path,
    source_dir: Path,
    moss_dir: Path,
    evidence_dir: Path,
    model_tree: Path | None,
    moss_tree: Path | None,
) -> int:
    blockers: list[str] = []
    model_files, model_file_blockers = files_under(model_dir)
    blockers.extend(model_file_blockers)
    model = {
        "repository": MODEL_REPOSITORY,
        "pinned_revision": MODEL_REVISION,
        "server_tree": tree_evidence(model_dir, model_tree, MODEL_REPOSITORY, MODEL_REVISION, blockers),
        "files": [file_identity(path, model_dir) for path in model_files],
        "config": config_evidence(model_dir, model_files, blockers),
        "tokenizer": tokenizer_evidence(model_dir, model_files, blockers),
        "safetensors": safetensors_evidence(model_dir, model_files, blockers),
        "onnx": onnx_evidence(model_dir, model_files, blockers),
        "license_files": license_evidence(model_dir, model_files),
    }
    if not has_license_declaration(model["license_files"]):
        blockers.append("model bundle has no parsed weight-card/license declaration")
    if has_license_declaration(model["license_files"]):
        model["license_status"] = "DECLARED_UNVERIFIED"
        blockers.append(MODEL_LICENSE_DECLARATION_UNREVIEWED_BLOCKER)
    else:
        model["license_status"] = "ABSENT"
        blockers.append(MODEL_LICENSE_ABSENT_BLOCKER)
    blockers.append(TOPOLOGY_UNVERIFIED_BLOCKER)
    payload = {
        "format": FORMAT,
        "model": model,
        "source": source_evidence(source_dir, blockers),
        "moss_audio_tokenizer": companion_evidence(moss_dir, moss_tree, blockers),
        "status": "BLOCKED",
        "evidence_stage": "INSPECTION_ONLY",
        "inspection_status": "AUTHENTICATED_EVIDENCE_COMPLETE",
        "collection_status": "AUTHENTICATED",
        "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
        "cpu_status": "UNSUPPORTED",
        "metal_status": "BLOCKED_BY_CPU",
        "parity_status": "NOT_RUN",
        "publication": "NO_UPLOAD",
        "license_status": "SEPARATE_WEIGHT_SOURCE_DEPENDENCY_EVIDENCE_REQUIRED",
        "voice_cloning_or_biometric_claim": "NOT_ASSESSED",
        "exit_code": 2,
        "blockers": sorted(set(blockers + [RUNTIME_BLOCKER])),
    }
    authentication_failures = []
    for label, evidence in (("model", model["server_tree"]), ("MOSS", payload["moss_audio_tokenizer"]["server_tree"])):
        if evidence.get("status") != "MATCHED":
            authentication_failures.append(f"{label} server/local identity is not authenticated")
    if payload["source"].get("identity_status") != "AUTHENTICATED":
        authentication_failures.append("VieNeu source checkout identity is not authenticated")
    if authentication_failures:
        raise RuntimeError("; ".join(authentication_failures))
    evidence_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = evidence_dir / "vieneu_v3_turbo_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"VieNeu inspection evidence is authenticated but blocked; evidence at {manifest_path}")
    return 2


def write_error_manifest(evidence_dir: Path, error: Exception) -> None:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "model_tree.json",
        "moss_tree.json",
        "tensor-inventory.json",
        "config.json",
        "source-inventory.json",
        "server-packet.json",
        "vieneu_v3_turbo_manifest.json",
    ):
        stale = evidence_dir / name
        if stale.is_file() or stale.is_symlink():
            stale.unlink()
    manifest_path = evidence_dir / "vieneu_v3_turbo_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": FORMAT,
                "status": "BLOCKED",
                "evidence_stage": "INSPECTION_ONLY",
                "inspection_status": "INSPECTION_ERROR",
                "collection_status": "FAILED",
                "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
                "cpu_status": "UNSUPPORTED",
                "metal_status": "BLOCKED_BY_CPU",
                "parity_status": "NOT_RUN",
                "publication": "NO_UPLOAD",
                "exit_code": 2,
                "error": str(error),
                "blockers": ["authenticated collection unavailable"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def self_test() -> None:
    root = Path(os.getcwd()).resolve()
    assert safe_relative(root / "tools", root) == "tools"
    try:
        safe_relative(root.parent / "escape", root)
    except ValueError:
        pass
    else:
        raise AssertionError("path traversal was accepted")
    assert token_id_fields({"bos_token_id": 1, "nested": {"eos_token": "<eos>"}}) == {
        "bos_token_id": 1,
        "nested.eos_token": "<eos>",
    }
    assert topology_fields({"hidden_size": 768, "nested": {"num_heads": 12}}) == {
        "hidden_size": 768,
        "nested.num_heads": 12,
    }
    with tempfile.TemporaryDirectory(prefix="vieneu-inspect-self-test-") as temporary:
        root = Path(temporary)
        local = root / "local"
        local.mkdir()
        readme = local / "README.md"
        readme.write_text("Project description only\n", encoding="utf-8")
        readme_license = license_evidence(local, [readme])
        assert readme_license[0]["status"] == "UNKNOWN"
        assert not has_license_declaration(readme_license)
        license_file = local / "LICENSE"
        license_file.write_text("MIT License\n", encoding="utf-8")
        parsed_license = license_evidence(local, [license_file])
        assert parsed_license[0]["status"] == "DECLARED_UNVERIFIED"
        assert has_license_declaration(parsed_license)
        tree = root / "tree.json"
        tree.write_text(
            json.dumps(
                {
                    "repository": MODEL_REPOSITORY,
                    "requested_revision": MODEL_REVISION,
                    "resolved_revision": MODEL_REVISION,
                    "files": [{"path": "other.bin", "type": "file", "size": 1, "git_blob_sha1": "0" * 40}],
                }
            ),
            encoding="utf-8",
        )
        blockers: list[str] = []
        assert tree_evidence(local, tree, MODEL_REPOSITORY, MODEL_REVISION, blockers)["status"] == "MISMATCH"
        assert any("tree mismatch" in blocker for blocker in blockers)
        (local / ".cache" / "huggingface").mkdir(parents=True)
        (local / ".cache" / "huggingface" / "transport.json").write_text("{}", encoding="utf-8")
        (local / "nested" / ".cache").mkdir(parents=True)
        (local / "nested" / ".cache" / "spoof").write_text("x", encoding="utf-8")
        _, cache_blockers = files_under(local)
        assert any("unauthenticated metadata path" in blocker for blocker in cache_blockers)
        lfs_root = root / "lfs"
        lfs_root.mkdir()
        payload = lfs_root / "weights.bin"
        payload.write_bytes(b"fixed payload")
        payload_sha = sha256(payload)
        pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{payload_sha}\nsize {payload.stat().st_size}\n".encode("ascii")
        pointer_sha = hashlib.sha1(f"blob {len(pointer)}\0".encode("ascii") + pointer).hexdigest()
        lfs_tree = root / "lfs-tree.json"
        lfs_tree.write_text(
            json.dumps(
                {
                    "repository": MODEL_REPOSITORY,
                    "requested_revision": MODEL_REVISION,
                    "resolved_revision": MODEL_REVISION,
                    "files": [
                        {
                            "path": "weights.bin",
                            "type": "file",
                            "size": payload.stat().st_size,
                            "git_blob_sha1": pointer_sha,
                            "lfs_sha256": payload_sha,
                            "lfs_size": payload.stat().st_size,
                            "lfs_pointer_sha1": pointer_sha,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        assert tree_evidence(lfs_root, lfs_tree, MODEL_REPOSITORY, MODEL_REVISION, [])[
            "status"
        ] == "MATCHED"
    with tempfile.TemporaryDirectory(prefix="vieneu-json-self-test-") as directory:
        duplicate = Path(directory) / "duplicate.json"
        duplicate.write_text('{"a": 1, "a": 2}', encoding="utf-8")
        try:
            json_load_unique(duplicate)
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate JSON key was accepted")
    with tempfile.TemporaryDirectory(prefix="vieneu-source-self-test-") as directory:
        source = Path(directory) / "source"
        source.mkdir()
        (source / "README.md").write_text("fixture\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(source), "init", "--quiet"], check=True)
        subprocess.run(["git", "-C", str(source), "add", "README.md"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "-c",
                "user.name=Vokra self-test",
                "-c",
                "user.email=vokra-self-test@example.invalid",
                "commit",
                "--quiet",
                "-m",
                "fixture",
            ],
            check=True,
        )
        source_blockers: list[str] = []
        source_result = source_evidence(source, source_blockers)
        assert source_result["tracked_files"]
        assert any("fixed role is missing" in blocker for blocker in source_blockers)
    try:
        validate_external_location("../weights.bin")
    except ValueError:
        pass
    else:
        raise AssertionError("ONNX external-data traversal was accepted")
    assert not valid_external_range(9, None, 8)
    assert not valid_external_range(4, 5, 8)
    assert valid_external_range(8, 0, 8)
    assert len(MODEL_REVISION) == 40 and len(SOURCE_REVISION) == 40 and len(MOSS_REVISION) == 40
    assert SOURCE_REVISION == SOURCE_TAG_OBJECT
    assert SOURCE_TAG_NAME == "v3.0.0"
    assert SOURCE_PEELED_COMMIT == "28392eee571db0da31632882ac7226faa2d09d5d"
    tag_content = f"object {SOURCE_PEELED_COMMIT}\ntype commit\ntag {SOURCE_TAG_NAME}\n\n"
    assert not validate_source_tag_identity(
        SOURCE_TAG_OBJECT, "tag", tag_content, SOURCE_PEELED_COMMIT
    )
    for invalid_tag, invalid_type, invalid_content, invalid_head in (
        ("0" * 40, "tag", tag_content, SOURCE_PEELED_COMMIT),
        (SOURCE_TAG_OBJECT, "commit", tag_content, SOURCE_PEELED_COMMIT),
        (
            SOURCE_TAG_OBJECT,
            "tag",
            f"object {'0' * 40}\ntype commit\ntag {SOURCE_TAG_NAME}\n\n",
            SOURCE_PEELED_COMMIT,
        ),
        (SOURCE_TAG_OBJECT, "tag", tag_content, "0" * 40),
    ):
        assert validate_source_tag_identity(invalid_tag, invalid_type, invalid_content, invalid_head)
    assert SOURCE_URL.startswith("https://github.com/")
    assert set(SOURCE_ROLE_BLOBS) == {
        "LICENSE",
        "pyproject.toml",
        "uv.lock",
        "src/vieneu/_v3_turbo_engine/__init__.py",
        "src/vieneu/_v3_turbo_engine/configuration_v3_turbo.py",
        "src/vieneu/_v3_turbo_engine/hub_load_v3_turbo.py",
        "src/vieneu/_v3_turbo_engine/inference_v3_turbo.py",
        "src/vieneu/_v3_turbo_engine/modeling_v3_turbo.py",
        "src/vieneu/_v3_turbo_engine/onnx_runtime_lite.py",
        "src/vieneu/_v3_turbo_engine/prompt_v3_turbo.py",
    }
    synthetic_roles = {
        relative: {"mode": "100644", "git_blob_sha1": blob}
        for relative, blob in SOURCE_ROLE_BLOBS.items()
    }
    validate_fixed_roles(synthetic_roles)
    spoofed_roles = dict(synthetic_roles)
    spoofed_roles["src/vieneu/_v3_turbo_engine/inference_v3_turbo.py"] = {
        "mode": "100755",
        "git_blob_sha1": SOURCE_ROLE_BLOBS["src/vieneu/_v3_turbo_engine/inference_v3_turbo.py"],
    }
    try:
        validate_fixed_roles(spoofed_roles)
    except RuntimeError:
        pass
    else:
        raise AssertionError("fixed source role mode drift was accepted")
    assert '"inspection_status": "AUTHENTICATED_EVIDENCE_COMPLETE"' in Path(__file__).read_text(encoding="utf-8")
    assert '"inspection_status": "INSPECTION_ERROR"' in Path(__file__).read_text(encoding="utf-8")
    print("vieneu_v3_turbo_inspect self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--moss-dir", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--model-tree", type=Path)
    parser.add_argument("--moss-tree", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.model_dir, args.source_dir, args.moss_dir, args.evidence_dir, args.model_tree, args.moss_tree)):
            parser.error("--self-test accepts no other arguments")
        self_test()
        return 0
    if any(value is None for value in (args.model_dir, args.source_dir, args.moss_dir, args.evidence_dir, args.model_tree, args.moss_tree)):
        parser.error("normal runs require model/source/MOSS dirs, evidence dir, and server tree evidence")
    try:
        return inspect(args.model_dir, args.source_dir, args.moss_dir, args.evidence_dir, args.model_tree, args.moss_tree)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        write_error_manifest(args.evidence_dir, error)
        print(f"VieNeu inspection error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
