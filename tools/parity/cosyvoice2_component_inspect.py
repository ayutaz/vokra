#!/usr/bin/env -S uv run --no-project --python 3.12 python
"""VAST-only structural inspection for one CosyVoice2 checkpoint component.

This helper is intentionally not a converter or a model runner.  It hashes one
fixed upstream artifact, reads only its ZIP ``data.pkl`` graph, and delegates
the restricted pickle parse to ``tools/audit/torch_pickle_manifest.py`` in a
fresh ``uv run`` subprocess.  Tensor storage members are never opened.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
import pickle
import platform
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

MODEL_REPOSITORY = "FunAudioLLM/CosyVoice2-0.5B"
MODEL_REVISION = "eec1ae6c79877dbd9379285cf8789c9e0879293d"
SOURCE_REPOSITORY = "https://github.com/FunAudioLLM/CosyVoice.git"
SOURCE_REVISION = "8555549e882236e6541748b1042d95693caa82ba"
SOURCE_CLOSURE_PATH = "tools/parity/cosyvoice2_flow_source_closure.json"
SOURCE_CLOSURE_SHA256 = "cc391a4a63e95b9cdf6373b5b41ac94239a89d6594a235bc142a17c542532673"
SOURCE_CLOSURE_NODE_COUNT = 15
MATCHA_PATH = "third_party/Matcha-TTS"
MATCHA_REPOSITORY = "https://github.com/shivammehta25/Matcha-TTS.git"
MATCHA_REVISION = "dd9105b34bf2be2230f4aa1e4769fb586a3c824e"
MATCHA_NODE_COUNT = 4
LICENSE_PATH = "LICENSE"
LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
LICENSE_GIT_BLOB_SHA1 = "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64"
CONFIG_PATH = "cosyvoice2.yaml"
CONFIG_BYTES = 7_330
CONFIG_SHA256 = "0af2c0d010c477187c39f3e8fd5f1ae2e4e6f90ad03ba37c10ed6c6a87b05959"
CONFIG_GIT_BLOB_SHA1 = "bc19267bbfd373c9a760b7667a74349ddd487db1"
QWEN_CONFIG_PATH = "CosyVoice-BlankEN/config.json"
QWEN_CONFIG_BYTES = 659
QWEN_CONFIG_SHA256 = "168aa1bd401abc3bc262ba15ba4e499627a8b4e006e9d050b47c22de20660185"
QWEN_CONFIG_GIT_BLOB_SHA1 = "463b055262b6c66c4629a74a4b300bfe2ed31d3c"

# These are the authenticated source roles already recorded for this exact
# CosyVoice revision.  Keeping the role hashes here makes this standalone
# component gate independent of the composite inspector.
SOURCE_ROLES = {
    "cosyvoice/cli/cosyvoice.py": (
        "8e44f0f0144378561a00ebc065fdb15a843bc4650e68683bebb6624827731859",
        "cc443bed44c651a47492fc7e2142e3a88fb47627",
        "CosyVoice2",
    ),
    "cosyvoice/llm/llm.py": (
        "6439d57fcf78bcdcad6d31812f3f4b02bd34f513333711ee317d71d1fd14d2de",
        "59ebd48fde1f1b69240391fdac6e2afc1035e123",
        "Qwen2LM",
    ),
    "cosyvoice/flow/flow.py": (
        "a8497feb58336e7566b1f085d11acff9cb4f1a24949abd2c244fbf97c76f9b6d",
        "a068288f889aff4079b0c54c612897d31d08882a",
        "CausalMaskedDiffWithXvec",
    ),
    "cosyvoice/tokenizer/tokenizer.py": (
        "94340fc7cdf270c69a3aeb63290c5241044e20714e01fea736f361f9e5a56df2",
        "43fb39a2b543cc7ba4ec95fca9327596c34dcff0",
        "Qwen",
    ),
    "cosyvoice/flow/flow_matching.py": (
        "b1ad671fe37f872c034bde8f75cc19c1b88758d54e375fa2b54e14a088addfe6",
        "7f92df5d24690fe89fc548ab60f37483f91b03a6",
        "CausalConditionalCFM",
    ),
    "cosyvoice/flow/decoder.py": (
        "ef5eceb9db7f63ddda1d5bca6bfa6b28b8ea11656c4b1f9c109d28f656cbbf29",
        "97768a459fbb89a2c99f98de302628d8ccafda67",
        "CausalConditionalDecoder",
    ),
    "cosyvoice/transformer/upsample_encoder.py": (
        "a8003c212ce64697ce43001f776902ee60696a3b7e373935479029dccaf7d569",
        "6ffda6acad25cc0cfcf1bc07b9211c326ca8d49f",
        "UpsampleConformerEncoder",
    ),
}

COMPONENTS = {
    "llm": {
        "path": "llm.pt",
        "bytes": 2_023_316_821,
        "sha256": "b144ef55b51ce8cfb79a73c90dbba0bdaba4e451c0ebcfab20f769264f84a608",
        "tensor_count": 295,
        "data_pickle_sha256": "eb6aeb033f8d864a1866dc72b6acef7ba9726241c82111a610380c2bfa6637f2",
        "manifest_sha256": "07cf10ae088c27a7c88e1c08fb231d00b01bba0c13f312a74d2fd4b35403bda2",
        "storage_manifest_sha256": "3015ec0b732f8af5b4cf7aab8b6c9fdbfb4339c4a828fb61489250b2f9c88264",
        "storage_member_count": 294,
        "member_count": 296,
        "roles": ("cosyvoice/cli/cosyvoice.py", "cosyvoice/llm/llm.py", "cosyvoice/tokenizer/tokenizer.py"),
    },
    "flow": {
        "path": "flow.pt",
        "bytes": 450_575_567,
        "sha256": "ff4c2f867674411e0a08cee702996df13fa67c1cd864c06108da88d16d088541",
        "tensor_count": 1121,
        "data_pickle_sha256": "d8977bfb852a57439b8e6ca0a656b69bf171c6f86e434bce2210d2c05f0a2448",
        "manifest_sha256": "68abcdcdc961091b9c9e6456146ac9da06cda9f86e45a8faaac22a3b0b26217d",
        "storage_manifest_sha256": "e2ec1a5009a0bf4f63eaebe82d4a1bc037f1cb26e82360f7d40c4c9700fc68ee",
        "storage_member_count": 1121,
        "member_count": 1123,
        "roles": ("cosyvoice/cli/cosyvoice.py", "cosyvoice/flow/flow.py", "cosyvoice/flow/flow_matching.py", "cosyvoice/flow/decoder.py", "cosyvoice/transformer/upsample_encoder.py"),
    },
}

# The model-side config identity is authenticated against the exact-head
# CosyVoice2 inspection record after the VAST runner acquires this sidecar.
MODEL_CONFIG = {"path": CONFIG_PATH, "bytes": CONFIG_BYTES, "sha256": CONFIG_SHA256, "git_blob_sha1": CONFIG_GIT_BLOB_SHA1}

FORMAT = "vokra-cosyvoice2-component-inspection-v1"
MAX_ARCHIVE_MEMBERS = 300_000
MAX_ARCHIVE_BYTES = 8_000_000_000
MAX_DATA_PICKLE_BYTES = 64 * 1024 * 1024
MAX_TENSORS = 300_000
MAX_DIMENSION = 1 << 40
MAX_STORAGE_NUMEL = 1 << 40
MAX_SUBPROCESS_OUTPUT = 1 << 20


class InspectionError(ValueError):
    """The artifact or authenticated source is not safe to inspect."""


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InspectionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_loads(raw: str) -> Any:
    try:
        return json.loads(raw, object_pairs_hook=_strict_object_pairs)
    except InspectionError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise InspectionError(f"invalid JSON: {error}") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_regular(path: Path, label: str) -> None:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise InspectionError(f"{label} must be an absolute regular non-symlink file")


def safe_member_name(name: str) -> None:
    path = PurePosixPath(name)
    if (
        not name
        or "\x00" in name
        or name.startswith("/")
        or "\\" in name
        or ".." in path.parts
        or path.is_absolute()
        or any(not component for component in path.parts)
        or len(name) > 4096
    ):
        raise InspectionError(f"unsafe ZIP member: {name!r}")


def read_data_pickle(path: Path) -> tuple[bytes, dict[str, Any]]:
    """Validate ZIP metadata and read exactly one bounded ``*/data.pkl``."""
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as error:
        raise InspectionError(f"checkpoint is not a readable ZIP: {error}") from error
    with archive:
        infos = archive.infolist()
        if not infos or len(infos) > MAX_ARCHIVE_MEMBERS:
            raise InspectionError("ZIP member count is outside the safe bound")
        names: set[str] = set()
        total_bytes = 0
        data_infos: list[zipfile.ZipInfo] = []
        for info in infos:
            name = info.filename
            safe_member_name(name)
            if name in names or info.is_dir() or info.flag_bits & 1:
                raise InspectionError(f"duplicate, directory, or encrypted ZIP member: {name!r}")
            mode = info.external_attr >> 16
            if mode not in (0, 0o600, 0o100644, 0o100755):
                raise InspectionError(f"non-regular ZIP member: {name!r}")
            total_bytes += info.file_size
            if total_bytes > MAX_ARCHIVE_BYTES:
                raise InspectionError("ZIP uncompressed byte bound exceeded")
            names.add(name)
            member = PurePosixPath(name)
            if member.name == "data.pkl" and len(member.parts) >= 2:
                data_infos.append(info)
        if len(data_infos) != 1:
            raise InspectionError(f"expected exactly one */data.pkl member, got {len(data_infos)}")
        data_info = data_infos[0]
        if data_info.file_size > MAX_DATA_PICKLE_BYTES:
            raise InspectionError("data.pkl exceeds bounded parser input size")
        data = bytearray()
        try:
            with archive.open(data_info, "r") as stream:
                while True:
                    block = stream.read(min(1 << 20, MAX_DATA_PICKLE_BYTES + 1 - len(data)))
                    if not block:
                        break
                    data.extend(block)
                    if len(data) > MAX_DATA_PICKLE_BYTES:
                        raise InspectionError("data.pkl read bound exceeded")
        except (OSError, RuntimeError, zipfile.BadZipFile) as error:
            raise InspectionError(f"data.pkl read failed: {error}") from error
        if len(data) != data_info.file_size:
            raise InspectionError("data.pkl size changed while reading")
        prefix = str(PurePosixPath(data_info.filename).parent)
        prefix = "" if prefix == "." else prefix + "/"
        storage_members = sorted(
            name[len(prefix + "data/") :]
            for name in names
            if name.startswith(prefix + "data/") and name != prefix + "data/"
        )
        return bytes(data), {
            "member_count": len(infos),
            "uncompressed_bytes": total_bytes,
            "data_pickle_member": data_info.filename,
            "data_pickle_bytes": data_info.file_size,
            "storage_member_count": len(storage_members),
            "storage_members": storage_members,
        }


def run_restricted_parser(raw: bytes, root: Path, source_label: str) -> dict[str, Any]:
    """Parse ``data.pkl`` in the repository parser's isolated uv process."""
    parser = root / "tools/audit/torch_pickle_manifest.py"
    require_regular(parser.resolve(), "restricted parser")
    with tempfile.TemporaryDirectory(prefix="cosyvoice2-component-pickle-") as temp:
        temp_root = Path(temp)
        pickle_path = temp_root / "data.pkl"
        manifest_path = temp_root / "manifest.json"
        pickle_path.write_bytes(raw)
        command = [
            "uv",
            "run",
            "--no-project",
            "--python",
            "3.12",
            "python",
            str(parser),
            str(pickle_path),
            str(manifest_path),
            "--source",
            source_label,
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except OSError as error:
            raise InspectionError(f"restricted parser subprocess failed to start: {error}") from error
        stdout = completed.stdout[-MAX_SUBPROCESS_OUTPUT:]
        stderr = completed.stderr[-MAX_SUBPROCESS_OUTPUT:]
        if completed.returncode != 0:
            detail = (stderr or stdout).strip().replace("\n", " ")
            raise InspectionError(f"restricted parser rejected data.pkl: {detail[:400]}")
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise InspectionError("restricted parser did not emit a manifest")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise InspectionError(f"restricted parser manifest is invalid: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("format") != "vokra-pytorch-state-dict-manifest-v1":
        raise InspectionError("restricted parser manifest format mismatch")
    tensors = manifest.get("tensors")
    if not isinstance(tensors, dict) or not tensors or len(tensors) > MAX_TENSORS:
        raise InspectionError("restricted parser tensor inventory is outside bounds")
    return manifest


def inspect_checkpoint(path: Path, root: Path, component: str) -> dict[str, Any]:
    raw, archive = read_data_pickle(path)
    manifest = run_restricted_parser(
        raw,
        root,
        f"hf://{MODEL_REPOSITORY}@{MODEL_REVISION}:{COMPONENTS[component]['path']}/data.pkl",
    )
    tensors = manifest["tensors"]
    for name, tensor in tensors.items():
        if not isinstance(name, str) or not name or "\x00" in name or "/" in name or "\\" in name:
            raise InspectionError(f"unsafe tensor name: {name!r}")
        if not isinstance(tensor, dict) or not isinstance(tensor.get("shape"), list):
            raise InspectionError(f"invalid tensor record: {name!r}")
        shape = tensor["shape"]
        stride = tensor.get("stride")
        if not isinstance(stride, list) or len(stride) != len(shape):
            raise InspectionError(f"tensor stride rank mismatch: {name}")
        if any(type(dimension) is not int or dimension < 0 or dimension > MAX_DIMENSION for dimension in shape):
            raise InspectionError(f"tensor dimension bound exceeded: {name}")
        if any(type(step) is not int or step < 0 or step > MAX_DIMENSION for step in stride):
            raise InspectionError(f"tensor stride bound exceeded: {name}")
        storage_key = tensor.get("storage_key")
        if not isinstance(storage_key, str) or storage_key not in archive["storage_members"]:
            raise InspectionError(f"tensor storage member is missing: {name}")
        storage_numel = tensor.get("storage_numel")
        storage_offset = tensor.get("storage_offset")
        if type(storage_numel) is not int or storage_numel < 0 or storage_numel > MAX_STORAGE_NUMEL:
            raise InspectionError(f"tensor storage_numel is outside bounds: {name}")
        if type(storage_offset) is not int or storage_offset < 0 or storage_offset > storage_numel:
            raise InspectionError(f"tensor storage_offset is outside bounds: {name}")
        if all(dimension > 0 for dimension in shape):
            max_address = storage_offset + sum(step * (dimension - 1) for step, dimension in zip(stride, shape))
            if max_address >= storage_numel:
                raise InspectionError(f"tensor strided view exceeds storage_numel: {name}")
    manifest["archive"] = archive
    manifest["payload_reads"] = "DATA_PKL_ONLY; TENSOR_STORAGE_MEMBERS_NOT_OPENED"
    manifest["component_contract"] = "NO_COMPONENT_SHAPE_ASSERTIONS_WITHOUT_AUTHENTICATED_FACTS"
    return manifest


def authenticate_config(config: Path) -> dict[str, Any]:
    require_regular(config, CONFIG_PATH)
    if config.stat().st_size != CONFIG_BYTES:
        raise InspectionError("cosyvoice2.yaml byte count does not match pinned config")
    actual_sha = sha256_file(config)
    actual_blob = git_blob_sha1(config)
    if actual_sha != CONFIG_SHA256 or actual_blob != CONFIG_GIT_BLOB_SHA1:
        raise InspectionError("cosyvoice2.yaml identity mismatch")
    return {**MODEL_CONFIG, "sha256": actual_sha, "git_blob_sha1": actual_blob, "verification": "ACQUIRED_AND_HASH_VERIFIED"}


def authenticate_qwen_config(config: Path) -> dict[str, Any]:
    require_regular(config, QWEN_CONFIG_PATH)
    if config.stat().st_size != QWEN_CONFIG_BYTES:
        raise InspectionError("Qwen config.json byte count does not match pinned config")
    actual_sha = sha256_file(config)
    actual_blob = git_blob_sha1(config)
    if actual_sha != QWEN_CONFIG_SHA256 or actual_blob != QWEN_CONFIG_GIT_BLOB_SHA1:
        raise InspectionError("Qwen config.json identity mismatch")
    return {
        "path": QWEN_CONFIG_PATH,
        "bytes": QWEN_CONFIG_BYTES,
        "sha256": actual_sha,
        "git_blob_sha1": actual_blob,
        "verification": "ACQUIRED_AND_HASH_VERIFIED",
    }


def validate_component_config_requirements(component: str, qwen_config: Path | None) -> None:
    if component == "llm" and qwen_config is None:
        raise InspectionError("llm inspection requires the exact Qwen config.json sidecar")
    if component == "flow" and qwen_config is not None:
        raise InspectionError("flow inspection must not receive a Qwen config sidecar")


def _safe_source_relative_path(path: str) -> None:
    candidate = PurePosixPath(path)
    if (
        not path
        or path.startswith("/")
        or "\\" in path
        or "\x00" in path
        or ".." in candidate.parts
        or any(not part for part in candidate.parts)
    ):
        raise InspectionError(f"unsafe source closure path: {path!r}")


def _require_source_regular(source: Path, relative: str, label: str) -> Path:
    _safe_source_relative_path(relative)
    path = source / relative
    current = source
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise InspectionError(f"{label} has a symlinked ancestor: {current}")
    require_regular(path, label)
    return path


def _require_source_tree(source: Path, relative: str, label: str) -> Path:
    _safe_source_relative_path(relative)
    path = source / relative
    current = source
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise InspectionError(f"{label} has a symlinked ancestor: {current}")
    if path.is_symlink() or not path.is_dir():
        raise InspectionError(f"{label} must be an absolute regular non-symlink directory")
    return path


def _strict_dependency_names(records: Any, label: str) -> set[str]:
    if not isinstance(records, list) or not records:
        raise InspectionError(f"{label} dependency records are missing")
    names: set[str] = set()
    for row in records:
        if not isinstance(row, dict) or set(row) != {"import", "reason"}:
            raise InspectionError(f"{label} dependency schema is invalid")
        name, reason = row["import"], row["reason"]
        if not isinstance(name, str) or not name or name in names or not isinstance(reason, str) or not reason:
            raise InspectionError(f"{label} dependency is empty or duplicated")
        names.add(name)
    return names


def authenticate_source_closure(source: Path, root: Path) -> dict[str, Any]:
    """Authenticate the fixed repo-local Flow implementation import closure."""
    manifest = root / SOURCE_CLOSURE_PATH
    _require_source_regular(root, SOURCE_CLOSURE_PATH, "Flow source closure manifest")
    if sha256_file(manifest) != SOURCE_CLOSURE_SHA256:
        raise InspectionError("Flow source closure manifest SHA-256 mismatch")
    try:
        payload = _strict_json_loads(manifest.read_text(encoding="utf-8"))
    except (OSError, InspectionError) as error:
        raise InspectionError(f"Flow source closure manifest is invalid: {error}") from error
    if not isinstance(payload, dict) or payload.get("format") != "vokra-cosyvoice2-flow-source-closure-v1":
        raise InspectionError("Flow source closure format mismatch")
    if payload.get("repository") != SOURCE_REPOSITORY or payload.get("revision") != SOURCE_REVISION:
        raise InspectionError("Flow source closure source identity mismatch")
    if payload.get("status") != "SOURCE_CLOSURE_COMPLETE":
        raise InspectionError("Flow source closure status is not complete")
    roots = payload.get("roots")
    expected_roots = [
        "cosyvoice/cli/cosyvoice.py",
        "cosyvoice/flow/flow.py",
        "cosyvoice/flow/flow_matching.py",
        "cosyvoice/flow/decoder.py",
        "cosyvoice/transformer/upsample_encoder.py",
    ]
    if roots != expected_roots:
        raise InspectionError("Flow source closure roots are not exact")
    license_record = payload.get("license")
    if license_record != {
        "path": LICENSE_PATH,
        "bytes": 11_357,
        "sha256": LICENSE_SHA256,
        "git_blob_sha1": LICENSE_GIT_BLOB_SHA1,
        "declared": "Apache-2.0",
    }:
        raise InspectionError("Flow source closure LICENSE identity mismatch")
    nodes = payload.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != SOURCE_CLOSURE_NODE_COUNT:
        raise InspectionError("Flow source closure node count is not exact")
    node_paths: set[str] = set()
    node_records: dict[str, Any] = {}
    for node in nodes:
        if not isinstance(node, dict) or set(node) != {"path", "bytes", "sha256", "git_blob_sha1", "role"}:
            raise InspectionError("Flow source closure node schema is not exact")
        path = node["path"]
        if not isinstance(path, str) or path in node_paths:
            raise InspectionError("Flow source closure contains duplicate paths")
        node_paths.add(path)
        if type(node["bytes"]) is not int or node["bytes"] <= 0:
            raise InspectionError(f"Flow source closure bytes are invalid: {path!r}")
        if not isinstance(node["sha256"], str) or len(node["sha256"]) != 64 or not all(c in "0123456789abcdef" for c in node["sha256"]):
            raise InspectionError(f"Flow source closure SHA-256 is invalid: {path!r}")
        if not isinstance(node["git_blob_sha1"], str) or len(node["git_blob_sha1"]) != 40 or not all(c in "0123456789abcdef" for c in node["git_blob_sha1"]):
            raise InspectionError(f"Flow source closure Git blob SHA-1 is invalid: {path!r}")
        if not isinstance(node["role"], str) or not node["role"]:
            raise InspectionError(f"Flow source closure role is invalid: {path!r}")
        node_records[path] = node
    for node in nodes:
        path = node["path"]
        actual = _require_source_regular(source, path, f"Flow source closure node {path}")
        if actual.stat().st_size != node["bytes"] or sha256_file(actual) != node["sha256"] or git_blob_sha1(actual) != node["git_blob_sha1"]:
            raise InspectionError(f"Flow source closure node identity mismatch: {path}")
    excluded_paths = set()
    edges = payload.get("edges")
    if not isinstance(edges, list) or not edges:
        raise InspectionError("Flow source closure edges are missing")
    edge_keys: set[tuple[str, str]] = set()
    for edge in edges:
        if not isinstance(edge, dict) or set(edge) != {"from", "to", "reason"}:
            raise InspectionError("Flow source closure edge schema is not exact")
        source_path, target_path, reason = edge["from"], edge["to"], edge["reason"]
        if source_path not in node_paths or target_path not in node_paths or not isinstance(reason, str) or not reason:
            raise InspectionError("Flow source closure edge references an unknown or empty node")
        if (source_path, target_path) in edge_keys:
            raise InspectionError("Flow source closure contains duplicate edges")
        edge_keys.add((source_path, target_path))
    external_names = _strict_dependency_names(payload.get("external_dependencies"), "Flow source closure external")
    excluded = payload.get("excluded_repo_imports")
    if not isinstance(excluded, list) or any(not isinstance(row, dict) or set(row) != {"path", "reason"} for row in excluded):
        raise InspectionError("Flow source closure exclusions are invalid")
    for row in excluded:
        path = row["path"]
        _safe_source_relative_path(path)
        if path in node_paths or path in excluded_paths or not isinstance(row["reason"], str) or not row["reason"]:
            raise InspectionError("Flow source closure exclusions overlap or have empty reasons")
        excluded_paths.add(path)
    actual_import_edges: set[tuple[str, str]] = set()
    actual_matcha_imports: set[str] = set()
    actual_external_imports: set[str] = set()
    for node in nodes:
        source_path = node["path"]
        try:
            tree = ast.parse((source / source_path).read_text(encoding="utf-8"), filename=source_path)
        except (OSError, UnicodeError, SyntaxError) as error:
            raise InspectionError(f"Flow source closure AST parse failed for {source_path}: {error}") from error
        for statement in ast.walk(tree):
            if isinstance(statement, ast.Import):
                imported_modules = [alias.name for alias in statement.names]
            elif isinstance(statement, ast.ImportFrom):
                imported_modules = [] if statement.module is None else [statement.module]
            else:
                continue
            for module in imported_modules:
                if not module.startswith("cosyvoice."):
                    if module.startswith("matcha."):
                        actual_matcha_imports.add(module)
                    else:
                        actual_external_imports.add(module)
                    continue
                target = module.replace(".", "/") + ".py"
                if target not in node_paths and target not in excluded_paths:
                    raise InspectionError(f"unresolved repo-local import in Flow closure: {source_path} -> {target}")
                if target in node_paths:
                    actual_import_edges.add((source_path, target))
    declared_edges = set()
    for edge in edges:
        declared_edges.add((edge["from"], edge["to"]))
    if declared_edges != actual_import_edges:
        raise InspectionError("Flow source closure edges do not exactly match parsed repo-local imports")
    if external_names != actual_external_imports:
        raise InspectionError("Flow source closure external dependency set mismatch")
    bindings = payload.get("config_bindings")
    if not isinstance(bindings, list) or any(not isinstance(row, dict) or set(row) != {"config_path", "implementation", "reason"} for row in bindings):
        raise InspectionError("Flow source closure config bindings are invalid")
    for row in bindings:
        implementation = row["implementation"]
        target = implementation.rsplit(".", 1)[0].replace(".", "/") + ".py" if isinstance(implementation, str) and "." in implementation else ""
        if target not in node_paths or not isinstance(row["config_path"], str) or not row["config_path"] or not isinstance(implementation, str) or not implementation or not isinstance(row["reason"], str) or not row["reason"]:
            raise InspectionError("Flow source closure config binding is unresolved")
    matcha = payload.get("matcha")
    if not isinstance(matcha, dict) or set(matcha) != {"repository", "revision", "status", "license", "roots", "nodes", "edges", "external_dependencies"}:
        raise InspectionError("Matcha closure schema is not exact")
    if matcha["repository"] != MATCHA_REPOSITORY or matcha["revision"] != MATCHA_REVISION or matcha["status"] != "MATCHA_GITLINK_AUTHENTICATED":
        raise InspectionError("Matcha gitlink identity/status mismatch")
    matcha_license = matcha["license"]
    expected_matcha_license = {
        "path": "LICENSE",
        "bytes": 1_069,
        "sha256": "874d84104bdc7b301f369b2f2e66b31f07826a67495af909655f3699c857620d",
        "git_blob_sha1": "858018e750da7be7b271bb7307e68d159ed67ef6",
        "declared": "MIT",
    }
    if matcha_license != expected_matcha_license:
        raise InspectionError("Matcha LICENSE identity mismatch")
    if matcha["roots"] != [
        "matcha/models/components/decoder.py",
        "matcha/models/components/flow_matching.py",
        "matcha/models/components/transformer.py",
    ]:
        raise InspectionError("Matcha closure roots are not exact")
    matcha_root = _require_source_tree(source, MATCHA_PATH, "Matcha gitlink checkout")

    try:
        gitlink = subprocess.check_output(
            ["git", "-C", str(source), "ls-tree", SOURCE_REVISION, MATCHA_PATH],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise InspectionError(f"CosyVoice Matcha gitlink lookup failed: {error}") from error
    if gitlink != f"160000 commit {MATCHA_REVISION}\t{MATCHA_PATH}":
        raise InspectionError("CosyVoice Matcha gitlink revision mismatch")

    def matcha_git(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(matcha_root), *args],
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise InspectionError(f"Matcha git command failed: {args!r}: {error}") from error

    if matcha_git("rev-parse", "HEAD") != MATCHA_REVISION or matcha_git("remote", "get-url", "origin") != MATCHA_REPOSITORY or matcha_git("status", "--porcelain", "--untracked-files=all"):
        raise InspectionError("Matcha gitlink revision/origin/clean checkout mismatch")
    matcha_license_path = _require_source_regular(matcha_root, "LICENSE", "Matcha LICENSE")
    if matcha_license_path.stat().st_size != expected_matcha_license["bytes"] or sha256_file(matcha_license_path) != expected_matcha_license["sha256"] or git_blob_sha1(matcha_license_path) != expected_matcha_license["git_blob_sha1"]:
        raise InspectionError("Matcha LICENSE file identity mismatch")
    matcha_nodes = matcha["nodes"]
    if not isinstance(matcha_nodes, list) or len(matcha_nodes) != MATCHA_NODE_COUNT:
        raise InspectionError("Matcha closure node count is not exact")
    matcha_node_paths: set[str] = set()
    for node in matcha_nodes:
        if not isinstance(node, dict) or set(node) != {"path", "bytes", "sha256", "git_blob_sha1", "role"}:
            raise InspectionError("Matcha closure node schema is not exact")
        path = node["path"]
        if not isinstance(path, str) or path in matcha_node_paths or not path.startswith("matcha/"):
            raise InspectionError("Matcha closure contains an invalid or duplicate path")
        matcha_node_paths.add(path)
        if type(node["bytes"]) is not int or node["bytes"] <= 0 or not isinstance(node["sha256"], str) or len(node["sha256"]) != 64 or not all(c in "0123456789abcdef" for c in node["sha256"]) or not isinstance(node["git_blob_sha1"], str) or len(node["git_blob_sha1"]) != 40 or not all(c in "0123456789abcdef" for c in node["git_blob_sha1"]) or not isinstance(node["role"], str) or not node["role"]:
            raise InspectionError(f"Matcha closure node identity fields are invalid: {path!r}")
    for node in matcha_nodes:
        path = node["path"]
        actual = _require_source_regular(matcha_root, path, f"Matcha closure node {path}")
        if actual.stat().st_size != node["bytes"] or sha256_file(actual) != node["sha256"] or git_blob_sha1(actual) != node["git_blob_sha1"]:
            raise InspectionError(f"Matcha closure node identity mismatch: {path}")
    matcha_edges = matcha["edges"]
    if not isinstance(matcha_edges, list) or not matcha_edges:
        raise InspectionError("Matcha closure edges are missing")
    matcha_edge_keys: set[tuple[str, str]] = set()
    for edge in matcha_edges:
        if not isinstance(edge, dict) or set(edge) != {"from", "to", "reason"}:
            raise InspectionError("Matcha closure edge schema is not exact")
        source_path, target_path, reason = edge["from"], edge["to"], edge["reason"]
        if source_path not in matcha_node_paths or target_path not in matcha_node_paths or not isinstance(reason, str) or not reason or (source_path, target_path) in matcha_edge_keys:
            raise InspectionError("Matcha closure edge references an unknown, empty, or duplicate node")
        matcha_edge_keys.add((source_path, target_path))
    matcha_external_names = _strict_dependency_names(matcha["external_dependencies"], "Matcha external")
    actual_matcha_edges: set[tuple[str, str]] = set()
    actual_matcha_external: set[str] = set()
    for node in matcha_nodes:
        source_path = node["path"]
        try:
            tree = ast.parse((matcha_root / source_path).read_text(encoding="utf-8"), filename=source_path)
        except (OSError, UnicodeError, SyntaxError) as error:
            raise InspectionError(f"Matcha closure AST parse failed for {source_path}: {error}") from error
        for statement in ast.walk(tree):
            if isinstance(statement, ast.Import):
                imported_modules = [alias.name for alias in statement.names]
            elif isinstance(statement, ast.ImportFrom):
                imported_modules = [] if statement.module is None else [statement.module]
            else:
                continue
            for module in imported_modules:
                if module.startswith("matcha."):
                    target = module.replace(".", "/") + ".py"
                    if target not in matcha_node_paths:
                        raise InspectionError(f"unresolved Matcha import: {source_path} -> {target}")
                    actual_matcha_edges.add((source_path, target))
                elif module.startswith("cosyvoice."):
                    raise InspectionError(f"Matcha closure imports CosyVoice implementation: {source_path} -> {module}")
                else:
                    actual_matcha_external.add(module)
    if actual_matcha_edges != matcha_edge_keys:
        raise InspectionError("Matcha closure edges do not exactly match parsed imports")
    if actual_matcha_external != matcha_external_names:
        raise InspectionError("Matcha external dependency set mismatch")
    contract = payload.get("execution_contract")
    if contract != {"model_download": "NOT_RUN", "model_execution": "NOT_RUN", "cpu": "NOT_RUN", "metal": "NOT_RUN", "publication": "NO_UPLOAD"}:
        raise InspectionError("Flow source closure execution contract mismatch")
    return {"path": SOURCE_CLOSURE_PATH, "sha256": SOURCE_CLOSURE_SHA256, "node_count": len(nodes), "nodes": node_records, "edge_count": len(edges), "matcha_node_count": len(matcha_nodes), "matcha_edge_count": len(matcha_edges), "status": payload["status"]}


def authenticate_source(source: Path, component: str, root: Path) -> dict[str, Any]:
    if not source.is_absolute() or source.is_symlink() or not source.is_dir():
        raise InspectionError("source checkout must be an absolute regular directory")

    def git(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(source), *args],
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise InspectionError(f"source git command failed: {args!r}: {error}") from error

    head = git("rev-parse", "HEAD")
    origin = git("remote", "get-url", "origin")
    clean = git("status", "--porcelain", "--untracked-files=all")
    if head != SOURCE_REVISION or origin != SOURCE_REPOSITORY or clean:
        raise InspectionError("CosyVoice source identity or clean-checkout mismatch")
    role_records: dict[str, Any] = {}
    for role in COMPONENTS[component]["roles"]:
        expected_sha, expected_blob, marker = SOURCE_ROLES[role]
        path = _require_source_regular(source, role, f"source role {role}")
        actual_sha = sha256_file(path)
        actual_blob = git_blob_sha1(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        if actual_sha != expected_sha or actual_blob != expected_blob or marker not in text:
            raise InspectionError(f"source role identity mismatch: {role}")
        role_records[role] = {
            "sha256": actual_sha,
            "git_blob_sha1": actual_blob,
            "bytes": path.stat().st_size,
            "marker": marker,
        }
    license_path = _require_source_regular(source, LICENSE_PATH, "CosyVoice Apache LICENSE")
    license_sha = sha256_file(license_path)
    license_blob = git_blob_sha1(license_path)
    if license_sha != LICENSE_SHA256 or license_blob != LICENSE_GIT_BLOB_SHA1:
        raise InspectionError("CosyVoice Apache LICENSE identity mismatch")
    if "Apache License" not in license_path.read_text(encoding="utf-8", errors="replace"):
        raise InspectionError("CosyVoice Apache LICENSE marker is missing")
    return {
        "repository": SOURCE_REPOSITORY,
        "revision": SOURCE_REVISION,
        "resolved_revision": head,
        "origin": origin,
        "clean": True,
        "roles": role_records,
        "flow_source_closure": authenticate_source_closure(source, root) if component == "flow" else None,
        "license": {"path": LICENSE_PATH, "bytes": license_path.stat().st_size, "sha256": license_sha, "git_blob_sha1": license_blob, "declared": "Apache-2.0"},
    }


def write_error(output: Path, component: str, message: str) -> None:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        return
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FORMAT,
        "status": "BLOCKED",
        "inspection_status": "INSPECTION_ERROR",
        "evidence_stage": "INSPECTION_ONLY",
        "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
        "cpu_status": "NOT_RUN",
        "metal_status": "NOT_RUN",
        "parity_status": "NOT_RUN",
        "publication": "NO_UPLOAD",
        "component": component,
        "blockers": [message],
    }
    (output / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def inspect(
    checkpoint: Path,
    source: Path,
    config: Path,
    qwen_config: Path | None,
    output: Path,
    root: Path,
    component: str,
) -> int:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise InspectionError("Linux x86_64 VAST is required")
    if os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1":
        raise InspectionError("VOKRA_PUBLISH_ON_VAST=1 is required")
    if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise InspectionError("inspection output must be absent or empty")
    validate_component_config_requirements(component, qwen_config)
    require_regular(checkpoint, COMPONENTS[component]["path"])
    expected = COMPONENTS[component]
    if checkpoint.stat().st_size != expected["bytes"]:
        raise InspectionError(f"{expected['path']} byte count does not match pinned artifact")
    actual_sha = sha256_file(checkpoint)
    if actual_sha != expected["sha256"]:
        raise InspectionError(f"{expected['path']} SHA-256 does not match pinned artifact")
    config_record = authenticate_config(config)
    qwen_config_record = authenticate_qwen_config(qwen_config) if qwen_config is not None else None
    source_record = authenticate_source(source, component, root)
    checkpoint_manifest = inspect_checkpoint(checkpoint, root, component)
    payload = {
        "format": FORMAT,
        "status": "BLOCKED",
        "inspection_status": "AUTHENTICATED_EVIDENCE_COMPLETE",
        "evidence_stage": "INSPECTION_ONLY",
        "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
        "cpu_status": "NOT_RUN",
        "metal_status": "NOT_RUN",
        "parity_status": "NOT_RUN",
        "publication": "NO_UPLOAD",
        "component": component,
        "artifact": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "path": expected["path"],
            "url": f"https://huggingface.co/{MODEL_REPOSITORY}/resolve/{MODEL_REVISION}/{expected['path']}?download=true",
            "bytes": expected["bytes"],
            "sha256": actual_sha,
        },
        "model_config": config_record,
        "official_source": source_record,
        "checkpoint": checkpoint_manifest,
        "blockers": [
            "Native CosyVoice2 component binder is not implemented; this is structural evidence only.",
            "No CPU, Metal, or numerical parity execution was performed.",
        ],
    }
    if qwen_config_record is not None:
        payload["qwen_config"] = qwen_config_record
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return 2


class _SyntheticStorage:
    pass


def _synthetic_rebuild(*args: Any) -> None:
    del args


class _SyntheticTensor:
    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (_synthetic_rebuild, (_SyntheticStorage(), 0, (2, 2), (2, 1), False, None))


def synthetic_pickle() -> bytes:
    class Pickler(pickle.Pickler):
        def persistent_id(self, value: Any) -> Any:
            if isinstance(value, _SyntheticStorage):
                return ("storage", "FloatStorage", "0", "cpu", 4)
            return None

    stream = io.BytesIO()
    Pickler(stream, protocol=2).dump({"w": _SyntheticTensor()})
    raw = stream.getvalue().replace(b"c__main__\n_synthetic_rebuild\n", b"ctorch._utils\n_rebuild_tensor_v2\n")
    return raw.replace(b"X\x0c\x00\x00\x00FloatStorage", b"ctorch\nFloatStorage\n")


def self_test() -> None:
    try:
        _strict_json_loads('{"duplicate": 1, "duplicate": 2}')
    except InspectionError:
        pass
    else:
        raise AssertionError("duplicate JSON key accepted")
    valid_dependency = [{"import": "torch", "reason": "synthetic"}]
    assert _strict_dependency_names(valid_dependency, "self-test") == {"torch"}
    for tampered in (
        [{"import": "torch", "reason": "synthetic"}, {"import": "torch", "reason": "duplicate"}],
        [{"import": 1, "reason": "non-string"}],
        [{"import": "", "reason": "empty"}],
        [{"import": "torch", "reason": "synthetic", "extra": True}],
    ):
        try:
            _strict_dependency_names(tampered, "self-test")
        except InspectionError:
            pass
        else:
            raise AssertionError("tampered external dependency declaration accepted")
    with tempfile.TemporaryDirectory(prefix="cosyvoice2-component-self-test-") as temp:
        root = Path(__file__).resolve().parents[2]
        work = Path(temp)
        valid = work / "valid.pt"
        with zipfile.ZipFile(valid, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("archive/data.pkl", synthetic_pickle())
            archive.writestr("archive/data/0", b"\0" * 16)
            archive.writestr("archive/version", b"3\n")
        raw, info = read_data_pickle(valid)
        assert raw == synthetic_pickle() and info["data_pickle_member"] == "archive/data.pkl"
        manifest = inspect_checkpoint(valid, root, "flow")
        assert manifest["tensor_count"] == 1 and manifest["tensors"]["w"]["shape"] == [2, 2]
        assert QWEN_CONFIG_BYTES == 659
        assert QWEN_CONFIG_SHA256 == "168aa1bd401abc3bc262ba15ba4e499627a8b4e006e9d050b47c22de20660185"
        assert QWEN_CONFIG_GIT_BLOB_SHA1 == "463b055262b6c66c4629a74a4b300bfe2ed31d3c"
        try:
            validate_component_config_requirements("llm", None)
        except InspectionError:
            pass
        else:
            raise AssertionError("LLM accepted a missing Qwen config")
        try:
            validate_component_config_requirements("flow", work / "qwen-config.json")
        except InspectionError:
            pass
        else:
            raise AssertionError("flow accepted a Qwen config")
        tampered_qwen = work / "tampered-qwen-config.json"
        tampered_qwen.write_bytes(b"tampered" + b"\0" * (QWEN_CONFIG_BYTES - len(b"tampered")))
        try:
            authenticate_qwen_config(tampered_qwen)
        except InspectionError:
            pass
        else:
            raise AssertionError("tampered Qwen config accepted")
        unsafe = work / "unsafe.pt"
        with zipfile.ZipFile(unsafe, "w") as archive:
            archive.writestr("../archive/data.pkl", synthetic_pickle())
        try:
            read_data_pickle(unsafe)
        except InspectionError:
            pass
        else:
            raise AssertionError("unsafe ZIP member accepted")
        multiple = work / "multiple.pt"
        with zipfile.ZipFile(multiple, "w") as archive:
            archive.writestr("a/data.pkl", synthetic_pickle())
            archive.writestr("b/data.pkl", synthetic_pickle())
        try:
            read_data_pickle(multiple)
        except InspectionError:
            pass
        else:
            raise AssertionError("multiple data.pkl members accepted")
        malicious = work / "malicious.pt"
        with zipfile.ZipFile(malicious, "w") as archive:
            archive.writestr("archive/data.pkl", b"\x80\x02cposix\nsystem\n.")
            archive.writestr("archive/data/0", b"")
        try:
            inspect_checkpoint(malicious, root, "llm")
        except InspectionError:
            pass
        else:
            raise AssertionError("malicious pickle global accepted")
        for name in ("/data.pkl", "archive/../data.pkl", "archive\\data.pkl", "archive/data.pkl\x00"):
            try:
                safe_member_name(name)
            except InspectionError:
                pass
            else:
                raise AssertionError(f"unsafe member accepted: {name!r}")
        for flag in ("--self-test", "--component", "--checkpoint", "--source", "--config", "--qwen-config", "--output"):
            try:
                cli_flag_counts([flag, flag])
            except InspectionError:
                pass
            else:
                raise AssertionError(f"duplicate CLI option accepted: {flag}")
    print("cosyvoice2_component_inspect self-test: OK")


def cli_flag_counts(argv: list[str]) -> dict[str, int]:
    def flag_count(name: str) -> int:
        return sum(arg == name or arg.startswith(name + "=") for arg in argv)

    counts = {name: flag_count(name) for name in ("--self-test", "--component", "--checkpoint", "--source", "--config", "--qwen-config", "--output")}
    if any(count > 1 for count in counts.values()):
        raise InspectionError("duplicate CLI option is not allowed")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--component", choices=tuple(COMPONENTS))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--qwen-config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        counts = cli_flag_counts(sys.argv[1:])
    except InspectionError as error:
        parser.error(str(error))
    if any(count > 1 for count in counts.values()):
        parser.error("duplicate CLI option is not allowed")
    component_flags = counts["--component"]
    if not args.self_test and component_flags != 1:
        parser.error("normal run requires exactly one --component llm|flow")
    if args.self_test and (component_flags or any(value is not None for value in (args.checkpoint, args.source, args.config, args.qwen_config, args.output))):
        parser.error("--self-test accepts no component or paths")
    if not args.self_test and any(value is None for value in (args.checkpoint, args.source, args.config, args.output)):
        parser.error("normal run requires --component, --checkpoint, --source, --config, and --output")
    if not args.self_test:
        if args.component == "llm" and args.qwen_config is None:
            parser.error("llm inspection requires --qwen-config")
        if args.component == "flow" and args.qwen_config is not None:
            parser.error("flow inspection must not receive --qwen-config")
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        try:
            self_test()
        except Exception as error:
            print(f"cosyvoice2_component_inspect self-test FAILED: {error}", file=sys.stderr)
            return 1
        return 0
    try:
        return inspect(args.checkpoint, args.source, args.config, args.qwen_config, args.output, Path(__file__).resolve().parents[2], args.component)
    except Exception as error:
        write_error(args.output, args.component, str(error))
        print(f"cosyvoice2_component_inspect: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
