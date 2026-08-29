#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Inspect the official ACE-Step 1.5 composite bundle on VAST.

This is an inspection oracle, not a native-runtime or parity implementation.
It inventories every expected component with safetensors metadata, safely
loads optional PyTorch containers with ``weights_only=True`` only, and records
the official source composition and locked Python dependency identities.  It
never creates a GGUF or executes a music-generation mirror.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

UPSTREAM_REPOSITORY = "ACE-Step/Ace-Step1.5"
UPSTREAM_REVISION = "19671f406d603126926c1b7e2adc169acbcade22"
SOURCE_REPOSITORY = "https://github.com/ace-step/ACE-Step-1.5"
SOURCE_REVISION = "7202bc354d7fc31d1c0e5a90b0b49fb610e52362"
FORMAT = "vokra-ace-step-v15-inspection-v1"
COMPONENTS = (
    "acestep-v15-turbo",
    "vae",
    "Qwen3-Embedding-0.6B",
    "acestep-5Hz-lm-1.7B",
)
CANONICAL_SILENCE_PATH = "acestep-v15-turbo/silence_latent.pt"
DEPENDENCIES = ("transformers", "diffusers")
TEXT_COMPONENTS = {"Qwen3-Embedding-0.6B", "acestep-5Hz-lm-1.7B"}

# These are source-object identities, not claims made by a downloaded packet.
# They were resolved with `git rev-parse HEAD:<path>` from the pinned official
# checkout.  A future source update must change this table deliberately.
SOURCE_ROLE_BLOBS = {
    "LICENSE": "600451d484a555c1273baa2602f32a37fdd0d0ab",
    "pyproject.toml": "b353723bd6c359c5ef3b9f6be3fdc04b494779a0",
    "uv.lock": "87c2442e7a852af1c19da39e0ee2008fb6e17b41",
    "acestep/constants.py": "9e6df5323d7e0a2f43237a63910d55ebbf2b39e8",
    "acestep/inference.py": "dc11604c94b2751baaa1650ff28975ce706240c9",
    "acestep/handler.py": "ff32ed9a86842bb9cc71d3affee025ccbc380778",
    "acestep/llm_inference.py": "3690af56d8fe5c6a51844917e8e56c54fc088953",
    "acestep/model_downloader.py": "30d9f9a3a1c3d0171a309c464d8306bc0059a143",
    "acestep/core/generation/handler/init_service_memory_basic.py": "b83fa52ea666bc92b86bb42f1cf66dbff81d32a5",
    "acestep/core/generation/handler/generate_music_payload.py": "ae355f9aa68e6ccadeb5123b9dd7927b98ee1480",
    "acestep/core/generation/handler/generate_music.py": "85130c91a92d88248efff37c0f619e0581f126cf",
    "acestep/core/generation/handler/diffusion.py": "020942219334af83e828391f28d365c8e48c2d97",
    "acestep/core/generation/handler/conditioning_batch.py": "9a4115adfc6c4fb6caa7e8072adb2ad0090197d9",
    "acestep/core/generation/handler/init_service_loader.py": "040c5ce114c7c25b03e3d1b0a3afa15e8eb1cc35",
    "acestep/core/generation/handler/init_service_loader_components.py": "8d8b19d86299b58709c7d5de806af16005de937b",
}
SOURCE_ROLE_MARKERS = {
    "acestep/model_downloader.py": ("MAIN_MODEL_COMPONENTS", "acestep-v15-turbo", "vae", "Qwen3-Embedding-0.6B", "acestep-5Hz-lm-1.7B"),
    "acestep/handler.py": ("sample_rate", "48000"),
    "acestep/core/generation/handler/init_service_loader.py": ("silence_latent.pt", "weights_only=True"),
    "acestep/inference.py": ("GenerationParams",),
    "acestep/core/generation/handler/generate_music.py": ("guidance", "diffusion"),
    "acestep/llm_inference.py": ("5Hz",),
    "acestep/core/generation/handler/init_service_loader_components.py": ("Qwen3-Embedding-0.6B",),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stream_file_identity(path: Path, block_size: int = 1 << 20) -> dict[str, Any]:
    """Hash a payload with bounded memory (including its Git blob identity)."""
    if block_size <= 0:
        raise ValueError("stream block size must be positive")
    size = path.stat().st_size
    payload_sha = hashlib.sha256()
    blob_sha = hashlib.sha1(f"blob {size}\0".encode("ascii"))
    counted = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            counted += len(block)
            payload_sha.update(block)
            blob_sha.update(block)
    if counted != size:
        raise RuntimeError(f"file changed while hashing: {path}")
    return {"bytes": size, "sha256": payload_sha.hexdigest(), "git_blob_sha1": blob_sha.hexdigest()}


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def git_blob_sha1(value: bytes) -> str:
    header = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(header + value).hexdigest()


def lfs_pointer_sha1(payload_sha256: str, payload_size: int) -> str:
    pointer = (
        b"version https://git-lfs.github.com/spec/v1\n"
        + f"oid sha256:{payload_sha256}\nsize {payload_size}\n".encode("ascii")
    )
    return git_blob_sha1(pointer)


def json_load_unique(path: Path) -> Any:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate)


def tracked_files(source: Path) -> list[Path]:
    return [Path(name) for name in git(source, "ls-files", "-z").split("\0") if name]


def classify_license(path: Path, text: str) -> str:
    lowered = text.lower()
    if "permission is hereby granted, free of charge" in lowered and "the software is provided" in lowered:
        return "MIT"
    if "apache license" in lowered and "version 2.0" in lowered:
        return "Apache-2.0"
    if "redistribution and use in source and binary forms" in lowered:
        return "BSD-family"
    raise RuntimeError(f"license identity is unknown: {path}")


def license_record(path: Path, root: Path, declared: str) -> dict[str, Any]:
    value = declared.strip()
    if not value or value.lower() in {"unknown", "none", "null", "-"}:
        raise RuntimeError(f"license declaration is unknown: {path}")
    return {
        "license": value,
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def license_records(root: Path, files: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in files:
        if "license" not in path.name.lower() and path.name.lower() not in {"notice", "copying"}:
            continue
        try:
            declared = classify_license(path, path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, RuntimeError) as error:
            records.append({"path": path.relative_to(root).as_posix(), "status": "UNKNOWN", "reason": str(error)})
        else:
            records.append(license_record(path, root, declared))
    return records


def regular_files(root: Path) -> list[Path]:
    """Enumerate materialized fixed-snapshot files; symlinks are rejected."""
    # Validate the one transport subtree before walking so a malformed root
    # cache can never be silently skipped by the inventory loop below.
    transport_metadata(root)
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        relative_parts = path.relative_to(root).parts
        if (relative_parts == (".cache",) or len(relative_parts) >= 2 and relative_parts[:2] == (".cache", "huggingface")):
            continue
        if any(part in {".cache", ".git"} for part in relative_parts):
            raise RuntimeError(f"snapshot contains unauthenticated metadata path: {path}")
        if path.is_dir() and not path.is_symlink():
            continue
        if path.is_symlink():
            resolved = path.resolve(strict=False)
            if not path.exists():
                raise RuntimeError(f"dangling snapshot symlink: {path}")
            if root.resolve() not in resolved.parents:
                raise RuntimeError(f"snapshot symlink escapes root: {path}")
            raise RuntimeError(f"snapshot symlinks are not accepted: {path}")
        if not path.is_file():
            raise RuntimeError(f"snapshot member is not a regular file: {path}")
        files.append(path)
    return files


def transport_metadata(root: Path) -> list[str]:
    """Return the only cache area excluded from the authenticated payload tree."""
    cache = root / ".cache"
    if not cache.exists() and not cache.is_symlink():
        return []
    if cache.is_symlink() or not cache.is_dir() or (cache / "huggingface").is_symlink() or not (cache / "huggingface").is_dir():
        raise RuntimeError("snapshot .cache must contain only root .cache/huggingface transport metadata")
    excluded = []
    for path in sorted((cache / "huggingface").rglob("*")):
        excluded.append(path.relative_to(root).as_posix())
    return [".cache/huggingface"] + excluded


def find_hf_license(snapshot: Path) -> dict[str, Any]:
    authenticated_files = regular_files(snapshot)
    candidates = [path for path in authenticated_files if path.name.lower() in {"license", "license.txt", "license.md"}]
    for path in sorted(candidates):
        try:
            identity = classify_license(path, path.read_text(encoding="utf-8", errors="replace"))
        except RuntimeError:
            continue
        return license_record(path, snapshot, identity)
    readmes = [path for path in authenticated_files if path.relative_to(snapshot).as_posix() == "README.md"]
    if readmes:
        readme = readmes[0]
        for line in readme.read_text(encoding="utf-8", errors="replace").splitlines()[:100]:
            key, separator, value = line.partition(":")
            if separator and key.strip().lower() in {"license", "license_spdx"}:
                return license_record(readme, snapshot, value)
    raise RuntimeError("HF bundle has no identifiable primary weight license")


def source_license(source: Path) -> dict[str, Any]:
    path = source / "LICENSE"
    if not path.is_file():
        raise RuntimeError("official ACE-Step source LICENSE is missing")
    return license_record(path, source, classify_license(path, path.read_text(encoding="utf-8", errors="replace")))


def source_identity(
    source: Path,
    git_runner: Any = git,
    file_identity_runner: Any = stream_file_identity,
) -> dict[str, Any]:
    if not (source / ".git").exists():
        raise RuntimeError("official source checkout lacks .git metadata")
    resolved_revision = git_runner(source, "rev-parse", "HEAD")
    remote = git_runner(source, "remote", "get-url", "origin").removesuffix("/").removesuffix(".git")
    dirty = git_runner(source, "status", "--porcelain", "--untracked-files=all")
    if resolved_revision != SOURCE_REVISION:
        raise RuntimeError(f"official source resolved revision mismatch: {resolved_revision}")
    if remote != SOURCE_REPOSITORY.removesuffix("/").removesuffix(".git"):
        raise RuntimeError(f"official source origin mismatch: {remote}")
    if dirty:
        raise RuntimeError(f"official source checkout is dirty: {dirty!r}")
    index_entries: dict[str, tuple[str, str]] = {}
    for record in git_runner(source, "ls-files", "-s", "-z").split("\0"):
        if not record:
            continue
        try:
            metadata, role = record.split("\t", 1)
            mode, object_id, stage = metadata.split()
        except ValueError as error:
            raise RuntimeError(f"official source index record is malformed: {record!r}") from error
        if mode not in {"100644", "100755"} or stage != "0":
            raise RuntimeError(f"official source has non-regular or staged tracked entry: {role}")
        role_path_value = Path(role)
        if role_path_value.is_absolute() or ".." in role_path_value.parts:
            raise RuntimeError(f"official source index contains unsafe path: {role!r}")
        tracked_path = source / role_path_value
        if tracked_path.is_symlink() or not tracked_path.is_file():
            raise RuntimeError(f"official source tracked file is missing or non-regular: {role}")
        if role in index_entries:
            raise RuntimeError(f"official source index has duplicate tracked role: {role}")
        index_entries[role] = (mode, object_id)
    role_blobs: list[dict[str, str]] = []
    for role, expected_blob in SOURCE_ROLE_BLOBS.items():
        role_path = source / role
        if not role_path.is_file() or role_path.is_symlink():
            raise RuntimeError(f"required official source role is missing or symlinked: {role}")
        mode_object = index_entries.get(role)
        if mode_object is None:
            raise RuntimeError(f"required official source role is not tracked in the index: {role}")
        mode, index_blob = mode_object
        if index_blob != expected_blob:
            raise RuntimeError(f"official source index blob mismatch: {role}: {index_blob} != {expected_blob}")
        head_blob = git_runner(source, "rev-parse", f"HEAD:{role}")
        if head_blob != expected_blob:
            raise RuntimeError(f"official source role blob mismatch: {role}: {head_blob} != {expected_blob}")
        working_blob = file_identity_runner(role_path)["git_blob_sha1"]
        if working_blob != expected_blob:
            raise RuntimeError(f"official source working file blob mismatch: {role}: {working_blob} != {expected_blob}")
        role_blobs.append({"path": role, "mode": mode, "git_blob_sha1": expected_blob})
    return {"resolved_revision": resolved_revision, "origin": remote, "worktree_status": "CLEAN", "required_source_roles": role_blobs}


def source_role_evidence(source: Path) -> list[dict[str, Any]]:
    composition: list[dict[str, Any]] = []
    for role, markers in SOURCE_ROLE_MARKERS.items():
        role_path = source / role
        role_text = role_path.read_text(encoding="utf-8", errors="replace")
        missing = [marker for marker in markers if marker.lower() not in role_text.lower()]
        if missing:
            raise RuntimeError(f"official source role markers missing from {role}: {missing!r}")
        composition.append({"path": role, "required_markers": list(markers), "status": "ROLE_BOUND"})
    return composition


def source_inventory(source: Path) -> dict[str, Any]:
    identity = source_identity(source)
    resolved_revision = identity["resolved_revision"]
    remote = identity["origin"]
    tracked = tracked_files(source)
    regular = [path for path in tracked if (source / path).is_file() and not (source / path).is_symlink()]
    if not regular:
        raise RuntimeError("official source has no tracked regular files")
    text_files = [
        path
        for path in regular
        if path.suffix.lower() in {".py", ".toml", ".md", ".yaml", ".yml", ".json", ".txt"}
    ]
    composition = source_role_evidence(source)
    pyproject_path = source / "pyproject.toml"
    lock_path = source / "uv.lock"
    if not pyproject_path.is_file() or not lock_path.is_file():
        raise RuntimeError("official source pyproject.toml and uv.lock are both required")
    project = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    dependency_text = [str(value) for value in project.get("project", {}).get("dependencies", [])]
    for dependency in ("transformers", "diffusers"):
        if not any(dependency in value.lower() for value in dependency_text):
            raise RuntimeError(f"official pyproject lacks dependency constraint: {dependency}")
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    packages = lock.get("package", [])
    locked: dict[str, Any] = {}
    for dependency in DEPENDENCIES:
        matches = [package for package in packages if package.get("name") == dependency]
        if len(matches) != 1:
            raise RuntimeError(f"uv.lock must contain exactly one {dependency} package")
        package = matches[0]
        artifacts = []
        if isinstance(package.get("sdist"), dict):
            artifacts.append(package["sdist"])
        artifacts.extend(item for item in package.get("wheels", []) if isinstance(item, dict))
        if not package.get("version") or not artifacts or any("hash" not in item for item in artifacts):
            raise RuntimeError(f"uv.lock lacks complete hashes for {dependency}")
        locked[dependency] = {
            "version": package["version"],
            "source": package.get("source"),
            "artifacts": artifacts,
        }
    files = [
        {"path": path.as_posix(), "bytes": (source / path).stat().st_size, "sha256": sha256(source / path)}
        for path in sorted(regular)
    ]
    all_packages = []
    for package in packages:
        if not isinstance(package, dict) or not package.get("name") or not package.get("version"):
            raise RuntimeError("uv.lock contains an incomplete package record")
        all_packages.append({"name": package["name"], "version": package["version"], "source": package.get("source")})
    return {
        "repository": SOURCE_REPOSITORY,
        "revision": SOURCE_REVISION,
        "resolved_revision": resolved_revision,
        "origin": remote,
        "worktree_status": identity["worktree_status"],
        "required_source_roles": identity["required_source_roles"],
        "license": source_license(source),
        "scope": "all_tracked_regular_files",
        "files": files,
        "composition_evidence": composition,
        "dependency_lock": {
            "path": "uv.lock",
            "sha256": sha256(lock_path),
            "pyproject_sha256": sha256(pyproject_path),
            "project_dependencies": dependency_text,
            "packages_all": all_packages,
            "packages": locked,
        },
    }


def tensor_metadata(handle: Any, name: str) -> dict[str, Any]:
    view = handle.get_slice(name)
    shape = [int(axis) for axis in view.get_shape()]
    dtype = str(view.get_dtype()).removeprefix("torch.")
    dtype = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}.get(dtype, dtype)
    elements = 1
    for axis in shape:
        elements *= axis
    return {"name": name, "shape": shape, "dtype": dtype, "elements": elements}


def json_evidence(path: Path, root: Path) -> dict[str, Any]:
    try:
        value = json_load_unique(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"canonical JSON parse failed for {path.relative_to(root)}: {error}") from error
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "status": "PARSED_CANONICAL_JSON",
        "type": type(value).__name__,
        "top_level_keys": sorted(value) if isinstance(value, dict) else None,
        "top_level_count": len(value) if isinstance(value, (dict, list)) else 1,
        "raw": value if path.name.endswith("index.json") else None,
    }


def config_semantics(path: Path, root: Path) -> dict[str, Any]:
    try:
        value = json_load_unique(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"component config parse failed for {path.relative_to(root)}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"component config must be a JSON object: {path.relative_to(root)}")
    model_type = value.get("model_type")
    architectures = value.get("architectures")
    if model_type is not None and not isinstance(model_type, str):
        raise RuntimeError(f"component config model_type is not a string: {path.relative_to(root)}")
    if architectures is not None and (
        not isinstance(architectures, list) or not architectures or not all(isinstance(item, str) for item in architectures)
    ):
        raise RuntimeError(f"component config architectures are not a non-empty string list: {path.relative_to(root)}")
    if model_type is None and architectures is None:
        raise RuntimeError(f"component config has no model_type or architectures: {path.relative_to(root)}")
    scalar_evidence = {
        key: value[key]
        for key in ("model_type", "architectures", "sample_rate", "sampling_rate", "latent_rate", "latent_sr", "audio_channels")
        if key in value and isinstance(value[key], (str, int, float, bool, list))
    }
    return {"path": path.relative_to(root).as_posix(), "model_type": model_type, "architectures": architectures, "scalar_evidence": scalar_evidence}


def strict_component_shard(value: Any) -> str:
    if not isinstance(value, str) or "\x00" in value or "\\" in value:
        raise ValueError("component shard must be a plain string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.name != value or not value.endswith(".safetensors"):
        raise ValueError(f"component shard is not a direct safetensors basename: {value!r}")
    return value


def inventory_component(snapshot: Path, component: str) -> dict[str, Any]:
    root = snapshot / component
    if not root.is_dir():
        raise RuntimeError(f"missing canonical ACE-Step component: {component}")
    files = regular_files(root)
    json_packets = [json_evidence(path, snapshot) for path in files if path.suffix.lower() == ".json"]
    config_paths = [path for path in files if path.name == "config.json"]
    if len(config_paths) != 1:
        raise RuntimeError(f"component lacks config.json: {component}")
    semantics = config_semantics(config_paths[0], snapshot)
    if component in TEXT_COMPONENTS and not any("tokenizer" in path.name.lower() for path in files):
        raise RuntimeError(f"text component lacks tokenizer companion: {component}")
    weights = sorted(path for path in files if path.suffix == ".safetensors")
    if not weights:
        raise RuntimeError(f"component has no safetensors weights: {component}")
    indexes = sorted(path for path in files if path.name.endswith(".index.json"))
    weight_map: dict[str, str] | None = None
    if indexes:
        if len(indexes) != 1:
            raise RuntimeError(f"component has multiple safetensors indexes: {component}")
        index = json_load_unique(indexes[0])
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(f"component index has no weight_map: {component}")
        if any(not isinstance(key, str) for key in weight_map) or any(not isinstance(value, str) for value in weight_map.values()):
            raise RuntimeError(f"component index weight_map keys/values must be strings: {component}")
        try:
            indexed = {strict_component_shard(value) for value in weight_map.values()}
        except ValueError as error:
            raise RuntimeError(f"component index shard path rejected: {component}: {error}") from error
        actual = {path.name for path in weights}
        if indexed != actual:
            raise RuntimeError(f"component index/shard mismatch: {component}")
    elif len(weights) != 1:
        raise RuntimeError(f"sharded component lacks a safetensors index: {component}")
    shards = []
    tensors = []
    seen: set[str] = set()
    for path in weights:
        local: set[str] = set()
        with safe_open(str(path), framework="pt") as handle:
            for name in sorted(handle.keys()):
                if name in seen or name in local:
                    raise RuntimeError(f"duplicate tensor name in component {component}: {name}")
                local.add(name)
                record = tensor_metadata(handle, name)
                record["component"] = component
                record["shard"] = path.relative_to(snapshot).as_posix()
                tensors.append(record)
        seen.update(local)
        shards.append({"path": path.relative_to(snapshot).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path), "tensor_count": len(local)})
    if weight_map is not None and set(weight_map) != {item["name"] for item in tensors}:
        raise RuntimeError(f"component weight_map keys do not match tensor names: {component}")
    if weight_map is not None and any(weight_map[item["name"]] != Path(item["shard"]).name for item in tensors):
        raise RuntimeError(f"component tensor-to-shard mapping mismatch: {component}")
    return {
        "name": component,
        "files": [{"path": path.relative_to(snapshot).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files],
        "license_records": license_records(snapshot, files),
        "json": json_packets,
        "config_semantics": semantics,
        "shards": shards,
        "tensors": tensors,
    }


def container_manifest(value: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            if not bool(torch.isfinite(value).all().item()):
                raise RuntimeError(f"non-finite tensor in PyTorch container: {path}")
        return [{"path": path or "<root>", "shape": [int(axis) for axis in value.shape], "dtype": str(value.dtype), "elements": int(value.numel()), "finite": True}]
    if isinstance(value, dict):
        records: list[dict[str, Any]] = []
        for key in sorted(value, key=str):
            records.extend(container_manifest(value[key], f"{path}.{key}" if path else str(key)))
        return records
    if isinstance(value, (list, tuple)):
        records = []
        for index, item in enumerate(value):
            records.extend(container_manifest(item, f"{path}[{index}]"))
        return records
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite scalar in PyTorch container: {path}")
    if value is None or isinstance(value, (str, int, float, bool)):
        return []
    raise RuntimeError(f"unsupported object in safe PyTorch container at {path}: {type(value).__name__}")


def inspect_container(path: Path, snapshot: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
        tensors = container_manifest(value)
    except Exception as error:
        raise RuntimeError(f"safe PyTorch container load failed for {path.relative_to(snapshot)}: {type(error).__name__}: {error}") from error
    if not tensors:
        raise RuntimeError(f"PyTorch container has no tensor manifest: {path.relative_to(snapshot)}")
    return {"path": path.relative_to(snapshot).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path), "safe_load": "weights_only=True", "tensors": tensors}


def server_tree(snapshot: Path, packet: Path, blockers: list[str]) -> dict[str, Any]:
    try:
        remote = json_load_unique(packet)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        blockers.append(f"server-tree packet parse failed: {error}")
        return {"status": "BLOCKED_PACKET_PARSE", "packet_sha256": sha256(packet) if packet.is_file() else None}
    if not isinstance(remote, dict):
        blockers.append("server-tree packet must be a JSON object")
        return {"status": "BLOCKED_PACKET_SCHEMA", "packet_sha256": sha256(packet)}
    allowed_packet_keys = {"repository", "requested_revision", "revision", "resolved_revision", "files"}
    unexpected_packet_keys = sorted(set(remote) - allowed_packet_keys)
    if unexpected_packet_keys:
        blockers.append(f"server-tree packet has unexpected top-level keys: {unexpected_packet_keys!r}")
    if (
        remote.get("repository") != UPSTREAM_REPOSITORY
        or remote.get("requested_revision") != UPSTREAM_REVISION
        or remote.get("revision") != UPSTREAM_REVISION
        or remote.get("resolved_revision") != UPSTREAM_REVISION
    ):
        blockers.append("server-tree packet repository/requested/resolved revision mismatch")
    rows = remote.get("files")
    if not isinstance(rows, list):
        blockers.append("server-tree packet files must be a list")
        rows = []
    excluded_transport = transport_metadata(snapshot)
    local = regular_files(snapshot)
    local_by_path = {path.relative_to(snapshot).as_posix(): path for path in local}
    remote_by_path: dict[str, dict[str, Any]] = {}
    for item in rows:
        if not isinstance(item, dict) or item.get("type") != "file":
            blockers.append("server-tree packet contains a non-file or malformed row")
            continue
        base_keys = {"path", "type", "size", "git_blob_sha1"}
        lfs_keys = {"lfs_sha256", "lfs_size", "lfs_pointer_sha1"}
        row_keys = set(item)
        if not (row_keys == base_keys or row_keys == base_keys | lfs_keys):
            blockers.append(f"server-tree packet row schema is not exact: {sorted(row_keys)!r}")
            continue
        path_value = item.get("path")
        if not isinstance(path_value, str):
            blockers.append("server-tree packet contains a non-string path")
            continue
        path = Path(path_value)
        if path.is_absolute() or "\\" in path_value or ".." in path.parts or path_value.startswith("./") or not path.name:
            blockers.append(f"server-tree packet unsafe path: {path_value!r}")
            continue
        if path_value in remote_by_path:
            blockers.append(f"server-tree packet duplicate path: {path_value}")
            continue
        remote_by_path[path_value] = item
    missing = sorted(set(local_by_path) - set(remote_by_path))
    extra = sorted(set(remote_by_path) - set(local_by_path))
    if missing or extra:
        blockers.append(f"server/local tree mismatch: missing={missing!r} extra={extra!r}")
    for path_value, item in remote_by_path.items():
        local_path = local_by_path.get(path_value)
        if local_path is None:
            continue
        identity = stream_file_identity(local_path)
        payload_size = identity["bytes"]
        actual_sha = identity["sha256"]
        actual_blob = identity["git_blob_sha1"]
        size = item.get("size")
        blob = item.get("git_blob_sha1")
        if isinstance(size, bool) or not isinstance(size, int) or size != payload_size:
            blockers.append(f"server/local size mismatch: {path_value}")
        if not isinstance(blob, str) or len(blob) != 40 or any(char not in "0123456789abcdef" for char in blob):
            blockers.append(f"server packet lacks canonical git blob SHA-1: {path_value}")
            continue
        is_lfs = set(item) == base_keys | lfs_keys
        if is_lfs:
            lfs_sha = item.get("lfs_sha256")
            lfs_size = item.get("lfs_size")
            pointer = item.get("lfs_pointer_sha1")
            if not isinstance(lfs_sha, str) or len(lfs_sha) != 64 or any(char not in "0123456789abcdef" for char in lfs_sha) or lfs_sha != actual_sha:
                blockers.append(f"server/local LFS payload SHA-256 mismatch: {path_value}")
            if isinstance(lfs_size, bool) or not isinstance(lfs_size, int) or lfs_size != payload_size:
                blockers.append(f"server/local LFS payload size mismatch: {path_value}")
            expected_pointer = lfs_pointer_sha1(actual_sha, payload_size)
            if pointer != expected_pointer:
                blockers.append(f"server packet LFS pointer blob mismatch: {path_value}")
            if blob != expected_pointer:
                blockers.append(f"server packet git blob is not canonical LFS pointer identity: {path_value}")
        elif blob != actual_blob:
            blockers.append(f"server/local git blob SHA-1 mismatch: {path_value}")
    return {
        "status": "MATCHED" if not blockers else "MISMATCH",
        "repository": remote.get("repository"),
        "requested_revision": remote.get("requested_revision"),
        "revision": remote.get("revision"),
        "resolved_revision": remote.get("resolved_revision"),
        "packet_sha256": sha256(packet),
        "missing_local": missing,
        "unexpected_local": extra,
        "authenticated_file_count": len(remote_by_path),
        "excluded_transport_metadata": excluded_transport,
    }


def validate_snapshot_tree(snapshot: Path) -> list[dict[str, Any]]:
    transport_metadata(snapshot)
    root_dirs = {path.name for path in snapshot.iterdir() if path.is_dir() and not path.is_symlink() and path.name != ".cache"}
    unexpected = sorted(root_dirs - set(COMPONENTS))
    if unexpected:
        raise RuntimeError(f"snapshot includes unrequested/community component directories: {unexpected}")
    root_files = [path for path in regular_files(snapshot) if path.parent == snapshot]
    if not any(path.name == "config.json" for path in root_files):
        raise RuntimeError("HF bundle lacks root config.json companion")
    if not any(path.name == "README.md" for path in root_files):
        raise RuntimeError("HF bundle lacks root README.md companion")
    silence = [
        path
        for path in regular_files(snapshot)
        if path.relative_to(snapshot).as_posix() == CANONICAL_SILENCE_PATH
    ]
    if len(silence) != 1:
        raise RuntimeError(f"HF bundle must contain exactly one canonical {CANONICAL_SILENCE_PATH}")
    pt_files = [path for path in regular_files(snapshot) if path.suffix.lower() == ".pt"]
    if pt_files != silence:
        raise RuntimeError(f"HF bundle contains an extra, nested, duplicate, or misplaced PyTorch container; canonical path is {CANONICAL_SILENCE_PATH}")
    return [{"path": path.relative_to(snapshot).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(root_files)]


def inspect(snapshot: Path, source: Path, output: Path, tree_packet: Path) -> None:
    snapshot = snapshot.resolve()
    source = source.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tree_blockers: list[str] = []
    tree = server_tree(snapshot, tree_packet, tree_blockers)
    if tree["status"] != "MATCHED":
        raise RuntimeError("server-tree packet failed independent object-identity validation: " + "; ".join(tree_blockers))
    hf_license = find_hf_license(snapshot)
    root_companions = validate_snapshot_tree(snapshot)
    root_json = [json_evidence(path, snapshot) for path in regular_files(snapshot) if path.parent == snapshot and path.suffix.lower() == ".json"]
    components = [inventory_component(snapshot, component) for component in COMPONENTS]
    if any(not item["license_records"] for item in components):
        tree_blockers.append("component repository-level licenses are unauthenticated")
    containers = [inspect_container(path, snapshot) for path in regular_files(snapshot) if path.suffix.lower() == ".pt"]
    silence = [record for record in containers if record["path"] == CANONICAL_SILENCE_PATH]
    if len(silence) != 1:
        raise RuntimeError(f"canonical bundle must contain exactly one safe {CANONICAL_SILENCE_PATH} container")
    sources = source_inventory(source)
    tensor_inventory = {"components": [{"name": item["name"], "tensor_count": len(item["tensors"]), "tensors": item["tensors"]} for item in components], "containers": containers}
    component_inventory = {"components": components}
    companion_inventory = {"root_files": root_companions, "json": root_json}
    source_inventory_payload = sources
    (output / "tensor-inventory.json").write_text(json.dumps(tensor_inventory, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "component-inventory.json").write_text(json.dumps(component_inventory, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "companion-inventory.json").write_text(json.dumps(companion_inventory, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "source-inventory.json").write_text(json.dumps(source_inventory_payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    packets = {name: {"bytes": (output / name).stat().st_size, "sha256": sha256(output / name)} for name in ("tensor-inventory.json", "component-inventory.json", "companion-inventory.json", "source-inventory.json")}
    blockers = tree_blockers + [
        "native ACE-Step composition/runtime is not implemented",
        "dependency licenses require separate audit",
        "dataset/training provenance is unauthenticated",
    ]
    manifest = {
        "format": FORMAT,
        "status": "BLOCKED",
        "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
        "cpu_status": "UNSUPPORTED",
        "metal_status": "BLOCKED_BY_CPU",
        "parity_status": "NOT_RUN",
        "publication": "NO_UPLOAD",
        "upstream": {"repository": UPSTREAM_REPOSITORY, "revision": UPSTREAM_REVISION, "license": hf_license, "bundle_scope": list(COMPONENTS)},
        "server_tree": tree,
        "components": [{"name": item["name"], "shards": item["shards"], "tensor_count": len(item["tensors"])} for item in components],
        "pytorch_containers": containers,
        "canonical_silence_latent": {"path": CANONICAL_SILENCE_PATH, "evidence": silence[0]},
        "root_companions": root_companions,
        "official_source": sources,
        "license_evidence": {
            "hf_bundle": hf_license,
            "components": [{"name": item["name"], "license_records": item["license_records"], "basis": "component-local declaration only; HF root license is not inherited", "independent_component_file": bool(item["license_records"])} for item in components],
            "root_companions": {"license": "NOT_INHERITED_FROM_HF_MODEL", "basis": "companion license requires independent declaration"},
            "official_source": sources["license"],
            "dependencies": {name: {"basis": "official source uv.lock artifact hashes", "version": sources["dependency_lock"]["packages"][name]["version"], "license_status": "UNREVIEWED_BLOCKER"} for name in DEPENDENCIES},
        },
        "composition_contract": "official source evidence records the component registry, 5Hz LM, text embedding, turbo DiT/flow guidance, VAE and 48kHz markers; no native composition or generation claim is made",
        "license_contract": "HF bundle, source, and uv-locked dependency identities are recorded separately; absent or ambiguous declarations block",
        "packets": packets,
        "dataset_training_provenance": "UNAUTHENTICATED_BLOCKER",
        "blockers": sorted(set(blockers)),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def write_blocked(output: Path, error: Exception, tree_packet: Path | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    packet = None
    if tree_packet is not None and tree_packet.is_file():
        packet = {"path": str(tree_packet), "bytes": tree_packet.stat().st_size, "sha256": sha256(tree_packet)}
    payload = {
        "format": FORMAT,
        "status": "BLOCKED",
        "error_type": type(error).__name__,
        "reason": str(error),
        "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
        "cpu_status": "UNSUPPORTED",
        "metal_status": "BLOCKED_BY_CPU",
        "parity_status": "NOT_RUN",
        "publication": "NO_UPLOAD",
        "upstream": {"repository": UPSTREAM_REPOSITORY, "revision": UPSTREAM_REVISION},
        "server_tree_packet": packet,
        "dataset_training_provenance": "UNAUTHENTICATED_BLOCKER",
        "blockers": [str(error), "native composition/runtime is not implemented", "dependency licenses require separate audit", "dataset/training provenance is unauthenticated"],
    }
    (output / "manifest.json").write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "blocker.txt").write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")


def self_test() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    assert hashlib.sha256(b"abc").hexdigest() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert git_blob_sha1(b"abc") != git_blob_sha1(b"abd")
    assert lfs_pointer_sha1("a" * 64, 3) != lfs_pointer_sha1("b" * 64, 3)
    assert len(UPSTREAM_REVISION) == 40 and len(SOURCE_REVISION) == 40
    assert "weights_only=True" in source and ("weights_only=" + "False") not in source
    assert "stream_file_identity" in source
    assert "INSPECTION_ONLY" in source and "safe_open" in source and "uv.lock" in source
    class BoundedReader:
        def __init__(self, payload: bytes, reads: list[int]) -> None:
            self.payload = payload
            self.offset = 0
            self.reads = reads

        def __enter__(self) -> "BoundedReader":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def read(self, size: int) -> bytes:
            self.reads.append(size)
            chunk = self.payload[self.offset : self.offset + size]
            self.offset += len(chunk)
            return chunk

    class BoundedPath:
        def __init__(self, payload: bytes, reads: list[int]) -> None:
            self.payload = payload
            self.reads = reads

        def stat(self) -> Any:
            return type("Stat", (), {"st_size": len(self.payload)})()

        def open(self, _mode: str) -> BoundedReader:
            return BoundedReader(self.payload, self.reads)

    bounded_reads: list[int] = []
    bounded_identity = stream_file_identity(BoundedPath(b"bounded-stream-fixture", bounded_reads), block_size=3)
    assert max(bounded_reads) <= 3 and len(bounded_reads) > 1
    assert bounded_identity["sha256"] == hashlib.sha256(b"bounded-stream-fixture").hexdigest()
    with tempfile.TemporaryDirectory(prefix="vokra-ace-step-inspect-") as directory:
        root = Path(directory)
        fake_source = root.parent / f"source-{root.name}"
        fake_source.mkdir()
        (fake_source / ".git").mkdir()
        for role in SOURCE_ROLE_BLOBS:
            role_path = fake_source / role
            role_path.parent.mkdir(parents=True, exist_ok=True)
            role_path.write_text("\n".join(SOURCE_ROLE_MARKERS.get(role, ())) or "fixture", encoding="utf-8")
        extra_source_file = fake_source / "extra-tracked.py"
        extra_source_file.write_text("tracked\n", encoding="utf-8")
        fake_index = "\0".join(f"100644 {blob} 0\t{role}" for role, blob in SOURCE_ROLE_BLOBS.items()) + "\0"
        fake_index += "100644 " + ("e" * 40) + " 0\textra-tracked.py\0"

        def fake_git(_source: Path, *args: str) -> str:
            if args == ("rev-parse", "HEAD"):
                return SOURCE_REVISION
            if args == ("remote", "get-url", "origin"):
                return SOURCE_REPOSITORY
            if args == ("status", "--porcelain", "--untracked-files=all"):
                return ""
            if args == ("ls-files", "-s", "-z"):
                return fake_index
            if args[0:1] == ("rev-parse",) and args[1].startswith("HEAD:"):
                return SOURCE_ROLE_BLOBS[args[1][5:]]
            raise AssertionError(args)

        def fake_file_identity(path: Path) -> dict[str, str]:
            return {"git_blob_sha1": SOURCE_ROLE_BLOBS[path.relative_to(fake_source).as_posix()]}

        assert source_identity(fake_source, fake_git, fake_file_identity)["worktree_status"] == "CLEAN"
        assert len(source_role_evidence(fake_source)) == len(SOURCE_ROLE_MARKERS)
        wrong_role = fake_source / "acestep/handler.py"
        wrong_role.write_text("GenerationParams\n", encoding="utf-8")
        try:
            source_role_evidence(fake_source)
        except RuntimeError as error:
            assert "handler.py" in str(error)
        else:
            raise AssertionError("marker from the wrong source role was accepted")
        wrong_role.write_text("\n".join(SOURCE_ROLE_MARKERS["acestep/handler.py"]) + "\n", encoding="utf-8")
        extra_source_file.unlink()
        try:
            source_identity(fake_source, fake_git, fake_file_identity)
        except RuntimeError as error:
            assert "tracked file is missing" in str(error)
        else:
            raise AssertionError("missing non-role tracked source file was accepted")
        extra_source_file.write_text("tracked\n", encoding="utf-8")
        for label, override, needle in (
            ("dirty", lambda args: " M acestep/inference.py", "dirty"),
            ("origin", lambda args: "https://example.invalid/other", "origin mismatch"),
            ("head", lambda args: "0" * 40, "revision mismatch"),
            ("blob", lambda args: "f" * 40, "role blob mismatch"),
            ("index-mode", lambda args: fake_index.replace("100644", "120000", 1), "non-regular"),
            ("working-bytes", lambda args: "f" * 40, "working file blob mismatch"),
        ):
            def spoofed_git(_source: Path, *args: str) -> str:
                if args == ("status", "--porcelain", "--untracked-files=all") and label == "dirty":
                    return override(args)
                if args == ("remote", "get-url", "origin") and label == "origin":
                    return override(args)
                if args == ("rev-parse", "HEAD") and label == "head":
                    return override(args)
                if args == ("ls-files", "-s", "-z") and label == "index-mode":
                    return override(args)
                if args[0:1] == ("rev-parse",) and args[1].startswith("HEAD:") and label == "blob":
                    return override(args)
                return fake_git(_source, *args)

            def spoofed_file_identity(path: Path) -> dict[str, str]:
                if label == "working-bytes" and path.name == "inference.py":
                    return {"git_blob_sha1": override(())}
                return fake_file_identity(path)

            try:
                source_identity(fake_source, spoofed_git, spoofed_file_identity)
            except RuntimeError as error:
                assert needle in str(error), (label, error)
            else:
                raise AssertionError(f"source {label} spoof was accepted")
        for number, component in enumerate(COMPONENTS, 1):
            path = root / component
            path.mkdir()
            (path / "config.json").write_text(json.dumps({"model_type": f"fixture-{number}"}) + "\n", encoding="utf-8")
            if component in TEXT_COMPONENTS:
                (path / "tokenizer.json").write_text("{}\n", encoding="utf-8")
            shard = path / "model.safetensors"
            save_file({f"layer.{number}": torch.tensor([float(number)], dtype=torch.bfloat16)}, str(shard))
            result = inventory_component(root, component)
            assert result["tensors"][0]["dtype"] == "bfloat16"
            assert result["tensors"][0]["shape"] == [1]
        (root / "config.json").write_text("{}\n", encoding="utf-8")
        (root / "README.md").write_text("license: mit\n", encoding="utf-8")
        target = root / "target.bin"
        target.write_bytes(b"cache target")
        cache_link = root / "cache-link.bin"
        cache_link.symlink_to(target)
        try:
            regular_files(root)
        except RuntimeError as error:
            assert "symlink" in str(error)
        else:
            raise AssertionError("snapshot symlink was accepted")
        cache_link.unlink()
        escape_link = root / "escape-link.bin"
        outside = root.parent / "outside.bin"
        outside.write_bytes(b"outside")
        escape_link.symlink_to(outside)
        try:
            regular_files(root)
        except RuntimeError as error:
            assert "escapes root" in str(error)
        else:
            raise AssertionError("escaping snapshot symlink was accepted")
        escape_link.unlink()
        outside.unlink()
        dangling_link = root / "dangling-link.bin"
        dangling_link.symlink_to(root / "missing.bin")
        try:
            regular_files(root)
        except RuntimeError as error:
            assert "dangling" in str(error)
        else:
            raise AssertionError("dangling snapshot symlink was accepted")
        dangling_link.unlink()
        transport = root / ".cache" / "huggingface"
        transport.mkdir(parents=True)
        (transport / "metadata.json").write_text("{}\n", encoding="utf-8")
        assert ".cache/huggingface/metadata.json" in transport_metadata(root)
        assert all(path.relative_to(root).as_posix() != ".cache/huggingface/metadata.json" for path in regular_files(root))
        for shape in ("cache-file", "cache-symlink", "transport-file", "transport-symlink"):
            malformed = root.parent / f"{shape}-{root.name}"
            malformed.mkdir()
            malformed_cache = malformed / ".cache"
            if shape.startswith("cache-"):
                if shape.endswith("file"):
                    malformed_cache.write_bytes(b"not-a-directory")
                else:
                    malformed_cache.symlink_to(root.parent / "outside-cache", target_is_directory=True)
            else:
                malformed_cache.mkdir()
                transport_path = malformed_cache / "huggingface"
                if shape.endswith("file"):
                    transport_path.write_bytes(b"not-a-directory")
                else:
                    transport_path.symlink_to(root.parent / "outside-transport", target_is_directory=True)
            try:
                regular_files(malformed)
            except RuntimeError as error:
                assert ".cache" in str(error)
            else:
                raise AssertionError(f"malformed {shape} transport cache was accepted")
        nested_cache = root / COMPONENTS[0] / ".cache"
        nested_cache.mkdir()
        try:
            regular_files(root)
        except RuntimeError as error:
            assert "metadata path" in str(error)
        else:
            raise AssertionError("nested component cache was accepted")
        nested_cache.rmdir()
        canonical_silence = root / CANONICAL_SILENCE_PATH
        torch.save({"latent": torch.zeros((1, 2), dtype=torch.float32)}, canonical_silence)
        assert len(validate_snapshot_tree(root)) == 3
        assert CANONICAL_SILENCE_PATH == "acestep-v15-turbo/silence_latent.pt"
        misplaced_root_silence = root / "silence_latent.pt"
        misplaced_root_silence.write_bytes(b"not the canonical silence container")
        try:
            validate_snapshot_tree(root)
        except RuntimeError as error:
            assert "canonical" in str(error) or "misplaced" in str(error)
        else:
            raise AssertionError("root silence container was accepted")
        misplaced_root_silence.unlink()
        misplaced_name = root / COMPONENTS[0] / "silence_latent_wrong.pt"
        misplaced_name.write_bytes(b"wrong suffix")
        try:
            validate_snapshot_tree(root)
        except RuntimeError as error:
            assert "canonical" in str(error)
        else:
            raise AssertionError("misnamed silence container was accepted")
        misplaced_name.unlink()
        packet_rows = []
        for path in regular_files(root):
            identity = stream_file_identity(path, block_size=2)
            packet_rows.append({"path": path.relative_to(root).as_posix(), "type": "file", "size": identity["bytes"], "git_blob_sha1": identity["git_blob_sha1"]})
        valid_tree = root.parent / "valid-tree.json"
        valid_tree.write_text(json.dumps({"repository": UPSTREAM_REPOSITORY, "requested_revision": UPSTREAM_REVISION, "revision": UPSTREAM_REVISION, "resolved_revision": UPSTREAM_REVISION, "files": packet_rows}), encoding="utf-8")
        tree_blockers: list[str] = []
        assert server_tree(root, valid_tree, tree_blockers)["status"] == "MATCHED"
        omitted_type_rows = [dict(item) for item in packet_rows]
        omitted_type_rows[0].pop("type")
        omitted_type_tree = root.parent / "omitted-type-tree.json"
        omitted_type_tree.write_text(json.dumps({"repository": UPSTREAM_REPOSITORY, "requested_revision": UPSTREAM_REVISION, "revision": UPSTREAM_REVISION, "resolved_revision": UPSTREAM_REVISION, "files": omitted_type_rows}), encoding="utf-8")
        omitted_type_blockers: list[str] = []
        assert server_tree(root, omitted_type_tree, omitted_type_blockers)["status"] == "MISMATCH"
        assert any("malformed row" in blocker for blocker in omitted_type_blockers)
        partial_lfs_rows = [dict(item) for item in packet_rows]
        partial_lfs_rows[0]["lfs_sha256"] = "0" * 64
        partial_lfs_tree = root.parent / "partial-lfs-tree.json"
        partial_lfs_tree.write_text(json.dumps({"repository": UPSTREAM_REPOSITORY, "requested_revision": UPSTREAM_REVISION, "revision": UPSTREAM_REVISION, "resolved_revision": UPSTREAM_REVISION, "files": partial_lfs_rows}), encoding="utf-8")
        partial_lfs_blockers: list[str] = []
        assert server_tree(root, partial_lfs_tree, partial_lfs_blockers)["status"] == "MISMATCH"
        assert any("schema is not exact" in blocker for blocker in partial_lfs_blockers)
        spoofed_rows = [dict(item) for item in packet_rows]
        spoofed_rows[0]["git_blob_sha1"] = "0" * 40
        spoofed_tree = root.parent / "spoofed-tree.json"
        spoofed_tree.write_text(json.dumps({"repository": UPSTREAM_REPOSITORY, "requested_revision": UPSTREAM_REVISION, "revision": UPSTREAM_REVISION, "resolved_revision": UPSTREAM_REVISION, "files": spoofed_rows}), encoding="utf-8")
        spoof_blockers: list[str] = []
        assert server_tree(root, spoofed_tree, spoof_blockers)["status"] == "MISMATCH"
        assert any("git blob SHA-1 mismatch" in blocker for blocker in spoof_blockers)
        lfs_rows = [dict(item) for item in packet_rows]
        lfs_rows[0].update({"lfs_sha256": "0" * 64, "lfs_size": lfs_rows[0]["size"], "lfs_pointer_sha1": "0" * 40})
        lfs_tree = root.parent / "lfs-spoof-tree.json"
        lfs_tree.write_text(json.dumps({"repository": UPSTREAM_REPOSITORY, "requested_revision": UPSTREAM_REVISION, "revision": UPSTREAM_REVISION, "resolved_revision": UPSTREAM_REVISION, "files": lfs_rows}), encoding="utf-8")
        lfs_blockers: list[str] = []
        assert server_tree(root, lfs_tree, lfs_blockers)["status"] == "MISMATCH"
        assert any("LFS payload SHA-256 mismatch" in blocker for blocker in lfs_blockers)
        traversal_tree = root.parent / "traversal-tree.json"
        traversal_tree.write_text(json.dumps({"repository": UPSTREAM_REPOSITORY, "requested_revision": UPSTREAM_REVISION, "revision": UPSTREAM_REVISION, "resolved_revision": UPSTREAM_REVISION, "files": [{"path": "../escape", "type": "file", "size": 1, "git_blob_sha1": "0" * 40}]}), encoding="utf-8")
        traversal_blockers: list[str] = []
        server_tree(root, traversal_tree, traversal_blockers)
        assert any("unsafe path" in blocker for blocker in traversal_blockers)
        duplicate_silence = root / "vae" / "silence_latent.pt"
        duplicate_silence.write_bytes(b"not an authenticated container")
        try:
            validate_snapshot_tree(root)
        except RuntimeError as error:
            assert "extra" in str(error) or "duplicate" in str(error)
        else:
            raise AssertionError("nested silence container was accepted")
        duplicate_silence.unlink()
        canonical_silence.unlink()
        try:
            validate_snapshot_tree(root)
        except RuntimeError as error:
            assert "canonical" in str(error)
        else:
            raise AssertionError("missing canonical silence container was accepted")
        torch.save({"latent": torch.zeros((1, 2), dtype=torch.float32)}, canonical_silence)
        duplicate_config = root / COMPONENTS[0] / "config.json"
        duplicate_config.write_text('{"model_type":"a","model_type":"b"}\n', encoding="utf-8")
        try:
            inventory_component(root, COMPONENTS[0])
        except RuntimeError as error:
            assert "duplicate JSON key" in str(error)
        else:
            raise AssertionError("duplicate JSON config key was accepted")
        duplicate_config.write_text(json.dumps({"model_type": "fixture-1"}) + "\n", encoding="utf-8")
        duplicate_config.write_text('{"model_type":[]}\n', encoding="utf-8")
        try:
            inventory_component(root, COMPONENTS[0])
        except RuntimeError as error:
            assert "model_type is not a string" in str(error)
        else:
            raise AssertionError("config topology drift was accepted")
        duplicate_config.write_text(json.dumps({"model_type": "fixture-1"}) + "\n", encoding="utf-8")
        (root / "community-xl").mkdir()
        try:
            validate_snapshot_tree(root)
        except RuntimeError:
            pass
        else:
            raise AssertionError("unexpected component directory was accepted")
        (root / "community-xl").rmdir()
        missing_component = root / COMPONENTS[-1]
        missing_component.rename(root / "missing-component")
        try:
            inventory_component(root, COMPONENTS[-1])
        except RuntimeError as error:
            assert "missing canonical" in str(error)
        else:
            raise AssertionError("missing component was accepted")
        (root / "missing-component").rename(missing_component)
        index = root / COMPONENTS[0] / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"layer.1": "missing.safetensors"}}), encoding="utf-8")
        try:
            inventory_component(root, COMPONENTS[0])
        except RuntimeError:
            pass
        else:
            raise AssertionError("index mismatch was accepted")
        index.write_text(json.dumps({"weight_map": {"missing.tensor": "model.safetensors"}}), encoding="utf-8")
        try:
            inventory_component(root, COMPONENTS[0])
        except RuntimeError:
            pass
        else:
            raise AssertionError("weight-map missing/extra tensor key was accepted")
        index.write_text(json.dumps({"weight_map": {"layer.1": "../model.safetensors"}}), encoding="utf-8")
        try:
            inventory_component(root, COMPONENTS[0])
        except RuntimeError:
            pass
        else:
            raise AssertionError("component index traversal was accepted")
        unsafe = root / "unsafe.pt"
        unsafe.write_bytes(pickle.dumps({"not": "a safe tensor archive"}, protocol=2))
        try:
            inspect_container(unsafe, root)
        except RuntimeError:
            pass
        else:
            raise AssertionError("unsafe pickle fixture was accepted")
        (root / "README.md").write_text("---\nlicense: unknown\n---\n", encoding="utf-8")
        (root / ".cache" / "huggingface" / "LICENSE").write_text("permission is hereby granted, free of charge\n", encoding="utf-8")
        try:
            find_hf_license(root)
        except RuntimeError:
            pass
        else:
            raise AssertionError("unknown license fixture was accepted")
        size_mismatch = root.parent / "size-tree.json"
        size_mismatch.write_text(json.dumps({"repository": UPSTREAM_REPOSITORY, "revision": UPSTREAM_REVISION, "files": [{"path": "README.md", "type": "file", "size": 999}]}), encoding="utf-8")
        tree_blockers: list[str] = []
        server_tree(root, size_mismatch, tree_blockers)
        assert any("server/local tree mismatch" in blocker for blocker in tree_blockers)
        revision_mismatch = root.parent / "revision-tree.json"
        revision_mismatch.write_text(json.dumps({"repository": UPSTREAM_REPOSITORY, "requested_revision": "a" * 40, "revision": "a" * 40, "resolved_revision": "a" * 40, "files": []}), encoding="utf-8")
        revision_blockers: list[str] = []
        server_tree(root, revision_mismatch, revision_blockers)
        assert any("repository/requested/resolved revision mismatch" in blocker for blocker in revision_blockers)
        blocked = root / "blocked"
        write_blocked(blocked, RuntimeError("license unknown fixture"))
        manifest = json.loads((blocked / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "BLOCKED" and manifest["error_type"] == "RuntimeError"
    source = Path(__file__).read_text(encoding="utf-8")
    for marker in ("NOT_IMPLEMENTED_FAIL_CLOSED", "UNSUPPORTED", "BLOCKED_BY_CPU", "NOT_RUN", "NO_UPLOAD", "UNAUTHENTICATED_BLOCKER", "UNREVIEWED_BLOCKER"):
        assert marker in source
    print("ace_step_v15_inspect.py self-test: OK (shards/container/license/blocker contracts)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--server-tree", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.snapshot, args.source, args.output, args.server_tree)):
            parser.error("--self-test accepts no other arguments")
        self_test()
        return 0
    if any(value is None for value in (args.snapshot, args.source, args.output, args.server_tree)):
        parser.error("--snapshot, --source, --output, and --server-tree are required")
    try:
        inspect(args.snapshot, args.source, args.output, args.server_tree)
    except Exception as error:
        write_blocked(args.output, error, args.server_tree)
        print(f"ACE-Step 1.5 inspection BLOCKED: {error}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
