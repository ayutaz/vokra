#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Fail-closed VAST-only inspection for facebook/audiogen-medium."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any

HF_REPOSITORY = "facebook/audiogen-medium"
HF_REVISION = "1277dd7dfd8fa57a205a70acc5de0ee90804502f"
SOURCE_REPOSITORY = "https://github.com/facebookresearch/audiocraft.git"
SOURCE_TAG = "v1.0.0"
SOURCE_REVISION = "a2b96756956846e194c9255d0cdadc2b47c93f1b"
FORMAT = "vokra-audiogen-medium-inspection-v2"
PROJECT = Path(__file__).with_name("audiogen_medium_reference")
HF_FILES = {".gitattributes", "README.md", "compression_state_dict.bin", "state_dict.bin"}
HF_FILE_IDENTITIES = {
    ".gitattributes": {"bytes": 1_519, "git_blob_sha1": "a6344aac8c09253b3b630fb776ae94478aa0275b"},
    "README.md": {"bytes": 2_240, "git_blob_sha1": "31a77819df582937de900237706f104a325e223f"},
    "compression_state_dict.bin": {"bytes": 235_740_815, "lfs_pointer_git_blob_sha1": "0cc8de6c4cf0c16326ee3c693385370b98bbf0f2", "lfs_payload_sha256": "5a520e64ca99226a9956f83b06df0617b713183fcdc384779883a6bb46dc1095"},
    "state_dict.bin": {"bytes": 3_678_455_287, "lfs_pointer_git_blob_sha1": "ae572ad32705a0a9ba679b0d2813cbae716d869e", "lfs_payload_sha256": "f3b20997834de1ca47d6a31d00a5dc37019b279c7c8f250fd482d56def04faaa"},
}
ARCHIVES = {name: row["bytes"] for name, row in HF_FILE_IDENTITIES.items() if name.endswith(".bin")}
HF_EXPECTED_LICENSE = "cc-by-nc-4.0"
SOURCE_WEIGHTS_LICENSE_BLOB = "108b5f002fc31efe11d881de2cd05329ebe8cc37"
HISTORICAL_WEIGHTS_LICENSE_BLOB = "dc1adf98654156baeb94d2e055c224a847e5820d"
MAX_ZIP_MEMBERS = 200_000
MAX_ZIP_UNCOMPRESSED = 8_000_000_000
MAX_CHECKPOINT_NODES = 1_000_000
MAX_CHECKPOINT_DEPTH = 64
MAX_CHECKPOINT_CONTAINER_ITEMS = 500_000
MAX_CHECKPOINT_STRING = 1_000_000
TRANSPORT_CACHE = ".cache/huggingface"
ROLE_MARKERS = {
    "builders": ("AudioGen", "get_audiogen"),
    "audiogen_model": ("class AudioGen", "CompressionModel"),
    "loaders": ("load_compression_model", "CompressionModel"),
    "language_model": ("class LMModel", "StreamingTransformer"),
    "conditioner": ("ConditionProvider", "T5"),
    "codebook_pattern": ("CodebooksPattern", "delay"),
    "transformer": ("StreamingTransformer", "Transformer"),
    "encodec_seanet": ("SEANet", "CompressionModel"),
}
SOURCE_ROLE_BLOBS = {"LICENSE": "b93be90515ccd0b9daedaa589e42bf5929693f1f", "LICENSE_weights": "108b5f002fc31efe11d881de2cd05329ebe8cc37", **{path: blob for path, blob in {
    "audiocraft/models/audiogen.py": "5cb889982ddc027e2588b7cfb8ef428b313ce88a",
    "audiocraft/models/builders.py": "038bf99c3d0fbbb86005683d5a2a1b4edcac4298",
    "audiocraft/models/encodec.py": "40d133017c0a0eddaafb07d291b3845789775bc3",
    "audiocraft/models/lm.py": "8cefd2c58c3a337378579d6cd6469fd038cbb1ee",
    "audiocraft/models/loaders.py": "7fd49d84e21ed26c01919dcb8e05315fb3bdf398",
    "audiocraft/modules/codebooks_patterns.py": "3cf3bb41774700a679ffe4325236d0324a99c546",
    "audiocraft/modules/conditioners.py": "d10ac8dc96466375379c883cd62f7c04a1bb0a73",
    "audiocraft/modules/transformer.py": "048c06dfbb0ab4167afce95dffb73dcc343c2344",
    "audiocraft/modules/conv.py": "d115cbf8729b642ed78608bd00a4d0fd5afae6fd",
    "audiocraft/modules/lstm.py": "c0866175950c1ca4f6cca98649525e6481853bba",
    "audiocraft/modules/seanet.py": "3e5998e9153afb6e68ea410d565e00ea835db248",
    "audiocraft/modules/quantization/core_vq.py": "da02a6ce3a7de15353f0fba9e826052beb67c436",
    "audiocraft/modules/quantization/vq.py": "aa57bea59db95ddae35e0657f723ca3a29ee943b",
}.items()}}
ROLE_MARKER_PATHS = {
    "audiocraft/models/audiogen.py": "audiogen_model",
    "audiocraft/models/builders.py": "builders",
    "audiocraft/models/encodec.py": "encodec_seanet",
    "audiocraft/models/lm.py": "language_model",
    "audiocraft/models/loaders.py": "loaders",
    "audiocraft/modules/codebooks_patterns.py": "codebook_pattern",
    "audiocraft/modules/conditioners.py": "conditioner",
    "audiocraft/modules/transformer.py": "transformer",
    "audiocraft/modules/conv.py": "conv",
    "audiocraft/modules/lstm.py": "lstm",
    "audiocraft/modules/seanet.py": "seanet",
    "audiocraft/modules/quantization/core_vq.py": "core_vq",
    "audiocraft/modules/quantization/vq.py": "vq",
}


def strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise RuntimeError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=strict_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError, RuntimeError) as error:
        raise RuntimeError(f"invalid strict JSON {path}: {error}") from error


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    value = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def lfs_pointer_sha1(payload_sha256: str, payload_size: int) -> str:
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{payload_sha256}\nsize {payload_size}\n".encode()
    value = hashlib.sha1(f"blob {len(pointer)}\0".encode())
    value.update(pointer)
    return value.hexdigest()


def safe_relative(value: str, label: str) -> None:
    path = Path(value)
    if not value or "\x00" in value or "\\" in value or path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"unsafe {label} path: {value!r}")


def snapshot_files(root: Path) -> tuple[list[Path], bool]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"snapshot is not a regular directory: {root}")
    result: list[Path] = []
    cache_excluded = False
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        parts = path.relative_to(root).parts
        if path.is_symlink():
            raise RuntimeError(f"snapshot symlink is forbidden: {relative}")
        if relative == ".cache":
            if not path.is_dir():
                raise RuntimeError("snapshot cache parent is not a directory")
            continue
        if relative == TRANSPORT_CACHE:
            if not path.is_dir():
                raise RuntimeError("HF transport cache is not a directory")
            cache_excluded = True
            continue
        if relative.startswith(TRANSPORT_CACHE + "/"):
            cache_excluded = True
            continue
        if ".cache" in parts or ".git" in parts:
            raise RuntimeError(f"unauthenticated cache/metadata path: {relative}")
        if path.is_dir():
            continue
        if not path.is_file() or stat.S_IFMT(path.stat().st_mode) != stat.S_IFREG:
            raise RuntimeError(f"snapshot member is not regular: {relative}")
        result.append(path)
    if not result:
        raise RuntimeError("snapshot is empty")
    return result, cache_excluded


def inventory_snapshot(root: Path, packet_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    packet = load_json(packet_path)
    required_envelope = {"repository", "requested_revision", "resolved_revision", "walk", "files"}
    if not isinstance(packet, dict) or set(packet) != required_envelope:
        raise RuntimeError("HF packet envelope is not exact")
    if (packet["repository"], packet["requested_revision"], packet["resolved_revision"], packet["walk"]) != (HF_REPOSITORY, HF_REVISION, HF_REVISION, "recursive_file_only"):
        raise RuntimeError("HF packet repository/revision/walk mismatch")
    expected = packet["files"]
    if not isinstance(expected, list):
        raise RuntimeError("HF packet files must be a list")
    local_paths, cache_excluded = snapshot_files(root)
    local_names = {path.relative_to(root).as_posix() for path in local_paths}
    if local_names != HF_FILES:
        raise RuntimeError(f"HF local tree mismatch: {sorted(local_names)}")
    required_row = {"path", "type", "size", "git_blob_sha1", "lfs_pointer_git_blob_sha1", "lfs_payload_sha256", "lfs_payload_size"}
    rows: list[dict[str, Any]] = []
    names: set[str] = set()
    for item in expected:
        if not isinstance(item, dict) or set(item) != required_row or item.get("type") != "file":
            raise RuntimeError("HF packet row schema/type mismatch")
        name, size = item["path"], item["size"]
        safe_relative(name, "HF packet")
        if name in names or name in {TRANSPORT_CACHE, ".cache"} or name.startswith(TRANSPORT_CACHE + "/"):
            raise RuntimeError(f"duplicate/unsafe HF packet path: {name}")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise RuntimeError(f"invalid HF packet size: {name}")
        fixed = HF_FILE_IDENTITIES.get(name)
        if fixed is None or size != fixed["bytes"]:
            raise RuntimeError(f"HF fixed artifact size mismatch: {name}")
        names.add(name)
        path = root / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size != size:
            raise RuntimeError(f"HF packet/local size or symlink mismatch: {name}")
        payload_sha = item["lfs_payload_sha256"]
        git_id = item["git_blob_sha1"]
        pointer_id = item["lfs_pointer_git_blob_sha1"]
        payload_size = item["lfs_payload_size"]
        if payload_sha is None:
            if "lfs_pointer_git_blob_sha1" in fixed or not isinstance(git_id, str) or not re.fullmatch(r"[0-9a-f]{40}", git_id) or pointer_id is not None or payload_size is not None or git_id != fixed["git_blob_sha1"] or git_blob_sha1(path) != git_id:
                raise RuntimeError(f"regular Git identity mismatch: {name}")
            row = {"path": name, "bytes": size, "sha256": digest(path), "git_blob_sha1": git_id, "lfs_pointer_git_blob_sha1": None, "lfs_payload_sha256": None, "lfs_payload_size": None}
        else:
            if "lfs_pointer_git_blob_sha1" not in fixed or git_id is not None or not isinstance(pointer_id, str) or not re.fullmatch(r"[0-9a-f]{40}", pointer_id) or pointer_id != fixed["lfs_pointer_git_blob_sha1"] or not isinstance(payload_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", payload_sha) or payload_sha != fixed["lfs_payload_sha256"] or not isinstance(payload_size, int) or isinstance(payload_size, bool) or payload_size != size or digest(path) != payload_sha or lfs_pointer_sha1(payload_sha, size) != pointer_id:
                raise RuntimeError(f"LFS payload/pointer identity mismatch: {name}")
            row = {"path": name, "bytes": size, "sha256": payload_sha, "git_blob_sha1": None, "lfs_pointer_git_blob_sha1": pointer_id, "lfs_payload_sha256": payload_sha, "lfs_payload_size": size}
        rows.append(row)
    if names != HF_FILES or len(rows) != len(HF_FILES):
        raise RuntimeError(f"HF packet tree mismatch: {sorted(names)}")
    return {"repository": packet["repository"], "requested_revision": packet["requested_revision"], "resolved_revision": packet["resolved_revision"], "walk": packet["walk"], "cache_excluded": cache_excluded}, sorted(rows, key=lambda row: row["path"])


def parse_model_card(text: str) -> dict[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise RuntimeError("model card frontmatter missing")
    try:
        end = next(index for index in range(1, len(lines)) if lines[index].strip() == "---")
    except StopIteration as error:
        raise RuntimeError("model card frontmatter unterminated") from error
    license_value: str | None = None
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#") or line[0].isspace() or line.startswith("-"):
            continue
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_-]*):[ \t]*(.*)", line)
        if match is None:
            raise RuntimeError("malformed top-level frontmatter")
        key, value = match.groups()
        if key != "license":
            continue
        if license_value is not None or not value.strip() or value.strip().startswith(("[", "{")):
            raise RuntimeError("license must be exactly one top-level scalar")
        license_value = value.strip().strip("\"'")
    if license_value != HF_EXPECTED_LICENSE:
        raise RuntimeError(f"model card license mismatch: {license_value!r}")
    return {"license": license_value}


def walk_checkpoint(value: Any, torch: Any, *, path: str = "$", depth: int = 0, seen: set[int] | None = None, state: dict[str, Any] | None = None) -> dict[str, Any]:
    seen = set() if seen is None else seen
    state = {"nodes": 0, "tensors": [], "scalars": [], "containers": 0} if state is None else state
    state["nodes"] += 1
    if state["nodes"] > MAX_CHECKPOINT_NODES or depth > MAX_CHECKPOINT_DEPTH:
        raise RuntimeError("checkpoint structure exceeds bounded walker limits")
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"non-finite checkpoint tensor: {path}")
        state["tensors"].append({"path": path, "dtype": str(value.dtype), "shape": list(value.shape), "elements": value.numel()})
        return state
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str) and len(value) > MAX_CHECKPOINT_STRING:
            raise RuntimeError(f"checkpoint string exceeds bound: {path}")
        if isinstance(value, float) and value != value:
            raise RuntimeError(f"non-finite checkpoint scalar: {path}")
        key = path.rsplit(".", 1)[-1].lower()
        if key in {"sample_rate", "sample_rate_hz", "frame_rate", "frame_rate_hz", "num_codebooks", "n_q", "n_codebooks", "codebook_size", "conditioner", "text_encoder", "text_encoder_name", "t5_name", "name", "dimension", "channels", "sample_rate"}:
            state["scalars"].append({"path": path, "key": key, "value": value})
        return state
    if not isinstance(value, (dict, list, tuple)):
        raise RuntimeError(f"unsupported checkpoint value type at {path}: {type(value).__name__}")
    object_id = id(value)
    if object_id in seen:
        raise RuntimeError(f"checkpoint contains a cycle at {path}")
    seen.add(object_id)
    state["containers"] += 1
    if state["containers"] > MAX_CHECKPOINT_CONTAINER_ITEMS:
        raise RuntimeError("checkpoint container count exceeds bound")
    if isinstance(value, dict):
        if len(value) > MAX_CHECKPOINT_CONTAINER_ITEMS:
            raise RuntimeError(f"checkpoint mapping exceeds bound: {path}")
        for key, child in value.items():
            if not isinstance(key, str) or len(key) > MAX_CHECKPOINT_STRING:
                raise RuntimeError(f"checkpoint mapping key is not bounded string: {path}")
            walk_checkpoint(child, torch, path=f"{path}.{key}" if path != "$" else key, depth=depth + 1, seen=seen, state=state)
    else:
        if len(value) > MAX_CHECKPOINT_CONTAINER_ITEMS:
            raise RuntimeError(f"checkpoint sequence exceeds bound: {path}")
        for index, child in enumerate(value):
            walk_checkpoint(child, torch, path=f"{path}[{index}]", depth=depth + 1, seen=seen, state=state)
    seen.remove(object_id)
    return state


def checkpoint_config(scalars: list[dict[str, Any]]) -> dict[str, Any]:
    observed: dict[str, list[Any]] = {}
    for item in scalars:
        observed.setdefault(item["key"], []).append(item["value"])
    def one(keys: tuple[str, ...]) -> Any:
        values = [value for key in keys for value in observed.get(key, [])]
        return values[0] if values and all(value == values[0] for value in values) else None
    config = {"sample_rate_hz": one(("sample_rate_hz", "sample_rate")), "frame_rate_hz": one(("frame_rate_hz", "frame_rate")), "num_codebooks": one(("num_codebooks", "n_q", "n_codebooks")), "codebook_size": one(("codebook_size",)), "conditioner_candidates": [value for key in ("conditioner", "text_encoder", "text_encoder_name", "t5_name", "name") for value in observed.get(key, []) if isinstance(value, str) and ("t5" in value.lower() or "text" in value.lower())]}
    missing = [key for key in ("sample_rate_hz", "frame_rate_hz", "num_codebooks") if config[key] is None]
    topology_mismatch = config["sample_rate_hz"] != 16_000 or config["frame_rate_hz"] != 50 or config["num_codebooks"] != 4
    if topology_mismatch or not config["conditioner_candidates"]:
        config["status"] = "BLOCKED_SEMANTICS"
        config["missing_or_mismatched"] = missing + (["fixed 16-kHz/50-Hz/4-codebook topology"] if topology_mismatch else [])
    else:
        config["status"] = "BLOCKED_CONDITIONER_SIZE_UNRESOLVED"
    return config


def inspect_torch_archive(path: Path) -> dict[str, Any]:
    if path.stat().st_size <= 0:
        raise RuntimeError(f"empty torch archive: {path.name}")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) == 0 or len(members) > MAX_ZIP_MEMBERS:
            raise RuntimeError(f"torch archive member count exceeds bound: {path.name}")
        names: set[str] = set()
        uncompressed = 0
        inventory: list[dict[str, Any]] = []
        for member in members:
            safe_relative(member.filename, "torch archive")
            if member.filename in names:
                raise RuntimeError(f"duplicate torch archive member: {member.filename}")
            names.add(member.filename)
            mode = (member.external_attr >> 16) & 0xFFFF
            if mode and stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise RuntimeError(f"torch archive link/device member: {member.filename}")
            uncompressed += member.file_size
            if uncompressed > MAX_ZIP_UNCOMPRESSED:
                raise RuntimeError(f"torch archive decompressed bound exceeded: {path.name}")
            inventory.append({"name": member.filename, "bytes": member.file_size, "compressed_bytes": member.compress_size})
    import torch
    unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(str(path))
    if unsafe:
        raise RuntimeError(f"torch archive unsafe globals: {unsafe}")
    value = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise RuntimeError(f"torch archive root is not a mapping: {path.name}")
    walked = walk_checkpoint(value, torch)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path), "zip_members": inventory, "tensor_count": len(walked["tensors"]), "tensors": walked["tensors"], "scalar_evidence": walked["scalars"], "container_count": walked["containers"], "config_evidence": checkpoint_config(walked["scalars"]), "safe_loader": "torch.load(weights_only=True)", "execution": "NOT_PERFORMED"}


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT).strip()


def license_evidence(root: Path) -> dict[str, Any]:
    path = root / "LICENSE"
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("AudioCraft LICENSE is missing/non-regular")
    text = path.read_text(encoding="utf-8", errors="strict").lower()
    clauses = {"apache_grant": "you may obtain a copy of the license" in text, "apache_warranty": "without warranties or conditions" in text, "mit_grant": "permission is hereby granted, free of charge" in text, "mit_warranty": "provided \"as is\"" in text}
    if not (clauses["apache_grant"] and clauses["apache_warranty"] or clauses["mit_grant"] and clauses["mit_warranty"]):
        raise RuntimeError("AudioCraft LICENSE grant/warranty clauses are not authenticated")
    return {"path": "LICENSE", "bytes": path.stat().st_size, "sha256": digest(path), "git_blob_sha1": git_blob_sha1(path), "license": "Apache-2.0" if clauses["apache_grant"] else "MIT", "clauses": clauses}


def weights_license_evidence(root: Path, tracked: dict[str, dict[str, Any]]) -> dict[str, Any]:
    path = root / "LICENSE_weights"
    row = tracked.get("LICENSE_weights")
    if row is None or not path.is_file() or path.is_symlink() or row["mode"] != "100644":
        raise RuntimeError("AudioCraft LICENSE_weights is missing/non-regular")
    if row["index_object_sha1"] != SOURCE_WEIGHTS_LICENSE_BLOB or row["head_object_sha1"] != SOURCE_WEIGHTS_LICENSE_BLOB or row["working_blob_sha1"] != SOURCE_WEIGHTS_LICENSE_BLOB:
        raise RuntimeError("AudioCraft LICENSE_weights Git identity mismatch")
    text = path.read_text(encoding="utf-8", errors="strict").lower()
    clauses = {"attribution": "attribution" in text, "noncommercial": "noncommercial" in text, "version_4": "4.0 international" in text or "4.0 international license" in text}
    if not all(clauses.values()):
        raise RuntimeError("AudioCraft LICENSE_weights CC-BY-NC clauses are not authenticated")
    return {"path": "LICENSE_weights", "bytes": path.stat().st_size, "sha256": digest(path), "git_blob_sha1": SOURCE_WEIGHTS_LICENSE_BLOB, "license": "CC-BY-NC-4.0", "clauses": clauses}


def source_inventory(source: Path) -> dict[str, Any]:
    if git(source, "rev-parse", "HEAD") != SOURCE_REVISION:
        raise RuntimeError("AudioCraft source revision mismatch")
    if git(source, "describe", "--exact-match", "--tags") != SOURCE_TAG:
        raise RuntimeError("AudioCraft source tag mismatch")
    origin = git(source, "remote", "get-url", "origin").removesuffix(".git").rstrip("/")
    if origin != SOURCE_REPOSITORY.removesuffix(".git"):
        raise RuntimeError(f"AudioCraft source origin mismatch: {origin}")
    if git(source, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("AudioCraft source checkout is dirty")
    tracked: list[dict[str, Any]] = []
    for entry in git(source, "ls-files", "--stage", "-z").split("\0"):
        if not entry:
            continue
        metadata, relative = entry.split("\t", 1)
        mode, index_object, stage = metadata.split()
        path = source / relative
        if stage != "0" or mode not in {"100644", "100755"} or not path.is_file() or path.is_symlink():
            raise RuntimeError(f"unsupported/non-regular tracked source entry: {relative}")
        expected_mode = 0o755 if mode == "100755" else 0o644
        if stat.S_IMODE(path.stat().st_mode) != expected_mode:
            raise RuntimeError(f"tracked source mode drift: {relative}")
        head_object = git(source, "rev-parse", f"HEAD:{relative}")
        working_object = git_blob_sha1(path)
        if index_object != head_object or head_object != working_object:
            raise RuntimeError(f"tracked source object drift: {relative}")
        tracked.append({"path": relative, "mode": mode, "stage": 0, "bytes": path.stat().st_size, "sha256": digest(path), "index_object_sha1": index_object, "head_object_sha1": head_object, "working_blob_sha1": working_object})
    by_path = {row["path"]: row for row in tracked}
    roles: dict[str, Any] = {}
    role_blockers: list[str] = []
    role_warnings: list[str] = []
    for relative, expected_blob in SOURCE_ROLE_BLOBS.items():
        if relative in {"LICENSE", "LICENSE_weights"}:
            continue
        row = by_path.get(relative)
        if row is None or row["mode"] != "100644":
            role_blockers.append(f"fixed source role missing/non-100644: {relative}")
            continue
        if row["index_object_sha1"] != expected_blob or row["head_object_sha1"] != expected_blob or row["working_blob_sha1"] != expected_blob:
            role_blockers.append(f"fixed source role Git object mismatch: {relative}")
            continue
        roles[relative] = row
        marker_name = ROLE_MARKER_PATHS.get(relative)
        markers = ROLE_MARKERS.get(marker_name, ())
        if markers and not all(marker in (source / relative).read_text(encoding="utf-8", errors="replace") for marker in markers):
            role_warnings.append(f"source role semantic markers missing: {relative}")
    license_row = by_path.get("LICENSE")
    if license_row is None or license_row["mode"] != "100644" or license_row["index_object_sha1"] != SOURCE_ROLE_BLOBS["LICENSE"] or license_row["head_object_sha1"] != SOURCE_ROLE_BLOBS["LICENSE"] or license_row["working_blob_sha1"] != SOURCE_ROLE_BLOBS["LICENSE"]:
        raise RuntimeError("AudioCraft LICENSE is not tracked")
    return {"repository": SOURCE_REPOSITORY, "tag": SOURCE_TAG, "revision": SOURCE_REVISION, "origin": origin, "clean": True, "tracked_files": sorted(tracked, key=lambda row: row["path"]), "roles": roles, "role_identity_allowlist": "AUTHENTICATED_FIXED_ROLE_BLOBS", "role_blockers": role_blockers, "role_warnings": role_warnings, "license": license_evidence(source), "weights_license": weights_license_evidence(source, by_path), "historical_weight_license": {"git_blob_sha1": HISTORICAL_WEIGHTS_LICENSE_BLOB, "license": "CC-BY-NC-ND-4.0", "status": "HISTORICAL_PROVENANCE_EVIDENCE"}}


def write_manifest(output: Path, **fields: Any) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = {"format": FORMAT, "status": "BLOCKED", "evidence_stage": "INSPECTION_ONLY", "runtime_status": "LM_ONLY_PCM_FAIL_CLOSED", "cpu_status": "UNSUPPORTED", "metal_status": "BLOCKED_BY_CPU", "parity_status": "NOT_RUN", "publication": "NO_UPLOAD", **fields}
    (output / "manifest.json").write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--server-tree", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not (PROJECT / "uv.lock").is_file():
        if args.output:
            write_manifest(args.output, inspection_status="INSPECTION_ERROR", collection_status="UNVERIFIED", upstream={"repository": HF_REPOSITORY, "requested_revision": HF_REVISION, "resolved_revision": None}, blockers=["dedicated uv.lock absent; fail before model/source acquisition"])
        return 2
    if any(value is None for value in (args.snapshot, args.source, args.server_tree, args.output)):
        parser.error("normal run requires snapshot, source, server-tree, and output")
    try:
        server, files = inventory_snapshot(args.snapshot, args.server_tree)
        readme = args.snapshot / "README.md"
        card = parse_model_card(readme.read_text(encoding="utf-8"))
        archives = {}
        for name, expected_size in ARCHIVES.items():
            path = args.snapshot / name
            if path.stat().st_size != expected_size:
                raise RuntimeError(f"fixed archive size mismatch: {name}")
            archives[name] = inspect_torch_archive(path)
        source = source_inventory(args.source)
        collection_blockers = source["role_blockers"]
        config_blockers = [f"{name} checkpoint config semantics are not fully authenticated" for name, archive in archives.items() if archive["config_evidence"]["status"] != "AUTHENTICATED"]
        blockers = ["release/source timing gap: HF weights uploaded 2023-07-27 before AudioCraft v1.0.0 execution source", "AudioCraft role identity is bound to v1.0.0 but weight-build provenance is not independently authenticated", "external text conditioner name/size is not fully recovered from authenticated checkpoint", "native AudioGen codec/LM composition is not implemented", "CPU/Metal parity is not run", "training-data provenance is unauthenticated", "source LICENSE_weights is CC-BY-NC-4.0; historical v0.0.2 LICENSE_weights was CC-BY-NC-ND-4.0 (provenance ambiguity)"] + config_blockers + collection_blockers
        complete = not collection_blockers
        write_manifest(args.output, inspection_status="AUTHENTICATED_EVIDENCE_COMPLETE" if complete else "INSPECTION_ERROR", collection_status="AUTHENTICATED" if complete else "UNVERIFIED", upstream={"repository": HF_REPOSITORY, "requested_revision": HF_REVISION, "resolved_revision": HF_REVISION, "walk": "recursive_file_only", "server_tree": server, "files": files, "model_card": {"path": "README.md", "license": card["license"], "sha256": digest(readme), "git_blob_sha1": git_blob_sha1(readme)}}, archives=archives, compression_companion={"role": "release-specific 16-kHz EnCodec/SEANet companion", "path": "compression_state_dict.bin"}, external_text_conditioner={"status": "UNRESOLVED_BLOCKER", "selection": None}, official_source=source, license_evidence={"weights": {"hf_model_card": {"license": HF_EXPECTED_LICENSE, "status": "AUTHENTICATED_FROM_MODEL_CARD"}, "source_LICENSE_weights": source["weights_license"], "historical_v0_0_2_LICENSE_weights": {"git_blob_sha1": HISTORICAL_WEIGHTS_LICENSE_BLOB, "license": "CC-BY-NC-ND-4.0", "status": "HISTORICAL_EVIDENCE_NOT_CURRENT_SOURCE"}, "status": "PROVENANCE_AMBIGUITY_BLOCKER"}, "code": source["license"], "training_data": "UNAUTHENTICATED_BLOCKER"}, blockers=sorted(set(blockers)))
        return 2
    except Exception as error:
        write_manifest(args.output or Path("."), inspection_status="INSPECTION_ERROR", collection_status="UNVERIFIED", upstream={"repository": HF_REPOSITORY, "requested_revision": HF_REVISION, "resolved_revision": None}, error_type=type(error).__name__, blockers=[str(error)])
        return 2


def self_test() -> None:
    global HF_FILE_IDENTITIES
    assert len(HF_REVISION) == 40 and len(SOURCE_REVISION) == 40
    assert HF_FILES == {".gitattributes", "README.md", "compression_state_dict.bin", "state_dict.bin"}
    assert ARCHIVES["state_dict.bin"] == 3_678_455_287
    assert SOURCE_WEIGHTS_LICENSE_BLOB == "108b5f002fc31efe11d881de2cd05329ebe8cc37"
    assert HISTORICAL_WEIGHTS_LICENSE_BLOB == "dc1adf98654156baeb94d2e055c224a847e5820d"
    assert "T5-large" not in ROLE_MARKERS
    import torch
    walked = walk_checkpoint({"best_state": {"layer": {"weight": torch.ones(2, 2)}}, "cfg": {"sample_rate": 16_000, "frame_rate": 50, "num_codebooks": 4, "text_encoder_name": "t5-small"}}, torch)
    assert walked["tensors"][0]["path"] == "best_state.layer.weight" and checkpoint_config(walked["scalars"])["sample_rate_hz"] == 16_000
    cycle: list[Any] = []
    cycle.append(cycle)
    try:
        walk_checkpoint(cycle, torch)
    except RuntimeError:
        pass
    else:
        raise AssertionError("checkpoint cycle was accepted")
    with tempfile.TemporaryDirectory(prefix="audiogen-medium-inspect-") as directory:
        root = Path(directory)
        card = "---\nlicense: cc-by-nc-4.0\ndatasets:\n- audio\n---\n"
        assert parse_model_card(card) == {"license": "cc-by-nc-4.0"}
        for invalid in ("license: cc-by-nc-4.0", "---\nlicense: cc-by-nc-4.0\nlicense: cc-by-nc-4.0\n---", "---\ndatasets:\n  license: cc-by-nc-4.0\n---", "---\nlicense:\n- cc-by-nc-4.0\n---"):
            try:
                parse_model_card(invalid)
            except RuntimeError:
                pass
            else:
                raise AssertionError("invalid model-card license accepted")
        packet = root / "packet.json"
        snapshot = root / "snapshot"
        snapshot.mkdir()
        for name in HF_FILES:
            (snapshot / name).write_bytes(b"x")
        original_identities = HF_FILE_IDENTITIES
        HF_FILE_IDENTITIES = {name: {"bytes": 1, "git_blob_sha1": git_blob_sha1(snapshot / name)} for name in HF_FILES}
        rows = []
        for name in sorted(HF_FILES):
            path = snapshot / name
            rows.append({"path": name, "type": "file", "size": 1, "git_blob_sha1": git_blob_sha1(path), "lfs_pointer_git_blob_sha1": None, "lfs_payload_sha256": None, "lfs_payload_size": None})
        packet.write_text(json.dumps({"repository": HF_REPOSITORY, "requested_revision": HF_REVISION, "resolved_revision": HF_REVISION, "walk": "recursive_file_only", "files": rows}))
        try:
            tree, checked = inventory_snapshot(snapshot, packet)
            assert tree["repository"] == HF_REPOSITORY and len(checked) == 4
            (snapshot / "extra").write_bytes(b"x")
            try:
                inventory_snapshot(snapshot, packet)
            except RuntimeError:
                pass
            else:
                raise AssertionError("extra HF file accepted")
        finally:
            HF_FILE_IDENTITIES = original_identities
    print("audiogen_medium_inspect --self-test: OK")


if __name__ == "__main__":
    raise SystemExit(main())
