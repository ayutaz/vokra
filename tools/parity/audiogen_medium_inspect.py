#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Fail-closed VAST-only inspection for facebook/audiogen-medium."""
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
sys.path.insert(0, str(PROJECT))
from companion_contract import contract as companion_contract
from companion_contract import canonical_sha256 as companion_contract_sha256
from dependency_audit import audit as dependency_audit
from t5_metadata_audit import load_packet as load_t5_metadata_packet
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
RUNTIME_STATUS = "LOUD_PARTIAL_FAIL_CLOSED"
CPU_STATUS = "NOT_RUN"
MAX_ZIP_MEMBERS = 200_000
MAX_ZIP_UNCOMPRESSED = 8_000_000_000
MAX_CHECKPOINT_NODES = 1_000_000
MAX_CHECKPOINT_DEPTH = 64
MAX_CHECKPOINT_CONTAINER_ITEMS = 500_000
MAX_CHECKPOINT_STRING = 1_000_000
TRANSPORT_CACHE = ".cache/huggingface"
APPROVAL_SCHEMA = "vokra-audiogen-medium-inspection-approval-v1"
APPROVAL_PLACEHOLDERS = {"", "owner", "owner@example.invalid", "placeholder", "tbd", "unknown"}
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
ROLE_SEMANTIC_PATTERNS = {
    "audiogen_lm_config": (r"conditioner\s*:\s*text2sound", r"n_q\s*:\s*4", r"card\s*:\s*2048", r"delays?\s*:\s*\[?\s*0\s*,\s*1\s*,\s*2\s*,\s*3"),
    "audiogen_solver_config": (r"sample_rate\s*:\s*16000", r"channels\s*:\s*1", r"compression"),
    "medium_config": (r"dim\s*:\s*1536", r"(?:num_heads|n_heads|heads)\s*:\s*24", r"(?:num_layers|n_layers|layers)\s*:\s*48"),
    "pretrained_grid": (r"facebook/audiogen-medium", r"medium"),
    "text2sound_config": (r"model\s*:\s*t5", r"name\s*:\s*t5-large", r"finetune\s*:\s*false"),
    "encodec_solver_config": (r"sample_rate\s*:\s*16000", r"channels\s*:\s*1", r"encodec_large_nq4_s320"),
    "encodec_config": (r"n_filters\s*:\s*64", r"bins\s*:\s*2048", r"n_q\s*:\s*4", r"q_dropout\s*:\s*false"),
    "encodec_default_config": (r"dimension\s*:\s*128", r"ratios\s*:\s*\[?\s*8\s*,\s*5\s*,\s*4\s*,\s*2"),
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
    "audiocraft/quantization/core_vq.py": "da02a6ce3a7de15353f0fba9e826052beb67c436",
    "audiocraft/quantization/vq.py": "aa57bea59db95ddae35e0657f723ca3a29ee943b",
    "config/model/lm/audiogen_lm.yaml": "696f74620af193c12208ce66fdb93a37f8ea9d80",
    "config/solver/audiogen/audiogen_base_16khz.yaml": "dd6aee785c74db19ce9d6f488e68e6eeb471c026",
    "config/model/lm/model_scale/medium.yaml": "c825d1ff6c3b8cc9ae4959a898e14b40409d95e8",
    "audiocraft/grids/audiogen/audiogen_pretrained_16khz_eval.py": "12f6d402a3c4a113d4c37be062790fa435b72104",
    "config/conditioner/text2sound.yaml": "555d4b7c3cecf0ec06c8cb25440b2f426c098ad2",
    "config/solver/compression/encodec_audiogen_16khz.yaml": "654deaa01ba9cace3f7144cc91921791c081b32a",
    "config/model/encodec/encodec_large_nq4_s320.yaml": "5f2d77590afd8a81185358c705a6e42853e257c3",
    "config/model/encodec/default.yaml": "ec62c6c8ef9a686890bdca8b8f27a2f1c232205d",
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
    "audiocraft/quantization/core_vq.py": "core_vq",
    "audiocraft/quantization/vq.py": "vq",
}
ROLE_SEMANTIC_PATHS = {
    "config/model/lm/audiogen_lm.yaml": "audiogen_lm_config",
    "config/solver/audiogen/audiogen_base_16khz.yaml": "audiogen_solver_config",
    "config/model/lm/model_scale/medium.yaml": "medium_config",
    "audiocraft/grids/audiogen/audiogen_pretrained_16khz_eval.py": "pretrained_grid",
    "config/conditioner/text2sound.yaml": "text2sound_config",
    "config/solver/compression/encodec_audiogen_16khz.yaml": "encodec_solver_config",
    "config/model/encodec/encodec_large_nq4_s320.yaml": "encodec_config",
    "config/model/encodec/default.yaml": "encodec_default_config",
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


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def approval_scope(expected_head: str) -> dict[str, Any]:
    dependency = dependency_audit(PROJECT)
    return {
        "schema": "vokra-audiogen-medium-approval-scope-v2",
        "expected_head": expected_head,
        "upstream_repository": HF_REPOSITORY,
        "upstream_revision": HF_REVISION,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "artifact_identity_sha256": hashlib.sha256(canonical_json(HF_FILE_IDENTITIES)).hexdigest(),
        "companion_contract_sha256": companion_contract_sha256(),
        "dependency_status": dependency["status"],
        "dependency_scope_sha256": dependency["scope_sha256"],
        "compression_weight_build_provenance": "UNRESOLVED_INTERNAL_CHECKPOINT",
        "license": "CC-BY-NC-4.0",
        "license_scope": "RESEARCH_ONLY",
        "publication": "NO_UPLOAD",
    }


def pending_approval_scope(expected_head: str, dependency: dict[str, Any]) -> dict[str, Any]:
    """Return a canonical, non-authorizing scope for the owner packet."""

    scope = approval_scope(expected_head)
    if dependency != dependency_audit(PROJECT):
        raise RuntimeError("dependency scope changed while building approval packet")
    return {
        "status": "PENDING_OWNER_APPROVAL",
        "scope": scope,
        "scope_sha256": hashlib.sha256(canonical_json(scope)).hexdigest(),
        "record": None,
        "signer": None,
        "decision": None,
        "publication": "NO_UPLOAD",
        "signable": False,
    }


def validate_approval(path: Path, expected_head: str, expected_sha256: str, *, raw_path: str | None = None) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", expected_head):
        raise RuntimeError("expected HEAD must be exactly 40 lowercase hexadecimal characters")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise RuntimeError("approval SHA-256 must be exactly 64 lowercase hexadecimal characters")
    raw_parts = (str(path) if raw_path is None else raw_path).replace("\\", "/").split("/")
    if any(part in {".", ".."} for part in raw_parts):
        raise RuntimeError("approval evidence path may not contain dot components")
    ancestor = path.parent
    while ancestor != ancestor.parent:
        # macOS exposes /var (and sometimes /tmp) as platform-owned symlinks;
        # reject every user-controlled symlink ancestor while allowing only
        # those fixed system mount aliases.
        if ancestor.is_symlink() and not (ancestor.parent == Path("/") and ancestor.name in {"var", "tmp"}):
            raise RuntimeError("approval evidence has a symlink ancestor")
        ancestor = ancestor.parent
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("approval evidence must be a regular non-symlink file")
    if path.stat().st_size <= 0 or path.stat().st_size > 1_000_000:
        raise RuntimeError("approval evidence size is outside the bounded range")
    actual_sha256 = digest(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError("approval evidence SHA-256 mismatch")
    approval = load_json(path)
    required = {"schema", "decision", "signer", "scope", "scope_sha256"}
    if not isinstance(approval, dict) or set(approval) != required:
        raise RuntimeError("approval evidence schema is not exact")
    if approval["schema"] != APPROVAL_SCHEMA or approval["decision"] != "APPROVED":
        raise RuntimeError("approval evidence is not an approved AudioGen inspection")
    signer = approval["signer"]
    if not isinstance(signer, str) or not signer.strip() or signer.strip().casefold() in APPROVAL_PLACEHOLDERS:
        raise RuntimeError("approval signer is missing or a placeholder")
    scope = approval["scope"]
    expected_scope = approval_scope(expected_head)
    if not isinstance(scope, dict) or scope != expected_scope:
        raise RuntimeError("approval scope does not bind exact AudioGen identities/policy")
    scope_sha256 = approval["scope_sha256"]
    if not isinstance(scope_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", scope_sha256) or scope_sha256 != hashlib.sha256(canonical_json(scope)).hexdigest():
        raise RuntimeError("approval scope SHA-256 mismatch")
    return {"schema": APPROVAL_SCHEMA, "signer": signer, "scope_sha256": scope_sha256, "evidence_sha256": actual_sha256}


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


def inventory_server_metadata(packet_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """Validate HF tree/card metadata without opening any checkpoint payload."""
    packet = load_json(packet_path)
    required = {"repository", "requested_revision", "resolved_revision", "walk", "files", "model_card"}
    if not isinstance(packet, dict) or set(packet) != required:
        raise RuntimeError("HF metadata packet envelope is not exact")
    if (packet["repository"], packet["requested_revision"], packet["resolved_revision"], packet["walk"]) != (HF_REPOSITORY, HF_REVISION, HF_REVISION, "recursive_file_only"):
        raise RuntimeError("HF metadata repository/revision/walk mismatch")
    card = packet["model_card"]
    if not isinstance(card, dict) or set(card) != {"path", "license"} or card != {"path": "README.md", "license": HF_EXPECTED_LICENSE}:
        raise RuntimeError("HF metadata model-card license is not authenticated")
    rows = packet["files"]
    if not isinstance(rows, list):
        raise RuntimeError("HF metadata files must be a list")
    names: set[str] = set()
    checked: list[dict[str, Any]] = []
    required_row = {"path", "type", "size", "git_blob_sha1", "lfs_pointer_git_blob_sha1", "lfs_payload_sha256", "lfs_payload_size"}
    for row in rows:
        if not isinstance(row, dict) or set(row) != required_row or row.get("type") != "file":
            raise RuntimeError("HF metadata row schema/type mismatch")
        name = row["path"]
        safe_relative(name, "HF metadata")
        if name in names or name not in HF_FILES:
            raise RuntimeError(f"unexpected/duplicate HF metadata path: {name}")
        fixed = HF_FILE_IDENTITIES[name]
        if row["size"] != fixed["bytes"]:
            raise RuntimeError(f"fixed HF metadata size mismatch: {name}")
        if name.endswith(".bin"):
            if row["git_blob_sha1"] is not None or row["lfs_pointer_git_blob_sha1"] != fixed["lfs_pointer_git_blob_sha1"] or row["lfs_payload_sha256"] != fixed["lfs_payload_sha256"] or row["lfs_payload_size"] != fixed["bytes"]:
                raise RuntimeError(f"fixed HF LFS metadata mismatch: {name}")
        elif row["git_blob_sha1"] != fixed["git_blob_sha1"] or any(row[key] is not None for key in ("lfs_pointer_git_blob_sha1", "lfs_payload_sha256", "lfs_payload_size")):
            raise RuntimeError(f"fixed HF Git metadata mismatch: {name}")
        names.add(name)
        checked.append({"path": name, "bytes": row["size"], "git_blob_sha1": row["git_blob_sha1"], "lfs_pointer_git_blob_sha1": row["lfs_pointer_git_blob_sha1"], "lfs_payload_sha256": row["lfs_payload_sha256"], "lfs_payload_size": row["lfs_payload_size"], "payload_status": "NOT_DOWNLOADED"})
    if names != HF_FILES:
        raise RuntimeError(f"HF metadata file set mismatch: {sorted(names)}")
    return {"repository": packet["repository"], "requested_revision": packet["requested_revision"], "resolved_revision": packet["resolved_revision"], "walk": packet["walk"], "payloads": "NOT_DOWNLOADED"}, sorted(checked, key=lambda row: row["path"]), card


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


def validate_clean_head(root: Path, expected_head: str) -> None:
    if git(root, "rev-parse", "HEAD") != expected_head:
        raise RuntimeError("Vokra checkout HEAD does not match the approved expected HEAD")
    if git(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Vokra checkout is dirty")


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


def source_semantic_contract(source: Path, relative: str) -> dict[str, Any] | None:
    contract_name = ROLE_SEMANTIC_PATHS.get(relative)
    if contract_name is None:
        return None
    text = (source / relative).read_text(encoding="utf-8", errors="strict")
    patterns = ROLE_SEMANTIC_PATTERNS[contract_name]
    missing = [pattern for pattern in patterns if re.search(pattern, text) is None]
    if missing:
        raise RuntimeError(f"fixed source semantic contract mismatch: {relative}: {missing}")
    return {"name": contract_name, "patterns": list(patterns), "status": "AUTHENTICATED_SOURCE_SEMANTICS"}


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
        try:
            semantic = source_semantic_contract(source, relative)
        except (OSError, UnicodeError, RuntimeError) as error:
            role_blockers.append(str(error))
            continue
        if semantic is not None:
            roles[relative]["semantic_contract"] = semantic
        marker_name = ROLE_MARKER_PATHS.get(relative)
        markers = ROLE_MARKERS.get(marker_name, ())
        if markers and not all(marker in (source / relative).read_text(encoding="utf-8", errors="replace") for marker in markers):
            role_warnings.append(f"source role semantic markers missing: {relative}")
    license_row = by_path.get("LICENSE")
    if license_row is None or license_row["mode"] != "100644" or license_row["index_object_sha1"] != SOURCE_ROLE_BLOBS["LICENSE"] or license_row["head_object_sha1"] != SOURCE_ROLE_BLOBS["LICENSE"] or license_row["working_blob_sha1"] != SOURCE_ROLE_BLOBS["LICENSE"]:
        raise RuntimeError("AudioCraft LICENSE is not tracked")
    return {"repository": SOURCE_REPOSITORY, "tag": SOURCE_TAG, "revision": SOURCE_REVISION, "origin": origin, "clean": True, "tracked_files": sorted(tracked, key=lambda row: row["path"]), "roles": roles, "role_identity_allowlist": "AUTHENTICATED_FIXED_ROLE_BLOBS", "role_blockers": role_blockers, "role_warnings": role_warnings, "license": license_evidence(source), "weights_license": weights_license_evidence(source, by_path), "historical_weight_license": {"git_blob_sha1": HISTORICAL_WEIGHTS_LICENSE_BLOB, "license": "CC-BY-NC-ND-4.0", "status": "HISTORICAL_PROVENANCE_EVIDENCE"}}


def write_manifest(output: Path, **fields: Any) -> None:
    if output.is_symlink() or output.exists() and not output.is_dir():
        raise RuntimeError("inspection output is not a regular directory")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() or manifest_path.is_symlink():
        raise RuntimeError("inspection manifest already exists; refusing to clobber evidence")
    payload = {"format": FORMAT, "status": "BLOCKED", "evidence_stage": "INSPECTION_ONLY", "runtime_status": RUNTIME_STATUS, "cpu_status": CPU_STATUS, "metal_status": "BLOCKED_BY_CPU", "parity_status": "NOT_RUN", "publication": "NO_UPLOAD", "companion_contract": companion_contract(), **fields}
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{manifest_path.name}.", dir=output)
    temporary = Path(temporary_name)
    temporary_info = os.fstat(descriptor)
    if not stat.S_ISREG(temporary_info.st_mode):
        os.close(descriptor)
        raise RuntimeError("inspection temporary is not a regular file")
    reserved_identity = (temporary_info.st_dev, temporary_info.st_ino)
    published = False

    def unlink_if_owned(path: Path) -> None:
        try:
            current = os.lstat(path)
            if (current.st_dev, current.st_ino) == reserved_identity and stat.S_ISREG(current.st_mode):
                path.unlink()
        except OSError:
            pass

    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        verify_descriptor = os.open(temporary, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            verified = os.fstat(verify_descriptor)
            if (verified.st_dev, verified.st_ino) != reserved_identity or not stat.S_ISREG(verified.st_mode):
                raise RuntimeError("inspection temporary identity changed")
            os.fsync(verify_descriptor)
        finally:
            os.close(verify_descriptor)
        try:
            os.link(temporary, manifest_path, follow_symlinks=False)
        except FileExistsError as error:
            raise RuntimeError("inspection manifest appeared during publication; refusing to clobber evidence") from error
        published = True
        final_descriptor = os.open(manifest_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            final_info = os.fstat(final_descriptor)
            if (final_info.st_dev, final_info.st_ino) != reserved_identity or not stat.S_ISREG(final_info.st_mode):
                raise RuntimeError("inspection manifest identity changed after publication")
        finally:
            os.close(final_descriptor)
    except Exception:
        if published:
            unlink_if_owned(manifest_path)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        unlink_if_owned(temporary)


def validate_model_free_options(*, snapshot: Path | None, approval_evidence: str | None, approval_sha256: str | None, expected_head: str | None, source: Path | None, server_tree: Path | None, output: Path | None, vokra_root: Path | None, t5_server_metadata: Path | None) -> None:
    """Enforce the owner-independent metadata boundary before approval handling."""
    if snapshot is not None or approval_evidence is not None or approval_sha256 is not None:
        raise RuntimeError("model-free metadata inspection cannot mix snapshot or approval arguments")
    if any(value is None for value in (expected_head, source, server_tree, output, vokra_root, t5_server_metadata)):
        raise RuntimeError("model-free metadata inspection requires expected-head, source, server-tree, output, vokra-root, and t5-server-metadata")


def model_free_archives(files: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Keep checkpoint archives in the manifest; other metadata stays upstream.files."""
    return {
        name: {
            "bytes": row["bytes"],
            "git_blob_sha1": row["git_blob_sha1"],
            "lfs_pointer_git_blob_sha1": row["lfs_pointer_git_blob_sha1"],
            "lfs_payload_sha256": row["lfs_payload_sha256"],
            "payload_status": row["payload_status"],
            "execution": "NOT_PERFORMED",
        }
        for name, row in ((item["path"], item) for item in files)
        if name in ARCHIVES
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--model-free", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--server-tree", type=Path)
    parser.add_argument("--t5-server-metadata", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-approval", action="store_true")
    parser.add_argument("--approval-evidence")
    parser.add_argument("--approval-sha256")
    parser.add_argument("--expected-head")
    parser.add_argument("--vokra-root", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if (args.model_free or args.metadata_only) and args.validate_approval:
        parser.error("model-free metadata inspection cannot mix with --validate-approval")
    if args.validate_approval:
        if args.approval_evidence is None or args.approval_sha256 is None or args.expected_head is None:
            parser.error("--validate-approval requires --approval-evidence, --approval-sha256, and --expected-head")
        try:
            approval = validate_approval(Path(args.approval_evidence), args.expected_head, args.approval_sha256, raw_path=args.approval_evidence)
            print(json.dumps({"status": "APPROVED", **approval}, sort_keys=True))
            return 0
        except (OSError, RuntimeError, UnicodeError, ValueError) as error:
            print(f"approval BLOCKED: {error}", file=sys.stderr)
            return 2
    if args.model_free and args.metadata_only:
        parser.error("--model-free and --metadata-only are aliases; pass only one")
    if args.model_free or args.metadata_only:
        try:
            validate_model_free_options(snapshot=args.snapshot, approval_evidence=args.approval_evidence, approval_sha256=args.approval_sha256, expected_head=args.expected_head, source=args.source, server_tree=args.server_tree, output=args.output, vokra_root=args.vokra_root, t5_server_metadata=args.t5_server_metadata)
        except RuntimeError as error:
            parser.error(str(error))
        try:
            if args.vokra_root is not None:
                validate_clean_head(args.vokra_root, args.expected_head)
            t5_metadata = load_t5_metadata_packet(args.t5_server_metadata)
            server, files, card = inventory_server_metadata(args.server_tree)
            source = source_inventory(args.source)
            dependency = dependency_audit(PROJECT)
            blockers = [
                "checkpoint payloads were intentionally not downloaded or loaded",
                "canonical external T5 repository/revision/file identity is pinned, but the historical AudioGen checkpoint linkage is not recorded by upstream",
                "release-specific 16-kHz EnCodec payload identity is pinned, but its tensor manifest and weight-build provenance require payload inspection",
                "HF weight-build provenance is not independently authenticated against AudioCraft v1.0.0 source",
                "training-data provenance is unauthenticated",
                "native AudioGen codec/LM composition is not implemented",
                "CPU/Metal parity is not run",
                "source LICENSE_weights is CC-BY-NC-4.0; historical v0.0.2 LICENSE_weights is CC-BY-NC-ND-4.0 (provenance ambiguity)",
                "owner approval evidence is pending; model-free closure does not authorize real inspection or publication",
                *dependency["blockers"],
            ] + source["role_blockers"]
            write_manifest(
                args.output,
                inspection_status="AUTHENTICATED_EVIDENCE_COMPLETE" if not source["role_blockers"] else "INSPECTION_ERROR",
                collection_status="AUTHENTICATED" if not source["role_blockers"] else "UNVERIFIED",
                expected_head=args.expected_head,
                approval_evidence={"status": "PENDING_OWNER_APPROVAL"},
                upstream={**server, "files": files, "model_card": card},
                archives=model_free_archives(files),
                compression_companion={**companion_contract()["compression_companion"], "role": "release-specific 16-kHz EnCodec/SEANet companion", "status": "SOURCE_CONFIG_AUTHENTICATED_PAYLOAD_BLOCKED", "codec": "encodec_large_nq4_s320"},
                external_text_conditioner={**companion_contract()["text_conditioner"], "family": "T5-family", "status": "CANONICAL_PIN_HISTORICAL_LINK_UNVERIFIED", "selection": "t5-large"},
                t5_server_metadata=t5_metadata,
                official_source=source,
                license_evidence={"weights": {"hf_model_card": {"license": HF_EXPECTED_LICENSE, "status": "AUTHENTICATED_FROM_METADATA"}, "source_LICENSE_weights": source["weights_license"], "historical_v0_0_2_LICENSE_weights": {"git_blob_sha1": HISTORICAL_WEIGHTS_LICENSE_BLOB, "license": "CC-BY-NC-ND-4.0", "status": "HISTORICAL_EVIDENCE_NOT_CURRENT_SOURCE"}, "status": "PROVENANCE_AMBIGUITY_BLOCKER"}, "code": source["license"], "training_data": "UNAUTHENTICATED_BLOCKER"},
                dependency_closure=dependency,
                approval_scope=pending_approval_scope(args.expected_head, dependency),
                blockers=sorted(set(blockers)),
            )
            return 2
        except Exception as error:
            write_manifest(args.output, inspection_status="INSPECTION_ERROR", collection_status="UNVERIFIED", expected_head=args.expected_head, approval_evidence={"status": "PENDING_OWNER_APPROVAL"}, upstream={"repository": HF_REPOSITORY, "requested_revision": HF_REVISION, "resolved_revision": None}, error_type=type(error).__name__, blockers=[str(error)])
            return 2
    if any(value is None for value in (args.approval_evidence, args.approval_sha256, args.expected_head, args.t5_server_metadata)):
        parser.error("normal runs require --approval-evidence, --approval-sha256, --expected-head, and --t5-server-metadata")
    try:
        approval = validate_approval(Path(args.approval_evidence), args.expected_head, args.approval_sha256, raw_path=args.approval_evidence)
        if args.vokra_root is not None:
            validate_clean_head(args.vokra_root, args.expected_head)
    except (OSError, RuntimeError, UnicodeError, ValueError) as error:
        if args.output:
            write_manifest(args.output, inspection_status="INSPECTION_ERROR", collection_status="UNVERIFIED", upstream={"repository": HF_REPOSITORY, "requested_revision": HF_REVISION, "resolved_revision": None}, blockers=[str(error)], expected_head=args.expected_head, approval_evidence=approval if "approval" in locals() else {"status": "UNVERIFIED"})
        return 2
    if not (PROJECT / "uv.lock").is_file():
        if args.output:
            write_manifest(args.output, inspection_status="INSPECTION_ERROR", collection_status="UNVERIFIED", upstream={"repository": HF_REPOSITORY, "requested_revision": HF_REVISION, "resolved_revision": None}, blockers=["dedicated uv.lock absent; fail before model/source acquisition"], expected_head=args.expected_head, approval_evidence=approval)
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
        dependency = dependency_audit(PROJECT)
        t5_metadata = load_t5_metadata_packet(args.t5_server_metadata)
        collection_blockers = source["role_blockers"]
        config_blockers = [f"{name} checkpoint config semantics are not fully authenticated" for name, archive in archives.items() if archive["config_evidence"]["status"] != "AUTHENTICATED"]
        blockers = ["release/source timing gap: HF weights uploaded 2023-07-27 before AudioCraft v1.0.0 execution source", "AudioCraft role identity is bound to v1.0.0 but weight-build provenance is not independently authenticated", "canonical external T5 repository/revision/file identity is pinned, but the historical AudioGen checkpoint linkage is not recorded by upstream", "release-specific 16-kHz EnCodec payload identity is pinned, but its tensor manifest and weight-build provenance require payload inspection", "native AudioGen codec/LM composition is not implemented", "CPU/Metal parity is not run", "training-data provenance is unauthenticated", "source LICENSE_weights is CC-BY-NC-4.0; historical v0.0.2 LICENSE_weights was CC-BY-NC-ND-4.0 (provenance ambiguity)"] + config_blockers + collection_blockers
        complete = not collection_blockers
        if args.vokra_root is not None:
            validate_clean_head(args.vokra_root, args.expected_head)
        write_manifest(args.output, inspection_status="AUTHENTICATED_EVIDENCE_COMPLETE" if complete else "INSPECTION_ERROR", collection_status="AUTHENTICATED" if complete else "UNVERIFIED", expected_head=args.expected_head, approval_evidence=approval, upstream={"repository": HF_REPOSITORY, "requested_revision": HF_REVISION, "resolved_revision": HF_REVISION, "walk": "recursive_file_only", "server_tree": server, "files": files, "model_card": {"path": "README.md", "license": card["license"], "sha256": digest(readme), "git_blob_sha1": git_blob_sha1(readme)}}, archives=archives, compression_companion={**companion_contract()["compression_companion"], "role": "release-specific 16-kHz EnCodec/SEANet companion", "status": "SOURCE_CONFIG_AUTHENTICATED_PAYLOAD_PRESENT", "codec": "encodec_large_nq4_s320", "payload": "PRESENT"}, external_text_conditioner={**companion_contract()["text_conditioner"], "family": "T5-family", "status": "CANONICAL_PIN_HISTORICAL_LINK_UNVERIFIED", "selection": "t5-large"}, t5_server_metadata=t5_metadata, official_source=source, license_evidence={"weights": {"hf_model_card": {"license": HF_EXPECTED_LICENSE, "status": "AUTHENTICATED_FROM_MODEL_CARD"}, "source_LICENSE_weights": source["weights_license"], "historical_v0_0_2_LICENSE_weights": {"git_blob_sha1": HISTORICAL_WEIGHTS_LICENSE_BLOB, "license": "CC-BY-NC-ND-4.0", "status": "HISTORICAL_EVIDENCE_NOT_CURRENT_SOURCE"}, "status": "PROVENANCE_AMBIGUITY_BLOCKER"}, "code": source["license"], "training_data": "UNAUTHENTICATED_BLOCKER"}, dependency_closure=dependency, blockers=sorted(set(blockers)))
        return 2
    except Exception as error:
        write_manifest(args.output or Path("."), inspection_status="INSPECTION_ERROR", collection_status="UNVERIFIED", expected_head=args.expected_head, approval_evidence=approval, upstream={"repository": HF_REPOSITORY, "requested_revision": HF_REVISION, "resolved_revision": None}, error_type=type(error).__name__, blockers=[str(error)])
        return 2


def self_test() -> None:
    global HF_FILE_IDENTITIES
    assert len(HF_REVISION) == 40 and len(SOURCE_REVISION) == 40
    dependency = dependency_audit(PROJECT)
    assert dependency["status"] == "BLOCKED_MISSING_LOCK_INPUTS"
    assert dependency["lock_present"] is False
    pending = pending_approval_scope("a" * 40, dependency)
    assert pending["status"] == "PENDING_OWNER_APPROVAL"
    assert pending["record"] is None and pending["signer"] is None and pending["decision"] is None
    tampered_pending = json.loads(json.dumps(pending))
    tampered_pending["scope"]["publication"] = "UPLOAD"
    assert tampered_pending["scope_sha256"] != hashlib.sha256(canonical_json(tampered_pending["scope"])).hexdigest()
    validate_model_free_options(snapshot=None, approval_evidence=None, approval_sha256=None, expected_head="a" * 40, source=Path("source"), server_tree=Path("tree.json"), output=Path("evidence"), vokra_root=Path("."), t5_server_metadata=Path("t5-tree.json"))
    for mixed in (
        {"snapshot": Path("snapshot"), "approval_evidence": None, "approval_sha256": None},
        {"snapshot": None, "approval_evidence": "approval.json", "approval_sha256": None},
    ):
        try:
            validate_model_free_options(**mixed, expected_head="a" * 40, source=Path("source"), server_tree=Path("tree.json"), output=Path("evidence"), vokra_root=Path("."), t5_server_metadata=Path("t5-tree.json"))
        except RuntimeError:
            pass
        else:
            raise AssertionError("model-free approval/snapshot mix was accepted")
    assert HF_FILES == {".gitattributes", "README.md", "compression_state_dict.bin", "state_dict.bin"}
    assert ARCHIVES["state_dict.bin"] == 3_678_455_287
    synthetic_files = [{"path": name, "bytes": 1, "git_blob_sha1": None, "lfs_pointer_git_blob_sha1": None, "lfs_payload_sha256": None, "payload_status": "NOT_DOWNLOADED"} for name in HF_FILES]
    assert set(model_free_archives(synthetic_files)) == set(ARCHIVES)
    assert "audiocraft/modules/quantization/core_vq.py" not in SOURCE_ROLE_BLOBS
    assert "audiocraft/modules/quantization/vq.py" not in SOURCE_ROLE_BLOBS
    assert set(ROLE_SEMANTIC_PATHS) <= set(SOURCE_ROLE_BLOBS)
    assert SOURCE_WEIGHTS_LICENSE_BLOB == "108b5f002fc31efe11d881de2cd05329ebe8cc37"
    assert HISTORICAL_WEIGHTS_LICENSE_BLOB == "dc1adf98654156baeb94d2e055c224a847e5820d"
    assert "T5-large" not in ROLE_MARKERS
    class DummyTorch:
        Tensor = type("Tensor", (), {})

    with tempfile.TemporaryDirectory(prefix="audiogen-medium-semantic-") as directory:
        semantic_root = Path(directory)
        semantic_path = semantic_root / "config/conditioner/text2sound.yaml"
        semantic_path.parent.mkdir(parents=True)
        semantic_path.write_text("model: t5\nname: t5-large\nfinetune: false\n", encoding="utf-8")
        assert source_semantic_contract(semantic_root, "config/conditioner/text2sound.yaml")["status"] == "AUTHENTICATED_SOURCE_SEMANTICS"

    walked = walk_checkpoint({"cfg": {"sample_rate": 16_000, "frame_rate": 50, "num_codebooks": 4, "text_encoder_name": "t5-small"}}, DummyTorch)
    assert not walked["tensors"] and checkpoint_config(walked["scalars"])["sample_rate_hz"] == 16_000
    cycle: list[Any] = []
    cycle.append(cycle)
    try:
        walk_checkpoint(cycle, DummyTorch)
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
        approval_path = root / "approval.json"
        expected_head = "a" * 40
        scope = approval_scope(expected_head)
        approval_path.write_text(json.dumps({"schema": APPROVAL_SCHEMA, "decision": "APPROVED", "signer": "owner@example.test", "scope": scope, "scope_sha256": hashlib.sha256(canonical_json(scope)).hexdigest()}, sort_keys=True) + "\n", encoding="utf-8")
        approval = validate_approval(approval_path, expected_head, digest(approval_path))
        assert approval["schema"] == APPROVAL_SCHEMA
        approval_path.write_text(approval_path.read_text(encoding="utf-8").replace("owner@example.test", "placeholder"), encoding="utf-8")
        try:
            validate_approval(approval_path, expected_head, digest(approval_path))
        except RuntimeError:
            pass
        else:
            raise AssertionError("placeholder approval signer was accepted")
        approval_path.write_text(json.dumps({"schema": APPROVAL_SCHEMA, "decision": "APPROVED", "signer": "owner@example.test", "scope": scope, "scope_sha256": hashlib.sha256(canonical_json(scope)).hexdigest()}, sort_keys=True) + "\n", encoding="utf-8")
        try:
            validate_approval(approval_path, expected_head, digest(approval_path), raw_path=f"{root}/./approval.json")
        except RuntimeError:
            pass
        else:
            raise AssertionError("dot-component approval path was accepted")
        target = root / "approval-target"
        target.mkdir()
        target_approval = target / "approval.json"
        target_approval.write_bytes(approval_path.read_bytes())
        linked = root / "approval-linked"
        linked.symlink_to(target, target_is_directory=True)
        linked_path = linked / "approval.json"
        try:
            validate_approval(linked_path, expected_head, digest(target_approval), raw_path=str(linked_path))
        except RuntimeError:
            pass
        else:
            raise AssertionError("symlink-ancestor approval path was accepted")
        (root / "duplicate.json").write_text('{"x":1,"x":2}\n', encoding="utf-8")
        try:
            load_json(root / "duplicate.json")
        except RuntimeError:
            pass
        else:
            raise AssertionError("duplicate approval JSON key was accepted")
        existing = root / "existing"
        existing.mkdir()
        (existing / "manifest.json").write_text("sentinel\n", encoding="utf-8")
        try:
            write_manifest(existing, test=True)
        except RuntimeError:
            pass
        else:
            raise AssertionError("existing inspection manifest was clobbered")
        fresh = root / "fresh"
        write_manifest(fresh, test=True)
        manifest = json.loads((fresh / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["runtime_status"] == RUNTIME_STATUS
        assert manifest["cpu_status"] == CPU_STATUS
        assert not list(fresh.glob(".manifest.json.*"))
        original_open = os.open
        temporary_attacker: Path | None = None
        def replace_temporary(path: str | bytes | Path, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
            nonlocal temporary_attacker
            candidate = Path(path)
            if temporary_attacker is None and candidate.parent == root / "temporary-race" and candidate.name.startswith(".manifest.json.") and not (flags & os.O_CREAT):
                candidate.unlink()
                temporary_attacker = candidate
                attacker = original_open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.write(attacker, b"attacker-temp\n")
                os.close(attacker)
            if dir_fd is None:
                return original_open(path, flags, mode)
            return original_open(path, flags, mode, dir_fd=dir_fd)
        os.open = replace_temporary  # type: ignore[assignment]
        temporary_race = root / "temporary-race"
        try:
            try: write_manifest(temporary_race, test=True)
            except RuntimeError as error: assert "temporary identity changed" in str(error)
            else: raise AssertionError("replaced temporary was published")
        finally: os.open = original_open  # type: ignore[assignment]
        assert not (temporary_race / "manifest.json").exists() and temporary_attacker is not None and temporary_attacker.read_bytes() == b"attacker-temp\n"
        temporary_attacker.unlink()
        original_open = os.open
        final_attacker = False
        final_race = root / "final-race"
        def replace_final(path: str | bytes | Path, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
            nonlocal final_attacker
            candidate = Path(path)
            if candidate == final_race / "manifest.json" and not (flags & os.O_CREAT) and not final_attacker:
                final_attacker = True
                candidate.unlink()
                candidate.write_bytes(b"attacker-final\n")
            if dir_fd is None:
                return original_open(path, flags, mode)
            return original_open(path, flags, mode, dir_fd=dir_fd)
        os.open = replace_final  # type: ignore[assignment]
        try:
            try: write_manifest(final_race, test=True)
            except RuntimeError as error: assert "manifest identity changed" in str(error)
            else: raise AssertionError("replaced final was accepted")
        finally: os.open = original_open  # type: ignore[assignment]
        assert final_attacker and (final_race / "manifest.json").read_bytes() == b"attacker-final\n"
        cleanup_race = root / "cleanup-race"
        original_unlink = Path.unlink
        def fail_temporary_cleanup(path: Path, missing_ok: bool = False) -> None:
            if path.parent == cleanup_race and path.name.startswith(".manifest.json."):
                raise OSError("injected cleanup failure")
            original_unlink(path, missing_ok=missing_ok)
        Path.unlink = fail_temporary_cleanup  # type: ignore[assignment]
        try: write_manifest(cleanup_race, test=True)
        finally: Path.unlink = original_unlink  # type: ignore[assignment]
        assert (cleanup_race / "manifest.json").is_file()
        for leaked in cleanup_race.glob(".manifest.json.*"): original_unlink(leaked)
    print("audiogen_medium_inspect --self-test: OK")


if __name__ == "__main__":
    raise SystemExit(main())
