"""Fail-closed packet and source contract for the Kyutai streaming oracle.

This module is intentionally stdlib-only.  It can validate source receipts
and synthetic packet shapes without importing torch or any upstream package.
The real runner is the only code allowed to import the official Moshi source.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Any

HF_REPOSITORY = "kyutai/stt-2.6b-en"
HF_REVISION = "a07aec56d22be5589cd0bc8709c75b6cf3e3039d"
MODEL_BYTES = 5_234_275_128
MODEL_SHA256 = "2471add7da1fdb2d5dc4561e88a9069376333d992760d55d29d1db46c52849b2"
MODEL_TENSOR_MANIFEST_SHA256 = "e62488c9d16953010c758ec17f4c70e8ee30d348adfab3811eb5dfecb435d5df"
CONFIG_SHA256 = "b79ea52a30329887a2d0ce2dd5473a63fc5083e441e7986f64f01050c06239c9"
MIMI_NAME = "mimi-pytorch-e351c8d8@125.safetensors"
MIMI_BYTES = 384_644_900
MIMI_SHA256 = "09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50"
TOKENIZER_NAME = "tokenizer_en_audio_4000.model"
TOKENIZER_BYTES = 59_339
TOKENIZER_SHA256 = "d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6"
DSM_REPOSITORY = "https://github.com/kyutai-labs/delayed-streams-modeling.git"
DSM_REVISION = "4c4f65e147df056adf3346290d64c7b9649b18c9"
MOSHI_REPOSITORY = "https://github.com/kyutai-labs/moshi.git"
MOSHI_REVISION = "e6a55d2722a65870ef52a6c9f6ecfc0e90f38362"
MOSHI_LM_SHA256 = "38991e83d7e3aa0ff1483b27a0b59e1ff43642ad6a32dc8dc4272ad2b6c0dc5f"
MOSHI_TRANSFORMER_SHA256 = "f5a73d752a5bde1eda2b0b14bebd13fd81db81017d361782a07164580a687622"
MODEL_NAME = "model.safetensors"
CONFIG_NAME = "config.json"
CONFIG_BYTES = 1_257
MOSHI_TREE_SHA = "2a6d6afe53d70bac490117651dfc478cf87a940e"
DSM_TREE_SHA = "1ab73718d99c5bb6ff94c1bc84a783a4d2e3a7e0"
MOSHI_ROLE_PACKET_FILES = {
    "moshi/moshi/models/lm.py": ("moshi-python-0010.py", 37312, MOSHI_LM_SHA256, "209b7a59c9c086810a81a7d8b99c5233a0ad87ff"),
    "moshi/moshi/modules/transformer.py": ("moshi-python-0024.py", 38159, MOSHI_TRANSFORMER_SHA256, "244e73a3bc012256325e5b759cd13588a96d2673"),
}
PCM_SOURCE_PACKET_MANIFEST_SHA256 = "85223a7ac8b947eaeafa7b2f337a1ac60dea84a75d44ee0c607df72ad25d6342"
DSM_PCM_ROLE_PACKET_FILES = {
    "scripts/stt_evaluate_on_dataset.py": (
        "dsm-python-0000.py",
        11674,
        "2832c048c77aa8ac4baa5535d5b723d17acb3ac91a33f943a3538d4c563d6dcb",
        "684fe5cc5512c6d2e7802ecfd6152f9b7dcf6373",
    ),
    "scripts/stt_from_file_rust_server.py": (
        "dsm-python-0003.py",
        4265,
        "597b72b69779dc8b9d90a3e1e3ff3d06274dae33f9a08bcd1947ed8b444e7828",
        "9333ca994c7daec57501585b7d7d1d90173a4229",
    ),
}
MOSHI_PCM_ROLE_PACKET_FILES = {
    "moshi/moshi/models/compression.py": (
        "moshi-python-0009.py",
        17429,
        "edc6fbf34e4d84b35f2c2f4d6f1f263c6ba86329a7c686b091f304a6b3cbfc8f",
        "33730232ebf320addc7a7ebfb7f61a978c6e8434",
    ),
    "moshi/moshi/modules/streaming.py": (
        "moshi-python-0023.py",
        8433,
        "bafaaafd12291727a6a613e9ac68623dc924fa6cdf57cdc9874f66fb30464d38",
        "7ea665fbd6669c21870e909916cc96ef70a8dbb2",
    ),
}
MOSHI_COMMIT_PACKET = ("moshi-commit.json", MOSHI_REVISION, MOSHI_TREE_SHA)
DSM_COMMIT_PACKET = ("dsm-commit.json", DSM_REVISION, DSM_TREE_SHA)
N_Q = 32
TEXT_CARD = 4000
AUDIO_CARD = 2048
CONTEXT = 375
MAX_CONTEXT = 4096
MAX_PACKET_BYTES = 2 * 1024 * 1024
MAX_EVENTS = 48 * 4096
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
STREAMING_SCHEMA = "vokra-kyutai-stt-independent-streaming-reference-v1"
STREAMING_SCOPE = (
    "official Moshi LMGen main-transformer KV cache only; no Mimi PCM, "
    "depformer, tokenizer, transcription, publication, or Apple parity claim"
)
NO_EXECUTION_STATUS = "NOT_RUN_NO_REAL_FIXTURES"
REAL_STATUS = "REFERENCE_READY"
COMPOSITE_APPROVAL_SCHEMA = "vokra-kyutai-stt-pytorch-pcm-oracle-approval-v1"
COMPOSITE_APPROVAL_SCOPE = "KYUTAI_STT_PYTORCH_PCM_ORACLE_CAPTURE"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SOURCE_MANIFEST_MAX_BYTES = 4 * 1024 * 1024
SOURCE_ROLE = "authenticated-source"

# Execution readiness is deliberately stricter than the six-field approval
# receipt below.  No dependency closure has passed the source/build/license/
# owner review required for execution, so this immutable set must remain empty
# until a separately reviewed exact closure is recorded in this module.
EXECUTION_APPROVAL_MAX_BYTES = 64 * 1024
EXECUTION_READINESS_STATUS = "BLOCKED_DEPENDENCY_CLOSURE"
REVIEWED_DEPENDENCY_CLOSURE_SHA256: frozenset[str] = frozenset()


class ExecutionReadinessBlocked(ValueError):
    """Fail-closed execution refusal with a machine-readable status."""

    status = EXECUTION_READINESS_STATUS


def _read_bounded(path: Path, label: str, limit: int, *, expected_size: int | None = None) -> bytes:
    """Read a bounded file once, optionally enforcing its manifest size."""
    real_path(path, label)
    if expected_size is not None and (expected_size < 0 or expected_size > limit):
        raise ValueError(f"{label} exceeds its bounded size")
    read_limit = limit + 1
    if expected_size is not None:
        read_limit = min(read_limit, expected_size + 1)
    body = bytearray()
    with path.open("rb") as stream:
        while len(body) < read_limit:
            block = stream.read(min(1 << 20, read_limit - len(body)))
            if not block:
                break
            body.extend(block)
    if len(body) > limit:
        raise ValueError(f"{label} exceeds its byte bound")
    if expected_size is not None and len(body) != expected_size:
        raise ValueError(f"{label} size changed while reading")
    return bytes(body)


def _json_object(body: bytes, label: str) -> dict[str, Any]:
    try:
        document = json.loads(body.decode("utf-8"), object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not valid unique-key JSON") from error
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be a JSON object")
    return document


def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(body: bytes) -> str:
    return hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()


def real_path(path: Path, label: str, *, directory: bool = False) -> Path:
    if not path.is_absolute() or path.is_symlink() or any(part in {".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be an absolute non-symlink path")
    if directory and not path.is_dir():
        raise ValueError(f"{label} must be a directory")
    if not directory and not path.is_file():
        raise ValueError(f"{label} must be a file")
    for ancestor in path.parents:
        # macOS exposes /private as the system's stable alias for the data
        # volume.  It is infrastructure, not a caller-controlled checkout
        # component; every other symlinked ancestor remains forbidden.
        if ancestor.is_symlink() and ancestor not in {Path("/private"), Path("/var"), Path("/tmp")}:
            raise ValueError(f"{label} has symlink ancestry")
    if directory and path.is_file():
        raise ValueError(f"{label} must be a directory")
    if not directory and not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    return path


def git_identity(root: Path, repository: str, revision: str) -> dict[str, str]:
    real_path(root, "source checkout", directory=True)
    try:
        actual_revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
        origin = subprocess.run(
            ["git", "-C", str(root), "remote", "get-url", "origin"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"cannot authenticate source checkout {root}: {error}") from error
    if actual_revision != revision:
        raise ValueError(f"source revision mismatch: {actual_revision!r} != {revision!r}")
    allowed = {repository, repository.removesuffix(".git"), repository.removesuffix(".git").replace("https://github.com/", "git@github.com:")}
    if origin not in allowed:
        raise ValueError(f"source origin mismatch: {origin!r}")
    if status:
        raise ValueError("source checkout is dirty")
    return {"repository": repository, "revision": revision, "origin": origin}


def require_source_packet(packet: Path | None) -> dict[str, Any] | None:
    """Validate the optional flat source receipt without treating it as code."""
    if packet is None:
        return None
    real_path(packet, "source packet", directory=True)
    manifest_path = real_path(packet / "manifest.json", "source packet manifest")
    manifest_body = _read_bounded(manifest_path, "source packet manifest", SOURCE_MANIFEST_MAX_BYTES)
    document = _json_object(manifest_body, "source packet manifest")
    if document.get("schema") != "vokra-flat-source-receipt-v1" or document.get("status") != "SOURCE_ONLY_PREPARATION":
        raise ValueError("source packet is not the authenticated source-only receipt")
    files = document.get("files", [])
    if not isinstance(files, list) or len(files) > 512:
        raise ValueError("source packet file inventory is malformed or too large")
    by_name: dict[str, dict[str, Any]] = {}
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"name", "role", "size", "sha256"}:
            raise ValueError("source packet manifest entry is malformed")
        name = entry.get("name")
        if not isinstance(name, str) or not name or name in by_name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("source packet manifest names must be unique safe basenames")
        if entry.get("role") != SOURCE_ROLE:
            raise ValueError("source packet manifest role is invalid")
        if not isinstance(entry.get("size"), int) or isinstance(entry.get("size"), bool) or entry["size"] < 0 or entry["size"] > MAX_PACKET_BYTES:
            raise ValueError("source packet entry size is invalid")
        if not isinstance(entry.get("sha256"), str) or not HEX64.fullmatch(entry["sha256"]):
            raise ValueError("source packet entry digest is invalid")
        by_name[name] = entry

    required_names = {
        MOSHI_COMMIT_PACKET[0],
        DSM_COMMIT_PACKET[0],
        "moshi-tree.json",
        *(packet_name for packet_name, _bytes, _digest, _blob in MOSHI_ROLE_PACKET_FILES.values()),
    }
    missing = required_names - by_name.keys()
    if missing:
        raise ValueError(f"source packet manifest is missing consumed files: {sorted(missing)}")

    authenticated: dict[str, bytes] = {}
    for name, entry in by_name.items():
        body = _read_bounded(
            packet / name,
            f"source packet {name}",
            MAX_PACKET_BYTES,
            expected_size=entry["size"],
        )
        if hashlib.sha256(body).hexdigest() != entry["sha256"]:
            raise ValueError(f"source packet manifest digest mismatch: {name}")
        if name in required_names:
            authenticated[name] = body

    for name, revision, tree in (MOSHI_COMMIT_PACKET, DSM_COMMIT_PACKET):
        payload = _json_object(authenticated[name], f"source packet {name}")
        commit = payload.get("commit")
        commit_tree = commit.get("tree") if isinstance(commit, dict) else None
        if payload.get("sha") != revision or not isinstance(commit_tree, dict) or commit_tree.get("sha") != tree:
            raise ValueError(f"source packet revision mismatch: {name}")
    tree_doc = _json_object(authenticated["moshi-tree.json"], "moshi tree")
    if tree_doc.get("sha") != MOSHI_TREE_SHA or tree_doc.get("truncated") is not False:
        raise ValueError("Moshi source tree is not the authenticated complete tree")
    tree_rows = tree_doc.get("tree")
    if not isinstance(tree_rows, list) or len(tree_rows) > 10000:
        raise ValueError("Moshi source tree is malformed")
    tree_by_path: dict[str, dict[str, Any]] = {}
    base_tree_keys = {"path", "mode", "type", "sha", "url"}
    for row in tree_rows:
        if not isinstance(row, dict) or (set(row) != base_tree_keys and set(row) != base_tree_keys | {"size"}):
            raise ValueError("Moshi source tree row is malformed")
        source_path = row.get("path")
        if not isinstance(source_path, str) or not source_path or "\x00" in source_path or source_path in tree_by_path:
            raise ValueError("Moshi source tree paths must be unique safe strings")
        if not isinstance(row.get("mode"), str) or not isinstance(row.get("type"), str) or not isinstance(row.get("url"), str):
            raise ValueError("Moshi source tree row types are malformed")
        if not isinstance(row.get("sha"), str) or not HEX40.fullmatch(row["sha"]):
            raise ValueError("Moshi source tree row digest is malformed")
        if "size" in row and (
            not isinstance(row["size"], int)
            or isinstance(row["size"], bool)
            or not 0 <= row["size"] <= MAX_ARTIFACT_BYTES
        ):
            raise ValueError("Moshi source tree row size is malformed")
        tree_by_path[source_path] = row
    for source_path, (packet_name, expected_bytes, expected_hash, expected_blob) in MOSHI_ROLE_PACKET_FILES.items():
        entry = by_name[packet_name]
        body = authenticated[packet_name]
        row = tree_by_path.get(source_path)
        if entry.get("size") != expected_bytes or hashlib.sha256(body).hexdigest() != expected_hash:
            raise ValueError(f"source packet role digest mismatch: {source_path}")
        if not isinstance(row, dict) or row.get("type") != "blob" or row.get("mode") != "100644" or row.get("size") != expected_bytes or row.get("sha") != expected_blob:
            raise ValueError(f"source tree role binding mismatch: {source_path}")
        if git_blob_sha1(body) != expected_blob:
            raise ValueError(f"source packet Git blob binding mismatch: {source_path}")
    return {
        "path": str(packet),
        "manifest_sha256": hashlib.sha256(manifest_body).hexdigest(),
        "moshi_lm_sha256": MOSHI_LM_SHA256,
        "moshi_transformer_sha256": MOSHI_TRANSFORMER_SHA256,
        "status": "SOURCE_ONLY_PREPARATION",
    }


def require_pcm_source_packet(packet: Path) -> dict[str, Any]:
    """Authenticate the retained DSM/Moshi source roles for full PCM work.

    This is intentionally separate from :func:`require_source_packet`: the
    streaming-LM receipt remains source-only and keeps its narrower role set.
    The returned receipt is also source-only; it cannot authorize approval,
    dependency closure, model loading, or upstream execution.
    """
    real_path(packet, "PCM source packet", directory=True)
    manifest_path = real_path(packet / "manifest.json", "PCM source packet manifest")
    manifest_body = _read_bounded(
        manifest_path, "PCM source packet manifest", SOURCE_MANIFEST_MAX_BYTES
    )
    if hashlib.sha256(manifest_body).hexdigest() != PCM_SOURCE_PACKET_MANIFEST_SHA256:
        raise ValueError("PCM source packet manifest digest mismatch")
    document = _json_object(manifest_body, "PCM source packet manifest")
    if set(document) != {"schema", "status", "source_revision", "total_bytes", "files"}:
        raise ValueError("PCM source packet manifest keys mismatch")
    if document["schema"] != "vokra-flat-source-receipt-v1" or document["status"] != "SOURCE_ONLY_PREPARATION":
        raise ValueError("PCM source packet is not source-only preparation")
    if document["source_revision"] != "authenticated":
        raise ValueError("PCM source packet source revision is not authenticated")
    files = document["files"]
    if not isinstance(files, list) or len(files) > 512:
        raise ValueError("PCM source packet file inventory is malformed or too large")
    total_bytes = document["total_bytes"]
    if not isinstance(total_bytes, int) or isinstance(total_bytes, bool) or not 0 <= total_bytes <= MAX_PACKET_BYTES:
        raise ValueError("PCM source packet total size is invalid")
    by_name: dict[str, dict[str, Any]] = {}
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"name", "role", "size", "sha256"}:
            raise ValueError("PCM source packet manifest entry is malformed")
        name = entry["name"]
        if (
            not isinstance(name, str)
            or not name
            or name in by_name
            or Path(name).name != name
            or name in {".", ".."}
        ):
            raise ValueError("PCM source packet manifest names must be unique safe basenames")
        if entry["role"] != SOURCE_ROLE:
            raise ValueError("PCM source packet manifest role is invalid")
        size = entry["size"]
        if not isinstance(size, int) or isinstance(size, bool) or not 0 <= size <= MAX_PACKET_BYTES:
            raise ValueError("PCM source packet entry size is invalid")
        digest = entry["sha256"]
        if not isinstance(digest, str) or HEX64.fullmatch(digest) is None:
            raise ValueError("PCM source packet entry digest is invalid")
        by_name[name] = entry
    if sum(entry["size"] for entry in by_name.values()) != total_bytes:
        raise ValueError("PCM source packet total size does not match entries")

    role_files = {**DSM_PCM_ROLE_PACKET_FILES, **MOSHI_PCM_ROLE_PACKET_FILES}
    required_names = {
        DSM_COMMIT_PACKET[0],
        MOSHI_COMMIT_PACKET[0],
        "dsm-tree.json",
        "moshi-tree.json",
        *(packet_name for packet_name, _size, _digest, _blob in role_files.values()),
    }
    missing = required_names - by_name.keys()
    if missing:
        raise ValueError(f"PCM source packet is missing consumed files: {sorted(missing)}")

    authenticated: dict[str, bytes] = {}
    for name, entry in by_name.items():
        body = _read_bounded(
            packet / name,
            f"PCM source packet {name}",
            MAX_PACKET_BYTES,
            expected_size=entry["size"],
        )
        if hashlib.sha256(body).hexdigest() != entry["sha256"]:
            raise ValueError(f"PCM source packet manifest digest mismatch: {name}")
        if name in required_names:
            authenticated[name] = body

    for name, revision, tree in (DSM_COMMIT_PACKET, MOSHI_COMMIT_PACKET):
        payload = _json_object(authenticated[name], f"PCM source packet {name}")
        commit = payload.get("commit")
        commit_tree = commit.get("tree") if isinstance(commit, dict) else None
        if payload.get("sha") != revision or not isinstance(commit_tree, dict) or commit_tree.get("sha") != tree:
            raise ValueError(f"PCM source packet revision mismatch: {name}")

    tree_by_source: dict[str, dict[str, dict[str, Any]]] = {}
    for tree_name, tree_digest, tree_key in (
        ("dsm-tree.json", DSM_TREE_SHA, "dsm"),
        ("moshi-tree.json", MOSHI_TREE_SHA, "moshi"),
    ):
        tree_doc = _json_object(authenticated[tree_name], f"PCM source packet {tree_name}")
        if tree_doc.get("sha") != tree_digest or tree_doc.get("truncated") is not False:
            raise ValueError(f"PCM source tree is not complete: {tree_key}")
        rows = tree_doc.get("tree")
        if not isinstance(rows, list) or len(rows) > 10000:
            raise ValueError(f"PCM source tree is malformed: {tree_key}")
        by_path: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"PCM source tree row is malformed: {tree_key}")
            if set(row) not in ({"path", "mode", "type", "sha", "url"}, {"path", "mode", "type", "sha", "url", "size"}):
                raise ValueError(f"PCM source tree row keys are malformed: {tree_key}")
            path = row.get("path")
            if not isinstance(path, str) or not path or "\x00" in path or path in by_path:
                raise ValueError(f"PCM source tree paths are not unique: {tree_key}")
            if not isinstance(row.get("mode"), str) or not isinstance(row.get("type"), str) or not isinstance(row.get("url"), str):
                raise ValueError(f"PCM source tree row types are malformed: {tree_key}")
            if not isinstance(row.get("sha"), str) or HEX40.fullmatch(row["sha"]) is None:
                raise ValueError(f"PCM source tree row digest is malformed: {tree_key}")
            if "size" in row and (not isinstance(row["size"], int) or isinstance(row["size"], bool) or not 0 <= row["size"] <= MAX_ARTIFACT_BYTES):
                raise ValueError(f"PCM source tree row size is malformed: {tree_key}")
            by_path[path] = row
        tree_by_source[tree_key] = by_path

    for source_path, (packet_name, expected_size, expected_sha, expected_blob) in role_files.items():
        tree_key = "dsm" if source_path.startswith("scripts/") else "moshi"
        entry = by_name[packet_name]
        body = authenticated[packet_name]
        row = tree_by_source[tree_key].get(source_path)
        if entry["size"] != expected_size or hashlib.sha256(body).hexdigest() != expected_sha:
            raise ValueError(f"PCM source role digest mismatch: {source_path}")
        if not isinstance(row, dict) or row.get("type") != "blob" or row.get("mode") != "100644" or row.get("size") != expected_size or row.get("sha") != expected_blob:
            raise ValueError(f"PCM source tree role binding mismatch: {source_path}")
        if git_blob_sha1(body) != expected_blob:
            raise ValueError(f"PCM source Git blob binding mismatch: {source_path}")

    return {
        "path": str(packet),
        "manifest_sha256": hashlib.sha256(manifest_body).hexdigest(),
        "dsm_tree_sha": DSM_TREE_SHA,
        "moshi_tree_sha": MOSHI_TREE_SHA,
        "status": "SOURCE_ONLY_PREPARATION",
    }


def validate_input_packet(document: Any, *, expected_context: int = CONTEXT) -> None:
    if not isinstance(document, dict) or set(document) != {"audio_codes", "context", "frames", "kind", "model_revision"}:
        raise ValueError("streaming input packet keys mismatch")
    if document["kind"] != "deterministic_mimi_code_boundary_packet":
        raise ValueError("streaming input is not the approved boundary packet kind")
    context = document["context"]
    if document["model_revision"] != HF_REVISION or context != expected_context:
        raise ValueError("streaming input identity mismatch")
    if not isinstance(context, int) or isinstance(context, bool) or not 1 <= context <= MAX_CONTEXT:
        raise ValueError("streaming context must be a bounded integer")
    codes = document["audio_codes"]
    if not isinstance(codes, list) or len(codes) != expected_context + 2 or document["frames"] != len(codes):
        raise ValueError("streaming input frame count mismatch")
    if len(codes) > MAX_CONTEXT + 2:
        raise ValueError("streaming input exceeds frame bound")
    for frame in codes:
        if not isinstance(frame, list) or len(frame) != N_Q or any(not isinstance(code, int) or isinstance(code, bool) or not 0 <= code < AUDIO_CARD for code in frame):
            raise ValueError("streaming input audio code shape/range mismatch")
    if len(json.dumps(document, separators=(",", ":"), sort_keys=True).encode()) > MAX_PACKET_BYTES:
        raise ValueError("streaming input packet exceeds byte bound")


def boundary_packet(context: int = CONTEXT) -> dict[str, Any]:
    if not isinstance(context, int) or isinstance(context, bool) or not 1 <= context <= MAX_CONTEXT:
        raise ValueError("streaming context must be a bounded positive integer")
    codes = [[(frame * 37 + channel * 11) % AUDIO_CARD for channel in range(N_Q)] for frame in range(context + 2)]
    return {
        "kind": "deterministic_mimi_code_boundary_packet",
        "model_revision": HF_REVISION,
        "context": context,
        "frames": len(codes),
        "audio_codes": codes,
    }


def validate_composite_approval(path: Path, expected_head: str, expected_sha256: str, checkout: Path) -> dict[str, Any]:
    """Authenticate the separate PCM-composite approval; decoder approval never substitutes."""
    real_path(path, "composite approval")
    if sha256(path) != expected_sha256:
        raise ValueError("composite approval digest mismatch")
    document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    if document.get("schema") != COMPOSITE_APPROVAL_SCHEMA or document.get("scope") != COMPOSITE_APPROVAL_SCOPE:
        raise ValueError("composite approval scope/schema mismatch")
    if document.get("decision") != "APPROVED" or document.get("execution") != "VAST_ONLY":
        raise ValueError("composite approval is not an approved VAST execution")
    if document.get("head") != expected_head or document.get("checkout") != str(checkout):
        raise ValueError("composite approval identity mismatch")
    return document


def _require_lower_hex(value: object, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase hexadecimal")
    return value


def _bounded_composite_approval(
    approval: Path,
    *,
    expected_head: str,
    expected_approval_sha256: str,
    checkout: Path,
) -> dict[str, object]:
    """Validate the external six-field receipt without unbounded reads."""

    approval = real_path(approval, "composite approval")
    checkout = real_path(checkout, "checkout", directory=True)
    try:
        approval.relative_to(checkout)
    except ValueError:
        pass
    else:
        raise ValueError("composite approval must be outside checkout")

    size = approval.stat().st_size
    body = _read_bounded(
        approval,
        "composite approval",
        EXECUTION_APPROVAL_MAX_BYTES,
        expected_size=size,
    )
    observed = hashlib.sha256(body).hexdigest()
    if observed != expected_approval_sha256:
        raise ValueError("composite approval sha256 mismatch")
    document = _json_object(body, "composite approval")
    expected_keys = {
        "schema",
        "scope",
        "decision",
        "execution",
        "head",
        "checkout",
    }
    if set(document) != expected_keys:
        raise ValueError("composite approval keys mismatch")
    if document["schema"] != COMPOSITE_APPROVAL_SCHEMA:
        raise ValueError("composite approval schema mismatch")
    if document["scope"] != COMPOSITE_APPROVAL_SCOPE:
        raise ValueError("composite approval scope mismatch")
    if document["decision"] != "APPROVED":
        raise ValueError("composite approval is not approved")
    if document["execution"] != "VAST_ONLY":
        raise ValueError("composite approval execution mismatch")
    if document["head"] != expected_head:
        raise ValueError("composite approval head mismatch")
    if document["checkout"] != str(checkout):
        raise ValueError("composite approval checkout mismatch")
    return document


def require_execution_readiness(
    approval: Path,
    *,
    expected_head: str,
    expected_approval_sha256: str,
    checkout: Path,
    dependency_closure_sha256: str,
    platform_system: str | None = None,
    platform_machine: str | None = None,
) -> dict[str, object]:
    """Require all preconditions before model authentication/import.

    This boundary performs only bounded receipt and path checks. It never opens
    model files. The code-side allowlist is intentionally empty at present, so
    a syntactically valid ``APPROVED`` receipt or caller-provided closure
    digest cannot authorize execution.
    """

    expected_head = _require_lower_hex(expected_head, HEX40, "expected head")
    expected_approval_sha256 = _require_lower_hex(
        expected_approval_sha256, HEX64, "expected approval sha256"
    )
    dependency_closure_sha256 = _require_lower_hex(
        dependency_closure_sha256, HEX64, "dependency closure sha256"
    )
    system = platform_system if platform_system is not None else platform.system()
    machine = platform_machine if platform_machine is not None else platform.machine()
    if system != "Linux" or machine != "x86_64":
        raise ExecutionReadinessBlocked(
            "execution requires Linux x86_64; model access is not permitted"
        )

    # Validate the receipt before checking the closure so malformed or
    # checkout-local approvals cannot be mistaken for a closure decision.
    document = _bounded_composite_approval(
        approval,
        expected_head=expected_head,
        expected_approval_sha256=expected_approval_sha256,
        checkout=checkout,
    )
    if dependency_closure_sha256 not in REVIEWED_DEPENDENCY_CLOSURE_SHA256:
        raise ExecutionReadinessBlocked(
            f"{EXECUTION_READINESS_STATUS}: no reviewed dependency closure"
        )
    # This branch is deliberately unreachable while the reviewed set is empty;
    # retaining the returned fields makes the future reviewed-closure contract
    # explicit without treating synthetic receipts as approval.
    return {
        "status": "READY",
        "head": expected_head,
        "approval_sha256": expected_approval_sha256,
        "dependency_closure_sha256": dependency_closure_sha256,
        "approval": document,
    }


def authenticate_composite_model(model_root: Path) -> dict[str, Any]:
    """Authenticate the four exact inputs for the later PCM composite run."""
    real_path(model_root, "model input directory", directory=True)
    required = {
        MODEL_NAME: (MODEL_BYTES, MODEL_SHA256),
        MIMI_NAME: (MIMI_BYTES, MIMI_SHA256),
        TOKENIZER_NAME: (TOKENIZER_BYTES, TOKENIZER_SHA256),
        CONFIG_NAME: (CONFIG_BYTES, CONFIG_SHA256),
    }
    records: dict[str, Any] = {}
    for name, (expected_bytes, expected_hash) in required.items():
        path = real_path(model_root / name, f"model input {name}")
        observed_bytes = path.stat().st_size
        if expected_bytes is not None and observed_bytes != expected_bytes:
            raise ValueError(f"model input size mismatch: {name}")
        observed_hash = sha256(path)
        if observed_hash != expected_hash:
            raise ValueError(f"model input digest mismatch: {name}")
        records[name] = {"bytes": observed_bytes, "sha256": observed_hash}
    return {"status": "AUTHENTICATED_FOUR_FILE_COMPOSITE_INPUT", "files": records}
