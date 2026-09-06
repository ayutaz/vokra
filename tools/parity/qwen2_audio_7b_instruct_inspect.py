#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Fail-closed inspection oracle for Qwen2-Audio-7B-Instruct.

The five-shard checkpoint is inspected only on VAST.  Safetensors headers and
data regions are checked without materializing tensor bodies; JSON/source,
license, and exact Transformers evidence are recorded independently.  This
tool does not convert, execute, or claim CPU/Metal/parity support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

UPSTREAM_REPOSITORY = "Qwen/Qwen2-Audio-7B-Instruct"
UPSTREAM_REVISION = "0a095220c30b7b31434169c3086508ef3ea5bf0a"
SOURCE_REPOSITORY = "https://github.com/QwenLM/Qwen2-Audio.git"
SOURCE_REPOSITORY_PATH = "QwenLM/Qwen2-Audio"
SOURCE_REVISION = "595360e82b5839c1507492ec83cae5bda6d5c7d4"
TRANSFORMERS_REPOSITORY = "https://github.com/huggingface/transformers"
TRANSFORMERS_REPOSITORY_PATH = "huggingface/transformers"
TRANSFORMERS_TAG = "v4.45.0"
TRANSFORMERS_REVISION = "2ef31dec1676249d26044a8aa8abe33dbecf0d10"
FORMAT = "vokra-qwen2-audio-7b-instruct-inspection-v2"
APPROVAL_SCHEMA = "vokra-qwen2-audio-7b-instruct-blocked-approval-v1"
APPROVAL_SCOPE = "QWEN2_AUDIO_7B_INSTRUCT_INSPECTION"
MAX_HEADER_BYTES = 64 * 1024 * 1024
SHARD_COUNT = 5
SHARD_NAMES = {f"model-{number:05d}-of-{SHARD_COUNT:05d}.safetensors" for number in range(1, SHARD_COUNT + 1)}
EXPECTED_COMPONENT_FIELDS = {
    "audio_encoder_layers": ("audio_config", "encoder_layers", 32),
    "audio_encoder_heads": ("audio_config", "encoder_attention_heads", 20),
    "audio_encoder_width": ("audio_config", "d_model", 1280),
    "audio_encoder_ffn": ("audio_config", "encoder_ffn_dim", 5120),
    "audio_mel_bins": ("audio_config", "num_mel_bins", 128),
    "audio_max_source_positions": ("audio_config", "max_source_positions", 1500),
    "audio_token_index": ("audio_token_index", "audio_token_index", 151646),
    "text_vocab": ("text_config", "vocab_size", 156032),
    "text_context": ("text_config", "max_position_embeddings", 8192),
    "text_ffn": ("text_config", "intermediate_size", 11008),
    "text_eos": ("text_config", "eos_token_id", 151645),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def external_path(raw: str, label: str, repo_root: Path, require_file: bool = False) -> Path:
    if not isinstance(raw, str) or not raw.startswith("/") or "//" in raw or "/./" in raw or "/../" in raw or raw.endswith(("/.", "/..")):
        raise RuntimeError(f"{label} must be an absolute dot-free path")
    path = Path(raw)
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise RuntimeError(f"{label} must be an absolute dot-free path")
    current = Path("/")
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise RuntimeError(f"{label} contains a symlink ancestor")
    root = repo_root.resolve()
    resolved = path.resolve(strict=False)
    if resolved == root or root in resolved.parents:
        raise RuntimeError(f"{label} must be outside the checkout")
    if path.is_symlink():
        raise RuntimeError(f"{label} must not be a symlink")
    if require_file and (not path.is_file() or path.is_symlink()):
        raise RuntimeError(f"{label} must be a regular file")
    return path


def validate_approval(path: str, expected_head: str, expected_sha256: str, repo_root: Path) -> dict[str, Any]:
    if len(expected_head) != 40 or any(character not in "0123456789abcdef" for character in expected_head) or len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256):
        raise RuntimeError("approval binding must use lowercase HEAD40 and SHA25664")
    approval = external_path(path, "approval evidence", repo_root, require_file=True)
    raw = approval.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError("approval evidence SHA-256 mismatch")
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, RuntimeError) as error:
        raise RuntimeError(f"approval evidence is not strict UTF-8 JSON: {error}") from error
    keys = {
        "schema", "status", "decision", "expected_head", "upstream_repository", "upstream_revision",
        "source_repository", "source_revision", "transformers_repository", "transformers_tag", "transformers_revision",
        "model_license", "source_license_status", "dependency_license_status", "dataset_status",
        "native_runtime_status", "audio_language_parity_status", "no_upload", "scope",
    }
    if not isinstance(data, dict) or set(data) != keys:
        raise RuntimeError("approval schema is not exact")
    if data.get("no_upload") is not True:
        raise RuntimeError("approval no_upload must be a JSON boolean true")
    expected = {
        "schema": APPROVAL_SCHEMA, "status": "BLOCKED", "decision": "BLOCKED_INSPECTION_ONLY",
        "expected_head": expected_head, "upstream_repository": UPSTREAM_REPOSITORY, "upstream_revision": UPSTREAM_REVISION,
        "source_repository": SOURCE_REPOSITORY, "source_revision": SOURCE_REVISION,
        "transformers_repository": TRANSFORMERS_REPOSITORY, "transformers_tag": TRANSFORMERS_TAG, "transformers_revision": TRANSFORMERS_REVISION,
        "model_license": "APACHE-2.0", "source_license_status": "SOURCE_LICENSE_UNKNOWN_BLOCKER",
        "dependency_license_status": "REQUIRES_PRIMARY_REVIEW", "dataset_status": "BLOCKED_UNAUTHENTICATED",
        "native_runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "audio_language_parity_status": "NOT_RUN",
        "no_upload": True, "scope": APPROVAL_SCOPE,
    }
    if data != expected:
        raise RuntimeError("approval identity/scope/disposition mismatch")
    return data


def require_clean_head(expected_head: str) -> Path:
    root = Path(__file__).resolve().parents[2]
    if len(expected_head) != 40 or any(character not in "0123456789abcdef" for character in expected_head):
        raise RuntimeError("expected_head must be lowercase HEX40")
    actual = git(root, "rev-parse", "HEAD")
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True, stderr=subprocess.STDOUT)
    if dirty or actual != expected_head:
        raise RuntimeError("checkout must be clean and match --expected-head")
    return root


def blocked_preflight(expected_head: str, approval_evidence: str, approval_sha256: str) -> dict[str, Any]:
    root = require_clean_head(expected_head)
    approval = validate_approval(approval_evidence, expected_head, approval_sha256, root)
    if approval["status"] != "BLOCKED" or approval["decision"] != "BLOCKED_INSPECTION_ONLY":
        raise RuntimeError("only the exact blocked inspection disposition is accepted")
    return approval


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    size = path.stat().st_size
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(source), *args], text=True, stderr=subprocess.STDOUT).strip()


def canonicalize_github_remote(remote: str, expected_path: str) -> str:
    """Accept only canonical GitHub HTTPS/SSH spellings for one repository."""
    value = remote[:-1] if remote.endswith("/") else remote
    if value.startswith("https://github.com/"):
        path = value.removeprefix("https://github.com/")
    elif value.startswith("ssh://git@github.com/"):
        path = value.removeprefix("ssh://git@github.com/")
    elif value.startswith("git@github.com:"):
        path = value.removeprefix("git@github.com:")
    else:
        raise RuntimeError(f"unsupported GitHub remote form: {remote!r}")
    if path.endswith("/"):
        raise RuntimeError(f"unsafe GitHub remote path: {remote!r}")
    if path.endswith(".git"):
        path = path[:-4]
    if path != expected_path:
        raise RuntimeError(f"GitHub remote repository mismatch: {remote!r}")
    return f"https://github.com/{path}"


def json_files(snapshot: Path) -> list[Path]:
    return sorted(path for path in snapshot.rglob("*.json") if path.is_file() and not path.is_symlink())


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def strict_json_loads(raw: str, source: str) -> Any:
    try:
        return json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, RuntimeError) as error:
        raise RuntimeError(f"invalid canonical JSON {source}: {error}") from error


def load_json(path: Path) -> Any:
    try:
        return strict_json_loads(path.read_text(encoding="utf-8"), str(path))
    except (OSError, RuntimeError) as error:
        raise RuntimeError(f"invalid canonical JSON {path}: {error}") from error


def license_record(path: Path, root: Path, declared: str) -> dict[str, Any]:
    value = declared.strip()
    if not value or value.lower() in {"unknown", "none", "null", "-"}:
        raise RuntimeError(f"license declaration is unknown: {path}")
    return {"license": value, "path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}


def find_license(root: Path, *, source_unknown_is_blocker: bool = False, require_apache: bool = False) -> dict[str, Any]:
    for path in sorted(root.glob("LICENSE*")):
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            if "apache license" in text and "version 2.0" in text:
                return license_record(path, root, "Apache-2.0")
            if not require_apache and "permission is hereby granted, free of charge" in text:
                return license_record(path, root, "MIT")
    if source_unknown_is_blocker:
        return {
            "license": "UNKNOWN",
            "status": "SOURCE_LICENSE_UNKNOWN_BLOCKER",
            "path": None,
            "bytes": None,
            "sha256": None,
        }
    readme = root / "README.md"
    if readme.is_file():
        for line in readme.read_text(encoding="utf-8", errors="replace").splitlines()[:100]:
            key, separator, value = line.partition(":")
            if separator and key.strip().lower() in {"license", "license_spdx"}:
                if value.strip().lower() in {"apache-2.0", "apache 2.0", "apache license 2.0"}:
                    return license_record(readme, root, "Apache-2.0")
                if not require_apache:
                    return license_record(readme, root, value)
    if require_apache:
        raise RuntimeError(f"canonical Apache-2.0 license evidence is missing in {root}")
    raise RuntimeError(f"no identifiable primary license in {root}")


def require_exact(document: Any, path: tuple[str, ...], expected: Any) -> str:
    value = document
    for component in path:
        if not isinstance(value, dict) or component not in value:
            raise RuntimeError(f"canonical JSON lacks exact field {'.'.join(path)}")
        value = value[component]
    if value != expected:
        raise RuntimeError(f"canonical JSON field {'.'.join(path)} mismatch: expected {expected!r}, actual={value!r}")
    return ".".join(path)


def require_exact_int_list(document: Any, path: tuple[str, ...], expected: list[int]) -> str:
    """Require an integer list with exact type, length, order, and values."""
    value = document
    for component in path:
        if not isinstance(value, dict) or component not in value:
            raise RuntimeError(f"canonical JSON lacks exact field {'.'.join(path)}")
        value = value[component]
    if type(value) is not list or len(value) != len(expected) or any(type(item) is not int for item in value) or value != expected:
        raise RuntimeError(f"canonical JSON field {'.'.join(path)} mismatch: expected exact integer list {expected!r}, actual={value!r}")
    return ".".join(path)


def parse_model_json(snapshot: Path) -> dict[str, Any]:
    config = snapshot / "config.json"
    processor = snapshot / "preprocessor_config.json"
    generation = snapshot / "generation_config.json"
    tokenizer = snapshot / "tokenizer_config.json"
    required = {"config": config, "preprocessor": processor, "generation": generation, "tokenizer": tokenizer}
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing canonical JSON companions: {missing}")
    documents = {name: load_json(path) for name, path in required.items()}
    paths = {"architectures": require_exact(documents["config"], ("architectures",), ["Qwen2AudioForConditionalGeneration"])}
    for label, (parent, key, expected) in EXPECTED_COMPONENT_FIELDS.items():
        path = (key,) if parent == key else (parent, key)
        paths[label] = require_exact(documents["config"], path, expected)
    preprocessor_paths = {
        "sampling_rate": require_exact(documents["preprocessor"], ("sampling_rate",), 16000),
        "n_fft": require_exact(documents["preprocessor"], ("n_fft",), 400),
        "hop_length": require_exact(documents["preprocessor"], ("hop_length",), 160),
        "chunk_length": require_exact(documents["preprocessor"], ("chunk_length",), 30),
        "feature_size": require_exact(documents["preprocessor"], ("feature_size",), 128),
        "max_frames": require_exact(documents["preprocessor"], ("nb_max_frames",), 3000),
    }
    generation_paths = {
        "eos_token_id": require_exact_int_list(documents["generation"], ("eos_token_id",), [151643, 151645]),
        "pad_token_id": require_exact(documents["generation"], ("pad_token_id",), 151643),
    }
    tokenizer_paths = {
        "tokenizer_class": require_exact(documents["tokenizer"], ("tokenizer_class",), "Qwen2Tokenizer"),
        "model_max_length": require_exact(documents["tokenizer"], ("model_max_length",), 8192),
        "eos_token": require_exact(documents["tokenizer"], ("eos_token",), "<|im_end|>"),
        "pad_token": require_exact(documents["tokenizer"], ("pad_token",), "<|endoftext|>"),
    }
    return {"config": documents["config"], "preprocessor": documents["preprocessor"], "generation": documents["generation"], "tokenizer": documents["tokenizer"], "required_config_paths": paths, "preprocessor_paths": preprocessor_paths, "generation_paths": generation_paths, "tokenizer_paths": tokenizer_paths}


def dtype_bytes(dtype: str) -> int:
    sizes = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
    if dtype not in sizes:
        raise RuntimeError(f"unsupported safetensors dtype: {dtype}")
    return sizes[dtype]


def inspect_safetensors(path: Path, snapshot: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from safetensors import safe_open

    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise RuntimeError(f"truncated safetensors header: {path}")
        header_bytes = int.from_bytes(prefix, "little")
        if header_bytes <= 0 or header_bytes > MAX_HEADER_BYTES or header_bytes > path.stat().st_size - 8:
            raise RuntimeError(f"invalid safetensors header length: {path}")
        header_raw = stream.read(header_bytes)
    try:
        header = strict_json_loads(header_raw.decode("utf-8"), str(path))
    except (UnicodeDecodeError, RuntimeError) as error:
        raise RuntimeError(f"invalid safetensors header JSON: {path}: {error}") from error
    if not isinstance(header, dict):
        raise RuntimeError(f"safetensors header is not an object: {path}")
    metadata = header.get("__metadata__")
    if metadata is not None and (not isinstance(metadata, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in metadata.items())):
        raise RuntimeError(f"safetensors __metadata__ must be a string map: {path}")
    payload_bytes = path.stat().st_size - 8 - header_bytes
    intervals = []
    metadata_names = {name for name in header if name == "__metadata__"}
    tensors = []
    for name, record in header.items():
        if name in metadata_names:
            continue
        if not isinstance(record, dict) or set(record) != {"dtype", "shape", "data_offsets"}:
            raise RuntimeError(f"invalid tensor header record {name} in {path}")
        dtype = str(record["dtype"])
        size = dtype_bytes(dtype)
        shape = record["shape"]
        offsets = record["data_offsets"]
        if not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise RuntimeError(f"invalid shape/offset header record {name} in {path}")
        elements = 1
        for dimension in shape:
            if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0:
                raise RuntimeError(f"invalid tensor shape {name} in {path}")
            elements *= dimension
        start, end = offsets
        if not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or isinstance(end, bool) or start < 0 or end < start or end > payload_bytes or end - start != elements * size:
            raise RuntimeError(f"tensor shape/dtype/data-region mismatch {name} in {path}")
        intervals.append((start, end, name))
        tensors.append({"name": name, "shape": [int(dimension) for dimension in shape], "dtype": dtype, "elements": elements, "data_bytes": end - start, "shard": path.relative_to(snapshot).as_posix()})
    cursor = 0
    for start, end, name in sorted(intervals):
        if start != cursor:
            raise RuntimeError(f"safetensors data-region overlap/gap before {name} in {path}")
        cursor = end
    if cursor != payload_bytes:
        raise RuntimeError(f"safetensors data-region tail gap in {path}")
    with safe_open(str(path), framework="pt") as handle:
        if set(handle.keys()) != {record["name"] for record in tensors}:
            raise RuntimeError(f"safe_open/header tensor-name mismatch: {path}")
        for record in tensors:
            view = handle.get_slice(record["name"])
            if [int(axis) for axis in view.get_shape()] != record["shape"] or str(view.get_dtype()) != record["dtype"]:
                raise RuntimeError(f"safe_open/header metadata mismatch: {path}:{record['name']}")
    shard = {"path": path.relative_to(snapshot).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path), "header_bytes": header_bytes, "data_bytes": payload_bytes, "tensor_count": len(tensors)}
    return shard, tensors


def inventory_weights(snapshot: Path, index: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index_path = snapshot / "model.safetensors.index.json"
    if index is None and not index_path.is_file():
        raise RuntimeError("missing model.safetensors.index.json")
    if index is None:
        index = load_json(index_path)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("safetensors index has no weight_map")
    if any(not isinstance(name, str) or not name for name in weight_map) or any(not isinstance(value, str) for value in weight_map.values()):
        raise RuntimeError("safetensors index weight_map keys and values must be strings")
    if any("\x00" in value or "\\" in value or Path(value).is_absolute() or ".." in Path(value).parts for value in weight_map.values()):
        raise RuntimeError("safetensors index weight_map contains unsafe path")
    actual = {path.name for path in snapshot.glob("*.safetensors")}
    if actual != SHARD_NAMES:
        raise RuntimeError(f"five-shard set mismatch: {sorted(actual)}")
    indexed = {str(value) for value in weight_map.values()}
    if indexed != SHARD_NAMES:
        raise RuntimeError(f"index shard set mismatch: {sorted(indexed)}")
    shards: list[dict[str, Any]] = []
    tensors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(snapshot.glob("*.safetensors")):
        shard, records = inspect_safetensors(path, snapshot)
        for record in records:
            if record["name"] in seen:
                raise RuntimeError(f"duplicate tensor name across shards: {record['name']}")
            if str(weight_map.get(record["name"], "")) != path.name:
                raise RuntimeError(f"weight_map mismatch for tensor: {record['name']}")
            seen.add(record["name"])
        tensors.extend(records)
        shards.append(shard)
    if seen != set(weight_map):
        raise RuntimeError("index tensor coverage has orphan or missing entries")
    return shards, tensors


def source_inventory(source: Path, transformers: Path) -> dict[str, Any]:
    if git(source, "rev-parse", "HEAD") != SOURCE_REVISION:
        raise RuntimeError("official Qwen2-Audio source is not the fixed detached revision")
    source_remote = canonicalize_github_remote(git(source, "remote", "get-url", "origin"), SOURCE_REPOSITORY_PATH)
    tracked = [path for path in (Path(name) for name in git(source, "ls-files", "-z").split("\0") if name) if (source / path).is_file() and not (source / path).is_symlink()]
    source_roles = {
        "readme": (Path("README.md"), ("Qwen2-Audio",)),
        "demo": (Path("demo/web_demo_audio.py"), ("Qwen2Audio",)),
    }
    for role, (relative, markers) in source_roles.items():
        if relative not in tracked:
            raise RuntimeError(f"official Qwen2-Audio project evidence file missing: {relative}")
        text = (source / relative).read_text(encoding="utf-8", errors="replace")
        if not all(marker.lower() in text.lower() for marker in markers):
            raise RuntimeError(f"official Qwen2-Audio project evidence marker missing: {relative}")
    eval_files = [path for path in tracked if path.as_posix().startswith("eval_audio/") and path.suffix == ".py"]
    if not eval_files:
        raise RuntimeError("official Qwen2-Audio evaluation project files are missing")
    source_roles["eval_audio"] = (eval_files[0], ())
    source_license = find_license(source, source_unknown_is_blocker=True)
    transformers_roles = {
        "configuration": "src/transformers/models/qwen2_audio/configuration_qwen2_audio.py",
        "modeling": "src/transformers/models/qwen2_audio/modeling_qwen2_audio.py",
        "processing": "src/transformers/models/qwen2_audio/processing_qwen2_audio.py",
        "audio_frontend": "src/transformers/models/whisper/feature_extraction_whisper.py",
        "decoder_configuration": "src/transformers/models/qwen2/configuration_qwen2.py",
        "decoder_modeling": "src/transformers/models/qwen2/modeling_qwen2.py",
    }
    transformer_markers = {
        "configuration": "Qwen2AudioConfig",
        "modeling": "Qwen2AudioForConditionalGeneration",
        "processing": "Qwen2AudioProcessor",
        "audio_frontend": "WhisperFeatureExtractor",
        "decoder_configuration": "Qwen2Config",
        "decoder_modeling": "Qwen2Model",
    }
    transformer_files = []
    for role, relative in transformers_roles.items():
        path = transformers / relative
        if not path.is_file() or transformer_markers[role].lower() not in path.read_text(encoding="utf-8", errors="replace").lower():
            raise RuntimeError(f"required Transformers Qwen2-Audio role missing or unrecognized: {relative}")
        transformer_files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    if git(transformers, "rev-parse", "HEAD") != TRANSFORMERS_REVISION or git(transformers, "describe", "--exact-match", "--tags") != TRANSFORMERS_TAG:
        raise RuntimeError("Transformers source is not the fixed v4.45.0 commit/tag")
    transformers_remote = canonicalize_github_remote(git(transformers, "remote", "get-url", "origin"), TRANSFORMERS_REPOSITORY_PATH)
    transformers_license = find_license(transformers, require_apache=True)
    def files_payload(root: Path, values: list[Path]) -> list[dict[str, Any]]:
        return [{"path": path.as_posix(), "bytes": (root / path).stat().st_size, "sha256": sha256(root / path)} for path in sorted(values)]
    source_role_records = {role: {"path": relative.as_posix(), "markers": list(markers), "bytes": (source / relative).stat().st_size, "sha256": sha256(source / relative)} for role, (relative, markers) in source_roles.items()}
    return {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION, "origin": source_remote, "role_files": source_role_records, "license": source_license, "files": files_payload(source, tracked), "transformers": {"repository": TRANSFORMERS_REPOSITORY, "tag": TRANSFORMERS_TAG, "revision": TRANSFORMERS_REVISION, "origin": transformers_remote, "role_files": transformer_files, "license": transformers_license}}


def write_exclusive(path: Path, payload: str) -> None:
    fd = path.open("x", encoding="utf-8")
    try:
        fd.write(payload)
    finally:
        fd.close()


def snapshot_inventory(snapshot: Path, server_tree: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = load_json(server_tree)
    if not isinstance(payload, dict):
        raise RuntimeError("server tree must be an identity envelope")
    if payload.get("repository") != UPSTREAM_REPOSITORY or payload.get("revision") != UPSTREAM_REVISION or payload.get("resolved_revision") != UPSTREAM_REVISION:
        raise RuntimeError("HF server tree identity mismatch")
    expected = payload.get("files")
    if not isinstance(expected, list) or not all(isinstance(item, dict) and isinstance(item.get("path"), str) and isinstance(item.get("size"), int) and not isinstance(item["size"], bool) and item["size"] >= 0 and isinstance(item.get("oid"), str) for item in expected):
        raise RuntimeError("server tree must contain path/size records for every regular file")
    # HF snapshots expose blob-backed files as symlinks.  Include readable
    # symlinks in the local inventory; ``is_file`` rejects directories and
    # dangling links while hashing follows the authenticated blob target.
    actual_paths = []
    for path in snapshot.rglob("*"):
        if path.is_symlink():
            if not path.exists() or not path.is_file():
                raise RuntimeError(f"HF snapshot contains dangling/non-file symlink: {path}")
            actual_paths.append(path.relative_to(snapshot).as_posix())
        elif path.is_file():
            actual_paths.append(path.relative_to(snapshot).as_posix())
        elif not path.is_dir():
            raise RuntimeError(f"HF snapshot contains non-regular member: {path}")
    expected_paths = [str(item["path"]) for item in expected]
    if any("\x00" in path or "\\" in path or Path(path).is_absolute() or ".." in Path(path).parts for path in expected_paths):
        raise RuntimeError("HF server tree contains unsafe path")
    if len(expected_paths) != len(set(expected_paths)):
        raise RuntimeError("HF server tree contains duplicate file paths")
    if set(actual_paths) != set(expected_paths):
        raise RuntimeError("HF server tree and local snapshot file sets differ")
    records = []
    for item in sorted(expected, key=lambda value: value["path"]):
        path = snapshot / item["path"]
        if path.stat().st_size != item["size"]:
            raise RuntimeError(f"HF server/local size mismatch: {item['path']}")
        local_digest = sha256(path)
        server_digest = item.get("lfs_sha256")
        oid = item["oid"]
        if server_digest is not None:
            if not isinstance(server_digest, str) or len(server_digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in server_digest):
                raise RuntimeError(f"HF server LFS SHA-256 is malformed: {item['path']}")
            if len(oid) != 64 or any(character not in "0123456789abcdefABCDEF" for character in oid) or oid.lower() != server_digest.lower():
                raise RuntimeError(f"HF server LFS OID is malformed: {item['path']}")
        elif len(oid) != 40 or any(character not in "0123456789abcdefABCDEF" for character in oid):
            raise RuntimeError(f"HF server Git blob OID is malformed: {item['path']}")
        elif git_blob_sha1(path) != oid:
            raise RuntimeError(f"HF server/local Git blob SHA-1 mismatch: {item['path']}")
        if server_digest is not None and server_digest != local_digest:
            raise RuntimeError(f"HF server/local SHA-256 mismatch: {item['path']}")
        records.append({"path": item["path"], "server_size": item["size"], "server_oid": item.get("oid"), "server_lfs_sha256": server_digest, "local_bytes": path.stat().st_size, "local_sha256": local_digest})
    return {"repository": payload["repository"], "revision": payload["revision"], "resolved_revision": payload["resolved_revision"]}, records


def inspect(snapshot: Path, source: Path, transformers: Path, server_tree: Path, output: Path) -> int:
    output.mkdir(parents=True, exist_ok=False)
    server_identity, files = snapshot_inventory(snapshot.resolve(), server_tree.resolve())
    hf_license = find_license(snapshot.resolve(), require_apache=True)
    parsed = parse_model_json(snapshot.resolve())
    shards, tensors = inventory_weights(snapshot.resolve())
    sources = source_inventory(source.resolve(), transformers.resolve())
    write_exclusive(output / "snapshot-inventory.json", json.dumps({"server_tree": server_identity, "files": files}, sort_keys=True, indent=2) + "\n")
    write_exclusive(output / "tensor-inventory.json", json.dumps({"shard_count": len(shards), "shards": shards, "tensor_count": len(tensors), "tensors": tensors}, sort_keys=True, indent=2) + "\n")
    write_exclusive(output / "parsed-json.json", json.dumps(parsed, sort_keys=True, indent=2) + "\n")
    write_exclusive(output / "source-inventory.json", json.dumps(sources, sort_keys=True, indent=2) + "\n")
    packets = {name: {"bytes": (output / name).stat().st_size, "sha256": sha256(output / name)} for name in ("snapshot-inventory.json", "tensor-inventory.json", "parsed-json.json", "source-inventory.json")}
    blockers = ["dataset/training provenance is unauthenticated; no runtime or publication claim"]
    if sources["license"]["status"] == "SOURCE_LICENSE_UNKNOWN_BLOCKER":
        blockers.insert(0, "SOURCE_LICENSE_UNKNOWN_BLOCKER: official Qwen2-Audio project has no recognized license file")
    manifest = {"format": FORMAT, "status": "BLOCKED", "evidence_stage": "INSPECTION_ONLY", "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "cpu_status": "UNSUPPORTED", "metal_status": "BLOCKED_BY_CPU", "parity_status": "NOT_RUN", "publication": "NO_UPLOAD", "task": "audio/text-to-text; not TTS", "upstream": {"repository": UPSTREAM_REPOSITORY, "revision": UPSTREAM_REVISION, "license": hf_license, "server_tree": server_identity, "files": files}, "shards": shards, "tensor_count": len(tensors), "parsed_json": parsed, "official_source": sources, "license_evidence": {"model": hf_license, "official_source": sources["license"], "transformers": sources["transformers"]["license"]}, "dataset_provenance": {"status": "BLOCKED_UNAUTHENTICATED", "reason": "dataset/training provenance was not authenticated by this inspection wave"}, "blockers": blockers, "packets": packets}
    write_exclusive(output / "manifest.json", json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return 2


def write_blocked(output: Path, error: Exception) -> None:
    output.mkdir(parents=True, exist_ok=False)
    payload = {
        "format": FORMAT,
        "status": "BLOCKED",
        "evidence_stage": "INSPECTION_ONLY",
        "error_type": type(error).__name__,
        "reason": str(error),
        "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
        "cpu_status": "UNSUPPORTED",
        "metal_status": "BLOCKED_BY_CPU",
        "parity_status": "NOT_RUN",
        "publication": "NO_UPLOAD",
        "upstream": {"repository": UPSTREAM_REPOSITORY, "revision": UPSTREAM_REVISION},
        "official_source": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION, "license": {"license": "UNKNOWN", "status": "SOURCE_LICENSE_UNKNOWN_BLOCKER"}},
        "transformers": {"repository": TRANSFORMERS_REPOSITORY, "tag": TRANSFORMERS_TAG, "revision": TRANSFORMERS_REVISION},
        "blockers": [str(error)],
    }
    write_exclusive(output / "manifest.json", json.dumps(payload, sort_keys=True, indent=2) + "\n")
    write_exclusive(output / "blocker.txt", f"{type(error).__name__}: {error}\n")


def gate_self_test() -> None:
    head = "a" * 40
    payload = {
        "schema": APPROVAL_SCHEMA, "status": "BLOCKED", "decision": "BLOCKED_INSPECTION_ONLY",
        "expected_head": head, "upstream_repository": UPSTREAM_REPOSITORY, "upstream_revision": UPSTREAM_REVISION,
        "source_repository": SOURCE_REPOSITORY, "source_revision": SOURCE_REVISION,
        "transformers_repository": TRANSFORMERS_REPOSITORY, "transformers_tag": TRANSFORMERS_TAG, "transformers_revision": TRANSFORMERS_REVISION,
        "model_license": "APACHE-2.0", "source_license_status": "SOURCE_LICENSE_UNKNOWN_BLOCKER",
        "dependency_license_status": "REQUIRES_PRIMARY_REVIEW", "dataset_status": "BLOCKED_UNAUTHENTICATED",
        "native_runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "audio_language_parity_status": "NOT_RUN",
        "no_upload": True, "scope": APPROVAL_SCOPE,
    }
    with tempfile.TemporaryDirectory(prefix="vokra-qwen2-audio-gate-") as directory:
        root = Path(directory)
        approval = root / "approval.json"
        approval.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        path = str(approval.resolve())
        digest = sha256(approval)
        validate_approval(path, head, digest, Path.cwd().parent)
        for invalid in (1, 0, "true"):
            approval.write_text(json.dumps({**payload, "no_upload": invalid}, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
            try:
                validate_approval(path, head, sha256(approval), Path.cwd().parent)
            except RuntimeError:
                pass
            else:
                raise AssertionError("non-boolean no_upload accepted")
        for bad_bytes in (b"{\xff", b"not-json", b'{"schema":1,"schema":2}'):
            approval.write_bytes(bad_bytes)
            try:
                validate_approval(path, head, sha256(approval), Path.cwd().parent)
            except RuntimeError:
                pass
            else:
                raise AssertionError("malformed/duplicate approval accepted")
        approval.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        for bad in ("/tmp/../approval.json", "/tmp/./approval.json", "relative.json"):
            try:
                validate_approval(bad, head, digest, Path.cwd().parent)
            except RuntimeError:
                pass
            else:
                raise AssertionError("unsafe approval path accepted")
        link = root / "approval-link.json"
        link.symlink_to(approval)
        try:
            validate_approval(str(link), head, digest, Path.cwd().parent)
        except RuntimeError:
            pass
        else:
            raise AssertionError("symlink approval accepted")
        with tempfile.TemporaryDirectory(prefix=".qwen2-audio-gate-", dir=Path.cwd()) as checkout_directory:
            checkout_approval = Path(checkout_directory) / "approval.json"
            checkout_approval.write_bytes(approval.read_bytes())
            try:
                validate_approval(str(checkout_approval), head, sha256(checkout_approval), Path.cwd())
            except RuntimeError:
                pass
            else:
                raise AssertionError("checkout-contained approval accepted")
        clean_head = globals()["require_clean_head"]
        globals()["require_clean_head"] = lambda _expected: Path.cwd()  # noqa: E731 - isolated gate test injection
        try:
            assert blocked_preflight(head, path, sha256(approval))["status"] == "BLOCKED"
        finally:
            globals()["require_clean_head"] = clean_head
    print("qwen2_audio_7b_instruct gate self-test: PASS")


def self_test() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    assert hashlib.sha256(b"abc").hexdigest() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert len(UPSTREAM_REVISION) == 40 and len(SOURCE_REVISION) == 40 and len(TRANSFORMERS_REVISION) == 40
    assert TRANSFORMERS_TAG == "v4.45.0" and TRANSFORMERS_REVISION == "2ef31dec1676249d26044a8aa8abe33dbecf0d10"
    assert ("v4." + "44.0") not in source and ("984bc11b0882" + "ff1e5b34ba717ea357e069ceced9") not in source
    assert ("weights_only=" + "False") not in source and "safe_open" in source and "NOT_IMPLEMENTED_FAIL_CLOSED" in source
    for remote in (
        "https://github.com/QwenLM/Qwen2-Audio",
        "https://github.com/QwenLM/Qwen2-Audio.git",
        "https://github.com/QwenLM/Qwen2-Audio.git/",
        "ssh://git@github.com/QwenLM/Qwen2-Audio.git",
        "git@github.com:QwenLM/Qwen2-Audio.git",
    ):
        assert canonicalize_github_remote(remote, SOURCE_REPOSITORY_PATH) == "https://github.com/QwenLM/Qwen2-Audio"
    for remote in (
        "https://github.com/other/Qwen2-Audio.git",
        "https://evil.example/QwenLM/Qwen2-Audio.git",
        "https://github.com/QwenLM/Qwen2-Audio/extra.git",
        "git://github.com/QwenLM/Qwen2-Audio.git",
    ):
        try:
            canonicalize_github_remote(remote, SOURCE_REPOSITORY_PATH)
        except RuntimeError:
            pass
        else:
            raise AssertionError("foreign/unsafe GitHub remote was accepted")
    with tempfile.TemporaryDirectory(prefix="vokra-qwen2-audio-inspect-") as directory:
        root = Path(directory)
        head = "a" * 40
        approval_data = {
            "schema": APPROVAL_SCHEMA, "status": "BLOCKED", "decision": "BLOCKED_INSPECTION_ONLY",
            "expected_head": head, "upstream_repository": UPSTREAM_REPOSITORY, "upstream_revision": UPSTREAM_REVISION,
            "source_repository": SOURCE_REPOSITORY, "source_revision": SOURCE_REVISION,
            "transformers_repository": TRANSFORMERS_REPOSITORY, "transformers_tag": TRANSFORMERS_TAG, "transformers_revision": TRANSFORMERS_REVISION,
            "model_license": "APACHE-2.0", "source_license_status": "SOURCE_LICENSE_UNKNOWN_BLOCKER",
            "dependency_license_status": "REQUIRES_PRIMARY_REVIEW", "dataset_status": "BLOCKED_UNAUTHENTICATED",
            "native_runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "audio_language_parity_status": "NOT_RUN",
            "no_upload": True, "scope": APPROVAL_SCOPE,
        }
        approval = root / "approval.json"
        approval.write_text(json.dumps(approval_data, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        approval_path = str(approval.resolve())
        approval_digest = sha256(approval)
        validate_approval(approval_path, head, approval_digest, Path.cwd().parent)
        try:
            validate_approval(approval_path, head, "0" * 64, Path.cwd().parent)
        except RuntimeError:
            pass
        else:
            raise AssertionError("wrong approval SHA accepted")
        for invalid in (1, 0, "true"):
            approval.write_text(json.dumps(dict(approval_data, no_upload=invalid), sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
            try:
                validate_approval(approval_path, head, sha256(approval), Path.cwd().parent)
            except RuntimeError:
                pass
            else:
                raise AssertionError("non-boolean no_upload accepted")
        for key, value in (("expected_head", "b" * 40), ("scope", "WRONG"), ("source_revision", "0" * 40)):
            approval.write_text(json.dumps(dict(approval_data, **{key: value}), sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
            try:
                validate_approval(approval_path, head, sha256(approval), Path.cwd().parent)
            except RuntimeError:
                pass
            else:
                raise AssertionError("invalid approval identity accepted")
        approval.write_bytes(b"{\xff")
        try:
            validate_approval(approval_path, head, sha256(approval), Path.cwd().parent)
        except RuntimeError:
            pass
        else:
            raise AssertionError("malformed UTF-8 approval accepted")
        approval.write_bytes(b"not-json")
        try:
            validate_approval(approval_path, head, sha256(approval), Path.cwd().parent)
        except RuntimeError:
            pass
        else:
            raise AssertionError("malformed JSON approval accepted")
        approval.write_text(json.dumps(approval_data, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        link = root / "approval-link.json"
        link.symlink_to(approval)
        try:
            validate_approval(str(link), head, approval_digest, Path.cwd().parent)
        except RuntimeError:
            pass
        else:
            raise AssertionError("symlink approval accepted")
        duplicate = root / "duplicate-approval.json"
        duplicate.write_bytes(b'{"schema":1,"schema":2}')
        try:
            validate_approval(str(duplicate), head, sha256(duplicate), Path.cwd().parent)
        except RuntimeError:
            pass
        else:
            raise AssertionError("duplicate approval key accepted")
        for bad in ("/tmp/../approval.json", "/tmp/./approval.json", "relative.json"):
            try:
                validate_approval(bad, head, approval_digest, Path.cwd().parent)
            except RuntimeError:
                pass
            else:
                raise AssertionError("unsafe approval path accepted")
        with tempfile.TemporaryDirectory(prefix=".qwen2-audio-approval-", dir=Path.cwd()) as checkout_directory:
            checkout_approval = Path(checkout_directory) / "approval.json"
            checkout_approval.write_bytes(approval.read_bytes())
            try:
                validate_approval(str(checkout_approval), head, sha256(checkout_approval), Path.cwd())
            except RuntimeError:
                pass
            else:
                raise AssertionError("checkout-contained approval accepted")
        clean_head = globals()["require_clean_head"]
        globals()["require_clean_head"] = lambda _expected: Path.cwd()  # noqa: E731 - isolated self-test gate injection
        try:
            assert blocked_preflight(head, approval_path, sha256(approval))["status"] == "BLOCKED"
        finally:
            globals()["require_clean_head"] = clean_head
        def write_shard(path: Path, name: str, payload: bytes = b"\0\0\0\0") -> None:
            header = json.dumps({name: {"dtype": "F32", "shape": [1], "data_offsets": [0, len(payload)]}}, separators=(",", ":")).encode()
            path.write_bytes(len(header).to_bytes(8, "little") + header + payload)
        weight_map = {}
        for number in range(1, SHARD_COUNT + 1):
            name = f"audio.tensor.{number}"
            shard = root / f"model-{number:05d}-of-{SHARD_COUNT:05d}.safetensors"
            write_shard(shard, name)
            weight_map[name] = shard.name
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
        shards, tensors = inventory_weights(root)
        assert len(shards) == SHARD_COUNT and len(tensors) == SHARD_COUNT
        duplicate_index = root / "duplicate-index.json"
        duplicate_index.write_text('{"weight_map": {}, "weight_map": {}}', encoding="utf-8")
        try:
            load_json(duplicate_index)
        except RuntimeError:
            pass
        else:
            raise AssertionError("duplicate index JSON key was accepted")
        for bad_index in (
            {**weight_map, "audio.orphan": next(iter(weight_map.values()))},
            {name: ("../" + value if index == 0 else value) for index, (name, value) in enumerate(weight_map.items())},
            {name: (True if index == 0 else value) for index, (name, value) in enumerate(weight_map.items())},
        ):
            try:
                inventory_weights(root, {"weight_map": bad_index})
            except RuntimeError:
                pass
            else:
                raise AssertionError("unsafe/orphan/non-string index entry was accepted")
        bad_region = root / "bad.safetensors"
        write_shard(bad_region, "bad", b"\0\0\0\0")
        bad_region.write_bytes(bad_region.read_bytes()[:-1])
        try:
            inspect_safetensors(bad_region, root)
        except RuntimeError:
            pass
        else:
            raise AssertionError("truncated data region was accepted")
        duplicate_header = root / "duplicate-header.safetensors"
        duplicate_header_raw = b'{"dup":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"dup":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
        duplicate_header.write_bytes(len(duplicate_header_raw).to_bytes(8, "little") + duplicate_header_raw + b"\0\0\0\0")
        try:
            inspect_safetensors(duplicate_header, root)
        except RuntimeError:
            pass
        else:
            raise AssertionError("duplicate tensor JSON key was accepted")
        huge_header = root / "huge-header.safetensors"
        huge_header.write_bytes((MAX_HEADER_BYTES + 1).to_bytes(8, "little"))
        try:
            inspect_safetensors(huge_header, root)
        except RuntimeError:
            pass
        else:
            raise AssertionError("oversized safetensors header was accepted")
        bad_header = root / "bad-header.safetensors"
        bad_header.write_bytes(
            len(json.dumps({"bad": {"dtype": "F32", "shape": [True], "data_offsets": [0, 4]}}).encode()).to_bytes(8, "little")
            + json.dumps({"bad": {"dtype": "F32", "shape": [True], "data_offsets": [0, 4]}}).encode()
            + b"\0\0\0\0"
        )
        try:
            inspect_safetensors(bad_header, root)
        except RuntimeError:
            pass
        else:
            raise AssertionError("boolean tensor shape was accepted")
        bad_offset = root / "bad-offset.safetensors"
        offset_header = json.dumps({"bad": {"dtype": "F32", "shape": [1], "data_offsets": [False, 4]}}).encode()
        bad_offset.write_bytes(len(offset_header).to_bytes(8, "little") + offset_header + b"\0\0\0\0")
        try:
            inspect_safetensors(bad_offset, root)
        except RuntimeError:
            pass
        else:
            raise AssertionError("boolean tensor offset was accepted")
        for name, value in {
            "config.json": {"architectures": ["Qwen2AudioForConditionalGeneration"], "audio_config": {"encoder_layers": 32, "encoder_attention_heads": 20, "d_model": 1280, "encoder_ffn_dim": 5120, "num_mel_bins": 128, "max_source_positions": 1500}, "audio_token_index": 151646, "text_config": {"vocab_size": 156032, "max_position_embeddings": 8192, "intermediate_size": 11008, "eos_token_id": 151645}},
            "preprocessor_config.json": {"sampling_rate": 16000, "n_fft": 400, "hop_length": 160, "chunk_length": 30, "feature_size": 128, "nb_max_frames": 3000},
            "generation_config.json": {"eos_token_id": [151643, 151645], "pad_token_id": 151643},
            "tokenizer_config.json": {"tokenizer_class": "Qwen2Tokenizer", "model_max_length": 8192, "eos_token": "<|im_end|>", "pad_token": "<|endoftext|>"},
        }.items():
            (root / name).write_text(json.dumps(value), encoding="utf-8")
        parsed = parse_model_json(root)
        assert parsed["config"]["audio_token_index"] == 151646
        assert parsed["config"]["text_config"]["eos_token_id"] == 151645
        assert type(parsed["generation"]["eos_token_id"]) is list and parsed["generation"]["eos_token_id"] == [151643, 151645]
        generation_path = root / "generation_config.json"
        for invalid_generation in (
            {"eos_token_id": [151645, 151643], "pad_token_id": 151643},
            {"eos_token_id": [151643], "pad_token_id": 151643},
            {"pad_token_id": 151643},
            {"eos_token_id": 151643, "pad_token_id": 151643},
            {"eos_token_id": [True, 151645], "pad_token_id": 151643},
        ):
            generation_path.write_text(json.dumps(invalid_generation), encoding="utf-8")
            try:
                parse_model_json(root)
            except RuntimeError:
                pass
            else:
                raise AssertionError("non-canonical generation eos_token_id was accepted")
        generation_path.write_text(json.dumps({"eos_token_id": [151643, 151645], "pad_token_id": 151643}), encoding="utf-8")
        (root / "README.md").write_text("license: unknown\n", encoding="utf-8")
        try:
            find_license(root)
        except RuntimeError:
            pass
        else:
            raise AssertionError("unknown license was accepted")
        assert find_license(root, source_unknown_is_blocker=True)["status"] == "SOURCE_LICENSE_UNKNOWN_BLOCKER"
        (root / "README.md").write_text("license: apache-2.0\n", encoding="utf-8")
        assert find_license(root, require_apache=True)["license"] == "Apache-2.0"
        server_tree = root.parent / "server-tree.json"
        expected = []
        for index, path in enumerate(path for path in root.rglob("*") if path.is_file()):
            item = {"path": path.relative_to(root).as_posix(), "size": path.stat().st_size}
            if index == 0:
                item["oid"] = sha256(path)
                item["lfs_sha256"] = item["oid"]
            else:
                item["oid"] = git_blob_sha1(path)
            expected.append(item)
        server_tree.write_text(json.dumps({"repository": UPSTREAM_REPOSITORY, "revision": UPSTREAM_REVISION, "resolved_revision": UPSTREAM_REVISION, "files": expected}), encoding="utf-8")
        identity, records = snapshot_inventory(root, server_tree)
        assert identity["resolved_revision"] == UPSTREAM_REVISION and len(records) == len(expected)
        server_tree.write_text(json.dumps({"repository": "wrong/repo", "revision": UPSTREAM_REVISION, "resolved_revision": UPSTREAM_REVISION, "files": expected}), encoding="utf-8")
        try:
            snapshot_inventory(root, server_tree)
        except RuntimeError:
            pass
        else:
            raise AssertionError("server identity mismatch was accepted")
        invalid_oid = list(expected)
        invalid_oid[0] = dict(invalid_oid[0])
        invalid_oid[0]["oid"] = "not-a-blob-id"
        server_tree.write_text(json.dumps({"repository": UPSTREAM_REPOSITORY, "revision": UPSTREAM_REVISION, "resolved_revision": UPSTREAM_REVISION, "files": invalid_oid}), encoding="utf-8")
        try:
            snapshot_inventory(root, server_tree)
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid server OID was accepted")
        dangling = root / "dangling-link"
        dangling.symlink_to(root / "missing-target")
        server_tree.write_text(json.dumps({"repository": UPSTREAM_REPOSITORY, "revision": UPSTREAM_REVISION, "resolved_revision": UPSTREAM_REVISION, "files": expected}), encoding="utf-8")
        try:
            snapshot_inventory(root, server_tree)
        except RuntimeError:
            pass
        else:
            raise AssertionError("dangling snapshot symlink was accepted")
        dangling.unlink()
        server_tree.write_text(json.dumps({"repository": UPSTREAM_REPOSITORY, "revision": UPSTREAM_REVISION, "resolved_revision": UPSTREAM_REVISION, "files": expected + [{"path": "orphan.bin", "size": 0}]}), encoding="utf-8")
        try:
            snapshot_inventory(root, server_tree)
        except RuntimeError:
            pass
        else:
            raise AssertionError("server/local tree mismatch was accepted")
        fake_source = root / "source"
        fake_transformers = root / "transformers"
        fake_source.mkdir()
        fake_transformers.mkdir()
        apache = "Apache License, Version 2.0\n"
        for relative, marker in (
            ("README.md", "Qwen2-Audio"),
            ("demo/web_demo_audio.py", "Qwen2Audio"),
            ("eval_audio/eval.py", "evaluation"),
            ("LICENSE", apache),
        ):
            path = fake_source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(marker, encoding="utf-8")
        transformer_files = {
            "src/transformers/models/qwen2_audio/configuration_qwen2_audio.py": "Qwen2AudioConfig",
            "src/transformers/models/qwen2_audio/modeling_qwen2_audio.py": "Qwen2AudioForConditionalGeneration",
            "src/transformers/models/qwen2_audio/processing_qwen2_audio.py": "Qwen2AudioProcessor",
            "src/transformers/models/whisper/feature_extraction_whisper.py": "WhisperFeatureExtractor",
            "src/transformers/models/qwen2/configuration_qwen2.py": "Qwen2Config",
            "src/transformers/models/qwen2/modeling_qwen2.py": "Qwen2Model",
        }
        for relative, marker in transformer_files.items():
            path = fake_transformers / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(marker, encoding="utf-8")
        (fake_transformers / "LICENSE").write_text(apache, encoding="utf-8")
        source_tracked = "README.md\0demo/web_demo_audio.py\0eval_audio/eval.py\0LICENSE\0"
        original_git = globals()["git"]
        def fake_git(path: Path, *args: str) -> str:
            if path == fake_source and args == ("rev-parse", "HEAD"):
                return SOURCE_REVISION
            if path == fake_source and args == ("remote", "get-url", "origin"):
                return SOURCE_REPOSITORY
            if path == fake_source and args == ("ls-files", "-z"):
                return source_tracked
            if path == fake_transformers and args == ("rev-parse", "HEAD"):
                return TRANSFORMERS_REVISION
            if path == fake_transformers and args == ("describe", "--exact-match", "--tags"):
                return TRANSFORMERS_TAG
            if path == fake_transformers and args == ("remote", "get-url", "origin"):
                return TRANSFORMERS_REPOSITORY
            raise AssertionError(f"unexpected fixture git call: {path} {args}")
        globals()["git"] = fake_git
        try:
            selected_sources = source_inventory(fake_source, fake_transformers)
            assert len(selected_sources["transformers"]["role_files"]) == len(transformer_files)
        finally:
            globals()["git"] = original_git
        blocked = root / "blocked"
        write_blocked(blocked, RuntimeError("fixture blocker"))
        blocked_manifest = strict_json_loads((blocked / "manifest.json").read_text(), "blocked fixture")
        assert blocked_manifest["status"] == "BLOCKED" and blocked_manifest["evidence_stage"] == "INSPECTION_ONLY"
        try:
            write_blocked(blocked, RuntimeError("must not clobber"))
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing blocked evidence was clobbered")
    print("qwen2_audio_7b_instruct_inspect.py self-test: OK (identity/header/json/source fail-closed contracts)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--transformers", type=Path)
    parser.add_argument("--server-tree", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--gate-self-test", action="store_true")
    parser.add_argument("--expected-head")
    parser.add_argument("--approval-evidence")
    parser.add_argument("--approval-sha256")
    args = parser.parse_args()
    if args.gate_self_test:
        if args.self_test or any(value is not None for value in (args.snapshot, args.source, args.transformers, args.server_tree, args.output, args.expected_head, args.approval_evidence, args.approval_sha256)):
            parser.error("--gate-self-test accepts no other arguments")
        gate_self_test()
        return 0
    if args.self_test:
        if any(value is not None for value in (args.snapshot, args.source, args.transformers, args.server_tree, args.output, args.expected_head, args.approval_evidence, args.approval_sha256)):
            parser.error("--self-test accepts no other arguments")
        self_test()
        return 0
    if any(value is None for value in (args.expected_head, args.approval_evidence, args.approval_sha256)):
        parser.error("--expected-head, --approval-evidence, and --approval-sha256 are required")
    try:
        blocked_preflight(args.expected_head, args.approval_evidence, args.approval_sha256)
    except Exception as error:
        print(f"QWEN2_AUDIO_BLOCKED_APPROVAL_INVALID: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print("QWEN2_AUDIO_BLOCKED_APPROVAL: status=BLOCKED decision=BLOCKED_INSPECTION_ONLY NO_UPLOAD", file=sys.stderr)
    return 2
    if any(value is None for value in (args.snapshot, args.source, args.transformers, args.server_tree, args.output)):
        parser.error("--snapshot, --source, --transformers, --server-tree, and --output are required")
    try:
        return inspect(args.snapshot, args.source, args.transformers, args.server_tree, args.output)
    except Exception as error:
        write_blocked(args.output, error)
        print(f"Qwen2-Audio inspection BLOCKED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
