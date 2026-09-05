#!/usr/bin/env python3
"""Dependency-free validator for transferred MOSS-TTS Local evidence."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

REPOSITORY = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
REVISION = "be7766a6735b98bd793f7c79fb720b4d0f5d13b8"
SOURCE_DIGESTS = {
    "configuration": "826f81f163b1b557ad13f83c4f35008f4fee5a6cb6311b4316ff3dbb25149411",
    "configuration_source": "ab6debcb92032cb9dc91ae80aed77dbadd2e59848208baef2b062bd6def3f3be",
    "modeling_source": "b0a66211943ae580b087f3e71495fea2f455701a4f6c29b6d3562218f7668c5f",
    "processing_source": "3fc5616b1ec3408162b7d859a7696725a40525313b20f9b31a06ee55c93bd7ad",
    "gpt2_source": "f2e877104669f1e6c7cd34680f0da1a8a159e032123ee56b660b63929b6c8989",
    "qwen3_source": "100163bd7ecf31a59bafacc0b032ace9339edc992a3eb4cc80662502e04e46f0",
    "processor_config": "db574bfebad009e05193196a63a4eeecd353eeca177ccfff28b9379d595d88b7",
}
SOURCE_BYTES = {"configuration": 10045, "configuration_source": 7160, "modeling_source": 26379, "processing_source": 37496, "gpt2_source": 30896, "qwen3_source": 25473, "processor_config": 210}
SOURCE_PATHS = {"configuration": "config.json", "configuration_source": "configuration_moss_tts.py", "modeling_source": "modeling_moss_tts.py", "processing_source": "processing_moss_tts.py", "gpt2_source": "gpt2_decoder.py", "qwen3_source": "qwen3_decoder.py", "processor_config": "processor_config.json"}
MODEL_BYTES = 9100859544
MODEL_SHA256 = "608f1ff64bc6caa9be836060fc7c78a15c4658c4a07b8d73c78d6f70d1b39c23"


def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def fail(message: str) -> None:
    raise SystemExit(message)


def exact(value: object, keys: set[str]) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        fail("exact schema mismatch")


def digest(value: object, length: int = 64) -> None:
    if not isinstance(value, str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
        fail("digest is not lowercase hexadecimal")


def safe_path(value: object) -> bool:
    return isinstance(value, str) and bool(value) and not value.startswith("/") and ".." not in Path(value).parts and ".cache" not in Path(value).parts


def validate(manifest: Path, prompt: Path, rows: Path, codes: Path, prompt_sha: str, rows_sha: str, codes_sha: str, snapshot_root: Path | None = None) -> None:
    m = json.loads(manifest.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)
    exact(m, {"repository", "revision", "snapshot", "loaded_custom_code", "prompt", "rows_from_audio_start", "assistant_codes", "terminal_row_present_in_official_output", "reference_status", "parity_status"})
    if (m["repository"], m["revision"], m["reference_status"], m["parity_status"]) != (REPOSITORY, REVISION, "AUTHENTICATED_EVIDENCE_COMPLETE", "MEASURED_NOT_GATED"):
        fail("manifest identity/status mismatch")
    s = m["snapshot"]
    exact(s, {"repository", "revision", "files", "model", "custom_code", "config_sha256", "server_tree"})
    if (s["repository"], s["revision"]) != (REPOSITORY, REVISION): fail("snapshot identity mismatch")
    digest(s["config_sha256"])
    files = s["files"]
    if not isinstance(files, list) or not files: fail("snapshot files missing")
    file_map: dict[str, dict] = {}
    for item in files:
        exact(item, {"path", "bytes", "sha256"})
        if not safe_path(item["path"]) or not isinstance(item["bytes"], int) or item["bytes"] <= 0: fail("unsafe snapshot file record")
        digest(item["sha256"])
        if item["path"] in file_map: fail("duplicate snapshot file")
        file_map[item["path"]] = item
    if file_map.get("model.safetensors", {}).get("bytes") != MODEL_BYTES or file_map.get("model.safetensors", {}).get("sha256") != MODEL_SHA256: fail("model payload identity mismatch")
    for role, path in SOURCE_PATHS.items():
        if file_map.get(path, {}).get("bytes") != SOURCE_BYTES[role] or file_map.get(path, {}).get("sha256") != SOURCE_DIGESTS[role]: fail("source payload identity mismatch")
    model = s["model"]
    exact(model, {"path", "bytes", "tensor_count", "manifest_sha256", "tensors", "metadata"})
    if model["path"] != "model.safetensors" or model["bytes"] != MODEL_BYTES or not isinstance(model["bytes"], int) or model["tensor_count"] != 438 or not isinstance(model["tensors"], list) or len(model["tensors"]) != 438 or not isinstance(model["metadata"], dict): fail("model contract mismatch")
    digest(model["manifest_sha256"])
    for tensor in model["tensors"]:
        exact(tensor, {"name", "dtype", "shape", "numel", "data_offsets"})
        if not isinstance(tensor["name"], str) or not tensor["name"] or not isinstance(tensor["shape"], list) or not isinstance(tensor["numel"], int) or tensor["numel"] < 0 or not isinstance(tensor["data_offsets"], list) or len(tensor["data_offsets"]) != 2: fail("tensor descriptor mismatch")
    custom = s["custom_code"]
    exact(custom, set(SOURCE_DIGESTS))
    for role, expected in SOURCE_DIGESTS.items():
        exact(custom[role], {"path", "sha256"})
        if not safe_path(custom[role]["path"]) or custom[role]["path"] != SOURCE_PATHS[role] or custom[role]["sha256"] != expected: fail("custom source identity mismatch")
    tree = s["server_tree"]
    exact(tree, {"repository", "revision", "resolved_revision", "files"})
    if (tree["repository"], tree["revision"], tree["resolved_revision"]) != (REPOSITORY, REVISION, REVISION): fail("server tree identity mismatch")
    tree_map: dict[str, dict] = {}
    if not isinstance(tree["files"], list) or len(tree["files"]) != len(file_map): fail("server tree cardinality mismatch")
    for item in tree["files"]:
        exact(item, {"path", "type", "size", "git_blob_sha1", "lfs_sha256", "lfs_size", "local_sha256"})
        if not safe_path(item["path"]) or item["type"] != "file" or not isinstance(item["size"], int) or item["size"] <= 0 or item["path"] in tree_map: fail("unsafe server tree record")
        digest(item["git_blob_sha1"], 40); digest(item["local_sha256"])
        if item["lfs_sha256"] is not None: digest(item["lfs_sha256"])
        if item["lfs_size"] is not None and (not isinstance(item["lfs_size"], int) or item["lfs_size"] != item["size"]): fail("LFS size mismatch")
        tree_map[item["path"]] = item
    if set(file_map) != set(tree_map) or any(tree_map[p]["size"] != file_map[p]["bytes"] or tree_map[p]["local_sha256"] != file_map[p]["sha256"] for p in file_map): fail("snapshot/server tree mismatch")
    if snapshot_root is not None:
        actual = set()
        for path in snapshot_root.rglob("*"):
            relative = path.relative_to(snapshot_root)
            if ".cache" in relative.parts: continue
            if path.is_symlink() or not path.is_file(): fail("snapshot contains symlink or non-regular file")
            actual.add(relative.as_posix())
            record = file_map.get(relative.as_posix())
            if record is None or path.stat().st_size != record["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]: fail("snapshot payload does not match authenticated inventory")
        if actual != set(file_map): fail("snapshot inventory is missing or has extra files")
    loaded = m["loaded_custom_code"]
    if not isinstance(loaded, list) or len(loaded) != 2: fail("loaded source cardinality mismatch")
    for item in loaded:
        exact(item, {"label", "role", "path", "sha256", "resolved_revision", "authenticated_snapshot_path"})
        if item["role"] not in {"modeling_source", "configuration_source"} or item["sha256"] != SOURCE_DIGESTS[item["role"]] or item["resolved_revision"] != REVISION or item["authenticated_snapshot_path"] != custom[item["role"]]["path"]: fail("loaded source mismatch")
    for item, path, width, expected in ((m["prompt"], prompt, 52, prompt_sha), (m["rows_from_audio_start"], rows, 52, rows_sha), (m["assistant_codes"], codes, 48, codes_sha)):
        wanted = {"path", "sha256", "rows", "columns"} if width == 52 and item is m["prompt"] else ({"path", "sha256", "rows", "columns", "start_length", "generated_frames"} if width == 52 else {"path", "sha256", "rows", "codebooks"})
        exact(item, wanted)
        if item["sha256"] != expected or not isinstance(item["rows"], int) or item["rows"] <= 0 or path.stat().st_size != item["rows"] * width: fail("transferred payload mismatch")
    if m["prompt"]["columns"] != 13 or m["rows_from_audio_start"]["columns"] != 13 or m["assistant_codes"]["codebooks"] != 12 or not isinstance(m["terminal_row_present_in_official_output"], bool): fail("payload contract mismatch")


if __name__ == "__main__":
    if len(sys.argv) not in {8, 9}: raise SystemExit("usage: reference_validator.py manifest prompt rows codes prompt_sha rows_sha codes_sha [snapshot_root]")
    values = [Path(arg) if index < 4 or index == 7 else arg for index, arg in enumerate(sys.argv[1:], 0)]
    validate(*values)
