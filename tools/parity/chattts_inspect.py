#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""VAST-only, fail-closed ChatTTS composite inspection.

This tool authenticates release/source evidence and safetensors headers only.
It never loads pickle assets, executes upstream code, converts weights, or
claims a native/runtime/parity result.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

HF_REPOSITORY = "2Noise/ChatTTS"
HF_REVISION = "1a3c04a8b0651689bd9242fbb55b1f4b5a9aef84"
SOURCE_REPOSITORY = "https://github.com/2noise/ChatTTS.git"
SOURCE_TAG = "v0.2.5"
SOURCE_REVISION = "77b89ee281cd479f5b1a787ada330dc975ca1f2a"
FORMAT = "vokra-chattts-inspection-v1"
MAX_HEADER_BYTES = 64 * 1024 * 1024
SELECTED = {
    "asset/DVAE.safetensors": (60359112, "9e1d2e66e74fbfdccd54de4e868fd6116ca2406d", "1d0b044a8368c0513100a2eca98456b289e6be6a18b7a63be1bcaa315ea874d9"),
    "asset/Decoder.safetensors": (103694920, "d221956fe494594af04b2b2cfb746f2abd3d81e9", "77aa55e0a977949c4733df3c6f876fa85860d3298cba63295a7bc6901729d4e0"),
    "asset/Embed.safetensors": (145598536, "335443c27d449651097717fbcf0328169d84a0b6", "2ff0be7134934155741b643b74e32fb6bf3eec41257984459b2ed60cdb4c48b0"),
    "asset/Vocos.safetensors": (54348240, "4331980d51b21224369bd02a0753900da5b98dc4", "07e5561491cce41f7f90cfdb94b2ff263ff5742c3d89339db99b17ad82cc3f44"),
    "asset/gpt/config.json": (762, "cfa091b355bda278d965767f147078cba6abf68c", None),
    "asset/gpt/model.safetensors": (853423872, "b61897b4aa30feff27e6dffb23c35dd464a718b6", "cd0806fd971f52f6a22c923ec64982b305e817bcc41ca83417fcf9141b984a0f"),
    "asset/tokenizer/special_tokens_map.json": (7847, "42cfc10c507baddea0a6ee0db879a93a6e7621c4", None),
    "asset/tokenizer/tokenizer.json": (448604, "d6e380b8cb88c10c9c68f6180c5ddd2ab3053f65", None),
    "asset/tokenizer/tokenizer_config.json": (11028, "b62fb7fbd3c9b91498b869b32343642d03a25fc0", None),
}
SOURCE_ROLES = (
    "ChatTTS/core.py", "ChatTTS/config/config.py", "ChatTTS/model/dvae.py",
    "ChatTTS/model/embed.py", "ChatTTS/model/gpt.py", "ChatTTS/model/speaker.py",
    "ChatTTS/model/tokenizer.py", "ChatTTS/model/processors.py", "ChatTTS/utils/io.py",
    "ChatTTS/res/sha256_map.json", "LICENSE", "README.md", "setup.py", "requirements.txt",
)
# Git blob identities are part of the source contract.  Revision/tag checks
# alone do not prove that the files consumed by the adapter are the pinned
# files (and a dirty checkout must never be accepted).
SOURCE_ROLE_BLOBS = {
    "ChatTTS/core.py": "5bd65336ffb6caad06c756105974fd341c7575f0",
    "ChatTTS/config/config.py": "c91d74c2182a76e8519fbf47783667a0993ccb2d",
    "ChatTTS/model/dvae.py": "01802b697c455d3714c57bc412196521a35893f3",
    "ChatTTS/model/embed.py": "bd8f7fe35013fca43cf04fafdd405d5ee55ba1d2",
    "ChatTTS/model/gpt.py": "e6108e52df48e628058c33416c944a2a5bf0b3ff",
    "ChatTTS/model/speaker.py": "5435922ab019a5ff50751e9054a9d06f1a51403b",
    "ChatTTS/model/tokenizer.py": "84a14527a9014b47d26dcb9f914e14ec4b7053c0",
    "ChatTTS/model/processors.py": "f774dd27f5af40eef0b9f517c3d037154698eedf",
    "ChatTTS/utils/io.py": "dc90e0e9dc6b248d0602939d330761fb30cfa149",
    "ChatTTS/res/sha256_map.json": "ae91128693a5c9519ae6acf3ca25330e3dbb6aa7",
    "LICENSE": "0ad25db4bd1d86c452db3f9602ccdbe172438f52",
    "README.md": "b21e908694a607d7c153e96e7d5891cffde33b95",
    "setup.py": "dde50e327955b052ba2fab0203b2001b2845564c",
    "requirements.txt": "bd108b79febd41db030c71566c99047665a97852",
}


def source_sha_key(path: str) -> str:
    """Return the exact key used by v0.2.5 ``res/sha256_map.json``."""
    return "sha256_" + path.replace("/", "_").replace(".", "_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    except Exception as error:
        raise RuntimeError(f"strict JSON failure at {path}: {error}") from error


class StrictYamlLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def yaml_mapping(loader: StrictYamlLoader, node: yaml.nodes.MappingNode) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if key in result:
            raise RuntimeError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node)
    return result


StrictYamlLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, yaml_mapping)


def safe_path(name: str, label: str) -> None:
    candidate = Path(name)
    if not name or "\0" in name or "\\" in name or candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeError(f"unsafe {label} path: {name!r}")


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT).strip()


def inspect_safetensors(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    if size < 8:
        raise RuntimeError(f"safetensors too short: {path}")
    with path.open("rb") as stream:
        header_len = int.from_bytes(stream.read(8), "little")
        if header_len > MAX_HEADER_BYTES or 8 + header_len > size:
            raise RuntimeError(f"invalid/bounded safetensors header: {path}")
        raw = stream.read(header_len)
    header = json.loads(raw.decode("utf-8"), object_pairs_hook=strict_pairs)
    if not isinstance(header, dict):
        raise RuntimeError("safetensors header must be object")
    data_end = size - 8 - header_len
    spans: list[tuple[int, int, str]] = []
    tensor_manifest: list[dict[str, Any]] = []
    for name, descriptor in header.items():
        if name == "__metadata__":
            if not isinstance(descriptor, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in descriptor.items()):
                raise RuntimeError("safetensors metadata must be string map")
            continue
        safe_path(name, "tensor")
        if not isinstance(descriptor, dict) or set(descriptor) != {"dtype", "shape", "data_offsets"}:
            raise RuntimeError(f"invalid tensor descriptor: {name}")
        dtype, shape, offsets = descriptor["dtype"], descriptor["shape"], descriptor["data_offsets"]
        if dtype not in {"F32", "F16", "BF16"} or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise RuntimeError(f"ChatTTS selected tensor must use a supported float dtype: {name}")
        if any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape):
            raise RuntimeError(f"invalid tensor shape: {name}")
        if any(isinstance(off, bool) or not isinstance(off, int) or off < 0 for off in offsets):
            raise RuntimeError(f"invalid tensor offsets: {name}")
        start, end = offsets
        if end < start or end > data_end:
            raise RuntimeError(f"tensor range out of bounds: {name}")
        expected = {"F32": 4, "F16": 2, "BF16": 2}[dtype]
        for dim in shape:
            expected *= dim
        if end - start != expected:
            raise RuntimeError(f"tensor byte-size mismatch: {name}")
        spans.append((start, end, name))
        tensor_manifest.append({"name": name, "dtype": dtype, "shape": shape, "data_offsets": [start, end]})
    spans.sort()
    cursor = 0
    for start, end, name in spans:
        if start != cursor:
            raise RuntimeError(f"safetensors gap/overlap before {name}")
        cursor = end
    if cursor != data_end or not spans:
        raise RuntimeError("safetensors data region is incomplete or empty")
    return {"bytes": size, "header_bytes": header_len, "tensor_count": len(spans), "data_bytes": data_end, "tensor_manifest": sorted(tensor_manifest, key=lambda item: item["name"])}


def inspect_model(snapshot: Path, packet: Path) -> dict[str, Any]:
    envelope = load_json(packet)
    if not isinstance(envelope, dict) or envelope.get("repository") != HF_REPOSITORY or envelope.get("revision") != HF_REVISION or envelope.get("resolved_revision") != HF_REVISION:
        raise RuntimeError("HF server-tree identity mismatch")
    rows = envelope.get("files")
    if not isinstance(rows, list) or len(rows) != 23:
        raise RuntimeError("ChatTTS release must have exactly 23 regular files")
    if sum(row.get("size", -1) for row in rows if isinstance(row, dict) and isinstance(row.get("size"), int)) != 2_365_218_661:
        raise RuntimeError("ChatTTS release total bytes mismatch")
    expected: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "type", "size", "git_blob_sha1", "lfs_sha256"}:
            raise RuntimeError("malformed server-tree row")
        name = row["path"]
        safe_path(name, "server")
        if row["type"] != "file" or name in expected or not isinstance(row["size"], int) or isinstance(row["size"], bool) or row["size"] < 0:
            raise RuntimeError(f"invalid/duplicate server row: {name!r}")
        if not isinstance(row["git_blob_sha1"], str) or not re.fullmatch(r"[0-9a-f]{40}", row["git_blob_sha1"].lower()):
            raise RuntimeError(f"missing Git identity: {name}")
        lfs = row["lfs_sha256"]
        if lfs is not None and (not isinstance(lfs, str) or not re.fullmatch(r"[0-9a-f]{64}", lfs.lower())):
            raise RuntimeError(f"invalid LFS identity: {name}")
        expected[name] = row
    actual: set[str] = set()
    for path in snapshot.rglob("*"):
        relative = path.relative_to(snapshot)
        if ".cache" in relative.parts:
            continue
        if path.is_symlink():
            if not path.exists() or not path.is_file():
                raise RuntimeError(f"invalid selected symlink: {relative}")
            try:
                path.resolve().relative_to(snapshot.resolve())
            except ValueError as error:
                raise RuntimeError(f"selected symlink escapes root: {relative}") from error
            actual.add(relative.as_posix())
        elif path.is_file():
            actual.add(relative.as_posix())
        elif not path.is_dir():
            raise RuntimeError(f"selected snapshot non-regular member: {relative}")
    # The VAST worker deliberately materializes nine safe release files plus
    # README for semantic/license inspection; legacy .pt files remain
    # authenticated in the complete server packet but are never downloaded or loaded.
    semantic_selected = {"README.md"}
    if actual != set(SELECTED) | semantic_selected:
        raise RuntimeError(f"selected/local file set mismatch: missing={sorted((set(SELECTED) | semantic_selected)-actual)} extra={sorted(actual-(set(SELECTED) | semantic_selected))}")
    records = []
    for name, row in expected.items():
        records.append({"path": name, "bytes": row["size"], "git_blob_sha1": row["git_blob_sha1"], "lfs_sha256": row["lfs_sha256"], "local_verified": name in SELECTED or name == "README.md"})
    selected = {}
    for name, (expected_bytes, expected_git, expected_lfs) in SELECTED.items():
        if name not in expected:
            raise RuntimeError(f"required selected release file missing: {name}")
        row = expected[name]
        if row["size"] != expected_bytes or row["git_blob_sha1"].lower() != expected_git.lower() or row["lfs_sha256"] != expected_lfs:
            raise RuntimeError(f"selected release identity mismatch: {name}")
        path = snapshot / name
        if path.is_symlink():
            if not path.exists() or not path.is_file():
                raise RuntimeError(f"invalid snapshot symlink: {name}")
            try:
                path.resolve().relative_to(snapshot.resolve())
            except ValueError as error:
                raise RuntimeError(f"snapshot symlink escapes root: {name}") from error
        if path.stat().st_size != expected_bytes:
            raise RuntimeError(f"size mismatch: {name}")
        digest = sha256(path)
        if expected_lfs is not None:
            if digest.lower() != expected_lfs.lower():
                raise RuntimeError(f"LFS SHA mismatch: {name}")
        elif blob_sha1(path).lower() != expected_git.lower():
            raise RuntimeError(f"Git blob mismatch: {name}")
        selected[name] = {"sha256": digest, "bytes": expected_bytes}
        if name.endswith(".safetensors"):
            selected[name]["header"] = inspect_safetensors(snapshot / name)
    # README is required for the semantic/license inspection but is not a
    # learned asset with a fixed local constant. Authenticate it directly
    # against the server packet's Git blob and byte identity.
    readme_row = expected.get("README.md")
    readme = snapshot / "README.md"
    if not isinstance(readme_row, dict) or not readme.is_file() or readme.stat().st_size != readme_row["size"] or blob_sha1(readme) != readme_row["git_blob_sha1"]:
        raise RuntimeError("README semantic evidence does not match authenticated server identity")
    selected["README.md"] = {"sha256": sha256(readme), "bytes": readme.stat().st_size}
    if len(selected) != len(SELECTED) + 1:
        raise RuntimeError("selected materialization is incomplete")
    return {"repository": HF_REPOSITORY, "revision": HF_REVISION, "resolved_revision": HF_REVISION, "files": records, "selected": selected}


def inspect_source(source: Path) -> dict[str, Any]:
    if git(source, "rev-parse", "HEAD") != SOURCE_REVISION or git(source, "describe", "--exact-match", "--tags", "HEAD") != SOURCE_TAG:
        raise RuntimeError("ChatTTS source HEAD/tag mismatch")
    if git(source, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("ChatTTS source checkout is dirty")
    origin = git(source, "remote", "get-url", "origin").rstrip("/")
    if origin.removesuffix(".git") != SOURCE_REPOSITORY.removesuffix(".git"):
        raise RuntimeError("ChatTTS source origin mismatch")
    missing = [role for role in SOURCE_ROLES if not (source / role).is_file()]
    if missing:
        raise RuntimeError(f"source role missing: {missing}")
    license_text = (source / "LICENSE").read_text(encoding="utf-8")
    setup_text = (source / "setup.py").read_text(encoding="utf-8")
    if "gnu affero general public license" not in license_text.casefold() or "agplv3+" not in setup_text.casefold():
        raise RuntimeError("source AGPLv3+ identity is not authenticated")
    source_map = load_json(source / "ChatTTS/res/sha256_map.json")
    if not isinstance(source_map, dict):
        raise RuntimeError("source sha256_map.json must be an object")
    selected_hashes = source_sha_map(source_map)
    source_config = validate_source_config(source / "ChatTTS/config/config.py")
    roles = {}
    for role in SOURCE_ROLES:
        actual = git(source, "rev-parse", f"HEAD:{role}")
        if actual != SOURCE_ROLE_BLOBS[role]:
            raise RuntimeError(f"source role Git blob mismatch: {role}")
        roles[role] = {"sha256": sha256(source / role), "git_blob_sha1": actual}
    return {"repository": SOURCE_REPOSITORY, "origin": origin, "revision": SOURCE_REVISION, "tag": SOURCE_TAG, "clean": True, "license": "AGPLv3+", "roles": roles, "release_sha_map": selected_hashes, "effective_config": source_config}


def source_sha_map(source_map: dict[str, Any]) -> dict[str, str]:
    selected_hashes = {}
    for name in SELECTED:
        key = source_sha_key(name)
        value = source_map.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value.lower()):
            raise RuntimeError(f"source SHA map missing unambiguous selected entry: {name}")
        selected_hashes[name] = value.lower()
    return selected_hashes


def validate_source_config(path: Path) -> dict[str, Any]:
    """Validate source config lexically after parsing its AST, never importing it."""
    text = path.read_text(encoding="utf-8")
    try:
        ast.parse(text, filename=str(path))
    except (SyntaxError, UnicodeError) as error:
        raise RuntimeError(f"source config AST parse failed: {error}") from error
    # The pinned release uses annotated dataclass fields (``idim: int = 384``).
    # Strip only the annotation immediately before an assignment so the
    # contract patterns below remain scoped to the source's actual values.
    text = re.sub(r":[ \t]*[^=,\n]+(?=[ \t]*=)", "", text)
    required = {
        "decoder": (r"\bidim\s*[=:]\s*384", r"\bodim\s*[=:]\s*384", r"\bhidden\s*[=:]\s*512", r"\bn_layer\s*[=:]\s*12", r"\bbn_dim\s*[=:]\s*128"),
        "vq": (r"\bdim\s*[=:]\s*1024", r"levels\s*[=:]\s*\(?\s*5\s*,\s*5\s*,\s*5\s*,\s*5", r"\bG\s*[=:]\s*2", r"\bR\s*[=:]\s*2"),
        "dvae_encoder": (r"encoder", r"(?:in_dim|idim)\s*[=:]\s*512", r"(?:out_dim|odim)\s*[=:]\s*1024", r"hidden\s*[=:]\s*256", r"n_layer\s*[=:]\s*12", r"bn_dim\s*[=:]\s*128"),
        "dvae_decoder": (r"decoder", r"(?:in_dim|idim)\s*[=:]\s*512", r"(?:out_dim|odim)\s*[=:]\s*512", r"hidden\s*[=:]\s*256", r"n_layer\s*[=:]\s*12", r"bn_dim\s*[=:]\s*128"),
        "gpt": (r"hidden_size\s*[=:]\s*768", r"intermediate_size\s*[=:]\s*3072", r"num_attention_heads\s*[=:]\s*12", r"num_hidden_layers\s*[=:]\s*20", r"use_cache\s*[=:]\s*False", r"max_position_embeddings\s*[=:]\s*4096", r"spk.*192", r"spk.*(?:False|0)", r"626", r"21178", r"num_vq\s*[=:]\s*4"),
        "embed": (r"Embed", r"768", r"626", r"21178", r"(?:num_vq|vq).*4"),
        "vocos": (r"sample_rate\s*[=:]\s*24000", r"(?:n_fft|fft)\s*[=:]\s*1024", r"hop(?:_length)?\s*[=:]\s*256", r"(?:n_mels|mel).*100", r"padding.*center", r"(?:input|mel).*100", r"dim\s*[=:]\s*512", r"intermediate.*1536", r"layers?\s*[=:]\s*8", r"iSTFT|ISTFT"),
        "standalone_decoder": (r"Decoder", r"idim\s*[=:]\s*384", r"odim\s*[=:]\s*384", r"hidden\s*[=:]\s*512", r"n_layer\s*[=:]\s*12", r"bn_dim\s*[=:]\s*128"),
    }
    missing = [label for label, patterns in required.items() if any(re.search(pattern, text, re.IGNORECASE) is None for pattern in patterns)]
    if missing:
        raise RuntimeError(f"effective ChatTTS source config markers missing: {missing}")
    return {
        "path": "ChatTTS/config/config.py",
        "sha256": sha256(path),
        "ast_parsed": True,
        "validated_sections": sorted(required),
        "contract": {
            "decoder": {"idim": 384, "odim": 384, "hidden": 512, "n_layer": 12, "bn_dim": 128},
            "vq": {"dim": 1024, "levels": [5, 5, 5, 5], "G": 2, "R": 2},
            "dvae_encoder": {"input": 512, "output": 1024, "hidden": 256, "layers": 12, "bn_dim": 128},
            "dvae_decoder": {"input": 512, "output": 512, "hidden": 256, "layers": 12, "bn_dim": 128},
            "gpt": {"hidden": 768, "intermediate": 3072, "heads": 12, "layers": 20, "use_cache": False, "max_position": 4096, "speaker": 192, "speaker_kl": False, "audio_tokens": 626, "text_tokens": 21178, "num_vq": 4},
            "embed": {"hidden": 768, "audio_tokens": 626, "text_tokens": 21178, "num_vq": 4},
            "vocos": {"sample_rate": 24000, "fft": 1024, "hop": 256, "mels": 100, "padding": "center", "backbone_dim": 512, "intermediate": 1536, "layers": 8, "istft_dim": 512},
            "standalone_decoder": {"idim": 384, "odim": 384, "hidden": 512, "layers": 12, "bn_dim": 128},
        },
    }


def source_mapping_fixture_test() -> bool:
    expected = {source_sha_key(name): "a" * 64 for name in SELECTED}
    if not source_sha_key("asset/gpt/config.json") == "sha256_asset_gpt_config_json" or not all(
        isinstance(expected.get(source_sha_key(name)), str) for name in SELECTED
    ) or "asset/gpt/config.json" in expected:
        return False
    if source_sha_key("asset/tokenizer/tokenizer.json") != "sha256_asset_tokenizer_tokenizer_json":
        return False
    expected.pop("sha256_asset_gpt_config_json")
    try:
        source_sha_map(expected)
    except RuntimeError:
        return True
    return False


def source_config_drift_fixture_test() -> bool:
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.py"
        path.write_text("class Drift: hidden_size = 1\n", encoding="utf-8")
        try:
            validate_source_config(path)
        except RuntimeError:
            return True
        return False


def inspect_semantics(snapshot: Path) -> dict[str, Any]:
    readme = (snapshot / "README.md").read_text(encoding="utf-8")
    if not readme.startswith("---\n") or "\n---\n" not in readme[4:]:
        raise RuntimeError("README strict YAML front matter missing")
    front = readme[4:readme.index("\n---\n", 4)]
    try:
        fields = yaml.load(front, Loader=StrictYamlLoader)
    except Exception as error:
        raise RuntimeError(f"README front matter strict YAML failure: {error}") from error
    if not isinstance(fields, dict) or fields.get("license") != "cc-by-nc-4.0" or fields.get("library_name") != "chat_tts" or fields.get("pipeline_tag") != "text-to-audio":
        raise RuntimeError("ChatTTS card identity mismatch")
    if not re.search(r"(?i)research|academic|non.?commercial", readme):
        raise RuntimeError("ChatTTS research-only card wording missing")
    config = load_json(snapshot / "asset/gpt/config.json")
    exact = {"model_type": "llama", "hidden_size": 768, "intermediate_size": 3072, "num_attention_heads": 12, "num_hidden_layers": 20, "max_position_embeddings": 4096, "num_key_value_heads": 12, "rms_norm_eps": 1e-6, "rope_theta": 10000, "hidden_act": "silu", "tie_word_embeddings": False, "use_cache": False, "vocab_size": 21178}
    for key, expected in exact.items():
        if config.get(key) != expected:
            raise RuntimeError(f"GPT config mismatch at {key}")
    tokenizer = load_json(snapshot / "asset/tokenizer/tokenizer.json")
    if tokenizer.get("version") != "1.0" or tokenizer.get("model", {}).get("type") != "WordPiece" or len(tokenizer.get("model", {}).get("vocab", {})) != 21128 or len(tokenizer.get("added_tokens", [])) != 55:
        raise RuntimeError("ChatTTS tokenizer structural contract mismatch")
    if tokenizer.get("normalizer", {}).get("type") != "BertNormalizer" or tokenizer.get("pre_tokenizer", {}).get("type") != "BertPreTokenizer" or tokenizer.get("decoder", {}).get("type") != "WordPiece" or tokenizer.get("post_processor", {}).get("type") != "TemplateProcessing":
        raise RuntimeError("ChatTTS tokenizer pipeline mismatch")
    expected_special_ids = {
        "[PAD]": 0, "[UNK]": 100, "[CLS]": 101, "[SEP]": 102, "[MASK]": 103,
        "[Ebreak]": 21136, "[spk_emb]": 21143, "[break_0]": 21147,
    }
    added_entries = tokenizer.get("added_tokens", [])
    if not isinstance(added_entries, list) or any(not isinstance(item, dict) for item in added_entries):
        raise RuntimeError("ChatTTS tokenizer added-token records are malformed")
    added_ids = [item.get("id") for item in added_entries]
    added_contents = [item.get("content") for item in added_entries]
    if len(set(added_ids)) != len(added_ids) or len(set(added_contents)) != len(added_contents):
        raise RuntimeError("ChatTTS tokenizer added-token IDs/content are duplicated")
    added_tokens = dict(zip(added_contents, added_ids))
    if any(added_tokens.get(token) != token_id for token, token_id in expected_special_ids.items()):
        raise RuntimeError("ChatTTS tokenizer special-token ID contract mismatch")
    tok_config = load_json(snapshot / "asset/tokenizer/tokenizer_config.json")
    if tok_config.get("tokenizer_class") != "BertTokenizer" or tok_config.get("pad_token") != "[PAD]" or tok_config.get("unk_token") != "[UNK]":
        raise RuntimeError("ChatTTS tokenizer config mismatch")
    specials = {entry.get("content") for entry in tokenizer.get("added_tokens", []) if isinstance(entry, dict)}
    for token in ("[laugh]", "[uv_break]", "[lbreak]", "[spk_emb]", "[break_0]"):
        if token not in specials:
            raise RuntimeError(f"ChatTTS special token missing: {token}")
    return {"card": fields, "gpt_config": config, "tokenizer": {"version": tokenizer["version"], "vocab": len(tokenizer["model"]["vocab"]), "added_tokens": len(tokenizer["added_tokens"]), "specials": sorted(specials), "special_ids": expected_special_ids}}


def blocked_manifest(output: Path, *, status: str, error: str | None = None, evidence: dict[str, Any] | None = None) -> None:
    manifest = {"format": FORMAT, "status": "BLOCKED", "inspection_status": status, "evidence_stage": "INSPECTION_ONLY", "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "native_status": "BLOCKED_NATIVE_BINDING", "cpu_status": "UNSUPPORTED", "metal_status": "BLOCKED_BY_CPU", "parity_status": "NOT_RUN", "publication": "NO_UPLOAD", "license_evidence": {"weights": "CC-BY-NC-4.0", "source": "AGPLv3+", "dependencies": "REVIEW_REQUIRED_BLOCKER"}, "model": {"repository": HF_REPOSITORY, "revision": HF_REVISION}, "source": {"repository": SOURCE_REPOSITORY, "tag": SOURCE_TAG, "revision": SOURCE_REVISION, "license": "AGPLv3+"}}
    if error:
        manifest["error"] = error
    if evidence:
        manifest["evidence"] = evidence
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def validate_inspection_manifest(manifest: dict[str, Any]) -> None:
    """Independent structural gate used by the Apple worker; never grep status."""
    required = {"format", "status", "inspection_status", "evidence_stage", "runtime_status", "native_status", "cpu_status", "metal_status", "parity_status", "publication", "license_evidence", "model", "source", "evidence"}
    if set(manifest) != required or manifest.get("format") != FORMAT or manifest.get("status") != "BLOCKED" or manifest.get("inspection_status") != "AUTHENTICATED_EVIDENCE_COMPLETE" or manifest.get("native_status") != "BLOCKED_NATIVE_BINDING" or manifest.get("parity_status") != "NOT_RUN" or manifest.get("publication") != "NO_UPLOAD":
        raise RuntimeError("inspection manifest is not an authenticated fail-closed packet")
    evidence = manifest["evidence"]
    if not isinstance(evidence, dict) or set(evidence) != {"model", "source", "semantics"}:
        raise RuntimeError("inspection evidence schema drift")
    model, source = evidence["model"], evidence["source"]
    if model.get("repository") != HF_REPOSITORY or model.get("revision") != HF_REVISION or model.get("resolved_revision") != HF_REVISION or len(model.get("files", [])) != 23 or set(model.get("selected", {})) != set(SELECTED) | {"README.md"}:
        raise RuntimeError("inspection model identity/cardinality mismatch")
    rows = model["files"]
    if any(not isinstance(row, dict) or set(row) != {"path", "bytes", "git_blob_sha1", "lfs_sha256", "local_verified"} or not re.fullmatch(r"[0-9a-f]{40}", row["git_blob_sha1"]) or (row["lfs_sha256"] is not None and not re.fullmatch(r"[0-9a-f]{64}", row["lfs_sha256"])) for row in rows) or len({row["path"] for row in rows}) != 23:
        raise RuntimeError("inspection server-tree rows are malformed or duplicated")
    for name, (size, blob, lfs) in SELECTED.items():
        selected = model["selected"].get(name)
        row = next((row for row in rows if row["path"] == name), None)
        if not isinstance(selected, dict) or selected.get("bytes") != size or not re.fullmatch(r"[0-9a-f]{64}", selected.get("sha256", "")) or not row or row["bytes"] != size or row["git_blob_sha1"] != blob or row["lfs_sha256"] != lfs:
            raise RuntimeError(f"inspection fixed release identity mismatch: {name}")
    if source.get("repository") != SOURCE_REPOSITORY or source.get("origin") != SOURCE_REPOSITORY or source.get("revision") != SOURCE_REVISION or source.get("tag") != SOURCE_TAG or source.get("clean") is not True or set(source.get("roles", {})) != set(SOURCE_ROLE_BLOBS):
        raise RuntimeError("inspection source identity/role mismatch")
    if set(source.get("release_sha_map", {})) != set(SELECTED) or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in source["release_sha_map"].values()):
        raise RuntimeError("inspection source release SHA map is incomplete")
    for role, record in source["roles"].items():
        if record.get("git_blob_sha1") != SOURCE_ROLE_BLOBS[role] or not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", "")):
            raise RuntimeError(f"inspection source role digest mismatch: {role}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--server-tree", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if inspect_safetensors_fixture_tests() and strict_json_fixture_test() and source_mapping_fixture_test() and source_config_drift_fixture_test():
            print("chattts_inspect.py self-test: OK")
            return 0
        return 1
    if not all((args.snapshot, args.source, args.server_tree, args.output)):
        parser.error("inspection requires --snapshot --source --server-tree --output")
    try:
        model = inspect_model(args.snapshot, args.server_tree)
        source = inspect_source(args.source)
        for name, record in model["selected"].items():
            # README is semantic/license evidence, not a learned release asset
            # and therefore is intentionally absent from sha256_map.json.
            if name in SELECTED and source["release_sha_map"].get(name) != record["sha256"]:
                raise RuntimeError(f"source SHA map disagrees with release SHA: {name}")
        semantics = inspect_semantics(args.snapshot)
        blocked_manifest(args.output, status="AUTHENTICATED_EVIDENCE_COMPLETE", evidence={"model": model, "source": source, "semantics": semantics})
    except Exception as error:
        blocked_manifest(args.output, status="INSPECTION_ERROR", error=f"{type(error).__name__}: {error}")
        print(f"ChatTTS inspection blocked: {error}", file=sys.stderr)
    return 2


def inspect_safetensors_fixture_tests() -> bool:
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "fixture.safetensors"
        header = json.dumps({"x": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}).encode()
        path.write_bytes(len(header).to_bytes(8, "little") + header + b"\0" * 4)
        if inspect_safetensors(path)["tensor_count"] != 1:
            return False
        bad = Path(directory) / "bad.safetensors"
        bad.write_bytes((MAX_HEADER_BYTES + 1).to_bytes(8, "little"))
        try:
            inspect_safetensors(bad)
        except RuntimeError:
            pass
        else:
            return False
        duplicate = Path(directory) / "duplicate.safetensors"
        duplicate_header = b'{"x":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"x":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}'
        duplicate.write_bytes(len(duplicate_header).to_bytes(8, "little") + duplicate_header + b"\0" * 4)
        try:
            inspect_safetensors(duplicate)
        except RuntimeError:
            return True
        return False


def strict_json_fixture_test() -> bool:
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "duplicate.json"
        path.write_text('{"x": 1, "x": 2}', encoding="utf-8")
        try:
            load_json(path)
        except RuntimeError:
            return True
        return False


if __name__ == "__main__":
    raise SystemExit(main())
