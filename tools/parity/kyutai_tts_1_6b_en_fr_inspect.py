#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""VAST-only inspection of the fixed Kyutai TTS composite release.

This tool authenticates bytes, headers, configuration evidence, source
identities, and the selected voice fixture.  It deliberately does not convert
or execute a Kyutai runtime: every result is BLOCKED/INSPECTION_ONLY.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

HF_REPOSITORY = "kyutai/tts-1.6b-en_fr"
HF_REVISION = "f65439609986c392cb12df63938abcc550c3fb15"
HF_ARTIFACT_BYTES = 4_068_484_990
HF_TOTAL_BYTES = 4_068_492_850
TTS_FILE = "dsm_tts_1e68beda@240.safetensors"
TTS_BYTES = 3_683_719_712
TTS_SHA256 = "726ddadd90a080c89cbc6b217745296ef32d8e25666d30f81a09e8ae5c9e0f0c"
MIMI_FILE = "tokenizer-e351c8d8-checkpoint125.safetensors"
MIMI_BYTES = 384_644_900
MIMI_SHA256 = "09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50"
SPM_FILE = "tokenizer_spm_8k_en_fr_audio.model"
SPM_BYTES = 120_378
SPM_SHA256 = "cd87dd5d17169151782ac700280ec057e5d658a9afbe238a048ea5ff318cce69"
HF_FILES = {".gitattributes", "README.md", "config.json", TTS_FILE, MIMI_FILE, SPM_FILE}
MOSHI_URL = "https://github.com/kyutai-labs/moshi.git"
MOSHI_REVISION = "e6a55d2722a65870ef52a6c9f6ecfc0e90f38362"
DSM_URL = "https://github.com/kyutai-labs/delayed-streams-modeling.git"
DSM_REVISION = "4c4f65e147df056adf3346290d64c7b9649b18c9"
VOICE_REPOSITORY = "kyutai/tts-voices"
VOICE_REVISION = "323332d33f997de8394f24a193e1a76df720e01a"
VOICE_FILE = "voice-donations/robert.wav.1e68beda@240.safetensors"
VOICE_BYTES = 256_136
VOICE_SHA256 = "bc79b0162c94862aadd6c5d351b5b4984274af0616e3a56b0df9973ff7c793c7"
MOSHI_ROLES = (
    "moshi/moshi/models/tts.py", "moshi/moshi/models/lm.py",
    "moshi/moshi/models/lm_utils.py", "moshi/moshi/models/loaders.py",
    "moshi/moshi/conditioners/__init__.py", "moshi/moshi/conditioners/base.py",
    "moshi/moshi/conditioners/tensors.py", "moshi/moshi/conditioners/text.py",
    "moshi/moshi/run_tts.py",
    "rust/moshi-core/src/tts.rs", "rust/moshi-core/src/tts_streaming.rs",
    "rust/moshi-core/src/lm.rs", "rust/moshi-core/src/lm_generate_multistream.rs",
    "rust/moshi-core/src/mimi.rs", "rust/moshi-core/src/conditioner.rs",
    "rust/moshi-server/src/main.rs",
)
DSM_ROLES = (
    "configs/config-tts.toml", "scripts/tts_pytorch.py",
    "scripts/tts_pytorch_streaming.py", "scripts/tts_mlx.py",
    "scripts/tts_mlx_streaming.py", "scripts/tts_rust_server.py",
    "README.md", "LICENSE-APACHE", "LICENSE-MIT",
)
MAX_HEADER_BYTES = 64 * 1024 * 1024
FORMAT = "vokra-kyutai-tts-1.6b-en-fr-inspection-v1"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
DTYPE_BYTES = {"F32": 4, "BF16": 2, "F16": 2, "I64": 8, "I32": 4, "U8": 1, "I8": 1}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    size = path.stat().st_size
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)


def safe_path(name: str) -> None:
    parts = PurePosixPath(name).parts
    if not name or "\x00" in name or "\\" in name or PurePosixPath(name).is_absolute() or ".." in parts:
        raise ValueError(f"unsafe path: {name!r}")


def selected_voice_readme_is_cc0(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return re.search(r"voices of volunteers.{0,240}licensed as cc0", normalized) is not None


def local_files(root: Path) -> dict[str, Path]:
    root = root.resolve()
    output: dict[str, Path] = {}
    for item in root.rglob("*"):
        relative = item.relative_to(root).as_posix()
        if relative == ".cache" or relative.startswith(".cache/"):
            continue
        safe_path(relative)
        resolved = item.resolve(strict=False)
        if not str(resolved).startswith(str(root) + os.sep):
            raise ValueError(f"path escapes snapshot: {relative}")
        if item.is_symlink() and not resolved.is_file():
            raise ValueError(f"dangling/non-regular symlink: {relative}")
        if item.is_file():
            output[relative] = item
        elif not item.is_dir():
            raise ValueError(f"non-regular snapshot entry: {relative}")
    return output


def validate_server_tree(packet: Any, root: Path, repository: str, revision: str) -> dict[str, Any]:
    if not isinstance(packet, dict) or set(packet) != {"repository", "revision", "resolved_revision", "files"}:
        raise ValueError("server tree envelope keys are not exact")
    if packet["repository"] != repository or packet["revision"] != revision or packet["resolved_revision"] != revision:
        raise ValueError("server tree repository/revision identity mismatch")
    files = packet["files"]
    if not isinstance(files, list):
        raise ValueError("server tree files is not a list")
    remote: dict[str, dict[str, Any]] = {}
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"path", "type", "size", "git_blob_sha1", "lfs_sha256"}:
            raise ValueError("malformed server tree file entry")
        path, kind, size, oid, lfs = (entry[k] for k in ("path", "type", "size", "git_blob_sha1", "lfs_sha256"))
        if not isinstance(path, str) or kind != "file" or not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("malformed server tree file fields")
        safe_path(path)
        if path in remote or not isinstance(oid, str):
            raise ValueError("duplicate/invalid server tree identity")
        if not HEX40.fullmatch(oid):
            raise ValueError(f"invalid Git blob identity for {path}")
        if lfs is not None and (not isinstance(lfs, str) or not HEX64.fullmatch(lfs)):
            raise ValueError(f"invalid LFS identity for {path}")
        remote[path] = entry
    local = local_files(root)
    if set(remote) != set(local):
        raise ValueError(f"server/local tree mismatch: remote={len(remote)} local={len(local)}")
    for path, entry in remote.items():
        file = local[path]
        if file.stat().st_size != entry["size"]:
            raise ValueError(f"size mismatch for {path}")
        actual = sha256(file) if entry["lfs_sha256"] is not None else git_blob_sha1(file)
        expected = entry["lfs_sha256"] or entry["git_blob_sha1"]
        if actual != expected:
            raise ValueError(f"content identity mismatch for {path}")
    return {"repository": repository, "revision": revision, "files": sorted(remote.values(), key=lambda x: x["path"]), "content_identity": "LFS SHA-256 or Git blob SHA-1; server git_blob_sha1 retained for every entry"}


def safetensors_header(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("short safetensors header length")
        length = int.from_bytes(raw, "little")
        if length <= 0 or length > MAX_HEADER_BYTES or length + 8 > size:
            raise ValueError("invalid/bounded safetensors header length")
        header = json.loads(stream.read(length).decode(), object_pairs_hook=unique_pairs)
    if not isinstance(header, dict):
        raise ValueError("safetensors root is not an object")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()):
        raise ValueError("safetensors metadata must be string map")
    data_size = size - 8 - length
    ranges: list[tuple[int, int]] = []
    tensors: dict[str, Any] = {}
    for name, descriptor in header.items():
        if not isinstance(name, str) or not name or not isinstance(descriptor, dict):
            raise ValueError("invalid tensor descriptor")
        safe_path(name)
        if set(descriptor) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"tensor descriptor keys are not exact: {name}")
        dtype, shape, offsets = descriptor.get("dtype"), descriptor.get("shape"), descriptor.get("data_offsets")
        if dtype not in DTYPE_BYTES or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"invalid tensor fields: {name}")
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in shape + offsets):
            raise ValueError(f"invalid tensor integer: {name}")
        begin, end = offsets
        elements = 1
        for dim in shape:
            elements *= dim
        if begin > end or end > data_size or end - begin != elements * DTYPE_BYTES[dtype]:
            raise ValueError(f"invalid tensor bounds: {name}")
        ranges.append((begin, end))
        tensors[name] = {"dtype": dtype, "shape": shape, "data_offsets": offsets, "bytes": end - begin}
    ranges.sort()
    cursor = 0
    for begin, end in ranges:
        if begin != cursor:
            raise ValueError("safetensors gap or overlap")
        cursor = end
    if cursor != data_size:
        raise ValueError("safetensors trailing gap")
    return {"header_bytes": length, "data_bytes": data_size, "metadata": metadata, "tensors": tensors}


def inspect_st(path: Path, expected_count: int, expected_dtypes: dict[str, int], expected_size: int, expected_hash: str) -> dict[str, Any]:
    if path.stat().st_size != expected_size or sha256(path) != expected_hash:
        raise ValueError(f"fixed file identity mismatch: {path.name}")
    header = safetensors_header(path)
    counts: dict[str, int] = {}
    for value in header["tensors"].values():
        counts[value["dtype"]] = counts.get(value["dtype"], 0) + 1
    if len(header["tensors"]) != expected_count or counts != expected_dtypes:
        raise ValueError(f"tensor count/dtype mismatch for {path.name}: {counts}")
    return {"path": path.name, "bytes": expected_size, "sha256": expected_hash, "header": header}


def archive_inventory(path: Path) -> dict[str, Any]:
    members: list[dict[str, Any]] = []
    seen: set[str] = set()
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                safe_path(info.filename)
                if info.filename in seen or (info.is_dir() and not info.filename.endswith("/")):
                    raise ValueError("duplicate/invalid zip member")
                seen.add(info.filename)
                mode = (info.external_attr >> 16) & 0o170000
                if mode not in {0, 0o040000, 0o100000}:
                    raise ValueError("unsafe zip member type")
                members.append({"name": info.filename, "type": "directory" if info.is_dir() else "file", "bytes": info.file_size})
    elif tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as archive:
            for info in archive:
                safe_path(info.name)
                if info.name in seen or not (info.isdir() or info.isfile()):
                    raise ValueError("duplicate/unsafe tar member")
                seen.add(info.name)
                members.append({"name": info.name, "type": "directory" if info.isdir() else "file", "bytes": info.size})
    else:
        raise ValueError("checkpoint is not a safe tar/zip archive")
    return {"path": path.name, "members": members}


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    inventory = archive_inventory(path)
    try:
        import torch

        unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(str(path))
        if unsafe:
            raise ValueError(f"checkpoint unsafe globals present: {unsafe!r}")
        torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001 - preserved as an evidence blocker
        raise ValueError(f"weights-only checkpoint inspection failed: {error}") from error
    return inventory


def git_identity(repo: Path, url: str, revision: str, roles: tuple[str, ...]) -> dict[str, Any]:
    origin = subprocess.check_output(["git", "-C", str(repo), "remote", "get-url", "origin"], text=True).strip()
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"], text=True)
    if status:
        raise ValueError("source checkout is dirty or contains untracked files")
    tracked = set(filter(None, subprocess.check_output(["git", "-C", str(repo), "ls-files", "-z"], text=False).decode().split("\0")))
    if origin != url or head != revision:
        raise ValueError(f"source identity mismatch: {origin}@{head}")
    licenses = []
    for relative in sorted(tracked):
        item = repo / relative
        name = item.name.upper()
        if item.is_file() and (name.startswith("LICENSE") or name in {"NOTICE", "README", "README.MD", "PYPROJECT.TOML"}):
            licenses.append({"path": relative, "sha256": sha256(item), "bytes": item.stat().st_size})
    if not licenses:
        raise ValueError("no tracked source license/readme record found")
    role_records = []
    for role in roles:
        path = repo / role
        if role not in tracked or not path.is_file():
            raise ValueError(f"fixed source role missing: {role}")
        role_records.append({"path": role, "bytes": path.stat().st_size, "sha256": sha256(path)})
    return {"origin": origin, "revision": head, "role_records": role_records, "license_records": sorted(licenses, key=lambda x: x["path"]), "code_license_status": "RECORDED_REQUIRES_PRIMARY_REVIEW", "dependency_license_status": "SEPARATE_REVIEW_REQUIRED"}


def config_evidence(snapshot: Path) -> dict[str, Any]:
    configs = []
    for relative, path in sorted(local_files(snapshot).items()):
        if path.suffix != ".json":
            continue
        value = json_file(path)
        configs.append({"path": relative, "sha256": sha256(path), "json": value})
    root = next((item["json"] for item in configs if item["path"] == "config.json"), None)
    if root is None:
        raise ValueError("fixed config.json missing")
    if not isinstance(root, dict):
        raise ValueError("config.json root must be an object")
    expected = {"card": 2048, "n_q": 32, "dep_q": 32, "delays": [0, 0] + [2] * 31, "dim": 2048, "text_card": 8000, "existing_text_padding_id": 3, "num_heads": 16, "num_layers": 16, "hidden_scale": 4.125, "causal": True, "layer_scale": None, "context": 500, "max_period": 10000, "gating": "silu", "norm": "rms_norm_f32", "positional_embedding": "rope", "depformer_dim": 1024, "depformer_num_heads": 16, "depformer_num_layers": 4, "depformer_dim_feedforward": 3072, "depformer_multi_linear": True, "depformer_pos_emb": "none", "depformer_weights_per_step": True, "depformer_low_rank_embeddings": 128, "demux_second_stream": True, "text_card_out": None, "cross_attention": True}
    for key, wanted in expected.items():
        if key not in root or root[key] != wanted:
            raise ValueError(f"exact root config axis {key} mismatch: {root.get(key)!r}")
    conditioners = root.get("conditioners")
    if not isinstance(conditioners, dict):
        raise ValueError("conditioners must be an exact object")
    speaker = conditioners.get("speaker_wavs")
    if speaker != {"type": "tensor", "tensor": {"dim": 512}}:
        raise ValueError(f"speaker_wavs conditioner mismatch: {speaker!r}")
    cfg = conditioners.get("cfg")
    if cfg != {"type": "lut", "lut": {"n_bins": 7, "dim": 16, "tokenizer": "noop", "possible_values": ["1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0"]}}:
        raise ValueError(f"cfg conditioner mismatch: {cfg!r}")
    control = conditioners.get("control")
    if control != {"type": "lut", "lut": {"dim": 2048, "n_bins": 1, "tokenizer": "noop", "possible_values": ["ok"]}}:
        raise ValueError(f"control conditioner mismatch: {control!r}")
    fuser = root.get("fuser")
    if fuser != {"cross_attention_pos_emb": True, "cross_attention_pos_emb_scale": 1, "sum": ["control", "cfg"], "prepend": [], "cross": ["speaker_wavs"]}:
        raise ValueError(f"fuser mismatch: {fuser!r}")
    tts_config = root.get("tts_config")
    if tts_config != {"audio_delay": 1.28, "second_stream_ahead": 2}:
        raise ValueError(f"tts_config mismatch: {tts_config!r}")
    model_id = root.get("model_id")
    if model_id != {"sig": "1e68beda", "epoch": 240}:
        raise ValueError(f"model_id mismatch: {model_id!r}")
    schedule = root.get("depformer_weights_per_step_schedule")
    if schedule != list(range(8)) + [8] * 8 + [9] * 8 + [10] * 8:
        raise ValueError("schedule mismatch")
    if root.get("model_type") != "tts":
        raise ValueError("model_type mismatch")
    lm_gen = root.get("lm_gen_config")
    if lm_gen != {"temp": 0.6, "text_temp": 0.6}:
        raise ValueError(f"lm_gen_config mismatch: {lm_gen!r}")
    filenames = {"tokenizer_name": SPM_FILE, "mimi_name": MIMI_FILE, "moshi_name": TTS_FILE}
    for key, wanted in filenames.items():
        if root.get(key) != wanted:
            raise ValueError(f"exact filename axis {key} mismatch: {root.get(key)!r}")
    axes = {"root": {key: root[key] for key in expected}, "filenames": filenames, "conditioners": conditioners, "fuser": fuser, "tts_config": tts_config, "model_id": model_id, "schedule": schedule, "model_type": root["model_type"], "lm_gen_config": lm_gen}
    axes["raw_configs"] = configs
    return axes


def sentencepiece_evidence(path: Path) -> dict[str, Any]:
    if path.stat().st_size != SPM_BYTES or sha256(path) != SPM_SHA256:
        raise ValueError("SentencePiece identity mismatch")
    try:
        import sentencepiece as spm

        processor = spm.SentencePieceProcessor(model_file=str(path))
        if processor.get_piece_size() != 8000:
            raise ValueError("SentencePiece base piece count is not 8000")
        pieces = [{"id": i, "piece": processor.id_to_piece(i)} for i in range(8000)]
    except Exception as error:  # noqa: BLE001
        raise ValueError(f"SentencePiece structural load failed: {error}") from error
    return {"path": path.name, "bytes": SPM_BYTES, "sha256": SPM_SHA256, "piece_count": 8000, "pieces": pieces}


def base_manifest() -> dict[str, Any]:
    return {"format": FORMAT, "status": "BLOCKED", "evidence_stage": "INSPECTION_ONLY", "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "cpu_status": "UNSUPPORTED", "metal_status": "BLOCKED_BY_CPU", "parity_status": "NOT_RUN", "publication": "NO_UPLOAD", "environment": {"python": sys.version, "platform": platform.platform()}, "model_license": "CC-BY-4.0 (HF model card; independently separated from source/dependency licenses)", "hf": {"repository": HF_REPOSITORY, "revision": HF_REVISION, "artifact_bytes": HF_ARTIFACT_BYTES, "total_bytes": HF_TOTAL_BYTES}, "fixed_artifacts": {TTS_FILE: {"bytes": TTS_BYTES, "sha256": TTS_SHA256}, MIMI_FILE: {"bytes": MIMI_BYTES, "sha256": MIMI_SHA256}, SPM_FILE: {"bytes": SPM_BYTES, "sha256": SPM_SHA256}, VOICE_FILE: {"bytes": VOICE_BYTES, "sha256": VOICE_SHA256}}, "source_identities": {"moshi": {"origin": MOSHI_URL, "revision": MOSHI_REVISION}, "delayed_streams_modeling": {"origin": DSM_URL, "revision": DSM_REVISION}}, "blockers": ["native Kyutai TTS state machine is not implemented", "second-stream demux and scheduled depformer weights are not implemented", "cross-attention speaker conditioner and CFG/control conditioners are not implemented", "complete Mimi decode is not implemented", "Moshi and delayed-streams source/dependency licenses require separate review"]}


def write_manifest(evidence: Path, manifest: dict[str, Any]) -> None:
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def inspect(args: argparse.Namespace) -> int:
    manifest = base_manifest()
    try:
        snapshot = Path(args.snapshot)
        tree = json_file(Path(args.server_tree))
        manifest["server_tree"] = validate_server_tree(tree, snapshot, HF_REPOSITORY, HF_REVISION)
        files = local_files(snapshot)
        if set(files) != HF_FILES:
            raise ValueError(f"HF snapshot file set mismatch: {sorted(files)}")
        if sum(path.stat().st_size for path in files.values()) != HF_TOTAL_BYTES:
            raise ValueError("HF snapshot total byte count mismatch")
        for required in (TTS_FILE, MIMI_FILE, SPM_FILE, "config.json"):
            if required not in files:
                raise ValueError(f"required HF file missing: {required}")
        readme = files.get("README.md")
        if readme is None:
            raise ValueError("HF model card README.md missing")
        readme_text = readme.read_text(encoding="utf-8").lower()
        if "cc-by-4.0" not in readme_text and "cc by 4.0" not in readme_text:
            raise ValueError("HF model card does not authenticate CC-BY-4.0")
        manifest["model_card"] = {"path": "README.md", "bytes": readme.stat().st_size, "sha256": sha256(readme), "license": "CC-BY-4.0"}
        manifest["weights"] = {"tts": inspect_st(files[TTS_FILE], 418, {"BF16": 410, "F32": 8}, TTS_BYTES, TTS_SHA256), "mimi": inspect_st(files[MIMI_FILE], 318, {"F32": 318}, MIMI_BYTES, MIMI_SHA256)}
        manifest["config"] = config_evidence(snapshot)
        manifest["sentencepiece"] = sentencepiece_evidence(files[SPM_FILE])
        if not args.voice_snapshot or not args.voice_server_tree:
            raise ValueError("selected voice snapshot and server tree are required")
        if args.voice_snapshot and args.voice_server_tree:
            voice_root = Path(args.voice_snapshot)
            voice_tree = json_file(Path(args.voice_server_tree))
            manifest["voice_server_tree"] = validate_server_tree(voice_tree, voice_root, VOICE_REPOSITORY, VOICE_REVISION)
            voice_files = local_files(voice_root)
            if not VOICE_FILE.startswith("voice-donations/") or set(voice_files) != {"README.md", VOICE_FILE}:
                raise ValueError("selected voice snapshot contains unexpected files")
            if voice_files[VOICE_FILE].stat().st_size != VOICE_BYTES or sha256(voice_files[VOICE_FILE]) != VOICE_SHA256:
                raise ValueError("selected voice fixture identity mismatch")
            voice_header = safetensors_header(voice_files[VOICE_FILE])
            if voice_header["metadata"] != {"sig": "1e68beda", "epoch": "240"} or voice_header["tensors"].get("speaker_wavs", {}).get("dtype") != "F32" or voice_header["tensors"].get("speaker_wavs", {}).get("shape") != [1, 512, 125]:
                raise ValueError("selected voice fixture header mismatch")
            if not selected_voice_readme_is_cc0(voice_files["README.md"].read_text(encoding="utf-8")):
                raise ValueError("selected voice fixture lacks its exact nearby README CC0 declaration")
            manifest["voice"] = {"selected_license": "CC0 (README-declared only)", "mixed_repository_license_blocker": True, "fixture": {"path": VOICE_FILE, "bytes": VOICE_BYTES, "sha256": VOICE_SHA256, "header": voice_header}}
        if not args.source or not args.dsm_source:
            raise ValueError("both fixed implementation source checkouts are required")
        manifest["moshi_source"] = git_identity(Path(args.source), MOSHI_URL, MOSHI_REVISION, MOSHI_ROLES)
        manifest["delayed_streams_source"] = git_identity(Path(args.dsm_source), DSM_URL, DSM_REVISION, DSM_ROLES)
        for component in ("audio_detokenizer/model.pt", "vocoder/model.pt"):
            if component in files:
                manifest.setdefault("pickle_components", {})[component] = inspect_checkpoint(files[component])
    except Exception as error:  # evidence must survive every failure
        manifest.setdefault("blockers", []).append(f"inspection error: {type(error).__name__}: {error}")
    write_manifest(Path(args.evidence), manifest)
    return 2


def self_test() -> None:
    assert safe_path("ok/name") is None
    for bad in ("../escape", "/absolute", "a\\b", "a\x00b"):
        try:
            safe_path(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe path accepted: {bad!r}")
    with tempfile.TemporaryDirectory(prefix="vokra-kyutai-selftest-") as directory:
        root = Path(directory) / "snapshot"
        root.mkdir()
        for unsafe_name in ("/absolute", "../parent", ""):
            raw_header = json.dumps({unsafe_name: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
            unsafe_tensor = root / "unsafe.safetensors"
            unsafe_tensor.write_bytes(len(raw_header).to_bytes(8, "little") + raw_header + b"\0" * 4)
            try:
                safetensors_header(unsafe_tensor)
            except ValueError:
                pass
            else:
                raise AssertionError(f"unsafe tensor name accepted: {unsafe_name!r}")
        unsafe_tensor.unlink()
        (root / ".cache").mkdir()
        (root / ".cache" / "ignored").write_text("cache", encoding="utf-8")
        payload = root / "x.txt"
        payload.write_text("x", encoding="utf-8")
        if set(local_files(root)) != {"x.txt"}:
            raise AssertionError("snapshot cache directory was not excluded")
        if not selected_voice_readme_is_cc0("Voices of volunteers are licensed as CC0") or selected_voice_readme_is_cc0("A random CC0 file"):
            raise AssertionError("voice README license proximity check is weak")
        packet = {"repository": HF_REPOSITORY, "revision": HF_REVISION, "resolved_revision": HF_REVISION, "files": [{"path": "x.txt", "type": "file", "size": 1, "git_blob_sha1": git_blob_sha1(payload), "lfs_sha256": None}]}
        validate_server_tree(packet, root, HF_REPOSITORY, HF_REVISION)
        lfs_digest = sha256(payload)
        lfs_packet = {"repository": HF_REPOSITORY, "revision": HF_REVISION, "resolved_revision": HF_REVISION, "files": [{"path": "x.txt", "type": "file", "size": 1, "git_blob_sha1": "1" * 40, "lfs_sha256": lfs_digest}]}
        validate_server_tree(lfs_packet, root, HF_REPOSITORY, HF_REVISION)
        wrong = dict(packet, files=[dict(packet["files"][0], git_blob_sha1="0" * 40)])
        try:
            validate_server_tree(wrong, root, HF_REPOSITORY, HF_REVISION)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong same-size content identity accepted")
        for malformed in (
            dict(packet, files=[dict(packet["files"][0], git_blob_sha1="not-a-git-oid")]),
            dict(packet, files=packet["files"] + [dict(packet["files"][0], path="extra.txt")]),
            dict(packet, files=[]),
        ):
            try:
                validate_server_tree(malformed, root, HF_REPOSITORY, HF_REVISION)
            except ValueError:
                pass
            else:
                raise AssertionError("malformed server tree accepted")
        try:
            validate_server_tree(packet, root, "wrong/repo", HF_REVISION)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong repository accepted")
        try:
            validate_server_tree(dict(packet, resolved_revision="0" * 40), root, HF_REPOSITORY, HF_REVISION)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong resolved revision accepted")
        archive = Path(directory) / "unsafe.tar"
        with tarfile.open(archive, "w") as tar:
            info = tarfile.TarInfo("../escape")
            info.size = 1
            import io
            tar.addfile(info, io.BytesIO(b"x"))
        try:
            archive_inventory(archive)
        except ValueError:
            pass
        else:
            raise AssertionError("traversal archive accepted")
        config = {
            "card": 2048, "n_q": 32, "dep_q": 32, "delays": [0, 0] + [2] * 31,
            "dim": 2048, "text_card": 8000, "existing_text_padding_id": 3,
            "num_heads": 16, "num_layers": 16, "hidden_scale": 4.125,
            "causal": True, "layer_scale": None, "context": 500,
            "max_period": 10000, "gating": "silu", "norm": "rms_norm_f32",
            "positional_embedding": "rope", "depformer_dim": 1024,
            "depformer_num_heads": 16, "depformer_num_layers": 4,
            "depformer_dim_feedforward": 3072, "depformer_multi_linear": True,
            "depformer_pos_emb": "none", "depformer_weights_per_step": True,
            "depformer_low_rank_embeddings": 128, "demux_second_stream": True,
            "text_card_out": None, "cross_attention": True,
            "conditioners": {"speaker_wavs": {"type": "tensor", "tensor": {"dim": 512}}, "cfg": {"type": "lut", "lut": {"n_bins": 7, "dim": 16, "tokenizer": "noop", "possible_values": ["1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0"]}}, "control": {"type": "lut", "lut": {"dim": 2048, "n_bins": 1, "tokenizer": "noop", "possible_values": ["ok"]}}},
            "fuser": {"cross_attention_pos_emb": True, "cross_attention_pos_emb_scale": 1, "sum": ["control", "cfg"], "prepend": [], "cross": ["speaker_wavs"]},
            "tts_config": {"audio_delay": 1.28, "second_stream_ahead": 2},
            "model_id": {"sig": "1e68beda", "epoch": 240},
            "depformer_weights_per_step_schedule": list(range(8)) + [8] * 8 + [9] * 8 + [10] * 8,
            "tokenizer_name": SPM_FILE, "mimi_name": MIMI_FILE, "moshi_name": TTS_FILE,
            "model_type": "tts", "lm_gen_config": {"temp": 0.6, "text_temp": 0.6},
        }
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        assert config_evidence(root)["root"]["card"] == 2048
        misplaced = dict(config)
        misplaced["conditioners"] = dict(config["conditioners"], card=2048)
        misplaced.pop("card")
        (root / "config.json").write_text(json.dumps(misplaced), encoding="utf-8")
        try:
            config_evidence(root)
        except ValueError:
            pass
        else:
            raise AssertionError("nested/missing root config axis accepted")
        legacy = dict(config)
        legacy["low_rank_embeddings"] = 128
        legacy.pop("depformer_low_rank_embeddings")
        (root / "config.json").write_text(json.dumps(legacy), encoding="utf-8")
        try:
            config_evidence(root)
        except ValueError:
            pass
        else:
            raise AssertionError("legacy config key accepted")
        source = Path(directory) / "source"
        (source / "role").mkdir(parents=True)
        (source / "role" / "tts.py").write_text("role", encoding="utf-8")
        (source / "LICENSE-MIT").write_text("MIT", encoding="utf-8")
        subprocess.run(["git", "-C", str(source), "init", "-q"], check=True)
        subprocess.run(["git", "-C", str(source), "config", "user.email", "self-test@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(source), "config", "user.name", "self-test"], check=True)
        subprocess.run(["git", "-C", str(source), "add", "role/tts.py", "LICENSE-MIT"], check=True)
        subprocess.run(["git", "-C", str(source), "commit", "-qm", "fixture"], check=True)
        subprocess.run(["git", "-C", str(source), "remote", "add", "origin", MOSHI_URL], check=True)
        head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        source_packet = git_identity(source, MOSHI_URL, head, ("role/tts.py",))
        if not any(record["path"] == "LICENSE-MIT" for record in source_packet["license_records"]):
            raise AssertionError("tracked LICENSE-MIT was not recorded")
        try:
            git_identity(source, MOSHI_URL, head, ("role/not-tracked.py",))
        except ValueError:
            pass
        else:
            raise AssertionError("untracked role path accepted")
        (source / "untracked.txt").write_text("dirty", encoding="utf-8")
        try:
            git_identity(source, MOSHI_URL, head, ("role/tts.py",))
        except ValueError:
            pass
        else:
            raise AssertionError("dirty source checkout accepted")
        evidence = Path(directory) / "evidence"
        rc = inspect(argparse.Namespace(snapshot=str(root / "missing"), server_tree=str(root / "missing.json"), voice_snapshot=None, voice_server_tree=None, source=None, dsm_source=None, evidence=str(evidence)))
        if rc != 2:
            raise AssertionError("inspection failure did not return exit 2")
        manifest = json_file(evidence / "manifest.json")
        if manifest.get("status") != "BLOCKED" or manifest.get("evidence_stage") != "INSPECTION_ONLY" or not manifest.get("blockers"):
            raise AssertionError("failure manifest is not a blocker evidence packet")
    print("kyutai TTS inspector self-test PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--snapshot")
    parser.add_argument("--server-tree")
    parser.add_argument("--voice-snapshot")
    parser.add_argument("--voice-server-tree")
    parser.add_argument("--source")
    parser.add_argument("--dsm-source")
    parser.add_argument("--evidence", required=False, default="evidence")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.snapshot or not args.server_tree:
        parser.error("--snapshot and --server-tree are required")
    return inspect(args)


if __name__ == "__main__":
    raise SystemExit(main())
