#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""VAST-only, fail-closed evidence inspector for Fun-CosyVoice3.

This tool authenticates a fixed composite release without converting or
executing it.  Pickles are inspected only through ``weights_only=True`` and
ONNX files are never executed.  The source-shaped native route is staged, but
remains ``BLOCKED`` because it has not been authenticated or numerically
compared against the complete official execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any

REPOSITORY = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
REVISION = "29e01c4e8d000f4bcd70751be16fa94bf3d85a18"
SOURCE_REPOSITORY = "https://github.com/FunAudioLLM/CosyVoice.git"
SOURCE_REVISION = "0d990d60740bf174904a5185cce910b847bd3684"
MATCHA_REPOSITORY = "https://github.com/shivammehta25/Matcha-TTS.git"
MATCHA_REVISION = "dd9105b34bf2be2230f4aa1e4769fb586a3c824e"
# The supplied fixed tree packet has 20 regular files.  Its exact per-file
# sizes sum to this value.  HF's API ``usedStorage`` value is a repository
# storage accounting number (including historical revisions), not the sum of
# the current recursive tree; retain both without treating that semantic
# distinction as a content mismatch.
TREE_TOTAL_BYTES = 9_747_516_745
HF_USED_STORAGE_BYTES = 11_767_984_206
FORMAT = "vokra-cosyvoice3-inspection-v1"
IGNORE = {".cache", ".git"}
MAX_HEADER = 64 * 1024 * 1024
MAX_ITEMS = 300_000
MAX_DEPTH = 64
MAX_ARCHIVE_MEMBERS = 100_000
MAX_ARCHIVE_BYTES = 8_000_000_000

# This is the complete fixed HF tree.  Git blob IDs and LFS IDs are kept
# separate: the former authenticates the repository tree, the latter the
# materialized large-file content.
TREE = {
    ".gitattributes": (1574, "cb90d04a72c0b51e3a115db0a9342dd88af903a4", None),
    "CosyVoice-BlankEN/config.json": (659, "463b055262b6c66c4629a74a4b300bfe2ed31d3c", None),
    "CosyVoice-BlankEN/generation_config.json": (242, "dfc11073787daf1b0f9c0f1499487ab5f4c93738", None),
    "CosyVoice-BlankEN/merges.txt": (1402109, "90d3d82d027eadcc6a5e77c38eb82d43fc51b53b", None),
    "CosyVoice-BlankEN/model.safetensors": (988097824, "3dff8ababe3dbf3bd7a556f5f143503ab2ef3c98", "130282af0dfa9fe5840737cc49a0d339d06075f83c5a315c3372c9a0740d0b96"),
    "CosyVoice-BlankEN/tokenizer_config.json": (1287, "ff55d7b9eb1384e5d4d7e75dc0f564c1a8833d6e", None),
    "CosyVoice-BlankEN/vocab.json": (2776833, "4783fe10ac3adce15ac8f358ef5462739852c569", None),
    "README.md": (11982, "d816a921470cff1b6926d31c89e4ec7dea185f32", None),
    "asset/dingding.png": (122824, "e407a9d3c0fc5a7fcac46aef09181a0bef330d37", "7f04815e2e676d31b089af6fa270135f3214f2193d5e0ad98b491d007d48f1c6"),
    "campplus.onnx": (28303423, "7b08523b2e28e437cfb1a0312723a5ab0bac287e", "a6ac6a63997761ae2997373e2ee1c47040854b4b759ea41ec48e4e42df0f4d73"),
    "config.json": (2, "9e26dfeeb6e641a33dae4961196235bdb965b21b", None),
    "configuration.json": (47, "5e812fae901c12933ac69ebf3eb79d0eb49bbab4", None),
    "cosyvoice3.yaml": (6934, "2eda7e5007d99f6b17fbe7bd751cf54e3cde29ea", None),
    "flow.decoder.estimator.fp32.onnx": (1326216933, "3f880eeae966a725cd7c875b8e4c929bf2035489", "9b51b9533a55937762b262bf2cf9c6220ce40760f76d6532cb16a6a6d84059a8"),
    "flow.pt": (1329116148, "074b96e9cfbf3e511067528bde8e76a308f94904", "a6fab32a7825e5b0bc855ddd948f8db9370b0a786fbc249caa4595e95b608e4b"),
    "hift.pt": (83202622, "c5088ac4f7db1314a4efe06ca60e9f47ea2a1900", "b279d7641eb97ae55b3b540cfba4f953c26492a2df758328a89a4d007ab87a65"),
    "llm.pt": (2024669519, "d9813d1f616910e9117d612e6e725b0350f98115", "69f43bd545131c30e98947fb360ea8b4dc9916d8e83dded7757c7ea4f5a24970"),
    "llm.rl.pt": (2024682701, "bc852bfe713ca73dcb2d900145731e6ce2c4a3c2", "74d34b01a80c7154670ae75ac372d1b1712c78bceae9f467eb9f1f6f61ec764f"),
    "speech_tokenizer_v3.batch.onnx": (969451579, "3a4fb34ea654cdfc4e4228b2d485c196af2985fa", "b156b8a7bbff436585e153f4637b9a368009005ac66efa108a6c8bfb34e5ee43"),
    "speech_tokenizer_v3.onnx": (969451503, "91daac1b6f0bbcb54b8885dc7a1cbf054de22f94", "23236a74175dbdda47afc66dbadd5bcb41303c467a57c261cb8539ad9db9208d"),
}
SOURCE_ROLE_BLOBS = {
    "LICENSE": "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64",
    "README.md": "6308ac2cdb8e611f6986806ecc7e9f89dd0b7f66",
    "cosyvoice/cli/model.py": "25610a4235c6a971821e64e275d2b3d1d9f0c669",
    "cosyvoice/cli/frontend.py": "6d397cc99be417e98b8d9fc7a1438ec722312276",
    "cosyvoice/flow/flow.py": "c25518621bf98e95d5ed75b83c5a2a610d0822be",
    "cosyvoice/flow/decoder.py": "97768a459fbb89a2c99f98de302628d8ccafda67",
    "cosyvoice/flow/flow_matching.py": "d3beb9ec2ce8c26972433080c458f90c3f2c0467",
    "cosyvoice/flow/DiT/dit.py": "0d637e4ad2fc8514b62c7ea7e564b7d29d4d91d4",
    "cosyvoice/hifigan/generator.py": "bbc2a2112bfd260963765af33760c95c3161fe14",
    "cosyvoice/hifigan/f0_predictor.py": "c896890612053d97a9567bfe261dce000dae0f52",
    "cosyvoice/llm/llm.py": "b17bd3af7abc3135c32f0b1d5d4dba7b59f15b1f",
    "cosyvoice/tokenizer/tokenizer.py": "6ecf4ae84b1b88d10c42e110db4665f944cc0317",
    "cosyvoice/utils/frontend_utils.py": "ea1c9fc8cc7438f257eecffe460a8b6f4ddf8584",
    "runtime/triton_trtllm/offline_inference.py": "326fb0ed890d65d60f0ca41eb27613cccf5c0543",
    "examples/libritts/cosyvoice3/conf/cosyvoice3.yaml": "36dfee4889b6f1c5a1e85e59a0e716005aa0063f",
}
MATCHA_ROLE_BLOBS = {
    "matcha/models/components/flow_matching.py": "5cad7431ef66a8d11da32a77c1af7f6e31d6b774",
    "matcha/models/components/decoder.py": "1137cd7008e9d07b4f306926a82e44c2b2cddbdf",
}
HISTORICAL = {
    "repository": "vokra/fun-cosyvoice3-0.5b-2512",
    "revision": "37e7d22a665d96dd7eb2e10e43ff4571783670cc",
    "filename": "model.gguf", "bytes": 2577517280,
    "git_blob_sha1": "cacb06cf521bf0bfeeaa0eb1d79113ab4bdf5bb2",
    "lfs_sha256": "d581891f7b25f8b3da80a73b750098108f065f03421e23acf0722f716c3cc84f",
    "tensor_count": 293, "status": "PARTIAL_LLM_ONLY_BLOCKER",
}


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def git_blob_bytes(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def lfs_pointer(size: int, sha256: str) -> bytes:
    return f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha256}\nsize {size}\n".encode()


def strict_json_pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def safe_path(name: str) -> None:
    if not isinstance(name, str) or not name or "\0" in name or "\\" in name:
        raise ValueError(f"unsafe path: {name!r}")
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe path: {name!r}")


def local_files(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise RuntimeError(f"missing snapshot: {root}")
    base = root.resolve()
    files: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in IGNORE for part in rel.parts):
            continue
        if path.is_dir() and not path.is_symlink():
            continue
        safe_path(rel.as_posix())
        if path.is_symlink():
            if not path.exists() or not path.is_file():
                raise RuntimeError(f"dangling/nonregular symlink: {rel}")
            resolved = path.resolve()
            if resolved != base and base not in resolved.parents:
                raise RuntimeError(f"escaping symlink: {rel}")
        if not path.is_file():
            raise RuntimeError(f"nonregular snapshot entry: {rel}")
        if rel.as_posix() in files:
            raise RuntimeError(f"duplicate local path: {rel}")
        files[rel.as_posix()] = path
    if not files:
        raise RuntimeError("empty snapshot")
    return files


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=strict_json_pairs)


def server_tree(snapshot: Path, packet: Path, blockers: list[str]) -> dict[str, Any]:
    remote = read_json(packet)
    if not isinstance(remote, dict):
        raise ValueError("server tree envelope must be an object")
    rows = remote.get("files")
    if not isinstance(rows, list):
        raise ValueError("server tree files must be an array")
    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "type", "size", "git_blob_sha1", "lfs_sha256"}:
            raise ValueError("server tree row schema mismatch")
        name, kind, size, blob, lfs = (row[k] for k in ("path", "type", "size", "git_blob_sha1", "lfs_sha256"))
        safe_path(name)
        if kind != "file" or isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"invalid server row: {name!r}")
        if not isinstance(blob, str) or not re.fullmatch(r"[0-9a-f]{40}", blob):
            raise ValueError(f"invalid Git blob identity: {name!r}")
        if lfs is not None and (not isinstance(lfs, str) or not re.fullmatch(r"[0-9a-f]{64}", lfs)):
            raise ValueError(f"invalid LFS identity: {name!r}")
        if name in records:
            raise ValueError(f"duplicate server path: {name!r}")
        records[name] = {"bytes": size, "git_blob_sha1": blob, "lfs_sha256": lfs}
    local = local_files(snapshot)
    missing, extra, changed = sorted(set(records) - set(local)), sorted(set(local) - set(records)), []
    for name in sorted(set(records) & set(local)):
        row, path = records[name], local[name]
        row["local_sha256"] = digest(path)
        okay = path.stat().st_size == row["bytes"]
        okay = okay and (row["local_sha256"] == row["lfs_sha256"] if row["lfs_sha256"] else git_blob(path) == row["git_blob_sha1"])
        if row["lfs_sha256"]:
            pointer = lfs_pointer(row["bytes"], row["local_sha256"])
            if git_blob_bytes(pointer) != row["git_blob_sha1"]:
                raise ValueError(f"LFS pointer Git blob mismatch: {name}")
        if not okay:
            changed.append(name)
    identity = remote.get("repository") == REPOSITORY and remote.get("revision") == REVISION and remote.get("resolved_revision") == REVISION and remote.get("walk") == "recursive_file_only"
    if not identity:
        blockers.append("server tree repository/revision/walk mismatch")
    if missing or extra or changed:
        blockers.append(f"server/local tree mismatch missing={missing} extra={extra} changed={changed}")
    return {"repository": remote.get("repository"), "revision": remote.get("revision"), "resolved_revision": remote.get("resolved_revision"), "walk": remote.get("walk"), "status": "MATCHED" if identity and not (missing or extra or changed) else "MISMATCH", "files": records, "missing": missing, "extra": extra, "content_mismatch": changed}


def safe_safetensors(path: Path, blockers: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path), "resident": "HEADER_ONLY"}
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError("missing header length")
            length = int.from_bytes(raw, "little")
            if length <= 2 or length > MAX_HEADER or length > size - 8:
                raise ValueError(f"invalid header length: {length}")
            header = json.loads(stream.read(length), object_pairs_hook=strict_json_pairs)
        if not isinstance(header, dict):
            raise ValueError("header is not an object")
        body = size - 8 - length
        desc = []
        allowed = {"F32": 4, "F16": 2, "BF16": 2, "I32": 4, "I64": 8, "U8": 1, "I8": 1, "BOOL": 1}
        for name, item in header.items():
            if name == "__metadata__":
                if not isinstance(item, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in item.items()):
                    raise ValueError("metadata must be string map")
                continue
            safe_path(name)
            if not isinstance(item, dict) or set(item) != {"dtype", "shape", "data_offsets"}:
                raise ValueError(f"descriptor schema: {name}")
            dtype, shape, offsets = item["dtype"], item["shape"], item["data_offsets"]
            if dtype not in allowed or not isinstance(shape, list) or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in shape) or not isinstance(offsets, list) or len(offsets) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in offsets) or offsets[0] > offsets[1] or offsets[1] > body:
                raise ValueError(f"descriptor range/dtype: {name}")
            desc.append((offsets[0], offsets[1], name, shape, dtype))
        desc.sort()
        cursor = 0
        for start, end, *_ in desc:
            if start != cursor:
                raise ValueError("tensor body has gap or overlap")
            cursor = end
        if cursor != body:
            raise ValueError("tensor body has trailing bytes")
        result.update({"status": "HEADER_AUTHENTICATED", "tensor_count": len(desc), "tensors": [{"name": n, "shape": s, "dtype": d, "offsets": [a, b]} for a, b, n, s, d in desc]})
    except Exception as exc:
        blockers.append(f"safetensors blocked ({path.name}): {exc}")
        result["status"] = "BLOCKED"
    return result


def safe_checkpoint(path: Path, blockers: list[str]) -> dict[str, Any]:
    result = {"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path), "resident": "BOUNDED", "safe_load": "WEIGHTS_ONLY"}
    try:
        try:
            archive = zipfile.ZipFile(path)
            infos = archive.infolist()
            members = []
            seen, total = set(), 0
            if len(infos) > MAX_ARCHIVE_MEMBERS:
                raise ValueError("archive member bound")
            for info in infos:
                safe_path(info.filename)
                mode = info.external_attr >> 16
                if info.filename in seen or info.flag_bits & 1 or info.is_dir() or (mode & 0o170000) not in (0, 0o100000) or len(info.filename) > 4096:
                    raise ValueError(f"unsafe ZIP member: {info.filename!r}")
                seen.add(info.filename); total += info.file_size
                if total > MAX_ARCHIVE_BYTES:
                    raise ValueError("archive uncompressed bound")
                members.append({"name": info.filename, "bytes": info.file_size, "type": "file"})
            archive.close()
            result["archive_type"] = "ZIP"
            result["archive_members"] = members
        except zipfile.BadZipFile:
            with tarfile.open(path, mode="r:*") as archive:
                members, seen, total = [], set(), 0
                entries = archive.getmembers()
                if len(entries) > MAX_ARCHIVE_MEMBERS:
                    raise ValueError("archive member bound")
                for info in entries:
                    safe_path(info.name)
                    if info.name in seen or info.issym() or info.islnk() or not (info.isfile() or info.isdir()) or len(info.name) > 4096:
                        raise ValueError(f"unsafe TAR member: {info.name!r}")
                    seen.add(info.name)
                    if info.isfile():
                        total += info.size
                        if total > MAX_ARCHIVE_BYTES:
                            raise ValueError("archive uncompressed bound")
                    members.append({"name": info.name, "bytes": info.size, "type": "directory" if info.isdir() else "file"})
                result["archive_type"] = "TAR"
                result["archive_members"] = members
    except (zipfile.BadZipFile, tarfile.ReadError):
        result["archive_members"] = []
    except Exception as exc:
        blockers.append(f"checkpoint archive blocked ({path.name}): {exc}")
    try:
        import torch
        unsafe_fn = getattr(torch.serialization, "get_unsafe_globals_in_checkpoint", None)
        unsafe = unsafe_fn(str(path)) if unsafe_fn else ["unavailable"]
        result["unsafe_globals"] = unsafe
        if unsafe:
            blockers.append(f"unsafe checkpoint globals ({path.name}): {unsafe}")
        # Deliberately no fallback: this invocation is the only accepted load.
        value = torch.load(path, map_location="cpu", weights_only=True)
        tensors, seen, count = [], set(), 0
        def walk(item: Any, name: str = "", depth: int = 0) -> None:
            nonlocal count
            count += 1
            if count > MAX_ITEMS or depth > MAX_DEPTH:
                raise ValueError("checkpoint walk bound exceeded")
            if isinstance(item, torch.Tensor):
                finite = bool(torch.isfinite(item).all().item()) if item.is_floating_point() else True
                row = {"name": name, "shape": [int(x) for x in item.shape], "dtype": str(item.dtype), "numel": int(item.numel()), "finite": finite}
                tensors.append(row)
                if not finite:
                    raise ValueError(f"non-finite tensor: {name}")
                return
            if item is None or isinstance(item, (bool, int, float, str)):
                return
            identity = id(item)
            if identity in seen:
                raise ValueError(f"checkpoint cycle: {name}")
            seen.add(identity)
            if isinstance(item, dict):
                for key, child in item.items():
                    safe_path(key)
                    walk(child, f"{name}.{key}" if name else key, depth + 1)
            elif isinstance(item, (list, tuple)):
                for index, child in enumerate(item):
                    walk(child, f"{name}[{index}]", depth + 1)
            else:
                raise ValueError(f"unsupported checkpoint object: {type(item).__name__}")
            seen.remove(identity)
        walk(value)
        manifest = json.dumps(tensors, separators=(",", ":"), sort_keys=True).encode()
        result.update({"loaded_type": type(value).__name__, "tensor_count": len(tensors), "tensor_manifest": tensors, "tensor_manifest_sha256": hashlib.sha256(manifest).hexdigest()})
    except Exception as exc:
        blockers.append(f"weights_only checkpoint blocked ({path.name}): {exc}")
        result["safe_load"] = "BLOCKED"
    return result


def qwen_evidence(snapshot: Path, blockers: list[str]) -> dict[str, Any]:
    root = read_json(snapshot / "CosyVoice-BlankEN/config.json")
    expected = {"architectures": ["Qwen2ForCausalLM"], "vocab_size": 151936, "hidden_size": 896, "intermediate_size": 4864, "num_hidden_layers": 24, "num_attention_heads": 14, "num_key_value_heads": 2, "hidden_act": "silu", "max_position_embeddings": 32768, "rms_norm_eps": 1e-6, "rope_theta": 1_000_000.0, "torch_dtype": "bfloat16", "tie_word_embeddings": True}
    if not isinstance(root, dict) or any(root.get(k) != v for k, v in expected.items()):
        blockers.append("Qwen config exact axes mismatch")
    tokenizer = read_json(snapshot / "CosyVoice-BlankEN/tokenizer_config.json")
    if not isinstance(tokenizer, dict) or tokenizer.get("tokenizer_class") != "Qwen2Tokenizer" or tokenizer.get("model_max_length") != 32768:
        blockers.append("Qwen tokenizer config mismatch")
    decoder = tokenizer.get("added_tokens_decoder", {}) if isinstance(tokenizer, dict) else {}
    expected_tokens = {"151643": "<|endoftext|>", "151644": "<|im_start|>", "151645": "<|im_end|>"}
    if not isinstance(decoder, dict) or any(not isinstance(decoder.get(key), dict) or decoder[key].get("content") != value or decoder[key].get("special") is not True for key, value in expected_tokens.items()):
        blockers.append("Qwen tokenizer special-token mapping mismatch")
    vocab = read_json(snapshot / "CosyVoice-BlankEN/vocab.json")
    if not isinstance(vocab, dict) or len(vocab) < 151643 or set(vocab.values()) != set(range(len(vocab))):
        blockers.append("Qwen vocabulary is not contiguous")
    merges = (snapshot / "CosyVoice-BlankEN/merges.txt").read_text(encoding="utf-8").splitlines()
    if not merges or not merges[0].startswith("#version:") or len([x for x in merges[1:] if x.strip()]) != 134839:
        blockers.append("Qwen merges structure mismatch")
    generation = read_json(snapshot / "CosyVoice-BlankEN/generation_config.json")
    if generation.get("bos_token_id") != 151643 or generation.get("pad_token_id") != 151643:
        blockers.append("Qwen generation token ids mismatch")
    return {"config": root, "tokenizer_config": tokenizer, "vocabulary_count": len(vocab) if isinstance(vocab, dict) else None, "merge_count": len([x for x in merges[1:] if x.strip()]), "generation_config": generation}


def yaml_evidence(snapshot: Path, blockers: list[str]) -> dict[str, Any]:
    path = snapshot / "cosyvoice3.yaml"
    record = {"sha256": digest(path), "expected_sha256": "f5a6b2c6f05139d0f18861a1fe506f751e787026b77c05f7e8fef9f8a4405965"}
    if record["sha256"] != record["expected_sha256"]:
        blockers.append("HF cosyvoice3.yaml SHA256 mismatch")
    text = path.read_text(encoding="utf-8")
    required = {"sample_rate": 24000, "llm_input_size": 896, "llm_output_size": 896, "spk_embed_dim": 192, "token_frame_rate": 25, "token_mel_ratio": 2, "chunk_size": 25, "speech_token_size": 6561, "input_size": 80, "output_size": 80, "pre_lookahead_len": 3, "channels": 1024, "dim": 1024, "depth": 22, "heads": 16, "dim_head": 64, "ff_mult": 2, "base_channels": 512, "nb_harmonics": 8, "n_fft": 16, "hop_len": 4}
    missing = [key for key, value in required.items() if not re.search(rf"(?m)^\s*{re.escape(key)}\s*:\s*{value}\s*(?:#.*)?$", text)]
    if missing:
        blockers.append(f"HF YAML runtime axes missing: {missing}")
    scalar_markers = {
        "mix_ratio": r"(?m)^\s*mix_ratio:\s*\[5,\s*15\]",
        "ras_top_p": r"(?m)^\s*top_p:\s*0\.8",
        "ras_top_k": r"(?m)^\s*top_k:\s*25",
        "ras_win_size": r"(?m)^\s*win_size:\s*10",
        "ras_tau": r"(?m)^\s*tau_r:\s*0\.1",
        "upsample_rates": r"(?m)^\s*upsample_rates:\s*\[8,\s*5,\s*3\]",
        "upsample_kernels": r"(?m)^\s*upsample_kernel_sizes:\s*\[16,\s*11,\s*7\]",
    }
    missing.extend(name for name, pattern in scalar_markers.items() if not re.search(pattern, text))
    if missing:
        blockers.append(f"HF YAML component axes missing: {missing}")
    try:
        import yaml
        class Loader(yaml.SafeLoader):
            pass
        def unknown(loader, tag, node):
            if isinstance(node, yaml.ScalarNode):
                return loader.construct_scalar(node)
            if isinstance(node, yaml.SequenceNode):
                return loader.construct_sequence(node)
            return loader.construct_mapping(node)
        def mapping(loader, node, deep=False):
            pairs = loader.construct_pairs(node, deep=deep)
            if len({key for key, _ in pairs}) != len(pairs):
                raise ValueError("duplicate YAML key")
            return dict(pairs)
        Loader.add_multi_constructor("!", unknown)
        Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
        parsed = yaml.load(text, Loader=Loader)
        if not isinstance(parsed, dict):
            raise ValueError("YAML root is not mapping")
        record["parsed_keys"] = sorted(parsed)
    except Exception as exc:
        blockers.append(f"YAML safe parse blocked: {exc}")
    return record


def source_inventory(root: Path, matcha: Path, blockers: list[str]) -> dict[str, Any]:
    # `runtime/triton_trtllm/infer_cosyvoice3.py` was never present in the
    # fixed upstream commit.  Requiring a later/development-only path would
    # turn an otherwise authenticated checkout into a false inspection error.
    roles = tuple(SOURCE_ROLE_BLOBS)
    matcha_roles = tuple(MATCHA_ROLE_BLOBS)
    result: dict[str, Any] = {"repository": SOURCE_REPOSITORY, "pinned_revision": SOURCE_REVISION, "matcha_repository": MATCHA_REPOSITORY, "matcha_pinned_revision": MATCHA_REVISION, "checkouts": []}
    for path, repo, rev, required in ((root, SOURCE_REPOSITORY, SOURCE_REVISION, roles), (matcha, MATCHA_REPOSITORY, MATCHA_REVISION, matcha_roles)):
        try:
            head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
            origin = subprocess.run(["git", "-C", str(path), "remote", "get-url", "origin"], check=True, capture_output=True, text=True).stdout.strip()
            dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"], check=True, capture_output=True, text=True).stdout
            tracked = set(subprocess.run(["git", "-C", str(path), "ls-files"], check=True, capture_output=True, text=True).stdout.splitlines())
            missing = [role for role in required if role not in tracked or not (path / role).is_file()]
            if head != rev or origin.rstrip("/") != repo.rstrip("/") or dirty or missing:
                blockers.append(f"source identity/roles/clean mismatch: {repo}")
            expected_blobs = SOURCE_ROLE_BLOBS if path == root else MATCHA_ROLE_BLOBS
            records = []
            for role in required:
                if (path / role).is_file():
                    blob = subprocess.run(["git", "-C", str(path), "rev-parse", f"HEAD:{role}"], check=True, capture_output=True, text=True).stdout.strip()
                    if blob != expected_blobs[role]:
                        blockers.append(f"source role Git blob mismatch: {repo}:{role}")
                    records.append({"path": role, "sha256": digest(path / role), "git_blob_sha1": blob, "expected_git_blob_sha1": expected_blobs[role]})
            checkout = {"repository": repo, "resolved_revision": head, "origin": origin, "clean": not bool(dirty), "tracked_roles": records, "missing_roles": missing}
            license_files = [role for role in tracked if role.upper().startswith("LICENSE") or role.upper().startswith("NOTICE")]
            license_records = []
            for role in sorted(license_files):
                license_path = path / role
                try:
                    license_text = license_path.read_text(encoding="utf-8")
                except Exception as exc:
                    blockers.append(f"license record unreadable: {repo}:{role}: {exc}")
                    continue
                license_records.append({"path": role, "sha256": digest(license_path), "utf8": True, "apache_marker": "apache" in license_text.lower(), "mit_marker": "mit license" in license_text.lower()})
            if not license_records:
                blockers.append(f"no tracked license record: {repo}")
            checkout["license_records"] = license_records
            if path == root:
                submodule = subprocess.run(["git", "-C", str(path), "ls-files", "-s", "third_party/Matcha-TTS"], check=True, capture_output=True, text=True).stdout.strip()
                configured = subprocess.run(["git", "-C", str(path), "config", "-f", ".gitmodules", "--get", "submodule.third_party/Matcha-TTS.url"], check=False, capture_output=True, text=True).stdout.strip()
                mode = submodule.split()[0] if submodule else ""
                gitlink_oid = submodule.split()[1] if len(submodule.split()) > 1 else None
                if mode != "160000" or gitlink_oid != MATCHA_REVISION or configured.rstrip("/") != MATCHA_REPOSITORY.rstrip("/"):
                    blockers.append("Matcha gitlink/origin is not authenticated")
                checkout["matcha_gitlink"] = {"mode": mode, "revision": gitlink_oid, "origin": configured, "expected_revision": MATCHA_REVISION, "expected_origin": MATCHA_REPOSITORY}
            result["checkouts"].append(checkout)
        except Exception as exc:
            blockers.append(f"source inventory blocked ({repo}): {exc}")
    return result


def base_manifest(blockers: list[str], inspection_status: str, **extra: Any) -> dict[str, Any]:
    return {"format": FORMAT, "status": "BLOCKED", "inspection_status": inspection_status, "evidence_stage": "INSPECTION_ONLY", "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "cpu_status": "UNSUPPORTED_FULL_TTS_PENDING", "metal_status": "BLOCKED_BY_CPU", "parity_status": "NOT_RUN", "native_status": "STAGED_UNAUTHENTICATED_UNCOMPARED", "publication": "NO_UPLOAD", "model_identity": {"repository": REPOSITORY, "revision": REVISION}, "source_identity": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION, "matcha_repository": MATCHA_REPOSITORY, "matcha_revision": MATCHA_REVISION}, "blockers": sorted(set(blockers)), **extra}


def inspect(snapshot: Path, source: Path, matcha: Path, tree: Path, out: Path) -> int:
    blockers: list[str] = []
    remote = server_tree(snapshot, tree, blockers)
    tree_sum = sum(row["bytes"] for row in remote["files"].values())
    if len(remote["files"]) != len(TREE) or tree_sum != TREE_TOTAL_BYTES:
        blockers.append("complete HF tree count/total mismatch")
    for name, (size, blob, lfs) in TREE.items():
        path, row = snapshot / name, remote["files"].get(name)
        if not path.is_file() or not row or row["bytes"] != size or row["git_blob_sha1"] != blob or row["lfs_sha256"] != lfs:
            blockers.append(f"fixed tree identity mismatch: {name}")
        if lfs and path.is_file() and digest(path) != lfs:
            blockers.append(f"fixed LFS content mismatch: {name}")
    if remote["files"] and remote["files"].keys() != TREE.keys():
        blockers.append("HF tree has unexpected/missing file")
    top_config = snapshot / "config.json"
    if top_config.is_file():
        try:
            if read_json(top_config) != {}:
                blockers.append("top-level config.json must be exactly empty object")
        except Exception as exc:
            blockers.append(f"top-level config.json blocked: {exc}")
    else:
        blockers.append("top-level config.json missing")
    package_config = snapshot / "configuration.json"
    if package_config.is_file():
        try:
            if read_json(package_config) != {"framework": "Pytorch", "task": "text-to-speech"}:
                blockers.append("configuration.json task/framework mismatch")
        except Exception as exc:
            blockers.append(f"configuration.json blocked: {exc}")
    else:
        blockers.append("configuration.json missing")
    qwen = qwen_evidence(snapshot, blockers)
    yaml = yaml_evidence(snapshot, blockers)
    model = safe_safetensors(snapshot / "CosyVoice-BlankEN/model.safetensors", blockers)
    checkpoints = {name: safe_checkpoint(snapshot / name, blockers) for name in ("llm.pt", "llm.rl.pt", "flow.pt", "hift.pt") if (snapshot / name).is_file()}
    onnx = [{"path": name, "bytes": row["bytes"], "git_blob_sha1": row["git_blob_sha1"], "lfs_sha256": row["lfs_sha256"], "execution": "NOT_RUN_NATIVE_BLOCKER"} for name, row in remote["files"].items() if name.endswith(".onnx")]
    source = source_inventory(source, matcha, blockers)
    complete = not blockers
    blockers.extend(["native composite TTS math is staged but not authenticated/compared", "CPU numerical parity is not run", "Metal parity is blocked by CPU", "speech-tokenizer/CampPlus/flow native inspection remains a blocker", "dataset and dependency licenses require separate audit"])
    payload = base_manifest(blockers, "AUTHENTICATED_EVIDENCE_COMPLETE" if complete else "INSPECTION_ERROR", model={"repository": REPOSITORY, "revision": REVISION, "hf_used_storage_bytes": HF_USED_STORAGE_BYTES, "fixed_tree_sum_bytes": TREE_TOTAL_BYTES, "server_tree": remote, "qwen": qwen, "yaml": yaml, "safetensors": model, "checkpoints": checkpoints, "onnx": onnx}, official_source=source, licenses={"weights": "apache-2.0", "source": "apache-2.0", "matcha": "MIT", "dependency": "REVIEW_REQUIRED"}, historical_public_artifact=HISTORICAL)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return 2


def self_test() -> None:
    assert len(TREE) == 20 and sum(v[0] for v in TREE.values()) == TREE_TOTAL_BYTES
    assert HF_USED_STORAGE_BYTES != TREE_TOTAL_BYTES
    assert HISTORICAL["tensor_count"] == 293
    with tempfile.TemporaryDirectory(prefix="cosyvoice3-inspect-") as tmp:
        root, snap = Path(tmp), Path(tmp) / "snapshot"; snap.mkdir()
        sample = snap / "x"; sample.write_bytes(b"payload")
        packet = root / "tree.json"; packet.write_text(json.dumps({"repository": REPOSITORY, "revision": REVISION, "resolved_revision": REVISION, "walk": "recursive_file_only", "files": [{"path": "x", "type": "file", "size": 7, "git_blob_sha1": git_blob(sample), "lfs_sha256": None}]}))
        bad: list[str] = []; assert server_tree(snap, packet, bad)["status"] == "MATCHED" and not bad
        sample.write_bytes(b"changed"); bad = []; assert server_tree(snap, packet, bad)["status"] == "MISMATCH" and bad
        lfs_snap = root / "lfs-snapshot"; lfs_snap.mkdir(); payload = lfs_snap / "payload"; payload.write_bytes(b"payload")
        payload_sha = digest(payload); pointer = lfs_pointer(payload.stat().st_size, payload_sha)
        pointer_blob = git_blob_bytes(pointer)
        lfs_packet = root / "lfs-tree.json"
        lfs_packet.write_text(json.dumps({"repository": REPOSITORY, "revision": REVISION, "resolved_revision": REVISION, "walk": "recursive_file_only", "files": [{"path": "payload", "type": "file", "size": payload.stat().st_size, "git_blob_sha1": pointer_blob, "lfs_sha256": payload_sha}]}))
        bad = []; assert server_tree(lfs_snap, lfs_packet, bad)["status"] == "MATCHED" and not bad
        for unsafe in ("../x", "/x", "a\\b", "a\x00b"):
            try: safe_path(unsafe); raise AssertionError(unsafe)
            except ValueError: pass
        archive = root / "bad.pt"
        with zipfile.ZipFile(archive, "w") as z: z.writestr("../escape", b"x")
        bad = []; safe_checkpoint(archive, bad); assert bad
        st = root / "bad.safetensors"; st.write_bytes((64).to_bytes(8, "little") + b"{}" + b"x")
        bad = []; safe_safetensors(st, bad); assert bad
        header = json.dumps({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        good = root / "good.safetensors"; good.write_bytes(len(header).to_bytes(8, "little") + header + b"\0\0\0\0")
        bad = []; assert safe_safetensors(good, bad)["status"] == "HEADER_AUTHENTICATED" and not bad
        unsafe_header = json.dumps({"../x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        unsafe = root / "unsafe.safetensors"; unsafe.write_bytes(len(unsafe_header).to_bytes(8, "little") + unsafe_header + b"\0\0\0\0")
        bad = []; safe_safetensors(unsafe, bad); assert bad
        assert any(v[2] for v in TREE.values())
    print("cosyvoice3_inspect self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--self-test", action="store_true"); parser.add_argument("--snapshot", type=Path); parser.add_argument("--source", type=Path); parser.add_argument("--matcha-source", type=Path); parser.add_argument("--server-tree", type=Path); parser.add_argument("--output", type=Path); args = parser.parse_args()
    if args.self_test:
        if any(x is not None for x in (args.snapshot, args.source, args.matcha_source, args.server_tree, args.output)): parser.error("--self-test accepts no paths")
        self_test(); return 0
    if any(x is None for x in (args.snapshot, args.source, args.matcha_source, args.server_tree, args.output)): parser.error("normal run requires snapshot/source/matcha-source/server-tree/output")
    try:
        return inspect(args.snapshot, args.source, args.matcha_source, args.server_tree, args.output)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        args.output.joinpath("manifest.json").write_text(json.dumps(base_manifest([f"inspection exception: {exc}"], "INSPECTION_ERROR"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
