#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Inspection-only evidence collector for IBM Granite Speech 4.1-2B."""
from __future__ import annotations

import argparse, base64, hashlib, json, os, re, struct, subprocess, sys, tempfile
from pathlib import Path, PurePosixPath
from typing import Any

HF_REPOSITORY = "ibm-granite/granite-speech-4.1-2b"
HF_REVISION = "de575db64086f84fdc79da4932d1076e965bc546"
SOURCE_REPOSITORY = "https://github.com/ibm-granite/granite-speech.git"
SOURCE_REVISION = "77b7b12fff71f577105b517645750717a1598caa"
TRANSFORMERS_REPOSITORY = "https://github.com/huggingface/transformers.git"
TRANSFORMERS_REVISION = "753d61104116eefc8ffc977327b441ee0c8d599f"
TRANSFORMERS_ROLE_FILES = (
    "src/transformers/models/granite_speech/configuration_granite_speech.py",
    "src/transformers/models/granite_speech/feature_extraction_granite_speech.py",
    "src/transformers/models/granite_speech/modeling_granite_speech.py",
    "src/transformers/models/granite_speech/processing_granite_speech.py",
    "src/transformers/models/granite/configuration_granite.py",
    "src/transformers/models/granite/modeling_granite.py",
)
# Exact IBM source roles are pinned independently from the checkout.  A
# revision-only clone is not sufficient: both the index object and the
# streamed working-tree blob must match these reviewed values.
SOURCE_ROLE_FILES = (
    "granite_speech/_backends/transformers.py",
    "granite_speech/_models.py",
    "granite_speech/audio.py",
    "granite_speech/loader.py",
    "granite_speech/model.py",
)
SOURCE_ROLE_BLOBS: dict[str, str] = {
    "granite_speech/_backends/transformers.py": "9734291e7083f62e8e508fd6d7f99352c37116fa",
    "granite_speech/_models.py": "97018670c6ce485a7572f3e4c7f424d92d789a03",
    "granite_speech/audio.py": "95bda5cbe9f861ab61d62ead507c2a919e08f5f7",
    "granite_speech/loader.py": "5e9ea6611da98b2e41b753f71eaccb6f2a5195e5",
    "granite_speech/model.py": "245c16dd2c78a4b275ffb4eef7e8a3b45acbfffa",
}
TRANSFORMERS_ROLE_BLOBS: dict[str, str] = {
    "src/transformers/models/granite_speech/configuration_granite_speech.py": "fede07b7b7e820e78f44538313a85d39afc811d7",
    "src/transformers/models/granite_speech/feature_extraction_granite_speech.py": "7528fc7ea5bd9efa6ae322d7fd2e40b567855359",
    "src/transformers/models/granite_speech/modeling_granite_speech.py": "1e44c9781dec683cf4b12bcb9d55d32b3635fb53",
    "src/transformers/models/granite_speech/processing_granite_speech.py": "84515d173c471198b987081198aeeed9415252c9",
    "src/transformers/models/granite/configuration_granite.py": "61d3ba9e7bb2775e537608d277e5973ec42a8cf9",
    "src/transformers/models/granite/modeling_granite.py": "846865c55508e223bf6b512ba03b2b64bd0e2434",
}
SOURCE_LICENSE_BLOB = "aab2b952eb7f3b4e848288271cd8fb0ed771c4a2"
TRANSFORMERS_LICENSE_BLOB = "68b7d66c97d66c58de883ed0c451af2b3183e6f3"
FORMAT = "vokra-granite-speech-4-1-2b-inspection-v1"
SHARDS = {
    "model-00001-of-00003.safetensors": (2_143_518_808, "3c987fdc29940c49d2498ea5925e8d57f88661af3ef30f73e56e2434ded3e42f"),
    "model-00002-of-00003.safetensors": (2_143_963_456, "8e18d6d3fbe009a95a4cf305e31c2aab4a3484eccbce29aa1aa1454fc8c046ee"),
    "model-00003-of-00003.safetensors": (339_045_512, "32f823497bc179f6f346efdd46984ab60e44b3d443bf40a18d757ddce626a2d2"),
}
AUXILIARY = {"out_llm.safetensors": (205_723_810, "6cc10d68fe05aec359aceffd597617c875b23f27211ee6dcdb7510d9e90fc64e")}
# The Hub API can expose Xet-backed blobs with ``lfs=None``.  These fixed
# weight paths are nevertheless LFS payloads: the packet must carry the
# canonical pointer Git object and the payload digest separately.  Never
# infer regular-Git status solely from the optional API ``lfs`` field.
LFS_ARTIFACTS = {**SHARDS, **AUXILIARY}
DTYPE_BYTES = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1, "I16": 2, "U16": 2, "I32": 4, "U32": 4, "I64": 8, "U64": 8}
EXPECTED_INDEX_BYTES = 84_396
EXPECTED_MAIN_TENSORS = 954
EXPECTED_MAIN_COUNTS = (221, 511, 222)
SIGSTORE_IGNORE_PATHS = (".gitattributes", ".github", ".gitignore", ".cache", ".git", "model.sig")
TRANSPORT_CACHE = (".cache", "huggingface")
CONFIG_FACTS = {
    ("architectures",): ["GraniteSpeechForConditionalGeneration"],
    ("audio_token_index",): 100352, ("downsample_rate",): 5, ("window_size",): 15,
    ("encoder_config", "num_layers"): 16, ("encoder_config", "hidden_dim"): 1024,
    ("encoder_config", "num_heads"): 8, ("encoder_config", "dim_head"): 128,
    ("encoder_config", "conv_kernel_size"): 15, ("encoder_config", "input_dim"): 160,
    ("encoder_config", "output_dim"): 348, ("encoder_config", "context_size"): 200,
    ("projector_config", "num_hidden_layers"): 2, ("projector_config", "hidden_size"): 1024,
    ("projector_config", "intermediate_size"): 4096, ("projector_config", "num_attention_heads"): 16,
    ("projector_config", "max_position_embeddings"): 2048,
    ("text_config", "num_hidden_layers"): 40, ("text_config", "hidden_size"): 2048,
    ("text_config", "intermediate_size"): 4096, ("text_config", "num_attention_heads"): 16,
    ("text_config", "num_key_value_heads"): 4, ("text_config", "vocab_size"): 100353,
    ("text_config", "max_position_embeddings"): 4096,
    ("text_config", "attention_multiplier"): 0.0078125, ("text_config", "embedding_multiplier"): 12.0,
    ("text_config", "logits_scaling"): 8.0, ("text_config", "residual_multiplier"): 0.22,
    ("transformers_version",): "4.57.6",
}
# The official snapshot only binds sampling_rate at the top level.  The
# feature extractor's window/feature defaults live in the pinned Transformers
# implementation, not in this JSON, so treating their absence as a mismatch
# would invent a model-file contract.
PREPROCESSOR_FACTS = {"sampling_rate": 16000}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()

def git_blob_sha1(path: Path) -> str:
    size=path.stat().st_size; digest=hashlib.sha1()  # Git's non-LFS blob identity.
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()

def git_blob_sha1_bytes(data: bytes) -> str:
    digest=hashlib.sha1(); digest.update(f"blob {len(data)}\0".encode()); digest.update(data); return digest.hexdigest()

def tracked_symlink(root: Path, relative: str, object_id: str, blockers: list[str]) -> dict[str, Any] | None:
    """Authenticate a Git symlink without dereferencing an untrusted target."""
    try:
        if not relative or "\x00" in relative or "\\" in relative or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
            raise ValueError("unsafe tracked symlink path")
        path = root / relative
        target_bytes = subprocess.run(
            ["git", "-C", str(root), "cat-file", "blob", object_id],
            check=True, capture_output=True,
        ).stdout
        target = target_bytes.decode("utf-8", errors="strict")
        if not target or "\x00" in target or "\\" in target or target.endswith("\n"):
            raise ValueError("unsafe Git symlink target encoding")
        if git_blob_sha1_bytes(target_bytes) != object_id:
            raise ValueError("Git symlink object does not match target bytes")
        if target.startswith("/"):
            raise ValueError("absolute symlink target")
        # Normalize only the link text. Do not call Path.resolve(): that would
        # follow a target supplied by the checkout and could escape the tree.
        parts: list[str] = []
        for component in (*PurePosixPath(relative).parent.parts, *target.split("/")):
            if component in ("", "."):
                continue
            if component == "..":
                if not parts:
                    raise ValueError("symlink target escapes checkout")
                parts.pop()
            elif "\x00" in component:
                raise ValueError("unsafe symlink target component")
            else:
                parts.append(component)
        normalized = PurePosixPath(*parts).as_posix()
        working_target = os.readlink(path)
        if working_target != target:
            raise ValueError("working-tree symlink target differs from Git index")
        return {
            "path": relative,
            "index_object_id": object_id,
            "index_target": target,
            "working_target": working_target,
            "normalized_target": normalized,
            "target_scope": "CHECKOUT_RELATIVE_NO_DEREFERENCE",
            "target_git_blob_sha1": git_blob_sha1_bytes(target_bytes),
        }
    except (OSError, UnicodeError, ValueError, subprocess.CalledProcessError) as error:
        blockers.append(f"unsafe tracked symlink {relative}: {error}")
        return None

def lfs_pointer_bytes(payload_sha256: str, payload_bytes: int) -> bytes:
    return f"version https://git-lfs.github.com/spec/v1\noid sha256:{payload_sha256}\nsize {payload_bytes}\n".encode()

def files(root: Path) -> list[Path]:
    if not root.is_dir(): raise RuntimeError(f"missing snapshot: {root}")
    result = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts == (TRANSPORT_CACHE[0],) or relative.parts[:2] == TRANSPORT_CACHE:
            if path.is_symlink():
                raise RuntimeError(f"transport cache symlink is not allowed: {path}")
            continue
        if any(part in {".cache", ".git"} for part in relative.parts):
            raise RuntimeError(f"unexpected cache/control-tree member: {path}")
        if path.is_dir() and not path.is_symlink(): continue
        if path.is_symlink():
            if not path.exists() or not path.is_file(): raise RuntimeError(f"dangling/nonregular symlink: {path}")
            raise RuntimeError(f"symlink is not an authenticated regular file: {path}")
        if not path.is_file(): raise RuntimeError(f"nonregular snapshot member: {path}")
        result.append(path)
    if not result: raise RuntimeError("empty snapshot")
    return result

def identity(path: Path, root: Path, include_git_blob: bool = False) -> dict[str, Any]:
    result = {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
    if include_git_blob: result["git_blob_sha1"] = git_blob_sha1(path)
    return result

def safe_direct_basename(value: Any, allowed: set[str]) -> bool:
    return (isinstance(value, str) and "\x00" not in value and "\\" not in value
            and not value.startswith("/") and Path(value).name == value
            and ".." not in Path(value).parts and value in allowed)

def contract_packet(path: Path, root: Path, facts: dict[tuple[str, ...], Any], blockers: list[str]) -> dict[str, Any]:
    packet=json_packet(path,root)
    value=json.loads(path.read_text(encoding="utf-8"),object_pairs_hook=_no_duplicate_keys)
    observed={}
    for parts, expected in facts.items():
        current=value
        try:
            for part in parts: current=current[part]
            observed[".".join(parts)]=current
            if current != expected: blockers.append(f"config fact mismatch at {path}:{'.'.join(parts)}")
        except (KeyError,TypeError): blockers.append(f"config fact missing at {path}:{'.'.join(parts)}")
    packet.update({"contract_status":"EXACT_FACTS_MATCHED" if not any(str(path) in item for item in blockers) else "BLOCKED_FACTS","expected_facts":{".".join(k):v for k,v in facts.items()},"observed_facts":observed})
    return packet

def server_tree(snapshot: Path, packet: Path, blockers: list[str]) -> dict[str, Any]:
    try: remote = json.loads(packet.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)
    except Exception as error: blockers.append(f"server tree parse failed: {error}"); return {"status": "BLOCKED"}
    schema_ok = isinstance(remote, dict) and set(remote) == {"repository", "requested_revision", "resolved_revision", "head_commit", "walk", "files"}
    if not schema_ok:
        blockers.append("server tree top-level schema mismatch")
    rows = remote.get("files", []) if isinstance(remote, dict) else []
    remote_records: dict[str, dict[str, Any]] = {}
    if not isinstance(rows, list):
        blockers.append("server tree files is not an array")
        rows = []
    for item in rows:
        if not isinstance(item, dict) or set(item) != {"path", "type", "size", "lfs_payload_size", "head_commit", "head_size", "head_etag", "git_blob_sha1", "lfs_pointer_git_blob_sha1", "lfs_sha256"}:
            blockers.append(f"server tree row schema mismatch: {item!r}")
            continue
        if not isinstance(item, dict) or item.get("type") != "file":
            blockers.append(f"server tree contains non-regular member: {item!r}")
            continue
        name, size, payload_size = item.get("path"), item.get("size"), item.get("lfs_payload_size")
        if not isinstance(name, str) or not isinstance(size, int) or isinstance(size, bool) or size < 0:
            blockers.append(f"server tree member has invalid path/size: {item!r}")
            continue
        if payload_size is not None and (not isinstance(payload_size, int) or isinstance(payload_size, bool) or payload_size < 0):
            blockers.append(f"server tree member has invalid LFS payload size: {item!r}")
            continue
        head_commit, head_size, head_etag = item.get("head_commit"), item.get("head_size"), item.get("head_etag")
        if head_commit != HF_REVISION or head_size != size or not isinstance(head_etag, str):
            blockers.append(f"server tree HEAD metadata mismatch: {name}")
        if not name or "\x00" in name or "\\" in name or name.startswith("/") or any(part in ("", ".", "..") for part in Path(name).parts):
            blockers.append(f"server tree member has unsafe path: {name!r}")
            continue
        git_sha, pointer_sha, lfs_sha = item.get("git_blob_sha1"), item.get("lfs_pointer_git_blob_sha1"), item.get("lfs_sha256")
        if git_sha is not None and (not isinstance(git_sha,str) or not re.fullmatch(r"[0-9a-f]{40}",git_sha)): blockers.append(f"server tree has invalid Git blob SHA1: {name}")
        if pointer_sha is not None and (not isinstance(pointer_sha,str) or not re.fullmatch(r"[0-9a-f]{40}",pointer_sha)): blockers.append(f"server tree has invalid LFS pointer Git blob SHA1: {name}")
        if lfs_sha is not None and (not isinstance(lfs_sha,str) or not re.fullmatch(r"[0-9a-f]{64}",lfs_sha)): blockers.append(f"server tree has invalid LFS SHA256: {name}")
        if (lfs_sha is None and (git_sha is None or pointer_sha is not None or payload_size is not None)) or (lfs_sha is not None and (git_sha is not None or pointer_sha is None or payload_size != size)): blockers.append(f"server tree regular/LFS identities are not distinct: {name}")
        if name in LFS_ARTIFACTS:
            expected_size, expected_sha = LFS_ARTIFACTS[name]
            if size != expected_size or lfs_sha != expected_sha or pointer_sha is None:
                blockers.append(f"fixed LFS artifact identity/classification mismatch: {name}")
        if name in remote_records: blockers.append(f"duplicate server tree member: {name}")
        remote_records[name]={"bytes":size,"lfs_payload_size":payload_size,"head_commit":head_commit,"head_size":head_size,"head_etag":head_etag,"git_blob_sha1":git_sha,"lfs_pointer_git_blob_sha1":pointer_sha,"lfs_sha256":lfs_sha}
    local = files(snapshot); local_records = {path.relative_to(snapshot).as_posix(): identity(path,snapshot) for path in local}
    missing, extra = sorted(set(remote_records)-set(local_records)), sorted(set(local_records)-set(remote_records))
    mismatched=[]
    for name in sorted(set(remote_records)&set(local_records)):
        expected, actual=remote_records[name], local_records[name]
        mismatch = (actual["bytes"] != expected["bytes"] or expected["head_commit"] != HF_REVISION
                    or expected["head_size"] != expected["bytes"] or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", expected["head_etag"]))
        if expected["lfs_sha256"] is None:
            mismatch = mismatch or git_blob_sha1(Path(snapshot) / name) != expected["git_blob_sha1"] or expected["head_etag"] != expected["git_blob_sha1"]
        else:
            mismatch = mismatch or actual["sha256"] != expected["lfs_sha256"] or expected["head_etag"] != expected["lfs_sha256"] or expected["head_size"] != expected["bytes"]
            pointer = git_blob_sha1_bytes(lfs_pointer_bytes(actual["sha256"], actual["bytes"]))
            mismatch = mismatch or pointer != expected["lfs_pointer_git_blob_sha1"]
        remote_records[name].update({"payload_bytes": actual["bytes"], "payload_sha256": actual["sha256"]})
        if mismatch: mismatched.append(name)
    identity_match=(remote.get("repository") == HF_REPOSITORY and remote.get("requested_revision") == HF_REVISION and remote.get("resolved_revision") == HF_REVISION and remote.get("head_commit") == HF_REVISION and remote.get("walk") == "recursive_file_only" and isinstance(remote.get("files"), list))
    if not identity_match: blockers.append("server tree identity/walk mismatch")
    if missing or extra: blockers.append(f"server/local tree mismatch: missing={missing!r} extra={extra!r}")
    if mismatched: blockers.append(f"server/local content identity mismatch: {mismatched!r}")
    return {"status": "MATCHED" if schema_ok and identity_match and not missing and not extra and not mismatched else "MISMATCH", "repository": remote.get("repository"), "requested_revision": remote.get("requested_revision"), "resolved_revision":remote.get("resolved_revision"), "head_commit":remote.get("head_commit"), "walk":remote.get("walk"), "packet_sha256": sha256(packet), "files":remote_records,"missing": missing, "extra": extra, "content_mismatch":mismatched}

def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result: raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

def json_packet(path: Path, root: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)
    return {**identity(path, root), "status": "PARSED_CANONICAL_JSON", "top_level_type": type(value).__name__, "top_level_keys": sorted(value) if isinstance(value, dict) else None, "raw": value if path.name.endswith("index.json") else value if path.name == "config.json" else None}

def safe_header(path: Path, root: Path, blockers: list[str]) -> dict[str, Any]:
    item = identity(path, root)
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8: raise ValueError("short header length")
            header_len = struct.unpack("<Q", prefix)[0]
            if header_len == 0 or header_len > max(0, size - 8) or header_len > 64 * 1024 * 1024: raise ValueError("unsafe header length")
            header = json.loads(stream.read(header_len), object_pairs_hook=_no_duplicate_keys)
    except Exception as error: blockers.append(f"safetensors header blocked {path}: {error}"); return {**item, "status": "BLOCKED_HEADER", "error": str(error)}
    if not isinstance(header, dict): blockers.append(f"header is not an object: {path}"); return {**item, "status": "BLOCKED_HEADER"}
    metadata = header.get("__metadata__", {})
    header_blocked=False
    if not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()): blockers.append(f"metadata is not string map: {path}"); header_blocked=True
    data_start = 8 + header_len; ranges=[]; tensors=[]
    for name, spec in sorted(header.items()):
        if name == "__metadata__": continue
        try:
            if not isinstance(name,str) or not name or "\x00" in name or "\\" in name or name.startswith("/") or any(part in ("", ".", "..") for part in Path(name).parts): raise ValueError("unsafe tensor name")
            if not isinstance(spec, dict) or set(spec) != {"dtype","shape","data_offsets"}: raise ValueError("strict tensor descriptor keys required")
            shape = spec["shape"]; offsets = spec["data_offsets"]; dtype = spec["dtype"]
            if not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2 or not isinstance(dtype, str) or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in shape+offsets): raise ValueError("invalid shape/offset types")
            start, end = offsets; elements=1
            for axis in shape: elements *= axis
            if dtype not in DTYPE_BYTES or end < start or end-start != elements*DTYPE_BYTES[dtype] or data_start+end > size: raise ValueError("invalid byte range")
            ranges.append((start,end,name)); tensors.append({"name":name,"shape":shape,"dtype":dtype,"elements":elements,"data_offsets":offsets,"finite":"NOT_CHECKED_HEADER_ONLY"})
        except Exception as error: blockers.append(f"invalid tensor header {path}:{name}: {error}"); header_blocked=True
    cursor=0
    for start,end,name in sorted(ranges):
        if start < cursor: blockers.append(f"overlap in tensor ranges: {path}:{name}"); header_blocked=True
        if start > cursor: blockers.append(f"gap in tensor ranges: {path}:{name}"); header_blocked=True
        cursor=max(cursor,end)
    if cursor != size-data_start: blockers.append(f"tensor data does not end at file boundary: {path}"); header_blocked=True
    item.update({"status":"BLOCKED_HEADER" if header_blocked else "HEADER_ONLY","header_bytes":header_len,"metadata":metadata,"tensor_count":len(tensors),"tensors":tensors,"resident_scope":"header-only; body never read"}); return item

def parse_sigstore_verification_material(value: Any, blockers: list[str]) -> dict[str, Any]:
    """Parse the v0.3 verification-material oneof without doing crypto.

    Sigstore's protobuf JSON represents the certificate/public-key oneof as
    either ``certificate`` or ``publicKey``.  The old collector required the
    certificate arm unconditionally, which rejected valid public-key bundles.
    Unknown top-level or nested fields remain a hard structural blocker.
    """
    if not isinstance(value, dict):
        blockers.append("model.sig verificationMaterial is not an object")
        return {"status": "BLOCKED_UNKNOWN_SCHEMA"}
    keys = set(value)
    arms = [key for key in ("certificate", "publicKey") if key in value]
    allowed_keys = {arms[0], "tlogEntries", "timestampVerificationData"} if len(arms) == 1 else set()
    if len(arms) != 1 or not keys <= allowed_keys or "tlogEntries" not in keys:
        blockers.append("model.sig verificationMaterial unknown schema")
        return {"status": "BLOCKED_UNKNOWN_SCHEMA", "keys": sorted(keys)}
    arm = arms[0]
    material = value[arm]
    if not isinstance(material, dict) or set(material) != {"rawBytes"}:
        blockers.append(f"model.sig verificationMaterial.{arm} unknown schema")
        return {"status": "BLOCKED_UNKNOWN_SCHEMA", "keys": sorted(keys), "arm": arm}
    raw_bytes = material["rawBytes"]
    try:
        decoded = base64.b64decode(raw_bytes, validate=True)
    except Exception:
        decoded = b""
    if not isinstance(raw_bytes, str) or not decoded:
        blockers.append(f"model.sig verificationMaterial.{arm}.rawBytes is not nonempty base64")
    entries = value["tlogEntries"]
    if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
        blockers.append("model.sig verificationMaterial.tlogEntries is not an object array")
        entries = []
    entry_inventory = []
    allowed_entry_keys = {
        "logIndex", "logId", "kindVersion", "integratedTime",
        "inclusionPromise", "inclusionProof", "canonicalizedBody",
    }
    for entry in entries:
        unknown = set(entry) - allowed_entry_keys
        if unknown:
            blockers.append(f"model.sig tlog entry unknown schema keys: {sorted(unknown)!r}")
        if not isinstance(entry.get("logId"), dict) or set(entry["logId"]) != {"keyId"} or not isinstance(entry["logId"].get("keyId"), str):
            blockers.append("model.sig tlog entry logId schema mismatch")
        if not isinstance(entry.get("kindVersion"), dict) or set(entry["kindVersion"]) != {"kind", "version"} or any(not isinstance(entry["kindVersion"].get(field), str) for field in ("kind", "version")):
            blockers.append("model.sig tlog entry kindVersion schema mismatch")
        for field in ("logIndex", "integratedTime", "canonicalizedBody"):
            if not isinstance(entry.get(field), str):
                blockers.append(f"model.sig tlog entry {field} schema mismatch")
        for field in ("inclusionPromise", "inclusionProof"):
            nested = entry.get(field)
            if nested is not None and not isinstance(nested, dict):
                blockers.append(f"model.sig tlog entry {field} schema mismatch")
        promise = entry.get("inclusionPromise")
        if promise is not None and (set(promise) != {"signedEntryTimestamp"} or not isinstance(promise.get("signedEntryTimestamp"), str)):
            blockers.append("model.sig tlog entry inclusionPromise schema mismatch")
        proof = entry.get("inclusionProof")
        if proof is not None:
            proof_keys = {"checkpoint", "logIndex", "rootHash", "treeSize", "hashes"}
            checkpoint = proof.get("checkpoint")
            if (set(proof) != proof_keys
                    or not isinstance(checkpoint, dict)
                    or set(checkpoint) != {"envelope"}
                    or not isinstance(checkpoint.get("envelope"), str)
                    or not checkpoint["envelope"]
                    or any(not isinstance(proof.get(field), str) for field in ("logIndex", "rootHash", "treeSize"))
                    or not isinstance(proof.get("hashes"), list)
                    or any(not isinstance(item, str) for item in proof["hashes"])
                ):
                blockers.append("model.sig tlog entry inclusionProof schema mismatch")
        entry_inventory.append({"keys": sorted(entry), "status": "STRUCTURE_PARSED"})
    timestamp_data = value.get("timestampVerificationData")
    timestamp_inventory = None
    if timestamp_data is not None:
        if not isinstance(timestamp_data, dict) or set(timestamp_data) != {"rfc3161Timestamps"} or not isinstance(timestamp_data["rfc3161Timestamps"], list):
            blockers.append("model.sig timestampVerificationData unknown schema")
        else:
            timestamp_inventory = []
            for timestamp in timestamp_data["rfc3161Timestamps"]:
                if not isinstance(timestamp, dict) or set(timestamp) != {"signedTimestamp"} or not isinstance(timestamp["signedTimestamp"], str):
                    blockers.append("model.sig RFC3161 timestamp schema mismatch")
                else:
                    timestamp_inventory.append({"keys": sorted(timestamp), "status": "STRUCTURE_PARSED"})
    return {
        "status": "STRUCTURE_PARSED",
        "arm": arm,
        "keys": sorted(keys),
        "raw_bytes": "PRESENT_BASE64",
        "tlog_entry_count": len(entries),
        "tlog_entries": entry_inventory,
        "rfc3161_timestamps": timestamp_inventory,
    }

def sigstore_evidence(path: Path, snapshot: Path, files: list[Path], blockers: list[str]) -> dict[str, Any]:
    item=identity(path,snapshot)
    try:
        value=json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)
        if value.get("mediaType") != "application/vnd.dev.sigstore.bundle.v0.3+json": blockers.append("model.sig mediaType mismatch")
        envelope=value["dsseEnvelope"]
        if envelope.get("payloadType") != "application/vnd.in-toto+json": blockers.append("model.sig DSSE payloadType mismatch")
        payload=envelope["payload"]
        decoded=json.loads(base64.b64decode(payload, validate=True), object_pairs_hook=_no_duplicate_keys)
        payload_type=decoded.get("_type") if isinstance(decoded,dict) else None
        if payload_type != "https://in-toto.io/Statement/v1": blockers.append("model.sig in-toto _type mismatch")
        if decoded.get("predicateType") != "https://model_signing/signature/v1.0": blockers.append("model.sig predicateType mismatch")
        predicate=decoded.get("predicate")
        serialization=predicate.get("serialization") if isinstance(predicate,dict) else None
        if not isinstance(serialization,dict) or set(serialization) != {"hash_type","method","allow_symlinks","ignore_paths"}: blockers.append("model.sig predicate.serialization structure mismatch")
        elif (serialization.get("hash_type") != "sha256" or serialization.get("method") != "files" or serialization.get("allow_symlinks") is not False or not isinstance(serialization.get("ignore_paths"),list) or any(not isinstance(item,str) for item in serialization["ignore_paths"]) or len(serialization["ignore_paths"]) != len(set(serialization["ignore_paths"])) or set(serialization["ignore_paths"]) != set(SIGSTORE_IGNORE_PATHS)): blockers.append("model.sig predicate.serialization values mismatch")
        resources=predicate.get("resources") if isinstance(predicate,dict) else None
        if not isinstance(resources,list) or not resources: blockers.append("model.sig predicate.resources missing/nonempty requirement"); resources=[]
        paths=set(); resource_hashes={}
        for resource in resources:
            name = resource.get("name") or resource.get("uri") if isinstance(resource, dict) else None
            if not isinstance(name, str) or not name or "\x00" in name or "\\" in name or name.startswith("/") or any(part in ("", ".", "..") for part in Path(name).parts):
                blockers.append(f"model.sig contains unsafe resource path: {name!r}")
                continue
            if name in paths: blockers.append(f"model.sig contains duplicate resource: {name}")
            paths.add(name)
            algorithm=resource.get("algorithm") if isinstance(resource,dict) else None
            digest=resource.get("digest") if isinstance(resource,dict) else None
            if not isinstance(resource,dict) or set(resource) != {"name","algorithm","digest"}: blockers.append(f"model.sig resource structure mismatch: {name}")
            if algorithm != "sha256": blockers.append(f"model.sig resource algorithm is not sha256: {name}")
            if not isinstance(digest,str) or not re.fullmatch(r"[0-9a-fA-F]{64}",digest): blockers.append(f"model.sig resource has no SHA256: {name}")
            else: resource_hashes[name]=digest.lower()
        ignored=set(SIGSTORE_IGNORE_PATHS)
        local={p.relative_to(snapshot).as_posix() for p in files if p.name not in ignored and not any(part in {".cache",".git",".github"} for part in p.relative_to(snapshot).parts)}
        if paths != local: blockers.append("model.sig resource set mismatch")
        for member in files:
            name=member.relative_to(snapshot).as_posix()
            expected=resource_hashes.get(name)
            if expected is not None and sha256(member).lower() != expected: blockers.append(f"model.sig resource SHA256 mismatch: {name}")
        signatures=envelope.get("signatures",[])
        if not isinstance(signatures,list) or not signatures: blockers.append("model.sig DSSE signatures missing")
        signature_shapes=[]
        for signature in signatures if isinstance(signatures,list) else []:
            encoded=signature.get("sig") if isinstance(signature,dict) else None
            try: base64.b64decode(encoded,validate=True)
            except Exception: blockers.append("model.sig signature is not valid base64"); continue
            signature_shapes.append({"keys":sorted(signature),"base64":"STRUCTURALLY_VALID"})
        verification=parse_sigstore_verification_material(value.get("verificationMaterial"), blockers)
        item.update({"status":"STRUCTURE_PARSED_CRYPTO_NOT_VERIFIED","payload_type":payload_type,"predicate_type":decoded.get("predicateType"),"serialization_keys":sorted(serialization) if isinstance(serialization,dict) else [],"resource_paths":sorted(paths),"resource_sha256":resource_hashes,"signature_inventory":signature_shapes,"verification_material":verification})
    except Exception as error: blockers.append(f"model.sig parse failed: {error}"); item.update({"status":"BLOCKED_SIGSTORE_PARSE","error":str(error)})
    blockers.append("SIGSTORE_CRYPTOGRAPHIC_VERIFICATION_NOT_PERFORMED")
    return item

def inspection_status(collection_status: str) -> str:
    """Only a complete authenticated collection may use the normal marker."""
    return "AUTHENTICATED_EVIDENCE_COMPLETE" if collection_status == "AUTHENTICATED" else "INSPECTION_ERROR"

def inspect(snapshot: Path, source: Path, transformers_source: Path, output: Path, tree: Path) -> int:
    blockers=[]; local=files(snapshot); tree_packet=server_tree(snapshot,tree,blockers)
    index_paths=[p for p in local if p.name=="model.safetensors.index.json"]
    if len(index_paths) != 1: blockers.append("exactly one model.safetensors.index.json is required")
    index={}
    if len(index_paths)==1:
        try: index=json.loads(index_paths[0].read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)
        except Exception as error: blockers.append(f"index JSON blocked: {error}")
        if index_paths[0].stat().st_size != EXPECTED_INDEX_BYTES: blockers.append("index byte size mismatch")
    weight_map=index.get("weight_map",{}) if isinstance(index,dict) else {}
    if not isinstance(weight_map,dict) or not weight_map: blockers.append("index weight_map missing/nonempty requirement")
    for key, value in weight_map.items() if isinstance(weight_map, dict) else []:
        if not isinstance(key, str) or not safe_direct_basename(value, set(SHARDS)):
            blockers.append(f"index has unsafe tensor/shard mapping: {key!r} -> {value!r}")
    shard_names={v for v in weight_map.values() if isinstance(v,str)} if isinstance(weight_map,dict) else set()
    if shard_names != set(SHARDS): blockers.append(f"main shard set mismatch: {sorted(shard_names)!r}")
    packets=[safe_header(snapshot/name,snapshot,blockers) for name in sorted(SHARDS) if (snapshot/name).is_file()]
    packet_names={packet.get("path"): {tensor["name"] for tensor in packet.get("tensors",[])} for packet in packets}
    seen_list=[tensor["name"] for packet in packets for tensor in packet.get("tensors",[])]
    seen=set(seen_list)
    if len(seen_list) != len(seen): blockers.append("duplicate tensor name across main shard headers")
    if set(weight_map) != seen: blockers.append("index tensor keys do not equal header tensor keys")
    for name, (expected_size, expected_sha) in {**SHARDS, **AUXILIARY}.items():
        path=snapshot/name
        if not path.is_file(): blockers.append(f"required artifact missing: {name}"); continue
        actual=identity(path,snapshot)
        if (actual["bytes"], actual["sha256"]) != (expected_size, expected_sha): blockers.append(f"fixed artifact identity mismatch: {name}")
    for tensor, shard in weight_map.items() if isinstance(weight_map,dict) else []:
        if isinstance(shard,str) and tensor not in packet_names.get(shard, set()): blockers.append(f"index mapping points tensor to wrong shard: {tensor} -> {shard}")
    expected_counts=list(EXPECTED_MAIN_COUNTS)
    observed_counts=[packet.get("tensor_count") for packet in packets]
    if observed_counts != expected_counts: blockers.append(f"main shard tensor counts mismatch: {observed_counts!r}")
    if len(seen) != EXPECTED_MAIN_TENSORS: blockers.append(f"expected {EXPECTED_MAIN_TENSORS} main tensors, observed {len(seen)}")
    metadata=index.get("metadata",{}) if isinstance(index,dict) else {}
    if not isinstance(metadata,dict) or metadata.get("total_size") != 4_626_414_392: blockers.append("index metadata.total_size mismatch")
    auxiliary=[safe_header(snapshot/name,snapshot,blockers) for name in AUXILIARY if (snapshot/name).is_file()]
    sig=sigstore_evidence(snapshot/"model.sig",snapshot,local,blockers) if (snapshot/"model.sig").is_file() else None
    if sig is None: blockers.append("model.sig is missing")
    jsons=[]
    for path in local:
        if path.suffix==".json":
            try: jsons.append(json_packet(path,snapshot))
            except Exception as error: blockers.append(f"duplicate/malformed JSON: {path}: {error}")
    source_inventory={"repository":SOURCE_REPOSITORY,"pinned_revision":SOURCE_REVISION,"tracked_files":[],"required_roles":list(SOURCE_ROLE_FILES),"role_blob_table":SOURCE_ROLE_BLOBS,"role_blob_table_status":"AUTHENTICATED" if set(SOURCE_ROLE_BLOBS)==set(SOURCE_ROLE_FILES) else "BLOCKED_UNAVAILABLE","transformers_source_inventory":{"repository":TRANSFORMERS_REPOSITORY,"pinned_revision":TRANSFORMERS_REVISION,"required_roles":list(TRANSFORMERS_ROLE_FILES),"role_blob_table":TRANSFORMERS_ROLE_BLOBS,"role_blob_table_status":"AUTHENTICATED" if set(TRANSFORMERS_ROLE_BLOBS)==set(TRANSFORMERS_ROLE_FILES) else "BLOCKED_UNAVAILABLE","status":"NOT_SUPPLIED_BLOCKER"}}
    try:
        actual=subprocess.run(["git","-C",str(source),"rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip(); origin=subprocess.run(["git","-C",str(source),"remote","get-url","origin"],check=True,capture_output=True,text=True).stdout.strip(); names=subprocess.run(["git","-C",str(source),"ls-files","-s","-z"],check=True,capture_output=True).stdout.split(b"\0")
        tags=subprocess.run(["git","-C",str(source),"tag","--points-at",actual],check=True,capture_output=True,text=True).stdout.splitlines()
        dirty=subprocess.run(["git","-C",str(source),"status","--porcelain","--untracked-files=all"],check=True,capture_output=True,text=True).stdout.strip()
        if actual!=SOURCE_REVISION or origin!=SOURCE_REPOSITORY: blockers.append("source identity/origin mismatch")
        if dirty: blockers.append(f"source checkout is dirty: {dirty}")
        paths=[]; gitlinks=[]; symlinks=[]
        for record in names:
            if not record: continue
            header, rel = record.split(b"\t",1); fields=header.split(); mode=fields[0].decode(); object_id=fields[1].decode() if len(fields)>1 else ""
            path=source/os.fsdecode(rel)
            if mode == "160000": gitlinks.append({"path":path.relative_to(source).as_posix(),"object_id":object_id,"status":"GITLINK_NOT_CHECKED_OUT"}); blockers.append(f"source gitlink not checked out: {path}")
            elif mode == "120000":
                relative_path = path.relative_to(source).as_posix()
                record = tracked_symlink(source, relative_path, object_id, blockers)
                if record is not None: symlinks.append(record)
            elif mode not in ("100644","100755"): blockers.append(f"source tracked non-regular member: {path}")
            elif path.is_file(): paths.append(path)
            else: blockers.append(f"source tracked file missing: {path}")
        license_paths=[p for p in sorted(source.glob("LICENSE*")) if p.is_file()]
        if not license_paths: blockers.append("source license file missing")
        if [p.relative_to(source).as_posix() for p in license_paths] != ["LICENSE"]: blockers.append("source LICENSE role set is not exactly LICENSE")
        role_records={}
        if set(SOURCE_ROLE_BLOBS)!=set(SOURCE_ROLE_FILES): blockers.append("IBM source role Git blob table is incomplete or has extra roles")
        for role in SOURCE_ROLE_FILES:
            path=source/role; expected=SOURCE_ROLE_BLOBS.get(role)
            if not path.is_file() or path.is_symlink(): blockers.append(f"IBM source role missing/nonregular: {role}"); continue
            actual_blob=git_blob_sha1(path)
            indexed_blob=subprocess.run(["git","-C",str(source),"rev-parse",f"HEAD:{role}"],check=True,capture_output=True,text=True).stdout.strip()
            if expected is None: blockers.append(f"IBM source role Git blob is unavailable: {role}")
            elif actual_blob!=expected or indexed_blob!=expected: blockers.append(f"IBM source role Git blob mismatch: {role}")
            role_records[role]={**identity(path,source),"git_blob_sha1":actual_blob,"indexed_git_blob_sha1":indexed_blob,"expected_git_blob_sha1":expected}
        license_records=[]
        for license_path in license_paths:
            license_blob=subprocess.run(["git","-C",str(source),"rev-parse",f"HEAD:{license_path.relative_to(source).as_posix()}"],check=True,capture_output=True,text=True).stdout.strip()
            license_records.append({**identity(license_path,source),"git_blob_sha1":git_blob_sha1(license_path),"indexed_git_blob_sha1":license_blob,"expected_git_blob_sha1":SOURCE_LICENSE_BLOB})
            if license_blob!=SOURCE_LICENSE_BLOB or git_blob_sha1(license_path)!=SOURCE_LICENSE_BLOB: blockers.append("IBM source LICENSE Git blob mismatch")
        source_inventory.update({"resolved_revision":actual,"origin":origin,"clean":not bool(dirty),"tags_at_revision":tags,"tracked_files":[identity(p,source,True) for p in sorted(paths)],"symlinks":symlinks,"required_role_files":role_records,"gitlinks":gitlinks,"license_files":license_records,"license_status":"DECLARATION_REQUIRES_PRIMARY_REVIEW"})
    except Exception as error: blockers.append(f"source inventory failed: {error}")
    try:
        actual=subprocess.run(["git","-C",str(transformers_source),"rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip()
        origin=subprocess.run(["git","-C",str(transformers_source),"remote","get-url","origin"],check=True,capture_output=True,text=True).stdout.strip()
        tags=subprocess.run(["git","-C",str(transformers_source),"tag","--points-at",actual],check=True,capture_output=True,text=True).stdout.splitlines()
        transformer_modes=subprocess.run(["git","-C",str(transformers_source),"ls-files","-s","-z"],check=True,capture_output=True).stdout.split(b"\0")
        transformer_gitlinks=[]; transformer_symlinks=[]
        for record in transformer_modes:
            if not record: continue
            header, raw_path = record.split(b"\t",1)
            fields = header.split(); mode=fields[0].decode(); object_id=fields[1].decode() if len(fields) > 1 else ""
            relative_path = raw_path.decode(errors="strict")
            if mode == "120000":
                record = tracked_symlink(transformers_source, relative_path, object_id, blockers)
                if record is not None: transformer_symlinks.append(record)
            elif mode not in ("100644","100755"):
                transformer_gitlinks.append(relative_path); blockers.append(f"Transformers tracked non-regular member: {transformer_gitlinks[-1]}")
        dirty=subprocess.run(["git","-C",str(transformers_source),"status","--porcelain","--untracked-files=all"],check=True,capture_output=True,text=True).stdout.strip()
        if actual!=TRANSFORMERS_REVISION or origin!=TRANSFORMERS_REPOSITORY: blockers.append("Transformers source identity/origin mismatch")
        if dirty: blockers.append(f"Transformers checkout is dirty: {dirty}")
        role_records=[]
        for role in TRANSFORMERS_ROLE_FILES:
            path=transformers_source/role
            if not path.is_file() or path.is_symlink(): blockers.append(f"Transformers role file missing/nonregular: {role}"); continue
            expected=TRANSFORMERS_ROLE_BLOBS.get(role)
            actual_blob=git_blob_sha1(path)
            indexed_blob=subprocess.run(["git","-C",str(transformers_source),"rev-parse",f"HEAD:{role}"],check=True,capture_output=True,text=True).stdout.strip()
            if expected is None: blockers.append(f"Transformers role Git blob is unavailable: {role}")
            elif actual_blob != expected or indexed_blob != expected: blockers.append(f"Transformers role Git blob mismatch: {role}")
            role_records.append({**identity(path,transformers_source),"git_blob_sha1":actual_blob,"indexed_git_blob_sha1":indexed_blob,"expected_git_blob_sha1":expected})
        licenses=[p for p in sorted(transformers_source.glob("LICENSE*")) if p.is_file() and not p.is_symlink()]
        if not licenses: blockers.append("Transformers source license file missing")
        if [p.relative_to(transformers_source).as_posix() for p in licenses] != ["LICENSE"]: blockers.append("Transformers LICENSE role set is not exactly LICENSE")
        license_records=[]
        for license_path in licenses:
            license_blob=subprocess.run(["git","-C",str(transformers_source),"rev-parse",f"HEAD:{license_path.relative_to(transformers_source).as_posix()}"],check=True,capture_output=True,text=True).stdout.strip()
            license_records.append({**identity(license_path,transformers_source),"git_blob_sha1":git_blob_sha1(license_path),"indexed_git_blob_sha1":license_blob,"expected_git_blob_sha1":TRANSFORMERS_LICENSE_BLOB})
            if license_blob!=TRANSFORMERS_LICENSE_BLOB or git_blob_sha1(license_path)!=TRANSFORMERS_LICENSE_BLOB: blockers.append("Transformers LICENSE Git blob mismatch")
        source_inventory["transformers_source_inventory"]={"repository":TRANSFORMERS_REPOSITORY,"pinned_revision":TRANSFORMERS_REVISION,"resolved_revision":actual,"origin":origin,"clean":not bool(dirty),"tags_at_revision":tags,"gitlinks":transformer_gitlinks,"symlinks":transformer_symlinks,"role_files":role_records,"license_files":license_records,"status":"ROLE_HASHES_AUTHENTICATED" if len(role_records)==len(TRANSFORMERS_ROLE_FILES) and set(TRANSFORMERS_ROLE_BLOBS)==set(TRANSFORMERS_ROLE_FILES) and not dirty and not transformer_gitlinks else "BLOCKED"}
    except Exception as error:
        blockers.append(f"Transformers source inventory failed: {error}")
    config_packets=[]
    config_paths=[p for p in local if p.name == "config.json"]
    if len(config_paths) == 1:
        try: config_packets.append(contract_packet(config_paths[0],snapshot,CONFIG_FACTS,blockers))
        except Exception as error: blockers.append(f"config contract parse failed: {error}")
    preprocessor_paths=[p for p in local if p.name == "preprocessor_config.json"]
    if len(preprocessor_paths) == 1:
        try: config_packets.append(contract_packet(preprocessor_paths[0],snapshot,{(key,):value for key,value in PREPROCESSOR_FACTS.items()},blockers))
        except Exception as error: blockers.append(f"preprocessor contract parse failed: {error}")
    if not config_packets: blockers.append("config.json is missing")
    model_license_files=[identity(p,snapshot) for p in local if p.name.upper() in {"LICENSE","LICENSE.TXT","LICENSE.MD","README.MD"}]
    collection_blockers = [item for item in blockers if not item.startswith("SIGSTORE_CRYPTOGRAPHIC_VERIFICATION_NOT_PERFORMED")]
    source_ok = source_inventory.get("license_files") and source_inventory.get("required_role_files")
    transformers_ok = source_inventory.get("transformers_source_inventory", {}).get("status") == "ROLE_HASHES_AUTHENTICATED"
    artifact_ok = not any(item.startswith(("required artifact missing", "fixed artifact identity mismatch", "main shard", "expected ", "index ", "duplicate tensor")) for item in collection_blockers)
    source_related_blocker = any(
        ("source" in item.lower() or "transformers" in item.lower())
        and "declaration requires primary-source review" not in item
        for item in collection_blockers
    )
    collection_status = "AUTHENTICATED" if tree_packet.get("status") == "MATCHED" and source_ok and transformers_ok and artifact_ok and not source_related_blocker and not collection_blockers else "UNVERIFIED"
    blockers += ["native Granite Speech composition/runtime is not implemented","model/weight Apache-2.0 declaration requires primary-source review","IBM wrapper source Apache-2.0 declaration requires primary-source review","dependency licenses are unreviewed","dataset/training provenance is unauthenticated"]
    payload={"format":FORMAT,"status":"BLOCKED","evidence_stage":"INSPECTION_ONLY","inspection_status":inspection_status(collection_status),"collection_status":collection_status,"runtime_status":"NOT_IMPLEMENTED_FAIL_CLOSED","cpu_status":"UNSUPPORTED","metal_status":"BLOCKED_BY_CPU","parity_status":"NOT_RUN","publication":"NO_UPLOAD","model":{"repository":HF_REPOSITORY,"requested_revision":HF_REVISION,"server_tree":tree_packet,"files":[identity(p,snapshot) for p in local],"expected_artifacts":{"main_shards":SHARDS,"auxiliary_out_llm":AUXILIARY,"index_bytes":EXPECTED_INDEX_BYTES,"index_total_size":4_626_414_392,"main_tensor_count":EXPECTED_MAIN_TENSORS,"main_shard_tensor_counts":list(EXPECTED_MAIN_COUNTS)},"main_shards":packets,"auxiliary_out_llm":auxiliary,"json":jsons,"config_evidence":config_packets,"model_sig":sig},"official_source":source_inventory,"license_evidence":{"model_card":"apache-2.0 declaration requires primary-source review","model_license_files":model_license_files,"source":"apache-2.0 declaration requires primary-source review","dependencies":"UNREVIEWED_BLOCKER","datasets":"UNAUTHENTICATED_BLOCKER"},"sigstore_crypto":{"status":"SIGSTORE_CRYPTOGRAPHIC_VERIFICATION_NOT_PERFORMED","dsse_pae":"NOT_VERIFIED","signature":"NOT_VERIFIED","fulcio_certificate_chain":"NOT_VERIFIED","fulcio_identity_validity":"NOT_VERIFIED","rekor_set":"NOT_VERIFIED","rekor_inclusion_proof":"NOT_VERIFIED","signed_resource_hash_set":"STRUCTURALLY_CHECKED_ONLY"},"blockers":sorted(set(blockers))}
    output.mkdir(parents=True,exist_ok=True); (output/"manifest.json").write_text(json.dumps(payload,sort_keys=True,indent=2)+"\n",encoding="utf-8"); return 2

def self_test() -> None:
    assert len(HF_REVISION)==len(SOURCE_REVISION)==len(TRANSFORMERS_REVISION)==40
    assert inspection_status("AUTHENTICATED") == "AUTHENTICATED_EVIDENCE_COMPLETE"
    assert inspection_status("UNVERIFIED") == "INSPECTION_ERROR"
    assert safe_direct_basename("model-00001-of-00003.safetensors", set(SHARDS))
    assert not safe_direct_basename("../model-00001-of-00003.safetensors", set(SHARDS))
    assert not safe_direct_basename("/model-00001-of-00003.safetensors", set(SHARDS))
    with tempfile.TemporaryDirectory(prefix="granite-inspect-") as d:
        root=Path(d); path=root/"x.safetensors"; header=json.dumps({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}).encode(); path.write_bytes(struct.pack("<Q",len(header))+header+b"\0"*4); blockers=[]; assert safe_header(path,root,blockers)["tensor_count"]==1 and not blockers
        transport_cache=root/".cache"/"huggingface"; transport_cache.mkdir(parents=True); (transport_cache/"transport.marker").write_text("ignored",encoding="utf-8"); assert path in files(root)
        unexpected_cache=root/"nested"/".cache"/"other"; unexpected_cache.mkdir(parents=True); (unexpected_cache/"x").write_text("reject",encoding="utf-8")
        try: files(root)
        except RuntimeError as error: assert "cache/control-tree" in str(error)
        else: raise AssertionError("non-transport cache accepted")
        (unexpected_cache / "x").unlink(); unexpected_cache.rmdir(); unexpected_cache.parent.rmdir(); unexpected_cache.parent.parent.rmdir()
        symlink=root/"payload-link"; symlink.symlink_to(path); bad_link=[]
        try: files(root)
        except RuntimeError as error: assert "symlink" in str(error)
        else: raise AssertionError("payload symlink accepted")
        symlink.unlink()
        symlink_tmp = tempfile.TemporaryDirectory(prefix="granite-symlink-")
        symlink_repo = Path(symlink_tmp.name)
        subprocess.run(["git", "init", "-q", str(symlink_repo)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(symlink_repo), "config", "user.email", "test@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(symlink_repo), "config", "user.name", "Granite self-test"], check=True)
        (symlink_repo / "README.md").write_text("ok\n", encoding="utf-8")
        (symlink_repo / "docs").mkdir()
        (symlink_repo / "docs" / "README.md").symlink_to("../README.md")
        subprocess.run(["git", "-C", str(symlink_repo), "add", "README.md", "docs/README.md"], check=True, capture_output=True)
        index_record = subprocess.run(["git", "-C", str(symlink_repo), "ls-files", "-s", "--", "docs/README.md"], check=True, capture_output=True, text=True).stdout.strip()
        index_fields, index_path = index_record.split("\t", 1)
        symlink_object = index_fields.split()[1]
        symlink_blockers: list[str] = []
        symlink_record = tracked_symlink(symlink_repo, index_path, symlink_object, symlink_blockers)
        assert symlink_record is not None and symlink_record["index_target"] == "../README.md" and not symlink_blockers
        (symlink_repo / "docs" / "README.md").unlink()
        (symlink_repo / "docs" / "README.md").symlink_to("../../outside")
        escaped_blockers: list[str] = []
        assert tracked_symlink(symlink_repo, index_path, symlink_object, escaped_blockers) is None and escaped_blockers
        symlink_tmp.cleanup()
        bool_header=json.dumps({"x":{"dtype":"F32","shape":[True],"data_offsets":[0,4]}}).encode(); (root/"bool.safetensors").write_bytes(struct.pack("<Q",len(bool_header))+bool_header+b"\0"*4); bad=[]; safe_header(root/"bool.safetensors",root,bad); assert bad
        unsafe_name=json.dumps({"../x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}).encode(); (root/"unsafe.safetensors").write_bytes(struct.pack("<Q",len(unsafe_name))+unsafe_name+b"\0"*4); bad=[]; safe_header(root/"unsafe.safetensors",root,bad); assert any("unsafe tensor name" in item for item in bad)
        duplicate_header=b'{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'; (root/"duplicate.safetensors").write_bytes(struct.pack("<Q",len(duplicate_header))+duplicate_header+b"\0"*4); bad=[]; safe_header(root/"duplicate.safetensors",root,bad); assert bad
        huge=root/"huge"; huge.write_bytes(struct.pack("<Q",65*1024*1024)+b"{}"); bad=[]; safe_header(huge,root,bad); assert bad
        duplicate=root/"duplicate.json"; duplicate.write_text('{"x":1,"x":2}',encoding="utf-8");
        try: json_packet(duplicate,root)
        except ValueError: pass
        else: raise AssertionError("duplicate JSON key accepted")
        config=root/"config.json"; config.write_text('{"x":{"y":3}}',encoding="utf-8"); config_blockers=[]; assert contract_packet(config,root,{("x","y"):3},config_blockers)["contract_status"] == "EXACT_FACTS_MATCHED" and not config_blockers
        config.write_text('{"x":{"y":4}}',encoding="utf-8"); config_blockers=[]; assert contract_packet(config,root,{("x","y"):3},config_blockers)["contract_status"] == "BLOCKED_FACTS" and config_blockers
        tree=root/"tree.json"; tree.write_text(json.dumps({"repository":HF_REPOSITORY,"requested_revision":HF_REVISION,"resolved_revision":HF_REVISION,"head_commit":HF_REVISION,"walk":"recursive_file_only","files":[{"path":"missing","type":"file","size":1,"lfs_payload_size":None,"git_blob_sha1":"0"*40,"lfs_pointer_git_blob_sha1":None,"lfs_sha256":None}]}),encoding="utf-8"); tree_blockers=[]; assert server_tree(root,tree,tree_blockers)["status"] == "MISMATCH" and tree_blockers
        unknown_tree=root/"unknown-tree.json"; unknown_tree.write_text(json.dumps({"repository":HF_REPOSITORY,"requested_revision":HF_REVISION,"resolved_revision":HF_REVISION,"head_commit":HF_REVISION,"walk":"recursive_file_only","files":[{"path":"x","type":"folder","size":4,"lfs_payload_size":None,"head_commit":HF_REVISION,"head_size":4,"head_etag":"0"*40,"git_blob_sha1":"0"*40,"lfs_pointer_git_blob_sha1":None,"lfs_sha256":None}]}),encoding="utf-8"); unknown_blockers=[]; assert server_tree(root,unknown_tree,unknown_blockers)["status"] == "MISMATCH" and any("non-regular" in item for item in unknown_blockers)
        snapshot=root/"snapshot"; snapshot.mkdir(); content=snapshot/"x"; content.write_bytes(b"abcd")
        def test_row(name: str, size: int, *, git: str | None, pointer: str | None, lfs: str | None) -> dict[str, Any]:
            return {"path":name,"type":"file","size":size,"lfs_payload_size":size if lfs is not None else None,"head_commit":HF_REVISION,"head_size":size,"head_etag":lfs if lfs is not None else git,"git_blob_sha1":git,"lfs_pointer_git_blob_sha1":pointer,"lfs_sha256":lfs}
        packet=root/"matching-tree.json"; packet.write_text(json.dumps({"repository":HF_REPOSITORY,"requested_revision":HF_REVISION,"resolved_revision":HF_REVISION,"head_commit":HF_REVISION,"walk":"recursive_file_only","files":[test_row("x",4,git=git_blob_sha1(content),pointer=None,lfs=None)]}),encoding="utf-8"); matching_blockers=[]; assert server_tree(snapshot,packet,matching_blockers)["status"] == "MATCHED" and not matching_blockers; content.write_bytes(b"abce"); mutated_blockers=[]; assert server_tree(snapshot,packet,mutated_blockers)["status"] == "MISMATCH" and any("content identity" in item for item in mutated_blockers)
        lfs_content=snapshot/"lfs"; lfs_content.write_bytes(b"lfs payload"); lfs_digest=sha256(lfs_content); pointer_digest=git_blob_sha1_bytes(lfs_pointer_bytes(lfs_digest,lfs_content.stat().st_size)); assert pointer_digest != git_blob_sha1(lfs_content)
        regular_row=test_row("x",4,git=git_blob_sha1(content),pointer=None,lfs=None); lfs_row=test_row("lfs",lfs_content.stat().st_size,git=None,pointer=pointer_digest,lfs=lfs_digest); lfs_packet=root/"lfs-tree.json"; lfs_packet.write_text(json.dumps({"repository":HF_REPOSITORY,"requested_revision":HF_REVISION,"resolved_revision":HF_REVISION,"head_commit":HF_REVISION,"walk":"recursive_file_only","files":[regular_row,lfs_row]}),encoding="utf-8"); lfs_blockers=[]; assert server_tree(snapshot,lfs_packet,lfs_blockers)["status"] == "MATCHED" and not lfs_blockers
        bad_head=json.loads(lfs_packet.read_text(encoding="utf-8")); bad_head["files"][0]["head_etag"]="0"*40; lfs_packet.write_text(json.dumps(bad_head),encoding="utf-8"); bad_head_blockers=[]; assert server_tree(snapshot,lfs_packet,bad_head_blockers)["status"] == "MISMATCH" and bad_head_blockers; lfs_packet.write_text(json.dumps({"repository":HF_REPOSITORY,"requested_revision":HF_REVISION,"resolved_revision":HF_REVISION,"head_commit":HF_REVISION,"walk":"recursive_file_only","files":[regular_row,lfs_row]}),encoding="utf-8")
        for field, value, phrase in (("lfs_pointer_git_blob_sha1","0"*40,"content identity"),("lfs_sha256","0"*64,"content identity"),("size",lfs_content.stat().st_size+1,"content identity")):
            spoof={"repository":HF_REPOSITORY,"requested_revision":HF_REVISION,"resolved_revision":HF_REVISION,"walk":"recursive_file_only","files":[dict(regular_row),dict(lfs_row)]}; spoof["files"][1][field]=value; spoof_path=root/f"spoof-{field}.json"; spoof_path.write_text(json.dumps(spoof),encoding="utf-8"); spoof_blockers=[]; assert server_tree(snapshot,spoof_path,spoof_blockers)["status"] == "MISMATCH" and any(phrase in item for item in spoof_blockers)
        identity_packet=root/"identity-tree.json"; identity_packet.write_text(json.dumps({"repository":HF_REPOSITORY,"requested_revision":"0"*40,"resolved_revision":"0"*40,"head_commit":"0"*40,"walk":"recursive_file_only","files":[regular_row,lfs_row]}),encoding="utf-8"); identity_blockers=[]; assert server_tree(snapshot,identity_packet,identity_blockers)["status"] == "MISMATCH" and any("identity/walk" in item for item in identity_blockers)
        top_level_spoof=root/"top-level-spoof.json"; top_level_spoof.write_text(json.dumps({"repository":HF_REPOSITORY,"requested_revision":HF_REVISION,"revision":HF_REVISION,"resolved_revision":HF_REVISION,"head_commit":HF_REVISION,"walk":"recursive_file_only","files":[regular_row,lfs_row]}),encoding="utf-8"); top_level_blockers=[]; assert server_tree(snapshot,top_level_spoof,top_level_blockers)["status"] == "MISMATCH" and any("top-level schema" in item for item in top_level_blockers)
        sig_snapshot=root/"sig-snapshot"; sig_snapshot.mkdir(); signed_file=sig_snapshot/"x.safetensors"; signed_file.write_bytes(path.read_bytes()); signed={"_type":"https://in-toto.io/Statement/v1","predicateType":"https://model_signing/signature/v1.0","predicate":{"serialization":{"hash_type":"sha256","method":"files","allow_symlinks":False,"ignore_paths":list(SIGSTORE_IGNORE_PATHS)},"resources":[{"name":"x.safetensors","algorithm":"sha256","digest":sha256(signed_file)}]}}
        payload=base64.b64encode(json.dumps(signed).encode()).decode(); sig=sig_snapshot/"model.sig"; sig.write_text(json.dumps({"mediaType":"application/vnd.dev.sigstore.bundle.v0.3+json","dsseEnvelope":{"payloadType":"application/vnd.in-toto+json","payload":payload,"signatures":[{"sig":base64.b64encode(b"sig").decode()}]},"verificationMaterial":{"certificate":{"rawBytes":base64.b64encode(b"cert").decode()},"tlogEntries":[],"timestampVerificationData":{"rfc3161Timestamps":[]}}}),encoding="utf-8"); sig_blockers=[]; evidence=sigstore_evidence(sig,sig_snapshot,[signed_file,sig],sig_blockers); assert evidence["payload_type"] == "https://in-toto.io/Statement/v1" and evidence["verification_material"]["status"] == "STRUCTURE_PARSED" and any("CRYPTOGRAPHIC_VERIFICATION_NOT_PERFORMED" in item for item in sig_blockers) and not any("resource set mismatch" in item for item in sig_blockers)
        sig.write_text(json.dumps({"mediaType":"application/vnd.dev.sigstore.bundle.v0.3+json","dsseEnvelope":{"payloadType":"application/vnd.in-toto+json","payload":payload,"signatures":[{"sig":base64.b64encode(b"sig").decode()}]},"verificationMaterial":{"publicKey":{"rawBytes":base64.b64encode(b"key").decode()},"tlogEntries":[]}}),encoding="utf-8"); public_key_blockers=[]; public_key_evidence=sigstore_evidence(sig,sig_snapshot,[signed_file,sig],public_key_blockers); assert public_key_evidence["verification_material"]["arm"] == "publicKey" and not any("unknown schema" in item for item in public_key_blockers)
        tlog_blockers=[]; tlog_evidence=parse_sigstore_verification_material({"certificate":{"rawBytes":base64.b64encode(b"cert").decode()},"tlogEntries":[{"logIndex":"1","logId":{"keyId":base64.b64encode(b"id").decode()},"kindVersion":{"kind":"hashedrekord","version":"0.0.1"},"integratedTime":"2","canonicalizedBody":base64.b64encode(b"body").decode(),"inclusionPromise":{"signedEntryTimestamp":base64.b64encode(b"set").decode()},"inclusionProof":{"checkpoint":{"envelope":"checkpoint"},"logIndex":"1","rootHash":base64.b64encode(b"root").decode(),"treeSize":"1","hashes":[]}}]},tlog_blockers); assert tlog_evidence["status"] == "STRUCTURE_PARSED" and not tlog_blockers
        sig.write_text(json.dumps({"mediaType":"application/vnd.dev.sigstore.bundle.v0.3+json","dsseEnvelope":{"payloadType":"application/vnd.in-toto+json","payload":payload,"signatures":[{"sig":base64.b64encode(b"sig").decode()}]},"verificationMaterial":{"certificate":{"rawBytes":base64.b64encode(b"cert").decode()},"tlogEntries":[],"unexpected":True}}),encoding="utf-8"); unknown_material_blockers=[]; sigstore_evidence(sig,sig_snapshot,[signed_file,sig],unknown_material_blockers); assert any("unknown schema" in item for item in unknown_material_blockers)
        signed["predicate"]["resources"][0]["name"]="../unsafe"; payload=base64.b64encode(json.dumps(signed).encode()).decode(); sig.write_text(json.dumps({"mediaType":"application/vnd.dev.sigstore.bundle.v0.3+json","dsseEnvelope":{"payloadType":"application/vnd.in-toto+json","payload":payload,"signatures":[]},"verificationMaterial":{}}),encoding="utf-8"); sig_blockers=[]; sigstore_evidence(sig,sig_snapshot,[signed_file,sig],sig_blockers); assert any("unsafe resource path" in item for item in sig_blockers)
        signed["predicate"]["resources"][0]["name"]="x.safetensors"; signed["predicate"]["serialization"]["hash_type"]="sha512"; payload=base64.b64encode(json.dumps(signed).encode()).decode(); sig.write_text(json.dumps({"mediaType":"application/vnd.dev.sigstore.bundle.v0.3+json","dsseEnvelope":{"payloadType":"application/vnd.in-toto+json","payload":payload,"signatures":[{"sig":base64.b64encode(b"sig").decode()}]},"verificationMaterial":{"certificate":{},"tlogEntries":[]}}),encoding="utf-8"); sig_blockers=[]; sigstore_evidence(sig,sig_snapshot,[signed_file,sig],sig_blockers); assert any("serialization values mismatch" in item for item in sig_blockers)
        signed["predicate"]["serialization"]["hash_type"]="sha256"; signed["predicate"]["serialization"]["ignore_paths"]=["model.sig"]; payload=base64.b64encode(json.dumps(signed).encode()).decode(); sig.write_text(json.dumps({"mediaType":"application/vnd.dev.sigstore.bundle.v0.3+json","dsseEnvelope":{"payloadType":"application/vnd.in-toto+json","payload":payload,"signatures":[{"sig":base64.b64encode(b"sig").decode()}]},"verificationMaterial":{}}),encoding="utf-8"); sig_blockers=[]; sigstore_evidence(sig,sig_snapshot,[signed_file,sig],sig_blockers); assert any("serialization values mismatch" in item for item in sig_blockers)
    print("granite_speech_4_1_2b_inspect self-test: OK")

def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--self-test",action="store_true"); parser.add_argument("--snapshot",type=Path); parser.add_argument("--source",type=Path); parser.add_argument("--transformers-source",type=Path); parser.add_argument("--server-tree",type=Path); parser.add_argument("--output",type=Path); args=parser.parse_args()
    if args.self_test:
        if any(v is not None for v in (args.snapshot,args.source,args.transformers_source,args.server_tree,args.output)): parser.error("--self-test accepts no other arguments")
        self_test(); return 0
    if any(v is None for v in (args.snapshot,args.source,args.transformers_source,args.server_tree,args.output)): parser.error("normal runs require snapshot/source/transformers-source/server-tree/output")
    try: return inspect(args.snapshot,args.source,args.transformers_source,args.output,args.server_tree)
    except Exception as error:
        args.output.mkdir(parents=True,exist_ok=True); (args.output/"manifest.json").write_text(json.dumps({"format":FORMAT,"status":"BLOCKED","evidence_stage":"INSPECTION_ONLY","inspection_status":"INSPECTION_ERROR","collection_status":"UNVERIFIED","runtime_status":"NOT_IMPLEMENTED_FAIL_CLOSED","cpu_status":"UNSUPPORTED","metal_status":"BLOCKED_BY_CPU","parity_status":"NOT_RUN","publication":"NO_UPLOAD","upstream":{"repository":HF_REPOSITORY,"requested_revision":HF_REVISION,"resolved_revision":None},"server_tree_packet":{"path":str(args.server_tree),"sha256":sha256(args.server_tree) if args.server_tree.is_file() else None},"error":str(error),"blockers":[str(error)]},indent=2)+"\n"); return 2
if __name__=="__main__": raise SystemExit(main())
