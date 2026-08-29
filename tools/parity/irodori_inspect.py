#!/usr/bin/env python3
"""VAST-only identity and format inspection for Irodori-TTS-500M-v3.

This is deliberately not a runtime or numerical oracle.  It authenticates
release/source/codec evidence and leaves the composite TTS route blocked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path
from typing import Any

MODEL_REPOSITORY = "Aratako/Irodori-TTS-500M-v3"
MODEL_REVISION = "236c1e56591279fc24e3c1bf6609fc06e48dde28"
MODEL_FILE = "model.safetensors"
MODEL_BYTES = 2_048_269_748
MODEL_SHA256 = "c4b8e7e982697664f829b7fb6bea307a25bd7ee013ad0d6114efc3e326acbd54"
SOURCE_REPOSITORY = "https://github.com/Aratako/Irodori-TTS.git"
# This is the immutable source used by the existing official text-block
# dumper.  It is an oracle pin, not a claim that the source is license-clean.
SOURCE_REVISION = "8224dafb46d0aba89209a8f905f1cb7e3299d9c1"
SOURCE_PYPROJECT_SHA256 = "a67e3494530cd9c29817507c67a496bb299a9a81e2edd4df6ffb80cf330dae71"
SOURCE_LOCK_SHA256 = "8175adbb9ad7ae77d1f048344343a63876e57c333b659314bcc054230b5b3e6c"
DACVAE_REVISION = "414c20785fc3a28373073ea8ef7a1316eeeaca6e"
CODEC_REPOSITORY = "Aratako/Semantic-DACVAE-Japanese-32dim"
CODEC_REVISION = "47376ee24834d7a05a48ebabfe3cde29b3c5e214"
# The model config names only this repository. This selected immutable
# snapshot is evidence for the adapted run, not an upstream source pin.
TOKENIZER_REPOSITORY = "llm-jp/llm-jp-3-150m"
TOKENIZER_REVISION = "b112feef602fff752e4dac4c30af6a2c2fa41c7a"
TOKENIZER_LICENSE = "Apache-2.0"
EXPECTED_TOKENIZER_PATHS = frozenset(
    {
        ".gitattributes",
        "README.md",
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
)
TOKENIZER_LOCAL_PATHS = EXPECTED_TOKENIZER_PATHS - {"model.safetensors"}
EXPECTED_CODEC_PATHS = frozenset({".gitattributes", "README.md", "weights.pth"})
PUBLIC_REPOSITORY = "vokra/irodori-tts-500m-v3"
PUBLIC_REVISION = "28e3efaf41f0890784d88f4744c34269e80bdd41"
PUBLIC_FILE = "irodori-tts-500m-v3.gguf"
PUBLIC_BYTES = 2_048_247_584
PUBLIC_SHA256 = "b64d970cf6a7b7cb81579147fa4b661761ee2c224c8da542926dc764fe04e09e"

# The release tree is intentionally explicit.  The API packet remains the
# authority for each small-file size/blob and for the LFS pointer identity.
EXPECTED_MODEL_PATHS = frozenset(
    {
        ".gitattributes",
        "EMOJI_ANNOTATIONS.md",
        "README.md",
        MODEL_FILE,
        "samples/clone_gen1.wav",
        "samples/clone_gen2.wav",
        "samples/clone_ref1.wav",
        "samples/clone_ref2.wav",
        "samples/emoji_sample1.wav",
        "samples/emoji_sample2.wav",
        "samples/emoji_sample3.wav",
        "samples/standard_sample1.wav",
        "samples/standard_sample2.wav",
    }
)
LFS_PAYLOADS = {
    MODEL_FILE: (MODEL_BYTES, MODEL_SHA256),
    "samples/clone_gen1.wav": (756558, "b660928f7416fbaeb80a9d79f7364988b77e87d779eb5db5aae9f0f3637856da"),
    "samples/clone_gen2.wav": (894798, "dad12f9ed444036e532fb29caa56b26dfca557597f880258ddb26a11b319a85b"),
    "samples/clone_ref1.wav": (729644, "99f6bedd737df9c5f7c831d8e471eaeaa59360efe771b33b0c45bfe5d47f6ac1"),
    "samples/clone_ref2.wav": (758564, "01f6baad852750e284f739527e7021c069a8313c30355d462fa3258397a9cc85"),
    "samples/emoji_sample1.wav": (1516878, "59812c149d04221d5bdcba798d61d9425c2fd99f50cdcd67a41c44db1a8d52b6"),
    "samples/emoji_sample2.wav": (625998, "0638ea1e1ba64e70ea42bff00dcb06ed8ce55bf806309d8b6e773aa5fb331db7"),
    "samples/emoji_sample3.wav": (1152078, "56f40efc8783af891d199aa9827416bd5d3a71b2529e15846e0d88dec7729228"),
    "samples/standard_sample1.wav": (956238, "1ae5a94cdb24d23170a01a33cda1a9b816a4d45c3e2be094a6a69f41210dd7b5"),
    "samples/standard_sample2.wav": (1969998, "02c240243a8e04239849b5a4e7d1c295997cb2fc7f03b0eece690aeafb776ca0"),
}
MAX_HEADER = 64 * 1024 * 1024
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
DTYPE_WIDTH = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
}

# This is an intentionally small, stdlib-auditable slice of the authenticated
# source lock.  It is not a replacement lock and must never be resolved or
# synchronized.  The route is the reason Irodori remains inspection-only:
# DACVAE -> descript-audiotools -> librosa -> soxr/soundfile -> cffi.
FORBIDDEN_DEPENDENCY_ROWS = {
    "dacvae": {
        "version": "1.0.0",
        "source": {"git": f"https://github.com/facebookresearch/dacvae#{DACVAE_REVISION}"},
        "dependencies": {"descript-audiotools"},
    },
    "descript-audiotools": {
        "version": "0.7.2",
        "source": {"registry": "https://pypi.org/simple"},
        "dependencies": {"librosa", "soundfile"},
    },
    "librosa": {
        "version": "0.11.0",
        "source": {"registry": "https://pypi.org/simple"},
        "dependencies": {"soxr", "soundfile"},
    },
    "soxr": {
        "version": "1.0.0",
        "source": {"registry": "https://pypi.org/simple"},
        "dependencies": set(),
    },
    "soundfile": {
        "version": "0.13.1",
        "source": {"registry": "https://pypi.org/simple"},
        "dependencies": {"cffi"},
    },
    "cffi": {
        "version": "2.0.0",
        "source": {"registry": "https://pypi.org/simple"},
        "dependencies": {"pycparser"},
    },
    "pycparser": {
        "version": "3.0",
        "source": {"registry": "https://pypi.org/simple"},
        "dependencies": set(),
    },
}
FORBIDDEN_IMPORT_ROUTE = {
    "irodori_tts/inference_runtime.py": ("from .codec import DACVAECodec", "import torchaudio"),
    "irodori_tts/codec.py": ("from dacvae import DACVAE", "import torchaudio", "import soundfile as sf"),
}


def strict_json(data: bytes | str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in items:
            if key in out:
                raise ValueError(f"duplicate JSON key: {key}")
            out[key] = value
        return out

    return json.loads(data, object_pairs_hook=pairs)


def safe_relative(path: str) -> None:
    if not path or "\x00" in path or "\\" in path or path.startswith("/"):
        raise ValueError(f"unsafe path: {path!r}")
    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"unsafe path: {path!r}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_dependency_lock(path: Path, *, expected_sha256: str = SOURCE_LOCK_SHA256) -> dict[str, Any]:
    """Authenticate the source lock and expose only its forbidden closure.

    ``tomllib`` is part of Python 3.12.  This helper deliberately performs no
    package resolution, import, subprocess other than the caller's git checks,
    or network operation.  A lock digest mismatch is a hard failure: a newer
    lock may not silently change the legal/native dependency decision.
    """
    if not path.is_file() or path.is_symlink():
        raise ValueError("source: uv.lock must be a regular file")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(f"source: uv.lock SHA-256 mismatch ({digest} != {expected_sha256})")
    try:
        lock = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"source: uv.lock is not valid TOML: {exc}") from exc
    if lock.get("version") != 1 or lock.get("revision") != 3:
        raise ValueError("source: uv.lock format revision mismatch")
    package_rows = lock.get("package")
    if not isinstance(package_rows, list):
        raise ValueError("source: uv.lock package table is missing")
    packages: dict[str, dict[str, Any]] = {}
    for row in package_rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ValueError("source: uv.lock contains a malformed package row")
        name = row["name"]
        # uv can legitimately emit multiple rows for one package when
        # resolution markers select alternate wheels.  The forbidden closure
        # rows are expected to be unique in this authenticated lock; unrelated
        # duplicate rows are not relevant to this gate.
        if name in FORBIDDEN_DEPENDENCY_ROWS:
            if name in packages:
                raise ValueError(f"source: forbidden package has ambiguous rows: {name}")
            packages[name] = row
    evidence: dict[str, dict[str, Any]] = {}
    for name, expected in FORBIDDEN_DEPENDENCY_ROWS.items():
        row = packages.get(name)
        if row is None:
            raise ValueError(f"source: authenticated forbidden package row missing: {name}")
        if row.get("version") != expected["version"] or row.get("source") != expected["source"]:
            raise ValueError(f"source: forbidden package identity changed: {name}")
        dependencies = row.get("dependencies", [])
        if not isinstance(dependencies, list):
            raise ValueError(f"source: malformed dependency list: {name}")
        dependency_names = {
            item.get("name") for item in dependencies
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        if not expected["dependencies"] <= dependency_names:
            raise ValueError(f"source: forbidden dependency route changed: {name}")
        evidence[name] = {
            "version": row["version"],
            "source": row["source"],
            "dependencies": sorted(dependency_names),
        }
    return {
        "lock_sha256": digest,
        "status": "AUTHENTICATED_SOURCE_LOCK_FORBIDDEN_CLOSURE",
        "execution": "BLOCKED_BEFORE_SYNC_OR_IMPORT",
        "route": [
            "dacvae@1.0.0 (git commit 414c20785fc3a28373073ea8ef7a1316eeeaca6e)",
            "descript-audiotools@0.7.2",
            "librosa@0.11.0",
            "soxr@1.0.0 + soundfile@0.13.1",
            "soundfile -> cffi@2.0.0 -> pycparser@3.0",
        ],
        "forbidden_rows": evidence,
        "forbidden_native": [
            "soxr@1.0.0",
            "soundfile@0.13.1",
            "libsndfile (native library loaded by soundfile)",
            "cffi@2.0.0",
        ],
        "reason": "librosa resolves soxr and soundfile; soundfile requires a native libsndfile path via cffi",
    }


def dependency_gate() -> int:
    """Run the pre-download, stdlib-only policy gate.

    The lock is intentionally not present at this stage: the worker must run
    this gate before cloning source or downloading any model.  The fixed source
    lock digest and closure above are the authenticated inspection contract;
    the known forbidden route therefore remains a deliberate exit-2 block.
    """
    if len(SOURCE_REVISION) != 40 or len(DACVAE_REVISION) != 40:
        raise ValueError("dependency gate: source revision is not immutable")
    if len(SOURCE_LOCK_SHA256) != 64 or any(c not in "0123456789abcdef" for c in SOURCE_LOCK_SHA256):
        raise ValueError("dependency gate: source lock digest is not canonical SHA-256")
    print(
        json.dumps(
            {
                "status": "BLOCKED",
                "reason": "forbidden native/audio dependency closure",
                "source_revision": SOURCE_REVISION,
                "source_pyproject_sha256": SOURCE_PYPROJECT_SHA256,
                "source_uv_lock_sha256": SOURCE_LOCK_SHA256,
                "closure": FORBIDDEN_IMPORT_ROUTE,
                "forbidden_rows": sorted(FORBIDDEN_DEPENDENCY_ROWS),
                "forbidden_native": [
                    "soxr@1.0.0",
                    "soundfile@0.13.1",
                    "libsndfile (native library loaded by soundfile)",
                    "cffi@2.0.0",
                ],
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return 2


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def pointer_sha1(sha256: str, size: int) -> str:
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha256}\nsize {size}\n".encode()
    return hashlib.sha1(b"blob " + str(len(pointer)).encode() + b"\0" + pointer).hexdigest()


def rows(packet: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(packet, dict) or not isinstance(packet.get("files"), list):
        raise ValueError(f"{label}: malformed tree envelope")
    out: dict[str, dict[str, Any]] = {}
    for row in packet["files"]:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError(f"{label}: malformed file row")
        path = row["path"]
        safe_relative(path)
        if path in out:
            raise ValueError(f"{label}: duplicate path {path}")
        if row.get("type", "file") != "file":
            raise ValueError(f"{label}: non-file entry {path}")
        out[path] = row
    return out


def verify_tree(packet: dict[str, Any], local: Path, repo: str, revision: str, expected: frozenset[str], label: str) -> dict[str, Any]:
    if packet.get("repository") != repo or packet.get("revision") != revision or packet.get("resolved_revision") != revision:
        raise ValueError(f"{label}: repository/revision identity mismatch")
    if not HEX40.fullmatch(revision):
        raise ValueError(f"{label}: revision is not immutable")
    remote = rows(packet, label)
    if set(remote) != expected:
        raise ValueError(f"{label}: path set mismatch (missing={sorted(expected-set(remote))}, extra={sorted(set(remote)-expected)})")
    if not local.is_dir():
        raise ValueError(f"{label}: local snapshot missing")
    local_rows: dict[str, Path] = {}
    for candidate in local.rglob("*"):
        rel = candidate.relative_to(local).as_posix()
        if rel == ".cache" or rel.startswith(".cache/"):
            continue
        safe_relative(rel)
        if candidate.is_symlink():
            target = candidate.resolve()
            if local not in target.parents:
                raise ValueError(f"{label}: symlink escapes snapshot: {rel}")
        if not candidate.is_file():
            raise ValueError(f"{label}: non-regular local member: {rel}")
        local_rows[rel] = candidate
    if set(local_rows) != expected:
        raise ValueError(f"{label}: local path set mismatch")
    evidence = []
    for path in sorted(expected):
        row = remote[path]
        file = local_rows[path]
        size = file.stat().st_size
        if not isinstance(row.get("size"), int) or isinstance(row["size"], bool) or row["size"] != size:
            raise ValueError(f"{label}: size mismatch: {path}")
        if path in LFS_PAYLOADS:
            want_size, want_sha = LFS_PAYLOADS[path]
            if size != want_size or sha256_file(file) != want_sha:
                raise ValueError(f"{label}: authenticated LFS payload mismatch: {path}")
            if row.get("lfs_sha256") != want_sha or row.get("lfs_size") != want_size:
                raise ValueError(f"{label}: remote LFS identity mismatch: {path}")
            if row.get("git_blob_sha1") != pointer_sha1(want_sha, want_size):
                raise ValueError(f"{label}: LFS pointer Git identity mismatch: {path}")
        else:
            remote_oid = row.get("git_blob_sha1")
            if not isinstance(remote_oid, str) or not HEX40.fullmatch(remote_oid) or remote_oid != git_blob_sha1(file):
                raise ValueError(f"{label}: Git blob identity mismatch: {path}")
        evidence.append({"path": path, "bytes": size, "sha256": sha256_file(file)})
    return {"repository": repo, "revision": revision, "files": evidence}


def verify_tokenizer_tree(packet: dict[str, Any], local: Path) -> dict[str, Any]:
    """Authenticate tokenizer metadata without downloading model weights."""
    if packet.get("repository") != TOKENIZER_REPOSITORY or packet.get("revision") != TOKENIZER_REVISION or packet.get("resolved_revision") != TOKENIZER_REVISION:
        raise ValueError("tokenizer: repository/revision identity mismatch")
    remote = rows(packet, "tokenizer")
    if set(remote) != EXPECTED_TOKENIZER_PATHS:
        raise ValueError("tokenizer: exact eight-file tree mismatch")
    if not local.is_dir():
        raise ValueError("tokenizer: local snapshot missing")
    local_rows = {p.relative_to(local).as_posix(): p for p in local.rglob("*") if p.is_file() and not p.is_symlink()}
    if set(local_rows) != TOKENIZER_LOCAL_PATHS:
        raise ValueError("tokenizer: local snapshot must contain only seven small assets")
    evidence = []
    for path in sorted(EXPECTED_TOKENIZER_PATHS):
        row = remote[path]
        oid = row.get("git_blob_sha1")
        if not isinstance(oid, str) or not HEX40.fullmatch(oid):
            raise ValueError(f"tokenizer: {path} lacks authenticated Git blob oid")
        if path == "model.safetensors":
            lfs_sha = row.get("lfs_sha256")
            lfs_size = row.get("lfs_size")
            if not isinstance(lfs_sha, str) or not HEX64.fullmatch(lfs_sha) or not isinstance(lfs_size, int) or lfs_size <= 0:
                raise ValueError("tokenizer: model.safetensors lacks authenticated LFS metadata")
            if row.get("size") != lfs_size or row.get("sha256") != lfs_sha or row.get("downloaded") is not False:
                raise ValueError("tokenizer: server-only model.safetensors identity mismatch")
            evidence.append({"path": path, "bytes": lfs_size, "git_blob_sha1": oid,
                             "lfs_sha256": lfs_sha, "downloaded": False})
            continue
        file = local_rows[path]
        if (file.stat().st_size != row.get("size") or row.get("downloaded") is not True
                or row.get("sha256") != sha256_file(file) or git_blob_sha1(file) != oid):
            raise ValueError(f"tokenizer: local Git blob mismatch: {path}")
        evidence.append({"path": path, "bytes": file.stat().st_size,
                         "git_blob_sha1": oid, "sha256": sha256_file(file), "downloaded": True})
    return {"repository": TOKENIZER_REPOSITORY, "revision": TOKENIZER_REVISION,
            "license": TOKENIZER_LICENSE, "files": evidence,
            "model_safetensors_downloaded": False,
            "tokenizer_status": "AUTHENTICATED_SMALL_ASSETS"}


def inspect_safetensors(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("safetensors: missing header length")
        header_len = struct.unpack("<Q", raw)[0]
        total = path.stat().st_size
        if header_len > MAX_HEADER or header_len > total - 8:
            raise ValueError("safetensors: header exceeds bounded file region")
        header = strict_json(stream.read(header_len))
    if not isinstance(header, dict):
        raise ValueError("safetensors: header is not an object")
    end = total - 8 - header_len
    spans: list[tuple[int, int, str]] = []
    manifest: list[dict[str, Any]] = []
    for name, descriptor in header.items():
        safe_relative(name)
        if name == "__metadata__":
            if not isinstance(descriptor, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in descriptor.items()):
                raise ValueError("safetensors: metadata must be a string map")
            continue
        if not isinstance(descriptor, dict) or set(descriptor) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"safetensors: malformed descriptor {name}")
        dtype, shape, offsets = descriptor["dtype"], descriptor["shape"], descriptor["data_offsets"]
        if not isinstance(dtype, str) or dtype not in DTYPE_WIDTH or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"safetensors: malformed descriptor types {name}")
        if any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in shape + offsets):
            raise ValueError(f"safetensors: invalid shape/offset {name}")
        start, stop = offsets
        if stop < start or stop > end:
            raise ValueError(f"safetensors: out-of-range span {name}")
        elements = 1
        for dimension in shape:
            elements *= dimension
            if elements > (1 << 63):
                raise ValueError(f"safetensors: shape product overflow {name}")
        if stop - start != elements * DTYPE_WIDTH[dtype]:
            raise ValueError(f"safetensors: span does not match dtype/shape {name}")
        spans.append((start, stop, name))
        manifest.append({"name": name, "dtype": dtype, "shape": shape, "data_offsets": offsets})
    spans.sort()
    cursor = 0
    for start, stop, name in spans:
        if start != cursor:
            raise ValueError(f"safetensors: gap/overlap before {name}")
        cursor = stop
    if cursor != end:
        raise ValueError("safetensors: trailing uncovered body bytes")
    raw_manifest = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    prefixes: dict[str, int] = {}
    for descriptor in manifest:
        name = descriptor["name"]
        prefix = name.split(".", 1)[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    return {
        "tensor_count": len(manifest),
        "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "body_bytes": end,
        "tensor_manifest": manifest,
        "top_level_prefix_counts": dict(sorted(prefixes.items())),
        "role_status": "OBSERVED_ONLY_UNREVIEWED_FOR_COMPOSITE_BINDING",
    }


def inspect_readme(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        raise ValueError("README: missing strict front matter")
    front, body = text[4:].split("\n---\n", 1)
    values: dict[str, str] = {}
    for line in front.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key in values:
            raise ValueError(f"README: duplicate front matter key {key}")
        values[key.strip()] = value.strip()
    if values.get("license") != "mit" or values.get("language") != "ja" or values.get("pipeline_tag") != "text-to-speech":
        raise ValueError("README: release metadata mismatch")
    required = ("Semantic-DACVAE-Japanese-32dim", "48", "duration", "SilentCipher")
    if any(marker not in body for marker in required):
        raise ValueError("README: required topology/policy marker missing")
    return {"license": values["license"], "language": values["language"], "pipeline_tag": values["pipeline_tag"]}


def inspect_public_contract(path: Path, artifact: Path) -> dict[str, Any]:
    data = strict_json(path.read_bytes())
    if data.get("repository") != PUBLIC_REPOSITORY or data.get("revision") != PUBLIC_REVISION or data.get("file") != PUBLIC_FILE:
        raise ValueError("public GGUF: fixed identity mismatch")
    if data.get("bytes") != PUBLIC_BYTES or data.get("sha256") != PUBLIC_SHA256 or data.get("tensor_count") != 637:
        raise ValueError("public GGUF: fixed artifact contract mismatch")
    if not isinstance(data.get("tensor_manifest_sha256"), str) or not HEX64.fullmatch(data["tensor_manifest_sha256"]):
        raise ValueError("public GGUF: authenticated tensor manifest hash required")
    tensor_manifest = data.get("tensor_manifest")
    if not isinstance(tensor_manifest, list) or len(tensor_manifest) != 637:
        raise ValueError("public GGUF: complete 637-tensor manifest required")
    canonical = json.dumps(tensor_manifest, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != data["tensor_manifest_sha256"]:
        raise ValueError("public GGUF: tensor manifest hash mismatch")
    if not artifact.is_file() or artifact.is_symlink() or artifact.stat().st_size != PUBLIC_BYTES:
        raise ValueError("public GGUF: downloaded artifact size mismatch")
    if sha256_file(artifact) != PUBLIC_SHA256:
        raise ValueError("public GGUF: downloaded artifact SHA-256 mismatch")
    return data


def inspect_public_gguf(artifact: Path, contract: dict[str, Any]) -> dict[str, Any]:
    """Derive the historical GGUF descriptors from its downloaded header."""
    with artifact.open("rb") as stream:
        raw = stream.read(64 * 1024 * 1024)
    cursor = 0

    def take(size: int) -> bytes:
        nonlocal cursor
        if cursor < 0 or cursor + size > len(raw):
            raise ValueError("public GGUF header truncated")
        result = raw[cursor : cursor + size]
        cursor += size
        return result

    def u32() -> int:
        return struct.unpack("<I", take(4))[0]

    def u64() -> int:
        return struct.unpack("<Q", take(8))[0]

    def string() -> str:
        size = u64()
        if size > 64 * 1024 * 1024:
            raise ValueError("public GGUF string exceeds bound")
        return take(size).decode("utf-8")

    def skip(kind: int) -> None:
        widths = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if kind in widths:
            take(widths[kind])
        elif kind == 8:
            string()
        elif kind == 9:
            element_kind, count = u32(), u64()
            if count > 1_000_000:
                raise ValueError("public GGUF metadata array exceeds bound")
            for _ in range(count):
                skip(element_kind)
        else:
            raise ValueError(f"unknown public GGUF metadata type {kind}")

    if take(4) != b"GGUF" or u32() not in (2, 3):
        raise ValueError("public artifact is not a supported GGUF")
    tensor_count, metadata_count = u64(), u64()
    if tensor_count != 637 or metadata_count > 1_000_000:
        raise ValueError("public GGUF count contract mismatch")
    for _ in range(metadata_count):
        string()
        skip(u32())
    dtypes = {0: "F32", 1: "F16", 30: "BF16"}
    descriptors = []
    for _ in range(tensor_count):
        name = string()
        rank = u32()
        if rank > 4:
            raise ValueError("public GGUF tensor rank exceeds bound")
        shape = [u64() for _ in range(rank)]
        dtype = dtypes.get(u32(), "UNKNOWN")
        descriptors.append({"dtype": dtype, "name": name, "offset": u64(), "shape": shape})
    if cursor != contract.get("header_bytes") or descriptors != contract.get("tensor_manifest"):
        raise ValueError("public GGUF header differs from reviewed descriptor manifest")
    manifest_hash = hashlib.sha256(json.dumps(descriptors, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if manifest_hash != contract.get("tensor_manifest_sha256"):
        raise ValueError("public GGUF descriptor manifest hash mismatch")
    return {"header_bytes": cursor, "tensor_count": tensor_count, "tensor_manifest_sha256": manifest_hash, "descriptor_source": "downloaded GGUF header"}


def inspect_codec(packet: dict[str, Any], local: Path) -> dict[str, Any]:
    if packet.get("repository") != CODEC_REPOSITORY:
        raise ValueError("codec: repository identity mismatch")
    revision = packet.get("resolved_revision")
    if revision != CODEC_REVISION:
        raise ValueError("codec: fixed immutable revision mismatch")
    remote = rows(packet, "codec")
    if set(remote) != EXPECTED_CODEC_PATHS:
        raise ValueError("codec: complete three-file release tree required")
    actual = {p.relative_to(local).as_posix(): p for p in local.rglob("*") if p.is_file() and not p.is_symlink()}
    if set(actual) != EXPECTED_CODEC_PATHS:
        raise ValueError("codec: local tree mismatch")
    for name, file in actual.items():
        row = remote[name]
        if row.get("size") != file.stat().st_size:
            raise ValueError(f"codec: size mismatch: {name}")
        if name == "weights.pth":
            lfs_sha = row.get("lfs_sha256")
            lfs_size = row.get("lfs_size")
            if not isinstance(lfs_sha, str) or not HEX64.fullmatch(lfs_sha) or lfs_size != file.stat().st_size:
                raise ValueError("codec: weights.pth lacks authenticated LFS identity")
            if sha256_file(file) != lfs_sha or row.get("git_blob_sha1") != pointer_sha1(lfs_sha, lfs_size):
                raise ValueError("codec: weights.pth payload/pointer mismatch")
        else:
            oid = row.get("git_blob_sha1")
            if not isinstance(oid, str) or not HEX40.fullmatch(oid) or oid != git_blob_sha1(file):
                raise ValueError(f"codec: Git blob mismatch: {name}")
    codec_readme = (local / "README.md").read_text(encoding="utf-8")
    if "license: mit" not in codec_readme.lower() or "latent dimension" not in codec_readme.lower():
        raise ValueError("codec: primary README license/topology declaration missing")
    members: list[str] = []
    try:
        with zipfile.ZipFile(actual["weights.pth"]) as archive:
            seen: set[str] = set()
            total_uncompressed = 0
            for info in archive.infolist():
                safe_relative(info.filename)
                if info.filename in seen:
                    raise ValueError("codec: duplicate archive member")
                seen.add(info.filename)
                if info.is_dir() or info.external_attr >> 16 & 0o170000 != 0o100000:
                    raise ValueError("codec: non-regular archive member")
                if info.file_size > actual["weights.pth"].stat().st_size:
                    raise ValueError("codec: archive member exceeds bounded size")
                total_uncompressed += info.file_size
                members.append(info.filename)
            if len(members) > 100_000 or total_uncompressed > actual["weights.pth"].stat().st_size:
                raise ValueError("codec: archive member bounds exceeded")
    except zipfile.BadZipFile as exc:
        raise ValueError(f"codec: weights.pth is not a bounded inspectable archive: {exc}") from exc
    try:
        import torch

        unsafe_globals = torch.serialization.get_unsafe_globals_in_checkpoint(str(actual["weights.pth"]))
        loaded = torch.load(actual["weights.pth"], weights_only=True, map_location="cpu")
    except Exception as exc:
        raise ValueError(f"codec: restricted torch.load failed; no unsafe fallback: {exc}") from exc
    tensors: list[dict[str, Any]] = []

    def walk(value: Any, prefix: str, depth: int = 0) -> None:
        if depth > 32 or len(tensors) > 100_000:
            raise ValueError("codec: recursive tensor manifest bounds exceeded")
        if isinstance(value, torch.Tensor):
            if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"codec: non-finite tensor {prefix}")
            tensors.append({"name": prefix, "shape": [int(x) for x in value.shape], "dtype": str(value.dtype), "elements": int(value.numel())})
            return
        if isinstance(value, dict):
            for key in sorted(value):
                if not isinstance(key, str):
                    raise ValueError("codec: tensor manifest key is not string")
                walk(value[key], f"{prefix}.{key}" if prefix else key, depth + 1)
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, f"{prefix}[{index}]", depth + 1)
            return
        if value is None or isinstance(value, (bool, int, float, str)):
            return
        raise ValueError(f"codec: unsupported loaded value at {prefix}")

    walk(loaded, "")
    if not tensors:
        raise ValueError("codec: safe checkpoint contains no tensors")
    manifest_hash = hashlib.sha256(json.dumps(tensors, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    file_evidence = [{"path": name, "bytes": file.stat().st_size,
                      "sha256": sha256_file(file),
                      "lfs_sha256": remote[name].get("lfs_sha256") if name == "weights.pth" else None}
                     for name, file in sorted(actual.items())]
    return {"repository": CODEC_REPOSITORY, "revision": revision, "files": file_evidence,
            "weights_archive_members": members,
            "unsafe_globals": sorted(unsafe_globals), "tensor_count": len(tensors), "tensor_manifest_sha256": manifest_hash,
            "tensors": tensors, "pickle_execution": "WEIGHTS_ONLY"}


def inspect_source(path: Path) -> dict[str, Any]:
    if not (path / ".git").exists():
        raise ValueError("source: git checkout required")
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()

    head = git("rev-parse", "HEAD")
    if head != SOURCE_REVISION:
        raise ValueError("source: immutable revision mismatch")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("source: checkout is dirty")
    required = {
        "LICENSE",
        "pyproject.toml",
        "uv.lock",
        "irodori_tts/config.py",
        "irodori_tts/model.py",
        "irodori_tts/inference_runtime.py",
        "irodori_tts/tokenizer.py",
        "irodori_tts/text_normalization.py",
        "irodori_tts/duration.py",
        "irodori_tts/rf.py",
        "irodori_tts/codec.py",
        "configs/train_500m_v3_phase1_body.yaml",
        "configs/train_500m_v3_phase2_duration.yaml",
    }
    tracked = set(git("ls-files").splitlines())
    if not required <= tracked:
        raise ValueError(f"source: required roles missing: {sorted(required-tracked)}")
    license_text = (path / "LICENSE").read_text(encoding="utf-8")
    if not license_text.strip() or "MIT" not in license_text.upper():
        raise ValueError("source: primary MIT license text missing")
    role_hashes = {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in sorted(required)}
    if sha256_file(path / "pyproject.toml") != SOURCE_PYPROJECT_SHA256:
        raise ValueError("source: pyproject.toml SHA-256 mismatch")
    lock_evidence = inspect_dependency_lock(path / "uv.lock")
    # Authenticate the semantic axes from the pinned source itself.  These
    # checks intentionally use exact source literals rather than comments in
    # the native implementation.
    body_config = (path / "configs/train_500m_v3_phase1_body.yaml").read_text(encoding="utf-8")
    duration_config = (path / "configs/train_500m_v3_phase2_duration.yaml").read_text(encoding="utf-8")
    model_source = (path / "irodori_tts/model.py").read_text(encoding="utf-8")
    config_source = (path / "irodori_tts/config.py").read_text(encoding="utf-8")
    rf_source = (path / "irodori_tts/rf.py").read_text(encoding="utf-8")
    codec_source = (path / "irodori_tts/codec.py").read_text(encoding="utf-8")
    norm_source = (path / "irodori_tts/text_normalization.py").read_text(encoding="utf-8")
    axes = {
        "latent_dim": "latent_dim: 32",
        "latent_patch_size": "latent_patch_size: 1",
        "text_vocab_size": "text_vocab_size: 99574",
        "text_tokenizer_repo": "text_tokenizer_repo: llm-jp/llm-jp-3-150m",
        "model_dim": "model_dim: 1280",
        "num_layers": "num_layers: 12",
        "num_heads": "num_heads: 20",
        "mlp_ratio": "mlp_ratio: 2.875",
        "text_dim": "text_dim: 512",
        "text_layers": "text_layers: 10",
        "text_heads": "text_heads: 8",
        "speaker_dim": "speaker_dim: 768",
        "speaker_layers": "speaker_layers: 8",
        "speaker_heads": "speaker_heads: 12",
        "timestep_embed_dim": "timestep_embed_dim: 512",
        "adaln_rank": "adaln_rank: 192",
        "text_add_bos": "text_add_bos: bool = True",
        "norm_eps": "norm_eps: float = 1e-5",
    }
    config_axes = {"text_add_bos", "norm_eps"}
    if any(
        marker not in (config_source if name in config_axes else body_config)
        for name, marker in axes.items()
    ):
        missing = [
            name
            for name, marker in axes.items()
            if marker not in (config_source if name in config_axes else body_config)
        ]
        raise ValueError(f"source: pinned v3 body axes missing or changed: {missing}")
    duration_axes = {
        "use_duration_predictor": "use_duration_predictor: true",
        "duration_aux_dim": "duration_aux_dim: 14",
        "duration_hidden_dim": "duration_hidden_dim: 1024",
        "duration_layers": "duration_layers: 3",
        "duration_attention_heads": "duration_attention_heads: 8",
        "duration_dropout": "duration_dropout: 0.1",
        "duration_architecture": "duration_architecture: token_sum_adarn_zero_no_aux",
        "duration_token_init_frames": "duration_token_init_frames: 9.0",
        "duration_speaker_fusion": "duration_speaker_fusion: adarn_zero",
    }
    if any(marker not in duration_config for marker in duration_axes.values()):
        missing = [name for name, marker in duration_axes.items() if marker not in duration_config]
        raise ValueError(f"source: pinned v3 duration axes missing or changed: {missing}")
    source_markers = {
        "half_head_rotary": "def _apply_rotary_half",
        "q_k_norm": "self.q_norm = RMSNorm((self.heads, self.head_dim)",
        "sigmoid_gate": "torch.sigmoid(self.gate(x))",
        "low_rank_adaln": "class LowRankAdaLN",
        "context_kv_cache": "build_context_kv_cache",
        "cfg_modes": 'cfg_guidance_mode not in {"independent", "joint", "alternating"}',
        "euler_sampler": "def sample_euler_rf_cfg",
        "sway_default": "sway_coeff: float = -1.0",
        "codec_patchify": "def patchify_latent",
        "codec_unpatchify": "def unpatchify_latent",
    }
    source_blobs = model_source + rf_source + codec_source
    missing = [name for name, marker in source_markers.items() if marker not in source_blobs]
    if missing:
        raise ValueError(f"source: required implementation roles missing: {missing}")
    if "def normalize_text" not in norm_source:
        raise ValueError("source: normalize_text role missing")
    for role, markers in FORBIDDEN_IMPORT_ROUTE.items():
        role_source = (path / role).read_text(encoding="utf-8")
        if any(marker not in role_source for marker in markers):
            raise ValueError(f"source: dependency import route changed: {role}")
    origins = git("remote", "get-url", "origin")
    if origins.removesuffix(".git") != SOURCE_REPOSITORY.removesuffix(".git"):
        raise ValueError("source: origin mismatch")
    return {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION,
            "license": "separate source license evidence", "role_sha256": role_hashes,
            "authenticated_axes": axes | duration_axes,
            "implementation_roles": source_markers,
            "tokenizer": {"repository": TOKENIZER_REPOSITORY,
                           "revision": TOKENIZER_REVISION,
                          "status": "SELECTED_IMMUTABLE_EVIDENCE",
                          "source_config_note": "source names only the repository; selected revision is adapted evidence"},
            "reference_environment": {
                "project": "official source pyproject.toml",
                "lock": "official source uv.lock",
                "status": "AUTHENTICATED_SOURCE_LOCK",
                "pyproject_sha256": sha256_file(path / "pyproject.toml"),
                "lock_sha256": lock_evidence["lock_sha256"],
                "dependency_closure_status": lock_evidence["status"],
                "dependency_closure": lock_evidence,
            }}


def blocked_manifest(error: str | None = None) -> dict[str, Any]:
    return {
        "model_identity": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION, "file": MODEL_FILE, "bytes": MODEL_BYTES, "sha256": MODEL_SHA256},
        "source_identity": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION},
        "codec_identity": {"repository": CODEC_REPOSITORY, "revision": CODEC_REVISION},
        "tokenizer_identity": {"repository": TOKENIZER_REPOSITORY, "revision": TOKENIZER_REVISION,
                                "license": TOKENIZER_LICENSE,
                                "status": "SELECTED_IMMUTABLE_EVIDENCE",
                                "source_config_note": "source names only the repository; selected revision is adapted evidence"},
        "model_composite_roles": {
            "status": "EXACT_TENSOR_ROLE_REVIEW_PENDING",
            "required": ["text_encoder", "speaker_encoder", "rf_dit", "duration_predictor"],
        },
        "historical_public_identity": {"repository": PUBLIC_REPOSITORY, "revision": PUBLIC_REVISION, "file": PUBLIC_FILE, "bytes": PUBLIC_BYTES, "sha256": PUBLIC_SHA256},
        "status": "BLOCKED",
        "evidence_stage": "INSPECTION_ONLY",
        "inspection_status": "INSPECTION_ERROR" if error else "AUTHENTICATED_EVIDENCE_COMPLETE",
        "runtime_status": "PARTIAL_RUNTIME_BLOCKED",
        "cpu_status": "UNSUPPORTED_FULL_TTS",
        "metal_status": "BLOCKED_BY_CPU",
        "parity_status": "NOT_RUN",
        "publication": "NO_UPLOAD",
        "error": error,
        "blockers": [
            "authenticated source uv.lock has a forbidden DACVAE -> audiotools/librosa -> soxr/soundfile native closure",
            "dependency gate must exit 2 before source/model download, uv sync, or official import",
            "official token-ID parity requires the selected tokenizer small-asset packet and actual runtime run",
            "model.safetensors exact tensor-name/shape/role manifest and full composite binder pending",
            "RF-DiT duration/CFG sampler and Semantic-DACVAE PCM parity pending",
            "Japanese tokenizer/G2P, dataset provenance, and legal review pending",
        ],
    }


def self_test() -> None:
    import tempfile

    def stensor(header: dict[str, Any], body: bytes) -> bytes:
        encoded = json.dumps(header, separators=(",", ":")).encode()
        return struct.pack("<Q", len(encoded)) + encoded + body

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        good = root / "good.safetensors"
        good.write_bytes(stensor({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}, b"\0" * 4))
        assert inspect_safetensors(good)["tensor_count"] == 1
        for bad in (
            stensor({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 3]}}, b"\0" * 4),
            stensor({"x": {"dtype": "F32", "shape": [1], "data_offsets": [1, 4]}}, b"\0" * 4),
            struct.pack("<Q", MAX_HEADER + 1),
        ):
            candidate = root / "bad.safetensors"
            candidate.write_bytes(bad)
            try:
                inspect_safetensors(candidate)
            except ValueError:
                pass
            else:
                raise AssertionError("malformed safetensors accepted")
        assert pointer_sha1(MODEL_SHA256, MODEL_BYTES) == pointer_sha1(MODEL_SHA256, MODEL_BYTES)
        assert pointer_sha1("0" * 64, MODEL_BYTES) != pointer_sha1(MODEL_SHA256, MODEL_BYTES)
        try:
            strict_json('{"x":1,"x":2}')
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate JSON accepted")
        try:
            safe_relative("samples/../model")
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe path accepted")
        try:
            rows({"files": [{"path": "x"}, {"path": "x"}]}, "fixture")
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate tree path accepted")
        lock = root / "uv.lock"
        lock.write_text(
            """version = 1
revision = 3

[[package]]
name = "dacvae"
version = "1.0.0"
source = { git = "https://github.com/facebookresearch/dacvae#414c20785fc3a28373073ea8ef7a1316eeeaca6e" }
dependencies = [{ name = "descript-audiotools" }]

[[package]]
name = "descript-audiotools"
version = "0.7.2"
source = { registry = "https://pypi.org/simple" }
dependencies = [{ name = "librosa" }, { name = "soundfile" }]

[[package]]
name = "librosa"
version = "0.11.0"
source = { registry = "https://pypi.org/simple" }
dependencies = [{ name = "soxr" }, { name = "soundfile" }]

[[package]]
name = "soxr"
version = "1.0.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "soundfile"
version = "0.13.1"
source = { registry = "https://pypi.org/simple" }
dependencies = [{ name = "cffi" }]

[[package]]
name = "cffi"
version = "2.0.0"
source = { registry = "https://pypi.org/simple" }
dependencies = [{ name = "pycparser" }]

[[package]]
name = "pycparser"
version = "3.0"
source = { registry = "https://pypi.org/simple" }
""",
            encoding="utf-8",
        )
        synthetic_digest = sha256_file(lock)
        lock_evidence = inspect_dependency_lock(lock, expected_sha256=synthetic_digest)
        assert lock_evidence["status"] == "AUTHENTICATED_SOURCE_LOCK_FORBIDDEN_CLOSURE"
        lock.write_text(lock.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        try:
            inspect_dependency_lock(lock, expected_sha256=synthetic_digest)
        except ValueError:
            pass
        else:
            raise AssertionError("tampered source lock accepted")
        tokenizer = root / "tokenizer"
        tokenizer.mkdir()
        tokenizer_packet_rows = []
        for path in sorted(TOKENIZER_LOCAL_PATHS):
            member = tokenizer / path
            member.write_bytes(path.encode("utf-8"))
            tokenizer_packet_rows.append({
                "path": path, "type": "file", "size": member.stat().st_size,
                "sha256": sha256_file(member), "downloaded": True,
                "git_blob_sha1": git_blob_sha1(member),
            })
        tokenizer_packet_rows.append({
            "path": "model.safetensors", "type": "file", "size": 123,
            "sha256": "b" * 64, "downloaded": False, "git_blob_sha1": "a" * 40,
            "lfs_sha256": "b" * 64, "lfs_size": 123,
        })
        tokenizer_packet = {
            "repository": TOKENIZER_REPOSITORY,
            "revision": TOKENIZER_REVISION,
            "resolved_revision": TOKENIZER_REVISION,
            "files": tokenizer_packet_rows,
        }
        evidence = verify_tokenizer_tree(tokenizer_packet, tokenizer)
        assert evidence["tokenizer_status"] == "AUTHENTICATED_SMALL_ASSETS"
        assert evidence["model_safetensors_downloaded"] is False
        bad_packet = dict(tokenizer_packet, revision="0" * 40)
        try:
            verify_tokenizer_tree(bad_packet, tokenizer)
        except ValueError:
            pass
        else:
            raise AssertionError("mutable tokenizer revision accepted")
    print("irodori inspector self-test: ok")


def inspect(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    try:
        model_packet = strict_json(Path(args.model_tree).read_bytes())
        evidence = blocked_manifest()
        evidence["model"] = verify_tree(model_packet, Path(args.model_dir), MODEL_REPOSITORY, MODEL_REVISION, EXPECTED_MODEL_PATHS, "model")
        evidence["model_readme"] = inspect_readme(Path(args.model_dir) / "README.md")
        evidence["model_safetensors"] = inspect_safetensors(Path(args.model_dir) / MODEL_FILE)
        public_contract = inspect_public_contract(Path(args.public_contract), Path(args.public_gguf))
        evidence["public_gguf"] = {**public_contract, **inspect_public_gguf(Path(args.public_gguf), public_contract)}
        if not args.codec_tree or not args.codec_dir:
            raise ValueError("codec evidence is required; Semantic-DACVAE is a distinct component")
        codec_packet = strict_json(Path(args.codec_tree).read_bytes())
        evidence["codec"] = inspect_codec(codec_packet, Path(args.codec_dir))
        if not args.tokenizer_tree or not args.tokenizer_dir:
            raise ValueError("tokenizer evidence is required; use the selected immutable snapshot")
        tokenizer_packet = strict_json(Path(args.tokenizer_tree).read_bytes())
        evidence["tokenizer"] = verify_tokenizer_tree(tokenizer_packet, Path(args.tokenizer_dir))
        if not args.source_dir:
            raise ValueError("source checkout is required")
        evidence["source"] = inspect_source(Path(args.source_dir))
        evidence["notes"] = [
            "model is authenticated only as a checkpoint; no end-to-end TTS or PCM verdict",
            "tensor names/shapes are recorded from safetensors but role ownership is not accepted",
        ]
        # All release/source/tokenizer identities are now authenticated. This
        # status is limited to inspection evidence; runtime/native/public
        # support remains blocked by the unreviewed composite tensor binding.
        evidence["inspection_status"] = "AUTHENTICATED_EVIDENCE_COMPLETE"
        evidence["error"] = None
    except Exception as exc:  # preserve an auditable failure manifest
        evidence = blocked_manifest(f"{type(exc).__name__}: {exc}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(evidence, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if evidence["inspection_status"] != "AUTHENTICATED_EVIDENCE_COMPLETE":
        return 2
    return 2  # evidence completion never authorizes runtime/publication


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dependency-gate", action="store_true")
    parser.add_argument("--model-dir")
    parser.add_argument("--model-tree")
    parser.add_argument("--codec-dir")
    parser.add_argument("--codec-tree")
    parser.add_argument("--tokenizer-dir")
    parser.add_argument("--tokenizer-tree")
    parser.add_argument("--source-dir")
    parser.add_argument("--public-contract")
    parser.add_argument("--public-gguf")
    parser.add_argument("--manifest", default="irodori-inspection.json")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.dependency_gate:
        return dependency_gate()
    required = (args.model_dir, args.model_tree, args.public_contract, args.public_gguf,
                args.tokenizer_dir, args.tokenizer_tree)
    if any(value is None for value in required):
        parser.error("inspection requires --model-dir --model-tree --public-contract")
    return inspect(args)


if __name__ == "__main__":
    raise SystemExit(main())
