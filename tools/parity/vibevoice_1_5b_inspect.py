#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""VAST-only inspection for the fixed VibeVoice-1.5B composite.

The inspector authenticates the complete HF tree, header-only safetensors
manifests, and pinned source roles.  It deliberately does not convert, load
tensor bodies, execute upstream code, or claim CPU/Metal parity: the HF
release has no text-tokenizer files and its official inference path still
needs the authenticated Qwen companion, prefill state, streaming tokenizer,
and DPMSolverMultistepScheduler implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

HF_REPOSITORY = "microsoft/VibeVoice-1.5B"
HF_REVISION = "142f4a5dda029212cda8b118e9d99c3da27018d8"
QWEN_REPOSITORY = "Qwen/Qwen2.5-1.5B"
QWEN_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
PUBLIC_REPOSITORY = "vokra/vibevoice-1.5b"
PUBLIC_REVISION = "dec190628f58928fc247b1205b9da2dabc58b9da"
PUBLIC_BYTES = 5_408_160_960
PUBLIC_SHA256 = "8ef5f259dfab0b048151ce52d27468040f72b35b6909528e6db7fbb332ccaeac"
PUBLIC_MANIFEST_SHA256 = "45cb011420fdb114c7ad61d80888663bcc861e33b7945873836aee2450eb5702"
SOURCE_REPOSITORY = "https://github.com/microsoft/VibeVoice.git"
# The default branch was reset and later lost the non-streaming TTS path.
# This immutable orphan is the last official Microsoft commit that still
# contains the 1.5B generation/prefill/diffusion/decode implementation.
SOURCE_REVISION = "2f9a3d79a0e51bd1cf2ab40d36884c8948e6bb9c"
TRANSFORMERS_REPOSITORY = "https://github.com/huggingface/transformers.git"
TRANSFORMERS_REVISION = "5f4ecf2d9f867a1255131d2461d75793c0cf1db2"
TRANSFORMERS_TAG = "v4.51.3"
FORMAT = "vokra-vibevoice-1-5b-inspection-v1"
MAX_HEADER_BYTES = 64 * 1024 * 1024
MAX_FILES = 128
MAX_TENSORS = 10_000
HF_TRANSPORT_CACHE = ".cache/huggingface"
MODEL_FILES = {
    ".gitattributes", "README.md", "config.json", "preprocessor_config.json",
    "model.safetensors.index.json", "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors",
}
QWEN_FILES = {
    "LICENSE", "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt",
}
class FixedFile(NamedTuple):
    """Authenticated server identity; regular Git and LFS are distinct."""

    size: int
    kind: str
    git_blob_sha1: str | None = None
    lfs_payload_sha256: str | None = None


MODEL_IDENTITIES: dict[str, FixedFile] = {
    ".gitattributes": FixedFile(1519, "regular", "a6344aac8c09253b3b630fb776ae94478aa0275b"),
    "README.md": FixedFile(6534, "regular", "96cb3a1adb721605ef38654d5c6fc88a7b69ab7d"),
    "config.json": FixedFile(2762, "regular", "17feb5528f948e364cde8640523c5ff3927131ae"),
    "preprocessor_config.json": FixedFile(351, "regular", "3222200e715275bc2ab78999d046ffba304e4ee9"),
    "model.safetensors.index.json": FixedFile(122616, "regular", "e18709430e1002a3a43906f82cc4a5cb0181fdd1"),
    "model-00001-of-00003.safetensors": FixedFile(1975317828, "lfs", "e66194eb854984fd466b12b2fdc066c5a4d03f55", "c5f0a61ddeaeb028e3af540ba4dee7933ad30f9f30b6e1320dd9c875a2daa033"),
    "model-00002-of-00003.safetensors": FixedFile(1983051688, "lfs", "9a20c335e35ef8380c746c31c491e2f41de9e5ba", "81c3891f7b2493eb48a9eb6f5be0df48d4f1a4bfd952d84e21683ca6d0bf7969"),
    "model-00003-of-00003.safetensors": FixedFile(1449832938, "lfs", "b6df3287b34e95a343c35def9f08a54b37057a08", "cb6e7e5e86b4a41fffbe1f3aaf445d0d50b5e21ed47574101b777f77d75fa196"),
}
QWEN_IDENTITIES: dict[str, FixedFile] = {
    "LICENSE": FixedFile(11343, "regular", "6634c8cc3133b3848ec74b9f275acaaa1ea618ab"),
    "tokenizer_config.json": FixedFile(7228, "regular", "ba7e4c5637b9732dadcd66286ce48334e8b31e9e"),
    "tokenizer.json": FixedFile(7031645, "regular", "443909a61d429dff23010e5bddd28ff530edda00"),
    "vocab.json": FixedFile(2776833, "regular", "4783fe10ac3adce15ac8f358ef5462739852c569"),
    "merges.txt": FixedFile(1671839, "regular", "20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0"),
}
SOURCE_ROLES = (
    "LICENSE",
    "pyproject.toml",
    "demo/inference_from_file.py",
    "vibevoice/modular/configuration_vibevoice.py",
    "vibevoice/modular/modeling_vibevoice.py",
    "vibevoice/modular/modeling_vibevoice_inference.py",
    "vibevoice/modular/modular_vibevoice_tokenizer.py",
    "vibevoice/modular/modular_vibevoice_text_tokenizer.py",
    "vibevoice/modular/modular_vibevoice_diffusion_head.py",
    "vibevoice/modular/streamer.py",
    "vibevoice/processor/vibevoice_processor.py",
    "vibevoice/processor/vibevoice_tokenizer_processor.py",
    "vibevoice/schedule/dpm_solver.py",
    "vibevoice/configs/qwen2.5_1.5b_64k.json",
)
TRANSFORMER_ROLES = (
    "LICENSE",
    "src/transformers/audio_utils.py",
    "src/transformers/feature_extraction_utils.py",
    "src/transformers/generation/utils.py",
    "src/transformers/processing_utils.py",
    "src/transformers/models/qwen2/configuration_qwen2.py",
    "src/transformers/models/qwen2/modeling_qwen2.py",
)
SOURCE_LICENSE_BLOB = "269a8973689dbb250d355f516f8a30c1cc66b8e4"
TRANSFORMER_LICENSE_BLOB = "68b7d66c97d66c58de883ed0c451af2b3183e6f3"
SOURCE_ROLE_BLOBS: dict[str, str] = {
    "LICENSE": "269a8973689dbb250d355f516f8a30c1cc66b8e4",
    "pyproject.toml": "ece97ec7b9177119f4fdd1fb1f329b430876dd89",
    "demo/inference_from_file.py": "078b53a11f0e4bf617171101655c11d0d394e66b",
    "vibevoice/configs/qwen2.5_1.5b_64k.json": "febd05cd76d2a5df49c39fabcde478aa18e1ba78",
    "vibevoice/modular/configuration_vibevoice.py": "fcffcb93afae6358f57a155d6fb6eb009b69a706",
    "vibevoice/modular/modeling_vibevoice.py": "016a38979ef74e1ea9c5dc0405c8ac13feb0a0d5",
    "vibevoice/modular/modeling_vibevoice_inference.py": "7e10af4a2bd1f5ba4ec454942e4a87bb312aa091",
    "vibevoice/modular/modular_vibevoice_diffusion_head.py": "59de50fb2fe80d6b1ba5a50c9de1ef9cffc4f614",
    "vibevoice/modular/modular_vibevoice_text_tokenizer.py": "bfa7bdd18783d67d488371071cc6425ceb80b376",
    "vibevoice/modular/modular_vibevoice_tokenizer.py": "fbd5182f82ba61898a09b762ec20e6f34270d053",
    "vibevoice/modular/streamer.py": "7a76cb063ec1b48a9e6397f113b47663ae6c5799",
    "vibevoice/processor/vibevoice_processor.py": "66d0a9de2e2beb3eeeaf0bb5a5eb523d5f61acae",
    "vibevoice/processor/vibevoice_tokenizer_processor.py": "0d854b7842658dbb573b6623c05d1326a71221cf",
    "vibevoice/schedule/dpm_solver.py": "806241f4352465f50114b587e0db2c63bc73c24f",
}
TRANSFORMER_ROLE_BLOBS: dict[str, str] = {
    "LICENSE": "68b7d66c97d66c58de883ed0c451af2b3183e6f3",
    "src/transformers/audio_utils.py": "8420a84e089e03be9a80fb63c237e34203ea28a0",
    "src/transformers/feature_extraction_utils.py": "ca2a3b5fde31d81554c76e26a24ebb4b806ed052",
    "src/transformers/generation/utils.py": "95d211cd5e31d79d4c926f78452fb5662f5125cf",
    "src/transformers/processing_utils.py": "b1c40e7ff2d7c08e8b8e741a59f933f58c13fb30",
    "src/transformers/models/qwen2/configuration_qwen2.py": "2e82f1976f3922f3620415f4eace6c6e046243f8",
    "src/transformers/models/qwen2/modeling_qwen2.py": "16a7316e2d0e56eafe301a7f2d8693d6cc6c73ec",
}


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    return digest_with_prefix(path, "sha1", f"blob {path.stat().st_size}\0".encode())


def git_blob_sha1_bytes(content: bytes) -> str:
    return hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()


def digest_with_prefix(path: Path, algorithm: str, prefix: bytes) -> str:
    h = hashlib.new(algorithm)
    h.update(prefix)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=strict_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError, RuntimeError) as error:
        raise RuntimeError(f"strict JSON failure at {path}: {error}") from error


def safe_relative(value: str, label: str = "path") -> None:
    path = Path(value)
    if not value or "\x00" in value or "\\" in value or path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"unsafe {label}: {value!r}")


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT).strip()


def lfs_pointer(sha256: str, size: int) -> bytes:
    if len(sha256) != 64 or size < 0:
        raise RuntimeError("invalid LFS identity")
    return f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha256}\nsize {size}\n".encode()


def local_files(root: Path) -> set[str]:
    result: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise RuntimeError(f"snapshot symlink is not allowed: {relative}")
        if relative.as_posix() == ".cache":
            if path.is_symlink() or not path.is_dir():
                raise RuntimeError("HF cache parent must be a real directory")
            continue
        if relative.as_posix() == HF_TRANSPORT_CACHE:
            if path.is_symlink() or not path.is_dir():
                raise RuntimeError("HF transport cache must be a real directory")
            continue
        if relative.as_posix().startswith(HF_TRANSPORT_CACHE + "/"):
            continue
        if ".cache" in relative.parts:
            raise RuntimeError(f"unexpected cache outside {HF_TRANSPORT_CACHE}: {relative}")
        if not path.is_file():
            if not path.is_dir():
                raise RuntimeError(f"non-regular snapshot member: {relative}")
            continue
        result.add(relative.as_posix())
    if len(result) > MAX_FILES:
        raise RuntimeError("snapshot file bound exceeded")
    return result


def server_inventory(snapshot: Path, packet: Path, repository: str = HF_REPOSITORY, revision: str = HF_REVISION, expected_files: set[str] = MODEL_FILES, check_fixed_sizes: bool = True) -> list[dict[str, Any]]:
    envelope = load_json(packet)
    if not isinstance(envelope, dict) or set(envelope) != {"repository", "requested_revision", "resolved_revision", "walk", "files"} or envelope.get("repository") != repository or envelope.get("requested_revision") != revision or envelope.get("resolved_revision") != revision or envelope.get("walk") != "recursive_file_only":
        raise RuntimeError("HF server-tree repository/revision mismatch")
    files = envelope.get("files")
    if not isinstance(files, list) or len(files) != len(expected_files):
        raise RuntimeError("HF server-tree is not complete")
    names: set[str] = set()
    rows: list[dict[str, Any]] = []
    fixed_identities = MODEL_IDENTITIES if repository == HF_REPOSITORY else QWEN_IDENTITIES
    if check_fixed_sizes and set(fixed_identities) != expected_files:
        raise RuntimeError(f"complete fixed server identity table is unavailable for {repository}")
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "type", "size", "git_blob_sha1", "lfs_pointer_git_blob_sha1", "lfs_payload_sha256", "lfs_payload_size"}:
            raise RuntimeError("server packet identity fields are not exact")
        name, kind, size = (item[key] for key in ("path", "type", "size"))
        git_id, pointer_id, lfs_id, payload_size = (item[key] for key in ("git_blob_sha1", "lfs_pointer_git_blob_sha1", "lfs_payload_sha256", "lfs_payload_size"))
        if kind != "file" or not isinstance(name, str) or not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise RuntimeError("server packet has invalid file entry")
        safe_relative(name, "server path")
        if name in names or name not in expected_files:
            raise RuntimeError(f"unexpected/duplicate HF path: {name}")
        if lfs_id is None:
            if not isinstance(git_id, str) or re.fullmatch(r"[0-9a-f]{40}", git_id) is None or pointer_id is not None or payload_size is not None:
                raise RuntimeError(f"invalid regular Git identity: {name}")
        elif git_id is not None or not isinstance(pointer_id, str) or re.fullmatch(r"[0-9a-f]{40}", pointer_id) is None or not isinstance(lfs_id, str) or re.fullmatch(r"[0-9a-f]{64}", lfs_id) is None or not isinstance(payload_size, int) or isinstance(payload_size, bool) or payload_size < 0 or payload_size != size:
            raise RuntimeError(f"invalid LFS identity: {name}")
        if check_fixed_sizes:
            fixed = fixed_identities.get(name)
            if fixed is None or fixed.size != size:
                raise RuntimeError(f"fixed server file size mismatch: {name}")
            if fixed.kind == "regular" and lfs_id is not None:
                raise RuntimeError(f"fixed regular file reported as LFS: {name}")
            if fixed.kind == "lfs" and lfs_id is None:
                raise RuntimeError(f"fixed LFS file reported as regular: {name}")
        path = snapshot / name
        if not path.is_file() or path.stat().st_size != size:
            raise RuntimeError(f"local HF path/size mismatch: {name}")
        if lfs_id is None:
            if git_blob_sha1(path).lower() != git_id.lower():
                raise RuntimeError(f"Git blob mismatch: {name}")
            if check_fixed_sizes and git_id != fixed_identities[name].git_blob_sha1:
                raise RuntimeError(f"fixed Git identity mismatch: {name}")
        else:
            if digest(path) != lfs_id.lower():
                raise RuntimeError(f"LFS content mismatch: {name}")
            pointer_git = git_blob_sha1_bytes(lfs_pointer(lfs_id, size))
            if pointer_git.lower() != pointer_id.lower():
                raise RuntimeError(f"LFS pointer Git blob mismatch: {name}")
            if check_fixed_sizes and pointer_id != fixed_identities[name].git_blob_sha1:
                raise RuntimeError(f"fixed LFS pointer identity mismatch: {name}")
            if check_fixed_sizes and lfs_id != fixed_identities[name].lfs_payload_sha256:
                raise RuntimeError(f"fixed LFS payload identity mismatch: {name}")
        names.add(name)
        rows.append({"path": name, "bytes": size, "git_blob_sha1": git_id, "lfs_pointer_git_blob_sha1": pointer_id, "lfs_payload_sha256": lfs_id, "lfs_payload_size": payload_size, "sha256": digest(path)})
    if names != expected_files or local_files(snapshot) != names:
        raise RuntimeError(f"HF tree mismatch: local={sorted(local_files(snapshot))} packet={sorted(names)}")
    return sorted(rows, key=lambda row: row["path"])

def inspect_safetensors(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise RuntimeError("truncated safetensors prefix")
        header_len = int.from_bytes(prefix, "little")
        if header_len <= 0 or header_len > MAX_HEADER_BYTES or header_len > size - 8:
            raise RuntimeError("invalid safetensors header length")
        raw = stream.read(header_len)
    header = json.loads(raw.decode("utf-8"), object_pairs_hook=strict_pairs)
    if not isinstance(header, dict):
        raise RuntimeError("safetensors header is not an object")
    metadata = header.pop("__metadata__", None)
    if metadata is not None and (not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items())):
        raise RuntimeError("safetensors metadata is not a string map")
    payload = size - 8 - header_len
    intervals: list[tuple[int, int, str]] = []
    rows: list[dict[str, Any]] = []
    for name, descriptor in header.items():
        safe_relative(name, "tensor name")
        if not isinstance(descriptor, dict) or set(descriptor) != {"dtype", "shape", "data_offsets"}:
            raise RuntimeError(f"invalid tensor descriptor: {name}")
        dtype, shape, offsets = (descriptor[key] for key in ("dtype", "shape", "data_offsets"))
        if dtype not in {"BF16", "F16", "F32"} or not isinstance(shape, list) or any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in shape) or not isinstance(offsets, list) or len(offsets) != 2 or any(not isinstance(x, int) or isinstance(x, bool) for x in offsets):
            raise RuntimeError(f"unsafe tensor descriptor: {name}")
        start, end = offsets
        if end < start or end > payload:
            raise RuntimeError(f"safetensors offset gap/overlap/range: {name}")
        elements = 1
        for dimension in shape:
            elements *= dimension
        width = {"BF16": 2, "F16": 2, "F32": 4}[dtype]
        if end - start != elements * width:
            raise RuntimeError(f"safetensors byte shape mismatch: {name}")
        rows.append({"name": name, "dtype": dtype, "shape": shape, "offsets": offsets, "elements": elements})
        intervals.append((start, end, name))
        if len(rows) > MAX_TENSORS:
            raise RuntimeError("tensor bound exceeded")
    cursor = 0
    for start, end, name in sorted(intervals):
        if start != cursor:
            raise RuntimeError(f"safetensors offset gap/overlap before {name}")
        cursor = end
    if cursor != payload:
        raise RuntimeError("safetensors trailing payload")
    return {"path": path.name, "bytes": size, "header_bytes": header_len, "payload_bytes": payload, "tensor_count": len(rows), "parameter_count": sum(row["elements"] for row in rows), "metadata": metadata or {}}, rows


def manifest_sha256(rows: list[dict[str, Any]]) -> str:
    canonical = bytearray()
    for row in sorted(rows, key=lambda item: item["name"]):
        canonical.extend(row["name"].encode())
        canonical.append(0)
        shape = row["shape"]
        canonical.extend(len(shape).to_bytes(8, "little"))
        for dimension in shape:
            canonical.extend(dimension.to_bytes(8, "little"))
    return hashlib.sha256(canonical).hexdigest()


def source_inventory(source: Path, transformers: Path) -> dict[str, Any]:
    def tracked_rows(root: Path, roles: tuple[str, ...], expected_license_blob: str) -> tuple[list[dict[str, Any]], str]:
        tracked_output = git(root, "ls-files", "--stage")
        tracked: dict[str, tuple[str, int, str]] = {}
        for line in tracked_output.splitlines():
            try:
                metadata, path_name = line.split("\t", 1)
                mode, object_id, stage = metadata.split()
                stage_number = int(stage)
            except (ValueError, TypeError) as error:
                raise RuntimeError(f"malformed tracked source row: {line!r}") from error
            if path_name in tracked:
                raise RuntimeError(f"duplicate tracked source path: {path_name}")
            if stage_number != 0 or mode not in {"100644", "100755"}:
                raise RuntimeError(f"tracked source path is not stage-0 regular 100644/100755: {path_name}")
            tracked[path_name] = (object_id, stage_number, mode)
        if not tracked:
            raise RuntimeError(f"source checkout has no tracked files: {root}")
        rows: list[dict[str, Any]] = []
        for path_name, (index_object, stage_number, git_mode) in sorted(tracked.items()):
            path = root / path_name
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"tracked source path is not a regular file: {path_name}")
            filesystem_mode = f"{path.stat().st_mode & 0o7777:04o}"
            expected_mode = git_mode[2:]
            if filesystem_mode != expected_mode:
                raise RuntimeError(f"tracked source filesystem mode drift: {path_name}")
            if path_name in roles and filesystem_mode != "0644":
                raise RuntimeError(f"fixed source role must be mode 100644: {path_name}")
            head_object = git(root, "rev-parse", f"HEAD:{path_name}")
            working_object = git_blob_sha1(path)
            if index_object != head_object or index_object != working_object:
                raise RuntimeError(f"tracked source object drift: {path_name}")
            rows.append({"path": path_name, "mode": filesystem_mode, "stage": stage_number, "index_object": index_object, "head_object": head_object, "working_git_blob_sha1": working_object, "bytes": path.stat().st_size, "sha256": digest(path)})
        license_row = next((row for row in rows if row["path"] == "LICENSE"), None)
        if license_row is None or license_row["head_object"].lower() != expected_license_blob:
            raise RuntimeError(f"pinned source LICENSE object drift: {root}")
        for role in roles:
            if role not in tracked:
                raise RuntimeError(f"missing tracked source role: {role}")
        return rows, license_row["head_object"]

    records: dict[str, Any] = {}
    for root, repo, revision, roles in ((source, SOURCE_REPOSITORY, SOURCE_REVISION, SOURCE_ROLES), (transformers, TRANSFORMERS_REPOSITORY, TRANSFORMERS_REVISION, TRANSFORMER_ROLES)):
        if git(root, "rev-parse", "HEAD") != revision or git(root, "status", "--porcelain", "--untracked-files=all"):
            raise RuntimeError(f"source checkout identity/cleanliness mismatch: {root}")
        origin = git(root, "remote", "get-url", "origin").rstrip("/")
        if origin.removesuffix(".git") != repo.removesuffix(".git"):
            raise RuntimeError(f"source origin mismatch: {root}")
        expected_license_blob = SOURCE_LICENSE_BLOB if repo == SOURCE_REPOSITORY else TRANSFORMER_LICENSE_BLOB
        expected_role_blobs = SOURCE_ROLE_BLOBS if repo == SOURCE_REPOSITORY else TRANSFORMER_ROLE_BLOBS
        if set(expected_role_blobs) != set(roles):
            raise RuntimeError(f"complete fixed role Git table is unavailable for {repo}")
        tracked_rows_value, license_blob = tracked_rows(root, roles, expected_license_blob)
        role_rows = [row for row in tracked_rows_value if row["path"] in roles]
        for row in role_rows:
            if row["head_object"].lower() != expected_role_blobs[row["path"]].lower():
                raise RuntimeError(f"fixed source role object drift: {row['path']}")
        license_path = root / "LICENSE"
        if not license_path.is_file():
            raise RuntimeError(f"pinned source license is missing: {root}")
        license_text = license_path.read_text(encoding="utf-8", errors="strict")
        if repo == SOURCE_REPOSITORY:
            marker = "MIT License"
            clauses = ("Permission is hereby granted, free of charge", "THE SOFTWARE IS PROVIDED \"AS IS\"")
        else:
            marker = "Apache License, Version 2.0"
            clauses = ("Licensed under the Apache License, Version 2.0", "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND")
        if marker not in license_text or any(clause not in license_text for clause in clauses):
            raise RuntimeError(f"pinned source license grant/warranty clauses missing: {repo}")
        records[repo] = {"repository": repo, "revision": revision, "origin": origin, "tracked_file_count": len(tracked_rows_value), "tracked_files": tracked_rows_value, "roles": role_rows, "license": {"path": "LICENSE", "bytes": license_path.stat().st_size, "sha256": digest(license_path), "git_blob_sha1": license_blob, "marker": marker, "clauses": list(clauses)}}
    source_text = "\n".join((source / role).read_text(encoding="utf-8", errors="strict") for role in SOURCE_ROLES)
    for marker in ("VibeVoiceForConditionalGeneration", "VibeVoiceTokenizer", "DPMSolverMultistepScheduler", "beta_schedule", "sample_speech_tokens", "speech_start_id", "speech_end_id", "speech_diffusion_id", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>"):
        if marker not in source_text:
            raise RuntimeError(f"official source marker missing: {marker}")
    scheduler_text = (source / "vibevoice/schedule/dpm_solver.py").read_text(encoding="utf-8", errors="strict")
    for marker in ("class DPMSolverMultistepScheduler", "algorithm_type", "dpmsolver++"):
        if marker not in scheduler_text:
            raise RuntimeError(f"official scheduler marker missing: {marker}")
    if "sde-dpmsolver++" in source_text or "sde-dpmsolver++" in scheduler_text:
        raise RuntimeError("unsupported SDE-DPMSolver contract found in official source")
    records["lineage"] = {
        "official_tts_revision": SOURCE_REVISION,
        "current_default_branch_not_used": True,
        "reason": "the Microsoft default branch reset removed the non-streaming TTS path; the pinned orphan retains it",
        "scheduler": "DPMSolverMultistepScheduler:dpmsolver++:squared-cosine (no SDE noise)",
    }
    return records


def config_evidence(snapshot: Path) -> dict[str, Any]:
    config = load_json(snapshot / "config.json")
    preprocessor = load_json(snapshot / "preprocessor_config.json")
    required = {"model_type": "vibevoice", "architectures": ["VibeVoiceForConditionalGeneration"], "acoustic_vae_dim": 64, "semantic_vae_dim": 128, "torch_dtype": "bfloat16"}
    for key, value in required.items():
        if config.get(key) != value:
            raise RuntimeError(f"config mismatch: {key}")
    acoustic = config.get("acoustic_tokenizer_config")
    if not isinstance(acoustic, dict):
        raise RuntimeError("acoustic_tokenizer_config missing")
    for key, value in {
        "model_type": "vibevoice_acoustic_tokenizer", "channels": 1,
        "causal": True, "vae_dim": 64, "corpus_normalize": 0.0,
        "fix_std": 0.5, "std_dist_type": "gaussian",
        "mixer_layer": "depthwise_conv", "conv_norm": "none",
        "pad_mode": "constant", "disable_last_norm": True,
        "layernorm": "RMSNorm", "layernorm_eps": 1e-5,
        "layernorm_elementwise_affine": True, "conv_bias": True,
        "layer_scale_init_value": 1e-6, "weight_init_value": 1e-2,
        "encoder_n_filters": 32, "encoder_ratios": [8, 5, 5, 4, 2, 2],
        "encoder_depths": "3-3-3-3-3-3-8", "decoder_n_filters": 32,
        "decoder_ratios": [8, 5, 5, 4, 2, 2], "decoder_depths": None,
    }.items():
        if acoustic.get(key) != value:
            raise RuntimeError(f"acoustic tokenizer config mismatch: {key}")
    semantic = config.get("semantic_tokenizer_config")
    if not isinstance(semantic, dict):
        raise RuntimeError("semantic_tokenizer_config missing")
    for key, value in {
        "model_type": "vibevoice_semantic_tokenizer", "channels": 1,
        "causal": True, "vae_dim": 128, "corpus_normalize": 0.0,
        "fix_std": 0, "std_dist_type": "none",
        "mixer_layer": "depthwise_conv", "conv_norm": "none",
        "pad_mode": "constant", "disable_last_norm": True,
        "layernorm": "RMSNorm", "layernorm_eps": 1e-5,
        "layernorm_elementwise_affine": True, "conv_bias": True,
        "layer_scale_init_value": 1e-6, "weight_init_value": 1e-2,
        "encoder_n_filters": 32, "encoder_ratios": [8, 5, 5, 4, 2, 2],
        "encoder_depths": "3-3-3-3-3-3-8",
    }.items():
        if semantic.get(key) != value:
            raise RuntimeError(f"semantic tokenizer config mismatch: {key}")
    decoder = config.get("decoder_config")
    if not isinstance(decoder, dict):
        raise RuntimeError("decoder_config missing")
    for key, value in {"hidden_size": 1536, "num_hidden_layers": 28, "num_attention_heads": 12, "num_key_value_heads": 2, "intermediate_size": 8960, "vocab_size": 151936, "max_position_embeddings": 65536, "rope_theta": 1_000_000, "rms_norm_eps": 1e-6, "tie_word_embeddings": True}.items():
        if decoder.get(key) != value:
            raise RuntimeError(f"decoder_config mismatch: {key}")
    diffusion = config.get("diffusion_head_config")
    if not isinstance(diffusion, dict):
        raise RuntimeError("diffusion_head_config missing")
    for key, value in {"hidden_size": 1536, "head_layers": 4, "latent_size": 64, "speech_vae_dim": 64, "prediction_type": "v_prediction", "diffusion_type": "ddpm", "ddpm_num_steps": 1000, "ddpm_num_inference_steps": 20, "ddpm_beta_schedule": "cosine"}.items():
        if diffusion.get(key) != value:
            raise RuntimeError(f"diffusion_head_config mismatch: {key}")
    if preprocessor.get("processor_class") != "VibeVoiceProcessor" or preprocessor.get("language_model_pretrained_name") != "Qwen/Qwen2.5-1.5B":
        raise RuntimeError("preprocessor does not pin the official processor/Qwen companion")
    if preprocessor.get("speech_tok_compress_ratio") != 3200 or preprocessor.get("audio_processor", {}).get("sampling_rate") != 24000:
        raise RuntimeError("preprocessor audio axes mismatch")
    return {"config": config, "preprocessor": preprocessor, "qwen_companion": f"{QWEN_REPOSITORY}@{QWEN_REVISION} (authenticated in a separate fixed tree; native integration remains gated)"}


def qwen_tokenizer_evidence(snapshot: Path) -> dict[str, Any]:
    tokenizer_config = load_json(snapshot / "tokenizer_config.json")
    tokenizer = load_json(snapshot / "tokenizer.json")
    vocab = load_json(snapshot / "vocab.json")
    if not isinstance(tokenizer_config, dict) or tokenizer_config.get("eos_token") != "<|endoftext|>":
        raise RuntimeError("Qwen tokenizer config does not authenticate end-of-text token")
    if tokenizer_config.get("pad_token") != "<|endoftext|>" or tokenizer_config.get("split_special_tokens") is not False:
        raise RuntimeError("Qwen tokenizer padding/splitting contract mismatch")
    if not isinstance(tokenizer, dict) or not isinstance(tokenizer.get("model"), dict) or tokenizer["model"].get("type") != "BPE":
        raise RuntimeError("Qwen tokenizer.json is not the authenticated BPE pipeline")
    model_vocab = tokenizer["model"].get("vocab")
    merges = tokenizer["model"].get("merges")
    if not isinstance(model_vocab, dict) or len(model_vocab) != 151_936 or any(not isinstance(value, int) or isinstance(value, bool) for value in model_vocab.values()) or set(model_vocab.values()) != set(range(151_936)) or not isinstance(merges, list) or not merges:
        raise RuntimeError("Qwen tokenizer model vocabulary/merges missing")
    if not isinstance(vocab, dict) or len(vocab) != 151_936 or any(not isinstance(value, int) or isinstance(value, bool) for value in vocab.values()) or set(vocab.values()) != set(range(151_936)):
        raise RuntimeError("Qwen vocab.json does not authenticate the 151936-token vocabulary")
    license_path = snapshot / "LICENSE"
    license_text = license_path.read_text(encoding="utf-8", errors="strict")
    lowered_license = license_text.lower()
    if not all(marker in lowered_license for marker in ("apache license, version 2.0", "you may obtain a copy", "distributed under the license", "without warranties or conditions")):
        raise RuntimeError("Qwen companion license is not authenticated as Apache-2.0")
    return {
        "repository": QWEN_REPOSITORY,
        "revision": QWEN_REVISION,
        "files": sorted(QWEN_FILES),
        "tokenizer_config_sha256": digest(snapshot / "tokenizer_config.json"),
        "tokenizer_json_sha256": digest(snapshot / "tokenizer.json"),
        "vocab_sha256": digest(snapshot / "vocab.json"),
        "vocab_size": len(vocab),
        "merge_count": len(merges),
        "license": {"path": "LICENSE", "sha256": digest(license_path), "marker": "Apache License Version 2.0"},
    }


def parse_model_card_frontmatter(text: str) -> dict[str, str]:
    """Read one top-level scalar ``license: mit`` without parsing all YAML."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise RuntimeError("model-card frontmatter is missing")
    try:
        end = next(index for index in range(1, len(lines)) if lines[index].strip() == "---")
    except StopIteration as error:
        raise RuntimeError("model-card frontmatter is unterminated") from error
    seen: set[str] = set()
    result: dict[str, str] = {}
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#") or line[0].isspace() or line.startswith("-"):
            continue
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_-]*):[ \t]*(.*)", line)
        if match is None:
            raise RuntimeError("model-card frontmatter has malformed top-level YAML")
        key, raw_value = match.groups()
        if key in seen:
            raise RuntimeError(f"model-card frontmatter key duplicated: {key}")
        seen.add(key)
        if key == "license":
            value = raw_value.strip().strip("\"'")
            if value != "mit":
                raise RuntimeError("model-card license is not exactly mit")
            result[key] = value
    if result.get("license") != "mit":
        raise RuntimeError("model-card license is not exactly one top-level mit")
    return result


def model_card_evidence(snapshot: Path) -> dict[str, Any]:
    path = snapshot / "README.md"
    parsed = parse_model_card_frontmatter(path.read_text(encoding="utf-8", errors="strict"))
    return {"path": "README.md", "bytes": path.stat().st_size, "sha256": digest(path), "license": "MIT", "frontmatter": parsed, "status": "AUTHENTICATED_MODEL_CARD"}


def blocked(output: Path, error: Exception, inspection_status: str = "INSPECTION_ERROR", **extra: Any) -> None:
    output.mkdir(parents=True, exist_ok=True)
    collection_status = "AUTHENTICATED" if inspection_status == "AUTHENTICATED_EVIDENCE_COMPLETE" else "UNVERIFIED"
    manifest = {"format": FORMAT, "status": "BLOCKED", "evidence_stage": "INSPECTION_ONLY", "inspection_status": inspection_status, "collection_status": collection_status, "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "cpu_status": "UNSUPPORTED", "metal_status": "BLOCKED_BY_CPU", "parity_status": "NOT_RUN", "publication": "NO_UPLOAD", "upstream": {"repository": HF_REPOSITORY, "requested_revision": HF_REVISION, "resolved_revision": HF_REVISION if collection_status == "AUTHENTICATED" else None}, "public_artifact": {"repository": PUBLIC_REPOSITORY, "revision": PUBLIC_REVISION, "bytes": PUBLIC_BYTES, "sha256": PUBLIC_SHA256, "manifest_sha256": PUBLIC_MANIFEST_SHA256}, "error_type": type(error).__name__, "reason": str(error), "blockers": [str(error)], **extra}
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def inspect(snapshot: Path, source: Path, transformers: Path, server_tree: Path, output: Path, qwen_snapshot: Path, qwen_tree: Path) -> int:
    files = server_inventory(snapshot, server_tree)
    qwen_files = server_inventory(qwen_snapshot, qwen_tree, QWEN_REPOSITORY, QWEN_REVISION, QWEN_FILES)
    qwen = qwen_tokenizer_evidence(qwen_snapshot)
    model_card = model_card_evidence(snapshot)
    config = config_evidence(snapshot)
    index = load_json(snapshot / "model.safetensors.index.json")
    metadata = index.get("metadata") if isinstance(index, dict) else None
    if not isinstance(metadata, dict) or metadata.get("total_size") != 5_408_043_974:
        raise RuntimeError("HF index total_size does not authenticate the fixed 1.5B snapshot")
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or len(weight_map) != 1204 or any(not isinstance(key, str) or not isinstance(value, str) for key, value in weight_map.items()):
        raise RuntimeError("HF index does not authenticate the exact 1,204 tensor map")
    all_rows: list[dict[str, Any]] = []
    shard_evidence = []
    for shard in sorted({value for value in weight_map.values()}):
        if shard not in MODEL_FILES or not shard.endswith(".safetensors"):
            raise RuntimeError(f"index references unexpected shard: {shard}")
        evidence, rows = inspect_safetensors(snapshot / shard)
        all_rows.extend(rows)
        shard_evidence.append(evidence)
        if {row["name"] for row in rows} != {key for key, value in weight_map.items() if value == shard}:
            raise RuntimeError(f"index/header tensor set mismatch: {shard}")
    if len(all_rows) != 1204 or {row["name"] for row in all_rows} != set(weight_map):
        raise RuntimeError("complete tensor manifest mismatch")
    source = source_inventory(source, transformers)
    output.mkdir(parents=True, exist_ok=True)
    evidence = {"server-tree.json": {"repository": HF_REPOSITORY, "requested_revision": HF_REVISION, "resolved_revision": HF_REVISION, "files": files}, "qwen-server-tree.json": {"repository": QWEN_REPOSITORY, "requested_revision": QWEN_REVISION, "resolved_revision": QWEN_REVISION, "files": qwen_files}, "qwen-tokenizer.json": qwen, "model-card.json": model_card, "config.json": config, "tensor-inventory.json": {"shards": shard_evidence, "tensor_count": len(all_rows), "manifest_sha256": manifest_sha256(all_rows)}, "source-inventory.json": source}
    for name, value in evidence.items():
        (output / name).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    blocked(output, RuntimeError("official Qwen tokenizer is authenticated but its VibeVoice prompt/prefill integration, streaming tokenizer, and native DPMSolverMultistepScheduler remain unbound; no CPU/Metal/parity claim"), inspection_status="AUTHENTICATED_EVIDENCE_COMPLETE", evidence_files=sorted(evidence), tensor_count=len(all_rows), tensor_manifest_sha256=manifest_sha256(all_rows), qwen_tokenizer="AUTHENTICATED_TREE_UNBOUND_RUNTIME", scheduler="DPMSolverMultistepScheduler:dpmsolver++:squared-cosine")
    return 2


def self_test() -> None:
    assert len(HF_REVISION) == len(SOURCE_REVISION) == len(QWEN_REVISION) == len(TRANSFORMERS_REVISION) == 40
    assert PUBLIC_BYTES > 5_000_000_000 and len(PUBLIC_SHA256) == len(PUBLIC_MANIFEST_SHA256) == 64
    assert set(MODEL_IDENTITIES) == MODEL_FILES and set(QWEN_IDENTITIES) == QWEN_FILES
    assert all(isinstance(identity, FixedFile) and identity.kind in {"regular", "lfs"} and identity.size >= 0 for identity in (*MODEL_IDENTITIES.values(), *QWEN_IDENTITIES.values()))
    assert all(identity.kind == "lfs" for name, identity in MODEL_IDENTITIES.items() if name.endswith(".safetensors"))
    assert set(SOURCE_ROLE_BLOBS) == set(SOURCE_ROLES) and set(TRANSFORMER_ROLE_BLOBS) == set(TRANSFORMER_ROLES)
    assert all(re.fullmatch(r"[0-9a-f]{40}", value) for value in (*SOURCE_ROLE_BLOBS.values(), *TRANSFORMER_ROLE_BLOBS.values()))
    try:
        json.loads('{"x":1,"x":2}', object_pairs_hook=strict_pairs)
    except RuntimeError:
        pass
    else:
        raise AssertionError("duplicate JSON accepted")
    with tempfile.TemporaryDirectory(prefix="vokra-vibevoice-1-5b-") as directory:
        root = Path(directory)
        config_path = root / "config.json"
        config_path.write_text(json.dumps({"model_type": "vibevoice", "architectures": ["VibeVoiceForConditionalGeneration"], "acoustic_vae_dim": 64, "semantic_vae_dim": 128, "torch_dtype": "bfloat16", "acoustic_tokenizer_config": {"model_type": "vibevoice_acoustic_tokenizer", "channels": 1, "causal": True, "vae_dim": 64, "corpus_normalize": 0.0, "fix_std": 0.5, "std_dist_type": "gaussian", "mixer_layer": "depthwise_conv", "conv_norm": "none", "pad_mode": "constant", "disable_last_norm": True, "layernorm": "RMSNorm", "layernorm_eps": 1e-5, "layernorm_elementwise_affine": True, "conv_bias": True, "layer_scale_init_value": 1e-6, "weight_init_value": 1e-2, "encoder_n_filters": 32, "encoder_ratios": [8, 5, 5, 4, 2, 2], "encoder_depths": "3-3-3-3-3-3-8", "decoder_n_filters": 32, "decoder_ratios": [8, 5, 5, 4, 2, 2], "decoder_depths": None}, "semantic_tokenizer_config": {"model_type": "vibevoice_semantic_tokenizer", "channels": 1, "causal": True, "vae_dim": 128, "corpus_normalize": 0.0, "fix_std": 0, "std_dist_type": "none", "mixer_layer": "depthwise_conv", "conv_norm": "none", "pad_mode": "constant", "disable_last_norm": True, "layernorm": "RMSNorm", "layernorm_eps": 1e-5, "layernorm_elementwise_affine": True, "conv_bias": True, "layer_scale_init_value": 1e-6, "weight_init_value": 1e-2, "encoder_n_filters": 32, "encoder_ratios": [8, 5, 5, 4, 2, 2], "encoder_depths": "3-3-3-3-3-3-8"}, "decoder_config": {"hidden_size": 1536, "num_hidden_layers": 28, "num_attention_heads": 12, "num_key_value_heads": 2, "intermediate_size": 8960, "vocab_size": 151936, "max_position_embeddings": 65536, "rope_theta": 1000000.0, "rms_norm_eps": 1e-6, "tie_word_embeddings": True}, "diffusion_head_config": {"hidden_size": 1536, "head_layers": 4, "latent_size": 64, "speech_vae_dim": 64, "prediction_type": "v_prediction", "diffusion_type": "ddpm", "ddpm_num_steps": 1000, "ddpm_num_inference_steps": 20, "ddpm_beta_schedule": "cosine"}}), encoding="utf-8")
        (root / "preprocessor_config.json").write_text(json.dumps({"processor_class": "VibeVoiceProcessor", "speech_tok_compress_ratio": 3200, "language_model_pretrained_name": "Qwen/Qwen2.5-1.5B", "audio_processor": {"sampling_rate": 24000}}), encoding="utf-8")
        (root / "README.md").write_text("---\nlicense: mit\ndatasets:\n- example/dataset\ntags:\n- audio\nmodel-index:\n  - name: fixture\n---\nVibeVoice fixture\n", encoding="utf-8")
        assert model_card_evidence(root)["status"] == "AUTHENTICATED_MODEL_CARD"
        try:
            parse_model_card_frontmatter("---\nlicense: mit\nlicense: apache-2.0\n---\n")
        except RuntimeError:
            pass
        else:
            raise AssertionError("duplicate model-card license accepted")
        try:
            parse_model_card_frontmatter("---\ntags:\n- audio\n---\n")
        except RuntimeError:
            pass
        else:
            raise AssertionError("missing model-card license accepted")
        assert config_evidence(root)["qwen_companion"].startswith("Qwen/Qwen2.5-1.5B")
        bad_config = json.loads(config_path.read_text(encoding="utf-8")); bad_config["decoder_config"]["hidden_size"] = 896; config_path.write_text(json.dumps(bad_config), encoding="utf-8")
        try:
            config_evidence(root)
        except RuntimeError:
            pass
        else:
            raise AssertionError("decoder topology drift accepted")
        qwen = root / "qwen"; qwen.mkdir()
        qwen_vocab = {f"token{i}": i for i in range(151_936)}
        (qwen / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|endoftext|>", "pad_token": "<|endoftext|>", "split_special_tokens": False}), encoding="utf-8")
        (qwen / "tokenizer.json").write_text(json.dumps({"model": {"type": "BPE", "vocab": qwen_vocab, "merges": ["token0 token1"]}}), encoding="utf-8")
        (qwen / "vocab.json").write_text(json.dumps(qwen_vocab), encoding="utf-8")
        (qwen / "LICENSE").write_text("Apache License, Version 2.0\nYou may obtain a copy of the License.\nDistributed under the License.\nWITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.\n", encoding="utf-8")
        assert qwen_tokenizer_evidence(qwen)["vocab_size"] == 151_936
        qwen_config = json.loads((qwen / "tokenizer_config.json").read_text(encoding="utf-8")); qwen_config["pad_token"] = "<bad>"; (qwen / "tokenizer_config.json").write_text(json.dumps(qwen_config), encoding="utf-8")
        try:
            qwen_tokenizer_evidence(qwen)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Qwen tokenizer drift accepted")
        good = root / "good.safetensors"
        raw = b'{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
        good.write_bytes(len(raw).to_bytes(8, "little") + raw + b"\0" * 4)
        evidence, rows = inspect_safetensors(good)
        assert evidence["tensor_count"] == 1 and rows[0]["shape"] == [1]
        bad = root / "bad.safetensors"
        raw = b'{"../bad":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
        bad.write_bytes(len(raw).to_bytes(8, "little") + raw + b"\0" * 4)
        try:
            inspect_safetensors(bad)
        except RuntimeError:
            pass
        else:
            raise AssertionError("unsafe tensor path accepted")
        snapshot = root / "snapshot"; snapshot.mkdir()
        rows = []
        for name in sorted(MODEL_FILES):
            path = snapshot / name; path.write_bytes(name.encode()); rows.append({"path": name, "type": "file", "size": path.stat().st_size, "git_blob_sha1": git_blob_sha1(path), "lfs_pointer_git_blob_sha1": None, "lfs_payload_sha256": None, "lfs_payload_size": None})
        packet = root / "tree.json"; packet.write_text(json.dumps({"repository": HF_REPOSITORY, "requested_revision": HF_REVISION, "resolved_revision": HF_REVISION, "walk": "recursive_file_only", "files": rows}), encoding="utf-8")
        assert len(server_inventory(snapshot, packet, check_fixed_sizes=False)) == len(MODEL_FILES)
        try:
            server_inventory(snapshot, packet)
        except RuntimeError as error:
            assert "fixed server file size mismatch" in str(error)
        else:
            raise AssertionError("self-described fixed server identity accepted")
        # Exercise the LFS content-vs-Git-pointer distinction without
        # materialising any model-sized body.
        lfs_path = snapshot / ".gitattributes"; lfs_path.write_bytes(b"payload")
        lfs_sha = digest(lfs_path); lfs_row = next(row for row in rows if row["path"] == ".gitattributes")
        lfs_row.update({"size": 7, "git_blob_sha1": None, "lfs_pointer_git_blob_sha1": git_blob_sha1_bytes(lfs_pointer(lfs_sha, 7)), "lfs_payload_sha256": lfs_sha, "lfs_payload_size": 7})
        packet.write_text(json.dumps({"repository": HF_REPOSITORY, "requested_revision": HF_REVISION, "resolved_revision": HF_REVISION, "walk": "recursive_file_only", "files": rows}), encoding="utf-8")
        assert len(server_inventory(snapshot, packet, check_fixed_sizes=False)) == len(MODEL_FILES)
        lfs_path.write_bytes(b"payload")
        (snapshot / "README.md").write_bytes(b"README.MD")
        try:
            server_inventory(snapshot, packet, check_fixed_sizes=False)
        except RuntimeError:
            pass
        else:
            raise AssertionError("same-size/content mutation accepted")
        (snapshot / "README.md").write_bytes(b"README.md")
        cache = snapshot / ".cache" / "huggingface"; cache.mkdir(parents=True)
        (cache / "transport.index").write_bytes(b"ignored transport cache")
        assert local_files(snapshot) == set(MODEL_FILES)
        (snapshot / ".cache" / "other").mkdir()
        try:
            local_files(snapshot)
        except RuntimeError:
            pass
        else:
            raise AssertionError("nested non-transport cache accepted")
        (snapshot / ".cache" / "other").rmdir()
        (snapshot / "payload-link").symlink_to(snapshot / "README.md")
        try:
            local_files(snapshot)
        except RuntimeError:
            pass
        else:
            raise AssertionError("payload symlink accepted")
        (snapshot / "payload-link").unlink()
        rows[-1]["path"] = "extra"
        packet.write_text(json.dumps({"repository": HF_REPOSITORY, "requested_revision": HF_REVISION, "resolved_revision": HF_REVISION, "walk": "recursive_file_only", "files": rows}), encoding="utf-8")
        try:
            server_inventory(snapshot, packet, check_fixed_sizes=False)
        except RuntimeError:
            pass
        else:
            raise AssertionError("incomplete fixed tree accepted")
        error = root / "error"; blocked(error, RuntimeError("fixture")); manifest = load_json(error / "manifest.json"); assert manifest["inspection_status"] == "INSPECTION_ERROR" and manifest["status"] == "BLOCKED"
    print("vibevoice_1_5b_inspect.py self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--snapshot", type=Path); parser.add_argument("--source", type=Path); parser.add_argument("--transformers", type=Path); parser.add_argument("--server-tree", type=Path); parser.add_argument("--qwen-snapshot", type=Path); parser.add_argument("--qwen-tree", type=Path); parser.add_argument("--output", type=Path); parser.add_argument("--self-test", action="store_true"); args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.snapshot, args.source, args.transformers, args.server_tree, args.qwen_snapshot, args.qwen_tree, args.output)): parser.error("--self-test accepts no paths")
        self_test(); return 0
    if any(value is None for value in (args.snapshot, args.source, args.transformers, args.server_tree, args.qwen_snapshot, args.qwen_tree, args.output)): parser.error("all inspection paths are required")
    try:
        return inspect(args.snapshot, args.source, args.transformers, args.server_tree, args.output, args.qwen_snapshot, args.qwen_tree)
    except Exception as error:
        blocked(args.output, error); print(f"VibeVoice 1.5B inspection BLOCKED: {error}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
