#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Inspect the fixed HF Sortformer tree without conversion or unpickling.

The official tree contains both ``model.safetensors`` and a ``.nemo`` tar.
This tool hashes and inventories both. It never extracts or loads pickle
payloads; an internal checkpoint is recorded as a blocker because no
``torch.load(..., weights_only=True, map_location="cpu")`` bridge exists.
The NeMo commit below is an independently reviewed public execution
reference. It is not weight-build provenance because the model card pins a
mutable ``NeMo@main`` lineage.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import stat
import sys
import tarfile
import subprocess
import tempfile
import struct
from pathlib import Path, PurePosixPath
from typing import Any

UPSTREAM_HF = "nvidia/diar_sortformer_4spk-v1"
HF_REVISION = "9f17b10df44c0a4c8f3c86fbddc9ee2d6ab9ac08"
SOURCE_REPOSITORY = "https://github.com/NVIDIA/NeMo.git"
SOURCE_REVISION = "505acacf6444a67ff9a4020fb03a5e6d59953e05"
SOURCE_ROLE_BLOBS = {
    "examples/speaker_tasks/diarization/conf/neural_diarizer/sortformer_diarizer_hybrid_loss_4spk-v1.yaml": "66cfc5fd1b61a7da870b8930764d02ac7c953a86",
    "examples/speaker_tasks/diarization/neural_diarizer/sortformer_diar_train.py": "ab6e418b107299a774232043c7aa08671947811a",
    "nemo/collections/asr/data/audio_to_diar_label.py": "0824c9c6ab51329430a812c8dd179fe895d54cba",
    "nemo/collections/asr/losses/bce_loss.py": "36a7a0166f2669cc632c1163d34645d683feebe3",
    "nemo/collections/asr/metrics/der.py": "c8dec24eaaca0849bba003b57c4aa81892b9995b",
    "nemo/collections/asr/models/sortformer_diar_models.py": "f6b0eab4c8950a6271ca71811a036e2c0bb9101e",
    "nemo/collections/asr/modules/conformer_encoder.py": "27d0cde33f8c1ec4bac21ce14d99c31944c298fb",
    "nemo/collections/asr/modules/sortformer_modules.py": "d99bf3b93e38ed696110b754e64097c8e688aa78",
    "nemo/collections/asr/parts/utils/asr_multispeaker_utils.py": "66cfcc75f49f472394757e9b785c82ca9402dd41",
    "nemo/collections/asr/parts/utils/speaker_utils.py": "223916e60a76183cda8fbbd97038315d8f0c0fbc",
    "LICENSE": "f49a4e16e68b128803cc2dcea614603632b04eac",
}
MODEL_SHA256 = "e8abcc5f3a82ff23134c98a37f70fef3f159611f394bb191a0ad0a6f4b052974"
MODEL_BYTES = 494206256
NEMO_SHA256 = "bc74dfd8ca314240abcdc7e2949901eeaa72947a04ce1fab893e373d81f1e689"
NEMO_BYTES = 493434880
FORMAT = "vokra-sortformer-diar-4spk-v2-inspection-v1"
ARRIVAL_ORDER_SENTENCE = (
    "Sortformer resolves permutation problem in diarization following the arrival-time order "
    "of the speech segments from each speaker."
)
EXPECTED_FILES = {
    ".gitattributes",
    "README.md",
    "config.json",
    "diar_sortformer_4spk-v1.nemo",
    "model.safetensors",
    "processor_config.json",
    "sortformer-v1-model.png",
    "sortformer_intro.png",
}
MAX_HEADER_BYTES = 64 * 1024 * 1024
MAX_NEMO_MEMBERS = 4096
MAX_NEMO_MEMBER_BYTES = 2 * NEMO_BYTES
MAX_NEMO_TOTAL_MEMBER_BYTES = 4 * NEMO_BYTES
HF_TRANSPORT_CACHE = ".cache/huggingface"
HF_CHRONOLOGY = {
    "original_nemo_commit": "5bd87d8c7e6fa303c6d9338f85a5e158537627e1",
    "safetensors_config_commit": "1dd84ea9d2126353c2ba61dc72fff27be68e00b1",
    "current_commit": HF_REVISION,
}
HEX40 = re.compile(r"^[0-9a-fA-F]{40}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1_bytes(data: bytes) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {len(data)}\0".encode())
    digest.update(data)
    return digest.hexdigest()


def lfs_pointer_bytes(payload_sha256: str, payload_bytes: int) -> bytes:
    return f"version https://git-lfs.github.com/spec/v1\noid sha256:{payload_sha256}\nsize {payload_bytes}\n".encode()


def parse_model_card_frontmatter(text: str) -> dict[str, str]:
    """Authenticate one top-level scalar license without pretending to parse YAML."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise RuntimeError("model card frontmatter is missing")
    try:
        end = next(index for index in range(1, len(lines)) if lines[index].strip() == "---")
    except StopIteration as error:
        raise RuntimeError("model card frontmatter is unterminated") from error
    result: dict[str, str] = {}
    seen_keys: set[str] = set()
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        # Lists and mappings belonging to a top-level field are indented in
        # the canonical card (datasets, tags, widget, model-index, ...).
        # They are deliberately tolerated but never interpreted as license.
        if line[0].isspace():
            continue
        # YAML permits an indentless sequence immediately below a mapping
        # key (the canonical card uses this for datasets/tags/widget).
        if line.startswith("-"):
            continue
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_-]*):[ \t]*(.*)", line)
        if match is None:
            raise RuntimeError("model card frontmatter has malformed top-level YAML")
        key, raw_value = match.groups()
        if key in seen_keys:
            raise RuntimeError(f"model card frontmatter key is duplicated: {key}")
        seen_keys.add(key)
        if key == "license":
            if not raw_value.strip():
                raise RuntimeError("model card frontmatter license is duplicated or non-scalar")
            value = raw_value.strip().strip('"\'')
            if value != "cc-by-nc-4.0":
                raise RuntimeError("model card frontmatter license is not exactly cc-by-nc-4.0")
            result[key] = value
    if result.get("license") != "cc-by-nc-4.0":
        raise RuntimeError("model card frontmatter license is not exactly cc-by-nc-4.0")
    return result


def source_selection_status(source_blockers: list[str]) -> str:
    return "REFERENCE_SOURCE_SELECTED" if not source_blockers else "BLOCKED"


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise RuntimeError(f"missing snapshot: {root}")
    result=[]
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative == ".cache":
            if path.is_symlink() or not path.is_dir():
                raise RuntimeError(f"cache parent must be a real directory: {path}")
            continue
        if relative == HF_TRANSPORT_CACHE:
            if path.is_symlink() or not path.is_dir():
                raise RuntimeError(f"transport cache must be a real directory: {path}")
            continue
        if relative.startswith(HF_TRANSPORT_CACHE + "/"):
            continue
        if ".cache" in path.relative_to(root).parts:
            raise RuntimeError(f"unexpected nested cache outside {HF_TRANSPORT_CACHE}: {path}")
        if path.is_dir() and not path.is_symlink():
            continue
        if path.is_symlink():
            if not path.exists() or not path.is_file():
                raise RuntimeError(f"dangling/nonregular symlink: {path}")
            raise RuntimeError(f"symlink is not an authenticated regular file: {path}")
        if not path.is_file():
            raise RuntimeError(f"nonregular snapshot member: {path}")
        result.append(path)
    if not result:
        raise RuntimeError("empty snapshot")
    return result


def tensor_inventory(checkpoint: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with checkpoint.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise RuntimeError("safetensors header prefix is truncated")
        header_length = int.from_bytes(raw_length, "little")
        if header_length > MAX_HEADER_BYTES:
            raise RuntimeError("safetensors header exceeds bounded cap")
        header = stream.read(header_length)
        if len(header) != header_length:
            raise RuntimeError("safetensors header is truncated")
    try:
        root = json.loads(header, object_pairs_hook=_no_duplicate_keys)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"safetensors header is not JSON: {error}") from error
    if not isinstance(root, dict):
        raise RuntimeError("safetensors header root is not an object")
    metadata = root.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise RuntimeError("safetensors __metadata__ must be a string map")
    data_base = 8 + header_length
    data_size = checkpoint.stat().st_size - data_base
    if data_size < 0:
        raise RuntimeError("safetensors has no data region")
    rows: list[dict[str, Any]] = []
    ranges: list[tuple[int, int, str]] = []
    dtype_sizes = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1}
    for name in sorted(root):
        entry = root[name]
        if not isinstance(name, str) or not name or "\x00" in name or "\\" in name or PurePosixPath(name).is_absolute() or any(part in {"", ".", ".."} for part in PurePosixPath(name).parts) or not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data_offsets"}:
            raise RuntimeError(f"malformed tensor entry: {name!r}")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if not isinstance(dtype, str) or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise RuntimeError(f"malformed tensor metadata: {name}")
        if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in shape + offsets):
            raise RuntimeError(f"negative/non-integer tensor metadata: {name}")
        begin, end = offsets
        if begin > end or end > data_size:
            raise RuntimeError(f"tensor ranges overlap or contain a gap: {name}")
        size = dtype_sizes.get(dtype)
        if size is None:
            raise RuntimeError(f"unsupported tensor dtype: {dtype}")
        elements = 1
        for dimension in shape:
            elements *= dimension
        if end - begin != elements * size:
            raise RuntimeError(f"tensor byte size does not match shape*dtype: {name}")
        ranges.append((begin, end, name))
        rows.append({"name": name, "dtype": dtype, "shape": shape, "data_offsets": offsets, "bytes": end - begin})
    previous_end = 0
    for begin, end, name in sorted(ranges):
        if begin != previous_end:
            raise RuntimeError(f"tensor ranges overlap or contain a gap: {name}")
        previous_end = end
    if previous_end != data_size:
        raise RuntimeError("safetensors data region has a trailing gap")
    if not rows:
        raise RuntimeError("checkpoint has no tensors")
    return metadata, rows


def validate_nemo_archive(source: Path | io.BytesIO) -> tuple[list[dict[str, Any]], list[str]]:
    """List a `.nemo` tar safely; return members and non-fatal blockers."""

    def reject_embedded_nul(stream: Any) -> None:
        """Reject raw tar names that tarfile would truncate at an embedded NUL."""
        position = stream.tell()
        prefix = stream.read(6)
        stream.seek(position)
        if prefix[:2] in {b"\x1f\x8b", b"BZ"} or prefix.startswith(b"\xfd7zXZ"):
            return
        while True:
            header = stream.read(512)
            if not header or header == b"\0" * 512:
                break
            if len(header) != 512:
                raise RuntimeError("truncated NeMo tar header")
            for field in (header[:100], header[345:500]):
                nul = field.find(b"\0")
                if nul >= 0 and any(field[nul + 1 :]):
                    raise RuntimeError("embedded NUL in NeMo archive member name")
            size_field = header[124:136].rstrip(b"\0 ")
            try:
                size = int(size_field or b"0", 8)
            except ValueError as error:
                raise RuntimeError("invalid NeMo tar member size") from error
            stream.seek(((size + 511) // 512) * 512, io.SEEK_CUR)

    seen: set[str] = set()
    members: list[dict[str, Any]] = []
    blockers: list[str] = []
    total_member_bytes = 0
    if isinstance(source, Path):
        with source.open("rb") as raw:
            reject_embedded_nul(raw)
        archive_context = tarfile.open(source, mode="r:*")
    else:
        source.seek(0)
        reject_embedded_nul(source)
        source.seek(0)
        archive_context = tarfile.open(fileobj=source, mode="r:*")
    with archive_context as archive:
        for member in archive:
            name = member.name
            path = PurePosixPath(name)
            if len(members) >= MAX_NEMO_MEMBERS:
                raise RuntimeError("NeMo archive exceeds bounded member count")
            if "\x00" in name or "\\" in name or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts) or not name:
                raise RuntimeError(f"unsafe NeMo archive path: {name!r}")
            if name in seen:
                raise RuntimeError(f"duplicate NeMo archive member: {name!r}")
            seen.add(name)
            if member.issym() or member.islnk() or member.isdev() or not (member.isfile() or member.isdir()):
                raise RuntimeError(f"unsafe NeMo archive member type: {name!r}")
            if member.size > MAX_NEMO_MEMBER_BYTES:
                raise RuntimeError(f"NeMo archive member exceeds bounded size: {name}")
            total_member_bytes += member.size
            if total_member_bytes > MAX_NEMO_TOTAL_MEMBER_BYTES:
                raise RuntimeError("NeMo archive cumulative member size exceeds bounded cap")
            suffix = Path(name).suffix.lower()
            if suffix in {".pkl", ".pickle", ".pt", ".pth", ".ckpt"}:
                blockers.append(f"unsafe pickle checkpoint payload requires weights_only=True: {name}")
            member_digest = None
            if member.isfile():
                stream = archive.extractfile(member)
                if stream is None:
                    raise RuntimeError(f"NeMo archive member cannot be read: {name}")
                digest = hashlib.sha256()
                read_bytes = 0
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    read_bytes += len(block)
                    if read_bytes > member.size:
                        raise RuntimeError(f"NeMo archive member read exceeds declared size: {name}")
                    digest.update(block)
                if read_bytes != member.size:
                    raise RuntimeError(f"NeMo archive member read-size mismatch: {name}")
                member_digest = digest.hexdigest()
            members.append({"name": name, "bytes": member.size, "type": "directory" if member.isdir() else "file", "sha256": member_digest})
    return members, blockers


def parse_config(config: Path) -> dict[str, Any]:
    try:
        value = json.loads(config.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"official config.json is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError("official config.json root is not an object")
    return value


def server_tree(snapshot: Path, packet: Path, blockers: list[str]) -> dict[str, Any]:
    local_blockers: list[str] = []
    try:
        remote = json.loads(packet.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)
    except Exception as error:
        local_blockers.append(f"server tree parse failed: {error}")
        blockers.extend(local_blockers)
        return {"status": "BLOCKED"}
    schema_ok = isinstance(remote, dict) and set(remote) == {"repository", "requested_revision", "resolved_revision", "walk", "files"}
    if not schema_ok:
        local_blockers.append("server tree top-level schema mismatch")
    rows = remote.get("files") if isinstance(remote, dict) else None
    if not isinstance(rows, list):
        local_blockers.append("server tree files must be a list")
        rows = []
    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "type", "size", "git_blob_sha1", "lfs_pointer_git_blob_sha1", "lfs_sha256"}:
            local_blockers.append("server tree row schema mismatch")
            continue
        name, size = row["path"], row["size"]
        path = PurePosixPath(name) if isinstance(name, str) else PurePosixPath(".")
        if row["type"] != "file" or not isinstance(name, str) or not name or "\x00" in name or "\\" in name or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts) or not isinstance(size, int) or isinstance(size, bool) or size < 0:
            local_blockers.append(f"server tree unsafe path/type/size: {name!r}")
            continue
        git_sha, pointer_sha, lfs_sha = row["git_blob_sha1"], row["lfs_pointer_git_blob_sha1"], row["lfs_sha256"]
        if git_sha is not None and (not isinstance(git_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", git_sha)): local_blockers.append(f"invalid regular Git blob SHA-1: {name}")
        if pointer_sha is not None and (not isinstance(pointer_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", pointer_sha)): local_blockers.append(f"invalid LFS pointer Git blob SHA-1: {name}")
        if lfs_sha is not None and (not isinstance(lfs_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", lfs_sha)): local_blockers.append(f"invalid LFS payload SHA-256: {name}")
        if (lfs_sha is None and (git_sha is None or pointer_sha is not None)) or (lfs_sha is not None and (git_sha is not None or pointer_sha is None)): local_blockers.append(f"regular/LFS identities are not distinct: {name}")
        if name in records: local_blockers.append(f"duplicate server tree path: {name}"); continue
        records[name] = row
    if set(records) != EXPECTED_FILES:
        local_blockers.append(f"server tree exact file set mismatch: {sorted(set(records) ^ EXPECTED_FILES)}")
    local = files(snapshot)
    local_names = {path.relative_to(snapshot).as_posix(): path for path in local}
    missing = sorted(set(records) - set(local_names)); extra = sorted(set(local_names) - set(records))
    mismatched=[]
    for name in sorted(set(records) & set(local_names)):
        row, path = records[name], local_names[name]
        payload_size = path.stat().st_size
        payload_sha = sha256(path)
        mismatch = payload_size != row["size"]
        if row["lfs_sha256"] is None:
            mismatch = mismatch or git_blob_sha1(path) != row["git_blob_sha1"]
        else:
            mismatch = mismatch or payload_sha != row["lfs_sha256"] or git_blob_sha1_bytes(lfs_pointer_bytes(payload_sha, payload_size)) != row["lfs_pointer_git_blob_sha1"]
        records[name] = {**row, "payload_bytes": payload_size, "payload_sha256": payload_sha}
        if mismatch: mismatched.append(name)
    if missing or extra: local_blockers.append(f"server/local tree mismatch: missing={missing!r} extra={extra!r}")
    if mismatched: local_blockers.append(f"server/local identity mismatch: {mismatched!r}")
    identity_ok = schema_ok and remote.get("repository") == UPSTREAM_HF and remote.get("requested_revision") == HF_REVISION and remote.get("resolved_revision") == HF_REVISION and remote.get("walk") == "recursive_file_only"
    if not identity_ok: local_blockers.append("server tree repository/revision/walk mismatch")
    blockers.extend(local_blockers)
    return {"status": "MATCHED" if identity_ok and not missing and not extra and not mismatched and not local_blockers else "MISMATCH", "repository": remote.get("repository"), "requested_revision": remote.get("requested_revision"), "resolved_revision": remote.get("resolved_revision"), "walk": remote.get("walk"), "packet_sha256": sha256(packet), "files": records, "missing": missing, "extra": extra, "content_mismatch": mismatched, "blockers": local_blockers}


def config_axes(config: dict[str, Any], processor: dict[str, Any], readme: str) -> dict[str, Any]:
    fc = config.get("fc_encoder_config", {})
    tf = config.get("tf_encoder_config", {})
    modules = config.get("modules_config", {})
    required = {
        "fc_hidden_size": fc.get("hidden_size"),
        "fc_layers": fc.get("num_hidden_layers"),
        "fc_heads": fc.get("num_attention_heads"),
        "tf_hidden_size": tf.get("d_model"),
        "tf_layers": tf.get("encoder_layers"),
        "tf_heads": tf.get("encoder_attention_heads"),
        "num_speakers": config.get("num_speakers", modules.get("num_speakers")),
        "subsampling_factor": fc.get("subsampling_factor", modules.get("subsampling_factor")),
    }
    if any(value is None for value in required.values()):
        raise RuntimeError(f"config.json is missing required raw axes: {required}")
    expected = {
        "fc_hidden_size": 512,
        "fc_layers": 18,
        "fc_heads": 8,
        "tf_hidden_size": 192,
        "tf_layers": 18,
        "tf_heads": 8,
        "num_speakers": 4,
        "subsampling_factor": 8,
    }
    if required != expected:
        raise RuntimeError(f"config.json raw axes mismatch: {required}")
    extractor = processor.get("feature_extractor", processor)
    processor_axes = {
        "sampling_rate": extractor.get("sampling_rate"),
        "hop_length": extractor.get("hop_length"),
        "feature_size": extractor.get("feature_size"),
    }
    if processor_axes != {"sampling_rate": 16000, "hop_length": 160, "feature_size": 80}:
        raise RuntimeError(f"processor_config.json frontend axes mismatch: {processor_axes}")
    frame_seconds = processor_axes["hop_length"] * required["subsampling_factor"] / processor_axes["sampling_rate"]
    if abs(frame_seconds - 0.08) > 1e-12:
        raise RuntimeError(f"derived frame duration is not 0.08 seconds: {frame_seconds}")
    normalized_readme = " ".join(readme.split()).casefold()
    normalized_sentence = " ".join(ARRIVAL_ORDER_SENTENCE.split()).casefold()
    if normalized_sentence not in normalized_readme:
        raise RuntimeError("README.md does not authenticate arrival-order semantics")
    return {
        **required,
        "processor": processor_axes,
        "arrival_order": "README_AUTHENTICATED",
        "frame_seconds": frame_seconds,
    }


def authenticate_source(
    source: Path,
    blockers: list[str],
    *,
    expected_repository: str = SOURCE_REPOSITORY,
    expected_revision: str = SOURCE_REVISION,
    expected_roles: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Authenticate the pinned public execution reference and every tracked file."""
    roles = dict(SOURCE_ROLE_BLOBS if expected_roles is None else expected_roles)
    inventory: dict[str, Any] = {
        "repository": expected_repository,
        "pinned_revision": expected_revision,
        "status": "UNVERIFIED",
        "role_files": {},
    }
    source_blockers: list[str] = []
    try:
        git = ["git", "-C", str(source)]
        actual = subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        origin = subprocess.run([*git, "remote", "get-url", "origin"], check=True, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run([*git, "status", "--porcelain", "--untracked-files=all"], check=True, capture_output=True, text=True).stdout.strip()
        entries = subprocess.run([*git, "ls-files", "-s", "-z"], check=True, capture_output=True).stdout.split(b"\0")
        gitlinks: list[dict[str, str]] = []
        tracked_modes: dict[str, str] = {}
        tracked_files: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        for entry in entries:
            if not entry:
                continue
            header, relative = entry.split(b"\t", 1)
            fields = header.split()
            if len(fields) != 3:
                source_blockers.append("NeMo source tracked entry schema mismatch")
                continue
            mode = fields[0].decode()
            index_object = fields[1].decode()
            stage = fields[2].decode()
            name = relative.decode()
            path = source / name
            if name in seen_paths:
                source_blockers.append(f"duplicate tracked source path: {name}")
            seen_paths.add(name)
            tracked_modes[name] = mode
            if stage != "0":
                source_blockers.append(f"NeMo source tracked entry is not stage 0: {name}")
            if mode not in ("100644", "100755"):
                gitlinks.append({"path": name, "mode": mode, "index_object_sha1": index_object})
                source_blockers.append(f"NeMo source tracked non-regular member: {name}")
            elif path.is_symlink() or not path.is_file():
                source_blockers.append(f"NeMo source tracked file missing/nonregular: {name}")
            else:
                expected_mode = 0o755 if mode == "100755" else 0o644
                if stat.S_IMODE(path.stat().st_mode) != expected_mode:
                    source_blockers.append(f"NeMo source tracked mode drift: {name}")
                head_object = subprocess.run([*git, "rev-parse", f"HEAD:{name}"], check=True, capture_output=True, text=True).stdout.strip()
                tracked_files.append({"path": name, "mode": mode, "stage": stage, "index_object_sha1": index_object, "head_object_sha1": head_object, "working_blob_sha1": git_blob_sha1(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
        role_files: dict[str, dict[str, Any]] = {}
        for role, expected in roles.items():
            path = source / role
            if not path.is_file() or path.is_symlink():
                source_blockers.append(f"NeMo source role missing/nonregular: {role}")
                continue
            tracked = next((row for row in tracked_files if row["path"] == role), None)
            if tracked is None:
                source_blockers.append(f"NeMo source role is not tracked: {role}")
                continue
            streamed = tracked["working_blob_sha1"]
            indexed = tracked["head_object_sha1"]
            index_object = tracked["index_object_sha1"]
            mode = tracked_modes.get(role)
            if mode != "100644":
                source_blockers.append(f"NeMo source role mode is not regular 100644: {role}")
            if streamed != expected or indexed != expected or index_object != expected:
                source_blockers.append(f"NeMo source role Git object mismatch: {role}")
            role_files[role] = {
                "path": role,
                "mode": mode,
                "working_blob_sha1": streamed,
                "head_object_sha1": indexed,
                "index_object_sha1": index_object,
                "expected_git_blob_sha1": expected,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        if actual != expected_revision or origin != expected_repository:
            source_blockers.append("NeMo source revision/origin mismatch")
        if dirty:
            source_blockers.append(f"NeMo source checkout is dirty: {dirty}")
        source_license = source / "LICENSE"
        license_text = source_license.read_text(encoding="utf-8", errors="replace").lower() if source_license.is_file() and not source_license.is_symlink() else ""
        license_markers = {"apache_license": "apache license, version 2.0" in license_text, "copyright": "copyright" in license_text, "grant": "distributed under the license" in license_text, "warranty": "without warranties or conditions of any kind" in license_text}
        license_auth = source_license.is_file() and not source_license.is_symlink() and git_blob_sha1(source_license) == roles["LICENSE"] and all(license_markers.values())
        if not license_auth:
            source_blockers.append("NeMo source LICENSE identity/Apache-2.0 clauses mismatch")
        role_auth = len(role_files) == len(roles) and all(
            row["working_blob_sha1"] == row["expected_git_blob_sha1"] == row["head_object_sha1"] == row["index_object_sha1"]
            for row in role_files.values()
        )
        auth_ok = actual == expected_revision and origin == expected_repository and not dirty and not gitlinks and role_auth and license_auth and not source_blockers
        inventory.update({
            "resolved_revision": actual,
            "origin": origin,
            "clean": not bool(dirty),
            "tracked_files": tracked_files,
            "tracked_modes": tracked_modes,
            "gitlinks": gitlinks,
            "role_files": role_files,
            "license": {"path": "LICENSE", "expected_git_blob_sha1": roles["LICENSE"], "authenticated": license_auth, "markers": license_markers},
            "provenance_status": "WEIGHT_BUILD_PROVENANCE_BLOCKED_MUTABLE_MAIN",
            "source_blockers": source_blockers,
            "status": "REFERENCE_SOURCE_SELECTED" if auth_ok else "BLOCKED",
        })
    except Exception as error:
        source_blockers.append(f"NeMo source inventory failed: {error}")
        inventory["source_blockers"] = source_blockers
        inventory["status"] = "BLOCKED"
    blockers.extend(source_blockers)
    return inventory


def _inspect(snapshot: Path, evidence: Path, packet: Path, source: Path) -> int:
    if not snapshot.is_dir():
        raise RuntimeError(f"HF snapshot directory is missing: {snapshot}")
    local_paths = files(snapshot)
    snapshot_files = sorted(path.relative_to(snapshot).as_posix() for path in local_paths)
    if set(snapshot_files) != EXPECTED_FILES:
        raise RuntimeError(f"HF snapshot exact file set mismatch: {sorted(set(snapshot_files) ^ EXPECTED_FILES)}")
    missing = sorted(EXPECTED_FILES - set(snapshot_files))
    if missing:
        raise RuntimeError(f"HF snapshot is incomplete; missing {missing}")
    model = snapshot / "model.safetensors"
    if model.stat().st_size != MODEL_BYTES:
        raise RuntimeError(f"model.safetensors byte-size mismatch: {model.stat().st_size}")
    model_digest = sha256(model)
    if model_digest != MODEL_SHA256:
        raise RuntimeError(f"model.safetensors SHA256 mismatch: {model_digest}")
    nemo = snapshot / "diar_sortformer_4spk-v1.nemo"
    if nemo.stat().st_size != NEMO_BYTES:
        raise RuntimeError(f"diar_sortformer_4spk-v1.nemo byte-size mismatch: {nemo.stat().st_size}")
    nemo_digest = sha256(nemo)
    if nemo_digest != NEMO_SHA256:
        raise RuntimeError(f"diar_sortformer_4spk-v1.nemo SHA256 mismatch: {nemo_digest}")
    metadata, tensors = tensor_inventory(model)
    config = parse_config(snapshot / "config.json")
    processor = parse_config(snapshot / "processor_config.json")
    readme = (snapshot / "README.md").read_text(encoding="utf-8")
    model_card_frontmatter = parse_model_card_frontmatter(readme)
    nemo_members, nemo_blockers = validate_nemo_archive(nemo)
    tree_blockers: list[str] = []
    server_tree_evidence = server_tree(snapshot, packet, tree_blockers)
    source_blockers: list[str] = []
    source_inventory = authenticate_source(source, source_blockers)
    blockers = [
        "weight_build_provenance=WEIGHT_BUILD_PROVENANCE_BLOCKED_MUTABLE_MAIN",
        "no reviewed prepared contract is emitted; runtime/converter remain fail-closed",
        *nemo_blockers,
        *tree_blockers,
        *source_blockers,
    ]
    evidence.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": FORMAT,
        "status": "BLOCKED",
        "evidence_stage": "INSPECTION_ONLY",
        "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
        "cpu_status": "UNSUPPORTED",
        "metal_status": "BLOCKED_BY_CPU",
        "parity_status": "NOT_RUN",
        "publication": "NO_UPLOAD",
        "collection_status": {"hf_tree": server_tree_evidence.get("status"), "fixed_artifacts": "MATCHED", "source": source_inventory.get("status"), "authenticated": server_tree_evidence.get("status") == "MATCHED" and source_inventory.get("status") == "REFERENCE_SOURCE_SELECTED"},
        "source_status": source_inventory.get("status", "UNVERIFIED"),
        "weight_build_provenance_status": "WEIGHT_BUILD_PROVENANCE_BLOCKED_MUTABLE_MAIN",
        "hf": {"repository": UPSTREAM_HF, "requested_revision": HF_REVISION, "resolved_revision": server_tree_evidence.get("resolved_revision"), "server_tree": server_tree_evidence},
        "transport_cache_exclusion": {"path": HF_TRANSPORT_CACHE, "scope": "exact subtree only"},
        "hf_chronology": HF_CHRONOLOGY,
        "license": {"weights": "CC-BY-NC-4.0", "source": "Apache-2.0", "source_provenance": "REFERENCE_ONLY_NOT_WEIGHT_BUILD"},
        "model_card_frontmatter": model_card_frontmatter,
        "files": {name: {"bytes": (snapshot / name).stat().st_size, "sha256": sha256(snapshot / name)} for name in snapshot_files},
        "config_json": {"canonical_axes": config_axes(config, processor, readme), "sha256": sha256(snapshot / "config.json")},
        "safetensors_metadata": metadata,
        "tensor_count": len(tensors),
        "tensors": tensors,
        "nemo_archive": {"members": nemo_members, "sha256": sha256(snapshot / "diar_sortformer_4spk-v1.nemo"), "bytes": (snapshot / "diar_sortformer_4spk-v1.nemo").stat().st_size},
        "official_source": source_inventory,
        "blockers": blockers,
    }
    (evidence / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 2


def inspect(snapshot: Path, evidence: Path, packet: Path, source: Path) -> int:
    """Always preserve fixed identity/status evidence, even on inspection errors."""
    try:
        return _inspect(snapshot, evidence, packet, source)
    except Exception as error:  # noqa: BLE001 - error is itself evidence
        evidence.mkdir(parents=True, exist_ok=True)
        manifest = {
            "format": FORMAT,
            "status": "BLOCKED",
            "evidence_stage": "INSPECTION_ONLY",
            "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
            "cpu_status": "UNSUPPORTED",
            "metal_status": "BLOCKED_BY_CPU",
            "parity_status": "NOT_RUN",
            "publication": "NO_UPLOAD",
            "collection_status": {"hf_tree": "UNVERIFIED", "fixed_artifacts": "UNVERIFIED", "source": "UNVERIFIED", "authenticated": False},
            "hf": {"repository": UPSTREAM_HF, "requested_revision": HF_REVISION},
            "source_status": "UNVERIFIED",
            "weight_build_provenance_status": "WEIGHT_BUILD_PROVENANCE_BLOCKED_MUTABLE_MAIN",
            "license": {"weights": "CC-BY-NC-4.0", "source": "Apache-2.0", "source_provenance": "REFERENCE_ONLY_NOT_WEIGHT_BUILD"},
            "error": f"{type(error).__name__}: {error}",
            "blockers": ["inspection_error", f"{type(error).__name__}: {error}"],
        }
        (evidence / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(f"Sortformer inspection blocked; evidence preserved at {evidence / 'manifest.json'}", file=sys.stderr)
        return 2


def self_test() -> None:
    global MODEL_BYTES, MODEL_SHA256, NEMO_BYTES, NEMO_SHA256
    source = Path(__file__).read_text(encoding="utf-8")
    assert HF_REVISION in source and MODEL_SHA256 in source and str(MODEL_BYTES) in source
    assert "CC-BY-NC-4.0" in source and "WEIGHT_BUILD_PROVENANCE_BLOCKED_MUTABLE_MAIN" in source
    assert "weights_only=True" in source and "NO_UPLOAD" in source

    def archive(name: str, kind: str = "file") -> io.BytesIO:
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w") as tar:
            member = tarfile.TarInfo(name)
            if kind == "symlink":
                member.type = tarfile.SYMTYPE
                member.linkname = "safe"
                tar.addfile(member)
            elif kind == "directory":
                member.type = tarfile.DIRTYPE
                tar.addfile(member)
            else:
                payload = b"unsafe"
                member.size = len(payload)
                tar.addfile(member, io.BytesIO(payload))
        output.seek(0)
        return output

    for fixture in (
        archive("../escape"),
        archive("nul\x00member"),
        archive("back\\slash"),
        archive("link", "symlink"),
    ):
        try:
            validate_nemo_archive(fixture)
        except RuntimeError:
            pass
        else:
            raise AssertionError("unsafe archive path/link fixture was accepted")
    members, blockers = validate_nemo_archive(archive("safe-dir", "directory"))
    assert members == [{"name": "safe-dir", "bytes": 0, "type": "directory", "sha256": None}]
    assert blockers == []
    _, blockers = validate_nemo_archive(archive("weights.ckpt"))
    assert any("weights_only=True" in blocker for blocker in blockers)
    root_temp = tempfile.TemporaryDirectory(prefix="sortformer-tree-")
    root = Path(root_temp.name)
    header = json.dumps({"z": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}, "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    offset_path = root / "offset-order.safetensors"
    offset_path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 8)
    _, offset_rows = tensor_inventory(offset_path)
    assert [row["name"] for row in offset_rows] == ["a", "z"]
    for offsets in (([0, 4], [3, 8]), ([0, 4], [8, 12])):
        malformed = json.dumps({"a": {"dtype": "F32", "shape": [1], "data_offsets": offsets[0]}, "z": {"dtype": "F32", "shape": [1], "data_offsets": offsets[1]}}).encode()
        malformed_path = root / "malformed-ranges.safetensors"
        malformed_path.write_bytes(struct.pack("<Q", len(malformed)) + malformed + b"\0" * 12)
        try: tensor_inventory(malformed_path)
        except RuntimeError: pass
        else: raise AssertionError("overlap/gap safetensors fixture was accepted")
    snapshot = root / "snapshot"
    snapshot.mkdir()
    for name in sorted(EXPECTED_FILES):
        (snapshot / name).write_bytes(name.encode())
    regular = snapshot / "config.json"
    lfs = snapshot / "README.md"
    lfs_sha = sha256(lfs)
    lfs_pointer = git_blob_sha1_bytes(lfs_pointer_bytes(lfs_sha, lfs.stat().st_size))
    rows = []
    for name in sorted(EXPECTED_FILES):
        path = snapshot / name
        is_lfs = path == lfs
        rows.append({"path": name, "type": "file", "size": path.stat().st_size, "git_blob_sha1": None if is_lfs else git_blob_sha1(path), "lfs_pointer_git_blob_sha1": lfs_pointer if is_lfs else None, "lfs_sha256": lfs_sha if is_lfs else None})
    tree = root / "server-tree.json"
    tree.write_text(json.dumps({"repository": UPSTREAM_HF, "requested_revision": HF_REVISION, "resolved_revision": HF_REVISION, "walk": "recursive_file_only", "files": rows}), encoding="utf-8")
    tree_blockers: list[str] = []
    assert server_tree(snapshot, tree, tree_blockers)["status"] == "MATCHED" and not tree_blockers
    for field, value in (("requested_revision", "0" * 40), ("lfs_pointer_git_blob_sha1", "0" * 40), ("lfs_sha256", "0" * 64)):
        spoof = {"repository": UPSTREAM_HF, "requested_revision": HF_REVISION, "resolved_revision": HF_REVISION, "walk": "recursive_file_only", "files": [dict(row) for row in rows]}
        if field == "requested_revision": spoof[field] = value
        else: spoof["files"][1][field] = value
        spoof_path = root / f"spoof-{field}.json"
        spoof_path.write_text(json.dumps(spoof), encoding="utf-8")
        spoof_blockers: list[str] = []
        assert server_tree(snapshot, spoof_path, spoof_blockers)["status"] == "MISMATCH" and spoof_blockers
    for packet_rows, label in ((rows[:1], "missing"), (rows + [{"path": "orphan", "type": "file", "size": 1, "git_blob_sha1": "0" * 40, "lfs_pointer_git_blob_sha1": None, "lfs_sha256": None}], "extra")):
        tree_path = root / f"{label}-tree.json"
        tree_path.write_text(json.dumps({"repository": UPSTREAM_HF, "requested_revision": HF_REVISION, "resolved_revision": HF_REVISION, "walk": "recursive_file_only", "files": packet_rows}), encoding="utf-8")
        tree_blockers = []
        assert server_tree(snapshot, tree_path, tree_blockers)["status"] == "MISMATCH" and tree_blockers
    transport = snapshot / HF_TRANSPORT_CACHE
    transport.mkdir(parents=True)
    (transport / "transport-marker").write_bytes(b"ignored transport cache")
    transport_blockers: list[str] = []
    assert server_tree(snapshot, tree, transport_blockers)["status"] == "MATCHED" and not transport_blockers
    (transport / "transport-marker").unlink()
    transport.rmdir()
    transport.symlink_to("/etc")
    try:
        server_tree(snapshot, tree, [])
    except RuntimeError:
        pass
    else:
        raise AssertionError("transport cache symlink was accepted")
    transport.unlink()
    (snapshot / ".cache").rmdir()
    cache = snapshot / ".cache"
    cache.write_bytes(b"transport cache must not be silently ignored")
    cache_blockers: list[str] = []
    try: server_tree(snapshot, tree, cache_blockers)
    except RuntimeError: pass
    else: raise AssertionError("nested cache outside the exact transport subtree was accepted")
    cache.unlink()
    symlink = snapshot / "escape-link"
    try:
        symlink.symlink_to("/etc/passwd")
        server_tree(snapshot, tree, [])
    except RuntimeError:
        pass
    else:
        raise AssertionError("snapshot symlink was accepted")
    duplicate = io.BytesIO()
    with tarfile.open(fileobj=duplicate, mode="w") as tar:
        for _ in range(2):
            member = tarfile.TarInfo("config.json")
            payload = b"{}"
            member.size = len(payload)
            tar.addfile(member, io.BytesIO(payload))
    duplicate.seek(0)
    try:
        validate_nemo_archive(duplicate)
    except RuntimeError:
        pass
    else:
        raise AssertionError("duplicate archive fixture was accepted")
    oversize = io.BytesIO()
    with tarfile.open(fileobj=oversize, mode="w") as tar:
        member = tarfile.TarInfo("oversize.bin"); member.size = MAX_NEMO_MEMBER_BYTES + 1; tar.addfile(member)
    oversize.seek(0)
    try: validate_nemo_archive(oversize)
    except RuntimeError: pass
    else: raise AssertionError("oversize archive member was accepted")
    config = {
        "fc_encoder_config": {"hidden_size": 512, "num_hidden_layers": 18, "num_attention_heads": 8, "subsampling_factor": 8},
        "tf_encoder_config": {"d_model": 192, "encoder_layers": 18, "encoder_attention_heads": 8},
        "modules_config": {"num_speakers": 4},
    }
    processor = {"feature_extractor": {"sampling_rate": 16000, "hop_length": 160, "feature_size": 80}}
    axes = config_axes(config, processor, ARRIVAL_ORDER_SENTENCE)
    assert axes["arrival_order"] == "README_AUTHENTICATED"
    assert parse_model_card_frontmatter("---\nlicense: cc-by-nc-4.0\n---\nprose") == {"license": "cc-by-nc-4.0"}
    canonical_card = "---\nlicense: cc-by-nc-4.0\ndatasets:\n- diarization/example\ntags:\n- audio\nwidget:\n- example: true\nmodel-index:\n- name: Sortformer\n  results:\n  - task:\n      type: audio\n---\nprose"
    assert parse_model_card_frontmatter(canonical_card) == {"license": "cc-by-nc-4.0"}
    for card in (
        "prose license: cc-by-nc-4.0",
        "---\nlicense: cc-by-nc-4.0\nlicense: cc-by-nc-4.0\n---",
        "---\ndatasets: one\ndatasets: two\nlicense: cc-by-nc-4.0\n---",
        "---\ndatasets:\n  license: cc-by-nc-4.0\n---",
        "---\nlicense:\n- cc-by-nc-4.0\n---",
    ):
        try:
            parse_model_card_frontmatter(card)
        except RuntimeError:
            pass
        else:
            raise AssertionError("malformed/prose model-card license was accepted")
    try:
        config_axes(config, processor, "The implementation emits speakers in arrival order.")
    except RuntimeError:
        pass
    else:
        raise AssertionError("weak arrival-order wording was accepted")
    auth_source = root / "source-checkout"
    auth_source.mkdir()
    subprocess.run(["git", "init", "-q", str(auth_source)], check=True)
    subprocess.run(["git", "-C", str(auth_source), "config", "user.email", "sortformer-selftest@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(auth_source), "config", "user.name", "Sortformer self-test"], check=True)
    license_text = "Apache License, Version 2.0\nCopyright 2026\nDistributed under the License\nwithout warranties or conditions of any kind\n"
    (auth_source / "LICENSE").write_text(license_text, encoding="utf-8")
    fixture_roles: dict[str, str] = {"LICENSE": git_blob_sha1(auth_source / "LICENSE")}
    for role in SOURCE_ROLE_BLOBS:
        if role == "LICENSE":
            continue
        role_path = auth_source / role
        role_path.parent.mkdir(parents=True, exist_ok=True)
        role_path.write_text(f"fixture role: {role}\n", encoding="utf-8")
        fixture_roles[role] = git_blob_sha1(role_path)
    assert set(fixture_roles) == set(SOURCE_ROLE_BLOBS)
    subprocess.run(["git", "-C", str(auth_source), "add", "LICENSE"], check=True)
    subprocess.run(["git", "-C", str(auth_source), "add", *fixture_roles.keys()], check=True)
    subprocess.run(["git", "-C", str(auth_source), "commit", "-q", "-m", "fixture"], check=True)
    subprocess.run(["git", "-C", str(auth_source), "remote", "add", "origin", SOURCE_REPOSITORY], check=True)
    fixture_revision = subprocess.run(["git", "-C", str(auth_source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    source_blockers: list[str] = []
    source_evidence = authenticate_source(auth_source, source_blockers, expected_revision=fixture_revision, expected_roles=fixture_roles)
    assert source_evidence["status"] == "REFERENCE_SOURCE_SELECTED"
    assert not source_blockers and source_evidence["clean"]
    assert all(row["index_object_sha1"] == row["head_object_sha1"] == row["working_blob_sha1"] == row["expected_git_blob_sha1"] for row in source_evidence["role_files"].values())
    spoof_blockers: list[str] = []
    spoofed_roles = dict(fixture_roles)
    spoofed_roles["LICENSE"] = "0" * 40
    spoof_evidence = authenticate_source(auth_source, spoof_blockers, expected_revision=fixture_revision, expected_roles=spoofed_roles)
    assert spoof_evidence["status"] == "BLOCKED" and any("Git object mismatch" in blocker for blocker in spoof_blockers)
    assert source_selection_status(["missing tracked file"]) == "BLOCKED"
    assert source_selection_status(["mode/object/dirty blocker"]) == "BLOCKED"
    assert source_selection_status([]) == "REFERENCE_SOURCE_SELECTED"
    subprocess.run(["git", "-C", str(auth_source), "update-index", "--chmod=+x", "LICENSE"], check=True)
    subprocess.run(["git", "-C", str(auth_source), "commit", "-q", "-m", "mode spoof"], check=True)
    mode_blockers: list[str] = []
    mode_evidence = authenticate_source(auth_source, mode_blockers)
    assert mode_evidence["status"] == "BLOCKED" and mode_blockers
    (auth_source / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    dirty_blockers: list[str] = []
    dirty_evidence = authenticate_source(auth_source, dirty_blockers)
    assert dirty_evidence["status"] == "BLOCKED" and dirty_blockers
    # Exercise the structurally successful collection path with tiny fixture
    # payloads while keeping the overall inspection verdict BLOCKED.
    fixture_snapshot = root / "successful-snapshot"
    fixture_snapshot.mkdir()
    fixture_header = json.dumps({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    fixture_model_bytes = struct.pack("<Q", len(fixture_header)) + fixture_header + b"\0" * 4
    (fixture_snapshot / "model.safetensors").write_bytes(fixture_model_bytes)
    fixture_nemo = io.BytesIO()
    with tarfile.open(fileobj=fixture_nemo, mode="w") as archive:
        member = tarfile.TarInfo("config.json")
        member.size = 2
        archive.addfile(member, io.BytesIO(b"{}"))
    (fixture_snapshot / "diar_sortformer_4spk-v1.nemo").write_bytes(fixture_nemo.getvalue())
    (fixture_snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (fixture_snapshot / "processor_config.json").write_text(json.dumps(processor), encoding="utf-8")
    (fixture_snapshot / "README.md").write_text("---\nlicense: cc-by-nc-4.0\n---\n" + ARRIVAL_ORDER_SENTENCE, encoding="utf-8")
    for name in EXPECTED_FILES - {"model.safetensors", "diar_sortformer_4spk-v1.nemo", "config.json", "processor_config.json", "README.md"}:
        (fixture_snapshot / name).write_bytes(name.encode())
    fixture_rows = []
    for path in files(fixture_snapshot):
        relative = path.relative_to(fixture_snapshot).as_posix()
        fixture_rows.append({"path": relative, "type": "file", "size": path.stat().st_size, "git_blob_sha1": git_blob_sha1(path), "lfs_pointer_git_blob_sha1": None, "lfs_sha256": None})
    fixture_tree = root / "successful-tree.json"
    fixture_tree.write_text(json.dumps({"repository": UPSTREAM_HF, "requested_revision": HF_REVISION, "resolved_revision": HF_REVISION, "walk": "recursive_file_only", "files": fixture_rows}), encoding="utf-8")
    saved = MODEL_BYTES, MODEL_SHA256, NEMO_BYTES, NEMO_SHA256
    MODEL_BYTES, MODEL_SHA256 = len(fixture_model_bytes), sha256(fixture_snapshot / "model.safetensors")
    NEMO_BYTES, NEMO_SHA256 = (fixture_snapshot / "diar_sortformer_4spk-v1.nemo").stat().st_size, sha256(fixture_snapshot / "diar_sortformer_4spk-v1.nemo")
    try:
        fixture_evidence = root / "successful-evidence"
        assert _inspect(fixture_snapshot, fixture_evidence, fixture_tree, auth_source) == 2
        fixture_manifest = json.loads((fixture_evidence / "manifest.json").read_text(encoding="utf-8"))
        assert fixture_manifest["status"] == "BLOCKED" and fixture_manifest["evidence_stage"] == "INSPECTION_ONLY"
        assert fixture_manifest["collection_status"] == {"hf_tree": "MATCHED", "fixed_artifacts": "MATCHED", "source": "BLOCKED", "authenticated": False}
        assert fixture_manifest["source_status"] == "BLOCKED"
    finally:
        MODEL_BYTES, MODEL_SHA256, NEMO_BYTES, NEMO_SHA256 = saved
    with tempfile.TemporaryDirectory(prefix="sortformer-error-evidence-") as temporary:
        evidence = Path(temporary) / "evidence"
        assert inspect(Path(temporary) / "missing-snapshot", evidence, Path(temporary) / "missing-tree", Path(temporary) / "missing-source") == 2
        failure_manifest = json.loads((evidence / "manifest.json").read_text(encoding="utf-8"))
        assert failure_manifest["hf"]["requested_revision"] == HF_REVISION
        assert failure_manifest["status"] == "BLOCKED"
        assert failure_manifest["evidence_stage"] == "INSPECTION_ONLY"
        assert failure_manifest["runtime_status"] == "NOT_IMPLEMENTED_FAIL_CLOSED"
        assert failure_manifest["source_status"] == "UNVERIFIED" and not failure_manifest["collection_status"]["authenticated"]
        assert failure_manifest["blockers"]
    print("sortformer_diar_4spk_v1_inspect.py self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--server-tree", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.snapshot, args.evidence, args.server_tree, args.source)):
            parser.error("--self-test accepts no other arguments")
        self_test()
        return 0
    if any(value is None for value in (args.snapshot, args.evidence, args.server_tree, args.source)):
        parser.error("--snapshot --evidence --server-tree --source are required")
    return inspect(args.snapshot, args.evidence, args.server_tree, args.source)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, tarfile.TarError) as error:
        print(f"Sortformer inspection: {error}", file=sys.stderr)
        raise SystemExit(2)
