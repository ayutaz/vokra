#!/usr/bin/env python3
"""Run the pinned MOSS-TTS Local implementation and record independent rows.

The real path is VAST-only. ``transformers`` executes the fixed HF custom
code; this module is only an evidence collector and never mirrors the model.
Before importing that code we authenticate the local snapshot, its bounded
safetensors header, and the source files used by transformers_modules.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
import struct
from pathlib import Path
from typing import Any

REPOSITORY = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
REVISION = "be7766a6735b98bd793f7c79fb720b4d0f5d13b8"
COLUMNS = 13
MAX_HEADER_BYTES = 64 * 1024 * 1024
MAX_FILES = 2_000
MAX_TENSORS = 1_000
AUDIO_START_TOKEN = 151_669
AUDIO_END_TOKEN = 151_670
AUDIO_ASSISTANT_SLOT_TOKEN = 151_656
SOURCE_DIGESTS = {
    "826f81f163b1b557ad13f83c4f35008f4fee5a6cb6311b4316ff3dbb25149411": "configuration",
    "ab6debcb92032cb9dc91ae80aed77dbadd2e59848208baef2b062bd6def3f3be": "configuration_source",
    "b0a66211943ae580b087f3e71495fea2f455701a4f6c29b6d3562218f7668c5f": "modeling_source",
    "3fc5616b1ec3408162b7d859a7696725a40525313b20f9b31a06ee55c93bd7ad": "processing_source",
    "f2e877104669f1e6c7cd34680f0da1a8a159e032123ee56b660b63929b6c8989": "gpt2_source",
    "100163bd7ecf31a59bafacc0b032ace9339edc992a3eb4cc80662502e04e46f0": "qwen3_source",
    "db574bfebad009e05193196a63a4eeecd353eeca177ccfff28b9379d595d88b7": "processor_config",
}


def strict_json(data: str | bytes, *, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    return json.loads(data, object_pairs_hook=reject_duplicates)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    size = path.stat().st_size
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1_bytes(data: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(data)}\0".encode("ascii"))
    digest.update(data)
    return digest.hexdigest()


def canonical_lfs_pointer(sha256: str, size: int) -> bytes:
    return (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{sha256}\n"
        f"size {size}\n"
    ).encode("ascii")


def authenticate_server_tree(snapshot: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare every local file with the fixed revision's recursive HF tree.

    This is deliberately called only in the VAST path.  A local directory
    that happens to contain a plausible model cannot become an oracle without
    the server-side revision and content identities.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.model_info(REPOSITORY, revision=REVISION)
    resolved = getattr(info, "sha", None)
    if resolved != REVISION:
        raise ValueError(f"HF resolved revision {resolved!r} != pinned {REVISION}")
    remote_items = list(
        api.list_repo_tree(REPOSITORY, revision=REVISION, recursive=True, expand=True)
    )
    remote: dict[str, dict[str, Any]] = {}
    for item in remote_items:
        kind = getattr(item, "type", "file")
        if kind in {"directory", "folder"}:
            continue
        if kind != "file":
            raise ValueError(f"HF tree contains unsupported member type {kind!r}")
        path = getattr(item, "path", None)
        size = getattr(item, "size", None)
        blob = getattr(item, "blob_id", None) or getattr(item, "oid", None)
        lfs_obj = getattr(item, "lfs", None)
        if isinstance(lfs_obj, dict):
            lfs = lfs_obj.get("sha256") or lfs_obj.get("oid")
            lfs_size = lfs_obj.get("size")
        else:
            lfs = getattr(lfs_obj, "sha256", None) or getattr(lfs_obj, "oid", None)
            lfs_size = getattr(lfs_obj, "size", None)
        if not isinstance(path, str) or not isinstance(size, int) or not isinstance(blob, str):
            raise ValueError("HF tree contains an untyped/non-file entry")
        if path in remote or path.startswith("/") or "\\" in path or "\x00" in path or any(part in {"", ".", ".."} for part in path.split("/")):
            raise ValueError(f"unsafe/duplicate HF tree path {path!r}")
        if not re.fullmatch(r"[0-9a-f]{40}", blob.lower()):
            raise ValueError(f"invalid HF Git blob id for {path}")
        if lfs is not None and not re.fullmatch(r"[0-9a-f]{64}", str(lfs).lower()):
            raise ValueError(f"invalid HF LFS digest for {path}")
        if lfs is not None and (not isinstance(lfs_size, int) or lfs_size != size):
            raise ValueError(f"HF LFS size is missing or differs from file size for {path}")
        if lfs is None and lfs_size is not None:
            raise ValueError(f"non-LFS HF entry has an LFS size for {path}")
        remote[path] = {"path": path, "type": "file", "size": size, "git_blob_sha1": blob.lower(), "lfs_sha256": str(lfs).lower() if lfs is not None else None, "lfs_size": lfs_size}
    local = {item["path"]: item for item in records}
    if set(local) != set(remote):
        raise ValueError(f"HF/local tree mismatch: missing={sorted(set(remote)-set(local))} extra={sorted(set(local)-set(remote))}")
    checked: list[dict[str, Any]] = []
    for path, expected in sorted(remote.items()):
        actual_path = snapshot / path
        actual = local[path]
        if actual["bytes"] != expected["size"]:
            raise ValueError(f"HF/local size mismatch for {path}")
        if expected["lfs_sha256"] is not None:
            if actual["sha256"] != expected["lfs_sha256"]:
                raise ValueError(f"HF/local LFS content mismatch for {path}")
            pointer = canonical_lfs_pointer(expected["lfs_sha256"], expected["lfs_size"])
            if git_blob_sha1_bytes(pointer) != expected["git_blob_sha1"]:
                raise ValueError(f"HF Git-LFS pointer identity mismatch for {path}")
        elif git_blob_sha1(actual_path) != expected["git_blob_sha1"]:
            raise ValueError(f"HF/local Git blob mismatch for {path}")
        checked.append({**expected, "local_sha256": actual["sha256"]})
    return {"repository": REPOSITORY, "revision": REVISION, "resolved_revision": resolved, "files": checked}


def safe_relative(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes snapshot: {path}") from exc
    value = relative.as_posix()
    if not value or "\x00" in value or "\\" in value or value.startswith("/"):
        raise ValueError(f"unsafe snapshot path: {value!r}")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"unsafe snapshot path: {value!r}")
    return value


def local_files(snapshot: Path) -> list[tuple[str, Path]]:
    if not snapshot.is_dir():
        raise ValueError(f"snapshot is not a directory: {snapshot}")
    files: list[tuple[str, Path]] = []
    root = snapshot.resolve()
    for path in snapshot.rglob("*"):
        relative = path.relative_to(snapshot)
        if ".cache" in relative.parts:
            continue
        if path.is_symlink():
            target = path.resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"snapshot symlink escapes root: {path}") from exc
            if not target.is_file():
                raise ValueError(f"snapshot symlink is not a regular file: {path}")
            path = target
        if path.is_file():
            files.append((safe_relative(path, snapshot), path))
        elif not path.is_dir():
            raise ValueError(f"snapshot member is not regular: {path}")
        if len(files) > MAX_FILES:
            raise ValueError("snapshot file count exceeds bound")
    if not files:
        raise ValueError("snapshot has no regular files")
    return sorted(files)


def _dtype_size(dtype: str) -> int:
    sizes = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1, "I16": 2, "I32": 4, "I64": 8, "BOOL": 1}
    if dtype not in sizes:
        raise ValueError(f"unsupported safetensors dtype {dtype!r}")
    return sizes[dtype]


def safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path}: truncated safetensors header length")
        header_len = struct.unpack("<Q", raw_len)[0]
        if header_len == 0 or header_len > MAX_HEADER_BYTES:
            raise ValueError(f"{path}: unsafe header length {header_len}")
        header = handle.read(header_len)
        if len(header) != header_len:
            raise ValueError(f"{path}: truncated safetensors header")
        total = path.stat().st_size
        descriptor = strict_json(header, label=str(path))
        if not isinstance(descriptor, dict):
            raise ValueError(f"{path}: header must be an object")
        entries = [(name, value) for name, value in descriptor.items() if name != "__metadata__"]
        if not entries or len(entries) > MAX_TENSORS:
            raise ValueError(f"{path}: invalid tensor count")
        intervals: list[tuple[int, int, str]] = []
        manifest: list[dict[str, Any]] = []
        for name, value in entries:
            if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
                raise ValueError(f"{path}: unsafe tensor name {name!r}")
            if name.startswith("/") or any(part in {"", ".", ".."} for part in name.split("/")):
                raise ValueError(f"{path}: unsafe tensor name {name!r}")
            if not isinstance(value, dict) or set(value) != {"dtype", "shape", "data_offsets"}:
                raise ValueError(f"{path}: malformed descriptor for {name}")
            dtype = value["dtype"]
            shape = value["shape"]
            offsets = value["data_offsets"]
            if not isinstance(dtype, str) or not isinstance(shape, list) or not isinstance(offsets, list):
                raise ValueError(f"{path}: malformed descriptor types for {name}")
            if any(type(axis) is not int or axis < 0 for axis in shape) or len(shape) > 16:
                raise ValueError(f"{path}: malformed tensor shape for {name}")
            if len(offsets) != 2 or any(type(item) is not int for item in offsets):
                raise ValueError(f"{path}: malformed tensor offsets for {name}")
            start, end = offsets
            if start < 0 or end < start:
                raise ValueError(f"{path}: invalid tensor offsets for {name}")
            numel = 1
            for axis in shape:
                numel *= axis
            if end - start != numel * _dtype_size(dtype):
                raise ValueError(f"{path}: tensor byte size mismatch for {name}")
            absolute_start = 8 + header_len + start
            absolute_end = 8 + header_len + end
            if absolute_end > total:
                raise ValueError(f"{path}: tensor body exceeds file for {name}")
            intervals.append((absolute_start, absolute_end, name))
            manifest.append({"name": name, "dtype": dtype, "shape": shape, "numel": numel, "data_offsets": offsets})
        intervals.sort()
        previous_end = 8 + header_len
        for start, end, name in intervals:
            if start != previous_end:
                raise ValueError(f"{path}: tensor body has gap/overlap before {name}")
            previous_end = end
        if previous_end != total:
            raise ValueError(f"{path}: trailing tensor bytes")
        metadata = descriptor.get("__metadata__", {})
        if not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()):
            raise ValueError(f"{path}: metadata must be a string map")
        manifest.sort(key=lambda item: item["name"])
        manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        return {"path": path.name, "bytes": total, "tensor_count": len(manifest), "manifest_sha256": hashlib.sha256(manifest_json.encode()).hexdigest(), "tensors": manifest, "metadata": metadata}


def authenticate_snapshot(snapshot: Path) -> dict[str, Any]:
    files = local_files(snapshot)
    records = [{"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)} for name, path in files]
    by_name = {item["path"]: item for item in records}
    model_path = snapshot / "model.safetensors"
    if not model_path.is_file():
        raise ValueError("fixed Local snapshot must contain model.safetensors")
    header = safetensors_header(model_path)
    if header["tensor_count"] != 438:
        raise ValueError(f"fixed Local snapshot expected 438 tensors, got {header['tensor_count']}")
    matched: dict[str, dict[str, str]] = {}
    for digest, role in SOURCE_DIGESTS.items():
        hits = [item for item in records if item["sha256"] == digest]
        if len(hits) != 1:
            raise ValueError(f"missing or ambiguous pinned source digest for {role}")
        matched[role] = {"path": hits[0]["path"], "sha256": digest}
    config = strict_json((snapshot / "config.json").read_bytes(), label="config.json")
    if not isinstance(config, dict):
        raise ValueError("config.json must be an object")
    return {"repository": REPOSITORY, "revision": REVISION, "files": records, "model": header, "custom_code": matched, "config_sha256": by_name["config.json"]["sha256"]}


def authenticate_loaded_source(
    obj: object, *, label: str, role: str, expected_digest: str
) -> dict[str, str]:
    source = inspect.getsourcefile(obj)
    if source is None:
        raise ValueError(f"{label} has no inspectable source")
    path = Path(source).resolve()
    if "transformers_modules" not in path.parts:
        raise ValueError(f"{label} was not loaded from HF transformers_modules: {path}")
    digest = sha256_file(path)
    if digest != expected_digest:
        raise ValueError(f"{label} source sha256 {digest} != pinned {expected_digest}")
    return {
        "label": label,
        "role": role,
        "path": str(path),
        "sha256": digest,
        "resolved_revision": REVISION,
    }


def read_rows(path: Path) -> list[list[int]]:
    data = path.read_bytes()
    width = COLUMNS * 4
    if not data or len(data) % width:
        raise ValueError(f"{path}: expected a non-empty [rows,13] u32le matrix")
    words = struct.unpack(f"<{len(data) // 4}I", data)
    return [list(words[offset : offset + COLUMNS]) for offset in range(0, len(words), COLUMNS)]


def write_rows(path: Path, rows: list[list[int]]) -> None:
    if not rows or any(len(row) != COLUMNS for row in rows):
        raise ValueError("normalized rows must be a non-empty [rows,13] matrix")
    data = struct.pack(f"<{len(rows) * COLUMNS}I", *(value for row in rows for value in row))
    path.write_bytes(data)


def normalize_generated(prompt: list[list[int]], generated: Any) -> tuple[list[list[int]], list[list[int]], int, int]:
    if not hasattr(generated, "ndim") or generated.ndim != 3 or tuple(generated.shape)[0] != 1 or tuple(generated.shape)[2] != COLUMNS:
        raise ValueError(f"official generation returned unexpected shape {getattr(generated, 'shape', None)}")
    output = generated[0].to(device="cpu").tolist()
    output = [[int(value) for value in row] for row in output]
    if any(value < 0 or value > 0xFFFFFFFF for row in output for value in row):
        raise ValueError("official generation returned a token outside u32")
    starts = [index for index, row in enumerate(prompt) if row[0] == AUDIO_START_TOKEN]
    if not starts:
        raise ValueError("prompt contains no audio-start row")
    suffix = prompt[starts[-1] :]
    if output[: len(prompt)] == prompt:
        normalized = output[starts[-1] :]
    elif output[: len(suffix)] == suffix:
        normalized = output
    else:
        raise ValueError("official output does not preserve the prompt/audio-start boundary")
    start_length = len(suffix) - 1
    first_generated = start_length + 1
    if first_generated >= len(normalized):
        raise ValueError("official generation returned no frame after audio-start boundary")
    assistant: list[list[int]] = []
    for row in normalized[first_generated:]:
        if row[0] == AUDIO_END_TOKEN:
            break
        if row[0] != AUDIO_ASSISTANT_SLOT_TOKEN:
            raise ValueError(f"unexpected generated row decision token {row[0]}")
        assistant.append(row)
    if not assistant:
        raise ValueError("official generation returned no assistant audio rows")
    return normalized[: first_generated + len(assistant)], assistant, start_length, len(assistant)


def dump(args: argparse.Namespace) -> None:
    snapshot_evidence = authenticate_snapshot(args.snapshot)
    snapshot_evidence["server_tree"] = authenticate_server_tree(args.snapshot, snapshot_evidence["files"])
    rows = read_rows(args.prompt_rows)
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.snapshot, revision=REVISION, trust_remote_code=True, local_files_only=True, torch_dtype=torch.float32)
    model.eval()
    loaded_sources = [
        authenticate_loaded_source(
            type(model),
            label="MOSS Local modeling source",
            role="modeling_source",
            expected_digest="b0a66211943ae580b087f3e71495fea2f455701a4f6c29b6d3562218f7668c5f",
        ),
        authenticate_loaded_source(
            type(model.config),
            label="MOSS Local configuration source",
            role="configuration_source",
            expected_digest="ab6debcb92032cb9dc91ae80aed77dbadd2e59848208baef2b062bd6def3f3be",
        ),
    ]
    for loaded in loaded_sources:
        snapshot_role = snapshot_evidence["custom_code"].get(loaded["role"])
        if snapshot_role is None or snapshot_role["sha256"] != loaded["sha256"]:
            raise ValueError(f"loaded {loaded['role']} is not the authenticated snapshot file")
        loaded["authenticated_snapshot_path"] = snapshot_role["path"]
    prompt = torch.tensor([rows], dtype=torch.long)
    with torch.no_grad():
        generated = model.generate(input_ids=prompt, max_new_tokens=args.max_new_frames, do_sample=False)
    normalized, assistant, start_length, generated_frames = normalize_generated(rows, generated)
    write_rows(args.output, normalized)
    assistant_path = args.assistant_codes or args.output.with_name(f"{args.output.stem}.assistant-codes.u32le")
    # Keep each assistant frame as twelve values; this is a separate format.
    assistant_path.write_bytes(struct.pack(f"<{len(assistant) * 12}I", *(value for row in assistant for value in row[1:])))
    manifest = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "snapshot": snapshot_evidence,
        "loaded_custom_code": loaded_sources,
        "prompt": {"path": str(args.prompt_rows), "sha256": sha256_file(args.prompt_rows), "rows": len(rows), "columns": COLUMNS},
        "rows_from_audio_start": {"path": str(args.output), "sha256": sha256_file(args.output), "rows": len(normalized), "columns": COLUMNS, "start_length": start_length, "generated_frames": generated_frames},
        "assistant_codes": {"path": str(assistant_path), "sha256": sha256_file(assistant_path), "rows": len(assistant), "codebooks": 12},
        "terminal_row_present_in_official_output": len(normalized) < len(generated[0]),
        "reference_status": "AUTHENTICATED_EVIDENCE_COMPLETE",
        "parity_status": "MEASURED_NOT_GATED",
    }
    if args.manifest_output:
        args.manifest_output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"source={REPOSITORY}@{REVISION} rows={len(normalized)} columns={COLUMNS} verdict=MEASURED_NOT_GATED")


def self_test() -> None:
    import tempfile

    class FakeGenerated:
        ndim = 3

        def __init__(self, rows: list[list[int]]):
            self.shape = (1, len(rows), COLUMNS)
            self._rows = rows

        def __getitem__(self, index: int) -> "FakeGenerated":
            assert index == 0
            return self

        def to(self, **_: object) -> "FakeGenerated":
            return self

        def tolist(self) -> list[list[int]]:
            return self._rows

    with tempfile.TemporaryDirectory(prefix="moss-local-ref-") as directory:
        source = Path(directory) / "prompt.u32le"
        source.write_bytes(struct.pack("<13I", *range(COLUMNS)))
        assert read_rows(source) == [list(range(COLUMNS))]
        bad = Path(directory) / "bad.u32le"
        bad.write_bytes(b"x")
        try:
            read_rows(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("missing prompt must fail closed")
        output = Path(directory) / "rows.u32le"
        write_rows(output, [list(range(COLUMNS))])
        assert read_rows(output)[0] == list(range(COLUMNS))
    prompt = [[AUDIO_START_TOKEN, *([0] * (COLUMNS - 1))]]
    assistant = [AUDIO_ASSISTANT_SLOT_TOKEN] + list(range(1, COLUMNS))
    terminal = [AUDIO_END_TOKEN] + [0] * (COLUMNS - 1)
    normalized, codes, start_length, frame_count = normalize_generated(
        prompt, FakeGenerated(prompt + [assistant, terminal])
    )
    assert normalized == prompt + [assistant]
    assert codes == [assistant]
    assert start_length == 0 and frame_count == 1
    try:
        normalize_generated(prompt, FakeGenerated(prompt + [[999] + [0] * (COLUMNS - 1)]))
    except ValueError:
        pass
    else:
        raise AssertionError("unexpected decision row was accepted")
    assert REPOSITORY.startswith("OpenMOSS-Team/")
    assert re.fullmatch(r"[0-9a-f]{40}", REVISION)
    assert len(SOURCE_DIGESTS) == 7
    payload = b"moss-lfs-self-test"
    payload_sha = hashlib.sha256(payload).hexdigest()
    pointer = canonical_lfs_pointer(payload_sha, len(payload))
    assert re.fullmatch(r"[0-9a-f]{40}", git_blob_sha1_bytes(pointer))
    assert git_blob_sha1_bytes(canonical_lfs_pointer(payload_sha, len(payload) + 1)) != git_blob_sha1_bytes(pointer)
    print("moss_tts_local_dump_reference.py self-test: OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--prompt-rows", type=Path)
    parser.add_argument("--max-new-frames", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--assistant-codes", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.snapshot is None or args.prompt_rows is None or args.output is None:
        parser.error("real reference requires --snapshot --prompt-rows --output")
    if args.max_new_frames is None or args.max_new_frames <= 0:
        parser.error("--max-new-frames must be positive")
    dump(args)


if __name__ == "__main__":
    main()
