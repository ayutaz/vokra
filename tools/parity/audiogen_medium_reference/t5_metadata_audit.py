"""Authenticate the AudioGen T5 companion from Hugging Face metadata only.

The audit calls the model-info endpoint with ``blobs=true``.  It never calls a
resolve URL and therefore never downloads a tokenizer or model payload.  The
resulting packet is an input to the next model-free AudioGen inspection; a
manifest must not be emitted as authenticated unless this packet validates.
"""

from __future__ import annotations

import json
import argparse
import os
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from companion_contract import T5_METADATA_FILES, T5_METADATA_SCHEMA, T5_REPOSITORY, T5_REVISION, T5_WEIGHT_PATH, t5_server_metadata_packet


SCHEMA = T5_METADATA_SCHEMA
LICENSE = "apache-2.0"
REQUIRED_FILES = tuple(sorted(T5_METADATA_FILES))
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


def _request() -> dict[str, Any]:
    encoded = urllib.parse.quote(T5_REPOSITORY, safe="/")
    url = f"https://huggingface.co/api/models/{encoded}/revision/{T5_REVISION}?blobs=true"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"T5 metadata request failed: {error}") from error
    if len(body) > MAX_RESPONSE_BYTES:
        raise RuntimeError("T5 metadata response exceeded the bounded size")
    try:
        value = json.loads(body, object_pairs_hook=_strict_pairs)
    except json.JSONDecodeError as error:
        raise RuntimeError("T5 metadata response was not JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError("T5 metadata response was not an object")
    return value


def _blob_id(entry: dict[str, Any]) -> str | None:
    for key in ("blobId", "blob_id", "oid"):
        value = entry.get(key)
        if isinstance(value, str):
            return value
    return None


def _lfs_value(entry: dict[str, Any], key: str) -> object:
    lfs = entry.get("lfs")
    return lfs.get(key) if isinstance(lfs, dict) else None


def build_packet(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("id") != T5_REPOSITORY or value.get("sha") != T5_REVISION:
        raise RuntimeError("T5 metadata repository/revision mismatch")
    card = value.get("cardData")
    if not isinstance(card, dict) or card.get("license") != LICENSE:
        raise RuntimeError("T5 metadata license mismatch")
    siblings = value.get("siblings")
    if not isinstance(siblings, list):
        raise RuntimeError("T5 metadata did not expose a sibling file list")
    by_name: dict[str, dict[str, Any]] = {}
    for entry in siblings:
        if not isinstance(entry, dict) or not isinstance(entry.get("rfilename"), str):
            raise RuntimeError("T5 metadata contained an invalid sibling entry")
        name = entry["rfilename"]
        if name in by_name:
            raise RuntimeError(f"duplicate T5 metadata sibling: {name}")
        by_name[name] = entry
    rows: list[dict[str, Any]] = []
    for name in REQUIRED_FILES:
        entry = by_name.get(name)
        if entry is None:
            raise RuntimeError(f"required T5 metadata file is missing: {name}")
        fixed = T5_METADATA_FILES[name]
        if entry.get("size") != fixed["bytes"]:
            raise RuntimeError(f"T5 metadata size mismatch: {name}")
        blob = _blob_id(entry)
        if name == T5_WEIGHT_PATH:
            if blob != fixed["lfs_pointer_git_blob_sha1"] or _lfs_value(entry, "sha256") != fixed["lfs_payload_sha256"] or _lfs_value(entry, "size") != fixed["lfs_payload_size"]:
                raise RuntimeError(f"T5 metadata LFS identity mismatch: {name}")
        else:
            if blob != fixed["git_blob_sha1"] or any(_lfs_value(entry, key) is not None for key in ("sha256", "size")):
                raise RuntimeError(f"T5 metadata Git identity mismatch: {name}")
        rows.append({"path": name, "type": "file", **fixed})
    packet = t5_server_metadata_packet()
    if packet["files"] != rows:
        raise RuntimeError("T5 metadata packet canonical file identity mismatch")
    return packet


def validate_packet(packet: dict[str, Any]) -> None:
    if packet != t5_server_metadata_packet():
        raise RuntimeError("T5 metadata packet identity drift")


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load_packet(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"invalid T5 metadata packet: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError("T5 metadata packet is not an object")
    validate_packet(value)
    return value


def write_packet(output: Path, packet: dict[str, Any]) -> None:
    if output.exists() or output.is_symlink():
        raise RuntimeError("refusing to clobber T5 metadata packet")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(packet, stream, sort_keys=True, indent=2)
        stream.write("\n")
    try:
        os.link(temporary, output)
    except FileExistsError as error:
        raise RuntimeError("refusing to clobber T5 metadata packet") from error
    finally:
        temporary.unlink(missing_ok=True)


def _safe_output(path: Path) -> None:
    if not path.is_absolute() or "\x00" in str(path) or any(part in {".", ".."} for part in path.parts):
        raise RuntimeError("T5 metadata output must be an absolute path without dot components")
    if path.exists() or path.is_symlink():
        raise RuntimeError("T5 metadata output already exists or is symlinked")
    ancestor = path.parent
    while ancestor != ancestor.parent:
        if ancestor.is_symlink():
            raise RuntimeError("T5 metadata output has a symlink ancestor")
        ancestor = ancestor.parent


def self_test() -> None:
    siblings = []
    for name, fixed in T5_METADATA_FILES.items():
        if name == T5_WEIGHT_PATH:
            siblings.append({"rfilename": name, "size": fixed["bytes"], "blobId": fixed["lfs_pointer_git_blob_sha1"], "lfs": {"sha256": fixed["lfs_payload_sha256"], "size": fixed["lfs_payload_size"]}})
        else:
            siblings.append({"rfilename": name, "size": fixed["bytes"], "blobId": fixed["git_blob_sha1"]})
    source = {"id": T5_REPOSITORY, "sha": T5_REVISION, "siblings": siblings, "cardData": {"license": LICENSE}}
    packet = build_packet(source)
    validate_packet(packet)
    with tempfile.TemporaryDirectory(prefix="audiogen-t5-metadata-") as directory:
        target = Path(directory) / "t5.json"
        write_packet(target, packet)
        try:
            write_packet(target, packet)
        except RuntimeError:
            pass
        else:
            raise AssertionError("T5 metadata packet clobber was accepted")
        duplicate = Path(directory) / "duplicate.json"
        duplicate.write_text('{"schema":"x","schema":"y"}', encoding="utf-8")
        try:
            load_packet(duplicate)
        except RuntimeError:
            pass
        else:
            raise AssertionError("duplicate metadata JSON keys were accepted")
    for invalid in (
        {**source, "sha": "0" * 40},
        {**source, "cardData": {"license": "mit"}},
        {**source, "siblings": [{**siblings[0], "blobId": "0" * 40}, *siblings[1:]]},
        {**source, "siblings": [*siblings[:-1], {**siblings[-1], "lfs": {"sha256": "0" * 64, "size": T5_METADATA_FILES[T5_WEIGHT_PATH]["bytes"]}}]},
    ):
        try:
            build_packet(invalid)
        except RuntimeError:
            pass
        else:
            raise AssertionError("T5 metadata drift was accepted")
    print("audiogen_medium_t5_metadata_audit --self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.output is not None:
            parser.error("--self-test cannot be combined with an output path")
        self_test()
        return 0
    if args.output is None:
        parser.error("an output path is required unless --self-test is used")
    try:
        output = Path(args.output)
        _safe_output(output)
        write_packet(output, build_packet(_request()))
    except (OSError, RuntimeError, UnicodeError, ValueError) as error:
        print(f"T5 metadata audit BLOCKED: {error}", file=sys.stderr)
        return 2
    print(f"PASS_T5_METADATA repository={T5_REPOSITORY} revision={T5_REVISION} payload=NOT_DOWNLOADED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
