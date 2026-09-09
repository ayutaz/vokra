#!/usr/bin/env python3
"""Collect AudioGen HF metadata without requesting repository payloads.

This audit intentionally uses only the public model-info endpoint.  It does
not call a resolve/download URL and fails closed if the endpoint omits any
identity required by the fixed inspection contract.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


REPOSITORY = "facebook/audiogen-medium"
REVISION = "1277dd7dfd8fa57a205a70acc5de0ee90804502f"
PAYLOAD_STATUS = "NOT_DOWNLOADED"
EXPECTED = {
    ".gitattributes": {
        "bytes": 1519,
        "git_blob_sha1": "a6344aac8c09253b3b630fb776ae94478aa0275b",
    },
    "README.md": {
        "bytes": 2240,
        "git_blob_sha1": "31a77819df582937de900237706f104a325e223f",
    },
    "compression_state_dict.bin": {
        "bytes": 235740815,
        "lfs_pointer_git_blob_sha1": "0cc8de6c4cf0c16326ee3c693385370b98bbf0f2",
        "lfs_payload_sha256": "5a520e64ca99226a9956f83b06df0617b713183fcdc384779883a6bb46dc1095",
    },
    "state_dict.bin": {
        "bytes": 3678455287,
        "lfs_pointer_git_blob_sha1": "ae572ad32705a0a9ba679b0d2813cbae716d869e",
        "lfs_payload_sha256": "f3b20997834de1ca47d6a31d00a5dc37019b279c7c8f250fd482d56def04faaa",
    },
}


def _request() -> dict:
    encoded = urllib.parse.quote(REPOSITORY, safe="/")
    url = f"https://huggingface.co/api/models/{encoded}/revision/{REVISION}?blobs=true"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"HF metadata request failed: {error}") from error
    try:
        value = json.loads(body)
    except json.JSONDecodeError as error:
        raise RuntimeError("HF metadata response was not JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError("HF metadata response was not an object")
    return value


def _blob_id(entry: dict) -> str | None:
    for key in ("blobId", "blob_id", "oid"):
        value = entry.get(key)
        if isinstance(value, str):
            return value
    return None


def _lfs_value(entry: dict, key: str) -> object:
    lfs = entry.get("lfs")
    if not isinstance(lfs, dict):
        return None
    return lfs.get(key)


def build_packet(value: dict) -> dict:
    if value.get("sha") != REVISION:
        raise RuntimeError("HF metadata resolved revision mismatch")
    siblings = value.get("siblings")
    if not isinstance(siblings, list):
        raise RuntimeError("HF metadata did not expose a sibling file list")
    by_name: dict[str, dict] = {}
    for entry in siblings:
        if not isinstance(entry, dict) or not isinstance(entry.get("rfilename"), str):
            raise RuntimeError("HF metadata contained an invalid sibling entry")
        name = entry["rfilename"]
        if name in by_name:
            raise RuntimeError(f"duplicate HF metadata sibling: {name}")
        by_name[name] = entry
    if set(by_name) != set(EXPECTED):
        raise RuntimeError(f"HF metadata file set mismatch: {sorted(by_name)}")
    rows = []
    for name in sorted(EXPECTED):
        entry = by_name[name]
        fixed = EXPECTED[name]
        if entry.get("size") != fixed["bytes"]:
            raise RuntimeError(f"HF metadata size mismatch: {name}")
        blob = _blob_id(entry)
        if "git_blob_sha1" in fixed:
            if blob != fixed["git_blob_sha1"] or _lfs_value(entry, "sha256") is not None:
                raise RuntimeError(f"HF metadata Git identity mismatch: {name}")
            row = {
                "path": name,
                "type": "file",
                "size": fixed["bytes"],
                "git_blob_sha1": blob,
                "lfs_pointer_git_blob_sha1": None,
                "lfs_payload_sha256": None,
                "lfs_payload_size": None,
            }
        else:
            if blob != fixed["lfs_pointer_git_blob_sha1"] or _lfs_value(entry, "sha256") != fixed["lfs_payload_sha256"] or _lfs_value(entry, "size") not in (None, fixed["bytes"]):
                raise RuntimeError(f"HF metadata LFS identity mismatch: {name}")
            row = {
                "path": name,
                "type": "file",
                "size": fixed["bytes"],
                "git_blob_sha1": None,
                "lfs_pointer_git_blob_sha1": blob,
                "lfs_payload_sha256": fixed["lfs_payload_sha256"],
                "lfs_payload_size": fixed["bytes"],
            }
        rows.append(row)
    card = value.get("cardData")
    license_value = card.get("license") if isinstance(card, dict) else None
    if license_value != "cc-by-nc-4.0":
        raise RuntimeError(f"HF model-card license mismatch: {license_value!r}")
    return {
        "repository": REPOSITORY,
        "requested_revision": REVISION,
        "resolved_revision": REVISION,
        "walk": "recursive_file_only",
        "files": rows,
        "model_card": {"path": "README.md", "license": license_value},
    }


def write_packet(output: Path, packet: dict) -> None:
    if output.exists() or output.is_symlink():
        raise RuntimeError("refusing to clobber metadata packet")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(packet, stream, sort_keys=True, indent=2)
        stream.write("\n")
    try:
        os.link(temporary, output)
    except FileExistsError as error:
        raise RuntimeError("refusing to clobber metadata packet") from error
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        siblings = []
        for name, fixed in EXPECTED.items():
            if "git_blob_sha1" in fixed:
                siblings.append({"rfilename": name, "size": fixed["bytes"], "blobId": fixed["git_blob_sha1"]})
            else:
                siblings.append({"rfilename": name, "size": fixed["bytes"], "blobId": fixed["lfs_pointer_git_blob_sha1"], "lfs": {"sha256": fixed["lfs_payload_sha256"], "size": fixed["bytes"]}})
        packet = build_packet({"sha": REVISION, "siblings": siblings, "cardData": {"license": "cc-by-nc-4.0"}})
        if set(packet) != {"repository", "requested_revision", "resolved_revision", "walk", "files", "model_card"} or PAYLOAD_STATUS != "NOT_DOWNLOADED":
            raise SystemExit("metadata audit self-test failed")
        with tempfile.TemporaryDirectory(prefix="audiogen-medium-metadata-") as directory:
            target = Path(directory) / "tree.json"
            write_packet(target, packet)
            try:
                write_packet(target, packet)
            except RuntimeError:
                pass
            else:
                raise SystemExit("metadata packet clobber was accepted")
        for invalid in (
            {"sha": "0" * 40, "siblings": siblings, "cardData": {"license": "cc-by-nc-4.0"}},
            {"sha": REVISION, "siblings": [{**siblings[0], "size": 1}, *siblings[1:]], "cardData": {"license": "cc-by-nc-4.0"}},
            {"sha": REVISION, "siblings": siblings, "cardData": {"license": "cc-by-4.0"}},
        ):
            try:
                build_packet(invalid)
            except RuntimeError:
                pass
            else:
                raise SystemExit("metadata audit accepted invalid metadata")
        print("audiogen_medium_metadata_audit --self-test: OK")
        return 0
    if len(sys.argv) != 2:
        raise SystemExit("usage: metadata_audit.py <output-packet>")
    output = Path(sys.argv[1])
    packet = build_packet(_request())
    write_packet(output, packet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
