#!/usr/bin/env -S uv run --no-project --offline --python 3.12 python
"""Fail-closed source-only observer seam candidate for Kyutai Moshi ASR.

This utility never compiles, executes, or imports the Rust source.  It accepts
only the exact pinned source and root Rust license bytes, then writes a new
patched file and a no-replace receipt.  The source input is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path


UPSTREAM_COMMIT = "e6a55d2722a65870ef52a6c9f6ecfc0e90f38362"
SOURCE_URL = (
    "https://raw.githubusercontent.com/kyutai-labs/moshi/"
    f"{UPSTREAM_COMMIT}/rust/moshi-core/src/asr.rs"
)
LICENSE_URL = (
    "https://raw.githubusercontent.com/kyutai-labs/moshi/"
    f"{UPSTREAM_COMMIT}/rust/LICENSE"
)
SOURCE_BLOB_SHA1 = "0f3ad70082f152e3250aa81ab4e0b48ffef550db"
SOURCE_BYTES = 9125
LICENSE_BLOB_SHA1 = "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64"
LICENSE_BYTES = 11357
PATCH_NAME = "moshi-asr-observer-seam-v1"
PATCH_MARKER = b"// Vokra candidate: optional ASR observation seam."


class PatchError(ValueError):
    """Raised for source drift, unsafe paths, duplicate patches, or writes."""


def raw_sha1(value: bytes) -> str:
    return hashlib.sha1(value).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_blob_sha1(value: bytes) -> str:
    header = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(header + value).hexdigest()


def _regular_file_without_symlink(path: Path, label: str) -> None:
    if not path.is_absolute():
        raise PatchError(f"{label} must be an absolute path")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            raise PatchError(f"{label} does not exist: {path}") from None
        if stat.S_ISLNK(mode):
            raise PatchError(f"{label} has a symlink component: {current}")
    if not stat.S_ISREG(os.lstat(path).st_mode):
        raise PatchError(f"{label} is not a regular file: {path}")


def _new_destination(path: Path, label: str) -> None:
    if not path.is_absolute():
        raise PatchError(f"{label} must be an absolute path")
    parent = path.parent
    current = Path(path.anchor)
    for part in path.parts[1:-1]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            raise PatchError(f"{label} parent does not exist: {parent}") from None
        if stat.S_ISLNK(mode):
            raise PatchError(f"{label} parent has a symlink component: {current}")
        if not stat.S_ISDIR(mode):
            raise PatchError(f"{label} parent component is not a directory: {current}")
    if not parent.is_dir():
        raise PatchError(f"{label} parent is not a directory: {parent}")
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return
    if stat.S_ISLNK(mode):
        raise PatchError(f"{label} is a symlink: {path}")
    raise PatchError(f"{label} already exists; refusing overwrite: {path}")


def _authenticated(path: Path, expected_sha1: str, expected_bytes: int, label: str) -> bytes:
    _regular_file_without_symlink(path, label)
    if expected_bytes < 0:
        raise PatchError(f"{label} expected byte count is invalid: {expected_bytes}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise PatchError(f"{label} could not be opened safely: {path}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise PatchError(f"{label} is not a regular file")
        if before.st_size != expected_bytes:
            raise PatchError(f"{label} byte count drift: {before.st_size} != {expected_bytes}")
        data = os.read(fd, expected_bytes + 1)
        after = os.fstat(fd)
        if len(data) != expected_bytes:
            raise PatchError(f"{label} short or oversized read: {len(data)} != {expected_bytes}")
        if after.st_size != expected_bytes:
            raise PatchError(f"{label} grew or shrank during read: {after.st_size} != {expected_bytes}")
        try:
            current = os.lstat(path)
        except FileNotFoundError as exc:
            raise PatchError(f"{label} was replaced during read: {path}") from exc
        if (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino):
            raise PatchError(f"{label} was replaced during read: {path}")
    finally:
        os.close(fd)
    digest = git_blob_sha1(data)
    if digest != expected_sha1:
        raise PatchError(f"{label} Git blob SHA-1 drift: {digest} != {expected_sha1}")
    return data


OBSERVER_TYPE = """pub type AsrObserver = Box<
    dyn FnMut(
        usize,
        &[ItemState],
        &crate::StreamMask,
        &Tensor,
        &Tensor,
        &[u32],
    ) -> Result<()> + Send,
>;

"""
OBSERVER_FIELD = "    observer: Option<AsrObserver>,\n"
OBSERVER_INIT = "            observer: None,\n"
OBSERVER_SETTER = """    pub fn set_observer(&mut self, observer: Option<AsrObserver>) {
        self.observer = observer;
    }

"""
OBSERVER_CALL = """            // Vokra candidate: optional ASR observation seam.
            if let Some(observer) = self.observer.as_mut() {
                observer(
                    self.model_step_idx,
                    self.batch.as_slice(),
                    mask,
                    &text_logits,
                    &transformer_out,
                    &text_tokens,
                )?;
            }
"""


def patch_bytes(source: bytes) -> bytes:
    """Apply only the five exact additive observer insertions."""
    if PATCH_MARKER in source:
        raise PatchError("observer patch marker already present; refusing double patch")
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PatchError("source is not UTF-8 Rust text") from exc
    replacements = (
        ("pub struct State {\n", OBSERVER_TYPE + "pub struct State {\n"),
        ("    batch: Vec<ItemState>,\n", "    batch: Vec<ItemState>,\n" + OBSERVER_FIELD),
        ("            batch: vec![item_state; batch_size],\n", "            batch: vec![item_state; batch_size],\n" + OBSERVER_INIT),
        (
            """    pub fn model_step_idx(&self) -> usize {
        self.model_step_idx
    }

""",
            """    pub fn model_step_idx(&self) -> usize {
        self.model_step_idx
    }

""" + OBSERVER_SETTER,
        ),
        (
            "            let text_tokens = text_tokens.to_vec1::<u32>()?;\n",
            "            let text_tokens = text_tokens.to_vec1::<u32>()?;\n" + OBSERVER_CALL,
        ),
    )
    for old, new in replacements:
        count = text.count(old)
        if count != 1:
            raise PatchError(f"source insertion anchor count is {count}, expected 1: {old!r}")
        text = text.replace(old, new, 1)
    patched = text.encode("utf-8")
    if patched == source or PATCH_MARKER not in patched:
        raise PatchError("observer patch did not produce the expected additive marker")
    return patched


def revert_bytes(patched: bytes) -> bytes:
    """Reverse the exact additive insertions for a byte-level proof."""
    if PATCH_MARKER not in patched:
        raise PatchError("observer patch marker is missing")
    text = patched.decode("utf-8")
    reversals = (
        ("pub type AsrObserver = Box<\n    dyn FnMut(\n        usize,\n        &[ItemState],\n        &crate::StreamMask,\n        &Tensor,\n        &Tensor,\n        &[u32],\n    ) -> Result<()> + Send,\n>;\n\n", ""),
        (OBSERVER_FIELD, ""),
        (OBSERVER_INIT, ""),
        (OBSERVER_SETTER, ""),
        (OBSERVER_CALL, ""),
    )
    for old, new in reversals:
        if text.count(old) != 1:
            raise PatchError(f"patched insertion count is not exactly one: {old!r}")
        text = text.replace(old, new, 1)
    return text.encode("utf-8")


def _write_exclusive(path: Path, data: bytes, label: str) -> tuple[int, int]:
    _new_destination(path, label)
    try:
        with path.open("xb") as handle:
            stat_result = os.fstat(handle.fileno())
            identity = (stat_result.st_dev, stat_result.st_ino)
            handle.write(data)
            handle.flush()
            return identity
    except FileExistsError as exc:
        raise PatchError(f"{label} appeared during write; refusing overwrite: {path}") from exc


def apply(source_path: Path, license_path: Path, output_path: Path, receipt_path: Path) -> dict[str, object]:
    source = _authenticated(source_path, SOURCE_BLOB_SHA1, SOURCE_BYTES, "source")
    license_bytes = _authenticated(license_path, LICENSE_BLOB_SHA1, LICENSE_BYTES, "root Rust LICENSE")
    if os.path.abspath(output_path) == os.path.abspath(receipt_path):
        raise PatchError("patched output and receipt must be distinct paths")
    _new_destination(output_path, "patched output")
    _new_destination(receipt_path, "receipt")
    patched = patch_bytes(source)
    receipt = {
        "schema": "vokra-moshi-asr-observer-source-only-v1",
        "patch": PATCH_NAME,
        "source": {
            "repository": "kyutai-labs/moshi",
            "commit": UPSTREAM_COMMIT,
            "path": "rust/moshi-core/src/asr.rs",
            "official_raw_url": SOURCE_URL,
            "git_blob_sha1": SOURCE_BLOB_SHA1,
            "bytes": SOURCE_BYTES,
            "authenticated_input": str(source_path),
        },
        "root_rust_license": {
            "path": "rust/LICENSE",
            "official_raw_url": LICENSE_URL,
            "git_blob_sha1": LICENSE_BLOB_SHA1,
            "bytes": LICENSE_BYTES,
            "authenticated_input": str(license_path),
        },
        "patched_output": {
            "path": str(output_path),
            "git_blob_sha1": git_blob_sha1(patched),
            "raw_sha1": raw_sha1(patched),
            "bytes": len(patched),
            "sha256": sha256_bytes(patched),
        },
        "reversible": revert_bytes(patched) == source,
        "observer_contract": {
            "callback_default": "None",
            "callback_error": "propagates as candle::Result error",
            "observed": ["global model_step_idx", "current ItemState slice", "StreamMask", "text_logits", "transformer_out", "selected tokens"],
            "call_site": "immediately after selected text_tokens.to_vec1 conversion, before state updates",
            "transformations": "none",
        },
        "gates": {
            "remote_build": "NOT_RUN",
            "runtime": "NOT_RUN",
            "dependency_license_owner_review": "BLOCKED_PENDING_REVIEW",
            "model_weights": "NOT_ACQUIRED",
            "publication": "NO_UPLOAD",
        },
        "source_blob_sha1": git_blob_sha1(source),
        "license_blob_sha1": git_blob_sha1(license_bytes),
        "source_sha256": sha256_bytes(source),
        "license_sha256": sha256_bytes(license_bytes),
    }
    _write_exclusive(output_path, patched, "patched output")
    _write_exclusive(receipt_path, json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n", "receipt")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--license", dest="license_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(apply(args.source, args.license_path, args.output, args.receipt), sort_keys=True))
    except (OSError, PatchError) as exc:
        print(f"observer patch refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
