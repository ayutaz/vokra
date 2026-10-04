#!/usr/bin/env -S uv run --no-project --python 3.12
"""Authenticate the fixed DSM PyTorch PCM caller source receipt.

This audit uses Python's standard library only. It verifies already acquired
source/API/receipt files; it never imports the upstream package and never
downloads model, config, tokenizer, or weight artifacts. Network acquisition
is intentionally not implicit: pass immutable local receipt files explicitly.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from typing import Any

MAX_RECEIPT_BYTES = 64 * 1024
MAX_API_BYTES = 256 * 1024
MAX_READ_CHUNK = 64 * 1024
LEGACY_API_BODY_SHA256 = "b8f00e48832c4f41747c56cef46040f884afa81e632ba21d147ddd4bb9f7d2b5"
EXPECTED_SCOPE = (
    "Caller schedule and PCM preparation source identity only; no model, config, "
    "tokenizer, weight, license, parity, or publication claim."
)

EXPECTED = {
    "repository": "kyutai-labs/delayed-streams-modeling",
    "revision": "4c4f65e147df056adf3346290d64c7b9649b18c9",
    "path": "scripts/stt_from_file_pytorch.py",
    "bytes": 8452,
    "git_blob_sha1": "cf3fb05b0e0c1f265a667276d2886ce2664d79ff",
    "raw_sha256": "2ac2d9bff71d3d6a874bed070eb9d4e60736209e3cbaa9697fbe735dc79d2955",
    "api_body_sha256": "c07733a52c67cbb78536de4ba6e8cbdbdd6a5a4ec7d169e8ee03ab356ee9062e",
    "raw_url": "https://raw.githubusercontent.com/kyutai-labs/delayed-streams-modeling/4c4f65e147df056adf3346290d64c7b9649b18c9/scripts/stt_from_file_pytorch.py",
    "api_url": "https://api.github.com/repos/kyutai-labs/delayed-streams-modeling/contents/scripts/stt_from_file_pytorch.py?ref=4c4f65e147df056adf3346290d64c7b9649b18c9",
    "html_url": "https://github.com/kyutai-labs/delayed-streams-modeling/blob/4c4f65e147df056adf3346290d64c7b9649b18c9/scripts/stt_from_file_pytorch.py",
    "git_url": "https://api.github.com/repos/kyutai-labs/delayed-streams-modeling/git/blobs/cf3fb05b0e0c1f265a667276d2886ce2664d79ff",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()  # noqa: S324 - Git identity


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def read_stable_regular_file(path: Path, max_bytes: int, label: str) -> bytes:
    """Read one bounded, regular, non-symlink file without a mutable path race."""

    if max_bytes < 0:
        raise ValueError(f"{label}: max_bytes must be non-negative")
    try:
        listed = os.lstat(path)
    except OSError as error:
        raise ValueError(f"{label}: cannot stat {path}: {error}") from error
    if stat.S_ISLNK(listed.st_mode) or not stat.S_ISREG(listed.st_mode):
        raise ValueError(f"{label}: {path} must be a regular non-symlink file")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise ValueError(f"{label}: platform lacks O_NOFOLLOW")
    try:
        descriptor = os.open(path, os.O_RDONLY | nofollow)
    except OSError as error:
        raise ValueError(f"{label}: cannot open {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or _stat_identity(before) != _stat_identity(listed):
            raise ValueError(f"{label}: {path} changed before it was read")
        data = bytearray()
        while len(data) <= max_bytes:
            remaining = max_bytes + 1 - len(data)
            chunk = os.read(descriptor, min(MAX_READ_CHUNK, remaining))
            if not chunk:
                break
            data.extend(chunk)
        after = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(after) or len(data) != before.st_size:
            raise ValueError(f"{label}: {path} changed while it was read")
        if len(data) > max_bytes:
            raise ValueError(f"{label}: {path} exceeds the {max_bytes}-byte limit")
        return bytes(data)
    finally:
        os.close(descriptor)


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"JSON non-finite constant is not allowed: {value}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def parse_json_object(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label}: invalid strict JSON: {error}") from error
    if type(value) is not dict:
        raise ValueError(f"{label}: JSON root must be an object")
    return value


def _expect(value: dict[str, Any], key: str, expected_type: type[Any], label: str) -> Any:
    if key not in value or type(value[key]) is not expected_type:
        raise ValueError(f"{label}: {key} must have JSON type {expected_type.__name__}")
    return value[key]


def validate_receipt(receipt: dict[str, Any]) -> None:
    receipt_expected_keys = (
        "repository",
        "revision",
        "path",
        "bytes",
        "git_blob_sha1",
        "raw_sha256",
        "api_body_sha256",
        "raw_url",
        "api_url",
    )
    for key in receipt_expected_keys:
        expected = EXPECTED[key]
        if key not in receipt or type(receipt[key]) is not type(expected) or receipt[key] != expected:
            raise ValueError(f"receipt {key} mismatch: {receipt.get(key)!r}")
    if receipt.get("status") != "AUTHENTICATED_SOURCE_ONLY_RECEIPT":
        raise ValueError("receipt status is not source-only authenticated")
    if receipt.get("execution") != "NOT_RUN":
        raise ValueError("receipt execution must remain NOT_RUN")
    if receipt.get("scope") != EXPECTED_SCOPE:
        raise ValueError("receipt scope must remain the exact source-only scope")


def validate_api_payload(api: dict[str, Any], raw: bytes, expected: dict[str, Any]) -> None:
    string_fields = {
        "name": expected["path"].rsplit("/", 1)[-1],
        "path": expected["path"],
        "sha": expected["git_blob_sha1"],
        "url": expected["api_url"],
        "html_url": expected["html_url"],
        "git_url": expected["git_url"],
        "download_url": expected["raw_url"],
        "type": "file",
        "encoding": "base64",
    }
    for key, expected_value in string_fields.items():
        if _expect(api, key, str, "GitHub API") != expected_value:
            raise ValueError(f"GitHub API {key} mismatch")
    size = _expect(api, "size", int, "GitHub API")
    if size != len(raw) or size != expected["bytes"]:
        raise ValueError("GitHub API size mismatch")
    content = _expect(api, "content", str, "GitHub API")
    try:
        compact = b"".join(content.encode("ascii").split())
        decoded = base64.b64decode(compact, validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as error:
        raise ValueError(f"GitHub API content is not strict base64: {error}") from error
    if decoded != raw:
        raise ValueError("GitHub API base64 content does not equal the retained raw source")
    links = _expect(api, "_links", dict, "GitHub API")
    for key, expected_value in {
        "self": expected["api_url"],
        "git": expected["git_url"],
        "html": expected["html_url"],
    }.items():
        if _expect(links, key, str, "GitHub API _links") != expected_value:
            raise ValueError(f"GitHub API _links.{key} mismatch")


def audit_contents(raw: bytes, api_bytes: bytes, receipt_bytes: bytes) -> dict[str, object]:
    if len(raw) != EXPECTED["bytes"]:
        raise ValueError(f"raw source byte count mismatch: {len(raw)}")
    receipt = parse_json_object(receipt_bytes, "receipt")
    validate_receipt(receipt)
    api = parse_json_object(api_bytes, "GitHub API")
    validate_api_payload(api, raw, EXPECTED)
    observed = {
        "repository": EXPECTED["repository"],
        "revision": EXPECTED["revision"],
        "path": EXPECTED["path"],
        "bytes": len(raw),
        "git_blob_sha1": git_blob_sha1(raw),
        "raw_sha256": sha256(raw),
        "api_body_sha256": sha256(api_bytes),
    }
    if observed != {key: EXPECTED[key] for key in observed}:
        raise ValueError(f"source receipt mismatch: {observed!r}")
    return {
        "status": "AUTHENTICATED_SOURCE_ONLY_RECEIPT",
        "observed": observed,
        "execution": "NOT_RUN",
    }


def audit(raw_path: Path, api_path: Path, receipt_path: Path) -> dict[str, object]:
    raw = read_stable_regular_file(raw_path, EXPECTED["bytes"], "raw source")
    api = read_stable_regular_file(api_path, MAX_API_BYTES, "GitHub API")
    receipt = read_stable_regular_file(receipt_path, MAX_RECEIPT_BYTES, "receipt")
    return audit_contents(raw, api, receipt)


class SourceAuditTests(unittest.TestCase):
    def test_current_api_snapshot_pin_accepts_and_legacy_pin_rejects(self) -> None:
        receipt = dict(EXPECTED)
        receipt.update(
            {
                "status": "AUTHENTICATED_SOURCE_ONLY_RECEIPT",
                "execution": "NOT_RUN",
                "scope": EXPECTED_SCOPE,
            }
        )
        validate_receipt(receipt)
        receipt["api_body_sha256"] = LEGACY_API_BODY_SHA256
        with self.assertRaisesRegex(ValueError, "api_body_sha256 mismatch"):
            validate_receipt(receipt)

    def test_other_source_identity_changes_are_rejected(self) -> None:
        receipt = dict(EXPECTED)
        receipt.update(
            {
                "status": "AUTHENTICATED_SOURCE_ONLY_RECEIPT",
                "execution": "NOT_RUN",
                "scope": EXPECTED_SCOPE,
            }
        )
        for key, value in (
            ("repository", "other/repository"),
            ("revision", "0" * 40),
            ("path", "scripts/other.py"),
            ("raw_url", "https://example.invalid/raw.py"),
            ("api_url", "https://example.invalid/api.json"),
        ):
            mutated = dict(receipt)
            mutated[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, rf"receipt {key} mismatch"):
                validate_receipt(mutated)

    def test_scope_must_remain_exact_and_source_only(self) -> None:
        receipt = dict(EXPECTED)
        receipt.update(
            {
                "status": "AUTHENTICATED_SOURCE_ONLY_RECEIPT",
                "execution": "NOT_RUN",
                "scope": EXPECTED_SCOPE,
            }
        )
        validate_receipt(receipt)
        for scope in (
            None,
            1,
            "model claim: executed weights and publication approved",
            "Caller schedule and PCM preparation source identity only; no model claim.",
        ):
            mutated = dict(receipt)
            mutated["scope"] = scope
            with self.subTest(scope=scope), self.assertRaisesRegex(ValueError, "exact source-only"):
                validate_receipt(mutated)

    def test_strict_duplicate_json_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_json_object(b'{"path":"a","path":"b"}', "synthetic")

    def test_bounded_regular_file_rejects_oversize_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            regular = root / "source.py"
            regular.write_bytes(b"abc")
            self.assertEqual(read_stable_regular_file(regular, 3, "test"), b"abc")
            with self.assertRaisesRegex(ValueError, "exceeds"):
                read_stable_regular_file(regular, 2, "test")
            link = root / "link.py"
            link.symlink_to(regular)
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                read_stable_regular_file(link, 3, "test")

    def test_corrupt_api_blob_and_base64_are_rejected(self) -> None:
        raw = b"synthetic-source"
        expected = dict(EXPECTED)
        expected.update(
            {
                "bytes": len(raw),
                "git_blob_sha1": git_blob_sha1(raw),
                "raw_sha256": sha256(raw),
            }
        )
        api = {
            "name": "stt_from_file_pytorch.py",
            "path": expected["path"],
            "sha": expected["git_blob_sha1"],
            "size": len(raw),
            "url": expected["api_url"],
            "html_url": expected["html_url"],
            "git_url": expected["git_url"],
            "download_url": expected["raw_url"],
            "type": "file",
            "encoding": "base64",
            "content": base64.b64encode(raw).decode("ascii"),
            "_links": {
                "self": expected["api_url"],
                "git": expected["git_url"],
                "html": expected["html_url"],
            },
        }
        validate_api_payload(api, raw, expected)
        api["sha"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "sha mismatch"):
            validate_api_payload(api, raw, expected)
        api["sha"] = expected["git_blob_sha1"]
        api["content"] = "not base64"
        with self.assertRaisesRegex(ValueError, "base64"):
            validate_api_payload(api, raw, expected)


def self_test() -> int:
    result = unittest.TextTestRunner(verbosity=1).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(SourceAuditTests)
    )
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-file", type=Path)
    parser.add_argument("--api-file", type=Path)
    parser.add_argument("--receipt-file", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.raw_file is None or args.api_file is None:
        parser.error("--raw-file and --api-file are required unless --self-test is used")
    receipt = args.receipt_file or Path(__file__).with_name("kyutai_stt_pytorch_pcm_source_receipt.json")
    print(json.dumps(audit(args.raw_file, args.api_file, receipt), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
