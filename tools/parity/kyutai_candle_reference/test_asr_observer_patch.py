"""Offline stdlib tests for the Moshi ASR observer candidate patcher."""

from __future__ import annotations

import contextlib
import io
import importlib.util
import json
import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path


LEAF = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("asr_observer_patch", LEAF / "asr_observer_patch.py")
assert SPEC is not None and SPEC.loader is not None
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)
PINNED_SOURCE = LEAF / "pinned-asr.rs"
PINNED_LICENSE = LEAF / "pinned-LICENSE"


@contextlib.contextmanager
def temporary_directory(prefix: str):
    """Use the system temp root after resolving Mac /tmp-style aliases."""
    canonical_root = Path(tempfile.gettempdir()).resolve()
    with tempfile.TemporaryDirectory(prefix=prefix, dir=canonical_root) as directory:
        yield Path(directory).resolve()


SYNTHETIC_SOURCE = b"""use candle::{Result, Tensor};

pub struct ItemState;
pub struct State {
    batch: Vec<ItemState>,
}
impl State {
    pub fn model_step_idx(&self) -> usize {
        self.model_step_idx
    }

    pub fn new() {
        let mut s = Self {
            batch: vec![item_state; batch_size],
        };
    }

    fn step() -> Result<()> {
            let text_tokens = text_tokens.to_vec1::<u32>()?;
            for (batch_idx, (text_token, item)) in text_tokens.into_iter().zip(self.batch.iter_mut()).enumerate()
            {
                let _ = (batch_idx, text_token, item);
            }
        Ok(())
    }
}
"""
SYNTHETIC_LICENSE = b"Apache License\n"


@contextlib.contextmanager
def synthetic_spec():
    names = ("SOURCE_BLOB_SHA1", "SOURCE_BYTES", "LICENSE_BLOB_SHA1", "LICENSE_BYTES")
    old = {name: getattr(PATCHER, name) for name in names}
    PATCHER.SOURCE_BLOB_SHA1 = PATCHER.git_blob_sha1(SYNTHETIC_SOURCE)
    PATCHER.SOURCE_BYTES = len(SYNTHETIC_SOURCE)
    PATCHER.LICENSE_BLOB_SHA1 = PATCHER.git_blob_sha1(SYNTHETIC_LICENSE)
    PATCHER.LICENSE_BYTES = len(SYNTHETIC_LICENSE)
    try:
        yield
    finally:
        for name, value in old.items():
            setattr(PATCHER, name, value)


class ObserverPatchTests(unittest.TestCase):
    def test_pinned_identity_and_urls_are_fixed(self) -> None:
        self.assertEqual(PATCHER.UPSTREAM_COMMIT, "e6a55d2722a65870ef52a6c9f6ecfc0e90f38362")
        self.assertEqual(PATCHER.SOURCE_BLOB_SHA1, "0f3ad70082f152e3250aa81ab4e0b48ffef550db")
        self.assertEqual(PATCHER.LICENSE_BLOB_SHA1, "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64")
        self.assertIn(PATCHER.UPSTREAM_COMMIT, PATCHER.SOURCE_URL)
        self.assertIn(PATCHER.UPSTREAM_COMMIT, PATCHER.LICENSE_URL)

    def test_authentication_accepts_exact_synthetic_fixture(self) -> None:
        with synthetic_spec(), self.fixture_files() as (source, license_path):
            self.assertEqual(PATCHER._authenticated(source, PATCHER.SOURCE_BLOB_SHA1, len(SYNTHETIC_SOURCE), "source"), SYNTHETIC_SOURCE)
            self.assertEqual(PATCHER._authenticated(license_path, PATCHER.LICENSE_BLOB_SHA1, len(SYNTHETIC_LICENSE), "license"), SYNTHETIC_LICENSE)

    def test_source_drift_wrong_size_and_license_drift_refuse(self) -> None:
        with synthetic_spec(), self.fixture_files() as (source, license_path):
            with self.assertRaises(PATCHER.PatchError):
                PATCHER._authenticated(source.parent / "missing.rs", PATCHER.SOURCE_BLOB_SHA1, PATCHER.SOURCE_BYTES, "source")
            source.write_bytes(SYNTHETIC_SOURCE + b"drift")
            with self.assertRaises(PATCHER.PatchError):
                PATCHER._authenticated(source, PATCHER.SOURCE_BLOB_SHA1, PATCHER.SOURCE_BYTES, "source")
            source.write_bytes(SYNTHETIC_SOURCE)
            license_path.write_bytes(b"wrong")
            with self.assertRaises(PATCHER.PatchError):
                PATCHER._authenticated(license_path, PATCHER.LICENSE_BLOB_SHA1, PATCHER.LICENSE_BYTES, "license")

    def test_same_size_mutations_fail_git_blob_authentication(self) -> None:
        with temporary_directory(prefix="moshi-asr-auth-") as directory:
            root = Path(directory)
            source = root / "asr.rs"
            license_path = root / "LICENSE"
            source_bytes = PINNED_SOURCE.read_bytes()
            license_bytes = PINNED_LICENSE.read_bytes()
            source.write_bytes(source_bytes[:100] + bytes([source_bytes[100] ^ 1]) + source_bytes[101:])
            license_path.write_bytes(license_bytes[:100] + bytes([license_bytes[100] ^ 1]) + license_bytes[101:])
            self.assertEqual(source.stat().st_size, PATCHER.SOURCE_BYTES)
            self.assertEqual(license_path.stat().st_size, PATCHER.LICENSE_BYTES)
            with self.assertRaises(PATCHER.PatchError):
                PATCHER._authenticated(source, PATCHER.SOURCE_BLOB_SHA1, PATCHER.SOURCE_BYTES, "source")
            with self.assertRaises(PATCHER.PatchError):
                PATCHER._authenticated(license_path, PATCHER.LICENSE_BLOB_SHA1, PATCHER.LICENSE_BYTES, "license")

    def test_oversize_rejected_before_payload_read(self) -> None:
        with synthetic_spec(), self.fixture_files() as (source, _license_path):
            source.write_bytes(SYNTHETIC_SOURCE + b"oversize")
            with mock.patch.object(PATCHER.os, "read", side_effect=AssertionError("payload read must not occur")) as read:
                with self.assertRaises(PATCHER.PatchError):
                    PATCHER._authenticated(source, PATCHER.SOURCE_BLOB_SHA1, PATCHER.SOURCE_BYTES, "source")
            read.assert_not_called()

    def test_temp_root_resolves_system_directory_alias(self) -> None:
        with tempfile.TemporaryDirectory(prefix="moshi-asr-temp-parent-") as parent:
            parent_path = Path(parent).resolve()
            alias = parent_path / "tmp-alias"
            alias.symlink_to(parent_path, target_is_directory=True)
            with mock.patch.object(tempfile, "gettempdir", return_value=str(alias)):
                with temporary_directory(prefix="moshi-asr-canonical-") as directory:
                    self.assertEqual(Path(directory).parent, parent_path)
                    self.assertEqual(Path(directory), Path(directory).resolve())

    def test_short_read_is_bounded_and_rejected(self) -> None:
        with synthetic_spec(), self.fixture_files() as (source, _license_path):
            real_read = PATCHER.os.read

            def short_read(fd: int, size: int) -> bytes:
                payload = real_read(fd, size)
                return payload[:-1]

            with mock.patch.object(PATCHER.os, "read", side_effect=short_read) as read:
                with self.assertRaises(PATCHER.PatchError):
                    PATCHER._authenticated(source, PATCHER.SOURCE_BLOB_SHA1, PATCHER.SOURCE_BYTES, "source")
            read.assert_called_once_with(mock.ANY, PATCHER.SOURCE_BYTES + 1)

    def test_growth_after_initial_size_check_is_rejected(self) -> None:
        with synthetic_spec(), self.fixture_files() as (source, _license_path):
            real_fstat = PATCHER.os.fstat
            calls = 0

            def growing_fstat(fd: int):
                nonlocal calls
                calls += 1
                result = real_fstat(fd)
                if calls == 2:
                    return mock.Mock(
                        st_mode=result.st_mode,
                        st_size=result.st_size + 1,
                        st_dev=result.st_dev,
                        st_ino=result.st_ino,
                    )
                return result

            with mock.patch.object(PATCHER.os, "fstat", side_effect=growing_fstat):
                with self.assertRaises(PATCHER.PatchError):
                    PATCHER._authenticated(source, PATCHER.SOURCE_BLOB_SHA1, PATCHER.SOURCE_BYTES, "source")

    def test_patch_is_additive_single_call_and_reversible(self) -> None:
        patched = PATCHER.patch_bytes(SYNTHETIC_SOURCE)
        self.assertIn(PATCHER.PATCH_MARKER, patched)
        self.assertEqual(patched.count(PATCHER.PATCH_MARKER), 1)
        self.assertEqual(PATCHER.revert_bytes(patched), SYNTHETIC_SOURCE)
        with self.assertRaises(PATCHER.PatchError):
            PATCHER.patch_bytes(patched)

    def test_apply_preserves_source_writes_receipt_without_overwrite(self) -> None:
        with synthetic_spec(), self.fixture_files() as (source, license_path):
            before = source.read_bytes()
            output = source.parent / "patched.rs"
            receipt_path = source.parent / "receipt.json"
            receipt = PATCHER.apply(source, license_path, output, receipt_path)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(receipt["source_blob_sha1"], PATCHER.git_blob_sha1(before))
            self.assertTrue(receipt["reversible"])
            self.assertEqual(json.loads(receipt_path.read_text())["patch"], PATCHER.PATCH_NAME)
            with self.assertRaises(PATCHER.PatchError):
                PATCHER.apply(source, license_path, output, source.parent / "second.json")
            self.assertEqual(output.read_bytes(), PATCHER.patch_bytes(before))

    def test_full_pinned_source_and_license_integration(self) -> None:
        with temporary_directory(prefix="moshi-asr-full-") as directory:
            root = Path(directory)
            output = root / "patched.rs"
            receipt_path = root / "receipt.json"
            before_source = PINNED_SOURCE.read_bytes()
            receipt = PATCHER.apply(PINNED_SOURCE, PINNED_LICENSE, output, receipt_path)
            self.assertEqual(receipt["source"]["git_blob_sha1"], PATCHER.SOURCE_BLOB_SHA1)
            self.assertEqual(receipt["root_rust_license"]["git_blob_sha1"], PATCHER.LICENSE_BLOB_SHA1)
            self.assertEqual(receipt["source_sha256"], PATCHER.sha256_bytes(before_source))
            self.assertTrue(receipt["reversible"])
            self.assertEqual(PATCHER.revert_bytes(output.read_bytes()), before_source)
            self.assertEqual(PINNED_SOURCE.read_bytes(), before_source)

    def test_preexisting_receipt_rejected_before_output_write(self) -> None:
        with temporary_directory(prefix="moshi-asr-receipt-") as directory:
            root = Path(directory)
            output = root / "patched.rs"
            receipt_path = root / "receipt.json"
            receipt_path.write_bytes(b"foreign receipt")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(
                    PATCHER.main(
                        [
                            "--source", str(PINNED_SOURCE),
                            "--license", str(PINNED_LICENSE),
                            "--output", str(output),
                            "--receipt", str(receipt_path),
                        ]
                    ),
                    2,
                )
            self.assertFalse(output.exists())
            self.assertEqual(receipt_path.read_bytes(), b"foreign receipt")

    def test_receipt_write_failure_preserves_own_unaccepted_output(self) -> None:
        with temporary_directory(prefix="moshi-asr-own-output-") as directory:
            root = Path(directory)
            output = root / "patched.rs"
            receipt_path = root / "receipt.json"
            real_write = PATCHER._write_exclusive

            def fail_receipt(path: Path, data: bytes, label: str):
                if label == "receipt":
                    raise OSError("simulated receipt write failure")
                return real_write(path, data, label)

            with mock.patch.object(PATCHER, "_write_exclusive", side_effect=fail_receipt):
                with self.assertRaises(OSError):
                    PATCHER.apply(PINNED_SOURCE, PINNED_LICENSE, output, receipt_path)
            self.assertEqual(output.read_bytes(), PATCHER.patch_bytes(PINNED_SOURCE.read_bytes()))
            self.assertFalse(receipt_path.exists())

    def test_racing_output_creator_is_preserved(self) -> None:
        with temporary_directory(prefix="moshi-asr-race-") as directory:
            root = Path(directory)
            output = root / "patched.rs"
            receipt_path = root / "receipt.json"

            def race(path: Path, data: bytes, label: str):
                if label == "patched output":
                    path.write_bytes(b"race winner")
                    raise OSError("simulated concurrent creator")
                return PATCHER._write_exclusive(path, data, label)

            with mock.patch.object(PATCHER, "_write_exclusive", side_effect=race):
                with self.assertRaises(OSError):
                    PATCHER.apply(PINNED_SOURCE, PINNED_LICENSE, output, receipt_path)
            self.assertEqual(output.read_bytes(), b"race winner")
            self.assertFalse(receipt_path.exists())

    def test_receipt_write_failure_preserves_replacement(self) -> None:
        with temporary_directory(prefix="moshi-asr-replacement-") as directory:
            root = Path(directory)
            output = root / "patched.rs"
            receipt_path = root / "receipt.json"
            real_write = PATCHER._write_exclusive
            original_identity: tuple[int, int] | None = None

            def fail_receipt(path: Path, data: bytes, label: str):
                nonlocal original_identity
                if label == "patched output":
                    original_identity = real_write(path, data, label)
                    replacement = root / "replacement.tmp"
                    replacement.write_bytes(b"replacement winner")
                    os.replace(replacement, path)
                    self.assertNotEqual(
                        original_identity,
                        (os.stat(path, follow_symlinks=False).st_dev, os.stat(path, follow_symlinks=False).st_ino),
                    )
                    return original_identity
                raise OSError("simulated receipt write failure")

            with mock.patch.object(PATCHER, "_write_exclusive", side_effect=fail_receipt):
                with self.assertRaises(OSError):
                    PATCHER.apply(PINNED_SOURCE, PINNED_LICENSE, output, receipt_path)
            self.assertEqual(output.read_bytes(), b"replacement winner")
            self.assertFalse(receipt_path.exists())

    def test_output_and_receipt_must_be_distinct(self) -> None:
        with temporary_directory(prefix="moshi-asr-distinct-") as directory:
            path = Path(directory) / "same"
            with self.assertRaises(PATCHER.PatchError):
                PATCHER.apply(PINNED_SOURCE, PINNED_LICENSE, path, path)
            self.assertFalse(path.exists())

    def test_symlink_inputs_ancestors_and_destinations_refuse(self) -> None:
        with synthetic_spec(), self.fixture_files() as (source, license_path):
            symlink_source = source.parent / "source-link.rs"
            os.symlink(source, symlink_source)
            with self.assertRaises(PATCHER.PatchError):
                PATCHER._authenticated(symlink_source, PATCHER.SOURCE_BLOB_SHA1, PATCHER.SOURCE_BYTES, "source")
            existing = source.parent / "existing.rs"
            existing.write_bytes(b"keep")
            with self.assertRaises(PATCHER.PatchError):
                PATCHER.apply(source, license_path, existing, source.parent / "receipt.json")
            output_link = source.parent / "output-link.rs"
            target = source.parent / "target.rs"
            target.write_bytes(b"keep")
            os.symlink(target, output_link)
            with self.assertRaises(PATCHER.PatchError):
                PATCHER.apply(source, license_path, output_link, source.parent / "receipt.json")
            linked_parent = source.parent / "linked-parent"
            os.symlink(source.parent, linked_parent)
            with self.assertRaises(PATCHER.PatchError):
                PATCHER.apply(source, license_path, linked_parent / "out.rs", source.parent / "receipt.json")

    @contextlib.contextmanager
    def fixture_files(self):
        with temporary_directory(prefix="moshi-asr-observer-test-") as directory:
            root = Path(directory)
            source = root / "asr.rs"
            license_path = root / "LICENSE"
            source.write_bytes(SYNTHETIC_SOURCE)
            license_path.write_bytes(SYNTHETIC_LICENSE)
            yield source, license_path


if __name__ == "__main__":
    unittest.main()
