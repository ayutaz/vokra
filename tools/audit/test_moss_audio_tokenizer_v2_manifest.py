"""Hermetic parser checks for the MOSS Audio Tokenizer v2 manifest audit."""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from tools.audit import moss_audio_tokenizer_v2_manifest as manifest


class JsonDuplicateKeyTests(unittest.TestCase):
    def test_duplicate_keys_are_rejected_at_any_object_depth(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON key: a"):
            manifest.load_json('{"a": 1, "a": 2}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON key: nested"):
            manifest.load_json('{"outer": {"nested": 1, "nested": 2}}')

    def test_safetensors_header_uses_duplicate_key_rejecting_parser(self) -> None:
        header = b'{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"weight":{}}'
        with tempfile.TemporaryDirectory(prefix="moss-v2-manifest-test-") as directory:
            path = Path(directory) / "duplicate.safetensors"
            path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
            with self.assertRaisesRegex(ValueError, "duplicate JSON key: weight"):
                manifest.safetensors_header(path)


if __name__ == "__main__":
    unittest.main()
