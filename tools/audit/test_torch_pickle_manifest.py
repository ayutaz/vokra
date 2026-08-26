#!/usr/bin/env python3
"""Tests for the fail-closed PyTorch state-dict manifest reader."""

from __future__ import annotations

import io
import unittest

from torch_pickle_manifest import ManifestError, load_manifest, render_manifest


def plain_state_dict_pickle() -> bytes:
    """Protocol-2 pickle for OrderedDict(weight=tensor([2, 3]))."""

    return (
        b"\x80\x02"
        b"ccollections\nOrderedDict\n)R("
        b"X\x06\x00\x00\x00weight"
        b"ctorch._utils\n_rebuild_tensor_v2\n("
        b"(X\x07\x00\x00\x00storage"
        b"ctorch\nFloatStorage\n"
        b"X\x01\x00\x00\x000"
        b"X\x03\x00\x00\x00cpu"
        b"K\x06tQ"
        b"K\x00"
        b"K\x02K\x03\x86"
        b"K\x03K\x01\x86"
        b"\x89"
        b"ccollections\nOrderedDict\n)R"
        b"tRu."
    )


class RestrictedUnpicklerTests(unittest.TestCase):
    def test_reads_plain_tensor_metadata_without_torch(self) -> None:
        state = load_manifest(io.BytesIO(plain_state_dict_pickle()))
        self.assertEqual(list(state), ["weight"])
        self.assertEqual(state["weight"].dtype, "F32")
        self.assertEqual(state["weight"].shape, (2, 3))
        self.assertEqual(state["weight"].stride, (3, 1))
        rendered = render_manifest(state, "fixture@revision:file", "00" * 32)
        self.assertEqual(rendered["tensor_count"], 1)
        self.assertEqual(
            rendered["manifest_sha256"],
            "a8fa6a1bd2c6ac5df4eee80571e43a8ce3b21cb08435bf27b9723daaa5152f8f",
        )
        self.assertEqual(
            rendered["storage_manifest_sha256"],
            "51e579a5e70959adf6da7bc6d2bddcfc0ce008348837a25053427ac00e79c03a",
        )

    def test_rejects_checkpoint_selected_global(self) -> None:
        hostile = b"\x80\x02cos\nsystem\n."
        with self.assertRaisesRegex(ManifestError, r"os\.system.*not allowed"):
            load_manifest(io.BytesIO(hostile))


if __name__ == "__main__":
    unittest.main()
