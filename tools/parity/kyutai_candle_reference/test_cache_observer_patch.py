"""Offline stdlib tests for the Moshi cache/state observer candidate patcher."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


LEAF = Path(__file__).resolve().parent
sys.path.insert(0, str(LEAF))
SPEC = importlib.util.spec_from_file_location("cache_observer_patch", LEAF / "cache_observer_patch.py")
assert SPEC is not None and SPEC.loader is not None
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)
PINNED_SOURCE = LEAF / "pinned-asr.rs"
PINNED_TRANSFORMER = LEAF / "pinned-transformer.rs"
PINNED_LM = LEAF / "pinned-lm.rs"
PINNED_LICENSE = LEAF / "pinned-LICENSE"


@contextlib.contextmanager
def temporary_directory(prefix: str):
    canonical_root = Path(tempfile.gettempdir()).resolve()
    with tempfile.TemporaryDirectory(prefix=prefix, dir=canonical_root) as directory:
        yield Path(directory).resolve()


class CacheObserverPatchTests(unittest.TestCase):
    def test_exact_pins_and_urls(self) -> None:
        self.assertEqual(PATCHER.UPSTREAM_COMMIT, "e6a55d2722a65870ef52a6c9f6ecfc0e90f38362")
        self.assertEqual(PATCHER.SOURCE_BLOB_SHA1, "0f3ad70082f152e3250aa81ab4e0b48ffef550db")
        self.assertEqual(PATCHER.TRANSFORMER_BLOB_SHA1, "4382d71044bd723ebbedade5043ff883683a70b0")
        self.assertEqual(PATCHER.LM_BLOB_SHA1, "6775ec8c5453c40a4f777303bd583db76e089dd8")
        self.assertEqual(PATCHER.LICENSE_BLOB_SHA1, "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64")
        self.assertEqual(PINNED_SOURCE.stat().st_size, 9125)
        self.assertEqual(PINNED_TRANSFORMER.stat().st_size, 40480)
        self.assertEqual(PINNED_LM.stat().st_size, 37671)
        self.assertEqual(PINNED_LICENSE.stat().st_size, 11357)
        expected_moshi_kv_cache_url = (
            "https://raw.githubusercontent.com/kyutai-labs/moshi/"
            "e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/rust/moshi-core/src/kv_cache.rs"
        )
        self.assertEqual(PATCHER.MOSHI_KV_CACHE_URL, expected_moshi_kv_cache_url)
        for url in (PATCHER.SOURCE_URL, PATCHER.TRANSFORMER_URL, PATCHER.LM_URL, PATCHER.LICENSE_URL, PATCHER.MOSHI_KV_CACHE_URL):
            self.assertIn(PATCHER.UPSTREAM_COMMIT, url)

    def test_pinned_fixture_attributes_disable_transforms(self) -> None:
        repo = LEAF.parents[2]
        paths = [
            "tools/parity/kyutai_candle_reference/pinned-asr.rs",
            "tools/parity/kyutai_candle_reference/pinned-LICENSE",
            "tools/parity/kyutai_candle_reference/pinned-transformer.rs",
            "tools/parity/kyutai_candle_reference/pinned-lm.rs",
        ]
        result = subprocess.run(
            ["git", "-C", str(repo), "check-attr", "filter", "diff", "merge", "text", "--", *paths],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for path in paths:
            for attribute in ("filter", "diff", "merge", "text"):
                self.assertIn(f"{path}: {attribute}: unset", result.stdout)

    def test_full_authentication_and_reversal(self) -> None:
        source = PINNED_SOURCE.read_bytes()
        transformer = PINNED_TRANSFORMER.read_bytes()
        lm = PINNED_LM.read_bytes()
        patched = PATCHER.patch_bytes(source, transformer, lm)
        self.assertEqual(PATCHER.revert_asr_bytes(patched[0]), source)
        self.assertEqual(PATCHER.revert_transformer_bytes(patched[1]), transformer)
        self.assertEqual(PATCHER.revert_lm_bytes(patched[2]), lm)
        for value in patched:
            self.assertIn(PATCHER.PATCH_MARKER, value)
        self.assertEqual(patched[0].count(PATCHER.PATCH_MARKER), 1)
        self.assertEqual(patched[1].count(PATCHER.PATCH_MARKER), 1)
        self.assertEqual(patched[2].count(PATCHER.PATCH_MARKER), 1)

    def test_same_size_mutations_rejected_for_all_inputs(self) -> None:
        with temporary_directory("moshi-cache-auth-") as root:
            for source_path, pinned, expected, size in (
                (root / "asr.rs", PINNED_SOURCE, PATCHER.SOURCE_BLOB_SHA1, PATCHER.SOURCE_BYTES),
                (root / "transformer.rs", PINNED_TRANSFORMER, PATCHER.TRANSFORMER_BLOB_SHA1, PATCHER.TRANSFORMER_BYTES),
                (root / "lm.rs", PINNED_LM, PATCHER.LM_BLOB_SHA1, PATCHER.LM_BYTES),
                (root / "LICENSE", PINNED_LICENSE, PATCHER.LICENSE_BLOB_SHA1, PATCHER.LICENSE_BYTES),
            ):
                value = pinned.read_bytes()
                source_path.write_bytes(value[:37] + bytes([value[37] ^ 1]) + value[38:])
                self.assertEqual(source_path.stat().st_size, size)
                with self.assertRaises(PATCHER.PatchError):
                    PATCHER._authenticated(source_path, expected, size, source_path.name)

    def test_missing_and_wrong_anchors_are_rejected(self) -> None:
        with self.assertRaises(PATCHER.PatchError):
            PATCHER.patch_transformer_bytes(PINNED_TRANSFORMER.read_bytes().replace(b"impl StreamingTransformer {\n", b"impl StreamingTransformer {", 1))
        with self.assertRaises(PATCHER.PatchError):
            PATCHER.patch_lm_bytes(PINNED_LM.read_bytes().replace(b"enum StreamingTransformer {\n", b"enum StreamingTransformer {", 1))
        with self.assertRaises(PATCHER.PatchError):
            PATCHER.patch_asr_bytes(PINNED_SOURCE.read_bytes().replace(b"pub struct State {\n", b"pub struct State {", 1))

    def test_duplicate_or_partial_patch_is_rejected(self) -> None:
        source = PINNED_SOURCE.read_bytes()
        transformer = PINNED_TRANSFORMER.read_bytes()
        lm = PINNED_LM.read_bytes()
        patched = PATCHER.patch_bytes(source, transformer, lm)
        with self.assertRaises(PATCHER.PatchError):
            PATCHER.patch_bytes(patched[0], transformer, lm)
        with self.assertRaises(PATCHER.PatchError):
            PATCHER.patch_transformer_bytes(patched[1])
        with self.assertRaises(PATCHER.PatchError):
            PATCHER.patch_lm_bytes(patched[2])
        with self.assertRaises(PATCHER.PatchError):
            PATCHER.revert_transformer_bytes(transformer)

    def test_contract_has_actual_names_order_and_no_default_fallible_work(self) -> None:
        asr = PATCHER.patch_asr_bytes(PINNED_SOURCE.read_bytes()).decode()
        transformer = PATCHER.patch_transformer_bytes(PINNED_TRANSFORMER.read_bytes()).decode()
        lm = PATCHER.patch_lm_bytes(PINNED_LM.read_bytes()).decode()
        self.assertIn("pub struct AsrObservation", asr)
        self.assertIn("global_model_step_idx", asr)
        self.assertIn("pub step_idx: usize", asr)
        self.assertIn("pub text_token: u32", asr)
        self.assertIn("if self.observer_v2.is_some()", asr)
        self.assertLess(asr.index("text_tokens.to_vec1::<u32>()?"), asr.index("cache/state observation seam v2"))
        self.assertLess(asr.index("cache/state observation seam v2"), asr.index("item.text_token = text_token;"))
        self.assertIn("pub upstream_current_seq_len: Vec<usize>", transformer)
        self.assertIn("pub upstream_positions_for_zero_query: Vec<Vec<usize>>", transformer)
        self.assertIn("pub last_reset_pos: Vec<usize>", transformer)
        self.assertIn("pub raw_kv: Vec<RawKvObservation>", transformer)
        self.assertIn("pub k: Option<Tensor>", transformer)
        self.assertIn("pub v: Option<Tensor>", transformer)
        self.assertIn("cache.k()?", transformer)
        self.assertIn("cache.v()?", transformer)
        for accessor in (
            "k_cache().offset()",
            "v_cache().offset()",
            "k_cache().dim()",
            "v_cache().dim()",
            "k_cache().max_seq_len()",
            "v_cache().max_seq_len()",
            "k_cache().current_seq_len()",
            "v_cache().current_seq_len()",
        ):
            self.assertIn(accessor, transformer)
        self.assertIn("KvCache::Rotating(cache)", transformer)
        self.assertIn("positions(0)", transformer)
        self.assertIn("StreamingTransformer::Batched(_) =>", lm)
        self.assertIn("cache observation is unsupported for batched transformer", lm)
        self.assertIn("pub fn cache_observation(&self) -> Result<CacheObservation>", transformer)
        self.assertIn("transformer.cache_observation()?", lm)
        callback = asr[asr.index("// Vokra candidate: cache/state observation seam v2."):]
        self.assertIn("self.lm.cache_observation()?", callback)
        self.assertIn("observer(", callback)

    def test_apply_writes_three_outputs_and_receipt_without_overwrite(self) -> None:
        with temporary_directory("moshi-cache-apply-") as root:
            paths = (root / "asr.rs", root / "transformer.rs", root / "lm.rs", root / "receipt.json")
            receipt = PATCHER.apply(PINNED_SOURCE, PINNED_TRANSFORMER, PINNED_LM, PINNED_LICENSE, *paths)
            self.assertEqual(receipt["schema"], "vokra-moshi-cache-observer-source-only-v2")
            self.assertEqual(json.loads(paths[-1].read_text())["patch"], PATCHER.PATCH_NAME)
            receipt_json = json.loads(paths[-1].read_text())
            candle = receipt_json["candle_api_provenance"]
            expected_candle_accessors = [
                "RotatingKvCache::k",
                "RotatingKvCache::v",
                "RotatingKvCache::k_cache",
                "RotatingKvCache::v_cache",
                "RotatingCache::offset",
                "RotatingCache::dim",
                "RotatingCache::max_seq_len",
                "RotatingCache::current_seq_len",
            ]
            expected_moshi_accessors = ["KvCache::current_seq_len", "KvCache::positions"]
            self.assertEqual(candle["used_accessors"], expected_candle_accessors)
            self.assertEqual(len(candle["used_accessors"]), 8)
            self.assertTrue(set(candle["used_accessors"]).isdisjoint(expected_moshi_accessors))
            self.assertEqual(candle["commit"], "cd96fa80da255e34f7b16b4ff98b6a31d557201b")
            self.assertEqual(candle["path_in_vcs"], "candle-nn")
            self.assertIsNone(candle["license_approval"])
            moshi = receipt_json["moshi_api_provenance"]
            self.assertEqual(moshi["repository"], "kyutai-labs/moshi")
            self.assertEqual(moshi["commit"], PATCHER.UPSTREAM_COMMIT)
            self.assertEqual(moshi["path"], "rust/moshi-core/src/kv_cache.rs")
            self.assertEqual(
                moshi["official_raw_url"],
                "https://raw.githubusercontent.com/kyutai-labs/moshi/"
                "e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/rust/moshi-core/src/kv_cache.rs",
            )
            self.assertEqual(moshi["used_accessors"], expected_moshi_accessors)
            self.assertTrue(set(moshi["used_accessors"]).isdisjoint(candle["used_accessors"]))
            self.assertIsNone(receipt_json["license_approval"])
            self.assertEqual(
                receipt_json["gates"],
                {
                    "remote_build": "NOT_RUN",
                    "runtime": "NOT_RUN",
                    "cpu": "NOT_RUN",
                    "metal": "NOT_RUN",
                    "raw_kv_capture": "NOT_RUN",
                    "independent_pcm_reference": "NOT_RUN",
                    "dependency_license_owner_review": "BLOCKED_PENDING_REVIEW",
                    "model_weights": "NOT_ACQUIRED",
                    "publication": "NO_UPLOAD",
                },
            )
            self.assertIn("current_data", receipt_json["raw_kv_semantics"])
            self.assertIn("physical rotating-buffer order", receipt_json["raw_kv_semantics"]["current_data"])
            self.assertIn("all_data", receipt_json["raw_kv_semantics"])
            self.assertIn("serialize synchronously", receipt_json["raw_kv_semantics"]["collector_rule"])
            for path in paths[:3]:
                self.assertTrue(path.exists())
            with self.assertRaises(PATCHER.PatchError):
                PATCHER.apply(PINNED_SOURCE, PINNED_TRANSFORMER, PINNED_LM, PINNED_LICENSE, *paths)

    def test_distinct_paths_required_and_existing_paths_preserved(self) -> None:
        with temporary_directory("moshi-cache-dest-") as root:
            same = root / "same"
            with self.assertRaises(PATCHER.PatchError):
                PATCHER.apply(PINNED_SOURCE, PINNED_TRANSFORMER, PINNED_LM, PINNED_LICENSE, same, same, root / "lm", root / "receipt")
            existing = root / "asr"
            existing.write_bytes(b"keep")
            with self.assertRaises(PATCHER.PatchError):
                PATCHER.apply(PINNED_SOURCE, PINNED_TRANSFORMER, PINNED_LM, PINNED_LICENSE, existing, root / "tr", root / "lm", root / "receipt")
            self.assertEqual(existing.read_bytes(), b"keep")

    def test_symlinks_and_unsafe_ancestors_rejected(self) -> None:
        with temporary_directory("moshi-cache-links-") as root:
            source_link = root / "asr-link"
            source_link.symlink_to(PINNED_SOURCE)
            with self.assertRaises(PATCHER.PatchError):
                PATCHER._authenticated(source_link, PATCHER.SOURCE_BLOB_SHA1, PATCHER.SOURCE_BYTES, "source")
            linked_parent = root / "linked-parent"
            linked_parent.symlink_to(root, target_is_directory=True)
            with self.assertRaises(PATCHER.PatchError):
                PATCHER.apply(PINNED_SOURCE, PINNED_TRANSFORMER, PINNED_LM, PINNED_LICENSE, linked_parent / "asr", root / "tr", root / "lm", root / "receipt")
            output = root / "output"
            output.symlink_to(root / "target")
            with self.assertRaises(PATCHER.PatchError):
                PATCHER.apply(PINNED_SOURCE, PINNED_TRANSFORMER, PINNED_LM, PINNED_LICENSE, output, root / "tr", root / "lm", root / "receipt")

    def test_partial_write_failure_has_no_receipt_and_preserves_partial_artifacts(self) -> None:
        with temporary_directory("moshi-cache-partial-") as root:
            paths = (root / "asr", root / "transformer", root / "lm", root / "receipt")
            real_write = PATCHER._write_exclusive

            def fail_lm(path: Path, data: bytes, label: str):
                if label == "patched lm output":
                    raise OSError("simulated LM output failure")
                return real_write(path, data, label)

            with mock.patch.object(PATCHER, "_write_exclusive", side_effect=fail_lm):
                with self.assertRaises(OSError):
                    PATCHER.apply(PINNED_SOURCE, PINNED_TRANSFORMER, PINNED_LM, PINNED_LICENSE, *paths)
            self.assertTrue(paths[0].exists())
            self.assertTrue(paths[1].exists())
            self.assertFalse(paths[2].exists())
            self.assertFalse(paths[3].exists())

    def test_partial_receipt_write_leaves_outputs_and_receipt_unaccepted(self) -> None:
        with temporary_directory("moshi-cache-receipt-failure-") as root:
            paths = (root / "asr", root / "transformer", root / "lm", root / "receipt")
            real_write = PATCHER._write_exclusive

            def fail_receipt(path: Path, data: bytes, label: str):
                if label == "receipt":
                    path.write_bytes(data[:17])
                    raise OSError("simulated receipt write failure")
                return real_write(path, data, label)

            with mock.patch.object(PATCHER, "_write_exclusive", side_effect=fail_receipt):
                with self.assertRaises(OSError):
                    PATCHER.apply(PINNED_SOURCE, PINNED_TRANSFORMER, PINNED_LM, PINNED_LICENSE, *paths)
            self.assertTrue(paths[0].exists())
            self.assertTrue(paths[1].exists())
            self.assertTrue(paths[2].exists())
            self.assertTrue(paths[3].exists())
            self.assertEqual(paths[3].stat().st_size, 17)
            with self.assertRaises(json.JSONDecodeError):
                json.loads(paths[3].read_text())

    def test_cli_self_test_is_offline_and_explicit(self) -> None:
        result = subprocess.run(
            ["uv", "run", "--no-project", "--offline", "--python", "3.12", "python", str(LEAF / "cache_observer_patch.py"), "--self-test"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cache observer self-test: PASS", result.stdout)

    def test_source_license_inputs_remain_byte_exact(self) -> None:
        before = {path: path.read_bytes() for path in (PINNED_SOURCE, PINNED_TRANSFORMER, PINNED_LM, PINNED_LICENSE)}
        with temporary_directory("moshi-cache-identity-") as root:
            PATCHER.apply(PINNED_SOURCE, PINNED_TRANSFORMER, PINNED_LM, PINNED_LICENSE, root / "asr", root / "tr", root / "lm", root / "receipt")
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_tampered_fixture_never_reaches_patch_bytes(self) -> None:
        with temporary_directory("moshi-cache-auth-order-") as root:
            source = root / "asr.rs"
            source_bytes = PINNED_SOURCE.read_bytes()
            source.write_bytes(source_bytes[:19] + bytes([source_bytes[19] ^ 1]) + source_bytes[20:])
            patch_mock = mock.patch.object(PATCHER, "patch_bytes", side_effect=AssertionError("patch must not run"))
            with patch_mock as patch_bytes, self.assertRaises(PATCHER.PatchError):
                PATCHER.apply(
                    source,
                    PINNED_TRANSFORMER,
                    PINNED_LM,
                    PINNED_LICENSE,
                    root / "asr.out",
                    root / "transformer.out",
                    root / "lm.out",
                    root / "receipt.json",
                )
            patch_bytes.assert_not_called()


if __name__ == "__main__":
    unittest.main()
