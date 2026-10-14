from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

import dump
from contract import COMPOSITE_APPROVAL_SCHEMA, COMPOSITE_APPROVAL_SCOPE, ExecutionReadinessBlocked
from dump import CONTEXT, RingCapture, _tensor_bytes, _write_output


DELTA_NUMEL = 2048
KV_HEADS = 32
KV_HEAD_WIDTH = 64
DELTA_BYTES = DELTA_NUMEL * 2
SNAPSHOT_BYTES = CONTEXT * DELTA_BYTES


class FakeTensor:
    def __init__(
        self,
        raw: bytes,
        shape: tuple[int, ...] = (1,),
        dtype: str = "torch.bfloat16",
        values=None,
        storage_offset: int = 0,
        element_size: int = 2,
    ):
        self.raw = raw
        self.shape = shape
        self.dtype = dtype
        self.values = [0] if values is None else values
        self._storage_offset = storage_offset
        self._element_size = element_size

    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def clone(self):
        return FakeTensor(
            bytes(self.raw), self.shape, self.dtype, list(self.values),
            self._storage_offset, self._element_size,
        )

    def untyped_storage(self):
        return self.raw

    def numel(self):
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    def element_size(self):
        return self._element_size

    def storage_offset(self):
        return self._storage_offset

    def tolist(self):
        return self.values

    def reshape(self, *_shape):
        return self


class FakeCache:
    def __init__(self):
        self.end_offset = FakeTensor(b"\x00" * 8, shape=(1,), dtype="torch.int64", element_size=8, values=[0])
        self.capacity = 375


class FakeResult:
    def __init__(self, raw: bytes):
        self.keys = FakeTensor(raw, shape=(1, KV_HEADS, CONTEXT, KV_HEAD_WIDTH))
        self.values = FakeTensor(raw, shape=(1, KV_HEADS, CONTEXT, KV_HEAD_WIDTH))
        self.positions = FakeTensor(
            b"\x00" * (CONTEXT * 8),
            shape=(1, CONTEXT),
            dtype="torch.int64",
            element_size=8,
            values=[list(range(CONTEXT))],
        )


class FakeFiniteFlag:
    def __init__(self, finite: bool):
        self.finite = finite

    def all(self):
        return self

    def item(self):
        return self.finite


class FakeLogits:
    def __init__(self, *, finite: bool = True, shape: tuple[int, ...] = (1, 1, 1, 4000)):
        self.shape = shape
        self.dtype = "torch.bfloat16"
        self._finite = finite

    def numel(self):
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    def isfinite(self):
        return FakeFiniteFlag(self._finite)

    def detach(self):
        raise AssertionError("logit metadata rejection must precede export copy")


class FakeTransformer:
    class RingKVCache:
        @staticmethod
        def complete(cache, keys, values, exec_mask):
            return FakeResult(b"\x01" * SNAPSHOT_BYTES)


class FakeLM:
    class Transformer:
        def __init__(self, caches):
            self.layers = [type("Layer", (), {"self_attn": type("Attention", (), {"_streaming_state": type("State", (), {"kv_cache": cache})()})()})() for cache in caches]

    def __init__(self, caches):
        self.transformer = self.Transformer(caches)


class ModuleList(list):
    pass


class TrustedTransformer(FakeTransformer):
    class nn:
        ModuleList = ModuleList


def delta_tensor(raw: bytes | None = None, *, dtype: str = "torch.bfloat16") -> FakeTensor:
    return FakeTensor(
        b"\x10" * DELTA_BYTES if raw is None else raw,
        shape=(1, KV_HEADS, 1, KV_HEAD_WIDTH),
        dtype=dtype,
    )


class CountingTransformer(FakeTransformer):
    calls = 0

    class RingKVCache:
        @staticmethod
        def complete(cache, keys, values, exec_mask):
            CountingTransformer.calls += 1
            return FakeResult(b"\x01" * SNAPSHOT_BYTES)


class MutatingTransformer(FakeTransformer):
    class RingKVCache:
        @staticmethod
        def complete(cache, keys, values, exec_mask):
            keys.raw = b"\xff" * DELTA_BYTES
            values.raw = b"\xee" * DELTA_BYTES
            return FakeResult(b"\x01" * SNAPSHOT_BYTES)


class UnknownLayerContainer:
    def __iter__(self):
        return iter(())


class DumpTests(unittest.TestCase):
    def _real_args(self, root: Path, approval: Path, approval_sha256: str) -> argparse.Namespace:
        return argparse.Namespace(
            out=root / "uncreated-reference",
            expected_head="0" * 40,
            approval_evidence=approval,
            approval_sha256=approval_sha256,
            dependency_closure_sha256="1" * 64,
            model=root / "model",
            dsm_source=root / "dsm",
            moshi_source=root / "moshi",
            source_packet=None,
        )

    def test_real_entry_blocks_darwin_before_model_authentication(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kyutai-readiness-darwin-") as raw:
            root = Path(raw)
            args = self._real_args(root, root / "fake-approval.json", "0" * 64)
            with patch.object(dump.platform, "system", return_value="Darwin"), \
                    patch.object(dump.platform, "machine", return_value="arm64"), \
                    patch.object(dump, "authenticate_composite_model", side_effect=AssertionError("model auth reached")), \
                    patch.object(dump, "_decoder_helpers", side_effect=AssertionError("decoder import reached")):
                with self.assertRaises(ExecutionReadinessBlocked):
                    dump.real(args)

    def test_real_entry_blocks_unreviewed_closure_before_model_authentication(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kyutai-readiness-closure-") as raw:
            root = Path(raw)
            document = {
                "schema": COMPOSITE_APPROVAL_SCHEMA,
                "scope": COMPOSITE_APPROVAL_SCOPE,
                "decision": "APPROVED",
                "execution": "VAST_ONLY",
                "head": "0" * 40,
                "checkout": str(Path.cwd()),
            }
            body = (json.dumps(document, sort_keys=True) + "\n").encode()
            approval = root / "approval.json"
            approval.write_bytes(body)
            digest = hashlib.sha256(body).hexdigest()
            args = self._real_args(root, approval, digest)
            with patch.object(dump.platform, "system", return_value="Linux"), \
                    patch.object(dump.platform, "machine", return_value="x86_64"), \
                    patch.object(dump, "authenticate_composite_model", side_effect=AssertionError("model auth reached")), \
                    patch.object(dump, "_decoder_helpers", side_effect=AssertionError("decoder import reached")):
                with self.assertRaises(ExecutionReadinessBlocked):
                    dump.real(args)

    def test_real_entry_requires_source_packet_after_readiness_gate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kyutai-readiness-source-") as raw:
            root = Path(raw)
            args = self._real_args(root, root / "unused-approval.json", "0" * 64)
            fake_readiness = {
                "approval": {"decision": "APPROVED", "scope": "synthetic-test-only"},
                "dependency_closure_sha256": "1" * 64,
            }
            with patch.object(dump, "require_execution_readiness", return_value=fake_readiness), \
                    patch.object(dump, "_decoder_helpers", side_effect=AssertionError("decoder import reached")), \
                    patch.object(dump, "authenticate_composite_model", side_effect=AssertionError("model auth reached")):
                with self.assertRaisesRegex(ValueError, "source packet"):
                    dump.real(args)

    def test_logits_metadata_is_bounded_and_finite_before_export(self) -> None:
        from dump import _validate_logits_for_export

        source_shape, comparison_shape, source_dtype, export_bytes = _validate_logits_for_export(
            FakeLogits(), reserved_bytes=0
        )
        self.assertEqual(source_shape, [1, 1, 1, 4000])
        self.assertEqual(comparison_shape, [1, 4000])
        self.assertEqual(source_dtype, "torch.bfloat16")
        self.assertEqual(export_bytes, 4000 * 4)
        with self.assertRaises(ValueError):
            _validate_logits_for_export(FakeLogits(finite=False), reserved_bytes=0)
        with self.assertRaises(ValueError):
            _validate_logits_for_export(FakeLogits(shape=(1, 4000)), reserved_bytes=0)
        with self.assertRaises(ValueError):
            _validate_logits_for_export(FakeLogits(shape=(1, 1, 1, 3999)), reserved_bytes=0)

    def test_raw_bfloat16_storage_does_not_use_numpy_conversion(self) -> None:
        tensor = FakeTensor(b"\x00\x3c\x80\x3c", shape=(2,), dtype="torch.bfloat16")
        self.assertEqual(_tensor_bytes(tensor), b"\x00\x3c\x80\x3c")

    def test_tensor_bytes_slices_logical_window_not_backing_storage(self) -> None:
        tensor = FakeTensor(
            b"PREFIX\x00\x3c\x80\x3cSUFFIX",
            shape=(2,),
            storage_offset=3,
            element_size=2,
        )
        self.assertEqual(_tensor_bytes(tensor), b"\x00\x3c\x80\x3c")

    def test_tensor_bytes_rejects_short_logical_window(self) -> None:
        tensor = FakeTensor(b"\x00\x3c", shape=(2,), element_size=2)
        with self.assertRaises(ValueError):
            _tensor_bytes(tensor)

    def test_tensor_bytes_caps_before_detach_or_cpu_copy(self) -> None:
        class HugeTensor(FakeTensor):
            def detach(self):
                raise AssertionError("detach must not run before logical cap")

        tensor = HugeTensor(b"", shape=(300_000_000,), element_size=2)
        with self.assertRaises(ValueError):
            _tensor_bytes(tensor)

    def test_tensor_bytes_slices_huge_backing_without_full_bytes_copy(self) -> None:
        class BoundedStorage:
            def __init__(self, body: bytes, total_bytes: int):
                self.body = body
                self.total_bytes = total_bytes
                self.slices = []

            def nbytes(self):
                return self.total_bytes

            def __getitem__(self, key):
                if not isinstance(key, slice):
                    raise TypeError("slice required")
                self.slices.append((key.start, key.stop))
                return self.body[key]

            def __bytes__(self):
                raise AssertionError("full backing storage must not be copied")

        storage = BoundedStorage(b"\x00\x3c\x80\x3c", 256 * 1024 * 1024)
        tensor = FakeTensor(storage, shape=(2,), element_size=2)
        self.assertEqual(_tensor_bytes(tensor), b"\x00\x3c\x80\x3c")
        self.assertEqual(storage.slices, [(0, 4)])

    def test_tensor_bytes_rejects_deprecated_typed_storage_fallback(self) -> None:
        class LegacyOnly(FakeTensor):
            def untyped_storage(self):
                return None

            def storage(self):
                raise AssertionError("typed storage fallback must not run")

        with self.assertRaises(TypeError):
            _tensor_bytes(LegacyOnly(b"\x00\x3c", shape=(1,), element_size=2))

    def test_capture_registers_exact_layers_and_clones_aliasing_result(self) -> None:
        transformer = FakeTransformer()
        caches = [FakeCache() for _ in range(48)]
        lm = FakeLM(caches)
        capture = RingCapture(frozenset({0}))
        capture.install(transformer, lm, trusted_layer_container_type=list)
        capture.record_state_snapshot("initial", 0)
        cache = caches[7]
        result = transformer.RingKVCache.complete(cache, delta_tensor(), delta_tensor(), None)
        self.assertEqual(capture.events[0]["layer"], 7)
        snapshot = bytes(capture.files["kv/warmup-step-0000-layer-07-keys.bin"])
        self.assertEqual(snapshot[:4], b"\x01\x01\x01\x01")
        self.assertEqual(len(snapshot), SNAPSHOT_BYTES)
        result.keys.raw = b"changed"
        self.assertEqual(bytes(capture.files["kv/warmup-step-0000-layer-07-keys.bin"]), snapshot)
        with self.assertRaises(RuntimeError):
            capture.require_complete_coverage()
        with self.assertRaises(RuntimeError):
            transformer.RingKVCache.complete(FakeCache(), delta_tensor(), delta_tensor(), None)
        self.assertEqual(len(capture.state_snapshots), 48)
        capture.uninstall(transformer)

    def test_delta_is_frozen_before_official_call_can_mutate_inputs(self) -> None:
        transformer = MutatingTransformer()
        caches = [FakeCache() for _ in range(48)]
        capture = RingCapture(frozenset())
        capture.install(transformer, FakeLM(caches), trusted_layer_container_type=list)
        transformer.RingKVCache.complete(caches[0], delta_tensor(b"\x11" * DELTA_BYTES), delta_tensor(b"\x22" * DELTA_BYTES), None)
        self.assertEqual(bytes(capture.files["kv/warmup-layer-00-delta-keys.bin"]), b"\x11" * DELTA_BYTES)
        self.assertEqual(bytes(capture.files["kv/warmup-layer-00-delta-values.bin"]), b"\x22" * DELTA_BYTES)
        capture.uninstall(transformer)

    def test_unknown_cache_is_rejected_before_original(self) -> None:
        CountingTransformer.calls = 0
        transformer = CountingTransformer()
        caches = [FakeCache() for _ in range(48)]
        capture = RingCapture(frozenset())
        capture.install(transformer, FakeLM(caches), trusted_layer_container_type=list)
        with self.assertRaises(RuntimeError):
            transformer.RingKVCache.complete(FakeCache(), delta_tensor(), delta_tensor(), None)
        self.assertEqual(CountingTransformer.calls, 0)
        capture.uninstall(transformer)

    def test_duplicate_event_is_rejected_before_original(self) -> None:
        CountingTransformer.calls = 0
        transformer = CountingTransformer()
        caches = [FakeCache() for _ in range(48)]
        capture = RingCapture(frozenset())
        capture.install(transformer, FakeLM(caches), trusted_layer_container_type=list)
        args = (caches[0], delta_tensor(), delta_tensor(), None)
        transformer.RingKVCache.complete(*args)
        with self.assertRaises(RuntimeError):
            transformer.RingKVCache.complete(*args)
        self.assertEqual(CountingTransformer.calls, 1)
        capture.uninstall(transformer)

    def test_budget_and_invalid_geometry_fail_before_copy_or_original(self) -> None:
        CountingTransformer.calls = 0
        transformer = CountingTransformer()
        caches = [FakeCache() for _ in range(48)]
        class CopyTrap(FakeTensor):
            def detach(self):
                raise AssertionError("budget must fail before bounded tensor copy")

        capture = RingCapture(frozenset(), max_artifact_bytes=1)
        capture.install(transformer, FakeLM(caches), trusted_layer_container_type=list)
        with self.assertRaises(RuntimeError):
            transformer.RingKVCache.complete(
                caches[0],
                CopyTrap(b"\x00" * DELTA_BYTES, shape=(1, KV_HEADS, 1, KV_HEAD_WIDTH)),
                CopyTrap(b"\x00" * DELTA_BYTES, shape=(1, KV_HEADS, 1, KV_HEAD_WIDTH)),
                None,
            )
        self.assertEqual(CountingTransformer.calls, 0)
        capture.uninstall(transformer)

        class BadTensor(FakeTensor):
            def detach(self):
                raise AssertionError("invalid geometry must fail before tensor copy")

        transformer = CountingTransformer()
        capture = RingCapture(frozenset())
        capture.install(transformer, FakeLM(caches), trusted_layer_container_type=list)
        bad = BadTensor(b"\x00\x00", shape=(1,), dtype="torch.float32")
        with self.assertRaises(ValueError):
            transformer.RingKVCache.complete(caches[0], bad, delta_tensor(), None)
        self.assertEqual(CountingTransformer.calls, 0)
        capture.uninstall(transformer)

        transformer = CountingTransformer()
        capture = RingCapture(frozenset())
        capture.install(transformer, FakeLM(caches), trusted_layer_container_type=list)
        bad_shape = BadTensor(b"\x00\x00", shape=(1,), dtype="torch.bfloat16")
        with self.assertRaises(ValueError):
            transformer.RingKVCache.complete(caches[0], bad_shape, delta_tensor(), None)
        self.assertEqual(CountingTransformer.calls, 0)
        capture.uninstall(transformer)

    def test_capture_accepts_official_module_list_container(self) -> None:
        caches = [FakeCache() for _ in range(48)]
        lm = FakeLM(caches)
        lm.transformer.layers = ModuleList(lm.transformer.layers)
        capture = RingCapture(frozenset({0}))
        transformer = TrustedTransformer()
        capture.install(transformer, lm)
        self.assertEqual(len(capture.layer_caches), 48)
        capture.uninstall(transformer)

    def test_capture_rejects_unknown_layer_container(self) -> None:
        caches = [FakeCache() for _ in range(48)]
        lm = FakeLM(caches)
        lm.transformer.layers = UnknownLayerContainer()
        with self.assertRaises(RuntimeError):
            RingCapture(frozenset({0})).register_layers(lm, ModuleList)

    def test_official_modulelist_must_come_from_transformer_nn(self) -> None:
        class DirectOnly:
            ModuleList = ModuleList

        with self.assertRaises(RuntimeError):
            from dump import _trusted_layer_container_type
            _trusted_layer_container_type(DirectOnly())

    def test_class_name_spoof_is_not_a_trusted_module_list(self) -> None:
        class Spoof(list):
            pass

        Spoof.__name__ = "ModuleList"
        caches = [FakeCache() for _ in range(48)]
        lm = FakeLM(caches)
        lm.transformer.layers = Spoof(lm.transformer.layers)
        with self.assertRaises(RuntimeError):
            RingCapture(frozenset({0})).register_layers(lm, ModuleList)

    def test_output_is_exclusive_and_does_not_stomp_existing_path(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kyutai-output-") as raw:
            parent = Path(raw)
            out = parent / "reference"
            _write_output(out, {"input.json": b"{}"}, {"status": "synthetic"})
            self.assertTrue((out / "manifest.json").is_file())
            with self.assertRaises(ValueError):
                _write_output(out, {"input.json": b"changed"}, {"status": "synthetic"})
            link = parent / "link"
            link.symlink_to(out, target_is_directory=True)
            with self.assertRaises(ValueError):
                _write_output(link, {"input.json": b"changed"}, {"status": "synthetic"})

    def test_output_rejects_reserved_and_prefix_colliding_names(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kyutai-output-names-") as raw:
            parent = Path(raw)
            with self.assertRaises(ValueError):
                _write_output(parent / "reserved", {"manifest.json": b"bad"}, {})
            with self.assertRaises(ValueError):
                _write_output(parent / "collision", {"a": b"file", "a/b": b"nested"}, {})

    def test_writer_failure_leaves_exclusive_partial_tree_without_recursive_cleanup(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kyutai-output-failure-") as raw:
            out = Path(raw) / "partial"
            with patch.object(dump.os, "write", side_effect=OSError("synthetic write failure")):
                with self.assertRaises(OSError):
                    _write_output(out, {"payload.bin": b"payload"}, {})
            self.assertTrue(out.is_dir())
            (out / "foreign-marker").write_bytes(b"preserve")
            with self.assertRaises(ValueError):
                _write_output(out, {"payload.bin": b"replacement"}, {})
            self.assertEqual((out / "foreign-marker").read_bytes(), b"preserve")


if __name__ == "__main__":
    unittest.main()
