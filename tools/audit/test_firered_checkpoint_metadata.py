"""Offline safety tests for the inert FireRed metadata reader."""

from __future__ import annotations

import argparse
import collections
import contextlib
import importlib.util
import io
import os
import pickle
import pickletools
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


LEAF = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("firered_checkpoint_metadata", LEAF / "firered_checkpoint_metadata.py")
assert SPEC is not None and SPEC.loader is not None
READER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = READER
SPEC.loader.exec_module(READER)


MANDATORY = {
    "idim": 80,
    "n_layers_enc": 6,
    "n_head": 4,
    "d_model": 256,
    "residual_dropout": 0.1,
    "dropout_rate": 0.1,
    "kernel_size": 3,
    "pe_maxlen": 5000,
    "sos_id": 1,
    "eos_id": 2,
    "pad_id": 0,
    "odim": 80,
    "n_layers_dec": 6,
}


def pickle_part(value: object) -> bytes:
    encoded = pickle.dumps(value, protocol=2)
    assert encoded[:2] == b"\x80\x02" and encoded[-1:] == b"."
    return encoded[2:-1]


def tensor_part(
    *,
    storage_key: str = "storage-key",
    numel: int = 8,
    shape: tuple[int, ...] = (2, 4),
    stride: tuple[int, ...] = (4, 1),
    mutate_descriptor: bool = False,
    mutate_storage: bool = False,
    mutate_storage_class: bool = False,
    ordered_hooks: bool = False,
    ordered_metadata: bool = False,
) -> bytes:
    result = bytearray(b"ctorch._utils\n_rebuild_tensor_v2\n(")
    result.extend(b"(")
    result.extend(pickle_part("storage"))
    result.extend(b"ctorch\nFloatStorage\n")
    if mutate_storage_class:
        result.extend(b"}")
        result.extend(pickle_part("dtype"))
        result.extend(pickle_part("F64"))
        result.extend(b"sb")
    result.extend(pickle_part(storage_key))
    result.extend(pickle_part("cpu"))
    result.extend(pickle_part(numel))
    result.extend(b"tQ")
    if mutate_storage:
        result.extend(b"}")
        result.extend(pickle_part("numel"))
        result.extend(pickle_part(1))
        result.extend(b"sb")
    result.extend(pickle_part(0))
    result.extend(b"(")
    for value in shape:
        result.extend(pickle_part(value))
    result.extend(b"t(")
    for value in stride:
        result.extend(pickle_part(value))
    result.extend(b"t\x89")
    result.extend(pickle_part(collections.OrderedDict()) if ordered_hooks else b"N")
    result.extend(pickle_part(collections.OrderedDict()) if ordered_metadata else b"N")
    result.extend(b"tR")
    if mutate_descriptor:
        result.extend(b"}")
        result.extend(pickle_part("shape"))
        result.extend(b"]sb")
    return bytes(result)


def envelope(
    fields: dict[str, object] | None = None,
    *,
    tensor: bytes | None = None,
    tensor_count: int = 1,
    model_cycle: bool = False,
    ordered_state: bool = False,
    ordered_metadata: str | None = None,
    omit: tuple[str, ...] = (),
) -> bytes:
    values = dict(MANDATORY)
    for field in omit:
        values.pop(field, None)
    values.update(fields or {})
    namespace = argparse.Namespace(**values)
    args_part = pickle_part(namespace)
    result = bytearray(b"\x80\x02}")
    result.extend(pickle_part("args"))
    result.extend(args_part)
    result.extend(b"s")
    result.extend(pickle_part("model_state_dict"))
    result.extend(pickle_part(collections.OrderedDict()) if ordered_state else b"}")
    if model_cycle:
        result.extend(pickle_part("self"))
        result.extend(b"q\x00h\x00s")
    else:
        if tensor_count < 1:
            raise ValueError("tensor_count must be positive")
        for index in range(tensor_count):
            result.extend(pickle_part("layer.weight" if index == 0 else f"layer.{index}"))
            result.extend(
                tensor
                if tensor is not None and index == 0
                else tensor_part(storage_key=f"storage-key-{index}")
            )
            result.extend(b"s")
        if ordered_metadata is not None:
            metadata_value = collections.OrderedDict()
            attr_key = "_metadata" if ordered_metadata == "valid" else ordered_metadata
            result.extend(pickle_part({attr_key: metadata_value}))
            result.extend(b"b")
    result.extend(b"s.")
    return bytes(result)


class SyntheticStorage:
    def __init__(self, key: str, numel: int) -> None:
        self.key = key
        self.numel = numel


class SyntheticTensor:
    def __init__(self, rebuild: object, index: int) -> None:
        self.rebuild = rebuild
        self.index = index

    def __reduce_ex__(self, protocol: int) -> tuple[object, tuple[object, ...]]:
        del protocol
        storage = SyntheticStorage(f"storage-key-{self.index}", 8)
        return (
            self.rebuild,
            (storage, 0, (2, 4), (4, 1), False, collections.OrderedDict(), collections.OrderedDict()),
        )


@contextlib.contextmanager
def fake_torch_globals():
    saved = {name: sys.modules.get(name) for name in ("torch", "torch._utils")}
    fake_torch = types.ModuleType("torch")
    fake_utils = types.ModuleType("torch._utils")

    class FakeFloatStorage:
        pass

    FakeFloatStorage.__module__ = "torch"
    FakeFloatStorage.__name__ = "FloatStorage"
    FakeFloatStorage.__qualname__ = "FloatStorage"

    def rebuild_tensor_v2(*args: object) -> None:
        del args

    rebuild_tensor_v2.__module__ = "torch._utils"
    rebuild_tensor_v2.__name__ = "_rebuild_tensor_v2"
    rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"
    fake_torch.FloatStorage = FakeFloatStorage
    fake_torch._utils = fake_utils
    fake_utils._rebuild_tensor_v2 = rebuild_tensor_v2
    sys.modules["torch"] = fake_torch
    sys.modules["torch._utils"] = fake_utils
    try:
        yield rebuild_tensor_v2
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def realistic_envelope(tensor_count: int = 940) -> bytes:
    with fake_torch_globals() as rebuild:
        state = collections.OrderedDict()
        state._metadata = collections.OrderedDict()
        for index in range(tensor_count):
            name = "layer.weight" if index == 0 else f"layer.{index}"
            state[name] = SyntheticTensor(rebuild, index)
        root = {"args": argparse.Namespace(**MANDATORY), "model_state_dict": state}

        class PersistentPickler(pickle.Pickler):
            def persistent_id(self, value: object) -> object:
                if isinstance(value, SyntheticStorage):
                    return ("storage", fake_storage_type, value.key, "cpu", value.numel)
                return None

        fake_storage_type = sys.modules["torch"].FloatStorage
        stream = io.BytesIO()
        PersistentPickler(stream, protocol=2).dump(root)
        return stream.getvalue()


@contextlib.contextmanager
def temp_root():
    with tempfile.TemporaryDirectory(prefix="firered-meta-") as directory:
        yield Path(directory).resolve()


class Attack:
    def __init__(self, path: str) -> None:
        self.path = path

    def __reduce__(self):
        return (os.system, (f"touch {self.path}",))


class FireRedMetadataTests(unittest.TestCase):
    def test_valid_capture_records_observed_args_and_tensor_descriptor(self) -> None:
        data = envelope({"token_list": [1, 2], "note": "observed"})
        metadata = READER.capture_bytes(data)
        self.assertEqual(metadata["status"], "METADATA_CAPTURED_REVIEW_PENDING")
        self.assertEqual(metadata["args"]["idim"], MANDATORY["idim"])
        self.assertEqual(metadata["args"]["token_list"], [1, 2])
        self.assertEqual(metadata["tensor_count"], 1)
        self.assertEqual(metadata["model_state_dict"]["layer.weight"]["dtype"], "F32")
        self.assertEqual(metadata["model"]["checkpoint_sha256"], None)
        self.assertEqual(metadata["publication"], "NO_UPLOAD")
        self.assertEqual(metadata["approval"], None)

    def test_standard_ordered_dict_surrogates_are_supported_only_in_expected_slots(self) -> None:
        data = envelope(
            ordered_state=True,
            ordered_metadata="valid",
            tensor=tensor_part(ordered_hooks=True, ordered_metadata=True),
        )
        metadata = READER.capture_bytes(data)
        self.assertEqual(metadata["tensor_count"], 1)
        self.assertEqual(metadata["model_state_dict_metadata"], {})
        with self.assertRaises(READER.MetadataError):
            READER.capture_bytes(envelope(ordered_state=True, ordered_metadata="unexpected"))
        forged_hooks = READER.InertOrderedDict()
        forged_hooks._metadata = {}
        with self.assertRaises(READER.MetadataError):
            READER.rebuild_tensor_v2(
                READER.StorageRef("F32", 4, "storage-key", "cpu", 8),
                0,
                (2, 4),
                (4, 1),
                False,
                forged_hooks,
                None,
            )

    def test_envelope_ordered_dict_attributes_are_rejected(self) -> None:
        plain = envelope(ordered_state=True)
        ordered_root = b"\x80\x02" + pickle_part(collections.OrderedDict()) + plain[3:]
        marker = pickle_part({"_metadata": {"unexpected": "root attribute"}})
        # Apply a forbidden attribute to the ordered envelope itself.
        forged = ordered_root[:-1] + marker + b"b."
        with self.assertRaises(READER.MetadataError):
            READER.capture_bytes(forged)

    def test_no_torch_import_and_revision_labels_are_not_authenticated(self) -> None:
        self.assertNotIn("torch", __import__("sys").modules)
        metadata = READER.capture_bytes(envelope())
        self.assertEqual(metadata["source"]["declared_revision"], READER.OFFICIAL_SOURCE_REVISION)
        self.assertEqual(metadata["source"]["identity_status"], "DECLARED_NOT_AUTHENTICATED")
        self.assertEqual(metadata["model"]["declared_revision"], READER.DECLARED_MODEL_REVISION)
        self.assertEqual(metadata["model"]["identity_status"], "DECLARED_NOT_AUTHENTICATED")
        self.assertEqual(metadata["gates"]["real_checkpoint"], "NOT_RUN")

    def test_missing_mandatory_dropout_and_position_fields_refuse(self) -> None:
        for field in ("dropout_rate", "residual_dropout", "pe_maxlen"):
            with self.subTest(field=field):
                fields = dict(MANDATORY)
                fields.pop(field)
                with self.assertRaises(READER.MetadataError):
                    READER.capture_bytes(envelope(omit=(field,)))

    def test_bool_as_integer_and_nonfinite_values_refuse(self) -> None:
        with self.assertRaises(READER.MetadataError):
            READER.capture_bytes(envelope({"idim": True}))
        with self.assertRaises(READER.MetadataError):
            READER.capture_bytes(envelope({"dropout_rate": float("nan")}))

    def test_credential_bearing_namespace_fields_refuse_without_echo(self) -> None:
        for field in ("api_key", "hf_token", "password", "secret", "scw_secret_key"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(READER.MetadataError, "credential-bearing"):
                    READER.capture_bytes(envelope({field: "not-printed"}))
        with self.assertRaisesRegex(READER.MetadataError, "credential-bearing"):
            READER.capture_bytes(envelope({"nested": {"scw_secret_key": "not-printed"}}))

    def test_global_reduce_attack_is_rejected_without_effect(self) -> None:
        with temp_root() as root:
            marker = root / "attack-marker"
            data = pickle.dumps(Attack(str(marker)), protocol=2)
            with self.assertRaises(READER.MetadataError):
                READER.capture_bytes(data)
            self.assertFalse(marker.exists())

    def test_sparse_memo_and_trailing_bytes_refuse_before_unpickle(self) -> None:
        sparse_memo = b"\x80\x02K\x00r" + struct.pack("<I", READER.MAX_MEMO_INDEX + 1) + b"."
        with mock.patch.object(READER, "RestrictedUnpickler", side_effect=AssertionError("must not unpickle")) as unpickler:
            with self.assertRaisesRegex(READER.MetadataError, "memo index"):
                READER.capture_bytes(sparse_memo)
            unpickler.assert_not_called()
        with mock.patch.object(READER, "RestrictedUnpickler", side_effect=AssertionError("must not unpickle")) as unpickler:
            with self.assertRaisesRegex(READER.MetadataError, "trailing bytes"):
                READER.capture_bytes(envelope() + b"trailing")
            unpickler.assert_not_called()

    def test_args_and_tensor_cycles_refuse(self) -> None:
        namespace = argparse.Namespace(**MANDATORY)
        namespace.self = namespace
        cyclic_args = pickle.dumps({"args": namespace, "model_state_dict": {}}, protocol=2)
        with self.assertRaises(READER.MetadataError):
            READER.capture_bytes(cyclic_args)
        with self.assertRaises(READER.MetadataError):
            READER.capture_bytes(envelope(model_cycle=True))

    def test_tensor_shape_stride_and_storage_overflow_refuse(self) -> None:
        with self.assertRaises(READER.MetadataError):
            READER.capture_bytes(envelope(tensor=tensor_part(numel=1, shape=(2,), stride=(1,))))
        with self.assertRaises(READER.MetadataError):
            READER.capture_bytes(
                envelope(
                    tensor=tensor_part(
                        numel=1,
                        shape=(2,),
                        stride=(READER.MAX_DIMENSION,),
                    )
                )
            )

    def test_pickle_build_cannot_mutate_inert_descriptors(self) -> None:
        for kwargs in (
            {"mutate_descriptor": True},
            {"mutate_storage": True},
            {"mutate_storage_class": True},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(READER.MetadataError):
                    READER.capture_bytes(envelope(tensor=tensor_part(**kwargs)))

    def test_bounded_reader_rejects_oversize_before_payload_read(self) -> None:
        with temp_root() as root:
            source = root / "data.pkl"
            source.write_bytes(b"x" * (READER.MAX_INPUT_BYTES + 1))
            with mock.patch.object(READER.os, "read", side_effect=AssertionError("payload read")) as read:
                with self.assertRaises(READER.MetadataError):
                    READER._read_bounded(source)
            read.assert_not_called()

    def test_bounded_reader_rejects_growth_and_short_read(self) -> None:
        with temp_root() as root:
            source = root / "data.pkl"
            source.write_bytes(b"abc")
            with mock.patch.object(READER.os, "read", return_value=b"abcd"):
                with self.assertRaises(READER.MetadataError):
                    READER._read_bounded(source)
            with mock.patch.object(READER.os, "read", side_effect=(b"ab", b"")):
                with self.assertRaises(READER.MetadataError):
                    READER._read_bounded(source)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO test requires POSIX")
    def test_fifo_is_rejected_before_open_read_can_block(self) -> None:
        with temp_root() as root:
            source = root / "data.pkl"
            os.mkfifo(source)
            with self.assertRaisesRegex(READER.MetadataError, "regular file"):
                READER._read_bounded(source)

    def test_parent_walk_closes_inner_fd_when_fstat_fails(self) -> None:
        with temp_root() as root:
            parent = root / "nested"
            parent.mkdir()
            target = parent / "metadata.json"
            real_fstat = READER.os.fstat
            fstat_calls = 0

            def fail_inner(fd: int) -> os.stat_result:
                nonlocal fstat_calls
                fstat_calls += 1
                if fstat_calls == 2:
                    raise OSError("synthetic inner fstat failure")
                return real_fstat(fd)

            real_close = READER.os.close
            with mock.patch.object(READER.os, "fstat", side_effect=fail_inner):
                with mock.patch.object(READER.os, "close", wraps=real_close) as close:
                    with self.assertRaises(READER.MetadataError):
                        READER._open_parent_dir(target, "output parent")
            self.assertGreaterEqual(close.call_count, 2)

    def test_shared_dag_output_budget_is_global(self) -> None:
        shared: list[object] = []
        for _ in range(14):
            shared = [shared, shared]
        with self.assertRaisesRegex(READER.MetadataError, "security budget"):
            READER.capture_bytes(envelope({"shared": shared}))

    def test_repeated_large_strings_hit_output_budget_before_rendering(self) -> None:
        with mock.patch.object(READER, "MAX_OUTPUT_BYTES", 16_384):
            control = READER.capture_bytes(envelope({"control": "x" * 512}))
            self.assertEqual(control["args"]["control"], "x" * 512)
            repeated = "x" * 512
            with self.assertRaisesRegex(READER.MetadataError, "output byte budget"):
                READER.capture_bytes(envelope({"repeated": [repeated] * 16}))

    def test_main_checks_exact_serialized_size_before_exclusive_write(self) -> None:
        with temp_root() as root:
            input_path = root / "data.pkl"
            output = root / "metadata.json"
            input_path.write_bytes(b"placeholder")
            oversized_metadata = {"observed": "x" * 256}
            with mock.patch.object(READER.sys, "platform", "linux"):
                with mock.patch.dict(READER.os.environ, {READER.REMOTE_MARKER: "1"}):
                    with mock.patch.object(READER, "_read_bounded", return_value=b"placeholder"):
                        with mock.patch.object(READER, "capture_bytes", return_value=oversized_metadata):
                            with mock.patch.object(READER, "MAX_OUTPUT_BYTES", 64):
                                with mock.patch.object(READER, "_write_exclusive") as write:
                                    self.assertEqual(READER.main([str(input_path), str(output)]), 2)
                                    write.assert_not_called()

    def test_known_scale_synthetic_fixture_has_940_distinct_descriptors(self) -> None:
        data = realistic_envelope(940)
        self.assertEqual(len(data), 93_568)
        self.assertLess(len(data), READER.MAX_INPUT_BYTES)
        operations = list(pickletools.genops(data))
        opcode_count = len(operations)
        memo_writes = [
            argument
            for operation, argument, _position in operations
            if operation.name in {"BINPUT", "LONG_BINPUT", "PUT", "MEMOIZE"}
        ]
        memo_indices = [
            argument
            for operation, argument, _position in operations
            if operation.name in {"BINPUT", "LONG_BINPUT", "BINGET", "LONG_BINGET", "PUT", "GET"}
        ]
        self.assertEqual(opcode_count, 28_286)
        self.assertEqual(len(memo_writes), 6_610)
        self.assertEqual(max(memo_indices), 6_609)
        self.assertLess(opcode_count, READER.MAX_OPCODE_COUNT)
        self.assertLessEqual(len(memo_writes), READER.MAX_MEMO_ENTRIES)
        self.assertLessEqual(max(memo_indices), READER.MAX_MEMO_INDEX)
        self.assertNotIn("torch", sys.modules)
        metadata = READER.capture_bytes(data)
        self.assertEqual(metadata["tensor_count"], 940)
        self.assertEqual(metadata["model_state_dict"]["layer.939"]["storage_key"], "storage-key-939")

    def test_output_no_overwrite_and_racing_creator_preserved(self) -> None:
        with temp_root() as root:
            output = root / "metadata.json"
            output.write_bytes(b"foreign")
            with self.assertRaises(READER.MetadataError):
                READER._write_exclusive(output, b"replacement")
            self.assertEqual(output.read_bytes(), b"foreign")

            parent = root / "output-parent"
            parent.mkdir()
            race = parent / "race.json"
            original_parent = root / "output-parent-original"
            real_os_open = READER.os.open

            def pin_then_replace(path: object, flags: int, *args: object, **kwargs: object) -> int:
                fd = real_os_open(path, flags, *args, **kwargs)
                if path == parent.name and kwargs.get("dir_fd") is not None:
                    os.rename(parent, original_parent)
                    parent.mkdir()
                    (parent / "race.json").write_bytes(b"race-winner")
                return fd

            with mock.patch.object(READER.os, "open", side_effect=pin_then_replace):
                READER._write_exclusive(race, b"candidate")
            self.assertEqual((original_parent / "race.json").read_bytes(), b"candidate")
            self.assertEqual(race.read_bytes(), b"race-winner")

    def test_input_parent_substitution_cannot_redirect_pinned_read(self) -> None:
        with temp_root() as root:
            parent = root / "input-parent"
            parent.mkdir()
            source = parent / "data.pkl"
            source.write_bytes(b"pinned-source")
            original_parent = root / "input-parent-original"
            real_os_open = READER.os.open

            def pin_then_replace(path: object, flags: int, *args: object, **kwargs: object) -> int:
                fd = real_os_open(path, flags, *args, **kwargs)
                if path == parent.name and kwargs.get("dir_fd") is not None:
                    os.rename(parent, original_parent)
                    parent.mkdir()
                    (parent / "data.pkl").write_bytes(b"wrong-parent")
                return fd

            with mock.patch.object(READER.os, "open", side_effect=pin_then_replace):
                self.assertEqual(READER._read_bounded(source), b"pinned-source")
            self.assertEqual(source.read_bytes(), b"wrong-parent")

    def test_symlinked_parent_is_rejected_for_input_and_output(self) -> None:
        with temp_root() as root:
            real_parent = root / "real-parent"
            real_parent.mkdir()
            (real_parent / "data.pkl").write_bytes(b"not-used")
            linked_parent = root / "linked-parent"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaises(READER.MetadataError):
                READER._read_bounded(linked_parent / "data.pkl")
            with self.assertRaises(READER.MetadataError):
                READER._write_exclusive(linked_parent / "metadata.json", b"not-used")

    def test_failed_output_write_leaves_unaccepted_artifact_without_cleanup(self) -> None:
        with temp_root() as root:
            output = root / "metadata.json"
            with mock.patch.object(READER.os, "write", side_effect=OSError("synthetic write failure")):
                with self.assertRaises(READER.MetadataError):
                    READER._write_exclusive(output, b"candidate")
            self.assertTrue(output.exists())
            self.assertEqual(output.read_bytes(), b"")

    def test_cli_darwin_guard_fails_before_input_read(self) -> None:
        with temp_root() as root:
            input_path = root / "data.pkl"
            output = root / "metadata.json"
            with mock.patch.object(READER.sys, "platform", "darwin"):
                with mock.patch.dict(READER.os.environ, {READER.REMOTE_MARKER: "1"}):
                    with mock.patch.object(READER, "_read_bounded", side_effect=AssertionError("must not read")) as reader:
                        self.assertEqual(READER.main([str(input_path), str(output)]), 2)
                        reader.assert_not_called()

    def test_cli_remote_guard_fails_before_input_read(self) -> None:
        with temp_root() as root:
            input_path = root / "data.pkl"
            output = root / "metadata.json"
            with mock.patch.object(READER.sys, "platform", "linux"):
                with mock.patch.dict(READER.os.environ, {READER.REMOTE_MARKER: "0"}):
                    with mock.patch.object(READER, "_read_bounded", side_effect=AssertionError("must not read")) as reader:
                        self.assertEqual(READER.main([str(input_path), str(output)]), 2)
                        reader.assert_not_called()

    def test_cli_synthetic_capture_requires_data_pkl_and_remote_marker(self) -> None:
        with temp_root() as root:
            input_path = root / "data.pkl"
            output = root / "metadata.json"
            input_path.write_bytes(envelope())
            with mock.patch.object(READER.sys, "platform", "linux"):
                with mock.patch.dict(READER.os.environ, {READER.REMOTE_MARKER: "1"}):
                    self.assertEqual(READER.main([str(input_path), str(output)]), 0)
            self.assertEqual(output.read_text(encoding="utf-8").splitlines()[0], "{")


if __name__ == "__main__":
    unittest.main()
