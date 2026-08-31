#!/usr/bin/env -S uv run --no-project --python 3.12 python
"""Dependency-free authenticated TFLite FlatBuffer tensor inventory.

This is intentionally not an interpreter: persistent constants are classified
only from FlatBuffer Tensor.buffer -> Buffer.data ownership, never from a
runtime ``get_tensor`` result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path
from typing import Any

FORMAT = "vokra-microwakeword-tflite-tensor-manifest-v1"
PRODUCER = {"name": "microwakeword_tensor_manifest.py", "version": "1.0", "method": "raw_flatbuffer"}
DTYPES = {0: ("float32", 4), 1: ("float16", 2), 2: ("int32", 4), 3: ("uint8", 1), 4: ("int64", 8), 5: ("string", 1), 6: ("bool", 1), 7: ("int16", 2), 8: ("complex64", 8), 9: ("int8", 1), 10: ("float64", 8), 11: ("resource", 1), 12: ("variant", 1), 13: ("uint16", 2), 14: ("uint32", 4), 15: ("uint64", 8), 16: ("int4", 1), 17: ("bfloat16", 2)}
SUPPORTED = {"float32", "int32", "int8"}
TFLITE_MODEL_VERSION = 3
# TensorFlow Lite's schema_v3.fbs declares Model.version as ``uint`` without
# an explicit default. FlatBuffers therefore supplies scalar default 0 when
# the slot is omitted; only an explicit value of 3 is accepted here.


class Reader:
    def __init__(self, data: bytes):
        self.data = data
        if len(data) < 8 or data[4:8] != b"TFL3":
            raise ValueError("TFL3 identifier missing")
        self.root = self.u32(0)
        self._check(self.root, 4)

    def _check(self, offset: int, size: int) -> None:
        if offset < 0 or size < 0 or offset + size > len(self.data):
            raise ValueError("FlatBuffer offset is out of bounds")

    def u8(self, offset: int) -> int:
        self._check(offset, 1)
        return self.data[offset]

    def u16(self, offset: int) -> int:
        self._check(offset, 2)
        return struct.unpack_from("<H", self.data, offset)[0]

    def i32(self, offset: int) -> int:
        self._check(offset, 4)
        return struct.unpack_from("<i", self.data, offset)[0]

    def u32(self, offset: int) -> int:
        self._check(offset, 4)
        return struct.unpack_from("<I", self.data, offset)[0]

    def table_field(self, table: int, field: int, width: int = 4) -> int | None:
        """Return a field address after validating the complete table bounds.

        ``width`` is the schema width of the value at this field (one byte for
        enum/bool, four bytes for scalar/uoffset fields). FlatBuffers permits
        trailing fields to be omitted; a missing vtable slot therefore means
        the schema default, while a present slot must stay within the object.
        """
        if width <= 0:
            raise ValueError("FlatBuffer field width must be positive")
        vtable_distance = self.i32(table)
        if vtable_distance <= 0 or vtable_distance > table:
            raise ValueError("FlatBuffer vtable offset is invalid")
        vtable = table - vtable_distance
        self._check(vtable, 4)
        vsize = self.u16(vtable)
        osize = self.u16(vtable + 2)
        if vsize < 4 or osize < 4:
            raise ValueError("FlatBuffer table sizes are invalid")
        self._check(vtable, vsize)
        self._check(table, osize)
        if vtable + vsize > table:
            raise ValueError("FlatBuffer vtable overlaps table object")
        slot = vtable + 4 + field * 2
        if slot + 2 > vtable + vsize:
            return None
        relative = self.u16(slot)
        if relative == 0:
            return None
        if relative < 4 or relative + width > osize:
            raise ValueError("FlatBuffer field extends beyond table object")
        address = table + relative
        self._check(address, width)
        return address

    def indirect(self, address: int) -> int:
        offset = self.u32(address)
        if offset == 0:
            raise ValueError("FlatBuffer uoffset must be nonzero")
        target = address + offset
        if target <= address:
            raise ValueError("FlatBuffer uoffset must point forward")
        self._check(target, 4)
        return target

    def vector(self, address: int, element_size: int) -> tuple[int, int]:
        target = self.indirect(address)
        length = self.u32(target)
        start = target + 4
        self._check(start, length * element_size)
        return start, length

    def vector_uoffsets(self, address: int) -> list[int]:
        start, length = self.vector(address, 4)
        return [self.indirect(start + index * 4) for index in range(length)]

    def string(self, address: int) -> str:
        target = self.indirect(address)
        length = self.u32(target)
        start = target + 4
        self._check(start, length + 1)
        if self.u8(start + length) != 0:
            raise ValueError("FlatBuffer string is not NUL terminated")
        try:
            return self.data[start:start + length].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("FlatBuffer string is not UTF-8") from error

    def bytes_vector(self, address: int) -> bytes:
        start, length = self.vector(address, 1)
        return self.data[start:start + length]

    def i32_vector(self, address: int) -> list[int]:
        start, length = self.vector(address, 4)
        return [struct.unpack_from("<i", self.data, start + index * 4)[0] for index in range(length)]


def parse(data: bytes) -> dict[str, Any]:
    reader = Reader(data)
    model = reader.root
    version_field = reader.table_field(model, 0, 4)
    version = reader.u32(version_field) if version_field is not None else 0
    if version != TFLITE_MODEL_VERSION:
        raise ValueError(f"unsupported TFLite Model schema version: {version}")
    subgraphs_field = reader.table_field(model, 2, 4)
    buffers_field = reader.table_field(model, 4, 4)
    if subgraphs_field is None or buffers_field is None:
        raise ValueError("TFLite Model lacks subgraphs or buffers")
    subgraphs = reader.vector_uoffsets(subgraphs_field)
    buffers = reader.vector_uoffsets(buffers_field)
    if len(subgraphs) != 1:
        raise ValueError(f"unsupported TFLite subgraph count: {len(subgraphs)}")
    if not buffers:
        raise ValueError("TFLite Model must contain the empty buffer-0 sentinel")
    buffer_data: list[bytes] = []
    for buffer in buffers:
        data_field = reader.table_field(buffer, 0, 4)
        # An omitted optional vector slot is the schema's null default.  A
        # present slot still carries a FlatBuffers uoffset and must go through
        # `indirect()` so zero/backward offsets cannot be silently reclassified
        # as an empty buffer.
        buffer_data.append(reader.bytes_vector(data_field) if data_field is not None else b"")
    if buffer_data[0]:
        raise ValueError("TFLite buffer 0 must be the empty sentinel")
    subgraph = subgraphs[0]
    tensors_field = reader.table_field(subgraph, 0, 4)
    if tensors_field is None:
        raise ValueError("TFLite subgraph lacks tensors")
    tensor_tables = reader.vector_uoffsets(tensors_field)
    tensors: list[dict[str, Any]] = []
    names: set[str] = set()
    ownership: dict[int, list[int]] = {}
    for index, tensor in enumerate(tensor_tables):
        shape_field = reader.table_field(tensor, 0, 4)
        type_field = reader.table_field(tensor, 1, 1)
        buffer_field = reader.table_field(tensor, 2, 4)
        name_field = reader.table_field(tensor, 3, 4)
        if shape_field is None or name_field is None:
            raise ValueError(f"tensor {index} has missing required fields")
        shape = reader.i32_vector(shape_field)
        if any(dimension < 0 for dimension in shape):
            raise ValueError(f"tensor {index} has ambiguous negative shape")
        signature_field = reader.table_field(tensor, 7, 4)
        if signature_field is not None and any(
            dimension < 0 for dimension in reader.i32_vector(signature_field)
        ):
            raise ValueError(f"tensor {index} has an ambiguous shape signature")
        name = reader.string(name_field)
        if not name or name in names:
            raise ValueError(f"tensor {index} has a missing or duplicate name")
        names.add(name)
        # Tensor.type defaults to FLOAT32 (0), and Tensor.buffer defaults to
        # buffer 0 when their vtable slots are omitted by FlatBuffers.
        dtype_code = reader.u8(type_field) if type_field is not None else 0
        if dtype_code not in DTYPES:
            raise ValueError(f"tensor {index} has unsupported dtype code {dtype_code}")
        dtype, item_size = DTYPES[dtype_code]
        buffer_index = reader.u32(buffer_field) if buffer_field is not None else 0
        if buffer_index >= len(buffer_data):
            raise ValueError(f"tensor {index} buffer index is out of bounds")
        variable_field = reader.table_field(tensor, 5, 1)
        if variable_field is not None and reader.u8(variable_field):
            raise ValueError(f"tensor {index} is variable")
        payload = buffer_data[buffer_index]
        if not payload:
            continue
        if dtype not in SUPPORTED:
            raise ValueError(f"constant tensor {index} has unsupported dtype {dtype}")
        elements = 1
        for dimension in shape:
            elements *= dimension
        expected = elements * item_size
        if expected != len(payload):
            raise ValueError(f"constant tensor {index} byte size {len(payload)} != shape element size {expected}")
        tensors.append({"index": index, "name": name, "type": dtype_code, "dtype": dtype, "shape": shape, "buffer_index": buffer_index, "buffer_size": len(payload), "buffer_sha256": hashlib.sha256(payload).hexdigest(), "kind": "constant"})
        ownership.setdefault(buffer_index, []).append(index)
    if not tensors:
        raise ValueError("TFLite model has no nonempty supported constant buffers")
    buffer_ownership = []
    for buffer_index, tensor_indices in sorted(ownership.items()):
        payload = buffer_data[buffer_index]
        buffer_ownership.append({"buffer_index": buffer_index, "tensor_indices": tensor_indices, "tensor_count": len(tensor_indices), "shared": len(tensor_indices) > 1, "buffer_size": len(payload), "buffer_sha256": hashlib.sha256(payload).hexdigest()})
    owned_indices = [index for item in buffer_ownership for index in item["tensor_indices"]]
    if len(owned_indices) != len(set(owned_indices)) or sorted(owned_indices) != sorted(item["index"] for item in tensors):
        raise ValueError("duplicate or incomplete nonempty buffer ownership")
    nonempty = [index for index, payload in enumerate(buffer_data) if payload]
    if any(not item["tensor_indices"] or item["buffer_size"] <= 0 for item in buffer_ownership):
        raise ValueError("invalid empty referenced buffer ownership")
    if any(item["buffer_index"] not in nonempty for item in buffer_ownership):
        raise ValueError("referenced buffer ownership is not nonempty")
    return {"format": FORMAT, "producer": PRODUCER, "complete": True, "source_sha256": hashlib.sha256(data).hexdigest(), "source_size": len(data), "subgraph_count": 1, "tensor_count": len(tensor_tables), "buffer_count": len(buffers), "constant_count": len(tensors), "nonempty_buffer_count": len(nonempty), "referenced_nonempty_buffer_count": len(buffer_ownership), "unreferenced_nonempty_buffer_indices": [index for index in nonempty if index not in ownership], "buffer_ownership": buffer_ownership, "tensors": tensors}


def publish(path: Path, value: dict[str, Any]) -> None:
    if path.parent.is_symlink():
        raise ValueError(f"manifest output exists or is unsafe: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ValueError(f"manifest output exists or is unsafe: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False, mode="w", encoding="utf-8") as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(value, sort_keys=True, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def self_test() -> None:
    # Independently assembled FlatBuffer fixture with one activation and one
    # float32 constant; the parser does not depend on a TFLite/runtime package.
    def fixture() -> bytes:
        out = bytearray(b"\x00\x00\x00\x00TFL3")
        def table(fields: int, size: int) -> int:
            vt = len(out); out.extend(struct.pack("<HH", 4 + fields * 2, size)); out.extend(b"\x00" * (fields * 2))
            tab = len(out); out.extend(struct.pack("<i", tab - vt)); out.extend(b"\x00" * (size - 4))
            for i in range(fields): struct.pack_into("<H", out, vt + 4 + i * 2, 4 + i * 4)
            return tab
        def vector_slots(count: int) -> int:
            start = len(out); out.extend(struct.pack("<I", count)); out.extend(b"\x00" * (4 * count)); return start
        def patch_vector(start: int, targets: list[int]) -> None:
            for i, target in enumerate(targets): struct.pack_into("<I", out, start + 4 + i * 4, target - (start + 4 + i * 4))
        def vector_i32(values: list[int]) -> int:
            start = len(out); out.extend(struct.pack("<I", len(values))); out.extend(struct.pack("<" + "i" * len(values), *values)); return start
        def string(value: str) -> int:
            start = len(out); encoded = value.encode(); out.extend(struct.pack("<I", len(encoded))); out.extend(encoded + b"\x00"); return start
        def field_ptr(tab: int, field: int, target: int) -> None: struct.pack_into("<I", out, tab + 4 + field * 4, target - (tab + 4 + field * 4))
        def omit_field(tab: int, field: int) -> None:
            vtable = tab - struct.unpack_from("<i", out, tab)[0]
            struct.pack_into("<H", out, vtable + 4 + field * 2, 0)
        mtab = table(8, 36); struct.pack_into("<I", out, mtab + 4, TFLITE_MODEL_VERSION)
        sv = vector_slots(1); bv = vector_slots(2); field_ptr(mtab, 2, sv); field_ptr(mtab, 4, bv)
        stab = table(5, 24); tv = vector_slots(2); field_ptr(stab, 0, tv)
        ttab = table(8, 36); struct.pack_into("<B", out, ttab + 8, 0); struct.pack_into("<I", out, ttab + 12, 1); omit_field(ttab, 1); omit_field(ttab, 7)
        shape = vector_i32([1]); name = string("constant"); field_ptr(ttab, 0, shape); field_ptr(ttab, 3, name)
        atab = table(8, 36); struct.pack_into("<B", out, atab + 8, 0); struct.pack_into("<I", out, atab + 12, 1); omit_field(atab, 1); omit_field(atab, 2); omit_field(atab, 7)
        shape2 = vector_i32([1]); name2 = string("activation"); field_ptr(atab, 0, shape2); field_ptr(atab, 3, name2)
        btab = table(1, 8); data_vec = len(out); out.extend(struct.pack("<I", 4) + struct.pack("<f", 1.0)); field_ptr(btab, 0, data_vec)
        empty_btab = table(1, 8); omit_field(empty_btab, 0)
        patch_vector(sv, [stab]); patch_vector(bv, [empty_btab, btab]); patch_vector(tv, [ttab, atab])
        struct.pack_into("<I", out, 0, mtab)
        return bytes(out)
    result = parse(fixture())
    assert result["complete"] and result["tensor_count"] == 2 and result["constant_count"] == 1
    assert result["tensors"][0]["name"] == "constant"
    # The fixture explicitly carries Model.version=3 and deliberately omits
    # Tensor.type and the activation Tensor.buffer slots. Their schema defaults
    # (FLOAT32 and buffer 0) must be applied rather than treated as missing.
    assert result["tensors"][0]["type"] == 0
    assert result["tensors"][0]["buffer_index"] == 1
    omitted_version = bytearray(fixture())
    root = struct.unpack_from("<I", omitted_version, 0)[0]
    vtable = root - struct.unpack_from("<i", omitted_version, root)[0]
    struct.pack_into("<H", omitted_version, vtable + 4, 0)
    try: parse(bytes(omitted_version))
    except ValueError: pass
    else: raise AssertionError("omitted Model.version was treated as schema version 3")
    with tempfile.TemporaryDirectory(prefix="mww-manifest-") as directory:
        path = Path(directory) / "manifest.json"; publish(path, result)
        try: publish(path, result)
        except ValueError: pass
        else: raise AssertionError("manifest clobber accepted")
        assert not list(path.parent.glob("*.tmp"))
    bad = bytearray(fixture()); bad[4:8] = b"BAD!"
    try: parse(bytes(bad))
    except ValueError: pass
    else: raise AssertionError("invalid identifier accepted")
    try: parse(fixture()[:-1])
    except ValueError: pass
    else: raise AssertionError("truncated FlatBuffer accepted")
    malformed_table = bytearray(fixture())
    root = struct.unpack_from("<I", malformed_table, 0)[0]
    vtable = root - struct.unpack_from("<i", malformed_table, root)[0]
    struct.pack_into("<H", malformed_table, vtable, 2)
    try: parse(bytes(malformed_table))
    except ValueError: pass
    else: raise AssertionError("malformed vtable size accepted")
    for distance in (-1, len(malformed_table)):
        malformed_offset = bytearray(fixture())
        root = struct.unpack_from("<I", malformed_offset, 0)[0]
        struct.pack_into("<i", malformed_offset, root, distance)
        try: parse(bytes(malformed_offset))
        except ValueError: pass
        else: raise AssertionError("invalid forward/negative vtable offset accepted")
    wrong_version = bytearray(fixture())
    root = struct.unpack_from("<I", wrong_version, 0)[0]
    vtable = root - struct.unpack_from("<i", wrong_version, root)[0]
    struct.pack_into("<H", wrong_version, vtable + 4, 4)
    struct.pack_into("<I", wrong_version, root + 4, 2)
    try: parse(bytes(wrong_version))
    except ValueError: pass
    else: raise AssertionError("unsupported Model schema version accepted")
    present_zero_data = bytearray(fixture())
    reader = Reader(bytes(present_zero_data))
    buffers_field = reader.table_field(reader.root, 4, 4)
    buffer_vector = reader.vector_uoffsets(buffers_field)
    data_field = reader.table_field(buffer_vector[1], 0, 4)
    struct.pack_into("<I", present_zero_data, data_field, 0)
    try: parse(bytes(present_zero_data))
    except ValueError: pass
    else: raise AssertionError("present-zero Buffer.data uoffset was accepted")
    present_zero_signature = bytearray(fixture())
    reader = Reader(bytes(present_zero_signature))
    tensors_field = reader.table_field(reader.vector_uoffsets(reader.table_field(reader.root, 2, 4))[0], 0, 4)
    tensor = reader.vector_uoffsets(tensors_field)[0]
    tensor_vtable = tensor - struct.unpack_from("<i", present_zero_signature, tensor)[0]
    struct.pack_into("<H", present_zero_signature, tensor_vtable + 4 + 7 * 2, 32)
    struct.pack_into("<I", present_zero_signature, tensor + 4 + 7 * 4, 0)
    try: parse(bytes(present_zero_signature))
    except ValueError: pass
    else: raise AssertionError("present-zero Tensor.shape_signature uoffset was accepted")
    nonempty_sentinel = bytearray(fixture())
    reader = Reader(bytes(nonempty_sentinel))
    model = reader.root
    buffers_field = reader.table_field(model, 4, 4)
    buffer_vector = reader.vector_uoffsets(buffers_field)
    first_buffer = buffer_vector[0]
    second_buffer = buffer_vector[1]
    second_data = reader.table_field(second_buffer, 0, 4)
    data_vector = reader.indirect(second_data)
    # Append a valid forward vector so the mutation isolates the buffer-0
    # sentinel rule rather than being rejected first as a backward uoffset.
    data_vector = len(nonempty_sentinel)
    nonempty_sentinel.extend(struct.pack("<I", 1) + b"x")
    first_data_slot = first_buffer + 4
    first_vtable = first_buffer - struct.unpack_from("<i", nonempty_sentinel, first_buffer)[0]
    struct.pack_into("<H", nonempty_sentinel, first_vtable + 4, 4)
    struct.pack_into("<I", nonempty_sentinel, first_data_slot, data_vector - first_data_slot)
    try: parse(bytes(nonempty_sentinel))
    except ValueError: pass
    else: raise AssertionError("nonempty buffer 0 was accepted")
    for offset in (0, 0xFFFF_FFF0):
        bad_uoffset = bytearray(fixture())
        reader = Reader(bytes(bad_uoffset))
        model = reader.root
        buffers_field = reader.table_field(model, 4, 4)
        vector_start, _ = reader.vector(buffers_field, 4)
        struct.pack_into("<I", bad_uoffset, vector_start, offset)
        try: parse(bytes(bad_uoffset))
        except ValueError: pass
        else: raise AssertionError("invalid FlatBuffer uoffset was accepted")
    with tempfile.TemporaryDirectory(prefix="mww-manifest-link-") as directory:
        root = Path(directory); source = root / "model.tflite"; source.write_bytes(fixture())
        target = root / "manifest.json"; publish(target, parse(source.read_bytes()))
        try: publish(target, parse(source.read_bytes()))
        except ValueError: pass
        else: raise AssertionError("rerun manifest publication clobbered destination")
        link = root / "manifest-link.json"; link.symlink_to(target)
        try: publish(link, parse(source.read_bytes()))
        except ValueError: pass
        else: raise AssertionError("symlink manifest destination accepted")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--input", type=Path); parser.add_argument("--output", type=Path); parser.add_argument("--self-test", action="store_true"); args = parser.parse_args()
    if args.self_test: self_test(); print("microwakeword tensor manifest self-test: PASS"); return 0
    if not args.input or not args.output: parser.error("--input and --output are required")
    if args.input.is_symlink() or not args.input.is_file(): raise SystemExit("authenticated TFLite input must be a regular file")
    if args.output.resolve(strict=False) == args.input.resolve(strict=False): raise SystemExit("manifest output aliases TFLite input")
    publish(args.output, parse(args.input.read_bytes())); return 0


if __name__ == "__main__":
    raise SystemExit(main())
