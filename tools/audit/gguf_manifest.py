#!/usr/bin/env python3
"""Print GGUF metadata and tensor shapes without loading tensor payloads.

Run through the repository Python policy:

    uv run --no-project --python 3.12 python tools/audit/gguf_manifest.py MODEL.gguf
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import BinaryIO, Any


VALUE_TYPES = {
    0: "u8",
    1: "i8",
    2: "u16",
    3: "i16",
    4: "u32",
    5: "i32",
    6: "f32",
    7: "bool",
    8: "string",
    9: "array",
    10: "u64",
    11: "i64",
    12: "f64",
}
SCALARS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}


def read_exact(handle: BinaryIO, size: int) -> bytes:
    value = handle.read(size)
    if len(value) != size:
        raise ValueError(f"unexpected EOF: needed {size} bytes, got {len(value)}")
    return value


def scalar(handle: BinaryIO, format_: str) -> int | float | bool:
    return struct.unpack(format_, read_exact(handle, struct.calcsize(format_)))[0]


def string(handle: BinaryIO) -> str:
    size = scalar(handle, "<Q")
    return read_exact(handle, int(size)).decode("utf-8")


def metadata_value(handle: BinaryIO, value_type: int) -> Any:
    if value_type in SCALARS:
        return scalar(handle, SCALARS[value_type])
    if value_type == 8:
        return string(handle)
    if value_type == 9:
        element_type = int(scalar(handle, "<I"))
        count = int(scalar(handle, "<Q"))
        if element_type not in VALUE_TYPES or element_type == 9:
            raise ValueError(f"unsupported GGUF array element type {element_type}")
        return [metadata_value(handle, element_type) for _ in range(count)]
    raise ValueError(f"unsupported GGUF metadata type {value_type}")


def read_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("rb") as handle:
        if read_exact(handle, 4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        version = int(scalar(handle, "<I"))
        if version not in {2, 3}:
            raise ValueError(f"{path}: unsupported GGUF version {version}")
        tensor_count = int(scalar(handle, "<Q"))
        metadata_count = int(scalar(handle, "<Q"))
        metadata: dict[str, Any] = {"general.gguf_version": version}
        for _ in range(metadata_count):
            key = string(handle)
            value_type = int(scalar(handle, "<I"))
            metadata[key] = metadata_value(handle, value_type)
        tensors = []
        for _ in range(tensor_count):
            name = string(handle)
            dimensions = [
                int(scalar(handle, "<Q"))
                for _ in range(int(scalar(handle, "<I")))
            ]
            ggml_type = int(scalar(handle, "<I"))
            offset = int(scalar(handle, "<Q"))
            tensors.append(
                {
                    "name": name,
                    "dimensions": dimensions,
                    "ggml_type": ggml_type,
                    "offset": offset,
                }
            )
    return metadata, tensors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    metadata, tensors = read_manifest(args.gguf)
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    if not args.metadata_only:
        for tensor in tensors:
            dims = ",".join(str(value) for value in tensor["dimensions"])
            print(
                f"{tensor['name']}\t[{dims}]\tggml_type={tensor['ggml_type']}"
                f"\toffset={tensor['offset']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
