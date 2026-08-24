#!/usr/bin/env python3
"""Read a public Hugging Face GGUF contract without downloading tensor data.

The GGUF metadata and tensor descriptors precede the tensor payload.  This
tool requests only that prefix, parses the version-3 header, and prints the
metadata plus the name/type/shape manifest used by runtime-binder audits.

Run through the repository Python policy:

    uv run --no-project --python 3.12 python tools/audit/hf_gguf_contract.py \
        vokra/wav2vec2-base-960h wav2vec2-base-960h.gguf
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


USER_AGENT = "vokra-hf-gguf-contract/1.0"
DEFAULT_HEAD_BYTES = 4 * 1024 * 1024


class HeaderError(ValueError):
    """Raised when a GGUF prefix is malformed or too short."""


@dataclass
class Reader:
    data: bytes
    pos: int = 0

    def take(self, size: int) -> bytes:
        end = self.pos + size
        if end > len(self.data):
            raise HeaderError(
                f"GGUF header needs more than {len(self.data)} bytes; "
                "increase --head-bytes"
            )
        value = self.data[self.pos : end]
        self.pos = end
        return value

    def unpack(self, fmt: str) -> Any:
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.take(size))[0]

    def string(self) -> str:
        size = self.unpack("<Q")
        return self.take(size).decode("utf-8")


SCALARS: dict[int, str] = {
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

GGML_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
}


def value(reader: Reader, kind: int) -> Any:
    if kind in SCALARS:
        return reader.unpack(SCALARS[kind])
    if kind == 8:
        return reader.string()
    if kind == 9:
        element_kind = reader.unpack("<I")
        length = reader.unpack("<Q")
        return [value(reader, element_kind) for _ in range(length)]
    raise HeaderError(f"unsupported GGUF metadata value type {kind}")


def parse_header(data: bytes) -> dict[str, Any]:
    reader = Reader(data)
    if reader.take(4) != b"GGUF":
        raise HeaderError("not a GGUF file")
    version = reader.unpack("<I")
    if version != 3:
        raise HeaderError(f"GGUF version {version} is unsupported (expected 3)")
    tensor_count = reader.unpack("<Q")
    metadata_count = reader.unpack("<Q")
    metadata: dict[str, Any] = {}
    for _ in range(metadata_count):
        key = reader.string()
        metadata[key] = value(reader, reader.unpack("<I"))
    tensors = []
    for _ in range(tensor_count):
        name = reader.string()
        n_dims = reader.unpack("<I")
        shape = [reader.unpack("<Q") for _ in range(n_dims)]
        dtype = reader.unpack("<I")
        offset = reader.unpack("<Q")
        tensors.append(
            {
                "name": name,
                "shape": shape,
                "dtype": GGML_TYPES.get(dtype, f"GGML_TYPE_{dtype}"),
                "offset": offset,
            }
        )
    return {
        "version": version,
        "tensor_count": tensor_count,
        "metadata_count": metadata_count,
        "header_bytes": reader.pos,
        "metadata": metadata,
        "tensors": tensors,
    }


def fetch_prefix(repo: str, revision: str, filename: str, size: int) -> bytes:
    quoted_file = urllib.parse.quote(filename, safe="/")
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{quoted_file}"
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes=0-{size - 1}",
            "User-Agent": USER_AGENT,
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read(size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="Hugging Face repo, for example vokra/model")
    parser.add_argument("filename", help="GGUF filename inside the repo")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--head-bytes", type=int, default=DEFAULT_HEAD_BYTES)
    parser.add_argument(
        "--metadata-prefix",
        action="append",
        default=["general.", "vokra."],
        help="metadata prefix to include; may be repeated",
    )
    parser.add_argument("--names-only", action="store_true")
    parser.add_argument(
        "--tensor-regex",
        help="include only tensor names matched by this regular expression",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.head_bytes <= 0:
        raise ValueError("--head-bytes must be positive")
    parsed = parse_header(
        fetch_prefix(args.repo, args.revision, args.filename, args.head_bytes)
    )
    parsed["repo"] = args.repo
    parsed["revision"] = args.revision
    parsed["filename"] = args.filename
    parsed["metadata"] = {
        key: item
        for key, item in parsed["metadata"].items()
        if any(key.startswith(prefix) for prefix in args.metadata_prefix)
    }
    if args.tensor_regex:
        pattern = re.compile(args.tensor_regex)
        parsed["tensors"] = [
            item for item in parsed["tensors"] if pattern.search(item["name"])
        ]
    if args.names_only:
        parsed["tensors"] = [item["name"] for item in parsed["tensors"]]
    print(json.dumps(parsed, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
