#!/usr/bin/env python3
"""Extract a tensor manifest from a PyTorch ``data.pkl`` without torch.

PyTorch ZIP checkpoints keep the state-dict graph in ``data.pkl`` and tensor
payloads in separate ``data/<storage>`` members.  This tool reads only the
pickle graph, replaces storage references with inert records, and emits the
name/dtype/shape contract without opening any tensor payload.

The unpickler is intentionally fail-closed.  Pickle can execute arbitrary
code, so only the four globals used by a plain PyTorch state dict are accepted:
``collections.OrderedDict``, ``torch._utils._rebuild_tensor_v2``, and the
explicit storage classes listed in ``STORAGE_DTYPES``.  Any other global or
any non-tensor value aborts instead of being imported or executed.

Run through the repository Python policy:

    uv run --no-project --python 3.12 python \
      tools/audit/torch_pickle_manifest.py data.pkl manifest.json \
      --source nvidia/model@<40-hex-revision>:model_weights.ckpt

This is a structural audit, not numerical parity: no reference forward runs
and no tensor values are read.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import pickle
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO


STORAGE_DTYPES = {
    ("torch", "FloatStorage"): "F32",
    ("torch", "DoubleStorage"): "F64",
    ("torch", "HalfStorage"): "F16",
    ("torch", "BFloat16Storage"): "BF16",
    ("torch", "ByteStorage"): "U8",
    ("torch", "CharStorage"): "I8",
    ("torch", "ShortStorage"): "I16",
    ("torch", "IntStorage"): "I32",
    ("torch", "LongStorage"): "I64",
    ("torch", "BoolStorage"): "BOOL",
}


class ManifestError(ValueError):
    """The pickle is not a fail-closed plain tensor state dict."""


@dataclass(frozen=True)
class StorageType:
    dtype: str


@dataclass(frozen=True)
class StorageRef:
    dtype: str
    key: str
    location: str
    numel: int


@dataclass(frozen=True)
class TensorRef:
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_key: str
    storage_offset: int
    storage_numel: int
    location: str


def rebuild_tensor_v2(
    storage: StorageRef,
    storage_offset: int,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    requires_grad: bool,
    backward_hooks: Any,
    metadata: Any = None,
) -> TensorRef:
    """Inert replacement for ``torch._utils._rebuild_tensor_v2``."""

    del requires_grad, backward_hooks, metadata
    if not isinstance(storage, StorageRef):
        raise ManifestError("tensor rebuild did not receive a storage reference")
    if not isinstance(storage_offset, int) or storage_offset < 0:
        raise ManifestError(f"invalid tensor storage offset {storage_offset!r}")
    if not isinstance(shape, tuple) or not all(
        isinstance(axis, int) and axis >= 0 for axis in shape
    ):
        raise ManifestError(f"invalid tensor shape {shape!r}")
    if not isinstance(stride, tuple) or not all(
        isinstance(axis, int) and axis >= 0 for axis in stride
    ):
        raise ManifestError(f"invalid tensor stride {stride!r}")
    return TensorRef(
        dtype=storage.dtype,
        shape=shape,
        stride=stride,
        storage_key=storage.key,
        storage_offset=storage_offset,
        storage_numel=storage.numel,
        location=storage.location,
    )


class RestrictedStateDictUnpickler(pickle.Unpickler):
    """Unpickler that never imports checkpoint-selected code."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return rebuild_tensor_v2
        dtype = STORAGE_DTYPES.get((module, name))
        if dtype is not None:
            return StorageType(dtype)
        raise ManifestError(f"pickle global `{module}.{name}` is not allowed")

    def persistent_load(self, persistent_id: Any) -> StorageRef:
        if not isinstance(persistent_id, tuple) or len(persistent_id) != 5:
            raise ManifestError(
                f"invalid PyTorch persistent id {persistent_id!r}; expected 5-tuple"
            )
        kind, storage_type, key, location, numel = persistent_id
        if kind != "storage" or not isinstance(storage_type, StorageType):
            raise ManifestError(
                f"unsupported PyTorch persistent id {persistent_id!r}"
            )
        if not isinstance(key, str) or not isinstance(location, str):
            raise ManifestError("storage key and location must be strings")
        if not isinstance(numel, int) or numel < 0:
            raise ManifestError(f"invalid storage element count {numel!r}")
        return StorageRef(storage_type.dtype, key, location, numel)


def load_manifest(source: BinaryIO) -> collections.OrderedDict[str, TensorRef]:
    """Load and validate a plain tensor state dict from ``data.pkl``."""

    root = RestrictedStateDictUnpickler(source).load()
    if not isinstance(root, collections.OrderedDict) or not root:
        raise ManifestError("pickle root must be a non-empty OrderedDict")
    for name, value in root.items():
        if not isinstance(name, str) or not name:
            raise ManifestError(f"state-dict key must be a non-empty string: {name!r}")
        if not isinstance(value, TensorRef):
            raise ManifestError(
                f"state-dict entry `{name}` is {type(value).__name__}, expected tensor"
            )
    return root


def render_manifest(
    state_dict: collections.OrderedDict[str, TensorRef],
    source_label: str,
    pickle_sha256: str,
) -> dict[str, Any]:
    tensors = {
        name: {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "stride": list(tensor.stride),
            "storage_key": tensor.storage_key,
            "storage_offset": tensor.storage_offset,
            "storage_numel": tensor.storage_numel,
            "location": tensor.location,
        }
        for name, tensor in sorted(state_dict.items())
    }
    storage_canonical = json.dumps(
        tensors, sort_keys=True, separators=(",", ":")
    ).encode()

    def name_shape_digest(items: list[tuple[str, TensorRef]]) -> str:
        canonical = bytearray()
        for name, tensor in items:
            canonical.extend(name.encode("utf-8"))
            canonical.append(0)
            canonical.extend(struct.pack("<Q", len(tensor.shape)))
            for dimension in tensor.shape:
                canonical.extend(struct.pack("<Q", dimension))
        return hashlib.sha256(canonical).hexdigest()

    sorted_items = sorted(state_dict.items())
    float_items = [
        item for item in sorted_items if item[1].dtype in {"F32", "F16", "BF16"}
    ]
    return {
        "format": "vokra-pytorch-state-dict-manifest-v1",
        "source": source_label,
        "data_pickle_sha256": pickle_sha256,
        "tensor_count": len(tensors),
        # Same canonical `(name, dimensions)` digest consumed by the Rust
        # StrictCheckpoint binder. Dtype/storage changes are tracked by the
        # richer digest below but do not block a legitimate GGUF quantization.
        "manifest_sha256": name_shape_digest(sorted_items),
        "float_tensor_count": len(float_items),
        "float_manifest_sha256": name_shape_digest(float_items),
        "storage_manifest_sha256": hashlib.sha256(storage_canonical).hexdigest(),
        "tensors": tensors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_pickle", type=Path, help="extracted PyTorch data.pkl")
    parser.add_argument("output", type=Path, help="manifest JSON to write")
    parser.add_argument(
        "--source",
        required=True,
        help="immutable source label, preferably repo@40-hex-revision:file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = args.data_pickle.read_bytes()
    state_dict = load_manifest(io.BytesIO(raw))
    manifest = render_manifest(
        state_dict,
        args.source,
        hashlib.sha256(raw).hexdigest(),
    )
    body = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.write_text(body, encoding="utf-8")
    print(
        f"tensor_count={manifest['tensor_count']} "
        f"manifest_sha256={manifest['manifest_sha256']} "
        f"data_pickle_sha256={manifest['data_pickle_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
