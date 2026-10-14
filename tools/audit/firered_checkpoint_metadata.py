#!/usr/bin/env -S uv run --no-project --offline --python 3.12 python
"""Capture restricted FireRed ``data.pkl`` structure without importing torch.

This reader accepts only a small, extracted pickle envelope containing an
``argparse.Namespace`` and a plain tensor state dictionary.  It never opens a
checkpoint archive or tensor storage payload, imports checkpoint-selected
code, executes a model, or establishes a network connection.  A successful
capture is review-pending structural metadata, not a checkpoint, license, or
parity approval.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import itertools
import json
import math
import os
import pickle
import pickletools
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO


OFFICIAL_SOURCE_REVISION = "834635e4cf277ed8ca92049fc375b17c3dc20748"
SOURCE_REPOSITORY = "FireRedTeam/FireRedASR"
DECLARED_MODEL_REPOSITORY = "FireRedTeam/FireRedASR-AED-L"
DECLARED_MODEL_REVISION = "e57f5960d03cff1071ff7acbb409314d1e70ed3d"
REMOTE_MARKER = "VOKRA_FIRERED_METADATA_REMOTE"
MAX_INPUT_BYTES = 512 * 1024
# This bounds normalized JSON construction, independently of the input cap.
MAX_OUTPUT_BYTES = 2 * 1024 * 1024
# The single-Pickler 940-descriptor synthetic fixture is 93,568 bytes / 28,286 opcodes;
# keep a small fixed headroom without allowing an unbounded opcode stream.
MAX_OPCODE_COUNT = 32_768
MAX_DEPTH = 64
MAX_CONTAINER_ITEMS = 4_096
MAX_OUTPUT_NODES = 8_192
# The realistic 940-descriptor Pickler fixture uses 6,610 memo writes and
# index 6,609; keep bounded headroom for that structural scale.
MAX_MEMO_ENTRIES = 8_192
MAX_MEMO_INDEX = 8_192
MAX_STRING_BYTES = 16 * 1024
MAX_DIMS = 32
MAX_DIMENSION = (1 << 31) - 1
MAX_STORAGE_NUMEL = (1 << 63) - 1
MAX_STORAGE_BYTES = (1 << 63) - 1
MAX_INTEGER = (1 << 63) - 1

MANDATORY_ARGS = (
    "idim",
    "n_layers_enc",
    "n_head",
    "d_model",
    "residual_dropout",
    "dropout_rate",
    "kernel_size",
    "pe_maxlen",
    "sos_id",
    "eos_id",
    "pad_id",
    "odim",
    "n_layers_dec",
)
SENSITIVE_ARG_KEYS = {
    "api_key",
    "apikey",
    "token",
    "hf_token",
    "hf_api_key",
    "access_token",
    "api_token",
    "auth_token",
    "secret",
    "secret_key",
    "client_secret",
    "password",
    "passwd",
    "passphrase",
    "credential",
    "credentials",
    "private_key",
    "aws_secret_access_key",
    "vast_api_key",
    "vastai_api_key",
    "scw_secret_key",
    "scw_access_key",
    "secret_access_key",
}

STORAGE_DTYPES = {
    "FloatStorage": ("F32", 4),
    "DoubleStorage": ("F64", 8),
    "HalfStorage": ("F16", 2),
    "BFloat16Storage": ("BF16", 2),
    "ByteStorage": ("U8", 1),
    "CharStorage": ("I8", 1),
    "ShortStorage": ("I16", 2),
    "IntStorage": ("I32", 4),
    "LongStorage": ("I64", 8),
    "BoolStorage": ("BOOL", 1),
}


class MetadataError(ValueError):
    """The bounded pickle is not the exact inert metadata envelope."""


@dataclass(frozen=True)
class StorageClass:
    dtype: str
    item_bytes: int

    def __setstate__(self, state: Any) -> None:
        raise MetadataError("storage class mutation is not allowed")


@dataclass(frozen=True)
class StorageRef:
    dtype: str
    item_bytes: int
    key: str
    location: str
    numel: int

    def __setstate__(self, state: Any) -> None:
        raise MetadataError("storage descriptor mutation is not allowed")


@dataclass(frozen=True)
class TensorRef:
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_key: str
    storage_offset: int
    storage_numel: int
    location: str

    def __setstate__(self, state: Any) -> None:
        raise MetadataError("tensor descriptor mutation is not allowed")


class InertNamespace:
    """Local surrogate for argparse.Namespace; it has no imported behavior."""

    def __setstate__(self, state: Any) -> None:
        if type(state) is not dict:
            raise MetadataError("namespace state is not a plain mapping")
        self.__dict__.update(state)


class InertOrderedDict(collections.OrderedDict):
    """OrderedDict surrogate with one explicit, bounded metadata attribute."""

    def __setstate__(self, state: Any) -> None:
        if type(state) is not dict or set(state) != {"_metadata"}:
            raise MetadataError("ordered mapping attributes are not allowed")
        metadata = state["_metadata"]
        if type(metadata) not in (dict, collections.OrderedDict, InertOrderedDict) or len(metadata) > MAX_CONTAINER_ITEMS:
            raise MetadataError("ordered mapping metadata is invalid")
        self._metadata = metadata


def _validate_inert_ordered_attributes(value: Any, *, allow_metadata: bool) -> None:
    if type(value) is not InertOrderedDict:
        return
    attributes = getattr(value, "__dict__", None)
    if type(attributes) is not dict:
        raise MetadataError("ordered mapping attributes are invalid")
    if not attributes:
        return
    if not allow_metadata or set(attributes) != {"_metadata"}:
        raise MetadataError("ordered mapping attributes are not allowed here")
    metadata = attributes["_metadata"]
    if type(metadata) not in (dict, collections.OrderedDict, InertOrderedDict) or len(metadata) > MAX_CONTAINER_ITEMS:
        raise MetadataError("ordered mapping metadata is invalid")


@dataclass
class TraversalBudget:
    remaining: int = MAX_OUTPUT_NODES
    output_remaining: int | None = None

    def __post_init__(self) -> None:
        if self.output_remaining is None:
            self.output_remaining = MAX_OUTPUT_BYTES

    def consume(self, amount: int = 1) -> None:
        if amount < 0 or self.remaining < amount:
            raise MetadataError("normalized metadata exceeds the security budget")
        self.remaining -= amount

    def consume_output(self, amount: int) -> None:
        if amount < 0 or self.output_remaining is None or self.output_remaining < amount:
            raise MetadataError("normalized metadata exceeds the output byte budget")
        self.output_remaining -= amount


def _strict_int(value: Any) -> bool:
    return type(value) is int and -MAX_INTEGER <= value <= MAX_INTEGER


def _bounded_text(value: Any) -> bool:
    return type(value) is str and 0 < len(value.encode("utf-8")) <= MAX_STRING_BYTES


def _tensor_span(shape: tuple[int, ...], stride: tuple[int, ...]) -> int:
    if any(dimension == 0 for dimension in shape):
        return 0
    span = 1
    for dimension, step in zip(shape, stride):
        span += (dimension - 1) * step
        if span > MAX_STORAGE_NUMEL:
            raise MetadataError("tensor storage span exceeds the security bound")
    return span


def _text_output_cost(value: str) -> int:
    encoded_length = len(value.encode("utf-8"))
    # JSON control-character escaping can expand one source character to six
    # output bytes; retain the UTF-8 bound for non-ASCII characters.
    return max(encoded_length, 6 * len(value)) + 64


def _descriptor_output_cost(name: str, tensor: TensorRef) -> int:
    return (
        _text_output_cost(name)
        + _text_output_cost(tensor.storage_key)
        + _text_output_cost(tensor.location)
        + 512
        + 64 * (len(tensor.shape) + len(tensor.stride))
    )


def rebuild_tensor_v2(
    storage: StorageRef,
    storage_offset: Any,
    shape: Any,
    stride: Any,
    requires_grad: Any,
    backward_hooks: Any,
    metadata: Any = None,
) -> TensorRef:
    """Inert replacement for the one accepted torch tensor reducer."""

    if not isinstance(storage, StorageRef):
        raise MetadataError("tensor reducer received an invalid storage")
    if not _strict_int(storage_offset) or storage_offset < 0:
        raise MetadataError("tensor storage offset is invalid")
    if type(shape) is not tuple or type(stride) is not tuple:
        raise MetadataError("tensor shape and stride must be tuples")
    if len(shape) != len(stride) or len(shape) > MAX_DIMS:
        raise MetadataError("tensor rank is invalid")
    if any(
        not _strict_int(dimension) or dimension < 0 or dimension > MAX_DIMENSION
        for dimension in shape
    ):
        raise MetadataError("tensor shape is invalid")
    if any(
        not _strict_int(step) or step < 0 or step > MAX_DIMENSION
        for step in stride
    ):
        raise MetadataError("tensor stride is invalid")
    _validate_inert_ordered_attributes(backward_hooks, allow_metadata=False)
    _validate_inert_ordered_attributes(metadata, allow_metadata=False)
    empty_hooks = type(backward_hooks) in (dict, collections.OrderedDict, InertOrderedDict) and not backward_hooks
    empty_metadata = type(metadata) in (dict, collections.OrderedDict, InertOrderedDict) and not metadata
    if type(requires_grad) is not bool or not (backward_hooks is None or empty_hooks) or not (metadata is None or empty_metadata):
        raise MetadataError("tensor reducer metadata is unsupported")
    if storage.numel * storage.item_bytes > MAX_STORAGE_BYTES:
        raise MetadataError("storage byte span exceeds the security bound")
    span = _tensor_span(shape, stride)
    if storage_offset > storage.numel or span > storage.numel - storage_offset:
        raise MetadataError("tensor view exceeds its storage")
    return TensorRef(
        dtype=storage.dtype,
        shape=shape,
        stride=stride,
        storage_key=storage.key,
        storage_offset=storage_offset,
        storage_numel=storage.numel,
        location=storage.location,
    )


class RestrictedUnpickler(pickle.Unpickler):
    """Resolve only local inert surrogates and known tensor descriptors."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "argparse" and name == "Namespace":
            return InertNamespace
        if module == "collections" and name == "OrderedDict":
            return InertOrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return rebuild_tensor_v2
        if module == "torch" and name in STORAGE_DTYPES:
            dtype, item_bytes = STORAGE_DTYPES[name]
            return StorageClass(dtype, item_bytes)
        raise MetadataError("pickle global is not allowed")

    def persistent_load(self, persistent_id: Any) -> StorageRef:
        if type(persistent_id) is not tuple or len(persistent_id) != 5:
            raise MetadataError("persistent storage identifier is invalid")
        kind, storage_type, key, location, numel = persistent_id
        if kind != "storage" or not isinstance(storage_type, StorageClass):
            raise MetadataError("persistent storage identifier is unsupported")
        expected = next(
            (
                (dtype, item_bytes)
                for dtype, item_bytes in STORAGE_DTYPES.values()
                if storage_type.dtype == dtype and storage_type.item_bytes == item_bytes
            ),
            None,
        )
        if expected is None:
            raise MetadataError("persistent storage type was mutated")
        if not _bounded_text(key) or not _bounded_text(location):
            raise MetadataError("persistent storage labels are invalid")
        if not _strict_int(numel) or numel < 0 or numel > MAX_STORAGE_NUMEL:
            raise MetadataError("persistent storage size is invalid")
        return StorageRef(storage_type.dtype, storage_type.item_bytes, key, location, numel)


_ALLOWED_OPCODES = {
    "PROTO",
    "FRAME",
    "STOP",
    "MARK",
    "POP",
    "POP_MARK",
    "DUP",
    "NONE",
    "NEWTRUE",
    "NEWFALSE",
    "BININT",
    "BININT1",
    "BININT2",
    "LONG1",
    "BINFLOAT",
    "BINUNICODE",
    "SHORT_BINSTRING",
    "BINSTRING",
    "EMPTY_TUPLE",
    "TUPLE",
    "TUPLE1",
    "TUPLE2",
    "TUPLE3",
    "EMPTY_DICT",
    "DICT",
    "SETITEM",
    "SETITEMS",
    "EMPTY_LIST",
    "LIST",
    "APPEND",
    "APPENDS",
    "EMPTY_SET",
    "ADDITEMS",
    "GLOBAL",
    "NEWOBJ",
    "REDUCE",
    "BUILD",
    "BINPUT",
    "LONG_BINPUT",
    "BINGET",
    "LONG_BINGET",
    "PUT",
    "GET",
    "MEMOIZE",
    "BINPERSID",
    "PERSID",
}


def _preflight(data: bytes) -> None:
    if len(data) > MAX_INPUT_BYTES:
        raise MetadataError("pickle input exceeds the security size limit")
    try:
        operations = pickletools.genops(data)
        first = next(operations)
    except (Exception,) as exc:
        raise MetadataError("pickle opcode stream is malformed") from exc
    if first[0].name != "PROTO":
        raise MetadataError("pickle protocol header is required")
    if first[1] not in (2, 3, 4):
        raise MetadataError("pickle protocol is unsupported")
    operation_count = 0
    memo_writes = 0
    mark_depth = 0
    stop_end = None
    try:
        for operation, argument, position in itertools.chain((first,), operations):
            operation_count += 1
            if operation_count > MAX_OPCODE_COUNT:
                raise MetadataError("pickle opcode count exceeds the security limit")
            name = operation.name
            if name not in _ALLOWED_OPCODES:
                raise MetadataError("pickle opcode is unsupported")
            if name == "PROTO" and position != 0:
                raise MetadataError("pickle protocol header is misplaced")
            if name == "FRAME":
                if type(argument) is not int or argument > len(data):
                    raise MetadataError("pickle frame exceeds the input")
            if name == "GLOBAL":
                if argument not in {
                    "argparse Namespace",
                    "collections OrderedDict",
                    "torch._utils _rebuild_tensor_v2",
                    *(f"torch {name}" for name in STORAGE_DTYPES),
                }:
                    raise MetadataError("pickle global is not allowed")
            if name in {"BINPUT", "BINGET", "PUT", "GET"}:
                if not _strict_int(argument) or argument < 0 or argument > MAX_MEMO_INDEX:
                    raise MetadataError("pickle memo index exceeds the security limit")
            if name == "LONG_BINPUT" or name == "LONG_BINGET":
                if not _strict_int(argument) or argument < 0 or argument > MAX_MEMO_INDEX:
                    raise MetadataError("pickle memo index exceeds the security limit")
            if name in {"BINPUT", "LONG_BINPUT", "PUT", "MEMOIZE"}:
                memo_writes += 1
                if memo_writes > MAX_MEMO_ENTRIES:
                    raise MetadataError("pickle memo exceeds the security limit")
            if name == "MARK":
                mark_depth += 1
                if mark_depth > MAX_DEPTH:
                    raise MetadataError("pickle nesting exceeds the security limit")
            elif name in {"TUPLE", "LIST", "DICT", "SETITEMS", "APPENDS", "ADDITEMS", "POP_MARK"}:
                mark_depth -= 1
                if mark_depth < 0:
                    raise MetadataError("pickle mark stack is malformed")
            if name in {"BINUNICODE", "SHORT_BINSTRING", "BINSTRING"}:
                if not isinstance(argument, (str, bytes)) or len(argument) > MAX_STRING_BYTES:
                    raise MetadataError("pickle string exceeds the security limit")
            if name == "STOP":
                stop_end = position + 1
    except MetadataError:
        raise
    except Exception as exc:
        raise MetadataError("pickle opcode stream is malformed") from exc
    if stop_end != len(data):
        raise MetadataError("pickle has trailing bytes")
    if mark_depth != 0:
        raise MetadataError("pickle mark stack is malformed")


def _safe_value(
    value: Any,
    active: set[int],
    budget: TraversalBudget,
    depth: int = 0,
) -> Any:
    if depth > MAX_DEPTH:
        raise MetadataError("argument nesting exceeds the security limit")
    budget.consume()
    if value is None or type(value) is bool:
        budget.consume_output(64)
        return value
    if type(value) is int:
        if abs(value) > MAX_INTEGER:
            raise MetadataError("argument integer exceeds the security limit")
        budget.consume_output(64)
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise MetadataError("argument number is non-finite")
        budget.consume_output(64)
        return value
    if type(value) is str:
        if len(value.encode("utf-8")) > MAX_STRING_BYTES:
            raise MetadataError("argument string exceeds the security limit")
        budget.consume_output(_text_output_cost(value))
        return value
    if type(value) is bytes:
        if len(value) > MAX_STRING_BYTES:
            raise MetadataError("argument bytes exceed the security limit")
        budget.consume_output(2 * len(value) + 96)
        return {"__bytes_hex__": value.hex()}
    identity = id(value)
    if identity in active:
        raise MetadataError("argument graph contains a cycle")
    active.add(identity)
    try:
        if type(value) is list:
            if len(value) > MAX_CONTAINER_ITEMS:
                raise MetadataError("argument list exceeds the security limit")
            budget.consume_output(64)
            result: list[Any] = []
            for item in value:
                result.append(_safe_value(item, active, budget, depth + 1))
            return result
        if type(value) is tuple:
            if len(value) > MAX_CONTAINER_ITEMS:
                raise MetadataError("argument tuple exceeds the security limit")
            budget.consume_output(96)
            result = []
            for item in value:
                result.append(_safe_value(item, active, budget, depth + 1))
            return {"__tuple__": result}
        if type(value) in (dict, collections.OrderedDict, InertOrderedDict):
            if len(value) > MAX_CONTAINER_ITEMS:
                raise MetadataError("argument mapping exceeds the security limit")
            _validate_inert_ordered_attributes(value, allow_metadata=False)
            budget.consume_output(96)
            result: dict[str, Any] = {}
            for key, item in value.items():
                if not _bounded_text(key):
                    raise MetadataError("argument mapping key is invalid")
                if key.casefold() in SENSITIVE_ARG_KEYS:
                    raise MetadataError("args contains a credential-bearing field")
                budget.consume_output(_text_output_cost(key))
                result[key] = _safe_value(item, active, budget, depth + 1)
            return result
    finally:
        active.remove(identity)
    raise MetadataError("argument value type is unsupported")


def _validate_args(namespace: InertNamespace, budget: TraversalBudget) -> dict[str, Any]:
    values = getattr(namespace, "__dict__", None)
    if type(values) is not dict:
        raise MetadataError("args is not an inert namespace")
    if len(values) > MAX_CONTAINER_ITEMS:
        raise MetadataError("args contains too many fields")
    for key in values:
        if not _bounded_text(key):
            raise MetadataError("args contains an invalid field name")
        if key.casefold() in SENSITIVE_ARG_KEYS:
            raise MetadataError("args contains a credential-bearing field")
    missing = [key for key in MANDATORY_ARGS if key not in values]
    if missing:
        raise MetadataError("args is missing a mandatory constructor field")
    integer_fields = set(MANDATORY_ARGS) - {"residual_dropout", "dropout_rate"}
    for key in integer_fields:
        if not _strict_int(values[key]) or values[key] < 0:
            raise MetadataError("mandatory integer argument is invalid")
    for key in ("residual_dropout", "dropout_rate"):
        value = values[key]
        if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(value):
            raise MetadataError("mandatory dropout argument is invalid")
        if value < 0 or value > 1:
            raise MetadataError("mandatory dropout argument is out of range")
    result: dict[str, Any] = {}
    for key in sorted(values):
        budget.consume_output(_text_output_cost(key))
        result[key] = _safe_value(values[key], set(), budget)
    return result


def _validate_tensor_ref(tensor: TensorRef) -> TensorRef:
    if type(tensor) is not TensorRef:
        raise MetadataError("model_state_dict contains an invalid tensor entry")
    if tensor.dtype not in {dtype for dtype, _item_bytes in STORAGE_DTYPES.values()}:
        raise MetadataError("tensor descriptor dtype is invalid")
    item_bytes = next(item_bytes for dtype, item_bytes in STORAGE_DTYPES.values() if dtype == tensor.dtype)
    if not _bounded_text(tensor.storage_key) or not _bounded_text(tensor.location):
        raise MetadataError("tensor descriptor labels are invalid")
    if not _strict_int(tensor.storage_numel) or tensor.storage_numel < 0 or tensor.storage_numel > MAX_STORAGE_NUMEL:
        raise MetadataError("tensor descriptor storage size is invalid")
    normalized = rebuild_tensor_v2(
        StorageRef(tensor.dtype, item_bytes, tensor.storage_key, tensor.location, tensor.storage_numel),
        tensor.storage_offset,
        tensor.shape,
        tensor.stride,
        False,
        None,
        None,
    )
    if normalized != tensor:
        raise MetadataError("tensor descriptor was mutated")
    return normalized


def _validate_state_dict(value: Any, budget: TraversalBudget) -> tuple[dict[str, TensorRef], Any]:
    if type(value) not in (dict, collections.OrderedDict, InertOrderedDict) or not value:
        raise MetadataError("model_state_dict must be a non-empty plain mapping")
    if len(value) > MAX_CONTAINER_ITEMS:
        raise MetadataError("model_state_dict has too many tensors")
    result: dict[str, TensorRef] = {}
    for key, tensor in value.items():
        if not _bounded_text(key):
            raise MetadataError("model_state_dict contains an invalid tensor entry")
        budget.consume()
        normalized = _validate_tensor_ref(tensor)
        budget.consume_output(_descriptor_output_cost(key, normalized))
        result[key] = normalized
    state_metadata = getattr(value, "_metadata", None)
    if state_metadata is not None:
        _validate_inert_ordered_attributes(value, allow_metadata=True)
        normalized_metadata = _safe_value(state_metadata, set(), budget)
    else:
        normalized_metadata = None
    return result, normalized_metadata


def _parse_root(
    data: bytes,
) -> tuple[dict[str, Any], tuple[dict[str, TensorRef], Any], TraversalBudget]:
    _preflight(data)
    try:
        root = RestrictedUnpickler(io.BytesIO(data)).load()
    except MetadataError:
        raise
    except Exception as exc:
        raise MetadataError("pickle cannot be safely decoded") from exc
    if type(root) not in (dict, collections.OrderedDict, InertOrderedDict) or set(root) != {"args", "model_state_dict"}:
        raise MetadataError("pickle root is not the exact FireRed envelope")
    _validate_inert_ordered_attributes(root, allow_metadata=False)
    if not isinstance(root.get("args"), InertNamespace):
        raise MetadataError("envelope args is not an argparse namespace")
    budget = TraversalBudget()
    return _validate_args(root["args"], budget), _validate_state_dict(root["model_state_dict"], budget), budget


def capture_bytes(data: bytes) -> dict[str, Any]:
    if type(data) is not bytes or not data or len(data) > MAX_INPUT_BYTES:
        raise MetadataError("pickle input is empty or exceeds the security limit")
    args, (tensors, state_metadata), budget = _parse_root(data)
    budget.consume_output(8192)
    rendered_tensors: dict[str, dict[str, Any]] = {}
    for name, tensor in sorted(tensors.items()):
        rendered_tensors[name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "stride": list(tensor.stride),
            "storage_key": tensor.storage_key,
            "storage_offset": tensor.storage_offset,
            "storage_numel": tensor.storage_numel,
            "location": tensor.location,
        }
    return {
        "format": "vokra-firered-checkpoint-metadata-v1",
        "status": "METADATA_CAPTURED_REVIEW_PENDING",
        "data_pickle_bytes": len(data),
        "data_pickle_sha256": hashlib.sha256(data).hexdigest(),
        "source": {
            "declared_repository": SOURCE_REPOSITORY,
            "declared_revision": OFFICIAL_SOURCE_REVISION,
            "identity_status": "DECLARED_NOT_AUTHENTICATED",
        },
        "model": {
            "declared_repository": DECLARED_MODEL_REPOSITORY,
            "declared_revision": DECLARED_MODEL_REVISION,
            "identity_status": "DECLARED_NOT_AUTHENTICATED",
            "checkpoint_sha256": None,
            "checkpoint_hash_status": "NOT_AUTHENTICATED_FULL_BYTES_REQUIRED",
        },
        "args": args,
        "model_state_dict": rendered_tensors,
        "tensor_count": len(rendered_tensors),
        "model_state_dict_metadata": state_metadata,
        "gates": {
            "real_checkpoint": "NOT_RUN",
            "model_execution": "NOT_RUN",
            "numerical_parity": "NOT_RUN",
            "license_review": "BLOCKED_PENDING_REVIEW",
        },
        "approval": None,
        "publication": "NO_UPLOAD",
    }


def load_capture(source: BinaryIO) -> dict[str, Any]:
    data = source.read(MAX_INPUT_BYTES + 1)
    if not isinstance(data, bytes) or len(data) > MAX_INPUT_BYTES:
        raise MetadataError("pickle input exceeds the security limit")
    return capture_bytes(data)


def _open_parent_dir(path: Path, label: str) -> tuple[int, str]:
    if not path.is_absolute():
        raise MetadataError(f"{label} path must be absolute")
    if not path.parts or path.name in {"", ".", ".."}:
        raise MetadataError(f"{label} path is invalid")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    current_fd: int | None = None
    try:
        current_fd = os.open(path.anchor, flags)
        if not stat.S_ISDIR(os.fstat(current_fd).st_mode):
            raise MetadataError(f"{label} parent is not a directory")
        for part in path.parts[1:-1]:
            if part in {"", ".", ".."}:
                raise MetadataError(f"{label} path is invalid")
            next_fd = os.open(part, flags, dir_fd=current_fd)
            try:
                if not stat.S_ISDIR(os.fstat(next_fd).st_mode):
                    raise MetadataError(f"{label} parent is not a directory")
            except BaseException:
                os.close(next_fd)
                raise
            previous_fd = current_fd
            current_fd = next_fd
            try:
                os.close(previous_fd)
            except BaseException:
                os.close(current_fd)
                current_fd = None
                raise
        return current_fd, path.name
    except FileNotFoundError as exc:
        if current_fd is not None:
            os.close(current_fd)
        raise MetadataError(f"{label} path is missing") from exc
    except MetadataError:
        if current_fd is not None:
            os.close(current_fd)
        raise
    except OSError as exc:
        if current_fd is not None:
            os.close(current_fd)
        raise MetadataError(f"{label} path cannot be opened safely") from exc


def _validate_input_path(path: Path) -> None:
    if path.name != "data.pkl":
        raise MetadataError("input filename must be data.pkl")
    if not path.is_absolute():
        raise MetadataError("input path must be absolute")


def _read_bounded(path: Path) -> bytes:
    _validate_input_path(path)
    parent_fd, leaf = _open_parent_dir(path, "input")
    try:
        candidate = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        os.close(parent_fd)
        raise MetadataError("input could not be inspected safely") from exc
    if not stat.S_ISREG(candidate.st_mode):
        os.close(parent_fd)
        raise MetadataError("input is not a regular file")
    if candidate.st_size <= 0 or candidate.st_size > MAX_INPUT_BYTES:
        os.close(parent_fd)
        raise MetadataError("input size is outside the security limit")
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
    fd: int | None = None
    try:
        fd = os.open(leaf, flags, dir_fd=parent_fd)
    except OSError as exc:
        os.close(parent_fd)
        raise MetadataError("input could not be opened") from exc
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or (before.st_dev, before.st_ino) != (candidate.st_dev, candidate.st_ino)
            or before.st_size <= 0
            or before.st_size > MAX_INPUT_BYTES
        ):
            raise MetadataError("input size is outside the security limit")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, remaining + 1)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
            if remaining < 0:
                break
        data = b"".join(chunks)
        after = os.fstat(fd)
        if len(data) != before.st_size or after.st_size != before.st_size:
            raise MetadataError("input changed or was not fully read")
        current = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino):
            raise MetadataError("input was replaced during read")
        return data
    finally:
        os.close(fd)
        os.close(parent_fd)


def _write_exclusive(path: Path, body: bytes) -> None:
    parent_fd, leaf = _open_parent_dir(path, "output parent")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd: int | None = None
    try:
        fd = os.open(leaf, flags, 0o600, dir_fd=parent_fd)
        offset = 0
        while offset < len(body):
            written = os.write(fd, body[offset:])
            if written <= 0:
                raise MetadataError("output write made no progress")
            offset += written
    except FileExistsError as exc:
        raise MetadataError("output appeared during exclusive write") from exc
    except OSError as exc:
        raise MetadataError("output could not be written") from exc
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent_fd)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_pickle", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if sys.platform != "linux":
        print("error: Linux remote worker is required", file=sys.stderr)
        return 2
    if os.environ.get(REMOTE_MARKER) != "1":
        print("error: explicit remote marker is required", file=sys.stderr)
        return 2
    try:
        data = _read_bounded(args.data_pickle)
        metadata = capture_bytes(data)
        body = (json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        if len(body) > MAX_OUTPUT_BYTES:
            raise MetadataError("serialized metadata exceeds the output byte budget")
        _write_exclusive(args.output, body)
    except (MetadataError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
