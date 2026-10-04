#!/usr/bin/env -S uv run --no-sync --project tools/parity --python 3.12 python
"""Run the official Kyutai streaming LM and capture an independent KV oracle.

The ``self-test`` mode is stdlib-only and never imports torch, Moshi, DSM, or
model files.  ``real`` is intentionally VAST-only: it authenticates the
pinned checkouts and checkpoint first, then hooks the *official* Moshi
``RingKVCache.complete`` method.  It records raw returned ring buffers before
any dtype conversion and keeps LMGen's scheduler cache in a separate field.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Callable

from contract import (
    AUDIO_CARD,
    CONTEXT,
    HF_REPOSITORY,
    HF_REVISION,
    MOSHI_REPOSITORY,
    MOSHI_REVISION,
    MODEL_TENSOR_MANIFEST_SHA256,
    N_Q,
    REAL_STATUS,
    STREAMING_SCHEMA,
    STREAMING_SCOPE,
    TEXT_CARD,
    boundary_packet,
    authenticate_composite_model,
    git_identity,
    require_execution_readiness,
    real_path,
    require_source_packet,
    sha256,
    unique,
    validate_input_packet,
)

def checkpoint_steps(context: int) -> frozenset[int]:
    return frozenset({0, 1, context - 1, context, context + 1})


def _decoder_helpers() -> Any:
    """Load only the existing stdlib/auth helpers, never its real path."""
    parity_root = str(Path(__file__).resolve().parents[1])
    if parity_root not in sys.path:
        sys.path.insert(0, parity_root)
    return importlib.import_module("kyutai_stt_decoder_dump_reference")


# Axes authenticated by the pinned LM contract in ``real`` below.
_MAIN_LAYER_COUNT = 48
_MAIN_DIM = 2048
_KV_HEADS = 32
_KV_HEAD_WIDTH = 64
_RAW_DTYPE = "torch.bfloat16"
_RAW_DTYPE_BYTES = 2
_CHECKPOINT_COUNT = 6  # five warm-up checkpoints plus the first post-reset step
_MAX_TENSOR_BYTES = 512 * 1024 * 1024
# Pinned source geometry is [batch=1, kv_heads=32, capacity=375,
# head_width=64].  The six complete K/V windows and 378 K/V delta calls total
# 1,033,371,648 raw bytes.  1.25 GiB leaves bounded room for manifest,
# scheduler/logit metadata and temporary per-call copies without pretending
# that a 512 MiB aggregate cap can contain the requested evidence.
_RAW_KV_BYTES = (
    (_CHECKPOINT_COUNT * _MAIN_LAYER_COUNT * 1 * _KV_HEADS * CONTEXT * _KV_HEAD_WIDTH * 2 * _RAW_DTYPE_BYTES)
    + ((CONTEXT + 2 + 1) * _MAIN_LAYER_COUNT * 1 * _KV_HEADS * 1 * _KV_HEAD_WIDTH * 2 * _RAW_DTYPE_BYTES)
)
_MAX_ARTIFACT_BYTES = (5 * 1024**3) // 4
_MAX_EVENTS = _MAIN_LAYER_COUNT * 4096
_MAX_LOGIT_ARTIFACT_BYTES = 8 * 1024 * 1024


def _logical_tensor_meta(tensor: Any) -> tuple[int, int, int, int]:
    numel = getattr(tensor, "numel", None)
    element_size = getattr(tensor, "element_size", None)
    storage_offset = getattr(tensor, "storage_offset", None)
    if not callable(numel) or not callable(element_size) or not callable(storage_offset):
        raise TypeError("tensor must expose numel, element_size, and storage_offset")
    logical_numel = int(numel())
    width = int(element_size())
    offset = int(storage_offset())
    if logical_numel < 0 or width <= 0 or offset < 0:
        raise ValueError("invalid tensor logical storage metadata")
    logical_bytes = logical_numel * width
    if logical_bytes > _MAX_TENSOR_BYTES:
        raise ValueError("tensor logical byte budget exceeded")
    return logical_numel, width, offset, logical_bytes


def _bounded_storage_slice(storage: Any, start: int, length: int) -> bytes:
    if length < 0 or length > _MAX_TENSOR_BYTES:
        raise ValueError("storage slice exceeds bounded byte budget")
    nbytes = getattr(storage, "nbytes", None)
    if callable(nbytes):
        storage_bytes = int(nbytes())
    else:
        try:
            storage_bytes = len(storage)
        except TypeError as exc:
            raise TypeError("untyped storage must expose nbytes or len") from exc
    if start < 0 or start + length > storage_bytes:
        raise ValueError("tensor logical window exceeds backing storage")
    try:
        view = memoryview(storage)
        if view.format != "B":
            view = view.cast("B")
        raw = view[start : start + length].tobytes()
        if len(raw) == length:
            return raw
    except (TypeError, ValueError):
        pass
    try:
        sliced = storage[start : start + length]
    except (TypeError, IndexError) as exc:
        raise TypeError("untyped storage does not support bounded byte slicing") from exc
    try:
        raw = bytes(sliced)
    except (TypeError, ValueError) as exc:
        raise TypeError("untyped storage slice is not byte-readable") from exc
    if len(raw) != length:
        raise ValueError("storage slice returned non-logical backing bytes")
    return raw


def _tensor_bytes(tensor: Any) -> bytes:
    """Copy only logical little-endian bytes without copying backing storage.

    The logical cap is checked before any detach/CPU/contiguous operation.
    Only ``UntypedStorage``-style bounded slicing is accepted; deprecated typed
    storage is deliberately rejected because its byte representation is not a
    trusted raw tensor contract.
    """
    if sys.byteorder != "little":
        raise RuntimeError("raw tensor evidence requires a little-endian host")
    logical_numel, width, _source_offset, logical_bytes = _logical_tensor_meta(tensor)
    cpu = tensor.detach().cpu().contiguous()
    compact_numel, compact_width, offset, compact_bytes = _logical_tensor_meta(cpu)
    if (compact_numel, compact_width, compact_bytes) != (logical_numel, width, logical_bytes):
        raise ValueError("CPU tensor logical metadata changed during copy")
    storage = getattr(cpu, "untyped_storage", None)
    if not callable(storage):
        raise TypeError("tensor does not expose supported untyped storage")
    raw_storage = storage()
    if raw_storage is None:
        raise TypeError("tensor untyped storage is unavailable")
    start = offset * width
    return _bounded_storage_slice(raw_storage, start, logical_bytes)


def _bounded_tensor_list(tensor: Any, *, label: str, expected_shape: tuple[int, ...] | None = None,
                         expected_dtype: str | None = None, max_numel: int = 4096) -> list[Any]:
    """Materialize only a validated, small metadata tensor."""
    metadata = _logical_tensor_meta(tensor)
    actual_shape = tuple(int(dimension) for dimension in getattr(tensor, "shape", ()))
    if expected_shape is not None and actual_shape != expected_shape:
        raise ValueError(f"{label} shape is not the authenticated contract")
    if expected_dtype is not None and str(getattr(tensor, "dtype", "")) != expected_dtype:
        raise ValueError(f"{label} dtype is not the authenticated contract")
    if metadata[0] > max_numel:
        raise ValueError(f"{label} metadata exceeds bounded list budget")
    return tensor.detach().cpu().tolist()


def _tensor_list(tensor: Any) -> list[Any]:
    return _bounded_tensor_list(tensor, label="bounded tensor metadata")


def _validate_end_offset(cache: Any) -> list[Any]:
    return _bounded_tensor_list(
        cache.end_offset,
        label="KV end_offset",
        expected_shape=(1,),
        expected_dtype="torch.int64",
        max_numel=1,
    )


def _environment_fingerprint() -> dict[str, Any]:
    """Capture host facts before importing or constructing the official LM."""
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "toolchain": sys.version,
        "torch_cpu_capability": "unresolved-before-import",
        "dtype": "unresolved-before-model",
    }


def _validate_logits_for_export(
    value: Any, *, reserved_bytes: int
) -> tuple[list[int], list[int], str, int]:
    """Validate bounded text logits before float32 conversion or device copy."""
    shape = tuple(int(dimension) for dimension in getattr(value, "shape", ()))
    source_shape = (1, 1, 1, TEXT_CARD)
    comparison_shape = (1, TEXT_CARD)
    if shape != source_shape:
        raise ValueError("text logits source shape is not the authenticated [1,1,1,4000] contract")
    numel = getattr(value, "numel", None)
    if not callable(numel) or int(numel()) != TEXT_CARD:
        raise ValueError("text logits cardinality is not the authenticated text card")
    isfinite = getattr(value, "isfinite", None)
    if not callable(isfinite):
        raise TypeError("text logits must expose an isfinite check")
    finite = isfinite()
    all_finite = getattr(finite, "all", None)
    if not callable(all_finite):
        raise TypeError("text logits finite check is malformed")
    finite_value = all_finite()
    item = getattr(finite_value, "item", None)
    if not callable(item) or not bool(item()):
        raise ValueError("text logits contain non-finite values")
    export_bytes = TEXT_CARD * 4
    if reserved_bytes + export_bytes > _MAX_LOGIT_ARTIFACT_BYTES:
        raise ValueError("text logits artifact budget exceeded")
    return list(source_shape), list(comparison_shape), str(getattr(value, "dtype", "")), export_bytes


def _trusted_layer_container_type(transformer_module: Any) -> type[Any]:
    """Obtain ModuleList identity from the authenticated official module."""
    # The pinned source constructs ``transformer.nn.ModuleList``.  Do not
    # accept a same-named convenience alias exported by another module.
    candidate = getattr(getattr(transformer_module, "nn", None), "ModuleList", None)
    if isinstance(candidate, type):
        return candidate
    raise RuntimeError("official transformer module exposes no trusted ModuleList type")


def _validate_tensor_geometry(
    tensor: Any, *, expected_shape: tuple[int, ...], label: str
) -> tuple[int, int, int, int]:
    """Validate raw KV geometry before any operation that can copy a tensor."""
    metadata = _logical_tensor_meta(tensor)
    dtype = str(getattr(tensor, "dtype", ""))
    actual_shape = tuple(int(dimension) for dimension in getattr(tensor, "shape", ()))
    expected_numel = 1
    for dimension in expected_shape:
        expected_numel *= dimension
    if (
        actual_shape != expected_shape
        or dtype != _RAW_DTYPE
        or metadata[0] != expected_numel
        or metadata[1] != _RAW_DTYPE_BYTES
    ):
        raise ValueError(f"{label} geometry/dtype is not the authenticated BF16 KV contract")
    return metadata


class RingCapture:
    """A narrow hook around the official RingKVCache, not a cache mirror."""

    def __init__(self, checkpoints: frozenset[int], *, max_artifact_bytes: int = _MAX_ARTIFACT_BYTES) -> None:
        self.original: Callable[..., Any] | None = None
        self.step = 0
        self.phase = "warmup"
        self.checkpoints = checkpoints
        self.layer_ids: dict[int, int] = {}
        self.layer_caches: list[Any] = []
        self.events: list[dict[str, Any]] = []
        self.files: dict[str, bytearray] = {}
        self.state_snapshots: list[dict[str, Any]] = []
        self._seen_snapshots: set[tuple[str, int, int]] = set()
        self._seen_events: set[tuple[str, int, int]] = set()
        self._reserved_bytes = 0
        if max_artifact_bytes <= 0 or max_artifact_bytes > _MAX_ARTIFACT_BYTES:
            raise ValueError("capture artifact budget is outside the authenticated bound")
        self.max_artifact_bytes = max_artifact_bytes

    def register_layers(self, lm: Any, trusted_layer_container_type: type[Any]) -> None:
        transformer_model = getattr(lm, "transformer", None)
        layers = getattr(transformer_model, "layers", None)
        if not isinstance(trusted_layer_container_type, type) or not isinstance(layers, trusted_layer_container_type):
            raise RuntimeError("official LM has no authenticated transformer layer container")
        try:
            layer_sequence = list(layers)
        except TypeError as exc:
            raise RuntimeError("official transformer layer container is not iterable") from exc
        ordered = []
        for layer in layer_sequence:
            attention = getattr(layer, "self_attn", None)
            state = getattr(attention, "_streaming_state", None)
            cache = getattr(state, "kv_cache", None)
            if cache is None:
                raise RuntimeError("official attention layer has no streaming kv_cache")
            ordered.append(cache)
        if len(ordered) != _MAIN_LAYER_COUNT or len({id(cache) for cache in ordered}) != _MAIN_LAYER_COUNT:
            raise RuntimeError("official main transformer must expose exactly 48 unique KV caches")
        self.layer_ids = {id(cache): layer for layer, cache in enumerate(ordered)}
        self.layer_caches = ordered

    def record_state_snapshot(self, phase: str, step: int) -> None:
        if phase not in {"initial", "after_reset"} or step != 0:
            raise ValueError("state snapshots are only initial/reset completion observations")
        ordered = getattr(self, "layer_caches", None)
        if not isinstance(ordered, list) or len(ordered) != _MAIN_LAYER_COUNT:
            raise RuntimeError("KV snapshot does not contain all 48 main-transformer layers")
        snapshot_keys = {(phase, step, layer) for layer in range(_MAIN_LAYER_COUNT)}
        if self._seen_snapshots.intersection(snapshot_keys):
            raise RuntimeError("duplicate phase/step/layer KV snapshot")
        for layer, cache in enumerate(ordered):
            snapshot_key = (phase, step, layer)
            if int(cache.capacity) != CONTEXT:
                raise RuntimeError("official KV cache capacity is not the authenticated 375")
            end_offset = _validate_end_offset(cache)
            if phase in {"initial", "after_reset"} and any(value != 0 for value in end_offset):
                raise RuntimeError(f"official {phase} KV end_offset is not zero")
            self.state_snapshots.append({
                "phase": phase,
                "step": step,
                "layer": layer,
                "end_offset": end_offset,
                "capacity": int(cache.capacity),
            })
            self._seen_snapshots.add(snapshot_key)

    def install(self, transformer: Any, lm: Any, trusted_layer_container_type: type[Any] | None = None) -> None:
        if self.original is not None:
            raise RuntimeError("RingKVCache hook already installed")
        if trusted_layer_container_type is None:
            trusted_layer_container_type = _trusted_layer_container_type(transformer)
        self.register_layers(lm, trusted_layer_container_type)
        original = transformer.RingKVCache.complete
        self.original = original
        owner = self

        def hooked(cache: Any, keys: Any, values: Any, exec_mask: Any) -> Any:
            layer = owner.layer_ids.get(id(cache))
            if layer is None:
                raise RuntimeError("unregistered non-main-transformer KV cache observed")
            if owner.phase not in {"warmup", "after_reset"} or owner.step < 0:
                raise ValueError("unsupported streaming phase or step")
            if owner.phase == "warmup" and owner.step > CONTEXT + 1:
                raise ValueError("warm-up capture step exceeds authenticated boundary")
            if owner.phase == "after_reset" and owner.step != 0:
                raise ValueError("post-reset capture has exactly one step")
            if int(cache.capacity) != CONTEXT:
                raise RuntimeError("official KV cache capacity is not the authenticated 375")
            event_key = (owner.phase, owner.step, layer)
            if event_key in owner._seen_events:
                raise RuntimeError("duplicate phase/step/layer KV event")
            # Validate and copy the logical deltas before the official method:
            # its implementation may mutate or alias the input tensors.
            key_meta = _validate_tensor_geometry(keys, expected_shape=(1, _KV_HEADS, 1, _KV_HEAD_WIDTH), label="delta keys")
            value_meta = _validate_tensor_geometry(values, expected_shape=(1, _KV_HEADS, 1, _KV_HEAD_WIDTH), label="delta values")
            snapshot_bytes = 0
            if owner.step in owner.checkpoints:
                snapshot_bytes = _KV_HEADS * CONTEXT * _KV_HEAD_WIDTH * _RAW_DTYPE_BYTES * 2
            reservation = key_meta[3] + value_meta[3] + snapshot_bytes + 1024
            if owner._reserved_bytes + reservation > owner.max_artifact_bytes:
                raise RuntimeError("KV capture artifact budget exceeded before official call")
            if len(owner.events) >= _MAX_EVENTS:
                raise RuntimeError("KV event budget exceeded before official call")
            key_bytes = _tensor_bytes(keys)
            value_bytes = _tensor_bytes(values)
            if (len(key_bytes), len(value_bytes)) != (key_meta[3], value_meta[3]):
                raise RuntimeError("delta tensor bytes changed during bounded copy")
            owner._reserved_bytes += reservation
            owner._seen_events.add(event_key)
            result = original(cache, keys, values, exec_mask)
            snapshot_data: tuple[Any, Any, list[Any], bytes, bytes] | None = None
            if owner.step in owner.checkpoints:
                raw_keys = result.keys
                raw_values = result.values
                raw_positions = result.positions
                _validate_tensor_geometry(raw_keys, expected_shape=(1, _KV_HEADS, CONTEXT, _KV_HEAD_WIDTH), label="snapshot keys")
                _validate_tensor_geometry(raw_values, expected_shape=(1, _KV_HEADS, CONTEXT, _KV_HEAD_WIDTH), label="snapshot values")
                positions = _bounded_tensor_list(
                    raw_positions,
                    label="snapshot positions",
                    expected_shape=(1, CONTEXT),
                    expected_dtype="torch.int64",
                    max_numel=CONTEXT,
                )
                raw_key_bytes = _tensor_bytes(raw_keys)
                raw_value_bytes = _tensor_bytes(raw_values)
                if len(raw_key_bytes) + len(raw_value_bytes) > snapshot_bytes:
                    raise RuntimeError("snapshot exceeded its reserved byte budget")
                snapshot_data = (raw_keys, raw_values, positions, raw_key_bytes, raw_value_bytes)
            delta_prefix = f"kv/{owner.phase}-layer-{layer:02d}"
            delta_key_name = f"{delta_prefix}-delta-keys.bin"
            delta_value_name = f"{delta_prefix}-delta-values.bin"
            delta_keys = owner.files.setdefault(delta_key_name, bytearray())
            delta_values = owner.files.setdefault(delta_value_name, bytearray())
            key_offset = len(delta_keys)
            value_offset = len(delta_values)
            delta_keys.extend(key_bytes)
            delta_values.extend(value_bytes)
            event: dict[str, Any] = {
                "phase": owner.phase,
                "step": owner.step,
                "layer": layer,
                "delta_shape": list(keys.shape),
                "delta_dtype": str(keys.dtype),
                "delta_keys": delta_key_name,
                "delta_keys_offset": key_offset,
                "delta_keys_bytes": len(key_bytes),
                "delta_values": delta_value_name,
                "delta_values_offset": value_offset,
                "delta_values_bytes": len(value_bytes),
                "end_offset": _validate_end_offset(cache),
                "capacity": int(cache.capacity),
            }
            if snapshot_data is not None:
                # _tensor_bytes returns an independent bounded bytes copy
                # immediately; result.keys/values may alias mutable storage.
                raw_keys, raw_values, positions, raw_key_bytes, raw_value_bytes = snapshot_data
                prefix = f"kv/{owner.phase}-step-{owner.step:04d}-layer-{layer:02d}"
                key_name = f"{prefix}-keys.bin"
                value_name = f"{prefix}-values.bin"
                owner.files[key_name] = bytearray(raw_key_bytes)
                owner.files[value_name] = bytearray(raw_value_bytes)
                event.update(
                    {
                        "shape": list(raw_keys.shape),
                        "dtype": str(raw_keys.dtype),
                        "positions": positions,
                        "keys": key_name,
                        "values": value_name,
                        "valid_positions": sorted(
                            position
                            for row in positions
                            for position in row
                            if position >= 0
                        ),
                    }
                )
            owner.events.append(event)
            return result

        transformer.RingKVCache.complete = hooked

    def require_complete_coverage(self) -> None:
        """Reject partial capture before any output can be marked reference-ready."""
        expected_events = {
            (phase, step, layer)
            for phase, steps in (("warmup", range(CONTEXT + 2)), ("after_reset", (0,)))
            for step in steps
            for layer in range(_MAIN_LAYER_COUNT)
        }
        if self._seen_events != expected_events or len(self.events) != len(expected_events):
            raise RuntimeError("KV capture is missing a phase/step/layer event")
        expected_snapshots = {
            (phase, 0, layer)
            for phase in ("initial", "after_reset")
            for layer in range(_MAIN_LAYER_COUNT)
        }
        if self._seen_snapshots != expected_snapshots or len(self.state_snapshots) != len(expected_snapshots):
            raise RuntimeError("KV capture is missing an initial/reset snapshot")
        for event in self.events:
            if event["step"] in self.checkpoints or (
                event["phase"] == "after_reset" and event["step"] == 0
            ):
                for key in ("keys", "values"):
                    name = event.get(key)
                    if not isinstance(name, str) or name not in self.files:
                        raise RuntimeError("KV checkpoint artifact is missing")

    def uninstall(self, transformer: Any) -> None:
        if self.original is None:
            return
        transformer.RingKVCache.complete = self.original
        self.original = None


def _artifact_record(files: dict[str, bytes]) -> dict[str, dict[str, Any]]:
    return {
        name: {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
        for name, body in sorted(files.items())
    }


def _write_output(out: Path, files: dict[str, bytes], manifest: dict[str, Any]) -> None:
    if out.exists() or out.is_symlink() or out.name in {"", ".", ".."}:
        raise ValueError("reference output must be absent")
    parent = real_path(out.parent, "reference output parent", directory=True)
    manifest_body = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    total_bytes = sum(len(body) for body in files.values()) + len(manifest_body)
    if total_bytes > _MAX_ARTIFACT_BYTES or len(files) > 4096:
        raise ValueError("reference artifact budget exceeded")
    names = set(files)
    if "manifest.json" in names:
        raise ValueError("manifest.json is reserved")
    path_parts: dict[str, tuple[str, ...]] = {}
    for name in files:
        candidate = Path(name)
        if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
            raise ValueError("artifact name is not a safe relative path")
        path_parts[name] = tuple(candidate.parts)
    for name, parts in path_parts.items():
        if any(parts[:index] in {tuple(Path(other).parts) for other in names} for index in range(1, len(parts))):
            raise ValueError(f"artifact path collides with a file: {name}")

    directory_parts = sorted(
        {parts[:index] for parts in path_parts.values() for index in range(1, len(parts))},
        key=lambda parts: (len(parts), parts),
    )
    parent_fd = -1
    output_fd = -1
    directory_fds: dict[tuple[str, ...], int] = {}
    try:
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | nofollow)
        # The output directory is created exactly once.  If anything fails
        # below, the exclusive partial directory remains as an invalid,
        # inspectable artifact; unknown files are never recursively removed.
        os.mkdir(out.name, 0o700, dir_fd=parent_fd)
        output_fd = os.open(out.name, os.O_RDONLY | os.O_DIRECTORY | nofollow, dir_fd=parent_fd)
        directory_fds[()] = os.dup(output_fd)
        for parts in directory_parts:
            parent_parts = parts[:-1]
            parent_dir_fd = directory_fds[parent_parts]
            os.mkdir(parts[-1], 0o700, dir_fd=parent_dir_fd)
            directory_fds[parts] = os.open(
                parts[-1], os.O_RDONLY | os.O_DIRECTORY | nofollow, dir_fd=parent_dir_fd
            )
        for name, body in files.items():
            parts = path_parts[name]
            directory_fd = directory_fds[parts[:-1]]
            fd = os.open(
                parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow,
                0o600,
                dir_fd=directory_fd,
            )
            try:
                view = memoryview(body)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise OSError("short artifact write")
                    view = view[written:]
                os.fsync(fd)
            except Exception:
                try:
                    os.close(fd)
                except OSError:
                    pass
                raise
            else:
                os.close(fd)
        manifest_fd = os.open(
            "manifest.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow,
            0o600,
            dir_fd=output_fd,
        )
        try:
            view = memoryview(manifest_body)
            while view:
                written = os.write(manifest_fd, view)
                if written <= 0:
                    raise OSError("short manifest write")
                view = view[written:]
            os.fsync(manifest_fd)
        finally:
            os.close(manifest_fd)
        for fd in sorted(directory_fds.values(), reverse=True):
            os.fsync(fd)
        os.fsync(output_fd)
        os.fsync(parent_fd)
    finally:
        for fd in directory_fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        if output_fd >= 0:
            os.close(output_fd)
        if parent_fd >= 0:
            os.close(parent_fd)


def _record_scheduler(lm_gen: Any, collector: dict[str, Any], phase: str, step: int) -> None:
    state = lm_gen._streaming_state
    if state is None:
        raise RuntimeError("LMGen scheduler state disappeared during official step")
    collector.setdefault("scheduler", []).append(
        {
            "phase": phase,
            "step": step,
            "offsets": _tensor_list(state.offsets),
            "cache_shape": list(state.cache.shape),
            "cache_dtype": str(state.cache.dtype),
            "cache": _tensor_list(state.cache),
        }
    )


def self_test() -> None:
    packet = boundary_packet()
    validate_input_packet(packet)
    assert checkpoint_steps(CONTEXT) == frozenset({0, 1, 374, 375, 376})
    assert STREAMING_SCHEMA.endswith("-v1")
    assert "no Mimi PCM" in STREAMING_SCOPE
    try:
        validate_input_packet(dict(packet, kind="official_pcm"))
    except ValueError:
        pass
    else:
        raise AssertionError("unapproved input kind accepted")
    try:
        validate_input_packet(dict(packet, audio_codes=packet["audio_codes"][:-1]))
    except ValueError:
        pass
    else:
        raise AssertionError("short boundary packet accepted")
    print("kyutai STT independent streaming reference self-test PASS")


def real(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise ValueError("refusing to overwrite an existing real fixture")
    checkout = Path.cwd()
    expected_head = args.expected_head
    readiness = require_execution_readiness(
        args.approval_evidence,
        expected_head=expected_head,
        expected_approval_sha256=args.approval_sha256,
        checkout=checkout,
        dependency_closure_sha256=args.dependency_closure_sha256,
    )
    if args.source_packet is None:
        raise ValueError("real reference capture requires an authenticated source packet")
    decoder = _decoder_helpers()
    decoder.require_clean_head(expected_head)
    source_packet = require_source_packet(args.source_packet)
    dsm_identity = git_identity(args.dsm_source, decoder.DSM_REPOSITORY, decoder.DSM_REVISION)
    moshi_identity = git_identity(args.moshi_source, MOSHI_REPOSITORY, MOSHI_REVISION)
    source_contract = decoder.authenticate_streaming_source_contract(
        args.dsm_source, args.moshi_source
    )
    model_record = authenticate_composite_model(args.model)
    decoder_record = decoder.authenticate_model(args.model / decoder.MODEL_NAME, args.model / "config.json")
    if decoder_record.get("tensor_manifest_sha256") != MODEL_TENSOR_MANIFEST_SHA256:
        raise ValueError("authenticated model tensor manifest mismatch")
    model_record["tensor_manifest_sha256"] = decoder_record["tensor_manifest_sha256"]
    environment = _environment_fingerprint()

    sys.path.insert(0, str(args.moshi_source))
    sys.path.insert(0, str(args.moshi_source / "moshi"))
    sys.path.insert(0, str(args.dsm_source))
    try:
        import torch  # type: ignore
        from moshi.models import LMGen, loaders  # type: ignore
        from moshi.modules import transformer  # type: ignore
    except ImportError as error:
        raise SystemExit(f"official Moshi implementation is required: {error}") from error
    environment["torch_cpu_capability"] = getattr(
        torch.backends.cpu, "get_cpu_capability", lambda: "unknown"
    )()
    environment["dtype"] = str(torch.bfloat16)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise ValueError("locked single-thread execution could not be established")
    torch.use_deterministic_algorithms(True)
    if not torch.are_deterministic_algorithms_enabled():
        raise ValueError("deterministic torch algorithms could not be enabled")

    input_packet = boundary_packet()
    validate_input_packet(input_packet)
    info = loaders.CheckpointInfo.from_hf_repo(
        HF_REPOSITORY,
        moshi_weights=args.model / decoder.MODEL_NAME,
        mimi_weights=args.model / decoder.MIMI_NAME,
        tokenizer=args.model / decoder.TOKENIZER_NAME,
        config_path=args.model / "config.json",
        revision=HF_REVISION,
    )
    # Keep checkpoint dtype visible.  The consumer may convert BF16 to f32,
    # but the official cache evidence must preserve this raw dtype first.
    lm = info.get_moshi(device="cpu", dtype=torch.bfloat16)
    if (lm.dep_q, lm.n_q, lm.text_card, lm.dim, lm.context) != (0, N_Q, TEXT_CARD, 2048, CONTEXT):
        raise ValueError("official streaming axes are not the authenticated contract")

    files: dict[str, bytes] = {"input.json": (json.dumps(input_packet, sort_keys=True, indent=2) + "\n").encode()}
    capture = RingCapture(checkpoint_steps(CONTEXT))
    scheduler: dict[str, Any] = {}
    logits: list[dict[str, Any]] = []
    logits_reserved_bytes = 0
    tokens: list[int] = []
    token_events: list[dict[str, Any]] = []
    current_phase = "warmup"
    current_step = 0

    def on_logits(value: Any) -> None:
        nonlocal logits_reserved_bytes
        if current_step in checkpoint_steps(CONTEXT):
            source_shape, comparison_shape, source_dtype, export_bytes = _validate_logits_for_export(
                value, reserved_bytes=logits_reserved_bytes
            )
            body = value.detach().cpu().float().reshape(1, TEXT_CARD).contiguous().numpy().tobytes()
            if len(body) != export_bytes:
                raise ValueError("text logits float32 export size mismatch")
            logits_reserved_bytes += export_bytes
            name = f"text/logits-{current_phase}-step-{current_step:04d}.f32"
            logits.append(
                {
                    "phase": current_phase,
                    "step": current_step,
                    "shape": comparison_shape,
                    "source_shape": source_shape,
                    "dtype": "torch.float32",
                    "source_dtype": source_dtype,
                    "conversion": "official logits converted to contiguous torch.float32 for comparison export",
                    "artifact": name,
                    "bytes": len(body),
                }
            )
            files[name] = body

    def on_text(value: Any) -> None:
        items = [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
        tokens.extend(items)
        token_events.append({"phase": current_phase, "step": current_step, "shape": list(value.shape), "dtype": str(value.dtype), "values": items})

    lm_gen = LMGen(
        lm,
        use_sampling=False,
        temp=0.0,
        temp_text=0.0,
        top_k_text=50,
        check=True,
        on_text_hook=on_text,
        on_text_logits_hook=on_logits,
    )
    try:
        with torch.no_grad(), lm_gen.streaming(1):
            if lm_gen._streaming_state is None:
                raise ValueError("official LMGen did not initialize streaming state")
            capture.install(transformer, lm)
            capture.record_state_snapshot("initial", 0)
            for current_step, frame in enumerate(input_packet["audio_codes"]):
                capture.phase = current_phase
                capture.step = current_step
                frame_tensor = torch.tensor(frame, dtype=torch.long, device=lm.device).view(1, N_Q, 1)
                output = lm_gen.step(frame_tensor)
                if output is None:
                    raise ValueError("official LMGen returned no streaming frame")
                if current_step in checkpoint_steps(CONTEXT):
                    _record_scheduler(lm_gen, scheduler, current_phase, current_step)
            lm_gen.reset_streaming(torch.ones(1, dtype=torch.bool, device=lm.device))
            current_phase = "after_reset"
            current_step = 0
            if lm_gen._streaming_state is None:
                raise ValueError("official LMGen reset removed streaming state")
            capture.record_state_snapshot(current_phase, current_step)
            capture.phase = current_phase
            capture.step = current_step
            frame_tensor = torch.tensor(input_packet["audio_codes"][0], dtype=torch.long, device=lm.device).view(1, N_Q, 1)
            if lm_gen.step(frame_tensor) is None:
                raise ValueError("official LMGen returned no post-reset frame")
            _record_scheduler(lm_gen, scheduler, current_phase, current_step)
    finally:
        capture.uninstall(transformer)

    capture.require_complete_coverage()
    files.update({name: bytes(body) for name, body in capture.files.items()})
    files["text_tokens.i64"] = b"".join(int(token).to_bytes(8, "little", signed=False) for token in tokens)
    if len(capture.events) > 48 * 4096 or len(token_events) > 4096:
        raise ValueError("streaming event budget exceeded")
    manifest = {
        "format": STREAMING_SCHEMA,
        "status": REAL_STATUS,
        "scope": STREAMING_SCOPE,
        "claim_boundary": "official streaming main-LM execution only; this is not PCM/ASR parity",
        "expected_head": expected_head,
        "dependency_closure_sha256": readiness["dependency_closure_sha256"],
        "approval_sha256": args.approval_sha256,
        "approval_decision": readiness["approval"]["decision"],
        "approval_scope": readiness["approval"]["scope"],
        "source_packet": source_packet,
        "model": model_record,
        "sources": {"dsm": dsm_identity, "moshi": moshi_identity, "contract": source_contract},
        "input": input_packet,
        "execution": {
            "implementation": "official Moshi LMGen.step with RingKVCache.complete hook",
            "device": "cpu",
            "weights_dtype": str(lm.dtype),
            "kv_dtype_recorded_before_conversion": True,
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "publication": "NO_UPLOAD",
            "environment": environment,
        },
        "ring_cache": {
            "events": capture.events,
            "state_snapshots": capture.state_snapshots,
            "reset_semantics": "official reset updates end_offset; reset exposes no positions API, so no positions are synthesized; stale physical backing slots are retained and are not compared",
            "comparison": "discard official positions=-1, reorder valid rows by absolute position, then compare native chronological view",
        },
        "scheduler": scheduler,
        "text": {"tokens": token_events, "logits": logits},
        "artifacts": _artifact_record(files),
    }
    _write_output(args.out, files, manifest)
    print(f"reference written: {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("self-test")
    real_parser = sub.add_parser("real")
    real_parser.add_argument("--model", type=Path, required=True)
    real_parser.add_argument("--dsm-source", type=Path, required=True)
    real_parser.add_argument("--moshi-source", type=Path, required=True)
    real_parser.add_argument("--source-packet", type=Path, required=True)
    real_parser.add_argument("--out", type=Path, required=True)
    real_parser.add_argument("--expected-head", required=True)
    real_parser.add_argument("--approval-evidence", type=Path, required=True)
    real_parser.add_argument("--approval-sha256", required=True)
    real_parser.add_argument("--dependency-closure-sha256", required=True)
    args = parser.parse_args()
    if args.mode == "self-test":
        self_test()
    else:
        real(args)


if __name__ == "__main__":
    main()
