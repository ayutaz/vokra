#!/usr/bin/env -S uv run --frozen --project tools/parity/vibevoice_realtime_0_5b_reference python
"""Export the fixed official Realtime Carter voice preset for native inspection.

This is deliberately an inspection-only bridge.  It authenticates the pinned
source checkout and the fixed Git blob before importing torch, then performs a
weights-only load with the two upstream safe globals used by Microsoft's demo.
It does not construct a model, call a forward method, decode audio, download,
or upload anything.

The resulting safetensors file contains only the four cached output branches
(``lm``, ``tts_lm``, ``neg_lm``, and ``neg_tts_lm``).  Framework KV tensors are
copied from ``[batch, kv_head, position, head_dim]`` into the native bridge's
``[position, kv_head, head_dim]`` layout.  Hidden-row and cache-position
lengths are recorded independently; no equality is inferred.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


SOURCE_REVISION = "94da20d98b2fa7688e9cbfaf7692ddb4954f7600"
MODEL_REVISION = "6bce5f06044837fe6d2c5d7a71a84f0416bd57e4"
PRESET_RELATIVE_PATH = Path("demo/voices/streaming_model/en-Carter_man.pt")
PRESET_BYTES = 4_256_002
PRESET_GIT_BLOB_SHA1 = "1d795ef667e6641eecb8b22452bb853b089bfdbe"
PRESET_PAYLOAD_SHA256 = "a7bfdf1cd4939c22469bcfc6f427ae9c4467b3df46c2c14303a39c294cfc6897"
HIDDEN = 896
KV_HEADS = 2
HEAD_DIM = 64
MAX_POSITIONS = 8_192
BRANCHES = ("lm", "tts_lm", "neg_lm", "neg_tts_lm")
LAYER_COUNTS = {"lm": 4, "neg_lm": 4, "tts_lm": 20, "neg_tts_lm": 20}
FORMAT = "vokra-vibevoice-realtime-0.5b-preset-v1"
CLASSIFICATION = "INSPECTION_ONLY"
PUBLICATION = "NO_UPLOAD"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _platform_guard(system: str, machine: str) -> None:
    if system != "Linux" or machine.lower() not in {"x86_64", "amd64"}:
        raise RuntimeError(
            "preset export is VAST/Linux x86_64 only; refusing torch/model imports "
            f"on {system}/{machine}"
        )


def _source_identity(source_root: Path, preset_override: Path | None) -> tuple[dict[str, Any], Path, bytes]:
    root = source_root.resolve()
    if not root.is_dir() or source_root.is_symlink():
        raise RuntimeError(f"official source root must be a regular directory: {source_root}")
    try:
        head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        origin = subprocess.check_output(
            ["git", "-C", str(root), "remote", "get-url", "origin"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("official source root must be a git checkout") from error
    if head != SOURCE_REVISION:
        raise RuntimeError(f"official source HEAD mismatch: {head} != {SOURCE_REVISION}")
    if status:
        raise RuntimeError("official source checkout is dirty; refusing unpinned source files")
    if origin not in {
        "https://github.com/microsoft/VibeVoice.git",
        "git@github.com:microsoft/VibeVoice.git",
        "ssh://git@github.com/microsoft/VibeVoice.git",
    }:
        raise RuntimeError(f"official source origin is not microsoft/VibeVoice: {origin!r}")

    raw_preset = preset_override or (root / PRESET_RELATIVE_PATH)
    if raw_preset.is_symlink():
        raise RuntimeError(f"fixed Carter preset must not be a symlink: {raw_preset}")
    preset = raw_preset.resolve()
    if not preset.is_file():
        raise RuntimeError(f"fixed Carter preset is not a regular file: {preset}")
    relative = PRESET_RELATIVE_PATH
    if preset.name != PRESET_RELATIVE_PATH.name:
        raise RuntimeError(f"unexpected preset filename: {preset.name}")
    size = preset.stat().st_size
    if size != PRESET_BYTES:
        raise RuntimeError(f"fixed Carter preset size {size} != authenticated {PRESET_BYTES}")
    git_blob_path = str(PRESET_RELATIVE_PATH) if preset_override is None else str(preset)
    try:
        blob = subprocess.check_output(
            ["git", "-C", str(root), "hash-object", "--", git_blob_path],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("could not authenticate the fixed Carter Git blob") from error
    if blob != PRESET_GIT_BLOB_SHA1:
        raise RuntimeError(f"fixed Carter Git blob mismatch: {blob} != {PRESET_GIT_BLOB_SHA1}")
    preset_bytes = preset.read_bytes()
    if len(preset_bytes) != PRESET_BYTES:
        raise RuntimeError(f"fixed Carter preset read size {len(preset_bytes)} != authenticated {PRESET_BYTES}")
    payload_sha256 = _sha256(preset_bytes)
    if payload_sha256 != PRESET_PAYLOAD_SHA256:
        raise RuntimeError(
            f"fixed Carter payload SHA-256 mismatch: {payload_sha256} != {PRESET_PAYLOAD_SHA256}"
        )
    return {
        "repository": "microsoft/VibeVoice",
        "revision": head,
        "relative_path": str(relative),
        "bytes": size,
        "git_blob_sha1": blob,
        "payload_sha256": payload_sha256,
        "model_revision": MODEL_REVISION,
        "origin": origin,
    }, preset, preset_bytes


def _require_tensor(value: Any, label: str, torch: Any) -> Any:
    if not torch.is_tensor(value):
        raise RuntimeError(f"{label} is not a torch tensor")
    if value.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise RuntimeError(f"{label} has unsupported dtype {value.dtype}; expected BF16/F16/F32")
    return value


def _as_f32(value: Any, label: str, torch: Any) -> tuple[Any, str]:
    value = _require_tensor(value, label, torch)
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"{label} contains non-finite values")
    source_dtype = str(value.dtype).removeprefix("torch.").upper()
    converted = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(converted).all().item()):
        raise RuntimeError(f"{label} became non-finite after F32 conversion")
    return converted, source_dtype


def _tensor_bytes(value: Any) -> bytes:
    # The tensors are CPU, contiguous, and F32 by construction.  This avoids
    # inventing a second serialization format for payload hashes.
    return value.numpy().tobytes(order="C")


def _cache_layers(
    cache: Any, branch: str, torch: Any, DynamicCache: Any
) -> tuple[list[tuple[Any, Any]], int, list[str]]:
    if not isinstance(cache, DynamicCache) or type(cache) is not DynamicCache:
        raise RuntimeError(
            f"{branch}.past_key_values is {type(cache).__module__}.{type(cache).__qualname__}, not the exact official DynamicCache; "
            "historical cache reconstruction is refused"
        )
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if not isinstance(key_cache, (list, tuple)) or not isinstance(value_cache, (list, tuple)):
        raise RuntimeError(
            f"{branch}.past_key_values DynamicCache lacks explicit key_cache/value_cache lists; "
            "no compatibility shim or guessed reconstruction is permitted"
        )
    expected_layers = LAYER_COUNTS[branch]
    if len(key_cache) != expected_layers or len(value_cache) != expected_layers:
        raise RuntimeError(
            f"{branch} DynamicCache layer counts {len(key_cache)}/{len(value_cache)} "
            f"!= authenticated {expected_layers}"
        )
    converted: list[tuple[Any, Any]] = []
    positions: list[int] = []
    source_dtypes: list[str] = []
    for index, (key, value) in enumerate(zip(key_cache, value_cache, strict=True)):
        key = _require_tensor(key, f"{branch}.cache[{index}].key", torch)
        value = _require_tensor(value, f"{branch}.cache[{index}].value", torch)
        if key.ndim != 4 or tuple(key.shape[:2]) != (1, KV_HEADS) or key.shape[3] != HEAD_DIM:
            raise RuntimeError(
                f"{branch}.cache[{index}].key shape {tuple(key.shape)} != [1,2,position,64]"
            )
        if tuple(value.shape) != tuple(key.shape):
            raise RuntimeError(
                f"{branch}.cache[{index}] key/value shapes differ: {tuple(key.shape)} vs {tuple(value.shape)}"
            )
        position = int(key.shape[2])
        if position > MAX_POSITIONS:
            raise RuntimeError(f"{branch}.cache[{index}] position {position} > {MAX_POSITIONS}")
        key_f32, key_dtype = _as_f32(key, f"{branch}.cache[{index}].key", torch)
        value_f32, value_dtype = _as_f32(value, f"{branch}.cache[{index}].value", torch)
        # Explicitly transpose [batch, kv-head, position, dim] to the native
        # [position, kv-head, dim] row-major layout.  No axis is squeezed.
        key_native = key_f32.permute(0, 2, 1, 3).reshape(position, KV_HEADS, HEAD_DIM).contiguous()
        value_native = value_f32.permute(0, 2, 1, 3).reshape(position, KV_HEADS, HEAD_DIM).contiguous()
        converted.append((key_native, value_native))
        positions.append(position)
        source_dtypes.extend((key_dtype, value_dtype))
    if len(set(positions)) != 1:
        raise RuntimeError(f"{branch} cache layers have independent positions {positions}; refusing to guess")
    return converted, positions[0], source_dtypes


def _extract_outputs(loaded: Any, torch: Any, BaseModelOutputWithPast: Any, DynamicCache: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(loaded, dict) or set(loaded) != set(BRANCHES):
        raise RuntimeError(f"preset top-level keys must be exactly {list(BRANCHES)}")
    tensors: dict[str, Any] = {}
    bindings: dict[str, Any] = {}
    for branch in BRANCHES:
        output = loaded[branch]
        if not isinstance(output, BaseModelOutputWithPast) or type(output) is not BaseModelOutputWithPast:
            raise RuntimeError(f"{branch} is not the exact safe BaseModelOutputWithPast output type")
        hidden = output.last_hidden_state
        if hidden is None or hidden.ndim != 3 or tuple(hidden.shape[:1]) != (1,) or hidden.shape[2] != HIDDEN:
            shape = None if hidden is None else tuple(hidden.shape)
            raise RuntimeError(f"{branch}.last_hidden_state shape {shape} != [1,hidden_rows,896]")
        hidden_rows = int(hidden.shape[1])
        if hidden_rows == 0 or hidden_rows > MAX_POSITIONS:
            raise RuntimeError(f"{branch} hidden rows {hidden_rows} > {MAX_POSITIONS}")
        hidden_f32, hidden_dtype = _as_f32(hidden, f"{branch}.last_hidden_state", torch)
        # Retain the authenticated batch axis in the serialized schema.  The
        # Rust bridge flattens only after checking [1, hidden_rows, 896], so a
        # malformed batch cannot be silently squeezed into a valid row matrix.
        tensors[f"{branch}.hidden"] = hidden_f32.contiguous()
        layers, cache_position, cache_dtypes = _cache_layers(
            output.past_key_values, branch, torch, DynamicCache
        )
        for layer_index, (key, value) in enumerate(layers):
            tensors[f"{branch}.cache.{layer_index}.key"] = key
            tensors[f"{branch}.cache.{layer_index}.value"] = value
        bindings[branch] = {
            "hidden_rows": hidden_rows,
            "cache_position": cache_position,
            "layer_count": len(layers),
            "source_dtypes": {"hidden": hidden_dtype, "cache": sorted(set(cache_dtypes))},
            "source_layout": "[batch,kv_head,position,head_dim]",
            "native_layout": "[position,kv_head,head_dim]",
        }
    return tensors, bindings


def _lock_identity() -> dict[str, str]:
    lock_path = Path(__file__).with_name("uv.lock")
    if not lock_path.is_file() or lock_path.is_symlink():
        raise RuntimeError(f"reference lock is missing or symlinked: {lock_path}")
    return {"relative_path": "tools/parity/vibevoice_realtime_0_5b_reference/uv.lock", "sha256": _sha256_file(lock_path)}


def _export(args: argparse.Namespace) -> int:
    _platform_guard(platform.system(), platform.machine())
    if os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1":
        raise RuntimeError("preset export requires VOKRA_PUBLISH_ON_VAST=1 on the disposable VAST worker")
    source, preset, preset_bytes = _source_identity(Path(args.source_root), Path(args.preset_path) if args.preset_path else None)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise RuntimeError(f"output directory must be absent; refusing to overwrite {output_dir}")

    # These imports intentionally occur only after Linux/x86_64 and source/blob
    # authentication.  The self-test path above never imports either package.
    import torch
    from safetensors.torch import save_file
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_outputs import BaseModelOutputWithPast

    with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
        loaded = torch.load(io.BytesIO(preset_bytes), map_location="cpu", weights_only=True)
    tensors, bindings = _extract_outputs(loaded, torch, BaseModelOutputWithPast, DynamicCache)

    output_dir.mkdir(parents=True)
    safetensors_path = output_dir / "cache.safetensors"
    save_file(tensors, str(safetensors_path), metadata={"format": FORMAT})
    tensor_manifest: dict[str, Any] = {}
    for name in sorted(tensors):
        value = tensors[name]
        tensor_manifest[name] = {
            "dtype": "F32",
            "shape": list(value.shape),
            "sha256": _sha256(_tensor_bytes(value)),
        }
    manifest = {
        "format": FORMAT,
        "classification": CLASSIFICATION,
        "publication": PUBLICATION,
        "voice_consent": "UNPROVEN",
        "execution": "NO_MODEL_FORWARD_NO_AUDIO",
        "source": source,
        "preset": {"id": "en-Carter_man", "rights": "UNPROVEN"},
        "reference_lock": _lock_identity(),
        "branches": list(BRANCHES),
        "branch_bindings": bindings,
        "tensor_layout_contract": {
            "hidden": "[1,hidden_rows,896] -> [hidden_rows,896]",
            "framework_cache": "[1,2,position,64]",
            "native_cache": "[position,2,64]",
            "max_positions": MAX_POSITIONS,
        },
        "tensors": tensor_manifest,
        "output": {
            "relative_path": "cache.safetensors",
            "sha256": _sha256_file(safetensors_path),
        },
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (output_dir / "manifest.json").write_bytes(manifest_bytes)
    print(json.dumps({"output_dir": str(output_dir), "manifest_sha256": _sha256(manifest_bytes), "tensor_count": len(tensors)}, sort_keys=True))
    return 0


def _self_test() -> int:
    """Run stdlib-only schema/layout checks without importing torch or models."""

    def transpose(values: list[int], positions: int) -> list[int]:
        # Synthetic [batch=1, kv=2, position, dim=2] source.
        return [values[k * positions * 2 + p * 2 + d] for p in range(positions) for k in range(2) for d in range(2)]

    assert transpose(list(range(8)), 2) == [0, 1, 4, 5, 2, 3, 6, 7]
    assert set(BRANCHES) == {"lm", "tts_lm", "neg_lm", "neg_tts_lm"}
    assert LAYER_COUNTS["lm"] == LAYER_COUNTS["neg_lm"] == 4
    assert LAYER_COUNTS["tts_lm"] == LAYER_COUNTS["neg_tts_lm"] == 20
    for bad in (("Darwin", "arm64"), ("Linux", "aarch64"), ("Windows", "x86_64")):
        try:
            _platform_guard(*bad)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"platform guard accepted {bad}")
    print("preset exporter self-test: PASS (stdlib-only; no torch/model/cache load)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", help="clean pinned VibeVoice checkout (VAST only)")
    parser.add_argument(
        "--preset-path",
        help="optional external fixed en-Carter_man.pt path; still checked against the authenticated Git blob",
    )
    parser.add_argument("--output-dir", help="new absent directory for cache.safetensors + manifest.json")
    parser.add_argument("--self-test", action="store_true", help="run stdlib-only model-free checks")
    args = parser.parse_args()
    if args.self_test:
        return _self_test()
    if not args.source_root or not args.output_dir:
        parser.error("--source-root and --output-dir are required for export")
    try:
        return _export(args)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"export_preset_cache.py: ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
