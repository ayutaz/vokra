"""kahrendt/microWakeWord TFLite → Vokra GGUF (M5-03b typed sidecar).

Offline sidecar tool (FR-LD-05: no Python / TFLite ever enters the runtime).
Fetches a canonical microWakeWord model release (Apache-2.0), inspects the
TFLite FlatBuffer via ``ai-edge-litert.Interpreter.get_tensor_details()``,
and emits a GGUF v3 directly (without a re-quantizing writer) whose metadata keys use the
``vokra.kws.*`` prefix so ``vokra_core::gguf::GgufFile::from_external`` (the
no_std GGUF reader the ``vokra-vad-micro`` sister crate already uses) can
open it on both host and thumbv8m Cortex-M55 (M5-03 IoT Tier-3).

# Why this file exists

This script bridges the upstream TFLite artifact to the Vokra GGUF shape
the ``vokra-kws-micro`` runtime reads. That crate's forward scaffold runs
log-mel -> INT8 quantise -> INT8 chain -> threshold when a validated chain is
attached via ``set_chain``; canonical TFLite topology binding remains an
owner-approved VAST task.

The GGUF preserves source INT8 bytes in Q8_0 identity blocks, source INT32
biases as dense I32, and stamps the complete TFLite quantization vectors. Operator topology and activation
parameters remain a separate inspected contract for the runtime binder.

# Contract — GGUF metadata keys (vokra.kws.* prefix)

The output GGUF carries these metadata keys (matching the vokra-vad-micro
``vokra.silero.*`` posture — a per-model prefix with the audio-dialect
category as a discriminator):

- ``vokra.kws.arch``         = ``"microwakeword"``  (distinct from
                                ``"openwakeword"`` — the two are separate
                                ecosystems; openWakeWord targets host CPUs
                                via a shared speech-embedding TFLite, while
                                microWakeWord targets microcontrollers via
                                a self-contained MC-MobileNet).
- ``vokra.kws.model``        = ``"hey_jarvis"`` (the only accepted name)
- ``vokra.kws.threshold``    = f32 (default 0.5; the wake-decision cutoff)
- ``vokra.kws.sample_rate``  = u32 (typically 16000)
- ``vokra.kws.hop_ms``       = u32 (typically 10 or 20)
- ``vokra.kws.window_ms``    = u32 (typically 30 or 32)
- ``vokra.kws.n_mels``       = u32 (typically 40)
- ``vokra.kws.feature_dim``  = u32 (per-frame feature vector length,
                                    equals ``n_mels`` for standard mel;
                                    kept as an independent key because a
                                    stacked-frame model may differ)
- ``vokra.kws.tflite_sha256`` = string (source TFLite hex digest for
                                        provenance audit)
- ``vokra.kws.upstream``     = string (upstream release URL for provenance)
- Provenance chunk group (``vokra.provenance.*``) written directly per
  ``license_class::Permissive`` +
  ``apache-2.0`` posture. The sidecar emits these provenance keys inline so
  the artifact can be checked by the FR-OP-32 catalog-reality gate.

# Tensor emission (Q8_0 source-byte carrier and I32 bias carrier)

The upstream TFLite is INT8-quantized for TFLite-Micro inference on
Cortex-M55. Each INT8 source tensor is emitted as Q8_0 with exact source
bytes and an identity FP16 block scale; complete scale/zero-point vectors and
``quantized_dimension`` are metadata. INT32 bias values are emitted as raw
little-endian GGML_TYPE_I32 (tag 26), with zero-point exactly zero. Q8_0
payloads require only that the total element count is a multiple of 32; the
writer fails closed rather than silently changing a declared shape.

# NOT REFERENCED (clean-room)

- ``kahrendt/microWakeWord`` Python training code (Apache-2.0 — we do
  not vendor or re-implement it; we consume the released ``.tflite`` as
  an opaque black-box).
- ``esphome/esphome`` micro_wake_word component (GPL-3.0 — never
  imported, never inspected; the ESPHome layer is out-of-scope for
  Vokra Apache-2.0 posture, see CLAUDE.md "Piper (piper1-gpl)" red-line).

The tensor extraction logic is derived from ``ai-edge-litert`` public
docs (``Interpreter.get_tensor_details()`` returning ``[{name, shape,
dtype, quantization}]``) — a black-box API contract, no source
transliteration.

# Usage

::

    cd tools/parity/microwakeword
    uv sync
    # The owner-approved VAST worker supplies the authenticated byte digest;
    # the model identity and URL are fixed by this script. It also supplies
    # an independently hashed JSON manifest whose `tensors` entries identify
    # persistent FlatBuffer buffers (not allocated activations):
    uv run python prepare_checkpoint.py \\
        --expected-sha256 <authenticated-hey_jarvis-tflite-sha256> \\
        --tensor-manifest /approved/path/hey_jarvis.tensors.json \\
        --tensor-manifest-sha256 <authenticated-tensor-manifest-sha256> \\
        --output /approved/path/hey_jarvis.gguf

    # ``--input`` is only an authenticated local transport for that same
    # canonical payload (it does not change provenance):
    uv run python prepare_checkpoint.py \\
        --input /approved/path/hey_jarvis.tflite \\
        --expected-sha256 <authenticated-hey_jarvis-tflite-sha256> \\
        --tensor-manifest /approved/path/hey_jarvis.tensors.json \\
        --tensor-manifest-sha256 <authenticated-tensor-manifest-sha256> \\
        --output /approved/path/hey_jarvis.gguf

Fails loudly on any anomaly (unsupported weight dtype, incomplete runtime affine
metadata, malformed FlatBuffer) rather than masking it — FR-EX-08
posture, matches every other sidecar in ``tools/parity/``.

The required tensor manifest is an owner-authenticated JSON object with
``format``, exact raw-parser ``producer`` identity, ``complete: true``,
``source_sha256``/``source_size``, one subgraph, and exact tensor/buffer/
constant counts. Each entry must carry the inspected FlatBuffer ``index``,
exact ``name``/``type``/``shape``, ``kind: "constant"``, a bounded
``buffer_index``, positive ``buffer_size``, and ``buffer_sha256``. Supported
constant dtypes are ``int8``, ``int32``, and ``float32``; dense element count
must exactly explain the buffer byte size. This is not inferred from
``get_tensor()``. INT32 bias entries use dense GGUF I32 and require positive
scales, exact zero-points of zero, and the same checked axis/shape contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Callable

# ----------------------------------------------------------------------
# Constants — Vokra GGUF metadata keys (``vokra.kws.*`` per ADR M5-03b)
# ----------------------------------------------------------------------

# Architecture discriminator: distinct from ``openwakeword`` (the two
# ecosystems target different tiers — MC-MobileNet on M55 vs speech-embed
# MLP on RPi/Linux). Downstream binders switch on this key.
ARCH: str = "microwakeword"

# Standard microWakeWord front-end defaults (upstream v2 release,
# owner-verifiable via ``strings <model>.tflite | grep -i mel``). Emitted
# as GGUF metadata so ``vokra-kws-micro/src/features.rs`` picks them up at
# load time rather than hard-coding them at compile time (the same
# posture Vokra takes for Whisper front-end via ``vokra.frontend.*``).
DEFAULT_SAMPLE_RATE: int = 16_000
DEFAULT_HOP_MS: int = 10
DEFAULT_WINDOW_MS: int = 32
DEFAULT_N_MELS: int = 40
DEFAULT_THRESHOLD: float = 0.5
DEFAULT_UPSTREAM_URL: str = (
    # ESPHome hosts the canonical curated v2 release. The model and URL are
    # fixed below; conversion refuses alternate identities.
    "https://github.com/esphome/micro-wake-word-models/raw/05b65922cc433c9df13e98e32a7fe520758c837e/models/v2/hey_jarvis.tflite"
)
CANONICAL_MODEL_NAME = "hey_jarvis"
CANONICAL_MODEL_REPOSITORY = "esphome/micro-wake-word-models"
CANONICAL_MODEL_REVISION = "05b65922cc433c9df13e98e32a7fe520758c837e"
CANONICAL_SOURCE_REPOSITORY = "https://github.com/kahrendt/microWakeWord"
CANONICAL_SOURCE_REVISION = "4665173cd35f1cff9a61e06fc427f124766c488e"
AUTHENTICATED_SHA_ENV = "MICROWAKEWORD_EXPECTED_SHA256"
TENSOR_MANIFEST_SHA_ENV = "MICROWAKEWORD_TENSOR_MANIFEST_SHA256"

# GGUF metadata key names — grouped so the ``add_metadata`` helper below
# reads top-down.
KEY_ARCH = "vokra.kws.arch"
KEY_MODEL = "vokra.kws.model"
KEY_THRESHOLD = "vokra.kws.threshold"
KEY_SAMPLE_RATE = "vokra.kws.sample_rate"
KEY_HOP_MS = "vokra.kws.hop_ms"
KEY_WINDOW_MS = "vokra.kws.window_ms"
KEY_N_MELS = "vokra.kws.n_mels"
KEY_FEATURE_DIM = "vokra.kws.feature_dim"
KEY_TFLITE_SHA256 = "vokra.kws.tflite_sha256"
KEY_UPSTREAM = "vokra.kws.upstream"
KEY_MODEL_REPOSITORY = "vokra.kws.model_repository"
KEY_MODEL_REVISION = "vokra.kws.model_revision"
KEY_SOURCE_REPOSITORY = "vokra.kws.source_repository"
KEY_SOURCE_REVISION = "vokra.kws.source_revision"

# Provenance chunk group (the sidecar emits these keys directly; the Rust
# converter has a separate provenance helper for converter-owned artifacts).
KEY_PROV_LICENSE = "vokra.provenance.license"
KEY_PROV_CLASS = "vokra.provenance.license_class"
KEY_PROV_UPSTREAM_HF = "vokra.provenance.upstream_hf"
KEY_PROV_UPSTREAM_NAME = "vokra.provenance.upstream_name"

GGUF_ALIGNMENT = 32
GGUF_VERSION = 3
GGML_TYPE_F32 = 0
GGML_TYPE_I32 = 26
GGML_TYPE_Q8_0 = 8
Q8_0_BLOCK_SIZE = 32
Q8_0_BLOCK_BYTES = 34

# ----------------------------------------------------------------------


def sha256_of_file(path: Path) -> str:
    """Hex sha256 of the entire file (streamed, no full-file read)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    """Streams the URL to ``dest``. Raises loudly on non-200.

    Kept in stdlib (``urllib.request``) — the microWakeWord release is a
    single ~200 KB TFLite file, no auth, no chunking. Adding ``requests``
    would double this file's dep footprint for zero win.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    # ``urlretrieve`` follows redirects (GitHub raw → objects.githubusercontent.com)
    # and raises ``HTTPError`` on 4xx/5xx by default, which is the loud-fail
    # behaviour we want.
    with urllib.request.urlopen(url) as response:
        if response.status != 200:
            raise SystemExit(
                f"HTTP {response.status} fetching {url!r}: {response.reason}"
            )
        with dest.open("wb") as f:
            shutil.copyfileobj(response, f)


def _exact_finite_integer(value: Any, field: str, name: str) -> int:
    """Return an integer only when the source value is finite and exact."""
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise SystemExit(f"tensor {name!r}: {field} must be a finite integer") from error
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise SystemExit(f"tensor {name!r}: {field} must be a finite integer")
    return int(numeric)


def _validate_affine_metadata(
    scales: Any,
    zero_points: Any,
    quantized_dimension: int,
    tensor_shape: Any,
    name: str,
    *,
    int32_bias: bool = False,
) -> int:
    """Validate TFLite affine vectors before conversion or GGUF emission."""
    if len(scales) == 0 or len(scales) != len(zero_points):
        raise SystemExit(f"tensor {name!r}: malformed TFLite quantization vectors")
    if any(not math.isfinite(float(scale)) or float(scale) <= 0.0 for scale in scales):
        kind = "INT32 bias" if int32_bias else "INT8"
        raise SystemExit(f"tensor {name!r}: TFLite {kind} scales must be finite and positive")
    normalized_zero_points = [
        _exact_finite_integer(zero_point, "INT8 zero_point", name)
        for zero_point in zero_points
    ]
    if int32_bias and any(zero_point != 0 for zero_point in normalized_zero_points):
        raise SystemExit(f"tensor {name!r}: INT32 bias zero_points must be exactly 0")
    if not int32_bias and any(zero_point < -128 or zero_point > 127 for zero_point in normalized_zero_points):
        raise SystemExit(f"tensor {name!r}: INT8 zero_points must be in [-128, 127]")
    normalized_dimension = _exact_finite_integer(
        quantized_dimension, "quantized_dimension", name
    )
    shape = tuple(int(dimension) for dimension in tensor_shape)
    rank = len(shape)
    if len(scales) == 1:
        if normalized_dimension != -1 and (
            normalized_dimension < 0 or normalized_dimension >= rank
        ):
            raise SystemExit(
                f"tensor {name!r}: scalar quantized_dimension {normalized_dimension} "
                f"is invalid for tensor rank {rank}"
            )
    elif normalized_dimension < 0 or normalized_dimension >= rank:
        raise SystemExit(
            f"tensor {name!r}: per-axis quantized_dimension {normalized_dimension} "
            f"is invalid for tensor shape {list(shape)}"
        )
    elif shape[normalized_dimension] != len(scales):
        raise SystemExit(
            f"tensor {name!r}: per-axis quantization has {len(scales)} scales but "
            f"source axis {normalized_dimension} has length {shape[normalized_dimension]}"
        )
    return normalized_dimension


def dequantize_int8_to_f32(
    quantized: np.ndarray,
    scales: np.ndarray,
    zero_points: np.ndarray,
    quantized_dimension: int,
) -> np.ndarray:
    """Standard TFLite affine dequantization, including per-axis tensors."""
    quantized_dimension = _validate_affine_metadata(
        scales, zero_points, quantized_dimension, quantized.shape, "<unnamed>"
    )
    if scales.size == 1:
        return (
            quantized.astype(np.int64) - int(zero_points[0])
        ).astype(np.float32) * float(scales[0])
    view_shape = [1] * quantized.ndim
    view_shape[quantized_dimension] = scales.size
    return (
        quantized.astype(np.int64)
        - zero_points.astype(np.int64).reshape(view_shape)
    ).astype(np.float32) * scales.astype(np.float32).reshape(view_shape)


def quantization_parameters(
    td: dict[str, Any], tensor_shape: Any, *, int32_bias: bool = False
) -> tuple[np.ndarray, np.ndarray, int]:
    """Returns the complete TFLite quantization vector for one tensor."""
    params = td.get("quantization_parameters")
    if isinstance(params, dict):
        scales = np.asarray(params.get("scales", []), dtype=np.float32).reshape(-1)
        raw_zero_points = np.asarray(params.get("zero_points", [])).reshape(-1)
        zero_points = raw_zero_points
        raw_dimension = params.get("quantized_dimension", -1)
        quantized_dimension = raw_dimension
    else:
        quantization = td.get("quantization", (0.0, 0))
        if not isinstance(quantization, tuple) or len(quantization) != 2:
            scales = np.empty(0, dtype=np.float32)
            zero_points = np.empty(0, dtype=np.int64)
        else:
            scales = np.asarray([quantization[0]], dtype=np.float32)
            zero_points = np.asarray([quantization[1]])
        quantized_dimension = -1
    quantized_dimension = _validate_affine_metadata(
        scales,
        zero_points,
        quantized_dimension,
        tensor_shape,
        td.get("name", "<unnamed>"),
        int32_bias=int32_bias,
    )
    zero_points = np.asarray(
        [_exact_finite_integer(value, "INT32 bias zero_point" if int32_bias else "INT8 zero_point", td.get("name", "<unnamed>"))
         for value in zero_points],
        dtype=np.int64,
    )
    return scales, zero_points, quantized_dimension


def load_tensor_manifest(
    path: Path, expected_sha256: str, source_sha256: str, source_size: int | None = None
) -> dict[int, dict[str, Any]]:
    """Load an independently authenticated constant-tensor manifest.

    ``Interpreter.get_tensor`` alone cannot distinguish persistent FlatBuffer
    buffers from allocated activations. The manifest must therefore be
    produced by the owner-approved VAST FlatBuffer inspection and authenticated
    independently before any tensor is emitted.
    """
    if path.is_symlink() or not path.is_file():
        raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
    try:
        manifest_sha256 = sha256_of_file(path)
    except OSError as error:
        raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED") from error
    if manifest_sha256 != expected_sha256:
        raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED") from error
    producer = document.get("producer") if isinstance(document, dict) else None
    if (
        not isinstance(document, dict)
        or document.get("format") != "vokra-microwakeword-tflite-tensor-manifest-v1"
        or producer != {"method": "raw_flatbuffer", "name": "microwakeword_tensor_manifest.py", "version": "1.0"}
        or document.get("source_sha256") != source_sha256
        or (source_size is not None and document.get("source_size") != source_size)
        or not isinstance(document.get("source_size"), int)
        or isinstance(document.get("source_size"), bool)
        or document.get("source_size", 0) <= 0
        or document.get("complete") is not True
        or document.get("subgraph_count") != 1
        or not isinstance(document.get("tensor_count"), int)
        or isinstance(document.get("tensor_count"), bool)
        or document.get("tensor_count") <= 0
        or not isinstance(document.get("buffer_count"), int)
        or isinstance(document.get("buffer_count"), bool)
        or document.get("buffer_count") <= 0
        or not isinstance(document.get("constant_count"), int)
        or isinstance(document.get("constant_count"), bool)
        or document.get("constant_count") <= 0
        or not isinstance(document.get("nonempty_buffer_count"), int)
        or isinstance(document.get("nonempty_buffer_count"), bool)
        or document.get("nonempty_buffer_count") <= 0
        or not isinstance(document.get("referenced_nonempty_buffer_count"), int)
        or isinstance(document.get("referenced_nonempty_buffer_count"), bool)
        or document.get("referenced_nonempty_buffer_count") <= 0
        or not isinstance(document.get("unreferenced_nonempty_buffer_indices"), list)
    ):
        raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
    entries = document.get("tensors")
    if not isinstance(entries, list) or not entries:
        raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
    result: dict[int, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
        index = entry.get("index")
        name = entry.get("name")
        kind = entry.get("kind")
        dtype = entry.get("dtype")
        buffer_index = entry.get("buffer_index")
        buffer_size = entry.get("buffer_size")
        buffer_sha256 = entry.get("buffer_sha256")
        shape = entry.get("shape")
        type_code = entry.get("type")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or not isinstance(name, str)
            or not name
            or kind != "constant"
            or dtype not in {"int8", "int32", "float32"}
            or not isinstance(buffer_index, int)
            or isinstance(buffer_index, bool)
            or buffer_index < 0
            or not isinstance(buffer_size, int)
            or isinstance(buffer_size, bool)
            or buffer_size <= 0
            or not isinstance(buffer_sha256, str)
            or len(buffer_sha256) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in buffer_sha256)
            or not isinstance(type_code, int)
            or isinstance(type_code, bool)
            or type_code != {"int8": 9, "int32": 2, "float32": 0}[dtype]
            or index >= document["tensor_count"]
            or buffer_index >= document["buffer_count"]
            or not isinstance(shape, list)
            or any(not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0 for dimension in shape)
            or index in result
        ):
            raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
        elements = 1
        for dimension in shape:
            elements *= dimension
        item_size = {"int8": 1, "float32": 4, "int32": 4}[dtype]
        if elements * item_size != buffer_size:
            raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
        result[index] = entry
    if len(result) != document["constant_count"] or document["tensor_count"] < len(result):
        raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
    ownership = document.get("buffer_ownership")
    if not isinstance(ownership, list) or len(ownership) != document["referenced_nonempty_buffer_count"]:
        raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
    expected_owners: dict[int, list[int]] = {}
    for index, entry in result.items():
        expected_owners.setdefault(entry["buffer_index"], []).append(index)
    actual_owners: dict[int, list[int]] = {}
    for owner in ownership:
        if not isinstance(owner, dict) or set(owner) != {"buffer_index", "tensor_indices", "tensor_count", "shared", "buffer_size", "buffer_sha256"}:
            raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
        buffer_index = owner["buffer_index"]
        tensor_indices = owner["tensor_indices"]
        if (not isinstance(buffer_index, int) or isinstance(buffer_index, bool) or buffer_index < 0
                or buffer_index >= document["buffer_count"] or not isinstance(tensor_indices, list)
                or not tensor_indices or tensor_indices != sorted(tensor_indices)
                or len(set(tensor_indices)) != len(tensor_indices)
                or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in tensor_indices)
                or owner["tensor_count"] != len(tensor_indices)
                or owner["shared"] != (len(tensor_indices) > 1)
                or not isinstance(owner["buffer_size"], int) or isinstance(owner["buffer_size"], bool) or owner["buffer_size"] <= 0
                or not isinstance(owner["buffer_sha256"], str) or len(owner["buffer_sha256"]) != 64
                or any(character not in "0123456789abcdefABCDEF" for character in owner["buffer_sha256"])):
            raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
        actual_owners[buffer_index] = tensor_indices
        owned_entries = [result.get(item) for item in tensor_indices]
        if any(item is None or item["buffer_index"] != buffer_index
               or item["buffer_size"] != owner["buffer_size"]
               or item["buffer_sha256"].lower() != owner["buffer_sha256"].lower()
               for item in owned_entries):
            raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
    if actual_owners != expected_owners or len(actual_owners) != len(ownership):
        raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
    unreferenced = document["unreferenced_nonempty_buffer_indices"]
    if (unreferenced != sorted(unreferenced) or len(set(unreferenced)) != len(unreferenced)
            or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 or item >= document["buffer_count"] for item in unreferenced)
            or set(unreferenced) & set(actual_owners)
            or document["nonempty_buffer_count"] != len(unreferenced) + len(actual_owners)):
        raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
    return result


def extract_tensors(
    interp: Interpreter, verbose: bool, constant_manifest: dict[int, dict[str, Any]]
) -> tuple[list[dict[str, Any]], int, int]:
    """Walks ``interp.get_tensor_details()`` and returns
    ``(records, weight_count, activation_count)`` — where ``records`` is
    the per-weight list including source int8 bytes and complete quantization
    vectors.

    Only tensor indices present in the authenticated manifest are treated as
    constants and become GGUF tensors. Unlisted interpreter details are
    treated as activations; ``get_tensor`` success is never used as proof of
    persistent FlatBuffer ownership.
    """
    weights: list[dict[str, Any]] = []
    n_weights = 0
    n_activations = 0
    details_by_index: dict[int, dict[str, Any]] = {}
    for td in interp.get_tensor_details():
        index = td.get("index")
        if not isinstance(index, (int, np.integer)) or isinstance(index, bool) or int(index) < 0 or int(index) in details_by_index:
            raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
        details_by_index[int(index)] = td
    if set(constant_manifest) - set(details_by_index):
        raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
    for idx, entry in constant_manifest.items():
        td = details_by_index[idx]
        if td.get("name") != entry["name"]:
            raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
    for td in details_by_index.values():
        idx = td["index"]
        name = td["name"]
        shape = td["shape"]
        dtype = td["dtype"]
        entry = constant_manifest.get(int(idx))
        if entry is None:
            n_activations += 1
            continue
        expected_dtype = {
            "int8": np.int8,
            "int32": np.int32,
            "float32": np.float32,
        }[entry["dtype"]]
        if dtype != expected_dtype:
            raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
        try:
            data = interp.get_tensor(idx)
        except (ValueError, RuntimeError) as error:
            raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED") from error
        # `data` shape and dtype are the ground truth (get_tensor_details()
        # can carry a stale shape when the interpreter has never been
        # allocated for the specific batch dimension).
        if not isinstance(data, np.ndarray) or not data.flags.c_contiguous or (
            data.dtype.byteorder == ">"
            or (data.dtype.byteorder == "=" and not np.little_endian)
        ):
            raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
        raw = data.tobytes(order="C")
        if list(data.shape) != entry.get("shape") or len(raw) != entry.get("buffer_size") or hashlib.sha256(raw).hexdigest() != entry.get("buffer_sha256"):
            raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
        n_weights += 1
        if dtype == np.int8:
            scales, zero_points, quantized_dimension = quantization_parameters(td, data.shape)
            f32 = dequantize_int8_to_f32(
                data, scales, zero_points, quantized_dimension
            )
            weights.append({
                "name": name,
                "shape": list(data.shape),
                "f32_data": f32,
                "i8_data": np.ascontiguousarray(data.astype(np.int8)),
                "orig_dtype": "int8",
                "scales": scales,
                "zero_points": zero_points,
                "quantized_dimension": quantized_dimension,
                "source_index": int(idx),
            })
        elif dtype == np.float32:
            weights.append({
                "name": name,
                "shape": list(data.shape),
                "f32_data": data.astype(np.float32),
                "orig_dtype": "float32",
                "source_index": int(idx),
            })
        elif dtype == np.int32:
            scales, zero_points, quantized_dimension = quantization_parameters(
                td, data.shape, int32_bias=True
            )
            weights.append({
                "name": name,
                "shape": list(data.shape),
                "i32_data": np.ascontiguousarray(data.astype("<i4", copy=False)),
                "orig_dtype": "int32",
                "scales": scales,
                "zero_points": zero_points,
                "quantized_dimension": quantized_dimension,
                "source_index": int(idx),
            })
        else:
            # Loud fail rather than mask — the Q8_0/F32 contract is deliberate.
            raise SystemExit(
                f"tensor {name!r}: unsupported dtype {dtype!r} (only INT8, INT32 + F32 "
                "are supported by this sidecar)"
            )
        if verbose:
            print(f"  emit[{weights[-1]['orig_dtype']:>7s}] idx={idx:3d} "
                  f"name={name!r} shape={list(data.shape)}",
                  file=sys.stderr)
    return weights, n_weights, n_activations


def _gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _gguf_kv_string(key: str, value: str) -> bytes:
    return _gguf_string(key) + struct.pack("<I", 8) + _gguf_string(value)


def _gguf_kv_u32(key: str, value: int) -> bytes:
    return _gguf_string(key) + struct.pack("<II", 4, value)


def _gguf_kv_f32(key: str, value: float) -> bytes:
    return _gguf_string(key) + struct.pack("<If", 6, value)


def _gguf_kv_i32(key: str, value: int) -> bytes:
    return _gguf_string(key) + struct.pack("<Ii", 5, value)


def _gguf_kv_f32_array(key: str, values: np.ndarray) -> bytes:
    flat = _plain_values(values)
    payload = struct.pack("<IIQ", 9, 6, len(flat))
    payload += struct.pack(f"<{len(flat)}f", *(float(value) for value in flat))
    return _gguf_string(key) + payload


def _gguf_kv_i32_array(key: str, values: np.ndarray) -> bytes:
    payload = struct.pack("<IIQ", 9, 5, int(values.size))
    payload += np.asarray(values, dtype="<i4").tobytes()
    return _gguf_string(key) + payload


def _gguf_kv_i64_array(key: str, values: np.ndarray) -> bytes:
    flat = _plain_values(values)
    payload = struct.pack("<IIQ", 9, 11, len(flat))
    payload += struct.pack(f"<{len(flat)}q", *(int(value) for value in flat))
    return _gguf_string(key) + payload


def _plain_values(values: Any) -> list[Any]:
    """Flattens NumPy arrays without making the wire writer NumPy-dependent."""
    if hasattr(values, "reshape"):
        values = values.reshape(-1)
    result = []
    for value in values:
        result.append(value.item() if hasattr(value, "item") else value)
    return result


def _q8_0_payload(values: Any, name: str) -> bytes:
    flat = []
    for value in _plain_values(values):
        integer = _exact_finite_integer(value, "Q8_0 INT8 value", name)
        if integer < -128 or integer > 127:
            raise SystemExit(
                f"tensor {name!r}: Q8_0 INT8 value {integer} is outside [-128, 127]"
            )
        flat.append(integer)
    if not flat or len(flat) % Q8_0_BLOCK_SIZE != 0:
        raise SystemExit(
            f"tensor {name!r}: {len(flat)} INT8 elements are not a whole number "
            f"of Q8_0 blocks ({Q8_0_BLOCK_SIZE}); refusing to alter its declared shape"
        )
    chunks = []
    for start in range(0, len(flat), Q8_0_BLOCK_SIZE):
        block = flat[start : start + Q8_0_BLOCK_SIZE]
        # This is a source-byte carrier: the runtime applies the TFLite affine
        # vector, so the GGUF Q8_0 block scale is an exact identity FP16 value.
        chunks.append(struct.pack("<H", 0x3C00) + struct.pack("<32b", *block))
    payload = b"".join(chunks)
    assert len(payload) == len(flat) // Q8_0_BLOCK_SIZE * Q8_0_BLOCK_BYTES
    return payload


def _f32_payload(values: Any) -> bytes:
    flat = _plain_values(values)
    return struct.pack(f"<{len(flat)}f", *(float(value) for value in flat))


def _i32_payload(values: Any, name: str) -> bytes:
    """Encode dense signed I32 values in the GGUF little-endian wire format."""
    flat = []
    for value in _plain_values(values):
        integer = _exact_finite_integer(value, "GGUF I32 value", name)
        if integer < -(1 << 31) or integer > (1 << 31) - 1:
            raise SystemExit(f"tensor {name!r}: I32 value is outside signed 32-bit range")
        flat.append(integer)
    return struct.pack(f"<{len(flat)}i", *flat)


def write_gguf(
    output: Path,
    weights: list[dict[str, Any]],
    *,
    model_name: str,
    threshold: float,
    sample_rate: int,
    hop_ms: int,
    window_ms: int,
    n_mels: int,
    tflite_sha256: str,
    upstream_url: str,
) -> None:
    """Emits a GGUF v3 file without re-quantizing source INT8 bytes."""
    metadata = [
        _gguf_kv_string(KEY_ARCH, ARCH),
        _gguf_kv_string(KEY_MODEL, model_name),
        _gguf_kv_f32(KEY_THRESHOLD, threshold),
        _gguf_kv_u32(KEY_SAMPLE_RATE, sample_rate),
        _gguf_kv_u32(KEY_HOP_MS, hop_ms),
        _gguf_kv_u32(KEY_WINDOW_MS, window_ms),
        _gguf_kv_u32(KEY_N_MELS, n_mels),
        _gguf_kv_u32(KEY_FEATURE_DIM, n_mels),
        _gguf_kv_string(KEY_TFLITE_SHA256, tflite_sha256),
        _gguf_kv_string(KEY_UPSTREAM, upstream_url),
        _gguf_kv_string(KEY_MODEL_REPOSITORY, CANONICAL_MODEL_REPOSITORY),
        _gguf_kv_string(KEY_MODEL_REVISION, CANONICAL_MODEL_REVISION),
        _gguf_kv_string(KEY_SOURCE_REPOSITORY, CANONICAL_SOURCE_REPOSITORY),
        _gguf_kv_string(KEY_SOURCE_REVISION, CANONICAL_SOURCE_REVISION),
        _gguf_kv_string(KEY_PROV_LICENSE, "apache-2.0"),
        _gguf_kv_string(KEY_PROV_CLASS, "Permissive"),
        _gguf_kv_string(KEY_PROV_UPSTREAM_HF, "kahrendt/microWakeWord"),
        _gguf_kv_string(KEY_PROV_UPSTREAM_NAME, model_name),
        _gguf_kv_u32("vokra.schema.version", 1),
        _gguf_kv_string("vokra.schema.producer", "microwakeword-sidecar 0.3.0"),
    ]
    tensor_payloads = []
    tensor_specs = []
    for ordinal, w in enumerate(weights):
        wire_shape = list(reversed(w["shape"]))
        if w["orig_dtype"] == "int8":
            quantized_dimension = _validate_affine_metadata(
                _plain_values(w["scales"]),
                _plain_values(w["zero_points"]),
                w["quantized_dimension"],
                w["shape"],
                w["name"],
            )
            payload = _q8_0_payload(w["i8_data"], w["name"])
            dtype = GGML_TYPE_Q8_0
            metadata.extend([
                _gguf_kv_string(f"vokra.kws.tensor.{ordinal}.name", w["name"]),
                _gguf_kv_f32_array(
                    f"vokra.kws.tensor.{ordinal}.quant.scales", w["scales"]
                ),
                _gguf_kv_i64_array(
                    f"vokra.kws.tensor.{ordinal}.quant.zero_points", w["zero_points"]
                ),
                _gguf_kv_i32(
                    f"vokra.kws.tensor.{ordinal}.quant.quantized_dimension",
                    quantized_dimension,
                ),
            ])
        elif w["orig_dtype"] == "int32":
            quantized_dimension = _validate_affine_metadata(
                _plain_values(w["scales"]),
                _plain_values(w["zero_points"]),
                w["quantized_dimension"],
                w["shape"],
                w["name"],
                int32_bias=True,
            )
            payload = _i32_payload(w["i32_data"], w["name"])
            dtype = GGML_TYPE_I32
            metadata.extend([
                _gguf_kv_string(f"vokra.kws.tensor.{ordinal}.name", w["name"]),
                _gguf_kv_f32_array(
                    f"vokra.kws.tensor.{ordinal}.quant.scales", w["scales"]
                ),
                _gguf_kv_i64_array(
                    f"vokra.kws.tensor.{ordinal}.quant.zero_points", w["zero_points"]
                ),
                _gguf_kv_i32(
                    f"vokra.kws.tensor.{ordinal}.quant.quantized_dimension",
                    quantized_dimension,
                ),
            ])
        elif w["orig_dtype"] == "float32":
            payload = _f32_payload(w["f32_data"])
            dtype = GGML_TYPE_F32
        else:
            raise SystemExit(f"tensor {w.get('name', '<unnamed>')!r}: unsupported source dtype")
        tensor_payloads.append(payload)
        # GGUF stores dimensions in innermost-first order; NumPy reports the
        # source array in outermost-first order. Keep the element count and
        # logical shape unchanged while applying only this wire conversion.
        tensor_specs.append((w["name"], wire_shape, dtype))

    header = bytearray()
    header.extend(b"GGUF")
    header.extend(struct.pack("<IQQ", GGUF_VERSION, len(tensor_specs), len(metadata)))
    header.extend(b"".join(metadata))
    offsets = []
    cursor = 0
    for payload in tensor_payloads:
        offsets.append(cursor)
        cursor = (cursor + len(payload) + GGUF_ALIGNMENT - 1) // GGUF_ALIGNMENT * GGUF_ALIGNMENT
    for (name, shape, dtype), offset in zip(tensor_specs, offsets):
        header.extend(_gguf_string(name))
        header.extend(struct.pack("<I", len(shape)))
        header.extend(b"".join(struct.pack("<Q", int(dim)) for dim in shape))
        header.extend(struct.pack("<IQ", dtype, offset))
    header.extend(b"\x00" * ((-len(header)) % GGUF_ALIGNMENT))
    body = bytearray(header)
    for payload in tensor_payloads:
        body.extend(payload)
        body.extend(b"\x00" * ((-len(body)) % GGUF_ALIGNMENT))
    _atomic_publish(output, bytes(body))


def _atomic_publish(
    output: Path, payload: bytes, *, before_link: Callable[[], None] | None = None
) -> None:
    """Publish bytes without replacing a destination created concurrently."""
    temporary: Path | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
        temporary = Path(temporary_name)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if before_link is not None:
            before_link()
        try:
            # A hard link is atomic and fails with EEXIST rather than replacing
            # a sentinel or symlink that appeared after the initial CLI check.
            os.link(temporary, output)
        except FileExistsError as error:
            raise SystemExit(
                f"--output was created concurrently; refusing to overwrite: {output}"
            ) from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _validate_output_destination(output: Path, input_path: Path | None) -> None:
    """Validate output/transport paths before importing or running TFLite."""
    if output.exists() or output.is_symlink():
        raise SystemExit(f"--output must be absent and non-symlink: {output}")
    parent = output.parent
    if not parent.is_dir() or parent.is_symlink():
        raise SystemExit(f"--output parent must be an existing non-symlink directory: {parent}")
    if input_path is not None:
        if input_path.is_symlink() or not input_path.is_file():
            raise SystemExit(f"--input must be a regular non-symlink file: {input_path}")
        if input_path.resolve() == output.resolve():
            raise SystemExit("--input and --output must not overlap")


def _validate_cli_values(
    threshold: float, sample_rate: int, hop_ms: int, window_ms: int, n_mels: int
) -> None:
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise SystemExit("--threshold must be finite and within [0, 1]")
    for option, value in (
        ("--sample-rate", sample_rate),
        ("--hop-ms", hop_ms),
        ("--window-ms", window_ms),
        ("--n-mels", n_mels),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 0xFFFFFFFF:
            raise SystemExit(f"{option} must be a positive uint32")


def _self_test_read_string(blob: bytes, cursor: int) -> tuple[str, int]:
    if cursor + 8 > len(blob):
        raise AssertionError("truncated GGUF string length")
    (length,) = struct.unpack_from("<Q", blob, cursor)
    cursor += 8
    end = cursor + length
    if end > len(blob):
        raise AssertionError("truncated GGUF string payload")
    return blob[cursor:end].decode("utf-8"), end


def _self_test_read_value(blob: bytes, cursor: int) -> tuple[int, Any, int]:
    if cursor + 4 > len(blob):
        raise AssertionError("truncated GGUF metadata value type")
    (tag,) = struct.unpack_from("<I", blob, cursor)
    cursor += 4
    sizes = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
             4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 10: ("<Q", 8),
             11: ("<q", 8), 12: ("<d", 8)}
    if tag in sizes:
        fmt, size = sizes[tag]
        if cursor + size > len(blob):
            raise AssertionError("truncated GGUF metadata scalar")
        return tag, struct.unpack_from(fmt, blob, cursor)[0], cursor + size
    if tag == 8:
        value, cursor = _self_test_read_string(blob, cursor)
        return tag, value, cursor
    if tag == 9:
        if cursor + 12 > len(blob):
            raise AssertionError("truncated GGUF metadata array header")
        element_tag, count = struct.unpack_from("<IQ", blob, cursor)
        cursor += 12
        values = []
        for _ in range(count):
            if element_tag in sizes:
                fmt, size = sizes[element_tag]
                if cursor + size > len(blob):
                    raise AssertionError("truncated GGUF array element")
                values.append(struct.unpack_from(fmt, blob, cursor)[0])
                cursor += size
            elif element_tag == 8:
                value, cursor = _self_test_read_string(blob, cursor)
                values.append(value)
            else:
                raise AssertionError(f"unexpected GGUF array element tag {element_tag}")
        return tag, (element_tag, values), cursor
    raise AssertionError(f"unexpected GGUF metadata tag {tag}")


def _self_test_parse_gguf(blob: bytes) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    if blob[:4] != b"GGUF":
        raise AssertionError("self-test output has bad GGUF magic")
    version, tensor_count, metadata_count = struct.unpack_from("<IQQ", blob, 4)
    if version != GGUF_VERSION:
        raise AssertionError(f"unexpected GGUF version {version}")
    cursor = 24
    metadata = {}
    for _ in range(metadata_count):
        key, cursor = _self_test_read_string(blob, cursor)
        tag, value, cursor = _self_test_read_value(blob, cursor)
        metadata[key] = (tag, value)
    tensors = []
    for _ in range(tensor_count):
        name, cursor = _self_test_read_string(blob, cursor)
        (rank,) = struct.unpack_from("<I", blob, cursor)
        cursor += 4
        dimensions = list(struct.unpack_from(f"<{rank}Q", blob, cursor))
        cursor += rank * 8
        dtype, offset = struct.unpack_from("<IQ", blob, cursor)
        cursor += 12
        tensors.append({"name": name, "dimensions": dimensions, "dtype": dtype, "offset": offset})
    data_offset = (cursor + GGUF_ALIGNMENT - 1) // GGUF_ALIGNMENT * GGUF_ALIGNMENT
    return metadata, tensors, data_offset


def self_test() -> None:
    """Exercise the direct writer and parse its wire output without model I/O."""
    class FakeDtype:
        byteorder = "="

    class FakeArray:
        dtype = FakeDtype()
        shape = (2,)
        nbytes = 8
        flags = type("Flags", (), {"c_contiguous": True})()

        def tobytes(self, order: str = "C") -> bytes:
            assert order == "C"
            return struct.pack("<ff", 1.0, 2.0)

        def astype(self, dtype: Any) -> "FakeArray":
            return self

    class FakeNumpy:
        ndarray = FakeArray
        integer = int
        int8 = object()
        int32 = object()
        float32 = FakeDtype()
        little_endian = True

        @staticmethod
        def ascontiguousarray(value: Any) -> Any:
            return value

    global np
    FakeNumpy.float32 = FakeArray.dtype
    np = FakeNumpy

    def expect_metadata_error(
        scales: list[float], zero_points: list[Any], qdim: Any, shape: list[int], text: str
    ) -> None:
        try:
            _validate_affine_metadata(scales, zero_points, qdim, shape, "metadata")
        except SystemExit as error:
            if text not in str(error):
                raise AssertionError(f"wrong affine metadata error: {error}")
        else:
            raise AssertionError("invalid affine metadata was accepted")

    expect_metadata_error([1.0], [0], 2, [4, 4], "scalar quantized_dimension")
    expect_metadata_error([1.0, 2.0], [0, 0], 0, [4, 32], "source axis 0")
    expect_metadata_error([1.0], [128], -1, [32], "zero_points")
    expect_metadata_error([1.0], [0.5], -1, [32], "zero_point must be a finite integer")
    expect_metadata_error([1.0], [float("inf")], -1, [32], "zero_point must be a finite integer")
    expect_metadata_error([1.0], [0], 1.5, [32, 2], "quantized_dimension must be a finite integer")
    expect_metadata_error([1.0], [0], float("nan"), [32], "quantized_dimension must be a finite integer")
    try:
        _validate_affine_metadata([1.0], [1], -1, [3], "bias", int32_bias=True)
    except SystemExit as error:
        if "INT32 bias zero_points" not in str(error):
            raise AssertionError(f"wrong I32 zero-point error: {error}")
    else:
        raise AssertionError("nonzero I32 bias zero point was accepted")
    try:
        _validate_affine_metadata([1.0, 2.0], [0, 0], 0, [3], "bias", int32_bias=True)
    except SystemExit as error:
        if "source axis 0" not in str(error):
            raise AssertionError(f"wrong I32 shape error: {error}")
    else:
        raise AssertionError("invalid I32 bias axis shape was accepted")
    for value, text in (
        (0.5, "Q8_0 INT8 value must be a finite integer"),
        (float("nan"), "Q8_0 INT8 value must be a finite integer"),
        (128, "outside [-128, 127]"),
    ):
        try:
            _q8_0_payload([value] * Q8_0_BLOCK_SIZE, "payload")
        except SystemExit as error:
            if text not in str(error):
                raise AssertionError(f"wrong Q8 payload error: {error}")
        else:
            raise AssertionError("invalid Q8 payload value was accepted")
    for value in ((1 << 31), -(1 << 31) - 1, 0.5):
        try:
            _i32_payload([value], "bias")
        except SystemExit:
            pass
        else:
            raise AssertionError("invalid I32 payload value was accepted")
    for values in ((float("nan"), 16_000, 10, 32, 40), (0.5, 0, 10, 32, 40)):
        try:
            _validate_cli_values(*values)
        except SystemExit:
            pass
        else:
            raise AssertionError("invalid CLI value was accepted")
    with tempfile.TemporaryDirectory(prefix="vokra-mww-self-test-") as directory:
        directory_path = Path(directory)
        sentinel = directory_path / "race.gguf"

        def create_sentinel() -> None:
            sentinel.write_bytes(b"sentinel")

        try:
            _atomic_publish(sentinel, b"replacement", before_link=create_sentinel)
        except SystemExit as error:
            if "created concurrently" not in str(error):
                raise AssertionError(f"wrong atomic publish error: {error}")
        else:
            raise AssertionError("atomic publish replaced a concurrent destination")
        if sentinel.read_bytes() != b"sentinel":
            raise AssertionError("atomic publish clobbered the concurrent sentinel")
        if list(directory_path.glob(f".{sentinel.name}.*")):
            raise AssertionError("atomic publish left a temporary file")

        output_parent = directory_path / "output"
        output_parent.mkdir()
        _validate_output_destination(output_parent / "ok.gguf", None)
        try:
            _validate_output_destination(directory_path / "missing" / "bad.gguf", None)
        except SystemExit as error:
            if "output parent" not in str(error):
                raise AssertionError(f"wrong output-parent error: {error}")
        else:
            raise AssertionError("missing output parent was accepted")

        manifest_path = directory_path / "tensor-manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "format": "vokra-microwakeword-tflite-tensor-manifest-v1",
                    "producer": {"method": "raw_flatbuffer", "name": "microwakeword_tensor_manifest.py", "version": "1.0"},
                    "source_sha256": "0" * 64,
                    "source_size": 4,
                    "complete": True,
                    "subgraph_count": 1,
                    "tensor_count": 1,
                    "buffer_count": 4,
                    "constant_count": 1,
                    "nonempty_buffer_count": 1,
                    "referenced_nonempty_buffer_count": 1,
                    "unreferenced_nonempty_buffer_indices": [],
                    "buffer_ownership": [{"buffer_index": 3, "tensor_indices": [0], "tensor_count": 1, "shared": False, "buffer_size": 32, "buffer_sha256": hashlib.sha256(b"\x00" * 32).hexdigest()}],
                    "tensors": [
                        {
                            "index": 0,
                            "name": "constant",
                            "kind": "constant",
                            "dtype": "int8",
                            "type": 9,
                            "shape": [32],
                            "buffer_index": 3,
                            "buffer_size": 32,
                            "buffer_sha256": hashlib.sha256(b"\x00" * 32).hexdigest(),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        manifest = load_tensor_manifest(
            manifest_path, sha256_of_file(manifest_path), "0" * 64
        )
        if manifest[0]["name"] != "constant":
            raise AssertionError("authenticated tensor manifest was not loaded")

        class FakeInterpreter:
            def __init__(self, details: list[dict[str, Any]], value: Any):
                self.details = details
                self.value = value

            def get_tensor_details(self) -> list[dict[str, Any]]:
                return self.details

            def get_tensor(self, index: int) -> Any:
                assert index == 0
                return self.value

        runtime_value = FakeArray()
        runtime_entry = {
            "index": 0,
            "name": "constant",
            "dtype": "float32",
            "shape": [2],
            "buffer_size": runtime_value.nbytes,
            "buffer_sha256": hashlib.sha256(runtime_value.tobytes(order="C")).hexdigest(),
        }
        runtime_details = [{"index": 0, "name": "constant", "shape": [2], "dtype": np.float32}]
        extract_tensors(FakeInterpreter(runtime_details, runtime_value), False, {0: runtime_entry})
        for malformed in (
            {**runtime_entry, "shape": [3]},
            {**runtime_entry, "buffer_size": runtime_value.nbytes + 4},
            {**runtime_entry, "buffer_sha256": "0" * 64},
        ):
            try:
                extract_tensors(FakeInterpreter(runtime_details, runtime_value), False, {0: malformed})
            except SystemExit:
                pass
            else:
                raise AssertionError("runtime tensor manifest mismatch was accepted")
        class NonContiguousArray(FakeArray):
            flags = type("Flags", (), {"c_contiguous": False})()

        try:
            extract_tensors(
                FakeInterpreter(runtime_details, NonContiguousArray()),
                False,
                {0: runtime_entry},
            )
        except SystemExit:
            pass
        else:
            raise AssertionError("non-contiguous runtime tensor was accepted")

        class BigEndianArray(FakeArray):
            dtype = type("BigEndianDtype", (), {"byteorder": ">"})()

        try:
            extract_tensors(
                FakeInterpreter(runtime_details, BigEndianArray()),
                False,
                {0: runtime_entry},
            )
        except SystemExit:
            pass
        else:
            raise AssertionError("big-endian runtime tensor was accepted")
        try:
            extract_tensors(FakeInterpreter(runtime_details + runtime_details, runtime_value), False, {0: runtime_entry})
        except SystemExit:
            pass
        else:
            raise AssertionError("duplicate runtime tensor index was accepted")
        invalid_manifest = directory_path / "invalid-tensor-manifest.json"
        invalid_manifest.write_text(
            json.dumps(
                {
                    "source_sha256": "0" * 64,
                    "complete": False,
                    "tensors": [
                        {
                            "index": 7,
                            "name": "activation",
                            "kind": "activation",
                            "dtype": "int8",
                            "buffer_index": 0,
                            "buffer_size": 32,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        try:
            load_tensor_manifest(invalid_manifest, sha256_of_file(invalid_manifest), "0" * 64)
        except SystemExit as error:
            if str(error) != "SOURCE_TENSOR_MANIFEST_REQUIRED":
                raise AssertionError(f"wrong tensor manifest error: {error}")
        else:
            raise AssertionError("invalid tensor manifest was accepted")

        writer_args = dict(
            model_name=CANONICAL_MODEL_NAME,
            threshold=DEFAULT_THRESHOLD,
            sample_rate=DEFAULT_SAMPLE_RATE,
            hop_ms=DEFAULT_HOP_MS,
            window_ms=DEFAULT_WINDOW_MS,
            n_mels=DEFAULT_N_MELS,
            tflite_sha256="0" * 64,
            upstream_url=DEFAULT_UPSTREAM_URL,
        )

        def expect_writer_error(record: dict[str, Any], text: str, label: str) -> None:
            destination = output_parent / f"{label}.gguf"
            try:
                write_gguf(destination, [record], **writer_args)
            except SystemExit as error:
                if text not in str(error):
                    raise AssertionError(f"wrong {label} writer error: {error}")
            else:
                raise AssertionError(f"invalid {label} writer record was accepted")
            if destination.exists() or destination.is_symlink():
                raise AssertionError(f"invalid {label} writer record created output")

        def q8_record(**overrides: Any) -> dict[str, Any]:
            record = {
                "name": "bad_q8",
                "shape": [32],
                "orig_dtype": "int8",
                "i8_data": [0] * Q8_0_BLOCK_SIZE,
                "scales": [1.0],
                "zero_points": [0],
                "quantized_dimension": -1,
            }
            record.update(overrides)
            return record

        expect_writer_error(
            q8_record(quantized_dimension=0.5),
            "quantized_dimension must be a finite integer",
            "fractional-qdim",
        )
        expect_writer_error(
            q8_record(quantized_dimension=float("nan")),
            "quantized_dimension must be a finite integer",
            "nan-qdim",
        )
        expect_writer_error(
            q8_record(quantized_dimension=float("inf")),
            "quantized_dimension must be a finite integer",
            "inf-qdim",
        )
        expect_writer_error(
            q8_record(zero_points=[0.5]),
            "zero_point must be a finite integer",
            "fractional-zero-point",
        )
        expect_writer_error(
            q8_record(zero_points=[float("nan")]),
            "zero_point must be a finite integer",
            "nan-zero-point",
        )
        expect_writer_error(
            q8_record(zero_points=[float("inf")]),
            "zero_point must be a finite integer",
            "inf-zero-point",
        )
        expect_writer_error(
            q8_record(i8_data=[128] * Q8_0_BLOCK_SIZE),
            "outside [-128, 127]",
            "overflow-payload",
        )
        expect_writer_error(
            q8_record(i8_data=[0.5] * Q8_0_BLOCK_SIZE),
            "Q8_0 INT8 value must be a finite integer",
            "fractional-payload",
        )
        expect_writer_error(
            q8_record(i8_data=[float("nan")] * Q8_0_BLOCK_SIZE),
            "Q8_0 INT8 value must be a finite integer",
            "nan-payload",
        )

        output = Path(directory) / "synthetic.gguf"
        try:
            _q8_0_payload([0] * (Q8_0_BLOCK_SIZE - 1), "bad-block")
        except SystemExit as error:
            if "31 INT8 elements" not in str(error):
                raise AssertionError(f"wrong Q8 block-size error: {error}")
        else:
            raise AssertionError("partial Q8 block was accepted")
        row_split_output = Path(directory) / "row-split.gguf"
        row_split_values = list(range(32))
        write_gguf(
            row_split_output,
            [{
                "name": "row_split",
                "shape": [2, 16],
                "orig_dtype": "int8",
                "i8_data": row_split_values,
                "scales": [1.0],
                "zero_points": [0],
                "quantized_dimension": -1,
            }],
            **writer_args,
        )
        row_metadata, row_tensors, row_data_offset = _self_test_parse_gguf(row_split_output.read_bytes())
        if row_tensors != [{"name": "row_split", "dimensions": [16, 2], "dtype": GGML_TYPE_Q8_0, "offset": 0}]:
            raise AssertionError(f"valid total-only Q8 shape was not retained: {row_tensors}")
        row_payload = row_split_output.read_bytes()[row_data_offset : row_data_offset + Q8_0_BLOCK_BYTES]
        if list(struct.unpack_from("<32b", row_payload, 2)) != row_split_values:
            raise AssertionError("total-only Q8 payload order changed")
        raw = list(range(-32, 32))
        write_gguf(
            output,
            [
                {
                    "name": "q8_weight",
                    "shape": [2, 32],
                    "orig_dtype": "int8",
                    "i8_data": raw,
                    "scales": [0.125],
                    "zero_points": [0],
                    "quantized_dimension": -1,
                },
                {
                    "name": "next",
                    "shape": [1],
                    "orig_dtype": "float32",
                    "f32_data": [1.5],
                },
            ],
            model_name=CANONICAL_MODEL_NAME,
            threshold=DEFAULT_THRESHOLD,
            sample_rate=DEFAULT_SAMPLE_RATE,
            hop_ms=DEFAULT_HOP_MS,
            window_ms=DEFAULT_WINDOW_MS,
            n_mels=DEFAULT_N_MELS,
            tflite_sha256="0" * 64,
            upstream_url=DEFAULT_UPSTREAM_URL,
        )
        metadata, tensors, data_offset = _self_test_parse_gguf(output.read_bytes())
        if [tensor["name"] for tensor in tensors] != ["q8_weight", "next"]:
            raise AssertionError("tensor declaration order was not preserved")
        q8, following = tensors
        if q8 != {"name": "q8_weight", "dimensions": [32, 2], "dtype": GGML_TYPE_Q8_0, "offset": 0}:
            raise AssertionError(f"unexpected Q8 tensor info: {q8}")
        if following["offset"] != 96 or following["dtype"] != GGML_TYPE_F32:
            raise AssertionError(f"unexpected aligned following tensor: {following}")
        q8_bytes = output.read_bytes()[data_offset : data_offset + 68]
        if any(q8_bytes[start : start + 2] != b"\x00\x3c" for start in (0, 34)):
            raise AssertionError("Q8 carrier block scale is not FP16 1.0")
        if list(struct.unpack("<64b", b"".join(
            q8_bytes[start + 2 : start + 34] for start in (0, 34)
        ))) != raw:
            raise AssertionError("Q8 carrier bytes changed")
        if metadata[f"vokra.kws.tensor.0.name"] != (8, "q8_weight"):
            raise AssertionError("Q8 source-name metadata is missing or wrong")
        scales = metadata["vokra.kws.tensor.0.quant.scales"]
        zero_points = metadata["vokra.kws.tensor.0.quant.zero_points"]
        if scales != (9, (6, [0.125])) or zero_points != (9, (11, [0])):
            raise AssertionError("Q8 quantization metadata is not typed as expected")
        if metadata["vokra.kws.tensor.0.quant.quantized_dimension"] != (5, -1):
            raise AssertionError("Q8 quantized_dimension metadata is wrong")

        i32_output = directory_path / "i32.gguf"
        bias_values = [-(1 << 31), (1 << 24) + 1, (1 << 31) - 1]
        write_gguf(
            i32_output,
            [{
                "name": "bias",
                "shape": [3],
                "orig_dtype": "int32",
                "i32_data": bias_values,
                "scales": [0.125],
                "zero_points": [0],
                "quantized_dimension": -1,
            }],
            model_name=CANONICAL_MODEL_NAME,
            threshold=DEFAULT_THRESHOLD,
            sample_rate=DEFAULT_SAMPLE_RATE,
            hop_ms=DEFAULT_HOP_MS,
            window_ms=DEFAULT_WINDOW_MS,
            n_mels=DEFAULT_N_MELS,
            tflite_sha256="0" * 64,
            upstream_url=DEFAULT_UPSTREAM_URL,
        )
        i32_metadata, i32_tensors, i32_data_offset = _self_test_parse_gguf(i32_output.read_bytes())
        if i32_tensors != [{"name": "bias", "dimensions": [3], "dtype": GGML_TYPE_I32, "offset": 0}]:
            raise AssertionError(f"unexpected I32 tensor info: {i32_tensors}")
        i32_payload = i32_output.read_bytes()[i32_data_offset : i32_data_offset + 12]
        if i32_payload != struct.pack("<3i", *bias_values):
            raise AssertionError("I32 payload was not exact little-endian storage")
        if i32_metadata["vokra.kws.tensor.0.quant.zero_points"] != (9, (11, [0])):
            raise AssertionError("I32 affine metadata is missing or not typed")
        try:
            write_gguf(
                directory_path / "bad-i32-zero-point.gguf",
                [{
                    "name": "bias", "shape": [1], "orig_dtype": "int32",
                    "i32_data": [1], "scales": [1.0], "zero_points": [1],
                    "quantized_dimension": -1,
                }],
                **writer_args,
            )
        except SystemExit as error:
            if "zero_points must be exactly 0" not in str(error):
                raise AssertionError(f"wrong I32 zero-point error: {error}")
        else:
            raise AssertionError("nonzero I32 bias zero point was accepted")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract kahrendt/microWakeWord TFLite → Vokra GGUF Q8_0 sidecar."
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="Exercise and wire-parse a synthetic GGUF without model I/O.",
    )
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument(
        "--input",
        type=Path,
        help="Local .tflite path (skip download). Mutually exclusive with --url.",
    )
    src.add_argument(
        "--url",
        type=str,
        default=DEFAULT_UPSTREAM_URL,
        help="URL to fetch the .tflite from (default: ESPHome micro-wake-word-models "
             "hey_jarvis v2 release).",
    )
    ap.add_argument("--name", default="hey_jarvis",
                    help="Model name for GGUF vokra.kws.model (default hey_jarvis).")
    ap.add_argument(
        "--expected-sha256",
        help=f"Authenticated canonical payload SHA-256 (or {AUTHENTICATED_SHA_ENV}).",
    )
    ap.add_argument(
        "--tensor-manifest",
        type=Path,
        help="Owner-approved VAST JSON manifest proving persistent source tensors.",
    )
    ap.add_argument(
        "--tensor-manifest-sha256",
        help=f"Authenticated tensor-manifest SHA-256 (or {TENSOR_MANIFEST_SHA_ENV}).",
    )
    ap.add_argument("--output", type=Path,
                    help="Output .gguf path.")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"Wake-decision threshold (default {DEFAULT_THRESHOLD}).")
    ap.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE,
                    help=f"Audio sample rate in Hz (default {DEFAULT_SAMPLE_RATE}).")
    ap.add_argument("--hop-ms", type=int, default=DEFAULT_HOP_MS,
                    help=f"Feature hop in ms (default {DEFAULT_HOP_MS}).")
    ap.add_argument("--window-ms", type=int, default=DEFAULT_WINDOW_MS,
                    help=f"Feature window in ms (default {DEFAULT_WINDOW_MS}).")
    ap.add_argument("--n-mels", type=int, default=DEFAULT_N_MELS,
                    help=f"Number of mel bands (default {DEFAULT_N_MELS}).")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Print per-tensor emit / skip records to stderr.")
    args = ap.parse_args()

    if args.self_test:
        if len(sys.argv) != 2:
            raise SystemExit("--self-test accepts no other arguments")
        self_test()
        print("prepare_checkpoint.py self-test: PASS", file=sys.stderr)
        return 0

    if args.output is None:
        raise SystemExit("--output is required unless --self-test is used")
    if args.name != CANONICAL_MODEL_NAME:
        raise SystemExit(
            f"canonical microWakeWord conversion requires --name {CANONICAL_MODEL_NAME!r}"
        )
    if args.url != DEFAULT_UPSTREAM_URL:
        raise SystemExit(
            "canonical microWakeWord conversion requires the fixed hey_jarvis URL "
            f"at revision {CANONICAL_MODEL_REVISION}"
        )
    expected_sha256 = args.expected_sha256 or os.environ.get(AUTHENTICATED_SHA_ENV, "")
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in expected_sha256):
        raise SystemExit("AUTHENTICATED_PAYLOAD_SHA_REQUIRED")
    expected_sha256 = expected_sha256.lower()
    manifest_sha256 = args.tensor_manifest_sha256 or os.environ.get(TENSOR_MANIFEST_SHA_ENV, "")
    if (
        args.tensor_manifest is None
        or len(manifest_sha256) != 64
        or any(c not in "0123456789abcdefABCDEF" for c in manifest_sha256)
    ):
        raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
    manifest_sha256 = manifest_sha256.lower()
    _validate_cli_values(
        args.threshold, args.sample_rate, args.hop_ms, args.window_ms, args.n_mels
    )
    _validate_output_destination(args.output, args.input)
    if args.tensor_manifest.resolve() == args.output.resolve() or (
        args.input is not None and args.tensor_manifest.resolve() == args.input.resolve()
    ):
        raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
    # Keep the expensive TFLite and NumPy imports out of the no-I/O self-test
    # and out of rejected unauthenticated conversion requests.
    global np
    import numpy as np
    from ai_edge_litert.interpreter import Interpreter

    # Resolve source .tflite path (download or use local).
    if args.input is not None:
        tflite_path = args.input
        # A local path is only an input transport. Once its bytes match the
        # authenticated digest below, provenance still names the fixed
        # canonical mirror/revision rather than pretending the local path is
        # an upstream identity.
        upstream_url = DEFAULT_UPSTREAM_URL
        tmpdir: tempfile.TemporaryDirectory[str] | None = None
    else:
        tmpdir = tempfile.TemporaryDirectory(prefix="vokra-mww-")
        tflite_path = Path(tmpdir.name) / "model.tflite"
        print(f"Downloading {args.url} …", file=sys.stderr)
        download(args.url, tflite_path)
        upstream_url = args.url

    try:
        tflite_sha256 = sha256_of_file(tflite_path)
        if tflite_sha256 != expected_sha256:
            raise SystemExit(
                "AUTHENTICATED_PAYLOAD_SHA_REQUIRED: source bytes do not match "
                f"expected SHA-256 {expected_sha256}"
            )
        size = tflite_path.stat().st_size
        print(f"Source: {tflite_path.name} ({size:,} bytes, sha256={tflite_sha256[:16]}…)",
              file=sys.stderr)
        constant_manifest = load_tensor_manifest(
            args.tensor_manifest, manifest_sha256, expected_sha256, size
        )

        # Parse the TFLite FlatBuffer via ai-edge-litert (successor of
        # tflite-runtime; get_tensor_details() is the same API).
        interp = Interpreter(model_path=str(tflite_path))
        interp.allocate_tensors()

        weights, n_weights, n_activations = extract_tensors(
            interp, args.verbose, constant_manifest
        )
        if not weights:
            raise SystemExit(
                "No weight tensors extracted — the source .tflite may be "
                "activation-only or malformed. Aborting to avoid emitting "
                "an empty GGUF (FR-EX-08)."
            )
        print(f"Extracted {n_weights} weight tensor(s), "
              f"skipped {n_activations} activation tensor(s).",
              file=sys.stderr)

        write_gguf(
            args.output,
            weights,
            model_name=args.name,
            threshold=args.threshold,
            sample_rate=args.sample_rate,
            hop_ms=args.hop_ms,
            window_ms=args.window_ms,
            n_mels=args.n_mels,
            tflite_sha256=tflite_sha256,
            upstream_url=upstream_url,
        )
        out_size = args.output.stat().st_size
        print(f"Wrote {args.output} ({out_size:,} bytes, {n_weights} tensors, "
              f"vokra.kws.arch={ARCH}, vokra.kws.model={args.name})",
              file=sys.stderr)
        print(f"sha256(output) = {sha256_of_file(args.output)}", file=sys.stderr)
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
