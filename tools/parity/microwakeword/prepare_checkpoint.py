"""kahrendt/microWakeWord TFLite → Vokra GGUF (M5-03b typed sidecar).

Offline sidecar tool (FR-LD-05: no Python / TFLite ever enters the runtime).
Consumes a VAST-authenticated raw FlatBuffer inventory (including exact
constant bytes) and emits a GGUF v3 directly (without a re-quantizing writer) whose metadata keys use the
``vokra.kws.*`` prefix so ``vokra_core::gguf::GgufFile::from_external`` (the
no_std GGUF reader the ``vokra-vad-micro`` sister crate already uses) can
open it on both host and thumbv8m Cortex-M55 (M5-03 IoT Tier-3).

# Why this file exists

This script bridges the upstream TFLite artifact to the Vokra GGUF shape
the ``vokra-kws-micro`` runtime reads. That crate's forward scaffold runs
log-mel -> INT8 quantise -> INT8 chain -> threshold when a validated chain is
attached via ``set_chain``; its typed topology is an explicitly untrusted
validation seam, while canonical TFLite topology binding remains an
owner-approved VAST task.

The GGUF preserves source INT8 bytes as dense GGML_TYPE_I8 logical tensors,
source INT32 biases as dense I32, and stamps the complete TFLite quantization vectors. Operator topology and activation
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

# Tensor emission (dense I8 source bytes and I32 bias carrier)

The upstream TFLite is INT8-quantized for TFLite-Micro inference on
Cortex-M55. Each INT8 source tensor is emitted as one dense GGML_TYPE_I8
(tag 24) tensor with exact source shape and bytes; complete scale/zero-point
vectors and ``quantized_dimension`` are metadata. INT32 bias values are emitted
as raw little-endian GGML_TYPE_I32 (tag 26), with zero-point exactly zero. The
legacy Q8_0 writer remains available only for compatibility fixtures and rejects
partial blocks; candidate/normal preparer output does not split or pad source
tensors. Candidate output remains unreviewed and cannot open production binding.

# NOT REFERENCED (clean-room)

- ``kahrendt/microWakeWord`` Python training code (Apache-2.0 — we do
  not vendor or re-implement it; we consume the released ``.tflite`` as
  an opaque black-box).
- ``esphome/esphome`` micro_wake_word component (GPL-3.0 — never
  imported, never inspected; the ESPHome layer is out-of-scope for
  Vokra Apache-2.0 posture, see CLAUDE.md "Piper (piper1-gpl)" red-line).

The tensor extraction logic consumes the independent raw FlatBuffer producer's
authenticated ``data_hex`` bytes. No interpreter, NumPy, FlatBuffers package,
or other model/runtime dependency is imported or executed here.

# Usage

::

    cd tools/parity/microwakeword
    # The owner-approved VAST worker supplies the authenticated byte digest;
    # the model identity and source revision are fixed by this script. It also supplies
    # an independently hashed JSON manifest whose `tensors` entries identify
    # persistent FlatBuffer buffers (not allocated activations):
    uv run python prepare_checkpoint.py \\
        --input /vast/hey_jarvis.tflite \\
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

Fails loudly on any anomaly (unsupported weight dtype, incomplete source affine
metadata, malformed FlatBuffer) rather than masking it — FR-EX-08
posture, matches every other sidecar in ``tools/parity/``.

The required tensor manifest is an owner-authenticated JSON object with
``format``, exact raw-parser ``producer`` identity, ``complete: true``,
``source_sha256``/``source_size``, one subgraph, and exact tensor/buffer/
constant counts. Each entry must carry the inspected FlatBuffer ``index``,
exact ``name``/``type``/``shape``, ``kind: "constant"``, a bounded
``buffer_index``, positive ``buffer_size``, ``buffer_sha256``, and exact
``data_hex`` bytes. Supported
constant dtypes are ``int8``, ``int32``, and ``float32``; dense element count
must exactly explain the buffer byte size. This is not inferred from a runtime
interpreter. INT32 bias entries use dense GGUF I32 and require positive
scales, exact zero-points of zero, and the same checked axis/shape contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

# ----------------------------------------------------------------------
# Constants — Vokra GGUF metadata keys (``vokra.kws.*`` per ADR M5-03b)
# ----------------------------------------------------------------------

# Architecture discriminator: distinct from ``openwakeword`` (the two
# ecosystems target different tiers — MC-MobileNet on M55 vs speech-embed
# MLP on RPi/Linux). Downstream binders switch on this key.
ARCH: str = "microwakeword"

# Closed production review authority. The caller-supplied manifest digest and
# CLI SHA authenticate transport bytes only; neither can set this value.
# This value is compiled from the reviewed VAST topology evidence; the caller
# cannot set or override it.
REVIEWED_TOPOLOGY_SHA256: str = "e17fa0cae8d504ce71b49ad2113fc6f7ebba9e74dd4070d26e7f291dcbfaf621"
REVIEWED_AUTHORITY = "VAST_REVIEWED_TOPOLOGY_PARITY"
REVIEWED_COMPUTE_TENSOR_NAMES = {
    "model/stream/conv2d/Conv2D", "model/depthwise_conv2d/depthwise",
    "model/depthwise_conv2d/depthwise1", "model/depthwise_conv2d/BiasAdd/ReadVariableOp",
    "model/conv2d_1/Conv2D", "model/batch_normalization/FusedBatchNormV3",
    "model/depthwise_conv2d_1/depthwise", "model/depthwise_conv2d_1/BiasAdd/ReadVariableOp",
    "model/conv2d_2/Conv2D", "model/batch_normalization_1/FusedBatchNormV3",
    "model/depthwise_conv2d_2/depthwise", "model/depthwise_conv2d_2/BiasAdd/ReadVariableOp",
    "model/conv2d_3/Conv2D", "model/batch_normalization_2/FusedBatchNormV3",
    "model/depthwise_conv2d_3/depthwise", "model/depthwise_conv2d_3/BiasAdd/ReadVariableOp",
    "model/conv2d_4/Conv2D", "model/batch_normalization_3/FusedBatchNormV3",
    "model/dense/MatMul", "model/dense/BiasAdd/ReadVariableOp",
}

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
# The model distributor is a GitHub repository, not a Hugging Face model
# namespace. Keep this provenance value distinct from ``KEY_UPSTREAM``, which
# identifies the exact raw model artifact above.
PROVENANCE_UPSTREAM_URL = "https://github.com/esphome/micro-wake-word-models"
CANONICAL_SOURCE_REPOSITORY = "https://github.com/kahrendt/microWakeWord"
CANONICAL_SOURCE_REVISION = "4665173cd35f1cff9a61e06fc427f124766c488e"
AUTHENTICATED_SHA_ENV = "MICROWAKEWORD_EXPECTED_SHA256"
TENSOR_MANIFEST_SHA_ENV = "MICROWAKEWORD_TENSOR_MANIFEST_SHA256"

# Owner-approved VAST evidence authenticates transport bytes only.  The
# topology remains unreviewed, so these constants must never unlock the strict
# production manifest path.
AUTHENTICATED_MODEL_SHA256 = "21a7976add39ee24ec96c63d96b7aaa18e24d1d9824b963e451da8feb4b78b77"
AUTHENTICATED_MODEL_SIZE = 52272
AUTHENTICATED_RAW_INVENTORY_SHA256 = "ce57a719f60af3a494cbd8fb22ff30fdb405b0a3037b049333f25f5794749989"
RAW_INVENTORY_FORMAT = "vokra-microwakeword-tflite-raw-inventory-v1"
RAW_INVENTORY_AUTHORITY = "EVIDENCE_ONLY_UNREVIEWED"
EXPECTED_STATE_NAMES = [
    "stream/states", "stream_1/states", "stream_2/states",
    "stream_3/states", "stream_4/states", "stream_5/states",
]
EXPECTED_STATE_SHAPES = {
    "stream/states": [1, 2, 1, 40],
    "stream_1/states": [1, 4, 1, 30],
    "stream_2/states": [1, 8, 1, 60],
    "stream_3/states": [1, 12, 1, 60],
    "stream_4/states": [1, 20, 1, 60],
    "stream_5/states": [1, 4, 1, 60],
}

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
KEY_PROV_UPSTREAM_URL = "vokra.provenance.upstream_url"
KEY_PROV_UPSTREAM_NAME = "vokra.provenance.upstream_name"

GGUF_ALIGNMENT = 32
GGUF_VERSION = 3
GGML_TYPE_F32 = 0
GGML_TYPE_I8 = 24
GGML_TYPE_I32 = 26
GGML_TYPE_Q8_0 = 8
Q8_0_BLOCK_SIZE = 32
Q8_0_BLOCK_BYTES = 34
MAX_MANIFEST_CONSTANT_BYTES = 256 * 1024 * 1024

# ----------------------------------------------------------------------


def sha256_of_file(path: Path) -> str:
    """Hex sha256 of the entire file (streamed, no full-file read)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys instead of silently taking the last value."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


EXPECTED_CANDIDATE_OPERATOR_SHA256 = (
    "37e180afb9fd79a3057f3e95e78eaa31c2cf55e135df26aba44d1dd1b0aacaf6"
)
_CANDIDATE_OPERATOR_CODES = [
    [129, 142, 142, 142, 142, 142, 142, 22, 143, 143, 143, 143, 143, 143,
     2, 45, 144, 3, 2, 45, 144, 4, 3, 2, 45, 144, 4, 3, 2, 45, 144, 4,
     3, 2, 45, 144, 4, 3, 2, 45, 144, 22, 9, 14, 114],
    [142, 144, 142, 144, 142, 144, 142, 144, 142, 144, 142, 144],
]


def _candidate_operator_digest(document: dict[str, Any]) -> str:
    operators = []
    for subgraph_index, subgraph in enumerate(document.get("subgraphs", [])):
        for operator in subgraph.get("operators", []):
            operators.append({"subgraph_index": subgraph_index, **operator})
    return hashlib.sha256(
        json.dumps(operators, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _candidate_streaming_plan(document: dict[str, Any]) -> dict[str, Any]:
    """Bind only the six resource states observed in evidence-only inventory."""
    subgraphs = document.get("subgraphs")
    if not isinstance(subgraphs, list) or len(subgraphs) != 2:
        raise SystemExit("CANDIDATE_TOPOLOGY_REQUIRED: expected two subgraphs")
    if any(
        not isinstance(subgraph, dict)
        or not isinstance(subgraph.get("operators"), list)
        or not isinstance(subgraph.get("tensors"), list)
        or any(not isinstance(operator, dict) for operator in subgraph["operators"])
        or any(not isinstance(tensor, dict) for tensor in subgraph["tensors"])
        for subgraph in subgraphs
    ):
        raise SystemExit("CANDIDATE_TOPOLOGY_REQUIRED: malformed subgraph evidence")
    for subgraph in subgraphs:
        indices = [tensor.get("index") for tensor in subgraph["tensors"]]
        if any(not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in indices):
            raise SystemExit("CANDIDATE_TENSOR_REQUIRED: tensor index is malformed")
        if len(indices) != len(set(indices)):
            raise SystemExit("CANDIDATE_TENSOR_REQUIRED: duplicate tensor index")
        tensor_count = len(indices)
        for operator in subgraph["operators"]:
            inputs = operator.get("inputs")
            outputs = operator.get("outputs")
            if (
                not isinstance(inputs, list)
                or not isinstance(outputs, list)
                or any(not isinstance(index, int) or isinstance(index, bool) or index < 0 or index >= tensor_count for index in inputs + outputs)
            ):
                raise SystemExit("CANDIDATE_TOPOLOGY_REQUIRED: operator edge index is malformed")
        for field in ("inputs", "outputs"):
            boundary = subgraph.get(field)
            if not isinstance(boundary, list) or any(
                not isinstance(index, int) or isinstance(index, bool) or index < 0 or index >= tensor_count
                for index in boundary
            ):
                raise SystemExit("CANDIDATE_TOPOLOGY_REQUIRED: subgraph edge index is malformed")
    if [
        [op.get("selected_code") for op in sg.get("operators", [])]
        for sg in subgraphs
    ] != _CANDIDATE_OPERATOR_CODES:
        raise SystemExit("CANDIDATE_TOPOLOGY_REQUIRED: operator code sequence drifted")
    if _candidate_operator_digest(document) != EXPECTED_CANDIDATE_OPERATOR_SHA256:
        raise SystemExit("CANDIDATE_TOPOLOGY_REQUIRED: operator/options evidence drifted")
    main_ops, init_ops = subgraphs[0]["operators"], subgraphs[1]["operators"]
    if any(not isinstance(tensor.get("index"), int) for tensor in subgraphs[0]["tensors"] + subgraphs[1]["tensors"]):
        raise SystemExit("CANDIDATE_TENSOR_REQUIRED: tensor indices are malformed")
    if len({tensor["index"] for tensor in subgraphs[0]["tensors"]}) != len(subgraphs[0]["tensors"]) or len({tensor["index"] for tensor in subgraphs[1]["tensors"]}) != len(subgraphs[1]["tensors"]):
        raise SystemExit("CANDIDATE_TENSOR_REQUIRED: duplicate tensor index")
    main_tensors = {tensor["index"]: tensor for tensor in subgraphs[0].get("tensors", [])}
    call_once = main_ops[0]
    if (
        call_once.get("official_name") != "CALL_ONCE"
        or call_once.get("version") != 1
        or call_once.get("inputs") != []
        or call_once.get("outputs") != []
    ):
        raise SystemExit("CANDIDATE_STATE_REQUIRED: CALL_ONCE evidence drifted")
    if call_once.get("builtin_options") != {
        "decoded": {"init_subgraph_index": 1}, "table_present": True, "type": 103
    }:
        raise SystemExit("CANDIDATE_STATE_REQUIRED: CALL_ONCE options drifted")
    handles = []
    expected_handles = []
    for ordinal, op in enumerate(main_ops[1:7]):
        options = op.get("builtin_options", {}).get("decoded", {})
        name = options.get("shared_name")
        expected_name = EXPECTED_STATE_NAMES[ordinal]
        if (
            op.get("official_name") != "VAR_HANDLE"
            or op.get("version") != 1
            or name != expected_name
            or op.get("builtin_options_type") != 111
            or not op.get("builtin_options_table_present")
            or options.get("container") != ""
            or len(op.get("outputs", [])) != 1
        ):
            raise SystemExit("CANDIDATE_STATE_REQUIRED: state handle evidence drifted")
        handles.append({"name": name, "main_handle_tensor": op["outputs"][0]})
        expected_handles.append(name)
    if expected_handles != EXPECTED_STATE_NAMES:
        raise SystemExit("CANDIDATE_STATE_REQUIRED: state order drifted")
    init_by_name: dict[str, dict[str, Any]] = {}
    for op_index in range(0, len(init_ops), 2):
        handle, assign = init_ops[op_index], init_ops[op_index + 1]
        name = handle.get("builtin_options", {}).get("decoded", {}).get("shared_name")
        if (
            handle.get("official_name") != "VAR_HANDLE"
            or assign.get("official_name") != "ASSIGN_VARIABLE"
            or handle.get("version") != 1
            or assign.get("version") != 1
            or len(handle.get("outputs", [])) != 1
            or len(assign.get("inputs", [])) != 2
            or assign["inputs"][0] != handle["outputs"][0]
            or assign.get("outputs") != []
            or assign.get("builtin_options_type") != 0
            or assign.get("builtin_options_table_present")
            or len(assign.get("mutating_variable_inputs", [])) != 0
            or handle.get("builtin_options", {}).get("decoded", {}).get("container") != ""
        ):
            raise SystemExit("CANDIDATE_STATE_REQUIRED: initializer plumbing drifted")
        if not isinstance(name, str) or name in init_by_name:
            raise SystemExit("CANDIDATE_STATE_REQUIRED: initializer state name invalid")
        init_by_name[name] = {
            "init_handle_tensor": handle["outputs"][0],
            "init_value_tensor": assign["inputs"][1],
            "init_assign_operator": assign["index"],
        }
    state_plan = []
    for handle in handles:
        name = handle["name"]
        if name not in init_by_name:
            raise SystemExit("CANDIDATE_STATE_REQUIRED: state lacks initializer")
        initializer = init_by_name[name]
        initial_tensor_index = initializer["init_value_tensor"]
        init_tensors = subgraphs[1].get("tensors", [])
        if not isinstance(initial_tensor_index, int) or not 0 <= initial_tensor_index < len(init_tensors):
            raise SystemExit("CANDIDATE_STATE_REQUIRED: initializer tensor is out of bounds")
        initial_tensor = init_tensors[initial_tensor_index]
        try:
            initial_bytes = bytes.fromhex(initial_tensor.get("data_hex", ""))
        except (TypeError, ValueError) as error:
            raise SystemExit("CANDIDATE_STATE_REQUIRED: initializer bytes are malformed") from error
        initial_quantization = initial_tensor.get("quantization")
        if (
            initial_tensor.get("kind") != "constant"
            or initial_tensor.get("dtype") != "int8"
            or initial_tensor.get("shape") != EXPECTED_STATE_SHAPES[name]
            or not isinstance(initial_tensor.get("data_hex"), str)
            or any(byte != 0x80 for byte in initial_bytes)
            or len(initial_bytes) != math.prod(EXPECTED_STATE_SHAPES[name])
            or hashlib.sha256(initial_bytes).hexdigest() != initial_tensor.get("buffer_sha256")
            or not isinstance(initial_quantization, dict)
            or initial_quantization.get("quantized_dimension") != 0
            or len(initial_quantization.get("scales", [])) != 1
            or len(initial_quantization.get("zero_points", [])) != 1
            or initial_quantization.get("zero_points") != [-128]
        ):
            raise SystemExit("CANDIDATE_STATE_REQUIRED: initializer bytes are not persistent evidence")
        reads = [
            op for op in main_ops
            if op.get("official_name") == "READ_VARIABLE"
            and op.get("inputs") == [handle["main_handle_tensor"]]
        ]
        assigns = [
            op for op in main_ops
            if op.get("official_name") == "ASSIGN_VARIABLE"
            and op.get("inputs", [None])[0] == handle["main_handle_tensor"]
        ]
        if len(reads) != 1 or len(assigns) != 1:
            raise SystemExit("CANDIDATE_STATE_REQUIRED: state read/write plumbing drifted")
        read_tensor = main_tensors.get(reads[0]["outputs"][0])
        if (
            not isinstance(read_tensor, dict)
            or read_tensor.get("dtype") != "int8"
            or read_tensor.get("shape") != EXPECTED_STATE_SHAPES[name]
            or read_tensor.get("quantization") != initial_tensor.get("quantization")
            or len(reads[0].get("inputs", [])) != 1
            or len(reads[0].get("outputs", [])) != 1
            or reads[0].get("version") != 1
            or reads[0].get("builtin_options_type") != 0
            or reads[0].get("builtin_options_table_present")
            or len(assigns[0].get("inputs", [])) != 2
            or assigns[0].get("outputs") != []
            or assigns[0].get("version") != 1
            or assigns[0].get("builtin_options_type") != 0
            or assigns[0].get("builtin_options_table_present")
            or len(assigns[0].get("mutating_variable_inputs", [])) != 0
        ):
            raise SystemExit("CANDIDATE_STATE_REQUIRED: state read shape is absent")
        state_plan.append({
            **handle,
            "main_read_tensor": reads[0]["outputs"][0],
            "state_shape": read_tensor["shape"],
            "state_dtype": read_tensor.get("dtype"),
            "main_read_operator": reads[0]["index"],
            "main_assign_value_tensor": assigns[0]["inputs"][1],
            "main_assign_operator": assigns[0]["index"],
            **initializer,
            "initial_value_shape": initial_tensor.get("shape"),
            "initial_value_dtype": initial_tensor.get("dtype"),
            "initial_value_buffer_index": initial_tensor.get("buffer_index"),
            "initial_value_buffer_sha256": initial_tensor.get("buffer_sha256"),
            "initial_value_data_hex": initial_tensor.get("data_hex"),
            "initial_value_quantization": initial_tensor.get("quantization"),
        })
    return {
        "kind": "MC-MobileNet-streaming-resource-plan",
        "state_count": 6,
        "states": state_plan,
        "call_once": {"subgraph_index": 0, "operator_index": 0, "init_subgraph_index": 1},
    }


def load_candidate_inventory(path: Path) -> tuple[dict[str, Any], str]:
    """Load the fixed VAST evidence; caller paths cannot become authority."""
    if path.is_symlink() or not path.is_file():
        raise SystemExit("CANDIDATE_RAW_INVENTORY_REQUIRED")
    if sha256_of_file(path) != AUTHENTICATED_RAW_INVENTORY_SHA256:
        raise SystemExit("CANDIDATE_RAW_INVENTORY_REQUIRED: evidence identity drifted")
    try:
        document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object)
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit("CANDIDATE_RAW_INVENTORY_REQUIRED") from error
    if (
        not isinstance(document, dict)
        or document.get("format") != RAW_INVENTORY_FORMAT
        or document.get("authority") != RAW_INVENTORY_AUTHORITY
        or document.get("source_sha256") != AUTHENTICATED_MODEL_SHA256
        or document.get("source_size") != AUTHENTICATED_MODEL_SIZE
        or document.get("subgraph_count") != 2
        or document.get("tensor_count") != 82
        or sum(len(sg.get("operators", [])) for sg in document.get("subgraphs", [])) != 57
        or any(key in document for key in ("canonical_identity", "canonical_topology_sha256", "reviewed_topology_sha256"))
    ):
        raise SystemExit("CANDIDATE_RAW_INVENTORY_REQUIRED: source evidence identity is incomplete")
    try:
        _candidate_streaming_plan(document)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
        raise SystemExit("CANDIDATE_STATE_REQUIRED: malformed state/topology evidence") from error
    return document, AUTHENTICATED_RAW_INVENTORY_SHA256


def build_candidate_streaming_manifest(document: dict[str, Any], inventory_sha256: str) -> dict[str, Any]:
    plan = _candidate_streaming_plan(document)
    unquantized_i32 = any(
        tensor.get("dtype") == "int32"
        and not (tensor.get("quantization") or {}).get("scales")
        for subgraph in document["subgraphs"]
        for tensor in subgraph["tensors"]
        if tensor.get("kind") == "constant"
    )
    payload = {
        "schema": "vokra-microwakeword-candidate-streaming-v1",
        "model_identity": {
            "repository": CANONICAL_MODEL_REPOSITORY,
            "revision": CANONICAL_MODEL_REVISION,
            "path": "models/v2/hey_jarvis.tflite",
            "bytes_sha256": document["source_sha256"],
            "size": document["source_size"],
        },
        "source_identity": {
            "repository": CANONICAL_SOURCE_REPOSITORY,
            "revision": CANONICAL_SOURCE_REVISION,
        },
        "candidate_transport": {
            "unquantized_i32_carrier": bool(unquantized_i32),
            "unquantized_i32_policy": (
                "GGUF I32 carrier uses synthetic scale=1, zero_point=0; transport-only, not production quantization authority"
                if unquantized_i32 else None
            ),
        },
        "tensor_storage_contract": {
            "int8": "GGML_TYPE_I8_dense_exact_source_bytes",
            "logical_shape": "preserved_source_shape",
            "production_binding": "closed_until_reviewed_topology_and_independent_parity",
        },
        "source_sha256": document["source_sha256"],
        "source_size": document["source_size"],
        "operator_codes": document["operator_codes"],
        "subgraphs": document["subgraphs"],
        "buffer_ownership": document["buffer_ownership"],
        "unreferenced_nonempty_buffer_indices": document["unreferenced_nonempty_buffer_indices"],
    }
    candidate_digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    tensor_contract = []
    for subgraph_index, subgraph in enumerate(document["subgraphs"]):
        for tensor in subgraph["tensors"]:
            tensor_contract.append({"subgraph_index": subgraph_index, **tensor})
    return {
        "format": "vokra-microwakeword-tflite-candidate-streaming-manifest-v1",
        "authority": "CANDIDATE_UNREVIEWED",
        "source_sha256": document["source_sha256"],
        "source_size": document["source_size"],
        "raw_inventory_sha256": inventory_sha256,
        "candidate_topology_sha256": candidate_digest,
        "model_identity": payload["model_identity"],
        "source_identity": payload["source_identity"],
        "streaming_plan": plan,
        "candidate_transport": payload["candidate_transport"],
        "tensor_storage_contract": payload["tensor_storage_contract"],
        "tensor_contract": tensor_contract,
        "operator_codes": document["operator_codes"],
        "subgraphs": document["subgraphs"],
        "buffer_ownership": document["buffer_ownership"],
        "unreferenced_nonempty_buffer_indices": document["unreferenced_nonempty_buffer_indices"],
    }


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


def quantization_parameters(
    td: dict[str, Any], tensor_shape: Any, *, int32_bias: bool = False
) -> tuple[list[float], list[int], int]:
    """Returns the complete TFLite quantization vector for one tensor."""
    params = td.get("quantization_parameters")
    if isinstance(params, dict):
        scales = list(params.get("scales", []))
        zero_points = list(params.get("zero_points", []))
        raw_dimension = params.get("quantized_dimension", -1)
        quantized_dimension = raw_dimension
    else:
        quantization = td.get("quantization", (0.0, 0))
        if not isinstance(quantization, tuple) or len(quantization) != 2:
            scales = []
            zero_points = []
        else:
            scales = [quantization[0]]
            zero_points = [quantization[1]]
        quantized_dimension = -1
    quantized_dimension = _validate_affine_metadata(
        scales,
        zero_points,
        quantized_dimension,
        tensor_shape,
        td.get("name", "<unnamed>"),
        int32_bias=int32_bias,
    )
    return [float(value) for value in scales], [
        _exact_finite_integer(value, "INT32 bias zero_point" if int32_bias else "INT8 zero_point", td.get("name", "<unnamed>"))
        for value in zero_points
    ], quantized_dimension


def load_tensor_manifest(
    path: Path,
    expected_sha256: str,
    source_sha256: str,
    source_size: int | None = None,
    *,
    allow_untrusted: bool = False,
) -> dict[int, dict[str, Any]]:
    """Load an independently authenticated constant-tensor manifest.

    Runtime tensor APIs cannot distinguish persistent FlatBuffer buffers from
    allocated activations. The manifest must therefore be
    produced by the owner-approved VAST FlatBuffer inspection and authenticated
    independently before any tensor is emitted. Production acceptance also
    requires the closed ``REVIEWED_TOPOLOGY_SHA256`` authority; a caller's
    manifest identity or file SHA cannot unlock it.
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
        document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object)
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED") from error
    producer = document.get("producer") if isinstance(document, dict) else None
    if (
        not isinstance(document, dict)
        or document.get("format") != "vokra-microwakeword-tflite-tensor-manifest-v1"
        or producer != {"method": "raw_flatbuffer", "name": "microwakeword_tensor_manifest.py", "version": "1.1"}
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
    tensor_contract = document.get("tensor_contract")
    if not isinstance(tensor_contract, list) or len(tensor_contract) != document["tensor_count"]:
        raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
    contract_by_index: dict[int, dict[str, Any]] = {}
    for record in tensor_contract:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("index"), int)
            or isinstance(record.get("index"), bool)
            or record["index"] < 0
            or record["index"] >= document["tensor_count"]
            or record["index"] in contract_by_index
            or not isinstance(record.get("name"), str)
            or not record["name"]
            or record.get("kind") not in {"constant", "activation"}
            or record.get("dtype") not in {"int8", "int32", "float32"}
            or not isinstance(record.get("type"), int)
            or isinstance(record.get("type"), bool)
            or record["type"] != {"int8": 9, "int32": 2, "float32": 0}[record["dtype"]]
            or not isinstance(record.get("shape"), list)
            or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in record["shape"])
            or not isinstance(record.get("buffer_index"), int)
            or isinstance(record.get("buffer_index"), bool)
            or record["buffer_index"] < 0
            or record["buffer_index"] >= document["buffer_count"]
            or not isinstance(record.get("buffer_size"), int)
            or isinstance(record.get("buffer_size"), bool)
            or record["buffer_size"] < 0
            or not isinstance(record.get("buffer_sha256"), str)
            or len(record["buffer_sha256"]) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in record["buffer_sha256"])
            or (record.get("kind") == "constant" and (
                not isinstance(record.get("data_hex"), str)
                or len(record["data_hex"]) % 2
                or len(record["data_hex"]) // 2 != record.get("buffer_size")
                or len(record["data_hex"]) // 2 > MAX_MANIFEST_CONSTANT_BYTES
                or any(character not in "0123456789abcdef" for character in record["data_hex"])
            ))
        ):
            raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
        if record.get("kind") == "constant":
            try:
                raw = bytes.fromhex(record["data_hex"])
            except ValueError as error:
                raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED") from error
            if hashlib.sha256(raw).hexdigest() != record["buffer_sha256"].lower():
                raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
        contract_by_index[record["index"]] = record
    if set(contract_by_index) != set(range(document["tensor_count"])):
        raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
    topology = document.get("topology")
    if (
        not isinstance(topology, dict)
        or topology.get("format") != "vokra-microwakeword-tflite-topology-v1"
        or topology.get("complete") is not True
        or not isinstance(topology.get("operator_code_count"), int)
        or isinstance(topology.get("operator_code_count"), bool)
        or topology.get("operator_code_count", 0) <= 0
        or not isinstance(topology.get("operator_count"), int)
        or isinstance(topology.get("operator_count"), bool)
        or not isinstance(topology.get("operators"), list)
        or not topology.get("operators")
        or not isinstance(topology.get("subgraph_inputs"), list)
        or len(topology["subgraph_inputs"]) != 1
        or not isinstance(topology["subgraph_inputs"][0], int)
        or isinstance(topology["subgraph_inputs"][0], bool)
        or topology["subgraph_inputs"][0] < 0
        or not isinstance(topology.get("subgraph_outputs"), list)
        or len(topology["subgraph_outputs"]) != 1
        or not isinstance(topology["subgraph_outputs"][0], int)
        or isinstance(topology["subgraph_outputs"][0], bool)
        or topology["subgraph_outputs"][0] < 0
    ):
        raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
    canonical_digest = topology.get("canonical_digest")
    if (
        not isinstance(canonical_digest, str)
        or len(canonical_digest) != 64
        or any(character not in "0123456789abcdef" for character in canonical_digest)
    ):
        raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
    canonical_payload = {
        "schema": "vokra-microwakeword-canonical-topology-v1",
        "tensor_contract": tensor_contract,
        "graph_inputs": topology["subgraph_inputs"],
        "graph_outputs": topology["subgraph_outputs"],
        "operators": topology["operators"],
    }
    expected_canonical_digest = hashlib.sha256(
        json.dumps(canonical_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if canonical_digest != expected_canonical_digest:
        raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
    canonical_identity = topology.get("canonical_identity")
    if not allow_untrusted:
        if (
            canonical_digest != REVIEWED_TOPOLOGY_SHA256
            or canonical_identity != REVIEWED_TOPOLOGY_SHA256
        ):
            raise SystemExit("AUTHENTICATED_TOPOLOGY_REQUIRED")
    if canonical_identity is not None and canonical_identity != canonical_digest:
        raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
    if topology["operator_count"] != len(topology["operators"]):
        raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
    previous_output: int | None = None
    for operator_index, operator in enumerate(topology["operators"]):
        if (
            not isinstance(operator, dict)
            or not isinstance(operator.get("index"), int)
            or isinstance(operator.get("index"), bool)
            or operator["index"] != operator_index
            or not isinstance(operator.get("opcode_index"), int)
            or isinstance(operator.get("opcode_index"), bool)
            or operator["opcode_index"] < 0
            or operator.get("builtin_name") not in {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "LOGISTIC", "SOFTMAX"}
            or operator.get("version") != 1
            or operator.get("builtin_code") != {"CONV_2D": 3, "DEPTHWISE_CONV_2D": 4, "FULLY_CONNECTED": 9, "LOGISTIC": 14, "SOFTMAX": 25}[operator.get("builtin_name", "")]
            or not isinstance(operator.get("inputs"), list)
            or not isinstance(operator.get("outputs"), list)
            or not operator["outputs"]
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 or value >= document["tensor_count"] for value in operator["inputs"] + operator["outputs"])
            or (previous_output is not None and (not operator["inputs"] or operator["inputs"][0] != previous_output))
        ):
            raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
        previous_output = operator["outputs"][0]
        options = operator.get("options")
        if not isinstance(options, dict):
            raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
        name = operator["builtin_name"]
        expected_inputs = 3 if name in {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"} else 1
        expected_type = {"CONV_2D": 1, "DEPTHWISE_CONV_2D": 2, "FULLY_CONNECTED": 8, "LOGISTIC": 0, "SOFTMAX": 9}[name]
        if len(operator["inputs"]) != expected_inputs or len(operator["outputs"]) != 1 or options.get("type") != expected_type:
            raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
        if name in {"CONV_2D", "DEPTHWISE_CONV_2D"} and (options.get("padding") not in {"SAME", "VALID"} or not isinstance(options.get("stride_h"), int) or not isinstance(options.get("stride_w"), int) or options["stride_h"] <= 0 or options["stride_w"] <= 0 or not isinstance(options.get("dilation_h"), int) or not isinstance(options.get("dilation_w"), int) or options["dilation_h"] <= 0 or options["dilation_w"] <= 0 or options.get("fused_activation") != "NONE"):
            raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
        if name == "CONV_2D" and options.get("quantized_bias_type") != 2:
            raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
        if name == "DEPTHWISE_CONV_2D" and options.get("depth_multiplier") != 1:
            raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
        if name == "FULLY_CONNECTED" and (options.get("fused_activation") != "NONE" or options.get("weights_format") != 0 or options.get("keep_num_dims") or options.get("asymmetric_quantize_inputs") or options.get("quantized_bias_type") != 2):
            raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
        if name == "SOFTMAX" and options.get("beta") != 1.0:
            raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
    if previous_output != topology["subgraph_outputs"][0]:
        raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
    if (
        topology["operators"][0]["inputs"][0] != topology["subgraph_inputs"][0]
        or topology["operators"][-1]["outputs"][0] != topology["subgraph_outputs"][0]
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
            or any(not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0 for dimension in shape)
            or index in result
        ):
            raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
        elements = 1
        for dimension in shape:
            elements *= dimension
        item_size = {"int8": 1, "float32": 4, "int32": 4}[dtype]
        if elements * item_size != buffer_size:
            raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
        contract = contract_by_index.get(index)
        if (
            contract is None
            or contract["name"] != name
            or contract["kind"] != "constant"
            or contract["dtype"] != dtype
            or contract["shape"] != shape
            or contract["buffer_index"] != buffer_index
            or contract["buffer_size"] != buffer_size
            or contract["buffer_sha256"].lower() != buffer_sha256.lower()
            or contract.get("data_hex") != entry.get("data_hex")
        ):
            raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
        try:
            raw = bytes.fromhex(entry["data_hex"])
        except (TypeError, ValueError) as error:
            raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED") from error
        if len(raw) != buffer_size or hashlib.sha256(raw).hexdigest() != buffer_sha256.lower():
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
    constant_manifest: dict[int, dict[str, Any]], verbose: bool = False
) -> tuple[list[dict[str, Any]], int, int]:
    """Decode authenticated FlatBuffer constant bytes without a model runtime."""
    weights: list[dict[str, Any]] = []
    for idx, entry in sorted(constant_manifest.items()):
        name = entry["name"]
        shape = list(entry["shape"])
        try:
            raw = bytes.fromhex(entry["data_hex"])
        except (TypeError, ValueError) as error:
            raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED") from error
        if len(raw) != entry["buffer_size"] or hashlib.sha256(raw).hexdigest() != entry["buffer_sha256"].lower():
            raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
        quant = entry.get("quantization")
        dtype = entry["dtype"]
        if dtype == "int8":
            scales, zero_points, qdim = quantization_parameters(
                {"name": name, "quantization_parameters": quant}, shape
            )
            values = list(struct.unpack(f"<{len(raw)}b", raw))
            record = {"name": name, "shape": shape, "i8_data": values,
                      "orig_dtype": dtype, "scales": scales,
                      "zero_points": zero_points, "quantized_dimension": qdim,
                      "source_index": idx}
        elif dtype == "float32":
            values = list(struct.unpack(f"<{len(raw) // 4}f", raw))
            if any(not math.isfinite(value) for value in values):
                raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
            record = {"name": name, "shape": shape, "f32_data": values,
                      "orig_dtype": dtype, "source_index": idx}
        elif dtype == "int32":
            scales, zero_points, qdim = quantization_parameters(
                {"name": name, "quantization_parameters": quant}, shape,
                int32_bias=True,
            )
            values = list(struct.unpack(f"<{len(raw) // 4}i", raw))
            record = {"name": name, "shape": shape, "i32_data": values,
                      "orig_dtype": dtype, "scales": scales,
                      "zero_points": zero_points, "quantized_dimension": qdim,
                      "source_index": idx}
        else:
            raise SystemExit("SOURCE_TENSOR_MANIFEST_REQUIRED")
        weights.append(record)
        if verbose:
            print(f"  emit[{dtype:>7s}] idx={idx:3d} name={name!r} shape={shape}", file=sys.stderr)
    return weights, len(weights), 0


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


def _gguf_kv_f32_array(key: str, values: Any) -> bytes:
    flat = _plain_values(values)
    payload = struct.pack("<IIQ", 9, 6, len(flat))
    payload += struct.pack(f"<{len(flat)}f", *(float(value) for value in flat))
    return _gguf_string(key) + payload


def _gguf_kv_i32_array(key: str, values: Any) -> bytes:
    flat = _plain_values(values)
    payload = struct.pack("<IIQ", 9, 5, len(flat))
    payload += struct.pack(f"<{len(flat)}i", *(int(value) for value in flat))
    return _gguf_string(key) + payload


def _gguf_kv_i64_array(key: str, values: Any) -> bytes:
    flat = _plain_values(values)
    payload = struct.pack("<IIQ", 9, 11, len(flat))
    payload += struct.pack(f"<{len(flat)}q", *(int(value) for value in flat))
    return _gguf_string(key) + payload


def _plain_values(values: Any) -> list[Any]:
    """Flatten list-like values without introducing a runtime dependency."""
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


def _i8_payload(values: Any, name: str) -> bytes:
    """Encode exact dense signed-I8 source bytes without quantization."""
    flat = []
    for value in _plain_values(values):
        integer = _exact_finite_integer(value, "GGML I8 value", name)
        if integer < -128 or integer > 127:
            raise SystemExit(
                f"tensor {name!r}: GGML I8 value {integer} is outside [-128, 127]"
            )
        flat.append(integer)
    if not flat:
        raise SystemExit(f"tensor {name!r}: GGML I8 tensor cannot be empty")
    return struct.pack(f"<{len(flat)}b", *flat)


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
    extra_metadata: dict[str, str] | None = None,
    dense_i8: bool = False,
    publish: bool = True,
) -> bytes:
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
        _gguf_kv_string(KEY_PROV_UPSTREAM_URL, PROVENANCE_UPSTREAM_URL),
        _gguf_kv_string(KEY_PROV_UPSTREAM_NAME, model_name),
        _gguf_kv_u32("vokra.schema.version", 1),
        _gguf_kv_string("vokra.schema.producer", "microwakeword-sidecar 0.3.0"),
    ]
    if extra_metadata:
        for key, value in sorted(extra_metadata.items()):
            if not (
                key.startswith("vokra.kws.candidate.")
                or key.startswith("vokra.kws.reviewed.")
            ):
                raise SystemExit("conversion metadata key is outside the reserved namespace")
            metadata.append(_gguf_kv_string(key, value))
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
            if dense_i8:
                payload = _i8_payload(w["i8_data"], w["name"])
                dtype = GGML_TYPE_I8
            else:
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
    payload = bytes(body)
    if publish:
        _atomic_publish(output, payload)
    return payload


def _candidate_weights(document: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """Build one dense-I8 logical tensor per authenticated INT8 constant.

    GGML_TYPE_I8 is deliberately used here because real source tensors such as
    hey_jarvis contain element counts that are not Q8_0 block multiples. The
    source shape and every source byte remain unchanged; the candidate is still
    unreviewed and cannot confer production binding authority.
    """
    weights: list[dict[str, Any]] = []
    ordinal = 0

    def checked_bytes(shape: Any, itemsize: int, raw_size: int, name: str) -> None:
        if not isinstance(shape, list) or not shape or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in shape
        ):
            raise SystemExit(f"CANDIDATE_TENSOR_REQUIRED: {name} shape is invalid")
        elements = 1
        for value in shape:
            if elements > MAX_MANIFEST_CONSTANT_BYTES // itemsize // value:
                raise SystemExit(f"CANDIDATE_TENSOR_REQUIRED: {name} shape overflows bounded size")
            elements *= value
        if elements * itemsize != raw_size:
            raise SystemExit(
                f"CANDIDATE_TENSOR_REQUIRED: {name} shape/itemsize does not explain constant bytes"
            )

    for subgraph_index, subgraph in enumerate(document["subgraphs"]):
        for tensor in subgraph["tensors"]:
            if tensor.get("kind") != "constant":
                continue
            if not isinstance(tensor.get("index"), int) or tensor["index"] < 0:
                raise SystemExit("CANDIDATE_TENSOR_REQUIRED: constant tensor index is malformed")
            raw_hex = tensor.get("data_hex")
            if not isinstance(raw_hex, str) or len(raw_hex) % 2:
                raise SystemExit("CANDIDATE_TENSOR_REQUIRED: constant bytes are missing")
            try:
                raw = bytes.fromhex(raw_hex)
            except ValueError as error:
                raise SystemExit("CANDIDATE_TENSOR_REQUIRED: constant bytes are malformed") from error
            if len(raw) != tensor.get("buffer_size") or hashlib.sha256(raw).hexdigest() != tensor.get("buffer_sha256"):
                raise SystemExit("CANDIDATE_TENSOR_REQUIRED: constant bytes are not authenticated")
            shape = tensor.get("shape")
            logical_name = f"subgraph.{subgraph_index}.tensor.{tensor['index']}.{tensor.get('name') or '<unnamed>'}"
            dtype = tensor.get("dtype")
            quant = tensor.get("quantization") or {}
            if not isinstance(quant, dict):
                raise SystemExit("CANDIDATE_TENSOR_REQUIRED: quantization record is malformed")
            scales = quant.get("scales", [])
            zero_points = quant.get("zero_points", [])
            qdim = quant.get("quantized_dimension", -1)
            if dtype == "int8":
                checked_bytes(shape, 1, len(raw), logical_name)
                signed = list(struct.unpack(f"<{len(raw)}b", raw))
                weights.append({
                    "name": logical_name, "shape": shape, "orig_dtype": "int8",
                    "i8_data": signed, "scales": scales, "zero_points": zero_points,
                    "quantized_dimension": qdim,
                })
                ordinal += 1
            elif dtype == "int32":
                checked_bytes(shape, 4, len(raw), logical_name)
                if len(raw) % 4:
                    raise SystemExit("CANDIDATE_TENSOR_REQUIRED: I32 bytes are not dense")
                values = struct.unpack(f"<{len(raw) // 4}i", raw)
                weights.append({
                    "name": logical_name, "shape": shape, "orig_dtype": "int32",
                    "i32_data": values, "scales": scales or [1.0],
                    "zero_points": zero_points or [0], "quantized_dimension": qdim if scales else -1,
                })
                ordinal += 1
            else:
                raise SystemExit(f"CANDIDATE_TENSOR_REQUIRED: unsupported dtype {dtype!r}")
    if not weights:
        raise SystemExit("CANDIDATE_TENSOR_REQUIRED: no constants were authenticated")
    return weights, ordinal


def write_candidate_gguf(
    output: Path,
    document: dict[str, Any],
    candidate_manifest: dict[str, Any],
    *,
    threshold: float,
    sample_rate: int,
    hop_ms: int,
    window_ms: int,
    n_mels: int,
    publish: bool = True,
) -> bytes:
    weights, count = _candidate_weights(document)
    return write_gguf(
        output,
        weights,
        model_name=CANONICAL_MODEL_NAME,
        threshold=threshold,
        sample_rate=sample_rate,
        hop_ms=hop_ms,
        window_ms=window_ms,
        n_mels=n_mels,
        tflite_sha256=AUTHENTICATED_MODEL_SHA256,
        upstream_url=DEFAULT_UPSTREAM_URL,
        extra_metadata={
            "vokra.kws.candidate.authority": "CANDIDATE_UNREVIEWED",
            "vokra.kws.candidate.storage": "GGML_TYPE_I8_dense_exact_source_bytes",
            "vokra.kws.candidate.logical_tensor_binding": "candidate_only_unreviewed",
            "vokra.kws.candidate.production_completion": "reviewed_topology_and_independent_parity_required",
            "vokra.kws.candidate.manifest_digest": hashlib.sha256(
                json.dumps(candidate_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "vokra.kws.candidate.topology_sha256": candidate_manifest["candidate_topology_sha256"],
            "vokra.kws.candidate.streaming_state_count": str(candidate_manifest["streaming_plan"]["state_count"]),
            "vokra.kws.candidate.constant_tensor_count": str(count),
            "vokra.kws.candidate.i32_quant_authority": "transport_only_synthetic_scale" if candidate_manifest["candidate_transport"]["unquantized_i32_carrier"] else "source_quantization_preserved",
        },
        dense_i8=True,
        publish=publish,
    )


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


def _atomic_publish_pair(
    first: Path, first_payload: bytes, second: Path, second_payload: bytes
) -> None:
    """Publish two related files without clobbering a concurrent creator.

    If the second link loses a race, only the first inode created by this call
    is removed; a pre-existing or concurrently-created destination is never
    deleted.  This keeps candidate manifest/GGUF publication fail-closed as a
    pair while retaining the no-clobber guarantee.
    """
    if first == second or first.exists() or first.is_symlink() or second.exists() or second.is_symlink():
        raise SystemExit("candidate output pair destinations must be absent and distinct")
    if not first.parent.is_dir() or not second.parent.is_dir():
        raise SystemExit("candidate output pair parents must exist")
    temporary: list[tuple[Path, Path, tuple[int, int]]] = []
    linked: list[tuple[Path, tuple[int, int]]] = []
    try:
        for destination, payload in ((first, first_payload), (second, second_payload)):
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
            )
            staged = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            identity = (staged.stat().st_dev, staged.stat().st_ino)
            temporary.append((staged, destination, identity))
        for staged, destination, identity in temporary:
            try:
                os.link(staged, destination)
            except FileExistsError as error:
                raise SystemExit("candidate output pair was created concurrently; refusing overwrite") from error
            linked.append((destination, identity))
    except BaseException:
        for destination, identity in linked:
            try:
                # lstat is intentional: a concurrent symlink must never be
                # followed during rollback or mistaken for our staged inode.
                current = destination.lstat()
            except FileNotFoundError:
                continue
            if (current.st_dev, current.st_ino) == identity:
                destination.unlink()
        raise
    finally:
        for staged, _, _ in temporary:
            staged.unlink(missing_ok=True)


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


def _validate_candidate_environment() -> None:
    if sys.platform != "linux" or os.uname().machine != "x86_64":
        raise SystemExit("CANDIDATE_VAST_REQUIRED: Linux x86_64 is required")
    if os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1":
        raise SystemExit("CANDIDATE_VAST_REQUIRED: VOKRA_PUBLISH_ON_VAST=1 is required")
    if os.environ.get("VOKRA_CANDIDATE_CONVERSION") != "1":
        raise SystemExit("CANDIDATE_CONVERSION_DISABLED: explicit candidate opt-in is required")


def _validate_reviewed_environment() -> None:
    """Keep the production conversion exclusively on the reviewed VAST host."""
    if sys.platform != "linux" or platform.machine() not in {"x86_64", "amd64"}:
        raise SystemExit("REVIEWED_VAST_REQUIRED: Linux x86_64 is required")
    if os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1":
        raise SystemExit("REVIEWED_VAST_REQUIRED: VOKRA_PUBLISH_ON_VAST=1 is required")
    if os.environ.get("VOKRA_REVIEWED_CONVERSION") != "1":
        raise SystemExit("REVIEWED_CONVERSION_DISABLED: explicit reviewed opt-in is required")


def load_reviewed_inventory(path: Path) -> tuple[dict[str, Any], str]:
    """Authenticate the fixed raw inventory and its reviewed topology identity."""
    document, inventory_sha256 = load_candidate_inventory(path)
    candidate = build_candidate_streaming_manifest(document, inventory_sha256)
    if candidate["candidate_topology_sha256"] != REVIEWED_TOPOLOGY_SHA256:
        raise SystemExit("REVIEWED_TOPOLOGY_REQUIRED")
    return document, inventory_sha256


def _reviewed_weights(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Decode every authenticated persistent constant preserving source names."""
    weights: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for subgraph in document["subgraphs"]:
        for tensor in subgraph["tensors"]:
            if tensor.get("kind") != "constant":
                continue
            name = tensor.get("name")
            if name not in REVIEWED_COMPUTE_TENSOR_NAMES:
                continue
            shape = tensor.get("shape")
            dtype = tensor.get("dtype")
            raw_hex = tensor.get("data_hex")
            if not isinstance(name, str) or not name or name in seen_names:
                raise SystemExit("REVIEWED_TENSOR_REQUIRED")
            if not isinstance(raw_hex, str) or len(raw_hex) % 2:
                raise SystemExit("REVIEWED_TENSOR_REQUIRED")
            try:
                raw = bytes.fromhex(raw_hex)
            except ValueError as error:
                raise SystemExit("REVIEWED_TENSOR_REQUIRED") from error
            if (
                not isinstance(shape, list)
                or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in shape)
                or len(raw) != tensor.get("buffer_size")
                or hashlib.sha256(raw).hexdigest() != tensor.get("buffer_sha256")
            ):
                raise SystemExit("REVIEWED_TENSOR_REQUIRED")
            quant = tensor.get("quantization")
            if not isinstance(quant, dict):
                raise SystemExit("REVIEWED_TENSOR_REQUIRED")
            if dtype == "int8":
                scales, zero_points, qdim = quantization_parameters(
                    {"name": name, "quantization_parameters": quant}, shape
                )
                values = list(struct.unpack(f"<{len(raw)}b", raw))
                record = {"name": name, "shape": shape, "i8_data": values, "orig_dtype": dtype,
                          "scales": scales, "zero_points": zero_points, "quantized_dimension": qdim}
            elif dtype == "int32":
                scales, zero_points, qdim = quantization_parameters(
                    {"name": name, "quantization_parameters": quant}, shape, int32_bias=True
                )
                if len(raw) % 4:
                    raise SystemExit("REVIEWED_TENSOR_REQUIRED")
                values = list(struct.unpack(f"<{len(raw) // 4}i", raw))
                record = {"name": name, "shape": shape, "i32_data": values, "orig_dtype": dtype,
                          "scales": scales, "zero_points": zero_points, "quantized_dimension": qdim}
            elif dtype == "float32":
                if len(raw) % 4:
                    raise SystemExit("REVIEWED_TENSOR_REQUIRED")
                values = list(struct.unpack(f"<{len(raw) // 4}f", raw))
                if any(not math.isfinite(value) for value in values):
                    raise SystemExit("REVIEWED_TENSOR_REQUIRED")
                record = {"name": name, "shape": shape, "f32_data": values, "orig_dtype": dtype}
            else:
                raise SystemExit("REVIEWED_TENSOR_REQUIRED")
            seen_names.add(name)
            weights.append(record)
    if len(weights) != len(REVIEWED_COMPUTE_TENSOR_NAMES):
        raise SystemExit("REVIEWED_TENSOR_REQUIRED")
    return weights


def _run_reviewed(args: argparse.Namespace) -> int:
    """VAST-only production conversion from the exact reviewed raw inventory."""
    _validate_reviewed_environment()
    if args.input is None or args.raw_inventory is None or args.output is None:
        raise SystemExit("REVIEWED_INPUT_REQUIRED: input, raw inventory, and output are required")
    if args.name != CANONICAL_MODEL_NAME:
        raise SystemExit("REVIEWED_SOURCE_REQUIRED")
    if args.expected_sha256 and args.expected_sha256.lower() != AUTHENTICATED_MODEL_SHA256:
        raise SystemExit("AUTHENTICATED_PAYLOAD_SHA_REQUIRED")
    _validate_cli_values(args.threshold, args.sample_rate, args.hop_ms, args.window_ms, args.n_mels)
    if (args.threshold, args.sample_rate, args.hop_ms, args.window_ms, args.n_mels) != (
        DEFAULT_THRESHOLD, DEFAULT_SAMPLE_RATE, DEFAULT_HOP_MS, DEFAULT_WINDOW_MS, DEFAULT_N_MELS
    ):
        raise SystemExit("REVIEWED_FRONTEND_REQUIRED")
    _validate_output_destination(args.output, args.input)
    if args.raw_inventory.resolve() == args.output.resolve() or args.raw_inventory.resolve() == args.input.resolve():
        raise SystemExit("REVIEWED_INPUT_REQUIRED")
    if sha256_of_file(args.input) != AUTHENTICATED_MODEL_SHA256 or args.input.stat().st_size != AUTHENTICATED_MODEL_SIZE:
        raise SystemExit("AUTHENTICATED_PAYLOAD_SHA_REQUIRED")
    document, inventory_sha256 = load_reviewed_inventory(args.raw_inventory)
    weights = _reviewed_weights(document)
    payload = write_gguf(
        args.output, weights, model_name=args.name, threshold=args.threshold,
        sample_rate=args.sample_rate, hop_ms=args.hop_ms, window_ms=args.window_ms,
        n_mels=args.n_mels, tflite_sha256=AUTHENTICATED_MODEL_SHA256,
        upstream_url=DEFAULT_UPSTREAM_URL,
        extra_metadata={
            "vokra.kws.reviewed.authority": REVIEWED_AUTHORITY,
            "vokra.kws.reviewed.topology_sha256": REVIEWED_TOPOLOGY_SHA256,
            "vokra.kws.reviewed.raw_inventory_sha256": inventory_sha256,
        }, dense_i8=True,
    )
    if not payload or sha256_of_file(args.output) == "":
        raise SystemExit("REVIEWED_OUTPUT_REQUIRED")
    print(f"Wrote reviewed production GGUF ({args.output.stat().st_size:,} bytes; NO_UPLOAD)", file=sys.stderr)
    return 0


def _run_candidate(args: argparse.Namespace) -> int:
    _validate_candidate_environment()
    if args.name != CANONICAL_MODEL_NAME:
        raise SystemExit("CANDIDATE_SOURCE_REQUIRED: canonical model name is fixed")
    if args.input is None or args.raw_inventory is None or args.candidate_manifest is None or args.output is None:
        raise SystemExit("CANDIDATE_INPUT_REQUIRED: input, raw inventory, candidate manifest, and output are required")
    if args.expected_sha256 and args.expected_sha256.lower() != AUTHENTICATED_MODEL_SHA256:
        raise SystemExit("AUTHENTICATED_PAYLOAD_SHA_REQUIRED: candidate identity is fixed")
    for path, label in ((args.input, "candidate input"), (args.raw_inventory, "candidate inventory")):
        if path.is_symlink() or not path.is_file():
            raise SystemExit(f"CANDIDATE_INPUT_REQUIRED: {label} must be a regular non-symlink file")
    _validate_output_destination(args.output, args.input)
    _validate_output_destination(args.candidate_manifest, args.input)
    if args.candidate_manifest.resolve() in {args.raw_inventory.resolve(), args.output.resolve()}:
        raise SystemExit("CANDIDATE_OUTPUT_REQUIRED: destinations must be disjoint")
    if args.input.stat().st_size != AUTHENTICATED_MODEL_SIZE or sha256_of_file(args.input) != AUTHENTICATED_MODEL_SHA256:
        raise SystemExit("AUTHENTICATED_PAYLOAD_SHA_REQUIRED: candidate bytes do not match fixed VAST evidence")
    document, inventory_sha256 = load_candidate_inventory(args.raw_inventory)
    candidate = build_candidate_streaming_manifest(document, inventory_sha256)
    _validate_cli_values(args.threshold, args.sample_rate, args.hop_ms, args.window_ms, args.n_mels)
    # Validate every source constant and fully construct both payloads before
    # publishing either artifact, so a malformed tail/CLI cannot leave a
    # misleading one-sided candidate.
    _candidate_weights(document)
    candidate_bytes = (json.dumps(candidate, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    gguf_bytes = write_candidate_gguf(
        args.output, document, candidate,
        threshold=args.threshold, sample_rate=args.sample_rate,
        hop_ms=args.hop_ms, window_ms=args.window_ms, n_mels=args.n_mels,
        publish=False,
    )
    _atomic_publish_pair(args.candidate_manifest, candidate_bytes, args.output, gguf_bytes)
    print("Wrote unreviewed candidate GGUF; production authority remains closed", file=sys.stderr)
    return 0


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
    global REVIEWED_TOPOLOGY_SHA256
    # Candidate gates are tested with synthetic documents only.  In particular,
    # no local model or VAST artifact is needed to prove that source/topology/
    # state/options tampering cannot enter the candidate path.
    for tampered in (
        {},
        {"subgraphs": []},
        {"subgraphs": [{"operators": []}, {"operators": []}]},
    ):
        try:
            _candidate_streaming_plan(tampered)
        except (SystemExit, KeyError, TypeError):
            pass
        else:
            raise AssertionError("synthetic candidate topology tamper was accepted")
    with tempfile.TemporaryDirectory(prefix="mww-candidate-gate-") as directory:
        tampered_path = Path(directory) / "tampered.json"
        tampered_path.write_text(
            json.dumps({"format": RAW_INVENTORY_FORMAT, "source_sha256": "0" * 64}),
            encoding="utf-8",
        )
        try:
            load_candidate_inventory(tampered_path)
        except SystemExit as error:
            if "CANDIDATE_RAW_INVENTORY_REQUIRED" not in str(error):
                raise AssertionError(f"wrong candidate source tamper error: {error}")
        else:
            raise AssertionError("synthetic candidate source tamper was accepted")
    saved_vast = os.environ.pop("VOKRA_PUBLISH_ON_VAST", None)
    saved_candidate = os.environ.pop("VOKRA_CANDIDATE_CONVERSION", None)
    try:
        try:
            _validate_candidate_environment()
        except SystemExit as error:
            if "CANDIDATE_VAST_REQUIRED" not in str(error):
                raise AssertionError(f"wrong candidate environment gate: {error}")
        else:
            raise AssertionError("candidate environment gate was bypassed")
    finally:
        if saved_vast is not None:
            os.environ["VOKRA_PUBLISH_ON_VAST"] = saved_vast
        if saved_candidate is not None:
            os.environ["VOKRA_CANDIDATE_CONVERSION"] = saved_candidate
    with tempfile.TemporaryDirectory(prefix="mww-candidate-pair-") as directory:
        pair_dir = Path(directory)
        first = pair_dir / "candidate.json"
        second = pair_dir / "candidate.gguf"
        _atomic_publish_pair(first, b"manifest", second, b"gguf")
        assert first.read_bytes() == b"manifest" and second.read_bytes() == b"gguf"
        race_first = pair_dir / "race.json"
        race_second = pair_dir / "race.gguf"
        race_second.write_bytes(b"sentinel")
        race_first.symlink_to(race_second.name)
        try:
            _atomic_publish_pair(race_first, b"new-manifest", race_second, b"new-gguf")
        except SystemExit:
            pass
        else:
            raise AssertionError("candidate race destination was overwritten")
        assert race_first.is_symlink() and race_second.read_bytes() == b"sentinel"
        race_first.unlink()
        invalid_manifest = {"subgraphs": [{"tensors": [{"shape": [3], "dtype": "int8", "kind": "constant", "data_hex": "00", "buffer_size": 1, "buffer_sha256": hashlib.sha256(b"\x00").hexdigest()}], "operators": []}, {"tensors": [], "operators": []}]}
        try:
            _candidate_weights(invalid_manifest)
        except SystemExit:
            pass
        else:
            raise AssertionError("candidate shape/byte mismatch was accepted")
        candidate_raw = bytes([0x80, 0xFF, 0x00, 0x7F, 0x01, 0xA5])
        candidate_doc = {
            "subgraphs": [{
                "tensors": [{
                    "index": 4,
                    "kind": "constant",
                    "dtype": "int8",
                    "name": "dense",
                    "shape": [3, 2],
                    "data_hex": candidate_raw.hex(),
                    "buffer_size": len(candidate_raw),
                    "buffer_sha256": hashlib.sha256(candidate_raw).hexdigest(),
                    "quantization": {
                        "scales": [0.125], "zero_points": [0],
                        "quantized_dimension": -1,
                    },
                }],
                "operators": [],
            }]}
        candidate_weights, candidate_count = _candidate_weights(candidate_doc)
        if candidate_count != 1 or candidate_weights[0]["shape"] != [3, 2] or candidate_weights[0]["i8_data"] != [-128, -1, 0, 127, 1, -91]:
            raise AssertionError("candidate dense-I8 logical shape or bytes changed")
        bad_output = pair_dir / "bad.gguf"
        bad_manifest = pair_dir / "bad.json"
        try:
            _validate_cli_values(float("nan"), 16000, 10, 32, 40)
        except SystemExit:
            pass
        else:
            raise AssertionError("invalid candidate CLI value was accepted")
        assert not bad_output.exists() and not bad_manifest.exists()
    candidate_source = Path(__file__).read_text(encoding="utf-8")
    candidate_run = candidate_source[candidate_source.index("def _run_candidate"):candidate_source.index("def _self_test_read_string")]
    if not (
        candidate_run.index("_validate_cli_values")
        < candidate_run.index("gguf_bytes = write_candidate_gguf")
        < candidate_run.index("_atomic_publish_pair")
    ):
        raise AssertionError("candidate pair publication occurs before complete validation")
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
        manifest_document = {
                    "format": "vokra-microwakeword-tflite-tensor-manifest-v1",
                    "producer": {"method": "raw_flatbuffer", "name": "microwakeword_tensor_manifest.py", "version": "1.1"},
                    "source_sha256": "0" * 64,
                    "source_size": 4,
                    "complete": True,
                    "subgraph_count": 1,
                    "tensor_count": 2,
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
                            "data_hex": (b"\x00" * 32).hex(),
                            "quantization": {"scales": [1.0], "zero_points": [0], "quantized_dimension": -1},
                        }
                    ],
                    "tensor_contract": [
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
                            "data_hex": (b"\x00" * 32).hex(),
                            "quantization": {"scales": [1.0], "zero_points": [0], "quantized_dimension": -1},
                        },
                        {
                            "index": 1,
                            "name": "activation",
                            "kind": "activation",
                            "dtype": "int8",
                            "type": 9,
                            "shape": [1],
                            "buffer_index": 0,
                            "buffer_size": 0,
                            "buffer_sha256": hashlib.sha256(b"").hexdigest(),
                            "quantization": None,
                        },
                    ],
                    "topology": {
                        "format": "vokra-microwakeword-tflite-topology-v1",
                        "complete": True,
                        "canonical_identity": None,
                        "operator_code_count": 1,
                        "operator_count": 1,
                        "subgraph_inputs": [0],
                        "subgraph_outputs": [1],
                        "operators": [{"index": 0, "opcode_index": 0, "builtin_code": 14, "builtin_name": "LOGISTIC", "version": 1, "inputs": [0], "outputs": [1], "options": {"type": 0}}],
                    },
                }
        manifest_document["topology"]["canonical_digest"] = hashlib.sha256(
            json.dumps(
                {
                    "schema": "vokra-microwakeword-canonical-topology-v1",
                    "tensor_contract": manifest_document["tensor_contract"],
                    "graph_inputs": manifest_document["topology"]["subgraph_inputs"],
                    "graph_outputs": manifest_document["topology"]["subgraph_outputs"],
                    "operators": manifest_document["topology"]["operators"],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest_document), encoding="utf-8")
        manifest = load_tensor_manifest(
            manifest_path,
            sha256_of_file(manifest_path),
            "0" * 64,
            allow_untrusted=True,
        )
        if manifest[0]["name"] != "constant":
            raise AssertionError("authenticated tensor manifest was not loaded")
        extracted, count, skipped = extract_tensors(manifest)
        if count != 1 or skipped != 0 or extracted[0]["i8_data"] != [0] * 32:
            raise AssertionError("raw authenticated constant bytes were not decoded exactly")
        tampered = {0: {**manifest[0], "data_hex": (b"\x01" * 32).hex()}}
        try:
            extract_tensors(tampered)
        except SystemExit:
            pass
        else:
            raise AssertionError("tampered authenticated constant bytes were accepted")
        try:
            load_tensor_manifest(manifest_path, sha256_of_file(manifest_path), "0" * 64)
        except SystemExit as error:
            if str(error) != "AUTHENTICATED_TOPOLOGY_REQUIRED":
                raise AssertionError(f"wrong production topology error: {error}")
        else:
            raise AssertionError("unreviewed topology entered production preparation")
        reviewed_document = json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object)
        reviewed_digest = reviewed_document["topology"]["canonical_digest"]
        reviewed_document["topology"]["canonical_identity"] = reviewed_digest
        manifest_path.write_text(json.dumps(reviewed_document), encoding="utf-8")
        # A matching caller self-stamp remains insufficient when it does not
        # match the compiled closed authority.
        try:
            load_tensor_manifest(manifest_path, sha256_of_file(manifest_path), "0" * 64)
        except SystemExit as error:
            if str(error) != "AUTHENTICATED_TOPOLOGY_REQUIRED":
                raise AssertionError(f"caller self-stamped topology was accepted: {error}")
        else:
            raise AssertionError("caller self-stamped topology unlocked production preparation")
        REVIEWED_TOPOLOGY_SHA256 = reviewed_digest
        assert load_tensor_manifest(
            manifest_path, sha256_of_file(manifest_path), "0" * 64
        )
        reviewed_document["topology"]["canonical_identity"] = None
        manifest_path.write_text(json.dumps(reviewed_document), encoding="utf-8")
        try:
            load_tensor_manifest(manifest_path, sha256_of_file(manifest_path), "0" * 64)
        except SystemExit as error:
            if str(error) != "AUTHENTICATED_TOPOLOGY_REQUIRED":
                raise AssertionError(f"missing canonical identity had wrong error: {error}")
        else:
            raise AssertionError("compiled authority accepted a missing identity")
        reviewed_document["topology"]["canonical_identity"] = reviewed_digest
        manifest_path.write_text(json.dumps(reviewed_document), encoding="utf-8")
        REVIEWED_TOPOLOGY_SHA256 = "f" * 64
        try:
            load_tensor_manifest(manifest_path, sha256_of_file(manifest_path), "0" * 64)
        except SystemExit as error:
            if str(error) != "AUTHENTICATED_TOPOLOGY_REQUIRED":
                raise AssertionError(f"compiled authority mismatch had wrong error: {error}")
        else:
            raise AssertionError("compiled authority mismatch was accepted")
        REVIEWED_TOPOLOGY_SHA256 = "e17fa0cae8d504ce71b49ad2113fc6f7ebba9e74dd4070d26e7f291dcbfaf621"

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
        duplicate_manifest = directory_path / "duplicate-manifest.json"
        duplicate_manifest.write_text('{"source_sha256":"' + "0" * 64 + '","source_sha256":"' + "0" * 64 + '"}', encoding="utf-8")
        try:
            load_tensor_manifest(duplicate_manifest, sha256_of_file(duplicate_manifest), "0" * 64)
        except SystemExit as error:
            if str(error) != "SOURCE_TENSOR_MANIFEST_REQUIRED":
                raise AssertionError(f"wrong duplicate-key error: {error}")
        else:
            raise AssertionError("duplicate manifest key was accepted")

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
        if metadata[KEY_PROV_UPSTREAM_URL] != (8, PROVENANCE_UPSTREAM_URL):
            raise AssertionError("GitHub distributor provenance URL is missing or wrong")
        if any(key == "vokra.provenance.upstream_hf" for key in metadata):
            raise AssertionError("obsolete Hugging Face provenance key was emitted")

        dense_i8_output = directory_path / "dense-i8.gguf"
        dense_i8_values = [-128, -1, 0, 1, 127, -91]
        write_gguf(
            dense_i8_output,
            [{
                "name": "dense_i8_weight",
                "shape": [3, 2],
                "orig_dtype": "int8",
                "i8_data": dense_i8_values,
                "scales": [0.125],
                "zero_points": [0],
                "quantized_dimension": -1,
            }],
            dense_i8=True,
            **writer_args,
        )
        _, dense_i8_tensors, dense_i8_offset = _self_test_parse_gguf(dense_i8_output.read_bytes())
        if dense_i8_tensors != [{
            "name": "dense_i8_weight", "dimensions": [2, 3],
            "dtype": GGML_TYPE_I8, "offset": 0,
        }]:
            raise AssertionError(f"dense I8 shape/tag changed: {dense_i8_tensors}")
        dense_i8_payload = dense_i8_output.read_bytes()[dense_i8_offset : dense_i8_offset + len(dense_i8_values)]
        if dense_i8_payload != struct.pack("<6b", *dense_i8_values):
            raise AssertionError("dense I8 source bytes changed")

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
        description="Extract kahrendt/microWakeWord TFLite → Vokra GGUF dense-I8 sidecar."
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="Exercise and wire-parse a synthetic GGUF without model I/O.",
    )
    ap.add_argument(
        "--candidate",
        action="store_true",
        help="VAST-only, NO_UPLOAD candidate conversion from fixed raw inventory.",
    )
    ap.add_argument(
        "--reviewed",
        action="store_true",
        help="VAST-only reviewed production conversion from the exact raw inventory (NO_UPLOAD).",
    )
    ap.add_argument(
        "--input",
        type=Path,
        help="VAST-materialized canonical .tflite transport path (required).",
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
    ap.add_argument(
        "--raw-inventory",
        type=Path,
        help="Fixed owner-approved raw inventory for --candidate (not production authority).",
    )
    ap.add_argument(
        "--candidate-manifest",
        type=Path,
        help="No-clobber candidate streaming manifest output for --candidate.",
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

    if args.candidate:
        return _run_candidate(args)
    if args.reviewed:
        return _run_reviewed(args)

    if args.output is None:
        raise SystemExit("--output is required unless --self-test is used")
    if args.name != CANONICAL_MODEL_NAME:
        raise SystemExit(
            f"canonical microWakeWord conversion requires --name {CANONICAL_MODEL_NAME!r}"
        )
    if args.input is None:
        raise SystemExit("AUTHENTICATED_PAYLOAD_REQUIRED: --input is required")
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
    tflite_path = args.input
    upstream_url = DEFAULT_UPSTREAM_URL
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

        weights, n_weights, n_activations = extract_tensors(constant_manifest, args.verbose)
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
            dense_i8=True,
        )
        out_size = args.output.stat().st_size
        print(f"Wrote {args.output} ({out_size:,} bytes, {n_weights} tensors, "
              f"vokra.kws.arch={ARCH}, vokra.kws.model={args.name})",
              file=sys.stderr)
        print(f"sha256(output) = {sha256_of_file(args.output)}", file=sys.stderr)
    finally:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
