#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=1.26",
#     "onnx>=1.17",
#     "onnxruntime>=1.19",
#     "safetensors>=0.4",
# ]
# ///
"""Prepare and independently exercise the official TEN-VAD v1.0 ONNX release.

The runtime never loads ONNX or the upstream shared library.  This offline
Python 3.12 bridge validates the exact v1.0-ONNX graph, rewrites its 19 float
initializers to stable Vokra tensor names, and can produce two independent
references:

* the official ONNX network driven by deterministic 3x41 features; and
* the official prebuilt TEN-VAD C ABI driven by deterministic PCM16 frames.

Pinned primary source: ``TEN-framework/ten-vad`` commit
``8e96899ba05a8e8c0e883ec7417e7a144bd9dec0`` (tag ``v1.0-ONNX``).
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np

UPSTREAM_REVISION = "8e96899ba05a8e8c0e883ec7417e7a144bd9dec0"
ONNX_SHA256 = "e10b98a0cab1c98e847fbdda14cb3d45a38336d47535a3f63a0fb6c4e0f4cdf4"
SAMPLE_RATE = 16_000
HOP_SIZE = 256
N_FEATURES = 41
CONTEXT_FRAMES = 3
HIDDEN_DIM = 64
TENSOR_COUNT = 19


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_spec() -> dict[str, tuple[str, tuple[int, ...]]]:
    return {
        "ten_vad.conv0.depthwise.weight": ("const_fold_opt__178", (1, 1, 3, 3)),
        "ten_vad.conv0.pointwise.weight": (
            "StatefulPartitionedCall/vad_model/separable_conv2d/separable_conv2d/ReadVariableOp_1:0",
            (16, 1, 1, 1),
        ),
        "ten_vad.conv0.pointwise.bias": (
            "StatefulPartitionedCall/vad_model/separable_conv2d/BiasAdd/ReadVariableOp:0",
            (16,),
        ),
        "ten_vad.conv1.depthwise.weight": ("const_fold_opt__179", (16, 1, 1, 3)),
        "ten_vad.conv1.pointwise.weight": (
            "StatefulPartitionedCall/vad_model/separable_conv1d/ExpandDims_2:0",
            (16, 16, 1, 1),
        ),
        "ten_vad.conv1.pointwise.bias": (
            "StatefulPartitionedCall/vad_model/separable_conv1d/BiasAdd/ReadVariableOp:0",
            (16,),
        ),
        "ten_vad.conv2.depthwise.weight": ("const_fold_opt__180", (16, 1, 1, 3)),
        "ten_vad.conv2.pointwise.weight": (
            "StatefulPartitionedCall/vad_model/separable_conv1d_1/ExpandDims_2:0",
            (16, 16, 1, 1),
        ),
        "ten_vad.conv2.pointwise.bias": (
            "StatefulPartitionedCall/vad_model/separable_conv1d_1/BiasAdd/ReadVariableOp:0",
            (16,),
        ),
        "ten_vad.lstm0.weight_ih": ("W0__70", (1, 256, 80)),
        "ten_vad.lstm0.weight_hh": ("R0__71", (1, 256, 64)),
        "ten_vad.lstm0.bias": ("B0__72", (1, 512)),
        "ten_vad.lstm1.weight_ih": ("W0__99", (1, 256, 64)),
        "ten_vad.lstm1.weight_hh": ("R0__100", (1, 256, 64)),
        "ten_vad.lstm1.bias": ("B0__101", (1, 512)),
        "ten_vad.dense0.weight": (
            "StatefulPartitionedCall/vad_model/dense_3/Tensordot/ReadVariableOp:0",
            (128, 32),
        ),
        "ten_vad.dense0.bias": (
            "StatefulPartitionedCall/vad_model/dense_3/BiasAdd/ReadVariableOp:0",
            (32,),
        ),
        "ten_vad.dense1.weight": (
            "StatefulPartitionedCall/vad_model/dense_5/Tensordot/ReadVariableOp:0",
            (32, 1),
        ),
        "ten_vad.dense1.bias": (
            "StatefulPartitionedCall/vad_model/dense_5/BiasAdd/ReadVariableOp:0",
            (1,),
        ),
    }


def load_graph(path: Path):
    import onnx

    if sha256(path) != ONNX_SHA256:
        raise SystemExit(
            f"TEN-VAD ONNX SHA-256 is {sha256(path)}, expected {ONNX_SHA256}"
        )
    model = onnx.load(path)
    inputs = [
        (item.name, tuple(dim.dim_value for dim in item.type.tensor_type.shape.dim))
        for item in model.graph.input
    ]
    outputs = [
        (item.name, tuple(dim.dim_value for dim in item.type.tensor_type.shape.dim))
        for item in model.graph.output
    ]
    if [name for name, _ in inputs] != [
        "input_1",
        "input_2",
        "input_3",
        "input_6",
        "input_7",
    ]:
        raise SystemExit(f"TEN-VAD ONNX input manifest drifted: {inputs}")
    if [name for name, _ in outputs] != [
        "output_1",
        "output_2",
        "output_3",
        "output_6",
        "output_7",
    ]:
        raise SystemExit(f"TEN-VAD ONNX output manifest drifted: {outputs}")
    return model


def extract_weights(onnx_path: Path) -> dict[str, np.ndarray]:
    from onnx import numpy_helper

    graph = load_graph(onnx_path).graph
    initializers = {
        item.name: numpy_helper.to_array(item) for item in graph.initializer
    }
    output: dict[str, np.ndarray] = {}
    used: set[str] = set()
    for canonical, (source, shape) in tensor_spec().items():
        if source not in initializers:
            raise SystemExit(f"official ONNX is missing initializer {source!r}")
        value = np.asarray(initializers[source])
        if value.shape != shape:
            raise SystemExit(
                f"initializer {source!r} has shape {value.shape}, expected {shape}"
            )
        if value.dtype != np.float32:
            raise SystemExit(
                f"initializer {source!r} is {value.dtype}, expected float32"
            )
        output[canonical] = np.ascontiguousarray(value)
        used.add(source)
    float_initializers = {
        name
        for name, value in initializers.items()
        if np.issubdtype(np.asarray(value).dtype, np.floating)
    }
    if used != float_initializers:
        raise SystemExit(
            "official ONNX float-initializer manifest drifted: "
            f"unmapped={sorted(float_initializers - used)}, stale={sorted(used - float_initializers)}"
        )
    if len(output) != TENSOR_COUNT:
        raise SystemExit(
            f"canonical bundle has {len(output)} tensors, expected {TENSOR_COUNT}"
        )
    return output


def deterministic_features(step: int) -> np.ndarray:
    index = np.arange(CONTEXT_FRAMES * N_FEATURES, dtype=np.float32)
    values = np.float32(0.37) * np.sin(
        index * np.float32(0.113) + np.float32(step) * np.float32(0.07)
    ) + np.float32(0.19) * np.cos(
        index * np.float32(0.037) - np.float32(step) * np.float32(0.11)
    )
    return np.ascontiguousarray(values.reshape(1, CONTEXT_FRAMES, N_FEATURES))


def onnx_reference(onnx_path: Path, steps: int) -> dict[str, object]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    states = [np.zeros((1, HIDDEN_DIM), dtype=np.float32) for _ in range(4)]
    probabilities: list[float] = []
    features: list[list[list[float]]] = []
    for step in range(steps):
        frame_features = deterministic_features(step)
        result = session.run(
            None,
            {
                "input_1": frame_features,
                "input_2": states[0],
                "input_3": states[1],
                "input_6": states[2],
                "input_7": states[3],
            },
        )
        probabilities.append(float(np.asarray(result[0]).reshape(-1)[0]))
        states = [np.ascontiguousarray(value, dtype=np.float32) for value in result[1:]]
        features.append(frame_features[0].tolist())
    return {
        "features": features,
        "probabilities": probabilities,
        "final_states": [state.reshape(-1).tolist() for state in states],
    }


def deterministic_pcm16(frames: int) -> np.ndarray:
    count = frames * HOP_SIZE
    index = np.arange(count, dtype=np.float64)
    time = index / SAMPLE_RATE
    envelope = np.minimum(1.0, index / 500.0) * np.minimum(1.0, (count - index) / 700.0)
    signal = envelope * (
        0.43 * np.sin(2.0 * np.pi * 173.0 * time)
        + 0.21 * np.sin(2.0 * np.pi * 487.0 * time + 0.31)
        + 0.08 * np.sin(2.0 * np.pi * (91.0 + 17.0 * time) * time)
    )
    return np.rint(np.clip(signal, -1.0, 1.0) * 32767.0).astype(np.int16)


def official_library_reference(library_path: Path, frames: int) -> dict[str, object]:
    class AedAllocation(ctypes.Structure):
        """Stable first two fields of the public-source ``Aed_St``."""

        _fields_ = [
            ("dynamic_memory", ctypes.c_void_p),
            ("dynamic_memory_size", ctypes.c_size_t),
        ]

    # `AUP_Aed_dynamMemPrepare` allocates, in order: two 512-float FIFOs,
    # 1024-float complex spectrum, aligned 513-float power spectrum, then the
    # 3*41 feature context. The power block starts at byte 8192 and the first
    # four aligned blocks occupy exactly 10248 bytes. Reading through this
    # allocation contract avoids depending on private C++ member/padding layout
    # while still using the official implementation as the independent
    # frontend oracle.
    bin_power_offset = 8_192
    feature_context_offset = 10_248

    library = ctypes.CDLL(str(library_path))
    handle = ctypes.c_void_p()
    library.ten_vad_create.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_float,
    ]
    library.ten_vad_create.restype = ctypes.c_int
    library.ten_vad_process.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int16),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
    ]
    library.ten_vad_process.restype = ctypes.c_int
    library.ten_vad_destroy.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    library.ten_vad_destroy.restype = ctypes.c_int
    if library.ten_vad_create(ctypes.byref(handle), HOP_SIZE, ctypes.c_float(0.5)) != 0:
        raise SystemExit("official ten_vad_create failed")
    pcm = deterministic_pcm16(frames)
    probabilities: list[float] = []
    flags: list[int] = []
    features: list[list[float]] = []
    bin_powers: list[list[float]] = []
    pitches_hz: list[float] = []
    try:
        for frame in pcm.reshape(frames, HOP_SIZE):
            probability = ctypes.c_float()
            flag = ctypes.c_int()
            ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
            if (
                library.ten_vad_process(
                    handle, ptr, HOP_SIZE, ctypes.byref(probability), ctypes.byref(flag)
                )
                != 0
            ):
                raise SystemExit("official ten_vad_process failed")
            probabilities.append(float(probability.value))
            flags.append(int(flag.value))
            allocation = ctypes.cast(handle, ctypes.POINTER(AedAllocation)).contents
            feature_bytes = CONTEXT_FRAMES * N_FEATURES * ctypes.sizeof(ctypes.c_float)
            if allocation.dynamic_memory_size < feature_context_offset + feature_bytes:
                raise SystemExit(
                    "official Aed_St dynamic allocation is too small for the pinned "
                    f"feature-context contract: {allocation.dynamic_memory_size}"
                )
            feature_pointer = ctypes.cast(
                allocation.dynamic_memory + feature_context_offset,
                ctypes.POINTER(ctypes.c_float),
            )
            feature_view = np.ctypeslib.as_array(
                feature_pointer, shape=(CONTEXT_FRAMES * N_FEATURES,)
            )
            feature_copy = np.asarray(feature_view, dtype=np.float32).copy()
            features.append(feature_copy.tolist())
            bin_power_pointer = ctypes.cast(
                allocation.dynamic_memory + bin_power_offset,
                ctypes.POINTER(ctypes.c_float),
            )
            bin_power_view = np.ctypeslib.as_array(bin_power_pointer, shape=(513,))
            bin_powers.append(
                np.asarray(bin_power_view, dtype=np.float32).copy().tolist()
            )
            pitches_hz.append(
                float(
                    feature_copy[-1] * np.float32(115.2136917114)
                    + np.float32(92.35690307617)
                )
            )
    finally:
        if library.ten_vad_destroy(ctypes.byref(handle)) != 0:
            raise SystemExit("official ten_vad_destroy failed")
    return {
        "pcm_i16": pcm.astype("<i2", copy=False).tolist(),
        "probabilities": probabilities,
        "flags": flags,
        "features": features,
        "bin_powers": bin_powers,
        "pitches_hz": pitches_hz,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--output-st", required=True, type=Path)
    parser.add_argument("--output-ref", required=True, type=Path)
    parser.add_argument("--official-library", type=Path)
    parser.add_argument("--network-steps", type=int, default=4)
    parser.add_argument("--pcm-frames", type=int, default=40)
    args = parser.parse_args()
    if not args.onnx.is_file():
        parser.error(f"--onnx is not a regular file: {args.onnx}")
    if args.official_library is not None and not args.official_library.is_file():
        parser.error(
            f"--official-library is not a regular file: {args.official_library}"
        )
    for output in (args.output_st, args.output_ref):
        if output.exists():
            parser.error(f"refusing to overwrite output: {output}")
    if args.network_steps <= 0 or args.pcm_frames <= 0:
        parser.error("--network-steps and --pcm-frames must be positive")

    from safetensors.numpy import save_file

    tensors = extract_weights(args.onnx)
    save_file(tensors, str(args.output_st))
    payload: dict[str, object] = {
        "upstream_revision": UPSTREAM_REVISION,
        "onnx_sha256": sha256(args.onnx),
        "safetensors_sha256": sha256(args.output_st),
        "sample_rate": SAMPLE_RATE,
        "hop_size": HOP_SIZE,
        "n_features": N_FEATURES,
        "context_frames": CONTEXT_FRAMES,
        "hidden_dim": HIDDEN_DIM,
        "network": onnx_reference(args.onnx, args.network_steps),
    }
    if args.official_library is not None:
        payload["official_library_sha256"] = sha256(args.official_library)
        payload["stream"] = official_library_reference(
            args.official_library, args.pcm_frames
        )
    args.output_ref.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"ten-vad prepared: tensors={len(tensors)}, network_steps={args.network_steps}, "
        f"stream_frames={args.pcm_frames if args.official_library is not None else 0}, "
        f"safetensors_sha256={payload['safetensors_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
