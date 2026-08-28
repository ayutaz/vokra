#!/usr/bin/env python3
"""Prepare the exact Microsoft DNSMOS P.808 + P.835 checkpoint bundle.

This is an offline ONNX parser, not a runtime dependency. It accepts only the
two audited official graphs at ``SOURCE_REVISION``, verifies their complete
file hashes and graph signatures, and emits the 38 F32 initializers consumed by
the strict Rust converter. No ONNX graph is executed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

SOURCE_REVISION = "591184a9fcb2cbdec02520fed81a32bbbf9d73ff"
P808_SHA256 = "9246480c58567bc6affd4200938e77eef49468c8bc7ed3776d109c07456f6e91"
P835_SHA256 = "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd"

P808_OPS = (
    "Unsqueeze", "Transpose", "Conv", "Relu", "MaxPool", "Conv", "Relu",
    "MaxPool", "Conv", "Relu", "Conv", "Relu", "MaxPool", "Conv", "Relu",
    "Transpose", "ReduceMax", "MatMul", "Add", "Relu", "MatMul", "Add",
    "Relu", "MatMul", "Add",
)
P835_OPS = (
    "Slice", "Slice", "Reshape", "Reshape", "Concat", "Transpose", "Transpose",
    "Conv", "Conv", "Transpose", "Transpose", "Mul", "Mul", "Add", "Sqrt",
    "Pow", "Max", "Log", "Div", "Unsqueeze", "Transpose", "Conv", "Relu",
    "Conv", "Relu", "Conv", "Relu", "Conv", "Relu", "MaxPool", "Conv",
    "Relu", "MaxPool", "Conv", "Relu", "MaxPool", "Conv", "Relu",
    "Transpose", "ReduceMax", "MatMul", "Add", "Relu", "MatMul", "Add",
    "Relu", "MatMul", "Add",
)

EXPECTED: dict[str, list[int]] = {
    "p808.conv2d_5/kernel:0": [32, 1, 3, 3],
    "p808.conv2d_5/bias:0": [32],
    "p808.conv2d_6/kernel:0": [32, 32, 3, 3],
    "p808.conv2d_6/bias:0": [32],
    "p808.conv2d_7/kernel:0": [32, 32, 3, 3],
    "p808.conv2d_7/bias:0": [32],
    "p808.conv2d_8/kernel:0": [32, 32, 3, 3],
    "p808.conv2d_8/bias:0": [32],
    "p808.conv2d_9/kernel:0": [64, 32, 3, 3],
    "p808.conv2d_9/bias:0": [64],
    "p808.mos_estimator_small_1/dense_3/MatMul/ReadVariableOp/resource:0": [64, 64],
    "p808.mos_estimator_small_1/dense_3/BiasAdd/ReadVariableOp/resource:0": [64],
    "p808.mos_estimator_small_1/dense_4/MatMul/ReadVariableOp/resource:0": [64, 64],
    "p808.mos_estimator_small_1/dense_4/BiasAdd/ReadVariableOp/resource:0": [64],
    "p808.mos_estimator_small_1/dense_5/MatMul/ReadVariableOp/resource:0": [64, 1],
    "p808.mos_estimator_small_1/dense_5/BiasAdd/ReadVariableOp/resource:0": [1],
    "p835.time2freq/stft-real/kernel:0": [161, 320, 1],
    "p835.time2freq/stft-imag/kernel:0": [161, 320, 1],
    "p835.conv2d/kernel:0": [128, 1, 3, 3],
    "p835.conv2d/bias:0": [128],
    "p835.conv2d_1/kernel:0": [64, 128, 3, 3],
    "p835.conv2d_1/bias:0": [64],
    "p835.conv2d_2/kernel:0": [64, 64, 3, 3],
    "p835.conv2d_2/bias:0": [64],
    "p835.conv2d_3/kernel:0": [32, 64, 3, 3],
    "p835.conv2d_3/bias:0": [32],
    "p835.conv2d_4/kernel:0": [32, 32, 3, 3],
    "p835.conv2d_4/bias:0": [32],
    "p835.conv2d_5/kernel:0": [32, 32, 3, 3],
    "p835.conv2d_5/bias:0": [32],
    "p835.conv2d_6/kernel:0": [64, 32, 3, 3],
    "p835.conv2d_6/bias:0": [64],
    "p835.mos_estimator_logpow/dense/MatMul/ReadVariableOp/resource:0": [64, 128],
    "p835.mos_estimator_logpow/dense/BiasAdd/ReadVariableOp/resource:0": [128],
    "p835.mos_estimator_logpow/dense_1/MatMul/ReadVariableOp/resource:0": [128, 64],
    "p835.mos_estimator_logpow/dense_1/BiasAdd/ReadVariableOp/resource:0": [64],
    "p835.mos_estimator_logpow/dense_3/MatMul/ReadVariableOp/resource:0": [64, 3],
    "p835.mos_estimator_logpow/dense_3/BiasAdd/ReadVariableOp/resource:0": [3],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(
            f"dnsmos_prepare_checkpoint: {path} sha256={actual}, expected {expected} "
            f"from microsoft/DNS-Challenge@{SOURCE_REVISION}"
        )


def graph_signature(model, label: str, expected_ops: tuple[str, ...]) -> None:
    opsets = [(entry.domain, entry.version) for entry in model.opset_import]
    actual_ops = tuple(node.op_type for node in model.graph.node)
    if model.ir_version != 7 or opsets != [("", 12)] or actual_ops != expected_ops:
        raise SystemExit(
            "dnsmos_prepare_checkpoint: "
            f"{label} graph signature drift: ir={model.ir_version}, "
            f"opsets={opsets}, ops={actual_ops}"
        )


def extract(path: Path, prefix: str, expected_hash: str, expected_ops: tuple[str, ...]):
    try:
        import numpy as np
        import onnx
        from onnx import numpy_helper
    except ImportError as error:
        raise SystemExit(
            "dnsmos_prepare_checkpoint: run with `uv run --project tools/parity python ...`"
        ) from error

    require_hash(path, expected_hash)
    model = onnx.load(str(path), load_external_data=False)
    graph_signature(model, prefix.rstrip("."), expected_ops)
    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    for initializer in model.graph.initializer:
        name = prefix + initializer.name
        if name not in EXPECTED:
            continue
        array = numpy_helper.to_array(initializer)
        if array.dtype != np.float32:
            raise SystemExit(
                f"dnsmos_prepare_checkpoint: {name} is {array.dtype}, expected float32"
            )
        shape = list(array.shape)
        if shape != EXPECTED[name]:
            raise SystemExit(
                f"dnsmos_prepare_checkpoint: {name} shape={shape}, expected={EXPECTED[name]}"
            )
        tensors[name] = ("F32", shape, array.tobytes(order="C"))
    return tensors


def write_safetensors(path: Path, tensors) -> None:
    header: dict[str, dict[str, object]] = {}
    payloads: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        dtype, shape, payload = tensors[name]
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(payload)],
        }
        payloads.append(payload)
        offset += len(payload)
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for payload in payloads:
            handle.write(payload)


def self_test() -> None:
    assert len(P808_OPS) == 25
    assert len(P835_OPS) == 48
    assert len(EXPECTED) == 38
    assert sum(name.startswith("p808.") for name in EXPECTED) == 16
    assert sum(name.startswith("p835.") for name in EXPECTED) == 22
    assert all(len(value) == 64 for value in (P808_SHA256, P835_SHA256))
    print("dnsmos_prepare_checkpoint: self-test OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p808", type=Path)
    parser.add_argument("--p835", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.p808 is None or args.p835 is None or args.output is None:
        raise SystemExit(
            "dnsmos_prepare_checkpoint: --p808, --p835, and --output are all required"
        )
    tensors = extract(args.p808, "p808.", P808_SHA256, P808_OPS)
    tensors.update(extract(args.p835, "p835.", P835_SHA256, P835_OPS))
    if set(tensors) != set(EXPECTED):
        missing = sorted(set(EXPECTED) - set(tensors))
        extra = sorted(set(tensors) - set(EXPECTED))
        raise SystemExit(
            f"dnsmos_prepare_checkpoint: manifest mismatch missing={missing}, extra={extra}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_safetensors(args.output, tensors)
    output_sha = sha256(args.output)
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest = {
        "source_revision": SOURCE_REVISION,
        "p808_onnx_sha256": P808_SHA256,
        "p835_onnx_sha256": P835_SHA256,
        "tensor_count": len(tensors),
        "safetensors_sha256": output_sha,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{args.output.name} {output_sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
