#!/usr/bin/env -S uv run --no-project --python 3.12 python
"""Dependency-free authenticated TFLite FlatBuffer tensor inventory.

This is intentionally not an interpreter: persistent constants are classified
only from FlatBuffer Tensor.buffer -> Buffer.data ownership, never from a
runtime ``get_tensor`` result. It emits a typed topology and a canonical
evidence digest, but ``canonical_identity`` remains ``None`` until the
owner-reviewed VAST artifact and parity evidence exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import tempfile
from pathlib import Path
from typing import Any

FORMAT = "vokra-microwakeword-tflite-tensor-manifest-v1"
TOPOLOGY_FORMAT = "vokra-microwakeword-tflite-topology-v1"
PRODUCER = {"name": "microwakeword_tensor_manifest.py", "version": "1.0", "method": "raw_flatbuffer"}
DTYPES = {0: ("float32", 4), 1: ("float16", 2), 2: ("int32", 4), 3: ("uint8", 1), 4: ("int64", 8), 5: ("string", 1), 6: ("bool", 1), 7: ("int16", 2), 8: ("complex64", 8), 9: ("int8", 1), 10: ("float64", 8), 11: ("resource", 1), 12: ("variant", 1), 13: ("uint16", 2), 14: ("uint32", 4), 15: ("uint64", 8), 16: ("int4", 1), 17: ("bfloat16", 2)}
SUPPORTED = {"float32", "int32", "int8"}
TFLITE_MODEL_VERSION = 3
# TensorFlow Lite's schema_v3.fbs declares Model.version as ``uint`` without
# an explicit default. FlatBuffers therefore supplies scalar default 0 when
# the slot is omitted; only an explicit value of 3 is accepted here.

# Values are pinned to TensorFlow Lite schema_v3.fbs.  Only the operators whose
# semantics have a first-class typed ChainConfig representation are admitted;
# CUSTOM/delegate/unknown values never enter the manifest as a guessed op.
BUILTIN_OPERATORS = {
    3: "CONV_2D",
    4: "DEPTHWISE_CONV_2D",
    9: "FULLY_CONNECTED",
    14: "LOGISTIC",
    25: "SOFTMAX",
}
BUILTIN_OPTIONS = {
    "CONV_2D": 1,
    "DEPTHWISE_CONV_2D": 2,
    "FULLY_CONNECTED": 8,
    "SOFTMAX": 9,
}
ACTIVATIONS = {0: "NONE", 1: "RELU", 2: "RELU_N1_TO_1", 3: "RELU6", 4: "TANH", 5: "SIGN_BIT"}
PADDINGS = {0: "SAME", 1: "VALID"}


def _canonical_topology_digest(tensor_contract: list[dict[str, Any]], topology: dict[str, Any]) -> str:
    """Hash all binding-relevant tensor, graph, edge, and option fields."""
    payload = {
        "schema": "vokra-microwakeword-canonical-topology-v1",
        "tensor_contract": tensor_contract,
        "graph_inputs": topology["subgraph_inputs"],
        "graph_outputs": topology["subgraph_outputs"],
        "operators": topology["operators"],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class Reader:
    def __init__(self, data: bytes):
        self.data = data
        if len(data) < 8 or data[4:8] != b"TFL3":
            raise ValueError("TFL3 identifier missing")
        self.root = self.u32(0)
        self._check(self.root, 4)

    def _check(self, offset: int, size: int) -> None:
        if offset < 0 or size < 0 or offset + size > len(self.data):
            raise ValueError("FlatBuffer offset is out of bounds")

    def u8(self, offset: int) -> int:
        self._check(offset, 1)
        return self.data[offset]

    def u16(self, offset: int) -> int:
        self._check(offset, 2)
        return struct.unpack_from("<H", self.data, offset)[0]

    def i32(self, offset: int) -> int:
        self._check(offset, 4)
        return struct.unpack_from("<i", self.data, offset)[0]

    def f32(self, offset: int) -> float:
        self._check(offset, 4)
        return struct.unpack_from("<f", self.data, offset)[0]

    def i64(self, offset: int) -> int:
        self._check(offset, 8)
        return struct.unpack_from("<q", self.data, offset)[0]

    def u32(self, offset: int) -> int:
        self._check(offset, 4)
        return struct.unpack_from("<I", self.data, offset)[0]

    def u64(self, offset: int) -> int:
        self._check(offset, 8)
        return struct.unpack_from("<Q", self.data, offset)[0]

    def table_field(self, table: int, field: int, width: int = 4) -> int | None:
        """Return a field address after validating the complete table bounds.

        ``width`` is the schema width of the value at this field (one byte for
        enum/bool, four bytes for scalar/uoffset fields). FlatBuffers permits
        trailing fields to be omitted; a missing vtable slot therefore means
        the schema default, while a present slot must stay within the object.
        """
        if width <= 0:
            raise ValueError("FlatBuffer field width must be positive")
        vtable_distance = self.i32(table)
        if vtable_distance <= 0 or vtable_distance > table:
            raise ValueError("FlatBuffer vtable offset is invalid")
        vtable = table - vtable_distance
        self._check(vtable, 4)
        vsize = self.u16(vtable)
        osize = self.u16(vtable + 2)
        if vsize < 4 or osize < 4:
            raise ValueError("FlatBuffer table sizes are invalid")
        self._check(vtable, vsize)
        self._check(table, osize)
        if vtable + vsize > table:
            raise ValueError("FlatBuffer vtable overlaps table object")
        slot = vtable + 4 + field * 2
        if slot + 2 > vtable + vsize:
            return None
        relative = self.u16(slot)
        if relative == 0:
            return None
        if relative < 4 or relative + width > osize:
            raise ValueError("FlatBuffer field extends beyond table object")
        address = table + relative
        self._check(address, width)
        return address

    def indirect(self, address: int) -> int:
        offset = self.u32(address)
        if offset == 0:
            raise ValueError("FlatBuffer uoffset must be nonzero")
        target = address + offset
        if target <= address:
            raise ValueError("FlatBuffer uoffset must point forward")
        self._check(target, 4)
        return target

    def vector(self, address: int, element_size: int) -> tuple[int, int]:
        target = self.indirect(address)
        length = self.u32(target)
        start = target + 4
        self._check(start, length * element_size)
        return start, length

    def vector_uoffsets(self, address: int) -> list[int]:
        start, length = self.vector(address, 4)
        return [self.indirect(start + index * 4) for index in range(length)]

    def string(self, address: int) -> str:
        target = self.indirect(address)
        length = self.u32(target)
        start = target + 4
        self._check(start, length + 1)
        if self.u8(start + length) != 0:
            raise ValueError("FlatBuffer string is not NUL terminated")
        try:
            return self.data[start:start + length].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("FlatBuffer string is not UTF-8") from error

    def bytes_vector(self, address: int) -> bytes:
        start, length = self.vector(address, 1)
        return self.data[start:start + length]

    def i32_vector(self, address: int) -> list[int]:
        start, length = self.vector(address, 4)
        return [struct.unpack_from("<i", self.data, start + index * 4)[0] for index in range(length)]

    def f32_vector(self, address: int) -> list[float]:
        start, length = self.vector(address, 4)
        return [struct.unpack_from("<f", self.data, start + index * 4)[0] for index in range(length)]

    def i64_vector(self, address: int) -> list[int]:
        start, length = self.vector(address, 8)
        return [struct.unpack_from("<q", self.data, start + index * 8)[0] for index in range(length)]

    def u8_vector(self, address: int) -> list[int]:
        start, length = self.vector(address, 1)
        return list(self.data[start:start + length])


def _optional_vector(reader: Reader, table: int, field: int, element_size: int, kind: str) -> list[Any]:
    address = reader.table_field(table, field, 4)
    if address is None:
        return []
    if kind == "i32":
        return reader.i32_vector(address)
    if kind == "i64":
        return reader.i64_vector(address)
    if kind == "f32":
        return reader.f32_vector(address)
    if kind == "u8":
        return reader.u8_vector(address)
    raise ValueError(f"unsupported vector kind {kind}")


def _tensor_quantization(reader: Reader, tensor: int) -> dict[str, Any] | None:
    address = reader.table_field(tensor, 4, 4)
    if address is None:
        return None
    quant = reader.indirect(address)
    scales = _optional_vector(reader, quant, 2, 4, "f32")
    zero_points = _optional_vector(reader, quant, 3, 8, "i64")
    # QuantizationParameters schema: details_type is field 4 and details is
    # field 5; quantized_dimension is field 6.  It is not field 4.
    details_type_field = reader.table_field(quant, 4, 1)
    details_type = reader.u8(details_type_field) if details_type_field is not None else 0
    details_value_field = reader.table_field(quant, 5, 4)
    if details_type != 0 or details_value_field is not None:
        raise ValueError("TFLite quantization details union is unsupported")
    qdim_field = reader.table_field(quant, 6, 4)
    qdim = reader.i32(qdim_field) if qdim_field is not None else 0
    if len(scales) != len(zero_points):
        raise ValueError("TFLite quantization scale/zero-point lengths differ")
    return {"scales": scales, "zero_points": zero_points, "quantized_dimension": qdim}


def _operator_code(reader: Reader, table: int) -> dict[str, Any]:
    deprecated_field = reader.table_field(table, 0, 1)
    # OperatorCode.builtin_code is a schema enum backed by int32. The legacy
    # deprecated_builtin_code remains a byte, so do not truncate the current
    # field to one byte when authenticating the operator identity.
    builtin_field = reader.table_field(table, 3, 4)
    deprecated = reader.u8(deprecated_field) if deprecated_field is not None else None
    builtin = reader.i32(builtin_field) if builtin_field is not None else None
    if builtin is not None and deprecated is not None and builtin != deprecated:
        raise ValueError("TFLite OperatorCode builtin code aliases disagree")
    code = builtin if builtin is not None else deprecated if deprecated is not None else 0
    custom_field = reader.table_field(table, 1, 4)
    custom = reader.string(custom_field) if custom_field is not None else None
    if custom:
        raise ValueError("custom TFLite operators are unsupported")
    version_field = reader.table_field(table, 2, 4)
    version = reader.i32(version_field) if version_field is not None else 1
    if version != 1:
        raise ValueError(f"unsupported TFLite operator version: {version}")
    name = BUILTIN_OPERATORS.get(code)
    if name is None:
        raise ValueError(f"unsupported TFLite builtin operator code: {code}")
    return {"builtin_code": code, "builtin_name": name, "version": version}


def _option_scalar(reader: Reader, table: int, field: int, width: int, default: Any) -> Any:
    address = reader.table_field(table, field, width)
    if address is None:
        return default
    if width == 1:
        return reader.u8(address)
    if width == 4:
        return reader.i32(address)
    if width == 8:
        return reader.u64(address)
    raise ValueError("unsupported option scalar width")


def _builtin_options(reader: Reader, operator: int, name: str) -> dict[str, Any]:
    type_field = reader.table_field(operator, 3, 1)
    options_type = reader.u8(type_field) if type_field is not None else 0
    expected_type = BUILTIN_OPTIONS.get(name, 0)
    if options_type != expected_type:
        raise ValueError(f"{name} has builtin option type {options_type}, expected {expected_type}")
    options_field = reader.table_field(operator, 4, 4)
    if expected_type == 0:
        if options_field is not None:
            raise ValueError(f"{name} must not carry builtin options")
        return {"type": 0}
    if options_field is None:
        raise ValueError(f"{name} lacks required builtin options")
    options = reader.indirect(options_field)
    if name == "CONV_2D":
        padding = _option_scalar(reader, options, 0, 1, 0)
        stride_w = _option_scalar(reader, options, 1, 4, 1)
        stride_h = _option_scalar(reader, options, 2, 4, 1)
        fused = _option_scalar(reader, options, 3, 1, 0)
        dilation_w = _option_scalar(reader, options, 4, 4, 1)
        dilation_h = _option_scalar(reader, options, 5, 4, 1)
        quantized_bias_type = _option_scalar(reader, options, 6, 1, 2)
        if padding not in PADDINGS or fused not in ACTIVATIONS or stride_w <= 0 or stride_h <= 0 or dilation_w <= 0 or dilation_h <= 0 or quantized_bias_type != 2:
            raise ValueError("invalid Conv2D builtin options")
        return {"type": options_type, "padding": PADDINGS[padding], "stride_w": stride_w, "stride_h": stride_h, "fused_activation": ACTIVATIONS[fused], "dilation_w": dilation_w, "dilation_h": dilation_h, "quantized_bias_type": quantized_bias_type}
    if name == "DEPTHWISE_CONV_2D":
        padding = _option_scalar(reader, options, 0, 1, 0)
        stride_w = _option_scalar(reader, options, 1, 4, 1)
        stride_h = _option_scalar(reader, options, 2, 4, 1)
        depth_multiplier = _option_scalar(reader, options, 3, 4, 1)
        fused = _option_scalar(reader, options, 4, 1, 0)
        dilation_w = _option_scalar(reader, options, 5, 4, 1)
        dilation_h = _option_scalar(reader, options, 6, 4, 1)
        if padding not in PADDINGS or fused not in ACTIVATIONS or stride_w <= 0 or stride_h <= 0 or depth_multiplier <= 0 or dilation_w <= 0 or dilation_h <= 0:
            raise ValueError("invalid DepthwiseConv2D builtin options")
        return {"type": options_type, "padding": PADDINGS[padding], "stride_w": stride_w, "stride_h": stride_h, "depth_multiplier": depth_multiplier, "fused_activation": ACTIVATIONS[fused], "dilation_w": dilation_w, "dilation_h": dilation_h}
    if name == "FULLY_CONNECTED":
        fused = _option_scalar(reader, options, 0, 1, 0)
        weights_format = _option_scalar(reader, options, 1, 1, 0)
        keep_num_dims = _option_scalar(reader, options, 2, 1, 0)
        asymmetric = _option_scalar(reader, options, 3, 1, 0)
        quantized_bias_type = _option_scalar(reader, options, 4, 1, 2)
        if fused not in ACTIVATIONS or weights_format != 0 or keep_num_dims or asymmetric or quantized_bias_type != 2:
            raise ValueError("unsupported FullyConnected options")
        return {"type": options_type, "fused_activation": ACTIVATIONS[fused], "weights_format": weights_format, "keep_num_dims": bool(keep_num_dims), "asymmetric_quantize_inputs": bool(asymmetric), "quantized_bias_type": quantized_bias_type}
    beta_field = reader.table_field(options, 0, 4)
    beta = reader.f32(beta_field) if beta_field is not None else 1.0
    if not math.isfinite(beta) or beta != 1.0:
        raise ValueError("unsupported Softmax beta")
    return {"type": options_type, "beta": beta}


def _shape_size(shape: list[int], label: str) -> int:
    size = 1
    for dimension in shape:
        if dimension <= 0:
            raise ValueError(f"{label} has a non-positive dimension")
        size *= dimension
    return size


def _require_tensor(records: dict[int, dict[str, Any]], index: int, label: str) -> dict[str, Any]:
    if index not in records:
        raise ValueError(f"{label} tensor index {index} is out of bounds")
    return records[index]


def _validate_quantization(record: dict[str, Any], label: str, *, scalar: bool = False, bias: bool = False) -> None:
    quantization = record.get("quantization")
    if not isinstance(quantization, dict):
        raise ValueError(f"{label} lacks TFLite quantization parameters")
    scales = quantization.get("scales")
    zero_points = quantization.get("zero_points")
    qdim = quantization.get("quantized_dimension")
    shape = record["shape"]
    if not isinstance(scales, list) or not isinstance(zero_points, list) or not scales or len(scales) != len(zero_points):
        raise ValueError(f"{label} has incomplete quantization vectors")
    if not isinstance(qdim, int) or (qdim != -1 and (qdim < 0 or qdim >= len(shape))):
        raise ValueError(f"{label} has an invalid quantized axis")
    if any(not isinstance(scale, (float, int)) or not math.isfinite(float(scale)) or float(scale) <= 0.0 for scale in scales):
        raise ValueError(f"{label} has a non-positive or non-finite quantization scale")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in zero_points):
        raise ValueError(f"{label} has a non-integer quantization zero point")
    if len(scales) > 1 and (qdim < 0 or len(scales) != shape[qdim]):
        raise ValueError(f"{label} quantization axis length does not match its shape")
    if scalar and len(scales) != 1:
        raise ValueError(f"{label} requires per-tensor quantization")
    if bias and any(value != 0 for value in zero_points):
        raise ValueError(f"{label} bias zero points must be zero")


def _conv_extent(size: int, kernel: int, stride: int, dilation: int, padding: str, label: str) -> tuple[int, int]:
    effective = (kernel - 1) * dilation + 1
    if padding == "VALID":
        if size < effective:
            raise ValueError(f"{label} VALID convolution has a negative output extent")
        return (size - effective) // stride + 1, 0
    output = (size + stride - 1) // stride
    total = max((output - 1) * stride + effective - size, 0)
    if total % 2:
        raise ValueError(f"{label} SAME padding is asymmetric and unsupported")
    return output, total // 2


def _validate_topology(
    reader: Reader,
    subgraph: int,
    operator_codes: list[dict[str, Any]],
    tensor_records: list[dict[str, Any]],
) -> dict[str, Any]:
    records = {record["index"]: record for record in tensor_records}
    inputs_field = reader.table_field(subgraph, 1, 4)
    outputs_field = reader.table_field(subgraph, 2, 4)
    operators_field = reader.table_field(subgraph, 3, 4)
    if inputs_field is None or outputs_field is None or operators_field is None:
        raise ValueError("TFLite subgraph lacks inputs, outputs, or operators")
    graph_inputs = reader.i32_vector(inputs_field)
    graph_outputs = reader.i32_vector(outputs_field)
    if len(graph_inputs) != 1 or len(graph_outputs) != 1 or any(index < 0 for index in graph_inputs + graph_outputs):
        raise ValueError("canonical microWakeWord topology requires one nonnegative input and output")
    if len(set(graph_inputs + graph_outputs)) != len(graph_inputs + graph_outputs):
        raise ValueError("canonical topology graph boundary has duplicate tensor indices")
    operators = reader.vector_uoffsets(operators_field)
    if not operators:
        raise ValueError("TFLite subgraph has no operators")
    parsed: list[dict[str, Any]] = []
    produced: dict[int, int] = {}
    consumed: dict[int, list[int]] = {}
    for index, operator in enumerate(operators):
        opcode_field = reader.table_field(operator, 0, 4)
        inputs = reader.i32_vector(reader.table_field(operator, 1, 4)) if reader.table_field(operator, 1, 4) is not None else []
        outputs = reader.i32_vector(reader.table_field(operator, 2, 4)) if reader.table_field(operator, 2, 4) is not None else []
        if opcode_field is None or not inputs or not outputs or any(value < 0 for value in inputs + outputs):
            raise ValueError(f"operator {index} has incomplete or negative tensor indices")
        opcode_index = reader.u32(opcode_field)
        if opcode_index >= len(operator_codes):
            raise ValueError(f"operator {index} opcode index is out of bounds")
        code = operator_codes[opcode_index]
        name = code["builtin_name"]
        custom_options_field = reader.table_field(operator, 5, 4)
        if custom_options_field is not None and reader.u8_vector(custom_options_field):
            raise ValueError(f"operator {index} carries unsupported custom options")
        # Operator schema fields 6..13 are custom-options format, mutable
        # inputs, intermediates, external custom payload, secondary builtin
        # options, and debug metadata. Only the schema defaults are admitted.
        custom_options_format_field = reader.table_field(operator, 6, 1)
        if custom_options_format_field is not None and reader.u8(custom_options_format_field) != 0:
            raise ValueError(f"operator {index} has unsupported custom options format")
        mutating_field = reader.table_field(operator, 7, 4)
        if mutating_field is not None and any(reader.u8_vector(mutating_field)):
            raise ValueError(f"operator {index} carries mutating variable inputs")
        intermediates_field = reader.table_field(operator, 8, 4)
        if intermediates_field is not None and reader.i32_vector(intermediates_field):
            raise ValueError(f"operator {index} carries unsupported intermediates")
        large_options_offset = _option_scalar(reader, operator, 9, 8, 0)
        large_options_size = _option_scalar(reader, operator, 10, 8, 0)
        if large_options_offset != 0 or large_options_size != 0:
            raise ValueError(f"operator {index} carries external custom options")
        secondary_type_field = reader.table_field(operator, 11, 1)
        secondary_type = reader.u8(secondary_type_field) if secondary_type_field is not None else 0
        secondary_value_field = reader.table_field(operator, 12, 4)
        if secondary_type != 0 or secondary_value_field is not None:
            raise ValueError(f"operator {index} carries secondary builtin options")
        debug_metadata_index = _option_scalar(reader, operator, 13, 4, -1)
        if debug_metadata_index != -1:
            raise ValueError(f"operator {index} carries unsupported debug metadata")
        options = _builtin_options(reader, operator, name)
        for tensor_index in outputs:
            _require_tensor(records, tensor_index, f"operator {index} output")
            if tensor_index in produced:
                raise ValueError(f"tensor {tensor_index} has multiple producers")
            produced[tensor_index] = index
        for tensor_index in inputs:
            _require_tensor(records, tensor_index, f"operator {index} input")
            consumed.setdefault(tensor_index, []).append(index)
        parsed.append({"index": index, "opcode_index": opcode_index, "builtin_code": code["builtin_code"], "builtin_name": name, "version": code["version"], "inputs": inputs, "outputs": outputs, "options": options})
    if any(len(indices) != 1 for indices in consumed.values()):
        raise ValueError("canonical topology requires a single consumer for every referenced tensor")
    if parsed[0]["inputs"][0] != graph_inputs[0] or parsed[-1]["outputs"][0] != graph_outputs[0]:
        raise ValueError("operator execution order does not match graph boundaries")
    for previous, current in zip(parsed, parsed[1:]):
        if current["inputs"][0] != previous["outputs"][0]:
            raise ValueError("canonical topology contains a branch, skip, or reordered activation")
    for record in tensor_records:
        index = record["index"]
        is_input = index in graph_inputs
        is_output = index in graph_outputs
        if record["kind"] == "constant":
            if index not in consumed or len(consumed[index]) != 1 or index in produced:
                raise ValueError(f"constant tensor {index} is not consumed exactly once")
        elif is_input:
            if index in produced or len(consumed.get(index, [])) != 1:
                raise ValueError(f"graph input tensor {index} has an invalid producer/consumer boundary")
        elif is_output:
            if index not in produced or len(consumed.get(index, [])) != 0:
                raise ValueError(f"graph output tensor {index} has an invalid producer/consumer boundary")
        elif index not in produced or len(consumed.get(index, [])) != 1:
            raise ValueError(f"activation tensor {index} is disconnected from the canonical chain")
    for op in parsed:
        name = op["builtin_name"]
        inputs = op["inputs"]
        output = _require_tensor(records, op["outputs"][0], f"{name} output")
        if name in {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"}:
            if len(inputs) != 3 or op["options"].get("fused_activation") != "NONE":
                raise ValueError(f"{name} requires activation/weight/bias inputs and no fused activation")
            activation = _require_tensor(records, inputs[0], f"{name} activation")
            weight = _require_tensor(records, inputs[1], f"{name} weight")
            bias = _require_tensor(records, inputs[2], f"{name} bias")
            if activation["dtype"] != "int8" or output["dtype"] != "int8" or weight["dtype"] != "int8" or bias["dtype"] != "int32" or weight["kind"] != "constant" or bias["kind"] != "constant":
                raise ValueError(f"{name} has unsupported activation/weight/bias types or ownership")
            _validate_quantization(activation, f"{name} activation", scalar=True)
            _validate_quantization(output, f"{name} output", scalar=True)
            _validate_quantization(weight, f"{name} weight")
            _validate_quantization(bias, f"{name} bias", scalar=True, bias=True)
            if name == "CONV_2D":
                if len(activation["shape"]) != 4 or len(weight["shape"]) != 4 or len(output["shape"]) != 4 or len(bias["shape"]) != 1:
                    raise ValueError("Conv2D requires rank-4 activation/weight/output and rank-1 bias")
                _, in_h, in_w, in_c = activation["shape"]
                out_c, kh, kw, weight_in_c = weight["shape"]
                if weight_in_c != in_c or bias["shape"] != [out_c] or output["shape"][0] != 1:
                    raise ValueError("Conv2D tensor shapes do not agree")
                oh, _ = _conv_extent(in_h, kh, op["options"]["stride_h"], op["options"]["dilation_h"], op["options"]["padding"], "Conv2D height")
                ow, _ = _conv_extent(in_w, kw, op["options"]["stride_w"], op["options"]["dilation_w"], op["options"]["padding"], "Conv2D width")
                if output["shape"] != [1, oh, ow, out_c]:
                    raise ValueError("Conv2D output shape does not match its options")
            elif name == "DEPTHWISE_CONV_2D":
                if len(activation["shape"]) != 4 or len(weight["shape"]) != 4 or len(output["shape"]) != 4 or len(bias["shape"]) != 1:
                    raise ValueError("DepthwiseConv2D requires rank-4 activation/weight/output and rank-1 bias")
                _, in_h, in_w, in_c = activation["shape"]
                first, kh, kw, channels = weight["shape"]
                multiplier = op["options"]["depth_multiplier"]
                if first != 1 or channels != in_c * multiplier or bias["shape"] != [channels] or output["shape"][0] != 1:
                    raise ValueError("DepthwiseConv2D tensor shapes do not agree")
                if multiplier != 1:
                    raise ValueError("DepthwiseConv2D depth_multiplier other than one is unsupported")
                oh, _ = _conv_extent(in_h, kh, op["options"]["stride_h"], op["options"]["dilation_h"], op["options"]["padding"], "DepthwiseConv2D height")
                ow, _ = _conv_extent(in_w, kw, op["options"]["stride_w"], op["options"]["dilation_w"], op["options"]["padding"], "DepthwiseConv2D width")
                if output["shape"] != [1, oh, ow, channels]:
                    raise ValueError("DepthwiseConv2D output shape does not match its options")
            else:
                in_dim = _shape_size(activation["shape"], "FullyConnected input")
                out_dim = _shape_size(weight["shape"], "FullyConnected weight")
                if len(weight["shape"]) != 2 or weight["shape"][1] != in_dim or bias["shape"] != [weight["shape"][0]] or _shape_size(output["shape"], "FullyConnected output") != weight["shape"][0]:
                    raise ValueError("FullyConnected tensor shapes do not agree")
        elif name == "LOGISTIC":
            if len(inputs) != 1 or output["dtype"] != "int8":
                raise ValueError("LOGISTIC requires one int8 input and output")
            activation = _require_tensor(records, inputs[0], "LOGISTIC input")
            if activation["dtype"] != "int8" or _shape_size(activation["shape"], "LOGISTIC input") != _shape_size(output["shape"], "LOGISTIC output"):
                raise ValueError("LOGISTIC input/output shapes do not agree")
            _validate_quantization(activation, "LOGISTIC input", scalar=True)
            _validate_quantization(output, "LOGISTIC output", scalar=True)
        elif name == "SOFTMAX":
            if len(inputs) != 1 or output["dtype"] != "int8":
                raise ValueError("SOFTMAX requires one int8 input and output")
            activation = _require_tensor(records, inputs[0], "SOFTMAX input")
            if activation["dtype"] != "int8" or _shape_size(activation["shape"], "SOFTMAX input") != _shape_size(output["shape"], "SOFTMAX output"):
                raise ValueError("SOFTMAX input/output shapes do not agree")
            _validate_quantization(activation, "SOFTMAX input", scalar=True)
            _validate_quantization(output, "SOFTMAX output", scalar=True)
    return {"format": TOPOLOGY_FORMAT, "complete": True, "canonical_identity": None, "operator_code_count": len(operator_codes), "operator_count": len(parsed), "subgraph_inputs": graph_inputs, "subgraph_outputs": graph_outputs, "operators": parsed}


def parse(data: bytes) -> dict[str, Any]:
    reader = Reader(data)
    model = reader.root
    version_field = reader.table_field(model, 0, 4)
    version = reader.u32(version_field) if version_field is not None else 0
    if version != TFLITE_MODEL_VERSION:
        raise ValueError(f"unsupported TFLite Model schema version: {version}")
    subgraphs_field = reader.table_field(model, 2, 4)
    buffers_field = reader.table_field(model, 4, 4)
    operator_codes_field = reader.table_field(model, 1, 4)
    if subgraphs_field is None or buffers_field is None or operator_codes_field is None:
        raise ValueError("TFLite Model lacks operator codes, subgraphs, or buffers")
    subgraphs = reader.vector_uoffsets(subgraphs_field)
    buffers = reader.vector_uoffsets(buffers_field)
    operator_codes = [_operator_code(reader, table) for table in reader.vector_uoffsets(operator_codes_field)]
    if not operator_codes:
        raise ValueError("TFLite Model has no operator codes")
    if len(subgraphs) != 1:
        raise ValueError(f"unsupported TFLite subgraph count: {len(subgraphs)}")
    if not buffers:
        raise ValueError("TFLite Model must contain the empty buffer-0 sentinel")
    buffer_data: list[bytes] = []
    for buffer in buffers:
        # schema.fbs Buffer: data=[ubyte] field 0, offset:ulong field 1,
        # size:ulong field 2. Only embedded Buffer.data is authenticated.
        data_field = reader.table_field(buffer, 0, 4)
        # An omitted optional vector slot is the schema's null default.  A
        # present slot still carries a FlatBuffers uoffset and must go through
        # `indirect()` so zero/backward offsets cannot be silently reclassified
        # as an empty buffer.
        external_offset = _option_scalar(reader, buffer, 1, 8, 0)
        external_size = _option_scalar(reader, buffer, 2, 8, 0)
        if external_offset != 0 or external_size != 0:
            raise ValueError("external Buffer payloads are unsupported")
        buffer_data.append(reader.bytes_vector(data_field) if data_field is not None else b"")
    if buffer_data[0]:
        raise ValueError("TFLite buffer 0 must be the empty sentinel")
    subgraph = subgraphs[0]
    tensors_field = reader.table_field(subgraph, 0, 4)
    if tensors_field is None:
        raise ValueError("TFLite subgraph lacks tensors")
    tensor_tables = reader.vector_uoffsets(tensors_field)
    tensors: list[dict[str, Any]] = []
    all_tensors: list[dict[str, Any]] = []
    names: set[str] = set()
    ownership: dict[int, list[int]] = {}
    for index, tensor in enumerate(tensor_tables):
        # schema.fbs Tensor: shape/type/buffer/name fields 0..3,
        # quantization field 4, is_variable field 5, sparsity field 6,
        # shape_signature field 7, has_rank field 8, variant_tensors field 9,
        # and external_buffer field 10.
        shape_field = reader.table_field(tensor, 0, 4)
        type_field = reader.table_field(tensor, 1, 1)
        buffer_field = reader.table_field(tensor, 2, 4)
        name_field = reader.table_field(tensor, 3, 4)
        if shape_field is None or name_field is None:
            raise ValueError(f"tensor {index} has missing required fields")
        shape = reader.i32_vector(shape_field)
        if any(dimension < 0 for dimension in shape):
            raise ValueError(f"tensor {index} has ambiguous negative shape")
        signature_field = reader.table_field(tensor, 7, 4)
        if signature_field is not None:
            signature = reader.i32_vector(signature_field)
            if any(dimension < 0 for dimension in signature) or signature != shape:
                raise ValueError(f"tensor {index} has an ambiguous shape signature")
        has_rank_field = reader.table_field(tensor, 8, 1)
        if has_rank_field is not None and reader.u8(has_rank_field) == 0:
            raise ValueError(f"tensor {index} has unknown rank")
        sparsity_field = reader.table_field(tensor, 6, 4)
        if sparsity_field is not None:
            reader.indirect(sparsity_field)
            raise ValueError(f"tensor {index} has unsupported sparsity metadata")
        variant_field = reader.table_field(tensor, 9, 4)
        if variant_field is not None:
            if reader.vector_uoffsets(variant_field):
                raise ValueError(f"tensor {index} has unsupported variant tensors")
            raise ValueError(f"tensor {index} has unsupported variant tensor metadata")
        external_buffer_field = reader.table_field(tensor, 10, 4)
        if external_buffer_field is not None and reader.u32(external_buffer_field) != 0:
            raise ValueError(f"tensor {index} has an external buffer")
        name = reader.string(name_field)
        if not name or name in names:
            raise ValueError(f"tensor {index} has a missing or duplicate name")
        names.add(name)
        # Tensor.type defaults to FLOAT32 (0), and Tensor.buffer defaults to
        # buffer 0 when their vtable slots are omitted by FlatBuffers.
        dtype_code = reader.u8(type_field) if type_field is not None else 0
        if dtype_code not in DTYPES:
            raise ValueError(f"tensor {index} has unsupported dtype code {dtype_code}")
        dtype, item_size = DTYPES[dtype_code]
        buffer_index = reader.u32(buffer_field) if buffer_field is not None else 0
        if buffer_index >= len(buffer_data):
            raise ValueError(f"tensor {index} buffer index is out of bounds")
        variable_field = reader.table_field(tensor, 5, 1)
        if variable_field is not None and reader.u8(variable_field):
            raise ValueError(f"tensor {index} is variable")
        payload = buffer_data[buffer_index]
        quantization = _tensor_quantization(reader, tensor)
        record = {"index": index, "name": name, "type": dtype_code, "dtype": dtype, "shape": shape, "buffer_index": buffer_index, "buffer_size": len(payload), "buffer_sha256": hashlib.sha256(payload).hexdigest(), "kind": "constant" if payload else "activation", "quantization": quantization}
        all_tensors.append(record)
        if not payload:
            continue
        if dtype not in SUPPORTED:
            raise ValueError(f"constant tensor {index} has unsupported dtype {dtype}")
        elements = 1
        for dimension in shape:
            elements *= dimension
        expected = elements * item_size
        if expected != len(payload):
            raise ValueError(f"constant tensor {index} byte size {len(payload)} != shape element size {expected}")
        tensors.append(record)
        ownership.setdefault(buffer_index, []).append(index)
    if not tensors:
        raise ValueError("TFLite model has no nonempty supported constant buffers")
    buffer_ownership = []
    for buffer_index, tensor_indices in sorted(ownership.items()):
        payload = buffer_data[buffer_index]
        buffer_ownership.append({"buffer_index": buffer_index, "tensor_indices": tensor_indices, "tensor_count": len(tensor_indices), "shared": len(tensor_indices) > 1, "buffer_size": len(payload), "buffer_sha256": hashlib.sha256(payload).hexdigest()})
    owned_indices = [index for item in buffer_ownership for index in item["tensor_indices"]]
    if len(owned_indices) != len(set(owned_indices)) or sorted(owned_indices) != sorted(item["index"] for item in tensors):
        raise ValueError("duplicate or incomplete nonempty buffer ownership")
    nonempty = [index for index, payload in enumerate(buffer_data) if payload]
    if any(not item["tensor_indices"] or item["buffer_size"] <= 0 for item in buffer_ownership):
        raise ValueError("invalid empty referenced buffer ownership")
    if any(item["buffer_index"] not in nonempty for item in buffer_ownership):
        raise ValueError("referenced buffer ownership is not nonempty")
    topology = _validate_topology(reader, subgraph, operator_codes, all_tensors)
    topology["canonical_digest"] = _canonical_topology_digest(all_tensors, topology)
    return {"format": FORMAT, "producer": PRODUCER, "complete": True, "source_sha256": hashlib.sha256(data).hexdigest(), "source_size": len(data), "subgraph_count": 1, "tensor_count": len(tensor_tables), "buffer_count": len(buffers), "constant_count": len(tensors), "nonempty_buffer_count": len(nonempty), "referenced_nonempty_buffer_count": len(buffer_ownership), "unreferenced_nonempty_buffer_indices": [index for index in nonempty if index not in ownership], "buffer_ownership": buffer_ownership, "tensors": tensors, "tensor_contract": all_tensors, "topology": topology}


def publish(path: Path, value: dict[str, Any]) -> None:
    if path.parent.is_symlink():
        raise ValueError(f"manifest output exists or is unsafe: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ValueError(f"manifest output exists or is unsafe: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False, mode="w", encoding="utf-8") as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(value, sort_keys=True, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def self_test() -> None:
    # Independently assembled FlatBuffer fixture with a complete one-op
    # FullyConnected chain; the parser does not depend on a TFLite/runtime
    # package.
    def fixture(*, external_buffer: bool = False, external_operator: bool = False) -> bytes:
        out = bytearray(b"\x00\x00\x00\x00TFL3")
        def table(fields: int, size: int) -> int:
            vt = len(out); out.extend(struct.pack("<HH", 4 + fields * 2, size)); out.extend(b"\x00" * (fields * 2))
            tab = len(out); out.extend(struct.pack("<i", tab - vt)); out.extend(b"\x00" * (size - 4))
            for i in range(fields): struct.pack_into("<H", out, vt + 4 + i * 2, 4 + i * 4)
            return tab
        def vector_slots(count: int) -> int:
            start = len(out); out.extend(struct.pack("<I", count)); out.extend(b"\x00" * (4 * count)); return start
        def patch_vector(start: int, targets: list[int]) -> None:
            for i, target in enumerate(targets): struct.pack_into("<I", out, start + 4 + i * 4, target - (start + 4 + i * 4))
        def vector_i32(values: list[int]) -> int:
            start = len(out); out.extend(struct.pack("<I", len(values))); out.extend(struct.pack("<" + "i" * len(values), *values)); return start
        def vector_i32_data(values: list[int]) -> int:
            return vector_i32(values)
        def vector_f32(values: list[float]) -> int:
            start = len(out); out.extend(struct.pack("<I", len(values))); out.extend(struct.pack("<" + "f" * len(values), *values)); return start
        def vector_i64(values: list[int]) -> int:
            start = len(out); out.extend(struct.pack("<I", len(values))); out.extend(struct.pack("<" + "q" * len(values), *values)); return start
        def string(value: str) -> int:
            start = len(out); encoded = value.encode(); out.extend(struct.pack("<I", len(encoded))); out.extend(encoded + b"\x00"); return start
        def field_ptr(tab: int, field: int, target: int) -> None: struct.pack_into("<I", out, tab + 4 + field * 4, target - (tab + 4 + field * 4))
        def omit_field(tab: int, field: int) -> None:
            vtable = tab - struct.unpack_from("<i", out, tab)[0]
            struct.pack_into("<H", out, vtable + 4 + field * 2, 0)
        mtab = table(8, 36); struct.pack_into("<I", out, mtab + 4, TFLITE_MODEL_VERSION)
        operator_code_vector = vector_slots(1)
        sv = vector_slots(1); bv = vector_slots(4 if external_buffer else 3)
        field_ptr(mtab, 1, operator_code_vector); field_ptr(mtab, 2, sv); field_ptr(mtab, 4, bv)
        opcode = table(4, 20); omit_field(opcode, 0); omit_field(opcode, 1); struct.pack_into("<i", out, opcode + 12, 1); struct.pack_into("<B", out, opcode + 16, 9)
        patch_vector(operator_code_vector, [opcode])
        stab = table(5, 24); tv = vector_slots(4); inputs = vector_i32([0]); outputs = vector_i32([3]); field_ptr(stab, 0, tv); field_ptr(stab, 1, inputs); field_ptr(stab, 2, outputs)
        op_vector = vector_slots(1)
        operator = table(14 if external_operator else 8, 60 if external_operator else 36); struct.pack_into("<I", out, operator + 4, 0); op_inputs = vector_i32([0, 1, 2]); op_outputs = vector_i32([3]); field_ptr(operator, 1, op_inputs); field_ptr(operator, 2, op_outputs); struct.pack_into("<B", out, operator + 16, 8); omit_field(operator, 5); omit_field(operator, 6); omit_field(operator, 7)
        if external_operator:
            for field in (8, 10, 11, 12, 13):
                omit_field(operator, field)
            struct.pack_into("<Q", out, operator + 4 + 9 * 4, 1 << 32)
        option = table(4, 20); struct.pack_into("<B", out, option + 4, 0); struct.pack_into("<B", out, option + 8, 0); struct.pack_into("<B", out, option + 12, 0); struct.pack_into("<B", out, option + 16, 0); field_ptr(operator, 4, option)
        patch_vector(op_vector, [operator]); field_ptr(stab, 3, op_vector)
        patch_vector(sv, [stab])
        empty_btab = table(1, 8); omit_field(empty_btab, 0)
        weight_btab = table(1, 8); weight_data = len(out); out.extend(struct.pack("<I", 1) + b"\x01"); field_ptr(weight_btab, 0, weight_data)
        bias_btab = table(1, 8); bias_data = len(out); out.extend(struct.pack("<I", 4) + struct.pack("<i", 0)); field_ptr(bias_btab, 0, bias_data)
        if external_buffer:
            external_btab = table(3, 28)
            struct.pack_into("<Q", out, external_btab + 8, 1 << 32)
            patch_vector(bv, [empty_btab, weight_btab, bias_btab, external_btab])
        else:
            patch_vector(bv, [empty_btab, weight_btab, bias_btab])
        def tensor(name: str, shape_values: list[int], dtype: int, buffer_index: int | None) -> int:
            tab = table(8, 36); shape = vector_i32(shape_values); name_offset = string(name); field_ptr(tab, 0, shape); struct.pack_into("<B", out, tab + 8, dtype); field_ptr(tab, 3, name_offset)
            if buffer_index is None: omit_field(tab, 2)
            else: struct.pack_into("<I", out, tab + 12, buffer_index)
            quant = table(7, 32); scales = vector_f32([1.0]); zero_points = vector_i64([0]); field_ptr(quant, 2, scales); field_ptr(quant, 3, zero_points); omit_field(quant, 4); omit_field(quant, 5); struct.pack_into("<i", out, quant + 28, -1); field_ptr(tab, 4, quant); omit_field(tab, 6); omit_field(tab, 7)
            return tab
        tensor_tables = [tensor("input", [1], 9, None), tensor("weight", [1, 1], 9, 1), tensor("bias", [1], 2, 2), tensor("output", [1], 9, None)]
        patch_vector(tv, tensor_tables)
        struct.pack_into("<I", out, 0, mtab)
        return bytes(out)
    result = parse(fixture())
    assert result["complete"] and result["tensor_count"] == 4 and result["constant_count"] == 2
    assert result["topology"]["canonical_identity"] is None
    assert result["topology"]["canonical_digest"] == _canonical_topology_digest(result["tensor_contract"], result["topology"])
    assert result["topology"]["operators"][0]["builtin_name"] == "FULLY_CONNECTED"
    assert result["tensors"][0]["name"] == "weight"
    try: parse(fixture(external_buffer=True))
    except ValueError: pass
    else: raise AssertionError("high-half Buffer external offset was truncated")
    try: parse(fixture(external_operator=True))
    except ValueError: pass
    else: raise AssertionError("high-half Operator external offset was truncated")
    wrong_details = bytearray(fixture())
    reader = Reader(bytes(wrong_details))
    tensor = reader.vector_uoffsets(reader.table_field(reader.vector_uoffsets(reader.table_field(reader.root, 2, 4))[0], 0, 4))[1]
    quant = reader.indirect(reader.table_field(tensor, 4, 4))
    quant_vtable = quant - struct.unpack_from("<i", wrong_details, quant)[0]
    struct.pack_into("<H", wrong_details, quant_vtable + 4 + 4 * 2, 20)
    struct.pack_into("<B", wrong_details, quant + 20, 1)
    try: parse(bytes(wrong_details))
    except ValueError: pass
    else: raise AssertionError("legacy quantized_dimension field was accepted as details")
    wrong_options = bytearray(fixture())
    reader = Reader(bytes(wrong_options))
    subgraph = reader.vector_uoffsets(reader.table_field(reader.root, 2, 4))[0]
    operator = reader.vector_uoffsets(reader.table_field(subgraph, 3, 4))[0]
    options_type = reader.table_field(operator, 3, 1)
    struct.pack_into("<B", wrong_options, options_type, 10)
    try: parse(bytes(wrong_options))
    except ValueError: pass
    else: raise AssertionError("legacy FullyConnected builtin option ordinal was accepted")
    unknown_operator = bytearray(fixture())
    reader = Reader(bytes(unknown_operator))
    operator_code = reader.vector_uoffsets(reader.table_field(reader.root, 1, 4))[0]
    builtin_field = reader.table_field(operator_code, 3, 4)
    struct.pack_into("<i", unknown_operator, builtin_field, 255)
    try: parse(bytes(unknown_operator))
    except ValueError: pass
    else: raise AssertionError("unknown builtin operator was accepted")
    broken_chain = bytearray(fixture())
    reader = Reader(bytes(broken_chain))
    subgraph = reader.vector_uoffsets(reader.table_field(reader.root, 2, 4))[0]
    operator = reader.vector_uoffsets(reader.table_field(subgraph, 3, 4))[0]
    outputs = reader.table_field(operator, 2, 4)
    output_vector, _ = reader.vector(outputs, 4)
    struct.pack_into("<i", broken_chain, output_vector, 0)
    try: parse(bytes(broken_chain))
    except ValueError: pass
    else: raise AssertionError("negative/reordered operator edge was accepted")
    # The fixture explicitly carries Model.version=3 and deliberately omits
    # Tensor.type and the activation Tensor.buffer slots. Their schema defaults
    # (FLOAT32 and buffer 0) must be applied rather than treated as missing.
    assert result["tensors"][0]["type"] == 9
    assert result["tensors"][0]["buffer_index"] == 1
    omitted_version = bytearray(fixture())
    root = struct.unpack_from("<I", omitted_version, 0)[0]
    vtable = root - struct.unpack_from("<i", omitted_version, root)[0]
    struct.pack_into("<H", omitted_version, vtable + 4, 0)
    try: parse(bytes(omitted_version))
    except ValueError: pass
    else: raise AssertionError("omitted Model.version was treated as schema version 3")
    with tempfile.TemporaryDirectory(prefix="mww-manifest-") as directory:
        path = Path(directory) / "manifest.json"; publish(path, result)
        try: publish(path, result)
        except ValueError: pass
        else: raise AssertionError("manifest clobber accepted")
        assert not list(path.parent.glob("*.tmp"))
    bad = bytearray(fixture()); bad[4:8] = b"BAD!"
    try: parse(bytes(bad))
    except ValueError: pass
    else: raise AssertionError("invalid identifier accepted")
    try: parse(fixture()[:-1])
    except ValueError: pass
    else: raise AssertionError("truncated FlatBuffer accepted")
    malformed_table = bytearray(fixture())
    root = struct.unpack_from("<I", malformed_table, 0)[0]
    vtable = root - struct.unpack_from("<i", malformed_table, root)[0]
    struct.pack_into("<H", malformed_table, vtable, 2)
    try: parse(bytes(malformed_table))
    except ValueError: pass
    else: raise AssertionError("malformed vtable size accepted")
    for distance in (-1, len(malformed_table)):
        malformed_offset = bytearray(fixture())
        root = struct.unpack_from("<I", malformed_offset, 0)[0]
        struct.pack_into("<i", malformed_offset, root, distance)
        try: parse(bytes(malformed_offset))
        except ValueError: pass
        else: raise AssertionError("invalid forward/negative vtable offset accepted")
    wrong_version = bytearray(fixture())
    root = struct.unpack_from("<I", wrong_version, 0)[0]
    vtable = root - struct.unpack_from("<i", wrong_version, root)[0]
    struct.pack_into("<H", wrong_version, vtable + 4, 4)
    struct.pack_into("<I", wrong_version, root + 4, 2)
    try: parse(bytes(wrong_version))
    except ValueError: pass
    else: raise AssertionError("unsupported Model schema version accepted")
    present_zero_data = bytearray(fixture())
    reader = Reader(bytes(present_zero_data))
    buffers_field = reader.table_field(reader.root, 4, 4)
    buffer_vector = reader.vector_uoffsets(buffers_field)
    data_field = reader.table_field(buffer_vector[1], 0, 4)
    struct.pack_into("<I", present_zero_data, data_field, 0)
    try: parse(bytes(present_zero_data))
    except ValueError: pass
    else: raise AssertionError("present-zero Buffer.data uoffset was accepted")
    present_zero_signature = bytearray(fixture())
    reader = Reader(bytes(present_zero_signature))
    tensors_field = reader.table_field(reader.vector_uoffsets(reader.table_field(reader.root, 2, 4))[0], 0, 4)
    tensor = reader.vector_uoffsets(tensors_field)[0]
    tensor_vtable = tensor - struct.unpack_from("<i", present_zero_signature, tensor)[0]
    struct.pack_into("<H", present_zero_signature, tensor_vtable + 4 + 7 * 2, 32)
    struct.pack_into("<I", present_zero_signature, tensor + 4 + 7 * 4, 0)
    try: parse(bytes(present_zero_signature))
    except ValueError: pass
    else: raise AssertionError("present-zero Tensor.shape_signature uoffset was accepted")
    nonempty_sentinel = bytearray(fixture())
    reader = Reader(bytes(nonempty_sentinel))
    model = reader.root
    buffers_field = reader.table_field(model, 4, 4)
    buffer_vector = reader.vector_uoffsets(buffers_field)
    first_buffer = buffer_vector[0]
    second_buffer = buffer_vector[1]
    second_data = reader.table_field(second_buffer, 0, 4)
    data_vector = reader.indirect(second_data)
    # Append a valid forward vector so the mutation isolates the buffer-0
    # sentinel rule rather than being rejected first as a backward uoffset.
    data_vector = len(nonempty_sentinel)
    nonempty_sentinel.extend(struct.pack("<I", 1) + b"x")
    first_data_slot = first_buffer + 4
    first_vtable = first_buffer - struct.unpack_from("<i", nonempty_sentinel, first_buffer)[0]
    struct.pack_into("<H", nonempty_sentinel, first_vtable + 4, 4)
    struct.pack_into("<I", nonempty_sentinel, first_data_slot, data_vector - first_data_slot)
    try: parse(bytes(nonempty_sentinel))
    except ValueError: pass
    else: raise AssertionError("nonempty buffer 0 was accepted")
    for offset in (0, 0xFFFF_FFF0):
        bad_uoffset = bytearray(fixture())
        reader = Reader(bytes(bad_uoffset))
        model = reader.root
        buffers_field = reader.table_field(model, 4, 4)
        vector_start, _ = reader.vector(buffers_field, 4)
        struct.pack_into("<I", bad_uoffset, vector_start, offset)
        try: parse(bytes(bad_uoffset))
        except ValueError: pass
        else: raise AssertionError("invalid FlatBuffer uoffset was accepted")
    with tempfile.TemporaryDirectory(prefix="mww-manifest-link-") as directory:
        root = Path(directory); source = root / "model.tflite"; source.write_bytes(fixture())
        target = root / "manifest.json"; publish(target, parse(source.read_bytes()))
        try: publish(target, parse(source.read_bytes()))
        except ValueError: pass
        else: raise AssertionError("rerun manifest publication clobbered destination")
        link = root / "manifest-link.json"; link.symlink_to(target)
        try: publish(link, parse(source.read_bytes()))
        except ValueError: pass
        else: raise AssertionError("symlink manifest destination accepted")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--input", type=Path); parser.add_argument("--output", type=Path); parser.add_argument("--self-test", action="store_true"); args = parser.parse_args()
    if args.self_test: self_test(); print("microwakeword tensor manifest self-test: PASS"); return 0
    if not args.input or not args.output: parser.error("--input and --output are required")
    if args.input.is_symlink() or not args.input.is_file(): raise SystemExit("authenticated TFLite input must be a regular file")
    if args.output.resolve(strict=False) == args.input.resolve(strict=False): raise SystemExit("manifest output aliases TFLite input")
    publish(args.output, parse(args.input.read_bytes())); return 0


if __name__ == "__main__":
    raise SystemExit(main())
