#!/usr/bin/env python3
"""PyTorch references for the remaining official TTS checkpoint consumers.

Usage (repository-managed Python only):

    uv run --with torch --with gguf \
      tools/parity/runtime_gap_tts_consumer_dump_reference.py \
      --model voxcpm2 --gguf /path/to/voxcpm-0.5b.gguf

The script executes PyTorch ``F.linear`` / ``F.embedding``. It does not mirror
the Rust loops. GGUF types F32 and BF16 are widened exactly as specified by
GGML; tensor names are the verbatim official state-dict names.

Pinned Vokra artifacts (revision, file SHA-256) used on 2026-08-21:

* CosyVoice3: ``37e7d22a665d96dd7eb2e10e43ff4571783670cc``,
  ``d581891f7b25f8b3da80a73b750098108f065f03421e23acf0722f716c3cc84f``;
* Dia: ``dd1df2a129fed7d15c365caeabaae227ccfe8537``,
  ``a90733e9e6806cae66abf3eca1d575ecf6dab9298c07d39fc4217a509c952a6d``;
* VibeVoice: ``dec190628f58928fc247b1205b9da2dabc58b9da``,
  ``8ef5f259dfab0b048151ce52d27468040f72b35b6909528e6db7fbb332ccaeac``;
* VoxCPM: ``ee0ca6d6728c947ecf170e6711bdfbd6decaf0d5``,
  ``2c5c3b2509368db3545ea44e66ddd3ef5050ceacd5b5a431d8d8acf1300c6cce``;
* Zonos: ``b1bf5c56d470eb9097e9b04f9deca364576574ba``,
  ``12d542bd219f7f31c91b893810d85b0d810285e603029c69fbd19fd3c7da2c5c``.

gguf-py eagerly constructs views for every tensor and is unusually slow on
the CosyVoice3 pass-through file. For that model, ``--weight-f32`` and
``--bias-f32`` accept little-endian f32 dumps of the two named tensors from a
first-party GGUF reader; PyTorch still independently evaluates the operator.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class LinearSpec:
    weight: str
    bias: str
    input_dim: int
    output_dim: int


LINEAR_SPECS = {
    "cosyvoice3": LinearSpec(
        "llm.model.model.layers.0.self_attn.q_proj.weight",
        "llm.model.model.layers.0.self_attn.q_proj.bias",
        896,
        896,
    ),
    "vibevoice": LinearSpec(
        "model.acoustic_connector.fc1.weight",
        "model.acoustic_connector.fc1.bias",
        64,
        1536,
    ),
    "voxcpm2": LinearSpec("stop_proj.weight", "stop_proj.bias", 1024, 1024),
    "zonos": LinearSpec(
        "prefix_conditioner.conditioners.1.project.weight",
        "prefix_conditioner.conditioners.1.project.bias",
        128,
        2048,
    ),
}


def reader(path: str):
    import gguf.gguf_reader as gguf_reader

    original = gguf_reader.quant_shape_to_byte_shape

    def scalar_safe(shape, tensor_type):
        if len(shape) == 0 and int(tensor_type) == 30:
            return np.array([2], dtype=np.int64)
        return original(shape, tensor_type)

    gguf_reader.quant_shape_to_byte_shape = scalar_safe
    return gguf_reader.GGUFReader(path)


def gguf_tensor(model_reader, name: str) -> torch.Tensor:
    item = next((tensor for tensor in model_reader.tensors if tensor.name == name), None)
    if item is None:
        raise KeyError(f"missing tensor {name!r}")
    shape = tuple(map(int, item.shape))
    flat = item.data.copy().reshape(-1)
    if int(item.tensor_type) == 0:  # GGML F32
        values = flat.astype(np.float32, copy=False)
    elif int(item.tensor_type) == 30:  # GGML BF16
        bits = flat.view(np.uint16).astype(np.uint32) << 16
        values = bits.view(np.float32)
    else:
        raise TypeError(f"{name}: unsupported GGML type {item.tensor_type}")
    return torch.from_numpy(values.reshape(shape))


def fixed_input(dimension: int) -> torch.Tensor:
    indices = np.arange(dimension, dtype=np.float32)
    values = (
        np.sin((indices + np.float32(0.25)) * np.float32(0.071))
        * np.float32(0.2)
    )
    return torch.from_numpy(values)


def print_prefix(output: torch.Tensor, count: int) -> None:
    flat = output.reshape(-1)
    print(", ".join(format(float(value), ".9g") for value in flat[:count]))
    print(
        f"count={flat.numel()} min={float(flat.min()):.9g} "
        f"max={float(flat.max()):.9g} finite={bool(flat.isfinite().all())}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[*LINEAR_SPECS, "dia"], required=True)
    parser.add_argument("--gguf")
    parser.add_argument("--weight-f32")
    parser.add_argument("--bias-f32")
    parser.add_argument("--prefix", type=int, default=32)
    args = parser.parse_args()

    if args.model == "dia":
        if not args.gguf:
            parser.error("dia requires --gguf")
        weight = gguf_tensor(reader(args.gguf), "encoder.embedding.weight")
        output = functional.embedding(torch.tensor([1, 42, 255]), weight)
        print_prefix(output, args.prefix)
        return

    spec = LINEAR_SPECS[args.model]
    if args.weight_f32 or args.bias_f32:
        if not args.weight_f32 or not args.bias_f32:
            parser.error("raw mode requires both --weight-f32 and --bias-f32")
        weight = torch.from_numpy(
            np.fromfile(args.weight_f32, dtype="<f4").reshape(
                spec.output_dim, spec.input_dim
            )
        )
        bias = torch.from_numpy(np.fromfile(args.bias_f32, dtype="<f4"))
    else:
        if not args.gguf:
            parser.error("linear GGUF mode requires --gguf")
        model_reader = reader(args.gguf)
        weight = gguf_tensor(model_reader, spec.weight)
        bias = gguf_tensor(model_reader, spec.bias).reshape(-1)
    output = functional.linear(fixed_input(spec.input_dim), weight, bias)
    print_prefix(output, args.prefix)


if __name__ == "__main__":
    main()
