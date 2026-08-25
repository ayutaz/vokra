#!/usr/bin/env python3
"""Dump an independent official NeuCodec token-to-PCM reference.

The oracle imports ``CodecDecoderVocos`` from the official Neuphonic source
tree pinned below, restores either audited public Vokra GGUF into those
official modules, and calls the upstream FSQ + decoder forward. It never
imports Vokra or mirrors the forward equations.

The released source imports ``RotaryPositionalEmbeddings`` through
``torchtune.__init__``, which also imports unrelated torchao quantization
modules. To keep the decoder-only oracle isolated, this tool loads the exact
``torchtune==0.3.1`` position-embedding source file directly after verifying
its SHA-256, then exposes that official class at ``torchtune.modules``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader


SOURCE_COMMIT = "ed3e6cd1bdc374ce14a21355e5eee66a777149ce"
BASE_GGUF_SHA256 = "b71d9d7867a4c244562caa2d735e93c9b744c70110c346f3f65e0862e41163fc"
DISTILL_GGUF_SHA256 = "15e60e7e5f7242255b18e1386b26c2a8f872c77a56ca241ee82c8aa5d8b6327f"
TORCHTUNE_VERSION = "0.3.1"
TORCHTUNE_ROPE_SHA256 = "8d79a03e1334fe6ecaff14b1e6a2d554e7e6209c95db058846973de013f92b80"
VECTOR_QUANTIZE_VERSION = "1.17.8"
CODEBOOK_SIZE = 65_536
HOP_LENGTH = 480
HIDDEN_DIM = 1_024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(source: Path) -> None:
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != SOURCE_COMMIT:
        raise RuntimeError(f"official source commit {commit!r} != {SOURCE_COMMIT!r}")


def install_official_rope_import() -> None:
    actual_version = importlib.metadata.version("torchtune")
    if actual_version != TORCHTUNE_VERSION:
        raise RuntimeError(
            f"torchtune version {actual_version!r} != {TORCHTUNE_VERSION!r}"
        )
    distribution = importlib.metadata.distribution("torchtune")
    relative = Path("torchtune/modules/position_embeddings.py")
    source_path = Path(distribution.locate_file(relative))
    if sha256_file(source_path) != TORCHTUNE_ROPE_SHA256:
        raise RuntimeError("torchtune official RoPE source SHA-256 mismatch")
    spec = importlib.util.spec_from_file_location(
        "_vokra_official_torchtune_position_embeddings", source_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official RoPE source {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    torchtune_package = types.ModuleType("torchtune")
    torchtune_modules = types.ModuleType("torchtune.modules")
    torchtune_modules.RotaryPositionalEmbeddings = (
        module.RotaryPositionalEmbeddings
    )
    torchtune_package.modules = torchtune_modules
    sys.modules["torchtune"] = torchtune_package
    sys.modules["torchtune.modules"] = torchtune_modules


def import_official_decoder(source: Path):
    install_official_rope_import()
    package = types.ModuleType("neucodec")
    package.__path__ = [str(source / "neucodec")]
    sys.modules["neucodec"] = package
    module = importlib.import_module("neucodec.codec_decoder_vocos")
    return module.CodecDecoderVocos


def gguf_tensor(item, expected_shape: torch.Size) -> torch.Tensor:
    tensor_type = int(item.tensor_type)
    flat = item.data.copy().reshape(-1)
    if tensor_type == 0:  # GGML_TYPE_F32
        values = flat.astype(np.float32, copy=False)
    elif tensor_type == 1:  # GGML_TYPE_F16
        values = flat.astype(np.float16, copy=False).astype(np.float32)
    elif tensor_type == 30:  # GGML_TYPE_BF16
        words = flat.astype(np.uint16, copy=False)
        values = (words.astype(np.uint32) << 16).view(np.float32)
    else:
        raise TypeError(f"{item.name}: unsupported oracle GGML type {tensor_type}")
    expected_elements = int(np.prod(expected_shape, dtype=np.int64))
    if values.size != expected_elements:
        raise RuntimeError(
            f"{item.name}: {values.size} values != expected {expected_elements}"
        )
    return torch.from_numpy(values.reshape(tuple(expected_shape)).copy())


def required(by_name: dict, name: str, shape: torch.Size) -> torch.Tensor:
    item = by_name.get(name)
    if item is None:
        raise RuntimeError(f"GGUF is missing official inference tensor {name!r}")
    return gguf_tensor(item, shape)


def base_tensor(by_name: dict, name: str, shape: torch.Size) -> torch.Tensor | None:
    if name.startswith("quantizer."):
        if ".layers." in name:
            return None
        return required(by_name, name, shape)
    if name == "backbone.embed.weight" or name == "backbone.embed.bias":
        return required(by_name, name.replace("backbone", "acoustic_decoder", 1), shape)
    if name.startswith("backbone.prior_net.") or name.startswith("backbone.post_net."):
        return required(by_name, name.replace("backbone", "acoustic_decoder", 1), shape)
    if name.startswith("backbone.transformers."):
        fields = name.split(".")
        layer = fields[2]
        suffix = ".".join(fields[3:])
        prefix = f"acoustic_decoder.layers.{layer}"
        if suffix == "att.c_attn.weight":
            parts = [
                required(
                    by_name,
                    f"{prefix}.self_attn.{projection}.weight",
                    torch.Size([HIDDEN_DIM, HIDDEN_DIM]),
                )
                for projection in ("q_proj", "k_proj", "v_proj")
            ]
            combined = torch.cat(parts, dim=0)
            if combined.shape != shape:
                raise RuntimeError(
                    f"combined base qkv shape {tuple(combined.shape)} != {tuple(shape)}"
                )
            return combined
        suffix_map = {
            "att_norm.weight": "input_layernorm.weight",
            "att.c_proj.weight": "self_attn.o_proj.weight",
            "ffn_norm.weight": "post_attention_layernorm.weight",
            "mlp.fc1.weight": "mlp.fc1.weight",
            "mlp.fc2.weight": "mlp.fc2.weight",
        }
        mapped = suffix_map.get(suffix)
        if mapped is None:
            raise RuntimeError(f"unmapped official base decoder tensor {name!r}")
        return required(by_name, f"{prefix}.{mapped}", shape)
    if name.startswith("backbone.final_layer_norm."):
        suffix = name.removeprefix("backbone.final_layer_norm.")
        return required(by_name, f"acoustic_decoder.norm.{suffix}", shape)
    if name.startswith("head.out."):
        suffix = name.removeprefix("head.out.")
        return required(by_name, f"acoustic_decoder.head.linear.{suffix}", shape)
    if name == "head.istft.window":
        return None
    raise RuntimeError(f"unmapped official base decoder tensor {name!r}")


def distill_tensor(by_name: dict, name: str, shape: torch.Size) -> torch.Tensor | None:
    item = by_name.get(f"generator.{name}")
    if item is not None:
        return gguf_tensor(item, shape)
    if name.startswith("quantizer.layers."):
        return None
    raise RuntimeError(f"GGUF is missing official distill decoder tensor {name!r}")


def load_official_modules(source: Path, gguf_path: Path, variant: str):
    decoder_class = import_official_decoder(source)
    decoder = decoder_class(hop_length=HOP_LENGTH)
    fc_post_a = torch.nn.Linear(2_048, HIDDEN_DIM)
    reader = GGUFReader(str(gguf_path))
    by_name = {item.name: item for item in reader.tensors}

    loaded = {}
    defaulted = []
    for name, target in decoder.state_dict().items():
        value = (
            base_tensor(by_name, name, target.shape)
            if variant == "base"
            else distill_tensor(by_name, name, target.shape)
        )
        if value is None:
            defaulted.append(name)
        else:
            loaded[name] = value
    incompatible = decoder.load_state_dict(loaded, strict=False)
    if sorted(incompatible.missing_keys) != sorted(defaulted):
        raise RuntimeError(
            f"official decoder missing keys {incompatible.missing_keys!r} != {defaulted!r}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(f"official decoder unexpected keys: {incompatible.unexpected_keys}")

    fc_prefix = "acoustic_decoder.fc" if variant == "base" else "fc_post_a"
    fc_post_a.load_state_dict(
        {
            "weight": required(by_name, f"{fc_prefix}.weight", fc_post_a.weight.shape),
            "bias": required(by_name, f"{fc_prefix}.bias", fc_post_a.bias.shape),
        },
        strict=True,
    )
    decoder.eval()
    fc_post_a.eval()
    return decoder, fc_post_a, len(loaded), defaulted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--codes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    verify_source(args.source)
    gguf_sha256 = sha256_file(args.gguf)
    variant_by_sha = {
        BASE_GGUF_SHA256: "base",
        DISTILL_GGUF_SHA256: "distill",
    }
    variant = variant_by_sha.get(gguf_sha256)
    if variant is None:
        raise RuntimeError("GGUF SHA-256 does not match either audited public artifact")
    if importlib.metadata.version("vector-quantize-pytorch") != VECTOR_QUANTIZE_VERSION:
        raise RuntimeError("vector-quantize-pytorch version mismatch")

    codes = np.fromfile(args.codes, dtype="<u4")
    if codes.size == 0 or np.any(codes >= CODEBOOK_SIZE):
        raise RuntimeError(f"codes must be non-empty and each below {CODEBOOK_SIZE}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    decoder, fc_post_a, loaded_count, defaulted = load_official_modules(
        args.source, args.gguf, variant
    )

    code_tensor = torch.from_numpy(codes.astype(np.int64)).reshape(1, 1, -1)
    with torch.inference_mode():
        features = decoder.quantizer.get_output_from_indices(code_tensor.transpose(1, 2))
        features = features.transpose(1, 2)
        features = fc_post_a(features.transpose(1, 2)).transpose(1, 2)
        decoded = decoder(features.transpose(1, 2), vq=False)[0]
    expected_shape = (1, 1, int(codes.size) * HOP_LENGTH)
    if tuple(features.shape) != (1, HIDDEN_DIM, int(codes.size)):
        raise RuntimeError(f"unexpected feature shape {tuple(features.shape)}")
    if tuple(decoded.shape) != expected_shape:
        raise RuntimeError(f"unexpected decoded shape {tuple(decoded.shape)}")
    if not bool(torch.isfinite(decoded).all()):
        raise RuntimeError("official decoder emitted non-finite PCM")

    args.output.mkdir(parents=True, exist_ok=True)
    codes_path = args.output / "codes.u32le"
    features_path = args.output / "features.f32"
    pcm_path = args.output / "decoded_pcm.f32"
    np.asarray(codes, dtype="<u4").tofile(codes_path)
    np.asarray(features.cpu().numpy(), dtype="<f4").tofile(features_path)
    np.asarray(decoded.cpu().numpy(), dtype="<f4").tofile(pcm_path)
    manifest = {
        "format": "vokra-neucodec-reference-v1",
        "oracle": "official CodecDecoderVocos FSQ + forward",
        "source_repository": "https://github.com/neuphonic/neucodec",
        "source_commit": SOURCE_COMMIT,
        "variant": variant,
        "gguf_sha256": gguf_sha256,
        "torchtune": TORCHTUNE_VERSION,
        "torchtune_rope_sha256": TORCHTUNE_ROPE_SHA256,
        "vector_quantize_pytorch": VECTOR_QUANTIZE_VERSION,
        "torch": str(torch.__version__),
        "official_state_tensors_loaded": loaded_count + 2,
        "official_defaulted_deterministic_buffers": defaulted,
        "code_count": int(codes.size),
        "feature_shape": list(features.shape),
        "decoded_shape": list(decoded.shape),
        "files": {
            path.name: sha256_file(path)
            for path in (codes_path, features_path, pcm_path)
        },
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
