#!/usr/bin/env python3
"""Dump an independent official X-Codec2 token-to-PCM reference.

The oracle imports ``CodecDecoderVocos`` from the official
``xcodec2==0.1.5`` PyPI package, restores the audited public Vokra GGUF into
those official modules, and calls the upstream FSQ + decoder forward. It never
imports Vokra or mirrors the forward equations.

The released source imports ``RotaryPositionalEmbeddings`` through
``torchtune.__init__``, which also imports unrelated torchao modules. This
tool loads the exact ``torchtune==0.3.1`` position-embedding source directly
after verifying its SHA-256, then exposes that official class at
``torchtune.modules``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader


XCODEC2_VERSION = "0.1.5"
XCODEC2_SDIST_SHA256 = (
    "dc1a73b32090706e65fb73b2469411bc27bb72048677a23b430ab21ad325e45b"
)
DECODER_SOURCE_SHA256 = (
    "8a770d35c4d90a3a82b38869b7b39bd6fab6ab7b2079a44915c7740549f19282"
)
TRANSFORMER_SOURCE_SHA256 = (
    "54786751f363ed6ea510c7a4a13d5c093cd392f79c545ce31dedcd745d6662d0"
)
GGUF_SHA256 = "7ab4b94006068226b0741930081f7e149316e045511c1cddb94769e7f598698e"
TORCHTUNE_VERSION = "0.3.1"
TORCHTUNE_ROPE_SHA256 = (
    "8d79a03e1334fe6ecaff14b1e6a2d554e7e6209c95db058846973de013f92b80"
)
VECTOR_QUANTIZE_VERSION = "1.17.8"
CODEBOOK_SIZE = 65_536
HOP_LENGTH = 320
HIDDEN_DIM = 1_024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def install_official_rope_import() -> None:
    actual_version = importlib.metadata.version("torchtune")
    if actual_version != TORCHTUNE_VERSION:
        raise RuntimeError(
            f"torchtune version {actual_version!r} != {TORCHTUNE_VERSION!r}"
        )
    distribution = importlib.metadata.distribution("torchtune")
    source_path = Path(
        distribution.locate_file("torchtune/modules/position_embeddings.py")
    )
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


def import_official_decoder():
    if importlib.metadata.version("xcodec2") != XCODEC2_VERSION:
        raise RuntimeError("xcodec2 package version mismatch")
    install_official_rope_import()
    module = importlib.import_module("xcodec2.vq.codec_decoder_vocos")
    decoder_source = Path(module.__file__ or "")
    transformer_source = decoder_source.with_name("bs_roformer5.py")
    if sha256_file(decoder_source) != DECODER_SOURCE_SHA256:
        raise RuntimeError("xcodec2 official decoder source SHA-256 mismatch")
    if sha256_file(transformer_source) != TRANSFORMER_SOURCE_SHA256:
        raise RuntimeError("xcodec2 official Transformer source SHA-256 mismatch")
    return module.CodecDecoderVocos


def gguf_tensor(item, expected_shape: torch.Size) -> torch.Tensor:
    if int(item.tensor_type) != 0:
        raise TypeError(f"{item.name}: public X-Codec2 tensor is not F32")
    values = item.data.copy().reshape(-1).astype(np.float32, copy=False)
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


def load_official_modules(gguf_path: Path):
    decoder_class = import_official_decoder()
    decoder = decoder_class(hop_length=HOP_LENGTH)
    fc_post_a = torch.nn.Linear(2_048, HIDDEN_DIM)
    reader = GGUFReader(str(gguf_path))
    by_name = {item.name: item for item in reader.tensors}

    loaded = {}
    defaulted = []
    for name, target in decoder.state_dict().items():
        item = by_name.get(f"generator.{name}")
        if item is None and name.startswith("quantizer.layers."):
            defaulted.append(name)
        elif item is None:
            raise RuntimeError(f"GGUF is missing official decoder tensor {name!r}")
        else:
            loaded[name] = gguf_tensor(item, target.shape)
    incompatible = decoder.load_state_dict(loaded, strict=False)
    if sorted(incompatible.missing_keys) != sorted(defaulted):
        raise RuntimeError(
            f"official decoder missing keys {incompatible.missing_keys!r} != {defaulted!r}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(f"official decoder unexpected keys: {incompatible.unexpected_keys}")

    fc_post_a.load_state_dict(
        {
            "weight": required(by_name, "fc_post_a.weight", fc_post_a.weight.shape),
            "bias": required(by_name, "fc_post_a.bias", fc_post_a.bias.shape),
        },
        strict=True,
    )
    decoder.eval()
    fc_post_a.eval()
    return decoder, fc_post_a, len(loaded), defaulted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--codes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gguf_sha256 = sha256_file(args.gguf)
    if gguf_sha256 != GGUF_SHA256:
        raise RuntimeError(f"GGUF SHA-256 {gguf_sha256} != {GGUF_SHA256}")
    if importlib.metadata.version("vector-quantize-pytorch") != VECTOR_QUANTIZE_VERSION:
        raise RuntimeError("vector-quantize-pytorch version mismatch")

    codes = np.fromfile(args.codes, dtype="<u4")
    if codes.size == 0 or np.any(codes >= CODEBOOK_SIZE):
        raise RuntimeError(f"codes must be non-empty and each below {CODEBOOK_SIZE}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    decoder, fc_post_a, loaded_count, defaulted = load_official_modules(args.gguf)

    code_tensor = torch.from_numpy(codes.astype(np.int64)).reshape(1, 1, -1)
    with torch.inference_mode():
        features = decoder.quantizer.get_output_from_indices(
            code_tensor.transpose(1, 2)
        )
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
        "format": "vokra-xcodec2-reference-v1",
        "oracle": "official xcodec2==0.1.5 CodecDecoderVocos FSQ + forward",
        "source_distribution": "xcodec2==0.1.5",
        "source_distribution_sha256": XCODEC2_SDIST_SHA256,
        "decoder_source_sha256": DECODER_SOURCE_SHA256,
        "transformer_source_sha256": TRANSFORMER_SOURCE_SHA256,
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
