#!/usr/bin/env python3
"""Audit the exact public Parakeet-TDT-1.1B GGUF tensor contract.

The public artifact is a 4.28 GB F32 checkpoint with 1,667 tensors. This tool
derives every expected name and shape from the pinned NeMo topology and checks
the GGUF header without decoding tensor payloads.

Run through the repository Python policy:

    uv run --no-project --python 3.12 python \
      tools/audit/parakeet_tdt_1_1b_manifest.py MODEL.gguf

``--header-only`` accepts an HTTP range prefix containing the complete GGUF
header. It authenticates the metadata and full name/shape/type/offset manifest,
but deliberately does not claim full-file SHA-256 verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


UPSTREAM_HF = "nvidia/parakeet-tdt-1.1b"
UPSTREAM_REVISION = "53276c6469d1f17a1352e30c4d11be3d0d7e9575"
UPSTREAM_NEMO = "parakeet-tdt-1.1b.nemo"
UPSTREAM_NEMO_BYTES = 4_283_136_000
UPSTREAM_NEMO_SHA256 = (
    "9c563d52bdffeacbac0c5b894fdea9be82fea3a6bd8bb8018ff57888e2b5d988"
)

PUBLIC_HF = "vokra/parakeet-tdt-1.1b"
PUBLIC_REVISION = "3bc0fb6f33204d39c8a76fcd1a7dd987f3662192"
PUBLIC_GGUF = "parakeet-tdt-1.1b.gguf"
PUBLIC_GGUF_BYTES = 4_282_300_800
PUBLIC_GGUF_SHA256 = (
    "5abd74cdffd5795b69b808f3c2164687cb906c181dfe19bf5e76bf3cad82126a"
)

TENSOR_COUNT = 1_667
PARAMETER_COUNT = 1_070_542_950
MANIFEST_SHA256 = (
    "988016b3f7f7562d9fd1f179b6784c6fe6d2fdf0acdbf3184e4428687ca139f5"
)
GGML_F32 = 0

EXPECTED_METADATA = {
    "vokra.model.arch": "parakeet-tdt-1_1b",
    "vokra.model.name": "parakeet-tdt-1.1b",
    "vokra.model.category": "asr",
    "vokra.provenance.upstream_hf": UPSTREAM_HF,
    "vokra.provenance.license": "cc-by-4.0",
    "vokra.provenance.weight_license": "attribution-required",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_manifest() -> dict[str, tuple[int, ...]]:
    manifest: dict[str, tuple[int, ...]] = {
        "preprocessor.featurizer.fb": (1, 80, 257),
        "preprocessor.featurizer.window": (400,),
        "encoder.pre_encode.conv.0.bias": (256,),
        "encoder.pre_encode.conv.0.weight": (256, 1, 3, 3),
        "encoder.pre_encode.conv.2.bias": (256,),
        "encoder.pre_encode.conv.2.weight": (256, 1, 3, 3),
        "encoder.pre_encode.conv.3.bias": (256,),
        "encoder.pre_encode.conv.3.weight": (256, 256, 1, 1),
        "encoder.pre_encode.conv.5.bias": (256,),
        "encoder.pre_encode.conv.5.weight": (256, 1, 3, 3),
        "encoder.pre_encode.conv.6.bias": (256,),
        "encoder.pre_encode.conv.6.weight": (256, 256, 1, 1),
        "encoder.pre_encode.out.bias": (1_024,),
        "encoder.pre_encode.out.weight": (1_024, 2_560),
    }

    for layer in range(42):
        prefix = f"encoder.layers.{layer}"
        for branch in ("feed_forward1", "feed_forward2"):
            manifest.update(
                {
                    f"{prefix}.{branch}.linear1.bias": (4_096,),
                    f"{prefix}.{branch}.linear1.weight": (4_096, 1_024),
                    f"{prefix}.{branch}.linear2.bias": (1_024,),
                    f"{prefix}.{branch}.linear2.weight": (1_024, 4_096),
                }
            )
        for norm in (
            "norm_feed_forward1",
            "norm_self_att",
            "norm_conv",
            "norm_feed_forward2",
            "norm_out",
        ):
            manifest[f"{prefix}.{norm}.bias"] = (1_024,)
            manifest[f"{prefix}.{norm}.weight"] = (1_024,)
        for projection in ("linear_q", "linear_k", "linear_v", "linear_out"):
            manifest[f"{prefix}.self_attn.{projection}.bias"] = (1_024,)
            manifest[f"{prefix}.self_attn.{projection}.weight"] = (1_024, 1_024)
        manifest.update(
            {
                f"{prefix}.self_attn.linear_pos.weight": (1_024, 1_024),
                f"{prefix}.self_attn.pos_bias_u": (8, 128),
                f"{prefix}.self_attn.pos_bias_v": (8, 128),
                f"{prefix}.conv.pointwise_conv1.bias": (2_048,),
                f"{prefix}.conv.pointwise_conv1.weight": (2_048, 1_024, 1),
                f"{prefix}.conv.depthwise_conv.bias": (1_024,),
                f"{prefix}.conv.depthwise_conv.weight": (1_024, 1, 9),
                f"{prefix}.conv.batch_norm.bias": (1_024,),
                f"{prefix}.conv.batch_norm.running_mean": (1_024,),
                f"{prefix}.conv.batch_norm.running_var": (1_024,),
                f"{prefix}.conv.batch_norm.weight": (1_024,),
                f"{prefix}.conv.pointwise_conv2.bias": (1_024,),
                f"{prefix}.conv.pointwise_conv2.weight": (1_024, 1_024, 1),
            }
        )

    manifest["decoder.prediction.embed.weight"] = (1_025, 640)
    for layer in range(2):
        prefix = "decoder.prediction.dec_rnn.lstm"
        manifest[f"{prefix}.bias_hh_l{layer}"] = (2_560,)
        manifest[f"{prefix}.bias_ih_l{layer}"] = (2_560,)
        manifest[f"{prefix}.weight_hh_l{layer}"] = (2_560, 640)
        manifest[f"{prefix}.weight_ih_l{layer}"] = (2_560, 640)
    manifest.update(
        {
            "joint.enc.bias": (640,),
            "joint.enc.weight": (640, 1_024),
            "joint.pred.bias": (640,),
            "joint.pred.weight": (640, 640),
            "joint.joint_net.2.bias": (1_030,),
            "joint.joint_net.2.weight": (1_030, 640),
        }
    )
    return manifest


def manifest_sha256(manifest: dict[str, tuple[int, ...]]) -> str:
    canonical = bytearray()
    for name in sorted(manifest):
        shape = manifest[name]
        canonical.extend(name.encode("utf-8"))
        canonical.append(0)
        canonical.extend(struct.pack("<Q", len(shape)))
        for dimension in shape:
            canonical.extend(struct.pack("<Q", dimension))
    return hashlib.sha256(canonical).hexdigest()


def validate_gguf(path: Path, *, header_only: bool) -> dict[str, object]:
    from gguf_manifest import read_manifest

    if not path.is_file():
        raise ValueError(f"GGUF does not exist: {path}")
    if not header_only:
        size = path.stat().st_size
        if size != PUBLIC_GGUF_BYTES:
            raise ValueError(f"GGUF size {size} != pinned {PUBLIC_GGUF_BYTES}")
        digest = sha256_file(path)
        if digest != PUBLIC_GGUF_SHA256:
            raise ValueError(
                f"GGUF SHA-256 {digest} != pinned {PUBLIC_GGUF_SHA256}"
            )

    metadata, tensors = read_manifest(path)
    for key, expected in EXPECTED_METADATA.items():
        actual = metadata.get(key)
        if actual != expected:
            raise ValueError(f"GGUF metadata {key}={actual!r}, expected {expected!r}")

    expected = expected_manifest()
    if len(expected) != TENSOR_COUNT:
        raise AssertionError(
            f"derived contract has {len(expected)} tensors, expected {TENSOR_COUNT}"
        )
    parameters = sum(math.prod(shape) for shape in expected.values())
    if parameters != PARAMETER_COUNT:
        raise AssertionError(
            f"derived contract has {parameters} parameters, expected {PARAMETER_COUNT}"
        )
    derived_digest = manifest_sha256(expected)
    if derived_digest != MANIFEST_SHA256:
        raise AssertionError(
            f"derived manifest {derived_digest} != pinned {MANIFEST_SHA256}"
        )

    if len(tensors) != TENSOR_COUNT:
        raise ValueError(f"GGUF has {len(tensors)} tensors, expected {TENSOR_COUNT}")
    actual: dict[str, tuple[int, ...]] = {}
    offsets: set[int] = set()
    max_payload_end = 0
    for tensor in tensors:
        name = tensor["name"]
        shape = tuple(tensor["dimensions"])
        ggml_type = tensor["ggml_type"]
        offset = tensor["offset"]
        if not isinstance(name, str) or name in actual:
            raise ValueError(f"invalid or duplicate tensor name: {name!r}")
        if shape != expected.get(name):
            raise ValueError(
                f"GGUF tensor {name} shape {shape}, expected {expected.get(name)}"
            )
        if ggml_type != GGML_F32:
            raise ValueError(f"GGUF tensor {name} type {ggml_type} != F32 (0)")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError(f"GGUF tensor {name} has invalid offset {offset!r}")
        if offset in offsets:
            raise ValueError(f"GGUF tensor {name} reuses payload offset {offset}")
        offsets.add(offset)
        actual[name] = shape
        max_payload_end = max(max_payload_end, offset + math.prod(shape) * 4)

    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise ValueError(
            f"GGUF manifest mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    actual_digest = manifest_sha256(actual)
    if actual_digest != MANIFEST_SHA256:
        raise ValueError(
            f"GGUF manifest {actual_digest} != pinned {MANIFEST_SHA256}"
        )
    if max_payload_end >= PUBLIC_GGUF_BYTES:
        raise ValueError(
            f"relative payload end {max_payload_end} exceeds artifact size "
            f"{PUBLIC_GGUF_BYTES}"
        )

    return {
        "public_hf": PUBLIC_HF,
        "public_revision": PUBLIC_REVISION,
        "public_file": PUBLIC_GGUF,
        "public_bytes": PUBLIC_GGUF_BYTES,
        "public_sha256": PUBLIC_GGUF_SHA256,
        "upstream_hf": UPSTREAM_HF,
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_nemo": UPSTREAM_NEMO,
        "upstream_nemo_bytes": UPSTREAM_NEMO_BYTES,
        "upstream_nemo_sha256": UPSTREAM_NEMO_SHA256,
        "tensor_count": len(actual),
        "parameter_count": parameters,
        "tensor_bytes_f32": parameters * 4,
        "max_relative_payload_end": max_payload_end,
        "manifest_sha256": actual_digest,
        "header_only": header_only,
        "requires_full_file_confirmation": header_only,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    parser.add_argument(
        "--header-only",
        action="store_true",
        help="validate an HTTP range prefix; never counts as artifact sign-off",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            validate_gguf(args.gguf, header_only=args.header_only), sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
