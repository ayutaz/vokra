#!/usr/bin/env python3
"""Audit the exact public MossFormer2-SS-16K GGUF tensor contract.

The public artifact is a real 223 MB F32 checkpoint with 1,076 tensors.  This
tool derives every expected name and shape from the pinned ClearerVoice-Studio
topology, then compares that contract with the GGUF header without decoding a
single tensor payload.

Normal mode authenticates the complete public file by byte size and SHA-256.
``--header-only`` exists for maintainer-side, low-memory inspection of an HTTP
range prefix; it still validates every metadata entry, tensor name, shape,
dtype, offset, parameter count, and manifest digest, but reports that full-file
authentication remains required.  Runtime/parity sign-off must use normal mode
on VAST.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


UPSTREAM_HF = "alibabasglab/MossFormer2_SS_16K"
UPSTREAM_REVISION = "407cb030cd66340918ebb6c8cc63b18f8592cdbe"
UPSTREAM_CHECKPOINT = "last_best_checkpoint.pt"
UPSTREAM_CHECKPOINT_BYTES = 670_353_271
UPSTREAM_CHECKPOINT_SHA256 = (
    "00a3a48bda492db1e829b85dd443f8f43a43039a3e90f1a24962ea9caf14a11a"
)
SOURCE_REPOSITORY = "https://github.com/modelscope/ClearerVoice-Studio"
SOURCE_REVISION = "6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61"

PUBLIC_HF = "vokra/mossformer2-ss-16k"
PUBLIC_REVISION = "0e9ba9258cead4252f8e5279598af296ada08bf7"
PUBLIC_GGUF = "mossformer2-ss-16k.gguf"
PUBLIC_GGUF_BYTES = 223_058_240
PUBLIC_GGUF_SHA256 = (
    "822516b75873dbeb814dac72f7ca0b5fb75254dd051dfdfdda54987347330f0c"
)

TENSOR_COUNT = 1_076
PARAMETER_COUNT = 55_735_666
MANIFEST_SHA256 = (
    "eb4b366872789b95228a172846259f6aa205a75c678f90941d5e8a3e9a47fb8b"
)
GGML_F32 = 0

EXPECTED_METADATA = {
    "vokra.model.arch": "mossformer2_ss_16k",
    "vokra.model.name": "mossformer2_ss_16k",
    "vokra.model.category": "source-separation",
    "vokra.provenance.license": "apache-2.0",
    "vokra.provenance.model_id": "mossformer2_ss_16k",
    "vokra.provenance.weight_license": "permissive",
    "vokra.provenance.upstream_hf": UPSTREAM_HF,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_manifest() -> dict[str, tuple[int, ...]]:
    manifest: dict[str, tuple[int, ...]] = {
        "dec.weight": (512, 1, 16),
        "enc.conv1d.weight": (512, 1, 16),
        "mask_net.conv1_decoder.weight": (512, 512, 1),
        "mask_net.conv1d_encoder.weight": (512, 512, 1),
        "mask_net.conv1d_out.bias": (1_024,),
        "mask_net.conv1d_out.weight": (1_024, 512, 1),
    }

    root = "mask_net.mdl.intra_mdl.mossformerM"
    for layer in range(24):
        fsmn = f"{root}.fsmn.{layer}"
        manifest.update(
            {
                f"{fsmn}.conv1.0.bias": (256,),
                f"{fsmn}.conv1.0.weight": (256, 512, 1),
                f"{fsmn}.conv1.1.weight": (1,),
                f"{fsmn}.conv2.bias": (512,),
                f"{fsmn}.conv2.weight": (512, 256, 1),
                f"{fsmn}.norm1.bias": (256,),
                f"{fsmn}.norm1.weight": (256,),
                f"{fsmn}.norm2.bias": (256,),
                f"{fsmn}.norm2.weight": (256,),
            }
        )

        core = f"{fsmn}.gated_fsmn"
        manifest.update(
            {
                f"{core}.fsmn.linear.bias": (256,),
                f"{core}.fsmn.linear.weight": (256, 256),
                f"{core}.fsmn.project.weight": (256, 256),
                f"{core}.fsmn.conv.conv1.weight": (256, 1, 39, 1),
                f"{core}.fsmn.conv.conv2.weight": (256, 2, 39, 1),
                f"{core}.fsmn.conv.norm1.bias": (256,),
                f"{core}.fsmn.conv.norm1.weight": (256,),
                f"{core}.fsmn.conv.norm2.bias": (256,),
                f"{core}.fsmn.conv.norm2.weight": (256,),
                f"{core}.fsmn.conv.prelu1.weight": (256,),
                f"{core}.fsmn.conv.prelu2.weight": (256,),
            }
        )
        for branch in ("to_u", "to_v"):
            prefix = f"{core}.{branch}.mdl"
            manifest.update(
                {
                    f"{prefix}.0.bias": (256,),
                    f"{prefix}.0.weight": (256,),
                    f"{prefix}.1.bias": (256,),
                    f"{prefix}.1.weight": (256, 256),
                    f"{prefix}.3.sequential.1.conv.weight": (256, 1, 17),
                }
            )

        attention = f"{root}.layers.{layer}"
        manifest.update(
            {
                f"{attention}.qk_offset_scale.beta": (4, 128),
                f"{attention}.qk_offset_scale.gamma": (4, 128),
            }
        )
        if layer == 0:
            # One RotaryEmbedding module is shared by all 24 FLASH layers;
            # PyTorch state_dict de-duplicates the shared registered buffer.
            manifest[f"{attention}.rotary_pos_emb.freqs"] = (16,)
        for projection, output, input_ in (
            ("to_hidden", 2_048, 512),
            ("to_qk", 128, 512),
            ("to_out", 512, 1_024),
        ):
            prefix = f"{attention}.{projection}.mdl"
            manifest.update(
                {
                    f"{prefix}.0.g": (1,),
                    f"{prefix}.1.bias": (output,),
                    f"{prefix}.1.weight": (output, input_),
                    f"{prefix}.3.sequential.1.conv.weight": (output, 1, 17),
                }
            )

    manifest.update(
        {
            "mask_net.mdl.intra_mdl.norm.bias": (512,),
            "mask_net.mdl.intra_mdl.norm.weight": (512,),
            "mask_net.mdl.intra_norm.bias": (512,),
            "mask_net.mdl.intra_norm.weight": (512,),
            "mask_net.norm.bias": (512,),
            "mask_net.norm.weight": (512,),
            "mask_net.output.0.bias": (512,),
            "mask_net.output.0.weight": (512, 512, 1),
            "mask_net.output_gate.0.bias": (512,),
            "mask_net.output_gate.0.weight": (512, 512, 1),
            "mask_net.pos_enc.inv_freq": (256,),
            "mask_net.pos_enc.scale": (1,),
            "mask_net.prelu.weight": (1,),
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
        dimensions = tensor["dimensions"]
        ggml_type = tensor["ggml_type"]
        offset = tensor["offset"]
        if not isinstance(name, str) or name in actual:
            raise ValueError(f"invalid or duplicate tensor name: {name!r}")
        if not isinstance(dimensions, list) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in dimensions
        ):
            raise ValueError(f"GGUF tensor {name} has invalid dimensions {dimensions!r}")
        shape = tuple(dimensions)
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
    # Tensor offsets are relative to the aligned payload start.  The largest
    # range must fit inside the pinned complete artifact; in header-only mode
    # it intentionally need not fit inside the downloaded prefix itself.
    if max_payload_end >= PUBLIC_GGUF_BYTES:
        raise ValueError(
            f"GGUF relative payload end {max_payload_end} exceeds artifact size "
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
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
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
    print(json.dumps(validate_gguf(args.gguf, header_only=args.header_only), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
