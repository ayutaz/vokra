#!/usr/bin/env python3
"""Prepare the canonical emotion2vec+ Large pickle for strict GGUF conversion.

Run only on vast.ai: the 1.95 GB source plus the prepared F32 safetensors
exceed the repository's 2 GB aggregate-artifact threshold.  Python/PyTorch is
an offline bridge only; neither enters the Vokra runtime.

The source is pinned to
``emotion2vec/emotion2vec_plus_large@6c303ba987b86b93193de93e34bb2b077a6bedc4/model.pt``.
After verifying bytes and SHA-256, this script unwraps the official ``model``
state dict, accepts exactly the 185 inference tensors, and performs the one
documented layout adaptation: ``alibi_scale [1,1,16,1,1] -> [16]``.  No
generic squeeze, tensor dropping, or dtype coercion is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

UPSTREAM_REPO = "emotion2vec/emotion2vec_plus_large"
UPSTREAM_REVISION = "6c303ba987b86b93193de93e34bb2b077a6bedc4"
CHECKPOINT_FILE = "model.pt"
CHECKPOINT_BYTES = 1_945_790_254
CHECKPOINT_SHA256 = (
    "be501a01f26fcdc7663a062dff86af839afbaef7c4de32f5e42d7e1ad2784da4"
)
TENSOR_COUNT = 185
GGUF_MANIFEST_SHA256 = (
    "f5f8f684302cf55fb399277a7446976a77f570816e7e3345a008e4d0b6774401"
)


def expected_manifest() -> dict[str, list[int]]:
    tensors: dict[str, list[int]] = {}

    def block(prefix: str) -> None:
        tensors.update(
            {
                f"{prefix}.attn.proj.bias": [1024],
                f"{prefix}.attn.proj.weight": [1024, 1024],
                f"{prefix}.attn.qkv.bias": [3072],
                f"{prefix}.attn.qkv.weight": [3072, 1024],
                f"{prefix}.mlp.fc1.bias": [4096],
                f"{prefix}.mlp.fc1.weight": [4096, 1024],
                f"{prefix}.mlp.fc2.bias": [1024],
                f"{prefix}.mlp.fc2.weight": [1024, 4096],
                f"{prefix}.norm1.bias": [1024],
                f"{prefix}.norm1.weight": [1024],
                f"{prefix}.norm2.bias": [1024],
                f"{prefix}.norm2.weight": [1024],
            }
        )

    for layer in range(8):
        block(f"d2v_model.blocks.{layer}")
    audio = "d2v_model.modality_encoders.AUDIO"
    tensors[f"{audio}.alibi_scale"] = [16]
    for layer in range(4):
        block(f"{audio}.context_encoder.blocks.{layer}")
    tensors[f"{audio}.context_encoder.norm.bias"] = [1024]
    tensors[f"{audio}.context_encoder.norm.weight"] = [1024]
    tensors[f"{audio}.extra_tokens"] = [1, 10, 1024]
    kernels = [10, 3, 3, 3, 3, 2, 2]
    for layer, kernel in enumerate(kernels):
        input_channels = 1 if layer == 0 else 512
        prefix = f"{audio}.local_encoder.conv_layers.{layer}"
        tensors[f"{prefix}.0.weight"] = [512, input_channels, kernel]
        tensors[f"{prefix}.2.1.bias"] = [512]
        tensors[f"{prefix}.2.1.weight"] = [512]
    tensors[f"{audio}.project_features.1.bias"] = [512]
    tensors[f"{audio}.project_features.1.weight"] = [512]
    tensors[f"{audio}.project_features.2.bias"] = [1024]
    tensors[f"{audio}.project_features.2.weight"] = [1024, 512]
    for layer in range(1, 6):
        prefix = f"{audio}.relative_positional_encoder.{layer}.0"
        tensors[f"{prefix}.bias"] = [1024]
        tensors[f"{prefix}.weight"] = [1024, 64, 19]
    tensors["proj.bias"] = [9]
    tensors["proj.weight"] = [9, 1024]
    assert len(tensors) == TENSOR_COUNT
    return tensors


def manifest_sha256(manifest: dict[str, list[int]]) -> str:
    digest = hashlib.sha256()
    for name, shape in sorted(manifest.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(shape).to_bytes(8, "little"))
        for dimension in shape:
            digest.update(dimension.to_bytes(8, "little"))
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(path: Path) -> None:
    if path.name != CHECKPOINT_FILE:
        sys.exit(f"expected source basename {CHECKPOINT_FILE!r}, got {path.name!r}")
    if path.stat().st_size != CHECKPOINT_BYTES:
        sys.exit(
            f"{path}: {path.stat().st_size} bytes, expected {CHECKPOINT_BYTES} "
            f"for {UPSTREAM_REPO}@{UPSTREAM_REVISION}"
        )
    actual = sha256_file(path)
    if actual != CHECKPOINT_SHA256:
        sys.exit(f"{path}: SHA-256 {actual}, expected {CHECKPOINT_SHA256}")


def prepare(source: Path, output: Path) -> None:
    verify_source(source)
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as error:
        sys.exit(f"missing offline parity dependency: {error}")

    # Pickle deserialization is reached only after the immutable official hash
    # passes.  The release predates weights_only-safe serialization.
    raw = torch.load(str(source), map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or not isinstance(raw.get("model"), dict):
        sys.exit("official checkpoint must contain a dict-valued top-level 'model'")
    state = raw["model"]
    expected = expected_manifest()
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        extra = sorted(set(state) - set(expected))
        sys.exit(
            f"official state_dict mismatch: missing={missing[:8]} extra={extra[:8]} "
            f"counts={len(state)}/{len(expected)}"
        )

    prepared: dict[str, object] = {}
    source_shapes: dict[str, list[int]] = {}
    for name in sorted(expected):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            sys.exit(f"{name}: expected torch.Tensor, got {type(tensor)!r}")
        if tensor.dtype != torch.float32:
            sys.exit(f"{name}: expected torch.float32, got {tensor.dtype}")
        source_shape = list(tensor.shape)
        source_shapes[name] = source_shape
        if name.endswith(".alibi_scale"):
            if source_shape != [1, 1, 16, 1, 1]:
                sys.exit(
                    f"{name}: expected official [1,1,16,1,1], got {source_shape}"
                )
            tensor = tensor.reshape(16)
        if list(tensor.shape) != expected[name]:
            sys.exit(
                f"{name}: prepared shape {list(tensor.shape)}, expected {expected[name]}"
            )
        prepared[name] = tensor.detach().contiguous()

    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(prepared, str(output))
    sidecar = output.with_suffix(output.suffix + ".manifest.json")
    sidecar.write_text(
        json.dumps(
            {
                "source_repo": UPSTREAM_REPO,
                "source_revision": UPSTREAM_REVISION,
                "source_file": CHECKPOINT_FILE,
                "source_bytes": CHECKPOINT_BYTES,
                "source_sha256": CHECKPOINT_SHA256,
                "tensor_count": TENSOR_COUNT,
                "gguf_manifest_sha256": GGUF_MANIFEST_SHA256,
                "adaptations": {
                    "d2v_model.modality_encoders.AUDIO.alibi_scale": {
                        "source": source_shapes[
                            "d2v_model.modality_encoders.AUDIO.alibi_scale"
                        ],
                        "prepared": [16],
                        "reason": "GGUF rank<=4; singleton-only reshape preserves per-head broadcast values",
                    }
                },
                "tensors": expected,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"emotion2vec: wrote {len(prepared)} tensors -> {output}")
    print(f"emotion2vec: audit manifest -> {sidecar}")


def self_test() -> None:
    manifest = expected_manifest()
    assert manifest_sha256(manifest) == GGUF_MANIFEST_SHA256
    assert manifest["d2v_model.blocks.7.attn.qkv.weight"] == [3072, 1024]
    assert manifest["proj.weight"] == [9, 1024]
    print("emotion2vec_prepare_checkpoint: self-test PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test and (args.input is None or args.output is None):
        parser.error("--input and --output are required unless --self-test is used")
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    prepare(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
