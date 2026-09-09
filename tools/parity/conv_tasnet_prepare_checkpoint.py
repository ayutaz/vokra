#!/usr/bin/env python3
"""Safely extract the official Asteroid Conv-TasNet state dict.

The upstream ``pytorch_model.bin`` is a seven-field Asteroid wrapper rather
than a bare state dict, so the generic tensor-only converter must reject it.
This dedicated prep step loads with ``weights_only=True``, validates the exact
Libri1Mix enhsingle topology, validates all 345 tensor names/shapes, and writes
a tensor-only safetensors file for the offline Rust converter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

UPSTREAM_HF = "JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k"
UPSTREAM_REVISION = "bb8a876bc157b5cf3c405994accb798c49146016"
CHECKPOINT_BYTES = 20_130_704
CHECKPOINT_SHA256 = "dd8ddefe95a35761f8a48643a618eba908572d04d33208a8ed5451fb5a4378d0"
EXPECTED_ARGS = {
    "fb_name": "FreeFB",
    "n_filters": 512,
    "kernel_size": 32,
    "stride": 16,
    "sample_rate": 16000.0,
    "in_chan": 512,
    "out_chan": 512,
    "bn_chan": 128,
    "hid_chan": 512,
    "skip_chan": 128,
    "conv_kernel_size": 3,
    "n_blocks": 8,
    "n_repeats": 3,
    "n_src": 1,
    "norm_type": "gLN",
    "mask_act": "relu",
    "encoder_activation": None,
}


def expected_shapes() -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "encoder.filterbank._filters": (512, 1, 32),
        "masker.bottleneck.0.gamma": (512,),
        "masker.bottleneck.0.beta": (512,),
        "masker.bottleneck.1.weight": (128, 512, 1),
        "masker.bottleneck.1.bias": (128,),
    }
    for block in range(24):
        prefix = f"masker.TCN.{block}"
        shapes.update(
            {
                f"{prefix}.shared_block.0.weight": (512, 128, 1),
                f"{prefix}.shared_block.0.bias": (512,),
                f"{prefix}.shared_block.1.weight": (1,),
                f"{prefix}.shared_block.2.gamma": (512,),
                f"{prefix}.shared_block.2.beta": (512,),
                f"{prefix}.shared_block.3.weight": (512, 1, 3),
                f"{prefix}.shared_block.3.bias": (512,),
                f"{prefix}.shared_block.4.weight": (1,),
                f"{prefix}.shared_block.5.gamma": (512,),
                f"{prefix}.shared_block.5.beta": (512,),
                f"{prefix}.res_conv.weight": (128, 512, 1),
                f"{prefix}.res_conv.bias": (128,),
                f"{prefix}.skip_conv.weight": (128, 512, 1),
                f"{prefix}.skip_conv.bias": (128,),
            }
        )
    shapes.update(
        {
            "masker.mask_net.0.weight": (1,),
            "masker.mask_net.1.weight": (512, 128, 1),
            "masker.mask_net.1.bias": (512,),
            "decoder.filterbank._filters": (512, 1, 32),
        }
    )
    assert len(shapes) == 345
    return shapes


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_absent(path: Path) -> bool:
    return not path.exists() and not path.is_symlink()


def clean_ancestors(path: Path) -> bool:
    absolute = Path(os.path.abspath(path))
    return all(not parent.is_symlink() for parent in absolute.parents)


def disjoint(left: Path, right: Path) -> bool:
    left, right = Path(os.path.abspath(left)), Path(os.path.abspath(right))
    return left != right and left not in right.parents and right not in left.parents


def self_test() -> None:
    assert EXPECTED_ARGS["n_filters"] == 512
    assert EXPECTED_ARGS["stride"] == 16
    assert len(expected_shapes()) == 345
    with __import__("tempfile").TemporaryDirectory(prefix="conv-tasnet-prep-") as raw:
        root = Path(raw)
        symlink_target = root / "target"
        symlink_target.write_bytes(b"fixture")
        link = root / "link"
        link.symlink_to(symlink_target)
        assert not regular_absent(link)
        assert regular_absent(root / "new-output")
        nested = root / "link" / "nested" / "output"
        assert not clean_ancestors(nested)
        assert disjoint(root / "new-output", symlink_target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        if any(value is not None for value in (args.checkpoint, args.output, args.manifest)):
            parser.error("--self-test accepts no other arguments")
        self_test()
        print("conv_tasnet_prepare_checkpoint self-test: OK")
        return 0

    if any(value is None for value in (args.checkpoint, args.output, args.manifest)):
        parser.error("normal runs require --checkpoint, --output, and --manifest")

    if args.checkpoint.is_symlink() or not args.checkpoint.is_file():
        raise SystemExit("checkpoint must be a regular non-symlink file")
    if not clean_ancestors(args.checkpoint):
        raise SystemExit("checkpoint has a symlinked lexical ancestor")
    if args.checkpoint.stat().st_size != CHECKPOINT_BYTES or sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise SystemExit("checkpoint bytes/SHA-256 do not match the authenticated official identity")
    if not regular_absent(args.output) or not regular_absent(args.manifest):
        raise SystemExit("output and manifest must be absent non-symlink paths")
    if not clean_ancestors(args.output) or not clean_ancestors(args.manifest):
        raise SystemExit("output paths have a symlinked lexical ancestor")
    if not disjoint(args.output, args.checkpoint) or not disjoint(args.manifest, args.checkpoint):
        raise SystemExit("output paths overlap the checkpoint")

    import torch
    from safetensors.torch import save_file

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise SystemExit(f"checkpoint wrapper must be dict, got {type(payload).__name__}")
    expected_wrapper = {
        "dataset",
        "infos",
        "licenses",
        "model_args",
        "model_name",
        "state_dict",
        "task",
    }
    if set(payload) != expected_wrapper:
        raise SystemExit(
            f"unexpected checkpoint wrapper keys: {sorted(payload)}; "
            f"expected {sorted(expected_wrapper)}"
        )
    model_args = payload["model_args"]
    if not isinstance(model_args, dict):
        raise SystemExit("model_args must be a dict")
    if set(model_args) != set(EXPECTED_ARGS):
        raise SystemExit(
            f"unexpected model_args keys: {sorted(model_args)}; "
            f"expected {sorted(EXPECTED_ARGS)}"
        )
    for key, expected in EXPECTED_ARGS.items():
        actual = model_args.get(key)
        if actual != expected:
            raise SystemExit(f"model_args[{key!r}]={actual!r}, expected {expected!r}")

    state = payload["state_dict"]
    if not isinstance(state, dict):
        raise SystemExit("state_dict must be a dict")
    expected = expected_shapes()
    missing = sorted(set(expected) - set(state))
    extra = sorted(set(state) - set(expected))
    if missing or extra:
        raise SystemExit(
            f"state_dict manifest mismatch: missing={missing[:8]!r}, extra={extra[:8]!r}"
        )
    for name, shape in expected.items():
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise SystemExit(f"{name}: expected Tensor, got {type(tensor).__name__}")
        if tuple(tensor.shape) != shape:
            raise SystemExit(f"{name}: shape={tuple(tensor.shape)}, expected={shape}")
        if not tensor.is_floating_point():
            raise SystemExit(f"{name}: non-floating dtype {tensor.dtype}")

    tensors = {name: state[name].detach().cpu().contiguous() for name in sorted(state)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output))
    manifest = {
        "format": "vokra-conv-tasnet-prep-v1",
        "upstream_hf": UPSTREAM_HF,
        "upstream_revision": UPSTREAM_REVISION,
        "input_sha256": sha256(args.checkpoint),
        "output_sha256": sha256(args.output),
        "tensor_count": len(tensors),
        "parameter_count": sum(tensor.numel() for tensor in tensors.values()),
        "model_args": EXPECTED_ARGS,
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
