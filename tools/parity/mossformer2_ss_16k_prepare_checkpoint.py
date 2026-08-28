#!/usr/bin/env python3
"""Extract the pinned official MossFormer2 state dict to safetensors.

Run only on VAST.  The upstream 670 MB PyTorch checkpoint is authenticated
before ``torch.load(weights_only=True)``.  Every canonical tensor name, shape
and floating dtype must match the independently derived 1,076-tensor public
contract; training/optimizer entries never enter the converter input.  The
only accepted raw-state difference is the complete set of 23 aliases created
when PyTorch serializes the one RotaryEmbedding shared by all 24 FLASH layers.
Those aliases must be exactly equal to layer 0 before they are removed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import torch
from safetensors.torch import save_file


UPSTREAM_HF = "alibabasglab/MossFormer2_SS_16K"
UPSTREAM_REVISION = "407cb030cd66340918ebb6c8cc63b18f8592cdbe"
CHECKPOINT_BYTES = 670_353_271
CHECKPOINT_SHA256 = (
    "00a3a48bda492db1e829b85dd443f8f43a43039a3e90f1a24962ea9caf14a11a"
)
EXPECTED_WRAPPER_KEYS = {"epoch", "model", "optimizer", "step"}
ROTARY_BASE = (
    "mask_net.mdl.intra_mdl.mossformerM.layers.0.rotary_pos_emb.freqs"
)
ROTARY_ALIASES = {
    f"mask_net.mdl.intra_mdl.mossformerM.layers.{layer}.rotary_pos_emb.freqs"
    for layer in range(1, 24)
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"missing pinned checkpoint: {path}")
    size = path.stat().st_size
    if size != CHECKPOINT_BYTES:
        raise ValueError(f"checkpoint size {size} != pinned {CHECKPOINT_BYTES}")
    digest = sha256_file(path)
    if digest != CHECKPOINT_SHA256:
        raise ValueError(
            f"checkpoint SHA-256 {digest} != pinned {CHECKPOINT_SHA256}"
        )


def audit_module(repository: Path) -> ModuleType:
    path = repository / "tools" / "audit" / "mossformer2_ss_16k_manifest.py"
    spec = importlib.util.spec_from_file_location("vokra_mossformer2_audit", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import manifest contract from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_state(payload: object) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise ValueError(
            f"checkpoint wrapper must be dict, got {type(payload).__name__}"
        )
    if set(payload) == EXPECTED_WRAPPER_KEYS:
        state = payload.get("model")
    elif payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
        state = payload
    else:
        raise ValueError(
            f"unexpected checkpoint wrapper keys: {sorted(map(str, payload))}; "
            f"expected {sorted(EXPECTED_WRAPPER_KEYS)} or a bare tensor state dict"
        )
    if not isinstance(state, dict):
        raise ValueError("checkpoint model entry is not a dict")
    if state and all(isinstance(key, str) and key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    invalid = sorted(
        str(key)
        for key, value in state.items()
        if not isinstance(key, str) or not isinstance(value, torch.Tensor)
    )
    if invalid:
        raise ValueError(f"state dict has non-tensor entries: {invalid[:8]}")
    return state  # type: ignore[return-value]


def canonical_state(
    state: dict[str, torch.Tensor],
    expected: dict[str, tuple[int, ...]],
) -> tuple[dict[str, torch.Tensor], int]:
    missing = sorted(set(expected) - set(state))
    unexpected = set(state) - set(expected)
    if missing:
        raise ValueError(f"state manifest is missing canonical tensors: {missing[:8]}")
    if unexpected and unexpected != ROTARY_ALIASES:
        raise ValueError(
            "unexpected checkpoint tensors outside the exact shared-rotary alias set: "
            f"{sorted(unexpected)[:8]}"
        )
    if unexpected:
        base = state.get(ROTARY_BASE)
        if base is None:
            raise ValueError("shared rotary aliases exist without the canonical layer-0 tensor")
        for name in sorted(ROTARY_ALIASES):
            alias = state[name]
            if tuple(alias.shape) != tuple(base.shape) or alias.dtype != base.dtype:
                raise ValueError(f"{name}: shared rotary alias shape/dtype differs from layer 0")
            if not torch.equal(alias, base):
                raise ValueError(f"{name}: shared rotary alias values differ from layer 0")
    canonical = {name: tensor for name, tensor in state.items() if name not in unexpected}
    return canonical, len(unexpected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    validate_file(args.checkpoint)
    audit = audit_module(args.repository.resolve())
    expected = audit.expected_manifest()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    raw_state = checkpoint_state(payload)
    state, collapsed_rotary_aliases = canonical_state(raw_state, expected)
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"state manifest mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    for name, shape in expected.items():
        tensor = state[name]
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name}: shape {tuple(tensor.shape)} != {shape}")
        if not tensor.is_floating_point():
            raise ValueError(f"{name}: dtype {tensor.dtype} is not floating point")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name}: tensor contains non-finite values")

    tensors = {
        name: state[name].detach().cpu().contiguous()
        for name in sorted(state)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output))
    result = {
        "format": "vokra-mossformer2-ss-16k-prep-v1",
        "upstream_hf": UPSTREAM_HF,
        "upstream_revision": UPSTREAM_REVISION,
        "checkpoint_bytes": CHECKPOINT_BYTES,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "output_bytes": args.output.stat().st_size,
        "output_sha256": sha256_file(args.output),
        "raw_tensor_count": len(raw_state),
        "tensor_count": len(tensors),
        "parameter_count": sum(tensor.numel() for tensor in tensors.values()),
        "manifest_sha256": audit.manifest_sha256(expected),
        "collapsed_shared_rotary_aliases": collapsed_rotary_aliases,
    }
    args.manifest.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
