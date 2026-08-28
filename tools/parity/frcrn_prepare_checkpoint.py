#!/usr/bin/env python3
"""Prepare the exact official FRCRN-SE-16K checkpoint for conversion.

This offline, VAST-only sidecar verifies the pinned Hugging Face checkpoint,
safe-loads its inference state, drops only BatchNorm tracking counters, and
writes the exact 812-tensor F32 safetensors manifest consumed by
``vokra-convert``. It never falls back to unsafe pickle loading and never
accepts a merely shape-compatible substitute.

Pinned inputs:

* weights: ``alibabasglab/FRCRN_SE_16K`` at revision
  ``3766e6a64b0d8cb58f08d913d617bf129f11ed53``;
* file: ``last_best_checkpoint.pt`` (161,053,751 bytes, SHA-256 below);
* source: ``modelscope/ClearerVoice-Studio`` at revision
  ``6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61``.

Run only through the repository uv environment, for example on VAST:

    cd tools/parity
    uv run python frcrn_prepare_checkpoint.py \
      --checkpoint /workspace/FRCRN_SE_16K/last_best_checkpoint.pt \
      --output /workspace/frcrn.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping


UPSTREAM_HF = "alibabasglab/FRCRN_SE_16K"
UPSTREAM_REVISION = "3766e6a64b0d8cb58f08d913d617bf129f11ed53"
SOURCE_REVISION = "6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61"
CHECKPOINT_SHA256 = "b22256adbb91b68cf5a3db8f6657a4fb17066eecd5f069803e59c186c1cf3ebb"
CHECKPOINT_BYTES = 161_053_751
TENSOR_MANIFEST_SHA256 = "ca71dad1ae5293d3d63628b71127c0efdf004cec684e5a341ab376ce3e2851b7"
TENSOR_COUNT = 812
PARAMETER_COUNT = 14_387_164


def _insert(out: dict[str, tuple[int, ...]], name: str, shape: tuple[int, ...]) -> None:
    if name in out:
        raise AssertionError(f"duplicate FRCRN manifest entry: {name}")
    out[name] = shape


def _add_complex_conv(
    out: dict[str, tuple[int, ...]],
    prefix: str,
    stem: str,
    weight_shape: tuple[int, ...],
    bias: int,
) -> None:
    for component in ("re", "im"):
        _insert(out, f"{prefix}.{stem}_{component}.weight", weight_shape)
        _insert(out, f"{prefix}.{stem}_{component}.bias", (bias,))


def _add_complex_bn(
    out: dict[str, tuple[int, ...]], prefix: str, channels: int
) -> None:
    for component in ("re", "im"):
        for field in ("weight", "bias", "running_mean", "running_var"):
            _insert(out, f"{prefix}.bn_{component}.{field}", (channels,))


def _add_real_fsmn(out: dict[str, tuple[int, ...]], prefix: str) -> None:
    _insert(out, f"{prefix}.linear.weight", (128, 128))
    _insert(out, f"{prefix}.linear.bias", (128,))
    _insert(out, f"{prefix}.project.weight", (128, 128))
    _insert(out, f"{prefix}.conv1.weight", (128, 1, 20, 1))


def _add_l1_fsmn(out: dict[str, tuple[int, ...]], prefix: str) -> None:
    for component in ("re", "im"):
        _add_real_fsmn(out, f"{prefix}.fsmn_{component}_L1")


def _add_central_fsmn(out: dict[str, tuple[int, ...]], prefix: str) -> None:
    for component in ("re", "im"):
        for level in ("L1", "L2"):
            _add_real_fsmn(out, f"{prefix}.fsmn_{component}_{level}")


def _add_se(out: dict[str, tuple[int, ...]], prefix: str) -> None:
    for component in ("r", "i"):
        _insert(out, f"{prefix}.fc_{component}.0.weight", (16, 128))
        _insert(out, f"{prefix}.fc_{component}.0.bias", (16,))
        _insert(out, f"{prefix}.fc_{component}.2.weight", (128, 16))
        _insert(out, f"{prefix}.fc_{component}.2.bias", (128,))


def _add_unet(out: dict[str, tuple[int, ...]], root: str) -> None:
    for layer in range(7):
        in_channels = 1 if layer == 0 else 128
        kernel_h = 2 if layer == 6 else 5
        prefix = f"{root}.encoder{layer}"
        _add_complex_conv(
            out,
            f"{prefix}.conv",
            "conv",
            (128, in_channels, kernel_h, 2),
            128,
        )
        _add_complex_bn(out, f"{prefix}.bn", 128)

    decoder_geometry = (
        (128, 128, 2),
        (256, 128, 5),
        (256, 128, 5),
        (256, 128, 5),
        (256, 128, 6),
        (256, 128, 5),
        (256, 1, 5),
    )
    for layer, (in_channels, out_channels, kernel_h) in enumerate(
        decoder_geometry
    ):
        prefix = f"{root}.decoder{layer}"
        _add_complex_conv(
            out,
            f"{prefix}.transconv",
            "tconv",
            (in_channels, out_channels, kernel_h, 2),
            out_channels,
        )
        _add_complex_bn(out, f"{prefix}.bn", out_channels)

    _add_central_fsmn(out, f"{root}.fsmn")
    for layer in range(7):
        _add_l1_fsmn(out, f"{root}.fsmn_enc{layer}")
        _add_l1_fsmn(out, f"{root}.fsmn_dec{layer}")
        _add_se(out, f"{root}.se_layer_enc{layer}")
        if layer < 6:
            _add_se(out, f"{root}.se_layer_dec{layer}")
    _add_complex_conv(out, f"{root}.linear", "conv", (1, 1, 1, 1), 1)


def expected_manifest() -> dict[str, tuple[int, ...]]:
    out: dict[str, tuple[int, ...]] = {}
    _insert(out, "stft.weight", (642, 1, 640))
    _insert(out, "istft.weight", (642, 1, 640))
    _insert(out, "istft.window", (1, 640, 1))
    _insert(out, "istft.enframe", (640, 1, 640))
    _add_unet(out, "unet")
    _add_unet(out, "unet2")
    if len(out) != TENSOR_COUNT:
        raise AssertionError(f"internal manifest count {len(out)} != {TENSOR_COUNT}")
    if sum(math.prod(shape) for shape in out.values()) != PARAMETER_COUNT:
        raise AssertionError("internal FRCRN parameter count drift")
    if manifest_sha256(out) != TENSOR_MANIFEST_SHA256:
        raise AssertionError("internal FRCRN manifest digest drift")
    return out


def manifest_sha256(manifest: Mapping[str, tuple[int, ...]]) -> str:
    canonical = bytearray()
    for name in sorted(manifest):
        shape = manifest[name]
        canonical.extend(name.encode("utf-8"))
        canonical.append(0)
        canonical.extend(struct.pack("<Q", len(shape)))
        for dimension in shape:
            canonical.extend(struct.pack("<Q", dimension))
    return hashlib.sha256(canonical).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def unwrap_official_state(obj: object, checkpoint: Path) -> OrderedDict[str, Any]:
    import torch

    if not isinstance(obj, dict):
        raise SystemExit(
            f"{checkpoint}: top level is {type(obj).__name__}, expected official dict"
        )
    candidate: object = obj.get("model", obj)
    if not isinstance(candidate, dict) or not candidate:
        raise SystemExit(f"{checkpoint}: official `model` state dict is absent or empty")
    if not all(isinstance(name, str) for name in candidate):
        raise SystemExit(f"{checkpoint}: state dict contains a non-string key")
    if not all(isinstance(value, torch.Tensor) for value in candidate.values()):
        offender = next(
            name
            for name, value in candidate.items()
            if not isinstance(value, torch.Tensor)
        )
        raise SystemExit(f"{checkpoint}: model[{offender!r}] is not a tensor")

    state = OrderedDict(candidate)
    prefixed = [name.startswith("module.") for name in state]
    if any(prefixed):
        if not all(prefixed):
            raise SystemExit(f"{checkpoint}: mixed `module.` prefix state is refused")
        state = OrderedDict((name[7:], value) for name, value in state.items())
    return state


def prepare_state(
    state: OrderedDict[str, Any], checkpoint: Path
) -> tuple[OrderedDict[str, Any], list[str]]:
    import torch

    expected = expected_manifest()
    kept: OrderedDict[str, torch.Tensor] = OrderedDict()
    dropped: list[str] = []
    for name, tensor in state.items():
        if name.endswith(".num_batches_tracked"):
            if tensor.dtype not in (torch.int32, torch.int64):
                raise SystemExit(
                    f"{checkpoint}: {name} is {tensor.dtype}, expected an integer BN counter"
                )
            dropped.append(name)
            continue
        if tensor.dtype != torch.float32:
            raise SystemExit(
                f"{checkpoint}: inference tensor {name!r} is {tensor.dtype}, expected torch.float32"
            )
        kept[name] = tensor.detach().cpu().contiguous()

    actual_names = set(kept)
    expected_names = set(expected)
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    wrong = [
        (name, tuple(kept[name].shape), expected[name])
        for name in sorted(actual_names & expected_names)
        if tuple(kept[name].shape) != expected[name]
    ]
    if missing or extra or wrong:
        raise SystemExit(
            f"{checkpoint}: FRCRN manifest mismatch; missing={missing[:8]} "
            f"extra={extra[:8]} wrong_shape={wrong[:8]}"
        )
    if len(kept) != TENSOR_COUNT:
        raise SystemExit(f"{checkpoint}: kept {len(kept)} tensors, expected {TENSOR_COUNT}")
    return kept, dropped


def self_test() -> None:
    manifest = expected_manifest()
    assert len(manifest) == 812
    assert manifest["stft.weight"] == (642, 1, 640)
    assert manifest["unet.encoder0.conv.conv_re.weight"] == (128, 1, 5, 2)
    assert manifest["unet2.decoder4.transconv.tconv_im.weight"] == (
        256,
        128,
        6,
        2,
    )
    print("frcrn_prepare_checkpoint self-test: ok")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "--ckpt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        if args.checkpoint is not None or args.output is not None:
            parser.error("--self-test does not accept --checkpoint/--output")
        self_test()
        return 0
    if args.checkpoint is None or args.output is None:
        parser.error("--checkpoint and --output are required outside --self-test")

    checkpoint: Path = args.checkpoint
    output: Path = args.output
    import torch
    from safetensors.torch import save_file

    if not checkpoint.is_file():
        raise SystemExit(f"--checkpoint is not a file: {checkpoint}")
    if checkpoint.stat().st_size != CHECKPOINT_BYTES:
        raise SystemExit(
            f"{checkpoint}: {checkpoint.stat().st_size} bytes, expected {CHECKPOINT_BYTES}"
        )
    actual_sha = sha256_file(checkpoint)
    if actual_sha != CHECKPOINT_SHA256:
        raise SystemExit(
            f"{checkpoint}: SHA-256 {actual_sha}, expected {CHECKPOINT_SHA256}"
        )

    obj = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = unwrap_official_state(obj, checkpoint)
    prepared, dropped = prepare_state(state, checkpoint)

    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(prepared, str(output))
    output_sha = sha256_file(output)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "checkpoint_bytes": CHECKPOINT_BYTES,
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "dropped_batch_norm_counters": dropped,
                "output_bytes": output.stat().st_size,
                "output_sha256": output_sha,
                "parameter_count": PARAMETER_COUNT,
                "source_revision": SOURCE_REVISION,
                "tensor_count": TENSOR_COUNT,
                "tensor_manifest_sha256": TENSOR_MANIFEST_SHA256,
                "upstream_hf": UPSTREAM_HF,
                "upstream_revision": UPSTREAM_REVISION,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"prepared {TENSOR_COUNT} F32 tensors ({PARAMETER_COUNT} values)")
    print(f"dropped_batch_norm_counters={len(dropped)}")
    print(f"sha256={output_sha}  {output}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
