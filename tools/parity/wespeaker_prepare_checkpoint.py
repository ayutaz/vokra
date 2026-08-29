#!/usr/bin/env python3
"""Prepare the fixed official WeSpeaker ResNet34-LM checkpoint.

The upstream ``avg_model.pt`` is a torch pickle.  This sidecar is the only
approved bridge for it: it uses ``torch.load(..., weights_only=True)``,
authenticates the complete source before loading, validates the exact
official-combined-bare-219 manifest, and writes safetensors for the strict
Vokra converter.  The 36 inference-inert BatchNorm counters are retained as
their actual values represented in F32 because the converter's safetensors
reader accepts floating dtypes only.  No values, names, or shapes are
synthesized.

The output sidecar records source/output SHA-256 values and the complete
source/output tensor manifest.  ``--self-test`` is pure offline and imports no
model or torch package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

UPSTREAM_HF = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
UPSTREAM_REVISION = "f0c48c298fd835726c27956a5d617bad7115627e"
CHECKPOINT_FILENAME = "avg_model.pt"
CHECKPOINT_SHA256 = "9872b375f2c6a3851ca471cbbf59e06efd23a627d78bf5872e1f0269fd298449"
CHECKPOINT_BYTES = 45053131
CHECKPOINT_GIT_OID = "7f92ddd059d244c7d2653650d3be85de9f136c41"
SOURCE_REVISION = "45941e7cba2c3ea99e232d02bedf617fc71b0dad"
TENSOR_COUNT = 219
COUNTER_COUNT = 36

STAGE_BLOCKS = (3, 4, 6, 3)
STAGE_CHANNELS = (32, 64, 128, 256)


def expected_manifest() -> dict[str, tuple[int, ...]]:
    """Return the exact bare state-dict names/shapes admitted by the Rust side."""

    manifest: dict[str, tuple[int, ...]] = {}

    def add(name: str, shape: tuple[int, ...]) -> None:
        if name in manifest:
            raise AssertionError(f"duplicate manifest entry: {name}")
        manifest[name] = shape

    def norm(prefix: str, channels: int) -> None:
        for field in ("weight", "bias", "running_mean", "running_var"):
            add(f"{prefix}.{field}", (channels,))
        add(f"{prefix}.num_batches_tracked", ())

    add("conv1.weight", (32, 1, 3, 3))
    norm("bn1", 32)
    input_channels = 32
    for stage, (blocks, output_channels) in enumerate(
        zip(STAGE_BLOCKS, STAGE_CHANNELS, strict=True), start=1
    ):
        for block in range(blocks):
            prefix = f"layer{stage}.{block}"
            add(f"{prefix}.conv1.weight", (output_channels, input_channels, 3, 3))
            norm(f"{prefix}.bn1", output_channels)
            add(f"{prefix}.conv2.weight", (output_channels, output_channels, 3, 3))
            norm(f"{prefix}.bn2", output_channels)
            if stage > 1 and block == 0:
                add(f"{prefix}.shortcut.0.weight", (output_channels, input_channels, 1, 1))
                norm(f"{prefix}.shortcut.1", output_channels)
            input_channels = output_channels
    add("seg_1.weight", (256, 5120))
    add("seg_1.bias", (256,))
    add("projection.weight", (17982, 256))
    if len(manifest) != TENSOR_COUNT:
        raise AssertionError(f"internal manifest count {len(manifest)} != {TENSOR_COUNT}")
    if sum(name.endswith(".num_batches_tracked") for name in manifest) != COUNTER_COUNT:
        raise AssertionError("internal counter manifest drift")
    return manifest


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sha256(manifest: Mapping[str, Mapping[str, Any]]) -> str:
    canonical = bytearray()
    for name in sorted(manifest):
        entry = manifest[name]
        canonical.extend(name.encode("utf-8"))
        canonical.append(0)
        canonical.extend(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode())
        canonical.append(0)
    return hashlib.sha256(canonical).hexdigest()


def unwrap_state_dict(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SystemExit(f"checkpoint root is not a mapping: {type(value).__name__}")
    current: Mapping[str, object] = value
    for _ in range(3):
        if all(isinstance(name, str) for name in current):
            if all(hasattr(item, "dtype") and hasattr(item, "shape") for item in current.values()):
                return current
        nested = [
            current[key]
            for key in ("model", "state_dict", "model_state_dict", "module")
            if key in current and isinstance(current[key], Mapping)
        ]
        if len(nested) != 1:
            break
        current = nested[0]
    raise SystemExit("could not find a string-to-tensor state dict under known wrappers")


def validate_paths(checkpoint: Path, output: Path) -> Path:
    for candidate in (checkpoint, output.parent):
        current = candidate
        while current != current.parent:
            # macOS exposes /var as the system /private/var alias; it is not
            # user-controlled output redirection and is safe to traverse.
            if current.is_symlink() and current != Path("/var"):
                raise SystemExit(f"path contains a symlink component: {current}")
            current = current.parent
    if checkpoint.name != CHECKPOINT_FILENAME:
        raise SystemExit(f"checkpoint must be named {CHECKPOINT_FILENAME}")
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise SystemExit(f"checkpoint is not a regular file: {checkpoint}")
    sidecar = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to overwrite output: {output}")
    if sidecar.exists() or sidecar.is_symlink():
        raise SystemExit(f"refusing to overwrite output manifest: {sidecar}")
    if not output.parent.exists() or output.parent.is_symlink() or not output.parent.is_dir():
        raise SystemExit(f"output parent must be an existing regular directory: {output.parent}")
    return sidecar


def self_test() -> int:
    manifest = expected_manifest()
    assert len(manifest) == TENSOR_COUNT
    counters = [name for name in manifest if name.endswith(".num_batches_tracked")]
    assert len(counters) == COUNTER_COUNT
    assert manifest["projection.weight"] == (17982, 256)
    assert manifest["layer4.2.bn2.num_batches_tracked"] == ()
    assert CHECKPOINT_FILENAME == "avg_model.pt"
    assert CHECKPOINT_BYTES == 45053131 and len(CHECKPOINT_SHA256) == 64
    assert len(CHECKPOINT_GIT_OID) == 40 and len(SOURCE_REVISION) == 40
    source = Path(__file__).read_text(encoding="utf-8")
    assert "torch.load(str(args.checkpoint), map_location=\"cpu\", weights_only=True)" in source
    with tempfile.TemporaryDirectory(prefix="wespeaker-prepare-self-test-") as directory:
        root = Path(directory)
        checkpoint = root / CHECKPOINT_FILENAME
        checkpoint.write_bytes(b"fixture")
        output = root / "out.safetensors"
        assert validate_paths(checkpoint, output).name == "out.safetensors.manifest.json"
        wrong_name = root / "avg_model"
        wrong_name.write_bytes(b"fixture")
        try:
            validate_paths(wrong_name, output)
        except SystemExit:
            pass
        else:
            raise AssertionError("wrong checkpoint filename accepted")
        output.symlink_to(root / "missing-output")
        try:
            validate_paths(checkpoint, output)
        except SystemExit:
            pass
        else:
            raise AssertionError("dangling output symlink accepted")
        output.unlink()
        sidecar = output.with_suffix(output.suffix + ".manifest.json")
        sidecar.symlink_to(root / "missing-manifest")
        try:
            validate_paths(checkpoint, output)
        except SystemExit:
            pass
        else:
            raise AssertionError("dangling output manifest symlink accepted")
        sidecar.unlink()
        real_parent = root / "real-parent"
        real_parent.mkdir()
        (real_parent / CHECKPOINT_FILENAME).write_bytes(b"fixture")
        input_link = root / "input-link"
        input_link.symlink_to(real_parent)
        try:
            validate_paths(input_link / CHECKPOINT_FILENAME, output)
        except SystemExit:
            pass
        else:
            raise AssertionError("intermediate checkpoint symlink accepted")
        output_parent = root / "output-link"
        output_parent.symlink_to(real_parent)
        try:
            validate_paths(checkpoint, output_parent / "out.safetensors")
        except SystemExit:
            pass
        else:
            raise AssertionError("intermediate output symlink accepted")
    print(f"wespeaker_prepare_checkpoint self-test: OK ({len(manifest)} tensors, {len(counters)} counters)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.checkpoint is not None or args.output is not None:
            parser.error("--self-test accepts no checkpoint/output")
        return self_test()
    if args.checkpoint is None or args.output is None:
        parser.error("--checkpoint and --output are required")
    sidecar = validate_paths(args.checkpoint, args.output)
    if args.checkpoint.stat().st_size != CHECKPOINT_BYTES:
        raise SystemExit(f"checkpoint byte size does not match pinned {CHECKPOINT_BYTES}")
    source_hash = file_sha256(args.checkpoint)
    if source_hash != CHECKPOINT_SHA256:
        raise SystemExit(f"checkpoint SHA-256 {source_hash} != pinned {CHECKPOINT_SHA256}")

    import torch
    from safetensors.torch import save_file

    loaded = torch.load(str(args.checkpoint), map_location="cpu", weights_only=True)
    state = unwrap_state_dict(loaded)
    expected = expected_manifest()
    actual_names = set(state)
    expected_names = set(expected)
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing or extra or len(state) != TENSOR_COUNT:
        raise SystemExit(f"exact 219-name manifest mismatch: count={len(state)} missing={missing[:3]} extra={extra[:3]}")

    converted: dict[str, object] = {}
    source_manifest: dict[str, dict[str, Any]] = {}
    output_manifest: dict[str, dict[str, Any]] = {}
    counter_names: set[str] = set()
    for name in sorted(expected):
        tensor = state[name]
        shape = tuple(int(dimension) for dimension in tensor.shape)
        if shape != expected[name]:
            raise SystemExit(f"tensor {name!r} shape {shape} != expected {expected[name]}")
        dtype = str(tensor.dtype)
        source_manifest[name] = {"dtype": dtype, "shape": list(shape), "numel": int(tensor.numel())}
        if name.endswith(".num_batches_tracked"):
            if dtype != "torch.int64" or shape != ():
                raise SystemExit(f"counter {name!r} must be scalar torch.int64, got {dtype} {shape}")
            counter_names.add(name)
            converted_tensor = tensor.to(dtype=torch.float32).contiguous()
            output_dtype = "torch.float32"
        else:
            if dtype not in {"torch.float32", "torch.float16", "torch.bfloat16"}:
                raise SystemExit(f"inference tensor {name!r} has unsupported dtype {dtype}")
            converted_tensor = tensor.detach().cpu().contiguous()
            output_dtype = dtype
        converted[name] = converted_tensor
        output_manifest[name] = {"dtype": output_dtype, "shape": list(shape), "numel": int(tensor.numel())}
    if len(counter_names) != COUNTER_COUNT:
        raise SystemExit(f"expected {COUNTER_COUNT} scalar counters, got {len(counter_names)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(args.output))
    output_hash = file_sha256(args.output)
    manifest = {
        "format": "vokra-wespeaker-prepared-v1",
        "model_id": UPSTREAM_HF,
        "model_revision": UPSTREAM_REVISION,
        "source_revision": SOURCE_REVISION,
        "checkpoint_filename": CHECKPOINT_FILENAME,
        "checkpoint_sha256": source_hash,
        "output_sha256": output_hash,
        "tensor_count": TENSOR_COUNT,
        "counter_count": COUNTER_COUNT,
        "source_manifest_sha256": manifest_sha256(source_manifest),
        "output_manifest_sha256": manifest_sha256(output_manifest),
        "source_manifest": source_manifest,
        "output_manifest": output_manifest,
    }
    manifest_path = sidecar
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wespeaker_prepare_checkpoint: wrote {args.output}")
    print(f"wespeaker_prepare_checkpoint: checkpoint_sha256={source_hash}")
    print(f"wespeaker_prepare_checkpoint: output_sha256={output_hash}")
    print(f"wespeaker_prepare_checkpoint: tensors={TENSOR_COUNT} counters={COUNTER_COUNT}")
    print(f"wespeaker_prepare_checkpoint: manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
