#!/usr/bin/env python3
"""Dump an independent official Asteroid 0.7.0 Conv-TasNet reference.

The oracle loads the pinned upstream ``pytorch_model.bin`` through Asteroid's
own ``ConvTasNet.from_pretrained`` and calls its encoder, masker and decoder.
It never reads a Vokra GGUF and contains no local network-layer mirror.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
from pathlib import Path

UPSTREAM_HF = "JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k"
UPSTREAM_REVISION = "bb8a876bc157b5cf3c405994accb798c49146016"
CHECKPOINT_BYTES = 20_130_704
CHECKPOINT_SHA256 = "dd8ddefe95a35761f8a48643a618eba908572d04d33208a8ed5451fb5a4378d0"
SAMPLE_RATE = 16_000
PCM_SAMPLES = 4_096


def write_f32(path: Path, values: np.ndarray) -> None:
    path.write_bytes(np.asarray(values, dtype="<f4").tobytes(order="C"))


def deterministic_pcm() -> np.ndarray:
    index = np.arange(PCM_SAMPLES, dtype=np.float64)
    time = index / SAMPLE_RATE
    signal = (
        0.18 * np.sin(2.0 * math.pi * 191.0 * time)
        + 0.09 * np.sin(2.0 * math.pi * 503.0 * time + 0.21)
        + 0.035 * np.cos(2.0 * math.pi * 1201.0 * time)
    )
    signal *= np.minimum(1.0, index / 192.0)
    return signal.astype(np.float32)


@contextlib.contextmanager
def weights_only_torch_load():
    """Force Asteroid's checkpoint load through PyTorch safe deserialization."""
    original = torch.load

    def guarded(*args, **kwargs):
        if kwargs.get("weights_only") is False:
            raise RuntimeError("Conv-TasNet oracle refuses torch.load(weights_only=False)")
        kwargs["weights_only"] = True
        return original(*args, **kwargs)

    torch.load = guarded
    try:
        yield
    finally:
        torch.load = original


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_ancestors(path: Path) -> bool:
    absolute = Path(os.path.abspath(path))
    return all(not parent.is_symlink() for parent in absolute.parents)


def disjoint(left: Path, right: Path) -> bool:
    left, right = Path(os.path.abspath(left)), Path(os.path.abspath(right))
    return left != right and left not in right.parents and right not in left.parents


def self_test() -> None:
    assert (UPSTREAM_REVISION, CHECKPOINT_BYTES, CHECKPOINT_SHA256) == ("bb8a876bc157b5cf3c405994accb798c49146016", 20_130_704, "dd8ddefe95a35761f8a48643a618eba908572d04d33208a8ed5451fb5a4378d0")
    assert (SAMPLE_RATE, PCM_SAMPLES) == (16_000, 4_096)
    with __import__("tempfile").TemporaryDirectory(prefix="conv-tasnet-dump-") as raw:
        root = Path(raw)
        target = root / "target"
        target.mkdir()
        link = root / "link"
        link.symlink_to(target)
        assert not clean_ancestors(link / "nested" / "output")
        assert disjoint(root / "reference", target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if args.checkpoint is not None or args.output_dir is not None:
            parser.error("--self-test accepts no other arguments")
        self_test()
        print("conv_tasnet_dump_reference self-test: OK")
        return 0
    if args.checkpoint is None or args.output_dir is None:
        parser.error("normal runs require --checkpoint and --output-dir")

    import numpy as np
    import torch
    globals()["np"] = np
    globals()["torch"] = torch

    import asteroid
    from asteroid.models import ConvTasNet

    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)

    if args.checkpoint.is_symlink() or not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not clean_ancestors(args.checkpoint):
        raise SystemExit("checkpoint has a symlinked lexical ancestor")
    if args.checkpoint.stat().st_size != CHECKPOINT_BYTES or sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise SystemExit("checkpoint bytes/SHA-256 do not match the authenticated official identity")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise SystemExit("output directory must be absent and non-symlink")
    if not clean_ancestors(args.output_dir) or not disjoint(args.output_dir, args.checkpoint):
        raise SystemExit("output directory has a symlinked ancestor or overlaps checkpoint")
    with weights_only_torch_load():
        model = ConvTasNet.from_pretrained(str(args.checkpoint))
    model.eval()
    pcm = deterministic_pcm()
    waveform = torch.from_numpy(pcm).reshape(1, 1, -1)
    encoded = model.forward_encoder(waveform)
    bottleneck = model.masker.bottleneck(encoded)
    masks = model.forward_masker(encoded)
    masked = model.apply_masks(encoded, masks)
    decoded = model.forward_decoder(masked)
    separated = model(waveform)

    expected = {
        "encoded": (1, 512, 255),
        "bottleneck": (1, 128, 255),
        "masks": (1, 1, 512, 255),
        "decoded": (1, 1, 4096),
        "separated": (1, 1, 4096),
    }
    actual = {
        "encoded": tuple(encoded.shape),
        "bottleneck": tuple(bottleneck.shape),
        "masks": tuple(masks.shape),
        "decoded": tuple(decoded.shape),
        "separated": tuple(separated.shape),
    }
    if actual != expected:
        raise SystemExit(f"unexpected reference shapes: {actual!r}, expected {expected!r}")
    if not all(np.isfinite(values).all() for values in (pcm, encoded, bottleneck, masks, separated)):
        raise SystemExit("reference contains non-finite values")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_f32(output / "pcm.f32.bin", pcm)
    write_f32(output / "encoder.f32.bin", encoded[0].cpu().numpy())
    write_f32(output / "bottleneck.f32.bin", bottleneck[0].cpu().numpy())
    write_f32(output / "mask.f32.bin", masks[0, 0].cpu().numpy())
    write_f32(output / "separated.f32.bin", separated[0, 0].cpu().numpy())
    artifact_shapes = {
        "pcm.f32.bin": (4096,),
        "encoder.f32.bin": (512, 255),
        "bottleneck.f32.bin": (128, 255),
        "mask.f32.bin": (512, 255),
        "separated.f32.bin": (4096,),
    }
    artifacts = {
        name: {
            "bytes": math.prod(shape) * 4,
            "sha256": sha256(output / name),
            "shape": list(shape),
            "dtype": "float32-le",
        }
        for name, shape in artifact_shapes.items()
    }

    manifest = {
        "format": "vokra-conv-tasnet-reference-v1",
        "model_id": UPSTREAM_HF,
        "revision": UPSTREAM_REVISION,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256(args.checkpoint),
            "identity": "pinned-revision-path-supplied-by-VAST",
        },
        "sample_rate": SAMPLE_RATE,
        "pcm_samples": PCM_SAMPLES,
        "shapes": {name: list(shape) for name, shape in actual.items()},
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "asteroid": asteroid.__version__,
        "runtime_status": "MEASURED_NOT_GATED",
        "parity_status": "MEASURED_NOT_GATED",
        "tolerance": None,
        "artifacts": artifacts,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
