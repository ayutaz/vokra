#!/usr/bin/env python3
"""Dump an independent official YuE-upsampler feature-decoder reference.

The oracle imports ``VocosBackbone`` and ``ISTFTHead`` directly from the
released ``vocos==0.1.0`` wheel pinned by ``tools/parity/uv.lock``. It never
calls Vokra code. The exact ``m-a-p/YuE-upsampler`` 151k checkpoint is pinned
by revision, byte length, and SHA-256 before restricted PyTorch deserialization.
No unsafe pickle fallback is permitted; unsupported pickle globals fail closed
before any output is created.

Run only through the repository parity environment, normally on VAST::

    uv run --project tools/parity python \
      tools/parity/yue_upsampler_dump_reference.py \
      --checkpoint decoder_151000.pth \
      --output-dir /path/to/reference
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


UPSTREAM_HF = "m-a-p/YuE-upsampler"
UPSTREAM_REVISION = "c6d7494a60555672be09ca809a40be400d682a53"
CHECKPOINT_FILE = "decoder_151000.pth"
CHECKPOINT_BYTES = 72_610_550
CHECKPOINT_SHA256 = (
    "8af97a29d3483f9d4a3755992837501bd7d6caa1a69382ed16e64039e0ea0998"
)
SOURCE_PACKAGE = "vocos==0.1.0"
SOURCE_PACKAGE_WHEEL_SHA256 = (
    "0ac13eaef68596074301e912d781399b3defa4b4ca60b6bc52c8a4b9209ca235"
)

INPUT_CHANNELS = 1024
DIM = 512
INTERMEDIATE_DIM = 1536
NUM_LAYERS = 8
N_FFT = 3528
HOP_LENGTH = 882
SAMPLE_RATE = 44_100
PADDING = "same"
BLOCKED_UNSAFE_PICKLE = "BLOCKED_UNSAFE_PICKLE"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: Path) -> None:
    if path.name != CHECKPOINT_FILE:
        raise ValueError(
            f"checkpoint filename {path.name!r}, expected {CHECKPOINT_FILE!r}"
        )
    size = path.stat().st_size
    if size != CHECKPOINT_BYTES:
        raise ValueError(
            f"checkpoint has {size} bytes, expected {CHECKPOINT_BYTES}"
        )
    actual = sha256_file(path)
    if actual != CHECKPOINT_SHA256:
        raise ValueError(
            f"checkpoint SHA-256 {actual}, expected {CHECKPOINT_SHA256}"
        )


def unwrap_state_dict(raw: object) -> dict:
    if not isinstance(raw, dict):
        raise TypeError("checkpoint must contain a dict")
    for wrapper in ("state_dict", "model_state_dict", "model", "module"):
        inner = raw.get(wrapper)
        if isinstance(inner, dict) and inner:
            raw = inner
            break
    if not isinstance(raw, dict) or not raw:
        raise TypeError("checkpoint yielded no state dict")
    return raw


def _assert_checkpoint_loader_contract() -> None:
    """Keep checkpoint deserialization restricted against future regressions."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    load_calls = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            if (
                isinstance(function, ast.Attribute)
                and function.attr == "load"
                and isinstance(function.value, ast.Name)
                and function.value.id == "torch"
            ):
                load_calls += 1
                weights_only = [
                    keyword
                    for keyword in node.keywords
                    if keyword.arg == "weights_only"
                ]
                assert len(weights_only) == 1
                assert isinstance(weights_only[0].value, ast.Constant)
                assert weights_only[0].value.value is True
            if (
                isinstance(function, ast.Name)
                and function.id == "setattr"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "torch"
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == "load"
            ):
                raise AssertionError("torch.load monkeypatch is forbidden")
            if (
                isinstance(function, ast.Attribute)
                and function.attr == "object"
                and isinstance(function.value, ast.Name)
                and function.value.id == "patch"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "torch"
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == "load"
            ):
                raise AssertionError("torch.load monkeypatch is forbidden")
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if isinstance(node, ast.AugAssign):
                targets = [node.target]
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "load"
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "torch"
                ):
                    raise AssertionError("torch.load monkeypatch is forbidden")
        if isinstance(node, ast.ImportFrom):
            assert all(alias.name != "Unpickler" for alias in node.names)
        if isinstance(node, (ast.Name, ast.Attribute)):
            name = node.id if isinstance(node, ast.Name) else node.attr
            assert not name.endswith("Unpickler")
    assert load_calls > 0


def self_test() -> int:
    assert len(UPSTREAM_REVISION) == 40
    assert len(CHECKPOINT_SHA256) == 64
    assert len(SOURCE_PACKAGE_WHEEL_SHA256) == 64
    assert N_FFT % 2 == 0
    assert N_FFT // HOP_LENGTH == 4
    assert SAMPLE_RATE // HOP_LENGTH == 50
    assert BLOCKED_UNSAFE_PICKLE == "BLOCKED_UNSAFE_PICKLE"
    _assert_checkpoint_loader_contract()
    print("yue_upsampler_dump_reference self-test: ok")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.checkpoint is None or args.output_dir is None:
        parser.error("--checkpoint and --output-dir are required")
    if args.frames <= 0:
        parser.error("--frames must be positive")

    verify_checkpoint(args.checkpoint)

    import torch
    from vocos.heads import ISTFTHead
    from vocos.models import VocosBackbone

    torch.set_num_threads(1)
    torch.manual_seed(0)
    try:
        raw = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001 - fail-closed deserialization boundary
        raise RuntimeError(
            f"{BLOCKED_UNSAFE_PICKLE}: YuE-upsampler checkpoint requires pickle "
            "globals outside PyTorch's restricted weights_only loader; unsafe "
            "fallback is forbidden"
        ) from error
    state = unwrap_state_dict(raw)

    backbone = VocosBackbone(
        input_channels=INPUT_CHANNELS,
        dim=DIM,
        intermediate_dim=INTERMEDIATE_DIM,
        num_layers=NUM_LAYERS,
        adanorm_num_embeddings=None,
    )
    head = ISTFTHead(
        dim=DIM,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        padding=PADDING,
    )
    backbone_state = {
        key.removeprefix("backbone."): value
        for key, value in state.items()
        if key.startswith("backbone.")
    }
    head_state = {
        key.removeprefix("head."): value
        for key, value in state.items()
        if key.startswith("head.")
    }
    backbone.load_state_dict(backbone_state, strict=True)
    head.load_state_dict(head_state, strict=True)
    expected = {
        *(f"backbone.{key}" for key in backbone.state_dict()),
        *(f"head.{key}" for key in head.state_dict()),
    }
    actual = set(state)
    if actual != expected:
        missing = sorted(expected - actual)[:5]
        extra = sorted(actual - expected)[:5]
        raise ValueError(
            f"official tensor manifest mismatch: missing={missing}, extra={extra}"
        )
    if len(actual) != 81:
        raise ValueError(f"official tensor count {len(actual)}, expected 81")

    backbone.eval()
    head.eval()
    grid = torch.arange(
        INPUT_CHANNELS * args.frames, dtype=torch.float32
    ).reshape(1, INPUT_CHANNELS, args.frames)
    features = 0.35 * torch.sin(grid * 0.017) + 0.15 * torch.cos(grid * 0.031)
    with torch.inference_mode():
        backbone_output = backbone(features)
        waveform = head(backbone_output)[0].contiguous()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    features[0].contiguous().numpy().astype("<f4", copy=False).tofile(
        args.output_dir / "features.f32le"
    )
    backbone_output[0].contiguous().numpy().astype("<f4", copy=False).tofile(
        args.output_dir / "backbone.f32le"
    )
    waveform.numpy().astype("<f4", copy=False).tofile(
        args.output_dir / "waveform.f32le"
    )
    metadata = {
        "upstream_hf": UPSTREAM_HF,
        "upstream_revision": UPSTREAM_REVISION,
        "checkpoint_file": CHECKPOINT_FILE,
        "checkpoint_bytes": CHECKPOINT_BYTES,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_package": SOURCE_PACKAGE,
        "source_package_wheel_sha256": SOURCE_PACKAGE_WHEEL_SHA256,
        "frames": args.frames,
        "input_channels": INPUT_CHANNELS,
        "dim": DIM,
        "intermediate_dim": INTERMEDIATE_DIM,
        "num_layers": NUM_LAYERS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "padding": PADDING,
        "sample_rate": SAMPLE_RATE,
        "samples": waveform.numel(),
        "torch": torch.__version__,
        "tensor_count": len(actual),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
