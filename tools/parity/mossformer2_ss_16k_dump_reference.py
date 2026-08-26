#!/usr/bin/env python3
"""Dump an independent official ClearerVoice MossFormer2 reference.

The oracle imports the model from an exact clean ClearerVoice-Studio checkout,
strictly loads the authenticated upstream PyTorch state dict, and calls the
official module.  It never imports Vokra and contains no mirror of the network
equations.  Execute the real model only on VAST; ``--self-test`` is stdlib-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SOURCE_REPOSITORY = "https://github.com/modelscope/ClearerVoice-Studio"
SOURCE_REVISION = "6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61"
UPSTREAM_HF = "alibabasglab/MossFormer2_SS_16K"
UPSTREAM_REVISION = "407cb030cd66340918ebb6c8cc63b18f8592cdbe"
CHECKPOINT_BYTES = 670_353_271
CHECKPOINT_SHA256 = (
    "00a3a48bda492db1e829b85dd443f8f43a43039a3e90f1a24962ea9caf14a11a"
)
SAMPLE_RATE = 16_000
PCM_SAMPLES = 4_096


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"missing pinned checkpoint: {path}")
    if path.stat().st_size != CHECKPOINT_BYTES:
        raise ValueError(
            f"checkpoint size {path.stat().st_size} != pinned {CHECKPOINT_BYTES}"
        )
    digest = sha256_file(path)
    if digest != CHECKPOINT_SHA256:
        raise ValueError(f"checkpoint SHA-256 {digest} != {CHECKPOINT_SHA256}")


def git_output(checkout: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def validate_source(checkout: Path) -> Path:
    checkout = checkout.resolve()
    if git_output(checkout, "rev-parse", "HEAD") != SOURCE_REVISION:
        raise ValueError(f"source checkout is not pinned revision {SOURCE_REVISION}")
    if git_output(checkout, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("official ClearerVoice-Studio checkout must be exactly clean")
    required = [
        "LICENSE",
        "clearvoice/clearvoice/models/mossformer2_ss/mossformer2.py",
        "clearvoice/clearvoice/models/mossformer2_ss/mossformer2_block.py",
        "clearvoice/clearvoice/models/mossformer2_ss/fsmn.py",
        "clearvoice/clearvoice/models/mossformer2_ss/conv_module.py",
        "clearvoice/clearvoice/models/mossformer2_ss/layer_norm.py",
    ]
    missing = [name for name in required if not (checkout / name).is_file()]
    if missing:
        raise ValueError(f"source checkout is incomplete: {missing}")
    return checkout


def checkpoint_state(payload: object) -> dict[str, Any]:
    import torch

    if not isinstance(payload, dict):
        raise ValueError("checkpoint wrapper is not a dict")
    state = payload.get("model") if "model" in payload else payload
    if not isinstance(state, dict):
        raise ValueError("checkpoint model entry is not a dict")
    if state and all(isinstance(key, str) and key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    if any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError("checkpoint state dict contains a non-tensor entry")
    return state


def reconcile_shared_rotary_aliases(
    state: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    """Reconcile only the aliases of the single shared RotaryEmbedding.

    PyTorch releases have differed in whether a shared submodule is repeated
    under every owning path in ``state_dict``.  The official source passes one
    RotaryEmbedding instance into all 24 FLASH layers.  Accept either spelling
    only after every serialized alias is exactly equal; no other missing or
    unexpected key is tolerated.
    """
    import torch

    base_name = (
        "mask_net.mdl.intra_mdl.mossformerM.layers.0.rotary_pos_emb.freqs"
    )
    aliases = {
        f"mask_net.mdl.intra_mdl.mossformerM.layers.{layer}.rotary_pos_emb.freqs"
        for layer in range(1, 24)
    }
    missing = set(expected) - set(state)
    unexpected = set(state) - set(expected)
    if (missing | unexpected) - aliases:
        raise ValueError(
            f"official state mismatch: missing={sorted(missing)[:8]}, "
            f"unexpected={sorted(unexpected)[:8]}"
        )
    base = state.get(base_name)
    if base is None:
        raise ValueError("official checkpoint is missing the shared layer-0 rotary tensor")
    for name in sorted(aliases & set(state)):
        alias = state[name]
        if not isinstance(alias, torch.Tensor) or tuple(alias.shape) != tuple(base.shape):
            raise ValueError(f"{name}: invalid shared rotary alias shape")
        if alias.dtype != base.dtype or not torch.equal(alias, base):
            raise ValueError(f"{name}: shared rotary alias differs from layer 0")
    reconciled = {name: value for name, value in state.items() if name in expected}
    for name in aliases & set(expected):
        reconciled.setdefault(name, base)
    if set(reconciled) != set(expected):
        raise ValueError("shared rotary reconciliation did not produce the official state")
    return reconciled


def deterministic_pcm(samples: int = PCM_SAMPLES) -> list[float]:
    values = []
    for index in range(samples):
        time = index / SAMPLE_RATE
        envelope = min(1.0, index / 192.0)
        values.append(
            envelope
            * (
                0.17 * math.sin(2.0 * math.pi * 173.0 * time + 0.11)
                + 0.08 * math.cos(2.0 * math.pi * 431.0 * time + 0.27)
                + 0.035 * math.sin(2.0 * math.pi * 997.0 * time)
            )
        )
    return values


def write_f32(path: Path, values: Any) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(values, dtype="<f4")
    path.write_bytes(array.tobytes(order="C"))
    return {
        "file": path.name,
        "dtype": "float32-le",
        "shape": list(array.shape),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def self_test() -> None:
    values = deterministic_pcm(32)
    payload = struct.pack(f"<{len(values)}f", *values)
    digest = hashlib.sha256(payload).hexdigest()
    if len(values) != 32 or not all(math.isfinite(value) for value in values):
        raise ValueError("deterministic PCM self-test failed")
    print(json.dumps({"samples": len(values), "sha256": digest}, sort_keys=True))


def dump(args: argparse.Namespace) -> None:
    import numpy as np
    import torch

    source = validate_source(args.source)
    validate_checkpoint(args.checkpoint)
    if args.output.exists():
        raise ValueError(f"output directory already exists: {args.output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requested but torch.cuda.is_available() is false")

    torch.set_grad_enabled(False)
    torch.manual_seed(1_234)
    torch.set_num_threads(1)
    np.random.seed(1_234)
    sys.dont_write_bytecode = True
    package_root = source / "clearvoice"
    sys.path.insert(0, str(package_root))
    try:
        from clearvoice.models.mossformer2_ss.mossformer2 import MossFormer2_SS_16K
        imported_module = sys.modules[MossFormer2_SS_16K.__module__]
    finally:
        sys.path.pop(0)
    imported = Path(imported_module.__file__).resolve()
    if source not in imported.parents:
        raise ValueError(f"imported official model from {imported}, outside {source}")

    model_args = SimpleNamespace(
        encoder_embedding_dim=512,
        mossformer_sequence_dim=512,
        num_mossformer_layer=24,
        encoder_kernel_size=16,
        num_spks=2,
    )
    model = MossFormer2_SS_16K(model_args).model
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint_state(payload)
    expected = model.state_dict()
    state = reconcile_shared_rotary_aliases(state, expected)
    for name, tensor in expected.items():
        if tuple(state[name].shape) != tuple(tensor.shape):
            raise ValueError(
                f"{name}: checkpoint shape {tuple(state[name].shape)} != official {tuple(tensor.shape)}"
            )
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    taps: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"tap {name} emitted {type(output).__name__}")
            taps[name] = output.detach().cpu().contiguous()

        return hook

    handles = [
        model.enc.register_forward_hook(capture("encoder")),
        model.mask_net.mdl.intra_mdl.mossformerM.layers[0].register_forward_hook(
            capture("attention_0")
        ),
        model.mask_net.mdl.intra_mdl.mossformerM.fsmn[0].register_forward_hook(
            capture("fsmn_0")
        ),
        model.mask_net.register_forward_hook(capture("mask")),
    ]
    pcm = torch.tensor(
        deterministic_pcm(), dtype=torch.float32, device=device
    ).reshape(1, -1)
    with torch.inference_mode():
        outputs = model(pcm)
    for handle in handles:
        handle.remove()
    if not isinstance(outputs, list) or len(outputs) != 2:
        raise ValueError("official model did not emit two output streams")
    expected_shapes = {
        "encoder": (1, 512, 511),
        "attention_0": (1, 511, 512),
        "fsmn_0": (1, 511, 512),
        "mask": (2, 1, 512, 511),
    }
    actual_shapes = {name: tuple(value.shape) for name, value in taps.items()}
    if actual_shapes != expected_shapes:
        raise ValueError(
            f"official tap shapes {actual_shapes} != expected {expected_shapes}"
        )
    for stream, output in enumerate(outputs):
        if tuple(output.shape) != (1, PCM_SAMPLES):
            raise ValueError(f"stream {stream} shape {tuple(output.shape)} is unexpected")
        if not bool(torch.isfinite(output).all()):
            raise ValueError(f"stream {stream} contains non-finite values")

    args.output.mkdir(parents=True)
    files = {
        "pcm": write_f32(args.output / "pcm.f32.bin", pcm[0].numpy()),
        **{
            name: write_f32(args.output / f"{name}.f32.bin", tensor.numpy())
            for name, tensor in taps.items()
        },
        "separated": write_f32(
            args.output / "separated.f32.bin",
            torch.stack([output[0] for output in outputs]).cpu().numpy(),
        ),
    }
    manifest = {
        "format": "vokra-mossformer2-ss-16k-reference-v1",
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "upstream_hf": UPSTREAM_HF,
        "upstream_revision": UPSTREAM_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "sample_rate": SAMPLE_RATE,
        "pcm_samples": PCM_SAMPLES,
        "files": files,
        "numeric_bounds": "UNSET_MEASURE_ON_VAST_BEFORE_RATIFICATION",
        "device": str(device),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.source, args.checkpoint, args.output)):
            parser.error("--self-test does not accept model inputs")
        if args.device != "cpu":
            parser.error("--self-test does not accept --device")
        self_test()
        return 0
    if any(value is None for value in (args.source, args.checkpoint, args.output)):
        parser.error("--source, --checkpoint and --output are required")
    dump(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
