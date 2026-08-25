#!/usr/bin/env python3
"""Prepare the exact public NISQA v2 multidimensional checkpoint.

The upstream release is a trusted torch pickle. This offline-only sidecar pins
the source tree and checkpoint hashes, validates the checkpoint-derived args,
removes only BatchNorm ``num_batches_tracked`` counters, and emits the exact 94
F32 tensors accepted by the strict Rust converter. Run real work on VAST.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


SOURCE_REVISION = "fe84f0f252abec382b24367d5b22498a7ce34dbb"
CHECKPOINT_SHA256 = "7ec4cf937514dd3f8860b21e66fabd8ca87a168572675ef8d979c4c4ad2e805c"
TENSOR_COUNT = 94
SOURCE_FILES = {
    Path("nisqa/NISQA_lib.py"): (
        77_206,
        "f3ace1c00e21ae06e5d0fed9710f4e988c13685b2316a3b3ded46607fb25b71e",
    ),
    Path("config/train_nisqa_cnn_sa_ap.yaml"): (
        None,
        "afa752835c45f5d052787c024b10eab26eba980e0bde85632e674dbe557ec764",
    ),
    Path("LICENSE"): (
        1_098,
        "6c6c762447306a0fa89b130d5df177a5c2a79f39fc8cbf58aad68c5245da3f16",
    ),
    Path("weights/LICENSE_model_weights"): (
        20_843,
        "5b8e7938e1b5e0a675869ffe429cc8e7cc187d76a7c6ea1e0546c412782a43da",
    ),
}
EXPECTED_ARGS = {
    "model": "NISQA_DIM",
    "ms_sr": None,
    "ms_fmax": 20_000,
    "ms_n_fft": 4_096,
    "ms_hop_length": 0.01,
    "ms_win_length": 0.02,
    "ms_n_mels": 48,
    "ms_seg_length": 15,
    "ms_seg_hop_length": 4,
    "ms_max_segments": 1_300,
    "cnn_model": "adapt",
    "cnn_c_out_1": 16,
    "cnn_c_out_2": 32,
    "cnn_c_out_3": 64,
    "cnn_kernel_size": [3, 3],
    "cnn_dropout": 0.2,
    "cnn_pool_1": [24, 7],
    "cnn_pool_2": [12, 5],
    "cnn_pool_3": [6, 3],
    "cnn_fc_out_h": None,
    "td": "self_att",
    "td_sa_d_model": 64,
    "td_sa_nhead": 1,
    "td_sa_pos_enc": None,
    "td_sa_num_layers": 2,
    "td_sa_h": 64,
    "td_sa_dropout": 0.1,
    "td_2": "skip",
    "pool": "att",
    "pool_att_h": 128,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(checkout: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def validate_source(checkout: Path) -> None:
    if git_output(checkout, "rev-parse", "HEAD") != SOURCE_REVISION:
        raise ValueError("NISQA source revision differs from the pinned release")
    if git_output(checkout, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("NISQA source checkout must be exactly clean")
    for relative, (expected_bytes, expected_sha256) in SOURCE_FILES.items():
        path = checkout / relative
        if expected_bytes is not None and path.stat().st_size != expected_bytes:
            raise ValueError(f"{relative}: unexpected byte length")
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"{relative}: unexpected SHA-256")


def normalize(value: object) -> object:
    if isinstance(value, tuple):
        return [normalize(item) for item in value]
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return value


def validate_args(args: dict[str, object]) -> None:
    for key, expected in EXPECTED_ARGS.items():
        actual = normalize(args.get(key))
        if actual != expected:
            raise ValueError(f"checkpoint arg {key}={actual!r}, expected {expected!r}")


def prepare(source: Path, checkpoint_path: Path, output: Path, manifest: Path) -> None:
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as error:
        raise SystemExit(
            "run with `uv run --project tools/parity python "
            "tools/parity/nisqa_v2_weight_prepare_checkpoint.py ...`"
        ) from error

    validate_source(source.resolve())
    if sha256_file(checkpoint_path) != CHECKPOINT_SHA256:
        raise ValueError("weights/nisqa.tar SHA-256 differs from the pinned release")
    if output.exists() or manifest.exists():
        raise ValueError("refusing to overwrite prepared output")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("official checkpoint root is not a dictionary")
    args = checkpoint.get("args")
    state = checkpoint.get("model_state_dict")
    if not isinstance(args, dict) or not isinstance(state, dict):
        raise TypeError("official checkpoint lacks args/model_state_dict dictionaries")
    validate_args(args)

    tensors = {}
    dropped = []
    for name, tensor in state.items():
        if name.endswith(".num_batches_tracked"):
            dropped.append(name)
            continue
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state entry {name!r} is not a tensor")
        if tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"state tensor {name!r} must be finite F32")
        tensors[name] = tensor.detach().cpu().contiguous()
    if len(tensors) != TENSOR_COUNT or len(dropped) != 6:
        raise ValueError(
            f"prepared tensor inventory is {len(tensors)} F32 + {len(dropped)} counters, "
            f"expected {TENSOR_COUNT} + 6"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(dict(sorted(tensors.items())), output)
    record = {
        "source_revision": SOURCE_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "safetensors_sha256": sha256_file(output),
        "tensor_count": len(tensors),
        "dropped_num_batches_tracked": sorted(dropped),
        "checkpoint_args": {key: normalize(args.get(key)) for key in EXPECTED_ARGS},
    }
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"{output.name} {record['safetensors_sha256']}")


def self_test() -> None:
    assert len(SOURCE_REVISION) == 40
    assert len(CHECKPOINT_SHA256) == 64
    assert TENSOR_COUNT == 94
    assert EXPECTED_ARGS["cnn_pool_3"] == [6, 3]
    assert EXPECTED_ARGS["td_sa_nhead"] == 1
    assert EXPECTED_ARGS["ms_sr"] is None
    print("nisqa_v2_weight_prepare_checkpoint: self-test OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if None in (args.source, args.checkpoint, args.output):
        raise SystemExit("--source, --checkpoint, and --output are required")
    manifest = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    prepare(args.source, args.checkpoint, args.output, manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
