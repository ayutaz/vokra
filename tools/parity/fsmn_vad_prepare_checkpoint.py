#!/usr/bin/env python3
"""Prepare the pinned FunASR FSMN-VAD checkpoint for strict GGUF conversion.

The official release splits its 24 PyTorch weights, Kaldi ``am.mvn`` affine
transform, and topology config across three files.  This offline-only bridge
verifies all three source hashes, unwraps the state dict, validates the exact
released tensor manifest, and embeds the two CMVN vectors as reserved
safetensors tensors.  The Rust converter consumes those vectors as GGUF
metadata and refuses an ordinary weight-only safetensors file.

Run only through the repository's uv-managed Python 3.12 environment:

    uv run --project tools/parity python tools/parity/fsmn_vad_prepare_checkpoint.py \
      --model-pt model.pt --cmvn am.mvn --config config.yaml \
      --output fsmn-vad.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

MODEL_REVISION = "df20e6b30c653645fa4ff125cacfcabd1020a669"
MODEL_SHA256 = "b3be75be477f0780277f3bae0fe489f48718f585f3a6e45d7dd1fbb1a4255fc5"
CMVN_SHA256 = "df189fd5f4352df84a0fd464eeab4e450a5e645665d6b38f13c832492261a739"
CONFIG_SHA256 = "486861ca26ddb79081663b6179cb204c6bfae71c52f04aafc48a9e9d8dde1e93"

CMVN_ADD_SHIFT = "__vokra__.fsmn_vad.cmvn_add_shift"
CMVN_RESCALE = "__vokra__.fsmn_vad.cmvn_rescale"

EXPECTED_SHAPES: dict[str, tuple[int, ...]] = {
    "encoder.in_linear1.linear.weight": (140, 400),
    "encoder.in_linear1.linear.bias": (140,),
    "encoder.in_linear2.linear.weight": (250, 140),
    "encoder.in_linear2.linear.bias": (250,),
    "encoder.out_linear1.linear.weight": (140, 250),
    "encoder.out_linear1.linear.bias": (140,),
    "encoder.out_linear2.linear.weight": (248, 140),
    "encoder.out_linear2.linear.bias": (248,),
}
for block in range(4):
    prefix = f"encoder.fsmn.{block}"
    EXPECTED_SHAPES[f"{prefix}.linear.linear.weight"] = (128, 250)
    EXPECTED_SHAPES[f"{prefix}.fsmn_block.conv_left.weight"] = (128, 1, 20, 1)
    EXPECTED_SHAPES[f"{prefix}.affine.linear.weight"] = (250, 128)
    EXPECTED_SHAPES[f"{prefix}.affine.linear.bias"] = (250,)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(
            f"{label} SHA-256 mismatch for {path}: got {actual}, expected {expected}"
        )


def parse_cmvn(path: Path) -> tuple[list[float], list[float]]:
    lines = path.read_text(encoding="utf-8").splitlines()

    def vector_after(marker: str) -> list[float]:
        for index, line in enumerate(lines[:-1]):
            if line.split()[:1] != [marker]:
                continue
            tokens = lines[index + 1].split()
            if len(tokens) < 5 or tokens[:3] != ["<LearnRateCoef>", "0", "["] or tokens[-1] != "]":
                raise SystemExit(f"{path}: malformed vector after {marker}")
            try:
                values = [float(token) for token in tokens[3:-1]]
            except ValueError as error:
                raise SystemExit(f"{path}: non-float value after {marker}: {error}") from error
            if len(values) != 400:
                raise SystemExit(
                    f"{path}: {marker} has {len(values)} values, expected 400"
                )
            if not all(math.isfinite(value) for value in values):
                raise SystemExit(f"{path}: {marker} contains a non-finite value")
            return values
        raise SystemExit(f"{path}: missing {marker}")

    add_shift = vector_after("<AddShift>")
    rescale = vector_after("<Rescale>")
    if not all(value > 0.0 for value in rescale):
        raise SystemExit(f"{path}: Rescale must contain 400 positive values")
    return add_shift, rescale


def unwrap_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"model.pt root must be a dict, got {type(checkpoint)!r}")
    for key in ("state_dict", "model_state_dict", "model", "module"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict) and candidate:
            first = next(iter(candidate.values()))
            if hasattr(first, "dtype") and hasattr(first, "shape"):
                return candidate
    return checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-pt", type=Path, required=True)
    parser.add_argument("--cmvn", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    require_hash(args.model_pt, MODEL_SHA256, "model.pt")
    require_hash(args.cmvn, CMVN_SHA256, "am.mvn")
    require_hash(args.config, CONFIG_SHA256, "config.yaml")

    import torch
    from safetensors.torch import save_file

    checkpoint = torch.load(args.model_pt, map_location="cpu", weights_only=True)
    state = unwrap_state_dict(checkpoint)
    actual_names = {name for name, value in state.items() if hasattr(value, "shape")}
    expected_names = set(EXPECTED_SHAPES)
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing or extra:
        raise SystemExit(f"FSMN tensor manifest mismatch: missing={missing}, extra={extra}")

    prepared: dict[str, torch.Tensor] = {}
    for name, shape in EXPECTED_SHAPES.items():
        tensor = state[name]
        actual_shape = tuple(int(axis) for axis in tensor.shape)
        if actual_shape != shape:
            raise SystemExit(f"{name}: shape {actual_shape}, expected {shape}")
        if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise SystemExit(f"{name}: unsupported dtype {tensor.dtype}")
        prepared[name] = tensor.detach().contiguous()

    add_shift, rescale = parse_cmvn(args.cmvn)
    prepared[CMVN_ADD_SHIFT] = torch.tensor(add_shift, dtype=torch.float32)
    prepared[CMVN_RESCALE] = torch.tensor(rescale, dtype=torch.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        prepared,
        str(args.output),
        metadata={
            "format": "pt",
            "vokra.source_revision": MODEL_REVISION,
            "vokra.model_sha256": MODEL_SHA256,
            "vokra.cmvn_sha256": CMVN_SHA256,
            "vokra.config_sha256": CONFIG_SHA256,
        },
    )
    print(
        f"fsmn_vad_prepare_checkpoint: wrote {len(prepared)} tensors "
        f"(24 weights + 2 CMVN vectors) to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
