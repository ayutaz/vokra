#!/usr/bin/env python3
"""Prepare an operator-held ESPnet JSUT VITS checkpoint for Vokra.

This tool never downloads a checkpoint. The canonical public model was trained
on JSUT, whose corpus terms prohibit redistribution, so the input must be a
local file the operator is permitted to use. The script safely extracts only
the inference-time ``VITSGenerator`` state dict, strips ESPnet wrapper prefixes,
and writes a safetensors file for ``vokra-cli convert --model vits-ja``.

The accepted manifest is derived from the official 22.05 kHz release recipe at
ESPnet commit ``628b46282537ce532d613d6bafb75e826e8455de`` (Zenodo record
5521354): 885 tensors and 42,011,890 parameters. The manifest SHA-256 covers
sorted tensor names and dimensions, not payload bytes.

Usage::

    uv run python tools/parity/vits_ja_prepare_checkpoint.py \
        --input /secure/path/train.total_count.ave.pth \
        --output /secure/path/vits-ja-generator.safetensors

The output remains corpus-restricted. Do not upload or redistribute it.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping
from pathlib import Path

EXPECTED_TENSOR_COUNT = 885
EXPECTED_PARAMETER_COUNT = 42_011_890
EXPECTED_MANIFEST_SHA256 = (
    "b5d039b6f6febfcb93f2ad17f1647311bb0c37869f54b5e5ceac23f7b951b284"
)

WRAPPER_KEYS = ("model", "state_dict", "weights", "module")
GENERATOR_PREFIXES = (
    "module.model.tts.generator.",
    "module.tts.generator.",
    "model.tts.generator.",
    "tts.generator.",
    "module.generator.",
    "model.generator.",
    "generator.",
)
DIRECT_ROOTS = (
    "text_encoder.",
    "decoder.",
    "posterior_encoder.",
    "flow.",
    "duration_predictor.",
)


def unwrap_checkpoint(value: object) -> Mapping[str, object]:
    """Unwrap only well-known tensor-state containers, fail closed otherwise."""

    if not isinstance(value, Mapping):
        raise SystemExit(
            "vits_ja_prepare_checkpoint: checkpoint root is not a mapping; "
            "refusing unsafe or unknown structure"
        )
    current = value
    for _ in range(3):
        if all(isinstance(key, str) for key in current):
            tensor_like = [key for key in current if key.startswith(DIRECT_ROOTS)]
            prefixed = [
                key for key in current if key.startswith(GENERATOR_PREFIXES)
            ]
            if tensor_like or prefixed:
                return current
        nested = [
            current[key]
            for key in WRAPPER_KEYS
            if key in current and isinstance(current[key], Mapping)
        ]
        if len(nested) != 1:
            break
        current = nested[0]
    raise SystemExit(
        "vits_ja_prepare_checkpoint: could not find a VITS generator under "
        "the known ESPnet model/state_dict wrappers"
    )


def normalized_generator_state(state: Mapping[str, object], torch: object) -> dict[str, object]:
    """Select generator tensors and normalize their names to VITSGenerator keys."""

    output: dict[str, object] = {}
    for key, value in state.items():
        if not isinstance(key, str):
            raise SystemExit(
                "vits_ja_prepare_checkpoint: state dict contains a non-string key"
            )
        normalized: str | None = None
        if key.startswith(DIRECT_ROOTS):
            normalized = key
        else:
            for prefix in GENERATOR_PREFIXES:
                if key.startswith(prefix):
                    normalized = key.removeprefix(prefix)
                    break
        if normalized is None:
            continue
        if not isinstance(value, torch.Tensor):
            raise SystemExit(
                f"vits_ja_prepare_checkpoint: {key!r} is "
                f"{type(value).__name__}, not a tensor"
            )
        if normalized in output:
            raise SystemExit(
                f"vits_ja_prepare_checkpoint: duplicate normalized key {normalized!r}"
            )
        if not value.dtype.is_floating_point:
            raise SystemExit(
                f"vits_ja_prepare_checkpoint: {key!r} has non-floating dtype "
                f"{value.dtype}; refusing a partial manifest"
            )
        output[normalized] = value.detach().cpu().contiguous().clone()
    return output


def manifest_sha256(state: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(tensor.shape).to_bytes(8, "little"))
        for dimension in tensor.shape:
            digest.update(int(dimension).to_bytes(8, "little"))
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"--input is not a regular file: {args.input}")
    if args.output.exists():
        parser.error(f"refusing to overwrite --output: {args.output}")

    import torch
    from safetensors.torch import save_file

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=True)
    state = normalized_generator_state(unwrap_checkpoint(checkpoint), torch)
    tensor_count = len(state)
    parameter_count = sum(tensor.numel() for tensor in state.values())
    manifest = manifest_sha256(state)

    failures: list[str] = []
    if tensor_count != EXPECTED_TENSOR_COUNT:
        failures.append(
            f"tensor count {tensor_count} != expected {EXPECTED_TENSOR_COUNT}"
        )
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        failures.append(
            f"parameter count {parameter_count} != expected {EXPECTED_PARAMETER_COUNT}"
        )
    if manifest != EXPECTED_MANIFEST_SHA256:
        failures.append(
            f"manifest SHA-256 {manifest} != expected {EXPECTED_MANIFEST_SHA256}"
        )
    if failures:
        raise SystemExit(
            "vits_ja_prepare_checkpoint: input is not the canonical 22.05 kHz "
            "ESPnet JSUT VITS generator:\n  - " + "\n  - ".join(failures)
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, args.output)
    output_sha256 = file_sha256(args.output)
    print(f"vits_ja_prepare_checkpoint: wrote {args.output}")
    print(f"vits_ja_prepare_checkpoint: tensors={tensor_count}")
    print(f"vits_ja_prepare_checkpoint: parameters={parameter_count}")
    print(f"vits_ja_prepare_checkpoint: manifest_sha256={manifest}")
    print(f"vits_ja_prepare_checkpoint: file_sha256={output_sha256}")
    print(
        "vits_ja_prepare_checkpoint: CORPUS-RESTRICTED — do not upload or "
        "redistribute the output"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
