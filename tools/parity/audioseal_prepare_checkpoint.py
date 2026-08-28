#!/usr/bin/env python3
"""Flatten the four pinned AudioSeal checkpoints into one safetensors file.

This is an offline-only bridge for the Rust converter.  Run it through the
repository Python policy, normally on a provisioned VAST instance::

    uv run --no-project --python 3.12 --with torch \
      python tools/parity/audioseal_prepare_checkpoint.py \
      --generator-base generator_base.pth \
      --detector-base detector_base.pth \
      --generator-streaming generator_streaming.pth \
      --detector-streaming detector_streaming.pth \
      --output audioseal-four-checkpoints.safetensors

The source files are deserialized only after their official SHA-256 digests
match.  At most one checkpoint state dict is resident at a time.  The output
contains raw F32 state-dict tensors; it never imports the AudioSeal package and
does not run inference.  Rust performs the final exact 310-name/shape gate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


CHECKPOINT_REVISION = "3c19eba53390776cf2cc9ed5f6c9ac67ce72ecba"
SOURCE_REVISION = "e63a8a0e5cdf7bb797159c92ba15961557fe9bd2"

INPUTS = {
    "generator_base": (
        "generator_base.pth",
        "7a845b5fbe9364a63a3909d8ab3fe064d13a76ae4c2e983573e08c69b7b51748",
        101,
    ),
    "detector_base": (
        "detector_base.pth",
        "8a78e8a83584113523e161fc599fcab10fd0e94c04d2eb9d2fa1e9ec91ab69d9",
        54,
    ),
    "generator_streaming": (
        "generator_streaming.pth",
        "f5eb3076c1748940578993edd18ecd4fbdbc387fe7613a77f420af921d83eb74",
        101,
    ),
    "detector_streaming": (
        "detector_streaming.pth",
        "b78b017411b661d77b1e37402b00af6ce5319bfba9a69590489c8bcd0e657e4c",
        54,
    ),
}


@dataclass(frozen=True)
class TensorSpec:
    output_name: str
    source_prefix: str
    source_name: str
    shape: tuple[int, ...]
    nbytes: int
    begin: int
    end: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def require_source(path: Path, prefix: str) -> None:
    filename, expected_digest, _ = INPUTS[prefix]
    if not path.is_file():
        raise ValueError(f"{prefix}: source is not a file: {path}")
    digest = sha256_file(path)
    if digest != expected_digest:
        raise ValueError(
            f"{prefix}: SHA-256 {digest} != pinned {expected_digest} for "
            f"facebook/audioseal/{filename} at {CHECKPOINT_REVISION}"
        )


def load_state(path: Path) -> Mapping[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "torch is required only for this offline pickle bridge; run with "
            "`uv run --no-project --python 3.12 --with torch python ...` on VAST"
        ) from error

    # Deserializing pickle is acceptable only after require_source verified the
    # exact official digest.  The public checkpoint carries xp.cfg objects, so
    # weights_only=True cannot represent the complete outer mapping.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{path}: checkpoint root is not a mapping")
    if "best_state" in checkpoint:
        best = checkpoint["best_state"]
        if not isinstance(best, Mapping) or "model" not in best:
            raise ValueError(f"{path}: malformed best_state.model checkpoint")
        checkpoint = best["model"]
    elif "model" in checkpoint:
        checkpoint = checkpoint["model"]
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{path}: resolved state dict is not a mapping")
    if not checkpoint:
        raise ValueError(f"{path}: resolved state dict is empty")
    return checkpoint


def inspect_one(path: Path, prefix: str, begin: int) -> tuple[list[TensorSpec], int]:
    import torch

    state = load_state(path)
    _, _, expected_count = INPUTS[prefix]
    if len(state) != expected_count:
        raise ValueError(
            f"{prefix}: state dict has {len(state)} entries, expected {expected_count}"
        )
    specs: list[TensorSpec] = []
    offset = begin
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not torch.is_tensor(value):
            raise ValueError(f"{prefix}: non-tensor state entry {name!r}")
        tensor = value.detach()
        if tensor.dtype != torch.float32:
            raise ValueError(
                f"{prefix}.{name}: dtype {tensor.dtype}, expected torch.float32"
            )
        shape = tuple(int(dim) for dim in tensor.shape)
        nbytes = tensor.numel() * 4
        specs.append(
            TensorSpec(
                output_name=f"{prefix}.{name}",
                source_prefix=prefix,
                source_name=name,
                shape=shape,
                nbytes=nbytes,
                begin=offset,
                end=offset + nbytes,
            )
        )
        offset += nbytes
    del state
    gc.collect()
    return specs, offset


def padded_header(specs: list[TensorSpec], input_digests: Mapping[str, str]) -> bytes:
    header: dict[str, Any] = {
        "__metadata__": {
            "vokra.audioseal.checkpoint_revision": CHECKPOINT_REVISION,
            "vokra.audioseal.source_revision": SOURCE_REVISION,
            **{
                f"vokra.audioseal.{prefix}.sha256": digest
                for prefix, digest in input_digests.items()
            },
        }
    }
    for spec in sorted(specs, key=lambda item: item.output_name):
        header[spec.output_name] = {
            "dtype": "F32",
            "shape": list(spec.shape),
            "data_offsets": [spec.begin, spec.end],
        }
    encoded = json.dumps(header, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return encoded + b" " * ((8 - len(encoded) % 8) % 8)


def tensor_bytes(value: Any, spec: TensorSpec) -> memoryview:
    import torch

    if not torch.is_tensor(value):
        raise ValueError(f"{spec.output_name}: no longer a tensor on second pass")
    tensor = value.detach().cpu().contiguous()
    shape = tuple(int(dim) for dim in tensor.shape)
    if tensor.dtype != torch.float32 or shape != spec.shape:
        raise ValueError(
            f"{spec.output_name}: second-pass contract changed to {tensor.dtype} {shape}; "
            f"expected torch.float32 {spec.shape}"
        )
    view = memoryview(tensor.numpy()).cast("B")
    if view.nbytes != spec.nbytes:
        raise ValueError(
            f"{spec.output_name}: payload is {view.nbytes} bytes, expected {spec.nbytes}"
        )
    return view


def prepare(paths: Mapping[str, Path], output: Path, force: bool) -> None:
    if output.exists() and not force:
        raise ValueError(f"output already exists: {output} (pass --force to replace it)")
    if output.resolve() in {path.resolve() for path in paths.values()}:
        raise ValueError("output must not overwrite one of the four source checkpoints")

    digests: dict[str, str] = {}
    for prefix, path in paths.items():
        require_source(path, prefix)
        digests[prefix] = sha256_file(path)

    specs: list[TensorSpec] = []
    offset = 0
    for prefix in sorted(paths):
        items, offset = inspect_one(paths[prefix], prefix, offset)
        specs.extend(items)
    if len(specs) != 310:
        raise ValueError(f"combined manifest has {len(specs)} tensors, expected 310")

    header = padded_header(specs, digests)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(struct.pack("<Q", len(header)))
            handle.write(header)
            by_prefix: dict[str, list[TensorSpec]] = {}
            for spec in specs:
                by_prefix.setdefault(spec.source_prefix, []).append(spec)
            for prefix in sorted(paths):
                state = load_state(paths[prefix])
                for spec in sorted(by_prefix[prefix], key=lambda item: item.begin):
                    handle.write(tensor_bytes(state[spec.source_name], spec))
                del state
                gc.collect()
            handle.flush()
            os.fsync(handle.fileno())
        expected_size = 8 + len(header) + offset
        if temporary.stat().st_size != expected_size:
            raise ValueError(
                f"temporary output is {temporary.stat().st_size} bytes, expected {expected_size}"
            )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    print(
        json.dumps(
            {
                "output": str(output),
                "tensor_count": len(specs),
                "tensor_bytes": offset,
                "checkpoint_revision": CHECKPOINT_REVISION,
                "source_revision": SOURCE_REVISION,
                "sha256": sha256_file(output),
            },
            sort_keys=True,
        )
    )


def self_test() -> None:
    assert sum(item[2] for item in INPUTS.values()) == 310
    assert len({item[1] for item in INPUTS.values()}) == 4
    assert all(len(item[1]) == 64 for item in INPUTS.values())
    sample = [TensorSpec("a", "p", "a", (2, 3), 24, 0, 24)]
    header = padded_header(sample, {"p": "0" * 64})
    assert len(header) % 8 == 0
    parsed = json.loads(header.decode("utf-8"))
    assert parsed["a"]["shape"] == [2, 3]
    assert parsed["a"]["data_offsets"] == [0, 24]
    print("audioseal_prepare_checkpoint self-test: ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator-base", type=Path)
    parser.add_argument("--detector-base", type=Path)
    parser.add_argument("--generator-streaming", type=Path)
    parser.add_argument("--detector-streaming", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        missing = [
            option
            for option in (
                "generator_base",
                "detector_base",
                "generator_streaming",
                "detector_streaming",
                "output",
            )
            if getattr(args, option) is None
        ]
        if missing:
            parser.error("missing required arguments: " + ", ".join(missing))
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    prepare(
        {
            "generator_base": args.generator_base,
            "detector_base": args.detector_base,
            "generator_streaming": args.generator_streaming,
            "detector_streaming": args.detector_streaming,
        },
        args.output,
        args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
