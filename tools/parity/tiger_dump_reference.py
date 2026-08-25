#!/usr/bin/env python3
"""Dump an independent official TIGER waveform reference.

The oracle imports ``TIGER`` / ``TIGERDNR`` from an exact clean checkout of
JusperLee/TIGER, validates the pinned Hugging Face config and safetensors
digests, strictly loads the official module, and calls its public ``forward``.
It never imports Vokra and does not mirror the TIGER equations.

Run model execution only on VAST through ``tools/parity/pyproject.toml``. The
``--self-test`` path is stdlib-only and merely checks deterministic PCM setup.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any


SOURCE_REPOSITORY = "https://github.com/JusperLee/TIGER"
SOURCE_REVISION = "9f18d4a10a7137e1ce8052cfb62215179f1287b6"
SOURCE_LICENSE_SHA256 = (
    "edc64d62aa021be7612337d2ced140375f52e4fd064b2f9cf6e656913d01bfa6"
)
DEFAULT_SAMPLES = 4_096


@dataclasses.dataclass(frozen=True)
class Variant:
    tag: str
    hf_repository: str
    hf_revision: str
    model_bytes: int
    model_sha256: str
    config_bytes: int
    config_sha256: str
    source_file: str
    source_file_bytes: int
    source_file_sha256: str
    tensor_count: int
    sample_rate: int
    streams: int
    config: dict[str, int]


VARIANTS = {
    "dnr": Variant(
        tag="dnr",
        hf_repository="JusperLee/TIGER-DnR",
        hf_revision="b7a59560bbca10febbcd46fb01600f868e587f57",
        model_bytes=17_130_568,
        model_sha256=(
            "dd1c696e72f6adea0085ef1af640882a8260519ad666422835e387a5b4abdd2a"
        ),
        config_bytes=250,
        config_sha256=(
            "ba9d2f833bf2f3a5855a35d0ccd11c786f6b92f1a482d84404bc4673edb29b54"
        ),
        source_file="look2hear/models/tiger_dnr.py",
        source_file_bytes=30_378,
        source_file_sha256=(
            "89605593bdfc05669e70f2b8647514077197f9870d32b5dd745913f6e03b50e0"
        ),
        tensor_count=2_304,
        sample_rate=44_100,
        streams=3,
        config={
            "att_hid_chan": 4,
            "att_kernel_size": 8,
            "att_n_head": 4,
            "att_stride": 1,
            "in_channels": 256,
            "num_blocks": 8,
            "num_sources": 3,
            "out_channels": 132,
            "sample_rate": 44_100,
            "stride": 512,
            "upsampling_depth": 5,
            "win": 2_048,
        },
    ),
    "speech": Variant(
        tag="speech",
        hf_repository="JusperLee/TIGER-speech",
        hf_revision="f0340340b2d9bbf72074edf8c076dcab59a10ba2",
        model_bytes=3_367_352,
        model_sha256=(
            "7e5fac7a9083c94b3a00c524f323188d4dd19ef09a54c29d1fec12ac114922db"
        ),
        config_bytes=249,
        config_sha256=(
            "1643c4e30cb97bc67024965aae13d631d44efdd304d8379cfd92143791017946"
        ),
        source_file="look2hear/models/tiger.py",
        source_file_bytes=23_531,
        source_file_sha256=(
            "a90ec403c5c024a1c6722a5143e0bd37bb642edec0e1506787ea212a65b287fe"
        ),
        tensor_count=838,
        sample_rate=16_000,
        streams=2,
        config={
            "att_hid_chan": 4,
            "att_kernel_size": 8,
            "att_n_head": 4,
            "att_stride": 1,
            "in_channels": 256,
            "num_blocks": 8,
            "num_sources": 2,
            "out_channels": 128,
            "sample_rate": 16_000,
            "stride": 160,
            "upsampling_depth": 5,
            "win": 640,
        },
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_file(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise ValueError(f"missing pinned input: {path}")
    size = path.stat().st_size
    if size != expected_bytes:
        raise ValueError(f"{path}: {size} bytes != pinned {expected_bytes}")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(f"{path}: SHA-256 {digest} != pinned {expected_sha256}")


def git_output(checkout: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def validate_source_checkout(checkout: Path, variant: Variant) -> Path:
    checkout = checkout.resolve()
    revision = git_output(checkout, "rev-parse", "HEAD")
    if revision != SOURCE_REVISION:
        raise ValueError(
            f"TIGER source revision {revision!r} != pinned {SOURCE_REVISION!r}"
        )
    dirty = git_output(checkout, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise ValueError("official TIGER source checkout must be exactly clean")
    validate_file(checkout / "LICENSE", 1_072, SOURCE_LICENSE_SHA256)
    source_path = checkout / variant.source_file
    validate_file(
        source_path, variant.source_file_bytes, variant.source_file_sha256
    )
    return checkout


def deterministic_pcm(sample_rate: int, samples: int) -> tuple[float, ...]:
    if samples < 1:
        raise ValueError("samples must be positive")
    values = []
    for index in range(samples):
        time = index / sample_rate
        value = (
            0.13 * math.sin(2.0 * math.pi * 173.0 * time + 0.1)
            + 0.07 * math.cos(2.0 * math.pi * 421.0 * time + 0.3)
            + 0.02 * math.sin(2.0 * math.pi * 37.0 * time + index * index * 1.0e-7)
        )
        values.append(value)
    return tuple(values)


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


def validate_config(path: Path, variant: Variant) -> None:
    validate_file(path, variant.config_bytes, variant.config_sha256)
    config = json.loads(path.read_text(encoding="utf-8"))
    if config != variant.config:
        raise ValueError(f"config content {config!r} != pinned {variant.config!r}")


def normalize_output(output: Any, variant: Variant, samples: int) -> Any:
    import torch

    if variant.tag == "dnr":
        if not isinstance(output, tuple) or len(output) != variant.streams:
            raise RuntimeError("official TIGERDNR returned an unexpected output tuple")
        streams = [value.reshape(-1) for value in output]
        normalized = torch.stack(streams, dim=0)
    else:
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("official TIGER-speech returned a non-tensor output")
        normalized = output.reshape(variant.streams, -1)
    expected_shape = (variant.streams, samples)
    if tuple(normalized.shape) != expected_shape:
        raise RuntimeError(
            f"official output shape {tuple(normalized.shape)} != {expected_shape}"
        )
    if not bool(torch.isfinite(normalized).all()):
        raise RuntimeError("official TIGER emitted non-finite samples")
    return normalized.detach().cpu().contiguous()


def dump(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from safetensors.torch import load_file

    variant = VARIANTS[args.variant]
    source = validate_source_checkout(args.source, variant)
    validate_file(args.weights, variant.model_bytes, variant.model_sha256)
    validate_config(args.config, variant)
    if args.output.exists():
        raise ValueError(f"output directory already exists: {args.output}")

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    try:
        if variant.tag == "dnr":
            from look2hear.models.tiger_dnr import TIGERDNR as OfficialTiger
        else:
            from look2hear.models.tiger import TIGER as OfficialTiger
        imported_module = sys.modules[OfficialTiger.__module__]
    finally:
        sys.path.pop(0)
    imported = Path(imported_module.__file__).resolve()
    if source not in imported.parents:
        raise ValueError(f"imported TIGER from {imported}, outside {source}")

    state = load_file(str(args.weights), device="cpu")
    if len(state) != variant.tensor_count:
        raise ValueError(
            f"checkpoint has {len(state)} tensors, expected {variant.tensor_count}"
        )
    non_f32 = sorted(name for name, value in state.items() if value.dtype != torch.float32)
    if non_f32:
        raise ValueError(f"checkpoint contains non-F32 tensors: {non_f32[:8]!r}")

    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model = OfficialTiger(**variant.config).cpu().eval()
    model.load_state_dict(state, strict=True)

    pcm = np.asarray(
        deterministic_pcm(variant.sample_rate, args.samples), dtype=np.float32
    )
    mixture = torch.from_numpy(pcm.copy()).reshape(1, 1, -1)
    with torch.inference_mode():
        separated = normalize_output(model(mixture), variant, args.samples)

    args.output.mkdir(parents=True)
    inputs = write_f32(args.output / "pcm.f32le", pcm)
    outputs = write_f32(args.output / "separated.f32le", separated.numpy())
    manifest = {
        "format": "vokra.tiger.official-parity.v1",
        "oracle": "official TIGER/TIGERDNR forward from pinned clean source",
        "variant": variant.tag,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_file": variant.source_file,
        "source_file_sha256": variant.source_file_sha256,
        "source_license": "MIT",
        "source_license_sha256": SOURCE_LICENSE_SHA256,
        "hf_repository": variant.hf_repository,
        "hf_revision": variant.hf_revision,
        "model_sha256": variant.model_sha256,
        "config_sha256": variant.config_sha256,
        "official_import": str(imported),
        "torch": str(torch.__version__),
        "sample_rate": variant.sample_rate,
        "samples": args.samples,
        "stream_order": (
            ["dialog", "effect", "music"]
            if variant.tag == "dnr"
            else ["speaker_1", "speaker_2"]
        ),
        "input": inputs,
        "output": outputs,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


def self_test() -> None:
    for variant in VARIANTS.values():
        pcm = deterministic_pcm(variant.sample_rate, DEFAULT_SAMPLES)
        if len(pcm) != DEFAULT_SAMPLES or not all(math.isfinite(value) for value in pcm):
            raise AssertionError(f"invalid deterministic PCM for {variant.tag}")
        payload = struct.pack(f"<{len(pcm)}f", *pcm)
        if len(payload) != DEFAULT_SAMPLES * 4:
            raise AssertionError(f"invalid PCM encoding for {variant.tag}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--variant", choices=sorted(VARIANTS))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    args = parser.parse_args(argv)
    if not args.self_test:
        missing = [
            name
            for name in ("variant", "source", "weights", "config", "output")
            if getattr(args, name) is None
        ]
        if missing:
            parser.error(f"model dump requires: {', '.join('--' + name for name in missing)}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.self_test:
        self_test()
    else:
        dump(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
