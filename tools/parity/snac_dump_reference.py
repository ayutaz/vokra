#!/usr/bin/env python3
"""Dump independent official-SNAC encoder, RVQ, and decoder fixtures.

The oracle is the pinned upstream ``hubertsiuzdak/snac`` package and the
official Hugging Face checkpoint revision for each released model.  Vokra is
not imported.  Decoder NoiseBlock tensors are captured from PyTorch itself so
Rust and Metal can exercise the learned graph with identical stochastic input
without pretending that two unrelated RNG algorithms share a seed contract.

Run through the repository Python 3.12 environment::

    uv run --project tools/parity python tools/parity/snac_dump_reference.py \
      --variant 24khz \
      --output crates/vokra-models/tests/fixtures/snac_24khz
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from snac import SNAC


SOURCE_REVISION = "8f79a718f1ad71f94f79999f0071348227aff22e"
VARIANTS = {
    "24khz": {
        "repo": "hubertsiuzdak/snac_24khz",
        "revision": "d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec",
        "sample_rate": 24_000,
        "num_samples": 1_567,
        "n_stages": 3,
    },
    "44khz": {
        "repo": "hubertsiuzdak/snac_44khz",
        "revision": "873ebef9718b89660340c6f55a2b515e98cfa1d9",
        "sample_rate": 44_100,
        "num_samples": 5_003,
        "n_stages": 4,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_f32(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().to(torch.float32).contiguous().numpy()
    path.write_bytes(np.asarray(array, dtype="<f4").tobytes(order="C"))


def write_u32(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().to(torch.int64).contiguous().numpy()
    if np.any(array < 0) or np.any(array > np.iinfo(np.uint32).max):
        raise RuntimeError(f"code outside u32 range in {path.name}")
    path.write_bytes(np.asarray(array, dtype="<u4").tobytes(order="C"))


def deterministic_pcm(num_samples: int, sample_rate: int) -> torch.Tensor:
    time = torch.arange(num_samples, dtype=torch.float32) / sample_rate
    pcm = (
        0.17 * torch.sin(2.0 * math.pi * 173.0 * time)
        + 0.09 * torch.sin(2.0 * math.pi * 997.0 * time + 0.37)
        + 0.035 * torch.cos(2.0 * math.pi * 2_137.0 * time)
        + 0.012 * torch.sin(2.0 * math.pi * 31.0 * time * time)
    )
    return pcm.clamp(-1.0, 1.0).contiguous()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spec = VARIANTS[args.variant]
    repo = str(spec["repo"])
    revision = str(spec["revision"])
    sample_rate = int(spec["sample_rate"])
    num_samples = int(spec["num_samples"])
    n_stages = int(spec["n_stages"])

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config_path = Path(
        hf_hub_download(repo_id=repo, filename="config.json", revision=revision)
    )
    checkpoint_path = Path(
        hf_hub_download(
            repo_id=repo, filename="pytorch_model.bin", revision=revision
        )
    )
    model = SNAC.from_pretrained(repo, revision=revision)
    model.eval()

    pcm = deterministic_pcm(num_samples, sample_rate)
    with torch.inference_mode():
        batch = pcm.reshape(1, 1, -1)
        padded = model.preprocess(batch)
        encoded = model.encoder(padded)
        codes = model.encode(batch)
        decoded_features = model.quantizer.from_codes(codes)

        captured_noise: list[torch.Tensor] = []
        original_randn = torch.randn

        def capture_randn(*size, **kwargs):
            value = original_randn(*size, **kwargs)
            captured_noise.append(value.detach().cpu().clone())
            return value

        torch.manual_seed(0x534E4143)
        torch.randn = capture_randn
        try:
            decoded_pcm = model.decode(codes)
        finally:
            torch.randn = original_randn

    if len(codes) != n_stages or len(captured_noise) != 4:
        raise RuntimeError(
            f"unexpected stages: codes={len(codes)}, noise={len(captured_noise)}"
        )
    if encoded.shape != decoded_features.shape:
        raise RuntimeError(
            f"encoder/decode feature shape mismatch: {encoded.shape} vs "
            f"{decoded_features.shape}"
        )
    if decoded_pcm.shape[-1] != padded.shape[-1]:
        raise RuntimeError(
            f"decoder extent {decoded_pcm.shape[-1]} != padded input {padded.shape[-1]}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def emit_f32(name: str, tensor: torch.Tensor) -> None:
        path = args.output / name
        write_f32(path, tensor)
        written.append(path)

    emit_f32("pcm.f32", pcm)
    emit_f32("encoded_features.f32", encoded)
    # Vokra's standalone RVQ helper deliberately returns time-major rows.
    emit_f32("decoded_features_time_major.f32", decoded_features.transpose(1, 2))
    emit_f32("decoded_pcm.f32", decoded_pcm)
    for stage, code in enumerate(codes):
        path = args.output / f"codes_{stage}.u32"
        write_u32(path, code)
        written.append(path)
    for stage, noise in enumerate(captured_noise):
        emit_f32(f"noise_{stage}.f32", noise)

    manifest = {
        "format": "vokra-snac-reference-v1",
        "oracle": "snac.SNAC.from_pretrained/encode/decode",
        "snac_version": importlib.metadata.version("snac"),
        "source_revision": SOURCE_REVISION,
        "upstream_hf": repo,
        "upstream_revision": revision,
        "config_sha256": sha256(config_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "torch": torch.__version__,
        "sample_rate": sample_rate,
        "num_samples": num_samples,
        "padded_samples": int(padded.shape[-1]),
        "feature_shape": list(encoded.shape),
        "code_shapes": [list(code.shape) for code in codes],
        "noise_shapes": [list(noise.shape) for noise in captured_noise],
        "files": {path.name: sha256(path) for path in sorted(written)},
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
