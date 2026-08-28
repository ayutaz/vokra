#!/usr/bin/env python3
"""Dump independent official-FocalCodec encode/decode parity fixtures.

The oracle is ``focalcodec.FocalCodec`` imported directly from the pinned
upstream Git commit.  The three supported checkpoints are pinned to immutable
Hugging Face revisions and audited safetensors SHA-256 digests.  Vokra is not
imported and none of its forward-pass code is reproduced here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path

import numpy as np
import torch
from focalcodec import FocalCodec
from huggingface_hub import hf_hub_download


SOURCE_COMMIT = "912b7f2c0cd43d54a8aed296bbcc925dec7d4ea3"
SOURCE_REPOSITORY = "github.com/lucadellalib/focalcodec"
NUM_SAMPLES = 3_200
VARIANTS = {
    "50hz": {
        "repo": "lucadellalib/focalcodec_50hz",
        "revision": "d6d6d9524e52155c85193c2c3b8da1cf8842f019",
        "checkpoint_sha256": "2700a916ca8d1c11a899995ef8d451ee53a481486650c4b78cd96feff9ac77f0",
        "downscale_factors": [1, 1, 1],
        "upscale_factors": [1, 1, 1],
    },
    "25hz": {
        "repo": "lucadellalib/focalcodec_25hz",
        "revision": "581aa401901b4985fbbdd3569d80a5a191740c1f",
        "checkpoint_sha256": "07a5ca4b92fb1fdc0499e761df6373d5b0811b8b950335fc40bcf6dc4f300441",
        "downscale_factors": [2, 1, 1],
        "upscale_factors": [1, 1, 2],
    },
    "12_5hz": {
        "repo": "lucadellalib/focalcodec_12_5hz",
        "revision": "96ddced5e284f109d9022a65d1062fcd92dc33eb",
        "checkpoint_sha256": "5362c33ed75801d9bced7e8573f8eece674592ea3c3156451a85f4b924b1c1e5",
        "downscale_factors": [2, 2, 1],
        "upscale_factors": [1, 2, 2],
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_reference_install() -> tuple[str, str]:
    distribution = importlib.metadata.distribution("focalcodec")
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError(
            "focalcodec has no PEP 610 direct_url.json; refusing an unpinned oracle"
        )
    direct_url = json.loads(direct_url_text)
    source_url = str(direct_url.get("url", ""))
    vcs_info = direct_url.get("vcs_info")
    commit = vcs_info.get("commit_id") if isinstance(vcs_info, dict) else None
    if commit != SOURCE_COMMIT:
        raise RuntimeError(
            f"imported focalcodec commit {commit!r} != pinned {SOURCE_COMMIT}"
        )
    if SOURCE_REPOSITORY not in source_url:
        raise RuntimeError(
            f"imported focalcodec source {source_url!r} is not the official repository"
        )
    return source_url, str(distribution.version)


def deterministic_pcm(num_samples: int, sample_rate: int) -> torch.Tensor:
    time = torch.arange(num_samples, dtype=torch.float32) / sample_rate
    pcm = (
        0.17 * torch.sin(2.0 * math.pi * 173.0 * time)
        + 0.09 * torch.sin(2.0 * math.pi * 997.0 * time + 0.37)
        + 0.035 * torch.cos(2.0 * math.pi * 2_137.0 * time)
        + 0.012 * torch.sin(2.0 * math.pi * 31.0 * time * time)
    )
    return pcm.clamp(-1.0, 1.0).contiguous()


def write_f32(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().to(torch.float32).contiguous().numpy()
    path.write_bytes(np.asarray(array, dtype="<f4").tobytes(order="C"))


def write_u32(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().to(torch.int64).contiguous().numpy()
    if np.any(array < 0) or np.any(array > np.iinfo(np.uint32).max):
        raise RuntimeError(f"token outside u32 range in {path.name}")
    path.write_bytes(np.asarray(array, dtype="<u4").tobytes(order="C"))


def execution_environment() -> dict[str, object]:
    get_cpu_capability = getattr(torch.backends.cpu, "get_cpu_capability", None)
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "torch": str(torch.__version__),
        "torch_cpu_capability": (
            str(get_cpu_capability())
            if callable(get_cpu_capability)
            else "unavailable"
        ),
        "torch_threads": torch.get_num_threads(),
    }


def require_topology(model: FocalCodec, spec: dict[str, object]) -> None:
    expected_classes = {
        "encoder": "WavLM",
        "compressor": "FocalEncoder",
        "quantizer": "BinarySphericalQuantizer",
        "decompressor": "FocalDecoder",
        "decoder": "Vocos",
    }
    for name, expected in expected_classes.items():
        actual = getattr(model, name).__class__.__name__
        if actual != expected:
            raise RuntimeError(f"{name} class {actual!r} != expected {expected!r}")
    if model.sample_rate != 16_000 or model.causal:
        raise RuntimeError(
            f"unexpected sample_rate/causal: {model.sample_rate}/{model.causal}"
        )
    if int(model.quantizer.codebook_size) != 8_192:
        raise RuntimeError(
            f"unexpected codebook size {model.quantizer.codebook_size}"
        )
    downscale = [int(value) for value in model.compressor.downscale_factors]
    upscale = [int(value) for value in model.decompressor.upscale_factors]
    if downscale != spec["downscale_factors"]:
        raise RuntimeError(
            f"checkpoint downscale factors {downscale} != {spec['downscale_factors']}"
        )
    if upscale != spec["upscale_factors"]:
        raise RuntimeError(
            f"checkpoint upscale factors {upscale} != {spec['upscale_factors']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_url, focalcodec_version = verify_reference_install()
    spec = VARIANTS[args.variant]
    repo = str(spec["repo"])
    revision = str(spec["revision"])

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0x464F4341)

    config_path = Path(
        hf_hub_download(repo_id=repo, filename="config.json", revision=revision)
    )
    checkpoint_path = Path(
        hf_hub_download(repo_id=repo, filename="model.safetensors", revision=revision)
    )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != spec["checkpoint_sha256"]:
        raise RuntimeError(
            f"{repo} checkpoint SHA-256 {checkpoint_sha256} != audited "
            f"{spec['checkpoint_sha256']}"
        )

    model = FocalCodec.from_pretrained(repo, revision=revision)
    model.eval()
    require_topology(model, spec)

    pcm = deterministic_pcm(NUM_SAMPLES, model.sample_rate)
    with torch.inference_mode():
        batch = pcm.reshape(1, -1)
        features = model.sig_to_feats(batch)
        latents = model.feats_to_lats(features)
        tokens = model.lats_to_toks(latents)
        codes = model.toks_to_codes(tokens)
        quantized_features = model.toks_to_qfeats(tokens)
        decoded_pcm = model.toks_to_sig(tokens)

    tensors = {
        "pcm.f32": pcm,
        "features.f32": features,
        "latents.f32": latents,
        "codes.f32": codes,
        "quantized_features.f32": quantized_features,
        "decoded_pcm.f32": decoded_pcm,
    }
    for name, tensor in tensors.items():
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"oracle emitted non-finite values in {name}")

    args.output.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, tensor in tensors.items():
        path = args.output / name
        write_f32(path, tensor)
        written[name] = path
    tokens_path = args.output / "tokens.u32"
    write_u32(tokens_path, tokens)
    written[tokens_path.name] = tokens_path

    manifest = {
        "format": "vokra-focalcodec-reference-v1",
        "oracle": "focalcodec.FocalCodec.from_pretrained public methods",
        "source_url": source_url,
        "source_commit": SOURCE_COMMIT,
        "focalcodec_version": focalcodec_version,
        "upstream_hf": repo,
        "upstream_revision": revision,
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": checkpoint_sha256,
        "variant": args.variant,
        "sample_rate": model.sample_rate,
        "num_samples": NUM_SAMPLES,
        "shapes": {
            "pcm": list(pcm.shape),
            "features": list(features.shape),
            "latents": list(latents.shape),
            "tokens": list(tokens.shape),
            "codes": list(codes.shape),
            "quantized_features": list(quantized_features.shape),
            "decoded_pcm": list(decoded_pcm.shape),
        },
        "environment": execution_environment(),
        "files": {
            name: sha256_file(path) for name, path in sorted(written.items())
        },
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
