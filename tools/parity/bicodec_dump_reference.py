#!/usr/bin/env -S uv run --script
"""Dump the official SparkAudio BiCodec decode reference (VAST-only).

This tool imports the pinned Spark-TTS source implementation and calls its
``BiCodec.detokenize`` subgraphs.  It never imports Vokra or mirrors the
forward equations.  The input token vectors are fixed literals (no random
source), and every materialized output is accompanied by a SHA-256 manifest.
This is an evidence producer, not a numerical pass/fail gate; tolerance
selection remains a separate review decision.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


UPSTREAM_HF_REVISION = "642071559bfc6346c2359d19dcb6be3f9dd8a05d"
CHECKPOINT_BYTES = 625_518_756
CHECKPOINT_SHA256 = "e9940cd48d4446e4340ced82d234bf5618350dd9f5db900ebe47a4fdb03867ec"
CONFIG_BYTES = 1_164
CONFIG_SHA256 = "744f4093ae2381a2eb44ea8c4a5268a8d1e581498e9bf0808c034d1b076429be"
SOURCE_REVISION = "2f1ea9082400547242641f5271b6f941c9f439d1"
SOURCE_REPOSITORY = "https://github.com/SparkAudio/Spark-TTS"
SAMPLE_RATE = 16_000
FRAME_HOP = 320
SEMANTIC_VOCAB = 8_192
SEMANTIC_CODEBOOK_DIM = 8
SEMANTIC_LATENT_DIM = 1_024
GLOBAL_VOCAB = 4_096
GLOBAL_TOKENS = 32


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is not a regular non-symlink file: {path}")


def require_identity(path: Path, expected_bytes: int, expected_sha256: str, label: str) -> None:
    require_regular_file(path, label)
    actual_bytes = path.stat().st_size
    actual_sha256 = sha256_file(path)
    if actual_bytes != expected_bytes or actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{label} identity mismatch: bytes={actual_bytes} sha256={actual_sha256}"
        )


def source_git(source: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def write_f32(path: Path, tensor: torch.Tensor) -> dict[str, object]:
    values = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    data = np.asarray(values, dtype="<f4").tobytes(order="C")
    path.write_bytes(data)
    return {
        "path": path.name,
        "shape": [int(value) for value in values.shape],
        "dtype": "F32",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.source_dir.is_symlink() or args.model_dir.is_symlink() or args.output.is_symlink():
        raise RuntimeError("source, model, and output paths must not be symlinks")
    source = args.source_dir.resolve()
    model_dir = args.model_dir.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise RuntimeError(f"source directory is not a regular directory: {source}")
    if source_git(source, "rev-parse", "HEAD") != SOURCE_REVISION:
        raise RuntimeError("official source revision mismatch")
    if source_git(source, "remote", "get-url", "origin") != SOURCE_REPOSITORY:
        raise RuntimeError("official source remote mismatch")

    checkpoint = model_dir / "model.safetensors"
    config = model_dir / "config.yaml"
    require_identity(checkpoint, CHECKPOINT_BYTES, CHECKPOINT_SHA256, "BiCodec checkpoint")
    require_identity(config, CONFIG_BYTES, CONFIG_SHA256, "BiCodec config")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(source))
    module = importlib.import_module("sparktts.models.bicodec")
    model = module.BiCodec.load_from_checkpoint(model_dir)
    model.eval()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    # Explicit literals exercise the lower/upper semantic and global ranges;
    # no random token or input generator is used.
    semantic_values = [0, 1, 4_096, 8_191]
    global_values = [
        0,
        1,
        4_095,
        16,
        255,
        1_024,
        2_048,
        3_072,
    ] * 4
    if len(global_values) != GLOBAL_TOKENS:
        raise AssertionError("fixed global token fixture must contain exactly 32 values")
    semantic_tokens = torch.tensor([semantic_values], dtype=torch.long)
    global_tokens = torch.tensor([global_values], dtype=torch.long).reshape(1, 1, GLOBAL_TOKENS)

    with torch.no_grad():
        semantic_latent = model.quantizer.detokenize(semantic_tokens)
        d_vector = model.speaker_encoder.detokenize(global_tokens)
        prenet_output = model.prenet(semantic_latent, d_vector)
        conditioned = prenet_output + d_vector.unsqueeze(-1)
        waveform = model.decoder(conditioned)

    expected_samples = len(semantic_values) * FRAME_HOP
    # The codebook emits SEMANTIC_CODEBOOK_DIM channels, then the official
    # quantizer out_project expands them to the model input width.
    if tuple(semantic_latent.shape) != (1, SEMANTIC_LATENT_DIM, len(semantic_values)):
        raise RuntimeError(f"unexpected semantic latent shape: {tuple(semantic_latent.shape)}")
    if tuple(d_vector.shape) != (1, 1_024):
        raise RuntimeError(f"unexpected d-vector shape: {tuple(d_vector.shape)}")
    if tuple(prenet_output.shape) != (1, 1_024, len(semantic_values)):
        raise RuntimeError(f"unexpected prenet shape: {tuple(prenet_output.shape)}")
    if tuple(waveform.shape) != (1, 1, expected_samples):
        raise RuntimeError(f"unexpected waveform shape: {tuple(waveform.shape)}")
    if not all(torch.isfinite(tensor).all().item() for tensor in (semantic_latent, d_vector, prenet_output, waveform)):
        raise RuntimeError("official reference produced a non-finite tensor")

    records = {
        "semantic_latent": write_f32(output / "semantic_latent.f32", semantic_latent),
        "d_vector": write_f32(output / "d_vector.f32", d_vector),
        "prenet_output": write_f32(output / "prenet_output.f32", prenet_output),
        "waveform": write_f32(output / "waveform.f32", waveform),
    }
    manifest = {
        "schema": "vokra-bicodec-official-reference-v1",
        "provenance": {
            "oracle": "SparkAudio/Spark-TTS BiCodec official source",
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
            "upstream_hf_revision": UPSTREAM_HF_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "config_sha256": CONFIG_SHA256,
            "randomness": "none (fixed literal token vectors)",
            "upload": "none",
        },
        "contract": {
            "sample_rate": SAMPLE_RATE,
            "frame_hop": FRAME_HOP,
            "semantic_vocab": SEMANTIC_VOCAB,
            "semantic_codebook_dim": SEMANTIC_CODEBOOK_DIM,
            "semantic_latent_dim": SEMANTIC_LATENT_DIM,
            "global_vocab": GLOBAL_VOCAB,
            "global_tokens": GLOBAL_TOKENS,
        },
        "tokens": {"semantic": semantic_values, "global": global_values},
        "token_contract": {
            "semantic_csv": ",".join(str(value) for value in semantic_values),
            "global_csv": ",".join(str(value) for value in global_values),
        },
        "tensors": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
