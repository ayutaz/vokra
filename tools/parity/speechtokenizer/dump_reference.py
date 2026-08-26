#!/usr/bin/env python3
"""Dump an independent official SpeechTokenizer decode reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import types
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch


SOURCE_COMMIT = "30c96fb32a9fc06a2258c98119e237def051e46c"
CHECKPOINT_SHA256 = (
    "d04593b6c9a4b475f91ca481141a6ef5b23e6ac112f347dd2b2717f193c1c728"
)
CONFIG_SHA256 = "ea343ad69ca7e70c8febf8fc4cda683b1c4b1c36709e5e577936ffb05d62e6eb"
SOURCE_HASHES = {
    "speechtokenizer/model.py": (
        "57af39db7ba83f43c67b3ab6b5881d13e81566cc06b9ab012088e1d1d50cc9dd"
    ),
    "speechtokenizer/modules/__init__.py": (
        "f9665a8fdefdf240cb6e52e015ea667a45e128db1eda68e57c6afd6ec25f4824"
    ),
    "speechtokenizer/modules/seanet.py": (
        "a11df17d632b05ea0311512dd2488c91aff58d4ced0b4ce9740372a204905e06"
    ),
    "speechtokenizer/modules/conv.py": (
        "341602819df01a80123701b5d31485e40182ad49d017aac65eabc675e930aab5"
    ),
    "speechtokenizer/modules/lstm.py": (
        "3390ea1378fef835b73ad5f48392b25ae7bab608aee1aa8099bbab96dfd83681"
    ),
    "speechtokenizer/modules/norm.py": (
        "23d5727c362bfcd43ee8ba2972531b6b63991a974c11a9156b179134ce2c87d3"
    ),
    "speechtokenizer/quantization/__init__.py": (
        "34c806bc1cafc8b835926b6f6450bee769f95eb467cf1c19b4427e9dd7e55bbc"
    ),
    "speechtokenizer/quantization/vq.py": (
        "def737933383759aba7ef279c6133a9625e8e8fbdf74a481122cbea3b85c99cb"
    ),
    "speechtokenizer/quantization/core_vq.py": (
        "7ab829c47657ac18d95305b13bfcfab836905537f387e5c85ab58225659e3116"
    ),
    "speechtokenizer/quantization/distrib.py": (
        "7d23e1e22d94d54988c3615dde43f32a87aa81d987f0ade16612a6736537640e"
    ),
}
SAMPLE_RATE = 16_000
FRAME_HOP = 320
DIMENSION = 1_024
NUM_CODEBOOKS = 8
CODEBOOK_SIZE = 1_024
EXPECTED_STATE_TENSORS = 166


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(source: Path) -> dict[str, str]:
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != SOURCE_COMMIT:
        raise RuntimeError(f"official source commit {commit!r} != {SOURCE_COMMIT!r}")
    actual = {}
    for relative, expected in SOURCE_HASHES.items():
        path = source / relative
        digest = sha256_file(path)
        if digest != expected:
            raise RuntimeError(
                f"official source {relative} SHA-256 {digest} != {expected}"
            )
        actual[relative] = digest
    return actual


def require_exact_config(config: Mapping) -> None:
    expected = {
        "n_filters": 64,
        "strides": [8, 5, 4, 2],
        "dimension": DIMENSION,
        "semantic_dimension": 768,
        "bidirectional": True,
        "dilation_base": 2,
        "residual_kernel_size": 3,
        "n_residual_layers": 1,
        "lstm_layers": 2,
        "activation": "ELU",
        "sampling_rate": SAMPLE_RATE,
        "sample_rate": SAMPLE_RATE,
        "codebook_size": CODEBOOK_SIZE,
        "n_q": NUM_CODEBOOKS,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"config {key}={config.get(key)!r} != {value!r}")


def load_official_model(source: Path, checkpoint: Path, config: Mapping):
    # Import the official inference package without executing its top-level
    # __init__, which also imports trainer-only audio dependencies irrelevant to
    # decode. Relative imports inside the untouched official model still run.
    package = types.ModuleType("speechtokenizer")
    package.__path__ = [str(source / "speechtokenizer")]
    package.__package__ = "speechtokenizer"
    sys.modules["speechtokenizer"] = package
    from speechtokenizer.model import SpeechTokenizer

    model = SpeechTokenizer(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise RuntimeError("official checkpoint is not a flat string-to-tensor state dict")
    if len(state) != EXPECTED_STATE_TENSORS:
        raise RuntimeError(
            f"official state tensor count {len(state)} != {EXPECTED_STATE_TENSORS}"
        )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, len(state)


def deterministic_codes(frames: int, num_quantizers: int) -> np.ndarray:
    return np.asarray(
        [
            (frame * 131 + quantizer * 37 + 17) % CODEBOOK_SIZE
            for frame in range(frames)
            for quantizer in range(num_quantizers)
        ],
        dtype="<u4",
    ).reshape(frames, num_quantizers)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--num-quantizers", type=int, default=NUM_CODEBOOKS)
    args = parser.parse_args()

    if args.frames <= 0:
        raise RuntimeError("frames must be positive")
    if not 1 <= args.num_quantizers <= NUM_CODEBOOKS:
        raise RuntimeError(f"num-quantizers must be in 1..{NUM_CODEBOOKS}")
    source_hashes = verify_source(args.source)
    if sha256_file(args.checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("official checkpoint SHA-256 mismatch")
    if sha256_file(args.config) != CONFIG_SHA256:
        raise RuntimeError("official config SHA-256 mismatch")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise RuntimeError("official config must be a mapping")
    require_exact_config(config)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model, state_tensors = load_official_model(args.source, args.checkpoint, config)
    codes = deterministic_codes(args.frames, args.num_quantizers)
    code_tensor = torch.from_numpy(codes.astype(np.int64)).transpose(0, 1).unsqueeze(1)
    with torch.inference_mode():
        decoded = model.decode(code_tensor)
    expected_shape = (1, 1, args.frames * FRAME_HOP)
    if tuple(decoded.shape) != expected_shape:
        raise RuntimeError(f"unexpected official PCM shape {tuple(decoded.shape)}")
    if not bool(torch.isfinite(decoded).all()):
        raise RuntimeError("official SpeechTokenizer emitted non-finite output")

    args.output.mkdir(parents=True, exist_ok=True)
    codes_path = args.output / "codes.u32le"
    pcm_path = args.output / "decoded_pcm.f32"
    np.asarray(codes, dtype="<u4").tofile(codes_path)
    np.asarray(decoded.cpu().numpy(), dtype="<f4").tofile(pcm_path)
    manifest = {
        "format": "vokra-speechtokenizer-reference-v1",
        "oracle": "official SpeechTokenizer.decode",
        "source_repository": "https://github.com/ZhangXInFD/SpeechTokenizer",
        "source_commit": SOURCE_COMMIT,
        "source_hashes": source_hashes,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "torch": str(torch.__version__),
        "frames": args.frames,
        "num_quantizers": args.num_quantizers,
        "sample_rate": SAMPLE_RATE,
        "frame_hop": FRAME_HOP,
        "decoded_shape": list(decoded.shape),
        "official_state_tensors": state_tensors,
        "files": {
            path.name: sha256_file(path) for path in (codes_path, pcm_path)
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
