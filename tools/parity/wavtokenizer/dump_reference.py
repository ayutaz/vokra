#!/usr/bin/env python3
"""Dump an independent official WavTokenizer token-to-PCM reference.

The oracle imports ``decoder.pretrained.WavTokenizer`` from the official
``jishengpeng/WavTokenizer`` tree pinned below. It does not reproduce the
forward in Python or import Vokra. The audited Vokra GGUF is a verbatim F32
state-dict conversion, so its tensors are loaded into the official modules and
the upstream ``codes_to_features`` + ``decode`` methods execute the reference.

Run from ``tools/parity`` with the repository-managed Python environment:

    uv run --frozen python wavtokenizer/dump_reference.py \
      --source /path/to/jishengpeng-WavTokenizer \
      --gguf /path/to/wavtokenizer-large.gguf \
      --codes /path/to/codes.u32le \
      --output /path/to/reference
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import typing
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader


SOURCE_COMMIT = "5cf440d91ac420ca338f117b7003a77450d64730"
GGUF_SHA256 = "99b7dce0426266f7f2f6615091d832cea71387ce57edfae66666143a5c33a36b"
CONFIG_RELATIVE = Path(
    "configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(source: Path) -> None:
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != SOURCE_COMMIT:
        raise RuntimeError(
            f"official source commit {commit!r} != pinned {SOURCE_COMMIT!r}"
        )
    if not (source / CONFIG_RELATIVE).is_file():
        raise RuntimeError(f"missing pinned official config {CONFIG_RELATIVE}")


def gguf_tensor(item, expected_shape: torch.Size) -> torch.Tensor:
    if int(item.tensor_type) != 0:
        raise TypeError(f"{item.name}: expected GGML F32, got {item.tensor_type}")
    flat = item.data.copy().reshape(-1).astype(np.float32, copy=False)
    expected_elements = int(np.prod(expected_shape, dtype=np.int64))
    if flat.size != expected_elements:
        raise RuntimeError(
            f"{item.name}: {flat.size} values != expected {expected_elements}"
        )
    return torch.from_numpy(flat.reshape(tuple(expected_shape)))


def load_official_model(source: Path, gguf_path: Path):
    sys.path.insert(0, str(source))
    # The pinned official models.py uses ``tp.List`` in VocosBackbone.__init__
    # but imports only ``Optional``. Supplying the missing typing alias fixes
    # that packaging typo without changing any model arithmetic.
    import decoder.models as official_models

    official_models.tp = typing
    from decoder.pretrained import WavTokenizer

    model = WavTokenizer.from_hparams0802(str(source / CONFIG_RELATIVE))
    reader = GGUFReader(str(gguf_path))
    by_name = {item.name: item for item in reader.tensors}
    expected = model.state_dict()
    loaded = {}
    for name, target in expected.items():
        item = by_name.get(name)
        if item is None:
            raise RuntimeError(f"GGUF is missing official inference tensor {name!r}")
        loaded[name] = gguf_tensor(item, target.shape)
    incompatible = model.load_state_dict(loaded, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"official state-dict mismatch: {incompatible}")
    model.eval()
    return model, len(expected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--codes", type=Path, required=True)
    parser.add_argument("--condition-id", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    verify_source(args.source)
    if sha256_file(args.gguf) != GGUF_SHA256:
        raise RuntimeError("GGUF SHA-256 does not match the audited public artifact")
    if not 0 <= args.condition_id < 4:
        raise RuntimeError("condition-id must be in 0..3")
    codes = np.fromfile(args.codes, dtype="<u4")
    if codes.size == 0 or np.any(codes >= 4096):
        raise RuntimeError("codes must be non-empty and each value must be below 4096")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model, state_tensor_count = load_official_model(args.source, args.gguf)

    token_tensor = torch.from_numpy(codes.astype(np.int64)).reshape(1, 1, -1)
    bandwidth_id = torch.tensor([args.condition_id], dtype=torch.long)
    with torch.inference_mode():
        features = model.codes_to_features(token_tensor)
        decoded = model.decode(features, bandwidth_id=bandwidth_id)
    if tuple(features.shape) != (1, 512, int(codes.size)):
        raise RuntimeError(f"unexpected feature shape {tuple(features.shape)}")
    if tuple(decoded.shape) != (1, int(codes.size) * 320):
        raise RuntimeError(f"unexpected decoded shape {tuple(decoded.shape)}")
    if not bool(torch.isfinite(decoded).all()):
        raise RuntimeError("official decoder emitted non-finite PCM")

    args.output.mkdir(parents=True, exist_ok=True)
    features_path = args.output / "features.f32"
    pcm_path = args.output / "decoded_pcm.f32"
    codes_path = args.output / "codes.u32le"
    np.asarray(features.cpu().numpy(), dtype="<f4").tofile(features_path)
    np.asarray(decoded.cpu().numpy(), dtype="<f4").tofile(pcm_path)
    np.asarray(codes, dtype="<u4").tofile(codes_path)
    manifest = {
        "format": "vokra-wavtokenizer-reference-v1",
        "oracle": "official WavTokenizer.codes_to_features + WavTokenizer.decode",
        "source_repository": "https://github.com/jishengpeng/WavTokenizer",
        "source_commit": SOURCE_COMMIT,
        "config": str(CONFIG_RELATIVE),
        "gguf_sha256": GGUF_SHA256,
        "state_tensor_count": state_tensor_count,
        "condition_id": args.condition_id,
        "code_count": int(codes.size),
        "feature_shape": list(features.shape),
        "decoded_shape": list(decoded.shape),
        "torch": str(torch.__version__),
        "files": {
            path.name: sha256_file(path)
            for path in [codes_path, features_path, pcm_path]
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
