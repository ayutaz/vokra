#!/usr/bin/env python3
"""Dump an independent official YuE xcodec-mini token-decode reference.

The oracle imports the fixed upstream ``ResidualVectorQuantizer`` directly
from ``m-a-p/xcodec_mini_infer`` and the released Vocos classes from the
``vocos==0.1.0`` wheel pinned by ``tools/parity/uv.lock``. It never calls
Vokra code and never recreates the Rust algorithm as a second local mirror.

Run only through the parity environment on VAST (the selected source
artifacts exceed the repository's 2 GB aggregate guard)::

    uv run --project tools/parity python \
      tools/parity/yue_xcodec_mini_dump_reference.py \
      --source-root /path/to/xcodec_mini_infer-at-fixed-revision \
      --codec-checkpoint final_ckpt/ckpt_00360000.pth \
      --decoder-checkpoint decoders/decoder_151000.pth \
      --output-dir /path/to/reference
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from pathlib import Path


UPSTREAM_HF = "m-a-p/xcodec_mini_infer"
UPSTREAM_REVISION = "fe781a67815ab47b4a3a5fce1e8d0a692da7e4e5"
CODEC_CHECKPOINT_FILE = "ckpt_00360000.pth"
CODEC_CHECKPOINT_BYTES = 1_360_444_883
CODEC_CHECKPOINT_SHA256 = (
    "c8c379ea2d3cbde1c8ba1b9717975220e79ba3f556bb161766fd5e4585dcd59c"
)
DECODER_CHECKPOINT_FILE = "decoder_151000.pth"
DECODER_CHECKPOINT_BYTES = 72_610_550
DECODER_CHECKPOINT_SHA256 = (
    "8af97a29d3483f9d4a3755992837501bd7d6caa1a69382ed16e64039e0ea0998"
)
SOURCE_PACKAGE = "vocos==0.1.0"
SOURCE_PACKAGE_WHEEL_SHA256 = (
    "0ac13eaef68596074301e912d781399b3defa4b4ca60b6bc52c8a4b9209ca235"
)

CODEBOOKS = 12
CODEBOOK_SIZE = 1024
FEATURE_DIM = 1024
TOKEN_SAMPLE_RATE = 16_000
TOKEN_HOP_LENGTH = 320
TOKEN_FRAME_RATE = 50
DIM = 512
INTERMEDIATE_DIM = 1536
NUM_LAYERS = 8
N_FFT = 3528
HOP_LENGTH = 882
OUTPUT_SAMPLE_RATE = 44_100
PADDING = "same"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, filename: str, size: int, sha256: str) -> None:
    if path.name != filename:
        raise ValueError(f"checkpoint filename {path.name!r}, expected {filename!r}")
    actual_size = path.stat().st_size
    if actual_size != size:
        raise ValueError(f"{path} has {actual_size} bytes, expected {size}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != sha256:
        raise ValueError(f"{path} SHA-256 {actual_sha256}, expected {sha256}")


def verify_source_root(path: Path) -> dict[str, str]:
    if not path.is_dir():
        raise ValueError(f"source root is not a directory: {path}")
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(
            f"source root revision {revision}, expected {UPSTREAM_REVISION}"
        )
    required = [
        Path("quantization/__init__.py"),
        Path("quantization/vq.py"),
        Path("quantization/core_vq_lsx_version.py"),
        Path("quantization/distrib.py"),
    ]
    hashes: dict[str, str] = {}
    for relative in required:
        source = path / relative
        if not source.is_file():
            raise ValueError(f"fixed upstream source is missing {relative}")
        hashes[relative.as_posix()] = sha256_file(source)
    return hashes


def load_pickle(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # This fallback is permitted only after exact size/SHA authentication.
        return torch.load(path, map_location="cpu", weights_only=False)


def unwrap_decoder_state(raw: object) -> dict:
    if not isinstance(raw, dict):
        raise TypeError("decoder checkpoint must contain a dict")
    for wrapper in ("state_dict", "model_state_dict", "model", "module"):
        inner = raw.get(wrapper)
        if isinstance(inner, dict) and inner:
            raw = inner
            break
    if not isinstance(raw, dict) or not raw:
        raise TypeError("decoder checkpoint yielded no state dict")
    return raw


def self_test() -> int:
    assert len(UPSTREAM_REVISION) == 40
    assert len(CODEC_CHECKPOINT_SHA256) == 64
    assert len(DECODER_CHECKPOINT_SHA256) == 64
    assert len(SOURCE_PACKAGE_WHEEL_SHA256) == 64
    assert 8 * 5 * 4 * 2 == TOKEN_HOP_LENGTH
    assert TOKEN_SAMPLE_RATE // TOKEN_HOP_LENGTH == TOKEN_FRAME_RATE
    assert CODEBOOKS == 12
    assert OUTPUT_SAMPLE_RATE // HOP_LENGTH == TOKEN_FRAME_RATE
    print("yue_xcodec_mini_dump_reference self-test: ok")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--codec-checkpoint", type=Path)
    parser.add_argument("--decoder-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if (
        args.source_root is None
        or args.codec_checkpoint is None
        or args.decoder_checkpoint is None
        or args.output_dir is None
    ):
        parser.error(
            "--source-root, --codec-checkpoint, --decoder-checkpoint and "
            "--output-dir are required"
        )
    if args.frames <= 0:
        parser.error("--frames must be positive")

    source_hashes = verify_source_root(args.source_root)
    verify_file(
        args.codec_checkpoint,
        CODEC_CHECKPOINT_FILE,
        CODEC_CHECKPOINT_BYTES,
        CODEC_CHECKPOINT_SHA256,
    )
    verify_file(
        args.decoder_checkpoint,
        DECODER_CHECKPOINT_FILE,
        DECODER_CHECKPOINT_BYTES,
        DECODER_CHECKPOINT_SHA256,
    )

    import torch
    from vocos.heads import ISTFTHead
    from vocos.models import VocosBackbone

    torch.set_num_threads(1)
    torch.manual_seed(0)
    sys.path.insert(0, str(args.source_root.resolve()))
    from quantization import ResidualVectorQuantizer  # type: ignore[import-not-found]

    raw_codec = load_pickle(args.codec_checkpoint)
    if not isinstance(raw_codec, dict):
        raise TypeError("codec checkpoint must contain a dict")
    codec_state = raw_codec.get("codec_model")
    if not isinstance(codec_state, dict) or not codec_state:
        raise TypeError("codec checkpoint is missing its non-empty `codec_model` mapping")
    quantizer_state = {
        key.removeprefix("quantizer."): value
        for key, value in codec_state.items()
        if key.startswith("quantizer.")
    }
    if not quantizer_state:
        raise ValueError("codec_model contains no quantizer tensors")
    quantizer = ResidualVectorQuantizer(
        dimension=FEATURE_DIM,
        n_q=CODEBOOKS,
        bins=CODEBOOK_SIZE,
    )
    quantizer.load_state_dict(quantizer_state, strict=True)
    quantizer.eval()
    del raw_codec, codec_state, quantizer_state
    gc.collect()

    raw_decoder = load_pickle(args.decoder_checkpoint)
    decoder_state = unwrap_decoder_state(raw_decoder)
    backbone = VocosBackbone(
        input_channels=FEATURE_DIM,
        dim=DIM,
        intermediate_dim=INTERMEDIATE_DIM,
        num_layers=NUM_LAYERS,
        adanorm_num_embeddings=None,
    )
    head = ISTFTHead(
        dim=DIM,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        padding=PADDING,
    )
    backbone_state = {
        key.removeprefix("backbone."): value
        for key, value in decoder_state.items()
        if key.startswith("backbone.")
    }
    head_state = {
        key.removeprefix("head."): value
        for key, value in decoder_state.items()
        if key.startswith("head.")
    }
    backbone.load_state_dict(backbone_state, strict=True)
    head.load_state_dict(head_state, strict=True)
    expected_decoder = {
        *(f"backbone.{key}" for key in backbone.state_dict()),
        *(f"head.{key}" for key in head.state_dict()),
    }
    if set(decoder_state) != expected_decoder or len(expected_decoder) != 81:
        missing = sorted(expected_decoder - set(decoder_state))[:5]
        extra = sorted(set(decoder_state) - expected_decoder)[:5]
        raise ValueError(
            f"official decoder manifest mismatch: missing={missing}, extra={extra}"
        )
    del raw_decoder, decoder_state, backbone_state, head_state
    gc.collect()

    quantizer.eval()
    backbone.eval()
    head.eval()
    q = torch.arange(CODEBOOKS, dtype=torch.long).reshape(CODEBOOKS, 1, 1)
    t = torch.arange(args.frames, dtype=torch.long).reshape(1, 1, args.frames)
    codes = (q * 97 + t * 131 + (q + t).square() * 17) % CODEBOOK_SIZE
    with torch.inference_mode():
        features = quantizer.decode(codes).contiguous()
        backbone_output = backbone(features)
        waveform = head(backbone_output)[0].contiguous()
    if list(features.shape) != [1, FEATURE_DIM, args.frames]:
        raise ValueError(f"official feature shape {list(features.shape)} is unexpected")
    expected_samples = args.frames * HOP_LENGTH
    if waveform.numel() != expected_samples:
        raise ValueError(
            f"official waveform has {waveform.numel()} samples, expected {expected_samples}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_major_codes = codes[:, 0, :].transpose(0, 1).contiguous()
    frame_major_codes.numpy().astype("<u4", copy=False).tofile(
        args.output_dir / "codes.u32le"
    )
    features[0].numpy().astype("<f4", copy=False).tofile(
        args.output_dir / "features.f32le"
    )
    backbone_output[0].contiguous().numpy().astype("<f4", copy=False).tofile(
        args.output_dir / "backbone.f32le"
    )
    waveform.numpy().astype("<f4", copy=False).tofile(
        args.output_dir / "waveform.f32le"
    )
    metadata = {
        "upstream_hf": UPSTREAM_HF,
        "upstream_revision": UPSTREAM_REVISION,
        "source_files_sha256": source_hashes,
        "codec_checkpoint_file": CODEC_CHECKPOINT_FILE,
        "codec_checkpoint_bytes": CODEC_CHECKPOINT_BYTES,
        "codec_checkpoint_sha256": CODEC_CHECKPOINT_SHA256,
        "decoder_checkpoint_file": DECODER_CHECKPOINT_FILE,
        "decoder_checkpoint_bytes": DECODER_CHECKPOINT_BYTES,
        "decoder_checkpoint_sha256": DECODER_CHECKPOINT_SHA256,
        "source_package": SOURCE_PACKAGE,
        "source_package_wheel_sha256": SOURCE_PACKAGE_WHEEL_SHA256,
        "frames": args.frames,
        "codebooks": CODEBOOKS,
        "codebook_size": CODEBOOK_SIZE,
        "feature_dim": FEATURE_DIM,
        "token_sample_rate": TOKEN_SAMPLE_RATE,
        "token_hop_length": TOKEN_HOP_LENGTH,
        "token_frame_rate": TOKEN_FRAME_RATE,
        "output_sample_rate": OUTPUT_SAMPLE_RATE,
        "output_hop_length": HOP_LENGTH,
        "samples": waveform.numel(),
        "torch": torch.__version__,
        "vocos_decoder_tensor_count": len(expected_decoder),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
