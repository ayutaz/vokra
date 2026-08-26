#!/usr/bin/env python3
"""Dump an independent official FunCodec token-to-waveform reference.

The oracle imports Alibaba DAMO's official ``CostumeQuantizer`` and
``SEANetDecoder`` from the exact source commit, restores their state directly
from the immutable upstream checkpoint, and calls their official decode and
forward methods. It never imports Vokra or reproduces the neural forward.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import yaml


SOURCE_COMMIT = "b467b73e4025a123a68e64de9ba445d6a57d1984"
CHECKPOINT_SHA256 = (
    "08dd881b74daa150c405418b613496e872bbad4edd2d3c1d6d94ecf7199ac42c"
)
CONFIG_SHA256 = "5830ffe0c8cad9e8678dca1e5c6873a89629c23007155068f485ca44b2af9c4e"
SOURCE_HASHES = {
    "funcodec/models/decoder/seanet_decoder.py": (
        "f6de4f855d596a64ce056a862bf704620d4a32863c55b14d27bd2bdbce63a1bf"
    ),
    "funcodec/models/codec_basic.py": (
        "c7e965a6af86eeb612128d7707aef453a693f0c6935c593c8aae40ae0e0708fc"
    ),
    "funcodec/models/quantizer/costume_quantizer.py": (
        "3e46f64b2e06c6fcfe4553b8a27f5b4e7dc9b83bb2f9d975661d6fa0d21582a6"
    ),
    "funcodec/modules/quantization/vq.py": (
        "3b7980f588a6144d268efe61928b409544834232295513d829fc89bceb96fa98"
    ),
    "funcodec/modules/quantization/ddp_core_vq.py": (
        "f01529de004b7b18e85df96c89232b33b6bb284119e0711621ffe12a3de8c0a8"
    ),
    "funcodec/modules/normed_modules/conv.py": (
        "58850e49038cbc30ee7afadc64eaef57ac930022663153d789ae5e50d1e9bcd8"
    ),
    "funcodec/modules/normed_modules/lstm.py": (
        "53d0511f2eb17f2a3e7dbf0f3b969e05a9392aa0f986a3ea918840da58aaffc4"
    ),
}
SAMPLE_RATE = 16_000
FRAME_HOP = 320
DIMENSION = 128
NUM_CODEBOOKS = 32
CODEBOOK_SIZE = 1_024


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
    expected_scalars = {
        "sampling_rate": SAMPLE_RATE,
        "decoder": "encodec_seanet_decoder",
        "quantizer": "costume_quantizer",
        "model": "encodec",
    }
    for key, expected in expected_scalars.items():
        if config.get(key) != expected:
            raise RuntimeError(f"config {key}={config.get(key)!r} != {expected!r}")
    quantizer = config.get("quantizer_conf")
    decoder = config.get("decoder_conf")
    model = config.get("model_conf")
    if not all(isinstance(value, Mapping) for value in (quantizer, decoder, model)):
        raise RuntimeError("config quantizer/decoder/model sections must be mappings")
    expected_quantizer = {
        "codebook_size": CODEBOOK_SIZE,
        "num_quantizers": NUM_CODEBOOKS,
        "sampling_rate": SAMPLE_RATE,
        "encoder_hop_length": FRAME_HOP,
        "use_ddp": True,
    }
    for key, expected in expected_quantizer.items():
        if quantizer.get(key) != expected:
            raise RuntimeError(
                f"config quantizer_conf.{key}={quantizer.get(key)!r} != {expected!r}"
            )
    if decoder.get("norm") != "time_group_norm" or decoder.get("causal") is not False:
        raise RuntimeError("config decoder must be non-causal time_group_norm")
    expected_model = {
        "odim": DIMENSION,
        "target_sample_hz": SAMPLE_RATE,
        "audio_normalize": True,
        "segment_dur": None,
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            raise RuntimeError(
                f"config model_conf.{key}={model.get(key)!r} != {expected!r}"
            )


def load_official_modules(source: Path, checkpoint: Path, config: Mapping):
    sys.path.insert(0, str(source))
    from funcodec.models.decoder.seanet_decoder import SEANetDecoder
    from funcodec.models.quantizer.costume_quantizer import CostumeQuantizer

    quantizer_conf = dict(config["quantizer_conf"])
    decoder_conf = dict(config["decoder_conf"])
    quantizer = CostumeQuantizer(input_size=DIMENSION, **quantizer_conf)
    decoder = SEANetDecoder(input_size=DIMENSION, **decoder_conf)

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise RuntimeError("official checkpoint is not a flat string-to-tensor state dict")
    quantizer_state = {
        name.removeprefix("quantizer."): value
        for name, value in state.items()
        if name.startswith("quantizer.")
    }
    decoder_state = {
        name.removeprefix("decoder."): value
        for name, value in state.items()
        if name.startswith("decoder.")
    }
    if not quantizer_state or not decoder_state:
        raise RuntimeError("official checkpoint lacks quantizer.* or decoder.* state")
    quantizer.load_state_dict(quantizer_state, strict=True)
    decoder.load_state_dict(decoder_state, strict=True)
    quantizer.eval()
    decoder.eval()
    return quantizer, decoder, len(quantizer_state), len(decoder_state)


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
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise RuntimeError("official config must be a mapping")
    require_exact_config(config)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    quantizer, decoder, quantizer_tensors, decoder_tensors = load_official_modules(
        args.source, args.checkpoint, config
    )
    codes = deterministic_codes(args.frames, args.num_quantizers)
    code_tensor = torch.from_numpy(codes.astype(np.int64)).transpose(0, 1).unsqueeze(1)
    with torch.inference_mode():
        latent = quantizer.decode(code_tensor)
        decoded = decoder(latent.transpose(1, 2))
    if tuple(latent.shape) != (1, DIMENSION, args.frames):
        raise RuntimeError(f"unexpected official latent shape {tuple(latent.shape)}")
    expected_pcm_shape = (1, 1, args.frames * FRAME_HOP)
    if tuple(decoded.shape) != expected_pcm_shape:
        raise RuntimeError(f"unexpected official PCM shape {tuple(decoded.shape)}")
    if not bool(torch.isfinite(latent).all()) or not bool(torch.isfinite(decoded).all()):
        raise RuntimeError("official FunCodec emitted non-finite output")

    args.output.mkdir(parents=True, exist_ok=True)
    codes_path = args.output / "codes.u32le"
    latent_path = args.output / "latent.f32"
    pcm_path = args.output / "decoded_pcm.f32"
    np.asarray(codes, dtype="<u4").tofile(codes_path)
    np.asarray(latent.cpu().numpy(), dtype="<f4").tofile(latent_path)
    np.asarray(decoded.cpu().numpy(), dtype="<f4").tofile(pcm_path)
    manifest = {
        "format": "vokra-funcodec-reference-v1",
        "oracle": "official CostumeQuantizer.decode + SEANetDecoder.forward",
        "source_repository": "https://github.com/modelscope/FunCodec",
        "source_commit": SOURCE_COMMIT,
        "source_hashes": source_hashes,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "torch": str(torch.__version__),
        "frames": args.frames,
        "num_quantizers": args.num_quantizers,
        "sample_rate": SAMPLE_RATE,
        "frame_hop": FRAME_HOP,
        "latent_shape": list(latent.shape),
        "decoded_shape": list(decoded.shape),
        "official_quantizer_state_tensors": quantizer_tensors,
        "official_decoder_state_tensors": decoder_tensors,
        "files": {
            path.name: sha256_file(path)
            for path in (codes_path, latent_path, pcm_path)
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
