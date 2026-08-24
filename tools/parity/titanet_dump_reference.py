#!/usr/bin/env python3
"""Dump an independent NVIDIA NeMo TitaNet-L reference fixture.

The oracle is NeMo's own ``EncDecSpeakerLabelModel.restore_from`` and its
preprocessor/encoder/decoder modules. This script deliberately does not import
or mirror Vokra's Rust implementation.

Run through the repository's pinned Python 3.12 environment:

    uv run --project tools/parity --extra titanet python \
      tools/parity/titanet_dump_reference.py \
      --nemo speakerverification_en_titanet_large.nemo \
      --output crates/vokra-models/tests/fixtures/titanet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import types
from pathlib import Path

import numpy as np
import torch


UPSTREAM_HF = "nvidia/speakerverification_en_titanet_large"
UPSTREAM_REVISION = "0dc382f40121a5fbd34db10a2bb04d826c2be6a8"
SOURCE_REVISION = "082c5ae26168796d3ebac6adcf54bb8b5354daa1"
EXPECTED_CHECKPOINT_SHA256 = (
    "e838520693f269e7984f55bc8eb3c2d60ccf246bf4b896d4be9bcabe3e4b0fe3"
)
SAMPLE_RATE = 16_000
NUM_SAMPLES = 8_173


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_pcm() -> torch.Tensor:
    time = torch.arange(NUM_SAMPLES, dtype=torch.float32) / SAMPLE_RATE
    generator = torch.Generator(device="cpu").manual_seed(0x54495441)
    noise = torch.randn(NUM_SAMPLES, generator=generator, dtype=torch.float32)
    pcm = (
        0.13 * torch.sin(2.0 * math.pi * 173.0 * time)
        + 0.07 * torch.sin(2.0 * math.pi * 947.0 * time + 0.31)
        + 0.025 * torch.cos(2.0 * math.pi * 2_113.0 * time)
        + 0.004 * noise
    )
    return pcm.clamp(-1.0, 1.0).contiguous()


def write_f32(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().to(torch.float32).contiguous().numpy()
    path.write_bytes(np.asarray(array, dtype="<f4").tobytes(order="C"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-checkpoint-sha-mismatch", action="store_true")
    args = parser.parse_args()

    checkpoint_sha = sha256(args.nemo)
    if (
        checkpoint_sha != EXPECTED_CHECKPOINT_SHA256
        and not args.allow_checkpoint_sha_mismatch
    ):
        raise SystemExit(
            f"checkpoint SHA-256 {checkpoint_sha} != pinned "
            f"{EXPECTED_CHECKPOINT_SHA256}"
        )

    from nemo.collections.asr.models import EncDecSpeakerLabelModel
    from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
    import nemo

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    model = EncDecSpeakerLabelModel.restore_from(
        restore_path=str(args.nemo), map_location="cpu"
    )
    model.eval()

    # The checkpoint was produced by NeMo 1.10.0. Its serialized config
    # predates the later `exact_pad` field, and NeMo 3 fills that missing field
    # with a new default. Recreate the official 1.10.0 preprocessor with every
    # relevant constructor value explicit so compatibility loading cannot
    # silently change center padding. The oracle remains NeMo's own module;
    # this script does not implement the frontend math.
    preprocessor = AudioToMelSpectrogramPreprocessor(
        sample_rate=SAMPLE_RATE,
        normalize="per_feature",
        window_size=0.025,
        window_stride=0.01,
        window="hann",
        features=80,
        n_fft=512,
        frame_splicing=1,
        dither=1.0e-5,
        exact_pad=False,
        pad_to=16,
    )
    preprocessor.load_state_dict(model.preprocessor.state_dict(), strict=True)
    preprocessor.eval()

    # NeMo 3 also changed two implementation details after 1.10.0: the
    # sequence-length helper dropped the historical `+ 1`, and STFT padding
    # became constant instead of torch.stft's then-default reflect mode. Bind
    # the two tiny methods verbatim to the pinned 1.10.0 contract while leaving
    # mel projection, log, normalization, encoder, pooling, and decoder in the
    # official NeMo implementation.
    def nemo_1_10_get_seq_len(featurizer, seq_len):
        pad_amount = (
            featurizer.stft_pad_amount * 2
            if featurizer.stft_pad_amount is not None
            else featurizer.n_fft // 2 * 2
        )
        return (
            torch.floor(
                (seq_len + pad_amount - featurizer.n_fft)
                / featurizer.hop_length
            )
            + 1
        ).to(dtype=torch.long)

    def nemo_1_10_stft(featurizer, values):
        return torch.stft(
            values,
            n_fft=featurizer.n_fft,
            hop_length=featurizer.hop_length,
            win_length=featurizer.win_length,
            center=True,
            window=featurizer.window.to(dtype=torch.float, device=values.device),
            return_complex=True,
            pad_mode="reflect",
        )

    preprocessor.featurizer.get_seq_len = types.MethodType(
        nemo_1_10_get_seq_len, preprocessor.featurizer
    )
    preprocessor.featurizer.stft = types.MethodType(
        nemo_1_10_stft, preprocessor.featurizer
    )

    pcm = deterministic_pcm()
    length = torch.tensor([pcm.numel()], dtype=torch.long)
    with torch.inference_mode():
        features, feature_length = preprocessor(
            input_signal=pcm.unsqueeze(0), length=length
        )
        encoded, encoded_length = model.encoder(
            audio_signal=features, length=feature_length
        )
        _, embedding = model.decoder(
            encoder_output=encoded, length=encoded_length
        )

    frames = int(feature_length[0].item())
    encoded_frames = int(encoded_length[0].item())
    if frames != encoded_frames:
        raise RuntimeError(
            f"stride-1 encoder changed frame count: {frames} -> {encoded_frames}"
        )
    features = features[0, :, :frames]
    embedding = embedding[0]
    if tuple(features.shape) != (80, frames):
        raise RuntimeError(f"unexpected frontend shape {tuple(features.shape)}")
    if tuple(embedding.shape) != (192,):
        raise RuntimeError(f"unexpected embedding shape {tuple(embedding.shape)}")

    args.output.mkdir(parents=True, exist_ok=True)
    write_f32(args.output / "pcm.f32.bin", pcm)
    write_f32(args.output / "features.f32.bin", features)
    write_f32(args.output / "embedding.f32.bin", embedding)
    manifest = {
        "format": "vokra-titanet-reference-v1",
        "oracle": "nemo.collections.asr.models.EncDecSpeakerLabelModel",
        "compatibility_contract": (
            "NeMo-1.10.0 exact_pad=false, reflect STFT, get_seq_len +1"
        ),
        "nemo_toolkit": getattr(nemo, "__version__", "unknown"),
        "upstream_hf": UPSTREAM_HF,
        "upstream_revision": UPSTREAM_REVISION,
        "source_revision": SOURCE_REVISION,
        "checkpoint_sha256": checkpoint_sha,
        "sample_rate": SAMPLE_RATE,
        "num_samples": int(pcm.numel()),
        "frames": frames,
        "feature_shape": [80, frames],
        "embedding_shape": [192],
        "torch": torch.__version__,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
