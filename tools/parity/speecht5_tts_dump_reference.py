#!/usr/bin/env python3
"""Dump an independent official-Transformers SpeechT5 TTS reference.

The numerical oracle is ``SpeechT5ForTextToSpeech.generate_speech`` from the
pinned Transformers 4.45.2 package. Vokra code is never imported. The only
injection is the decoder-prenet dropout mask: SpeechT5 deliberately keeps that
dropout active during evaluation, so this tool supplies Vokra's documented
SplitMix64 mask stream to the *official* prenet. All encoder, attention,
decoder, stop-head and postnet arithmetic remains the official implementation.

Actual model execution is Linux/VAST-only. The small ``--self-test`` path has
no third-party imports and is safe on a maintainer machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import types
from pathlib import Path


UPSTREAM_HF = "microsoft/speecht5_tts"
UPSTREAM_REVISION = "30fcde30f19b87502b8435427b5f5068e401d5f6"
REFERENCE_IMPLEMENTATION = (
    "transformers.models.speecht5.modeling_speecht5."
    "SpeechT5ForTextToSpeech.generate_speech"
)
REFERENCE_PACKAGE = "transformers==4.45.2"
DETERMINISTIC_SEED = 0x5350_4545_4348_5435
DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog."
SPEAKER_DIM = 512
PRENET_UNITS = 256
PRENET_LAYERS = 2
MEL_BINS = 80
REDUCTION_FACTOR = 2

# Exact files consumed by from_pretrained at the pinned immutable revision.
PINNED_FILES = {
    "pytorch_model.bin": (
        585_476_837,
        "d60d28067349ef66b50d8cd643ae56b6d6b8f27def929bc4ef6fcad907954190",
    ),
    "spm_char.model": (
        238_473,
        "7fcc48f3e225f627b1641db410ceb0c8649bd2b0c982e150b03f8be3728ab560",
    ),
    "config.json": (
        2_062,
        "2caf62dde93699a90cfc35ff2a8de27b02b479a0c98881cbc55f9682cc43e258",
    ),
    "tokenizer_config.json": (
        232,
        "d589430c619db2d95ff0fa757a187b55ef5ea44eff7fb08a6fbf0e78e32a6247",
    ),
    "added_tokens.json": (
        40,
        "74be21ecff0a1fb1f304fe7c72ab21e4f0c046f8359fdf2852eb1b80967069ad",
    ),
    "special_tokens_map.json": (
        234,
        "2a098b61fe8ec4cfd7674832ca00b4268c07569743a4ad15c8164e8f60ebf981",
    ),
}


def digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def verify_checkpoint(checkpoint: Path) -> dict[str, str]:
    verified = {}
    for name, (expected_size, expected_sha256) in PINNED_FILES.items():
        path = checkpoint / name
        if not path.is_file():
            raise SystemExit(f"SpeechT5 parity: missing pinned file: {path}")
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise SystemExit(
                f"SpeechT5 parity: {name} has {actual_size} bytes, "
                f"expected {expected_size}"
            )
        actual_sha256 = digest_file(path)
        if actual_sha256 != expected_sha256:
            raise SystemExit(
                f"SpeechT5 parity: {name} SHA-256 {actual_sha256} != "
                f"pinned {expected_sha256}"
            )
        verified[name] = actual_sha256
    return verified


def require_vast() -> None:
    if platform.system() != "Linux":
        raise SystemExit(
            "SpeechT5 reference generation is Linux/VAST-only; refusing "
            f"model execution on {platform.system()}"
        )
    if os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1":
        raise SystemExit(
            "VOKRA_PUBLISH_ON_VAST=1 is absent; run the repository VAST "
            "provisioner before loading the released checkpoint"
        )


class SplitMix64:
    """The public deterministic mask stream used by SpeechT5GenerationOptions."""

    MASK = (1 << 64) - 1

    def __init__(self, state: int) -> None:
        self.state = state & self.MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E37_79B9_7F4A_7C15) & self.MASK
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58_476D_1CE4_E5B9) & self.MASK
        value = ((value ^ (value >> 27)) * 0x94D0_49BB_1331_11EB) & self.MASK
        return (value ^ (value >> 31)) & self.MASK

    def next_unit_f32(self) -> float:
        return float(self.next_u64() >> 40) / float(1 << 24)


def speaker_values() -> list[float]:
    # Exact binary fractions exercise every speaker-projection column while
    # avoiding any dependence on a separately licensed x-vector dataset.
    return [float((index % 17) - 8) / 8.0 for index in range(SPEAKER_DIM)]


class OfficialPrenetDropout:
    """Inject SplitMix64 masks into the official prenet, without mirroring it."""

    def __init__(self, seed: int) -> None:
        self.rng = SplitMix64(seed)
        self.calls = 0

    def apply(self, inputs_embeds, probability: float):
        import torch

        if probability != 0.5:
            raise RuntimeError(
                f"official SpeechT5 prenet dropout changed to {probability}; "
                "review the deterministic mask contract"
            )
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1:
            raise RuntimeError(
                "parity fixture requires one official batch, got "
                f"shape={tuple(inputs_embeds.shape)}"
            )
        if inputs_embeds.shape[-1] != PRENET_UNITS:
            raise RuntimeError(
                f"official prenet width {inputs_embeds.shape[-1]} != {PRENET_UNITS}"
            )

        # _generate_speech runs the prenet over the complete output sequence,
        # then passes only the final row to the cached decoder. Prenet linears
        # are position-independent, so earlier rows cannot affect that row.
        # Fill only the consumed row from Vokra's stream and zero irrelevant
        # rows; this avoids advancing SplitMix for work the native cached path
        # intentionally does not perform.
        scale = 1.0 / (1.0 - probability)
        keep = [
            scale if self.rng.next_unit_f32() < probability else 0.0
            for _ in range(PRENET_UNITS)
        ]
        mask = torch.zeros_like(inputs_embeds)
        mask[:, -1, :] = torch.tensor(
            keep, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        self.calls += 1
        return inputs_embeds * mask


def self_test() -> None:
    rng = SplitMix64(0)
    assert rng.next_u64() == 0xE220_A839_7B1D_CDAF
    assert rng.next_u64() == 0x6E78_9E6A_A1B9_65F4
    first = SplitMix64(DETERMINISTIC_SEED)
    second = SplitMix64(DETERMINISTIC_SEED)
    assert [first.next_u64() for _ in range(32)] == [
        second.next_u64() for _ in range(32)
    ]
    speaker = speaker_values()
    assert len(speaker) == SPEAKER_DIM
    assert all(math.isfinite(value) for value in speaker)
    assert any(value < 0.0 for value in speaker)
    assert any(value > 0.0 for value in speaker)
    print("speecht5_tts_dump_reference: self-test PASS")


def write_f32(path: Path, array, numpy_module) -> tuple[int, str]:
    payload = numpy_module.asarray(array, dtype="<f4").tobytes(order="C")
    path.write_bytes(payload)
    return len(payload) // 4, digest_bytes(payload)


def write_u32(path: Path, values: list[int], numpy_module) -> tuple[int, str]:
    payload = numpy_module.asarray(values, dtype="<u4").tobytes(order="C")
    path.write_bytes(payload)
    return len(payload) // 4, digest_bytes(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.checkpoint is None or args.output_dir is None:
        parser.error(
            "--checkpoint and --output-dir are required unless --self-test is used"
        )
    if not args.text or args.text != args.text.strip():
        parser.error("--text must be non-empty and have no leading/trailing space")

    require_vast()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_dir():
        parser.error(f"checkpoint is not a directory: {checkpoint}")
    verified_files = verify_checkpoint(checkpoint)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"--output-dir must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import numpy as np
        import torch
        import transformers
        from transformers import SpeechT5ForTextToSpeech, SpeechT5Tokenizer
    except ImportError as error:
        raise SystemExit(
            "pinned SpeechT5 parity dependencies are unavailable; run with "
            "uv --project tools/parity/speecht5_tts --frozen. Import failed: "
            f"{error}"
        ) from error

    if transformers.__version__ != "4.45.2":
        raise RuntimeError(
            f"transformers {transformers.__version__} != pinned 4.45.2"
        )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    tokenizer = SpeechT5Tokenizer.from_pretrained(
        checkpoint, local_files_only=True
    )
    encoded = tokenizer(
        args.text, return_tensors="pt", return_attention_mask=True
    )
    input_ids = encoded["input_ids"].to(dtype=torch.long, device="cpu")
    attention_mask = encoded["attention_mask"].to(dtype=torch.long, device="cpu")
    tokens = [int(token) for token in input_ids[0].tolist()]
    if not tokens or tokens[-1] != 2 or any(token < 0 or token >= 81 for token in tokens):
        raise RuntimeError(
            f"official tokenizer emitted an invalid SpeechT5 sequence: {tokens}"
        )

    model = SpeechT5ForTextToSpeech.from_pretrained(
        checkpoint,
        local_files_only=True,
        use_safetensors=False,
    ).eval().to(device="cpu", dtype=torch.float32)
    config_contract = {
        "hidden_size": 768,
        "encoder_layers": 12,
        "decoder_layers": 6,
        "num_mel_bins": MEL_BINS,
        "reduction_factor": REDUCTION_FACTOR,
        "speaker_embedding_dim": SPEAKER_DIM,
        "speech_decoder_prenet_units": PRENET_UNITS,
        "speech_decoder_prenet_layers": PRENET_LAYERS,
    }
    for name, expected in config_contract.items():
        actual = getattr(model.config, name)
        if actual != expected:
            raise RuntimeError(
                f"official config {name}={actual!r}, expected {expected!r}"
            )

    dropout = OfficialPrenetDropout(DETERMINISTIC_SEED)
    prenet = model.speecht5.decoder.prenet

    def deterministic_dropout(_self, inputs_embeds, probability):
        return dropout.apply(inputs_embeds, probability)

    prenet._consistent_dropout = types.MethodType(  # noqa: SLF001
        deterministic_dropout, prenet
    )

    before_postnet = []
    postnet = model.speech_decoder_postnet
    official_postnet = postnet.postnet

    def capture_postnet(_self, hidden_states):
        before_postnet.append(hidden_states.detach().cpu().clone())
        return official_postnet(hidden_states)

    postnet.postnet = types.MethodType(capture_postnet, postnet)
    speaker = torch.tensor(
        speaker_values(), dtype=torch.float32, device="cpu"
    ).unsqueeze(0)

    with torch.inference_mode():
        generated, generated_lengths = model.generate_speech(
            input_ids,
            speaker_embeddings=speaker,
            attention_mask=attention_mask,
            threshold=0.5,
            minlenratio=0.0,
            maxlenratio=20.0,
            vocoder=None,
            output_cross_attentions=False,
            return_output_lengths=True,
        )

    if len(generated_lengths) != 1:
        raise RuntimeError(
            f"official generator returned lengths={generated_lengths!r}"
        )
    frames = int(generated_lengths[0])
    if frames <= 0 or frames % REDUCTION_FACTOR != 0:
        raise RuntimeError(f"official generated frame count is invalid: {frames}")
    if tuple(generated.shape) != (1, frames, MEL_BINS):
        raise RuntimeError(
            "official postnet output shape "
            f"{tuple(generated.shape)} != (1, {frames}, {MEL_BINS})"
        )
    if len(before_postnet) != 1 or tuple(before_postnet[0].shape) != (
        1,
        frames,
        MEL_BINS,
    ):
        raise RuntimeError(
            "official postnet capture is not one complete spectrogram: "
            f"{[tuple(value.shape) for value in before_postnet]}"
        )
    decoder_steps = frames // REDUCTION_FACTOR
    expected_dropout_calls = decoder_steps * PRENET_LAYERS
    if dropout.calls != expected_dropout_calls:
        raise RuntimeError(
            f"official prenet made {dropout.calls} dropout calls, expected "
            f"{expected_dropout_calls} for {decoder_steps} decoder steps"
        )

    speaker_count, speaker_sha256 = write_f32(
        output_dir / "speaker.f32", speaker[0].numpy(), np
    )
    token_count, tokens_sha256 = write_u32(output_dir / "tokens.u32", tokens, np)
    before_count, before_sha256 = write_f32(
        output_dir / "before_postnet.f32", before_postnet[0][0].numpy(), np
    )
    after_count, after_sha256 = write_f32(
        output_dir / "after_postnet.f32", generated[0].cpu().numpy(), np
    )
    (output_dir / "text.txt").write_text(args.text + "\n", encoding="utf-8")
    (output_dir / "frames.txt").write_text(f"{frames}\n", encoding="ascii")
    (output_dir / "decoder_steps.txt").write_text(
        f"{decoder_steps}\n", encoding="ascii"
    )

    report = {
        "format": "vokra-speecht5-tts-reference-v1",
        "reference_implementation": REFERENCE_IMPLEMENTATION,
        "reference_package": REFERENCE_PACKAGE,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "upstream_hf": UPSTREAM_HF,
        "upstream_revision": UPSTREAM_REVISION,
        "verified_files": verified_files,
        "text": args.text,
        "tokens": tokens,
        "token_count": token_count,
        "tokens_sha256": tokens_sha256,
        "speaker_count": speaker_count,
        "speaker_sha256": speaker_sha256,
        "dropout_seed": DETERMINISTIC_SEED,
        "dropout_algorithm": "SplitMix64 top-24-bit unit interval; keep when u < 0.5; scale 2",
        "dropout_calls": dropout.calls,
        "decoder_steps": decoder_steps,
        "frames": frames,
        "mel_bins": MEL_BINS,
        "before_postnet_count": before_count,
        "before_postnet_sha256": before_sha256,
        "after_postnet_count": after_count,
        "after_postnet_sha256": after_sha256,
    }
    (output_dir / "reference.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
