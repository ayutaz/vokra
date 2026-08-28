#!/usr/bin/env python3
"""Dump an independent official-NeMo ReazonSpeech v2 reference.

The oracle is ``EncDecRNNTBPEModel.restore_from`` from NVIDIA NeMo. This file
does not reproduce the frontend, Longformer attention, RNN-T decoder, greedy
search, or tokenizer. It records the official encoder output and exact emitted
token IDs for the committed 16 kHz JFK clip, and aborts if NeMo/checkpoint/API
provenance is unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

UPSTREAM_HF = "reazon-research/reazonspeech-nemo-v2"
UPSTREAM_REVISION = "33693408be76b7cba9fd4a7546a0a8772430211b"
ARCHIVE_SIZE = 2_477_946_880
ARCHIVE_SHA256 = "d196d43ad03466ca88beeda4bf5fafb07bab7202d4b663b8e4f12cb0a4381fae"
REFERENCE_IMPLEMENTATION = (
    "nemo.collections.asr.models.EncDecRNNTBPEModel.restore_from"
)
REFERENCE_PACKAGE = "nemo-toolkit[asr]==3.0.0"
JFK_SHA256 = "58adb4ea501d955fcd40bfbb69128f8f40428b81d8716b9ed337949773be253f"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIO = REPO_ROOT / "tests/fixtures/audio/jfk-30s.wav"


def digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require_vast() -> None:
    if platform.system() != "Linux":
        raise SystemExit(
            "ReazonSpeech reference generation is Linux/VAST-only; refusing "
            f"model execution on {platform.system()}"
        )
    if os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1":
        raise SystemExit(
            "VOKRA_PUBLISH_ON_VAST=1 is absent; run the repository VAST "
            "provisioner before loading the released checkpoint"
        )


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if line.startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def hypothesis_tokens(hypothesis: object) -> list[int]:
    sequence = getattr(hypothesis, "y_sequence", None)
    if sequence is None:
        sequence = getattr(hypothesis, "tokens", None)
    if sequence is None:
        raise RuntimeError(
            "official NeMo hypothesis exposes neither y_sequence nor tokens; "
            "refusing a text-only success-shaped reference"
        )
    if hasattr(sequence, "detach"):
        sequence = sequence.detach()
    if hasattr(sequence, "cpu"):
        sequence = sequence.cpu()
    if hasattr(sequence, "tolist"):
        sequence = sequence.tolist()
    if not isinstance(sequence, list):
        sequence = list(sequence)
    tokens = [int(token) for token in sequence]
    if not tokens:
        raise RuntimeError(
            "official NeMo emitted no tokens for the fixed clip; refusing an "
            "encoder-only success-shaped RNN-T gate"
        )
    if any(token < 0 or token >= 3_000 for token in tokens):
        raise RuntimeError(f"official NeMo emitted an invalid nonblank token: {tokens}")
    return tokens


def self_test() -> None:
    class Hypothesis:
        y_sequence = [1, 2, 2]

    assert hypothesis_tokens(Hypothesis()) == [1, 2, 2]
    assert len(ARCHIVE_SHA256) == len(JFK_SHA256) == 64
    print("reazonspeech_nemo_v2_dump_reference: self-test PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo", type=Path)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.nemo is None or args.output_dir is None:
        parser.error("--nemo and --output-dir are required unless --self-test is used")

    require_vast()
    nemo_path = args.nemo.resolve()
    audio_path = args.audio.resolve()
    if not nemo_path.is_file() or nemo_path.stat().st_size != ARCHIVE_SIZE:
        raise SystemExit(
            f"checkpoint must be the pinned {ARCHIVE_SIZE}-byte `.nemo`: {nemo_path}"
        )
    nemo_sha256 = digest_file(nemo_path)
    if nemo_sha256 != ARCHIVE_SHA256:
        raise SystemExit(
            f"checkpoint SHA-256 {nemo_sha256} != pinned {ARCHIVE_SHA256}"
        )
    if not audio_path.is_file():
        parser.error(f"audio is not a regular file: {audio_path}")
    audio_sha256 = digest_file(audio_path)
    if audio_path == DEFAULT_AUDIO.resolve() and audio_sha256 != JFK_SHA256:
        raise SystemExit(
            f"committed JFK fixture SHA-256 {audio_sha256} != pinned {JFK_SHA256}"
        )

    try:
        import nemo
        import numpy as np
        import soundfile as sf
        import torch
        from nemo.collections.asr.models import EncDecRNNTBPEModel
    except ImportError as error:
        raise SystemExit(
            "official NVIDIA NeMo is required; run through tools/parity with "
            f"--extra titanet. Import failed: {error}"
        ) from error

    pcm, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    if sample_rate != 16_000 or pcm.shape[1] != 1:
        raise RuntimeError(
            f"reference audio must be 16 kHz mono, got rate={sample_rate}, "
            f"shape={pcm.shape}"
        )

    torch.set_num_threads(1)
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_capability = getattr(torch.backends.cpu, "get_cpu_capability", None)
    environment = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "torch_cpu_capability": (
            cpu_capability() if callable(cpu_capability) else "unavailable"
        ),
        "device": str(device),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }
    print(
        json.dumps({"reference_environment": environment}, sort_keys=True),
        flush=True,
    )

    model = EncDecRNNTBPEModel.restore_from(
        restore_path=str(nemo_path), map_location=device
    )
    model.eval()
    model.freeze()
    model.to(device)
    decoding = getattr(model.cfg, "decoding", None)
    decoding_strategy = str(getattr(decoding, "strategy", "unknown"))
    if "greedy" not in decoding_strategy.lower():
        raise RuntimeError(
            f"released decoding strategy {decoding_strategy!r} is not greedy; "
            "the native oracle contract must be reviewed rather than inferred"
        )

    signal = torch.from_numpy(pcm[:, 0].copy()).to(device).unsqueeze(0)
    signal_length = torch.tensor([signal.shape[1]], dtype=torch.int64, device=device)
    with torch.inference_mode():
        forward = model.forward(
            input_signal=signal,
            input_signal_length=signal_length,
        )
        hypotheses = model.transcribe(
            [str(audio_path)], batch_size=1, return_hypotheses=True
        )
    if not isinstance(forward, (tuple, list)) or len(forward) != 2:
        raise RuntimeError(
            "official EncDecRNNTBPEModel.forward did not return "
            f"(encoded, encoded_length): {type(forward)}"
        )
    encoded, encoded_length = forward
    if encoded.ndim != 3 or encoded.shape[0] != 1 or encoded.shape[1] != 1_024:
        raise RuntimeError(
            f"official encoder shape must be [1,1024,T], got {tuple(encoded.shape)}"
        )
    frames = int(encoded_length[0].item())
    if frames <= 0 or frames > encoded.shape[2]:
        raise RuntimeError(
            f"official encoded length {frames} is invalid for {tuple(encoded.shape)}"
        )
    encoder_time_major = encoded[0, :, :frames].transpose(0, 1).contiguous()

    if not isinstance(hypotheses, list) or len(hypotheses) != 1:
        raise RuntimeError(
            f"official NeMo returned {type(hypotheses)} with unexpected length"
        )
    hypothesis = hypotheses[0]
    text = getattr(hypothesis, "text", None)
    if not isinstance(text, str):
        raise RuntimeError("official NeMo hypothesis has no string text field")
    tokens = hypothesis_tokens(hypothesis)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pcm_bytes = np.asarray(pcm[:, 0], dtype="<f4").tobytes(order="C")
    encoder_bytes = (
        encoder_time_major.detach()
        .cpu()
        .numpy()
        .astype("<f4", copy=False)
        .tobytes(order="C")
    )
    token_bytes = np.asarray(tokens, dtype="<u4").tobytes(order="C")
    (output_dir / "pcm.f32").write_bytes(pcm_bytes)
    (output_dir / "encoder.f32").write_bytes(encoder_bytes)
    (output_dir / "tokens.u32").write_bytes(token_bytes)
    (output_dir / "text.txt").write_text(text + "\n", encoding="utf-8")
    (output_dir / "encoder.frames.txt").write_text(
        f"{frames}\n", encoding="utf-8"
    )

    report = {
        "format": "vokra-reazonspeech-nemo-v2-reference-v1",
        "reference_implementation": REFERENCE_IMPLEMENTATION,
        "reference_package": REFERENCE_PACKAGE,
        "nemo_version": getattr(nemo, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "environment": environment,
        "upstream_hf": UPSTREAM_HF,
        "upstream_revision": UPSTREAM_REVISION,
        "checkpoint_sha256": nemo_sha256,
        "audio": (
            str(audio_path.relative_to(REPO_ROOT))
            if audio_path.is_relative_to(REPO_ROOT)
            else str(audio_path)
        ),
        "audio_sha256": audio_sha256,
        "sample_rate": sample_rate,
        "sample_count": int(pcm.shape[0]),
        "decoding_strategy": decoding_strategy,
        "encoder_frames": frames,
        "encoder_width": int(encoder_time_major.shape[1]),
        "encoder_sha256": digest_bytes(encoder_bytes),
        "tokens": tokens,
        "tokens_sha256": digest_bytes(token_bytes),
        "text": text,
    }
    (output_dir / "reference.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
