#!/usr/bin/env python3
"""Dump an independent NVIDIA NeMo reference for Canary-1B-Flash.

The oracle is the official ``EncDecMultiTaskModel`` imported from
``nemo.collections.asr.models``. This script contains no reimplementation of
the frontend, encoder, decoder, tokenizer, or greedy search. It runs only on a
provisioned Linux/VAST host and records exact token IDs plus decoded text for
the committed 16 kHz JFK clip.

Run with the pinned NeMo optional environment on VAST::

    VOKRA_PUBLISH_ON_VAST=1 uv run --project tools/parity --extra titanet \
      --python 3.12 python tools/parity/canary_1b_flash_dump_reference.py \
      --nemo /workspace/canary-1b-flash.nemo \
      --source-language en --target-language en \
      --output /workspace/canary/reference-en-en.json

No reference values are synthesized when NeMo or the released checkpoint is
unavailable; the script aborts loudly instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

UPSTREAM_HF = "nvidia/canary-1b-flash"
UPSTREAM_REVISION = "2b6e4d2dacb11cc1b1724de31bb48fe68c26c12e"
ARCHIVE_SIZE = 3_540_715_520
ARCHIVE_SHA256 = "3887cce1afdd425429cfc5109575a8f2cffeb07c02c503a9faff7612bd74e324"
REFERENCE_IMPLEMENTATION = (
    "nemo.collections.asr.models.EncDecMultiTaskModel.restore_from"
)
REFERENCE_PACKAGE = "nemo-toolkit[asr]==3.0.0"
# Source used to audit the decoder/prompt/hypothesis semantics. The executed
# oracle remains the separately pinned PyPI package above; do not mislabel this
# audit revision as the package build's provenance commit.
REFERENCE_SOURCE_AUDIT_COMMIT = "837a31fa7a810a3de9e4826837e97dea837a5c42"
LANGUAGES = ("en", "de", "es", "fr")
JFK_SHA256 = "58adb4ea501d955fcd40bfbb69128f8f40428b81d8716b9ed337949773be253f"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIO = REPO_ROOT / "tests/fixtures/audio/jfk-30s.wav"


def digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def require_vast() -> None:
    if platform.system() != "Linux":
        raise SystemExit(
            "Canary reference generation is Linux/VAST-only; refusing model "
            f"execution on {platform.system()}"
        )
    if os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1":
        raise SystemExit(
            "VOKRA_PUBLISH_ON_VAST=1 is absent; run the repository VAST "
            "provisioner before loading the released checkpoint"
        )


def validate_language(code: str) -> str:
    if code not in LANGUAGES:
        raise ValueError(f"language must be one of {LANGUAGES}, got {code!r}")
    return code


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
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
        raise RuntimeError("official NeMo hypothesis returned an empty token sequence")
    return tokens


def self_test() -> None:
    for language in LANGUAGES:
        assert validate_language(language) == language
    try:
        validate_language("ja")
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported language must fail")
    print("canary_1b_flash_dump_reference: self-test PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo", type=Path)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--source-language", default="en")
    parser.add_argument("--target-language")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.nemo is None or args.output is None:
        parser.error("--nemo and --output are required unless --self-test is used")

    require_vast()
    source = validate_language(args.source_language)
    target = validate_language(args.target_language or source)
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
        from nemo.collections.asr.models import EncDecMultiTaskModel
    except ImportError as error:
        raise SystemExit(
            "official NVIDIA NeMo is required; run through tools/parity with "
            f"--extra titanet. Import failed: {error}"
        ) from error

    pcm, sample_rate = sf.read(
        str(audio_path), dtype="float32", always_2d=True
    )
    if sample_rate != 16_000 or pcm.shape[1] != 1:
        raise RuntimeError(
            f"reference audio must be 16 kHz mono, got rate={sample_rate}, shape={pcm.shape}"
        )

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
    # Numerical-parity policy: record the execution environment before the
    # model emits values, so a later platform-specific discrepancy is
    # diagnosable rather than dismissed as an untracked rerun.
    print(json.dumps({"reference_environment": environment}, sort_keys=True), flush=True)
    model = EncDecMultiTaskModel.restore_from(
        restore_path=str(nemo_path), map_location=device
    )
    model.eval()
    model.to(device)

    taskname = "asr" if source == target else "ast"
    manifest_row = {
        "audio_filepath": str(audio_path),
        "duration": None,
        "taskname": taskname,
        "source_lang": source,
        "target_lang": target,
        "decodercontext": "",
        "emotion": "<|emo:undefined|>",
        "pnc": "yes",
        "itn": "noitn",
        "timestamp": "notimestamp",
        "diarize": "nodiarize",
        "answer": "",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vokra-canary-reference-") as temp_dir:
        manifest_path = Path(temp_dir) / "manifest.jsonl"
        manifest_path.write_text(json.dumps(manifest_row) + "\n", encoding="utf-8")
        with torch.inference_mode():
            hypotheses = model.transcribe(
                str(manifest_path), batch_size=1, return_hypotheses=True
            )

    if not isinstance(hypotheses, list) or len(hypotheses) != 1:
        raise RuntimeError(
            f"official NeMo returned {type(hypotheses)} / len={getattr(hypotheses, '__len__', lambda: '?')()}"
        )
    hypothesis = hypotheses[0]
    text = getattr(hypothesis, "text", None)
    if not isinstance(text, str):
        raise RuntimeError("official NeMo hypothesis has no string text field")
    tokens = hypothesis_tokens(hypothesis)

    report = {
        "format": "vokra-canary-1b-flash-nemo-reference-v1",
        "reference_implementation": REFERENCE_IMPLEMENTATION,
        "reference_package": REFERENCE_PACKAGE,
        "reference_source_audit_commit": REFERENCE_SOURCE_AUDIT_COMMIT,
        "nemo_version": getattr(nemo, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "environment": environment,
        "upstream_hf": UPSTREAM_HF,
        "upstream_revision": UPSTREAM_REVISION,
        "checkpoint_sha256": nemo_sha256,
        "audio": str(audio_path.relative_to(REPO_ROOT))
        if audio_path.is_relative_to(REPO_ROOT)
        else str(audio_path),
        "audio_sha256": audio_sha256,
        "sample_rate": sample_rate,
        "sample_count": int(pcm.shape[0]),
        "source_language": source,
        "target_language": target,
        "taskname": taskname,
        "text": text,
        "tokens": tokens,
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".tokens.txt").write_text(
        " ".join(str(token) for token in tokens) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".text.txt").write_text(text + "\n", encoding="utf-8")
    args.output.with_suffix(".pcm.f32").write_bytes(
        np.asarray(pcm[:, 0], dtype="<f4").tobytes(order="C")
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
