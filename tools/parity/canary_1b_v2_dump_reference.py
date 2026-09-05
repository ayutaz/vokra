#!/usr/bin/env python3
"""Dump an independent official NVIDIA NeMo Canary-1B-v2 reference.

The oracle is ``EncDecMultiTaskModel.restore_from`` from NVIDIA NeMo. This
script reimplements none of the frontend, encoder, decoder, tokenizer, prompt,
or greedy search. It runs only on a provisioned Linux/VAST host and records
the execution environment before producing exact token IDs and decoded text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from pathlib import Path

UPSTREAM_HF = "nvidia/canary-1b-v2"
UPSTREAM_REVISION = "87bc52657add533cd0156b3fc1aef027280754bf"
ARCHIVE_SIZE = 6_358_958_080
ARCHIVE_SHA256 = "ae5ef1bf06812a95a1594a8f5f0ee9c51f35418e5ba96939fa6b98ab00431094"
REFERENCE_IMPLEMENTATION = (
    "nemo.collections.asr.models.EncDecMultiTaskModel.restore_from"
)
REFERENCE_PACKAGE = "nemo-toolkit[asr]==3.0.0"
REFERENCE_SOURCE_AUDIT_COMMIT = "837a31fa7a810a3de9e4826837e97dea837a5c42"
LANGUAGES = (
    "bg",
    "hr",
    "cs",
    "da",
    "nl",
    "en",
    "et",
    "fi",
    "fr",
    "de",
    "el",
    "hu",
    "it",
    "lv",
    "lt",
    "mt",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sl",
    "es",
    "sv",
    "uk",
)
JFK_SHA256 = "58adb4ea501d955fcd40bfbb69128f8f40428b81d8716b9ed337949773be253f"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIO = REPO_ROOT / "tests/fixtures/audio/jfk-30s.wav"
DIAGNOSTIC_DEFAULT_TOP_K = 8


def digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def require_vast() -> None:
    if platform.system() != "Linux":
        raise SystemExit(
            "Canary-1B-v2 reference generation is Linux/VAST-only; refusing "
            f"model execution on {platform.system()}"
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


def parse_diagnostic_prefix(value: str) -> list[int]:
    """Parse an explicitly requested forced decoder prefix.

    This is deliberately a diagnostic-only input.  The normal reference path
    continues to obtain its prompt and generated sequence from NeMo.
    """
    fields = value.replace(",", " ").split()
    if not fields:
        raise ValueError("diagnostic prefix must contain at least one token id")
    try:
        tokens = [int(field, 10) for field in fields]
    except ValueError as error:
        raise ValueError("diagnostic prefix must contain decimal token ids") from error
    if any(token < 0 for token in tokens):
        raise ValueError("diagnostic prefix token ids must be non-negative")
    return tokens


def top_k_logits(values: list[float], k: int) -> list[tuple[int, float]]:
    """Return finite logits ordered like an argmax with deterministic ties."""
    if k <= 0:
        raise ValueError("diagnostic top-k must be positive")
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("diagnostic logits must be non-empty and finite")
    ranked = sorted(enumerate(values), key=lambda item: (-item[1], item[0]))
    return ranked[: min(k, len(ranked))]


def official_forced_prefix_diagnostic(
    model: object,
    torch: object,
    np: object,
    pcm: object,
    source: str,
    target: str,
    forced_prefix: list[int],
    top_k: int,
) -> dict[str, object]:
    """Score a forced prefix with NeMo's real encoder/decoder forward.

    ``model.forward`` is the official ``EncDecMultiTaskModel`` path: it runs
    the checkpoint's preprocessor, encoder, ``transf_decoder`` and
    ``log_softmax``.  No decoder or tokenizer mirror is used here.  Disabling
    the classifier's final log-softmax is only to expose the raw classifier
    logits for this opt-in diagnostic; the previous setting is restored before
    returning.
    """
    turns = [
        {
            "role": "user",
            "slots": {
                "decodercontext": "",
                "emotion": "<|emo:undefined|>",
                "source_lang": source,
                "target_lang": target,
                "pnc": "yes",
                "itn": "noitn",
                "timestamp": "notimestamp",
                "diarize": "nodiarize",
            },
        }
    ]
    prompt_tensor = model.prompt.encode_dialog(turns=turns)["context_ids"]
    prompt_ids = [int(token) for token in prompt_tensor.detach().cpu().tolist()]
    all_ids = prompt_ids + forced_prefix
    device = torch.device("cpu")
    audio = torch.from_numpy(np.asarray(pcm[:, 0], dtype=np.float32)).unsqueeze(0).to(device)
    audio_length = torch.tensor([audio.shape[1]], dtype=torch.long, device=device)
    transcript = torch.tensor([all_ids], dtype=torch.long, device=device)
    transcript_length = torch.tensor([len(all_ids)], dtype=torch.long, device=device)
    classifier = getattr(model, "log_softmax", None)
    mlp = getattr(classifier, "mlp", None)
    if mlp is None or not hasattr(mlp, "log_softmax"):
        raise RuntimeError(
            "official NeMo model does not expose TokenClassifier.mlp.log_softmax; "
            "refusing a mirror or text-only diagnostic"
        )
    previous_log_softmax = mlp.log_softmax
    mlp.log_softmax = False
    try:
        with torch.inference_mode():
            outputs = model.forward(
                input_signal=audio,
                input_signal_length=audio_length,
                transcript=transcript,
                transcript_length=transcript_length,
            )
    finally:
        mlp.log_softmax = previous_log_softmax
    raw_logits = outputs[0]
    if raw_logits is None:
        raise RuntimeError("official NeMo decoder returned no forced-prefix logits")
    values = [float(value) for value in raw_logits[0, -1].detach().cpu().float().tolist()]
    ranked = top_k_logits(values, top_k)
    top_two = top_k_logits(values, 2)
    return {
        "prompt_ids": prompt_ids,
        "forced_prefix": forced_prefix,
        "decoder_position": len(all_ids),
        "top_k": [
            {"token_id": token_id, "logit": logit} for token_id, logit in ranked
        ],
        "top1_top2_margin": top_two[0][1] - top_two[1][1]
        if len(top_two) > 1
        else None,
        "finite": all(math.isfinite(value) for value in values),
        "implementation": "official_nemo_model.forward",
    }


def self_test() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    assert 'torch.device("cpu")' in source
    assert "torch." + "cuda" not in source
    assert '"cuda_device": None' in source
    production = source[source.index("def main") :]
    lines = production.splitlines()
    cpu_env_line = next(
        index
        for index, line in enumerate(lines)
        if line.strip() == 'os.environ["CUDA_VISIBLE_DEVICES"] = ""'
    )
    nemo_import_line = next(
        index
        for index, line in enumerate(lines)
        if line.strip() == "import " + "nemo"
    )
    assert cpu_env_line < nemo_import_line
    assert len(LANGUAGES) == 25
    assert len(set(LANGUAGES)) == 25
    for language in LANGUAGES:
        assert validate_language(language) == language
    try:
        validate_language("ja")
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported language must fail")
    assert parse_diagnostic_prefix("3651, 1402 16067") == [3651, 1402, 16067]
    try:
        parse_diagnostic_prefix(" ")
    except ValueError:
        pass
    else:
        raise AssertionError("empty diagnostic prefix must fail")
    assert top_k_logits([1.0, 4.0, 4.0, -1.0], 3) == [
        (1, 4.0),
        (2, 4.0),
        (0, 1.0),
    ]
    assert top_k_logits([1.0, 4.0], 8) == [(1, 4.0), (0, 1.0)]
    print("canary_1b_v2_dump_reference: self-test PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo", type=Path)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--source-language", default="en")
    parser.add_argument("--target-language")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--diagnostic-prefix",
        help="opt-in comma/space-separated generated token prefix for official logits",
    )
    parser.add_argument(
        "--diagnostic-top-k",
        type=int,
        default=DIAGNOSTIC_DEFAULT_TOP_K,
        help="top-k logits to emit with --diagnostic-prefix (default: 8)",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.nemo is None or args.output is None:
        parser.error("--nemo and --output are required unless --self-test is used")
    if args.diagnostic_prefix is not None and args.diagnostic_top_k <= 0:
        parser.error("--diagnostic-top-k must be positive")
    diagnostic_prefix = (
        parse_diagnostic_prefix(args.diagnostic_prefix)
        if args.diagnostic_prefix is not None
        else None
    )

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

    # NeMo's ModelPT constructor probes CUDA during import/restore. This
    # worker is a CPU oracle, so suppress that internal probe before imports;
    # do not honor a caller-provided GPU visibility setting.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
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

    pcm, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    if sample_rate != 16_000 or pcm.shape[1] != 1:
        raise RuntimeError(
            f"reference audio must be 16 kHz mono, got rate={sample_rate}, shape={pcm.shape}"
        )

    # Numerical-parity policy: the official NeMo run is a CPU oracle. Do not
    # probe CUDA or fall back to it, since a visible GPU is not part of this
    # worker's reproducibility contract.
    device = torch.device("cpu")
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
        "cuda_device": None,
        "cuda_visible_devices": "",
    }
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
    if diagnostic_prefix is not None:
        diagnostic = official_forced_prefix_diagnostic(
            model,
            torch,
            np,
            pcm,
            source,
            target,
            diagnostic_prefix,
            args.diagnostic_top_k,
        )
        print(json.dumps({"canary_v2_diagnostic": diagnostic}, sort_keys=True), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vokra-canary-v2-reference-") as temp_dir:
        manifest_path = Path(temp_dir) / "manifest.jsonl"
        manifest_path.write_text(json.dumps(manifest_row) + "\n", encoding="utf-8")
        with torch.inference_mode():
            hypotheses = model.transcribe(
                str(manifest_path), batch_size=1, return_hypotheses=True
            )

    if not isinstance(hypotheses, list) or len(hypotheses) != 1:
        length = getattr(hypotheses, "__len__", lambda: "?")()
        raise RuntimeError(
            f"official NeMo returned {type(hypotheses)} / len={length}"
        )
    hypothesis = hypotheses[0]
    text = getattr(hypothesis, "text", None)
    if not isinstance(text, str):
        raise RuntimeError("official NeMo hypothesis has no string text field")
    tokens = hypothesis_tokens(hypothesis)

    report = {
        "format": "vokra-canary-1b-v2-nemo-reference-v1",
        "reference_implementation": REFERENCE_IMPLEMENTATION,
        "reference_package": REFERENCE_PACKAGE,
        "reference_source_audit_commit": REFERENCE_SOURCE_AUDIT_COMMIT,
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
