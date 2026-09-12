#!/usr/bin/env python3
"""Generate an independent real-weight Whisper-extras reference packet.

The only model implementation imported here is Hugging Face Transformers. In
particular, this script never imports Vokra or reimplements Whisper layers;
the packet therefore remains an oracle for the native Rust forward. It is
intended for the disposable VAST worker and must not be run with a large
checkpoint on the maintainer Mac.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import struct
import sys
from pathlib import Path

SCHEMA = "vokra-whisper-extras-reference-v1"
IDENTITY_FILES = {
    "model.safetensors": ("checkpoint_bytes", "checkpoint_sha256"),
    "config.json": ("config_bytes", "config_sha256"),
    "generation_config.json": ("generation_config_bytes", "generation_config_sha256"),
    "tokenizer.json": ("tokenizer_bytes", "tokenizer_sha256"),
}
MODELS = {
    "distil_whisper": {
        "repo": "distil-whisper/distil-large-v3.5",
        "revision": "728a7691f3ff1d3d971528d3203a6e9559165d41",
        "license": "MIT",
        "language": "en",
        "task": "transcribe",
        "no_timestamps": True,
        "checkpoint_bytes": 3025686376,
        "checkpoint_sha256": "76ec9f754fc4b4810845dc36b71d1897c1342e702810c179e1569690084cfb0c",
        "config_bytes": 1249,
        "config_sha256": "515a10a9979258d3fc71cf79b2cd055c189f07d78879a15bd9bc282673308b85",
        "generation_config_bytes": 4249,
        "generation_config_sha256": "b521c66612bd95be36c154f2d3904f6e4ea3be481a18a48f293d66791f60cf98",
        "tokenizer_bytes": 2480645,
        "tokenizer_sha256": "b3c8202bbf06d8ee4232c5984baa563784ac4737e2e7fdc42fa180200d3cfcdb",
    },
    "kotoba_whisper": {
        "repo": "kotoba-tech/kotoba-whisper-v2.2",
        "revision": "9d33482a0eb9b57f1ad80708e8ac5538246d8355",
        "license": "Apache-2.0",
        # The official v2.2 pipeline passes language="ja" and task="transcribe".
        # The token id is resolved from this snapshot's tokenizer below; do
        # not hand-code the id in the independent reference.
        "language": "ja",
        "task": "transcribe",
        "no_timestamps": True,
        "checkpoint_bytes": 3025686376,
        "checkpoint_sha256": "e0ef3e7b379515f0c35d0e7885638ddef0fa9f5c8e3e3f88cbc6da9b39edd1e9",
        "config_bytes": 1499,
        "config_sha256": "75e0166afbf44308af4908793fc5ade1707890ed15760f0529b468dcfc378aec",
        "generation_config_bytes": 3898,
        "generation_config_sha256": "20d28b9169207ab6ca402ec9342393a88ec3f341d88c9da24e516e1da71c24de",
        "tokenizer_bytes": 3931381,
        "tokenizer_sha256": "615928f5a25409279b47b47d87a4ca2aaee3bd09f65e1a3df6c9d23c718cfdb0",
    },
}
PCM_SAMPLES = 30 * 16_000
ENCODER_ROWS = 32
MAX_NEW_TOKENS = 224


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_keys(name: str) -> tuple[str, str]:
    try:
        return IDENTITY_FILES[name]
    except KeyError as exc:
        raise SystemExit(f"unknown checkpoint identity filename: {name}") from exc


def verify_checkpoint_identity(checkpoint: Path, spec: dict) -> dict[str, int | str]:
    """Fail closed unless every pinned source file is byte-identical."""
    identities: dict[str, int | str] = {}
    for name in ("model.safetensors", "config.json", "generation_config.json", "tokenizer.json"):
        path = checkpoint / name
        if not path.is_file():
            raise SystemExit(f"checkpoint is missing pinned {name}")
        size = path.stat().st_size
        size_key, sha_key = identity_keys(name)
        expected_size = spec[size_key]
        expected_sha = spec[sha_key]
        actual_sha = sha256(path)
        if size != expected_size or actual_sha != expected_sha:
            raise SystemExit(
                f"{name} identity mismatch: bytes={size} sha256={actual_sha}; "
                f"expected bytes={expected_size} sha256={expected_sha}"
            )
        identities[size_key] = size
        identities[sha_key] = actual_sha
    return identities


def write_f32(path: Path, values: np.ndarray | torch.Tensor) -> None:
    import numpy as np
    import torch

    array = values.detach().float().cpu().contiguous().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    path.write_bytes(np.asarray(array, dtype="<f4").reshape(-1).tobytes())


def write_u32(path: Path, values: list[int]) -> None:
    path.write_bytes(struct.pack("<{}I".format(len(values)), *values))


def load_pcm(path: Path) -> np.ndarray:
    """Read a real, already-normalized 16 kHz mono WAV; never resample/mix."""
    import numpy as np

    data = path.read_bytes()
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise SystemExit(f"{path}: expected RIFF/WAVE")
    fmt = payload = None
    offset = 12
    while offset + 8 <= len(data):
        chunk = data[offset : offset + 4]
        size = struct.unpack_from("<I", data, offset + 4)[0]
        start, end = offset + 8, offset + 8 + size
        if end > len(data):
            raise SystemExit(f"{path}: truncated WAV chunk")
        body = data[start:end]
        if chunk == b"fmt ":
            if len(body) < 16:
                raise SystemExit(f"{path}: short fmt chunk")
            audio_format, channels, rate = struct.unpack_from("<HHI", body, 0)
            bits = struct.unpack_from("<H", body, 14)[0]
            fmt = (audio_format, channels, rate, bits)
        elif chunk == b"data":
            payload = body
        offset = end + (size & 1)
    if fmt is None or payload is None:
        raise SystemExit(f"{path}: missing fmt or data chunk")
    audio_format, channels, rate, bits = fmt
    if (channels, rate) != (1, 16_000):
        raise SystemExit(f"{path}: expected mono 16 kHz, got channels={channels}, rate={rate}")
    if audio_format == 1 and bits == 16:
        pcm = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    elif audio_format == 3 and bits == 32:
        pcm = np.frombuffer(payload, dtype="<f4").astype(np.float32)
    else:
        raise SystemExit(f"{path}: unsupported WAV format={audio_format}, bits={bits}")
    if not np.isfinite(pcm).all():
        raise SystemExit(f"{path}: non-finite PCM")
    if len(pcm) >= PCM_SAMPLES:
        return np.ascontiguousarray(pcm[:PCM_SAMPLES])
    padded = np.zeros(PCM_SAMPLES, dtype=np.float32)
    padded[: len(pcm)] = pcm
    return padded


def special_tokens(processor, checkpoint: Path, spec: dict) -> tuple[list[int], int]:
    tokenizer = processor.tokenizer
    language = spec["language"]
    task = spec["task"]
    generation_path = checkpoint / "generation_config.json"
    try:
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read official generation_config.json: {exc}") from exc
    no_timestamps = generation.get("return_timestamps") is False
    if no_timestamps != spec["no_timestamps"]:
        raise SystemExit("model semantics disagree with generation_config return_timestamps")
    names = [
        "<|startoftranscript|>",
        f"<|{language}|>",
        f"<|{task}|>",
        "<|notimestamps|>",
        "<|endoftext|>",
    ]
    ids = [tokenizer.convert_tokens_to_ids(name) for name in names]
    for name, token_id in zip(names, ids):
        if token_id is None or tokenizer.convert_ids_to_tokens(token_id) != name:
            raise SystemExit(f"tokenizer special-token round-trip failed for {name}: {token_id}")
    if generation.get("decoder_start_token_id") != ids[0]:
        raise SystemExit("generation_config decoder_start_token_id disagrees with tokenizer")
    task_id = generation.get("task_to_id", {}).get(task)
    if task_id != ids[2]:
        raise SystemExit(f"generation_config task_to_id[{task!r}] disagrees with tokenizer")
    no_timestamps_id = generation.get("no_timestamps_token_id")
    if no_timestamps and no_timestamps_id != ids[3]:
        raise SystemExit("generation_config no_timestamps_token_id disagrees with tokenizer")
    language_id = generation.get("lang_to_id", {}).get(names[1])
    if language_id != ids[1]:
        raise SystemExit(f"generation_config lang_to_id[{names[1]!r}] disagrees with tokenizer")
    return ids[:4], ids[4]


def greedy(model, encoder_outputs, prefix: list[int], eot: int) -> list[int]:
    import torch

    tokens = list(prefix)
    generated: list[int] = []
    for _ in range(MAX_NEW_TOKENS):
        ids = torch.tensor([tokens], dtype=torch.long)
        logits = model(encoder_outputs=encoder_outputs, decoder_input_ids=ids).logits
        token = int(logits[0, -1].argmax().item())
        generated.append(token)
        if token == eot:
            break
        tokens.append(token)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model", choices=sorted(MODELS))
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        expected = {
            "model.safetensors": ("checkpoint_bytes", "checkpoint_sha256"),
            "config.json": ("config_bytes", "config_sha256"),
            "generation_config.json": ("generation_config_bytes", "generation_config_sha256"),
            "tokenizer.json": ("tokenizer_bytes", "tokenizer_sha256"),
        }
        if IDENTITY_FILES != expected:
            raise SystemExit("identity filename mapping drift")
        for name, keys in expected.items():
            if identity_keys(name) != keys:
                raise SystemExit(f"identity mapping drift for {name}")
        try:
            identity_keys("unexpected.json")
        except SystemExit as exc:
            if "unknown checkpoint identity filename" not in str(exc):
                raise
        else:
            raise SystemExit("unknown identity filename was accepted")
        print("dump_reference self-test: OK")
        return
    if not all((args.model, args.checkpoint_dir, args.audio, args.output_dir)):
        parser.error("--model, --checkpoint-dir, --audio, and --output-dir are required")
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    spec = MODELS[args.model]
    checkpoint = args.checkpoint_dir
    if not (checkpoint.is_dir() and (checkpoint / "config.json").is_file()):
        raise SystemExit("checkpoint-dir must be an HF snapshot with config.json")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise SystemExit(f"output-dir must be absent (dumper owns creation): {args.output_dir}")
    checkpoint_identity = verify_checkpoint_identity(checkpoint, spec)
    pcm = load_pcm(args.audio)
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    # These are the official independent classes, loaded from the pinned
    # snapshot. Do not replace with a Vokra mirror or a hand-written forward.
    processor = WhisperProcessor.from_pretrained(str(checkpoint), local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(str(checkpoint), local_files_only=True)
    model.eval()
    prefix, eot = special_tokens(processor, checkpoint, spec)
    with torch.inference_mode():
        features = processor.feature_extractor(pcm, sampling_rate=16_000, return_tensors="pt").input_features
        encoder_outputs = model.model.encoder(features)
        encoder = encoder_outputs.last_hidden_state
        decoder_input = torch.tensor([prefix], dtype=torch.long)
        logits = model(encoder_outputs=encoder_outputs, decoder_input_ids=decoder_input).logits
        generated = greedy(model, encoder_outputs, prefix, eot)
    if encoder.shape[1] < ENCODER_ROWS:
        raise SystemExit(f"encoder has only {encoder.shape[1]} rows")
    write_f32(args.output_dir / "input_pcm.f32le", pcm)
    write_f32(args.output_dir / "encoder.f32le", encoder[0, :ENCODER_ROWS])
    write_f32(args.output_dir / "logits_last.f32le", logits[0, -1])
    write_u32(args.output_dir / "greedy_tokens.u32le", generated)
    files = {
        name: sha256(args.output_dir / name)
        for name in ("input_pcm.f32le", "encoder.f32le", "logits_last.f32le", "greedy_tokens.u32le")
    }
    manifest = {
        "schema": SCHEMA,
        "model": args.model,
        "upstream_repo": spec["repo"],
        "upstream_revision": spec["revision"],
        "weight_license_spdx": spec["license"],
        "reference_implementation": "transformers.WhisperForConditionalGeneration",
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "torch_num_threads": torch.get_num_threads(),
        "seed": 0,
        "sample_rate": 16_000,
        "pcm_samples": len(pcm),
        "audio_sha256": sha256(args.audio),
        **checkpoint_identity,
        "encoder_rows": ENCODER_ROWS,
        "d_model": int(encoder.shape[-1]),
        "vocab": int(logits.shape[-1]),
        "decoder_prefix": prefix,
        "language": spec["language"],
        "task": spec["task"],
        "no_timestamps": spec["no_timestamps"],
        "semantics_source": (
            "official Kotoba v2.2 pipeline + pinned tokenizer/generation_config"
            if args.model == "kotoba_whisper"
            else "pinned tokenizer/generation_config"
        ),
        "eot": eot,
        "greedy_tokens": generated,
        "files": files,
        "atol": 0.01,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"wrote independent {args.model} reference packet to {args.output_dir}")


if __name__ == "__main__":
    main()
