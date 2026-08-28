#!/usr/bin/env python3
"""Dump an independent official Qwen3-ASR real-checkpoint reference.

This script imports the immutable official ``qwen-asr==0.0.6`` package.  It
does not mirror any Qwen layer.  The tap named ``audio_embeddings.f32le`` is
the return value of the package's
``Qwen3ASRThinkerForConditionalGeneration.get_audio_features`` method.  Prompt
construction, feature extraction, generation, batch decoding and output
parsing all call the official package as well.

The model snapshot must already be local and must have been downloaded at the
exact revision selected by ``--variant``.  Network fallback is disabled before
the package is imported.  This is a VAST-only tool: both released snapshots
and the generated GGUF validation run cross the repository's local-memory
guard.  It never uploads or publishes anything.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Variant:
    slug: str
    repo: str
    revision: str
    model_name: str
    hidden_size: int
    tensor_count: int


VARIANTS = {
    "0.6b": Variant(
        slug="0.6b",
        repo="Qwen/Qwen3-ASR-0.6B",
        revision="5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
        model_name="qwen3-asr-0.6b",
        hidden_size=1024,
        tensor_count=612,
    ),
    "1.7b": Variant(
        slug="1.7b",
        repo="Qwen/Qwen3-ASR-1.7B",
        revision="7278e1e70fe206f11671096ffdd38061171dd6e5",
        model_name="qwen3-asr-1.7b",
        hidden_size=2048,
        tensor_count=708,
    ),
}

EXPECTED_ASSETS = {
    "vocab.json": (2_776_833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
    "merges.txt": (1_671_853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
    "tokenizer_config.json": (12_487, "4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c"),
    "chat_template.json": (1_161, "75a8cfca24f00de72d796fbfed6858fc9614ef3dabd8696684cc3bc03a9c58ff"),
    "generation_config.json": (142, "1da527824d81e07118facff437e03f2e24a23311e3bdeb2368973fe77e5f275c"),
}

SAMPLE_RATE = 16_000
QWEN_ASR_VERSION = "0.0.6"
TRANSFORMERS_VERSION = "4.57.6"
SCHEMA = "vokra-qwen3-asr-reference-v1"


def die(message: str) -> "None":
    raise SystemExit(f"qwen3_asr reference: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def safe_manifest_value(value: object) -> str:
    text = str(value)
    if "\n" in text or "\r" in text or "=" in text:
        raise ValueError(f"unsafe manifest value {text!r}")
    return text


def write_manifest(path: Path, values: dict[str, object]) -> None:
    lines = [f"{key}={safe_manifest_value(value)}" for key, value in sorted(values.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cpu_model() -> str:
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def cpu_flags() -> str:
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if re.match(r"^(flags|features)\s*:", line, flags=re.IGNORECASE):
                return line.split(":", 1)[1].strip()
    return "unknown"


def require_empty_output(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    entries = list(path.iterdir())
    if entries:
        die(f"--output must be empty, found {entries[0]}")


def require_model_identity(model_dir: Path, variant: Variant) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        die(f"missing local config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_asr":
        die(f"config model_type={config.get('model_type')!r}, expected 'qwen3_asr'")
    architectures = config.get("architectures")
    if architectures != ["Qwen3ASRForConditionalGeneration"]:
        die(f"config architectures={architectures!r}, expected the official Qwen3-ASR class")
    try:
        hidden_size = int(config["thinker_config"]["text_config"]["hidden_size"])
    except (KeyError, TypeError, ValueError) as error:
        die(f"config lacks thinker_config.text_config.hidden_size: {error}")
    if hidden_size != variant.hidden_size:
        die(f"config hidden_size={hidden_size}, expected {variant.hidden_size} for {variant.slug}")

    for name, (expected_bytes, expected_hash) in EXPECTED_ASSETS.items():
        path = model_dir / name
        if not path.is_file():
            die(f"missing pinned sidecar: {path}")
        actual_bytes = path.stat().st_size
        actual_hash = sha256_file(path)
        if (actual_bytes, actual_hash) != (expected_bytes, expected_hash):
            die(
                f"{name} identity drift: bytes={actual_bytes} sha256={actual_hash}; "
                f"expected bytes={expected_bytes} sha256={expected_hash}"
            )
    return config


def source_inventory(model_dir: Path) -> dict[str, dict[str, object]]:
    names = {
        "config.json",
        "model.safetensors.index.json",
        *EXPECTED_ASSETS.keys(),
    }
    names.update(path.name for path in model_dir.glob("*.safetensors"))
    inventory: dict[str, dict[str, object]] = {}
    for name in sorted(names):
        path = model_dir / name
        if path.is_file():
            inventory[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if not any(name.endswith(".safetensors") for name in inventory):
        die(f"no local safetensors files found in {model_dir}")
    return inventory


def write_f32(path: Path, array: Any, numpy: Any) -> None:
    values = numpy.asarray(array, dtype=numpy.float32)
    if not numpy.isfinite(values).all():
        die(f"non-finite values in {path.name}")
    path.write_bytes(values.astype("<f4", copy=False).tobytes(order="C"))


def write_u32(path: Path, values: Any, numpy: Any) -> None:
    array = numpy.asarray(values)
    if array.size and (array.min() < 0 or array.max() > 0xFFFFFFFF):
        die(f"token id outside u32 in {path.name}")
    path.write_bytes(array.astype("<u4", copy=False).tobytes(order="C"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", default="")
    parser.add_argument(
        "--language",
        default="English",
        help="official language name, or 'auto'; forced English is the stable parity default",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    variant = VARIANTS[args.variant]
    model_dir = args.model_dir.resolve()
    audio_path = args.audio.resolve()
    output = args.output.resolve()
    if not model_dir.is_dir():
        die(f"--model-dir is not a directory: {model_dir}")
    if not audio_path.is_file():
        die(f"--audio is not a file: {audio_path}")
    if not 1 <= args.max_new_tokens <= 512:
        die("--max-new-tokens must be in 1..=512")
    language = None if args.language.lower() == "auto" else args.language
    require_empty_output(output)
    config = require_model_identity(model_dir, variant)

    # The official model and all reference dependencies must resolve from the
    # pinned uv environment and the already-downloaded exact snapshot.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        import numpy
        import soundfile
        import torch
        import transformers
        from qwen_asr import Qwen3ASRModel
        from qwen_asr.inference.utils import parse_asr_output
    except ImportError as error:
        die(
            "official qwen-asr imports are required; run with "
            f"`uv run --project tools/parity/qwen3_asr --frozen`: {error}"
        )

    package_version = importlib.metadata.version("qwen-asr")
    if package_version != QWEN_ASR_VERSION:
        die(f"qwen-asr={package_version}, expected {QWEN_ASR_VERSION}")
    if transformers.__version__ != TRANSFORMERS_VERSION:
        die(f"transformers={transformers.__version__}, expected {TRANSFORMERS_VERSION}")

    pcm, sample_rate = soundfile.read(str(audio_path), dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE or pcm.shape[1] != 1:
        die(f"audio must be mono 16 kHz, got rate={sample_rate} shape={pcm.shape}")
    pcm = numpy.ascontiguousarray(pcm[:, 0], dtype=numpy.float32)
    if pcm.size == 0 or not numpy.isfinite(pcm).all():
        die("audio is empty or non-finite")

    torch.manual_seed(1234)
    numpy.random.seed(1234)
    torch.set_num_threads(max(1, int(os.environ.get("VOKRA_REFERENCE_TORCH_THREADS", "1"))))
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)

    cpu_capability = getattr(torch.backends.cpu, "get_cpu_capability", None)
    environment = {
        "schema": SCHEMA,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model(),
        "cpu_flags": cpu_flags(),
        "logical_cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "torch_cpu_capability": cpu_capability() if callable(cpu_capability) else "unavailable",
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "qwen_asr": package_version,
        "device": "cpu",
        "dtype": "float32",
    }
    print(json.dumps({"reference_environment": environment}, sort_keys=True), flush=True)

    # Calling the official wrapper also registers its exact custom AutoModel
    # and AutoProcessor classes.  No local layer implementation exists here.
    asr = Qwen3ASRModel.from_pretrained(
        str(model_dir),
        max_inference_batch_size=1,
        max_new_tokens=args.max_new_tokens,
        local_files_only=True,
        dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    if asr.backend != "transformers" or asr.model.device.type != "cpu":
        die(f"official wrapper selected unexpected backend/device: {asr.backend}/{asr.model.device}")
    if asr.model.dtype != torch.float32:
        die(f"official model dtype={asr.model.dtype}, expected torch.float32")

    prompt = asr._build_text_prompt(context=args.context, force_language=language)
    inputs = asr.processor(text=[prompt], audio=[pcm], return_tensors="pt", padding=True)
    inputs = inputs.to(asr.model.device).to(asr.model.dtype)
    prompt_ids = inputs["input_ids"][0].detach().cpu().to(torch.int64)

    with torch.inference_mode():
        audio_embeddings = asr.model.thinker.get_audio_features(
            inputs["input_features"],
            feature_attention_mask=inputs["feature_attention_mask"],
        )
        generated = asr.model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    if audio_embeddings.ndim != 2 or audio_embeddings.shape[1] != variant.hidden_size:
        die(
            f"official audio tap shape={tuple(audio_embeddings.shape)}, "
            f"expected [frames,{variant.hidden_size}]"
        )
    sequences = generated.sequences
    if sequences.ndim != 2 or sequences.shape[0] != 1:
        die(f"official generate returned unexpected sequences shape={tuple(sequences.shape)}")
    generated_ids = sequences[0, prompt_ids.numel() :].detach().cpu().to(torch.int64)
    raw_text = asr.processor.batch_decode(
        generated_ids.unsqueeze(0),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    result_language, result_text = parse_asr_output(raw_text, user_language=language)

    files = {
        "pcm.f32le": pcm,
        "prompt_ids.u32le": prompt_ids.numpy(),
        "audio_embeddings.f32le": audio_embeddings.detach().cpu().float().numpy(),
        "generated_ids.u32le": generated_ids.numpy(),
    }
    write_f32(output / "pcm.f32le", files["pcm.f32le"], numpy)
    write_u32(output / "prompt_ids.u32le", files["prompt_ids.u32le"], numpy)
    write_f32(output / "audio_embeddings.f32le", files["audio_embeddings.f32le"], numpy)
    write_u32(output / "generated_ids.u32le", files["generated_ids.u32le"], numpy)
    (output / "context.txt").write_text(args.context, encoding="utf-8")
    (output / "forced_language.txt").write_text(language or "", encoding="utf-8")
    (output / "raw_text.txt").write_text(raw_text, encoding="utf-8")
    (output / "result_language.txt").write_text(result_language, encoding="utf-8")
    (output / "result_text.txt").write_text(result_text, encoding="utf-8")
    (output / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "source_files.json").write_text(
        json.dumps(source_inventory(model_dir), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    artifact_names = [
        "pcm.f32le",
        "prompt_ids.u32le",
        "audio_embeddings.f32le",
        "generated_ids.u32le",
        "context.txt",
        "forced_language.txt",
        "raw_text.txt",
        "result_language.txt",
        "result_text.txt",
        "environment.json",
        "source_files.json",
    ]
    manifest: dict[str, object] = {
        "schema": SCHEMA,
        "variant": variant.slug,
        "model_name": variant.model_name,
        "upstream_repo": variant.repo,
        "upstream_revision": variant.revision,
        "qwen_asr_version": package_version,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "sample_rate": SAMPLE_RATE,
        "pcm_samples": pcm.size,
        "audio_frames": audio_embeddings.shape[0],
        "hidden_size": audio_embeddings.shape[1],
        "prompt_tokens": prompt_ids.numel(),
        "generated_tokens": generated_ids.numel(),
        "max_new_tokens": args.max_new_tokens,
        "tensor_count": variant.tensor_count,
        "source_config_sha256": sha256_file(model_dir / "config.json"),
        "source_audio_sha256": sha256_file(audio_path),
        "config_model_type": config["model_type"],
    }
    for name in artifact_names:
        path = output / name
        manifest[f"sha256_{name.replace('.', '_')}"] = sha256_file(path)
    write_manifest(output / "manifest.txt", manifest)
    print(
        f"QWEN3_ASR_OFFICIAL_REFERENCE variant={variant.slug} "
        f"audio_shape={tuple(audio_embeddings.shape)} prompt_tokens={prompt_ids.numel()} "
        f"generated_tokens={generated_ids.numel()} output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
