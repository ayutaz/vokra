#!/usr/bin/env python3
"""Dump an independent official NeuTTS Air language-model reference.

The neural oracle is Hugging Face's official ``Qwen2ForCausalLM`` loaded from
the exact Neuphonic snapshot. Prompt construction calls the immutable released
``NeuTTSAir._apply_chat_template`` method directly. No Qwen layer or prompt
template is reimplemented here. The only substituted boundary is phonemization:
the inputs are already IPA strings and ``_to_phones`` is identity, keeping this
gate focused on the learned LM rather than a host eSpeak installation.

The snapshot and source checkout must already be local. Network fallback is
disabled before Transformers is imported. This script is VAST-only and has no
upload or publication path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import sys
import types
from pathlib import Path
from typing import Any


SCHEMA = "vokra-neutts-air-reference-v1"
UPSTREAM_REPO = "neuphonic/neutts-air"
UPSTREAM_REVISION = "3b58b776406b62fdc137e31ea53d728f5c22a4ed"
SOURCE_REVISION = "3e9415df12633f8a74ac6f92418c7cd5c8c4bf0e"
SOURCE_BYTES = 9_035
SOURCE_SHA256 = "e68b87dae6718903337a08eff56afbd58ba261d829624ea5a00a343c8cefb7c1"
TRANSFORMERS_VERSION = "4.57.6"
MODEL_SAFETENSORS_BYTES = 1_495_893_752

TEXT_PROMPT_START = 151_666
TEXT_PROMPT_END = 151_667
SPEECH_GENERATION_START = 151_669
SPEECH_GENERATION_END = 151_670
SPEECH_TOKEN_BASE = 151_671
SPEECH_TOKEN_LAST = 217_206
VOCAB_SIZE = 217_652

REFERENCE_CODES = [0, 1, 2, 3, 7, 31, 255, 1_023]
REFERENCE_PHONES = "həˈloʊ"
TARGET_PHONES = "wɝld"


def die(message: str) -> "None":
    raise SystemExit(f"neutts_air reference: {message}")


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


def require_empty_output(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    entries = list(path.iterdir())
    if entries:
        die(f"--output must be empty, found {entries[0]}")


def require_source(source_file: Path) -> None:
    if not source_file.is_file():
        die(f"missing pinned Neuphonic source: {source_file}")
    actual = (source_file.stat().st_size, sha256_file(source_file))
    expected = (SOURCE_BYTES, SOURCE_SHA256)
    if actual != expected:
        die(f"source identity drift: bytes={actual[0]} sha256={actual[1]}; expected {expected}")


def require_model(model_dir: Path) -> dict[str, Any]:
    revision_stamp = model_dir / ".vokra-source-revision"
    if not revision_stamp.is_file():
        die(f"missing exact-revision stamp: {revision_stamp}")
    revision = revision_stamp.read_text(encoding="utf-8").strip()
    if revision != UPSTREAM_REVISION:
        die(f"snapshot revision={revision!r}, expected {UPSTREAM_REVISION}")
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        die("snapshot must contain config.json and single-file model.safetensors")
    if weights_path.stat().st_size != MODEL_SAFETENSORS_BYTES:
        die(
            f"model.safetensors bytes={weights_path.stat().st_size}, "
            f"expected {MODEL_SAFETENSORS_BYTES}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "model_type": "qwen2",
        "hidden_size": 896,
        "intermediate_size": 4_864,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32_768,
        "vocab_size": VOCAB_SIZE,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            die(f"config {key}={config.get(key)!r}, expected {value!r}")
    if float(config.get("rope_theta", 0.0)) != 1_000_000.0:
        die(f"config rope_theta={config.get('rope_theta')!r}, expected 1000000.0")
    if float(config.get("rms_norm_eps", 0.0)) != 1.0e-6:
        die(f"config rms_norm_eps={config.get('rms_norm_eps')!r}, expected 1e-6")
    return config


def source_inventory(model_dir: Path, source_file: Path) -> dict[str, dict[str, object]]:
    names = {
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
    inventory: dict[str, dict[str, object]] = {}
    for name in sorted(names):
        path = model_dir / name
        if not path.is_file():
            die(f"missing pinned model asset: {path}")
        inventory[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    inventory["neuttsair/neutts.py"] = {
        "bytes": source_file.stat().st_size,
        "sha256": sha256_file(source_file),
    }
    return inventory


def stub_module(name: str, **attributes: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def load_official_release_class(source_file: Path) -> type[Any]:
    # The immutable source imports codec, phonemizer, audio and watermark
    # packages at module scope although `_apply_chat_template` uses none of
    # them. Stubs make only those unused imports resolvable; the method body
    # itself is executed verbatim from the authenticated official file.
    stub_module("librosa")
    stub_module("perth")
    dummy_codec = type("UnusedCodec", (), {})
    stub_module("neucodec", NeuCodec=dummy_codec, DistillNeuCodec=dummy_codec)
    stub_module("phonemizer")
    backend = stub_module("phonemizer.backend", EspeakBackend=type("UnusedEspeak", (), {}))
    sys.modules["phonemizer"].backend = backend

    spec = importlib.util.spec_from_file_location("vokra_official_neutts_air", source_file)
    if spec is None or spec.loader is None:
        die(f"cannot import official source: {source_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    release_class = getattr(module, "NeuTTSAir", None)
    if not isinstance(release_class, type):
        die("official source does not define NeuTTSAir")
    return release_class


def official_prompt_ids(release_class: type[Any], tokenizer: Any) -> list[int]:
    instance = object.__new__(release_class)
    instance.tokenizer = tokenizer
    instance._to_phones = types.MethodType(lambda _self, text: text, instance)
    ids = release_class._apply_chat_template(
        instance,
        REFERENCE_CODES,
        REFERENCE_PHONES,
        TARGET_PHONES,
    )
    if not isinstance(ids, list) or not ids or not all(isinstance(value, int) for value in ids):
        die(f"official prompt returned invalid token list: {type(ids).__name__}")
    if ids.count(SPEECH_GENERATION_START) != 1:
        die("official prompt does not contain exactly one speech-generation-start token")
    start = ids.index(SPEECH_GENERATION_START)
    expected_suffix = [SPEECH_TOKEN_BASE + code for code in REFERENCE_CODES]
    if ids[start + 1 :] != expected_suffix:
        die(f"official reference-code suffix drift: {ids[start + 1 :]} != {expected_suffix}")
    if TEXT_PROMPT_START not in ids or TEXT_PROMPT_END not in ids:
        die("official prompt is missing text prompt delimiters")
    return ids


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


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_dir = args.model_dir.resolve()
    source_file = args.source_file.resolve()
    output = args.output.resolve()
    if not model_dir.is_dir():
        die(f"--model-dir is not a directory: {model_dir}")
    if not 1 <= args.max_new_tokens <= 32:
        die("--max-new-tokens must be in 1..=32")
    require_source(source_file)
    config = require_model(model_dir)
    require_empty_output(output)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        import numpy
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        die(
            "official Transformers dependencies are required; run with "
            f"`uv run --project tools/parity/neutts_air --frozen`: {error}"
        )
    if transformers.__version__ != TRANSFORMERS_VERSION:
        die(f"transformers={transformers.__version__}, expected {TRANSFORMERS_VERSION}")

    torch.manual_seed(1_234)
    numpy.random.seed(1_234)
    torch.set_num_threads(max(1, int(os.environ.get("VOKRA_REFERENCE_TORCH_THREADS", "1"))))
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    token_checks = {
        "<|TEXT_PROMPT_START|>": TEXT_PROMPT_START,
        "<|TEXT_PROMPT_END|>": TEXT_PROMPT_END,
        "<|SPEECH_GENERATION_START|>": SPEECH_GENERATION_START,
        "<|SPEECH_GENERATION_END|>": SPEECH_GENERATION_END,
        "<|speech_0|>": SPEECH_TOKEN_BASE,
        "<|speech_65535|>": SPEECH_TOKEN_LAST,
    }
    for token, expected_id in token_checks.items():
        actual_id = tokenizer.convert_tokens_to_ids(token)
        if actual_id != expected_id:
            die(f"official tokenizer maps {token!r} to {actual_id}, expected {expected_id}")

    release_class = load_official_release_class(source_file)
    prompt_ids = official_prompt_ids(release_class, tokenizer)
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device="cpu")

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    if model.device.type != "cpu" or model.dtype != torch.float32:
        die(f"official model selected unexpected device/dtype: {model.device}/{model.dtype}")
    with torch.inference_mode():
        output_values = model(input_ids=prompt, use_cache=False, return_dict=True)
        next_logits = output_values.logits[0, -1].detach().cpu().float()
        generated = model.generate(
            input_ids=prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
            eos_token_id=SPEECH_GENERATION_END,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    if next_logits.shape != (VOCAB_SIZE,):
        die(f"official next logits shape={tuple(next_logits.shape)}, expected ({VOCAB_SIZE},)")
    generated_ids = generated[0, len(prompt_ids) :].detach().cpu().to(torch.int64)
    if generated_ids.numel() == 0:
        die("official greedy generation returned no token")

    write_u32(output / "prompt_ids.u32le", prompt_ids, numpy)
    write_f32(output / "next_logits.f32le", next_logits.numpy(), numpy)
    write_u32(output / "generated_ids.u32le", generated_ids.numpy(), numpy)
    inventory = source_inventory(model_dir, source_file)
    (output / "source_files.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    environment = {
        "schema": SCHEMA,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "device": "cpu",
        "dtype": "float32",
    }
    (output / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    artifacts = [
        "prompt_ids.u32le",
        "next_logits.f32le",
        "generated_ids.u32le",
        "source_files.json",
        "environment.json",
    ]
    manifest: dict[str, object] = {
        "schema": SCHEMA,
        "upstream_repo": UPSTREAM_REPO,
        "upstream_revision": UPSTREAM_REVISION,
        "source_revision": SOURCE_REVISION,
        "source_sha256": SOURCE_SHA256,
        "source_config_sha256": sha256_file(model_dir / "config.json"),
        "source_weights_sha256": sha256_file(model_dir / "model.safetensors"),
        "source_tokenizer_sha256": sha256_file(model_dir / "tokenizer.json"),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "vocab_size": VOCAB_SIZE,
        "prompt_tokens": len(prompt_ids),
        "prompt_ids_csv": ",".join(str(token) for token in prompt_ids),
        "generated_tokens": generated_ids.numel(),
        "max_new_tokens": args.max_new_tokens,
        "reference_codes": ",".join(str(code) for code in REFERENCE_CODES),
        "config_model_type": config["model_type"],
    }
    for name in artifacts:
        manifest[f"sha256_{name.replace('.', '_')}"] = sha256_file(output / name)
    write_manifest(output / "manifest.txt", manifest)
    print(
        f"NEUTTS_AIR_OFFICIAL_REFERENCE prompt_tokens={len(prompt_ids)} "
        f"generated_tokens={generated_ids.numel()} logits={next_logits.numel()} output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
