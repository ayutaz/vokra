#!/usr/bin/env python3
"""Dump an independent official MOSS-Audio real-checkpoint reference.

The oracle imports ``MossAudioModel`` and ``MossAudioProcessor`` from the
immutable official OpenMOSS source checkout supplied through ``--source-dir``.
It does not mirror an encoder, adapter, Qwen layer, prompt builder or decoder.
The four audio taps are direct outputs of ``get_audio_features``,
``audio_adapter`` and ``deepstack_audio_merger_list``; generation and text
decode call the official model and processor entry points.

Both checkpoints exceed the repository's local artifact threshold. Run only
through ``scripts/publish/vast-ai/run-moss-audio-validation.sh``. The script
never downloads, uploads or publishes and refuses network fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SOURCE_CODE_REVISION = "5cbb1d823937cd5b5de3d8fa4d3a7253ebd3b883"
CONFIGURATION_SOURCE_SHA256 = (
    "e597dca441ff7fb58a5ec43186fafdfce19f31dada4955b4910059baa5d52ebd"
)
MODELING_SOURCE_SHA256 = (
    "a52513e518c68a0ba7c636a1ab0e12f7755ceebd0ae033235dc5e2551bfcbf9c"
)
PROCESSING_SOURCE_SHA256 = (
    "05fb788cbdc6482eded8d70f7d2f524bc0cdca47d001acab5661c11f02cc6fe6"
)
REFERENCE_AUDIO_SHA256 = (
    "241c0d93cc7ed8792c85c525d1e02b8c33850b791902a5e75b79c2d500e71a1a"
)
SAMPLE_RATE = 16_000
SCHEMA = "vokra-moss-audio-reference-v1"
DEFAULT_PROMPT = "Describe this audio."
TRANSFORMERS_VERSION = "4.57.1"


@dataclass(frozen=True)
class Variant:
    slug: str
    repo: str
    revision: str
    model_name: str
    hidden_size: int
    intermediate_size: int
    tensor_count: int
    config_sha256: str
    tokenizer_config_bytes: int
    tokenizer_config_sha256: str
    processor_config_bytes: int
    processor_config_sha256: str


VARIANTS = {
    "4b": Variant(
        slug="4b",
        repo="OpenMOSS-Team/MOSS-Audio-4B-Instruct",
        revision="6907a499dc0e87cc77c8ae0fe23fd0eb5476a02d",
        model_name="moss-audio-4b-instruct",
        hidden_size=2_560,
        intermediate_size=9_728,
        tensor_count=901,
        config_sha256=(
            "e528a941446f4443f1b9fede12ea484e58a79d494c28d21ef1e73b5148abfbfa"
        ),
        tokenizer_config_bytes=5_404,
        tokenizer_config_sha256=(
            "443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0"
        ),
        processor_config_bytes=426,
        processor_config_sha256=(
            "0749d81701d2a2a2e83ca4d549fbebb1a205acac1ac7bdccea7965c1913b2cbf"
        ),
    ),
    "8b": Variant(
        slug="8b",
        repo="OpenMOSS-Team/MOSS-Audio-8B-Instruct",
        revision="6521a39181b47a18f2d9f4b3acfb5bca7b76b57f",
        model_name="moss-audio-8b-instruct",
        hidden_size=4_096,
        intermediate_size=12_288,
        tensor_count=901,
        config_sha256=(
            "535154c2a5bcbd0e18e2f92bcf370ac74b530eec97ad4fd9317993ba0a316536"
        ),
        tokenizer_config_bytes=6_114,
        tokenizer_config_sha256=(
            "0869e41f5d123ff144a811f0d83c5d18871dcd4b4064f46bf9def194bfbc6f41"
        ),
        processor_config_bytes=427,
        processor_config_sha256=(
            "6a5c462858acb299db0d2d967b63d520b72d178f44d1619c33fc860f25fdccbf"
        ),
    ),
}

COMMON_ASSETS = {
    "vocab.json": (
        3_383_407,
        "87a257b04b17642a0688c98cd1df89c398bda4fee532d6f88b38a659ecb4ac8d",
    ),
    "merges.txt": (
        1_671_853,
        "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    ),
    "chat_template.jinja": (
        4_116,
        "87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5",
    ),
    "generation_config.json": (
        121,
        "bb52bfdd308deaea4ec800bf0165e75770b0a4e5c105963bee1b0398f4043d3e",
    ),
}

SOURCE_FILES = {
    "src/configuration_moss_audio.py": CONFIGURATION_SOURCE_SHA256,
    "src/modeling_moss_audio.py": MODELING_SOURCE_SHA256,
    "src/processing_moss_audio.py": PROCESSING_SOURCE_SHA256,
}


def die(message: str) -> "None":
    raise SystemExit(f"moss_audio reference: {message}")


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
    lines = [
        f"{key}={safe_manifest_value(value)}"
        for key, value in sorted(values.items())
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cpu_model() -> str:
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def cpu_flags() -> str:
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if re.match(r"^(flags|features)\s*:", line, flags=re.IGNORECASE):
                return line.split(":", 1)[1].strip()
    return "unknown"


def require_empty_output(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    entries = list(path.iterdir())
    if entries:
        die(f"--output must be empty, found {entries[0]}")


def require_exact_file(path: Path, expected_bytes: int, expected_hash: str) -> None:
    if not path.is_file():
        die(f"missing pinned file: {path}")
    actual = (path.stat().st_size, sha256_file(path))
    expected = (expected_bytes, expected_hash)
    if actual != expected:
        die(
            f"identity drift for {path.name}: bytes={actual[0]} sha256={actual[1]}; "
            f"expected bytes={expected[0]} sha256={expected[1]}"
        )


def source_revision(source_dir: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def require_source_identity(source_dir: Path) -> dict[str, str]:
    actual_revision = source_revision(source_dir)
    if actual_revision != SOURCE_CODE_REVISION:
        die(
            f"official source checkout is not {SOURCE_CODE_REVISION}: "
            f"{actual_revision}"
        )
    inventory: dict[str, str] = {}
    for relative, expected_hash in SOURCE_FILES.items():
        path = (source_dir / relative).resolve()
        if not path.is_file():
            die(f"missing official source file: {path}")
        actual = sha256_file(path)
        if actual != expected_hash:
            die(f"official source hash drift for {relative}: {actual} != {expected_hash}")
        inventory[relative] = actual
    return inventory


def require_model_identity(model_dir: Path, variant: Variant) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        die(f"missing pinned file: {config_path}")
    config_hash = sha256_file(config_path)
    if config_hash != variant.config_sha256:
        die(
            f"identity drift for config.json: sha256={config_hash}; "
            f"expected sha256={variant.config_sha256}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "moss_audio":
        die(f"config model_type={config.get('model_type')!r}, expected 'moss_audio'")
    if config.get("architectures") != ["MossAudioModel"]:
        die(f"config architectures={config.get('architectures')!r}")
    audio = config.get("audio_config")
    language = config.get("language_config")
    if not isinstance(audio, dict) or not isinstance(language, dict):
        die("config lacks audio_config/language_config objects")
    expected_audio = {
        "d_model": 1_280,
        "output_dim": 1_280,
        "num_mel_bins": 128,
        "encoder_layers": 32,
        "encoder_attention_heads": 20,
        "encoder_ffn_dim": 5_120,
        "downsample_rate": 8,
        "downsample_hidden_size": 480,
        "encoder_attention_window_size": 100,
        "max_source_positions": 1_500,
        "activation_function": "gelu",
        "n_window": 200,
        "deepstack_encoder_layer_indexes": [8, 16, 24],
    }
    for key, expected in expected_audio.items():
        if audio.get(key) != expected:
            die(f"config audio_config.{key}={audio.get(key)!r}, expected {expected!r}")
    expected_language = {
        "hidden_size": variant.hidden_size,
        "intermediate_size": variant.intermediate_size,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151_936,
        "max_position_embeddings": 40_960,
        "rope_theta": 1_000_000.0,
        "rms_norm_eps": 1.0e-6,
        "attention_bias": False,
    }
    for key, expected in expected_language.items():
        if language.get(key) != expected:
            die(f"config language_config.{key}={language.get(key)!r}, expected {expected!r}")
    if config.get("adapter_hidden_size") != 8_192:
        die(f"config adapter_hidden_size={config.get('adapter_hidden_size')!r}")
    if config.get("deepstack_num_inject_layers") != 3:
        die(f"config deepstack_num_inject_layers={config.get('deepstack_num_inject_layers')!r}")

    assets = dict(COMMON_ASSETS)
    assets["tokenizer_config.json"] = (
        variant.tokenizer_config_bytes,
        variant.tokenizer_config_sha256,
    )
    assets["processor_config.json"] = (
        variant.processor_config_bytes,
        variant.processor_config_sha256,
    )
    for name, (expected_bytes, expected_hash) in assets.items():
        require_exact_file(model_dir / name, expected_bytes, expected_hash)

    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        die(f"missing sharded checkpoint index: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or len(weight_map) != variant.tensor_count:
        die(
            f"checkpoint index has {len(weight_map) if isinstance(weight_map, dict) else 'invalid'} "
            f"tensor entries, expected {variant.tensor_count}"
        )
    return config


def require_import_source(obj: object, expected: Path, inspect_module: object) -> None:
    source = inspect_module.getsourcefile(obj)
    if source is None:
        die(f"{obj!r} has no inspectable source")
    actual = Path(source).resolve()
    if actual != expected.resolve():
        die(f"official class came from {actual}, expected {expected.resolve()}")


def write_f32(path: Path, tensor: object, numpy_module: object) -> None:
    array = tensor.detach().cpu().float().contiguous().numpy()
    if not bool(numpy_module.isfinite(array).all()):
        die(f"{path.name} contains non-finite values")
    array.astype("<f4", copy=False).tofile(path)


def write_pcm_f32(path: Path, array: object, numpy_module: object) -> None:
    value = numpy_module.asarray(array, dtype=numpy_module.float32)
    if value.ndim != 1 or not bool(numpy_module.isfinite(value).all()):
        die("reference PCM must be finite mono rank-1 float32")
    value.astype("<f4", copy=False).tofile(path)


def write_u32(path: Path, tensor: object, numpy_module: object) -> None:
    array = tensor.detach().cpu().contiguous().numpy()
    if not bool(numpy_module.issubdtype(array.dtype, numpy_module.integer)):
        die(f"{path.name} is not an integer tensor: dtype={array.dtype}")
    if array.size and (int(array.min()) < 0 or int(array.max()) > 0xFFFF_FFFF):
        die(f"{path.name} contains an id outside u32")
    array.astype("<u4", copy=False).tofile(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    variant = VARIANTS[args.variant]
    model_dir = args.model_dir.resolve()
    source_dir = args.source_dir.resolve()
    audio_path = args.audio.resolve()
    output = args.output.resolve()
    if args.prompt != DEFAULT_PROMPT:
        die(f"parity prompt must remain the official example {DEFAULT_PROMPT!r}")
    if not 1 <= args.max_new_tokens <= 16:
        die("--max-new-tokens must be in 1..=16")
    if not model_dir.is_dir() or not source_dir.is_dir():
        die("--model-dir and --source-dir must be existing directories")
    if not audio_path.is_file():
        die(f"missing reference audio: {audio_path}")
    if sha256_file(audio_path) != REFERENCE_AUDIO_SHA256:
        die(f"reference audio SHA-256 drift: {sha256_file(audio_path)}")
    require_empty_output(output)
    source_inventory = require_source_identity(source_dir)
    config_json = require_model_identity(model_dir, variant)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    source_text = str(source_dir)
    if source_text in sys.path:
        die("official source path was already present before oracle setup")
    sys.path.insert(0, source_text)

    import inspect
    import numpy
    import soundfile
    import torch
    import transformers
    from src.configuration_moss_audio import MossAudioConfig
    from src.modeling_moss_audio import MossAudioModel
    from src.processing_moss_audio import MossAudioProcessor

    if transformers.__version__ != TRANSFORMERS_VERSION:
        die(
            f"transformers={transformers.__version__}, expected {TRANSFORMERS_VERSION}"
        )
    if not torch.__version__.startswith("2.9.1"):
        die(f"torch={torch.__version__}, expected the pinned 2.9.1 line")
    require_import_source(
        MossAudioConfig,
        source_dir / "src/configuration_moss_audio.py",
        inspect,
    )
    require_import_source(
        MossAudioModel,
        source_dir / "src/modeling_moss_audio.py",
        inspect,
    )
    require_import_source(
        MossAudioProcessor,
        source_dir / "src/processing_moss_audio.py",
        inspect,
    )

    thread_count = int(os.environ.get("VOKRA_REFERENCE_TORCH_THREADS", "8"))
    if not 1 <= thread_count <= 64:
        die("VOKRA_REFERENCE_TORCH_THREADS must be in 1..=64")
    torch.manual_seed(0)
    torch.set_num_threads(thread_count)
    torch.set_num_interop_threads(1)
    capability = getattr(torch.backends.cpu, "get_cpu_capability", None)
    environment = {
        "cpu_model": cpu_model(),
        "cpu_flags": cpu_flags(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "torch_cpu_capability": capability() if callable(capability) else "unavailable",
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": "cpu",
        "dtype": "float32",
    }
    print(json.dumps({"reference_environment": environment}, sort_keys=True), flush=True)

    pcm, sample_rate = soundfile.read(
        audio_path, dtype="float32", always_2d=False
    )
    if sample_rate != SAMPLE_RATE or pcm.ndim != 1:
        die(
            f"reference audio is shape={pcm.shape} rate={sample_rate}; "
            f"expected mono {SAMPLE_RATE} Hz"
        )

    processor = MossAudioProcessor.from_pretrained(
        str(model_dir),
        local_files_only=True,
        enable_time_marker=True,
    )
    model = MossAudioModel.from_pretrained(
        str(model_dir),
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    first_parameter = next(model.parameters())
    if first_parameter.device.type != "cpu" or first_parameter.dtype != torch.float32:
        die(
            f"official model loaded on {first_parameter.device}/{first_parameter.dtype}, "
            "expected CPU/float32"
        )
    if model.config.language_config.hidden_size != variant.hidden_size:
        die("loaded official model hidden width drifted after config construction")

    inputs = processor(
        text=args.prompt,
        audios=[pcm],
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    inputs["audio_data"] = inputs["audio_data"].to(dtype=model.dtype)
    audio_input_mask = inputs["input_ids"] == processor.audio_token_id
    inputs["audio_input_mask"] = audio_input_mask
    prompt_ids = inputs["input_ids"][0].detach().cpu().to(torch.int64)

    with torch.inference_mode():
        encoder_audio, encoder_deepstack = model.get_audio_features(
            inputs["audio_data"], inputs["audio_data_seqlens"]
        )
        primary_audio = model.audio_adapter(encoder_audio)
        if encoder_deepstack is None or len(encoder_deepstack) != 3:
            die(f"official encoder returned {encoder_deepstack!r} DeepStack taps")
        if len(model.deepstack_audio_merger_list) != 3:
            die(
                "official model exposes "
                f"{len(model.deepstack_audio_merger_list)} DeepStack adapters"
            )
        deepstack_audio = [
            adapter(values)
            for adapter, values in zip(
                model.deepstack_audio_merger_list, encoder_deepstack, strict=True
            )
        ]
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )

    expected_shape = (1, int(primary_audio.shape[1]), variant.hidden_size)
    if tuple(primary_audio.shape) != expected_shape:
        die(
            f"official primary adapter shape={tuple(primary_audio.shape)}, "
            f"expected {expected_shape}"
        )
    for index, values in enumerate(deepstack_audio):
        if tuple(values.shape) != expected_shape:
            die(f"official DeepStack {index} shape={tuple(values.shape)} != {expected_shape}")
    audio_frames = int(primary_audio.shape[1])
    if int(audio_input_mask.to(torch.int64).sum().item()) != audio_frames:
        die("official prompt audio placeholder count differs from adapter rows")

    sequences = generated.sequences if hasattr(generated, "sequences") else generated
    if sequences.ndim != 2 or sequences.shape[0] != 1:
        die(f"official generate returned shape={tuple(sequences.shape)}")
    generated_ids = sequences[0, prompt_ids.numel() :].detach().cpu().to(torch.int64)
    result_text = processor.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    write_pcm_f32(output / "pcm.f32le", pcm, numpy)
    write_u32(output / "prompt_ids.u32le", prompt_ids, numpy)
    write_f32(output / "primary_audio.f32le", primary_audio, numpy)
    for index, values in enumerate(deepstack_audio):
        write_f32(output / f"deepstack_audio_{index}.f32le", values, numpy)
    write_u32(output / "generated_ids.u32le", generated_ids, numpy)
    (output / "prompt.txt").write_text(args.prompt, encoding="utf-8")
    (output / "result_text.txt").write_text(result_text, encoding="utf-8")
    (output / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "source_files.json").write_text(
        json.dumps(source_inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    artifact_names = [
        "pcm.f32le",
        "prompt_ids.u32le",
        "primary_audio.f32le",
        "deepstack_audio_0.f32le",
        "deepstack_audio_1.f32le",
        "deepstack_audio_2.f32le",
        "generated_ids.u32le",
        "prompt.txt",
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
        "source_code_revision": SOURCE_CODE_REVISION,
        "configuration_source_sha256": CONFIGURATION_SOURCE_SHA256,
        "modeling_source_sha256": MODELING_SOURCE_SHA256,
        "processing_source_sha256": PROCESSING_SOURCE_SHA256,
        "config_sha256": variant.config_sha256,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "sample_rate": SAMPLE_RATE,
        "pcm_samples": int(pcm.size),
        "audio_frames": audio_frames,
        "hidden_size": variant.hidden_size,
        "prompt_tokens": int(prompt_ids.numel()),
        "generated_tokens": int(generated_ids.numel()),
        "max_new_tokens": args.max_new_tokens,
        "tensor_count": variant.tensor_count,
        "config_model_type": config_json["model_type"],
        "source_audio_sha256": REFERENCE_AUDIO_SHA256,
    }
    for name in artifact_names:
        manifest[f"sha256_{name.replace('.', '_')}"] = sha256_file(output / name)
    write_manifest(output / "manifest.txt", manifest)
    print(
        f"MOSS_AUDIO_OFFICIAL_REFERENCE variant={variant.slug} "
        f"audio_frames={audio_frames} prompt_tokens={prompt_ids.numel()} "
        f"generated_tokens={generated_ids.numel()} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
