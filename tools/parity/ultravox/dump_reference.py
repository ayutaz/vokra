#!/usr/bin/env python3
"""Dump an independent official Ultravox v0.5 FP32 reference.

The oracle imports Fixie's authenticated Hugging Face custom code and executes
its released UltravoxModel, ModifiedWhisperEncoder, UltravoxProjector,
UltravoxProcessor, tokenizer chat template, and Llama language model. The
official projector is hooked only to capture its returned tensor during the
official forward. No Vokra module is imported and no Ultravox layer is
reimplemented. Both fixed snapshots must already be local; network fallback is
disabled before Transformers loads.

This script is VAST-only. It has no download, upload, or publication path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any


SCHEMA = "vokra-ultravox-reference-v1"
UPSTREAM_REPO = "fixie-ai/ultravox-v0_5-llama-3_2-1b"
UPSTREAM_REVISION = "b95bec8ab291eeb04b5cd600dd473377f6b79026"
COMPANION_REPO = "meta-llama/Llama-3.2-1B-Instruct"
COMPANION_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
PUBLIC_REPO = "vokra/ultravox-v0-5-llama-3-2-1b"
PUBLIC_REVISION = "ddbbeec5bfcb09c71a1f88971b794e3e5da811f9"
PUBLIC_FILENAME = "ultravox-v0-5-llama-3-2-1b.gguf"
PUBLIC_FILE_BYTES = 1_366_275_264
PUBLIC_FILE_SHA256 = "376c79a7219bb38fc6a857b0bd9ccf57daff878e7bb4723c4801000c0d7b8c9c"
TRANSFORMERS_VERSION = "5.5.0"
SAMPLE_RATE = 16_000
SAMPLE_COUNT = 16_000
N_MELS = 128
STACK_FACTOR = 8
ENCODER_DS_FACTOR = 2
TEXT_HIDDEN_SIZE = 2_048
VOCAB_SIZE = 128_256

CUSTOM_CODE: dict[str, tuple[int, str]] = {
    "ultravox_config.py": (
        7_057,
        "99cf5ad911189f2351c2232234025db56b23763283583c0a848ebf2a1ecc40fc",
    ),
    "ultravox_model.py": (
        41_578,
        "df618218561375da01bb53bd2764ea123e0cbf782f3326753f669f63ff6c6d3f",
    ),
    "ultravox_processing.py": (
        17_087,
        "2ae6682f3deecb22539fae6a6631688fc1675282f1a5b31145d9f95d2347ff7b",
    ),
}


def die(message: str) -> "None":
    raise SystemExit(f"ultravox reference: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if not path.is_file():
        die(f"missing pinned file: {path}")
    actual = (path.stat().st_size, sha256_file(path))
    expected = (expected_bytes, expected_sha256)
    if actual != expected:
        die(
            f"identity drift for {path.name}: bytes={actual[0]} sha256={actual[1]}; "
            f"expected bytes={expected[0]} sha256={expected[1]}"
        )


def require_revision(directory: Path, expected: str, label: str) -> None:
    stamp = directory / ".vokra-source-revision"
    if not stamp.is_file():
        die(f"{label} has no exact-revision stamp: {stamp}")
    actual = stamp.read_text(encoding="utf-8").strip()
    if actual != expected:
        die(f"{label} revision={actual!r}, expected {expected}")


def require_snapshot_inventory(
    directory: Path,
    expected_repo: str,
    expected_revision: str,
    expected_files: set[str],
    label: str,
) -> dict[str, Any]:
    """Verify the authenticated downloader's exact input closure before import."""
    path = directory / ".vokra-source-inventory.json"
    if not path.is_file():
        die(f"{label} has no authenticated input inventory: {path}")
    try:
        inventory = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        die(f"{label} input inventory is invalid: {error}")
    if inventory.get("repo") != expected_repo:
        die(f"{label} inventory repo={inventory.get('repo')!r}, expected {expected_repo!r}")
    if inventory.get("revision") != expected_revision:
        die(
            f"{label} inventory revision={inventory.get('revision')!r}, "
            f"expected {expected_revision}"
        )
    files = inventory.get("files")
    if not isinstance(files, dict) or set(files) != expected_files:
        die(f"{label} inventory file closure is not exact: {files!r}")
    for name, record in files.items():
        local = record.get("local") if isinstance(record, dict) else None
        remote = record.get("remote") if isinstance(record, dict) else None
        if not isinstance(local, dict) or not isinstance(remote, dict):
            die(f"{label} inventory entry {name!r} is incomplete")
        verify_file(
            directory / name,
            int(local.get("bytes", -1)),
            str(local.get("sha256", "")),
        )
        if not isinstance(remote.get("size"), int) or remote["size"] != local["bytes"]:
            die(f"{label} remote/local size mismatch for {name}")
        if name == "model.safetensors":
            remote_sha = remote.get("lfs_sha256")
            if not isinstance(remote_sha, str) or len(remote_sha) != 64:
                die(f"{label} model.safetensors lacks authenticated LFS SHA-256")
            if remote_sha != local["sha256"]:
                die(f"{label} model.safetensors differs from authenticated LFS SHA-256")
        else:
            blob_id = remote.get("blob_id")
            if not isinstance(blob_id, str) or len(blob_id) != 40:
                die(f"{label} {name} lacks authenticated Git blob identity")
    return inventory


def require_public_snapshot(directory: Path) -> dict[str, Any]:
    require_revision(directory, UPSTREAM_REVISION, "Ultravox snapshot")
    for name, (size, digest) in CUSTOM_CODE.items():
        verify_file(directory / name, size, digest)
    required = [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    require_snapshot_inventory(
        directory,
        UPSTREAM_REPO,
        UPSTREAM_REVISION,
        set(required + list(CUSTOM_CODE)),
        "Ultravox snapshot",
    )
    verify_file(
        directory / "model.safetensors",
        PUBLIC_FILE_BYTES,
        "f3a3bf7e9137f3219a0d27ba71668deeee8c60aaf0ea587b48d8f71178763f31",
    )
    for name in required:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            die(f"Ultravox snapshot is missing non-empty {name}")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    audio = config.get("audio_config", {})
    expected = {
        "model_type": "ultravox",
        "text_model_id": COMPANION_REPO,
        "hidden_size": 4_096,
        "stack_factor": STACK_FACTOR,
        "projector_act": "swiglu",
        "projector_ln_mid": True,
        "vocab_size": VOCAB_SIZE,
        "audio_model_id": None,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            die(f"Ultravox config {key}={config.get(key)!r}, expected {value!r}")
    audio_expected = {
        "model_type": "whisper",
        "d_model": 1_280,
        "encoder_layers": 32,
        "encoder_attention_heads": 20,
        "encoder_ffn_dim": 5_120,
        "max_source_positions": 1_500,
        "num_mel_bins": N_MELS,
    }
    for key, value in audio_expected.items():
        if audio.get(key) != value:
            die(f"Ultravox audio config {key}={audio.get(key)!r}, expected {value!r}")
    return config


def require_companion_snapshot(directory: Path) -> dict[str, Any]:
    require_revision(directory, COMPANION_REVISION, "Llama companion snapshot")
    require_snapshot_inventory(
        directory,
        COMPANION_REPO,
        COMPANION_REVISION,
        {"config.json", "model.safetensors"},
        "Llama companion snapshot",
    )
    for name in ["config.json", "model.safetensors"]:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            die(f"Llama companion snapshot is missing non-empty {name}")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    expected = {
        "model_type": "llama",
        "hidden_size": TEXT_HIDDEN_SIZE,
        "intermediate_size": 8_192,
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "max_position_embeddings": 131_072,
        "vocab_size": VOCAB_SIZE,
        "tie_word_embeddings": True,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            die(f"Llama config {key}={config.get(key)!r}, expected {value!r}")
    if float(config.get("rope_theta", 0.0)) != 500_000.0:
        die(f"Llama rope_theta={config.get('rope_theta')!r}, expected 500000")
    return config


def require_empty_output(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    entries = list(path.iterdir())
    if entries:
        die(f"--output must be empty, found {entries[0]}")


def safe_manifest_value(value: object) -> str:
    text = str(value)
    if "\n" in text or "\r" in text or "=" in text:
        raise ValueError(f"unsafe manifest value {text!r}")
    return text


def write_manifest(path: Path, values: dict[str, object]) -> None:
    lines = [f"{key}={safe_manifest_value(value)}" for key, value in sorted(values.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_f32(path: Path, values: Any, numpy: Any) -> None:
    array = numpy.asarray(values, dtype=numpy.float32)
    if not numpy.isfinite(array).all():
        die(f"non-finite values in {path.name}")
    path.write_bytes(array.astype("<f4", copy=False).tobytes(order="C"))


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


def source_inventory(*directories: Path) -> dict[str, dict[str, object]]:
    inventory: dict[str, dict[str, object]] = {}
    for directory in directories:
        prefix = "ultravox" if directory == directories[0] else "llama"
        for path in sorted(item for item in directory.iterdir() if item.is_file()):
            if path.name == ".vokra-source-revision":
                continue
            inventory[f"{prefix}/{path.name}"] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return inventory


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ultravox-dir", type=Path)
    parser.add_argument("--companion-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def verify_reference_manifest(directory: Path) -> None:
    """Recheck every emitted artifact hash, including the source inventory."""
    manifest_path = directory / "manifest.txt"
    if not manifest_path.is_file():
        die(f"reference manifest is missing: {manifest_path}")
    values: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or key in values:
            die(f"reference manifest is malformed near {line!r}")
        values[key] = value
    artifacts = [
        "pcm.f32le",
        "input_features.f32le",
        "audio_embeddings.f32le",
        "prompt_ids.u32le",
        "next_logits.f32le",
        "generated_ids.u32le",
        "source_files.json",
        "environment.json",
    ]
    for name in artifacts:
        expected = values.get(f"sha256_{name.replace('.', '_')}")
        if not expected:
            die(f"reference manifest lacks hash for {name}")
        actual = sha256_file(directory / name)
        if actual != expected:
            die(f"reference artifact {name} differs from its manifest hash")


def self_test() -> int:
    """Offline tamper checks for revision, source hash, and reference hashes."""
    with tempfile.TemporaryDirectory(prefix="vokra-ultravox-reference-") as raw:
        directory = Path(raw)
        payload = directory / "payload"
        payload.write_bytes(b"abc")
        verify_file(payload, 3, hashlib.sha256(b"abc").hexdigest())
        try:
            verify_file(payload, 3, hashlib.sha256(b"tampered").hexdigest())
        except SystemExit:
            pass
        else:
            die("self-test accepted a tampered source hash")
        stamp = directory / ".vokra-source-revision"
        stamp.write_text(UPSTREAM_REVISION + "\n", encoding="utf-8")
        require_revision(directory, UPSTREAM_REVISION, "self-test snapshot")
        stamp.write_text("0" * 40 + "\n", encoding="utf-8")
        try:
            require_revision(directory, UPSTREAM_REVISION, "self-test snapshot")
        except SystemExit:
            pass
        else:
            die("self-test accepted a tampered revision")
        artifacts = [
            "pcm.f32le",
            "input_features.f32le",
            "audio_embeddings.f32le",
            "prompt_ids.u32le",
            "next_logits.f32le",
            "generated_ids.u32le",
            "source_files.json",
            "environment.json",
        ]
        for name in artifacts:
            (directory / name).write_bytes(b"\x00\x00\x00\x00")
        manifest = directory / "manifest.txt"
        manifest.write_text(
            "".join(
                f"sha256_{name.replace('.', '_')}={sha256_file(directory / name)}\n"
                for name in artifacts
            ),
            encoding="utf-8",
        )
        verify_reference_manifest(directory)
        (directory / "pcm.f32le").write_bytes(b"\x01\x00\x00\x00")
        try:
            verify_reference_manifest(directory)
        except SystemExit:
            pass
        else:
            die("self-test accepted a tampered reference hash")
    print("dump_reference.py self-test: PASS", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        if args.ultravox_dir or args.companion_dir or args.output:
            die("--self-test accepts no snapshot/output arguments")
        return self_test()
    if not args.ultravox_dir or not args.companion_dir or not args.output:
        die("--ultravox-dir, --companion-dir, and --output are required")
    ultravox_dir = args.ultravox_dir.resolve()
    companion_dir = args.companion_dir.resolve()
    output = args.output.resolve()
    if not ultravox_dir.is_dir() or not companion_dir.is_dir():
        die("--ultravox-dir and --companion-dir must be existing directories")
    if not 1 <= args.max_new_tokens <= 16:
        die("--max-new-tokens must be in 1..=16")
    public_config = require_public_snapshot(ultravox_dir)
    companion_config = require_companion_snapshot(companion_dir)
    require_empty_output(output)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        import numpy
        import torch
        import transformers
        from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
    except ImportError as error:
        die(
            "official dependencies are required; run with "
            f"`uv run --project tools/parity/ultravox --frozen`: {error}"
        )
    if transformers.__version__ != TRANSFORMERS_VERSION:
        die(f"transformers={transformers.__version__}, expected {TRANSFORMERS_VERSION}")

    torch.manual_seed(1_234)
    numpy.random.seed(1_234)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, int(os.environ.get("VOKRA_REFERENCE_TORCH_THREADS", "1"))))
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)

    config_class = get_class_from_dynamic_module(
        "ultravox_config.UltravoxConfig",
        str(ultravox_dir),
        local_files_only=True,
    )
    config_values = dict(public_config)
    # The released config carries a remote text_model_id and its constructor
    # immediately resolves that ID. Build the same official class from the
    # authenticated local Llama config first, then point model construction at
    # the fixed local snapshot. No config or weight may resolve over network.
    config_values["text_model_id"] = None
    config_values["text_config"] = companion_config
    config_values["audio_model_id"] = None
    config = config_class(**config_values)
    config.text_model_id = str(companion_dir)
    config.audio_model_id = None
    config._name_or_path = str(ultravox_dir)
    model = AutoModel.from_pretrained(
        str(ultravox_dir),
        config=config,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval()
    model.to("cpu")
    if int(model.config.text_config.vocab_size) != VOCAB_SIZE:
        die(f"official model vocab={model.config.text_config.vocab_size}, expected {VOCAB_SIZE}")

    processor_class = get_class_from_dynamic_module(
        "ultravox_processing.UltravoxProcessor",
        str(ultravox_dir),
        local_files_only=True,
    )
    audio_processor = AutoFeatureExtractor.from_pretrained(
        str(ultravox_dir), local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(str(ultravox_dir), local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    processor = processor_class(
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        stack_factor=STACK_FACTOR,
        audio_context_size=model.audio_tower_context_length,
    )

    sample_index = numpy.arange(SAMPLE_COUNT, dtype=numpy.float64)
    seconds = sample_index / float(SAMPLE_RATE)
    pcm = (
        0.20 * numpy.sin(2.0 * numpy.pi * 440.0 * seconds)
        + 0.08 * numpy.sin(2.0 * numpy.pi * (180.0 * seconds + 160.0 * seconds**2))
    ).astype(numpy.float32)
    turns = [
        {
            "role": "system",
            "content": "You are a precise speech assistant.",
        },
        {"role": "user", "content": "<|audio|>"},
    ]
    text = tokenizer.apply_chat_template(turns, add_generation_prompt=True, tokenize=False)
    batch = processor(
        text=text,
        audio=pcm,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )
    required_keys = {
        "input_ids",
        "attention_mask",
        "audio_values",
        "audio_lens",
        "audio_token_len",
        "audio_token_start_idx",
        "audio_batch_size",
    }
    missing = sorted(required_keys.difference(batch.keys()))
    if missing:
        die(f"official processor omitted fields: {missing}")
    if tuple(batch["audio_values"].shape[:2]) != (1, N_MELS):
        die(f"official audio_values shape={tuple(batch['audio_values'].shape)}")
    audio_frames = int(batch["audio_values"].shape[-1])
    audio_len = int(batch["audio_lens"][0])
    audio_token_len = int(batch["audio_token_len"][0])
    audio_start = int(batch["audio_token_start_idx"][0])
    expected_token_len = (audio_len + ENCODER_DS_FACTOR * STACK_FACTOR - 1) // (
        ENCODER_DS_FACTOR * STACK_FACTOR
    )
    if audio_frames != audio_len or audio_token_len != expected_token_len:
        die(
            f"official audio contract frames={audio_frames} len={audio_len} "
            f"tokens={audio_token_len}, expected tokens={expected_token_len}"
        )

    prompt_ids = batch["input_ids"][0].detach().cpu().to(torch.int64)
    if audio_start + audio_token_len > prompt_ids.numel():
        die("official audio placeholder span exceeds prompt")
    terminators = [int(tokenizer.eos_token_id)]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and 0 <= eot_id < VOCAB_SIZE and eot_id not in terminators:
        terminators.append(eot_id)

    captured_projector: list[Any] = []
    official_projector = model.multi_modal_projector.forward

    def capture_projector(*projector_args: Any, **projector_kwargs: Any) -> Any:
        values = official_projector(*projector_args, **projector_kwargs)
        captured_projector.append(values.detach().cpu().float())
        return values

    with torch.inference_mode():
        model.multi_modal_projector.forward = capture_projector
        try:
            official_output = model(**batch, use_cache=False, return_dict=True)
        finally:
            model.multi_modal_projector.forward = official_projector
        next_logits = official_output.logits[0, -1].detach().cpu().float()
        generated = model.generate(
            **batch,
            do_sample=False,
            repetition_penalty=1.0,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=terminators,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    if len(captured_projector) != 1:
        die(f"official projector call count={len(captured_projector)}, expected 1")
    audio_embeddings = captured_projector[0][0, :audio_token_len]
    if tuple(audio_embeddings.shape) != (audio_token_len, TEXT_HIDDEN_SIZE):
        die(f"official projected audio shape={tuple(audio_embeddings.shape)}")
    if tuple(next_logits.shape) != (VOCAB_SIZE,):
        die(f"official next logits shape={tuple(next_logits.shape)}")
    generated_ids = generated[0, prompt_ids.numel() :].detach().cpu().to(torch.int64)
    if generated_ids.numel() == 0:
        die("official greedy generation emitted no token")

    write_f32(output / "pcm.f32le", pcm, numpy)
    write_f32(output / "input_features.f32le", batch["audio_values"][0].numpy(), numpy)
    write_f32(output / "audio_embeddings.f32le", audio_embeddings.numpy(), numpy)
    write_u32(output / "prompt_ids.u32le", prompt_ids.numpy(), numpy)
    write_f32(output / "next_logits.f32le", next_logits.numpy(), numpy)
    write_u32(output / "generated_ids.u32le", generated_ids.numpy(), numpy)

    inventory = source_inventory(ultravox_dir, companion_dir)
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
        "accelerate": importlib.metadata.version("accelerate"),
        "peft": importlib.metadata.version("peft"),
        "torch_cpu_capability": (
            torch.backends.cpu.get_cpu_capability()
            if hasattr(torch.backends.cpu, "get_cpu_capability")
            else "unavailable"
        ),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "device": "cpu",
        "dtype": "float32",
    }
    (output / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    artifacts = [
        "pcm.f32le",
        "input_features.f32le",
        "audio_embeddings.f32le",
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
        "companion_repo": COMPANION_REPO,
        "companion_revision": COMPANION_REVISION,
        "public_repo": PUBLIC_REPO,
        "public_revision": PUBLIC_REVISION,
        "public_filename": PUBLIC_FILENAME,
        "public_file_bytes": PUBLIC_FILE_BYTES,
        "public_file_sha256": PUBLIC_FILE_SHA256,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "sample_rate": SAMPLE_RATE,
        "sample_count": pcm.size,
        "n_mels": N_MELS,
        "audio_frames": audio_frames,
        "audio_token_len": audio_token_len,
        "audio_token_start_idx": audio_start,
        "audio_embedding_hidden": TEXT_HIDDEN_SIZE,
        "vocab_size": VOCAB_SIZE,
        "prompt_tokens": prompt_ids.numel(),
        "prompt_ids_csv": ",".join(str(int(token)) for token in prompt_ids),
        "stop_token_ids_csv": ",".join(str(token) for token in terminators),
        "max_new_tokens": args.max_new_tokens,
        "generated_tokens": generated_ids.numel(),
        "input_formula": "0.20*sin(2*pi*440*t)+0.08*sin(2*pi*(180*t+160*t^2))",
        "public_config_sha256": sha256_file(ultravox_dir / "config.json"),
        "public_weights_sha256": sha256_file(ultravox_dir / "model.safetensors"),
        "companion_config_sha256": sha256_file(companion_dir / "config.json"),
        "companion_weights_sha256": sha256_file(companion_dir / "model.safetensors"),
        "ultravox_inventory_sha256": sha256_file(ultravox_dir / ".vokra-source-inventory.json"),
        "companion_inventory_sha256": sha256_file(companion_dir / ".vokra-source-inventory.json"),
        "source_ultravox_config_sha256": CUSTOM_CODE["ultravox_config.py"][1],
        "source_ultravox_model_sha256": CUSTOM_CODE["ultravox_model.py"][1],
        "source_ultravox_processing_sha256": CUSTOM_CODE["ultravox_processing.py"][1],
        "config_model_type": public_config["model_type"],
    }
    for name in artifacts:
        manifest[f"sha256_{name.replace('.', '_')}"] = sha256_file(output / name)
    write_manifest(output / "manifest.txt", manifest)
    verify_reference_manifest(output)
    print(
        f"ULTRAVOX_OFFICIAL_REFERENCE frames={audio_frames} audio_tokens={audio_token_len} "
        f"prompt_tokens={prompt_ids.numel()} generated_tokens={generated_ids.numel()} "
        f"output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
