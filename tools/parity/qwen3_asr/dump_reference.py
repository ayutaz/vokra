#!/usr/bin/env python3
"""Dump an independent official Qwen3-ASR real-checkpoint reference.

The official backend source is loaded from an authenticated wheel.  The
package root and its inference wrapper are deliberately never imported: the
wrapper eagerly imports the optional forced-aligner/librosa closure.  Prompt
helpers and output parsing are AST-lifted from the exact, hash-bound official
source files, while the Transformers backend is imported unchanged.

The model snapshot must already be local and must have been downloaded at the
exact revision selected by ``--variant``.  Network fallback is disabled before
the package is imported.  This is a VAST-only tool: both released snapshots
and the generated GGUF validation run cross the repository's local-memory
guard.  It never uploads or publishes anything.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import platform
import re
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from wheel_audit import audit_wheel, extract_backend
except ModuleNotFoundError:  # pragma: no cover - package import path
    from tools.parity.qwen3_asr.wheel_audit import audit_wheel, extract_backend


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
TRANSFORMERS_VERSION = "5.10.4"
SCHEMA = "vokra-qwen3-asr-reference-v1"


def die(message: str) -> "None":
    raise SystemExit(f"qwen3_asr reference: {message}")


def require_vast_x86_64() -> None:
    if os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1":
        die("VOKRA_PUBLISH_ON_VAST=1 is required for model reference execution")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        die("reference execution is restricted to Linux x86_64 VAST")


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
    require_no_symlink_ancestors(path, "--output")
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        die(f"--output must be a regular directory path, not a symlink/file: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        die(f"--output directory is symlinked: {path}")
    entries = list(path.iterdir())
    if entries:
        die(f"--output must be empty, found {entries[0]}")


def require_no_symlink_ancestors(path: Path, label: str) -> None:
    candidate = path if path.is_absolute() else Path.cwd() / path
    while True:
        if candidate.is_symlink():
            die(f"{label} path has a symlink ancestor: {candidate}")
        parent = candidate.parent
        if parent == candidate:
            return
        candidate = parent


def require_model_identity(model_dir: Path, variant: Variant) -> dict[str, Any]:
    if model_dir.is_symlink() or not model_dir.is_dir():
        die(f"model directory must be a regular non-symlink directory: {model_dir}")
    config_path = model_dir / "config.json"
    if config_path.is_symlink() or not config_path.is_file():
        die(f"missing local config: {config_path}")
    config = strict_json_load(config_path, "config.json")
    if not isinstance(config, dict):
        die("config.json top-level value must be an object")
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
        if path.is_symlink() or not path.is_file():
            die(f"missing pinned sidecar: {path}")
        actual_bytes = path.stat().st_size
        actual_hash = sha256_file(path)
        if (actual_bytes, actual_hash) != (expected_bytes, expected_hash):
            die(
                f"{name} identity drift: bytes={actual_bytes} sha256={actual_hash}; "
                f"expected bytes={expected_bytes} sha256={expected_hash}"
            )
    return config


def strict_json_load(path: Path, label: str) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        die(f"{label} is not valid authenticated JSON: {error}")


def source_inventory(model_dir: Path) -> dict[str, dict[str, object]]:
    expected_names = {
        ".gitattributes",
        "LICENSE",
        "README.md",
        "config.json",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        *EXPECTED_ASSETS.keys(),
    }
    index_path = model_dir / "model.safetensors.index.json"
    single_checkpoint = model_dir / "model.safetensors"
    if index_path.exists():
        if index_path.is_symlink() or not index_path.is_file():
            die(f"model shard index is symlinked or not a regular file: {index_path}")
        index = strict_json_load(index_path, "model.safetensors.index.json")
        if not isinstance(index, dict) or set(index) != {"metadata", "weight_map"}:
            die("model.safetensors.index.json must contain exactly metadata and weight_map")
        if not isinstance(index["metadata"], dict):
            die("model.safetensors.index.json metadata must be an object")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            die("model.safetensors.index.json weight_map is missing or empty")
        expected_shards: set[str] = set()
        for tensor, shard in weight_map.items():
            if not isinstance(tensor, str) or not tensor or not isinstance(shard, str) or not shard:
                die("model.safetensors.index.json weight_map has a non-string/empty entry")
            shard_path = Path(shard)
            if shard_path.name != shard or shard_path.is_absolute() or shard_path.suffix != ".safetensors":
                die(f"model.safetensors.index.json uses unsafe shard path: {shard!r}")
            expected_shards.add(shard)
        actual_shards = {
            path.name
            for path in model_dir.iterdir()
            if path.name.endswith(".safetensors")
        }
        if actual_shards != expected_shards:
            die(
                "model safetensors set differs from authenticated index: "
                f"missing={sorted(expected_shards - actual_shards)} "
                f"extra={sorted(actual_shards - expected_shards)}"
            )
    elif single_checkpoint.exists():
        if single_checkpoint.is_symlink() or not single_checkpoint.is_file():
            die(f"model checkpoint is symlinked or not a regular file: {single_checkpoint}")
        expected_shards = {single_checkpoint.name}
    else:
        die("model snapshot lacks model.safetensors.index.json or model.safetensors")

    inventory: dict[str, dict[str, object]] = {}
    for path in sorted(model_dir.iterdir(), key=lambda candidate: candidate.name):
        if path.name == ".cache":
            if path.is_symlink() or not path.is_dir():
                die(f"model transport cache is symlinked or not a directory: {path}")
            continue
        if path.is_symlink():
            die(f"model snapshot entry is symlinked: {path}")
        if path.name not in expected_names and path.name not in expected_shards:
            die(f"model snapshot contains unexpected entry: {path.name}")
        if not path.is_file():
            die(f"model snapshot entry is not a regular file: {path}")
        inventory[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if set(name for name in inventory if name.endswith(".safetensors")) != expected_shards:
        die("model snapshot safetensors inventory does not match authenticated shard set")
    return inventory


def _lift_official_functions(source: str, names: tuple[str, ...]) -> dict[str, Any]:
    """Compile only named functions from a hash-authenticated official file.

    No source body is copied into this project.  The wheel auditor authenticates
    the complete file; this additionally rejects duplicate/missing definitions
    and executes only the exact AST nodes needed by the reference.
    """
    tree = ast.parse(source, mode="exec")
    found: dict[str, ast.AST] = {}
    constants: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            if node.name in found:
                raise ValueError(f"duplicate official function: {node.name}")
            found[node.name] = node
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"_ASR_TEXT_TAG", "_LANG_PREFIX"}:
                    if target.id in constants:
                        raise ValueError(f"duplicate official constant: {target.id}")
                    constants[target.id] = node
    if set(found) != set(names):
        raise ValueError(f"official function set drifted: expected {names}, got {sorted(found)}")
    if set(constants) != {"_ASR_TEXT_TAG", "_LANG_PREFIX"}:
        raise ValueError("official parser constants drifted")
    module = ast.Module(
        body=[
            ast.ImportFrom(module="typing", names=[ast.alias(name="Optional"), ast.alias(name="Tuple")], level=0),
            *constants.values(),
            *[found[name] for name in names],
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {}
    exec(compile(module, "<authenticated-qwen3-asr-utils>", "exec"), namespace, namespace)
    return {name: namespace[name] for name in names}


def _lift_official_prompt_helpers(source: str) -> Any:
    tree = ast.parse(source, mode="exec")
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen3ASRModel"]
    if len(classes) != 1:
        raise ValueError("official Qwen3ASRModel class shape drifted")
    wanted = ("_build_messages", "_build_text_prompt")
    methods = [node for node in classes[0].body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in methods} != set(wanted) or len(methods) != len(wanted):
        raise ValueError("official prompt helper method set drifted")
    helper_class = ast.ClassDef(
        name="AuthenticatedPromptHelpers",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="typing",
                names=[ast.alias(name="Any"), ast.alias(name="Dict"), ast.alias(name="List"), ast.alias(name="Optional")],
                level=0,
            ),
            helper_class,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {}
    exec(compile(module, "<authenticated-qwen3-asr-wrapper>", "exec"), namespace, namespace)
    return namespace["AuthenticatedPromptHelpers"]


def load_official_support(backend_root: Path) -> tuple[Any, Any]:
    """Load backend unchanged and AST-lift wrapper utilities without imports."""
    package_root = backend_root / "qwen_asr"
    package = types.ModuleType("qwen_asr")
    package.__path__ = [str(package_root)]
    sys.modules["qwen_asr"] = package
    core = types.ModuleType("qwen_asr.core")
    core.__path__ = [str(package_root / "core")]
    sys.modules["qwen_asr.core"] = core
    backend_path = package_root / "core" / "transformers_backend" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "qwen_asr.core.transformers_backend",
        backend_path,
        submodule_search_locations=[str(backend_path.parent)],
    )
    if spec is None or spec.loader is None:
        raise ImportError("unable to load authenticated Transformers backend")
    backend = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = backend
    spec.loader.exec_module(backend)
    wrapper_source = (package_root / "inference" / "qwen3_asr.py").read_text(encoding="utf-8")
    utils_source = (package_root / "inference" / "utils.py").read_text(encoding="utf-8")
    prompt_helpers = _lift_official_prompt_helpers(wrapper_source)
    utils = _lift_official_functions(utils_source, ("normalize_language_name", "detect_and_fix_repetitions", "parse_asr_output"))
    return backend, (prompt_helpers, utils["parse_asr_output"])


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
    parser.add_argument("--wheel", type=Path, required=True)
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
    require_vast_x86_64()
    variant = VARIANTS[args.variant]
    for label, path in (
        ("--model-dir", args.model_dir),
        ("--audio", args.audio),
        ("--wheel", args.wheel),
        ("--output", args.output),
    ):
        require_no_symlink_ancestors(path, label)
    model_dir = args.model_dir.resolve()
    audio_path = args.audio.resolve()
    wheel_path = args.wheel.resolve()
    output = args.output.resolve()
    if not model_dir.is_dir():
        die(f"--model-dir is not a directory: {model_dir}")
    if not audio_path.is_file():
        die(f"--audio is not a file: {audio_path}")
    if wheel_path.is_symlink() or not wheel_path.is_file():
        die(f"--wheel is not a regular non-symlink file: {wheel_path}")
    if not 1 <= args.max_new_tokens <= 512:
        die("--max-new-tokens must be in 1..=512")
    language = None if args.language.lower() == "auto" else args.language
    require_empty_output(output)
    config = require_model_identity(model_dir, variant)
    try:
        wheel_evidence = audit_wheel(wheel_path)
    except ValueError as error:
        die(f"official wheel audit failed: {error}")

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
    except ImportError as error:
        die(
            "official reference dependencies are required; run with "
            f"`uv run --project tools/parity/qwen3_asr --frozen`: {error}"
        )

    if transformers.__version__ != TRANSFORMERS_VERSION:
        die(f"transformers={transformers.__version__}, expected {TRANSFORMERS_VERSION}")

    wheel_directory = tempfile.TemporaryDirectory(prefix="qwen3-asr-official-wheel-")
    wheel_root = Path(wheel_directory.name)
    try:
        extract_backend(wheel_path, wheel_root)
        backend, (prompt_helper_type, parse_asr_output) = load_official_support(wheel_root)
    except (ImportError, SyntaxError, ValueError) as error:
        wheel_directory.cleanup()
        die(f"authenticated official backend/support load failed: {error}")

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
        "qwen_asr": QWEN_ASR_VERSION,
        "device": "cpu",
        "dtype": "float32",
    }
    print(json.dumps({"reference_environment": environment}, sort_keys=True), flush=True)

    # Load the official Transformers backend directly; no wrapper registration
    # or optional forced-aligner/librosa closure is involved.
    model = backend.Qwen3ASRForConditionalGeneration.from_pretrained(
        str(model_dir),
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    model.to("cpu").eval()
    processor = backend.Qwen3ASRProcessor.from_pretrained(str(model_dir), local_files_only=True)

    prompt_helper = prompt_helper_type.__new__(prompt_helper_type)
    prompt_helper.processor = processor
    prompt = prompt_helper._build_text_prompt(context=args.context, force_language=language)
    inputs = processor(text=[prompt], audio=[pcm], return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(torch.float32)
    prompt_ids = inputs["input_ids"][0].detach().cpu().to(torch.int64)

    with torch.inference_mode():
        audio_embeddings = model.thinker.get_audio_features(
            inputs["input_features"],
            feature_attention_mask=inputs["feature_attention_mask"],
        )
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    if audio_embeddings.ndim != 2 or audio_embeddings.shape[1] != variant.hidden_size:
        die(
            f"official audio tap shape={tuple(audio_embeddings.shape)}, "
            f"expected [frames,{variant.hidden_size}]"
        )
    sequences = generated.sequences
    if sequences.ndim != 2 or sequences.shape[0] != 1:
        die(f"official generate returned unexpected sequences shape={tuple(sequences.shape)}")
    generated_ids = sequences[0, prompt_ids.numel() :].detach().cpu().to(torch.int64)
    raw_text = processor.batch_decode(
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
        "qwen_asr_version": QWEN_ASR_VERSION,
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
        "official_wheel": wheel_evidence,
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
