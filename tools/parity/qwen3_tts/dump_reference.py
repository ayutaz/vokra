#!/usr/bin/env python3
"""Dump an independent official Qwen3-TTS real-weight reference.

The QwenLM/Qwen3-TTS package is imported from the immutable source revision in
the lockfile. No layer or prompt logic is reimplemented here. The official
wrapper is used to build prompts and generation kwargs; a temporary hook on
the official speech tokenizer captures the exact generated code packet before
the same official tokenizer decodes it to PCM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SOURCE_REPO = "QwenLM/Qwen3-TTS"
SOURCE_REVISION = "022e286b98fbec7e1e916cb940cdf532cd9f488e"
PACKAGE_VERSION = "0.1.1"
SCHEMA = "vokra-qwen3-tts-reference-v1"
CODEBOOKS = 16
OUTPUT_SAMPLE_RATE = 24_000
TEXT = "The Vokra parity packet is short and deterministic."
LANGUAGE = "English"
MAX_NEW_TOKENS = 8
MIN_NEW_TOKENS = 2
SPEAKER = "Serena"
DECODER_REPO = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
DECODER_REVISION = "a87c50897bb00837eb857d0538b29d117541d7f6"
DECODER_CHECKPOINT_SHA256 = "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258"
TRANSFORMERS_COMPATIBILITY_STATUS = "BLOCKED_UNVERIFIED_API_SMOKE"


@dataclass(frozen=True)
class Variant:
    slug: str
    repo: str
    revision: str
    model_name: str
    kind: str
    speaker_dim: int
    config_bytes: int
    config_sha256: str


VARIANTS = {
    "0.6b-base": Variant(
        "0.6b-base", "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "5d83992436eae1d760afd27aff78a71d676296fc",
        "qwen3-tts-12hz-0.6b-base", "base", 1024, 4494,
        "2e714c787c8edb98b05432685cddb634add2de4d4e645f653d68251ef72ba011",
    ),
    "0.6b-customvoice": Variant(
        "0.6b-customvoice", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "85e237c12c027371202489a0ec509ded67b5e4b5",
        "qwen3-tts-12hz-0.6b-customvoice", "custom_voice", 0, 4908,
        "81aca2b6fac304944d8acf345272d8a9a727d5fc2e2e66b222ab4729340c7455",
    ),
    "1.7b-base": Variant(
        "1.7b-base", "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "fd4b254389122332181a7c3db7f27e918eec64e3",
        "qwen3-tts-12hz-1.7b-base", "base", 2048, 4494,
        "b4f01752d15a488abde3e1ab44723ae4f4b9e68a4037257b098b3737893cc1f9",
    ),
    "1.7b-customvoice": Variant(
        "1.7b-customvoice", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "0c0e3051f131929182e2c023b9537f8b1c68adfe",
        "qwen3-tts-12hz-1.7b-customvoice", "custom_voice", 0, 4908,
        "17a07f527a1c25ea30b4e023a184482a23d3e279d697b1dc81b1bde498d29cf9",
    ),
}

COMMON_ASSETS = {
    "vocab.json": (2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
    "merges.txt": (1671839, "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"),
    "tokenizer_config.json": (7344, "dc3c31c3bdaedd5016382bb3cbe07323026775ad51f5a4fb564505992ae4a670"),
    "generation_config.json": (245, "f1b90b4513f3b34c62851049e2492d7b4c5940daf1276f89c82b8ef04127f3aa"),
}


def die(message: str) -> "None":
    raise SystemExit(f"qwen3_tts reference: {message}")


def require_transformers_api_smoke() -> None:
    if TRANSFORMERS_COMPATIBILITY_STATUS == "AUTHENTICATED_API_SMOKE":
        return
    if TRANSFORMERS_COMPATIBILITY_STATUS == "BLOCKED_UNVERIFIED_API_SMOKE":
        die("Transformers API smoke is not authenticated; refusing official reference imports")
    die(f"unknown Transformers API smoke status: {TRANSFORMERS_COMPATIBILITY_STATUS}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_f32(path: Path, values: Any, numpy: Any) -> None:
    array = numpy.asarray(values, dtype=numpy.float32)
    if array.size == 0 or not numpy.isfinite(array).all():
        die(f"{path.name} is empty or contains non-finite values")
    path.write_bytes(numpy.ascontiguousarray(array, dtype="<f4").tobytes())


def write_u32(path: Path, values: Any, numpy: Any) -> None:
    array = numpy.asarray(values)
    if array.size == 0 or array.min() < 0 or array.max() > 0xFFFFFFFF:
        die(f"{path.name} is empty or contains an out-of-range token")
    path.write_bytes(numpy.ascontiguousarray(array, dtype="<u4").tobytes())


def require_snapshot(model_dir: Path, variant: Variant) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        die(f"missing {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_tts":
        die(f"config model_type={config.get('model_type')!r} is not qwen3_tts")
    if config.get("architectures") != ["Qwen3TTSForConditionalGeneration"]:
        die(f"unexpected official architectures: {config.get('architectures')!r}")
    if config.get("tts_model_type") != variant.kind:
        die(f"config tts_model_type={config.get('tts_model_type')!r}, expected {variant.kind!r}")
    if config.get("tts_model_size") != ("0b6" if variant.slug.startswith("0.6b") else "1b7"):
        die(f"config tts_model_size does not match {variant.slug}")
    expected = (variant.config_bytes, variant.config_sha256)
    actual = (config_path.stat().st_size, sha256_file(config_path))
    if actual != expected:
        die(f"config identity drift: got {actual}, expected {expected}")
    for name, identity in COMMON_ASSETS.items():
        path = model_dir / name
        actual = (path.stat().st_size, sha256_file(path)) if path.is_file() else None
        if actual != identity:
            die(f"{name} identity drift: got {actual}, expected {identity}")
    if not list(model_dir.glob("*.safetensors")):
        die(f"no main safetensors file in {model_dir}")
    if not (model_dir / "speech_tokenizer").is_dir():
        die(f"official speech_tokenizer directory is missing in {model_dir}")
    return config


def require_empty(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        die(f"output directory must be empty: {path}")


def require_decoder_snapshot(model_dir: Path, decoder_dir: Path) -> tuple[str, str]:
    standalone = decoder_dir / "model.safetensors"
    nested = model_dir / "speech_tokenizer" / "model.safetensors"
    if not standalone.is_file() or sha256_file(standalone) != DECODER_CHECKPOINT_SHA256:
        die("standalone decoder checkpoint is missing or has the wrong authenticated SHA-256")
    if not nested.is_file():
        die(f"nested decoder checkpoint is missing: {nested}")
    nested_sha = sha256_file(nested)
    if nested_sha != DECODER_CHECKPOINT_SHA256:
        die(f"nested decoder SHA-256 {nested_sha} differs from standalone decoder")
    return DECODER_CHECKPOINT_SHA256, nested_sha


def require_source_tree(source_dir: Path) -> None:
    if not source_dir.is_dir() or not (source_dir / ".git").is_dir():
        die(f"authenticated official source tree is missing: {source_dir}")
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT
        ).strip()
        metadata = tomllib.loads((source_dir / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, subprocess.CalledProcessError, tomllib.TOMLDecodeError) as error:
        die(f"authenticated official source metadata is unreadable: {error}")
    if revision != SOURCE_REVISION:
        die(f"official source revision {revision!r} != pinned {SOURCE_REVISION!r}")
    if metadata.get("project", {}).get("version") != PACKAGE_VERSION:
        die("official source package version drifted")
    if not (source_dir / "qwen_tts" / "__init__.py").is_file():
        die("official qwen_tts package is missing from the authenticated source tree")


def environment() -> dict[str, object]:
    return {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(), "device": "cpu", "torch_threads": 1}


def run_self_test() -> int:
    """Exercise the immutable packet contract without importing torch or weights."""
    global TRANSFORMERS_COMPATIBILITY_STATUS
    saved_status = TRANSFORMERS_COMPATIBILITY_STATUS
    try:
        TRANSFORMERS_COMPATIBILITY_STATUS = "BLOCKED_UNVERIFIED_API_SMOKE"
        try:
            require_transformers_api_smoke()
        except SystemExit:
            pass
        else:
            die("blocked Transformers API smoke status was accepted")
        TRANSFORMERS_COMPATIBILITY_STATUS = "AUTHENTICATED_API_SMOKE"
        require_transformers_api_smoke()
        TRANSFORMERS_COMPATIBILITY_STATUS = "UNKNOWN_STATUS"
        try:
            require_transformers_api_smoke()
        except SystemExit:
            pass
        else:
            die("unknown Transformers API smoke status was accepted")
    finally:
        TRANSFORMERS_COMPATIBILITY_STATUS = saved_status
    if len(SOURCE_REVISION) != 40 or any(c not in "0123456789abcdef" for c in SOURCE_REVISION):
        die("official source revision is not an immutable SHA-1")
    if len(VARIANTS) != 4 or set(VARIANTS) != {"0.6b-base", "0.6b-customvoice", "1.7b-base", "1.7b-customvoice"}:
        die("the four public variants are not all registered")
    for variant in VARIANTS.values():
        if len(variant.revision) != 40 or any(c not in "0123456789abcdef" for c in variant.revision):
            die(f"{variant.slug} revision is not immutable")
        if variant.speaker_dim and variant.kind != "base":
            die(f"{variant.slug} has an invalid speaker embedding contract")
        if variant.kind == "base" and variant.speaker_dim not in (1024, 2048):
            die(f"{variant.slug} has an invalid speaker embedding width")
    if CODEBOOKS != 16 or OUTPUT_SAMPLE_RATE != 24_000 or MAX_NEW_TOKENS != 8 or MIN_NEW_TOKENS != 2:
        die("fixed packet contract drifted")
    if DECODER_REPO != "Qwen/Qwen3-TTS-Tokenizer-12Hz" or len(DECODER_REVISION) != 40:
        die("decoder identity drifted")
    source = Path(__file__).read_text(encoding="utf-8")
    if "from qwen_tts import Qwen3TTSModel" not in source or "local_files_only=True" not in source or "nested_decoder_sha256" not in source or "--source-dir" not in source:
        die("reference is not using the official local-only wrapper")
    if "pickle." + "loads" in source or "weights_only=" + "False" in source:
        die("unsafe pickle loading appeared in the reference dumper")
    print("qwen3_tts reference self-test: PASS")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS))
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--decoder-dir", type=Path)
    parser.add_argument("--source-dir", type=Path, help="authenticated official Qwen3-TTS source checkout")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-audio", type=Path, help="required for Base speaker embedding")
    parser.add_argument("--self-test", action="store_true", help="check the packet contract without loading weights")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        if any(value is not None for value in (args.variant, args.model_dir, args.decoder_dir, args.source_dir, args.output, args.reference_audio)):
            die("--self-test accepts no model or output arguments")
        return run_self_test()
    require_transformers_api_smoke()
    if args.variant is None or args.model_dir is None or args.decoder_dir is None or args.source_dir is None or args.output is None:
        die("--variant, --model-dir, --decoder-dir, --source-dir, and --output are required")
    variant = VARIANTS[args.variant]
    model_dir = args.model_dir.resolve()
    output = args.output.resolve()
    if not model_dir.is_dir():
        die(f"model directory is missing: {model_dir}")
    decoder_dir = args.decoder_dir.resolve()
    if not decoder_dir.is_dir():
        die(f"decoder directory is missing: {decoder_dir}")
    if variant.kind == "base" and (args.reference_audio is None or not args.reference_audio.is_file()):
        die("Base variants require --reference-audio")
    source_dir = args.source_dir.resolve()
    require_source_tree(source_dir)
    sys.path.insert(0, str(source_dir))
    require_empty(output)
    config = require_snapshot(model_dir, variant)
    decoder_sha, nested_decoder_sha = require_decoder_snapshot(model_dir, decoder_dir)
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    try:
        import numpy
        import qwen_tts
        import soundfile
        import torch
        from qwen_tts import Qwen3TTSModel
    except ImportError as error:
        die(f"official qwen-tts import failed; use the frozen project: {error}")
    imported_root = Path(qwen_tts.__file__).resolve().parents[1]
    if imported_root != source_dir:
        die(f"imported qwen_tts from {imported_root}, expected authenticated source {source_dir}")
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)
    torch.manual_seed(1234)
    numpy.random.seed(1234)
    tts = Qwen3TTSModel.from_pretrained(str(model_dir), local_files_only=True, dtype=torch.float32, device_map="cpu")
    if tts.device.type != "cpu":
        die(f"official model selected {tts.device}, expected CPU")
    input_ids = tts._tokenize_texts([tts._build_assistant_text(TEXT)])[0][0].detach().cpu()
    prompt = None
    if variant.kind == "base":
        prompt = tts.create_voice_clone_prompt(ref_audio=str(args.reference_audio), x_vector_only_mode=True)[0]
        if prompt.ref_spk_embedding.numel() != variant.speaker_dim:
            die(f"speaker embedding has {prompt.ref_spk_embedding.numel()} values, expected {variant.speaker_dim}")
        write_f32(output / "speaker_embedding.f32le", prompt.ref_spk_embedding.detach().cpu().numpy(), numpy)
    captured: list[Any] = []
    decoder = tts.model.speech_tokenizer
    original_decode = decoder.decode
    def capture(packet: Any) -> Any:
        captured.append(packet[0]["audio_codes"].detach().cpu().clone())
        return original_decode(packet)
    decoder.decode = capture
    try:
        kwargs = tts._merge_generate_kwargs(do_sample=False, top_k=None, top_p=1.0, temperature=0.0, repetition_penalty=1.0, subtalker_dosample=False, subtalker_top_k=None, subtalker_top_p=1.0, subtalker_temperature=0.0, max_new_tokens=MAX_NEW_TOKENS, min_new_tokens=MIN_NEW_TOKENS)
        if variant.kind == "base":
            wavs, sample_rate = tts.generate_voice_clone(TEXT, language=LANGUAGE, voice_clone_prompt=[prompt], non_streaming_mode=False, **kwargs)
        else:
            wavs, sample_rate = tts.generate_custom_voice(TEXT, language=LANGUAGE, speaker=SPEAKER, instruct=None, non_streaming_mode=True, **kwargs)
    finally:
        decoder.decode = original_decode
    if len(captured) != 1:
        die(f"official decoder hook captured {len(captured)} packets, expected one")
    codes = captured[0]
    if codes.ndim != 2 or codes.shape[1] != CODEBOOKS:
        die(f"official code packet shape={tuple(codes.shape)}, expected [frames,{CODEBOOKS}]")
    pcm = numpy.asarray(wavs[0], dtype=numpy.float32)
    if int(sample_rate) != OUTPUT_SAMPLE_RATE:
        die(f"official decoder sample rate={sample_rate}, expected {OUTPUT_SAMPLE_RATE}")
    write_u32(output / "prompt_ids.u32le", input_ids.numpy(), numpy)
    write_u32(output / "codes.u32le", codes.numpy(), numpy)
    write_f32(output / "pcm.f32le", pcm, numpy)
    env = environment() | {"numpy": numpy.__version__, "torch": torch.__version__, "qwen_tts": PACKAGE_VERSION}
    (output / "environment.json").write_text(json.dumps(env, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest: dict[str, object] = {
        "schema": SCHEMA, "variant": variant.slug, "model_name": variant.model_name,
        "upstream_repo": variant.repo, "upstream_revision": variant.revision,
        "official_source_repo": SOURCE_REPO, "official_source_revision": SOURCE_REVISION,
        "qwen_tts_version": PACKAGE_VERSION, "text": TEXT, "language": LANGUAGE,
        "speaker": SPEAKER if variant.kind != "base" else "official_x_vector_only",
        "max_new_tokens": MAX_NEW_TOKENS, "min_new_tokens": MIN_NEW_TOKENS, "sampling": "greedy",
        "sample_rate": OUTPUT_SAMPLE_RATE, "frames": int(codes.shape[0]), "codebooks": CODEBOOKS,
        "decoder_repo": DECODER_REPO, "decoder_revision": DECODER_REVISION,
        "decoder_checkpoint_sha256": decoder_sha, "nested_decoder_sha256": nested_decoder_sha,
        "config_sha256": sha256_file(model_dir / "config.json"),
        "main_safetensors": sorted((p.name, p.stat().st_size, sha256_file(p)) for p in model_dir.glob("*.safetensors")),
    }
    for path in sorted(output.iterdir()):
        if path.is_file():
            manifest[f"sha256_{path.name.replace('.', '_')}"] = sha256_file(path)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"QWEN3_TTS_OFFICIAL_REFERENCE variant={variant.slug} frames={codes.shape[0]} codebooks={CODEBOOKS} output={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
