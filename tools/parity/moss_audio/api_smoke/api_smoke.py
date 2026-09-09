#!/usr/bin/env -S uv run --script
"""Model-free official MOSS-Audio Transformers API smoke.

This probe authenticates only the pinned source files and non-weight model
metadata.  It imports the official configuration and processor classes and
constructs a configuration/processor without loading a checkpoint.  It never
downloads a model and never imports Vokra.
"""

from __future__ import annotations

import argparse
import builtins
from contextlib import contextmanager, nullcontext
import hashlib
import importlib
import inspect
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SOURCE_REPO = "https://github.com/OpenMOSS/MOSS-Audio.git"
SOURCE_REVISION = "5cbb1d823937cd5b5de3d8fa4d3a7253ebd3b883"
SOURCE_FILES = {
    "src/configuration_moss_audio.py": "e597dca441ff7fb58a5ec43186fafdfce19f31dada4955b4910059baa5d52ebd",
    "src/modeling_moss_audio.py": "a52513e518c68a0ba7c636a1ab0e12f7755ceebd0ae033235dc5e2551bfcbf9c",
    "src/processing_moss_audio.py": "05fb788cbdc6482eded8d70f7d2f524bc0cdca47d001acab5661c11f02cc6fe6",
}
VARIANTS = {
    "4b": {
        "repo": "OpenMOSS-Team/MOSS-Audio-4B-Instruct",
        "revision": "6907a499dc0e87cc77c8ae0fe23fd0eb5476a02d",
        "model_name": "moss-audio-4b-instruct",
        "config_sha256": "e528a941446f4443f1b9fede12ea484e58a79d494c28d21ef1e73b5148abfbfa",
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "metadata": {
            "tokenizer_config.json": "443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0",
            "processor_config.json": "0749d81701d2a2a2e83ca4d549fbebb1a205acac1ac7bdccea7965c1913b2cbf",
            "vocab.json": "87a257b04b17642a0688c98cd1df89c398bda4fee532d6f88b38a659ecb4ac8d",
            "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
            "chat_template.jinja": "87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5",
            "generation_config.json": "bb52bfdd308deaea4ec800bf0165e75770b0a4e5c105963bee1b0398f4043d3e",
        },
    },
    "8b": {
        "repo": "OpenMOSS-Team/MOSS-Audio-8B-Instruct",
        "revision": "6521a39181b47a18f2d9f4b3acfb5bca7b76b57f",
        "model_name": "moss-audio-8b-instruct",
        "config_sha256": "535154c2a5bcbd0e18e2f92bcf370ac74b530eec97ad4fd9317993ba0a316536",
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "metadata": {
            "tokenizer_config.json": "0869e41f5d123ff144a811f0d83c5d18871dcd4b4064f46bf9def194bfbc6f41",
            "processor_config.json": "6a5c462858acb299db0d2d967b63d520b72d178f44d1619c33fc860f25fdccbf",
            "vocab.json": "87a257b04b17642a0688c98cd1df89c398bda4fee532d6f88b38a659ecb4ac8d",
            "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
            "chat_template.jinja": "87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5",
            "generation_config.json": "bb52bfdd308deaea4ec800bf0165e75770b0a4e5c105963bee1b0398f4043d3e",
        },
    },
}
PROJECT_SHA256 = "af7486c52182a23cc64185e6ad95c42b8863673dc4e032fda79637a9ca0d427e"
LOCK_SHA256 = "fd8c1f2342da9512d1fb1e97ed0e7640ffa4a83db279aeaba655ce6cf89c1085"
REQUIRED_DEPENDENCIES = {
    "einops==0.8.1", "numpy==2.3.5", "safetensors==0.7.0",
    "scipy==1.16.3", "soundfile==0.13.1", "tiktoken==0.12.0", "torch==2.9.1",
    "torchaudio==2.9.1", "transformers==5.10.4",
}
FORMAT = "vokra-moss-audio-transformers-api-smoke-v1"
MODEL_FREE_FORMAT = "vokra-moss-audio-model-free-api-smoke-v1"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
UNRESOLVED = {"", "none", "null", "unresolved", "pending", "todo", "owner_review_required"}
TRUST_REMOTE_CODE_PROMPT_MARKERS = ("custom code", "trust_remote_code", "trust remote code")
MODEL_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx")
MODEL_WEIGHT_PREFIXES = ("model-", "pytorch_model", "consolidated.", "adapter_model")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def strict_json(path: Path) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)


def pending_approval() -> dict[str, Any]:
    return {
        "source_license": "PENDING_OWNER_APPROVAL",
        "model_license": "PENDING_OWNER_APPROVAL",
        "operator": "PENDING_OWNER_APPROVAL",
        "signer": None,
        "scope_sha256": None,
    }


def write_model_free_evidence(path: Path, evidence: dict[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise ValueError("model-free evidence output must be an absent absolute path")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(evidence, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class NonInteractiveInputRefusal:
    def __init__(self) -> None:
        self.prompt_digests: list[dict[str, Any]] = []

    def __call__(self, prompt: object = "") -> str:
        prompt_text = str(prompt)
        if len(prompt_text) > 4096:
            raise ValueError("trust-remote-code prompt exceeds bounded length")
        prompt_digest = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        record = {"bytes": len(prompt_text.encode("utf-8")), "sha256": prompt_digest}
        self.prompt_digests.append(record)
        if len(self.prompt_digests) > 1:
            raise ValueError("unexpected second trust-remote-code prompt")
        lowered = prompt_text.casefold()
        if not any(marker in lowered for marker in TRUST_REMOTE_CODE_PROMPT_MARKERS):
            raise ValueError("unexpected non-trust-remote-code prompt")
        return "n"

    def evidence(self) -> dict[str, Any]:
        return {
            "installed": True,
            "decision": "n",
            "prompt_count": len(self.prompt_digests),
            "prompt_digests": list(self.prompt_digests),
        }


class ModelFreeApiFailure(ValueError):
    """A model-free API failure carrying bounded input-prompt evidence."""

    def __init__(self, message: str, *, input_refusal: dict[str, Any]) -> None:
        super().__init__(message)
        self.input_refusal = input_refusal


@contextmanager
def install_noninteractive_input_refusal() -> Any:
    previous = builtins.input
    refusal = NonInteractiveInputRefusal()
    builtins.input = refusal
    try:
        yield refusal
    finally:
        if builtins.input is not refusal:
            builtins.input = previous
            raise ValueError("input refusal sentinel was overwritten")
        builtins.input = previous


def require_regular(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"{label} is missing, symlinked, or empty: {path}")


def require_clean_head(root: Path, expected: str) -> None:
    if not HEX40.fullmatch(expected):
        raise ValueError("expected HEAD must be lowercase 40-hex")
    actual = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if actual != expected:
        raise ValueError(f"Vokra HEAD drifted: {actual} != {expected}")
    status = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], check=True, capture_output=True, text=True).stdout
    if status:
        raise ValueError("Vokra checkout is dirty")


def package_rows(lock: dict[str, Any]) -> list[dict[str, Any]]:
    expected_top_level = {
        "version", "revision", "requires-python", "resolution-markers",
        "supported-markers", "manifest", "package",
    }
    if set(lock) != expected_top_level:
        raise ValueError("uv.lock top-level schema drifted")
    if lock["manifest"] != {
        "constraints": [{"name": "setuptools", "specifier": ">=83.0.0"}]
    }:
        raise ValueError("uv.lock manifest constraints drifted")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for package in lock["package"]:
        if not isinstance(package, dict) or not isinstance(package.get("name"), str) or not isinstance(package.get("version"), str):
            raise ValueError("uv.lock package identity is malformed")
        source = package.get("source")
        if not isinstance(source, dict) or set(source) != {"registry"} or source["registry"] not in {"https://pypi.org/simple", "https://download.pytorch.org/whl/cpu"}:
            if source == {"virtual": "."}:
                continue
            raise ValueError(f"uv.lock package source is not approved: {package['name']}")
        identity = (package["name"], package["version"], source["registry"])
        if identity in seen:
            raise ValueError("uv.lock contains duplicate package identity")
        seen.add(identity)
        for artifact_name in ("sdist", "wheels"):
            artifacts = package.get(artifact_name, [] if artifact_name == "wheels" else None)
            candidates = [] if artifacts is None else (artifacts if isinstance(artifacts, list) else [artifacts])
            for artifact in candidates:
                # The PyTorch CPU index currently emits no size field in uv's
                # lock record.  Accept that one authenticated uv schema only;
                # PyPI records must retain the complete size-bearing schema.
                expected_keys = {"url", "hash", "upload-time"}
                if source["registry"] == "https://pypi.org/simple":
                    expected_keys.add("size")
                if not isinstance(artifact, dict) or set(artifact) != expected_keys:
                    raise ValueError(f"{package['name']} {artifact_name} artifact schema is not exact")
                expected_host = "download-r2.pytorch.org" if source["registry"] == "https://download.pytorch.org/whl/cpu" else "files.pythonhosted.org"
                parsed = urlsplit(artifact["url"]) if isinstance(artifact["url"], str) else None
                size_valid = source["registry"] != "https://pypi.org/simple" or (isinstance(artifact["size"], int) and artifact["size"] > 0)
                if parsed is None or parsed.scheme != "https" or parsed.netloc != expected_host or not parsed.path or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(artifact["hash"])) or not size_valid or not isinstance(artifact["upload-time"], str):
                    raise ValueError(f"{package['name']} {artifact_name} artifact identity is malformed")
        rows.append({"name": package["name"], "version": package["version"], "source": source})
    return sorted(rows, key=lambda row: (row["name"], row["version"], row["source"]["registry"]))


def verify_project(project: Path) -> tuple[list[dict[str, Any]], str, str]:
    pyproject = project / "pyproject.toml"
    lock_path = project / "uv.lock"
    require_regular(pyproject, "API smoke pyproject")
    require_regular(lock_path, "API smoke uv.lock")
    project_hash, lock_hash = sha256_file(pyproject), sha256_file(lock_path)
    if project_hash != PROJECT_SHA256 or lock_hash != LOCK_SHA256:
        raise ValueError("API smoke project or lock bytes drifted")
    project_data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    dependencies = project_data.get("project", {}).get("dependencies", [])
    if set(dependencies) != REQUIRED_DEPENDENCIES:
        raise ValueError("patched reference dependency closure is not exact")
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    rows = package_rows(lock)
    return rows, project_hash, lock_hash


def verify_approval(path: Path, expected_head: str, variants: list[str], rows: list[dict[str, Any]], project_hash: str, lock_hash: str) -> tuple[str, dict[str, Any]]:
    require_regular(path, "API smoke approval")
    approval = strict_json(path)
    if not isinstance(approval, dict) or set(approval) != {"schema", "decision", "signer", "scope", "scope_sha256"}:
        raise ValueError("API smoke approval schema is not exact")
    if approval["schema"] != "vokra-moss-audio-api-approval-v1" or approval["decision"] != "APPROVED" or not isinstance(approval["signer"], str) or approval["signer"].strip().casefold() in UNRESOLVED:
        raise ValueError("API smoke approval is not an explicit owner approval")
    reviews = approval["scope"].get("package_reviews") if isinstance(approval["scope"], dict) else None
    expected_review_keys = {"name", "version", "source", "status", "license", "native_review", "bundled_review"}
    if not isinstance(reviews, list) or len(reviews) != len(rows):
        raise ValueError("API smoke package review closure is incomplete")
    review_by_id: dict[tuple[str, str, str], dict[str, Any]] = {}
    for review in reviews:
        if not isinstance(review, dict) or set(review) != expected_review_keys or review.get("status") != "REVIEWED":
            raise ValueError("API smoke package review is unresolved or malformed")
        if any(not isinstance(review.get(key), str) or review[key].strip().casefold() in UNRESOLVED for key in ("license", "native_review", "bundled_review")):
            raise ValueError("API smoke package license/native review is unresolved")
        source = review.get("source")
        if not isinstance(source, dict) or set(source) != {"registry"}:
            raise ValueError("API smoke package review source is malformed")
        key = (review["name"], review["version"], source["registry"])
        if key in review_by_id:
            raise ValueError("API smoke package review is duplicated")
        review_by_id[key] = review
    if set(review_by_id) != {(r["name"], r["version"], r["source"]["registry"]) for r in rows}:
        raise ValueError("API smoke package review identities do not match uv.lock")
    license_reviews = approval["scope"].get("license_reviews") if isinstance(approval["scope"], dict) else None
    if not isinstance(license_reviews, dict) or set(license_reviews) != {"source", "4b", "8b"}:
        raise ValueError("API smoke source/model license review closure is incomplete")
    for label, review in license_reviews.items():
        if not isinstance(review, dict) or set(review) != {"status", "spdx", "evidence_sha256"} or review.get("status") != "REVIEWED" or not isinstance(review.get("spdx"), str) or review["spdx"].strip().casefold() in UNRESOLVED or not HEX64.fullmatch(str(review.get("evidence_sha256"))):
            raise ValueError(f"API smoke license review is unresolved: {label}")
    scope = {
        "expected_head": expected_head,
        "variants": variants,
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "project_sha256": project_hash,
        "lock_sha256": lock_hash,
        "package_rows_sha256": digest(rows),
        "package_reviews": reviews,
        "license_reviews": license_reviews,
    }
    if approval["scope"] != scope or approval["scope_sha256"] != digest(scope):
        raise ValueError("API smoke approval is not bound to exact head/project/package scope")
    return approval["signer"], scope


def verify_source(source: Path) -> dict[str, Any]:
    if source.is_symlink() or not source.is_dir():
        raise ValueError("official source root is not a real directory")
    actual_revision = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if actual_revision != SOURCE_REVISION:
        raise ValueError(f"official source revision drifted: {actual_revision}")
    if subprocess.run(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"], check=True, capture_output=True, text=True).stdout:
        raise ValueError("official source checkout is dirty")
    files: dict[str, Any] = {}
    for relative, expected in SOURCE_FILES.items():
        path = source / relative
        require_regular(path, f"official source {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"official source hash drifted for {relative}: {actual}")
        files[relative] = {"sha256": actual, "bytes": path.stat().st_size}
    return {"repo": SOURCE_REPO, "revision": SOURCE_REVISION, "files": files}


def expected_language_config(variant: str) -> dict[str, Any]:
    identity = VARIANTS[variant]
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "attention_dropout": 0.0,
        "hidden_size": identity["hidden_size"],
        "hidden_act": "silu",
        "intermediate_size": identity["intermediate_size"],
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "initializer_range": 0.02,
        "layer_types": ["full_attention"] * 36,
        "vocab_size": 151936,
        "max_position_embeddings": 40960,
        "max_window_layers": 36,
        "model_type": "qwen3",
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1.0e-6,
        "attention_bias": False,
        "rope_scaling": None,
        "sliding_window": None,
        "use_cache": True,
        "use_sliding_window": False,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
    }


def expected_api_language_config(variant: str) -> dict[str, Any]:
    expected = expected_language_config(variant)
    del expected["rope_theta"]
    expected["rope_parameters"] = {"rope_theta": 1000000, "rope_type": "default"}
    return expected


def expected_audio_config() -> dict[str, Any]:
    return {
        "_attn_implementation": "eager",
        "activation_dropout": 0.0,
        "activation_function": "gelu",
        "attention_dropout": 0.1,
        "d_model": 1280,
        "deepstack_encoder_layer_indexes": [8, 16, 24],
        "downsample_hidden_size": 480,
        "downsample_rate": 8,
        "dropout": 0.1,
        "encoder_attention_heads": 20,
        "encoder_attention_window_size": 100,
        "encoder_ffn_dim": 5120,
        "encoder_layers": 32,
        "layer_norm_eps": 1.0e-5,
        "max_source_positions": 1500,
        "num_mel_bins": 128,
        "output_dim": 1280,
        "pretrained_path": "",
    }


def expected_processor_config() -> dict[str, Any]:
    return {
        "processor_class": "MossAudioProcessor",
        "auto_map": {"AutoProcessor": "processing_moss_audio.MossAudioProcessor"},
        "mel_config": {
            "mel_sr": 16000,
            "mel_dim": 128,
            "mel_n_fft": 400,
            "mel_hop_length": 160,
            "mel_dtype": "bfloat16",
            "use_whisper_feature_extractor": True,
        },
        "enable_time_marker": True,
        "audio_token_id": 151654,
        "audio_start_id": 151669,
        "audio_end_id": 151670,
    }


def validate_config_topology(config: dict[str, Any], variant: str) -> None:
    expected_root = {
        "adapter_hidden_size": 8192,
        "architectures": ["MossAudioModel"],
        "auto_map": {
            "AutoConfig": "configuration_moss_audio.MossAudioConfig",
            "AutoProcessor": "processing_moss_audio.MossAudioProcessor",
        },
        "bos_token_id": 151643,
        "deepstack_num_inject_layers": 3,
        "dtype": "bfloat16",
        "eos_token_id": 151645,
        "ignore_index": -100,
        "model_type": "moss_audio",
        "num_hidden_layers": 36,
        "tie_word_embeddings": False,
        "transformers_version": "4.57.1",
        "vocab_size": 151936,
    }
    for key, expected in expected_root.items():
        if config.get(key) != expected:
            raise ValueError(f"{variant} root config.{key} topology metadata drifted")
    audio_config = config.get("audio_config")
    if not isinstance(audio_config, dict):
        raise ValueError("MOSS-Audio audio_config is not an object")
    for key, expected in expected_audio_config().items():
        if audio_config.get(key) != expected:
            raise ValueError(f"{variant} audio_config.{key} topology metadata drifted")
    if config.get("model_type") != "moss_audio" or config.get("architectures") != ["MossAudioModel"]:
        raise ValueError("MOSS-Audio config model identity is not exact")
    language_config = config.get("language_config")
    if not isinstance(language_config, dict):
        raise ValueError("MOSS-Audio language_config is not an object")
    for key, expected in expected_language_config(variant).items():
        if language_config.get(key) != expected:
            raise ValueError(f"{variant} language_config.{key} topology metadata drifted")
    if "hidden_size" in config or "intermediate_size" in config:
        raise ValueError("MOSS-Audio topology must be nested under language_config")


def validate_api_config(config: Any, config_class: type[Any], config_dict: dict[str, Any], variant: str) -> None:
    """Validate the normalized object returned by MossAudioConfig.from_pretrained."""
    if not isinstance(config, config_class):
        raise TypeError("official MOSS-Audio config object has the wrong class")
    if not isinstance(config_dict, dict):
        raise TypeError("official MOSS-Audio API config did not expose an object")
    expected_root = {
        "adapter_hidden_size": 8192,
        "auto_map": {
            "AutoConfig": "configuration_moss_audio.MossAudioConfig",
            "AutoProcessor": "processing_moss_audio.MossAudioProcessor",
        },
        "bos_token_id": 151643,
        "deepstack_num_inject_layers": 3,
        "eos_token_id": 151645,
        "ignore_index": -100,
        "model_type": "moss_audio",
        "num_hidden_layers": 36,
        "tie_word_embeddings": False,
        "vocab_size": 151936,
    }
    for key, expected in expected_root.items():
        if config_dict.get(key) != expected:
            raise ValueError(f"{variant} API config.{key} propagated metadata drifted")
    normalized_root = {
        "architectures": None,
        "dtype": None,
        "transformers_version": "5.10.4",
    }
    for key, expected in normalized_root.items():
        if key not in config_dict or config_dict[key] != expected:
            raise ValueError(f"{variant} API config.{key} normalization drifted")
    audio_config = config_dict.get("audio_config")
    if not isinstance(audio_config, dict):
        raise ValueError("MOSS-Audio API audio_config is not an object")
    for key, expected in expected_audio_config().items():
        if audio_config.get(key) != expected:
            raise ValueError(f"{variant} API audio_config.{key} topology metadata drifted")
    language_config = config_dict.get("language_config")
    if not isinstance(language_config, dict):
        raise ValueError("MOSS-Audio API language_config is not an object")
    expected_language = expected_api_language_config(variant)
    if "rope_theta" in language_config:
        raise ValueError("MOSS-Audio API language_config.rope_theta normalization drifted")
    rope_parameters = language_config.get("rope_parameters")
    if (
        not isinstance(rope_parameters, dict)
        or set(rope_parameters) != {"rope_theta", "rope_type"}
        or type(rope_parameters["rope_theta"]) is not int
        or rope_parameters["rope_theta"] != 1000000
        or rope_parameters["rope_type"] != "default"
    ):
        raise ValueError(f"{variant} API language_config.rope_parameters normalization drifted")
    for key, expected in expected_language.items():
        if language_config.get(key) != expected:
            raise ValueError(f"{variant} API language_config.{key} topology metadata drifted")
    if any(key in config_dict for key in ("hidden_size", "intermediate_size")):
        raise ValueError("MOSS-Audio API topology must be nested under language_config")


def validate_processor_config(processor_config: dict[str, Any], variant: str) -> None:
    if processor_config != expected_processor_config():
        raise ValueError(f"{variant} processor topology metadata drifted")


def is_checkpoint_path(path: Path) -> bool:
    """Return whether a transported snapshot path looks like model weights."""

    name = path.name.casefold()
    return name.endswith(MODEL_WEIGHT_SUFFIXES) or name.startswith(MODEL_WEIGHT_PREFIXES)


def verify_snapshot(snapshot: Path, variant: str) -> dict[str, Any]:
    identity = VARIANTS[variant]
    expected = {"config.json": identity["config_sha256"], **identity["metadata"]}
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise ValueError("model metadata snapshot root is not a real directory")
    entries = list(snapshot.iterdir())
    cache = snapshot / ".cache"
    if cache in entries and (cache.is_symlink() or not cache.is_dir()):
        raise ValueError("metadata snapshot transport cache is invalid")
    actual = sorted(path.name for path in entries if path.name != ".cache")
    if actual != sorted(expected):
        raise ValueError(f"metadata snapshot closure drifted: {actual}")
    nested_weights = [path for path in snapshot.rglob("*") if is_checkpoint_path(path)]
    if nested_weights:
        raise ValueError(f"checkpoint file reached model-free snapshot: {nested_weights[0]}")
    files: dict[str, Any] = {}
    for name, expected_hash in expected.items():
        path = snapshot / name
        require_regular(path, f"model metadata {name}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"model metadata hash drifted for {name}: {actual_hash}")
        files[name] = {"sha256": actual_hash, "bytes": path.stat().st_size}
    config = strict_json(snapshot / "config.json")
    if not isinstance(config, dict):
        raise ValueError("MOSS-Audio config is not an object")
    validate_config_topology(config, variant)
    processor_config = strict_json(snapshot / "processor_config.json")
    if not isinstance(processor_config, dict):
        raise ValueError("MOSS-Audio processor_config is not an object")
    validate_processor_config(processor_config, variant)
    return {"repo": identity["repo"], "revision": identity["revision"], "files": files, "model_type": config["model_type"]}


def api_probe(source: Path, snapshot: Path, variant: str, *, model_free: bool = False) -> dict[str, Any]:
    sys.path.insert(0, str(source))
    input_refusal: NonInteractiveInputRefusal | None = None
    try:
        import transformers
        if transformers.__version__ != "5.10.4":
            raise ValueError(f"Transformers runtime drifted: {transformers.__version__}")
        configuration = importlib.import_module("src.configuration_moss_audio")
        modeling = importlib.import_module("src.modeling_moss_audio")
        processing = importlib.import_module("src.processing_moss_audio")
        config_class = getattr(configuration, "MossAudioConfig")
        model_class = getattr(modeling, "MossAudioModel")
        processor_class = getattr(processing, "MossAudioProcessor")
        if not all(inspect.isclass(cls) for cls in (config_class, model_class, processor_class)):
            raise TypeError("official MOSS-Audio symbols are not classes")
        input_context = install_noninteractive_input_refusal() if model_free else nullcontext(None)
        with input_context as input_refusal:
            config = config_class.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=True)
            config_dict = config.to_dict() if hasattr(config, "to_dict") else None
            if not isinstance(config_dict, dict):
                raise TypeError("official MOSS-Audio config did not expose a JSON object")
            validate_api_config(config, config_class, config_dict, variant)
            config_signature = str(inspect.signature(config_class.__init__))
            model_signature = str(inspect.signature(model_class.__init__))
            processor_signature = str(inspect.signature(processor_class.__init__))
            from_pretrained_signature = str(inspect.signature(processor_class.from_pretrained))
            processor = processor_class.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=True)
            if processor is None or config.model_type != "moss_audio":
                raise RuntimeError("official processor/config construction returned an invalid object")
            if model_free and input_refusal.evidence()["prompt_count"] != 1:
                raise ValueError("expected exactly one trust-remote-code prompt per variant")
            result = {
                "transformers": transformers.__version__,
                "config_class": f"{config_class.__module__}.{config_class.__name__}",
                "model_class": f"{model_class.__module__}.{model_class.__name__}",
                "processor_class": f"{processor_class.__module__}.{processor_class.__name__}",
                "config_signature": config_signature,
                "model_signature": model_signature,
                "processor_signature": processor_signature,
                "processor_from_pretrained_signature": from_pretrained_signature,
                "config_construction": "PASS",
                "processor_construction": "PASS",
                "checkpoint_load": "NOT_PERFORMED",
            }
            if model_free:
                result["input_refusal"] = input_refusal.evidence()
            return result
    except Exception as exc:  # noqa: BLE001 - model-free failures carry structured prompt evidence
        if model_free:
            raise ModelFreeApiFailure(
                str(exc),
                input_refusal=input_refusal.evidence() if input_refusal is not None else {"installed": False},
            ) from None
        raise
    finally:
        if sys.path and sys.path[0] == str(source):
            sys.path.pop(0)


def run(args: argparse.Namespace) -> int:
    root = Path(args.vokra_root)
    project = Path(args.project)
    source = Path(args.source_dir)
    approval = Path(args.approval_evidence)
    output = Path(args.output)
    variants = args.variant if args.variant != "all" else "4b,8b"
    selected = variants.split(",")
    if selected != [v for v in selected if v in VARIANTS] or len(set(selected)) != len(selected):
        raise ValueError("variant selection is invalid")
    require_clean_head(root, args.expected_head)
    rows, project_hash, lock_hash = verify_project(project)
    signer, scope = verify_approval(approval, args.expected_head, selected, rows, project_hash, lock_hash)
    source_record = verify_source(source)
    variant_records: dict[str, Any] = {}
    for variant in selected:
        variant_records[variant] = verify_snapshot(Path(args.snapshot_root) / variant, variant)
    api_records: dict[str, Any] = {}
    for variant in selected:
        try:
            api_records[variant] = api_probe(source, Path(args.snapshot_root) / variant, variant)
        except Exception as exc:  # noqa: BLE001 - failure evidence is part of the contract
            evidence = {
                "format": FORMAT,
                "status": "BLOCKED_INCOMPATIBLE_API",
                "publication": "NO_UPLOAD",
                "expected_head": args.expected_head,
                "approval_signer": signer,
                "approval_scope_sha256": digest(scope),
                "approval_scope": scope,
                "source": source_record,
                "variants": variant_records,
                "project": {
                    "sha256": project_hash,
                    "lock_sha256": lock_hash,
                    "package_rows_sha256": digest(rows),
                    "packages": rows,
                },
                "api": {"variant": variant, "error_type": type(exc).__name__, "error": str(exc)},
                "checkpoint_load": "NOT_PERFORMED",
                "environment": {"python": platform.python_version(), "platform": platform.platform()},
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(evidence, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
            print(f"BLOCKED_INCOMPATIBLE_API: {exc}", file=sys.stderr)
            return 2
    evidence = {
        "format": FORMAT,
        "status": "PASS",
        "publication": "NO_UPLOAD",
        "expected_head": args.expected_head,
        "approval_signer": signer,
        "approval_scope_sha256": digest(scope),
        "approval_scope": scope,
        "source": source_record,
        "variants": variant_records,
        "project": {
            "sha256": project_hash,
            "lock_sha256": lock_hash,
            "package_rows_sha256": digest(rows),
            "packages": rows,
        },
        "api": api_records,
        "checkpoint_load": "NOT_PERFORMED",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(evidence, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
    print("MOSS_AUDIO_API_SMOKE PASS (no checkpoint load, no upload)")
    return 0


def blocked_model_free(
    args: argparse.Namespace,
    project_record: dict[str, Any],
    source_record: dict[str, Any],
    variant_records: dict[str, Any],
    stage: str,
    error: Exception,
) -> int:
    evidence = {
        "format": MODEL_FREE_FORMAT,
        "status": "BLOCKED_INCOMPATIBLE_API",
        "publication": "NO_UPLOAD",
        "expected_head": args.expected_head,
        "source": source_record,
        "variants": variant_records,
        "project": project_record,
        "failure": {"stage": stage, "error_type": type(error).__name__, "error": str(error)},
        "input_refusal": getattr(error, "input_refusal", {"installed": False}),
        "checkpoint_load": "NOT_PERFORMED",
        "approval": pending_approval(),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }
    write_model_free_evidence(Path(args.output), evidence)
    print(f"BLOCKED_INCOMPATIBLE_API ({stage}): {error}", file=sys.stderr)
    return 2


def run_model_free(args: argparse.Namespace) -> int:
    if os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1":
        raise ValueError("VOKRA_PUBLISH_ON_VAST=1 is required")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("model-free API smoke requires VAST Linux x86_64")
    root = Path(args.vokra_root)
    project = Path(args.project)
    source = Path(args.source_dir)
    snapshot_root = Path(args.snapshot_root)
    output = Path(args.output)
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise ValueError("model-free evidence output must be an absent absolute path")
    selected = ["4b", "8b"] if args.variant == "all" else [args.variant]
    if any(variant not in VARIANTS for variant in selected) or len(set(selected)) != len(selected):
        raise ValueError("variant selection is invalid")
    require_clean_head(root, args.expected_head)
    rows, project_hash, lock_hash = verify_project(project)
    try:
        source_record = verify_source(source)
    except Exception as exc:  # noqa: BLE001 - authenticated source failures are structured
        return blocked_model_free(args, {"sha256": project_hash, "lock_sha256": lock_hash, "packages": rows}, {}, {}, "source", exc)
    variant_records: dict[str, Any] = {}
    for variant in selected:
        try:
            variant_records[variant] = verify_snapshot(snapshot_root / variant, variant)
        except Exception as exc:  # noqa: BLE001 - metadata failures are structured
            return blocked_model_free(
                args,
                {"sha256": project_hash, "lock_sha256": lock_hash, "packages": rows},
                source_record,
                variant_records,
                f"metadata:{variant}",
                exc,
            )
    api_records: dict[str, Any] = {}
    try:
        for variant in selected:
            api_records[variant] = api_probe(source, snapshot_root / variant, variant, model_free=True)
    except Exception as exc:  # noqa: BLE001 - blocked evidence is part of the contract
        return blocked_model_free(
            args,
            {"sha256": project_hash, "lock_sha256": lock_hash, "packages": rows},
            source_record,
            variant_records,
            "api",
            exc,
        )
    evidence = {
        "format": MODEL_FREE_FORMAT,
        "status": "PASS_MODEL_FREE",
        "publication": "NO_UPLOAD",
        "expected_head": args.expected_head,
        "source": source_record,
        "variants": variant_records,
        "project": {"sha256": project_hash, "lock_sha256": lock_hash, "packages": rows},
        "api": api_records,
        "checkpoint_load": "NOT_PERFORMED",
        "approval": pending_approval(),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }
    write_model_free_evidence(output, evidence)
    print("MOSS_AUDIO_MODEL_FREE_API_SMOKE PASS_MODEL_FREE (no checkpoint load, no upload)")
    return 0


def closure_only(args: argparse.Namespace) -> int:
    root = Path(args.vokra_root)
    project = Path(args.project)
    approval = Path(args.approval_evidence)
    require_clean_head(root, args.expected_head)
    rows, project_hash, lock_hash = verify_project(project)
    selected = args.variant.split(",") if args.variant != "all" else ["4b", "8b"]
    signer, scope = verify_approval(approval, args.expected_head, selected, rows, project_hash, lock_hash)
    print(f"MOSS_AUDIO_API_SMOKE CLOSURE_PASS signer={signer} scope_sha256={digest(scope)}")
    return 0


def self_test() -> int:
    try:
        with tempfile.TemporaryDirectory(prefix="moss-audio-api-smoke-") as temporary:
            strict_json_path = Path(temporary) / "duplicate.json"
            strict_json_path.write_text('{"a":1,"a":2}', encoding="utf-8")
            try:
                strict_json(strict_json_path)
            except ValueError:
                pass
            else:
                raise AssertionError("duplicate JSON key accepted")
            valid_topology = {
                "model_type": "moss_audio",
                "architectures": ["MossAudioModel"],
                "adapter_hidden_size": 8192,
                "auto_map": {
                    "AutoConfig": "configuration_moss_audio.MossAudioConfig",
                    "AutoProcessor": "processing_moss_audio.MossAudioProcessor",
                },
                "bos_token_id": 151643,
                "deepstack_num_inject_layers": 3,
                "dtype": "bfloat16",
                "eos_token_id": 151645,
                "ignore_index": -100,
                "num_hidden_layers": 36,
                "tie_word_embeddings": False,
                "transformers_version": "4.57.1",
                "vocab_size": 151936,
                "audio_config": expected_audio_config(),
                "language_config": {
                    **expected_language_config("4b"),
                },
            }
            validate_config_topology(valid_topology, "4b")
            validate_processor_config(expected_processor_config(), "4b")
            class MossAudioConfigFixture:
                pass

            constructed_topology = dict(valid_topology)
            constructed_topology.update({"architectures": None, "dtype": None, "transformers_version": "5.10.4"})
            constructed_topology["language_config"] = expected_api_language_config("4b")
            validate_api_config(MossAudioConfigFixture(), MossAudioConfigFixture, constructed_topology, "4b")
            for label, mutation in (
                ("missing rope_parameters", lambda language: language.pop("rope_parameters")),
                ("tampered rope_parameters", lambda language: language["rope_parameters"].update({"rope_theta": 2_000_000})),
                ("extra rope_parameters key", lambda language: language["rope_parameters"].update({"extra": True})),
                ("retained rope_theta", lambda language: language.update({"rope_theta": 1_000_000})),
            ):
                tampered_normalized = dict(constructed_topology)
                tampered_normalized["language_config"] = dict(constructed_topology["language_config"])
                tampered_normalized["language_config"]["rope_parameters"] = dict(constructed_topology["language_config"]["rope_parameters"])
                mutation(tampered_normalized["language_config"])
                try:
                    validate_api_config(MossAudioConfigFixture(), MossAudioConfigFixture, tampered_normalized, "4b")
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"{label} accepted")
            for missing_key in ("architectures", "dtype", "transformers_version"):
                missing_normalized_key = dict(constructed_topology)
                del missing_normalized_key[missing_key]
                try:
                    validate_api_config(MossAudioConfigFixture(), MossAudioConfigFixture, missing_normalized_key, "4b")
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"missing constructed normalization key accepted: {missing_key}")
            try:
                validate_config_topology(constructed_topology, "4b")
            except ValueError:
                pass
            else:
                raise AssertionError("constructed normalized config accepted as raw transport config")
            valid_8b = dict(valid_topology)
            valid_8b["language_config"] = expected_language_config("8b")
            validate_config_topology(valid_8b, "8b")
            constructed_8b = dict(constructed_topology)
            constructed_8b["language_config"] = expected_api_language_config("8b")
            validate_api_config(MossAudioConfigFixture(), MossAudioConfigFixture, constructed_8b, "8b")
            for tampered_api_config, target_variant, label in (
                (constructed_topology, "8b", "constructed 4B config accepted as 8B"),
                (constructed_8b, "4b", "constructed 8B config accepted as 4B"),
            ):
                try:
                    validate_api_config(MossAudioConfigFixture(), MossAudioConfigFixture, tampered_api_config, target_variant)
                except ValueError:
                    pass
                else:
                    raise AssertionError(label)
            try:
                validate_api_config(object(), MossAudioConfigFixture, constructed_topology, "4b")
            except TypeError:
                pass
            else:
                raise AssertionError("wrong constructed config class accepted")
            for mismatched_config, target_variant, label in (
                (valid_topology, "8b", "4B config accepted as 8B"),
                (valid_8b, "4b", "8B config accepted as 4B"),
            ):
                try:
                    validate_config_topology(mismatched_config, target_variant)
                except ValueError:
                    pass
                else:
                    raise AssertionError(label)
            tampered_topology = dict(valid_topology)
            tampered_topology["language_config"] = dict(valid_topology["language_config"])
            tampered_topology["language_config"]["hidden_size"] = 4096
            try:
                validate_config_topology(tampered_topology, "4b")
            except ValueError:
                pass
            else:
                raise AssertionError("tampered nested language topology accepted")
            tampered_audio = dict(valid_topology)
            tampered_audio["audio_config"] = dict(valid_topology["audio_config"])
            tampered_audio["audio_config"]["downsample_rate"] = 4
            try:
                validate_config_topology(tampered_audio, "4b")
            except ValueError:
                pass
            else:
                raise AssertionError("tampered audio topology accepted")
            tampered_processor = dict(expected_processor_config())
            tampered_processor["mel_config"] = dict(expected_processor_config()["mel_config"])
            tampered_processor["mel_config"]["mel_hop_length"] = 80
            try:
                validate_processor_config(tampered_processor, "4b")
            except ValueError:
                pass
            else:
                raise AssertionError("tampered processor topology accepted")
            tampered_layer_types = dict(valid_topology)
            tampered_layer_types["language_config"] = dict(valid_topology["language_config"])
            tampered_layer_types["language_config"]["layer_types"] = ["sliding_attention"] * 36
            try:
                validate_config_topology(tampered_layer_types, "4b")
            except ValueError:
                pass
            else:
                raise AssertionError("tampered language layer topology accepted")
            root_topology = dict(valid_topology)
            root_topology["hidden_size"] = 2560
            try:
                validate_config_topology(root_topology, "4b")
            except ValueError:
                pass
            else:
                raise AssertionError("root-level language topology accepted")
            blocked_path = Path(temporary) / "blocked.json"
            blocked_evidence = {
                "format": MODEL_FREE_FORMAT,
                "status": "BLOCKED_INCOMPATIBLE_API",
                "publication": "NO_UPLOAD",
                "checkpoint_load": "NOT_PERFORMED",
                "approval": pending_approval(),
                "failure": {"stage": "metadata:4b", "error": "topology drift"},
            }
            write_model_free_evidence(blocked_path, blocked_evidence)
            before = blocked_path.read_bytes()
            try:
                write_model_free_evidence(blocked_path, blocked_evidence)
            except ValueError:
                pass
            else:
                raise AssertionError("blocked evidence output was clobbered")
            assert blocked_path.read_bytes() == before
            parsed_blocked = strict_json(blocked_path)
            assert parsed_blocked["status"] == "BLOCKED_INCOMPATIBLE_API"
            assert parsed_blocked["checkpoint_load"] == "NOT_PERFORMED"
            assert parsed_blocked["approval"]["source_license"] == "PENDING_OWNER_APPROVAL"
            assert not list(Path(temporary).glob(".blocked.json.*.tmp"))
            previous_input = builtins.input
            with install_noninteractive_input_refusal() as refusal:
                assert builtins.input is refusal
                assert refusal("The repository contains custom code. Trust remote code? [y/N]") == "n"
                assert refusal.evidence()["prompt_count"] == 1
                assert len(refusal.evidence()["prompt_digests"][0]["sha256"]) == 64
            assert builtins.input is previous_input
            with install_noninteractive_input_refusal() as refusal:
                try:
                    refusal("Unrelated question? [y/N]")
                except ValueError:
                    pass
                else:
                    raise AssertionError("unknown input prompt was accepted")
                assert refusal.evidence()["prompt_count"] == 1
                assert len(refusal.evidence()["prompt_digests"][0]["sha256"]) == 64
            assert builtins.input is previous_input
            replacement = lambda prompt="": "y"
            builtins.input = replacement
            try:
                try:
                    with install_noninteractive_input_refusal():
                        builtins.input = replacement
                except ValueError:
                    pass
                else:
                    raise AssertionError("overwritten input sentinel was not rejected")
                assert builtins.input is replacement
            finally:
                builtins.input = previous_input
        assert HEX40.fullmatch(SOURCE_REVISION)
        assert all(HEX64.fullmatch(value) for value in SOURCE_FILES.values())
        assert VARIANTS["4b"]["hidden_size"] != VARIANTS["8b"]["hidden_size"]
        assert MODEL_FREE_FORMAT.endswith("-v1")
        assert is_checkpoint_path(Path("pytorch_model.bin"))
        assert is_checkpoint_path(Path("weights/model-00001-of-00002.safetensors"))
        assert is_checkpoint_path(Path("adapter_model.pt"))
        assert is_checkpoint_path(Path("checkpoint.pth"))
        assert is_checkpoint_path(Path("weights.gguf"))
        assert is_checkpoint_path(Path("export.onnx"))
        assert not is_checkpoint_path(Path("config.json"))
        assert not is_checkpoint_path(Path("tokenizer_config.json"))
        print("moss_audio API smoke self-test PASS (stdlib-only, no model, no network)")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"moss_audio API smoke self-test FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--vokra-root")
    parser.add_argument("--project")
    parser.add_argument("--source-dir")
    parser.add_argument("--snapshot-root")
    parser.add_argument("--approval-evidence")
    parser.add_argument("--expected-head")
    parser.add_argument("--variant", choices=["4b", "8b", "all"])
    parser.add_argument("--output")
    parser.add_argument("--closure-only", action="store_true")
    parser.add_argument("--model-free", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.model_free or args.closure_only or any(value is not None for value in (args.vokra_root, args.project, args.source_dir, args.snapshot_root, args.approval_evidence, args.expected_head, args.variant, args.output)):
            parser.error("--self-test accepts no other options")
        raise SystemExit(self_test())
    if args.model_free:
        if args.closure_only or args.approval_evidence is not None:
            parser.error("--model-free does not accept --closure-only or approval evidence")
        required = (args.vokra_root, args.project, args.source_dir, args.snapshot_root, args.expected_head, args.variant, args.output)
        if any(value is None for value in required):
            parser.error("model-free requires Vokra root, project, source, snapshots, expected head, variant, and output")
        raise SystemExit(run_model_free(args))
    if args.closure_only:
        required = (args.vokra_root, args.project, args.approval_evidence, args.expected_head, args.variant)
        if any(value is None for value in required):
            parser.error("closure-only requires Vokra root, project, approval, expected head, and variant")
        raise SystemExit(closure_only(args))
    required = (args.vokra_root, args.project, args.source_dir, args.snapshot_root, args.approval_evidence, args.expected_head, args.variant, args.output)
    if any(value is None for value in required):
        parser.error("all smoke inputs are required")
    raise SystemExit(run(args))
