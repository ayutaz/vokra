#!/usr/bin/env -S uv run --script
"""Model-free official Qwen3-TTS API probe.

This probe is deliberately separate from the real-weight API smoke.  It
imports the pinned QwenLM source and exercises only configuration,
tokenizer/processor construction, and wrapper API introspection.  It rejects
checkpoint files and never calls ``Qwen3TTSModel.from_pretrained``.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import importlib.machinery
import importlib.util
import inspect
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import tomllib
import types
from pathlib import Path
from typing import Any

from qwen_source_compat import (
    PATCH_TARGET as COMPATIBILITY_PATCH_TARGET,
    PATCHED_BYTES as COMPATIBILITY_PATCHED_BYTES,
    PATCHED_SHA256 as COMPATIBILITY_PATCHED_SHA256,
    patch_source_checkout,
    self_test_filesystem,
    CompatibilityPatchError,
)

SCHEMA = "vokra-qwen3-tts-model-free-api-smoke-v1"
SOURCE_REPOSITORY = "QwenLM/Qwen3-TTS"
SOURCE_URL = "https://github.com/QwenLM/Qwen3-TTS.git"
SOURCE_REVISION = "022e286b98fbec7e1e916cb940cdf532cd9f488e"
SOURCE_PACKAGE_VERSION = "0.1.1"
SOURCE_FILES = (
    "qwen_tts/__init__.py",
    "qwen_tts/core/models/configuration_qwen3_tts.py",
    "qwen_tts/core/models/processing_qwen3_tts.py",
    "qwen_tts/inference/qwen3_tts_model.py",
    "qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py",
)
VARIANTS: dict[str, dict[str, Any]] = {
    "0.6b-base": {
        "repository": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "revision": "5d83992436eae1d760afd27aff78a71d676296fc",
        "config_bytes": 4494,
        "config_sha256": "2e714c787c8edb98b05432685cddb634add2de4d4e645f653d68251ef72ba011",
        "tts_model_type": "base",
    },
    "0.6b-customvoice": {
        "repository": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "revision": "85e237c12c027371202489a0ec509ded67b5e4b5",
        "config_bytes": 4908,
        "config_sha256": "81aca2b6fac304944d8acf345272d8a9a727d5fc2e2e66b222ab4729340c7455",
        "tts_model_type": "custom_voice",
    },
    "1.7b-base": {
        "repository": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "revision": "fd4b254389122332181a7c3db7f27e918eec64e3",
        "config_bytes": 4494,
        "config_sha256": "b4f01752d15a488abde3e1ab44723ae4f4b9e68a4037257b098b3737893cc1f9",
        "tts_model_type": "base",
    },
    "1.7b-customvoice": {
        "repository": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "revision": "0c0e3051f131929182e2c023b9537f8b1c68adfe",
        "config_bytes": 4908,
        "config_sha256": "17a07f527a1c25ea30b4e023a184482a23d3e279d697b1dc81b1bde498d29cf9",
        "tts_model_type": "custom_voice",
    },
}
COMMON_ASSETS = {
    "vocab.json": (2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
    "merges.txt": (1671839, "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"),
    "tokenizer_config.json": (7344, "dc3c31c3bdaedd5016382bb3cbe07323026775ad51f5a4fb564505992ae4a670"),
    "generation_config.json": (245, "f1b90b4513f3b34c62851049e2492d7b4c5940daf1276f89c82b8ef04127f3aa"),
}
PROJECT_SHA256 = "022e792fb7862641b81a896ed9e482ddae75a34bff1a0270fb4005088ce57e1b"
LOCK_SHA256 = "865514909ea6b9253d8883fd1acabfcc1d51ad58361da6966965102bdf67bc58"
REQUIRED_DEPENDENCIES = {
    "accelerate==1.12.0", "einops==0.8.2", "librosa==1.0.0", "numpy==2.5.2",
    "soundfile==0.14.0", "torch==2.7.1", "torchaudio==2.7.1",
    "transformers==5.10.4",
}
EXPECTED_PACKAGE_VERSIONS = {
    "accelerate": "1.12.0",
    "einops": "0.8.2",
    "librosa": "1.0.0",
    "numpy": "2.5.2",
    "soundfile": "0.14.0",
    "torch": "2.7.1",
    "torchaudio": "2.7.1",
    "transformers": "5.10.4",
}
FORBIDDEN_PACKAGES = {"gradio", "onnxruntime", "protobuf", "setuptools", "sox"}
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
EXPECTED_TORCH_FAMILY = "2.7.1"
CUDA_RUNTIME_PREFIXES = ("nvidia-", "cuda-")
CUDA_RUNTIME_NAMES = {"cuda", "cudatoolkit", "cudnn"}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_OPTIONAL_MODULES = {
    "sox": "/__vokra_import_only_sox_sentinel__.py",
    "onnxruntime": "/__vokra_import_only_onnxruntime_sentinel__.py",
}
ALLOWED_OPTIONAL_METADATA = ["__file__", "__spec__"]
SOURCE_FILE_SET = set(SOURCE_FILES)
SOURCE_FILE_CLASS_PATHS = {
    "config": "qwen_tts/core/models/configuration_qwen3_tts.py",
    "processor": "qwen_tts/core/models/processing_qwen3_tts.py",
    "wrapper": "qwen_tts/inference/qwen3_tts_model.py",
}
PROJECT_RECORD_KEYS = {"project_sha256", "lock_sha256", "packages"}
PACKAGE_RECORD_KEYS = {"name", "version", "source"}
VARIANT_RECORD_KEYS = {"repository", "revision", "files", "model_type", "tts_model_type", "checkpoint_files"}
METADATA_RECORD_KEYS = {"bytes", "sha256"}
API_RECORD_KEYS = {
    "imports", "package_versions", "config_class", "processor_class", "wrapper_class",
    "config_from_pretrained", "processor_from_pretrained", "wrapper_from_pretrained",
    "wrapper_signature", "generate_voice_clone_signature", "checkpoint_load",
    "optional_sentinels", "source_facts",
}
API_SOURCE_FACT_KEYS = {"path", "bytes", "sha256"}
SENTINEL_RECORD_KEYS = {"installed", "allowed_metadata", "sentinel_file", "metadata_reads", "metadata_keys", "accesses"}
APPROVAL_KEYS = {"source_license", "model_license", "operator", "signer", "scope_sha256"}
ENVIRONMENT_KEYS = {"python", "platform", "machine"}


class ProbeError(RuntimeError):
    """A fail-closed model-free probe failure."""


class ForbiddenOptionalModuleAccessError(ProbeError):
    """A forbidden optional module was accessed during import."""


class ApiProbeFailure(ProbeError):
    """An official API import/introspection failure with sentinel evidence."""

    def __init__(self, message: str, *, sentinel_records: dict[str, dict[str, Any]]) -> None:
        super().__init__(message)
        self.sentinel_records = sentinel_records


class _ForbiddenOptionalModuleSentinel(types.ModuleType):
    def __init__(self, module_name: str, sentinel_file: str) -> None:
        super().__init__(module_name)
        self._sentinel_file = sentinel_file
        self.__spec__ = importlib.machinery.ModuleSpec(module_name, loader=None)
        self._accesses = 0
        self._metadata_reads = 0
        self._metadata_keys: list[str] = []

    def __getattribute__(self, name: str) -> Any:
        if name == "__file__":
            reads = object.__getattribute__(self, "_metadata_reads")
            object.__setattr__(self, "_metadata_reads", reads + 1)
            object.__getattribute__(self, "_metadata_keys").append(name)
            return object.__getattribute__(self, "_sentinel_file")
        if name == "__spec__":
            reads = object.__getattribute__(self, "_metadata_reads")
            object.__setattr__(self, "_metadata_reads", reads + 1)
            object.__getattribute__(self, "_metadata_keys").append(name)
            return super().__getattribute__(name)
        if name.startswith("__") and name.endswith("__"):
            return super().__getattribute__(name)
        accesses = object.__getattribute__(self, "_accesses") + 1
        object.__setattr__(self, "_accesses", accesses)
        module_name = object.__getattribute__(self, "__name__")
        raise ForbiddenOptionalModuleAccessError(f"forbidden {module_name} access: {name}")

    def __getattr__(self, name: str) -> Any:
        accesses = object.__getattribute__(self, "_accesses") + 1
        object.__setattr__(self, "_accesses", accesses)
        module_name = object.__getattribute__(self, "__name__")
        raise ForbiddenOptionalModuleAccessError(f"forbidden {module_name} access: {name}")


def optional_sentinel_records(sentinels: dict[str, _ForbiddenOptionalModuleSentinel]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for module_name, sentinel_file in FORBIDDEN_OPTIONAL_MODULES.items():
        sentinel = sentinels.get(module_name)
        records[module_name] = {
            "installed": sentinel is not None,
            "allowed_metadata": list(ALLOWED_OPTIONAL_METADATA),
            "sentinel_file": sentinel_file,
            "metadata_reads": object.__getattribute__(sentinel, "_metadata_reads") if sentinel is not None else 0,
            "metadata_keys": list(object.__getattribute__(sentinel, "_metadata_keys")) if sentinel is not None else [],
            "accesses": object.__getattribute__(sentinel, "_accesses") if sentinel is not None else 0,
        }
    return records


@contextmanager
def install_forbidden_optional_sentinels() -> Any:
    """Provide inert import-only modules and restore ``sys.modules`` exactly."""

    for module_name in FORBIDDEN_OPTIONAL_MODULES:
        if module_name in sys.modules:
            raise ProbeError(f"real or pre-existing {module_name} module is installed")
        if importlib.util.find_spec(module_name) is not None:
            raise ProbeError(f"real {module_name} package is installed")
    sentinels = {
        module_name: _ForbiddenOptionalModuleSentinel(module_name, sentinel_file)
        for module_name, sentinel_file in FORBIDDEN_OPTIONAL_MODULES.items()
    }
    for module_name, sentinel in sentinels.items():
        sys.modules[module_name] = sentinel
    try:
        yield sentinels
    finally:
        overwritten: list[str] = []
        for module_name, sentinel in sentinels.items():
            if module_name not in sys.modules:
                continue
            if sys.modules[module_name] is not sentinel:
                overwritten.append(module_name)
            del sys.modules[module_name]
        if overwritten:
            raise ProbeError(f"optional module sentinels were overwritten during API probe: {overwritten}")


def pending_approval() -> dict[str, Any]:
    return {
        "source_license": "PENDING_OWNER_APPROVAL",
        "model_license": "PENDING_OWNER_APPROVAL",
        "operator": "PENDING_OWNER_APPROVAL",
        "signer": None,
        "scope_sha256": None,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(text: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(text, object_pairs_hook=reject_duplicates)


def require_exact_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ProbeError(f"{label} keys drifted: expected={sorted(expected)} actual={actual}")


def source_fact(cls: Any, source_root: Path) -> dict[str, Any]:
    source = inspect.getsourcefile(cls)
    if source is None:
        raise ProbeError(f"cannot locate source for {cls.__name__}")
    path = Path(source).resolve()
    try:
        relative = path.relative_to(source_root.resolve()).as_posix()
    except ValueError as error:
        raise ProbeError(f"{cls.__name__} source escapes official checkout") from error
    if relative not in SOURCE_FILE_SET:
        raise ProbeError(f"{cls.__name__} source path is outside fixed source contract: {relative}")
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def validate_source_record(source: dict[str, Any]) -> None:
    require_exact_keys(source, {"repository", "url", "revision", "package_version", "files", "compatibility_patch"}, "source")
    if source["repository"] != SOURCE_REPOSITORY or source["url"] != SOURCE_URL or source["revision"] != SOURCE_REVISION:
        raise ProbeError("official source identity drifted")
    if source["package_version"] != SOURCE_PACKAGE_VERSION:
        raise ProbeError("official source package version drifted")
    files = source["files"]
    if not isinstance(files, dict) or set(files) != SOURCE_FILE_SET:
        raise ProbeError("official source file identity set drifted")
    for relative, record in files.items():
        if relative not in SOURCE_FILE_SET or not isinstance(record, dict):
            raise ProbeError(f"official source file record is malformed: {relative}")
        if relative == COMPATIBILITY_PATCH_TARGET:
            require_exact_keys(record, {"original_bytes", "original_sha256", "bytes", "sha256"}, f"source file {relative}")
            if record["original_bytes"] != 40519 or record["original_sha256"] != "844e8dd8c0182ef9c6463c874631c22ef3c5a4fd1899dd657016164cc5379628":
                raise ProbeError(f"official source original identity drifted: {relative}")
        else:
            require_exact_keys(record, {"bytes", "sha256"}, f"source file {relative}")
        if not isinstance(record["bytes"], int) or record["bytes"] <= 0 or not isinstance(record["sha256"], str) or not HEX64.fullmatch(record["sha256"]):
            raise ProbeError(f"official source file identity is malformed: {relative}")
    patch = source["compatibility_patch"]
    require_exact_keys(patch, {"status", "target", "operation", "original_bytes", "original_sha256", "patched_bytes", "patched_sha256", "replacement_count", "transformers_api"}, "compatibility patch")
    expected_patch = {
        "status": "COMPATIBILITY_PATCH_APPLIED",
        "target": COMPATIBILITY_PATCH_TARGET,
        "operation": "replace_exactly_one_decorator",
        "original_bytes": 40519,
        "original_sha256": "844e8dd8c0182ef9c6463c874631c22ef3c5a4fd1899dd657016164cc5379628",
        "patched_bytes": COMPATIBILITY_PATCHED_BYTES,
        "patched_sha256": COMPATIBILITY_PATCHED_SHA256,
        "replacement_count": 1,
        "transformers_api": "check_model_inputs(func)",
    }
    if patch != expected_patch:
        raise ProbeError("compatibility patch identity drifted")
    patch_file = files[COMPATIBILITY_PATCH_TARGET]
    if patch_file["bytes"] != COMPATIBILITY_PATCHED_BYTES or patch_file["sha256"] != COMPATIBILITY_PATCHED_SHA256:
        raise ProbeError("patched source file identity drifted")


def validate_project_record(project: dict[str, Any]) -> None:
    require_exact_keys(project, PROJECT_RECORD_KEYS, "project")
    if project["project_sha256"] != PROJECT_SHA256 or project["lock_sha256"] != LOCK_SHA256:
        raise ProbeError("Qwen3-TTS project/lock identity drifted")
    packages = project["packages"]
    if not isinstance(packages, list) or not packages:
        raise ProbeError("Qwen3-TTS locked package inventory is empty")
    seen: set[tuple[str, str, str]] = set()
    for package in packages:
        require_exact_keys(package, PACKAGE_RECORD_KEYS, "locked package")
        name, version, source = package["name"], package["version"], package["source"]
        if not isinstance(name, str) or not isinstance(version, str) or not isinstance(source, dict) or set(source) != {"registry"} or not isinstance(source["registry"], str):
            raise ProbeError("locked package identity is malformed")
        key = (name, version, source["registry"])
        if key in seen:
            raise ProbeError(f"duplicate locked package: {key}")
        seen.add(key)
    expected_packages = verify_project(Path(__file__).resolve().parent)["packages"]
    if packages != expected_packages:
        raise ProbeError("locked package inventory differs from the fixed uv.lock")


def validate_variant_record(variant: str, record: dict[str, Any]) -> None:
    require_exact_keys(record, VARIANT_RECORD_KEYS, f"variant {variant}")
    expected = VARIANTS.get(variant)
    if expected is None or record["repository"] != expected["repository"] or record["revision"] != expected["revision"]:
        raise ProbeError(f"variant identity drifted: {variant}")
    if record["model_type"] != "qwen3_tts" or record["tts_model_type"] != expected["tts_model_type"] or record["checkpoint_files"] != "NONE_PRESENT":
        raise ProbeError(f"variant checkpoint/config contract drifted: {variant}")
    files = record["files"]
    if not isinstance(files, dict) or set(files) != set(COMMON_ASSETS) | {"config.json"}:
        raise ProbeError(f"variant metadata file set drifted: {variant}")
    expected_files = {"config.json": (expected["config_bytes"], expected["config_sha256"]), **COMMON_ASSETS}
    for name, file_record in files.items():
        require_exact_keys(file_record, METADATA_RECORD_KEYS, f"{variant} metadata {name}")
        if file_record["bytes"] != expected_files[name][0] or file_record["sha256"] != expected_files[name][1]:
            raise ProbeError(f"variant metadata identity drifted: {variant}/{name}")


def validate_sentinel_records(records: dict[str, Any], label: str) -> None:
    if not isinstance(records, dict) or set(records) != set(FORBIDDEN_OPTIONAL_MODULES):
        raise ProbeError(f"{label} optional sentinel set drifted")
    for module_name, record in records.items():
        require_exact_keys(record, SENTINEL_RECORD_KEYS, f"{label} sentinel {module_name}")
        if record["installed"] is not True or record["allowed_metadata"] != ALLOWED_OPTIONAL_METADATA or record["sentinel_file"] != FORBIDDEN_OPTIONAL_MODULES[module_name] or record["accesses"] != 0:
            raise ProbeError(f"{label} optional sentinel contract drifted: {module_name}")
        if not isinstance(record["metadata_reads"], int) or record["metadata_reads"] < 0 or not isinstance(record["metadata_keys"], list) or len(record["metadata_keys"]) != record["metadata_reads"] or any(key not in ALLOWED_OPTIONAL_METADATA for key in record["metadata_keys"]):
            raise ProbeError(f"{label} optional sentinel metadata reads malformed: {module_name}")


def validate_api_record(variant: str, api: dict[str, Any], source_files: dict[str, Any]) -> None:
    require_exact_keys(api, API_RECORD_KEYS, f"API {variant}")
    expected_classes = {
        "config_class": "qwen_tts.core.models.configuration_qwen3_tts.Qwen3TTSConfig",
        "processor_class": "qwen_tts.core.models.processing_qwen3_tts.Qwen3TTSProcessor",
        "wrapper_class": "qwen_tts.inference.qwen3_tts_model.Qwen3TTSModel",
    }
    for key, expected in expected_classes.items():
        if api[key] != expected:
            raise ProbeError(f"API {variant} {key} drifted")
    if api["config_from_pretrained"] != "CALLED_LOCAL_ONLY" or api["processor_from_pretrained"] != "CALLED_LOCAL_ONLY" or api["wrapper_from_pretrained"] != "NOT_CALLED" or api["checkpoint_load"] != "NOT_PERFORMED":
        raise ProbeError(f"API {variant} checkpoint/API call contract drifted")
    if not isinstance(api["wrapper_signature"], str) or not api["wrapper_signature"] or not isinstance(api["generate_voice_clone_signature"], str) or not api["generate_voice_clone_signature"]:
        raise ProbeError(f"API {variant} signature evidence is missing")
    imports = api["imports"]
    if not isinstance(imports, list) or imports != ["qwen_tts.Qwen3TTSModel", "qwen_tts.core.models.Qwen3TTSConfig", "qwen_tts.core.models.Qwen3TTSProcessor"]:
        raise ProbeError(f"API {variant} import contract drifted")
    versions = api["package_versions"]
    if versions != EXPECTED_PACKAGE_VERSIONS:
        raise ProbeError(f"API {variant} package version evidence drifted")
    source_facts = api["source_facts"]
    if not isinstance(source_facts, dict) or set(source_facts) != set(SOURCE_FILE_CLASS_PATHS):
        raise ProbeError(f"API {variant} source fact set drifted")
    for label, relative in SOURCE_FILE_CLASS_PATHS.items():
        fact = source_facts[label]
        require_exact_keys(fact, API_SOURCE_FACT_KEYS, f"API {variant} source {label}")
        if fact["path"] != relative or fact["bytes"] != source_files[relative]["bytes"] or fact["sha256"] != source_files[relative]["sha256"]:
            raise ProbeError(f"API {variant} source identity drifted: {label}")
    validate_sentinel_records(api["optional_sentinels"], f"API {variant}")


def validate_approval(approval: dict[str, Any]) -> None:
    require_exact_keys(approval, APPROVAL_KEYS, "approval")
    if approval != pending_approval():
        raise ProbeError("owner approval state is not pending")


def validate_evidence(path: Path, expected_head: str, expected_variant: str) -> dict[str, Any]:
    require_regular(path, "model-free API smoke evidence")
    if not HEX40.fullmatch(expected_head):
        raise ProbeError("expected HEAD must be exactly 40 lowercase hex")
    if expected_variant not in {*VARIANTS, "all"}:
        raise ProbeError("expected variant scope is invalid")
    evidence = strict_json(path.read_text(encoding="utf-8"))
    require_exact_keys(evidence, {"schema", "status", "publication", "expected_head", "source", "variants", "project", "api", "optional_sentinels", "checkpoint_load", "approval", "environment"}, "evidence")
    if evidence["schema"] != SCHEMA or evidence["status"] != "PASS_MODEL_FREE" or evidence["publication"] != "NO_UPLOAD" or evidence["expected_head"] != expected_head or evidence["checkpoint_load"] != "NOT_PERFORMED":
        raise ProbeError("model-free evidence status/HEAD/checkpoint contract drifted")
    validate_source_record(evidence["source"])
    validate_project_record(evidence["project"])
    selected = list(VARIANTS) if expected_variant == "all" else [expected_variant]
    variants = evidence["variants"]
    api = evidence["api"]
    if not isinstance(variants, dict) or set(variants) != set(selected) or not isinstance(api, dict) or set(api) != set(selected):
        raise ProbeError("evidence variant scope is missing or contains unexpected variants")
    for variant in selected:
        validate_variant_record(variant, variants[variant])
        validate_api_record(variant, api[variant], evidence["source"]["files"])
    validate_sentinel_records(evidence["optional_sentinels"], "aggregate")
    validate_approval(evidence["approval"])
    require_exact_keys(evidence["environment"], ENVIRONMENT_KEYS, "environment")
    if any(not isinstance(evidence["environment"][key], str) or not evidence["environment"][key] for key in ENVIRONMENT_KEYS):
        raise ProbeError("environment evidence is malformed")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "expected_head": expected_head, "variant_scope": expected_variant}


def require_regular(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ProbeError(f"{label} is missing, symlinked, or empty: {path}")


def require_clean_head(root: Path, expected_head: str) -> None:
    if not HEX40.fullmatch(expected_head):
        raise ProbeError("expected HEAD must be exactly 40 lowercase hex")
    actual = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if actual != expected_head:
        raise ProbeError(f"Vokra checkout HEAD drifted: {actual} != {expected_head}")
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True, capture_output=True, text=True,
    ).stdout
    if status:
        raise ProbeError("Vokra checkout is dirty")


def validate_cpu_torch_closure(packages: list[dict[str, Any]]) -> None:
    """Reject PyPI/CUDA torch stacks before any official import is attempted."""
    names = {str(package.get("name", "")).casefold() for package in packages}
    forbidden_cuda = sorted(
        name for name in names
        if name.startswith(CUDA_RUNTIME_PREFIXES) or name in CUDA_RUNTIME_NAMES
    )
    if forbidden_cuda:
        raise ProbeError(f"CUDA/NVIDIA runtime packages are forbidden: {forbidden_cuda}")
    selected = [package for package in packages if package.get("name") in {"torch", "torchaudio"}]
    if len(selected) != 4 or {package.get("name") for package in selected} != {"torch", "torchaudio"}:
        raise ProbeError("uv.lock must contain both CPU-index torch/torchaudio variants")
    for package in selected:
        if package.get("source") != {"registry": PYTORCH_CPU_INDEX}:
            raise ProbeError(f"{package.get('name')} is not resolved from the explicit CPU index")
        if str(package.get("version", "")).split("+", 1)[0] != EXPECTED_TORCH_FAMILY:
            raise ProbeError("torch/torchaudio version family is not 2.7.1")


def verify_project(project: Path) -> dict[str, Any]:
    pyproject = project / "pyproject.toml"
    lock_path = project / "uv.lock"
    require_regular(pyproject, "Qwen3-TTS project")
    require_regular(lock_path, "Qwen3-TTS uv.lock")
    project_hash = sha256_file(pyproject)
    if project_hash != PROJECT_SHA256:
        raise ProbeError(f"project SHA-256 drifted: {project_hash}")
    lock_hash = sha256_file(lock_path)
    if lock_hash != LOCK_SHA256:
        raise ProbeError(f"uv.lock SHA-256 drifted: {lock_hash}")
    project_data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    dependencies = project_data.get("project", {}).get("dependencies")
    if not isinstance(dependencies, list) or set(dependencies) != REQUIRED_DEPENDENCIES:
        raise ProbeError("Qwen3-TTS dependency closure is not exact")
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    if set(lock) != {"version", "revision", "requires-python", "resolution-markers", "manifest", "package"}:
        raise ProbeError("uv.lock top-level schema drifted")
    if lock["manifest"] != {"overrides": [{"name": "setuptools", "marker": "python_full_version < '0'"}]}:
        raise ProbeError("uv.lock manifest override drifted")
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise ProbeError("uv.lock package table is malformed")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for package in packages:
        if not isinstance(package, dict):
            raise ProbeError("uv.lock package row is malformed")
        name, version, source = package.get("name"), package.get("version"), package.get("source")
        if not isinstance(name, str) or not isinstance(version, str) or not isinstance(source, dict):
            raise ProbeError("uv.lock package identity is malformed")
        if source == {"virtual": "."}:
            continue
        if set(source) != {"registry"} or source.get("registry") not in {"https://pypi.org/simple", "https://download.pytorch.org/whl/cpu"}:
            raise ProbeError(f"uv.lock package source is not approved: {name}")
        key = (name, version, source["registry"])
        if key in seen:
            raise ProbeError(f"duplicate locked package: {key}")
        seen.add(key)
        if name in FORBIDDEN_PACKAGES:
            raise ProbeError(f"forbidden package is locked: {name}")
        rows.append({"name": name, "version": version, "source": source})
    validate_cpu_torch_closure(rows)
    return {
        "project_sha256": project_hash,
        "lock_sha256": lock_hash,
        "packages": sorted(rows, key=lambda row: (row["name"], row["version"], row["source"]["registry"])),
    }


def verify_source(source: Path) -> dict[str, Any]:
    if source.is_symlink() or not source.is_dir() or not (source / ".git").is_dir() or (source / ".git").is_symlink():
        raise ProbeError("official source checkout is missing or symlinked")
    actual = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if actual != SOURCE_REVISION:
        raise ProbeError(f"official source revision drifted: {actual}")
    if subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        check=True, capture_output=True, text=True,
    ).stdout:
        raise ProbeError("official source checkout is dirty")
    metadata = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))
    version = metadata.get("project", {}).get("version")
    if version != SOURCE_PACKAGE_VERSION:
        raise ProbeError(f"official source package version drifted: {version!r}")
    files: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_FILES:
        path = source / relative
        require_regular(path, f"official source {relative}")
        if relative != COMPATIBILITY_PATCH_TARGET:
            files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    try:
        patch = patch_source_checkout(source)
    except CompatibilityPatchError as error:
        raise ProbeError(str(error)) from error
    files[COMPATIBILITY_PATCH_TARGET] = {
        "original_bytes": patch["original_bytes"],
        "original_sha256": patch["original_sha256"],
        "bytes": patch["patched_bytes"],
        "sha256": patch["patched_sha256"],
    }
    return {"repository": SOURCE_REPOSITORY, "url": SOURCE_URL, "revision": SOURCE_REVISION,
            "package_version": SOURCE_PACKAGE_VERSION, "files": files,
            "compatibility_patch": patch}


def verify_metadata(snapshot: Path, variant: str) -> dict[str, Any]:
    expected = {"config.json": (VARIANTS[variant]["config_bytes"], VARIANTS[variant]["config_sha256"])}
    expected.update(COMMON_ASSETS)
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise ProbeError(f"metadata snapshot is missing or symlinked: {snapshot}")
    entries = list(snapshot.iterdir())
    cache = snapshot / ".cache"
    if cache in entries and (cache.is_symlink() or not cache.is_dir()):
        raise ProbeError("metadata snapshot .cache is invalid")
    visible = [entry for entry in entries if entry.name != ".cache"]
    nested_weights = [
        path for path in snapshot.rglob("*")
        if path.name.endswith(".safetensors") or path.name.startswith("model-")
    ]
    if nested_weights:
        raise ProbeError(f"checkpoint file reached model-free snapshot: {nested_weights[0]}")
    if sorted(entry.name for entry in visible) != sorted(expected):
        raise ProbeError("metadata snapshot contains an extra or missing file")
    records: dict[str, dict[str, Any]] = {}
    for name, (size, expected_hash) in expected.items():
        path = snapshot / name
        require_regular(path, f"metadata {name}")
        actual = sha256_file(path)
        if path.stat().st_size != size or actual != expected_hash:
            raise ProbeError(f"metadata identity drifted for {name}")
        records[name] = {"bytes": size, "sha256": actual}
    config = strict_json((snapshot / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("model_type") != "qwen3_tts" or config.get("tts_model_type") != VARIANTS[variant]["tts_model_type"]:
        raise ProbeError(f"{variant} config contract is not exact")
    if any(path.name.endswith(".safetensors") or path.name.startswith("model-") for path in visible):
        raise ProbeError("checkpoint file reached model-free snapshot")
    return {"repository": VARIANTS[variant]["repository"], "revision": VARIANTS[variant]["revision"], "files": records,
            "model_type": config["model_type"], "tts_model_type": config["tts_model_type"],
            "checkpoint_files": "NONE_PRESENT"}


def api_probe(source: Path, snapshot: Path) -> dict[str, Any]:
    sys.path.insert(0, str(source))
    sentinels: dict[str, _ForbiddenOptionalModuleSentinel] = {}
    try:
        with install_forbidden_optional_sentinels() as sentinels:
            import qwen_tts
            from qwen_tts import Qwen3TTSModel
            from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSProcessor
            config = Qwen3TTSConfig.from_pretrained(str(snapshot), local_files_only=True)
            processor = Qwen3TTSProcessor.from_pretrained(str(snapshot), local_files_only=True)
            if processor is None or config.model_type != "qwen3_tts":
                raise ProbeError("official processor/config construction returned an invalid object")
            package_root = Path(qwen_tts.__file__).resolve().parents[1]
            if package_root != source.resolve():
                raise ProbeError(f"qwen_tts imported from unexpected path: {package_root}")
            versions = {
                name: importlib.metadata.version(name)
                for name in ("accelerate", "einops", "librosa", "numpy", "soundfile", "torch", "torchaudio", "transformers")
            }
            if versions["transformers"] != "5.10.4":
                raise ProbeError(f"Transformers runtime drifted: {versions['transformers']}")
            sentinel_records = optional_sentinel_records(sentinels)
            if any(record["accesses"] != 0 for record in sentinel_records.values()):
                raise ProbeError(f"forbidden optional module access counts: {sentinel_records}")
            return {
                "imports": [
                    "qwen_tts.Qwen3TTSModel",
                    "qwen_tts.core.models.Qwen3TTSConfig",
                    "qwen_tts.core.models.Qwen3TTSProcessor",
                ],
                "package_versions": versions,
                "config_class": f"{Qwen3TTSConfig.__module__}.{Qwen3TTSConfig.__name__}",
                "processor_class": f"{Qwen3TTSProcessor.__module__}.{Qwen3TTSProcessor.__name__}",
                "wrapper_class": f"{Qwen3TTSModel.__module__}.{Qwen3TTSModel.__name__}",
                "config_from_pretrained": "CALLED_LOCAL_ONLY",
                "processor_from_pretrained": "CALLED_LOCAL_ONLY",
                "wrapper_from_pretrained": "NOT_CALLED",
                "wrapper_signature": str(inspect.signature(Qwen3TTSModel.from_pretrained)),
                "generate_voice_clone_signature": str(inspect.signature(Qwen3TTSModel.generate_voice_clone)),
                "source_facts": {
                    "config": source_fact(Qwen3TTSConfig, source),
                    "processor": source_fact(Qwen3TTSProcessor, source),
                    "wrapper": source_fact(Qwen3TTSModel, source),
                },
                "checkpoint_load": "NOT_PERFORMED",
                "optional_sentinels": sentinel_records,
            }
    except Exception as exc:  # noqa: BLE001 - API incompatibility is evidence, not a traceback
        raise ApiProbeFailure(
            str(exc),
            sentinel_records=optional_sentinel_records(sentinels),
        ) from None
    finally:
        if sys.path and sys.path[0] == str(source):
            sys.path.pop(0)


def write_output(path: Path, evidence: dict[str, Any]) -> None:
    if not path.is_absolute():
        raise ProbeError("output must be an absolute path")
    if path.exists() or path.is_symlink():
        raise ProbeError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(evidence, stream, sort_keys=True, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run(args: argparse.Namespace) -> int:
    if os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1":
        raise ProbeError("VOKRA_PUBLISH_ON_VAST=1 is required")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ProbeError("model-free API smoke requires VAST Linux x86_64")
    root = Path(args.vokra_root)
    project = Path(args.project)
    source = Path(args.source_dir)
    snapshot_root = Path(args.snapshot_root)
    output = Path(args.output)
    require_clean_head(root, args.expected_head)
    project_record = verify_project(project)
    source_record = verify_source(source)
    variant_names = list(VARIANTS) if args.variant == "all" else [args.variant]
    metadata = {variant: verify_metadata(snapshot_root / variant, variant) for variant in variant_names}
    api: dict[str, Any] = {}
    try:
        for variant in variant_names:
            api[variant] = api_probe(source, snapshot_root / variant)
    except Exception as exc:  # noqa: BLE001 - API incompatibility is emitted atomically
        sentinel_records = getattr(exc, "sentinel_records", optional_sentinel_records({}))
        blocked = {
            "schema": SCHEMA,
            "status": "BLOCKED_INCOMPATIBLE_API",
            "publication": "NO_UPLOAD",
            "expected_head": args.expected_head,
            "source": source_record,
            "variants": metadata,
            "project": project_record,
            "api": {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "optional_sentinels": sentinel_records,
            },
            "checkpoint_load": "NOT_PERFORMED",
            "approval": pending_approval(),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
        }
        write_output(output, blocked)
        print(f"BLOCKED_INCOMPATIBLE_API: {exc}", file=sys.stderr)
        return 2
    sentinel_records = [record["optional_sentinels"] for record in api.values()]
    evidence = {
        "schema": SCHEMA,
        "status": "PASS_MODEL_FREE",
        "publication": "NO_UPLOAD",
        "expected_head": args.expected_head,
        "source": source_record,
        "variants": metadata,
        "project": project_record,
        "api": api,
        "optional_sentinels": {
            module_name: {
                "installed": all(record[module_name]["installed"] for record in sentinel_records),
                "allowed_metadata": list(ALLOWED_OPTIONAL_METADATA),
                "sentinel_file": FORBIDDEN_OPTIONAL_MODULES[module_name],
                "metadata_reads": sum(record[module_name]["metadata_reads"] for record in sentinel_records),
                "metadata_keys": [
                    key
                    for record in sentinel_records
                    for key in record[module_name]["metadata_keys"]
                ],
                "accesses": sum(record[module_name]["accesses"] for record in sentinel_records),
            }
            for module_name in FORBIDDEN_OPTIONAL_MODULES
        },
        "checkpoint_load": "NOT_PERFORMED",
        "approval": pending_approval(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }
    write_output(output, evidence)
    print("QWEN3_TTS_MODEL_FREE_API_SMOKE PASS_MODEL_FREE (no checkpoint load, no upload)")
    return 0


def self_test() -> int:
    try:
        assert all(HEX40.fullmatch(identity["revision"]) for identity in VARIANTS.values())
        assert all(HEX64.fullmatch(identity["config_sha256"]) for identity in VARIANTS.values())
        assert HEX64.fullmatch(PROJECT_SHA256) and HEX64.fullmatch(LOCK_SHA256)
        assert COMPATIBILITY_PATCHED_BYTES == 40517
        assert HEX64.fullmatch(COMPATIBILITY_PATCHED_SHA256)
        self_test_filesystem()
        lock = tomllib.loads((Path(__file__).resolve().parent / "uv.lock").read_text(encoding="utf-8"))
        validate_cpu_torch_closure(lock["package"])
        for bad in (
            [{"name": "torch", "version": "2.7.1", "source": {"registry": "https://pypi.org/simple"}}] * 4,
            [{"name": "torch", "version": "2.7.1", "source": {"registry": PYTORCH_CPU_INDEX}}] * 2
            + [{"name": "torchaudio", "version": "2.11.0", "source": {"registry": PYTORCH_CPU_INDEX}}] * 2,
            [{"name": "torch", "version": "2.7.1", "source": {"registry": PYTORCH_CPU_INDEX}}] * 2
            + [{"name": "torchaudio", "version": "2.7.1", "source": {"registry": PYTORCH_CPU_INDEX}}] * 2
            + [{"name": "nvidia-cuda-runtime", "version": "12", "source": {"registry": "https://pypi.org/simple"}}],
        ):
            try:
                validate_cpu_torch_closure(bad)
            except ProbeError:
                pass
            else:
                raise AssertionError("unsafe torch/torchaudio closure was accepted")
        with tempfile.TemporaryDirectory(prefix="qwen3-tts-model-free-self-test-") as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"x":1,"x":2}', encoding="utf-8")
            try:
                strict_json(path.read_text(encoding="utf-8"))
            except ValueError:
                pass
            else:
                raise AssertionError("duplicate JSON key accepted")
            if any(importlib.util.find_spec(name) is not None for name in FORBIDDEN_OPTIONAL_MODULES):
                raise AssertionError("real forbidden optional package is installed")
            with install_forbidden_optional_sentinels() as sentinels:
                assert all(sys.modules[name] is sentinels[name] for name in FORBIDDEN_OPTIONAL_MODULES)
                assert sentinels["sox"].__file__ == FORBIDDEN_OPTIONAL_MODULES["sox"]
                assert sentinels["onnxruntime"].__file__ == FORBIDDEN_OPTIONAL_MODULES["onnxruntime"]
                assert sentinels["sox"].__spec__.name == "sox"
                assert sentinels["onnxruntime"].__spec__.name == "onnxruntime"
                assert sentinels["sox"].__spec__.loader is None
                assert sentinels["onnxruntime"].__spec__.loader is None
                assert importlib.util.find_spec("sox").name == "sox"
                assert importlib.util.find_spec("onnxruntime").name == "onnxruntime"
                for module_name, functional_attribute in (("sox", "Transformer"), ("onnxruntime", "InferenceSession"), ("sox", "accesses"), ("sox", "sentinel_file")):
                    try:
                        getattr(sentinels[module_name], functional_attribute)
                    except ForbiddenOptionalModuleAccessError:
                        pass
                    else:
                        raise AssertionError(f"{module_name}.{functional_attribute} was allowed")
                assert object.__getattribute__(sentinels["sox"], "_metadata_reads") >= 3
                assert object.__getattribute__(sentinels["onnxruntime"], "_metadata_reads") >= 3
                assert object.__getattribute__(sentinels["sox"], "_accesses") == 3
                assert object.__getattribute__(sentinels["onnxruntime"], "_accesses") == 1
            assert all(name not in sys.modules for name in FORBIDDEN_OPTIONAL_MODULES)
            for module_name in FORBIDDEN_OPTIONAL_MODULES:
                prior = types.ModuleType(module_name)
                sys.modules[module_name] = prior
                try:
                    try:
                        with install_forbidden_optional_sentinels():
                            raise AssertionError(f"pre-existing {module_name} module was clobbered")
                    except ProbeError:
                        pass
                    assert sys.modules[module_name] is prior
                finally:
                    del sys.modules[module_name]
            try:
                with install_forbidden_optional_sentinels() as sentinels:
                    sys.modules["sox"] = types.ModuleType("replacement-sox")
                    sys.modules["onnxruntime"] = types.ModuleType("replacement-onnxruntime")
                    _ = sentinels
            except ProbeError:
                pass
            else:
                raise AssertionError("overwritten optional sentinels were accepted")
            assert all(name not in sys.modules for name in FORBIDDEN_OPTIONAL_MODULES)
            expected_head = "a" * 40
            generic_hash = "b" * 64
            source_files = {
                relative: {"bytes": 1, "sha256": generic_hash}
                for relative in SOURCE_FILES
            }
            source_files[COMPATIBILITY_PATCH_TARGET] = {
                "original_bytes": 40519,
                "original_sha256": "844e8dd8c0182ef9c6463c874631c22ef3c5a4fd1899dd657016164cc5379628",
                "bytes": COMPATIBILITY_PATCHED_BYTES,
                "sha256": COMPATIBILITY_PATCHED_SHA256,
            }
            source_record = {
                "repository": SOURCE_REPOSITORY,
                "url": SOURCE_URL,
                "revision": SOURCE_REVISION,
                "package_version": SOURCE_PACKAGE_VERSION,
                "files": source_files,
                "compatibility_patch": {
                    "status": "COMPATIBILITY_PATCH_APPLIED",
                    "target": COMPATIBILITY_PATCH_TARGET,
                    "operation": "replace_exactly_one_decorator",
                    "original_bytes": 40519,
                    "original_sha256": "844e8dd8c0182ef9c6463c874631c22ef3c5a4fd1899dd657016164cc5379628",
                    "patched_bytes": COMPATIBILITY_PATCHED_BYTES,
                    "patched_sha256": COMPATIBILITY_PATCHED_SHA256,
                    "replacement_count": 1,
                    "transformers_api": "check_model_inputs(func)",
                },
            }
            source_facts = {
                label: {"path": relative, "bytes": source_files[relative]["bytes"], "sha256": source_files[relative]["sha256"]}
                for label, relative in SOURCE_FILE_CLASS_PATHS.items()
            }
            package_versions = dict(EXPECTED_PACKAGE_VERSIONS)
            sentinel_record = {
                module: {
                    "installed": True,
                    "allowed_metadata": list(ALLOWED_OPTIONAL_METADATA),
                    "sentinel_file": sentinel,
                    "metadata_reads": 0,
                    "metadata_keys": [],
                    "accesses": 0,
                }
                for module, sentinel in FORBIDDEN_OPTIONAL_MODULES.items()
            }
            api_record = {
                "imports": ["qwen_tts.Qwen3TTSModel", "qwen_tts.core.models.Qwen3TTSConfig", "qwen_tts.core.models.Qwen3TTSProcessor"],
                "package_versions": package_versions,
                "config_class": "qwen_tts.core.models.configuration_qwen3_tts.Qwen3TTSConfig",
                "processor_class": "qwen_tts.core.models.processing_qwen3_tts.Qwen3TTSProcessor",
                "wrapper_class": "qwen_tts.inference.qwen3_tts_model.Qwen3TTSModel",
                "config_from_pretrained": "CALLED_LOCAL_ONLY",
                "processor_from_pretrained": "CALLED_LOCAL_ONLY",
                "wrapper_from_pretrained": "NOT_CALLED",
                "wrapper_signature": "(model_path)",
                "generate_voice_clone_signature": "(text)",
                "source_facts": source_facts,
                "checkpoint_load": "NOT_PERFORMED",
                "optional_sentinels": sentinel_record,
            }
            metadata_records = {}
            for variant, identity in VARIANTS.items():
                expected_files = {"config.json": (identity["config_bytes"], identity["config_sha256"]), **COMMON_ASSETS}
                metadata_records[variant] = {
                    "repository": identity["repository"],
                    "revision": identity["revision"],
                    "files": {name: {"bytes": size, "sha256": digest} for name, (size, digest) in expected_files.items()},
                    "model_type": "qwen3_tts",
                    "tts_model_type": identity["tts_model_type"],
                    "checkpoint_files": "NONE_PRESENT",
                }
            valid_evidence = {
                "schema": SCHEMA,
                "status": "PASS_MODEL_FREE",
                "publication": "NO_UPLOAD",
                "expected_head": expected_head,
                "source": source_record,
                "variants": metadata_records,
                "project": {
                    "project_sha256": PROJECT_SHA256,
                    "lock_sha256": LOCK_SHA256,
                    "packages": verify_project(Path(__file__).resolve().parent)["packages"],
                },
                "api": {variant: json.loads(json.dumps(api_record)) for variant in VARIANTS},
                "optional_sentinels": sentinel_record,
                "checkpoint_load": "NOT_PERFORMED",
                "approval": pending_approval(),
                "environment": {"python": "3.12.0", "platform": "Linux", "machine": "x86_64"},
            }
            evidence_path = Path(directory) / "valid-evidence.json"
            evidence_path.write_text(json.dumps(valid_evidence, sort_keys=True) + "\n", encoding="utf-8")
            assert validate_evidence(evidence_path, expected_head, "all")["variant_scope"] == "all"
            tamper_cases = {
                "stale-head": lambda value: value.update(expected_head="c" * 40),
                "missing-variant": lambda value: value["variants"].pop("1.7b-base"),
                "status": lambda value: value.update(status="BLOCKED_INCOMPATIBLE_API"),
                "publication": lambda value: value.update(publication="UPLOAD"),
                "checkpoint": lambda value: value.update(checkpoint_load="PERFORMED"),
                "api": lambda value: value["api"]["0.6b-base"].update(wrapper_from_pretrained="CALLED"),
                "package-version": lambda value: value["api"]["0.6b-base"]["package_versions"].update(torch="2.7.2"),
                "source": lambda value: value["source"]["files"].pop(SOURCE_FILES[0]),
                "lock": lambda value: value["project"].update(lock_sha256="0" * 64),
                "unknown": lambda value: value.update(unexpected=True),
            }
            for name, mutate in tamper_cases.items():
                tampered = json.loads(json.dumps(valid_evidence))
                mutate(tampered)
                evidence_path.write_text(json.dumps(tampered, sort_keys=True) + "\n", encoding="utf-8")
                try:
                    validate_evidence(evidence_path, expected_head, "all")
                except ProbeError:
                    pass
                else:
                    raise AssertionError(f"tampered evidence was accepted: {name}")
            evidence_path.write_text('{"schema":1,"schema":2}\n', encoding="utf-8")
            try:
                validate_evidence(evidence_path, expected_head, "all")
            except (ProbeError, ValueError):
                pass
            else:
                raise AssertionError("duplicate evidence JSON key was accepted")
            blocked_path = Path(directory) / "blocked.json"
            write_output(blocked_path, {
                "schema": SCHEMA,
                "status": "BLOCKED_INCOMPATIBLE_API",
                "publication": "NO_UPLOAD",
                "checkpoint_load": "NOT_PERFORMED",
                "api": {
                    "optional_sentinels": optional_sentinel_records({
                        name: _ForbiddenOptionalModuleSentinel(name, sentinel_file)
                        for name, sentinel_file in FORBIDDEN_OPTIONAL_MODULES.items()
                    }),
                },
                "approval": pending_approval(),
            })
            blocked = strict_json(blocked_path.read_text(encoding="utf-8"))
            assert blocked["status"] == "BLOCKED_INCOMPATIBLE_API"
            assert blocked["publication"] == "NO_UPLOAD"
            assert blocked["checkpoint_load"] == "NOT_PERFORMED"
            assert blocked["approval"]["source_license"] == "PENDING_OWNER_APPROVAL"
            assert set(blocked["api"]["optional_sentinels"]) == set(FORBIDDEN_OPTIONAL_MODULES)
            assert all(
                record["allowed_metadata"] == ALLOWED_OPTIONAL_METADATA
                for record in blocked["api"]["optional_sentinels"].values()
            )
            assert not list(Path(directory).glob(".blocked.json.*.tmp"))
        probe_source = inspect.getsource(api_probe)
        assert "Qwen3TTSModel.from_pretrained(" not in probe_source
        assert '"wrapper_from_pretrained": "NOT_CALLED"' in probe_source
        print("qwen3_tts model-free API smoke self-test PASS (no model, no network)")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"qwen3_tts model-free API smoke self-test FAIL: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate-evidence", action="store_true")
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--vokra-root", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--expected-head")
    parser.add_argument("--variant", choices=[*VARIANTS, "all"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.validate_evidence:
        if any(value is not None for value in (args.vokra_root, args.project, args.source_dir, args.snapshot_root, args.output)) or args.self_test:
            parser.error("--validate-evidence accepts only --evidence, --expected-head, and --variant")
        if args.evidence is None or args.expected_head is None or args.variant is None:
            parser.error("--validate-evidence requires --evidence, --expected-head, and --variant")
        try:
            result = validate_evidence(args.evidence, args.expected_head, args.variant)
        except (ProbeError, OSError, ValueError, UnicodeError) as error:
            print(f"qwen3_tts evidence validation: BLOCKED: {error}", file=sys.stderr)
            return 2
        print(f"QWEN3_TTS_MODEL_FREE_API_EVIDENCE VALIDATED sha256={result['sha256']} variant_scope={result['variant_scope']}")
        return 0
    if args.self_test:
        if any(value is not None for value in (args.vokra_root, args.project, args.source_dir, args.snapshot_root, args.expected_head, args.variant, args.output, args.evidence)):
            parser.error("--self-test accepts no other arguments")
        return self_test()
    required = (args.vokra_root, args.project, args.source_dir, args.snapshot_root, args.expected_head, args.variant, args.output)
    if any(value is None for value in required):
        parser.error("all production paths are required")
    try:
        return run(args)
    except (ProbeError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"qwen3_tts model-free API smoke: BLOCKED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
