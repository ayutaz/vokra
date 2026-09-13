#!/usr/bin/env -S uv run --no-project --offline --python 3.12
"""Model-free Transformers compatibility probe for the pinned Zonos source.

The run path imports official Zonos modules from a source checkout without
constructing a model or reading a checkpoint. The validation path is
stdlib-only and authenticates evidence produced by a disposable VAST worker.
"""
from __future__ import annotations
import argparse
import ast
import builtins
import copy
from contextlib import contextmanager
import dataclasses
import hashlib
import io
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import typing
from pathlib import Path
from typing import Any, Iterator

SOURCE_REPOSITORY = "https://github.com/Zyphra/Zonos.git"
SOURCE_REVISION = "bc40d98e1e1ab54fc65c483be127a90e3c7c0645"
SOURCE_LICENSE_PATH = "LICENSE"
SOURCE_LICENSE_SPDX = "Apache-2.0"
SOURCE_LICENSE_BYTES = 11357
SOURCE_LICENSE_SHA256 = "58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd"
SOURCE_LICENSE_GIT_BLOB_SHA1 = "7a4a3ea2424c09fbe48d455aed1eaa94d9124835"
PROJECT_RELATIVE = "tools/parity/zonos_v0_1_reference"
PROJECT_SHA256 = "2c700b2335ae6b09d0e7744b39d3f26d30ded6608e25a36bf444f93eba9c708e"
LOCK_SHA256 = "a056d13d68926d8cc173d94853272d31b23ad6595fcc5cd1ce2bd2a6924675eb"
PROBE_RELATIVE = f"{PROJECT_RELATIVE}/transformers_compatibility.py"
FORMAT = "vokra-zonos-transformers-compatibility-v1"
PASS = "PASS_COMPATIBLE"
NO_UPLOAD = "NO_UPLOAD"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MODEL_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx")
TOKEN_ENV_NAMES = ("HF", "HF_TOKEN", "HF_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_ACCESS_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_TOKEN", "HUGGINGFACE_API_TOKEN", "HUGGING_FACE_TOKEN")
SOURCE_MODULE_FILES = {
    "zonos.model": "zonos/model.py",
    "zonos.config": "zonos/config.py",
    "zonos.conditioning": "zonos/conditioning.py",
    "zonos.autoencoder": "zonos/autoencoder.py",
    "zonos.sampling": "zonos/sampling.py",
    "zonos.backbone": "zonos/backbone/__init__.py",
}
EXPECTED_API_PARAMETERS = {
    "zonos.model.Zonos.from_local": [("cls", "POSITIONAL_OR_KEYWORD", "<class 'inspect._empty'>"), ("config_path", "POSITIONAL_OR_KEYWORD", "<class 'inspect._empty'>"), ("model_path", "POSITIONAL_OR_KEYWORD", "<class 'inspect._empty'>"), ("device", "POSITIONAL_OR_KEYWORD", "device(type='cpu')"), ("backbone", "POSITIONAL_OR_KEYWORD", "None")],
    "zonos.model.Zonos.from_pretrained": [("cls", "POSITIONAL_OR_KEYWORD", "<class 'inspect._empty'>"), ("repo_id", "POSITIONAL_OR_KEYWORD", "<class 'inspect._empty'>"), ("revision", "POSITIONAL_OR_KEYWORD", "None"), ("device", "POSITIONAL_OR_KEYWORD", "device(type='cpu')"), ("kwargs", "VAR_KEYWORD", "<class 'inspect._empty'>")],
    "zonos.model.Zonos.prepare_conditioning": [("self", "POSITIONAL_OR_KEYWORD", "<class 'inspect._empty'>"), ("cond_dict", "POSITIONAL_OR_KEYWORD", "<class 'inspect._empty'>"), ("uncond_dict", "POSITIONAL_OR_KEYWORD", "None")],
    "zonos.model.Zonos.generate": [("self", "POSITIONAL_OR_KEYWORD", "<class 'inspect._empty'>"), ("prefix_conditioning", "POSITIONAL_OR_KEYWORD", "<class 'inspect._empty'>"), ("audio_prefix_codes", "POSITIONAL_OR_KEYWORD", "None"), ("max_new_tokens", "POSITIONAL_OR_KEYWORD", "2580"), ("cfg_scale", "POSITIONAL_OR_KEYWORD", "2.0"), ("batch_size", "POSITIONAL_OR_KEYWORD", "1"), ("sampling_params", "POSITIONAL_OR_KEYWORD", "{'min_p': 0.1}"), ("progress_bar", "POSITIONAL_OR_KEYWORD", "True"), ("disable_torch_compile", "POSITIONAL_OR_KEYWORD", "False"), ("callback", "POSITIONAL_OR_KEYWORD", "None")],
    "zonos.conditioning.PrefixConditioner.forward": [("self", "POSITIONAL_OR_KEYWORD", "<class 'inspect._empty'>"), ("cond_dict", "POSITIONAL_OR_KEYWORD", "<class 'inspect._empty'>")],
    "zonos.autoencoder.DACAutoencoder.__init__": [("self", "POSITIONAL_OR_KEYWORD", "<class 'inspect._empty'>")],
    "zonos.sampling.sample_from_logits": [("logits", "POSITIONAL_OR_KEYWORD", "<class 'inspect._empty'>"), ("temperature", "POSITIONAL_OR_KEYWORD", "1.0"), ("top_p", "POSITIONAL_OR_KEYWORD", "0.0"), ("top_k", "POSITIONAL_OR_KEYWORD", "0"), ("min_p", "POSITIONAL_OR_KEYWORD", "0.0"), ("linear", "POSITIONAL_OR_KEYWORD", "0.0"), ("conf", "POSITIONAL_OR_KEYWORD", "0.0"), ("quad", "POSITIONAL_OR_KEYWORD", "0.0"), ("generated_tokens", "POSITIONAL_OR_KEYWORD", "None"), ("repetition_penalty", "POSITIONAL_OR_KEYWORD", "3.0"), ("repetition_penalty_window", "POSITIONAL_OR_KEYWORD", "2")],
}
TRANSFORMERS_IMPORT = {"status": "IMPORTED", "file": "transformers/__init__.py", "distribution": "transformers==5.10.4"}

class ProbeError(ValueError):
    """A fail-closed probe or evidence error."""

def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]

def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()

def strict_json(path: Path) -> tuple[dict[str, Any], bytes]:
    def unique_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ProbeError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProbeError(f"invalid evidence JSON: {path}") from error
    if not isinstance(value, dict):
        raise ProbeError("evidence JSON root is not an object")
    return value, raw

def regular(path: Path, label: str, nonempty: bool = True) -> None:
    if path.is_symlink() or not path.is_file() or (nonempty and path.stat().st_size == 0):
        raise ProbeError(f"{label} is missing, symlinked, or empty: {path}")

def ancestor_symlink(path: Path) -> bool:
    current = path.absolute().parent
    while True:
        if current.is_symlink():
            return True
        if current == current.parent:
            return False
        current = current.parent

def external_output(path: Path) -> None:
    if not path.is_absolute() or path == Path(path.anchor):
        raise ProbeError("evidence output must be an absolute path")
    if path.exists() or path.is_symlink() or ancestor_symlink(path):
        raise ProbeError("evidence output must be absent and free of symlink ancestry")

def overlaps(left: Path, right: Path) -> bool:
    left, right = left.resolve(), right.resolve()
    return left == right or left in right.parents or right in left.parents

def clean_head(root: Path, expected: str) -> str:
    if not HEX40.fullmatch(expected):
        raise ProbeError("expected HEAD must be lowercase 40-hex")
    actual = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if actual != expected:
        raise ProbeError(f"Vokra HEAD differs: {actual} != {expected}")
    status = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], capture_output=True, text=True, check=True).stdout
    if status:
        raise ProbeError("Vokra checkout is dirty")
    return actual

def project_identity(root: Path) -> dict[str, Any]:
    project = root / PROJECT_RELATIVE
    pyproject, lock = project / "pyproject.toml", project / "uv.lock"
    regular(pyproject, "Zonos pyproject.toml")
    regular(lock, "Zonos uv.lock")
    project_sha, lock_sha = sha256_file(pyproject), sha256_file(lock)
    if project_sha != PROJECT_SHA256 or lock_sha != LOCK_SHA256:
        raise ProbeError("dedicated Zonos project or lock identity drifted")
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    dependencies = sorted(data.get("project", {}).get("dependencies", []))
    expected = sorted(["huggingface-hub==1.5.0", "numpy==2.2.2", "safetensors==0.5.3", "setuptools==84.0.0", "torch==2.11.0", "torchaudio==2.11.0", "tqdm==4.67.1", "transformers==5.10.4"])
    if dependencies != expected:
        raise ProbeError("dedicated Zonos dependency contract drifted")
    return {"path": PROJECT_RELATIVE, "pyproject_sha256": project_sha, "uv_lock_sha256": lock_sha, "dependencies": dependencies}

def source_identity(source: Path) -> dict[str, Any]:
    if source.is_symlink() or ancestor_symlink(source) or not source.is_dir():
        raise ProbeError("Zonos source checkout is missing or symlinked")
    actual = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if actual != SOURCE_REVISION:
        raise ProbeError(f"Zonos source revision differs: {actual}")
    dirty = subprocess.run(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"], capture_output=True, text=True, check=True).stdout
    if dirty:
        raise ProbeError("Zonos source checkout is dirty")
    origin = subprocess.run(["git", "-C", str(source), "remote", "get-url", "origin"], capture_output=True, text=True, check=True).stdout.strip()
    if origin != SOURCE_REPOSITORY:
        raise ProbeError(f"Zonos source origin differs: {origin}")
    license_path = source / SOURCE_LICENSE_PATH
    regular(license_path, "Zonos source LICENSE")
    size, license_sha = license_path.stat().st_size, sha256_file(license_path)
    blob = subprocess.run(["git", "-C", str(source), "rev-parse", f"{SOURCE_REVISION}:{SOURCE_LICENSE_PATH}"], capture_output=True, text=True, check=True).stdout.strip()
    if size != SOURCE_LICENSE_BYTES or license_sha != SOURCE_LICENSE_SHA256 or blob != SOURCE_LICENSE_GIT_BLOB_SHA1:
        raise ProbeError("Zonos source LICENSE identity differs")
    for item in source.rglob("*"):
        if item.is_file() and item.suffix.casefold() in MODEL_SUFFIXES:
            raise ProbeError(f"source checkout contains a model-looking file: {item}")
    return {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION, "license": {"path": SOURCE_LICENSE_PATH, "spdx": SOURCE_LICENSE_SPDX, "bytes": size, "sha256": license_sha, "git_blob_sha1": blob}}

class AccessRefusal:
    def __init__(self) -> None:
        self.events: list[str] = []
    def fail(self, operation: str) -> None:
        self.events.append(operation)
        raise ProbeError(f"model/checkpoint access attempted: {operation}")

@contextmanager
def refuse_model_access() -> Iterator[AccessRefusal]:
    refusal = AccessRefusal()
    old_open = builtins.open
    old_path_open = Path.open
    old_io_open = io.open
    old_hf: list[tuple[Any, str, Any]] = []
    old_load: Any = None
    old_jit_load: Any = None
    old_safe_open: Any = None
    old_safe_tensor_loaders: list[tuple[Any, str, Any]] = []
    def guarded_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        name = os.fspath(file) if isinstance(file, (str, bytes, os.PathLike)) else repr(file)
        if any(str(name).casefold().endswith(suffix) for suffix in MODEL_SUFFIXES):
            refusal.fail(f"open:{name}")
        return old_open(file, *args, **kwargs)
    builtins.open = guarded_open
    def guarded_path_open(path_object: Path, *args: Any, **kwargs: Any) -> Any:
        if any(str(path_object).casefold().endswith(suffix) for suffix in MODEL_SUFFIXES):
            refusal.fail(f"pathlib.Path.open:{path_object}")
        return old_path_open(path_object, *args, **kwargs)
    def guarded_io_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        name = os.fspath(file) if isinstance(file, (str, bytes, os.PathLike)) else repr(file)
        if any(str(name).casefold().endswith(suffix) for suffix in MODEL_SUFFIXES):
            refusal.fail(f"io.open:{name}")
        return old_io_open(file, *args, **kwargs)
    Path.open = guarded_path_open
    io.open = guarded_io_open
    try:
        try:
            hub = importlib.import_module("huggingface_hub")
            for name in ("hf_hub_download", "snapshot_download", "hf_hub_url"):
                if hasattr(hub, name):
                    old_hf.append((hub, name, getattr(hub, name)))
                    setattr(hub, name, lambda *args, _name=name, **kwargs: refusal.fail(_name))
        except ImportError:
            pass
        try:
            torch = importlib.import_module("torch")
            old_load = torch.load
            torch.load = lambda *args, **kwargs: refusal.fail("torch.load")
            old_jit_load = getattr(getattr(torch, "jit", None), "load", None)
            if old_jit_load is not None:
                torch.jit.load = lambda *args, **kwargs: refusal.fail("torch.jit.load")
        except ImportError:
            pass
        try:
            safetensors = importlib.import_module("safetensors")
            old_safe_open = getattr(safetensors, "safe_open", None)
            if old_safe_open is not None:
                safetensors.safe_open = lambda *args, **kwargs: refusal.fail("safetensors.safe_open")
            try:
                safe_torch = importlib.import_module("safetensors.torch")
                for name in ("load_file", "load_model"):
                    if hasattr(safe_torch, name):
                        old_safe_tensor_loaders.append((safe_torch, name, getattr(safe_torch, name)))
                        setattr(safe_torch, name, lambda *args, _name=name, **kwargs: refusal.fail(f"safetensors.torch.{_name}"))
            except ImportError:
                pass
        except ImportError:
            pass
        yield refusal
    finally:
        builtins.open = old_open
        Path.open = old_path_open
        io.open = old_io_open
        for module, name, value in old_hf:
            setattr(module, name, value)
        if old_load is not None:
            importlib.import_module("torch").load = old_load
        if old_jit_load is not None:
            importlib.import_module("torch").jit.load = old_jit_load
        if old_safe_open is not None:
            importlib.import_module("safetensors").safe_open = old_safe_open
        for module, name, value in old_safe_tensor_loaders:
            setattr(module, name, value)

def signature_contract(callable_object: Any, required: list[str]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_object)
    except (TypeError, ValueError) as error:
        raise ProbeError(f"cannot inspect API signature: {callable_object}") from error
    parameters = list(signature.parameters.values())
    names = [parameter.name for parameter in parameters]
    if names != required:
        raise ProbeError(f"API parameter contract drifted: {names} != {required}")
    return {"parameters": [{"name": p.name, "kind": p.kind.name, "default": repr(p.default)} for p in parameters]}

def dac_signature_surface(callable_object: Any, *, positional_input: bool = False, keyword_input: str | None = None) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_object)
    except (TypeError, ValueError) as error:
        raise ProbeError(f"cannot inspect DAC API signature: {callable_object}") from error
    parameters = list(signature.parameters.values())
    records = [{"name": p.name, "kind": p.kind.name, "default": repr(p.default)} for p in parameters]
    result: dict[str, Any] = {"callable": callable(callable_object), "parameters": records}
    if positional_input:
        candidates = [p for p in parameters if p.name not in {"self", "cls"} and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        if not candidates:
            raise ProbeError("DAC encode does not accept a positional input")
        result["positional_input"] = {"name": candidates[0].name, "kind": candidates[0].kind.name}
    if keyword_input is not None:
        parameter = next((p for p in parameters if p.name == keyword_input), None)
        if parameter is None or parameter.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            raise ProbeError(f"DAC API does not accept keyword {keyword_input}")
        result["keyword_input"] = {"name": keyword_input, "kind": parameter.kind.name}
    return result

def dac_output_surface(callable_object: Any, required_field: str, expected_type_name: str) -> dict[str, Any]:
    try:
        hints = typing.get_type_hints(callable_object)
    except (NameError, TypeError, AttributeError) as error:
        raise ProbeError(f"cannot resolve DAC return type hints: {callable_object}") from error
    annotation = hints.get("return")
    if annotation is None:
        raise ProbeError(f"DAC {callable_object} has no resolved return annotation")
    candidates = list(typing.get_args(annotation)) or [annotation]
    matches = []
    for candidate in candidates:
        if not inspect.isclass(candidate) or candidate.__name__ != expected_type_name or candidate.__module__ != "transformers.models.dac.modeling_dac":
            continue
        if dataclasses.is_dataclass(candidate):
            fields = [field.name for field in dataclasses.fields(candidate)]
        else:
            fields = list(getattr(candidate, "__annotations__", {}))
        if required_field not in fields:
            raise ProbeError(f"DAC output type {expected_type_name} lacks {required_field}")
        matches.append({"identity": f"{candidate.__module__}.{candidate.__qualname__}", "fields": fields})
    if len(matches) != 1:
        raise ProbeError(f"DAC return type does not resolve to {expected_type_name}")
    return {"required_field": required_field, "types": matches}

def dac_config_surface(config_class: Any) -> dict[str, Any]:
    try:
        signature = inspect.signature(config_class.__init__)
    except (TypeError, ValueError) as error:
        raise ProbeError("cannot inspect DacConfig constructor") from error
    fields = []
    for name, expected_default in (("codebook_size", "1024"), ("sampling_rate", "16000")):
        parameter = signature.parameters.get(name)
        if parameter is None or parameter.kind is not inspect.Parameter.KEYWORD_ONLY or repr(parameter.default) != expected_default:
            raise ProbeError(f"DacConfig field contract drifted: {name}")
        fields.append({"name": name, "kind": parameter.kind.name, "default": repr(parameter.default)})
    return {"identity": f"{config_class.__module__}.{config_class.__qualname__}", "fields": fields}

def _attribute_path(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _attribute_path(node.value)
        return f"{parent}.{node.attr}" if parent is not None else None
    return None


def _assignment(tree: ast.AST, target_path: str, value_path: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        if value is not None and _attribute_path(value) == value_path and any(_attribute_path(target) == target_path for target in targets):
            return True
    return False


def _module_list_range(tree: ast.AST, target_path: str, range_path: str) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(_attribute_path(target) == target_path for target in targets):
            continue
        value = node.value
        if not isinstance(value, ast.Call) or _attribute_path(value.func) != "nn.ModuleList" or not value.args:
            continue
        for child in ast.walk(value):
            if not isinstance(child, ast.Call) or _attribute_path(child.func) != "range" or not child.args:
                continue
            if _attribute_path(child.args[0]) == range_path:
                return True
    return False


def _method_attribute_uses(class_tree: ast.ClassDef, attribute_path: str) -> list[str]:
    methods: list[str] = []
    for node in class_tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name == "__init__":
            continue
        if any(isinstance(child, ast.Attribute) and _attribute_path(child) == attribute_path for child in ast.walk(node)):
            methods.append(node.name)
    return sorted(set(methods))


def dac_quantizer_surface(modeling_module: Any, distribution: importlib.metadata.Distribution) -> dict[str, Any]:
    expected_identity = "transformers.models.dac.modeling_dac.DacResidualVectorQuantizer"
    try:
        candidate = getattr(modeling_module, "DacResidualVectorQuantizer")
    except AttributeError as error:
        raise ProbeError("Transformers DAC quantizer class is not exported by modeling_dac") from error
    if (not inspect.isclass(candidate) or candidate.__module__ != "transformers.models.dac.modeling_dac" or candidate.__qualname__ != "DacResidualVectorQuantizer"):
        raise ProbeError("Transformers DAC quantizer class identity drifted")
    source_path_raw = Path(inspect.getsourcefile(candidate) or "")
    expected_source = Path(distribution.locate_file("transformers/models/dac/modeling_dac.py"))
    if (not source_path_raw.is_file() or source_path_raw.is_symlink() or ancestor_symlink(source_path_raw)
            or source_path_raw.resolve() != expected_source.resolve()):
        raise ProbeError("Transformers DAC quantizer source is not the installed modeling_dac module")
    try:
        source = inspect.getsource(candidate)
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError) as error:
        raise ProbeError("Transformers DAC quantizer source is not introspectable") from error
    class_tree = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DacResidualVectorQuantizer"), None)
    if class_tree is None:
        raise ProbeError("Transformers DAC quantizer class source is incomplete")
    if not _assignment(class_tree, "n_codebooks", "config.n_codebooks") or not _assignment(class_tree, "self.n_codebooks", "n_codebooks"):
        raise ProbeError("Transformers DAC quantizer config.n_codebooks assignment is missing")
    if not _module_list_range(class_tree, "self.quantizers", "config.n_codebooks"):
        raise ProbeError("Transformers DAC quantizer ModuleList range contract is missing")
    methods = _method_attribute_uses(class_tree, "self.n_codebooks")
    if "forward" not in methods:
        raise ProbeError("Transformers DAC quantizer forward n_codebooks use is missing")
    return {
        "identity": expected_identity,
        "source_file": "transformers/models/dac/modeling_dac.py",
        "attribute": "n_codebooks",
        "config_source": "config.n_codebooks",
        "local_assignment": "n_codebooks",
        "self_assignment": "self.n_codebooks",
        "quantizers_module_list": "self.quantizers",
        "quantizers_range_source": "config.n_codebooks",
        "self_attribute_use_methods": methods,
    }

def api_contract(root: Path, source: Path) -> dict[str, Any]:
    sys.path.insert(0, str(root / PROJECT_RELATIVE))
    sys.path.insert(0, str(source))
    policy = importlib.import_module("import_policy")
    policy.install()
    with refuse_model_access() as refusal:
        transformers = importlib.import_module("transformers")
        transformers_dac = importlib.import_module("transformers.models.dac")
        transformers_dac_modeling = importlib.import_module("transformers.models.dac.modeling_dac")
        model = importlib.import_module("zonos.model")
        conditioning = importlib.import_module("zonos.conditioning")
        autoencoder = importlib.import_module("zonos.autoencoder")
        sampling = importlib.import_module("zonos.sampling")
        importlib.import_module("zonos.config")
        importlib.import_module("zonos.backbone")
    imports = {}
    for module_name, relative_file in SOURCE_MODULE_FILES.items():
        module = importlib.import_module(module_name)
        module_file_raw = Path(module.__file__)
        module_file = module_file_raw.resolve()
        expected_file = source / relative_file
        if module_file != expected_file.resolve() or module_file_raw.is_symlink() or ancestor_symlink(module_file_raw) or not module_file.is_file():
            raise ProbeError(f"official source module resolved outside fixed checkout: {module_name}")
        imports[module_name] = {"status": "IMPORTED", "file": relative_file}
    transformer_file = Path(transformers.__file__).resolve()
    transformers_distribution = importlib.metadata.distribution("transformers")
    distribution_file = Path(transformers_distribution.locate_file("transformers/__init__.py")).resolve()
    if transformer_file != distribution_file or transformer_file.name != "__init__.py" or transformer_file.parent.name != "transformers":
        raise ProbeError("Transformers import did not resolve to the fixed installed distribution")
    imports["transformers"] = dict(TRANSFORMERS_IMPORT)
    contracts = {
        "zonos.model.Zonos.from_local": signature_contract(model.Zonos.__dict__["from_local"].__func__, ["cls", "config_path", "model_path", "device", "backbone"]),
        "zonos.model.Zonos.from_pretrained": signature_contract(model.Zonos.__dict__["from_pretrained"].__func__, ["cls", "repo_id", "revision", "device", "kwargs"]),
        "zonos.model.Zonos.prepare_conditioning": signature_contract(model.Zonos.prepare_conditioning, ["self", "cond_dict", "uncond_dict"]),
        "zonos.model.Zonos.generate": signature_contract(model.Zonos.generate, ["self", "prefix_conditioning", "audio_prefix_codes", "max_new_tokens", "cfg_scale", "batch_size", "sampling_params", "progress_bar", "disable_torch_compile", "callback"]),
        "zonos.conditioning.PrefixConditioner.forward": signature_contract(conditioning.PrefixConditioner.forward, ["self", "cond_dict"]),
        "zonos.autoencoder.DACAutoencoder.__init__": signature_contract(autoencoder.DACAutoencoder.__init__, ["self"]),
        "zonos.sampling.sample_from_logits": signature_contract(sampling.sample_from_logits, ["logits", "temperature", "top_p", "top_k", "min_p", "linear", "conf", "quad", "generated_tokens", "repetition_penalty", "repetition_penalty_window"]),
    }
    dac_class = transformers_dac.DacModel
    quantizer_class = transformers_dac_modeling.DacResidualVectorQuantizer
    if quantizer_class.__module__ != "transformers.models.dac.modeling_dac" or quantizer_class.__qualname__ != "DacResidualVectorQuantizer":
        raise ProbeError("Transformers DAC quantizer did not resolve from modeling_dac")
    if getattr(autoencoder, "DacModel", None) is not dac_class:
        raise ProbeError("official autoencoder.DacModel is not the imported Transformers DacModel")
    dac_contract = {
        "class_identity": "transformers.models.dac.DacModel",
        "same_class_object": True,
        "from_pretrained": dac_signature_surface(dac_class.from_pretrained, positional_input=True),
        "encode": dac_signature_surface(dac_class.encode, positional_input=True) | {"output": dac_output_surface(dac_class.encode, "audio_codes", "DacEncoderOutput")},
        "decode": dac_signature_surface(dac_class.decode, keyword_input="audio_codes") | {"output": dac_output_surface(dac_class.decode, "audio_values", "DacDecoderOutput")},
        "config": dac_config_surface(transformers_dac.DacConfig),
        "quantizer": dac_quantizer_surface(transformers_dac_modeling, transformers_distribution),
        "caller_flow": {
            "encode_result": "audio_codes",
            "decode_keyword": "audio_codes",
            "decode_result": "audio_values",
            "config_codebook_size": "config.codebook_size",
            "quantizer_codebooks": "quantizer.n_codebooks",
            "config_sampling_rate": "config.sampling_rate",
        },
    }
    if refusal.events:
        raise ProbeError(f"model access events recorded: {refusal.events}")
    return {"modules": imports, "callables": contracts, "dac_api_contract": dac_contract, "constructor_calls": 0, "model_access_events": []}

def package_versions() -> dict[str, str]:
    names = ("huggingface-hub", "numpy", "safetensors", "torch", "torchaudio", "tqdm", "transformers")
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise ProbeError(f"required distribution is missing: {name}") from error
    expected = {"huggingface-hub": "1.5.0", "numpy": "2.2.2", "safetensors": "0.5.3", "torch": "2.11.0+cpu", "torchaudio": "2.11.0+cpu", "tqdm": "4.67.1", "transformers": "5.10.4"}
    if result != expected:
        raise ProbeError(f"installed versions drifted: {result}")
    return result

def caller_identity(root: Path, caller: Path) -> dict[str, str]:
    expected = root / "scripts/publish/vast-ai/run-zonos-transformers-compatibility.sh"
    if caller.resolve() != expected.resolve() or caller.is_symlink() or ancestor_symlink(caller):
        raise ProbeError("caller is not the dedicated Zonos compatibility wrapper")
    regular(caller, "caller script")
    probe = Path(__file__).resolve()
    return {"script": "scripts/publish/vast-ai/run-zonos-transformers-compatibility.sh", "script_sha256": sha256_file(caller), "probe": PROBE_RELATIVE, "probe_sha256": sha256_file(probe)}

def write_evidence(path: Path, evidence: dict[str, Any]) -> None:
    external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(evidence, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        # Hard-link publication is atomic and never replaces a concurrently
        # created destination. Both paths are in the same evidence directory.
        os.link(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    else:
        temporary.unlink(missing_ok=True)

def verify_bound_hash(path: Path, supplied: str) -> None:
    if not HEX64.fullmatch(supplied) or sha256_file(path) != supplied:
        raise ProbeError("evidence SHA-256 differs from caller binding")

def verify_safety(evidence: dict[str, Any]) -> None:
    if evidence["source_clean_after_import"] is not True or evidence["python_dont_write_bytecode"] is not True:
        raise ProbeError("compatibility evidence source/import safety is not exact")
    if evidence["model_access"] is not False or evidence["checkpoint_access"] is not False or evidence["hf_token_present"] is not False:
        raise ProbeError("compatibility evidence records forbidden access")
    if evidence["constructor_calls"] != 0 or evidence["model_access_events"] != [] or evidence["publication"] != NO_UPLOAD:
        raise ProbeError("compatibility evidence access/publication contract is not fail-closed")

def validate_dac_contract(dac: Any) -> None:
    if not isinstance(dac, dict) or set(dac) != {"class_identity", "same_class_object", "from_pretrained", "encode", "decode", "config", "quantizer", "caller_flow"} or dac["class_identity"] != "transformers.models.dac.DacModel" or dac["same_class_object"] is not True:
        raise ProbeError("DAC class identity contract is incomplete")
    valid_kinds = {kind.name for kind in inspect._ParameterKind}
    for key in ("from_pretrained", "encode", "decode"):
        record = dac[key]
        expected_keys = {"callable", "parameters"}
        if key == "from_pretrained":
            expected_keys.add("positional_input")
        if key in {"encode", "decode"}:
            expected_keys.add("output")
        if key == "encode":
            expected_keys.add("positional_input")
        if key == "decode":
            expected_keys.add("keyword_input")
        if not isinstance(record, dict) or set(record) != expected_keys or record["callable"] is not True or not isinstance(record["parameters"], list):
            raise ProbeError(f"DAC {key} signature record is malformed")
        for item in record["parameters"]:
            if not isinstance(item, dict) or set(item) != {"name", "kind", "default"} or not isinstance(item["name"], str) or item["kind"] not in valid_kinds or not isinstance(item["default"], str):
                raise ProbeError(f"DAC {key} parameter record is malformed")
        if key in {"from_pretrained", "encode"}:
            positional = record["positional_input"]
            if not isinstance(positional, dict) or set(positional) != {"name", "kind"} or positional["kind"] not in {"POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD"} or not any(item["name"] == positional["name"] and item["kind"] == positional["kind"] for item in record["parameters"]):
                raise ProbeError(f"DAC {key} positional caller contract is invalid")
        if key == "decode":
            keyword = record["keyword_input"]
            if not isinstance(keyword, dict) or set(keyword) != {"name", "kind"} or keyword.get("name") != "audio_codes" or keyword["kind"] not in {"POSITIONAL_OR_KEYWORD", "KEYWORD_ONLY"} or not any(item["name"] == "audio_codes" and item["kind"] == keyword["kind"] for item in record["parameters"]):
                raise ProbeError("DAC decode audio_codes contract is invalid")
        if key in {"encode", "decode"}:
            output = record["output"]
            expected_field = "audio_codes" if key == "encode" else "audio_values"
            expected_type = "DacEncoderOutput" if key == "encode" else "DacDecoderOutput"
            if not isinstance(output, dict) or set(output) != {"required_field", "types"} or output["required_field"] != expected_field or not isinstance(output["types"], list) or len(output["types"]) != 1:
                raise ProbeError(f"DAC {key} output contract is malformed")
            output_type = output["types"][0]
            if not isinstance(output_type, dict) or set(output_type) != {"identity", "fields"} or output_type["identity"] != f"transformers.models.dac.modeling_dac.{expected_type}" or not isinstance(output_type["fields"], list) or any(not isinstance(field, str) for field in output_type["fields"]) or expected_field not in output_type["fields"]:
                raise ProbeError(f"DAC {key} output field contract is invalid")
    config = dac["config"]
    if not isinstance(config, dict) or set(config) != {"identity", "fields"} or config["identity"] != "transformers.models.dac.configuration_dac.DacConfig" or config["fields"] != [{"name": "codebook_size", "kind": "KEYWORD_ONLY", "default": "1024"}, {"name": "sampling_rate", "kind": "KEYWORD_ONLY", "default": "16000"}]:
        raise ProbeError("DacConfig field contract is invalid")
    quantizer = dac["quantizer"]
    expected_quantizer = {
        "identity": "transformers.models.dac.modeling_dac.DacResidualVectorQuantizer",
        "source_file": "transformers/models/dac/modeling_dac.py",
        "attribute": "n_codebooks",
        "config_source": "config.n_codebooks",
        "local_assignment": "n_codebooks",
        "self_assignment": "self.n_codebooks",
        "quantizers_module_list": "self.quantizers",
        "quantizers_range_source": "config.n_codebooks",
        "self_attribute_use_methods": ["forward"],
    }
    if quantizer != expected_quantizer:
        raise ProbeError("DAC quantizer n_codebooks implementation contract is invalid")
    flow = dac["caller_flow"]
    if flow != {"encode_result": "audio_codes", "decode_keyword": "audio_codes", "decode_result": "audio_values", "config_codebook_size": "config.codebook_size", "quantizer_codebooks": "quantizer.n_codebooks", "config_sampling_rate": "config.sampling_rate"}:
        raise ProbeError("DAC caller flow contract is invalid")

def validate_environment(environment: Any) -> None:
    expected_versions = {"huggingface-hub": "1.5.0", "numpy": "2.2.2", "safetensors": "0.5.3", "setuptools": "84.0.0", "torch": "2.11.0+cpu", "torchaudio": "2.11.0+cpu", "tqdm": "4.67.1", "transformers": "5.10.4"}
    if not isinstance(environment, dict) or set(environment) != {"system", "release", "machine", "python", "sys_platform", "python_dont_write_bytecode", "package_versions"} or environment.get("system") != "Linux" or not isinstance(environment.get("release"), str) or not environment["release"] or environment.get("machine") != "x86_64" or not isinstance(environment.get("python"), str) or not re.fullmatch(r"3\.12\.[0-9]+", environment["python"]) or environment.get("sys_platform") != "linux" or environment.get("python_dont_write_bytecode") is not True or environment.get("package_versions") != expected_versions:
        raise ProbeError("compatibility evidence environment is not exact Linux x86_64")

def run_probe(args: argparse.Namespace) -> None:
    root, source, output = Path(args.vokra_root).resolve(), Path(args.source_dir), Path(args.output)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if any(os.environ.get(name) for name in TOKEN_ENV_NAMES):
        raise ProbeError("Hugging Face token environment variables must be absent")
    head = clean_head(root, args.expected_head)
    project = project_identity(root)
    source_facts = source_identity(source)
    caller = caller_identity(root, Path(args.caller_script))
    versions = package_versions()
    contracts = api_contract(root, source)
    source_after = source_identity(source)
    if source_after != source_facts:
        raise ProbeError("Zonos source checkout changed during import")
    evidence = {"schema": FORMAT, "status": PASS, "expected_head": head, "caller": caller, "source": source_facts, "project": project,
        "environment": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "python": platform.python_version(), "sys_platform": sys.platform, "python_dont_write_bytecode": True, "package_versions": versions},
                "imports": contracts["modules"], "api_contract": contracts["callables"], "dac_api_contract": contracts["dac_api_contract"], "source_clean_after_import": True,
                "python_dont_write_bytecode": True, "model_access": False, "checkpoint_access": False,
                "hf_token_present": False, "constructor_calls": 0, "model_access_events": [], "publication": NO_UPLOAD}
    write_evidence(output, evidence)
    print(f"{PASS}: evidence={output} sha256={sha256_file(output)}")

def validate_evidence(args: argparse.Namespace) -> None:
    root, path = Path(args.vokra_root).resolve(), Path(args.evidence)
    if not path.is_absolute() or ancestor_symlink(path):
        raise ProbeError("evidence must be absolute and free of symlink ancestry")
    if overlaps(root, path):
        raise ProbeError("compatibility evidence must be outside the Vokra checkout")
    regular(path, "compatibility evidence")
    verify_bound_hash(path, args.evidence_sha256)
    evidence, _ = strict_json(path)
    required = {"schema", "status", "expected_head", "caller", "source", "project", "environment", "imports", "api_contract", "dac_api_contract", "source_clean_after_import", "python_dont_write_bytecode", "model_access", "checkpoint_access", "hf_token_present", "constructor_calls", "model_access_events", "publication"}
    if set(evidence) != required or evidence["schema"] != FORMAT or evidence["status"] != PASS or evidence["expected_head"] != args.expected_head:
        raise ProbeError("compatibility evidence schema/status/HEAD is not exact")
    clean_head(root, args.expected_head)
    if evidence["project"] != project_identity(root):
        raise ProbeError("compatibility evidence project identity differs")
    source = {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION, "license": {"path": SOURCE_LICENSE_PATH, "spdx": SOURCE_LICENSE_SPDX, "bytes": SOURCE_LICENSE_BYTES, "sha256": SOURCE_LICENSE_SHA256, "git_blob_sha1": SOURCE_LICENSE_GIT_BLOB_SHA1}}
    if evidence["source"] != source:
        raise ProbeError("compatibility evidence source/license identity differs")
    caller = evidence["caller"]
    if not isinstance(caller, dict) or set(caller) != {"script", "script_sha256", "probe", "probe_sha256"}:
        raise ProbeError("compatibility caller identity is malformed")
    caller_path, probe_path = root / "scripts/publish/vast-ai/run-zonos-transformers-compatibility.sh", root / PROBE_RELATIVE
    if caller["script"] != caller_path.relative_to(root).as_posix() or caller["probe"] != PROBE_RELATIVE:
        raise ProbeError("compatibility caller paths differ")
    if caller["script_sha256"] != sha256_file(caller_path) or caller["probe_sha256"] != sha256_file(probe_path):
        raise ProbeError("compatibility caller hashes differ")
    verify_safety(evidence)
    environment = evidence["environment"]
    validate_environment(environment)
    expected_imports = {"zonos.model", "zonos.config", "zonos.conditioning", "zonos.autoencoder", "zonos.sampling", "zonos.backbone", "transformers"}
    expected_files = {name: relative for name, relative in SOURCE_MODULE_FILES.items()}
    if not isinstance(evidence["imports"], dict) or set(evidence["imports"]) != expected_imports or any(not isinstance(row, dict) or set(row) != {"status", "file"} or row.get("status") != "IMPORTED" or row.get("file") != expected_files.get(name, row.get("file")) for name, row in evidence["imports"].items() if name in SOURCE_MODULE_FILES):
        raise ProbeError("compatibility evidence imports are incomplete")
    transformer_import = evidence["imports"].get("transformers", {})
    if transformer_import != TRANSFORMERS_IMPORT:
        raise ProbeError("Transformers import evidence is incomplete")
    expected_contracts = set(EXPECTED_API_PARAMETERS)
    contracts = evidence["api_contract"]
    if not isinstance(contracts, dict) or set(contracts) != expected_contracts:
        raise ProbeError("compatibility evidence API contract is incomplete")
    for name, record in contracts.items():
        if not isinstance(record, dict) or set(record) != {"parameters"} or not isinstance(record["parameters"], list):
            raise ProbeError("compatibility API contract record is malformed")
        actual_parameters = []
        for item in record["parameters"]:
            if not isinstance(item, dict) or set(item) != {"name", "kind", "default"}:
                raise ProbeError("compatibility API parameter record is malformed")
            actual_parameters.append((item["name"], item["kind"], item["default"]))
        if actual_parameters != EXPECTED_API_PARAMETERS[name]:
            raise ProbeError(f"compatibility API parameter contract drifted: {name}")
    validate_dac_contract(evidence["dac_api_contract"])
    print("zonos Transformers compatibility evidence: PASS")

def self_test() -> None:
    def expect_error(action: Any, label: str) -> None:
        try:
            action()
        except (ProbeError, OSError, subprocess.CalledProcessError):
            return
        raise AssertionError(f"{label} was accepted")

    with tempfile.TemporaryDirectory(prefix="vokra-zonos-compat-self-test-") as temporary:
        # macOS commonly exposes TMPDIR through /var -> /private/var. Keep
        # the synthetic workspace on its physical path so the portable
        # ancestry check does not reject the platform's own temp alias; the
        # explicit symlink ancestry case below still exercises rejection.
        root = Path(temporary).resolve()
        output = root / "evidence.json"
        external_output(output)
        write_evidence(output, {"synthetic": True})
        assert output.is_file(), "published evidence disappeared before work cleanup"
        original = output.read_bytes()
        expect_error(lambda: external_output(output), "preexisting output")
        expect_error(lambda: write_evidence(output, {"synthetic": False}), "no-replace evidence publication")
        assert output.read_bytes() == original, "preexisting output was overwritten"
        symlink_target = root / "real"
        symlink_target.mkdir()
        symlink_dir = root / "link"
        symlink_dir.symlink_to(symlink_target, target_is_directory=True)
        expect_error(lambda: external_output(symlink_dir / "new.json"), "symlink ancestry")
        duplicate = root / "duplicate.json"
        duplicate.write_text('{"a":1,"a":2}\n', encoding="utf-8")
        expect_error(lambda: strict_json(duplicate), "duplicate JSON key")
        expect_error(lambda: verify_bound_hash(output, "0" * 64), "hash mismatch")
        work = root / "work"
        work.mkdir()
        write_evidence(root / "survives.json", {"synthetic": True})
        shutil.rmtree(work)
        assert (root / "survives.json").is_file(), "evidence was removed with work directory"
        safe = {"source_clean_after_import": True, "python_dont_write_bytecode": True, "model_access": False, "checkpoint_access": False, "hf_token_present": False, "constructor_calls": 0, "model_access_events": [], "publication": NO_UPLOAD}
        verify_safety(safe)
        for key, value in (("model_access", True), ("checkpoint_access", True), ("hf_token_present", True), ("publication", "UPLOAD")):
            tampered = dict(safe)
            tampered[key] = value
            expect_error(lambda tampered=tampered: verify_safety(tampered), f"tampered {key}")
        dac_safe = {
            "class_identity": "transformers.models.dac.DacModel", "same_class_object": True,
            "from_pretrained": {"callable": True, "parameters": [{"name": "model", "kind": "POSITIONAL_OR_KEYWORD", "default": "<class 'inspect._empty'>"}], "positional_input": {"name": "model", "kind": "POSITIONAL_OR_KEYWORD"}},
            "encode": {"callable": True, "parameters": [{"name": "wav", "kind": "POSITIONAL_OR_KEYWORD", "default": "<class 'inspect._empty'>"}], "positional_input": {"name": "wav", "kind": "POSITIONAL_OR_KEYWORD"}, "output": {"required_field": "audio_codes", "types": [{"identity": "transformers.models.dac.modeling_dac.DacEncoderOutput", "fields": ["audio_codes"]}]}},
            "decode": {"callable": True, "parameters": [{"name": "audio_codes", "kind": "KEYWORD_ONLY", "default": "<class 'inspect._empty'>"}], "keyword_input": {"name": "audio_codes", "kind": "KEYWORD_ONLY"}, "output": {"required_field": "audio_values", "types": [{"identity": "transformers.models.dac.modeling_dac.DacDecoderOutput", "fields": ["audio_values"]}]}},
            "config": {"identity": "transformers.models.dac.configuration_dac.DacConfig", "fields": [{"name": "codebook_size", "kind": "KEYWORD_ONLY", "default": "1024"}, {"name": "sampling_rate", "kind": "KEYWORD_ONLY", "default": "16000"}]},
            "quantizer": {
                "identity": "transformers.models.dac.modeling_dac.DacResidualVectorQuantizer",
                "source_file": "transformers/models/dac/modeling_dac.py",
                "attribute": "n_codebooks",
                "config_source": "config.n_codebooks",
                "local_assignment": "n_codebooks",
                "self_assignment": "self.n_codebooks",
                "quantizers_module_list": "self.quantizers",
                "quantizers_range_source": "config.n_codebooks",
                "self_attribute_use_methods": ["forward"],
            },
            "caller_flow": {"encode_result": "audio_codes", "decode_keyword": "audio_codes", "decode_result": "audio_values", "config_codebook_size": "config.codebook_size", "quantizer_codebooks": "quantizer.n_codebooks", "config_sampling_rate": "config.sampling_rate"},
        }
        validate_dac_contract(dac_safe)
        # Exercise the same hash-bound evidence validator used by the gate,
        # including the generated DacConfig KEYWORD_ONLY records. This stays
        # model/source-free; all identities come from the checked-in contract
        # and the current clean Vokra checkout.
        validation_root = root / "validator-evidence"
        validation_root.mkdir()
        validation_path = validation_root / "synthetic-evidence.json"
        project_root = repository_root()
        expected_head = subprocess.run(["git", "-C", str(project_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        generated_imports = {name: {"status": "IMPORTED", "file": relative} for name, relative in SOURCE_MODULE_FILES.items()}
        generated_imports["transformers"] = dict(TRANSFORMERS_IMPORT)
        generated_api = {name: {"parameters": [{"name": parameter_name, "kind": kind, "default": default} for parameter_name, kind, default in parameters]} for name, parameters in EXPECTED_API_PARAMETERS.items()}
        synthetic_evidence = {
            "schema": FORMAT,
            "status": PASS,
            "expected_head": expected_head,
            "caller": caller_identity(project_root, project_root / "scripts/publish/vast-ai/run-zonos-transformers-compatibility.sh"),
            "source": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION, "license": {"path": SOURCE_LICENSE_PATH, "spdx": SOURCE_LICENSE_SPDX, "bytes": SOURCE_LICENSE_BYTES, "sha256": SOURCE_LICENSE_SHA256, "git_blob_sha1": SOURCE_LICENSE_GIT_BLOB_SHA1}},
            "project": project_identity(project_root),
            "environment": {"system": "Linux", "release": "vast-kernel", "machine": "x86_64", "python": "3.12.9", "sys_platform": "linux", "python_dont_write_bytecode": True, "package_versions": {"huggingface-hub": "1.5.0", "numpy": "2.2.2", "safetensors": "0.5.3", "setuptools": "84.0.0", "torch": "2.11.0+cpu", "torchaudio": "2.11.0+cpu", "tqdm": "4.67.1", "transformers": "5.10.4"}},
            "imports": generated_imports,
            "api_contract": generated_api,
            "dac_api_contract": dac_safe,
            "source_clean_after_import": True,
            "python_dont_write_bytecode": True,
            "model_access": False,
            "checkpoint_access": False,
            "hf_token_present": False,
            "constructor_calls": 0,
            "model_access_events": [],
            "publication": NO_UPLOAD,
        }
        write_evidence(validation_path, synthetic_evidence)
        validate_evidence(argparse.Namespace(vokra_root=str(project_root), expected_head=expected_head, evidence=str(validation_path), evidence_sha256=sha256_file(validation_path)))
        for field in ("class_identity", "same_class_object", "caller_flow"):
            tampered = dict(dac_safe)
            tampered[field] = "tampered" if field != "same_class_object" else False
            expect_error(lambda tampered=tampered: validate_dac_contract(tampered), f"tampered DAC {field}")
        for path in (("from_pretrained", "positional_input"), ("encode", "output"), ("decode", "output"), ("config", "fields"), ("quantizer", "config_source"), ("quantizer", "self_attribute_use_methods")):
            tampered = copy.deepcopy(dac_safe)
            if path == ("from_pretrained", "positional_input"):
                tampered[path[0]][path[1]]["kind"] = "KEYWORD_ONLY"
            elif path[1] == "output":
                tampered[path[0]][path[1]]["required_field"] = "tampered"
            elif path == ("config", "fields"):
                tampered[path[0]][path[1]][0]["default"] = "0"
            elif path == ("quantizer", "self_attribute_use_methods"):
                tampered[path[0]][path[1]] = ["tampered"]
            else:
                tampered[path[0]][path[1]] = "tampered"
            expect_error(lambda tampered=tampered: validate_dac_contract(tampered), f"tampered DAC {path[0]}.{path[1]}")
        original_path_open, original_io_open = Path.open, io.open
        with refuse_model_access():
            expect_error(lambda: Path(root / "synthetic.safetensors").open("rb"), "pathlib model loader")
            expect_error(lambda: io.open(root / "synthetic.safetensors", "rb"), "io model loader")
            try:
                torch_available = importlib.util.find_spec("torch") is not None
            except ModuleNotFoundError:
                torch_available = False
            if torch_available:
                torch = importlib.import_module("torch")
                expect_error(lambda: torch.jit.load(root / "synthetic.pt"), "torch.jit model loader")
            try:
                safetensors_available = importlib.util.find_spec("safetensors.torch") is not None
            except ModuleNotFoundError:
                safetensors_available = False
            if safetensors_available:
                safe_torch = importlib.import_module("safetensors.torch")
                for loader in ("load_file", "load_model"):
                    if hasattr(safe_torch, loader):
                        expect_error(lambda loader=loader: getattr(safe_torch, loader)(root / "synthetic.safetensors"), f"safetensors.torch.{loader} model loader")
        assert Path.open is original_path_open and io.open is original_io_open, "model access guards were not restored"
        environment_safe = {"system": "Linux", "release": "vast-kernel", "machine": "x86_64", "python": "3.12.9", "sys_platform": "linux", "python_dont_write_bytecode": True, "package_versions": {"huggingface-hub": "1.5.0", "numpy": "2.2.2", "safetensors": "0.5.3", "setuptools": "84.0.0", "torch": "2.11.0+cpu", "torchaudio": "2.11.0+cpu", "tqdm": "4.67.1", "transformers": "5.10.4"}}
        validate_environment(environment_safe)
        for field, value in (("python", "3.11.9"), ("sys_platform", "darwin"), ("release", "")):
            tampered = dict(environment_safe)
            tampered[field] = value
            expect_error(lambda tampered=tampered: validate_environment(tampered), f"tampered environment {field}")
    assert overlaps(repository_root(), repository_root() / "inside.json"), "checkout overlap was not detected"
    current = subprocess.run(["git", "-C", str(repository_root()), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert HEX40.fullmatch(current)
    expect_error(lambda: clean_head(repository_root(), "0" * 40), "stale HEAD")
    assert FORMAT == "vokra-zonos-transformers-compatibility-v1"
    assert SOURCE_REVISION == "bc40d98e1e1ab54fc65c483be127a90e3c7c0645"
    print("zonos Transformers compatibility probe self-test: PASS")

def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--validate-evidence", action="store_true")
    parser.add_argument("--vokra-root")
    parser.add_argument("--source-dir")
    parser.add_argument("--caller-script")
    parser.add_argument("--expected-head")
    parser.add_argument("--output")
    parser.add_argument("--evidence")
    parser.add_argument("--evidence-sha256")
    args = parser.parse_args()
    try:
        if args.self_test:
            if any(getattr(args, name) is not None for name in ("vokra_root", "source_dir", "caller_script", "expected_head", "output", "evidence", "evidence_sha256")):
                raise ProbeError("--self-test accepts no other arguments")
            self_test()
        elif args.run:
            if any(getattr(args, name) is None for name in ("vokra_root", "source_dir", "caller_script", "expected_head", "output")):
                raise ProbeError("--run requires root, source, caller, expected HEAD, and output")
            run_probe(args)
        else:
            if any(getattr(args, name) is None for name in ("vokra_root", "expected_head", "evidence", "evidence_sha256")):
                raise ProbeError("--validate-evidence requires root, expected HEAD, evidence, and SHA-256")
            validate_evidence(args)
    except (ProbeError, OSError, subprocess.CalledProcessError) as error:
        print(f"zonos Transformers compatibility probe: ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)

if __name__ == "__main__":
    main()
