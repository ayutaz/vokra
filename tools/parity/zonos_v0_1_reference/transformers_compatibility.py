#!/usr/bin/env -S uv run --no-project --offline --python 3.12
"""Model-free Transformers compatibility probe for the pinned Zonos source.

The run path imports official Zonos modules from a source checkout without
constructing a model or reading a checkpoint. The validation path is
stdlib-only and authenticates evidence produced by a disposable VAST worker.
"""
from __future__ import annotations
import argparse
import builtins
from contextlib import contextmanager
import hashlib
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
PROJECT_SHA256 = "d1147745ce62515adfa4aaa28a896a1f8765592f77404ed3e9b8ba4a4c0d7f97"
LOCK_SHA256 = "d533b5917f820cca4bb0776b282ffa3749d89152acf94fca755dc78fdfcb82a1"
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
    expected = sorted(["huggingface-hub==1.5.0", "numpy==2.2.2", "safetensors==0.5.3", "torch==2.6.0", "torchaudio==2.6.0", "tqdm==4.67.1", "transformers==5.10.4"])
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
    old_hf: list[tuple[Any, str, Any]] = []
    old_load: Any = None
    old_safe_open: Any = None
    def guarded_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        name = os.fspath(file) if isinstance(file, (str, bytes, os.PathLike)) else repr(file)
        if any(str(name).casefold().endswith(suffix) for suffix in MODEL_SUFFIXES):
            refusal.fail(f"open:{name}")
        return old_open(file, *args, **kwargs)
    builtins.open = guarded_open
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
        except ImportError:
            pass
        try:
            safetensors = importlib.import_module("safetensors")
            old_safe_open = getattr(safetensors, "safe_open", None)
            if old_safe_open is not None:
                safetensors.safe_open = lambda *args, **kwargs: refusal.fail("safetensors.safe_open")
        except ImportError:
            pass
        yield refusal
    finally:
        builtins.open = old_open
        for module, name, value in old_hf:
            setattr(module, name, value)
        if old_load is not None:
            importlib.import_module("torch").load = old_load
        if old_safe_open is not None:
            importlib.import_module("safetensors").safe_open = old_safe_open

def signature_contract(callable_object: Any, required: list[str]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_object)
    except (TypeError, ValueError) as error:
        raise ProbeError(f"cannot inspect API signature: {callable_object}") from error
    parameters = list(signature.parameters.values())
    names = [parameter.name for parameter in parameters]
    if names != required:
        raise ProbeError(f"API parameter contract drifted: {names} != {required}")
    return {"parameters": [{"name": p.name, "kind": p.kind.name, "default": repr(p.default)} for p in parameters], "return_annotation": repr(signature.return_annotation)}

def api_contract(root: Path, source: Path) -> dict[str, Any]:
    sys.path.insert(0, str(root / PROJECT_RELATIVE))
    sys.path.insert(0, str(source))
    policy = importlib.import_module("import_policy")
    policy.install()
    with refuse_model_access() as refusal:
        transformers = importlib.import_module("transformers")
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
    imports["transformers"] = {"status": "IMPORTED", "file": str(transformer_file)}
    contracts = {
        "zonos.model.Zonos.from_local": signature_contract(model.Zonos.__dict__["from_local"].__func__, ["cls", "config_path", "model_path", "device", "backbone"]),
        "zonos.model.Zonos.from_pretrained": signature_contract(model.Zonos.__dict__["from_pretrained"].__func__, ["cls", "repo_id", "revision", "device", "kwargs"]),
        "zonos.model.Zonos.prepare_conditioning": signature_contract(model.Zonos.prepare_conditioning, ["self", "cond_dict", "uncond_dict"]),
        "zonos.model.Zonos.generate": signature_contract(model.Zonos.generate, ["self", "prefix_conditioning", "audio_prefix_codes", "max_new_tokens", "cfg_scale", "batch_size", "sampling_params", "progress_bar", "disable_torch_compile", "callback"]),
        "zonos.conditioning.PrefixConditioner.forward": signature_contract(conditioning.PrefixConditioner.forward, ["self", "cond_dict"]),
        "zonos.autoencoder.DACAutoencoder.__init__": signature_contract(autoencoder.DACAutoencoder.__init__, ["self"]),
        "zonos.sampling.sample_from_logits": signature_contract(sampling.sample_from_logits, ["logits", "temperature", "top_p", "top_k", "min_p", "linear", "conf", "quad", "generated_tokens", "repetition_penalty", "repetition_penalty_window"]),
    }
    if refusal.events:
        raise ProbeError(f"model access events recorded: {refusal.events}")
    if not str(transformer_file).endswith("/transformers/__init__.py"):
        raise ProbeError("Transformers import resolved to an unexpected module file")
    return {"modules": imports, "callables": contracts, "constructor_calls": 0, "model_access_events": []}

def package_versions() -> dict[str, str]:
    names = ("huggingface-hub", "numpy", "safetensors", "torch", "torchaudio", "tqdm", "transformers")
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise ProbeError(f"required distribution is missing: {name}") from error
    expected = {"huggingface-hub": "1.5.0", "numpy": "2.2.2", "safetensors": "0.5.3", "torch": "2.6.0+cpu", "torchaudio": "2.6.0+cpu", "tqdm": "4.67.1", "transformers": "5.10.4"}
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
                "imports": contracts["modules"], "api_contract": contracts["callables"], "source_clean_after_import": True,
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
    required = {"schema", "status", "expected_head", "caller", "source", "project", "environment", "imports", "api_contract", "source_clean_after_import", "python_dont_write_bytecode", "model_access", "checkpoint_access", "hf_token_present", "constructor_calls", "model_access_events", "publication"}
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
    expected_versions = {"huggingface-hub": "1.5.0", "numpy": "2.2.2", "safetensors": "0.5.3", "torch": "2.6.0+cpu", "torchaudio": "2.6.0+cpu", "tqdm": "4.67.1", "transformers": "5.10.4"}
    if not isinstance(environment, dict) or environment.get("system") != "Linux" or environment.get("machine") != "x86_64" or environment.get("python_dont_write_bytecode") is not True or environment.get("package_versions") != expected_versions:
        raise ProbeError("compatibility evidence environment is not exact Linux x86_64")
    expected_imports = {"zonos.model", "zonos.config", "zonos.conditioning", "zonos.autoencoder", "zonos.sampling", "zonos.backbone", "transformers"}
    expected_files = {name: relative for name, relative in SOURCE_MODULE_FILES.items()}
    if not isinstance(evidence["imports"], dict) or set(evidence["imports"]) != expected_imports or any(not isinstance(row, dict) or set(row) != {"status", "file"} or row.get("status") != "IMPORTED" or row.get("file") != expected_files.get(name, row.get("file")) for name, row in evidence["imports"].items() if name in SOURCE_MODULE_FILES):
        raise ProbeError("compatibility evidence imports are incomplete")
    transformer_import = evidence["imports"].get("transformers", {})
    if not isinstance(transformer_import, dict) or not isinstance(transformer_import.get("file"), str) or not transformer_import["file"].endswith("/transformers/__init__.py"):
        raise ProbeError("Transformers import evidence is incomplete")
    expected_contracts = set(EXPECTED_API_PARAMETERS)
    contracts = evidence["api_contract"]
    if not isinstance(contracts, dict) or set(contracts) != expected_contracts:
        raise ProbeError("compatibility evidence API contract is incomplete")
    for name, record in contracts.items():
        if not isinstance(record, dict) or set(record) != {"parameters", "return_annotation"} or not isinstance(record["parameters"], list):
            raise ProbeError("compatibility API contract record is malformed")
        actual_parameters = []
        for item in record["parameters"]:
            if not isinstance(item, dict) or set(item) != {"name", "kind", "default"}:
                raise ProbeError("compatibility API parameter record is malformed")
            actual_parameters.append((item["name"], item["kind"], item["default"]))
        if actual_parameters != EXPECTED_API_PARAMETERS[name]:
            raise ProbeError(f"compatibility API parameter contract drifted: {name}")
    print("zonos Transformers compatibility evidence: PASS")

def self_test() -> None:
    def expect_error(action: Any, label: str) -> None:
        try:
            action()
        except (ProbeError, OSError, subprocess.CalledProcessError):
            return
        raise AssertionError(f"{label} was accepted")

    with tempfile.TemporaryDirectory(prefix="vokra-zonos-compat-self-test-") as temporary:
        root = Path(temporary)
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
