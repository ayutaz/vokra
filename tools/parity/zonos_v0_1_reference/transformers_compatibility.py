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

class ProbeError(ValueError):
    """A fail-closed probe or evidence error."""

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
    if source.is_symlink() or not source.is_dir():
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

def api_contract(root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(root / PROJECT_RELATIVE))
    policy = importlib.import_module("import_policy")
    policy.install()
    with refuse_model_access() as refusal:
        model = importlib.import_module("zonos.model")
        conditioning = importlib.import_module("zonos.conditioning")
        autoencoder = importlib.import_module("zonos.autoencoder")
        sampling = importlib.import_module("zonos.sampling")
        importlib.import_module("zonos.config")
        importlib.import_module("zonos.backbone")
    imports = {}
    for module_name in ("zonos.model", "zonos.config", "zonos.conditioning", "zonos.autoencoder", "zonos.sampling", "zonos.backbone"):
        module = importlib.import_module(module_name)
        imports[module_name] = {"status": "IMPORTED", "file": str(module.__file__)}
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
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

def run_probe(args: argparse.Namespace) -> None:
    root, source, output = Path(args.vokra_root).resolve(), Path(args.source_dir).resolve(), Path(args.output)
    if os.environ.get("HF_TOKEN") or os.environ.get("HF"):
        raise ProbeError("HF token environment variables must be absent")
    head = clean_head(root, args.expected_head)
    project = project_identity(root)
    source_facts = source_identity(source)
    caller = caller_identity(root, Path(args.caller_script))
    versions = package_versions()
    contracts = api_contract(root)
    evidence = {"schema": FORMAT, "status": PASS, "expected_head": head, "caller": caller, "source": source_facts, "project": project,
                "environment": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "python": platform.python_version(), "sys_platform": sys.platform, "package_versions": versions},
                "imports": contracts["modules"], "api_contract": contracts["callables"], "model_access": False, "checkpoint_access": False,
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
    if not HEX64.fullmatch(args.evidence_sha256) or sha256_file(path) != args.evidence_sha256:
        raise ProbeError("evidence SHA-256 differs from caller binding")
    evidence, _ = strict_json(path)
    required = {"schema", "status", "expected_head", "caller", "source", "project", "environment", "imports", "api_contract", "model_access", "checkpoint_access", "hf_token_present", "constructor_calls", "model_access_events", "publication"}
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
    if evidence["model_access"] is not False or evidence["checkpoint_access"] is not False or evidence["hf_token_present"] is not False:
        raise ProbeError("compatibility evidence records forbidden access")
    if evidence["constructor_calls"] != 0 or evidence["model_access_events"] != [] or evidence["publication"] != NO_UPLOAD:
        raise ProbeError("compatibility evidence access/publication contract is not fail-closed")
    environment = evidence["environment"]
    expected_versions = {"huggingface-hub": "1.5.0", "numpy": "2.2.2", "safetensors": "0.5.3", "torch": "2.6.0+cpu", "torchaudio": "2.6.0+cpu", "tqdm": "4.67.1", "transformers": "5.10.4"}
    if not isinstance(environment, dict) or environment.get("system") != "Linux" or environment.get("machine") != "x86_64" or environment.get("package_versions") != expected_versions:
        raise ProbeError("compatibility evidence environment is not exact Linux x86_64")
    expected_imports = {"zonos.model", "zonos.config", "zonos.conditioning", "zonos.autoencoder", "zonos.sampling", "zonos.backbone"}
    if not isinstance(evidence["imports"], dict) or set(evidence["imports"]) != expected_imports or any(not isinstance(row, dict) or set(row) != {"status", "file"} or row.get("status") != "IMPORTED" or not isinstance(row.get("file"), str) or "/zonos/" not in row["file"] for row in evidence["imports"].values()):
        raise ProbeError("compatibility evidence imports are incomplete")
    expected_contracts = {"zonos.model.Zonos.from_local", "zonos.model.Zonos.from_pretrained", "zonos.model.Zonos.prepare_conditioning", "zonos.model.Zonos.generate", "zonos.conditioning.PrefixConditioner.forward", "zonos.autoencoder.DACAutoencoder.__init__", "zonos.sampling.sample_from_logits"}
    if not isinstance(evidence["api_contract"], dict) or set(evidence["api_contract"]) != expected_contracts:
        raise ProbeError("compatibility evidence API contract is incomplete")
    print("zonos Transformers compatibility evidence: PASS")

def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="vokra-zonos-compat-self-test-") as temporary:
        output = Path(temporary) / "evidence.json"
        try:
            external_output(output)
        except ProbeError:
            pass
        else:
            raise AssertionError("self-test output precondition failed")
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
