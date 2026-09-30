#!/usr/bin/env -S uv run --frozen --project tools/parity/dia_1_6b_reference python
"""Probe the pinned Dia source against the Torch 2.13 security candidate.

This is deliberately a model-free probe.  It authenticates the already
checked-out upstream source, parses its dependency and public method
contracts, and (only on an explicitly marked VAST worker) imports the source
module without constructing a model or calling a loader.  A successful import
does not override the upstream dependency pin: the official source still
declares Torch/TorchAudio 2.6.0, so the candidate remains blocked until a
separate compatibility decision and real-weight CPU parity are recorded.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tomllib
from typing import Any


SOURCE_REPOSITORY = "https://github.com/nari-labs/dia.git"
SOURCE_REVISION = "2811af1c5f476b1f49f4744fabf56cf352be21e5"
SOURCE_PYPROJECT_BLOB = "dd844dd2fb0ab0c016520c4b070beaa7c159e3e1"
UPSTREAM_TORCH_PIN = "2.6.0"
UPSTREAM_TORCHAUDIO_PIN = "2.6.0"
CANDIDATE_TORCH_PIN = "2.13.0"
CANDIDATE_TORCHAUDIO_PIN = "2.11.0"
CANDIDATE_TORCHAUDIO_IN_LOCK = True
CANDIDATE_LOCK_SHA256 = "06d1f30607934c822c12fdef1db62369f2af0a72372e19d2ba782ffb95583449"
CANDIDATE_PYPROJECT_SHA256 = "4dcc396ff3f7387b4b00b32db00ad79fa38f3cf1ef7ad22e3e7f3f8f563be4eb"
SCHEMA = "vokra-dia-upstream-compatibility-probe-v1"
MIN_VAST_MEM_KIB = 60_000_000


EXPECTED_SIGNATURES = {
    "module._sample_next_token": ["logits_BCxV", "temperature", "top_p", "top_k", "audio_eos_value"],
    "Dia.__init__": ["self", "config", "compute_dtype", "device", "load_dac"],
    "Dia._encode_text": ["self", "text"],
    "Dia._decoder_step": ["self", "tokens_Bx1xC", "dec_state", "cfg_scale", "temperature", "top_p", "top_k", "current_idx"],
    "Dia.generate": [
        "self",
        "text",
        "max_tokens",
        "cfg_scale",
        "temperature",
        "top_p",
        "use_torch_compile",
        "cfg_filter_top_k",
        "audio_prompt",
        "audio_prompt_path",
        "use_cfg_filter",
        "verbose",
    ],
}


class ProbeError(RuntimeError):
    """A malformed source or candidate input."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def absolute_without_symlinks(path: Path, label: str) -> Path:
    """Return an absolute path only when every existing component is regular."""
    if not path.is_absolute():
        raise ProbeError(f"{label} path must be absolute")
    if any(component in {".", ".."} for component in path.parts):
        raise ProbeError(f"{label} path must not contain dot components")
    absolute = Path(os.path.abspath(os.fspath(path)))
    cursor = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        cursor /= component
        if cursor.is_symlink():
            raise ProbeError(f"{label} path contains a symlink component: {cursor}")
    return absolute


def vast_memtotal_kib() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            key, _, value = line.partition(":")
            if key == "MemTotal":
                return int(value.strip().split()[0])
    except (OSError, UnicodeError, ValueError, IndexError):
        pass
    raise ProbeError("VAST RAM cannot be established from /proc/meminfo")


def require_vast_runtime(*, system: str | None = None, machine: str | None = None, memory_kib: int | None = None) -> None:
    if os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1":
        raise ProbeError("runtime source import is VAST-only; set VOKRA_PUBLISH_ON_VAST=1")
    system = platform.system() if system is None else system
    machine = platform.machine() if machine is None else machine
    if system != "Linux" or machine != "x86_64":
        raise ProbeError("runtime source import requires a Linux x86_64 VAST worker")
    memory_kib = vast_memtotal_kib() if memory_kib is None else memory_kib
    if memory_kib < MIN_VAST_MEM_KIB:
        raise ProbeError(f"VAST RAM is below the {MIN_VAST_MEM_KIB}-KiB guard")


def git(source: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(source), *args], text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProbeError(f"cannot inspect upstream source git state: {exc}") from exc


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def authenticate_source(source: Path) -> tuple[dict[str, Any], str]:
    if not source.is_dir() or source.is_symlink():
        raise ProbeError("upstream source must be a regular directory")
    if git(source, "rev-parse", "HEAD") != SOURCE_REVISION:
        raise ProbeError("upstream source revision is not the authenticated Dia revision")
    origin = git(source, "remote", "get-url", "origin").removesuffix(".git").rstrip("/")
    if origin != SOURCE_REPOSITORY.removesuffix(".git"):
        raise ProbeError("upstream source origin is not the authenticated Dia repository")
    if git(source, "status", "--porcelain", "--untracked-files=all"):
        raise ProbeError("upstream source checkout is dirty")
    pyproject = source / "pyproject.toml"
    model = source / "dia" / "model.py"
    if not pyproject.is_file() or pyproject.is_symlink() or not model.is_file() or model.is_symlink():
        raise ProbeError("authenticated Dia source files are missing")
    pyproject_bytes = pyproject.read_bytes()
    if git_blob_sha1(pyproject_bytes) != SOURCE_PYPROJECT_BLOB:
        raise ProbeError("upstream pyproject blob does not match the authenticated source")
    try:
        metadata = tomllib.loads(pyproject_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ProbeError(f"upstream pyproject is not valid TOML: {exc}") from exc
    dependencies = metadata.get("project", {}).get("dependencies", [])
    if not isinstance(dependencies, list):
        raise ProbeError("upstream dependency list is malformed")
    dependency_map = {str(item).split(";", 1)[0].strip().split("==", 1)[0].lower(): str(item) for item in dependencies}
    if dependency_map.get("torch") != f"torch=={UPSTREAM_TORCH_PIN}" or dependency_map.get("torchaudio") != f"torchaudio=={UPSTREAM_TORCHAUDIO_PIN}":
        raise ProbeError("upstream Torch/TorchAudio pins drifted from the authenticated source")
    return {"revision": SOURCE_REVISION, "origin": origin, "pyproject_blob_sha1": SOURCE_PYPROJECT_BLOB}, model.read_text(encoding="utf-8")


def parameter_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    args = function.args
    positional = [*args.posonlyargs, *args.args]
    return [item.arg for item in positional] + [item.arg for item in args.kwonlyargs]


def source_signatures(source_text: str) -> dict[str, list[str]]:
    try:
        tree = ast.parse(source_text, filename="dia/model.py")
    except SyntaxError as exc:
        raise ProbeError(f"upstream Dia model source is not valid Python: {exc}") from exc
    found: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "_sample_next_token" and "module._sample_next_token" not in found:
                found["module._sample_next_token"] = parameter_names(node)
        if isinstance(node, ast.ClassDef) and node.name == "Dia":
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and f"Dia.{child.name}" in EXPECTED_SIGNATURES:
                    found[f"Dia.{child.name}"] = parameter_names(child)
    return found


def candidate_identity(project: Path) -> dict[str, Any]:
    if not project.is_dir() or project.is_symlink():
        raise ProbeError("candidate reference project is not a regular directory")
    lock = project / "uv.lock"
    pyproject = project / "pyproject.toml"
    if sha256(lock) != CANDIDATE_LOCK_SHA256 or sha256(pyproject) != CANDIDATE_PYPROJECT_SHA256:
        raise ProbeError("candidate project hash does not match the reviewed 2.13.0 candidate")
    try:
        lock_data = tomllib.loads(lock.read_text(encoding="utf-8"))
        project_data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ProbeError(f"candidate project metadata is unreadable: {exc}") from exc
    direct = project_data.get("project", {}).get("dependencies", [])
    if f"torch=={CANDIDATE_TORCH_PIN}" not in direct:
        raise ProbeError("candidate project does not pin Torch 2.13.0")
    torch_rows = [row for row in lock_data.get("package", []) if row.get("name") == "torch"]
    if {row.get("version") for row in torch_rows} != {"2.13.0", "2.13.0+cpu"}:
        raise ProbeError("candidate lock does not contain both expected Torch 2.13.0 platform rows")
    torchaudio_rows = [row for row in lock_data.get("package", []) if row.get("name") == "torchaudio"]
    if {row.get("version") for row in torchaudio_rows} != {CANDIDATE_TORCHAUDIO_PIN, f"{CANDIDATE_TORCHAUDIO_PIN}+cpu"}:
        raise ProbeError("candidate lock does not contain both expected Torchaudio 2.11.0 platform rows")
    return {"pyproject_sha256": CANDIDATE_PYPROJECT_SHA256, "uv_lock_sha256": CANDIDATE_LOCK_SHA256, "torch": CANDIDATE_TORCH_PIN, "torchaudio": CANDIDATE_TORCHAUDIO_PIN}


def classify_runtime_probe(payload: dict[str, Any]) -> dict[str, str]:
    module_status = payload.get("module_status")
    torch_status = payload.get("torch_status")
    torchaudio_status = payload.get("torchaudio_status")
    closure_status = "BLOCKED_TORCH_TORCHAUDIO_CLOSURE"
    if CANDIDATE_TORCHAUDIO_IN_LOCK and torch_status == "PASS_TORCH_IMPORT" and torchaudio_status == "PASS_TORCHAUDIO_IMPORT":
        closure_status = "PASS_TORCH_TORCHAUDIO_CLOSURE"
    if module_status != "PASS_SOURCE_IMPORT":
        status = "BLOCKED_SOURCE_IMPORT"
    elif closure_status != "PASS_TORCH_TORCHAUDIO_CLOSURE":
        status = "BLOCKED_TORCH_TORCHAUDIO_CLOSURE"
    else:
        status = "PASS_SOURCE_IMPORT_TORCH_TORCHAUDIO"
    return {"status": status, "closure_status": closure_status}


def import_probe(source: Path) -> dict[str, Any]:
    require_vast_runtime()
    environment = os.environ.copy()
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "PYTHONNOUSERSITE": "1"})
    code = (
        "import importlib, json\n"
        "facts = {}\n"
        "try:\n"
        " import torch\n"
        " facts.update(torch_status='PASS_TORCH_IMPORT', torch_version=torch.__version__)\n"
        "except Exception as exc:\n"
        " facts.update(torch_status='BLOCKED_TORCH_IMPORT', torch_error=f'{type(exc).__name__}: {exc}')\n"
        "try:\n"
        " import torchaudio\n"
        " facts.update(torchaudio_status='PASS_TORCHAUDIO_IMPORT', torchaudio_version=torchaudio.__version__)\n"
        "except Exception as exc:\n"
        " facts.update(torchaudio_status='BLOCKED_TORCHAUDIO_IMPORT', torchaudio_error=f'{type(exc).__name__}: {exc}')\n"
        "try:\n"
        " importlib.import_module('dia.model')\n"
        " facts['module_status'] = 'PASS_SOURCE_IMPORT'\n"
        "except Exception as exc:\n"
        " facts.update(module_status='BLOCKED_SOURCE_IMPORT', module_error=f'{type(exc).__name__}: {exc}')\n"
        "print(json.dumps(facts, sort_keys=True))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=source, env={**environment, "PYTHONPATH": str(source)}, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return {"status": "BLOCKED_SOURCE_IMPORT", "stderr_tail": result.stderr[-2000:], "stdout_tail": result.stdout[-2000:]}
    try:
        versions = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"status": "BLOCKED_SOURCE_IMPORT_OUTPUT", "stdout_tail": result.stdout[-2000:]}
    return {**versions, **classify_runtime_probe(versions)}


def open_parent_without_symlinks(path: Path) -> tuple[int, str]:
    """Open the output parent by directory fd, rejecting ancestor aliases."""
    absolute = absolute_without_symlinks(path, "probe output")
    if absolute.exists() or absolute.is_symlink():
        raise ProbeError("probe output must be absent")
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise ProbeError("platform lacks required no-follow directory flags")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parent_fd = os.open(os.sep, flags)
    try:
        for component in absolute.parent.parts[1:]:
            next_fd = os.open(component, flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        return parent_fd, absolute.name
    except OSError as exc:
        os.close(parent_fd)
        raise ProbeError(f"probe output parent is not a regular directory: {exc}") from exc


def write_json(path: Path, value: dict[str, Any]) -> None:
    parent_fd, name = open_parent_without_symlinks(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    payload = json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    try:
        try:
            output_fd = os.open(name, flags, 0o600, dir_fd=parent_fd)
        except OSError as exc:
            raise ProbeError(f"probe output creation failed closed: {exc}") from exc
        with os.fdopen(output_fd, "wb") as stream:
            stream.write(payload)
    finally:
        os.close(parent_fd)


def self_test() -> None:
    synthetic = """
def _sample_next_token(logits_BCxV, temperature, top_p, top_k, audio_eos_value): pass
class Dia:
    def __init__(self, config, compute_dtype, device, load_dac): pass
    def _encode_text(self, text): pass
    def _decoder_step(self, tokens_Bx1xC, dec_state, cfg_scale, temperature, top_p, top_k, current_idx): pass
    def generate(self, text, max_tokens, cfg_scale, temperature, top_p, use_torch_compile, cfg_filter_top_k, audio_prompt, audio_prompt_path, use_cfg_filter, verbose): pass
"""
    assert source_signatures(synthetic) == EXPECTED_SIGNATURES
    assert UPSTREAM_TORCH_PIN != CANDIDATE_TORCH_PIN
    assert SOURCE_REPOSITORY.removesuffix(".git").endswith("nari-labs/dia")
    assert classify_runtime_probe({"module_status": "PASS_SOURCE_IMPORT", "torch_status": "PASS_TORCH_IMPORT", "torchaudio_status": "BLOCKED_TORCHAUDIO_IMPORT"}) == {"status": "BLOCKED_TORCH_TORCHAUDIO_CLOSURE", "closure_status": "BLOCKED_TORCH_TORCHAUDIO_CLOSURE"}
    assert classify_runtime_probe({"module_status": "PASS_SOURCE_IMPORT", "torch_status": "PASS_TORCH_IMPORT", "torchaudio_status": "PASS_TORCHAUDIO_IMPORT"}) == {"status": "PASS_SOURCE_IMPORT_TORCH_TORCHAUDIO", "closure_status": "PASS_TORCH_TORCHAUDIO_CLOSURE"}
    tempfile = __import__("tempfile")
    temp_parent = "/private/tmp" if Path("/private/tmp").is_dir() and not Path("/private/tmp").is_symlink() else None
    with tempfile.TemporaryDirectory(prefix="dia-compat-probe-", dir=temp_parent) as directory:
        root = Path(directory)
        real = root / "real"
        real.mkdir()
        alias = root / "alias"
        alias.symlink_to(real, target_is_directory=True)
        for unsafe in (Path("relative.json"), root / "alias" / "input", root / "alias" / "output.json"):
            try:
                absolute_without_symlinks(unsafe, "self-test")
            except ProbeError:
                pass
            else:
                raise AssertionError(f"symlinked/relative path accepted: {unsafe}")
        output = root / "output.json"
        write_json(output, {"status": "self-test"})
        final_alias = root / "final-alias"
        final_alias.symlink_to(real / "target")
        try:
            write_json(final_alias, {"status": "symlink"})
        except ProbeError:
            pass
        else:
            raise AssertionError("symlinked output was accepted")
        try:
            write_json(output, {"status": "overwrite"})
        except ProbeError:
            pass
        else:
            raise AssertionError("existing output was overwritten")
    original_marker = os.environ.pop("VOKRA_PUBLISH_ON_VAST", None)
    try:
        try:
            require_vast_runtime(system="Darwin", machine="x86_64", memory_kib=MIN_VAST_MEM_KIB)
        except ProbeError:
            pass
        else:
            raise AssertionError("non-Linux runtime accepted")
        os.environ["VOKRA_PUBLISH_ON_VAST"] = "1"
        try:
            require_vast_runtime(system="Linux", machine="x86_64", memory_kib=MIN_VAST_MEM_KIB - 1)
        except ProbeError:
            pass
        else:
            raise AssertionError("low-memory runtime accepted")
    finally:
        if original_marker is not None:
            os.environ["VOKRA_PUBLISH_ON_VAST"] = original_marker
        else:
            os.environ.pop("VOKRA_PUBLISH_ON_VAST", None)
    print("dia upstream compatibility probe: self-test PASS (model-free, no source import, no weights)")


def run(source: Path, project: Path, output: Path | None) -> int:
    source = absolute_without_symlinks(source, "upstream source")
    project = absolute_without_symlinks(project, "candidate project")
    if output is not None:
        output = absolute_without_symlinks(output, "probe output")
    upstream, source_text = authenticate_source(source)
    candidate = candidate_identity(project)
    signatures = source_signatures(source_text)
    signature_status = "PASS_SOURCE_SIGNATURES" if signatures == EXPECTED_SIGNATURES else "BLOCKED_SOURCE_SIGNATURE_DRIFT"
    runtime = import_probe(source)
    result = {
        "schema": SCHEMA,
        "status": "BLOCKED_UPSTREAM_PINNED_COMPATIBILITY",
        "publication": "NO_UPLOAD",
        "weights": "NOT_ACQUIRED",
        "forward": "NOT_PERFORMED",
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "upstream": upstream,
        "candidate": candidate,
        "source_signature": {"status": signature_status, "expected": EXPECTED_SIGNATURES, "observed": signatures},
        "runtime_import": runtime,
        "compatibility": {"upstream_torch": UPSTREAM_TORCH_PIN, "upstream_torchaudio": UPSTREAM_TORCHAUDIO_PIN, "candidate_torch": CANDIDATE_TORCH_PIN, "candidate_torchaudio": CANDIDATE_TORCHAUDIO_PIN, "decision": "BLOCKED_UNTIL_VAST_API_AND_OWNER_APPROVED_CPU_PARITY"},
    }
    if output is not None:
        write_json(output, result)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--project", type=Path, default=Path(os.path.abspath(__file__)).parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.source, args.output)):
            parser.error("--self-test accepts no source/output")
        self_test()
        return 0
    if args.source is None:
        parser.error("--source is required unless --self-test is used")
    try:
        return run(args.source, args.project, args.output)
    except ProbeError as exc:
        print(f"dia upstream compatibility probe: BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
