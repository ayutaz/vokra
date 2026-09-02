#!/usr/bin/env python3
"""Audit the synchronized SpeechT5 oracle environment before model download.

This is deliberately model-free and standard-library-only.  It checks the
fresh worker environment rather than accepting native-file claims from the
committed VAST audit report.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
import re
import subprocess
import sys
import sysconfig
from pathlib import Path


COMPACT_SCHEMA = "vokra-speecht5-dependency-audit-compact-v1"
FULL_AUDIT_SHA256 = "b7a4c6ffcbc68109d8743b127432dfedb4897cd52641b251433945da3f4b4d3d"
TORCH_GOMP = "torch/lib/libgomp-a34b3233.so.1"
TORCH_GOMP_SHA256 = "570455c2902d6cc2a7f367703c06dac07495dd7f8a1ed2c8fc4cea628c881b13"
BUILD_ONLY = {"cython", "meson-python", "meson", "pyproject-metadata", "ninja", "patchelf"}
EXPECTED = {
    "anyio": "4.14.2",
    "certifi": "2026.7.22",
    "click": "8.5.0",
    "filelock": "3.32.4",
    "fsspec": "2026.7.0",
    "h11": "0.16.0",
    "hf-xet": "1.6.0",
    "httpcore": "1.0.9",
    "httpx": "0.28.1",
    "huggingface-hub": "1.5.0",
    "idna": "3.19",
    "jinja2": "3.1.6",
    "markupsafe": "3.0.3",
    "mpmath": "1.3.0",
    "networkx": "3.6.1",
    "numpy": "1.26.4",
    "packaging": "26.3",
    "pyyaml": "6.0.3",
    "regex": "2026.7.19",
    "safetensors": "0.8.0",
    "sentencepiece": "0.2.2",
    "setuptools": "83.0.0",
    "sympy": "1.14.0",
    "tokenizers": "0.22.2",
    "torch": "2.4.1+cpu",
    "tqdm": "4.70.0",
    "transformers": "5.10.4",
    "typer": "0.9.0",
    "typing-extensions": "4.16.0",
}
NATIVE_TOP_LEVELS = {"hf_xet", "markupsafe", "numpy", "yaml", "regex", "safetensors", "sentencepiece", "tokenizers", "torch"}
SYSTEM_NEEDED = {
    "linux-vdso.so.1",
    "libc.so.6",
    "libdl.so.2",
    "libgcc_s.so.1",
    "libm.so.6",
    "libpthread.so.0",
    "librt.so.1",
    "libstdc++.so.6",
    "ld-linux-x86-64.so.2",
}
TORCH_NEEDED = SYSTEM_NEEDED | {
    "libc10.so",
    "libshm.so",
    "libtorch.so",
    "libtorch_cpu.so",
    "libtorch_python.so",
    "libgomp-a34b3233.so.1",
}
NEEDED_RE = re.compile(r"Shared library: \[([^]]+)\]")


def fail(message: str) -> None:
    raise RuntimeError(message)


def normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).casefold()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_compact(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        fail(f"compact audit is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"compact audit is unreadable: {exc}")
    if not isinstance(value, dict) or value.get("schema") != COMPACT_SCHEMA or value.get("full_audit_sha256") != FULL_AUDIT_SHA256:
        fail("compact audit is not the reviewed VAST dependency audit")
    if value.get("operator_approval") != "PENDING_REVIEW":
        fail("compact audit operator status changed without owner review")
    return value


def distributions() -> dict[str, metadata.Distribution]:
    found: dict[str, metadata.Distribution] = {}
    for dist in metadata.distributions():
        key = normalized(dist.metadata.get("Name", dist.name))
        if key in found:
            fail(f"duplicate installed distribution: {key}")
        found[key] = dist
    if set(found) != set(EXPECTED):
        fail(f"installed distribution closure drifted: unexpected={sorted(set(found) - set(EXPECTED))}, missing={sorted(set(EXPECTED) - set(found))}")
    for key, version in EXPECTED.items():
        if found[key].version != version:
            fail(f"installed version drifted: {key}={found[key].version}, expected {version}")
    if set(found) & BUILD_ONLY:
        fail(f"isolated build dependency leaked into final environment: {sorted(set(found) & BUILD_ONLY)}")
    return found


def native_files(site_packages: Path) -> list[Path]:
    files: list[Path] = []
    for path in site_packages.rglob("*"):
        if not path.is_file() or ".dist-info" in path.parts:
            continue
        name = path.name
        if not (".so" in name or name.endswith(".a")):
            continue
        relative = path.relative_to(site_packages)
        top = relative.parts[0]
        if top not in NATIVE_TOP_LEVELS:
            fail(f"unexpected native artifact outside reviewed packages: {relative}")
        files.append(path)
    return sorted(files)


def elf_needed(path: Path) -> list[str]:
    if not ".so" in path.name:
        return []
    result = subprocess.run(["readelf", "-d", str(path)], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        fail(f"readelf failed for native artifact {path}: {result.stderr.strip()}")
    return sorted(set(NEEDED_RE.findall(result.stdout)))


def run(compact_path: Path, output_path: Path) -> int:
    compact = audit_compact(compact_path)
    if sys.platform != "linux" or sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12):
        fail(f"reviewed environment is Linux CPython 3.12, got {sys.platform} {sys.implementation.name} {sys.version_info[:2]}")
    site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()
    installed = distributions()
    observed_native: list[dict[str, object]] = []
    for path in native_files(site_packages):
        relative = path.relative_to(site_packages).as_posix()
        needed = elf_needed(path)
        allowed = TORCH_NEEDED if relative.startswith("torch/") else SYSTEM_NEEDED
        unknown = sorted(set(needed) - allowed)
        if unknown:
            fail(f"unreviewed ELF NEEDED entries in {relative}: {unknown}")
        observed_native.append({"path": relative, "sha256": sha256(path), "needed": needed})
    numpy_root = Path(installed["numpy"].locate_file("numpy")).resolve()
    if (numpy_root / "libs").exists() or (numpy_root.parent / "numpy.libs").exists():
        fail("source-built NumPy contains a numpy.libs directory")
    forbidden = {"libgfortran", "libquadmath", "libopenblas", "openblas"}
    for path in numpy_root.rglob("*"):
        if path.is_file() and any(token in path.name.casefold() for token in forbidden):
            fail(f"forbidden NumPy bundled library: {path.relative_to(site_packages)}")
    gomp = site_packages / TORCH_GOMP
    if not gomp.is_file() or sha256(gomp) != TORCH_GOMP_SHA256:
        fail("torch bundled libgomp identity drifted")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    for module in ("numpy", "torch", "transformers"):
        importlib.import_module(module)
    transformers = importlib.import_module("transformers")
    if not hasattr(transformers, "SpeechT5ForTextToSpeech"):
        fail("locked Transformers package does not expose SpeechT5ForTextToSpeech")
    result = {
        "schema": "vokra-speecht5-post-sync-audit-v1",
        "full_audit_sha256": compact["full_audit_sha256"],
        "site_packages": str(site_packages),
        "packages": {key: installed[key].version for key in sorted(installed)},
        "build_only_absent": sorted(BUILD_ONLY),
        "native_files": observed_native,
        "numpy_source_build": {"numpy_libs_entries": 0, "forbidden_bundled_libraries": [], "setup_args": ["-Dblas=none", "-Dlapack=none"]},
        "torch_gomp": {"path": TORCH_GOMP, "sha256": TORCH_GOMP_SHA256},
        "verdict": "PASS",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"SPEECHT5_POST_SYNC_AUDIT packages={len(installed)} native_files={len(observed_native)} verdict=PASS")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        raise SystemExit(run(args.compact_evidence, args.output))
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"speecht5 post-sync audit: BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
