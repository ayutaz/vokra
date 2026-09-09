#!/usr/bin/env python3
"""Audit the isolated, upstream Irodori TextBlock import closure.

This is a source-only gate.  It does not resolve packages, download source or
weights, construct a model, or execute a reference.  The full Irodori
inference environment is intentionally excluded because its authenticated
lock reaches the forbidden DACVAE/librosa/soxr/soundfile closure.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import tempfile
import tomllib
from pathlib import Path

UPSTREAM_REPOSITORY = "https://github.com/Aratako/Irodori-TTS"
UPSTREAM_REVISION = "8224dafb46d0aba89209a8f905f1cb7e3299d9c1"
SOURCE_LICENSE_DISPOSITION = "MIT"
SOURCE_LICENSE_BYTES = 1064
SOURCE_LICENSE_SHA256 = "dfd47cc99fd79cced8e0cab04ed81f70b3d4f2f475a75746b341671bd3991c00"
REFERENCE_PROJECT = Path(__file__).with_name("irodori_text_block_reference")
REFERENCE_PROJECT_SHA256 = "bec4c4f7eed1fb2f8838a8eab807c6996818e2eda54a447aee04ddc29555b645"
REFERENCE_LOCK_SHA256 = "a25fac5c9bc859c68e3e3f1badee59f4a79afb8ffea629dc7ab174d81ff126f7"

REQUIRED_FILES = frozenset(
    {
        "LICENSE",
        "irodori_tts/config.py",
        "irodori_tts/model.py",
        "irodori_tts/speaker_inversion.py",
    }
)

# This is the complete closure for importing TextBlock through the package
# shell used by irodori_text_block_dump_reference.py.  Keep this exact: a
# newly introduced upstream import must be reviewed rather than silently
# pulling the full inference environment into the reference project.
ALLOWED_IMPORTS = frozenset(
    {
        "__future__",
        "dataclasses",
        "json",
        "math",
        "pathlib",
        "safetensors",
        "torch",
        "typing",
    }
)
ALLOWED_RELATIVE_IMPORTS = frozenset({"config", "speaker_inversion"})
FORBIDDEN_IMPORTS = frozenset(
    {
        "dacvae",
        "descript_audiotools",
        "librosa",
        "soxr",
        "soundfile",
        "torchaudio",
        "transformers",
    }
)
REQUIRED_SOURCE_MARKERS = {
    "irodori_tts/model.py": (
        "class TextBlock",
        "def precompute_freqs_cis",
        "F.scaled_dot_product_attention",
    ),
    "irodori_tts/speaker_inversion.py": (
        "from safetensors.torch import load_file",
    ),
}

EXPECTED_LOCK_PACKAGES = {
    "filelock": {"3.32.5"},
    "fsspec": {"2026.7.0"},
    "jinja2": {"3.1.6"},
    "markupsafe": {"3.0.3"},
    "mpmath": {"1.3.0"},
    "networkx": {"3.6.1"},
    "safetensors": {"0.8.0"},
    "sympy": {"1.14.0"},
    "torch": {"2.13.0", "2.13.0+cpu"},
    "typing-extensions": {"4.16.0"},
    "vokra-irodori-text-block-reference": {"0.1.0"},
}

EXPECTED_PROJECT_IDENTITY = (
    "vokra-irodori-text-block-reference",
    "0.1.0",
    "==3.12.*",
    ("safetensors==0.8.0", "torch==2.13.0"),
)
EXPECTED_PROJECT_CONTRACT = {
    "status": "PENDING_DEPENDENCY_LICENSE_AUDIT",
    "dependency_license_status": "PENDING_OWNER_REVIEW",
    "execution": "NOT_AUTHORIZED",
    "publication": "NO_UPLOAD",
}


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def _module_name(value: str) -> str:
    return value.partition(".")[0]


def _imports(path: Path) -> tuple[set[str], set[str], set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    absolute: set[str] = set()
    relative: set[str] = set()
    forbidden: set[str] = set()
    # Only module-level imports execute while importing the TextBlock closure.
    # Optional helpers in the same upstream files (for example YAML config
    # loading or a pretrained Transformers backbone) are not part of this
    # oracle and must not be resolved merely because they exist in the source.
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = _module_name(alias.name)
                absolute.add(module)
                if module in FORBIDDEN_IMPORTS:
                    forbidden.add(module)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if node.module:
                    relative.add(_module_name(node.module))
            else:
                module = _module_name(node.module or "")
                absolute.add(module)
                if module in FORBIDDEN_IMPORTS:
                    forbidden.add(module)
    return absolute, relative, forbidden


def _validate_project_identity(project_data: dict[str, object]) -> None:
    identity = (
        project_data.get("name"),
        project_data.get("version"),
        project_data.get("requires-python"),
        tuple(project_data.get("dependencies", ())),
    )
    if identity != EXPECTED_PROJECT_IDENTITY:
        raise ValueError("isolated reference pyproject identity or dependencies drifted")


def validate_reference_project(project: Path = REFERENCE_PROJECT) -> dict[str, object]:
    """Validate the committed Python-3.12 project and its complete lock."""

    pyproject = project / "pyproject.toml"
    lockfile = project / "uv.lock"
    if not pyproject.is_file() or pyproject.is_symlink():
        raise ValueError("isolated reference pyproject.toml is missing or symlinked")
    if not lockfile.is_file() or lockfile.is_symlink():
        raise ValueError("isolated reference uv.lock is missing or symlinked")
    if hashlib.sha256(pyproject.read_bytes()).hexdigest() != REFERENCE_PROJECT_SHA256:
        raise ValueError("isolated reference pyproject.toml SHA-256 mismatch")
    if hashlib.sha256(lockfile.read_bytes()).hexdigest() != REFERENCE_LOCK_SHA256:
        raise ValueError("isolated reference uv.lock SHA-256 mismatch")
    pyproject_data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project_data = pyproject_data.get("project")
    if not isinstance(project_data, dict):
        raise ValueError("isolated reference project table is missing")
    _validate_project_identity(project_data)
    if pyproject_data.get("tool", {}).get("uv", {}).get("package") is not False:
        raise ValueError("isolated reference must remain a non-package project")
    uv_data = pyproject_data["tool"]["uv"]
    contract = pyproject_data.get("tool", {}).get("vokra", {}).get("irodori_text_block_reference")
    if contract != EXPECTED_PROJECT_CONTRACT:
        raise ValueError("isolated reference status contract drifted")
    if uv_data.get("override-dependencies") != ["setuptools ; python_version < '0'"]:
        raise ValueError("isolated reference setuptools omission policy drifted")
    if pyproject_data.get("tool", {}).get("uv", {}).get("sources", {}).get("torch") != {
        "index": "pytorch-cpu",
        "marker": "sys_platform == 'linux' or sys_platform == 'win32'",
    }:
        raise ValueError("isolated reference torch source is not the explicit CPU index")
    indexes = uv_data.get("index")
    if indexes != [{"name": "pytorch-cpu", "url": "https://download.pytorch.org/whl/cpu", "explicit": True}]:
        raise ValueError("isolated reference index is not the pinned PyTorch CPU index")

    lock = tomllib.loads(lockfile.read_text(encoding="utf-8"))
    if (lock.get("version"), lock.get("revision"), lock.get("requires-python")) != (1, 3, "==3.12.*"):
        raise ValueError("isolated reference lock format or Python pin drifted")
    rows = lock.get("package")
    if not isinstance(rows, list):
        raise ValueError("isolated reference lock package table is missing")
    observed: dict[str, set[str]] = {}
    forbidden: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not isinstance(row.get("version"), str):
            raise ValueError("isolated reference lock contains a malformed package row")
        name = row["name"]
        observed.setdefault(name, set()).add(row["version"])
        for dependency in row.get("dependencies", []):
            if isinstance(dependency, dict) and isinstance(dependency.get("name"), str):
                if dependency["name"] in FORBIDDEN_IMPORTS:
                    forbidden.add(dependency["name"])
        if name in FORBIDDEN_IMPORTS:
            forbidden.add(name)
    if observed != EXPECTED_LOCK_PACKAGES:
        raise ValueError(f"isolated reference lock package set drifted: {observed}")
    if forbidden:
        raise ValueError(f"isolated reference lock contains forbidden packages: {sorted(forbidden)}")
    return {
        "status": "PASS_ISOLATED_LOCK_STRUCTURAL_ONLY",
        "dependency_license_status": "PENDING_OWNER_REVIEW",
        "dependency_audit_status": "PENDING_DEPENDENCY_LICENSE_AUDIT",
        "execution": "NOT_AUTHORIZED",
        "publication": "NO_UPLOAD",
        "project_sha256": REFERENCE_PROJECT_SHA256,
        "lock_sha256": REFERENCE_LOCK_SHA256,
        "packages": {name: sorted(versions) for name, versions in sorted(observed.items())},
        "forbidden_packages": [],
    }


def audit_source(upstream: Path) -> dict[str, object]:
    """Authenticate the exact checkout and isolated TextBlock source closure."""

    lock_evidence = validate_reference_project()
    if not upstream.is_dir() or upstream.is_symlink() or not (upstream / ".git").is_dir():
        raise ValueError("upstream must be a regular git checkout")
    if _git(upstream, "rev-parse", "HEAD") != UPSTREAM_REVISION:
        raise ValueError("upstream revision does not match the immutable pin")
    if _git(upstream, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("upstream checkout is dirty")
    origin = _git(upstream, "remote", "get-url", "origin").removesuffix(".git")
    if origin != UPSTREAM_REPOSITORY:
        raise ValueError(f"upstream origin mismatch: {origin}")
    tracked = set(_git(upstream, "ls-files").splitlines())
    if not REQUIRED_FILES <= tracked:
        raise ValueError(f"required upstream files missing: {sorted(REQUIRED_FILES - tracked)}")
    license_path = upstream / "LICENSE"
    license_bytes = license_path.read_bytes()
    if len(license_bytes) != SOURCE_LICENSE_BYTES or hashlib.sha256(license_bytes).hexdigest() != SOURCE_LICENSE_SHA256:
        raise ValueError("upstream primary LICENSE bytes do not match the immutable evidence")

    source_files = sorted(path for path in REQUIRED_FILES if path.endswith(".py"))
    role_hashes: dict[str, str] = {}
    all_absolute: set[str] = set()
    all_relative: set[str] = set()
    forbidden: set[str] = set()
    for name in source_files:
        path = upstream / name
        absolute, relative, found_forbidden = _imports(path)
        role_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        all_absolute.update(absolute)
        all_relative.update(relative)
        forbidden.update(found_forbidden)
        for marker in REQUIRED_SOURCE_MARKERS.get(name, ()):
            if marker not in path.read_text(encoding="utf-8"):
                raise ValueError(f"required TextBlock marker missing: {name}: {marker}")
    if forbidden:
        raise ValueError(f"forbidden imports in isolated closure: {sorted(forbidden)}")
    if not all_absolute <= ALLOWED_IMPORTS:
        raise ValueError(f"unreviewed absolute imports: {sorted(all_absolute - ALLOWED_IMPORTS)}")
    if not all_relative <= ALLOWED_RELATIVE_IMPORTS:
        raise ValueError(f"unreviewed relative imports: {sorted(all_relative - ALLOWED_RELATIVE_IMPORTS)}")
    return {
        "status": "PASS_ISOLATED_SOURCE_CLOSURE_STRUCTURAL_ONLY",
        "repository": UPSTREAM_REPOSITORY,
        "revision": UPSTREAM_REVISION,
        "license": {
            "disposition": SOURCE_LICENSE_DISPOSITION,
            "bytes": SOURCE_LICENSE_BYTES,
            "sha256": SOURCE_LICENSE_SHA256,
            "path": "LICENSE",
        },
        "required_files": sorted(REQUIRED_FILES),
        "role_sha256": role_hashes,
        "absolute_imports": sorted(all_absolute),
        "relative_imports": sorted(all_relative),
        "external_packages": ["safetensors", "torch"],
        "full_inference_status": "BLOCKED_FORBIDDEN_DACVAE_LIBROSA_SOXR_SOUNDFILE_CLOSURE",
        "dependency_license_status": "PENDING_OWNER_REVIEW",
        "dependency_audit_status": "PENDING_DEPENDENCY_LICENSE_AUDIT",
        "execution": "SOURCE_ONLY_NO_SYNC_NO_DOWNLOAD_NO_MODEL_EXECUTION_NOT_AUTHORIZED",
        "publication": "NO_UPLOAD",
        "reference_project": lock_evidence,
    }


def self_test() -> None:
    assert validate_reference_project()["status"] == "PASS_ISOLATED_LOCK_STRUCTURAL_ONLY"
    identity = {
        "name": EXPECTED_PROJECT_IDENTITY[0],
        "version": EXPECTED_PROJECT_IDENTITY[1],
        "requires-python": EXPECTED_PROJECT_IDENTITY[2],
        "dependencies": list(EXPECTED_PROJECT_IDENTITY[3]),
    }
    _validate_project_identity(identity)
    identity["dependencies"] = list(reversed(identity["dependencies"]))
    try:
        _validate_project_identity(identity)
    except ValueError:
        pass
    else:
        raise AssertionError("permuted project dependencies accepted")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for name in REQUIRED_FILES:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if name == "LICENSE":
                path.write_text("MIT License\n", encoding="utf-8")
            else:
                path.write_text("from .config import ModelConfig\n", encoding="utf-8")
        # Exercise the AST closure independently of git/source authentication.
        safe, relative, forbidden = _imports(root / "irodori_tts/model.py")
        assert safe == set()
        assert relative == {"config"}
        assert not forbidden
        (root / "irodori_tts/model.py").write_text("import librosa\n", encoding="utf-8")
        _, _, forbidden = _imports(root / "irodori_tts/model.py")
        assert forbidden == {"librosa"}
    assert _module_name("torch.nn.functional") == "torch"
    assert _module_name("safetensors.torch") == "safetensors"
    print("irodori_text_block_source_audit.py self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.upstream is not None:
            parser.error("--self-test accepts no --upstream")
        self_test()
        return 0
    if args.upstream is None:
        parser.error("--upstream is required unless --self-test is used")
    print(json.dumps(audit_source(args.upstream), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
