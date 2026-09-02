#!/usr/bin/env -S uv run --frozen --project tools/parity/chattts --python 3.12 python
"""Run the pinned ChatTTS implementation for VAST-only evidence.

This is an adapter around the upstream ``Chat`` object, not a reimplementation.
It is intentionally gated by ``VOKRA_CHATTS_RUN_REFERENCE=1`` because the
composite checkpoint is large and the source has third-party runtime
dependencies.  Without that gate, only hermetic contract self-tests run.
"""
from __future__ import annotations

import argparse
from array import array
import hashlib
from importlib.metadata import PackageNotFoundError, version as installed_version
import json
import math
import os
import sys
import re
import struct
import tomllib
import tempfile
from pathlib import Path
from typing import Any

HF_REPOSITORY = "2Noise/ChatTTS"
HF_REVISION = "1a3c04a8b0651689bd9242fbb55b1f4b5a9aef84"
SOURCE_REPOSITORY = "https://github.com/2noise/ChatTTS.git"
SOURCE_REVISION = "77b89ee281cd479f5b1a787ada330dc975ca1f2a"
SAMPLE_RATE_HZ = 24_000
FORMAT = "vokra-chattts-reference-v1"
REFERENCE_PROJECT = Path(__file__).parent / "chattts"
PROJECT_VERSIONS = {
    "huggingface-hub": "1.5.0", "numba": "0.63.1", "numpy": "2.3.5",
    "pybase16384": "0.3.0", "pyyaml": "6.0.3", "requests": "2.33.0", "safetensors": "0.7.0",
    "torch": "2.7.1",
    "torchaudio": "2.7.1", "tqdm": "4.67.1", "transformers": "5.10.4",
    "vector-quantize-pytorch": "1.27.15", "vocos": "0.1.0",
}
SOURCE_TRANSFORMERS_CONSTRAINT = "transformers>=4.41.1"
TRANSFORMERS_SECURITY_ADVISORY = "GHSA-xrqw-3rrv-vx5w"
TRANSFORMERS_SECURITY_PATCHED_MINIMUM = "5.10.0"
ISOLATED_TRANSFORMERS_PIN = "5.10.4"
REFERENCE_LOCK = REFERENCE_PROJECT / "uv.lock"
REFERENCE_LOCK_SHA256 = "6099870e3685fec99e8ae68745d37ce4e71138d353cf056540d092b3d55ac4c5"
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
FORBIDDEN_PACKAGES = {"soxr", "rubberband", "triton"}
REFERENCE_DISTRIBUTION_VERSIONS = {
    "torch": {"2.7.1", "2.7.1+cpu"},
    "torchaudio": {"2.7.1", "2.7.1+cpu"},
}
REFERENCE_PACKAGE_INVENTORY_SHA256 = "74c0c3ef9afd095594e24afc48e0d2148717308aa317ea9762eb9b10d2f0ec7f"
REFERENCE_LOCK_PACKAGE_ROWS_SHA256 = "19395b8e7796dc26af01df77e3b786299391c38f3f861d2e9b59e29175b1cb4c"
REFERENCE_LICENSE_AUDIT_SHA256 = "3e5b662aa2134be84ee6645a7c483345d46550b690c582f414896c990a7f1dff"
SOURCE_LICENSE_EVIDENCE = "https://raw.githubusercontent.com/2noise/ChatTTS/77b89ee281cd479f5b1a787ada330dc975ca1f2a/LICENSE"
WEIGHT_LICENSE_EVIDENCE = "https://huggingface.co/api/models/2Noise/ChatTTS?revision=1a3c04a8b0651689bd9242fbb55b1f4b5a9aef84"
AUDIT_PRIMARY_METADATA = "PyPI JSON release metadata, upstream package LICENSE files, and official PyTorch CPU index metadata; wheel/native notices are reviewed separately."
AUDIT_SOURCE_ROUTE = "AGPLv3+ source is independent VAST-only reference code and must never be copied into Vokra runtime/product artifacts."
AUDIT_WEIGHT_ROUTE = "CC-BY-NC-4.0 weights are inspection/reference-only and are not eligible for model-zoo publication."
AUDIT_BLOCKER = "Owner approval is required before dedicated uv sync or authenticated model/source acquisition. The lock contains PSF-2.0 typing-extensions, MPL-2.0 certifi/tqdm files, Unlicense filelock, and bundled numerical-library notices; no project exception/sign-off exists."
# Canonical rows are deliberately independent of uv's TOML formatting.  The
# row order follows uv.lock and each row binds name/version/registry plus the
# exact resolution markers used to select platform distributions.
REFERENCE_PACKAGE_INVENTORY = (
    ("annotated-doc", "0.0.5", "https://pypi.org/simple", ()),
    ("anyio", "4.14.2", "https://pypi.org/simple", ()),
    ("certifi", "2026.7.22", "https://pypi.org/simple", ()),
    ("cffi", "2.1.1", "https://pypi.org/simple", ()),
    ("charset-normalizer", "3.5.1", "https://pypi.org/simple", ()),
    ("colorama", "0.4.6", "https://pypi.org/simple", ()),
    ("einops", "0.8.2", "https://pypi.org/simple", ()),
    ("einx", "0.4.3", "https://pypi.org/simple", ()),
    ("encodec", "0.1.1", "https://pypi.org/simple", ()),
    ("filelock", "3.32.4", "https://pypi.org/simple", ()),
    ("frozendict", "2.4.7", "https://pypi.org/simple", ()),
    ("fsspec", "2026.7.0", "https://pypi.org/simple", ()),
    ("h11", "0.16.0", "https://pypi.org/simple", ()),
    ("hf-xet", "1.6.0", "https://pypi.org/simple", ()),
    ("httpcore", "1.0.9", "https://pypi.org/simple", ()),
    ("httpx", "0.28.1", "https://pypi.org/simple", ()),
    ("huggingface-hub", "1.5.0", "https://pypi.org/simple", ()),
    ("idna", "3.19", "https://pypi.org/simple", ()),
    ("jinja2", "3.1.6", "https://pypi.org/simple", ()),
    ("llvmlite", "0.46.0", "https://pypi.org/simple", ()),
    ("markdown-it-py", "4.2.0", "https://pypi.org/simple", ()),
    ("markupsafe", "3.0.3", "https://pypi.org/simple", ()),
    ("mdurl", "0.1.2", "https://pypi.org/simple", ()),
    ("mpmath", "1.3.0", "https://pypi.org/simple", ()),
    ("networkx", "3.6.1", "https://pypi.org/simple", ()),
    ("numba", "0.63.1", "https://pypi.org/simple", ()),
    ("numpy", "2.3.5", "https://pypi.org/simple", ()),
    ("packaging", "26.3", "https://pypi.org/simple", ()),
    ("pybase16384", "0.3.0", "https://pypi.org/simple", ()),
    ("pycparser", "3.0", "https://pypi.org/simple", ()),
    ("pygments", "2.21.0", "https://pypi.org/simple", ()),
    ("pyyaml", "6.0.3", "https://pypi.org/simple", ()),
    ("regex", "2026.7.19", "https://pypi.org/simple", ()),
    ("requests", "2.33.0", "https://pypi.org/simple", ()),
    ("rich", "15.0.0", "https://pypi.org/simple", ()),
    ("safetensors", "0.7.0", "https://pypi.org/simple", ()),
    ("scipy", "1.18.1", "https://pypi.org/simple", ()),
    ("setuptools", "84.0.0", "https://pypi.org/simple", ()),
    ("shellingham", "1.5.4", "https://pypi.org/simple", ()),
    ("sympy", "1.14.0", "https://pypi.org/simple", ()),
    ("tokenizers", "0.22.2", "https://pypi.org/simple", ()),
    ("torch", "2.7.1", "https://download.pytorch.org/whl/cpu", ("sys_platform == 'darwin'",)),
    ("torch", "2.7.1+cpu", "https://download.pytorch.org/whl/cpu", ("(platform_machine != 'aarch64' and sys_platform == 'linux') or (platform_python_implementation != 'CPython' and sys_platform == 'linux') or (sys_platform != 'darwin' and sys_platform != 'linux')", "platform_machine == 'aarch64' and platform_python_implementation == 'CPython' and sys_platform == 'linux'")),
    ("torchaudio", "2.7.1", "https://download.pytorch.org/whl/cpu", ("platform_machine == 'aarch64' and platform_python_implementation == 'CPython' and sys_platform == 'linux'", "sys_platform == 'darwin'")),
    ("torchaudio", "2.7.1+cpu", "https://download.pytorch.org/whl/cpu", ("(platform_machine != 'aarch64' and sys_platform == 'linux') or (platform_python_implementation != 'CPython' and sys_platform == 'linux') or (sys_platform != 'darwin' and sys_platform != 'linux')",)),
    ("tqdm", "4.67.1", "https://pypi.org/simple", ()),
    ("transformers", "5.10.4", "https://pypi.org/simple", ()),
    ("typer", "0.27.2", "https://pypi.org/simple", ()),
    ("typing-extensions", "4.16.0", "https://pypi.org/simple", ()),
    ("urllib3", "2.7.0", "https://pypi.org/simple", ()),
    ("vector-quantize-pytorch", "1.27.15", "https://pypi.org/simple", ()),
    ("vocos", "0.1.0", "https://pypi.org/simple", ()),
)
# Exact dependency-level marker/source details for every concrete and virtual
# lock row.  None means uv omitted that field; retaining it here makes an
# unmarked dependency distinct from one resolved from another index.
REFERENCE_DEPENDENCY_DETAILS = {
    ("anyio", "4.14.2"): (("idna", None, None, None), ("typing-extensions", None, None, None)),
    ("cffi", "2.1.1"): (("pycparser", None, None, "implementation_name != 'PyPy'"),),
    ("einx", "0.4.3"): (("frozendict", None, None, None), ("numpy", None, None, None), ("sympy", None, None, None)),
    ("encodec", "0.1.1"): (("einops", None, None, None), ("numpy", None, None, None), ("torch", "2.7.1", "https://download.pytorch.org/whl/cpu", "sys_platform == 'darwin'"), ("torch", "2.7.1+cpu", "https://download.pytorch.org/whl/cpu", "sys_platform != 'darwin'"), ("torchaudio", "2.7.1", "https://download.pytorch.org/whl/cpu", "(platform_machine == 'aarch64' and platform_python_implementation == 'CPython' and sys_platform == 'linux') or sys_platform == 'darwin'"), ("torchaudio", "2.7.1+cpu", "https://download.pytorch.org/whl/cpu", "(platform_machine != 'aarch64' and sys_platform == 'linux') or (platform_python_implementation != 'CPython' and sys_platform == 'linux') or (sys_platform != 'darwin' and sys_platform != 'linux')")),
    ("huggingface-hub", "1.5.0"): (("filelock", None, None, None), ("fsspec", None, None, None), ("hf-xet", None, None, "platform_machine == 'AMD64' or platform_machine == 'aarch64' or platform_machine == 'amd64' or platform_machine == 'arm64' or platform_machine == 'x86_64'"), ("httpx", None, None, None), ("packaging", None, None, None), ("pyyaml", None, None, None), ("tqdm", None, None, None), ("typer", None, None, None), ("typing-extensions", None, None, None)),
    ("httpcore", "1.0.9"): (("certifi", None, None, None), ("h11", None, None, None)),
    ("httpx", "0.28.1"): (("anyio", None, None, None), ("certifi", None, None, None), ("httpcore", None, None, None), ("idna", None, None, None)),
    ("jinja2", "3.1.6"): (("markupsafe", None, None, None),),
    ("markdown-it-py", "4.2.0"): (("mdurl", None, None, None),),
    ("numba", "0.63.1"): (("llvmlite", None, None, None), ("numpy", None, None, None)),
    ("pybase16384", "0.3.0"): (("cffi", None, None, None),),
    ("requests", "2.33.0"): (("certifi", None, None, None), ("charset-normalizer", None, None, None), ("idna", None, None, None), ("urllib3", None, None, None)),
    ("scipy", "1.18.1"): (("numpy", None, None, None),),
    ("sympy", "1.14.0"): (("mpmath", None, None, None),),
    ("tokenizers", "0.22.2"): (("huggingface-hub", None, None, None),),
    ("torch", "2.7.1"): (("filelock", None, None, "sys_platform == 'darwin'"), ("fsspec", None, None, "sys_platform == 'darwin'"), ("jinja2", None, None, "sys_platform == 'darwin'"), ("networkx", None, None, "sys_platform == 'darwin'"), ("setuptools", None, None, "sys_platform == 'darwin'"), ("sympy", None, None, "sys_platform == 'darwin'"), ("typing-extensions", None, None, "sys_platform == 'darwin'")),
    ("torch", "2.7.1+cpu"): (("filelock", None, None, "sys_platform != 'darwin'"), ("fsspec", None, None, "sys_platform != 'darwin'"), ("jinja2", None, None, "sys_platform != 'darwin'"), ("networkx", None, None, "sys_platform != 'darwin'"), ("setuptools", None, None, "sys_platform != 'darwin'"), ("sympy", None, None, "sys_platform != 'darwin'"), ("typing-extensions", None, None, "sys_platform != 'darwin'")),
    ("torchaudio", "2.7.1"): (("torch", "2.7.1", "https://download.pytorch.org/whl/cpu", "sys_platform == 'darwin'"), ("torch", "2.7.1+cpu", "https://download.pytorch.org/whl/cpu", "platform_machine == 'aarch64' and platform_python_implementation == 'CPython' and sys_platform == 'linux'")),
    ("torchaudio", "2.7.1+cpu"): (("torch", "2.7.1+cpu", "https://download.pytorch.org/whl/cpu", "(platform_machine != 'aarch64' and sys_platform == 'linux') or (platform_python_implementation != 'CPython' and sys_platform == 'linux') or (sys_platform != 'darwin' and sys_platform != 'linux')"),),
    ("tqdm", "4.67.1"): (("colorama", None, None, "sys_platform == 'win32'"),),
    ("transformers", "5.10.4"): (("huggingface-hub", None, None, None), ("numpy", None, None, None), ("packaging", None, None, None), ("pyyaml", None, None, None), ("regex", None, None, None), ("safetensors", None, None, None), ("tokenizers", None, None, None), ("tqdm", None, None, None), ("typer", None, None, None)),
    ("typer", "0.27.2"): (("annotated-doc", None, None, None), ("colorama", None, None, "sys_platform == 'win32'"), ("rich", None, None, None), ("shellingham", None, None, None)),
    ("rich", "15.0.0"): (("markdown-it-py", None, None, None), ("pygments", None, None, None)),
    ("vector-quantize-pytorch", "1.27.15"): (("einops", None, None, None), ("einx", None, None, None), ("torch", "2.7.1", "https://download.pytorch.org/whl/cpu", "sys_platform == 'darwin'"), ("torch", "2.7.1+cpu", "https://download.pytorch.org/whl/cpu", "sys_platform != 'darwin'")),
    ("vocos", "0.1.0"): (("einops", None, None, None), ("encodec", None, None, None), ("huggingface-hub", None, None, None), ("numpy", None, None, None), ("pyyaml", None, None, None), ("scipy", None, None, None), ("torch", "2.7.1", "https://download.pytorch.org/whl/cpu", "sys_platform == 'darwin'"), ("torch", "2.7.1+cpu", "https://download.pytorch.org/whl/cpu", "sys_platform != 'darwin'"), ("torchaudio", "2.7.1", "https://download.pytorch.org/whl/cpu", "(platform_machine == 'aarch64' and platform_python_implementation == 'CPython' and sys_platform == 'linux') or sys_platform == 'darwin'"), ("torchaudio", "2.7.1+cpu", "https://download.pytorch.org/whl/cpu", "(platform_machine != 'aarch64' and sys_platform == 'linux') or (platform_python_implementation != 'CPython' and sys_platform == 'linux') or (sys_platform != 'darwin' and sys_platform != 'linux')")),
    ("vokra-chattts-reference", "0.1.0"): (("huggingface-hub", None, None, None), ("numba", None, None, None), ("numpy", None, None, None), ("pybase16384", None, None, None), ("pyyaml", None, None, None), ("requests", None, None, None), ("safetensors", None, None, None), ("torch", "2.7.1", "https://download.pytorch.org/whl/cpu", "sys_platform == 'darwin'"), ("torch", "2.7.1+cpu", "https://download.pytorch.org/whl/cpu", "sys_platform != 'darwin'"), ("torchaudio", "2.7.1", "https://download.pytorch.org/whl/cpu", "(platform_machine == 'aarch64' and platform_python_implementation == 'CPython' and sys_platform == 'linux') or sys_platform == 'darwin'"), ("torchaudio", "2.7.1+cpu", "https://download.pytorch.org/whl/cpu", "(platform_machine != 'aarch64' and sys_platform == 'linux') or (platform_python_implementation != 'CPython' and sys_platform == 'linux') or (sys_platform != 'darwin' and sys_platform != 'linux')"), ("tqdm", None, None, None), ("transformers", None, None, None), ("vector-quantize-pytorch", None, None, None), ("vocos", None, None, None)),
}
DEPENDENCY_GATE_FIELDS = {
    "source_repository", "source_revision", "source_license", "weight_license",
    "selection_status", "selection_note", "upstream_install_requires",
    "excluded_upstream_optional", "torch_index", "forbidden_packages", "lock_sha256",
    "package_inventory_sha256", "lock_package_rows_sha256",
    "transformers_security_advisory", "transformers_security_patched_minimum",
    "isolated_transformers_pin",
}
LICENSE_AUDIT_FIELDS = {
    "status", "primary_metadata", "source_route", "weight_route", "blocker",
    "package_records", "source_license_evidence", "weight_license_evidence",
    "record_sha256",
}
REQUIRED_RELEASE_FILES = {
    "asset/DVAE.safetensors", "asset/Decoder.safetensors", "asset/Embed.safetensors",
    "asset/Vocos.safetensors", "asset/gpt/config.json", "asset/gpt/model.safetensors",
    "asset/tokenizer/special_tokens_map.json", "asset/tokenizer/tokenizer.json",
    "asset/tokenizer/tokenizer_config.json", "README.md",
}
SOURCE_ROLE_BLOBS = {
    "ChatTTS/core.py": "5bd65336ffb6caad06c756105974fd341c7575f0",
    "ChatTTS/config/config.py": "c91d74c2182a76e8519fbf47783667a0993ccb2d",
    "ChatTTS/model/dvae.py": "01802b697c455d3714c57bc412196521a35893f3",
    "ChatTTS/model/embed.py": "bd8f7fe35013fca43cf04fafdd405d5ee55ba1d2",
    "ChatTTS/model/gpt.py": "e6108e52df48e628058c33416c944a2a5bf0b3ff",
    "ChatTTS/model/speaker.py": "5435922ab019a5ff50751e9054a9d06f1a51403b",
    "ChatTTS/model/tokenizer.py": "84a14527a9014b47d26dcb9f914e14ec4b7053c0",
    "ChatTTS/model/processors.py": "f774dd27f5af40eef0b9f517c3d037154698eedf",
    "ChatTTS/utils/io.py": "dc90e0e9dc6b248d0602939d330761fb30cfa149",
    "ChatTTS/res/sha256_map.json": "ae91128693a5c9519ae6acf3ca25330e3dbb6aa7",
    "LICENSE": "0ad25db4bd1d86c452db3f9602ccdbe172438f52",
    "README.md": "b21e908694a607d7c153e96e7d5891cffde33b95",
    "setup.py": "dde50e327955b052ba2fab0203b2001b2845564c",
    "requirements.txt": "bd108b79febd41db030c71566c99047665a97852",
}


def canonical_lock_inventory(lock: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize every concrete uv package row and dependency edge."""
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise RuntimeError("ChatTTS uv.lock package inventory is missing")
    rows = []
    for item in packages:
        if not isinstance(item, dict) or item.get("source", {}).get("registry") is None:
            continue
        source = item["source"]
        dependencies = item.get("dependencies", [])
        if not isinstance(dependencies, list):
            raise RuntimeError(f"ChatTTS lock dependencies are malformed: {item.get('name')}")
        details = []
        for dependency in dependencies:
            if not isinstance(dependency, dict) or not isinstance(dependency.get("name"), str):
                raise RuntimeError(f"ChatTTS lock dependency row is malformed: {item.get('name')}")
            dep_source = dependency.get("source")
            if dep_source is not None and (not isinstance(dep_source, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in dep_source.items())):
                raise RuntimeError(f"ChatTTS lock dependency source is malformed: {item.get('name')}")
            marker = dependency.get("marker")
            if marker is not None and not isinstance(marker, str):
                raise RuntimeError(f"ChatTTS lock dependency marker is malformed: {item.get('name')}")
            details.append({
                "name": dependency["name"], "version": dependency.get("version"),
                "source": None if dep_source is None else {key: dep_source[key] for key in sorted(dep_source)},
                "marker": marker,
            })
        rows.append({
            "name": item.get("name"),
            "version": item.get("version"),
            "source": {"registry": source.get("registry")},
            "resolution_markers": sorted(item.get("resolution-markers", [])),
            "dependency_details": sorted(details, key=lambda detail: (detail["name"], detail["version"] or "", json.dumps(detail["source"], sort_keys=True), detail["marker"] or "")),
        })
    return rows


def expected_lock_inventory() -> list[dict[str, Any]]:
    return [
        {"name": name, "version": version, "source": {"registry": registry}, "resolution_markers": list(markers), "dependency_details": [
            {"name": dep_name, "version": dep_version, "source": None if dep_source is None else {"registry": dep_source}, "marker": marker}
            for dep_name, dep_version, dep_source, marker in REFERENCE_DEPENDENCY_DETAILS.get((name, version), ())
        ]}
        for name, version, registry, markers in REFERENCE_PACKAGE_INVENTORY
    ]


def canonical_lock_rows(lock: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize registry and virtual rows without silently dropping either."""
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise RuntimeError("ChatTTS uv.lock package rows are missing")
    rows = []
    for item in packages:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not isinstance(item.get("version"), str):
            raise RuntimeError("ChatTTS uv.lock contains a malformed package row")
        source = item.get("source")
        if not isinstance(source, dict) or set(source) not in ({"registry"}, {"virtual"}) or not isinstance(next(iter(source.values()), None), str):
            raise RuntimeError(f"ChatTTS uv.lock package source is malformed: {item.get('name')}")
        rows.append(item)
    # canonical_lock_inventory intentionally contains only the concrete
    # registry rows; this companion identity binds the virtual root too.
    concrete = canonical_lock_inventory({"package": rows})
    virtual = [item for item in rows if item["source"].get("virtual") == "."]
    if len(virtual) != 1:
        raise RuntimeError("ChatTTS uv.lock must contain exactly one virtual root")
    root = virtual[0]
    details = []
    for dependency in root.get("dependencies", []):
        dep_source = dependency.get("source")
        details.append({"name": dependency["name"], "version": dependency.get("version"), "source": None if dep_source is None else {key: dep_source[key] for key in sorted(dep_source)}, "marker": dependency.get("marker")})
    root_row = {"name": root["name"], "version": root["version"], "source": {"virtual": root["source"]["virtual"]}, "resolution_markers": sorted(root.get("resolution-markers", [])), "dependency_details": sorted(details, key=lambda detail: (detail["name"], detail["version"] or "", json.dumps(detail["source"], sort_keys=True), detail["marker"] or ""))}
    return concrete + [root_row]


def expected_lock_rows() -> list[dict[str, Any]]:
    return expected_lock_inventory() + expected_lock_inventory_for_virtual()


def expected_lock_inventory_for_virtual() -> list[dict[str, Any]]:
    name, version, source, markers = ("vokra-chattts-reference", "0.1.0", ".", ())
    return [{"name": name, "version": version, "source": {"virtual": source}, "resolution_markers": list(markers), "dependency_details": [
        {"name": dep_name, "version": dep_version, "source": None if dep_source is None else {"registry": dep_source}, "marker": marker}
        for dep_name, dep_version, dep_source, marker in REFERENCE_DEPENDENCY_DETAILS[(name, version)]
    ]}]


def canonical_json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def reference_project_identity() -> dict[str, Any]:
    """Authenticate the exact frozen Python environment before model loading."""
    pyproject = REFERENCE_PROJECT / "pyproject.toml"
    lock = REFERENCE_PROJECT / "uv.lock"
    if not pyproject.is_file() or not lock.is_file():
        raise RuntimeError("ChatTTS dedicated pyproject.toml and uv.lock are required before acquisition")
    config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    dependencies = config.get("project", {}).get("dependencies")
    if not isinstance(dependencies, list):
        raise RuntimeError("dedicated ChatTTS dependency inventory is missing")
    expected = {f"{name}=={version}" for name, version in PROJECT_VERSIONS.items()}
    if set(dependencies) != expected:
        raise RuntimeError("dedicated ChatTTS dependency inventory drifted")
    lock_text = lock.read_text(encoding="utf-8")
    lock_document = tomllib.loads(lock_text)
    inventory = canonical_lock_inventory(lock_document)
    if inventory != expected_lock_inventory() or canonical_json_digest(inventory) != REFERENCE_PACKAGE_INVENTORY_SHA256:
        raise RuntimeError("dedicated ChatTTS lock package inventory/source markers drifted")
    lock_rows = canonical_lock_rows(lock_document)
    if lock_rows != expected_lock_rows() or canonical_json_digest(lock_rows) != REFERENCE_LOCK_PACKAGE_ROWS_SHA256:
        raise RuntimeError("dedicated ChatTTS registry/virtual dependency edges drifted")
    reference = config.get("tool", {}).get("vokra", {}).get("chattts_reference", {})
    audit = reference.get("license_audit", {}) if isinstance(reference, dict) else {}
    if not isinstance(reference, dict) or reference.get("package_inventory_sha256") != REFERENCE_PACKAGE_INVENTORY_SHA256 or not isinstance(audit, dict) or audit.get("primary_metadata") != AUDIT_PRIMARY_METADATA or audit.get("source_route") != AUDIT_SOURCE_ROUTE or audit.get("weight_route") != AUDIT_WEIGHT_ROUTE or audit.get("blocker") != AUDIT_BLOCKER or audit.get("source_license_evidence") != SOURCE_LICENSE_EVIDENCE or audit.get("weight_license_evidence") != WEIGHT_LICENSE_EVIDENCE or audit.get("record_sha256") != REFERENCE_LICENSE_AUDIT_SHA256 or canonical_json_digest(audit.get("package_records")) != REFERENCE_LICENSE_AUDIT_SHA256:
        raise RuntimeError("dedicated ChatTTS license audit record digest drifted")
    locked: dict[str, str] = {}
    for name in PROJECT_VERSIONS:
        match = re.search(rf'(?m)^name = "{re.escape(name)}"\nversion = "([^"]+)"', lock_text)
        if not match or match.group(1) != PROJECT_VERSIONS[name]:
            raise RuntimeError(f"dedicated lock package/version mismatch: {name}")
        locked[name] = match.group(1)
    actual_versions = {}
    for name, expected_version in PROJECT_VERSIONS.items():
        try:
            actual = installed_version(name)
        except PackageNotFoundError as error:
            raise RuntimeError(f"dedicated environment is missing direct dependency: {name}") from error
        allowed_versions = REFERENCE_DISTRIBUTION_VERSIONS.get(name, {expected_version})
        if actual not in allowed_versions or locked[name] not in allowed_versions:
            raise RuntimeError(f"dedicated installed package drift: {name}={actual}")
        actual_versions[name] = actual
    return {
        "python": ">=3.12,<3.13",
        "pyproject_sha256": hashlib.sha256(pyproject.read_bytes()).hexdigest(),
        "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "package_inventory": inventory,
        "package_inventory_sha256": REFERENCE_PACKAGE_INVENTORY_SHA256,
        "package_lock_rows": lock_rows,
        "package_lock_rows_sha256": REFERENCE_LOCK_PACKAGE_ROWS_SHA256,
        "license_audit_sha256": REFERENCE_LICENSE_AUDIT_SHA256,
        "dependencies": locked,
        "actual_versions": actual_versions,
    }


def validate_dependency_gate(project_path: Path = REFERENCE_PROJECT / "pyproject.toml", lock_path: Path = REFERENCE_LOCK) -> None:
    """Refuse sync/acquisition until the exact lock has explicit owner approval.

    This check intentionally uses only Python's standard library.  Workers call
    it through ``uv run --no-project`` before any dedicated environment sync,
    or model download.  ``BLOCKED_UNRESOLVED`` is the checked-in default until
    the owner signs off the primary-source audit; changing a lock or status
    cannot silently grant permission because the digest and package inventory
    are both bound here.
    """
    try:
        project = tomllib.loads(project_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError("ChatTTS dependency gate cannot read pyproject.toml") from error
    reference = project.get("tool", {}).get("vokra", {}).get("chattts_reference")
    if not isinstance(reference, dict) or not DEPENDENCY_GATE_FIELDS.issubset(reference):
        raise RuntimeError("ChatTTS dependency gate metadata is incomplete")
    audit = reference.get("license_audit")
    if not isinstance(audit, dict) or set(audit) != LICENSE_AUDIT_FIELDS:
        raise RuntimeError("ChatTTS dependency license audit metadata is incomplete")
    records = audit.get("package_records")
    if not isinstance(records, list) or not records:
        raise RuntimeError("ChatTTS package license audit records are missing")
    if reference.get("source_repository") != SOURCE_REPOSITORY or reference.get("source_revision") != SOURCE_REVISION or reference.get("source_license") != "AGPLv3+":
        raise RuntimeError("ChatTTS source identity/license gate drifted")
    if reference.get("weight_license") != "CC-BY-NC-4.0":
        raise RuntimeError("ChatTTS weight license gate drifted")
    if reference.get("transformers_security_advisory") != TRANSFORMERS_SECURITY_ADVISORY or reference.get("transformers_security_patched_minimum") != TRANSFORMERS_SECURITY_PATCHED_MINIMUM or reference.get("isolated_transformers_pin") != ISOLATED_TRANSFORMERS_PIN:
        raise RuntimeError("ChatTTS transformers security provenance drifted")
    if reference.get("torch_index") != PYTORCH_CPU_INDEX:
        raise RuntimeError("ChatTTS PyTorch CPU index gate drifted")
    if reference.get("package_inventory_sha256") != REFERENCE_PACKAGE_INVENTORY_SHA256:
        raise RuntimeError("ChatTTS package inventory digest binding drifted")
    if reference.get("lock_package_rows_sha256") != REFERENCE_LOCK_PACKAGE_ROWS_SHA256:
        raise RuntimeError("ChatTTS registry/virtual package-row digest binding drifted")
    if set(reference.get("forbidden_packages", [])) != FORBIDDEN_PACKAGES | {"nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12"}:
        raise RuntimeError("ChatTTS forbidden-package gate drifted")
    actual_digest = hashlib.sha256(lock_path.read_bytes()).hexdigest() if lock_path.is_file() else None
    if actual_digest is None or actual_digest != reference.get("lock_sha256") or actual_digest == "PENDING":
        raise RuntimeError("ChatTTS uv.lock identity is not bound to pyproject.toml")
    try:
        lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError("ChatTTS dependency gate cannot read uv.lock") from error
    packages = lock.get("package")
    if lock.get("requires-python") != "==3.12.*" or not isinstance(packages, list):
        raise RuntimeError("ChatTTS uv.lock package inventory is missing")
    inventory = canonical_lock_inventory(lock)
    if inventory != expected_lock_inventory() or canonical_json_digest(inventory) != REFERENCE_PACKAGE_INVENTORY_SHA256:
        raise RuntimeError("ChatTTS uv.lock canonical package inventory drifted")
    lock_rows = canonical_lock_rows(lock)
    if lock_rows != expected_lock_rows() or canonical_json_digest(lock_rows) != REFERENCE_LOCK_PACKAGE_ROWS_SHA256:
        raise RuntimeError("ChatTTS uv.lock registry/virtual dependency edges drifted")
    names = {item.get("name") for item in packages if isinstance(item, dict) and isinstance(item.get("name"), str)}
    if any(name in FORBIDDEN_PACKAGES or name.startswith("nvidia-") for name in names):
        raise RuntimeError("ChatTTS uv.lock contains forbidden GPL/CUDA dependency")
    locked_records = {(item.get("name"), item.get("version")) for item in packages if isinstance(item, dict) and item.get("source", {}).get("registry")}
    audited_records = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {"name", "version", "license", "primary_source", "route"}:
            raise RuntimeError("ChatTTS package license audit record schema drifted")
        if not all(isinstance(record[key], str) and record[key] for key in record):
            raise RuntimeError("ChatTTS package license audit record is malformed")
        if record["route"] != "reference-only" or not record["primary_source"].startswith(("https://pypi.org/pypi/", "https://download.pytorch.org/whl/cpu/")):
            raise RuntimeError("ChatTTS package license evidence is not primary-source/reference-only")
        audited_records.add((record["name"], record["version"]))
    if audited_records != locked_records:
        raise RuntimeError("ChatTTS package license records do not cover the exact lock")
    if audit.get("primary_metadata") != AUDIT_PRIMARY_METADATA or audit.get("source_route") != AUDIT_SOURCE_ROUTE or audit.get("weight_route") != AUDIT_WEIGHT_ROUTE or audit.get("blocker") != AUDIT_BLOCKER or audit.get("source_license_evidence") != SOURCE_LICENSE_EVIDENCE or audit.get("weight_license_evidence") != WEIGHT_LICENSE_EVIDENCE:
        raise RuntimeError("ChatTTS source/weight license routes or blocker text drifted")
    if audit.get("record_sha256") != REFERENCE_LICENSE_AUDIT_SHA256 or canonical_json_digest(records) != REFERENCE_LICENSE_AUDIT_SHA256:
        raise RuntimeError("ChatTTS dependency license audit digest is not immutable")
    virtual = [item for item in packages if isinstance(item, dict) and item.get("source", {}).get("virtual") == "."]
    if len(virtual) != 1 or virtual[0].get("name") != "vokra-chattts-reference" or virtual[0].get("version") != "0.1.0":
        raise RuntimeError("ChatTTS virtual project record is malformed")
    for item in packages:
        if not isinstance(item, dict) or item.get("name") == "vokra-chattts-reference":
            continue
        source = item.get("source")
        if not isinstance(source, dict) or not source:
            raise RuntimeError("ChatTTS lock package source identity is missing")
        registry = source.get("registry")
        if item.get("name") in {"torch", "torchaudio"}:
            if registry != PYTORCH_CPU_INDEX:
                raise RuntimeError("ChatTTS torch/torchaudio package is not sourced from the official CPU index")
        elif registry != "https://pypi.org/simple":
            raise RuntimeError(f"ChatTTS package has an unexpected registry source: {item.get('name')}")
    root_requires = virtual[0].get("metadata", {}).get("requires-dist", [])
    if {item.get("name"): str(item.get("specifier", "")).removeprefix("==") for item in root_requires if isinstance(item, dict)} != PROJECT_VERSIONS:
        raise RuntimeError("ChatTTS lock direct requirements drifted")
    if reference.get("selection_status") != "AUDITED_ALLOW" or audit.get("status") != "AUDITED_ALLOW":
        raise RuntimeError("ChatTTS dependency/license audit is not affirmatively approved")


def dependency_gate_fixture_tests() -> None:
    """Negative tests ensure missing/tampered approval never opens the gate."""
    with tempfile.TemporaryDirectory(prefix="vokra-chattts-gate-") as directory:
        root = Path(directory)
        project = root / "pyproject.toml"
        lock = root / "uv.lock"
        if not (REFERENCE_PROJECT / "pyproject.toml").is_file() or not REFERENCE_LOCK.is_file():
            raise AssertionError("dedicated project files are required for gate self-test")
        project.write_bytes((REFERENCE_PROJECT / "pyproject.toml").read_bytes())
        lock.write_bytes(REFERENCE_LOCK.read_bytes())
        original_project = project.read_bytes()
        original_lock = lock.read_bytes()

        def rejected(project_bytes: bytes, lock_bytes: bytes, label: str) -> None:
            project.write_bytes(project_bytes)
            lock.write_bytes(lock_bytes)
            try:
                validate_dependency_gate(project, lock)
            except RuntimeError:
                return
            raise AssertionError(f"{label} unexpectedly opened the gate")

        try:
            validate_dependency_gate(project, lock)
        except RuntimeError:
            pass
        else:
            raise AssertionError("blocked ChatTTS audit unexpectedly opened the gate")
        altered = project.read_text(encoding="utf-8").replace(
            'selection_status = "AUTHENTICATED_CPU_INDEX_METADATA_LOCKED"',
            'selection_status = "NOT_APPROVED"',
        )
        rejected(altered.encode(), original_lock, "unknown approval status")
        marker_tampered = original_lock.replace(b"sys_platform == 'darwin'", b"sys_platform == 'linux'", 1)
        rejected(original_project, marker_tampered, "resolution-marker tamper")
        source_tampered = original_lock.replace(b"https://pypi.org/simple", b"https://evil.invalid/simple", 1)
        rejected(original_project, source_tampered, "package-source tamper")
        license_tampered = original_project.replace(b'license = "MPL-2.0"', b'license = "MIT"', 1)
        rejected(license_tampered, original_lock, "license-record tamper")
        rejected(original_project, original_lock + b"\n# tampered\n", "lock digest tamper")
        lock_document = tomllib.loads(original_lock.decode("utf-8"))
        for label, package_name, dependency_index, field, value in (
            ("dependency-marker", "huggingface-hub", 0, "marker", "sys_platform == 'win32'"),
            ("dependency-source", "encodec", 2, "source", {"registry": "https://evil.invalid/simple"}),
        ):
            candidate = json.loads(json.dumps(lock_document))
            package = next(item for item in candidate["package"] if item.get("name") == package_name)
            package["dependencies"][dependency_index][field] = value
            if canonical_json_digest(canonical_lock_rows(candidate)) == REFERENCE_LOCK_PACKAGE_ROWS_SHA256:
                raise AssertionError(f"{label} tamper was accepted")


def digest(value: Any) -> str:
    """Return a stable digest for bounded JSON-compatible evidence."""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_request(request: Any) -> dict[str, Any]:
    """Validate the small request packet without choosing hidden defaults."""
    if not isinstance(request, dict) or set(request) != {"text", "seed", "max_new_token", "temperature", "top_p", "top_k", "repetition_penalty"}:
        raise ValueError("ChatTTS reference request fields must be exact")
    if not isinstance(request["text"], str) or not request["text"].strip() or "\x00" in request["text"] or len(request["text"]) > 16_384:
        raise ValueError("reference text must be bounded non-empty UTF-8")
    if not isinstance(request["seed"], int) or isinstance(request["seed"], bool) or request["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")
    if request["max_new_token"] != 1:
        raise ValueError("ChatTTS evidence contract is fixed to exactly one generated time step")
    for name in ("temperature", "top_p", "repetition_penalty"):
        if not isinstance(request[name], (int, float)) or isinstance(request[name], bool) or not float(request[name]) == float(request[name]) or abs(float(request[name])) == float("inf"):
            raise ValueError(f"{name} must be finite")
    if request["temperature"] <= 0 or not 0 < request["top_p"] <= 1 or request["top_k"] <= 0 or request["repetition_penalty"] <= 0:
        raise ValueError("sampling controls are outside the fixed source contract")
    if not isinstance(request["top_k"], int) or isinstance(request["top_k"], bool):
        raise ValueError("top_k must be an integer")
    return request


def _checked_numel(shape: list[int]) -> int:
    """Multiply dimensions with an explicit bounded overflow check."""
    total = 1
    for dimension in shape:
        if dimension < 0 or total > 16_777_216 // max(1, dimension):
            raise ValueError("tensor tap exceeds the bounded element budget")
        total *= dimension
    return total


def _artifact_path(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError("artifact path is unsafe")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("artifact path escapes evidence root")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("artifact path escapes evidence root") from error
    return path


def _finite_f32(raw: bytes) -> bool:
    """Check the stored little-endian float32 stream without trusting metadata."""
    if len(raw) % 4:
        return False
    values = array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return all(math.isfinite(value) for value in values)


def _validate_discrete_f32_artifact(artifact: dict[str, Any], root: Path, upper: int, label: str, expected_dtypes: set[str]) -> None:
    """Re-read stored f32 bytes; metadata alone cannot authenticate IDs."""
    source = artifact.get("source")
    if not isinstance(source, dict) or source.get("dtype") not in expected_dtypes:
        raise ValueError(f"{label} source dtype is not discrete")
    path = _artifact_path(root, artifact.get("path"))
    raw = path.read_bytes()
    if len(raw) % 4:
        raise ValueError(f"{label} f32 byte count is invalid")
    values = [value for (value,) in struct.iter_unpack("<f", raw)]
    if len(values) != source.get("numel"):
        raise ValueError(f"{label} f32 cardinality differs from source metadata")
    for value in values:
        if not math.isfinite(value) or value != math.trunc(value) or value < 0 or value > upper:
            raise ValueError(f"{label} contains fractional or out-of-range value")


def _tensor_record(value: Any, name: str, *, include_values: bool = False) -> dict[str, Any]:
    """Record only bounded metadata from a real upstream tensor."""
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"upstream tap {name} was not tensor-like")
    shape = [int(dimension) for dimension in value.shape]
    numel = _checked_numel(shape)
    if not shape or len(shape) > 4 or numel <= 0 or any(dimension <= 0 for dimension in shape):
        raise ValueError(f"upstream tap {name} exceeds evidence bounds")
    finite = bool(value.isfinite().all().item()) if value.is_floating_point() else True
    if not finite:
        raise ValueError(f"upstream tap {name} contains non-finite values")
    record = {"name": name, "shape": shape, "numel": numel, "dtype": str(value.dtype), "finite": finite}
    if include_values:
        flat = value.detach().cpu().reshape(-1).tolist()
        if len(flat) != numel or len(flat) > 65_536:
            raise ValueError(f"upstream tap {name} exceeds value evidence bound")
        record["values"] = flat
    return record


def _validate_discrete(value: Any, name: str, upper: int, *, mask: bool = False) -> None:
    """Reject float/self-asserted ID and mask packets at the official tap."""
    dtype = value.dtype
    if getattr(dtype, "is_floating_point", False) or getattr(dtype, "is_complex", False):
        raise ValueError(f"{name} must be an integer/bool tensor")
    values = value.detach().cpu().reshape(-1).tolist()
    if any(isinstance(item, bool) and not mask for item in values):
        raise ValueError(f"{name} must use integer IDs")
    for item in values:
        if not isinstance(item, int) or item < 0 or item > (1 if mask else upper):
            raise ValueError(f"{name} contains an out-of-range value")


def validate_reference_evidence(root: Path, evidence: Any) -> None:
    """Validate the bounded binary evidence emitted by the official adapter."""
    if not isinstance(evidence, dict) or set(evidence) != {"model", "source", "reference_project", "request", "execution_id", "sample_rate_hz", "routes", "artifacts", "taps", "rng_records"}:
        raise ValueError("reference evidence fields are incomplete")
    request = validate_request(evidence["request"])
    if evidence["execution_id"] != digest(request):
        raise ValueError("execution identity is not bound to the request")
    project = evidence["reference_project"]
    if not isinstance(project, dict) or set(project) != {"python", "pyproject_sha256", "uv_lock_sha256", "package_inventory", "package_inventory_sha256", "package_lock_rows", "package_lock_rows_sha256", "license_audit_sha256", "dependencies", "actual_versions"} or project["python"] != ">=3.12,<3.13" or not re.fullmatch(r"[0-9a-f]{64}", project["pyproject_sha256"]) or not re.fullmatch(r"[0-9a-f]{64}", project["uv_lock_sha256"]) or project["package_inventory"] != expected_lock_inventory() or project["package_inventory_sha256"] != REFERENCE_PACKAGE_INVENTORY_SHA256 or project["package_lock_rows"] != expected_lock_rows() or project["package_lock_rows_sha256"] != REFERENCE_LOCK_PACKAGE_ROWS_SHA256 or project["license_audit_sha256"] != REFERENCE_LICENSE_AUDIT_SHA256 or project["dependencies"] != PROJECT_VERSIONS or any(project["actual_versions"].get(name) not in REFERENCE_DISTRIBUTION_VERSIONS.get(name, {version}) for name, version in PROJECT_VERSIONS.items()):
        raise ValueError("dedicated ChatTTS project/lock identity is missing")
    model = evidence["model"]
    source = evidence["source"]
    if not isinstance(model, dict) or model.get("repository") != HF_REPOSITORY or model.get("revision") != HF_REVISION or model.get("resolved_revision") != HF_REVISION or not isinstance(model.get("files"), list) or len(model["files"]) != 23 or len({row.get("path") for row in model["files"] if isinstance(row, dict)}) != 23 or not isinstance(model.get("selected"), dict) or set(model["selected"]) != REQUIRED_RELEASE_FILES:
        raise ValueError("authenticated model identity/evidence is missing")
    if any(not isinstance(row, dict) or set(row) != {"path", "bytes", "git_blob_sha1", "lfs_sha256", "local_verified"} or not isinstance(row["bytes"], int) or not re.fullmatch(r"[0-9a-f]{40}", row["git_blob_sha1"]) or (row["lfs_sha256"] is not None and not re.fullmatch(r"[0-9a-f]{64}", row["lfs_sha256"])) for row in model["files"]):
        raise ValueError("authenticated model server-tree rows are malformed")
    from chattts_inspect import SELECTED
    for name, (size, blob, lfs) in SELECTED.items():
        row = next(row for row in model["files"] if row["path"] == name)
        selected = model["selected"][name]
        if row["bytes"] != size or row["git_blob_sha1"] != blob or row["lfs_sha256"] != lfs or not isinstance(selected, dict) or selected.get("bytes") != size or not re.fullmatch(r"[0-9a-f]{64}", selected.get("sha256", "")):
            raise ValueError(f"fixed release identity mismatch: {name}")
    if not isinstance(source, dict) or source.get("repository") != SOURCE_REPOSITORY or source.get("origin") != SOURCE_REPOSITORY or source.get("revision") != SOURCE_REVISION or source.get("tag") != "v0.2.5" or source.get("clean") is not True or not isinstance(source.get("roles"), dict) or set(source["roles"]) != set(SOURCE_ROLE_BLOBS):
        raise ValueError("authenticated source identity/evidence is missing")
    for role, record in source["roles"].items():
        if not isinstance(record, dict) or set(record) != {"sha256", "git_blob_sha1"} or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) or record["git_blob_sha1"] != SOURCE_ROLE_BLOBS[role]:
            raise ValueError(f"source role identity mismatch: {role}")
    if evidence["sample_rate_hz"] != SAMPLE_RATE_HZ or set(evidence["routes"]) != {"dvae", "decoder"}:
        raise ValueError("both official decode routes are required")
    for route, record in evidence["routes"].items():
        if not isinstance(record, dict) or set(record) != {"sample_rate_hz", "samples", "dtype", "bytes", "path", "sha256", "execution_id"} or record["execution_id"] != evidence["execution_id"] or record["sample_rate_hz"] != SAMPLE_RATE_HZ or record["dtype"] != "float32":
            raise ValueError(f"invalid {route} PCM record")
        if not isinstance(record["samples"], int) or isinstance(record["samples"], bool) or record["samples"] <= 0 or not isinstance(record["bytes"], int) or isinstance(record["bytes"], bool) or record["bytes"] <= 0 or not isinstance(record["sha256"], str) or len(record["sha256"]) != 64 or any(character not in "0123456789abcdef" for character in record["sha256"].lower()):
            raise ValueError(f"invalid {route} PCM schema")
        path = _artifact_path(root, record["path"])
        raw = path.read_bytes()
        if record["bytes"] > 256 * 1024 * 1024 or len(raw) != record["bytes"] or len(raw) != record["samples"] * 4 or not _finite_f32(raw) or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError(f"{route} PCM artifact identity mismatch")
    if not isinstance(evidence["artifacts"], list) or len(evidence["artifacts"]) < 10:
        raise ValueError("reference tensor artifacts are missing")
    artifact_paths: set[str] = set()
    for index, artifact in enumerate(evidence["artifacts"]):
        if not isinstance(artifact, dict) or set(artifact) != {"path", "source", "storage_dtype", "bytes", "sha256", "execution_id", "call_index", "route"} or artifact["storage_dtype"] != "float32" or artifact["execution_id"] != evidence["execution_id"] or not isinstance(artifact["call_index"], int) or artifact["call_index"] != index or artifact["route"] not in {"dvae", "decoder"} or not isinstance(artifact["path"], str) or artifact["path"] in artifact_paths:
            raise ValueError("malformed tensor artifact record")
        if not isinstance(artifact["bytes"], int) or isinstance(artifact["bytes"], bool) or artifact["bytes"] <= 0 or not isinstance(artifact["sha256"], str) or len(artifact["sha256"]) != 64 or any(character not in "0123456789abcdef" for character in artifact["sha256"].lower()):
            raise ValueError("malformed tensor artifact identity")
        artifact_paths.add(artifact["path"])
        tensor_source = artifact["source"]
        if not isinstance(tensor_source, dict) or set(tensor_source) != {"name", "shape", "numel", "dtype", "finite"} or not tensor_source["finite"]:
            raise ValueError("tensor artifact is non-finite or missing shape")
        shape = tensor_source["shape"]
        if not isinstance(shape, list) or not shape or len(shape) > 4 or any(not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0 for dimension in shape) or not isinstance(tensor_source["numel"], int) or isinstance(tensor_source["numel"], bool) or _checked_numel(shape) != tensor_source["numel"]:
            raise ValueError("tensor artifact numel does not match checked shape product")
        path = _artifact_path(root, artifact["path"])
        raw = path.read_bytes()
        if artifact["bytes"] > 256 * 1024 * 1024 or len(raw) != artifact["bytes"] or len(raw) != tensor_source["numel"] * 4 or not _finite_f32(raw) or hashlib.sha256(raw).hexdigest() != artifact["sha256"]:
            raise ValueError("tensor artifact identity mismatch")
    artifact_refs: set[str] = set()
    def collect_refs(value: Any) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("path"), str):
                artifact_refs.add(value["path"])
            for child in value.values():
                collect_refs(child)
        elif isinstance(value, list):
            for child in value:
                collect_refs(child)
    collect_refs(evidence["taps"])
    collect_refs(evidence["rng_records"])
    pcm_paths = {record["path"] for record in evidence["routes"].values()}
    if artifact_refs != artifact_paths or artifact_paths & pcm_paths:
        raise ValueError("tensor artifacts contain stale/extra or unreferenced files")
    expected_files = artifact_paths | pcm_paths
    actual_files = {path.relative_to(root).as_posix() for path in root.iterdir() if path.is_file() and path.name != "manifest.json"}
    if actual_files != expected_files:
        raise ValueError("evidence directory contains stale/orphan/extra files")
    if not isinstance(evidence["rng_records"], list) or len(evidence["rng_records"]) != 2:
        raise ValueError("exactly one GPT RNG record per official route is required")
    routes_seen: set[str] = set()
    for index, record in enumerate(evidence["rng_records"]):
        if not isinstance(record, dict) or set(record) != {"call_index", "route", "source", "call_site", "execution_id", "probabilities", "sample_ids"} or record.get("execution_id") != evidence["execution_id"] or record.get("call_index") != index or record.get("source") != "ChatTTS/model/gpt.py::GPT.generate" or record.get("route") not in {"dvae", "decoder"} or record["route"] in routes_seen or record["call_site"] != "torch.multinomial":
            raise ValueError("GPT RNG call order/source is not authenticated")
        routes_seen.add(record["route"])
        for key in ("probabilities", "sample_ids"):
            artifact = record[key]
            if not isinstance(artifact, dict) or artifact.get("path") not in artifact_paths:
                raise ValueError("GPT RNG artifact reference is missing")
        if record["probabilities"]["source"].get("shape") != [4, 626] or record["probabilities"]["source"].get("numel") != 4 * 626:
            raise ValueError("GPT probability capture must be the four-codebook [4,626] row")
        if record["sample_ids"]["source"].get("shape") != [4, 1] or record["sample_ids"]["source"].get("dtype") not in {"torch.int64", "torch.int32"}:
            raise ValueError("GPT sampled IDs must be integer [4,1] tensors")
    taps = evidence["taps"]
    expected_taps = {"tokenizer_encode", "generated", "dvae_embed_output", "dvae_dvae_output", "dvae_vocos_output", "decoder_embed_output", "decoder_output", "decoder_vocos_output"}
    if not isinstance(taps, dict) or set(taps) != expected_taps or set(taps.get("tokenizer_encode", {})) != {"dvae", "decoder"} or set(taps.get("generated", {})) != {"dvae", "decoder"}:
        raise ValueError("per-route tokenizer/generated taps are missing")
    for route, required in {
        "dvae": {"dvae_embed_output", "dvae_dvae_output", "dvae_vocos_output"},
        "decoder": {"decoder_embed_output", "decoder_output", "decoder_vocos_output"},
    }.items():
        if any(key not in taps or not isinstance(taps[key].get(route), list) or len(taps[key][route]) != 1 for key in required):
            raise ValueError(f"official {route} tap set is incomplete")
        generated = taps["generated"][route]
        if not isinstance(generated, list) or len(generated) != 1 or generated[0].get("count") != 1:
            raise ValueError(f"official {route} must contain exactly one generated step")
        if generated[0].get("execution_id") != evidence["execution_id"] or generated[0].get("call_index") != 0:
            raise ValueError(f"official {route} generated call identity is missing")
        if not isinstance(generated[0].get("ids"), list) or len(generated[0]["ids"]) != 1 or not isinstance(generated[0].get("hiddens"), list) or len(generated[0]["hiddens"]) != 1:
            raise ValueError(f"official {route} generated IDs/hiddens are missing")
        id_record = generated[0]["ids"][0]
        if id_record.get("source", {}).get("shape") != [1, 4] or id_record["source"].get("dtype") not in {"torch.int64", "torch.int32"}:
            raise ValueError(f"official {route} generated ID time axis/type is not source-exact")
        _validate_discrete_f32_artifact(id_record, root, 625, f"{route} generated IDs", {"torch.int64", "torch.int32"})
    for record in evidence["rng_records"]:
        _validate_discrete_f32_artifact(record["sample_ids"], root, 625, f"{record['route']} sampled IDs", {"torch.int64", "torch.int32"})
    for route in ("dvae", "decoder"):
        tokenizer = taps["tokenizer_encode"][route]
        if not isinstance(tokenizer, list) or len(tokenizer) != 1 or set(tokenizer[0]) != {"input_ids", "attention_mask", "text_mask", "execution_id", "call_index"} or tokenizer[0]["execution_id"] != evidence["execution_id"] or tokenizer[0]["call_index"] != 0 or any(not isinstance(tokenizer[0][name], dict) or tokenizer[0][name].get("path") not in artifact_paths for name in ("input_ids", "attention_mask", "text_mask")):
            raise ValueError(f"official {route} tokenizer tap schema is incomplete")
        input_source = tokenizer[0]["input_ids"]["source"]
        if input_source.get("dtype") not in {"torch.int64", "torch.int32"} or not isinstance(input_source.get("shape"), list) or len(input_source["shape"]) != 3 or input_source["shape"][-1] != 4:
            raise ValueError(f"official {route} input IDs are not integer [batch,time,4]")
        _validate_discrete_f32_artifact(tokenizer[0]["input_ids"], root, 21_177, f"{route} input IDs", {"torch.int64", "torch.int32"})
        for mask_name in ("attention_mask", "text_mask"):
            if tokenizer[0][mask_name]["source"].get("dtype") not in {"torch.bool", "torch.int64", "torch.int32"}:
                raise ValueError(f"official {route} {mask_name} is not discrete")
            mask_shape = tokenizer[0][mask_name]["source"].get("shape")
            if not isinstance(mask_shape, list) or len(mask_shape) != 2 or mask_shape != input_source["shape"][:2]:
                raise ValueError(f"official {route} {mask_name} shape is not aligned to input IDs")
            _validate_discrete_f32_artifact(tokenizer[0][mask_name], root, 1, f"{route} {mask_name}", {"torch.bool", "torch.int64", "torch.int32"})
        for key in expected_taps - {"tokenizer_encode", "generated"}:
            if key.startswith(route + "_") and taps[key][route][0].get("execution_id") != evidence["execution_id"]:
                raise ValueError(f"official {route} tap execution identity is missing")


def run_official(snapshot: Path, source: Path, server_tree: Path, request: dict[str, Any], output: Path) -> dict[str, Any]:
    """Call the actual pinned ChatTTS source and capture its returned values."""
    if os.environ.get("VOKRA_CHATTS_RUN_REFERENCE") != "1":
        raise RuntimeError("official ChatTTS execution requires VOKRA_CHATTS_RUN_REFERENCE=1 on VAST")
    project_identity = reference_project_identity()
    execution_id = digest(request)
    sys.path.insert(0, str(Path(__file__).parent))
    from chattts_inspect import inspect_model, inspect_source

    model_evidence = inspect_model(snapshot, server_tree)
    source_evidence = inspect_source(source)
    gpt_source = (source / "ChatTTS/model/gpt.py").read_text(encoding="utf-8")
    if "torch.multinomial" not in gpt_source or "manual_seed" not in gpt_source:
        raise RuntimeError("fixed GPT source randomness callsite could not be authenticated")
    sys.path.insert(0, str(source))
    try:
        import torch
        from ChatTTS.core import Chat
    except Exception as error:
        raise RuntimeError(f"pinned ChatTTS dependencies unavailable: {error}") from error

    chat = Chat()
    if not chat.load(source="custom", custom_path=str(snapshot), device=torch.device("cpu"), enable_cache=True):
        raise RuntimeError("upstream Chat.load(custom=...) rejected the authenticated asset bundle")
    output.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    rng_records: list[dict[str, Any]] = []
    active_route = "dvae"
    rng_active = False

    def save_tensor(value: Any, label: str, route: str) -> dict[str, Any]:
        record = _tensor_record(value, label)
        if record["numel"] > 65_536:
            raise ValueError(f"upstream tap {label} exceeds binary evidence bound")
        tensor = value.detach().to(torch.float32).cpu().contiguous()
        raw = tensor.numpy().tobytes()
        relative = f"artifact-{len(artifacts):04d}-{label}.f32le"
        (output / relative).write_bytes(raw)
        artifact = {"path": relative, "source": record, "storage_dtype": "float32", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "execution_id": execution_id, "call_index": len(artifacts), "route": route}
        artifacts.append(artifact)
        return artifact

    taps: dict[str, Any] = {}
    original_encode = chat.tokenizer.encode
    original_embed = chat.embed.forward
    original_generate = chat.gpt.generate
    original_dvae = chat.dvae.forward
    original_vocos = chat.vocos.decode

    def encode(*args: Any, **kwargs: Any) -> Any:
        result = original_encode(*args, **kwargs)
        _validate_discrete(result[0], f"{active_route} input_ids", 21_177)
        _validate_discrete(result[1], f"{active_route} attention_mask", 1, mask=True)
        _validate_discrete(result[2], f"{active_route} text_mask", 1, mask=True)
        record = {"input_ids": save_tensor(result[0], f"{active_route}_input_ids", active_route), "attention_mask": save_tensor(result[1], f"{active_route}_attention_mask", active_route), "text_mask": save_tensor(result[2], f"{active_route}_text_mask", active_route), "execution_id": execution_id, "call_index": len(taps.setdefault("tokenizer_encode", {}).setdefault(active_route, []))}
        taps["tokenizer_encode"][active_route].append(record)
        return result

    def embed(*args: Any, **kwargs: Any) -> Any:
        result = original_embed(*args, **kwargs)
        taps.setdefault(f"{active_route}_embed_output", {}).setdefault(active_route, []).append({"path": save_tensor(result, f"{active_route}_embed_output", active_route)["path"], "execution_id": execution_id, "call_index": len(taps[f"{active_route}_embed_output"][active_route])})
        return result

    def generate(*args: Any, **kwargs: Any) -> Any:
        nonlocal rng_active
        kwargs["return_hidden"] = True
        rng_active = True
        try:
            for result in original_generate(*args, **kwargs):
                for item in result.ids:
                    _validate_discrete(item, f"{active_route} generated IDs", 625)
                ids = [save_tensor(item, f"{active_route}_generated_ids_{index}", active_route) for index, item in enumerate(result.ids)]
                hiddens = [save_tensor(item, f"{active_route}_gpt_last_hidden_{index}", active_route) for index, item in enumerate(result.hiddens)]
                taps.setdefault("generated", {}).setdefault(active_route, []).append({"route": active_route, "ids": ids, "hiddens": hiddens, "count": len(result.ids), "execution_id": execution_id, "call_index": len(taps["generated"][active_route])})
                yield result
        finally:
            rng_active = False

    def dvae(*args: Any, **kwargs: Any) -> Any:
        result = original_dvae(*args, **kwargs)
        taps.setdefault("dvae_dvae_output", {}).setdefault(active_route, []).append({"path": save_tensor(result, "dvae_dvae_output", active_route)["path"], "execution_id": execution_id, "call_index": len(taps["dvae_dvae_output"][active_route])})
        return result

    original_decoder = chat.decoder.forward

    def decoder(*args: Any, **kwargs: Any) -> Any:
        result = original_decoder(*args, **kwargs)
        taps.setdefault("decoder_output", {}).setdefault(active_route, []).append({"path": save_tensor(result, "decoder_output", active_route)["path"], "execution_id": execution_id, "call_index": len(taps["decoder_output"][active_route])})
        return result

    def vocos(*args: Any, **kwargs: Any) -> Any:
        result = original_vocos(*args, **kwargs)
        taps.setdefault(f"{active_route}_vocos_output", {}).setdefault(active_route, []).append({"path": save_tensor(result, f"{active_route}_vocos_output", active_route)["path"], "execution_id": execution_id, "call_index": len(taps[f"{active_route}_vocos_output"][active_route])})
        return result

    chat.tokenizer.encode = encode
    chat.embed.forward = embed
    chat.gpt.generate = generate
    chat.dvae.forward = dvae
    chat.decoder.forward = decoder
    chat.vocos.decode = vocos
    original_multinomial = torch.multinomial

    def multinomial(*args: Any, **kwargs: Any) -> Any:
        result = original_multinomial(*args, **kwargs)
        if rng_active:
            probabilities = args[0] if args else kwargs.get("input")
            if probabilities is None:
                raise RuntimeError("torch.multinomial call had no probability input")
            if tuple(probabilities.shape) != (4, 626):
                raise RuntimeError(f"ChatTTS code probability shape must be [4,626], got {tuple(probabilities.shape)}")
            probability_artifact = save_tensor(probabilities, f"{active_route}_rng_{len(rng_records)}_probabilities", active_route)
            if tuple(result.shape) != (4, 1):
                raise RuntimeError(f"ChatTTS sampled code shape must be [4,1], got {tuple(result.shape)}")
            _validate_discrete(result, f"{active_route} sampled IDs", 625)
            sample_artifact = save_tensor(result, f"{active_route}_rng_{len(rng_records)}_sample_ids", active_route)
            rng_records.append({"call_index": len(rng_records), "route": active_route, "source": "ChatTTS/model/gpt.py::GPT.generate", "call_site": "torch.multinomial", "execution_id": execution_id, "probabilities": probability_artifact, "sample_ids": sample_artifact})
        return result

    torch.multinomial = multinomial
    try:
        params = Chat.InferCodeParams(
            temperature=request["temperature"],
            top_P=request["top_p"],
            top_K=request["top_k"],
            repetition_penalty=request["repetition_penalty"],
            max_new_token=request["max_new_token"],
            # A one-position trace must exercise the intended sampling call;
            # do not let an immediate EOS/retry path decide whether it ran.
            min_new_token=1,
            manual_seed=request["seed"],
            show_tqdm=False,
            ensure_non_empty=True,
        )
        route_results = {}
        for active_route, use_decoder in (("dvae", False), ("decoder", True)):
            route_results[active_route] = chat.infer(
                request["text"],
                split_text=False,
                skip_refine_text=True,
                use_decoder=use_decoder,
                params_infer_code=params,
            )
    finally:
        torch.multinomial = original_multinomial
        chat.tokenizer.encode = original_encode
        chat.embed.forward = original_embed
        chat.gpt.generate = original_generate
        chat.dvae.forward = original_dvae
        chat.decoder.forward = original_decoder
        chat.vocos.decode = original_vocos
    route_pcm: dict[str, dict[str, Any]] = {}
    for route, result in route_results.items():
        if not isinstance(result, list) or len(result) != 1:
            raise RuntimeError(f"upstream Chat.infer did not return one {route} waveform")
        waveform = result[0]
        if getattr(waveform, "ndim", 0) != 1 or not bool(waveform.size):
            raise RuntimeError(f"upstream Chat.infer returned an empty/non-mono {route} waveform")
        if not bool(torch.isfinite(torch.as_tensor(waveform)).all().item()):
            raise RuntimeError(f"upstream {route} waveform contains non-finite samples")
        pcm = torch.as_tensor(waveform, dtype=torch.float32).cpu().contiguous().numpy()
        relative = f"official_{route}_pcm.f32le"
        raw = pcm.tobytes()
        (output / relative).write_bytes(raw)
        route_pcm[route] = {"sample_rate_hz": SAMPLE_RATE_HZ, "samples": int(pcm.size), "dtype": "float32", "bytes": len(raw), "path": relative, "sha256": hashlib.sha256(raw).hexdigest(), "execution_id": execution_id}
    required_taps = {"tokenizer_encode", "generated", "dvae_embed_output", "dvae_dvae_output", "dvae_vocos_output", "decoder_embed_output", "decoder_output", "decoder_vocos_output"}
    if not required_taps.issubset(taps):
        raise RuntimeError(f"official composite tap set is incomplete: missing={sorted(required_taps - set(taps))}")
    if len(rng_records) != 2 or {record["route"] for record in rng_records} != {"dvae", "decoder"}:
        raise RuntimeError("official GPT one-step packet did not consume exactly one torch.multinomial call per route")
    evidence = {"model": model_evidence, "source": source_evidence, "reference_project": project_identity, "request": request, "execution_id": execution_id, "sample_rate_hz": SAMPLE_RATE_HZ, "routes": route_pcm, "artifacts": artifacts, "taps": taps, "rng_records": rng_records}
    validate_reference_evidence(output, evidence)
    return evidence


def write_blocked(output: Path, status: str, error: str | None = None, evidence: dict[str, Any] | None = None) -> None:
    manifest: dict[str, Any] = {
        "format": FORMAT,
        "status": "BLOCKED",
        "inspection_status": status,
        "evidence_stage": "REFERENCE_INSPECTION_ONLY",
        "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
        "native_status": "BLOCKED_NATIVE_BINDING",
        "cpu_status": "UNSUPPORTED",
        "metal_status": "BLOCKED_BY_CPU",
        "parity_status": "NOT_RUN",
        "publication": "NO_UPLOAD",
        "license_evidence": {"weights": "CC-BY-NC-4.0", "source": "AGPLv3+", "dependencies": "REVIEW_REQUIRED_BLOCKER"},
        "model": {"repository": HF_REPOSITORY, "revision": HF_REVISION},
        "source": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION, "license": "AGPLv3+"},
    }
    if error:
        manifest["error"] = error
    if evidence:
        manifest["evidence"] = evidence
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def self_test() -> None:
    dependency_gate_fixture_tests()
    base = {"text": "hello", "seed": 7, "max_new_token": 1, "temperature": 0.3, "top_p": 0.7, "top_k": 20, "repetition_penalty": 1.05}
    assert validate_request(base)["seed"] == 7
    bad = dict(base, max_new_token=0)
    try:
        validate_request(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted invalid max_new_token")
    try:
        validate_request(dict(base, max_new_token=2))
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a multi-step request for the one-step packet")
    bad = dict(base, text="\x00")
    try:
        validate_request(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted NUL text")
    assert _checked_numel([2, 3]) == 6
    try:
        _checked_numel([4_097, 4_097])
    except ValueError:
        pass
    else:
        raise AssertionError("accepted an unbounded tensor shape")
    import tempfile
    with tempfile.TemporaryDirectory(prefix="vokra-chattts-ref-") as directory:
        out = Path(directory) / "evidence"
        write_blocked(out, "INSPECTION_ERROR", "fixture")
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "BLOCKED" and manifest["publication"] == "NO_UPLOAD"
        assert manifest["inspection_status"] == "INSPECTION_ERROR"
        for invalid_value in (1.25, 21_178.0):
            bad_path = out / f"bad-{invalid_value}.f32le"
            bad_path.write_bytes(struct.pack("<f", invalid_value))
            bad_record = {
                "path": bad_path.name,
                "source": {"dtype": "torch.int64", "numel": 1},
            }
            try:
                _validate_discrete_f32_artifact(bad_record, out, 21_177, "bad input IDs", {"torch.int64"})
            except ValueError:
                pass
            else:
                raise AssertionError("accepted fractional/out-of-range stored ID")
            bad_path.unlink()
        raw = b"\0" * 8
        (out / "dvae_pcm.f32le").write_bytes(raw)
        (out / "decoder_pcm.f32le").write_bytes(raw)
        valid_artifacts = []
        def fixture_artifact(name: str, shape: list[int], route: str, dtype: str = "torch.float32") -> dict[str, Any]:
            raw_value = b"\0" * (_checked_numel(shape) * 4)
            path = f"artifact-{len(valid_artifacts):04d}-{name}.f32le"
            (out / path).write_bytes(raw_value)
            record = {"name": name, "shape": shape, "numel": _checked_numel(shape), "dtype": dtype, "finite": True}
            artifact = {"path": path, "source": record, "storage_dtype": "float32", "bytes": len(raw_value), "sha256": hashlib.sha256(raw_value).hexdigest(), "execution_id": digest(base), "call_index": len(valid_artifacts), "route": route}
            valid_artifacts.append(artifact)
            return artifact
        taps: dict[str, Any] = {"tokenizer_encode": {}, "generated": {}}
        rng_records = []
        for route in ("dvae", "decoder"):
            tokenizer = {name: fixture_artifact(f"{route}_{name}", [1, 1, 4] if name == "input_ids" else [1, 1], route, "torch.int64") for name in ("input_ids", "attention_mask", "text_mask")}
            embed = fixture_artifact(f"{route}_embed_output", [1, 2], route)
            ids = fixture_artifact(f"{route}_generated_ids", [1, 4], route, "torch.int64")
            hidden = fixture_artifact(f"{route}_hidden", [1, 2], route)
            probability = fixture_artifact(f"{route}_probabilities", [4, 626], route)
            samples = fixture_artifact(f"{route}_sample_ids", [4, 1], route, "torch.int64")
            taps["tokenizer_encode"][route] = [{**tokenizer, "execution_id": digest(base), "call_index": 0}]
            taps["generated"][route] = [{"route": route, "ids": [ids], "hiddens": [hidden], "count": 1, "execution_id": digest(base), "call_index": 0}]
            taps[f"{route}_embed_output"] = {route: [{"path": embed["path"], "execution_id": digest(base), "call_index": 0}]}
            output_key = "dvae_dvae_output" if route == "dvae" else "decoder_output"
            taps[output_key] = {route: [{"path": fixture_artifact(f"{route}_decode_output", [1, 2], route)["path"], "execution_id": digest(base), "call_index": 0}]}
            taps[f"{route}_vocos_output"] = {route: [{"path": fixture_artifact(f"{route}_vocos_output", [1, 2], route)["path"], "execution_id": digest(base), "call_index": 0}]}
            rng_records.append({"call_index": len(rng_records), "route": route, "source": "ChatTTS/model/gpt.py::GPT.generate", "call_site": "torch.multinomial", "execution_id": digest(base), "probabilities": probability, "sample_ids": samples})
        for route in ("dvae", "decoder"):
            (out / f"{route}_pcm.f32le").write_bytes(raw)
        from chattts_inspect import SELECTED
        fixed_rows = [{"path": name, "bytes": size, "git_blob_sha1": blob, "lfs_sha256": lfs, "local_verified": True} for name, (size, blob, lfs) in SELECTED.items()]
        fixed_rows.extend({"path": f"legacy-{index}", "bytes": 1, "git_blob_sha1": "a" * 40, "lfs_sha256": None, "local_verified": False} for index in range(14))
        selected_rows = {name: {"bytes": size, "sha256": "a" * 64} for name, (size, _blob, _lfs) in SELECTED.items()}
        selected_rows["README.md"] = {}
        evidence = {"model": {"repository": HF_REPOSITORY, "revision": HF_REVISION, "resolved_revision": HF_REVISION, "files": fixed_rows, "selected": selected_rows}, "source": {"repository": SOURCE_REPOSITORY, "origin": SOURCE_REPOSITORY, "revision": SOURCE_REVISION, "tag": "v0.2.5", "clean": True, "roles": {role: {"sha256": "a" * 64, "git_blob_sha1": blob} for role, blob in SOURCE_ROLE_BLOBS.items()}}, "reference_project": {"python": ">=3.12,<3.13", "pyproject_sha256": "a" * 64, "uv_lock_sha256": "b" * 64, "package_inventory": expected_lock_inventory(), "package_inventory_sha256": REFERENCE_PACKAGE_INVENTORY_SHA256, "package_lock_rows": expected_lock_rows(), "package_lock_rows_sha256": REFERENCE_LOCK_PACKAGE_ROWS_SHA256, "license_audit_sha256": REFERENCE_LICENSE_AUDIT_SHA256, "dependencies": PROJECT_VERSIONS, "actual_versions": PROJECT_VERSIONS}, "request": base, "execution_id": digest(base), "sample_rate_hz": SAMPLE_RATE_HZ, "routes": {route: {"sample_rate_hz": SAMPLE_RATE_HZ, "samples": 2, "dtype": "float32", "bytes": 8, "path": f"{route}_pcm.f32le", "sha256": hashlib.sha256(raw).hexdigest(), "execution_id": digest(base)} for route in ("dvae", "decoder")}, "artifacts": valid_artifacts, "taps": taps, "rng_records": rng_records}
        try:
            validate_reference_evidence(out, {**evidence, "model": {}, "source": {}})
        except ValueError:
            pass
        else:
            raise AssertionError("accepted empty model/source evidence")
        validate_reference_evidence(out, evidence)
        for invalid in (
            dict(evidence, taps={}),
            dict(evidence, rng_records=[evidence["rng_records"][0]]),
            dict(evidence, routes={"dvae": evidence["routes"]["dvae"], "decoder": dict(evidence["routes"]["decoder"], path="../escape.f32le")}),
        ):
            try:
                validate_reference_evidence(out, invalid)
            except (ValueError, FileNotFoundError):
                pass
            else:
                raise AssertionError("accepted incomplete or unsafe reference evidence")
        malformed = dict(evidence, artifacts=[dict(valid_artifacts[0], source=dict(valid_artifacts[0]["source"], numel=3))] + valid_artifacts[1:])
        try:
            validate_reference_evidence(out, malformed)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted binary artifact shape mismatch")
    print("chattts_dump_reference.py self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--server-tree", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--text", default="Hello.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-token", type=int, default=1)
    parser.add_argument("--dependency-gate", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.dependency_gate:
        try:
            validate_dependency_gate()
        except RuntimeError as error:
            print(f"ChatTTS dependency/license gate: BLOCKED: {error}", file=sys.stderr)
            return 2
        print(
            "ChatTTS dependency/license gate: APPROVED "
            f"lock={REFERENCE_LOCK_SHA256} inventory={REFERENCE_PACKAGE_INVENTORY_SHA256} "
            f"lock_rows={REFERENCE_LOCK_PACKAGE_ROWS_SHA256} license_audit={REFERENCE_LICENSE_AUDIT_SHA256}"
        )
        return 0
    if args.self_test:
        self_test()
        return 0
    if not all((args.snapshot, args.source, args.server_tree, args.output)):
        parser.error("reference requires --snapshot --source --server-tree --output")
    request = {"text": args.text, "seed": args.seed, "max_new_token": args.max_new_token, "temperature": 0.3, "top_p": 0.7, "top_k": 20, "repetition_penalty": 1.05}
    if args.output.exists():
        if not args.output.is_dir() or any(args.output.iterdir()):
            parser.error("--output must be absent or empty")
    else:
        args.output.mkdir(parents=True)
    try:
        evidence = run_official(args.snapshot, args.source, args.server_tree, validate_request(request), args.output)
        write_blocked(args.output, "AUTHENTICATED_REFERENCE_EVIDENCE", evidence=evidence)
    except Exception as error:
        write_blocked(args.output, "INSPECTION_ERROR", f"{type(error).__name__}: {error}")
        print(f"ChatTTS reference blocked: {error}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
