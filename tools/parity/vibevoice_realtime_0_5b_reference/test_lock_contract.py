"""Static security checks for the model-free Realtime reference lock.

This test deliberately reads only the project metadata and lock.  It must not
install packages, import torch, download an upstream checkout, or construct a
checkpoint.  The real API smoke remains a separate VAST-only operation.
"""

from __future__ import annotations

import hashlib
import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "pyproject.toml"
LOCK = ROOT / "uv.lock"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
LINUX_X86_MARKER = "platform_machine == 'x86_64' and sys_platform == 'linux'"


def test_torch_is_explicitly_cpu_only() -> None:
    project = tomllib.loads(PROJECT.read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]
    torch_requirement = next(item for item in dependencies if item.startswith("torch=="))
    assert "torch==2.13.0" in torch_requirement
    assert LINUX_X86_MARKER in torch_requirement

    uv = project["tool"]["uv"]
    assert uv["sources"]["torch"] == {"index": "pytorch-cpu"}
    index = next(item for item in uv["index"] if item["name"] == "pytorch-cpu")
    assert index["url"] == CPU_INDEX
    assert index["explicit"] is True


def test_lock_has_no_unreviewed_cuda_payload() -> None:
    lock_text = LOCK.read_text(encoding="utf-8")
    lock = tomllib.loads(lock_text)
    torch = next(package for package in lock["package"] if package["name"] == "torch")
    assert torch["version"] == "2.13.0+cpu"
    assert torch["source"] == {"registry": CPU_INDEX}

    package_names = [package["name"] for package in lock["package"]]
    assert not [name for name in package_names if name.startswith("nvidia-")]
    assert not re.search(r"(?i)(cuda|cudnn|cublas|cufft|nccl|nvjitlink)", lock_text)


def test_lock_metadata_declares_cpu_security_boundary() -> None:
    project = tomllib.loads(PROJECT.read_text(encoding="utf-8"))
    reference = project["tool"]["vokra"]["reference"]
    assert reference["torch_distribution"] == "2.13.0+cpu"
    assert reference["torch_index"] == CPU_INDEX
    assert reference["dependency_lock_sha256"] == hashlib.sha256(LOCK.read_bytes()).hexdigest()
    assert reference["cuda_native_payload"] == "EXCLUDED_UNREVIEWED"
    assert reference["gpu_timing_status"] == "DEFERRED_CPU_ONLY_SECURITY_LOCK"
