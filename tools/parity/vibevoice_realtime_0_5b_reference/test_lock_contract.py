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
OLD_LOCK_SHA256 = "cbf0ce675cdc8bc3c8cd32a4528f3a67f283b2e7b46841cb6dc32e4949af666e"
COMPATIBILITY_SMOKE_SHA256 = "1d5f9d037ef15e8cded3db06d323d86de4ef5e08bd9fc52e00f96655e241a189"
INSTALLED_CLOSURE_SHA256 = "23bc546d7fcf47f1dc3b66f587c53d37b8025e5b418c185148f3201f1417e06e"


def test_torch_is_explicitly_cpu_only() -> None:
    project = tomllib.loads(PROJECT.read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]
    torch_requirement = next(item for item in dependencies if item.startswith("torch=="))
    assert torch_requirement == f"torch==2.13.0 ; {LINUX_X86_MARKER}"
    assert [item for item in dependencies if item.startswith("torch")] == [torch_requirement]

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
    assert reference["dependency_lock_status"] == "IDENTITY_ENRICHED_VAST_NOT_RUN_OWNER_REVIEW_REQUIRED"
    assert reference["historical_compatibility_smoke_dependency_lock_sha256"] == OLD_LOCK_SHA256
    assert reference["historical_installed_closure_dependency_lock_sha256"] == OLD_LOCK_SHA256
    assert reference["compatibility_smoke_evidence_sha256"] == COMPATIBILITY_SMOKE_SHA256
    assert reference["installed_closure_audit_sha256"] == INSTALLED_CLOSURE_SHA256
    assert reference["cuda_native_payload"] == "EXCLUDED_UNREVIEWED"
    assert reference["gpu_timing_status"] == "DEFERRED_CPU_ONLY_SECURITY_LOCK"


def test_all_locked_wheels_have_declared_hash_and_positive_size() -> None:
    lock = tomllib.loads(LOCK.read_text(encoding="utf-8"))
    packages = [row for row in lock["package"] if "registry" in row.get("source", {})]
    assert len(packages) == 41
    for package in packages:
        for wheel in package.get("wheels", []):
            assert re.fullmatch(r"sha256:[0-9a-f]{64}", wheel["hash"])
            assert isinstance(wheel["size"], int) and wheel["size"] > 0
    selected = {
        "jinja2": (134899, "sha256:85ece4451f492d0c13c5dd7c13a64681a86afae63a5f347908daf103ce6d2f67"),
        "markupsafe": (22947, "sha256:d6dd0be5b5b189d31db7cda48b91d7e0a9795f31430b7f271219ab30f1d3ac9d"),
        "torch": (191817609, "sha256:4ca4a9394b0c771238a4f73590fdbbc4debad85ed0fa63d026ae1b085da7d6e2"),
    }
    for name, (size, digest) in selected.items():
        wheel = next(row for package in packages if package["name"] == name for row in package["wheels"])
        assert (wheel["size"], wheel["hash"]) == (size, digest)


def test_missing_wheel_hash_is_rejected() -> None:
    lock = tomllib.loads(LOCK.read_text(encoding="utf-8"))
    torch = next(row for row in lock["package"] if row["name"] == "torch")
    torch["wheels"][0].pop("hash")
    assert "hash" not in torch["wheels"][0]

    # This is intentionally a structural regression: the collector's public
    # lock validator must reject a registry row without an artifact digest.
    import importlib.util

    spec = importlib.util.spec_from_file_location("vokra_lock_audit", ROOT / "audit_installed_closure.py")
    assert spec is not None and spec.loader is not None
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    try:
        audit.validate_lock(lock)
    except audit.AuditError:
        return
    raise AssertionError("lock validator accepted a selected wheel without hash")


if __name__ == "__main__":
    test_torch_is_explicitly_cpu_only()
    test_lock_has_no_unreviewed_cuda_payload()
    test_lock_metadata_declares_cpu_security_boundary()
    test_all_locked_wheels_have_declared_hash_and_positive_size()
    test_missing_wheel_hash_is_rejected()
    print("vibevoice realtime CPU lock contract: OK")
