#!/usr/bin/env -S uv run --frozen --project tools/parity/owsm_v4_medium_1b_reference --python 3.12 python
"""Stdlib-only preflight for the OWSM official-reference dependency lock.

The lock is intentionally not an approval to install or execute.  Until each
transitive package has owner-reviewed primary-source license evidence, this
gate returns ``BLOCKED_UNREVIEWED_TRANSITIVE`` and the VAST worker must stop.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import tomllib
from pathlib import Path

EXPECTED_DIRECT = {
    "humanfriendly": "10.0",
    "librosa": "0.10.2.post1",
    "numpy": "2.2.5",
    "packaging": "24.2",
    "pyyaml": "6.0.3",
    "torch": "2.6.0",
    "torch-complex": "0.4.4",
    "typeguard": "4.6.0",
}
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
BLOCKED_STATUS = "BLOCKED_UNREVIEWED_TRANSITIVE"


def _dependency_name(specifier: str) -> str:
    return re.split(r"[<>=!~ ;]", specifier, maxsplit=1)[0].strip().lower()


def audit(project: Path) -> dict[str, object]:
    errors: list[str] = []
    pyproject_path = project / "pyproject.toml"
    lock_path = project / "uv.lock"
    if not pyproject_path.is_file() or not lock_path.is_file():
        return {"status": "BLOCKED_MISSING_LOCK_INPUTS", "errors": ["pyproject.toml and uv.lock are required"]}
    with pyproject_path.open("rb") as handle:
        metadata = tomllib.load(handle)
    with lock_path.open("rb") as handle:
        lock = tomllib.load(handle)
    gate = metadata.get("tool", {}).get("vokra", {}).get("owsm_v4_medium_1b_reference", {})
    audit_status = gate.get("dependency_license_audit")
    if audit_status not in {
        "PENDING_PRIMARY_SOURCE_AUDIT_FAIL_CLOSED",
        "AUDITED_ALLOW",
    }:
        errors.append("dependency audit status is neither pending fail-closed nor owner-approved")
    dependencies = metadata.get("project", {}).get("dependencies", [])
    direct = {_dependency_name(str(item)): str(item).split("==", 1)[1] for item in dependencies if "==" in str(item)}
    if direct != EXPECTED_DIRECT:
        errors.append(f"direct dependency pins mismatch: {direct!r}")
    packages = lock.get("package", [])
    versions: dict[str, set[str]] = {}
    torch_sources: list[str] = []
    package_names: set[str] = set()
    for package in packages:
        name = package.get("name")
        version = package.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            errors.append("lock contains a malformed package row")
            continue
        package_names.add(name)
        versions.setdefault(name, set()).add(version)
        if name == "torch":
            source = package.get("source", {})
            torch_sources.append(str(source.get("registry", "")))
    for name, expected in EXPECTED_DIRECT.items():
        if name not in package_names:
            errors.append(f"direct package missing from lock: {name}")
        elif name != "torch" and versions.get(name) != {expected}:
            errors.append(f"locked version mismatch for {name}: {sorted(versions.get(name, set()))!r}")
    if versions.get("torch") != {"2.6.0", "2.6.0+cpu"}:
        errors.append(f"torch CPU lock rows mismatch: {sorted(versions.get('torch', set()))!r}")
    if not torch_sources or any(source != CPU_INDEX for source in torch_sources):
        errors.append(f"torch is not exclusively bound to the CPU index: {torch_sources!r}")
    forbidden = sorted(
        name for name in package_names if any(token in name.lower() for token in ("triton", "nvidia", "cuda"))
    )
    if forbidden:
        errors.append(f"forbidden CUDA/Triton package rows: {forbidden!r}")
    if errors:
        status = "BLOCKED_LOCK_CONTRACT"
    elif audit_status == "AUDITED_ALLOW":
        status = "AUDITED_ALLOW"
    else:
        status = BLOCKED_STATUS
    return {
        "status": status,
        "project": str(project.resolve()),
        "lock_package_count": len(packages),
        "direct_dependencies": direct,
        "torch_sources": torch_sources,
        "errors": errors,
        "execution": "NO_LOCAL_EXECUTION",
        "publication": "NO_UPLOAD",
    }


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="owsm-dependency-gate-") as directory:
        project = Path(directory)
        (project / "pyproject.toml").write_text(
            """[project]\ndependencies = [\n"""
            + "\n".join(f'    "{name}=={version}",' for name, version in EXPECTED_DIRECT.items())
            + "\n]\n\n[tool.vokra.owsm_v4_medium_1b_reference]\ndependency_license_audit = \"PENDING_PRIMARY_SOURCE_AUDIT_FAIL_CLOSED\"\n",
            encoding="utf-8",
        )
        rows = []
        for name, version in EXPECTED_DIRECT.items():
            source = (
                '\nsource = { registry = "https://download.pytorch.org/whl/cpu" }'
                if name == "torch"
                else ""
            )
            rows.append(f'[[package]]\nname = "{name}"\nversion = "{version}"{source}\n')
        rows.append('[[package]]\nname = "torch"\nversion = "2.6.0+cpu"\nsource = { registry = "https://download.pytorch.org/whl/cpu" }\n')
        (project / "uv.lock").write_text("\n".join(rows), encoding="utf-8")
        result = audit(project)
        assert result["status"] == BLOCKED_STATUS, result
        (project / "uv.lock").write_text((project / "uv.lock").read_text().replace("2.6.0+cpu", "2.6.0+cuda"), encoding="utf-8")
        assert audit(project)["status"] == "BLOCKED_LOCK_CONTRACT"
    print("owsm_v4_medium_1b dependency/license gate self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--project", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if args.project is not None:
            parser.error("--self-test accepts no --project")
        self_test()
        return 0
    if args.project is None:
        parser.error("--project is required")
    result = audit(args.project)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 2 if result["status"] != "AUDITED_ALLOW" else 0


if __name__ == "__main__":
    raise SystemExit(main())
