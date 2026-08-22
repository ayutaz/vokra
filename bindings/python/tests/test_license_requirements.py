"""Tests for Python license-audit requirement extraction."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

if sys.version_info < (3, 11):
    pytest.skip("license audit tooling uses stdlib tomllib on Python 3.11+", allow_module_level=True)

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "license_requirements.py"
_SPEC = importlib.util.spec_from_file_location("vokra_license_requirements", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
license_requirements = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(license_requirements)


def test_audit_requirements_covers_build_and_optional_dependencies() -> None:
    requirements = license_requirements.audit_requirements(_ROOT / "pyproject.toml")
    assert "hatchling==1.32.0" in requirements
    assert "numpy>=1.23,<2" in requirements
    assert "pytest>=9.0.3; python_version >= '3.10'" in requirements
    assert "pip-licenses>=4.3" in requirements


def test_audit_requirements_rejects_runtime_dependency(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[build-system]
requires = ["hatchling==1.32.0"]
[project]
dependencies = ["requests"]
[project.optional-dependencies]
dev = ["pip-licenses>=4.3"]
""",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="must remain empty"):
        license_requirements.audit_requirements(pyproject)


def test_verify_report_allows_only_documented_pathspec_mpl_exception(tmp_path: Path) -> None:
    report = tmp_path / "licenses.json"
    report.write_text(
        json.dumps(
            [
                {"Name": "hatchling", "Version": "1.32.0", "License": "MIT"},
                {
                    "Name": "pathspec",
                    "Version": "1.1.1",
                    "License": "Mozilla Public License 2.0 (MPL 2.0)",
                },
            ]
        ),
        encoding="utf-8",
    )
    license_requirements.verify_report(report)


def test_verify_report_rejects_mpl_for_another_package(tmp_path: Path) -> None:
    report = tmp_path / "licenses.json"
    report.write_text(
        json.dumps(
            [
                {
                    "Name": "unexpected",
                    "Version": "1.0",
                    "License": "Mozilla Public License 2.0 (MPL 2.0)",
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="unapproved licenses"):
        license_requirements.verify_report(report)
