#!/usr/bin/env python3
"""Extract the dependency closure roots used by the Python license audit."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import NoReturn

_ALLOWED_LICENSES = {
    "Apache-2.0",
    "Apache Software License",
    "Apache-2.0 OR BSD-2-Clause",
    "MIT",
    "MIT License",
    "BSD-3-Clause",
    "BSD License",
    "BSD-2-Clause",
    "PSF-2.0",
    "Python-2.0",
}
_PACKAGE_EXCEPTIONS = {
    (
        "pathspec",
        "Mozilla Public License 2.0 (MPL 2.0)",
    ),
}


def _fail(message: str) -> NoReturn:
    raise SystemExit(f"Python license requirement check failed: {message}")


def audit_requirements(pyproject: Path) -> list[str]:
    """Return build/optional requirements after enforcing zero runtime deps."""
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})
    runtime = project.get("dependencies", [])
    if runtime != []:
        _fail(f"project.dependencies must remain empty, got {runtime!r}")

    build = data.get("build-system", {}).get("requires", [])
    optional = project.get("optional-dependencies", {})
    if not isinstance(build, list) or not isinstance(optional, dict):
        _fail("build-system.requires must be a list and optional-dependencies a table")

    requirements = [*build]
    for group in sorted(optional):
        group_requirements = optional[group]
        if not isinstance(group_requirements, list):
            _fail(f"optional dependency group {group!r} must be a list")
        requirements.extend(group_requirements)

    if not all(isinstance(requirement, str) and requirement.strip() for requirement in requirements):
        _fail("all dependency declarations must be non-empty strings")
    if not any(requirement.lower().startswith("pip-licenses") for requirement in requirements):
        _fail("the dev dependency group must include pip-licenses")
    return sorted(set(requirements), key=str.lower)


def verify_report(report: Path) -> None:
    """Enforce the license allowlist plus documented package-scoped exceptions."""
    rows = json.loads(report.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        _fail("pip-licenses report must be a non-empty JSON list")
    rejected: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            _fail(f"pip-licenses report row must be an object, got {row!r}")
        name = row.get("Name")
        license_name = row.get("License")
        if not isinstance(name, str) or not isinstance(license_name, str):
            _fail(f"pip-licenses report row has invalid fields: {row!r}")
        if license_name in _ALLOWED_LICENSES:
            continue
        if (name.casefold(), license_name) in _PACKAGE_EXCEPTIONS:
            continue
        rejected.append(f"{name}={row.get('Version', '?')}: {license_name}")
    if rejected:
        _fail("unapproved licenses: " + "; ".join(sorted(rejected)))


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract")
    extract.add_argument("pyproject", type=Path)
    verify = subparsers.add_parser("verify-report")
    verify.add_argument("report", type=Path)
    args = parser.parse_args()
    if args.command == "extract":
        for requirement in audit_requirements(args.pyproject):
            print(requirement)
    else:
        verify_report(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
