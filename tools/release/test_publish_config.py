#!/usr/bin/env python3
"""Oracle for explicit, fail-closed release publication configuration."""

from __future__ import annotations

import os
import subprocess
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHECK = os.path.join(ROOT, "tools", "release", "publish_config.py")
RELEASE = os.path.join(ROOT, ".github", "workflows", "release.yml")
HANDOFF = os.path.join(ROOT, "docs", "handoff", "x-07.md")
VARIABLES = (
    "CRATES_IO_PUBLISH_ENABLED",
    "NPM_PUBLISH_ENABLED",
    "OPENUPM_PUBLISH_ENABLED",
    "PYPI_PUBLISH_ENABLED",
    "DESKTOP_AAR_ENABLED",
)


def run(strict: bool, values: dict[str, str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    for name in (*VARIABLES, "CRATES_IO_TOKEN", "NPM_TOKEN", "OPENUPM_TOKEN", "PYPI_API_TOKEN", "PYPI_TRUSTED_PUBLISHER_CONFIGURED"):
        env.pop(name, None)
    env.update(values)
    return subprocess.run(
        [sys.executable, CHECK, "--strict", str(strict).lower()],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def main() -> None:
    all_disabled = {name: "false" for name in VARIABLES}
    checks: list[tuple[bool, str]] = []
    checks.append((run(True, all_disabled).returncode == 0, "explicit GitHub-only release is valid"))

    missing = run(True, {})
    checks.append((missing.returncode != 0 and "must be explicitly true or false" in missing.stderr, "missing decisions fail strict preflight"))

    crates_missing = run(True, {**all_disabled, "CRATES_IO_PUBLISH_ENABLED": "true"})
    checks.append((crates_missing.returncode != 0 and "requires secret CRATES_IO_TOKEN" in crates_missing.stderr, "enabled crates.io requires token"))

    crates_ready = run(True, {**all_disabled, "CRATES_IO_PUBLISH_ENABLED": "true", "CRATES_IO_TOKEN": "test-only"})
    checks.append((crates_ready.returncode == 0, "enabled crates.io accepts present token"))

    pypi_oidc = run(True, {**all_disabled, "PYPI_PUBLISH_ENABLED": "true", "PYPI_TRUSTED_PUBLISHER_CONFIGURED": "true"})
    checks.append((pypi_oidc.returncode == 0, "PyPI OIDC owner attestation satisfies preflight"))

    release = open(RELEASE, encoding="utf-8").read()
    handoff = open(HANDOFF, encoding="utf-8").read()
    checks.extend(
        [
            ("enforce_publish_config:" in release, "workflow exposes strict dry-run input"),
            ("tools/release/publish_config.py" in release, "workflow runs publish preflight"),
            ("needs: [validate-tag, release-config]" in release, "Release creation waits for config"),
            ('name: pypi' in release and 'url: https://pypi.org/p/vokra' in release, "PyPI job declares trusted-publisher environment"),
            (all(name in handoff for name in VARIABLES), "handoff lists every channel decision"),
            ("PYPI_TRUSTED_PUBLISHER_CONFIGURED" in handoff, "handoff records PyPI OIDC attestation"),
        ]
    )

    failures = [label for passed, label in checks if not passed]
    for passed, label in checks:
        print(f"  {'ok' if passed else 'FAIL'}: {label}")
    print()
    print(f"publish-config oracle: {len(checks) - len(failures)} passed, {len(failures)} failed")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
