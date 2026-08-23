#!/usr/bin/env python3
"""Oracle for the release version single-source-of-truth contract."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONTRACT = os.path.join(ROOT, "tools", "release", "version_contract.py")
RELEASE_YML = os.path.join(ROOT, ".github", "workflows", "release.yml")
GODOT_BUILD = os.path.join(ROOT, "scripts", "build-godot-gdextension.sh")


def run(cargo_toml: str, ref: str, candidate: str = "") -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            CONTRACT,
            "--cargo-toml",
            cargo_toml,
            "--github-ref",
            ref,
            "--dispatch-version",
            candidate,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def main() -> None:
    checks: list[tuple[bool, str]] = []
    with tempfile.TemporaryDirectory(prefix="release-version-contract.") as scratch:
        cargo_toml = os.path.join(scratch, "Cargo.toml")
        with open(cargo_toml, "w", encoding="utf-8") as handle:
            handle.write('[workspace.package]\nversion = "1.2.3-rc.1"\n')

        matched = run(cargo_toml, "refs/tags/v1.2.3-rc.1")
        checks.append((matched.returncode == 0 and matched.stdout.strip() == "1.2.3-rc.1", "matching tag"))

        mismatch = run(cargo_toml, "refs/tags/v1.2.3")
        checks.append((mismatch.returncode != 0 and "does not match workspace" in mismatch.stderr, "mismatched tag fails"))

        default_dispatch = run(cargo_toml, "refs/heads/main")
        checks.append((default_dispatch.returncode == 0 and default_dispatch.stdout.strip() == "1.2.3-rc.1", "dispatch defaults to workspace"))

        candidate = run(cargo_toml, "refs/heads/release-prep", "2.0.0-beta.2")
        checks.append((candidate.returncode == 0 and candidate.stdout.strip() == "2.0.0-beta.2", "dispatch candidate"))

        invalid = run(cargo_toml, "refs/heads/main", "v2.0.0")
        checks.append((invalid.returncode != 0 and "unprefixed SemVer" in invalid.stderr, "invalid dispatch version fails"))

    release = open(RELEASE_YML, encoding="utf-8").read()
    checks.extend(
        [
            ("release_version:" in release, "workflow exposes dry-run candidate"),
            ("tools/release/version_contract.py" in release, "workflow calls version contract"),
            ("needs.validate-tag.outputs.version" in release, "jobs consume validated output"),
            ("npm version --no-git-tag-version --allow-same-version" in release, "Unity manifest is stamped"),
            ("VOKRA_RELEASE_VERSION" in release, "Godot pack receives validated version"),
            ("VOKRA_RELEASE_VERSION" in open(GODOT_BUILD, encoding="utf-8").read(), "Godot pack honors override"),
        ]
    )

    failures = [label for passed, label in checks if not passed]
    for passed, label in checks:
        print(f"  {'ok' if passed else 'FAIL'}: {label}")
    print()
    print(f"version-contract oracle: {len(checks) - len(failures)} passed, {len(failures)} failed")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
