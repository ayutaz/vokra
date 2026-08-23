#!/usr/bin/env python3
"""Oracle for the release version single-source-of-truth contract."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import tomllib


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONTRACT = os.path.join(ROOT, "tools", "release", "version_contract.py")
RELEASE_YML = os.path.join(ROOT, ".github", "workflows", "release.yml")
GODOT_BUILD = os.path.join(ROOT, "scripts", "build-godot-gdextension.sh")
ROOT_CARGO = os.path.join(ROOT, "Cargo.toml")
ROOT_LOCK = os.path.join(ROOT, "Cargo.lock")
UNITY_PACKAGE = os.path.join(
    ROOT, "bindings", "unity", "com.vokra.unity", "package.json"
)
UNITY_CHANGELOG = os.path.join(
    ROOT, "bindings", "unity", "com.vokra.unity", "CHANGELOG.md"
)
GODOT_CARGO = os.path.join(ROOT, "integrations", "vokra-godot", "Cargo.toml")
CHANGELOG = os.path.join(ROOT, "CHANGELOG.md")
EXTRACT_CHANGELOG = os.path.join(ROOT, "scripts", "release", "extract-changelog.py")
LOCKFILES = (
    ROOT_LOCK,
    os.path.join(ROOT, "fuzz", "Cargo.lock"),
    os.path.join(ROOT, "integrations", "vokra-android", "Cargo.lock"),
    os.path.join(ROOT, "integrations", "vokra-godot", "Cargo.lock"),
    os.path.join(ROOT, "integrations", "vokra-misaki-g2p", "Cargo.lock"),
    os.path.join(ROOT, "integrations", "vokra-piper-g2p", "Cargo.lock"),
    os.path.join(ROOT, "integrations", "vokra-server", "Cargo.lock"),
)


def load_toml(path: str) -> dict:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


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

    root_cargo = load_toml(ROOT_CARGO)
    canonical = root_cargo["workspace"]["package"]["version"]
    path_deps = [
        (name, spec.get("version"))
        for name, spec in root_cargo["workspace"]["dependencies"].items()
        if isinstance(spec, dict) and "path" in spec
    ]
    checks.append(
        (
            bool(path_deps) and all(version == canonical for _, version in path_deps),
            "workspace path dependencies use canonical version",
        )
    )

    root_lock = load_toml(ROOT_LOCK)
    workspace_names = {package["name"] for package in root_lock["package"]}
    lock_mismatches: list[str] = []
    for lockfile in LOCKFILES:
        for package in load_toml(lockfile)["package"]:
            if package["name"] in workspace_names and package["version"] != canonical:
                lock_mismatches.append(
                    f"{os.path.relpath(lockfile, ROOT)}:{package['name']}={package['version']}"
                )
    checks.append(
        (
            not lock_mismatches,
            "tracked lockfiles use canonical first-party version",
        )
    )

    with open(UNITY_PACKAGE, encoding="utf-8") as handle:
        unity_package = json.load(handle)
    checks.append(
        (
            unity_package["version"] == canonical,
            "Unity source manifest uses canonical version",
        )
    )
    checks.append(
        (
            load_toml(GODOT_CARGO)["package"]["version"] == canonical,
            "Godot source manifest uses canonical version",
        )
    )

    changelog = subprocess.run(
        [
            sys.executable,
            EXTRACT_CHANGELOG,
            "--version",
            canonical,
            "--changelog",
            CHANGELOG,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    checks.append(
        (
            changelog.returncode == 0 and bool(changelog.stdout.strip()),
            "root CHANGELOG has non-empty canonical release notes",
        )
    )
    with open(UNITY_CHANGELOG, encoding="utf-8") as handle:
        unity_changelog = handle.read()
    checks.append(
        (
            f"## [{canonical}]" in unity_changelog
            and f"releases/tag/v{canonical}" in unity_changelog,
            "Unity CHANGELOG has canonical release section and link",
        )
    )

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
