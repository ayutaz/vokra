#!/usr/bin/env python3
"""Resolve the one canonical version used by every release artifact.

The Rust workspace version is the release source of truth. A tag-triggered
release must use exactly ``v<workspace-version>``. A workflow-dispatch dry-run
may supply a candidate version explicitly so the same packaging paths can be
exercised before the source version is bumped; otherwise it uses the current
workspace version.
"""

from __future__ import annotations

import argparse
import os
import re
import sys


SEMVER_RE = re.compile(
    r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)


def fail(message: str) -> "None":
    print(f"version-contract: FAIL {message}", file=sys.stderr)
    raise SystemExit(1)


def workspace_version(cargo_toml: str) -> str:
    in_workspace_package = False
    with open(cargo_toml, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("[") and line.endswith("]"):
                in_workspace_package = line == "[workspace.package]"
                continue
            if in_workspace_package:
                match = re.fullmatch(r'version\s*=\s*"([^"]+)"', line)
                if match:
                    return match.group(1)
    fail(f"could not read [workspace.package] version from {cargo_toml}")


def resolve_version(github_ref: str, dispatch_version: str, canonical: str) -> str:
    if not SEMVER_RE.fullmatch(canonical):
        fail(f"workspace version '{canonical}' is not SemVer")

    if github_ref.startswith("refs/tags/"):
        tag = github_ref.removeprefix("refs/tags/")
        if not tag.startswith("v") or not SEMVER_RE.fullmatch(tag[1:]):
            fail(f"tag '{tag}' is not v-prefixed SemVer")
        version = tag[1:]
        if version != canonical:
            fail(
                f"tag version '{version}' does not match workspace version "
                f"'{canonical}'; bump all release metadata before tagging"
            )
        return version

    version = dispatch_version or canonical
    if version.startswith("v") or not SEMVER_RE.fullmatch(version):
        fail(
            f"dry-run version '{version}' must be unprefixed SemVer "
            "(for example 1.0.0-rc.1)"
        )
    return version


def main() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--github-ref", required=True)
    parser.add_argument("--dispatch-version", default="")
    parser.add_argument(
        "--cargo-toml", default=os.path.join(root, "Cargo.toml")
    )
    parser.add_argument("--github-output")
    args = parser.parse_args()

    canonical = workspace_version(args.cargo_toml)
    version = resolve_version(args.github_ref, args.dispatch_version, canonical)
    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as handle:
            handle.write(f"version={version}\n")
    print(version)


if __name__ == "__main__":
    main()
