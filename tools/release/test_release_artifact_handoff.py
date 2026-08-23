#!/usr/bin/env python3
"""Oracle for fail-loud, same-run release artifact handoff."""

from __future__ import annotations

import os
import re


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RELEASE = os.path.join(ROOT, ".github", "workflows", "release.yml")
DESKTOP = os.path.join(ROOT, ".github", "workflows", "release-desktop-preflight.yml")
GODOT = os.path.join(ROOT, ".github", "workflows", "godot-crossbuild.yml")


def job_block(text: str, job_id: str) -> str:
    match = re.search(
        rf"^  {re.escape(job_id)}:\s*$([\s\S]*?)(?=^  [A-Za-z_][\w-]*:\s*$|\Z)",
        text,
        flags=re.MULTILINE,
    )
    if not match:
        raise AssertionError(f"missing job {job_id}")
    return match.group(0)


def main() -> None:
    release = open(RELEASE, encoding="utf-8").read()
    desktop = open(DESKTOP, encoding="utf-8").read()
    godot = open(GODOT, encoding="utf-8").read()
    checks: list[tuple[bool, str]] = []

    desktop_call = job_block(release, "release-desktop-preflight")
    godot_call = job_block(release, "release-godot-preflight")
    unity = job_block(release, "unity-package-release")
    godot_release = job_block(release, "godot-package-release")
    desktop_release = job_block(release, "desktop-release")

    checks.extend(
        [
            ("release-desktop-preflight.yml" in desktop_call, "release calls desktop preflight"),
            ("godot-crossbuild.yml" in godot_call, "release calls Godot preflight"),
            ("release-desktop-preflight" in unity, "Unity waits for desktop preflight"),
            (unity.count("actions/download-artifact@") >= 3, "Unity downloads same-run iOS/macOS/Windows artifacts"),
            ("gh run list" not in unity and "gh run download" not in unity, "Unity has no cross-run lookup"),
            ("release-godot-preflight" in godot_release, "Godot release waits for crossbuild"),
            ("pattern: vokra-godot-*" in godot_release, "Godot downloads same-run matrix artifacts"),
            (all(target in godot_release for target in (
                "x86_64-apple-darwin:libvokra_godot.dylib",
                "aarch64-apple-darwin:libvokra_godot.dylib",
                "x86_64-unknown-linux-gnu:libvokra_godot.so",
                "x86_64-pc-windows-msvc:vokra_godot.dll",
                "aarch64-linux-android:libvokra_godot.so",
            )), "Godot requires all five exact target files"),
            ("continue-on-error: true" not in godot_release, "Godot handoff is fail-loud"),
            ("gh run list" not in godot_release and "gh run download" not in godot_release, "Godot has no cross-run lookup"),
            ("gh run list" not in desktop_release and "gh run download" not in desktop_release, "desktop release has no cross-run lookup"),
            ("workflow_call:" in desktop, "desktop preflight is reusable"),
            ("vokra-capi-macos" in desktop and "vokra-capi-windows" in desktop, "desktop preflight emits both artifacts"),
            ("if-no-files-found: error" in desktop, "desktop upload is fail-loud"),
            ("workflow_call:" in godot, "Godot crossbuild is reusable"),
        ]
    )

    failures = [label for passed, label in checks if not passed]
    for passed, label in checks:
        print(f"  {'ok' if passed else 'FAIL'}: {label}")
    print()
    print(
        f"release-artifact-handoff oracle: "
        f"{len(checks) - len(failures)} passed, {len(failures)} failed"
    )
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
