#!/usr/bin/env python3
"""Static contract checks for the GitHub-only v0.4.0-rc.1 release path."""

from __future__ import annotations

import json
import os
import re


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKFLOW = os.path.join(ROOT, ".github", "workflows", "release-rc1-gh-only.yml")
PREFLIGHT = os.path.join(ROOT, ".github", "workflows", "release-desktop-preflight.yml")
STABLE_RELEASE = os.path.join(ROOT, ".github", "workflows", "release.yml")
PINS = os.path.join(ROOT, "tools", "parity", "release_candidate_rc1_pins.json")


def _job(text: str, job_id: str) -> str:
    match = re.search(
        rf"^  {re.escape(job_id)}:\s*$([\s\S]*?)(?=^  [A-Za-z_][\w-]*:\s*$|\Z)",
        text,
        flags=re.MULTILINE,
    )
    if not match:
        raise AssertionError(f"missing job {job_id}")
    return match.group(0)


def _contains_in_order(text: str, first: str, second: str) -> bool:
    """Require both workflow fragments and their execution order."""
    return (
        first in text
        and second in text
        and text.index(first) < text.index(second)
    )


def _self_test_contains_in_order() -> None:
    """Keep missing/reversed fragment checks fail-closed."""
    assert _contains_in_order("chmod help", "chmod", "help")
    assert not _contains_in_order("help", "chmod", "help")
    assert not _contains_in_order("help chmod", "chmod", "help")


def main() -> None:
    _self_test_contains_in_order()
    print("  ok: chmod order helper rejects missing and reversed fragments")
    workflow = open(WORKFLOW, encoding="utf-8").read()
    preflight = open(PREFLIGHT, encoding="utf-8").read()
    stable_release = open(STABLE_RELEASE, encoding="utf-8").read()
    pins = json.load(open(PINS, encoding="utf-8"))
    checks: list[tuple[bool, str]] = []

    publish = _job(workflow, "publish-github-prerelease")
    assemble = _job(workflow, "assemble-desktop")
    smoke = _job(workflow, "candidate-cpu-smoke")
    linux = _job(workflow, "clean-install-linux")
    macos = _job(workflow, "clean-install-macos")
    windows = _job(workflow, "clean-install-windows")
    checks.extend(
        [
            ("workflow_dispatch:" in workflow and "  push:" not in workflow, "RC workflow is manual-only"),
            ('RC_TAG: v0.4.0-rc.1' in workflow and 'RC_VERSION: 0.4.0-rc.1' in workflow, "RC identity is fixed"),
            ('source_sha:' in workflow and 'required: true' in workflow, "exact source SHA is required"),
            ('git/ref/heads/main' in workflow and 'MAIN_SHA=' in workflow and 'SOURCE_SHA,,}' in workflow, "RC source must be current merged main"),
            ('GITHUB_REF" == refs/heads/main' in workflow and 'GITHUB_SHA,,}' in workflow, "RC dispatch is bound to the main workflow commit"),
            ('git/ref/tags/${RC_TAG}' in workflow and 'tag already exists' in workflow, "existing RC tag is refused before build"),
            ('version_contract.py' in _job(workflow, "validate-input") and 'refs/tags/${RC_TAG}' in workflow, "candidate version contract is validated before build"),
            ('source_ref: ${{ needs.validate-input.outputs.source_sha }}' in workflow, "preflight receives exact source SHA"),
            ('ref: ${{ needs.validate-input.outputs.source_sha }}' in workflow, "consumer jobs checkout exact source SHA"),
            ('release-desktop-preflight.yml' in workflow, "RC reuses desktop preflight"),
            ('release_candidate_rc1_pins.json' in workflow, "RC uses fixed reviewed model pins"),
            ('--pins-json "$PINS_PATH" --validate-pins-only' in smoke, "RC validates reviewed pins before smoke"),
            ('release_candidate_cpu_smoke.py' in smoke and 'EXPECTED_HEAD' in smoke, "RC smoke binds to exact source"),
            ('dist/desktop/vokra-cli-${RC_VERSION}-x86_64-linux' in smoke, "CPU smoke uses the assembled distribution filename"),
            ('desktop_release.py' in assemble and 'manifest --version "$RC_VERSION"' in assemble, "RC verifies complete desktop manifest"),
            ('vokra-desktop-capi.spdx.json' in assemble and 'vokra-desktop-cli.spdx.json' in assemble, "RC includes C ABI and CLI SBOMs"),
            ('native-smoke' in linux and 'vokra-cli-${RC_VERSION}-x86_64-linux-musl' in linux, "Linux clean-install QA covers both CLI forms and C ABI"),
            (_contains_in_order(linux, 'chmod +x "dist/desktop/vokra-cli-${RC_VERSION}-x86_64-linux"', '"dist/desktop/vokra-cli-${RC_VERSION}-x86_64-linux" --help'), "Linux clean-install restores CLI mode before help"),
            ('runs-on: macos-latest' in macos and 'uname -m' in macos and 'native-smoke' in macos and '--help' in macos, "macOS clean-install QA covers CLI and C ABI"),
            (_contains_in_order(macos, 'chmod +x "$CLI"', '"$CLI" --help'), "macOS clean-install restores CLI mode before help"),
            ('runs-on: windows-latest' in windows and 'Expand-Archive' in windows and 'native-smoke' in windows, "Windows clean-install QA covers ZIP CLI and DLL"),
            (all(job_id in publish for job_id in ("clean-install-linux", "clean-install-macos", "clean-install-windows")), "publication waits for every clean-install QA"),
            ('--draft --prerelease --latest=false' in publish, "release is created as draft non-latest prerelease"),
            ('gh release edit "$RC_TAG" --draft=false --prerelease --latest=false' in publish, "release is published only after all gates"),
            ('gh release view "$RC_TAG"' in publish and 'refusing to reuse' in publish, "existing release/tag is refused"),
            ('--target "$SOURCE_SHA"' in publish, "release tag targets exact source SHA"),
            ('git/refs"' in publish and 'refs/tags/${RC_TAG}' in publish and 'sha=${SOURCE_SHA}' in publish, "missing tag is created only at the reviewed source SHA"),
            ('targetCommitish' in publish and 'grep -Fx "$SOURCE_SHA"' in publish, "draft release target is checked against source SHA"),
            ('extract-changelog.py' in publish and '--notes-file release-notes.md' in publish, "RC notes come from the candidate changelog"),
            ('contents: write' in publish and 'contents: write' not in workflow.split('  publish-github-prerelease:', 1)[0], "write permission is publish-job-only"),
            ('cargo publish' not in workflow and 'npm publish' not in workflow and 'twine' not in workflow, "RC has no external registry publish"),
            ('continue-on-error' not in workflow and 'continue-on-error' not in preflight, "RC gates fail closed"),
            ('source_ref:' in preflight and 'inputs.source_ref || github.sha' in preflight, "reusable preflight preserves default callers"),
            ('sparse-checkout-cone-mode: false' in preflight and 'Verify protected trees are excluded' in preflight, "desktop preflight excludes protected trees"),
            ('prerelease tags are handled only by the explicit GitHub-only RC workflow' in stable_release and 'refs/tags/v*-*' in stable_release, "stable release workflow refuses prerelease tags"),
        ]
    )

    expected = {
        "whisper-base": {
            "repo": "vokra/whisper-base",
            "revision": "dddd93ff6301a82aa3750d398cb2e0a599cd9e83",
            "filename": "whisper-base.gguf",
            "sha256": "7e774425585d6e9ba58ac5337b406522b1408ab6ff6765d2938b092aef4c8e27",
            "size_bytes": 290859360,
        },
        "silero-vad-v5": {
            "repo": "vokra/silero-vad-v5",
            "revision": "3670dfed054980307eeb1711c6329c29b80bdf14",
            "filename": "silero-vad-v5.gguf",
            "sha256": "23b19eadba72fb9726fa73a0bfd095afcba2f8c5bcb4f03a1b5e57a67eb13df6",
            "size_bytes": 2183744,
        },
    }
    checks.append((pins == expected, "fixed RC pin manifest matches reviewed records"))

    failures = [label for passed, label in checks if not passed]
    for passed, label in checks:
        print(f"  {'ok' if passed else 'FAIL'}: {label}")
    print()
    print(f"RC release workflow oracle: {len(checks) - len(failures)} passed, {len(failures)} failed")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
