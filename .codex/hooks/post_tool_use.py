#!/usr/bin/env python3
"""Codex PostToolUse policy for apply_patch/Edit/Write.

The legacy PostToolUse payload exposed a file path directly. Codex's
apply_patch tool exposes the patch text as ``tool_input.command`` instead, so
this hook extracts changed paths before applying the same Rust-format and
zero-dependency checks.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def git_root(cwd: str) -> Path | None:
    result = subprocess.run(
        ["git", "-C", cwd, "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip())


def changed_paths(tool_input: object) -> list[str]:
    if not isinstance(tool_input, dict):
        return []

    paths: list[str] = []
    for key in ("file_path", "path"):
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            paths.append(value)

    command = tool_input.get("command")
    if not isinstance(command, str):
        return list(dict.fromkeys(paths))

    patterns = (
        r"^\*\*\* (?:Update|Add|Delete) File: (.+)$",
        r"^\+\+\+ b/(.+)$",
        r"^diff --git a/(\S+) b/(\S+)$",
    )
    for line in command.splitlines():
        for index, pattern in enumerate(patterns):
            match = re.match(pattern, line)
            if not match:
                continue
            if index == 2:
                paths.append(match.group(2))
            else:
                paths.append(match.group(1).strip())
            break

    return list(dict.fromkeys(paths))


def resolve(root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else root / path


def feedback_for_payload(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return []
    cwd = payload.get("cwd")
    root = git_root(cwd if isinstance(cwd, str) else os.getcwd())
    if root is None:
        return []

    paths = [resolve(root, path) for path in changed_paths(payload.get("tool_input"))]
    feedback: list[str] = []

    for path in paths:
        if path.suffix != ".rs" or not path.is_file():
            continue
        result = subprocess.run(
            ["rustfmt", "--edition", "2024", str(path)],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            feedback.append(
                f"rustfmt failed for {path.relative_to(root)}"
                + (f": {detail[-1]}" if detail else "")
            )

    if any(path.name in {"Cargo.toml", "Cargo.lock"} for path in paths):
        checker = root / "scripts/check-zero-deps.sh"
        if checker.is_file():
            result = subprocess.run(
                ["bash", str(checker)],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                detail = (result.stdout + result.stderr).strip()
                feedback.append(
                    "zero-dependency check failed after Cargo metadata changed"
                    + (f": {detail[-1200:]}" if detail else "")
                )
    return feedback


def self_test() -> int:
    root = Path(__file__).resolve().parents[2]
    patch = "*** Update File: docs/example.md\n+@@\n+"
    assert changed_paths({"command": patch}) == ["docs/example.md"]
    assert changed_paths({"file_path": "docs/example.md", "path": "docs/example.md"}) == [
        "docs/example.md"
    ]
    with tempfile.TemporaryDirectory(prefix="vokra-post-hook-"):
        # A documentation-only patch is a safe no-op and must not invoke
        # rustfmt or the Cargo dependency gate.
        assert feedback_for_payload({"cwd": str(root), "tool_input": {"command": patch}}) == []
    print("post_tool_use --self-test: OK")
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        return self_test()
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError):
        return 0
    feedback = feedback_for_payload(payload)

    if feedback:
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PostToolUse",
                        "additionalContext": "\n".join(feedback),
                    }
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
