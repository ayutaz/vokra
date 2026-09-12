#!/usr/bin/env -S uv run --script
"""Reject Rust test exclusions without an auditable reason."""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path


EMPTY_IGNORE = re.compile(
    r"^\s*#\[\s*ignore\s*(?:=\s*[\"']\s*[\"'])?\s*\]\s*(?://.*)?$"
)
CFG_ATTR_IGNORE = re.compile(r"^\s*#\[cfg_attr\([^]]*\bignore\s*\)\]\s*(?://.*)?$")


def violations(files: list[Path]) -> list[str]:
    found: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), 1):
            if EMPTY_IGNORE.search(line) or CFG_ATTR_IGNORE.search(line):
                found.append(f"{path}:{line_number}: #[ignore] requires a non-empty reason")
    return found


def tracked_test_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--", "*.rs"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "git ls-files failed")
    return [root / line for line in result.stdout.splitlines() if line]


def self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="vokra-ignore-") as directory:
        root = Path(directory)
        bad = root / "bad.rs"
        good = root / "good.rs"
        bad.write_text(
            "#[ignore]\nfn t() {}\n#[ignore = \"\"] // trailing\nfn u() {}\n"
            "#[cfg_attr(feature = \"real\", ignore)]\nfn v() {}\n",
            encoding="utf-8",
        )
        good.write_text(
            "#[ignore = \"requires an external fixture\"] // documented\nfn t() {}\n",
            encoding="utf-8",
        )
        assert len(violations([bad])) == 3, "bare, empty, and cfg_attr ignores must fail"
        assert not violations([good]), "reasoned ignore must pass"
    print("check-test-exclusion-hygiene --self-test: OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    root = Path(__file__).resolve().parents[1]
    errors = violations(tracked_test_files(root))
    if errors:
        print("\n".join(f"check-test-exclusion-hygiene: {item}" for item in errors))
        return 1
    print("Rust test exclusion hygiene: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
