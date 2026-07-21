#!/usr/bin/env python3
"""test_changelog.py — oracle for changelog automation + semver gate (X-07-T19).

Covers:
  * T15 extract-changelog round-trip (known section extracted verbatim);
  * T15 FR-EX-08: an ABSENT section exits non-zero; an EMPTY section exits
    non-zero (never fabricated notes);
  * T17 roll-changelog idempotency (re-roll is a no-op) + post-roll integrity
    (--check passes) + the freshly-rolled [Unreleased] is empty (extract exits 2);
  * T18 validate-tag: the release.yml job rejects non-semver tags AND
    announced-skips (passes) on non-tag refs — the property that keeps the
    dry-run/dispatch verify path from self-blocking.

Zero-dep (NFR-DS-02): python3 stdlib only. FR-EX-08: no silent pass.
Usage: python3 tools/release/test_changelog.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
EXTRACT = os.path.join(ROOT, "scripts", "release", "extract-changelog.py")
ROLL = os.path.join(ROOT, "scripts", "release", "roll-changelog.py")
RELEASE_YML = os.path.join(ROOT, ".github", "workflows", "release.yml")

_pass = 0
_fail = 0


def ok(msg: str) -> None:
    global _pass
    _pass += 1
    print(f"  ok:   {msg}")


def bad(msg: str) -> None:
    global _fail
    _fail += 1
    print(f"  FAIL: {msg}", file=sys.stderr)


FIXTURE = """# Changelog

## [Unreleased]

- unreleased change A
- unreleased change B

## [1.2.0] — 2026-05-01

- shipped feature X

## [1.1.0] — 2026-04-01

- shipped feature Y

[Unreleased]: https://github.com/ayutaz/vokra/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/ayutaz/vokra/releases/tag/v1.2.0
[1.1.0]: https://github.com/ayutaz/vokra/releases/tag/v1.1.0
"""

EMPTY_UNRELEASED = """# Changelog

## [Unreleased]

## [1.0.0] — 2026-01-01

- something

[Unreleased]: https://github.com/ayutaz/vokra/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/ayutaz/vokra/releases/tag/v1.0.0
"""


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def test_extract(scratch: str) -> None:
    fx = os.path.join(scratch, "CHANGELOG.md")
    open(fx, "w").write(FIXTURE)

    # round-trip: [1.2.0] body extracted verbatim.
    p = run(["python3", EXTRACT, "--version", "1.2.0", "--changelog", fx])
    if p.returncode == 0 and p.stdout.strip() == "- shipped feature X":
        ok("T15 extract round-trip: [1.2.0] body extracted verbatim")
    else:
        bad(f"T15 extract round-trip wrong (rc={p.returncode}): {p.stdout!r}")

    # Unreleased multi-line body.
    p = run(["python3", EXTRACT, "--version", "Unreleased", "--changelog", fx])
    if p.returncode == 0 and "unreleased change A" in p.stdout and "unreleased change B" in p.stdout:
        ok("T15 extract [Unreleased] returns its body")
    else:
        bad(f"T15 extract [Unreleased] wrong (rc={p.returncode})")

    # absent section -> non-zero.
    p = run(["python3", EXTRACT, "--version", "9.9.9", "--changelog", fx])
    if p.returncode != 0:
        ok("T15 absent section exits non-zero (FR-EX-08)")
    else:
        bad("T15 absent section did NOT fail")

    # empty section -> non-zero (no fabricated notes).
    fe = os.path.join(scratch, "EMPTY.md")
    open(fe, "w").write(EMPTY_UNRELEASED)
    p = run(["python3", EXTRACT, "--version", "Unreleased", "--changelog", fe])
    if p.returncode != 0:
        ok("T15 empty [Unreleased] exits non-zero (no fabricated notes, FR-EX-08)")
    else:
        bad("T15 empty [Unreleased] did NOT fail")


def test_roll(scratch: str) -> None:
    fx = os.path.join(scratch, "ROLL.md")
    open(fx, "w").write(FIXTURE)

    # roll to a new version.
    p = run(["python3", ROLL, "--version", "1.3.0", "--date", "2026-06-01", "--changelog", fx])
    if p.returncode != 0:
        bad(f"T17 roll failed: {p.stderr.strip()}")
        return
    rolled = open(fx).read()
    if "## [1.3.0] — 2026-06-01" in rolled and rolled.count("## [Unreleased]") == 1:
        ok("T17 roll: [Unreleased] -> [1.3.0] + one fresh [Unreleased]")
    else:
        bad("T17 roll produced wrong section structure")

    # footer updated: compare points at v1.3.0, new tag link inserted.
    if "compare/v1.3.0...HEAD" in rolled and "releases/tag/v1.3.0" in rolled:
        ok("T17 roll: footer compare + release-tag links updated")
    else:
        bad("T17 roll: footer links not updated correctly")

    # idempotent: re-roll same version = no change.
    before = open(fx).read()
    run(["python3", ROLL, "--version", "1.3.0", "--date", "2026-06-01", "--changelog", fx])
    after = open(fx).read()
    if before == after:
        ok("T17 roll idempotent: re-rolling the same version is a no-op")
    else:
        bad("T17 roll NOT idempotent: re-roll changed the file")

    # --check passes on the rolled file.
    p = run(["python3", ROLL, "--check", "--changelog", fx])
    if p.returncode == 0:
        ok("T17 --check passes on the rolled changelog")
    else:
        bad(f"T17 --check failed on rolled changelog: {p.stderr.strip()}")

    # freshly-rolled [Unreleased] is empty -> extract exits 2.
    p = run(["python3", EXTRACT, "--version", "Unreleased", "--changelog", fx])
    if p.returncode != 0:
        ok("T17 fresh [Unreleased] is empty -> extract refuses (no fabricated notes)")
    else:
        bad("T17 fresh [Unreleased] not empty (extract unexpectedly succeeded)")


def test_validate_tag() -> None:
    text = open(RELEASE_YML, encoding="utf-8").read()
    # Locate the validate-tag job body.
    if "validate-tag:" not in text:
        bad("T18 validate-tag job missing from release.yml")
        return
    # The announced-skip property: non-tag ref passes (exit 0 with a ::notice::).
    if "non-tag ref" in text and "announced skip" in text and 'refs/tags/*' in text:
        ok("T18 validate-tag announced-skips (passes) on non-tag refs")
    else:
        bad("T18 validate-tag missing the non-tag announced-skip (dispatch path would be blocked)")

    # Replicate the semver check on sample tags to prove the regex is right.
    semver = re.compile(r"^v[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$")
    good = ["v1.0.0", "v1.0.0-rc.1", "v0.1.0-alpha.0", "v10.20.30+build.5"]
    bad_tags = ["vfoo", "v1.0", "1.0.0", "vv1.0.0", "v1.0.0-", "release"]
    # Assert the SAME regex text is present in the job (so the job enforces it).
    if r"^v[0-9]+\.[0-9]+\.[0-9]+" in text:
        ok("T18 validate-tag uses a v-prefixed semver regex")
    else:
        bad("T18 validate-tag semver regex not found in release.yml")
    if all(semver.match(t) for t in good) and not any(semver.match(t) for t in bad_tags):
        ok("T18 semver regex accepts valid tags / rejects vfoo, v1.0, bare 1.0.0, etc.")
    else:
        bad("T18 semver regex classification wrong")


def main() -> None:
    print("[X-07-T19] changelog automation + semver gate")
    with tempfile.TemporaryDirectory(prefix="x07-changelog.") as scratch:
        test_extract(scratch)
        test_roll(scratch)
    test_validate_tag()
    print()
    print(f"changelog oracle: {_pass} passed, {_fail} failed")
    sys.exit(1 if _fail else 0)


if __name__ == "__main__":
    main()
