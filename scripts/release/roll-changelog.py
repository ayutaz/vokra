#!/usr/bin/env python3
"""roll-changelog.py — roll [Unreleased] into a released section (X-07-T17).

At release time the `## [Unreleased]` section is renamed to `## [X.Y.Z] — <date>`,
a fresh empty `## [Unreleased]` is inserted above it, and the compare/tag link
footer is updated to the repo's convention (CHANGELOG.md:553-555):

    [Unreleased]: <base>/compare/vX.Y.Z...HEAD      (was: .../compare/<prev>...HEAD)
    [X.Y.Z]:      <base>/releases/tag/vX.Y.Z         (newly inserted)

This is an OWNER-run helper (it edits a tracked file; it is NOT auto-run in CI —
red-line 6: no auto-commits). `--check` validates the current CHANGELOG's
integrity without modifying it.

IDEMPOTENT (X-07-T19): rolling to a version whose section already exists is a
no-op (exit 0, no change), so a re-run never duplicates a section or corrupts
the footer.

The freshly-inserted [Unreleased] is EMPTY on purpose — `extract-changelog.py`
then honestly reports "nothing to release" (exit 2) until real entries land, so
a release can never ship fabricated/placeholder notes (FR-EX-08).

Zero-dep (NFR-DS-02): python3 stdlib only.

Usage:
    python3 scripts/release/roll-changelog.py --version 1.0.0-rc.1 --date 2026-08-01
    python3 scripts/release/roll-changelog.py --check
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CHANGELOG = os.path.join(ROOT, "CHANGELOG.md")

HEADING_RE = re.compile(r"^##\s+\[(?P<ver>[^\]]+)\]\s*(?:[—-]\s*(?P<date>.+))?\s*$")
FOOTER_RE = re.compile(r"^\[(?P<ver>[^\]]+)\]:\s+(?P<url>\S+)\s*$")
# MULTILINE: this is `.search()`ed / `.sub()`ed against the whole document, so
# ^ and $ must anchor to line boundaries, not the start/end of the full text.
UNRELEASED_COMPARE_RE = re.compile(
    r"^\[Unreleased\]:\s+(?P<base>\S+?)/compare/(?P<prev>\S+?)\.\.\.HEAD\s*$",
    re.MULTILINE,
)


def fail(msg: str, code: int = 1) -> "None":
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def headings(text: str) -> list[str]:
    out = []
    for ln in text.splitlines():
        m = HEADING_RE.match(ln)
        if m:
            out.append(m.group("ver").strip())
    return out


def footer_versions(text: str) -> dict[str, str]:
    out = {}
    for ln in text.splitlines():
        m = FOOTER_RE.match(ln)
        if m:
            out[m.group("ver").strip()] = m.group("url")
    return out


def check(text: str) -> int:
    problems: list[str] = []
    heads = headings(text)
    foot = footer_versions(text)

    if "Unreleased" not in heads:
        problems.append("no `## [Unreleased]` section heading")
    if "Unreleased" not in foot:
        problems.append("no `[Unreleased]: ...` footer link")
    elif not UNRELEASED_COMPARE_RE.search(text):
        problems.append("`[Unreleased]:` footer is not the `.../compare/<prev>...HEAD` form")

    # Every section heading must have a matching footer link (and vice versa is
    # nice but headings-first is the load-bearing direction for release notes).
    for h in heads:
        if h not in foot:
            problems.append(f"section [{h}] has no matching footer link")

    # Duplicate headings would corrupt extraction.
    seen = set()
    for h in heads:
        if h in seen:
            problems.append(f"duplicate section heading [{h}]")
        seen.add(h)

    if problems:
        print("roll-changelog --check: FAIL", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    print(f"roll-changelog --check: OK ({len(heads)} section(s), footer consistent)")
    return 0


def roll(text: str, version: str, date: str) -> str | None:
    """Return the rolled text, or None if it's already rolled (idempotent)."""
    heads = headings(text)
    if version in heads:
        # Idempotent: the section already exists — nothing to do.
        return None
    if "Unreleased" not in heads:
        fail("cannot roll: no `## [Unreleased]` section")

    lines = text.splitlines(keepends=True)
    out: list[str] = []
    rolled_heading = False
    for ln in lines:
        m = HEADING_RE.match(ln.rstrip("\n"))
        if m and m.group("ver").strip() == "Unreleased" and not rolled_heading:
            # Insert a fresh empty Unreleased, then the renamed release heading.
            nl = "\n" if ln.endswith("\n") else ""
            out.append(f"## [Unreleased]{nl}")
            out.append("\n")
            out.append(f"## [{version}] — {date}{nl}")
            rolled_heading = True
            continue
        out.append(ln)
    result = "".join(out)

    # Footer: repoint [Unreleased] to compare/vX.Y.Z...HEAD and insert the
    # [X.Y.Z] release-tag link right after it.
    m = UNRELEASED_COMPARE_RE.search(result)
    if not m:
        fail("cannot roll: `[Unreleased]:` compare footer not found")
    base = m.group("base")
    new_unreleased = f"[Unreleased]: {base}/compare/v{version}...HEAD"
    new_release = f"[{version}]: {base}/releases/tag/v{version}"
    result = UNRELEASED_COMPARE_RE.sub(
        lambda _m: new_unreleased + "\n" + new_release, result, count=1
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", default=None, help="release version, e.g. 1.0.0-rc.1")
    ap.add_argument("--date", default=None, help="release date YYYY-MM-DD (default: today UTC)")
    ap.add_argument("--changelog", default=CHANGELOG)
    ap.add_argument("--output", default=None, help="write here (default: in place)")
    ap.add_argument("--check", action="store_true", help="validate integrity only")
    args = ap.parse_args()

    if not os.path.isfile(args.changelog):
        fail(f"changelog not found: {args.changelog}")
    text = open(args.changelog, encoding="utf-8").read()

    if args.check:
        sys.exit(check(text))

    if not args.version:
        fail("--version is required to roll (or pass --check)")
    if not re.match(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.\-]+)?$", args.version):
        fail(f"not a semver version: {args.version!r}")
    date = args.date or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

    rolled = roll(text, args.version, date)
    if rolled is None:
        print(f"roll-changelog: [{args.version}] already present — no-op (idempotent)")
        sys.exit(0)

    dest = args.output or args.changelog
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write(rolled)
    print(f"roll-changelog: rolled [Unreleased] -> [{args.version}] — {date} "
          f"(wrote {dest})")


if __name__ == "__main__":
    main()
