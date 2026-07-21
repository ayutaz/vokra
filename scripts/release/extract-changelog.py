#!/usr/bin/env python3
"""extract-changelog.py — pull one version's notes out of CHANGELOG.md (X-07-T15).

CHANGELOG.md follows Keep a Changelog: `## [X.Y.Z] — <date>` / `## [Unreleased]`
section headings, with `[X.Y.Z]: <url>` compare/tag link lines at the foot. This
tool extracts the body of ONE section so `release.yml` can attach it as the
GitHub Release notes (X-07-T16), replacing the empty auto-generated notes a bare
tag push produces.

FAILURE POLICY (FR-EX-08, mirroring scripts/sbom/generate_spdx.py:38-40):
  * a requested version whose `## [<version>]` heading is ABSENT -> exit 1;
  * a heading that exists but whose body is EMPTY -> exit 2 (do NOT emit
    fabricated / placeholder notes — an empty [Unreleased] means there is
    nothing to release, and the release step should fail rather than publish
    blank notes).
The tool never invents notes.

Zero-dep (NFR-DS-02): python3 stdlib only.

Usage:
    python3 scripts/release/extract-changelog.py --version Unreleased
    python3 scripts/release/extract-changelog.py --version 1.0.0-rc.1 \
        --changelog CHANGELOG.md --output release-notes.md
"""

from __future__ import annotations

import argparse
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# `## [Unreleased]` or `## [1.0.0-rc.1] — 2026-07-04` (em dash or hyphen, or
# nothing). The version token inside the brackets is what we match on.
HEADING_RE = re.compile(r"^##\s+\[(?P<ver>[^\]]+)\]\s*(?:[—-]\s*(?P<date>.+))?\s*$")
# Any `## ` section heading ends the current section.
ANY_HEADING_RE = re.compile(r"^##\s+")
# Link-footer line: `[Unreleased]: https://...`.
FOOTER_RE = re.compile(r"^\[[^\]]+\]:\s+\S+")


def fail(msg: str, code: int = 1) -> "None":
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def extract(text: str, version: str) -> tuple[str, str | None]:
    """Return (body, date-or-None) for the `## [<version>]` section.

    `version` matches the token inside the brackets, case-insensitively
    (so `unreleased` and `Unreleased` both work).
    """
    lines = text.splitlines()
    start = None
    date = None
    for i, ln in enumerate(lines):
        m = HEADING_RE.match(ln)
        if m and m.group("ver").strip().lower() == version.strip().lower():
            start = i
            date = (m.group("date") or "").strip() or None
            break
    if start is None:
        fail(f"CHANGELOG section not found: [{version}]", code=1)

    body: list[str] = []
    for ln in lines[start + 1:]:
        if ANY_HEADING_RE.match(ln):
            break
        if FOOTER_RE.match(ln):
            break
        body.append(ln)

    # Trim leading/trailing blank lines but keep the internal shape.
    while body and body[0].strip() == "":
        body.pop(0)
    while body and body[-1].strip() == "":
        body.pop()

    if not body:
        fail(
            f"CHANGELOG section [{version}] is present but EMPTY — nothing to "
            "release; refusing to emit fabricated notes (FR-EX-08)",
            code=2,
        )
    return "\n".join(body) + "\n", date


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", default="Unreleased",
                    help="section version token, e.g. Unreleased or 1.0.0-rc.1")
    ap.add_argument("--changelog", default=os.path.join(ROOT, "CHANGELOG.md"))
    ap.add_argument("--output", default=None,
                    help="write here (default: stdout)")
    args = ap.parse_args()

    if not os.path.isfile(args.changelog):
        fail(f"changelog not found: {args.changelog}")
    text = open(args.changelog, encoding="utf-8").read()

    body, _date = extract(text, args.version)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(body)
        print(f"wrote {args.output}: {len(body.splitlines())} line(s) "
              f"for [{args.version}]", file=sys.stderr)
    else:
        sys.stdout.write(body)


if __name__ == "__main__":
    main()
