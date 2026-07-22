#!/usr/bin/env python3
"""cadence.py — 4-week release-train cadence measurement (X-07-T20).

Computes whether the next release is DUE (>= 28 days since the last release tag)
and emits a machine-readable cadence judgment. This is ADVISORY: it NEVER tags
anything (tag push is owner-only, red-line 6). The judgment contract is the
Source-of-Truth consumed by M5-12-T06 (GA DoD item 3 "release train stable
operation") — see docs/tickets/m5/M5-12-ga-dod-judgment.md:75, so the definition
lives HERE and is not re-implemented there (drift avoidance).

CADENCE JUDGMENT CONTRACT (schema "vokra.release-cadence.v1"), fixed by
ADR docs/adr/X-07-release-train.md §cadence:

  (i)   data_source     = "git-tags:refs/tags/v*" — release tags are the SoT
                          (release.yml triggers on `v*` tag pushes). Only
                          semver-valid tags (vX.Y.Z[-pre][+build]) count.
  (ii)  boundary        = ">=" — due when days_since_last >= threshold_days
                          (28-day boundary is CLOSED: exactly 28 days == due).
  (iii) cadence_established: >= 2 releases AND every consecutive interval
                          <= threshold_days. This is the BACKWARD-looking
                          "has been operating on a 4-week cadence" evidence
                          M5-12 needs; `due` alone is the forward-looking
                          "is the next one overdue".

Zero-dep (NFR-DS-02): python3 stdlib only, driving `git for-each-ref`. For
deterministic testing, tag data may be injected via --tags-json (no git call).

Usage:
    python3 tools/release/cadence.py                    # JSON from git tags
    python3 tools/release/cadence.py --emit-summary     # + step-summary material
    python3 tools/release/cadence.py --now 1700000000 --tags-json fixtures.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

SCHEMA = "vokra.release-cadence.v1"
DATA_SOURCE = "git-tags:refs/tags/v*"
THRESHOLD_DAYS = 28
BOUNDARY = ">="  # closed boundary: days_since >= 28 == due
DAY = 86400

# v-prefixed semver: vX.Y.Z with optional -prerelease and +build.
SEMVER_TAG_RE = re.compile(r"^v[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")


def read_git_tags() -> list[tuple[str, int]]:
    """Return [(tag, epoch), ...] for semver v-tags, sorted by creation asc."""
    proc = subprocess.run(
        ["git", "for-each-ref", "--sort=creatordate",
         "--format=%(refname:short) %(creatordate:unix)", "refs/tags/v*"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"error: git for-each-ref failed: {proc.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    out = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        tag, epoch_s = parts
        if not SEMVER_TAG_RE.match(tag):
            continue  # non-semver v-tag: ignored (validate-tag rejects at release)
        try:
            out.append((tag, int(epoch_s)))
        except ValueError:
            continue
    return out


def compute(tags: list[tuple[str, int]], now: int) -> dict:
    """Pure cadence judgment from (tag, epoch) pairs sorted ascending."""
    tags = sorted(tags, key=lambda t: t[1])
    count = len(tags)
    intervals = [
        round((tags[i + 1][1] - tags[i][1]) / DAY, 2) for i in range(count - 1)
    ]
    if count == 0:
        last_tag = None
        last_epoch = None
        days_since = None
        due = True  # nothing released yet — a first release is "due"
    else:
        last_tag, last_epoch = tags[-1]
        days_since = round((now - last_epoch) / DAY, 2)
        due = days_since >= THRESHOLD_DAYS  # boundary ">="

    cadence_established = count >= 2 and all(iv <= THRESHOLD_DAYS for iv in intervals)

    return {
        "schema": SCHEMA,
        "data_source": DATA_SOURCE,
        "threshold_days": THRESHOLD_DAYS,
        "boundary": BOUNDARY,
        "now_epoch": now,
        "release_count": count,
        "last_tag": last_tag,
        "last_tag_epoch": last_epoch,
        "days_since_last": days_since,
        "due": due,
        "intervals_days": intervals,
        "cadence_established": cadence_established,
        # Honest note for the 0/1-release reality (git tag 0 件 at intake).
        "note": (
            "no releases yet — cadence not established; a first release is due"
            if count == 0 else
            "single release — cadence not established until a second interval exists"
            if count == 1 else
            "cadence established" if cadence_established else
            "cadence NOT established — an interval exceeded the threshold"
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--now", type=int, default=None, help="override now (epoch, for tests)")
    ap.add_argument("--tags-json", default=None,
                    help="JSON [[tag, epoch], ...] to use instead of git (tests)")
    ap.add_argument("--emit-summary", action="store_true",
                    help="also append material to $GITHUB_STEP_SUMMARY (X-07-T21)")
    args = ap.parse_args()

    now = args.now if args.now is not None else int(time.time())
    if args.tags_json:
        raw = json.load(open(args.tags_json, encoding="utf-8"))
        tags = [(t[0], int(t[1])) for t in raw]
    else:
        tags = read_git_tags()

    result = compute(tags, now)
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")

    if args.emit_summary:
        emit_summary(result)


def emit_summary(result: dict) -> None:
    """X-07-T21: write cadence material to the GitHub step summary (advisory)."""
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    lines = [
        "## Release cadence (advisory, X-07-T20/T21)",
        "",
        f"- data source: `{result['data_source']}`",
        f"- releases so far: **{result['release_count']}**",
        f"- last tag: `{result['last_tag']}`",
        f"- days since last: **{result['days_since_last']}** "
        f"(threshold {result['threshold_days']}, boundary `{result['boundary']}`)",
        f"- **DUE: {result['due']}**",
        f"- cadence established: {result['cadence_established']}",
        f"- intervals (days): {result['intervals_days']}",
        f"- note: {result['note']}",
        "",
        "> Advisory only — this workflow NEVER pushes a tag (owner-only, "
        "red-line 6). The judgment contract (schema `"
        + result["schema"] + "`) is consumed by M5-12-T06.",
    ]
    text = "\n".join(lines) + "\n"
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
