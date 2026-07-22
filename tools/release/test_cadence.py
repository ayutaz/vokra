#!/usr/bin/env python3
"""test_cadence.py — oracle for the 4-week cadence mechanism (X-07-T22).

Covers:
  * the 28-day threshold with a CLOSED (">=") boundary — exactly 28 days is
    DUE, 27 days is not;
  * NO auto-tag — neither the workflow nor cadence.py ever pushes a tag /
    creates a release (red-line 6 mechanized);
  * the JUDGMENT CONTRACT M5-12-T06 consumes: schema, data_source, boundary,
    threshold_days, and the cadence_established definition are pinned HERE so
    M5-12 can rely on them without re-implementing the threshold (drift guard);
  * the stale Package.swift sentinel comment in release.yml is corrected AND the
    release.yml regex actually matches the real Package.swift binaryTarget form.

Zero-dep (NFR-DS-02): python3 stdlib only.
Usage: python3 tools/release/test_cadence.py
"""

from __future__ import annotations

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CADENCE_YML = os.path.join(ROOT, ".github", "workflows", "release-cadence.yml")
CADENCE_PY = os.path.join(HERE, "cadence.py")
RELEASE_YML = os.path.join(ROOT, ".github", "workflows", "release.yml")
PACKAGE_SWIFT = os.path.join(ROOT, "Package.swift")

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


def load_cadence():
    spec = importlib.util.spec_from_file_location("cadence", CADENCE_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DAY = 86400
NOW = 1_700_000_000


def test_threshold(cad) -> None:
    # exactly 28 days -> due (closed boundary).
    r = cad.compute([("v1.0.0", NOW - 28 * DAY)], NOW)
    if r["due"] is True and r["days_since_last"] == 28.0:
        ok("T22 exactly 28 days -> DUE (closed '>=' boundary)")
    else:
        bad(f"T22 28-day boundary wrong: due={r['due']} days={r['days_since_last']}")

    # 27 days -> not due.
    r = cad.compute([("v1.0.0", NOW - 27 * DAY)], NOW)
    if r["due"] is False:
        ok("T22 27 days -> not due")
    else:
        bad("T22 27 days incorrectly reported due")

    # 29 days -> due.
    r = cad.compute([("v1.0.0", NOW - 29 * DAY)], NOW)
    if r["due"] is True:
        ok("T22 29 days -> due")
    else:
        bad("T22 29 days incorrectly reported not-due")

    # 0 releases -> a first release is due; cadence not established.
    r = cad.compute([], NOW)
    if r["due"] is True and r["release_count"] == 0 and r["cadence_established"] is False:
        ok("T22 0 releases -> first release due, cadence NOT established (honest)")
    else:
        bad(f"T22 0-release case wrong: {r}")

    # 1 release -> not established (needs an interval).
    r = cad.compute([("v1.0.0", NOW - 10 * DAY)], NOW)
    if r["cadence_established"] is False:
        ok("T22 1 release -> cadence not established (needs >= 2 for an interval)")
    else:
        bad("T22 1-release incorrectly reported established")

    # 3 releases all within 28d -> established; one interval over -> not.
    good = [("v1.0.0", NOW - 84 * DAY), ("v1.0.1", NOW - 56 * DAY), ("v1.0.2", NOW - 28 * DAY)]
    r = cad.compute(good, NOW)
    if r["cadence_established"] is True and r["intervals_days"] == [28.0, 28.0]:
        ok("T22 3 releases all <=28d apart -> cadence established")
    else:
        bad(f"T22 established-case wrong: {r['intervals_days']} est={r['cadence_established']}")

    over = [("v1.0.0", NOW - 100 * DAY), ("v1.0.1", NOW - 56 * DAY), ("v1.0.2", NOW - 28 * DAY)]
    r = cad.compute(over, NOW)
    if r["cadence_established"] is False:
        ok("T22 an interval > 28d -> cadence NOT established")
    else:
        bad("T22 over-threshold interval incorrectly reported established")


def test_contract(cad) -> None:
    # Pin the M5-12-T06 contract: fields + values that M5-12 relies on.
    r = cad.compute([("v1.0.0", NOW - 10 * DAY)], NOW)
    expected_keys = {
        "schema", "data_source", "threshold_days", "boundary", "now_epoch",
        "release_count", "last_tag", "last_tag_epoch", "days_since_last",
        "due", "intervals_days", "cadence_established", "note",
    }
    missing = expected_keys - set(r)
    if not missing:
        ok("T22 contract: all judgment fields present (M5-12-T06 consumer surface)")
    else:
        bad(f"T22 contract missing fields: {missing}")

    if r["schema"] == "vokra.release-cadence.v1":
        ok("T22 contract: schema is vokra.release-cadence.v1")
    else:
        bad(f"T22 contract: unexpected schema {r['schema']!r}")

    if r["threshold_days"] == 28 and r["boundary"] == ">=" and \
            r["data_source"] == "git-tags:refs/tags/v*":
        ok("T22 contract: threshold=28, boundary='>=', data_source=git-tags (pinned)")
    else:
        bad(f"T22 contract: threshold/boundary/data_source drift: "
            f"{r['threshold_days']} / {r['boundary']!r} / {r['data_source']!r}")


def test_no_auto_tag(cad) -> None:
    # cadence.py must never tag / create a release. Use CODE-FORM patterns so
    # prose in the docstring ("tag push is owner-only") is not a false hit:
    #   '"tag"'  = a `["git", "tag", <name>]` subprocess element (creation);
    #   "git push" / "gh release create|edit" = the other mutation calls.
    # The only git call in cadence.py is the read-only `git for-each-ref`.
    src = open(CADENCE_PY, encoding="utf-8").read()
    forbidden = ['"tag"', "git push", "gh release create", "gh release edit"]
    hit = [f for f in forbidden if f in src]
    if not hit:
        ok("T22 cadence.py contains no tag/push/release-create call")
    else:
        bad(f"T22 cadence.py must not tag/push/create: found {hit}")

    yml = open(CADENCE_YML, encoding="utf-8").read()
    # Only inspect non-comment lines (a comment explains the deliberate absence).
    yml_code = "\n".join(ln for ln in yml.splitlines() if not ln.lstrip().startswith("#"))
    hit = [f for f in ["git tag ", "git push", "gh release create"] if f in yml_code]
    if not hit:
        ok("T22 release-cadence.yml never pushes a tag / creates a release (advisory)")
    else:
        bad(f"T22 release-cadence.yml must not tag/create: found {hit}")


def test_sentinel_fix() -> None:
    # The stale `// M2-02-T09: release binaryTarget` sentinel must no longer be
    # presented as an existing marker, and the release.yml regex must match the
    # ACTUAL Package.swift binaryTarget form.
    rel = open(RELEASE_YML, encoding="utf-8").read()
    if "There is NO sentinel comment in Package.swift" in rel and "X-07-T22" in rel:
        ok("T22 stale Package.swift sentinel comment corrected in release.yml")
    else:
        bad("T22 stale sentinel comment not corrected")

    if not os.path.isfile(PACKAGE_SWIFT):
        bad("T22 Package.swift not found")
        return
    import re
    swift = open(PACKAGE_SWIFT, encoding="utf-8").read()
    pat = re.compile(r'\.binaryTarget\(\s*name:\s*"Vokra"\s*,\s*path:\s*"[^"]+"\s*\)')
    if pat.search(swift):
        ok("T22 release.yml regex matches the real Package.swift .binaryTarget(path:) form")
    else:
        bad("T22 Package.swift .binaryTarget(path:) form no longer matches the release.yml regex")


def main() -> None:
    print("[X-07-T22] cadence mechanism + judgment contract")
    cad = load_cadence()
    test_threshold(cad)
    test_contract(cad)
    test_no_auto_tag(cad)
    test_sentinel_fix()
    print()
    print(f"cadence oracle: {_pass} passed, {_fail} failed")
    sys.exit(1 if _fail else 0)


if __name__ == "__main__":
    main()
