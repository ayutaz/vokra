#!/usr/bin/env bash
# check-community-docs.sh — X-05-T07 contributor-funnel integrity check.
#
# WHAT IT GATES
#   The path an outside contributor actually walks: README -> CONTRIBUTING ->
#   good-first-tasks -> issue / PR templates. Every hop is a plain markdown
#   link, and nothing was checking any of them — which is exactly how
#   CONTRIBUTING.md accumulated four separate stale statements before X-05.
#   This script walks the funnel and fails when a hop is broken.
#
# LEGS
#   (a) required community files exist and are non-empty
#   (b) each GitHub Issue Form carries its required top-level keys
#   (c) relative markdown links in the funnel documents resolve on disk
#   (d) en/ja twins exist in pairs and have the same heading count
#       (a one-sided edit is the usual way a translated page rots)
#   (e) the SSOT pointers CONTRIBUTING hands the reader — the requirement-ID
#       glossary and the architecture map from X-09a — actually exist.
#       Without this leg, CONTRIBUTING can go back to pointing at documents
#       that are not in a public clone, which is the defect X-05-T09 fixed.
#   (f) absolute github.com/<repo>/blob/main/<path> URLs used by the .github
#       templates map back to a path that exists. Those templates must use
#       absolute URLs (a relative link is unreliable in a rendered PR or
#       issue body), which would otherwise put them outside link checking.
#
# PENDING vs DRIFT — and why this script has a third exit code
#   CODE_OF_CONDUCT.md and SECURITY.md are deliberately NOT landed yet. Their
#   enforcement / vulnerability-reporting contact points are undecided
#   (X-05-T04, owner), and publishing a reporting channel that does not reach
#   anybody is worse than publishing none: it converts a would-be private
#   report into either a public issue or silence. Vokra ships a C ABI, raw FFI
#   and nine `unsafe`-permitted crates, so that matters here.
#
#   Collapsing "blocked on an owner decision" into the same FAIL as "somebody
#   broke a link" would produce a permanently red check, and a permanently red
#   check is one people learn to ignore — which would then hide real drift.
#   Reporting them as OK would be worse: the gap would be forgotten.
#
#   So they are reported as PENDING and get their own exit code. When the
#   contacts are decided, drop the drafts prepared under docs/tickets/m5/
#   into place and it turns 0. Do not add a bypass for the pending set.
#
#   CI wiring (revised 2026-07-20, wave 1 integration)
#     This block previously concluded "this script is therefore NOT wired into
#     CI", on the reasoning above: wiring it would make CI permanently red
#     until T04, and a permanently red check is one people learn to ignore.
#     That reasoning is sound but it was choosing between only two options —
#     wire it (permanently red) or leave it out. There is a third: wire it and
#     let the *exit code* carry the distinction. Leg (a) is the only pending
#     one; legs (b)-(f) check real drift (dead links, en/ja twins, missing
#     SSOT targets) and, unwired, nothing enforced them at all — which is the
#     very defect class X-08 exists to close (a gate that never runs is not a
#     gate). So the CI step treats exit 3 as a pass that annotates PENDING and
#     exit 1 as a failure. The original concern is preserved: the check is not
#     permanently red, so it does not train people to ignore it.
#
# MODES
#   scripts/check-community-docs.sh              verify (default)
#   scripts/check-community-docs.sh --list       print what would be checked
#   scripts/check-community-docs.sh --self-test  unit-test the parsers
#   scripts/check-community-docs.sh --help       this text
#
# ZERO-DEP (NFR-DS-02)
#   bash + python3 standard library only.
#   LIMITATION, stated honestly: the Python standard library has no YAML
#   parser, so leg (b) is a structural check (required top-level keys present,
#   file non-empty) and NOT a full parse. A malformed Issue Form that happens
#   to contain the right key lines will pass here and be rejected by GitHub.
#   Final validation of an Issue Form is GitHub's UI.
#
# EXIT CODES
#   0  everything green
#   1  drift — something that should hold right now is broken
#   3  pending only — the X-05-T04 files are absent, everything else is green
#   2  usage / setup error

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

usage() {
    sed -n '3,63p' "$0" | sed 's/^# \{0,1\}//'
}

analyze() {
    python3 - "$1" "$2" <<'PY'
import os
import re
import sys

root, mode = sys.argv[1], sys.argv[2]

# ---- what must exist now -------------------------------------------------
REQUIRED = [
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/question.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
    "docs/good-first-tasks.md",
    "docs/good-first-tasks.ja.md",
]

# ---- blocked on X-05-T04 (owner): contact points undecided ---------------
PENDING = [
    "CODE_OF_CONDUCT.md",
    "CODE_OF_CONDUCT.ja.md",
    "SECURITY.md",
    "SECURITY.ja.md",
]

ISSUE_FORMS = [
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/question.yml",
]
# GitHub Issue Form top-level keys we rely on.
FORM_KEYS = ["name:", "description:", "body:"]

# ---- documents whose relative links must resolve -------------------------
LINK_DOCS = [
    "README.md",
    "README.ja.md",
    "CONTRIBUTING.md",
    "docs/good-first-tasks.md",
    "docs/good-first-tasks.ja.md",
    "docs/governance/quarterly-reviews/README.md",
    "docs/governance/vokra-go-nogo-v0.5.md",
    "docs/governance/exit-path-playbook.md",
]

# ---- en/ja twins that must stay structurally in step ---------------------
TWINS = [
    ("docs/good-first-tasks.md", "docs/good-first-tasks.ja.md"),
]
PENDING_TWINS = [
    ("CODE_OF_CONDUCT.md", "CODE_OF_CONDUCT.ja.md"),
    ("SECURITY.md", "SECURITY.ja.md"),
]

# ---- leg (e): the SSOT targets CONTRIBUTING points the reader at ---------
SSOT_TARGETS = [
    "docs/requirement-ids.md",
    "docs/architecture.md",
]

# ---- leg (f): absolute in-repo URLs used by the .github templates --------
# PR bodies and Issue Form descriptions are rendered outside the file tree, so
# a relative link there is unreliable; those templates use absolute
# github.com/<repo>/blob/<branch>/<path> URLs instead. That would normally
# forfeit drift checking, so map them back to a local path and verify.
SELF_URL_RE = re.compile(
    r"https://github\.com/ayutaz/vokra/blob/main/([^)\s\"']+)"
)
URL_SCANNED = [
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/question.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
]

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
HEADING_RE = re.compile(r"^#{1,6}\s+\S")
FENCE_RE = re.compile(r"^\s*```")


def read(rel):
    path = os.path.join(root, rel)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


def headings(text):
    """Count ATX headings outside fenced code blocks."""
    count, in_fence = 0, False
    for line in text.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence and HEADING_RE.match(line):
            count += 1
    return count


def local_links(text):
    """Relative link targets, with external schemes and pure anchors dropped."""
    out = []
    for raw in LINK_RE.findall(text):
        target = raw.strip().split()[0] if raw.strip() else ""
        if not target or target.startswith("#"):
            continue
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):  # http:, mailto:, …
            continue
        out.append(target.split("#", 1)[0])
    return [t for t in out if t]


if mode == "list":
    print("required :")
    for rel in REQUIRED:
        print(f"  {'OK ' if read(rel) else 'MISS'}  {rel}")
    print("pending (X-05-T04, owner):")
    for rel in PENDING:
        print(f"  {'OK ' if read(rel) else 'PEND'}  {rel}")
    print("link-scanned documents:")
    for rel in LINK_DOCS:
        print(f"  {'OK ' if read(rel) else 'MISS'}  {rel}")
    print("SSOT targets (X-09a):")
    for rel in SSOT_TARGETS:
        print(f"  {'OK ' if read(rel) else 'MISS'}  {rel}")
    sys.exit(0)

drift = []
pending = []


def fail(leg, msg, items=()):
    drift.append(leg)
    print(f"[{leg}] FAIL — {msg}", file=sys.stderr)
    for item in sorted(items):
        print(f"        {item}", file=sys.stderr)


# ---- leg (a) --------------------------------------------------------------
missing, empty = [], []
for rel in REQUIRED:
    text = read(rel)
    if text is None:
        missing.append(rel)
    elif not text.strip():
        empty.append(rel)
if missing:
    fail("a", f"{len(missing)} required community file(s) absent", missing)
if empty:
    fail("a", f"{len(empty)} required community file(s) empty", empty)
if not missing and not empty:
    print(f"[a] OK — {len(REQUIRED)} required community file(s) present")

for rel in PENDING:
    if read(rel) is None:
        pending.append(rel)
if pending:
    print(
        f"[a] PENDING — {len(pending)} file(s) intentionally not landed until "
        "X-05-T04 (owner) fixes the contact points:"
    )
    for rel in sorted(pending):
        print(f"        {rel}")

# ---- leg (b) --------------------------------------------------------------
form_problems = []
for rel in ISSUE_FORMS:
    text = read(rel)
    if text is None:
        continue  # already reported by leg (a)
    for key in FORM_KEYS:
        if not any(line.startswith(key) for line in text.splitlines()):
            form_problems.append(f"{rel}: missing top-level `{key}`")
if form_problems:
    fail("b", f"{len(form_problems)} Issue Form key problem(s)", form_problems)
elif all(read(rel) is not None for rel in ISSUE_FORMS):
    print(f"[b] OK — {len(ISSUE_FORMS)} Issue Form(s) carry their required keys")

# ---- leg (c) --------------------------------------------------------------
broken = []
scanned = 0
for rel in LINK_DOCS:
    text = read(rel)
    if text is None:
        continue  # reported elsewhere; do not double-count
    scanned += 1
    base = os.path.dirname(os.path.join(root, rel))
    for target in local_links(text):
        if not os.path.exists(os.path.normpath(os.path.join(base, target))):
            broken.append(f"{rel} -> {target}")
if broken:
    fail("c", f"{len(broken)} relative link(s) do not resolve", broken)
else:
    print(f"[c] OK — relative links resolve in {scanned} funnel document(s)")

# ---- leg (d) --------------------------------------------------------------
twin_problems = []
for en, ja in TWINS:
    en_text, ja_text = read(en), read(ja)
    if en_text is None or ja_text is None:
        twin_problems.append(f"{en} / {ja}: one side missing")
        continue
    en_h, ja_h = headings(en_text), headings(ja_text)
    if en_h != ja_h:
        twin_problems.append(
            f"{en} ({en_h} headings) vs {ja} ({ja_h}) — one side was edited alone"
        )
if twin_problems:
    fail("d", f"{len(twin_problems)} en/ja twin problem(s)", twin_problems)
else:
    print(f"[d] OK — {len(TWINS)} en/ja twin pair(s) in step")

for en, ja in PENDING_TWINS:
    if read(en) is None and read(ja) is None:
        continue
    if (read(en) is None) != (read(ja) is None):
        fail("d", f"pending pair landed one-sided: {en} / {ja}")

# ---- leg (e) --------------------------------------------------------------
ssot_missing = [rel for rel in SSOT_TARGETS if read(rel) is None]
if ssot_missing:
    fail(
        "e",
        "CONTRIBUTING points the reader at document(s) that do not exist "
        "(X-09a not landed?)",
        ssot_missing,
    )
else:
    print(f"[e] OK — {len(SSOT_TARGETS)} SSOT target(s) exist")

# ---- leg (f) --------------------------------------------------------------
url_broken = []
url_count = 0
for rel in URL_SCANNED:
    text = read(rel)
    if text is None:
        continue  # reported by leg (a)
    for target in SELF_URL_RE.findall(text):
        url_count += 1
        if not os.path.exists(os.path.join(root, target)):
            url_broken.append(f"{rel} -> {target}")
if url_broken:
    fail(
        "f",
        f"{len(url_broken)} absolute in-repo URL(s) point at a missing path",
        url_broken,
    )
else:
    print(f"[f] OK — {url_count} absolute in-repo URL(s) resolve")

# ---- verdict --------------------------------------------------------------
if drift:
    uniq = sorted(set(drift))
    print(
        f"check-community-docs: FAIL — leg(s) {', '.join(uniq)} failed",
        file=sys.stderr,
    )
    sys.exit(1)
if pending:
    print(
        f"check-community-docs: PENDING — {len(pending)} file(s) await "
        "X-05-T04 (owner). Everything else is green."
    )
    sys.exit(3)
print("check-community-docs: OK (all legs pass)")
sys.exit(0)
PY
}

# -------------------------------------------------------------- self-test ---
self_test() {
    local tmproot rc=0
    tmproot="$(mktemp -d -t vokra-comm-check.XXXXXX)"
    trap 'rm -rf "$tmproot"' RETURN

    _scaffold() {
        local dir="$1"
        rm -rf "$dir"
        mkdir -p "$dir/.github/ISSUE_TEMPLATE" "$dir/docs/governance/quarterly-reviews"
        printf '# pr template\n' >"$dir/.github/PULL_REQUEST_TEMPLATE.md"
        for f in bug_report question; do
            cat >"$dir/.github/ISSUE_TEMPLATE/$f.yml" <<'YML'
name: form
description: a form
body:
  - type: markdown
    attributes:
      value: hi
YML
        done
        printf 'blank_issues_enabled: false\n' >"$dir/.github/ISSUE_TEMPLATE/config.yml"
        printf '# tasks\n\n## one\n[c](../CONTRIBUTING.md)\n' >"$dir/docs/good-first-tasks.md"
        printf '# tasks\n\n## one\n[c](../CONTRIBUTING.md)\n' >"$dir/docs/good-first-tasks.ja.md"
        printf '# readme\n' >"$dir/README.md"
        printf '# readme ja\n' >"$dir/README.ja.md"
        printf '# contributing\n[g](docs/good-first-tasks.md)\n' >"$dir/CONTRIBUTING.md"
        printf '# reviews\n' >"$dir/docs/governance/quarterly-reviews/README.md"
        printf '# go/no-go\n' >"$dir/docs/governance/vokra-go-nogo-v0.5.md"
        printf '# exit paths\n' >"$dir/docs/governance/exit-path-playbook.md"
        printf '# ids\n' >"$dir/docs/requirement-ids.md"
        printf '# arch\n' >"$dir/docs/architecture.md"
    }

    _rc() { analyze "$1" verify >/dev/null 2>&1; echo $?; }

    # (1) healthy tree with the pending pair absent -> exit 3, not 0 and not 1
    local d="$tmproot/pending"
    _scaffold "$d"
    if [ "$(_rc "$d")" != "3" ]; then
        echo "self-test FAILED: absent CoC/SECURITY should be PENDING (exit 3)" >&2
        rc=1
    fi

    # (2) the pending pair landed -> fully green
    d="$tmproot/complete"
    _scaffold "$d"
    for f in CODE_OF_CONDUCT SECURITY; do
        printf '# %s\n' "$f" >"$d/$f.md"
        printf '# %s ja\n' "$f" >"$d/$f.ja.md"
    done
    if [ "$(_rc "$d")" != "0" ]; then
        echo "self-test FAILED: a complete tree should exit 0" >&2
        rc=1
    fi

    # (3) a required file missing -> drift (1), and drift must WIN over pending
    d="$tmproot/missing"
    _scaffold "$d"
    rm "$d/.github/PULL_REQUEST_TEMPLATE.md"
    if [ "$(_rc "$d")" != "1" ]; then
        echo "self-test FAILED: a missing required file should be drift (exit 1)" >&2
        rc=1
    fi

    # (4) a broken relative link -> leg (c)
    d="$tmproot/deadlink"
    _scaffold "$d"
    printf '[gone](docs/nope.md)\n' >>"$d/CONTRIBUTING.md"
    if [ "$(_rc "$d")" != "1" ]; then
        echo "self-test FAILED: a dead relative link should fail leg (c)" >&2
        rc=1
    fi

    # (5) external and anchor-only links must NOT be treated as broken
    d="$tmproot/extlink"
    _scaffold "$d"
    printf '[x](https://example.com) [y](#section) [z](mailto:a@b.c)\n' \
        >>"$d/CONTRIBUTING.md"
    if [ "$(_rc "$d")" != "3" ]; then
        echo "self-test FAILED: external/anchor links must not count as broken" >&2
        rc=1
    fi

    # (6) one-sided en/ja edit -> leg (d)
    d="$tmproot/twinskew"
    _scaffold "$d"
    printf '\n## two\n' >>"$d/docs/good-first-tasks.md"
    if [ "$(_rc "$d")" != "1" ]; then
        echo "self-test FAILED: a one-sided en/ja edit should fail leg (d)" >&2
        rc=1
    fi

    # (7) an Issue Form missing a required key -> leg (b)
    d="$tmproot/badform"
    _scaffold "$d"
    printf 'description: only this\n' >"$d/.github/ISSUE_TEMPLATE/question.yml"
    if [ "$(_rc "$d")" != "1" ]; then
        echo "self-test FAILED: an Issue Form missing keys should fail leg (b)" >&2
        rc=1
    fi

    # (8) X-09a SSOT target absent -> leg (e); this is the tripwire that keeps
    #     CONTRIBUTING from pointing at documents a public clone lacks
    d="$tmproot/nossot"
    _scaffold "$d"
    rm "$d/docs/requirement-ids.md"
    if [ "$(_rc "$d")" != "1" ]; then
        echo "self-test FAILED: a missing SSOT target should fail leg (e)" >&2
        rc=1
    fi

    # (9) a heading inside a fenced code block must not count toward parity
    d="$tmproot/fence"
    _scaffold "$d"
    printf '\n```\n# not a heading\n```\n' >>"$d/docs/good-first-tasks.md"
    if [ "$(_rc "$d")" != "3" ]; then
        echo "self-test FAILED: a fenced '#' line must not count as a heading" >&2
        rc=1
    fi

    # (10) landing only one side of a pending pair is drift, not pending
    d="$tmproot/halfpend"
    _scaffold "$d"
    printf '# security\n' >"$d/SECURITY.md"
    if [ "$(_rc "$d")" != "1" ]; then
        echo "self-test FAILED: a one-sided pending pair should be drift" >&2
        rc=1
    fi

    # (11) an absolute in-repo URL in a .github template that points at a
    #      missing path -> leg (f). These URLs are absolute precisely because
    #      relative links are unreliable in a rendered PR/issue body, so
    #      without this leg they would escape drift checking entirely.
    d="$tmproot/deadurl"
    _scaffold "$d"
    printf '[x](https://github.com/ayutaz/vokra/blob/main/docs/gone.md)\n' \
        >>"$d/.github/PULL_REQUEST_TEMPLATE.md"
    if [ "$(_rc "$d")" != "1" ]; then
        echo "self-test FAILED: a dead absolute in-repo URL should fail leg (f)" >&2
        rc=1
    fi

    # (12) an absolute in-repo URL that DOES resolve must not be flagged
    d="$tmproot/liveurl"
    _scaffold "$d"
    printf '[x](https://github.com/ayutaz/vokra/blob/main/CONTRIBUTING.md)\n' \
        >>"$d/.github/PULL_REQUEST_TEMPLATE.md"
    if [ "$(_rc "$d")" != "3" ]; then
        echo "self-test FAILED: a resolving in-repo URL must not be flagged" >&2
        rc=1
    fi

    if [ "$rc" -eq 0 ]; then
        echo "check-community-docs --self-test: OK (12 cases)"
    else
        echo "check-community-docs --self-test: FAILED" >&2
    fi
    return "$rc"
}

case "${1:---verify}" in
    --verify)    analyze "$ROOT" verify ;;
    --list)      analyze "$ROOT" list ;;
    --self-test) self_test ;;
    -h|--help)   usage ;;
    *)
        echo "error: unknown option: $1" >&2
        echo "try: $0 --help" >&2
        exit 2
        ;;
esac
