#!/usr/bin/env bash
# check-doc-references.sh — X-09a-T02 contributor-reference doc drift tripwire.
#
# WHAT IT GATES
#   Vokra's public docs cite requirement IDs (BR / FR / NFR / IF) whose
#   defining documents are deliberately NOT published. `docs/requirement-ids.md`
#   is the public glossary that lets an outside contributor resolve them, and
#   `docs/architecture.md` is the public crate/execution-model map that stands
#   in for the unpublished ADR tree. Both rot silently: a PR can introduce a
#   new requirement ID without adding a glossary entry, drop the Japanese twin
#   out of sync, or rename a crate and leave the architecture doc pointing at
#   a path that no longer exists. This script fails loudly on all four.
#
# LEGS
#   (a)  glossary <-> public docs, BOTH directions
#          - an ID cited anywhere in the public docs but absent from the
#            glossary  = a reader cannot resolve it   (掲載漏れ)
#          - an ID listed in the glossary but cited nowhere
#            = the glossary has drifted / bloated     (掲載過剰)
#   (b)  docs/requirement-ids.md  <->  docs/requirement-ids.ja.md
#          identical listed-ID sets (NFR-MT-04 en/ja parity; catches a
#          one-sided edit)
#   (c)  every `<!-- anchor: <path> -->` in docs/architecture.md resolves to
#        a file that exists
#   (c') docs/architecture.ja.md exists AND carries the same anchor set as
#        the English original
#
#   Legs (b) and (c') exist so that ALL FOUR deliverables of X-09a are covered
#   by at least one leg. A checker that only inspected the English pages would
#   re-introduce, for the Japanese twins, exactly the "nothing is checked"
#   failure mode this script was written to end.
#
#   (d)  every `<!-- anchor: <path> -->` in the X-09b doc-hierarchy files
#        (docs/backend-guide.md + the four new platform tutorials + the
#        api-reference index, each en/ja) resolves to a file that exists, and
#        every one of those files is present  (掲載漏れ / crate リネーム腐り)
#   (d), (e), (f) are the X-09b (doc-hierarchy-completion) extension. They
#   EXTEND this X-09a script rather than living in a second checker, so there
#   is one tool and one CI wiring (the advisory `license`-job step owned by
#   X-09a-T10). See X-09b-T02 / X-09b unresolved-item 1.
#   (e)  each X-09b en/ja pair carries the same number of `##` section
#        headings (NFR-MT-04 two-language parity; tutorials have no requirement
#        IDs, so heading count is the drift proxy that leg (b) is for glossary
#        IDs — a one-sided translation edit is caught)
#   (f)  every repo-relative markdown link in the hierarchy ENTRY pages
#        (docs/getting-started.{md,ja.md} + docs/api-reference.{md,ja.md})
#        resolves to a path that exists — external URLs and pure #fragments are
#        out of scope. getting-started's "Next steps" fans out to every
#        tutorial, so a renamed/absent tutorial surfaces here as a dead link.
#
# ID SYNTAX (the master-set regex — see docs/requirement-ids.md meta section)
#   \b(BR|FR|NFR|IF)-([A-Z]{2}-)?[0-9]+
#   The middle [A-Z]{2}- segment is OPTIONAL. A 3-segment-only regex silently
#   drops the 2-segment IDs (BR-02 / BR-04 / IF-01 / IF-05 / IF-07) and
#   undercounts the master set by 5 (91 -> 86). Do not "simplify" it.
#
# ANCHOR SYNTAX (in docs/architecture.md, mirroring
#                scripts/check-platform-support.sh)
#   <!-- anchor: <repo-relative-path> -->     the path must exist
#   The target may be a file or a directory, and may sit at the repository
#   root (Cargo.toml, CONTRIBUTING.md) — existence is the whole test. A token
#   containing whitespace is prose that leaked into an anchor comment and is
#   rejected with a distinct message.
#   Anchor-driven, never a prose grep — so a sentence mentioning a crate name
#   can neither create a false positive nor mask a real drift.
#
# CORPUS
#   Tracked files only, via `git ls-files`:
#     README.md, README.ja.md, CONTRIBUTING.md, docs/*.md, docs/**/*.md
#   The two glossary files are EXCLUDED from the corpus: the glossary defines
#   IDs rather than citing them, so counting itself would make the
#   "listed but cited nowhere" direction of leg (a) vacuous.
#   Using `git ls-files` (not `find`) is load-bearing: the untracked/ignored
#   internal planning tree (docs/adr/, docs/tickets/) cites many IDs that are
#   deliberately not public, and must never enter the master set.
#
# MODES
#   scripts/check-doc-references.sh              verify (default)
#   scripts/check-doc-references.sh --list       print the resolved sets
#   scripts/check-doc-references.sh --self-test  unit-test the parsers
#   scripts/check-doc-references.sh --help       this text
#
# ZERO-DEP (NFR-DS-02)
#   bash + git + python3 standard library only. No Rust toolchain, no crate,
#   no pip package. Same family as scripts/check-zero-deps.sh and
#   scripts/check-platform-support.sh.
#
# CI WIRING
#   Advisory only (X-09a-T10). Runs as a continue-on-error step in the
#   `license` job of .github/workflows/ci.yml, mirroring the
#   check-platform-support.sh posture. Promotion to a required
#   branch-protection check is an owner decision (NFR-MT-07) tracked by
#   X-08-T28.
#
# EXIT CODES
#   0  all legs pass (or --list / --self-test / --help success)
#   1  one or more legs failed (doc drift)
#   2  usage / setup error (not a git repo, empty corpus, bad flag)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

usage() {
    sed -n '3,80p' "$0" | sed 's/^# \{0,1\}//'
}

# ------------------------------------------------------------------ core ---
# analyze <root> <mode>  — mode is "verify" or "list".
# Returns 0 (all legs pass), 1 (>=1 leg failed), 2 (setup error).
# Never calls `exit`, so the self-test can invoke it repeatedly.
analyze() {
    python3 - "$1" "$2" <<'PY'
import os
import re
import subprocess
import sys

root, mode = sys.argv[1], sys.argv[2]

GLOSSARY = "docs/requirement-ids.md"
GLOSSARY_JA = "docs/requirement-ids.ja.md"
ARCH = "docs/architecture.md"
ARCH_JA = "docs/architecture.ja.md"

CORPUS_SPECS = [
    "README.md",
    "README.ja.md",
    "CONTRIBUTING.md",
    "docs/*.md",
    "docs/**/*.md",
]
# The glossary defines IDs; it does not cite them. Counting it would make the
# "listed in the glossary but cited nowhere" direction of leg (a) vacuous.
CORPUS_EXCLUDE = {GLOSSARY, GLOSSARY_JA}

# Master-set regex. The middle [A-Z]{2}- segment is OPTIONAL (2-segment IDs
# such as BR-02 / IF-01 exist); requiring it undercounts by 5.
ID_RE = re.compile(r"\b(?:BR|FR|NFR|IF)-(?:[A-Z]{2}-)?[0-9]+")

# A glossary ENTRY is the first cell of a markdown table row. Prose mentions
# in the meta section (e.g. "most cited: FR-EX-08") must not count as entries,
# otherwise the glossary could "pass" leg (a) without actually defining the ID.
ROW_RE = re.compile(
    r"^\|\s*[`*]*((?:BR|FR|NFR|IF)-(?:[A-Z]{2}-)?[0-9]+)[`*]*\s*\|"
)

ANCHOR_RE = re.compile(r"<!--\s*anchor:\s*(.*?)\s*-->")

# --- X-09b (doc-hierarchy-completion) extension --------------------------
# The doc-hierarchy deliverables: backend-guide + the four platform tutorials
# that X-09a left out (cli / android / godot / server) + the api-reference
# index, each an en/ja pair (NFR-MT-04). Legs (d)/(e)/(f) fail loudly if any
# of these rot. See X-09b-T02.
X09B_PAIRS = [
    ("docs/backend-guide.md", "docs/backend-guide.ja.md"),
    ("docs/tutorials/cli.md", "docs/tutorials/cli.ja.md"),
    ("docs/tutorials/android.md", "docs/tutorials/android.ja.md"),
    ("docs/tutorials/godot.md", "docs/tutorials/godot.ja.md"),
    ("docs/tutorials/server.md", "docs/tutorials/server.ja.md"),
    ("docs/api-reference.md", "docs/api-reference.ja.md"),
]
# Entry pages whose relative links must resolve (leg f): the top of the
# hierarchy and the API index.
X09B_LINK_DOCS = [
    "docs/getting-started.md",
    "docs/getting-started.ja.md",
    "docs/api-reference.md",
    "docs/api-reference.ja.md",
]

# A `## ` section heading (leg e). ATX level-2; deeper/shallower levels are
# not counted so the proxy stays coarse and translation-robust.
HEADING_RE = re.compile(r"^##\s+\S")

# An inline markdown link target: `[text](target)`. Only repo-relative targets
# are existence-checked (leg f); external URLs / mail / pure #fragments are
# skipped by the caller.
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def headings_in(text):
    return sum(1 for ln in text.splitlines() if HEADING_RE.match(ln)) if text else 0


def links_in(text):
    return LINK_RE.findall(text) if text else []


def read(rel):
    path = os.path.join(root, rel)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


def corpus_files():
    """Tracked corpus paths, sorted-unique, glossary files removed."""
    proc = subprocess.run(
        ["git", "-C", root, "ls-files", "--"] + CORPUS_SPECS,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    paths = {p for p in proc.stdout.splitlines() if p}
    return sorted(paths - CORPUS_EXCLUDE)


def ids_in(text):
    return set(ID_RE.findall(text)) if text else set()


def entries_in(text):
    """IDs that occupy the first cell of a table row."""
    if not text:
        return set()
    out = set()
    for line in text.splitlines():
        match = ROW_RE.match(line.strip())
        if match:
            out.add(match.group(1))
    return out


def anchors_in(text):
    if not text:
        return set()
    return {a for a in (m.strip() for m in ANCHOR_RE.findall(text)) if a}


files = corpus_files()
if files is None:
    print("error: `git ls-files` failed (not a git repository?)", file=sys.stderr)
    sys.exit(2)
if not files:
    print("error: corpus is empty (no tracked public docs found)", file=sys.stderr)
    sys.exit(2)

cited = set()
for rel in files:
    text = read(rel)
    if text is not None:
        cited |= ids_in(text)

glossary_text = read(GLOSSARY)
glossary_ja_text = read(GLOSSARY_JA)
arch_text = read(ARCH)
arch_ja_text = read(ARCH_JA)

listed = entries_in(glossary_text)
listed_ja = entries_in(glossary_ja_text)
anchors = anchors_in(arch_text)
anchors_ja = anchors_in(arch_ja_text)

if mode == "list":
    print(f"corpus files      : {len(files)}")
    print(f"cited IDs         : {len(cited)}")
    print(f"glossary entries  : {len(listed)}   ({GLOSSARY})")
    print(f"glossary ja       : {len(listed_ja)}   ({GLOSSARY_JA})")
    print(f"architecture anch : {len(anchors)}   ({ARCH})")
    print(f"architecture ja   : {len(anchors_ja)}   ({ARCH_JA})")
    print()
    for i in sorted(cited):
        print(f"  ID     {i}")
    for a in sorted(anchors):
        print(f"  ANCHOR {a}")
    print()
    print("X-09b doc-hierarchy files (leg d/e):")
    for en, ja in X09B_PAIRS:
        for rel in (en, ja):
            t = read(rel)
            mark = "OK " if t is not None else "MISS"
            nheads = headings_in(t) if t is not None else 0
            print(f"  {mark} {rel}   ({nheads} '##')")
    sys.exit(0)

failed = []


def fail(leg, msg, items=()):
    failed.append(leg)
    print(f"[{leg}] FAIL — {msg}", file=sys.stderr)
    for item in sorted(items):
        print(f"        {item}", file=sys.stderr)


# ---- leg (a): glossary <-> public docs, both directions -------------------
if glossary_text is None:
    fail("a", f"{GLOSSARY} not found")
else:
    missing = cited - listed
    extra = listed - cited
    if missing:
        fail(
            "a",
            f"{len(missing)} ID(s) cited in public docs but absent from the "
            "glossary (add a row, or the reader cannot resolve them)",
            missing,
        )
    if extra:
        fail(
            "a",
            f"{len(extra)} ID(s) listed in the glossary but cited nowhere in "
            "the public docs (stale entry — remove it or cite it)",
            extra,
        )
    if not missing and not extra:
        print(f"[a] OK — glossary and public docs agree on {len(listed)} ID(s)")

# ---- leg (b): en/ja glossary ID-set parity -------------------------------
if glossary_text is None or glossary_ja_text is None:
    which = GLOSSARY if glossary_text is None else GLOSSARY_JA
    fail("b", f"{which} not found (en/ja glossary parity cannot hold)")
else:
    only_en = listed - listed_ja
    only_ja = listed_ja - listed
    if only_en:
        fail("b", f"{len(only_en)} ID(s) only in the English glossary", only_en)
    if only_ja:
        fail("b", f"{len(only_ja)} ID(s) only in the Japanese glossary", only_ja)
    if not only_en and not only_ja:
        print(f"[b] OK — en/ja glossaries list the same {len(listed)} ID(s)")

# ---- leg (c): architecture anchors resolve -------------------------------
if arch_text is None:
    fail("c", f"{ARCH} not found")
elif not anchors:
    fail("c", f"{ARCH} carries no <!-- anchor: ... --> lines (drift undetectable)")
else:
    broken = []
    for anchor in sorted(anchors):
        # A whitespace-bearing token is prose that slipped into an anchor
        # comment, not a path. Everything else is judged purely by whether it
        # exists, so root-level targets (Cargo.toml, CONTRIBUTING.md) and
        # directories are both legal anchors.
        if any(ch.isspace() for ch in anchor):
            broken.append(f"{anchor}  (not a path — prose in an anchor comment)")
        elif not os.path.exists(os.path.join(root, anchor)):
            broken.append(f"{anchor}  (path not found)")
    if broken:
        fail("c", f"{len(broken)} anchor(s) in {ARCH} do not resolve", broken)
    else:
        print(f"[c] OK — {len(anchors)} architecture anchor(s) resolve")

# ---- leg (c'): ja twin exists and mirrors the anchor set -----------------
if arch_ja_text is None:
    fail("c'", f"{ARCH_JA} not found (NFR-MT-04 en/ja pair incomplete)")
elif arch_text is None:
    fail("c'", f"{ARCH} not found (cannot compare anchor sets)")
else:
    only_en = anchors - anchors_ja
    only_ja = anchors_ja - anchors
    if only_en:
        fail("c'", f"{len(only_en)} anchor(s) only in {ARCH}", only_en)
    if only_ja:
        fail("c'", f"{len(only_ja)} anchor(s) only in {ARCH_JA}", only_ja)
    if not only_en and not only_ja:
        print(f"[c'] OK — en/ja architecture docs share {len(anchors)} anchor(s)")

# ---- leg (d): X-09b doc-hierarchy files exist and their anchors resolve ---
d_missing = []
d_broken = []
for en, ja in X09B_PAIRS:
    for rel in (en, ja):
        txt = read(rel)
        if txt is None:
            d_missing.append(rel)
            continue
        for anchor in sorted(anchors_in(txt)):
            if any(ch.isspace() for ch in anchor):
                d_broken.append(f"{rel}: {anchor}  (not a path — prose in an anchor comment)")
            elif not os.path.exists(os.path.join(root, anchor)):
                d_broken.append(f"{rel}: {anchor}  (path not found)")
if d_missing:
    fail("d", f"{len(d_missing)} X-09b hierarchy doc(s) missing (NFR-MT-04 two-language pair)", d_missing)
if d_broken:
    fail("d", f"{len(d_broken)} anchor(s) in the X-09b docs do not resolve", d_broken)
if not d_missing and not d_broken:
    print(f"[d] OK — {len(X09B_PAIRS) * 2} X-09b doc(s) present, all anchors resolve")

# ---- leg (e): X-09b en/ja heading-count parity ---------------------------
e_skew = []
e_compared = 0
for en, ja in X09B_PAIRS:
    en_txt, ja_txt = read(en), read(ja)
    if en_txt is None or ja_txt is None:
        continue  # absence already reported by leg (d)
    e_compared += 1
    ne, nj = headings_in(en_txt), headings_in(ja_txt)
    if ne != nj:
        e_skew.append(f"{en} has {ne} '##' heading(s) but {ja} has {nj} (en/ja drift)")
if e_skew:
    fail("e", f"{len(e_skew)} X-09b pair(s) have an en/ja heading-count skew (NFR-MT-04)", e_skew)
elif e_compared:
    print(f"[e] OK — {e_compared} X-09b en/ja pair(s) share their '##' heading count")
else:
    print("[e] SKIP — no X-09b pair had both halves present (see leg d)")

# ---- leg (f): entry-page relative links resolve --------------------------
f_missing = []
f_broken = []
for rel in X09B_LINK_DOCS:
    txt = read(rel)
    if txt is None:
        f_missing.append(rel)
        continue
    base = os.path.dirname(rel)
    for target in links_in(txt):
        t = target.strip()
        if t.startswith(("http://", "https://", "mailto:", "#")):
            continue
        t = t.split("#", 1)[0]  # strip in-page #fragment
        if not t:
            continue
        resolved = os.path.normpath(os.path.join(base, t))
        if not os.path.exists(os.path.join(root, resolved)):
            f_broken.append(f"{rel}: [..]({target}) -> {resolved} (not found)")
if f_missing:
    fail("f", f"{len(f_missing)} hierarchy entry page(s) missing", f_missing)
if f_broken:
    fail("f", f"{len(f_broken)} relative link(s) in the entry pages do not resolve", f_broken)
if not f_missing and not f_broken:
    print("[f] OK — entry-page relative links resolve")

if failed:
    uniq = sorted(set(failed))
    print(
        f"check-doc-references: FAIL — leg(s) {', '.join(uniq)} failed "
        "(contributor-reference doc drift)",
        file=sys.stderr,
    )
    sys.exit(1)

print("check-doc-references: OK (all legs pass)")
sys.exit(0)
PY
}

# -------------------------------------------------------------- self-test ---
# self_test — build throwaway git repos and assert each leg fires when it
# should and stays quiet when it should. Guards the parsers (ID regex, table
# row extraction, anchor extraction, set diff) against regression, so a
# checker bug can never manufacture a passing run.
self_test() {
    local tmproot rc=0
    tmproot="$(mktemp -d -t vokra-docref-check.XXXXXX)"
    trap 'rm -rf "$tmproot"' RETURN

    # --- scaffold a minimal but REALISTIC repo -----------------------------
    _scaffold() {
        local dir="$1"
        rm -rf "$dir"
        mkdir -p "$dir/docs" "$dir/crates/demo/src"
        git -C "$dir" init -q 2>/dev/null || return 1
        printf 'fn main() {}\n' >"$dir/crates/demo/src/lib.rs"
        # Public docs cite three IDs, one of them 2-segment (IF-01) so the
        # regex's optional middle segment is genuinely exercised.
        printf '# readme\nSee FR-EX-08 and NFR-DS-02.\n' >"$dir/README.md"
        printf '# contributing\nAlso IF-01.\n' >"$dir/CONTRIBUTING.md"
        cat >"$dir/docs/requirement-ids.md" <<'MD'
# glossary
Most cited: NFR-PF-03 (prose only — must NOT count as an entry).
| ID | meaning |
|---|---|
| `FR-EX-08` | explicit errors |
| `NFR-DS-02` | zero dependency |
| `IF-01` | C ABI consumers |
MD
        cp "$dir/docs/requirement-ids.md" "$dir/docs/requirement-ids.ja.md"
        cat >"$dir/docs/architecture.md" <<'MD'
# architecture
<!-- anchor: crates/demo/src/lib.rs -->
MD
        cp "$dir/docs/architecture.md" "$dir/docs/architecture.ja.md"
        # A gitignored internal tree citing an ID that is NOT public: proves
        # the corpus is index-driven and never leaks internal scope.
        mkdir -p "$dir/docs/adr"
        printf 'internal only: FR-OP-99\n' >"$dir/docs/adr/secret.md"
        printf '/docs/adr/\n' >"$dir/.gitignore"
        # X-09b doc-hierarchy files (legs d/e/f). Anchors point at the demo
        # crate; en/ja heading counts are balanced (cp); entry-page links
        # resolve. A healthy scaffold must pass legs d/e/f as well as a/b/c/c'.
        mkdir -p "$dir/docs/tutorials"
        local t
        for t in cli android godot server; do
            cat >"$dir/docs/tutorials/$t.md" <<'MD'
# tutorial
<!-- anchor: crates/demo/src/lib.rs -->
## 1. one
## 2. two
MD
            cp "$dir/docs/tutorials/$t.md" "$dir/docs/tutorials/$t.ja.md"
        done
        cat >"$dir/docs/backend-guide.md" <<'MD'
# backend guide
<!-- anchor: crates/demo/src/lib.rs -->
## 1. one
## 2. two
MD
        cp "$dir/docs/backend-guide.md" "$dir/docs/backend-guide.ja.md"
        cat >"$dir/docs/api-reference.md" <<'MD'
# api reference
## 1. rust
See [the backend guide](backend-guide.md).
MD
        cp "$dir/docs/api-reference.md" "$dir/docs/api-reference.ja.md"
        cat >"$dir/docs/getting-started.md" <<'MD'
# getting started
## next steps
- [cli tutorial](tutorials/cli.md)
- [backend guide](backend-guide.md)
- [api reference](api-reference.md)
- [external](https://example.com)
MD
        cp "$dir/docs/getting-started.md" "$dir/docs/getting-started.ja.md"
        git -C "$dir" add -A >/dev/null 2>&1
        return 0
    }

    local good="$tmproot/good"
    if ! _scaffold "$good"; then
        echo "self-test SKIPPED: git unavailable" >&2
        return 0
    fi

    # (1) a healthy repo passes every leg
    if ! analyze "$good" verify >/dev/null 2>&1; then
        echo "self-test FAILED: a healthy doc set should pass" >&2
        rc=1
    fi

    # (2) an ID cited in public docs but missing from the glossary -> leg (a)
    local d="$tmproot/miss"
    _scaffold "$d"
    printf 'New requirement NFR-QL-01 landed.\n' >>"$d/README.md"
    git -C "$d" add -A >/dev/null 2>&1
    if analyze "$d" verify >/dev/null 2>&1; then
        echo "self-test FAILED: an uncataloged ID should fail leg (a)" >&2
        rc=1
    fi

    # (3) an ID listed in the glossary but cited nowhere -> leg (a) reverse
    d="$tmproot/extra"
    _scaffold "$d"
    printf '| `NFR-MT-07` | ci gates |\n' >>"$d/docs/requirement-ids.md"
    printf '| `NFR-MT-07` | ci gates |\n' >>"$d/docs/requirement-ids.ja.md"
    git -C "$d" add -A >/dev/null 2>&1
    if analyze "$d" verify >/dev/null 2>&1; then
        echo "self-test FAILED: a stale glossary entry should fail leg (a)" >&2
        rc=1
    fi

    # (4) one-sided en/ja glossary edit -> leg (b)
    d="$tmproot/jaskew"
    _scaffold "$d"
    printf '| `NFR-MT-07` | ci gates |\n' >>"$d/docs/requirement-ids.md"
    printf 'Cites NFR-MT-07 too.\n' >>"$d/README.md"
    git -C "$d" add -A >/dev/null 2>&1
    if analyze "$d" verify >/dev/null 2>&1; then
        echo "self-test FAILED: a one-sided en/ja edit should fail leg (b)" >&2
        rc=1
    fi

    # (5) architecture anchor pointing at a deleted path -> leg (c)
    d="$tmproot/anchor"
    _scaffold "$d"
    printf '<!-- anchor: crates/gone/src/lib.rs -->\n' >>"$d/docs/architecture.md"
    printf '<!-- anchor: crates/gone/src/lib.rs -->\n' >>"$d/docs/architecture.ja.md"
    git -C "$d" add -A >/dev/null 2>&1
    if analyze "$d" verify >/dev/null 2>&1; then
        echo "self-test FAILED: a dangling anchor should fail leg (c)" >&2
        rc=1
    fi

    # (6) missing Japanese architecture twin -> leg (c')
    d="$tmproot/nojatwin"
    _scaffold "$d"
    rm "$d/docs/architecture.ja.md"
    git -C "$d" add -A >/dev/null 2>&1
    if analyze "$d" verify >/dev/null 2>&1; then
        echo "self-test FAILED: a missing ja architecture twin should fail leg (c')" >&2
        rc=1
    fi

    # (7) ja architecture twin that lost an anchor -> leg (c') parity
    d="$tmproot/jaanchor"
    _scaffold "$d"
    printf '# architecture (ja)\nno anchors here\n' >"$d/docs/architecture.ja.md"
    git -C "$d" add -A >/dev/null 2>&1
    if analyze "$d" verify >/dev/null 2>&1; then
        echo "self-test FAILED: an en/ja anchor mismatch should fail leg (c')" >&2
        rc=1
    fi

    # (8) architecture doc with no anchors at all -> leg (c)
    d="$tmproot/noanchor"
    _scaffold "$d"
    printf '# architecture\nprose only\n' >"$d/docs/architecture.md"
    printf '# architecture\nprose only\n' >"$d/docs/architecture.ja.md"
    git -C "$d" add -A >/dev/null 2>&1
    if analyze "$d" verify >/dev/null 2>&1; then
        echo "self-test FAILED: an anchor-less architecture doc should fail leg (c)" >&2
        rc=1
    fi

    # (9) prose mention must not satisfy leg (a): NFR-PF-03 appears in the
    #     glossary's meta prose only. Cite it from a public doc and the
    #     checker must still demand a real table row.
    d="$tmproot/prose"
    _scaffold "$d"
    printf 'Cites NFR-PF-03.\n' >>"$d/README.md"
    git -C "$d" add -A >/dev/null 2>&1
    if analyze "$d" verify >/dev/null 2>&1; then
        echo "self-test FAILED: a prose-only glossary mention should not count as an entry" >&2
        rc=1
    fi

    # (10) an ID that lives only in the gitignored internal tree must NOT be
    #      demanded of the glossary (corpus is git-index driven).
    #      The healthy scaffold already contains docs/adr/secret.md citing
    #      FR-OP-99; case (1) passing is the proof. Assert explicitly that the
    #      internal ID never shows up in --list output.
    if analyze "$good" list 2>/dev/null | grep -q 'FR-OP-99'; then
        echo "self-test FAILED: an ignored internal doc leaked into the corpus" >&2
        rc=1
    fi

    # (11) a root-level anchor (no slash) is legal as long as it exists —
    #      docs/architecture.md anchors Cargo.toml and CONTRIBUTING.md.
    d="$tmproot/rootanchor"
    _scaffold "$d"
    printf '<!-- anchor: CONTRIBUTING.md -->\n' >>"$d/docs/architecture.md"
    printf '<!-- anchor: CONTRIBUTING.md -->\n' >>"$d/docs/architecture.ja.md"
    git -C "$d" add -A >/dev/null 2>&1
    if ! analyze "$d" verify >/dev/null 2>&1; then
        echo "self-test FAILED: an existing root-level anchor should pass" >&2
        rc=1
    fi

    # (12) prose that leaked into an anchor comment -> leg (c) malformed
    d="$tmproot/prosanchor"
    _scaffold "$d"
    printf '<!-- anchor: see the crate layout -->\n' >>"$d/docs/architecture.md"
    printf '<!-- anchor: see the crate layout -->\n' >>"$d/docs/architecture.ja.md"
    git -C "$d" add -A >/dev/null 2>&1
    if analyze "$d" verify >/dev/null 2>&1; then
        echo "self-test FAILED: prose in an anchor comment should fail leg (c)" >&2
        rc=1
    fi

    # (13) a dangling anchor in an X-09b hierarchy doc -> leg (d)
    d="$tmproot/x09b-anchor"
    _scaffold "$d"
    printf '<!-- anchor: crates/gone/src/lib.rs -->\n' >>"$d/docs/backend-guide.md"
    printf '<!-- anchor: crates/gone/src/lib.rs -->\n' >>"$d/docs/backend-guide.ja.md"
    git -C "$d" add -A >/dev/null 2>&1
    if analyze "$d" verify >/dev/null 2>&1; then
        echo "self-test FAILED: a dangling X-09b anchor should fail leg (d)" >&2
        rc=1
    fi

    # (14) a one-sided en/ja heading edit in an X-09b tutorial -> leg (e)
    d="$tmproot/x09b-heading"
    _scaffold "$d"
    printf '## 3. three\n' >>"$d/docs/tutorials/cli.md"   # ja twin left unchanged
    git -C "$d" add -A >/dev/null 2>&1
    if analyze "$d" verify >/dev/null 2>&1; then
        echo "self-test FAILED: an en/ja heading skew should fail leg (e)" >&2
        rc=1
    fi

    # (15) a dead relative link on a hierarchy entry page -> leg (f)
    d="$tmproot/x09b-link"
    _scaffold "$d"
    printf '%s\n' '- [dead](tutorials/nonexistent.md)' >>"$d/docs/getting-started.md"
    git -C "$d" add -A >/dev/null 2>&1
    if analyze "$d" verify >/dev/null 2>&1; then
        echo "self-test FAILED: a dead entry-page link should fail leg (f)" >&2
        rc=1
    fi

    # (16) a missing Japanese twin of an X-09b doc -> leg (d)
    d="$tmproot/x09b-noja"
    _scaffold "$d"
    rm "$d/docs/tutorials/server.ja.md"
    git -C "$d" add -A >/dev/null 2>&1
    if analyze "$d" verify >/dev/null 2>&1; then
        echo "self-test FAILED: a missing X-09b ja twin should fail leg (d)" >&2
        rc=1
    fi

    if [ "$rc" -eq 0 ]; then
        echo "check-doc-references --self-test: OK (16 cases)"
    else
        echo "check-doc-references --self-test: FAILED" >&2
    fi
    return "$rc"
}

# ------------------------------------------------------------------ main ---
case "${1:---verify}" in
    --verify)
        analyze "$ROOT" verify
        ;;
    --list)
        analyze "$ROOT" list
        ;;
    --self-test)
        self_test
        ;;
    -h|--help)
        usage
        ;;
    *)
        echo "error: unknown option: $1" >&2
        echo "try: $0 --help" >&2
        exit 2
        ;;
esac
