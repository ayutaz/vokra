#!/usr/bin/env bash
# check-owner-checklist-drift.sh — M4-RESIDUAL-A owner-queue doc drift tripwire.
#
# WHAT IT GATES
#   docs/m4-owner-verification-checklist.md,
#   docs/m3-owner-verification-checklist.md and docs/handoff/m4-19.md are the
#   owner's hand-off surface: the owner reads them and runs the commands they
#   name. When the implementation moves and these docs do not, the owner is
#   handed a procedure that cannot work. The concrete regression this script
#   exists to prevent: two of those docs told the owner to pass vokra-server
#   CLI flags that do not exist in HEAD, so the very first command of the
#   hand-off would have died with an unknown-flag error.
#
#   Two independent checks:
#
#   (1) FORBIDDEN TOKENS — literal strings confirmed ABSENT from HEAD (dead
#       CLI flag names) must not appear in any tracked markdown. A dead flag
#       name has no legitimate use in a procedure doc, so this check cannot
#       produce a false positive. Same shape as
#       scripts/check-forbidden-symbols.sh.
#
#   (2) CLAIM-EVIDENCE ANCHORS — every "CC reached this far" claim that these
#       docs make carries a machine-checkable pointer at the implementation or
#       committed record backing it. If the evidence file is renamed, deleted,
#       or loses the cited token, the claim is no longer substantiated and this
#       fails. The check is anchor-driven — it never guesses from prose (the
#       design principle of scripts/check-platform-support.sh), so no false
#       positive/negative from a natural-language grep.
#
# ANCHOR SYNTAX (in the owner docs)
#   <!-- claim-evidence: <path> -->          file must exist
#   <!-- claim-evidence: <path>#<token> -->  file must exist AND contain the
#                                            literal <token>
#
# TRACKED-ONLY RULE (authoring error, exit 2)
#   An anchor target must be a TRACKED path. The three owner docs are public;
#   gitignore-local internal docs (/docs/milestones.md, /docs/tickets/,
#   /docs/adr/ — .gitignore lines 57 / 61 / 62) are absent from a fresh clone,
#   so resolving one would fail in public CI for the wrong reason. Embedding a
#   local path in a tracked doc also breaks the project rule that tracked docs
#   reference local internal docs by ID only. Anchoring a gitignored target is
#   therefore an authoring mistake, reported as exit 2 (setup error), never as
#   drift.
#
# MODES
#   scripts/check-owner-checklist-drift.sh              verify (default)
#   scripts/check-owner-checklist-drift.sh --list       print resolved anchors
#   scripts/check-owner-checklist-drift.sh --self-test  unit-test the parser
#   scripts/check-owner-checklist-drift.sh --help       this text
#
# ZERO-DEP (NFR-DS-02)
#   Pure bash + grep + sed + git. No Rust toolchain, no crate, no external
#   binary beyond coreutils and the git already required to check out the repo.
#   Same family as scripts/check-zero-deps.sh, scripts/check-abi-changelog.sh
#   and scripts/check-platform-support.sh.
#
# CI WIRING
#   Advisory only (M4-RESIDUAL-A-T11). Runs as a continue-on-error step in the
#   `license` job of .github/workflows/ci.yml, mirroring the
#   `Platform-support matrix drift check (advisory, M4-11-T08)` step. Promotion
#   to a required branch-protection check is an owner decision (NFR-MT-07).
#
# EXIT CODES
#   0  no forbidden token, all anchors resolve (or --list/--self-test/--help)
#   1  drift: a forbidden token reappeared, or an anchor lost its target
#   2  usage / setup / authoring error (doc missing, no anchors, gitignored
#      anchor target, bad flag)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# The owner-facing docs that carry claim-evidence anchors. All three are
# tracked (public); see the TRACKED-ONLY RULE above.
CLAIM_DOCS=(
    "docs/m4-owner-verification-checklist.md"
    "docs/m3-owner-verification-checklist.md"
    "docs/handoff/m4-19.md"
)

# vokra-server CLI flag names that never existed in a shipped HEAD. The real
# names are parsed in integrations/vokra-server/src/config.rs
# (--whisper-base / --piper-plus / --piper-g2p). bash 3.2 (macOS) compatible.
FORBIDDEN_TOKENS=(
    "--asr-base"
    "--tts-piper"
)

usage() {
    sed -n '3,74p' "$0" | sed 's/^# \{0,1\}//'
}

# --------------------------------------------------------------- helpers ---
# is_ignored <root> <path> — 0 when <path> is gitignored under <root>.
# Outside a git work tree git errors out and we report "not ignored": the
# tracked-only rule can only be enforced where the ignore rules are readable,
# and .gitignore is itself tracked so a real checkout always has them.
is_ignored() {
    local root="$1" path="$2"
    git -C "$root" check-ignore -q -- "$path" 2>/dev/null
}

# tracked_markdown <root> — every tracked *.md path, one per line.
# Falls back to `find` when <root> is not a git work tree (or has nothing
# staged) so the self-test can run against a throwaway tree.
tracked_markdown() {
    local root="$1" listed=""
    listed="$(git -C "$root" ls-files -- '*.md' 2>/dev/null || true)"
    if [ -n "$listed" ]; then
        printf '%s\n' "$listed"
        return 0
    fi
    (cd "$root" && find . -name '*.md' -type f 2>/dev/null | sed 's|^\./||') || true
}

# --------------------------------------------------------------- extract ---
# extract_claims <doc-path> — emit one anchor token per line (unsorted, so a
# caller can report per-doc); an anchor-less doc yields nothing without
# tripping `set -e`.
extract_claims() {
    local doc="$1"
    grep -oE '<!-- claim-evidence:[^>]*-->' "$doc" 2>/dev/null \
        | sed -E 's/^<!-- claim-evidence:[[:space:]]*//; s/[[:space:]]*-->$//' \
        | sed -E 's/[[:space:]]+$//' \
        || true
}

# --------------------------------------------------------------- resolve ---
# check_claim <root> <token> — 0 resolves, 1 drift, 2 authoring error.
# Prints a one-line reason to stderr on failure.
check_claim() {
    local root="$1" token="$2"
    local path tok full
    if [[ "$token" == *"#"* ]]; then
        path="${token%%#*}"
        tok="${token#*#}"
    else
        path="$token"
        tok=""
    fi
    # A real anchor is always a repo-relative path (contains a '/'). A token
    # without one is a stray placeholder or typo, not a valid anchor.
    case "$path" in
        */*) : ;;
        *)
            echo "  MALFORMED    : $token  (anchor token is not a repo-relative path)" >&2
            return 1
            ;;
    esac
    if is_ignored "$root" "$path"; then
        echo "  LOCAL TARGET : $token  ($path is gitignore-local — tracked docs must reference internal docs by ID only)" >&2
        return 2
    fi
    full="$root/$path"
    if [ ! -f "$full" ]; then
        echo "  MISSING FILE : $token  ($path not found)" >&2
        return 1
    fi
    if [ -n "$tok" ] && ! grep -qF -- "$tok" "$full"; then
        echo "  MISSING TOKEN: $token  ('$tok' not in $path)" >&2
        return 1
    fi
    return 0
}

# ----------------------------------------------------------- forbidden ------
# scan_forbidden <root> — 0 clean, 1 a dead token reappeared in tracked docs.
scan_forbidden() {
    local root="$1" docs sym matches rc=0
    docs="$(tracked_markdown "$root")"
    if [ -z "$docs" ]; then
        echo "error: no tracked markdown found under $root" >&2
        return 2
    fi
    for sym in "${FORBIDDEN_TOKENS[@]}"; do
        matches="$(
            printf '%s\n' "$docs" \
                | (cd "$root" && xargs grep -nF -e "$sym" 2>/dev/null) \
                || true
        )"
        if [ -n "$matches" ]; then
            echo "  DEAD FLAG    : '$sym' is not parsed by integrations/vokra-server/src/config.rs — an owner following this doc hits an unknown-flag error:" >&2
            printf '    %s\n' "$matches" >&2
            rc=1
        fi
    done
    return "$rc"
}

# ----------------------------------------------------------------- verify ---
# run_verify <root> <doc>... — never calls `exit`, so the self-test can invoke
# it repeatedly. Returns 0 (clean), 1 (drift) or 2 (setup/authoring error).
run_verify() {
    local root="$1"
    shift
    local docs=("$@")
    local doc total=0 bad=0 setup=0 token rc

    # `|| rc=$?` keeps every fallible call in a condition context: toggling
    # `set -e` inside a function would leak back to the caller (and re-arming
    # errexit here would abort the script on this function's own non-zero
    # return, before the caller could inspect it).
    rc=0
    scan_forbidden "$root" || rc=$?
    case "$rc" in
        0) : ;;
        2) setup=1 ;;
        *) bad=$((bad + 1)) ;;
    esac

    for doc in "${docs[@]}"; do
        if [ ! -f "$root/$doc" ]; then
            echo "error: owner doc not found: $doc" >&2
            setup=1
            continue
        fi
        local anchors
        anchors="$(extract_claims "$root/$doc")"
        if [ -z "$anchors" ]; then
            echo "error: no <!-- claim-evidence: ... --> anchors in $doc" >&2
            setup=1
            continue
        fi
        while IFS= read -r token; do
            [ -n "$token" ] || continue
            total=$((total + 1))
            rc=0
            check_claim "$root" "$token" || rc=$?
            case "$rc" in
                0) : ;;
                2) setup=1 ;;
                *) bad=$((bad + 1)) ;;
            esac
        done <<EOF
$anchors
EOF
    done

    if [ "$setup" -ne 0 ]; then
        echo "check-owner-checklist-drift: SETUP ERROR — see messages above" >&2
        return 2
    fi
    if [ "$bad" -ne 0 ]; then
        echo "check-owner-checklist-drift: FAIL — $bad problem(s) (owner-doc drift)" >&2
        return 1
    fi
    echo "check-owner-checklist-drift: OK ($total claim-evidence anchors resolve, no dead flag names)"
    return 0
}

# -------------------------------------------------------------- self-test ---
# self_test — synthesize a throwaway git repo and assert the checker passes a
# good tree and fails on each drift / authoring mode. Catches parser drift
# without touching the real docs.
self_test() {
    local tmproot rc=0 out
    tmproot="$(mktemp -d -t vokra-owner-drift.XXXXXX)"
    trap 'rm -rf "$tmproot"' RETURN

    mkdir -p "$tmproot/docs" "$tmproot/src" "$tmproot/local"
    printf '/local/\n' >"$tmproot/.gitignore"
    printf 'fn main() { let flag = "--whisper-base"; }\n' >"$tmproot/src/config.rs"
    printf '# local internal doc\n' >"$tmproot/local/plan.md"

    git -C "$tmproot" init -q 2>/dev/null
    # Stage everything so `git ls-files` reflects tracked-vs-ignored exactly as
    # it does in the real repo (an ignored path simply never gets added).
    git -C "$tmproot" add -A 2>/dev/null

    # (1) clean tree -> pass
    cat >"$tmproot/docs/good.md" <<'MD'
Owner runs the server with the real flag.
<!-- claim-evidence: src/config.rs#--whisper-base -->
<!-- claim-evidence: src/config.rs -->
MD
    git -C "$tmproot" add -A 2>/dev/null
    if ! run_verify "$tmproot" docs/good.md >/dev/null 2>&1; then
        echo "self-test FAILED: a clean tree should pass" >&2; rc=1
    fi

    # (2) a dead flag name reappears in tracked markdown -> drift (1)
    cat >"$tmproot/docs/dead.md" <<'MD'
Pass --asr-base to the server.
<!-- claim-evidence: src/config.rs -->
MD
    git -C "$tmproot" add -A 2>/dev/null
    out=0
    run_verify "$tmproot" docs/dead.md >/dev/null 2>&1 || out=$?
    if [ "$out" -ne 1 ]; then
        echo "self-test FAILED: a dead flag name should be drift (1), got $out" >&2; rc=1
    fi
    rm -f "$tmproot/docs/dead.md"
    git -C "$tmproot" add -A 2>/dev/null

    # (3) anchor target missing -> drift (1)
    printf '%s\n' '<!-- claim-evidence: src/gone.rs -->' >"$tmproot/docs/badfile.md"
    git -C "$tmproot" add -A 2>/dev/null
    out=0
    run_verify "$tmproot" docs/badfile.md >/dev/null 2>&1 || out=$?
    if [ "$out" -ne 1 ]; then
        echo "self-test FAILED: a missing anchor file should be drift (1), got $out" >&2; rc=1
    fi

    # (4) anchor token missing from an existing file -> drift (1)
    printf '%s\n' '<!-- claim-evidence: src/config.rs#AbsentToken -->' \
        >"$tmproot/docs/badtoken.md"
    git -C "$tmproot" add -A 2>/dev/null
    out=0
    run_verify "$tmproot" docs/badtoken.md >/dev/null 2>&1 || out=$?
    if [ "$out" -ne 1 ]; then
        echo "self-test FAILED: a missing anchor token should be drift (1), got $out" >&2; rc=1
    fi

    # (5) anchor pointing at a gitignore-local target -> authoring error (2)
    printf '%s\n' '<!-- claim-evidence: local/plan.md -->' \
        >"$tmproot/docs/localtarget.md"
    git -C "$tmproot" add -A 2>/dev/null
    out=0
    run_verify "$tmproot" docs/localtarget.md >/dev/null 2>&1 || out=$?
    if [ "$out" -ne 2 ]; then
        echo "self-test FAILED: a gitignored anchor target should be a setup error (2), got $out" >&2; rc=1
    fi

    # (6) malformed anchor (not a repo-relative path) -> drift (1)
    printf '%s\n' '<!-- claim-evidence: ... -->' >"$tmproot/docs/malformed.md"
    git -C "$tmproot" add -A 2>/dev/null
    out=0
    run_verify "$tmproot" docs/malformed.md >/dev/null 2>&1 || out=$?
    if [ "$out" -ne 1 ]; then
        echo "self-test FAILED: a non-path anchor token should be drift (1), got $out" >&2; rc=1
    fi

    # (7) doc with no anchors at all -> setup error (2)
    printf '# empty\n' >"$tmproot/docs/empty.md"
    git -C "$tmproot" add -A 2>/dev/null
    out=0
    run_verify "$tmproot" docs/empty.md >/dev/null 2>&1 || out=$?
    if [ "$out" -ne 2 ]; then
        echo "self-test FAILED: an anchor-less doc should be a setup error (2), got $out" >&2; rc=1
    fi

    # (8) missing doc entirely -> setup error (2)
    out=0
    run_verify "$tmproot" docs/nope.md >/dev/null 2>&1 || out=$?
    if [ "$out" -ne 2 ]; then
        echo "self-test FAILED: a missing doc should be a setup error (2), got $out" >&2; rc=1
    fi

    if [ "$rc" -eq 0 ]; then
        echo "check-owner-checklist-drift --self-test: OK (8 cases)"
    fi
    return "$rc"
}

# ------------------------------------------------------------------ main ---
mode="${1:-verify}"
case "$mode" in
    verify|"")
        echo "Vokra owner-checklist drift check (M4-RESIDUAL-A-T01, advisory)"
        for d in "${CLAIM_DOCS[@]}"; do
            echo "  doc    : $d"
        done
        rc=0
        run_verify "$ROOT" "${CLAIM_DOCS[@]}" || rc=$?
        exit "$rc"
        ;;
    --list)
        for d in "${CLAIM_DOCS[@]}"; do
            [ -f "$ROOT/$d" ] || continue
            extract_claims "$ROOT/$d" | sed "s|^|$d -> |"
        done
        ;;
    --self-test)
        rc=0
        self_test || rc=$?
        exit "$rc"
        ;;
    --help|-h)
        usage
        exit 0
        ;;
    *)
        echo "error: unknown argument '$mode'" >&2
        echo "usage: $0 [--list | --self-test | --help]" >&2
        exit 2
        ;;
esac
