#!/usr/bin/env bash
# check-platform-support.sh — M4-11-T08 platform-support matrix drift tripwire.
#
# WHAT IT GATES
#   docs/platform-support/v1.0-rc-support-matrix.md cites evidence anchors —
#   CI workflow jobs, build scripts, and owner benchmark docs — as the proof
#   that each of the 6 platforms (Windows / macOS / Linux / Android / iOS /
#   Web) is officially supported (NFR-PT-01 / NFR-MT-02 / NFR-DS-03, M4-11).
#   Over time a workflow job can be renamed or a script deleted, silently
#   rotting the matrix's evidence. This script re-reads every machine-readable
#   anchor line in the matrix and fails if the anchor's target no longer
#   exists — so the confirmation record can never drift into a fabricated
#   pass without CI noticing.
#
# ANCHOR SYNTAX (in the matrix markdown)
#   <!-- anchor: <path> -->            file must exist
#   <!-- anchor: <path>#<job> -->      file must exist AND, for a YAML
#                                      workflow, a top-level job id `<job>`
#                                      (2-space indent under `jobs:`) must
#                                      exist; for a non-YAML file, the literal
#                                      token <job> must appear.
#   The check is anchor-driven — it never guesses from prose (no false
#   positive/negative from a natural-language grep).
#
# MODES
#   scripts/check-platform-support.sh              verify (default)
#   scripts/check-platform-support.sh --list       print resolved anchors
#   scripts/check-platform-support.sh --self-test  unit-test the parser
#   scripts/check-platform-support.sh --help       this text
#
# ZERO-DEP (NFR-DS-02)
#   Pure bash + grep + sed. No Rust toolchain, no crate, no external binary
#   beyond coreutils. Same family as scripts/check-zero-deps.sh and
#   scripts/check-abi-changelog.sh.
#
# CI WIRING
#   Advisory only (M4-11-T08). Runs as a continue-on-error step in the
#   `license` job of .github/workflows/ci.yml. Promotion to a required
#   branch-protection check is an owner decision (NFR-MT-07), mirroring the
#   gpu-vulkan-parity.yml cool-off posture.
#
# EXIT CODES
#   0  all anchors resolve (or --list / --self-test / --help success)
#   1  one or more anchors point at a missing file / job / token (drift)
#   2  usage / setup error (matrix doc missing, no anchors, bad flag)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MATRIX="$ROOT/docs/platform-support/v1.0-rc-support-matrix.md"

usage() {
    sed -n '3,45p' "$0" | sed 's/^# \{0,1\}//'
}

# ---------------------------------------------------------------- extract ---
# extract_anchors <doc-path> — emit one anchor token per line, sorted-unique.
# Always returns 0 (an empty result is handled by the caller) so `set -e`
# does not abort on a doc that legitimately has no anchors yet.
extract_anchors() {
    local doc="$1"
    grep -oE '<!-- anchor:[^>]*-->' "$doc" 2>/dev/null \
        | sed -E 's/^<!-- anchor:[[:space:]]*//; s/[[:space:]]*-->$//' \
        | sed -E 's/[[:space:]]+$//' \
        | LC_ALL=C sort -u \
        || true
}

# --------------------------------------------------------------- resolve ---
# check_anchor <root> <token> — 0 if the anchor target exists, else 1.
# Prints a one-line reason to stderr on failure.
check_anchor() {
    local root="$1" token="$2"
    local path job full
    if [[ "$token" == *"#"* ]]; then
        path="${token%%#*}"
        job="${token#*#}"
    else
        path="$token"
        job=""
    fi
    # A real anchor is always a repo-relative path (contains a '/'). A token
    # without one is a stray documentation placeholder or typo, NOT a valid
    # anchor — fail loudly with a distinct message rather than a confusing
    # "missing file". (The matrix prose must not embed example anchors.)
    case "$path" in
        */*) : ;;
        *)
            echo "  MALFORMED    : $token  (anchor token is not a repo-relative path)" >&2
            return 1
            ;;
    esac
    full="$root/$path"
    if [ ! -f "$full" ]; then
        echo "  MISSING FILE : $token  ($path not found)" >&2
        return 1
    fi
    if [ -n "$job" ]; then
        case "$path" in
            *.yml|*.yaml)
                # GitHub Actions job id: a key at exactly 2-space indent
                # under `jobs:`. Anchors only ever name real job ids, so this
                # is precise (no prose grep).
                if ! grep -qE "^[[:space:]]{2}${job}:([[:space:]]|$)" "$full"; then
                    echo "  MISSING JOB  : $token  (no '^  ${job}:' in $path)" >&2
                    return 1
                fi
                ;;
            *)
                if ! grep -qF -- "$job" "$full"; then
                    echo "  MISSING TOKEN: $token  ('$job' not in $path)" >&2
                    return 1
                fi
                ;;
        esac
    fi
    return 0
}

# ----------------------------------------------------------------- verify ---
# run_verify <root> <doc> — check every anchor in <doc> against <root>.
# Returns 0 (all resolve), 1 (>=1 missing), or 2 (setup error). Never calls
# `exit`, so callers (self-test) can invoke it repeatedly.
run_verify() {
    local root="$1" doc="$2"
    if [ ! -f "$doc" ]; then
        echo "error: matrix doc not found: $doc" >&2
        return 2
    fi
    local anchors
    anchors="$(extract_anchors "$doc")"
    if [ -z "$anchors" ]; then
        echo "error: no <!-- anchor: ... --> lines found in $doc" >&2
        return 2
    fi
    local total=0 bad=0 token
    while IFS= read -r token; do
        [ -n "$token" ] || continue
        total=$((total + 1))
        if ! check_anchor "$root" "$token"; then
            bad=$((bad + 1))
        fi
    done <<EOF
$anchors
EOF
    if [ "$bad" -ne 0 ]; then
        echo "check-platform-support: FAIL — $bad of $total anchor(s) missing (matrix drift)" >&2
        return 1
    fi
    echo "check-platform-support: OK ($total anchors resolve)"
    return 0
}

# -------------------------------------------------------------- self-test ---
# self_test — synthesize a throwaway root + matrix docs and assert the
# parser passes a good doc and fails on a missing file / missing job /
# missing token. Catches parser drift without touching the real matrix.
self_test() {
    local tmproot rc=0
    tmproot="$(mktemp -d -t vokra-plat-check.XXXXXX)"
    trap 'rm -rf "$tmproot"' RETURN

    mkdir -p "$tmproot/.github/workflows" "$tmproot/scripts" "$tmproot/docs"
    cat >"$tmproot/.github/workflows/fake.yml" <<'YML'
name: fake
on: [push]
jobs:
  foo:
    runs-on: ubuntu-latest
    steps:
      - run: echo hi
YML
    printf '#!/bin/sh\necho hi\n' >"$tmproot/scripts/fake.sh"
    printf '# fake benchmark doc\n' >"$tmproot/docs/fake.md"

    # (1) all-good doc -> pass
    cat >"$tmproot/docs/good.md" <<'MD'
<!-- anchor: .github/workflows/fake.yml#foo -->
<!-- anchor: scripts/fake.sh -->
<!-- anchor: docs/fake.md -->
MD
    if ! run_verify "$tmproot" "$tmproot/docs/good.md" >/dev/null 2>&1; then
        echo "self-test FAILED: a valid doc should pass" >&2; rc=1
    fi

    # (2) missing file -> fail
    printf '%s\n' '<!-- anchor: .github/workflows/gone.yml#foo -->' \
        >"$tmproot/docs/badfile.md"
    if run_verify "$tmproot" "$tmproot/docs/badfile.md" >/dev/null 2>&1; then
        echo "self-test FAILED: a missing file should fail" >&2; rc=1
    fi

    # (3) missing job -> fail
    printf '%s\n' '<!-- anchor: .github/workflows/fake.yml#nope -->' \
        >"$tmproot/docs/badjob.md"
    if run_verify "$tmproot" "$tmproot/docs/badjob.md" >/dev/null 2>&1; then
        echo "self-test FAILED: a missing job id should fail" >&2; rc=1
    fi

    # (4) missing literal token in a non-YAML file -> fail
    printf '%s\n' '<!-- anchor: scripts/fake.sh#AbsentToken -->' \
        >"$tmproot/docs/badtoken.md"
    if run_verify "$tmproot" "$tmproot/docs/badtoken.md" >/dev/null 2>&1; then
        echo "self-test FAILED: a missing token should fail" >&2; rc=1
    fi

    # (5) malformed anchor (not a repo-relative path) -> fail
    printf '%s\n' '<!-- anchor: ... -->' >"$tmproot/docs/malformed.md"
    if run_verify "$tmproot" "$tmproot/docs/malformed.md" >/dev/null 2>&1; then
        echo "self-test FAILED: a non-path anchor token should fail" >&2; rc=1
    fi

    # (6) doc with no anchors at all -> setup error (rc 2)
    printf '# empty\n' >"$tmproot/docs/empty.md"
    run_verify "$tmproot" "$tmproot/docs/empty.md" >/dev/null 2>&1
    if [ "$?" -ne 2 ]; then
        echo "self-test FAILED: a doc with no anchors should be a setup error (2)" >&2; rc=1
    fi

    if [ "$rc" -eq 0 ]; then
        echo "check-platform-support --self-test: OK"
    fi
    return "$rc"
}

# ------------------------------------------------------------------ main ---
mode="${1:-verify}"
case "$mode" in
    verify|"")
        echo "Vokra platform-support matrix drift check (M4-11-T08, advisory)"
        echo "  matrix : $MATRIX"
        run_verify "$ROOT" "$MATRIX"
        exit $?
        ;;
    --list)
        extract_anchors "$MATRIX"
        ;;
    --self-test)
        set +e
        self_test
        exit $?
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
