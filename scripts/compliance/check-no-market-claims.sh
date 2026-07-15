#!/usr/bin/env bash
# check-no-market-claims.sh — forbidden-words review gate for the CPU +
# Vulkan-only build target documents (M4-15-T08, ADR M4-15 §(e)).
#
# WHY: the 2026-07-14 scope decision (hard gate (f), scope-expansion
# §BIG-3) split FR-BE-09 in two — M4-15 ships the *build target* only,
# while the market positioning of the corresponding SKU stays in
# M5-08/M5-11. Putting positioning language into M4 documents would
# invite B2B compliance requirements into the M4 window, so every
# WP-owned document must stick to the neutral wording
# "CPU + Vulkan-only build target". This gate makes that mechanical
# (machine tier); T09 adds a human review pass on top (2-tier review).
#
# WHAT IT SCANS: the file / directory arguments it is given — the CI
# `build-target-vulkan-only` job passes the assembled artifact dir
# (NOTICE variant + SBOM), CHANGELOG.md, and the M4-15 scripts. Binary
# files are ignored (grep -I), so the cdylib inside the artifact dir
# cannot false-positive. The gate excludes ITSELF from the scan — this
# file necessarily embeds the ban list.
#
# WHEN M5-08/M5-11 legitimately introduce the positioning language, that
# WP must consciously re-scope this gate's invocation (e.g. stop passing
# CHANGELOG.md wholesale) instead of deleting terms from the list.
#
# Usage: bash scripts/compliance/check-no-market-claims.sh <file-or-dir>...
# Exit:  0 = clean, 1 = forbidden term found, 2 = usage error.

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "usage: $0 <file-or-dir>..." >&2
    exit 2
fi

SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

# --- collect files -----------------------------------------------------------
files=()
for arg in "$@"; do
    if [ -d "$arg" ]; then
        while IFS= read -r -d '' f; do
            files+=("$f")
        done < <(find "$arg" -type f -print0)
    elif [ -f "$arg" ]; then
        files+=("$arg")
    else
        echo "error: no such file or directory: $arg" >&2
        exit 2
    fi
done

# --- ban list ----------------------------------------------------------------
# English terms: extended regex, case-insensitive, whole-word (-w keeps
# "translate" / "island" from matching). Kept verbatim here on purpose —
# this file is the canonical list and is self-excluded from the scan.
EN_CLAIMS='critical[- ]safe|safety[- ]critical|mission[- ]critical|medical|automotive|military|ISO[ ]?26262|IEC[ ]?62304'
# Uppercase acronym: case-SENSITIVE whole word (the lowercase letter
# sequence appears in ordinary words).
ACRONYM='SLA'
# Japanese terms: fixed-string match (no word boundaries in Japanese).
JA_CLAIMS=(医療 車載 軍事)

fail=0
scanned=0

for f in "${files[@]}"; do
    # self-exclusion (the ban list lives here).
    abs="$(cd "$(dirname "$f")" && pwd)/$(basename "$f")"
    [ "$abs" = "$SELF" ] && continue
    scanned=$((scanned + 1))

    hits=""
    h="$(grep -nIiEw "$EN_CLAIMS" "$f" 2>/dev/null || true)"
    [ -n "$h" ] && hits="$hits$h"$'\n'
    h="$(grep -nIw "$ACRONYM" "$f" 2>/dev/null || true)"
    [ -n "$h" ] && hits="$hits$h"$'\n'
    for term in "${JA_CLAIMS[@]}"; do
        h="$(grep -nIF "$term" "$f" 2>/dev/null || true)"
        [ -n "$h" ] && hits="$hits$h"$'\n'
    done

    if [ -n "$hits" ]; then
        echo "FAIL: market-claim term(s) in $f:" >&2
        printf '%s' "$hits" | sed 's/^/  /' >&2
        fail=1
    fi
done

if [ "$fail" -ne 0 ]; then
    echo "" >&2
    echo "M4-15 documents must use the neutral wording 'CPU + Vulkan-only" >&2
    echo "build target' — market positioning language is deferred to" >&2
    echo "M5-08/M5-11 (ADR M4-15 §(e), 2026-07-14 hard gate (f))." >&2
    exit 1
fi

echo "OK: no market-claim terms in $scanned scanned file(s)"
