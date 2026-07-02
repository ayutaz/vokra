#!/usr/bin/env bash
# Claude Code PostToolUse hook: after a Cargo.toml / Cargo.lock edit, re-assert
# the zero-external-dependency invariant (NFR-DS-02) and alert Claude (exit 2)
# if the resolved lockfile now contains a non-vokra crate.
#
# This is an EARLY WARNING only. Hard enforcement lives in the pre-commit hook
# and the CI `license` job (cargo-deny bans + check-zero-deps). A Cargo.toml
# edit that has not yet been resolved into Cargo.lock is caught later by those
# gates, not here.

set -uo pipefail

payload="$(cat)"

file=""
if command -v jq >/dev/null 2>&1; then
    file="$(printf '%s' "$payload" | jq -r '.tool_input.file_path // empty' 2>/dev/null || true)"
elif command -v python3 >/dev/null 2>&1; then
    file="$(printf '%s' "$payload" \
        | python3 -c 'import sys,json; print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))' \
        2>/dev/null || true)"
fi

case "$file" in
    Cargo.toml|*/Cargo.toml|Cargo.lock|*/Cargo.lock) ;;
    *) exit 0 ;;
esac

ROOT="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
[ -f "$ROOT/scripts/check-zero-deps.sh" ] || exit 0

if ! out="$(bash "$ROOT/scripts/check-zero-deps.sh" 2>&1)"; then
    {
        echo "vokra zero-dependency invariant tripped after editing $file:"
        echo "$out"
        echo "Vokra links only first-party vokra-* crates (NFR-DS-02). If a dependency"
        echo "was added, remove it or escalate it as a design-red-line decision."
    } >&2
    exit 2
fi
exit 0
