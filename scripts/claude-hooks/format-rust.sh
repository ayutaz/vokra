#!/usr/bin/env bash
# Claude Code PostToolUse hook: keep Rust edits formatted, matching the CI
# `fmt` required check. A formatter must never interrupt work, so this ALWAYS
# exits 0 (it is best-effort).
#
# Reads the hook JSON on stdin, extracts tool_input.file_path, and runs
# rustfmt on it when it is a *.rs file. Edition 2024 matches the workspace
# (Cargo.toml); rustfmt.toml (defaults) is discovered from the file upward.

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

[ -n "$file" ] || exit 0
case "$file" in
    *.rs) ;;
    *) exit 0 ;;
esac
[ -f "$file" ] || exit 0
command -v rustfmt >/dev/null 2>&1 || exit 0

rustfmt --edition 2024 "$file" >/dev/null 2>&1 || true
exit 0
