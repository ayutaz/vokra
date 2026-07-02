#!/usr/bin/env bash
# Claude Code PreToolUse hook (Bash): block commands that would add an external
# dependency, which Vokra forbids (NFR-DS-02 — the workspace links only
# first-party vokra-* path crates; the ONNX/protobuf families are additionally
# banned in deny.toml). Currently blocks `cargo add`.
#
# Exit 2 = block the tool call and return the message to Claude.

set -uo pipefail

payload="$(cat)"

cmd=""
if command -v jq >/dev/null 2>&1; then
    cmd="$(printf '%s' "$payload" | jq -r '.tool_input.command // empty' 2>/dev/null || true)"
elif command -v python3 >/dev/null 2>&1; then
    cmd="$(printf '%s' "$payload" \
        | python3 -c 'import sys,json; print(json.load(sys.stdin).get("tool_input",{}).get("command",""))' \
        2>/dev/null || true)"
fi

# Match `cargo add` as a command word (line start, or after ; & | or whitespace),
# so a path or a comment mention of the string does not trip it.
if printf '%s' "$cmd" | grep -Eq '(^|[;&|[:space:]])cargo[[:space:]]+add([[:space:]]|$)'; then
    {
        echo "Blocked: 'cargo add' introduces an external crate dependency, which Vokra"
        echo "forbids (NFR-DS-02 — the workspace links only first-party vokra-* path"
        echo "crates; ONNX/protobuf families are additionally banned in deny.toml)."
        echo "Implement the capability in std / first-party code, or escalate it as a"
        echo "design-red-line decision (CONTRIBUTING.md §3 / §5, CLAUDE.md)."
    } >&2
    exit 2
fi
exit 0
