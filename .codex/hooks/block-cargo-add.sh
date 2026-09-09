#!/usr/bin/env bash
# Codex PreToolUse hook (Bash): block commands that would add an external
# dependency, which Vokra forbids (NFR-DS-02 — the workspace links only
# first-party vokra-* path crates; the ONNX/protobuf families are additionally
# banned in deny.toml). Currently blocks `cargo add`.
#
# Exit 2 = block the tool call and return the message to Codex.

set -uo pipefail

if [ "${1:-}" = --self-test ]; then
    fails=0
    json_payload() {
        local command="$1"
        if command -v jq >/dev/null 2>&1; then
            jq -cn --arg command "$command" '{tool_input:{command:$command}}'
        elif command -v uv >/dev/null 2>&1; then
            printf '%s' "$command" \
                | UV_CACHE_DIR="${TMPDIR:-/tmp}/vokra-uv-cache" \
                  uv run --no-project --python 3.12 python -c \
                  'import json,sys; print(json.dumps({"tool_input":{"command":sys.stdin.read()}}))'
        else
            return 1
        fi
    }
    check() {
        local name="$1" expected="$2" command="$3" rc got
        json_payload "$command" \
            | bash "$0" >/dev/null 2>&1
        rc=$?
        case "$rc" in
            0) got=allow ;;
            2) got=block ;;
            *) got="error($rc)" ;;
        esac
        if [ "$got" = "$expected" ]; then
            printf '  ok    %-46s %s\n' "$name" "$got"
        else
            printf '  FAIL  %-46s expected %s, got %s\n' "$name" "$expected" "$got"
            fails=$((fails + 1))
        fi
    }
    echo 'block-cargo-add --self-test'
    check 'direct cargo add' block 'cargo add serde'
    check 'chained cargo add' block 'git status --short && cargo add serde'
    check 'cargo add after assignment' block 'CARGO_TERM_COLOR=never cargo add serde'
    check 'cargo subcommand prose' allow 'echo "cargo add is forbidden"'
    check 'cargo build' allow 'cargo build -p vokra-cli'
    check 'path-like cargo add prose' allow 'echo scripts/cargo add helper'
    if [ "$fails" -eq 0 ]; then
        echo 'block-cargo-add --self-test: OK'
        exit 0
    fi
    echo "block-cargo-add --self-test: FAIL ($fails)"
    exit 1
fi

payload="$(cat)"

cmd=""
if command -v jq >/dev/null 2>&1; then
    cmd="$(printf '%s' "$payload" | jq -r '.tool_input.command // empty' 2>/dev/null || true)"
elif command -v uv >/dev/null 2>&1; then
    cmd="$(printf '%s' "$payload" \
        | UV_CACHE_DIR="${TMPDIR:-/tmp}/vokra-uv-cache" uv run --no-project --python 3.12 python -c 'import sys,json; print(json.load(sys.stdin).get("tool_input",{}).get("command",""))' \
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
        echo "design-red-line decision (CONTRIBUTING.md §3 / §5, AGENTS.md)."
    } >&2
    exit 2
fi
exit 0
