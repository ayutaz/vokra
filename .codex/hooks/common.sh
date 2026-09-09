#!/usr/bin/env bash
# Shared, deliberately small helpers for Codex PreToolUse hooks.
#
# This file never reads model contents.  Its JSON fallback is routed through
# uv because repository Python is uv-managed (AGENTS.md).

set -uo pipefail

hook_json_value() {
    local payload="$1" filter="$2"
    if command -v jq >/dev/null 2>&1; then
        printf '%s' "$payload" | jq -r "$filter" 2>/dev/null || true
    elif command -v uv >/dev/null 2>&1; then
        local field
        case "$filter" in
            '.tool_input.command // empty') field="command" ;;
            '.tool_input.file_path // .tool_input.path // empty') field="file_path" ;;
            '.cwd // empty') field="cwd" ;;
            *) return 0 ;;
        esac
        printf '%s' "$payload" \
            | UV_CACHE_DIR="${TMPDIR:-/tmp}/vokra-uv-cache" \
              uv run --no-project --python 3.12 python -c \
              'import json,sys; d=json.load(sys.stdin); k=sys.argv[1]; t=d.get("tool_input",{}); v=t.get("command", "") if k == "command" else ((t.get("file_path") or t.get("path") or "") if k == "file_path" else d.get(k, "")); print(v if isinstance(v,str) else "")' "$field" \
              2>/dev/null || true
    fi
}

hook_command() {
    hook_json_value "$1" '.tool_input.command // empty'
}

hook_file_path() {
    hook_json_value "$1" '.tool_input.file_path // .tool_input.path // empty'
}

# Emit one approximate shell segment per line.  Hooks intentionally do not
# evaluate shell syntax; splitting separators is enough to avoid an `echo`
# mention in one segment affecting a real command in another.
hook_segments() {
    printf '%s\n' "$1" | tr ';|&' '\n\n\n\n'
}

hook_trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

# Remove harmless command wrappers while retaining the command's original
# text for policy checks.  This is intentionally not a shell interpreter.
hook_normalize_segment() {
    local segment
    segment="$(hook_trim "$1")"
    while printf '%s' "$segment" | grep -Eq '^[A-Za-z_][A-Za-z0-9_]*=[^[:space:]]+[[:space:]]+'; do
        segment="${segment#* }"
        segment="$(hook_trim "$segment")"
    done
    while printf '%s' "$segment" | grep -Eq '^(command|exec|time|sudo)[[:space:]]+'; do
        segment="${segment#* }"
        segment="$(hook_trim "$segment")"
    done
    if printf '%s' "$segment" | grep -Eq '^env[[:space:]]+'; then
        segment="${segment#* }"
        segment="$(hook_trim "$segment")"
        while printf '%s' "$segment" \
            | grep -Eq '^(-[^[:space:]]+|[A-Za-z_][A-Za-z0-9_]*=[^[:space:]]+)[[:space:]]+'; do
            segment="${segment#* }"
            segment="$(hook_trim "$segment")"
        done
    fi
    printf '%s' "$segment"
}
