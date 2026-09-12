#!/usr/bin/env bash
# Offline contract gate for the Codex project hooks.
# It validates the complete hook schema, target files, shell/Python syntax and
# hook self-tests without invoking a model, Cargo, VAST, Scaleway or network.

set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
HOOKS="$ROOT/.codex/hooks"

# Keep successful output compact, but preserve the exact failing check and its
# captured diagnostics. This matters on hosted runners where a generic exit 1
# otherwise hides whether jq, syntax, a named self-test, or diff-check failed.
run_check() {
    local label="$1"
    shift
    local output rc
    if output="$("$@" 2>&1)"; then
        return 0
    else
        rc=$?
    fi
    echo "check-codex-hooks: FAIL: $label (exit $rc)" >&2
    if [ -n "$output" ]; then
        printf '%s\n' "$output" >&2
    else
        echo "check-codex-hooks: command produced no diagnostics" >&2
    fi
    return "$rc"
}

if [ "${1:-}" = --self-test ]; then
    self_test_output=""
    if self_test_output="$(run_check 'intentional failure' bash -c 'printf intentional-diagnostic >&2; exit 7' 2>&1)"; then
        echo 'check-codex-hooks --self-test: FAIL: run_check accepted an intentional failure' >&2
        exit 1
    else
        self_test_rc=$?
    fi
    if [ "$self_test_rc" -eq 0 ] \
        || ! printf '%s' "$self_test_output" | grep -q 'FAIL: intentional failure (exit 7)' \
        || ! printf '%s' "$self_test_output" | grep -q 'intentional-diagnostic'; then
        echo 'check-codex-hooks --self-test: FAIL: failure diagnostics were not preserved' >&2
        exit 1
    fi
    echo 'check-codex-hooks --self-test: OK'
    exit 0
fi

command -v jq >/dev/null 2>&1 || {
    echo 'check-codex-hooks: jq is required to parse .codex/hooks.json' >&2
    exit 1
}

run_check 'hooks object schema' \
    jq -e '.hooks and (.hooks | type == "object")' "$ROOT/.codex/hooks.json"
for event in PreToolUse PostToolUse UserPromptSubmit; do
    run_check "event array: $event" \
        jq -e --arg event "$event" '.hooks[$event] and (.hooks[$event] | type == "array")' \
        "$ROOT/.codex/hooks.json"
done
run_check 'tool event matcher schema' \
    jq -e '[.hooks.PreToolUse[], .hooks.PostToolUse[]]
        | all(.[]; (.matcher | type == "string" and length > 0))' \
    "$ROOT/.codex/hooks.json"
run_check 'UserPromptSubmit matcher omission' \
    jq -e '[.hooks.UserPromptSubmit[]] | all(.[]; has("matcher") | not)' \
    "$ROOT/.codex/hooks.json"
run_check 'command hook type/timeout schema' \
    jq -e '[.hooks[] | .[] | .hooks[]]
        | length > 0
        and all(.[]; .type == "command"
            and (.command | type == "string" and length > 0)
            and (.timeout | type == "number" and . >= 1 and . <= 120))' \
    "$ROOT/.codex/hooks.json"
run_check 'PreToolUse Bash dispatcher wiring' \
    jq -e '.hooks.PreToolUse[] | select(.matcher == "^Bash$") | .hooks[] | select(.command | contains("pre_tool_use.sh"))' \
    "$ROOT/.codex/hooks.json"
run_check 'protected-path tool wiring' \
    jq -e '.hooks.PreToolUse[] | select(.matcher == "^(apply_patch|Edit|Write)$") | .hooks[] | select(.command | contains("guard-protected-path.sh"))' \
    "$ROOT/.codex/hooks.json"
run_check 'PostToolUse formatter wiring' \
    jq -e '.hooks.PostToolUse[] | .hooks[] | select(.command | contains("post_tool_use.py"))' \
    "$ROOT/.codex/hooks.json"
run_check 'UserPromptSubmit secret guard wiring' \
    jq -e '.hooks.UserPromptSubmit[] | .hooks[] | select(.command | contains("prompt-secret-guard.py"))' \
    "$ROOT/.codex/hooks.json"

check_target_files() {
    local command target commands
    commands="$(jq -er '.hooks[] | .[] | .hooks[] | .command' "$ROOT/.codex/hooks.json")"
    while IFS= read -r command; do
        while IFS= read -r target; do
            [ -z "$target" ] && continue
            [ -f "$ROOT/$target" ] || {
                echo "missing hook command target: $target"
                return 1
            }
        done < <(printf '%s' "$command" | grep -oE '\.codex/hooks/[A-Za-z0-9._-]+' || true)
    done <<< "$commands"
}
run_check 'hook command target files' check_target_files

check_shell_syntax() {
    local file
    while IFS= read -r file; do
        bash -n "$file" || return $?
    done < <(find "$HOOKS" -maxdepth 1 -type f -name '*.sh' -print | sort)
}
run_check 'all hook shell syntax' check_shell_syntax
hook_count="$(find "$HOOKS" -maxdepth 1 -type f -name '*.sh' -print | wc -l | tr -d ' ')"

pycache_dir="$(mktemp -d "${TMPDIR:-/tmp}/vokra-hook-pycache.XXXXXX")"
trap 'rm -rf "$pycache_dir"' EXIT
run_check 'all hook Python compile' env \
    UV_CACHE_DIR="${TMPDIR:-/tmp}/vokra-uv-cache" \
    PYTHONPYCACHEPREFIX="$pycache_dir" \
    uv run --no-project --python 3.12 python -m py_compile \
    "$HOOKS"/*.py

for file in \
    block-cargo-add.sh \
    block-pip-conda.sh \
    guard-local-memory.sh \
    guard-local-models.sh \
    guard-protected-path.sh \
    guard-cloud-mutations.sh \
    pre_tool_use.sh
do
    run_check "shell self-test: $file" bash "$HOOKS/$file" --self-test
done

run_check 'Python self-test: post_tool_use.py' \
    env UV_CACHE_DIR="${TMPDIR:-/tmp}/vokra-uv-cache" \
    uv run --no-project --python 3.12 python "$HOOKS/post_tool_use.py" --self-test
run_check 'Python self-test: prompt-secret-guard.py' \
    env UV_CACHE_DIR="${TMPDIR:-/tmp}/vokra-uv-cache" \
    uv run --no-project --python 3.12 python "$HOOKS/prompt-secret-guard.py" --self-test

run_check 'working-tree diff check' \
    git -C "$ROOT" diff --check -- . ':(exclude)tools/parity/cosyvoice2_llm_reference/license_gate_manifest.json'
echo "Codex hook contract: PASS ($hook_count shell files, JSON schema/targets, syntax, self-tests, diff check)"
