#!/usr/bin/env bash
# Offline contract gate for the Codex project hooks.
# It validates the complete hook schema, target files, shell/Python syntax and
# hook self-tests without invoking a model, Cargo, VAST, Scaleway or network.

set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
HOOKS="$ROOT/.codex/hooks"

command -v jq >/dev/null 2>&1 || {
    echo 'check-codex-hooks: jq is required to parse .codex/hooks.json' >&2
    exit 1
}

jq -e '.hooks and (.hooks | type == "object")' "$ROOT/.codex/hooks.json" >/dev/null
for event in PreToolUse PostToolUse UserPromptSubmit; do
    jq -e --arg event "$event" '.hooks[$event] and (.hooks[$event] | type == "array")' \
        "$ROOT/.codex/hooks.json" >/dev/null
done
jq -e '[.hooks.PreToolUse[], .hooks.PostToolUse[]]
    | all(.[]; (.matcher | type == "string" and length > 0))' \
    "$ROOT/.codex/hooks.json" >/dev/null
jq -e '[.hooks.UserPromptSubmit[]] | all(.[]; has("matcher") | not)' \
    "$ROOT/.codex/hooks.json" >/dev/null
jq -e '[.hooks[] | .[] | .hooks[]]
    | length > 0
    and all(.[]; .type == "command"
        and (.command | type == "string" and length > 0)
        and (.timeout | type == "number" and . >= 1 and . <= 120))' \
    "$ROOT/.codex/hooks.json" >/dev/null
jq -e '.hooks.PreToolUse[] | select(.matcher == "^Bash$") | .hooks[] | select(.command | contains("pre_tool_use.sh"))' \
    "$ROOT/.codex/hooks.json" >/dev/null
jq -e '.hooks.PreToolUse[] | select(.matcher == "^(apply_patch|Edit|Write)$") | .hooks[] | select(.command | contains("guard-protected-path.sh"))' \
    "$ROOT/.codex/hooks.json" >/dev/null
jq -e '.hooks.PostToolUse[] | .hooks[] | select(.command | contains("post_tool_use.py"))' \
    "$ROOT/.codex/hooks.json" >/dev/null
jq -e '.hooks.UserPromptSubmit[] | .hooks[] | select(.command | contains("prompt-secret-guard.py"))' \
    "$ROOT/.codex/hooks.json" >/dev/null

# Every path embedded in a hook command must resolve inside this checkout.
while IFS= read -r command; do
    while IFS= read -r target; do
        [ -z "$target" ] && continue
        [ -f "$ROOT/$target" ] || {
            echo "check-codex-hooks: missing hook command target: $target" >&2
            exit 1
        }
    done < <(printf '%s' "$command" | grep -oE '\.codex/hooks/[A-Za-z0-9._-]+' || true)
done < <(jq -r '.hooks[] | .[] | .hooks[] | .command' "$ROOT/.codex/hooks.json")

hook_count=0
while IFS= read -r file; do
    bash -n "$file"
    hook_count=$((hook_count + 1))
done < <(find "$HOOKS" -maxdepth 1 -type f -name '*.sh' -print | sort)

pycache_dir="$(mktemp -d "${TMPDIR:-/tmp}/vokra-hook-pycache.XXXXXX")"
trap 'rm -rf "$pycache_dir"' EXIT
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
    bash "$HOOKS/$file" --self-test >/dev/null
done

UV_CACHE_DIR="${TMPDIR:-/tmp}/vokra-uv-cache" \
    uv run --no-project --python 3.12 python "$HOOKS/post_tool_use.py" --self-test >/dev/null
UV_CACHE_DIR="${TMPDIR:-/tmp}/vokra-uv-cache" \
    uv run --no-project --python 3.12 python "$HOOKS/prompt-secret-guard.py" --self-test >/dev/null

git -C "$ROOT" diff --check -- . ':(exclude)tools/parity/cosyvoice2_llm_reference/license_gate_manifest.json'
echo "Codex hook contract: PASS ($hook_count shell files, JSON schema/targets, syntax, self-tests, diff check)"
