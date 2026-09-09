#!/usr/bin/env bash
# Offline contract gate for the Codex project hooks.
# It validates wiring, shell syntax and hook self-tests without invoking a
# model, Cargo, VAST, Scaleway or any network service.

set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
HOOKS="$ROOT/.codex/hooks"

command -v jq >/dev/null 2>&1 || {
    echo 'check-codex-hooks: jq is required to parse .codex/hooks.json' >&2
    exit 1
}

jq -e '.hooks.PreToolUse and (.hooks.PreToolUse | type == "array")' \
    "$ROOT/.codex/hooks.json" >/dev/null
jq -e '.hooks.PreToolUse[] | select(.matcher == "^Bash$") | .hooks[] | select(.command | contains("pre_tool_use.sh"))' \
    "$ROOT/.codex/hooks.json" >/dev/null
jq -e '.hooks.PreToolUse[] | select(.matcher == "^(apply_patch|Edit|Write)$") | .hooks[] | select(.command | contains("guard-protected-path.sh"))' \
    "$ROOT/.codex/hooks.json" >/dev/null

hook_count=0
while IFS= read -r file; do
    bash -n "$file"
    hook_count=$((hook_count + 1))
done < <(find "$HOOKS" -maxdepth 1 -type f -name '*.sh' -print | sort)

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

git -C "$ROOT" diff --check -- . ':(exclude)tools/parity/cosyvoice2_llm_reference/license_gate_manifest.json'
echo "Codex hook contract: PASS ($hook_count shell files, JSON wiring, self-tests, diff check)"
