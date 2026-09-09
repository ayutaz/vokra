#!/usr/bin/env bash
# Codex PreToolUse(Bash) policy dispatcher.
#
# The individual guards deliberately remain small and stdin-compatible with
# the Codex hook payload. Exit 2 denies the pending Bash tool call.

set -uo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0

if [ "${1:-}" = --self-test ]; then
    fails=0
    for guard in \
        block-cargo-add.sh \
        block-pip-conda.sh \
        guard-local-memory.sh \
        guard-local-models.sh \
        guard-protected-path.sh \
        guard-cloud-mutations.sh
    do
        if ! bash "$ROOT/.codex/hooks/$guard" --self-test; then
            fails=$((fails + 1))
        fi
    done

    check() {
        local name="$1" expected="$2" payload="$3" got
        if printf '%s' "$payload" | bash "$ROOT/.codex/hooks/pre_tool_use.sh" >/dev/null 2>&1; then
            got=allow
        else
            got=block
        fi
        if [ "$got" = "$expected" ]; then
            printf '  ok    %-56s %s\n' "$name" "$got"
        else
            printf '  FAIL  %-56s expected %s, got %s\n' "$name" "$expected" "$got"
            fails=$((fails + 1))
        fi
    }

    echo 'pre_tool_use dispatcher --self-test'
    check 'dispatcher blocks local model run' block \
        '{"tool_input":{"command":"vokra-cli run --model ./tiny.gguf"}}'
    check 'dispatcher blocks direct VAST mutation' block \
        '{"tool_input":{"command":"vastai destroy instance 123"}}'
    check 'dispatcher blocks dirty broad add' block \
        '{"tool_input":{"command":"git add ."}}'
    check 'dispatcher allows hash inspection' allow \
        '{"tool_input":{"command":"sha256sum ./tiny.gguf"}}'
    check 'dispatcher allows VAST status' allow \
        '{"tool_input":{"command":"scripts/publish/vast-ai/vastai-safe.sh show instance 123"}}'
    check 'dispatcher allows static gate' allow \
        '{"tool_input":{"command":"scripts/check-doc-references.sh --self-test"}}'

    if [ "$fails" -eq 0 ]; then
        echo 'pre_tool_use dispatcher --self-test: OK'
        exit 0
    fi
    echo "pre_tool_use dispatcher --self-test: FAIL ($fails)"
    exit 1
fi

payload="$(cat)"

for guard in \
    block-cargo-add.sh \
    block-pip-conda.sh \
    guard-local-memory.sh \
    guard-local-models.sh \
    guard-protected-path.sh \
    guard-cloud-mutations.sh
do
    if ! printf '%s' "$payload" | bash "$ROOT/.codex/hooks/$guard"; then
        exit 2
    fi
done

exit 0
