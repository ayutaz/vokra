#!/usr/bin/env bash
# Codex PreToolUse(Bash) policy dispatcher.
#
# The individual guards deliberately remain small and stdin-compatible with
# the Codex hook payload. Exit 2 denies the pending Bash tool call.

set -uo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0
payload="$(cat)"

for guard in \
    block-cargo-add.sh \
    block-pip-conda.sh \
    guard-local-memory.sh
do
    if ! printf '%s' "$payload" | bash "$ROOT/.codex/hooks/$guard"; then
        exit 2
    fi
done

exit 0
