#!/usr/bin/env bash
# check-zero-deps.sh — enforce the zero-external-dependency invariant (NFR-DS-02).
#
# Vokra's dependency graph must contain ONLY first-party `vokra-*` crates. No
# third-party crates.io dependency is linked into the runtime, the C ABI, the
# models, or (for M0) the offline `vokra-convert` tool. This is STRICTER than
# `cargo deny` (which merely allows a permitted-license dependency): here the
# allowed number of external crates is exactly ZERO.
#
# Mechanism: scan the resolved lockfile (Cargo.lock — the full transitive
# graph) and fail if any `[[package]]` name does not start with `vokra`.
#
# CI wiring is owned by M0-01 (proposed: alongside the license check). This
# script is also run by the pre-commit hook (.githooks/pre-commit).
# Exit code: 0 = clean, 1 = an external crate is present.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCK="$ROOT/Cargo.lock"

if [ ! -f "$LOCK" ]; then
    echo "error: Cargo.lock not found at $LOCK" >&2
    echo "hint: run 'cargo generate-lockfile' first (Cargo.lock is committed)." >&2
    exit 1
fi

# Every resolved package name that is not a first-party vokra-* crate.
foreign="$(
    grep -E '^name = "' "$LOCK" \
        | sed -E 's/^name = "(.*)"$/\1/' \
        | grep -vE '^vokra' \
        || true
)"

if [ -n "$foreign" ]; then
    echo "error: external (non-vokra) crate(s) found in Cargo.lock" >&2
    echo "       NFR-DS-02: the workspace must have ZERO external dependencies." >&2
    printf '  - %s\n' $foreign >&2
    echo "Vokra links only first-party vokra-* crates. Remove the dependency, or —" >&2
    echo "if it is genuinely unavoidable — escalate it as a design-red-line decision" >&2
    echo "(CONTRIBUTING.md §3 / §5, CLAUDE.md). ONNX/protobuf families are also banned" >&2
    echo "outright in deny.toml." >&2
    exit 1
fi

echo "check-zero-deps: OK (Cargo.lock contains only vokra-* crates)"
