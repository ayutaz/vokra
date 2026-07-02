#!/usr/bin/env bash
# check-forbidden-symbols.sh — guard against locale-dependent C parsing APIs.
#
# Ticket: M0-02-T05. Basis: NFR-RL-01 (and CLAUDE.md "LC_NUMERIC trap"):
# `strtod` must never be used — under European comma-decimal locales
# (LC_NUMERIC) it misparses/crashes. Numeric parsing policy for the whole
# workspace: string-to-number conversion MUST use Rust's locale-independent
# `str::parse`; C's `strtod` (or any locale-sensitive parser) must not be
# introduced, including via future C glue code or FFI declarations.
# (The policy is also recorded in the vokra-core crate-level rustdoc.)
#
# Scope: scans source files under crates/ and the C ABI smoke tests under
# tests/capi/ (M0-09-T01: the C ABI boundary must not introduce strtod either)
# (*.rs, *.c, *.h, *.cc, *.cpp, *.hpp, *.m, *.mm). Comment-only lines (`//`,
# `///`, `//!`, `/*`, `*`) are excluded so documentation may mention the
# forbidden symbol by name; any use in actual code cannot live on a
# comment-only line and is caught.
#
# Extending the symbol list to other locale-dependent APIs (strtof/strtold,
# setlocale-sensitive printf/scanf families, ...) is a spike-time decision
# (M0-02-T05); add entries to FORBIDDEN_SYMBOLS below.
#
# CI wiring is owned by M0-01 (proposed: run in the same job as the license
# check). Exit code: 0 = clean, 1 = forbidden symbol found.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# bash 3.2 (macOS default) compatible.
FORBIDDEN_SYMBOLS=(strtod)

# Scan the Rust crates plus the C ABI smoke tests (M0-09-T01). tests/capi is
# added only when present so the check still runs in a crates-only checkout.
SCAN_DIRS=("$ROOT/crates")
[ -d "$ROOT/tests/capi" ] && SCAN_DIRS+=("$ROOT/tests/capi")

if [ ! -d "$ROOT/crates" ]; then
    echo "error: scan directory not found: $ROOT/crates" >&2
    exit 1
fi

status=0
for sym in "${FORBIDDEN_SYMBOLS[@]}"; do
    # Word-boundary emulation portable across BSD/GNU grep.
    pattern="(^|[^A-Za-z0-9_])${sym}([^A-Za-z0-9_]|\$)"
    matches="$(
        grep -RInE \
            --include='*.rs' --include='*.c' --include='*.h' \
            --include='*.cc' --include='*.cpp' --include='*.hpp' \
            --include='*.m' --include='*.mm' \
            -e "$pattern" "${SCAN_DIRS[@]}" 2>/dev/null \
            | grep -vE '^[^:]+:[0-9]+:[[:space:]]*(//|/\*|\*)' || true
    )"
    if [ -n "$matches" ]; then
        echo "error: forbidden symbol '${sym}' found (NFR-RL-01 — use Rust str::parse instead):" >&2
        printf '%s\n' "$matches" >&2
        status=1
    fi
done

if [ "$status" -eq 0 ]; then
    echo "check-forbidden-symbols: OK (no forbidden symbols under crates/ or tests/capi/)"
fi
exit "$status"
