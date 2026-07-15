#!/usr/bin/env bash
# check-fa-v3-confinement.sh — M4-07-T15: the FA v3 containment red-line,
# machine-checked.
#
# Basis: design constraint §5-(7) ("FA v3 pushed out to v1.5+") unlocked at
# M4-07 ONLY inside `crates/vokra-backend-cuda` (milestones.md §8 invariant
# row: "FA v3 前倒し禁止（設計制約 §5-(7)、M4-07 内スコープに閉じる)").
# This script SUPERSEDES the M3-01 ADR §2-(b)-(iii) "no FA v3 anywhere" grep
# gate (which necessarily went non-empty when M4-07 landed): the invariant is
# now
#
#   FA v3 symbols (flash_attn_v3 / WGMMA / wgmma / TMA_ / compute_90a)
#   appear ONLY under crates/vokra-backend-cuda; every other crate in the
#   workspace stays clean.
#
# Doc-comment mentions of the prohibition itself (e.g.
# vokra-backend-vulkan/src/lib.rs "no Hopper WGMMA/TMA equivalent") must NOT
# trip the gate, so comment-only lines (`//`, `///`, `//!`, `/*`, `*`) are
# excluded — the same discipline check-forbidden-symbols.sh uses: real code
# cannot live on a comment-only line.
#
# CI wiring: advisory job in ci.yml (fa-v3-confinement). Promotion to a
# branch-protection required check is the owner's call after sustained green
# (the standard workflow-promotion discipline).
#
# Env override CRATES_DIR is for the fixture tests
# (crates/vokra-backend-cuda/tests/fa_v3_confinement.rs), which build scratch
# trees with deliberate violations and assert red/green.
#
# Exit code: 0 = confined, 1 = leak found (each offending line printed).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CRATES_DIR="${CRATES_DIR:-$ROOT/crates}"
ALLOWED_SUBTREE="vokra-backend-cuda"

if [ ! -d "$CRATES_DIR" ]; then
    echo "error: scan directory not found: $CRATES_DIR" >&2
    exit 1
fi

# Case-sensitive on purpose: `WGMMA` (doc shorthand) and `wgmma` (PTX
# mnemonic) are both listed; `TMA_` catches the descriptor/API constant
# family (CU_TENSOR_MAP is matched via the dedicated pattern below).
PATTERN='flash_attn_v3|WGMMA|wgmma|TMA_|compute_90a|cuTensorMapEncodeTiled'

# grep -E over .rs files outside the allowed subtree, then drop comment-only
# lines (leading whitespace + //, ///, //!, /* or a block-comment
# continuation *). grep exits 1 on "no match", which is our success case —
# hence the `|| true` capture.
violations="$(
    grep -rnE --include='*.rs' "$PATTERN" "$CRATES_DIR" 2>/dev/null |
        grep -v "/$ALLOWED_SUBTREE/" |
        grep -vE '^[^:]*:[0-9]+:[[:space:]]*(//|/\*|\*)' || true
)"

if [ -n "$violations" ]; then
    echo "FA v3 confinement violation (design constraint §5-(7) / milestones §8 red-line):" >&2
    echo "FA v3 symbols are legal ONLY under crates/$ALLOWED_SUBTREE — found outside it:" >&2
    printf '%s\n' "$violations" >&2
    exit 1
fi

echo "OK: FA v3 symbols confined to crates/$ALLOWED_SUBTREE (pattern: $PATTERN)"
