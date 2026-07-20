#!/usr/bin/env bash
# check-m5-residual-no-abi.sh — M5-ORPHAN-SCOPE-T06.
#
# WHAT IT GATES
#   The M5-residual op catalogue (crates/vokra-core/src/m5_residual_ops.rs) is
#   a set of *reserved but unlanded* op-kind ids. A core part of the
#   reservation contract (module doc + docs/abi-changelog.md "Reserved
#   additions") is that these ops add **no C ABI symbol** until they actually
#   land in M5. This gate makes that claim machine-checkable: it asserts that
#   none of the reserved op-kind id strings appears in the exported C ABI
#   symbol list.
#
#   If one of these ops ever legitimately lands with a C-ABI function (e.g. a
#   future `vokra_ctc_decode`), this gate fires — that is the intended signal
#   that the op left the reserved-no-symbol state and the abi-changelog +
#   module doc reservation must be reconciled (not a false alarm).
#
#   NOTE: the sibling `OpKind`-non-registration dimension is deliberately NOT
#   gated here — it is a structural property (OpKind has no string-resolution
#   path, so a `&str` can never become a variant) with no runtime assertion
#   target. See m5_residual_ops.rs module doc and
#   docs/adr/M5-ORPHAN-SCOPE-residual-ops-amx-sme.md §(6).
#
# WHY THE SYMBOL LIST, NOT THE RAW HEADER
#   We check against the extracted FUNC/TYPEDEF *names*
#   (scripts/check-abi-changelog.sh --list), not a grep over include/vokra.h.
#   The header carries rustdoc-derived comments; a doc comment mentioning an op
#   name by word would false-positive on a raw grep. The symbol list is
#   comment-stripped and is the actual ABI surface.
#
# SOURCE OF TRUTH FOR THE OP IDS
#   Extracted live from the Rust source so the gate cannot drift from the
#   catalogue:
#     - the six `pub const *_OP: &str = "..."` in m5_residual_ops.rs
#     - BIGVGAN_GENERATOR_OP (re-exported there) from quant/registry.rs
#   Exactly seven ids are expected (the catalogue size guarded by
#   m5_residual_ops::tests::catalogue_is_the_seven_residual_ops); a different
#   count is a setup error here too.
#
# MODES
#   scripts/check-m5-residual-no-abi.sh              -- verify (default)
#   scripts/check-m5-residual-no-abi.sh --list-ops   -- print the reserved ids
#   scripts/check-m5-residual-no-abi.sh --self-test  -- unit-test the logic
#   scripts/check-m5-residual-no-abi.sh --help       -- this text
#
# ZERO-DEP
#   Pure bash + sed + grep. bash 3.2 (macOS default) compatible. No Rust
#   toolchain, no cbindgen — it reads the committed include/vokra.h via
#   check-abi-changelog.sh --list (which is itself pure text).
#
# CI
#   Wired into the `abi-surface` job (advisory, continue-on-error) next to the
#   ABI changelog gate — the enforcement locus for C-ABI symbol drift during
#   the v1.0-rc window (the IF-01 freeze fires at M5-13).
#
# EXIT CODES
#   0  clean (no reserved op id appears in the C ABI symbol list) / --list-ops
#      / --self-test / --help success
#   1  a reserved op id was found in the C ABI symbol list
#   2  usage / setup error (missing source, wrong op-id count, missing gate)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESIDUAL_SRC="$ROOT/crates/vokra-core/src/m5_residual_ops.rs"
REGISTRY_SRC="$ROOT/crates/vokra-core/src/quant/registry.rs"
ABICHECK="$ROOT/scripts/check-abi-changelog.sh"

usage() {
    sed -n '3,52p' "$0" | sed 's/^# \{0,1\}//'
}

# --------------------------------------------------------------- extract ---
# extract_op_ids — emit the reserved op-kind id string literals, one per line
# (unsorted; the caller sorts/dedups). Reads them from the Rust source so the
# gate tracks the catalogue automatically.
extract_op_ids() {
    if [ ! -f "$RESIDUAL_SRC" ]; then
        echo "error: source not found: $RESIDUAL_SRC" >&2
        return 2
    fi
    if [ ! -f "$REGISTRY_SRC" ]; then
        echo "error: source not found: $REGISTRY_SRC" >&2
        return 2
    fi
    # The six declared-here anchors: `pub const NAME_OP: &str = "literal";`.
    sed -nE 's/^pub const [A-Z0-9_]+_OP: &str = "([a-z0-9_]+)";.*$/\1/p' "$RESIDUAL_SRC"
    # BigVGAN is re-exported here; its literal lives in the M2-08 registry.
    sed -nE 's/^pub const BIGVGAN_GENERATOR_OP: &str = "([a-z0-9_]+)";.*$/\1/p' "$REGISTRY_SRC"
}

# abi_symbol_names — the exported C ABI symbol *names* (FUNC + TYPEDEF), one
# per line. Strips the `FUNC `/`TYPEDEF ` kind prefix and the `|<signature>`
# tail, leaving just the identifier we test against.
abi_symbol_names() {
    if [ ! -x "$ABICHECK" ] && [ ! -f "$ABICHECK" ]; then
        echo "error: ABI symbol gate not found: $ABICHECK" >&2
        return 2
    fi
    bash "$ABICHECK" --list | sed -E 's/^(FUNC|TYPEDEF) //; s/\|.*$//'
}

# ---------------------------------------------------------- core logic ---
# find_collisions <opids> <names> — for each op id (newline-separated in $1),
# report every C ABI symbol name (newline-separated in $2) that contains it as
# a substring. Prints collisions to stderr. Returns 1 iff any collision found,
# 0 otherwise. `grep -F` (fixed string) — op ids have no regex metachars, but
# fixed-string keeps a future rename honest.
find_collisions() {
    local opids="$1" names="$2" found=0 opid hit
    while IFS= read -r opid; do
        [ -z "$opid" ] && continue
        hit="$(printf '%s\n' "$names" | grep -F -- "$opid" || true)"
        if [ -n "$hit" ]; then
            found=1
            echo "  COLLISION: reserved M5-residual op-kind id '$opid' appears" >&2
            echo "             in the exported C ABI symbol(s):" >&2
            printf '%s\n' "$hit" | sed 's/^/               /' >&2
        fi
    done <<EOF
$opids
EOF
    return "$found"
}

# ------------------------------------------------------------ self-test ---
# self_test — exercise the collision detector against synthetic inputs and
# confirm the live extractor yields exactly the seven catalogue ids, so a
# future edit to the sed patterns or the detector cannot silently break the
# gate.
self_test() {
    local opids_syn names_clean names_dirty
    opids_syn="$(printf '%s\n' ctc_decode diarize bigvgan_generator)"
    names_clean="$(printf '%s\n' vokra_asr_transcribe vokra_session_t vokra_string_free)"
    names_dirty="$(printf '%s\n' vokra_asr_transcribe vokra_ctc_decode vokra_session_t)"

    # Clean list must NOT be flagged.
    if ! find_collisions "$opids_syn" "$names_clean" 2>/dev/null; then
        echo "self-test FAILED: clean symbol list wrongly flagged" >&2
        return 1
    fi
    # Dirty list (contains vokra_ctc_decode) MUST be flagged.
    if find_collisions "$opids_syn" "$names_dirty" 2>/dev/null; then
        echo "self-test FAILED: colliding symbol 'vokra_ctc_decode' not detected" >&2
        return 1
    fi
    # The live extractor must produce exactly the seven catalogue ids.
    local extracted count
    extracted="$(extract_op_ids | LC_ALL=C sort -u)"
    count="$(printf '%s\n' "$extracted" | grep -c . || true)"
    if [ "$count" -ne 7 ]; then
        echo "self-test FAILED: expected 7 reserved op ids, extracted $count:" >&2
        printf '%s\n' "$extracted" | sed 's/^/  /' >&2
        return 1
    fi

    echo "check-m5-residual-no-abi --self-test: OK"
    return 0
}

# ----------------------------------------------------------------- main ---
mode="${1:-verify}"
case "$mode" in
    verify|"")
        opids="$(extract_op_ids | LC_ALL=C sort -u)"
        count="$(printf '%s\n' "$opids" | grep -c . || true)"
        if [ "$count" -ne 7 ]; then
            echo "error: expected 7 reserved M5-residual op ids, extracted $count" >&2
            echo "       (catalogue drift? see m5_residual_ops.rs)" >&2
            printf '%s\n' "$opids" | sed 's/^/  /' >&2
            exit 2
        fi
        names="$(abi_symbol_names)"

        echo "M5-residual no-C-ABI-symbol gate (M5-ORPHAN-SCOPE-T06)"
        echo "  reserved op ids : $count (from m5_residual_ops.rs + quant/registry.rs)"
        echo "  ABI symbols     : $(printf '%s\n' "$names" | grep -c . || true) (scripts/check-abi-changelog.sh --list)"

        if find_collisions "$opids" "$names"; then
            echo ""
            echo "check-m5-residual-no-abi: OK (no reserved op id in the C ABI surface)"
            exit 0
        else
            cat >&2 <<EOF

check-m5-residual-no-abi: FAIL — a reserved M5-residual op-kind id appears in
the exported C ABI symbol list (include/vokra.h).

If an M5 op landed with a C-ABI function, this is expected: reconcile the
reservation — move the op out of m5_residual_ops.rs, drop its
docs/abi-changelog.md "Reserved additions" row, and record the new symbol in a
dated abi-changelog entry (scripts/check-abi-changelog.sh). Otherwise the
symbol was added by accident; revert the vokra-capi change.
EOF
            exit 1
        fi
        ;;

    --list-ops)
        extract_op_ids | LC_ALL=C sort -u
        ;;

    --self-test)
        set +e
        self_test
        exit $?
        ;;

    --help|-h)
        usage
        exit 0
        ;;

    *)
        echo "error: unknown argument '$mode'" >&2
        echo "usage: $0 [--list-ops | --self-test | --help]" >&2
        exit 2
        ;;
esac
