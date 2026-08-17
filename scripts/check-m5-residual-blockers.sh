#!/usr/bin/env bash
# check-m5-residual-blockers.sh — M5-residual blocker-column honesty gate.
#
# WHAT IT GATES
#   The M5-residual op catalogue (crates/vokra-core/src/m5_residual_ops.rs)
#   carries a `blocker` column per reserved op-kind id. ADR M4-20 §D-6 and the
#   M5-13 freeze decision read that column to decide what may land after the
#   IF-01 freeze as a backward-compatible additive — so a blocker that lies
#   about the tree directly misinforms the freeze scope.
#
#   The specific failure this gate catches is a *decaying factual claim*. A
#   blocker like "no trigger model" or "NeMo-family trigger pending" asserts
#   that something does not exist yet. That is true only until it lands, and
#   nothing re-reads the column when it does. On 2026-08-15 three rows
#   (bigvgan_generator, ctc_decode, rnnt_decode) were found stale in exactly
#   this way: all three runtime primitives had landed, and rnnt_decode even had
#   a live consumer (ParakeetTdt11b::decode_tdt).
#
#   The durable phrasing is the one ADR M4-20 §D-5 implies: these primitives
#   are deliberately *runtime functions*, NOT `OpKind` variants, so what a row
#   reserves is the graph-side variant + C ABI export. That statement stays
#   true after the primitive ships. This gate pushes rows toward it.
#
# THE RULE (one rule, deliberately narrow)
#   For each reserved op-kind id `X`: if `crates/vokra-ops/src/X.rs` exists AND
#   declares at least one `pub fn`, then X's blocker text MUST NOT contain an
#   absence-claim phrase (the closed list in ABSENCE_PHRASES below).
#
#   Both halves must hold for a row to be flagged, which is what keeps this
#   free of false positives:
#     * The module filename must match the reserved op id EXACTLY — not a
#       substring, not a fuzzy match. Ops with no `vokra-ops` module of that
#       name are never examined at all, so legitimately-absent rows keep their
#       natural wording. Today that exempts ecapa_tdnn_speaker_encode,
#       wespeaker_speaker_encode, titanet_speaker_encode ("CAM++ already covers
#       speaker embedding") and diarize ("trigger only ..."), none of which has
#       a same-named ops module.
#     * The phrase list is a closed, reviewed set of absence assertions. It
#       deliberately does NOT contain the bare word "trigger" — "trigger only"
#       is a legitimate blocker shape (diarize) and must not fire.
#
# WHAT IT DOES NOT CATCH (stated so nobody over-trusts it)
#   A blocker can still be wrong in ways no script can see — a novel phrasing
#   of "does not exist", a wrong FR-OP mapping, or a claim about a model that
#   lives outside vokra-ops. This gate is a tripwire for the one drift shape
#   that actually bit us and is mechanically decidable, not a proof of
#   correctness. Widening the phrase list is cheap; widening the *rule* (e.g.
#   grepping for "live call sites") is not, because call-site detection by grep
#   trips over doc comments and over op names quoted inside loud-partial error
#   messages — which is exactly how several models legitimately mention
#   `ctc_decode_greedy` today without calling it.
#
# SOURCE OF TRUTH
#   Extracted live from the Rust source so the gate cannot drift from the
#   catalogue:
#     - `pub const *_OP: &str = "..."` in m5_residual_ops.rs (six anchors)
#     - BIGVGAN_GENERATOR_OP (re-exported there) from quant/registry.rs
#     - the `op_id:` / `blocker:` pairs inside `m5_residual_op_anchors()`
#   Exactly seven anchors are expected (the catalogue size is separately
#   pinned by m5_residual_ops::tests::catalogue_is_the_seven_residual_ops); a
#   different count is a setup error here too.
#
# MODES
#   scripts/check-m5-residual-blockers.sh              -- verify (default)
#   scripts/check-m5-residual-blockers.sh --list       -- print id/module/blocker
#   scripts/check-m5-residual-blockers.sh --self-test  -- unit-test the logic
#   scripts/check-m5-residual-blockers.sh --help       -- this text
#
# ZERO-DEP
#   Pure bash + sed + grep + awk. bash 3.2 (macOS default) compatible. No Rust
#   toolchain, no cargo — it reads committed source text only.
#
# CI
#   Wired into the `abi-surface` job (advisory, continue-on-error) next to
#   check-m5-residual-no-abi.sh, its sibling gate on the same catalogue.
#
# EXIT CODES
#   0  clean / --list / --self-test / --help success
#   1  a blocker asserts absence for an op whose vokra-ops module exists
#   2  usage / setup error (missing source, wrong anchor count, parse failure)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Overridable so --self-test can point the extractors at synthetic fixtures.
RESIDUAL_SRC="${RESIDUAL_SRC:-$ROOT/crates/vokra-core/src/m5_residual_ops.rs}"
REGISTRY_SRC="${REGISTRY_SRC:-$ROOT/crates/vokra-core/src/quant/registry.rs}"
OPS_DIR="${OPS_DIR:-$ROOT/crates/vokra-ops/src}"

# Closed list of absence-claim phrases (case-insensitive, ERE alternation).
# Adding to this list is a review decision: every entry must be a phrase that
# asserts a thing does NOT exist, never a phrase describing what is reserved.
ABSENCE_PHRASES='no trigger model|trigger pending|trigger is pending|trigger model pending|awaiting trigger|pending trigger|unimplemented|not implemented|no implementation|does not exist|not yet exist|no live implementation'

usage() {
    sed -n '3,90p' "$0" | sed 's/^# \{0,1\}//'
}

# ---------------------------------------------------------------- extract ---
# const_literal_map — emit `CONST_NAME|literal` for every reserved op-kind
# constant, from both source files.
const_literal_map() {
    if [ ! -f "$RESIDUAL_SRC" ]; then
        echo "error: source not found: $RESIDUAL_SRC" >&2
        return 2
    fi
    if [ ! -f "$REGISTRY_SRC" ]; then
        echo "error: source not found: $REGISTRY_SRC" >&2
        return 2
    fi
    sed -nE 's/^pub const ([A-Z0-9_]+_OP): &str = "([a-z0-9_]+)";.*$/\1|\2/p' "$RESIDUAL_SRC"
    sed -nE 's/^pub const (BIGVGAN_GENERATOR_OP): &str = "([a-z0-9_]+)";.*$/\1|\2/p' "$REGISTRY_SRC"
}

# anchor_blockers — emit `CONST_NAME|blocker text` for every entry in
# `m5_residual_op_anchors()`. Joins Rust `\` line continuations into one line
# so multi-line blocker literals are matched as a whole.
anchor_blockers() {
    awk '
        function closed(s) { return (s ~ /",[ \t]*$/) }
        function emit(   t) {
            t = acc
            gsub(/\\[ \t]*/, " ", t)          # fold Rust line continuations
            sub(/^[ \t]*"/, "", t)            # strip opening quote
            sub(/",[ \t]*$/, "", t)           # strip closing quote + comma
            gsub(/[ \t]+/, " ", t)            # collapse whitespace
            sub(/^ /, "", t); sub(/ $/, "", t)
            print cur "|" t
            collecting = 0; acc = ""; cur = ""
        }
        /m5_residual_op_anchors\(\)/ { inblk = 1; next }
        inblk == 0 { next }
        /^\}/ { inblk = 0; next }
        {
            if (collecting == 1) {
                acc = acc " " $0
                if (closed($0)) emit()
                next
            }
            if (match($0, /op_id:[ \t]*[A-Za-z0-9_]+/)) {
                s = substr($0, RSTART, RLENGTH)
                sub(/op_id:[ \t]*/, "", s)
                cur = s
            }
            i = index($0, "blocker:")
            if (i > 0) {
                acc = substr($0, i + 8)
                collecting = 1
                if (closed(acc)) emit()
            }
        }
    ' "$RESIDUAL_SRC"
}

# ops_module_has_pub_fn <literal> — 0 iff crates/vokra-ops/src/<literal>.rs
# exists and declares at least one `pub fn`.
ops_module_has_pub_fn() {
    local lit="$1" f="$OPS_DIR/$1.rs"
    [ -n "$lit" ] || return 1
    [ -f "$f" ] || return 1
    grep -qE '^[[:space:]]*pub fn ' "$f"
}

# has_absence_claim <text> — 0 iff text contains a listed absence phrase.
has_absence_claim() {
    printf '%s\n' "$1" | grep -qiE "$ABSENCE_PHRASES"
}

# --------------------------------------------------------------- core ---
# resolve_rows — emit `literal|module_state|blocker`, where module_state is
# `module` or `nomodule`. Fails (2) if a blocker's op_id constant has no known
# literal, which would mean the parser or the catalogue drifted.
resolve_rows() {
    local map rows line cname blocker lit
    map="$(const_literal_map)" || return 2
    rows="$(anchor_blockers)" || return 2
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        cname="${line%%|*}"
        blocker="${line#*|}"
        lit="$(printf '%s\n' "$map" | grep -E "^${cname}\|" | head -n 1 | cut -d'|' -f2)"
        if [ -z "$lit" ]; then
            echo "error: no string literal found for op_id constant '$cname'" >&2
            echo "       (parser drift, or the constant moved out of the two known sources)" >&2
            return 2
        fi
        if ops_module_has_pub_fn "$lit"; then
            printf '%s|module|%s\n' "$lit" "$blocker"
        else
            printf '%s|nomodule|%s\n' "$lit" "$blocker"
        fi
    done <<EOF
$rows
EOF
}

# find_violations <rows> — print offending rows to stderr; return 1 iff any.
find_violations() {
    local rows="$1" found=0 line lit state blocker rest
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        lit="${line%%|*}"
        rest="${line#*|}"
        state="${rest%%|*}"
        blocker="${rest#*|}"
        [ "$state" = "module" ] || continue
        if has_absence_claim "$blocker"; then
            found=1
            echo "  STALE BLOCKER: reserved op '$lit' claims absence, but" >&2
            echo "                 crates/vokra-ops/src/$lit.rs exists with a pub fn." >&2
            echo "                 blocker: $blocker" >&2
        fi
    done <<EOF
$rows
EOF
    return "$found"
}

# ------------------------------------------------------------ self-test ---
self_test() {
    local tmp rows out rc
    tmp="$(mktemp -d)"
    # shellcheck disable=SC2064
    trap "rm -rf '$tmp'" RETURN

    # --- 1. absence-phrase detector -------------------------------------
    if ! has_absence_claim "NeMo-family trigger pending"; then
        echo "self-test FAILED: 'trigger pending' not detected as absence claim" >&2
        return 1
    fi
    if ! has_absence_claim "no trigger model; min-dtype anchor already registered"; then
        echo "self-test FAILED: 'no trigger model' not detected as absence claim" >&2
        return 1
    fi
    if has_absence_claim "graph-side OpKind variant + C ABI export reserved"; then
        echo "self-test FAILED: durable wording wrongly flagged as absence claim" >&2
        return 1
    fi
    # The legitimate diarize wording must NOT be treated as an absence claim,
    # so the bare word "trigger" must stay out of the phrase list.
    if has_absence_claim "trigger only (pyannote license MIT signed 2026-07-30)"; then
        echo "self-test FAILED: 'trigger only' wrongly flagged (bare 'trigger' in list?)" >&2
        return 1
    fi

    # --- 2. parser: single-line and continued blockers -------------------
    cat >"$tmp/src.rs" <<'RS'
pub const CTC_DECODE_OP: &str = "ctc_decode";
pub const DIARIZE_OP: &str = "diarize";
pub fn m5_residual_op_anchors() -> &'static [M5ResidualAnchor] {
    &[
        M5ResidualAnchor {
            op_id: CTC_DECODE_OP,
            fr_op: "FR-OP-41",
            blocker: "NeMo-family trigger \
                      pending",
        },
        M5ResidualAnchor {
            op_id: DIARIZE_OP,
            fr_op: "FR-OP-82",
            blocker: "trigger only (license signed)",
        },
    ]
}
RS
    rows="$(RESIDUAL_SRC="$tmp/src.rs" anchor_blockers)"
    if ! printf '%s\n' "$rows" | grep -q '^CTC_DECODE_OP|NeMo-family trigger pending$'; then
        echo "self-test FAILED: continued blocker literal not rejoined; got:" >&2
        printf '%s\n' "$rows" | sed 's/^/  /' >&2
        return 1
    fi
    if ! printf '%s\n' "$rows" | grep -q '^DIARIZE_OP|trigger only (license signed)$'; then
        echo "self-test FAILED: single-line blocker not parsed; got:" >&2
        printf '%s\n' "$rows" | sed 's/^/  /' >&2
        return 1
    fi

    # --- 3. end-to-end: stale row must fire, legit row must not ----------
    # ctc_decode has a real vokra-ops module -> its absence claim is a
    # violation. diarize has none -> "trigger only" must stay clean.
    rows="$(RESIDUAL_SRC="$tmp/src.rs" resolve_rows)" || {
        echo "self-test FAILED: resolve_rows errored on the fixture" >&2
        return 1
    }
    if ! printf '%s\n' "$rows" | grep -q '^ctc_decode|module|'; then
        echo "self-test FAILED: ctc_decode not resolved as having a module" >&2
        return 1
    fi
    if ! printf '%s\n' "$rows" | grep -q '^diarize|nomodule|'; then
        echo "self-test FAILED: diarize wrongly resolved as having a module" >&2
        return 1
    fi
    set +e
    find_violations "$rows" 2>/dev/null
    rc=$?
    set -e
    if [ "$rc" -ne 1 ]; then
        echo "self-test FAILED: stale ctc_decode blocker not flagged (rc=$rc)" >&2
        return 1
    fi

    # --- 4. the same fixture, corrected, must pass -----------------------
    sed 's/NeMo-family trigger \\/graph-side OpKind variant reserved; runtime \\/; s/^                      pending",$/                      primitive landed",/' \
        "$tmp/src.rs" >"$tmp/fixed.rs"
    rows="$(RESIDUAL_SRC="$tmp/fixed.rs" resolve_rows)" || {
        echo "self-test FAILED: resolve_rows errored on the corrected fixture" >&2
        return 1
    }
    set +e
    find_violations "$rows" 2>/dev/null
    rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        echo "self-test FAILED: corrected blocker still flagged (rc=$rc); rows:" >&2
        printf '%s\n' "$rows" | sed 's/^/  /' >&2
        return 1
    fi

    # --- 5. live catalogue shape ----------------------------------------
    out="$(resolve_rows)" || {
        echo "self-test FAILED: resolve_rows errored on the live catalogue" >&2
        return 1
    }
    local count
    count="$(printf '%s\n' "$out" | grep -c . || true)"
    if [ "$count" -ne 7 ]; then
        echo "self-test FAILED: expected 7 live anchors, resolved $count:" >&2
        printf '%s\n' "$out" | sed 's/^/  /' >&2
        return 1
    fi

    echo "check-m5-residual-blockers --self-test: OK"
    return 0
}

# ----------------------------------------------------------------- main ---
mode="${1:-verify}"
case "$mode" in
    verify|"")
        rows="$(resolve_rows)" || exit 2
        count="$(printf '%s\n' "$rows" | grep -c . || true)"
        if [ "$count" -ne 7 ]; then
            echo "error: expected 7 M5-residual anchors, resolved $count" >&2
            echo "       (catalogue drift? see m5_residual_ops.rs)" >&2
            printf '%s\n' "$rows" | sed 's/^/  /' >&2
            exit 2
        fi
        withmod="$(printf '%s\n' "$rows" | grep -c '|module|' || true)"

        echo "M5-residual blocker-column honesty gate"
        echo "  anchors            : $count (from m5_residual_ops.rs)"
        echo "  with vokra-ops mod : $withmod (these may not claim absence)"

        if find_violations "$rows"; then
            echo ""
            echo "check-m5-residual-blockers: OK (no blocker claims absence for a landed op)"
            exit 0
        else
            cat >&2 <<'EOF'

check-m5-residual-blockers: FAIL — a reserved op's blocker asserts that
something does not exist, while its vokra-ops module is present with a pub fn.

ADR M4-20 §D-6 and the M5-13 freeze decision read this column, so a false
blocker misinforms the freeze scope. Fix the DESCRIPTION, not the policy: the
reservation itself stays valid, because per ADR M4-20 §D-5 these primitives are
deliberately runtime functions and NOT OpKind variants. Restate the row as what
is actually still reserved, e.g.

  "graph-side OpKind variant + C ABI export reserved; the runtime primitive
   landed (vokra_ops::<op>, live consumer <path>:<line>)"

Do NOT resolve this by landing an OpKind variant — that is an owner/WP decision
(M5-17), not a gate fix.
EOF
            exit 1
        fi
        ;;

    --list)
        resolve_rows || exit 2
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
        echo "usage: $0 [--list | --self-test | --help]" >&2
        exit 2
        ;;
esac
