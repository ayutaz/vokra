#!/usr/bin/env bash
# check-binary-size.sh — binary-size budget gate for the shipped Vokra runtime
# library (M1-11a; NFR-DS-01 "<10MB core single binary", <5MB mobile goal).
#
# WHAT IT GATES
#   The C ABI shared library `libvokra` — libvokra.dylib (macOS) / libvokra.so
#   (Linux) / vokra.dll (Windows) — built by crates/vokra-capi as a `cdylib`
#   (output name `vokra`). This is the single loadable binary that Unity / Godot /
#   iOS / Android and the CLI embed, so it is the artifact NFR-DS-01's "<10MB core
#   single binary" budget applies to.
#   The staticlib (libvokra.a) is an *archive* of all object code — naturally much
#   larger, and the linker dead-strips it into the consumer binary — so it is
#   reported for information only and is NOT gated. The CLI / convert bins are
#   likewise out of scope of this gate (decisions_to_flag #4: the gate is libvokra
#   specifically, not the fatter bins).
#
# SIZE LEVERS (root Cargo.toml [profile.release], added in M1-11a)
#   lto="fat" + codegen-units=1 + strip=true, with opt-level kept at 3 (RTF is the
#   first-priority NFR; the size win comes from LTO+strip, not from lowering
#   opt-level). panic stays "unwind" because vokra-capi's FFI firewall uses
#   std::panic::catch_unwind (panic="abort" would break it). See
#   docs/design/size-budget.md.
#
# SOFT GATE (today — M1-11a)
#   The spike libvokra is already well under budget because no full model weights
#   are linked into the binary — weights load from external GGUF files at runtime.
#   The real <10MB verification against a full-model build is M1-11b (blocked on
#   real GGUFs / hardware). Until then this is a regression tripwire on the *code*
#   size, with a provisional 10 MiB budget; raise it (VOKRA_SIZE_BUDGET_BYTES) or
#   run soft (VOKRA_SIZE_SOFT=1) while the budget is being calibrated.
#
# CONFIG (environment variables)
#   VOKRA_SIZE_BUDGET_BYTES       hard budget for the cdylib      (default 10 MiB)
#   VOKRA_SIZE_MOBILE_GOAL_BYTES  informational mobile goal        (default  5 MiB)
#   VOKRA_SIZE_SOFT=1             over-budget warns instead of failing (exit 0)
#   VOKRA_SIZE_SKIP_BUILD=1       measure the existing artifact; skip cargo build
#   VOKRA_SIZE_CDYLIB=<path>      measure this file instead of auto-locating
#   VOKRA_SIZE_TEST_BYTES=<n>     use <n> as the measured size (no build; testing)
#
# USAGE
#   scripts/check-binary-size.sh              # build release cdylib, measure, gate
#   scripts/check-binary-size.sh --self-test  # unit-test the compare logic (no build)
#
# Exit code: 0 = within budget (or soft warn / self-test pass); 1 = over budget
# (hard mode); 2 = usage / build / artifact-not-found error.
#
# CI wiring (a `size` job in .github/workflows/ci.yml) is intentionally left to a
# follow-up so this WP does not clobber concurrent CI edits; run this in the
# `build` or `license` job with VOKRA_SIZE_SKIP_BUILD=1 after `cargo build
# --release`. A `cargo bloat` top-symbols step is an OPTIONAL build-time dev tool
# (never a Cargo.lock dependency — same non-dependency status as
# cargo-deny / cargo-audit), used only for diagnosis when the gate trips.
#
# bash 3.2 (macOS default) compatible: no associative arrays / bashisms >4.0.

set -euo pipefail

readonly DEFAULT_BUDGET=$((10 * 1024 * 1024))     # 10 MiB
readonly DEFAULT_MOBILE_GOAL=$((5 * 1024 * 1024)) #  5 MiB

BUDGET="${VOKRA_SIZE_BUDGET_BYTES:-$DEFAULT_BUDGET}"
MOBILE_GOAL="${VOKRA_SIZE_MOBILE_GOAL_BYTES:-$DEFAULT_MOBILE_GOAL}"
SOFT="${VOKRA_SIZE_SOFT:-0}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"

# human <bytes> — render a byte count as a short human string (B / KiB / MiB)
# using integer-only arithmetic (bash has no floating point).
human() {
    local b="$1"
    if [ "$b" -ge $((1024 * 1024)) ]; then
        printf '%d.%02d MiB' \
            $((b / (1024 * 1024))) \
            $(((b % (1024 * 1024)) * 100 / (1024 * 1024)))
    elif [ "$b" -ge 1024 ]; then
        printf '%d.%02d KiB' \
            $((b / 1024)) \
            $(((b % 1024) * 100 / 1024))
    else
        printf '%d B' "$b"
    fi
}

# gate_decision <size> <budget> <soft> — echoes PASS | WARN | FAIL and returns 0
# unless it is a hard (non-soft) over-budget, which returns 1. Pure: it does no
# build and no I/O beyond the single verdict word, so --self-test exercises the
# size compare/parse logic directly with synthetic sizes.
gate_decision() {
    local size="$1" budget="$2" soft="$3"
    if [ "$size" -le "$budget" ]; then
        echo PASS
        return 0
    fi
    if [ "$soft" = "1" ]; then
        echo WARN
        return 0
    fi
    echo FAIL
    return 1
}

# ------------------------------------------------------------------ self-test ---
self_test() {
    local fail=0 v rc
    # check <desc> <want-verdict> <want-rc> <got-verdict> <got-rc>
    check() {
        if [ "$2" = "$4" ] && [ "$3" = "$5" ]; then
            echo "  ok: $1 -> $4 (rc=$5)"
        else
            echo "  FAIL: $1 -> got $4 (rc=$5), want $2 (rc=$3)" >&2
            fail=1
        fi
    }

    v=$(gate_decision 1048576 10485760 0); rc=$?
    check "1MiB under 10MiB budget" PASS 0 "$v" "$rc"

    v=$(gate_decision 10485760 10485760 0); rc=$?
    check "exactly 10MiB == budget (must-not-exceed)" PASS 0 "$v" "$rc"

    v=$(gate_decision 20971520 10485760 0); rc=$?
    check "20MiB over 10MiB budget (hard)" FAIL 1 "$v" "$rc"

    v=$(gate_decision 20971520 10485760 1); rc=$?
    check "20MiB over 10MiB budget (soft)" WARN 0 "$v" "$rc"

    # human() formatter sanity (integer-only rounding is truncating).
    local h
    h=$(human 5242880); [ "$h" = "5.00 MiB" ] || { echo "  FAIL: human(5MiB)=$h" >&2; fail=1; }
    h=$(human 1536);    [ "$h" = "1.50 KiB" ] || { echo "  FAIL: human(1536)=$h" >&2; fail=1; }
    h=$(human 512);     [ "$h" = "512 B" ]    || { echo "  FAIL: human(512)=$h" >&2; fail=1; }

    if [ "$fail" = 0 ]; then
        echo "check-binary-size --self-test: OK"
        return 0
    fi
    echo "check-binary-size --self-test: FAILED" >&2
    return 1
}

if [ "${1:-}" = "--self-test" ]; then
    # --self-test intentionally exercises the non-zero (FAIL) return path, so
    # disable errexit for the duration.
    set +e
    self_test
    exit $?
fi

if [ "$#" -gt 0 ]; then
    echo "error: unknown argument '$1'" >&2
    echo "usage: $0 [--self-test]" >&2
    exit 2
fi

# --------------------------------------------------------------- measurement ---
if [ -n "${VOKRA_SIZE_TEST_BYTES:-}" ]; then
    # Testing / override hook: skip the build and use a synthetic size.
    SIZE="$VOKRA_SIZE_TEST_BYTES"
    SIZE_SRC="synthetic (VOKRA_SIZE_TEST_BYTES=$SIZE)"
else
    if [ "${VOKRA_SIZE_SKIP_BUILD:-0}" != "1" ]; then
        echo "building release cdylib (cargo build --release -p vokra-capi) ..."
        # -p vokra-capi builds exactly the gated artifact (and its vokra-* deps).
        # A full-workspace `cargo build --release` produces the same libvokra.
        ( cd "$ROOT" && cargo build --release -p vokra-capi )
    fi

    CDYLIB="${VOKRA_SIZE_CDYLIB:-}"
    if [ -z "$CDYLIB" ]; then
        for n in libvokra.dylib libvokra.so vokra.dll; do
            if [ -f "$TARGET_DIR/release/$n" ]; then
                CDYLIB="$TARGET_DIR/release/$n"
                break
            fi
        done
    fi

    if [ -z "$CDYLIB" ] || [ ! -f "$CDYLIB" ]; then
        echo "error: libvokra cdylib not found under $TARGET_DIR/release" >&2
        echo "       expected one of: libvokra.dylib / libvokra.so / vokra.dll" >&2
        echo "       build it with: cargo build --release -p vokra-capi" >&2
        exit 2
    fi

    SIZE="$(wc -c < "$CDYLIB" | tr -d '[:space:]')"
    SIZE_SRC="$CDYLIB"
fi

# ------------------------------------------------------------------- report ----
echo ""
echo "Vokra binary-size gate (M1-11a; NFR-DS-01 core single binary)"
echo "  artifact : $SIZE_SRC"
echo "  size     : $(human "$SIZE")  ($SIZE bytes)"
echo "  budget   : $(human "$BUDGET")  ($BUDGET bytes)  [hard gate]"
echo "  mobile   : $(human "$MOBILE_GOAL")  ($MOBILE_GOAL bytes)  [informational goal]"

# Informational: staticlib archive size (naturally large; NOT gated).
for n in libvokra.a vokra.lib; do
    if [ -f "$TARGET_DIR/release/$n" ]; then
        asize="$(wc -c < "$TARGET_DIR/release/$n" | tr -d '[:space:]')"
        echo "  staticlib: $(human "$asize")  ($n — archive, linker dead-strips it; NOT gated)"
        break
    fi
done

if [ "$SIZE" -gt "$MOBILE_GOAL" ]; then
    echo "  note     : over the informational <$(human "$MOBILE_GOAL") mobile goal (not a failure)."
fi

# errexit off: gate_decision returns 1 for a hard over-budget, which we surface
# ourselves via the explicit exit below.
set +e
verdict=$(gate_decision "$SIZE" "$BUDGET" "$SOFT")
gate_rc=$?
set -e

echo ""
case "$verdict" in
    PASS)
        echo "check-binary-size: OK (libvokra within the <$(human "$BUDGET") budget)"
        ;;
    WARN)
        echo "check-binary-size: WARNING — over budget but VOKRA_SIZE_SOFT=1 (soft gate, not failing)" >&2
        ;;
    FAIL)
        echo "check-binary-size: FAIL — libvokra exceeds the $(human "$BUDGET") budget" >&2
        echo "  Reduce it (feature-gate optional ops per docs/design/size-budget.md; inspect" >&2
        echo "  the top symbols with 'cargo bloat --release -p vokra-capi'), or, if the growth" >&2
        echo "  is expected, raise VOKRA_SIZE_BUDGET_BYTES with a documented rationale." >&2
        ;;
esac

exit "$gate_rc"
