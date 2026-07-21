#!/usr/bin/env bash
# check-nostd-subset.sh — enforce that Vokra's no_std(+alloc) subset keeps
# cross-compiling for bare-metal Cortex-M55 (M5-03-T07/T09/T15; NFR-PT-03
# Tier 3, NFR-DS-02).
#
# Vokra's IoT Tier 3 revival rests on a core-clean subset that builds under
# `#![no_std]` (with `alloc`):
#   * `vokra-core`      — the error type + the GGUF reader parse / from_external
#                         decode path (M5-03 Wave 1);
#   * `vokra-vad-micro` — the Silero VAD v5 forward core (weights binding +
#                         pseudo-STFT + encoder + LSTM + head + the
#                         self-contained scalar exp/tanh/sqrt), Wave 2 (T09).
# This tripwire is a permanent-invariant guard in the check-zero-deps.sh mould —
# it does NOT test behavior, it asserts a structural property that must never
# regress.
#
# Mechanism (two layers):
#   1. COMPILE GATE (authoritative). Build each crate with
#      `--no-default-features` — which flips it to `#![no_std]` — for the
#      Cortex-M55 rustc targets. If any std-only item leaks into the no_std
#      surface (a stray `use std::…`, or an exposed module that is not
#      `#[cfg(feature = "std")]`-gated), the build fails with a compile error.
#      This is the strongest possible detector: `#![no_std]` cannot reach `std`
#      without an explicit `extern crate std`.
#   2. SYMBOL SCAN (belt-and-suspenders). A clean no_std+alloc rlib links only
#      `core` and `alloc`; it contains NO `std::` symbols. `nm` each built rlib
#      and fail if any legacy-mangled `std` namespace symbol appears. Skipped
#      (with a note) when `nm` is unavailable.
#
# Local: `bash scripts/check-nostd-subset.sh`. CI wiring is M5-03-T15 (it runs
# per-PR in ci.yml's `license` job; the weekly Tier-3 platform leg is
# .github/workflows/silero-nostd-cross-build.yml). `--self-test` proves the
# compile gate has real detection power by injecting a temporary `use std::fs;`
# leak into an exposed module and confirming the build then fails (the file is
# always restored).
#
# Exit code: 0 = subset still no_std-clean; 1 = a std leak (or a self-test that
# failed to detect an injected leak).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# The no_std subset crates, in dependency order (vokra-vad-micro depends on
# vokra-core). Each must cross-compile with --no-default-features.
CRATES=("vokra-core" "vokra-vad-micro")
# The crate whose exposed module the self-test injects a leak into (the leak
# must be in a crate that is part of the no_std subset).
SELFTEST_CRATE="vokra-core"
# Cortex-M55 rustc targets: hard-float (FP-armv8) and soft-float ABI.
TARGETS=("thumbv8m.main-none-eabihf" "thumbv8m.main-none-eabi")

# --- build one crate's no_std subset for one target; return cargo's status ----
build_nostd() {
    local crate="$1" target="$2"
    cargo build -p "$crate" --no-default-features --target "$target" --quiet
}

# --- self-test: prove the compile gate detects an injected std leak -----------
if [ "${1:-}" = "--self-test" ]; then
    LEAK_FILE="crates/vokra-core/src/error.rs"
    BACKUP="$(mktemp)"
    cp "$LEAK_FILE" "$BACKUP"
    # Always restore the source, even on error / interrupt.
    trap 'cp "$BACKUP" "$LEAK_FILE"; rm -f "$BACKUP"' EXIT
    # Inject a std-only import into an exposed (no_std) module.
    printf '\n#[cfg(not(feature = "std"))]\nuse std::fs as _leak_probe;\n' >> "$LEAK_FILE"
    target="${TARGETS[0]}"
    if ! rustup target list --installed | grep -qx "$target"; then
        echo "self-test: SKIP (target $target not installed)" >&2
        exit 0
    fi
    if build_nostd "$SELFTEST_CRATE" "$target" >/dev/null 2>&1; then
        echo "self-test: FAIL — the compile gate did NOT detect an injected std leak" >&2
        exit 1
    fi
    echo "self-test: OK — the compile gate detected the injected std leak"
    exit 0
fi

fail=0
built_any=0
for t in "${TARGETS[@]}"; do
    if ! rustup target list --installed | grep -qx "$t"; then
        echo "warning: rust target '$t' not installed; skipping." >&2
        echo "         install with: rustup target add $t" >&2
        continue
    fi
    for crate in "${CRATES[@]}"; do
        echo "check-nostd-subset: building $crate (--no-default-features = #![no_std]) for $t ..."
        if ! build_nostd "$crate" "$t"; then
            echo "error: the no_std subset of $crate failed to compile for $t." >&2
            echo "       A std-only item leaked into the core-clean subset. Gate it" >&2
            echo "       behind #[cfg(feature = \"std\")] or hold it out of the no_std" >&2
            echo "       surface (M5-03, NFR-PT-03 Tier 3)." >&2
            fail=1
            continue
        fi
        built_any=1

        # Symbol scan: a clean no_std+alloc rlib carries no `std` symbols.
        crate_us="${crate//-/_}"
        rlib="$(ls -t "target/$t/debug/deps/lib${crate_us}-"*.rlib 2>/dev/null | head -1 || true)"
        if [ -z "$rlib" ]; then
            rlib="$(ls -t "target/$t/debug/lib${crate_us}.rlib" 2>/dev/null | head -1 || true)"
        fi
        if ! command -v nm >/dev/null 2>&1; then
            echo "check-nostd-subset: note — 'nm' unavailable; skipping symbol scan for $crate/$t."
        elif [ -z "$rlib" ]; then
            echo "check-nostd-subset: note — no rlib found for $crate/$t; skipping symbol scan."
        else
            # Legacy Rust mangling puts std's namespace as `…3std…` (the `3` is
            # the length prefix of "std"). A core/alloc-only rlib has none.
            if nm "$rlib" 2>/dev/null | grep -qE '_ZN[0-9]*3std|3std[0-9E]'; then
                echo "error: the no_std $crate rlib for $t contains std-namespace symbols." >&2
                echo "       A std item leaked past the compile gate — investigate:" >&2
                echo "       nm '$rlib' | grep 3std" >&2
                fail=1
            fi
        fi
    done
done

if [ "$built_any" -eq 0 ] && [ "$fail" -eq 0 ]; then
    echo "check-nostd-subset: WARNING — no Cortex-M55 target installed; nothing checked." >&2
    echo "         install one with: rustup target add ${TARGETS[0]}" >&2
    # Not a hard failure locally (a dev may lack the target); CI (T15) installs it.
    exit 0
fi

if [ "$fail" -ne 0 ]; then
    exit 1
fi
echo "check-nostd-subset: OK (${CRATES[*]} no_std subset cross-compiles clean for Cortex-M55)"
