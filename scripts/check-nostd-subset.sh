#!/usr/bin/env bash
# check-nostd-subset.sh — enforce that vokra-core's no_std(+alloc) subset keeps
# cross-compiling for bare-metal Cortex-M55 (M5-03-T07; NFR-PT-03 Tier 3,
# NFR-DS-02).
#
# Vokra's IoT Tier 3 revival rests on a core-clean subset of `vokra-core` that
# builds under `#![no_std]` (with `alloc`): the error type and the GGUF reader
# parse / from_external decode path. This tripwire is a permanent-invariant
# guard in the check-zero-deps.sh mould — it does NOT test behavior, it asserts
# a structural property that must never regress.
#
# Mechanism (two layers):
#   1. COMPILE GATE (authoritative). Build `vokra-core` with
#      `--no-default-features` — which flips the crate to `#![no_std]` — for the
#      Cortex-M55 rustc targets. If any std-only item leaks into the no_std
#      surface (a stray `use std::…` in error.rs / gguf reader, or an exposed
#      module that is not `#[cfg(feature = "std")]`-gated), the build fails with
#      a compile error. This is the strongest possible detector: `#![no_std]`
#      cannot reach `std` without an explicit `extern crate std`.
#   2. SYMBOL SCAN (belt-and-suspenders). A clean no_std+alloc rlib links only
#      `core` and `alloc`; it contains NO `std::` symbols. `nm` the built rlib
#      and fail if any legacy-mangled `std` namespace symbol appears. Skipped
#      (with a note) when `nm` is unavailable.
#
# Local: `bash scripts/check-nostd-subset.sh`. CI wiring is M5-03-T15 (Wave 3).
# `--self-test` proves the compile gate has real detection power by injecting a
# temporary `use std::fs;` leak into an exposed module and confirming the build
# then fails (the file is always restored).
#
# Exit code: 0 = subset still no_std-clean; 1 = a std leak (or a self-test that
# failed to detect an injected leak).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CRATE="vokra-core"
# Cortex-M55 rustc targets: hard-float (FP-armv8) and soft-float ABI.
TARGETS=("thumbv8m.main-none-eabihf" "thumbv8m.main-none-eabi")

# --- build the no_std subset for one target; return cargo's exit status -------
build_nostd() {
    local target="$1"
    cargo build -p "$CRATE" --no-default-features --target "$target" --quiet
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
    if build_nostd "$target" >/dev/null 2>&1; then
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
    echo "check-nostd-subset: building $CRATE (--no-default-features = #![no_std]) for $t ..."
    if ! build_nostd "$t"; then
        echo "error: the no_std subset of $CRATE failed to compile for $t." >&2
        echo "       A std-only item leaked into the core-clean subset (error /" >&2
        echo "       gguf reader). Gate it behind #[cfg(feature = \"std\")] or hold" >&2
        echo "       it out of the no_std surface (M5-03, NFR-PT-03 Tier 3)." >&2
        fail=1
        continue
    fi
    built_any=1

    # Symbol scan: a clean no_std+alloc rlib carries no `std` namespace symbols.
    rlib="$(ls -t "target/$t/debug/deps/libvokra_core-"*.rlib 2>/dev/null | head -1 || true)"
    if [ -z "$rlib" ]; then
        rlib="$(ls -t "target/$t/debug/libvokra_core.rlib" 2>/dev/null | head -1 || true)"
    fi
    if ! command -v nm >/dev/null 2>&1; then
        echo "check-nostd-subset: note — 'nm' unavailable; skipping symbol scan for $t."
    elif [ -z "$rlib" ]; then
        echo "check-nostd-subset: note — no rlib found for $t; skipping symbol scan."
    else
        # Legacy Rust mangling puts std's namespace as `…3std…` (the `3` is the
        # length prefix of "std"). A core/alloc-only rlib has none.
        if nm "$rlib" 2>/dev/null | grep -qE '_ZN[0-9]*3std|3std[0-9E]'; then
            echo "error: the no_std $CRATE rlib for $t contains std-namespace symbols." >&2
            echo "       A std item leaked past the compile gate — investigate:" >&2
            echo "       nm '$rlib' | grep 3std" >&2
            fail=1
        fi
    fi
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
echo "check-nostd-subset: OK ($CRATE no_std subset cross-compiles clean for Cortex-M55)"
