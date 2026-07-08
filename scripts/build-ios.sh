#!/usr/bin/env bash
# build-ios.sh — build the Vokra iOS XCFramework (M2-02-T05).
#
# Ticket: M2-02-T05 (iOS build scaffold). Basis: NFR-RL-03 (iOS static-only,
# libvokra.a via DllImport("__Internal")-style), NFR-RL-05 (JIT-free), FR-EX-08
# (no silent fallback: CUDA red-lined on iOS). Design in docs/adr/000N-ios-build.md.
#
# Produces build/ios/Vokra.xcframework with two slices:
#   ios-arm64                        — device (aarch64-apple-ios)
#   ios-arm64_x86_64-simulator       — simulator fat (arm64-sim + x86_64-sim via lipo)
#
# Metal backend is ON (vokra-models/metal); CUDA is UNCONDITIONALLY OFF (App Store
# forbids user dlopen, and cudarc/CUDA absent on iOS). The script REFUSES if the
# caller tries to force CUDA via VOKRA_IOS_ENABLE_CUDA=1.
#
# panic = "unwind" is inherited from the workspace root Cargo.toml; this script
# MUST NOT pass -C panic=abort or export a RUSTFLAGS that overrides it (ffi_guard
# depends on catch_unwind — see R5 in the M2-02 plan).
#
# Usage:
#   scripts/build-ios.sh              # build the XCFramework
#   scripts/build-ios.sh --check      # verify header drift first, then build,
#                                     # then run the symbol whitelist per slice
#
# Env:
#   VOKRA_IOS_MIN_VERSION   iOS deployment target (default 15.0)
#   VOKRA_IOS_OUT_DIR       output directory      (default build/ios)
#   VOKRA_IOS_ENABLE_CUDA   MUST be unset or 0    (any other value = hard error)
#
# Exit codes: 0 = success; 78 = Xcode CLI missing; 1 = any other failure.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${VOKRA_IOS_OUT_DIR:-$ROOT/build/ios}"
MIN_VERSION="${VOKRA_IOS_MIN_VERSION:-15.0}"
MODE="${1:-}"

# --- guardrails ---------------------------------------------------------------

if [ "${VOKRA_IOS_ENABLE_CUDA:-0}" != "0" ]; then
    echo "error: VOKRA_IOS_ENABLE_CUDA is set. CUDA on iOS is red-lined" >&2
    echo "       (App Store forbids user dlopen; libcuda absent; see M2-02 ADR)." >&2
    exit 1
fi

# R5: refuse to run if RUSTFLAGS smuggles a panic override.
if [ -n "${RUSTFLAGS:-}" ] && printf '%s' "$RUSTFLAGS" | grep -q 'panic'; then
    echo "error: RUSTFLAGS overrides panic (\"$RUSTFLAGS\"). ffi_guard needs unwind." >&2
    exit 1
fi

# R2: Xcode CLI present + version floor (XCFramework format is stable since 14).
if ! xcode-select -p >/dev/null 2>&1; then
    echo "ERROR: Xcode Command Line Tools not installed." >&2
    echo "       Run: xcode-select --install" >&2
    exit 78
fi
if ! command -v xcodebuild >/dev/null 2>&1; then
    echo "ERROR: xcodebuild not on PATH (install full Xcode from the App Store)." >&2
    exit 78
fi
xcode_ver="$(xcodebuild -version 2>/dev/null | awk 'NR==1 {print $2}')"
xcode_major="${xcode_ver%%.*}"
if [ -z "$xcode_major" ] || [ "$xcode_major" -lt 14 ]; then
    echo "ERROR: Xcode $xcode_ver detected; Vokra iOS build requires Xcode 14+." >&2
    exit 78
fi

# --- rustup targets (idempotent) ---------------------------------------------

TRIPLES_DEVICE="aarch64-apple-ios"
TRIPLES_SIM_ARM="aarch64-apple-ios-sim"
TRIPLES_SIM_X86="x86_64-apple-ios"

installed="$(rustup target list --installed 2>/dev/null || true)"
for t in "$TRIPLES_DEVICE" "$TRIPLES_SIM_ARM" "$TRIPLES_SIM_X86"; do
    if ! printf '%s\n' "$installed" | grep -q "^$t$"; then
        echo "build-ios: installing rustup target $t"
        rustup target add "$t"
    fi
done

# --- --check mode: header drift gate BEFORE building -------------------------

if [ "$MODE" = "--check" ]; then
    echo "build-ios: --check → verifying include/vokra.h drift"
    "$ROOT/scripts/gen-c-abi.sh" --check
elif [ -n "$MODE" ] && [ "$MODE" != "--check" ]; then
    echo "usage: $0 [--check]" >&2
    exit 1
fi

# --- build the three static libraries ----------------------------------------

FEATURES="vokra-models/metal"
build_one() {
    local triple="$1"
    echo "build-ios: cargo build -p vokra-capi --release --target $triple"
    ( cd "$ROOT" && cargo build -p vokra-capi --release \
        --target "$triple" \
        --no-default-features \
        --features "$FEATURES" )
}
build_one "$TRIPLES_DEVICE"
build_one "$TRIPLES_SIM_ARM"
build_one "$TRIPLES_SIM_X86"

LIB_DEVICE="$ROOT/target/$TRIPLES_DEVICE/release/libvokra.a"
LIB_SIM_ARM="$ROOT/target/$TRIPLES_SIM_ARM/release/libvokra.a"
LIB_SIM_X86="$ROOT/target/$TRIPLES_SIM_X86/release/libvokra.a"
for f in "$LIB_DEVICE" "$LIB_SIM_ARM" "$LIB_SIM_X86"; do
    [ -f "$f" ] || { echo "error: expected staticlib not found: $f" >&2; exit 1; }
done

# --- lipo simulator slices into a fat archive --------------------------------

mkdir -p "$OUT_DIR"
# Name the simulator fat archive `libvokra.a` (matching the device slice)
# rather than `libvokra-sim.a`. `xcodebuild -create-xcframework` preserves
# the input file name inside the resulting `.xcframework`, so any downstream
# consumer that walks the tree (`scripts/verify-ios-xcframework.sh`, Swift
# Package Manager, Unity Package Manager) sees the same `libvokra.a` in
# both `ios-arm64/` and `ios-arm64_x86_64-simulator/`. Keep the two inputs
# in separate subdirectories so `xcodebuild` can distinguish them.
SIM_FAT_DIR="$OUT_DIR/sim-fat"
mkdir -p "$SIM_FAT_DIR"
LIB_SIM_FAT="$SIM_FAT_DIR/libvokra.a"
echo "build-ios: lipo -create → $LIB_SIM_FAT"
lipo -create "$LIB_SIM_ARM" "$LIB_SIM_X86" -output "$LIB_SIM_FAT"

# --- stage headers + module.modulemap ----------------------------------------

HDR_DIR="$OUT_DIR/headers"
rm -rf "$HDR_DIR"
mkdir -p "$HDR_DIR"
cp "$ROOT/include/vokra.h" "$HDR_DIR/vokra.h"
cat > "$HDR_DIR/module.modulemap" <<'EOF'
module Vokra {
    header "vokra.h"
    export *
}
EOF

# --- --check mode: symbol whitelist per slice --------------------------------

if [ "$MODE" = "--check" ]; then
    check_syms() {
        local lib="$1"
        local label="$2"
        local unexpected
        unexpected="$(nm -g "$lib" 2>/dev/null \
            | awk 'NF>=2 {print $NF}' \
            | sed 's/^_//' \
            | grep -vE '^vokra_' \
            | grep -vE '^$' || true)"
        if [ -n "$unexpected" ]; then
            echo "build-ios: FAIL $label exports non-vokra symbols:" >&2
            printf '  %s\n' "$unexpected" | head -20 >&2
            exit 1
        fi
        echo "build-ios: OK $label — all exports vokra_*"
    }
    check_syms "$LIB_DEVICE"  "ios-arm64"
    check_syms "$LIB_SIM_FAT" "ios-arm64_x86_64-simulator"
fi

# --- assemble the XCFramework ------------------------------------------------

XCF="$OUT_DIR/Vokra.xcframework"
rm -rf "$XCF"
echo "build-ios: xcodebuild -create-xcframework → $XCF"
xcodebuild -create-xcframework \
    -library "$LIB_DEVICE"   -headers "$HDR_DIR" \
    -library "$LIB_SIM_FAT"  -headers "$HDR_DIR" \
    -output "$XCF"

# xcodebuild has been observed to return 0 even when a plug-in load failure
# (IDESimulatorFoundation etc.) prevents the output from being written. Assert
# the artifact + both slices are actually on disk before declaring success.
for expected in "$XCF" "$XCF/ios-arm64/libvokra.a" "$XCF/ios-arm64_x86_64-simulator/libvokra.a"; do
    if [ ! -e "$expected" ]; then
        echo "error: xcodebuild returned 0 but $expected is missing." >&2
        echo "       If IDESimulatorFoundation warnings appeared above, run" >&2
        echo "       'xcodebuild -runFirstLaunch' and reinstall Xcode components." >&2
        exit 1
    fi
done

echo "build-ios: OK — $XCF (iOS $MIN_VERSION+, panic=unwind, metal on, cuda off)"
