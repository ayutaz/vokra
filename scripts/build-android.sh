#!/usr/bin/env bash
# build-android.sh — cross-compile the Vokra Android JNI library (M2-11-T07).
#
# Ticket: M2-11-T07 (Android cross-build). Basis: NFR-RL-04 (Android
# StreamingAssets → persistentDataPath expansion, real filesystem path so mmap
# via C ABI works), FR-EX-08 (no silent fallback — CPU-only baseline SKU for
# the shipped UPM Package; Metal/CUDA are host-desktop optional features that
# never apply to Android). Design in docs/adr/0007-unity-official-plugin.md.
#
# Produces:
#   target/aarch64-linux-android/release/libvokra.so
#   bindings/unity/com.vokra.unity/Plugins/Android/libs/arm64-v8a/libvokra.so
#     (staged into the UPM Package tree; the accompanying .meta is
#      hand-authored with a fixed GUID and lives alongside in git.)
#
# CPU-only build (--no-default-features): NO Metal, NO CUDA. The root
# Cargo.lock invariant (root Cargo.lock has vokra-* only) is preserved because
# adding a rustup target does not touch dependency resolution — no new crates
# enter the graph.
#
# Toolchain: Android NDK r25+ (LLVM/clang). API level 24 (Android 7.0) is the
# floor per Unity 2022.3 LTS minimum supported Android target.
#
# Usage:
#   ANDROID_NDK_HOME=/path/to/ndk scripts/build-android.sh
#
# Env:
#   ANDROID_NDK_HOME   REQUIRED. Path to the Android NDK root. Script exits 1
#                     with "ANDROID_NDK_HOME must be set" when unset or empty.
#
# Exit codes: 0 = success; 1 = ANDROID_NDK_HOME unset or any build failure.

set -euo pipefail

# --- guardrail: ANDROID_NDK_HOME required ------------------------------------

if [ -z "${ANDROID_NDK_HOME:-}" ]; then
    echo "error: ANDROID_NDK_HOME must be set" >&2
    echo "       (Android NDK r25+ required; see docs/adr/0007-unity-official-plugin.md)" >&2
    exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRIPLE="aarch64-linux-android"
API=24

# --- resolve NDK host tag (darwin-x86_64 / linux-x86_64) ---------------------

case "$(uname -s)" in
    Darwin) NDK_HOST="darwin-x86_64" ;;
    Linux)  NDK_HOST="linux-x86_64" ;;
    *)
        echo "error: unsupported host OS $(uname -s) for Android cross-build" >&2
        exit 1
        ;;
esac

NDK_BIN="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/$NDK_HOST/bin"
CLANG="$NDK_BIN/${TRIPLE}${API}-clang"

if [ ! -x "$CLANG" ]; then
    echo "error: NDK clang not found or not executable: $CLANG" >&2
    echo "       Check ANDROID_NDK_HOME=$ANDROID_NDK_HOME and NDK version (r25+)." >&2
    exit 1
fi

# --- rustup target (idempotent) ----------------------------------------------

installed="$(rustup target list --installed 2>/dev/null || true)"
if ! printf '%s\n' "$installed" | grep -q "^$TRIPLE$"; then
    echo "build-android: installing rustup target $TRIPLE"
    rustup target add "$TRIPLE"
fi

# --- CC + linker exports for cargo/cc-rs -------------------------------------
# cc-rs looks at CC_<target> (with hyphens preserved); cargo looks at
# CARGO_TARGET_<target-uppercased-underscored>_LINKER. Bash env-var names
# cannot contain hyphens, so we use `env` on the cargo invocation for CC_ and
# a plain export for the CARGO_TARGET_..._LINKER variant.

TRIPLE_UPPER="AARCH64_LINUX_ANDROID"
export "CARGO_TARGET_${TRIPLE_UPPER}_LINKER=$CLANG"

# --- CPU-only build ----------------------------------------------------------
# --no-default-features guarantees no Metal (macOS) / no CUDA (desktop) code
# reaches the Android artifact even if defaults change upstream.

echo "build-android: cargo build -p vokra-capi --release --target $TRIPLE"
(
    cd "$ROOT"
    env "CC_${TRIPLE}=$CLANG" \
        cargo build -p vokra-capi --release \
        --target "$TRIPLE" \
        --no-default-features
)

LIB_SRC="$ROOT/target/$TRIPLE/release/libvokra.so"
if [ ! -f "$LIB_SRC" ]; then
    echo "error: expected shared library not found: $LIB_SRC" >&2
    exit 1
fi

# --- stage into the UPM Package tree ----------------------------------------

DEST_DIR="$ROOT/bindings/unity/com.vokra.unity/Plugins/Android/libs/arm64-v8a"
mkdir -p "$DEST_DIR"
cp "$LIB_SRC" "$DEST_DIR/libvokra.so"

echo "build-android: OK — $DEST_DIR/libvokra.so (API $API, CPU-only, arm64-v8a)"
