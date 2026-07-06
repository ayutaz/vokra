#!/usr/bin/env bash
# collect-ios-lib.sh — extract iOS device staticlib from Vokra.xcframework into
# the Unity UPM package (M2-11-T05).
#
# Consumes a Vokra.xcframework built by scripts/build-ios.sh (M2-02) or
# downloaded from the CI artifact `Vokra-xcframework` / GitHub Release asset
# `Vokra.xcframework.zip`. Extracts the ios-arm64 device slice's libvokra.a
# into bindings/unity/com.vokra.unity/Plugins/iOS/libvokra.a, then authors the
# accompanying PluginImporter .meta file.
#
# Plan alignment (M2-11 plan §3 T05, §5 R4):
#   - Runs scripts/verify-ios-xcframework.sh FIRST so a spec drift from M2-02
#     (e.g. LibraryIdentifier rename) fails loudly before we blindly extract.
#   - Asserts extracted libvokra.a is a SINGLE arm64 slice via `lipo -info`;
#     App Store rejects fat archives on device-only slices.
#   - Simulator slice `ios-arm64_x86_64-simulator/libvokra-sim.a` is NOT
#     copied (device-only for TestFlight / App Store; NFR-RL-03).
#   - .meta declares iOS enabled=1, AddToEmbeddedBinaries=false,
#     FrameworkDependencies=Metal;Accelerate (matches M2-02 build spec:
#     Metal backend ON for iOS, Accelerate for BLAS/FFT).
#   - Fixed guid so re-runs are idempotent; existing .meta is preserved to
#     keep the guid stable if Unity Editor has adjusted importer defaults.
#
# Usage:
#   scripts/collect-ios-lib.sh [<path/to/Vokra.xcframework>]
#
# Default XCF path: build/ios/Vokra.xcframework (matches build-ios.sh output).
#
# Exit 0 = libvokra.a + libvokra.a.meta staged; non-zero on the first failure.

set -euo pipefail

XCF="${1:-build/ios/Vokra.xcframework}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_DIR="$ROOT/bindings/unity/com.vokra.unity/Plugins/iOS"
DEST_LIB="$DEST_DIR/libvokra.a"
DEST_META="$DEST_LIB.meta"

# Fixed guid — hand-generated once, MUST NOT change across re-runs (Unity
# references plugins by guid; churn breaks scene / prefab references). Do not
# regenerate. If Unity Editor rewrites the .meta with a different guid the
# operator MUST commit that .meta as-is; this script only writes when absent.
LIBVOKRA_A_META_GUID="7c9f4a2b8d6e4c1a9b3f5d7e2a8c6f4d"

if [ ! -d "$XCF" ]; then
    echo "collect-ios-lib: FAIL missing XCFramework at $XCF" >&2
    echo "  Build it first with scripts/build-ios.sh or download the CI artifact" >&2
    echo "  Vokra-xcframework and unzip to $XCF." >&2
    exit 1
fi

echo "== collect-ios-lib $XCF -> $DEST_LIB =="

# --- (a) verify the XCFramework before extraction (R4 mitigation) --------------
# If M2-02's XCFramework layout has drifted, verify-ios-xcframework.sh will
# exit non-zero here; -e propagates and we do not touch DEST_LIB.
"$ROOT/scripts/verify-ios-xcframework.sh" "$XCF"

# --- (b) extract the ios-arm64 device slice ------------------------------------
SRC_LIB="$XCF/ios-arm64/libvokra.a"
if [ ! -f "$SRC_LIB" ]; then
    echo "collect-ios-lib: FAIL missing $SRC_LIB (device slice not present)" >&2
    exit 1
fi

mkdir -p "$DEST_DIR"
cp "$SRC_LIB" "$DEST_LIB"

# --- (c) assert single arm64 arch via lipo -info (R4 mitigation) ---------------
# App Store submission rejects fat device slices; the device libvokra.a MUST be
# arm64-only. `lipo -info` prints e.g.:
#   Non-fat file: … is architecture: arm64
#   Architectures in the fat file: … are: arm64 armv7 …
# We require exactly the single-arch, non-fat form with arm64.
LIPO_INFO="$(lipo -info "$DEST_LIB" 2>&1)"
if ! printf '%s' "$LIPO_INFO" | grep -qE 'Non-fat file:.*architecture: arm64$'; then
    echo "collect-ios-lib: FAIL $DEST_LIB is not a single arm64 slice" >&2
    printf '  lipo -info: %s\n' "$LIPO_INFO" >&2
    echo "  M2-02 device slice must be arm64-only (NFR-RL-03, App Store)." >&2
    rm -f "$DEST_LIB"
    exit 1
fi
echo "  lipo -info: single arm64 slice OK"

# --- (d) author the PluginImporter .meta (idempotent) --------------------------
# Preserve an existing .meta so Unity Editor-authored importer edits and the
# original guid survive. Only write when absent.
if [ -f "$DEST_META" ]; then
    echo "  .meta already present: $DEST_META (preserved; guid unchanged)"
else
    cat >"$DEST_META" <<META
fileFormatVersion: 2
guid: $LIBVOKRA_A_META_GUID
PluginImporter:
  externalObjects: {}
  serializedVersion: 2
  iconMap: {}
  executionOrder: {}
  defineConstraints: []
  isPreloaded: 0
  isOverridable: 0
  isExplicitlyReferenced: 0
  validateReferences: 1
  platformData:
  - first:
      : Any
    second:
      enabled: 0
      settings:
        Exclude Android: 1
        Exclude Editor: 1
        Exclude Linux64: 1
        Exclude OSXUniversal: 1
        Exclude Win: 1
        Exclude Win64: 1
        Exclude iOS: 0
  - first:
      Any:
    second:
      enabled: 0
      settings: {}
  - first:
      Editor: Editor
    second:
      enabled: 0
      settings:
        DefaultValueInitialized: true
  - first:
      iPhone: iOS
    second:
      enabled: 1
      settings:
        AddToEmbeddedBinaries: false
        CPU: ARM64
        CompileFlags:
        FrameworkDependencies: Metal;Accelerate
  userData:
  assetBundleName:
  assetBundleVariant:
META
    echo "  wrote .meta: $DEST_META (guid $LIBVOKRA_A_META_GUID)"
fi

# --- (e) summary ---------------------------------------------------------------
BYTES="$(stat -f %z "$DEST_LIB" 2>/dev/null || stat -c %s "$DEST_LIB")"
echo "collect-ios-lib: staged libvokra.a ($(printf '%d' "$BYTES") bytes) + .meta"
