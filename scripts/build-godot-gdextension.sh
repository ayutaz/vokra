#!/usr/bin/env bash
# build-godot-gdextension.sh — build the Vokra Godot 4.x GDExtension package
# (M3-11-T11, FR-TL-04, ADR-0011 §D8).
#
# =============================================================================
#  CANONICAL DISTRIBUTION = CD  (NFR-MT-08 / FR-API-05)
#
#  正規発行は CD 経由 (.github/workflows/release.yml の godot-package-release
#  job — T17)。Local runs of this script are for dev iteration only
#  (host-OS-only cdylib sync + optional local zip smoke-test). The
#  release-signed vokra-godot-<version>.zip that consumers install via Godot
#  AssetLib is produced by CI, not by hand.
# =============================================================================
#
# What this script does (T11 initial + T12 crossbuild expansion):
#   (1) cargo build --release --manifest-path integrations/vokra-godot/Cargo.toml
#       to produce the cdylib for the CURRENT host platform.
#       Linux  x86_64  → libvokra_godot.so
#       macOS  arm64   → libvokra_godot.dylib  (universal in T12 via lipo)
#       macOS  x86_64  → libvokra_godot.dylib
#       Windows        → vokra_godot.dll         (bash under MinGW/MSYS/Cygwin)
#   (2) Assemble the AssetLib package skeleton
#       (ADR-0011 §D9 layout) at dist/godot/vokra-godot/.
#   (3) With --pack, produce dist/godot/vokra-godot-<version>.zip
#       (deterministic mtime / owner / order for reproducible builds).
#
# TODO (later tickets):
#   - T12: cross-target builds for the missing 3 host platforms
#     (Windows x86_64 msvc / Linux x86_64 gnu / Android aarch64) driven by
#     CI matrix. Placeholder branches below reject unknown hosts explicitly
#     instead of silent-CPU-fallback-style graceful degradation
#     (FR-EX-08 spirit applied to the build system).
#   - T13: `vokra.gdextension` fully-populated with all 4-platform library
#     paths (template lives in integrations/vokra-godot/vokra.gdextension —
#     this script SUBSTITUTES that template into dist/).
#   - T16: CI required check.
#   - T17: CD release train + Godot AssetLib auto-publish.
#   - T18: NVIDIA-runtime non-bundle scan (mirror
#     scripts/check-unity-package-no-nvidia.sh).
#
# Usage:
#   bash scripts/build-godot-gdextension.sh           # host-only cdylib sync
#   bash scripts/build-godot-gdextension.sh --pack    # + assemble zip
#   bash scripts/build-godot-gdextension.sh --no-build  # skip cargo build
#   bash scripts/build-godot-gdextension.sh -h | --help
#
# Exit code: 0 on OK, non-zero on build / sync / pack failure. Unknown flag
# = exit 2 (usage error).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GODOT_CRATE="$ROOT/integrations/vokra-godot"
DIST_DIR="$ROOT/dist/godot"
ADDONS_DIR="$DIST_DIR/vokra-godot/addons/vokra"
BIN_DIR="$ADDONS_DIR/bin"

DO_PACK=0
DO_BUILD=1

usage() {
    sed -n '2,42p' "$0"
}

for arg in "$@"; do
    case "$arg" in
        --pack)     DO_PACK=1 ;;
        --no-build) DO_BUILD=0 ;;
        -h|--help)  usage; exit 0 ;;
        *)
            echo "build-godot-gdextension: unknown flag: $arg" >&2
            usage >&2
            exit 2
            ;;
    esac
done

echo "== build-godot-gdextension (canonical publish = CD, NFR-MT-08) =="

# ---- Detect host triple → cdylib basename + AssetLib subdir ---------------
case "$(uname -s)" in
    Darwin)
        SRC_NAME="libvokra_godot.dylib"
        ARCH="$(uname -m)"
        case "$ARCH" in
            arm64|aarch64) DEST_SUBDIR="macos/arm64" ;;
            x86_64)        DEST_SUBDIR="macos/x86_64" ;;
            *)
                echo "build-godot-gdextension: unsupported macOS arch: $ARCH" >&2
                exit 1
                ;;
        esac
        ;;
    Linux)
        SRC_NAME="libvokra_godot.so"
        DEST_SUBDIR="linux/x86_64"
        ;;
    MINGW*|MSYS*|CYGWIN*)
        SRC_NAME="vokra_godot.dll"
        DEST_SUBDIR="windows/x86_64"
        ;;
    *)
        echo "build-godot-gdextension: unsupported host OS: $(uname -s)" >&2
        echo "                        Vokra Godot binding targets Windows /" >&2
        echo "                        macOS / Linux / Android only (ADR-0011 §D4)." >&2
        echo "                        Cross-target support is T12; iOS/Web are M4+." >&2
        exit 1
        ;;
esac

SRC="$GODOT_CRATE/target/release/$SRC_NAME"

# ---- Build the Rust cdylib -----------------------------------------------
if [ "$DO_BUILD" -eq 1 ]; then
    echo "== cargo build --release (--manifest-path integrations/vokra-godot/Cargo.toml) =="
    ( cd "$GODOT_CRATE" && cargo build --release )
fi

if [ ! -f "$SRC" ]; then
    echo "build-godot-gdextension: FAIL expected cdylib not found: $SRC" >&2
    echo "                        (rerun without --no-build to compile it)" >&2
    exit 1
fi

# ---- Sync into AssetLib package skeleton ---------------------------------
echo "== assemble AssetLib package skeleton at dist/godot/vokra-godot/ =="
DEST_DIR="$BIN_DIR/$DEST_SUBDIR"
DEST="$DEST_DIR/$SRC_NAME"

mkdir -p "$DEST_DIR"
cp -f "$SRC" "$DEST"
echo "build-godot-gdextension: synced $DEST_SUBDIR"
echo "                        src : $SRC"
echo "                        dst : $DEST"

# ---- .gdextension + LICENSE / NOTICE -------------------------------------
# Copy the template `vokra.gdextension` that ships with the crate. T12
# expansion may render a fresh copy per crossbuild matrix step; for now the
# static template names all four platforms and Godot silently ignores paths
# whose file is missing from the current package build.
TEMPLATE="$GODOT_CRATE/vokra.gdextension"
if [ -f "$TEMPLATE" ]; then
    cp -f "$TEMPLATE" "$ADDONS_DIR/vokra.gdextension"
else
    echo "build-godot-gdextension: WARN template not found: $TEMPLATE" >&2
    echo "                        (T13 will land it; running against T11 stub)" >&2
fi

# LICENSE (Apache-2.0) + NOTICE + README fall back to the repo root when
# the crate doesn't ship its own. In the release train the CI job assembles
# a package-scoped NOTICE with the dependency roll-up; the local dev build
# ships the root files verbatim so the AssetLib package is never license-less.
for f in LICENSE NOTICE README.md; do
    if [ -f "$GODOT_CRATE/$f" ]; then
        cp -f "$GODOT_CRATE/$f" "$ADDONS_DIR/$f"
    elif [ -f "$ROOT/$f" ]; then
        cp -f "$ROOT/$f" "$ADDONS_DIR/$f"
    else
        echo "build-godot-gdextension: WARN $f not found (in crate or repo root)" >&2
    fi
done

# ---- Optional: assemble the AssetLib zip (dev smoke-test) ----------------
if [ "$DO_PACK" -eq 1 ]; then
    # Extract version from the crate's Cargo.toml without a TOML parser
    # (keep host requirements minimal — zero-dep spirit).
    CARGO_TOML="$GODOT_CRATE/Cargo.toml"
    VERSION="$(
        awk -F'"' '/^[[:space:]]*version[[:space:]]*=/ { print $2; exit }' "$CARGO_TOML"
    )"
    if [ -z "$VERSION" ]; then
        echo "build-godot-gdextension: FAIL could not parse version from $CARGO_TOML" >&2
        exit 1
    fi

    ZIP="$DIST_DIR/vokra-godot-${VERSION}.zip"

    # Reproducible-build stamp — same idiom as build-unity-plugin.sh.
    if [ -z "${SOURCE_DATE_EPOCH:-}" ]; then
        if command -v git >/dev/null 2>&1 && \
           git -C "$ROOT" rev-parse --show-toplevel >/dev/null 2>&1; then
            SOURCE_DATE_EPOCH="$(git -C "$ROOT" log -1 --pretty=%ct 2>/dev/null || echo 0)"
        else
            SOURCE_DATE_EPOCH=0
        fi
        export SOURCE_DATE_EPOCH
    fi

    echo "== packing AssetLib zip =="
    echo "   version : $VERSION"
    echo "   zip     : $ZIP"
    echo "   sde     : $SOURCE_DATE_EPOCH (reproducible-build stamp)"

    rm -f "$ZIP"
    if command -v zip >/dev/null 2>&1; then
        # -X strips extra file attributes; -q quiet.
        ( cd "$DIST_DIR" && zip -X -qr "$(basename "$ZIP")" vokra-godot )
    else
        # Fallback for hosts without zip (e.g. minimal Alpine): produce a
        # deterministic tar instead. The AssetLib requires zip; CI hosts
        # ALWAYS have it, so this is dev-only. Fail loudly so the CD path
        # never silently falls back.
        echo "build-godot-gdextension: FAIL 'zip' is not installed on this host" >&2
        echo "                        (Godot AssetLib requires zip format)" >&2
        exit 1
    fi

    if [ ! -f "$ZIP" ]; then
        echo "build-godot-gdextension: FAIL zip not produced: $ZIP" >&2
        exit 1
    fi

    # sha256 alongside the zip for downstream audit.
    if command -v shasum >/dev/null 2>&1; then
        ( cd "$DIST_DIR" && shasum -a 256 "$(basename "$ZIP")" > "${ZIP}.sha256" )
    elif command -v sha256sum >/dev/null 2>&1; then
        ( cd "$DIST_DIR" && sha256sum "$(basename "$ZIP")" > "${ZIP}.sha256" )
    fi

    echo "build-godot-gdextension: packed $ZIP"
fi

echo "build-godot-gdextension: OK"
echo
echo "Reminder: canonical distribution is the release-signed .zip produced by"
echo "          .github/workflows/release.yml godot-package-release job"
echo "          (NFR-MT-08, T17). Local runs are for dev iteration only."
