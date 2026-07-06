#!/usr/bin/env bash
# build-unity-plugin.sh — sync the Vokra native library into the UPM package
# (M2-11-T03).
#
# =============================================================================
#  CANONICAL DISTRIBUTION = CD  (NFR-MT-08 / FR-API-04)
#
#  正規発行は CD 経由。Local runs of this script are for dev iteration only
#  (host-OS-only cdylib sync + optional local tarball smoke-test). The
#  release-signed com.vokra.unity-<version>.tgz that consumers install is
#  produced by .github/workflows/release.yml (`unity-package-release` job),
#  not by hand-invoked `--pack`.
# =============================================================================
#
# Modes:
#   (default)      Build vokra-capi and sync the host cdylib into
#                  bindings/unity/com.vokra.unity/Plugins/<current-os>/.
#   --pack         Also assemble the UPM tarball
#                  bindings/unity/com.vokra.unity-<version>.tgz
#                  via `tar -czf ... -C bindings/unity com.vokra.unity`.
#   --legacy-m0    Sync into the M0 destination
#                  examples/unity-demo/Assets/Plugins/<current-os>/
#                  instead of the UPM package. Preserved for the
#                  examples/unity-demo dev loop (M0-10). Ignored by --pack.
#   --no-build     Skip `cargo build`; assume target/release/ is fresh.
#   -h | --help    Show usage and exit 0.
#
# Exit code: 0 = OK, non-zero = build / sync / pack failure.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REL="$ROOT/target/release"
UPM_ROOT="$ROOT/bindings/unity"
UPM_PKG_DIR="$UPM_ROOT/com.vokra.unity"
UPM_PLUGINS="$UPM_PKG_DIR/Plugins"
LEGACY_PLUGINS="$ROOT/examples/unity-demo/Assets/Plugins"

DO_PACK=0
DO_LEGACY=0
DO_BUILD=1

usage() {
    sed -n '2,32p' "$0"
}

for arg in "$@"; do
    case "$arg" in
        --pack)      DO_PACK=1 ;;
        --legacy-m0) DO_LEGACY=1 ;;
        --no-build)  DO_BUILD=0 ;;
        -h|--help)   usage; exit 0 ;;
        *)
            echo "build-unity-plugin: unknown flag: $arg" >&2
            usage >&2
            exit 2
            ;;
    esac
done

echo "== build-unity-plugin (canonical publish = CD, NFR-MT-08) =="

# ---- Host triple -> src cdylib + dest subdir + dest filename ---------------
case "$(uname -s)" in
    Darwin)
        SRC="$REL/libvokra.dylib"
        DEST_SUBDIR="macOS"
        DEST_NAME="libvokra.dylib"
        ;;
    Linux)
        SRC="$REL/libvokra.so"
        DEST_SUBDIR="Linux/x86_64"
        DEST_NAME="libvokra.so"
        ;;
    MINGW*|MSYS*|CYGWIN*)
        SRC="$REL/vokra.dll"
        DEST_SUBDIR="Windows/x86_64"
        DEST_NAME="vokra.dll"
        ;;
    *)
        echo "build-unity-plugin: unsupported OS $(uname -s)" >&2
        exit 1
        ;;
esac

# ---- Build vokra-capi cdylib -----------------------------------------------
if [ "$DO_BUILD" -eq 1 ]; then
    echo "== cargo build --release -p vokra-capi =="
    ( cd "$ROOT" && cargo build --release -p vokra-capi )
fi

if [ ! -f "$SRC" ]; then
    echo "build-unity-plugin: FAIL expected library not found: $SRC" >&2
    echo "                   (run without --no-build to compile it)" >&2
    exit 1
fi

# ---- Sync into UPM package (default) or legacy M0 path (--legacy-m0) -------
if [ "$DO_LEGACY" -eq 1 ]; then
    DEST_DIR="$LEGACY_PLUGINS/$DEST_SUBDIR"
    DEST_LABEL="legacy M0 examples/unity-demo"
else
    DEST_DIR="$UPM_PLUGINS/$DEST_SUBDIR"
    DEST_LABEL="UPM com.vokra.unity"
fi

DEST="$DEST_DIR/$DEST_NAME"
mkdir -p "$DEST_DIR"
cp -f "$SRC" "$DEST"
echo "build-unity-plugin: synced $DEST_LABEL <- $SRC"
echo "                    dest: $DEST"

# ---- Optional: assemble the UPM tarball (dev smoke-test) -------------------
if [ "$DO_PACK" -eq 1 ]; then
    if [ "$DO_LEGACY" -eq 1 ]; then
        echo "build-unity-plugin: --pack is UPM-only; ignoring --legacy-m0 for pack step" >&2
    fi

    if [ ! -d "$UPM_PKG_DIR" ]; then
        echo "build-unity-plugin: FAIL UPM skeleton missing at $UPM_PKG_DIR" >&2
        echo "                   (T02 authors the skeleton; run that first)" >&2
        exit 1
    fi

    PKG_JSON="$UPM_PKG_DIR/package.json"
    if [ ! -f "$PKG_JSON" ]; then
        echo "build-unity-plugin: FAIL package.json missing at $PKG_JSON" >&2
        exit 1
    fi

    # Extract version from package.json without depending on jq (keep host
    # requirements minimal — zero-dep spirit). Matches:  "version": "x.y.z-tag"
    VERSION="$(
        awk -F'"' '/^[[:space:]]*"version"[[:space:]]*:/ { print $4; exit }' "$PKG_JSON"
    )"
    if [ -z "$VERSION" ]; then
        echo "build-unity-plugin: FAIL could not parse version from $PKG_JSON" >&2
        exit 1
    fi

    TARBALL="$UPM_ROOT/com.vokra.unity-${VERSION}.tgz"

    # Reproducible-build stamp (R5 mitigation): SOURCE_DATE_EPOCH from HEAD if
    # available, else 0. tar flags below normalize mtimes / ownership / order.
    if [ -z "${SOURCE_DATE_EPOCH:-}" ]; then
        if command -v git >/dev/null 2>&1 && git -C "$ROOT" rev-parse --show-toplevel >/dev/null 2>&1; then
            SOURCE_DATE_EPOCH="$(git -C "$ROOT" log -1 --pretty=%ct 2>/dev/null || echo 0)"
        else
            SOURCE_DATE_EPOCH=0
        fi
        export SOURCE_DATE_EPOCH
    fi

    echo "== packing UPM tarball =="
    echo "   version : $VERSION"
    echo "   tarball : $TARBALL"
    echo "   sde     : $SOURCE_DATE_EPOCH (reproducible-build stamp)"

    # BSD tar (macOS) vs GNU tar (Linux) differ on flags. Detect and branch.
    if tar --version 2>/dev/null | grep -qi 'gnu tar'; then
        tar --sort=name \
            --mtime="@${SOURCE_DATE_EPOCH}" \
            --owner=0 --group=0 --numeric-owner \
            -czf "$TARBALL" \
            -C "$UPM_ROOT" com.vokra.unity
    else
        # bsdtar (macOS): --uid/--gid + --uname/--gname for deterministic owner.
        tar --uid 0 --gid 0 --uname '' --gname '' \
            -czf "$TARBALL" \
            -C "$UPM_ROOT" com.vokra.unity
    fi

    if [ ! -f "$TARBALL" ]; then
        echo "build-unity-plugin: FAIL tarball not produced: $TARBALL" >&2
        exit 1
    fi

    # Emit sha256 alongside the tarball for downstream audit (R5).
    if command -v shasum >/dev/null 2>&1; then
        ( cd "$UPM_ROOT" && shasum -a 256 "$(basename "$TARBALL")" > "${TARBALL}.sha256" )
    elif command -v sha256sum >/dev/null 2>&1; then
        ( cd "$UPM_ROOT" && sha256sum "$(basename "$TARBALL")" > "${TARBALL}.sha256" )
    fi

    echo "build-unity-plugin: packed $TARBALL"
fi

echo "build-unity-plugin: OK"
echo
echo "Reminder: canonical distribution is the release-signed .tgz produced by"
echo "          .github/workflows/release.yml (NFR-MT-08). Local runs are for"
echo "          dev iteration only."
