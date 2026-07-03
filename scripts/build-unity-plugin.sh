#!/usr/bin/env bash
# build-unity-plugin.sh — place the Vokra native library into the Unity demo
# (M0-10-T02).
#
# M0 SCOPE: this only builds the `vokra-capi` cdylib and copies it into
# examples/unity-demo/Assets/Plugins/<platform>/. Generating a distributable
# Unity Package (.unitypackage / UPM) and release automation (NFR-MT-08,
# FR-API-04 official plugin) are v0.5 scope — NOT done here.
#
# No cross-compilation: run this ON each target OS to place that OS's library
# (macOS = libvokra.dylib, Linux = libvokra.so, Windows = vokra.dll via Git Bash;
# or fetch vokra.dll from the M0-01 CI Windows runner artifact).
#
# Exit code: 0 = library placed, non-zero = build/copy failure.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REL="$ROOT/target/release"
PLUGINS="$ROOT/examples/unity-demo/Assets/Plugins"

case "$(uname -s)" in
    Darwin)
        SRC="$REL/libvokra.dylib"
        DEST_DIR="$PLUGINS/macOS"
        DEST="$DEST_DIR/libvokra.dylib"
        ;;
    Linux)
        SRC="$REL/libvokra.so"
        DEST_DIR="$PLUGINS/Linux/x86_64"
        DEST="$DEST_DIR/libvokra.so"
        ;;
    MINGW* | MSYS* | CYGWIN*)
        SRC="$REL/vokra.dll"
        DEST_DIR="$PLUGINS/Windows/x86_64"
        DEST="$DEST_DIR/vokra.dll"
        ;;
    *)
        echo "build-unity-plugin: unsupported OS $(uname -s)" >&2
        exit 1
        ;;
esac

echo "== cargo build --release -p vokra-capi =="
( cd "$ROOT" && cargo build --release -p vokra-capi )

if [ ! -f "$SRC" ]; then
    echo "build-unity-plugin: FAIL expected library not found: $SRC" >&2
    exit 1
fi

mkdir -p "$DEST_DIR"
cp -f "$SRC" "$DEST"
echo "build-unity-plugin: placed $DEST"
echo "build-unity-plugin: OK"
echo
echo "Next: open examples/unity-demo in Unity and set the plugin's platform import"
echo "settings in the Inspector (target OS/CPU). See Assets/Plugins/README.md."
