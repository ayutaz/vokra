#!/usr/bin/env bash
# build-godot-gdextension.sh — build the Vokra Godot 4.x GDExtension package
# (M3-11-T11 + T12, FR-TL-04, ADR-0011 §D8).
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
#       [--target ${TARGET_TRIPLE}] to produce the cdylib for either the
#       CURRENT host platform (default) or the explicitly requested cross
#       target.
#       Linux    x86_64          → libvokra_godot.so
#       macOS    arm64 / x86_64  → libvokra_godot.dylib
#       Windows  x86_64          → vokra_godot.dll   (bash under MinGW/MSYS/
#                                                       Cygwin, or MSVC cross
#                                                       from Linux via
#                                                       cargo-xwin/cross)
#       Android  aarch64         → libvokra_godot.so (NDK-driven, T12 CI
#                                                       matrix invokes with
#                                                       CC/AR set)
#   (2) Assemble the AssetLib package skeleton
#       (ADR-0011 §D9 layout) at dist/godot/vokra-godot/.
#   (3) With --pack, produce dist/godot/vokra-godot-<version>.zip
#       (deterministic mtime / owner / order for reproducible builds).
#
# T12 crossbuild matrix (ADR-0011 §D4 / D9):
#   Setting ${TARGET_TRIPLE} switches this script into cross-target mode:
#     TARGET_TRIPLE=x86_64-apple-darwin       ← macOS Intel
#     TARGET_TRIPLE=aarch64-apple-darwin      ← macOS Apple Silicon
#     TARGET_TRIPLE=x86_64-unknown-linux-gnu  ← Linux x64
#     TARGET_TRIPLE=x86_64-pc-windows-msvc    ← Windows x64  (Rust cross)
#     TARGET_TRIPLE=aarch64-linux-android     ← Android arm64 (NDK required)
#   In cross mode the cdylib is:
#     (a) fetched from `${GODOT_CRATE}/target/${TARGET_TRIPLE}/release/…`
#         (cargo's per-target output tree), and
#     (b) copied *both* into the AssetLib layout under
#         `dist/godot/vokra-godot/addons/vokra/bin/<platform>/<arch>/`  AND
#         into the flat per-target staging tree
#         `integrations/vokra-godot/pkg/lib/${TARGET_TRIPLE}/`  so the CI
#         matrix (T16 `godot-crossbuild.yml`) can upload one artifact per
#         target without inspecting the nested AssetLib path.
#   Cross-target builds do NOT auto-install `rustup target add` — CI does that
#   in a preceding step, and local dev is expected to have added the target
#   already (FR-EX-08 spirit: no silent behavior on missing toolchain).
#
# Follow-up tickets:
#   - T13: `vokra.gdextension` fully-populated with all 4-platform library
#     paths (template lives in integrations/vokra-godot/vokra.gdextension —
#     this script SUBSTITUTES that template into dist/). [landed alongside T11]
#   - T16: CI required check.  → .github/workflows/godot-crossbuild.yml
#   - T17: CD release train + Godot AssetLib auto-publish.
#          → .github/workflows/release.yml `godot-package-release` job
#   - T18: NVIDIA-runtime non-bundle scan (mirror
#     scripts/check-unity-package-no-nvidia.sh).
#     → scripts/compliance/check-godot-package-no-nvidia.sh
#
# Usage:
#   bash scripts/build-godot-gdextension.sh           # host-only cdylib sync
#   bash scripts/build-godot-gdextension.sh --pack    # + assemble zip
#   bash scripts/build-godot-gdextension.sh --no-build  # skip cargo build
#   TARGET_TRIPLE=aarch64-linux-android bash scripts/build-godot-gdextension.sh
#       # cross-build for Android arm64 (assumes NDK / cargo cross is
#       # pre-configured in the caller's env)
#   bash scripts/build-godot-gdextension.sh -h | --help
#
# Exit code: 0 on OK, non-zero on build / sync / pack failure. Unknown flag
# = exit 2 (usage error). Unknown ${TARGET_TRIPLE} = exit 1.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GODOT_CRATE="$ROOT/integrations/vokra-godot"
DIST_DIR="$ROOT/dist/godot"
ADDONS_DIR="$DIST_DIR/vokra-godot/addons/vokra"
BIN_DIR="$ADDONS_DIR/bin"
PKG_STAGE_ROOT="$GODOT_CRATE/pkg/lib"

DO_PACK=0
DO_BUILD=1

usage() {
    sed -n '2,75p' "$0"
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

# ---- Resolve triple → cdylib basename + AssetLib subdir -------------------
#
# T12 crossbuild matrix. When ${TARGET_TRIPLE} is set we treat it as the
# authoritative selector and skip host detection entirely; every supported
# triple is listed explicitly so a typo (`x86_64-linux-gnu` instead of
# `x86_64-unknown-linux-gnu`) fails loudly rather than silent-CPU-fallbacks.
#
# When ${TARGET_TRIPLE} is empty we fall back to `uname -s`/`uname -m` host
# detection, which is the T11 baseline behavior (dev iteration on the
# author's machine, no CI matrix indirection).
#
# Layout invariant (both modes):
#   $DEST_SUBDIR  = "<platform>/<arch>"  (AssetLib bin/ tree; ADR-0011 §D9)
#   $SRC_NAME     = "libvokra_godot.so" | "libvokra_godot.dylib" | "vokra_godot.dll"

TARGET_TRIPLE="${TARGET_TRIPLE:-}"

if [ -n "$TARGET_TRIPLE" ]; then
    case "$TARGET_TRIPLE" in
        x86_64-apple-darwin)
            SRC_NAME="libvokra_godot.dylib";  DEST_SUBDIR="macos/x86_64"     ;;
        aarch64-apple-darwin)
            SRC_NAME="libvokra_godot.dylib";  DEST_SUBDIR="macos/arm64"      ;;
        x86_64-unknown-linux-gnu)
            SRC_NAME="libvokra_godot.so";     DEST_SUBDIR="linux/x86_64"     ;;
        x86_64-pc-windows-msvc)
            SRC_NAME="vokra_godot.dll";       DEST_SUBDIR="windows/x86_64"   ;;
        aarch64-linux-android)
            SRC_NAME="libvokra_godot.so";     DEST_SUBDIR="android/arm64-v8a";;
        *)
            echo "build-godot-gdextension: unsupported TARGET_TRIPLE: $TARGET_TRIPLE" >&2
            echo "                        Supported (ADR-0011 §D4):" >&2
            echo "                          x86_64-apple-darwin" >&2
            echo "                          aarch64-apple-darwin" >&2
            echo "                          x86_64-unknown-linux-gnu" >&2
            echo "                          x86_64-pc-windows-msvc" >&2
            echo "                          aarch64-linux-android" >&2
            echo "                        iOS / Web are M4+ (ADR-0011 §D4)." >&2
            exit 1
            ;;
    esac
    echo "build-godot-gdextension: crossbuild mode (TARGET_TRIPLE=$TARGET_TRIPLE)"
    # Cargo places cross-target artifacts under target/<triple>/release/, not
    # the top-level target/release/. Diverging tree = separate SRC location.
    SRC="$GODOT_CRATE/target/$TARGET_TRIPLE/release/$SRC_NAME"
else
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
            echo "                        Set TARGET_TRIPLE=… for cross builds." >&2
            exit 1
            ;;
    esac
    SRC="$GODOT_CRATE/target/release/$SRC_NAME"
fi

# ---- Build the Rust cdylib -----------------------------------------------
if [ "$DO_BUILD" -eq 1 ]; then
    if [ -n "$TARGET_TRIPLE" ]; then
        echo "== cargo build --release --target ${TARGET_TRIPLE} (--manifest-path integrations/vokra-godot/Cargo.toml) =="
        ( cd "$GODOT_CRATE" && cargo build --release --target "$TARGET_TRIPLE" )
    else
        echo "== cargo build --release (--manifest-path integrations/vokra-godot/Cargo.toml) =="
        ( cd "$GODOT_CRATE" && cargo build --release )
    fi
fi

if [ ! -f "$SRC" ]; then
    echo "build-godot-gdextension: FAIL expected cdylib not found: $SRC" >&2
    echo "                        (rerun without --no-build to compile it)" >&2
    if [ -n "$TARGET_TRIPLE" ]; then
        echo "                        Cross-target hint: has rustup added" >&2
        echo "                        the target?  rustup target add $TARGET_TRIPLE" >&2
    fi
    exit 1
fi

# ---- Sync into AssetLib package skeleton ---------------------------------
echo "== assemble AssetLib package skeleton at dist/godot/vokra-godot/ =="
DEST_DIR="$BIN_DIR/$DEST_SUBDIR"
DEST="$DEST_DIR/$SRC_NAME"

mkdir -p "$DEST_DIR"
cp -f "$SRC" "$DEST"
echo "build-godot-gdextension: synced AssetLib slot $DEST_SUBDIR"
echo "                        src : $SRC"
echo "                        dst : $DEST"

# ---- Also stage into pkg/lib/${TARGET_TRIPLE}/ (T12 flat matrix output) --
# The CI matrix in .github/workflows/godot-crossbuild.yml uploads one
# artifact per matrix leg by tarring `pkg/lib/${TARGET_TRIPLE}/` — a flat
# tree keyed off the Rust target triple, sibling to the AssetLib layout.
# For the host-detected path we key off the mapped platform slot instead of
# a synthetic triple so both modes have a deterministic staging path.
if [ -n "$TARGET_TRIPLE" ]; then
    PKG_STAGE_DIR="$PKG_STAGE_ROOT/$TARGET_TRIPLE"
else
    # Host-mode fallback stamp — use DEST_SUBDIR with `/` → `_` normalization
    # so the path is a single directory component (e.g., "macos_arm64").
    PKG_STAGE_KEY="host-$(printf '%s' "$DEST_SUBDIR" | tr '/' '_')"
    PKG_STAGE_DIR="$PKG_STAGE_ROOT/$PKG_STAGE_KEY"
fi
mkdir -p "$PKG_STAGE_DIR"
cp -f "$SRC" "$PKG_STAGE_DIR/$SRC_NAME"
echo "build-godot-gdextension: synced pkg stage $PKG_STAGE_DIR/$SRC_NAME"

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

# ---- fetch-demo-models.sh (M3-11 addons/vokra Godot mirror) --------------
# Ship the MIT-only demo weight fetcher inside the AssetLib addon tree so
# consumers can run `bash addons/vokra/fetch-demo-models.sh` from their
# Godot project root (the demo GDScripts in `demos/asr_demo/main.gd:12`
# and `demos/tts_demo/main.gd:11` document that exact invocation).
#
# Mirrors bindings/unity/com.vokra.unity/Samples~/VadAsrTts/Scripts/fetch-demo-models.sh
# (Unity precedent) with Godot-flavored destination + filenames — see the
# script header for the divergence points.
#
# `cp -p` preserves the source file's exec bit so consumers don't have to
# `chmod +x` after unzip. Zero-dep: pure bash, no new tooling in the
# packaging path.
FETCH_SCRIPT_SRC="$GODOT_CRATE/addons/vokra/fetch-demo-models.sh"
if [ -f "$FETCH_SCRIPT_SRC" ]; then
    cp -p -f "$FETCH_SCRIPT_SRC" "$ADDONS_DIR/fetch-demo-models.sh"
    # Belt + braces: some cp implementations (e.g., BSD `cp -p` without
    # source exec bit; older busybox variants) don't materialize the exec
    # bit. Explicitly stamp it so `bash addons/vokra/fetch-demo-models.sh`
    # AND `./addons/vokra/fetch-demo-models.sh` both work post-unzip.
    chmod +x "$ADDONS_DIR/fetch-demo-models.sh"
else
    echo "build-godot-gdextension: WARN fetch-demo-models.sh not found at $FETCH_SCRIPT_SRC" >&2
    echo "                        (M3-11 addons/vokra/fetch-demo-models.sh is missing;" >&2
    echo "                        demos will fail to bootstrap weights.)" >&2
fi

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
