#!/usr/bin/env bash
# M3-02-T13 / ADR M3-02-spirv-generation §4 (a).
#
# Compile every `crates/vokra-backend-vulkan/kernels/glsl/*.comp` shader with
# `glslc` (Vulkan SDK, developer-side) into a matching
# `crates/vokra-backend-vulkan/kernels/precompiled/*.spv` blob, and emit a
# SHA-256 manifest to the same directory (`SHA256SUMS`) so that
# `crates/vokra-backend-vulkan/src/spirv.rs::verify_pinned_hashes` can pin the
# expected hash of every blob.
#
# This script is a **developer tool** and NOT a runtime dependency. `cargo
# build -p vokra-backend-vulkan` never invokes it (NFR-DS-02 zero-dep + NFR-RL-05
# no CPU-side JIT). CI recompiles by re-running this script and diffing the
# `.spv` output against the committed blobs (T36 follow-up).
#
# Dependencies: bash (>= 4.x), coreutils (`sha256sum` or macOS `shasum -a 256`),
# and `glslc` from the Vulkan SDK. See `scripts/install-vulkan-toolchain.md` for
# install instructions per OS. No Python, no crate, no cargo.
#
# Usage:
#     scripts/compile-vulkan-shaders.sh                # recompile everything
#     scripts/compile-vulkan-shaders.sh gemm_subgroup  # recompile one kernel
#     GLSLC=/opt/vulkan/bin/glslc scripts/compile-vulkan-shaders.sh  # override tool
#
# Exit status:
#     0 — all blobs produced and hashed, SHA256SUMS refreshed
#     1 — glslc missing or reported errors
#     2 — sha256 tool missing
#     3 — GLSL source or precompiled/ directory layout unexpected

set -euo pipefail

# ------------------------------------------------------------------------------
# 1. Locate repository + kernels root.
# ------------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
KERNELS_ROOT="$REPO_ROOT/crates/vokra-backend-vulkan/kernels"
GLSL_DIR="$KERNELS_ROOT/glsl"
SPV_DIR="$KERNELS_ROOT/precompiled"

if [ ! -d "$GLSL_DIR" ]; then
    echo "error: GLSL source directory not found at $GLSL_DIR" >&2
    exit 3
fi
if [ ! -d "$SPV_DIR" ]; then
    echo "error: precompiled/ directory not found at $SPV_DIR" >&2
    echo "       (create it with: mkdir -p '$SPV_DIR')" >&2
    exit 3
fi

# ------------------------------------------------------------------------------
# 2. Resolve glslc + sha256 tool.
# ------------------------------------------------------------------------------

GLSLC_BIN="${GLSLC:-glslc}"
if ! command -v "$GLSLC_BIN" >/dev/null 2>&1; then
    cat >&2 <<EOF
error: glslc not found (tried: $GLSLC_BIN)

The Vulkan SDK provides glslc. To install:
    macOS:   brew install glslang         # then glslc is on PATH as glslangValidator's sibling
             or download the LunarG Vulkan SDK: https://vulkan.lunarg.com/sdk/home#mac
    Ubuntu:  sudo apt install glslang-tools
    Windows: download the LunarG Vulkan SDK: https://vulkan.lunarg.com/sdk/home#windows
             and add %VULKAN_SDK%\Bin to PATH.

Or set GLSLC=/path/to/glslc and rerun.

See scripts/install-vulkan-toolchain.md for detailed per-OS install instructions.
EOF
    exit 1
fi

if command -v sha256sum >/dev/null 2>&1; then
    SHA256_TOOL="sha256sum"
elif command -v shasum >/dev/null 2>&1; then
    # macOS ships `shasum -a 256`
    SHA256_TOOL="shasum -a 256"
else
    echo "error: neither sha256sum (Linux) nor shasum (macOS) is available on PATH" >&2
    exit 2
fi

# ------------------------------------------------------------------------------
# 3. Determine kernel filter (optional argument is a basename to recompile only).
# ------------------------------------------------------------------------------

FILTER="${1:-}"

# ------------------------------------------------------------------------------
# 4. Recompile each *.comp -> *.spv.
# ------------------------------------------------------------------------------

# Bash: enable nullglob so an empty directory becomes an empty loop (no literal
# `*.comp` iteration).
shopt -s nullglob

TARGET_ENV="vulkan1.1"  # Vokra targets Vulkan 1.1+ per M3-02 ADR §T01(c)

COMPILED_COUNT=0
FAILED_COUNT=0

for src in "$GLSL_DIR"/*.comp; do
    base="$(basename "$src" .comp)"
    if [ -n "$FILTER" ] && [ "$base" != "$FILTER" ]; then
        continue
    fi
    dst="$SPV_DIR/${base}.spv"
    # -O2 (or --optimize) is intentionally omitted so the compiled bytecode is
    # a faithful reflection of the source. T14+ can revisit if size/speed
    # matters, but for parity CI the un-optimised form is more debuggable.
    if "$GLSLC_BIN" --target-env="$TARGET_ENV" -o "$dst" "$src"; then
        printf '[ok]  %s -> %s\n' "$src" "$dst"
        COMPILED_COUNT=$((COMPILED_COUNT + 1))
    else
        printf '[err] %s (glslc failed)\n' "$src" >&2
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

if [ "$FAILED_COUNT" -gt 0 ]; then
    echo "error: $FAILED_COUNT shader(s) failed to compile" >&2
    exit 1
fi

if [ "$COMPILED_COUNT" -eq 0 ]; then
    if [ -n "$FILTER" ]; then
        echo "error: filter '$FILTER' matched no .comp under $GLSL_DIR" >&2
        exit 3
    fi
    echo "warning: no .comp sources found under $GLSL_DIR (nothing to do)" >&2
fi

# ------------------------------------------------------------------------------
# 5. Emit SHA256SUMS (over ALL .spv, not just the ones we recompiled — so that
#    a single-kernel rebuild still updates the manifest coherently).
# ------------------------------------------------------------------------------

pushd "$SPV_DIR" >/dev/null

SUMS_FILE="SHA256SUMS"
: > "$SUMS_FILE.new"
for f in *.spv; do
    if [ ! -f "$f" ]; then
        continue
    fi
    # sha256sum output: "<hex>  <name>" — same on shasum -a 256.
    $SHA256_TOOL "$f" >> "$SUMS_FILE.new"
done

# Sort deterministically (name column) so the manifest diff is small when only
# one shader changes.
sort -k 2 "$SUMS_FILE.new" -o "$SUMS_FILE"
rm -f "$SUMS_FILE.new"

popd >/dev/null

# ------------------------------------------------------------------------------
# 6. Emit human-readable summary for the developer to paste into spirv.rs's
#    `expected_sha256_hex` field.
# ------------------------------------------------------------------------------

echo
echo "Generated $COMPILED_COUNT .spv blob(s). SHA-256 manifest at:"
echo "    $SPV_DIR/SHA256SUMS"
echo
echo "Paste each hash into crates/vokra-backend-vulkan/src/spirv.rs's SHADERS"
echo "manifest (SpirvShader::expected_sha256_hex) and rerun \`cargo test -p"
echo "vokra-backend-vulkan verify_pinned_hashes_is_ok\` to confirm the runtime"
echo "load path picks up the same bytes."
