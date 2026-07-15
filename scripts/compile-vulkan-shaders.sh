#!/usr/bin/env bash
# M3-02-T13 + M4-13-T11 / ADR M3-02-spirv-generation §4 (a) + (b).
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
# no CPU-side JIT). CI drift-gates by re-running this script in `--check` mode
# and diffing the recompiled output against the committed blobs
# (gpu-vulkan-parity.yml, M4-13-T14; ADR §4 (b)).
#
# Dependencies: bash (>= 4.x), coreutils (`sha256sum` or macOS `shasum -a 256`),
# and `glslc` from the Vulkan SDK (fallback: `glslangValidator` from
# glslang-tools — see the tool-resolution note below). See
# `scripts/install-vulkan-toolchain.md` for install instructions per OS. No
# Python, no crate, no cargo.
#
# Modes (ADR §4 (b)):
#     --update   compile .comp → precompiled/*.spv IN PLACE and refresh
#                SHA256SUMS (the owner M4-13-T16 workflow). DEFAULT when no
#                mode flag is given (backwards compatible with the M3-02
#                invocation).
#     --check    recompile every .comp into a temp dir and DIFF the SHA-256
#                of each result against the committed precompiled/*.spv.
#                Non-zero exit on any drift (silent-divergence gate for CI).
#                When NO .spv is committed yet (the placeholder slice before
#                the owner's M4-13-T16 commit), reports that honestly and
#                exits 0 WITHOUT requiring a compiler — a clean skip, not a
#                fabricated pass.
#
# Per-shader target environment (M4-13-T11): each `.comp` header carries its
# own `Compile with: glslc --target-env=vulkanX.Y …` line (gemm_coopmat needs
# vulkan1.3 for VK_KHR_cooperative_matrix; everything else is vulkan1.1).
# The script parses that line per file and falls back to vulkan1.1 when the
# header names none — the committed source is the single source of truth,
# not a script-side constant.
#
# Tool resolution: `glslc` (shaderc; ships in the LunarG SDK and Homebrew
# `shaderc`) is preferred. When absent, `glslangValidator -V` (Khronos
# glslang; Ubuntu `glslang-tools`, Homebrew `glslang`) is used as a
# fallback. NOTE: the two compilers do NOT emit byte-identical SPIR-V — a
# `--check` run must use the same tool family that produced the committed
# blobs, or drift will be reported (which is honest: the report names the
# tool used).
#
# Usage:
#     scripts/compile-vulkan-shaders.sh                       # --update, all kernels
#     scripts/compile-vulkan-shaders.sh --update gemm_subgroup # one kernel
#     scripts/compile-vulkan-shaders.sh --check                # CI drift gate
#     GLSLC=/opt/vulkan/bin/glslc scripts/compile-vulkan-shaders.sh  # tool override
#
# Exit status:
#     0 — success (--update: blobs + SHA256SUMS refreshed; --check: no drift,
#         or honest clean skip when nothing is committed yet)
#     1 — compiler missing (when required) or compile errors or drift found
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
# 2. Parse mode + optional kernel filter.
# ------------------------------------------------------------------------------

MODE="update"
FILTER=""
for arg in "$@"; do
    case "$arg" in
        --update) MODE="update" ;;
        --check)  MODE="check" ;;
        --*)
            echo "error: unknown flag '$arg' (expected --update or --check)" >&2
            exit 3
            ;;
        *)
            if [ -n "$FILTER" ]; then
                echo "error: at most one kernel filter may be given (got '$FILTER' and '$arg')" >&2
                exit 3
            fi
            FILTER="$arg"
            ;;
    esac
done

# ------------------------------------------------------------------------------
# 3. --check clean-skip path: nothing committed yet → honest report, exit 0.
#    Runs BEFORE tool resolution so the placeholder slice needs no compiler
#    (the owner M4-13-T16 commit flips this path off automatically).
# ------------------------------------------------------------------------------

shopt -s nullglob
COMMITTED_SPV=("$SPV_DIR"/*.spv)
if [ "$MODE" = "check" ] && [ "${#COMMITTED_SPV[@]}" -eq 0 ]; then
    echo "[check] no committed .spv under $SPV_DIR"
    echo "[check] placeholder slice (owner glslc commit pending, M4-13-T16) — drift check"
    echo "[check] has nothing to compare against; skipping cleanly (NOT a fabricated pass:"
    echo "[check] the runtime treats these kernels as UnsupportedOp until blobs land)."
    exit 0
fi

# ------------------------------------------------------------------------------
# 4. Resolve compiler + sha256 tool.
# ------------------------------------------------------------------------------

# Preferred: glslc (or $GLSLC override). Fallback: glslangValidator -V.
GLSLC_BIN="${GLSLC:-glslc}"
COMPILER_KIND=""
if command -v "$GLSLC_BIN" >/dev/null 2>&1; then
    COMPILER_KIND="glslc"
elif command -v glslangValidator >/dev/null 2>&1; then
    COMPILER_KIND="glslangValidator"
    echo "note: glslc not found; falling back to glslangValidator (glslang-tools)." >&2
    echo "      SPIR-V output differs between the two compilers — in --check mode," >&2
    echo "      compare only against blobs produced by the same tool family." >&2
else
    cat >&2 <<EOF
error: no SPIR-V compiler found (tried: $GLSLC_BIN, glslangValidator)

To install:
    macOS:   brew install shaderc          # provides glslc
             or: brew install glslang      # provides glslangValidator
             or download the LunarG Vulkan SDK: https://vulkan.lunarg.com/sdk/home#mac
    Ubuntu:  sudo apt install glslang-tools   # provides glslangValidator
             (glslc ships in the LunarG SDK tarball for Linux)
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

hash_of() {
    # Prints just the hex digest of "$1".
    $SHA256_TOOL "$1" | awk '{print $1}'
}

# ------------------------------------------------------------------------------
# 5. Helpers: per-shader target-env (parsed from the .comp header) + compile.
# ------------------------------------------------------------------------------

target_env_of() {
    # Reads the "Compile with: glslc --target-env=vulkanX.Y ..." header line
    # of "$1"; defaults to vulkan1.1 (M3-02 ADR §T01(c) baseline) when the
    # header names none.
    local env
    env="$(grep -o -- '--target-env=vulkan[0-9]\.[0-9]' "$1" | head -n 1 | cut -d= -f2 || true)"
    echo "${env:-vulkan1.1}"
}

compile_one() {
    # compile_one <src.comp> <dst.spv> — honours the per-file target env.
    local src="$1" dst="$2" tenv
    tenv="$(target_env_of "$src")"
    case "$COMPILER_KIND" in
        glslc)
            # -O / --optimize intentionally omitted: the committed bytecode
            # stays a faithful, debuggable reflection of the source.
            "$GLSLC_BIN" --target-env="$tenv" -o "$dst" "$src"
            ;;
        glslangValidator)
            glslangValidator -V --target-env "$tenv" -o "$dst" "$src" >/dev/null
            ;;
    esac
}

# ------------------------------------------------------------------------------
# 6. Main loop.
# ------------------------------------------------------------------------------

if [ "$MODE" = "check" ]; then
    TMP_DIR="$(mktemp -d)"
    trap 'rm -rf "$TMP_DIR"' EXIT
    CHECKED=0
    DRIFTED=0
    MISSING=0
    for src in "$GLSL_DIR"/*.comp; do
        base="$(basename "$src" .comp)"
        if [ -n "$FILTER" ] && [ "$base" != "$FILTER" ]; then
            continue
        fi
        committed="$SPV_DIR/${base}.spv"
        if [ ! -f "$committed" ]; then
            # Partial commits are normal mid-T16 (owner lands blobs
            # shader-by-shader); report but do not fail — the runtime keeps
            # treating the op as UnsupportedOp, which is already honest.
            echo "[check] $base: no committed .spv yet (placeholder) — skipped"
            MISSING=$((MISSING + 1))
            continue
        fi
        rebuilt="$TMP_DIR/${base}.spv"
        if ! compile_one "$src" "$rebuilt"; then
            echo "[check] $base: recompile FAILED (compiler=$COMPILER_KIND)" >&2
            DRIFTED=$((DRIFTED + 1))
            continue
        fi
        want="$(hash_of "$committed")"
        got="$(hash_of "$rebuilt")"
        if [ "$want" = "$got" ]; then
            echo "[check] $base: OK ($got)"
        else
            echo "[check] $base: DRIFT — committed $want vs recompiled $got (compiler=$COMPILER_KIND)" >&2
            DRIFTED=$((DRIFTED + 1))
        fi
        CHECKED=$((CHECKED + 1))
    done
    # Committed .spv with no matching source is also drift (stale blob).
    for spv in "${COMMITTED_SPV[@]}"; do
        base="$(basename "$spv" .spv)"
        if [ -n "$FILTER" ] && [ "$base" != "$FILTER" ]; then
            continue
        fi
        if [ ! -f "$GLSL_DIR/${base}.comp" ]; then
            echo "[check] $base: committed .spv has NO GLSL source (stale blob)" >&2
            DRIFTED=$((DRIFTED + 1))
        fi
    done
    echo "[check] done: $CHECKED compared, $MISSING not yet committed, $DRIFTED drifted"
    if [ "$DRIFTED" -gt 0 ]; then
        echo "error: SPIR-V drift detected — rerun with --update and commit, or fix the source" >&2
        exit 1
    fi
    exit 0
fi

# ---- update mode -------------------------------------------------------------

COMPILED_COUNT=0
FAILED_COUNT=0

for src in "$GLSL_DIR"/*.comp; do
    base="$(basename "$src" .comp)"
    if [ -n "$FILTER" ] && [ "$base" != "$FILTER" ]; then
        continue
    fi
    dst="$SPV_DIR/${base}.spv"
    if compile_one "$src" "$dst"; then
        printf '[ok]  %s -> %s (%s)\n' "$src" "$dst" "$(target_env_of "$src")"
        COMPILED_COUNT=$((COMPILED_COUNT + 1))
    else
        printf '[err] %s (compile failed, compiler=%s)\n' "$src" "$COMPILER_KIND" >&2
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
# 7. Emit SHA256SUMS (over ALL .spv, not just the ones we recompiled — so that
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
# 8. Emit human-readable summary for the developer to paste into spirv.rs's
#    `expected_sha256_hex` field.
# ------------------------------------------------------------------------------

echo
echo "Generated $COMPILED_COUNT .spv blob(s). SHA-256 manifest at:"
echo "    $SPV_DIR/SHA256SUMS"
echo
echo "Paste each hash into crates/vokra-backend-vulkan/src/spirv.rs's SHADERS"
echo "manifest (SpirvShader::expected_sha256_hex), switch the matching"
echo "load_spv arm to include_bytes!, and rerun \`cargo test -p"
echo "vokra-backend-vulkan\` (verify_pinned_hashes + the blob-gated parity"
echo "tests light up automatically — M4-13-T16)."
