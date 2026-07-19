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
#                SHA256SUMS + PROVENANCE (the M4-13-T16 workflow). DEFAULT
#                when no mode flag is given (backwards compatible with the
#                M3-02 invocation).
#     --check    two-stage drift gate (silent-divergence gate for CI):
#                  (1) SOURCE-HASH GATE (compiler-independent): every
#                      committed .spv's .comp source is SHA-256-compared
#                      against the hash recorded in precompiled/PROVENANCE
#                      at --update time. An edited source without a
#                      recompile is drift → exit 1. Runs on ANY host, no
#                      compiler needed.
#                  (2) RECOMPILE BYTE-DIFF: recompile every .comp into a
#                      temp dir and SHA-256-diff against the committed
#                      .spv. Runs ONLY when the local compiler family AND
#                      version match the PROVENANCE pin — different
#                      compiler families (glslc vs glslangValidator) and
#                      even different glslang versions emit different,
#                      equally valid, SPIR-V bytes, so a cross-tool
#                      byte-diff would spuriously report drift (e.g. the
#                      CI runner's apt glslang-tools vs the Homebrew
#                      glslang that produced the committed blobs). On a
#                      family/version mismatch the byte-diff is skipped
#                      with an honest note; blob-byte integrity is still
#                      enforced everywhere by the SHA-256 pins in
#                      crates/vokra-backend-vulkan/src/spirv.rs
#                      (`cargo test -p vokra-backend-vulkan`, no Vulkan
#                      driver needed).
#                When NO .spv is committed yet (the placeholder slice
#                before the M4-13-T16 commit), reports that honestly and
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
# fallback. NOTE: the two compilers do NOT emit byte-identical SPIR-V — the
# committed blobs' producing tool family + version is pinned in
# precompiled/PROVENANCE (written by --update), and --check only
# byte-compares against a matching local tool (see Modes above). --update
# refuses a single-kernel rebuild with a tool that differs from the
# PROVENANCE pin: mixed-toolchain blob sets would make --check dishonest —
# rerun a FULL --update to switch toolchains.
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
# In --check mode a missing compiler is tolerated when PROVENANCE is present
# (the source-hash gate needs only a sha256 tool; the byte-diff is skipped
# with an honest note) — --update always requires one.
GLSLC_BIN="${GLSLC:-glslc}"
COMPILER_KIND=""
if command -v "$GLSLC_BIN" >/dev/null 2>&1; then
    COMPILER_KIND="glslc"
elif command -v glslangValidator >/dev/null 2>&1; then
    COMPILER_KIND="glslangValidator"
    echo "note: glslc not found; falling back to glslangValidator (glslang-tools)." >&2
    echo "      SPIR-V output differs between the two compilers — --check byte-compares" >&2
    echo "      only when this tool family+version matches precompiled/PROVENANCE." >&2
fi
if [ -z "$COMPILER_KIND" ] && [ "$MODE" = "update" ]; then
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

# One-line version identity of the resolved compiler — pinned into
# PROVENANCE by --update and compared by --check. glslangValidator's first
# --version line is e.g. "Glslang Version: 11:16.4.0"; glslc's names the
# shaderc release. Empty when no compiler was resolved.
compiler_version_line() {
    case "$COMPILER_KIND" in
        glslc)            "$GLSLC_BIN" --version 2>/dev/null | head -n 1 ;;
        glslangValidator) glslangValidator --version 2>/dev/null | head -n 1 ;;
        *)                echo "" ;;
    esac
}

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
    PROV_FILE="$SPV_DIR/PROVENANCE"
    CHECKED=0
    DRIFTED=0
    MISSING=0
    SRC_CHECKED=0

    # ---- Stage 1: compiler-independent source-hash gate (PROVENANCE). ----
    # Detects the dangerous case — a .comp edited without recompiling its
    # committed .spv — on ANY host, regardless of which (or whether a)
    # SPIR-V compiler is installed. This is what keeps the CI drift gate
    # honest when the runner's apt glslang-tools version differs from the
    # pinned toolchain that produced the blobs.
    if [ -f "$PROV_FILE" ]; then
        for spv in "${COMMITTED_SPV[@]}"; do
            base="$(basename "$spv" .spv)"
            if [ -n "$FILTER" ] && [ "$base" != "$FILTER" ]; then
                continue
            fi
            src="$GLSL_DIR/${base}.comp"
            if [ ! -f "$src" ]; then
                continue # reported by the stale-blob loop below
            fi
            want_src="$(grep "^source_sha256 ${base}\.comp " "$PROV_FILE" | head -n 1 | awk '{print $3}' || true)"
            if [ -z "$want_src" ]; then
                echo "[check] $base: NO source_sha256 row in PROVENANCE (rerun --update and commit)" >&2
                DRIFTED=$((DRIFTED + 1))
                continue
            fi
            got_src="$(hash_of "$src")"
            if [ "$want_src" = "$got_src" ]; then
                SRC_CHECKED=$((SRC_CHECKED + 1))
            else
                echo "[check] $base: SOURCE DRIFT — ${base}.comp changed since its committed .spv was compiled (recorded $want_src vs current $got_src)" >&2
                DRIFTED=$((DRIFTED + 1))
            fi
        done
        # Stale provenance rows (a recorded source whose .spv is gone).
        if [ -z "$FILTER" ]; then
            while read -r _tag pname _hash; do
                pbase="${pname%.comp}"
                if [ ! -f "$SPV_DIR/${pbase}.spv" ]; then
                    echo "[check] $pbase: PROVENANCE row has NO committed .spv (stale provenance — rerun --update)" >&2
                    DRIFTED=$((DRIFTED + 1))
                fi
            done < <(grep '^source_sha256 ' "$PROV_FILE" || true)
        fi
        echo "[check] source-hash gate: $SRC_CHECKED source(s) match PROVENANCE"
    else
        echo "[check] WARNING: $PROV_FILE missing — committed blobs have no recorded compiler" >&2
        echo "[check] provenance. Rerun --update with the tool that produced them and commit" >&2
        echo "[check] PROVENANCE; falling back to the recompile byte-diff below." >&2
    fi

    # ---- Stage 2: recompile byte-diff (needs a provenance-matching tool). ----
    BYTE_DIFF=1
    if [ -f "$PROV_FILE" ]; then
        REC_FAMILY="$(grep '^compiler_family=' "$PROV_FILE" | head -n 1 | cut -d= -f2- || true)"
        REC_VERSION="$(grep '^compiler_version=' "$PROV_FILE" | head -n 1 | cut -d= -f2- || true)"
        LOC_VERSION="$(compiler_version_line)"
        if [ -z "$COMPILER_KIND" ]; then
            echo "[check] no local SPIR-V compiler — recompile byte-diff skipped (the source-hash"
            echo "[check] gate above is the drift signal; blob-byte integrity is enforced by the"
            echo "[check] SHA-256 pins in crates/vokra-backend-vulkan/src/spirv.rs via cargo test)."
            BYTE_DIFF=0
        elif [ "$COMPILER_KIND" != "$REC_FAMILY" ] || [ "$LOC_VERSION" != "$REC_VERSION" ]; then
            echo "[check] compiler provenance mismatch — recompile byte-diff skipped honestly:"
            echo "[check]     committed blobs: $REC_FAMILY ($REC_VERSION)"
            echo "[check]     local tool:      $COMPILER_KIND ($LOC_VERSION)"
            echo "[check] Different compiler families/versions emit different (equally valid)"
            echo "[check] SPIR-V bytes, so a cross-tool byte-diff would spuriously report drift."
            echo "[check] The source-hash gate above is the compiler-independent drift signal;"
            echo "[check] blob-byte integrity is enforced by the SHA-256 pins in"
            echo "[check] crates/vokra-backend-vulkan/src/spirv.rs (cargo test, no Vulkan needed)."
            BYTE_DIFF=0
        fi
    elif [ -z "$COMPILER_KIND" ]; then
        echo "error: no PROVENANCE and no SPIR-V compiler — committed blobs cannot be checked." >&2
        echo "       Install a compiler (scripts/install-vulkan-toolchain.md) or rerun --update" >&2
        echo "       with the original tool to record PROVENANCE." >&2
        exit 1
    fi

    if [ "$BYTE_DIFF" -eq 1 ]; then
        TMP_DIR="$(mktemp -d)"
        trap 'rm -rf "$TMP_DIR"' EXIT
        for src in "$GLSL_DIR"/*.comp; do
            base="$(basename "$src" .comp)"
            if [ -n "$FILTER" ] && [ "$base" != "$FILTER" ]; then
                continue
            fi
            committed="$SPV_DIR/${base}.spv"
            if [ ! -f "$committed" ]; then
                # Partial commits are normal mid-T16 (blobs may land
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
    else
        # Byte-diff skipped: still enumerate sources with no committed blob so
        # the "not yet committed" count stays meaningful in the summary.
        for src in "$GLSL_DIR"/*.comp; do
            base="$(basename "$src" .comp)"
            if [ -n "$FILTER" ] && [ "$base" != "$FILTER" ]; then
                continue
            fi
            if [ ! -f "$SPV_DIR/${base}.spv" ]; then
                echo "[check] $base: no committed .spv yet (placeholder) — skipped"
                MISSING=$((MISSING + 1))
            fi
        done
    fi

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
    echo "[check] done: $CHECKED byte-compared, $SRC_CHECKED source-verified, $MISSING not yet committed, $DRIFTED drifted"
    if [ "$DRIFTED" -gt 0 ]; then
        echo "error: SPIR-V drift detected — rerun with --update and commit, or fix the source" >&2
        exit 1
    fi
    exit 0
fi

# ---- update mode -------------------------------------------------------------

# Mixed-toolchain guard: a single-kernel rebuild with a tool that differs
# from the recorded provenance would leave blobs from two compiler
# families/versions side by side — --check's byte-diff would then be
# dishonest for one half or the other. Switching toolchains requires a FULL
# --update (which rewrites every blob + the PROVENANCE pin coherently).
PROV_FILE="$SPV_DIR/PROVENANCE"
if [ -n "$FILTER" ] && [ -f "$PROV_FILE" ]; then
    REC_FAMILY="$(grep '^compiler_family=' "$PROV_FILE" | head -n 1 | cut -d= -f2- || true)"
    REC_VERSION="$(grep '^compiler_version=' "$PROV_FILE" | head -n 1 | cut -d= -f2- || true)"
    LOC_VERSION="$(compiler_version_line)"
    if [ "$COMPILER_KIND" != "$REC_FAMILY" ] || [ "$LOC_VERSION" != "$REC_VERSION" ]; then
        echo "error: single-kernel --update with a compiler that differs from PROVENANCE:" >&2
        echo "           recorded: $REC_FAMILY ($REC_VERSION)" >&2
        echo "           local:    $COMPILER_KIND ($LOC_VERSION)" >&2
        echo "       Mixed-toolchain blob sets make --check dishonest. Either use the" >&2
        echo "       recorded tool, or rerun a FULL --update (no kernel filter) to rebuild" >&2
        echo "       every blob and re-pin the toolchain — then re-paste the new hashes" >&2
        echo "       into crates/vokra-backend-vulkan/src/spirv.rs." >&2
        exit 1
    fi
fi

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
# 7.5. Emit PROVENANCE: the compiler family+version that produced the
#      committed blobs, plus the SHA-256 of each .comp SOURCE at compile
#      time. --check's two-stage drift gate consumes both (see the Modes
#      note in the header). Regenerated over ALL committed .spv (like
#      SHA256SUMS) so a single-kernel rebuild keeps the file coherent —
#      the mixed-toolchain guard above ensures the compiler pin stays
#      truthful for every blob.
# ------------------------------------------------------------------------------

{
    echo "# SPIR-V blob provenance — written by scripts/compile-vulkan-shaders.sh --update."
    echo "# Do not edit by hand."
    echo "#"
    echo "# compiler_family/compiler_version pin the toolchain that produced every"
    echo "# committed kernels/precompiled/*.spv. SPIR-V output is NOT byte-stable"
    echo "# across compiler families (glslc vs glslangValidator) or versions, so"
    echo "# scripts/compile-vulkan-shaders.sh --check only recompile-byte-diffs when"
    echo "# the local tool matches this pin; otherwise it relies on the source_sha256"
    echo "# rows below (compiler-independent source-drift gate) and on the blob"
    echo "# SHA-256 pins in crates/vokra-backend-vulkan/src/spirv.rs (cargo test)."
    echo "compiler_family=$COMPILER_KIND"
    echo "compiler_version=$(compiler_version_line)"
    for f in "$SPV_DIR"/*.spv; do
        [ -f "$f" ] || continue
        b="$(basename "$f" .spv)"
        s="$GLSL_DIR/${b}.comp"
        if [ -f "$s" ]; then
            echo "source_sha256 ${b}.comp $(hash_of "$s")"
        fi
    done
} > "$PROV_FILE"

# ------------------------------------------------------------------------------
# 8. Emit human-readable summary for the developer to paste into spirv.rs's
#    `expected_sha256_hex` field.
# ------------------------------------------------------------------------------

echo
echo "Generated $COMPILED_COUNT .spv blob(s). SHA-256 manifest at:"
echo "    $SPV_DIR/SHA256SUMS"
echo "Compiler provenance ($COMPILER_KIND, $(compiler_version_line)) pinned at:"
echo "    $PROV_FILE"
echo
echo "Paste each hash into crates/vokra-backend-vulkan/src/spirv.rs's SHADERS"
echo "manifest (SpirvShader::expected_sha256_hex), switch the matching"
echo "load_spv arm to include_bytes!, and rerun \`cargo test -p"
echo "vokra-backend-vulkan\` (verify_pinned_hashes + the blob-gated parity"
echo "tests light up automatically — M4-13-T16). Commit the .spv blobs,"
echo "SHA256SUMS and PROVENANCE together."
