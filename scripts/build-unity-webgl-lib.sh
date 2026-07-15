#!/usr/bin/env bash
# build-unity-webgl-lib.sh — build the Unity WebGL native plugin staticlib and
# sync it into the UPM package (M4-02-T04, ADR M4-02).
#
# =============================================================================
#  CANONICAL DISTRIBUTION = CD  (NFR-MT-08 / FR-API-04)
#
#  正規発行は CD 経由。Local runs of this script are for dev iteration only.
#  The release-signed com.vokra.unity-<version>.tgz that consumers install is
#  produced by .github/workflows/release.yml (`unity-package-release` job).
# =============================================================================
#
# What it builds: crates/vokra-capi as a `staticlib` (`libvokra.a`, GNU
# archive of WebAssembly object files) for `wasm32-unknown-emscripten` —
# the Unity WebGL native plugin format (Unity Manual "WebGL native plug-ins
# for Emscripten", preferred since 2021.2). Unity's own Emscripten performs
# the final link; **emsdk is NOT needed to build the .a** (rustc emits the
# wasm objects itself — ADR M4-02 §2). emsdk + node are needed only for the
# optional `--verify` harness.
#
# Non-negotiable flags (ADR M4-02 §2, empirically derived):
#   * `-C panic=abort` — default (unwind) objects reference the wasm-EH tag
#     `__cpp_exception`, which Unity-bundled Emscripten (3.1.8 / 3.1.38)
#     fails to link. panic=abort objects link cleanly on both.
#   * `--no-default-features --features cpu` — CPU-only audit trail
#     (M2-11-T13); no Metal/CUDA/Vulkan/WebGPU code can enter the archive
#     (mechanically asserted by the GPU-symbol audit below).
#   * SIMD128 is OFF for the package artifact (browser floor + Unity's own
#     SIMD setting independence — ADR M4-02 §7); `--simd` builds an opt-in
#     variant that is NOT synced into the package.
#
# Usage:
#   scripts/build-unity-webgl-lib.sh                 # build + audit + gate + sync
#   scripts/build-unity-webgl-lib.sh --simd          # also build the simd128 variant (not synced)
#   scripts/build-unity-webgl-lib.sh --no-build      # skip cargo; use existing artifact
#   scripts/build-unity-webgl-lib.sh --skip-audit    # skip the llvm-nm GPU-symbol audit (visible opt-out)
#   scripts/build-unity-webgl-lib.sh --verify        # + emcc(pin) link of smoke_vad_bytes + node run
#
# Env:
#   VOKRA_WEBGL_EMSDK_VERSION  emcc version pin for --verify (default 3.1.38 =
#                              Unity 6 line; 3.1.8 = Unity 2022.3 line is also
#                              a measured-good pin — ADR M4-02 §2 matrix).
#   VOKRA_LLVM_NM              explicit llvm-nm path for the audit.
#
# Exit code: 0 = OK; non-zero = build / audit / gate / verify failure
# (explicit errors, FR-EX-08 — no silent skip except the flagged opt-outs).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRIPLE="wasm32-unknown-emscripten"
PKG_WEBGL_DIR="$ROOT/bindings/unity/com.vokra.unity/Plugins/WebGL"
ARCHIVE="$ROOT/target/webgl/$TRIPLE/release/libvokra.a"
ARCHIVE_SIMD="$ROOT/target/webgl-simd/$TRIPLE/release/libvokra.a"
EMSDK_PIN="${VOKRA_WEBGL_EMSDK_VERSION:-3.1.38}"

DO_BUILD=1
DO_SIMD=0
DO_AUDIT=1
DO_VERIFY=0

for arg in "$@"; do
    case "$arg" in
        --no-build)   DO_BUILD=0 ;;
        --simd)       DO_SIMD=1 ;;
        --skip-audit) DO_AUDIT=0 ;;
        --verify)     DO_VERIFY=1 ;;
        -h|--help)    sed -n '2,45p' "$0"; exit 0 ;;
        *)
            echo "build-unity-webgl-lib: unknown flag: $arg" >&2
            exit 2
            ;;
    esac
done

echo "== build-unity-webgl-lib (canonical publish = CD, NFR-MT-08) =="

# ---- rustup target (idempotent, mirrors build-android.sh) -------------------
installed="$(rustup target list --installed 2>/dev/null || true)"
if ! printf '%s\n' "$installed" | grep -q "^$TRIPLE$"; then
    echo "== rustup target add $TRIPLE (one-time)"
    rustup target add "$TRIPLE"
fi

# ---- root Cargo.lock tripwire (NFR-DS-02) -----------------------------------
LOCK_BEFORE="$(shasum -a 256 "$ROOT/Cargo.lock" | awk '{print $1}')"
check_lock() {
    local after
    after="$(shasum -a 256 "$ROOT/Cargo.lock" | awk '{print $1}')"
    if [ "$LOCK_BEFORE" != "$after" ]; then
        echo "build-unity-webgl-lib: FAIL root Cargo.lock changed (NFR-DS-02 tripwire)" >&2
        exit 1
    fi
}

# ---- build -------------------------------------------------------------------
if [ "$DO_BUILD" -eq 1 ]; then
    echo "== cargo rustc -p vokra-capi ($TRIPLE, staticlib, panic=abort, CPU-only) =="
    ( cd "$ROOT" && RUSTFLAGS="-C panic=abort" cargo rustc -p vokra-capi --release \
        --target "$TRIPLE" --no-default-features --features cpu \
        --crate-type staticlib --target-dir target/webgl )
    if [ "$DO_SIMD" -eq 1 ]; then
        echo "== cargo rustc simd128 variant (NOT synced into the package — ADR M4-02 §7) =="
        ( cd "$ROOT" && RUSTFLAGS="-C panic=abort -C target-feature=+simd128" \
            cargo rustc -p vokra-capi --release \
            --target "$TRIPLE" --no-default-features --features cpu \
            --crate-type staticlib --target-dir target/webgl-simd )
    fi
    check_lock
fi

if [ ! -f "$ARCHIVE" ]; then
    echo "build-unity-webgl-lib: FAIL staticlib not found: $ARCHIVE" >&2
    echo "                       (run without --no-build to compile it)" >&2
    exit 1
fi

# ---- llvm-nm lookup (audit needs a wasm-capable nm) --------------------------
find_llvm_nm() {
    if [ -n "${VOKRA_LLVM_NM:-}" ]; then
        echo "$VOKRA_LLVM_NM"
        return 0
    fi
    if command -v llvm-nm >/dev/null 2>&1; then
        command -v llvm-nm
        return 0
    fi
    if [ -n "${EMSDK:-}" ] && [ -x "$EMSDK/upstream/bin/llvm-nm" ]; then
        echo "$EMSDK/upstream/bin/llvm-nm"
        return 0
    fi
    if command -v rustc >/dev/null 2>&1; then
        local sysroot host cand
        sysroot="$(rustc --print sysroot 2>/dev/null || true)"
        host="$(rustc -vV 2>/dev/null | awk '/^host:/ { print $2 }')"
        cand="$sysroot/lib/rustlib/$host/bin/llvm-nm"
        if [ -x "$cand" ]; then
            echo "$cand"
            return 0
        fi
    fi
    return 1
}

# ---- GPU-symbol non-mixin audit (M4-02-T08, NFR-LC-01 posture) ---------------
# The WebGL build is CPU-only by feature selection; this audit asserts it at
# the object level: no Metal / CUDA-driver / NVRTC / Vulkan-loader / WebGPU
# extern-import identifier may appear in any archive symbol.
if [ "$DO_AUDIT" -eq 1 ]; then
    NM="$(find_llvm_nm)" || {
        echo "build-unity-webgl-lib: FAIL llvm-nm not found for the GPU-symbol audit." >&2
        echo "  Provide VOKRA_LLVM_NM, put llvm-nm on PATH, set \$EMSDK, or" >&2
        echo "  'rustup component add llvm-tools'. (--skip-audit to bypass, visibly.)" >&2
        exit 1
    }
    echo "== GPU-symbol audit (llvm-nm: $NM) =="
    GPU_TOKENS='vokra_webgpu_|MTLCreateSystemDefaultDevice|objc_msgSend|cuInit|cuDeviceGet|nvrtcCompileProgram|vkGetInstanceProcAddr'
    hits="$("$NM" -A "$ARCHIVE" 2>/dev/null | grep -E "$GPU_TOKENS" || true)"
    if [ -n "$hits" ]; then
        echo "build-unity-webgl-lib: FAIL GPU backend symbol(s) in the WebGL staticlib:" >&2
        echo "$hits" >&2
        exit 1
    fi
    echo "   OK: no Metal/CUDA/Vulkan/WebGPU symbol in $ARCHIVE"
else
    echo "== GPU-symbol audit SKIPPED (--skip-audit) =="
fi

# ---- production thread-free gate (M4-02-T02) ---------------------------------
bash "$ROOT/scripts/check-capi-thread-free.sh"

# ---- sync into the UPM package ------------------------------------------------
mkdir -p "$PKG_WEBGL_DIR"
cp -f "$ARCHIVE" "$PKG_WEBGL_DIR/libvokra.a"
echo "build-unity-webgl-lib: synced $PKG_WEBGL_DIR/libvokra.a"
if [ "$DO_SIMD" -eq 1 ] && [ -f "$ARCHIVE_SIMD" ]; then
    echo "build-unity-webgl-lib: simd128 variant at $ARCHIVE_SIMD"
    echo "  (opt-in: replace Plugins/WebGL/libvokra.a manually ONLY for projects"
    echo "   that enable Unity's WebAssembly SIMD player setting — TESTING.md)"
fi

# ---- optional verify harness (emcc pin + node run) ----------------------------
if [ "$DO_VERIFY" -eq 1 ]; then
    echo "== verify: emcc link + node run of tests/capi/smoke_vad_bytes.c =="
    command -v emcc >/dev/null 2>&1 || {
        echo "build-unity-webgl-lib: FAIL --verify needs emcc (emsdk $EMSDK_PIN)." >&2
        echo "  emsdk install $EMSDK_PIN && emsdk activate $EMSDK_PIN && source emsdk_env.sh" >&2
        exit 1
    }
    EMCC_VER="$(emcc --version 2>/dev/null | head -n1 | sed -nE 's/^emcc .[^0-9]*([0-9]+\.[0-9]+\.[0-9]+).*$/\1/p')"
    if [ "$EMCC_VER" != "$EMSDK_PIN" ]; then
        echo "build-unity-webgl-lib: FAIL emcc version $EMCC_VER != pin $EMSDK_PIN" >&2
        echo "  (VOKRA_WEBGL_EMSDK_VERSION overrides the pin; mismatched toolchains" >&2
        echo "   are exactly the failure mode this check makes loud — ADR M4-02 §2)" >&2
        exit 1
    fi
    command -v node >/dev/null 2>&1 || {
        echo "build-unity-webgl-lib: FAIL --verify needs node" >&2
        exit 1
    }
    VERIFY_DIR="$ROOT/target/webgl-verify"
    mkdir -p "$VERIFY_DIR"
    emcc "$ROOT/tests/capi/smoke_vad_bytes.c" "$ARCHIVE" \
        -I "$ROOT/include" \
        -sALLOW_MEMORY_GROWTH=1 -sSINGLE_FILE=1 \
        --embed-file "$ROOT/tests/parity/silero_vad/silero-vad-v5.gguf@/vokra-fixtures/silero-vad-v5.gguf" \
        --embed-file "$ROOT/tests/capi/fixtures/vad_input_16k.f32@/vokra-fixtures/vad_input_16k.f32" \
        -o "$VERIFY_DIR/smoke_vad_bytes.js"
    OUT="$(node "$VERIFY_DIR/smoke_vad_bytes.js" \
        /vokra-fixtures/silero-vad-v5.gguf /vokra-fixtures/vad_input_16k.f32)"
    echo "$OUT"
    if ! printf '%s\n' "$OUT" | grep -q '^smoke_vad_bytes: PASS$'; then
        echo "build-unity-webgl-lib: FAIL verify harness did not print PASS" >&2
        exit 1
    fi
    echo "   OK: bytes-path VAD chain PASS under emcc $EMCC_VER + node"
fi

check_lock
echo "build-unity-webgl-lib: OK"
echo
echo "Reminder: canonical distribution is the release-signed .tgz produced by"
echo "          .github/workflows/release.yml (NFR-MT-08). Local runs are for"
echo "          dev iteration only."
