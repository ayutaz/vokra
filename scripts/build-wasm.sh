#!/usr/bin/env bash
# build-wasm.sh — wasm32-unknown-unknown cross-build driver (M4-01-T03/T04/T20).
#
# Ticket: M4-01-T03 (cross-build base), T04 (SIMD128 2-artifact policy), T20
# (npm package assembly). Basis: FR-BE-05 / NFR-PF-08 (Web = WASM + WebGPU),
# NFR-DS-02 (zero-dep: adding a rustup target never touches dependency
# resolution — the root Cargo.lock stays `vokra-*`-only, enforced below with
# a diff tripwire), FR-EX-08 (no silent fallback — missing tools are explicit
# errors), ADR M4-01-webgpu-wasm §4 (SIMD128 2-artifact + JS loader select).
#
# WASM has NO runtime CPU feature detection: SIMD128 acceptance is decided at
# module *validation* time, so the AVX2/NEON-style CPUID runtime dispatch of
# the CPU backend cannot work on wasm32. Instead this script builds TWO
# artifacts of the Web entry crate (`tests/wasm-harness`, package
# `vokra-wasm-harness`):
#
#   web/dist/vokra_wasm_base.wasm      — no SIMD (baseline MVP wasm)
#   web/dist/vokra_wasm_simd128.wasm   — RUSTFLAGS="-C target-feature=+simd128"
#
# and the JS loader (web/pkg/index.js) picks one at load time with a
# `WebAssembly.validate` probe (ADR M4-01 §4).
#
# Memory64 status (M4-01-T03, survey only — no implementation): Whisper base
# (~74M params, fp16 GGUF ~150 MB) fits comfortably inside wasm32 linear
# memory (4 GiB architectural limit), so the wasm64 target is NOT built here.
# Rust `wasm64-unknown-unknown` is Tier 3 (std not guaranteed) as of the
# 2026-07 toolchain; browser Memory64 is shipped in Chromium/Firefox lines
# and not in Safari (owner spot check T28 re-confirms). Large models are a
# follow-up (ADR M4-01 §9).
#
# Usage:
#   scripts/build-wasm.sh check     # wasm32 cargo check of the 4 core crates (T03 gate)
#   scripts/build-wasm.sh harness   # build base + simd128 test-entry artifacts (T06)
#   scripts/build-wasm.sh pkg       # harness (production entries only) + assemble web/pkg (T20)
#   scripts/build-wasm.sh all       # check + harness + pkg   (default)
#
# Exit codes: 0 = success; 1 = missing tool / target / build failure / lock drift.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRIPLE="wasm32-unknown-unknown"
MODE="${1:-all}"

cd "$ROOT"

# --- guardrails ---------------------------------------------------------------

if ! command -v cargo >/dev/null 2>&1; then
    echo "error: cargo not found on PATH" >&2
    exit 1
fi

# rustup target (idempotent — mirrors scripts/build-android.sh).
installed="$(rustup target list --installed 2>/dev/null || true)"
if ! printf '%s\n' "$installed" | grep -q "^$TRIPLE$"; then
    echo "== rustup target add $TRIPLE (one-time)"
    rustup target add "$TRIPLE"
fi

# Root Cargo.lock tripwire (NFR-DS-02): snapshot before, diff after. A wasm
# cross-build must never change dependency resolution.
LOCK_BEFORE="$(shasum -a 256 Cargo.lock | awk '{print $1}')"

check_lock() {
    local after
    after="$(shasum -a 256 Cargo.lock | awk '{print $1}')"
    if [ "$LOCK_BEFORE" != "$after" ]; then
        echo "error: root Cargo.lock changed during the wasm build (NFR-DS-02 tripwire)" >&2
        git diff --stat Cargo.lock >&2 || true
        exit 1
    fi
}

# --- check: wasm32 buildability of the core crates (M4-01-T03 gate) -----------

do_check() {
    echo "== cargo check --target $TRIPLE (vokra-core / vokra-ops / vokra-backend-cpu / vokra-models, default features)"
    cargo check --target "$TRIPLE" \
        -p vokra-core -p vokra-ops -p vokra-backend-cpu -p vokra-models
    echo "== cargo check --target $TRIPLE (vokra-models --features webgpu)"
    cargo check --target "$TRIPLE" -p vokra-models --features webgpu
    echo "== cargo check --target $TRIPLE with +simd128 (T04 gate)"
    RUSTFLAGS="-C target-feature=+simd128" \
        cargo check --target "$TRIPLE" --target-dir target/wasm-simd128-check \
        -p vokra-core -p vokra-ops -p vokra-backend-cpu -p vokra-models
    check_lock
    echo "== wasm32 check OK (root Cargo.lock unchanged)"
}

# --- harness / pkg: 2-artifact builds of the Web entry crate -------------------

# $1 = destination dir; $2 = extra cargo args (e.g. --no-default-features for
#      the production pkg slice without the vokra_test_* kernel-parity
#      exports). The two feature slices land in DIFFERENT directories —
#      web/dist keeps the test-entry artifacts the Node/browser harnesses
#      drive, web/pkg gets the production exports-only artifacts — so a
#      `pkg` run never clobbers the harness artifacts (`all` runs both).
build_two_artifacts() {
    local dest="$1"
    local extra_args="${2:-}"
    mkdir -p "$dest"

    echo "== build base artifact (no SIMD) -> $dest $extra_args"
    # shellcheck disable=SC2086  # word-splitting of extra cargo args is intended
    cargo build --release --target "$TRIPLE" -p vokra-wasm-harness $extra_args
    cp "target/$TRIPLE/release/vokra_wasm_harness.wasm" "$dest/vokra_wasm_base.wasm"

    echo "== build simd128 artifact (RUSTFLAGS=-C target-feature=+simd128) -> $dest $extra_args"
    # A separate --target-dir keeps the two flag sets from thrashing each
    # other's incremental cache and keeps both artifacts addressable.
    # shellcheck disable=SC2086
    RUSTFLAGS="-C target-feature=+simd128" \
        cargo build --release --target "$TRIPLE" --target-dir target/wasm-simd128 \
        -p vokra-wasm-harness $extra_args
    cp "target/wasm-simd128/$TRIPLE/release/vokra_wasm_harness.wasm" "$dest/vokra_wasm_simd128.wasm"

    check_lock
    ls -la "$dest"/*.wasm
}

do_harness() {
    build_two_artifacts web/dist ""
}

# --- pkg: assemble the npm package layout (M4-01-T20) --------------------------

do_pkg() {
    # Production slice: --no-default-features drops the `test-entries` feature
    # so the shipped .wasm exports only the vokra_wasm_* session API. Built
    # straight into web/pkg — web/dist (test-entry artifacts) stays intact.
    build_two_artifacts web/pkg "--no-default-features"

    echo "== assemble web/pkg (npm layout)"
    cp crates/vokra-backend-webgpu/glue/vokra_webgpu.js web/pkg/
    cp crates/vokra-backend-webgpu/glue/vokra_worker.js web/pkg/
    cp LICENSE web/pkg/LICENSE
    cp NOTICE web/pkg/NOTICE
    echo "== web/pkg contents:"
    ls -la web/pkg/
    echo "== npm pack --dry-run (self-contained tarball sanity)"
    (cd web/pkg && npm pack --dry-run)
}

case "$MODE" in
    check)   do_check ;;
    harness) do_check; do_harness ;;
    pkg)     do_check; do_pkg ;;
    all)     do_check; do_harness; do_pkg ;;
    *)
        echo "error: unknown mode '$MODE' (expected: check | harness | pkg | all)" >&2
        exit 1
        ;;
esac
