#!/usr/bin/env bash
# test-m4-15-build-target.sh — scratch-tree tests for the M4-15 CPU +
# Vulkan-only build-target tooling (ADR M4-15).
#
# Mirrors the scratch-tree test style of the M3-11 Godot compliance
# scanner: build tiny fake trees under mktemp, run the real tool against
# them, and assert on the observed exit codes / output. No network, no
# cargo, no third-party tool — plain bash + cc(optional) so the suite runs
# both locally and as a step of the `build-target-vulkan-only` CI job.
#
# Tools under test (grown per ticket, TDD):
#   [T05] scripts/gen-notice-cpu-vulkan-only.sh       NOTICE variant derivation
#   [T04] scripts/compliance/check-cpu-vulkan-only-no-nvidia.sh  non-link scanner
#   [T08] scripts/compliance/check-no-market-claims.sh forbidden-words gate
#
# Usage: bash scripts/compliance/test-m4-15-build-target.sh
# Exit:  0 = all tests pass, 1 = at least one failure.

set -uo pipefail # deliberately NOT -e: we assert on nonzero exits of the tools

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GEN_NOTICE="$ROOT/scripts/gen-notice-cpu-vulkan-only.sh"

SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/m4-15-tests.XXXXXX")"
trap 'rm -rf "$SCRATCH"' EXIT

pass=0
fail=0
skip=0
ok() {
    pass=$((pass + 1))
    echo "  ok:   $1"
}
bad() {
    fail=$((fail + 1))
    echo "  FAIL: $1" >&2
}
skipped() {
    skip=$((skip + 1))
    echo "  skip: $1 (announced, not fabricated — FR-EX-08 spirit)"
}

# ===========================================================================
# [T05] NOTICE variant generator
# ===========================================================================
echo "[T05] gen-notice-cpu-vulkan-only.sh"

if [ ! -f "$GEN_NOTICE" ]; then
    bad "generator script missing: $GEN_NOTICE"
else
    VARIANT="$SCRATCH/NOTICE.variant"

    # t1 — generator succeeds on the real root NOTICE.
    if bash "$GEN_NOTICE" --notice "$ROOT/NOTICE" --out "$VARIANT" >/dev/null 2>&1 \
        && [ -s "$VARIANT" ]; then
        ok "t1 generates a non-empty variant from the root NOTICE"
    else
        bad "t1 generator failed on the real root NOTICE"
    fi

    # t2 — the NVIDIA runtime section is dropped.
    if [ -s "$VARIANT" ] \
        && ! grep -q "NVIDIA CUDA / cuDNN / cuBLAS runtime dependencies" "$VARIANT"; then
        ok "t2 NVIDIA runtime section dropped"
    else
        bad "t2 NVIDIA runtime section still present in the variant"
    fi

    # t3 — no reference to the NVIDIA EULA document survives.
    if [ -s "$VARIANT" ] && ! grep -q "NVIDIA-EULA" "$VARIANT"; then
        ok "t3 variant does not reference the NVIDIA EULA document"
    else
        bad "t3 variant still references NVIDIA-EULA"
    fi

    # t4 — backend-independent attributions are all retained.
    t4_ok=1
    for marker in \
        "BigVGAN" \
        "pocketfft" \
        "piper-plus" \
        "Mimi codec" \
        "EnCodec" \
        "SpeexDSP echo canceller" \
        "Maintaining this file"; do
        if ! grep -q "$marker" "$VARIANT" 2>/dev/null; then
            bad "t4 retained-section marker missing from variant: $marker"
            t4_ok=0
        fi
    done
    [ "$t4_ok" -eq 1 ] && ok "t4 backend-independent attributions retained"

    # t5 — section numbers are renumbered to a contiguous 1..N (and the
    # variant has exactly one section fewer than the root NOTICE).
    root_sections="$(grep -cE '^[0-9]+\. ' "$ROOT/NOTICE" || true)"
    variant_numbers="$(grep -oE '^[0-9]+\. ' "$VARIANT" 2>/dev/null | grep -oE '^[0-9]+' | tr '\n' ' ')"
    variant_count="$(echo "$variant_numbers" | wc -w | tr -d ' ')"
    expected_numbers="$(seq 1 "$variant_count" | tr '\n' ' ')"
    if [ "$variant_count" -eq $((root_sections - 1)) ] \
        && [ "$variant_numbers" = "$expected_numbers" ]; then
        ok "t5 sections renumbered contiguously (1..$variant_count, root had $root_sections)"
    else
        bad "t5 renumbering wrong: got [$variant_numbers], expected [$expected_numbers] (root $root_sections)"
    fi

    # t6 — fail-loud when the input has no NVIDIA runtime section (the
    # derivation rule went stale — silent pass is forbidden).
    cat >"$SCRATCH/notice-no-nvidia" <<'EOF'
Fake NOTICE preamble.

--------------------------------------------------------------------------
1. Something unrelated
--------------------------------------------------------------------------
Body text.
EOF
    if bash "$GEN_NOTICE" --notice "$SCRATCH/notice-no-nvidia" --out "$SCRATCH/out-should-not-exist" >/dev/null 2>&1; then
        bad "t6 generator silently passed on a NOTICE without the NVIDIA section"
    else
        ok "t6 generator fails loudly when the NVIDIA section is absent"
    fi

    # t7 — deterministic: two runs are byte-identical.
    bash "$GEN_NOTICE" --notice "$ROOT/NOTICE" --out "$SCRATCH/NOTICE.variant2" >/dev/null 2>&1
    if cmp -s "$VARIANT" "$SCRATCH/NOTICE.variant2"; then
        ok "t7 two runs are byte-identical"
    else
        bad "t7 output is not deterministic"
    fi

    # t8 — the variant preamble names the build target neutrally.
    if grep -q "CPU + Vulkan-only build target" "$VARIANT" 2>/dev/null; then
        ok "t8 variant preamble names the CPU + Vulkan-only build target"
    else
        bad "t8 variant preamble does not name the build target"
    fi
fi

# ===========================================================================
echo ""
echo "m4-15 scratch-tree tests: $pass passed, $fail failed, $skip skipped"
[ "$fail" -eq 0 ] || exit 1
exit 0
