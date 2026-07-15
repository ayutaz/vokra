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
# [T04] NVIDIA / Metal non-link scanner
# ===========================================================================
echo ""
echo "[T04] check-cpu-vulkan-only-no-nvidia.sh"

SCANNER="$ROOT/scripts/compliance/check-cpu-vulkan-only-no-nvidia.sh"

# Helper: assemble a well-formed artifact dir (cdylib + LICENSE + NOTICE
# variant) under $1. The cdylib is a real innocuous shared object when a C
# compiler is available (higher fidelity: the nm tier actually parses it),
# else a placeholder file (the nm tier then has nothing to report — the
# same graceful shape the Godot scanner has for non-object files).
CC_BIN=""
command -v cc >/dev/null 2>&1 && CC_BIN="cc"

make_lib() {
    # make_lib <out.so> <c-source-body>
    local out="$1" body="$2" extra=""
    [ -n "$CC_BIN" ] || return 1
    case "$(uname -s)" in
        Darwin) extra="-Wl,-undefined,dynamic_lookup" ;;
    esac
    # shellcheck disable=SC2086
    printf '%s\n' "$body" | "$CC_BIN" -shared -o "$out" -xc - $extra 2>/dev/null
}

make_artifact_dir() {
    local dir="$1"
    mkdir -p "$dir"
    if [ -n "$CC_BIN" ]; then
        make_lib "$dir/libvokra.so" 'int vokra_fake(void) { return 42; }' || return 1
    else
        printf 'not-a-real-object\n' >"$dir/libvokra.so"
    fi
    cp "$ROOT/LICENSE" "$dir/LICENSE"
    bash "$GEN_NOTICE" --notice "$ROOT/NOTICE" --out "$dir/NOTICE" >/dev/null 2>&1
}

if [ ! -f "$SCANNER" ]; then
    bad "scanner script missing: $SCANNER"
else
    # s1 — clean artifact dir passes.
    make_artifact_dir "$SCRATCH/art-clean"
    if bash "$SCANNER" "$SCRATCH/art-clean" >/dev/null 2>&1; then
        ok "s1 clean artifact dir passes"
    else
        bad "s1 scanner rejected a clean artifact dir"
    fi

    # s2 — bundled NVIDIA runtime FILE is rejected (filename tier; includes
    # the lib-prefixed Linux name the Unity mirror's glob would have missed).
    make_artifact_dir "$SCRATCH/art-file"
    printf 'fake\n' >"$SCRATCH/art-file/libcudart.so.12"
    if bash "$SCANNER" "$SCRATCH/art-file" >/dev/null 2>&1; then
        bad "s2 scanner accepted a bundled libcudart.so.12"
    else
        ok "s2 bundled libcudart.so.12 rejected (exit 1)"
    fi

    # s3 — undefined NVIDIA runtime/driver symbol in the cdylib is rejected.
    if [ -n "$CC_BIN" ]; then
        make_artifact_dir "$SCRATCH/art-cuda"
        if make_lib "$SCRATCH/art-cuda/libvokra.so" \
            'extern int cudaMalloc(void); int f(void) { return cudaMalloc(); }'; then
            if bash "$SCANNER" "$SCRATCH/art-cuda" >/dev/null 2>&1; then
                bad "s3 scanner accepted a cdylib referencing cudaMalloc"
            else
                ok "s3 undefined cudaMalloc reference rejected (exit 1)"
            fi
        else
            skipped "s3 could not build the NVIDIA-symbol fixture"
        fi

        # s3b — CUDA Driver API camelCase symbol (cuInit) is also caught.
        make_artifact_dir "$SCRATCH/art-cu"
        if make_lib "$SCRATCH/art-cu/libvokra.so" \
            'extern int cuInit(void); int f(void) { return cuInit(); }'; then
            if bash "$SCANNER" "$SCRATCH/art-cu" >/dev/null 2>&1; then
                bad "s3b scanner accepted a cdylib referencing cuInit"
            else
                ok "s3b undefined cuInit reference rejected (exit 1)"
            fi
        else
            skipped "s3b could not build the driver-API fixture"
        fi

        # s4 — Metal / Objective-C runtime symbol is rejected (this build
        # target must not contain the Metal backend either).
        make_artifact_dir "$SCRATCH/art-objc"
        if make_lib "$SCRATCH/art-objc/libvokra.so" \
            'extern int objc_msgSend(void); int f(void) { return objc_msgSend(); }'; then
            if bash "$SCANNER" "$SCRATCH/art-objc" >/dev/null 2>&1; then
                bad "s4 scanner accepted a cdylib referencing objc_msgSend"
            else
                ok "s4 undefined objc_msgSend reference rejected (exit 1)"
            fi
        else
            skipped "s4 could not build the objc-symbol fixture"
        fi
    else
        skipped "s3/s3b/s4 need a C compiler for symbol fixtures"
    fi

    # s5 — missing NOTICE is rejected.
    make_artifact_dir "$SCRATCH/art-nonotice"
    rm -f "$SCRATCH/art-nonotice/NOTICE"
    if bash "$SCANNER" "$SCRATCH/art-nonotice" >/dev/null 2>&1; then
        bad "s5 scanner accepted an artifact dir without NOTICE"
    else
        ok "s5 missing NOTICE rejected (exit 1)"
    fi

    # s6 — a NOTICE that still carries the NVIDIA runtime section (root
    # NOTICE copied verbatim instead of the T05 variant) is rejected.
    make_artifact_dir "$SCRATCH/art-rootnotice"
    cp "$ROOT/NOTICE" "$SCRATCH/art-rootnotice/NOTICE"
    if bash "$SCANNER" "$SCRATCH/art-rootnotice" >/dev/null 2>&1; then
        bad "s6 scanner accepted the root NOTICE (NVIDIA section present)"
    else
        ok "s6 verbatim root NOTICE rejected (exit 1)"
    fi

    # s7 — empty LICENSE is rejected.
    make_artifact_dir "$SCRATCH/art-emptylic"
    : >"$SCRATCH/art-emptylic/LICENSE"
    if bash "$SCANNER" "$SCRATCH/art-emptylic" >/dev/null 2>&1; then
        bad "s7 scanner accepted an empty LICENSE"
    else
        ok "s7 empty LICENSE rejected (exit 1)"
    fi

    # s8 — no native library at all is rejected (the assembled artifact dir
    # must actually contain the cdylib; an empty stage is a broken pipeline,
    # not a pass — stricter than the Godot --no-build allowance, per the
    # scanner contract).
    mkdir -p "$SCRATCH/art-nolib"
    cp "$ROOT/LICENSE" "$SCRATCH/art-nolib/LICENSE"
    bash "$GEN_NOTICE" --notice "$ROOT/NOTICE" --out "$SCRATCH/art-nolib/NOTICE" >/dev/null 2>&1
    if bash "$SCANNER" "$SCRATCH/art-nolib" >/dev/null 2>&1; then
        bad "s8 scanner accepted an artifact dir without any native library"
    else
        ok "s8 artifact dir without a native library rejected (exit 1)"
    fi
fi

# ===========================================================================
echo ""
echo "m4-15 scratch-tree tests: $pass passed, $fail failed, $skip skipped"
[ "$fail" -eq 0 ] || exit 1
exit 0
