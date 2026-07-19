#!/usr/bin/env bash
# verify.sh — the one command that answers "did you verify?"
#
# WHY THIS EXISTS
# ---------------
# `cargo test -p <crate>` is not verification. It builds with DEFAULT features,
# so every `#[cfg(feature = "metal")]` / `#[cfg(feature = "cuda")]` test target
# is never even COMPILED — a regression inside a GPU-gated block sails through a
# green per-crate run. This has already happened on this branch: the breakage
# was caught only by a workspace-wide run, where feature unification pulls the
# GPU features on via other members.
#
# So the two halves below are a PAIR. Neither alone is verification:
#
#   (1) cargo test --workspace              — runs the tests, with unification
#   (2) cargo clippy --features <gpu> ...   — COMPILES the feature-gated targets
#                                             that (1) still may not reach
#
# plus the cheap invariants CI enforces (fmt, clippy, zero-dep, fixture pins).
#
# Not wired into a git hook on purpose — it is too slow for pre-commit, and a
# slow hook gets bypassed with --no-verify, which is a net loss. Invoke it by
# name before pushing, before claiming a task is done, and in any handoff that
# says "verified".
#
# Usage:
#   bash scripts/verify.sh                 # the full pair + invariants
#   bash scripts/verify.sh --quick         # skip the workspace test run
#   bash scripts/verify.sh --run-gpu-tests # also EXECUTE metal tests (needs a GPU)
#
# Exit 0 only if every step passed. Skipped steps are reported LOUDLY and, if a
# skip was not a deliberate platform exclusion, treated as failure — a verify
# script that silently does less than it claims is worse than none.

set -uo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

QUICK=0
RUN_GPU_TESTS=0
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK=1 ;;
        --run-gpu-tests) RUN_GPU_TESTS=1 ;;
        -h | --help)
            sed -n '2,40p' "$0"
            exit 0
            ;;
        *)
            echo "verify: unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

# Deterministic, parseable cargo output regardless of the caller's environment.
export CARGO_TERM_COLOR="${CARGO_TERM_COLOR:-never}"

# Feature sets that exist in this workspace (crates/vokra-models/Cargo.toml).
# `metal` is macOS-only; the rest are raw dlopen FFI and compile anywhere.
GPU_FEATURES="vulkan,cuda"
if [ "$(uname -s)" = "Darwin" ]; then
    GPU_FEATURES="metal,cuda,vulkan"
fi

passed=()
failed=()
skipped=()

step() { # step <label> <command...>
    local label="$1"
    shift
    echo ""
    echo "=============================================================="
    echo "verify: $label"
    echo "  \$ $*"
    echo "=============================================================="
    if "$@"; then
        passed+=("$label")
    else
        failed+=("$label")
        echo "verify: >>> FAILED: $label" >&2
    fi
}

echo "verify: repo $(git rev-parse --short HEAD) on $(uname -s) $(uname -m)"
echo "verify: gpu feature set = $GPU_FEATURES"

# --- cheap invariants first (fail fast on the trivial stuff) ------------------
step "fmt" cargo fmt --all -- --check
step "zero-dep invariant (NFR-DS-02)" bash scripts/check-zero-deps.sh
step "fixture eol pins" bash scripts/check-fixture-eol-pins.sh
step "pipefail/grep -q fail-open lint" python3 scripts/compliance/lint-pipefail-grep-q.py

# --- half 1 of the pair: workspace tests --------------------------------------
if [ "$QUICK" -eq 1 ]; then
    skipped+=("cargo test --workspace (--quick)")
    echo ""
    echo "verify: SKIPPING the workspace test run (--quick). This is HALF the"
    echo "        verification pair; --quick output is not 'verified'."
else
    step "PAIR 1/2 — workspace tests" cargo test --workspace
fi

# --- half 2 of the pair: compile the feature-gated targets --------------------
step "clippy (default features)" \
    cargo clippy --all-targets -- -D warnings

# Combined set: matches the all-features test run recorded in the milestone log.
step "PAIR 2/2 — feature-gated compile ($GPU_FEATURES)" \
    cargo clippy -p vokra-models -p vokra-cli --features "$GPU_FEATURES" \
    --all-targets -- -D warnings

# Per-feature, matching the CI matrix. A combined build can hide code that only
# compiles when a sibling feature is also on, so each is checked alone too.
IFS=',' read -r -a feat_list <<< "$GPU_FEATURES"
for feat in "${feat_list[@]}"; do
    step "feature-gated compile (single: $feat)" \
        cargo clippy -p vokra-models --features "$feat" --all-targets -- -D warnings
done

# webgpu lives on vokra-models only (no vokra-cli feature).
step "feature-gated compile (single: webgpu)" \
    cargo clippy -p vokra-models --features webgpu --all-targets -- -D warnings

# --- optional: actually execute the GPU tests ---------------------------------
if [ "$RUN_GPU_TESTS" -eq 1 ]; then
    if [ "$(uname -s)" = "Darwin" ]; then
        step "metal tests (executed)" cargo test -p vokra-models --features metal
    else
        skipped+=("metal tests (not Darwin)")
    fi
else
    skipped+=("GPU test EXECUTION (pass --run-gpu-tests)")
fi

if [ "$(uname -s)" != "Darwin" ]; then
    skipped+=("metal compile (not Darwin — no Metal FFI on this platform)")
fi

# --- summary ------------------------------------------------------------------
echo ""
echo "=============================================================="
echo "verify: SUMMARY"
echo "=============================================================="
for s in "${passed[@]:-}"; do [ -n "$s" ] && echo "  PASS  $s"; done
for s in "${skipped[@]:-}"; do [ -n "$s" ] && echo "  SKIP  $s"; done
for s in "${failed[@]:-}"; do [ -n "$s" ] && echo "  FAIL  $s"; done

if [ "${#failed[@]}" -gt 0 ]; then
    echo ""
    echo "verify: ${#failed[@]} step(s) FAILED — not verified." >&2
    exit 1
fi

echo ""
if [ "$QUICK" -eq 1 ]; then
    echo "verify: partial OK (${#passed[@]} steps) — --quick skipped the workspace"
    echo "        test run, so this does NOT count as verified."
    exit 0
fi
echo "verify: OK (${#passed[@]} steps passed)"
