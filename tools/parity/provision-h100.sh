#!/usr/bin/env bash
# provision-h100.sh — one-shot setup for a fresh vast.ai H100 instance
# so it can run the M4-07 FlashAttention v3 Hopper bench
# (docs/handoff/m4-07-hopper-bench-handover.md).
#
# **Position in the plan** — this is the sibling of
# ``scripts/publish/vast-ai/provision.sh`` (which targets HF publish
# workloads) narrowed to the H100 / FA v3 measurement flow. The two
# scripts intentionally do NOT share code:
#
# - ``provision.sh`` handles the HF publish path (uv + hf-transfer +
#   uv Python 3.12 + huggingface_hub<0.30 pin + Docker image hardening).
# - ``provision-h100.sh`` handles the FA v3 measurement path (Hopper
#   compute_cap probe + Rust toolchain + vokra-cli release build +
#   ``VOKRA_CUDA_FA_V3_ENCODER=1`` sanity check).
#
# Keeping them separate means the FA v3 script cannot accidentally pull
# in the HF publish machinery (which needs 40 GB of scratch and a
# huggingface_hub install) and vice versa. Both are idempotent and
# rerun-safe.
#
# **Hopper gate** — the script exits 1 with a clear message if the host
# GPU is not Hopper (compute capability 9.0). Ada / Ampere / Hopper
# probes all pass the ``nvidia-smi`` sniff, but FA v3 kernels only
# compile for ``compute_90a`` — running this script on RTX 4090 would
# waste ~5 minutes of Rust build only to discover the delegate probe
# skips. The gate fires early, before any expensive step.
#
# **Zero-dep + NVIDIA EULA red-lines** (NFR-DS-02 / ``CLAUDE.md``): the
# script installs ``rustup`` + a stable toolchain but does NOT
# ``pip install`` anything, does NOT ``apt install`` cuDNN / cuBLAS /
# cuFFT, and does NOT bundle cudart. ``vokra-cli`` discovers CUDA at
# runtime via ``dlopen("libcuda.so.1")`` + ``dlopen("libnvrtc.so.12")``
# — the toolkit driver + NVRTC must already be present on the vast.ai
# base image (all ``nvidia/cuda:12.x-devel-ubuntu22.04`` variants ship
# both).
#
# **Owner does**:
#   1. Rent a vast.ai H100 instance (PCIe or SXM, VRAM 80 GB):
#      ``vastai search offers 'gpu_name=H100_PCIE num_gpus=1' --order 'dph_total'``
#      ``vastai create instance <OFFER_ID> --image nvidia/cuda:12.4.1-devel-ubuntu22.04 --disk 60``
#   2. SSH in.
#   3. ``git clone https://github.com/ayutaz/vokra.git && cd vokra`` (or
#      ``git checkout <M4-07 merge commit>`` if measuring a specific rev).
#   4. Run this script from the repo root:
#      ``./tools/parity/provision-h100.sh``
#   5. Follow ``docs/handoff/m4-07-hopper-bench-handover.md`` §1-§5 to
#      run the FA v3 measurement, then ``vastai destroy <INSTANCE_ID>``.
#
# **Options**:
#   provision-h100.sh                    # full install
#   provision-h100.sh --self-test        # dry-run: probes only, no install
#   provision-h100.sh --skip-build       # provision toolchains, skip cargo build
#   provision-h100.sh --skip-hopper-gate # bypass the SM 9.0 gate (dev only)
#   provision-h100.sh --repo-root PATH   # explicit vokra repo root
#                                        # (default: cwd, must contain Cargo.toml)

set -euo pipefail

VOKRA_ROOT="${VOKRA_ROOT:-$PWD}"
SKIP_BUILD=0
SKIP_HOPPER_GATE=0
SELF_TEST=0

log() { printf '[provision-h100] %s\n' "$*" >&2; }
step() { printf '\n\033[1;36m[provision-h100] ==== %s ====\033[0m\n' "$*" >&2; }
warn() { printf '\033[1;33m[provision-h100] WARN: %s\033[0m\n' "$*" >&2; }
err()  { printf '\033[1;31m[provision-h100] ERROR: %s\033[0m\n' "$*" >&2; }

usage() {
  cat <<'EOF' >&2
usage: provision-h100.sh [--repo-root PATH] [--skip-build] [--skip-hopper-gate]
       provision-h100.sh --self-test

Provisions a fresh vast.ai H100 instance for the M4-07 FA v3 bench:
  1. Hopper gate: nvidia-smi compute capability >= 9.0
  2. Rust toolchain (rustup + stable, once)
  3. Repo sanity (Cargo.toml at $VOKRA_ROOT, vokra-cli member present)
  4. Cargo release build of vokra-cli
  5. FA v3 lazy-compile smoke via `vokra-cli probe --backend cuda`
     (owner-visible failure surface — never silently promotes to "OK")

After it finishes, run docs/handoff/m4-07-hopper-bench-handover.md §1-§5
to collect the measurement, then `vastai destroy <INSTANCE_ID>`.

The Hopper gate can be bypassed with --skip-hopper-gate for local dev on
a non-H100 host; the FA v3 probe will report "unavailable" and the
downstream measurement will honest-skip, but the toolchain / repo /
build steps still complete so an SSH environment can be smoke-tested.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --repo-root)         VOKRA_ROOT="$2";      shift 2 ;;
    --skip-build)        SKIP_BUILD=1;         shift ;;
    --skip-hopper-gate)  SKIP_HOPPER_GATE=1;   shift ;;
    --self-test)         SELF_TEST=1;          shift ;;
    -h|--help)           usage; exit 0 ;;
    *) err "unexpected argument: '$1'"; usage; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# Probes — idempotent 0-cost checks
# ---------------------------------------------------------------------------

have_rust()      { command -v cargo    >/dev/null 2>&1; }
have_curl()      { command -v curl     >/dev/null 2>&1; }
have_nvidia_smi(){ command -v nvidia-smi >/dev/null 2>&1; }
have_repo()      { [ -f "$VOKRA_ROOT/Cargo.toml" ]; }
have_vokra_cli() { [ -x "$VOKRA_ROOT/target/release/vokra-cli" ]; }

# ---------------------------------------------------------------------------
# Hopper gate — bail early if not on H100
#
# ``nvidia-smi --query-gpu=compute_cap`` reports the capability as
# ``9.0`` for H100 / H200, ``8.9`` for Ada (RTX 4090 / L40 / L4),
# ``8.0`` for Ampere data-center (A100 / A30), ``8.6`` for Ampere
# consumer (RTX 30). Anything below 9.0 fails the gate — FA v3 kernels
# only compile for ``compute_90a`` (see the WGMMA + TMA instructions in
# the M4-07 kernel per ADR-M4-07).
#
# Compat: `--query-gpu=compute_cap` was added in driver 495 / CUDA 11.5.
# All ``nvidia/cuda:12.x-devel`` base images bundle a driver newer than
# that, so the query is safe. We still handle the "field not found"
# case defensively — a stale driver ships an older ``nvidia-smi`` that
# does not recognise the field name and prints nothing.
# ---------------------------------------------------------------------------

hopper_gate() {
  step "Hopper gate (SM 9.0 probe via nvidia-smi)"

  if [ "$SKIP_HOPPER_GATE" -eq 1 ]; then
    warn "--skip-hopper-gate set — proceeding without SM 9.0 verification"
    return 0
  fi

  if ! have_nvidia_smi; then
    err "nvidia-smi not on PATH — is this an nvidia/cuda:* base image?"
    err "the driver ships with the vast.ai runtime; a missing nvidia-smi"
    err "means the instance did not attach a GPU. Destroy and reprovision."
    exit 1
  fi

  # `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` prints
  # one row per GPU. On multi-GPU H100 SXM hosts that's 4/8 rows all
  # reading `9.0`; we take the max so a mixed configuration (H100 +
  # older card in the same box, which vast.ai occasionally lists) is
  # not blocked when at least one card is Hopper.
  local cap
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | awk -F. 'NF>=2 {v=$1*10+$2; if (v>m) m=v} END{ if (m) print m; else print "0" }')"
  if [ "$cap" = "0" ]; then
    err "nvidia-smi did not report compute_cap — the driver is too old"
    err "for the --query-gpu=compute_cap field (needs driver >= 495)."
    err "Base image should be nvidia/cuda:12.x-devel-ubuntu22.04 or newer."
    err "Bypass with --skip-hopper-gate at your own risk."
    exit 1
  fi

  if [ "$cap" -lt 90 ]; then
    err "GPU compute capability $((cap / 10)).$((cap % 10)) < 9.0 (Hopper)."
    err "  compute_cap reported: $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1)"
    err "FA v3 kernels only compile for compute_90a. Rerun on an H100"
    err "instance (\`gpu_name=H100_PCIE\` or \`H100_SXM\` in vast.ai search),"
    err "or pass --skip-hopper-gate to continue with the toolchain install"
    err "for a dev-only smoke test (the FA v3 probe will honest-skip)."
    exit 1
  fi

  log "OK: Hopper detected (compute_cap = $((cap / 10)).$((cap % 10)))"
  log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  log "Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
}

# ---------------------------------------------------------------------------
# Rust toolchain — install once via rustup
# ---------------------------------------------------------------------------

install_rust() {
  step "Rust toolchain (rustup + stable)"

  if have_rust; then
    log "OK: cargo already on PATH ($(cargo --version 2>/dev/null || echo unknown))"
    return 0
  fi

  if ! have_curl; then
    err "curl not on PATH — cannot download rustup. Install curl first."
    exit 1
  fi

  # Standard rustup one-liner. Pinned to the same recipe the CUDA
  # measurement handoff uses (see tools/parity/README-cuda-rtf-variance.md
  # §"Owner workflow (vast.ai)"). We NOT pin to a fixed rustc version
  # here — the M4-07 handoff instructs the owner to `git checkout` the
  # M4-07 branch first, so the checkout's `rust-toolchain.toml` (if
  # any) becomes the source of truth.
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain stable

  # shellcheck disable=SC1091
  . "$HOME/.cargo/env"

  log "installed: $(cargo --version 2>/dev/null || echo unknown)"
}

# ---------------------------------------------------------------------------
# Repo sanity — the caller must run this from the vokra repo root, or
# pass --repo-root PATH
# ---------------------------------------------------------------------------

check_repo() {
  step "Repo sanity ($VOKRA_ROOT)"

  if ! have_repo; then
    err "$VOKRA_ROOT does not contain Cargo.toml"
    err "cd to the vokra repo root first, or pass --repo-root PATH"
    exit 1
  fi

  if ! grep -q '"crates/vokra-cli"' "$VOKRA_ROOT/Cargo.toml" 2>/dev/null; then
    err "$VOKRA_ROOT/Cargo.toml does not list crates/vokra-cli — wrong repo?"
    exit 1
  fi

  log "OK: $VOKRA_ROOT is a vokra workspace"
}

# ---------------------------------------------------------------------------
# Build vokra-cli in release mode
# ---------------------------------------------------------------------------

build_vokra_cli() {
  step "Cargo release build (vokra-cli)"

  if [ "$SKIP_BUILD" -eq 1 ]; then
    warn "--skip-build set — skipping cargo build"
    return 0
  fi

  if have_vokra_cli; then
    log "OK: vokra-cli release binary already at $VOKRA_ROOT/target/release/vokra-cli"
    return 0
  fi

  (cd "$VOKRA_ROOT" && cargo build --release -p vokra-cli 2>&1 | tail -3)

  if ! have_vokra_cli; then
    err "cargo build succeeded but vokra-cli binary was not produced?"
    exit 1
  fi

  log "OK: built $VOKRA_ROOT/target/release/vokra-cli"
}

# ---------------------------------------------------------------------------
# FA v3 lazy-compile smoke via `vokra-cli probe`
#
# This is a fast (< 5 s) confidence check that the FA v3 kernel path is
# reachable from the built binary. It does NOT run the parity tests
# (that is docs/handoff/m4-07-hopper-bench-handover.md §2) — the goal
# here is only to catch "wrong branch checkout" / "CUDA driver too old
# for compute_90a NVRTC" cases before the owner burns 4 minutes on the
# real RTF variance harness.
#
# Deliberately silent on non-Hopper hosts (already gated) or on a
# vokra-cli that predates the M4-07 `probe --backend cuda` flag — this
# is an *advisory* probe, never a gate. The M4-07 handoff §2 is where
# the real verification happens.
# ---------------------------------------------------------------------------

probe_fa_v3() {
  step "FA v3 probe (advisory — real verify is m4-07 handoff §2)"

  if [ "$SKIP_HOPPER_GATE" -eq 1 ]; then
    warn "--skip-hopper-gate was set — FA v3 probe will report unavailable"
  fi

  local out
  # `probe --backend cuda` is expected to print backend + capabilities.
  # Fall back to `--help` if the subcommand is absent (older revs).
  if ! out="$("$VOKRA_ROOT/target/release/vokra-cli" probe --backend cuda 2>&1)"; then
    warn "vokra-cli probe --backend cuda exited non-zero; not fatal for provisioning"
    warn "reason: '$(printf '%s' "$out" | head -1)'"
    warn "proceed to m4-07 handoff §2 to run the real verification"
    return 0
  fi

  printf '%s\n' "$out" | head -20 >&2
  log "probe complete — proceed to docs/handoff/m4-07-hopper-bench-handover.md §2"
}

# ---------------------------------------------------------------------------
# Self-test — probes only, no install
# ---------------------------------------------------------------------------

self_test() {
  step "Self-test (probes only, no install)"

  local rc=0

  if have_rust;       then log "cargo:      present"; else log "cargo:      absent (would install rustup)"; rc=1; fi
  if have_curl;       then log "curl:       present"; else log "curl:       absent (needed for rustup)"; rc=1; fi
  if have_nvidia_smi; then log "nvidia-smi: present"; else log "nvidia-smi: absent (not a CUDA host?)"; rc=1; fi
  if have_repo;       then log "repo:       present at $VOKRA_ROOT"; else log "repo:       Cargo.toml missing at $VOKRA_ROOT"; rc=1; fi
  if have_vokra_cli;  then log "vokra-cli:  present";  else log "vokra-cli:  not built (would run cargo build)"; fi

  if [ "$SKIP_HOPPER_GATE" -eq 0 ] && have_nvidia_smi; then
    local cap
    cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
          | awk -F. 'NF>=2 {v=$1*10+$2; if (v>m) m=v} END{ if (m) print m; else print "0" }')"
    if [ "$cap" -ge 90 ]; then
      log "hopper:     OK (compute_cap = $((cap / 10)).$((cap % 10)))"
    else
      log "hopper:     FAIL (compute_cap = $((cap / 10)).$((cap % 10)) < 9.0)"
      rc=1
    fi
  else
    log "hopper:     skipped"
  fi

  if [ "$rc" -eq 0 ]; then
    log "self-test: all probes passed"
  else
    warn "self-test: at least one probe failed — see logs above"
  fi
  return "$rc"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if [ "$SELF_TEST" -eq 1 ]; then
  self_test
  exit $?
fi

hopper_gate
install_rust
check_repo
build_vokra_cli
probe_fa_v3

step "Done"
log "Next steps: docs/handoff/m4-07-hopper-bench-handover.md §2 onward"
log "Remember: after measurement, destroy the instance (\`vastai destroy <ID>\`)"
