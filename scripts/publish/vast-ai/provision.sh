#!/usr/bin/env bash
# provision.sh — one-shot setup for a fresh vast.ai instance so it can
# run scripts/publish/vast-ai/run-one.sh.
#
# 2026-07-28 policy (memory feedback-large-models-on-vast-ai): convert +
# upload for a HF weight default to vast.ai. This script encodes the
# runbook (docs/handoff/vast-ai-large-model-publish.md §2.3 / §2.4) so
# owner runs one command per fresh instance, not 15.
#
# 2026-08-03 Wave 12 addition (harden_vast_docker_image): the stock
# nvidia/cuda:13.0.0 vast.ai image ships with (A) a hf_config.pth
# site-packages shim that reroutes huggingface_hub through a broken
# mirror, (B) a huggingface_hub >= 0.30 that regressed non-xet routes,
# (C) an empty/stale certifi CA bundle, and (D) no torch/numpy/
# safetensors at the system layer that resilient_batch.sh's uv_cmd
# fallback + ad-hoc `python3 -c` consume. Waves 9-11 spent ~day burning
# down these four root causes reactively. Fix is now pre-handled at
# provision time so a fresh box comes up clean.
#
# Idempotent: rerun-safe. Each step probes for its own artifact and skips
# if already installed. Safe to invoke after `git pull` on the same box.
#
# Owner does:
#   1. Rent a vast.ai instance (≥64 GB RAM, ≥200 GB disk, cheapest GPU).
#   2. SSH in.
#   3. Set HF_TOKEN in the shell:  export HF_TOKEN='hf_xxxxxxxx'
#   4. curl this script to /root/, or `git clone` the repo first.
#   5. Run this script.
#
# After it finishes, run scripts/publish/vast-ai/run-one.sh per model.
#
# Usage:
#   provision.sh                       # full install
#   provision.sh --self-test           # dry-run: check idempotency probes
#   provision.sh --repo-url <url>      # clone from custom URL (default: public GitHub)
#   provision.sh --branch <name>       # checkout non-main branch (default: main)
#   provision.sh --skip-build          # provision toolchains but skip cargo build

set -euo pipefail

VOKRA_REPO_URL="${VOKRA_REPO_URL:-https://github.com/ayutaz/vokra.git}"
VOKRA_BRANCH="${VOKRA_BRANCH:-main}"
VOKRA_ROOT="${VOKRA_ROOT:-$HOME/vokra}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"

log() { printf '[provision] %s\n' "$*" >&2; }
step() { printf '\n\033[1;36m[provision] ==== %s ====\033[0m\n' "$*" >&2; }

usage() {
  cat <<'EOF' >&2
usage: provision.sh [--repo-url <url>] [--branch <name>] [--skip-build]
       provision.sh --self-test

Provisions a fresh vast.ai instance for Vokra HF publish:
  1. Rust toolchain (rustup + stable)
  2. uv + Python 3.12
  3. hf-transfer for 40x HF upload speedup
  4. Repo clone at $VOKRA_ROOT
  5. Cargo release build of vokra-cli
  6. $VOKRA_SCRATCH scratch dirs (hf-cache, staging)
  7. Adds `export VOKRA_PUBLISH_ON_VAST=1` to ~/.bashrc so publish-one.sh
     gate 7 auto-bypasses on this instance.

HF_TOKEN must be set in env before publish (not needed for provision itself).
EOF
}

# Probes — each returns 0 if the artifact exists and no install is needed.
have_rust()        { command -v cargo    >/dev/null 2>&1; }
have_hf_shim()     {
  [[ -e /usr/local/lib/python3.10/dist-packages/hf_config.pth \
    || -e /usr/lib/python3/dist-packages/pip/_vendor/hf_config.pth ]]
}
have_uv()          { command -v uv       >/dev/null 2>&1; }
have_hf_transfer() {
  uv run --with hf-transfer python -c 'import hf_transfer' 2>/dev/null
}
have_repo()        { [[ -d "$VOKRA_ROOT/.git" ]]; }
have_vokra_cli()   { [[ -x "$VOKRA_ROOT/target/release/vokra-cli" ]]; }

# --- Wave 12 pre-handle (vast.ai nvidia/cuda:13.0.0 image hardening) ---
# Fixes four root causes reactively burned down in Waves 9-11:
#   (A) hf_config.pth mirror shim in site-packages reroutes hh downloads
#   (B) huggingface_hub >= 0.30 regressed non-xet routes
#   (C) stale/empty certifi CA bundle breaks urllib3/hh internals
#   (D) system-layer torch/numpy/safetensors absent — resilient_batch.sh's
#       uv_cmd fallback + ad-hoc `python3 -c` ImportError before use
#
# Idempotent. Non-vast guarded: bails cleanly on macOS/local dev via
# `command -v apt-get` + EUID probe. Set VOKRA_FORCE_HARDEN=1 to force
# on a rooted local Debian container. MUST run before install_uv /
# install_hf_transfer so shim (A) does not intercept uv operations.
harden_vast_docker_image() {
  step "Harden Docker image (rm HF shim / refresh CA / pin hh<0.30)"

  # Non-vast guard: skip cleanly on macOS/local dev where apt is absent.
  if ! command -v apt-get >/dev/null 2>&1; then
    log "no apt-get on PATH — not a Debian/Ubuntu image, skipping hardening"
    return 0
  fi
  if [[ $EUID -ne 0 ]]; then
    if [[ "${VOKRA_FORCE_HARDEN:-0}" != "1" ]]; then
      log "not root — skipping system hardening (set VOKRA_FORCE_HARDEN=1 to try)"
      return 0
    fi
    log "VOKRA_FORCE_HARDEN=1 set — attempting hardening as EUID=$EUID"
  fi

  # Fix (A): remove HF mirror shim. Load-bearing: must fire before any
  # pip/uv operation runs, otherwise huggingface_hub installs go through
  # the mirror during install itself. `rm -f` is no-op when clean; but
  # if the file exists and rm fails, set -e kills provision.sh, which is
  # intentional (proceeding with a live shim guarantees the bug recurs).
  local shim
  for shim in \
    /usr/local/lib/python3.10/dist-packages/hf_config.pth \
    /usr/lib/python3/dist-packages/pip/_vendor/hf_config.pth
  do
    if [[ -e "$shim" ]]; then
      rm -f "$shim"
      log "removed shim: $shim"
    else
      log "no shim at $shim (clean)"
    fi
  done

  # Fix (C, prep): restore CA bundle. --reinstall is designed for repeat.
  # Run before Fix (B) so pip's TLS to PyPI is trustworthy.
  if apt-get install --reinstall -y ca-certificates >/dev/null 2>&1 \
    && update-ca-certificates >/dev/null 2>&1; then
    log "ca-certificates reinstalled + refreshed"
  else
    log "WARN: ca-certificates reinstall failed — continuing"
  fi

  # Fix (B) + Fix (D): pin huggingface_hub<0.30 (Wave 9-11 empirical
  # ceiling — relax when upstream restores non-xet backward compat or
  # vast.ai's mirror stops mangling xet routes) + pre-install
  # torch/numpy/safetensors/certifi at the SYSTEM layer where
  # resilient_batch.sh's uv_cmd fallback + ad-hoc `python3 -c` actually
  # consume them. tools/parity's per-tree uv env stays lean.
  if command -v pip3 >/dev/null 2>&1; then
    if pip3 install --quiet --upgrade \
      'huggingface_hub<0.30' torch numpy safetensors certifi; then
      log "pip3: hh<0.30 pin + torch/numpy/safetensors/certifi installed"
    else
      log "WARN: pip3 install failed — uv path may still work"
    fi
  else
    log "note: pip3 not on PATH — skipping system-layer python installs"
  fi

  # Fix (C, apply): overwrite certifi bundle with system CA. This covers
  # code that resolves through certifi.where() directly (hh internals,
  # urllib3, some torch loaders) — resilient_batch.sh's runtime
  # SSL_CERT_FILE / REQUESTS_CA_BUNDLE exports do NOT cover this path.
  local certifi_where
  certifi_where="$(python3 -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
  if [[ -n "$certifi_where" && -f /etc/ssl/certs/ca-certificates.crt ]]; then
    if cp /etc/ssl/certs/ca-certificates.crt "$certifi_where" 2>/dev/null; then
      log "certifi bundle synced: $certifi_where"
    else
      log "WARN: certifi bundle cp failed — continuing"
    fi
  else
    log "note: certifi.where() unresolved or system CA missing — skipping"
  fi
}

install_rust() {
  step "Rust toolchain"
  if have_rust; then
    log "rustup/cargo already present ($(cargo --version)) — skipping"
    return 0
  fi
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
  log "installed: $(cargo --version)"
}

install_uv() {
  step "uv + Python 3.12"
  if have_uv; then
    log "uv already present ($(uv --version)) — skipping install"
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv installer writes to ~/.local/bin. Not always on PATH in fresh shells.
    export PATH="$HOME/.local/bin:$PATH"
    # shellcheck disable=SC1091
    [[ -f "$HOME/.local/bin/env" ]] && source "$HOME/.local/bin/env" || true
    log "installed: $(uv --version)"
  fi
  # feedback-python-3-12: pin CPython 3.12, not older.
  uv python install 3.12 || log "uv python install 3.12 exited non-zero (may already be installed)"
}

install_hf_transfer() {
  step "hf-transfer (40x HF upload speedup)"
  if have_hf_transfer; then
    log "hf-transfer already resolvable via uv — skipping"
    return 0
  fi
  # hf-transfer is a Rust-backed helper huggingface_hub picks up when the
  # HF_HUB_ENABLE_HF_TRANSFER=1 env var is set. Installing into a uv-managed
  # venv keeps it isolated from the system Python (feedback-python-uses-uv).
  # We put the shim in $VOKRA_ROOT/tools/parity where uv sync will resolve it.
  if [[ -d "$VOKRA_ROOT/tools/parity" && -f "$VOKRA_ROOT/tools/parity/pyproject.toml" ]]; then
    ( cd "$VOKRA_ROOT/tools/parity" && uv add hf-transfer huggingface_hub )
  else
    log "note: $VOKRA_ROOT/tools/parity not present yet — hf-transfer will be resolved on first run-one.sh via --with"
  fi
}

clone_repo() {
  step "Repo clone ($VOKRA_REPO_URL @ $VOKRA_BRANCH)"
  if have_repo; then
    log "$VOKRA_ROOT already a git checkout — running git fetch + checkout"
    ( cd "$VOKRA_ROOT" && git fetch origin && git checkout "$VOKRA_BRANCH" && git pull --ff-only origin "$VOKRA_BRANCH" )
    return 0
  fi
  git clone --branch "$VOKRA_BRANCH" "$VOKRA_REPO_URL" "$VOKRA_ROOT"
  log "cloned to $VOKRA_ROOT"
}

build_vokra_cli() {
  step "cargo build --release -p vokra-cli"
  if have_vokra_cli; then
    log "target/release/vokra-cli already built — skipping (rebuild manually if repo pulled)"
    return 0
  fi
  ( cd "$VOKRA_ROOT" && cargo build --release -p vokra-cli )
  log "built: $VOKRA_ROOT/target/release/vokra-cli"
}

setup_scratch() {
  step "Scratch dirs"
  mkdir -p "$VOKRA_SCRATCH/hf-cache" "$VOKRA_SCRATCH/staging"
  log "hf-cache : $VOKRA_SCRATCH/hf-cache"
  log "staging  : $VOKRA_SCRATCH/staging"
}

# Adds `export VOKRA_PUBLISH_ON_VAST=1` to ~/.bashrc idempotently so gate 7
# in publish-one.sh auto-bypasses on this instance. Also to ~/.profile for
# non-interactive shells (some vast.ai images run scripts via /bin/sh).
mark_as_vast_instance() {
  step "Mark shell as vast.ai (VOKRA_PUBLISH_ON_VAST=1)"
  local marker='export VOKRA_PUBLISH_ON_VAST=1  # vokra provision.sh: publish-one.sh gate 7 auto-bypass'
  local rc
  for rc in "$HOME/.bashrc" "$HOME/.profile"; do
    [[ -e "$rc" ]] || touch "$rc"
    if ! grep -Fq "VOKRA_PUBLISH_ON_VAST=1" "$rc"; then
      printf '\n%s\n' "$marker" >> "$rc"
      log "added marker to $rc"
    else
      log "marker already in $rc — skipping"
    fi
  done
  # Also export for the current shell so a same-session run-one.sh sees it.
  export VOKRA_PUBLISH_ON_VAST=1
}

sanity_hf_token() {
  step "HF_TOKEN sanity"
  if [[ -z "${HF_TOKEN:-${HF:-}}" ]]; then
    log "WARN: HF_TOKEN / HF not set in env"
    log "  Publish will fail. Before run-one.sh:  export HF_TOKEN='hf_xxxxxx'"
    log "  (HF token has to be set per-session; a fresh instance will forget it.)"
  else
    log "HF_TOKEN present (len=${#HF_TOKEN:-0}, first 6 chars: ${HF_TOKEN:0:6}...)"
  fi
}

# --- self-test -----------------------------------------------------------
# Non-mutating dry-run: probe each artifact and print the verdict, so owner
# can pre-check idempotency on an already-provisioned box without kicking
# off any install.
run_self_test() {
  local cases=0 fail=0
  echo "provision.sh self-test — probes only, no installs"
  cases=$((cases + 1))
  if have_rust;        then echo "  [ok]   Rust:         $(cargo --version 2>/dev/null || echo 'unknown')"; else echo "  [need] Rust:         not installed"; fi
  cases=$((cases + 1))
  if have_uv;          then echo "  [ok]   uv:           $(uv --version 2>/dev/null || echo 'unknown')"; else echo "  [need] uv:           not installed"; fi
  cases=$((cases + 1))
  if have_hf_transfer; then echo "  [ok]   hf-transfer:  resolvable"; else echo "  [need] hf-transfer:  not resolvable via uv"; fi
  cases=$((cases + 1))
  if have_repo;        then echo "  [ok]   repo:         $VOKRA_ROOT is a git checkout"; else echo "  [need] repo:         $VOKRA_ROOT not a git checkout"; fi
  cases=$((cases + 1))
  if have_vokra_cli;   then echo "  [ok]   vokra-cli:    $VOKRA_ROOT/target/release/vokra-cli"; else echo "  [need] vokra-cli:    not built"; fi

  # ~/.bashrc marker probe
  cases=$((cases + 1))
  if [[ -f "$HOME/.bashrc" ]] && grep -Fq "VOKRA_PUBLISH_ON_VAST=1" "$HOME/.bashrc"; then
    echo "  [ok]   bashrc:       VOKRA_PUBLISH_ON_VAST=1 present"
  else
    echo "  [need] bashrc:       VOKRA_PUBLISH_ON_VAST=1 not persisted"
  fi

  # Wave 12: HF mirror shim probe (both known site-packages paths).
  cases=$((cases + 1))
  if ! have_hf_shim; then
    echo "  [ok]   hf-shim:      absent"
  else
    echo "  [need] hf-shim:      present — re-run provision.sh to purge"
  fi

  # HF_TOKEN sanity (not really a probe — always advisory)
  cases=$((cases + 1))
  if [[ -n "${HF_TOKEN:-${HF:-}}" ]]; then
    echo "  [ok]   HF_TOKEN:     set in env"
  else
    echo "  [need] HF_TOKEN:     unset (fine now, but must be set before run-one.sh)"
  fi

  echo "provision.sh self-test: $cases probes evaluated (no side effects)"
  return 0
}

# --- main ----------------------------------------------------------------

main() {
  local self_test=0 skip_build=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --self-test)  self_test=1;   shift ;;
      --repo-url)   VOKRA_REPO_URL="$2"; shift 2 ;;
      --branch)     VOKRA_BRANCH="$2";   shift 2 ;;
      --skip-build) skip_build=1;  shift ;;
      -h|--help)    usage; exit 0 ;;
      *)            echo "provision.sh: unknown flag '$1'" >&2; usage; exit 2 ;;
    esac
  done

  if [[ $self_test -eq 1 ]]; then
    run_self_test
    exit $?
  fi

  harden_vast_docker_image   # Wave 12 pre-handle — MUST run before install_uv
  install_rust
  install_uv
  clone_repo
  install_hf_transfer   # after clone: tools/parity is present
  setup_scratch
  if [[ $skip_build -eq 0 ]]; then
    build_vokra_cli
  else
    log "--skip-build: leaving cargo build to owner"
  fi
  mark_as_vast_instance
  sanity_hf_token

  step "provision complete"
  log "next: source ~/.bashrc     # or open a new shell — picks up VOKRA_PUBLISH_ON_VAST=1"
  log "      $VOKRA_ROOT/scripts/publish/vast-ai/run-one.sh --help"
}

main "$@"
