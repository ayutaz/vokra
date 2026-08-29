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
# safetensors at the system layer needed for image compatibility. Vokra
# scripts themselves always execute Python through uv. Waves 9-11 spent
# ~day burning down these four root causes reactively. Fix is now
# pre-handled at provision time so a fresh box comes up clean.
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
RUSTUP_INIT_VERSION="1.29.0"
RUSTUP_INIT_SHA256="4acc9acc76d5079515b46346a485974457b5a79893cfb01112423c89aeb5aa10"
RUSTUP_INIT_URL="https://static.rust-lang.org/rustup/archive/${RUSTUP_INIT_VERSION}/x86_64-unknown-linux-gnu/rustup-init"
UV_VERSION="0.12.5"
UV_ARCHIVE_SHA256="68a509da24b06b4223a1c0175fb5eb5bc79342b76cbeff0cfe51ac3f5b17b6b2"
UV_ARCHIVE_URL="https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz"

log() { printf '[provision] %s\n' "$*" >&2; }
step() { printf '\n\033[1;36m[provision] ==== %s ====\033[0m\n' "$*" >&2; }

usage() {
  cat <<'EOF' >&2
usage: provision.sh [--repo-url <url>] [--branch <name>] [--skip-build]
       provision.sh --self-test

Provisions a fresh vast.ai instance for Vokra HF publish:
  1. Rust toolchain (rustup + stable)
  2. uv + Python 3.12
  3. Repo clone at $VOKRA_ROOT
  4. Cargo release build of vokra-cli
  5. $VOKRA_SCRATCH scratch dirs (hf-cache, staging)
  6. Adds `export VOKRA_PUBLISH_ON_VAST=1` to ~/.bashrc so publish-one.sh
     gate 7 auto-bypasses on this instance.

The run-one download path resolves its pinned Hugging Face dependencies in
its own uv invocation. Provisioning does not edit any repository dependency
file or create a shared parity environment.

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
have_repo()        { [[ -d "$VOKRA_ROOT/.git" ]]; }
have_vokra_cli()   { [[ -x "$VOKRA_ROOT/target/release/vokra-cli" ]]; }

# --- Wave 12 pre-handle (vast.ai nvidia/cuda:13.0.0 image hardening) ---
# Fixes four root causes reactively burned down in Waves 9-11:
#   (A) hf_config.pth mirror shim in site-packages reroutes hh downloads
#   (B) huggingface_hub >= 0.30 regressed non-xet routes
#   (C) stale/empty certifi CA bundle breaks urllib3/hh internals
#   (D) system-layer torch/numpy/safetensors absent for image compatibility
#       tools (Vokra's own Python execution remains uv-managed)
#
# Idempotent. Non-vast guarded: bails cleanly on macOS/local dev via
# `command -v apt-get` + EUID probe. Set VOKRA_FORCE_HARDEN=1 to force
# on a rooted local Debian container. This runs immediately after install_uv:
# Fix (A) is performed before its first Hugging Face package operation, while
# every Python dependency operation itself stays on uv.
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
  # Hugging Face package operation runs, otherwise huggingface_hub installs
  # go through the mirror during install itself. `rm -f` is no-op when clean; but
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
  # torch/numpy/safetensors/certifi at the SYSTEM layer as a VAST image
  # compatibility layer. tools/parity's per-tree uv environment remains the
  # only Python execution path for Vokra itself.
  if command -v uv >/dev/null 2>&1; then
    if uv pip install --system --break-system-packages --quiet --upgrade \
      'huggingface_hub<0.30' torch numpy safetensors certifi; then
      log "uv pip --system: hh<0.30 pin + torch/numpy/safetensors/certifi installed"
    else
      log "WARN: uv pip --system install failed — project uv env may still work"
    fi
  else
    log "note: uv not on PATH — skipping system-layer Python installs"
  fi

  # Fix (C, apply): overwrite certifi bundle with system CA. This covers
  # code that resolves through certifi.where() directly (hh internals,
  # urllib3, some torch loaders) — resilient_batch.sh's runtime
  # SSL_CERT_FILE / REQUESTS_CA_BUNDLE exports do NOT cover this path.
  local certifi_location certifi_where
  certifi_location="$(uv pip show --system certifi 2>/dev/null | awk -F ': ' '$1 == "Location" { print $2; exit }' || true)"
  certifi_where="${certifi_location:+$certifi_location/certifi/cacert.pem}"
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
  local rustup_dir rustup_init actual_sha256
  rustup_dir="$(mktemp -d "${TMPDIR:-/tmp}/vokra-rustup-init.XXXXXX")"
  rustup_init="$rustup_dir/rustup-init"
  curl --proto '=https' --tlsv1.2 -sSfL \
    --output "$rustup_init" "$RUSTUP_INIT_URL"
  actual_sha256="$(sha256sum "$rustup_init" | awk '{print $1}')"
  if [[ "$actual_sha256" != "$RUSTUP_INIT_SHA256" ]]; then
    rm -f "$rustup_init"
    rmdir "$rustup_dir"
    log "ERROR: rustup-init SHA-256 mismatch (expected $RUSTUP_INIT_SHA256, got $actual_sha256)"
    exit 1
  fi
  chmod 700 "$rustup_init"
  "$rustup_init" -y --default-toolchain stable
  rm -f "$rustup_init"
  rmdir "$rustup_dir"
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
  log "installed: $(cargo --version)"
}

install_uv() {
  step "uv + Python 3.12"
  if have_uv; then
    log "uv already present ($(uv --version)) — skipping install"
  else
    local uv_archive uv_unpack actual_sha256
    uv_archive="$(mktemp "${TMPDIR:-/tmp}/vokra-uv.XXXXXX.tar.gz")"
    uv_unpack="$(mktemp -d "${TMPDIR:-/tmp}/vokra-uv-unpack.XXXXXX")"
    curl --proto '=https' --tlsv1.2 -sSfL \
      --output "$uv_archive" "$UV_ARCHIVE_URL"
    actual_sha256="$(sha256sum "$uv_archive" | awk '{print $1}')"
    if [[ "$actual_sha256" != "$UV_ARCHIVE_SHA256" ]]; then
      rm -f "$uv_archive"
      rm -rf "$uv_unpack"
      log "ERROR: uv archive SHA-256 mismatch (expected $UV_ARCHIVE_SHA256, got $actual_sha256)"
      exit 1
    fi
    tar -xzf "$uv_archive" -C "$uv_unpack"
    mkdir -p "$HOME/.local/bin"
    install -m 755 "$uv_unpack/uv-x86_64-unknown-linux-gnu/uv" "$HOME/.local/bin/uv"
    install -m 755 "$uv_unpack/uv-x86_64-unknown-linux-gnu/uvx" "$HOME/.local/bin/uvx"
    rm -f "$uv_archive"
    rm -rf "$uv_unpack"
    # The verified archive is installed to ~/.local/bin. Not always on PATH in fresh shells.
    export PATH="$HOME/.local/bin:$PATH"
    # shellcheck disable=SC1091
    [[ -f "$HOME/.local/bin/env" ]] && source "$HOME/.local/bin/env" || true
    log "installed: $(uv --version)"
  fi
  # feedback-python-3-12: pin CPython 3.12, not older.
  uv python install 3.12 || log "uv python install 3.12 exited non-zero (may already be installed)"
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
  local token="${HF_TOKEN:-${HF:-}}"
  if [[ -z "$token" ]]; then
    log "WARN: HF_TOKEN / HF not set in env"
    log "  Publish will fail. Before run-one.sh:  export HF_TOKEN='hf_xxxxxx'"
    log "  (HF token has to be set per-session; a fresh instance will forget it.)"
  else
    log "HF_TOKEN present (len=${#token}, first 6 chars: ${token:0:6}...)"
  fi
}

# --- self-test -----------------------------------------------------------
# Non-mutating dry-run: probe each artifact and print the verdict, so owner
# can pre-check idempotency on an already-provisioned box without kicking
# off any install.
run_self_test() {
  local cases=0
  echo "provision.sh self-test — probes only, no installs"
  cases=$((cases + 1))
  if have_rust;        then echo "  [ok]   Rust:         $(cargo --version 2>/dev/null || echo 'unknown')"; else echo "  [need] Rust:         not installed"; fi
  cases=$((cases + 1))
  if have_uv;          then echo "  [ok]   uv:           $(uv --version 2>/dev/null || echo 'unknown')"; else echo "  [need] uv:           not installed"; fi
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

  install_rust
  install_uv
  harden_vast_docker_image   # Wave 12 pre-handle — before any HF package operation
  clone_repo
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
