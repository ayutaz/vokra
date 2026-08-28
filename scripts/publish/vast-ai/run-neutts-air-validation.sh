#!/usr/bin/env bash
# Validate exact public NeuTTS Air against the official model. VAST-only; no upload.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/neutts_air"
REFERENCE_DUMPER="$PARITY_PROJECT/dump_reference.py"

PUBLIC_REPO="vokra/neutts-air"
PUBLIC_REVISION="df2b47ec81862f0e3a19eb2638a6a2bcd2f13b8c"
PUBLIC_FILE="neutts-air.gguf"
PUBLIC_BYTES="1495883328"
PUBLIC_SHA256="f6caf559e919b16d77ac28177e59ee5427a5de92bdeedd719ecab00b4afbb754"

COMPANION_REPO="vokra/distill-neucodec"
COMPANION_REVISION="1471e4d9b82bfb98ae201f02e746fca346c3eb56"
COMPANION_FILE="model.gguf"
COMPANION_BYTES="1025417504"
COMPANION_SHA256="15e60e7e5f7242255b18e1386b26c2a8f872c77a56ca241ee82c8aa5d8b6327f"

UPSTREAM_REPO="neuphonic/neutts-air"
UPSTREAM_REVISION="3b58b776406b62fdc137e31ea53d728f5c22a4ed"
SOURCE_REPO="https://github.com/neuphonic/neutts.git"
SOURCE_REVISION="3e9415df12633f8a74ac6f92418c7cd5c8c4bf0e"
SOURCE_RELATIVE="neuttsair/neutts.py"
SOURCE_BYTES="9035"
SOURCE_SHA256="e68b87dae6718903337a08eff56afbd58ba261d829624ea5a00a343c8cefb7c1"

MIN_VAST_MEM_KIB=48000000
MIN_FREE_DISK_KIB=25000000
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

log() { printf '[neutts-air-vast] %s\n' "$*" >&2; }
step() { printf '\n[neutts-air-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-neutts-air-validation.sh [--work-dir <empty-dir>]
       run-neutts-air-validation.sh --self-test

VAST-only, non-publishing gate for the exact public NeuTTS Air release. It
downloads the pinned public LM GGUF and Distill NeuCodec companion, downloads
the exact gated upstream snapshot, executes the fixed Neuphonic prompt method
and official Transformers Qwen2 model in CPU FP32, then compares Vokra logits
at atol=0.01 and greedy ids exactly. Workspace and Apple-target feature builds
run only after the real CPU gate.

There is no --push flag and no upload path. Pull only the small evidence and
reference directories, never model payloads, then destroy the VAST instance.
EOF
}

sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  else
    die "neither sha256sum nor shasum is available"
  fi
}

require_vast_host() {
  local mem_kib free_kib
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] \
    || die "VOKRA_PUBLISH_ON_VAST=1 is absent; run provision.sh first"
  [[ "$(uname -s)" == "Linux" ]] \
    || die "NeuTTS Air model work is VAST/Linux-only; refusing $(uname -s)"
  [[ -r /proc/meminfo ]] || die "/proc/meminfo is unavailable"
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$mem_kib" =~ ^[0-9]+$ ]] || die "could not read MemTotal"
  if (( mem_kib < MIN_VAST_MEM_KIB )); then
    die "MemTotal=${mem_kib} KiB is below the 48-GB-class guard"
  fi
  mkdir -p "$VOKRA_SCRATCH"
  free_kib="$(df -Pk "$VOKRA_SCRATCH" | awk 'NR == 2 {print $4}')"
  [[ "$free_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
  if (( free_kib < MIN_FREE_DISK_KIB )); then
    die "free disk=${free_kib} KiB is below the 25-GB run guard"
  fi
  [[ -n "${HF_TOKEN:-${HF:-}}" ]] \
    || die "HF_TOKEN/HF is required for the gated upstream snapshot"
}

require_tooling() {
  local tool
  for tool in uv cargo rustc rustup git awk find tee wc df grep; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" ]] || die "$VOKRA_ROOT is not a git checkout"
  [[ -f "$PARITY_PROJECT/uv.lock" ]] || die "NeuTTS Air parity uv.lock is missing"
  [[ -f "$REFERENCE_DUMPER" ]] || die "official reference dumper is missing"
  if [[ -n "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]]; then
    die "VAST checkout must be clean so evidence names one exact commit"
  fi
}

require_identity() {
  local label="$1" path="$2" expected_bytes="$3" expected_sha="$4"
  [[ -f "$path" ]] || die "$label is missing: $path"
  local actual_bytes actual_sha
  actual_bytes="$(wc -c < "$path" | tr -d '[:space:]')"
  actual_sha="$(sha256_file "$path")"
  [[ "$actual_bytes" == "$expected_bytes" ]] \
    || die "$label bytes=$actual_bytes, expected $expected_bytes"
  [[ "$actual_sha" == "$expected_sha" ]] \
    || die "$label SHA-256=$actual_sha, expected $expected_sha"
}

record_environment() {
  local output="$1" cpu_model cpu_flags
  cpu_model="$(awk -F ':' '$1 ~ /model name/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
  cpu_flags="$(awk -F ':' '$1 ~ /flags/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
  {
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "git_branch=$(git -C "$VOKRA_ROOT" branch --show-current)"
    echo "uname=$(uname -a)"
    echo "cpu_model=$cpu_model"
    echo "cpu_flags=$cpu_flags"
    echo "nproc=$(nproc)"
    awk '$1 == "MemTotal:" {print "mem_total_kib=" $2; exit}' /proc/meminfo
    rustc --version --verbose
    cargo --version
    uv --version
  } > "$output"
}

download_hf_file() {
  local repo="$1" revision="$2" filename="$3" output_dir="$4"
  mkdir -p "$output_dir"
  uv run --no-project --python 3.12 \
    --with 'huggingface_hub<0.30' python -c \
    'import os,sys
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id=sys.argv[1], revision=sys.argv[2], filename=sys.argv[3],
    local_dir=sys.argv[4], token=os.environ.get("HF_TOKEN") or os.environ.get("HF"),
)' "$repo" "$revision" "$filename" "$output_dir"
}

download_upstream_snapshot() {
  local output_dir="$1"
  mkdir -p "$output_dir"
  uv run --no-project --python 3.12 \
    --with 'huggingface_hub<0.30' python -c \
    'import os,sys
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3],
    allow_patterns=["config.json", "generation_config.json", "model.safetensors", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"],
    token=os.environ.get("HF_TOKEN") or os.environ.get("HF"),
)' "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$output_dir"
  printf '%s\n' "$UPSTREAM_REVISION" > "$output_dir/.vokra-source-revision"
}

checkout_source() {
  local output_dir="$1"
  git init -q "$output_dir"
  git -C "$output_dir" remote add origin "$SOURCE_REPO"
  git -C "$output_dir" fetch --depth 1 origin "$SOURCE_REVISION"
  git -C "$output_dir" checkout --detach -q FETCH_HEAD
  [[ "$(git -C "$output_dir" rev-parse HEAD)" == "$SOURCE_REVISION" ]] \
    || die "Neuphonic source checkout revision drift"
  require_identity "Neuphonic release source" "$output_dir/$SOURCE_RELATIVE" \
    "$SOURCE_BYTES" "$SOURCE_SHA256"
}

run_self_test() {
  local failed=0
  [[ "$PUBLIC_REVISION" =~ ^[0-9a-f]{40}$ ]] || failed=1
  [[ "$UPSTREAM_REVISION" =~ ^[0-9a-f]{40}$ ]] || failed=1
  [[ "$SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]] || failed=1
  [[ "$PUBLIC_SHA256" =~ ^[0-9a-f]{64}$ ]] || failed=1
  [[ "$COMPANION_SHA256" =~ ^[0-9a-f]{64}$ ]] || failed=1
  [[ "$SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]] || failed=1
  if (( failed != 0 )); then
    die "self-test FAIL"
  fi
  log "self-test PASS"
}

main() {
  local work_dir='' self_test=0
  while (( $# > 0 )); do
    case "$1" in
      --work-dir)
        [[ $# -ge 2 ]] || { usage; return 2; }
        work_dir="$2"
        shift 2
        ;;
      --self-test)
        self_test=1
        shift
        ;;
      -h|--help)
        usage
        return 0
        ;;
      *)
        usage
        die "unknown argument $1"
        ;;
    esac
  done
  if (( self_test == 1 )); then
    [[ -z "$work_dir" ]] || die "--self-test accepts no other arguments"
    run_self_test
    return
  fi

  require_vast_host
  require_tooling
  if [[ -z "$work_dir" ]]; then
    work_dir="$VOKRA_SCRATCH/neutts-air-validation-$(git -C "$VOKRA_ROOT" rev-parse --short=12 HEAD)"
  fi
  if [[ -e "$work_dir" ]]; then
    [[ -d "$work_dir" && -z "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
      || die "--work-dir must not exist or must be empty: $work_dir"
  else
    mkdir -p "$work_dir"
  fi

  local evidence_dir="$work_dir/evidence"
  local public_dir="$work_dir/public-neutts"
  local companion_dir="$work_dir/public-neucodec"
  local upstream_dir="$work_dir/upstream"
  local source_dir="$work_dir/source"
  local reference_dir="$evidence_dir/reference"
  local gguf="$public_dir/$PUBLIC_FILE"
  local companion="$companion_dir/$COMPANION_FILE"
  mkdir -p "$evidence_dir"
  record_environment "$evidence_dir/environment.txt"

  step "Download and authenticate exact public GGUFs"
  download_hf_file "$PUBLIC_REPO" "$PUBLIC_REVISION" "$PUBLIC_FILE" "$public_dir"
  download_hf_file "$COMPANION_REPO" "$COMPANION_REVISION" "$COMPANION_FILE" "$companion_dir"
  require_identity "NeuTTS Air public GGUF" "$gguf" "$PUBLIC_BYTES" "$PUBLIC_SHA256"
  require_identity "Distill NeuCodec public GGUF" "$companion" \
    "$COMPANION_BYTES" "$COMPANION_SHA256"

  step "Download exact gated upstream snapshot and official source"
  download_upstream_snapshot "$upstream_dir"
  checkout_source "$source_dir"

  step "Install locked official reference environment"
  uv sync --project "$PARITY_PROJECT" --frozen --python 3.12 \
    2>&1 | tee "$evidence_dir/uv-sync.log"

  step "Generate independent official FP32 reference"
  VOKRA_REFERENCE_TORCH_THREADS="${VOKRA_REFERENCE_TORCH_THREADS:-8}" \
    uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python \
      "$REFERENCE_DUMPER" \
      --model-dir "$upstream_dir" \
      --source-file "$source_dir/$SOURCE_RELATIVE" \
      --output "$reference_dir" \
      --max-new-tokens 4 \
      2>&1 | tee "$evidence_dir/reference.log"

  step "Build Vokra and compare CPU logits/greedy generation/composition"
  cargo build --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release -p vokra-cli \
    2>&1 | tee "$evidence_dir/build-cli.log"
  env \
    VOKRA_NEUTTS_AIR_GGUF="$gguf" \
    VOKRA_NEUTTS_AIR_REFERENCE_DIR="$reference_dir" \
    VOKRA_NEUTTS_AIR_COMPANION_GGUF="$companion" \
    VOKRA_NEUTTS_AIR_BACKEND=cpu \
    RUST_TEST_THREADS=1 \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --test neutts_air_real \
      neutts_air_public_cpu_or_metal_matches_official_reference -- --exact --nocapture \
      2>&1 | tee "$evidence_dir/parity-cpu.log"
  grep -F 'NEUTTS_AIR_PARITY Cpu_vs_official logits_atol=0.01 greedy_ids=exact PASS' \
    "$evidence_dir/parity-cpu.log" >/dev/null \
    || die "CPU numerical PASS marker is absent"
  grep -F 'NEUTTS_AIR_COMPOSITION Cpu' "$evidence_dir/parity-cpu.log" >/dev/null \
    || die "CPU composition PASS marker is absent"

  local prompt_ids
  prompt_ids="$(awk -F= '$1 == "prompt_ids_csv" {print substr($0, index($0, "=") + 1); exit}' "$reference_dir/manifest.txt")"
  [[ -n "$prompt_ids" ]] || die "reference manifest has no prompt_ids_csv"
  "$VOKRA_ROOT/target/release/vokra-cli" run \
    --model "$gguf" \
    --neutts-companion "$companion" \
    --token-ids "$prompt_ids" \
    --neutts-greedy \
    --neutts-max-new-tokens 4 \
    --output "$evidence_dir/neutts-air-cpu.wav" \
    2>&1 | tee "$evidence_dir/cli-cpu.log"
  [[ -s "$evidence_dir/neutts-air-cpu.wav" ]] || die "CLI emitted no WAV"

  step "Run repository gates and full VAST verification"
  cargo fmt --manifest-path "$VOKRA_ROOT/Cargo.toml" --all -- --check
  bash "$VOKRA_ROOT/scripts/check-zero-deps.sh"
  bash "$VOKRA_ROOT/scripts/check-forbidden-symbols.sh"
  bash "$VOKRA_ROOT/scripts/check-bound-arch-coverage.sh"
  bash "$VOKRA_ROOT/scripts/check-arch-handshake.sh"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --workspace \
    2>&1 | tee "$evidence_dir/workspace-test.log"
  cargo clippy --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --workspace \
    --all-targets -- -D warnings 2>&1 | tee "$evidence_dir/workspace-clippy.log"

  step "Cross-check Apple Metal feature compilation"
  rustup target add aarch64-apple-darwin
  cargo check --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked \
    -p vokra-models --features metal --target aarch64-apple-darwin \
    2>&1 | tee "$evidence_dir/apple-metal-cross-check.log"

  {
    echo "verdict=PASS"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "public_repo=$PUBLIC_REPO"
    echo "public_revision=$PUBLIC_REVISION"
    echo "public_sha256=$PUBLIC_SHA256"
    echo "companion_repo=$COMPANION_REPO"
    echo "companion_revision=$COMPANION_REVISION"
    echo "companion_sha256=$COMPANION_SHA256"
    echo "upstream_repo=$UPSTREAM_REPO"
    echo "upstream_revision=$UPSTREAM_REVISION"
    echo "source_revision=$SOURCE_REVISION"
    echo "reference_manifest_sha256=$(sha256_file "$reference_dir/manifest.txt")"
    echo "next_logits_atol=$FP32_ATOL"
    echo "greedy_ids=exact"
    echo "composition=PASS"
    echo "upload=NOT_PERFORMED"
  } > "$evidence_dir/summary.txt"
  log "PASS: pull only $evidence_dir, then destroy the VAST instance"
}

FP32_ATOL="0.01"
main "$@"
