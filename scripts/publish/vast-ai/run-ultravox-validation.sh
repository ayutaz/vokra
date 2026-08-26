#!/usr/bin/env bash
# Validate exact public Ultravox against the official model. VAST-only; no upload.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/ultravox"
REFERENCE_DUMPER="$PARITY_PROJECT/dump_reference.py"

PUBLIC_REPO="vokra/ultravox-v0-5-llama-3-2-1b"
PUBLIC_REVISION="ddbbeec5bfcb09c71a1f88971b794e3e5da811f9"
PUBLIC_FILE="ultravox-v0-5-llama-3-2-1b.gguf"
PUBLIC_BYTES="1366275264"
PUBLIC_SHA256="376c79a7219bb38fc6a857b0bd9ccf57daff878e7bb4723c4801000c0d7b8c9c"

UPSTREAM_REPO="fixie-ai/ultravox-v0_5-llama-3_2-1b"
UPSTREAM_REVISION="b95bec8ab291eeb04b5cd600dd473377f6b79026"
COMPANION_REPO="meta-llama/Llama-3.2-1B-Instruct"
COMPANION_REVISION="9213176726f574b556790deb65791e0c5aa438b6"

MODEL_SOURCE_BYTES="41578"
MODEL_SOURCE_SHA256="df618218561375da01bb53bd2764ea123e0cbf782f3326753f669f63ff6c6d3f"
PROCESSOR_SOURCE_BYTES="17087"
PROCESSOR_SOURCE_SHA256="2ae6682f3deecb22539fae6a6631688fc1675282f1a5b31145d9f95d2347ff7b"
CONFIG_SOURCE_BYTES="7057"
CONFIG_SOURCE_SHA256="99cf5ad911189f2351c2232234025db56b23763283583c0a848ebf2a1ecc40fc"

MIN_VAST_MEM_KIB=64000000
MIN_FREE_DISK_KIB=30000000
FP32_ATOL="0.01"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

log() { printf '[ultravox-vast] %s\n' "$*" >&2; }
step() { printf '\n[ultravox-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-ultravox-validation.sh [--work-dir <empty-dir>]
       run-ultravox-validation.sh --self-test

VAST-only, non-publishing gate for the exact public Ultravox v0.5 audio GGUF.
It downloads and authenticates the fixed public artifact, fixed Fixie snapshot,
and exact gated Meta Llama companion. It converts the companion locally,
executes the official custom model and processor in CPU FP32, compares Vokra
frontend/audio embeddings/logits at atol=0.01 and greedy IDs exactly, then runs
the repository verification gates.

There is no --push flag and no upload path. Pull only the small evidence and
reference directories. Do not pull model payloads to the maintainer Mac; send
them directly to a disposable Apple worker if Metal parity is required, then
destroy the VAST instance.
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

file_bytes() {
  wc -c < "$1" | tr -d '[:space:]'
}

require_identity() {
  local label="$1" path="$2" expected_bytes="$3" expected_sha="$4"
  [[ -f "$path" ]] || die "$label is missing: $path"
  local actual_bytes actual_sha
  actual_bytes="$(file_bytes "$path")"
  actual_sha="$(sha256_file "$path")"
  [[ "$actual_bytes" == "$expected_bytes" ]] \
    || die "$label bytes=$actual_bytes, expected $expected_bytes"
  [[ "$actual_sha" == "$expected_sha" ]] \
    || die "$label SHA-256=$actual_sha, expected $expected_sha"
}

require_vast_host() {
  local mem_kib free_kib
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] \
    || die "VOKRA_PUBLISH_ON_VAST=1 is absent; run provision.sh first"
  [[ "$(uname -s)" == "Linux" ]] \
    || die "Ultravox model work is VAST/Linux-only; refusing $(uname -s)"
  [[ -r /proc/meminfo ]] || die "/proc/meminfo is unavailable"
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$mem_kib" =~ ^[0-9]+$ ]] || die "could not read MemTotal"
  if (( mem_kib < MIN_VAST_MEM_KIB )); then
    die "MemTotal=${mem_kib} KiB is below the 64-GB guard"
  fi
  mkdir -p "$VOKRA_SCRATCH"
  free_kib="$(df -Pk "$VOKRA_SCRATCH" | awk 'NR == 2 {print $4}')"
  [[ "$free_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
  if (( free_kib < MIN_FREE_DISK_KIB )); then
    die "free disk=${free_kib} KiB is below the 30-GB run guard"
  fi
  [[ -n "${HF_TOKEN:-${HF:-}}" ]] \
    || die "HF_TOKEN/HF is required for the gated Meta snapshot"
}

require_tooling() {
  local tool
  for tool in uv cargo rustc rustup git awk find tee wc df grep tr cargo-deny; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" ]] || die "$VOKRA_ROOT is not a git checkout"
  [[ -f "$PARITY_PROJECT/uv.lock" ]] || die "Ultravox parity uv.lock is missing"
  [[ -f "$REFERENCE_DUMPER" ]] || die "official reference dumper is missing"
  if [[ -n "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]]; then
    die "VAST checkout must be clean so evidence names one exact commit"
  fi
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
    cargo deny --version
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

download_ultravox_snapshot() {
  local output_dir="$1"
  mkdir -p "$output_dir"
  uv run --no-project --python 3.12 \
    --with 'huggingface_hub<0.30' python -c \
    'import os,sys
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3],
    allow_patterns=[
        "config.json", "generation_config.json", "model.safetensors",
        "preprocessor_config.json", "processor_config.json",
        "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
        "ultravox_config.py", "ultravox_model.py", "ultravox_processing.py",
    ],
    token=os.environ.get("HF_TOKEN") or os.environ.get("HF"),
)' "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$output_dir"
  printf '%s\n' "$UPSTREAM_REVISION" > "$output_dir/.vokra-source-revision"
  require_identity "official Ultravox model source" "$output_dir/ultravox_model.py" \
    "$MODEL_SOURCE_BYTES" "$MODEL_SOURCE_SHA256"
  require_identity "official Ultravox processor source" "$output_dir/ultravox_processing.py" \
    "$PROCESSOR_SOURCE_BYTES" "$PROCESSOR_SOURCE_SHA256"
  require_identity "official Ultravox config source" "$output_dir/ultravox_config.py" \
    "$CONFIG_SOURCE_BYTES" "$CONFIG_SOURCE_SHA256"
}

download_companion_snapshot() {
  local output_dir="$1"
  mkdir -p "$output_dir"
  uv run --no-project --python 3.12 \
    --with 'huggingface_hub<0.30' python -c \
    'import os,sys
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3],
    allow_patterns=["config.json", "model.safetensors"],
    token=os.environ.get("HF_TOKEN") or os.environ.get("HF"),
)' "$COMPANION_REPO" "$COMPANION_REVISION" "$output_dir"
  printf '%s\n' "$COMPANION_REVISION" > "$output_dir/.vokra-source-revision"
  [[ -s "$output_dir/config.json" && -s "$output_dir/model.safetensors" ]] \
    || die "gated companion snapshot is incomplete"
}

run_self_test() {
  local failed=0
  [[ "$PUBLIC_REVISION" =~ ^[0-9a-f]{40}$ ]] || failed=1
  [[ "$UPSTREAM_REVISION" =~ ^[0-9a-f]{40}$ ]] || failed=1
  [[ "$COMPANION_REVISION" =~ ^[0-9a-f]{40}$ ]] || failed=1
  [[ "$PUBLIC_SHA256" =~ ^[0-9a-f]{64}$ ]] || failed=1
  [[ "$MODEL_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]] || failed=1
  [[ "$PROCESSOR_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]] || failed=1
  [[ "$CONFIG_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]] || failed=1
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
    work_dir="$VOKRA_SCRATCH/ultravox-validation-$(git -C "$VOKRA_ROOT" rev-parse --short=12 HEAD)"
  fi
  if [[ -e "$work_dir" ]]; then
    [[ -d "$work_dir" && -z "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
      || die "--work-dir must not exist or must be empty: $work_dir"
  else
    mkdir -p "$work_dir"
  fi

  local evidence_dir="$work_dir/evidence"
  local public_dir="$work_dir/public-vokra"
  local upstream_dir="$work_dir/upstream-ultravox"
  local companion_source_dir="$work_dir/upstream-llama"
  local converted_dir="$work_dir/converted"
  local reference_dir="$evidence_dir/reference"
  local gguf="$public_dir/$PUBLIC_FILE"
  local companion_gguf="$converted_dir/ultravox-llama-companion.gguf"
  mkdir -p "$evidence_dir" "$converted_dir"
  record_environment "$evidence_dir/environment.txt"

  step "Download and authenticate exact public GGUF"
  download_hf_file "$PUBLIC_REPO" "$PUBLIC_REVISION" "$PUBLIC_FILE" "$public_dir"
  require_identity "Ultravox public GGUF" "$gguf" "$PUBLIC_BYTES" "$PUBLIC_SHA256"

  step "Download exact official Ultravox and gated Llama snapshots"
  download_ultravox_snapshot "$upstream_dir"
  download_companion_snapshot "$companion_source_dir"

  step "Build Vokra and stream-convert the separately licensed companion"
  cargo build --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release -p vokra-cli \
    2>&1 | tee "$evidence_dir/build-cli.log"
  "$VOKRA_ROOT/target/release/vokra-cli" convert \
    --model ultravox-llama-companion \
    --input "$companion_source_dir/model.safetensors" \
    --config "$companion_source_dir/config.json" \
    --revision "$COMPANION_REVISION" \
    --output "$companion_gguf" \
    2>&1 | tee "$evidence_dir/convert-companion.log"
  [[ -s "$companion_gguf" ]] || die "companion conversion emitted no GGUF"

  step "Install locked official reference environment"
  uv sync --project "$PARITY_PROJECT" --frozen --python 3.12 \
    2>&1 | tee "$evidence_dir/uv-sync.log"

  step "Generate independent official FP32 reference"
  VOKRA_REFERENCE_TORCH_THREADS="${VOKRA_REFERENCE_TORCH_THREADS:-8}" \
    uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python \
      "$REFERENCE_DUMPER" \
      --ultravox-dir "$upstream_dir" \
      --companion-dir "$companion_source_dir" \
      --output "$reference_dir" \
      --max-new-tokens 4 \
      2>&1 | tee "$evidence_dir/reference.log"

  step "Compare real Vokra CPU frontend, embeddings, logits and greedy IDs"
  env \
    VOKRA_ULTRAVOX_GGUF="$gguf" \
    VOKRA_ULTRAVOX_COMPANION_GGUF="$companion_gguf" \
    VOKRA_ULTRAVOX_REFERENCE_DIR="$reference_dir" \
    VOKRA_ULTRAVOX_BACKEND=cpu \
    RUST_TEST_THREADS=1 \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --test ultravox_real \
      ultravox_public_cpu_or_metal_matches_official_reference -- --exact --nocapture \
      2>&1 | tee "$evidence_dir/parity-cpu.log"
  grep -F 'ULTRAVOX_PARITY Cpu_vs_official frontend_atol=0.01 audio_embeddings_atol=0.01 logits_atol=0.01 greedy_ids=exact PASS' \
    "$evidence_dir/parity-cpu.log" >/dev/null \
    || die "CPU numerical PASS marker is absent"

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
  cargo deny --manifest-path "$VOKRA_ROOT/Cargo.toml" check licenses advisories bans \
    2>&1 | tee "$evidence_dir/cargo-deny.log"

  step "Cross-check Apple Metal feature compilation"
  rustup target add aarch64-apple-darwin
  cargo check --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked \
    -p vokra-models --features metal --target aarch64-apple-darwin \
    2>&1 | tee "$evidence_dir/apple-metal-cross-check.log"

  {
    echo "public_gguf_bytes=$(file_bytes "$gguf")"
    echo "public_gguf_sha256=$(sha256_file "$gguf")"
    echo "companion_source_bytes=$(file_bytes "$companion_source_dir/model.safetensors")"
    echo "companion_source_sha256=$(sha256_file "$companion_source_dir/model.safetensors")"
    echo "companion_gguf_bytes=$(file_bytes "$companion_gguf")"
    echo "companion_gguf_sha256=$(sha256_file "$companion_gguf")"
    echo "reference_manifest_sha256=$(sha256_file "$reference_dir/manifest.txt")"
  } > "$evidence_dir/input-hashes.txt"
  {
    echo "verdict=PASS"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "public_repo=$PUBLIC_REPO"
    echo "public_revision=$PUBLIC_REVISION"
    echo "public_sha256=$PUBLIC_SHA256"
    echo "upstream_repo=$UPSTREAM_REPO"
    echo "upstream_revision=$UPSTREAM_REVISION"
    echo "companion_repo=$COMPANION_REPO"
    echo "companion_revision=$COMPANION_REVISION"
    echo "companion_gguf_sha256=$(sha256_file "$companion_gguf")"
    echo "reference_manifest_sha256=$(sha256_file "$reference_dir/manifest.txt")"
    echo "frontend_atol=$FP32_ATOL"
    echo "audio_embeddings_atol=$FP32_ATOL"
    echo "next_logits_atol=$FP32_ATOL"
    echo "greedy_ids=exact"
    echo "upload=NOT_PERFORMED"
  } > "$evidence_dir/summary.txt"
  log "PASS: pull only $evidence_dir; send model data directly to the Apple worker if needed, then destroy VAST"
}

main "$@"
