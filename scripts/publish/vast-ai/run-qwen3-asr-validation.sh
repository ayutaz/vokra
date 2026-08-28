#!/usr/bin/env bash
# Convert and validate the two exact Qwen3-ASR releases. VAST-only, no upload.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/qwen3_asr"
REFERENCE_DUMPER="$PARITY_PROJECT/dump_reference.py"
REFERENCE_AUDIO="$VOKRA_ROOT/tests/parity/utmos/ref-clip.wav"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

MIN_VAST_MEM_KIB=60000000
MIN_FREE_DISK_KIB=50000000
REFERENCE_AUDIO_SHA256="241c0d93cc7ed8792c85c525d1e02b8c33850b791902a5e75b79c2d500e71a1a"

log() { printf '[qwen3-asr-vast] %s\n' "$*" >&2; }
step() { printf '\n[qwen3-asr-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-qwen3-asr-validation.sh --variant <0.6b|1.7b|all> [--work-dir <empty-dir>]
       run-qwen3-asr-validation.sh --self-test

VAST-only, non-publishing gate for Qwen3-ASR. For each requested exact release
it downloads the immutable Hugging Face snapshot, streams the BF16 checkpoint
to a self-contained GGUF, generates an independent FP32 CPU reference through
official qwen-asr==0.0.6, and compares Vokra CPU projected audio, prompt ids,
greedy ids, language, and text. It then runs workspace and Apple cross-build
verification once.

There is deliberately no --push flag and no upload path. Pull only the small
reference/evidence directory, never the source snapshots or GGUFs, then destroy
the VAST instance rather than stopping it.
EOF
}

variant_repo() {
  case "$1" in
    0.6b) printf '%s\n' 'Qwen/Qwen3-ASR-0.6B' ;;
    1.7b) printf '%s\n' 'Qwen/Qwen3-ASR-1.7B' ;;
    *) die "unknown variant $1" ;;
  esac
}

variant_revision() {
  case "$1" in
    0.6b) printf '%s\n' '5eb144179a02acc5e5ba31e748d22b0cf3e303b0' ;;
    1.7b) printf '%s\n' '7278e1e70fe206f11671096ffdd38061171dd6e5' ;;
    *) die "unknown variant $1" ;;
  esac
}

variant_model_kind() {
  case "$1" in
    0.6b) printf '%s\n' 'qwen3-asr-0.6b' ;;
    1.7b) printf '%s\n' 'qwen3-asr-1.7b' ;;
    *) die "unknown variant $1" ;;
  esac
}

variant_test() {
  case "$1" in
    0.6b) printf '%s\n' 'qwen3_asr_0_6b_cpu_matches_official_reference' ;;
    1.7b) printf '%s\n' 'qwen3_asr_1_7b_cpu_matches_official_reference' ;;
    *) die "unknown variant $1" ;;
  esac
}

variant_gguf_env() {
  case "$1" in
    0.6b) printf '%s\n' 'VOKRA_QWEN3_ASR_0_6B_GGUF' ;;
    1.7b) printf '%s\n' 'VOKRA_QWEN3_ASR_1_7B_GGUF' ;;
    *) die "unknown variant $1" ;;
  esac
}

variant_reference_env() {
  case "$1" in
    0.6b) printf '%s\n' 'VOKRA_QWEN3_ASR_0_6B_REFERENCE_DIR' ;;
    1.7b) printf '%s\n' 'VOKRA_QWEN3_ASR_1_7B_REFERENCE_DIR' ;;
    *) die "unknown variant $1" ;;
  esac
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
    || die "Qwen3-ASR checkpoint work is VAST/Linux-only; refusing $(uname -s)"
  [[ -r /proc/meminfo ]] || die "/proc/meminfo is unavailable"
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$mem_kib" =~ ^[0-9]+$ ]] || die "could not read MemTotal"
  if (( mem_kib < MIN_VAST_MEM_KIB )); then
    die "MemTotal=${mem_kib} KiB is below the 64-GB-class guard"
  fi
  mkdir -p "$VOKRA_SCRATCH"
  free_kib="$(df -Pk "$VOKRA_SCRATCH" | awk 'NR == 2 {print $4}')"
  [[ "$free_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
  if (( free_kib < MIN_FREE_DISK_KIB )); then
    die "free disk=${free_kib} KiB is below the 50-GB run guard"
  fi
}

require_tooling() {
  local tool
  for tool in uv cargo rustc rustup git awk find tee wc df; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" ]] || die "$VOKRA_ROOT is not a git checkout"
  [[ -f "$PARITY_PROJECT/uv.lock" ]] || die "Qwen3-ASR parity uv.lock is missing"
  [[ -f "$REFERENCE_DUMPER" ]] || die "official reference dumper is missing"
  [[ -f "$REFERENCE_AUDIO" ]] || die "reference audio is missing"
  local audio_hash
  audio_hash="$(sha256_file "$REFERENCE_AUDIO")"
  [[ "$audio_hash" == "$REFERENCE_AUDIO_SHA256" ]] \
    || die "reference audio SHA-256 drift: $audio_hash"
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
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    else
      echo "gpu=unavailable (reference is intentionally CPU FP32)"
    fi
    rustc --version --verbose
    cargo --version
    uv --version
  } > "$output"
}

download_snapshot() {
  local repo="$1" revision="$2" output="$3"
  mkdir -p "$output"
  (
    export HF_HUB_ENABLE_HF_TRANSFER=1
    uv run --no-project --python 3.12 \
      --with 'huggingface_hub<0.30' --with hf-transfer python -c \
      'import os,sys
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=sys.argv[1],
    revision=sys.argv[2],
    local_dir=sys.argv[3],
    allow_patterns=["LICENSE", "README.md", "config.json", "*.safetensors", "model.safetensors.index.json", "vocab.json", "merges.txt", "tokenizer_config.json", "chat_template.json", "generation_config.json", "preprocessor_config.json"],
    token=os.environ.get("HF_TOKEN") or os.environ.get("HF"),
)' "$repo" "$revision" "$output"
  )
}

checkpoint_input() {
  local snapshot="$1"
  if [[ -f "$snapshot/model.safetensors.index.json" ]]; then
    printf '%s\n' "$snapshot/model.safetensors.index.json"
  elif [[ -f "$snapshot/model.safetensors" ]]; then
    printf '%s\n' "$snapshot/model.safetensors"
  else
    die "snapshot has neither model.safetensors.index.json nor model.safetensors: $snapshot"
  fi
}

run_variant() {
  local variant="$1" work_dir="$2" evidence_dir="$3"
  local repo revision model_kind snapshot input gguf reference_dir
  local test_name gguf_env reference_env reference_threads parity_log
  repo="$(variant_repo "$variant")"
  revision="$(variant_revision "$variant")"
  model_kind="$(variant_model_kind "$variant")"
  snapshot="$work_dir/source-$variant"
  gguf="$work_dir/$model_kind.gguf"
  reference_dir="$evidence_dir/reference-$variant"
  test_name="$(variant_test "$variant")"
  gguf_env="$(variant_gguf_env "$variant")"
  reference_env="$(variant_reference_env "$variant")"
  reference_threads="${VOKRA_REFERENCE_TORCH_THREADS:-8}"
  parity_log="$evidence_dir/parity-$variant.log"

  step "Download $repo@$revision"
  download_snapshot "$repo" "$revision" "$snapshot"
  input="$(checkpoint_input "$snapshot")"

  step "Convert $model_kind without quantization or upload"
  "$VOKRA_ROOT/target/release/vokra-cli" convert \
    --model "$model_kind" \
    --input "$input" \
    --output "$gguf" \
    --license apache-2.0 \
    2>&1 | tee "$evidence_dir/convert-$variant.log"
  [[ -s "$gguf" ]] || die "converter emitted no GGUF: $gguf"

  step "Generate independent official FP32 CPU reference for $variant"
  VOKRA_REFERENCE_TORCH_THREADS="$reference_threads" \
    uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python \
      "$REFERENCE_DUMPER" \
      --variant "$variant" \
      --model-dir "$snapshot" \
      --audio "$REFERENCE_AUDIO" \
      --output "$reference_dir" \
      --language English \
      --max-new-tokens 8 \
      2>&1 | tee "$evidence_dir/reference-$variant.log"

  step "Compare Vokra CPU with official reference for $variant"
  env "$gguf_env=$gguf" "$reference_env=$reference_dir" RUST_TEST_THREADS=1 \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --test qwen3_asr_real "$test_name" -- --exact --nocapture \
      2>&1 | tee "$parity_log"
  grep -F "QWEN3_ASR_PARITY $model_kind CPU_vs_official token_ids=exact text=exact PASS" \
    "$parity_log" >/dev/null \
    || die "expected exact-token parity marker is absent for $variant"

  {
    echo "variant=$variant"
    echo "upstream_repo=$repo"
    echo "upstream_revision=$revision"
    echo "gguf_sha256=$(sha256_file "$gguf")"
    echo "reference_manifest_sha256=$(sha256_file "$reference_dir/manifest.txt")"
    echo "numeric_bound=0.01"
    echo "greedy_ids=exact"
    echo "language=exact"
    echo "text=exact"
    echo "verdict=PASS"
  } > "$evidence_dir/summary-$variant.txt"
}

run_self_test() {
  local failed=0
  [[ "$(variant_repo 0.6b)" == "Qwen/Qwen3-ASR-0.6B" ]] || failed=1
  [[ "$(variant_revision 0.6b)" =~ ^[0-9a-f]{40}$ ]] || failed=1
  [[ "$(variant_model_kind 1.7b)" == "qwen3-asr-1.7b" ]] || failed=1
  [[ "$(variant_test 1.7b)" == "qwen3_asr_1_7b_cpu_matches_official_reference" ]] || failed=1
  if variant_repo bad >/dev/null 2>&1; then
    failed=1
  fi
  if (( failed != 0 )); then
    log "self-test FAIL"
    return 1
  fi
  log "self-test PASS"
}

main() {
  local selection='' work_dir='' self_test=0
  while (( $# > 0 )); do
    case "$1" in
      --variant)
        [[ $# -ge 2 ]] || { usage; return 2; }
        selection="$2"
        shift 2
        ;;
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
    [[ -z "$selection" && -z "$work_dir" ]] || die "--self-test accepts no other arguments"
    run_self_test
    return
  fi
  case "$selection" in
    0.6b|1.7b|all) ;;
    *) usage; die "--variant must be 0.6b, 1.7b, or all" ;;
  esac

  require_vast_host
  require_tooling
  if [[ -z "$work_dir" ]]; then
    work_dir="$VOKRA_SCRATCH/qwen3-asr-validation-$(git -C "$VOKRA_ROOT" rev-parse --short=12 HEAD)"
  fi
  if [[ -e "$work_dir" ]]; then
    [[ -d "$work_dir" && -z "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
      || die "--work-dir must not exist or must be empty: $work_dir"
  else
    mkdir -p "$work_dir"
  fi
  local evidence_dir="$work_dir/evidence"
  mkdir -p "$evidence_dir"
  record_environment "$evidence_dir/environment.txt"

  step "Install the locked official reference environment"
  uv sync --project "$PARITY_PROJECT" --frozen --python 3.12 \
    2>&1 | tee "$evidence_dir/uv-sync.log"

  step "Build the current Vokra CLI on VAST"
  cargo build --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release -p vokra-cli \
    2>&1 | tee "$evidence_dir/build-cli.log"

  if [[ "$selection" == "0.6b" || "$selection" == "all" ]]; then
    run_variant 0.6b "$work_dir" "$evidence_dir"
  fi
  if [[ "$selection" == "1.7b" || "$selection" == "all" ]]; then
    run_variant 1.7b "$work_dir" "$evidence_dir"
  fi

  step "Run repository gates and full workspace verification on VAST"
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
    echo "selection=$selection"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "workspace_test=PASS"
    echo "workspace_clippy=PASS"
    echo "apple_metal_cross_compile=PASS"
    echo "apple_real_weight_runtime=PENDING_SEPARATE_APPLE_SILICON_RUN"
    echo "upload=NOT_PERFORMED"
  } > "$evidence_dir/summary.txt"
  log "PASS: pull only $evidence_dir (including reference-*), never source-* or *.gguf"
  log "After evidence is pulled, destroy the VAST instance; do not merely stop it"
}

main "$@"
