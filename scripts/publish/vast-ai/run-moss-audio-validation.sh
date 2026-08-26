#!/usr/bin/env bash
# Convert and validate the two exact MOSS-Audio Instruct releases. VAST-only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/moss_audio"
REFERENCE_DUMPER="$PARITY_PROJECT/dump_reference.py"
REFERENCE_AUDIO="$VOKRA_ROOT/tests/parity/utmos/ref-clip.wav"
SOURCE_REPO="https://github.com/OpenMOSS/MOSS-Audio.git"
SOURCE_REVISION="5cbb1d823937cd5b5de3d8fa4d3a7253ebd3b883"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# The 8B FP32 official oracle and the shard merger can each retain tens of GB.
# A 128-GB-class CPU host leaves headroom for allocator and Cargo peaks.
MIN_VAST_MEM_KIB=120000000
MIN_FREE_DISK_KIB=150000000
REFERENCE_AUDIO_SHA256="241c0d93cc7ed8792c85c525d1e02b8c33850b791902a5e75b79c2d500e71a1a"

log() { printf '[moss-audio-vast] %s\n' "$*" >&2; }
step() { printf '\n[moss-audio-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-moss-audio-validation.sh --variant <4b|8b|all> [--work-dir <empty-dir>]
       run-moss-audio-validation.sh --self-test

VAST-only, non-publishing gate for the two pinned MOSS-Audio Instruct
releases. It downloads exact immutable source/model revisions, merges the
shards, converts a self-contained GGUF, generates an independent FP32 CPU
reference through the official OpenMOSS model and processor, and compares
Vokra CPU audio projections, prompt ids, greedy ids and decoded text. It then
runs workspace and Apple Metal cross-build verification once.

There is no publishing option or artifact-upload path. Pull only the small
evidence directory, never the snapshots, merged checkpoints or GGUFs. Destroy
the VAST instance after the evidence is recovered; do not merely stop it.
EOF
}

variant_repo() {
  case "$1" in
    4b) printf '%s\n' 'OpenMOSS-Team/MOSS-Audio-4B-Instruct' ;;
    8b) printf '%s\n' 'OpenMOSS-Team/MOSS-Audio-8B-Instruct' ;;
    *) die "unknown variant $1" ;;
  esac
}

variant_revision() {
  case "$1" in
    4b) printf '%s\n' '6907a499dc0e87cc77c8ae0fe23fd0eb5476a02d' ;;
    8b) printf '%s\n' '6521a39181b47a18f2d9f4b3acfb5bca7b76b57f' ;;
    *) die "unknown variant $1" ;;
  esac
}

variant_model_kind() {
  case "$1" in
    4b) printf '%s\n' 'moss-audio-4b-instruct' ;;
    8b) printf '%s\n' 'moss-audio-8b-instruct' ;;
    *) die "unknown variant $1" ;;
  esac
}

variant_preparer() {
  case "$1" in
    4b) printf '%s\n' "$VOKRA_ROOT/tools/parity/moss_audio_4b_instruct_prepare_checkpoint.py" ;;
    8b) printf '%s\n' "$VOKRA_ROOT/tools/parity/moss_audio_8b_instruct_prepare_checkpoint.py" ;;
    *) die "unknown variant $1" ;;
  esac
}

variant_test() {
  case "$1" in
    4b) printf '%s\n' 'moss_audio_4b_cpu_matches_official_reference' ;;
    8b) printf '%s\n' 'moss_audio_8b_cpu_matches_official_reference' ;;
    *) die "unknown variant $1" ;;
  esac
}

variant_gguf_env() {
  case "$1" in
    4b) printf '%s\n' 'VOKRA_MOSS_AUDIO_4B_GGUF' ;;
    8b) printf '%s\n' 'VOKRA_MOSS_AUDIO_8B_GGUF' ;;
    *) die "unknown variant $1" ;;
  esac
}

variant_reference_env() {
  case "$1" in
    4b) printf '%s\n' 'VOKRA_MOSS_AUDIO_4B_REFERENCE_DIR' ;;
    8b) printf '%s\n' 'VOKRA_MOSS_AUDIO_8B_REFERENCE_DIR' ;;
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
    || die "MOSS-Audio checkpoint work is VAST/Linux-only; refusing $(uname -s)"
  [[ -r /proc/meminfo ]] || die "/proc/meminfo is unavailable"
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$mem_kib" =~ ^[0-9]+$ ]] || die "could not read MemTotal"
  if (( mem_kib < MIN_VAST_MEM_KIB )); then
    die "MemTotal=${mem_kib} KiB is below the 128-GB-class guard"
  fi
  mkdir -p "$VOKRA_SCRATCH"
  free_kib="$(df -Pk "$VOKRA_SCRATCH" | awk 'NR == 2 {print $4}')"
  [[ "$free_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
  if (( free_kib < MIN_FREE_DISK_KIB )); then
    die "free disk=${free_kib} KiB is below the 150-GB run guard"
  fi
}

require_tooling() {
  local tool preparer
  for tool in uv cargo rustc rustup git awk find tee grep wc df; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" ]] || die "$VOKRA_ROOT is not a git checkout"
  [[ -f "$PARITY_PROJECT/uv.lock" ]] || die "MOSS-Audio parity uv.lock is missing"
  [[ -f "$REFERENCE_DUMPER" ]] || die "official reference dumper is missing"
  [[ -f "$REFERENCE_AUDIO" ]] || die "reference audio is missing"
  for preparer in "$(variant_preparer 4b)" "$(variant_preparer 8b)"; do
    [[ -f "$preparer" ]] || die "checkpoint preparer is missing: $preparer"
  done
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

checkout_official_source() {
  local output="$1"
  git clone --filter=blob:none --no-checkout "$SOURCE_REPO" "$output"
  git -C "$output" checkout --detach "$SOURCE_REVISION"
  [[ "$(git -C "$output" rev-parse HEAD)" == "$SOURCE_REVISION" ]] \
    || die "official source checkout revision mismatch"
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
    allow_patterns=["LICENSE", "README.md", "config.json", "*.safetensors", "model.safetensors.index.json", "vocab.json", "merges.txt", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "processor_config.json"],
    token=os.environ.get("HF_TOKEN") or os.environ.get("HF"),
)' "$repo" "$revision" "$output"
  )
}

run_variant() {
  local variant="$1" work_dir="$2" evidence_dir="$3" source_dir="$4"
  local repo revision model_kind preparer snapshot merged gguf reference_dir
  local test_name gguf_env reference_env reference_threads parity_log
  repo="$(variant_repo "$variant")"
  revision="$(variant_revision "$variant")"
  model_kind="$(variant_model_kind "$variant")"
  preparer="$(variant_preparer "$variant")"
  snapshot="$work_dir/source-$variant"
  merged="$snapshot/model.merged.safetensors"
  gguf="$work_dir/$model_kind.gguf"
  reference_dir="$evidence_dir/reference-$variant"
  test_name="$(variant_test "$variant")"
  gguf_env="$(variant_gguf_env "$variant")"
  reference_env="$(variant_reference_env "$variant")"
  reference_threads="${VOKRA_REFERENCE_TORCH_THREADS:-8}"
  parity_log="$evidence_dir/parity-$variant.log"

  step "Download $repo@$revision"
  download_snapshot "$repo" "$revision" "$snapshot"

  step "Merge the pinned $variant sharded checkpoint"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python \
    "$preparer" --input-dir "$snapshot" --output "$merged" --strict \
    2>&1 | tee "$evidence_dir/prepare-$variant.log"
  [[ -s "$merged" ]] || die "checkpoint merger emitted no file: $merged"

  step "Convert $model_kind without quantization or publication"
  "$VOKRA_ROOT/target/release/vokra-cli" convert \
    --model "$model_kind" \
    --input "$merged" \
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
      --source-dir "$source_dir" \
      --audio "$REFERENCE_AUDIO" \
      --output "$reference_dir" \
      --max-new-tokens 4 \
      2>&1 | tee "$evidence_dir/reference-$variant.log"

  step "Compare Vokra CPU with official reference for $variant"
  env "$gguf_env=$gguf" "$reference_env=$reference_dir" RUST_TEST_THREADS=1 \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --test moss_audio_real "$test_name" -- --exact --nocapture \
      2>&1 | tee "$parity_log"
  grep -F "MOSS_AUDIO_PARITY $model_kind CPU_vs_official token_ids=exact text=exact PASS" \
    "$parity_log" >/dev/null \
    || die "expected exact-token parity marker is absent for $variant"

  {
    echo "variant=$variant"
    echo "upstream_repo=$repo"
    echo "upstream_revision=$revision"
    echo "source_repo=$SOURCE_REPO"
    echo "source_revision=$SOURCE_REVISION"
    echo "gguf_sha256=$(sha256_file "$gguf")"
    echo "reference_manifest_sha256=$(sha256_file "$reference_dir/manifest.txt")"
    echo "numeric_bound=0.01"
    echo "greedy_ids=exact"
    echo "text=exact"
    echo "verdict=PASS"
  } > "$evidence_dir/summary-$variant.txt"
}

run_self_test() {
  local failed=0
  [[ "$(variant_repo 4b)" == "OpenMOSS-Team/MOSS-Audio-4B-Instruct" ]] || failed=1
  [[ "$(variant_revision 8b)" =~ ^[0-9a-f]{40}$ ]] || failed=1
  [[ "$(variant_model_kind 8b)" == "moss-audio-8b-instruct" ]] || failed=1
  [[ "$(variant_test 4b)" == "moss_audio_4b_cpu_matches_official_reference" ]] || failed=1
  [[ "$(variant_gguf_env 8b)" == "VOKRA_MOSS_AUDIO_8B_GGUF" ]] || failed=1
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
    4b|8b|all) ;;
    *) usage; die "--variant must be 4b, 8b, or all" ;;
  esac

  require_vast_host
  require_tooling
  if [[ -z "$work_dir" ]]; then
    work_dir="$VOKRA_SCRATCH/moss-audio-validation-$(git -C "$VOKRA_ROOT" rev-parse --short=12 HEAD)"
  fi
  if [[ -e "$work_dir" ]]; then
    [[ -d "$work_dir" && -z "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
      || die "--work-dir must not exist or must be empty: $work_dir"
  else
    mkdir -p "$work_dir"
  fi
  local evidence_dir="$work_dir/evidence"
  local source_dir="$work_dir/official-source"
  mkdir -p "$evidence_dir"
  record_environment "$evidence_dir/environment.txt"

  step "Install the locked official reference environment"
  uv sync --project "$PARITY_PROJECT" --frozen --python 3.12 \
    2>&1 | tee "$evidence_dir/uv-sync.log"

  step "Checkout the immutable official OpenMOSS source"
  checkout_official_source "$source_dir" \
    2>&1 | tee "$evidence_dir/source-checkout.log"

  step "Build the current Vokra CLI on VAST"
  cargo build --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release -p vokra-cli \
    2>&1 | tee "$evidence_dir/build-cli.log"

  if [[ "$selection" == "4b" || "$selection" == "all" ]]; then
    run_variant 4b "$work_dir" "$evidence_dir" "$source_dir"
  fi
  if [[ "$selection" == "8b" || "$selection" == "all" ]]; then
    run_variant 8b "$work_dir" "$evidence_dir" "$source_dir"
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
