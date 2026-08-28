#!/usr/bin/env bash
# Authenticate and measure the native MossFormer2-SS-16K forward.
# VAST-only: this worker never publishes, uploads, or mutates a remote model.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/mossformer2_ss_16k"
AUDITOR="$VOKRA_ROOT/tools/audit/mossformer2_ss_16k_manifest.py"
PREPARER="$VOKRA_ROOT/tools/parity/mossformer2_ss_16k_prepare_checkpoint.py"
REFERENCE_DUMPER="$VOKRA_ROOT/tools/parity/mossformer2_ss_16k_dump_reference.py"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

PUBLIC_REPO="vokra/mossformer2-ss-16k"
PUBLIC_REVISION="0e9ba9258cead4252f8e5279598af296ada08bf7"
PUBLIC_FILE="mossformer2-ss-16k.gguf"
PUBLIC_BYTES="223058240"
PUBLIC_SHA256="822516b75873dbeb814dac72f7ca0b5fb75254dd051dfdfdda54987347330f0c"
UPSTREAM_REPO="alibabasglab/MossFormer2_SS_16K"
UPSTREAM_REVISION="407cb030cd66340918ebb6c8cc63b18f8592cdbe"
UPSTREAM_FILE="last_best_checkpoint.pt"
UPSTREAM_BYTES="670353271"
UPSTREAM_SHA256="00a3a48bda492db1e829b85dd443f8f43a43039a3e90f1a24962ea9caf14a11a"
SOURCE_REPO="https://github.com/modelscope/ClearerVoice-Studio"
SOURCE_REVISION="6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61"
MANIFEST_SHA256="eb4b366872789b95228a172846259f6aa205a75c678f90941d5e8a3e9a47fb8b"

MIN_VAST_MEM_KIB=60000000
MIN_FREE_DISK_KIB=20000000
MIN_GPU_MEM_MIB=20000

log() { printf '[mossformer2-vast] %s\n' "$*" >&2; }
step() { printf '\n[mossformer2-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-mossformer2-ss-16k-validation.sh [--work-dir <empty-dir>]
       run-mossformer2-ss-16k-validation.sh --self-test

VAST-only source-separation gate. It downloads and authenticates the exact
public GGUF and upstream checkpoint, checks out the exact official source,
validates all 1,076 checkpoint tensors, generates an independent official CUDA
reference, compiles the Rust CPU/CLI paths, and records the first native CPU
measurement without inventing a numerical tolerance.

The Apple Metal path is cross-compiled here; its numerical run remains an
Apple-silicon step over the same evidence. This worker has no publication,
Hugging Face push, or model upload operation. Pull logs and reference evidence,
then destroy the instance.

Actual runs require Linux, VOKRA_PUBLISH_ON_VAST=1 from provision.sh, at least
60,000,000 KiB RAM, 20,000,000 KiB free disk, and a CUDA GPU with at least
20,000 MiB memory. Both Hugging Face repositories are public.
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

verify_file() {
  local path="$1" expected_bytes="$2" expected_hash="$3" actual_bytes actual_hash
  [[ -f "$path" ]] || die "missing pinned input: $path"
  actual_bytes="$(wc -c < "$path" | tr -d '[:space:]')"
  if [[ "$actual_bytes" != "$expected_bytes" ]]; then
    die "byte-size mismatch for $path: got $actual_bytes, expected $expected_bytes"
    return 2
  fi
  actual_hash="$(sha256_file "$path")"
  if [[ "$actual_hash" != "$expected_hash" ]]; then
    die "SHA-256 mismatch for $path: got $actual_hash, expected $expected_hash"
    return 2
  fi
  log "identity OK: $(basename "$path") bytes=$actual_bytes sha256=$actual_hash"
}

require_vast_marker() {
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] \
    || die "VOKRA_PUBLISH_ON_VAST=1 is absent; run provision.sh on VAST first"
}

require_vast_host() {
  local mem_kib free_kib gpu_mem_mib
  require_vast_marker
  [[ "$(uname -s)" == "Linux" ]] \
    || die "model work is Linux/VAST-only; refusing host $(uname -s)"
  [[ -r /proc/meminfo ]] || die "/proc/meminfo is unavailable"
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ -n "$mem_kib" ]] || die "could not read MemTotal"
  if (( mem_kib < MIN_VAST_MEM_KIB )); then
    die "MemTotal=${mem_kib} KiB is below the 64-GB class guard"
  fi
  mkdir -p "$VOKRA_SCRATCH"
  free_kib="$(df -Pk "$VOKRA_SCRATCH" | awk 'NR == 2 {print $4}')"
  [[ -n "$free_kib" ]] || die "could not read free disk"
  if (( free_kib < MIN_FREE_DISK_KIB )); then
    die "free disk=${free_kib} KiB is below the 20-GB run guard"
  fi
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
  gpu_mem_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]')"
  [[ "$gpu_mem_mib" =~ ^[0-9]+$ ]] || die "could not read CUDA GPU memory"
  if (( gpu_mem_mib < MIN_GPU_MEM_MIB )); then
    die "GPU memory=${gpu_mem_mib} MiB is below the 20,000-MiB reference guard"
  fi
}

require_tooling() {
  local tool
  for tool in uv cargo rustc rustup git awk grep find sort xargs tee wc tr nvidia-smi; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" ]] || die "$VOKRA_ROOT is not a git checkout"
  [[ -f "$PARITY_PROJECT/uv.lock" ]] || die "MossFormer2 parity uv.lock is missing"
  [[ -f "$AUDITOR" ]] || die "MossFormer2 manifest auditor is missing"
  [[ -f "$PREPARER" ]] || die "MossFormer2 checkpoint preparer is missing"
  [[ -f "$REFERENCE_DUMPER" ]] || die "MossFormer2 reference dumper is missing"
  if [[ -n "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]]; then
    die "VAST checkout must be clean so evidence names an exact commit"
  fi
}

checkout_exact_source() {
  local output="$1"
  [[ ! -e "$output" ]] || die "source target already exists: $output"
  mkdir -p "$output"
  git -C "$output" init -q
  git -C "$output" remote add origin "$SOURCE_REPO"
  git -C "$output" fetch -q --depth=1 origin "$SOURCE_REVISION"
  git -C "$output" checkout -q --detach FETCH_HEAD
  [[ "$(git -C "$output" rev-parse HEAD)" == "$SOURCE_REVISION" ]] \
    || die "source checkout did not land on $SOURCE_REVISION"
  [[ -z "$(git -C "$output" status --porcelain --untracked-files=all)" ]] \
    || die "official source checkout is not clean"
}

download_hf_file() {
  local repository="$1" revision="$2" filename="$3" output="$4"
  mkdir -p "$output"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
    'import sys; from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id=sys.argv[1], revision=sys.argv[2], filename=sys.argv[3], local_dir=sys.argv[4]))' \
    "$repository" "$revision" "$filename" "$output"
  [[ -f "$output/$filename" ]] \
    || die "Hugging Face download did not produce $output/$filename"
}

record_environment() {
  local output="$1" cpu_model cpu_flags
  cpu_model="$(awk -F ':' '$1 ~ /model name/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
  cpu_flags="$(awk -F ':' '$1 ~ /flags/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
  [[ -n "$cpu_model" && -n "$cpu_flags" ]] || die "CPU provenance unavailable"
  {
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "uname=$(uname -a)"
    echo "cpu_model=$cpu_model"
    echo "cpu_flags=$cpu_flags"
    echo "nproc=$(nproc)"
    awk '$1 == "MemTotal:" {print "mem_total_kib=" $2; exit}' /proc/meminfo
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    rustc --version --verbose
    cargo --version
    uv --version
    uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
      'import platform, huggingface_hub, torch; print(f"python={platform.python_version()}"); print(f"huggingface_hub={huggingface_hub.__version__}"); print(f"torch={torch.__version__}"); print(f"cuda={torch.version.cuda}"); print(f"cuda_available={torch.cuda.is_available()}")'
  } | tee "$output"
}

run_self_test() {
  local tmp payload actual script_path cases=0 fail=0
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp'" EXIT
  payload="$tmp/payload"
  printf 'vokra-mossformer2-self-test\n' > "$payload"
  actual="$(sha256_file "$payload")"

  cases=$((cases + 1))
  verify_file "$payload" "$(wc -c < "$payload" | tr -d '[:space:]')" "$actual" \
    >/dev/null 2>&1 || { log "self-test FAIL: valid identity rejected"; fail=1; }
  cases=$((cases + 1))
  if verify_file "$payload" 1 "$actual" >/dev/null 2>&1; then
    log "self-test FAIL: invalid size accepted"
    fail=1
  fi
  cases=$((cases + 1))
  if verify_file "$payload" "$(wc -c < "$payload" | tr -d '[:space:]')" \
    "$(printf '%064d' 0)" >/dev/null 2>&1; then
    log "self-test FAIL: invalid hash accepted"
    fail=1
  fi
  cases=$((cases + 1))
  if (unset VOKRA_PUBLISH_ON_VAST; require_vast_marker) >/dev/null 2>&1; then
    log "self-test FAIL: missing VAST marker accepted"
    fail=1
  fi
  cases=$((cases + 1))
  VOKRA_PUBLISH_ON_VAST=1 require_vast_marker >/dev/null 2>&1 \
    || { log "self-test FAIL: explicit VAST marker rejected"; fail=1; }

  cases=$((cases + 1))
  script_path="${BASH_SOURCE[0]}"
  for required in \
    "$PUBLIC_REVISION" "$PUBLIC_SHA256" "$UPSTREAM_REVISION" \
    "$UPSTREAM_SHA256" "$SOURCE_REVISION" "$MANIFEST_SHA256" \
    "mossformer2_ss_16k_manifest.py" \
    "mossformer2_ss_16k_prepare_checkpoint.py" \
    "mossformer2_ss_16k_dump_reference.py" \
    "--device cuda" "--frozen --python 3.12" \
    "measure_real_cpu_and_optional_metal_against_official" \
    "numeric_bounds=UNSET" "aarch64-apple-darwin"; do
    if ! grep -Fq -- "$required" "$script_path"; then
      log "self-test FAIL: worker contract lost token: $required"
      fail=1
    fi
  done
  cases=$((cases + 1))
  if grep -En '^[[:space:]]*(python3|python|pip)([[:space:]]|$)' "$script_path" >/dev/null; then
    log "self-test FAIL: direct Python/pip command found"
    fail=1
  fi
  cases=$((cases + 1))
  if grep -En '(publish-one\.sh|upload\.sh|--push([[:space:]]|$))' "$script_path" >/dev/null; then
    log "self-test FAIL: external publication operation found"
    fail=1
  fi

  rm -rf "$tmp"
  trap - EXIT
  if [[ $fail -eq 0 ]]; then
    echo "run-mossformer2-ss-16k-validation.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

main() {
  local self_test=0 requested_work_dir="" run_stamp work_dir inputs source stage logs reference
  local public_dir upstream_dir gguf checkpoint prepared prepared_manifest
  local audit_json reference_manifest run_log env_log compile_log cpu_log cli_log cross_log summary_file
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --work-dir)
        [[ $# -ge 2 && -n "$2" ]] || { die "--work-dir requires a directory"; return 2; }
        requested_work_dir="$2"
        shift 2
        ;;
      --self-test) self_test=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown argument: $1"; usage; return 2 ;;
    esac
  done
  if [[ $self_test -eq 1 ]]; then
    run_self_test
    return $?
  fi

  require_vast_host
  require_tooling
  run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  work_dir="${requested_work_dir:-$VOKRA_SCRATCH/mossformer2-ss-16k-validation/$run_stamp}"
  if [[ -e "$work_dir" ]] && [[ -n "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    die "--work-dir must be absent or empty: $work_dir"
  fi
  inputs="$work_dir/inputs"
  source="$work_dir/source/ClearerVoice-Studio"
  stage="$work_dir/stage"
  logs="$work_dir/logs"
  reference="$work_dir/reference"
  public_dir="$inputs/public"
  upstream_dir="$inputs/upstream"
  gguf="$public_dir/$PUBLIC_FILE"
  checkpoint="$upstream_dir/$UPSTREAM_FILE"
  prepared="$stage/mossformer2-ss-16k.safetensors"
  prepared_manifest="$logs/prepared-checkpoint-manifest.json"
  audit_json="$logs/public-gguf-audit.json"
  reference_manifest="$reference/manifest.json"
  mkdir -p "$inputs" "$stage" "$logs" "$reference"
  export UV_CACHE_DIR="$VOKRA_SCRATCH/uv-cache-mossformer2"
  export HF_HOME="$VOKRA_SCRATCH/hf-home-mossformer2"
  run_log="$logs/run.log"
  env_log="$logs/environment.txt"
  compile_log="$logs/models-compile.log"
  cpu_log="$logs/cpu-measurement.log"
  cli_log="$logs/cli-build.log"
  cross_log="$logs/apple-metal-cross-check.log"
  summary_file="$logs/summary.txt"
  exec > >(tee -a "$run_log") 2>&1
  trap 'rc=$?; if [[ -n "${summary_file:-}" && ! -f "$summary_file" ]]; then printf "execution_status=FAIL\nexit_code=%s\n" "$rc" > "$summary_file"; fi; exit "$rc"' EXIT

  step "Sync locked Python 3.12 parity environment"
  uv sync --project "$PARITY_PROJECT" --frozen --python 3.12

  step "Download immutable public GGUF and upstream checkpoint"
  download_hf_file "$PUBLIC_REPO" "$PUBLIC_REVISION" "$PUBLIC_FILE" "$public_dir"
  download_hf_file "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$UPSTREAM_FILE" "$upstream_dir"

  step "Check out exact official ClearerVoice source"
  checkout_exact_source "$source"

  step "Authenticate all model inputs"
  verify_file "$gguf" "$PUBLIC_BYTES" "$PUBLIC_SHA256"
  verify_file "$checkpoint" "$UPSTREAM_BYTES" "$UPSTREAM_SHA256"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python "$AUDITOR" \
    "$gguf" | tee "$audit_json"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
    'import json,pathlib,sys; data=json.loads(pathlib.Path(sys.argv[1]).read_text()); assert data["manifest_sha256"] == sys.argv[2]; assert data["tensor_count"] == 1076; assert data["parameter_count"] == 55735666; assert data["requires_full_file_confirmation"] is False; print("public GGUF contract authenticated")' \
    "$audit_json" "$MANIFEST_SHA256"

  step "Validate and normalize the exact upstream checkpoint manifest"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python "$PREPARER" \
    --repository "$VOKRA_ROOT" \
    --checkpoint "$checkpoint" \
    --output "$prepared" \
    --manifest "$prepared_manifest"

  step "Record environment before numerical output"
  record_environment "$env_log"

  step "Generate independent official CUDA reference"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python "$REFERENCE_DUMPER" \
    --source "$source" \
    --checkpoint "$checkpoint" \
    --device cuda \
    --output "$reference"
  [[ -f "$reference_manifest" ]] || die "official reference manifest is missing"
  grep -F 'UNSET_MEASURE_ON_VAST_BEFORE_RATIFICATION' "$reference_manifest" >/dev/null \
    || die "reference manifest lost its measurement-only numeric state"

  step "Compile native model tests and the CLI on VAST"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-models --lib --no-run 2>&1 | tee "$compile_log"
  cargo build --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-cli 2>&1 | tee "$cli_log"

  step "Measure native CPU core against the independent official reference"
  VOKRA_MOSSFORMER2_GGUF="$gguf" \
  VOKRA_MOSSFORMER2_REFERENCE_DIR="$reference" \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --lib \
      mossformer2_ss_16k::tests::measure_real_cpu_and_optional_metal_against_official \
      -- --ignored --exact --nocapture 2>&1 | tee "$cpu_log"
  grep -F 'MOSSFORMER2_SS_16K_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET' \
    "$cpu_log" >/dev/null || die "CPU measurement sentinel is missing"

  step "Cross-check Apple Metal model and CLI compilation"
  rustup target add aarch64-apple-darwin
  cargo check --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-cli --features metal --target aarch64-apple-darwin \
    2>&1 | tee "$cross_log"

  step "Write evidence summary and checksums"
  {
    echo "execution_status=PASS"
    echo "scope=AUTHENTICATED_PUBLIC_GGUF_OFFICIAL_REFERENCE_CPU_MEASUREMENT"
    echo "numeric_verdict=MEASURED_NOT_GATED"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "public_repo=$PUBLIC_REPO"
    echo "public_revision=$PUBLIC_REVISION"
    echo "upstream_repo=$UPSTREAM_REPO"
    echo "upstream_revision=$UPSTREAM_REVISION"
    echo "source_revision=$SOURCE_REVISION"
    echo "manifest_sha256=$MANIFEST_SHA256"
    echo "gguf_sha256=$(sha256_file "$gguf")"
    echo "checkpoint_sha256=$(sha256_file "$checkpoint")"
    echo "reference_manifest_sha256=$(sha256_file "$reference_manifest")"
    echo "metal_runtime=NOT_RUN_LINUX_VAST"
    echo "metal_cross_compile=PASS"
    grep -F 'MOSSFORMER2_SS_16K_MEASUREMENT_ONLY backend=cpu' "$cpu_log"
  } | tee "$summary_file"
  (
    cd "$work_dir"
    find logs reference -type f ! -name SHA256SUMS -print0 \
      | sort -z \
      | xargs -0 sha256sum > logs/SHA256SUMS
  )
  trap - EXIT
  log "PASS: pull $logs and $reference, then destroy the VAST instance"
}

main "$@"
