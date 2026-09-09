#!/usr/bin/env bash
# Reproduce the first real-weight YuE-upsampler CPU measurement on VAST.
# Downloads public inputs only; never publishes or uploads.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

PUBLIC_REPO="vokra/yue-upsampler"
PUBLIC_REVISION="6eea19bd301c5214123ee69217a61a989ffe80d0"
PUBLIC_FILE="yue-upsampler.gguf"
PUBLIC_SHA256="17df9c667c931544cf84545266d07e3598a9528d751ca6f281fffd305f4409ff"
UPSTREAM_REPO="m-a-p/YuE-upsampler"
UPSTREAM_REVISION="c6d7494a60555672be09ca809a40be400d682a53"
CHECKPOINT_FILE="decoder_151000.pth"
CHECKPOINT_SHA256="8af97a29d3483f9d4a3755992837501bd7d6caa1a69382ed16e64039e0ea0998"
MIN_VAST_MEM_KIB=60000000
MIN_FREE_DISK_KIB=12000000

log() { printf '[yue-upsampler-vast] %s\n' "$*" >&2; }
step() { printf '\n[yue-upsampler-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-yue-upsampler-parity.sh [--work-dir <empty-dir>]
       run-yue-upsampler-parity.sh --self-test

VAST-only measurement worker. It downloads the exact public GGUF and official
151k checkpoint, verifies their hashes, generates an independent reference by
directly importing vocos==0.1.0, compiles vokra-models on VAST, and runs the
ignored CPU measurement test. Numeric bounds remain unset until the output is
reviewed together with Apple-silicon CPU/Metal observations.
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

verify_hash() {
  local path="$1" expected="$2" actual
  [[ -f "$path" ]] || die "missing pinned input: $path"
  actual="$(sha256_file "$path")"
  if [[ "$actual" != "$expected" ]]; then
    die "SHA-256 mismatch for $path: got $actual, expected $expected"
    return 2
  fi
  log "SHA-256 OK: $(basename "$path") = $actual"
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

require_vast_host() {
  local mem_kib free_kib
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] \
    || die "VOKRA_PUBLISH_ON_VAST=1 is absent; run provision.sh first"
  [[ "$(uname -s)" == "Linux" ]] \
    || die "model work is Linux/VAST-only; refusing host $(uname -s)"
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ -n "$mem_kib" ]] || die "could not read MemTotal"
  if (( mem_kib < MIN_VAST_MEM_KIB )); then
    die "MemTotal=${mem_kib} KiB is below the VAST 64 GB class guard"
  fi
  mkdir -p "$VOKRA_SCRATCH"
  free_kib="$(df -Pk "$VOKRA_SCRATCH" | awk 'NR == 2 {print $4}')"
  [[ -n "$free_kib" ]] || die "could not read free disk"
  if (( free_kib < MIN_FREE_DISK_KIB )); then
    die "free disk=${free_kib} KiB is below the 12 GB run guard"
  fi
}

require_tooling() {
  local tool
  for tool in uv cargo rustc git awk grep tee; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" ]] || die "$VOKRA_ROOT is not a git checkout"
  [[ -f "$PARITY_PROJECT/uv.lock" ]] || die "parity uv.lock is missing"
  if [[ -n "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]]; then
    die "VAST checkout must be clean so evidence names an exact commit"
  fi
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
    rustc --version --verbose
    cargo --version
    uv --version
    uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
      'import platform, torch, vocos; print(f"python={platform.python_version()}"); print(f"torch={torch.__version__}"); print("vocos=0.1.0")'
  } | tee "$output"
}

run_self_test() {
  local tmp payload actual cases=0 fail=0 script_path
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp'" EXIT
  payload="$tmp/payload"
  printf 'vokra-yue-upsampler-self-test\n' > "$payload"
  actual="$(sha256_file "$payload")"
  cases=$((cases + 1))
  verify_hash "$payload" "$actual" >/dev/null 2>&1 \
    || { log "self-test FAIL: valid hash rejected"; fail=1; }
  cases=$((cases + 1))
  if verify_hash "$payload" "$(printf '%064d' 0)" >/dev/null 2>&1; then
    log "self-test FAIL: invalid hash accepted"
    fail=1
  fi
  cases=$((cases + 1))
  script_path="${BASH_SOURCE[0]}"
  for required in "$PUBLIC_REVISION" "$UPSTREAM_REVISION" "$CHECKPOINT_SHA256" \
    "yue_upsampler_dump_reference.py" \
    "yue_upsampler::tests::measure_real_cpu_against_official_vocos" \
    "--frozen --python 3.12" "--ignored --exact --nocapture"; do
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
  rm -rf "$tmp"
  trap - EXIT
  if [[ $fail -eq 0 ]]; then
    echo "run-yue-upsampler-parity.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

write_failure_summary_on_exit() {
  local rc=$?
  if [[ -n "${summary_file:-}" && ! -f "$summary_file" ]]; then
    printf "execution_status=FAIL\nexit_code=%s\n" "$rc" > "$summary_file"
  fi
  exit "$rc"
}

main() {
  local self_test=0 requested_work_dir="" run_stamp work_dir inputs_dir logs_dir
  local public_dir upstream_dir gguf checkpoint reference run_log env_log compile_log cpu_log summary_file
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
  work_dir="${requested_work_dir:-$VOKRA_SCRATCH/yue-upsampler-parity/$run_stamp}"
  if [[ -e "$work_dir" ]] && [[ -n "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    die "--work-dir must be absent or empty: $work_dir"
  fi
  inputs_dir="$work_dir/inputs"
  logs_dir="$work_dir/logs"
  public_dir="$inputs_dir/public"
  upstream_dir="$inputs_dir/upstream"
  reference="$work_dir/reference"
  mkdir -p "$logs_dir" "$public_dir" "$upstream_dir"
  export UV_CACHE_DIR="$VOKRA_SCRATCH/uv-cache"
  run_log="$logs_dir/run.log"
  env_log="$logs_dir/environment.txt"
  compile_log="$logs_dir/compile.log"
  cpu_log="$logs_dir/cpu.log"
  summary_file="$logs_dir/summary.txt"
  exec > >(tee -a "$run_log") 2>&1
  trap write_failure_summary_on_exit EXIT

  step "Sync locked Python 3.12 environment"
  uv sync --project "$PARITY_PROJECT" --frozen --python 3.12

  step "Download exact public GGUF and official checkpoint"
  download_hf_file "$PUBLIC_REPO" "$PUBLIC_REVISION" "$PUBLIC_FILE" "$public_dir"
  download_hf_file "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$CHECKPOINT_FILE" "$upstream_dir"
  gguf="$public_dir/$PUBLIC_FILE"
  checkpoint="$upstream_dir/$CHECKPOINT_FILE"
  verify_hash "$gguf" "$PUBLIC_SHA256"
  verify_hash "$checkpoint" "$CHECKPOINT_SHA256"

  step "Record environment before numerical output"
  record_environment "$env_log"

  step "Generate independent official vocos==0.1.0 reference"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python \
    "$PARITY_PROJECT/yue_upsampler_dump_reference.py" \
    --checkpoint "$checkpoint" --output-dir "$reference"
  cp "$reference/manifest.json" "$logs_dir/reference-manifest.json"

  step "Compile complete vokra-models library test target on VAST"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-models --lib --no-run 2>&1 | tee "$compile_log"

  step "Measure CPU against independent official forward"
  VOKRA_YUE_UPSAMPLER_GGUF="$gguf" \
  VOKRA_YUE_UPSAMPLER_REFERENCE_DIR="$reference" \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --lib \
      yue_upsampler::tests::measure_real_cpu_against_official_vocos \
      -- --ignored --exact --nocapture 2>&1 | tee "$cpu_log"

  grep -F "YUE_UPSAMPLER_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET" "$cpu_log" >/dev/null \
    || die "CPU measurement sentinel missing"
  {
    echo "execution_status=PASS"
    echo "numeric_verdict=MEASURED_NOT_GATED"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "public_revision=$PUBLIC_REVISION"
    echo "public_sha256=$PUBLIC_SHA256"
    echo "upstream_revision=$UPSTREAM_REVISION"
    echo "checkpoint_sha256=$CHECKPOINT_SHA256"
    echo "reference_manifest_sha256=$(sha256_file "$reference/manifest.json")"
  } | tee "$summary_file"
  trap - EXIT
  log "PASS: pull $logs_dir, then destroy the VAST instance"
}

main "$@"
