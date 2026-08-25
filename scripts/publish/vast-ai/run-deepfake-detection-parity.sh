#!/usr/bin/env bash
# Reproduce the first real-weight deepfake Wav2Vec2 CPU measurement on VAST.
# Downloads public inputs only; never publishes or uploads.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/deepfake_detection"
REFERENCE_SCRIPT="$VOKRA_ROOT/tools/parity/deepfake_detection_dump_reference.py"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

PUBLIC_REPO="vokra/deepfake-audio-detection-v2"
PUBLIC_REVISION="3eb02d838debc59d2c92d7e028f8cc2b52fcba30"
PUBLIC_FILE="deepfake-detection.gguf"
PUBLIC_BYTES=378295456
PUBLIC_SHA256="aa190afeb56f57042c8104d43b49b094e1efda069b7072e6e6d220894f7d4cf7"

UPSTREAM_REPO="MelodyMachine/Deepfake-audio-detection-V2"
UPSTREAM_REVISION="de3cde5a29c449bb5268814e421b46bf6ebdcd72"
CHECKPOINT_FILE="model.safetensors"
CHECKPOINT_BYTES=378302360
CHECKPOINT_SHA256="997d9ce59e63151d5e444a6fa7c863986d0e56d515f67321bd705ac3b01bc38c"
CONFIG_FILE="config.json"
CONFIG_BYTES=2509
CONFIG_SHA256="a7ff31ca7ba4dc7fb5c4847d6dff0cb8daa1f0ec512e6ff8190664874c5b2806"
PREPROCESSOR_FILE="preprocessor_config.json"
PREPROCESSOR_BYTES=215
PREPROCESSOR_SHA256="8cdfd65ff4115423185a1512bdae100e2e0cd744f5b322417429944aaafd0827"
SIGNAL_SHA256="b95320de8c0182cc0a916dbbfe03fa8a1103e5a2ab71cb56e217f8e712f51585"

MIN_VAST_MEM_KIB=30000000
MIN_FREE_DISK_KIB=12000000

log() { printf '[deepfake-vast] %s\n' "$*" >&2; }
step() { printf '\n[deepfake-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-deepfake-detection-parity.sh [--work-dir <empty-dir>]
       run-deepfake-detection-parity.sh --self-test

VAST-only real-weight CPU measurement worker. It downloads the exact public
Vokra GGUF and immutable upstream checkpoint/config/preprocessor, generates a
reference by directly importing Transformers 4.41.2, compiles vokra-models and
vokra-cli on VAST, and runs the ignored native CPU measurement. Numeric bounds
remain unset until this output and a real Apple CPU/Metal observation have been
reviewed.

Actual runs require Linux, VOKRA_PUBLISH_ON_VAST=1 from provision.sh, at least
30 GB RAM and 12 GB free disk. This script cannot publish or upload. Pull the
small logs/manifests, then destroy the instance.
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
  log "identity OK: $path bytes=$actual_bytes sha256=$actual_hash"
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
    die "MemTotal=${mem_kib} KiB is below the 30-GB guard"
  fi
  mkdir -p "$VOKRA_SCRATCH"
  free_kib="$(df -Pk "$VOKRA_SCRATCH" | awk 'NR == 2 {print $4}')"
  [[ -n "$free_kib" ]] || die "could not read free disk"
  if (( free_kib < MIN_FREE_DISK_KIB )); then
    die "free disk=${free_kib} KiB is below the 12-GB guard"
  fi
}

require_tooling() {
  local tool
  for tool in uv cargo rustc git awk grep find tee wc tr; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" ]] || die "$VOKRA_ROOT is not a git checkout"
  [[ -f "$PARITY_PROJECT/uv.lock" ]] || die "deepfake parity uv.lock is missing"
  [[ -f "$REFERENCE_SCRIPT" ]] || die "deepfake reference dumper is missing"
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
      'import platform, torch, transformers; print(f"python={platform.python_version()}"); print(f"torch={torch.__version__}"); print(f"transformers={transformers.__version__}")'
  } | tee "$output"
}

run_self_test() {
  local tmp payload actual script_path cases=0 fail=0
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp'" EXIT
  payload="$tmp/payload"
  printf 'vokra-deepfake-self-test\n' > "$payload"
  actual="$(sha256_file "$payload")"

  cases=$((cases + 1))
  verify_file "$payload" "$(wc -c < "$payload" | tr -d '[:space:]')" "$actual" \
    >/dev/null 2>&1 || { log "self-test FAIL: valid identity rejected"; fail=1; }

  cases=$((cases + 1))
  if verify_file "$payload" 1 "$actual" >/dev/null 2>&1; then
    log "self-test FAIL: invalid byte size accepted"
    fail=1
  fi

  cases=$((cases + 1))
  if verify_file "$payload" "$(wc -c < "$payload" | tr -d '[:space:]')" \
    "$(printf '%064d' 0)" >/dev/null 2>&1; then
    log "self-test FAIL: invalid SHA-256 accepted"
    fail=1
  fi

  cases=$((cases + 1))
  script_path="${BASH_SOURCE[0]}"
  for required in "$PUBLIC_REVISION" "$UPSTREAM_REVISION" "$CHECKPOINT_SHA256" \
    "$CONFIG_SHA256" "$PREPROCESSOR_SHA256" "$SIGNAL_SHA256" \
    "deepfake_detection_dump_reference.py" \
    "deepfake_detection::tests::measure_official_cpu_against_transformers" \
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
    echo "run-deepfake-detection-parity.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

main() {
  local self_test=0 requested_work_dir="" run_stamp work_dir inputs_dir logs_dir
  local public_dir upstream_dir gguf checkpoint config preprocessor reference
  local run_log env_log compile_log cli_log cpu_log summary_file
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
  work_dir="${requested_work_dir:-$VOKRA_SCRATCH/deepfake-parity/$run_stamp}"
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
  compile_log="$logs_dir/models-compile.log"
  cli_log="$logs_dir/cli-route.log"
  cpu_log="$logs_dir/cpu.log"
  summary_file="$logs_dir/summary.txt"
  exec > >(tee -a "$run_log") 2>&1
  # shellcheck disable=SC2154
  trap 'rc=$?; if [[ -n "${summary_file:-}" && ! -f "$summary_file" ]]; then printf "execution_status=FAIL\nexit_code=%s\n" "$rc" > "$summary_file"; fi; exit "$rc"' EXIT

  step "Sync locked Python 3.12 oracle"
  uv sync --project "$PARITY_PROJECT" --frozen --python 3.12

  step "Download exact public GGUF and immutable upstream inputs"
  download_hf_file "$PUBLIC_REPO" "$PUBLIC_REVISION" "$PUBLIC_FILE" "$public_dir"
  download_hf_file "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$CHECKPOINT_FILE" "$upstream_dir"
  download_hf_file "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$CONFIG_FILE" "$upstream_dir"
  download_hf_file "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$PREPROCESSOR_FILE" "$upstream_dir"
  gguf="$public_dir/$PUBLIC_FILE"
  checkpoint="$upstream_dir/$CHECKPOINT_FILE"
  config="$upstream_dir/$CONFIG_FILE"
  preprocessor="$upstream_dir/$PREPROCESSOR_FILE"
  verify_file "$gguf" "$PUBLIC_BYTES" "$PUBLIC_SHA256"
  verify_file "$checkpoint" "$CHECKPOINT_BYTES" "$CHECKPOINT_SHA256"
  verify_file "$config" "$CONFIG_BYTES" "$CONFIG_SHA256"
  verify_file "$preprocessor" "$PREPROCESSOR_BYTES" "$PREPROCESSOR_SHA256"

  step "Record environment before numerical output"
  record_environment "$env_log"

  step "Generate independent official Transformers reference"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python \
    "$REFERENCE_SCRIPT" --input-dir "$upstream_dir" --output-dir "$reference"
  verify_file "$reference/input_pcm.f32" 64000 "$SIGNAL_SHA256"
  cp "$reference/manifest.json" "$logs_dir/reference-manifest.json"

  step "Compile complete vokra-models library test target on VAST"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-models --lib --no-run 2>&1 | tee "$compile_log"

  step "Compile and verify the deepfake CLI dispatch route on VAST"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-cli engine::tests::load_session_routes_deepfake_detection_to_classification \
    -- --exact --nocapture 2>&1 | tee "$cli_log"

  step "Measure native CPU against independent official Transformers"
  VOKRA_DEEPFAKE_GGUF="$gguf" \
  VOKRA_DEEPFAKE_REFERENCE_DIR="$reference" \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --lib \
      deepfake_detection::tests::measure_official_cpu_against_transformers \
      -- --ignored --exact --nocapture 2>&1 | tee "$cpu_log"

  grep -F "DEEPFAKE_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET" "$cpu_log" >/dev/null \
    || die "CPU measurement sentinel missing"
  {
    echo "execution_status=PASS"
    echo "numeric_verdict=MEASURED_NOT_GATED"
    echo "numeric_bounds=UNSET"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "public_revision=$PUBLIC_REVISION"
    echo "public_sha256=$PUBLIC_SHA256"
    echo "upstream_revision=$UPSTREAM_REVISION"
    echo "checkpoint_sha256=$CHECKPOINT_SHA256"
    echo "transformers_version=4.41.2"
    echo "signal_sha256=$SIGNAL_SHA256"
    echo "reference_manifest_sha256=$(sha256_file "$reference/manifest.json")"
    grep -F "DEEPFAKE_MEASUREMENT" "$cpu_log"
  } | tee "$summary_file"
  trap - EXIT
  log "PASS: pull $logs_dir, then destroy the VAST instance"
}

main "$@"
