#!/usr/bin/env bash
# Reproduce the first real-weight emotion2vec+ Large CPU measurement on VAST.
# Downloads public inputs only; never publishes or uploads.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

PUBLIC_REPO="vokra/emotion2vec"
PUBLIC_REVISION="fcdce49fd5ce07ffd37c2f18aaa3ec6fd6c3b78e"
PUBLIC_FILE="model.gguf"
PUBLIC_BYTES=648576992
PUBLIC_SHA256="052efcdaa000208933bfe1633ae81115fa9aa05b043920bb1cfa92f2827f02bc"

UPSTREAM_REPO="emotion2vec/emotion2vec_plus_large"
UPSTREAM_REVISION="6c303ba987b86b93193de93e34bb2b077a6bedc4"
CHECKPOINT_FILE="model.pt"
CHECKPOINT_BYTES=1945790254
CHECKPOINT_SHA256="be501a01f26fcdc7663a062dff86af839afbaef7c4de32f5e42d7e1ad2784da4"
CONFIG_FILE="config.yaml"
CONFIG_BYTES=5552
CONFIG_SHA256="f4fa0eb82cc78bfebb43c56d68791afb01788085a18897d20999af7bc45d51d3"
TOKENS_FILE="tokens.txt"
TOKENS_BYTES=119
TOKENS_SHA256="866121e470057b847d7a50e9923509141fb2924392f53385a186482a1ec0fb7f"
WAV_FILE="example/test.wav"
WAV_BYTES=131376
WAV_SHA256="a4839eaaa3d54bd2db6eb48aa3d40def1b5c5004df3fd163a8dcd045097f8a23"

FUNASR_REPO="https://github.com/modelscope/FunASR.git"
FUNASR_REVISION="2f7dcbad90e82e964ab381ad63ff5109dd92327d"

MIN_VAST_MEM_KIB=60000000
MIN_FREE_DISK_KIB=20000000

log() { printf '[emotion2vec-vast] %s\n' "$*" >&2; }
step() { printf '\n[emotion2vec-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-emotion2vec-parity.sh [--work-dir <empty-dir>]
       run-emotion2vec-parity.sh --self-test

VAST-only real-weight CPU measurement worker. It downloads the exact public
Vokra GGUF and immutable official checkpoint/config/tokens/example WAV, checks
out the pinned FunASR source, generates an independent official reference,
compiles vokra-models and vokra-cli on VAST, and runs the ignored native CPU
measurement. Numeric bounds remain unset until this output and a real Apple
CPU/Metal observation have been reviewed.

Actual runs require Linux, VOKRA_PUBLISH_ON_VAST=1 from provision.sh, the
64-GB RAM class and at least 20 GB free disk. This script has no publish or
upload operation. Pull the small logs/manifests, then destroy the instance.
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

checkout_exact_source() {
  local repository="$1" revision="$2" output="$3"
  [[ ! -e "$output" ]] || die "source target already exists: $output"
  mkdir -p "$output"
  git -C "$output" init -q
  git -C "$output" remote add origin "$repository"
  git -C "$output" fetch -q --depth=1 origin "$revision"
  git -C "$output" checkout -q --detach FETCH_HEAD
  [[ "$(git -C "$output" rev-parse HEAD)" == "$revision" ]] \
    || die "source checkout did not land on $revision"
  [[ -z "$(git -C "$output" status --porcelain --untracked-files=all)" ]] \
    || die "source checkout is not clean: $output"
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
    die "MemTotal=${mem_kib} KiB is below the VAST 64-GB class guard"
  fi
  mkdir -p "$VOKRA_SCRATCH"
  free_kib="$(df -Pk "$VOKRA_SCRATCH" | awk 'NR == 2 {print $4}')"
  [[ -n "$free_kib" ]] || die "could not read free disk"
  if (( free_kib < MIN_FREE_DISK_KIB )); then
    die "free disk=${free_kib} KiB is below the 20-GB run guard"
  fi
}

require_tooling() {
  local tool
  for tool in uv cargo rustc git awk grep find tee wc tr; do
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
      'import platform, torch; print(f"python={platform.python_version()}"); print(f"torch={torch.__version__}"); print(f"torch_cpu_capability={torch.backends.cpu.get_cpu_capability()}")'
  } | tee "$output"
}

run_self_test() {
  local tmp payload actual script_path cases=0 fail=0
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp'" EXIT
  payload="$tmp/payload"
  printf 'vokra-emotion2vec-self-test\n' > "$payload"
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
  for required in "$PUBLIC_REVISION" "$UPSTREAM_REVISION" "$FUNASR_REVISION" \
    "$CHECKPOINT_SHA256" "example/test.wav" \
    "emotion2vec_dump_reference.py" \
    "emotion2vec::tests::measure_official_cpu_against_funasr" \
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
    echo "run-emotion2vec-parity.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

main() {
  local self_test=0 requested_work_dir="" run_stamp work_dir inputs_dir sources_dir logs_dir
  local public_dir upstream_dir funasr_source gguf checkpoint config tokens wav reference
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
  work_dir="${requested_work_dir:-$VOKRA_SCRATCH/emotion2vec-parity/$run_stamp}"
  if [[ -e "$work_dir" ]] && [[ -n "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    die "--work-dir must be absent or empty: $work_dir"
  fi
  inputs_dir="$work_dir/inputs"
  sources_dir="$work_dir/sources"
  logs_dir="$work_dir/logs"
  public_dir="$inputs_dir/public"
  upstream_dir="$inputs_dir/upstream"
  funasr_source="$sources_dir/funasr"
  reference="$work_dir/reference"
  mkdir -p "$logs_dir" "$public_dir" "$upstream_dir" "$sources_dir"
  export UV_CACHE_DIR="$VOKRA_SCRATCH/uv-cache"
  run_log="$logs_dir/run.log"
  env_log="$logs_dir/environment.txt"
  compile_log="$logs_dir/models-compile.log"
  cli_log="$logs_dir/cli-route.log"
  cpu_log="$logs_dir/cpu.log"
  summary_file="$logs_dir/summary.txt"
  exec > >(tee -a "$run_log") 2>&1
  # `rc` is assigned inside the single-quoted EXIT trap body.
  # shellcheck disable=SC2154
  trap 'rc=$?; if [[ -n "${summary_file:-}" && ! -f "$summary_file" ]]; then printf "execution_status=FAIL\nexit_code=%s\n" "$rc" > "$summary_file"; fi; exit "$rc"' EXIT

  step "Sync locked Python 3.12 environment"
  uv sync --project "$PARITY_PROJECT" --frozen --python 3.12

  step "Download exact public GGUF and official inputs"
  download_hf_file "$PUBLIC_REPO" "$PUBLIC_REVISION" "$PUBLIC_FILE" "$public_dir"
  download_hf_file "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$CHECKPOINT_FILE" "$upstream_dir"
  download_hf_file "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$CONFIG_FILE" "$upstream_dir"
  download_hf_file "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$TOKENS_FILE" "$upstream_dir"
  download_hf_file "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$WAV_FILE" "$upstream_dir"
  gguf="$public_dir/$PUBLIC_FILE"
  checkpoint="$upstream_dir/$CHECKPOINT_FILE"
  config="$upstream_dir/$CONFIG_FILE"
  tokens="$upstream_dir/$TOKENS_FILE"
  wav="$upstream_dir/$WAV_FILE"
  verify_file "$gguf" "$PUBLIC_BYTES" "$PUBLIC_SHA256"
  verify_file "$checkpoint" "$CHECKPOINT_BYTES" "$CHECKPOINT_SHA256"
  verify_file "$config" "$CONFIG_BYTES" "$CONFIG_SHA256"
  verify_file "$tokens" "$TOKENS_BYTES" "$TOKENS_SHA256"
  verify_file "$wav" "$WAV_BYTES" "$WAV_SHA256"

  step "Check out exact official FunASR source"
  checkout_exact_source "$FUNASR_REPO" "$FUNASR_REVISION" "$funasr_source"

  step "Record environment before numerical output"
  record_environment "$env_log"

  step "Generate independent official FunASR reference"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python \
    "$PARITY_PROJECT/emotion2vec_dump_reference.py" \
    --funasr-source "$funasr_source" \
    --checkpoint "$checkpoint" --config "$config" --tokens "$tokens" \
    --wav "$wav" --output-dir "$reference"
  cp "$reference/manifest.json" "$logs_dir/reference-manifest.json"

  step "Compile complete vokra-models library test target on VAST"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-models --lib --no-run 2>&1 | tee "$compile_log"

  step "Compile and verify the emotion2vec CLI dispatch route on VAST"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-cli engine::tests::load_session_routes_emotion2vec_to_classification \
    -- --exact --nocapture 2>&1 | tee "$cli_log"

  step "Measure native CPU against independent official FunASR"
  VOKRA_EMOTION2VEC_GGUF="$gguf" \
  VOKRA_EMOTION2VEC_REFERENCE_DIR="$reference" \
  VOKRA_EMOTION2VEC_WAV="$wav" \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --lib \
      emotion2vec::tests::measure_official_cpu_against_funasr \
      -- --ignored --exact --nocapture 2>&1 | tee "$cpu_log"

  grep -F "EMOTION2VEC_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET" "$cpu_log" >/dev/null \
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
    echo "funasr_revision=$FUNASR_REVISION"
    echo "reference_manifest_sha256=$(sha256_file "$reference/manifest.json")"
    grep -F "EMOTION2VEC_MEASUREMENT" "$cpu_log"
  } | tee "$summary_file"
  trap - EXIT
  log "PASS: pull $logs_dir, then destroy the VAST instance"
}

main "$@"
