#!/usr/bin/env bash
# Reproduce the first real-public-GGUF FunCodec CPU parity run on VAST.
# Downloads public inputs only; never converts, publishes, or uploads.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/funcodec"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

PUBLIC_REPO="vokra/funcodec"
PUBLIC_REVISION="ea8be2e051ede0365496e8cd3b24d732c8bc6ffb"
PUBLIC_FILE="model.gguf"
PUBLIC_BYTES=95072832
PUBLIC_SHA256="b6fa6c903e23b1785f517f4e6c33c5d323a227a94ea757442e5d177d48d5781d"

UPSTREAM_REPO="alibaba-damo/audio_codec-encodec-zh_en-general-16k-nq32ds320-pytorch"
UPSTREAM_REVISION="ef9fbae4943cb272b8803e8a0f3c974fa1003b1f"
CHECKPOINT_FILE="model.pth"
CHECKPOINT_BYTES=95149521
CHECKPOINT_SHA256="08dd881b74daa150c405418b613496e872bbad4edd2d3c1d6d94ecf7199ac42c"
CONFIG_FILE="config.yaml"
CONFIG_SHA256="5830ffe0c8cad9e8678dca1e5c6873a89629c23007155068f485ca44b2af9c4e"

SOURCE_REPO="https://github.com/modelscope/FunCodec.git"
SOURCE_REVISION="b467b73e4025a123a68e64de9ba445d6a57d1984"

MIN_VAST_MEM_KIB=15000000
MIN_FREE_DISK_KIB=20000000

log() { printf '[funcodec-vast] %s\n' "$*" >&2; }
step() { printf '\n[funcodec-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-funcodec-parity.sh [--work-dir <empty-dir>]
       run-funcodec-parity.sh --self-test

VAST-only real-weight CPU parity worker. It downloads the exact public Vokra
GGUF and immutable official checkpoint/config, checks out the pinned official
FunCodec source, verifies every identity, generates the independent official
token-to-waveform reference, compiles the relevant Rust targets, verifies the
CLI dispatch, and runs the native CPU comparison.

Actual runs require Linux, VOKRA_PUBLISH_ON_VAST=1 from provision.sh, at least
16 GB RAM and 20 GB free disk. The worker contains no publish/upload operation.
Pull the small logs and reference manifest, then destroy the instance.
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
  if [[ -n "$expected_bytes" && "$actual_bytes" != "$expected_bytes" ]]; then
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
  local repository="$1" revision="$2" filename="$3" output="$4" url
  mkdir -p "$(dirname "$output")"
  url="https://huggingface.co/$repository/resolve/$revision/$filename?download=true"
  curl --fail --location --retry 5 --retry-delay 2 --output "$output" "$url"
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
    die "MemTotal=${mem_kib} KiB is below the VAST 16-GB guard"
  fi
  mkdir -p "$VOKRA_SCRATCH"
  free_kib="$(df -Pk "$VOKRA_SCRATCH" | awk 'NR == 2 {print $4}')"
  [[ -n "$free_kib" ]] || die "could not read free disk"
  if (( free_kib < MIN_FREE_DISK_KIB )); then
    die "free disk=${free_kib} KiB is below the 20-GB guard"
  fi
}

require_tooling() {
  local tool
  for tool in uv cargo rustc git curl awk grep find tee wc tr; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" ]] || die "$VOKRA_ROOT is not a git checkout"
  [[ -f "$PARITY_PROJECT/uv.lock" ]] || die "FunCodec parity uv.lock is missing"
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
  printf 'vokra-funcodec-self-test\n' > "$payload"
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
  for required in "$PUBLIC_REVISION" "$UPSTREAM_REVISION" "$SOURCE_REVISION" \
    "$CHECKPOINT_SHA256" "funcodec/dump_reference.py" \
    "parity_funcodec_real" "load_session_routes_funcodec_to_the_residual_vq_codec_task" \
    "--frozen --python 3.12"; do
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
    echo "run-funcodec-parity.sh self-test: OK ($cases cases)"
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
  local self_test=0 requested_work_dir="" run_stamp work_dir inputs_dir sources_dir logs_dir
  local public_dir upstream_dir source_dir gguf checkpoint config reference
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
  work_dir="${requested_work_dir:-$VOKRA_SCRATCH/funcodec-parity/$run_stamp}"
  if [[ -e "$work_dir" ]] && [[ -n "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    die "--work-dir must be absent or empty: $work_dir"
  fi
  inputs_dir="$work_dir/inputs"
  sources_dir="$work_dir/sources"
  logs_dir="$work_dir/logs"
  public_dir="$inputs_dir/public"
  upstream_dir="$inputs_dir/upstream"
  source_dir="$sources_dir/funcodec"
  reference="$work_dir/reference"
  mkdir -p "$logs_dir" "$public_dir" "$upstream_dir" "$sources_dir"
  export UV_CACHE_DIR="$VOKRA_SCRATCH/uv-cache"
  run_log="$logs_dir/run.log"
  env_log="$logs_dir/environment.txt"
  compile_log="$logs_dir/compile.log"
  cli_log="$logs_dir/cli-route.log"
  cpu_log="$logs_dir/cpu.log"
  summary_file="$logs_dir/summary.txt"
  exec > >(tee -a "$run_log") 2>&1
  trap write_failure_summary_on_exit EXIT

  step "Sync locked Python 3.12 environment"
  uv sync --project "$PARITY_PROJECT" --frozen --python 3.12

  step "Download exact public GGUF and official checkpoint/config"
  gguf="$public_dir/$PUBLIC_FILE"
  checkpoint="$upstream_dir/$CHECKPOINT_FILE"
  config="$upstream_dir/$CONFIG_FILE"
  download_hf_file "$PUBLIC_REPO" "$PUBLIC_REVISION" "$PUBLIC_FILE" "$gguf"
  download_hf_file "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$CHECKPOINT_FILE" "$checkpoint"
  download_hf_file "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$CONFIG_FILE" "$config"
  verify_file "$gguf" "$PUBLIC_BYTES" "$PUBLIC_SHA256"
  verify_file "$checkpoint" "$CHECKPOINT_BYTES" "$CHECKPOINT_SHA256"
  verify_file "$config" "" "$CONFIG_SHA256"

  step "Check out exact official FunCodec source"
  checkout_exact_source "$SOURCE_REPO" "$SOURCE_REVISION" "$source_dir"

  step "Record environment before numerical output"
  record_environment "$env_log"

  step "Generate independent official FunCodec reference"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python \
    "$PARITY_PROJECT/dump_reference.py" \
    --source "$source_dir" --checkpoint "$checkpoint" --config "$config" \
    --output "$reference" --frames 4 --num-quantizers 32
  cp "$reference/manifest.json" "$logs_dir/reference-manifest.json"

  step "Compile relevant vokra-models integration target on VAST"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-models --test parity_funcodec_real --no-run 2>&1 | tee "$compile_log"

  step "Compile and verify the FunCodec CLI dispatch route on VAST"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-cli engine::tests::load_session_routes_funcodec_to_the_residual_vq_codec_task \
    -- --exact --nocapture 2>&1 | tee "$cli_log"

  step "Compare native CPU decode with the independent official reference"
  VOKRA_FUNCODEC_GGUF="$gguf" \
  VOKRA_FUNCODEC_REFERENCE_DIR="$reference" \
  VOKRA_FUNCODEC_BACKEND=cpu \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --test parity_funcodec_real \
      real_funcodec_decode_matches_official -- --exact --nocapture 2>&1 | tee "$cpu_log"
  grep -F "FunCodec Cpu:" "$cpu_log" >/dev/null || die "CPU parity measurement sentinel missing"

  {
    echo "execution_status=PASS"
    echo "numeric_verdict=FP32_ATOL_0.01_PASS"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "public_revision=$PUBLIC_REVISION"
    echo "public_sha256=$PUBLIC_SHA256"
    echo "upstream_revision=$UPSTREAM_REVISION"
    echo "checkpoint_sha256=$CHECKPOINT_SHA256"
    echo "source_revision=$SOURCE_REVISION"
    echo "reference_manifest_sha256=$(sha256_file "$reference/manifest.json")"
  } | tee "$summary_file"
  trap - EXIT
  log "PASS: pull $logs_dir, then destroy the VAST instance"
}

main "$@"
