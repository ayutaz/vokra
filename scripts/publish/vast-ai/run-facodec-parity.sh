#!/usr/bin/env bash
# Reproduce NaturalSpeech 3 FACodec V2 CPU parity on VAST. No upload path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/facodec"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

PUBLIC_REPO="vokra/naturalspeech3-facodec-v2"
PUBLIC_REVISION="da6263e2c1a203641a5d4346a8a04d4eab4c738f"
PUBLIC_FILE="naturalspeech3-facodec-v2.gguf"
PUBLIC_BYTES=449251040
PUBLIC_SHA256="ee1d1e23266d6d2a898152d18bde156a2de008b8fe1eae9eeb392feca24c3084"

UPSTREAM_REPO="amphion/naturalspeech3_facodec"
UPSTREAM_REVISION="314afc3ea1455ba881a0e484ef9408b6cb996736"
ENCODER_FILE="ns3_facodec_encoder_v2.bin"
ENCODER_BYTES=17089391
ENCODER_SHA256="26636b05867f02f8da3690efb8c36f82909f0a8801ccc4bfdc73cdecf5f9c470"
DECODER_FILE="ns3_facodec_decoder_v2.bin"
DECODER_BYTES=432395901
DECODER_SHA256="e6a38d81916affae40a72f5517f39ebadeec4fefea67b074f21d4ec3a0156e3a"

SOURCE_REPO="https://github.com/open-mmlab/Amphion.git"
SOURCE_REVISION="26f6883110181f1dbfe95c70a7c7dbaf4de5f42a"

MIN_VAST_MEM_KIB=15000000
MIN_FREE_DISK_KIB=20000000

log() { printf '[facodec-vast] %s\n' "$*" >&2; }
step() { printf '\n[facodec-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-facodec-parity.sh [--work-dir <empty-dir>]
       run-facodec-parity.sh --self-test

VAST-only, non-publishing NaturalSpeech 3 FACodec V2 validation. The worker
downloads and verifies the exact public Vokra GGUF and official V2 encoder /
decoder, checks out the immutable official Amphion source, creates an
independent CPU reference, compiles the workspace plus Apple target, verifies
CLI routing, and compares native CPU encode/decode.

There is no --push flag and no upload command. Pull the small logs/reference
directory and destroy the VAST instance rather than stopping it. Real Metal
execution is a separate remote Apple Silicon gate; never run it on the
maintainer Mac.
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
  if [[ ! -f "$path" ]]; then
    die "missing pinned input: $path"
    return 2
  fi
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
    || die "real FACodec model work is VAST/Linux-only; refusing $(uname -s)"
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$mem_kib" =~ ^[0-9]+$ ]] || die "could not read MemTotal"
  if (( mem_kib < MIN_VAST_MEM_KIB )); then
    die "MemTotal=${mem_kib} KiB is below the VAST 16-GB guard"
  fi
  mkdir -p "$VOKRA_SCRATCH"
  free_kib="$(df -Pk "$VOKRA_SCRATCH" | awk 'NR == 2 {print $4}')"
  [[ "$free_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
  if (( free_kib < MIN_FREE_DISK_KIB )); then
    die "free disk=${free_kib} KiB is below the 20-GB guard"
  fi
}

require_tooling() {
  local tool
  for tool in uv cargo rustc rustup git curl awk find tee wc tr; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" ]] || die "$VOKRA_ROOT is not a git checkout"
  [[ -f "$PARITY_PROJECT/uv.lock" ]] || die "FACodec parity uv.lock is missing"
  [[ -f "$PARITY_PROJECT/dump_reference.py" ]] || die "FACodec reference dumper is missing"
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
  printf 'vokra-facodec-self-test\n' > "$payload"
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
    "$PUBLIC_SHA256" "$ENCODER_SHA256" "$DECODER_SHA256" \
    "facodec/dump_reference.py" "parity_facodec_real" \
    "load_session_routes_facodec_to_the_factorized_codec_task" \
    "aarch64-apple-darwin" "--frozen --python 3.12"; do
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
  if grep -En -- '^[[:space:]]*(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh)([[:space:]]|$)' "$script_path" >/dev/null; then
    log "self-test FAIL: publishing command found"
    fail=1
  fi

  rm -rf "$tmp"
  trap - EXIT
  if [[ $fail -eq 0 ]]; then
    echo "run-facodec-parity.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

main() {
  local self_test=0 requested_work_dir="" run_stamp work_dir inputs_dir sources_dir logs_dir
  local public_dir upstream_dir source_dir gguf encoder decoder reference
  local run_log env_log compile_log apple_log cli_log cpu_log summary_file
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
  work_dir="${requested_work_dir:-$VOKRA_SCRATCH/facodec-parity/$run_stamp}"
  if [[ -e "$work_dir" ]] && [[ -n "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    die "--work-dir must be absent or empty: $work_dir"
  fi
  inputs_dir="$work_dir/inputs"
  sources_dir="$work_dir/sources"
  logs_dir="$work_dir/logs"
  public_dir="$inputs_dir/public"
  upstream_dir="$inputs_dir/upstream"
  source_dir="$sources_dir/Amphion"
  reference="$work_dir/reference"
  mkdir -p "$logs_dir" "$public_dir" "$upstream_dir" "$sources_dir"
  export UV_CACHE_DIR="$VOKRA_SCRATCH/uv-cache"
  run_log="$logs_dir/run.log"
  env_log="$logs_dir/environment.txt"
  compile_log="$logs_dir/compile.log"
  apple_log="$logs_dir/apple-cross-check.log"
  cli_log="$logs_dir/cli-route.log"
  cpu_log="$logs_dir/cpu.log"
  summary_file="$logs_dir/summary.txt"
  exec > >(tee -a "$run_log") 2>&1
  trap 'rc=$?; if [[ -n "${summary_file:-}" && ! -f "$summary_file" ]]; then printf "execution_status=FAIL\nexit_code=%s\n" "$rc" > "$summary_file"; fi; exit "$rc"' EXIT

  step "Sync locked Python 3.12 reference environment"
  uv sync --project "$PARITY_PROJECT" --frozen --python 3.12

  step "Download exact public GGUF and official V2 checkpoints"
  gguf="$public_dir/$PUBLIC_FILE"
  encoder="$upstream_dir/$ENCODER_FILE"
  decoder="$upstream_dir/$DECODER_FILE"
  download_hf_file "$PUBLIC_REPO" "$PUBLIC_REVISION" "$PUBLIC_FILE" "$gguf"
  download_hf_file "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$ENCODER_FILE" "$encoder"
  download_hf_file "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$DECODER_FILE" "$decoder"
  verify_file "$gguf" "$PUBLIC_BYTES" "$PUBLIC_SHA256"
  verify_file "$encoder" "$ENCODER_BYTES" "$ENCODER_SHA256"
  verify_file "$decoder" "$DECODER_BYTES" "$DECODER_SHA256"

  step "Check out exact official Amphion source"
  checkout_exact_source "$SOURCE_REPO" "$SOURCE_REVISION" "$source_dir"

  step "Record environment before numerical output"
  record_environment "$env_log"

  step "Generate independent official FACodec V2 reference"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python \
    "$PARITY_PROJECT/dump_reference.py" \
    --source "$source_dir" --encoder "$encoder" --decoder "$decoder" \
    --output "$reference"
  cp "$reference/manifest.json" "$logs_dir/reference-manifest.json"

  step "Compile all workspace targets on VAST"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    --workspace --no-run 2>&1 | tee "$compile_log"

  step "Cross-check the Apple Metal target compiles"
  rustup target add aarch64-apple-darwin
  cargo check --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-models --features metal --target aarch64-apple-darwin \
    2>&1 | tee "$apple_log"

  step "Verify the FACodec CLI dispatch route"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-cli engine::tests::load_session_routes_facodec_to_the_factorized_codec_task \
    -- --exact --nocapture 2>&1 | tee "$cli_log"

  step "Compare native CPU encode/decode with the official reference"
  VOKRA_FACODEC_GGUF="$gguf" \
  VOKRA_FACODEC_PARITY_DIR="$reference" \
  VOKRA_FACODEC_BACKEND=cpu \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --test parity_facodec_real \
      real_facodec_v2_encode_decode_matches_official -- --exact --nocapture \
      2>&1 | tee "$cpu_log"
  grep -F "FACodec V2 Cpu:" "$cpu_log" >/dev/null \
    || die "CPU parity measurement sentinel missing"

  {
    echo "execution_status=PASS"
    echo "numeric_verdict=FP32_ATOL_0.01_PASS"
    echo "codes=exact"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "public_revision=$PUBLIC_REVISION"
    echo "public_sha256=$PUBLIC_SHA256"
    echo "upstream_revision=$UPSTREAM_REVISION"
    echo "encoder_sha256=$ENCODER_SHA256"
    echo "decoder_sha256=$DECODER_SHA256"
    echo "source_revision=$SOURCE_REVISION"
    echo "reference_manifest_sha256=$(sha256_file "$reference/manifest.json")"
    echo "metal_runtime=REQUIRES_REMOTE_APPLE_SILICON"
  } | tee "$summary_file"
  trap - EXIT
  log "PASS: pull $logs_dir and $reference, then destroy the VAST instance"
}

main "$@"
