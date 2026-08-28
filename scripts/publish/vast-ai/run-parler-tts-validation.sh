#!/usr/bin/env bash
# Reproduce Parler-TTS Mini English/Multilingual CPU parity on VAST. No upload path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/parler_tts"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

PARLER_SOURCE_REVISION="d108732cd57788ec86bc857d99a6cabd66663d68"

ENGLISH_PUBLIC_REPO="vokra/parler-tts-mini-v1"
ENGLISH_PUBLIC_REVISION="cb02a124c8d125231b396a293608f2488ae2e4d2"
ENGLISH_PUBLIC_FILE="parler-tts-mini-v1.gguf"
ENGLISH_PUBLIC_BYTES=3511459168
ENGLISH_PUBLIC_SHA256="7f69b811edae6cbe82fdfa8e72e6181945d4466748349aa74d994fb566785ddc"
ENGLISH_UPSTREAM_REPO="parler-tts/parler-tts-mini-v1"
ENGLISH_UPSTREAM_REVISION="0392b9451a601e528fd863bbb0598431fee810d9"
ENGLISH_CHECKPOINT_BYTES=3511490560
ENGLISH_CHECKPOINT_SHA256="bc430eb6752b96ffb3f67036d1a6e207fbd031575a775716ffa64ef1eeb03692"
ENGLISH_CONFIG_BYTES=6930
ENGLISH_CONFIG_SHA256="d8d2afa72bf3b098263a073c4d4df18627b76e1eb454c48f60bc5f787b2433b1"
ENGLISH_GENERATION_BYTES=265
ENGLISH_GENERATION_SHA256="77831b39a5e0c4dba09b4dcbe37ce082e10f94c646920b20678c9c5289e52440"

MULTILINGUAL_PUBLIC_REPO="vokra/parler-tts-mini-multilingual"
MULTILINGUAL_PUBLIC_REVISION="6f0f56788f06e6d514e0fab8530663b8af8b1fe2"
MULTILINGUAL_PUBLIC_FILE="parler-tts-mini.gguf"
MULTILINGUAL_PUBLIC_BYTES=3751292736
MULTILINGUAL_PUBLIC_SHA256="d1edf792305a486192be73dfb279891febb6e81735abf06b2ae90b29da94134d"
MULTILINGUAL_UPSTREAM_REPO="parler-tts/parler-tts-mini-multilingual-v1.1"
MULTILINGUAL_UPSTREAM_REVISION="11b27d57855dec1ce0914ba1f12363bf2ea75ba3"
MULTILINGUAL_CHECKPOINT_BYTES=3751321772
MULTILINGUAL_CHECKPOINT_SHA256="79c64e3705e0ccce122988c7817f0d65efa3fd37625906d90765858bdab38412"
MULTILINGUAL_CONFIG_BYTES=7467
MULTILINGUAL_CONFIG_SHA256="06d4cb727521542cab6b26d3ad1c8517d51fd1f551600ec67a59575364e221c6"
MULTILINGUAL_GENERATION_BYTES=218
MULTILINGUAL_GENERATION_SHA256="3bb518e78ea5f32fbbcfc7f0aaed388e7aefede474d2bf4b8cf4502fd6b27a92"

MIN_VAST_MEM_KIB=30000000
MIN_FREE_DISK_KIB=60000000

log() { printf '[parler-vast] %s\n' "$*" >&2; }
step() { printf '\n[parler-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-parler-tts-validation.sh [--work-dir <empty-dir>]
       run-parler-tts-validation.sh --self-test

VAST-only, non-publishing Parler-TTS Mini English/Multilingual validation. The
worker downloads and verifies both exact public Vokra GGUFs and exact immutable
upstream checkpoints, uses the locked official Parler-TTS source plus
Transformers 4.46.1 for independent greedy references, compiles the workspace
and Apple target, verifies CLI routing, and compares native CPU T5 states,
generated codes, and embedded-DAC PCM.

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
  [[ -f "$path" ]] || { die "missing pinned input: $path"; return 2; }
  actual_bytes="$(wc -c < "$path" | tr -d '[:space:]')"
  [[ "$actual_bytes" == "$expected_bytes" ]] \
    || { die "byte-size mismatch for $path: got $actual_bytes, expected $expected_bytes"; return 2; }
  actual_hash="$(sha256_file "$path")"
  [[ "$actual_hash" == "$expected_hash" ]] \
    || { die "SHA-256 mismatch for $path: got $actual_hash, expected $expected_hash"; return 2; }
  log "identity OK: $path bytes=$actual_bytes sha256=$actual_hash"
}

download_hf_file() {
  local repository="$1" revision="$2" filename="$3" output="$4" url
  mkdir -p "$(dirname "$output")"
  url="https://huggingface.co/$repository/resolve/$revision/$filename?download=true"
  curl --fail --location --retry 5 --retry-delay 2 --output "$output" "$url"
}

require_vast_host() {
  local mem_kib free_kib
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] \
    || die "VOKRA_PUBLISH_ON_VAST=1 is absent; run provision.sh first"
  [[ "$(uname -s)" == "Linux" ]] \
    || die "real Parler model work is VAST/Linux-only; refusing $(uname -s)"
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$mem_kib" =~ ^[0-9]+$ ]] || die "could not read MemTotal"
  if (( mem_kib < MIN_VAST_MEM_KIB )); then
    die "MemTotal=${mem_kib} KiB is below the VAST 32-GB guard"
  fi
  mkdir -p "$VOKRA_SCRATCH"
  free_kib="$(df -Pk "$VOKRA_SCRATCH" | awk 'NR == 2 {print $4}')"
  [[ "$free_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
  if (( free_kib < MIN_FREE_DISK_KIB )); then
    die "free disk=${free_kib} KiB is below the 60-GB guard"
  fi
}

require_tooling() {
  local tool
  for tool in uv cargo rustc rustup git curl awk find tee wc tr; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" ]] || die "$VOKRA_ROOT is not a git checkout"
  [[ -f "$PARITY_PROJECT/uv.lock" ]] || die "Parler parity uv.lock is missing"
  [[ -f "$PARITY_PROJECT/dump_reference.py" ]] || die "Parler reference dumper is missing"
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
      'import platform, torch, transformers, parler_tts; print(f"python={platform.python_version()}"); print(f"torch={torch.__version__}"); print(f"transformers={transformers.__version__}"); print(f"parler_tts={parler_tts.__version__}"); print(f"torch_cpu_capability={torch.backends.cpu.get_cpu_capability()}")'
  } | tee "$output"
}

run_self_test() {
  local tmp payload actual script_path cases=0 fail=0
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp'" EXIT
  payload="$tmp/payload"
  printf 'vokra-parler-self-test\n' > "$payload"
  actual="$(sha256_file "$payload")"

  cases=$((cases + 1))
  verify_file "$payload" "$(wc -c < "$payload" | tr -d '[:space:]')" "$actual" \
    >/dev/null 2>&1 || { log "self-test FAIL: valid identity rejected"; fail=1; }
  cases=$((cases + 1))
  if verify_file "$payload" 1 "$actual" >/dev/null 2>&1; then
    log "self-test FAIL: invalid byte size accepted"; fail=1
  fi
  cases=$((cases + 1))
  if verify_file "$payload" "$(wc -c < "$payload" | tr -d '[:space:]')" \
    "$(printf '%064d' 0)" >/dev/null 2>&1; then
    log "self-test FAIL: invalid SHA-256 accepted"; fail=1
  fi
  cases=$((cases + 1))
  script_path="${BASH_SOURCE[0]}"
  for required in "$ENGLISH_PUBLIC_REVISION" "$MULTILINGUAL_PUBLIC_REVISION" \
    "$ENGLISH_UPSTREAM_REVISION" "$MULTILINGUAL_UPSTREAM_REVISION" \
    "$PARLER_SOURCE_REVISION" "$ENGLISH_PUBLIC_SHA256" "$MULTILINGUAL_PUBLIC_SHA256" \
    "$ENGLISH_CHECKPOINT_SHA256" "$MULTILINGUAL_CHECKPOINT_SHA256" \
    "parler_tts/dump_reference.py" "parity_parler_tts_real" \
    "load_session_routes_only_named_parler_releases_to_tts" \
    "aarch64-apple-darwin" "--test-threads=1" "--frozen --python 3.12"; do
    if ! grep -Fq -- "$required" "$script_path"; then
      log "self-test FAIL: worker contract lost token: $required"; fail=1
    fi
  done
  cases=$((cases + 1))
  if grep -En '^[[:space:]]*(python3|python|pip)([[:space:]]|$)' "$script_path" >/dev/null; then
    log "self-test FAIL: direct Python/pip command found"; fail=1
  fi
  cases=$((cases + 1))
  if grep -En -- '^[[:space:]]*(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh)([[:space:]]|$)' "$script_path" >/dev/null; then
    log "self-test FAIL: publishing command found"; fail=1
  fi

  rm -rf "$tmp"
  trap - EXIT
  if [[ $fail -eq 0 ]]; then
    echo "run-parler-tts-validation.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

download_variant() {
  local public_repo="$1" public_revision="$2" public_file="$3" public_output="$4"
  local upstream_repo="$5" upstream_revision="$6" upstream_dir="$7"
  download_hf_file "$public_repo" "$public_revision" "$public_file" "$public_output"
  download_hf_file "$upstream_repo" "$upstream_revision" model.safetensors "$upstream_dir/model.safetensors"
  download_hf_file "$upstream_repo" "$upstream_revision" config.json "$upstream_dir/config.json"
  download_hf_file "$upstream_repo" "$upstream_revision" generation_config.json "$upstream_dir/generation_config.json"
}

main() {
  local self_test=0 requested_work_dir="" run_stamp work_dir inputs_dir logs_dir reference_dir
  local english_public english_upstream multilingual_public multilingual_upstream
  local run_log env_log compile_log apple_log cli_log cpu_log summary_file
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --work-dir)
        [[ $# -ge 2 && -n "$2" ]] || { die "--work-dir requires a directory"; return 2; }
        requested_work_dir="$2"; shift 2 ;;
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
  work_dir="${requested_work_dir:-$VOKRA_SCRATCH/parler-validation/$run_stamp}"
  if [[ -e "$work_dir" ]] && [[ -n "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    die "--work-dir must be absent or empty: $work_dir"
  fi
  inputs_dir="$work_dir/inputs"
  logs_dir="$work_dir/logs"
  reference_dir="$work_dir/reference"
  english_public="$inputs_dir/public-english/model.gguf"
  english_upstream="$inputs_dir/upstream-english"
  multilingual_public="$inputs_dir/public-multilingual/model.gguf"
  multilingual_upstream="$inputs_dir/upstream-multilingual"
  mkdir -p "$logs_dir" "$english_upstream" "$multilingual_upstream" \
    "$(dirname "$english_public")" "$(dirname "$multilingual_public")" "$reference_dir"
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

  step "Sync locked Python 3.12 official-reference environment"
  uv sync --project "$PARITY_PROJECT" --frozen --python 3.12

  step "Download exact public and upstream English Mini inputs"
  download_variant "$ENGLISH_PUBLIC_REPO" "$ENGLISH_PUBLIC_REVISION" \
    "$ENGLISH_PUBLIC_FILE" "$english_public" "$ENGLISH_UPSTREAM_REPO" \
    "$ENGLISH_UPSTREAM_REVISION" "$english_upstream"
  verify_file "$english_public" "$ENGLISH_PUBLIC_BYTES" "$ENGLISH_PUBLIC_SHA256"
  verify_file "$english_upstream/model.safetensors" "$ENGLISH_CHECKPOINT_BYTES" "$ENGLISH_CHECKPOINT_SHA256"
  verify_file "$english_upstream/config.json" "$ENGLISH_CONFIG_BYTES" "$ENGLISH_CONFIG_SHA256"
  verify_file "$english_upstream/generation_config.json" "$ENGLISH_GENERATION_BYTES" "$ENGLISH_GENERATION_SHA256"

  step "Download exact public and upstream Multilingual Mini inputs"
  download_variant "$MULTILINGUAL_PUBLIC_REPO" "$MULTILINGUAL_PUBLIC_REVISION" \
    "$MULTILINGUAL_PUBLIC_FILE" "$multilingual_public" "$MULTILINGUAL_UPSTREAM_REPO" \
    "$MULTILINGUAL_UPSTREAM_REVISION" "$multilingual_upstream"
  verify_file "$multilingual_public" "$MULTILINGUAL_PUBLIC_BYTES" "$MULTILINGUAL_PUBLIC_SHA256"
  verify_file "$multilingual_upstream/model.safetensors" "$MULTILINGUAL_CHECKPOINT_BYTES" "$MULTILINGUAL_CHECKPOINT_SHA256"
  verify_file "$multilingual_upstream/config.json" "$MULTILINGUAL_CONFIG_BYTES" "$MULTILINGUAL_CONFIG_SHA256"
  verify_file "$multilingual_upstream/generation_config.json" "$MULTILINGUAL_GENERATION_BYTES" "$MULTILINGUAL_GENERATION_SHA256"

  step "Record environment before numerical output"
  record_environment "$env_log"

  step "Generate independent official English Mini reference"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python \
    "$PARITY_PROJECT/dump_reference.py" --variant english \
    --model-dir "$english_upstream" --output "$reference_dir/english"
  cp "$reference_dir/english/manifest.json" "$logs_dir/reference-english-manifest.json"

  step "Generate independent official Multilingual Mini reference"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python \
    "$PARITY_PROJECT/dump_reference.py" --variant multilingual \
    --model-dir "$multilingual_upstream" --output "$reference_dir/multilingual"
  cp "$reference_dir/multilingual/manifest.json" "$logs_dir/reference-multilingual-manifest.json"

  step "Compile all workspace targets on VAST"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    --workspace --no-run 2>&1 | tee "$compile_log"

  step "Cross-check the Apple Metal target compiles"
  rustup target add aarch64-apple-darwin
  cargo check --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-models --features metal --target aarch64-apple-darwin \
    2>&1 | tee "$apple_log"

  step "Verify the Parler CLI dispatch route"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-cli engine::tests::load_session_routes_only_named_parler_releases_to_tts \
    -- --exact --nocapture 2>&1 | tee "$cli_log"

  step "Compare native CPU T5/generation/codec with both official references"
  VOKRA_PARLER_ENGLISH_GGUF="$english_public" \
  VOKRA_PARLER_ENGLISH_PARITY_DIR="$reference_dir/english" \
  VOKRA_PARLER_MULTILINGUAL_GGUF="$multilingual_public" \
  VOKRA_PARLER_MULTILINGUAL_PARITY_DIR="$reference_dir/multilingual" \
  VOKRA_PARLER_BACKEND=cpu \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --test parity_parler_tts_real -- --nocapture --test-threads=1 \
      2>&1 | tee "$cpu_log"
  grep -F "Parler ENGLISH Cpu:" "$cpu_log" >/dev/null \
    || die "Parler English CPU parity measurement sentinel missing"
  grep -F "Parler MULTILINGUAL Cpu:" "$cpu_log" >/dev/null \
    || die "Parler Multilingual CPU parity measurement sentinel missing"

  {
    echo "execution_status=PASS"
    echo "text_hidden_verdict=FP32_ATOL_0.01_PASS"
    echo "generated_codes=exact"
    echo "pcm_verdict=FP32_ATOL_0.01_PASS"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "english_public_revision=$ENGLISH_PUBLIC_REVISION"
    echo "english_public_sha256=$ENGLISH_PUBLIC_SHA256"
    echo "english_upstream_revision=$ENGLISH_UPSTREAM_REVISION"
    echo "english_checkpoint_sha256=$ENGLISH_CHECKPOINT_SHA256"
    echo "multilingual_public_revision=$MULTILINGUAL_PUBLIC_REVISION"
    echo "multilingual_public_sha256=$MULTILINGUAL_PUBLIC_SHA256"
    echo "multilingual_upstream_revision=$MULTILINGUAL_UPSTREAM_REVISION"
    echo "multilingual_checkpoint_sha256=$MULTILINGUAL_CHECKPOINT_SHA256"
    echo "parler_source_revision=$PARLER_SOURCE_REVISION"
    echo "english_reference_manifest_sha256=$(sha256_file "$reference_dir/english/manifest.json")"
    echo "multilingual_reference_manifest_sha256=$(sha256_file "$reference_dir/multilingual/manifest.json")"
    echo "metal_runtime=REQUIRES_REMOTE_APPLE_SILICON"
  } | tee "$summary_file"
  trap - EXIT
  log "PASS: pull $logs_dir and $reference_dir, then destroy the VAST instance"
}

main "$@"
