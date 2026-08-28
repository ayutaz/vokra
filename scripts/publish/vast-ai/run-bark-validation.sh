#!/usr/bin/env bash
# Reproduce Bark Small/Full CPU parity on VAST. No upload path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/bark"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

TRANSFORMERS_REVISION="e42587f596181396e1c4b63660abf0c736b10dae"
GENERATION_CONFIG_BYTES=4908
GENERATION_CONFIG_SHA256="ab2969fcd40e085bc924ad99ad419c27f62f5acb61afac5de7490ab0c796b5b9"

SMALL_PUBLIC_REPO="vokra/bark-small"
SMALL_PUBLIC_REVISION="09802c56a2b2e8ad87835115b94b38031fde29b6"
SMALL_PUBLIC_BYTES=1674074848
SMALL_PUBLIC_SHA256="43b781a0dcd66f1e7451005e461ec20e2141bc9c4f529feb4a9a8c0e352ea137"
SMALL_UPSTREAM_REPO="suno/bark-small"
SMALL_UPSTREAM_REVISION="1dbd7a128513b8ae4a4e2130fed57b7ac9da5bcd"
SMALL_CHECKPOINT_BYTES=1676663913
SMALL_CHECKPOINT_SHA256="f0f7f16b24f65789ce42b3c491aa6a1cdf219f7ef425066fcd194485245e65d9"
SMALL_CONFIG_BYTES=8803
SMALL_CONFIG_SHA256="9d95e9c3027cd79cf5f762cc03a69b6393cea87c51e9dd6b998fde3a7f01510e"

FULL_PUBLIC_REPO="vokra/bark"
FULL_PUBLIC_REVISION="f304ddcdfd9218994731ec3b09e89b9961b8b751"
FULL_PUBLIC_BYTES=4466390272
FULL_PUBLIC_SHA256="fd628312ce7d8e1cbc41718741614116d5c7f08d0763f81622edbac320b208ec"
FULL_UPSTREAM_REPO="suno/bark"
FULL_UPSTREAM_REVISION="70a8a7d34168586dc5d028fa9666aceade177992"
FULL_CHECKPOINT_BYTES=4486643861
FULL_CHECKPOINT_SHA256="4e3d407b9b3b619da184c85786c88e5e35f90f9089303e16db696ed0be477989"
FULL_CONFIG_BYTES=8806
FULL_CONFIG_SHA256="48be144c0232acd8c55786d1eea9161ae6c973f21ec4a2f02627c844065ea695"

MIN_VAST_MEM_KIB=23000000
MIN_FREE_DISK_KIB=40000000

log() { printf '[bark-vast] %s\n' "$*" >&2; }
step() { printf '\n[bark-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-bark-validation.sh [--work-dir <empty-dir>]
       run-bark-validation.sh --self-test

VAST-only, non-publishing Bark Small/Full validation. The worker downloads and
verifies both exact public Vokra GGUFs and exact immutable Suno checkpoints,
uses locked official Transformers 4.31.0 for independent greedy references,
compiles the workspace plus Apple target, verifies CLI routing, and compares
native CPU generated codes plus embedded-codec PCM.

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
    || die "real Bark model work is VAST/Linux-only; refusing $(uname -s)"
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$mem_kib" =~ ^[0-9]+$ ]] || die "could not read MemTotal"
  if (( mem_kib < MIN_VAST_MEM_KIB )); then
    die "MemTotal=${mem_kib} KiB is below the VAST 24-GB guard"
  fi
  mkdir -p "$VOKRA_SCRATCH"
  free_kib="$(df -Pk "$VOKRA_SCRATCH" | awk 'NR == 2 {print $4}')"
  [[ "$free_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
  if (( free_kib < MIN_FREE_DISK_KIB )); then
    die "free disk=${free_kib} KiB is below the 40-GB guard"
  fi
}

require_tooling() {
  local tool
  for tool in uv cargo rustc rustup git curl awk find tee wc tr; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" ]] || die "$VOKRA_ROOT is not a git checkout"
  [[ -f "$PARITY_PROJECT/uv.lock" ]] || die "Bark parity uv.lock is missing"
  [[ -f "$PARITY_PROJECT/dump_reference.py" ]] || die "Bark reference dumper is missing"
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
      'import platform, torch, transformers; print(f"python={platform.python_version()}"); print(f"torch={torch.__version__}"); print(f"transformers={transformers.__version__}"); print(f"torch_cpu_capability={torch.backends.cpu.get_cpu_capability()}")'
  } | tee "$output"
}

run_self_test() {
  local tmp payload actual script_path cases=0 fail=0
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp'" EXIT
  payload="$tmp/payload"
  printf 'vokra-bark-self-test\n' > "$payload"
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
  for required in "$SMALL_PUBLIC_REVISION" "$FULL_PUBLIC_REVISION" \
    "$SMALL_UPSTREAM_REVISION" "$FULL_UPSTREAM_REVISION" \
    "$TRANSFORMERS_REVISION" "$SMALL_PUBLIC_SHA256" "$FULL_PUBLIC_SHA256" \
    "$SMALL_CHECKPOINT_SHA256" "$FULL_CHECKPOINT_SHA256" \
    "bark/dump_reference.py" "parity_bark_real" \
    "load_session_routes_only_named_bark_releases_to_tts" \
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
    echo "run-bark-validation.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

main() {
  local self_test=0 requested_work_dir="" run_stamp work_dir inputs_dir logs_dir reference_dir
  local small_public small_upstream full_public full_upstream
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
  work_dir="${requested_work_dir:-$VOKRA_SCRATCH/bark-validation/$run_stamp}"
  if [[ -e "$work_dir" ]] && [[ -n "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    die "--work-dir must be absent or empty: $work_dir"
  fi
  inputs_dir="$work_dir/inputs"
  logs_dir="$work_dir/logs"
  reference_dir="$work_dir/reference"
  small_public="$inputs_dir/public-small/model.gguf"
  small_upstream="$inputs_dir/upstream-small"
  full_public="$inputs_dir/public-full/model.gguf"
  full_upstream="$inputs_dir/upstream-full"
  mkdir -p "$logs_dir" "$small_upstream" "$full_upstream" \
    "$(dirname "$small_public")" "$(dirname "$full_public")" "$reference_dir"
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

  step "Download exact public and upstream Bark Small inputs"
  download_hf_file "$SMALL_PUBLIC_REPO" "$SMALL_PUBLIC_REVISION" model.gguf "$small_public"
  download_hf_file "$SMALL_UPSTREAM_REPO" "$SMALL_UPSTREAM_REVISION" pytorch_model.bin "$small_upstream/pytorch_model.bin"
  download_hf_file "$SMALL_UPSTREAM_REPO" "$SMALL_UPSTREAM_REVISION" config.json "$small_upstream/config.json"
  download_hf_file "$SMALL_UPSTREAM_REPO" "$SMALL_UPSTREAM_REVISION" generation_config.json "$small_upstream/generation_config.json"
  verify_file "$small_public" "$SMALL_PUBLIC_BYTES" "$SMALL_PUBLIC_SHA256"
  verify_file "$small_upstream/pytorch_model.bin" "$SMALL_CHECKPOINT_BYTES" "$SMALL_CHECKPOINT_SHA256"
  verify_file "$small_upstream/config.json" "$SMALL_CONFIG_BYTES" "$SMALL_CONFIG_SHA256"
  verify_file "$small_upstream/generation_config.json" "$GENERATION_CONFIG_BYTES" "$GENERATION_CONFIG_SHA256"

  step "Download exact public and upstream Bark Full inputs"
  download_hf_file "$FULL_PUBLIC_REPO" "$FULL_PUBLIC_REVISION" model.gguf "$full_public"
  download_hf_file "$FULL_UPSTREAM_REPO" "$FULL_UPSTREAM_REVISION" pytorch_model.bin "$full_upstream/pytorch_model.bin"
  download_hf_file "$FULL_UPSTREAM_REPO" "$FULL_UPSTREAM_REVISION" config.json "$full_upstream/config.json"
  download_hf_file "$FULL_UPSTREAM_REPO" "$FULL_UPSTREAM_REVISION" generation_config.json "$full_upstream/generation_config.json"
  verify_file "$full_public" "$FULL_PUBLIC_BYTES" "$FULL_PUBLIC_SHA256"
  verify_file "$full_upstream/pytorch_model.bin" "$FULL_CHECKPOINT_BYTES" "$FULL_CHECKPOINT_SHA256"
  verify_file "$full_upstream/config.json" "$FULL_CONFIG_BYTES" "$FULL_CONFIG_SHA256"
  verify_file "$full_upstream/generation_config.json" "$GENERATION_CONFIG_BYTES" "$GENERATION_CONFIG_SHA256"

  step "Record environment before numerical output"
  record_environment "$env_log"

  step "Generate independent official Bark Small reference"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python \
    "$PARITY_PROJECT/dump_reference.py" --variant small \
    --model-dir "$small_upstream" --output "$reference_dir/small"
  cp "$reference_dir/small/manifest.json" "$logs_dir/reference-small-manifest.json"

  step "Generate independent official Bark Full reference"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python \
    "$PARITY_PROJECT/dump_reference.py" --variant full \
    --model-dir "$full_upstream" --output "$reference_dir/full"
  cp "$reference_dir/full/manifest.json" "$logs_dir/reference-full-manifest.json"

  step "Compile all workspace targets on VAST"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    --workspace --no-run 2>&1 | tee "$compile_log"

  step "Cross-check the Apple Metal target compiles"
  rustup target add aarch64-apple-darwin
  cargo check --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-models --features metal --target aarch64-apple-darwin \
    2>&1 | tee "$apple_log"

  step "Verify the Bark CLI dispatch route"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-cli engine::tests::load_session_routes_only_named_bark_releases_to_tts \
    -- --exact --nocapture 2>&1 | tee "$cli_log"

  step "Compare native CPU generation/codec with both official references"
  VOKRA_BARK_SMALL_GGUF="$small_public" \
  VOKRA_BARK_SMALL_PARITY_DIR="$reference_dir/small" \
  VOKRA_BARK_FULL_GGUF="$full_public" \
  VOKRA_BARK_FULL_PARITY_DIR="$reference_dir/full" \
  VOKRA_BARK_BACKEND=cpu \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --test parity_bark_real -- --nocapture --test-threads=1 \
      2>&1 | tee "$cpu_log"
  grep -F "Bark SMALL Cpu:" "$cpu_log" >/dev/null \
    || die "Bark Small CPU parity measurement sentinel missing"
  grep -F "Bark FULL Cpu:" "$cpu_log" >/dev/null \
    || die "Bark Full CPU parity measurement sentinel missing"

  {
    echo "execution_status=PASS"
    echo "numeric_verdict=FP32_ATOL_0.01_PASS"
    echo "generated_codes=exact"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "small_public_revision=$SMALL_PUBLIC_REVISION"
    echo "small_public_sha256=$SMALL_PUBLIC_SHA256"
    echo "small_upstream_revision=$SMALL_UPSTREAM_REVISION"
    echo "small_checkpoint_sha256=$SMALL_CHECKPOINT_SHA256"
    echo "full_public_revision=$FULL_PUBLIC_REVISION"
    echo "full_public_sha256=$FULL_PUBLIC_SHA256"
    echo "full_upstream_revision=$FULL_UPSTREAM_REVISION"
    echo "full_checkpoint_sha256=$FULL_CHECKPOINT_SHA256"
    echo "transformers_source_revision=$TRANSFORMERS_REVISION"
    echo "small_reference_manifest_sha256=$(sha256_file "$reference_dir/small/manifest.json")"
    echo "full_reference_manifest_sha256=$(sha256_file "$reference_dir/full/manifest.json")"
    echo "metal_runtime=REQUIRES_REMOTE_APPLE_SILICON"
  } | tee "$summary_file"
  trap - EXIT
  log "PASS: pull $logs_dir and $reference_dir, then destroy the VAST instance"
}

main "$@"
