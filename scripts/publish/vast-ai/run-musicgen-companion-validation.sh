#!/usr/bin/env bash
# Validate exact public MusicGen Medium/Large + Small companion on VAST CPU.
# This worker never uploads, publishes, pushes, stops, or destroys instances.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

SMALL_REPO="vokra/musicgen-small"
SMALL_REVISION="30e7e356c9d8326c42965a337e810162d7cdbc70"
SMALL_FILE="model.gguf"
SMALL_BYTES=2364405568
SMALL_SHA256="be0a1d823cd4b4570e39cb87ce05a707959ffdffdc0aef23eb90fffa5c084a98"

MEDIUM_REPO="vokra/musicgen-medium"
MEDIUM_REVISION="29b20532e56d3a4803ce1488e03aace0f976e5cc"
MEDIUM_FILE="musicgen-medium.gguf"
MEDIUM_BYTES=3677520768
MEDIUM_SHA256="574072a7058c4a7bd5f60b7a773e219f659a029dc35cc6a4fd167b08e62fbc1c"

LARGE_REPO="vokra/musicgen-large"
LARGE_REVISION="306a9091012eb15e8ad3e108a72dd2ea0bfd8586"
LARGE_FILE="musicgen-large.gguf"
LARGE_BYTES=6513958784
LARGE_SHA256="d015b2dbe60b1ab85d0778d98c818413f46e71c91f9dcf04b2ff1088a9bc6ca9"

MIN_VAST_MEM_KIB=60000000
MIN_FREE_DISK_KIB=80000000

log() { printf '[musicgen-companion-vast] %s\n' "$*" >&2; }
step() { printf '\n[musicgen-companion-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-musicgen-companion-validation.sh [--work-dir <empty-dir>]
       run-musicgen-companion-validation.sh --self-test

VAST-only validation of the exact public MusicGen Medium/Large LM-only GGUFs
with the exact public MusicGen Small T5/EnCodec companion. The worker verifies
all three immutable identities, strict-binds every complete manifest, runs a
three-frame CPU T5 -> target LM -> companion EnCodec route for Medium and
Large, exercises the Medium CLI route, and cross-checks Metal-feature source
compilation for aarch64-apple-darwin.

This is a finite/non-zero route smoke, not an independent AudioCraft numerical
parity result. There is no upload or publication operation. Pull the small log
directory, then destroy the disposable VAST instance rather than stopping it.
Real Metal execution remains a separate external Apple Silicon gate.
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
  curl --fail --location --retry 5 --retry-delay 2 --retry-all-errors \
    --output "$output" "$url"
}

require_vast_host() {
  local mem_kib free_kib
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] \
    || die "VOKRA_PUBLISH_ON_VAST=1 is absent; run provision.sh first"
  [[ "$(uname -s)" == "Linux" ]] \
    || die "real MusicGen model work is VAST/Linux-only; refusing $(uname -s)"
  [[ "$(uname -m)" == "x86_64" ]] \
    || die "the recorded CPU route requires Linux x86_64, got $(uname -m)"
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$mem_kib" =~ ^[0-9]+$ ]] || die "could not read MemTotal"
  if (( mem_kib < MIN_VAST_MEM_KIB )); then
    die "MemTotal=${mem_kib} KiB is below the VAST 64-GB class guard"
  fi
  mkdir -p "$VOKRA_SCRATCH"
  free_kib="$(df -Pk "$VOKRA_SCRATCH" | awk 'NR == 2 {print $4}')"
  [[ "$free_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
  if (( free_kib < MIN_FREE_DISK_KIB )); then
    die "free disk=${free_kib} KiB is below the 80-GB run guard"
  fi
}

require_tooling() {
  local tool
  for tool in cargo rustc rustup git curl awk find tee wc tr grep; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" ]] \
    || die "VOKRA_ROOT is not the repository checkout: $VOKRA_ROOT"
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] \
    || die "VAST checkout must be clean so evidence names one exact commit"
}

record_environment() {
  local output="$1" cpu_model cpu_flags
  cpu_model="$(awk -F ':' '$1 ~ /model name/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
  cpu_flags="$(awk -F ':' '$1 ~ /flags/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
  [[ -n "$cpu_model" && -n "$cpu_flags" ]] || die "CPU provenance unavailable"
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
  } | tee "$output"
}

run_self_test() {
  local tmp payload actual script_path cases=0 fail=0
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp'" EXIT
  payload="$tmp/payload"
  printf 'vokra-musicgen-companion-self-test\n' > "$payload"
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
  for required in "$SMALL_REVISION" "$SMALL_SHA256" "$MEDIUM_REVISION" \
    "$MEDIUM_SHA256" "$LARGE_REVISION" "$LARGE_SHA256" \
    "public_musicgen_lm_only_companion_generates_finite_pcm" \
    "MUSICGEN_COMPANION_ROUTE backend=cpu" "aarch64-apple-darwin" \
    "numerical_parity=NOT_CLAIMED"; do
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
    echo "run-musicgen-companion-validation.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

main() {
  local self_test=0 requested_work_dir="" run_stamp work_dir inputs_dir logs_dir outputs_dir
  local small medium large cpu_log strict_log cli_log compile_log apple_log summary_file run_log
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
  work_dir="${requested_work_dir:-$VOKRA_SCRATCH/musicgen-companion-validation/$run_stamp}"
  if [[ -e "$work_dir" ]] && [[ -n "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    die "--work-dir must be absent or empty: $work_dir"
  fi
  inputs_dir="$work_dir/inputs"
  logs_dir="$work_dir/evidence/logs"
  outputs_dir="$work_dir/outputs"
  small="$inputs_dir/musicgen-small.gguf"
  medium="$inputs_dir/musicgen-medium.gguf"
  large="$inputs_dir/musicgen-large.gguf"
  mkdir -p "$inputs_dir" "$logs_dir" "$outputs_dir"
  export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}"
  export RUST_BACKTRACE=1
  run_log="$logs_dir/run.log"
  strict_log="$logs_dir/strict-contract.log"
  cpu_log="$logs_dir/cpu-route.log"
  cli_log="$logs_dir/cli-route.log"
  compile_log="$logs_dir/compile.log"
  apple_log="$logs_dir/apple-cross-check.log"
  summary_file="$logs_dir/summary.txt"
  exec > >(tee -a "$run_log") 2>&1
  # shellcheck disable=SC2154
  trap 'rc=$?; if [[ -n "${summary_file:-}" && ! -f "$summary_file" ]]; then printf "execution_status=FAIL\nexit_code=%s\n" "$rc" > "$summary_file"; fi; exit "$rc"' EXIT

  step "Download and authenticate exact public MusicGen artifacts"
  download_hf_file "$SMALL_REPO" "$SMALL_REVISION" "$SMALL_FILE" "$small"
  verify_file "$small" "$SMALL_BYTES" "$SMALL_SHA256"
  download_hf_file "$MEDIUM_REPO" "$MEDIUM_REVISION" "$MEDIUM_FILE" "$medium"
  verify_file "$medium" "$MEDIUM_BYTES" "$MEDIUM_SHA256"
  download_hf_file "$LARGE_REPO" "$LARGE_REVISION" "$LARGE_FILE" "$large"
  verify_file "$large" "$LARGE_BYTES" "$LARGE_SHA256"

  step "Record immutable inputs and VAST CPU environment"
  record_environment "$logs_dir/environment.txt"
  {
    echo "small_revision=$SMALL_REVISION"
    echo "small_bytes=$SMALL_BYTES"
    echo "small_sha256=$SMALL_SHA256"
    echo "medium_revision=$MEDIUM_REVISION"
    echo "medium_bytes=$MEDIUM_BYTES"
    echo "medium_sha256=$MEDIUM_SHA256"
    echo "large_revision=$LARGE_REVISION"
    echo "large_bytes=$LARGE_BYTES"
    echo "large_sha256=$LARGE_SHA256"
  } > "$logs_dir/input-identities.txt"

  step "Compile focused model contract and CLI on VAST"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-models --test musicgen_public_contract --no-run \
    2>&1 | tee "$compile_log"
  cargo build --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-cli 2>&1 | tee -a "$compile_log"

  step "Cross-check Metal-feature source for Apple arm64"
  rustup target add aarch64-apple-darwin
  cargo check --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-models --features metal --target aarch64-apple-darwin \
    2>&1 | tee "$apple_log"

  step "Strict-bind all three exact public manifests"
  VOKRA_MUSICGEN_SMALL_GGUF="$small" \
  VOKRA_MUSICGEN_MEDIUM_GGUF="$medium" \
  VOKRA_MUSICGEN_LARGE_GGUF="$large" \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --test musicgen_public_contract \
      public_musicgen_and_audiogen_artifacts_match_strict_runtime_contracts \
      -- --exact --nocapture 2>&1 | tee "$strict_log"

  step "Run complete Medium and Large CPU companion routes"
  VOKRA_MUSICGEN_SMALL_GGUF="$small" \
  VOKRA_MUSICGEN_MEDIUM_GGUF="$medium" \
  VOKRA_MUSICGEN_LARGE_GGUF="$large" \
  VOKRA_MUSICGEN_ROUTE_BACKEND=cpu \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --test musicgen_public_contract \
      public_musicgen_lm_only_companion_generates_finite_pcm \
      -- --exact --ignored --nocapture --test-threads=1 \
      2>&1 | tee "$cpu_log"
  for target in musicgen-medium musicgen-large; do
    grep -F "MUSICGEN_COMPANION_ROUTE backend=cpu target=$target" "$cpu_log" \
      | grep -F "verdict=PASS" >/dev/null \
      || die "$target CPU route PASS marker is absent"
  done

  step "Exercise the real Medium CLI companion route"
  VOKRA_ALLOW_RESEARCH_LICENSE=1 \
    "$VOKRA_ROOT/target/release/vokra-cli" run \
      --model "$medium" --backend cpu --musicgen-companion "$small" \
      --token-ids 1 --music-unconditional-token-ids 0 \
      --music-frames 3 --music-seed 0 --output "$outputs_dir/medium.wav" \
      2>&1 | tee "$cli_log"
  [[ -s "$outputs_dir/medium.wav" ]] || die "Medium CLI output WAV is absent or empty"
  grep -F "musicgen: wrote 1920 samples @ 32000 Hz" "$cli_log" >/dev/null \
    || die "Medium CLI output contract is absent"

  {
    echo "execution_status=PASS"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "small_revision=$SMALL_REVISION"
    echo "small_sha256=$SMALL_SHA256"
    echo "medium_revision=$MEDIUM_REVISION"
    echo "medium_sha256=$MEDIUM_SHA256"
    echo "large_revision=$LARGE_REVISION"
    echo "large_sha256=$LARGE_SHA256"
    echo "strict_manifest_contracts=PASS"
    echo "medium_cpu_route=FINITE_NONZERO_PASS"
    echo "large_cpu_route=FINITE_NONZERO_PASS"
    echo "medium_cli_route=PASS"
    echo "medium_cli_wav_sha256=$(sha256_file "$outputs_dir/medium.wav")"
    echo "apple_arm64_metal_feature_compile=PASS"
    echo "metal_runtime=REQUIRES_EXTERNAL_APPLE_SILICON"
    echo "numerical_parity=NOT_CLAIMED"
    echo "upload=NOT_PERFORMED"
  } | tee "$summary_file"
  trap - EXIT
  log "PASS: pull $logs_dir only, then destroy the disposable VAST instance"
}

main "$@"
