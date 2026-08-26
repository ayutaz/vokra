#!/usr/bin/env bash
# Real-weight Ultravox CPU/Metal parity on a disposable remote Apple host.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"

PUBLIC_BYTES="1366275264"
PUBLIC_SHA256="376c79a7219bb38fc6a857b0bd9ccf57daff878e7bb4723c4801000c0d7b8c9c"
MIN_MEMORY_BYTES=24000000000
MIN_FREE_DISK_KIB=12000000

log() { printf '[ultravox-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: apple-silicon-ultravox.sh \
  --gguf <ultravox.gguf> \
  --companion <ultravox-llama-companion.gguf> \
  --companion-sha256 <VAST-recorded-sha256> \
  --reference <VAST-reference-dir> \
  --evidence-dir <empty-dir>
       apple-silicon-ultravox.sh --self-test

Runs the fixed Ultravox pair first on Apple CPU and then Metal against the same
VAST-produced official reference. It refuses the maintainer's 16-GB machine
class and requires VOKRA_REMOTE_APPLE_SILICON=1, Darwin arm64, a clean checkout,
at least 24 GB physical memory, and authenticated inputs before Cargo starts.

This script does not download, upload, convert, publish, or delete a model.
Transfer model data directly from VAST to a disposable Apple host. Pull only
the evidence directory, then delete staged data or destroy the worker.
EOF
}

sha256_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

file_bytes() {
  wc -c < "$1" | tr -d '[:space:]'
}

require_empty_directory() {
  local directory="$1"
  if [[ -e "$directory" ]]; then
    [[ -d "$directory" ]] || die "evidence path is not a directory: $directory"
    [[ -z "$(find "$directory" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
      || die "evidence directory must be empty: $directory"
  else
    mkdir -p "$directory"
  fi
}

require_file() {
  local label="$1" path="$2"
  [[ -f "$path" && -s "$path" ]] || die "$label is missing or empty: $path"
}

require_identity() {
  local label="$1" path="$2" expected_bytes="$3" expected_sha="$4"
  require_file "$label" "$path"
  local actual_bytes actual_sha
  actual_bytes="$(file_bytes "$path")"
  actual_sha="$(sha256_file "$path")"
  [[ "$actual_bytes" == "$expected_bytes" ]] \
    || die "$label bytes=$actual_bytes, expected $expected_bytes"
  [[ "$actual_sha" == "$expected_sha" ]] \
    || die "$label SHA-256=$actual_sha, expected $expected_sha"
}

require_reference() {
  local directory="$1"
  [[ -d "$directory" ]] || die "reference is not a directory: $directory"
  local name
  for name in manifest.txt pcm.f32le input_features.f32le audio_embeddings.f32le \
    prompt_ids.u32le next_logits.f32le generated_ids.u32le environment.json source_files.json; do
    require_file "reference $name" "$directory/$name"
  done
}

require_remote_apple_host() {
  local memory_bytes free_disk_kib
  [[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == "1" ]] \
    || die "VOKRA_REMOTE_APPLE_SILICON=1 is absent; refusing possible maintainer-Mac execution"
  [[ "$(uname -s)" == "Darwin" ]] || die "remote Metal parity requires Darwin"
  [[ "$(uname -m)" == "arm64" ]] || die "remote Metal parity requires Apple arm64"
  memory_bytes="$(sysctl -n hw.memsize)"
  [[ "$memory_bytes" =~ ^[0-9]+$ ]] || die "could not read hw.memsize"
  if (( memory_bytes < MIN_MEMORY_BYTES )); then
    die "physical memory $memory_bytes bytes is below the 24-GB remote-worker guard"
  fi
  free_disk_kib="$(df -Pk "$VOKRA_ROOT" | awk 'NR == 2 {print $4}')"
  [[ "$free_disk_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
  if (( free_disk_kib < MIN_FREE_DISK_KIB )); then
    die "free disk $free_disk_kib KiB is below the 12-GB run guard"
  fi
}

require_tooling() {
  local tool
  for tool in cargo rustc git shasum awk find tee grep sysctl sw_vers system_profiler xcrun wc tr; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" ]] || die "$VOKRA_ROOT is not a git checkout"
  if [[ -n "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]]; then
    die "remote Apple checkout must be clean so evidence names one exact commit"
  fi
  xcrun -f metal >/dev/null 2>&1 || die "Xcode Metal compiler is unavailable"
}

record_environment() {
  local output="$1"
  {
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "git_branch=$(git -C "$VOKRA_ROOT" branch --show-current)"
    echo "uname=$(uname -a)"
    echo "machine=$(uname -m)"
    echo "memory_bytes=$(sysctl -n hw.memsize)"
    echo "physical_cpu=$(sysctl -n hw.physicalcpu)"
    echo "logical_cpu=$(sysctl -n hw.logicalcpu)"
    sw_vers
    rustc --version --verbose
    cargo --version
    echo "metal_compiler=$(xcrun -f metal)"
    system_profiler SPDisplaysDataType
  } > "$output"
}

run_self_test() (
  local temporary
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-ultravox-apple.XXXXXX")"
  trap 'rm -rf "$temporary"' EXIT
  printf 'abc' > "$temporary/value"
  [[ "$(sha256_file "$temporary/value")" == \
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" ]] \
    || die "SHA-256 helper self-test failed"
  require_empty_directory "$temporary/evidence"
  [[ -d "$temporary/evidence" ]] || die "directory helper self-test failed"
  log "self-test PASS"
)

run_parity() {
  local backend="$1" gguf="$2" companion="$3" reference="$4" log_path="$5"
  env \
    VOKRA_ULTRAVOX_GGUF="$gguf" \
    VOKRA_ULTRAVOX_COMPANION_GGUF="$companion" \
    VOKRA_ULTRAVOX_REFERENCE_DIR="$reference" \
    VOKRA_ULTRAVOX_BACKEND="$backend" \
    CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}" \
    RUST_TEST_THREADS=1 \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --features metal --test ultravox_real \
      ultravox_public_cpu_or_metal_matches_official_reference -- --exact --nocapture \
      2>&1 | tee "$log_path"
}

main() {
  local gguf='' companion='' companion_sha='' reference='' evidence_dir='' self_test=0
  while (( $# > 0 )); do
    case "$1" in
      --gguf)
        [[ $# -ge 2 ]] || { usage; return 2; }
        gguf="$2"
        shift 2
        ;;
      --companion)
        [[ $# -ge 2 ]] || { usage; return 2; }
        companion="$2"
        shift 2
        ;;
      --companion-sha256)
        [[ $# -ge 2 ]] || { usage; return 2; }
        companion_sha="$2"
        shift 2
        ;;
      --reference)
        [[ $# -ge 2 ]] || { usage; return 2; }
        reference="$2"
        shift 2
        ;;
      --evidence-dir)
        [[ $# -ge 2 ]] || { usage; return 2; }
        evidence_dir="$2"
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
    [[ -z "$gguf$companion$companion_sha$reference$evidence_dir" ]] \
      || die "--self-test accepts no other arguments"
    run_self_test
    return
  fi
  [[ -n "$gguf" && -n "$companion" && -n "$companion_sha" && -n "$reference" && -n "$evidence_dir" ]] \
    || { usage; die "all five input options are required"; }
  [[ "$companion_sha" =~ ^[0-9a-f]{64}$ ]] \
    || die "--companion-sha256 must be 64 lowercase hex characters"

  require_remote_apple_host
  require_tooling
  require_identity "Ultravox GGUF" "$gguf" "$PUBLIC_BYTES" "$PUBLIC_SHA256"
  require_file "Ultravox Llama companion GGUF" "$companion"
  [[ "$(sha256_file "$companion")" == "$companion_sha" ]] \
    || die "companion GGUF SHA-256 differs from the VAST-recorded identity"
  require_reference "$reference"
  require_empty_directory "$evidence_dir"
  record_environment "$evidence_dir/environment.txt"

  {
    echo "gguf_sha256=$(sha256_file "$gguf")"
    echo "companion_sha256=$(sha256_file "$companion")"
    echo "reference_manifest_sha256=$(sha256_file "$reference/manifest.txt")"
  } > "$evidence_dir/input-hashes.txt"

  log "running real-weight Apple CPU parity against official reference"
  run_parity cpu "$gguf" "$companion" "$reference" "$evidence_dir/parity-cpu.log"
  grep -F 'ULTRAVOX_PARITY Cpu_vs_official frontend_atol=0.01 audio_embeddings_atol=0.01 logits_atol=0.01 greedy_ids=exact PASS' \
    "$evidence_dir/parity-cpu.log" >/dev/null \
    || die "CPU PASS marker is absent"

  log "running real-weight Metal parity against official reference"
  run_parity metal "$gguf" "$companion" "$reference" "$evidence_dir/parity-metal.log"
  grep -F 'ULTRAVOX_PARITY Metal_vs_official frontend_atol=0.01 audio_embeddings_atol=0.01 logits_atol=0.01 greedy_ids=exact PASS' \
    "$evidence_dir/parity-metal.log" >/dev/null \
    || die "Metal PASS marker is absent"

  {
    echo "verdict=PASS"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "cpu_vs_official_frontend_atol=0.01"
    echo "cpu_vs_official_audio_embeddings_atol=0.01"
    echo "cpu_vs_official_logits_atol=0.01"
    echo "metal_vs_official_frontend_atol=0.01"
    echo "metal_vs_official_audio_embeddings_atol=0.01"
    echo "metal_vs_official_logits_atol=0.01"
    echo "cpu_greedy_ids=exact"
    echo "metal_greedy_ids=exact"
    echo "upload=NOT_PERFORMED"
  } > "$evidence_dir/summary.txt"
  log "PASS: pull only $evidence_dir, then delete staged model/reference data or destroy the worker"
}

main "$@"
