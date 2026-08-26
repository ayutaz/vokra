#!/usr/bin/env bash
# Real-weight Qwen3-ASR CPU/Metal parity on a disposable remote Apple host.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"

MIN_MEMORY_BYTES=32000000000
MIN_FREE_DISK_KIB=20000000

log() { printf '[qwen3-asr-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: apple-silicon-qwen3-asr.sh \
  --gguf-0.6b <path> --reference-0.6b <dir> \
  --gguf-1.7b <path> --reference-1.7b <dir> \
  --evidence-dir <empty-dir>
       apple-silicon-qwen3-asr.sh --self-test

Runs the exact-token and projected-audio CPU/Metal parity test for both pinned
Qwen3-ASR releases. It refuses the 16-GB maintainer class of machine and also
requires VOKRA_REMOTE_APPLE_SILICON=1, Darwin arm64, a clean checkout, at
least 32 GB physical memory, and all four real inputs before Cargo starts.

This script does not download, upload, convert, publish, or delete a model.
Transfer the VAST-produced GGUFs directly to a disposable remote Apple host;
do not route them through the maintainer Mac. Pull only the evidence directory
after the run, then delete the staged model data or destroy the remote worker.
EOF
}

sha256_file() {
  shasum -a 256 "$1" | awk '{print $1}'
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

require_reference() {
  local label="$1" directory="$2"
  [[ -d "$directory" ]] || die "$label is not a directory: $directory"
  require_file "$label manifest" "$directory/manifest.txt"
  require_file "$label PCM" "$directory/pcm.f32le"
  require_file "$label projected audio" "$directory/audio_embeddings.f32le"
  require_file "$label prompt ids" "$directory/prompt_ids.u32le"
  require_file "$label generated ids" "$directory/generated_ids.u32le"
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
    die "physical memory $memory_bytes bytes is below the 32-GB remote-worker guard"
  fi
  free_disk_kib="$(df -Pk "$VOKRA_ROOT" | awk 'NR == 2 {print $4}')"
  [[ "$free_disk_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
  if (( free_disk_kib < MIN_FREE_DISK_KIB )); then
    die "free disk $free_disk_kib KiB is below the 20-GB run guard"
  fi
}

require_tooling() {
  local tool
  for tool in cargo rustc git shasum awk find tee grep sysctl sw_vers system_profiler xcrun; do
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
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-qwen3-asr-apple.XXXXXX")"
  trap 'rm -rf "$temporary"' EXIT
  printf 'abc' > "$temporary/value"
  [[ "$(sha256_file "$temporary/value")" == \
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" ]] \
    || die "SHA-256 helper self-test failed"
  require_empty_directory "$temporary/evidence"
  [[ -d "$temporary/evidence" ]] || die "directory helper self-test failed"
  log "self-test PASS"
)

main() {
  local gguf_06='' reference_06='' gguf_17='' reference_17=''
  local evidence_dir='' self_test=0
  while (( $# > 0 )); do
    case "$1" in
      --gguf-0.6b)
        [[ $# -ge 2 ]] || { usage; return 2; }
        gguf_06="$2"
        shift 2
        ;;
      --reference-0.6b)
        [[ $# -ge 2 ]] || { usage; return 2; }
        reference_06="$2"
        shift 2
        ;;
      --gguf-1.7b)
        [[ $# -ge 2 ]] || { usage; return 2; }
        gguf_17="$2"
        shift 2
        ;;
      --reference-1.7b)
        [[ $# -ge 2 ]] || { usage; return 2; }
        reference_17="$2"
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
    [[ -z "$gguf_06$reference_06$gguf_17$reference_17$evidence_dir" ]] \
      || die "--self-test accepts no other arguments"
    run_self_test
    return
  fi
  [[ -n "$gguf_06" && -n "$reference_06" && -n "$gguf_17" && \
    -n "$reference_17" && -n "$evidence_dir" ]] \
    || { usage; die "all two-variant model/reference arguments and --evidence-dir are required"; }

  require_remote_apple_host
  require_tooling
  require_file "Qwen3-ASR 0.6B GGUF" "$gguf_06"
  require_file "Qwen3-ASR 1.7B GGUF" "$gguf_17"
  require_reference "Qwen3-ASR 0.6B reference" "$reference_06"
  require_reference "Qwen3-ASR 1.7B reference" "$reference_17"
  require_empty_directory "$evidence_dir"
  record_environment "$evidence_dir/environment.txt"

  {
    echo "gguf_0_6b_sha256=$(sha256_file "$gguf_06")"
    echo "gguf_1_7b_sha256=$(sha256_file "$gguf_17")"
    echo "reference_0_6b_manifest_sha256=$(sha256_file "$reference_06/manifest.txt")"
    echo "reference_1_7b_manifest_sha256=$(sha256_file "$reference_17/manifest.txt")"
  } > "$evidence_dir/input-hashes.txt"

  log "running both real-weight CPU/Metal parity cases on remote Apple Silicon"
  env \
    VOKRA_QWEN3_ASR_0_6B_GGUF="$gguf_06" \
    VOKRA_QWEN3_ASR_0_6B_REFERENCE_DIR="$reference_06" \
    VOKRA_QWEN3_ASR_1_7B_GGUF="$gguf_17" \
    VOKRA_QWEN3_ASR_1_7B_REFERENCE_DIR="$reference_17" \
    CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}" \
    RUST_TEST_THREADS=1 \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --features metal --test qwen3_asr_real \
      qwen3_asr_real_metal_matches_cpu_exact_greedy -- --exact --nocapture \
      2>&1 | tee "$evidence_dir/parity.log"

  local marker_06 marker_17
  marker_06='QWEN3_ASR_PARITY qwen3-asr-0.6b Metal_vs_CPU token_ids=exact text=exact PASS'
  marker_17='QWEN3_ASR_PARITY qwen3-asr-1.7b Metal_vs_CPU token_ids=exact text=exact PASS'
  grep -F "$marker_06" "$evidence_dir/parity.log" >/dev/null \
    || die "0.6B Metal PASS marker is absent"
  grep -F "$marker_17" "$evidence_dir/parity.log" >/dev/null \
    || die "1.7B Metal PASS marker is absent"

  {
    echo "verdict=PASS"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "qwen3_asr_0_6b_cpu_vs_metal=PASS"
    echo "qwen3_asr_1_7b_cpu_vs_metal=PASS"
    echo "projected_audio_atol=0.01"
    echo "greedy_ids=exact"
    echo "language=exact"
    echo "text=exact"
    echo "upload=NOT_PERFORMED"
  } > "$evidence_dir/summary.txt"
  log "PASS: pull only $evidence_dir, then remove the staged GGUF/reference data or destroy the remote worker"
}

main "$@"
