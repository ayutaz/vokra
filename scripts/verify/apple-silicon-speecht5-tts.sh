#!/usr/bin/env bash
# Exact public SpeechT5-TTS CPU/Metal parity on a disposable remote Apple host.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"

PUBLIC_GGUF_SHA256="f26019f5e2f7106d834b0b1fd4f66286839e000350caad169388467452c8dde0"
MIN_MEMORY_BYTES=32000000000
MIN_FREE_DISK_KIB=10000000

log() { printf '[speecht5-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: apple-silicon-speecht5-tts.sh \
  --gguf <public-speecht5.gguf> --reference <official-reference-dir> \
  --evidence-dir <empty-dir>
       apple-silicon-speecht5-tts.sh --self-test

Runs the official-reference CPU/Metal text-to-mel parity test against the
exact public vokra/speecht5-tts GGUF. It refuses the maintainer class of
machine and requires VOKRA_REMOTE_APPLE_SILICON=1, Darwin arm64, at least
32 GB physical memory, a clean checkout, and pre-staged real inputs.

This script does not download, upload, convert, publish, or delete a model.
Transfer the fixed public GGUF and VAST-produced reference directly to a
disposable remote Apple host. Pull only the evidence directory afterward,
then remove staged model data or destroy the remote worker.
EOF
}

sha256_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

require_file() {
  local label="$1" path="$2"
  [[ -f "$path" && -s "$path" ]] || die "$label is missing or empty: $path"
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

require_reference() {
  local directory="$1" filename
  [[ -d "$directory" ]] || die "reference path is not a directory: $directory"
  for filename in text.txt tokens.u32 speaker.f32 before_postnet.f32 \
    after_postnet.f32 frames.txt decoder_steps.txt reference.json; do
    require_file "SpeechT5 reference $filename" "$directory/$filename"
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
    die "physical memory $memory_bytes bytes is below the 32-GB remote-worker guard"
  fi
  free_disk_kib="$(df -Pk "$VOKRA_ROOT" | awk 'NR == 2 {print $4}')"
  [[ "$free_disk_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
  if (( free_disk_kib < MIN_FREE_DISK_KIB )); then
    die "free disk $free_disk_kib KiB is below the 10-GB run guard"
  fi
}

require_tooling() {
  local tool
  for tool in cargo rustc git shasum awk find tee grep sysctl sw_vers \
    system_profiler xcrun; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" ]] || die "$VOKRA_ROOT is not a git checkout"
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] \
    || die "remote Apple checkout must be clean so evidence names one exact commit"
  xcrun -f metal >/dev/null 2>&1 || die "Xcode Metal compiler is unavailable"
}

record_environment() {
  local output="$1"
  {
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "git_branch=$(git -C "$VOKRA_ROOT" branch --show-current)"
    echo "uname=$(uname -a)"
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
  local temporary script_path
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-speecht5-apple.XXXXXX")"
  trap 'rm -rf "$temporary"' EXIT
  printf 'abc' > "$temporary/value"
  [[ "$(sha256_file "$temporary/value")" == \
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" ]] \
    || die "SHA-256 helper self-test failed"
  require_empty_directory "$temporary/evidence"
  script_path="${BASH_SOURCE[0]}"
  grep -F "$PUBLIC_GGUF_SHA256" "$script_path" >/dev/null \
    || die "public GGUF SHA contract is missing"
  grep -F "SPEECHT5_TTS_OFFICIAL_PARITY backend=metal" "$script_path" >/dev/null \
    || die "Metal PASS marker contract is missing"
  log "self-test PASS"
)

main() {
  local gguf='' reference='' evidence_dir='' self_test=0 gguf_sha
  while (( $# > 0 )); do
    case "$1" in
      --gguf)
        [[ $# -ge 2 ]] || { usage; return 2; }
        gguf="$2"
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
    [[ -z "$gguf$reference$evidence_dir" ]] \
      || die "--self-test accepts no other arguments"
    run_self_test
    return
  fi
  [[ -n "$gguf" && -n "$reference" && -n "$evidence_dir" ]] \
    || { usage; die "--gguf, --reference and --evidence-dir are required"; }

  require_remote_apple_host
  require_tooling
  require_file "public SpeechT5 GGUF" "$gguf"
  require_reference "$reference"
  gguf_sha="$(sha256_file "$gguf")"
  [[ "$gguf_sha" == "$PUBLIC_GGUF_SHA256" ]] \
    || die "GGUF SHA-256 $gguf_sha != exact public artifact $PUBLIC_GGUF_SHA256"
  require_empty_directory "$evidence_dir"
  record_environment "$evidence_dir/environment.txt"

  {
    echo "public_gguf_sha256=$gguf_sha"
    echo "reference_manifest_sha256=$(sha256_file "$reference/reference.json")"
  } > "$evidence_dir/input-hashes.txt"

  log "running exact public SpeechT5 CPU/Metal parity on remote Apple Silicon"
  env \
    VOKRA_SPEECHT5_TTS_GGUF="$gguf" \
    VOKRA_SPEECHT5_TTS_REFERENCE_DIR="$reference" \
    CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}" \
    RUST_TEST_THREADS=1 \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --features metal --test parity_speecht5_tts_real \
      released_cpu_mel_matches_official_transformers -- --exact --nocapture \
      2>&1 | tee "$evidence_dir/parity.log"

  grep -F "SPEECHT5_TTS_OFFICIAL_PARITY backend=cpu" "$evidence_dir/parity.log" \
    | grep -F "verdict=PASS" >/dev/null \
    || die "CPU official-reference PASS marker is absent"
  grep -F "SPEECHT5_TTS_OFFICIAL_PARITY backend=metal" "$evidence_dir/parity.log" \
    | grep -F "verdict=PASS" >/dev/null \
    || die "Metal official-reference PASS marker is absent"

  {
    echo "verdict=PASS"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "public_gguf_sha256=$gguf_sha"
    echo "speecht5_cpu_vs_official=PASS"
    echo "speecht5_metal_vs_official=PASS"
    echo "speecht5_metal_vs_cpu=PASS"
    echo "bound=0.01"
    echo "upload=NOT_PERFORMED"
  } > "$evidence_dir/summary.txt"
  log "PASS: pull only $evidence_dir, then remove staged data or destroy the remote worker"
}

main "$@"
