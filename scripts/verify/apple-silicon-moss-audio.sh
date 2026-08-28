#!/usr/bin/env bash
# Real-weight MOSS-Audio CPU/Metal parity on a disposable Apple host.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"

MIN_MEMORY_BYTES=64000000000
MIN_FREE_DISK_KIB=50000000

log() { printf '[moss-audio-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: apple-silicon-moss-audio.sh \
  --gguf-4b <path> --reference-4b <dir> \
  --gguf-8b <path> --reference-8b <dir> \
  --evidence-dir <empty-dir>
       apple-silicon-moss-audio.sh --self-test

Runs projected-audio and exact-token CPU/Metal parity for both pinned
MOSS-Audio Instruct releases. It refuses the 16-GB maintainer class and
requires VOKRA_REMOTE_APPLE_SILICON=1, Darwin arm64, a clean checkout, at
least 64 GB physical memory, and all four real inputs before Cargo starts.

This script does not download, upload, convert, publish, or delete a model.
Transfer VAST-produced inputs directly to a disposable Apple host. Pull only
the evidence directory after the run, then remove the staged data or destroy
the remote worker.
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
  local label="$1" directory="$2" name
  [[ -d "$directory" ]] || die "$label is not a directory: $directory"
  for name in manifest.txt pcm.f32le prompt_ids.u32le primary_audio.f32le \
    deepstack_audio_0.f32le deepstack_audio_1.f32le deepstack_audio_2.f32le \
    generated_ids.u32le prompt.txt environment.json source_files.json; do
    require_file "$label $name" "$directory/$name"
  done
  # A valid greedy sequence can decode only to special tokens.
  [[ -f "$directory/result_text.txt" ]] \
    || die "$label result_text.txt is missing: $directory/result_text.txt"
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
    die "physical memory $memory_bytes bytes is below the 64-GB remote-worker guard"
  fi
  free_disk_kib="$(df -Pk "$VOKRA_ROOT" | awk 'NR == 2 {print $4}')"
  [[ "$free_disk_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
  if (( free_disk_kib < MIN_FREE_DISK_KIB )); then
    die "free disk $free_disk_kib KiB is below the 50-GB run guard"
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
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-moss-audio-apple.XXXXXX")"
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
  local gguf_4b='' reference_4b='' gguf_8b='' reference_8b=''
  local evidence_dir='' self_test=0
  while (( $# > 0 )); do
    case "$1" in
      --gguf-4b)
        [[ $# -ge 2 ]] || { usage; return 2; }
        gguf_4b="$2"
        shift 2
        ;;
      --reference-4b)
        [[ $# -ge 2 ]] || { usage; return 2; }
        reference_4b="$2"
        shift 2
        ;;
      --gguf-8b)
        [[ $# -ge 2 ]] || { usage; return 2; }
        gguf_8b="$2"
        shift 2
        ;;
      --reference-8b)
        [[ $# -ge 2 ]] || { usage; return 2; }
        reference_8b="$2"
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
    [[ -z "$gguf_4b$reference_4b$gguf_8b$reference_8b$evidence_dir" ]] \
      || die "--self-test accepts no other arguments"
    run_self_test
    return
  fi
  [[ -n "$gguf_4b" && -n "$reference_4b" && -n "$gguf_8b" && \
    -n "$reference_8b" && -n "$evidence_dir" ]] \
    || { usage; die "all two-variant model/reference arguments and --evidence-dir are required"; }

  require_remote_apple_host
  require_tooling
  require_file "MOSS-Audio 4B GGUF" "$gguf_4b"
  require_file "MOSS-Audio 8B GGUF" "$gguf_8b"
  require_reference "MOSS-Audio 4B reference" "$reference_4b"
  require_reference "MOSS-Audio 8B reference" "$reference_8b"
  require_empty_directory "$evidence_dir"
  record_environment "$evidence_dir/environment.txt"

  {
    echo "gguf_4b_sha256=$(sha256_file "$gguf_4b")"
    echo "gguf_8b_sha256=$(sha256_file "$gguf_8b")"
    echo "reference_4b_manifest_sha256=$(sha256_file "$reference_4b/manifest.txt")"
    echo "reference_8b_manifest_sha256=$(sha256_file "$reference_8b/manifest.txt")"
  } > "$evidence_dir/input-hashes.txt"

  log "running both real-weight CPU/Metal parity cases on remote Apple Silicon"
  env \
    VOKRA_MOSS_AUDIO_4B_GGUF="$gguf_4b" \
    VOKRA_MOSS_AUDIO_4B_REFERENCE_DIR="$reference_4b" \
    VOKRA_MOSS_AUDIO_8B_GGUF="$gguf_8b" \
    VOKRA_MOSS_AUDIO_8B_REFERENCE_DIR="$reference_8b" \
    CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}" \
    RUST_TEST_THREADS=1 \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --features metal --test moss_audio_real \
      moss_audio_real_metal_matches_cpu_exact_greedy -- --exact --nocapture \
      2>&1 | tee "$evidence_dir/parity.log"

  local marker_4b marker_8b
  marker_4b='MOSS_AUDIO_PARITY moss-audio-4b-instruct Metal_vs_CPU token_ids=exact text=exact PASS'
  marker_8b='MOSS_AUDIO_PARITY moss-audio-8b-instruct Metal_vs_CPU token_ids=exact text=exact PASS'
  grep -F "$marker_4b" "$evidence_dir/parity.log" >/dev/null \
    || die "4B Metal PASS marker is absent"
  grep -F "$marker_8b" "$evidence_dir/parity.log" >/dev/null \
    || die "8B Metal PASS marker is absent"

  {
    echo "verdict=PASS"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "moss_audio_4b_cpu_vs_metal=PASS"
    echo "moss_audio_8b_cpu_vs_metal=PASS"
    echo "audio_projection_atol=0.01"
    echo "greedy_ids=exact"
    echo "text=exact"
    echo "upload=NOT_PERFORMED"
  } > "$evidence_dir/summary.txt"
  log "PASS: pull only $evidence_dir, then remove staged data or destroy the remote worker"
}

main "$@"
