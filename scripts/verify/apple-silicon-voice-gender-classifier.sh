#!/usr/bin/env bash
# Real-weight voice-gender CPU/upstream and Metal-vs-CPU parity on a
# disposable Apple Silicon host. GGUF and fixtures must already be staged by
# the VAST worker; this verifier never downloads, converts, publishes, or
# manufactures a test result.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
PARITY_SOURCE="$VOKRA_ROOT/crates/vokra-models/tests/parity_voice_gender_classifier.rs"
PARITY_DUMPER="$VOKRA_ROOT/tools/parity/voice_gender_classifier_dump_reference.py"

MIN_MEMORY_BYTES=32000000000
MIN_FREE_DISK_KIB=10000000
UPSTREAM_REPOSITORY="https://github.com/JaesungHuh/voice-gender-classifier.git"
UPSTREAM_REVISION="49bcbecfd929ba5a043bde645fdff1a375eb79c7"

log() { printf '[voice-gender-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

preflight_gate() {
  die "fixed upstream checkpoint byte identity/license gate is unresolved; Apple execution is blocked"
}

usage() {
  cat <<'EOF' >&2
usage: apple-silicon-voice-gender-classifier.sh \
  --gguf <vast-corrected-voice-gender-classifier.gguf> \
  --reference-dir <vast-voice-gender-fixtures> --evidence-dir <empty-dir>
       apple-silicon-voice-gender-classifier.sh --self-test

Runs the exact official CPU parity and Metal-vs-CPU test using only VAST
outputs. It requires VOKRA_REMOTE_APPLE_SILICON=1, Darwin arm64, at least
32 GB physical memory, free disk, a clean checkout, and the Metal compiler.
EOF
}

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

require_file() {
  local label="$1" path="$2"
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] || die "$label is missing, symlinked, or empty: $path"
}

require_empty_directory() {
  local directory="$1"
  [[ ! -e "$directory" && ! -L "$directory" ]] \
    || die "evidence directory must be absent and not a symlink: $directory"
  mkdir -p "$directory"
}

require_remote_apple_host() {
  local memory_bytes free_disk_kib
  [[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == 1 ]] \
    || die "VOKRA_REMOTE_APPLE_SILICON=1 is absent; refusing maintainer-Mac execution"
  [[ "$(uname -s)" == Darwin ]] || die "voice-gender Metal parity requires Darwin"
  [[ "$(uname -m)" == arm64 ]] || die "voice-gender Metal parity requires Apple arm64"
  memory_bytes="$(sysctl -n hw.memsize)"
  [[ "$memory_bytes" =~ ^[0-9]+$ ]] || die "could not read hw.memsize"
  (( memory_bytes >= MIN_MEMORY_BYTES )) \
    || die "physical memory $memory_bytes bytes is below the exact 32-GB guard"
  free_disk_kib="$(df -Pk "$VOKRA_ROOT" | awk 'NR == 2 {print $4}')"
  [[ "$free_disk_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
  (( free_disk_kib >= MIN_FREE_DISK_KIB )) \
    || die "free disk $free_disk_kib KiB is below the exact 10-GB guard"
}

require_tooling() {
  local tool
  for tool in cargo rustc git shasum awk find tee grep sysctl sw_vers \
    system_profiler xcrun; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" ]] \
    || die "$VOKRA_ROOT is not a Vokra git checkout"
  [[ -f "$PARITY_SOURCE" ]] || die "parity source is missing: $PARITY_SOURCE"
  grep -Fq 'fn real_voice_gender_classifier_matches_official_reference' \
    "$PARITY_SOURCE" || die "official CPU parity test is missing"
  grep -Fq 'VOICE_GENDER_OFFICIAL_PARITY MEASURED_NOT_GATED' "$PARITY_SOURCE" \
    || die "CPU parity test does not own its measurement marker"
  grep -Fq 'VOICE_GENDER_METAL_VS_CPU MEASURED_NOT_GATED' "$PARITY_SOURCE" \
    || die "Metal parity test does not own its measurement marker"
  grep -Fq 'BackendKind::Metal' "$PARITY_SOURCE" \
    || die "parity source lacks explicit Metal backend selection"
  grep -Fq 'assert!(metal_error.is_finite())' "$PARITY_SOURCE" \
    || die "Metal/CPU finite metric assertion is missing"
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] \
    || die "remote Apple checkout must be clean so evidence names one exact commit"
  xcrun -f metal >/dev/null 2>&1 || die "Xcode Metal compiler is unavailable"
}

require_reference() {
  local directory="$1" path field
  [[ -d "$directory" ]] || die "reference directory is missing: $directory"
  for path in pcm.f32 features.f32 embedding.f32 logits.f32 probabilities.f32 \
    argmax.u32 meta.json; do
    require_file "voice-gender reference $path" "$directory/$path"
  done
  for field in \
    '"upstream_repository": "https://github.com/JaesungHuh/voice-gender-classifier.git"' \
    '"upstream_revision": "49bcbecfd929ba5a043bde645fdff1a375eb79c7"' \
    '"upstream_hf_revision": "db1222153bd60337e900be22add7af180452adc0"' \
    '"upstream_class": "model.ECAPA_gender"' \
    '"sample_rate": 16000' '"n_mels": 80' '"n_fft": 512' \
    '"win_length": 400' '"hop_length": 160' '"embedding_dim": 192'; do
    grep -Fq -- "$field" "$directory/meta.json" \
      || die "reference metadata is missing exact field: $field"
  done
}

record_environment() {
  local output="$1"
  {
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "git_branch=$(git -C "$VOKRA_ROOT" branch --show-current)"
    echo "upstream_repository=$UPSTREAM_REPOSITORY"
    echo "upstream_revision=$UPSTREAM_REVISION"
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

hash_reference_directory() {
  local directory="$1" output="$2" path
  find "$directory" -mindepth 1 -maxdepth 1 -type f -print \
    | LC_ALL=C sort \
    | while IFS= read -r path; do
        printf '%s  %s\n' "$(sha256_file "$path")" "${path#"$directory"/}"
      done > "$output"
}

run_self_test() (
  local script_path="${BASH_SOURCE[0]}" temporary fail=0 required
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-voice-gender-apple.XXXXXX")"
  trap 'rm -rf "$temporary"' EXIT
  printf 'abc' > "$temporary/value"
  [[ "$(sha256_file "$temporary/value")" == \
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" ]] \
    || die "SHA-256 helper self-test failed"
  require_empty_directory "$temporary/evidence"
  # shellcheck disable=SC2016 # literal contract token intentionally keeps quoting
  for required in \
    'VOKRA_REMOTE_APPLE_SILICON=1' 'Darwin' 'arm64' \
    'MIN_MEMORY_BYTES=32000000000' 'MIN_FREE_DISK_KIB=10000000' \
    'hw.memsize' 'df -Pk' 'xcrun -f metal' \
    "$UPSTREAM_REPOSITORY" "$UPSTREAM_REVISION" \
    'real_voice_gender_classifier_matches_official_reference' \
    'VOKRA_VOICE_GENDER_GGUF' 'VOKRA_VOICE_GENDER_PCM' \
    'VOKRA_VOICE_GENDER_FEATURES' 'VOKRA_VOICE_GENDER_LOGITS' \
    'VOKRA_VOICE_GENDER_EMBEDDING' \
    'VOKRA_VOICE_GENDER_PROBABILITIES' \
    'VOICE_GENDER_OFFICIAL_PARITY MEASURED_NOT_GATED' \
    'VOICE_GENDER_METAL_VS_CPU MEASURED_NOT_GATED' \
    'test result: ok. 1 passed' '--features metal' \
    'cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml"'; do
    grep -Fq -- "$required" "$script_path" \
      || { log "self-test missing contract token: $required"; fail=1; }
  done
  if grep -En '(^|[[:space:]])(curl|wget|python3?|pip|.*convert|git[[:space:]]+(clone|fetch|pull)|.*(upload|publish)|git[[:space:]]+push)([[:space:]]|$)' \
    "$script_path" | grep -v 'UV_NO_CACHE=1 uv run' >/dev/null; then
    log "self-test found acquisition/conversion/publication command"
    fail=1
  fi
  local printf_token='printf'
  if grep -F 'VOICE_GENDER_METAL_VS_CPU' "$script_path" | grep -Fq "$printf_token "; then
    log "self-test found a manufactured Metal PASS marker"
    fail=1
  fi
  if "$script_path" --self-test --gguf "$temporary/model.gguf" >/dev/null 2>&1; then
    log "self-test accepted an extra argument"
    fail=1
  fi
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$PARITY_DUMPER" --self-test >/dev/null || fail=1
  (( fail == 0 )) || return 1
  log "self-test PASS"
)

main() {
  local gguf='' reference_dir='' evidence_dir='' self_test=0 gguf_seen=0 reference_seen=0 evidence_seen=0
  local gguf_sha
  while (( $# > 0 )); do
    case "$1" in
      --gguf) (( gguf_seen == 0 )) || die 'duplicate --gguf'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; gguf="$2"; gguf_seen=1; shift 2 ;;
      --reference-dir) (( reference_seen == 0 )) || die 'duplicate --reference-dir'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; reference_dir="$2"; reference_seen=1; shift 2 ;;
      --evidence-dir) (( evidence_seen == 0 )) || die 'duplicate --evidence-dir'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; evidence_dir="$2"; evidence_seen=1; shift 2 ;;
      --self-test) (( self_test == 0 )) || die 'duplicate --self-test'; self_test=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) usage; die "unknown argument: $1"; return 2 ;;
    esac
  done

  if (( self_test == 1 )); then
    [[ -z "$gguf$reference_dir$evidence_dir" ]] \
      || die "--self-test accepts no other arguments"
    run_self_test
    return
  fi
  [[ -n "$gguf" && -n "$reference_dir" && -n "$evidence_dir" ]] \
    || { usage; die "--gguf, --reference-dir and --evidence-dir are required"; }

  preflight_gate
  require_remote_apple_host
  require_tooling
  require_file "VAST-produced corrected voice-gender GGUF" "$gguf"
  require_reference "$reference_dir"
  require_empty_directory "$evidence_dir"
  record_environment "$evidence_dir/environment.txt"
  gguf_sha="$(sha256_file "$gguf")"
  hash_reference_directory "$reference_dir" "$evidence_dir/reference-hashes.txt"
  {
    echo "gguf=$gguf"
    echo "gguf_sha256=$gguf_sha"
    echo "reference_dir=$reference_dir"
    echo "reference_meta_sha256=$(sha256_file "$reference_dir/meta.json")"
  } > "$evidence_dir/input-hashes.txt"

  log "running exact official CPU and Metal-vs-CPU parity"
  env \
    VOKRA_VOICE_GENDER_GGUF="$gguf" \
    VOKRA_VOICE_GENDER_PCM="$reference_dir/pcm.f32" \
    VOKRA_VOICE_GENDER_FEATURES="$reference_dir/features.f32" \
    VOKRA_VOICE_GENDER_EMBEDDING="$reference_dir/embedding.f32" \
    VOKRA_VOICE_GENDER_LOGITS="$reference_dir/logits.f32" \
    VOKRA_VOICE_GENDER_PROBABILITIES="$reference_dir/probabilities.f32" \
    RUST_TEST_THREADS=1 \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --features metal --test parity_voice_gender_classifier \
      -- --nocapture 2>&1 | tee "$evidence_dir/parity.log"
  grep -Fq 'test real_voice_gender_classifier_matches_official_reference ... ok' \
    "$evidence_dir/parity.log" || die "official CPU parity test did not pass"
  grep -Eq 'test result: ok\. [1-9][0-9]* passed' "$evidence_dir/parity.log" \
    || die "parity log does not prove a nonzero passing test count"
  grep -Fq 'VOICE_GENDER_OFFICIAL_PARITY MEASURED_NOT_GATED' "$evidence_dir/parity.log" \
    || die "official CPU parity measurement marker is absent"
  grep -Fq 'VOICE_GENDER_METAL_VS_CPU MEASURED_NOT_GATED' "$evidence_dir/parity.log" \
    || die "Metal-vs-CPU measurement marker is absent"

  {
    echo 'verdict=MEASURED_NOT_GATED'
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "upstream_repository=$UPSTREAM_REPOSITORY"
    echo "upstream_revision=$UPSTREAM_REVISION"
    echo "gguf_sha256=$gguf_sha"
    echo "reference_meta_sha256=$(sha256_file "$reference_dir/meta.json")"
    echo 'official_cpu_parity=MEASURED_NOT_GATED'
    echo 'metal_vs_cpu=MEASURED_NOT_GATED'
    echo "metal_compiler=$(xcrun -f metal)"
    echo 'download=NOT_PERFORMED'
    echo 'conversion=NOT_PERFORMED'
    echo 'publication=NOT_PERFORMED'
  } > "$evidence_dir/summary.txt"
  log "MEASURED_NOT_GATED: pull only $evidence_dir; remove staged model data after evidence capture"
}

main "$@"
