#!/usr/bin/env bash
# Real-weight WeSpeaker/ECAPA CPU-vs-Metal parity on a disposable Apple host.
# Inputs are produced and authenticated by the VAST worker; this script never
# downloads, converts, publishes, uploads, or pushes model artifacts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"

MIN_MEMORY_BYTES=32000000000
MIN_FREE_DISK_KIB=10000000

log() { printf '[speaker-replacements-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: apple-silicon-speaker-replacements.sh \
  --wespeaker-gguf <vast-official-combined-219.gguf> \
  --wespeaker-gguf-sha256 <sha256> --ecapa-gguf <vast-corrected-ecapa.gguf> \
  --ecapa-gguf-sha256 <sha256> --evidence-dir <absent-dir>
       apple-silicon-speaker-replacements.sh --self-test

Runs the exact real WeSpeaker official-combined-219 and corrected ECAPA
CPU-vs-Metal integration tests on a remote Apple Silicon host. It requires
VOKRA_REMOTE_APPLE_SILICON=1, Darwin arm64, a clean checkout, at least 32 GB
physical memory, free disk, and the Xcode Metal compiler.

This script does not download, convert, publish, upload, or push model files.
The two GGUFs must be staged by the VAST workflow. Pull only evidence after
the run, then remove staged model data or destroy the remote worker.
EOF
}

sha256_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

require_sha256() {
  [[ "$1" =~ ^[0-9a-fA-F]{64}$ ]] || die "expected SHA-256 is malformed"
}

reject_symlink_ancestors() {
  local path="$1" component rest current
  [[ "$path" == /* ]] || { die "path must be absolute: $path"; return 2; }
  rest="${path#/}"
  current="/"
  while [[ -n "$rest" ]]; do
    if [[ "$rest" == */* ]]; then component="${rest%%/*}"; rest="${rest#*/}"; else component="$rest"; rest=""; fi
    [[ -n "$component" ]] || continue
    current="$current$component"
    [[ ! -L "$current" ]] || { die "path contains symlink ancestor: $path"; return 2; }
    current="$current/"
  done
}

require_file() {
  local label="$1" path="$2"
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] || die "$label is missing, empty, or symlinked: $path"
  reject_symlink_ancestors "$path" || return 2
}

require_empty_directory() {
  local directory="$1"
  reject_symlink_ancestors "$directory" || return 2
  [[ ! -e "$directory" && ! -L "$directory" ]] \
    || { die "evidence directory must be absent and non-symlinked: $directory"; return 2; }
}

validate_evidence_path() {
  local evidence="$1" first_input="$2" second_input="$3" component rest current parent candidate item root_real input_parent input_real
  local -a suffix=()
  reject_symlink_ancestors "$evidence" || return 2
  [[ ! -e "$evidence" && ! -L "$evidence" ]] || { die "evidence directory must be absent: $evidence"; return 2; }
  parent="$evidence"
  while [[ ! -e "$parent" ]]; do
    item="${parent##*/}"
    [[ -n "$item" ]] || { die "evidence path has an invalid parent: $evidence"; return 2; }
    suffix+=("$item")
    [[ "$parent" != / ]] || { die "evidence path parent does not exist: $evidence"; return 2; }
    parent="${parent%/*}"
    [[ -n "$parent" ]] || parent=/
  done
  [[ -d "$parent" && ! -L "$parent" ]] || { die "evidence path parent is not a directory: $parent"; return 2; }
  candidate="$(cd -P "$parent" && pwd)" || { die "could not resolve evidence path parent"; return 2; }
  for (( item = ${#suffix[@]} - 1; item >= 0; item-- )); do candidate="$candidate/${suffix[item]}"; done
  root_real="$(cd -P "$VOKRA_ROOT" && pwd)" || { die "could not resolve Vokra checkout"; return 2; }
  [[ "$candidate" != "$root_real" && "$candidate/" != "$root_real/"* && "$root_real/" != "$candidate/"* ]] || { die "evidence directory overlaps checkout"; return 2; }
  for input in "$first_input" "$second_input"; do
    input_parent="$(cd -P "$(dirname "$input")" && pwd)" || { die "could not resolve input parent"; return 2; }
    input_real="$input_parent/$(basename "$input")"
    [[ "$candidate" != "$input_real" && "$candidate/" != "$input_real/"* && "$input_real/" != "$candidate/"* ]] || { die "evidence directory overlaps model input"; return 2; }
  done
}

require_cargo_singleton() {
  local log_file="$1" test_name="$2" named_count result_count all_test_lines
  named_count="$(grep -Ec "^test ${test_name} \.\.\. ok$" "$log_file" || true)"
  result_count="$(grep -Ec '^test result:' "$log_file" || true)"
  all_test_lines="$(grep -Ec '^test ' "$log_file" || true)"
  (( named_count == 1 && result_count == 1 && all_test_lines - result_count == 1 )) || { die "Cargo log is not one exact passing test"; return 2; }
  grep -Eq '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+(\.[0-9]+)?s)?$' "$log_file" || { die "Cargo result is not an exact singleton pass"; return 2; }
}

require_sentinel_line() {
  local log_file="$1" prefix="$2" expected="$3" label="$4" family_count exact_count
  family_count="$(grep -Ec "^${prefix}" "$log_file" || true)"
  exact_count="$(grep -Fxc -- "$expected" "$log_file" || true)"
  [[ "$family_count" == 1 && "$exact_count" == 1 ]] \
    || { die "$label is not one exact line"; return 2; }
}

require_metric_sentinel() {
  local log_file="$1" prefix="$2"
  [[ "$(grep -Ec "^${prefix} max_abs=[0-9]+\.[0-9]{9}e[+-][0-9]{2} at [0-9]+ \(actual=-?[0-9]+\.[0-9]{9}e[+-][0-9]{2}, reference=-?[0-9]+\.[0-9]{9}e[+-][0-9]{2}\), mean_abs=[0-9]+\.[0-9]{9}e[+-][0-9]{2}, relative_l1=[0-9]+\.[0-9]{9}e[+-][0-9]{2}, cosine=-?[0-9]+\.[0-9]{9}$" "$log_file" || true)" == 1 ]] || { die "metric sentinel is not one complete line"; return 2; }
}

require_remote_apple_host() {
  local memory_bytes free_disk_kib
  [[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == "1" ]] \
    || die "VOKRA_REMOTE_APPLE_SILICON=1 is absent; refusing maintainer-Mac execution"
  [[ "$(uname -s)" == "Darwin" ]] || die "remote Metal parity requires Darwin"
  [[ "$(uname -m)" == "arm64" ]] || die "remote Metal parity requires Apple arm64"
  memory_bytes="$(sysctl -n hw.memsize)"
  [[ "$memory_bytes" =~ ^[0-9]+$ ]] || die "could not read hw.memsize"
  (( memory_bytes >= MIN_MEMORY_BYTES )) \
    || die "physical memory $memory_bytes bytes is below the 32-GB guard"
  free_disk_kib="$(df -Pk "$VOKRA_ROOT" | awk 'NR == 2 {print $4}')"
  [[ "$free_disk_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
  (( free_disk_kib >= MIN_FREE_DISK_KIB )) \
    || die "free disk $free_disk_kib KiB is below the 10-GB run guard"
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
  local temporary script_path synthetic_log metric_log fail=0
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-speakers-apple.XXXXXX")"
  temporary="$(cd -P "$temporary" && pwd)"
  trap 'rm -rf "$temporary"' EXIT
  printf 'abc' > "$temporary/value"
  [[ "$(sha256_file "$temporary/value")" == \
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" ]] \
    || die "SHA-256 helper self-test failed"
  require_empty_directory "$temporary/evidence"
  synthetic_log="$temporary/synthetic-cargo.log"
  printf '%s\n' \
    'test official_combined_artifact_matches_upstream_wespeaker ... ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s' \
    'WESPEAKER_OFFICIAL_COMBINED_CPU_VS_UPSTREAM PASS' \
    'WESPEAKER_OFFICIAL_COMBINED_METAL_VS_CPU PASS' > "$synthetic_log"
  require_cargo_singleton "$synthetic_log" official_combined_artifact_matches_upstream_wespeaker
  require_sentinel_line "$synthetic_log" 'WESPEAKER_OFFICIAL_COMBINED_CPU_VS_UPSTREAM' \
    'WESPEAKER_OFFICIAL_COMBINED_CPU_VS_UPSTREAM PASS' 'WeSpeaker CPU sentinel'
  require_sentinel_line "$synthetic_log" 'WESPEAKER_OFFICIAL_COMBINED_METAL_VS_CPU' \
    'WESPEAKER_OFFICIAL_COMBINED_METAL_VS_CPU PASS' 'WeSpeaker Metal sentinel'
  printf '%s\n' 'WESPEAKER_OFFICIAL_COMBINED_METAL_VS_CPU PASS extra' >> "$synthetic_log"
  if require_sentinel_line "$synthetic_log" 'WESPEAKER_OFFICIAL_COMBINED_METAL_VS_CPU' \
    'WESPEAKER_OFFICIAL_COMBINED_METAL_VS_CPU PASS' 'WeSpeaker Metal sentinel' >/dev/null 2>&1; then
    log 'self-test FAIL: sentinel suffix was accepted'
    fail=1
  fi
  metric_log="$temporary/metric.log"
  printf '%s\n' 'ECAPA-TDNN Metal embedding vs CPU: max_abs=1.000000000e-04 at 2 (actual=1.000000000e-01, reference=1.000000000e-01), mean_abs=1.000000000e-05, relative_l1=1.000000000e-04, cosine=0.999999999' > "$metric_log"
  require_metric_sentinel "$metric_log" 'ECAPA-TDNN Metal embedding vs CPU:'
  script_path="${BASH_SOURCE[0]}"
  # shellcheck disable=SC2016 # contract token intentionally keeps literal quoting
  for required in "VOKRA_REMOTE_APPLE_SILICON=1" "Darwin" "arm64" \
    "MIN_MEMORY_BYTES=32000000000" "MIN_FREE_DISK_KIB=10000000" \
    "hw.memsize" "df -Pk" \
    'git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all' \
    "xcrun -f metal" "parity_wespeaker_real" \
    "official_combined_artifact_matches_upstream_wespeaker" \
    "parity_ecapa_tdnn_real" "public_artifact_matches_speechbrain" \
    "WESPEAKER_OFFICIAL_COMBINED_METAL_VS_CPU PASS" \
    "ECAPA-TDNN Metal embedding vs CPU:" "require_metric_sentinel" \
    "--features metal" \
    "test result: ok. 1 passed" \
    "test official_combined_artifact_matches_upstream_wespeaker ... ok" \
    "test public_artifact_matches_speechbrain ... ok"; do
    grep -F -- "$required" "$script_path" >/dev/null \
      || die "contract token is missing: $required"
  done
  grep -E '(^|[[:space:]])(curl|wget|hf_hub_download|git[[:space:]]+push|upload\.sh|publish-one\.sh)([[:space:]]|$)' \
    "$script_path" >/dev/null \
    && die "download or publication command found"
  if "$script_path" --self-test --evidence-dir "$temporary/other" >/dev/null 2>&1; then
    die "--self-test accepted an extra argument"
  fi
  if "$script_path" --self-test --self-test >/dev/null 2>&1; then
    die "--self-test accepted duplicate invocation"
  fi
  if "$script_path" \
    --wespeaker-gguf "$temporary/value" \
    --wespeaker-gguf-sha256 ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad \
    --ecapa-gguf "$temporary/value" \
    --ecapa-gguf-sha256 ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad \
    --evidence-dir "$temporary/evidence" >/dev/null 2>&1; then
    log "self-test: production-shaped incomplete host probe unexpectedly passed"
    fail=1
  fi
  [[ ! -e "$temporary/evidence" && ! -L "$temporary/evidence" ]] \
    || { log "self-test: blocked production probe created evidence"; fail=1; }
  (( fail == 0 )) || return 1
  log "self-test PASS"
)

main() {
  local wespeaker_gguf='' wespeaker_expected='' ecapa_gguf='' ecapa_expected='' evidence_dir='' self_test=0
  local wespeaker_sha ecapa_sha
  local seen_wespeaker=0 seen_wespeaker_sha=0 seen_ecapa=0 seen_ecapa_sha=0 seen_evidence=0 seen_self=0
  while (( $# > 0 )); do
    case "$1" in
      --wespeaker-gguf)
        (( seen_wespeaker == 0 )) || die 'duplicate --wespeaker-gguf'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; seen_wespeaker=1
        wespeaker_gguf="$2"; shift 2 ;;
      --wespeaker-gguf-sha256)
        (( seen_wespeaker_sha == 0 )) || die 'duplicate --wespeaker-gguf-sha256'; [[ $# -ge 2 ]] || { usage; return 2; }; require_sha256 "$2"; seen_wespeaker_sha=1
        wespeaker_expected="${2,,}"; shift 2 ;;
      --ecapa-gguf)
        (( seen_ecapa == 0 )) || die 'duplicate --ecapa-gguf'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; seen_ecapa=1
        ecapa_gguf="$2"; shift 2 ;;
      --ecapa-gguf-sha256)
        (( seen_ecapa_sha == 0 )) || die 'duplicate --ecapa-gguf-sha256'; [[ $# -ge 2 ]] || { usage; return 2; }; require_sha256 "$2"; seen_ecapa_sha=1
        ecapa_expected="${2,,}"; shift 2 ;;
      --evidence-dir)
        (( seen_evidence == 0 )) || die 'duplicate --evidence-dir'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; seen_evidence=1
        evidence_dir="$2"; shift 2 ;;
      --self-test)
        (( seen_self == 0 )) || die 'duplicate --self-test'; seen_self=1; self_test=1; shift ;;
      -h|--help)
        [[ $self_test == 0 && $# == 1 ]] || die '--help cannot be combined with other arguments'; usage; return 0 ;;
      *)
        usage; die "unknown argument $1" ;;
    esac
  done

  if (( self_test == 1 )); then
    [[ "$seen_wespeaker" == 0 && "$seen_wespeaker_sha" == 0 && "$seen_ecapa" == 0 && "$seen_ecapa_sha" == 0 && "$seen_evidence" == 0 ]] \
      || die "--self-test accepts no other arguments"
    run_self_test
    return
  fi
  [[ -n "$wespeaker_gguf" && -n "$wespeaker_expected" && -n "$ecapa_gguf" && -n "$ecapa_expected" && -n "$evidence_dir" ]] \
    || { usage; die "all GGUF, expected SHA-256, and --evidence-dir arguments are required"; }

  require_file "official combined WeSpeaker 219 GGUF" "$wespeaker_gguf"
  wespeaker_sha="$(sha256_file "$wespeaker_gguf")"
  [[ "$wespeaker_sha" == "$wespeaker_expected" ]] || die "WeSpeaker GGUF SHA-256 mismatch"
  require_file "corrected ECAPA GGUF" "$ecapa_gguf"
  ecapa_sha="$(sha256_file "$ecapa_gguf")"
  [[ "$ecapa_sha" == "$ecapa_expected" ]] || die "ECAPA GGUF SHA-256 mismatch"
  validate_evidence_path "$evidence_dir" "$wespeaker_gguf" "$ecapa_gguf"

  require_remote_apple_host
  require_tooling
  mkdir -p "$evidence_dir"
  record_environment "$evidence_dir/environment.txt"
  {
    echo "wespeaker_official_combined_219_gguf_sha256=$wespeaker_sha"
    echo "ecapa_corrected_gguf_sha256=$ecapa_sha"
  } > "$evidence_dir/input-hashes.txt"

  log "running WeSpeaker official-combined-219 CPU/Metal parity"
  env VOKRA_WESPEAKER_OFFICIAL_GGUF="$wespeaker_gguf" \
    CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}" RUST_TEST_THREADS=1 \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --features metal --test parity_wespeaker_real \
      official_combined_artifact_matches_upstream_wespeaker \
      -- --exact --nocapture 2>&1 | tee "$evidence_dir/wespeaker-parity.log"
  require_cargo_singleton "$evidence_dir/wespeaker-parity.log" official_combined_artifact_matches_upstream_wespeaker
  require_sentinel_line "$evidence_dir/wespeaker-parity.log" \
    'WESPEAKER_OFFICIAL_COMBINED_CPU_VS_UPSTREAM' \
    'WESPEAKER_OFFICIAL_COMBINED_CPU_VS_UPSTREAM PASS' 'WeSpeaker CPU sentinel'
  require_sentinel_line "$evidence_dir/wespeaker-parity.log" \
    'WESPEAKER_OFFICIAL_COMBINED_METAL_VS_CPU' \
    'WESPEAKER_OFFICIAL_COMBINED_METAL_VS_CPU PASS' 'WeSpeaker Metal-vs-CPU sentinel'

  log "running corrected ECAPA CPU/Metal parity"
  env VOKRA_ECAPA_GGUF="$ecapa_gguf" \
    CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}" RUST_TEST_THREADS=1 \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --features metal --test parity_ecapa_tdnn_real \
      public_artifact_matches_speechbrain \
      -- --exact --nocapture 2>&1 | tee "$evidence_dir/ecapa-parity.log"
  require_cargo_singleton "$evidence_dir/ecapa-parity.log" public_artifact_matches_speechbrain
  require_metric_sentinel "$evidence_dir/ecapa-parity.log" 'ECAPA-TDNN Metal embedding vs CPU:'

  {
    echo "verdict=PASS"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "wespeaker_official_combined_219_cpu_vs_metal=PASS"
    echo "ecapa_corrected_cpu_vs_metal=PASS"
    echo "metal_compiler=$(xcrun -f metal)"
    echo "upload=NOT_PERFORMED"
  } > "$evidence_dir/summary.txt"
  log "PASS: pull only $evidence_dir, then remove staged data or destroy the remote worker"
}

main "$@"
