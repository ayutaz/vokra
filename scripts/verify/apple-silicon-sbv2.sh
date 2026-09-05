#!/usr/bin/env bash
# SBV2 CPU/reference and Metal-vs-CPU measurement on disposable Apple Silicon.
# All model and reference inputs must already be staged by VAST. This worker
# performs no acquisition, conversion, publication, upload, or push.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
MIN_MEMORY_BYTES=32000000000
MIN_FREE_DISK_KIB=10000000
PARITY_SOURCE="$VOKRA_ROOT/crates/vokra-models/tests/parity_sbv2_real.rs"

log() { printf '[sbv2-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: apple-silicon-sbv2.sh --gguf <vast-sbv2-main.gguf> \
  --gguf-sha256 <sha256> --reference-dir <vast-sbv2-bundle> \
  --reference-manifest-sha256 <sha256> --evidence-dir <absent-dir>
       apple-silicon-sbv2.sh --self-test

The bundle contains the main GGUF, BERT sidecars, manifest and raw upstream
reference fixtures produced by VAST. The same request/seed is run through the
existing CPU parity test and the Metal backend; Metal metrics remain
MEASURED_NOT_GATED until a real hardware campaign establishes bounds.
EOF
}

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

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

reject_checkout_descendant() {
  local candidate="$1" canonical_root canonical_candidate parent leaf
  reject_symlink_ancestors "$candidate" || return 2
  canonical_root="$(cd -P "$VOKRA_ROOT" && pwd)" \
    || die "could not resolve Vokra checkout path: $VOKRA_ROOT"
  if [[ -e "$candidate" ]]; then
    [[ -d "$candidate" && ! -L "$candidate" ]] || { die "evidence path is not a regular directory: $candidate"; return 2; }
    canonical_candidate="$(cd -P "$candidate" && pwd)" \
      || die "could not resolve evidence path: $candidate"
  else
    parent="$candidate"
    local -a suffix=()
    while [[ ! -e "$parent" ]]; do
      leaf="${parent##*/}"
      [[ -n "$leaf" ]] || { die "evidence path has an invalid parent: $candidate"; return 2; }
      suffix+=("$leaf")
      [[ "$parent" != / ]] || { die "evidence path parent does not exist: $candidate"; return 2; }
      parent="${parent%/*}"
      [[ -n "$parent" ]] || parent=/
    done
    [[ -d "$parent" && ! -L "$parent" ]] || { die "evidence path parent is not a directory: $parent"; return 2; }
    parent="$(cd -P "$parent" && pwd)" || { die "could not resolve evidence path parent: $parent"; return 2; }
    canonical_candidate="$parent"
    for (( leaf = ${#suffix[@]} - 1; leaf >= 0; leaf-- )); do canonical_candidate="$canonical_candidate/${suffix[leaf]}"; done
  fi
  case "$canonical_candidate" in
    "$canonical_root"|"$canonical_root"/*)
      die "evidence directory must be outside the Vokra checkout: $candidate"
      return 2
      ;;
  esac
}

require_file() {
  local label="$1" path="$2"
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] || die "$label is missing, empty, or symlinked: $path"
  reject_symlink_ancestors "$path"
}

require_empty_directory() {
  local directory="$1"
  reject_symlink_ancestors "$directory"
  [[ ! -e "$directory" && ! -L "$directory" ]] \
    || { die "evidence directory must be absent and non-symlinked: $directory"; return 2; }
}

require_remote_apple_host() {
  local memory_bytes free_disk_kib
  [[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == 1 ]] \
    || die "VOKRA_REMOTE_APPLE_SILICON=1 is absent; refusing maintainer-Mac execution"
  [[ "$(uname -s)" == Darwin ]] || die "SBV2 Metal parity requires Darwin"
  [[ "$(uname -m)" == arm64 ]] || die "SBV2 Metal parity requires Apple arm64"
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
  [[ -f "$PARITY_SOURCE" ]] || die "SBV2 parity source is missing: $PARITY_SOURCE"
  grep -Fq 'fn parity_sbv2_real_waveform_matches_reference_dump' "$PARITY_SOURCE" \
    || die "SBV2 CPU/reference parity test is missing"
  grep -Fq 'VOKRA_SBV2_METAL_VS_CPU' "$PARITY_SOURCE" \
    || die "SBV2 Metal-vs-CPU measurement path is missing"
  grep -Fq 'SBV2_METAL_VS_CPU MEASURED_NOT_GATED' "$PARITY_SOURCE" \
    || die "SBV2 Metal result is not explicitly measurement-only"
  grep -Fq 'BackendKind::Metal' "$PARITY_SOURCE" \
    || die "SBV2 parity source lacks explicit Metal selection"
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] \
    || die "remote Apple checkout must be clean"
  xcrun -f metal >/dev/null 2>&1 || die "Xcode Metal compiler is unavailable"
}

require_reference() {
  local directory="$1"
  [[ -d "$directory" && ! -L "$directory" ]] || die "reference directory is missing or symlinked: $directory"
  reject_symlink_ancestors "$directory"
  require_file "SBV2 manifest" "$directory/reference_dump.manifest.json"
  require_file "SBV2 main GGUF" "$directory/sbv2-v2-multilingual-base.gguf"
  grep -Fq '"sbv2_main"' "$directory/reference_dump.manifest.json" \
    || die "SBV2 manifest lacks checkpoint.sbv2_main"
  grep -Fq '"bert_ja"' "$directory/reference_dump.manifest.json" \
    || die "SBV2 manifest lacks checkpoint.bert_ja"
  grep -Fq '"bert_en"' "$directory/reference_dump.manifest.json" \
    || die "SBV2 manifest lacks checkpoint.bert_en"
  grep -Fq '"request"' "$directory/reference_dump.manifest.json" \
    || die "SBV2 manifest lacks the exact request contract"
  grep -Fq '"seed"' "$directory/reference_dump.manifest.json" \
    || die "SBV2 manifest lacks the exact deterministic seed"
  require_file "SBV2 reference phoneme ids" "$directory/reference_dump/phoneme_ids.bin"
  require_file "SBV2 reference text hidden" "$directory/reference_dump/text_hidden.bin"
  require_file "SBV2 reference waveform" "$directory/reference_dump/waveform.bin"
}

require_cargo_singleton() {
  local log_file="$1" test_name="$2" named_count result_count all_test_lines
  named_count="$(grep -Ec "^test ${test_name} \.\.\. ok$" "$log_file" || true)"
  result_count="$(grep -Ec '^test result:' "$log_file" || true)"
  all_test_lines="$(grep -Ec '^test ' "$log_file" || true)"
  (( named_count == 1 && result_count == 1 && all_test_lines - result_count == 1 )) \
    || { die "Cargo log is not one exact passing test: $log_file"; return 2; }
  grep -Eq '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+(\.[0-9]+)?s)?$' "$log_file" \
    || { die "Cargo result is not an exact singleton pass: $log_file"; return 2; }
}

require_sbv2_metal_sentinel() {
  local log_file="$1"
  [[ "$(grep -Ec '^SBV2_METAL_VS_CPU MEASURED_NOT_GATED waveform_max_abs=[0-9]+\.[0-9]{6}e[+-][0-9]{2} intermediate_max_abs=[0-9]+\.[0-9]{6}e[+-][0-9]{2}$' "$log_file" || true)" == 1 ]] \
    || { die "SBV2 Metal measurement sentinel is not one complete line"; return 2; }
}

require_sbv2_cpu_sentinel() {
  local log_file="$1"
  [[ "$(grep -Ec '^\[parity_sbv2_real\] waveform parity OK: rust=[0-9]+ samples ref=[0-9]+ samples \(ratio [0-9]+\.[0-9]{4}, band ±[0-9]+(\.[0-9]+)?%, overlap [0-9]+ samples: max \|Δ\| = [0-9]+\.[0-9]+e[+-][0-9]+, RMS \|Δ\| = [0-9]+\.[0-9]+e[+-][0-9]+ <= atol [0-9]+(\.[0-9]+)?\)$' "$log_file" || true)" == 1 ]] \
    || { die "SBV2 CPU parity sentinel is not one complete line"; return 2; }
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

hash_directory() {
  local directory="$1" output="$2" path
  find "$directory" -type f -print | LC_ALL=C sort | while IFS= read -r path; do
    printf '%s  %s\n' "$(sha256_file "$path")" "${path#"$directory"/}"
  done > "$output"
}

run_self_test() (
  local script_path="${BASH_SOURCE[0]}" temporary fail=0 required marker_prefix
  marker_prefix='SBV2_METAL_VS_CPU'
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-sbv2-apple.XXXXXX")"
  temporary="$(cd -P "$temporary" && pwd)"
  trap 'rm -rf "$temporary"' EXIT
  printf abc > "$temporary/value"
  [[ "$(sha256_file "$temporary/value")" == \
    ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad ]] \
    || die "SHA-256 helper self-test failed"
  local synthetic_log="$temporary/cargo.log"
  printf 'test parity_sbv2_real_waveform_matches_reference_dump ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s\n[parity_sbv2_real] waveform parity OK: rust=10 samples ref=10 samples (ratio 1.0000, band ±10.0000%%, overlap 10 samples: max |Δ| = 1.000000e-04, RMS |Δ| = 1.000000e-05 <= atol 0.01)\nSBV2_METAL_VS_CPU MEASURED_NOT_GATED waveform_max_abs=1.000000e-04 intermediate_max_abs=2.000000e-04\n' >"$synthetic_log"
  require_cargo_singleton "$synthetic_log" parity_sbv2_real_waveform_matches_reference_dump
  require_sbv2_cpu_sentinel "$synthetic_log"
  require_sbv2_metal_sentinel "$synthetic_log"
  printf 'test extra_case ... ok\n' >>"$synthetic_log"
  if require_cargo_singleton "$synthetic_log" parity_sbv2_real_waveform_matches_reference_dump >/dev/null 2>&1; then
    log "self-test accepted an extra Cargo test line"
    fail=1
  fi
  reject_symlink_ancestors "$temporary/evidence"
  if [[ -e "$temporary/evidence" ]]; then
    log "self-test: existing evidence path was accepted"
    fail=1
  fi
  reject_checkout_descendant "$temporary/evidence" \
    || { log "self-test rejected an external evidence directory"; fail=1; }
  if reject_checkout_descendant "$VOKRA_ROOT" >/dev/null 2>&1; then
    log "self-test accepted the checkout itself as evidence"
    fail=1
  fi
  if reject_checkout_descendant "$VOKRA_ROOT/.git" >/dev/null 2>&1; then
    log "self-test accepted a checkout descendant as evidence"
    fail=1
  fi
  if "$script_path" --gguf "$temporary/value" \
    --gguf-sha256 ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad \
    --reference-dir "$temporary/missing-reference" \
    --reference-manifest-sha256 ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad \
    --evidence-dir "$temporary/evidence" >/dev/null 2>&1; then
    log "self-test accepted a production-shaped incomplete input"
    fail=1
  fi
  [[ ! -e "$temporary/evidence" && ! -L "$temporary/evidence" ]] \
    || { log "self-test: blocked production probe created evidence"; fail=1; }
  # shellcheck disable=SC2016 # literal contract tokens
  for required in \
    'VOKRA_REMOTE_APPLE_SILICON=1' 'Darwin' 'arm64' \
    'MIN_MEMORY_BYTES=32000000000' 'hw.memsize' 'xcrun -f metal' \
    'parity_sbv2_real_waveform_matches_reference_dump' \
    'VOKRA_SBV2_FIXTURE_DIR' 'VOKRA_SBV2_METAL_VS_CPU=1' \
    'SBV2_METAL_VS_CPU MEASURED_NOT_GATED' \
    'CARGO_BUILD_JOBS=1 cargo test' \
    'reject_checkout_descendant' \
    'cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml"' \
    '--features metal' '--test parity_sbv2_real'; do
    grep -Fq -- "$required" "$script_path" \
      || { log "self-test missing contract token: $required"; fail=1; }
  done
  # The Rust test must finish all CPU/reference gates before it can enter the
  # optional Metal branch. This prevents a Metal run from obscuring a failed
  # official CPU fixture check.
  local cpu_gate_line metal_branch_line
  cpu_gate_line="$(grep -n 'waveform parity OK' "$PARITY_SOURCE" | head -n1 | cut -d: -f1)"
  metal_branch_line="$(grep -n 'model.set_backend(BackendKind::Metal)' "$PARITY_SOURCE" | head -n1 | cut -d: -f1)"
  [[ "$cpu_gate_line" =~ ^[0-9]+$ && "$metal_branch_line" =~ ^[0-9]+$ && \
    "$cpu_gate_line" -lt "$metal_branch_line" ]] \
    || { log "self-test could not prove CPU gates precede Metal execution"; fail=1; }
  if grep -En '(^|[[:space:]])(curl|wget|python3?|pip|.*convert|git[[:space:]]+(clone|fetch|pull)|.*(upload|publish)|git[[:space:]]+push)([[:space:]]|$)' \
    "$script_path" >/dev/null; then
    log "self-test found acquisition/conversion/publication command"
    fail=1
  fi
  if grep -F "printf \"$marker_prefix" "$script_path"; then
    log "self-test found a manufactured Metal result"
    fail=1
  fi
  if "$script_path" --self-test --gguf "$temporary/model.gguf" >/dev/null 2>&1; then
    log "self-test accepted an extra argument"
    fail=1
  fi
  if "$script_path" --self-test --self-test >/dev/null 2>&1; then
    log "self-test accepted duplicate --self-test"
    fail=1
  fi
  (( fail == 0 )) || return 1
  log "self-test PASS"
)

main() {
  local gguf='' gguf_sha_expected='' reference_dir='' reference_manifest_sha_expected='' evidence_dir='' self_test=0 output gguf_sha reference_manifest_sha
  local seen_gguf=0 seen_gguf_sha=0 seen_reference=0 seen_reference_sha=0 seen_evidence=0 seen_self=0
  while (( $# > 0 )); do
    case "$1" in
      --gguf) (( seen_gguf == 0 )) || die 'duplicate --gguf'; (( $# >= 2 )) || die '--gguf requires a path'; [[ -n "$2" && "$2" != -* ]] || die '--gguf path is empty or starts with -'; seen_gguf=1; gguf="$2"; shift 2 ;;
      --gguf-sha256) (( seen_gguf_sha == 0 )) || die 'duplicate --gguf-sha256'; (( $# >= 2 )) || die '--gguf-sha256 requires a value'; require_sha256 "$2"; seen_gguf_sha=1; gguf_sha_expected="${2,,}"; shift 2 ;;
      --reference-dir) (( seen_reference == 0 )) || die 'duplicate --reference-dir'; (( $# >= 2 )) || die '--reference-dir requires a path'; [[ -n "$2" && "$2" != -* ]] || die '--reference-dir path is empty or starts with -'; seen_reference=1; reference_dir="$2"; shift 2 ;;
      --reference-manifest-sha256) (( seen_reference_sha == 0 )) || die 'duplicate --reference-manifest-sha256'; (( $# >= 2 )) || die '--reference-manifest-sha256 requires a value'; require_sha256 "$2"; seen_reference_sha=1; reference_manifest_sha_expected="${2,,}"; shift 2 ;;
      --evidence-dir) (( seen_evidence == 0 )) || die 'duplicate --evidence-dir'; (( $# >= 2 )) || die '--evidence-dir requires a path'; [[ -n "$2" && "$2" != -* ]] || die '--evidence-dir path is empty or starts with -'; seen_evidence=1; evidence_dir="$2"; shift 2 ;;
      --self-test) (( seen_self == 0 )) || die 'duplicate --self-test'; seen_self=1; self_test=1; shift ;;
      -h|--help) [[ $self_test == 0 && $# == 1 ]] || die '--help cannot be combined with other arguments'; usage; return 0 ;;
      *) usage; die "unknown argument: $1"; return 2 ;;
    esac
  done
  if (( self_test == 1 )); then
    [[ "$seen_gguf" == 0 && "$seen_gguf_sha" == 0 && "$seen_reference" == 0 && "$seen_reference_sha" == 0 && "$seen_evidence" == 0 ]] || die "--self-test accepts no other arguments"
    run_self_test
    return
  fi
  [[ -n "$gguf" && -n "$gguf_sha_expected" && -n "$reference_dir" && -n "$reference_manifest_sha_expected" && -n "$evidence_dir" ]] \
    || { usage; die "--gguf, --gguf-sha256, --reference-dir, --reference-manifest-sha256 and --evidence-dir are required"; }
  require_file "VAST-produced SBV2 GGUF" "$gguf"
  gguf_sha="$(sha256_file "$gguf")"
  [[ "$gguf_sha" == "$gguf_sha_expected" ]] || die "SBV2 GGUF SHA-256 mismatch"
  require_reference "$reference_dir"
  reference_manifest_sha="$(sha256_file "$reference_dir/reference_dump.manifest.json")"
  [[ "$reference_manifest_sha" == "$reference_manifest_sha_expected" ]] || die "SBV2 reference manifest SHA-256 mismatch"
  reject_checkout_descendant "$evidence_dir"
  require_empty_directory "$evidence_dir"
  require_remote_apple_host
  require_tooling
  [[ "$gguf" -ef "$reference_dir/sbv2-v2-multilingual-base.gguf" ]] \
    || die "--gguf must be the manifest's staged SBV2 main path"
  mkdir -p "$evidence_dir"
  record_environment "$evidence_dir/environment.txt"
  hash_directory "$reference_dir" "$evidence_dir/input-hashes.txt"
  {
    echo "gguf=$gguf"
    echo "gguf_sha256=$gguf_sha"
    echo "reference_dir=$reference_dir"
  } >> "$evidence_dir/input-hashes.txt"
  output="$evidence_dir/parity.log"
  log "running SBV2 CPU/reference and Metal-vs-CPU measurement"
  if ! VOKRA_SBV2_FIXTURE_DIR="$reference_dir" \
    VOKRA_SBV2_METAL_VS_CPU=1 \
    CARGO_BUILD_JOBS=1 cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --features metal --test parity_sbv2_real \
      parity_sbv2_real_waveform_matches_reference_dump -- --exact --nocapture 2>&1 | tee "$output"; then
    die "SBV2 CPU/Metal parity test failed; see $output"
  fi
  require_cargo_singleton "$output" parity_sbv2_real_waveform_matches_reference_dump
  require_sbv2_cpu_sentinel "$output"
  require_sbv2_metal_sentinel "$output"
  log "MEASURED_NOT_GATED: evidence is in $evidence_dir; no model publication performed"
}

main "$@"
