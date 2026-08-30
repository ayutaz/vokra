#!/usr/bin/env bash
# Real-weight NVIDIA Canary-1B-v2 CPU/reference/Metal parity on a disposable
# remote Apple Silicon host.
#
# The complete GGUF and the two official NeMo reference cases are produced on
# VAST.  This verifier never downloads, converts, publishes, uploads, or
# mutates model artifacts.  It refuses the historical CPU-only test contract
# instead of presenting a Metal-feature CPU rerun as device parity.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"

MIN_MEMORY_BYTES=32000000000
MIN_FREE_DISK_KIB=20000000
GGUF_ENV="VOKRA_CANARY_V2_REAL_GGUF"
REFERENCE_PCM_ENV="VOKRA_CANARY_V2_REFERENCE_PCM"
REFERENCE_TOKENS_ENV="VOKRA_CANARY_V2_REFERENCE_TOKENS"
SOURCE_LANGUAGE_ENV="VOKRA_CANARY_V2_SOURCE_LANGUAGE"
TARGET_LANGUAGE_ENV="VOKRA_CANARY_V2_TARGET_LANGUAGE"
PARITY_SOURCE="$VOKRA_ROOT/crates/vokra-models/src/canary/mod.rs"
TEST_TARGET="canary::tests::canary_v2_released_checkpoint_matches_official_nemo_greedy_tokens"
TEST_NAME="canary_v2_released_checkpoint_matches_official_nemo_greedy_tokens"
PREFLIGHT_GATE="$VOKRA_ROOT/tools/parity/canary_1b/preflight_gate.py"
PREFLIGHT_MANIFEST="$VOKRA_ROOT/tools/parity/canary_1b/license_gate_manifest.json"
VARIANT="canary-1b-v2"

log() { printf '[canary-1b-v2-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: apple-silicon-canary-1b-v2.sh \
  --gguf <vast-generated-canary-1b-v2.gguf> \
  --reference <vast-official-reference-dir> \
  --approval-evidence <owner-approval.json> --evidence-dir <absent-dir>
       apple-silicon-canary-1b-v2.sh --self-test

Runs the exact existing Canary-1B-v2 real-weight test for the English ASR and
English-to-German AST cases on a disposable Darwin/arm64 host.  The test must
contain both the independent official-NeMo comparison and its explicit
Metal-vs-CPU leg; this wrapper refuses the historical CPU-only implementation.
It requires VOKRA_REMOTE_APPLE_SILICON=1, a clean checkout, at least 32 GB
physical memory, free disk, and the Xcode Metal compiler.

This script performs no download, conversion, upload, publication, or model
mutation.  Pull only the evidence directory after the run, then remove staged
inputs or destroy the disposable Apple worker.
EOF
}

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

require_file() {
  local label="$1" path="$2"
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] || die "$label is missing, empty, or symlinked: $path"
}

license_preflight() {
  local approval="$1"
  [[ -f "$PREFLIGHT_GATE" && ! -L "$PREFLIGHT_GATE" && \
    -f "$PREFLIGHT_MANIFEST" && ! -L "$PREFLIGHT_MANIFEST" ]] \
    || die "Canary-1B approval gate or manifest is missing or symlinked"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python \
    "$PREFLIGHT_GATE" --manifest "$PREFLIGHT_MANIFEST" \
    --approval "$approval" --variant "$VARIANT" \
    || die "Canary-1B-v2 approval preflight is unresolved"
}

require_absent_directory() {
  local directory="$1"
  [[ ! -e "$directory" && ! -L "$directory" ]] || die "evidence directory must be absent: $directory"
}

canonical_existing_path() {
  local target="$1" lexical current="/" component parent
  [[ "$target" = /* ]] || target="$PWD/$target"
  lexical="${target#/}"
  while [[ -n "$lexical" ]]; do
    component="${lexical%%/*}"
    if [[ "$lexical" == "$component" ]]; then lexical=""; else lexical="${lexical#*/}"; fi
    [[ "$component" == "." || -z "$component" ]] && continue
    [[ "$component" != ".." ]] || { die "path contains ..: $target"; return 2; }
    current="${current%/}/$component"
    [[ ! -L "$current" ]] || { die "path contains symlinked ancestor: $target"; return 2; }
  done
  [[ -e "$target" && ! -L "$target" ]] || { die "path is missing or symlinked: $target"; return 2; }
  if [[ -d "$target" ]]; then (cd -P "$target" && pwd); else
    parent="$(cd -P "$(dirname "$target")" 2>/dev/null && pwd)" || return 2
    printf '%s/%s\n' "$parent" "$(basename "$target")"
  fi
}

canonical_absent_path() {
  local target="$1" lexical current="/" component suffix="" real
  [[ "$target" = /* ]] || target="$PWD/$target"
  lexical="${target#/}"
  while [[ -n "$lexical" ]]; do
    component="${lexical%%/*}"
    if [[ "$lexical" == "$component" ]]; then lexical=""; else lexical="${lexical#*/}"; fi
    [[ "$component" == "." || -z "$component" ]] && continue
    [[ "$component" != ".." ]] || { die "path contains ..: $target"; return 2; }
    current="${current%/}/$component"
    [[ ! -L "$current" ]] || { die "path contains symlinked ancestor: $target"; return 2; }
  done
  current="$target"
  while [[ ! -e "$current" && ! -L "$current" ]]; do
    component="$(basename "$current")"; suffix="/$component$suffix"; current="$(dirname "$current")"
  done
  [[ -d "$current" && ! -L "$current" ]] || { die "path parent is missing or symlinked"; return 2; }
  real="$(cd -P "$current" 2>/dev/null && pwd)" || return 2
  printf '%s%s\n' "$real" "$suffix"
}

paths_overlap() { local left="$1" right="$2"; [[ "$left" == "$right" || "$left/" == "$right/"* || "$right/" == "$left/"* ]]; }

require_disjoint_evidence() {
  local evidence="$1" gguf="$2" reference="$3" approval="$4" root_real evidence_real protected
  root_real="$(canonical_existing_path "$VOKRA_ROOT")" || return 2
  canonical_existing_path "$gguf" >/dev/null || return 2
  canonical_existing_path "$reference" >/dev/null || return 2
  canonical_existing_path "$approval" >/dev/null || return 2
  evidence_real="$(canonical_absent_path "$evidence")" || return 2
  for protected in "$root_real" "$(canonical_existing_path "$gguf")" "$(canonical_existing_path "$reference")" "$(canonical_existing_path "$approval")"; do
    paths_overlap "$evidence_real" "$protected" && { die "evidence directory overlaps protected input"; return 2; }
  done
}

production_order_ok() {
  local script_path="$1" gate_pattern="$2" host_pattern="$3" resource_pattern="$4"
  local checkpoint_pattern="$5" scratch_pattern="$6" cargo_pattern="$7"
  local gate_line host_line resource_line checkpoint_line scratch_line cargo_line
  gate_line="$(grep -nE "$gate_pattern" "$script_path" | tail -1 | cut -d: -f1 || true)"
  host_line="$(grep -nE "$host_pattern" "$script_path" | tail -1 | cut -d: -f1 || true)"
  resource_line="$(grep -nE "$resource_pattern" "$script_path" | tail -1 | cut -d: -f1 || true)"
  checkpoint_line="$(grep -nE "$checkpoint_pattern" "$script_path" | tail -1 | cut -d: -f1 || true)"
  scratch_line="$(grep -nE "$scratch_pattern" "$script_path" | tail -1 | cut -d: -f1 || true)"
  cargo_line="$(grep -nE "$cargo_pattern" "$script_path" | tail -1 | cut -d: -f1 || true)"
  [[ -n "$gate_line" && -n "$host_line" && -n "$resource_line" \
    && -n "$checkpoint_line" && -n "$scratch_line" && -n "$cargo_line" \
    && "$gate_line" -lt "$host_line" && "$gate_line" -lt "$resource_line" \
    && "$gate_line" -lt "$checkpoint_line" && "$gate_line" -lt "$scratch_line" \
    && "$gate_line" -lt "$cargo_line" ]]
}

require_reference() {
  local directory="$1" suffix name
  [[ -d "$directory" && ! -L "$directory" ]] || die "reference is missing or symlinked: $directory"
  for suffix in en-en en-de; do
    for name in json pcm.f32 tokens.txt text.txt; do
      require_file "Canary-v2 reference ${suffix}.${name}" \
        "$directory/reference-${suffix}.${name}"
    done
  done
}

require_remote_apple_host() {
  local memory_bytes free_disk_kib
  [[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == "1" ]] \
    || die "VOKRA_REMOTE_APPLE_SILICON=1 is absent; refusing possible maintainer-Mac execution"
  [[ "$(uname -s)" == "Darwin" ]] \
    || die "real Metal parity requires Darwin"
  [[ "$(uname -m)" == "arm64" ]] \
    || die "real Metal parity requires Apple arm64"
  memory_bytes="$(sysctl -n hw.memsize)"
  [[ "$memory_bytes" =~ ^[0-9]+$ ]] || die "could not read hw.memsize"
  (( memory_bytes >= MIN_MEMORY_BYTES )) \
    || die "physical memory $memory_bytes bytes is below the exact 32-GB guard"
  free_disk_kib="$(df -Pk "$VOKRA_ROOT" | awk 'NR == 2 {print $4}')"
  [[ "$free_disk_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
  (( free_disk_kib >= MIN_FREE_DISK_KIB )) \
    || die "free disk $free_disk_kib KiB is below the exact 20-GB guard"
}

require_tooling() {
  local tool
  for tool in cargo rustc git shasum awk find tee grep sysctl sw_vers \
    system_profiler xcrun sed; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" ]] \
    || die "$VOKRA_ROOT is not a Vokra git checkout"
  [[ -f "$PARITY_SOURCE" ]] \
    || die "Canary-v2 parity source is missing: $PARITY_SOURCE"
  grep -Fq "fn $TEST_NAME" "$PARITY_SOURCE" \
    || die "Canary-v2 real-weight test is missing: $TEST_NAME"
  grep -Fq 'official NeMo token fixture' "$PARITY_SOURCE" \
    || die "Canary-v2 test lacks the independent official-NeMo token input"
  grep -Fq 'CanaryAsr::from_gguf_with_backend(&gguf, BackendKind::Cpu)' \
    "$PARITY_SOURCE" \
    || die "Canary-v2 test lacks its CPU real-weight bind"
  # Restrict this check to the real-weight function body; a separate Metal
  # coverage unit test must never substitute for its real-weight Metal leg.
  local test_line test_body
  test_line="$(grep -n -m1 -F "fn $TEST_NAME" "$PARITY_SOURCE" | cut -d: -f1)"
  [[ "$test_line" =~ ^[0-9]+$ ]] || die "could not locate Canary-v2 real-weight test body"
  test_body="$(sed -n "${test_line},$((test_line + 180))p" "$PARITY_SOURCE")"
  grep -Fq 'CanaryAsr::from_gguf_with_backend' <<<"$test_body" \
    || die "Canary-v2 test has no explicit real-weight Metal bind; refusing CPU-only PASS"
  grep -Fq 'BackendKind::Metal' <<<"$test_body" \
    || die "Canary-v2 real-weight test lacks a Metal-vs-CPU leg"
  grep -Fq 'assert_eq!(metal_actual, actual, "Canary-v2 Metal IDs must equal CPU");' \
    <<<"$test_body" \
    || die "Canary-v2 test lacks exact Metal-vs-CPU token equality"
  grep -Fq 'CANARY_1B_V2_CPU_VS_OFFICIAL PASS' <<<"$test_body" \
    || die "Canary-v2 test lacks its CPU-vs-official sentinel"
  grep -Fq 'CANARY_1B_V2_METAL_VS_CPU PASS' <<<"$test_body" \
    || die "Canary-v2 test lacks its Metal-vs-CPU sentinel"
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] \
    || die "remote Apple checkout must be clean so evidence names one exact commit"
  xcrun -f metal >/dev/null 2>&1 \
    || die "Xcode Metal compiler is unavailable"
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

hash_reference_directory() {
  local directory="$1" output="$2" path
  find "$directory" -mindepth 1 -maxdepth 1 -type f -print \
    | LC_ALL=C sort \
    | while IFS= read -r path; do
        printf '%s  %s\n' "$(sha256_file "$path")" "${path#"$directory"/}"
      done > "$output"
}

# shellcheck disable=SC2016
run_self_test() (
  local script_path="${BASH_SOURCE[0]}" temporary fail=0 required
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-canary-v2-apple.XXXXXX")"
  trap 'rm -rf -- "$temporary"' EXIT
  printf 'abc' > "$temporary/value"
  [[ "$(sha256_file "$temporary/value")" == \
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" ]] \
    || die "SHA-256 helper self-test failed"
  if require_absent_directory "$temporary/evidence"; then :; else die "absent evidence self-test failed"; fi
  mkdir "$temporary/evidence"
  if require_absent_directory "$temporary/evidence" >/dev/null 2>&1; then die "existing empty evidence accepted"; fi
  for required in \
    'VOKRA_REMOTE_APPLE_SILICON=1' 'Darwin' 'arm64' \
    'MIN_MEMORY_BYTES=32000000000' 'MIN_FREE_DISK_KIB=20000000' \
    'xcrun -f metal' "$GGUF_ENV" "$REFERENCE_PCM_ENV" \
    "$REFERENCE_TOKENS_ENV" "$SOURCE_LANGUAGE_ENV" "$TARGET_LANGUAGE_ENV" \
    'license_preflight' '--approval-evidence' 'preflight_gate.py' \
    'license_gate_manifest.json' "--variant \"\$VARIANT\"" \
    "$TEST_TARGET" '--features metal' '-- --exact --ignored --nocapture' \
    'CanaryAsr::from_gguf_with_backend' 'BackendKind::Metal' \
    'assert_eq!(metal_actual, actual, "Canary-v2 Metal IDs must equal CPU");' \
    'test canary_v2_released_checkpoint_matches_official_nemo_greedy_tokens ... ok' \
    'test result: ok. 1 passed' \
    'CANARY_1B_V2_CPU_VS_OFFICIAL PASS' \
    'CANARY_1B_V2_METAL_VS_CPU PASS' \
    'network=NOT_PERFORMED' 'conversion=NOT_PERFORMED'; do
    if ! grep -Fq -- "$required" "$script_path"; then
      log "self-test FAIL: contract token missing: $required"
      fail=1
    fi
  done
  if grep -En -- '(^|[[:space:]])(curl|wget|python3?|pip|git[[:space:]]+(clone|fetch|pull)|.*(upload|publish))([[:space:]]|$)' \
    "$script_path" | grep -Ev 'UV_NO_CACHE=1 uv run.*python' >/dev/null; then
    log "self-test FAIL: download, direct Python, or publication command found"
    fail=1
  fi
  local gate_pattern='^[[:space:]]*license_preflight "\$approval"[[:space:]]*$'
  local host_pattern='^[[:space:]]*require_remote_apple_host[[:space:]]*$'
  local resource_pattern='^[[:space:]]*require_tooling[[:space:]]*$'
  local checkpoint_pattern='^[[:space:]]*require_file "VAST-generated complete Canary-1B-v2 GGUF" "\$gguf"[[:space:]]*$'
  local scratch_pattern='^[[:space:]]*mkdir -p "\$evidence_dir"[[:space:]]*$'
  local cargo_pattern='^[[:space:]]*run_case en-en'
  if ! production_order_ok "$script_path" "$gate_pattern" "$host_pattern" \
    "$resource_pattern" "$checkpoint_pattern" "$scratch_pattern" "$cargo_pattern"; then
    log 'self-test FAIL: preflight is not before production boundaries'
    fail=1
  fi
  if grep -vE "$gate_pattern" "$script_path" > "$temporary/without-preflight.sh" \
    && production_order_ok "$temporary/without-preflight.sh" "$gate_pattern" "$host_pattern" \
      "$resource_pattern" "$checkpoint_pattern" "$scratch_pattern" "$cargo_pattern"; then
    log 'self-test FAIL: deleted production preflight was accepted'
    fail=1
  fi
  if "$script_path" --self-test --gguf "$temporary/model.gguf" >/dev/null 2>&1; then
    log "self-test FAIL: extra --self-test argument accepted"
    fail=1
  fi
  if "$script_path" --gguf >/dev/null 2>&1; then
    log "self-test FAIL: missing --gguf value accepted"
    fail=1
  fi
  if "$script_path" --unknown-flag >/dev/null 2>&1; then
    log "self-test FAIL: unknown argument accepted"
    fail=1
  fi
  if "$script_path" --self-test --approval-evidence "$temporary/approval.json" >/dev/null 2>&1; then
    log "self-test FAIL: extra approval argument accepted"
    fail=1
  fi
  if "$script_path" --gguf "$temporary/model.gguf" --reference "$temporary/reference" --approval-evidence >/dev/null 2>&1; then
    log "self-test FAIL: missing approval value accepted"
    fail=1
  fi
  if "$script_path" --gguf "$temporary/model.gguf" --reference "$temporary/reference" \
    --approval-evidence "$temporary/a" --approval-evidence "$temporary/b" >/dev/null 2>&1; then
    log "self-test FAIL: duplicate approval accepted"
    fail=1
  fi
  (( fail == 0 )) || return 1
  log "self-test PASS"
)

run_case() {
  local language_pair="$1" source_language="$2" target_language="$3"
  local gguf="$4" reference="$5" log_path="$6"
  env \
    "$GGUF_ENV=$gguf" \
    "$REFERENCE_PCM_ENV=$reference/reference-${language_pair}.pcm.f32" \
    "$REFERENCE_TOKENS_ENV=$reference/reference-${language_pair}.tokens.txt" \
    "$SOURCE_LANGUAGE_ENV=$source_language" \
    "$TARGET_LANGUAGE_ENV=$target_language" \
    CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}" RUST_TEST_THREADS=1 \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --features metal --lib "$TEST_TARGET" \
      -- --exact --ignored --nocapture --test-threads=1 \
      2>&1 | tee "$log_path"
  grep -F "test $TEST_TARGET ... ok" "$log_path" >/dev/null \
    || die "Canary-v2 ${language_pair} exact real-weight test did not report success"
  grep -F 'test result: ok. 1 passed' "$log_path" >/dev/null \
    || die "Canary-v2 ${language_pair} log does not prove one nonzero test ran"
  grep -F 'CANARY_1B_V2_CPU_VS_OFFICIAL PASS' "$log_path" >/dev/null \
    || die "Canary-v2 ${language_pair} CPU-vs-official sentinel is absent"
  grep -F 'CANARY_1B_V2_METAL_VS_CPU PASS' "$log_path" >/dev/null \
    || die "Canary-v2 ${language_pair} Metal-vs-CPU sentinel is absent"
}

main() {
  local gguf='' reference='' approval='' evidence_dir='' self_test=0 seen_gguf=0 seen_reference=0 seen_approval=0 seen_evidence=0
  while (( $# > 0 )); do
    case "$1" in
      --gguf)
        (( seen_gguf == 0 )) && (( $# >= 2 )) && [[ -n "$2" && "$2" != -* ]] || { usage; return 2; }
        seen_gguf=1
        gguf="$2"; shift 2 ;;
      --reference)
        (( seen_reference == 0 )) && (( $# >= 2 )) && [[ -n "$2" && "$2" != -* ]] || { usage; return 2; }
        seen_reference=1
        reference="$2"; shift 2 ;;
      --approval-evidence)
        (( seen_approval == 0 && $# >= 2 )) && [[ -n "$2" && "$2" != -* ]] || { usage; return 2; }
        seen_approval=1
        approval="$2"; shift 2 ;;
      --evidence-dir)
        (( seen_evidence == 0 )) && (( $# >= 2 )) && [[ -n "$2" && "$2" != -* ]] || { usage; return 2; }
        seen_evidence=1
        evidence_dir="$2"; shift 2 ;;
      --self-test)
        self_test=1; shift ;;
      -h|--help)
        usage; return 0 ;;
      *)
        usage; die "unknown argument $1" ;;
    esac
  done

  if (( self_test == 1 )); then
    [[ "$seen_gguf$seen_reference$seen_approval$seen_evidence" == 0000 ]] \
      || die "--self-test accepts no other arguments"
    run_self_test
    return
  fi
  [[ "$seen_gguf$seen_reference$seen_approval$seen_evidence" == 1111 ]] \
    || { usage; die "--gguf, --reference, --approval-evidence and --evidence-dir are required"; }

  # Keep approval ahead of every normal-run host/resource or input operation.
  license_preflight "$approval"
  require_remote_apple_host
  require_tooling
  require_file "VAST-generated complete Canary-1B-v2 GGUF" "$gguf"
  require_reference "$reference"
  require_absent_directory "$evidence_dir"
  require_disjoint_evidence "$evidence_dir" "$gguf" "$reference" "$approval"
  mkdir -p "$evidence_dir"
  record_environment "$evidence_dir/environment.txt"
  {
    echo "gguf=$gguf"
    echo "gguf_sha256=$(sha256_file "$gguf")"
    echo "approval_sha256=$(sha256_file "$approval")"
    hash_reference_directory "$reference" "$evidence_dir/reference-hashes.txt"
  } > "$evidence_dir/input-hashes.txt"

  log "running exact real-weight Canary-v2 ASR CPU/official/Metal comparison"
  run_case en-en en en "$gguf" "$reference" \
    "$evidence_dir/parity-en-en.log"
  log "running exact real-weight Canary-v2 AST CPU/official/Metal comparison"
  run_case en-de en de "$gguf" "$reference" \
    "$evidence_dir/parity-en-de.log"
  grep -F 'CANARY_1B_V2_CPU_VS_OFFICIAL PASS' \
    "$evidence_dir/parity-en-en.log" >/dev/null \
    || die "Canary-v2 ASR CPU/official marker is absent"
  grep -F 'CANARY_1B_V2_METAL_VS_CPU PASS' \
    "$evidence_dir/parity-en-en.log" >/dev/null \
    || die "Canary-v2 ASR Metal/CPU marker is absent"
  grep -F 'CANARY_1B_V2_CPU_VS_OFFICIAL PASS' \
    "$evidence_dir/parity-en-de.log" >/dev/null \
    || die "Canary-v2 AST CPU/official marker is absent"
  grep -F 'CANARY_1B_V2_METAL_VS_CPU PASS' \
    "$evidence_dir/parity-en-de.log" >/dev/null \
    || die "Canary-v2 AST Metal/CPU marker is absent"

  {
    echo "verdict=PASS"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "gguf_sha256=$(sha256_file "$gguf")"
    echo "approval_sha256=$(sha256_file "$approval")"
    echo "asr_cpu_vs_official=PASS"
    echo "asr_metal_vs_cpu=PASS"
    echo "ast_cpu_vs_official=PASS"
    echo "ast_metal_vs_cpu=PASS"
    echo "test=$TEST_TARGET"
    echo "network=NOT_PERFORMED"
    echo "conversion=NOT_PERFORMED"
    echo "upload=NOT_PERFORMED"
    echo "publication=NOT_PERFORMED"
  } > "$evidence_dir/summary.txt"
  log "PASS: pull only $evidence_dir, then remove staged inputs or destroy the remote worker"
}

main "$@"
