#!/usr/bin/env bash
# Real-weight Qwen3-ASR CPU/Metal parity on a disposable remote Apple host.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/qwen3_asr"
PREFLIGHT_GATE="$PARITY_PROJECT/preflight_gate.py"
PREFLIGHT_MANIFEST="$PARITY_PROJECT/license_gate_manifest.json"

MIN_MEMORY_BYTES=32000000000
MIN_FREE_DISK_KIB=20000000
REFERENCE_AUDIO_SHA256="241c0d93cc7ed8792c85c525d1e02b8c33850b791902a5e75b79c2d500e71a1a"

log() { printf '[qwen3-asr-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: apple-silicon-qwen3-asr.sh \
  --gguf-0.6b <path> --gguf-0.6b-sha256 <64-hex> --reference-0.6b <dir> \
  --reference-0.6b-sha256 <64-hex> \
  --gguf-1.7b <path> --gguf-1.7b-sha256 <64-hex> --reference-1.7b <dir> \
  --reference-1.7b-sha256 <64-hex> \
  --approval-evidence <external-approval.json> \
  --evidence-dir <empty-dir>
       apple-silicon-qwen3-asr.sh --self-test

Runs the exact-token and projected-audio CPU/Metal parity test for both pinned
Qwen3-ASR releases. It refuses the 16-GB maintainer class of machine and also
requires VOKRA_REMOTE_APPLE_SILICON=1, Darwin arm64, a clean checkout, at
least 32 GB physical memory, and all four real inputs before Cargo starts.

This script does not download, upload, convert, publish, or delete a model.
Transfer the VAST-produced GGUFs directly to a disposable remote Apple host;
the expected GGUF hashes are mandatory and must be copied from VAST evidence.
Pull only the evidence directory
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

canonicalize_uncreated() {
  local path="$1" suffix='' name parent scan rest component
  [[ ! -L "$path" ]] || return 1
  [[ "$path" == /* ]] || path="$PWD/$path"
  rest="${path#/}"; scan=''
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; rest="${rest#*/}"
    [[ "$component" == "$rest" ]] && rest=''
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || continue
    scan="$scan/$component"; [[ ! -L "$scan" || "$scan" == "/var" ]] || return 1
  done
  while [[ ! -d "$path" || -L "$path" ]]; do
    name="${path##*/}"; [[ -n "$name" ]] && suffix="/$name$suffix"
    parent="${path%/*}"; [[ "$parent" == "$path" ]] && parent='.'
    [[ -n "$parent" ]] || parent='/'; path="$parent"
    [[ ! -L "$path" ]] || return 1
  done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}
paths_overlap() { [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]; }
require_absent_evidence_dir() {
  local target="$1"; shift; local canonical protected other
  [[ ! -e "$target" && ! -L "$target" ]] || { die "evidence directory must be absent and non-symlink: $target"; return 2; }
  canonical="$(canonicalize_uncreated "$target")" || { die "cannot canonicalize evidence directory: $target"; return 2; }
  for protected in "$VOKRA_ROOT" "$PARITY_PROJECT" "$PREFLIGHT_GATE" "$PREFLIGHT_MANIFEST" "$@"; do
    [[ -n "$protected" && ( -e "$protected" || -L "$protected" ) ]] || continue
    [[ ! -L "$protected" ]] || { die "protected input is symlinked: $protected"; return 2; }
    other="$(canonicalize_uncreated "$protected")" || { die "cannot canonicalize protected input: $protected"; return 2; }
    paths_overlap "$canonical" "$other" && { die "evidence directory overlaps protected input: $protected"; return 2; }
  done
  return 0
}

require_file() {
  local label="$1" path="$2"
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] || die "$label is missing, symlinked, or empty: $path"
}

license_preflight() {
  local approval="$1"
  [[ -f "$approval" && -s "$approval" && ! -L "$approval" ]] || die "approval evidence must be a non-empty regular non-symlink file"
  [[ -f "$PARITY_PROJECT/uv.lock" && ! -L "$PARITY_PROJECT/uv.lock" && -f "$PARITY_PROJECT/pyproject.toml" && ! -L "$PARITY_PROJECT/pyproject.toml" && -f "$PREFLIGHT_GATE" && ! -L "$PREFLIGHT_GATE" && -f "$PREFLIGHT_MANIFEST" && ! -L "$PREFLIGHT_MANIFEST" ]] || die "Qwen3-ASR gate inputs are missing or symlinked"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$PREFLIGHT_GATE" --project "$PARITY_PROJECT" --manifest "$PREFLIGHT_MANIFEST" --evidence "$approval"
}

require_reference_manifest_digest() {
  local label="$1" manifest="$2" expected="$3" actual
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || { die "$label expected manifest SHA-256 is missing or malformed"; return 2; }
  require_file "$label manifest" "$manifest"
  actual="$(sha256_file "$manifest")"
  [[ "$actual" == "$expected" ]] || { die "$label manifest SHA-256 $actual != VAST evidence $expected"; return 2; }
}

manifest_value() {
  local manifest="$1" key="$2" count value
  count="$(awk -F= -v key="$key" '$1 == key {count++; value=substr($0, index($0, "=") + 1)} END {print count+0}' "$manifest")"
  [[ "$count" == 1 ]] || die "manifest key is missing or duplicated: $key"
  value="$(awk -F= -v key="$key" '$1 == key {print substr($0, index($0, "=") + 1)}' "$manifest")"
  [[ -n "$value" ]] || die "manifest key is empty: $key"
  printf '%s\n' "$value"
}

verify_manifest_artifact() {
  local manifest="$1" path="$2" filename key expected actual
  filename="$(basename "$path")"
  key="sha256_${filename//./_}"
  expected="$(manifest_value "$manifest" "$key")"
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || die "invalid hash in $manifest: $key"
  actual="$(sha256_file "$path")"
  [[ "$actual" == "$expected" ]] || die "reference artifact hash mismatch: $path"
}

require_reference() {
  local label="$1" directory="$2" variant="$3" repository="$4" revision="$5" manifest
  [[ -d "$directory" && ! -L "$directory" ]] || die "$label is not a directory or is symlinked: $directory"
  manifest="$directory/manifest.txt"
  require_file "$label manifest" "$manifest"
  for artifact in pcm.f32le prompt_ids.u32le audio_embeddings.f32le generated_ids.u32le \
    context.txt forced_language.txt raw_text.txt result_language.txt result_text.txt \
    environment.json source_files.json; do
    require_file "$label $artifact" "$directory/$artifact"
    verify_manifest_artifact "$manifest" "$directory/$artifact"
  done
  [[ "$(manifest_value "$manifest" schema)" == "vokra-qwen3-asr-reference-v1" ]] || die "$label schema drifted"
  [[ "$(manifest_value "$manifest" variant)" == "$variant" ]] || die "$label variant drifted"
  [[ "$(manifest_value "$manifest" upstream_repo)" == "$repository" ]] || die "$label repository drifted"
  [[ "$(manifest_value "$manifest" upstream_revision)" == "$revision" ]] || die "$label revision drifted"
  [[ "$(manifest_value "$manifest" source_audio_sha256)" == "$REFERENCE_AUDIO_SHA256" ]] || die "$label source audio identity drifted"
  [[ "$(manifest_value "$manifest" qwen_asr_version)" == "0.0.6" ]] || die "$label qwen-asr version drifted"
  [[ "$(manifest_value "$manifest" transformers_version)" == "4.57.6" ]] || die "$label transformers version drifted"
  [[ "$(manifest_value "$manifest" sample_rate)" == "16000" ]] || die "$label sample rate drifted"
}

require_one_named_test_passed() {
  local log_path="$1" test_name="$2" marker="$3" test_count result_count marker_count
  test_count="$(grep -Ec "^test ${test_name} \.\.\. ok$" "$log_path" || true)"
  result_count="$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in .+)?$' "$log_path" || true)"
  local total_result_count
  total_result_count="$(grep -Ec '^test result:' "$log_path" || true)"
  marker_count="$(grep -Fxc "$marker" "$log_path" || true)"
  [[ "$test_count" == 1 ]] || { die "expected exactly one passing $test_name, got $test_count"; return 2; }
  [[ "$result_count" == 1 && "$total_result_count" == 1 ]] || { die "expected exactly one exact Cargo result with 1 passed/0 failed/0 ignored"; return 2; }
  [[ "$marker_count" == 1 ]] || { die "expected exactly one full-line parity marker for $test_name, got $marker_count"; return 2; }
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
  # shellcheck disable=SC2016
  grep -Fq 'require_absent_evidence_dir "$evidence_dir" "$gguf_06" "$gguf_17" "$reference_06" "$reference_17" "$approval"' "$0" || return 1
  local temporary
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-qwen3-asr-apple.XXXXXX")"
  trap 'rm -rf "$temporary"' EXIT
  printf 'abc' > "$temporary/value"
  printf '{}\n' > "$temporary/approval.json"
  require_absent_evidence_dir "$temporary/new-evidence" "$temporary/value" "$temporary/approval.json" || die "absent evidence path self-test failed"
  mkdir "$temporary/empty-evidence"
  if require_absent_evidence_dir "$temporary/empty-evidence" "$temporary/value" "$temporary/approval.json" >/dev/null 2>&1; then die "existing empty evidence was accepted"; fi
  rmdir "$temporary/empty-evidence"
  ln -s "$temporary/missing-evidence" "$temporary/link-evidence"
  if require_absent_evidence_dir "$temporary/link-evidence" "$temporary/value" "$temporary/approval.json" >/dev/null 2>&1; then die "evidence symlink was accepted"; fi
  rm "$temporary/link-evidence"
  mkdir -p "$temporary/real-parent/child"
  ln -s "$temporary/real-parent" "$temporary/link-parent"
  if require_absent_evidence_dir "$temporary/link-parent/child/new-evidence" "$temporary/value" "$temporary/approval.json" >/dev/null 2>&1; then die "intermediate evidence symlink was accepted"; fi
  rm -rf "$temporary/real-parent" "$temporary/link-parent"
  if require_absent_evidence_dir "$VOKRA_ROOT/qwen3-asr-apple-self-test" "$temporary/value" "$temporary/approval.json" >/dev/null 2>&1; then die "checkout overlap was accepted"; fi
  if require_absent_evidence_dir "$temporary/approval.json/child" "$temporary/value" "$temporary/approval.json" >/dev/null 2>&1; then die "approval overlap was accepted"; fi
  if require_absent_evidence_dir "$temporary/value/child" "$temporary/value" "$temporary/approval.json" >/dev/null 2>&1; then die "input overlap was accepted"; fi
  [[ "$(sha256_file "$temporary/value")" == \
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" ]] \
    || die "SHA-256 helper self-test failed"
  require_empty_directory "$temporary/evidence"
  [[ -d "$temporary/evidence" ]] || die "directory helper self-test failed"
  printf 'sha256_value=ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\n' > "$temporary/manifest"
  verify_manifest_artifact "$temporary/manifest" "$temporary/value"
  printf 'sha256_value=0000000000000000000000000000000000000000000000000000000000000000\n' > "$temporary/manifest"
  if verify_manifest_artifact "$temporary/manifest" "$temporary/value" >/dev/null 2>&1; then
    die "artifact tamper self-test failed"
  fi
  if require_reference_manifest_digest 'missing digest' "$temporary/missing" "$(printf '%064d' 0)" >/dev/null 2>&1; then
    die "missing reference manifest was accepted"
  fi
  printf '%s\n' \
    'test qwen3_asr_real_metal_matches_cpu_exact_greedy ... ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' \
    'QWEN3_ASR_PARITY qwen3-asr-0.6b Metal_vs_CPU token_ids=exact text=exact PASS' \
    'QWEN3_ASR_PARITY qwen3-asr-1.7b Metal_vs_CPU token_ids=exact text=exact PASS' > "$temporary/parity.log"
  require_one_named_test_passed "$temporary/parity.log" qwen3_asr_real_metal_matches_cpu_exact_greedy \
    'QWEN3_ASR_PARITY qwen3-asr-0.6b Metal_vs_CPU token_ids=exact text=exact PASS'
  require_one_named_test_passed "$temporary/parity.log" qwen3_asr_real_metal_matches_cpu_exact_greedy \
    'QWEN3_ASR_PARITY qwen3-asr-1.7b Metal_vs_CPU token_ids=exact text=exact PASS'
  for malformed in duplicate prefix suffix FAIL; do
    cp "$temporary/parity.log" "$temporary/$malformed.log"
    case "$malformed" in
      duplicate) printf '%s\n' 'QWEN3_ASR_PARITY qwen3-asr-0.6b Metal_vs_CPU token_ids=exact text=exact PASS' >> "$temporary/$malformed.log" ;;
      prefix) sed 's/^QWEN3_ASR_/prefix QWEN3_ASR_/' "$temporary/$malformed.log" > "$temporary/$malformed.tmp" && mv "$temporary/$malformed.tmp" "$temporary/$malformed.log" ;;
      suffix) sed 's/ PASS$/ PASS trailing/' "$temporary/$malformed.log" > "$temporary/$malformed.tmp" && mv "$temporary/$malformed.tmp" "$temporary/$malformed.log" ;;
      FAIL) sed 's/ PASS$/ FAIL/' "$temporary/$malformed.log" > "$temporary/$malformed.tmp" && mv "$temporary/$malformed.tmp" "$temporary/$malformed.log" ;;
    esac
    if require_one_named_test_passed "$temporary/$malformed.log" qwen3_asr_real_metal_matches_cpu_exact_greedy \
      'QWEN3_ASR_PARITY qwen3-asr-0.6b Metal_vs_CPU token_ids=exact text=exact PASS' >/dev/null 2>&1; then
      die "malformed $malformed parity marker was accepted"
    fi
  done
  log "self-test PASS"
)

main() {
  local gguf_06='' gguf_06_digest='' reference_06='' reference_06_digest=''
  local gguf_17='' gguf_17_digest='' reference_17='' reference_17_digest=''
  local approval='' evidence_dir='' self_test=0 seen=''
  while (( $# > 0 )); do
    case "$1" in
      --gguf-*|--reference-*|--evidence-dir|--approval-evidence)
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }
        [[ "$seen" != *"|$1|"* ]] || { usage; return 2; }
        seen+="|$1|" ;;
    esac
    case "$1" in
      --gguf-0.6b)
        [[ $# -ge 2 ]] || { usage; return 2; }
        gguf_06="$2"
        shift 2
        ;;
      --gguf-0.6b-sha256)
        [[ $# -ge 2 ]] || { usage; return 2; }
        gguf_06_digest="$2"
        shift 2
        ;;
      --reference-0.6b)
        [[ $# -ge 2 ]] || { usage; return 2; }
        reference_06="$2"
        shift 2
        ;;
      --reference-0.6b-sha256)
        [[ $# -ge 2 ]] || { usage; return 2; }
        reference_06_digest="$2"
        shift 2
        ;;
      --gguf-1.7b)
        [[ $# -ge 2 ]] || { usage; return 2; }
        gguf_17="$2"
        shift 2
        ;;
      --gguf-1.7b-sha256)
        [[ $# -ge 2 ]] || { usage; return 2; }
        gguf_17_digest="$2"
        shift 2
        ;;
      --reference-1.7b)
        [[ $# -ge 2 ]] || { usage; return 2; }
        reference_17="$2"
        shift 2
        ;;
      --reference-1.7b-sha256)
        [[ $# -ge 2 ]] || { usage; return 2; }
        reference_17_digest="$2"
        shift 2
        ;;
      --evidence-dir)
        [[ $# -ge 2 ]] || { usage; return 2; }
        evidence_dir="$2"
        shift 2
        ;;
      --approval-evidence)
        approval="$2"
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
    [[ -z "$gguf_06$gguf_06_digest$reference_06$reference_06_digest$gguf_17$gguf_17_digest$reference_17$reference_17_digest$approval$evidence_dir" ]] \
      || die "--self-test accepts no other arguments"
    run_self_test
    return
  fi
  [[ -n "$gguf_06" && -n "$gguf_06_digest" && -n "$reference_06" && -n "$reference_06_digest" && \
    -n "$gguf_17" && -n "$gguf_17_digest" && -n "$reference_17" && -n "$reference_17_digest" && -n "$approval" && -n "$evidence_dir" ]] \
    || { usage; die "all two-variant model/reference arguments and --evidence-dir are required"; }
  [[ "$gguf_06_digest" =~ ^[0-9a-f]{64}$ && "$gguf_17_digest" =~ ^[0-9a-f]{64}$ ]] \
    || die "GGUF SHA-256 arguments must be 64 lowercase hex characters"

  license_preflight "$approval"
  require_remote_apple_host
  require_tooling
  require_file "Qwen3-ASR 0.6B GGUF" "$gguf_06"
  require_file "Qwen3-ASR 1.7B GGUF" "$gguf_17"
  [[ "$(sha256_file "$gguf_06")" == "$gguf_06_digest" ]] || die "0.6B GGUF SHA-256 mismatch"
  [[ "$(sha256_file "$gguf_17")" == "$gguf_17_digest" ]] || die "1.7B GGUF SHA-256 mismatch"
  require_reference_manifest_digest "Qwen3-ASR 0.6B" "$reference_06/manifest.txt" "$reference_06_digest"
  require_reference_manifest_digest "Qwen3-ASR 1.7B" "$reference_17/manifest.txt" "$reference_17_digest"
  require_reference "Qwen3-ASR 0.6B reference" "$reference_06" 0.6b \
    "Qwen/Qwen3-ASR-0.6B" 5eb144179a02acc5e5ba31e748d22b0cf3e303b0
  require_reference "Qwen3-ASR 1.7B reference" "$reference_17" 1.7b \
    "Qwen/Qwen3-ASR-1.7B" 7278e1e70fe206f11671096ffdd38061171dd6e5
  require_absent_evidence_dir "$evidence_dir" "$gguf_06" "$gguf_17" "$reference_06" "$reference_17" "$approval"
  mkdir -p "$evidence_dir"
  record_environment "$evidence_dir/environment.txt"

  {
    echo "gguf_0_6b_sha256=$(sha256_file "$gguf_06")"
    echo "gguf_1_7b_sha256=$(sha256_file "$gguf_17")"
    echo "expected_reference_0_6b_manifest_sha256=$reference_06_digest"
    echo "actual_reference_0_6b_manifest_sha256=$(sha256_file "$reference_06/manifest.txt")"
    echo "expected_reference_1_7b_manifest_sha256=$reference_17_digest"
    echo "actual_reference_1_7b_manifest_sha256=$(sha256_file "$reference_17/manifest.txt")"
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
  require_one_named_test_passed "$evidence_dir/parity.log" \
    qwen3_asr_real_metal_matches_cpu_exact_greedy "$marker_06"
  [[ "$(grep -Fxc "$marker_17" "$evidence_dir/parity.log" || true)" == 1 ]] \
    || die "1.7B Metal PASS marker must occur exactly once"

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
