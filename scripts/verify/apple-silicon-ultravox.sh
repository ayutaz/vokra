#!/usr/bin/env bash
# Real-weight Ultravox CPU/Metal parity on a disposable remote Apple host.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/ultravox"
LICENSE_GATE="$PARITY_PROJECT/license_gate.py"
LICENSE_MANIFEST="$PARITY_PROJECT/license_gate_manifest.json"

PUBLIC_BYTES="1366275264"
PUBLIC_SHA256="376c79a7219bb38fc6a857b0bd9ccf57daff878e7bb4723c4801000c0d7b8c9c"
PUBLIC_REPO="vokra/ultravox-v0-5-llama-3-2-1b"
PUBLIC_REVISION="ddbbeec5bfcb09c71a1f88971b794e3e5da811f9"
PUBLIC_FILENAME="ultravox-v0-5-llama-3-2-1b.gguf"
UPSTREAM_REPO="fixie-ai/ultravox-v0_5-llama-3_2-1b"
UPSTREAM_REVISION="b95bec8ab291eeb04b5cd600dd473377f6b79026"
COMPANION_REPO="meta-llama/Llama-3.2-1B-Instruct"
COMPANION_REVISION="9213176726f574b556790deb65791e0c5aa438b6"
MIN_MEMORY_BYTES=24000000000
MIN_FREE_DISK_KIB=12000000
TEST_NAME="ultravox_public_cpu_or_metal_matches_official_reference"

log() { printf '[ultravox-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: apple-silicon-ultravox.sh \
  --gguf <ultravox.gguf> \
  --companion <ultravox-llama-companion.gguf> \
  --companion-sha256 <VAST-recorded-sha256> \
  --reference <VAST-reference-dir> \
  --reference-manifest-sha256 <VAST-recorded-sha256> \
  --approval-evidence <external-approval.json> \
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
  for protected in "$VOKRA_ROOT" "$PARITY_PROJECT" "$LICENSE_GATE" "$LICENSE_MANIFEST" "$@"; do
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
  [[ -f "$LICENSE_GATE" && ! -L "$LICENSE_GATE" && -f "$LICENSE_MANIFEST" && ! -L "$LICENSE_MANIFEST" && -f "$PARITY_PROJECT/uv.lock" && ! -L "$PARITY_PROJECT/uv.lock" && -f "$PARITY_PROJECT/pyproject.toml" && ! -L "$PARITY_PROJECT/pyproject.toml" ]] || die "Ultravox gate inputs are missing or symlinked"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$LICENSE_GATE" --lock "$PARITY_PROJECT/uv.lock" --project "$PARITY_PROJECT/pyproject.toml" --manifest "$LICENSE_MANIFEST" --approval "$approval" --public-repo "$PUBLIC_REPO" --public-revision "$PUBLIC_REVISION" --public-file "$PUBLIC_FILENAME" --public-sha256 "$PUBLIC_SHA256" --upstream-repo "$UPSTREAM_REPO" --upstream-revision "$UPSTREAM_REVISION" --companion-repo "$COMPANION_REPO" --companion-revision "$COMPANION_REVISION" --upstream-model-sha256 f3a3bf7e9137f3219a0d27ba71668deeee8c60aaf0ea587b48d8f71178763f31
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

require_pass_marker() {
  local log_path="$1" marker="$2"
  grep -F "$marker" "$log_path" >/dev/null \
    || die "required PASS sentinel is absent: $marker"
}

require_test_pass() {
  local log_path="$1" backend="$2" marker="$3" test_count result_count marker_count
  test_count="$(grep -Ec "^test ${TEST_NAME} \.\.\. ok$" "$log_path" || true)"
  result_count="$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out($|;)' "$log_path" || true)"
  marker_count="$(grep -Fc "$marker" "$log_path" || true)"
  if [[ "$test_count" != 1 ]]; then
    die "expected exactly one passing $TEST_NAME for $backend, got $test_count"
    return 2
  fi
  if [[ "$result_count" != 1 ]]; then
    die "expected exactly one Cargo result with 1 passed/0 failed/0 ignored for $backend"
    return 2
  fi
  if [[ "$marker_count" != 1 ]]; then
    die "expected exactly one parity marker for $backend, got $marker_count"
    return 2
  fi
}

require_reference() {
  local directory="$1"
  [[ -d "$directory" && ! -L "$directory" ]] || die "reference is not a directory or is symlinked: $directory"
  local name
  for name in manifest.txt pcm.f32le input_features.f32le audio_embeddings.f32le \
    prompt_ids.u32le next_logits.f32le generated_ids.u32le environment.json source_files.json; do
    require_file "reference $name" "$directory/$name"
  done
  local manifest="$directory/manifest.txt"
  for expected in \
    "schema=vokra-ultravox-reference-v1" \
    "public_repo=$PUBLIC_REPO" \
    "public_revision=$PUBLIC_REVISION" \
    "public_filename=$PUBLIC_FILENAME" \
    "public_file_bytes=$PUBLIC_BYTES" \
    "public_file_sha256=$PUBLIC_SHA256" \
    "upstream_repo=$UPSTREAM_REPO" \
    "upstream_revision=$UPSTREAM_REVISION" \
    "companion_repo=$COMPANION_REPO" \
    "companion_revision=$COMPANION_REVISION" \
    "transformers_version=5.5.0"; do
    grep -Fxq "$expected" "$manifest" \
      || die "reference manifest is missing exact identity: $expected"
  done
  for name in pcm.f32le input_features.f32le audio_embeddings.f32le \
    prompt_ids.u32le next_logits.f32le generated_ids.u32le source_files.json environment.json; do
    local key="sha256_${name//./_}" expected_hash actual_hash
    expected_hash="$(awk -F= -v key="$key" '$1 == key {print $2; found=1} END {if (!found) exit 1}' "$manifest")" \
      || die "reference manifest lacks $key"
    [[ "$expected_hash" =~ ^[0-9a-f]{64}$ ]] || die "reference $name hash is invalid"
    actual_hash="$(sha256_file "$directory/$name")"
    [[ "$actual_hash" == "$expected_hash" ]] \
      || die "reference $name differs from its authenticated manifest hash"
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
  # shellcheck disable=SC2016
  grep -Fq 'require_absent_evidence_dir "$evidence_dir" "$gguf" "$companion" "$reference" "$approval"' "$0" || return 1
  local temporary
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-ultravox-apple.XXXXXX")"
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
  if require_absent_evidence_dir "$VOKRA_ROOT/ultravox-apple-self-test" "$temporary/value" "$temporary/approval.json" >/dev/null 2>&1; then die "checkout overlap was accepted"; fi
  if require_absent_evidence_dir "$temporary/approval.json/child" "$temporary/value" "$temporary/approval.json" >/dev/null 2>&1; then die "approval overlap was accepted"; fi
  if require_absent_evidence_dir "$temporary/value/child" "$temporary/value" "$temporary/approval.json" >/dev/null 2>&1; then die "input overlap was accepted"; fi
  [[ "$(sha256_file "$temporary/value")" == \
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" ]] \
    || die "SHA-256 helper self-test failed"
  require_empty_directory "$temporary/evidence"
  [[ -d "$temporary/evidence" ]] || die "directory helper self-test failed"
  printf 'ULTRAVOX_TEST_SENTINEL\n' > "$temporary/sentinel"
  require_pass_marker "$temporary/sentinel" 'ULTRAVOX_TEST_SENTINEL'
  if require_pass_marker "$temporary/sentinel" 'ULTRAVOX_MISSING_SENTINEL' >/dev/null 2>&1; then
    die "self-test accepted a missing PASS sentinel"
  fi
  if require_identity "tampered value" "$temporary/value" "3" \
    "ba7816bf8f01cfea414140de5dae41?" >/dev/null 2>&1; then
    die "self-test accepted a tampered input hash"
  fi
  for backend in CPU Metal; do
    marker="ULTRAVOX_PARITY ${backend}_vs_official frontend_atol=0.01 audio_embeddings_atol=0.01 logits_atol=0.01 greedy_ids=exact PASS"
    printf 'test %s ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n%s\n' \
      "$TEST_NAME" "$marker" > "$temporary/test-$backend.log"
    require_test_pass "$temporary/test-$backend.log" "$backend" "$marker"
  done
  printf 'test %s ... ok\n%s\n' "$TEST_NAME" "$marker" > "$temporary/test-missing-result.log"
  if require_test_pass "$temporary/test-missing-result.log" self-test "$marker" >/dev/null 2>&1; then
    die "self-test accepted a missing exact test result"
  fi
  cp "$temporary/test-CPU.log" "$temporary/test-duplicate.log"
  printf 'test %s ... ok\n' "$TEST_NAME" >> "$temporary/test-duplicate.log"
  if require_test_pass "$temporary/test-duplicate.log" self-test "ULTRAVOX_PARITY CPU_vs_official frontend_atol=0.01 audio_embeddings_atol=0.01 logits_atol=0.01 greedy_ids=exact PASS" >/dev/null 2>&1; then
    die "self-test accepted a duplicate named test"
  fi
  cp "$temporary/test-CPU.log" "$temporary/test-duplicate-result.log"
  grep -F 'test result: ok.' "$temporary/test-CPU.log" >> "$temporary/test-duplicate-result.log"
  if require_test_pass "$temporary/test-duplicate-result.log" self-test "ULTRAVOX_PARITY CPU_vs_official frontend_atol=0.01 audio_embeddings_atol=0.01 logits_atol=0.01 greedy_ids=exact PASS" >/dev/null 2>&1; then
    die "self-test accepted a duplicate test result"
  fi
  cp "$temporary/test-CPU.log" "$temporary/test-duplicate-marker.log"
  printf 'ULTRAVOX_PARITY CPU_vs_official frontend_atol=0.01 audio_embeddings_atol=0.01 logits_atol=0.01 greedy_ids=exact PASS\n' >> "$temporary/test-duplicate-marker.log"
  if require_test_pass "$temporary/test-duplicate-marker.log" self-test "ULTRAVOX_PARITY CPU_vs_official frontend_atol=0.01 audio_embeddings_atol=0.01 logits_atol=0.01 greedy_ids=exact PASS" >/dev/null 2>&1; then
    die "self-test accepted a duplicate parity sentinel"
  fi
  log "self-test PASS"
)

run_parity() {
  local backend="$1" gguf="$2" companion="$3" companion_sha="$4" reference="$5" log_path="$6"
  env \
    VOKRA_ULTRAVOX_GGUF="$gguf" \
    VOKRA_ULTRAVOX_COMPANION_GGUF="$companion" \
    VOKRA_ULTRAVOX_COMPANION_GGUF_SHA256="$companion_sha" \
    VOKRA_ULTRAVOX_REFERENCE_DIR="$reference" \
    VOKRA_ULTRAVOX_BACKEND="$backend" \
    CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}" \
    RUST_TEST_THREADS=1 \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --features metal --test ultravox_real \
      "$TEST_NAME" -- --exact --nocapture \
      2>&1 | tee "$log_path"
}

main() {
  local gguf='' companion='' companion_sha='' reference='' reference_manifest_sha='' approval='' evidence_dir='' self_test=0 seen=''
  while (( $# > 0 )); do
    case "$1" in
      --gguf|--companion|--companion-sha256|--reference|--reference-manifest-sha256|--approval-evidence|--evidence-dir)
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }
        [[ "$seen" != *"|$1|"* ]] || { usage; return 2; }
        seen+="|$1|" ;;
    esac
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
      --reference-manifest-sha256)
        [[ $# -ge 2 ]] || { usage; return 2; }
        reference_manifest_sha="$2"
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
    [[ -z "$gguf$companion$companion_sha$reference$reference_manifest_sha$approval$evidence_dir" ]] \
      || die "--self-test accepts no other arguments"
    run_self_test
    return
  fi
  [[ -n "$gguf" && -n "$companion" && -n "$companion_sha" && -n "$reference" && \
    -n "$reference_manifest_sha" && -n "$approval" && -n "$evidence_dir" ]] \
    || { usage; die "all six input options are required"; }
  [[ "$companion_sha" =~ ^[0-9a-f]{64}$ ]] \
    || die "--companion-sha256 must be 64 lowercase hex characters"
  [[ "$reference_manifest_sha" =~ ^[0-9a-f]{64}$ ]] \
    || die "--reference-manifest-sha256 must be 64 lowercase hex characters"

  license_preflight "$approval"
  require_remote_apple_host
  require_tooling
  require_identity "Ultravox GGUF" "$gguf" "$PUBLIC_BYTES" "$PUBLIC_SHA256"
  require_file "Ultravox Llama companion GGUF" "$companion"
  [[ "$(sha256_file "$companion")" == "$companion_sha" ]] \
    || die "companion GGUF SHA-256 differs from the VAST-recorded identity"
  require_reference "$reference"
  [[ "$(sha256_file "$reference/manifest.txt")" == "$reference_manifest_sha" ]] \
    || die "reference manifest SHA-256 differs from the VAST-recorded identity"
  require_absent_evidence_dir "$evidence_dir" "$gguf" "$companion" "$reference" "$approval"
  mkdir -p "$evidence_dir"
  record_environment "$evidence_dir/environment.txt"

  {
    echo "gguf_sha256=$(sha256_file "$gguf")"
    echo "companion_sha256=$(sha256_file "$companion")"
    echo "reference_manifest_sha256=$reference_manifest_sha"
  } > "$evidence_dir/input-hashes.txt"

  log "running real-weight Apple CPU parity against official reference"
  run_parity cpu "$gguf" "$companion" "$companion_sha" "$reference" "$evidence_dir/parity-cpu.log"
  require_test_pass "$evidence_dir/parity-cpu.log" CPU \
    'ULTRAVOX_PARITY Cpu_vs_official frontend_atol=0.01 audio_embeddings_atol=0.01 logits_atol=0.01 greedy_ids=exact PASS'

  log "running real-weight Metal parity against official reference"
  run_parity metal "$gguf" "$companion" "$companion_sha" "$reference" "$evidence_dir/parity-metal.log"
  require_test_pass "$evidence_dir/parity-metal.log" Metal \
    'ULTRAVOX_PARITY Metal_vs_official frontend_atol=0.01 audio_embeddings_atol=0.01 logits_atol=0.01 greedy_ids=exact PASS'

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
