#!/usr/bin/env bash
# Disposable Apple Silicon real-weight CPU/Metal parity for four Qwen3-TTS
# main GGUFs and the VAST-produced official 12-Hz decoder companion.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/qwen3_tts"
LICENSE_GATE="$PARITY_PROJECT/license_gate.py"
LICENSE_MANIFEST="$PARITY_PROJECT/license_gate_manifest.json"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"

MIN_MEMORY_BYTES=32000000000
MIN_FREE_DISK_KIB=40000000
DECODER_REPO="Qwen/Qwen3-TTS-Tokenizer-12Hz"
DECODER_REVISION="a87c50897bb00837eb857d0538b29d117541d7f6"
DECODER_CHECKPOINT_SHA256="836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258"
OFFICIAL_SOURCE_REVISION="022e286b98fbec7e1e916cb940cdf532cd9f488e"
MIN_NEW_TOKENS=2
TRANSFORMERS_VERSION="5.10.4"
TRANSFORMERS_COMPATIBILITY_STATUS="BLOCKED_UNVERIFIED_API_SMOKE"
TEST_SOURCE="$VOKRA_ROOT/crates/vokra-models/tests/qwen3_tts_real.rs"
TEST_NAME="qwen3_tts_real_metal_matches_cpu_and_official_reference"

log() { printf '[qwen3-tts-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }
require_transformers_api_smoke() {
  case "$TRANSFORMERS_COMPATIBILITY_STATUS" in
    AUTHENTICATED_API_SMOKE) ;;
    BLOCKED_UNVERIFIED_API_SMOKE) die 'Transformers API smoke is not authenticated; refusing reference imports and parity' ;;
    *) die "unknown Transformers API smoke status: $TRANSFORMERS_COMPATIBILITY_STATUS" ;;
  esac
}
variant_revision() {
  case "$1" in
    0.6b-base) printf '%s\n' 5d83992436eae1d760afd27aff78a71d676296fc ;;
    0.6b-customvoice) printf '%s\n' 85e237c12c027371202489a0ec509ded67b5e4b5 ;;
    1.7b-base) printf '%s\n' fd4b254389122332181a7c3db7f27e918eec64e3 ;;
    1.7b-customvoice) printf '%s\n' 0c0e3051f131929182e2c023b9537f8b1c68adfe ;;
    *) die "unknown Qwen3-TTS variant: $1"; return 2 ;;
  esac
}

usage() {
  cat <<'EOF' >&2
usage: apple-silicon-qwen3-tts.sh \
  --gguf-0.6b-base <path> --reference-0.6b-base <dir> \
  --gguf-0.6b-base-sha256 <hex> \
  --reference-0.6b-base-sha256 <hex> \
  --gguf-0.6b-customvoice <path> --reference-0.6b-customvoice <dir> \
  --gguf-0.6b-customvoice-sha256 <hex> \
  --reference-0.6b-customvoice-sha256 <hex> \
  --gguf-1.7b-base <path> --reference-1.7b-base <dir> \
  --gguf-1.7b-base-sha256 <hex> \
  --reference-1.7b-base-sha256 <hex> \
  --gguf-1.7b-customvoice <path> --reference-1.7b-customvoice <dir> \
  --gguf-1.7b-customvoice-sha256 <hex> \
  --reference-1.7b-customvoice-sha256 <hex> \
  --decoder-gguf <path> --decoder-gguf-sha256 <hex> --approval-evidence <json> --evidence-dir <empty-dir>
       apple-silicon-qwen3-tts.sh --self-test

Consumes only VAST-staged corrected main/decoder GGUFs and official reference
directories. It requires a disposable Darwin arm64 host, real Metal, a clean
checkout, and VOKRA_REMOTE_APPLE_SILICON=1. It does not download, convert,
upload, publish, or use a CPU fallback for the Metal leg.
EOF
}

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

require_expected_sha256() {
  local label="$1" expected="$2" path="$3" actual
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || die "$label expected SHA-256 is missing or malformed"
  actual="$(sha256_file "$path")"
  [[ "$actual" == "$expected" ]] || die "$label SHA-256 mismatch: expected $expected, got $actual"
}

require_distinct_reference_hashes() {
  local first="$1" second="$2" third="$3" fourth="$4"
  [[ "$first" != "$second" && "$first" != "$third" && "$first" != "$fourth" && "$second" != "$third" && "$second" != "$fourth" && "$third" != "$fourth" ]] || die 'reference manifest SHA-256 values contain a duplicate'
}

verify_reference_hashes() {
  local directory="$1" slug="$2" file key expected actual
  for file in prompt_ids.u32le codes.u32le pcm.f32le environment.json; do
    key="sha256_${file//./_}"
    expected="$(grep -F -- "\"$key\":" "$directory/manifest.json" | sed -E 's/.*"([0-9a-f]{64})".*/\1/')"
    [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || die "$slug manifest lacks $key"
    actual="$(sha256_file "$directory/$file")"
    [[ "$actual" == "$expected" ]] || die "$slug/$file hash differs from manifest"
  done
  if [[ "$slug" == *-base ]]; then
    file=speaker_embedding.f32le; key="sha256_${file//./_}"
    expected="$(grep -F -- "\"$key\":" "$directory/manifest.json" | sed -E 's/.*"([0-9a-f]{64})".*/\1/')"
    [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || die "$slug manifest lacks $key"
    actual="$(sha256_file "$directory/$file")"
    [[ "$actual" == "$expected" ]] || die "$slug/$file hash differs from manifest"
  fi
}

require_file() {
  local label="$1" path="$2"
  [[ -f "$path" && -s "$path" && ! -L "$path" ]] || die "$label is missing, symlinked, or empty: $path"
}

require_empty_dir() {
  local path="$1"
  if [[ -e "$path" ]]; then
    [[ -d "$path" ]] || die "not a directory: $path"
    [[ -z "$(find "$path" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die "directory is not empty: $path"
  else mkdir -p "$path"; fi
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

require_reference() {
  local slug="$1" directory="$2" manifest required repo revision speaker
  case "$slug" in
    0.6b-base) repo='Qwen/Qwen3-TTS-12Hz-0.6B-Base'; revision='5d83992436eae1d760afd27aff78a71d676296fc'; speaker='official_x_vector_only' ;;
    0.6b-customvoice) repo='Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice'; revision='85e237c12c027371202489a0ec509ded67b5e4b5'; speaker='Serena' ;;
    1.7b-base) repo='Qwen/Qwen3-TTS-12Hz-1.7B-Base'; revision='fd4b254389122332181a7c3db7f27e918eec64e3'; speaker='official_x_vector_only' ;;
    1.7b-customvoice) repo='Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'; revision='0c0e3051f131929182e2c023b9537f8b1c68adfe'; speaker='Serena' ;;
    *) die "unknown Qwen3-TTS reference variant: $slug" ;;
  esac
  [[ -d "$directory" && ! -L "$directory" ]] || die "$slug reference directory is missing or symlinked: $directory"
  manifest="$directory/manifest.json"; require_file "$slug manifest" "$manifest"
  for required in \
    '"schema": "vokra-qwen3-tts-reference-v1"' \
    "\"upstream_repo\": \"$repo\"" "\"upstream_revision\": \"$revision\"" \
    "\"model_name\": \"qwen3-tts-12hz-$slug\"" \
    '"official_source_repo": "QwenLM/Qwen3-TTS"' \
    '"official_source_revision": "022e286b98fbec7e1e916cb940cdf532cd9f488e"' \
    '"decoder_repo": "Qwen/Qwen3-TTS-Tokenizer-12Hz"' \
    '"decoder_revision": "a87c50897bb00837eb857d0538b29d117541d7f6"' \
    '"decoder_checkpoint_sha256": "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258"' \
    '"nested_decoder_sha256": "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258"' \
    '"text": "The Vokra parity packet is short and deterministic."' '"language": "English"' \
    "\"speaker\": \"$speaker\"" \
    '"max_new_tokens": 8' '"min_new_tokens": 2' '"sample_rate": 24000' \
    '"qwen_tts_version": "0.1.1"' '"sampling": "greedy"' '"codebooks": 16'; do
    grep -Fq -- "$required" "$manifest" || die "$slug manifest lost $required"
  done
  for file in prompt_ids.u32le codes.u32le pcm.f32le; do require_file "$slug $file" "$directory/$file"; done
  case "$slug" in *-base) require_file "$slug speaker embedding" "$directory/speaker_embedding.f32le" ;; esac
  verify_reference_hashes "$directory" "$slug"
}

require_exact_test_result() {
  local log_file="$1" test_name="$2" test_count result_count result_total
  test_count="$(grep -Ecx "^test ${test_name} \.\.\. ok$" "$log_file" || true)"
  result_count="$(grep -Ecx '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in .+)?$' "$log_file" || true)"
  result_total="$(grep -Ec '^test result:' "$log_file" || true)"
  [[ "$test_count" == 1 && "$result_count" == 1 && "$result_total" == 1 ]] || die "${test_name} did not produce exactly one passing, non-ignored result"
}

require_exact_marker() {
  local log_file="$1" marker="$2" count
  count="$(grep -Fxc "$marker" "$log_file" || true)"
  [[ "$count" == 1 ]] || die "parity marker is missing or duplicated: $marker"
}

require_remote_host() {
  local memory free
  [[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == '1' ]] || die 'VOKRA_REMOTE_APPLE_SILICON=1 is absent'
  [[ "$(uname -s)" == 'Darwin' ]] || die 'real Metal parity requires Darwin'
  [[ "$(uname -m)" == 'arm64' ]] || die 'real Metal parity requires arm64'
  memory="$(sysctl -n hw.memsize)"; [[ "$memory" =~ ^[0-9]+$ ]] || die 'cannot read hw.memsize'
  (( memory >= MIN_MEMORY_BYTES )) || die 'physical memory is below 32-GB guard'
  free="$(df -Pk "$VOKRA_ROOT" | awk 'NR == 2 {print $4}')"; [[ "$free" =~ ^[0-9]+$ ]] || die 'cannot read free disk'
  (( free >= MIN_FREE_DISK_KIB )) || die 'free disk is below 40-GB guard'
}

require_tooling() {
  local tool
  for tool in cargo rustc git shasum awk find grep sed tee sysctl sw_vers system_profiler xcrun wc; do command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"; done
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" && -f "$TEST_SOURCE" ]] || die 'Vokra checkout or real test source is missing'
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'remote Apple checkout must be clean'
  xcrun -f metal >/dev/null 2>&1 || die 'Xcode Metal compiler is unavailable'
}

run_self_test() {
  # shellcheck disable=SC2016
  grep -Fq 'require_absent_evidence_dir "$evidence" "$base06" "$custom06" "$base17" "$custom17" "$decoder" "$approval"' "$0" || return 1
  local script_path="${BASH_SOURCE[0]}" failed=0 required
  local saved_status="$TRANSFORMERS_COMPATIBILITY_STATUS"
  TRANSFORMERS_COMPATIBILITY_STATUS='BLOCKED_UNVERIFIED_API_SMOKE'; require_transformers_api_smoke && failed=1 || :
  TRANSFORMERS_COMPATIBILITY_STATUS='AUTHENTICATED_API_SMOKE'; require_transformers_api_smoke || failed=1
  TRANSFORMERS_COMPATIBILITY_STATUS='UNKNOWN_STATUS'; require_transformers_api_smoke && failed=1 || :
  TRANSFORMERS_COMPATIBILITY_STATUS="$saved_status"
  for required in 'VOKRA_REMOTE_APPLE_SILICON=1' 'Darwin' 'arm64' 'MIN_MEMORY_BYTES=32000000000' 'xcrun -f metal' \
    'Qwen/Qwen3-TTS-Tokenizer-12Hz' 'a87c50897bb00837eb857d0538b29d117541d7f6' \
    '022e286b98fbec7e1e916cb940cdf532cd9f488e' \
    '5d83992436eae1d760afd27aff78a71d676296fc' '85e237c12c027371202489a0ec509ded67b5e4b5' \
    'fd4b254389122332181a7c3db7f27e918eec64e3' '0c0e3051f131929182e2c023b9537f8b1c68adfe' \
    '836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258' 'qwen3_tts_real.rs' "$TEST_NAME" \
    '--features metal --test qwen3_tts_real' '-- --ignored --exact --nocapture' \
    'MIN_NEW_TOKENS=2' 'QWEN3_TTS_0_6B_BASE_GGUF' 'QWEN3_TTS_0_6B_BASE_DECODER_GGUF' 'QWEN3_TTS_0_6B_BASE_REFERENCE_DIR' \
    '--gguf-0.6b-base-sha256' '--gguf-0.6b-customvoice-sha256' '--gguf-1.7b-base-sha256' '--gguf-1.7b-customvoice-sha256' '--reference-0.6b-base-sha256' '--reference-0.6b-customvoice-sha256' '--reference-1.7b-base-sha256' '--reference-1.7b-customvoice-sha256' '--decoder-gguf-sha256' 'require_expected_sha256' 'require_distinct_reference_hashes' \
    "require_exact_test_result \"\$evidence/parity.log\" \"\$TEST_NAME\"" '0 failed; 0 ignored; 0 measured' \
    'QWEN3_TTS_PARITY' 'codes_exact=PASS' 'QWEN3_TTS_METAL_CPU' 'MEASURED_NOT_GATED' 'upload, publish' \
    'TRANSFORMERS_VERSION="5.10.4"' 'previous_isolated_transformers_pin=transformers==4.57.3' \
    'transformers_security_advisory=GHSA-xrqw-3rrv-vx5w' 'transformers_security_patched_minimum=5.10.0' \
    'transformers_compatibility_status=BLOCKED_UNVERIFIED_API_SMOKE' 'require_transformers_api_smoke' \
    'AUTHENTICATED_API_SMOKE' 'UNKNOWN_STATUS'; do
    grep -Fq -- "$required" "$script_path" || { log "self-test missing token: $required"; failed=1; }
  done
  if grep -En -- '(^|[[:space:]])(curl|wget|python3?|pip|git[[:space:]]+(clone|fetch|pull|push)|convert|upload|publish)([[:space:]]|$)' "$script_path" | grep -Ev 'does not download, convert|does not.*upload|does not.*publish|uv run.*python' >/dev/null; then
    log 'self-test found a forbidden external/download operation'; failed=1
  fi
  local forbidden_marker='MEASURED_NOT_GATED'; forbidden_marker+=' PASS'
  if grep -Fq "$forbidden_marker" "$script_path"; then
    log 'self-test found a contradictory full PASS marker'; failed=1
  fi
  local sha_probe sha_expected
  sha_probe="$(mktemp "${TMPDIR:-/tmp}/qwen3-tts-apple-sha-selftest.XXXXXX")"
  printf '%s\n' qwen3-tts-sha-self-test > "$sha_probe"
  sha_expected="$(sha256_file "$sha_probe")"
  if ! require_expected_sha256 self-test "$sha_expected" "$sha_probe"; then failed=1; fi
  if require_expected_sha256 self-test malformed "$sha_probe"; then failed=1; fi
  if require_expected_sha256 self-test "$(printf '%064d' 0)" "$sha_probe"; then failed=1; fi
  if require_expected_sha256 self-test "$sha_expected" "$sha_probe.missing"; then failed=1; fi
  if require_distinct_reference_hashes "$(printf '1%.0s' {1..64})" "$(printf '1%.0s' {1..64})" "$(printf '2%.0s' {1..64})" "$(printf '3%.0s' {1..64})"; then failed=1; fi
  local path_probe
  path_probe="$(mktemp -d "${TMPDIR:-/tmp}/qwen3-tts-apple-path-selftest.XXXXXX")"
  printf '{}\n' > "$path_probe/approval.json"
  printf x > "$path_probe/value"
  require_absent_evidence_dir "$path_probe/new-evidence" "$sha_probe" "$path_probe/approval.json" || failed=1
  mkdir "$path_probe/empty-evidence"
  if require_absent_evidence_dir "$path_probe/empty-evidence" "$sha_probe" "$path_probe/approval.json" >/dev/null 2>&1; then failed=1; fi
  rmdir "$path_probe/empty-evidence"
  ln -s "$path_probe/missing-evidence" "$path_probe/link-evidence"
  if require_absent_evidence_dir "$path_probe/link-evidence" "$sha_probe" "$path_probe/approval.json" >/dev/null 2>&1; then failed=1; fi
  rm "$path_probe/link-evidence"
  mkdir -p "$path_probe/real-parent/child"
  ln -s "$path_probe/real-parent" "$path_probe/link-parent"
  if require_absent_evidence_dir "$path_probe/link-parent/child/new-evidence" "$path_probe/sha-probe" "$path_probe/approval.json" >/dev/null 2>&1; then failed=1; fi
  rm -rf "$path_probe/real-parent" "$path_probe/link-parent"
  if require_absent_evidence_dir "$VOKRA_ROOT/qwen3-tts-apple-self-test" "$sha_probe" "$path_probe/approval.json" >/dev/null 2>&1; then failed=1; fi
  if require_absent_evidence_dir "$path_probe/approval.json/child" "$sha_probe" "$path_probe/approval.json" >/dev/null 2>&1; then failed=1; fi
  if require_absent_evidence_dir "$path_probe/value/child" "$path_probe/value" "$path_probe/approval.json" >/dev/null 2>&1; then failed=1; fi
  rm -rf "$path_probe"
  rm -f "$sha_probe"
  local result_probe duplicate_probe malformed_probe
  result_probe="$(mktemp "${TMPDIR:-/tmp}/qwen3-tts-apple-result-selftest.XXXXXX")"
  printf '%s\n' \
    "test $TEST_NAME ... ok" \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' \
    'QWEN3_TTS_PARITY variant=0.6b-base backend=cpu prompt_ids=exact codes_exact=PASS pcm=MEASURED_NOT_GATED' \
    'QWEN3_TTS_PARITY variant=0.6b-base backend=metal prompt_ids=exact codes_exact=PASS pcm=MEASURED_NOT_GATED' \
    'QWEN3_TTS_METAL_CPU variant=0.6b-base codes_exact=PASS pcm=MEASURED_NOT_GATED' > "$result_probe"
  if ! require_exact_test_result "$result_probe" "$TEST_NAME"; then failed=1; fi
  duplicate_probe="$(mktemp "${TMPDIR:-/tmp}/qwen3-tts-apple-result-duplicate.XXXXXX")"
  cat "$result_probe" "$result_probe" > "$duplicate_probe"
  if require_exact_test_result "$duplicate_probe" "$TEST_NAME"; then failed=1; fi
  malformed_probe="$(mktemp "${TMPDIR:-/tmp}/qwen3-tts-apple-result-malformed.XXXXXX")"
  { sed -n '1p' "$result_probe"; echo 'test result: ok. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out'; sed -n '2,$p' "$result_probe"; } > "$malformed_probe"
  if require_exact_test_result "$malformed_probe" "$TEST_NAME"; then failed=1; fi
  if ! require_exact_marker "$result_probe" 'QWEN3_TTS_PARITY variant=0.6b-base backend=cpu prompt_ids=exact codes_exact=PASS pcm=MEASURED_NOT_GATED'; then failed=1; fi
  if require_exact_marker "$result_probe" 'QWEN3_TTS_PARITY variant=0.6b-customvoice backend=metal prompt_ids=exact codes_exact=PASS pcm=MEASURED_NOT_GATED'; then failed=1; fi
  rm -f "$result_probe" "$duplicate_probe" "$malformed_probe"
  (( failed == 0 )) || return 1
  log 'self-test PASS'
}

main() {
  local base06='' custom06='' base17='' custom17='' ref_base06='' ref_custom06='' ref_base17='' ref_custom17='' decoder='' approval='' evidence='' base06_sha='' custom06_sha='' base17_sha='' custom17_sha='' decoder_sha='' ref_base06_sha='' ref_custom06_sha='' ref_base17_sha='' ref_custom17_sha='' self_test=0 seen=''
  while (( $# > 0 )); do
    case "$1" in
      --gguf-*|--reference-*|--decoder-gguf|--decoder-gguf-sha256|--approval-evidence|--evidence-dir)
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }
        [[ "$seen" != *"|$1|"* ]] || { usage; return 2; }
        seen+="|$1|" ;;
    esac
    case "$1" in
      --gguf-0.6b-base) base06="$2"; shift 2 ;; --gguf-0.6b-base-sha256) base06_sha="$2"; shift 2 ;; --reference-0.6b-base) ref_base06="$2"; shift 2 ;; --reference-0.6b-base-sha256) ref_base06_sha="$2"; shift 2 ;;
      --gguf-0.6b-customvoice) custom06="$2"; shift 2 ;; --gguf-0.6b-customvoice-sha256) custom06_sha="$2"; shift 2 ;; --reference-0.6b-customvoice) ref_custom06="$2"; shift 2 ;; --reference-0.6b-customvoice-sha256) ref_custom06_sha="$2"; shift 2 ;;
      --gguf-1.7b-base) base17="$2"; shift 2 ;; --gguf-1.7b-base-sha256) base17_sha="$2"; shift 2 ;; --reference-1.7b-base) ref_base17="$2"; shift 2 ;; --reference-1.7b-base-sha256) ref_base17_sha="$2"; shift 2 ;;
      --gguf-1.7b-customvoice) custom17="$2"; shift 2 ;; --gguf-1.7b-customvoice-sha256) custom17_sha="$2"; shift 2 ;; --reference-1.7b-customvoice) ref_custom17="$2"; shift 2 ;; --reference-1.7b-customvoice-sha256) ref_custom17_sha="$2"; shift 2 ;;
      --decoder-gguf) decoder="$2"; shift 2 ;; --decoder-gguf-sha256) decoder_sha="$2"; shift 2 ;; --approval-evidence) approval="$2"; shift 2 ;; --evidence-dir) evidence="$2"; shift 2 ;;
      --self-test) self_test=1; shift ;; -h|--help) usage; return 0 ;; *) usage; die "unknown argument: $1" ;;
    esac
  done
  if (( self_test == 1 )); then [[ -z "$base06$custom06$base17$custom17$ref_base06$ref_custom06$ref_base17$ref_custom17$decoder$approval$evidence$base06_sha$custom06_sha$base17_sha$custom17_sha$decoder_sha$ref_base06_sha$ref_custom06_sha$ref_base17_sha$ref_custom17_sha" ]] || die '--self-test accepts no other arguments'; run_self_test; return; fi
  [[ -n "$base06" && -n "$custom06" && -n "$base17" && -n "$custom17" && \
    -n "$ref_base06" && -n "$ref_custom06" && -n "$ref_base17" && -n "$ref_custom17" && \
    -n "$decoder" && -n "$approval" && -n "$evidence" && \
    -n "$base06_sha" && -n "$custom06_sha" && -n "$base17_sha" && -n "$custom17_sha" && \
    -n "$decoder_sha" && -n "$ref_base06_sha" && -n "$ref_custom06_sha" && \
    -n "$ref_base17_sha" && -n "$ref_custom17_sha" ]] \
    || { usage; die 'all four GGUF/reference pairs, nine artifact SHA-256 values, decoder, approval evidence and evidence dir are required'; }
  [[ -f "$approval" && -s "$approval" && ! -L "$approval" ]] || die 'approval evidence must be a non-empty regular non-symlink file'
  require_transformers_api_smoke
  [[ -f "$PARITY_PROJECT/uv.lock" && ! -L "$PARITY_PROJECT/uv.lock" && -f "$PARITY_PROJECT/pyproject.toml" && ! -L "$PARITY_PROJECT/pyproject.toml" && -f "$LICENSE_GATE" && ! -L "$LICENSE_GATE" && -f "$LICENSE_MANIFEST" && ! -L "$LICENSE_MANIFEST" ]] || die 'Qwen3-TTS gate inputs are missing or symlinked'
  local -a gate_args=(--lock "$PARITY_PROJECT/uv.lock" --project "$PARITY_PROJECT/pyproject.toml" --manifest "$LICENSE_MANIFEST" --approval "$approval" --source-revision "$OFFICIAL_SOURCE_REVISION" --decoder-revision "$DECODER_REVISION" --decoder-checkpoint-sha256 "$DECODER_CHECKPOINT_SHA256")
  local variant
  for variant in 0.6b-base 0.6b-customvoice 1.7b-base 1.7b-customvoice; do gate_args+=(--variant-revision "$variant=$(variant_revision "$variant")"); done
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$LICENSE_GATE" "${gate_args[@]}" || return 2
  require_remote_host; require_tooling
  require_file '0.6B Base corrected main GGUF' "$base06"; require_file '0.6B CustomVoice corrected main GGUF' "$custom06"; require_file '1.7B Base corrected main GGUF' "$base17"; require_file '1.7B CustomVoice corrected main GGUF' "$custom17"; require_file 'official decoder GGUF' "$decoder"
  require_expected_sha256 '0.6B Base corrected main GGUF' "$base06_sha" "$base06"; require_expected_sha256 '0.6B CustomVoice corrected main GGUF' "$custom06_sha" "$custom06"; require_expected_sha256 '1.7B Base corrected main GGUF' "$base17_sha" "$base17"; require_expected_sha256 '1.7B CustomVoice corrected main GGUF' "$custom17_sha" "$custom17"; require_expected_sha256 'official decoder GGUF' "$decoder_sha" "$decoder"
  require_file '0.6B Base reference manifest' "$ref_base06/manifest.json"; require_file '0.6B CustomVoice reference manifest' "$ref_custom06/manifest.json"; require_file '1.7B Base reference manifest' "$ref_base17/manifest.json"; require_file '1.7B CustomVoice reference manifest' "$ref_custom17/manifest.json"
  require_expected_sha256 '0.6B Base reference manifest' "$ref_base06_sha" "$ref_base06/manifest.json"; require_expected_sha256 '0.6B CustomVoice reference manifest' "$ref_custom06_sha" "$ref_custom06/manifest.json"; require_expected_sha256 '1.7B Base reference manifest' "$ref_base17_sha" "$ref_base17/manifest.json"; require_expected_sha256 '1.7B CustomVoice reference manifest' "$ref_custom17_sha" "$ref_custom17/manifest.json"
  require_distinct_reference_hashes "$ref_base06_sha" "$ref_custom06_sha" "$ref_base17_sha" "$ref_custom17_sha"
  require_reference 0.6b-base "$ref_base06"; require_reference 0.6b-customvoice "$ref_custom06"; require_reference 1.7b-base "$ref_base17"; require_reference 1.7b-customvoice "$ref_custom17"
  require_absent_evidence_dir "$evidence" "$base06" "$custom06" "$base17" "$custom17" "$decoder" "$approval" \
    "$ref_base06" "$ref_custom06" "$ref_base17" "$ref_custom17"
  mkdir -p "$evidence"
  {
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"; echo "decoder_repo=$DECODER_REPO"; echo "decoder_revision=$DECODER_REVISION"; echo "official_source_revision=$OFFICIAL_SOURCE_REVISION"
    echo 'model_0_6b_base_repo=Qwen/Qwen3-TTS-12Hz-0.6B-Base'; echo 'model_0_6b_base_revision=5d83992436eae1d760afd27aff78a71d676296fc'; echo 'model_0_6b_base_config_bytes=4494'; echo 'model_0_6b_base_config_sha256=2e714c787c8edb98b05432685cddb634add2de4d4e645f653d68251ef72ba011'
    echo 'model_0_6b_customvoice_repo=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice'; echo 'model_0_6b_customvoice_revision=85e237c12c027371202489a0ec509ded67b5e4b5'; echo 'model_0_6b_customvoice_config_bytes=4908'; echo 'model_0_6b_customvoice_config_sha256=81aca2b6fac304944d8acf345272d8a9a727d5fc2e2e66b222ab4729340c7455'
    echo 'model_1_7b_base_repo=Qwen/Qwen3-TTS-12Hz-1.7B-Base'; echo 'model_1_7b_base_revision=fd4b254389122332181a7c3db7f27e918eec64e3'; echo 'model_1_7b_base_config_bytes=4494'; echo 'model_1_7b_base_config_sha256=b4f01752d15a488abde3e1ab44723ae4f4b9e68a4037257b098b3737893cc1f9'
    echo 'model_1_7b_customvoice_repo=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'; echo 'model_1_7b_customvoice_revision=0c0e3051f131929182e2c023b9537f8b1c68adfe'; echo 'model_1_7b_customvoice_config_bytes=4908'; echo 'model_1_7b_customvoice_config_sha256=17a07f527a1c25ea30b4e023a184482a23d3e279d697b1dc81b1bde498d29cf9'
    echo "expected_gguf_0_6b_base_sha256=$base06_sha"; echo "expected_gguf_0_6b_customvoice_sha256=$custom06_sha"; echo "expected_gguf_1_7b_base_sha256=$base17_sha"; echo "expected_gguf_1_7b_customvoice_sha256=$custom17_sha"; echo "expected_decoder_gguf_sha256=$decoder_sha"
    echo "expected_reference_0_6b_base_manifest_sha256=$ref_base06_sha"; echo "expected_reference_0_6b_customvoice_manifest_sha256=$ref_custom06_sha"; echo "expected_reference_1_7b_base_manifest_sha256=$ref_base17_sha"; echo "expected_reference_1_7b_customvoice_manifest_sha256=$ref_custom17_sha"
    echo "gguf_0_6b_base_sha256=$(sha256_file "$base06")"; echo "gguf_0_6b_customvoice_sha256=$(sha256_file "$custom06")"; echo "gguf_1_7b_base_sha256=$(sha256_file "$base17")"; echo "gguf_1_7b_customvoice_sha256=$(sha256_file "$custom17")"; echo "decoder_gguf_sha256=$(sha256_file "$decoder")"
    echo "reference_0_6b_base_manifest_sha256=$(sha256_file "$ref_base06/manifest.json")"; echo "reference_0_6b_customvoice_manifest_sha256=$(sha256_file "$ref_custom06/manifest.json")"; echo "reference_1_7b_base_manifest_sha256=$(sha256_file "$ref_base17/manifest.json")"; echo "reference_1_7b_customvoice_manifest_sha256=$(sha256_file "$ref_custom17/manifest.json")"
    echo 'main_artifacts=VAST_corrected_replacements'; echo 'public_precontract_ggufs=NOT_USED'; echo 'numeric_bound=UNSET'; echo 'verdict=MEASURED_NOT_GATED'; echo 'upload=NOT_PERFORMED'
  } > "$evidence/input-hashes.txt"
  env \
    VOKRA_QWEN3_TTS_0_6B_BASE_GGUF="$base06" VOKRA_QWEN3_TTS_0_6B_BASE_DECODER_GGUF="$decoder" VOKRA_QWEN3_TTS_0_6B_BASE_REFERENCE_DIR="$ref_base06" \
    VOKRA_QWEN3_TTS_0_6B_CUSTOMVOICE_GGUF="$custom06" VOKRA_QWEN3_TTS_0_6B_CUSTOMVOICE_DECODER_GGUF="$decoder" VOKRA_QWEN3_TTS_0_6B_CUSTOMVOICE_REFERENCE_DIR="$ref_custom06" \
    VOKRA_QWEN3_TTS_1_7B_BASE_GGUF="$base17" VOKRA_QWEN3_TTS_1_7B_BASE_DECODER_GGUF="$decoder" VOKRA_QWEN3_TTS_1_7B_BASE_REFERENCE_DIR="$ref_base17" \
    VOKRA_QWEN3_TTS_1_7B_CUSTOMVOICE_GGUF="$custom17" VOKRA_QWEN3_TTS_1_7B_CUSTOMVOICE_DECODER_GGUF="$decoder" VOKRA_QWEN3_TTS_1_7B_CUSTOMVOICE_REFERENCE_DIR="$ref_custom17" \
    RUST_TEST_THREADS=1 cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release -p vokra-models --features metal --test qwen3_tts_real "$TEST_NAME" -- --ignored --exact --nocapture --test-threads=1 2>&1 | tee "$evidence/parity.log"
  require_exact_test_result "$evidence/parity.log" "$TEST_NAME"
  for marker in 'variant=0.6b-base backend=cpu' 'variant=0.6b-base backend=metal' 'variant=0.6b-customvoice backend=cpu' 'variant=0.6b-customvoice backend=metal' 'variant=1.7b-base backend=cpu' 'variant=1.7b-base backend=metal' 'variant=1.7b-customvoice backend=cpu' 'variant=1.7b-customvoice backend=metal'; do require_exact_marker "$evidence/parity.log" "QWEN3_TTS_PARITY $marker prompt_ids=exact codes_exact=PASS pcm=MEASURED_NOT_GATED"; done
  for variant in 0.6b-base 0.6b-customvoice 1.7b-base 1.7b-customvoice; do require_exact_marker "$evidence/parity.log" "QWEN3_TTS_METAL_CPU variant=$variant codes_exact=PASS pcm=MEASURED_NOT_GATED"; done
  { echo 'verdict=MEASURED_NOT_GATED'; echo 'numeric_bound=UNSET'; echo "min_new_tokens=$MIN_NEW_TOKENS"; echo 'previous_isolated_transformers_pin=transformers==4.57.3'; echo 'transformers_security_advisory=GHSA-xrqw-3rrv-vx5w'; echo 'transformers_security_patched_minimum=5.10.0'; echo "isolated_transformers_pin=transformers==$TRANSFORMERS_VERSION"; echo "transformers_compatibility_status=$TRANSFORMERS_COMPATIBILITY_STATUS"; echo 'cpu_reference=MEASURED_NOT_GATED'; echo 'metal_vs_cpu=MEASURED_NOT_GATED'; echo 'public_precontract_ggufs=NOT_USED'; echo 'upload=NOT_PERFORMED'; } > "$evidence/summary.txt"
  log 'MEASURED_NOT_GATED: pull evidence only, then remove staged artifacts or destroy the remote Apple host'
}

main "$@"
