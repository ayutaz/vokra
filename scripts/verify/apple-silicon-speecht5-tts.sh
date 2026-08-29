#!/usr/bin/env bash
# Exact public SpeechT5-TTS CPU/Metal parity on a disposable remote Apple host.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/speecht5_tts"
PREFLIGHT_GATE="$PARITY_PROJECT/preflight_gate.py"
PREFLIGHT_MANIFEST="$PARITY_PROJECT/license_gate_manifest.json"

PUBLIC_GGUF_SHA256="f26019f5e2f7106d834b0b1fd4f66286839e000350caad169388467452c8dde0"
TTS_REVISION="30fcde30f19b87502b8435427b5f5068e401d5f6"
MIN_MEMORY_BYTES=32000000000
MIN_FREE_DISK_KIB=10000000

log() { printf '[speecht5-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: apple-silicon-speecht5-tts.sh \
  --gguf <public-speecht5.gguf> --reference <official-reference-dir> \
  --reference-sha256 <64-hex-from-vast-evidence> \
  --approval-evidence <external-approval.json> \
  --evidence-dir <empty-dir>
       apple-silicon-speecht5-tts.sh --self-test

Runs the official-reference CPU/Metal text-to-mel parity test against the
exact public vokra/speecht5-tts GGUF. It refuses the maintainer class of
machine and requires VOKRA_REMOTE_APPLE_SILICON=1, Darwin arm64, at least
32 GB physical memory, a clean checkout, and pre-staged real inputs.

This script does not download, upload, convert, publish, or delete a model.
Transfer the fixed public GGUF and VAST-produced reference directly to a
disposable remote Apple host. The expected reference.json hash is mandatory
and must come from VAST evidence. Pull only the evidence directory afterward,
then remove staged model data or destroy the remote worker.
EOF
}

sha256_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

license_preflight() {
  local approval="$1"
  [[ -f "$PARITY_PROJECT/uv.lock" && ! -L "$PARITY_PROJECT/uv.lock" && -f "$PARITY_PROJECT/pyproject.toml" && ! -L "$PARITY_PROJECT/pyproject.toml" && -f "$PREFLIGHT_GATE" && ! -L "$PREFLIGHT_GATE" && -f "$PREFLIGHT_MANIFEST" && ! -L "$PREFLIGHT_MANIFEST" ]] || die "SpeechT5 preflight inputs are missing or symlinked"
  [[ -f "$approval" && -s "$approval" && ! -L "$approval" ]] || die "approval evidence must be a non-empty regular non-symlink file"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$PREFLIGHT_GATE" --project "$PARITY_PROJECT" --manifest "$PREFLIGHT_MANIFEST" --evidence "$approval"
}

require_file() {
  local label="$1" path="$2"
  [[ -f "$path" && -s "$path" && ! -L "$path" ]] || die "$label is missing, symlinked, or empty: $path"
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

require_reference() {
  local directory="$1" filename reference_json
  [[ -d "$directory" && ! -L "$directory" ]] || die "reference path is not a directory or is symlinked: $directory"
  reference_json="$directory/reference.json"
  for filename in text.txt tokens.u32 speaker.f32 before_postnet.f32 \
    after_postnet.f32 frames.txt decoder_steps.txt reference.json; do
    require_file "SpeechT5 reference $filename" "$directory/$filename"
  done
  for filename in tokens.u32 speaker.f32 before_postnet.f32 after_postnet.f32; do
    verify_reference_hash "$reference_json" "$directory/$filename"
  done
  grep -F '"format": "vokra-speecht5-tts-reference-v1"' "$reference_json" >/dev/null \
    || die "reference format is not the pinned SpeechT5 schema"
  grep -F '"reference_package": "transformers==5.5.0"' "$reference_json" >/dev/null \
    || die "reference Transformers route is not 5.5.0"
  grep -F "\"upstream_revision\": \"$TTS_REVISION\"" "$reference_json" >/dev/null \
    || die "reference TTS revision is not pinned"
  verify_reference_scalars "$directory"
}

verify_reference_hash() {
  local reference_json="$1" artifact="$2" filename field expected actual field_lines matches
  filename="$(basename "$artifact")"
  case "$filename" in
    tokens.u32) field=tokens_sha256 ;;
    speaker.f32) field=speaker_sha256 ;;
    before_postnet.f32) field=before_postnet_sha256 ;;
    after_postnet.f32) field=after_postnet_sha256 ;;
    *) die "no authenticated reference hash field for $filename" ;;
  esac
  field_lines="$(grep -Ec "^[[:space:]]+\"$field\": " "$reference_json" || true)"
  [[ "$field_lines" == 1 ]] || { die "reference hash field missing or duplicated: $field"; return 2; }
  matches="$(grep -Ec "^[[:space:]]+\"$field\": \"[0-9a-f]{64}\",?$" "$reference_json" || true)"
  [[ "$matches" == 1 ]] || { die "reference hash field is malformed: $field"; return 2; }
  expected="$(grep -E "^[[:space:]]+\"$field\": " "$reference_json" | tr -cd '0-9a-f' | tail -c 64)"
  actual="$(sha256_file "$artifact")"
  [[ "$actual" == "$expected" ]] || { die "reference artifact hash mismatch: $artifact"; return 2; }
}

verify_reference_manifest_digest() {
  local reference_json="$1" expected="$2" actual
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || die "reference SHA-256 must be 64 lowercase hex characters"
  require_file "SpeechT5 reference manifest" "$reference_json"
  actual="$(sha256_file "$reference_json")"
  [[ "$actual" == "$expected" ]] || die "reference.json SHA-256 mismatch"
}

verify_reference_scalars() {
  local directory="$1" reference_json="$1/reference.json"
  local text_lines json_text text_file frames_lines json_frames file_frames
  local steps_lines json_steps file_steps

  text_lines="$(grep -Ec '^[[:space:]]+"text": "[^"\\]*(\\\\.[^"\\\\]*)*",?$' "$reference_json" || true)"
  [[ "$text_lines" == 1 ]] || { die "reference text field missing or duplicated"; return 2; }
  if ! json_text="$(plutil -extract text raw -o - "$reference_json" 2>/dev/null)"; then
    die "reference text field is not valid JSON"
    return 2
  fi
  [[ -n "$json_text" ]] || { die "reference text field is empty or malformed"; return 2; }
  text_file="$(<"$directory/text.txt")"
  [[ "$(wc -l < "$directory/text.txt" | tr -d '[:space:]')" == 1 ]] \
    || { die "reference text.txt must contain exactly one newline-terminated line"; return 2; }
  [[ "$text_file" == "$json_text" ]] \
    || { die "reference text.txt does not match reference.json text"; return 2; }

  frames_lines="$(grep -Ec '^[[:space:]]+"frames": [0-9]+,?$' "$reference_json" || true)"
  [[ "$frames_lines" == 1 ]] || { die "reference frames field missing or duplicated"; return 2; }
  if ! json_frames="$(plutil -extract frames raw -o - "$reference_json" 2>/dev/null)"; then
    die "reference frames field is not valid JSON"
    return 2
  fi
  file_frames="$(<"$directory/frames.txt")"
  [[ "$file_frames" =~ ^[0-9]+$ && "$file_frames" == "$json_frames" ]] \
    || { die "reference frames.txt does not match reference.json frames"; return 2; }

  steps_lines="$(grep -Ec '^[[:space:]]+"decoder_steps": [0-9]+,?$' "$reference_json" || true)"
  [[ "$steps_lines" == 1 ]] || { die "reference decoder_steps field missing or duplicated"; return 2; }
  if ! json_steps="$(plutil -extract decoder_steps raw -o - "$reference_json" 2>/dev/null)"; then
    die "reference decoder_steps field is not valid JSON"
    return 2
  fi
  file_steps="$(<"$directory/decoder_steps.txt")"
  [[ "$file_steps" =~ ^[0-9]+$ && "$file_steps" == "$json_steps" ]] \
    || { die "reference decoder_steps.txt does not match reference.json decoder_steps"; return 2; }
}

require_one_named_test_passed() {
  local log_path="$1" test_name="$2" test_count named_line_count result_count total_result_count
  test_count="$(grep -Ec "^test ${test_name} \.\.\. ok$" "$log_path" || true)"
  named_line_count="$(grep -Ec "^test ${test_name} \.\.\." "$log_path" || true)"
  result_count="$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$log_path" || true)"
  total_result_count="$(grep -Ec '^test result:' "$log_path" || true)"
  [[ "$test_count" == 1 ]] || { die "expected exactly one passing named test, got $test_count"; return 2; }
  [[ "$named_line_count" == 1 ]] || { die "expected exactly one total named test line, got $named_line_count"; return 2; }
  [[ "$result_count" == 1 ]] || { die "expected exactly one Cargo result with 1 passed/0 failed/0 ignored"; return 2; }
  [[ "$total_result_count" == 1 ]] || { die "expected exactly one total Cargo result line, got $total_result_count"; return 2; }
}

require_exact_parity_sentinels() {
  local log_path="$1" cpu_count metal_count
  local cpu_pattern='^SPEECHT5_TTS_OFFICIAL_PARITY backend=cpu frames=[0-9]+ decoder_steps=[0-9]+ before_max_abs=[-+0-9.eE]+ before_index=[0-9]+ after_max_abs=[-+0-9.eE]+ after_index=[0-9]+ bound=[-+0-9.eE]+ verdict=PASS$'
  local metal_pattern='^SPEECHT5_TTS_OFFICIAL_PARITY backend=metal frames=[0-9]+ decoder_steps=[0-9]+ before_max_abs=[-+0-9.eE]+ before_index=[0-9]+ after_max_abs=[-+0-9.eE]+ after_index=[0-9]+ cpu_max_abs=[-+0-9.eE]+ bound=[-+0-9.eE]+ verdict=PASS$'
  cpu_count="$(grep -Ec "$cpu_pattern" "$log_path" || true)"
  metal_count="$(grep -Ec "$metal_pattern" "$log_path" || true)"
  [[ "$cpu_count" == 1 && "$metal_count" == 1 ]] || die "expected exactly one complete CPU and Metal parity sentinel"
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
  for tool in cargo rustc git shasum awk find tee grep plutil sysctl sw_vers \
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
  # shellcheck disable=SC2016
  grep -Fq 'require_absent_evidence_dir "$evidence_dir" "$gguf" "$reference" "$approval"' "$0" || return 1
  local temporary script_path
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-speecht5-apple.XXXXXX")"
  trap 'rm -rf "$temporary"' EXIT
  printf 'abc' > "$temporary/speaker.f32"
  printf '{}\n' > "$temporary/approval.json"
  require_absent_evidence_dir "$temporary/new-evidence" "$temporary/speaker.f32" "$temporary/approval.json" || die "absent evidence path self-test failed"
  mkdir "$temporary/empty-evidence"
  if require_absent_evidence_dir "$temporary/empty-evidence" "$temporary/speaker.f32" "$temporary/approval.json" >/dev/null 2>&1; then die "existing empty evidence was accepted"; fi
  rmdir "$temporary/empty-evidence"
  ln -s "$temporary/missing-evidence" "$temporary/link-evidence"
  if require_absent_evidence_dir "$temporary/link-evidence" "$temporary/speaker.f32" "$temporary/approval.json" >/dev/null 2>&1; then die "evidence symlink was accepted"; fi
  rm "$temporary/link-evidence"
  mkdir -p "$temporary/real-parent/child"
  ln -s "$temporary/real-parent" "$temporary/link-parent"
  if require_absent_evidence_dir "$temporary/link-parent/child/new-evidence" "$temporary/speaker.f32" "$temporary/approval.json" >/dev/null 2>&1; then die "intermediate evidence symlink was accepted"; fi
  rm -rf "$temporary/real-parent" "$temporary/link-parent"
  if require_absent_evidence_dir "$VOKRA_ROOT/speecht5-apple-self-test" "$temporary/speaker.f32" "$temporary/approval.json" >/dev/null 2>&1; then die "checkout overlap was accepted"; fi
  if require_absent_evidence_dir "$temporary/approval.json/child" "$temporary/speaker.f32" "$temporary/approval.json" >/dev/null 2>&1; then die "approval overlap was accepted"; fi
  if require_absent_evidence_dir "$temporary/speaker.f32/child" "$temporary/speaker.f32" "$temporary/approval.json" >/dev/null 2>&1; then die "input overlap was accepted"; fi
  [[ "$(sha256_file "$temporary/speaker.f32")" == \
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" ]] \
    || die "SHA-256 helper self-test failed"
  printf '  "speaker_sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",\n' > "$temporary/reference.json"
  local reference_digest
  reference_digest="$(sha256_file "$temporary/reference.json")"
  verify_reference_manifest_digest "$temporary/reference.json" "$reference_digest"
  if verify_reference_manifest_digest "$temporary/reference.json" "0000000000000000000000000000000000000000000000000000000000000000" >/dev/null 2>&1; then
    die "reference manifest mismatch self-test failed"
  fi
  if verify_reference_manifest_digest "$temporary/missing.json" "$reference_digest" >/dev/null 2>&1; then
    die "missing reference manifest self-test failed"
  fi
  verify_reference_hash "$temporary/reference.json" "$temporary/speaker.f32"
  {
    printf '  "speaker_sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",\n'
    printf '  "speaker_sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "tampered": true,\n'
  } > "$temporary/reference.json"
  if verify_reference_hash "$temporary/reference.json" "$temporary/speaker.f32" >/dev/null 2>&1; then
    die "duplicate reference hash self-test failed"
  fi
  printf '  "speaker_sha256": "0000000000000000000000000000000000000000000000000000000000000000",\n' > "$temporary/reference.json"
  if verify_reference_hash "$temporary/reference.json" "$temporary/speaker.f32" >/dev/null 2>&1; then
    die "reference artifact tamper self-test failed"
  fi
  mkdir "$temporary/scalars"
  printf 'Hello, SpeechT5!\n' > "$temporary/scalars/text.txt"
  printf '2\n' > "$temporary/scalars/frames.txt"
  printf '1\n' > "$temporary/scalars/decoder_steps.txt"
  cat > "$temporary/scalars/reference.json" <<'EOF'
{
  "text": "Hello, SpeechT5!",
  "decoder_steps": 1,
  "frames": 2
}
EOF
  verify_reference_scalars "$temporary/scalars"
  printf '3\n' > "$temporary/scalars/frames.txt"
  if verify_reference_scalars "$temporary/scalars" >/dev/null 2>&1; then
    die "reference scalar tamper self-test failed"
  fi
  local cargo_log
  cargo_log="$temporary/cargo.log"
  printf '%s\n%s\n' \
    'test released_cpu_mel_matches_official_transformers ... ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' \
    > "$cargo_log"
  require_one_named_test_passed "$cargo_log" released_cpu_mel_matches_official_transformers
  printf '%s\n%s\n' \
    'test released_cpu_mel_matches_official_transformers ... ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s' \
    > "$cargo_log"
  require_one_named_test_passed "$cargo_log" released_cpu_mel_matches_official_transformers
  printf '%s\n%s\n' \
    'test released_cpu_mel_matches_official_transformers ... ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; unexpected' \
    > "$cargo_log"
  if require_one_named_test_passed "$cargo_log" released_cpu_mel_matches_official_transformers >/dev/null 2>&1; then
    die "malformed Cargo result suffix self-test failed"
  fi
  printf '%s\n%s\n%s\n' \
    'test released_cpu_mel_matches_official_transformers ... ok' \
    'test result: ok. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' \
    > "$cargo_log"
  if require_one_named_test_passed "$cargo_log" released_cpu_mel_matches_official_transformers >/dev/null 2>&1; then
    die "duplicate Cargo result self-test failed"
  fi
  printf '%s\n%s\n' \
    'test released_cpu_mel_matches_official_transformers ... FAILED' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' \
    > "$cargo_log"
  if require_one_named_test_passed "$cargo_log" released_cpu_mel_matches_official_transformers >/dev/null 2>&1; then
    die "failed named test self-test failed"
  fi
  printf '%s\n%s\n%s\n' \
    'test released_cpu_mel_matches_official_transformers ... ok' \
    'test released_cpu_mel_matches_official_transformers ... ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' \
    > "$cargo_log"
  if require_one_named_test_passed "$cargo_log" released_cpu_mel_matches_official_transformers >/dev/null 2>&1; then
    die "duplicate named test self-test failed"
  fi
  local sentinel_log sentinel_cpu sentinel_metal
  sentinel_log="$temporary/sentinels.log"
  sentinel_cpu='SPEECHT5_TTS_OFFICIAL_PARITY backend=cpu frames=2 decoder_steps=1 before_max_abs=1.000000000e-3 before_index=0 after_max_abs=2.000000000e-3 after_index=1 bound=1.000000000e-2 verdict=PASS'
  sentinel_metal='SPEECHT5_TTS_OFFICIAL_PARITY backend=metal frames=2 decoder_steps=1 before_max_abs=1.000000000e-3 before_index=0 after_max_abs=2.000000000e-3 after_index=1 cpu_max_abs=3.000000000e-3 bound=1.000000000e-2 verdict=PASS'
  printf '%s\n%s\n' "$sentinel_cpu" "$sentinel_metal" > "$sentinel_log"
  require_exact_parity_sentinels "$sentinel_log"
  printf '%s\n%s\n%s\n' "$sentinel_cpu" "$sentinel_cpu" "$sentinel_metal" > "$sentinel_log"
  if require_exact_parity_sentinels "$sentinel_log" >/dev/null 2>&1; then die "duplicate sentinel self-test failed"; fi
  printf 'prefix%s\n%s\n' "$sentinel_cpu" "$sentinel_metal" > "$sentinel_log"
  if require_exact_parity_sentinels "$sentinel_log" >/dev/null 2>&1; then die "prefix sentinel self-test failed"; fi
  printf '%s suffix\n%s\n' "$sentinel_cpu" "$sentinel_metal" > "$sentinel_log"
  if require_exact_parity_sentinels "$sentinel_log" >/dev/null 2>&1; then die "suffix sentinel self-test failed"; fi
  printf '%s\n%s\n' "${sentinel_cpu/verdict=PASS/verdict=FAIL}" "$sentinel_metal" > "$sentinel_log"
  if require_exact_parity_sentinels "$sentinel_log" >/dev/null 2>&1; then die "FAIL sentinel self-test failed"; fi
  require_empty_directory "$temporary/evidence"
  script_path="${BASH_SOURCE[0]}"
  grep -F "$PUBLIC_GGUF_SHA256" "$script_path" >/dev/null \
    || die "public GGUF SHA contract is missing"
  grep -F "SPEECHT5_TTS_OFFICIAL_PARITY backend=metal" "$script_path" >/dev/null \
    || die "Metal PASS marker contract is missing"
  log "self-test PASS"
)

main() {
  local gguf='' reference='' reference_digest='' approval='' evidence_dir='' self_test=0 gguf_sha
  while (( $# > 0 )); do
    case "$1" in
      --gguf)
        [[ $# -ge 2 && -n "$2" && "$2" != -* && -z "$gguf" ]] || { usage; return 2; }
        gguf="$2"
        shift 2
        ;;
      --reference)
        [[ $# -ge 2 && -n "$2" && "$2" != -* && -z "$reference" ]] || { usage; return 2; }
        reference="$2"
        shift 2
        ;;
      --reference-sha256)
        [[ $# -ge 2 && -n "$2" && "$2" != -* && -z "$reference_digest" ]] || { usage; return 2; }
        reference_digest="$2"
        shift 2
        ;;
      --approval-evidence)
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }
        [[ -z "$approval" ]] || { usage; return 2; }
        approval="$2"
        shift 2
        ;;
      --evidence-dir)
        [[ $# -ge 2 && -n "$2" && "$2" != -* && -z "$evidence_dir" ]] || { usage; return 2; }
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
    [[ -z "$gguf$reference$reference_digest$approval$evidence_dir" ]] \
      || die "--self-test accepts no other arguments"
    run_self_test
    return
  fi
  [[ -n "$gguf" && -n "$reference" && -n "$reference_digest" && -n "$approval" && -n "$evidence_dir" ]] \
    || { usage; die "--gguf, --reference, --reference-sha256, --approval-evidence and --evidence-dir are required"; }
  license_preflight "$approval"
  [[ "$reference_digest" =~ ^[0-9a-f]{64}$ ]] \
    || die "--reference-sha256 must be 64 lowercase hex characters"

  require_remote_apple_host
  require_tooling
  require_file "public SpeechT5 GGUF" "$gguf"
  verify_reference_manifest_digest "$reference/reference.json" "$reference_digest"
  require_reference "$reference"
  gguf_sha="$(sha256_file "$gguf")"
  [[ "$gguf_sha" == "$PUBLIC_GGUF_SHA256" ]] \
    || die "GGUF SHA-256 $gguf_sha != exact public artifact $PUBLIC_GGUF_SHA256"
  require_absent_evidence_dir "$evidence_dir" "$gguf" "$reference" "$approval"
  mkdir -p "$evidence_dir"
  record_environment "$evidence_dir/environment.txt"

  {
    echo "public_gguf_sha256=$gguf_sha"
    echo "reference_manifest_sha256=$reference_digest"
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

  require_one_named_test_passed "$evidence_dir/parity.log" \
    released_cpu_mel_matches_official_transformers
  require_exact_parity_sentinels "$evidence_dir/parity.log"

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
