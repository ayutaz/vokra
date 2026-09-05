#!/usr/bin/env bash
# Real-weight NeuTTS Air CPU/Metal parity on a disposable remote Apple host.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/neutts_air"
PREFLIGHT_GATE="$PARITY_PROJECT/preflight_gate.py"
PREFLIGHT_MANIFEST="$PARITY_PROJECT/license_gate_manifest.json"

PUBLIC_BYTES="1495883328"
PUBLIC_SHA256="f6caf559e919b16d77ac28177e59ee5427a5de92bdeedd719ecab00b4afbb754"
COMPANION_BYTES="1025417504"
COMPANION_SHA256="15e60e7e5f7242255b18e1386b26c2a8f872c77a56ca241ee82c8aa5d8b6327f"
UPSTREAM_REPO="neuphonic/neutts-air"
UPSTREAM_REVISION="3b58b776406b62fdc137e31ea53d728f5c22a4ed"
SOURCE_REPO="https://github.com/neuphonic/neutts.git"
SOURCE_REVISION="3e9415df12633f8a74ac6f92418c7cd5c8c4bf0e"
SOURCE_PATH="neuttsair/neutts.py"
SOURCE_SHA256="e68b87dae6718903337a08eff56afbd58ba261d829624ea5a00a343c8cefb7c1"
SOURCE_BYTES="9035"
MIN_MEMORY_BYTES=24000000000
MIN_FREE_DISK_KIB=12000000

log() { printf '[neutts-air-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: apple-silicon-neutts-air.sh \
  --gguf <neutts-air.gguf> \
  --gguf-sha256 <64-hex> \
  --companion <distill-neucodec.gguf> \
  --companion-sha256 <64-hex> \
  --reference <VAST-reference-dir> \
  --reference-sha256 <manifest.txt-64-hex> \
  --approval-evidence <file> \
  --evidence-dir <absent-dir>
       apple-silicon-neutts-air.sh --self-test

Runs the exact public NeuTTS Air and Distill NeuCodec pair first on Apple CPU
and then Metal against the same VAST-produced official reference. It refuses
the maintainer's 16-GB machine class and requires
VOKRA_REMOTE_APPLE_SILICON=1, Darwin arm64, a clean checkout, at least 24 GB
physical memory, and all authenticated inputs before Cargo starts.

This script does not download, upload, convert, publish, or delete a model.
The evidence directory must be absent/nonexistent before validation and is created only after approval and input checks. Transfer model data directly from VAST to a disposable Apple host. Pull only
the evidence directory, then delete staged data or destroy the worker.
EOF
}

sha256_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

license_preflight() {
  local approval="$1"
  command -v uv >/dev/null 2>&1 || die "required tool missing: uv"
  [[ -f "$PREFLIGHT_GATE" && ! -L "$PREFLIGHT_GATE" && -f "$PREFLIGHT_MANIFEST" && ! -L "$PREFLIGHT_MANIFEST" ]] || die "NeuTTS Air preflight inputs are missing"
  [[ -f "$approval" && ! -L "$approval" && -s "$approval" ]] || die "approval evidence is missing, symlinked, or empty"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$PREFLIGHT_GATE" --project "$PARITY_PROJECT" --manifest "$PREFLIGHT_MANIFEST" --evidence "$approval"
}

file_bytes() {
  wc -c < "$1" | tr -d '[:space:]'
}

require_disjoint_evidence() {
  local evidence="$1" parent candidate other other_parent real
  [[ ! -e "$evidence" && ! -L "$evidence" ]] || { die "evidence directory must be absent before validation"; return 2; }
  parent="$(cd -P "$(dirname "$evidence")" 2>/dev/null && pwd)" || { die "evidence parent is inaccessible"; return 2; }
  candidate="$parent/$(basename "$evidence")"
  shift
  for other in "$@"; do
    [[ ! -L "$other" ]] || { die "validation input is a symlink: $other"; return 2; }
    other_parent="$(cd -P "$(dirname "$other")" 2>/dev/null && pwd)" || { die "validation input is inaccessible: $other"; return 2; }
    real="$other_parent/$(basename "$other")"
    [[ "$candidate" != "$real" && "$candidate/" != "$real/"* && "$real/" != "$candidate/"* ]] || { die "evidence directory overlaps validation input"; return 2; }
  done
  mkdir -p "$evidence"
}

require_file() {
  local label="$1" path="$2"
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] || die "$label is missing, symlinked, or empty: $path"
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

manifest_value() {
  local manifest="$1" key="$2" value count
  count="$(awk -F= -v key="$key" '$1 == key {count++} END {print count + 0}' "$manifest")"
  [[ "$count" == 1 ]] || return 1
  value="$(awk -F= -v key="$key" '$1 == key {print substr($0, length(key) + 2)}' "$manifest")"
  [[ -n "$value" ]] || return 1
  printf '%s\n' "$value"
}

require_reference_file_set() {
  local directory="$1" entry name
  local expected_names='manifest.txt prompt_ids.u32le next_logits.f32le generated_ids.u32le source_files.json environment.json'
  [[ -d "$directory" ]] || die "reference is not a directory: $directory"
  while IFS= read -r entry; do
    name="$(basename "$entry")"
    [[ -f "$entry" && ! -L "$entry" ]] \
      || { die "reference output is not a regular non-symlink file: $name"; return 2; }
    case " $expected_names " in
      *" $name "*) ;;
      *) die "unexpected reference output file: $name"; return 2 ;;
    esac
  done < <(find -P "$directory" -mindepth 1 -maxdepth 1 -print)
  for name in $expected_names; do
    [[ -f "$directory/$name" && ! -L "$directory/$name" ]] \
      || { die "missing regular reference output: $name"; return 2; }
  done
}

require_reference_manifest() {
  local directory="$1"
  local manifest="$directory/manifest.txt" value
  require_file "reference manifest" "$manifest"
  local key
  for key in schema upstream_repo upstream_revision source_repo source_revision source_path source_bytes source_sha256 \
    source_config_sha256 source_weights_sha256 source_tokenizer_sha256 \
    transformers_version torch_version vocab_size prompt_tokens prompt_ids_csv \
    generated_tokens max_new_tokens reference_codes config_model_type \
    sha256_prompt_ids_u32le sha256_next_logits_f32le sha256_generated_ids_u32le \
    sha256_source_files_json sha256_environment_json; do
    value="$(manifest_value "$manifest" "$key")" \
      || die "reference manifest key is missing/duplicated/empty: $key"
  done
  [[ "$(manifest_value "$manifest" schema)" == "vokra-neutts-air-reference-v1" ]] \
    || die "reference schema drifted"
  [[ "$(manifest_value "$manifest" upstream_repo)" == "$UPSTREAM_REPO" && \
    "$(manifest_value "$manifest" upstream_revision)" == "$UPSTREAM_REVISION" ]] \
    || die "gated upstream identity drifted"
  [[ "$(manifest_value "$manifest" source_repo)" == "$SOURCE_REPO" && \
    "$(manifest_value "$manifest" source_revision)" == "$SOURCE_REVISION" && \
    "$(manifest_value "$manifest" source_path)" == "$SOURCE_PATH" && \
    "$(manifest_value "$manifest" source_bytes)" == "$SOURCE_BYTES" && \
    "$(manifest_value "$manifest" source_sha256)" == "$SOURCE_SHA256" ]] \
    || die "official source identity drifted"
  [[ "$(manifest_value "$manifest" transformers_version)" == "5.5.0" && \
    "$(manifest_value "$manifest" torch_version)" == "2.13.0+cpu" && \
    "$(manifest_value "$manifest" config_model_type)" == "qwen2" && \
    "$(manifest_value "$manifest" vocab_size)" == "217652" && \
    "$(manifest_value "$manifest" reference_codes)" == "0,1,2,3,7,31,255,1023" ]] \
    || die "reference runtime/contract identity drifted"
  for key in sha256_prompt_ids_u32le sha256_next_logits_f32le sha256_generated_ids_u32le sha256_source_files_json sha256_environment_json; do
    [[ "$(manifest_value "$manifest" "$key")" =~ ^[0-9a-f]{64}$ ]] \
      || die "reference artifact hash is malformed: $key"
  done
  local artifact expected actual
  for artifact in prompt_ids.u32le next_logits.f32le generated_ids.u32le source_files.json environment.json; do
    key="sha256_${artifact//./_}"
    expected="$(manifest_value "$manifest" "$key")"
    actual="$(sha256_file "$directory/$artifact")"
    [[ "$actual" == "$expected" ]] || die "reference artifact SHA-256 mismatch: $artifact"
  done
  UV_CACHE_DIR="${NEUTTS_AIR_UV_CACHE_DIR:-/private/tmp/vokra-neutts-air-uv-cache}" \
    uv run --no-project --offline --python 3.12 python - "$directory/source_files.json" "$manifest" <<'PY'
import json, re, sys
from pathlib import Path
source = json.loads(Path(sys.argv[1]).read_text())
if set(source) != {"config.json", "generation_config.json", "model.safetensors", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "neuttsair/neutts.py"}:
    raise SystemExit("source_files.json identity set drifted")
for name, row in source.items():
    if not isinstance(row, dict) or not isinstance(row.get("bytes"), int) or row["bytes"] <= 0 or not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", "")):
        raise SystemExit(f"malformed source_files.json row: {name}")
if source["neuttsair/neutts.py"] != {"bytes": 9035, "sha256": "e68b87dae6718903337a08eff56afbd58ba261d829624ea5a00a343c8cefb7c1"}:
    raise SystemExit("official source file identity drifted")
values = {}
for line in Path(sys.argv[2]).read_text().splitlines():
    key, sep, value = line.partition("=")
    if sep:
        values[key] = value
for manifest_key, name in (("source_config_sha256", "config.json"), ("source_weights_sha256", "model.safetensors"), ("source_tokenizer_sha256", "tokenizer.json")):
    if values.get(manifest_key) != source[name]["sha256"]:
        raise SystemExit(f"manifest/source_files hash mismatch: {name}")
PY
}

require_reference() {
  local directory="$1"
  require_reference_file_set "$directory"
  require_reference_manifest "$directory"
  require_file "reference prompt" "$directory/prompt_ids.u32le"
  require_file "reference logits" "$directory/next_logits.f32le"
  require_file "reference generated ids" "$directory/generated_ids.u32le"
}

require_one_named_test_passed() {
  local log_path="$1" test_name="$2" backend="$3"
  local test_count named_line_count result_count total_result_count parity_count composition_count
  test_count="$(grep -Ec "^test ${test_name} \.\.\. ok$" "$log_path" || true)"
  named_line_count="$(grep -Ec "^test ${test_name} \.\.\." "$log_path" || true)"
  result_count="$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$log_path" || true)"
  total_result_count="$(grep -Ec '^test result:' "$log_path" || true)"
  parity_count="$(grep -Fxc "NEUTTS_AIR_PARITY ${backend}_vs_official logits_atol=0.01 greedy_ids=exact PASS" "$log_path" || true)"
  composition_count="$(grep -Exc "NEUTTS_AIR_COMPOSITION ${backend} codes=[0-9]+ samples=[0-9]+ PASS" "$log_path" || true)"
  [[ "$test_count" == 1 ]] || { die "expected exactly one passing $test_name, got $test_count"; return 2; }
  [[ "$named_line_count" == 1 ]] || { die "expected exactly one total $test_name result line, got $named_line_count"; return 2; }
  [[ "$result_count" == 1 && "$total_result_count" == 1 ]] || { die "expected exactly one exact Cargo result"; return 2; }
  [[ "$parity_count" == 1 ]] || { die "expected exactly one full-line ${backend} parity marker"; return 2; }
  [[ "$composition_count" == 1 ]] || { die "expected exactly one full-line ${backend} composition marker"; return 2; }
  ! grep -Eq '^NEUTTS_AIR_(PARITY|COMPOSITION).*FAIL$' "$log_path" \
    || { die "a NEUTTS_AIR FAIL marker is present"; return 2; }
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
  for tool in cargo rustc git shasum awk find tee grep sysctl sw_vers system_profiler xcrun wc tr uv; do
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
  local temporary malformed
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-neutts-air-apple.XXXXXX")"
  trap 'rm -rf "$temporary"' EXIT
  printf 'abc' > "$temporary/value"
  [[ "$(sha256_file "$temporary/value")" == \
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" ]] \
    || die "SHA-256 helper self-test failed"
  mkdir "$temporary/evidence"
  if require_disjoint_evidence "$temporary/evidence" "$VOKRA_ROOT" >/dev/null 2>&1; then
    die "existing evidence directory was accepted"
  fi
  rmdir "$temporary/evidence"
  require_disjoint_evidence "$temporary/evidence" "$VOKRA_ROOT"
  [[ -d "$temporary/evidence" ]] || die "directory helper self-test failed"
  mkdir "$temporary/reference"
  for name in manifest.txt prompt_ids.u32le next_logits.f32le generated_ids.u32le source_files.json environment.json; do
    : > "$temporary/reference/$name"
  done
  require_reference_file_set "$temporary/reference"
  : > "$temporary/reference/unexpected.bin"
  if require_reference_file_set "$temporary/reference" >/dev/null 2>&1; then
    die "extra reference output was accepted"
  fi
  rm "$temporary/reference/unexpected.bin"
  printf '%s\n' \
    'test neutts_air_public_cpu_or_metal_matches_official_reference ... ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s' \
    'NEUTTS_AIR_PARITY Metal_vs_official logits_atol=0.01 greedy_ids=exact PASS' \
    'NEUTTS_AIR_COMPOSITION Metal codes=8 samples=3840 PASS' > "$temporary/valid.log"
  require_one_named_test_passed "$temporary/valid.log" neutts_air_public_cpu_or_metal_matches_official_reference Metal
  for malformed in duplicate_named duplicate_result duplicate_marker prefix suffix result_suffix FAIL; do
    cp "$temporary/valid.log" "$temporary/$malformed.log"
    case "$malformed" in
      duplicate_named) printf '%s\n' 'test neutts_air_public_cpu_or_metal_matches_official_reference ... FAILED' >> "$temporary/$malformed.log" ;;
      duplicate_result) printf '%s\n' 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' >> "$temporary/$malformed.log" ;;
      duplicate_marker) printf '%s\n' 'NEUTTS_AIR_COMPOSITION Metal codes=8 samples=3840 PASS' >> "$temporary/$malformed.log" ;;
      prefix) sed 's/^NEUTTS_AIR_/prefix NEUTTS_AIR_/' "$temporary/$malformed.log" > "$temporary/$malformed.tmp" && mv "$temporary/$malformed.tmp" "$temporary/$malformed.log" ;;
      suffix) sed 's/ PASS$/ PASS trailing/' "$temporary/$malformed.log" > "$temporary/$malformed.tmp" && mv "$temporary/$malformed.tmp" "$temporary/$malformed.log" ;;
      result_suffix) sed 's/filtered out; finished in 0.01s$/filtered out; finished in nonsense/' "$temporary/$malformed.log" > "$temporary/$malformed.tmp" && mv "$temporary/$malformed.tmp" "$temporary/$malformed.log" ;;
      FAIL) sed 's/ PASS$/ FAIL/' "$temporary/$malformed.log" > "$temporary/$malformed.tmp" && mv "$temporary/$malformed.tmp" "$temporary/$malformed.log" ;;
    esac
    if require_one_named_test_passed "$temporary/$malformed.log" neutts_air_public_cpu_or_metal_matches_official_reference Metal >/dev/null 2>&1; then
      die "malformed $malformed marker was accepted"
    fi
  done
  # shellcheck disable=SC2086 # Each case intentionally models argv tokenization.
  for bad_args in "--self-test --approval-evidence x" "--self-test --self-test" "--gguf x --gguf y" "--approval-evidence" "--unknown x"; do
    if bash "$0" $bad_args >/dev/null 2>&1; then die "accepted malformed parser case: $bad_args"; fi
  done
  log "self-test PASS"
)

run_parity() {
  local backend="$1" gguf="$2" companion="$3" reference="$4" log_path="$5"
  env \
    VOKRA_NEUTTS_AIR_GGUF="$gguf" \
    VOKRA_NEUTTS_AIR_REFERENCE_DIR="$reference" \
    VOKRA_NEUTTS_AIR_COMPANION_GGUF="$companion" \
    VOKRA_NEUTTS_AIR_BACKEND="$backend" \
    CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}" \
    RUST_TEST_THREADS=1 \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --features metal --test neutts_air_real \
      neutts_air_public_cpu_or_metal_matches_official_reference -- --exact --nocapture \
      2>&1 | tee "$log_path"
}

main() {
  local gguf='' gguf_digest='' companion='' companion_digest='' reference='' reference_digest='' approval='' evidence_dir='' self_test=0
  local seen_gguf=0 seen_gguf_digest=0 seen_companion=0 seen_companion_digest=0 seen_reference=0 seen_reference_digest=0 seen_approval=0 seen_evidence=0 seen_self_test=0
  while (( $# > 0 )); do
    case "$1" in
      --gguf)
        (( seen_gguf == 0 )) || die 'duplicate --gguf'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--gguf requires a nonempty value'; seen_gguf=1
        gguf="$2"
        shift 2
        ;;
      --gguf-sha256)
        (( seen_gguf_digest == 0 )) || die 'duplicate --gguf-sha256'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--gguf-sha256 requires a nonempty value'; seen_gguf_digest=1
        gguf_digest="$2"
        shift 2
        ;;
      --companion)
        (( seen_companion == 0 )) || die 'duplicate --companion'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--companion requires a nonempty value'; seen_companion=1
        companion="$2"
        shift 2
        ;;
      --companion-sha256)
        (( seen_companion_digest == 0 )) || die 'duplicate --companion-sha256'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--companion-sha256 requires a nonempty value'; seen_companion_digest=1
        companion_digest="$2"
        shift 2
        ;;
      --reference)
        (( seen_reference == 0 )) || die 'duplicate --reference'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--reference requires a nonempty value'; seen_reference=1
        reference="$2"
        shift 2
        ;;
      --reference-sha256)
        (( seen_reference_digest == 0 )) || die 'duplicate --reference-sha256'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--reference-sha256 requires a nonempty value'; seen_reference_digest=1
        reference_digest="$2"
        shift 2
        ;;
      --evidence-dir)
        (( seen_evidence == 0 )) || die 'duplicate --evidence-dir'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--evidence-dir requires a nonempty value'; seen_evidence=1
        evidence_dir="$2"
        shift 2
        ;;
      --self-test)
        (( seen_self_test == 0 )) || die 'duplicate --self-test'; seen_self_test=1
        self_test=1
        shift
        ;;
      --approval-evidence)
        (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--approval-evidence requires a nonempty value'; seen_approval=1
        approval="$2"; shift 2 ;;
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
    [[ -z "$gguf$gguf_digest$companion$companion_digest$reference$reference_digest$approval$evidence_dir" ]] \
      || die "--self-test accepts no other arguments"
    run_self_test
    return
  fi
  [[ -n "$gguf" && -n "$gguf_digest" && -n "$companion" && -n "$companion_digest" && -n "$reference" && -n "$reference_digest" && -n "$approval" && -n "$evidence_dir" ]] \
    || { usage; die "all model/reference hashes, --approval-evidence, and --evidence-dir are required"; }
  [[ "$gguf_digest" =~ ^[0-9a-f]{64}$ && "$companion_digest" =~ ^[0-9a-f]{64}$ && "$reference_digest" =~ ^[0-9a-f]{64}$ ]] \
    || die "all expected SHA-256 values must be lowercase 64-hex"
  [[ "$gguf_digest" == "$PUBLIC_SHA256" ]] || die "expected public GGUF hash is not the fixed identity"
  [[ "$companion_digest" == "$COMPANION_SHA256" ]] || die "expected companion hash is not the fixed identity"

  license_preflight "$approval"
  require_remote_apple_host
  require_tooling
  require_identity "NeuTTS Air GGUF" "$gguf" "$PUBLIC_BYTES" "$gguf_digest"
  require_identity "Distill NeuCodec GGUF" "$companion" \
    "$COMPANION_BYTES" "$companion_digest"
  [[ "$(sha256_file "$reference/manifest.txt")" == "$reference_digest" ]] \
    || die "reference manifest SHA-256 mismatch"
  require_reference "$reference"
  require_disjoint_evidence "$evidence_dir" "$VOKRA_ROOT" "$gguf" "$companion" "$reference" "$approval"
  record_environment "$evidence_dir/environment.txt"

  {
    echo "expected_gguf_sha256=$gguf_digest"
    echo "actual_gguf_sha256=$(sha256_file "$gguf")"
    echo "expected_companion_sha256=$companion_digest"
    echo "actual_companion_sha256=$(sha256_file "$companion")"
    echo "expected_reference_manifest_sha256=$reference_digest"
    echo "actual_reference_manifest_sha256=$(sha256_file "$reference/manifest.txt")"
  } > "$evidence_dir/input-hashes.txt"

  log "running real-weight CPU parity against official reference"
  run_parity cpu "$gguf" "$companion" "$reference" "$evidence_dir/parity-cpu.log"
  require_one_named_test_passed "$evidence_dir/parity-cpu.log" \
    neutts_air_public_cpu_or_metal_matches_official_reference Cpu

  log "running real-weight Metal parity against official reference"
  run_parity metal "$gguf" "$companion" "$reference" "$evidence_dir/parity-metal.log"
  require_one_named_test_passed "$evidence_dir/parity-metal.log" \
    neutts_air_public_cpu_or_metal_matches_official_reference Metal

  {
    echo "verdict=PASS"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "cpu_vs_official_logits_atol=0.01"
    echo "metal_vs_official_logits_atol=0.01"
    echo "cpu_greedy_ids=exact"
    echo "metal_greedy_ids=exact"
    echo "cpu_composition=PASS"
    echo "metal_composition=PASS"
    echo "upload=NOT_PERFORMED"
  } > "$evidence_dir/summary.txt"
  log "PASS: pull only $evidence_dir, then remove staged model/reference data or destroy the worker"
}

main "$@"
