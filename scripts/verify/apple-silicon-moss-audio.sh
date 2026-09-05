#!/usr/bin/env bash
# Real-weight MOSS-Audio CPU/Metal parity on a disposable Apple host.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/moss_audio"
PREFLIGHT_GATE="$PARITY_PROJECT/preflight_gate.py"
PREFLIGHT_MANIFEST="$PARITY_PROJECT/license_gate_manifest.json"

MIN_MEMORY_BYTES=64000000000
MIN_FREE_DISK_KIB=50000000
SOURCE_REVISION="5cbb1d823937cd5b5de3d8fa4d3a7253ebd3b883"
REFERENCE_FILES=(manifest.txt pcm.f32le prompt_ids.u32le primary_audio.f32le \
  deepstack_audio_0.f32le deepstack_audio_1.f32le deepstack_audio_2.f32le \
  generated_ids.u32le prompt.txt result_text.txt environment.json source_files.json)
REFERENCE_MANIFEST_KEYS=(schema variant model_name upstream_repo upstream_revision \
  source_code_revision configuration_source_sha256 modeling_source_sha256 \
  processing_source_sha256 config_sha256 torch_version transformers_version \
  sample_rate pcm_samples audio_frames hidden_size prompt_tokens generated_tokens \
  max_new_tokens tensor_count config_model_type source_audio_sha256 \
  sha256_pcm_f32le sha256_prompt_ids_u32le sha256_primary_audio_f32le \
  sha256_deepstack_audio_0_f32le sha256_deepstack_audio_1_f32le \
  sha256_deepstack_audio_2_f32le sha256_generated_ids_u32le sha256_prompt_txt \
  sha256_result_text_txt sha256_environment_json sha256_source_files_json)

log() { printf '[moss-audio-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: apple-silicon-moss-audio.sh \
  --gguf-4b <path> --gguf-4b-sha256 <hash> --reference-4b <dir> \
  --reference-4b-sha256 <hash> --gguf-8b <path> --gguf-8b-sha256 <hash> \
  --reference-8b <dir> --reference-8b-sha256 <hash> \
  --approval-evidence <file> --evidence-dir <absent-dir>
       apple-silicon-moss-audio.sh --self-test

Runs projected-audio and exact-token CPU/Metal parity for both pinned
MOSS-Audio Instruct releases. It refuses the 16-GB maintainer class and
requires VOKRA_REMOTE_APPLE_SILICON=1, Darwin arm64, a clean checkout, at
least 64 GB physical memory, and all four real inputs before Cargo starts.

This script does not download, upload, convert, publish, or delete a model.
Transfer VAST-produced inputs directly to a disposable Apple host. The
evidence directory must be absent/nonexistent before validation and is created
only after approval and input checks succeed. Pull only the evidence directory
after the run, then remove the staged data or destroy
the remote worker.
EOF
}

sha256_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

license_preflight() {
  local approval="$1"
  command -v uv >/dev/null 2>&1 || die "required tool missing: uv"
  [[ -f "$PREFLIGHT_GATE" && ! -L "$PREFLIGHT_GATE" && -f "$PREFLIGHT_MANIFEST" && ! -L "$PREFLIGHT_MANIFEST" ]] \
    || die "MOSS-Audio preflight gate or manifest is missing"
  require_file "approval evidence" "$approval" || return 2
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$PREFLIGHT_GATE" \
    --project "$PARITY_PROJECT" --manifest "$PREFLIGHT_MANIFEST" --evidence "$approval"
}

require_disjoint_evidence() {
  local evidence="$1" candidate parent other real other_parent
  if [[ -e "$evidence" || -L "$evidence" ]]; then
    die "evidence directory must be absent before validation: $evidence"
    return 2
  fi
  if ! parent="$(cd -P "$(dirname "$evidence")" 2>/dev/null && pwd)"; then
    die "evidence parent is inaccessible"
    return 2
  fi
  candidate="$parent/$(basename "$evidence")"
  shift
  for other in "$@"; do
    if [[ -L "$other" ]]; then
      die "validation input is a symlink: $other"
      return 2
    fi
    if ! other_parent="$(cd -P "$(dirname "$other")" 2>/dev/null && pwd)"; then
      die "validation input is inaccessible: $other"
      return 2
    fi
    real="$other_parent/$(basename "$other")"
    if [[ "$candidate" == "$real" || "$candidate/" == "$real/"* || "$real/" == "$candidate/"* ]]; then
      die "evidence directory overlaps validation input"
      return 2
    fi
  done
  mkdir -p "$evidence"
}

require_file() {
  local label="$1" path="$2"
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] || die "$label is missing, symlinked, or empty: $path"
}

require_expected_sha256() {
  local label="$1" path="$2" expected="$3" actual
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || { die "$label expected SHA-256 is malformed"; return 2; }
  require_file "$label" "$path" || return 2
  actual="$(sha256_file "$path")"
  [[ "$actual" == "$expected" ]] || { die "$label SHA-256 mismatch: $actual != $expected"; return 2; }
}

require_manifest_value() {
  local manifest="$1" key="$2" expected="$3" count
  count="$(grep -Ec "^${key}=${expected}$" "$manifest" || true)"
  [[ "$count" == 1 ]] || { die "reference manifest field $key is missing, duplicated or wrong"; return 2; }
}

require_manifest_pattern() {
  local manifest="$1" key="$2" pattern="$3" count
  count="$(grep -Ec "^${key}=${pattern}$" "$manifest" || true)"
  [[ "$count" == 1 ]] || { die "reference manifest field $key is missing, duplicated or malformed"; return 2; }
}

require_exact_manifest_keys() {
  local manifest="$1" actual expected duplicate
  if grep -Ev '^[A-Za-z0-9_]+=[^=[:space:]]*$' "$manifest" >/dev/null; then
    die "reference manifest contains malformed or ambiguous lines"
    return 2
  fi
  duplicate="$(cut -d= -f1 "$manifest" | sort | uniq -d | head -n 1)"
  [[ -z "$duplicate" ]] || { die "reference manifest key is duplicated: $duplicate"; return 2; }
  actual="$(cut -d= -f1 "$manifest" | sort)"
  expected="$(printf '%s\n' "${REFERENCE_MANIFEST_KEYS[@]}" | sort)"
  [[ "$actual" == "$expected" ]] || { die "reference manifest key set is not exact"; return 2; }
}

verify_manifest_file_hash() {
  local manifest="$1" directory="$2" artifact="$3" key expected count
  key="sha256_${artifact//./_}"
  count="$(grep -Ec "^${key}=" "$manifest" || true)"
  [[ "$count" == 1 ]] || { die "reference manifest hash field is missing or duplicated: $key"; return 2; }
  expected="$(grep -E "^${key}=" "$manifest" | cut -d= -f2-)"
  require_expected_sha256 "reference $artifact" "$directory/$artifact" "$expected"
}

require_reference() {
  local label="$1" directory="$2" variant="$3" name manifest expected_files actual_files
  [[ -d "$directory" && ! -L "$directory" ]] || { die "$label is missing or symlinked: $directory"; return 2; }
  manifest="$directory/manifest.txt"
  require_exact_manifest_keys "$manifest" || return 2
  for name in "${REFERENCE_FILES[@]}"; do
    if [[ "$name" == result_text.txt ]]; then
      [[ -f "$directory/$name" && ! -L "$directory/$name" ]] || { die "$label $name is missing or symlinked"; return 2; }
    else
      require_file "$label $name" "$directory/$name" || return 2
    fi
  done
  expected_files="$(printf '%s\n' "${REFERENCE_FILES[@]}" | sort)"
  actual_files="$(find "$directory" -mindepth 1 -maxdepth 1 -exec basename {} \; | sort)"
  [[ "$actual_files" == "$expected_files" ]] || { die "$label contains unexpected or missing files"; return 2; }
  case "$variant" in
    4b)
      require_manifest_value "$manifest" variant 4b || return 2
      require_manifest_value "$manifest" model_name moss-audio-4b-instruct || return 2
      require_manifest_value "$manifest" upstream_repo OpenMOSS-Team/MOSS-Audio-4B-Instruct || return 2
      require_manifest_value "$manifest" upstream_revision 6907a499dc0e87cc77c8ae0fe23fd0eb5476a02d || return 2
      require_manifest_value "$manifest" config_sha256 e528a941446f4443f1b9fede12ea484e58a79d494c28d21ef1e73b5148abfbfa || return 2
      ;;
    8b)
      require_manifest_value "$manifest" variant 8b || return 2
      require_manifest_value "$manifest" model_name moss-audio-8b-instruct || return 2
      require_manifest_value "$manifest" upstream_repo OpenMOSS-Team/MOSS-Audio-8B-Instruct || return 2
      require_manifest_value "$manifest" upstream_revision 6521a39181b47a18f2d9f4b3acfb5bca7b76b57f || return 2
      require_manifest_value "$manifest" config_sha256 535154c2a5bcbd0e18e2f92bcf370ac74b530eec97ad4fd9317993ba0a316536 || return 2
      ;;
    *) die "unknown MOSS-Audio reference variant: $variant"; return 2 ;;
  esac
  require_manifest_value "$manifest" schema vokra-moss-audio-reference-v1 || return 2
  require_manifest_value "$manifest" source_code_revision "$SOURCE_REVISION" || return 2
  require_manifest_value "$manifest" configuration_source_sha256 e597dca441ff7fb58a5ec43186fafdfce19f31dada4955b4910059baa5d52ebd || return 2
  require_manifest_value "$manifest" modeling_source_sha256 a52513e518c68a0ba7c636a1ab0e12f7755ceebd0ae033235dc5e2551bfcbf9c || return 2
  require_manifest_value "$manifest" processing_source_sha256 05fb788cbdc6482eded8d70f7d2f524bc0cdca47d001acab5661c11f02cc6fe6 || return 2
  require_manifest_value "$manifest" config_model_type moss_audio || return 2
  require_manifest_value "$manifest" tensor_count 901 || return 2
  require_manifest_value "$manifest" transformers_version 5.5.0 || return 2
  require_manifest_pattern "$manifest" torch_version '2\.9\.1(\+cpu)?' || return 2
  require_manifest_value "$manifest" sample_rate 16000 || return 2
  require_manifest_value "$manifest" source_audio_sha256 241c0d93cc7ed8792c85c525d1e02b8c33850b791902a5e75b79c2d500e71a1a || return 2
  require_manifest_pattern "$manifest" pcm_samples '[1-9][0-9]*' || return 2
  require_manifest_pattern "$manifest" audio_frames '[1-9][0-9]*' || return 2
  require_manifest_pattern "$manifest" prompt_tokens '[1-9][0-9]*' || return 2
  require_manifest_pattern "$manifest" generated_tokens '[0-9]+' || return 2
  require_manifest_value "$manifest" max_new_tokens 4 || return 2
  case "$variant" in
    4b) require_manifest_value "$manifest" hidden_size 2560 || return 2 ;;
    8b) require_manifest_value "$manifest" hidden_size 4096 || return 2 ;;
  esac
  for name in "${REFERENCE_FILES[@]:1}"; do
    verify_manifest_file_hash "$manifest" "$directory" "$name" || return 2
  done
}

require_one_named_test_passed() {
  local log_path="$1" test_name="$2" test_count named_count total_test_count result_count total_result_count
  test_count="$(grep -Ec "^test ${test_name} \.\.\. ok$" "$log_path" || true)"
  named_count="$(grep -Ec "^test ${test_name} \.\.\." "$log_path" || true)"
  total_test_count="$(grep -Ec '^test [^ ]+ \.\.\.' "$log_path" || true)"
  result_count="$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$log_path" || true)"
  total_result_count="$(grep -Ec '^test result:' "$log_path" || true)"
  [[ "$test_count" == 1 ]] || { die "expected exactly one passing named MOSS-Audio test"; return 2; }
  [[ "$named_count" == 1 ]] || { die "expected exactly one total named MOSS-Audio test line"; return 2; }
  [[ "$total_test_count" == 1 ]] || { die "expected exactly one total Cargo test line"; return 2; }
  [[ "$result_count" == 1 ]] || { die "expected one standard Cargo result"; return 2; }
  [[ "$total_result_count" == 1 ]] || { die "expected exactly one total Cargo result line"; return 2; }
}

require_exact_parity_sentinels() {
  local log_path="$1" count_4b count_8b family_4b family_8b
  family_4b="$(grep -Ec '^MOSS_AUDIO_PARITY moss-audio-4b-instruct Metal_vs_CPU ' "$log_path" || true)"
  family_8b="$(grep -Ec '^MOSS_AUDIO_PARITY moss-audio-8b-instruct Metal_vs_CPU ' "$log_path" || true)"
  count_4b="$(grep -Ec '^MOSS_AUDIO_PARITY moss-audio-4b-instruct Metal_vs_CPU token_ids=exact text=exact PASS$' "$log_path" || true)"
  count_8b="$(grep -Ec '^MOSS_AUDIO_PARITY moss-audio-8b-instruct Metal_vs_CPU token_ids=exact text=exact PASS$' "$log_path" || true)"
  [[ "$family_4b" == 1 && "$family_8b" == 1 && "$count_4b" == 1 && "$count_8b" == 1 ]] || { die "expected exactly one complete 4B and 8B Metal sentinel families"; return 2; }
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
  require_expected_sha256 value "$temporary/value" \
    ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
  if require_expected_sha256 value "$temporary/value" bad >/dev/null 2>&1; then
    die "malformed expected SHA self-test failed"
  fi
  if require_expected_sha256 value "$temporary/value" \
    0000000000000000000000000000000000000000000000000000000000000000 >/dev/null 2>&1; then
    die "mismatched expected SHA self-test failed"
  fi
  local reference="$temporary/reference-4b" name
  mkdir "$reference"
  for name in "${REFERENCE_FILES[@]}"; do
    [[ "$name" == manifest.txt ]] || printf 'abc' > "$reference/$name"
  done
  {
    printf '%s\n' \
      'schema=vokra-moss-audio-reference-v1' \
      'variant=4b' \
      'model_name=moss-audio-4b-instruct' \
      'upstream_repo=OpenMOSS-Team/MOSS-Audio-4B-Instruct' \
      'upstream_revision=6907a499dc0e87cc77c8ae0fe23fd0eb5476a02d' \
      'source_code_revision=5cbb1d823937cd5b5de3d8fa4d3a7253ebd3b883' \
      'configuration_source_sha256=e597dca441ff7fb58a5ec43186fafdfce19f31dada4955b4910059baa5d52ebd' \
      'modeling_source_sha256=a52513e518c68a0ba7c636a1ab0e12f7755ceebd0ae033235dc5e2551bfcbf9c' \
      'processing_source_sha256=05fb788cbdc6482eded8d70f7d2f524bc0cdca47d001acab5661c11f02cc6fe6' \
      'config_sha256=e528a941446f4443f1b9fede12ea484e58a79d494c28d21ef1e73b5148abfbfa' \
      'config_model_type=moss_audio' \
      'tensor_count=901' \
      'transformers_version=5.5.0' \
      'torch_version=2.9.1+cpu' \
      'sample_rate=16000' \
      'source_audio_sha256=241c0d93cc7ed8792c85c525d1e02b8c33850b791902a5e75b79c2d500e71a1a' \
      'pcm_samples=1' \
      'audio_frames=1' \
      'prompt_tokens=1' \
      'generated_tokens=0' \
      'max_new_tokens=4' \
      'hidden_size=2560'
    for name in "${REFERENCE_FILES[@]:1}"; do
      printf 'sha256_%s=%s\n' "${name//./_}" ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
    done
  } > "$reference/manifest.txt"
  require_reference "self-test 4B reference" "$reference" 4b
  local reference_link="$temporary/reference-link"
  ln -s "$reference" "$reference_link"
  if require_reference "self-test symlinked reference" "$reference_link" 4b >/dev/null 2>&1; then
    die "reference root symlink tamper self-test failed"
  fi
  rm "$reference_link"
  local manifest_good="$temporary/manifest.good"
  cp "$reference/manifest.txt" "$manifest_good"
  printf '%s\n' 'extra_key=unexpected' >> "$reference/manifest.txt"
  if require_reference "self-test 4B reference" "$reference" 4b >/dev/null 2>&1; then
    die "inner reference extra-manifest-key tamper self-test failed"
  fi
  cp "$manifest_good" "$reference/manifest.txt"
  printf '%s\n' 'schema=vokra-moss-audio-reference-v1' >> "$reference/manifest.txt"
  if require_reference "self-test 4B reference" "$reference" 4b >/dev/null 2>&1; then
    die "inner reference duplicate-manifest-key tamper self-test failed"
  fi
  cp "$manifest_good" "$reference/manifest.txt"
  printf 'tampered' > "$reference/pcm.f32le"
  if require_reference "self-test 4B reference" "$reference" 4b >/dev/null 2>&1; then
    die "inner reference hash tamper self-test failed"
  fi
  printf 'abc' > "$reference/pcm.f32le"
  printf 'extra' > "$reference/extra.bin"
  if require_reference "self-test 4B reference" "$reference" 4b >/dev/null 2>&1; then
    die "inner reference file-set tamper self-test failed"
  fi
  rm -f "$reference/extra.bin"
  mkdir "$reference/extra-dir"
  if require_reference "self-test 4B reference" "$reference" 4b >/dev/null 2>&1; then
    die "inner reference directory tamper self-test failed"
  fi
  rmdir "$reference/extra-dir"
  ln -s pcm.f32le "$reference/extra-link"
  if require_reference "self-test 4B reference" "$reference" 4b >/dev/null 2>&1; then
    die "inner reference symlink tamper self-test failed"
  fi
  local evidence_probe="$temporary/evidence"
  if mkdir -p "$evidence_probe" && require_disjoint_evidence "$evidence_probe" "$VOKRA_ROOT" >/dev/null 2>&1; then
    die "existing evidence directory was accepted"
  fi
  [[ ! -e "$evidence_probe" ]] || rmdir "$evidence_probe"
  require_disjoint_evidence "$evidence_probe" "$VOKRA_ROOT"
  [[ -d "$evidence_probe" ]] || die "absent evidence directory helper self-test failed"
  local cargo_log="$temporary/cargo.log"
  printf '%s\n%s\n' \
    'test moss_audio_real_metal_matches_cpu_exact_greedy ... ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' > "$cargo_log"
  require_one_named_test_passed "$cargo_log" moss_audio_real_metal_matches_cpu_exact_greedy
  printf '%s\n%s\n%s\n' \
    'test moss_audio_real_metal_matches_cpu_exact_greedy ... ok' \
    'test result: ok. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' > "$cargo_log"
  if require_one_named_test_passed "$cargo_log" moss_audio_real_metal_matches_cpu_exact_greedy >/dev/null 2>&1; then
    die "duplicate Cargo result self-test failed"
  fi
  printf '%s\n%s\n' \
    'test moss_audio_real_metal_matches_cpu_exact_greedy ... ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.2x' > "$cargo_log"
  if require_one_named_test_passed "$cargo_log" moss_audio_real_metal_matches_cpu_exact_greedy >/dev/null 2>&1; then
    die "malformed Cargo timing self-test failed"
  fi
  printf '%s\n%s\n%s\n' \
    'test moss_audio_real_metal_matches_cpu_exact_greedy ... ok' \
    'test moss_audio_real_metal_matches_cpu_exact_greedy ... ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' > "$cargo_log"
  if require_one_named_test_passed "$cargo_log" moss_audio_real_metal_matches_cpu_exact_greedy >/dev/null 2>&1; then
    die "duplicate named Cargo test self-test failed"
  fi
  local sentinel_log="$temporary/sentinels.log"
  printf '%s\n%s\n' \
    'MOSS_AUDIO_PARITY moss-audio-4b-instruct Metal_vs_CPU token_ids=exact text=exact PASS' \
    'MOSS_AUDIO_PARITY moss-audio-8b-instruct Metal_vs_CPU token_ids=exact text=exact PASS' > "$sentinel_log"
  require_exact_parity_sentinels "$sentinel_log"
  printf '%s\n%s\n%s\n' \
    'MOSS_AUDIO_PARITY moss-audio-4b-instruct Metal_vs_CPU token_ids=exact text=exact PASS' \
    'MOSS_AUDIO_PARITY moss-audio-4b-instruct Metal_vs_CPU token_ids=exact text=exact PASS' \
    'MOSS_AUDIO_PARITY moss-audio-8b-instruct Metal_vs_CPU token_ids=exact text=exact PASS' > "$sentinel_log"
  if require_exact_parity_sentinels "$sentinel_log" >/dev/null 2>&1; then
    die "duplicate sentinel self-test failed"
  fi
  printf 'prefix%s\n%s\n' \
    'MOSS_AUDIO_PARITY moss-audio-4b-instruct Metal_vs_CPU token_ids=exact text=exact PASS' \
    'MOSS_AUDIO_PARITY moss-audio-8b-instruct Metal_vs_CPU token_ids=exact text=exact PASS' > "$sentinel_log"
  if require_exact_parity_sentinels "$sentinel_log" >/dev/null 2>&1; then
    die "prefix sentinel self-test failed"
  fi
  printf '%s suffix\n%s\n' \
    'MOSS_AUDIO_PARITY moss-audio-4b-instruct Metal_vs_CPU token_ids=exact text=exact PASS' \
    'MOSS_AUDIO_PARITY moss-audio-8b-instruct Metal_vs_CPU token_ids=exact text=exact PASS' > "$sentinel_log"
  if require_exact_parity_sentinels "$sentinel_log" >/dev/null 2>&1; then
    die "suffix sentinel self-test failed"
  fi
  printf '%s\n%s\n%s\n' \
    'MOSS_AUDIO_PARITY moss-audio-4b-instruct Metal_vs_CPU token_ids=exact text=exact PASS' \
    'MOSS_AUDIO_PARITY moss-audio-4b-instruct Metal_vs_CPU token_ids=exact text=exact FAIL' \
    'MOSS_AUDIO_PARITY moss-audio-8b-instruct Metal_vs_CPU token_ids=exact text=exact PASS' > "$sentinel_log"
  if require_exact_parity_sentinels "$sentinel_log" >/dev/null 2>&1; then
    die "FAIL sentinel self-test failed"
  fi
  # shellcheck disable=SC2086 # Each case intentionally models argv tokenization.
  for bad_args in "--self-test --approval-evidence x" "--self-test --self-test" "--approval-evidence" "--evidence-dir" "--unknown x"; do
    if bash "$0" $bad_args >/dev/null 2>&1; then die "accepted malformed parser case: $bad_args"; fi
  done
  log "self-test PASS"
)

main() {
  local gguf_4b='' reference_4b='' gguf_8b='' reference_8b=''
  local gguf_4b_sha='' reference_4b_sha='' gguf_8b_sha='' reference_8b_sha=''
  local approval_evidence='' evidence_dir='' self_test=0
  local seen_gguf_4b=0 seen_reference_4b=0 seen_gguf_4b_sha=0 seen_reference_4b_sha=0
  local seen_gguf_8b=0 seen_reference_8b=0 seen_gguf_8b_sha=0 seen_reference_8b_sha=0
  local seen_approval=0 seen_evidence=0 seen_self_test=0
  while (( $# > 0 )); do
    case "$1" in
      --gguf-4b)
        (( seen_gguf_4b == 0 )) || die 'duplicate --gguf-4b'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--gguf-4b requires a nonempty value'; seen_gguf_4b=1
        gguf_4b="$2"
        shift 2
        ;;
      --reference-4b)
        (( seen_reference_4b == 0 )) || die 'duplicate --reference-4b'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--reference-4b requires a nonempty value'; seen_reference_4b=1
        reference_4b="$2"
        shift 2
        ;;
      --gguf-4b-sha256)
        (( seen_gguf_4b_sha == 0 )) || die 'duplicate --gguf-4b-sha256'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--gguf-4b-sha256 requires a nonempty value'; seen_gguf_4b_sha=1
        gguf_4b_sha="$2"
        shift 2
        ;;
      --reference-4b-sha256)
        (( seen_reference_4b_sha == 0 )) || die 'duplicate --reference-4b-sha256'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--reference-4b-sha256 requires a nonempty value'; seen_reference_4b_sha=1
        reference_4b_sha="$2"
        shift 2
        ;;
      --gguf-8b)
        (( seen_gguf_8b == 0 )) || die 'duplicate --gguf-8b'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--gguf-8b requires a nonempty value'; seen_gguf_8b=1
        gguf_8b="$2"
        shift 2
        ;;
      --reference-8b)
        (( seen_reference_8b == 0 )) || die 'duplicate --reference-8b'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--reference-8b requires a nonempty value'; seen_reference_8b=1
        reference_8b="$2"
        shift 2
        ;;
      --gguf-8b-sha256)
        (( seen_gguf_8b_sha == 0 )) || die 'duplicate --gguf-8b-sha256'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--gguf-8b-sha256 requires a nonempty value'; seen_gguf_8b_sha=1
        gguf_8b_sha="$2"
        shift 2
        ;;
      --reference-8b-sha256)
        (( seen_reference_8b_sha == 0 )) || die 'duplicate --reference-8b-sha256'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--reference-8b-sha256 requires a nonempty value'; seen_reference_8b_sha=1
        reference_8b_sha="$2"
        shift 2
        ;;
      --approval-evidence)
        (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--approval-evidence requires a nonempty value'; seen_approval=1
        approval_evidence="$2"; shift 2 ;;
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
    [[ -z "$gguf_4b$reference_4b$gguf_8b$reference_8b$gguf_4b_sha$reference_4b_sha$gguf_8b_sha$reference_8b_sha$approval_evidence$evidence_dir" ]] \
      || die "--self-test accepts no other arguments"
    run_self_test
    return
  fi
  [[ -n "$gguf_4b" && -n "$reference_4b" && -n "$gguf_8b" && \
    -n "$reference_8b" && -n "$gguf_4b_sha" && -n "$reference_4b_sha" && \
    -n "$gguf_8b_sha" && -n "$reference_8b_sha" && -n "$approval_evidence" && -n "$evidence_dir" ]] \
    || { usage; die "all model/reference arguments, --approval-evidence, and --evidence-dir are required"; }
  for expected in "$gguf_4b_sha" "$reference_4b_sha" "$gguf_8b_sha" "$reference_8b_sha"; do
    [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || die "all four expected SHA-256 values must be lowercase 64-hex"
  done

  license_preflight "$approval_evidence"
  require_remote_apple_host
  require_tooling
  require_file "MOSS-Audio 4B GGUF" "$gguf_4b"
  require_file "MOSS-Audio 8B GGUF" "$gguf_8b"
  require_expected_sha256 "MOSS-Audio 4B GGUF" "$gguf_4b" "$gguf_4b_sha"
  require_expected_sha256 "MOSS-Audio 8B GGUF" "$gguf_8b" "$gguf_8b_sha"
  require_expected_sha256 "MOSS-Audio 4B reference manifest" "$reference_4b/manifest.txt" "$reference_4b_sha"
  require_expected_sha256 "MOSS-Audio 8B reference manifest" "$reference_8b/manifest.txt" "$reference_8b_sha"
  require_reference "MOSS-Audio 4B reference" "$reference_4b" 4b
  require_reference "MOSS-Audio 8B reference" "$reference_8b" 8b
  require_disjoint_evidence "$evidence_dir" "$VOKRA_ROOT" "$gguf_4b" "$reference_4b" "$gguf_8b" "$reference_8b" "$approval_evidence"
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
  require_one_named_test_passed "$evidence_dir/parity.log" moss_audio_real_metal_matches_cpu_exact_greedy
  require_exact_parity_sentinels "$evidence_dir/parity.log"


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
