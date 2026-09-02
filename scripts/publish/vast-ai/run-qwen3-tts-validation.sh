#!/usr/bin/env bash
# VAST-only conversion and real-weight validation for the four released
# Qwen3-TTS 12-Hz main checkpoints plus their official decoder companion.
# Corrected GGUFs are staged on VAST only; this script never uploads them.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/qwen3_tts"
REFERENCE_DUMPER="$PARITY_PROJECT/dump_reference.py"
LICENSE_GATE="$PARITY_PROJECT/license_gate.py"
LICENSE_MANIFEST="$PARITY_PROJECT/license_gate_manifest.json"
REFERENCE_AUDIO="$VOKRA_ROOT/tests/parity/utmos/ref-clip.wav"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"

DECODER_REPO="Qwen/Qwen3-TTS-Tokenizer-12Hz"
DECODER_REVISION="a87c50897bb00837eb857d0538b29d117541d7f6"
DECODER_CHECKPOINT_SHA256="836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258"
OFFICIAL_SOURCE_REPO="QwenLM/Qwen3-TTS"
OFFICIAL_SOURCE_URL="https://github.com/QwenLM/Qwen3-TTS.git"
OFFICIAL_SOURCE_REVISION="022e286b98fbec7e1e916cb940cdf532cd9f488e"
REFERENCE_AUDIO_SHA256="241c0d93cc7ed8792c85c525d1e02b8c33850b791902a5e75b79c2d500e71a1a"
MIN_NEW_TOKENS=2
TRANSFORMERS_VERSION="5.10.4"
TRANSFORMERS_COMPATIBILITY_STATUS="BLOCKED_UNVERIFIED_API_SMOKE"

MIN_VAST_MEM_KIB=60000000
MIN_FREE_DISK_KIB=100000000

log() { printf '[qwen3-tts-vast] %s\n' "$*" >&2; }
step() { printf '\n[qwen3-tts-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

canonicalize_uncreated() {
  local path="$1" suffix='' name parent
  local scan rest component
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
require_absent_work_dir() {
  local target="$1" approval="$2" canonical protected other
  [[ ! -e "$target" && ! -L "$target" ]] || { die "--work-dir must be absent and non-symlink: $target"; return 2; }
  canonical="$(canonicalize_uncreated "$target")" || { die "cannot canonicalize --work-dir: $target"; return 2; }
  for protected in "$VOKRA_ROOT" "$PARITY_PROJECT" "$LICENSE_GATE" "$LICENSE_MANIFEST" \
    "$PARITY_PROJECT/uv.lock" "$PARITY_PROJECT/pyproject.toml" "$approval"; do
    [[ -e "$protected" || -L "$protected" ]] || continue
    [[ ! -L "$protected" ]] || { die "protected path is symlinked: $protected"; return 2; }
    other="$(canonicalize_uncreated "$protected")" || { die "cannot canonicalize protected path: $protected"; return 2; }
    paths_overlap "$canonical" "$other" && { die "--work-dir overlaps protected path: $protected"; return 2; }
  done
  return 0
}

usage() {
  cat <<'EOF' >&2
usage: run-qwen3-tts-validation.sh --approval-evidence <json> [--variant <slug|all>] [--work-dir <absent-dir>]
       run-qwen3-tts-validation.sh --self-test

Converts the exact immutable Qwen3-TTS main release into corrected GGUFs and
the separately authenticated official 12-Hz decoder on VAST, then runs the
independent official CPU reference and native real-weight parity test. The
public pre-contract GGUFs are never downloaded or treated as canonical.
There is no upload, publish, push, or download path for generated artifacts.
EOF
}

variant_repo() {
  case "$1" in
    0.6b-base) printf '%s\n' 'Qwen/Qwen3-TTS-12Hz-0.6B-Base' ;;
    0.6b-customvoice) printf '%s\n' 'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice' ;;
    1.7b-base) printf '%s\n' 'Qwen/Qwen3-TTS-12Hz-1.7B-Base' ;;
    1.7b-customvoice) printf '%s\n' 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice' ;;
    *) die "unknown variant: $1" ;;
  esac
}

variant_revision() {
  case "$1" in
    0.6b-base) printf '%s\n' '5d83992436eae1d760afd27aff78a71d676296fc' ;;
    0.6b-customvoice) printf '%s\n' '85e237c12c027371202489a0ec509ded67b5e4b5' ;;
    1.7b-base) printf '%s\n' 'fd4b254389122332181a7c3db7f27e918eec64e3' ;;
    1.7b-customvoice) printf '%s\n' '0c0e3051f131929182e2c023b9537f8b1c68adfe' ;;
    *) die "unknown variant: $1" ;;
  esac
}

variant_model_kind() {
  case "$1" in
    0.6b-base) printf '%s\n' 'qwen3-tts-12hz-0.6b-base' ;;
    0.6b-customvoice) printf '%s\n' 'qwen3-tts-12hz-0.6b-customvoice' ;;
    1.7b-base) printf '%s\n' 'qwen3-tts-12hz-1.7b-base' ;;
    1.7b-customvoice) printf '%s\n' 'qwen3-tts-12hz-1.7b-customvoice' ;;
    *) die "unknown variant: $1" ;;
  esac
}

variant_config_bytes() {
  case "$1" in
    0.6b-base|1.7b-base) printf '%s\n' 4494 ;;
    0.6b-customvoice|1.7b-customvoice) printf '%s\n' 4908 ;;
    *) die "unknown variant: $1" ;;
  esac
}

variant_config_sha256() {
  case "$1" in
    0.6b-base) printf '%s\n' 2e714c787c8edb98b05432685cddb634add2de4d4e645f653d68251ef72ba011 ;;
    0.6b-customvoice) printf '%s\n' 81aca2b6fac304944d8acf345272d8a9a727d5fc2e2e66b222ab4729340c7455 ;;
    1.7b-base) printf '%s\n' b4f01752d15a488abde3e1ab44723ae4f4b9e68a4037257b098b3737893cc1f9 ;;
    1.7b-customvoice) printf '%s\n' 17a07f527a1c25ea30b4e023a184482a23d3e279d697b1dc81b1bde498d29cf9 ;;
    *) die "unknown variant: $1" ;;
  esac
}

variant_env_prefix() {
  case "$1" in
    0.6b-base) printf '%s\n' 'VOKRA_QWEN3_TTS_0_6B_BASE' ;;
    0.6b-customvoice) printf '%s\n' 'VOKRA_QWEN3_TTS_0_6B_CUSTOMVOICE' ;;
    1.7b-base) printf '%s\n' 'VOKRA_QWEN3_TTS_1_7B_BASE' ;;
    1.7b-customvoice) printf '%s\n' 'VOKRA_QWEN3_TTS_1_7B_CUSTOMVOICE' ;;
    *) die "unknown variant: $1" ;;
  esac
}

sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$path" | awk '{print $1}';
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$path" | awk '{print $1}';
  else die 'no SHA-256 utility available'; fi
}

verify_reference_hashes() {
  local directory="$1" file key expected actual
  for file in prompt_ids.u32le codes.u32le pcm.f32le environment.json; do
    key="sha256_${file//./_}"
    expected="$(grep -F -- "\"$key\":" "$directory/manifest.json" | sed -E 's/.*"([0-9a-f]{64})".*/\1/')"
    [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || die "$directory manifest lacks $key"
    actual="$(sha256_file "$directory/$file")"
    [[ "$actual" == "$expected" ]] || die "$directory/$file hash differs from manifest"
  done
  if [[ "$directory" == *-0.6b-base || "$directory" == *-1.7b-base ]]; then
    file=speaker_embedding.f32le
    key="sha256_${file//./_}"
    expected="$(grep -F -- "\"$key\":" "$directory/manifest.json" | sed -E 's/.*"([0-9a-f]{64})".*/\1/')"
    [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || die "$directory manifest lacks $key"
    actual="$(sha256_file "$directory/$file")"
    [[ "$actual" == "$expected" ]] || die "$directory/$file hash differs from manifest"
  fi
}

require_vast_host() {
  local mem_kib free_kib
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == '1' ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
  [[ "$(uname -s)" == 'Linux' ]] || die 'Qwen3-TTS conversion is VAST/Linux-only'
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$mem_kib" =~ ^[0-9]+$ ]] || die 'cannot read MemTotal'
  (( mem_kib >= MIN_VAST_MEM_KIB )) || die 'VAST host has less than 60-GB memory'
  mkdir -p "$VOKRA_SCRATCH"
  free_kib="$(df -Pk "$VOKRA_SCRATCH" | awk 'NR == 2 {print $4}')"
  [[ "$free_kib" =~ ^[0-9]+$ ]] || die 'cannot read free disk'
  (( free_kib >= MIN_FREE_DISK_KIB )) || die 'VAST scratch has less than 100-GB free disk'
}

require_tooling() {
  local tool
  for tool in uv cargo rustc git awk find tee wc df grep sed; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" ]] || die 'not a Vokra checkout'
  [[ -f "$PARITY_PROJECT/uv.lock" && -f "$REFERENCE_DUMPER" && -f "$LICENSE_GATE" && -f "$LICENSE_MANIFEST" ]] || die 'Qwen3-TTS parity project or license gate is incomplete'
  [[ -f "$REFERENCE_AUDIO" ]] || die 'reference audio is missing'
  [[ "$(sha256_file "$REFERENCE_AUDIO")" == "$REFERENCE_AUDIO_SHA256" ]] || die 'reference audio hash drifted'
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'VAST checkout must be clean'
}

license_gate() {
  local approval="$1"
  local args=(
    --project "$PARITY_PROJECT/pyproject.toml" --lock "$PARITY_PROJECT/uv.lock" --manifest "$LICENSE_MANIFEST"
    --source-revision "$OFFICIAL_SOURCE_REVISION"
    --decoder-revision "$DECODER_REVISION"
    --decoder-checkpoint-sha256 "$DECODER_CHECKPOINT_SHA256"
  )
  local variant
  for variant in 0.6b-base 0.6b-customvoice 1.7b-base 1.7b-customvoice; do
    args+=(--variant-revision "$variant=$(variant_revision "$variant")")
  done
  args+=(--license-evidence "$approval")
  UV_NO_CACHE=1 UV_CACHE_DIR="${QWEN3_TTS_UV_CACHE_DIR:-/tmp/vokra-qwen3-tts-uv-cache}" \
    uv run --no-cache --no-project --offline --python 3.12 python "$LICENSE_GATE" "${args[@]}"
}

preflight() {
  local approval="$1"
  [[ -f "$PARITY_PROJECT/pyproject.toml" && ! -L "$PARITY_PROJECT/pyproject.toml" && -f "$PARITY_PROJECT/uv.lock" && ! -L "$PARITY_PROJECT/uv.lock" && -f "$LICENSE_GATE" && ! -L "$LICENSE_GATE" && -f "$LICENSE_MANIFEST" && ! -L "$LICENSE_MANIFEST" ]] || die 'Qwen3-TTS preflight gate inputs are incomplete or symlinked'
  [[ -s "$approval" && ! -L "$approval" ]] || die 'approval evidence must be a non-empty regular non-symlink file'
  license_gate "$approval"
}

require_transformers_api_smoke() {
  case "$TRANSFORMERS_COMPATIBILITY_STATUS" in
    AUTHENTICATED_API_SMOKE) ;;
    BLOCKED_UNVERIFIED_API_SMOKE) die 'Transformers API smoke is not authenticated; refusing reference imports and acquisition' ;;
    *) die "unknown Transformers API smoke status: $TRANSFORMERS_COMPATIBILITY_STATUS" ;;
  esac
}

download_snapshot() {
  local repo="$1" revision="$2" output="$3"
  if [[ -e "$output" ]]; then
    [[ -d "$output" && -z "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die "snapshot target must be new or empty: $output"
  else
    mkdir -p "$output"
  fi
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
    'import os,sys; from huggingface_hub import snapshot_download; snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], token=os.environ.get("HF_TOKEN") or os.environ.get("HF"), allow_patterns=["LICENSE", "README.md", "config.json", "generation_config.json", "merges.txt", "model.safetensors", "model-*.safetensors", "model.safetensors.index.json", "preprocessor_config.json", "tokenizer_config.json", "vocab.json", "speech_tokenizer/**"])' \
    "$repo" "$revision" "$output"
}

download_source_tree() {
  local output="$1"
  [[ ! -e "$output" ]] || die "official source target must not already exist: $output"
  mkdir -p "$output"
  git -C "$output" init -q
  git -C "$output" remote add origin "$OFFICIAL_SOURCE_URL"
  git -C "$output" fetch --depth 1 origin "$OFFICIAL_SOURCE_REVISION"
  git -C "$output" checkout --detach FETCH_HEAD
  [[ "$(git -C "$output" rev-parse HEAD)" == "$OFFICIAL_SOURCE_REVISION" ]] || die 'official source checkout revision drifted'
  [[ -f "$output/pyproject.toml" && -f "$output/qwen_tts/__init__.py" ]] || die 'official source tree is incomplete'
  [[ -z "$(git -C "$output" status --porcelain --untracked-files=all)" ]] || die 'official source checkout is dirty'
}

require_single_file_snapshot() {
  local directory="$1" shards index
  [[ -f "$directory/model.safetensors" ]] || die "single-file checkpoint missing: $directory/model.safetensors"
  shards=("$directory"/model-*.safetensors)
  if [[ -e "${shards[0]}" ]]; then
    die "sharded checkpoint detected at $directory; this validator requires authenticated single-file releases"
  fi
  index="$directory/model.safetensors.index.json"
  [[ ! -e "$index" ]] || die "sharded index detected at $index; refusing implicit merge"
  [[ ! -e "$directory/.cache/huggingface" && ! -L "$directory/.cache/huggingface" ]] || die "snapshot contains a .cache/huggingface entry that could invalidate exact closure"
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

run_self_test() {
  local script_path="${BASH_SOURCE[0]}" failed=0 required gate_line sync_line cpu_command cpu_log_token cpu_sentinel_token
  for required in \
    '0.6b-base' '0.6b-customvoice' '1.7b-base' '1.7b-customvoice' \
    '5d83992436eae1d760afd27aff78a71d676296fc' \
    '85e237c12c027371202489a0ec509ded67b5e4b5' \
    'fd4b254389122332181a7c3db7f27e918eec64e3' \
    '0c0e3051f131929182e2c023b9537f8b1c68adfe' \
    '2e714c787c8edb98b05432685cddb634add2de4d4e645f653d68251ef72ba011' \
    '81aca2b6fac304944d8acf345272d8a9a727d5fc2e2e66b222ab4729340c7455' \
    'b4f01752d15a488abde3e1ab44723ae4f4b9e68a4037257b098b3737893cc1f9' \
    '17a07f527a1c25ea30b4e023a184482a23d3e279d697b1dc81b1bde498d29cf9' \
    'Qwen/Qwen3-TTS-Tokenizer-12Hz' 'a87c50897bb00837eb857d0538b29d117541d7f6' \
    'https://github.com/QwenLM/Qwen3-TTS.git' 'download_source_tree' 'git init' 'remote add origin' 'fetch --depth 1 origin' 'FETCH_HEAD' '--source-dir' \
    '022e286b98fbec7e1e916cb940cdf532cd9f488e' "$DECODER_CHECKPOINT_SHA256" \
    'nested_decoder_sha256' 'min_new_tokens' \
    'qwen3-tts-tokenizer-12hz' 'MIN_NEW_TOKENS=2' 'qwen3_tts_real_cpu_matches_official_reference' \
    'qwen3_tts_real_metal_matches_cpu_and_official_reference' 'single-file checkpoint' '.cache/huggingface' \
    '--ignored --exact --nocapture' 'QWEN3_TTS_PARITY' 'codes_exact=PASS' \
    'pcm=MEASURED_NOT_GATED' 'corrected GGUFs' 'CARGO_BUILD_JOBS' 'license_gate.py' \
    'TRANSFORMERS_VERSION="5.10.4"' 'previous_isolated_transformers_pin=transformers==4.57.3' \
    'transformers_security_advisory=GHSA-xrqw-3rrv-vx5w' 'transformers_security_patched_minimum=5.10.0' \
    'transformers_compatibility_status=BLOCKED_UNVERIFIED_API_SMOKE' 'require_transformers_api_smoke' \
    'AUTHENTICATED_API_SMOKE' 'UNKNOWN_STATUS' \
    'license_gate_manifest.json' '--no-project --offline --python 3.12' 'test result: ok. 1 passed' \
    '--gguf-0.6b-base-sha256' '--gguf-0.6b-customvoice-sha256' '--gguf-1.7b-base-sha256' \
    '--gguf-1.7b-customvoice-sha256' '--decoder-gguf-sha256' '--reference-0.6b-base-sha256' \
    '--reference-0.6b-customvoice-sha256' '--reference-1.7b-base-sha256' '--reference-1.7b-customvoice-sha256' \
    '--decoder-gguf %q' '--reference-0.6b-base %q' '<APPLE_QWEN3_TTS_APPROVAL_EVIDENCE>' '<APPLE_QWEN3_TTS_EVIDENCE_DIR>'; do
    grep -Fq -- "$required" "$script_path" || { log "self-test missing token: $required"; failed=1; }
  done
  local apple_hash_flag apple_hash_echo_count
  for apple_hash_flag in \
    '--gguf-0.6b-base-sha256' '--gguf-0.6b-customvoice-sha256' '--gguf-1.7b-base-sha256' \
    '--gguf-1.7b-customvoice-sha256' '--decoder-gguf-sha256' '--reference-0.6b-base-sha256' \
    '--reference-0.6b-customvoice-sha256' '--reference-1.7b-base-sha256' '--reference-1.7b-customvoice-sha256'; do
    apple_hash_echo_count="$(grep -Fc -- " $apple_hash_flag %q" "$script_path" || true)"
    [[ "$apple_hash_echo_count" == 1 ]] || { log "self-test Apple transfer flag is not emitted exactly once: $apple_hash_flag"; failed=1; }
  done
  if grep -En '^[[:space:]]*(python3?|pip)([[:space:]]|$)' "$script_path" >/dev/null; then failed=1; fi
  local forbidden_marker='MEASURED_NOT_GATED'; forbidden_marker+=' PASS'
  if grep -Fq "$forbidden_marker" "$script_path"; then failed=1; fi
  if grep -En '(upload|publish|push|--push|huggingface-cli)' "$script_path" | grep -Ev 'never uploads|never.*publish|no upload|not.*push|NOT_PERFORMED|--push|scripts/publish/' >/dev/null; then failed=1; fi
  local download_block path_probe
  download_block="$(awk '/^download_snapshot\(\)/,/^\}/ {print}' "$script_path")"
  [[ "$download_block" != *"--with"* && "$download_block" != *"--no-project"* ]] || { log 'self-test download path uses an unreviewed uv environment'; failed=1; }
  path_probe="$(mktemp -d "${TMPDIR:-/tmp}/qwen3-tts-path-selftest.XXXXXX")"
  printf '{}\n' > "$path_probe/approval.json"
  require_absent_work_dir "$path_probe/new-work" "$path_probe/approval.json" || failed=1
  mkdir "$path_probe/empty-work"
  if require_absent_work_dir "$path_probe/empty-work" "$path_probe/approval.json" >/dev/null 2>&1; then failed=1; fi
  rmdir "$path_probe/empty-work"
  ln -s "$path_probe/missing-work" "$path_probe/link-work"
  if require_absent_work_dir "$path_probe/link-work" "$path_probe/approval.json" >/dev/null 2>&1; then failed=1; fi
  rm "$path_probe/link-work"
  mkdir -p "$path_probe/real-parent/child"
  ln -s "$path_probe/real-parent" "$path_probe/link-parent"
  if require_absent_work_dir "$path_probe/link-parent/child/new-work" "$path_probe/approval.json" >/dev/null 2>&1; then failed=1; fi
  rm -rf "$path_probe/real-parent" "$path_probe/link-parent"
  if require_absent_work_dir "$VOKRA_ROOT/qwen3-tts-self-test-work" "$path_probe/approval.json" >/dev/null 2>&1; then failed=1; fi
  if require_absent_work_dir "$path_probe/approval.json/child" "$path_probe/approval.json" >/dev/null 2>&1; then failed=1; fi
  rm -rf "$path_probe"
  cpu_command="$(grep -F 'cargo test --manifest-path' "$script_path" | grep -F 'qwen3_tts_real_cpu_matches_official_reference' || true)"
  [[ "$cpu_command" == *'--ignored --exact --nocapture --test-threads=1'* ]] || { log 'self-test CPU command is not exact/ignored/nocapture'; failed=1; }
  cpu_log_token="tee \"\$evidence/parity-cpu.log\""
  cpu_sentinel_token="QWEN3_TTS_PARITY variant=\$variant backend=cpu prompt_ids=exact codes_exact=PASS pcm=MEASURED_NOT_GATED"
  [[ "$cpu_command" == *"$cpu_log_token"* ]] || { log 'self-test CPU command does not capture a dedicated result log'; failed=1; }
  grep -Fq "require_exact_test_result \"\$evidence/parity-cpu.log\" qwen3_tts_real_cpu_matches_official_reference" "$script_path" || { log 'self-test does not require exactly one CPU test pass'; failed=1; }
  grep -Fq '0 failed; 0 ignored; 0 measured' "$script_path" || { log 'self-test does not reject failed/ignored/filtered test results'; failed=1; }
  grep -Fq "$cpu_sentinel_token" "$script_path" || { log 'self-test does not require per-variant CPU sentinels'; failed=1; }
  local result_probe duplicate_probe malformed_probe
  result_probe="$(mktemp "${TMPDIR:-/tmp}/qwen3-tts-result-selftest.XXXXXX")"
  printf '%s\n' \
    'test qwen3_tts_real_cpu_matches_official_reference ... ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s' \
    'QWEN3_TTS_PARITY variant=0.6b-base backend=cpu prompt_ids=exact codes_exact=PASS pcm=MEASURED_NOT_GATED' \
    'QWEN3_TTS_PARITY variant=0.6b-customvoice backend=cpu prompt_ids=exact codes_exact=PASS pcm=MEASURED_NOT_GATED' \
    'QWEN3_TTS_PARITY variant=1.7b-base backend=cpu prompt_ids=exact codes_exact=PASS pcm=MEASURED_NOT_GATED' \
    'QWEN3_TTS_PARITY variant=1.7b-customvoice backend=cpu prompt_ids=exact codes_exact=PASS pcm=MEASURED_NOT_GATED' > "$result_probe"
  if ! require_exact_test_result "$result_probe" qwen3_tts_real_cpu_matches_official_reference; then failed=1; fi
  duplicate_probe="$(mktemp "${TMPDIR:-/tmp}/qwen3-tts-result-duplicate.XXXXXX")"
  cat "$result_probe" "$result_probe" > "$duplicate_probe"
  if require_exact_test_result "$duplicate_probe" qwen3_tts_real_cpu_matches_official_reference; then failed=1; fi
  malformed_probe="$(mktemp "${TMPDIR:-/tmp}/qwen3-tts-result-malformed.XXXXXX")"
  { sed -n '1p' "$result_probe"; echo 'test result: ok. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out'; sed -n '2,$p' "$result_probe"; } > "$malformed_probe"
  if require_exact_test_result "$malformed_probe" qwen3_tts_real_cpu_matches_official_reference; then failed=1; fi
  if require_exact_marker "$result_probe" 'QWEN3_TTS_PARITY variant=0.6b-base backend=metal prompt_ids=exact codes_exact=PASS pcm=MEASURED_NOT_GATED'; then failed=1; fi
  rm -f "$result_probe" "$duplicate_probe" "$malformed_probe"
  # shellcheck disable=SC2016
  gate_line="$(grep -n '^  preflight "\$approval"; require_tooling;' "$script_path" | cut -d: -f1)"
  sync_line="$(grep -n 'uv sync --project' "$script_path" | tail -n 1 | cut -d: -f1)"
  [[ "$gate_line" =~ ^[0-9]+$ && "$sync_line" =~ ^[0-9]+$ && "$gate_line" -lt "$sync_line" ]] || { log 'self-test gate ordering is invalid'; failed=1; }
  local sandbox trace worker_log real_uv fake_bin fake_uv fake_curl fake_cargo rc
  sandbox="$(mktemp -d "${TMPDIR:-/tmp}/qwen3-tts-worker-selftest.XXXXXX")"
  trace="$(mktemp "${TMPDIR:-/tmp}/qwen3-tts-worker-trace.XXXXXX")"
  worker_log="$(mktemp "${TMPDIR:-/tmp}/qwen3-tts-worker-log.XXXXXX")"
  real_uv="$(command -v uv)"; fake_bin="$sandbox/bin"
  mkdir -p "$fake_bin" "$sandbox/tools/parity/qwen3_tts" "$sandbox/scripts/publish/vast-ai" "$sandbox/tests/parity/utmos"
  cp "$script_path" "$sandbox/scripts/publish/vast-ai/run-qwen3-tts-validation.sh"
  cp "$LICENSE_GATE" "$sandbox/tools/parity/qwen3_tts/license_gate.py"
  cp "$LICENSE_MANIFEST" "$sandbox/tools/parity/qwen3_tts/license_gate_manifest.json"
  cp "$PARITY_PROJECT/pyproject.toml" "$sandbox/tools/parity/qwen3_tts/pyproject.toml"
  cp "$PARITY_PROJECT/uv.lock" "$sandbox/tools/parity/qwen3_tts/uv.lock"
  cp "$REFERENCE_AUDIO" "$sandbox/tests/parity/utmos/ref-clip.wav"
  printf '{}\n' > "$sandbox/approval.json"
  : > "$sandbox/tools/parity/qwen3_tts/dump_reference.py"
  : > "$sandbox/Cargo.toml"
  git -C "$sandbox" init -q
  git -C "$sandbox" config user.email qwen3-tts-selftest@example.invalid
  git -C "$sandbox" config user.name qwen3-tts-selftest
  fake_uv="$fake_bin/uv"; fake_curl="$fake_bin/curl"; fake_cargo="$fake_bin/cargo"
  # shellcheck disable=SC2016
  printf '%s\n' '#!/usr/bin/env bash' 'printf "uv %s\n" "$*" >> "$QWEN3_TTS_SELFTEST_TRACE"' 'if [[ " $* " == *" --no-project "* ]]; then exec "$QWEN3_TTS_SELFTEST_REAL_UV" "$@"; fi' 'exit 97' > "$fake_uv"
  # shellcheck disable=SC2016
  printf '%s\n' '#!/usr/bin/env bash' 'printf "curl %s\n" "$*" >> "$QWEN3_TTS_SELFTEST_TRACE"' 'exit 97' > "$fake_curl"
  # shellcheck disable=SC2016
  printf '%s\n' '#!/usr/bin/env bash' 'printf "cargo %s\n" "$*" >> "$QWEN3_TTS_SELFTEST_TRACE"' 'exit 97' > "$fake_cargo"
  chmod +x "$fake_uv" "$fake_curl" "$fake_cargo"
  git -C "$sandbox" add .
  git -C "$sandbox" -c user.email=qwen3-tts-selftest@example.invalid -c user.name=qwen3-tts-selftest commit -qm self-test
  printf '%s\n' dirty > "$sandbox/dirty-unrelated-file"
  set +e
  PATH="$fake_bin:$PATH" HOME="$sandbox/home" VOKRA_ROOT="$sandbox" VOKRA_SCRATCH="$sandbox/scratch" VOKRA_PUBLISH_ON_VAST=1 \
    QWEN3_TTS_SELFTEST_TRACE="$trace" QWEN3_TTS_SELFTEST_REAL_UV="$real_uv" \
    bash "$sandbox/scripts/publish/vast-ai/run-qwen3-tts-validation.sh" --approval-evidence "$sandbox/approval.json" >"$worker_log" 2>&1
  rc=$?
  set -e
  [[ "$rc" -eq 2 ]] || { log "self-test blocked worker returned $rc, expected 2"; failed=1; }
  [[ ! -s "$trace" ]] || { log "self-test blocked worker reached a gate, sync, download, or build: $(sed -n l "$worker_log")"; failed=1; }
  [[ ! -d "$sandbox/scratch" ]] || { log 'self-test worker created scratch output before the gate'; failed=1; }
  rm -rf -- "$sandbox" "$trace" "$worker_log"
  (( failed == 0 )) || return 1
  echo 'run-qwen3-tts-validation.sh self-test: PASS'
}

run_variant() {
  local variant="$1" work_dir="$2" evidence="$3" source_tree="$4"
  local repo revision model_kind source gguf ref
  repo="$(variant_repo "$variant")"; revision="$(variant_revision "$variant")"; model_kind="$(variant_model_kind "$variant")"
  source="$work_dir/source-$variant"; gguf="$work_dir/$model_kind.gguf"; ref="$evidence/reference-$variant"
  step "Download exact $repo@$revision"
  download_snapshot "$repo" "$revision" "$source"
  require_single_file_snapshot "$source"
  step "Convert corrected $model_kind GGUF on VAST"
  "$VOKRA_ROOT/target/release/vokra-cli" convert --model "$model_kind" --input "$source/model.safetensors" --output "$gguf" --license apache-2.0 2>&1 | tee "$evidence/convert-$variant.log"
  [[ -s "$gguf" ]] || die "converter emitted no GGUF for $variant"
  step "Generate independent official reference for $variant"
  local audio_args=()
  [[ "$variant" == *-base ]] && audio_args=(--reference-audio "$REFERENCE_AUDIO")
  PYTHONPATH="$source_tree${PYTHONPATH:+:$PYTHONPATH}" uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python "$REFERENCE_DUMPER" --variant "$variant" --model-dir "$source" --decoder-dir "$work_dir/source-decoder" --source-dir "$source_tree" --output "$ref" "${audio_args[@]}" 2>&1 | tee "$evidence/reference-$variant.log"
  verify_reference_hashes "$ref"
  step "Hash and record $variant corrected inputs"
  {
    echo "variant=$variant"; echo "upstream_repo=$repo"; echo "upstream_revision=$revision"
    echo "official_source_repo=$OFFICIAL_SOURCE_REPO"; echo "official_source_revision=$OFFICIAL_SOURCE_REVISION"
    echo "config_bytes=$(variant_config_bytes "$variant")"; echo "config_sha256=$(variant_config_sha256 "$variant")"
    echo "decoder_repo=$DECODER_REPO"; echo "decoder_revision=$DECODER_REVISION"; echo "decoder_checkpoint_sha256=$DECODER_CHECKPOINT_SHA256"
    echo "gguf_sha256=$(sha256_file "$gguf")"; echo "reference_manifest_sha256=$(sha256_file "$ref/manifest.json")"
    echo "runtime_artifact=corrected_vokra_conversion"; echo "public_precontract_artifact=NOT_USED"; echo "upload=NOT_PERFORMED"
  } > "$evidence/inputs-$variant.txt"
}

main() {
  local selection='all' work_dir='' approval='' self_test=0 variant_seen=0
  while (( $# > 0 )); do
    case "$1" in
      --variant) [[ $# -ge 2 && -n "$2" && "$2" != -* && "$variant_seen" == 0 ]] || { usage; return 2; }; selection="$2"; variant_seen=1; shift 2 ;;
      --work-dir) [[ $# -ge 2 && -n "$2" && "$2" != -* && -z "$work_dir" ]] || { usage; return 2; }; work_dir="$2"; shift 2 ;;
      --approval-evidence) [[ $# -ge 2 && -n "$2" && "$2" != -* && -z "$approval" ]] || { usage; return 2; }; approval="$2"; shift 2 ;;
      --self-test) self_test=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) usage; die "unknown argument: $1" ;;
    esac
  done
  if (( self_test == 1 )); then
    [[ "$selection" == all && -z "$work_dir" && -z "$approval" ]] || die '--self-test accepts no other arguments'
    local saved_status="$TRANSFORMERS_COMPATIBILITY_STATUS"
    TRANSFORMERS_COMPATIBILITY_STATUS='BLOCKED_UNVERIFIED_API_SMOKE'; require_transformers_api_smoke && return 1 || :
    TRANSFORMERS_COMPATIBILITY_STATUS='AUTHENTICATED_API_SMOKE'; require_transformers_api_smoke || return 1
    TRANSFORMERS_COMPATIBILITY_STATUS='UNKNOWN_STATUS'; require_transformers_api_smoke && return 1 || :
    TRANSFORMERS_COMPATIBILITY_STATUS="$saved_status"
    run_self_test; return
  fi
  case "$selection" in all) ;; *) die 'this four-variant validation requires --variant all' ;; esac
  [[ -n "$approval" ]] || { usage; die '--approval-evidence is required'; }
  require_transformers_api_smoke
  preflight "$approval"; require_tooling; require_vast_host
  [[ -n "$work_dir" ]] || work_dir="$VOKRA_SCRATCH/qwen3-tts-validation-$(git -C "$VOKRA_ROOT" rev-parse --short=12 HEAD)"
  require_absent_work_dir "$work_dir" "$approval"
  mkdir -p "$work_dir"
  local evidence="$work_dir/evidence" decoder_source decoder_gguf source_tree
  mkdir -p "$evidence"
  export HF_HOME="$work_dir/hf-home" HF_HUB_CACHE="$work_dir/hf-home/hub"
  exec > >(tee -a "$evidence/run.log") 2>&1
  step 'Install frozen official reference environment'
  uv sync --project "$PARITY_PROJECT" --frozen --python 3.12
  step 'Build Vokra CLI on VAST'
  cargo build --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release -p vokra-cli 2>&1 | tee "$evidence/build-cli.log"
  step "Stage authenticated official source $OFFICIAL_SOURCE_REPO@$OFFICIAL_SOURCE_REVISION"
  source_tree="$work_dir/source-qwen3-tts"
  download_source_tree "$source_tree"
  step "Download and convert official decoder $DECODER_REPO@$DECODER_REVISION"
  decoder_source="$work_dir/source-decoder"; decoder_gguf="$work_dir/qwen3-tts-tokenizer-12hz.gguf"
  download_snapshot "$DECODER_REPO" "$DECODER_REVISION" "$decoder_source"
  require_single_file_snapshot "$decoder_source"
  [[ "$(sha256_file "$decoder_source/model.safetensors")" == "$DECODER_CHECKPOINT_SHA256" ]] || die 'official decoder checkpoint SHA-256 drifted'
  "$VOKRA_ROOT/target/release/vokra-cli" convert --model qwen3-tts-tokenizer-12hz --input "$decoder_source/model.safetensors" --output "$decoder_gguf" --license apache-2.0 2>&1 | tee "$evidence/convert-decoder.log"
  [[ -s "$decoder_gguf" ]] || die 'decoder converter emitted no GGUF'
  for variant in 0.6b-base 0.6b-customvoice 1.7b-base 1.7b-customvoice; do
    [[ "$selection" == all || "$selection" == "$variant" ]] || continue
    run_variant "$variant" "$work_dir" "$evidence" "$source_tree"
  done
  {
    printf 'scripts/verify/apple-silicon-qwen3-tts.sh'
    printf " --gguf-0.6b-base '%s' --gguf-0.6b-base-sha256 %q" '<APPLE_QWEN3_TTS_0_6B_BASE_GGUF>' "$(sha256_file "$work_dir/qwen3-tts-12hz-0.6b-base.gguf")"
    printf " --gguf-0.6b-customvoice '%s' --gguf-0.6b-customvoice-sha256 %q" '<APPLE_QWEN3_TTS_0_6B_CUSTOMVOICE_GGUF>' "$(sha256_file "$work_dir/qwen3-tts-12hz-0.6b-customvoice.gguf")"
    printf " --gguf-1.7b-base '%s' --gguf-1.7b-base-sha256 %q" '<APPLE_QWEN3_TTS_1_7B_BASE_GGUF>' "$(sha256_file "$work_dir/qwen3-tts-12hz-1.7b-base.gguf")"
    printf " --gguf-1.7b-customvoice '%s' --gguf-1.7b-customvoice-sha256 %q" '<APPLE_QWEN3_TTS_1_7B_CUSTOMVOICE_GGUF>' "$(sha256_file "$work_dir/qwen3-tts-12hz-1.7b-customvoice.gguf")"
    printf " --decoder-gguf '%s' --decoder-gguf-sha256 %q" '<APPLE_QWEN3_TTS_DECODER_GGUF>' "$(sha256_file "$decoder_gguf")"
    printf " --reference-0.6b-base '%s' --reference-0.6b-base-sha256 %q" '<APPLE_QWEN3_TTS_0_6B_BASE_REFERENCE>' "$(sha256_file "$evidence/reference-0.6b-base/manifest.json")"
    printf " --reference-0.6b-customvoice '%s' --reference-0.6b-customvoice-sha256 %q" '<APPLE_QWEN3_TTS_0_6B_CUSTOMVOICE_REFERENCE>' "$(sha256_file "$evidence/reference-0.6b-customvoice/manifest.json")"
    printf " --reference-1.7b-base '%s' --reference-1.7b-base-sha256 %q" '<APPLE_QWEN3_TTS_1_7B_BASE_REFERENCE>' "$(sha256_file "$evidence/reference-1.7b-base/manifest.json")"
    printf " --reference-1.7b-customvoice '%s' --reference-1.7b-customvoice-sha256 %q" '<APPLE_QWEN3_TTS_1_7B_CUSTOMVOICE_REFERENCE>' "$(sha256_file "$evidence/reference-1.7b-customvoice/manifest.json")"
    printf " --approval-evidence '%s' --evidence-dir '%s'\n" '<APPLE_QWEN3_TTS_APPROVAL_EVIDENCE>' '<APPLE_QWEN3_TTS_EVIDENCE_DIR>'
  } > "$evidence/apple-gguf-sha256-args.txt"
  step 'Run four-variant real CPU parity on VAST'
  local env_args=()
  for variant in 0.6b-base 0.6b-customvoice 1.7b-base 1.7b-customvoice; do
    [[ "$selection" == all || "$selection" == "$variant" ]] || continue
    env_args+=("$(variant_env_prefix "$variant")_GGUF=$work_dir/$(variant_model_kind "$variant").gguf")
    env_args+=("$(variant_env_prefix "$variant")_DECODER_GGUF=$decoder_gguf")
    env_args+=("$(variant_env_prefix "$variant")_REFERENCE_DIR=$evidence/reference-$variant")
  done
  env "${env_args[@]}" RUST_TEST_THREADS=1 cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release -p vokra-models --test qwen3_tts_real qwen3_tts_real_cpu_matches_official_reference -- --ignored --exact --nocapture --test-threads=1 2>&1 | tee "$evidence/parity-cpu.log"
  require_exact_test_result "$evidence/parity-cpu.log" qwen3_tts_real_cpu_matches_official_reference
  for variant in 0.6b-base 0.6b-customvoice 1.7b-base 1.7b-customvoice; do
    require_exact_marker "$evidence/parity-cpu.log" "QWEN3_TTS_PARITY variant=$variant backend=cpu prompt_ids=exact codes_exact=PASS pcm=MEASURED_NOT_GATED"
  done
  {
    echo 'verdict=MEASURED_NOT_GATED'; echo 'numeric_bound=UNSET'; echo "min_new_tokens=$MIN_NEW_TOKENS"; echo 'previous_isolated_transformers_pin=transformers==4.57.3'; echo 'transformers_security_advisory=GHSA-xrqw-3rrv-vx5w'; echo 'transformers_security_patched_minimum=5.10.0'; echo "isolated_transformers_pin=transformers==$TRANSFORMERS_VERSION"; echo "transformers_compatibility_status=$TRANSFORMERS_COMPATIBILITY_STATUS"; echo 'nested_decoder_sha256=validated_in_reference'; echo "decoder_gguf_sha256=$(sha256_file "$decoder_gguf")"; echo "official_source_revision=$OFFICIAL_SOURCE_REVISION"; echo 'public_precontract_artifacts=NOT_USED'; echo 'upload=NOT_PERFORMED'
  } > "$evidence/summary.txt"
  (cd "$work_dir" && find evidence -type f -print0 | sort -z | xargs -0 sha256sum > evidence/SHA256SUMS)
  log 'MEASURED_NOT_GATED: pull evidence only, then destroy the VAST instance'
}

main "$@"
