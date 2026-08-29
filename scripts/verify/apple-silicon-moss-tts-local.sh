#!/usr/bin/env bash
# Disposable Apple-Silicon consumer for the VAST MOSS-TTS Local composite.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
LOCAL_PROJECT="$VOKRA_ROOT/tools/parity/moss_tts_local"
LOCAL_GATE="$LOCAL_PROJECT/preflight_gate.py"
LOCAL_MANIFEST="$LOCAL_PROJECT/license_gate_manifest.json"
V2_PROJECT="$VOKRA_ROOT/tools/parity/moss_audio_tokenizer_v2"
V2_GATE="$V2_PROJECT/license_gate.py"
V2_MANIFEST="$V2_PROJECT/license_gate_manifest.json"
V2_REPOSITORY="OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
V2_REVISION="f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169"
V2_MODEL_SOURCE_SHA256="7f807e6ee77a60d512e5aa4a8f58a1d5af4e3722f4ab350d70dd538429391cb9"
V2_CONFIG_SOURCE_SHA256="f87a7a975868ce3f0077f374f46ebd2aab610fd7a26cd7569d16827a14e29529"
MIN_MEMORY_BYTES=32000000000
MIN_FREE_DISK_KIB=20000000

log() { printf '[moss-tts-local-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }
sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
require_hash() {
  local path="$1" expected="$2"
  [[ -n "$path" && -f "$path" && ! -L "$path" ]] || { die "input is not a regular non-symlink file: $path"; return 2; }
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || { die "expected hash is not lowercase SHA-256: $path"; return 2; }
  [[ "$(sha256_file "$path")" == "$expected" ]] || { die "input SHA-256 mismatch: $path"; return 2; }
}
require_approval_file() {
  local path="$1"
  [[ -n "$path" && -f "$path" && ! -L "$path" && -s "$path" ]] || { die "approval must be a nonempty regular non-symlink file: $path"; return 2; }
}
canonical_candidate() {
  local value="$1" suffix='' parent rest component scan
  [[ "$value" = /* ]] || value="$PWD/$value"
  rest="${value#/}"; scan=''
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; rest="${rest#*/}"
    [[ "$component" == "$rest" ]] && rest=''
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || continue
    scan="$scan/$component"
    [[ ! -L "$scan" || "$scan" == "/var" ]] || { die "path contains symlink component: $scan"; return 2; }
  done
  while [[ ! -e "$value" && ! -L "$value" ]]; do
    parent="$(dirname "$value")"
    suffix="/$(basename "$value")$suffix"
    [[ "$parent" != "$value" ]] || { die "path has no canonical parent: $value"; return 2; }
    value="$parent"
  done
  [[ -d "$value" && ! -L "$value" ]] || { die "path parent is not a real directory: $value"; return 2; }
  (cd -P "$value" && printf '%s%s\n' "$PWD" "$suffix")
}
paths_overlap() {
  local left="${1%/}" right="${2%/}"
  [[ "$left" == "$right" || "$left" == "$right"/* || "$right" == "$left"/* ]]
}
validate_evidence_target() {
  local evidence="$1" input input_dir canonical_evidence canonical_input canonical_root
  shift
  [[ -n "$evidence" && ! -e "$evidence" && ! -L "$evidence" ]] || { die 'evidence directory must be absent before run'; return 2; }
  canonical_evidence="$(canonical_candidate "$evidence")" || return 2
  canonical_root="$(canonical_candidate "$VOKRA_ROOT")" || return 2
  paths_overlap "$canonical_evidence" "$canonical_root" && { die 'evidence must be outside the Vokra checkout'; return 2; }
  for input in "$@"; do
    input_dir="$(dirname "$input")"
    if [[ -d "$input" ]]; then
      canonical_input="$(canonical_candidate "$input")" || return 2
    else
      canonical_input="$(canonical_candidate "$input_dir")/$(basename "$input")" || return 2
    fi
    paths_overlap "$canonical_evidence" "$canonical_input" && { die 'evidence must be separate from the transferred input bundle'; return 2; }
  done
  return 0
}
pre_sync_gates() {
  local local_approval="$1" v2_approval="$2" rc=0
  command -v uv >/dev/null 2>&1 || { die 'uv is required before approval gates'; return 2; }
  [[ -f "$LOCAL_GATE" && -f "$LOCAL_MANIFEST" && -f "$V2_GATE" && -f "$V2_MANIFEST" ]] || { die 'both dedicated gate inputs are missing'; return 2; }
  require_approval_file "$local_approval" || rc=$?
  require_approval_file "$v2_approval" || rc=$?
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$LOCAL_GATE" \
    --project "$LOCAL_PROJECT" --manifest "$LOCAL_MANIFEST" --approval-evidence "$local_approval" || rc=$?
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$V2_GATE" \
    --lock "$V2_PROJECT/uv.lock" --project "$V2_PROJECT/pyproject.toml" \
    --manifest "$V2_MANIFEST" --approval "$v2_approval" || rc=$?
  return "$rc"
}
require_prompt_shape() {
  local path="$1" bytes; bytes="$(wc -c < "$path" | tr -d '[:space:]')"
  [[ "$bytes" =~ ^[0-9]+$ && "$bytes" -gt 0 && $((bytes % 52)) -eq 0 ]] || { die "prompt/reference rows are not non-empty [rows,13] u32le"; return 2; }
}
require_codes_shape() {
  local path="$1" bytes; bytes="$(wc -c < "$path" | tr -d '[:space:]')"
  [[ "$bytes" =~ ^[0-9]+$ && "$bytes" -gt 0 && $((bytes % 48)) -eq 0 ]] || { die 'assistant codes are not non-empty [rows,12] u32le'; return 2; }
}
require_remote_host() {
  local memory_bytes free_disk_kib
  [[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == 1 ]] || die 'VOKRA_REMOTE_APPLE_SILICON=1 is required'
  [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || die 'disposable Darwin arm64 is required'
  memory_bytes="$(sysctl -n hw.memsize)"
  [[ "$memory_bytes" =~ ^[0-9]+$ && "$memory_bytes" -ge "$MIN_MEMORY_BYTES" ]] || die 'Apple host must have at least 32 GB RAM'
  free_disk_kib="$(df -Pk "$VOKRA_ROOT" | awk 'NR == 2 {print $4}')"
  [[ "$free_disk_kib" =~ ^[0-9]+$ && "$free_disk_kib" -ge "$MIN_FREE_DISK_KIB" ]] || die 'Apple host must have at least 20 GB free disk'
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" ]] || die 'Vokra checkout is missing'
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'Apple checkout must be clean'
  for tool in cargo rustc git shasum awk find grep tee sysctl df wc tr uv; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  xcrun -f metal >/dev/null 2>&1 || die 'Xcode Metal compiler is unavailable'
}
verify_reference_manifest() {
  local manifest="$1" expected_manifest="$2" prompt_path="$3" rows_path="$4" codes_path="$5" prompt_sha="$6" rows_sha="$7" codes_sha="$8"
  require_hash "$manifest" "$expected_manifest"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python \
    "$VOKRA_ROOT/tools/parity/moss_tts_local/reference_validator.py" \
    "$manifest" "$prompt_path" "$rows_path" "$codes_path" "$prompt_sha" "$rows_sha" "$codes_sha"
}
verify_v2_reference() {
  local reference="$1"
  [[ -f "$reference" && ! -L "$reference" ]] || die 'v2 reference is not a regular file'
  [[ "$(wc -l < "$reference" | tr -d '[:space:]')" == 22 ]] || { die 'v2 reference has unexpected row count'; return 2; }
  [[ "$(grep -Fxc "source,v2,$V2_REPOSITORY,$V2_REVISION" "$reference" || true)" == 1 ]] || { die 'v2 reference source identity is not a singleton'; return 2; }
  [[ "$(grep -Ec '^runtime,torch-[^,]+,transformers-[^,]+$' "$reference" || true)" == 1 ]] || { die 'v2 reference runtime row is not a singleton'; return 2; }
  [[ "$(grep -Fxc 'environment,device,cuda' "$reference" || true)" == 1 ]] || { die 'v2 reference is not a CUDA reference'; return 2; }
  [[ "$(grep -Fxc 'contract,1,12,1024,48000,2,3840' "$reference" || true)" == 1 ]] || { die 'v2 reference contract drifted'; return 2; }
  [[ "$(grep -Ec '^codes(,[0-9]+){12}$' "$reference" || true)" == 1 ]] || { die 'v2 code packet schema/cardinality drifted'; return 2; }
  [[ "$(grep -Ec '^environment,cpu,[^,]+,machine-[^,]+,logical-[0-9]+,torch-capability-.+$' "$reference" || true)" == 1 ]] || { die 'v2 CPU environment row schema/cardinality drifted'; return 2; }
  for row in model config; do
    local expected
    if [[ "$row" == model ]]; then expected="$V2_MODEL_SOURCE_SHA256"; else expected="$V2_CONFIG_SOURCE_SHA256"; fi
    [[ "$(awk -F, -v expected="$expected" -v row="$row" '$1 == "source_file" && $2 == row && NF == 4 && $3 ~ /transformers_modules/ && $4 == expected {count++} END {print count + 0}' "$reference")" == 1 ]] || { die "v2 source-file row is not authenticated: $row"; return 2; }
  done
  for label in decoder_0 decoder_1 decoder_2 decoder_3 decoder_4 decoder_5 decoder_6 decoder_7 decoder_8 decoder_9 decoder_10 decoder_11 quantizer audio; do
    [[ "$(awk -F, -v label="$label" '$1 == "tensor" && $2 == label {count++} END {print count + 0}' "$reference")" == 1 ]] || { die "v2 tensor row is missing or duplicated: $label"; return 2; }
  done
  [[ "$(grep -Ec '^(source|runtime|environment|source_file|contract|codes|tensor),' "$reference" || true)" == 22 ]] || { die 'v2 reference contains extra or malformed rows'; return 2; }
  [[ "$(grep -Ec '^tensor,(decoder_(0|[1-9]|10|11)|quantizer|audio),[^,]+,.+$' "$reference" || true)" == 14 ]] || { die 'v2 tensor row family/cardinality drifted'; return 2; }
}
require_one_result() {
  local log_path="$1" backend="$2" name='moss_tts::local_transformer::tests::measure_local_real_cpu_and_optional_metal_against_official'
  local test_count ok_count name_count result_count total_result rows_count codes_count pcm_count composite_count cpu_rows_mentions cpu_codes_mentions cpu_pcm_mentions metal_rows_mentions metal_codes_mentions metal_pcm_mentions
  test_count="$(grep -Ec '^test .* \.\.\.' "$log_path" || true)"; ok_count="$(grep -Ec "^test ${name} \.\.\. ok$" "$log_path" || true)"; name_count="$(grep -Ec "^test ${name} \.\.\." "$log_path" || true)"
  result_count="$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$log_path" || true)"; total_result="$(grep -Ec '^test result:' "$log_path" || true)"
  if [[ "$backend" == cpu ]]; then
    rows_count="$(grep -Ec '^MOSS_TTS_LOCAL_ROWS_MEASURED backend=cpu exact=true differing_values=0$' "$log_path" || true)"
    codes_count="$(grep -Ec '^MOSS_TTS_LOCAL_CODES_MEASURED backend=cpu exact=true differing_values=0$' "$log_path" || true)"
    pcm_count="$(grep -Ec '^MOSS_TTS_LOCAL_PCM_MEASURED backend=cpu samples=[0-9]+ channels=[0-9]+ rms=[0-9eE.+-]+ peak=[0-9eE.+-]+$' "$log_path" || true)"
  else
    rows_count="$(grep -Ec '^MOSS_TTS_LOCAL_METAL_ROWS_MEASURED exact_to_cpu=true differing_values=0$' "$log_path" || true)"
    codes_count="$(grep -Ec '^MOSS_TTS_LOCAL_METAL_CODES_MEASURED exact_to_cpu=true exact_to_reference=true differing_values=0$' "$log_path" || true)"
    pcm_count="$(grep -Ec '^MOSS_TTS_LOCAL_METAL_PCM_MEASURED samples=[0-9]+ channels=[0-9]+ rms=[0-9eE.+-]+ peak=[0-9eE.+-]+ max_abs_to_cpu=[0-9eE.+-]+$' "$log_path" || true)"
  fi
  composite_count="$(grep -Ec '^COMPOSITE_PCM_NOT_RUN reason=official_v2_pcm_sidecar_not_supplied$' "$log_path" || true)"
  cpu_rows_mentions="$(grep -Ec 'MOSS_TTS_LOCAL_ROWS_MEASURED' "$log_path" || true)"; cpu_codes_mentions="$(grep -Ec 'MOSS_TTS_LOCAL_CODES_MEASURED' "$log_path" || true)"; cpu_pcm_mentions="$(grep -Ec 'MOSS_TTS_LOCAL_PCM_MEASURED' "$log_path" || true)"
  metal_rows_mentions="$(grep -Ec 'MOSS_TTS_LOCAL_METAL_ROWS_MEASURED' "$log_path" || true)"; metal_codes_mentions="$(grep -Ec 'MOSS_TTS_LOCAL_METAL_CODES_MEASURED' "$log_path" || true)"; metal_pcm_mentions="$(grep -Ec 'MOSS_TTS_LOCAL_METAL_PCM_MEASURED' "$log_path" || true)"
  if [[ "$backend" == cpu ]]; then
    [[ "$cpu_rows_mentions" == 1 && "$cpu_codes_mentions" == 1 && "$cpu_pcm_mentions" == 1 && "$metal_rows_mentions" == 0 && "$metal_codes_mentions" == 0 && "$metal_pcm_mentions" == 0 ]] || { die 'CPU sentinel family is not exact'; return 2; }
  else
    local cpu_rows_count cpu_codes_count cpu_pcm_count
    cpu_rows_count="$(grep -Ec '^MOSS_TTS_LOCAL_ROWS_MEASURED backend=cpu exact=true differing_values=0$' "$log_path" || true)"; cpu_codes_count="$(grep -Ec '^MOSS_TTS_LOCAL_CODES_MEASURED backend=cpu exact=true differing_values=0$' "$log_path" || true)"; cpu_pcm_count="$(grep -Ec '^MOSS_TTS_LOCAL_PCM_MEASURED backend=cpu samples=[0-9]+ channels=[0-9]+ rms=[0-9eE.+-]+ peak=[0-9eE.+-]+$' "$log_path" || true)"
    [[ "$cpu_rows_count" == 1 && "$cpu_codes_count" == 1 && "$cpu_pcm_count" == 1 && "$cpu_rows_mentions" == 1 && "$cpu_codes_mentions" == 1 && "$cpu_pcm_mentions" == 1 && "$metal_rows_mentions" == 1 && "$metal_codes_mentions" == 1 && "$metal_pcm_mentions" == 1 ]] || { die 'Metal sentinel family is not exact'; return 2; }
  fi
  [[ "$test_count" == 1 && "$ok_count" == 1 && "$name_count" == 1 && "$result_count" == 1 && "$total_result" == 1 && "$rows_count" == 1 && "$codes_count" == 1 && "$pcm_count" == 1 && "$composite_count" == 1 ]] || { die "${backend} Cargo/sentinel lines are not exact singletons"; return 2; }
  ! grep -Eq '^MOSS_TTS_LOCAL_(ROWS|CODES|PCM)_MEASURED .*FAIL$' "$log_path" || { die "${backend} measurement FAIL marker present"; return 2; }
}
self_test() {
  local tmp name gate_line host_line; tmp="$(mktemp -d "${TMPDIR:-/tmp}/moss-tts-local-apple.XXXXXX")"; trap 'rm -rf "${tmp:-}"' EXIT
  gate_line="$(grep -n '^pre_sync_gates()' "$0" | head -1 | cut -d: -f1)"
  host_line="$(grep -n '^require_remote_host()' "$0" | head -1 | cut -d: -f1)"
  (( gate_line > 0 && host_line > gate_line )) || die 'approval gates are not before host attestation'
  name='moss_tts::local_transformer::tests::measure_local_real_cpu_and_optional_metal_against_official'
  printf '%s\n' "test $name ... ok" 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' 'MOSS_TTS_LOCAL_ROWS_MEASURED backend=cpu exact=true differing_values=0' 'MOSS_TTS_LOCAL_CODES_MEASURED backend=cpu exact=true differing_values=0' 'MOSS_TTS_LOCAL_PCM_MEASURED backend=cpu samples=3840 channels=2 rms=0.000000000e+00 peak=0.000000000e+00' 'COMPOSITE_PCM_NOT_RUN reason=official_v2_pcm_sidecar_not_supplied' > "$tmp/log"; require_one_result "$tmp/log" cpu
  for malformed in extra failed failed-plus-ok duplicate malformed-suffix fail prefix nonzero; do
    cp "$tmp/log" "$tmp/$malformed"; case "$malformed" in
      extra) printf '%s\n' 'test another::test ... ok' >> "$tmp/$malformed";; failed) sed 's/\.\.\. ok$/... FAILED/' "$tmp/$malformed" > "$tmp/x" && mv "$tmp/x" "$tmp/$malformed";; failed-plus-ok) sed 's/\.\.\. ok$/... FAILED/' "$tmp/$malformed" > "$tmp/x" && printf '%s\n' "test $name ... ok" >> "$tmp/x" && mv "$tmp/x" "$tmp/$malformed";; duplicate) printf '%s\n' 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' >> "$tmp/$malformed";; malformed-suffix) sed 's/filtered out$/filtered out; finished in nope/' "$tmp/$malformed" > "$tmp/x" && mv "$tmp/x" "$tmp/$malformed";; fail) printf '%s\n' 'MOSS_TTS_LOCAL_PCM_MEASURED backend=cpu FAIL' >> "$tmp/$malformed";; prefix) sed 's/^MOSS_TTS_LOCAL_ROWS_MEASURED /prefix MOSS_TTS_LOCAL_ROWS_MEASURED /' "$tmp/$malformed" > "$tmp/x" && mv "$tmp/x" "$tmp/$malformed";; nonzero) sed 's/differing_values=0/differing_values=1/g' "$tmp/$malformed" > "$tmp/x" && mv "$tmp/x" "$tmp/$malformed";; esac
    if require_one_result "$tmp/$malformed" cpu >/dev/null 2>&1; then die "accepted malformed Cargo evidence: $malformed"; fi
  done
  printf '%s\n' "test $name ... ok" 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' 'MOSS_TTS_LOCAL_ROWS_MEASURED backend=cpu exact=true differing_values=0' 'MOSS_TTS_LOCAL_CODES_MEASURED backend=cpu exact=true differing_values=0' 'MOSS_TTS_LOCAL_PCM_MEASURED backend=cpu samples=3840 channels=2 rms=0.000000000e+00 peak=0.000000000e+00' 'COMPOSITE_PCM_NOT_RUN reason=official_v2_pcm_sidecar_not_supplied' 'MOSS_TTS_LOCAL_METAL_ROWS_MEASURED exact_to_cpu=true differing_values=0' 'MOSS_TTS_LOCAL_METAL_CODES_MEASURED exact_to_cpu=true exact_to_reference=true differing_values=0' 'MOSS_TTS_LOCAL_METAL_PCM_MEASURED samples=3840 channels=2 rms=0.000000000e+00 peak=0.000000000e+00 max_abs_to_cpu=0.000000000e+00' > "$tmp/metal.log"
  require_one_result "$tmp/metal.log" metal
  for malformed in missing-cpu extra-metal metal-prefix metal-fail metal-nonzero; do
    cp "$tmp/metal.log" "$tmp/$malformed"
    case "$malformed" in
      missing-cpu) sed -E '/^MOSS_TTS_LOCAL_(ROWS|CODES|PCM)_MEASURED /d' "$tmp/$malformed" > "$tmp/x" && mv "$tmp/x" "$tmp/$malformed";;
      extra-metal) printf '%s\n' 'MOSS_TTS_LOCAL_METAL_ROWS_MEASURED exact_to_cpu=true differing_values=0' >> "$tmp/$malformed";;
      metal-prefix) sed 's/^MOSS_TTS_LOCAL_METAL_ROWS_MEASURED /prefix MOSS_TTS_LOCAL_METAL_ROWS_MEASURED /' "$tmp/$malformed" > "$tmp/x" && mv "$tmp/x" "$tmp/$malformed";;
      metal-fail) printf '%s\n' 'MOSS_TTS_LOCAL_METAL_PCM_MEASURED FAIL' >> "$tmp/$malformed";;
      metal-nonzero) sed 's/differing_values=0/differing_values=1/g' "$tmp/$malformed" > "$tmp/x" && mv "$tmp/x" "$tmp/$malformed";;
    esac
    if require_one_result "$tmp/$malformed" metal >/dev/null 2>&1; then die "accepted malformed Metal evidence: $malformed"; fi
  done
  if "$0" --self-test unexpected >/dev/null 2>&1; then die '--self-test accepted an extra argument'; fi
  if "$0" --local-gguf only-one-argument >/dev/null 2>&1; then die 'missing explicit input/hash arguments were accepted'; fi
  if "$0" --local-gguf -bad >/dev/null 2>&1; then die 'leading-dash input value was accepted'; fi
  if "$0" --local-approval-evidence >/dev/null 2>&1; then die 'missing Local approval value was accepted'; fi
  if "$0" --v2-approval-evidence >/dev/null 2>&1; then die 'missing v2 approval value was accepted'; fi
  if "$0" --local-approval-evidence -bad >/dev/null 2>&1; then die 'leading-dash Local approval value was accepted'; fi
  if "$0" --v2-approval-evidence -bad >/dev/null 2>&1; then die 'leading-dash v2 approval value was accepted'; fi
  if "$0" --local-approval-evidence a --local-approval-evidence b >/dev/null 2>&1; then die 'duplicate Local approval option was accepted'; fi
  if "$0" --v2-approval-evidence a --v2-approval-evidence b >/dev/null 2>&1; then die 'duplicate v2 approval option was accepted'; fi
  if "$0" --local-gguf a --local-gguf b >/dev/null 2>&1; then die 'duplicate option was accepted'; fi
  if "$0" --unknown value >/dev/null 2>&1; then die 'unknown option was accepted'; fi
  if "$0" trailing >/dev/null 2>&1; then die 'trailing positional argument was accepted'; fi
  printf '%052d' 0 > "$tmp/prompt"
  printf '%052d' 0 > "$tmp/rows"
  printf '%048d' 0 > "$tmp/codes"
  local prompt_sha rows_sha codes_sha manifest_sha
  prompt_sha="$(sha256_file "$tmp/prompt")"; rows_sha="$(sha256_file "$tmp/rows")"; codes_sha="$(sha256_file "$tmp/codes")"
  printf '%s\n' "{\"repository\":\"OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5\",\"revision\":\"be7766a6735b98bd793f7c79fb720b4d0f5d13b8\",\"snapshot\":{\"model\":{\"tensor_count\":438}},\"loaded_custom_code\":[{\"label\":\"a\",\"role\":\"a\",\"path\":\"a\",\"sha256\":\"$prompt_sha\",\"resolved_revision\":\"be7766a6735b98bd793f7c79fb720b4d0f5d13b8\"},{\"label\":\"b\",\"role\":\"b\",\"path\":\"b\",\"sha256\":\"$rows_sha\",\"resolved_revision\":\"be7766a6735b98bd793f7c79fb720b4d0f5d13b8\"}],\"prompt\":{\"path\":\"prompt\",\"sha256\":\"$prompt_sha\",\"rows\":1,\"columns\":13},\"rows_from_audio_start\":{\"path\":\"rows\",\"sha256\":\"$rows_sha\",\"rows\":1,\"columns\":13,\"start_length\":1,\"generated_frames\":1},\"assistant_codes\":{\"path\":\"codes\",\"sha256\":\"$codes_sha\",\"rows\":1,\"codebooks\":12},\"terminal_row_present_in_official_output\":true,\"reference_status\":\"AUTHENTICATED_EVIDENCE_COMPLETE\",\"parity_status\":\"MEASURED_NOT_GATED\"}" > "$tmp/manifest"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$tmp/manifest" "$tmp/prompt" "$tmp/rows" "$tmp/codes" <<'PY'
import hashlib, json, sys
from pathlib import Path
manifest, prompt, rows, codes = map(Path, sys.argv[1:])
repo = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"; rev = "be7766a6735b98bd793f7c79fb720b4d0f5d13b8"
digests = {"configuration":"826f81f163b1b557ad13f83c4f35008f4fee5a6cb6311b4316ff3dbb25149411", "configuration_source":"ab6debcb92032cb9dc91ae80aed77dbadd2e59848208baef2b062bd6def3f3be", "modeling_source":"b0a66211943ae580b087f3e71495fea2f455701a4f6c29b6d3562218f7668c5f", "processing_source":"3fc5616b1ec3408162b7d859a7696725a40525313b20f9b31a06ee55c93bd7ad", "gpt2_source":"f2e877104669f1e6c7cd34680f0da1a8a159e032123ee56b660b63929b6c8989", "qwen3_source":"100163bd7ecf31a59bafacc0b032ace9339edc992a3eb4cc80662502e04e46f0", "processor_config":"db574bfebad009e05193196a63a4eeecd353eeca177ccfff28b9379d595d88b7"}
files = [{"path":"model.safetensors","bytes":9100859544,"sha256":"608f1ff64bc6caa9be836060fc7c78a15c4658c4a07b8d73c78d6f70d1b39c23"}]
paths = {"configuration":"config.json","configuration_source":"configuration_moss_tts.py","modeling_source":"modeling_moss_tts.py","processing_source":"processing_moss_tts.py","gpt2_source":"gpt2_decoder.py","qwen3_source":"qwen3_decoder.py","processor_config":"processor_config.json"}
sizes = {"configuration":10045,"configuration_source":7160,"modeling_source":26379,"processing_source":37496,"gpt2_source":30896,"qwen3_source":25473,"processor_config":210}
for role, path in paths.items(): files.append({"path":path,"bytes":sizes[role],"sha256":digests[role]})
tree = [{"path": f["path"], "type":"file", "size":f["bytes"], "git_blob_sha1":"0"*40, "lfs_sha256":None, "lfs_size":None, "local_sha256":f["sha256"]} for f in files]
custom = {role:{"path":path,"sha256":digests[role]} for role,path in paths.items()}
tensors = [{"name":f"tensor_{i}","dtype":"F32","shape":[1],"numel":1,"data_offsets":[0,4]} for i in range(438)]
s = {"repository":repo,"revision":rev,"files":files,"model":{"path":"model.safetensors","bytes":9100859544,"tensor_count":438,"manifest_sha256":"0"*64,"tensors":tensors,"metadata":{}},"custom_code":custom,"config_sha256":digests["configuration"],"server_tree":{"repository":repo,"revision":rev,"resolved_revision":rev,"files":tree}}
loaded = [{"label":"MOSS Local modeling source","role":"modeling_source","path":"modeling_moss_tts.py","sha256":digests["modeling_source"],"resolved_revision":rev,"authenticated_snapshot_path":"modeling_moss_tts.py"},{"label":"MOSS Local configuration source","role":"configuration_source","path":"configuration_moss_tts.py","sha256":digests["configuration_source"],"resolved_revision":rev,"authenticated_snapshot_path":"configuration_moss_tts.py"}]
old=json.loads(manifest.read_text()); old.update({"snapshot":s,"loaded_custom_code":loaded}); manifest.write_text(json.dumps(old), encoding="utf-8")
PY
  manifest_sha="$(sha256_file "$tmp/manifest")"
  verify_reference_manifest "$tmp/manifest" "$manifest_sha" "$tmp/prompt" "$tmp/rows" "$tmp/codes" "$prompt_sha" "$rows_sha" "$codes_sha"
  sed 's/"columns": 13/"columns": 12/' "$tmp/manifest" > "$tmp/tampered-manifest"
  if verify_reference_manifest "$tmp/tampered-manifest" "$(sha256_file "$tmp/tampered-manifest")" "$tmp/prompt" "$tmp/rows" "$tmp/codes" "$prompt_sha" "$rows_sha" "$codes_sha" >/dev/null 2>&1; then die 'manifest schema tamper was accepted'; fi
  {
    printf '%s\n' "source,v2,$V2_REPOSITORY,$V2_REVISION" 'runtime,torch-2.7.1,transformers-5.5.0' 'environment,cpu,arm64,machine-selftest,logical-1,torch-capability-cpu' 'environment,device,cuda' "source_file,model,transformers_modules/moss/modeling.py,$V2_MODEL_SOURCE_SHA256" "source_file,config,transformers_modules/moss/configuration.py,$V2_CONFIG_SOURCE_SHA256" 'contract,1,12,1024,48000,2,3840' 'codes,0,1,2,3,4,5,6,7,8,9,10,11'
    for label in decoder_0 decoder_1 decoder_2 decoder_3 decoder_4 decoder_5 decoder_6 decoder_7 decoder_8 decoder_9 decoder_10 decoder_11 quantizer audio; do printf 'tensor,%s,[1],0\n' "$label"; done
  } > "$tmp/v2-reference.csv"
  verify_v2_reference "$tmp/v2-reference.csv"
  printf '%s\n' 'extra,unexpected' >> "$tmp/v2-reference.csv"
  if verify_v2_reference "$tmp/v2-reference.csv" >/dev/null 2>&1; then die 'v2 extra row was accepted'; fi
  ln -s "$tmp/prompt" "$tmp/prompt-symlink"
  if require_hash "$tmp/prompt-symlink" "$prompt_sha" >/dev/null 2>&1; then die 'input symlink was accepted'; fi
  printf '{}\n' > "$tmp/local-approval.json"
  printf '{}\n' > "$tmp/v2-approval.json"
  if VOKRA_REMOTE_APPLE_SILICON=1 "$0" \
    --local-gguf "$tmp/missing-local.gguf" --local-gguf-sha256 "$(printf '%064d' 1)" \
    --v2-gguf "$tmp/missing-v2.gguf" --v2-gguf-sha256 "$(printf '%064d' 2)" \
    --prompt "$tmp/missing-prompt" --prompt-sha256 "$(printf '%064d' 3)" \
    --reference-rows "$tmp/missing-rows" --reference-rows-sha256 "$(printf '%064d' 4)" \
    --assistant-codes "$tmp/missing-codes" --assistant-codes-sha256 "$(printf '%064d' 5)" \
    --local-reference-manifest "$tmp/missing-manifest" --local-reference-manifest-sha256 "$(printf '%064d' 6)" \
    --v2-reference "$tmp/missing-v2-reference" --v2-reference-sha256 "$(printf '%064d' 7)" \
    --local-approval-evidence "$tmp/local-approval.json" --v2-approval-evidence "$tmp/v2-approval.json" \
    --evidence-dir "$tmp/evidence" > "$tmp/gate-first.log" 2>&1; then
    die 'production-shaped invocation unexpectedly passed pending approval gates'
  fi
  grep -Eq 'BLOCKED|approval|license' "$tmp/gate-first.log" || die 'gate-first production proof emitted no gate diagnostics'
  [[ ! -e "$tmp/evidence" ]] || die 'blocked Apple invocation created evidence'
  mkdir "$tmp/inputs"
  validate_evidence_target "$tmp/evidence-valid" "$tmp/inputs/input.bin" || die 'disjoint absent evidence target was rejected'
  mkdir "$tmp/evidence-existing"
  if validate_evidence_target "$tmp/evidence-existing" "$tmp/inputs/input.bin" >/dev/null 2>&1; then die 'existing empty evidence directory was accepted'; fi
  ln -s "$tmp/evidence-target" "$tmp/evidence-symlink"
  if validate_evidence_target "$tmp/evidence-symlink" "$tmp/inputs/input.bin" >/dev/null 2>&1; then die 'symlink evidence directory was accepted'; fi
  mkdir -p "$tmp/real-parent/child"; ln -s "$tmp/real-parent" "$tmp/link-parent"
  if validate_evidence_target "$tmp/link-parent/child/new" "$tmp/inputs/input.bin" >/dev/null 2>&1; then die 'descendant under symlink evidence directory was accepted'; fi
  mkdir "$tmp/inputs/input-dir"
  if validate_evidence_target "$tmp/inputs/input-dir/child" "$tmp/inputs/input-dir" >/dev/null 2>&1; then die 'input-overlapping evidence directory was accepted'; fi
  if validate_evidence_target "$VOKRA_ROOT/outside-evidence" "$tmp/inputs/input.bin" >/dev/null 2>&1; then die 'checkout-overlapping evidence directory was accepted'; fi
  grep -Fq -- '--features metal' "$0" || die 'Metal feature is not enabled'; grep -Fq -- 'UV_NO_CACHE=1 uv run --no-cache' "$0" || die 'stdlib validation is not no-cache'; grep -Fq 'pre_sync_gates' "$0" || die 'dual approval gates are missing'; log 'self-test: PASS'
}
main() {
  if [[ "${1:-}" == --self-test ]]; then (($# == 1)) || { die '--self-test does not accept extra arguments'; return 2; }; self_test; return 0; fi
  local local_gguf='' local_gguf_sha='' v2_gguf='' v2_gguf_sha='' prompt='' prompt_sha='' rows='' rows_sha='' codes='' codes_sha='' manifest='' manifest_sha='' v2_reference='' v2_reference_sha='' local_approval='' v2_approval='' evidence=''
  local seen_local_gguf=0 seen_local_gguf_sha=0 seen_v2_gguf=0 seen_v2_gguf_sha=0 seen_prompt=0 seen_prompt_sha=0 seen_rows=0 seen_rows_sha=0 seen_codes=0 seen_codes_sha=0 seen_manifest=0 seen_manifest_sha=0 seen_v2_reference=0 seen_v2_reference_sha=0 seen_local_approval=0 seen_v2_approval=0 seen_evidence=0
  while (($#)); do
    local option="$1"
    [[ "$option" == --* ]] || { usage; die "unexpected trailing argument: $option"; return 2; }
    (($# >= 2)) && [[ -n "${2:-}" && "${2:-}" != -* ]] || { usage; die "$option requires a non-empty value that does not begin with a dash"; return 2; }
    case "$option" in
      --local-gguf) ((seen_local_gguf == 0)) || { die 'duplicate --local-gguf'; return 2; }; seen_local_gguf=1; local_gguf="$2";;
      --local-gguf-sha256) ((seen_local_gguf_sha == 0)) || { die 'duplicate --local-gguf-sha256'; return 2; }; seen_local_gguf_sha=1; local_gguf_sha="$2";;
      --v2-gguf) ((seen_v2_gguf == 0)) || { die 'duplicate --v2-gguf'; return 2; }; seen_v2_gguf=1; v2_gguf="$2";;
      --v2-gguf-sha256) ((seen_v2_gguf_sha == 0)) || { die 'duplicate --v2-gguf-sha256'; return 2; }; seen_v2_gguf_sha=1; v2_gguf_sha="$2";;
      --prompt) ((seen_prompt == 0)) || { die 'duplicate --prompt'; return 2; }; seen_prompt=1; prompt="$2";;
      --prompt-sha256) ((seen_prompt_sha == 0)) || { die 'duplicate --prompt-sha256'; return 2; }; seen_prompt_sha=1; prompt_sha="$2";;
      --reference-rows) ((seen_rows == 0)) || { die 'duplicate --reference-rows'; return 2; }; seen_rows=1; rows="$2";;
      --reference-rows-sha256) ((seen_rows_sha == 0)) || { die 'duplicate --reference-rows-sha256'; return 2; }; seen_rows_sha=1; rows_sha="$2";;
      --assistant-codes) ((seen_codes == 0)) || { die 'duplicate --assistant-codes'; return 2; }; seen_codes=1; codes="$2";;
      --assistant-codes-sha256) ((seen_codes_sha == 0)) || { die 'duplicate --assistant-codes-sha256'; return 2; }; seen_codes_sha=1; codes_sha="$2";;
      --local-reference-manifest) ((seen_manifest == 0)) || { die 'duplicate --local-reference-manifest'; return 2; }; seen_manifest=1; manifest="$2";;
      --local-reference-manifest-sha256) ((seen_manifest_sha == 0)) || { die 'duplicate --local-reference-manifest-sha256'; return 2; }; seen_manifest_sha=1; manifest_sha="$2";;
      --v2-reference) ((seen_v2_reference == 0)) || { die 'duplicate --v2-reference'; return 2; }; seen_v2_reference=1; v2_reference="$2";;
      --v2-reference-sha256) ((seen_v2_reference_sha == 0)) || { die 'duplicate --v2-reference-sha256'; return 2; }; seen_v2_reference_sha=1; v2_reference_sha="$2";;
      --local-approval-evidence) ((seen_local_approval == 0)) || { die 'duplicate --local-approval-evidence'; return 2; }; [[ "$2" != -* ]] || { die 'approval path may not begin with a dash'; return 2; }; seen_local_approval=1; local_approval="$2";;
      --v2-approval-evidence) ((seen_v2_approval == 0)) || { die 'duplicate --v2-approval-evidence'; return 2; }; [[ "$2" != -* ]] || { die 'approval path may not begin with a dash'; return 2; }; seen_v2_approval=1; v2_approval="$2";;
      --evidence-dir) ((seen_evidence == 0)) || { die 'duplicate --evidence-dir'; return 2; }; seen_evidence=1; evidence="$2";;
      *) usage; die "unknown argument: $option"; return 2;;
    esac
    shift 2
  done
  [[ -n "$local_gguf" && -n "$local_gguf_sha" && -n "$v2_gguf" && -n "$v2_gguf_sha" && -n "$prompt" && -n "$prompt_sha" && -n "$rows" && -n "$rows_sha" && -n "$codes" && -n "$codes_sha" && -n "$manifest" && -n "$manifest_sha" && -n "$v2_reference" && -n "$v2_reference_sha" && -n "$local_approval" && -n "$v2_approval" && -n "$evidence" ]] || die 'all input paths, hashes, approvals, and evidence directory are required'
  pre_sync_gates "$local_approval" "$v2_approval"
  require_remote_host
  require_approval_file "$local_approval"; require_approval_file "$v2_approval"; require_hash "$local_gguf" "$local_gguf_sha"; require_hash "$v2_gguf" "$v2_gguf_sha"; require_hash "$prompt" "$prompt_sha"; require_prompt_shape "$prompt"; require_hash "$rows" "$rows_sha"; require_prompt_shape "$rows"; require_hash "$codes" "$codes_sha"; require_codes_shape "$codes"; require_hash "$v2_reference" "$v2_reference_sha"; verify_reference_manifest "$manifest" "$manifest_sha" "$prompt" "$rows" "$codes" "$prompt_sha" "$rows_sha" "$codes_sha"; verify_v2_reference "$v2_reference"
  validate_evidence_target "$evidence" "$local_gguf" "$v2_gguf" "$prompt" "$rows" "$codes" "$manifest" "$v2_reference" "$local_approval" "$v2_approval"
  mkdir -p "$evidence"
  local selector='moss_tts::local_transformer::tests::measure_local_real_cpu_and_optional_metal_against_official'
  local common_env=(
    "VOKRA_MOSS_TTS_LOCAL_GGUF=$local_gguf"
    "VOKRA_MOSS_AUDIO_TOKENIZER_V2_GGUF=$v2_gguf"
    "VOKRA_MOSS_TTS_LOCAL_PROMPT_ROWS=$prompt"
    "VOKRA_MOSS_TTS_LOCAL_REFERENCE_ROWS=$rows"
    "VOKRA_MOSS_TTS_LOCAL_REFERENCE_CODES=$codes"
    "VOKRA_MOSS_TTS_LOCAL_MAX_FRAMES=${VOKRA_MOSS_TTS_LOCAL_MAX_FRAMES:-1}"
  )
  for backend in cpu metal; do
    if [[ "$backend" == metal ]]; then env "${common_env[@]}" VOKRA_MOSS_TTS_LOCAL_RUN_METAL=1 cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release --features metal -p vokra-models --lib "$selector" -- --ignored --exact --nocapture 2>&1 | tee "$evidence/$backend.log"; else env "${common_env[@]}" cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release -p vokra-models --lib "$selector" -- --ignored --exact --nocapture 2>&1 | tee "$evidence/$backend.log"; fi
    require_one_result "$evidence/$backend.log" "$backend"
  done
}
main "$@"
