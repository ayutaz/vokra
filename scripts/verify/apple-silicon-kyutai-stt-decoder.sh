#!/usr/bin/env bash
# Apple verifier for the Kyutai STT dep_q=0 decoder component.
# It consumes only pre-existing authenticated inputs and never publishes.
set -euo pipefail

ROOT="${VOKRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TEST_NAME="parity_kyutai_stt_decoder_real_apple_cpu_metal"
TOKENIZER_NAME="tokenizer.gguf"
MIMI_NAME="mimi-pytorch-e351c8d8@125.safetensors"
MIMI_SHA256="09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50"
usage() { printf '%s\n' "usage: apple-silicon-kyutai-stt-decoder.sh --gguf ABS --gguf-sha256 HEX64 --tokenizer-gguf ABS --tokenizer-gguf-sha256 HEX64 --tokenizer-table-sha256 HEX64 --mimi ABS --mimi-sha256 HEX64 --composite-bind-log ABS --composite-bind-log-sha256 HEX64 --reference ABS --reference-manifest-sha256 HEX64 --reference-packet-sha256 HEX64 --cpu-log ABS --cpu-log-sha256 HEX64 --transfer-manifest ABS --transfer-manifest-sha256 HEX64 --approval-evidence ABS --approval-sha256 HEX64 --evidence ABS --expected-head HEX40 | --self-test"; }
die() { printf '[kyutai-stt-apple] ERROR: %s\n' "$*" >&2; exit 2; }
validate_composite_log() {
  local path="$1"
  [[ "$(grep -Ec '^test [^r].* \.\.\. ok$' "$path")" == 1 ]] || return 1
  [[ "$(grep -Ec '^test parity_kyutai_stt_composite_bind_real \.\.\. ok$' "$path")" == 1 ]] || return 1
  [[ "$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' "$path")" == 1 ]] || return 1
  [[ "$(grep -Ec '^KYUTAI_STT_COMPOSITE_BIND backend=metadata mmap verdict=IDENTITY_SCHEMA_GATE pcm_streaming_parity=BLOCKED$' "$path")" == 1 ]] || return 1
  [[ "$(grep -Ec 'verdict=PASS' "$path")" == 0 ]] || return 1
}

self_test() {
  local self="${BASH_SOURCE[0]}" fail=0 token
  for token in "${TEST_NAME}" 'VOKRA_REMOTE_APPLE_SILICON=1' 'CARGO_NET_OFFLINE=true' 'NO_UPLOAD' 'MEASUREMENT_ONLY_NOT_APPLE_READY' 'APPLE_READY' 'FIXED_ATOL' 'CPU' 'Metal' 'verdict=PASS' '1 passed' 'manifest-sha256' 'reference-packet-sha256' 'tokenizer-gguf' 'tokenizer-table-sha256' 'mimi-pytorch-e351c8d8@125.safetensors' 'composite-bind-log' 'IDENTITY_SCHEMA_GATE' 'pcm_streaming_parity=BLOCKED' 'cpu-log' 'transfer-manifest' '--approval-evidence' '--approval-sha256' '--expected-head' 'vokra-kyutai-stt-decoder-transfer-v2' 'scope=dep_q=0 decoder component numerical parity only' 'status=PASS'; do
    grep -Fq -- "$token" "$self" || { printf '[kyutai-stt-apple] missing self-test token: %s\n' "$token" >&2; fail=1; }
  done
  grep -Eq '(^|[;&|[:space:]])(curl|wget|git[[:space:]]+(clone|fetch|push)|hf_hub_download|upload_file|--push)([[:space:]]|$)' "$self" && fail=1 || true
  local guard_line cargo_line
  guard_line="$(grep -n 'not Apple-ready; reviewed fixed bound is required' "$self" | head -n 1 | cut -d: -f1)"
  cargo_line="$(grep -n 'cargo test --locked --offline' "$self" | head -n 1 | cut -d: -f1)"
  [[ "$guard_line" =~ ^[0-9]+$ && "$cargo_line" =~ ^[0-9]+$ && "$guard_line" -lt "$cargo_line" ]] || { printf '%s\n' 'measurement-only transfer guard must precede Cargo' >&2; fail=1; }
  if "$self" --expected-head "${TEST_NAME//[^a-z]/a}" --expected-head "${TEST_NAME//[^a-z]/b}" >/dev/null 2>&1; then
    printf '%s\n' 'duplicate --expected-head was accepted' >&2
    fail=1
  fi
  local valid_hash='0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef'
  if "$self" --gguf /private/tmp/decoder.gguf --gguf-sha256 "$valid_hash" \
      --tokenizer-gguf /private/tmp/tokenizer.gguf --tokenizer-gguf-sha256 "$valid_hash" \
      --tokenizer-table-sha256 "A${valid_hash:1}" --mimi "/private/tmp/${MIMI_NAME}" \
      --mimi-sha256 "$valid_hash" --composite-bind-log /private/tmp/composite-bind.log \
      --composite-bind-log-sha256 "$valid_hash" --reference /private/tmp/reference \
      --reference-manifest-sha256 "$valid_hash" --reference-packet-sha256 "$valid_hash" \
      --cpu-log /private/tmp/validation.log --cpu-log-sha256 "$valid_hash" \
      --transfer-manifest /private/tmp/transfer-manifest.json --transfer-manifest-sha256 "$valid_hash" \
      --approval-evidence /private/tmp/approval.json --approval-sha256 "$valid_hash" \
      --evidence /private/tmp/apple-evidence --expected-head '0123456789abcdef0123456789abcdef01234567' >/dev/null 2>&1; then
    printf '%s\n' 'uppercase tokenizer table digest was accepted' >&2
    fail=1
  fi
  local composite_synthetic
  composite_synthetic="$(mktemp)"
  printf '%s\n' 'test parity_kyutai_stt_composite_bind_real ... ok' 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' 'KYUTAI_STT_COMPOSITE_BIND backend=metadata mmap verdict=IDENTITY_SCHEMA_GATE pcm_streaming_parity=BLOCKED' > "$composite_synthetic"
  validate_composite_log "$composite_synthetic" || fail=1
  printf '%s\n' 'test unexpected_extra ... ok' >> "$composite_synthetic"
  validate_composite_log "$composite_synthetic" && fail=1 || true
  printf '%s\n' 'test parity_kyutai_stt_composite_bind_real ... ok' 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' 'KYUTAI_STT_COMPOSITE_BIND backend=metadata mmap verdict=IDENTITY_SCHEMA_GATE pcm_streaming_parity=BLOCKED' 'KYUTAI_STT_COMPOSITE_BIND backend=metadata mmap verdict=IDENTITY_SCHEMA_GATE pcm_streaming_parity=BLOCKED' > "$composite_synthetic"
  validate_composite_log "$composite_synthetic" && fail=1 || true
  rm -f "$composite_synthetic"
  (( fail == 0 )) || return 1
  printf '%s\n' 'kyutai STT Apple verifier self-test PASS'
}

[[ "${1:-}" == --self-test ]] && { [[ $# == 1 ]] || die '--self-test accepts no arguments'; self_test; exit $?; }
gguf=""; gguf_sha=""; tokenizer=""; tokenizer_sha=""; tokenizer_table_sha=""; mimi=""; mimi_sha=""; composite_log=""; composite_log_sha=""; reference=""; reference_sha=""; reference_packet_sha=""; cpu_log=""; cpu_log_sha=""; transfer_manifest=""; transfer_manifest_sha=""; approval=""; approval_sha=""; evidence=""; expected_head=""
gguf_seen=0; gguf_sha_seen=0; tokenizer_seen=0; tokenizer_sha_seen=0; tokenizer_table_sha_seen=0; mimi_seen=0; mimi_sha_seen=0; composite_log_seen=0; composite_log_sha_seen=0; reference_seen=0; reference_sha_seen=0; reference_packet_sha_seen=0; cpu_log_seen=0; cpu_log_sha_seen=0; transfer_manifest_seen=0; transfer_manifest_sha_seen=0; approval_seen=0; approval_sha_seen=0; evidence_seen=0; expected_head_seen=0
while (($#)); do
  case "$1" in
    --gguf) (($# >= 2)) || die '--gguf requires ABS'; (( gguf_seen == 0 )) || die 'duplicate --gguf'; gguf="$2"; gguf_seen=1; shift 2;;
    --gguf-sha256) (($# >= 2)) || die '--gguf-sha256 requires HEX64'; (( gguf_sha_seen == 0 )) || die 'duplicate --gguf-sha256'; gguf_sha="$2"; gguf_sha_seen=1; shift 2;;
    --tokenizer-gguf) (($# >= 2)) || die '--tokenizer-gguf requires ABS'; (( tokenizer_seen == 0 )) || die 'duplicate --tokenizer-gguf'; tokenizer="$2"; tokenizer_seen=1; shift 2;;
    --tokenizer-gguf-sha256) (($# >= 2)) || die '--tokenizer-gguf-sha256 requires HEX64'; (( tokenizer_sha_seen == 0 )) || die 'duplicate --tokenizer-gguf-sha256'; tokenizer_sha="$2"; tokenizer_sha_seen=1; shift 2;;
    --tokenizer-table-sha256) (($# >= 2)) || die '--tokenizer-table-sha256 requires HEX64'; (( tokenizer_table_sha_seen == 0 )) || die 'duplicate --tokenizer-table-sha256'; tokenizer_table_sha="$2"; tokenizer_table_sha_seen=1; shift 2;;
    --mimi) (($# >= 2)) || die '--mimi requires ABS'; (( mimi_seen == 0 )) || die 'duplicate --mimi'; mimi="$2"; mimi_seen=1; shift 2;;
    --mimi-sha256) (($# >= 2)) || die '--mimi-sha256 requires HEX64'; (( mimi_sha_seen == 0 )) || die 'duplicate --mimi-sha256'; mimi_sha="$2"; mimi_sha_seen=1; shift 2;;
    --composite-bind-log) (($# >= 2)) || die '--composite-bind-log requires ABS'; (( composite_log_seen == 0 )) || die 'duplicate --composite-bind-log'; composite_log="$2"; composite_log_seen=1; shift 2;;
    --composite-bind-log-sha256) (($# >= 2)) || die '--composite-bind-log-sha256 requires HEX64'; (( composite_log_sha_seen == 0 )) || die 'duplicate --composite-bind-log-sha256'; composite_log_sha="$2"; composite_log_sha_seen=1; shift 2;;
    --reference) (($# >= 2)) || die '--reference requires ABS'; (( reference_seen == 0 )) || die 'duplicate --reference'; reference="$2"; reference_seen=1; shift 2;;
    --reference-manifest-sha256) (($# >= 2)) || die '--reference-manifest-sha256 requires HEX64'; (( reference_sha_seen == 0 )) || die 'duplicate --reference-manifest-sha256'; reference_sha="$2"; reference_sha_seen=1; shift 2;;
    --reference-packet-sha256) (($# >= 2)) || die '--reference-packet-sha256 requires HEX64'; (( reference_packet_sha_seen == 0 )) || die 'duplicate --reference-packet-sha256'; reference_packet_sha="$2"; reference_packet_sha_seen=1; shift 2;;
    --cpu-log) (($# >= 2)) || die '--cpu-log requires ABS'; (( cpu_log_seen == 0 )) || die 'duplicate --cpu-log'; cpu_log="$2"; cpu_log_seen=1; shift 2;;
    --cpu-log-sha256) (($# >= 2)) || die '--cpu-log-sha256 requires HEX64'; (( cpu_log_sha_seen == 0 )) || die 'duplicate --cpu-log-sha256'; cpu_log_sha="$2"; cpu_log_sha_seen=1; shift 2;;
    --transfer-manifest) (($# >= 2)) || die '--transfer-manifest requires ABS'; (( transfer_manifest_seen == 0 )) || die 'duplicate --transfer-manifest'; transfer_manifest="$2"; transfer_manifest_seen=1; shift 2;;
    --transfer-manifest-sha256) (($# >= 2)) || die '--transfer-manifest-sha256 requires HEX64'; (( transfer_manifest_sha_seen == 0 )) || die 'duplicate --transfer-manifest-sha256'; transfer_manifest_sha="$2"; transfer_manifest_sha_seen=1; shift 2;;
    --approval-evidence) (($# >= 2)) || die '--approval-evidence requires ABS'; (( approval_seen == 0 )) || die 'duplicate --approval-evidence'; approval="$2"; approval_seen=1; shift 2;;
    --approval-sha256) (($# >= 2)) || die '--approval-sha256 requires HEX64'; (( approval_sha_seen == 0 )) || die 'duplicate --approval-sha256'; approval_sha="$2"; approval_sha_seen=1; shift 2;;
    --evidence) (($# >= 2)) || die '--evidence requires ABS'; (( evidence_seen == 0 )) || die 'duplicate --evidence'; evidence="$2"; evidence_seen=1; shift 2;;
    --expected-head) (($# >= 2)) || die '--expected-head requires HEX40'; (( expected_head_seen == 0 )) || die 'duplicate --expected-head'; expected_head="$2"; expected_head_seen=1; shift 2;;
    -h|--help) usage; exit 0;;
    *) usage; die "unknown argument: $1";;
  esac
done
[[ "$expected_head" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head must be lowercase HEX40'
[[ "$gguf_sha" =~ ^[0-9a-f]{64}$ && "$tokenizer_sha" =~ ^[0-9a-f]{64}$ && "$tokenizer_table_sha" =~ ^[0-9a-f]{64}$ && "$mimi_sha" =~ ^[0-9a-f]{64}$ && "$composite_log_sha" =~ ^[0-9a-f]{64}$ && "$reference_sha" =~ ^[0-9a-f]{64}$ && "$reference_packet_sha" =~ ^[0-9a-f]{64}$ && "$cpu_log_sha" =~ ^[0-9a-f]{64}$ && "$transfer_manifest_sha" =~ ^[0-9a-f]{64}$ && "$approval_sha" =~ ^[0-9a-f]{64}$ ]] || die 'all artifact and approval digests must be lowercase HEX64'
[[ "$gguf_seen$gguf_sha_seen$tokenizer_seen$tokenizer_sha_seen$tokenizer_table_sha_seen$mimi_seen$mimi_sha_seen$composite_log_seen$composite_log_sha_seen$reference_seen$reference_sha_seen$reference_packet_sha_seen$cpu_log_seen$cpu_log_sha_seen$transfer_manifest_seen$transfer_manifest_sha_seen$approval_seen$approval_sha_seen$evidence_seen$expected_head_seen" == 11111111111111111111 ]] || die 'all decoder/tokenizer/Mimi/composite/reference/hash/CPU-log/transfer/approval/evidence/head arguments are required'
UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$ROOT/tools/parity/kyutai_stt_decoder_dump_reference.py" validate-approval --expected-head "$expected_head" --approval-evidence "$approval" --approval-sha256 "$approval_sha" >/dev/null || die 'external decoder approval is invalid'

[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || die 'requires Darwin arm64 Apple Silicon'
[[ "${VOKRA_REMOTE_APPLE_SILICON:-}" == 1 ]] || die 'VOKRA_REMOTE_APPLE_SILICON=1 is required'
[[ -f "$ROOT/Cargo.toml" && -d "$ROOT/.git" ]] || die 'not a Vokra checkout'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
actual_head="$(git -C "$ROOT" rev-parse HEAD)"
[[ "$actual_head" == "$expected_head" ]] || die "checkout HEAD $actual_head does not match --expected-head $expected_head"
[[ "$gguf" == /* && "$tokenizer" == /* && "$mimi" == /* && "$composite_log" == /* && "$reference" == /* && "$cpu_log" == /* && "$transfer_manifest" == /* && "$approval" == /* && "$evidence" == /* ]] || die 'all paths must be absolute'
canonical_path() {
  local path="$1" current=/ rest component
  [[ "$path" == /* && "$path" != *'//' && "$path" != *'/./'* && "$path" != *'/../'* && "$path" != */. && "$path" != */.. ]] || return 1
  rest="${path#/}"
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; [[ "$rest" == "$component" ]] && rest="" || rest="${rest#*/}"
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || return 1
    current="${current%/}/$component"; [[ ! -L "$current" ]] || return 1
  done
}
for path in "$gguf" "$tokenizer" "$mimi" "$composite_log" "$reference" "$cpu_log" "$transfer_manifest" "$approval" "$evidence"; do canonical_path "$path" || die 'absolute input path has unsafe dot/empty/symlink ancestry'; done
[[ "$(basename "$tokenizer")" == "$TOKENIZER_NAME" ]] || die "tokenizer filename must be $TOKENIZER_NAME"
[[ "$(basename "$mimi")" == "$MIMI_NAME" ]] || die "Mimi filename must be $MIMI_NAME"
[[ -f "$gguf" && ! -L "$gguf" ]] || die 'decoder GGUF must be a regular file'
[[ -f "$tokenizer" && ! -L "$tokenizer" ]] || die 'tokenizer GGUF must be a regular file'
[[ -f "$mimi" && ! -L "$mimi" ]] || die 'Mimi input must be a regular file'
[[ -d "$reference" && ! -L "$reference" ]] || die 'reference must be a regular directory'
[[ -f "$reference/manifest.json" && ! -L "$reference/manifest.json" ]] || die 'reference manifest is missing'
[[ ! -e "$evidence" ]] || die 'evidence must be absent (no-clobber)'
[[ -f "$transfer_manifest" && ! -L "$transfer_manifest" && -f "$approval" && ! -L "$approval" ]] || die 'transfer manifest and approval must be regular files'
[[ -f "$cpu_log" && ! -L "$cpu_log" ]] || die 'CPU evidence log must be a regular file'
[[ -f "$composite_log" && ! -L "$composite_log" ]] || die 'composite bind log must be a regular file'
[[ "$mimi_sha" == "$MIMI_SHA256" ]] || die 'Mimi digest is not the pinned exact identity'
[[ "$(shasum -a 256 "$cpu_log" | awk '{print $1}')" == "$cpu_log_sha" ]] || die 'CPU evidence log digest mismatch'
[[ "$(grep -Ec '^test parity_kyutai_stt_decoder_real_cpu \.\.\. ok$' "$cpu_log")" == 1 ]] || die 'CPU evidence test singleton missing'
[[ "$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' "$cpu_log")" == 1 ]] || die 'CPU evidence result singleton missing'
[[ "$(grep -Ec '^KYUTAI_STT_DECODER_MEASUREMENT backend=Cpu .* verdict=MEASUREMENT_ONLY$' "$cpu_log")" == 1 ]] || die 'CPU measurement sentinel missing'
validate_composite_log "$composite_log" || die 'composite bind log singleton/sentinel validation failed'
actual_gguf_sha="$(shasum -a 256 "$gguf" | awk '{print $1}')"
actual_tokenizer_sha="$(shasum -a 256 "$tokenizer" | awk '{print $1}')"
actual_mimi_sha="$(shasum -a 256 "$mimi" | awk '{print $1}')"
actual_composite_log_sha="$(shasum -a 256 "$composite_log" | awk '{print $1}')"
actual_reference_sha="$(shasum -a 256 "$reference/manifest.json" | awk '{print $1}')"
[[ "$actual_gguf_sha" == "$gguf_sha" ]] || die 'decoder GGUF digest mismatch'
[[ "$actual_tokenizer_sha" == "$tokenizer_sha" ]] || die 'tokenizer GGUF digest mismatch'
[[ "$actual_mimi_sha" == "$mimi_sha" ]] || die 'Mimi digest mismatch'
[[ "$actual_composite_log_sha" == "$composite_log_sha" ]] || die 'composite bind log digest mismatch'
[[ "$actual_reference_sha" == "$reference_sha" ]] || die 'reference manifest digest mismatch'
actual_packet_sha="$(UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$reference" <<'PY'
import hashlib, pathlib, sys
root=pathlib.Path(sys.argv[1]); expected={"input.json","hidden.f32","logits.f32","manifest.json"}; rows=[]
for path in root.iterdir():
    if path.is_symlink() or not path.is_file() or path.name not in expected: raise SystemExit("reference packet has an unexpected or unsafe entry")
    rows.append((path.name, hashlib.sha256(path.read_bytes()).hexdigest()))
if {name for name,_ in rows} != expected: raise SystemExit("reference packet closure is incomplete")
print(hashlib.sha256(b"".join(name.encode()+b"\0"+digest.encode()+b"\n" for name,digest in sorted(rows))).hexdigest())
PY
)"
[[ "$actual_packet_sha" == "$reference_packet_sha" ]] || die 'reference packet digest mismatch'
UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$reference/manifest.json" "$expected_head" "$approval_sha" "$tokenizer_table_sha" <<'PY'
import json, pathlib, sys
path, head, approval, tokenizer_table = sys.argv[1:]
def unique(pairs):
    out={}
    for key,value in pairs:
        if key in out: raise SystemExit("duplicate reference manifest key")
        out[key]=value
    return out
data=json.loads(pathlib.Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)
if data.get("expected_head") != head or data.get("approval_sha256") != approval or data.get("approval_decision") != "APPROVED_FOR_NO_UPLOAD_PARITY" or data.get("approval_scope") != "KYUTAI_STT_DECODER_PARITY": raise SystemExit("reference approval binding drifted")
structure = data.get("model", {}).get("tokenizer_structure")
if not isinstance(structure, dict) or structure.get("table_sha256") != tokenizer_table: raise SystemExit("reference tokenizer table digest binding drifted")
PY
[[ "$(shasum -a 256 "$transfer_manifest" | awk '{print $1}')" == "$transfer_manifest_sha" ]] || die 'transfer manifest digest mismatch'
UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$transfer_manifest" "$expected_head" "$approval_sha" "$gguf_sha" "$tokenizer_sha" "$tokenizer_table_sha" "$mimi_sha" "$composite_log_sha" "$reference_sha" "$reference_packet_sha" "$cpu_log_sha" <<'PY'
import json, pathlib, sys
path, head, approval, gguf, tokenizer, tokenizer_table, mimi, composite_log, ref, packet, cpu_log=sys.argv[1:]
def unique(pairs):
    out={}
    for key,value in pairs:
        if key in out: raise SystemExit("duplicate transfer manifest key")
        out[key]=value
    return out
data=json.loads(pathlib.Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)
if data.get("status") != "APPLE_READY": raise SystemExit("Kyutai CPU measurement is not Apple-ready; reviewed fixed bound is required")
expected={"format":"vokra-kyutai-stt-decoder-transfer-v2","status":"APPLE_READY","expected_head":head,"approval_sha256":approval,"gguf_name":"decoder.gguf","gguf_sha256":gguf,"tokenizer_gguf_name":"tokenizer.gguf","tokenizer_sha256":tokenizer,"tokenizer_table_sha256":tokenizer_table,"mimi_name":"mimi-pytorch-e351c8d8@125.safetensors","mimi_sha256":mimi,"reference_manifest_sha256":ref,"reference_packet_sha256":packet,"cpu_log_name":"validation.log","cpu_log_sha256":cpu_log,"cpu_test_name":"parity_kyutai_stt_decoder_real_cpu","composite_bind_log_name":"composite-bind.log","composite_bind_log_sha256":composite_log,"composite_bind_test_name":"parity_kyutai_stt_composite_bind_real","composite_bind_verdict":"IDENTITY_SCHEMA_GATE_ONLY","reference_files":["hidden.f32","input.json","logits.f32","manifest.json"],"no_upload":True}
if data != expected: raise SystemExit("portable transfer manifest identity/closure mismatch")
PY
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$expected_head" ]] || die 'checkout HEAD changed before hardware execution'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout became dirty before hardware execution'
gguf_real="$(cd "$(dirname "$gguf")" && pwd -P)/$(basename "$gguf")"
tokenizer_real="$(cd "$(dirname "$tokenizer")" && pwd -P)/$(basename "$tokenizer")"
mimi_real="$(cd "$(dirname "$mimi")" && pwd -P)/$(basename "$mimi")"
composite_log_real="$(cd "$(dirname "$composite_log")" && pwd -P)/$(basename "$composite_log")"
reference_real="$(cd "$reference" && pwd -P)"
evidence_parent="$(cd "$(dirname "$evidence")" && pwd -P)"
evidence_real="$evidence_parent/$(basename "$evidence")"
cpu_log_real="$(cd "$(dirname "$cpu_log")" && pwd -P)/$(basename "$cpu_log")"
transfer_real="$(cd "$(dirname "$transfer_manifest")" && pwd -P)/$(basename "$transfer_manifest")"
approval_real="$(cd "$(dirname "$approval")" && pwd -P)/$(basename "$approval")"
root_real="$(cd "$ROOT" && pwd -P)"
logical_paths=("$gguf_real" "$tokenizer_real" "$mimi_real" "$composite_log_real" "$reference_real" "$cpu_log_real" "$transfer_real" "$approval_real" "$evidence_real" "$root_real")
logical_names=('decoder GGUF' 'tokenizer GGUF' 'Mimi' 'composite log' 'reference' 'CPU log' 'transfer manifest' 'approval' 'evidence' 'checkout')
path_contains() {
  local child="$1" parent="$2" parent_prefix="$2"
  if [[ "$parent_prefix" == / ]]; then
    [[ "$child" == /* ]]
  else
    parent_prefix="${parent_prefix%/}"
    [[ "$child" == "$parent" || "$child" == "$parent_prefix/"* ]]
  fi
}
for ((i = 0; i < ${#logical_paths[@]}; i++)); do
  for ((j = i + 1; j < ${#logical_paths[@]}; j++)); do
    if [[ "${logical_paths[$i]}" == "${logical_paths[$j]}" ]] || \
      path_contains "${logical_paths[$i]}" "${logical_paths[$j]}" || \
      path_contains "${logical_paths[$j]}" "${logical_paths[$i]}"; then
      die "${logical_names[$i]} overlaps ${logical_names[$j]}"
    fi
  done
done
mkdir -m 700 "$evidence"
log="$evidence/run.log"
[[ ! -e "$log" ]] || die 'log already exists'
set -o noclobber
{
  echo 'phase=RUNNING'
  echo 'scope=dep_q=0 decoder component numerical parity only; tokenizer/Mimi/composite identity-schema gate; no PCM/streaming parity claim'
  echo "test=$TEST_NAME"
  echo "gguf_sha256=$gguf_sha"
  echo "tokenizer_gguf_sha256=$tokenizer_sha"
  echo "tokenizer_table_sha256=$tokenizer_table_sha"
  echo "mimi_sha256=$mimi_sha"
  echo "composite_bind_log_sha256=$composite_log_sha"
  echo "reference_manifest_sha256=$reference_sha"
  echo "reference_packet_sha256=$reference_packet_sha"
  echo "transfer_manifest_sha256=$transfer_manifest_sha"
  echo "approval_sha256=$approval_sha"
  echo "expected_head=$expected_head"
  echo "actual_head=$actual_head"
  echo 'backend_contract=decoder CPU/reference, decoder Metal/reference, decoder Metal/CPU exact argmax; composite identity/schema gate only'
  echo 'publication=NO_UPLOAD'
  echo 'cargo_contract=CARGO_NET_OFFLINE=true cargo test --features metal --exact'
} > "$log"
set +o noclobber
set +e
VOKRA_REMOTE_APPLE_SILICON=1 CARGO_NET_OFFLINE=true \
  VOKRA_KYUTAI_STT_DECODER_GGUF="$gguf" \
  VOKRA_KYUTAI_STT_DECODER_GGUF_SHA256="$gguf_sha" \
  VOKRA_KYUTAI_STT_DECODER_REFERENCE="$reference" \
  VOKRA_KYUTAI_STT_DECODER_REFERENCE_MANIFEST_SHA256="$reference_sha" \
  cargo test --locked --offline --features metal --manifest-path "$ROOT/Cargo.toml" -p vokra-models \
    --test parity_kyutai_stt_decoder_real -- --ignored --exact "$TEST_NAME" --nocapture \
    >> "$log" 2>&1
test_status=$?
set -e
(( test_status == 0 )) || die 'Apple parity test failed; evidence log preserved'
[[ "$(grep -Ec "^test ${TEST_NAME} \.\.\. ok$" "$log")" == 1 ]] || die 'target test singleton/result missing'
[[ "$(grep -Ec '^test [^r].* \.\.\. ok$' "$log")" == 1 ]] || die 'extra or missing test result line'
[[ "$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' "$log")" == 1 ]] || die 'Cargo result is not exactly 1 passed / 0 failed / 0 ignored / 0 measured / 0 filtered out'
[[ "$(grep -Ec '^KYUTAI_STT_DECODER_APPLE .* verdict=PASS$' "$log")" == 1 ]] || die 'Apple PASS sentinel missing or duplicated'
[[ "$(grep -Ec 'verdict=PASS' "$log")" == 1 ]] || die 'verdict is not an exact singleton'
[[ "$(grep -Ec '0 failed' "$log")" == 1 ]] || die 'failed count is not an exact singleton'
printf '%s\n' 'status=PASS' >> "$log"
[[ "$(grep -Ec '^status=PASS$' "$log")" == 1 ]] || die 'final PASS status missing or duplicated'
log_sha="$(shasum -a 256 "$log" | awk '{print $1}')"
printf '[kyutai-stt-apple] PASS: %s (log_sha256=%s)\n' "$log" "$log_sha" >&2
