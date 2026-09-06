#!/usr/bin/env bash
# Apple verifier for the Kyutai STT dep_q=0 decoder component.
# It consumes only pre-existing authenticated inputs and never publishes.
set -euo pipefail

ROOT="${VOKRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TEST_NAME="parity_kyutai_stt_decoder_real_apple_cpu_metal"
usage() { printf '%s\n' "usage: apple-silicon-kyutai-stt-decoder.sh --gguf ABS --gguf-sha256 HEX64 --reference ABS --reference-manifest-sha256 HEX64 --reference-packet-sha256 HEX64 --cpu-log ABS --cpu-log-sha256 HEX64 --transfer-manifest ABS --transfer-manifest-sha256 HEX64 --approval-evidence ABS --approval-sha256 HEX64 --evidence ABS --expected-head HEX40 | --self-test"; }
die() { printf '[kyutai-stt-apple] ERROR: %s\n' "$*" >&2; exit 2; }

self_test() {
  local self="${BASH_SOURCE[0]}" fail=0 token
  for token in "${TEST_NAME}" 'VOKRA_REMOTE_APPLE_SILICON=1' 'CARGO_NET_OFFLINE=true' 'NO_UPLOAD' 'MEASUREMENT_ONLY_NOT_APPLE_READY' 'APPLE_READY' 'FIXED_ATOL' 'CPU' 'Metal' 'verdict=PASS' '1 passed' 'manifest-sha256' 'reference-packet-sha256' 'cpu-log' 'transfer-manifest' '--approval-evidence' '--approval-sha256' '--expected-head' 'status=PASS'; do
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
  (( fail == 0 )) || return 1
  printf '%s\n' 'kyutai STT Apple verifier self-test PASS'
}

[[ "${1:-}" == --self-test ]] && { [[ $# == 1 ]] || die '--self-test accepts no arguments'; self_test; exit $?; }
gguf=""; gguf_sha=""; reference=""; reference_sha=""; reference_packet_sha=""; cpu_log=""; cpu_log_sha=""; transfer_manifest=""; transfer_manifest_sha=""; approval=""; approval_sha=""; evidence=""; expected_head=""
gguf_seen=0; gguf_sha_seen=0; reference_seen=0; reference_sha_seen=0; reference_packet_sha_seen=0; cpu_log_seen=0; cpu_log_sha_seen=0; transfer_manifest_seen=0; transfer_manifest_sha_seen=0; approval_seen=0; approval_sha_seen=0; evidence_seen=0; expected_head_seen=0
while (($#)); do
  case "$1" in
    --gguf) (($# >= 2)) || die '--gguf requires ABS'; (( gguf_seen == 0 )) || die 'duplicate --gguf'; gguf="$2"; gguf_seen=1; shift 2;;
    --gguf-sha256) (($# >= 2)) || die '--gguf-sha256 requires HEX64'; (( gguf_sha_seen == 0 )) || die 'duplicate --gguf-sha256'; gguf_sha="$2"; gguf_sha_seen=1; shift 2;;
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
[[ "$gguf_sha" =~ ^[0-9a-f]{64}$ && "$reference_sha" =~ ^[0-9a-f]{64}$ && "$reference_packet_sha" =~ ^[0-9a-f]{64}$ && "$cpu_log_sha" =~ ^[0-9a-f]{64}$ && "$transfer_manifest_sha" =~ ^[0-9a-f]{64}$ && "$approval_sha" =~ ^[0-9a-f]{64}$ ]] || die 'all artifact and approval digests must be lowercase HEX64'
[[ "$gguf_seen$gguf_sha_seen$reference_seen$reference_sha_seen$reference_packet_sha_seen$cpu_log_seen$cpu_log_sha_seen$transfer_manifest_seen$transfer_manifest_sha_seen$approval_seen$approval_sha_seen$evidence_seen$expected_head_seen" == 1111111111111 ]] || die 'all GGUF/reference/hash/CPU-log/transfer/approval/evidence/head arguments are required'
UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$ROOT/tools/parity/kyutai_stt_decoder_dump_reference.py" validate-approval --expected-head "$expected_head" --approval-evidence "$approval" --approval-sha256 "$approval_sha" >/dev/null || die 'external decoder approval is invalid'

[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || die 'requires Darwin arm64 Apple Silicon'
[[ "${VOKRA_REMOTE_APPLE_SILICON:-}" == 1 ]] || die 'VOKRA_REMOTE_APPLE_SILICON=1 is required'
[[ -f "$ROOT/Cargo.toml" && -d "$ROOT/.git" ]] || die 'not a Vokra checkout'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
actual_head="$(git -C "$ROOT" rev-parse HEAD)"
[[ "$actual_head" == "$expected_head" ]] || die "checkout HEAD $actual_head does not match --expected-head $expected_head"
[[ "$gguf" == /* && "$reference" == /* && "$cpu_log" == /* && "$transfer_manifest" == /* && "$approval" == /* && "$evidence" == /* ]] || die 'all paths must be absolute'
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
for path in "$gguf" "$reference" "$cpu_log" "$transfer_manifest" "$approval" "$evidence"; do canonical_path "$path" || die 'absolute input path has unsafe dot/empty/symlink ancestry'; done
[[ -f "$gguf" && ! -L "$gguf" ]] || die 'GGUF must be a regular file'
[[ -d "$reference" && ! -L "$reference" ]] || die 'reference must be a regular directory'
[[ -f "$reference/manifest.json" && ! -L "$reference/manifest.json" ]] || die 'reference manifest is missing'
[[ ! -e "$evidence" ]] || die 'evidence must be absent (no-clobber)'
[[ -f "$transfer_manifest" && ! -L "$transfer_manifest" && -f "$approval" && ! -L "$approval" ]] || die 'transfer manifest and approval must be regular files'
[[ -f "$cpu_log" && ! -L "$cpu_log" ]] || die 'CPU evidence log must be a regular file'
[[ "$(shasum -a 256 "$cpu_log" | awk '{print $1}')" == "$cpu_log_sha" ]] || die 'CPU evidence log digest mismatch'
[[ "$(grep -Ec '^test parity_kyutai_stt_decoder_real_cpu \.\.\. ok$' "$cpu_log")" == 1 ]] || die 'CPU evidence test singleton missing'
[[ "$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' "$cpu_log")" == 1 ]] || die 'CPU evidence result singleton missing'
[[ "$(grep -Ec '^KYUTAI_STT_DECODER_MEASUREMENT backend=Cpu .* verdict=MEASUREMENT_ONLY$' "$cpu_log")" == 1 ]] || die 'CPU measurement sentinel missing'
actual_gguf_sha="$(shasum -a 256 "$gguf" | awk '{print $1}')"
actual_reference_sha="$(shasum -a 256 "$reference/manifest.json" | awk '{print $1}')"
[[ "$actual_gguf_sha" == "$gguf_sha" ]] || die 'GGUF digest mismatch'
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
UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$reference/manifest.json" "$expected_head" "$approval_sha" <<'PY'
import json, pathlib, sys
path, head, approval = sys.argv[1:]
def unique(pairs):
    out={}
    for key,value in pairs:
        if key in out: raise SystemExit("duplicate reference manifest key")
        out[key]=value
    return out
data=json.loads(pathlib.Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)
if data.get("expected_head") != head or data.get("approval_sha256") != approval or data.get("approval_decision") != "APPROVED_FOR_NO_UPLOAD_PARITY" or data.get("approval_scope") != "KYUTAI_STT_DECODER_PARITY": raise SystemExit("reference approval binding drifted")
PY
[[ "$(shasum -a 256 "$transfer_manifest" | awk '{print $1}')" == "$transfer_manifest_sha" ]] || die 'transfer manifest digest mismatch'
UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$transfer_manifest" "$expected_head" "$approval_sha" "$gguf_sha" "$reference_sha" "$reference_packet_sha" "$cpu_log_sha" <<'PY'
import json, pathlib, sys
path, head, approval, gguf, ref, packet, cpu_log=sys.argv[1:]
def unique(pairs):
    out={}
    for key,value in pairs:
        if key in out: raise SystemExit("duplicate transfer manifest key")
        out[key]=value
    return out
data=json.loads(pathlib.Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)
if data.get("status") != "APPLE_READY": raise SystemExit("Kyutai CPU measurement is not Apple-ready; reviewed fixed bound is required")
expected={"format":"vokra-kyutai-stt-decoder-transfer-v1","status":"APPLE_READY","expected_head":head,"approval_sha256":approval,"gguf_name":"decoder.gguf","gguf_sha256":gguf,"reference_manifest_sha256":ref,"reference_packet_sha256":packet,"cpu_log_name":"validation.log","cpu_log_sha256":cpu_log,"cpu_test_name":"parity_kyutai_stt_decoder_real_cpu","reference_files":["hidden.f32","input.json","logits.f32","manifest.json"],"no_upload":True}
if data != expected: raise SystemExit("portable transfer manifest identity/closure mismatch")
PY
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$expected_head" ]] || die 'checkout HEAD changed before hardware execution'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout became dirty before hardware execution'
gguf_real="$(cd "$(dirname "$gguf")" && pwd -P)/$(basename "$gguf")"
reference_real="$(cd "$reference" && pwd -P)"
evidence_parent="$(cd "$(dirname "$evidence")" && pwd -P)"
evidence_real="$evidence_parent/$(basename "$evidence")"
cpu_log_real="$(cd "$(dirname "$cpu_log")" && pwd -P)/$(basename "$cpu_log")"
transfer_real="$(cd "$(dirname "$transfer_manifest")" && pwd -P)/$(basename "$transfer_manifest")"
approval_real="$(cd "$(dirname "$approval")" && pwd -P)/$(basename "$approval")"
case "$gguf_real/" in "$reference_real/"*|"$evidence_real/"*|"$cpu_log_real/"*|"$transfer_real/"*|"$approval_real/"*|"$ROOT/"*) die 'GGUF overlaps another path';; esac
case "$reference_real/" in "$gguf_real/"*|"$evidence_real/"*|"$cpu_log_real/"*|"$transfer_real/"*|"$approval_real/"*|"$ROOT/"*) die 'reference overlaps another path';; esac
case "$ROOT/" in "$gguf_real/"*|"$reference_real/"*|"$cpu_log_real/"*|"$transfer_real/"*|"$approval_real/"*) die 'checkout overlaps an input';; esac
mkdir -m 700 "$evidence"
log="$evidence/run.log"
[[ ! -e "$log" ]] || die 'log already exists'
set -o noclobber
{
  echo 'phase=RUNNING'
  echo 'scope=dep_q=0 decoder component only'
  echo "test=$TEST_NAME"
  echo "gguf_sha256=$gguf_sha"
  echo "reference_manifest_sha256=$reference_sha"
  echo "reference_packet_sha256=$reference_packet_sha"
  echo "transfer_manifest_sha256=$transfer_manifest_sha"
  echo "approval_sha256=$approval_sha"
  echo "expected_head=$expected_head"
  echo "actual_head=$actual_head"
  echo 'backend_contract=CPU/reference, Metal/reference, Metal/CPU exact argmax'
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
