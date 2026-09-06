#!/usr/bin/env bash
# Apple verifier for the Kyutai STT dep_q=0 decoder component.
# It consumes only pre-existing authenticated inputs and never publishes.
set -euo pipefail

ROOT="${VOKRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TEST_NAME="parity_kyutai_stt_decoder_real_apple_cpu_metal"
usage() { printf '%s\n' "usage: apple-silicon-kyutai-stt-decoder.sh --gguf ABS --gguf-sha256 HEX64 --reference ABS --reference-manifest-sha256 HEX64 --evidence ABS --expected-head HEX40 | --self-test"; }
die() { printf '[kyutai-stt-apple] ERROR: %s\n' "$*" >&2; exit 2; }

self_test() {
  local self="${BASH_SOURCE[0]}" fail=0 token
  for token in "${TEST_NAME}" 'VOKRA_REMOTE_APPLE_SILICON=1' 'CARGO_NET_OFFLINE=true' 'NO_UPLOAD' 'FIXED_ATOL' 'CPU' 'Metal' 'verdict=PASS' '1 passed' 'manifest-sha256' '--expected-head' 'status=PASS'; do
    grep -Fq -- "$token" "$self" || { printf '[kyutai-stt-apple] missing self-test token: %s\n' "$token" >&2; fail=1; }
  done
  grep -Eq '(^|[;&|[:space:]])(curl|wget|git[[:space:]]+(clone|fetch|push)|hf_hub_download|upload_file|--push)([[:space:]]|$)' "$self" && fail=1 || true
  if "$self" --expected-head "${TEST_NAME//[^a-z]/a}" --expected-head "${TEST_NAME//[^a-z]/b}" >/dev/null 2>&1; then
    printf '%s\n' 'duplicate --expected-head was accepted' >&2
    fail=1
  fi
  (( fail == 0 )) || return 1
  printf '%s\n' 'kyutai STT Apple verifier self-test PASS'
}

[[ "${1:-}" == --self-test ]] && { [[ $# == 1 ]] || die '--self-test accepts no arguments'; self_test; exit $?; }
gguf=""; gguf_sha=""; reference=""; reference_sha=""; evidence=""; expected_head=""
gguf_seen=0; gguf_sha_seen=0; reference_seen=0; reference_sha_seen=0; evidence_seen=0; expected_head_seen=0
while (($#)); do
  case "$1" in
    --gguf) (($# >= 2)) || die '--gguf requires ABS'; (( gguf_seen == 0 )) || die 'duplicate --gguf'; gguf="$2"; gguf_seen=1; shift 2;;
    --gguf-sha256) (($# >= 2)) || die '--gguf-sha256 requires HEX64'; (( gguf_sha_seen == 0 )) || die 'duplicate --gguf-sha256'; gguf_sha="$2"; gguf_sha_seen=1; shift 2;;
    --reference) (($# >= 2)) || die '--reference requires ABS'; (( reference_seen == 0 )) || die 'duplicate --reference'; reference="$2"; reference_seen=1; shift 2;;
    --reference-manifest-sha256) (($# >= 2)) || die '--reference-manifest-sha256 requires HEX64'; (( reference_sha_seen == 0 )) || die 'duplicate --reference-manifest-sha256'; reference_sha="$2"; reference_sha_seen=1; shift 2;;
    --evidence) (($# >= 2)) || die '--evidence requires ABS'; (( evidence_seen == 0 )) || die 'duplicate --evidence'; evidence="$2"; evidence_seen=1; shift 2;;
    --expected-head) (($# >= 2)) || die '--expected-head requires HEX40'; (( expected_head_seen == 0 )) || die 'duplicate --expected-head'; expected_head="$2"; expected_head_seen=1; shift 2;;
    -h|--help) usage; exit 0;;
    *) usage; die "unknown argument: $1";;
  esac
done
[[ "$expected_head" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head must be lowercase HEX40'

[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || die 'requires Darwin arm64 Apple Silicon'
[[ "${VOKRA_REMOTE_APPLE_SILICON:-}" == 1 ]] || die 'VOKRA_REMOTE_APPLE_SILICON=1 is required'
[[ -f "$ROOT/Cargo.toml" && -d "$ROOT/.git" ]] || die 'not a Vokra checkout'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
actual_head="$(git -C "$ROOT" rev-parse HEAD)"
[[ "$actual_head" == "$expected_head" ]] || die "checkout HEAD $actual_head does not match --expected-head $expected_head"
[[ "$gguf" == /* && "$reference" == /* && "$evidence" == /* ]] || die 'all paths must be absolute'
for path in "$gguf" "$reference" "$evidence"; do
  [[ "$path" != *'/./'* && "$path" != *'/../'* && "$path" != */. && "$path" != */.. ]] || die 'dot path component rejected'
  [[ ! -L "$path" ]] || die 'symlink input/evidence rejected'
done
[[ -f "$gguf" && ! -L "$gguf" ]] || die 'GGUF must be a regular file'
[[ -d "$reference" && ! -L "$reference" ]] || die 'reference must be a regular directory'
[[ -f "$reference/manifest.json" && ! -L "$reference/manifest.json" ]] || die 'reference manifest is missing'
[[ ! -e "$evidence" ]] || die 'evidence must be absent (no-clobber)'
[[ "$gguf_sha" =~ ^[0-9a-f]{64}$ && "$reference_sha" =~ ^[0-9a-f]{64}$ ]] || die 'digests must be lowercase HEX64'
actual_gguf_sha="$(shasum -a 256 "$gguf" | awk '{print $1}')"
actual_reference_sha="$(shasum -a 256 "$reference/manifest.json" | awk '{print $1}')"
[[ "$actual_gguf_sha" == "$gguf_sha" ]] || die 'GGUF digest mismatch'
[[ "$actual_reference_sha" == "$reference_sha" ]] || die 'reference manifest digest mismatch'
gguf_real="$(cd "$(dirname "$gguf")" && pwd -P)/$(basename "$gguf")"
reference_real="$(cd "$reference" && pwd -P)"
evidence_parent="$(cd "$(dirname "$evidence")" && pwd -P)"
evidence_real="$evidence_parent/$(basename "$evidence")"
case "$gguf_real/" in "$reference_real/"*|"$evidence_real/"*|"$ROOT/"*) die 'GGUF overlaps another path';; esac
case "$reference_real/" in "$gguf_real/"*|"$evidence_real/"*|"$ROOT/"*) die 'reference overlaps another path';; esac
case "$ROOT/" in "$gguf_real/"*|"$reference_real/"*) die 'checkout overlaps an input';; esac
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
[[ "$(grep -Ec '^test result: ok\. 1 passed; 0 failed;' "$log")" == 1 ]] || die 'Cargo result is not exactly 1 passed / 0 failed'
[[ "$(grep -Ec '^KYUTAI_STT_DECODER_APPLE .* verdict=PASS$' "$log")" == 1 ]] || die 'Apple PASS sentinel missing or duplicated'
[[ "$(grep -Ec 'verdict=PASS' "$log")" == 1 ]] || die 'verdict is not an exact singleton'
[[ "$(grep -Ec '0 failed' "$log")" == 1 ]] || die 'failed count is not an exact singleton'
printf '%s\n' 'status=PASS' >> "$log"
[[ "$(grep -Ec '^status=PASS$' "$log")" == 1 ]] || die 'final PASS status missing or duplicated'
log_sha="$(shasum -a 256 "$log" | awk '{print $1}')"
printf '[kyutai-stt-apple] PASS: %s (log_sha256=%s)\n' "$log" "$log_sha" >&2
