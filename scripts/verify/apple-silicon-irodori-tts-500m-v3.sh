#!/usr/bin/env bash
# Apple Silicon preflight for Irodori-TTS-500M-v3.  It stops at the same
# stdlib-only dependency gate as VAST; no dependency sync or model execution.
set -euo pipefail

ROOT="${VOKRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SOURCE_REV='8224dafb46d0aba89209a8f905f1cb7e3299d9c1'
SOURCE_LOCK_SHA256='8175adbb9ad7ae77d1f048344343a63876e57c333b659314bcc054230b5b3e6c'

die() { printf '[irodori-apple] ERROR: %s\n' "$*" >&2; exit 2; }

run_dependency_gate() {
  local gate_output gate_status
  set +e
  gate_output="$(UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$ROOT/tools/parity/irodori_inspect.py" --dependency-gate 2>&1)"
  gate_status=$?
  set -e
  printf '%s\n' "$gate_output" >&2
  [[ $gate_status == 2 ]] || die "dependency gate returned unexpected status: $gate_status"
  return 1
}

self_test() {
  local token
  for token in Darwin arm64 IRODORI_VAST_BUNDLE ACCEPTED_VAST_CPU_BASELINE \
    BLOCKED_NATIVE_BINDING MEASURED_NOT_GATED NO_UPLOAD "$SOURCE_REV" "$SOURCE_LOCK_SHA256"; do
    grep -Fq -- "$token" "$0" || { echo "self-test missing $token" >&2; return 1; }
  done
  if grep -En '(^|[;&|][[:space:]]*)(curl|wget|snapshot_download|git[[:space:]]+push|cargo[[:space:]]+test|uv[[:space:]]+sync)([[:space:]]|$)' "$0" >/dev/null; then
    echo 'self-test found download/publication/sync/native cargo command' >&2
    return 1
  fi
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$ROOT/tools/parity/irodori_500m_v3_dump_reference.py" --self-test >/dev/null
  echo 'apple-silicon-irodori-tts-500m-v3.sh self-test: OK'
}

if [[ ${1:-} == --self-test ]]; then
  [[ $# == 1 ]] || die '--self-test accepts no arguments'
  self_test
  exit 0
fi
[[ $# == 1 ]] || die 'usage: apple-silicon-irodori-tts-500m-v3.sh VAST_BUNDLE | --self-test'
run_dependency_gate || die 'Irodori dependency/native closure is unresolved; no input probing or CPU/Metal execution is permitted'
[[ ${VOKRA_REMOTE_APPLE_SILICON:-0} == 1 ]] || die 'VOKRA_REMOTE_APPLE_SILICON=1 is required'
[[ $(uname -s) == Darwin && $(uname -m) == arm64 ]] || die 'Darwin arm64 is required'
memory="$(sysctl -n hw.memsize 2>/dev/null || true)"
[[ $memory =~ ^[0-9]+$ && $memory -ge 34359738368 ]] || die 'at least 32 GiB RAM is required'
command -v uv >/dev/null || die 'uv is required'
command -v xcrun >/dev/null || die 'xcrun is required'
xcrun -sdk macosx metal -v >/dev/null 2>&1 || die 'Metal compiler unavailable'

echo "[irodori-apple] BLOCKED_NATIVE_BINDING: source=$SOURCE_REV lock=$SOURCE_LOCK_SHA256; no dependency sync, source import, or model execution attempted; PCM=MEASURED_NOT_GATED; NO_UPLOAD." >&2
exit 2
