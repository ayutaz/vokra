#!/usr/bin/env bash
set -euo pipefail

# Apple worker is intentionally a fail-closed readiness gate.  It must not
# claim CPU/Metal support from synthetic metadata or an LLM-only GGUF.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REFERENCE="$ROOT/tools/parity/cosyvoice2_dump_reference.py"
die() { echo "cosyvoice2-apple: ERROR: $*" >&2; exit 2; }

run_dependency_gate() {
  local gate_output gate_status
  set +e
  gate_output="$(UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$REFERENCE" --dependency-gate 2>&1)"
  gate_status=$?
  set -e
  printf '%s\n' "$gate_output" >&2
  [[ "$gate_status" == 2 ]] || die "dependency gate returned unexpected status: $gate_status"
  return 1
}

self_test() {
  local fail=0 token
  for token in "AUTHENTICATED_REFERENCE_EVIDENCE" "NOT_IMPLEMENTED_FAIL_CLOSED" "sample_rate" "token_mel_ratio" "flow_rand_noise_full" "ras_nucleus_probability" "NO_UPLOAD" "native_status"; do
    grep -Fq -- "$token" "$REFERENCE" "$0" || { echo "missing contract: $token" >&2; fail=1; }
  done
  if grep -En 'git[[:space:]]+push|upload\.sh|publish-one\.sh|--push|--upload' "$0" | grep -v 'grep -En' >/dev/null; then fail=1; fi
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$REFERENCE" --self-test || fail=1
  (( fail == 0 )) || return 1
  echo 'apple-silicon-cosyvoice2.sh self-test: OK'
}

if [[ "${1:-}" == --self-test ]]; then
  [[ $# == 1 ]] || die '--self-test accepts no arguments'
  self_test
  exit 0
fi
[[ $# == 0 ]] || die 'usage: apple-silicon-cosyvoice2.sh [--self-test]'
run_dependency_gate || die 'CosyVoice2 dependency/license closure is unresolved; no input probing or CPU/Metal execution is permitted'
[[ "$(uname -s)" == Darwin ]] || die 'Apple Silicon worker requires macOS'
[[ "$(uname -m)" == arm64 ]] || die 'Apple Silicon worker requires arm64'
die 'CosyVoice2 CPU/Metal e2e is blocked until the complete composite binder and VAST reference evidence exist'
