#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

self_test() {
  grep -Fq 'INSPECTION_ONLY' "$ROOT/scripts/verify/apple-silicon-zonos-inspection.sh"
  grep -Fq 'CPU/Metal parity is not available' "$ROOT/scripts/verify/apple-silicon-zonos-inspection.sh"
  if grep -Eq '^echo .* (MEASURED|PASS|parity=PASS)' "$ROOT/scripts/verify/apple-silicon-zonos-inspection.sh"; then
    echo 'placeholder parity marker found' >&2
    return 1
  fi
  echo 'apple-silicon-zonos-inspection.sh self-test: OK'
}

if [[ "${1:-}" == --self-test ]]; then
  [[ $# == 1 ]] || { echo 'self-test accepts no arguments' >&2; exit 2; }
  self_test
  exit 0
fi
[[ $# == 0 ]] || { echo 'arguments are fixed' >&2; exit 2; }
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || {
  echo 'Zonos Apple verification requires real Darwin arm64 hardware' >&2
  exit 2
}

# This is a readiness blocker, not parity evidence. The fixed Zonos revision,
# complete 246 tensor manifest, crate::dac::Dac route, and conditioning packet
# are not authenticated; therefore no CPU or Metal binary is compiled/run and
# no PASS/MEASURED marker is emitted.
echo 'Zonos Apple verification INSPECTION_ONLY: CPU/Metal parity is not available until VAST authenticates the complete contract' >&2
exit 2
