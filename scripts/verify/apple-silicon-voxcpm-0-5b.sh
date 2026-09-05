#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
die() { echo "voxcpm-apple: ERROR: $*" >&2; exit 2; }
self_test() {
  local failed=0 token
  for token in 'VOKRA_REMOTE_APPLE_SILICON=1' 'darwin' 'arm64' 'Metal' \
    'INSPECTION_ONLY' 'MEASURED_NOT_GATED' 'NO_UPLOAD' 'CPU' 'no fallback'; do
    grep -Fqi -- "$token" "$0" || { echo "missing Apple contract: $token" >&2; failed=1; }
  done
  if grep -En 'git[[:space:]]+push|upload\.sh|publish-one\.sh|--push|--upload|PASS' "$0" | grep -v 'grep -En' >/dev/null; then
    echo 'forbidden success/upload marker found' >&2; failed=1
  fi
  (( failed == 0 )) || return 1
  echo 'apple-silicon-voxcpm-0-5b.sh self-test: OK'
}
if [[ "${1:-}" == --self-test ]]; then [[ $# == 1 ]] || die '--self-test accepts no arguments'; self_test; exit 0; fi
[[ $# == 0 ]] || die 'usage: apple-silicon-voxcpm-0-5b.sh [--self-test]'
[[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == 1 ]] || die 'VOKRA_REMOTE_APPLE_SILICON=1 is absent'
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || die 'Apple Silicon Darwin host required'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
command -v xcrun >/dev/null || die 'Xcode xcrun is required'
xcrun metal -help >/dev/null 2>&1 || die 'Xcode Metal compiler is unavailable'
ram_bytes="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
[[ "$ram_bytes" =~ ^[0-9]+$ && "$ram_bytes" -ge $((32*1024*1024*1024)) ]] || die '32 GiB RAM guard failed'
gguf="${VOXCPM_APPLE_GGUF:-}"
reference="${VOXCPM_APPLE_REFERENCE:-}"
[[ -n "$gguf" && -f "$gguf" ]] || die 'VAST-staged corrected GGUF is required'
[[ -n "$reference" && -f "$reference" ]] || die 'VAST-staged reference/evidence is required'
[[ "${VOXCPM_APPLE_INPUT_CONVERSION:-VAST}" == VAST ]] || die 'input must be VAST-converted/staged'
echo 'VoxCPM-0.5B Apple route: INSPECTION_ONLY / MEASURED_NOT_GATED; real CPU+Metal route is not implemented, no fallback, no upload.' >&2
exit 2
