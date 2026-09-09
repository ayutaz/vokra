#!/usr/bin/env bash
set -euo pipefail
die(){ echo "apple-silicon-audioldm2: BLOCKED: $*" >&2; exit 2; }
if [[ "${1:-}" == --self-test ]]; then
  [[ $# == 1 ]] || die '--self-test accepts no arguments'
  grep -Fq 'CPU/Metal' "$0"
  grep -Fq 'NOT_RUN' "$0"
  echo 'apple-silicon-audioldm2.sh self-test: OK'
  exit 0
fi
[[ $# == 0 ]] || die 'usage: apple-silicon-audioldm2.sh [--self-test]'
die 'CPU/Metal native generation and parity are NOT_RUN; AudioLDM2 remains inspection-only'
