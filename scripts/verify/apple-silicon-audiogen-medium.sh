#!/usr/bin/env bash
set -euo pipefail
die(){ echo "apple-silicon-audiogen-medium: BLOCKED: $*" >&2; exit 2; }
if [[ "${1:-}" == --self-test ]]; then
  [[ $# == 1 ]] || die '--self-test accepts no arguments'
  grep -Fq 'CPU/Metal' "$0"
  grep -Fq 'NOT_RUN' "$0"
  echo 'apple-silicon-audiogen-medium.sh self-test: OK'
  exit 0
fi
[[ $# == 0 ]] || die 'usage: apple-silicon-audiogen-medium.sh [--self-test]'
die 'CPU/Metal PCM generation and parity are NOT_RUN; public artifact remains LM-only'
