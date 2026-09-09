#!/usr/bin/env bash
set -euo pipefail
die(){ echo "apple-silicon-audiogen-medium: BLOCKED: $*" >&2; exit 2; }
if [[ "${1:-}" == --self-test ]]; then
  [[ $# == 1 ]] || die '--self-test accepts no arguments'
  grep -Fq 'CPU/Metal' "$0"
  grep -Fq 'NOT_RUN' "$0"
  grep -Fq 'public artifact remains LM-only' "$0"
  grep -Fq 'VAST-to-Apple transfer packet is unavailable' "$0"
  pass_word='PA'; pass_suffix='SS';
  if grep -Eq "verdict=${pass_word}${pass_suffix}|metal_vs_(cpu|official)=${pass_word}${pass_suffix}" "$0"; then die 'self-test found an unearned PASS marker'; fi
  grep -Fq 'CPU/Metal PCM generation and parity are NOT_RUN' "$0"
  echo 'apple-silicon-audiogen-medium.sh self-test: OK'
  exit 0
fi
[[ $# == 0 ]] || die 'usage: apple-silicon-audiogen-medium.sh [--self-test]'
die 'CPU/Metal PCM generation and parity are NOT_RUN; public artifact remains LM-only; VAST-to-Apple transfer packet is unavailable'
