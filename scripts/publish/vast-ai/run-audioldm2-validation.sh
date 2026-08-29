#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT="$ROOT/tools/parity/audioldm2_reference"
die(){ echo "audioldm2-validation: BLOCKED: $*" >&2; exit 2; }
if [[ "${1:-}" == --self-test ]]; then
  [[ $# == 1 ]] || die '--self-test accepts no arguments'
  grep -Fq 'uv.lock' "$0"
  grep -Fq 'dedicated AudioLDM2 uv.lock is absent' "$0"
  grep -Fq 'UNRESOLVED_EXACT_PYTHON312_GRAPH' "$PROJECT/pyproject.toml"
  grep -Fq 'SOURCE_COMMIT' "$ROOT/tools/parity/audioldm2_dump_reference.py"
  grep -Fq 'NO_UPLOAD' "$ROOT/tools/parity/audioldm2_inspect.py"
  echo 'run-audioldm2-validation.sh self-test: OK'
  exit 0
fi
[[ $# == 0 ]] || die 'usage: run-audioldm2-validation.sh [--self-test]'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'VAST requires Linux x86_64'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -f "$PROJECT/uv.lock" ]] || die 'dedicated AudioLDM2 uv.lock is absent; fail before downloads'
die 'native PCM/parity validation is BLOCKED until authenticated official evidence exists'
