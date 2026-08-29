#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT="$ROOT/tools/parity/audiogen_medium_reference"
INSPECTOR="$ROOT/tools/parity/audiogen_medium_inspect.py"
REFERENCE="$ROOT/tools/parity/audiogen_medium_dump_reference.py"
die(){ echo "audiogen-medium-validation: BLOCKED: $*" >&2; exit 2; }
if [[ "${1:-}" == --self-test ]]; then
  [[ $# == 1 ]] || die '--self-test accepts no arguments'
  grep -Fq 'torch.load' "$INSPECTOR"
  grep -Fq 'weights_only=True' "$INSPECTOR"
  grep -Fq 'NO_UPLOAD' "$INSPECTOR"
  UV_CACHE_DIR="${AUDIOGEN_UV_CACHE_DIR:-/private/tmp/vokra-audiogen-uv-cache}" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$REFERENCE" --self-test
  echo 'run-audiogen-medium-validation.sh self-test: OK'
  exit 0
fi
[[ $# == 0 ]] || die 'usage: run-audiogen-medium-validation.sh [--self-test]'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'VAST requires Linux x86_64'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -f "$PROJECT/uv.lock" ]] || die 'dedicated AudioGen uv.lock absent; fail before downloads'
die 'native AudioGen validation remains blocked pending authenticated native/composite parity'
