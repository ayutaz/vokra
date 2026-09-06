#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT="$ROOT/tools/parity/audioldm2_reference"
die(){ echo "audioldm2-validation: BLOCKED: $*" >&2; exit 2; }
INSPECTOR="$ROOT/tools/parity/audioldm2_inspect.py"
usage(){ echo 'usage: run-audioldm2-validation.sh --expected-head <40-hex> --approval-evidence <file> --approval-sha256 <64-hex> [--self-test]' >&2; }
if [[ "${1:-}" == --self-test ]]; then
  [[ $# == 1 ]] || die '--self-test accepts no arguments'
  grep -Fq 'uv.lock' "$0"
  grep -Fq 'dedicated AudioLDM2 uv.lock is absent' "$0"
  grep -Fq 'UNRESOLVED_EXACT_PYTHON312_GRAPH' "$PROJECT/pyproject.toml"
  grep -Fq 'SOURCE_COMMIT' "$ROOT/tools/parity/audioldm2_dump_reference.py"
  grep -Fq 'NO_UPLOAD' "$ROOT/tools/parity/audioldm2_inspect.py"
  grep -Fq -- '--expected-head' "$0"
  grep -Fq -- '--approval-sha256' "$0"
  if "$0" --expected-head 0000000000000000000000000000000000000000 --expected-head 1111111111111111111111111111111111111111 >/dev/null 2>&1; then die 'duplicate --expected-head was accepted'; fi
  echo 'run-audioldm2-validation.sh self-test: OK'
  exit 0
fi
expected_head=''; approval_evidence=''; approval_sha256=''; seen_head=0; seen_approval=0; seen_approval_sha=0
while (($#)); do
  case "$1" in
    --expected-head) (( seen_head == 0 )) || die 'duplicate --expected-head'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head requires lowercase 40-hex'; expected_head="$2"; seen_head=1; shift 2 ;;
    --approval-evidence) (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--approval-evidence requires a path'; approval_evidence="$2"; seen_approval=1; shift 2 ;;
    --approval-sha256) (( seen_approval_sha == 0 )) || die 'duplicate --approval-sha256'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{64}$ ]] || die '--approval-sha256 requires lowercase 64-hex'; approval_sha256="$2"; seen_approval_sha=1; shift 2 ;;
    *) usage; die "unexpected argument: $1" ;;
  esac
done
(( seen_head == 1 )) || { usage; die '--expected-head is required'; }
(( seen_approval == 1 )) || { usage; die '--approval-evidence is required'; }
(( seen_approval_sha == 1 )) || { usage; die '--approval-sha256 is required'; }
[[ -f "$PROJECT/uv.lock" ]] || die 'dedicated AudioLDM2 uv.lock is absent; fail before downloads'
[[ -f "$approval_evidence" && ! -L "$approval_evidence" && -s "$approval_evidence" ]] || die 'approval evidence must be a non-empty regular file'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'VAST checkout must be clean'
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$expected_head" ]] || die 'checkout HEAD does not match --expected-head'
UV_CACHE_DIR="${AUDIO_LDM2_UV_CACHE_DIR:-/private/tmp/vokra-audioldm2-uv-cache}" \
  uv run --no-project --offline --python 3.12 python "$INSPECTOR" --validate-approval \
    --approval-evidence "$approval_evidence" --approval-sha256 "$approval_sha256" \
    --expected-head "$expected_head" >/dev/null || die 'external approval evidence is invalid'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'VAST requires Linux x86_64'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
die 'native PCM/parity validation is BLOCKED until authenticated official evidence exists'
