#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
die(){ echo "dia-apple: ERROR: $*" >&2; exit 2; }
self_test(){
  local file="$ROOT/tools/parity/dia_1_6b_dump_reference.py"
  for token in 'REFERENCE_COMPLETE' 'BLOCKED_UNTIL_VAST_AND_APPLE_EVIDENCE' 'NOT_RUN_OFFICIAL_ONLY' 'text_ids' 'selected_ids' 'delayed_codes' 'reverted_codes' 'dac_latent' 'pcm' 'NO_UPLOAD' 'DAC evidence is missing exact checkpoint'; do
    grep -Fq -- "$token" "$file" || die "missing reference contract: $token"
  done
  grep -Fq 'stale/orphan evidence file' "$ROOT/tools/parity/dia_1_6b_validate_evidence.py" || die 'independent validator missing'
  # This contract test is stdlib-only.  Never invoke the dedicated reference
  # project here: a frozen lock can still create/sync its environment before
  # the affirmative dependency-license gate is granted.
  grep -Fq 'uv run --no-project --python 3.12 python' "$0" || die 'self-test must use the no-project stdlib route'
  grep -Fq 'validate_evidence.py' "$0" || die 'Apple path must invoke the independent validator'
  local uv_cmd='uv'
  if grep -Fq "$uv_cmd sync" "$0" || grep -Fq "$uv_cmd lock" "$0"; then die 'Apple self-test must not sync or lock dependencies'; fi
  UV_CACHE_DIR="${DIA_UV_CACHE_DIR:-/private/tmp/vokra-dia-uv-cache}" uv run --no-project --python 3.12 python "$file" --self-test
  echo 'apple-silicon-dia-1-6b.sh self-test: OK'
}
if [[ "${1:-}" == --self-test ]]; then [[ $# == 1 ]] || die '--self-test accepts no arguments'; self_test; exit 0; fi
[[ $# == 1 ]] || die 'usage: apple-silicon-dia-1-6b.sh EVIDENCE_DIR'
[[ "$(uname -s)" == Darwin ]] || die 'Apple Silicon verification requires macOS'
[[ "$(uname -m)" == arm64 ]] || die 'Apple Silicon verification requires arm64'
evidence="$1"
[[ -s "$evidence/manifest.json" ]] || die 'same-execution evidence manifest is missing'
UV_CACHE_DIR="${DIA_UV_CACHE_DIR:-/private/tmp/vokra-dia-uv-cache}" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$ROOT/tools/parity/dia_1_6b_validate_evidence.py" "$evidence" || die 'same-execution evidence schema/hash validation failed'
die 'Dia CPU/Metal route remains closed until independent VAST parity and this Apple worker evidence are reviewed'
