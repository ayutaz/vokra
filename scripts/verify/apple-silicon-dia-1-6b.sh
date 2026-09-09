#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INSPECTOR="$ROOT/tools/parity/dia_1_6b_inspect.py"
die(){ echo "dia-apple: ERROR: $*" >&2; exit 2; }
self_test(){
  local file="$ROOT/tools/parity/dia_1_6b_dump_reference.py"
  for token in 'REFERENCE_COMPLETE' 'BLOCKED_UNTIL_VAST_AND_APPLE_EVIDENCE' 'NOT_RUN_OFFICIAL_ONLY' 'text_ids' 'selected_ids' 'delayed_codes' 'reverted_codes' 'dac_latent' 'pcm' 'NO_UPLOAD' 'DAC evidence is missing exact checkpoint' '--expected-head' '--approval-sha256'; do
    grep -Fq -- "$token" "$file" || die "missing reference contract: $token"
  done
  grep -Fq 'stale/orphan evidence file' "$ROOT/tools/parity/dia_1_6b_validate_evidence.py" || die 'independent validator missing'
  # This contract test is stdlib-only.  Never invoke the dedicated reference
  # project here: a frozen lock can still create/sync its environment before
  # the affirmative dependency-license gate is granted.
  grep -Fq 'UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python' "$0" || die 'self-test must use the offline no-cache stdlib route'
  grep -Fq 'validate_evidence.py' "$0" || die 'Apple path must invoke the independent validator'
  grep -Fq 'canonical_existing_path' "$0" || die 'canonical input path gate missing'
  grep -Fq 'evidence overlaps approval' "$0" || die 'approval/evidence disjoint gate missing'
  local uv_cmd='uv'
  if grep -Fq "$uv_cmd sync" "$0" || grep -Fq "$uv_cmd lock" "$0"; then die 'Apple self-test must not sync or lock dependencies'; fi
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$file" --self-test
  echo 'apple-silicon-dia-1-6b.sh self-test: OK'
}
if [[ "${1:-}" == --self-test ]]; then [[ $# == 1 ]] || die '--self-test accepts no arguments'; self_test; exit 0; fi
usage(){ echo 'usage: apple-silicon-dia-1-6b.sh --expected-head HEAD --approval-evidence FILE --approval-sha256 SHA --evidence-dir DIR' >&2; }
expected_head=''; approval_evidence=''; approval_sha256=''; evidence=''; seen_head=0; seen_approval=0; seen_sha=0; seen_evidence=0
while (($#)); do case "$1" in
 --expected-head) (( seen_head == 0 )) || die 'duplicate --expected-head'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head requires lowercase 40-hex'; expected_head="$2"; seen_head=1; shift 2 ;;
 --approval-evidence) (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--approval-evidence requires a path'; approval_evidence="$2"; seen_approval=1; shift 2 ;;
 --approval-sha256) (( seen_sha == 0 )) || die 'duplicate --approval-sha256'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{64}$ ]] || die '--approval-sha256 requires lowercase 64-hex'; approval_sha256="$2"; seen_sha=1; shift 2 ;;
 --evidence-dir) (( seen_evidence == 0 )) || die 'duplicate --evidence-dir'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--evidence-dir requires a path'; evidence="$2"; seen_evidence=1; shift 2 ;;
 *) usage; die "unexpected argument: $1" ;;
 esac; done
(( seen_head == 1 && seen_approval == 1 && seen_sha == 1 && seen_evidence == 1 )) || { usage; die 'all approval, HEAD, and evidence arguments are required'; }
canonical_existing_path() {
  local path="$1" rest component current=/ parent base
  [[ "$path" == /* && "$path" != */ && -e "$path" && ! -L "$path" ]] || return 1
  rest="${path#/}"
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; [[ "$rest" == "$component" ]] && rest='' || rest="${rest#*/}"
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || return 1
    current="${current%/}/$component"; [[ ! -L "$current" ]] || return 1
  done
  if [[ -d "$path" ]]; then (cd -P "$path" && pwd); else parent="$(dirname "$path")"; base="$(basename "$path")"; parent="$(cd -P "$parent" && pwd)" || return 1; printf '%s/%s\n' "$parent" "$base"; fi
}
approval_real="$(canonical_existing_path "$approval_evidence")" || die 'approval path has invalid absolute/canonical/symlink ancestry'
evidence_real="$(canonical_existing_path "$evidence")" || die 'evidence path has invalid absolute/canonical/symlink ancestry'
root_real="$(canonical_existing_path "$ROOT")" || die 'checkout path is not canonical/non-symlink'
paths_overlap(){ local left="$1" right="$2"; [[ "$left" == "$right" || "$left/" == "$right/"* || "$right/" == "$left/"* ]]; }
paths_overlap "$evidence_real" "$root_real" && die 'evidence overlaps checkout'
paths_overlap "$evidence_real" "$approval_real" && die 'evidence overlaps approval'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'Apple checkout must be clean'
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$expected_head" ]] || die 'checkout HEAD does not match --expected-head'
UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$INSPECTOR" --validate-approval --approval-evidence "$approval_evidence" --approval-sha256 "$approval_sha256" --expected-head "$expected_head" >/dev/null || die 'external approval evidence is invalid'
[[ "$(uname -s)" == Darwin ]] || die 'Apple Silicon verification requires macOS'
[[ "$(uname -m)" == arm64 ]] || die 'Apple Silicon verification requires arm64'
[[ -s "$evidence/manifest.json" ]] || die 'same-execution evidence manifest is missing'
UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$ROOT/tools/parity/dia_1_6b_validate_evidence.py" "$evidence" --expected-head "$expected_head" --approval-sha256 "$approval_sha256" || die 'same-execution evidence schema/hash validation failed'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'Apple checkout became dirty during evidence validation'
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$expected_head" ]] || die 'Apple checkout HEAD changed during evidence validation'
die 'Dia CPU/Metal route remains closed until independent VAST parity and this Apple worker evidence are reviewed'
