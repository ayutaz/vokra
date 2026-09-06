#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT="$ROOT/tools/parity/dia_1_6b_reference"
LOCK_SHA256="ccdfaf4cfedd7780f8c1032a42341f28ac56bec7353f4563f9a1b44b764cf29c"
PYPROJECT_SHA256="56430b6f50620df9ce3383f535dec1755843a4a9bab9758e34cf69e9913b6fc2"
die(){ echo "dia-validation: ERROR: $*" >&2; exit 2; }

check_project_identity() {
  [[ -f "$PROJECT/uv.lock" ]] || die 'dedicated Dia reference uv.lock is absent; refuse validation'
  [[ "$(sha256sum "$PROJECT/uv.lock" | awk '{print $1}')" == "$LOCK_SHA256" ]] || die 'dedicated Dia uv.lock identity mismatch'
  [[ "$(sha256sum "$PROJECT/pyproject.toml" | awk '{print $1}')" == "$PYPROJECT_SHA256" ]] || die 'dedicated Dia pyproject identity mismatch'
}

self_test(){
  grep -Fq -- '--expected-head' "$0" || die 'expected HEAD gate missing'
  grep -Fq -- '--approval-sha256' "$0" || die 'approval SHA gate missing'
  grep -Fq 'REFERENCE_COMPLETE' "$ROOT/tools/parity/dia_1_6b_dump_reference.py" || die 'reference completion marker missing'
  grep -Fq 'uv.lock' "$ROOT/tools/parity/dia_1_6b_dump_reference.py" || die 'lock contract missing'
  check_project_identity
  grep -Fq 'dependency_license_audit = "BLOCKED_UNREVIEWED_TRANSITIVE"' "$PROJECT/pyproject.toml" || die 'dependency audit gate missing'
  grep -Fq 'duplicate --expected-head' "$0" || die 'duplicate expected-head rejection missing'
  grep -Fq 'duplicate --approval-evidence' "$0" || die 'duplicate approval rejection missing'
  grep -Fq 'duplicate --approval-sha256' "$0" || die 'duplicate approval SHA rejection missing'
  grep -Fq -- '--validate-approval' "$0" || die 'approval validation mode missing'
  grep -Fq 'canonical_existing_path' "$0" || die 'canonical input path gate missing'
  grep -Fq 'canonical_absent_path' "$0" || die 'canonical evidence path gate missing'
  grep -Fq 'adapter log claim failed' "$0" || die 'adapter log no-clobber gate missing'
  grep -Fq 'checkout HEAD changed during validation' "$0" || die 'post-run HEAD gate missing'
  if grep -Eq 'librosa|soxr|gradio|triton|nvidia-|descript-audio-codec' "$PROJECT/uv.lock"; then die 'forbidden/UI/GPL/CUDA reference dependency in lock'; fi
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$ROOT/tools/parity/dia_1_6b_dump_reference.py" --self-test
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$ROOT/tools/parity/dia_1_6b_validate_evidence.py" --self-test
  echo 'run-dia-1-6b-validation.sh self-test: OK'
}
if [[ "${1:-}" == --self-test ]]; then [[ $# == 1 ]] || die '--self-test accepts no arguments'; self_test; exit 0; fi
check_project_identity
grep -Fq 'dependency_license_audit = "AUDITED_ALLOW"' "$PROJECT/pyproject.toml" || die 'dependency license/provenance audit is not affirmatively allowed; refuse reference execution'
usage(){ echo 'usage: run-dia-1-6b-validation.sh --expected-head HEAD --approval-evidence FILE --approval-sha256 SHA SOURCE_DIR MODEL_DIR PUBLIC_DIR DAC_SOURCE DAC_EVIDENCE DAC_CHECKPOINT EVIDENCE_DIR' >&2; }
expected_head=''; approval_evidence=''; approval_sha256=''; seen_head=0; seen_approval=0; seen_sha=0; positional=()
while (($#)); do case "$1" in
 --expected-head) (( seen_head == 0 )) || die 'duplicate --expected-head'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head requires lowercase 40-hex'; expected_head="$2"; seen_head=1; shift 2 ;;
 --approval-evidence) (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--approval-evidence requires a path'; approval_evidence="$2"; seen_approval=1; shift 2 ;;
 --approval-sha256) (( seen_sha == 0 )) || die 'duplicate --approval-sha256'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{64}$ ]] || die '--approval-sha256 requires lowercase 64-hex'; approval_sha256="$2"; seen_sha=1; shift 2 ;;
 *) positional+=("$1"); shift ;;
 esac; done
(( seen_head == 1 && seen_approval == 1 && seen_sha == 1 && ${#positional[@]} == 7 )) || { usage; die 'expected HEAD, approval, and seven input paths'; }
source_dir="${positional[0]}"; model_dir="${positional[1]}"; public_dir="${positional[2]}"; dac_source="${positional[3]}"; dac_evidence="${positional[4]}"; dac_checkpoint="${positional[5]}"; evidence="${positional[6]}"
[[ -f "$approval_evidence" && ! -L "$approval_evidence" && -s "$approval_evidence" ]] || die 'approval evidence is missing or symlinked'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$expected_head" ]] || die 'checkout HEAD does not match --expected-head'
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
canonical_absent_path() {
  local path="$1" target="$1" rest component current=/ suffix='' parent
  [[ "$path" == /* && "$path" != */ && ! -e "$path" && ! -L "$path" ]] || return 1
  rest="${path#/}"
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; [[ "$rest" == "$component" ]] && rest='' || rest="${rest#*/}"
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || return 1
    current="${current%/}/$component"; [[ ! -L "$current" ]] || return 1
  done
  while [[ ! -e "$target" && ! -L "$target" ]]; do
    suffix="/$(basename "$target")$suffix"; parent="$(dirname "$target")"; [[ "$parent" != "$target" ]] || return 1; target="$parent"
  done
  [[ -d "$target" && ! -L "$target" ]] || return 1
  printf '%s%s\n' "$(cd -P "$target" && pwd)" "$suffix"
}
root_real="$(canonical_existing_path "$ROOT")" || die 'checkout path is not canonical/non-symlink'
approval_real="$(canonical_existing_path "$approval_evidence")" || die 'approval path has invalid absolute/canonical/symlink ancestry'
UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$ROOT/tools/parity/dia_1_6b_inspect.py" --validate-approval --approval-evidence "$approval_evidence" --approval-sha256 "$approval_sha256" --expected-head "$expected_head" >/dev/null || die 'external approval evidence is invalid'
for input in "$source_dir" "$model_dir" "$public_dir" "$dac_source" "$dac_evidence" "$dac_checkpoint"; do
  canonical_existing_path "$input" >/dev/null || die "input path has invalid absolute/canonical/symlink ancestry: $input"
done
[[ -d "$source_dir" && -d "$model_dir" && -d "$public_dir" && -d "$dac_source" ]] || die 'source/model/public/DAC source directory is missing or not a directory'
evidence_real="$(canonical_absent_path "$evidence")" || die 'evidence path must be an absent absolute path without dot/symlink ancestry'
adapter_log="${evidence}.adapter.log"
canonical_absent_path "$adapter_log" >/dev/null || die 'adapter log must be absent and canonical'
paths_overlap(){ local left="$1" right="$2"; [[ "$left" == "$right" || "$left/" == "$right/"* || "$right/" == "$left/"* ]]; }
for protected in "$root_real" "$approval_real"; do paths_overlap "$evidence_real" "$protected" && die 'evidence overlaps checkout or approval'; done
for input in "$source_dir" "$model_dir" "$public_dir" "$dac_source" "$dac_evidence" "$dac_checkpoint"; do
  input_real="$(canonical_existing_path "$input")" || die "input path cannot be canonicalized: $input"
  paths_overlap "$evidence_real" "$input_real" && die 'evidence overlaps an input path'
done
mkdir "$evidence"
( set -C; : > "$adapter_log" ) || die 'adapter log claim failed (existing path or race)'
UV_CACHE_DIR="${DIA_UV_CACHE_DIR:-/private/tmp/vokra-dia-uv-cache}" uv run --frozen --project "$PROJECT" --python 3.12 python "$ROOT/tools/parity/dia_1_6b_dump_reference.py" --source "$source_dir" --model "$model_dir" --public "$public_dir" --dac-source "$dac_source" --dac-evidence "$dac_evidence" --dac-checkpoint "$dac_checkpoint" --output "$evidence" --expected-head "$expected_head" --approval-sha256 "$approval_sha256" >>"$adapter_log" 2>&1 || die 'official reference adapter failed; inspect INSPECTION_ERROR'
UV_CACHE_DIR="${DIA_UV_CACHE_DIR:-/private/tmp/vokra-dia-uv-cache}" uv run --frozen --project "$PROJECT" --python 3.12 python "$ROOT/tools/parity/dia_1_6b_validate_evidence.py" "$evidence" --expected-head "$expected_head" --approval-sha256 "$approval_sha256"
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout became dirty during validation'
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$expected_head" ]] || die 'checkout HEAD changed during validation'
