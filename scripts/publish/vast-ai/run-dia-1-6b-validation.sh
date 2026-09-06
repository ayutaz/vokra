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
  if grep -Eq 'librosa|soxr|gradio|triton|nvidia-|descript-audio-codec' "$PROJECT/uv.lock"; then die 'forbidden/UI/GPL/CUDA reference dependency in lock'; fi
  UV_CACHE_DIR="${DIA_UV_CACHE_DIR:-/private/tmp/vokra-dia-uv-cache}" uv run --no-project --python 3.12 python "$ROOT/tools/parity/dia_1_6b_dump_reference.py" --self-test
  UV_CACHE_DIR="${DIA_UV_CACHE_DIR:-/private/tmp/vokra-dia-uv-cache}" uv run --no-project --python 3.12 python "$ROOT/tools/parity/dia_1_6b_validate_evidence.py" --self-test
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
UV_CACHE_DIR="${DIA_UV_CACHE_DIR:-/private/tmp/vokra-dia-uv-cache}" uv run --no-project --python 3.12 python "$ROOT/tools/parity/dia_1_6b_inspect.py" --validate-approval --approval-evidence "$approval_evidence" --approval-sha256 "$approval_sha256" --expected-head "$expected_head" >/dev/null || die 'external approval evidence is invalid'
[[ -d "$source_dir" && ! -L "$source_dir" && -d "$model_dir" && ! -L "$model_dir" && -d "$public_dir" && ! -L "$public_dir" && -d "$dac_source" && ! -L "$dac_source" ]] || die 'source/model/public/DAC source directory is missing or symlinked'
[[ ! -e "$evidence" && ! -L "$evidence" ]] || die 'evidence directory must be absent/non-symlink'
root_real="$(cd -P "$ROOT" && pwd)"; evidence_real="$(cd -P "$(dirname "$evidence")" && pwd)/$(basename "$evidence")"; approval_real="$(cd -P "$(dirname "$approval_evidence")" && pwd)/$(basename "$approval_evidence")"
paths_overlap(){ local left="$1" right="$2"; [[ "$left" == "$right" || "$left/" == "$right/"* || "$right/" == "$left/"* ]]; }
paths_overlap "$evidence_real" "$root_real" && die 'evidence overlaps checkout'
paths_overlap "$evidence_real" "$approval_real" && die 'evidence overlaps approval'
mkdir "$evidence"
UV_CACHE_DIR="${DIA_UV_CACHE_DIR:-/private/tmp/vokra-dia-uv-cache}" uv run --frozen --project "$PROJECT" --python 3.12 python "$ROOT/tools/parity/dia_1_6b_dump_reference.py" --source "$source_dir" --model "$model_dir" --public "$public_dir" --dac-source "$dac_source" --dac-evidence "$dac_evidence" --dac-checkpoint "$dac_checkpoint" --output "$evidence" --expected-head "$expected_head" --approval-sha256 "$approval_sha256" >>"${evidence}.adapter.log" 2>&1 || die 'official reference adapter failed; inspect INSPECTION_ERROR'
UV_CACHE_DIR="${DIA_UV_CACHE_DIR:-/private/tmp/vokra-dia-uv-cache}" uv run --frozen --project "$PROJECT" --python 3.12 python "$ROOT/tools/parity/dia_1_6b_validate_evidence.py" "$evidence" --expected-head "$expected_head" --approval-sha256 "$approval_sha256"
