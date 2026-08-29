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
  grep -Fq 'REFERENCE_COMPLETE' "$ROOT/tools/parity/dia_1_6b_dump_reference.py" || die 'reference completion marker missing'
  grep -Fq 'uv.lock' "$ROOT/tools/parity/dia_1_6b_dump_reference.py" || die 'lock contract missing'
  check_project_identity
  grep -Fq 'dependency_license_audit = "BLOCKED_UNREVIEWED_TRANSITIVE"' "$PROJECT/pyproject.toml" || die 'dependency audit gate missing'
  if grep -Eq 'librosa|soxr|gradio|triton|nvidia-|descript-audio-codec' "$PROJECT/uv.lock"; then die 'forbidden/UI/GPL/CUDA reference dependency in lock'; fi
  UV_CACHE_DIR="${DIA_UV_CACHE_DIR:-/private/tmp/vokra-dia-uv-cache}" uv run --no-project --python 3.12 python "$ROOT/tools/parity/dia_1_6b_dump_reference.py" --self-test
  UV_CACHE_DIR="${DIA_UV_CACHE_DIR:-/private/tmp/vokra-dia-uv-cache}" uv run --no-project --python 3.12 python "$ROOT/tools/parity/dia_1_6b_validate_evidence.py" --self-test
  echo 'run-dia-1-6b-validation.sh self-test: OK'
}
if [[ "${1:-}" == --self-test ]]; then [[ $# == 1 ]] || die '--self-test accepts no arguments'; self_test; exit 0; fi
check_project_identity
grep -Fq 'dependency_license_audit = "AUDITED_ALLOW"' "$PROJECT/pyproject.toml" || die 'dependency license/provenance audit is not affirmatively allowed; refuse reference execution'
[[ $# == 7 ]] || die 'usage: run-dia-1-6b-validation.sh SOURCE_DIR MODEL_DIR PUBLIC_DIR DAC_SOURCE DAC_EVIDENCE DAC_CHECKPOINT EVIDENCE_DIR'
source_dir="$1"; model_dir="$2"; public_dir="$3"; dac_source="$4"; dac_evidence="$5"; dac_checkpoint="$6"; evidence="$7"
[[ -d "$source_dir" && -d "$model_dir" && -d "$public_dir" && -d "$dac_source" ]] || die 'source/model/public/DAC source directory is missing'
[[ ! -e "$evidence" || -z "$(find "$evidence" -mindepth 1 -print -quit 2>/dev/null)" ]] || die 'evidence directory must be absent or empty'
mkdir -p "$evidence"
UV_CACHE_DIR="${DIA_UV_CACHE_DIR:-/private/tmp/vokra-dia-uv-cache}" uv run --frozen --project "$PROJECT" --python 3.12 python "$ROOT/tools/parity/dia_1_6b_dump_reference.py" --source "$source_dir" --model "$model_dir" --public "$public_dir" --dac-source "$dac_source" --dac-evidence "$dac_evidence" --dac-checkpoint "$dac_checkpoint" --output "$evidence" >>"${evidence}.adapter.log" 2>&1 || die 'official reference adapter failed; inspect INSPECTION_ERROR'
UV_CACHE_DIR="${DIA_UV_CACHE_DIR:-/private/tmp/vokra-dia-uv-cache}" uv run --frozen --project "$PROJECT" --python 3.12 python "$ROOT/tools/parity/dia_1_6b_validate_evidence.py" "$evidence"
