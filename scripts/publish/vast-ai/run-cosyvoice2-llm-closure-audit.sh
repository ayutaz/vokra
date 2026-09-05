#!/usr/bin/env bash
# VAST-only lock generation, sync, and model-free closure audit.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PROJECT="$ROOT/tools/parity/cosyvoice2_llm_reference"
PREFLIGHT="$PROJECT/preflight_gate.py"
AUDIT="$PROJECT/audit_installed_closure.py"
WORK="${COSYVOICE2_LLM_AUDIT_WORK_DIR:-/dev/shm/vokra-cosyvoice2-llm-closure-audit}"
UV_CACHE_DIR="${COSYVOICE2_LLM_UV_CACHE_DIR:-/tmp/vokra-cosyvoice2-llm-uv-cache}"; export UV_CACHE_DIR
log() { printf '[cosyvoice2-llm-closure-vast] %s\n' "$*" >&2; }; die() { log "ERROR: $*"; exit 2; }
self_test() {
  local fail=0 token
  for token in 'uv lock' 'uv sync --frozen' 'audit_installed_closure.py' 'NO_MODEL_DOWNLOAD' 'NO_MODEL_EXECUTION' 'NO_UPLOAD' 'OWNER_REVIEW_REQUIRED' 'Linux x86_64 VAST' 'atomic' 'no-replace'; do
    grep -Fq -- "$token" "$AUDIT" "$PREFLIGHT" "$0" || { log "missing contract: $token"; fail=1; }
  done
  grep -En '(^|[[:space:]])(curl|wget|huggingface|git[[:space:]]+clone|.*llm\.pt|.*model.*download)([[:space:]]|$)' "$0" >/dev/null && { log 'model acquisition command found'; fail=1; } || true
  uv run --no-project --offline --python 3.12 python "$PREFLIGHT" --self-test >/dev/null || fail=1
  uv run --no-project --offline --python 3.12 python "$AUDIT" --self-test >/dev/null || fail=1
  (( fail == 0 )) && log 'self-test: OK' || return 1
}
self=0; audit_mode=0
while (($#)); do
  case "$1" in
    --self-test) (( self == 0 )) || die 'duplicate --self-test'; self=1; shift ;;
    --audit) (( audit_mode == 0 )) || die 'duplicate --audit'; audit_mode=1; shift ;;
    -h|--help) echo "usage: $0 --audit | --self-test"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if (( self )); then (( audit_mode == 0 )) || die '--self-test accepts no mode'; self_test; exit $?; fi
(( audit_mode == 1 )) || die 'explicit --audit is required'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'Linux x86_64 VAST required'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 required'
[[ -d "$ROOT/.git" && -f "$ROOT/Cargo.toml" ]] || die 'Vokra checkout required'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'clean checkout required'
for tool in awk df find findmnt git uv; do command -v "$tool" >/dev/null 2>&1 || die "missing tool: $tool"; done
mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"; [[ "$mem_kib" =~ ^[0-9]+$ && "$mem_kib" -ge $((16*1024*1024)) ]] || die 'RAM below 16 GiB'
parent="$(dirname "$WORK")"; [[ "$(findmnt -T "$parent" -no FSTYPE 2>/dev/null || true)" == tmpfs ]] || die 'work parent must be tmpfs'
free_kib="$(df -Pk "$parent" | awk 'NR == 2 {print $4}')"; [[ "$free_kib" =~ ^[0-9]+$ && "$free_kib" -ge $((8*1024*1024)) ]] || die 'tmpfs free space below 8 GiB'
[[ ! -e "$WORK" || -z "$(find "$WORK" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die 'work directory must be absent or empty'
mkdir -p "$WORK/evidence" "$WORK/logs"; WORK="$(cd "$WORK" && pwd)"; log_file="$WORK/logs/validation.log"
log 'generating the dedicated uv.lock before any model acquisition'
uv lock --project "$PROJECT" >>"$log_file" 2>&1 || die 'uv lock failed'
[[ -f "$PROJECT/uv.lock" && ! -L "$PROJECT/uv.lock" ]] || die 'uv.lock was not generated as a regular file'
log 'syncing the frozen dedicated project into VAST work storage'
UV_PROJECT_ENVIRONMENT="$WORK/.venv" uv sync --frozen --project "$PROJECT" >>"$log_file" 2>&1 || die 'uv sync --frozen failed'
report="$WORK/evidence/closure.json"
set +e
VOKRA_PUBLISH_ON_VAST=1 UV_PROJECT_ENVIRONMENT="$WORK/.venv" uv run --frozen --project "$PROJECT" python "$AUDIT" --project "$PROJECT/pyproject.toml" --lock "$PROJECT/uv.lock" --license-manifest "$PROJECT/license_gate_manifest.json" --output "$report" >>"$log_file" 2>&1
rc=$?
set -e
[[ "$rc" == 2 && -f "$report" ]] || die "closure audit returned unexpected status or no evidence: $rc"
mv "$log_file" "$WORK/evidence/validation.log"
log "closure evidence written to $report; owner review remains required; publication=NO_UPLOAD"
exit 2

