#!/usr/bin/env bash
# Disposable Apple readiness gate for ChatTTS evidence. No download or upload.
set -euo pipefail

REFERENCE_LOCK_SHA256="6099870e3685fec99e8ae68745d37ce4e71138d353cf056540d092b3d55ac4c5"
REFERENCE_PYPROJECT_SHA256="d16bc02489442ee0134fb82903f3d08dbdc3f9d7b23a67d730cf9202fee23b9e"
REFERENCE_PACKAGE_INVENTORY_SHA256="74c0c3ef9afd095594e24afc48e0d2148717308aa317ea9762eb9b10d2f0ec7f"
REFERENCE_LOCK_PACKAGE_ROWS_SHA256="19395b8e7796dc26af01df77e3b786299391c38f3f861d2e9b59e29175b1cb4c"
REFERENCE_LICENSE_AUDIT_SHA256="3e5b662aa2134be84ee6645a7c483345d46550b690c582f414896c990a7f1dff"
UV_GATE_CMD=(uv run --no-cache --no-project --offline --python 3.12 python)

die() { echo "apple-silicon-chattts: $*" >&2; exit 2; }
self_test() {
  local self="${BASH_SOURCE[0]}" fail=0 needle
  for needle in Darwin arm64 VOKRA_REMOTE_APPLE_SILICON=1 AUTHENTICATED_EVIDENCE_COMPLETE AUTHENTICATED_REFERENCE_EVIDENCE BLOCKED_NATIVE_BINDING NO_UPLOAD INSPECTION_ONLY --no-cache --offline --validate-approval --dependency-gate expected-head approval-sha256 canonical_existing_path evidence-overlaps "$REFERENCE_LOCK_SHA256" "$REFERENCE_PYPROJECT_SHA256" "$REFERENCE_PACKAGE_INVENTORY_SHA256" "$REFERENCE_LOCK_PACKAGE_ROWS_SHA256" "$REFERENCE_LICENSE_AUDIT_SHA256"; do
    grep -Fq -- "$needle" "$self" || { echo "self-test FAIL: missing $needle" >&2; fail=1; }
  done
  grep -Fq 'validate_reference_evidence' "$self" || fail=1
  if grep -En '(^|[[:space:]])(curl|wget|git[[:space:]]+clone|git[[:space:]]+push|cargo[[:space:]]+(run|test|check)|uv[[:space:]]+sync)' "$self" >/dev/null; then
    echo "self-test FAIL: download/publication/heavy Cargo/sync path found" >&2; fail=1
  fi
  (( fail == 0 )) && echo 'apple-silicon-chattts.sh self-test: OK' || return 1
}
if [[ "${1:-}" == --self-test ]]; then [[ $# == 1 ]] || die '--self-test accepts no extra arguments'; self_test; exit 0; fi

usage() { echo 'usage: apple-silicon-chattts.sh --expected-head HEAD --approval-evidence FILE --approval-sha256 SHA --evidence-dir DIR' >&2; }
expected_head=''; approval_evidence=''; approval_sha256=''; evidence=''; seen_head=0; seen_approval=0; seen_sha=0; seen_evidence=0
while (($#)); do case "$1" in
  --expected-head) (( seen_head == 0 )) || die 'duplicate --expected-head'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head requires lowercase 40-hex'; expected_head="$2"; seen_head=1; shift 2;;
  --approval-evidence) (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--approval-evidence requires a path'; approval_evidence="$2"; seen_approval=1; shift 2;;
  --approval-sha256) (( seen_sha == 0 )) || die 'duplicate --approval-sha256'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{64}$ ]] || die '--approval-sha256 requires lowercase 64-hex'; approval_sha256="$2"; seen_sha=1; shift 2;;
  --evidence-dir) (( seen_evidence == 0 )) || die 'duplicate --evidence-dir'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--evidence-dir requires a path'; evidence="$2"; seen_evidence=1; shift 2;;
  *) usage; die "unexpected argument: $1";;
esac; done
(( seen_head == 1 && seen_approval == 1 && seen_sha == 1 && seen_evidence == 1 )) || { usage; die 'all approval, HEAD, and evidence arguments are required'; }
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
canonical_existing_path() {
  local path="$1" rest component current=/ parent base
  [[ "$path" == /* && "$path" != */ && -e "$path" && ! -L "$path" ]] || return 1
  rest="${path#/}"
  while [[ -n "$rest" ]]; do component="${rest%%/*}"; [[ "$rest" == "$component" ]] && rest='' || rest="${rest#*/}"; [[ -n "$component" && "$component" != . && "$component" != .. ]] || return 1; current="${current%/}/$component"; [[ ! -L "$current" ]] || return 1; done
  if [[ -d "$path" ]]; then (cd -P "$path" && pwd); else parent="$(dirname "$path")"; base="$(basename "$path")"; parent="$(cd -P "$parent" && pwd)" || return 1; printf '%s/%s\n' "$parent" "$base"; fi
}
root_real="$(canonical_existing_path "$root")" || die 'checkout path is not canonical'
approval_real="$(canonical_existing_path "$approval_evidence")" || die 'approval path has invalid canonical/symlink ancestry'
evidence_real="$(canonical_existing_path "$evidence")" || die 'evidence path has invalid canonical/symlink ancestry'
paths_overlap(){ local left="$1" right="$2"; [[ "$left" == "$right" || "$left/" == "$right/"* || "$right/" == "$left/"* ]]; }
paths_overlap "$evidence_real" "$root_real" && die 'evidence directory must be outside the Vokra checkout'
paths_overlap "$evidence_real" "$approval_real" && die 'evidence-overlaps-approval'
[[ -f "$approval_evidence" && ! -L "$approval_evidence" && -s "$approval_evidence" ]] || die 'approval evidence missing or symlinked'
[[ -f "$root/tools/parity/chattts/uv.lock" && -f "$root/tools/parity/chattts/pyproject.toml" ]] || die 'ChatTTS dedicated project files missing'
[[ "$(shasum -a 256 "$root/tools/parity/chattts/uv.lock" | awk '{print $1}')" == "$REFERENCE_LOCK_SHA256" ]] || die 'ChatTTS uv.lock identity mismatch'
[[ "$(shasum -a 256 "$root/tools/parity/chattts/pyproject.toml" | awk '{print $1}')" == "$REFERENCE_PYPROJECT_SHA256" ]] || die 'ChatTTS pyproject identity mismatch'
[[ "$(git -C "$root" rev-parse HEAD)" == "$expected_head" ]] || die 'checkout HEAD does not match --expected-head'
UV_NO_CACHE=1 "${UV_GATE_CMD[@]}" "$root/tools/parity/chattts_dump_reference.py" --validate-approval --approval-evidence "$approval_evidence" --approval-sha256 "$approval_sha256" --expected-head "$expected_head" >/dev/null || die 'external approval evidence invalid'
UV_NO_CACHE=1 "${UV_GATE_CMD[@]}" "$root/tools/parity/chattts_dump_reference.py" --dependency-gate || die 'ChatTTS dependency/license gate is not approved'
die 'ChatTTS approval scope is INSPECTION_ONLY; no transfer packet or native execution is authorized'
[[ -f "$evidence/inspection/manifest.json" && ! -L "$evidence/inspection/manifest.json" && -f "$evidence/reference/manifest.json" && ! -L "$evidence/reference/manifest.json" ]] || die 'VAST inspection/reference manifests required'
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || die 'disposable Darwin arm64 required'
[[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == 1 ]] || die 'VOKRA_REMOTE_APPLE_SILICON=1 is absent'
cd "$root"
UV_NO_CACHE=1 "${UV_GATE_CMD[@]}" - "$evidence/inspection/manifest.json" "$evidence/reference/manifest.json" "$root/tools/parity/chattts/pyproject.toml" "$root/tools/parity/chattts/uv.lock" "$expected_head" "$approval_sha256" <<'PY'
import json, sys
from pathlib import Path
def pairs(items):
    out = {}
    for key, value in items:
        if key in out: raise SystemExit(f"duplicate manifest key: {key}")
        out[key] = value
    return out
inspection = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"), object_pairs_hook=pairs)
reference = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"), object_pairs_hook=pairs)
sys.path.insert(0, str(Path.cwd() / "tools" / "parity"))
from chattts_dump_reference import validate_dependency_gate, validate_reference_evidence
validate_dependency_gate(Path(sys.argv[3]), Path(sys.argv[4]))
required = {"format","status","inspection_status","evidence_stage","runtime_status","native_status","cpu_status","metal_status","parity_status","publication","license_evidence","model","source","evidence","expected_head","approval_sha256"}
if set(inspection) != required or inspection.get("status") != "BLOCKED" or inspection.get("inspection_status") != "AUTHENTICATED_EVIDENCE_COMPLETE" or inspection.get("native_status") != "BLOCKED_NATIVE_BINDING" or inspection.get("publication") != "NO_UPLOAD": raise SystemExit("inspection fail-closed schema drift")
if inspection.get("expected_head") != sys.argv[5] or inspection.get("approval_sha256") != sys.argv[6]: raise SystemExit("inspection approval binding drifted")
if set(reference) != required or reference.get("status") != "BLOCKED" or reference.get("inspection_status") != "AUTHENTICATED_REFERENCE_EVIDENCE" or reference.get("native_status") != "BLOCKED_NATIVE_BINDING" or reference.get("publication") != "NO_UPLOAD": raise SystemExit("reference fail-closed schema drift")
if reference.get("expected_head") != sys.argv[5] or reference.get("approval_sha256") != sys.argv[6]: raise SystemExit("reference approval binding drifted")
validate_reference_evidence(Path(sys.argv[2]).parent, reference.get("evidence"))
PY
[[ -z "$(git -C "$root" status --porcelain --untracked-files=all)" ]] || die 'Apple checkout became dirty'
[[ "$(git -C "$root" rev-parse HEAD)" == "$expected_head" ]] || die 'Apple checkout HEAD changed'
echo 'ChatTTS Apple BLOCKED_NATIVE_BINDING: native CPU/Metal and composite parity remain blocked; no upload' >&2
exit 2
