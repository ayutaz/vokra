#!/usr/bin/env bash
# Disposable Apple readiness gate for ChatTTS evidence. No download or upload.
set -euo pipefail

REFERENCE_LOCK_SHA256="265acb8b9ea68bdc7d5a1e8247e44a8410dfe6c4e3736fff9a5e6ef8d0ed5619"
REFERENCE_PACKAGE_INVENTORY_SHA256="a4d2d83a5b1d95419c9c913262cc596ea63e90f4532ad09e8d021c3ad07097bd"
REFERENCE_LOCK_PACKAGE_ROWS_SHA256="22247bf02552f4d31b613fa9a2944a339a0c066f652d7d2f25c8fe72b10d2307"
REFERENCE_LICENSE_AUDIT_SHA256="364ca38d14d269705641c9ccf0a350085fbfbe05705b98a09ee6ba1636f95fc5"

die() { echo "apple-silicon-chattts: $*" >&2; exit 2; }
self_test() {
  local self="${BASH_SOURCE[0]}" fail=0 needle
  for needle in "Darwin" "arm64" "VOKRA_REMOTE_APPLE_SILICON=1" "AUTHENTICATED_EVIDENCE_COMPLETE" "AUTHENTICATED_REFERENCE_EVIDENCE" "BLOCKED_NATIVE_BINDING" "NO_UPLOAD" "evidence directory must be outside" "uv.lock" "--no-sync" "--dependency-gate" "tools/parity" "$REFERENCE_LOCK_SHA256" "$REFERENCE_PACKAGE_INVENTORY_SHA256" "$REFERENCE_LOCK_PACKAGE_ROWS_SHA256" "$REFERENCE_LICENSE_AUDIT_SHA256"; do
    grep -Fq -- "$needle" "$self" || { echo "self-test FAIL: missing $needle" >&2; fail=1; }
  done
  if grep -En '(^|[[:space:]])(curl|wget|git[[:space:]]+clone|git[[:space:]]+push|cargo[[:space:]]+(run|test|check))([[:space:]]|$)' "$self" >/dev/null; then
    echo "self-test FAIL: download/publication/heavy Cargo path found" >&2; fail=1
  fi
  if grep -En 'uv[[:space:]]+sync|uv[[:space:]]+run.*--project[[:space:]]+tools/parity/chattts' "$self" >/dev/null; then
    echo "self-test FAIL: dedicated ChatTTS sync/runtime path found" >&2; fail=1
  fi
  (( fail == 0 )) && echo "apple-silicon-chattts.sh self-test: OK" || return 1
}
if [[ "${1:-}" == "--self-test" ]]; then
  [[ $# -eq 1 ]] || die "--self-test accepts no extra arguments"
  self_test
  exit 0
fi
[[ $# -eq 1 ]] || die "usage: $0 /path/to/VAST/evidence"
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || die "disposable Darwin arm64 required"
[[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == 1 ]] || die "VOKRA_REMOTE_APPLE_SILICON=1 is absent"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
evidence="$(cd "$1" 2>/dev/null && pwd)" || die "evidence directory unavailable"
case "$evidence/" in
  "$root/"*) die "evidence directory must be outside the Vokra checkout";;
esac
[[ -f "$evidence/inspection/manifest.json" && -f "$evidence/reference/manifest.json" ]] || die "VAST inspection/reference manifests required"
[[ -f "$root/tools/parity/chattts/uv.lock" ]] || die "ChatTTS dedicated locked environment missing"
[[ "$(shasum -a 256 "$root/tools/parity/chattts/uv.lock" | awk '{print $1}')" == "$REFERENCE_LOCK_SHA256" ]] || die "ChatTTS dedicated uv.lock identity mismatch"
# The Apple host validates the lock/license gate in the already-provisioned
# general parity environment.  It must never sync or execute the dedicated
# ChatTTS environment; the gate is checked before any manifest is trusted.
UV_CACHE_DIR="${UV_CACHE_DIR:-/private/tmp/vokra-uv-cache}" uv run --no-sync --frozen --project tools/parity --python 3.12 python - "$evidence/inspection/manifest.json" "$evidence/reference/manifest.json" "$root/tools/parity" "$root/tools/parity/chattts/pyproject.toml" "$root/tools/parity/chattts/uv.lock" <<'PY'
import json, sys
from pathlib import Path
def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise SystemExit(f"duplicate manifest key: {key}")
        result[key] = value
    return result
inspection, reference = [json.loads(Path(p).read_text(encoding="utf-8"), object_pairs_hook=pairs) for p in sys.argv[1:3]]
sys.path.insert(0, sys.argv[3])
from chattts_inspect import validate_inspection_manifest
from chattts_dump_reference import validate_dependency_gate, validate_reference_evidence
validate_dependency_gate(Path(sys.argv[4]), Path(sys.argv[5]))
validate_inspection_manifest(inspection)
if reference.get("inspection_status") != "AUTHENTICATED_REFERENCE_EVIDENCE":
    raise SystemExit("official reference evidence is incomplete")
validate_reference_evidence(Path(sys.argv[2]).parent, reference.get("evidence"))
for manifest in (inspection, reference):
    if manifest.get("status") != "BLOCKED" or manifest.get("native_status") != "BLOCKED_NATIVE_BINDING" or manifest.get("publication") != "NO_UPLOAD":
        raise SystemExit("fail-closed status/publication contract failed")
PY
echo "ChatTTS Apple BLOCKED_NATIVE_BINDING: native CPU/Metal and composite parity remain blocked; no upload" >&2
exit 2
