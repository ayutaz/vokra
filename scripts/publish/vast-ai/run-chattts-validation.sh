#!/usr/bin/env bash
# VAST-only ChatTTS composite evidence staging. Never converts, uploads, or publishes.
set -euo pipefail

INSPECTION_WORKER="scripts/publish/vast-ai/run-chattts-inspection.sh"
REFERENCE="tools/parity/chattts_dump_reference.py"
REFERENCE_LOCK_SHA256="36986402c3badb45b50c9d18ffbc811409be618cf45e2438f97e99c6751235db"
REFERENCE_PACKAGE_INVENTORY_SHA256="f8b00a8226662347ccf2e0ef7420922614ec570524ca6216852ee699f32db98a"
REFERENCE_LOCK_PACKAGE_ROWS_SHA256="9714e1a005af4800608f608c9617e0ce90dec0c563427e7c693d9c603ea2cf52"
REFERENCE_LICENSE_AUDIT_SHA256="38d0b49ad2b3fafd34bf19eaf1c955e53f0d7b5eb362612d0292a23d3e59148a"
UV_GATE_CMD=(uv run --no-project --python 3.12 python)
WORK_DIR="/dev/shm/vokra-chattts-validation"

die() { echo "run-chattts-validation: $*" >&2; exit 2; }

self_test() {
  local root fail=0 needle gate_pattern
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
  [[ -f "$root/$INSPECTION_WORKER" ]] || die "inspection worker missing"
  [[ -f "$root/$REFERENCE" ]] || die "official reference missing"
  for needle in "VOKRA_CHATTS_RUN_REFERENCE=1" "AUTHENTICATED_REFERENCE_EVIDENCE" "INSPECTION_ERROR" "NO_UPLOAD" "run-chattts-inspection.sh" "uv.lock" "$REFERENCE_LOCK_SHA256" "$REFERENCE_PACKAGE_INVENTORY_SHA256" "$REFERENCE_LOCK_PACKAGE_ROWS_SHA256" "$REFERENCE_LICENSE_AUDIT_SHA256" "dependency-gate" "validate_reference_evidence"; do
    if ! grep -Fq -- "$needle" "$root/$REFERENCE" && ! grep -Fq -- "$needle" "${BASH_SOURCE[0]}"; then
      echo "self-test FAIL: missing $needle" >&2; fail=1
    fi
  done
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh|vokra-cli[[:space:]]+convert|cargo[[:space:]]+(run|test|check))([[:space:]]|$)' "${BASH_SOURCE[0]}" >/dev/null; then
    echo "self-test FAIL: mutation/conversion/Cargo test found" >&2; fail=1
  fi
  if grep -En '^[[:space:]]*(python|python3|pip)([[:space:]]|$)' "${BASH_SOURCE[0]}" >/dev/null; then
    echo "self-test FAIL: raw Python/pip found" >&2; fail=1
  fi
  gate_pattern="\"\$root/\$REFERENCE\" --dependency-gate"
  gate_line="$(grep -nF "$gate_pattern" "${BASH_SOURCE[0]}" | tail -1 | cut -d: -f1)"
  sync_line="$(grep -n '^uv sync --project' "${BASH_SOURCE[0]}" | tail -1 | cut -d: -f1)"
  gate_command="$(sed -n "${gate_line}p" "${BASH_SOURCE[0]}")"
  if [[ -z "$gate_line" || -z "$sync_line" || "$gate_line" -ge "$sync_line" || "$gate_command" != *"UV_GATE_CMD"* || "$gate_command" == *"--project tools/parity/chattts"* ]] || ! grep -Fq 'UV_GATE_CMD=(uv run --no-project --python 3.12 python)' "${BASH_SOURCE[0]}"; then
    echo "self-test FAIL: dedicated sync must follow the no-project dependency gate" >&2; fail=1
  fi
  (( fail == 0 )) && echo "run-chattts-validation.sh self-test: OK" || return 1
}

if [[ "${1:-}" == "--self-test" ]]; then
  [[ $# -eq 1 ]] || die "--self-test accepts no extra arguments"
  self_test
  exit 0
fi
[[ $# -eq 0 ]] || die "no runtime arguments are accepted"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "Linux x86_64 VAST required"
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die "VOKRA_PUBLISH_ON_VAST=1 is absent"
[[ "${VOKRA_CHATTS_RUN_REFERENCE:-0}" == 1 ]] || die "VOKRA_CHATTS_RUN_REFERENCE=1 is absent"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$root"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || die "worktree is not clean"
[[ -f "$root/tools/parity/chattts/uv.lock" ]] || die "ChatTTS dedicated locked environment missing"
[[ "$(sha256sum "$root/tools/parity/chattts/uv.lock" | awk '{print $1}')" == "$REFERENCE_LOCK_SHA256" ]] || die "ChatTTS dedicated uv.lock identity mismatch"
[[ "$(findmnt -T "$(dirname "$WORK_DIR")" -no FSTYPE 2>/dev/null || true)" == tmpfs ]] || die "validation parent must be tmpfs"
mem="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
[[ "$mem" =~ ^[0-9]+$ && "$mem" -ge $((128 * 1024 * 1024)) ]] || die "RAM below 128 GiB"
for command in bash git cargo rustfmt uv findmnt awk; do command -v "$command" >/dev/null 2>&1 || die "missing tool $command"; done
export CARGO_BUILD_JOBS=1
UV_CACHE_DIR="${UV_CACHE_DIR:-/dev/shm/vokra-uv-cache}" "${UV_GATE_CMD[@]}" "$root/$REFERENCE" --dependency-gate || die "ChatTTS dependency/license gate is not explicitly approved"
uv sync --project "$root/tools/parity/chattts" --frozen --python 3.12
[[ ! -e "$WORK_DIR" || -d "$WORK_DIR" ]] || die "validation work path is not a directory"
[[ ! -e "$WORK_DIR" || -z "$(find "$WORK_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die "validation work path must be absent or empty"
mkdir -p "$WORK_DIR"
cargo fmt --all -- --check || die "cargo fmt check failed"
cargo metadata --locked --no-deps --format-version 1 >/dev/null || die "cargo metadata failed"

set +e
VOKRA_PUBLISH_ON_VAST=1 bash "$root/$INSPECTION_WORKER" --work-dir "$WORK_DIR/inspection"
inspection_rc=$?
set -e
[[ "$inspection_rc" == 2 ]] || die "inspection worker returned $inspection_rc"
[[ -f "$WORK_DIR/inspection/evidence/manifest.json" ]] || die "inspection manifest missing"
reference_rc=0
UV_CACHE_DIR="${UV_CACHE_DIR:-/dev/shm/vokra-uv-cache}" uv run --frozen --project tools/parity/chattts --python 3.12 python "$root/$REFERENCE" \
  --snapshot "$WORK_DIR/inspection/model" --source "$WORK_DIR/inspection/source" \
  --server-tree "$WORK_DIR/inspection/server-tree.json" \
  --output "$WORK_DIR/reference" --text "Hello." --seed 7 --max-new-token 1 || reference_rc=$?
[[ "$reference_rc" == 2 ]] || die "reference returned $reference_rc"
UV_CACHE_DIR="${UV_CACHE_DIR:-/dev/shm/vokra-uv-cache}" uv run --frozen --project tools/parity/chattts --python 3.12 python - "$WORK_DIR/reference/manifest.json" "$root/tools/parity" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from chattts_dump_reference import reference_project_identity, validate_reference_evidence
manifest_path = Path(sys.argv[1])
def pairs(items):
    out={}
    for key,value in items:
        if key in out: raise ValueError("duplicate manifest key")
        out[key]=value
    return out
manifest = json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=pairs)
if set(manifest) != {"format","status","inspection_status","evidence_stage","runtime_status","native_status","cpu_status","metal_status","parity_status","publication","license_evidence","model","source","evidence"}:
    raise SystemExit("reference manifest schema drift")
if manifest.get("status") != "BLOCKED" or manifest.get("native_status") != "BLOCKED_NATIVE_BINDING" or manifest.get("publication") != "NO_UPLOAD":
    raise SystemExit("reference manifest is not fail-closed")
if manifest.get("inspection_status") != "AUTHENTICATED_REFERENCE_EVIDENCE":
    raise SystemExit(f"official reference evidence incomplete: {manifest.get('inspection_status')}")
validate_reference_evidence(manifest_path.parent, manifest.get("evidence"))
if reference_project_identity() != manifest["evidence"]["reference_project"]:
    raise SystemExit("dedicated ChatTTS project/lock identity is not independently bound")
PY
echo "ChatTTS composite evidence staged but runtime/parity/publication remain BLOCKED; no upload" >&2
exit 2
