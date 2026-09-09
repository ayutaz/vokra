#!/usr/bin/env bash
# VAST-only ChatTTS composite evidence staging. Never converts, uploads, or publishes.
set -euo pipefail

INSPECTION_WORKER="scripts/publish/vast-ai/run-chattts-inspection.sh"
REFERENCE="tools/parity/chattts_dump_reference.py"
REFERENCE_LOCK_SHA256="6099870e3685fec99e8ae68745d37ce4e71138d353cf056540d092b3d55ac4c5"
REFERENCE_PYPROJECT_SHA256="d16bc02489442ee0134fb82903f3d08dbdc3f9d7b23a67d730cf9202fee23b9e"
REFERENCE_PACKAGE_INVENTORY_SHA256="74c0c3ef9afd095594e24afc48e0d2148717308aa317ea9762eb9b10d2f0ec7f"
REFERENCE_LOCK_PACKAGE_ROWS_SHA256="19395b8e7796dc26af01df77e3b786299391c38f3f861d2e9b59e29175b1cb4c"
REFERENCE_LICENSE_AUDIT_SHA256="3e5b662aa2134be84ee6645a7c483345d46550b690c582f414896c990a7f1dff"
UV_GATE_CMD=(uv run --no-cache --no-project --offline --python 3.12 python)
WORK_DIR="/dev/shm/vokra-chattts-validation"

die() { echo "run-chattts-validation: $*" >&2; exit 2; }

self_test() {
  local root fail=0 needle gate_pattern
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
  [[ -f "$root/$INSPECTION_WORKER" ]] || die "inspection worker missing"
  [[ -f "$root/$REFERENCE" ]] || die "official reference missing"
  for needle in "VOKRA_CHATTS_RUN_REFERENCE=1" "AUTHENTICATED_REFERENCE_EVIDENCE" "INSPECTION_ERROR" "NO_UPLOAD" "run-chattts-inspection.sh" "uv.lock" "$REFERENCE_LOCK_SHA256" "$REFERENCE_PYPROJECT_SHA256" "$REFERENCE_PACKAGE_INVENTORY_SHA256" "$REFERENCE_LOCK_PACKAGE_ROWS_SHA256" "$REFERENCE_LICENSE_AUDIT_SHA256" "transformers==5.10.4" "huggingface-hub==1.5.0" "GHSA-xrqw-3rrv-vx5w" "dependency-gate" "--validate-approval" "validate_reference_evidence" "canonical_absent_path" "INSPECTION_ONLY"; do
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
  if [[ -z "$gate_line" || -z "$sync_line" || "$gate_line" -ge "$sync_line" || "$gate_command" != *"UV_GATE_CMD"* || "$gate_command" == *"--project tools/parity/chattts"* ]] || ! grep -Fq 'UV_GATE_CMD=(uv run --no-cache --no-project --offline --python 3.12 python)' "${BASH_SOURCE[0]}"; then
    echo "self-test FAIL: dedicated sync must follow the no-project dependency gate" >&2; fail=1
  fi
  (( fail == 0 )) && echo "run-chattts-validation.sh self-test: OK" || return 1
}

if [[ "${1:-}" == "--self-test" ]]; then
  [[ $# -eq 1 ]] || die "--self-test accepts no extra arguments"
  self_test
  exit 0
fi
expected_head=''; approval_evidence=''; approval_sha256=''; seen_head=0; seen_approval=0; seen_sha=0
while (($#)); do case "$1" in
  --expected-head) (( seen_head == 0 )) || die 'duplicate --expected-head'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head requires lowercase 40-hex'; expected_head="$2"; seen_head=1; shift 2 ;;
  --approval-evidence) (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--approval-evidence requires a path'; approval_evidence="$2"; seen_approval=1; shift 2 ;;
  --approval-sha256) (( seen_sha == 0 )) || die 'duplicate --approval-sha256'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{64}$ ]] || die '--approval-sha256 requires lowercase 64-hex'; approval_sha256="$2"; seen_sha=1; shift 2 ;;
  *) die "unknown argument: $1" ;;
esac; done
(( seen_head == 1 && seen_approval == 1 && seen_sha == 1 )) || die 'expected HEAD and external approval are required'
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$root"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || die "worktree is not clean"
[[ -f "$root/tools/parity/chattts/uv.lock" ]] || die "ChatTTS dedicated locked environment missing"
[[ "$(sha256sum "$root/tools/parity/chattts/uv.lock" | awk '{print $1}')" == "$REFERENCE_LOCK_SHA256" ]] || die "ChatTTS dedicated uv.lock identity mismatch"
[[ "$(sha256sum "$root/tools/parity/chattts/pyproject.toml" | awk '{print $1}')" == "$REFERENCE_PYPROJECT_SHA256" ]] || die "ChatTTS dedicated pyproject identity mismatch"
[[ "$(git rev-parse HEAD)" == "$expected_head" ]] || die "checkout HEAD does not match --expected-head"
[[ -f "$approval_evidence" && ! -L "$approval_evidence" && -s "$approval_evidence" ]] || die "approval evidence is missing or symlinked"
UV_NO_CACHE=1 "${UV_GATE_CMD[@]}" "$root/$REFERENCE" --validate-approval --approval-evidence "$approval_evidence" --approval-sha256 "$approval_sha256" --expected-head "$expected_head" >/dev/null || die 'external approval evidence is invalid'
UV_NO_CACHE=1 "${UV_GATE_CMD[@]}" "$root/$REFERENCE" --dependency-gate || die "ChatTTS dependency/license gate is not explicitly approved"
die 'ChatTTS approval scope is INSPECTION_ONLY; acquisition and official reference execution remain unconditionally blocked'
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
root_real="$(canonical_existing_path "$root")" || die 'checkout path is not canonical'
approval_real="$(canonical_existing_path "$approval_evidence")" || die 'approval path has invalid canonical/symlink ancestry'
work_real="$(canonical_absent_path "$WORK_DIR")" || die 'validation work path must be absent and free of dot/symlink ancestry'
paths_overlap(){ local left="$1" right="$2"; [[ "$left" == "$right" || "$left/" == "$right/"* || "$right/" == "$left/"* ]]; }
paths_overlap "$work_real" "$root_real" && die 'validation work overlaps checkout'
paths_overlap "$work_real" "$approval_real" && die 'validation work overlaps approval'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "Linux x86_64 VAST required"
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die "VOKRA_PUBLISH_ON_VAST=1 is absent"
[[ "${VOKRA_CHATTS_RUN_REFERENCE:-0}" == 1 ]] || die "VOKRA_CHATTS_RUN_REFERENCE=1 is absent"
[[ "$(findmnt -T "$(dirname "$WORK_DIR")" -no FSTYPE 2>/dev/null || true)" == tmpfs ]] || die "validation parent must be tmpfs"
mem="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
[[ "$mem" =~ ^[0-9]+$ && "$mem" -ge $((128 * 1024 * 1024)) ]] || die "RAM below 128 GiB"
for command in bash git cargo rustfmt uv findmnt awk; do command -v "$command" >/dev/null 2>&1 || die "missing tool $command"; done
export CARGO_BUILD_JOBS=1
uv sync --project "$root/tools/parity/chattts" --frozen --python 3.12
mkdir "$WORK_DIR"
cargo fmt --all -- --check || die "cargo fmt check failed"
cargo metadata --locked --no-deps --format-version 1 >/dev/null || die "cargo metadata failed"

set +e
VOKRA_PUBLISH_ON_VAST=1 bash "$root/$INSPECTION_WORKER" --work-dir "$WORK_DIR/inspection" --expected-head "$expected_head" --approval-evidence "$approval_evidence" --approval-sha256 "$approval_sha256"
inspection_rc=$?
set -e
[[ "$inspection_rc" == 2 ]] || die "inspection worker returned $inspection_rc"
[[ -f "$WORK_DIR/inspection/evidence/manifest.json" ]] || die "inspection manifest missing"
[[ "$(git rev-parse HEAD)" == "$expected_head" ]] || die "checkout HEAD changed after inspection"
reference_rc=0
UV_CACHE_DIR="${UV_CACHE_DIR:-/dev/shm/vokra-uv-cache}" uv run --frozen --project tools/parity/chattts --python 3.12 python "$root/$REFERENCE" \
  --snapshot "$WORK_DIR/inspection/model" --source "$WORK_DIR/inspection/source" \
  --server-tree "$WORK_DIR/inspection/server-tree.json" \
  --output "$WORK_DIR/reference" --text "Hello." --seed 7 --max-new-token 1 --approval-evidence "$approval_evidence" --approval-sha256 "$approval_sha256" --expected-head "$expected_head" || reference_rc=$?
[[ "$reference_rc" == 2 ]] || die "reference returned $reference_rc"
UV_CACHE_DIR="${UV_CACHE_DIR:-/dev/shm/vokra-uv-cache}" uv run --frozen --project tools/parity/chattts --python 3.12 python - "$WORK_DIR/reference/manifest.json" "$root/tools/parity" "$expected_head" "$approval_sha256" <<'PY'
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
if set(manifest) != {"format","status","inspection_status","evidence_stage","runtime_status","native_status","cpu_status","metal_status","parity_status","publication","license_evidence","model","source","evidence","expected_head","approval_sha256"}:
    raise SystemExit("reference manifest schema drift")
if manifest.get("status") != "BLOCKED" or manifest.get("native_status") != "BLOCKED_NATIVE_BINDING" or manifest.get("publication") != "NO_UPLOAD":
    raise SystemExit("reference manifest is not fail-closed")
if manifest.get("inspection_status") != "AUTHENTICATED_REFERENCE_EVIDENCE":
    raise SystemExit(f"official reference evidence incomplete: {manifest.get('inspection_status')}")
if manifest.get("expected_head") != sys.argv[3] or manifest.get("approval_sha256") != sys.argv[4]:
    raise SystemExit("reference approval binding drifted")
validate_reference_evidence(manifest_path.parent, manifest.get("evidence"))
if reference_project_identity() != manifest["evidence"]["reference_project"]:
    raise SystemExit("dedicated ChatTTS project/lock identity is not independently bound")
PY
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || die "worktree became dirty during validation"
[[ "$(git rev-parse HEAD)" == "$expected_head" ]] || die "checkout HEAD changed during validation"
echo "ChatTTS composite evidence staged but runtime/parity/publication remain BLOCKED; no upload" >&2
exit 2
