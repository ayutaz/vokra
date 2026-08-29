#!/usr/bin/env bash
# Apple Silicon inspection gate for MMS 1B all.
# The native MMS binder is intentionally fail-closed until VAST supplies an
# audited backbone+adapter manifest, so hardware discovery is not parity.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PARITY_SOURCE="$VOKRA_ROOT/crates/vokra-models/tests/parity_mms_1b_all_real.rs"

log() { printf '[mms-1b-all-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF'
usage: apple-silicon-mms-1b-all.sh --approval-evidence <file> --evidence-dir <absent-dir>
       apple-silicon-mms-1b-all.sh --self-test

Requires a disposable Darwin/arm64 host with VOKRA_REMOTE_APPLE_SILICON=1
and real Metal tooling. The current result is INSPECTION_ONLY because the
pinned MMS checkpoint manifest and native route are not reviewed. A hardware
probe is not CPU/Metal parity evidence; no model download or publication is
performed.
EOF
}

license_preflight() {
  local approval="$1" project="$VOKRA_ROOT/tools/parity/pyproject.toml" lock="$VOKRA_ROOT/tools/parity/uv.lock" project_sha lock_sha
  [[ -f "$approval" && ! -L "$approval" && -s "$approval" ]] || die 'approval evidence must be a nonempty regular non-symlink file'
  project_sha="$(shasum -a 256 "$project" | awk '{print $1}')"; lock_sha="$(shasum -a 256 "$lock" | awk '{print $1}')"
  if UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$approval" "$project_sha" "$lock_sha" <<'PY'
import hashlib, json, pathlib, sys
def reject(pairs):
    d = {}
    for k, v in pairs:
        if k in d: raise ValueError("duplicate JSON key: " + k)
        d[k] = v
    return d
try:
    d = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"), object_pairs_hook=reject)
    keys = {"schema", "model", "upstream_repo", "upstream_revision", "license_spdx", "project_sha256", "lock_sha256", "no_upload", "decision", "signer", "scope_sha256"}
    if set(d) != keys: raise ValueError("approval schema is not exact")
    if d["schema"] != "vokra-validation-approval-v1" or d["model"] != "mms-1b-all" or d["upstream_repo"] != "facebook/mms-1b-all" or d["upstream_revision"] != "3d33597edbdaaba14a8e858e2c8caa76e3cec0cd": raise ValueError("MMS identity mismatch")
    if d["license_spdx"] != "cc-by-nc-4.0" or d["project_sha256"] != sys.argv[2] or d["lock_sha256"] != sys.argv[3] or d["no_upload"] is not True or d["decision"] != "APPROVED": raise ValueError("approval facts mismatch")
    if not isinstance(d["signer"], str) or not d["signer"].strip() or d["signer"].strip().upper() in {"TODO", "UNRESOLVED", "OWNER_SIGNOFF_REQUIRED"}: raise ValueError("approval signer unresolved")
    scope = {"license_spdx": d["license_spdx"], "lock_sha256": sys.argv[3], "model": d["model"], "no_upload": True, "project_sha256": sys.argv[2], "upstream_repo": d["upstream_repo"], "upstream_revision": d["upstream_revision"]}
    if d["scope_sha256"] != hashlib.sha256(json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()).hexdigest(): raise ValueError("approval scope digest mismatch")
except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
    raise SystemExit("approval gate BLOCKED: " + str(exc))
PY
  then :; else die 'approval evidence is invalid or offline Python is unavailable'; fi
  if UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$VOKRA_ROOT/scripts/publish/signoff_match.py" --check-repo mms-1b-all --audit "$VOKRA_ROOT/docs/license-audit.md"; then :; else die 'MMS license/noncommercial signoff is unresolved'; fi
}

canonical_absent_path() {
  local path="$1" suffix='' rest component scan name parent
  [[ "$path" == /* ]] || path="$PWD/$path"; rest="${path#/}"; scan=''
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; rest="${rest#*/}"; [[ "$component" == "$rest" ]] && rest=''
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || continue
    scan="$scan/$component"; [[ ! -L "$scan" || "$scan" == /var ]] || return 1
  done
  while [[ ! -d "$path" || -L "$path" ]]; do
    name="${path##*/}"; [[ -n "$name" ]] && suffix="/$name$suffix"; parent="${path%/*}"; [[ "$parent" == "$path" ]] && parent=/; path="$parent"
  done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}
paths_overlap() { [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]; }
require_absent_evidence() {
  local target="$1" approval="$2" candidate other
  [[ ! -e "$target" && ! -L "$target" ]] || { die 'evidence directory must be absent and non-symlink'; return 2; }
  candidate="$(canonical_absent_path "$target")" || { die 'evidence directory has a symlinked ancestor'; return 2; }
  for other in "$VOKRA_ROOT" "$approval"; do
    [[ ! -L "$other" ]] || { die 'protected input is symlinked'; return 2; }
    local resolved; resolved="$(canonical_absent_path "$other")" || return 2
    paths_overlap "$candidate" "$resolved" && { die 'evidence directory overlaps protected input'; return 2; }
  done
}

self_test() {
  local path="${BASH_SOURCE[0]}" fail=0 token
  for token in 'VOKRA_REMOTE_APPLE_SILICON=1' 'Darwin' 'arm64' \
    'xcrun -f metal' 'INSPECTION_ONLY' 'hardware_probe_is_not_parity_evidence' \
    'backbone+adapter manifest' 'git status --porcelain' 'parity_mms_1b_all_real.rs'; do
    if ! grep -Fq -- "$token" "$path" && ! grep -Fq -- "$token" "$PARITY_SOURCE"; then
      log "self-test FAIL: missing contract token: $token"
      fail=1
    fi
  done
  if grep -En '^[[:space:]]*(curl|wget|python3?|pip|git[[:space:]]+push)([[:space:]]|$)' "$path" >/dev/null; then
    log 'self-test FAIL: acquisition or publication command found'
    fail=1
  fi
  if grep -En '^[[:space:]]*printf[^#]*PASS' "$path" >/dev/null; then
    log 'self-test FAIL: verifier manufactures a PASS marker'
    fail=1
  fi
  if "$path" --self-test --evidence-dir /tmp/not-accepted >/dev/null 2>&1; then
    log 'self-test FAIL: extra argument accepted'
    fail=1
  fi
  if "$path" --unknown-flag >/dev/null 2>&1; then
    log 'self-test FAIL: unknown argument accepted'
    fail=1
  fi
  for bad in '--approval-evidence' '--approval-evidence -bad' '--approval-evidence a --approval-evidence b' '--evidence-dir' '--evidence-dir -bad' '--evidence-dir a --evidence-dir b'; do
    if eval "\"$path\" $bad" >/dev/null 2>&1; then
      log "self-test FAIL: malformed or duplicate option accepted: $bad"
      fail=1
    fi
  done
  (( fail == 0 )) || return 1
  log 'self-test PASS'
}

evidence_dir=''; approval_evidence=''; self=0
seen_self=0; seen_evidence=0; seen_approval=0
while (($#)); do
  case "$1" in
    --self-test) (( seen_self == 0 )) || die 'duplicate --self-test'; seen_self=1; self=1; shift ;;
    --approval-evidence) (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--approval-evidence requires a nonempty path'; seen_approval=1; approval_evidence="$2"; shift 2 ;;
    --evidence-dir) (( seen_evidence == 0 )) || die 'duplicate --evidence-dir'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--evidence-dir requires a nonempty path'; seen_evidence=1; evidence_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if (( self )); then
  [[ -z "$evidence_dir$approval_evidence" ]] || die '--self-test accepts no other arguments'
  self_test
  exit $?
fi

[[ "$seen_approval" == 1 && "$seen_evidence" == 1 ]] || die '--approval-evidence and --evidence-dir are required'
license_preflight "$approval_evidence"
[[ -n "$evidence_dir" ]] || die '--evidence-dir is required'
[[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == 1 ]] || die 'VOKRA_REMOTE_APPLE_SILICON=1 is absent'
[[ "$(uname -s)" == Darwin ]] || die 'real Metal inspection requires Darwin'
[[ "$(uname -m)" == arm64 ]] || die 'real Metal inspection requires Apple arm64'
command -v xcrun >/dev/null 2>&1 || die 'xcrun is unavailable'
xcrun -f metal >/dev/null 2>&1 || die 'Xcode Metal compiler is unavailable'
[[ -f "$VOKRA_ROOT/Cargo.toml" && -d "$VOKRA_ROOT/.git" ]] || die 'not a Vokra checkout'
[[ -f "$PARITY_SOURCE" ]] || die 'MMS parity gate source is missing'
[[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'Apple checkout must be clean'
require_absent_evidence "$evidence_dir" "$approval_evidence"
mkdir -p "$evidence_dir"

{
  echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
  echo "host=$(uname -a)"
  echo "metal_compiler=$(xcrun -f metal)"
  echo 'runtime_status=INSPECTION_ONLY'
  echo 'parity_status=INSPECTION_ONLY'
  echo 'verdict=NO_CPU_OR_METAL_PASS'
  echo 'reason=pinned MMS backbone+adapter manifest and native route are not audited'
  echo 'hardware_probe_is_not_parity_evidence=true'
} > "$evidence_dir/mms-1b-all-apple-inspection.txt"
log "recorded inspection-only evidence at $evidence_dir; no CPU/Metal parity claim was emitted"
