#!/usr/bin/env bash
# Collect the Dia locked dependency/license/native facts on VAST only.
# This job intentionally exits 2 while the owner/license gate is blocked.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
PROJECT="$VOKRA_ROOT/tools/parity/dia_1_6b_reference"
AUDITOR="$PROJECT/dependency_audit.py"
OWNER_SCOPE="$PROJECT/owner_review_scope.py"
DEPENDENCY_APPROVAL="$PROJECT/dependency_approval.py"
PREPARER="$VOKRA_ROOT/scripts/publish/vast-ai/prepare-dia-1-6b-reference.sh"
LOCK_SHA256="58218102471c94979b1e9147759abf50fa3784793c193ff30cdde908400650dc"
PYPROJECT_SHA256="fa675f2c7542bd9eebedcc6ba29963f49093305c7a518542d71fad424449e77b"
MIN_VAST_MEM_KIB=60000000

log() { printf '[dia-dependency-audit] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

paths_overlap() { [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]; }

canonicalize_uncreated() {
  local path="$1" suffix='' name parent component rest scan
  [[ "$path" == /* ]] || return 1
  rest="${path#/}"; scan=''
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"
    [[ "$rest" == "$component" ]] && rest='' || rest="${rest#*/}"
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || return 1
    scan="$scan/$component"
    [[ ! -L "$scan" ]] || return 1
  done
  while [[ ! -d "$path" || -L "$path" ]]; do
    name="${path##*/}"; [[ -n "$name" ]] && suffix="/$name$suffix"
    parent="${path%/*}"; [[ "$parent" == "$path" ]] && parent='/'
    path="$parent"; [[ ! -L "$path" ]] || return 1
  done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}

require_absent_output() {
  local output="$1" canonical root project
  [[ "$output" == /* && ! -e "$output" && ! -L "$output" ]] || return 1
  canonical="$(canonicalize_uncreated "$output")" || return 1
  root="$(canonicalize_uncreated "$VOKRA_ROOT")" || return 1
  project="$(canonicalize_uncreated "$PROJECT")" || return 1
  paths_overlap "$canonical" "$root" && return 1
  paths_overlap "$canonical" "$project" && return 1
  return 0
}

require_vast() {
  local memory
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || { die 'VOKRA_PUBLISH_ON_VAST=1 is required'; return 2; }
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { die 'Linux x86_64 VAST host required'; return 2; }
  command -v uv >/dev/null 2>&1 || { die 'uv is required'; return 2; }
  command -v readelf >/dev/null 2>&1 || { die 'readelf is required'; return 2; }
  command -v sha256sum >/dev/null 2>&1 || { die 'sha256sum is required'; return 2; }
  memory="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$memory" =~ ^[0-9]+$ && "$memory" -ge "$MIN_VAST_MEM_KIB" ]] || { die 'VAST RAM is below the 60-GB guard'; return 2; }
}

require_contract() {
  local path
  [[ -d "$VOKRA_ROOT/.git" || -f "$VOKRA_ROOT/.git" ]] || { die 'VOKRA_ROOT is not a Vokra checkout'; return 2; }
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || { die 'VAST checkout must be clean'; return 2; }
  for path in pyproject.toml uv.lock dependency_audit.py owner_review_scope.py dependency_approval.py; do
    [[ -f "$PROJECT/$path" && ! -L "$PROJECT/$path" ]] || { die "missing or symlinked Dia audit input: $path"; return 2; }
  done
  [[ -x "$PREPARER" && ! -L "$PREPARER" ]] || { die 'missing or symlinked Dia preparation helper'; return 2; }
  [[ "$(sha256sum "$PROJECT/uv.lock" | awk '{print $1}')" == "$LOCK_SHA256" ]] || { die 'Dia uv.lock identity mismatch'; return 2; }
  [[ "$(sha256sum "$PROJECT/pyproject.toml" | awk '{print $1}')" == "$PYPROJECT_SHA256" ]] || { die 'Dia pyproject identity mismatch'; return 2; }
}

normalize_audit_exit() {
  local report="$1" audit_rc="$2" gate_status
  (( audit_rc != 0 )) && return "$audit_rc"
  gate_status="$(UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$report" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)
if report.get("status") == "FACTS_COLLECTED_GATE_BLOCKED" and report.get("dependency_license_audit") == "BLOCKED_UNREVIEWED_TRANSITIVE" and report.get("publication") == "NO_UPLOAD":
    print("BLOCKED")
else:
    print("UNEXPECTED")
PY
)"
  [[ "$gate_status" == BLOCKED ]] && return 2
  return 0
}

run_audit() {
  local output="$1" out log_path rc scope_rc head environment preparation scope
  require_vast || return 2
  require_contract || return 2
  require_absent_output "$output" || { die 'output must be an absent absolute path outside the checkout/project'; return 2; }
  mkdir -p "$output"
  out="$(canonicalize_uncreated "$output")/dependency-audit.json"
  log_path="$(canonicalize_uncreated "$output")/audit.log"
  preparation="$(canonicalize_uncreated "$output")/preparation"
  environment="$preparation/venv"
  [[ "$environment" == /* && ! -L "$environment" ]] || { die 'DIA_AUDIT_ENVIRONMENT must be an absolute non-symlink path'; return 2; }
  log 'Preparing the exact frozen Dia project and no-BLAS NumPy wheel; no model/source/checkpoint acquisition'
  set +e
  VOKRA_PUBLISH_ON_VAST=1 DIA_REFERENCE_UV_CACHE_DIR="${DIA_AUDIT_UV_CACHE_DIR:-/tmp/vokra-dia-audit-uv-cache}" \
    "$PREPARER" --output-dir "$preparation" 2>&1 | tee "$log_path"
  rc="${PIPESTATUS[0]}"
  set -e
  (( rc == 0 )) || { log "Dia preparation failed (rc=$rc)"; return "$rc"; }
  log 'Collecting installed metadata, publisher LICENSE/NOTICE bytes, and native payload facts'
  set +e
  UV_PROJECT_ENVIRONMENT="$environment" UV_NO_CACHE=1 UV_CACHE_DIR="${DIA_AUDIT_UV_CACHE_DIR:-/tmp/vokra-dia-audit-uv-cache}" \
    uv run --project "$PROJECT" --frozen --no-sync --python 3.12 python "$AUDITOR" \
      --project "$PROJECT" --output "$out" 2>&1 | tee -a "$log_path"
  rc="${PIPESTATUS[0]}"
  set -e
  [[ -f "$out" && -f "$log_path" ]] || { log 'dependency audit did not emit its report'; return 2; }
  scope="$(canonicalize_uncreated "$output")/owner-review-scope.json"
  head="$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
  set +e
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$OWNER_SCOPE" \
    --report "$out" --preparation "$(canonicalize_uncreated "$preparation")/preparation.json" --expected-head "$head" --output "$scope" 2>&1 | tee -a "$log_path"
  scope_rc="${PIPESTATUS[0]}"
  set -e
  (( scope_rc == 0 )) || { log "owner-review scope generation failed (rc=$scope_rc)"; return 2; }
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$OWNER_SCOPE" \
    --validate --report "$out" --preparation "$(canonicalize_uncreated "$preparation")/preparation.json" --expected-head "$head" --output "$scope" >>"$log_path" 2>&1 || { log 'owner-review scope validation failed'; return 2; }
  (cd "$(canonicalize_uncreated "$output")" && sha256sum audit.log dependency-audit.json owner-review-scope.json preparation/preparation.json preparation/build-dependency-evidence.json preparation/numpy-config.json preparation/numpy-2.2.5.tar.gz preparation/wheelhouse/*.whl > SHA256SUMS)
  normalize_audit_exit "$out" "$rc"
}

self_test() {
  local temp_root fake_repo fake_project fake_output fake_log rc temp_parent failed=0 exit_probe
  for token in 'VOKRA_PUBLISH_ON_VAST=1' 'prepare-dia-1-6b-reference.sh' '--no-install-package numpy' '--no-sync' 'dependency_audit.py' 'owner_review_scope.py' 'dependency_approval.py' 'owner-review-scope.json' 'dependency-audit.json' 'publisher LICENSE/NOTICE bytes' 'native payload facts' 'NO_UPLOAD'; do
    grep -Fq -- "$token" "$0" || failed=1
  done
  if grep -En '^[[:space:]]*(python3?|pip)([[:space:]]|$)' "$0" | grep -v 'grep -En' >/dev/null; then failed=1; fi
  if grep -En 'snapshot_download|git[[:space:]]+clone|cargo[[:space:]]+(build|test|check|clippy)|publish-one\.sh|--push|--upload' "$0" | grep -v 'grep -En' >/dev/null; then failed=1; fi
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$AUDITOR" --self-test >/dev/null 2>&1 || failed=1
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$OWNER_SCOPE" --self-test >/dev/null 2>&1 || failed=1
  if temp_root="$(mktemp -d "${TMPDIR:-/tmp}/dia-dependency-exit-contract.XXXXXXXX")"; then
    UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$temp_root/blocked.json" <<'PY' || failed=1
import json, sys
json.dump({"status": "FACTS_COLLECTED_GATE_BLOCKED", "dependency_license_audit": "BLOCKED_UNREVIEWED_TRANSITIVE", "publication": "NO_UPLOAD"}, open(sys.argv[1], "w", encoding="utf-8"))
PY
    set +e
    normalize_audit_exit "$temp_root/blocked.json" 0
    exit_probe="$?"
    set -e
    (( exit_probe == 2 )) || failed=1
    set +e
    normalize_audit_exit "$temp_root/blocked.json" 7
    exit_probe="$?"
    set -e
    (( exit_probe == 7 )) || failed=1
    rm -rf "$temp_root"
  else
    failed=1
  fi
  temp_parent="${TMPDIR:-/tmp}"
  [[ -d /private/tmp && ! -L /private/tmp ]] && temp_parent=/private/tmp
  if temp_root="$(mktemp -d "$temp_parent/dia-dependency-wrapper.XXXXXXXX")"; then
    trap 'rm -rf "$temp_root"' EXIT
    if VOKRA_PUBLISH_ON_VAST=0 run_audit "$temp_root/blocked" >/dev/null 2>&1; then failed=1; fi
    [[ ! -e "$temp_root/blocked" ]] || failed=1
    if ! require_absent_output "$temp_root/valid-output"; then failed=1; fi
    if require_absent_output "$VOKRA_ROOT/audit.json"; then failed=1; fi
    if require_absent_output "$PROJECT/audit.json"; then failed=1; fi
    fake_repo="$temp_root/fake-repo"
    fake_project="$fake_repo/tools/parity/dia_1_6b_reference"
    fake_output="$temp_root/exception-report.json"
    fake_log="$temp_root/exception-report.log"
    mkdir -p "$fake_project"
    : > "$fake_project/pyproject.toml"
    : > "$fake_project/uv.lock"
    set +e
    UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$AUDITOR" \
      --project "$fake_project" --output "$fake_output" >"$fake_log" 2>&1
    rc="$?"
    set -e
    (( rc == 2 )) || failed=1
    UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$fake_output" <<'PY' || failed=1
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["schema"] == "vokra-dia-dependency-audit-v1"
assert report["status"] == "BLOCKED"
assert report["dependency_license_audit"] == "BLOCKED_UNREVIEWED_TRANSITIVE"
assert report["publication"] == "NO_UPLOAD"
PY
    rm -rf "$temp_root"
    trap - EXIT
  else
    failed=1
  fi
  (( failed == 0 )) || { log 'self-test FAIL'; return 1; }
  echo 'audit-dia-1-6b-dependencies.sh self-test: PASS (model-free, NO_UPLOAD)'
}

main() {
  local output='' self=0
  while (($#)); do
    case "$1" in
      --output-dir) [[ $# -eq 2 && -z "$output" && "$2" != -* ]] || { die 'invalid or duplicate --output-dir'; return 2; }; output="$2"; shift 2 ;;
      --self-test) (( self++ == 0 )) || { die 'duplicate --self-test'; return 2; }; shift ;;
      -h|--help) echo 'usage: audit-dia-1-6b-dependencies.sh --output-dir ABSENT_DIR | --self-test' >&2; return 0 ;;
      *) die "unknown argument: $1"; return 2 ;;
    esac
  done
  if (( self )); then [[ -z "$output" ]] || { die '--self-test accepts no output'; return 2; }; self_test; return $?; fi
  [[ -n "$output" ]] || { die '--output-dir is required'; return 2; }
  run_audit "$output"
}

main "$@"
