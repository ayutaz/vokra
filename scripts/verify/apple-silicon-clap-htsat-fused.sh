#!/usr/bin/env bash
# Apple Silicon inspection gate for CLAP HTSAT.
# There is no live native binder until the VAST tensor manifest is audited;
# this wrapper therefore records an explicit INSPECTION_ONLY result and
# never fabricates CPU/Metal parity.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PARITY_SOURCE="$VOKRA_ROOT/crates/vokra-models/tests/parity_clap_htsat_fused_real.rs"
DEDICATED_PROJECT="$VOKRA_ROOT/tools/parity/clap_htsat_fused_reference"

log() { printf '[clap-htsat-fused-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }
CLAP_APPLE_SELF_TEST_TMP=""
# shellcheck disable=SC2329 # Invoked by the EXIT trap below.
cleanup_self_test() {
  [[ -n "$CLAP_APPLE_SELF_TEST_TMP" ]] && rm -rf -- "$CLAP_APPLE_SELF_TEST_TMP"
}

require_preflight() {
  local project="${1:-$DEDICATED_PROJECT}" approval="${2:-}"
  local gate="$project/license_gate.py" manifest="$project/license_gate_manifest.json"
  [[ -d "$project" && ! -L "$project" ]] || { log 'dedicated CLAP reference project is missing; identity/license gate is unresolved'; return 2; }
  [[ -f "$project/pyproject.toml" && ! -L "$project/pyproject.toml" ]] || { log 'dedicated CLAP pyproject.toml is missing'; return 2; }
  [[ -f "$project/uv.lock" && ! -L "$project/uv.lock" ]] || { log 'dedicated CLAP uv.lock is missing; refuse before evidence'; return 2; }
  [[ -f "$gate" && ! -L "$gate" && -f "$manifest" && ! -L "$manifest" ]] || { log 'CLAP license gate/manifest is missing; refuse before evidence'; return 2; }
  [[ -f "$approval" && ! -L "$approval" && -s "$approval" ]] || { log 'approval evidence must be a nonempty regular file'; return 2; }
}

validate_absent_evidence() {
  local path="$1" approval="$2" component rest current parent candidate item approval_parent approval_real root_real
  local -a suffix=()
  [[ "$path" == /* && "$path" != *$'\n'* && "$path" != *$'\r'* ]] || return 2
  [[ "$path" != */../* && "$path" != */.. && "$path" != *'/./'* && "$path" != *'/.' ]] || return 2
  rest="${path#/}"
  current="/"
  while [[ -n "$rest" ]]; do
    if [[ "$rest" == */* ]]; then component="${rest%%/*}"; rest="${rest#*/}"; else component="$rest"; rest=""; fi
    [[ -n "$component" ]] || continue
    current="$current$component"
    [[ ! -L "$current" ]] || return 2
    current="$current/"
  done
  [[ ! -e "$path" && ! -L "$path" ]] || return 2
  parent="$path"
  while [[ ! -e "$parent" ]]; do
    [[ ! -L "$parent" ]] || return 2
    item="${parent##*/}"
    [[ -n "$item" ]] || return 2
    suffix+=("$item")
    [[ "$parent" != / ]] || return 2
    parent="${parent%/*}"
    [[ -n "$parent" ]] || parent=/
  done
  [[ -d "$parent" && ! -L "$parent" ]] || return 2
  candidate="$(cd -P "$parent" && pwd)"
  for (( item = ${#suffix[@]} - 1; item >= 0; item-- )); do candidate="$candidate/${suffix[item]}"; done
  root_real="$(cd -P "$VOKRA_ROOT" 2>/dev/null && pwd)" || return 2
  [[ "$candidate" != "$root_real" && "$candidate/" != "$root_real/"* && "$root_real/" != "$candidate/"* ]] || return 2
  approval_parent="$(cd -P "$(dirname "$approval")" 2>/dev/null && pwd)" || return 2
  approval_real="$approval_parent/$(basename "$approval")"
  [[ "$candidate" != "$approval_real" && "$candidate/" != "$approval_real/"* && "$approval_real/" != "$candidate/"* ]] || return 2
}

usage() {
  cat <<'EOF'
usage: apple-silicon-clap-htsat-fused.sh --approval-evidence <file> --evidence-dir <absent-dir>
       apple-silicon-clap-htsat-fused.sh --self-test

Requires a disposable Darwin/arm64 host with VOKRA_REMOTE_APPLE_SILICON=1,
real Metal tooling, and a clean checkout. The current result is explicitly
INSPECTION_ONLY because the VAST tensor-name/shape manifest is not yet
reviewed; no model download, conversion, or publication is done here.
EOF
}

self_test() {
  local path="${BASH_SOURCE[0]}" fail=0 token tmp approval evidence rc
  for token in 'VOKRA_REMOTE_APPLE_SILICON=1' 'Darwin' 'arm64' \
    'xcrun -f metal' 'INSPECTION_ONLY' 'CLAP_METAL_VS_CPU' \
    'tensor-name/shape manifest' 'git status --porcelain'; do
    if ! grep -Fq -- "$token" "$path" && ! grep -Fq -- "$token" "$PARITY_SOURCE"; then
      log "self-test FAIL: missing contract token: $token"
      fail=1
    fi
  done
  if grep -En '(^|[[:space:]])(curl|wget|python3?|pip|.*convert|.*upload|.*publish|git[[:space:]]+push)([[:space:]]|$)' "$path" >/dev/null; then
    log 'self-test FAIL: acquisition or publication command found'
    fail=1
  fi
  if grep -En '^[[:space:]]*printf[^#]*CLAP_METAL_VS_CPU PASS' "$path" >/dev/null; then
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
  if "$path" --self-test --self-test >/dev/null 2>&1; then
    log 'self-test FAIL: duplicate --self-test accepted'
    fail=1
  fi
  tmp="$(mktemp -d)"
  tmp="$(cd -P "$tmp" && pwd)"
  CLAP_APPLE_SELF_TEST_TMP="$tmp"
  trap cleanup_self_test EXIT
  approval="$tmp/approval.json"
  evidence="$tmp/evidence/nested"
  printf '{}\n' >"$approval"
  validate_absent_evidence "$evidence" "$approval" || { log 'self-test FAIL: safe absent evidence path rejected'; fail=1; }
  set +e
  VOKRA_REMOTE_APPLE_SILICON=1 "$path" --approval-evidence "$approval" --evidence-dir "$evidence" >/dev/null 2>&1
  rc=$?
  set -e
  if [[ "$rc" != 2 || -e "$evidence" ]]; then
    log 'self-test FAIL: production-shaped missing-lock probe had effects or wrong status'
    fail=1
  fi
  (( fail == 0 )) || return 1
  log 'self-test PASS'
}

evidence_dir=''
approval_evidence=''
self=0
seen_self=0
seen_evidence=0
seen_approval=0
while (($#)); do
  case "$1" in
    --self-test) (( seen_self == 0 )) || die 'duplicate --self-test'; seen_self=1; self=1; shift ;;
    --evidence-dir) (( seen_evidence == 0 )) || die 'duplicate --evidence-dir'; (( $# >= 2 )) || die '--evidence-dir requires a path'; [[ -n "$2" && "$2" != -* ]] || die '--evidence-dir must be a nonempty path'; seen_evidence=1; evidence_dir="$2"; shift 2 ;;
    --approval-evidence) (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; (( $# >= 2 )) || die '--approval-evidence requires a file'; [[ -n "$2" && "$2" != -* ]] || die '--approval-evidence must be a nonempty file path'; seen_approval=1; approval_evidence="$2"; shift 2 ;;
    -h|--help) [[ $self == 0 && $# == 1 ]] || die '--help cannot be combined with other arguments'; usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if (( self )); then
  [[ "$seen_evidence" == 0 && "$seen_approval" == 0 ]] || die '--self-test accepts no other arguments'
  self_test
  exit $?
fi

[[ -n "$approval_evidence" ]] || die '--approval-evidence is required'
[[ -n "$evidence_dir" ]] || die '--evidence-dir is required'
require_preflight "$DEDICATED_PROJECT" "$approval_evidence" || die 'CLAP dedicated lock/license/approval gate is unresolved; refuse before host/input/evidence'
[[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == 1 ]] || die 'VOKRA_REMOTE_APPLE_SILICON=1 is absent'
[[ "$(uname -s)" == Darwin ]] || die 'real Metal inspection requires Darwin'
[[ "$(uname -m)" == arm64 ]] || die 'real Metal inspection requires Apple arm64'
command -v xcrun >/dev/null 2>&1 || die 'xcrun is unavailable'
xcrun -f metal >/dev/null 2>&1 || die 'Xcode Metal compiler is unavailable'
[[ -f "$VOKRA_ROOT/Cargo.toml" && -d "$VOKRA_ROOT/.git" ]] || die 'not a Vokra checkout'
[[ -f "$PARITY_SOURCE" ]] || die 'CLAP parity gate source is missing'
[[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'Apple checkout must be clean'
validate_absent_evidence "$evidence_dir" "$approval_evidence" || die 'evidence directory must be absent, disjoint, and free of symlink ancestors'
mkdir -p "$evidence_dir"

{
  echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
  echo "host=$(uname -a)"
  echo "metal_compiler=$(xcrun -f metal)"
  echo 'runtime_status=INSPECTION_ONLY'
  echo 'parity_status=INSPECTION_ONLY'
  echo 'verdict=NO_CPU_OR_METAL_PASS'
  echo 'reason=VAST tensor-name/shape manifest and checkpoint identity are not audited'
  echo 'hardware_probe_is_not_clap_parity_evidence=true'
} > "$evidence_dir/clap-htsat-fused-apple-inspection.txt"
log "recorded inspection-only evidence at $evidence_dir; no PASS marker was emitted"
