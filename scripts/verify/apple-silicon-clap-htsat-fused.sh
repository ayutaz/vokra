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

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

require_clean_expected_head() {
  local expected="$1" actual
  [[ "$expected" =~ ^[0-9a-f]{40}$ ]] || { log 'expected HEAD must be exactly 40 lowercase hexadecimal characters'; return 2; }
  [[ -d "$VOKRA_ROOT/.git" ]] || { log 'checkout is missing .git'; return 2; }
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || { log 'checkout must be clean'; return 2; }
  actual="$(git -C "$VOKRA_ROOT" rev-parse HEAD)" || return 2
  [[ "$actual" == "$expected" ]] || { log "checkout HEAD $actual differs from expected $expected"; return 2; }
}

require_regular_approval_path() {
  local input="$1" path="$1" rest component current base
  [[ -n "$path" && "$path" != *$'\n'* && "$path" != *$'\r'* ]] || return 2
  if [[ "$path" != /* ]]; then
    base="$(pwd -P)" || return 2
    path="$base/$path"
  fi
  [[ "$path" != */../* && "$path" != */.. && "$path" != *'/./'* && "$path" != *'/.' ]] || return 2
  rest="${path#/}"
  current="/"
  while [[ -n "$rest" ]]; do
    if [[ "$rest" == */* ]]; then component="${rest%%/*}"; rest="${rest#*/}"; else component="$rest"; rest=""; fi
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || return 2
    current="$current$component"
    [[ ! -L "$current" ]] || return 2
    current="$current/"
  done
  [[ -f "$input" && ! -L "$input" ]] || return 2
}

require_approval_binding() {
  local approval="$1" expected_sha="$2"
  [[ "$expected_sha" =~ ^[0-9a-f]{64}$ ]] || { log 'approval SHA must be exactly 64 lowercase hexadecimal characters'; return 2; }
  require_regular_approval_path "$approval" || { log 'approval evidence must be a regular file with safe non-symlink ancestry'; return 2; }
  [[ "$(sha256_file "$approval")" == "$expected_sha" ]] || { log 'approval evidence SHA-256 differs from caller binding'; return 2; }
}

claim_absent_directory() {
  local path="$1"
  [[ ! -e "$path" && ! -L "$path" ]] || return 2
  mkdir "$path" || return 2
  [[ -d "$path" && ! -L "$path" ]] || return 2
}

require_preflight() {
  # No authenticated CLAP reference project, lock, license gate, or native
  # binder exists in this source tree. Approval is only a caller-bound hash
  # for this blocked disposition; it can never authorize acquisition.
  log 'BLOCKED_MISSING_AUTHENTICATED_REFERENCE_LOCK_LICENSE_GATE/NO_UPLOAD'
  return 2
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
usage: apple-silicon-clap-htsat-fused.sh --approval-evidence <file> --approval-sha256 <64-hex> --expected-head <40-hex> --evidence-dir <absent-dir>
       apple-silicon-clap-htsat-fused.sh --self-test

Requires a disposable Darwin/arm64 host with VOKRA_REMOTE_APPLE_SILICON=1,
real Metal tooling, and a clean checkout. The current result is explicitly
INSPECTION_ONLY because the VAST tensor-name/shape manifest is not yet
reviewed; no model download, conversion, or publication is done here.
EOF
}

self_test() {
  local path="${BASH_SOURCE[0]}" fail=0 token tmp approval approval_sha evidence rc
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
  if "$path" --approval-evidence /tmp/a --approval-sha256 "$(printf '0%.0s' {1..64})" --evidence-dir /tmp/e >/dev/null 2>&1; then
    log 'self-test FAIL: missing --expected-head accepted'
    fail=1
  fi
  if "$path" --approval-evidence /tmp/a --approval-sha256 "$(printf '0%.0s' {1..64})" --expected-head "$(printf '0%.0s' {1..40})" --expected-head "$(printf '1%.0s' {1..40})" --evidence-dir /tmp/e >/dev/null 2>&1; then
    log 'self-test FAIL: duplicate --expected-head accepted'
    fail=1
  fi
  if "$path" --approval-evidence /tmp/a --approval-sha256 "$(printf '0%.0s' {1..64})" --approval-sha256 "$(printf '1%.0s' {1..64})" --expected-head "$(printf '0%.0s' {1..40})" --evidence-dir /tmp/e >/dev/null 2>&1; then
    log 'self-test FAIL: duplicate --approval-sha256 accepted'
    fail=1
  fi
  tmp="$(mktemp -d)"
  tmp="$(cd -P "$tmp" && pwd)"
  CLAP_APPLE_SELF_TEST_TMP="$tmp"
  trap cleanup_self_test EXIT
  approval="$tmp/approval.json"
  printf '{}\n' >"$approval"
  approval_sha="$(sha256_file "$approval")"
  evidence="$tmp/evidence"
  validate_absent_evidence "$evidence" "$approval" || { log 'self-test FAIL: safe absent evidence path rejected'; fail=1; }
  if validate_absent_evidence "$tmp/../escape" "$approval" >/dev/null 2>&1; then
    log 'self-test FAIL: dot-dot evidence path accepted'; fail=1
  fi
  mkdir "$tmp/real"
  ln -s "$tmp/real" "$tmp/link"
  if validate_absent_evidence "$tmp/link/evidence" "$approval" >/dev/null 2>&1; then
    log 'self-test FAIL: symlink-ancestor evidence path accepted'; fail=1
  fi
  if require_approval_binding "$approval" "$(printf '0%.0s' {1..64})" >/dev/null 2>&1; then
    log 'self-test FAIL: wrong approval SHA accepted'; fail=1
  fi
  require_approval_binding "$approval" "$approval_sha" || { log 'self-test FAIL: correct approval SHA rejected'; fail=1; }
  mkdir "$tmp/approval-real"
  ln -s "$tmp/approval-real" "$tmp/approval-link"
  printf '{}\n' >"$tmp/approval-real/evidence.json"
  if require_approval_binding "$tmp/approval-link/evidence.json" "$approval_sha" >/dev/null 2>&1; then
    log 'self-test FAIL: symlink-ancestor approval path accepted'; fail=1
  fi
  mkdir "$tmp/project"
  : >"$tmp/project/pyproject.toml"
  : >"$tmp/project/uv.lock"
  : >"$tmp/project/license_gate.py"
  : >"$tmp/project/license_gate_manifest.json"
  if require_preflight "$tmp/project" "$approval" >/dev/null 2>&1; then
    log 'self-test FAIL: placeholder dedicated lock/gate overrode blocked disposition'; fail=1
  fi
  set +e
  VOKRA_REMOTE_APPLE_SILICON=1 "$path" --approval-evidence "$approval" --approval-sha256 "$approval_sha" \
    --expected-head "$(printf '0%.0s' {1..40})" --evidence-dir "$evidence" >/dev/null 2>&1
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
approval_sha256=''
expected_head=''
self=0
seen_self=0
seen_evidence=0
seen_approval=0
seen_approval_sha=0
seen_head=0
while (($#)); do
  case "$1" in
    --self-test) (( seen_self == 0 )) || die 'duplicate --self-test'; seen_self=1; self=1; shift ;;
    --evidence-dir) (( seen_evidence == 0 )) || die 'duplicate --evidence-dir'; (( $# >= 2 )) || die '--evidence-dir requires a path'; [[ -n "$2" && "$2" != -* ]] || die '--evidence-dir must be a nonempty path'; seen_evidence=1; evidence_dir="$2"; shift 2 ;;
    --approval-evidence) (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; (( $# >= 2 )) || die '--approval-evidence requires a file'; [[ -n "$2" && "$2" != -* ]] || die '--approval-evidence must be a nonempty file path'; seen_approval=1; approval_evidence="$2"; shift 2 ;;
    --approval-sha256) (( seen_approval_sha == 0 )) || die 'duplicate --approval-sha256'; (( $# >= 2 )) || die '--approval-sha256 requires a SHA'; [[ "$2" =~ ^[0-9a-f]{64}$ ]] || die '--approval-sha256 requires lowercase 64-hex'; seen_approval_sha=1; approval_sha256="$2"; shift 2 ;;
    --expected-head) (( seen_head == 0 )) || die 'duplicate --expected-head'; (( $# >= 2 )) || die '--expected-head requires a commit'; [[ "$2" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head requires lowercase 40-hex'; seen_head=1; expected_head="$2"; shift 2 ;;
    -h|--help) [[ $self == 0 && $# == 1 ]] || die '--help cannot be combined with other arguments'; usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if (( self )); then
  [[ "$seen_evidence" == 0 && "$seen_approval" == 0 && "$seen_approval_sha" == 0 && "$seen_head" == 0 ]] || die '--self-test accepts no other arguments'
  self_test
  exit $?
fi

[[ -n "$approval_evidence" ]] || die '--approval-evidence is required'
[[ "$seen_approval_sha" == 1 ]] || die '--approval-sha256 is required'
[[ "$seen_head" == 1 ]] || die '--expected-head is required'
[[ -n "$evidence_dir" ]] || die '--evidence-dir is required'
command -v shasum >/dev/null 2>&1 || die 'shasum is required for caller binding'
require_clean_expected_head "$expected_head" || die 'checkout is not the caller-bound clean expected HEAD'
require_approval_binding "$approval_evidence" "$approval_sha256" || die 'approval evidence caller binding is invalid'
require_preflight "$DEDICATED_PROJECT" "$approval_evidence" || die 'CLAP dedicated lock/license/approval gate is unresolved; refuse before host/input/evidence'
[[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == 1 ]] || die 'VOKRA_REMOTE_APPLE_SILICON=1 is absent'
[[ "$(uname -s)" == Darwin ]] || die 'real Metal inspection requires Darwin'
[[ "$(uname -m)" == arm64 ]] || die 'real Metal inspection requires Apple arm64'
command -v xcrun >/dev/null 2>&1 || die 'xcrun is unavailable'
xcrun -f metal >/dev/null 2>&1 || die 'Xcode Metal compiler is unavailable'
[[ -f "$VOKRA_ROOT/Cargo.toml" && -d "$VOKRA_ROOT/.git" ]] || die 'not a Vokra checkout'
[[ -f "$PARITY_SOURCE" ]] || die 'CLAP parity gate source is missing'
[[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'Apple checkout must be clean'
require_clean_expected_head "$expected_head" || die 'checkout HEAD/clean state changed before final inspection evidence'
validate_absent_evidence "$evidence_dir" "$approval_evidence" || die 'evidence directory must be absent, disjoint, and free of symlink ancestors'
claim_absent_directory "$evidence_dir" || die 'evidence directory could not be atomically claimed as absent'

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
