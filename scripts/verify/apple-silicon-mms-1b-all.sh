#!/usr/bin/env bash
# Apple Silicon inspection gate for MMS 1B all.
# The native MMS binder is intentionally fail-closed until VAST supplies an
# audited backbone+adapter manifest, so hardware discovery is not parity.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PARITY_SOURCE="$VOKRA_ROOT/crates/vokra-models/tests/parity_mms_1b_all_real.rs"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/mms_1b_all"
PREFLIGHT_GATE="$PARITY_PROJECT/license_gate.py"
PREFLIGHT_MANIFEST="$PARITY_PROJECT/license_gate_manifest.json"

log() { printf '[mms-1b-all-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF'
usage: apple-silicon-mms-1b-all.sh --language <official-code> --approval-evidence <file> --evidence-dir <absent-dir>
       apple-silicon-mms-1b-all.sh --self-test

Requires a disposable Darwin/arm64 host with VOKRA_REMOTE_APPLE_SILICON=1
and real Metal tooling. The current result is INSPECTION_ONLY because the
pinned MMS checkpoint manifest and native route are not reviewed. A hardware
probe is not CPU/Metal parity evidence; no model download or publication is
performed.
EOF
}

license_preflight() {
  local language="$1" approval="$2"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$PREFLIGHT_GATE" \
    --lock "$PARITY_PROJECT/uv.lock" --project "$PARITY_PROJECT/pyproject.toml" \
    --manifest "$PREFLIGHT_MANIFEST" --approval-evidence "$approval" --language "$language" \
    || die 'dedicated MMS closure/license/approval gate is unresolved'
}

canonical_absent_path() {
  local path="$1" suffix='' rest component scan name parent
  [[ "$path" == /* ]] || path="$PWD/$path"; rest="${path#/}"; scan=''
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; rest="${rest#*/}"; [[ "$component" == "$rest" ]] && rest=''
    [[ -n "$component" && "$component" != . ]] || continue
    [[ "$component" != .. ]] || return 1
    scan="$scan/$component"; [[ ! -L "$scan" ]] || return 1
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
  # shellcheck disable=SC2016 # literal source contract tokens
  for token in 'VOKRA_REMOTE_APPLE_SILICON=1' 'Darwin' 'arm64' \
    'xcrun -f metal' 'INSPECTION_ONLY' 'hardware_probe_is_not_parity_evidence' \
    'backbone+adapter manifest' 'git status --porcelain' 'parity_mms_1b_all_real.rs' \
    'tools/parity/mms_1b_all/license_gate.py' \
    '--language "$language"'; do
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
  local gate_line path_line host_line
  # shellcheck disable=SC2016 # match literal source token
  gate_line="$(grep -n 'license_preflight "\$language" "\$approval_evidence"' "$path" | tail -n 1 | cut -d: -f1)"
  # shellcheck disable=SC2016 # match literal source token
  path_line="$(grep -n 'require_absent_evidence "\$evidence_dir"' "$path" | tail -n 1 | cut -d: -f1)"
  host_line="$(grep -n 'uname -s' "$path" | tail -n 1 | cut -d: -f1)"
  [[ "$gate_line" =~ ^[0-9]+$ && "$path_line" =~ ^[0-9]+$ && "$host_line" =~ ^[0-9]+$ && "$gate_line" -lt "$path_line" && "$path_line" -lt "$host_line" ]] || {
    log 'self-test FAIL: closure/path gate is not before host probe'
    fail=1
  }
  (( fail == 0 )) || return 1
  log 'self-test PASS'
}

evidence_dir=''; approval_evidence=''; language=''; self=0
seen_self=0; seen_evidence=0; seen_approval=0; seen_language=0
while (($#)); do
  case "$1" in
    --self-test) (( seen_self == 0 )) || die 'duplicate --self-test'; seen_self=1; self=1; shift ;;
    --language) (( seen_language == 0 )) || die 'duplicate --language'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--language requires a nonempty official adapter code'; seen_language=1; language="$2"; shift 2 ;;
    --approval-evidence) (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--approval-evidence requires a nonempty path'; seen_approval=1; approval_evidence="$2"; shift 2 ;;
    --evidence-dir) (( seen_evidence == 0 )) || die 'duplicate --evidence-dir'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--evidence-dir requires a nonempty path'; seen_evidence=1; evidence_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if (( self )); then
  [[ -z "$evidence_dir$approval_evidence$language" ]] || die '--self-test accepts no other arguments'
  self_test
  exit $?
fi

[[ "$seen_language" == 1 && "$seen_approval" == 1 && "$seen_evidence" == 1 ]] || die '--language, --approval-evidence, and --evidence-dir are required'
[[ -n "$evidence_dir" ]] || die '--evidence-dir is required'
[[ "$language" =~ ^[a-z0-9]+([_-][a-z0-9]+)*$ ]] || die '--language contains unsafe filename characters'
license_preflight "$language" "$approval_evidence"
require_absent_evidence "$evidence_dir" "$approval_evidence"
[[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == 1 ]] || die 'VOKRA_REMOTE_APPLE_SILICON=1 is absent'
[[ "$(uname -s)" == Darwin ]] || die 'real Metal inspection requires Darwin'
[[ "$(uname -m)" == arm64 ]] || die 'real Metal inspection requires Apple arm64'
command -v xcrun >/dev/null 2>&1 || die 'xcrun is unavailable'
xcrun -f metal >/dev/null 2>&1 || die 'Xcode Metal compiler is unavailable'
[[ -f "$VOKRA_ROOT/Cargo.toml" && -d "$VOKRA_ROOT/.git" ]] || die 'not a Vokra checkout'
[[ -f "$PARITY_SOURCE" ]] || die 'MMS parity gate source is missing'
[[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'Apple checkout must be clean'
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
