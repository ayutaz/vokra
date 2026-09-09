#!/usr/bin/env bash
# Collect exact installed YuE xcodec-mini dependency evidence on VAST.
# This worker never acquires/imports/executes model or checkpoint weights.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/yue_xcodec_mini"
AUDIT="$PARITY_PROJECT/dependency_audit.py"
OUTPUT=""
EXPECTED_HEAD=""
SELF_TEST=0
MIN_VAST_MEM_KIB=60000000
MIN_FREE_DISK_KIB=30000000

usage() {
  cat <<'EOF' >&2
usage: audit-yue-xcodec-mini-dependencies.sh --output <audit.json> --expected-head <40-lowercase-hex>
       audit-yue-xcodec-mini-dependencies.sh --self-test

The production path synchronizes only the frozen Python dependency closure,
then records importlib.metadata/license/native payload facts. It performs no
model download, model import, model execution, Cargo operation, upload, or
publication. The report is expected to remain BLOCKED_OWNER_REVIEW until the
owner reviews the pending rows and component boundaries.
EOF
}

canonicalize_uncreated() {
  local path="$1" suffix='' name parent component rest scan=''
  [[ "$path" == /* ]] || path="$PWD/$path"
  rest="${path#/}"
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; rest="${rest#*/}"
    [[ "$component" == "$rest" ]] && rest=''
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || return 1
    scan="$scan/$component"
    [[ ! -L "$scan" ]] || return 1
  done
  while [[ ! -d "$path" || -L "$path" ]]; do
    name="${path##*/}"; [[ -n "$name" ]] && suffix="/$name$suffix"
    parent="${path%/*}"; [[ "$parent" == "$path" ]] && parent='/'; path="$parent"
    [[ ! -L "$path" ]] || return 1
  done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}

paths_overlap() { [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]; }

require_expected_head() {
  [[ "$1" =~ ^[0-9a-f]{40}$ ]] || { echo 'YuE dependency audit: expected-head must be 40 lowercase hexadecimal characters' >&2; return 2; }
}

head_matches_expected() {
  local expected="$1" actual
  require_expected_head "$expected" || return 2
  actual="$(git -C "$VOKRA_ROOT" rev-parse HEAD 2>/dev/null)" || return 2
  [[ "$actual" == "$expected" ]] || { echo "YuE dependency audit: expected HEAD $expected but checkout is $actual" >&2; return 2; }
}

require_vast_linux() {
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || { echo 'YuE dependency audit: VOKRA_PUBLISH_ON_VAST=1 is required' >&2; return 2; }
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'YuE dependency audit: Linux x86_64 VAST is required' >&2; return 2; }
  local memory; memory="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$memory" =~ ^[0-9]+$ && "$memory" -ge "$MIN_VAST_MEM_KIB" ]] || { echo 'YuE dependency audit: 60-GB-class VAST memory is required' >&2; return 2; }
  local free; free="$(df -Pk "$VOKRA_ROOT" | awk 'NR == 2 {print $4}')"
  [[ "$free" =~ ^[0-9]+$ && "$free" -ge "$MIN_FREE_DISK_KIB" ]] || { echo 'YuE dependency audit: free disk is below 30 GB' >&2; return 2; }
}

require_clean_checkout() {
  local expected_head="$1"
  [[ -d "$VOKRA_ROOT/.git" || -f "$VOKRA_ROOT/.git" ]] || return 2
  head_matches_expected "$expected_head" || return 2
  local root; root="$(git -C "$VOKRA_ROOT" rev-parse --show-toplevel 2>/dev/null)" || return 2
  [[ "$(canonicalize_uncreated "$VOKRA_ROOT")" == "$(canonicalize_uncreated "$root")" ]] || return 2
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || { echo 'YuE dependency audit: checkout must be clean' >&2; return 2; }
}

require_absent_output() {
  local output="$1" canonical root project
  [[ "$output" == /* && ! -e "$output" && ! -L "$output" ]] || return 2
  canonical="$(canonicalize_uncreated "$output")" || return 2
  root="$(canonicalize_uncreated "$VOKRA_ROOT")" || return 2
  project="$(canonicalize_uncreated "$PARITY_PROJECT")" || return 2
  paths_overlap "$canonical" "$root" && return 2
  paths_overlap "$canonical" "$project" && return 2
  return 0
}

run_audit() {
  local output="$1" expected_head="$2" rc input
  require_vast_linux || return 2
  require_expected_head "$expected_head" || return 2
  require_clean_checkout "$expected_head" || return 2
  for input in pyproject.toml uv.lock license_gate_manifest.json preflight_gate.py dependency_audit.py; do
    [[ -f "$PARITY_PROJECT/$input" && ! -L "$PARITY_PROJECT/$input" ]] || { echo "YuE dependency audit: missing input $input" >&2; return 2; }
  done
  command -v uv >/dev/null 2>&1 || return 2
  command -v readelf >/dev/null 2>&1 || return 2
  command -v git >/dev/null 2>&1 || return 2
  require_absent_output "$output" || return 2
  require_absent_output "$output.sha256" || return 2
  mkdir -p "$(dirname "$output")"
  local cache_dir="${YUE_AUDIT_UV_CACHE_DIR:-/tmp/vokra-yue-xcodec-audit-uv-cache}"
  UV_NO_CACHE=1 UV_CACHE_DIR="$cache_dir" uv sync --project "$PARITY_PROJECT" --frozen --python 3.12 --no-install-project
  set +e
  UV_NO_CACHE=1 UV_CACHE_DIR="$cache_dir" uv run --project "$PARITY_PROJECT" --frozen --no-sync --python 3.12 python "$AUDIT" --project "$PARITY_PROJECT" --repo-root "$VOKRA_ROOT" --expected-head "$expected_head" --output "$output"
  rc=$?
  set -e
  [[ -f "$output" && -f "$output.sha256" ]] || { echo 'YuE dependency audit: report or sidecar was not atomically created' >&2; return 2; }
  return "$rc"
}

self_test() {
  local failed=0 tmp parent=/tmp
  [[ -d /private/tmp && ! -L /private/tmp ]] && parent=/private/tmp
  grep -Fq -- 'no model download' "$0" || failed=1
  grep -Fq -- '--no-install-project' "$0" || failed=1
  grep -Fq -- '--expected-head' "$0" || failed=1
  ! grep -Eq '^[[:space:]]*(snapshot_download|huggingface-cli|cargo[[:space:]]+(build|test|check|clippy)|python3|python|pip)([[:space:]]|$)' "$0" || failed=1
  tmp="$(mktemp -d "$parent/yue-xcodec-audit-wrapper.XXXXXX")"
  trap 'rm -rf "$tmp"' EXIT
  local actual_head; actual_head="$(git -C "$VOKRA_ROOT" rev-parse HEAD 2>/dev/null)" || failed=1
  if head_matches_expected "$(printf '%040d' 0)" >/dev/null 2>&1; then failed=1; fi
  if ! head_matches_expected "$actual_head" >/dev/null 2>&1; then failed=1; fi
  if VOKRA_PUBLISH_ON_VAST=0 run_audit "$tmp/blocked.json" "$(printf '%040d' 1)" >/dev/null 2>&1; then failed=1; fi
  [[ ! -e "$tmp/blocked.json" ]] || failed=1
  require_absent_output "$VOKRA_ROOT" >/dev/null 2>&1 && failed=1
  require_absent_output "$PARITY_PROJECT" >/dev/null 2>&1 && failed=1
  require_absent_output "$tmp/nested/audit.json" || failed=1
  [[ ! -e "$tmp/nested" ]] || failed=1
  require_absent_output "$tmp/./dot.json" >/dev/null 2>&1 && failed=1
  require_absent_output "$tmp/../dot-dot.json" >/dev/null 2>&1 && failed=1
  touch "$tmp/existing.json"
  require_absent_output "$tmp/existing.json" >/dev/null 2>&1 && failed=1
  touch "$tmp/existing.json.sha256"
  require_absent_output "$tmp/existing.json.sha256" >/dev/null 2>&1 && failed=1
  ln -s "$tmp" "$tmp/link"
  require_absent_output "$tmp/link/new.json" >/dev/null 2>&1 && failed=1
  rm -rf "$tmp"
  trap - EXIT
  UV_NO_CACHE=1 UV_CACHE_DIR="$parent/yue-audit-selftest-cache" uv run --no-cache --no-project --offline --python 3.12 python "$AUDIT" --self-test >/dev/null 2>&1 || failed=1
  (( failed == 0 )) || { echo 'audit-yue-xcodec-mini-dependencies.sh self-test FAIL' >&2; return 1; }
  echo 'audit-yue-xcodec-mini-dependencies.sh self-test: PASS (model-free, NO_UPLOAD)'
}

main() {
  local self=0
  while (( $# > 0 )); do
    case "$1" in
      --output) [[ $# -ge 2 && -z "$OUTPUT" && "$2" != -* ]] || { usage; return 2; }; OUTPUT="$2"; shift 2 ;;
      --expected-head) [[ $# -ge 2 && -z "$EXPECTED_HEAD" && "$2" != -* ]] || { usage; return 2; }; EXPECTED_HEAD="$2"; shift 2 ;;
      --self-test) (( self == 0 )) || { usage; return 2; }; self=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) usage; return 2 ;;
    esac
  done
  if (( self )); then [[ -z "$OUTPUT$EXPECTED_HEAD" ]] || { usage; return 2; }; self_test; return $?; fi
  [[ -n "$OUTPUT" && -n "$EXPECTED_HEAD" && "$EXPECTED_HEAD" =~ ^[0-9a-f]{40}$ ]] || { usage; return 2; }
  run_audit "$OUTPUT" "$EXPECTED_HEAD"
}

main "$@"
