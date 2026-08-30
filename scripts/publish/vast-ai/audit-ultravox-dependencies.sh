#!/usr/bin/env bash
# Audit the exact Ultravox Python closure on an already-frozen VAST host.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/ultravox"
AUDIT="$PARITY_PROJECT/dependency_audit.py"
OUTPUT=''
MIN_VAST_MEM_KIB=60000000

usage() {
  cat <<'EOF' >&2
usage: audit-ultravox-dependencies.sh --output <audit.json>
       audit-ultravox-dependencies.sh --self-test

The target is an already synchronized, separately authorized named VAST job.
This wrapper never runs uv sync, imports model/Torch code, invokes Cargo, or
downloads weights. It may fetch only exact locked PyPI sdists for missing
publisher evidence and the fixed source/model/Meta companion LICENSE paths.
EOF
}

canonicalize_uncreated() {
  local path="$1" suffix='' name parent component rest scan
  [[ "$path" == /* ]] || path="$PWD/$path"
  rest="${path#/}"; scan=''
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; rest="${rest#*/}"
    [[ "$component" == "$rest" ]] && rest=''
    [[ "$component" == .. ]] && return 1
    [[ -n "$component" && "$component" != . ]] || continue
    scan="$scan/$component"
    [[ ! -L "$scan" ]] || return 1
  done
  while [[ ! -d "$path" || -L "$path" ]]; do
    name="${path##*/}"; [[ -n "$name" ]] && suffix="/$name$suffix"
    parent="${path%/*}"; [[ "$parent" == "$path" ]] && parent='.'
    [[ -n "$parent" ]] || parent='/'; path="$parent"
    [[ ! -L "$path" ]] || return 1
  done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}

paths_overlap() { [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]; }

require_vast_linux() {
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || { echo 'ultravox dependency audit: VOKRA_PUBLISH_ON_VAST=1 is required' >&2; return 2; }
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'ultravox dependency audit: Linux x86_64 VAST host required' >&2; return 2; }
  local memory
  memory="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$memory" =~ ^[0-9]+$ && memory -ge MIN_VAST_MEM_KIB ]] || { echo 'ultravox dependency audit: 60-GB-class VAST memory required' >&2; return 2; }
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
  local output="$1" input root
  require_vast_linux
  [[ -d "$VOKRA_ROOT/.git" || -f "$VOKRA_ROOT/.git" ]] || return 2
  root="$(git -C "$VOKRA_ROOT" rev-parse --show-toplevel 2>/dev/null)" || return 2
  [[ "$(canonicalize_uncreated "$VOKRA_ROOT")" == "$(canonicalize_uncreated "$root")" ]] || return 2
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || { echo 'ultravox dependency audit: checkout must be clean' >&2; return 2; }
  command -v uv >/dev/null 2>&1 || return 2
  command -v readelf >/dev/null 2>&1 || return 2
  for input in pyproject.toml uv.lock license_gate_manifest.json dependency_audit.py; do
    [[ -f "$PARITY_PROJECT/$input" && ! -L "$PARITY_PROJECT/$input" ]] || { echo "ultravox dependency audit: missing/symlinked input: $input" >&2; return 2; }
  done
  require_absent_output "$output"
  mkdir -p "$(dirname "$output")"
  UV_NO_CACHE=1 uv run --no-cache --project "$PARITY_PROJECT" --frozen --no-sync --python 3.12 \
    python "$AUDIT" --project "$PARITY_PROJECT" --output "$output" --fetch-model-licenses
}

self_test() {
  local failed=0 probe_root probe_parent=/tmp
  [[ -d /private/tmp && ! -L /private/tmp ]] && probe_parent=/private/tmp
  grep -Fq -- '--no-sync' "$0" || failed=1
  grep -Fq -- 'exact locked PyPI sdist' "$0" || failed=1
  grep -Fq -- 'fixed source/model/Meta companion LICENSE paths' "$0" || failed=1
  ! grep -Eq '^[[:space:]]*(uv[[:space:]]+sync|snapshot_download|huggingface-cli|cargo[[:space:]]+(build|test|check|clippy))([[:space:]]|$)' "$0" || failed=1
  if VOKRA_PUBLISH_ON_VAST=0 run_audit /private/tmp/ultravox-audit-self-test.json >/dev/null 2>&1; then failed=1; fi
  probe_root="$(mktemp -d "$probe_parent/ultravox-audit-wrapper.XXXXXX")"
  if require_absent_output "$VOKRA_ROOT" >/dev/null 2>&1; then failed=1; fi
  if require_absent_output "$PARITY_PROJECT" >/dev/null 2>&1; then failed=1; fi
  require_absent_output "$probe_root/nested/audit.json" || failed=1
  [[ ! -e "$probe_root/nested" ]] || failed=1
  if require_absent_output "$probe_root/../escaped.json" >/dev/null 2>&1; then failed=1; fi
  mkdir -p "$probe_root/real-parent"
  ln -s "$probe_root/real-parent" "$probe_root/link-parent"
  if require_absent_output "$probe_root/link-parent/audit.json" >/dev/null 2>&1; then failed=1; fi
  touch "$probe_root/existing.json"
  if require_absent_output "$probe_root/existing.json" >/dev/null 2>&1; then failed=1; fi
  rm -rf "$probe_root"
  (cd "$PARITY_PROJECT" && UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$AUDIT" --self-test) || failed=1
  (( failed == 0 )) || { echo 'audit-ultravox-dependencies.sh self-test FAIL' >&2; return 1; }
  echo 'audit-ultravox-dependencies.sh self-test PASS'
}

while (( $# > 0 )); do
  case "$1" in
    --output) [[ $# -ge 2 && -n "$2" && "$2" != -* && -z "$OUTPUT" ]] || { usage; exit 2; }; OUTPUT="$2"; shift 2 ;;
    --self-test) [[ -z "$OUTPUT" ]] || { usage; exit 2; }; OUTPUT=self-test; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

if [[ "$OUTPUT" == self-test ]]; then self_test; else [[ -n "$OUTPUT" ]] || { usage; exit 2; }; run_audit "$OUTPUT"; fi
