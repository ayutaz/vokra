#!/usr/bin/env bash
# Audit the exact NeuTTS Air Python closure on an authorized VAST/Linux host.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/neutts_air"
AUDIT="$PARITY_PROJECT/dependency_audit.py"
MIN_VAST_MEM_KIB=48000000

usage() {
  cat <<'EOF' >&2
usage: audit-neutts-air-dependencies.sh --output <audit.json>
       audit-neutts-air-dependencies.sh --self-test

VAST/Linux-only post-sync audit. The environment must have been synchronized
by a separately authorized exact frozen setup step. This wrapper never runs
uv sync, never downloads weights, never imports model code, and never runs
Cargo. It fetches only exact locked PyPI sdists for missing publisher files
and the four fixed LICENSE paths in the NeuTTS Air contract.
EOF
}

canonical_uncreated() {
  local path="$1" rest component scan suffix parent
  [[ "$path" == /* ]] || path="$PWD/$path"
  rest="${path#/}"; scan=""
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; rest="${rest#*/}"
    [[ "$component" == "$rest" ]] && rest=''
    [[ "$component" != .. ]] || return 1
    [[ -n "$component" && "$component" != . ]] || continue
    scan="$scan/$component"
    [[ ! -L "$scan" ]] || return 1
  done
  suffix=''
  while [[ ! -d "$path" || -L "$path" ]]; do
    component="${path##*/}"; [[ -n "$component" ]] && suffix="/$component$suffix"
    parent="${path%/*}"; [[ "$parent" == "$path" ]] && parent='.'
    path="$parent"
    [[ ! -L "$path" ]] || return 1
  done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}

paths_overlap() { [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]; }

require_vast_linux() {
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] || return 2
  [[ "$(uname -s)" == "Linux" ]] || return 2
  [[ -r /proc/meminfo ]] || return 2
  local memory
  memory="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$memory" =~ ^[0-9]+$ && memory -ge MIN_VAST_MEM_KIB ]] || return 2
}

require_clean_checkout() {
  local root
  root="$(git -C "$VOKRA_ROOT" rev-parse --show-toplevel 2>/dev/null)" || return 2
  [[ "$(canonical_uncreated "$VOKRA_ROOT")" == "$(canonical_uncreated "$root")" ]] || return 2
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || return 2
}

require_absent_output() {
  local output="$1" canonical root project
  [[ "$output" == /* && ! -e "$output" && ! -L "$output" ]] || return 2
  canonical="$(canonical_uncreated "$output")" || return 2
  root="$(canonical_uncreated "$VOKRA_ROOT")" || return 2
  project="$(canonical_uncreated "$PARITY_PROJECT")" || return 2
  paths_overlap "$canonical" "$root" && return 2
  paths_overlap "$canonical" "$project" && return 2
  return 0
}

run_audit() {
  local output="$1"
  require_vast_linux || return 2
  require_clean_checkout || return 2
  command -v uv >/dev/null 2>&1 || return 2
  command -v readelf >/dev/null 2>&1 || return 2
  [[ -f "$PARITY_PROJECT/pyproject.toml" && ! -L "$PARITY_PROJECT/pyproject.toml" ]] || return 2
  [[ -f "$PARITY_PROJECT/uv.lock" && ! -L "$PARITY_PROJECT/uv.lock" ]] || return 2
  [[ -f "$AUDIT" && ! -L "$AUDIT" ]] || return 2
  require_absent_output "$output" || return 2
  mkdir -p "$(dirname "$output")" || return 2
  UV_NO_CACHE=1 uv run --no-cache --project "$PARITY_PROJECT" --frozen --no-sync --python 3.12 \
    python "$AUDIT" --project "$PARITY_PROJECT" --output "$output" --fetch-model-licenses
}

self_test() {
  local failed=0 probe blocked_parent blocked_output temp_parent=/tmp
  [[ -d /private/tmp && ! -L /private/tmp ]] && temp_parent=/private/tmp
  probe="$(mktemp -d "$temp_parent/neutts-air-audit-wrapper.XXXXXX")"
  blocked_parent="$probe/blocked-parent"
  blocked_output="$blocked_parent/audit.json"
  mkdir "$probe/bin"
  printf '%s\n' '#!/usr/bin/env bash' "touch '$probe/auditor-invoked'" > "$probe/bin/uv"
  chmod +x "$probe/bin/uv"
  if PATH="$probe/bin:$PATH" VOKRA_PUBLISH_ON_VAST=0 run_audit "$blocked_output" >/dev/null 2>&1; then failed=1; fi
  [[ ! -e "$probe/auditor-invoked" ]] || failed=1
  [[ ! -e "$blocked_parent" && ! -e "$blocked_output" ]] || failed=1
  if require_absent_output "$VOKRA_ROOT" >/dev/null 2>&1; then failed=1; fi
  if require_absent_output "$PARITY_PROJECT" >/dev/null 2>&1; then failed=1; fi
  if ! require_absent_output "$probe/nested/audit.json"; then failed=1; fi
  [[ ! -e "$probe/nested" ]] || failed=1
  if require_absent_output "$probe/../escaped.json" >/dev/null 2>&1; then failed=1; fi
  touch "$probe/existing.json"
  if require_absent_output "$probe/existing.json" >/dev/null 2>&1; then failed=1; fi
  ln -s "$probe" "$probe/link-parent"
  if require_absent_output "$probe/link-parent/new.json" >/dev/null 2>&1; then failed=1; fi
  rm -rf "$probe"
  if ! UV_CACHE_DIR=/private/tmp/vokra-neutts-air-uv-cache uv run --offline --no-project --python 3.12 python "$AUDIT" --self-test; then failed=1; fi
  if (( failed != 0 )); then
    echo "audit-neutts-air-dependencies.sh self-test FAIL" >&2
    return 1
  fi
  echo "audit-neutts-air-dependencies.sh self-test PASS"
}

case "${1:-}" in
  --self-test)
    [[ $# -eq 1 ]] || { usage; exit 2; }
    self_test
    ;;
  --output)
    [[ $# -eq 2 && "$2" == /* ]] || { usage; exit 2; }
    run_audit "$2"
    ;;
  -h|--help) usage; exit 0 ;;
  *) usage; exit 2 ;;
esac
