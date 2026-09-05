#!/usr/bin/env bash
# Fresh model-free SpeechT5 dependency/license/native audit on VAST.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/speecht5_tts"
AUDITOR="$PARITY_PROJECT/dependency_audit.py"
MIN_VAST_MEM_KIB=60000000
MIN_FREE_DISK_KIB=30000000

log() { printf '[speecht5-dependency-audit] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

canonical_path() {
  local target="$1" current="/" rest component suffix="" real
  [[ "$target" == /* ]] || { die 'path must be absolute'; return 2; }
  rest="${target#/}"
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"
    [[ "$rest" == "$component" ]] && rest="" || rest="${rest#*/}"
    [[ -z "$component" || "$component" == "." ]] && continue
    [[ "$component" != ".." ]] || { die 'path contains ..'; return 2; }
    current="${current%/}/$component"
    if [[ -L "$current" ]]; then
      real="$(cd -P "$current" 2>/dev/null && pwd)" || { die 'path has inaccessible symlink component'; return 2; }
      [[ "$current:$real" == "/var:/private/var" || "$current:$real" == "/tmp:/private/tmp" ]] || { die 'path has a symlinked component'; return 2; }
      current="$real"
    fi
  done
  current="$target"
  while [[ ! -e "$current" && ! -L "$current" && "$current" != / ]]; do
    component="$(basename "$current")"
    suffix="/$component$suffix"
    current="$(dirname "$current")"
  done
  [[ -d "$current" && ! -L "$current" ]] || { die 'existing parent is missing or symlinked'; return 2; }
  real="$(cd -P "$current" 2>/dev/null && pwd)" || { die 'existing parent is inaccessible'; return 2; }
  printf '%s%s\n' "$real" "$suffix"
}

paths_overlap() { [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]; }

require_absent_output() {
  local output="$1" canonical root project
  [[ "$output" == /* && ! -e "$output" && ! -L "$output" ]] || { die "output must be an absent absolute non-symlink path: $output"; return 2; }
  canonical="$(canonical_path "$output")" || return 2
  root="$(canonical_path "$VOKRA_ROOT")" || return 2
  project="$(canonical_path "$PARITY_PROJECT")" || return 2
  paths_overlap "$canonical" "$root" && { die 'output overlaps checkout'; return 2; }
  paths_overlap "$canonical" "$project" && { die 'output overlaps parity project'; return 2; }
  return 0
}

require_vast() {
  local memory free disk_path="$1"
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] || { die 'VOKRA_PUBLISH_ON_VAST=1 is required for an authorized VAST audit job'; return 2; }
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { die 'fresh dependency audit requires Linux x86_64 VAST'; return 2; }
  memory="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$memory" =~ ^[0-9]+$ && "$memory" -ge "$MIN_VAST_MEM_KIB" ]] || { die 'VAST RAM is below the 60-GB guard'; return 2; }
  while [[ ! -e "$disk_path" && "$disk_path" != / ]]; do disk_path="$(dirname "$disk_path")"; done
  free="$(df -Pk "$disk_path" | awk 'NR == 2 {print $4}')"
  [[ "$free" =~ ^[0-9]+$ && "$free" -ge "$MIN_FREE_DISK_KIB" ]] || { die 'free disk is below the 30-GB guard'; return 2; }
}

require_clean_checkout() {
  [[ "$VOKRA_ROOT" == /* && -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" ]] || { die 'VOKRA_ROOT is not an absolute Vokra checkout'; return 2; }
  [[ "$(canonical_path "$VOKRA_ROOT")" == "$VOKRA_ROOT" ]] || { die 'VOKRA_ROOT uses a symlinked path'; return 2; }
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || { die 'VAST checkout must be clean including untracked files'; return 2; }
}

require_inputs() {
  local path
  for path in pyproject.toml uv.lock license_gate_manifest.json preflight_gate.py dependency_audit.py; do
    [[ -f "$PARITY_PROJECT/$path" && ! -L "$PARITY_PROJECT/$path" ]] || { die "missing or symlinked SpeechT5 audit input: $path"; return 2; }
  done
  for path in uv git awk df readelf sha256sum tee; do
    command -v "$path" >/dev/null 2>&1 || { die "required tool missing: $path"; return 2; }
  done
}

run_audit() {
  local output="$1" rc
  local full="$output/full-audit.json" compact="$output/compact-audit.json" log_path="$output/audit.log"
  require_vast "$(dirname "$output")"
  require_clean_checkout
  require_inputs
  require_absent_output "$output"
  mkdir -p "$output"
  log 'Synchronizing the frozen active SpeechT5 project (no model/source acquisition)'
  set +e
  UV_NO_CACHE=1 UV_CACHE_DIR="${SPEECHT5_AUDIT_UV_CACHE_DIR:-/tmp/vokra-speecht5-audit-uv-cache}" \
    uv sync --project "$PARITY_PROJECT" --frozen --python 3.12 2>&1 | tee "$log_path"
  rc="${PIPESTATUS[0]}"
  set -e
  (( rc == 0 )) || { log "uv sync failed (rc=$rc)"; return "$rc"; }
  log 'Collecting exact closure, license, native, bundled, and build-only facts'
  set +e
  UV_CACHE_DIR="${SPEECHT5_AUDIT_UV_CACHE_DIR:-/tmp/vokra-speecht5-audit-uv-cache}" \
    uv run --project "$PARITY_PROJECT" --frozen --no-sync --python 3.12 python "$AUDITOR" \
      --project "$PARITY_PROJECT" --full-output "$full" --compact-output "$compact" 2>&1 | tee -a "$log_path"
  rc="${PIPESTATUS[0]}"
  set -e
  if [[ -f "$log_path" && -f "$full" && -f "$compact" ]]; then
    (cd "$output" && sha256sum audit.log full-audit.json compact-audit.json > SHA256SUMS)
  fi
  return "$rc"
}

self_test() {
  local tmp parent=/tmp failed=0
  [[ -d /private/tmp && ! -L /private/tmp ]] && parent=/private/tmp
  tmp="$(mktemp -d "$parent/speecht5-dependency-audit-wrapper.XXXXXX")"
  trap 'rm -rf "$tmp"' EXIT
  grep -Fq -- 'VOKRA_PUBLISH_ON_VAST=1' "$0" || failed=1
  grep -Fq -- 'uv sync --project' "$0" || failed=1
  grep -Fq -- '--frozen --no-sync --python 3.12' "$0" || failed=1
  grep -Fq -- 'dependency_audit.py' "$0" || failed=1
  grep -Fq -- 'compact-audit.json' "$0" || failed=1
  grep -Fq -- 'sha256sum audit.log full-audit.json compact-audit.json > SHA256SUMS' "$0" || failed=1
  ! grep -Eq '(^|[;&|[:space:]])find[[:space:]]+\.[[:space:]]+-type[[:space:]]+f([[:space:]]|$)' "$0" || failed=1
  ! grep -Eq '(^|[;&|[:space:]])xargs([[:space:]]|$)' "$0" || failed=1
  ! grep -Eq '^[[:space:]]*(snapshot_download|huggingface-cli|cargo[[:space:]]+(build|test|check|clippy))([[:space:]]|$)' "$0" || failed=1
  if VOKRA_PUBLISH_ON_VAST=0 run_audit "$tmp/blocked" >/dev/null 2>&1; then failed=1; fi
  if require_absent_output "$VOKRA_ROOT" >/dev/null 2>&1; then failed=1; fi
  if require_absent_output "$PARITY_PROJECT" >/dev/null 2>&1; then failed=1; fi
  if ! require_absent_output "$tmp/new/nested" >/dev/null 2>&1; then failed=1; fi
  mkdir "$tmp/existing"
  if require_absent_output "$tmp/existing" >/dev/null 2>&1; then failed=1; fi
  ln -s "$tmp/existing" "$tmp/link"
  if require_absent_output "$tmp/link/new" >/dev/null 2>&1; then failed=1; fi
  rm "$tmp/link"
  UV_CACHE_DIR="$tmp/cache" uv run --no-cache --no-project --offline --python 3.12 python "$AUDITOR" --self-test >/dev/null 2>&1 || failed=1
  rm -rf "$tmp"
  trap - EXIT
  (( failed == 0 )) || { log 'self-test FAIL'; return 1; }
  echo 'audit-speecht5-tts-dependencies.sh self-test: PASS (model-free, NO_UPLOAD)'
}

main() {
  local output='' self=0
  while (( $# > 0 )); do
    case "$1" in
      --output-dir) [[ $# -eq 2 && -z "$output" && "$2" != -* ]] || { die 'invalid or duplicate --output-dir'; return 2; }; output="$2"; shift 2 ;;
      --self-test) (( self == 0 )) || { die 'duplicate --self-test'; return 2; }; self=1; shift ;;
      -h|--help) printf 'usage: audit-speecht5-tts-dependencies.sh --output-dir ABSENT_DIR\n       audit-speecht5-tts-dependencies.sh --self-test\n' >&2; return 0 ;;
      *) die "unknown argument: $1"; return 2 ;;
    esac
  done
  if (( self )); then [[ -z "$output" ]] || { die '--self-test accepts no output'; return 2; }; self_test; return $?; fi
  [[ -n "$output" ]] || { die '--output-dir is required'; return 2; }
  run_audit "$output"
}

main "$@"
