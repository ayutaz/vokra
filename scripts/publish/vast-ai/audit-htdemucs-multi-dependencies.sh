#!/usr/bin/env bash
# VAST/Linux-only, model-free frozen dependency evidence for HT-Demucs.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
PROJECT="$VOKRA_ROOT/tools/parity/htdemucs_multi"
COLLECTOR="$PROJECT/collect_dependency_evidence.py"

die() { printf '[htdemucs-dependency-evidence] ERROR: %s\n' "$*" >&2; return 2; }

usage() {
  cat >&2 <<'EOF'
usage: audit-htdemucs-multi-dependencies.sh --expected-head <40-hex-commit> --output <absent-json>
       audit-htdemucs-multi-dependencies.sh --self-test

Run only on a clean Linux/x86_64 VAST checkout.  The production path freezes
the dedicated Python 3.12 project, then runs the collector with --no-sync. It
does not acquire models, weights, audio, source repositories, Cargo artifacts,
or publish/upload anything.  Evidence is always BLOCKED_OWNER_REVIEW.
EOF
}

canonical_uncreated() {
  local path="$1" component rest suffix='' parent scan=''
  [[ "$path" = /* ]] || return 1
  rest="${path#/}"
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; rest="${rest#*/}"
    [[ "$component" == "$rest" ]] && rest=''
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || return 1
    scan="$scan/$component"; [[ ! -L "$scan" ]] || return 1
  done
  while [[ ! -d "$path" || -L "$path" ]]; do
    component="${path##*/}"; [[ -n "$component" ]] && suffix="/$component$suffix"
    parent="${path%/*}"; [[ "$parent" == "$path" ]] && parent=/; path="$parent"
    [[ ! -L "$path" ]] || return 1
  done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}

paths_overlap() { [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]; }

require_absent_output() {
  local output="$1" canonical root project
  [[ "$output" = /* && ! -e "$output" && ! -L "$output" ]] || { die '--output must be an absent absolute path'; return 2; }
  canonical="$(canonical_uncreated "$output")" || { die 'output path has unsafe/symlink ancestry'; return 2; }
  root="$(canonical_uncreated "$VOKRA_ROOT")" || { die 'checkout canonicalization failed'; return 2; }
  project="$(canonical_uncreated "$PROJECT")" || { die 'project canonicalization failed'; return 2; }
  paths_overlap "$canonical" "$root" && { die 'output overlaps checkout'; return 2; }
  paths_overlap "$canonical" "$project" && { die 'output overlaps HT-Demucs project'; return 2; }
  return 0
}

check_environment() {
  local expected_head="$1" actual status free_kib disk_kib
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || { die 'VOKRA_PUBLISH_ON_VAST=1 is required'; return 2; }
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { die 'Linux x86_64 VAST host required'; return 2; }
  for command in uv readelf git; do command -v "$command" >/dev/null 2>&1 || { die "$command is required"; return 2; }; done
  [[ -f "$PROJECT/pyproject.toml" && -f "$PROJECT/uv.lock" && -f "$PROJECT/license_gate_manifest.json" && -f "$PROJECT/dependency_audit.json" && -f "$PROJECT/audit.py" && -f "$COLLECTOR" ]] || { die 'exact HT-Demucs audit inputs are missing'; return 2; }
  [[ -d "$VOKRA_ROOT/.git" ]] || { die 'checkout .git metadata is missing'; return 2; }
  [[ "$expected_head" =~ ^[0-9a-f]{40}$ ]] || { die '--expected-head requires lowercase 40-hex'; return 2; }
  status="$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)"; [[ -z "$status" ]] || { die 'VAST checkout must be clean'; return 2; }
  actual="$(git -C "$VOKRA_ROOT" rev-parse HEAD)"; [[ "$actual" == "$expected_head" ]] || { die "checkout HEAD $actual differs from --expected-head $expected_head"; return 2; }
  free_kib="$(awk '/MemAvailable:/ {print $2; exit}' /proc/meminfo 2>/dev/null || printf 0)"
  disk_kib="$(df -Pk "$VOKRA_ROOT" | awk 'NR==2 {print $4}')"
  [[ "${free_kib:-0}" -ge 8388608 ]] || { die 'at least 8 GiB available RAM is required'; return 2; }
  [[ "${disk_kib:-0}" -ge 20971520 ]] || { die 'at least 20 GiB free checkout filesystem space is required'; return 2; }
}

self_test() {
  local temp
  for token in VOKRA_PUBLISH_ON_VAST PYTHONDONTWRITEBYTECODE uv sync --no-install-project --no-sync --frozen --project --expected-head readelf dependency_audit.py BLOCKED_OWNER_REVIEW NO_UPLOAD weights audio source Cargo; do
    grep -Fq -- "$token" "$0" || { die "wrapper contract missing: $token"; return 1; }
  done
  if grep -En '(^|[;&|][[:space:]])(python|python3|pip|conda)([[:space:]]|$)' "$0" >/dev/null; then
    die 'bare Python/pip/conda invocation found'; return 1
  fi
  PYTHONDONTWRITEBYTECODE=1 UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$COLLECTOR" --self-test
  temp="$(mktemp -d "/private/tmp/htdemucs-dependency-evidence.XXXXXXXX")"
  trap 'if [[ -n "${temp:-}" ]]; then rm -rf "$temp"; fi' RETURN
  mkdir -p "$temp/real"
  require_absent_output "$temp/new/deeper/report.json"
  if require_absent_output "$temp/../escape.json" >/dev/null 2>&1; then die 'dot-dot output accepted'; return 1; fi
  ln -s "$temp/real" "$temp/link"
  if require_absent_output "$temp/link/report.json" >/dev/null 2>&1; then die 'symlink output ancestor accepted'; return 1; fi
  if require_absent_output "$VOKRA_ROOT/new-report.json" >/dev/null 2>&1; then die 'checkout-overlap output accepted'; return 1; fi
  printf 'audit-htdemucs-multi-dependencies.sh self-test: OK\n'
}

main() {
  local output='' expected_head='' self=0 destination
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --output) [[ $# -ge 2 && -n "$2" ]] || { die '--output requires a path'; return 2; }; output="$2"; shift 2;;
      --expected-head) [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{40}$ ]] || { die '--expected-head requires lowercase 40-hex'; return 2; }; expected_head="$2"; shift 2;;
      --self-test) (( self++ == 0 )) || { die 'duplicate --self-test'; return 2; }; shift;;
      -h|--help) usage; return 0;;
      *) die "unknown argument: $1"; return 2;;
    esac
  done
  if (( self )); then [[ -z "$output" && -z "$expected_head" ]] || { die '--self-test accepts no output/head'; return 2; }; self_test; return $?; fi
  [[ -n "$output" && -n "$expected_head" ]] || { die '--expected-head and --output are required'; return 2; }
  check_environment "$expected_head"
  require_absent_output "$output"
  destination="$(canonical_uncreated "$output")"
  uv sync --project "$PROJECT" --frozen --python 3.12 --no-install-project
  PYTHONDONTWRITEBYTECODE=1 VOKRA_PUBLISH_ON_VAST=1 uv run --project "$PROJECT" --frozen --no-sync --python 3.12 python "$COLLECTOR" --project "$PROJECT" --output "$destination" --expected-head "$expected_head"
}

main "$@"
