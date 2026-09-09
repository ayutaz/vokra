#!/usr/bin/env bash
# VAST/Linux-only, model-free dependency/native-payload audit for MOSS Nano.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
PROJECT="$VOKRA_ROOT/tools/parity/moss_audio_tokenizer_nano"
AUDIT="$PROJECT/dependency_audit.py"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

die() { printf '[moss-nano-dependency-audit] ERROR: %s\n' "$*" >&2; return 2; }

usage() {
  cat >&2 <<'EOF'
usage: audit-moss-audio-tokenizer-nano-dependencies.sh --expected-head <40-hex-commit> --output <absent-json>
       audit-moss-audio-tokenizer-nano-dependencies.sh --self-test

Run after the exact Nano uv project has been synced on a disposable VAST
Linux/x86_64 host.  The audit uses --no-sync and never imports model code,
downloads weights, invokes Cargo, converts, publishes, or uploads anything.
It records locked artifact identities, installed package license/EULA bytes,
native ELF hashes/NEEDED facts, and explicitly rejects CUDA/NVIDIA/Triton
payloads; the only accepted Torch identity is 2.7.1+cpu from the official CPU
index.
The report remains BLOCKED while owner review rows are unresolved.
EOF
}

canonical_uncreated() {
  local path="$1" suffix='' component rest scan parent
  [[ "$path" = /* ]] || return 1
  rest="${path#/}"; scan=''
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; rest="${rest#*/}"
    [[ "$component" == "$rest" ]] && rest=''
    [[ -n "$component" ]] || return 1
    [[ "$component" != . && "$component" != .. ]] || return 1
    scan="$scan/$component"
    [[ ! -L "$scan" ]] || return 1
  done
  while [[ ! -d "$path" || -L "$path" ]]; do
    component="${path##*/}"
    [[ -n "$component" ]] && suffix="/$component$suffix"
    parent="${path%/*}"; [[ "$parent" == "$path" ]] && parent='/'
    path="$parent"
    [[ ! -L "$path" ]] || return 1
  done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}

paths_overlap() { [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]; }

require_absent_output() {
  local output="$1" canonical root project
  [[ "$output" = /* && ! -e "$output" && ! -L "$output" ]] || { die '--output must be an absent absolute path'; return 2; }
  canonical="$(canonical_uncreated "$output")" || { die 'output path has an unsafe/symlink ancestor'; return 2; }
  root="$(canonical_uncreated "$VOKRA_ROOT")" || { die 'checkout canonicalization failed'; return 2; }
  project="$(canonical_uncreated "$PROJECT")" || { die 'project canonicalization failed'; return 2; }
  paths_overlap "$canonical" "$root" && { die 'output overlaps checkout'; return 2; }
  paths_overlap "$canonical" "$project" && { die 'output overlaps Nano project'; return 2; }
  return 0
}

check_environment() {
  local expected_head="$1" actual status
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || { die 'VOKRA_PUBLISH_ON_VAST=1 is required'; return 2; }
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { die 'Linux x86_64 VAST host required'; return 2; }
  for command in uv readelf git; do command -v "$command" >/dev/null 2>&1 || { die "$command is required"; return 2; }; done
  [[ -f "$PROJECT/pyproject.toml" && -f "$PROJECT/uv.lock" && -f "$PROJECT/license_gate_manifest.json" && -f "$PROJECT/license_gate.py" && -f "$AUDIT" ]] || { die 'exact Nano audit inputs are missing'; return 2; }
  [[ -d "$VOKRA_ROOT/.git" && -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || { die 'VAST checkout must be clean'; return 2; }
  actual="$(git -C "$VOKRA_ROOT" rev-parse HEAD)" || { die 'cannot resolve checkout HEAD'; return 2; }
  [[ "$actual" == "$expected_head" ]] || { die "checkout HEAD $actual differs from --expected-head $expected_head"; return 2; }
  status="$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)"
  [[ -z "$status" ]] || { die 'VAST checkout became dirty'; return 2; }
}

self_test() {
  local temp temp_root
  for token in VOKRA_PUBLISH_ON_VAST --no-sync --frozen --project --expected-head readelf dependency_audit.py CPU CUDA NVIDIA Triton '2.7.1+cpu' 'download.pytorch.org/whl/cpu' weights Cargo; do
    grep -Fq -- "$token" "$0" || { die "wrapper contract missing: $token"; return 1; }
  done
  if grep -En '(^|[;&|][[:space:]])(python|python3|pip)([[:space:]]|$)' "$0" >/dev/null; then
    die 'bare Python/pip invocation found'; return 1
  fi
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$AUDIT" --self-test
  temp_root="$(cd -P "${TMPDIR:-/tmp}" && pwd -P)" || { die 'temporary directory root is unavailable'; return 1; }
  temp="$(mktemp -d "$temp_root/moss-nano-audit.XXXXXXXX")"
  trap 'if [[ -n "${temp:-}" ]]; then rm -rf "$temp"; fi' RETURN
  mkdir -p "$temp/real"
  require_absent_output "$temp/new/deeper/report.json" || return 1
  if require_absent_output "$temp/../escape.json" >/dev/null 2>&1; then die 'dot-dot output was accepted'; return 1; fi
  if "$0" --output "$temp/missing-head.json" >/dev/null 2>&1; then die 'missing expected-head was accepted'; return 1; fi
  ln -s "$temp/real" "$temp/link"
  if require_absent_output "$temp/link/report.json" >/dev/null 2>&1; then die 'symlink ancestor output was accepted'; return 1; fi
  if require_absent_output "$VOKRA_ROOT/new-report.json" >/dev/null 2>&1; then die 'checkout-overlap output was accepted'; return 1; fi
  printf 'audit-moss-audio-tokenizer-nano-dependencies.sh self-test: OK\n'
}

main() {
  local output='' expected_head='' self=0
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
  local destination; destination="$(canonical_uncreated "$output")"
  VOKRA_PUBLISH_ON_VAST=1 uv run --no-cache --project "$PROJECT" --frozen --no-sync --python 3.12 python "$AUDIT" --project "$PROJECT" --output "$destination" --expected-head "$expected_head"
}

main "$@"
