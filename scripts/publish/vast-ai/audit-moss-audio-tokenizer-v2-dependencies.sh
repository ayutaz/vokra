#!/usr/bin/env bash
# Run the model-free MOSS v2 factual dependency audit in an already-synced VAST env.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
PROJECT="$VOKRA_ROOT/tools/parity/moss_audio_tokenizer_v2"
AUDIT="$PROJECT/dependency_audit.py"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

die() { printf '[moss-dependency-audit] ERROR: %s\n' "$*" >&2; return 2; }
usage() {
  cat >&2 <<'EOF'
usage: audit-moss-audio-tokenizer-v2-dependencies.sh --output <absent-json>
       audit-moss-audio-tokenizer-v2-dependencies.sh --self-test

VAST-only, model-free factual audit. Run after a separately authorized
`uv sync --frozen` on VAST; this wrapper itself uses --no-sync. It records
exact installed metadata, native/ELF facts, exact locked PyPI sdist license
bytes when publisher files are absent, and the pinned upstream LICENSE only.
It never downloads weights, imports model/Torch code, invokes Cargo, or syncs.
EOF
}
canonicalize_uncreated() {
  local path="$1" suffix='' name parent component rest scan
  [[ "$path" == /* ]] || return 1
  rest="${path#/}"; scan=''
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; rest="${rest#*/}"
    [[ "$component" == "$rest" ]] && rest=''
    [[ "$component" != .. ]] || return 1
    [[ -n "$component" && "$component" != . ]] || continue
    scan="$scan/$component"
    [[ ! -L "$scan" ]] || return 1
  done
  while [[ ! -d "$path" || -L "$path" ]]; do
    name="${path##*/}"
    [[ -n "$name" ]] && suffix="/$name$suffix"
    parent="${path%/*}"
    [[ "$parent" == "$path" ]] && parent='.'
    [[ -n "$parent" ]] || parent='/'
    path="$parent"
    [[ ! -L "$path" ]] || return 1
  done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}
paths_overlap() { [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]; }
require_absent_output() {
  local output="$1" canonical root project
  [[ "$output" == /* && ! -e "$output" && ! -L "$output" ]] || { die "output must be an absent absolute path"; return 2; }
  canonical="$(canonicalize_uncreated "$output")" || { die "output path is unsafe or has a symlink ancestor"; return 2; }
  root="$(canonicalize_uncreated "$VOKRA_ROOT")" || { die "checkout canonicalization failed"; return 2; }
  project="$(canonicalize_uncreated "$PROJECT")" || { die "project canonicalization failed"; return 2; }
  paths_overlap "$canonical" "$root" && { die "output overlaps checkout"; return 2; }
  paths_overlap "$canonical" "$project" && { die "output overlaps project"; return 2; }
  return 0
}
check_environment() {
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || { die "VOKRA_PUBLISH_ON_VAST=1 is required"; return 2; }
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { die "Linux x86_64 VAST host required"; return 2; }
  command -v uv >/dev/null 2>&1 || { die "uv is required"; return 2; }
  command -v readelf >/dev/null 2>&1 || { die "readelf is required"; return 2; }
  command -v git >/dev/null 2>&1 || { die "git is required"; return 2; }
  [[ -f "$PROJECT/pyproject.toml" && -f "$PROJECT/uv.lock" && -f "$PROJECT/license_gate_manifest.json" && -f "$PROJECT/license_gate.py" && -f "$AUDIT" ]] || { die "exact MOSS audit inputs are missing"; return 2; }
  [[ -d "$VOKRA_ROOT/.git" && -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || { die "checkout must be clean"; return 2; }
}
self_test() {
  local script="$0" required tmp nested real link
  for required in VOKRA_PUBLISH_ON_VAST --no-sync --frozen --project readelf dependency_audit.py LICENSE Cargo weights; do
    grep -Fq -- "$required" "$script" || { die "wrapper contract missing: $required"; return 1; }
  done
  if grep -En '^[[:space:]]*(python3|python|pip)([[:space:]]|$)' "$script" >/dev/null; then
    die 'direct Python/pip invocation found'; return 1
  fi
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$AUDIT" --self-test
  local tmp_base="/private/tmp"
  [[ -d "$tmp_base" ]] || tmp_base="${TMPDIR:-/tmp}"
  tmp="$(mktemp -d "$tmp_base/moss-audit.XXXXXXXX")"
  real="$tmp/real"; mkdir -p "$real"
  nested="$tmp/new/deeper/audit.json"
  if ! require_absent_output "$nested"; then die 'nonexistent nested output was rejected'; rm -rf "$tmp"; return 1; fi
  if require_absent_output "$tmp/../escape.json" >/dev/null 2>&1; then die 'dot-dot output was accepted'; rm -rf "$tmp"; return 1; fi
  link="$tmp/link"; ln -s "$real" "$link"
  if require_absent_output "$link/child/audit.json" >/dev/null 2>&1; then die 'symlink-ancestor output was accepted'; rm -rf "$tmp"; return 1; fi
  if require_absent_output "$VOKRA_ROOT/new-audit.json" >/dev/null 2>&1; then die 'checkout-overlap output was accepted'; rm -rf "$tmp"; return 1; fi
  rm -rf "$tmp"
  printf 'audit-moss-audio-tokenizer-v2-dependencies.sh self-test: OK\n'
}
main() {
  local output='' self=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --output) [[ $# -ge 2 && -n "$2" ]] || { die '--output requires a path'; return 2; }; output="$2"; shift 2;;
      --self-test) (( self++ == 0 )) || { die 'duplicate --self-test'; return 2; }; shift;;
      -h|--help) usage; return 0;;
      *) die "unknown argument: $1"; return 2;;
    esac
  done
  if (( self )); then [[ -z "$output" ]] || { die '--self-test accepts no output'; return 2; }; self_test; return $?; fi
  [[ -n "$output" ]] || { die '--output is required'; return 2; }
  check_environment
  require_absent_output "$output"
  local out; out="$(canonicalize_uncreated "$output")"
  VOKRA_PUBLISH_ON_VAST=1 uv run --no-cache --project "$PROJECT" --frozen --no-sync --python 3.12 python "$AUDIT" \
    --project "$PROJECT" --output "$out" --fetch-model-license
}
main "$@"
