#!/usr/bin/env bash
# Apple Silicon host-readiness inspection for HT-Demucs multi.
# The ensemble manifest and native binder are not audited, so this script
# records no CPU/Metal execution or parity verdict.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

log() { printf '[htdemucs-multi-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF'
usage: apple-silicon-htdemucs-multi.sh --evidence-dir <absent-dir>
       apple-silicon-htdemucs-multi.sh --self-test

Checks Darwin/arm64 Metal host readiness only. Until VAST authenticates the
official ensemble member manifests and a native binder exists, the output is
INSPECTION_ONLY; this script runs no model, parity test, conversion, or upload.
EOF
}

reject_symlink_ancestors() {
  local path="$1" component rest current
  [[ "$path" == /* ]] || { die "path must be absolute: $path"; return 2; }
  rest="${path#/}"
  current="/"
  while [[ -n "$rest" ]]; do
    if [[ "$rest" == */* ]]; then
      component="${rest%%/*}"
      rest="${rest#*/}"
    else
      component="$rest"
      rest=""
    fi
    [[ -n "$component" ]] || continue
    current="$current$component"
    [[ ! -L "$current" ]] || { die "path contains symlink ancestor: $path"; return 2; }
    current="$current/"
  done
}

validate_evidence_path() {
  local evidence="$1" root_real parent leaf candidate
  local -a suffix=()
  reject_symlink_ancestors "$evidence" || return 2
  [[ ! -e "$evidence" && ! -L "$evidence" ]] \
    || { die "evidence directory must be absent and non-symlinked: $evidence"; return 2; }

  # Resolve the nearest existing parent. This permits a nested absent path
  # without ever traversing an untrusted intermediate component.
  parent="$evidence"
  while [[ ! -e "$parent" ]]; do
    leaf="${parent##*/}"
    [[ -n "$leaf" ]] || { die "evidence path has an invalid parent: $evidence"; return 2; }
    suffix+=("$leaf")
    [[ "$parent" != / ]] || { die "evidence path parent does not exist: $evidence"; return 2; }
    parent="${parent%/*}"
    [[ -n "$parent" ]] || parent=/
  done
  [[ -d "$parent" && ! -L "$parent" ]] || { die "evidence parent is not a directory: $parent"; return 2; }
  candidate="$(cd -P "$parent" && pwd)" || { die "could not resolve evidence parent"; return 2; }
  for (( leaf = ${#suffix[@]} - 1; leaf >= 0; leaf-- )); do
    candidate="$candidate/${suffix[leaf]}"
  done
  root_real="$(cd -P "$VOKRA_ROOT" && pwd)" || { die "could not resolve Vokra checkout"; return 2; }
  [[ "$candidate" != "$root_real" && "$candidate/" != "$root_real/"* && \
    "$root_real/" != "$candidate/"* ]] \
    || { die "evidence directory overlaps the Vokra checkout"; return 2; }
}

self_test() (
  local path="${BASH_SOURCE[0]}" fail=0 token temporary real_path
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-htdemucs-apple.XXXXXX")"
  temporary="$(cd -P "$temporary" && pwd)"
  trap 'rm -rf "$temporary"' EXIT
  mkdir -p "$temporary/real"
  validate_evidence_path "$temporary/nested/new/evidence" \
    || { log 'self-test FAIL: external nested absent evidence path was rejected'; fail=1; }
  if validate_evidence_path "$VOKRA_ROOT" >/dev/null 2>&1; then
    log 'self-test FAIL: checkout itself was accepted as evidence'
    fail=1
  fi
  if validate_evidence_path "$VOKRA_ROOT/.git/new-evidence" >/dev/null 2>&1; then
    log 'self-test FAIL: checkout descendant was accepted as evidence'
    fail=1
  fi
  mkdir -p "$temporary/real/existing"
  if validate_evidence_path "$temporary/real/existing" >/dev/null 2>&1; then
    log 'self-test FAIL: existing empty evidence directory was accepted'
    fail=1
  fi
  ln -s "$temporary/real" "$temporary/link"
  if validate_evidence_path "$temporary/link/existing-descendant/new" >/dev/null 2>&1; then
    log 'self-test FAIL: symlink ancestor was accepted'
    fail=1
  fi
  real_path="$temporary/real"
  [[ -d "$real_path" && ! -e "$temporary/probe/evidence" ]] \
    || { log 'self-test FAIL: temporary path setup is inconsistent'; fail=1; }
  if VOKRA_REMOTE_APPLE_SILICON=0 "$path" --evidence-dir "$temporary/probe/evidence" >/dev/null 2>&1; then
    log 'self-test FAIL: production-shaped host probe unexpectedly passed'
    fail=1
  fi
  [[ ! -e "$temporary/probe/evidence" && ! -L "$temporary/probe/evidence" ]] \
    || { log 'self-test FAIL: blocked probe created evidence'; fail=1; }
  for token in 'VOKRA_REMOTE_APPLE_SILICON=1' 'Darwin' 'arm64' \
    'xcrun -f metal' 'INSPECTION_ONLY' 'official ensemble member manifests' \
    'hardware_probe_is_not_htdemucs_parity_evidence=true' \
    'runtime_status=INSPECTION_ONLY' 'parity_status=INSPECTION_ONLY' \
    'git status --porcelain' 'reject_symlink_ancestors' \
    'validate_evidence_path' 'evidence directory must be absent'; do
    if ! grep -Fq -- "$token" "$path"; then
      log "self-test FAIL: missing contract token: $token"
      fail=1
    fi
  done
  if grep -En '(^|[[:space:]])(curl|wget|python3?|pip|.*convert|.*upload|.*publish|git[[:space:]]+push)([[:space:]]|$)' "$path" >/dev/null; then
    log 'self-test FAIL: acquisition or publication command found'
    fail=1
  fi
  if grep -En '^[[:space:]]*printf[^#]*HTDEMUCS[^#]*PASS' "$path" >/dev/null; then
    log 'self-test FAIL: verifier manufactures a parity PASS marker'
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
  (( fail == 0 )) || return 1
  log 'self-test PASS'
)

evidence_dir=''
self=0
seen_evidence=0
seen_self=0
while (($#)); do
  case "$1" in
    --self-test)
      (( seen_self == 0 )) || die 'duplicate --self-test'
      seen_self=1
      self=1
      shift
      ;;
    --evidence-dir)
      (( seen_evidence == 0 )) || die 'duplicate --evidence-dir'
      [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--evidence-dir requires a nonempty path'
      seen_evidence=1
      evidence_dir="$2"
      shift 2
      ;;
    -h|--help)
      [[ $# == 1 && $self == 0 ]] || die '--help cannot be combined with other arguments'
      usage
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done
if (( self )); then
  (( seen_evidence == 0 )) || die '--self-test accepts no other arguments'
  self_test
  exit $?
fi

[[ -n "$evidence_dir" ]] || die '--evidence-dir is required'
validate_evidence_path "$evidence_dir"
[[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == 1 ]] || die 'VOKRA_REMOTE_APPLE_SILICON=1 is absent'
[[ "$(uname -s)" == Darwin ]] || die 'host readiness requires Darwin'
[[ "$(uname -m)" == arm64 ]] || die 'host readiness requires Apple arm64'
command -v xcrun >/dev/null 2>&1 || die 'xcrun is unavailable'
xcrun -f metal >/dev/null 2>&1 || die 'Xcode Metal tooling is unavailable'
[[ -f "$VOKRA_ROOT/Cargo.toml" && -d "$VOKRA_ROOT/.git" ]] || die 'not a Vokra checkout'
[[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'Apple checkout must be clean'
mkdir -p "$evidence_dir"

{
  echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
  echo "host=$(uname -a)"
  echo "metal_compiler=$(xcrun -f metal)"
  echo 'runtime_status=INSPECTION_ONLY'
  echo 'parity_status=INSPECTION_ONLY'
  echo 'verdict=NO_CPU_OR_METAL_PASS'
  echo 'reason=official ensemble member manifests and native binder are not audited'
  echo 'hardware_probe_is_not_htdemucs_parity_evidence=true'
} > "$evidence_dir/htdemucs-multi-apple-inspection.txt"
log "recorded inspection-only host evidence at $evidence_dir; no model or parity execution"
