#!/usr/bin/env bash
set -euo pipefail

# This wrapper is deliberately model/source/checkpoint-free.  It only installs
# the dedicated, locked Python dependency project and collects local facts.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT="$ROOT/tools/parity/zonos_v0_1_reference"
AUDITOR="$PROJECT/dependency_audit.py"

die() { echo "zonos-dependency-audit: ERROR: $*" >&2; exit 2; }

require_clean_head() {
  local expected="$1" actual
  [[ "$expected" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head must be 40 lowercase hexadecimal characters'
  [[ -d "$ROOT/.git" ]] || die 'checkout is missing .git'
  [[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
  actual="$(git -C "$ROOT" rev-parse HEAD)"
  [[ "$actual" == "$expected" ]] || die "checkout HEAD $actual differs from expected $expected"
}

require_absent_canonical_output() {
  local output="$1" parent canonical_root cursor
  [[ "$output" = /* ]] || die '--output must be an absolute path'
  [[ "$output" != "$ROOT" && "$output" != "$ROOT"/* ]] || die '--output must be outside the checkout'
  [[ "$output" != */ ]] || die '--output must name a file'
  [[ ! -e "$output" && ! -L "$output" ]] || die 'canonical output already exists or is symlinked'
  parent="${output%/*}"
  [[ -n "$parent" && -d "$parent" && ! -L "$parent" ]] || die 'output parent must be an existing directory'
  cursor="$parent"
  while [[ "$cursor" != / ]]; do
    [[ ! -L "$cursor" ]] || die 'output path has a symlinked ancestor'
    cursor="${cursor%/*}"
    [[ -n "$cursor" ]] || cursor=/
  done
  canonical_root="$(realpath "$ROOT")"
  [[ "$(realpath "$parent")" != "$canonical_root"/* ]] || die '--output parent is inside the checkout'
}

require_host() {
  local mem_kib
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'VAST audit requires Linux x86_64'
  mem_kib="$(awk '$1 == "MemTotal:" { print $2; exit }' /proc/meminfo)"
  [[ "$mem_kib" =~ ^[0-9]+$ && "$mem_kib" -ge $((60 * 1024 * 1024)) ]] || die '60 GiB host memory guard failed'
}

self_test() {
  local failed=0 temporary output
  [[ -f "$PROJECT/pyproject.toml" && -f "$PROJECT/uv.lock" ]] || failed=1
  grep -Fq 'zonos_v0_1_reference' "$0" || failed=1
  grep -Fq -- 'uv sync --frozen --no-install-project' "$0" || failed=1
  for forbidden in \
    'git '"clone" \
    'hugging'"face" \
    'H'"F_" \
    'carg'"o" \
    'publish-'"one" \
    'upl'"oad."; do
    if grep -Fq -- "$forbidden" "$0"; then
      echo "forbidden acquisition/publication token found: $forbidden" >&2
      failed=1
    fi
  done
  temporary="$(mktemp -d "$(realpath "${TMPDIR:-/tmp}")/vokra-zonos-audit-selftest.XXXXXX")"
  output="$temporary/evidence.json"
  (require_absent_canonical_output "$output") || failed=1
  touch "$output"
  if (require_absent_canonical_output "$output") >/dev/null 2>&1; then
    echo 'existing output was accepted' >&2
    failed=1
  fi
  mkdir "$temporary/real"
  ln -s "$temporary/real" "$temporary/link"
  if (require_absent_canonical_output "$temporary/link/blocked.json") >/dev/null 2>&1; then
    echo 'symlink output parent was accepted' >&2
    failed=1
  fi
  mkdir "$temporary/real/sub"
  if (require_absent_canonical_output "$temporary/link/sub/nested.json") >/dev/null 2>&1; then
    echo 'symlink output ancestor was accepted' >&2
    failed=1
  fi
  rm -rf "$temporary"
  (( failed == 0 )) || return 1
  echo 'audit-zonos-v0-1-dependencies.sh self-test: OK'
}

if [[ "${1:-}" == --self-test ]]; then
  [[ $# == 1 ]] || die '--self-test accepts no arguments'
  self_test
  exit 0
fi
[[ $# == 4 ]] || die 'expected --expected-head HEX40 --output ABSENT_FILE'
[[ "$1" == --expected-head && "$3" == --output ]] || die 'arguments must be --expected-head and --output'
expected_head="$2"
output="$4"
require_clean_head "$expected_head"
require_absent_canonical_output "$output"
require_host
[[ "${VOKRA_ZONOS_DEPENDENCY_AUDIT:-0}" == 1 ]] || die 'VOKRA_ZONOS_DEPENDENCY_AUDIT=1 is absent'
for command in git realpath awk uv; do command -v "$command" >/dev/null || die "missing tool: $command"; done

cd "$ROOT"
UV_CACHE_DIR="${ZONOS_UV_CACHE_DIR:-/tmp/vokra-zonos-uv-cache}" \
  uv sync --frozen --no-install-project --project "$PROJECT" --python 3.12
set +e
UV_CACHE_DIR="${ZONOS_UV_CACHE_DIR:-/tmp/vokra-zonos-uv-cache}" \
  uv run --frozen --project "$PROJECT" --no-sync --python 3.12 python "$AUDITOR" \
  --installed --output "$output"
audit_status=$?
set -e
[[ "$audit_status" == 0 || "$audit_status" == 2 ]] || die "dependency collector failed with status $audit_status"
[[ -f "$output" && ! -L "$output" ]] || die 'dependency audit did not create regular evidence'
grep -Fq '"status": "BLOCKED_UNREVIEWED_TRANSITIVE"' "$output" || die 'blocked status marker missing'
grep -Fq '"publication": "NO_UPLOAD"' "$output" || die 'NO_UPLOAD marker missing'
grep -Fq '"model_access": false' "$output" || die 'model access boundary missing'
grep -Fq '"source_access": false' "$output" || die 'source access boundary missing'
grep -Fq '"checkpoint_access": false' "$output" || die 'checkpoint access boundary missing'
grep -Fq '"installed_closure_sha256"' "$output" || die 'installed closure digest missing'
grep -Fq '"native_files_sha256"' "$output" || die 'native digest missing'
grep -Fq '"publisher_files_sha256"' "$output" || die 'publisher digest missing'
echo 'zonos dependency audit is BLOCKED_UNREVIEWED_TRANSITIVE; evidence was written; NO_UPLOAD' >&2
exit 2
