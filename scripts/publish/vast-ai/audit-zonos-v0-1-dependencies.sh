#!/usr/bin/env bash
set -euo pipefail

# This wrapper is deliberately model/source/checkpoint-free.  It only installs
# the dedicated, locked Python dependency project and collects local facts.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT="$ROOT/tools/parity/zonos_v0_1_reference"
AUDITOR="$PROJECT/dependency_audit.py"
PREPARER="$PROJECT/prepare_numpy_no_blas.sh"

die() { echo "zonos-dependency-audit: ERROR: $*" >&2; exit 2; }

require_clean_head() {
  local expected="$1" actual
  [[ "$expected" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head must be 40 lowercase hexadecimal characters'
  [[ ! -L "$ROOT/.git" && ( -d "$ROOT/.git" || -f "$ROOT/.git" ) ]] || die 'checkout is missing a regular .git directory or gitfile'
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
  local failed=0 temporary output archive test_repo test_worktree test_head saved_root
  [[ -f "$PROJECT/pyproject.toml" && -f "$PROJECT/uv.lock" ]] || failed=1
  [[ -x "$PREPARER" && ! -L "$PREPARER" ]] || failed=1
  grep -Fq 'zonos_v0_1_reference' "$0" || failed=1
  grep -Fq 'prepare_numpy_no_blas.sh' "$0" || failed=1
  grep -Fq -- 'uv sync --frozen --no-install-project' "$0" || failed=1
  grep -Fq -- '--preparation' "$0" || failed=1
  grep -Fq -- ' -d "$ROOT/.git" || -f "$ROOT/.git" ' "$0" || failed=1
  local env_token="UV_PROJECT_"'ENVIRONMENT='
  [[ "$(grep -Fc "$env_token\"\$environment\"" "$0")" == 2 ]] || failed=1
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
  test_repo="$temporary/repo"
  test_worktree="$temporary/worktree"
  git init -q "$test_repo"
  git -C "$test_repo" config user.email audit-self-test@example.invalid
  git -C "$test_repo" config user.name audit-self-test
  printf '%s\n' gitfile-self-test > "$test_repo/file"
  git -C "$test_repo" add file
  git -C "$test_repo" commit -q -m init
  git -C "$test_repo" worktree add -q --detach "$test_worktree" HEAD
  test_head="$(git -C "$test_worktree" rev-parse HEAD)"
  saved_root="$ROOT"
  ROOT="$test_worktree"
  (require_clean_head "$test_head") || { echo 'linked worktree gitfile was rejected' >&2; failed=1; }
  ROOT="$saved_root"
  output="$temporary/evidence.json"
  archive="$temporary/archive"
  (require_absent_canonical_output "$output") || failed=1
  (require_absent_canonical_output "$archive") || failed=1
  touch "$output"
  mkdir "$archive"
  if (require_absent_canonical_output "$archive") >/dev/null 2>&1; then
    echo 'existing archive directory was accepted' >&2
    failed=1
  fi
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
  bash "$PREPARER" --self-test || failed=1
  (( failed == 0 )) || return 1
  echo 'audit-zonos-v0-1-dependencies.sh self-test: OK'
}

if [[ "${1:-}" == --self-test ]]; then
  [[ $# == 1 ]] || die '--self-test accepts no arguments'
  self_test
  exit 0
fi
[[ $# == 6 ]] || die 'expected --expected-head HEX40 --output ABSENT_FILE --publisher-archive ABSENT_DIR'
[[ "$1" == --expected-head && "$3" == --output && "$5" == --publisher-archive ]] || die 'arguments must be --expected-head, --output, and --publisher-archive'
expected_head="$2"
output="$4"
publisher_archive="$6"
require_clean_head "$expected_head"
require_absent_canonical_output "$output"
require_absent_canonical_output "$publisher_archive"
preparation="${ZONOS_NUMPY_PREPARATION:-${publisher_archive}.numpy-no-blas}"
require_absent_canonical_output "$preparation"
require_host
[[ "${VOKRA_ZONOS_DEPENDENCY_AUDIT:-0}" == 1 ]] || die 'VOKRA_ZONOS_DEPENDENCY_AUDIT=1 is absent'
for command in git realpath awk uv bash; do command -v "$command" >/dev/null || die "missing tool: $command"; done
[[ -x "$PREPARER" && ! -L "$PREPARER" ]] || die 'NumPy no-BLAS preparation helper is missing or symlinked'

cd "$ROOT"
VOKRA_PUBLISH_ON_VAST=1 bash "$PREPARER" --output-dir "$preparation"
environment="$preparation/venv"
[[ -d "$environment" && ! -L "$environment" ]] || die 'no-BLAS preparation environment is missing or symlinked'
[[ -f "$preparation/preparation.json" && ! -L "$preparation/preparation.json" ]] || die 'no-BLAS preparation identity is missing or symlinked'
set +e
UV_PROJECT_ENVIRONMENT="$environment" UV_CACHE_DIR="${ZONOS_UV_CACHE_DIR:-/tmp/vokra-zonos-uv-cache}" \
  uv run --frozen --project "$PROJECT" --no-sync --python 3.12 python "$AUDITOR" \
  --installed --expected-head "$expected_head" --publisher-archive "$publisher_archive" \
  --preparation "$preparation/preparation.json" --output "$output"
audit_status=$?
set -e
[[ "$audit_status" == 0 || "$audit_status" == 2 ]] || die "dependency collector failed with status $audit_status"
[[ -f "$output" && ! -L "$output" ]] || die 'dependency audit did not create regular evidence'
grep -Fq '"status": "BLOCKED_UNREVIEWED_TRANSITIVE"' "$output" || die 'blocked status marker missing'
grep -Fq '"publication": "NO_UPLOAD"' "$output" || die 'NO_UPLOAD marker missing'
set +e
UV_PROJECT_ENVIRONMENT="$environment" UV_CACHE_DIR="${ZONOS_UV_CACHE_DIR:-/tmp/vokra-zonos-uv-cache}" \
  uv run --frozen --project "$PROJECT" --no-sync --python 3.12 python "$AUDITOR" \
  --validate-output "$output"
validation_status=$?
set -e
[[ "$validation_status" == 0 ]] || die 'dependency audit evidence is incomplete or not hash-bound'
echo 'zonos dependency audit is BLOCKED_UNREVIEWED_TRANSITIVE; evidence was written; NO_UPLOAD' >&2
exit 2
