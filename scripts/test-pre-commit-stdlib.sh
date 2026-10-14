#!/usr/bin/env bash
# Regression test for the production pre-commit hook's stdlib-only linter path.
#
# The hook is executed as-is inside an isolated temporary Git worktree. Only
# Cargo and its three preceding shell checks are mocked. The pipefail linter
# itself always runs through the real uv command with UV_OFFLINE=1, against
# deliberately unresolvable root and tools/parity projects. The caller's
# worktree is never used as the hook ROOT or mutated by this test.

set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_TMP="$(mktemp -d)"
FIXTURE_ROOT="$TEST_TMP/worktree"
MOCK_BIN="$TEST_TMP/bin"
LOG="$TEST_TMP/calls.log"
trap 'rm -rf "$TEST_TMP"' EXIT

REAL_UV="$(command -v uv)"
SOURCE_STATUS_BEFORE="$(git -C "$SOURCE_ROOT" status --porcelain)"
mkdir -p "$FIXTURE_ROOT/.githooks" "$FIXTURE_ROOT/scripts/compliance" "$FIXTURE_ROOT/tools/parity" "$MOCK_BIN"
cp "$SOURCE_ROOT/.githooks/pre-commit" "$FIXTURE_ROOT/.githooks/pre-commit"
cp "$SOURCE_ROOT/scripts/compliance/lint-pipefail-grep-q.py" "$FIXTURE_ROOT/scripts/compliance/lint-pipefail-grep-q.py"
chmod +x "$FIXTURE_ROOT/.githooks/pre-commit"
git -C "$FIXTURE_ROOT" init -q

cat >"$FIXTURE_ROOT/pyproject.toml" <<'EOF'
[project]
name = "pre-commit-root-sentinel-no-sync"
version = "0.0.0"
requires-python = ">=3.12"
dependencies = ["package-that-must-never-be-synced==0.0.0"]
EOF
cat >"$FIXTURE_ROOT/tools/parity/pyproject.toml" <<'EOF'
[project]
name = "pre-commit-tools-sentinel-no-sync"
version = "0.0.0"
requires-python = ">=3.12"
dependencies = ["another-package-that-must-never-be-synced==0.0.0"]
EOF
printf '%s\n' 'preserve this fixture sentinel' >"$FIXTURE_ROOT/preexisting-sentinel.txt"

ROOT_PROJECT_SHA="$(shasum -a 256 "$FIXTURE_ROOT/pyproject.toml" | awk '{print $1}')"
TOOLS_PROJECT_SHA="$(shasum -a 256 "$FIXTURE_ROOT/tools/parity/pyproject.toml" | awk '{print $1}')"
SENTINEL_SHA="$(shasum -a 256 "$FIXTURE_ROOT/preexisting-sentinel.txt" | awk '{print $1}')"

cat >"$MOCK_BIN/cargo" <<'MOCK'
#!/bin/bash
printf 'cargo %s\n' "$*" >>"$HOOK_TEST_LOG"
if [ "${HOOK_FAIL_COMMAND:-}" = "cargo" ]; then
    exit 17
fi
exit 0
MOCK

cat >"$MOCK_BIN/bash" <<'MOCK'
#!/bin/bash
printf 'bash %s\n' "$*" >>"$HOOK_TEST_LOG"
case "${1:-}" in
    scripts/check-forbidden-symbols.sh) command_name="forbidden" ;;
    scripts/check-zero-deps.sh) command_name="zero-deps" ;;
    scripts/check-fixture-eol-pins.sh) command_name="fixture-eol" ;;
    *) exec /bin/bash "$@" ;;
esac
if [ "${HOOK_FAIL_COMMAND:-}" = "$command_name" ]; then
    exit 23
fi
exit 0
MOCK

cat >"$MOCK_BIN/uv" <<'MOCK'
#!/bin/bash
printf 'uv %s\n' "$*" >>"$HOOK_TEST_LOG"
if [ "${MOCK_UV_FAIL:-0}" = "1" ]; then
    exit 42
fi
exec "$HOOK_REAL_UV" "$@"
MOCK
chmod +x "$MOCK_BIN/cargo" "$MOCK_BIN/bash" "$MOCK_BIN/uv"

run_hook() {
    local output_file="$1"
    shift
    : >"$LOG"
    set +e
    (
        cd "$FIXTURE_ROOT"
        env \
            -u VIRTUAL_ENV \
            PATH="$MOCK_BIN:$PATH" \
            HOOK_TEST_LOG="$LOG" \
            HOOK_REAL_UV="$REAL_UV" \
            UV_OFFLINE=1 \
            UV_CACHE_DIR="$TEST_TMP/uv-cache" \
            UV_PROJECT_ENVIRONMENT="$TEST_TMP/uv-project-env" \
            UV_NO_SYNC=0 \
            "$@" \
            /bin/bash "$FIXTURE_ROOT/.githooks/pre-commit" >"$output_file" 2>&1
    )
    local rc=$?
    set -e
    printf '%s\n' "$rc"
}

assert_log_exact() {
    local expected="$1"
    local actual
    actual="$(cat "$LOG")"
    if [ "$actual" != "$expected" ]; then
        printf 'unexpected hook calls:\n%s\nexpected exactly:\n%s\n' "$actual" "$expected" >&2
        exit 1
    fi
}

assert_fixture_unchanged() {
    local current
    current="$(shasum -a 256 "$FIXTURE_ROOT/pyproject.toml" | awk '{print $1}')"
    [ "$current" = "$ROOT_PROJECT_SHA" ]
    current="$(shasum -a 256 "$FIXTURE_ROOT/tools/parity/pyproject.toml" | awk '{print $1}')"
    [ "$current" = "$TOOLS_PROJECT_SHA" ]
    current="$(shasum -a 256 "$FIXTURE_ROOT/preexisting-sentinel.txt" | awk '{print $1}')"
    [ "$current" = "$SENTINEL_SHA" ]
    [ ! -e "$FIXTURE_ROOT/.venv" ] && [ ! -L "$FIXTURE_ROOT/.venv" ]
    [ ! -e "$FIXTURE_ROOT/tools/parity/.venv" ] && [ ! -L "$FIXTURE_ROOT/tools/parity/.venv" ]
    [ ! -e "$FIXTURE_ROOT/uv.lock" ] && [ ! -L "$FIXTURE_ROOT/uv.lock" ]
    [ ! -e "$FIXTURE_ROOT/tools/parity/uv.lock" ] && [ ! -L "$FIXTURE_ROOT/tools/parity/uv.lock" ]
    [ ! -e "$TEST_TMP/uv-project-env" ] && [ ! -L "$TEST_TMP/uv-project-env" ]
}

echo "test-pre-commit-stdlib: production hook and stdlib linter regression"

# A clean run must reach the real linter after Cargo plus the three shell
# checks. The explicit offline environment proves no dependency resolution.
clean_output="$TEST_TMP/clean.out"
clean_rc="$(run_hook "$clean_output")"
[ "$clean_rc" -eq 0 ] || { cat "$clean_output" >&2; exit 1; }
assert_log_exact "$(printf '%s\n' \
    'cargo fmt --all -- --check' \
    'bash scripts/check-forbidden-symbols.sh' \
    'bash scripts/check-zero-deps.sh' \
    'bash scripts/check-fixture-eol-pins.sh' \
    'uv run --no-project --python 3.12 python scripts/compliance/lint-pipefail-grep-q.py')"
assert_fixture_unchanged
grep -Fq 'lint-pipefail-grep-q: OK' "$clean_output"
echo "OK   clean hook uses real offline stdlib linter without sync"

# Each preceding command must remain present, ordered, and fail closed. The
# four cases are Cargo plus the three shell checks; expected statuses are the
# mock's exact 17 (Cargo) and 23 (shell-check) failures.
declare -a names=(cargo forbidden zero-deps fixture-eol)
declare -a expected_rcs=(17 23 23 23)
declare -a expected_logs=(
    'cargo fmt --all -- --check'
    $'cargo fmt --all -- --check\nbash scripts/check-forbidden-symbols.sh'
    $'cargo fmt --all -- --check\nbash scripts/check-forbidden-symbols.sh\nbash scripts/check-zero-deps.sh'
    $'cargo fmt --all -- --check\nbash scripts/check-forbidden-symbols.sh\nbash scripts/check-zero-deps.sh\nbash scripts/check-fixture-eol-pins.sh'
)
for index in "${!names[@]}"; do
    output="$TEST_TMP/fail-${names[index]}.out"
    rc="$(run_hook "$output" HOOK_FAIL_COMMAND="${names[index]}")"
    [ "$rc" -eq "${expected_rcs[index]}" ] || { cat "$output" >&2; exit 1; }
    assert_log_exact "${expected_logs[index]}"
    assert_fixture_unchanged
done
echo "OK   Cargo plus three shell checks retain exact order and failures"

# Skip semantics remain explicit and untouched.
skip_output="$TEST_TMP/skip.out"
skip_rc="$(run_hook "$skip_output" VOKRA_SKIP_HOOKS=1)"
[ "$skip_rc" -eq 0 ] || { cat "$skip_output" >&2; exit 1; }
assert_log_exact ""
assert_fixture_unchanged
grep -Fq 'pre-commit: skipped (VOKRA_SKIP_HOOKS=1)' "$skip_output"
echo "OK   explicit VOKRA_SKIP_HOOKS=1 semantics remain unchanged"

# A linter finding must propagate through the real hook. The bad fixture is
# created only inside the isolated worktree and is removed before final checks.
bad_producer='find . -maxdepth 1'
bad_filter='grep -q .'
{
    printf '%s\n' '#!/bin/bash' 'set -o pipefail'
    printf '%s %s\n' "$bad_producer |" "$bad_filter"
} >"$FIXTURE_ROOT/.githooks/test-pre-commit-bad.sh"
bad_output="$TEST_TMP/bad-lint.out"
bad_rc="$(run_hook "$bad_output")"
[ "$bad_rc" -ne 0 ] || { cat "$bad_output" >&2; exit 1; }
grep -Fq 'lint-pipefail-grep-q: FAIL' "$bad_output"
rm -f "$FIXTURE_ROOT/.githooks/test-pre-commit-bad.sh"
assert_fixture_unchanged
echo "OK   linter finding remains nonzero through production hook"

# A failure to invoke uv must also propagate, rather than being swallowed.
uv_output="$TEST_TMP/uv-fail.out"
uv_rc="$(run_hook "$uv_output" MOCK_UV_FAIL=1)"
[ "$uv_rc" -eq 42 ] || { cat "$uv_output" >&2; exit 1; }
assert_fixture_unchanged
echo "OK   uv invocation failure remains nonzero"

if [ "$(git -C "$SOURCE_ROOT" status --porcelain)" != "$SOURCE_STATUS_BEFORE" ]; then
    git -C "$SOURCE_ROOT" status --short >&2
    echo "caller worktree changed during isolated production-hook regression" >&2
    exit 1
fi

echo "test-pre-commit-stdlib: PASS"
