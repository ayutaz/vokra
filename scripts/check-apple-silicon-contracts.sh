#!/usr/bin/env bash
# Check every Apple Silicon worker's executable and hermetic contract test.
#
# Keep discovery dynamic: adding scripts/verify/apple-silicon-*.sh must make
# this check exercise the new worker automatically.  This check intentionally
# does not run Cargo or a model; the worker's --self-test mode is the static /
# synthetic contract path, while the workers themselves own real-device work.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERIFY_DIR="$ROOT/scripts/verify"
temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-apple-contracts.XXXXXX")"
trap 'rm -rf -- "$temporary"' EXIT

scripts=()
while IFS= read -r script; do
  scripts+=("$script")
done < <(find "$VERIFY_DIR" -type f -name 'apple-silicon-*.sh' -print | sort)

if (( ${#scripts[@]} == 0 )); then
  echo "Apple Silicon worker contract check: no worker scripts found" >&2
  exit 1
fi

syntax_failures=0
self_test_failures=0
unsupported=0
for script in "${scripts[@]}"; do
  relative="${script#"$ROOT/"}"
  log="$temporary/${#scripts[@]}-${relative##*/}.log"
  echo "::group::$relative"

  if bash -n "$script"; then
    echo "APPLE_WORKER_SYNTAX_PASS script=$relative"
  else
    echo "APPLE_WORKER_SYNTAX_FAIL script=$relative" >&2
    syntax_failures=$((syntax_failures + 1))
  fi

  # Do not use `||` here: retain the worker's exact status so exit 2 cannot be
  # mistaken for a successful test.  The pipe keeps each worker's output in
  # the Actions log and in a per-worker file for post-run audit.
  set +e
  bash "$script" --self-test 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  set -e
  case "$status" in
    0)
      echo "APPLE_WORKER_SELF_TEST_PASS script=$relative"
      ;;
    2)
      echo "::error file=$relative::--self-test returned 2 (unsupported/blocked contract); this is not a pass"
      echo "APPLE_WORKER_SELF_TEST_UNSUPPORTED script=$relative status=2"
      unsupported=$((unsupported + 1))
      self_test_failures=$((self_test_failures + 1))
      ;;
    *)
      echo "::error file=$relative::--self-test failed with status $status"
      echo "APPLE_WORKER_SELF_TEST_FAIL script=$relative status=$status"
      self_test_failures=$((self_test_failures + 1))
      ;;
  esac
  echo "::endgroup::"
done

printf 'Apple worker contract summary: scripts=%d syntax_failures=%d self_test_failures=%d unsupported_or_blocked=%d\n' \
  "${#scripts[@]}" "$syntax_failures" "$self_test_failures" "$unsupported"

if (( syntax_failures != 0 || self_test_failures != 0 )); then
  exit 1
fi
