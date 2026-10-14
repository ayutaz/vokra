#!/usr/bin/env bash
# Regression test for the NSNet2 VAST NumPy native-payload audit.
#
# The audit runs before the pinned ONNX download, so this test intentionally
# uses only a synthetic site-packages tree and mocked ldd/grep functions. The
# audit function is extracted from the production worker at runtime; keeping
# the test coupled to that function prevents a duplicated test implementation
# from going green while the worker regresses.
#
# The large forbidden fixture is the regression control: a forbidden runtime on
# the first ldd line followed by >5 MiB of output used to make
# `printf "$ldd_output" | grep -q` return 141 under pipefail, which skipped the
# rejection. The clean fixture prevents the test from becoming an always-fail
# gate. The ldd and grep error fixtures lock in fail-closed handling for errors
# other than grep's ordinary no-match status 1.

set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AUDIT_SCRIPT="$ROOT/scripts/publish/vast-ai/run-nsnet2-validation.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

pass=0
fail=0

ok() {
  printf '  PASS: %s\n' "$1"
  pass=$((pass + 1))
}

bad() {
  printf '  FAIL: %s\n' "$1" >&2
  fail=$((fail + 1))
}

function_source="$(sed -n '/^is_forbidden_native_basename()/,/^download_upstream_onnx()/p' "$AUDIT_SCRIPT" | sed '$d')"
if [[ -z "$function_source" || "$function_source" != *'audit_numpy_native_payload()'* ]]; then
  printf 'test-nsnet2-native-audit-sigpipe: production audit extraction failed\n' >&2
  exit 1
fi

# This is executed by a child shell with set -e enabled and as an unconditional
# command, matching main()'s production call context. The caller deliberately
# handles the child status outside that shell so negative controls can assert
# the exact fail-closed result.
# shellcheck disable=SC2016 # the child shell expands these variables at runtime
runner='
set -euo pipefail
log() { printf "[nsnet2-vast] %s\\n" "$*" >&2; }
die() { log "ERROR: $*"; return 2; }
eval "$FUNCTION_SOURCE"
ldd() { cat "$MOCK_LDD_FILE"; return "${MOCK_LDD_RC:-0}"; }
grep() {
  if [[ "${MOCK_GREP_RC:-0}" != 0 ]]; then
    return "$MOCK_GREP_RC"
  fi
  command grep "$@"
}
audit_numpy_native_payload
'

make_tree() {
  local root="$1" mode="$2"
  mkdir -p "$root/.venv/lib/python3.12/site-packages/numpy/core"
  touch "$root/.venv/lib/python3.12/site-packages/numpy/core/fake_extension.so"
  case "$mode" in
    small-forbidden)
      printf '%s\n' 'libopenblas.so.0 => /usr/lib/libopenblas.so.0 (0x0)' > "$root/ldd.out"
      ;;
    large-forbidden)
      {
        printf '%s\n' 'libopenblas.so.0 => /usr/lib/libopenblas.so.0 (0x0)'
        awk 'BEGIN { for (i = 0; i < 300000; i++) print "clean-padding-line" }'
      } > "$root/ldd.out"
      ;;
    large-clean)
      awk 'BEGIN { for (i = 0; i < 300000; i++) print "clean-padding-line" }' > "$root/ldd.out"
      ;;
    ldd-error|grep-error)
      printf '%s\n' 'libc.so.6 => /usr/lib/libc.so.6 (0x0)' > "$root/ldd.out"
      ;;
    *)
      printf 'unknown fixture: %s\n' "$mode" >&2
      return 2
      ;;
  esac
}

run_case() {
  local label="$1" mode="$2" expected_rc="$3" ldd_rc="$4" grep_rc="$5"
  local root="$TMP/$label" rc ldd_bytes
  mkdir "$root"
  make_tree "$root" "$mode"
  ldd_bytes="$(wc -c < "$root/ldd.out" | tr -d '[:space:]')"

  # Do not put the child command in an if/while condition: Bash suppresses
  # errexit in that context and would no longer model main()'s call site.
  PARITY_PROJECT="$root" FUNCTION_SOURCE="$function_source" \
    MOCK_LDD_FILE="$root/ldd.out" MOCK_LDD_RC="$ldd_rc" MOCK_GREP_RC="$grep_rc" \
    bash -c "$runner" > "$root/stdout" 2> "$root/stderr"
  rc=$?

  if [[ "$rc" == "$expected_rc" ]]; then
    ok "$label (ldd bytes=$ldd_bytes, exit=$rc)"
  else
    bad "$label expected exit=$expected_rc, got exit=$rc (ldd bytes=$ldd_bytes)"
    sed -n '1,3p' "$root/stderr" >&2
  fi
}

printf 'test-nsnet2-native-audit-sigpipe: production audit regression\n'
run_case small-forbidden small-forbidden 2 0 0
run_case large-forbidden large-forbidden 2 0 0
run_case large-clean large-clean 0 0 0
run_case ldd-error ldd-error 2 7 0
run_case grep-error grep-error 2 0 2

printf '\n'
if [[ "$fail" -ne 0 ]]; then
  printf 'test-nsnet2-native-audit-sigpipe: %d failed, %d passed\n' "$fail" "$pass" >&2
  exit 1
fi
printf 'test-nsnet2-native-audit-sigpipe: OK (%d passed)\n' "$pass"
