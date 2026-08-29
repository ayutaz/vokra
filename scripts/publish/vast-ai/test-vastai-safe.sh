#!/usr/bin/env bash
# Offline contract test for vastai-safe.sh. No Vast CLI, network, or Cargo.

set -euo pipefail

script_dir="$(dirname -- "$0")"
script_dir="$(cd "$script_dir" && pwd)"
wrapper="$script_dir/vastai-safe.sh"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/vokra-vastai-safe-test.XXXXXX")"
trap 'rm -rf "$tmp_dir"' EXIT HUP INT TERM

stdout_file="$tmp_dir/stdout"
stderr_file="$tmp_dir/stderr"

run_case() {
  local expected_status="$1"
  local expected_stdout="$2"
  local expected_stderr="$3"
  local actual_status

  set +e
  # Values are exported to the offline `sh -c` fixture below.
  # shellcheck disable=SC2016,SC2034
  FAKE_STDOUT='ordinary output https://example.test/?api_key=dummy-secret&keep=1;tokenizer=kept-tokenizer' \
    FAKE_STDERR="ordinary warning 'https://example.test/?keep=1;access_token=another-dummy;tokenizer=kept-tokenizer'" \
  FAKE_STATUS="$expected_status" VASTAI_BIN=sh "$wrapper" -c \
    'printf "%s\\n" "$FAKE_STDOUT"; printf "%s\\n" "$FAKE_STDERR" >&2; exit "$FAKE_STATUS"' \
    >"$stdout_file" 2>"$stderr_file"
  actual_status=$?
  set -e

  if [[ "$actual_status" != "$expected_status" ]]; then
    echo "vastai-safe self-test: wrapped status was not preserved" >&2
    return 1
  fi
  if ! grep -Fqx -- "$expected_stdout" "$stdout_file"; then
    echo "vastai-safe self-test: ordinary/redacted stdout mismatch" >&2
    return 1
  fi
  if ! grep -Fqx -- "$expected_stderr" "$stderr_file"; then
    echo "vastai-safe self-test: ordinary/redacted stderr mismatch" >&2
    return 1
  fi
  if grep -Fq -- 'api_key=dummy-secret' "$stdout_file" || \
     grep -Fq -- 'api_key=dummy-secret' "$stderr_file" || \
     grep -Fq -- 'access_token=another-dummy' "$stdout_file" || \
     grep -Fq -- 'access_token=another-dummy' "$stderr_file"; then
    echo "vastai-safe self-test: credential fixture survived redaction" >&2
    return 1
  fi
}

run_case 0 \
  'ordinary output https://example.test/?api_key=[REDACTED]&keep=1;tokenizer=kept-tokenizer' \
  "ordinary warning 'https://example.test/?keep=1;access_token=[REDACTED];tokenizer=kept-tokenizer'"
run_case 7 \
  'ordinary output https://example.test/?api_key=[REDACTED]&keep=1;tokenizer=kept-tokenizer' \
  "ordinary warning 'https://example.test/?keep=1;access_token=[REDACTED];tokenizer=kept-tokenizer'"

echo 'test-vastai-safe.sh: OK (redaction + exit-status preservation)'
