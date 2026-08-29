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
  FAKE_STDOUT='ordinary output https://example.test/?api_key=dummy-secret&keep=1;tokenizer=kept-tokenizer json={"extra_env": "JUPYTER_TOKEN=dummy-jupyter\\n", "TOKEN": "dummy-json", "tokenizer": "kept-tokenizer"}' \
    FAKE_STDERR="ordinary warning 'https://example.test/?keep=1;ACCESS_TOKEN=another-dummy;tokenizer=kept-tokenizer' kv CONTAINER_API_KEY=dummy-kv token_value=keep onstart='IMAGE_LOGIN_PASS=dummy-image'" \
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
     grep -Fq -- 'access_token=another-dummy' "$stderr_file" || \
     grep -Fq -- 'dummy-jupyter' "$stdout_file" || \
     grep -Fq -- 'dummy-json' "$stdout_file" || \
     grep -Fq -- 'dummy-kv' "$stderr_file" || \
     grep -Fq -- 'dummy-image' "$stderr_file"; then
    echo "vastai-safe self-test: credential fixture survived redaction" >&2
    return 1
  fi
}

run_case 0 \
  'ordinary output https://example.test/?api_key=[REDACTED]&keep=1;tokenizer=kept-tokenizer json={"extra_env": "JUPYTER_TOKEN=[REDACTED]", "TOKEN": "[REDACTED]", "tokenizer": "kept-tokenizer"}' \
  "ordinary warning 'https://example.test/?keep=1;ACCESS_TOKEN=[REDACTED];tokenizer=kept-tokenizer' kv CONTAINER_API_KEY=[REDACTED] token_value=keep onstart='IMAGE_LOGIN_PASS=[REDACTED]'"
run_case 7 \
  'ordinary output https://example.test/?api_key=[REDACTED]&keep=1;tokenizer=kept-tokenizer json={"extra_env": "JUPYTER_TOKEN=[REDACTED]", "TOKEN": "[REDACTED]", "tokenizer": "kept-tokenizer"}' \
  "ordinary warning 'https://example.test/?keep=1;ACCESS_TOKEN=[REDACTED];tokenizer=kept-tokenizer' kv CONTAINER_API_KEY=[REDACTED] token_value=keep onstart='IMAGE_LOGIN_PASS=[REDACTED]'"

run_json_key_cases() {
  local key actual_status
  for key in api_key API_KEY apikey APIKEY api-key API-KEY container_api_key CONTAINER_API_KEY \
    access_token ACCESS_TOKEN auth_token AUTH_TOKEN client_secret CLIENT_SECRET \
    hf_token HF_TOKEN jupyter_token JUPYTER_TOKEN token TOKEN secret SECRET \
    password PASSWORD docker_login_pass DOCKER_LOGIN_PASS image_login_pass IMAGE_LOGIN_PASS; do
    set +e
    # shellcheck disable=SC2016
    FAKE_STDOUT="{\"$key\":\"secret-$key\"}" FAKE_STDERR="json warning {\"$key\": \"secret-$key\"}" \
      VASTAI_BIN=sh "$wrapper" -c \
      'printf "%s\\n" "$FAKE_STDOUT"; printf "%s\\n" "$FAKE_STDERR" >&2' \
      >"$stdout_file" 2>"$stderr_file"
    actual_status=$?
    set -e
    [[ "$actual_status" == 0 ]] || { echo "vastai-safe self-test: $key status changed" >&2; return 1; }
    grep -Fqx -- "{\"$key\":\"[REDACTED]\"}" "$stdout_file" || return 1
    grep -Fqx -- "json warning {\"$key\": \"[REDACTED]\"}" "$stderr_file" || return 1
    if grep -Fq -- "secret-$key" "$stdout_file" "$stderr_file"; then
      return 1
    fi
  done
}

run_json_key_cases

run_python_dict_key_cases() {
  local key actual_status
  for key in api_key API_KEY apikey APIKEY api-key API-KEY container_api_key CONTAINER_API_KEY \
    access_token ACCESS_TOKEN auth_token AUTH_TOKEN client_secret CLIENT_SECRET \
    hf_token HF_TOKEN jupyter_token JUPYTER_TOKEN instance_api_key INSTANCE_API_KEY \
    token TOKEN secret SECRET password PASSWORD docker_login_pass DOCKER_LOGIN_PASS \
    image_login_pass IMAGE_LOGIN_PASS; do
    set +e
    # shellcheck disable=SC2016
    FAKE_STDOUT="{'$key': 'secret-$key'}" FAKE_STDERR="python warning {'$key': 'secret-$key'}" \
      VASTAI_BIN=sh "$wrapper" -c \
      'printf "%s\\n" "$FAKE_STDOUT"; printf "%s\\n" "$FAKE_STDERR" >&2' \
      >"$stdout_file" 2>"$stderr_file"
    actual_status=$?
    set -e
    [[ "$actual_status" == 0 ]] || { echo "vastai-safe self-test: Python dict $key status changed" >&2; return 1; }
    grep -Fqx -- "{'$key': '[REDACTED]'}" "$stdout_file" || return 1
    grep -Fqx -- "python warning {'$key': '[REDACTED]'}" "$stderr_file" || return 1
    if grep -Fq -- "secret-$key" "$stdout_file" "$stderr_file"; then
      echo "vastai-safe self-test: Python dict $key survived redaction" >&2
      return 1
    fi

    set +e
    # Also cover dict-like diagnostics that omit quotes around the field name.
    # shellcheck disable=SC2016
    FAKE_STDOUT="$key: 'secret-$key'" FAKE_STDERR="python warning $key: 'secret-$key'" \
      VASTAI_BIN=sh "$wrapper" -c \
      'printf "%s\\n" "$FAKE_STDOUT"; printf "%s\\n" "$FAKE_STDERR" >&2' \
      >"$stdout_file" 2>"$stderr_file"
    actual_status=$?
    set -e
    [[ "$actual_status" == 0 ]] || { echo "vastai-safe self-test: Python field $key status changed" >&2; return 1; }
    grep -Fqx -- "$key: '[REDACTED]'" "$stdout_file" || return 1
    grep -Fqx -- "python warning $key: '[REDACTED]'" "$stderr_file" || return 1
    if grep -Fq -- "secret-$key" "$stdout_file" "$stderr_file"; then
      echo "vastai-safe self-test: Python field $key survived redaction" >&2
      return 1
    fi
  done
}

run_python_dict_key_cases

echo 'test-vastai-safe.sh: OK (redaction + exit-status preservation)'
