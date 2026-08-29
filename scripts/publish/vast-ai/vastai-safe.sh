#!/usr/bin/env bash
# vastai-safe.sh — invoke the Vast CLI without echoing credentials.
#
# The CLI has, in some failure paths, included its API URL or credential fields
# in an exception. Capture both streams, redact credential-valued URL query and
# JSON/key-value fields, then replay them while returning the CLI's original
# exit status.
#
# Usage:
#   scripts/publish/vast-ai/vastai-safe.sh search offers ...
#   VASTAI_BIN=/path/to/vastai scripts/publish/vast-ai/vastai-safe.sh ...

set -euo pipefail
set +x

VASTAI_BIN="${VASTAI_BIN:-vastai}"

if (($# == 1)) && [[ "$1" == --self-test ]]; then
  test_script_dir="$(dirname -- "$0")"
  exec "$test_script_dir/test-vastai-safe.sh"
fi

usage() {
  cat >&2 <<'EOF'
usage: vastai-safe.sh <vastai arguments...>

Runs ${VASTAI_BIN:-vastai} and redacts credential-valued URL query/JSON fields
from stdout and stderr. The wrapped command's exit status is preserved.
Set VASTAI_BIN to a different executable for offline tests or a pinned CLI.

Run the offline contract test with:
  vastai-safe.sh --self-test
EOF
}

if (($# == 0)); then
  usage
  exit 64
fi

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/vokra-vastai-safe.XXXXXX")"
stdout_file="$tmp_dir/stdout"
stderr_file="$tmp_dir/stderr"

# shellcheck disable=SC2329
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT HUP INT TERM

# Query parameter names are intentionally explicit. The separators prevent a
# value such as `tokenizer=` from being treated as the `token=` credential.
# `;` is accepted because some URL parsers allow it as a query separator.
redact_stream() {
  # Keep the key and JSON punctuation intact. The JSON value expression also
  # accepts escaped characters (including an escaped quote), while the exact
  # key list avoids redacting near-matches such as `tokenizer`.
  local credential_keys='(api_key|API_KEY|apikey|APIKEY|api-key|API-KEY|container_api_key|CONTAINER_API_KEY|access_token|ACCESS_TOKEN|auth_token|AUTH_TOKEN|client_secret|CLIENT_SECRET|hf_token|HF_TOKEN|jupyter_token|JUPYTER_TOKEN|token|TOKEN|secret|SECRET|password|PASSWORD|docker_login_pass|DOCKER_LOGIN_PASS|image_login_pass|IMAGE_LOGIN_PASS)'
  local json_pattern="s/(\"${credential_keys}\"[[:space:]]*:[[:space:]]*\")([^\"\\\\]|\\\\.)*(\")/\\1[REDACTED]\\4/g"
  local query_pattern="s/((^|[?&;])${credential_keys}=)[^&;[:space:]\"'<>)]*/\\1[REDACTED]/g"
  local kv_pattern="s/((^|[[:space:],;{\"'])${credential_keys}[[:space:]]*=[[:space:]]*)[^&;[:space:]\"'<>),}]*/\\1[REDACTED]/g"
  sed -E -e "$json_pattern" -e "$query_pattern" -e "$kv_pattern"
}

command_status=0
if "$VASTAI_BIN" "$@" >"$stdout_file" 2>"$stderr_file"; then
  command_status=0
else
  command_status=$?
fi

output_status=0
redact_stream <"$stdout_file" || output_status=$?
redact_stream <"$stderr_file" >&2 || output_status=$?

if ((output_status != 0)); then
  # A local output failure is distinct from the wrapped command's status.
  # This should only be reachable when the temporary files cannot be read.
  exit 125
fi
exit "$command_status"
