#!/usr/bin/env bash
# vastai-safe.sh — invoke the Vast CLI without echoing query credentials.
#
# The CLI has, in some failure paths, included its API URL in a requests
# exception. Capture both streams, redact credential-valued query parameters,
# then replay them while returning the CLI's original exit status.
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

Runs ${VASTAI_BIN:-vastai} and redacts credential-valued URL query parameters
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
  local pattern="s/((^|[?&;])(api_key|apikey|api-key|access_token|auth_token|client_secret|hf_token|token|secret|password)=)[^&;[:space:]\"'<>)]*/\\1[REDACTED]/g"
  sed -E "$pattern"
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
