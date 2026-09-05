#!/usr/bin/env bash
set -euo pipefail

# Readiness gate only.  No synthetic or LLM-only artifact may be reported as
# Apple CPU/Metal support before the complete composite has passed VAST parity.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REFERENCE="$ROOT/tools/parity/cosyvoice3_dump_reference.py"
die(){ echo "cosyvoice3-apple: ERROR: $*" >&2; exit 2; }
COSYVOICE3_SELF_TEST_TMP=""
# shellcheck disable=SC2329 # Invoked by the EXIT trap below.
cleanup_self_test() {
  [[ -n "$COSYVOICE3_SELF_TEST_TMP" ]] && rm -rf -- "$COSYVOICE3_SELF_TEST_TMP"
}
self_test(){
  local fail=0 token tmp valid duplicate link
  for token in 'AUTHENTICATED_REFERENCE_EVIDENCE' 'REFERENCE_ERROR' 'NOT_IMPLEMENTED_FAIL_CLOSED' 'sample_rate' 'flow_rand_noise_full' 'native_status' 'comparison_status' 'NO_UPLOAD' 'manifest.json' 'approved_lock_status'; do
    grep -Fq -- "$token" "$REFERENCE" "$0" || { echo "missing contract: $token" >&2; fail=1; }
  done
  if grep -En 'git[[:space:]]+push|upload\.sh|publish-one\.sh|--push|--upload' "$0" | grep -v 'grep -En' >/dev/null; then fail=1; fi
  # This is stdlib-only.  Never invoke the dedicated project: its inventory
  # is intentionally blocked and a frozen lock could still create/sync an
  # environment before the affirmative license gate exists.
  if grep -Eq '^[[:space:]]*UV_CACHE_DIR=.*uv run .*--project .*cosyvoice3_reference' "$0"; then
    echo 'self-test found dedicated CosyVoice3 environment use' >&2; fail=1
  fi
  tmp="$(mktemp -d)"
  COSYVOICE3_SELF_TEST_TMP="$tmp"
  trap cleanup_self_test EXIT
  valid="$tmp/valid.json"
  duplicate="$tmp/duplicate.json"
  link="$tmp/link.json"
  printf '{"status":"BLOCKED","runtime_status":"NOT_IMPLEMENTED_FAIL_CLOSED","native_status":"BLOCKED","metal_status":"BLOCKED_BY_CPU","publication":"NO_UPLOAD"}\n' >"$valid"
  printf '{"status":"BLOCKED","status":"BLOCKED"}\n' >"$duplicate"
  ln -s "$valid" "$link"
  if validate_manifest "$duplicate" >/dev/null 2>&1; then fail=1; fi
  if validate_manifest "$link" >/dev/null 2>&1; then fail=1; fi
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$REFERENCE" --self-test || fail=1
  (( fail == 0 )) || return 1
  echo 'apple-silicon-cosyvoice3.sh self-test: OK'
}

validate_manifest() {
  local manifest="$1"
  [[ -f "$manifest" && ! -L "$manifest" && -s "$manifest" ]] || { echo 'CosyVoice3 evidence manifest is missing, symlinked, or empty' >&2; return 2; }
  # Status and lock approval are checked in a general stdlib-only environment.
  # A blocked dedicated project is never installed or imported here.
  if UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$manifest" <<'PY'
import json
import sys

def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

path = sys.argv[1]
try:
    with open(path, encoding="utf-8") as stream:
        manifest = json.load(stream, object_pairs_hook=reject_duplicates)
except (OSError, ValueError) as error:
    raise SystemExit(f"{path}: malformed or duplicate-key JSON: {error}") from error
required = {
    "status": "BLOCKED",
    "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
    "native_status": "BLOCKED",
    "metal_status": "BLOCKED_BY_CPU",
    "publication": "NO_UPLOAD",
}
for key, expected in required.items():
    if manifest.get(key) != expected:
        raise SystemExit(f"{path}: {key} is not fail-closed")
project = manifest.get("project")
if project is not None:
    if not isinstance(project, dict):
        raise SystemExit(f"{path}: project identity is malformed")
    if project.get("approved_lock_status") != "APPROVED_CPU_REFERENCE_LOCK":
        raise SystemExit(f"{path}: dedicated lock is not explicitly approved; staying on general tools env")
PY
  then
    :
  else
    echo 'CosyVoice3 evidence manifest JSON is malformed or not fail-closed' >&2
    return 2
  fi
}

if [[ "${1:-}" == --self-test ]]; then [[ $# == 1 ]] || die '--self-test accepts no arguments'; self_test; exit 0; fi
if [[ $# == 1 ]]; then
  validate_manifest "$1"
  die 'CosyVoice3 CPU/Metal e2e remains blocked; manifest validation used the general tools environment'
fi
[[ $# == 0 ]] || die 'usage: apple-silicon-cosyvoice3.sh [MANIFEST] | --self-test'
[[ "$(uname -s)" == Darwin ]] || die 'Apple Silicon worker requires macOS'
[[ "$(uname -m)" == arm64 ]] || die 'Apple Silicon worker requires arm64'
die 'CosyVoice3 CPU/Metal e2e is blocked until complete composite VAST reference and native parity evidence exist'
