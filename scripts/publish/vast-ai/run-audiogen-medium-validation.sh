#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT="$ROOT/tools/parity/audiogen_medium_reference"
INSPECTOR="$ROOT/tools/parity/audiogen_medium_inspect.py"
REFERENCE="$ROOT/tools/parity/audiogen_medium_dump_reference.py"
DEPENDENCY_AUDIT="$ROOT/tools/parity/audiogen_medium_reference/dependency_audit.py"
die(){ echo "audiogen-medium-validation: BLOCKED: $*" >&2; exit 2; }
usage(){ cat >&2 <<'EOF'
usage: run-audiogen-medium-validation.sh --expected-head <40-hex> \
       --approval-evidence <file> --approval-sha256 <64-hex> [--self-test]
EOF
}
if [[ "${1:-}" == --self-test ]]; then
  [[ $# == 1 ]] || die '--self-test accepts no arguments'
  grep -Fq '${TMPDIR:-/tmp}/vokra-audiogen-uv-cache' "$0"
  linux_cache="$(AUDIOGEN_UV_CACHE_DIR= TMPDIR=/var/tmp bash -c 'printf "%s" "${AUDIOGEN_UV_CACHE_DIR:-${TMPDIR:-/tmp}/vokra-audiogen-uv-cache}"')"
  [[ "$linux_cache" == /var/tmp/vokra-audiogen-uv-cache ]] || die 'Linux TMPDIR cache fallback drifted'
  grep -Fq 'metadata-only' "$INSPECTOR"
  grep -Fq 'model-free' "$INSPECTOR"
  grep -Fq 'PENDING_OWNER_APPROVAL' "$INSPECTOR"
  grep -Fq 'checkpoint payloads were intentionally not downloaded or loaded' "$INSPECTOR"
  grep -Fq 'NO_UPLOAD' "$INSPECTOR"
  grep -Fq 'LOUD_PARTIAL_FAIL_CLOSED' "$INSPECTOR"
  grep -Fq 'BLOCKED_MISSING_LOCK_INPUTS' "$DEPENDENCY_AUDIT"
  grep -Fq 'NO_LOCAL_EXECUTION' "$DEPENDENCY_AUDIT"
  download_word='snapshot'; download_suffix='_download'
  if grep -Fq "${download_word}${download_suffix}" "$0"; then die 'validation script contains a payload download route'; fi
  uv_word='uv'; sync_word='sy'; sync_suffix='nc'; project_prefix='--pro'; project_suffix='ject'
  if grep -Fq -- "${uv_word} ${sync_word}${sync_suffix}" "$0" || grep -Fq -- "${uv_word} run ${project_prefix}${project_suffix}" "$0"; then die 'validation self-test found an executable project route'; fi
  UV_CACHE_DIR="${AUDIOGEN_UV_CACHE_DIR:-${TMPDIR:-/tmp}/vokra-audiogen-uv-cache}" uv run --no-project --offline --python 3.12 python "$INSPECTOR" --self-test
  UV_CACHE_DIR="${AUDIOGEN_UV_CACHE_DIR:-${TMPDIR:-/tmp}/vokra-audiogen-uv-cache}" uv run --no-project --offline --python 3.12 python "$REFERENCE" --self-test
  UV_CACHE_DIR="${AUDIOGEN_UV_CACHE_DIR:-${TMPDIR:-/tmp}/vokra-audiogen-uv-cache}" uv run --no-project --offline --python 3.12 python "$DEPENDENCY_AUDIT" --self-test
  if "$0" --expected-head 0000000000000000000000000000000000000000 --expected-head 1111111111111111111111111111111111111111 >/dev/null 2>&1; then die 'duplicate --expected-head was accepted'; fi
  echo 'run-audiogen-medium-validation.sh self-test: OK'
  exit 0
fi
expected_head=''; approval_evidence=''; approval_sha256=''; seen_head=0; seen_approval=0; seen_approval_sha=0
while (($#)); do
  case "$1" in
    --expected-head) (( seen_head == 0 )) || die 'duplicate --expected-head'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head requires lowercase 40-hex'; expected_head="$2"; seen_head=1; shift 2 ;;
    --approval-evidence) (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--approval-evidence requires a nonempty path'; approval_evidence="$2"; seen_approval=1; shift 2 ;;
    --approval-sha256) (( seen_approval_sha == 0 )) || die 'duplicate --approval-sha256'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{64}$ ]] || die '--approval-sha256 requires lowercase 64-hex'; approval_sha256="$2"; seen_approval_sha=1; shift 2 ;;
    *) usage; die "unexpected argument: $1" ;;
  esac
done
(( seen_head == 1 )) || die '--expected-head is required'
(( seen_approval == 1 )) || die '--approval-evidence is required'
(( seen_approval_sha == 1 )) || die '--approval-sha256 is required'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'VAST requires Linux x86_64'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -f "$PROJECT/uv.lock" ]] || die 'dedicated AudioGen uv.lock absent; fail before downloads'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'VAST checkout must be clean'
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$expected_head" ]] || die 'checkout HEAD does not match --expected-head'
UV_CACHE_DIR="${AUDIOGEN_UV_CACHE_DIR:-${TMPDIR:-/tmp}/vokra-audiogen-uv-cache}" uv run --no-project --offline --python 3.12 python "$INSPECTOR" --validate-approval --approval-evidence "$approval_evidence" --approval-sha256 "$approval_sha256" --expected-head "$expected_head" >/dev/null || die 'external approval evidence is invalid'
die 'native AudioGen validation remains blocked pending authenticated native/composite parity'
