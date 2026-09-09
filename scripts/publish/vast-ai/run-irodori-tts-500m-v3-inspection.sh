#!/usr/bin/env bash
# Irodori inspection is intentionally stopped before any source/model fetch.
# The authenticated source lock contains a forbidden native/audio closure.
set -euo pipefail

SOURCE_REPO='https://github.com/Aratako/Irodori-TTS.git'
SOURCE_REV='8224dafb46d0aba89209a8f905f1cb7e3299d9c1'
SOURCE_LOCK_SHA256='8175adbb9ad7ae77d1f048344343a63876e57c333b659314bcc054230b5b3e6c'
CODEC_REPO='Aratako/Semantic-DACVAE-Japanese-32dim'
CODEC_REV='47376ee24834d7a05a48ebabfe3cde29b3c5e214'
TOKENIZER_REPO='llm-jp/llm-jp-3-150m'
TOKENIZER_REV='b112feef602fff752e4dac4c30af6a2c2fa41c7a'
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
UV_GATE_CMD=(uv run --no-cache --no-project --offline --python 3.12 python)

die() { echo "irodori inspection: $*" >&2; exit 2; }

self_test() {
  cd "$ROOT"
  grep -Fq 'INSPECTION_ONLY' tools/parity/irodori_inspect.py
  grep -Fq 'NO_UPLOAD' tools/parity/irodori_inspect.py
  grep -Fq 'resolved_revision' tools/parity/irodori_inspect.py
  grep -Fq -- '--validate-approval' "$0"
  grep -Fq 'canonical' "$0"
  grep -Fq 'CODEC_REPO' "$0" && grep -Fq 'CODEC_REV=' "$0" && grep -Fq 'SOURCE_REV' "$0" && grep -Fq 'TOKENIZER_REV=' "$0"
  grep -Fq 'uv run --no-cache --no-project --offline --python 3.12' "$0"
  grep -Fq "$SOURCE_LOCK_SHA256" "$0"
  if grep -En '(^|[;&|][[:space:]]*)(curl|wget|snapshot_download|git[[:space:]]+push|vokra-cli[[:space:]]+convert)([[:space:]]|$)' "$0" >/dev/null; then
    echo 'irodori worker self-test: forbidden download/conversion/publication marker' >&2
    return 1
  fi
  UV_NO_CACHE=1 "${UV_GATE_CMD[@]}" tools/parity/irodori_inspect.py --self-test
  UV_NO_CACHE=1 "${UV_GATE_CMD[@]}" tools/parity/irodori_text_block_dump_reference.py --self-test
  echo 'irodori worker self-test: ok'
}

if [[ ${1:-} == --self-test ]]; then
  [[ $# == 1 ]] || die '--self-test accepts no arguments'
  self_test
  exit 0
fi

expected_head=''; approval_evidence=''; approval_sha256=''; seen_head=0; seen_approval=0; seen_sha=0
while (($#)); do
  case "$1" in
    --expected-head) (( seen_head == 0 )) || die 'duplicate --expected-head'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head requires lowercase 40-hex'; expected_head="$2"; seen_head=1; shift 2 ;;
    --approval-evidence) (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; [[ $# -ge 2 && "$2" != -* && "$2" == /* ]] || die '--approval-evidence requires an absolute path'; approval_evidence="$2"; seen_approval=1; shift 2 ;;
    --approval-sha256) (( seen_sha == 0 )) || die 'duplicate --approval-sha256'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{64}$ ]] || die '--approval-sha256 requires lowercase 64-hex'; approval_sha256="$2"; seen_sha=1; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done
(( seen_head == 1 && seen_approval == 1 && seen_sha == 1 )) || die 'expected HEAD and external approval are required'
cd "$ROOT"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || die 'worktree is not clean'
[[ "$(git rev-parse HEAD)" == "$expected_head" ]] || die 'checkout HEAD does not match --expected-head'
[[ "$approval_evidence" != */ && "$approval_evidence" != *'/./'* && "$approval_evidence" != *'/../'* && -f "$approval_evidence" && ! -L "$approval_evidence" ]] || die 'approval path is not canonical'

# This gate must stay before git clone, HF access, and every dependency-aware
# command.  It uses only Python 3.12 stdlib and intentionally exits 2.
UV_NO_CACHE=1 "${UV_GATE_CMD[@]}" tools/parity/irodori_inspect.py --validate-approval --approval-evidence "$approval_evidence" --approval-sha256 "$approval_sha256" --expected-head "$expected_head" >/dev/null || die 'external approval evidence is invalid'
set +e
gate_output="$(UV_NO_CACHE=1 "${UV_GATE_CMD[@]}" tools/parity/irodori_inspect.py --dependency-gate 2>&1)"
gate_status=$?
set -e
printf '%s\n' "$gate_output" >&2
[[ $gate_status == 2 ]] || die "dependency gate returned unexpected status: $gate_status"
echo "irodori inspection: BLOCKED by forbidden closure (source=$SOURCE_REPO@$SOURCE_REV lock=$SOURCE_LOCK_SHA256 codec=$CODEC_REPO@$CODEC_REV tokenizer=$TOKENIZER_REPO@$TOKENIZER_REV); no source/model download or sync attempted" >&2
exit 2
