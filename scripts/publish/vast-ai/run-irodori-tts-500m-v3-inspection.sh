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

die() { echo "irodori inspection: $*" >&2; exit 2; }

self_test() {
  grep -Fq 'INSPECTION_ONLY' tools/parity/irodori_inspect.py
  grep -Fq 'NO_UPLOAD' tools/parity/irodori_inspect.py
  grep -Fq 'resolved_revision' tools/parity/irodori_inspect.py
  grep -Fq 'CODEC_REPO' "$0" && grep -Fq 'CODEC_REV=' "$0" && grep -Fq 'SOURCE_REV' "$0" && grep -Fq 'TOKENIZER_REV=' "$0"
  grep -Fq 'uv run --no-project --python 3.12' "$0"
  grep -Fq "$SOURCE_LOCK_SHA256" "$0"
  if grep -En '(^|[;&|][[:space:]]*)(curl|wget|snapshot_download|git[[:space:]]+push|vokra-cli[[:space:]]+convert)([[:space:]]|$)' "$0" >/dev/null; then
    echo 'irodori worker self-test: forbidden download/conversion/publication marker' >&2
    return 1
  fi
  UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/vokra-uv-cache}" uv run --no-project --python 3.12 \
    python tools/parity/irodori_inspect.py --self-test
  echo 'irodori worker self-test: ok'
}

if [[ ${1:-} == --self-test ]]; then
  [[ $# == 1 ]] || die '--self-test accepts no arguments'
  self_test
  exit 0
fi
[[ ${VOKRA_PUBLISH_ON_VAST:-} == 1 ]] || die 'set VOKRA_PUBLISH_ON_VAST=1 on the disposable VAST host'
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || die 'requires Linux x86_64 VAST host'
command -v uv >/dev/null || die 'uv is required'

# This gate must stay before git clone, HF access, and every dependency-aware
# command.  It uses only Python 3.12 stdlib and intentionally exits 2.
set +e
gate_output="$(uv run --no-project --python 3.12 python tools/parity/irodori_inspect.py --dependency-gate 2>&1)"
gate_status=$?
set -e
printf '%s\n' "$gate_output" >&2
[[ $gate_status == 2 ]] || die "dependency gate returned unexpected status: $gate_status"
echo "irodori inspection: BLOCKED by forbidden closure (source=$SOURCE_REPO@$SOURCE_REV lock=$SOURCE_LOCK_SHA256 codec=$CODEC_REPO@$CODEC_REV tokenizer=$TOKENIZER_REPO@$TOKENIZER_REV); no source/model download or sync attempted" >&2
exit 2
