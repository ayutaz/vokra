#!/usr/bin/env bash
# VAST-only no-upload Irodori validation worker.  The authenticated source
# lock is inspection evidence only; its forbidden native/audio closure blocks
# before any source lock sync, import, or model execution.
set -euo pipefail

ROOT_DIR="${VOKRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
SOURCE_REV='8224dafb46d0aba89209a8f905f1cb7e3299d9c1'
SOURCE_LOCK_SHA256='8175adbb9ad7ae77d1f048344343a63876e57c333b659314bcc054230b5b3e6c'
# IRODORI_NATIVE_BINDER_ACCEPTED is intentionally unavailable while the
# forbidden closure remains unresolved.

die() { echo "irodori validation: $*" >&2; exit 2; }

run_dependency_gate() {
  local gate_output gate_status
  set +e
  gate_output="$(UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$ROOT_DIR/tools/parity/irodori_inspect.py" --dependency-gate 2>&1)"
  gate_status=$?
  set -e
  printf '%s\n' "$gate_output" >&2
  [[ $gate_status == 2 ]] || die "dependency gate returned unexpected status: $gate_status"
  return 1
}

self_test() {
  grep -Fq 'REFERENCE_BLOCKED' tools/parity/irodori_500m_v3_dump_reference.py
  grep -Fq 'NO_UPLOAD' tools/parity/irodori_500m_v3_dump_reference.py
  grep -Fq 'IRODORI_NATIVE_BINDER_ACCEPTED' "$0"
  grep -Fq -- '--no-project' "$0"
  grep -Fq 'uv run --no-cache --no-project --offline --python 3.12' "$0"
  grep -Fq "$SOURCE_REV" "$0"
  grep -Fq "$SOURCE_LOCK_SHA256" "$0"
  if grep -En '(^|[;&|][[:space:]]*)(curl|wget|snapshot_download|git[[:space:]]+push|vokra-cli[[:space:]]+convert)([[:space:]]|$)' "$0" >/dev/null; then
    echo 'irodori validation self-test: forbidden download/conversion/publication marker' >&2
    return 1
  fi
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$ROOT_DIR/tools/parity/irodori_500m_v3_dump_reference.py" --self-test >/dev/null
  echo 'irodori validation worker self-test: OK'
}

if [[ ${1:-} == --self-test ]]; then
  [[ $# == 1 ]] || die '--self-test accepts no arguments'
  self_test
  exit 0
fi
run_dependency_gate || die 'Irodori dependency/native closure is unresolved; no source/model sync or execution is permitted'
[[ ${VOKRA_PUBLISH_ON_VAST:-} == 1 ]] || die 'set VOKRA_PUBLISH_ON_VAST=1 on disposable VAST'
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || die 'requires Linux x86_64 VAST host'
command -v uv >/dev/null || die 'uv is required'

echo "irodori validation: BLOCKED by forbidden closure (source=$SOURCE_REV lock=$SOURCE_LOCK_SHA256); no source/model sync or execution attempted" >&2
exit 2
