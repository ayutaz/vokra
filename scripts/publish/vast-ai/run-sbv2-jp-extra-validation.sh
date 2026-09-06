#!/usr/bin/env bash
# SBV2 JP-Extra handoff.  The implementation lives in the established
# four-checkpoint worker; this entry point fixes the production language to
# Japanese without duplicating its acquisition, audit, and packet gates.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="$SCRIPT_DIR/run-sbv2-zh-parity.sh"

if [[ ! -x "$WORKER" ]]; then
  printf '%s\n' "SBV2 worker is missing or not executable: $WORKER" >&2
  exit 2
fi

if [[ "${1:-}" == --self-test ]]; then
  [[ "${2:-}" != --self-test ]] || {
    printf '%s\n' 'duplicate --self-test' >&2
    exit 2
  }
  grep -Fq -- '--language' "$WORKER"
  grep -Fq -- 'production_japanese_g2p=UNRESOLVED' "$WORKER"
  exec "$WORKER" --self-test
fi

exec "$WORKER" --language ja "$@"
