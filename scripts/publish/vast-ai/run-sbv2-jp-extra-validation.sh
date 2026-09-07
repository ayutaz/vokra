#!/usr/bin/env bash
# SBV2 JP-Extra handoff.  The implementation lives in the established
# four-checkpoint worker; this entry point fixes the production language to
# Japanese without duplicating its acquisition, audit, and packet gates.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="$SCRIPT_DIR/run-sbv2-zh-parity.sh"
CONTRACT_VALIDATOR="$(cd "$SCRIPT_DIR/../../.." && pwd)/tools/parity/sbv2_jp_extra/validate_contract.py"

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
  grep -Fq -- '--g2p-contract' "${BASH_SOURCE[0]}"
  [[ -x "$SCRIPT_DIR/run-sbv2-jp-extra-g2p-contract.sh" ]]
  "$SCRIPT_DIR/run-sbv2-jp-extra-g2p-contract.sh" --self-test
  exec "$WORKER" --self-test
fi

CONTRACT=""
FORWARDED=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --g2p-contract)
      [[ -z "$CONTRACT" && $# -ge 2 ]] || { printf '%s\n' 'duplicate or missing --g2p-contract' >&2; exit 2; }
      CONTRACT="$2"
      shift 2
      ;;
    *)
      FORWARDED+=("$1")
      shift
      ;;
  esac
done
[[ -n "$CONTRACT" ]] || { printf '%s\n' '--g2p-contract is required; generate it on VAST first' >&2; exit 2; }
[[ "$CONTRACT" == /* && -f "$CONTRACT" && ! -L "$CONTRACT" ]] || { printf '%s\n' 'G2P contract must be an absolute regular file' >&2; exit 2; }
[[ -x "$CONTRACT_VALIDATOR" ]] || { printf '%s\n' 'G2P contract validator is missing' >&2; exit 2; }
uv run --no-project --offline --python 3.12 python "$CONTRACT_VALIDATOR" --contract "$CONTRACT"
sidecar="$CONTRACT.sha256"
[[ -f "$sidecar" && ! -L "$sidecar" ]] || { printf '%s\n' 'G2P contract SHA-256 sidecar is missing' >&2; exit 2; }
(cd "$(dirname "$CONTRACT")" && sha256sum --strict --check "$(basename "$sidecar")")
exec "$WORKER" --language ja "${FORWARDED[@]}"
