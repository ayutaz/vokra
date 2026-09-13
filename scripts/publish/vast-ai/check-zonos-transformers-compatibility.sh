#!/usr/bin/env bash
# Fail-closed Zonos Transformers compatibility gate.
# A lock pin is not API evidence: this gate blocks source/checkpoint work until
# an authorized VAST API smoke test records a compatible upstream result.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../tools/parity/zonos_v0_1_reference" && pwd)"
LOCK="$PROJECT_DIR/uv.lock"
PROBE="$PROJECT_DIR/transformers_compatibility.py"
BLOCKED="BLOCKED_UNVERIFIED_TRANSFORMERS_API_SMOKE"

lock_transformers_version() {
  awk '
    function finish() {
      if (name == "transformers") {
        entries++
        if (version == "5.10.4") exact++
      }
    }
    /^\[\[package\]\]$/ {
      finish()
      name = ""
      version = ""
      next
    }
    /^name = "/ {
      name = $0
      sub(/^name = "/, "", name)
      sub(/"$/, "", name)
    }
    /^version = "/ {
      version = $0
      sub(/^version = "/, "", version)
      sub(/"$/, "", version)
    }
    END {
      finish()
      printf "entries=%d exact=%d\n", entries + 0, exact + 0
    }
  ' "$1"
}

run_self_test() {
  [[ -f "$LOCK" && ! -L "$LOCK" ]] || {
    echo "compatibility gate self-test: lock is missing or symlinked" >&2
    return 1
  }
  [[ "$(lock_transformers_version "$LOCK")" == "entries=1 exact=1" ]] || {
    echo "compatibility gate self-test: tracked lock pin contract failed" >&2
    return 1
  }
  grep -Fq -- "$BLOCKED" "$0" || {
    echo "compatibility gate self-test: blocked marker contract is missing" >&2
    return 1
  }
  UV_CACHE_DIR="${ZONOS_UV_CACHE_DIR:-/tmp/vokra-zonos-uv-cache}" \
    uv run --no-cache --no-project --offline --python 3.12 python "$PROBE" --self-test || return 1
  echo "zonos Transformers compatibility gate self-test: PASS"
}

if [[ "${1:-}" == --self-test ]]; then
  [[ $# == 1 ]] || { echo "compatibility gate self-test: unexpected arguments" >&2; exit 1; }
  run_self_test
  exit 0
fi
if [[ $# == 0 ]]; then
  echo "$BLOCKED: external VAST API evidence is required" >&2
  exit 2
fi
[[ $# == 6 && "$1" == --evidence && "$3" == --evidence-sha256 && "$5" == --expected-head ]] || {
  echo "usage: check-zonos-transformers-compatibility.sh --evidence FILE --evidence-sha256 SHA --expected-head HEX40" >&2
  exit 1
}
evidence="$2"
evidence_sha="$4"
expected_head="$6"
[[ -f "$LOCK" && ! -L "$LOCK" ]] || { echo "$BLOCKED: lock is missing or symlinked" >&2; exit 2; }
summary="$(lock_transformers_version "$LOCK")"
if [[ "$summary" != "entries=1 exact=1" ]]; then
  echo "$BLOCKED: tracked Transformers lock contract failed ($summary)" >&2
  exit 2
fi
[[ -f "$PROBE" && ! -L "$PROBE" ]] || { echo "$BLOCKED: compatibility probe is missing or symlinked" >&2; exit 2; }
set +e
UV_CACHE_DIR="${ZONOS_UV_CACHE_DIR:-/tmp/vokra-zonos-uv-cache}" \
  uv run --no-cache --no-project --offline --python 3.12 python "$PROBE" \
    --validate-evidence --vokra-root "$ROOT" --expected-head "$expected_head" \
    --evidence "$evidence" --evidence-sha256 "$evidence_sha"
status=$?
set -e
if [[ "$status" != 0 ]]; then
  echo "$BLOCKED: external VAST API evidence failed strict validation" >&2
  exit 2
fi
echo "Zonos Transformers compatibility gate: PASS_COMPATIBLE (external evidence authenticated)" >&2
