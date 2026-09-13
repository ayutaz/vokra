#!/usr/bin/env bash
# Fail-closed Zonos Transformers compatibility gate.
# A lock pin is not API evidence: this gate blocks source/checkpoint work until
# an authorized VAST API smoke test records a compatible upstream result.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../tools/parity/zonos_v0_1_reference" && pwd)"
LOCK="$PROJECT_DIR/uv.lock"
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
  echo "zonos Transformers compatibility gate self-test: PASS"
}

if [[ "${1:-}" == --self-test ]]; then
  [[ $# == 1 ]] || { echo "compatibility gate self-test: unexpected arguments" >&2; exit 1; }
  run_self_test
  exit 0
fi
[[ $# == 0 ]] || { echo "usage: check-zonos-transformers-compatibility.sh [--self-test]" >&2; exit 1; }
[[ -f "$LOCK" && ! -L "$LOCK" ]] || { echo "$BLOCKED: lock is missing or symlinked" >&2; exit 2; }
summary="$(lock_transformers_version "$LOCK")"
if [[ "$summary" != "entries=1 exact=1" ]]; then
  echo "$BLOCKED: tracked Transformers lock contract failed ($summary)" >&2
  exit 2
fi
echo "$BLOCKED: transformers==5.10.4 is security-fixed but upstream Zonos API compatibility is unverified" >&2
exit 2
