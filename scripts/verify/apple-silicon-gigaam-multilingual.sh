#!/usr/bin/env bash
set -euo pipefail

# Real-weight Apple verification is intentionally gated until a VAST-prepared
# GGUF and independent reference fixture are supplied. This worker must never
# turn an absent fixture or an unimplemented Metal path into PASS.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
die() { echo "apple-gigaam-multilingual: BLOCKED: $*" >&2; exit 2; }

if [[ "${1:-}" == --self-test ]]; then
  [[ $# == 1 ]] || die "--self-test accepts no other arguments"
  grep -Fq 'Metal/CUDA backend is not wired' "$ROOT/crates/vokra-models/src/gigaam/multilingual.rs" || die "Metal contract drift"
  grep -Fq 'AsrGigaamMultilingual' "$ROOT/crates/vokra-cli/src/engine.rs" || die "CLI route contract drift"
  grep -Fq 'OPEN_UNSUPPORTED' "$ROOT/scripts/publish/vast-ai/run-gigaam-multilingual-validation.sh" || die "publication status contract drift"
  if grep -En 'cargo (run|test)|git push|upload\.sh|publish-one\.sh' "$ROOT/scripts/verify/apple-silicon-gigaam-multilingual.sh" | grep -v 'grep -En' >/dev/null; then
    die "Apple worker must not execute or publish"
  fi
  echo "apple-silicon-gigaam-multilingual contract self-test: OK (CPU route/Metal block; no model execution)"
  exit 0
fi

[[ "$(uname -s)" == Darwin ]] || die "requires macOS"
[[ "$(uname -m)" == arm64 ]] || die "requires Apple Silicon"

if [[ -n "${GIGAAM_GGUF:-}" || -n "${GIGAAM_REFERENCE_DIR:-}" ]]; then
  die "real-weight Apple execution is OPEN_UNSUPPORTED: Metal learned-op route is not wired; CPU parity belongs to the VAST worker and no PASS is emitted"
fi
die "Apple CPU/Metal verification is OPEN_UNSUPPORTED: provide a reviewed VAST CPU parity result after Metal learned-op wiring"
