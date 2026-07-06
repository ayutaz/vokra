#!/usr/bin/env bash
# vokra Python bindings — CI drift-check wrapper.
#
# Copyright The Vokra Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Reruns `gen-py-bindings.py --check` to detect stale ctypes declarations vs
# the canonical C ABI header `include/vokra.h`. This exists as a separate
# entry point so `.github/workflows/ci.yml` can wire it into the required
# checks alongside `scripts/gen-c-abi.sh --check` (M2-12 T07 / plan D12).
#
# Contract:
#   - exit 0: on-disk `bindings/python/src/vokra/_bindings.py` matches the
#     generator output byte-for-byte (no drift).
#   - exit 1: DRIFT detected (unified diff printed to stderr by the codegen)
#     OR the file is missing entirely. Fix by running:
#         python3 bindings/python/scripts/gen-py-bindings.py
#     and committing the regenerated `_bindings.py`.
#   - exit 2: usage error, or `include/vokra.h` / codegen / python3 missing
#     (upstream `scripts/gen-c-abi.sh` was not run) — surface upstream
#     failure, do not paper over.
#
# Usage:
#   bash bindings/python/scripts/check-py-bindings.sh --check
#
# The `--check` flag is mandatory (this script is *always* a check) to keep
# the CLI symmetric with `scripts/gen-c-abi.sh --check` (ADR-0003 §3-a) and
# to reject typos in CI YAML rather than silently no-op.
#
# Zero non-stdlib deps (NFR-DS-02): pure bash + python3 (stdlib argparse /
# difflib in the codegen). No pip install, no Rust rebuild, no root
# Cargo.lock touch.

set -euo pipefail

if [[ "${1:-}" != "--check" ]]; then
    echo "usage: $(basename "$0") --check" >&2
    echo "  Runs gen-py-bindings.py --check to enforce that _bindings.py is up to date." >&2
    exit 2
fi
shift

# Reject stray extra args to avoid silent misconfiguration.
if [[ $# -ne 0 ]]; then
    echo "check-py-bindings: unexpected argument: $1" >&2
    echo "usage: $(basename "$0") --check" >&2
    exit 2
fi

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${HERE}/../../.." && pwd)"
CODEGEN="${HERE}/gen-py-bindings.py"
HEADER="${REPO_ROOT}/include/vokra.h"

if [[ ! -f "${CODEGEN}" ]]; then
    echo "check-py-bindings: codegen missing: ${CODEGEN}" >&2
    exit 2
fi

if [[ ! -f "${HEADER}" ]]; then
    echo "check-py-bindings: header missing: ${HEADER}" >&2
    echo "  Run scripts/gen-c-abi.sh first (upstream cbindgen step)." >&2
    exit 2
fi

# Locate python3 without hard-coding a path (macOS Xcode CLT / Linux distro
# / CI runner all ship it on PATH). Do not fall back to `python` — Python 2
# would silently break argparse's `--check`. Honour ${PYTHON} override for
# venv / pyenv workflows (matches the pre-existing convention).
PY="${PYTHON:-python3}"
if ! command -v "${PY}" >/dev/null 2>&1; then
    echo "check-py-bindings: interpreter not found on PATH: ${PY}" >&2
    exit 2
fi

# Delegate the actual byte-for-byte comparison to the codegen's --check
# mode. It prints a unified diff on drift; we forward its exit code so CI
# fails loudly (exit 1 = drift/missing, exit 2 = header not found — the
# latter already handled above).
exec "${PY}" "${CODEGEN}" --check --header "${HEADER}"
