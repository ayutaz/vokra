#!/usr/bin/env bash
# check-converter-signoff.sh — every converter under crates/vokra-convert/src/
# models/ must be declared in scripts/publish/signoff_match.py, either with
# a §3.1 row it can produce or as an intentional main-repo exclusion.
#
# WHY THIS EXISTS
#
# Adding a new converter is a common step in the SoTA plan waves. A converter
# produces GGUF artifacts, and publishing a GGUF is a legal act — someone had
# to grant it. §3.1 is where that grant lives. Without this gate a scaffold
# converter can land, be forgotten, and then be publish-eligible via
# publish-one.sh the moment someone types the right --repo flag.
#
# WHAT THIS DOES
#
# Delegates to scripts/publish/signoff_match.py (which owns the explicit
# `converter stem → §3.1 row(s)` alias map) and turns its report into a
# shell exit code. Two directions of drift are caught:
#
#   * NEW converter on disk with no entry in signoff_match.CONVERTER_TO_SIGNOFF_ROWS
#     AND not in CONVERTER_NO_SIGNOFF_ROW. -> add the row + mapping.
#   * STALE entry in either map whose .rs file no longer exists.
#     -> remove the entry.
#
# Fail-closed: an untracked converter blocks CI. The gate is what keeps
# signoff_match.py honest as the tree evolves.
#
# Usage:
#   scripts/publish/check-converter-signoff.sh
#   scripts/publish/check-converter-signoff.sh --self-test

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
matcher="$repo_root/scripts/publish/signoff_match.py"
models_dir="$repo_root/crates/vokra-convert/src/models"
audit="$repo_root/docs/license-audit.md"

run_python() {
  uv run --no-project --python 3.12 python "$@"
}

if [[ "${1:-}" == "--self-test" ]]; then
  # Shell wrapper self-test. The Python module owns the exhaustive coverage
  # semantics (real maps vs. synthetic converters dir); we exercise the
  # bash -> uv-managed Python hand-off here so a broken invocation cannot
  # silently coast on a green matcher.
  fail=0

  # 1. Delegated matcher self-test — the semantic core.
  if ! run_python "$matcher" --self-test >/dev/null 2>&1; then
    echo "check-converter-signoff self-test: FAIL — signoff_match.py --self-test failed" >&2
    fail=1
  fi

  # 2. Real tree must currently pass. This is the "did a converter land
  #    without its signoff_match.py entry" tripwire, which is the point.
  if ! run_python "$matcher" --check-converters "$models_dir" --audit "$audit" >/dev/null; then
    echo "check-converter-signoff self-test: FAIL — real tree is not clean" >&2
    run_python "$matcher" --check-converters "$models_dir" --audit "$audit" >&2 || true
    fail=1
  fi

  # 3. Injected stray must fail. Build a scratch dir that mirrors the real
  #    tree via symlinks (so signoff_match.py's stale_mapped / stale_excluded
  #    checks stay happy), then plant a `never_declared_stem.rs`. The wrapper
  #    must exit non-zero and name the stray in its report.
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  mkdir -p "$tmp/models"
  for f in "$models_dir"/*.rs; do
    ln -s "$f" "$tmp/models/$(basename "$f")"
  done
  : >"$tmp/models/never_declared_stem.rs"
  if out="$(run_python "$matcher" --check-converters "$tmp/models" --audit "$audit" 2>&1)"; then
    echo "check-converter-signoff self-test: FAIL — undeclared converter was accepted" >&2
    fail=1
  elif ! grep -q "never_declared_stem" <<<"$out"; then
    echo "check-converter-signoff self-test: FAIL — undeclared converter was not named in the report" >&2
    printf '%s\n' "$out" >&2
    fail=1
  fi

  if [[ $fail -eq 0 ]]; then
    echo "check-converter-signoff self-test: OK (3 cases: matcher self-test + real tree + stray fail)"
    exit 0
  fi
  exit 1
fi

# Production run: enumerate the real tree.
run_python "$matcher" --check-converters "$models_dir" --audit "$audit"
