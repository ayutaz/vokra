#!/usr/bin/env bash
# install-git-hooks.sh — activate Vokra's version-controlled git hooks.
#
# Vokra keeps its git hooks under version control in .githooks/ and turns them
# on with `core.hooksPath` — no external hook manager (husky / lefthook /
# pre-commit) and therefore no added dependency, consistent with the
# zero-dependency policy (NFR-DS-02). Run this once per clone.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

git config core.hooksPath .githooks

echo "Installed: core.hooksPath -> .githooks"
echo "  pre-commit (fast):  cargo fmt --check, forbidden symbols, zero-dependency invariant"
echo "  pre-push  (full):   cargo clippy -D warnings, cargo test --workspace"
echo
echo "Bypass once:  git commit --no-verify   /   git push --no-verify"
echo "Or per-run:   VOKRA_SKIP_HOOKS=1 git ..."
echo "Uninstall:    git config --unset core.hooksPath"
