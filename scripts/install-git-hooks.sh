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
echo "  pre-commit (fast):  cargo fmt --check, forbidden symbols, zero-dependency invariant,"
echo "                      fixture-eol pins, pipefail/grep-q lint"
echo "  pre-push  (compiling):"
echo "    * compliance scanner sigpipe test (always, ~5 s)"
echo "    * cargo clippy --all-targets -- -D warnings"
echo "    * cargo test --workspace   (or cargo nextest run --workspace + cargo test --doc"
echo "                                if cargo-nextest is installed; ~60% faster)"
echo
echo "Iteration-speed fast-paths (silent skip forbidden — reason is always printed):"
echo "  * documentation-only diffs (docs/**, .github/**, *.md, *.yml, *.yaml, etc.)"
echo "    skip clippy + test; the compliance scanner still runs."
echo "  * VOKRA_HOOK_DEEP=1 forces the full check regardless of diff shape."
echo
echo "Optional (recommended): install cargo-nextest for the parallel test runner:"
echo "  cargo install cargo-nextest --locked"
echo
echo "Bypass once:  git commit --no-verify   /   git push --no-verify"
echo "Or per-run:   VOKRA_SKIP_HOOKS=1 git ..."
echo "Uninstall:    git config --unset core.hooksPath"
