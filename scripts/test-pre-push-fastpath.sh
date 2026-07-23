#!/usr/bin/env bash
# scripts/test-pre-push-fastpath.sh
#
# Regression test for the .githooks/pre-push docs-only fast-path.
#
# Sources .githooks/lib-fastpath.sh (production classifier) and drives
# `is_docs_only_diff` per case by shadowing `git diff --name-only` and
# `diff_base` in a subshell. No real git activity, no cargo — the test runs
# in milliseconds and can sit inside `scripts/verify.sh` or a CI leg without
# added cost.
#
# Cases exercise both directions of the classifier:
#   * docs-only inputs must land on fast-path (return 0)
#   * anything Rust-adjacent must land on deep-path (return 1)
#   * defensive inputs (empty diff, VOKRA_HOOK_DEEP=1) must land on deep-path

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

pass=0
fail=0

# One test case: name, expected verdict ("fast" | "deep"), diff-line list
# (newline-separated, may be empty), optional env override (e.g.
# "VOKRA_HOOK_DEEP=1"). The diff line list is fed to a fake
# `git diff --name-only` inside a subshell so the real repo is not touched.
run_case() {
    local name="$1"
    local expected="$2"
    local files="$3"
    local envvar="${4:-}"

    local verdict
    verdict=$(
        set +e
        if [ -n "$envvar" ]; then
            export "${envvar?}"
        fi
        # shellcheck source=../.githooks/lib-fastpath.sh
        source "$ROOT/.githooks/lib-fastpath.sh"

        # Shadow the two calls the classifier makes.
        FAKE_FILES="$files"
        diff_base() { echo "fake-base"; }
        git() {
            if [ "${1:-}" = "diff" ] && [ "${2:-}" = "--name-only" ]; then
                if [ -n "$FAKE_FILES" ]; then
                    printf '%s\n' "$FAKE_FILES"
                fi
            else
                command git "$@"
            fi
        }

        if is_docs_only_diff; then
            echo "fast"
        else
            echo "deep"
        fi
    )

    if [ "$verdict" = "$expected" ]; then
        pass=$((pass + 1))
        printf 'OK   %-52s → %s\n' "$name" "$verdict"
    else
        fail=$((fail + 1))
        printf 'FAIL %-52s expected=%s got=%s\n' "$name" "$expected" "$verdict"
    fi
}

echo "test-pre-push-fastpath: 17 cases"
echo

# --- FAST-PATH cases (all inputs are docs-shape) ---
run_case "single markdown file" \
    "fast" \
    "CLAUDE.md"

run_case "several docs files" \
    "fast" \
    "$(printf 'docs/handoff/x-10.md\ndocs/license-audit.md\n.github/workflows/ci.yml\n')"

run_case "yaml catalog only" \
    "fast" \
    ".github/pins.yaml"

run_case "generated C header (include/*.h)" \
    "fast" \
    "include/vokra.h"

run_case "root dotfiles (gitignore/gitattributes/editorconfig)" \
    "fast" \
    "$(printf '.gitattributes\n.gitignore\n.editorconfig\n')"

run_case "LICENSE + NOTICE + README + CHANGELOG" \
    "fast" \
    "$(printf 'LICENSE\nNOTICE\nREADME.md\nCHANGELOG.md\n')"

# --- DEEP-PATH cases (anything Rust-adjacent kills the fast-path) ---
run_case ".rs kills fast-path" \
    "deep" \
    "$(printf 'docs/foo.md\ncrates/vokra-core/src/lib.rs\n')"

run_case "Cargo.toml kills fast-path" \
    "deep" \
    "Cargo.toml"

run_case "crate Cargo.toml kills fast-path (path pattern *.toml)" \
    "deep" \
    "crates/vokra-core/Cargo.toml"

run_case "scripts/ kills fast-path" \
    "deep" \
    "scripts/check-zero-deps.sh"

run_case "tools/ kills fast-path" \
    "deep" \
    "tools/eval/librispeech_wer.py"

run_case ".githooks/ (hook self-change) kills fast-path" \
    "deep" \
    ".githooks/pre-push"

run_case "tests/ kills fast-path" \
    "deep" \
    "tests/fixtures/audio/README.md"

run_case "integrations/ kills fast-path" \
    "deep" \
    "integrations/vokra-server/Cargo.toml"

run_case "unrecognised extension kills fast-path" \
    "deep" \
    "web/pkg/index.html"

# --- DEFENSIVE cases (must fall through to deep) ---
run_case "empty diff falls through to deep" \
    "deep" \
    ""

run_case "VOKRA_HOOK_DEEP=1 forces deep on docs-only input" \
    "deep" \
    "CLAUDE.md" \
    "VOKRA_HOOK_DEEP=1"

echo
if [ "$fail" -eq 0 ]; then
    echo "test-pre-push-fastpath: OK (${pass} cases)"
    exit 0
else
    echo "test-pre-push-fastpath: FAIL (${pass} ok / ${fail} bad)" >&2
    exit 1
fi
