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

echo "test-pre-push-fastpath: 36 cases"
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

# --- FAST-PATH cases for tools/parity offline sidecars (Python, uv, vendor) ---
run_case "tools/parity Python script only" \
    "fast" \
    "tools/parity/sbv2_prepare_checkpoint.py"

run_case "tools/parity pyproject + uv.lock" \
    "fast" \
    "$(printf 'tools/parity/pyproject.toml\ntools/parity/uv.lock\n')"

run_case "tools/parity vendor tree (Python + docs)" \
    "fast" \
    "$(printf 'tools/parity/vendor/vits/text_encoder.py\ntools/parity/vendor/vits/README.md\ntools/parity/vendor/vits/LICENSE\n')"

run_case "tools/parity Python + docs mixed" \
    "fast" \
    "$(printf 'tools/parity/bin_to_safetensors.py\ndocs/handoff/x.md\n')"

# --- FAST-PATH cases for fixture hash sidecars ---
run_case "SBV2 fixture hash sidecar only" \
    "fast" \
    "tests/fixtures/sbv2/sbv2-v2-multilingual-base.gguf.sha256"

run_case "audio fixture hash sidecar only" \
    "fast" \
    "tests/fixtures/audio/jfk-30s.wav.sha256"

run_case "several fixture hash sidecars at once" \
    "fast" \
    "$(printf 'tests/fixtures/sbv2/sbv2-v2-multilingual-base.gguf.sha256\ntests/fixtures/sbv2/deberta-v2-large-japanese-char-wwm.gguf.sha256\ntests/fixtures/sbv2/deberta-v3-large.gguf.sha256\n')"

# --- FAST-PATH case for _typos.toml (CI advisory config only) ---
run_case "_typos.toml only (CI advisory config)" \
    "fast" \
    "_typos.toml"

# --- FAST-PATH cases for scripts/publish/** (HF publish helpers, Rust deps = 0) ---
run_case "scripts/publish shell script only" \
    "fast" \
    "scripts/publish/publish-one.sh"

run_case "scripts/publish new script + doc together" \
    "fast" \
    "$(printf 'scripts/publish/check-model-size.sh\ndocs/handoff/vast-ai-large-model-publish.md\n')"

run_case "scripts/publish/vast-ai subdir" \
    "fast" \
    "$(printf 'scripts/publish/vast-ai/provision.sh\nscripts/publish/vast-ai/run-one.sh\n')"

run_case "scripts/publish python helper" \
    "fast" \
    "scripts/publish/make_model_card.py"

run_case "scripts/claude-hooks only" \
    "fast" \
    "scripts/claude-hooks/on-tool-use.sh"

# --- DEEP-PATH cases (anything Rust-adjacent kills the fast-path) ---

run_case "tools/parity Python + .rs together kills fast-path" \
    "deep" \
    "$(printf 'tools/parity/sbv2_prepare_checkpoint.py\ncrates/vokra-models/src/sbv2/mod.rs\n')"

run_case "tools/parity non-Python (bash script) kills fast-path" \
    "deep" \
    "tools/parity/cuda_rtf_variance.sh"

run_case "tests/fixtures non-.sha256 (README) kills fast-path" \
    "deep" \
    "tests/fixtures/sbv2/README.md"

run_case "tests/fixtures manifest.json kills fast-path" \
    "deep" \
    "tests/fixtures/sbv2/reference_dump.manifest.json"

run_case ".rs kills fast-path" \
    "deep" \
    "$(printf 'docs/foo.md\ncrates/vokra-core/src/lib.rs\n')"

run_case "Cargo.toml kills fast-path" \
    "deep" \
    "Cargo.toml"

run_case "crate Cargo.toml kills fast-path (path pattern *.toml)" \
    "deep" \
    "crates/vokra-core/Cargo.toml"

run_case "scripts/ (general, non-publish) kills fast-path" \
    "deep" \
    "scripts/check-zero-deps.sh"

run_case "scripts/check-fa-v3-confinement.sh kills fast-path (Rust test dep)" \
    "deep" \
    "scripts/check-fa-v3-confinement.sh"

run_case "scripts/publish + generic script kills fast-path" \
    "deep" \
    "$(printf 'scripts/publish/publish-one.sh\nscripts/verify.sh\n')"

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
