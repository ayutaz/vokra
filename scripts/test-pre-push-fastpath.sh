#!/usr/bin/env bash
# scripts/test-pre-push-fastpath.sh
# shellcheck source-path=SCRIPTDIR
#
# Regression test for the .githooks/pre-push ref-update and diff fast-paths.
#
# Sources .githooks/lib-fastpath.sh (production classifier) and drives
# `is_docs_only_diff` per case by shadowing `git diff --name-only` and
# `diff_base` in a subshell. No real git activity, no cargo — the test runs
# in milliseconds and can sit inside `scripts/verify.sh` or a CI leg without
# added cost.
#
# Cases exercise both directions of the classifiers:
#   * deletion-only push updates may skip cargo after compliance passes
#   * normal, mixed, empty, or malformed push updates must not skip cargo
#   * docs-only inputs must land on fast-path (return 0)
#   * anything Rust-adjacent must land on deep-path (return 1)
#   * defensive inputs (empty diff, VOKRA_HOOK_DEEP=1) must land on deep-path
#   * production control flow runs compliance before every non-bypassed path
#   * deletion/docs paths skip Cargo; Darwin deep paths refuse before Cargo
#   * explicit Darwin override and Linux deep paths reach mocked Cargo

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

pass=0
fail=0
TEST_TMP="$(mktemp -d)"
trap 'rm -rf "$TEST_TMP"' EXIT

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

run_update_case() {
    local name="$1"
    local expected="$2"
    local updates="$3"
    local verdict

    # shellcheck source=../.githooks/lib-fastpath.sh
    source "$ROOT/.githooks/lib-fastpath.sh"
    if is_deletion_only_push_updates "$updates"; then
        verdict="delete-skip-cargo"
    else
        verdict="continue"
    fi

    if [ "$verdict" = "$expected" ]; then
        pass=$((pass + 1))
        printf 'OK   %-52s → %s\n' "$name" "$verdict"
    else
        fail=$((fail + 1))
        printf 'FAIL %-52s expected=%s got=%s\n' "$name" "$expected" "$verdict"
    fi
}

# Execute the production hook with only its external boundaries mocked. This
# complements the classifier unit cases above: a correct helper that is called
# too late (after Cargo) or whose result is ignored is still a broken hook.
run_hook_case() {
    local name="$1" expected_rc="$2" updates="$3" os="$4" files="$5"
    local mode="$6" compliance_rc="$7" expected_text="$8"
    local expected_cargo="$9" expected_compliance="${10}"
    local case_dir="$TEST_TMP/hook-$pass-$fail" mock_bin output rc
    local log="$case_dir/calls.log"

    mkdir -p "$case_dir/bin"
    mock_bin="$case_dir/bin"
    : >"$log"

    cat >"$mock_bin/git" <<'MOCK'
#!/bin/bash
printf 'git %s\n' "$*" >>"$HOOK_TEST_LOG"
case "${1:-}" in
    rev-parse)
        case "${2:-}" in
            --show-toplevel) printf '%s\n' "$HOOK_TEST_ROOT" ;;
            --abbrev-ref) exit 1 ;;
            --verify)
                [ "${3:-}" = "origin/main" ] || exit 1
                printf '%s\n' origin/main
                ;;
            *) exit 1 ;;
        esac
        ;;
    merge-base) printf '%s\n' fake-base ;;
    diff)
        [ "${2:-}" = "--name-only" ] || exit 1
        [ -z "$HOOK_TEST_FILES" ] || printf '%s\n' "$HOOK_TEST_FILES"
        ;;
    *) exit 1 ;;
esac
MOCK

    cat >"$mock_bin/bash" <<'MOCK'
#!/bin/bash
printf 'bash %s\n' "$*" >>"$HOOK_TEST_LOG"
if [ "${1:-}" = "scripts/compliance/test-nvidia-scanner-sigpipe.sh" ]; then
    exit "$HOOK_TEST_COMPLIANCE_RC"
fi
exec /bin/bash "$@"
MOCK

    cat >"$mock_bin/uname" <<'MOCK'
#!/bin/bash
if [ "${1:-}" = "-s" ]; then
    printf '%s\n' "$HOOK_TEST_OS"
else
    printf '%s\n' "$HOOK_TEST_OS"
fi
MOCK

    cat >"$mock_bin/cargo" <<'MOCK'
#!/bin/bash
printf 'cargo %s\n' "$*" >>"$HOOK_TEST_LOG"
exit 0
MOCK

    cat >"$mock_bin/cargo-nextest" <<'MOCK'
#!/bin/bash
exit 0
MOCK

    cat >"$mock_bin/sccache" <<'MOCK'
#!/bin/bash
printf 'sccache %s\n' "$*" >>"$HOOK_TEST_LOG"
exit 0
MOCK
    chmod +x "$mock_bin"/*

    local skip=0 allow=0
    case "$mode" in
        skip) skip=1 ;;
        allow-heavy) allow=1 ;;
        normal) ;;
        *) echo "test setup error: unknown hook mode '$mode'" >&2; exit 2 ;;
    esac

    set +e
    output="$(printf '%s' "$updates" | env \
        PATH="$mock_bin:$PATH" \
        HOOK_TEST_LOG="$log" \
        HOOK_TEST_ROOT="$ROOT" \
        HOOK_TEST_FILES="$files" \
        HOOK_TEST_OS="$os" \
        HOOK_TEST_COMPLIANCE_RC="$compliance_rc" \
        VOKRA_SKIP_HOOKS="$skip" \
        VOKRA_ALLOW_LOCAL_HEAVY="$allow" \
        /bin/bash "$ROOT/.githooks/pre-push" 2>&1)"
    rc=$?
    set -e

    local bad=""
    [ "$rc" -eq "$expected_rc" ] || bad="exit=$rc (expected $expected_rc)"
    if ! printf '%s' "$output" | grep -Fq "$expected_text"; then
        bad="${bad:+$bad; }missing output '$expected_text'"
    fi
    if [ "$expected_cargo" = "yes" ]; then
        grep -q '^cargo ' "$log" || bad="${bad:+$bad; }Cargo was not invoked"
    elif grep -q '^cargo ' "$log"; then
        bad="${bad:+$bad; }Cargo was invoked"
    fi
    if [ "$expected_compliance" = "yes" ]; then
        grep -q '^bash scripts/compliance/test-nvidia-scanner-sigpipe.sh$' "$log" \
            || bad="${bad:+$bad; }compliance was not invoked"
    elif grep -q '^bash scripts/compliance/test-nvidia-scanner-sigpipe.sh$' "$log"; then
        bad="${bad:+$bad; }compliance was invoked"
    fi

    if [ -z "$bad" ]; then
        pass=$((pass + 1))
        printf 'OK   %-52s → exit %s\n' "$name" "$rc"
    else
        fail=$((fail + 1))
        printf 'FAIL %-52s %s\n' "$name" "$bad"
        printf '     output: %s\n' "$(printf '%s' "$output" | tr '\n' ' ')"
        printf '     calls:\n'
        sed 's/^/       /' "$log"
    fi
}

zero40="0000000000000000000000000000000000000000"
zero64="0000000000000000000000000000000000000000000000000000000000000000"
sha40="1111111111111111111111111111111111111111"

echo "test-pre-push-fastpath: classifier + production-hook integration"
echo

# --- REF-UPDATE cases (read from git pre-push stdin) ---
run_update_case "single SHA-1 branch deletion skips cargo" \
    "delete-skip-cargo" \
    "(delete) $zero40 refs/heads/old $sha40"

run_update_case "multiple SHA-1/SHA-256 deletions skip cargo" \
    "delete-skip-cargo" \
    "$(printf '(delete) %s refs/heads/old-a %s\n(delete) %s refs/heads/old-b %s\n' "$zero40" "$sha40" "$zero64" "$zero64")"

run_update_case "normal update continues through checks" \
    "continue" \
    "refs/heads/main $sha40 refs/heads/main $zero40"

run_update_case "mixed deletion and update continues through checks" \
    "continue" \
    "$(printf '(delete) %s refs/heads/old %s\nrefs/heads/main %s refs/heads/main %s\n' "$zero40" "$sha40" "$sha40" "$sha40")"

run_update_case "empty push-update input does not skip" \
    "continue" \
    ""

run_update_case "malformed push-update input does not skip" \
    "continue" \
    "(delete) $zero40 refs/heads/old"

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
echo "--- PRODUCTION HOOK integration cases (Cargo/compliance mocked) ---"

run_hook_case "deletion-only runs compliance and skips Cargo" \
    0 "(delete) $zero40 refs/heads/old $sha40" Darwin "" normal 0 \
    "pre-push: OK (ref deletion" no yes

run_hook_case "docs-only on Darwin stays on the light path" \
    0 "refs/heads/topic $sha40 refs/heads/topic $sha40" Darwin "docs/README.md" normal 0 \
    "pre-push: OK (fast-path" no yes

run_hook_case "deep path on Darwin refuses before Cargo" \
    1 "refs/heads/topic $sha40 refs/heads/topic $sha40" Darwin "scripts/check-zero-deps.sh" normal 0 \
    "REFUSE deep cargo path" no yes

run_hook_case "normal update with empty diff still refuses on Darwin" \
    1 "refs/heads/topic $sha40 refs/heads/topic $sha40" Darwin "" normal 0 \
    "REFUSE deep cargo path" no yes

run_hook_case "Darwin explicit heavy override reaches mocked Cargo" \
    0 "refs/heads/topic $sha40 refs/heads/topic $sha40" Darwin "scripts/check-zero-deps.sh" allow-heavy 0 \
    "pre-push: OK" yes yes

run_hook_case "Linux deep path reaches mocked Cargo" \
    0 "refs/heads/topic $sha40 refs/heads/topic $sha40" Linux "scripts/check-zero-deps.sh" normal 0 \
    "pre-push: OK" yes yes

run_hook_case "compliance failure aborts deletion fast-path" \
    17 "(delete) $zero40 refs/heads/old $sha40" Darwin "" normal 17 \
    "compliance scanner fail-open regression test" no yes

run_hook_case "VOKRA_SKIP_HOOKS bypasses all hook work" \
    0 "refs/heads/topic $sha40 refs/heads/topic $sha40" Darwin "scripts/check-zero-deps.sh" skip 0 \
    "pre-push: skipped" no no

echo
if [ "$fail" -eq 0 ]; then
    echo "test-pre-push-fastpath: OK (${pass} cases)"
    exit 0
else
    echo "test-pre-push-fastpath: FAIL (${pass} ok / ${fail} bad)" >&2
    exit 1
fi
