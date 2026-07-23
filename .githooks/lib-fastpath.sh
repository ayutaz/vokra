# .githooks/lib-fastpath.sh
#
# Diff-shape classifier for the pre-push fast-paths. Sourced by
# `.githooks/pre-push` (production) and `scripts/test-pre-push-fastpath.sh`
# (regression tests). Not standalone-executable — always sourced.
#
# Two functions:
#
#   * `diff_base` — echoes the commit id to diff HEAD against, or fails
#     (returns 1, prints nothing). Prefers the tracking upstream; falls
#     back to origin/main.
#
#   * `is_docs_only_diff` — sets `fastpath_reason` and returns 0 if every
#     file changed since `diff_base` matches a documentation-shape pattern.
#     Otherwise returns 1 with the reason set to the first offending file.
#     `VOKRA_HOOK_DEEP=1` forces a non-zero return regardless of diff.

# shellcheck disable=SC2034  # fastpath_reason is set for callers to read.

fastpath_reason=""

diff_base() {
    local upstream
    if upstream=$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null); then
        if [ -n "$upstream" ] && git rev-parse --verify "$upstream" >/dev/null 2>&1; then
            git merge-base HEAD "$upstream"
            return 0
        fi
    fi
    if git rev-parse --verify origin/main >/dev/null 2>&1; then
        git merge-base HEAD origin/main
        return 0
    fi
    return 1
}

is_docs_only_diff() {
    if [ "${VOKRA_HOOK_DEEP:-0}" = "1" ]; then
        fastpath_reason="VOKRA_HOOK_DEEP=1 (forcing deep path)"
        return 1
    fi
    local base
    if ! base=$(diff_base 2>/dev/null); then
        fastpath_reason="cannot determine diff base — taking the deep path"
        return 1
    fi
    if [ -z "$base" ]; then
        fastpath_reason="empty diff base — taking the deep path"
        return 1
    fi
    local files
    files=$(git diff --name-only "$base" HEAD)
    if [ -z "$files" ]; then
        fastpath_reason="no files changed since $base — taking the deep path"
        return 1
    fi
    local trigger=""
    while IFS= read -r f; do
        # SAFETY: any of these paths, if touched, means the change may affect
        # compiled output OR the hook itself. Keep this list conservative;
        # over-inclusion loses the fast-path, under-inclusion loses safety.
        case "$f" in
            # Rust / build (highest priority — must not skip):
            *.rs|Cargo.toml|Cargo.lock|rust-toolchain*|deny.toml|.cargo/*|build.rs)
                trigger="$f"; break ;;
            # Scripts / tooling that may be exercised elsewhere in the hook or in tests:
            scripts/*|tools/*|.githooks/*)
                trigger="$f"; break ;;
            # Test fixtures / harness (may bind test output):
            tests/*|integrations/*)
                trigger="$f"; break ;;
            # Documentation-shape files → OK to skip:
            docs/*|.github/*|*.md|*.yml|*.yaml|.gitattributes|.gitignore|.editorconfig|LICENSE|NOTICE|README|CONTRIBUTING*|CHANGELOG*|include/*.h)
                ;;
            # Everything else → deep path (safe default).
            *) trigger="$f"; break ;;
        esac
    done <<<"$files"
    if [ -n "$trigger" ]; then
        fastpath_reason="deep path required (first non-docs file: $trigger)"
        return 1
    fi
    fastpath_reason="only documentation-shape files changed since $base"
    return 0
}
