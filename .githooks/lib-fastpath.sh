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
            # Offline sidecar Python tools under tools/parity/ (FR-LD-05: they
            # are not linked into the Rust runtime). Any Rust reference is
            # docstring-only ("regenerate with `python tools/parity/foo.py`"),
            # never `include_str!` / `include_bytes!` / runtime import — grep
            # `crates/**/*.rs` for `tools/parity` hits confirms only comments.
            # Matched BEFORE the generic tools/* trigger so a Python-only
            # touch takes the fast-path. Rust code / config anywhere still
            # trips the *.rs / Cargo.toml lines above.
            tools/parity/*.py|tools/parity/pyproject.toml|tools/parity/uv.lock|tools/parity/vendor/*/*.py|tools/parity/vendor/*/*.md|tools/parity/vendor/*/LICENSE)
                ;;
            # Fixture hash sidecar files (`*.gguf.sha256` / `*.wav.sha256`).
            # Consumed only by CI workflow gate scripts (`grep -vE '^\s*(#|$)'`
            # for a non-comment line) and by owner-recipe README's, never by
            # Rust tests directly — parity tests read the actual GGUF/WAV
            # binary, not its hash sidecar. Matched BEFORE the tests/* trigger.
            tests/fixtures/*/*.sha256)
                ;;
            # Root-level CI-only tool configs. `_typos.toml` (crate-ci/typos
            # allowlist) tunes a CI advisory job; it has no effect on the
            # Rust build/test or on `Cargo.*`. Same class as .github/* — a
            # config that only the CI job reads. Matched before the fallback
            # deep-path *) trigger below.
            _typos.toml)
                ;;
            # HF publish helpers (scripts/publish/**). Callers are:
            #   * owner shell (upload / stage / check-model-size)
            #   * scripts/publish/vast-ai/* (in-instance driver)
            # Rust references are docstring-only (`crates/vokra-convert/src/
            # models/mimi.rs:368` mentions `make_model_card.py` in a comment).
            # No `include_str!` / `include_bytes!` / `Command::new` from Rust
            # code — verified 2026-07-29. Matched BEFORE the generic scripts/*
            # deep-trigger below so a publish-helper-only touch takes the
            # fast-path.
            scripts/publish/*|scripts/publish/*/*|scripts/publish/*/*/*)
                ;;
            # Claude Code hook helpers (scripts/claude-hooks/**). Session
            # tooling, never invoked by Rust code or CI Rust builds.
            scripts/claude-hooks/*|scripts/claude-hooks/*/*)
                ;;
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
    fastpath_reason="only Rust-build-neutral files changed since $base (docs / Python sidecar / fixture hash — see .githooks/lib-fastpath.sh)"
    return 0
}

# ---------------------------------------------------------------------------
# `changed_workspace_crates` — echo the newline-separated list of `crates/<name>`
# names touched by the diff since `diff_base`. Only crates/* (root workspace
# members) — integrations/* live in isolated workspaces and are not `-p`-able
# from root.
#
# Prints nothing (return 1) when:
#   * diff base is unresolvable,
#   * diff is empty,
#   * the diff touches ANY file outside `crates/<name>/` that could affect the
#     workspace build globally (root Cargo.toml/Cargo.lock, `.cargo/*`,
#     `rust-toolchain*`, `deny.toml`, `build.rs`, `scripts/*`, `.githooks/*`,
#     `tools/*` non-Python-sidecar, `tests/*` non-hash-sidecar, or any of the
#     fastpath-eligible categories) — the whole workspace is at risk, run it
#     all.
#
# Callers should treat a non-empty list as "safe to package-scope test to
# these crates + their reverse-deps"; empty means "run --workspace".
# ---------------------------------------------------------------------------
changed_workspace_crates() {
    local base
    if ! base=$(diff_base 2>/dev/null); then return 1; fi
    if [ -z "$base" ]; then return 1; fi
    local files
    files=$(git diff --name-only "$base" HEAD)
    if [ -z "$files" ]; then return 1; fi

    local crates=""
    while IFS= read -r f; do
        case "$f" in
            # Fastpath-eligible = build-neutral, skip without disqualifying:
            docs/*|.github/*|*.md|*.yml|*.yaml|.gitattributes|.gitignore|.editorconfig|LICENSE|NOTICE|README|CONTRIBUTING*|CHANGELOG*|include/*.h|_typos.toml)
                ;;
            tools/parity/*.py|tools/parity/pyproject.toml|tools/parity/uv.lock|tools/parity/vendor/*/*.py|tools/parity/vendor/*/*.md|tools/parity/vendor/*/LICENSE)
                ;;
            tests/fixtures/*/*.sha256)
                ;;
            # HF publish helpers + Claude hooks — build-neutral, same rationale
            # as is_docs_only_diff. Verified 2026-07-29 that Rust code only
            # references these in comments.
            scripts/publish/*|scripts/publish/*/*|scripts/publish/*/*/*)
                ;;
            scripts/claude-hooks/*|scripts/claude-hooks/*/*)
                ;;
            # Root-level build config → disqualifies package scoping (must run
            # full workspace):
            Cargo.toml|Cargo.lock|rust-toolchain*|deny.toml|.cargo/*|build.rs)
                return 1 ;;
            # Anything under a crates/<name>/ tree → include that crate:
            crates/*)
                local name
                name=$(printf '%s\n' "$f" | awk -F/ '{print $2}')
                if [ -n "$name" ]; then
                    case "$crates" in
                        *"$name"$'\n'*|"$name"$'\n'*|*$'\n'"$name") : ;;
                        "") crates="$name" ;;
                        *) crates="$crates"$'\n'"$name" ;;
                    esac
                fi
                ;;
            # Anything else (scripts / .githooks / tools/* non-python /
            # integrations/* / tests/* non-hash / etc) → disqualifies scoping:
            *) return 1 ;;
        esac
    done <<<"$files"

    if [ -z "$crates" ]; then return 1; fi
    printf '%s\n' "$crates"
    return 0
}

# ---------------------------------------------------------------------------
# `expand_reverse_deps <crate1> [<crate2> …]` — echo the given crates plus
# every workspace crate that transitively depends on them, one per line.
# Uses `cargo tree -p <c> --invert` per crate. Silent when cargo tree fails
# (returns 1) so the caller can fall back to --workspace.
# ---------------------------------------------------------------------------
expand_reverse_deps() {
    if [ $# -eq 0 ]; then return 1; fi
    local out=""
    local c line rev
    for c in "$@"; do
        # `cargo tree --workspace -i <c>` (invert) prints the crate + everything
        # that depends on it, across the whole workspace. Filter to `vokra-*`
        # names since every workspace member on this repo is `vokra-*` — the
        # `(*)` "already-printed subtree" markers and blank lines are dropped
        # by the awk `{print $1}` extraction below.
        if ! rev=$(cargo tree --workspace -i "$c" --prefix=none --edges=normal 2>/dev/null); then
            return 1
        fi
        while IFS= read -r line; do
            local pkg
            pkg=$(printf '%s\n' "$line" | awk '{print $1}')
            case "$pkg" in
                vokra-*|"")
                    if [ -n "$pkg" ]; then
                        case $'\n'"$out"$'\n' in
                            *$'\n'"$pkg"$'\n'*) : ;;
                            *) out="${out:+$out$'\n'}$pkg" ;;
                        esac
                    fi
                    ;;
            esac
        done <<<"$rev"
    done
    if [ -z "$out" ]; then return 1; fi
    printf '%s\n' "$out"
    return 0
}
