#!/usr/bin/env bash
# Codex PreToolUse hook: protect the owner's dirty CosyVoice2 license
# manifest from both explicit and broad Git/file mutations.
#
# The manifest is deliberately never opened.  `git status -- <path>` only
# asks Git whether that path is dirty; its contents are not read by this hook.

set -uo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

readonly PROTECTED_REL='tools/parity/cosyvoice2_llm_reference/license_gate_manifest.json'

protected_path_in_text() {
    local text="$1"
    case "$text" in
        *"$PROTECTED_REL"*|*/"$PROTECTED_REL"*) return 0 ;;
        *) return 1 ;;
    esac
}

protected_scope_in_text() {
    # Match only the protected file's ancestor pathspecs.  `tools/coreml` is
    # intentionally not included: a different explicit path must remain
    # usable even while the owner manifest is dirty.  Git pathspec magic and
    # relative parent prefixes are covered because they can name repo/tools.
    local scope_re="(^|[[:space:]'\"])((\./|\.\./)+|:/|:\([^)]*\))?(tools/?([[:space:]'\"]|$)|tools/(\\*{1,2}|\\?|\\[|\\{)([[:space:]'\"]|$)|tools/parity([/[:space:]'\"]|$)|tools/parity/cosyvoice2_llm_reference([/[:space:]'\"]|$))"
    printf '%s' "$1" | grep -Eiq "$scope_re"
}

normalize_git_segment() {
    local normalized="$1"
    while printf '%s' "$normalized" | grep -Eq '^git[[:space:]]+-C[[:space:]]+[^[:space:]]+[[:space:]]+'; do
        normalized="$(printf '%s' "$normalized" \
            | sed -E 's/^git[[:space:]]+-C[[:space:]]+[^[:space:]]+[[:space:]]+/git /')"
    done
    printf '%s' "$normalized"
}

protected_path_dirty() {
    local root="$1" status
    status="$(git -C "$root" status --porcelain --untracked-files=no -- "$PROTECTED_REL" 2>/dev/null || true)"
    [ -n "$status" ]
}

segment_is_git_mutation() {
    local segment="$1" normalized
    normalized="$(normalize_git_segment "$(hook_normalize_segment "$segment")")"
    printf '%s' "$normalized" \
        | grep -Eq '^git[[:space:]]+(add|commit|restore|reset|checkout|clean|rm|mv|stash|switch)([[:space:]]|$)'
}

segment_is_shell_mutation() {
    local segment="$1" normalized
    normalized="$(normalize_git_segment "$(hook_normalize_segment "$segment")")"
    printf '%s' "$normalized" \
        | grep -Eq '^(rm|mv|cp|install|tee|dd|truncate|touch|sed[[:space:]]+.*-[^[:space:]]*i|perl[[:space:]]+.*-[^[:space:]]*i)([[:space:]]|$)' \
        || printf '%s' "$normalized" | grep -Eq '[>][>]*[[:space:]]*[^;&]*tools/parity/cosyvoice2_llm_reference/license_gate_manifest\.json([[:space:]]|$)'
}

segment_is_broad_git_mutation() {
    local segment="$1" normalized
    normalized="$(normalize_git_segment "$(hook_normalize_segment "$segment")")"

    # Parent/pathspec magic only has mutation meaning for Git commands.
    # Without this command-position check, a read-only `git diff -- .` (or an
    # unrelated shell command mentioning `..`) would be mistaken for a broad
    # protected-path mutation.
    if ! printf '%s' "$normalized" \
        | grep -Eq '^git[[:space:]]+(add|commit|restore|reset|checkout|clean|rm|mv|stash|switch)([[:space:]]|$)'; then
        return 1
    fi

    if printf '%s' "$normalized" | grep -Eiq "(^|[[:space:]'\"])\.\.?(/|[[:space:]'\"]|$)|(^|[[:space:]'\"])(:(top|literal|glob)(,[^)]*)?\)|:/)([[:space:]'\"]|$)"; then
        return 0
    fi
    if printf '%s' "$normalized" | grep -Eq '^git[[:space:]]+(clean|stash)([[:space:]]|$)'; then
        # `git clean/stash -- <other-path>` scopes the operation away from
        # the protected file.  Without a pathspec it is branch-wide/broad.
        if ! printf '%s' "$normalized" | grep -Eq '(^|[[:space:]])--[[:space:]]+(\./)?[^.[:space:]][^[:space:]]*([[:space:]]|$)'; then
            return 0
        fi
    fi
    if printf '%s' "$normalized" | grep -Eq '^git[[:space:]]+switch([[:space:]]|$)'; then
        return 0
    fi

    # A restore/reset/checkout/rm with an explicit non-protected path does not
    # touch the protected manifest and remains usable.  No-path, hard/reset,
    # branch-switch and dot/path-root forms are branch-wide mutations.
    if printf '%s' "$normalized" | grep -Eq '^git[[:space:]]+(restore|reset|checkout|rm)([[:space:]]|$)'; then
        if printf '%s' "$normalized" | grep -Eq '(^|[[:space:]])(--hard|--merge|--keep|--discard-changes)([[:space:]]|$)'; then
            return 0
        fi
        if printf '%s' "$normalized" | grep -Eq '^git[[:space:]]+(restore|reset|checkout|rm)[[:space:]]*$'; then
            return 0
        fi
        if printf '%s' "$normalized" | grep -Eq '^git[[:space:]]+(reset|checkout)[[:space:]]+[^-[:space:]][^[:space:]]*([[:space:]]|$)' \
            && ! printf '%s' "$normalized" | grep -Eq '(^|[[:space:]])--([[:space:]]|$)'; then
            return 0
        fi
        if printf '%s' "$normalized" | grep -Eq '^git[[:space:]]+rm[[:space:]]+(-r[[:space:]]+)?\.?/?([[:space:]]|$)'; then
            return 0
        fi
    fi

    if printf '%s' "$normalized" | grep -Eq '^git[[:space:]]+add([[:space:]]|$)'; then
        printf '%s' "$normalized" \
            | grep -Eq '(^|[[:space:]])(-A|--all|-u|--update|-p|--patch|-i|--interactive)([[:space:]]|$)|(^|[[:space:]])\.[/[:space:]]*($|[[:space:]])' \
            && return 0
    fi

    if printf '%s' "$normalized" | grep -Eq '^git[[:space:]]+commit([[:space:]]|$)'; then
        printf '%s' "$normalized" \
            | grep -Eq '(^|[[:space:]])(-a|-am|--all)([[:space:]]|$)' \
            && return 0
    fi
    return 1
}

analyse_command() {
    local root="$1" command="$2" dirty="$3" segment

    # apply_patch's command is a patch payload, not a shell command.
    if printf '%s\n' "$command" \
        | grep -Eq '^\*\*\* (Update|Add|Delete) File: .*(tools/parity/cosyvoice2_llm_reference/license_gate_manifest\.json)([[:space:]]*)$'; then
        echo 'explicit protected manifest in patch payload'
        return 0
    fi

    if protected_path_in_text "$command" || protected_scope_in_text "$command"; then
        while IFS= read -r segment; do
            [ -n "$(hook_trim "$segment")" ] || continue
            if segment_is_git_mutation "$segment" || segment_is_shell_mutation "$segment"; then
                echo 'explicit protected manifest mutation'
                return 0
            fi
        done < <(hook_segments "$command")
    fi

    if [ "$dirty" = dirty ]; then
        while IFS= read -r segment; do
            [ -n "$(hook_trim "$segment")" ] || continue
            if segment_is_broad_git_mutation "$segment"; then
                echo 'broad Git mutation while protected manifest is dirty'
                return 0
            fi
        done < <(hook_segments "$command")
    fi
    return 1
}

self_test() {
    local fails=0 got test_repo test_manifest
    check() {
        local name="$1" expected="$2" command="$3"
        if analyse_command /tmp "$command" dirty >/dev/null; then got=block; else got=allow; fi
        if [ "$got" = "$expected" ]; then
            printf '  ok    %-56s %s\n' "$name" "$got"
        else
            printf '  FAIL  %-56s expected %s, got %s\n' "$name" "$expected" "$got"
            fails=$((fails + 1))
        fi
    }
    check_payload() {
        local name="$1" expected="$2" payload="$3" command dirty got
        command="$(hook_command "$payload")"
        dirty=dirty
        if analyse_command /tmp "$command" "$dirty" >/dev/null; then got=block; else got=allow; fi
        if [ "$got" = "$expected" ]; then
            printf '  ok    %-56s %s\n' "$name" "$got"
        else
            printf '  FAIL  %-56s expected %s, got %s\n' "$name" "$expected" "$got"
            fails=$((fails + 1))
        fi
    }

    echo 'guard-protected-path --self-test'
    check 'explicit git add target' block "git add $PROTECTED_REL"
    check 'ancestor git add tools' block 'git add tools'
    check 'ancestor git add parity' block 'git add tools/parity'
    check 'ancestor git add dot slash' block 'git add ./tools/parity/'
    check 'quoted ancestor tools' block 'git add "tools"'
    check 'quoted ancestor parity' block "git checkout -- 'tools/parity'"
    check 'quoted parent ancestor' block 'git add "../tools"'
    check 'ancestor checkout tools' block 'git checkout -- tools'
    check 'ancestor restore parity' block 'git restore tools/parity'
    check 'ancestor rm directory' block 'git rm -r tools/parity/cosyvoice2_llm_reference'
    check 'ancestor stash tools' block 'git stash push -- tools'
    check 'top magic ancestor' block 'git add :(top)tools'
    check 'top magic wildcard ancestor' block 'git add :(top,glob)tools/**'
    check 'root magic ancestor' block 'git add :/tools'
    check 'parent pathspec' block 'git add ..'
    check 'git -C broad add' block 'git -C docs add ../tools'
    check 'git -C spaced broad add' block 'git  -C  docs add ../tools'
    check 'git -C broad restore' block 'git -C docs restore tools/parity'
    check 'explicit remove target' block "rm -- $PROTECTED_REL"
    check 'apply_patch marker' block "*** Update File: $PROTECTED_REL"
    check 'dirty git add dot' block 'git add .'
    check 'dirty git add all' block 'git add --all'
    check 'dirty git add update' block 'git add -u'
    check 'dirty commit all' block 'git commit --all -m docs'
    check 'explicit other restore' allow 'git restore --staged other.rs'
    check 'explicit other rm' allow 'git rm other.rs'
    check 'dirty reset broad' block 'git reset HEAD~1'
    check 'dirty checkout broad' block 'git checkout main'
    check 'dirty clean broad' block 'git clean -fd'
    check 'explicit other clean' allow 'git clean -fd -- docs/README.md'
    check 'dirty stash broad' block 'git stash push -m save'
    check 'explicit other stash' allow 'git stash push -m save -- docs/README.md'
    check 'dirty switch broad' block 'git switch main'
    check 'redirection to manifest' block 'printf x > tools/parity/cosyvoice2_llm_reference/license_gate_manifest.json'
    check 'truncate manifest' block 'truncate -s 0 tools/parity/cosyvoice2_llm_reference/license_gate_manifest.json'
    check 'touch manifest' block 'touch tools/parity/cosyvoice2_llm_reference/license_gate_manifest.json'
    check 'chained broad mutation' block 'git status --short && git add -A'
    check 'manifest prose' allow "echo '$PROTECTED_REL is protected'"
    check 'manifest read-only diff' allow "git diff -- $PROTECTED_REL"
    check 'read-only repository diff root' allow 'git diff -- .'
    check 'unrelated parent prose' allow 'echo "../tools is discussed"'
    check 'other explicit add' allow 'git add docs/README.md'
    check 'other tools path' allow 'git add tools/coreml/foo.py'
    check 'quoted other tools path' allow 'git add "tools/coreml/foo.py"'
    check_payload 'JSON command extraction' block \
        "{\"tool_input\":{\"command\":\"git add $PROTECTED_REL\"}}"
    check_payload 'JSON prose extraction' allow \
        "{\"tool_input\":{\"command\":\"echo '$PROTECTED_REL'\"}}"

    test_repo="$(mktemp -d "${TMPDIR:-/tmp}/vokra-protected-path.XXXXXX")"
    test_manifest="$test_repo/$PROTECTED_REL"
    mkdir -p "$(dirname "$test_manifest")" "$test_repo/docs"
    printf 'baseline\n' > "$test_manifest"
    git -C "$test_repo" init -q
    git -C "$test_repo" add "$PROTECTED_REL"
    git -C "$test_repo" -c user.name=self-test -c user.email=self-test@example.invalid commit -qm baseline
    printf 'dirty\n' > "$test_manifest"
    cwd_payload="$(jq -cn --arg cwd "$test_repo/docs" '{cwd:$cwd,tool_input:{command:"git add ."}}')"
    if printf '%s' "$cwd_payload" | bash "$0" >/dev/null 2>&1; then
        printf '  FAIL  %-56s expected block, got allow\n' 'subdirectory cwd resolves to git root'
        fails=$((fails + 1))
    else
        printf '  ok    %-56s block\n' 'subdirectory cwd resolves to git root'
    fi
    rm -rf "$test_repo"

    if [ "$fails" -eq 0 ]; then
        echo 'guard-protected-path --self-test: OK'
        return 0
    fi
    echo "guard-protected-path --self-test: FAIL ($fails)"
    return 1
}

if [ "${1:-}" = --self-test ]; then
    self_test
    exit $?
fi

payload="$(cat)"
cwd="$(hook_json_value "$payload" '.cwd // empty')"
if [ -n "$cwd" ]; then
    root="$(git -C "$cwd" rev-parse --show-toplevel 2>/dev/null || true)"
else
    root=""
fi
root="${root:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
command="$(hook_command "$payload")"
file_path="$(hook_file_path "$payload")"

if protected_path_in_text "$file_path"; then
    echo "Blocked: edits to the owner's dirty manifest are outside this task: $PROTECTED_REL" >&2
    exit 2
fi

if reason="$(analyse_command "$root" "$command" "$(protected_path_dirty "$root" && echo dirty || echo clean)")"; then
    {
        echo "Blocked: $reason."
        echo "The owner-owned dirty path is protected and must not be staged,"
        echo "restored, reset, checked out, cleaned, removed, or overwritten."
        echo "Read-only diff/status inspection remains allowed."
    } >&2
    exit 2
fi
exit 0
