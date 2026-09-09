#!/usr/bin/env bash
# Codex PreToolUse hook: make cloud lifecycle mutations explicit and routed.
#
# VAST mutations must use vastai-safe.sh so credential-valued output is
# redacted.  `stop` is denied unless the retained-handoff storage tradeoff is
# explicitly acknowledged.  Scaleway create/start additionally requires an
# exact-head preflight token plus a separate user approval token.

set -uo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

readonly VAST_SAFE='vastai-safe.sh'

is_vast_mutation() {
    local normalized="$1"
    printf '%s' "$normalized" | grep -Eiq \
        '(^|[[:space:]])vastai[[:space:]].*([[:space:]])(create|destroy|delete|remove|stop|start|reboot|relaunch|update|attach|detach)([[:space:]]|$)|(^|[[:space:]])vastai[[:space:]]+(create|destroy|delete|remove|stop|start|reboot|relaunch|update|attach|detach)([[:space:]]|$)'
}

is_vast_stop() {
    printf '%s' "$1" | grep -Eiq \
        '(^|[[:space:]])vastai[[:space:]].*[[:space:]]stop([[:space:]]|$)|(^|[[:space:]])vastai[[:space:]]+stop([[:space:]]|$)|vastai-safe\.sh[[:space:]]+stop([[:space:]]|$)'
}

is_safe_vast_wrapper() {
    # The wrapper must be the executable at the command position, optionally
    # after `bash`/`sh`.  Matching an arbitrary token would let
    # `vastai destroy ... vastai-safe.sh` masquerade as a safe invocation.
    printf '%s' "$1" | grep -Eq "^((bash|sh)[[:space:]]+)?([^[:space:]]*/)?${VAST_SAFE}([[:space:]]|$)"
}

is_scaleway_cli_create_start() {
    local normalized="$1"
    # Official `scw` CLI and common API/IaC mutation surfaces. Read-only
    # listing/status commands do not match these verbs.
    if printf '%s' "$normalized" | grep -Eiq \
        '^(scw|scaleway)([[:space:]]|$).*([[:space:]])(create|start|boot|launch)([[:space:]]|$)'; then
        return 0
    fi
    if printf '%s' "$normalized" | grep -Eiq \
        '^(curl|wget)([[:space:]]|$).*(api\.scaleway\.com|scaleway\.com/api)' \
        && is_http_mutation "$normalized"; then
        return 0
    fi
    if printf '%s' "$normalized" | grep -Eiq \
        '^(terraform[[:space:]]+apply|pulumi[[:space:]]+up)([[:space:]]|$).*scaleway|scaleway.*(terraform|pulumi).*(apply|up)'; then
        return 0
    fi
    return 1
}

is_vast_api_mutation() {
    local normalized="$1"
    printf '%s' "$normalized" | grep -Eiq \
        '^(curl|wget)([[:space:]]|$).*(api\.vast\.ai|vast\.ai/api)' \
        && is_http_mutation "$normalized"
}

is_http_mutation() {
    local normalized="$1"
    # Explicit mutating methods, including curl's -XPOST and wget's
    # --method=POST forms.
    if printf '%s' "$normalized" | grep -Eiq \
        '(^|[[:space:]])(--request|--method)([=[:space:]]*)?(POST|PUT|PATCH|DELETE)([[:space:]]|$)|(^|[[:space:]])-X([=[:space:]]*)?(POST|PUT|PATCH|DELETE)([[:space:]]|$)'; then
        return 0
    fi
    # curl data/form/upload flags imply a request body and therefore an
    # implicit POST even when -X/--request is absent.
    printf '%s' "$normalized" | grep -Eiq \
        '(^|[[:space:]])(--data([=-]|[[:space:]])|--form([=-]|[[:space:]])|--upload-file([=-]|[[:space:]])|-d[^[:space:]]*([[:space:]]|$)|-F[^[:space:]]*([[:space:]]|$)|-T[^[:space:]]*([[:space:]]|$))'
}

is_non_mutating_mode() {
    printf '%s' "$1" | grep -Eq '(^|[[:space:]])(--help|--self-test)([[:space:]]|$)'
}

has_scaleway_gate() {
    local root="$1" segment="$2" expected_head
    # PASS@<40-hex> binds the preflight to an exact reviewed HEAD; the second
    # token is an independent, explicit user authorization. Neither is a
    # generic allow-local escape hatch and neither is accepted for VAST.
    expected_head="$(git -C "$root" rev-parse HEAD 2>/dev/null || true)"
    [ -n "$expected_head" ] \
        && printf '%s' "$segment" | grep -Eq "(^|[[:space:]])VOKRA_SCALEWAY_PREFLIGHT=PASS@$expected_head([[:space:]]|$)" \
        && printf '%s' "$segment" | grep -Eq '(^|[[:space:]])VOKRA_SCALEWAY_USER_APPROVAL=1([[:space:]]|$)'
}

analyse_segment() {
    local root="$1" segment="$2" normalized
    normalized="$(hook_normalize_segment "$segment")"
    [ -n "$normalized" ] || return 1

    # CLI help/self-tests do not create, start or stop resources.  A dry-run
    # flag is not a trusted bypass: wrappers and CLIs may still load or
    # mutate cached state before honoring it.
    is_non_mutating_mode "$normalized" && return 1

    # A safe wrapper is the only accepted VAST API entry point. SSH commands
    # are remote worker commands, not local VAST API mutations.
    if is_vast_mutation "$normalized"; then
        if ! is_safe_vast_wrapper "$normalized"; then
            echo 'direct VAST mutation (use scripts/publish/vast-ai/vastai-safe.sh)'
            return 0
        fi
    fi
    if is_safe_vast_wrapper "$normalized" && is_vast_stop "$normalized" \
        && ! printf '%s' "$segment" | grep -Eq '(^|[[:space:]])VOKRA_VAST_RETAINED_HANDOFF_APPROVAL=1([[:space:]]|$)'; then
        echo 'VAST stop without retained-handoff approval'
        return 0
    fi

    if is_vast_api_mutation "$normalized"; then
        echo 'direct VAST API mutation (use scripts/publish/vast-ai/vastai-safe.sh)'
        return 0
    fi

    if is_scaleway_cli_create_start "$normalized" && ! has_scaleway_gate "$root" "$segment"; then
        echo 'Scaleway create/start without exact-head preflight and user approval'
        return 0
    fi
    return 1
}

analyse_command() {
    local root="$1" command="$2" segment reason
    while IFS= read -r segment; do
        [ -n "$(hook_trim "$segment")" ] || continue
        if reason="$(analyse_segment "$root" "$segment")"; then
            echo "$reason"
            return 0
        fi
    done < <(hook_segments "$command")
    return 1
}

self_test() {
    local fails=0 got root head gate
    root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    head="$(git -C "$root" rev-parse HEAD)"
    gate="VOKRA_SCALEWAY_PREFLIGHT=PASS@$head VOKRA_SCALEWAY_USER_APPROVAL=1"
    check() {
        local name="$1" expected="$2" command="$3"
        if analyse_command "$root" "$command" >/dev/null; then got=block; else got=allow; fi
        if [ "$got" = "$expected" ]; then
            printf '  ok    %-58s %s\n' "$name" "$got"
        else
            printf '  FAIL  %-58s expected %s, got %s\n' "$name" "$expected" "$got"
            fails=$((fails + 1))
        fi
    }

    echo 'guard-cloud-mutations --self-test'
    check 'direct VAST create' block 'vastai create instance 123'
    check 'direct VAST destroy' block 'vastai destroy instance 123'
    check 'wrapper token in direct args' block 'vastai destroy instance 123 scripts/publish/vast-ai/vastai-safe.sh'
    check 'direct VAST API curl' block 'curl -X POST https://api.vast.ai/api/v0/instances'
    check 'VAST API curl with dry-run' block 'curl -X POST https://api.vast.ai/api/v0/instances --dry-run'
    check 'VAST API implicit data POST' block 'curl https://api.vast.ai/api/v0/instances -d payload'
    check 'VAST API form POST' block 'curl --form payload https://api.vast.ai/api/v0/instances'
    check 'VAST API upload mutation' block 'curl --upload-file model.gguf https://api.vast.ai/api/v0/instances'
    check 'VAST API wget POST' block 'wget --method=POST https://api.vast.ai/api/v0/instances'
    check 'safe VAST show' allow 'scripts/publish/vast-ai/vastai-safe.sh show instance 123'
    check 'bash safe VAST show' allow 'bash scripts/publish/vast-ai/vastai-safe.sh show instance 123'
    check 'safe VAST destroy' allow 'scripts/publish/vast-ai/vastai-safe.sh destroy instance 123'
    check 'safe VAST stop no approval' block 'scripts/publish/vast-ai/vastai-safe.sh stop instance 123'
    check 'safe VAST stop approved' allow 'VOKRA_VAST_RETAINED_HANDOFF_APPROVAL=1 scripts/publish/vast-ai/vastai-safe.sh stop instance 123'
    check 'remote VAST SSH' allow 'ssh vast-worker vastai status'
    check 'Scaleway create no gate' block 'scw apple-silicon server create --name test'
    check 'Scaleway create help' allow 'scw apple-silicon server create --help'
    check 'Scaleway API curl' block 'curl -X POST https://api.scaleway.com/instances'
    check 'Scaleway API curl with dry-run' block 'curl -X POST https://api.scaleway.com/instances --dry-run'
    check 'Scaleway API implicit data POST' block 'curl https://api.scaleway.com/instances --data payload'
    check 'Scaleway API wget POST' block 'wget --method=POST https://api.scaleway.com/instances'
    check 'VAST API remains blocked with Scaleway tokens' block "VOKRA_SCALEWAY_PREFLIGHT=PASS@$head VOKRA_SCALEWAY_USER_APPROVAL=1 curl https://api.vast.ai/api/v0/instances --data payload"
    check 'Scaleway start no gate' block 'scw instance server start 123'
    check 'Scaleway exact current HEAD gate' allow "$gate scw apple-silicon server create --name test"
    check 'Scaleway different 40-hex HEAD' block 'VOKRA_SCALEWAY_PREFLIGHT=PASS@0123456789abcdef0123456789abcdef01234567 VOKRA_SCALEWAY_USER_APPROVAL=1 scw server create'
    check 'Scaleway missing user approval' block "VOKRA_SCALEWAY_PREFLIGHT=PASS@$head scw server create"
    check 'Scaleway list read-only' allow 'scw apple-silicon server list'
    check 'Scaleway status read-only' allow 'scw instance server get 123'
    check 'prose false positive' allow 'echo "scw server create requires approval"'
    check 'chained mutation' block 'git status --short && vastai start instance 123'

    if [ "$fails" -eq 0 ]; then
        echo 'guard-cloud-mutations --self-test: OK'
        return 0
    fi
    echo "guard-cloud-mutations --self-test: FAIL ($fails)"
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
[ -n "$command" ] || exit 0
if reason="$(analyse_command "$root" "$command")"; then
    {
        echo "Blocked: $reason."
        echo "Read-only VAST/Scaleway listing and status commands remain allowed."
        echo "VAST stop requires VOKRA_VAST_RETAINED_HANDOFF_APPROVAL=1."
        echo "Scaleway create/start requires VOKRA_SCALEWAY_PREFLIGHT=PASS@<40-hex-HEAD> and VOKRA_SCALEWAY_USER_APPROVAL=1."
    } >&2
    exit 2
fi
exit 0
