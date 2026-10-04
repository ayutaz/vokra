#!/usr/bin/env bash
# Codex PreToolUse hook: keep model acquisition and model execution off the
# maintainer Mac.  This is intentionally separate from guard-local-memory.sh:
# the memory guard is size/Cargo based, while this guard applies even to a
# small checkpoint and does not offer a general local-heavy escape hatch.

set -uo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

readonly REVIEWED_ARCHIVE_CONTROLLER="/private/tmp/vokra-realtime-archive-collection-vast-controller-v6-20261003.sh"
readonly REVIEWED_ARCHIVE_CONTROLLER_SHA256="1d81150ebe1b464233191a535e85f83d6ecaa00e421f8724f3ca683329d8c031"
readonly REVIEWED_ARCHIVE_AUDIT="/private/tmp/vokra-realtime-dependency-audit-20260930.json"
readonly REVIEWED_ARCHIVE_LOCK="/private/tmp/vokra-realtime-historical-lock-evidence-20261002/uv.lock"
readonly REVIEWED_ARCHIVE_MANIFEST="/private/tmp/vokra-realtime-historical-lock-evidence-20261002/artifact-rows.json"
readonly REVIEWED_ARCHIVE_HELPER="/private/tmp/vokra-ecapa-target-identity-20261002/tools/parity/vibevoice_realtime_0_5b_reference/audit_dependency_archives.py"
readonly REVIEWED_ARCHIVE_LEAF="/private/tmp/vokra-realtime-dependency-archive-collection-20261002.py"
readonly REVIEWED_ARCHIVE_UV_SHA256="58488ae8dbd0773134c92c85e901430e33f99d975bd7f929d26aa9ab0c2f9390"

model_marker() {
    printf '%s' "$1" | grep -Eiq \
        "(^|[[:space:]=/])(--?(model|checkpoint|weights)([-_](name|id|path|file))?|--?(gguf|safetensors|state-dict))([=[:space:]]|$)|\\.(gguf|safetensors|pt|pth|bin|ckpt)([[:space:]\"'<>|;&?]|$)|huggingface\\.co|/resolve/|/(model|checkpoint)([/._-]|[[:space:]]|$)|hf_hub_download|snapshot_download|from_pretrained|(^|[[:space:]])(hf|huggingface-cli)[[:space:]]+download([[:space:]]|$)"
}

is_help_or_self_test() {
    printf '%s' "$1" | grep -Eq '(^|[[:space:]])(--help|--self-test)([[:space:]]|$)'
}

is_read_only_inspection() {
    local normalized="$1"
    printf '%s' "$normalized" | grep -Eq \
        '^(sha(256|512)sum|shasum|b3sum|md5|stat|file|ls|find|rg|grep|head|tail|jq|od|xxd|cmp|diff|wc|sort|awk|sed|cat)([[:space:]]|$)|^git[[:space:]]+(status|diff|show|log|ls-files|rev-parse)([[:space:]]|$)'
}

is_model_shell_substitution() {
    local command="$1"
    case "$command" in
        *'$('*|*'`'*|*'<('*|*'>('*) ;;
        *) return 1 ;;
    esac
    if model_marker "$command"; then
        return 0
    fi
    # These workers can acquire and execute real weights without a model
    # filename in the outer command.  Do not let an inspection command hide
    # one inside command/process substitution.
    printf '%s' "$command" | grep -Eiq \
        '(^|[[:space:]/])(scripts/verify/apple-silicon|scripts/publish/vast-ai/run-[^[:space:]]+-(validation|parity))[^[:space:]]*|(^|[[:space:]/])tools/parity/[^[:space:]]+\.py([[:space:]]|\)|$)|(^|[[:space:]/])apple-silicon-[^[:space:]]*([[:space:]]|\)|$)'
}

is_literal_git_inspection() {
    local command="$1" token subcommand="" path resolved
    local index=1 count after_double_dash=0
    local -a argv

    case "$command" in
        *'$('*|*'`'*|*'>'*|*'<'*|*'&'*|*'|'*|*';'*|*'\\'*|*'"'*|*"'"*) return 1 ;;
    esac
    read -r -a argv <<< "$command"
    count="${#argv[@]}"
    [ "$count" -ge 2 ] || return 1
    [ "${argv[0]}" = git ] || return 1
    while [ "$index" -lt "$count" ]; do
        token="${argv[$index]}"
        case "$token" in
            -C)
                index=$((index + 1))
                [ "$index" -lt "$count" ] || return 1
                path="${argv[$index]}"
                case "$path" in
                    /*) ;;
                    *) return 1 ;;
                esac
                case "$path" in
                    *'/../'*|*/..|*'/./'*|*/.|*'"'*|*"'"*) return 1 ;;
                esac
                [ -d "$path" ] && [ ! -L "$path" ] || return 1
                resolved="$(CDPATH= cd -P -- "$path" 2>/dev/null && pwd -P)" || return 1
                [ "$resolved" = "$path" ] || return 1
                ;;
            status|diff|show|log|ls-files|rev-parse)
                subcommand="$token"
                index=$((index + 1))
                break
                ;;
            *) return 1 ;;
        esac
        index=$((index + 1))
    done
    [ -n "$subcommand" ] || return 1
    local saw_no_ext_diff=0 saw_no_textconv=0
    while [ "$index" -lt "$count" ]; do
        token="${argv[$index]}"
        if [ "$after_double_dash" -eq 1 ]; then
            index=$((index + 1))
            continue
        fi
        if [ "$token" = -- ]; then
            after_double_dash=1
            index=$((index + 1))
            continue
        fi
        case "$token" in
            --no-ext-diff) saw_no_ext_diff=1 ;;
            --no-textconv) saw_no_textconv=1 ;;
            --ext-diff|--textconv|--config*|--exec-path|--upload-pack=*|-c)
                return 1
                ;;
            --short|--porcelain|--branch|--ahead-behind|--no-ahead-behind|--ignored|--no-renames|-u|-uno|-unormal|-uall|--stat|--name-only|--name-status|--check|--cached|--staged|--no-color|--color=never|--oneline|--decorate|--graph|--format=*|--abbrev-commit|--deleted|--modified|--others|--stage|--eol|--exclude-standard|--show-toplevel|--git-dir|--is-inside-work-tree|--verify|--abbrev-ref|--show-prefix|--show-cdup|--absolute-git-dir|--git-path|-U[0-9]*)
                ;;
            -*) return 1 ;;
            *) ;;
        esac
        index=$((index + 1))
    done
    if [ "$subcommand" = diff ] || [ "$subcommand" = show ]; then
        [ "$saw_no_ext_diff" -eq 1 ] && [ "$saw_no_textconv" -eq 1 ] || return 1
    fi
    return 0
}

is_explicit_git_external_helper() {
    local normalized="$1"
    printf '%s' "$normalized" | grep -Eq '^git([[:space:]]|$)' || return 1
    printf '%s' "$normalized" | grep -Eq '(^|[[:space:]])(diff|show)([[:space:]]|$)' || return 1
    printf '%s' "$normalized" | grep -Eq '(^|[[:space:]])(--ext-diff|--textconv)([[:space:]]|$)'
}

is_static_command() {
    local normalized="$1"
    printf '%s' "$normalized" | grep -Eq \
        '^(bash[[:space:]]+-n|zsh[[:space:]]+-n|shellcheck|cargo[[:space:]]+(fmt|metadata|tree)|scripts/check-[^[:space:]]+|bash[[:space:]]+scripts/check-[^[:space:]]+|uv[[:space:]]+run[[:space:]].*(check_doc_examples|check[_-](doc|platform|abi|workflow|community)|self-test)|uv[[:space:]]+run[[:space:]]+.*tools/audit/|test[[:space:]]+-[fdb])([[:space:]]|$)'
}

is_model_download() {
    local normalized="$1"
    if ! model_marker "$normalized"; then return 1; fi

    # HEAD-only metadata probes do not acquire a model.
    if printf '%s' "$normalized" | grep -Eq '^(curl|wget)([[:space:]]|$)' \
        && printf '%s' "$normalized" | grep -Eq '(^|[[:space:]])(-I|--head|--spider)([[:space:]]|$)'; then
        return 1
    fi

    printf '%s' "$normalized" | grep -Eiq \
        '^(curl|wget|aria2c|scp|rsync|huggingface-cli|hf[[:space:]]+download|git[[:space:]]+clone|uv[[:space:]]+run|uv[[:space:]]+tool)([[:space:]]|$)|(^|[[:space:]])(curl|wget|aria2c)[[:space:]].*(https?://|--output|-O[[:space:]])'
}

is_literal_git_add() {
    local normalized="$1" shell_substitution="\$(" backtick="\`"
    # Indexing an already-present path is not model acquisition or execution.
    # Keep this deliberately narrower than generic Git handling: aliases,
    # `git -c`/`-C`, commits, and shell substitutions remain on the guarded
    # path.  Reject shell metacharacters before exempting the exact command.
    printf '%s' "$normalized" | grep -Eq '^git[[:space:]]+add([[:space:]]|$)' || return 1
    case "$normalized" in
        *"$shell_substitution"*|*"$backtick"*|*'>'*|*'<'*|*'&'*|*'|'*|*';'*) return 1 ;;
    esac
    return 0
}

archive_controller_digest_ok() {
    local path="$1" expected_path="${2:-$REVIEWED_ARCHIVE_CONTROLLER}" expected_sha="${3:-$REVIEWED_ARCHIVE_CONTROLLER_SHA256}" actual
    [ "$path" = "$expected_path" ] || return 1
    [ -f "$path" ] && [ ! -L "$path" ] || return 1
    actual="$(shasum -a 256 -- "$path" 2>/dev/null)" || return 1
    [ "${actual%% *}" = "$expected_sha" ]
}

is_reviewed_archive_controller_run() {
    local command="$1" controller_path="${2:-$REVIEWED_ARCHIVE_CONTROLLER}" controller_sha="${3:-$REVIEWED_ARCHIVE_CONTROLLER_SHA256}"
    local shell_substitution="\$(" backtick="\`" double_quote='"' single_quote="'" backslash="\\"
    local -a argv
    case "$command" in
        *"$shell_substitution"*|*"$backtick"*) return 1 ;;
        *'>'*|*'<'*|*'&'*|*'|'*|*';'*) return 1 ;;
        *"$double_quote"*|*"$single_quote"*|*"$backslash"*) return 1 ;;
    esac
    read -r -a argv <<< "$command"
    [ "${#argv[@]}" -ge 17 ] || return 1
    [ "${argv[0]}" = bash ] || return 1
    archive_controller_digest_ok "${argv[1]}" "$controller_path" "$controller_sha" || return 1
    [ "${argv[2]}" = --run ] || return 1
    [ "${argv[3]}" = --audit-json ] && [ "${argv[4]}" = "$REVIEWED_ARCHIVE_AUDIT" ] || return 1
    [ "${argv[5]}" = --lock ] && [ "${argv[6]}" = "$REVIEWED_ARCHIVE_LOCK" ] || return 1
    [ "${argv[7]}" = --manifest ] && [ "${argv[8]}" = "$REVIEWED_ARCHIVE_MANIFEST" ] || return 1
    [ "${argv[9]}" = --helper ] && [ "${argv[10]}" = "$REVIEWED_ARCHIVE_HELPER" ] || return 1
    [ "${argv[11]}" = --leaf ] && [ "${argv[12]}" = "$REVIEWED_ARCHIVE_LEAF" ] || return 1
    [ "${argv[13]}" = --offer-id ] && [[ "${argv[14]}" =~ ^[1-9][0-9]*$ ]] || return 1
    [ "${argv[15]}" = --uv-installer-sha256 ] && [ "${argv[16]}" = "$REVIEWED_ARCHIVE_UV_SHA256" ] || return 1
    if [ "${#argv[@]}" -eq 17 ]; then
        return 0
    fi
    [ "${#argv[@]}" -eq 19 ] && [ "${argv[17]}" = --output-root ] || return 1
    [[ "${argv[18]}" =~ ^/private/tmp/vokra-realtime-archive-collection-[A-Za-z0-9._-]+$ ]]
}

is_model_execution() {
    local normalized="$1"

    # The commands below are inspection-only even when they mention a model
    # file.  This keeps hash/manifest/license checks usable on the Mac.
    is_read_only_inspection "$normalized" && return 1
    is_static_command "$normalized" && return 1

    # A parity/reference/Apple worker can acquire its model internally, so it
    # is guarded even when the command line has no filename marker.
    if printf '%s' "$normalized" | grep -Eiq \
        '(^|[[:space:]/])(scripts/verify/apple-silicon|scripts/publish/vast-ai/run-[^[:space:]]+-(validation|parity))[^[:space:]]*|(^|[[:space:]/])tools/parity/[^[:space:]]+\.py([[:space:]]|$)'; then
        return 0
    fi

    model_marker "$normalized" || return 1

    if printf '%s' "$normalized" | grep -Eiq \
        '(^|[[:space:]])(([^[:space:]]*/)?vokra-cli|vokra|cargo[[:space:]]+run|cargo[[:space:]]+test|cargo[[:space:]]+bench)([[:space:]]|$)'; then
        printf '%s' "$normalized" | grep -Eiq \
            '(^|[[:space:]])(run|infer|forward|generate|transcribe|synthesize|convert|parity|validate|validation|bench)([[:space:]]|$)|(^|[[:space:]])(--?(model|checkpoint|weights|gguf|input))([=[:space:]]|$)'
        return $?
    fi

    # Unknown local runners are not trusted merely because their executable
    # name is unfamiliar.  A model filename passed to `bash runner.sh` or an
    # executable path is still model execution; only known read-only/static
    # commands were exempted above.
    printf '%s' "$normalized" | grep -Eiq \
        '^(bash|sh|zsh|node|deno|ruby|perl)[[:space:]]+[^[:space:]]+|^\.?/[^[:space:]]+([[:space:]]|$)|^/[^[:space:]]+([[:space:]]|$)'
    [ "$?" -eq 0 ] && return 0

    # Existing Apple/model validation workers are real-weight runners.  Their
    # self-test, help and shell-syntax modes were exempted above.
    printf '%s' "$normalized" | grep -Eiq \
        '(^|[[:space:]/])(scripts/(verify/apple-silicon|publish/vast-ai/run-[^[:space:]]+-(validation|parity))[^[:space:]]*|tools/parity/[^[:space:]]+\.py|apple-silicon-[^[:space:]]*)([[:space:]]|$)'
}

is_maintainer_mac() {
    [ "$1" = Darwin ]
}

is_local_ssh() {
    local normalized="$1" token host="" skip_value=0
    normalized="${normalized#ssh}"
    for token in $normalized; do
        if [ "$skip_value" -eq 1 ]; then
            skip_value=0
            continue
        fi
        case "$token" in
            -p|-o|-i|-F|-J|-l) skip_value=1 ;;
            -*) ;;
            *) host="$token"; break ;;
        esac
    done
    case "$host" in
        localhost|127.0.0.1|::1|*@localhost|*@127.0.0.1|*@::1) return 0 ;;
        *) return 1 ;;
    esac
}

analyse_segment() {
    local segment="$1" normalized
    normalized="$(hook_normalize_segment "$segment")"
    [ -n "$normalized" ] || return 1

    # Read-only/static/help exceptions are only safe when the command itself
    # is inspected.  A model-bearing command substitution or backtick can
    # execute before the outer inspection command, so fail closed before any
    # of those exceptions are considered.  This also catches substitutions in
    # quoted arguments; the hook deliberately does not attempt to interpret
    # shell quoting.
    if is_model_shell_substitution "$segment"; then
        echo "model/checkpoint execution"
        return 0
    fi

    is_literal_git_inspection "$segment" && return 1

    # One immutable, hash-bound remote controller is allowed to carry its
    # remote archive inputs through this local hook.  The recognizer is exact:
    # it does not permit wrappers, aliases, substitutions, shell chaining, or
    # any unreviewed path/value.  All other tools/parity Python remains blocked.
    is_reviewed_archive_controller_run "$segment" && return 1

    # Remote execution is deliberately outside this local-machine guard. The
    # VAST mutation guard separately controls cloud lifecycle operations.
    case "$normalized" in
        ssh\ *|scripts/publish/vast-ai/vastai-safe.sh\ *|*/vastai-safe.sh\ *)
            if is_local_ssh "$normalized"; then
                :
            else
                return 1
            fi
            ;;
        scp\ *|rsync\ *)
            # Copying a model/checkpoint to the Mac is acquisition, not a
            # remote worker exemption; model_marker is checked below.
            ;;
    esac
    is_help_or_self_test "$normalized" && return 1
    is_static_command "$normalized" && return 1
    is_literal_git_add "$normalized" && return 1
    if is_explicit_git_external_helper "$normalized"; then
        echo "unsafe git external diff/textconv request"
        return 0
    fi

    if is_model_download "$normalized"; then
        echo "model/checkpoint download"
        return 0
    fi
    if is_model_execution "$normalized"; then
        echo "model/checkpoint execution"
        return 0
    fi
    return 1
}

analyse_command() {
    local command="$1" segment reason
    while IFS= read -r segment; do
        [ -n "$(hook_trim "$segment")" ] || continue
        if reason="$(analyse_segment "$segment")"; then
            echo "$reason"
            return 0
        fi
    done < <(hook_segments "$command")
    return 1
}

self_test() {
    local fails=0 got
    check() {
        local name="$1" expected="$2" command="$3"
        if analyse_command "$command" >/dev/null; then got=block; else got=allow; fi
        if [ "$got" = "$expected" ]; then
            printf '  ok    %-54s %s\n' "$name" "$got"
        else
            printf '  FAIL  %-54s expected %s, got %s\n' "$name" "$expected" "$got"
            fails=$((fails + 1))
        fi
    }

    echo 'guard-local-models --self-test'
    check 'small GGUF execution' block 'vokra-cli run --model ./tiny.gguf'
    check 'checkpoint conversion' block 'vokra-cli convert --input ./model.safetensors'
    check 'target path model execution' block './target/release/vokra-cli run --model ./model.gguf'
    check 'unknown bash runner' block 'bash custom-runner.sh model.gguf'
    check 'unknown executable runner' block './custom-runner model.gguf'
    check 'HF download' block 'hf download org/model --include *.safetensors'
    check 'HF download without include' block 'huggingface-cli download org/model'
    check 'snapshot download' block 'uv run python -c snapshot_download()'
    check 'curl model download' block 'curl -L https://huggingface.co/org/model/resolve/main/model.gguf -o model.gguf'
    check 'uv reference download' block 'uv run --project tools/parity python -c from_pretrained()'
    check 'Apple worker execution' block 'bash scripts/verify/apple-silicon-omniasr-ctc.sh --packet packet.gguf'
    check 'chained download' block 'echo begin && wget https://example.invalid/a.gguf'
    check 'remote SSH model command' allow 'ssh vast-worker vokra-cli run --model /remote/model.gguf'
    check 'localhost SSH model command' block 'ssh localhost vokra-cli run --model /local/model.gguf'
    check 'loopback SSH model command' block 'ssh 127.0.0.1 vokra-cli run --model /local/model.gguf'
    check 'loopback SSH with port' block 'ssh -p 22 localhost vokra-cli run --model /local/model.gguf'
    check 'loopback SSH user host' block 'ssh user@localhost vokra-cli run --model /local/model.gguf'
    check 'scp checkpoint copy' block 'scp vast-worker:/remote/model.gguf ./model.gguf'
    check 'rsync checkpoint copy' block 'rsync vast-worker:/remote/model.safetensors ./model.safetensors'
    check 'VAST safe wrapper' allow 'scripts/publish/vast-ai/vastai-safe.sh show instance 123'
    check 'hash inspection' allow 'sha256sum ./model.gguf'
    check 'manifest inspection' allow 'jq . tools/parity/license_manifest.json'
    check 'metadata HEAD probe' allow 'curl -I https://huggingface.co/org/model/resolve/main/model.gguf'
    check 'metadata spider probe' allow 'wget --spider https://huggingface.co/org/model/resolve/main/model.gguf'
    check 'static shell syntax' allow 'bash -n scripts/verify/apple-silicon-omniasr-ctc.sh'
    check 'static repository gate' allow 'scripts/check-doc-references.sh --self-test'
    check 'help is safe' allow 'vokra-cli run --model ./model.gguf --help'
    check 'self-test is safe' allow 'bash scripts/verify/apple-silicon-omniasr-ctc.sh --self-test'
    reviewed_controller_cmd="bash $REVIEWED_ARCHIVE_CONTROLLER --run --audit-json $REVIEWED_ARCHIVE_AUDIT --lock $REVIEWED_ARCHIVE_LOCK --manifest $REVIEWED_ARCHIVE_MANIFEST --helper $REVIEWED_ARCHIVE_HELPER --leaf $REVIEWED_ARCHIVE_LEAF --offer-id 48161395 --uv-installer-sha256 $REVIEWED_ARCHIVE_UV_SHA256 --output-root /private/tmp/vokra-realtime-archive-collection-selftest"
    if [ -f "$REVIEWED_ARCHIVE_CONTROLLER" ] && [ ! -L "$REVIEWED_ARCHIVE_CONTROLLER" ] && archive_controller_digest_ok "$REVIEWED_ARCHIVE_CONTROLLER"; then
        check 'reviewed archive controller exact remote run' allow "$reviewed_controller_cmd"
    else
        check 'reviewed archive controller absent or mismatched' block "$reviewed_controller_cmd"
    fi
    check 'reviewed controller unknown runner' block "zsh ${reviewed_controller_cmd#bash }"
    check 'reviewed controller changed argument' block "${reviewed_controller_cmd/--offer-id 48161395/--offer-id 48161395 --unexpected}"
    check 'reviewed controller chained local model' block "$reviewed_controller_cmd && vokra-cli run --model ./tiny.gguf"
    check 'direct Python parity remains blocked' block 'uv run --project tools/parity python tools/parity/firered_decoder.py'
    production_target_state=absent
    production_target_hash=
    if [ -L "$REVIEWED_ARCHIVE_CONTROLLER" ]; then
        production_target_state=symlink
    elif [ -f "$REVIEWED_ARCHIVE_CONTROLLER" ]; then
        production_target_state=regular
        production_target_hash="$(shasum -a 256 -- "$REVIEWED_ARCHIVE_CONTROLLER" 2>/dev/null | awk '{print $1}')"
    elif [ -e "$REVIEWED_ARCHIVE_CONTROLLER" ]; then
        production_target_state=other
    fi
    fixture_root="$(mktemp -d "${TMPDIR:-/tmp}/vokra-archive-hook.XXXXXX")"
    fixture_controller="$fixture_root/$(basename "$REVIEWED_ARCHIVE_CONTROLLER")"
    rm -f "$fixture_controller"
    printf '%s\n' 'synthetic inert controller fixture' > "$fixture_controller"
    fixture_sha="$(shasum -a 256 -- "$fixture_controller" | awk '{print $1}')"
    fixture_command="${reviewed_controller_cmd/$REVIEWED_ARCHIVE_CONTROLLER/$fixture_controller}"
    if is_reviewed_archive_controller_run "$fixture_command" "$fixture_controller" "$fixture_sha"; then
        printf '  ok    %-54s %s\n' 'fixture trusted controller exact hash' allow
    else
        printf '  FAIL  %-54s\n' 'fixture trusted controller exact hash'; fails=$((fails + 1))
    fi
    printf '# changed\n' >> "$fixture_controller"
    if is_reviewed_archive_controller_run "$fixture_command" "$fixture_controller" "$fixture_sha"; then
        printf '  FAIL  %-54s\n' 'fixture changed hash remains blocked'; fails=$((fails + 1))
    else
        printf '  ok    %-54s %s\n' 'fixture changed hash remains blocked' block
    fi
    rm -f "$fixture_controller"
    ln -s "$REVIEWED_ARCHIVE_CONTROLLER" "$fixture_controller"
    if is_reviewed_archive_controller_run "$fixture_command" "$fixture_controller" "$REVIEWED_ARCHIVE_CONTROLLER_SHA256"; then
        printf '  FAIL  %-54s\n' 'fixture symlink remains blocked'; fails=$((fails + 1))
    else
        printf '  ok    %-54s %s\n' 'fixture symlink remains blocked' block
    fi
    rm -f "$fixture_controller"
    printf '%s\n' 'synthetic inert controller fixture' > "$fixture_controller"
    fixture_sha="$(shasum -a 256 -- "$fixture_controller" | awk '{print $1}')"
    if is_reviewed_archive_controller_run "env FOO=bar $fixture_command" "$fixture_controller" "$fixture_sha"; then
        printf '  FAIL  %-54s\n' 'fixture env wrapper remains blocked'; fails=$((fails + 1))
    else
        printf '  ok    %-54s %s\n' 'fixture env wrapper remains blocked' block
    fi
    if is_reviewed_archive_controller_run "${fixture_command/--offer-id 48161395/--offer-id \$(printf 48161395)}" "$fixture_controller" "$fixture_sha"; then
        printf '  FAIL  %-54s\n' 'fixture substitution remains blocked'; fails=$((fails + 1))
    else
        printf '  ok    %-54s %s\n' 'fixture substitution remains blocked' block
    fi
    if is_reviewed_archive_controller_run "${fixture_command/--output-root \/private\/tmp\/vokra-realtime-archive-collection-selftest/--output-root \/private\/tmp\/vokra-realtime-archive-collection-..\/escape}" "$fixture_controller" "$fixture_sha"; then
        printf '  FAIL  %-54s\n' 'fixture output traversal remains blocked'; fails=$((fails + 1))
    else
        printf '  ok    %-54s %s\n' 'fixture output traversal remains blocked' block
    fi
    case "$production_target_state" in
        absent)
            if [ -e "$REVIEWED_ARCHIVE_CONTROLLER" ] || [ -L "$REVIEWED_ARCHIVE_CONTROLLER" ]; then
                printf '  FAIL  %-54s\n' 'production controller self-test target unchanged'; fails=$((fails + 1))
            else
                printf '  ok    %-54s %s\n' 'production controller self-test target unchanged' absent
            fi
            ;;
        regular)
            production_target_hash_after="$(shasum -a 256 -- "$REVIEWED_ARCHIVE_CONTROLLER" 2>/dev/null | awk '{print $1}')"
            if [ -L "$REVIEWED_ARCHIVE_CONTROLLER" ] || [ ! -f "$REVIEWED_ARCHIVE_CONTROLLER" ] || [ "$production_target_hash_after" != "$production_target_hash" ]; then
                printf '  FAIL  %-54s\n' 'production controller self-test target unchanged'; fails=$((fails + 1))
            else
                printf '  ok    %-54s %s\n' 'production controller self-test target unchanged' "$production_target_hash_after"
            fi
            ;;
        symlink)
            if [ ! -L "$REVIEWED_ARCHIVE_CONTROLLER" ]; then
                printf '  FAIL  %-54s\n' 'production controller self-test target unchanged'; fails=$((fails + 1))
            else
                printf '  ok    %-54s %s\n' 'production controller self-test target unchanged' symlink
            fi
            ;;
        other)
            if [ ! -e "$REVIEWED_ARCHIVE_CONTROLLER" ] && [ ! -L "$REVIEWED_ARCHIVE_CONTROLLER" ]; then
                printf '  FAIL  %-54s\n' 'production controller self-test target unchanged'; fails=$((fails + 1))
            else
                printf '  ok    %-54s %s\n' 'production controller self-test target unchanged' other
            fi
            ;;
    esac
    if [ -f "$REVIEWED_ARCHIVE_CONTROLLER" ] && [ ! -L "$REVIEWED_ARCHIVE_CONTROLLER" ] && archive_controller_digest_ok "$REVIEWED_ARCHIVE_CONTROLLER"; then
        check 'production controller exact immutable path' allow "$reviewed_controller_cmd"
    else
        check 'production controller absent or mismatched remains blocked' block "$reviewed_controller_cmd"
    fi
    rm -rf "$fixture_root"
    check 'dry-run is not a bypass' block 'vokra-cli convert --model ./model.gguf --dry-run'
    check 'no-download is not a bypass' block 'uv run python -c from_pretrained() --no-download'
    check 'ordinary parity script is guarded' block 'uv run --project tools/parity python tools/parity/foo.py'
    check 'audit script is model-free' allow 'uv run --no-project python tools/audit/hf_mac_coverage.py'
    check 'static check script with marker' allow 'bash scripts/check-doc-references.sh model.gguf'
    check 'git inspection substitution with model is blocked' block 'git diff --no-ext-diff --no-textconv -- README.md $(vokra-cli run --model ./model.gguf)'
    check 'read-only substitution with model is blocked' block 'cat $(vokra-cli run --model ./model.gguf)'
    check 'quoted substitution with model is blocked' block 'git diff --no-ext-diff --no-textconv -- README.md "$(vokra-cli run --model ./model.gguf)"'
    check 'backtick substitution with model is blocked' block 'sha256sum `vokra-cli run --model ./model.gguf`'
    check 'parity substitution without model marker is blocked' block 'cat $(uv run --project tools/parity python tools/parity/foo.py)'
    check 'process substitution with model is blocked' block 'cat <(vokra-cli run --model ./tiny.gguf)'
    check 'tee process substitution with model is blocked' block 'tee >(vokra-cli run --model ./tiny.gguf) >/dev/null'
    check 'bash syntax process substitution with model is blocked' block 'bash -n <(vokra-cli run --model ./tiny.gguf)'
    check 'help plus model substitution is blocked' block 'vokra-cli run --model ./model.gguf --help $(printf x)'
    check 'self-test plus model substitution is blocked' block 'bash custom-runner.sh model.gguf --self-test $(printf x)'
    check 'ordinary inspection substitution remains allowed' allow 'git diff --no-ext-diff --no-textconv -- README.md $(printf README.md)'
    git_inspection_fixture="$(mktemp -d "${TMPDIR:-/tmp}/vokra-git-inspection.XXXXXX")"
    git_inspection_fixture="$(CDPATH= cd -P -- "$git_inspection_fixture" 2>/dev/null && pwd -P)"
    git_inspection_link="${git_inspection_fixture}-link"
    ln -s "$git_inspection_fixture" "$git_inspection_link"
    git_inspection_prefix="git -C $git_inspection_fixture"
    check 'git -C status inspection' allow "$git_inspection_prefix status --short"
    check 'git -C log inspection' allow "$git_inspection_prefix log --oneline -- tools/parity/probe.py"
    check 'git -C ls-files inspection' allow "$git_inspection_prefix ls-files -- tools/parity/probe.py"
    check 'git -C rev-parse inspection' allow "$git_inspection_prefix rev-parse --show-toplevel"
    check 'git -C diff with helper protections' allow "$git_inspection_prefix diff --no-ext-diff --no-textconv -- model.gguf tools/parity/probe.py"
    check 'git -C show with helper protections' allow "$git_inspection_prefix show --no-ext-diff --no-textconv HEAD -- model.gguf"
    check 'git -C diff without helper ordinary path' allow "$git_inspection_prefix diff -- README.md"
    check 'plain git diff inspection remains allowed' allow 'git diff -- tools/parity/probe.py'
    check 'plain git show inspection remains allowed' allow 'git show HEAD:tools/parity/probe.py'
    check 'plain git diff external helper rejected' block 'git diff --ext-diff -- tools/parity/probe.py'
    check 'plain git show textconv rejected' block 'git show --textconv HEAD:tools/parity/probe.py'
    check 'git -C external diff rejected' block "$git_inspection_prefix diff --ext-diff --no-textconv -- tools/parity/probe.py"
    check 'git -C textconv rejected' block "$git_inspection_prefix show --no-ext-diff --textconv HEAD"
    check 'git -C config remains ordinary Git' allow "git -c user.name=bot -C $git_inspection_fixture status"
    check 'git env wrapper remains ordinary Git' allow "env GIT_EXTERNAL_DIFF=helper $git_inspection_prefix status"
    check 'git symlink directory remains ordinary Git' allow "git -C $git_inspection_link status"
    check 'git traversal directory remains ordinary Git' allow "git -C $git_inspection_fixture/../$(basename "$git_inspection_fixture") status"
    check 'git unknown subcommand remains ordinary Git' allow "$git_inspection_prefix remote -v"
    check 'git commit remains ordinary Git' allow 'git commit -m docs'
    check 'git fetch remains ordinary Git' allow 'git fetch origin main'
    check 'git bundle verify remains ordinary Git' allow 'git bundle verify candidate.bundle'
    check 'git chained model execution rejected' block "$git_inspection_prefix diff --no-ext-diff --no-textconv -- tools/parity/probe.py && uv run --project tools/parity python tools/parity/foo.py"
    rm -rf "$git_inspection_fixture" "$git_inspection_link"
    check 'literal git add parity source' allow 'git add -- tools/parity/new_probe.py'
    check 'literal git add checkpoint path' allow 'git add -- model.gguf'
    check 'git add chained model execution' block 'git add -- tools/parity/new_probe.py && uv run --project tools/parity python tools/parity/foo.py'
    check 'git -c add remains guarded' block 'git -c user.name=bot add -- tools/parity/new_probe.py'
    check 'git commit with parity source remains guarded' block 'git commit --only -- tools/parity/new_probe.py -m docs'
    check 'prose false positive' allow 'echo "download model.gguf later"'
    check 'prose chained false positive' allow 'echo "ssh vast worker" && git status --short'
    if is_maintainer_mac Darwin && ! is_maintainer_mac Linux; then
        printf '  ok    %-54s %s\n' 'OS gate Darwin-only' allow
    else
        printf '  FAIL  %-54s expected Darwin-only gate\n' 'OS gate Darwin-only'
        fails=$((fails + 1))
    fi

    if [ "$fails" -eq 0 ]; then
        echo 'guard-local-models --self-test: OK'
        return 0
    fi
    echo "guard-local-models --self-test: FAIL ($fails)"
    return 1
}

if [ "${1:-}" = --self-test ]; then
    self_test
    exit $?
fi

payload="$(cat)"
command="$(hook_command "$payload")"
[ -n "$command" ] || exit 0
# This guard is for the maintainer Mac only.  The remote VAST/Linux checkout
# is precisely where real-weight work is expected to run.
is_maintainer_mac "$(uname -s 2>/dev/null || echo unknown)" || exit 0
if reason="$(analyse_command "$command")"; then
    {
        echo "Blocked: local $reason is prohibited on the maintainer Mac."
        echo "Use the audited VAST/remote worker path for real model work."
        echo "Read-only hash, manifest, metadata, --help and --self-test operations remain allowed."
    } >&2
    exit 2
fi
exit 0
