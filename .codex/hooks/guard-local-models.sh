#!/usr/bin/env bash
# Codex PreToolUse hook: keep model acquisition and model execution off the
# maintainer Mac.  This is intentionally separate from guard-local-memory.sh:
# the memory guard is size/Cargo based, while this guard applies even to a
# small checkpoint and does not offer a general local-heavy escape hatch.

set -uo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

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
        '^(git[[:space:]]+(status|diff|show|log|ls-files|rev-parse)|sha(256|512)sum|shasum|b3sum|md5|stat|file|ls|find|rg|grep|head|tail|jq|od|xxd|cmp|diff|wc|sort|awk|sed|cat)([[:space:]]|$)'
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
    check 'dry-run is not a bypass' block 'vokra-cli convert --model ./model.gguf --dry-run'
    check 'no-download is not a bypass' block 'uv run python -c from_pretrained() --no-download'
    check 'ordinary parity script is guarded' block 'uv run --project tools/parity python tools/parity/foo.py'
    check 'audit script is model-free' allow 'uv run --no-project python tools/audit/hf_mac_coverage.py'
    check 'static check script with marker' allow 'bash scripts/check-doc-references.sh model.gguf'
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
