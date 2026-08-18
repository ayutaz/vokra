#!/usr/bin/env bash
# Claude Code PreToolUse hook (Bash): refuse commands that have been measured
# to exhaust this machine's memory, and say where to run them instead.
#
# WHY THIS EXISTS
# ---------------
# 2026-08-16: `cargo test -p vokra-models --lib kyutai_stt` was OOM-killed
# (exit 137) on the maintainer's 16 GB M1 and **rebooted macOS**. A memory
# note recording "offload >8 GB models to vast.ai" already existed and did
# not prevent it: guidance is advisory, and an agent under task pressure
# reaches for the local command anyway. Enforcement has to live where the
# harness runs it, not where the agent remembers it.
#
# The threshold for model artefacts is 2 GB by maintainer instruction
# (2026-08-16), which is deliberately stricter than the measured-safe set
# (whisper-* at 2.9 GB and csm-1b at 6.21 GB have both converted locally).
# Being stricter than measured is the point: the cost of a false block is
# one env var, the cost of a false allow is a reboot.
#
# GIT PUSH
# --------
# This command guard does not parse git's ref-update stream. `.githooks/pre-push`
# owns that boundary: deletion-only pushes run compliance but skip cargo,
# docs-only pushes use the light path, and a deep path on the maintainer Mac
# refuses before cargo.
#
# ESCAPE HATCH
#   VOKRA_ALLOW_LOCAL_HEAVY=1  — you have decided this specific run is safe.
#
# SELF-TEST
#   bash scripts/claude-hooks/guard-local-memory.sh --self-test
#
# Exit 2 = block the tool call and return the message to Claude.

set -uo pipefail

# Model artefacts at or above this size go to vast.ai, never local.
readonly SIZE_LIMIT_BYTES=$((2 * 1024 * 1024 * 1024))

# The owner explicitly routes every compiling/testing/checking/documenting/
# auditing `vokra-models` Cargo command to VAST, even when a particular test has
# since been made smaller. This is an operational memory ceiling, not a claim
# that every invocation always OOMs.
readonly HEAVY_CRATES='vokra-models'

emit_heavy_cargo_block() {
    {
        echo "Blocked: this cargo scope is reserved for VAST to protect the"
        echo "maintainer machine (16 GB M1). A local vokra-models run was"
        echo "OOM-killed and rebooted macOS on 2026-08-16."
        echo
        echo "Matched: $1"
        echo
        echo "Run it on vast.ai instead — load the 'vast-ai-workflow' skill. A"
        echo "48-core / 125 GB box costs about \$0.08/hr and ran the full workspace"
        echo "(6965 tests) in minutes for \$0.03. Ship unpushed commits to it with"
        echo "'git bundle create <f> <base>..HEAD' so nothing unverified is pushed."
        echo
        echo "Locally, scope to one light crate instead:"
        echo "  CARGO_BUILD_JOBS=1 cargo test -p vokra-convert   # or -cli / -eval / -core"
        echo
        echo "If you have genuinely decided this run is safe, re-issue it with"
        echo "VOKRA_ALLOW_LOCAL_HEAVY=1 prefixed."
    } >&2
}

emit_large_artefact_block() {
    {
        echo "Blocked: this command reads a model artefact of $2 (>= 2 GB), which"
        echo "the maintainer has directed must not be processed on this machine."
        echo
        echo "Path: $1"
        echo
        echo "Convert it on vast.ai — load the 'vast-ai-workflow' skill for the"
        echo "rent -> provision -> work -> destroy lifecycle. Remember to destroy"
        echo "the instance ('vastai destroy instance <id>'); it bills per hour."
        echo
        echo "Exception worth checking first: if you only need to change provenance"
        echo "metadata and not the tensors, 'restamp_provenance' rewrites in-place"
        echo "via mmap and has published an 8.7 GB file at 6.4 MB peak locally."
        echo
        echo "If you have genuinely decided this run is safe, re-issue it with"
        echo "VOKRA_ALLOW_LOCAL_HEAVY=1 prefixed."
    } >&2
}

human_size() {
    awk -v b="$1" 'BEGIN { printf "%.1f GB", b / 1073741824 }'
}

# `stat` is not portable: macOS (BSD) wants -f '%z', Linux (GNU) wants -c '%s'.
#
# The first version tried BSD then fell back with `||`, which does not work and
# CI caught it: GNU `stat -f` is not an error, it is a DIFFERENT valid option
# (print filesystem status). So on Linux the BSD form SUCCEEDED, printed
# filesystem info, the `||` never fired, and the arithmetic that consumed the
# result died with "File: unbound variable". Every size check would have been
# garbage rather than merely wrong.
#
# Two fixes, because exit status alone is not trustworthy here:
#   1. try GNU first — BSD `stat -c` is genuinely invalid and fails cleanly,
#      so the asymmetry now runs the safe way round;
#   2. validate the answer is a plain integer regardless, and report 0 if not.
file_size_bytes() {
    local out
    out="$(stat -c '%s' "$1" 2>/dev/null)" || out="$(stat -f '%z' "$1" 2>/dev/null)" || out=""
    case "$out" in
        '' | *[!0-9]*) echo 0 ;;
        *) echo "$out" ;;
    esac
}

# Size of a file, or the recursive sum for a directory (sharded safetensors
# live as many files in one directory, and each shard alone can be under the
# limit while the set is far over it).
path_size_bytes() {
    local p="$1" total=0 f
    if [ -d "$p" ]; then
        while IFS= read -r f; do
            total=$((total + $(file_size_bytes "$f")))
        done <<EOF
$(find "$p" -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' \
    -o -name '*.pth' -o -name '*.gguf' -o -name '*.ckpt' \) 2>/dev/null)
EOF
        echo "$total"
    elif [ -f "$p" ]; then
        file_size_bytes "$p"
    else
        echo 0
    fi
}

# True when some command segment actually INVOKES a workspace-reading Cargo
# subcommand covered by the owner routing rule, as opposed to merely containing
# the words. Segments are split on
# ; && || | and newline; leading VAR=value assignments and a +toolchain are
# skipped so `FOO=1 cargo +stable test` still matches.
segment_invokes_heavy_cargo() {
    printf '%s\n' "$1" \
        | tr ';\n' '\n\n' \
        | sed -e 's/&&/\n/g' -e 's/||/\n/g' -e 's/|/\n/g' \
        | while IFS= read -r seg; do
            # Drop leading whitespace, then leading NAME=VALUE assignments.
            seg="${seg#"${seg%%[![:space:]]*}"}"
            while printf '%s' "$seg" | grep -Eq '^[A-Za-z_][A-Za-z0-9_]*='; do
                seg="${seg#* }"
                seg="${seg#"${seg%%[![:space:]]*}"}"
            done
            printf '%s' "$seg" \
                | grep -Eq '^cargo[[:space:]]+(\+[^[:space:]]+[[:space:]]+)?(test|build|check|clippy|bench|nextest|run|doc|rustdoc|deny|audit)([[:space:]]|$)' \
                && echo MATCH
        done | grep -q MATCH
}

# ---------------------------------------------------------------- analysis --
# Returns 0 and prints a reason when the command must be blocked.
analyse() {
    local cmd="$1"

    # An explicit override anywhere in the command line is honoured.
    if printf '%s' "$cmd" | grep -Eq '(^|[;&|[:space:]])VOKRA_ALLOW_LOCAL_HEAVY=1'; then
        return 1
    fi

    # -- (a) memory-heavy cargo -------------------------------------------
    # Cargo subcommands that compile, test, document, run, or audit the selected
    # workspace scope. `cargo fmt`, `cargo metadata`, and `cargo tree` are cheap
    # and must not be caught.
    #
    # `cargo` must be the FIRST WORD of a command segment, not merely present.
    # Matching it anywhere blocked `echo 'run cargo test later'` — caught by
    # this file's own self-test. A guard that trips on prose about a command
    # gets switched off, and then it guards nothing.
    if segment_invokes_heavy_cargo "$cmd"; then

        if printf '%s' "$cmd" | grep -Eq '[[:space:]]--(workspace|all)([[:space:]]|$)'; then
            echo "HEAVY_CARGO:workspace-wide scope (--workspace / --all)"
            return 0
        fi

        # Skipped explicitly when the list is empty. Interpolating an empty
        # string would leave `(...)` as an empty alternation, which matches
        # the empty string — a regex that is one stray space away from
        # blocking every `-p` invocation there is.
        if [ -n "$HEAVY_CRATES" ] \
            && printf '%s' "$cmd" | grep -Eq "[[:space:]]-p[[:space:]]+($HEAVY_CRATES)([[:space:]]|$)"; then
            echo "HEAVY_CARGO:-p names a crate measured to exhaust this machine"
            return 0
        fi

        # A bare `cargo test` in a virtual workspace IS the workspace. Treat
        # "no -p and no --package" as workspace scope rather than assuming
        # the caller meant something small.
        if ! printf '%s' "$cmd" | grep -Eq '[[:space:]](-p|--package)[[:space:]]'; then
            echo "HEAVY_CARGO:no -p given, which in this virtual workspace means every crate"
            return 0
        fi
    fi

    # -- (b) model artefacts at or over the limit --------------------------
    # Candidate paths: values of --input/--config/--model/--output, plus any
    # bare token that looks like a checkpoint. Quoting is not fully parsed;
    # a path with spaces simply will not resolve and is left alone, which
    # errs toward allowing rather than blocking on a mis-parse.
    local tok prev="" size
    for tok in $cmd; do
        case "$prev" in
            --input|--config|--model|--output|-i|-o) : ;;
            *)
                case "$tok" in
                    *.safetensors|*.gguf|*.pt|*.pth|*.bin|*.ckpt) : ;;
                    *) prev="$tok"; continue ;;
                esac
                ;;
        esac
        prev="$tok"

        # Strip surrounding quotes a naive split leaves behind.
        tok="${tok%\"}"; tok="${tok#\"}"
        tok="${tok%\'}"; tok="${tok#\'}"
        [ -e "$tok" ] || continue

        size="$(path_size_bytes "$tok")"
        if [ "${size:-0}" -ge "$SIZE_LIMIT_BYTES" ] 2>/dev/null; then
            echo "LARGE_ARTEFACT:$tok:$(human_size "$size")"
            return 0
        fi
    done

    return 1
}

# --------------------------------------------------------------- self-test --
# Plants each defect where the guard must see it. A guard that has stopped
# being able to see one is worse than no guard: it certifies what it missed.
if [ "${1:-}" = "--self-test" ]; then
    fails=0
    check() { # name expect-block command
        local name="$1" expect="$2" c="$3" got
        if analyse "$c" >/dev/null; then got=block; else got=allow; fi
        if [ "$got" = "$expect" ]; then
            printf '  ok    %-52s %s\n' "$name" "$got"
        else
            printf '  FAIL  %-52s expected %s, got %s\n' "$name" "$expect" "$got"
            fails=$((fails + 1))
        fi
    }

    echo "guard-local-memory --self-test"

    # (a) heavy cargo — must block
    check "cargo test --workspace"            block "cargo test --workspace"
    check "bare cargo test (= workspace)"     block "cargo test"
    check "cargo nextest run --workspace"     block "cargo nextest run --workspace"
    check "cargo build --all"                 block "cargo build --all --release"
    check "cargo check --workspace"           block "cargo check --workspace --all-targets"
    check "bare cargo deny (= workspace)"     block "cargo deny check licenses advisories bans"
    check "bare cargo audit (= workspace)"    block "cargo audit"
    check "toolchain-pinned workspace test"   block "cargo +stable test --workspace"
    check "chained after &&"                  block "cd /tmp && cargo test --workspace"

    # (a) explicit heavy crate — must block even when the named test is small.
    check "cargo test -p vokra-models"        block "cargo test -p vokra-models --lib kyutai_stt"
    check "cargo check -p vokra-models"       block "cargo check -p vokra-models"

    # (a) light cargo — must NOT block
    check "cargo test -p vokra-convert"       allow "cargo test -p vokra-convert"
    check "cargo test -p vokra-cli"           allow "CARGO_BUILD_JOBS=1 cargo test -p vokra-cli"
    check "cargo fmt"                         allow "cargo fmt --all -- --check"
    check "cargo metadata"                    allow "cargo metadata --no-deps"
    check "cargo tree"                        allow "cargo tree -p vokra-core"
    check "override honoured"                 allow "VOKRA_ALLOW_LOCAL_HEAVY=1 cargo test --workspace"
    check "unrelated command"                 allow "git status --short"
    check "the word cargo in prose"           allow "echo 'run cargo test later'"

    # (b) artefact size — needs real files on disk
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    mkdir -p "$tmp/shards"
    # Sparse files: apparent size is what stat reports, no disk is consumed.
    dd if=/dev/null of="$tmp/small.safetensors" bs=1 count=0 seek=1000000 2>/dev/null
    dd if=/dev/null of="$tmp/big.safetensors"   bs=1 count=0 seek=3000000000 2>/dev/null
    # Each shard is under the limit; the set is over it.
    dd if=/dev/null of="$tmp/shards/a.safetensors" bs=1 count=0 seek=1500000000 2>/dev/null
    dd if=/dev/null of="$tmp/shards/b.safetensors" bs=1 count=0 seek=1500000000 2>/dev/null

    check "3 GB checkpoint"                   block "vokra-cli convert --model whisper --input $tmp/big.safetensors"
    check "sharded dir summing over limit"    block "vokra-cli convert --model x --input $tmp/shards"
    check "1 MB checkpoint"                   allow "vokra-cli convert --model x --input $tmp/small.safetensors"
    check "path that does not exist"          allow "vokra-cli convert --model x --input /nope/absent.safetensors"
    check "big file but override set"         allow "VOKRA_ALLOW_LOCAL_HEAVY=1 vokra-cli convert --input $tmp/big.safetensors"

    echo
    if [ "$fails" -eq 0 ]; then
        echo "guard-local-memory --self-test: OK"
        exit 0
    fi
    echo "guard-local-memory --self-test: FAIL ($fails)"
    exit 1
fi

# ------------------------------------------------------------------ driver --
payload="$(cat)"

cmd=""
if command -v jq >/dev/null 2>&1; then
    cmd="$(printf '%s' "$payload" | jq -r '.tool_input.command // empty' 2>/dev/null || true)"
elif command -v uv >/dev/null 2>&1; then
    cmd="$(printf '%s' "$payload" \
        | UV_CACHE_DIR="${TMPDIR:-/tmp}/vokra-uv-cache" uv run --no-project --python 3.12 python -c 'import sys,json; print(json.load(sys.stdin).get("tool_input",{}).get("command",""))' \
        2>/dev/null || true)"
fi

[ -n "$cmd" ] || exit 0

# The environment variable may also be exported rather than inlined.
[ "${VOKRA_ALLOW_LOCAL_HEAVY:-0}" = "1" ] && exit 0

if reason="$(analyse "$cmd")"; then
    case "$reason" in
        HEAVY_CARGO:*)
            emit_heavy_cargo_block "${reason#HEAVY_CARGO:}"
            ;;
        LARGE_ARTEFACT:*)
            rest="${reason#LARGE_ARTEFACT:}"
            emit_large_artefact_block "${rest%:*}" "${rest##*:}"
            ;;
    esac
    exit 2
fi

exit 0
