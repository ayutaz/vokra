#!/usr/bin/env bash
# check-capi-thread-free.sh — machine gate for the production C ABI
# thread-spawn-0 invariant (M4-02-T02, ADR M4-02 §4).
#
# Why: Unity WebGL executes the statically linked Vokra objects on the
# browser main thread of a single-threaded WASM build (threads OFF baseline,
# NFR-RL-02 / ADR M4-02 §4). The rustup-distributed std for
# wasm32-unknown-emscripten has no real thread support (no atomics), so any
# `std::thread::spawn` reachable from the C ABI would be a guaranteed runtime
# failure. The M4-02 spec audit found every `std::thread` use lives under a
# trailing `#[cfg(test)]` module (production streaming is poll-based, M1-08);
# this script turns that one-time audit into a repeatable source-level gate.
#
# How: greps the capi-reachable production crates for thread-spawn tokens
# (`thread::spawn` / `thread::Builder`). A hit is a violation unless it
# appears AFTER the first `#[cfg(test)]` line of its file — the repo-wide
# convention (rustfmt-enforced layout keeps the tests module last in the
# file). Adding production code below a tests module would defeat the gate;
# that layout is itself forbidden by the convention this gate encodes.
#
# NOTE (honest negative, ADR M4-02 §4): an object-level gate (llvm-nm scan of
# libvokra.a for `pthread_create` refs in vokra members) was prototyped first
# and DISCARDED — on wasm32-unknown-emscripten, std's spawn path is an
# "unsupported operation" stub, so a spawning crate leaves NO pthread_create
# trace in its own object (verified empirically with a poisoned fixture).
# Source level is the honest layer for this invariant.
#
# Usage:
#   scripts/check-capi-thread-free.sh                # scan the production crates
#   scripts/check-capi-thread-free.sh --self-test    # fixture-based negative/positive
#
# Exit code: 0 = invariant holds (or self-test green); 1 = violation
# (explicit error, FR-EX-08 — never a silent pass).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Crates whose code is reachable from the C ABI surface (vokra-capi and its
# workspace dependency closure — see crates/vokra-capi/Cargo.toml).
CRATES=(
    vokra-capi
    vokra-core
    vokra-models
    vokra-ops
    vokra-backend-cpu
    vokra-piper-plus
)

# Thread-spawn tokens. `thread::spawn` covers `std::thread::spawn` and
# `use std::thread; thread::spawn`; `thread::Builder` covers the named-thread
# builder path (`Builder::new().spawn(...)`).
PATTERN='thread::(spawn|Builder)'

# wasm_excluded_modules <dir> — prints "dir/src-relative" file/dir prefixes of
# modules whose declaration carries `not(target_family = "wasm")` (i.e. code
# that provably never compiles into the WebGL staticlib, like the `parallel`
# CPU worker pool — M1-12 / ADR M4-01 §9). Attribute + `mod x;` may be
# separated by further attribute/comment lines.
wasm_excluded_modules() {
    local dir="$1"
    grep -rE --include='*.rs' -A 5 'not\(target_family = "wasm"\)' "$dir" 2>/dev/null \
        | sed -nE 's/^.*[:-][[:space:]]*(pub[[:space:]]+)?mod[[:space:]]+([a-z_0-9]+);.*$/\2/p' \
        | sort -u
}

# scan_file <file> — prints violations ("file:line: text"); returns 0 if the
# file is clean, 1 otherwise. A hit before the file's first `#[cfg(test)]`
# line (or in a file with none) is a violation.
scan_file() {
    local file="$1"
    awk -v file="$file" '
        /^[[:space:]]*#\[cfg\(test\)\]/ { in_test = 1 }
        !in_test && $0 ~ /thread::(spawn|Builder)/ {
            printf "%s:%d: %s\n", file, NR, $0
            bad = 1
        }
        END { exit bad ? 1 : 0 }
    ' "$file"
}

# scan_tree <dir...> — scans every .rs file under the given directories,
# skipping modules that are cfg-gated off wasm (they never reach the WebGL
# staticlib, which is what this invariant protects).
scan_tree() {
    local fail=0 f dir m skip
    local -a excluded=()
    for dir in "$@"; do
        while IFS= read -r m; do
            [ -n "$m" ] || continue
            excluded+=("$dir/$m.rs" "$dir/$m/")
        done < <(wasm_excluded_modules "$dir")
    done
    while IFS= read -r f; do
        skip=0
        for m in ${excluded[@]+"${excluded[@]}"}; do
            case "$f" in
                "$m" | "$m"*) skip=1 ;;
            esac
        done
        if [ "$skip" -eq 1 ]; then
            echo "  (skipping $f — module cfg-gated off wasm, never in the WebGL staticlib)"
            continue
        fi
        if ! scan_file "$f"; then
            fail=1
        fi
    done < <(grep -rlE "$PATTERN" --include='*.rs' "$@" 2>/dev/null || true)
    return "$fail"
}

# ---- self-test ---------------------------------------------------------------
self_test() {
    local tmp
    tmp="$(mktemp -d "${TMPDIR:-/tmp}/vokra-threadfree-XXXXXX")"
    # Expand $tmp NOW: the trap fires after this function's locals are gone.
    # shellcheck disable=SC2064
    trap "rm -rf '$tmp'" EXIT
    mkdir -p "$tmp/src"

    # Negative fixture: spawn in production position (before any cfg(test)).
    cat > "$tmp/src/poison.rs" <<'EOF'
pub fn poison() {
    let h = std::thread::spawn(|| 7);
    let _ = h.join();
}

#[cfg(test)]
mod tests {}
EOF
    if scan_tree "$tmp/src" >/dev/null 2>&1; then
        echo "self-test FAILED: production spawn passed the scan" >&2
        exit 1
    fi
    echo "self-test: production spawn rejected (expected)"
    rm "$tmp/src/poison.rs"

    # Negative fixture 2: Builder path in production position.
    cat > "$tmp/src/builder.rs" <<'EOF'
pub fn builder() {
    let _ = std::thread::Builder::new();
}
EOF
    if scan_tree "$tmp/src" >/dev/null 2>&1; then
        echo "self-test FAILED: production thread::Builder passed the scan" >&2
        exit 1
    fi
    echo "self-test: production thread::Builder rejected (expected)"
    rm "$tmp/src/builder.rs"

    # Positive fixture: spawn only inside the trailing tests module.
    cat > "$tmp/src/clean.rs" <<'EOF'
pub fn clean(x: u32) -> u32 {
    x.wrapping_mul(2654435761)
}

#[cfg(test)]
mod tests {
    #[test]
    fn spawn_in_tests_is_fine() {
        let h = std::thread::spawn(|| 7);
        assert_eq!(h.join().unwrap(), 7);
    }
}
EOF
    if ! scan_tree "$tmp/src" >/dev/null; then
        echo "self-test FAILED: test-module spawn was rejected" >&2
        exit 1
    fi
    echo "self-test: test-module spawn accepted (expected)"
    rm "$tmp/src/clean.rs"

    # Positive fixture 2: a module cfg-gated off wasm may spawn (it never
    # reaches the WebGL staticlib — e.g. the `parallel` CPU pool, M1-12).
    cat > "$tmp/src/lib.rs" <<'EOF'
#[cfg(all(feature = "parallel", not(target_family = "wasm")))]
mod pool;
EOF
    cat > "$tmp/src/pool.rs" <<'EOF'
pub fn spawn_worker() {
    let _ = std::thread::Builder::new();
}
EOF
    if ! scan_tree "$tmp/src" >/dev/null; then
        echo "self-test FAILED: wasm-excluded module spawn was rejected" >&2
        exit 1
    fi
    echo "self-test: wasm-excluded module spawn accepted (expected)"
    echo "check-capi-thread-free: self-test OK"
}

# ---- entry -------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ]; then
    self_test
    exit 0
fi

DIRS=()
for crate in "${CRATES[@]}"; do
    DIRS+=("$ROOT/crates/$crate/src")
done

if scan_tree "${DIRS[@]}"; then
    echo "check-capi-thread-free: OK (no production thread spawn in: ${CRATES[*]})"
else
    echo "check-capi-thread-free: FAILED — thread spawn outside #[cfg(test)] in a" >&2
    echo "  C-ABI-reachable crate. The Unity WebGL baseline is single-threaded" >&2
    echo "  (ADR M4-02 §4); use the poll-based streaming design (M1-08) instead." >&2
    exit 1
fi
