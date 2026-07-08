#!/usr/bin/env bash
# verify-ios-xcframework.sh — post-build verifier for Vokra.xcframework (M2-02).
#
# Enforces NFR-RL-03 (static-only iOS distribution) + NFR-RL-05 (JIT-free) +
# FR-EX-08 (no silent fallback) + the symbol-hygiene rules established in
# scripts/run-capi-smoke.sh (only ^_?vokra_* exported).
#
# Covers plan tickets:
#   T06  Info.plist LibraryIdentifier + Mach-O arch checks
#   T11  static-only assertions (no .framework, no .dylib, no _dlopen refs)
#   T12  symbol whitelist against Mach-O `_vokra_` form
#   T13  clang -fmodules sanity compile of module.modulemap + vokra.h
#
# Usage:
#   scripts/verify-ios-xcframework.sh <path/to/Vokra.xcframework>
#
# Exit 0 = all checks green; non-zero on the first failure (fail-loud).

set -euo pipefail

XCF="${1:-build/ios/Vokra.xcframework}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# R5 mitigation: refuse to run if RUSTFLAGS carries a `panic` override — that
# would silently defeat ffi_guard's catch_unwind (root Cargo.toml:99-107,
# panic = "unwind" is mandatory for iOS). Checked first so a stale artifact
# does not mask a poisoned build env.
if [ "${RUSTFLAGS:-}" != "${RUSTFLAGS/panic/}" ]; then
    echo "verify-ios-xcframework: FAIL RUSTFLAGS contains 'panic' override: $RUSTFLAGS" >&2
    echo "  iOS build must keep panic = \"unwind\" (ffi_guard requirement)." >&2
    exit 1
fi

if [ ! -d "$XCF" ]; then
    echo "verify-ios-xcframework: FAIL missing $XCF" >&2
    exit 1
fi

echo "== verify-ios-xcframework $XCF =="

# --- (a) Info.plist LibraryIdentifier check (T06) ------------------------------
PLIST="$XCF/Info.plist"
if [ ! -f "$PLIST" ]; then
    echo "verify-ios-xcframework: FAIL missing $PLIST" >&2
    exit 1
fi
PLIST_XML="$(plutil -convert xml1 -o - "$PLIST")"
for id in ios-arm64 ios-arm64_x86_64-simulator; do
    # See (b) below for why we use herestrings here instead of pipes — with
    # `set -o pipefail`, a large enough $PLIST_XML causes `printf` to see
    # SIGPIPE after `grep -q` closes stdin, which trips pipefail even though
    # the pattern was found. The herestring keeps a single-command form.
    if ! grep -qF "<string>$id</string>" <<<"$PLIST_XML"; then
        echo "verify-ios-xcframework: FAIL Info.plist missing LibraryIdentifier '$id'" >&2
        exit 1
    fi
    echo "  Info.plist has LibraryIdentifier: $id"
done

# --- (d) static-only tree assertions (T11) -------------------------------------
if find "$XCF" -type d -name '*.framework' | grep -q .; then
    echo "verify-ios-xcframework: FAIL .framework directory in tree (dynamic; violates NFR-RL-03)" >&2
    find "$XCF" -type d -name '*.framework' >&2
    exit 1
fi
if find "$XCF" -type f -name '*.dylib' | grep -q .; then
    echo "verify-ios-xcframework: FAIL .dylib in tree (dynamic; violates NFR-RL-03)" >&2
    find "$XCF" -type f -name '*.dylib' >&2
    exit 1
fi
echo "  static-only tree: no .framework, no .dylib"

# --- per-slice checks: arch, static archive, symbols, dlopen refs, size --------
verify_slice() {
    local id="$1" arch="$2"
    local lib="$XCF/$id/libvokra.a"
    if [ ! -f "$lib" ]; then
        echo "verify-ios-xcframework: FAIL missing $lib" >&2
        exit 1
    fi

    # (d.i) static archive per file(1) (T11)
    if ! file "$lib" | grep -q 'current ar archive'; then
        echo "verify-ios-xcframework: FAIL $lib is not 'current ar archive':" >&2
        file "$lib" >&2
        exit 1
    fi

    # (b) otool -hv per slice: MH_MAGIC_64 + expected arch (T06)
    local hdrs
    hdrs="$(otool -hv -arch "$arch" "$lib" 2>/dev/null || true)"
    # Use `grep -c` (count) + shell arithmetic rather than piping into
    # `grep -q`: with `set -o pipefail`, the archive of an XCFramework
    # produces enough header output that `grep -q` closes stdin before
    # `printf` finishes writing (`printf: write error: Broken pipe`), and
    # the SIGPIPE-triggered `printf` exit trips the pipefail gate even
    # though the pattern was found. Using a herestring keeps a single
    # command and no pipe, so pipefail never fires here.
    if ! grep -q 'MH_MAGIC_64' <<<"$hdrs"; then
        echo "verify-ios-xcframework: FAIL $lib ($arch) missing MH_MAGIC_64" >&2
        printf '%s\n' "$hdrs" >&2
        exit 1
    fi
    # otool -hv prints the CPU type in the mach header as `ARM64` / `X86_64`
    # (uppercase). Match case-insensitively against the uppercased form —
    # `\bARM64\b` matches every `MH_MAGIC_64    ARM64` header line the arch
    # selection produced. The prior form (`\barm64\b`) only ever passed by
    # coincidence on the device slice because the archive PATH
    # (`.../ios-arm64/libvokra.a`) contained the lowercase word `arm64`;
    # the simulator slice's identifier `ios-arm64_x86_64-simulator` has
    # `arm64_x86_64` (underscore is a word char, no `\b` boundary between
    # `arm64` and `_x86_64`), so the same regex silently missed and reported
    # "does not contain arch 'arm64'" even though otool did list `ARM64`
    # headers throughout. Uppercase + case-insensitive is what the actual
    # header field looks like.
    local arch_upper
    arch_upper="$(printf '%s' "$arch" | tr '[:lower:]' '[:upper:]')"
    if ! grep -qE "\\b$arch_upper\\b" <<<"$hdrs"; then
        echo "verify-ios-xcframework: FAIL $lib does not contain arch '$arch'" >&2
        printf '%s\n' "$hdrs" >&2
        exit 1
    fi

    # (c) symbol whitelist — Mach-O `_vokra_` form (T12; extends
    # run-capi-smoke.sh:52-58 ^vokra_ gate). nm -g -arch selects the slice; we
    # keep only defined externs (T/S/D/C/B) and require every "user" one to be
    # _vokra_-prefixed. Undefined refs are checked separately below.
    #
    # Rust's static library form (`libvokra.a`) also exports a handful of
    # Rust-runtime / compiler-builtins symbols that *cannot* be stripped
    # without breaking linkage. These are namespaced and cannot collide with
    # the Vokra C ABI surface, so we allow them explicitly:
    #   * `__rust_*`   — allocator shims (`__rust_alloc`, `__rust_dealloc`,
    #                    `__rust_realloc`, `__rust_no_alloc_shim_is_unstable`,
    #                    `__rust_alloc_error_handler`, etc.)
    #   * `__R[a-z]*`  — Rust v0 mangled symbols (`__RNvCs...`, `__RINvNtCs...`).
    #                    These are internal Rust items marked `#[used]` or
    #                    `#[no_mangle]` in unusual ways; they cannot conflict
    #                    with any C caller because of the `_R` prefix.
    #   * `_atomic_*`  — compiler-builtins atomic fences (`_atomic_thread_fence`).
    #                    Emitted by the toolchain when std spawns anything
    #                    atomic-shaped.
    #   * `___[cdlp]*` / `___divti3` etc. — LLVM compiler-rt intrinsics
    #                    (integer / float helpers). Some Rust math paths
    #                    emit these; the leading `___` triple underscore
    #                    is the Mach-O form of `__` C-symbol double-underscore.
    #   * `__aarch64_*` — LLVM AArch64 atomic outlined helpers
    #                    (`__aarch64_cas{1,2,4,8}_{relax,acq,rel,acq_rel}`,
    #                    `__aarch64_ldadd*`, `__aarch64_swp*`). These are
    #                    generated by the compiler for `AtomicUsize::cmpxchg`
    #                    et al. when targeting `aarch64-apple-ios` and are
    #                    NOT vendored — they live in `compiler-builtins`.
    #                    Cannot conflict with a C caller (leading double
    #                    underscore is a reserved namespace in ISO C).
    #
    # Non-allowlisted defined externs (i.e. anything that doesn't start with
    # `_vokra_` or one of the runtime allowances above) still fail this gate.
    local defined unexpected count
    defined="$(nm -g -arch "$arch" "$lib" 2>/dev/null \
        | awk '/^[0-9a-fA-F]+ [TSDCB] / {print $3}' || true)"
    unexpected="$(printf '%s\n' "$defined" \
        | grep -vE '^_vokra_' \
        | grep -vE '^__rust_' \
        | grep -vE '^__R[a-zA-Z]' \
        | grep -vE '^_atomic_' \
        | grep -vE '^___[a-z]' \
        | grep -vE '^__aarch64_' \
        | grep -vE '^_?$' || true)"
    if [ -n "$unexpected" ]; then
        echo "verify-ios-xcframework: FAIL unexpected defined symbols in $lib ($arch):" >&2
        printf '  %s\n' "$unexpected" >&2
        exit 1
    fi
    count="$(printf '%s\n' "$defined" | grep -cE '^_vokra_' || true)"

    # (d.iii) no _dlopen references (T11) — capi does not dlopen; CUDA gated out
    if nm -u -arch "$arch" "$lib" 2>/dev/null | grep -qE '(^| )_dlopen( |$)'; then
        echo "verify-ios-xcframework: FAIL $lib ($arch) references _dlopen (violates NFR-RL-03 static-only)" >&2
        exit 1
    fi

    # (g) informational size print (per check-binary-size.sh conventions)
    local bytes
    bytes="$(stat -f %z "$lib" 2>/dev/null || stat -c %s "$lib")"
    echo "  $id/libvokra.a: $arch, $count _vokra_* symbols, $(printf '%d' "$bytes") bytes"
}

verify_slice ios-arm64 arm64
# Simulator slice is fat (arm64 + x86_64) — verify both arches.
verify_slice ios-arm64_x86_64-simulator arm64
verify_slice ios-arm64_x86_64-simulator x86_64

# --- (e) clang -fmodules sanity compile of module.modulemap + vokra.h (T13) ----
# Catches VOKRA_H macro guard issues and any header/modulemap mismatch. We look
# for module.modulemap alongside vokra.h in either slice's Headers/ directory.
HDR_DIR=""
for cand in "$XCF/ios-arm64/Headers" "$XCF/ios-arm64_x86_64-simulator/Headers"; do
    if [ -f "$cand/vokra.h" ] && [ -f "$cand/module.modulemap" ]; then
        HDR_DIR="$cand"
        break
    fi
done
if [ -z "$HDR_DIR" ]; then
    echo "verify-ios-xcframework: FAIL no Headers/{vokra.h,module.modulemap} pair in slices" >&2
    exit 1
fi

TMP="$(mktemp -d -t vokra-verify-ios.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

# Minimal C TU that pulls the module map and touches an API symbol so the
# preprocessor + modules loader actually parses vokra.h through the modulemap.
cat >"$TMP/probe.c" <<'PROBE'
#include "vokra.h"
int probe(void) { return (int)vokra_version_major(); }
PROBE

if ! clang -x c -fsyntax-only \
        -fmodules -fmodules-cache-path="$TMP/mcache" \
        -fmodule-map-file="$HDR_DIR/module.modulemap" \
        -I"$HDR_DIR" \
        "$TMP/probe.c" 2>"$TMP/clang.err"; then
    echo "verify-ios-xcframework: FAIL clang -fmodules sanity compile failed" >&2
    cat "$TMP/clang.err" >&2
    exit 1
fi
echo "  clang -fmodules sanity compile OK ($HDR_DIR/module.modulemap)"

echo "verify-ios-xcframework: all checks passed"
