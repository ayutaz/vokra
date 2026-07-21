#!/usr/bin/env bash
# check-console-static.sh — console-portability static-link gate (M5-04).
#
# WP M5-04 (docs/tickets/m5/M5-04-console-portability-static-link.md). The
# console SDK itself is owner territory (Nintendo / Sony / Microsoft NDA — CC
# cannot substitute). What CC lands before the NDA is the machine gate that
# proves the *static-link base* is sound so that, once a real console triple
# arrives, the owner only has to swap `VOKRA_STATIC_TRIPLE` and re-run this.
#
# This gate asserts three things about the CPU-only static library
# (`libvokra.a`, the artifact a console links directly, no dynamic loader):
#
#   (1) C ABI completeness — every function symbol declared in the committed
#       header `include/vokra.h` is actually DEFINED in the static archive.
#       The symbol set is extracted from the header at runtime (NOT hardcoded)
#       so the gate tracks IF-01 / M5-13 without a second source of truth.
#
#   (2) FFI panic firewall intact — the workspace `[profile.release]` must NOT
#       carry `panic = "abort"`. crates/vokra-capi/src/ffi_guard.rs turns a
#       Rust panic into VOKRA_ERROR_PANIC via std::panic::catch_unwind; under
#       panic="abort" that catch_unwind is a no-op and a panic aborts the whole
#       host process. `panic` is a whole-compilation setting, so the shipped
#       libvokra MUST build with unwinding. Only the non-FFI-crossing
#       [profile.release-min] may opt into abort (root Cargo.toml documents it).
#
#   (3) no-dynamic-load — the CPU-only archive references no dlopen/dlsym/
#       LoadLibrary/GetProcAddress (those live only in the opt-in GPU backends,
#       which are NOT in this build). This is the literal meaning of a console
#       static-link build: nothing is resolved at runtime by a dynamic loader.
#
# --self-test runs a negative-fixture suite: it proves each leg has real
# detection power (a "gate that always passes" is a fabricated pass — red-line).
#
# Tooling contract (zero-dep, NFR-DS-02): bash + cargo + nm only. No python, no
# third-party crate, no committed binary. nm is chosen by CAPABILITY PROBE
# (`$VOKRA_NM` → `nm` → `llvm-nm`), never by binary name — see pick_nm.
#
# ===========================================================================
# WHY THE DEFAULT TARGET IS x86_64-unknown-linux-musl (measured 2026-07-21)
# ===========================================================================
# A console = a no-dynamic-loading environment. The nearest CC-runnable,
# fully-static, stock-rustup proxy (and the one CI already builds in the
# `server-deployment` job) is `x86_64-unknown-linux-musl`.
#
# It is ALSO the only reliable target for symbol enumeration on macOS. Measured
# on this machine (Apple M1, rustc 1.95.0, Xcode /usr/bin/nm = llvm-nm 2100):
#   * HOST build (aarch64-apple-darwin, `[profile.release]` lto="fat"): the
#     vokra-capi codegen object is left as LLVM-22 BITCODE for the host linker's
#     LTO plugin. `nm` on that object fails with
#     `Unknown attribute kind (105) (Producer: LLVM22.1.2 Reader: LLVM 2100)`,
#     so none of the 33 `_vokra_*` symbols are visible → the presence check
#     would false-FAIL through no fault of the archive.
#   * musl CROSS build (same profile): rustc must emit NATIVE ELF objects (no
#     host LTO plugin to defer to), so `nm` enumerates all 33 `vokra_*` defined
#     symbols and 0 dlopen undefined-refs cleanly.
# Real console triples are cross-targets like musl, so they emit native objects
# too. The owner overrides the target via `VOKRA_STATIC_TRIPLE`.

set -uo pipefail # NOT -e: the self-test asserts on nonzero exits of the legs.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HEADER="$ROOT/include/vokra.h"
CARGO_TOML="$ROOT/Cargo.toml"
TRIPLE="${VOKRA_STATIC_TRIPLE:-x86_64-unknown-linux-musl}"

# Dynamic-loader entry points that must never be referenced by a static-link
# console build. Kept as one source of truth (used by the gate and self-test).
DYNLOAD_RE='(^|[^A-Za-z0-9_])_?(dlopen|dlsym|dlvsym|LoadLibraryA|LoadLibraryW|LoadLibrary|GetProcAddress)([^A-Za-z0-9_]|$)'

log()  { printf '%s\n' "$*"; }
fail() { printf 'check-console-static: FAIL %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<EOF
usage: $0 [<path/to/libvokra.a>] | --self-test

  (no args)     build the CPU-only release staticlib for \$VOKRA_STATIC_TRIPLE
                (default: $TRIPLE) and run all gate legs against it.
  <path.a>      skip the build; run the gate legs against an existing archive.
  --self-test   run the negative-fixture suite (no build; hermetic).

Environment:
  VOKRA_STATIC_TRIPLE   target triple to build/verify (owner injects the real
                        console triple here — never committed to a tracked file).
  VOKRA_NM              nm-compatible tool to prefer (e.g. an SDK-supplied nm).
EOF
}

# --------------------------------------------------------------------------
# nm capability probe. Try candidates in order; adopt the first that actually
# enumerates at least one DEFINED symbol line from <archive>. Never keys off
# the binary NAME: on this machine /usr/bin/nm IS llvm-nm and reads ELF fine,
# so failing when a literal `llvm-nm` is absent would wrongly block a healthy
# host. Fail-loud only if NOTHING can read the archive (FR-EX-08).
# Prints the chosen command on stdout; returns 1 if none work.
# --------------------------------------------------------------------------
pick_nm() {
    local archive="$1" cand out hits
    for cand in ${VOKRA_NM:-} ${LLVM_NM:-} nm llvm-nm; do
        command -v "$cand" >/dev/null 2>&1 || continue
        out="$("$cand" "$archive" 2>/dev/null || true)"
        # Count defined-symbol-shaped lines. `grep -c` reads ALL input (no early
        # exit), so there is no SIGPIPE-into-pipefail hazard here. A defined line
        # is `<addr-or-dashes> <TYPE> <name>`; undefined lines start with blanks.
        hits="$(printf '%s\n' "$out" | grep -cE '^[0-9a-fA-F-]+[[:space:]]+[A-Za-z][[:space:]]' || true)"
        if [ "${hits:-0}" -gt 0 ]; then
            printf '%s\n' "$cand"
            return 0
        fi
    done
    return 1
}

# Extract the C ABI FUNCTION symbol set from the committed header at runtime.
# cbindgen emits `<ret> vokra_name(...)`; the name is the token immediately
# before `(`. Return types / opaque struct typedefs (`vokra_session_t`) are
# followed by whitespace or `;`/`)` , never `(`, so they are excluded.
extract_header_symbols() {
    grep -oE 'vokra_[A-Za-z0-9_]+[[:space:]]*\(' "$1" \
        | sed -E 's/[[:space:]]*\(.*$//' \
        | sort -u
}

# Defined EXTERNAL symbols of an archive, one name per line. Uppercase type
# code (T/D/S/B/G/R/W...) = defined external; lowercase = local; `U` = undefined
# (NF<3 after awk drops the blank addr column). Robust to the Mach-O archive
# form where the address column is `----------------` (still NF==3).
nm_defined_symbols() {
    local nm="$1" archive="$2" out
    out="$("$nm" "$archive" 2>/dev/null || true)"
    printf '%s\n' "$out" | awk 'NF>=3 && $(NF-1) ~ /^[A-Z]$/ { print $NF }'
}

# Undefined references of an archive (nm -u), one name-bearing line per entry.
nm_undefined_lines() {
    local nm="$1" archive="$2" out
    out="$("$nm" -u "$archive" 2>/dev/null || true)"
    printf '%s\n' "$out"
}

# --- pure-text analyzers (fed synthetic input by --self-test) --------------

# missing_symbols <expected-file> <defined-file> : print header symbols that
# are NOT present (as `sym` or Mach-O `_sym`) in the defined set. grep reads
# from a FILE (no upstream producer), so `-q` is safe here — the SIGPIPE
# fail-open trap only bites when grep -q sits downstream of a pipe.
missing_symbols() {
    local expected="$1" defined="$2" sym
    while IFS= read -r sym; do
        [ -n "$sym" ] || continue
        grep -qxE "_?${sym}" "$defined" || printf '%s\n' "$sym"
    done < "$expected"
}

# dynamic_load_refs <undefined-file> : print any line naming a dynamic-loader
# entry point (ELF `dlopen` or Mach-O `_dlopen` form). Reads from a FILE.
dynamic_load_refs() {
    grep -E "$DYNLOAD_RE" "$1" 2>/dev/null || true
}

# check_release_panic_unwind <Cargo.toml> : return 1 if [profile.release]
# carries a real `panic = "abort"` ASSIGNMENT. The section header is matched
# exactly (`[profile.release]`, not the `[profile.release-min]` prefix) and
# capture stops at the next `[...]`. Comments are stripped BEFORE the match so
# the root manifest's own prose ("do NOT set panic=\"abort\"") never trips it.
check_release_panic_unwind() {
    local toml="$1" section hit
    section="$(awk '
        /^\[profile\.release\][[:space:]]*$/ { inrel=1; next }
        /^\[/ { inrel=0 }
        inrel { print }
    ' "$toml")"
    hit="$(printf '%s\n' "$section" \
        | sed 's/#.*//' \
        | grep -E "^[[:space:]]*panic[[:space:]]*=[[:space:]]*[\"']?abort" || true)"
    [ -z "$hit" ]
}

# ===========================================================================
# gate
# ===========================================================================
run_gate() {
    local archive="$1"

    # (0) RUSTFLAGS panic override guard (mirrors verify-ios-xcframework.sh).
    # A `panic` override in RUSTFLAGS defeats ffi_guard the same way
    # [profile.release] panic="abort" does, and it is compilation-wide.
    local rf="${RUSTFLAGS:-}"
    if [ "$rf" != "${rf/panic/}" ]; then
        fail "RUSTFLAGS carries a 'panic' override ($rf); the libvokra staticlib must build with panic=\"unwind\" (ffi_guard)."
    fi

    # (1) FFI panic firewall — [profile.release] must not be panic="abort".
    if check_release_panic_unwind "$CARGO_TOML"; then
        log "  [profile.release] keeps panic=unwind (ffi_guard firewall intact)"
    else
        fail "[profile.release] sets panic=\"abort\" — that no-ops ffi_guard's catch_unwind (a panic would abort the host process). Only [profile.release-min] may use abort."
    fi

    # (2) build the CPU-only staticlib if no archive was supplied.
    if [ -z "$archive" ]; then
        log "== build CPU-only staticlib ($TRIPLE) =="
        if ! ( cd "$ROOT" && cargo build -p vokra-capi --release \
                --target "$TRIPLE" --no-default-features --features cpu ); then
            fail "cargo build of the CPU-only staticlib failed for target '$TRIPLE'."
        fi
        archive="$ROOT/target/$TRIPLE/release/libvokra.a"
    fi
    [ -f "$archive" ] || fail "static archive not found: $archive"

    log "== verify static archive =="
    log "  archive: $archive"

    # (3) file(1) — it must be an ar archive.
    if ! file "$archive" | grep -q 'ar archive'; then
        fail "$archive is not an ar archive: $(file "$archive")"
    fi
    log "  file(1): ar archive"

    # (4) pick an nm that can actually read this archive.
    local NM
    if ! NM="$(pick_nm "$archive")"; then
        printf 'check-console-static: FAIL no nm could enumerate symbols from %s\n' "$archive" >&2
        printf '  tried: ${VOKRA_NM} ${LLVM_NM} nm llvm-nm\n' >&2
        printf '  install one, e.g.:  rustup component add llvm-tools-preview  (its bin dir has llvm-nm)\n' >&2
        printf '  or point VOKRA_NM / LLVM_NM at an SDK-supplied nm.\n' >&2
        exit 1
    fi
    log "  nm: $NM"

    # (5) C ABI completeness — every header symbol is DEFINED in the archive.
    local tmp; tmp="$(mktemp -d "${TMPDIR:-/tmp}/vokra-console.XXXXXX")"
    # shellcheck disable=SC2064
    trap "rm -rf '$tmp'" RETURN
    extract_header_symbols "$HEADER" > "$tmp/expected"
    nm_defined_symbols "$NM" "$archive" | sort -u > "$tmp/defined"
    local n_expected; n_expected="$(grep -c . "$tmp/expected" || true)"
    if [ "${n_expected:-0}" -eq 0 ]; then
        fail "extracted 0 C ABI symbols from $HEADER — extraction is broken (would be a vacuous pass)."
    fi
    local missing; missing="$(missing_symbols "$tmp/expected" "$tmp/defined")"
    if [ -n "$missing" ]; then
        printf 'check-console-static: FAIL %d/%d C ABI symbols are NOT defined in the staticlib:\n' \
            "$(printf '%s\n' "$missing" | grep -c .)" "$n_expected" >&2
        printf '  %s\n' "$missing" >&2
        exit 1
    fi
    log "  C ABI: all $n_expected header symbols defined in the archive"

    # (6) no-dynamic-load — no dlopen/dlsym/LoadLibrary/GetProcAddress refs.
    # Captured into a file, NEVER `nm -u | grep -q` (SIGPIPE + pipefail =
    # fail-open on a large undefined list — see verify-ios-xcframework.sh:222).
    nm_undefined_lines "$NM" "$archive" > "$tmp/undef"
    local dynrefs; dynrefs="$(dynamic_load_refs "$tmp/undef")"
    if [ -n "$dynrefs" ]; then
        printf 'check-console-static: FAIL archive references a dynamic loader (violates static-link/no-dynamic-load):\n' >&2
        printf '  %s\n' "$dynrefs" >&2
        exit 1
    fi
    log "  no-dynamic-load: no dlopen/dlsym/LoadLibrary/GetProcAddress refs"

    log "check-console-static: all checks passed ($NM, $TRIPLE)"
}

# ===========================================================================
# --self-test : negative fixtures (prove detection power; no build)
# ===========================================================================
run_self_test() {
    local scratch pass=0 fail=0
    scratch="$(mktemp -d "${TMPDIR:-/tmp}/vokra-console-selftest.XXXXXX")"
    trap 'rm -rf "$scratch"' EXIT
    ok()  { pass=$((pass + 1)); printf '  ok:   %s\n' "$1"; }
    bad() { fail=$((fail + 1)); printf '  FAIL: %s\n' "$1" >&2; }

    log "== check-console-static --self-test =="

    # --- panic firewall detector (the primary --self-test requirement) ---
    log "[panic] check_release_panic_unwind"

    # n1 — panic="abort" injected into [profile.release] must be DETECTED (red).
    cat >"$scratch/toml-abort" <<'EOF'
[profile.release]
opt-level = 3
panic = "abort"
EOF
    if check_release_panic_unwind "$scratch/toml-abort"; then
        bad "n1 panic=\"abort\" in [profile.release] was NOT detected (gate would fail open)"
    else
        ok "n1 panic=\"abort\" in [profile.release] detected (red)"
    fi

    # n2 — panic="abort" only in [profile.release-min] must PASS (green): that
    # profile does not cross the C ABI, so abort is allowed there.
    cat >"$scratch/toml-min" <<'EOF'
[profile.release]
opt-level = 3
lto = "fat"

[profile.release-min]
inherits = "release"
panic = "abort"
EOF
    if check_release_panic_unwind "$scratch/toml-min"; then
        ok "n2 panic=\"abort\" confined to [profile.release-min] passes (section discrimination works)"
    else
        bad "n2 gate wrongly tripped on [profile.release-min] abort"
    fi

    # n3 — a COMMENT mentioning panic="abort" in [profile.release] must PASS.
    # This is the exact shape of the real root Cargo.toml, whose prose says
    # 'do NOT set panic="abort"'. Comment-stripping must not false-positive.
    cat >"$scratch/toml-comment" <<'EOF'
[profile.release]
# panic is LEFT AT THE DEFAULT ("unwind") — do NOT set panic="abort" here.
opt-level = 3
EOF
    if check_release_panic_unwind "$scratch/toml-comment"; then
        ok "n3 commented panic=\"abort\" passes (comment-stripping works)"
    else
        bad "n3 gate false-positived on a commented panic=\"abort\""
    fi

    # n4 — the real shipped manifest passes (positive control on the actual
    # profile this WP must not regress).
    if check_release_panic_unwind "$CARGO_TOML"; then
        ok "n4 the real root Cargo.toml [profile.release] passes (panic=unwind)"
    else
        bad "n4 the real root Cargo.toml unexpectedly reads as panic=abort"
    fi

    # --- C ABI presence detector ---
    log "[symbols] missing_symbols"

    # n5 — a header symbol absent from the archive must be reported missing.
    printf 'vokra_version\nvokra_absent_probe\n' > "$scratch/expected"
    printf '_vokra_version\nvokra_other\n'        > "$scratch/defined"
    local miss; miss="$(missing_symbols "$scratch/expected" "$scratch/defined")"
    if printf '%s\n' "$miss" | grep -qx 'vokra_absent_probe' \
        && ! printf '%s\n' "$miss" | grep -qx 'vokra_version'; then
        ok "n5 missing symbol reported, present symbol (Mach-O _-form) accepted"
    else
        bad "n5 missing_symbols mis-detected (got: $(printf '%s' "$miss" | tr '\n' ' '))"
    fi

    # n6 — all present ⇒ empty missing set (no false alarm).
    printf 'vokra_version\n'  > "$scratch/expected2"
    printf 'vokra_version\n'  > "$scratch/defined2"
    if [ -z "$(missing_symbols "$scratch/expected2" "$scratch/defined2")" ]; then
        ok "n6 fully-present set yields no missing symbols"
    else
        bad "n6 missing_symbols false-alarmed on a complete set"
    fi

    # n7 — header extraction on the real header yields the 33-symbol contract
    # (a vacuous 0-symbol extraction would make the gate a fabricated pass).
    local n_hdr; n_hdr="$(extract_header_symbols "$HEADER" | grep -c . || true)"
    if [ "${n_hdr:-0}" -ge 1 ]; then
        ok "n7 extract_header_symbols yields $n_hdr symbols from the real header"
    else
        bad "n7 extract_header_symbols yielded 0 symbols (extraction broken)"
    fi

    # --- no-dynamic-load detector ---
    log "[dynload] dynamic_load_refs"

    # n8 — an undefined dlopen ref (ELF and Mach-O forms) must be DETECTED.
    printf '                 U dlopen\n                 U malloc\n' > "$scratch/undef-elf"
    printf '                 U _dlopen\n'                            > "$scratch/undef-macho"
    if [ -n "$(dynamic_load_refs "$scratch/undef-elf")" ] \
        && [ -n "$(dynamic_load_refs "$scratch/undef-macho")" ]; then
        ok "n8 dlopen undefined-ref detected (ELF and Mach-O forms)"
    else
        bad "n8 dynamic_load_refs missed a dlopen reference (gate would fail open)"
    fi

    # n9 — a clean undefined list (no loader entry points) yields no refs, and
    # a benign lookalike (my_dlopen_wrapper) does NOT false-match.
    printf '                 U malloc\n                 U my_dlopen_wrapper\n' > "$scratch/undef-clean"
    if [ -z "$(dynamic_load_refs "$scratch/undef-clean")" ]; then
        ok "n9 clean undefined list passes; word-boundary negative (my_dlopen_wrapper) not matched"
    else
        bad "n9 dynamic_load_refs false-positived on a clean list"
    fi

    # --- nm capability probe fail-loud path ---
    log "[nm] pick_nm"

    # n10 — pick_nm must REJECT a non-object input (no nm can enumerate
    # symbols), so the gate's fail-loud branch is reachable, not dead code.
    printf 'this is not an object file\n' > "$scratch/notanobject"
    if pick_nm "$scratch/notanobject" >/dev/null 2>&1; then
        bad "n10 pick_nm accepted a non-object file (fail-loud branch would be dead)"
    else
        ok "n10 pick_nm rejects a non-object input (fail-loud path reachable)"
    fi

    log ""
    log "self-test: $pass passed, $fail failed"
    [ "$fail" -eq 0 ] || exit 1
    exit 0
}

# ===========================================================================
case "${1:-}" in
    -h | --help) usage; exit 0 ;;
    --self-test) run_self_test ;;
    "")          run_gate "" ;;
    -*)          usage >&2; exit 2 ;;
    *)           run_gate "$1" ;;
esac
