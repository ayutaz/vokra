#!/usr/bin/env bash
# check-no-dynamic-load.sh — source-level no-dynamic-load scanner (M5-04).
#
# Complementary to `scripts/check-console-static.sh`, which enforces the
# no-dynamic-load rule at the ARCHIVE level (nm -u | grep -vE dlopen/dlsym/…)
# on the CPU-only staticlib for a console-portable triple. This scanner
# enforces the same rule at the SOURCE level, so the CPU-only crate set stays
# clean before an archive is even built. Together the two gates catch
# regressions on both sides: an accidental `use libloading::…` slipped into a
# CPU crate (source-level) and a stray `_dlopen` undefined-ref that survives
# link (archive-level).
#
# Scope (explicit list — NOT a negation — so a new crate under `crates/` must
# be added here consciously; silent scope drift would let a stray dynamic
# loader sneak in the next time somebody `cargo new`'s a support crate):
#   crates/vokra-core/src
#   crates/vokra-ops/src
#   crates/vokra-backend-cpu/src
#   crates/vokra-models/src
#   crates/vokra-piper-plus/src
#   crates/vokra-capi/src
#   crates/vokra-mmap/src
#   crates/vokra-cli/src
#   crates/vokra-convert/src
#   crates/vokra-eval/src
#   crates/vokra-bert/src
#   crates/vokra-kws-micro/src
#   crates/vokra-vad-micro/src
#
# Explicit EXCLUDE (intentional dlopen; Metal is framework-linked, not dlopen,
# but is listed for symmetry / future-proofing so this list can be dropped in
# wholesale as the `--include-backends` adversarial argument):
#   crates/vokra-backend-cuda/src    — CUDA Driver API + NVRTC (M2-03)
#   crates/vokra-backend-vulkan/src  — Vulkan libvulkan  (M3-02)
#   crates/vokra-backend-webgpu/src  — WebGPU/wgpu shim  (M4-01)
#   crates/vokra-backend-metal/src   — Metal framework   (kept for symmetry)
#   crates/vokra-backend-coreml/src  — CoreML delegate   (v2.0)
#   crates/vokra-backend-qnn/src     — Qualcomm QNN      (v2.0)
#
# Match (mirrors DYNLOAD_RE in scripts/check-console-static.sh:68, minus the
# Mach-O `_?` undefined-symbol prefix that only exists in archive form; source
# code uses the bare identifier).  `libloading` is added as a Rust-crate-usage
# indicator so a `use libloading::Library;` line is caught even when the
# `dlopen` symbol itself is not present in that file.
#
#   (^|[^A-Za-z0-9_])(dlopen|dlsym|dlvsym|LoadLibraryA|LoadLibraryW|LoadLibrary|GetProcAddress|libloading)([^A-Za-z0-9_]|$)
#
# The `[^A-Za-z0-9_]` anchors are a word-boundary equivalent that rejects
# `my_dlopen_wrapper` (a benign helper name that happens to embed the token)
# without a false positive.
#
# Line-level exclusions (mirrors scripts/compliance/check-encodec-exclusion.sh
# lines 141-157):
#   * The line is a Rust doc comment (starts with `///` or `//!`), OR
#   * A `//` inline comment appears BEFORE the token on the line, OR
#   * The line is after the first `#[cfg(test)]` marker in the file.
#
# Exit codes:
#   0 = clean, 1 = a hit was found or scope broken, 2 = usage/CLI error.
#
# Tooling contract (zero-dep, NFR-DS-02): bash + grep + awk + sed +
# coreutils only. No python, no cargo, no third-party crate. The producer
# pipeline never feeds `grep -q` (see scripts/compliance/lint-pipefail-grep-q.py
# for why: under `set -o pipefail`, a producer piped to `grep -q` fails OPEN
# once the producer's output exceeds the pipe buffer).
#
# `--self-test` runs 4 negative fixtures + 1 adversarial (real
# vokra-backend-cuda source tree) to prove detection power (mirrors
# M5-04-T06 red-line — a scanner that always passes is a fabricated pass).

set -uo pipefail  # NOT -e: self-test asserts on nonzero exits of the gate.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# CPU-only scope. Adding a new CPU crate is a conscious edit here — do not
# reformulate as a negation ("everything under crates/ except backend-*").
CPU_SCOPE=(
    "$ROOT/crates/vokra-core/src"
    "$ROOT/crates/vokra-ops/src"
    "$ROOT/crates/vokra-backend-cpu/src"
    "$ROOT/crates/vokra-models/src"
    "$ROOT/crates/vokra-piper-plus/src"
    "$ROOT/crates/vokra-capi/src"
    "$ROOT/crates/vokra-mmap/src"
    "$ROOT/crates/vokra-cli/src"
    "$ROOT/crates/vokra-convert/src"
    "$ROOT/crates/vokra-eval/src"
    "$ROOT/crates/vokra-bert/src"
    "$ROOT/crates/vokra-kws-micro/src"
    "$ROOT/crates/vokra-vad-micro/src"
)

# Backend crates that intentionally dlopen — expected OUT of default scope.
BACKEND_SCOPE=(
    "$ROOT/crates/vokra-backend-cuda/src"
    "$ROOT/crates/vokra-backend-vulkan/src"
    "$ROOT/crates/vokra-backend-webgpu/src"
    "$ROOT/crates/vokra-backend-metal/src"
    "$ROOT/crates/vokra-backend-coreml/src"
    "$ROOT/crates/vokra-backend-qnn/src"
)

# One source of truth for the token regex, shared between gate and self-test.
DYNLOAD_RE='(^|[^A-Za-z0-9_])(dlopen|dlsym|dlvsym|LoadLibraryA|LoadLibraryW|LoadLibrary|GetProcAddress|libloading)([^A-Za-z0-9_]|$)'

log()  { printf '%s\n' "$*"; }

usage() {
    cat <<EOF
usage: $0 [--include-backends | --self-test | --help]

  (no args)             scan the CPU-only crate set (13 paths); exit 1 on any
                        hit that survives the comment and #[cfg(test)] filters.
  --include-backends    additionally include the 6 backend crates in scope
                        (adversarial one-shot — MUST exit 1 since those crates
                        legitimately dlopen; proves the pattern has teeth on
                        real code).
  --self-test           run 4 synthetic negative fixtures + 1 adversarial (no
                        build; hermetic); proves detection power per
                        M5-04-T06.

Environment: none.
Exit codes: 0 = clean, 1 = hit or scope broken, 2 = usage/CLI error.
EOF
}

# --- pure-text analyzer (also fed synthetic input by --self-test) -----------
#
# scan_paths <paths...>
# Return 0 if no violation survives the filters. Return 1 if any hit
# remains or if the scope matched 0 .rs files (vacuous-pass avoidance,
# mirroring check-console-static.sh:238-240 `n_expected == 0` guard).
scan_paths() {
    local paths=("$@")
    local existing=()
    local p
    for p in "${paths[@]}"; do
        if [ -e "$p" ]; then
            existing+=("$p")
        fi
    done
    if [ "${#existing[@]}" -eq 0 ]; then
        printf 'check-no-dynamic-load: FAIL no scope path exists\n' >&2
        for p in "${paths[@]}"; do
            printf '  missing: %s\n' "$p" >&2
        done
        return 1
    fi

    # Vacuous-pass guard: count .rs files across existing scope paths. `find`
    # feeding `wc -l` is not the SIGPIPE-fail-open shape (`producer | grep -q`);
    # `wc -l` reads to EOF.
    local file_count
    file_count="$(find "${existing[@]}" -type f -name '*.rs' 2>/dev/null | wc -l | tr -d ' ')"
    if [ "${file_count:-0}" -eq 0 ]; then
        printf 'check-no-dynamic-load: FAIL scope matched 0 .rs files (vacuous-pass avoidance)\n' >&2
        for p in "${existing[@]}"; do
            printf '  scope: %s\n' "$p" >&2
        done
        return 1
    fi

    # Grep once, capture into a variable, then filter per hit. Never
    # `producer | grep -q` — see scripts/compliance/lint-pipefail-grep-q.py.
    local grep_hits
    grep_hits="$(grep -rnHE --include='*.rs' "$DYNLOAD_RE" "${existing[@]}" 2>/dev/null || true)"

    if [ -z "$grep_hits" ]; then
        printf 'check-no-dynamic-load: OK (%d .rs files scanned, 0 dynload tokens)\n' "$file_count"
        return 0
    fi

    local offending=""
    local hit file rest linenum content stripped cfg_test_line
    while IFS= read -r hit; do
        [ -z "$hit" ] && continue
        # grep -nH output: file:linenum:content
        file="${hit%%:*}"
        rest="${hit#*:}"
        linenum="${rest%%:*}"
        content="${rest#*:}"

        # Strip everything from `//` onward. This collapses both `///` and
        # `//!` doc comments (whole line disappears) AND inline `code(); //
        # comment` (the trailing comment disappears). If the token is no
        # longer present in the stripped code, the hit was in a comment.
        #
        # This is a heuristic: it would false-negative if a token appeared in
        # a URL string literal after `//` (e.g. "https://…/dlopen.so"). Such
        # a URL, if it ever appeared, would be data, not a live dlopen call —
        # so the false negative is defensible for a source-level gate.
        stripped="$(printf '%s\n' "$content" | sed 's|//.*||')"
        if ! printf '%s\n' "$stripped" | grep -qE "$DYNLOAD_RE"; then
            continue
        fi

        # Skip lines after the first `#[cfg(test)]` marker in the file (Rust
        # convention: tests live at end of file in `#[cfg(test)] mod tests`).
        cfg_test_line="$(awk '/^[[:space:]]*#\[cfg\(test\)\]/ { print NR; exit }' "$file" 2>/dev/null || true)"
        if [ -n "$cfg_test_line" ] && [ "$linenum" -gt "$cfg_test_line" ]; then
            continue
        fi

        offending+="$hit"$'\n'
    done <<< "$grep_hits"

    if [ -n "$offending" ]; then
        printf 'check-no-dynamic-load: FAIL dynamic-loader token(s) found outside comments and #[cfg(test)]:\n' >&2
        printf '%s' "$offending" >&2
        printf '\n' >&2
        printf 'A CPU-only crate must not reference dlopen / dlsym / LoadLibrary / GetProcAddress / libloading.\n' >&2
        printf 'Those live only in the opt-in backend crates (crates/vokra-backend-{cuda,vulkan,webgpu,metal,coreml,qnn}).\n' >&2
        return 1
    fi

    printf 'check-no-dynamic-load: OK (%d .rs files scanned, dynload hits were all doc-comment or #[cfg(test)])\n' "$file_count"
    return 0
}

# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------
run_gate() {
    local include_backends="${1:-0}"
    local scope=("${CPU_SCOPE[@]}")
    if [ "$include_backends" = "1" ]; then
        scope+=("${BACKEND_SCOPE[@]}")
        log "== check-no-dynamic-load: CPU + backend crates (adversarial one-shot) =="
    else
        log "== check-no-dynamic-load: CPU-only crate set =="
    fi
    if scan_paths "${scope[@]}"; then
        exit 0
    else
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# --self-test : 4 negative fixtures + 1 adversarial (no build; hermetic)
# ---------------------------------------------------------------------------
run_self_test() {
    local scratch pass=0 fails=0
    scratch="$(mktemp -d "${TMPDIR:-/tmp}/vokra-no-dynload-selftest.XXXXXX")"
    # shellcheck disable=SC2064
    trap "rm -rf '$scratch'" EXIT

    ok()  { pass=$((pass + 1));  printf '  ok:   %s\n' "$1"; }
    bad() { fails=$((fails + 1)); printf '  FAIL: %s\n' "$1" >&2; }

    log "== check-no-dynamic-load --self-test =="

    # n1 — real `unsafe extern "C" { fn dlopen(...); }` MUST be flagged.
    mkdir -p "$scratch/n1"
    cat >"$scratch/n1/lib.rs" <<'EOF'
use std::os::raw::{c_char, c_int, c_void};

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
}

pub unsafe fn load_lib(name: *const c_char) -> *mut c_void {
    unsafe { dlopen(name, 2) }
}
EOF
    if scan_paths "$scratch/n1" >/dev/null 2>&1; then
        bad "n1 real dlopen extern was NOT flagged (fabricated pass!)"
    else
        ok "n1 real dlopen extern flagged (positive control)"
    fi

    # n2 — a `/// uses dlopen …` doc comment MUST NOT be flagged.
    mkdir -p "$scratch/n2"
    cat >"$scratch/n2/lib.rs" <<'EOF'
/// This backend uses dlopen at runtime, see docs/adr/M2-03-cuda-raw-ffi.md.
///
/// The actual `dlopen` primitive is called from the CUDA backend crate, not
/// from this CPU-only crate.
pub fn stub() {}
EOF
    if scan_paths "$scratch/n2" >/dev/null 2>&1; then
        ok "n2 /// doc comment mentioning dlopen not flagged (doc-comment exclusion works)"
    else
        bad "n2 /// doc comment mentioning dlopen was flagged (false positive)"
    fi

    # n3 — dlopen inside a `#[cfg(test)]` module MUST NOT be flagged. The
    # marker is placed BEFORE the token so the "lines after cfg(test)" gate
    # kicks in.
    mkdir -p "$scratch/n3"
    cat >"$scratch/n3/lib.rs" <<'EOF'
pub fn foo() {}

#[cfg(test)]
mod tests {
    fn spawn_helper() {
        // Test-only helper: pretend to dlopen a shim library.
        unsafe { dlopen(std::ptr::null(), 2); }
    }
}
EOF
    if scan_paths "$scratch/n3" >/dev/null 2>&1; then
        ok "n3 dlopen inside #[cfg(test)] not flagged (test-block exclusion works)"
    else
        bad "n3 dlopen inside #[cfg(test)] was flagged (false positive)"
    fi

    # n4 — `my_dlopen_wrapper()` benign function name MUST NOT be flagged.
    # The word-boundary regex must reject a substring embedding. The fixture
    # deliberately contains ONLY the benign name — no bare `dlopen` token
    # anywhere else — so the test isolates the word-boundary property.
    mkdir -p "$scratch/n4"
    cat >"$scratch/n4/lib.rs" <<'EOF'
pub fn my_dlopen_wrapper() {
    println!("benign helper — name embeds the token but does not load anything");
}

pub fn helper_call() {
    my_dlopen_wrapper();
}
EOF
    if scan_paths "$scratch/n4" >/dev/null 2>&1; then
        ok "n4 my_dlopen_wrapper() not flagged (word-boundary regex works)"
    else
        bad "n4 my_dlopen_wrapper() was flagged (word-boundary regex broken)"
    fi

    # n5 — adversarial: the real vokra-backend-cuda/src tree MUST be flagged.
    # Proves the pattern has teeth on production code (mirrors T02 spec
    # §"必須の実装順" (1) — a scanner that always passes is a fabricated
    # pass and would be worse than no gate at all).
    if [ -d "$ROOT/crates/vokra-backend-cuda/src" ]; then
        if scan_paths "$ROOT/crates/vokra-backend-cuda/src" >/dev/null 2>&1; then
            bad "n5 adversarial: real vokra-backend-cuda/src was NOT flagged — pattern has no teeth on production code (fabricated pass!)"
        else
            ok "n5 adversarial: real vokra-backend-cuda/src flagged on genuine dlopen use (pattern has teeth)"
        fi
    else
        bad "n5 adversarial: crates/vokra-backend-cuda/src missing — cannot verify pattern has teeth"
    fi

    log ""
    log "self-test: $pass passed, $fails failed"
    if [ "$fails" -ne 0 ]; then
        exit 1
    fi
    exit 0
}

# ---------------------------------------------------------------------------
case "${1:-}" in
    -h | --help)          usage; exit 0 ;;
    --self-test)          run_self_test ;;
    --include-backends)   run_gate 1 ;;
    "")                   run_gate 0 ;;
    -*)                   usage >&2; exit 2 ;;
    *)                    usage >&2; exit 2 ;;
esac
