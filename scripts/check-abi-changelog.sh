#!/usr/bin/env bash
# check-abi-changelog.sh — M3-16 v0.9 ABI changelog scaffold.
#
# WHAT IT GATES
#   For the v1.0-rc (M4) prerelease window the C ABI is a moving target
#   (the IF-01 semver freeze fires at M5-13 / v1.0 GA, not here — see
#   docs/abi-changelog.md and the STABILITY block at the top of
#   include/vokra.h). During this window we still want every symbol delta
#   to be *observable* on-disk: this script diffs the working-tree
#   include/vokra.h against a committed anchor snapshot and, if it finds a
#   delta, requires every changed symbol to have a row in
#   docs/abi-changelog.md. The entry's date is historical metadata (the PR
#   opening date), so a later CI run must not make an already documented delta
#   fail merely because the calendar moved on.
#
# ARTEFACTS
#   include/vokra.h                                 -- current C header (cbindgen)
#   docs/abi/vokra.h.v1.0-rc-baseline.symbols       -- anchor snapshot
#   docs/abi-changelog.md                           -- narrative + entries
#
# SECOND LEG: GGUF CHUNK PREFIXES
#   docs/abi-changelog.md §Scope puts the GGUF metadata schema under the
#   `vokra.*` prefix IN scope, on the grounds that model files are content-
#   addressed by these chunks. Nothing enforced that, and 50 converter-
#   stamped prefixes accumulated with no entry. The verify mode now also
#   asserts: every `vokra.<group>` a converter stamps a key under must
#   appear somewhere in docs/abi-changelog.md, unless it is listed in the
#   declared-exception ledger below with a reason.
#
# MODES
#   scripts/check-abi-changelog.sh                  -- verify (both legs)
#   scripts/check-abi-changelog.sh --list           -- print current symbols
#   scripts/check-abi-changelog.sh --update-snapshot-- rewrite the anchor
#                                                     (owner action, requires
#                                                     a paired changelog entry)
#   scripts/check-abi-changelog.sh --check-gguf-prefixes
#                                                  -- run only the GGUF
#                                                     chunk-prefix leg
#   scripts/check-abi-changelog.sh --list-gguf-prefixes
#                                                  -- print the prefixes the
#                                                     converter crate stamps
#   scripts/check-abi-changelog.sh --self-test      -- unit-test both scanners
#   scripts/check-abi-changelog.sh --help           -- this text
#
# NOT WIRED INTO CI YET
#   The wiring into .github/workflows/ci.yml is deliberately left to a
#   later WP (M5-13) so this scaffold can land without blocking still-in-
#   flight M3 WPs whose ABI additions are only half-typed. Today, run it
#   from the pre-commit hook or manually before opening a PR.
#
# ZERO-DEP
#   Pure bash + awk + grep + diff. No `cbindgen`, no Rust toolchain, no
#   external crate needed to run the gate. It DOES NOT regenerate the
#   header — call `scripts/gen-c-abi.sh` first if you touched the FFI.
#
# EXIT CODES
#   0  clean (no delta, or delta covered by recorded symbol rows,
#      or a --list / --self-test / --update-snapshot success)
#   1  delta detected AND one or more symbols have no changelog row, OR a converter
#      stamps a `vokra.<group>.*` chunk prefix with no mention in
#      docs/abi-changelog.md and no ledger exception
#   2  usage / setup error (missing header, missing anchor, missing
#      converter tree, scanner returned nothing, bad flag)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HEADER="$ROOT/include/vokra.h"
ANCHOR="$ROOT/docs/abi/vokra.h.v1.0-rc-baseline.symbols"
CHANGELOG="$ROOT/docs/abi-changelog.md"

usage() {
    # Print the whole banner (line 3 .. the last `#` line before `set -e`)
    # rather than a hard-coded upper bound, so adding a section to the
    # header cannot silently truncate --help.
    local last
    last="$(grep -nE '^set -euo pipefail' "$0" | head -1 | cut -d: -f1)"
    sed -n "3,$((last - 2))p" "$0" | sed 's/^# \{0,1\}//'
}

# ---------------------------------------------------------------- extract ---
# extract_symbols <header-path>
#
# Reads a C header and emits one normalized "symbol line" per exported
# entity, sorted (LC_ALL=C so the order is portable). A symbol is:
#
#   FUNC <name>|<normalized-prototype>
#   TYPEDEF <name>|<normalized-declaration>
#
# Pipeline: strip block+line comments, join lines (preserve ;), split on ;,
# keep statements that either (a) declare a vokra_* function or (b) start
# with `typedef` and mention `vokra_`, collapse internal whitespace.
#
# The extractor is intentionally conservative: it does NOT try to be a C
# parser. It works because `include/vokra.h` is a cbindgen output with a
# very small vocabulary (function prototypes + typedef struct/enum), which
# lets us round-trip it through this awk-driven normalizer without loss.
extract_symbols() {
    local header="$1"
    if [ ! -f "$header" ]; then
        echo "error: header not found: $header" >&2
        return 2
    fi

    # Single awk that does the whole pipeline in one process:
    #   phase A -- state machine over the file, strips /* ... */ (multi-
    #              line safe) and // comments AND drops any line whose
    #              first non-space char is `#` (preprocessor). Emits the
    #              surviving text char-by-char into a growing buffer.
    #   phase B -- once EOF hits, brace-aware split of the buffer on `;`
    #              at brace-depth 0. Struct bodies (which contain
    #              intra-brace `;`) survive as one logical statement.
    #   phase C -- for each statement: collapse whitespace, drop obvious
    #              noise (`extern "C" {`, `}`), then classify as either a
    #              vokra_ function prototype or a typedef that names a
    #              vokra_ identifier. Emits the FUNC/TYPEDEF line.
    awk '
        BEGIN {
            in_block = 0
            buf = ""
        }
        {
            # First: if this line is a preprocessor directive (first
            # non-space char is `#`), drop it entirely BEFORE we start
            # accumulating characters. Preprocessor is not part of the
            # ABI surface; skipping it wholesale avoids polluting the
            # brace-aware splitter.
            probe = $0
            sub(/^[[:space:]]+/, "", probe)
            if (in_block == 0 && substr(probe, 1, 1) == "#") next

            line = $0
            n = length(line)
            i = 1
            while (i <= n) {
                if (in_block) {
                    p = index(substr(line, i), "*/")
                    if (p == 0) {
                        i = n + 1
                    } else {
                        i = i + p + 1
                        in_block = 0
                    }
                    continue
                }
                rest = substr(line, i)
                bo = index(rest, "/*")
                lo = index(rest, "//")
                if (bo > 0 && (lo == 0 || bo < lo)) {
                    buf = buf substr(rest, 1, bo - 1)
                    i = i + bo + 1
                    in_block = 1
                } else if (lo > 0) {
                    buf = buf substr(rest, 1, lo - 1)
                    i = n + 1
                } else {
                    buf = buf rest
                    i = n + 1
                }
            }
            buf = buf " "  # newline -> space so tokens do not fuse
        }
        END {
            # `extern "C" {` (and the C++ variant) wraps prototypes but
            # holds no semicolons of its own — its braces are depth-
            # neutral for the splitter. Strip the opener and the trailing
            # closer so phase B does not treat the whole extern block as
            # one giant statement.
            gsub(/extern[[:space:]]+"C"[[:space:]]*\{/, " ", buf)
            gsub(/extern[[:space:]]+"C\+\+"[[:space:]]*\{/, " ", buf)
            sub(/[[:space:]]*\}[[:space:]]*$/, "", buf)

            # Phase B: brace-aware split of `buf` on `;` at depth 0.
            depth = 0
            stmt = ""
            L = length(buf)
            for (k = 1; k <= L; k++) {
                c = substr(buf, k, 1)
                if (c == "{") { depth++; stmt = stmt c; continue }
                if (c == "}") { depth--; stmt = stmt c; continue }
                if (c == ";" && depth == 0) {
                    emit(stmt)
                    stmt = ""
                    continue
                }
                stmt = stmt c
            }
            # Trailing chunk without a `;` — normally empty, but emit for
            # safety so a stray `extern "C" {` block at EOF is discarded
            # by the classifier rather than lost as a false negative.
            emit(stmt)
        }

        function emit(s,    name, last, tail) {
            gsub(/[[:space:]]+/, " ", s)
            sub(/^[[:space:]]+/, "", s)
            sub(/[[:space:]]+$/, "", s)
            if (length(s) == 0) return
            if (s == "extern \"C\" {") return
            if (s == "}") return

            # Function prototype: contains `vokra_<ident>(`.
            if (match(s, /vokra_[A-Za-z0-9_]+[[:space:]]*\(/)) {
                name = substr(s, RSTART, RLENGTH)
                sub(/[[:space:]]*\($/, "", name)
                print "FUNC " name "|" s
                return
            }

            # Typedef that names a vokra_ identifier as the alias.
            # We take the LAST `vokra_<ident>` occurrence: for both
            #   `typedef struct X X`
            #   `typedef enum X { ... } X`
            #   `typedef struct X { ... } X`
            # the alias is the last one.
            if (match(s, /^typedef[[:space:]]/) && match(s, /vokra_[A-Za-z0-9_]+/)) {
                tail = s
                last = ""
                while (match(tail, /vokra_[A-Za-z0-9_]+/)) {
                    last = substr(tail, RSTART, RLENGTH)
                    tail = substr(tail, RSTART + RLENGTH)
                }
                print "TYPEDEF " last "|" s
                return
            }
        }
    ' "$header" \
    | LC_ALL=C sort -u
}

# ------------------------------------------------------------- self-test ---
# self_test — exercise the extractor against a small synthetic header so a
# future change to the awk pipeline can be caught without touching the real
# include/vokra.h. Uses a here-doc scratch file under $TMPDIR.
self_test() {
    local tmp
    tmp="$(mktemp -t vokra-abi-check.XXXXXX)"
    trap 'rm -f "$tmp"' RETURN

    cat >"$tmp" <<'EOF'
/* Fake header for the self-test.
 * Multi-line block comment on purpose.
 */
#ifndef VOKRA_TEST_H
#define VOKRA_TEST_H

typedef enum vokra_status_t {
    VOKRA_OK = 0,        // line comment
    VOKRA_ERROR_IO = 1,  /* inline block */
} vokra_status_t;

typedef struct vokra_session_t vokra_session_t;

// This is a decoy line, not an exported symbol.
enum vokra_status_t vokra_asr_transcribe(const struct vokra_session_t *session,
                                         const float *pcm,
                                         size_t num_samples);

void vokra_string_free(char *s);

// Not a Vokra symbol; must NOT be picked up.
int unrelated_function(int x);

#endif
EOF

    local got want
    got="$(extract_symbols "$tmp")"
    want="$(printf '%s\n' \
        'FUNC vokra_asr_transcribe|enum vokra_status_t vokra_asr_transcribe(const struct vokra_session_t *session, const float *pcm, size_t num_samples)' \
        'FUNC vokra_string_free|void vokra_string_free(char *s)' \
        'TYPEDEF vokra_session_t|typedef struct vokra_session_t vokra_session_t' \
        'TYPEDEF vokra_status_t|typedef enum vokra_status_t { VOKRA_OK = 0, VOKRA_ERROR_IO = 1, } vokra_status_t' \
        | LC_ALL=C sort -u)"

    if [ "$got" != "$want" ]; then
        echo "self-test FAILED — extractor drift:" >&2
        diff -u <(printf '%s\n' "$want") <(printf '%s\n' "$got") >&2 || true
        return 1
    fi
    echo "check-abi-changelog --self-test: extractor OK"
    self_test_gguf_prefixes || return 1
    echo "check-abi-changelog --self-test: OK"
    return 0
}

# self_test_gguf_prefixes — drive converter_gguf_prefixes over a synthetic
# converter tree. Pins the two narrowings that keep the leg from crying
# wolf, so a future edit to the grep/sed pipeline cannot quietly widen it.
self_test_gguf_prefixes() {
    local dir
    dir="$(mktemp -d -t vokra-abi-gguf.XXXXXX)"
    # shellcheck disable=SC2064
    trap "rm -rf '$dir'" RETURN

    mkdir -p "$dir/models"
    cat >"$dir/models/fixture.rs" <<'EOF'
//! Doc comment naming "vokra.ghost.beta" — prose must never invent a group.
/// Another doc line with "vokra.phantom.gamma".
// A plain comment with "vokra.spectre.delta".

const KEY_DEMO_ALPHA: &str = "vokra.demo.alpha";
const KEY_DEMO_NESTED: &str = "vokra.demo.arch.width";

fn stamp(b: &mut GgufBuilder) {
    b.add_u32(KEY_DEMO_ALPHA, 1);
    b.add_u32(KEY_DEMO_NESTED, 2);
    b.add_string("vokra.inline.gamma", "x");
    b.add_u32(&format!("vokra.indexed.ratio_{i}"), 3);
    b.add_string("general.name", "not-a-vokra-chunk");
}

#[cfg(test)]
mod tests {
    // The voila shape: a bare prefix used as a starts_with guard, in a test
    // that asserts NO such key is emitted. Must NOT be reported as stamped.
    fn guard(key: &str) -> bool {
        key.starts_with("vokra.voila.")
    }
}
EOF

    local got want
    got="$(converter_gguf_prefixes "$dir")"
    want="$(printf '%s\n' 'vokra.demo' 'vokra.indexed' 'vokra.inline' | LC_ALL=C sort -u)"

    if [ "$got" != "$want" ]; then
        echo "self-test FAILED — GGUF prefix scanner drift:" >&2
        diff -u <(printf '%s\n' "$want") <(printf '%s\n' "$got") >&2 || true
        return 1
    fi
    echo "check-abi-changelog --self-test: gguf prefix scanner OK"
    return 0
}

# --------------------------------------------------------- changelog gate ---
# changelog_mentions_symbol <symbol>
# True iff docs/abi-changelog.md contains a table row for the exact ABI symbol.
# Dates describe when a PR was opened; they are not a valid proxy for whether
# the symbol itself has already been documented when CI runs later.
changelog_mentions_symbol() {
    local symbol="$1"
    if [ ! -f "$CHANGELOG" ]; then
        return 0
    fi
    grep -qF "| \`$symbol\` |" "$CHANGELOG"
}

# ------------------------------------------------ GGUF chunk-prefix gate ---
# Declared-exception ledger for the GGUF chunk-prefix leg.
#
# A prefix listed here is one the converter crate mentions in a key literal
# but which legitimately needs no docs/abi-changelog.md row. Every line MUST
# carry a reason: an undocumented exception is a hole in the gate, and a
# ledger nobody has to justify is how the 50-prefix backlog happened in the
# first place.
#
# Format: `vokra.<group>` then whitespace then `# <reason>`. `#`-only lines
# and blanks are comments.
#
# EMPTY BY DESIGN as of 2026-08-15 — every prefix the converter crate stamps
# now has a row. The plausible future entry is a key that exists only inside
# a `#[cfg(test)]` fixture and never reaches a shipped `.gguf`; the scanner
# cannot tell that from a real stamp, so such a case belongs here with the
# test named.
GGUF_PREFIX_EXCEPTIONS="$(
    cat <<'LEDGER'
LEDGER
)"

# converter_gguf_prefixes [dir]
#
# Emits the sorted, unique set of `vokra.<group>` chunk prefixes that the
# converter crate stamps a key under. `dir` defaults to the converter source
# tree and exists so --self-test can drive the same code over a fixture.
#
# Two deliberate narrowings keep this from crying wolf:
#
#   1. Whole-line comments are dropped BEFORE the literal scan, so prose in
#      a rustdoc block cannot invent a chunk group.
#   2. A key literal must have a LEAF: `"vokra.foo.bar"` counts,
#      `"vokra.foo."` does not. That bare form is the `starts_with` guard
#      shape, and matching it would flag `models/voila.rs` — which mentions
#      `vokra.voila.` only to assert in a test that no such key is ever
#      emitted. Flagging a converter for refusing to stamp axes would be
#      exactly backwards.
#
# Known scope limit (documented, not hidden): this covers
# crates/vokra-convert only. `vokra.denoise.*` is written by
# `DenoiseModel::to_gguf_bytes` in vokra-ops, and `vokra.schema.*` /
# `vokra.provenance.*` / `vokra.model.*` by `GgufBuilder::effective_metadata`
# in vokra-core. Both are already recorded in docs/abi-changelog.md; widening
# the scan to those crates would also sweep up every READER of a key (the
# `from_gguf` side in vokra-models names the same strings), which is a false
# positive the ledger would have to absorb wholesale.
converter_gguf_prefixes() {
    local src="${1:-$ROOT/crates/vokra-convert/src}"
    if [ ! -d "$src" ]; then
        echo "error: converter source tree not found: $src" >&2
        return 2
    fi
    grep -rhE --include='*.rs' -v '^[[:space:]]*(//|\*|/\*)' "$src" 2>/dev/null \
        | grep -ohE '"vokra\.[a-z0-9_]+\.[a-z0-9_][a-z0-9_.{}]*"' \
        | sed -E 's/^"(vokra\.[a-z0-9_]+)\..*$/\1/' \
        | LC_ALL=C sort -u \
        || true
}

# changelog_mentions_prefix <vokra.group>
#
# True iff docs/abi-changelog.md names the prefix. The trailing character
# class stops `vokra.canary` from being satisfied by a `vokra.canary_qwen`
# mention (and vice versa) — the two are different chunk groups and each
# needs its own record.
changelog_mentions_prefix() {
    local esc
    esc="$(printf '%s' "$1" | sed 's/\./\\./g')"
    grep -qE "${esc}([^A-Za-z0-9_]|\$)" "$CHANGELOG"
}

# check_gguf_prefixes [dir] — the leg itself. 0 = clean, 1 = unrecorded
# prefix found, 2 = setup problem.
check_gguf_prefixes() {
    local src="${1:-$ROOT/crates/vokra-convert/src}"
    local prefixes exceptions missing=() skipped=0 pre total

    if [ ! -f "$CHANGELOG" ]; then
        echo "error: changelog not found: $CHANGELOG" >&2
        return 2
    fi

    prefixes="$(converter_gguf_prefixes "$src")" || return 2
    if [ -z "$prefixes" ]; then
        # A scanner that quietly matches nothing is worse than no scanner:
        # it reports success forever. Treat an empty sweep of the real tree
        # as a setup error, not a pass.
        echo "error: no vokra.* chunk prefix found under $src" >&2
        echo "       the scanner regressed, or the tree moved — not a pass" >&2
        return 2
    fi

    exceptions="$(printf '%s\n' "$GGUF_PREFIX_EXCEPTIONS" \
        | grep -Ev '^[[:space:]]*(#|$)' \
        | awk '{print $1}' || true)"

    total=0
    while IFS= read -r pre; do
        [ -n "$pre" ] || continue
        total=$((total + 1))
        if [ -n "$exceptions" ] && printf '%s\n' "$exceptions" | grep -qx "$pre"; then
            skipped=$((skipped + 1))
            continue
        fi
        changelog_mentions_prefix "$pre" || missing+=("$pre")
    done <<EOF
$prefixes
EOF

    echo "  gguf    : $total converter-stamped chunk prefixes, ${skipped} ledger exception(s)"

    if [ ${#missing[@]} -eq 0 ]; then
        echo "  gguf    : OK (every prefix appears in docs/abi-changelog.md)"
        return 0
    fi

    {
        echo ""
        echo "check-abi-changelog: FAIL — ${#missing[@]} GGUF chunk prefix(es)"
        echo "are stamped by a converter but appear nowhere in"
        echo "docs/abi-changelog.md:"
        printf '  %s.*\n' "${missing[@]}"
        echo ""
        echo "docs/abi-changelog.md §\"Scope: what belongs in this file\" puts the"
        echo "GGUF metadata schema in scope — model files are content-addressed by"
        echo "these chunks, so an unrecorded group is an undocumented on-disk"
        echo "compatibility surface."
        echo ""
        echo "Fix (pick one, honestly):"
        echo "  1. Add a row to the GGUF metadata additions block naming the"
        echo "     prefix, the converter that stamps it, and whether it is"
        echo "     additive. Follow the existing table columns."
        echo "  2. If the prefix genuinely needs no row (e.g. it only exists in"
        echo "     a #[cfg(test)] fixture), add it to GGUF_PREFIX_EXCEPTIONS in"
        echo "     this script WITH a reason naming the test."
    } >&2
    return 1
}

# ------------------------------------------------------------------ main ---
mode="${1:-verify}"
case "$mode" in
    verify|"")
        # Leg 2 (GGUF chunk prefixes) runs first and its status is carried
        # to the end, so a C-ABI-clean tree still fails on an unrecorded
        # chunk group instead of exiting 0 before the leg is reached.
        prefix_rc=0
        check_gguf_prefixes || prefix_rc=$?
        if [ "$prefix_rc" -eq 2 ]; then
            exit 2
        fi

        # Extract fresh, compare against anchor.
        if [ ! -f "$ANCHOR" ]; then
            echo "error: anchor snapshot missing: $ANCHOR" >&2
            echo "       run: scripts/check-abi-changelog.sh --update-snapshot" >&2
            exit 2
        fi
        current="$(extract_symbols "$HEADER")"
        # Strip the `#`-prefixed banner from the anchor before comparing;
        # only FUNC/TYPEDEF rows are payload. Order in the anchor is
        # already `sort -u`-stable, but we re-sort defensively so a hand
        # edit to the anchor cannot make the gate falsely diff-clean.
        anchor="$(grep -Ev '^[[:space:]]*(#|$)' "$ANCHOR" | LC_ALL=C sort -u)"

        # Count symbols in the anchor for the human-readable summary.
        func_count=$(printf '%s\n' "$anchor" | grep -c '^FUNC ' || true)
        type_count=$(printf '%s\n' "$anchor" | grep -c '^TYPEDEF ' || true)

        echo "Vokra ABI changelog gate (M3-16; IF-01 fires at M5-13, not here)"
        echo "  header  : $HEADER"
        echo "  anchor  : $ANCHOR"
        echo "  anchor  : $func_count exported functions, $type_count typedefs"

        if diff_out="$(diff -u <(printf '%s\n' "$anchor") <(printf '%s\n' "$current"))"; then
            echo ""
            echo "check-abi-changelog: OK (baseline unchanged)"
            exit "$prefix_rc"
        fi

        echo ""
        echo "ABI delta detected between include/vokra.h and the v1.0-rc anchor:"
        printf '%s\n' "$diff_out" | sed 's/^/  /'
        echo ""

        # The anchor is intentionally long-lived through the rc window. Do
        # not require a heading dated *today*: that made a documented ABI
        # addition fail on every subsequent CI day. Extract the symbols from
        # the delta and require each one to have an explicit changelog row.
        delta_symbols="$(printf '%s' "$diff_out" | awk '/^[+-](FUNC|TYPEDEF) / { line = substr($0, 2); sub(/^(FUNC|TYPEDEF) /, "", line); sub(/\|.*/, "", line); print line }' | LC_ALL=C sort -u)"
        missing_symbols=()
        while IFS= read -r symbol; do
            [ -n "$symbol" ] || continue
            if ! changelog_mentions_symbol "$symbol"; then
                missing_symbols+=("$symbol")
            fi
        done <<< "$delta_symbols"

        if [ "${#missing_symbols[@]}" -eq 0 ]; then
            echo "check-abi-changelog: OK (every changed C ABI symbol has a changelog row)"
            echo ""
            echo "reminder: once the change is merged into the release cut,"
            echo "run 'scripts/check-abi-changelog.sh --update-snapshot' to"
            echo "advance the anchor and drop the entries into the immutable"
            echo "release section."
            exit "$prefix_rc"
        fi

        cat >&2 <<EOF
check-abi-changelog: FAIL — the C ABI moved but these changed symbols have
no row in docs/abi-changelog.md:

$(printf '  %s\n' "${missing_symbols[@]}")

Fix:
  1. If the change is intentional, add one table row per missing symbol to
     docs/abi-changelog.md following the schema at the top of that file
     (inside a dated section, with rationale + WP/PR id).
  2. If the change is accidental (e.g. cbindgen drift on an unrelated
     refactor), revert the include/vokra.h diff or fix the vokra-capi Rust
     source that produced it.

The v1.0-rc anchor at $ANCHOR is only rotated by
'scripts/check-abi-changelog.sh --update-snapshot' — do not edit it by
hand.
EOF
        exit 1
        ;;

    --list)
        extract_symbols "$HEADER"
        ;;

    --check-gguf-prefixes)
        set +e
        check_gguf_prefixes
        exit $?
        ;;

    --list-gguf-prefixes)
        converter_gguf_prefixes
        ;;

    --update-snapshot)
        # Owner action: replace the anchor with the current extraction and
        # commit it alongside the changelog entry that describes the delta.
        # We deliberately do NOT auto-commit; the caller must review.
        mkdir -p "$(dirname "$ANCHOR")"
        {
            echo "# Vokra C ABI anchor snapshot — v1.0-rc (M4) window."
            echo "#"
            echo "# Origin: the v1.0-rc-window ABI anchor for the M4 prerelease"
            echo "# series (semver 1.0.0-rc.N). Rotated here from the v0.9 anchor"
            echo "# by M4-12 (2026-07-14 v-label reassignment #2 re-scope). The"
            echo "# capture commit + exported-symbol counts are recorded in"
            echo '# docs/abi-changelog.md "Baseline snapshot: v1.0-rc".'
            echo "#"
            echo "# Regenerate with: scripts/check-abi-changelog.sh --update-snapshot"
            echo "# Diff against with: scripts/check-abi-changelog.sh"
            echo "# Historical diff:  scripts/abi-diff.sh --anchor v1.0-rc"
            echo "#"
            echo "# One line per exported symbol, format:"
            echo "#   FUNC <name>|<normalized prototype>"
            echo "#   TYPEDEF <name>|<normalized declaration>"
            echo "#"
            echo "# NOT frozen: the IF-01 freeze fires at v1.0 GA (M5-13), not"
            echo "# here. The pre-1.0 prerelease policy (free rename/remove with a"
            echo "# dated changelog entry) stays in force through the rc series."
            echo "# See docs/abi-changelog.md for the schema and freeze policy."
            extract_symbols "$HEADER"
        } >"$ANCHOR"
        func_count=$(grep -c '^FUNC ' "$ANCHOR" || true)
        type_count=$(grep -c '^TYPEDEF ' "$ANCHOR" || true)
        echo "check-abi-changelog: wrote $ANCHOR"
        echo "  anchored $func_count exported functions, $type_count typedefs"
        ;;

    --self-test)
        set +e
        self_test
        exit $?
        ;;

    --help|-h)
        usage
        exit 0
        ;;

    *)
        echo "error: unknown argument '$mode'" >&2
        echo "usage: $0 [--list | --update-snapshot | --self-test | --help]" >&2
        exit 2
        ;;
esac
