#!/usr/bin/env bash
# check-abi-changelog.sh — M3-16 v0.9 ABI changelog scaffold.
#
# WHAT IT GATES
#   For the v0.9 (M3) release window the C ABI is a moving target (the
#   IF-01 semver freeze fires at M4-12 / v1.0 GA, not here — see
#   docs/abi-changelog.md and the STABILITY block at the top of
#   include/vokra.h). During this window we still want every symbol delta
#   to be *observable* on-disk: this script diffs the working-tree
#   include/vokra.h against a committed anchor snapshot and, if it finds a
#   delta, requires docs/abi-changelog.md to have an entry dated today.
#
# ARTEFACTS
#   include/vokra.h                                 -- current C header (cbindgen)
#   docs/abi/vokra.h.v0.9-baseline.symbols          -- anchor snapshot
#   docs/abi-changelog.md                           -- narrative + entries
#
# MODES
#   scripts/check-abi-changelog.sh                  -- verify (default)
#   scripts/check-abi-changelog.sh --list           -- print current symbols
#   scripts/check-abi-changelog.sh --update-snapshot-- rewrite the anchor
#                                                     (owner action, requires
#                                                     a paired changelog
#                                                     entry dated today)
#   scripts/check-abi-changelog.sh --self-test      -- unit-test the extractor
#   scripts/check-abi-changelog.sh --help           -- this text
#
# NOT WIRED INTO CI YET
#   The wiring into .github/workflows/ci.yml is deliberately left to a
#   later WP (M4-12) so this scaffold can land without blocking still-in-
#   flight M3 WPs whose ABI additions are only half-typed. Today, run it
#   from the pre-commit hook or manually before opening a PR.
#
# ZERO-DEP
#   Pure bash + awk + grep + diff. No `cbindgen`, no Rust toolchain, no
#   external crate needed to run the gate. It DOES NOT regenerate the
#   header — call `scripts/gen-c-abi.sh` first if you touched the FFI.
#
# EXIT CODES
#   0  clean (no delta, or delta covered by a today-dated changelog entry,
#      or a --list / --self-test / --update-snapshot success)
#   1  delta detected AND no today-dated changelog entry
#   2  usage / setup error (missing header, missing anchor, bad flag)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HEADER="$ROOT/include/vokra.h"
ANCHOR="$ROOT/docs/abi/vokra.h.v0.9-baseline.symbols"
CHANGELOG="$ROOT/docs/abi-changelog.md"

usage() {
    sed -n '3,32p' "$0" | sed 's/^# \{0,1\}//'
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
    echo "check-abi-changelog --self-test: OK"
    return 0
}

# --------------------------------------------------------- changelog gate ---
# has_today_entry — echo `yes` iff docs/abi-changelog.md contains an
# `### YYYY-MM-DD ...` header dated today. We use `date -u +%F` so the gate
# does not flake across time zones. UTC is the canonical date for the file.
has_today_entry() {
    local today
    today="$(date -u +%F)"
    if [ ! -f "$CHANGELOG" ]; then
        echo "no"
        return 0
    fi
    if grep -qE "^### ${today}( |—|$)" "$CHANGELOG"; then
        echo "yes"
    else
        echo "no"
    fi
}

# ------------------------------------------------------------------ main ---
mode="${1:-verify}"
case "$mode" in
    verify|"")
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

        echo "Vokra ABI changelog gate (M3-16; IF-01 fires at M4-12, not here)"
        echo "  header  : $HEADER"
        echo "  anchor  : $ANCHOR"
        echo "  anchor  : $func_count exported functions, $type_count typedefs"

        if diff_out="$(diff -u <(printf '%s\n' "$anchor") <(printf '%s\n' "$current"))"; then
            echo ""
            echo "check-abi-changelog: OK (baseline unchanged)"
            exit 0
        fi

        echo ""
        echo "ABI delta detected between include/vokra.h and the v0.9 anchor:"
        printf '%s\n' "$diff_out" | sed 's/^/  /'
        echo ""

        if [ "$(has_today_entry)" = "yes" ]; then
            today="$(date -u +%F)"
            echo "check-abi-changelog: OK (delta covered by a $today entry in docs/abi-changelog.md)"
            echo ""
            echo "reminder: once the change is merged into the release cut,"
            echo "run 'scripts/check-abi-changelog.sh --update-snapshot' to"
            echo "advance the anchor and drop the entries into the immutable"
            echo "release section."
            exit 0
        fi

        cat >&2 <<EOF
check-abi-changelog: FAIL — the C ABI moved but docs/abi-changelog.md has
no entry dated $(date -u +%F).

Fix:
  1. If the change is intentional, add a section
       ### $(date -u +%F) — 0.9.0-dev
     to docs/abi-changelog.md following the schema at the top of that file
     (one row per symbol, with rationale + WP/PR id).
  2. If the change is accidental (e.g. cbindgen drift on an unrelated
     refactor), revert the include/vokra.h diff or fix the vokra-capi Rust
     source that produced it.

The v0.9 anchor at $ANCHOR is only rotated by
'scripts/check-abi-changelog.sh --update-snapshot' — do not edit it by
hand.
EOF
        exit 1
        ;;

    --list)
        extract_symbols "$HEADER"
        ;;

    --update-snapshot)
        # Owner action: replace the anchor with the current extraction and
        # commit it alongside the changelog entry that describes the delta.
        # We deliberately do NOT auto-commit; the caller must review.
        mkdir -p "$(dirname "$ANCHOR")"
        {
            echo "# Vokra C ABI anchor snapshot — v0.9 window."
            echo "#"
            echo "# Regenerate with: scripts/check-abi-changelog.sh --update-snapshot"
            echo "# Diff against with: scripts/check-abi-changelog.sh"
            echo "#"
            echo "# One line per exported symbol, format:"
            echo "#   FUNC <name>|<normalized prototype>"
            echo "#   TYPEDEF <name>|<normalized declaration>"
            echo "#"
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
