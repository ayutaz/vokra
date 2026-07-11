#!/usr/bin/env bash
# abi-diff.sh — M3-16-T02 C ABI historical diff tool.
#
# WHAT IT DOES
#   Compares the working-tree `include/vokra.h` against a historical anchor
#   (default: the M0 baseline `include/vokra.h.m0-anchor` — the raw M0
#   cbindgen output verbatim from commit fe9b67f, captured per M3-16-T02
#   with a top-of-file provenance/retrieval comment) and emits an added /
#   removed / changed report over the exported C surface (functions +
#   typedefs). Feeds the M3-16-T04 changelog aggregation.
#
#   This is intentionally SEPARATE from `scripts/check-abi-changelog.sh`:
#   - `check-abi-changelog.sh` gates the v0.9 window (asks: "does a today-
#      dated changelog entry cover the drift against the v0.9 baseline?"),
#      and is the pre-commit / advisory CI hook.
#   - `abi-diff.sh` (this script) is a REPORT tool. It never fails on a
#      delta; it just classifies the delta so the M3-16-T04 aggregator, and
#      later the M4-12 freeze-flip, can consume it as input. Exit codes are
#      reserved for setup errors (missing anchor / bad flag / missing
#      header).
#
# ARTEFACTS
#   include/vokra.h                                 -- current C header (cbindgen)
#   include/vokra.h.m0-anchor                       -- M0 (v0.1 spike) raw
#                                                      header snapshot at
#                                                      commit fe9b67f;
#                                                      consumed by `--anchor
#                                                      m0` (default). See
#                                                      that file's top-of-
#                                                      file comment block
#                                                      for the retrieval
#                                                      command per M3-16-T02.
#   docs/abi/vokra.h.v0.9-baseline.symbols          -- v0.9 window baseline
#                                                      (`--anchor v0.9`,
#                                                      symbols-format,
#                                                      rotated by check-abi-
#                                                      changelog.sh
#                                                      --update-snapshot).
#
# ANCHOR FORMATS
#   The script auto-detects the anchor format so callers can point
#   `--anchor <path>` at either a raw C header (like `include/vokra.h`)
#   or a pre-extracted symbols file (like the v0.9 baseline). Detection
#   is content-based: if the file has any `FUNC <name>|...` or
#   `TYPEDEF <name>|...` payload lines (after stripping `#`-comments and
#   blanks), it is treated as pre-extracted symbols; otherwise it is a
#   raw C header and the same awk pipeline as `check-abi-changelog.sh`
#   is applied. This lets us honour the M3-16-T02 spec (raw M0 header
#   snapshot at `include/vokra.h.m0-anchor`) without breaking callers
#   that pass in a symbols-format anchor.
#
# MODES
#   scripts/abi-diff.sh                          -- diff current vs. M0 anchor (default)
#   scripts/abi-diff.sh --anchor <path|label>    -- diff against a named anchor.
#                                                   Label shortcuts: `m0` (default,
#                                                   raw header), `v0.9` (symbols).
#                                                   Anything else is treated as a
#                                                   path; the format is auto-
#                                                   detected (see ANCHOR FORMATS).
#   scripts/abi-diff.sh --header <path>          -- override the header path
#                                                   (defaults to include/vokra.h)
#   scripts/abi-diff.sh --regenerate             -- before diffing, invoke
#                                                   `scripts/gen-c-abi.sh` to
#                                                   regenerate `include/vokra.h`
#                                                   into a tempfile (the working-
#                                                   tree header is left untouched)
#                                                   and diff against that. This is
#                                                   the "gen-c-abi.sh --check style
#                                                   flow to a tempfile" the M3-16
#                                                   spec calls for. Requires
#                                                   cbindgen on PATH; without it
#                                                   the flag exits 2.
#   scripts/abi-diff.sh --format text            -- human-readable (default)
#   scripts/abi-diff.sh --format machine         -- one line per delta, prefixed
#                                                   `+ FUNC ...` / `- FUNC ...` /
#                                                   `~ FUNC ...` for scripts.
#   scripts/abi-diff.sh --self-test              -- unit-test the diff classifier
#   scripts/abi-diff.sh --help                   -- this text
#
# NOT WIRED INTO CI
#   Per M3-16 spec §T02 last paragraph: CI gating is left to M4-12
#   (`docs/tickets/m3/M3-16-abi-changelog.md` §T02 内容 bullet 4). This
#   script produces a report — the caller decides whether a delta is
#   expected.
#
# ZERO-DEP
#   Pure bash + awk + grep + diff + sed. Reuses the extractor pattern
#   established by `scripts/check-abi-changelog.sh` (the M0 raw header was
#   captured verbatim from commit fe9b67f into `include/vokra.h.m0-anchor`
#   per M3-16-T02, with a top-of-file retrieval-command comment). The
#   optional `--regenerate` flag additionally shells out to
#   `scripts/gen-c-abi.sh`, which itself requires cbindgen (build-time
#   tool only, not a runtime dep — same policy as gen-c-abi.sh, so this
#   does not perturb NFR-DS-02).
#
# EXIT CODES
#   0  ran to completion (regardless of whether a delta was found)
#   2  usage / setup error (missing header, missing anchor, bad flag,
#      or `--regenerate` requested but cbindgen unavailable / regen failed)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_HEADER="$ROOT/include/vokra.h"
# --anchor m0 -> the raw M0 header snapshot at commit fe9b67f (captured
# per M3-16-T02, see the file's top-of-file comment). The script auto-
# detects the format (raw C header vs. pre-extracted symbols) so callers
# can point --anchor at either kind of file.
ANCHOR_M0="$ROOT/include/vokra.h.m0-anchor"
ANCHOR_V09="$ROOT/docs/abi/vokra.h.v0.9-baseline.symbols"
GEN_C_ABI="$ROOT/scripts/gen-c-abi.sh"

usage() {
    # The help banner is the block-comment header at the top of this file.
    # Emit lines 3-<end-of-header> stripped of the leading `# `. The header
    # ends at the first blank hash-less line (`^set -euo pipefail` etc).
    awk '
        NR < 3 { next }
        /^set -/ { exit }
        /^[^#]/  { exit }
        { sub(/^# ?/, ""); print }
    ' "$0"
}

# ---------------------------------------------------------------- extract ---
# extract_symbols <header-path>
#
# Same pipeline as `scripts/check-abi-changelog.sh::extract_symbols`. We
# duplicate the awk here rather than sourcing check-abi-changelog.sh so
# this script has no cross-file coupling (either script can be moved
# without breaking the other). If the two ever drift, the self-test
# below catches it.
extract_symbols() {
    local header="$1"
    if [ ! -f "$header" ]; then
        echo "error: header not found: $header" >&2
        return 2
    fi

    awk '
        BEGIN {
            in_block = 0
            buf = ""
        }
        {
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
            buf = buf " "
        }
        END {
            gsub(/extern[[:space:]]+"C"[[:space:]]*\{/, " ", buf)
            gsub(/extern[[:space:]]+"C\+\+"[[:space:]]*\{/, " ", buf)
            sub(/[[:space:]]*\}[[:space:]]*$/, "", buf)

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
            emit(stmt)
        }

        function emit(s,    name, last, tail) {
            gsub(/[[:space:]]+/, " ", s)
            sub(/^[[:space:]]+/, "", s)
            sub(/[[:space:]]+$/, "", s)
            if (length(s) == 0) return
            if (s == "extern \"C\" {") return
            if (s == "}") return

            if (match(s, /vokra_[A-Za-z0-9_]+[[:space:]]*\(/)) {
                name = substr(s, RSTART, RLENGTH)
                sub(/[[:space:]]*\($/, "", name)
                print "FUNC " name "|" s
                return
            }

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

# ---------------------------------------------------------------- anchor ----
# read_anchor <path>
#
# Reads a pre-extracted `.symbols` file (e.g. the v0.9 baseline produced by
# `scripts/check-abi-changelog.sh --update-snapshot`), strips the `#`-
# prefixed banner + blank lines, and re-sorts defensively so a hand edit
# to the anchor cannot silently produce a false-clean diff. Caller MUST
# have decided the file is symbols-format; use `load_anchor_symbols`
# below to auto-route between raw-header and symbols-format anchors.
read_anchor() {
    local anchor="$1"
    if [ ! -f "$anchor" ]; then
        echo "error: anchor not found: $anchor" >&2
        return 2
    fi
    grep -Ev '^[[:space:]]*(#|$)' "$anchor" | LC_ALL=C sort -u
}

# is_symbols_format <path>
#
# Returns 0 (true) if the file at <path> looks like a pre-extracted
# symbols file (contains at least one `FUNC <name>|...` or `TYPEDEF
# <name>|...` line after stripping `#`-comments and blanks), else 1
# (false — treat as a raw C header). Content-based detection so callers
# do NOT need to name their anchors with a specific extension; both
# `include/vokra.h.m0-anchor` (raw) and `docs/abi/*.symbols` (extracted)
# route correctly.
#
# Why grep -q on a purpose-built regex instead of "just look at the
# first non-comment line": the anchor might have an empty payload
# section (e.g. a placeholder) or interleave comments with payload
# rows. Scanning for the `KIND name|` shape is unambiguous — a raw C
# header never contains that literal syntax at column 0, because a `|`
# in C is bitwise-OR and would not begin a line after
# `FUNC ` / `TYPEDEF `.
is_symbols_format() {
    local path="$1"
    if [ ! -f "$path" ]; then
        return 1
    fi
    if grep -Ev '^[[:space:]]*(#|$)' "$path" \
        | grep -Eq '^(FUNC|TYPEDEF) [A-Za-z_][A-Za-z0-9_]*\|'; then
        return 0
    fi
    return 1
}

# load_anchor_symbols <path>
#
# Emits the normalized symbol list for the anchor at <path>, auto-
# detecting the format:
#   - if `is_symbols_format` returns true -> read_anchor (fast path,
#     the anchor is already normalized)
#   - else                                -> extract_symbols (the anchor
#     is a raw C header, run the same awk pipeline check-abi-changelog.sh
#     uses so the extractor cannot silently drift between the two tools)
#
# The M3-16-T02 M0 anchor `include/vokra.h.m0-anchor` is raw-header
# form (task explicit spec: "snapshot the current include/vokra.h as
# it existed at the M0 tag"). The v0.9 baseline is symbols form
# (unchanged from check-abi-changelog.sh --update-snapshot).
load_anchor_symbols() {
    local path="$1"
    if [ ! -f "$path" ]; then
        echo "error: anchor not found: $path" >&2
        return 2
    fi
    if is_symbols_format "$path"; then
        read_anchor "$path"
    else
        extract_symbols "$path"
    fi
}

# --------------------------------------------------------------- classify ---
# classify <anchor-symbols> <current-symbols>
#
# Consumes two symbol lists (each line = `KIND name|prototype`) already
# sorted, and prints a categorized report on stdout. The report has four
# lines per non-empty class:
#
#   ADDED    <count>
#     <symbol> :: <prototype>
#   REMOVED  <count>
#     <symbol> :: <prototype>
#   CHANGED  <count>
#     <symbol>
#       -    <old-prototype>
#       +    <new-prototype>
#   UNCHANGED <count>
#     (no rows unless --verbose-unchanged, reserved for a future flag)
#
# The classifier is line-oriented: a "changed" row is one whose `KIND name`
# key appears in both sides but the payload (`|`-delimited prototype)
# differs. That is stricter than a plain `diff -u` because we want the
# M3-16-T04 aggregator to see a rename or a signature tweak as one row,
# not two (add + remove). This is what `docs/tickets/m3/M3-16-abi-changelog.md`
# §T02 抽出項目 bullet "signature 変更" asks for.
#
# The output format is intentionally simple text so the report renders
# well both in a terminal and pasted into a changelog PR. The machine
# format (`--format machine`) is a strict single-line-per-delta stream
# consumable by `awk '{print $2}'` etc.
classify() {
    local anchor_syms="$1"
    local current_syms="$2"
    local format="$3"

    # Build key-only lists (KIND name) for set arithmetic. We use LC_ALL=C
    # everywhere so the order is portable across GNU/BSD sort.
    local anchor_keys current_keys
    anchor_keys="$(printf '%s\n' "$anchor_syms" | awk -F'|' '{print $1}' | LC_ALL=C sort -u)"
    current_keys="$(printf '%s\n' "$current_syms" | awk -F'|' '{print $1}' | LC_ALL=C sort -u)"

    # `comm` gives us the three sets in one process each:
    #   comm -23  -> anchor-only  (removed)
    #   comm -13  -> current-only (added)
    #   comm -12  -> in both      (candidates for changed vs unchanged)
    #
    # `comm` needs sorted inputs (already true) and BSD/GNU-portable flags.
    local removed_keys added_keys common_keys
    removed_keys="$(comm -23 <(printf '%s\n' "$anchor_keys") <(printf '%s\n' "$current_keys"))"
    added_keys="$(comm -13 <(printf '%s\n' "$anchor_keys") <(printf '%s\n' "$current_keys"))"
    common_keys="$(comm -12 <(printf '%s\n' "$anchor_keys") <(printf '%s\n' "$current_keys"))"

    # Look up prototype for a key on a given side. We use grep -F -x on the
    # start-of-line prefix to avoid regex escaping on function names — the
    # anchor is `sort -u`-stable, so at most one line matches.
    lookup() {
        local key="$1" side_syms="$2"
        printf '%s\n' "$side_syms" | awk -F'|' -v k="$key" '$1 == k { print $0; exit }'
    }

    # Split "changed" from "unchanged" by comparing full lines within the
    # common_keys set.
    local changed_rows=""
    local unchanged_count=0
    if [ -n "$common_keys" ]; then
        while IFS= read -r key; do
            [ -z "$key" ] && continue
            local a c
            a="$(lookup "$key" "$anchor_syms")"
            c="$(lookup "$key" "$current_syms")"
            if [ "$a" = "$c" ]; then
                unchanged_count=$((unchanged_count + 1))
            else
                # Emit a marker line so the caller can regroup after read.
                changed_rows+="$key"$'\n'"$a"$'\n'"$c"$'\n'"---"$'\n'
            fi
        done <<<"$common_keys"
    fi

    # Counts.
    local added_count removed_count changed_count
    added_count=$([ -z "$added_keys" ] && echo 0 || printf '%s\n' "$added_keys" | grep -c . || true)
    removed_count=$([ -z "$removed_keys" ] && echo 0 || printf '%s\n' "$removed_keys" | grep -c . || true)
    changed_count=0
    if [ -n "$changed_rows" ]; then
        changed_count=$(printf '%s' "$changed_rows" | grep -c '^---$' || true)
    fi

    if [ "$format" = "machine" ]; then
        # One line per delta, prefixed +/-/~. Downstream can grep by prefix.
        if [ -n "$added_keys" ]; then
            while IFS= read -r key; do
                [ -z "$key" ] && continue
                local c
                c="$(lookup "$key" "$current_syms")"
                # payload after `KIND name|` — keep the whole thing.
                printf '+ %s\n' "$c"
            done <<<"$added_keys"
        fi
        if [ -n "$removed_keys" ]; then
            while IFS= read -r key; do
                [ -z "$key" ] && continue
                local a
                a="$(lookup "$key" "$anchor_syms")"
                # `printf -- ...` because a leading `-` in the format string is
                # otherwise parsed as a flag by bash-builtin printf.
                printf -- '- %s\n' "$a"
            done <<<"$removed_keys"
        fi
        if [ -n "$changed_rows" ]; then
            # changed_rows encodes 4-line groups (key, old, new, ---).
            local key old new
            key=""; old=""; new=""
            while IFS= read -r ln; do
                if [ "$ln" = "---" ]; then
                    printf '~ %s\n' "$new"
                    printf '~-%s\n' "$old"
                    key=""; old=""; new=""
                    continue
                fi
                if [ -z "$key" ]; then key="$ln"; continue; fi
                if [ -z "$old" ]; then old="$ln"; continue; fi
                if [ -z "$new" ]; then new="$ln"; continue; fi
            done <<<"$changed_rows"
        fi
        return 0
    fi

    # Human-readable (default).
    echo "ADDED    $added_count"
    if [ -n "$added_keys" ]; then
        while IFS= read -r key; do
            [ -z "$key" ] && continue
            local c
            c="$(lookup "$key" "$current_syms")"
            # Strip the leading `KIND name|` so we show `KIND name :: <proto>`.
            local kind name proto
            kind="$(echo "$key" | awk '{print $1}')"
            name="$(echo "$key" | awk '{print $2}')"
            proto="$(echo "$c" | awk -F'|' '{ for (i=2; i<=NF; i++) { if (i>2) printf "|"; printf "%s", $i } printf "\n" }')"
            printf '  %s %s :: %s\n' "$kind" "$name" "$proto"
        done <<<"$added_keys"
    fi

    echo "REMOVED  $removed_count"
    if [ -n "$removed_keys" ]; then
        while IFS= read -r key; do
            [ -z "$key" ] && continue
            local a
            a="$(lookup "$key" "$anchor_syms")"
            local kind name proto
            kind="$(echo "$key" | awk '{print $1}')"
            name="$(echo "$key" | awk '{print $2}')"
            proto="$(echo "$a" | awk -F'|' '{ for (i=2; i<=NF; i++) { if (i>2) printf "|"; printf "%s", $i } printf "\n" }')"
            printf '  %s %s :: %s\n' "$kind" "$name" "$proto"
        done <<<"$removed_keys"
    fi

    echo "CHANGED  $changed_count"
    if [ -n "$changed_rows" ]; then
        local key old new
        key=""; old=""; new=""
        while IFS= read -r ln; do
            if [ "$ln" = "---" ]; then
                local kind name old_proto new_proto
                kind="$(echo "$key" | awk '{print $1}')"
                name="$(echo "$key" | awk '{print $2}')"
                old_proto="$(echo "$old" | awk -F'|' '{ for (i=2; i<=NF; i++) { if (i>2) printf "|"; printf "%s", $i } printf "\n" }')"
                new_proto="$(echo "$new" | awk -F'|' '{ for (i=2; i<=NF; i++) { if (i>2) printf "|"; printf "%s", $i } printf "\n" }')"
                printf '  %s %s\n' "$kind" "$name"
                printf '    -    %s\n' "$old_proto"
                printf '    +    %s\n' "$new_proto"
                key=""; old=""; new=""
                continue
            fi
            if [ -z "$key" ]; then key="$ln"; continue; fi
            if [ -z "$old" ]; then old="$ln"; continue; fi
            if [ -z "$new" ]; then new="$ln"; continue; fi
        done <<<"$changed_rows"
    fi

    echo "UNCHANGED $unchanged_count"
    return 0
}

# ------------------------------------------------------------- self-test ---
# self_test — exercise the classifier with a synthetic anchor + current
# pair so a future change to the extractor or classifier can be caught
# without touching the real header / anchors. Verifies that:
#   1. an added FUNC is reported under ADDED
#   2. a removed TYPEDEF is reported under REMOVED
#   3. a signature-changed FUNC is reported under CHANGED (once, not as
#      add+remove)
#   4. an unchanged FUNC is counted under UNCHANGED
self_test() {
    local tmp_anchor tmp_current
    tmp_anchor="$(mktemp -t vokra-abi-diff-anchor.XXXXXX)"
    tmp_current="$(mktemp -t vokra-abi-diff-current.XXXXXX)"
    trap 'rm -f "$tmp_anchor" "$tmp_current"' RETURN

    cat >"$tmp_anchor" <<'EOF'
# anchor snapshot
FUNC vokra_kept_func|enum vokra_status_t vokra_kept_func(int32_t x)
FUNC vokra_changed_func|enum vokra_status_t vokra_changed_func(int32_t x)
FUNC vokra_removed_func|void vokra_removed_func(void)
TYPEDEF vokra_kept_t|typedef struct vokra_kept_t vokra_kept_t
TYPEDEF vokra_removed_t|typedef struct vokra_removed_t vokra_removed_t
EOF

    cat >"$tmp_current" <<'EOF'
# current snapshot
FUNC vokra_kept_func|enum vokra_status_t vokra_kept_func(int32_t x)
FUNC vokra_changed_func|enum vokra_status_t vokra_changed_func(int32_t x, int32_t y)
FUNC vokra_added_func|void vokra_added_func(void)
TYPEDEF vokra_kept_t|typedef struct vokra_kept_t vokra_kept_t
TYPEDEF vokra_added_t|typedef struct vokra_added_t vokra_added_t
EOF

    local anchor_syms current_syms
    anchor_syms="$(read_anchor "$tmp_anchor")"
    current_syms="$(read_anchor "$tmp_current")"

    local report
    report="$(classify "$anchor_syms" "$current_syms" text)"

    local ok=1
    _st_check() {
        local pat="$1" label="$2"
        if ! printf '%s\n' "$report" | grep -Eq "$pat"; then
            echo "self-test FAILED: $label" >&2
            echo "  pattern: $pat" >&2
            ok=0
        fi
    }

    # 1. ADDED must list both the FUNC and TYPEDEF additions.
    _st_check '^ADDED    2$'                                    'ADDED count == 2'
    _st_check '^  FUNC vokra_added_func :: '                    'ADDED includes vokra_added_func'
    _st_check '^  TYPEDEF vokra_added_t :: '                    'ADDED includes vokra_added_t'

    # 2. REMOVED must list the FUNC and TYPEDEF removals.
    _st_check '^REMOVED  2$'                                    'REMOVED count == 2'
    _st_check '^  FUNC vokra_removed_func :: '                  'REMOVED includes vokra_removed_func'
    _st_check '^  TYPEDEF vokra_removed_t :: '                  'REMOVED includes vokra_removed_t'

    # 3. CHANGED must have exactly one row, both old and new prototypes.
    _st_check '^CHANGED  1$'                                    'CHANGED count == 1'
    _st_check '^  FUNC vokra_changed_func$'                     'CHANGED includes vokra_changed_func'
    _st_check '^    -    enum vokra_status_t vokra_changed_func\(int32_t x\)$'          'CHANGED shows old proto'
    _st_check '^    \+    enum vokra_status_t vokra_changed_func\(int32_t x, int32_t y\)$' 'CHANGED shows new proto'

    # 4. UNCHANGED must count the two kept symbols.
    _st_check '^UNCHANGED 2$'                                   'UNCHANGED count == 2'

    # 5. Machine format one-liners must appear too.
    local mreport
    mreport="$(classify "$anchor_syms" "$current_syms" machine)"
    if ! printf '%s\n' "$mreport" | grep -q '^\+ FUNC vokra_added_func|'; then
        echo "self-test FAILED: machine ADDED"; ok=0
    fi
    if ! printf '%s\n' "$mreport" | grep -q '^- TYPEDEF vokra_removed_t|'; then
        echo "self-test FAILED: machine REMOVED"; ok=0
    fi
    if ! printf '%s\n' "$mreport" | grep -q '^~ FUNC vokra_changed_func|.*int32_t y'; then
        echo "self-test FAILED: machine CHANGED new"; ok=0
    fi
    if ! printf '%s\n' "$mreport" | grep -q '^~-FUNC vokra_changed_func|.*(int32_t x)$'; then
        echo "self-test FAILED: machine CHANGED old"; ok=0
    fi

    # 6. Extractor drift guard: re-run the awk pipeline over the same
    #    synthetic header that check-abi-changelog.sh uses in its own self-
    #    test, and verify we produce the same FUNC / TYPEDEF surface. This
    #    catches any drift between the two extractors (the whole reason we
    #    duplicated the awk in this file rather than sourcing it).
    local tmp_hdr got want
    tmp_hdr="$(mktemp -t vokra-abi-diff-hdr.XXXXXX)"
    cat >"$tmp_hdr" <<'EOF'
/* self-test header (mirrors check-abi-changelog.sh) */
#ifndef VOKRA_TEST_H
#define VOKRA_TEST_H
typedef enum vokra_status_t {
    VOKRA_OK = 0,
    VOKRA_ERROR_IO = 1,
} vokra_status_t;
typedef struct vokra_session_t vokra_session_t;
enum vokra_status_t vokra_asr_transcribe(const struct vokra_session_t *session,
                                         const float *pcm,
                                         size_t num_samples);
void vokra_string_free(char *s);
int unrelated_function(int x);
#endif
EOF
    got="$(extract_symbols "$tmp_hdr")"
    want="$(printf '%s\n' \
        'FUNC vokra_asr_transcribe|enum vokra_status_t vokra_asr_transcribe(const struct vokra_session_t *session, const float *pcm, size_t num_samples)' \
        'FUNC vokra_string_free|void vokra_string_free(char *s)' \
        'TYPEDEF vokra_session_t|typedef struct vokra_session_t vokra_session_t' \
        'TYPEDEF vokra_status_t|typedef enum vokra_status_t { VOKRA_OK = 0, VOKRA_ERROR_IO = 1, } vokra_status_t' \
        | LC_ALL=C sort -u)"
    if [ "$got" != "$want" ]; then
        echo "self-test FAILED: extractor drift vs. check-abi-changelog.sh" >&2
        diff -u <(printf '%s\n' "$want") <(printf '%s\n' "$got") >&2 || true
        ok=0
    fi

    # 7. Format-detection guard: verify is_symbols_format returns TRUE for
    #    a pre-extracted symbols file and FALSE for a raw C header. This
    #    is what routes --anchor m0 (raw header) vs. --anchor v0.9
    #    (symbols) to the right loader.
    if ! is_symbols_format "$tmp_anchor"; then
        echo "self-test FAILED: is_symbols_format missed symbols file" >&2
        ok=0
    fi
    if is_symbols_format "$tmp_hdr"; then
        echo "self-test FAILED: is_symbols_format false-positive on raw header" >&2
        ok=0
    fi

    # 8. load_anchor_symbols must produce IDENTICAL output for the raw
    #    header and a pre-extracted symbols file derived from that same
    #    header. This is the drift guard between the two anchor formats
    #    and is why M3-16-T02 can commit only the raw M0 header
    #    (include/vokra.h.m0-anchor) without also committing a redundant
    #    .symbols file: the extractor is deterministic.
    local raw_via_load extracted
    raw_via_load="$(load_anchor_symbols "$tmp_hdr")"
    extracted="$(extract_symbols "$tmp_hdr")"
    if [ "$raw_via_load" != "$extracted" ]; then
        echo "self-test FAILED: load_anchor_symbols disagrees with extract_symbols on a raw header" >&2
        diff -u <(printf '%s\n' "$extracted") <(printf '%s\n' "$raw_via_load") >&2 || true
        ok=0
    fi

    local tmp_syms
    tmp_syms="$(mktemp -t vokra-abi-diff-syms.XXXXXX)"
    {
        echo "# synthesized from tmp_hdr for the load_anchor_symbols round-trip"
        extract_symbols "$tmp_hdr"
    } >"$tmp_syms"
    local sym_via_load sym_via_read
    sym_via_load="$(load_anchor_symbols "$tmp_syms")"
    sym_via_read="$(read_anchor "$tmp_syms")"
    if [ "$sym_via_load" != "$sym_via_read" ]; then
        echo "self-test FAILED: load_anchor_symbols disagrees with read_anchor on a symbols file" >&2
        diff -u <(printf '%s\n' "$sym_via_read") <(printf '%s\n' "$sym_via_load") >&2 || true
        ok=0
    fi
    if [ "$sym_via_load" != "$raw_via_load" ]; then
        echo "self-test FAILED: symbols-form vs raw-form anchors diverge for the same header" >&2
        diff -u <(printf '%s\n' "$raw_via_load") <(printf '%s\n' "$sym_via_load") >&2 || true
        ok=0
    fi
    rm -f "$tmp_syms" "$tmp_hdr"

    if [ "$ok" -eq 1 ]; then
        echo "abi-diff --self-test: OK"
        return 0
    fi
    return 1
}

# ------------------------------------------------------------------ main ---
anchor_arg="m0"
header=""            # empty -> resolve to DEFAULT_HEADER after arg parsing
format="text"
mode="diff"
regenerate=0

while [ $# -gt 0 ]; do
    case "$1" in
        --anchor)
            if [ $# -lt 2 ]; then
                echo "error: --anchor requires an argument" >&2
                exit 2
            fi
            anchor_arg="$2"
            shift 2
            ;;
        --header)
            if [ $# -lt 2 ]; then
                echo "error: --header requires an argument" >&2
                exit 2
            fi
            header="$2"
            shift 2
            ;;
        --format)
            if [ $# -lt 2 ]; then
                echo "error: --format requires an argument" >&2
                exit 2
            fi
            case "$2" in
                text|machine) format="$2" ;;
                *) echo "error: --format must be 'text' or 'machine'" >&2; exit 2 ;;
            esac
            shift 2
            ;;
        --regenerate)
            regenerate=1
            shift
            ;;
        --self-test)
            mode="self-test"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument '$1'" >&2
            echo "usage: $0 [--anchor <path|m0|v0.9>] [--header <path>] [--regenerate] [--format text|machine] [--self-test | --help]" >&2
            exit 2
            ;;
    esac
done

case "$mode" in
    self-test)
        set +e
        self_test
        exit $?
        ;;
esac

# Resolve anchor label -> path.
case "$anchor_arg" in
    m0)   anchor_path="$ANCHOR_M0" ;;
    v0.9) anchor_path="$ANCHOR_V09" ;;
    *)    anchor_path="$anchor_arg" ;;
esac

if [ ! -f "$anchor_path" ]; then
    echo "error: anchor not found: $anchor_path" >&2
    echo "       known labels: m0 -> $ANCHOR_M0" >&2
    echo "                     v0.9 -> $ANCHOR_V09" >&2
    exit 2
fi

# --regenerate: shell out to scripts/gen-c-abi.sh to write a fresh
# include/vokra.h into a tempfile, then use that tempfile as the
# "current" header for the diff. Mirrors the gen-c-abi.sh --check
# tempfile pattern per M3-16-T02, so an M3-16-T04 aggregator can be
# certain the report reflects the source tree even if the caller
# forgot to run scripts/gen-c-abi.sh first. --header wins if the
# caller passed it explicitly — --regenerate implies "use the freshly
# generated header", and combining the two would be ambiguous.
tmp_regen_header=""
if [ "$regenerate" -eq 1 ]; then
    if [ -n "$header" ]; then
        echo "error: --regenerate and --header are mutually exclusive" >&2
        echo "       (drop --header to regenerate, or drop --regenerate to point at a fixed file)" >&2
        exit 2
    fi
    if [ ! -x "$GEN_C_ABI" ]; then
        echo "error: --regenerate needs $GEN_C_ABI to be executable" >&2
        exit 2
    fi
    if ! command -v cbindgen >/dev/null 2>&1 \
        && [ ! -x "${CARGO_HOME:-$HOME/.cargo}/bin/cbindgen" ]; then
        echo "error: --regenerate requires cbindgen on PATH (or in \$CARGO_HOME/bin)" >&2
        echo "       install it with: cargo install cbindgen" >&2
        exit 2
    fi
    tmp_regen_header="$(mktemp -t vokra-abi-diff-regen.XXXXXX)"
    # shellcheck disable=SC2064
    trap "rm -f '$tmp_regen_header'" EXIT
    # Reuse gen-c-abi.sh's exact cbindgen invocation via HEADER
    # override -- but that script always writes to a fixed path, so
    # invoke cbindgen ourselves with the same config/crate to a
    # tempfile. This is the same pattern gen-c-abi.sh --check uses
    # internally (see scripts/gen-c-abi.sh line 45).
    if command -v cbindgen >/dev/null 2>&1; then
        CBINDGEN="cbindgen"
    else
        CBINDGEN="${CARGO_HOME:-$HOME/.cargo}/bin/cbindgen"
    fi
    if ! ( cd "$ROOT" && \
            "$CBINDGEN" \
                --config "$ROOT/crates/vokra-capi/cbindgen.toml" \
                --crate vokra-capi \
                --output "$tmp_regen_header" \
                --quiet ); then
        echo "error: --regenerate failed while running cbindgen" >&2
        exit 2
    fi
    header="$tmp_regen_header"
elif [ -z "$header" ]; then
    header="$DEFAULT_HEADER"
fi

if [ ! -f "$header" ]; then
    echo "error: header not found: $header" >&2
    exit 2
fi

anchor_syms="$(load_anchor_symbols "$anchor_path")"
current_syms="$(extract_symbols "$header")"

if [ "$format" = "text" ]; then
    echo "abi-diff: current header vs. anchor"
    echo "  header : $header"
    echo "  anchor : $anchor_path"
    if [ "$regenerate" -eq 1 ]; then
        echo "  note   : header was regenerated by cbindgen into a tempfile (--regenerate)"
    fi
    echo ""
fi

classify "$anchor_syms" "$current_syms" "$format"
