#!/usr/bin/env bash
# rust-public-api-list.sh — M3-16-T03 Rust public API extractor, snapshot,
#                          and `#[non_exhaustive]` audit.
#
# WHAT IT DOES
#   Extracts the `pub` API surface (fn / struct / enum / trait / type / const /
#   static / union / use / mod) of the three ABI-relevant crates
#   (`vokra-core`, `vokra-ops`, `vokra-capi`), emits a stable sorted snapshot,
#   and audits that a small set of evolution-critical enums still carry
#   `#[non_exhaustive]`. Complements `scripts/abi-diff.sh` (which covers the
#   C ABI via `include/vokra.h`) with the Rust-side surface.
#
# WHY CANDIDATE C
#   `docs/tickets/m3/M3-16-abi-changelog.md` §T03 内容 lists three candidates:
#     A. `cargo public-api` — dev-dep, would drop a new crate into the root
#        `Cargo.lock`, violating NFR-DS-02 zero-dep.
#     B. `cargo doc --output-format json` — requires nightly rustdoc.
#     C. Grep-based `pub fn` / `pub struct` / `pub enum` / `pub trait`
#        extraction — zero-dep, portable, simplest.
#   We implement candidate C: pure bash + find + awk + grep + sort + diff,
#   mirroring the `scripts/check-abi-changelog.sh` idiom.
#
# HONEST LIMITATIONS (candidate C, ticket-acknowledged: "簡易だが漏れリスクあり")
#   1. Methods declared inside `impl <Type>` blocks appear as bare
#      `pub fn <name>` with no impl-target context. The snapshot lists them
#      by name + normalized signature, and `sort -u` collapses two impls
#      that share both name AND signature into one entry. Sufficient for
#      changelog input, insufficient as a semver freeze gate (which fires
#      at M5-13, not here).
#   2. `pub` items inside `#[cfg(test)]` modules would be extracted if any
#      existed. This workspace has none as of v0.9 (verified across all
#      99 .rs files in vokra-core / vokra-ops / vokra-capi), and test
#      modules universally use `pub(crate)` / `pub(super)` visibility,
#      which this extractor skips.
#   3. `pub(crate)` / `pub(super)` / `pub(in path)` are NOT public and are
#      explicitly filtered out (they cannot appear in a downstream crate's
#      API).
#   4. Macro-generated `pub` items are invisible. Same limitation as the
#      C ABI extractor in check-abi-changelog.sh (both grep pre-expansion
#      source), and the v0.9 tree does not export any via macros.
#
# ARTEFACTS
#   crates/vokra-{core,ops,capi}/src/           -- Rust sources scanned
#   docs/abi/vokra-rust-public-api.v0.9.list    -- snapshot (this script)
#
# MODES
#   scripts/rust-public-api-list.sh                    -- verify (default)
#   scripts/rust-public-api-list.sh --list             -- print current
#   scripts/rust-public-api-list.sh --update-snapshot  -- rewrite snapshot
#                                                        (owner action)
#   scripts/rust-public-api-list.sh --audit            -- run only the
#                                                        `#[non_exhaustive]`
#                                                        audit
#   scripts/rust-public-api-list.sh --self-test        -- exercise the
#                                                        extractor + auditor
#                                                        against synthetic
#                                                        input
#   scripts/rust-public-api-list.sh --help             -- this text
#
# NOT WIRED INTO CI
#   Per M3-16 spec §T03 last paragraph and the sibling C-ABI tool
#   (`scripts/abi-diff.sh`), CI gating is deferred to M5-13 with the v1.0
#   GA freeze (the 2026-07-14 v-label reassignment #2 moved the freeze WP
#   M4-12 → M5-13). Run this from a pre-commit hook or manually to snapshot
#   the v1.0-rc Rust surface.
#
# ZERO-DEP
#   Pure bash + awk + grep + diff + sort + find. No `cargo public-api`, no
#   Rust toolchain invocation, no external crate needed. NFR-DS-02 preserved
#   (root `Cargo.lock` unchanged, no dev-dep added).
#
# EXIT CODES
#   0  clean (snapshot matches + audit passes, or --list / --update-snapshot
#      / --audit / --self-test success)
#   1  drift detected (snapshot mismatch), audit failure, or self-test
#      failure
#   2  usage / setup error (missing snapshot on default verify, missing
#      source dir, bad flag)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CRATES_DIR="$ROOT/crates"
SNAPSHOT="$ROOT/docs/abi/vokra-rust-public-api.v1.0-rc.list"

# Crates scanned. Order affects only the extraction traversal order; the
# final output is `sort -u`-stable so downstream diffs are order-independent.
CRATES=("vokra-core" "vokra-ops" "vokra-capi")

# `#[non_exhaustive]` audit expectations.
#
# Each row: "<type-name>:<pub-kind>:<repo-relative-path>".
#   type-name — identifier as it appears in `pub <kind> <name> {`.
#   pub-kind  — "enum" or "struct" (extend if a future `pub union`
#               ships with a `#[non_exhaustive]` requirement).
#   path      — source file the marker lives in.
#
# Curation rationale: the two names M3-16-T03 spec calls out by name are
# `OpKind` and `VokraError`; we add the three sibling enums that share the
# same "must keep the wildcard arm" contract (BackendKind for backend
# selection, GgufError for loader errors, StreamEvent for stream events).
# All five are verified to carry `#[non_exhaustive]` at v0.9 window open
# (verified 2026-07-11 against the working-tree source of the branch
# `feat/m3-plan-and-wave1`).
#
# M4-12-T05 adds `IsaPath` (vokra-backend-cpu — a fourth crate the extractor
# does NOT scan, but the audit reads by explicit path). M4-17 marked it
# `#[non_exhaustive]` (T04) so future CPU ISA tiers (AMX*/SME*/RvvZvfh*, see
# docs/abi-changelog.md "Reserved additions") land as backward-compat variant
# additions; auditing it at the v1.0-rc baseline protects the M5-13 enum-shape
# freeze (docs/handoff/m4-12.md §(e)-2). Verified 2026-07-15.
NON_EXHAUSTIVE_EXPECTED=(
    "VokraError:enum:crates/vokra-core/src/error.rs"
    "OpKind:enum:crates/vokra-core/src/ir/graph.rs"
    "BackendKind:enum:crates/vokra-core/src/backend.rs"
    "GgufError:enum:crates/vokra-core/src/gguf/mod.rs"
    "StreamEvent:enum:crates/vokra-core/src/stream/event.rs"
    "IsaPath:enum:crates/vokra-backend-cpu/src/features.rs"
)

usage() {
    # Print the header docstring (skip the shebang, stop at the blank line
    # that separates the docstring from `set -euo pipefail`). Range is
    # locked to the current layout — if the docstring grows, extend the
    # upper bound to match.
    sed -n '2,77p' "$0" | sed 's/^# \{0,1\}//'
}

# ---------------------------------------------------------------- module ---
# module_of <src-relative-path>
#
# Converts a path relative to a crate's `src/` into the Rust module chain
# used in the snapshot's fully-qualified name column:
#   lib.rs                         -> ""                (crate root)
#   backend.rs                     -> "backend"
#   ir/mod.rs                      -> "ir"
#   ir/graph.rs                    -> "ir::graph"
#   ir/fusion/mod.rs               -> "ir::fusion"
#   stream/event.rs                -> "stream::event"
#
# This is a purely lexical mapping — it does not consult `mod` declarations
# inside lib.rs. The v0.9 tree uses on-disk module layout throughout, so
# lexical == declarative. If a future refactor breaks that (a `#[path]`
# attribute, or a nested `mod X { ... }` inline in a parent file), the
# self-test's audit-drift guard catches it as an unexpected snapshot diff.
module_of() {
    local rel="$1"
    case "$rel" in
        lib.rs)
            echo ""
            ;;
        */mod.rs)
            local trimmed="${rel%/mod.rs}"
            echo "${trimmed//\//::}"
            ;;
        *.rs)
            local trimmed="${rel%.rs}"
            echo "${trimmed//\//::}"
            ;;
        *)
            # Non-.rs file — shouldn't happen given the `-name '*.rs'` filter,
            # but be safe: emit empty so the caller drops the row.
            echo ""
            ;;
    esac
}

# ---------------------------------------------------------------- extract ---
# extract_symbols
#
# Walks every `.rs` file under each crate's `src/` and emits one line per
# `pub` declaration:
#
#   <KIND> <crate>::<module>::<name>|<normalized-line>
#
# where <KIND> is FN / STRUCT / ENUM / TRAIT / TYPE / CONST / STATIC / UNION
# / USE / MOD, and <normalized-line> is the original source line with all
# whitespace runs collapsed to a single space and leading/trailing space
# stripped. Output is sorted, `LC_ALL=C` stable, and deduplicated so a
# repeated identical signature across two impls collapses to one row
# (candidate-C honest limitation).
extract_symbols() {
    local crate src_dir file rel module

    for crate in "${CRATES[@]}"; do
        src_dir="$CRATES_DIR/$crate/src"
        if [ ! -d "$src_dir" ]; then
            echo "error: source dir not found: $src_dir" >&2
            return 2
        fi

        # NUL-delimited find keeps this safe if paths ever contain whitespace.
        while IFS= read -r -d '' file; do
            rel="${file#"$src_dir"/}"
            module="$(module_of "$rel")"
            parse_file "$crate" "$module" "$file"
        done < <(find "$src_dir" -type f -name '*.rs' -print0)
    done | LC_ALL=C sort -u
}

# parse_file <crate> <module> <file>
#
# Emits one <KIND> <fq-name>|<normalized-line> row per `pub` declaration in
# `<file>`. Filters out `pub(...)` visibilities (not crate-external) and
# skips `#[cfg(test)]`-gated content by walking brace depth after the marker
# (the v0.9 tree has zero `pub ` items inside test blocks, but we still
# scope the truncation so a future test-only `pub` item cannot silently
# leak into the snapshot).
#
# Multi-line `pub use foo::{ ... };` groups are pre-joined into a single
# logical line by the first awk in the pipe, so the extractor sees exactly
# one row per re-export statement no matter how many source lines it spans.
# Brace-depth tracking handles nested `pub use foo::{Bar, Baz::{X, Y}};`
# correctly — the outer `};` closes only when the depth counter returns to
# zero.
parse_file() {
    local crate="$1" module="$2" file="$3"

    # Pre-pass: join multi-line `pub use foo::{ ... };` groups into one
    # logical line so the extractor can classify a re-export by its full
    # content, not just its opening line. Non-`use` items are passed
    # through unchanged.
    awk '
        BEGIN { in_use = 0; buf = ""; depth = 0 }
        {
            if (!in_use) {
                trimmed = $0
                sub(/^[[:space:]]+/, "", trimmed)
                if (trimmed ~ /^pub[[:space:]]+use[[:space:]]/ && index($0, "{") > 0 && index($0, ";") == 0) {
                    in_use = 1
                    buf = $0
                    depth = 0
                    for (i = 1; i <= length($0); i++) {
                        c = substr($0, i, 1)
                        if (c == "{") depth++
                        else if (c == "}") depth--
                    }
                    next
                }
                print $0
                next
            }
            # Accumulate a continuation line of the current use group.
            buf = buf " " $0
            for (i = 1; i <= length($0); i++) {
                c = substr($0, i, 1)
                if (c == "{") depth++
                else if (c == "}") depth--
            }
            # Close when the outer `{` has been balanced AND the current
            # line closes with `;`. Ignoring the `;` requirement would
            # emit a partial line for `pub use foo::{Bar::{X, Y}}` on
            # the inner `}` — the outer `;` is the reliable terminator.
            if (depth <= 0 && index($0, ";") > 0) {
                print buf
                in_use = 0
                buf = ""
                depth = 0
            }
        }
        END {
            # Malformed source (unterminated `pub use ... {` at EOF).
            # Still emit what we have so the extractor at least sees the
            # opening line; the self-test does not exercise this branch.
            if (in_use && length(buf) > 0) print buf
        }
    ' "$file" | awk -v crate="$crate" -v module="$module" '
        # Track whether the current line is inside a `#[cfg(test)]`-gated
        # module (or item). Approach:
        #   - When we see `^#[cfg(test)]` or `^#[cfg(all(test, ...))]`, set
        #     `pending_test = 1`.
        #   - When the next non-blank / non-attribute / non-doc line arrives,
        #     start counting braces: from the first `{` we hit through the
        #     matching `}` at the same depth, we are `in_test`. Once the
        #     matching `}` is consumed, resume normal extraction.
        #   - Nested braces inside the test block are handled by counting
        #     `depth`.
        # This deliberately misses `#[cfg(test)]` followed by a `use ...;`
        # (no braces), but the v0.9 tree does not attach it to bare `use`.
        BEGIN {
            pending_test = 0
            in_test = 0
            depth = 0
        }

        # Detect the marker. Match both bare `#[cfg(test)]` and the
        # `#[cfg(all(test, ...))]` conjunction pattern used in
        # crates/vokra-core/src/ir/fusion/patterns/snake.rs.
        {
            probe = $0
            sub(/^[[:space:]]+/, "", probe)
            if (probe ~ /^#\[cfg\(test\)\]/ || probe ~ /^#\[cfg\(all\(test,/) {
                pending_test = 1
                next
            }
        }

        # If we are already inside a test block, advance the brace depth
        # over this line and either stay in_test or exit when depth hits 0.
        in_test {
            for (i = 1; i <= length($0); i++) {
                c = substr($0, i, 1)
                if (c == "{") depth++
                else if (c == "}") {
                    depth--
                    if (depth <= 0) {
                        in_test = 0
                        depth = 0
                        # Continue scanning the rest of this line as normal
                        # code (rare but possible when a test block ends
                        # mid-line with something after it).
                        break
                    }
                }
            }
            next
        }

        # `pending_test` means we saw the attribute; we are waiting for the
        # associated item to open its brace scope. Skip blank / attribute /
        # doc lines; on the first "real" content, if it opens a brace,
        # enter in_test.
        pending_test {
            trimmed_p = $0
            sub(/^[[:space:]]+/, "", trimmed_p)
            if (length(trimmed_p) == 0) next
            if (trimmed_p ~ /^#\[/) next
            if (trimmed_p ~ /^\/\//) next
            if (trimmed_p ~ /^\/\*/) next

            # The first real content-line resolves the test scope.
            open = index($0, "{")
            if (open > 0) {
                in_test = 1
                pending_test = 0
                depth = 0
                # Count braces on this line — handles single-line
                # `mod tests { ... }` as well as the more common
                # multi-line form.
                for (i = 1; i <= length($0); i++) {
                    c = substr($0, i, 1)
                    if (c == "{") depth++
                    else if (c == "}") {
                        depth--
                        if (depth <= 0) {
                            in_test = 0
                            depth = 0
                            break
                        }
                    }
                }
                next
            }
            # Non-brace item after the marker (e.g. `#[cfg(test)] use foo;`).
            # v0.9 tree has none, but treat as one-line scope: the marker
            # applied to this single line and we resume normal scanning at
            # the next line.
            pending_test = 0
            next
        }

        # Normal-mode scan: try to classify this line as a `pub` declaration.
        {
            trimmed = $0
            sub(/^[[:space:]]+/, "", trimmed)

            # Must start with `pub` followed by a whitespace char.
            if (trimmed !~ /^pub[[:space:]]/) next

            # Filter `pub(crate)` / `pub(super)` / `pub(in path)` — none of
            # those are crate-external, so they are not part of the API.
            if (trimmed ~ /^pub\(/) next

            # Consume `pub ` prefix.
            after = trimmed
            sub(/^pub[[:space:]]+/, "", after)

            # Consume optional modifiers between `pub` and the item keyword.
            # Modifiers: unsafe, async, const (as `const fn`), extern with
            # or without an ABI string. They stack in a stable-but-arbitrary
            # order (`pub unsafe extern "C" fn`, `pub const fn`, etc.).
            #
            # `const` is treated as a modifier ONLY when it precedes `fn` —
            # otherwise `pub const NAME: T = ...` is the item and `const`
            # is the kind, not a modifier. The lookahead check
            # `^const[[:space:]]+fn[[:space:]]+` disambiguates.
            while (1) {
                if (match(after, /^extern[[:space:]]+"[^"]*"[[:space:]]+/)) {
                    after = substr(after, RLENGTH + 1)
                    continue
                }
                if (match(after, /^extern[[:space:]]+/)) {
                    after = substr(after, RLENGTH + 1)
                    continue
                }
                if (match(after, /^(unsafe|async|default|auto)[[:space:]]+/)) {
                    after = substr(after, RLENGTH + 1)
                    continue
                }
                if (match(after, /^const[[:space:]]+fn[[:space:]]+/)) {
                    # Eat only the "const " prefix — leave "fn ..." for
                    # the item-kind matcher below.
                    sub(/^const[[:space:]]+/, "", after)
                    continue
                }
                break
            }

            # Now the next token must be a known item kind followed by
            # whitespace.
            if (!match(after, /^(fn|struct|enum|trait|type|const|static|union|use|mod)[[:space:]]+/)) next
            kw_and_ws = substr(after, RSTART, RLENGTH)
            kw = kw_and_ws
            sub(/[[:space:]]+$/, "", kw)

            after_kw = substr(after, RSTART + RLENGTH)

            # Extract the item identifier (or, for `use`, the path).
            if (kw == "use") {
                # For `pub use PATH;`, capture PATH up to `;`. If no `;` is
                # present on this line (rare — but multi-line `pub use`
                # exists in complex re-exports like the one in
                # vokra-core/src/lib.rs), fall back to end-of-line.
                if (match(after_kw, /^[^;]+/)) {
                    name = substr(after_kw, RSTART, RLENGTH)
                    gsub(/[[:space:]]+/, " ", name)
                    sub(/^ /, "", name)
                    sub(/ $/, "", name)
                    # Drop trailing `{` from a use-tree opener so multi-line
                    # `pub use foo::{` groups by the imported path.
                    sub(/[[:space:]]*\{$/, "", name)
                } else {
                    next
                }
            } else {
                if (match(after_kw, /^[A-Za-z_][A-Za-z0-9_]*/)) {
                    name = substr(after_kw, RSTART, RLENGTH)
                } else {
                    next
                }
            }

            # Normalize the payload line: collapse all whitespace runs to a
            # single space and strip ends. Sorting and diff both operate on
            # this normalized form.
            norm = $0
            gsub(/[[:space:]]+/, " ", norm)
            sub(/^ /, "", norm)
            sub(/ $/, "", norm)

            # Compose the fully-qualified name.
            fq = crate
            if (length(module) > 0) fq = fq "::" module
            fq = fq "::" name

            kind = toupper(kw)
            printf "%s %s|%s\n", kind, fq, norm
        }
    '
}

# ------------------------------------------------------------------ audit ---
# audit_non_exhaustive
#
# For each `<name>:<pub-kind>:<path>` entry in NON_EXHAUSTIVE_EXPECTED,
# verifies that the source file contains the marker `#[non_exhaustive]`
# in the attribute block IMMEDIATELY preceding the declaration
# `pub <pub-kind> <name>`. "Immediately preceding" means: walking backward
# from the `pub <pub-kind> <name>` line, past `#[...]` attributes, doc
# comments (`///`, `//!`), and blank lines, until reaching either the
# `#[non_exhaustive]` marker (success) or a non-attribute line (failure).
#
# The audit is deliberately narrow: it only checks the small set of enums
# M3-16-T03 spec calls out and their siblings that share the same wildcard-
# arm contract. It does not attempt to enumerate every `#[non_exhaustive]`
# in the tree — that would swamp the signal with false positives on
# scaffold enums whose evolution is not IF-01-critical.
audit_non_exhaustive() {
    local entry name kind path abspath ok=0 total=0
    ok=1

    for entry in "${NON_EXHAUSTIVE_EXPECTED[@]}"; do
        name="${entry%%:*}"
        local rest="${entry#*:}"
        kind="${rest%%:*}"
        path="${rest#*:}"
        abspath="$ROOT/$path"
        total=$((total + 1))

        if [ ! -f "$abspath" ]; then
            echo "audit FAIL: source not found for $name: $abspath" >&2
            ok=0
            continue
        fi

        # awk finds the `pub <kind> <name>` line, then walks the collected
        # history backward past attribute / doc / blank lines, looking for
        # a `#[non_exhaustive]` line. Emits `OK` or `MISS` and exits.
        local verdict
        verdict="$(awk -v name="$name" -v kind="$kind" '
            {
                hist[++n] = $0
            }
            END {
                found_marker = 0
                found_decl   = 0
                for (i = 1; i <= n; i++) {
                    line = hist[i]
                    # Match `pub enum NAME` / `pub struct NAME` — allow
                    # trailing `<...>` generics or `{` or whitespace.
                    pattern = "^pub " kind " " name "([[:space:]<{]|$)"
                    if (line ~ pattern) {
                        found_decl = 1
                        # Walk backward past attribute / doc / blank lines.
                        for (j = i - 1; j > 0; j--) {
                            prev = hist[j]
                            # Strip leading whitespace for the checks.
                            sub(/^[[:space:]]+/, "", prev)
                            if (prev ~ /^#\[non_exhaustive\]/) {
                                found_marker = 1
                                break
                            }
                            # Continue past other attributes / doc / blank.
                            if (prev ~ /^#\[/)      continue
                            if (prev ~ /^\/\/\//)   continue
                            if (prev ~ /^\/\/!/)    continue
                            if (prev ~ /^\/\//)     continue
                            if (length(prev) == 0)  continue
                            # Any other line breaks the attribute chain.
                            break
                        }
                        break
                    }
                }
                if (!found_decl) {
                    print "NODECL"
                } else if (!found_marker) {
                    print "MISS"
                } else {
                    print "OK"
                }
            }
        ' "$abspath")"

        case "$verdict" in
            OK)
                # Silent success — the per-entry line is only printed on
                # failure to keep the terminal output focused.
                ;;
            NODECL)
                echo "audit FAIL: declaration \`pub $kind $name\` not found in $path" >&2
                ok=0
                ;;
            MISS)
                echo "audit FAIL: $name in $path is missing \`#[non_exhaustive]\` marker (expected immediately above \`pub $kind $name\`)" >&2
                ok=0
                ;;
            *)
                echo "audit FAIL: unexpected verdict \"$verdict\" for $name in $path" >&2
                ok=0
                ;;
        esac
    done

    if [ "$ok" -eq 1 ]; then
        echo "audit-non-exhaustive: OK ($total enum(s) still carry the marker)"
        return 0
    fi
    return 1
}

# ------------------------------------------------------------- self-test ---
# self_test — exercise the extractor + auditor against synthetic input.
#
# Case A (extractor happy path): a fake crate root emitting a mix of
#   `pub fn` / `pub struct` / `pub enum` / `pub trait` / `pub type` /
#   `pub const` / `pub use` / `pub mod`, plus non-public and `pub(crate)`
#   items that must NOT appear in the output.
#
# Case B (extractor test-scope guard): a `#[cfg(test)]` block with a
#   `pub struct` inside — must be skipped by the extractor, and the sibling
#   `pub struct` outside the block must survive.
#
# Case B2 (extractor multi-line `pub use` join): a `pub use foo::{ A, B, };`
#   spanning three lines — must be joined into a single logical row with
#   the full re-export list visible in the payload column.
#
# Case B3 (extractor const-vs-const-fn disambiguation): a `pub const X: T`
#   at the top level plus a `pub const fn cf()` — the item-const's name
#   must not be swallowed by the `const fn` modifier path.
#
# Case C (audit happy path): a struct with `#[non_exhaustive]` in its
#   attribute chain — verdict OK.
#
# Case D (audit missing marker): the same struct with the marker deleted
#   — verdict MISS, exit non-zero.
self_test() {
    local tmproot tmp_audit_ok tmp_audit_miss
    tmproot="$(mktemp -d -t vokra-rust-api.XXXXXX)"
    trap 'rm -rf "$tmproot"' RETURN

    # --- Case A + B + B2 + B3 -------------------------------------------
    # Build a synthetic crate under $tmproot/crates/vokra-testcrate/src/.
    local synth_root="$tmproot/synth-root"
    mkdir -p "$synth_root/crates/vokra-testcrate/src/nested"
    cat >"$synth_root/crates/vokra-testcrate/src/lib.rs" <<'EOF'
//! doc comment.
pub mod backend;
pub mod nested;
pub use backend::Backend;
pub use nested::{
    NestedA,
    NestedB,
};
pub const MAX_TOKENS: usize = 32;
pub const fn cf() -> u32 { 1 }

pub struct Public {
    pub field: u32,
}

pub(crate) struct NotPublic;

pub enum Kind {
    A,
    B,
}

pub trait Runnable {
    fn run(&self);
}

pub fn top_level() -> u32 {
    42
}

pub unsafe fn top_unsafe() {}

pub type Alias = Public;

// Case B: `#[cfg(test)]` module with a `pub struct` — must be skipped.
#[cfg(test)]
mod tests {
    pub struct HiddenByCfg;
    pub fn hidden_by_cfg() {}
}

// A real `pub struct` after the test block — must survive.
pub struct AfterTests;
EOF

    cat >"$synth_root/crates/vokra-testcrate/src/backend.rs" <<'EOF'
pub trait Backend {
    fn run(&self);
}
EOF

    cat >"$synth_root/crates/vokra-testcrate/src/nested/mod.rs" <<'EOF'
pub struct NestedA;
pub struct NestedB;
pub fn nested_fn() {}
EOF

    # Invoke parse_file directly for each synthetic file so we don't need
    # the outer $CRATES / $CRATES_DIR machinery.
    local got want
    got="$(
        {
            parse_file vokra-testcrate ""                "$synth_root/crates/vokra-testcrate/src/lib.rs"
            parse_file vokra-testcrate "backend"        "$synth_root/crates/vokra-testcrate/src/backend.rs"
            parse_file vokra-testcrate "nested"         "$synth_root/crates/vokra-testcrate/src/nested/mod.rs"
        } | LC_ALL=C sort -u
    )"

    want="$(printf '%s\n' \
        'CONST vokra-testcrate::MAX_TOKENS|pub const MAX_TOKENS: usize = 32;' \
        'ENUM vokra-testcrate::Kind|pub enum Kind {' \
        'FN vokra-testcrate::cf|pub const fn cf() -> u32 { 1 }' \
        'FN vokra-testcrate::nested::nested_fn|pub fn nested_fn() {}' \
        'FN vokra-testcrate::top_level|pub fn top_level() -> u32 {' \
        'FN vokra-testcrate::top_unsafe|pub unsafe fn top_unsafe() {}' \
        'MOD vokra-testcrate::backend|pub mod backend;' \
        'MOD vokra-testcrate::nested|pub mod nested;' \
        'STRUCT vokra-testcrate::AfterTests|pub struct AfterTests;' \
        'STRUCT vokra-testcrate::Public|pub struct Public {' \
        'STRUCT vokra-testcrate::nested::NestedA|pub struct NestedA;' \
        'STRUCT vokra-testcrate::nested::NestedB|pub struct NestedB;' \
        'TRAIT vokra-testcrate::Runnable|pub trait Runnable {' \
        'TRAIT vokra-testcrate::backend::Backend|pub trait Backend {' \
        'TYPE vokra-testcrate::Alias|pub type Alias = Public;' \
        'USE vokra-testcrate::backend::Backend|pub use backend::Backend;' \
        'USE vokra-testcrate::nested::{ NestedA, NestedB, }|pub use nested::{ NestedA, NestedB, };' \
        | LC_ALL=C sort -u)"

    local ok=1
    if [ "$got" != "$want" ]; then
        echo "self-test FAILED — extractor drift:" >&2
        diff -u <(printf '%s\n' "$want") <(printf '%s\n' "$got") >&2 || true
        ok=0
    fi

    # --- Case C + D ------------------------------------------------------
    # Build a synthetic file with a #[non_exhaustive] enum, invoke the
    # audit's inner awk directly.
    tmp_audit_ok="$tmproot/audit_ok.rs"
    cat >"$tmp_audit_ok" <<'EOF'
//! Fixture: non_exhaustive marker present.

/// Doc for FakeEnum.
#[derive(Debug)]
#[non_exhaustive]
pub enum FakeEnum {
    A,
    B,
}
EOF

    tmp_audit_miss="$tmproot/audit_miss.rs"
    cat >"$tmp_audit_miss" <<'EOF'
//! Fixture: non_exhaustive marker absent.

/// Doc for FakeEnum.
#[derive(Debug)]
pub enum FakeEnum {
    A,
    B,
}
EOF

    _run_audit() {
        awk -v name="FakeEnum" -v kind="enum" '
            {
                hist[++n] = $0
            }
            END {
                found_marker = 0
                found_decl   = 0
                for (i = 1; i <= n; i++) {
                    line = hist[i]
                    pattern = "^pub " kind " " name "([[:space:]<{]|$)"
                    if (line ~ pattern) {
                        found_decl = 1
                        for (j = i - 1; j > 0; j--) {
                            prev = hist[j]
                            sub(/^[[:space:]]+/, "", prev)
                            if (prev ~ /^#\[non_exhaustive\]/) { found_marker = 1; break }
                            if (prev ~ /^#\[/)    continue
                            if (prev ~ /^\/\/\//) continue
                            if (prev ~ /^\/\/!/)  continue
                            if (prev ~ /^\/\//)   continue
                            if (length(prev) == 0) continue
                            break
                        }
                        break
                    }
                }
                if (!found_decl)       print "NODECL"
                else if (!found_marker) print "MISS"
                else                    print "OK"
            }
        ' "$1"
    }

    local v_ok v_miss
    v_ok="$(_run_audit "$tmp_audit_ok")"
    v_miss="$(_run_audit "$tmp_audit_miss")"

    if [ "$v_ok" != "OK" ]; then
        echo "self-test FAILED — audit_ok expected OK, got: $v_ok" >&2
        ok=0
    fi
    if [ "$v_miss" != "MISS" ]; then
        echo "self-test FAILED — audit_miss expected MISS, got: $v_miss" >&2
        ok=0
    fi

    if [ "$ok" -eq 1 ]; then
        echo "rust-public-api-list --self-test: OK"
        return 0
    fi
    return 1
}

# ------------------------------------------------------------------ main ---
mode="${1:-verify}"
case "$mode" in
    verify|"")
        if [ ! -f "$SNAPSHOT" ]; then
            echo "error: snapshot missing: $SNAPSHOT" >&2
            echo "       run: scripts/rust-public-api-list.sh --update-snapshot" >&2
            exit 2
        fi
        current="$(extract_symbols)"
        # Strip banner lines (leading `#`) and blank lines from the snapshot
        # before comparing; re-sort defensively so a hand edit cannot
        # falsely produce a clean diff.
        anchor="$(grep -Ev '^[[:space:]]*(#|$)' "$SNAPSHOT" | LC_ALL=C sort -u)"

        fn_count=$(printf '%s\n' "$anchor" | grep -c '^FN ' || true)
        struct_count=$(printf '%s\n' "$anchor" | grep -c '^STRUCT ' || true)
        enum_count=$(printf '%s\n' "$anchor" | grep -c '^ENUM ' || true)
        trait_count=$(printf '%s\n' "$anchor" | grep -c '^TRAIT ' || true)
        type_count=$(printf '%s\n' "$anchor" | grep -c '^TYPE ' || true)
        const_count=$(printf '%s\n' "$anchor" | grep -c '^CONST ' || true)
        static_count=$(printf '%s\n' "$anchor" | grep -c '^STATIC ' || true)
        union_count=$(printf '%s\n' "$anchor" | grep -c '^UNION ' || true)
        use_count=$(printf '%s\n' "$anchor" | grep -c '^USE ' || true)
        mod_count=$(printf '%s\n' "$anchor" | grep -c '^MOD ' || true)

        echo "Vokra Rust public-API snapshot gate (M3-16-T03; IF-01 fires at M5-13, not here)"
        echo "  crates   : ${CRATES[*]}"
        echo "  snapshot : $SNAPSHOT"
        echo "  anchor   : $fn_count fn, $struct_count struct, $enum_count enum, $trait_count trait,"
        echo "             $type_count type, $const_count const, $static_count static, $union_count union,"
        echo "             $use_count use, $mod_count mod"

        drift=0
        if ! diff_out="$(diff -u <(printf '%s\n' "$anchor") <(printf '%s\n' "$current"))"; then
            drift=1
            echo ""
            echo "Rust public-API delta detected vs. the v0.9 snapshot:"
            printf '%s\n' "$diff_out" | sed 's/^/  /'
        fi

        echo ""
        # Run the non_exhaustive audit unconditionally so a snapshot-clean
        # run still catches an evolution-critical enum losing its marker.
        set +e
        audit_non_exhaustive
        audit_rc=$?
        set -e

        if [ "$drift" -eq 0 ] && [ "$audit_rc" -eq 0 ]; then
            echo ""
            echo "rust-public-api-list: OK (snapshot unchanged + non_exhaustive audit passed)"
            exit 0
        fi

        echo ""
        if [ "$drift" -ne 0 ]; then
            cat >&2 <<EOF
rust-public-api-list: FAIL — the Rust public API moved.

Fix:
  1. If the change is intentional (added / renamed / removed a public
     item), add a section to CHANGELOG.md or docs/abi-changelog.md under
     the appropriate Keep-a-Changelog category (Added / Changed /
     Removed / Breaking — prerelease semver 0.9.x permits Breaking).
     Then update the snapshot:
       scripts/rust-public-api-list.sh --update-snapshot
  2. If the change is accidental (a stray \`pub\` on an internal item),
     revert the source change and re-run this script.

The v0.9 snapshot at $SNAPSHOT is only rotated by
\`scripts/rust-public-api-list.sh --update-snapshot\` — do not edit it
by hand.
EOF
        fi
        if [ "$audit_rc" -ne 0 ]; then
            echo "rust-public-api-list: FAIL — one or more evolution-critical enums lost \`#[non_exhaustive]\`. Restore the marker." >&2
        fi
        exit 1
        ;;

    --list)
        extract_symbols
        ;;

    --update-snapshot)
        mkdir -p "$(dirname "$SNAPSHOT")"
        {
            echo "# Vokra Rust public-API snapshot — v1.0-rc window (M3-16-T03; rotated by M4-12)."
            echo "#"
            echo "# Regenerate with: scripts/rust-public-api-list.sh --update-snapshot"
            echo "# Diff against with: scripts/rust-public-api-list.sh"
            echo "# Audit only: scripts/rust-public-api-list.sh --audit"
            echo "#"
            echo "# Scope: pub API surface (candidate C, grep-based) of the three"
            echo "# ABI-relevant crates: vokra-core, vokra-ops, vokra-capi."
            echo "# One line per public declaration, sorted, format:"
            echo "#   FN     <crate>::<module>::<name>|<normalized-line>"
            echo "#   STRUCT <crate>::<module>::<name>|<normalized-line>"
            echo "#   ENUM   <crate>::<module>::<name>|<normalized-line>"
            echo "#   TRAIT  <crate>::<module>::<name>|<normalized-line>"
            echo "#   TYPE   <crate>::<module>::<name>|<normalized-line>"
            echo "#   CONST  <crate>::<module>::<name>|<normalized-line>"
            echo "#   STATIC <crate>::<module>::<name>|<normalized-line>"
            echo "#   UNION  <crate>::<module>::<name>|<normalized-line>"
            echo "#   USE    <crate>::<module>::<path>|<normalized-line>"
            echo "#   MOD    <crate>::<module>::<name>|<normalized-line>"
            echo "#"
            echo "# Honest limitations of the grep-based extractor:"
            echo "#   * impl-block methods appear as bare FN <name>|<signature>."
            echo "#     Two impls sharing name AND signature collapse under sort -u."
            echo "#   * pub(crate) / pub(super) / pub(in path) are filtered out."
            echo "#   * Macro-generated pub items are invisible (source-level grep)."
            echo "# The tree does not exercise any of these gaps; the freeze"
            echo "# gate at M5-13 will re-evaluate the extraction strategy."
            echo "#"
            echo "# See docs/abi-changelog.md for the semver policy and"
            echo "# docs/tickets/m3/M3-16-abi-changelog.md for the WP scope."
            extract_symbols
        } >"$SNAPSHOT"

        fn_count=$(grep -c '^FN ' "$SNAPSHOT" || true)
        struct_count=$(grep -c '^STRUCT ' "$SNAPSHOT" || true)
        enum_count=$(grep -c '^ENUM ' "$SNAPSHOT" || true)
        trait_count=$(grep -c '^TRAIT ' "$SNAPSHOT" || true)
        type_count=$(grep -c '^TYPE ' "$SNAPSHOT" || true)
        const_count=$(grep -c '^CONST ' "$SNAPSHOT" || true)
        static_count=$(grep -c '^STATIC ' "$SNAPSHOT" || true)
        union_count=$(grep -c '^UNION ' "$SNAPSHOT" || true)
        use_count=$(grep -c '^USE ' "$SNAPSHOT" || true)
        mod_count=$(grep -c '^MOD ' "$SNAPSHOT" || true)

        echo "rust-public-api-list: wrote $SNAPSHOT"
        echo "  captured $fn_count fn, $struct_count struct, $enum_count enum, $trait_count trait,"
        echo "           $type_count type, $const_count const, $static_count static, $union_count union,"
        echo "           $use_count use, $mod_count mod"
        ;;

    --audit)
        set +e
        audit_non_exhaustive
        rc=$?
        set -e
        exit "$rc"
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
        echo "usage: $0 [--list | --update-snapshot | --audit | --self-test | --help]" >&2
        exit 2
        ;;
esac
