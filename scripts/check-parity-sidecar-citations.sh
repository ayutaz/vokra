#!/usr/bin/env bash
# check-parity-sidecar-citations.sh — the "cited bridge actually exists" gate.
#
# WHY THIS GATE EXISTS
#   Rust source under `crates/` names `tools/parity/*.py` sidecars constantly:
#   converter module docs say how to flatten an upstream torch pickle, binder
#   error strings tell an operator which bridge to re-run, parity harnesses
#   print a numbered fixture recipe. Those citations are INSTRUCTIONS. An
#   operator reads one, types it, and gets `No such file or directory` — after
#   which they stop trusting every other docstring in the tree, including the
#   ones that were right.
#
#   Measured on 2026-08-15: 109 distinct `tools/parity/*.py` paths were cited
#   across `crates/`, and 47 of them were not on disk. Most were phrased in the
#   PRESENT TENSE, as an existing tool ("Callers pre-flatten … offline through
#   `tools/parity/dtln_aec_prepare_checkpoint.py`"). About seven sites were
#   already honest ("a future `tools/parity/panns_prepare_checkpoint.py`"), so
#   the honest form existed and simply was not applied consistently.
#
#   THE SHARPEST SHAPE — a parity gate made permanently vacuous.
#   `crates/vokra-models/tests/parity_dnsmos.rs` carried a five-step recipe
#   whose step 3 ran `tools/parity/dnsmos_score_reference.py`. At the
#   2026-08-15 baseline that file had never existed, so
#   `VOKRA_DNSMOS_REFERENCE_JSONL` could not be produced by anything and the
#   MOS-comparison test always took its skip branch. The independent official
#   oracle and real comparison harness landed on 2026-08-26; this historical
#   incident remains the reason the general citation gate exists.
#
#   The tree already knew this class existed — `vokra-models/src/lib.rs` calls
#   out "the MISSING SIDECAR `tools/parity/nisqa_v2_weight_prepare_checkpoint.py`,
#   which the converter's docstring names but which has never been written".
#   One instance was handled by hand; ~35 were not. Handling them by hand again
#   is not a fix, because nothing stops the next one.
#
# THE RULE
#   Every `tools/parity/<name>.py` literal in a source or prose file under
#   `crates/` (see `SCANNED_EXTS`) must either
#     (a) exist on disk, or
#     (b) be marked, near the citation, as not-yet-written.
#
#   That is the whole invariant. It deliberately does NOT require the file to
#   exist: naming a bridge you intend to write is useful, and a citation that
#   says so is honest. What is forbidden is a citation that READS as an
#   existing tool while being absent.
#
# WHAT COUNTS AS A MARKER
#   `ABSENCE_MARKERS` below, matched case-insensitively against a WINDOW of the
#   citing line plus `WINDOW` lines either side. The vocabulary is the set of
#   phrasings already in use in the tree, not an invented convention — every
#   entry is annotated with what it was drawn from. The window exists because
#   the explanation and the path are routinely on different lines: doc comments
#   wrap (`//! … via a future` / `//! \`tools/parity/x.py\``), and a `pub const`
#   puts its whole justification in a doc block above the initialiser. The
#   window is also matched against NORMALIZED text (see `normalize`), because
#   the same wrapping splits the marker PHRASES themselves — the tree really
#   contains `the MISSING` / `// SIDECAR`, and `does **not** exist yet`.
#
#   Preferred phrasing for NEW citations, and the one the 2026-08-15 sweep
#   applied: a future `tools/parity/x.py` (not yet written).
#
#   `future` is only accepted ADJACENT to a `tools/parity/` path (see
#   `FUTURE_ADJACENT`), never as a bare word. "a follow-up wave in the future
#   will…" three lines above an unrelated citation must not launder it.
#
# WHY NOT scripts/check-doc-references.sh
#   That gate reads PUBLISHED DOCS under `docs/`. Every defect above lives
#   under `crates/` — Rust source, crate-local READMEs, fixture provenance
#   headers — none of which it opens. Extending it would also mix two different
#   questions — "is the handbook self-consistent" versus "does source cite
#   tools that exist" — into one failure message.
#
# NOT A LEDGER OF EXCEPTIONS, ON PURPOSE
#   Siblings like `check-arch-handshake.sh` carry double-sided `NO_READER` /
#   `NO_CONVERTER` ledgers because their accepted gaps are project-level facts
#   with reasons that belong in one reviewable place. This gate is the opposite
#   shape: the exception IS the sentence the reader sees, so recording it in a
#   shell array would put the honesty somewhere the operator never looks. Fix
#   the prose, and the gate goes green because the docstring got truthful —
#   which is the only outcome worth having. The self-test is correspondingly
#   double-sided in its own way: it checks that a hedge is accepted AND that an
#   un-hedged citation fails AND that a hedge just outside the window does not
#   count.
#
# Zero-dep: bash + python3 stdlib only (no jq, no pip, no cargo). Not a Vokra
# runtime dep.
# Exit: 0 = every cited sidecar exists or is marked absent, 1 = an unmarked
# citation of a missing file / a parser guard trip / a bad argument.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRATES_DEFAULT="$ROOT/crates"
TOOLS_DEFAULT="$ROOT/tools/parity"

usage() {
    cat <<'USAGE'
check-parity-sidecar-citations.sh — cited `tools/parity/*.py` bridges must exist

Usage:
  bash scripts/check-parity-sidecar-citations.sh
  bash scripts/check-parity-sidecar-citations.sh --help
  bash scripts/check-parity-sidecar-citations.sh --self-test

Walks every .rs / .md / .expected / .txt / .toml file under crates/ for
`tools/parity/<name>.py` literals. A citation passes if the file exists on
disk, OR if an absence marker appears within 8 lines of it (wide enough for a
`pub const` whose doc block carries the explanation). Preferred phrasing for a
bridge that has not been written:

    a future `tools/parity/x.py` (not yet written)

Rationale, the measured 47-of-109 baseline, and why the DNSMOS instance made a
parity gate permanently vacuous are documented at the top of this script.

Exit 1 on any unmarked citation of a missing file, or on a parser guard trip.
USAGE
}

# The checker. Args:
#   $1 crates root  (walked for SCANNED_EXTS files)
#   $2 tools/parity dir (existence is resolved against it)
# stdlib only. Reused verbatim by the main path and --self-test.
run_check() {
    python3 - "$1" "$2" <<'PY'
import os, re, sys

crates_root, tools_dir = sys.argv[1:3]

# How many lines either side of the citation the marker may appear on.
#
# MEASURED, not guessed. The dominant shape is not a wrapped sentence but a
# `pub const` whose DOC BLOCK carries the explanation while the path sits in
# the initialiser several lines below:
#
#     /// The offline sidecar that does not exist yet. It is the place a real
#     /// checkpoint's topology must be transcribed from and the
#     /// [`FIREREDVAD_SPEC_KEYS`] group emitted, mirroring the sibling
#     /// `tools/parity/*_prepare_checkpoint.py` bridges. Never shipped inside
#     /// the `vokra-*` runtime (NFR-DS-02).
#     pub const SIDECAR_PATH: &str = "tools/parity/firered_vad_prepare_checkpoint.py";
#
# Marker-to-citation distances across the honest sites on 2026-08-15: 0-2 for
# wrapped prose, 5 for `firered_vad::SIDECAR_PATH`, 6 for
# `chattts::PREP_SCRIPT_PATH` and for the DNSMOS recipe's step-3 NOTE. 8 takes
# the measured worst case plus two lines of headroom.
#
# KNOWN LIMITATION, stated rather than papered over: if two DIFFERENT missing
# paths are cited within 8 lines of each other and only one is marked, the
# marker covers both. Repeated citations of the SAME missing path are the
# common case and are harmless (one explanation, several mentions); the mixed
# case is not currently distinguishable and would need per-path attribution.
# CHECKED 2026-08-15: no file in the tree has two different missing paths
# within 8 lines, so nothing is relying on this hole today. If that stops
# being true the honest fix is per-path attribution, not a wider window.
WINDOW = 8

# A `tools/parity/...py` path, possibly with subdirectories (the vendored
# reference trees under `tools/parity/vendor/...` are cited too).
CITATION = re.compile(r'tools/parity/[A-Za-z0-9_./-]+\.py')

# `future` counts ONLY when it is IMMEDIATELY followed by the path, with
# nothing between but wrap noise: whitespace, a `//` / `///` / `//!` comment
# marker, a backtick, a rustdoc `*`, a markdown `#` / `>` / `-`. That is
# exactly the idiom the honest sites use —
#
#     //! Callers pre-flatten via a future
#     //! `tools/parity/beats_prepare_checkpoint.py` uv-managed sidecar.
#
# — and it admits no prose at all. The looser "within N characters"
# formulation was tried first and is wrong in both directions: it rejected
# the wrapped form above (the newline is not "a character within N" once the
# window is line-joined) while ACCEPTING "In future we may rewrite this. See
# `tools/parity/x.py`", where the word is about something else entirely.
# Requiring adjacency rather than proximity removes the threshold, and with
# it the judgement call about how far is too far. Matched against the
# NORMALIZED window (see `normalize`), so the wrap has already collapsed to
# a single space by the time this runs and only quoting/bracket noise is
# left to allow for.
FUTURE_ADJACENT = re.compile(r'future[\s(\[\'"#>-]*tools/parity/', re.IGNORECASE)

# Every entry is a phrasing ALREADY PRESENT in the tree, so this vocabulary
# documents the honest sites rather than imposing a new convention on them.
# Drawn from, in order: the canonical form applied by the 2026-08-15 sweep;
# nisqa/mod.rs + utmosv2/mod.rs + voila.rs; ten_vad/mod.rs + chattts/mod.rs +
# lib.rs:1500; lib.rs:1274 + lib.rs:1432; lib.rs:1213; firered_vad/mod.rs +
# smart_turn/mod.rs; sbv2_sdp_torch_parity.rs; whisper_medusa_v1.rs;
# clap/mod.rs + emotion2vec/mod.rs + speaker_3d_eres2net/mod.rs +
# voila/mod.rs; parity_rmvpe.rs's stale-path guard; and the two synthetic
# fixture strings in parity_tts_japanese.rs / parity_tts_continuous_vae.rs.
#
# Kept deliberately OUT of this list: `unreachable`. It reads like a fine
# marker and the DNSMOS module doc does use the word — but `unreachable!()`
# is an ordinary Rust macro, so accepting it would let a control-flow
# assertion three lines from a citation launder it. The DNSMOS doc says
# "not yet written" on the same line as the path and does not need it.
ABSENCE_MARKERS = [
    r'not yet written',
    r'does not exist',
    r'has (?:never|not) been written',
    r'never been written',
    r'missing sidecar',
    r'absent sidecar',
    r'not yet populated',
    r'would grow a',
    r'should live in',
    r'will front',
    r'would front',
    r'not as a shipped tool',
    r'must stay absent',
    r'previously documented',
    r'synthetic fixture',
]
MARKER_RE = re.compile("|".join(ABSENCE_MARKERS), re.IGNORECASE)


# Extensions walked. `.rs` is where the 47-of-109 defect lived, but the same
# citation class appears in crate-local prose and in fixture provenance
# headers — `crates/vokra-kws-micro/README.md` names a prepare script,
# `crates/vokra-core/tests/parity/fixtures/m5-06/*.expected` record which
# dumper produced them. Those are operator-facing instructions too, and a
# renamed tool would rot them just as silently. All three cited from non-`.rs`
# files were verified present on 2026-08-15, so widening cost nothing then —
# the point is that it cannot quietly break later. An explicit allowlist
# rather than "every text file" keeps binary fixtures (`.gguf` / `.bin` /
# `.wav`) out without a heuristic.
SCANNED_EXTS = (".rs", ".md", ".expected", ".txt", ".toml")


def source_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "target")
        for fn in sorted(filenames):
            if fn.endswith(SCANNED_EXTS):
                yield os.path.join(dirpath, fn)


# Leading comment noise on a wrapped line: indentation, then `//` / `///` /
# `//!` / a rustdoc `*` / a markdown `#`, then spacing.
LINE_PREFIX = re.compile(r'^[ \t]*(?://[/!]?|[*#])?[ \t]*')
# Markdown emphasis and Rust string-continuation backslashes, which sit
# INSIDE marker phrases in the tree: `does **not** exist yet`,
# `**That file has never been written.**`, `via a future \` + newline.
EMPHASIS = re.compile(r'[*`\\]')


def normalize(window_lines):
    """Flatten a window to one space-separated line for phrase matching.

    Every marker in `ABSENCE_MARKERS` is a PHRASE, and Rust comments wrap
    wherever rustfmt's column limit falls — so the tree really contains
    `the MISSING` / `// SIDECAR \\`tools/...\\``, `has not been` / `//!
    written`, and `does **not** exist yet`. Matching against the raw
    newline-joined window misses all three, and the failure is silent in
    the dangerous direction: an already-honest site reads as unmarked, and
    whoever is fixing the gate "corrects" prose that was fine.

    So: strip each line's comment prefix, drop markdown emphasis and
    string-continuation backslashes, and collapse whitespace. This is a
    matching aid only — the reported line number is still the citation's.
    """
    flat = " ".join(LINE_PREFIX.sub("", ln) for ln in window_lines)
    return re.sub(r'\s+', ' ', EMPHASIS.sub("", flat))


def marked(lines, idx):
    """True if an absence marker sits within WINDOW lines of `idx`."""
    lo = max(0, idx - WINDOW)
    hi = min(len(lines), idx + WINDOW + 1)
    window = normalize(lines[lo:hi])
    if MARKER_RE.search(window):
        return True
    return bool(FUTURE_ADJACENT.search(window))


errors = []
scanned_files = 0
citations = 0
cited_paths = set()
missing_paths = set()
marked_sites = 0

for path in source_files(crates_root):
    scanned_files += 1
    rel = os.path.relpath(path, crates_root)
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    for idx, line in enumerate(lines):
        for m in CITATION.finditer(line):
            cited = m.group(0)
            citations += 1
            cited_paths.add(cited)
            # `tools/parity/x.py` is repo-root-relative; resolve the tail
            # against the tools dir the caller handed us so --self-test can
            # point at a scratch tree.
            tail = cited[len("tools/parity/"):]
            if os.path.isfile(os.path.join(tools_dir, tail)):
                continue
            missing_paths.add(cited)
            if marked(lines, idx):
                marked_sites += 1
                continue
            errors.append(
                f"crates/{rel}:{idx + 1} cites `{cited}`, which does NOT exist. "
                f"An operator who follows this line gets `No such file or directory` "
                f"— and then has no reason to trust the next docstring either. Fix: "
                f"do NOT write the Python file to make the sentence true; make the "
                f"sentence true instead, e.g. \"a future `{cited}` (not yet "
                f"written)\". If a DIFFERENT bridge already does this job, name that "
                f"one. If the string is deliberately not a real path (a synthetic "
                f"test fixture), say so in a comment beside it."
            )

# ---- parser guards --------------------------------------------------------
# A checker that scanned nothing passes every run — the fabricated-pass shape
# this gate exists to prevent. Each guard fires only if the tree moved out
# from under the walk.
if scanned_files == 0:
    errors.append(
        f"no source files ({', '.join(SCANNED_EXTS)}) found anywhere under "
        f"{crates_root} — the walk is broken, so "
        f"a pass here would be vacuous."
    )
if citations == 0:
    errors.append(
        f"zero `tools/parity/*.py` citations found under {crates_root} — either the "
        f"CITATION regex stopped matching or the walk is broken. This gate has never "
        f"legitimately seen zero (the 2026-08-15 baseline was 109 distinct paths), so "
        f"treat it as a scanner defect, not as good news."
    )

if errors:
    print(f"check-parity-sidecar-citations: FAIL — {len(errors)} problem(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)

print(
    f"check-parity-sidecar-citations: OK — {citations} citation(s) across "
    f"{scanned_files} source file(s); {len(cited_paths)} distinct path(s), "
    f"{len(cited_paths) - len(missing_paths)} present on disk, "
    f"{len(missing_paths)} absent and marked as such at {marked_sites} site(s)."
)
PY
}

self_test() {
    local status=0
    local tmp
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' RETURN

    mkdir -p "$tmp/crates/alpha/src" "$tmp/crates/beta/src" "$tmp/tools"

    # One sidecar that really exists, so "present" is exercised too.
    printf '# a real bridge\n' >"$tmp/tools/real_bridge.py"

    local out

    # 1. A citation of a file that EXISTS passes with no marker at all.
    printf '//! Flatten via `tools/parity/real_bridge.py` first.\n' \
        >"$tmp/crates/alpha/src/lib.rs"
    if run_check "$tmp/crates" "$tmp/tools" >/dev/null 2>&1; then
        echo "self-test PASS: a citation of an existing sidecar needs no marker"
    else
        echo "self-test FAIL: an existing sidecar should pass unmarked" >&2
        status=1
    fi

    # 2. A PRESENT-TENSE citation of a MISSING file fails, naming the path
    #    and the file:line. This is the 47-of-109 defect itself.
    printf '//! Callers pre-flatten via `tools/parity/ghost_prepare.py` offline.\n' \
        >"$tmp/crates/beta/src/lib.rs"
    if out="$(run_check "$tmp/crates" "$tmp/tools" 2>&1)"; then
        echo "self-test FAIL: an unmarked citation of a missing file should fail" >&2
        status=1
    elif grep -q 'ghost_prepare.py' <<<"$out" && grep -q 'beta/src/lib.rs:1' <<<"$out"; then
        echo "self-test PASS: an unmarked missing citation fails, naming path and site"
    else
        echo "self-test FAIL: the failure did not name \`ghost_prepare.py\` at its site" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi

    # 3. The canonical hedge on the SAME line passes.
    printf '//! Via a future `tools/parity/ghost_prepare.py` (not yet written).\n' \
        >"$tmp/crates/beta/src/lib.rs"
    if run_check "$tmp/crates" "$tmp/tools" >/dev/null 2>&1; then
        echo "self-test PASS: the canonical \"(not yet written)\" hedge is accepted"
    else
        echo "self-test FAIL: the canonical hedge should pass" >&2
        status=1
    fi

    # 4. A hedge WRAPPED onto the previous line passes — the single most
    #    common real shape, and the whole reason WINDOW is not zero.
    {
        printf '//! Callers pre-flatten via a future\n'
        printf '//! `tools/parity/ghost_prepare.py` uv-managed Python 3.12 sidecar.\n'
    } >"$tmp/crates/beta/src/lib.rs"
    if run_check "$tmp/crates" "$tmp/tools" >/dev/null 2>&1; then
        echo "self-test PASS: a hedge wrapped onto the preceding line is accepted"
    else
        echo "self-test FAIL: a wrapped hedge should pass" >&2
        status=1
    fi

    # 5. A `pub const` whose DOC BLOCK explains the absence passes even
    #    though the path is several lines below it. This is the dominant
    #    real shape (firered_vad::SIDECAR_PATH, chattts::PREP_SCRIPT_PATH)
    #    and the reason WINDOW is 8 rather than 3.
    {
        printf '/// The offline sidecar that does not exist yet. It is the place\n'
        printf '/// a real topology must be transcribed from, mirroring the\n'
        printf '/// sibling bridges. Never shipped inside the runtime.\n'
        printf '///\n'
        printf '/// Named in the loud-partial error because it is the first thing\n'
        printf '/// that must land before a real forward is possible.\n'
        printf 'pub const SIDECAR_PATH: &str = "tools/parity/ghost_prepare.py";\n'
    } >"$tmp/crates/beta/src/lib.rs"
    if run_check "$tmp/crates" "$tmp/tools" >/dev/null 2>&1; then
        echo "self-test PASS: a doc block explaining the absence covers a const 6 lines below"
    else
        echo "self-test FAIL: a const whose doc block hedges it should pass" >&2
        status=1
    fi

    # 6. The window has an EDGE, and it is enforced. A marker 9 lines above
    #    the citation (one past WINDOW) does not launder it — otherwise
    #    "8 lines" would be decoration rather than a bound.
    {
        printf '//! This sidecar does not exist yet.\n'
        printf '//!\n//!\n//!\n//!\n//!\n//!\n//!\n//!\n'
        printf '//! Callers pre-flatten via `tools/parity/ghost_prepare.py`.\n'
    } >"$tmp/crates/beta/src/lib.rs"
    if out="$(run_check "$tmp/crates" "$tmp/tools" 2>&1)"; then
        echo "self-test FAIL: a marker 9 lines away should NOT launder a citation" >&2
        status=1
    elif grep -q 'ghost_prepare.py' <<<"$out"; then
        echo "self-test PASS: a marker one line past the window does not launder a citation"
    else
        echo "self-test FAIL: out-of-window case failed for the wrong reason" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi

    # 7. "future" in a sentence ABOUT SOMETHING ELSE does not count, even
    #    on the same line and only a few words away. Adjacency, not
    #    proximity: any prose between the word and the path disqualifies it.
    printf '//! In future we may rewrite this. See `tools/parity/ghost_prepare.py`.\n' \
        >"$tmp/crates/beta/src/lib.rs"
    if out="$(run_check "$tmp/crates" "$tmp/tools" 2>&1)"; then
        echo "self-test FAIL: a far-away \"future\" should not satisfy the marker" >&2
        status=1
    elif grep -q 'ghost_prepare.py' <<<"$out"; then
        echo "self-test PASS: \"future\" far from the path is not accepted as a marker"
    else
        echo "self-test FAIL: distant-\"future\" case failed for the wrong reason" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi

    # 8. The other real phrasings in the tree are accepted, one at a time,
    #    so a vocabulary entry cannot silently stop working.
    local phrase
    local -a phrases=(
        '//! The MISSING SIDECAR `tools/parity/ghost_prepare.py` was never written.'
        '//! `tools/parity/ghost_prepare.py` **does not exist yet**.'
        '//! The sidecar `tools/parity/ghost_prepare.py` has never been written.'
        '//! Named in the error, the absent sidecar `tools/parity/ghost_prepare.py`.'
        '//! A future v2 would grow a `tools/parity/ghost_prepare.py` bridge.'
        '//! Recipe should live in `tools/parity/ghost_prepare.py` (mirror).'
        '//! A `tools/parity/ghost_prepare.py` sidecar will front the converter.'
        '//! `tools/parity/ghost_prepare.py` is synthetic fixture text, not a path.'
    )
    for phrase in "${phrases[@]}"; do
        printf '%s\n' "$phrase" >"$tmp/crates/beta/src/lib.rs"
        if ! run_check "$tmp/crates" "$tmp/tools" >/dev/null 2>&1; then
            echo "self-test FAIL: this in-tree phrasing was rejected: $phrase" >&2
            status=1
        fi
    done
    echo "self-test PASS: all ${#phrases[@]} in-tree absence phrasings are accepted"

    # 9. A subdirectory citation (the vendored reference trees) resolves.
    mkdir -p "$tmp/tools/vendor/vits"
    printf 'class Generator: pass\n' >"$tmp/tools/vendor/vits/decoder.py"
    printf '//! Upstream reference (`tools/parity/vendor/vits/decoder.py`).\n' \
        >"$tmp/crates/beta/src/lib.rs"
    if run_check "$tmp/crates" "$tmp/tools" >/dev/null 2>&1; then
        echo "self-test PASS: a nested tools/parity/<dir>/<file>.py citation resolves"
    else
        echo "self-test FAIL: a nested vendored citation should pass" >&2
        status=1
    fi

    # 10. A nested citation that does NOT exist still fails — proving case 8
    #    passed because the file was there, not because nesting is skipped.
    printf '//! Upstream reference (`tools/parity/vendor/vits/models.py`).\n' \
        >"$tmp/crates/beta/src/lib.rs"
    if out="$(run_check "$tmp/crates" "$tmp/tools" 2>&1)"; then
        echo "self-test FAIL: a missing nested citation should fail" >&2
        status=1
    elif grep -q 'vendor/vits/models.py' <<<"$out"; then
        echo "self-test PASS: a missing nested citation fails, naming the nested path"
    else
        echo "self-test FAIL: missing-nested case failed for the wrong reason" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi
    rm -f "$tmp/crates/beta/src/lib.rs"

    # 11. A crate-local README is scanned too, not just `.rs`. Three real
    #     citations live in `.md` / `.expected` files under crates/; all
    #     pointed at existing tools on 2026-08-15, so this case is what
    #     keeps that true rather than what fixed it.
    printf '# See `tools/parity/ghost_prepare.py` to rebuild the fixtures.\n' \
        >"$tmp/crates/beta/README.md"
    if out="$(run_check "$tmp/crates" "$tmp/tools" 2>&1)"; then
        echo "self-test FAIL: a missing citation in a README should fail" >&2
        status=1
    elif grep -q 'README.md:1' <<<"$out"; then
        echo "self-test PASS: non-.rs files under crates/ are scanned too"
    else
        echo "self-test FAIL: README case failed for the wrong reason" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi
    rm -f "$tmp/crates/beta/README.md"

    # 12. Parser guard: a tree with source files but ZERO citations fails
    #     rather than reporting a confident clean run over nothing.
    printf 'pub fn f() {}\n' >"$tmp/crates/alpha/src/lib.rs"
    if out="$(run_check "$tmp/crates" "$tmp/tools" 2>&1)"; then
        echo "self-test FAIL: zero citations should trip the parser guard" >&2
        status=1
    elif grep -q 'zero .tools/parity' <<<"$out"; then
        echo "self-test PASS: a scan that found no citations fails rather than passing vacuously"
    else
        echo "self-test FAIL: zero-citation guard did not fire" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi

    # 13. Parser guard: no source files at all.
    rm -rf "$tmp/crates"
    mkdir -p "$tmp/crates"
    if run_check "$tmp/crates" "$tmp/tools" >/dev/null 2>&1; then
        echo "self-test FAIL: an empty crates tree should trip the parser guard" >&2
        status=1
    else
        echo "self-test PASS: an empty crates tree fails the parser guard"
    fi

    if [ "$status" -eq 0 ]; then
        echo "check-parity-sidecar-citations --self-test: OK (13 cases)"
    fi
    return "$status"
}

case "${1:-}" in
    --help | -h)
        usage
        exit 0
        ;;
    --self-test)
        self_test
        exit $?
        ;;
    "")
        if [ ! -d "$CRATES_DEFAULT" ]; then
            echo "error: required directory not found: $CRATES_DEFAULT" >&2
            exit 1
        fi
        if [ ! -d "$TOOLS_DEFAULT" ]; then
            echo "error: required directory not found: $TOOLS_DEFAULT" >&2
            exit 1
        fi
        run_check "$CRATES_DEFAULT" "$TOOLS_DEFAULT"
        ;;
    *)
        echo "error: unknown argument '$1'" >&2
        usage >&2
        exit 1
        ;;
esac
