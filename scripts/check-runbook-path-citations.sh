#!/usr/bin/env bash
# check-runbook-path-citations.sh — the "runbook sends you somewhere real" gate.
#
# WHY THIS GATE EXISTS
#   `tools/parity/**/README.md` and `docs/handoff/*.md` are RUNBOOKS. An owner
#   reads one while a rented vast.ai box bills by the minute, types the command
#   it gives, and either proceeds or gets `No such file or directory` with the
#   meter running. Unlike a stale docstring, a wrong path here costs money and
#   an interrupted multi-GB convert.
#
#   Measured on 2026-08-15: three separate runbooks sent the owner to a
#   provisioning script that has never existed at the cited path —
#     tools/parity/firered_asr_llm_l/README.md:60   bash scripts/vast-ai/provision.sh
#     tools/parity/higgs_audio_v3_tts_4b/README.md:56 bash scripts/vast-ai/provision.sh
#     tools/parity/melodyflow_t24_30secs/README.md:78 "run scripts/provision.sh"
#   — the real path being `scripts/publish/vast-ai/provision.sh`. Step 2 of an
#   8-step paid walkthrough, in all three.
#
#   The sweep that followed found the same class was not confined to those
#   three lines:
#     - `crates/vokra-bert/src/sentencepiece.rs` — commit cb2cd7b actually
#       created `crates/vokra-convert/src/spm_proto.rs`.
#     - `crates/vokra-models/src/speaker/campplus.rs` — the real file is
#       `camplus.rs`, ONE `p`. (The converter-side sibling really does have
#       two, so the asymmetry is easy to reproduce and impossible to notice.)
#     - `crates/vokra-kws-micro/tests/host_parity.rs` — renamed to
#       `parity_microwakeword.rs`.
#     - `crates/vokra-backend-{metal,cuda}/src/kernels/*.metal|*.cu.rs`, cited
#       as the "既存 GPU kernel pattern". No `.metal` file and no `kernels/`
#       directory exists in either backend: MSL and NVRTC sources are inline
#       `const` strings in `context.rs`. An owner following that recipe on a
#       rented GPU box would have written a kernel the build never compiles —
#       and commit 66d0077 proves the point, having landed the very kernel the
#       runbook uses as its example into `context.rs` instead.
#
# THE RULE
#   Every repo-relative path cited in a scanned runbook must either
#     (a) exist on disk, or
#     (b) be marked, near the citation, as not-yet-written.
#   Naming a file you intend to create is useful; what is forbidden is a
#   citation that READS as an existing file while being absent.
#
# WHAT IS IN SCOPE — the anchor rule, and why it is not "every path-like token"
#   Only paths ANCHORED at a known repo top-level directory are checked:
#   `scripts/ tools/ crates/ docs/ integrations/ .github/` (see `ANCHORS`),
#   and only when they end in a known extension or a `/`.
#
#   This is the whole answer to the false-positive problem, and it is a
#   deliberate design choice rather than an accident. A gate that cries wolf on
#   placeholders gets ignored, which is worse than no gate — so the following
#   are OUT OF SCOPE BY CONSTRUCTION, not by a growing list of exceptions:
#
#     - Remote/produced artifacts. `merged.safetensors`, `flat.safetensors`,
#       `<output>.sha256`, the `.gguf` a convert emits, `model.safetensors.index.json`
#       downloaded onto the rented box. None are anchored — they are bare
#       basenames or live under `/root/...` — so none are considered.
#     - Absolute paths on the remote box: `/root/vokra/target/release/vokra-cli`.
#       Not anchored (leading `/`).
#     - Bare basenames used prosaically: "`publish-one.sh` reads the signoff",
#       "the sibling `qwen3_tts.rs`". A basename is not a path claim; resolving
#       it would mean guessing a directory, and guessing is how a gate starts
#       lying. Cite it with its directory and the gate will check it.
#     - URLs. `https://astral.sh/uv/install.sh` ends in `.sh` but is skipped by
#       the preceding-scheme test in `is_url_context`.
#
#   Placeholders are excluded on top of the anchor rule, by `PLACEHOLDER`:
#   any `< > * ? { } $` or `...` (`<model>.gguf`, `crates/**/tests/*_cuda.rs`),
#   and the date/serial templates the bakeoff docs are built from —
#   `docs/handoff/m5-01-coreml-bakeoff-YYYY-MM-DD.md` is a filename the owner
#   fills in, not a file that should exist.
#
#   GITIGNORED PATHS ARE SKIPPED, and this one is subtle enough to spell out.
#   `docs/tickets/` and `docs/adr/` are gitignore-local by project policy —
#   present on the owner's machine, absent from the public checkout CI runs in.
#   Checking existence alone would make this gate pass locally and fail in CI
#   on citations that are perfectly correct. So an absent path is consulted
#   against `git check-ignore` (which matches RULES, and therefore works on
#   paths that do not exist) and skipped if ignored. Consequence, stated
#   plainly: a typo inside `docs/tickets/...` is NOT caught. That is the price
#   of not failing CI on correct citations, and it is bounded — those are
#   internal PM artefacts, not paid-runbook commands.
#
# WHAT COUNTS AS A MARKER
#   `ABSENCE_MARKERS`, matched case-insensitively against a window of the
#   citing line plus `WINDOW` lines either side, normalized (see `normalize`)
#   so that a phrase split across a markdown wrap still matches.
#
#   Every entry is a phrasing ALREADY IN USE in these runbooks — the
#   vocabulary documents the honest sites rather than imposing a convention on
#   them. Note that these docs are bilingual, so the vocabulary is too:
#   `未実装` / `存在しない` / `新規` carry exactly the weight of "not
#   implemented" / "does not exist" / "new" and appear far more often than
#   their English equivalents in the vast.ai runbooks.
#
#   Preferred phrasing for NEW citations, matching the sibling gate
#   `check-parity-sidecar-citations.sh`:
#
#       a future `crates/foo/src/bar.rs` (not yet written)
#
#   Creation verbs (`を追加` / `に追記` / `scaffold`) are accepted because a
#   line that says "add this file" is self-evidently talking about a file that
#   does not exist yet. They were checked against the real defects above: none
#   of the five sat near a creation verb or a future marker, because each was
#   phrased in the present tense as an existing thing. That is precisely what
#   makes the defect class detectable.
#
# WHY NOT scripts/check-parity-sidecar-citations.sh
#   That sibling scans `crates/` for `tools/parity/*.py` sidecar citations —
#   source citing tools. This one scans runbooks citing anything in the repo.
#   Different surface, different direction, and merging them would blur one
#   clear failure message into two questions. The two overlap on exactly one
#   file class (`tools/parity/**/README.md`) and agree there.
#
# Zero-dep: bash + python3 stdlib + git only (no jq, no pip, no cargo). Not a
# Vokra runtime dep.
# Exit: 0 = every cited path exists, is marked absent, or is out of scope.
#       1 = an unmarked citation of a missing file / a parser guard trip / a
#           bad argument.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    cat <<'USAGE'
check-runbook-path-citations.sh — cited paths in runbooks must exist

Usage:
  bash scripts/check-runbook-path-citations.sh
  bash scripts/check-runbook-path-citations.sh --help
  bash scripts/check-runbook-path-citations.sh --self-test

Scans `tools/parity/**/README.md` and `docs/handoff/*.md` for repo-relative
paths anchored at scripts/ tools/ crates/ docs/ integrations/ .github/. A
citation passes if the path exists, is gitignored (intentionally local), or
carries an absence marker within 4 lines. Preferred phrasing for a file that
has not been written:

    a future `crates/foo/src/bar.rs` (not yet written)

Placeholders (<model>.gguf, *.metal, YYYY-MM-DD templates), bare basenames,
absolute remote-box paths, and URLs are out of scope by construction — see the
header for why that matters more than breadth.

Exit 1 on any unmarked citation of a missing path, or on a parser guard trip.
USAGE
}

# The checker. Args:
#   $1 repo root (scanned, and paths resolve against it)
# stdlib + git only. Reused verbatim by the main path and --self-test.
run_check() {
    python3 - "$1" <<'PY'
import os, re, subprocess, sys

root = sys.argv[1]

# How many lines either side of the citation a marker may appear on.
#
# MEASURED on the 2026-08-15 sweep, not guessed. Markdown wraps hard and the
# explanation routinely sits on a neighbouring line or in the list-item stem:
#
#     - **Real audio inference accuracy verification is fixture-gated** —
#       when the future runtime binder in
#       `crates/vokra-models/src/firered_asr_llm_l/` lands, it will reach
#
# Distances observed across the honest sites: 0 for inline hedges, 1-2 for
# wrapped prose, 3 for a `## What this directory still does NOT do` heading
# above its bullet, 4 for the `新規` in a fenced-code comment above the path
# it introduces. 4 covers the measured worst case exactly; 5+ started pulling
# unrelated bullets into the window during tuning.
#
# KNOWN LIMITATION, stated rather than papered over: two DIFFERENT missing
# paths within 4 lines of each other, only one marked, means the marker covers
# both. Repeated citations of the SAME path are the common case and harmless.
# CHECKED 2026-08-15: the mixed case does not occur in the tree today.
WINDOW = 4

# Repo top-level directories a citation may be anchored at. Anything not
# starting with one of these is out of scope — see the header's rationale.
ANCHORS = ("scripts", "tools", "crates", "docs", "integrations", ".github")

# File extensions treated as a concrete file claim. A trailing `/` is also
# accepted, so a directory citation like `crates/vokra-models/src/pyannote/`
# is checked.
EXTS = ("sh", "py", "rs", "md", "toml", "yml", "yaml", "json")

CITATION = re.compile(
    r"(?<![\w./-])("
    + r"(?:" + "|".join(re.escape(a) for a in ANCHORS) + r")"
    + r"/[A-Za-z0-9_./+-]*"
    + r"(?:\.(?:" + "|".join(EXTS) + r")\b|/)"
    + r")"
)

# Anything with a fill-in-the-blank shape is not a claim that a file exists.
#   <model>.gguf / <output>          — angle placeholders
#   crates/**/tests/*_cuda.rs        — globs
#   ${VAR} / {a,b}                   — shell expansion
#   docs/handoff/m5-01-...-YYYY-MM-DD.md — date templates the owner fills in
#   docs/bench-baselines/vast-YYYY-MM-DD/ — same, for a results directory
PLACEHOLDER = re.compile(r"[<>*?{}$]|\.\.\.|YYYY|MM-DD")

# Characters that may continue a path-ish token, INCLUDING placeholder
# metacharacters. Used only to widen a match before the placeholder test.
TOKEN_TAIL = re.compile(r"[A-Za-z0-9_./+*?<>{}$-]")
# Placeholder metacharacters that may PRECEDE the anchor, e.g. `<repo>/docs/x.md`.
TOKEN_HEAD = re.compile(r"[*?<>{}$]")


def full_token(line, start, end):
    """Widen a match to the whole path-ish token before placeholder testing.

    Necessary because `CITATION` stops at the first `/` or known extension,
    so a glob like `crates/vokra-models/tests/*_cuda_bit_identical.rs` matches
    only its prefix `crates/vokra-models/tests/` — which looks like a plain
    directory citation and would be reported as missing. That is the exact
    cry-wolf failure the anchor rule exists to avoid, and the self-test pins
    it (case 6).

    Widening right over placeholder metacharacters, and left over the few that
    can precede an anchor, makes the placeholder test see the real token. The
    leftward walk is safe because `CITATION`'s lookbehind already guarantees
    the preceding character is not `[\\w./-]`.
    """
    while end < len(line) and TOKEN_TAIL.match(line[end]):
        end += 1
    while start > 0 and TOKEN_HEAD.match(line[start - 1]):
        start -= 1
    return line[start:end]

# Bilingual on purpose: these runbooks are written in mixed EN/JA and the
# Japanese forms are the more common ones in the vast.ai docs.
ABSENCE_MARKERS = [
    # --- explicit absence -------------------------------------------------
    r"not yet written",          # the canonical form, shared with the sibling gate
    r"does not exist",           # melodyflow / vocoder-gpu-kernels
    r"never been written",       # sibling-gate vocabulary, kept in step
    r"there is no",              # residual-wave3's m5-05 correction
    r"未実装",                    # vast-ai-publish-* §7 "runtime forward は未実装"
    r"存在しない",                 # vocoder-gpu-kernels "`.metal` ファイルも … 存在しない"
    # --- explicitly future ------------------------------------------------
    r"future wave",              # vast-ai-publish-firered / higgs / large-model
    r"future runtime binder",    # firered + higgs sidecar READMEs
    r"新規",                      # vocoder-cuda "（新規）", the dominant JA form
    r"\(new\b",                  # pyannote plan "(new)" / "(new op)"
    r"実装候補",                   # pyannote plan "Vokra 実装候補"
    r"deferred",                 # m5-04 "(d) Deferred"
    r"did not build",            # m5-04 "this wave intentionally did not build"
    r"gitignore-local",          # m5-04's console-portability doc
    # --- creation verbs: the line is an instruction to CREATE the file ----
    r"scaffold",                 # "scaffold 追加" / "section 7.1 scaffold"
    r"を追加",                     # "MSL kernel を追加"
    r"に追記",                     # "context.rs に … 追記"
    r"を commit",                 # "…yml を commit"
]
MARKER_RE = re.compile("|".join(ABSENCE_MARKERS), re.IGNORECASE)


def scanned_files(root):
    """The two runbook surfaces, in a stable order."""
    out = []
    parity = os.path.join(root, "tools", "parity")
    for dirpath, dirnames, filenames in os.walk(parity):
        # Skip virtualenvs / caches that can contain vendored READMEs.
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in (".venv", "parity-venv", "__pycache__", ".pytest_cache",
                         "node_modules", "scratchpad", "vendor")
        )
        for fn in sorted(filenames):
            if fn == "README.md":
                out.append(os.path.join(dirpath, fn))
    handoff = os.path.join(root, "docs", "handoff")
    if os.path.isdir(handoff):
        for fn in sorted(os.listdir(handoff)):
            if fn.endswith(".md"):
                out.append(os.path.join(handoff, fn))
    return out


def is_url_context(line, start):
    """True if the match sits inside a URL (…//host/path.sh)."""
    return re.search(r"https?://\S*$", line[:start]) is not None


# Leading markdown/comment noise on a wrapped line, so a marker phrase split
# across lines still matches: indentation, list bullets, blockquote markers,
# heading hashes, and shell-comment `#` inside fenced blocks.
LINE_PREFIX = re.compile(r"^[ \t]*(?:[-*>+]|#{1,6}|//[/!]?)?[ \t]*")
# Markdown emphasis and backticks sit INSIDE marker phrases in these docs
# (`**does not exist**`, `` `新規` ``), so drop them before matching.
EMPHASIS = re.compile(r"[*`\\]")


def normalize(window_lines):
    """Flatten a window to one space-separated line for phrase matching.

    Markers are PHRASES and markdown wraps wherever the column limit falls, so
    the tree really contains `when the future` / `runtime binder in`. Matching
    the raw newline-joined window misses those, and the failure is silent in
    the dangerous direction: an already-honest site reads as unmarked, and
    whoever fixes the gate "corrects" prose that was fine.

    Matching aid only — the reported line number is still the citation's.
    """
    flat = " ".join(LINE_PREFIX.sub("", ln) for ln in window_lines)
    return re.sub(r"\s+", " ", EMPHASIS.sub("", flat))


def marked(lines, idx):
    """True if an absence marker sits within WINDOW lines of `idx`."""
    lo = max(0, idx - WINDOW)
    hi = min(len(lines), idx + WINDOW + 1)
    return bool(MARKER_RE.search(normalize(lines[lo:hi])))


def git_ignored(root, rel):
    """True if `rel` matches a .gitignore RULE (works on absent paths).

    This is what keeps the gate honest in CI, where `docs/tickets/` and
    `docs/adr/` are absent by policy but their citations are correct.
    """
    try:
        return subprocess.run(
            ["git", "-C", root, "check-ignore", "-q", "--no-index", rel],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0
    except OSError:
        # No git available: fail OPEN for this one test rather than
        # inventing violations across every internal-doc citation.
        return False


errors = []
scanned = 0
citations = 0
cited_paths = set()
missing_marked = 0
skipped_ignored = 0

for path in scanned_files(root):
    scanned += 1
    rel_file = os.path.relpath(path, root)
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    for idx, line in enumerate(lines):
        for m in CITATION.finditer(line):
            cited = m.group(1)
            # Test the WIDENED token, not the match: see `full_token`.
            if PLACEHOLDER.search(full_token(line, m.start(1), m.end(1))):
                continue
            if is_url_context(line, m.start()):
                continue
            citations += 1
            cited_paths.add(cited)
            if os.path.exists(os.path.join(root, cited)):
                continue
            if git_ignored(root, cited):
                skipped_ignored += 1
                continue
            if marked(lines, idx):
                missing_marked += 1
                continue
            errors.append(
                f"{rel_file}:{idx + 1} cites `{cited}`, which does NOT exist. "
                f"This is a runbook: an owner follows it with a rented box "
                f"billing by the minute, and gets `No such file or directory`. "
                f"Fix: do NOT create a file to make the sentence true — make "
                f"the sentence true. If the path moved or was renamed, cite the "
                f"real one. If it is genuinely unwritten, say so, e.g. "
                f"\"a future `{cited}` (not yet written)\"."
            )

# ---- parser guards --------------------------------------------------------
# A checker that scanned nothing passes every run — the fabricated-pass shape
# this gate exists to prevent. Each guard fires only if the tree moved out
# from under the walk.
if scanned == 0:
    errors.append(
        f"no runbooks found under {root}/tools/parity or {root}/docs/handoff — "
        f"the walk is broken, so a pass here would be vacuous."
    )
if citations == 0:
    errors.append(
        f"zero anchored path citations found across {scanned} runbook(s) — "
        f"either the CITATION regex stopped matching or the walk is broken. "
        f"This gate has never legitimately seen zero (the 2026-08-15 baseline "
        f"was ~930 citations), so treat it as a scanner defect, not good news."
    )

if errors:
    print(f"check-runbook-path-citations: FAIL — {len(errors)} problem(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)

print(
    f"check-runbook-path-citations: OK — {citations} anchored citation(s) "
    f"across {scanned} runbook(s); {len(cited_paths)} distinct path(s), "
    f"{missing_marked} absent-but-marked, {skipped_ignored} gitignore-local."
)
PY
}

self_test() {
    local status=0
    local tmp
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' RETURN

    mkdir -p "$tmp/tools/parity/demo" "$tmp/docs/handoff" "$tmp/scripts/publish/vast-ai"
    # A real script, so "present" is exercised rather than assumed.
    printf '#!/usr/bin/env bash\n' >"$tmp/scripts/publish/vast-ai/provision.sh"
    # A git repo so `git check-ignore` has something to answer against.
    git -C "$tmp" init -q 2>/dev/null || true
    printf '/docs/tickets/\n' >"$tmp/.gitignore"

    local out rc

    # Helper: run the checker over the scratch tree, capturing output+status.
    # The assignment is the `if` CONDITION on purpose: that suspends `set -e`
    # for it, so a legitimately-failing case (most of them below) records rc
    # instead of aborting the whole self-test on its first red case.
    run_scratch() {
        if out="$(run_check "$tmp" 2>&1)"; then rc=0; else rc=$?; fi
        return 0
    }

    # 1. A citation of a path that EXISTS passes with no marker at all.
    printf 'Run `bash scripts/publish/vast-ai/provision.sh` first.\n' \
        >"$tmp/tools/parity/demo/README.md"
    run_scratch
    if [ "$rc" -eq 0 ]; then
        echo "self-test PASS: an existing path needs no marker"
    else
        echo "self-test FAIL: an existing path should pass unmarked" >&2
        printf '%s\n' "$out" >&2; status=1
    fi

    # 2. THE DEFECT ITSELF — the exact wrong path from three real runbooks.
    #    Must fail, naming the path and the file:line.
    printf 'Run `bash scripts/vast-ai/provision.sh` first.\n' \
        >"$tmp/tools/parity/demo/README.md"
    run_scratch
    if [ "$rc" -eq 0 ]; then
        echo "self-test FAIL: the real 2026-08-15 defect should fail" >&2; status=1
    elif grep -q 'scripts/vast-ai/provision.sh' <<<"$out" \
        && grep -q 'demo/README.md:1' <<<"$out"; then
        echo "self-test PASS: the real provision.sh defect fails, naming path and site"
    else
        echo "self-test FAIL: failure did not name the path at its site" >&2
        printf '%s\n' "$out" >&2; status=1
    fi

    # 3. The canonical hedge on the same line passes.
    printf 'A future `crates/foo/src/bar.rs` (not yet written) will land.\n' \
        >"$tmp/tools/parity/demo/README.md"
    run_scratch
    [ "$rc" -eq 0 ] \
        && echo "self-test PASS: the canonical \"(not yet written)\" hedge is accepted" \
        || { echo "self-test FAIL: canonical hedge should pass" >&2; status=1; }

    # 4. A hedge WRAPPED onto the previous line passes — the dominant real
    #    shape, and the whole reason WINDOW is not zero.
    {
        printf -- '- **Verification is fixture-gated** — when the future\n'
        printf '  runtime binder in\n'
        printf '  `crates/vokra-models/src/firered_asr_llm_l/` lands, it will\n'
    } >"$tmp/tools/parity/demo/README.md"
    run_scratch
    [ "$rc" -eq 0 ] \
        && echo "self-test PASS: a hedge wrapped across lines is accepted" \
        || { echo "self-test FAIL: wrapped hedge should pass" >&2; status=1; }

    # 5. The window has an EDGE and it is enforced: a marker 5 lines above
    #    (one past WINDOW) does not launder the citation.
    {
        printf 'This binder does not exist yet.\n'
        printf '\n\n\n\n'
        printf 'Then run `crates/ghost/src/lib.rs` to finish.\n'
    } >"$tmp/tools/parity/demo/README.md"
    run_scratch
    if [ "$rc" -ne 0 ] && grep -q 'crates/ghost/src/lib.rs' <<<"$out"; then
        echo "self-test PASS: a marker one line past the window does not launder"
    else
        echo "self-test FAIL: out-of-window marker should not launder" >&2
        printf '%s\n' "$out" >&2; status=1
    fi

    # 6. PLACEHOLDERS ARE NOT FLAGGED. This is the case that decides whether
    #    the gate is usable: it must stay silent on globs, angle-brackets,
    #    and the YYYY-MM-DD templates the bakeoff runbooks are built from.
    #    A gate that cries wolf here gets ignored.
    {
        printf 'Fill in `docs/handoff/m5-01-coreml-bakeoff-YYYY-MM-DD.md`.\n'
        printf 'Record into `docs/bench-baselines/vast-YYYY-MM-DD/out.jsonl`.\n'
        printf 'Mirror every `crates/vokra-models/tests/*_cuda_bit_identical.rs`.\n'
        printf 'Convert to `<model>.gguf` via `scripts/${TOOL}/run.sh`.\n'
        printf 'Also see `bash scripts/publish/vast-ai/provision.sh`.\n'
    } >"$tmp/tools/parity/demo/README.md"
    run_scratch
    if [ "$rc" -eq 0 ]; then
        echo "self-test PASS: placeholders/globs/date-templates are not flagged"
    else
        echo "self-test FAIL: a placeholder was flagged — the gate would be ignored" >&2
        printf '%s\n' "$out" >&2; status=1
    fi

    # 7. Out-of-scope shapes stay silent: bare basenames, absolute paths on
    #    the rented box, and URLs that happen to end in .sh.
    {
        printf 'The `publish-one.sh` script reads the signoff from `upload.sh`.\n'
        printf 'Run `/root/vokra/target/release/vokra-cli convert`.\n'
        printf 'Install: `curl -LsSf https://astral.sh/uv/install.sh | sh`.\n'
        printf 'Then `bash scripts/publish/vast-ai/provision.sh`.\n'
    } >"$tmp/tools/parity/demo/README.md"
    run_scratch
    if [ "$rc" -eq 0 ]; then
        echo "self-test PASS: basenames, remote-absolute paths and URLs are out of scope"
    else
        echo "self-test FAIL: an out-of-scope shape was flagged" >&2
        printf '%s\n' "$out" >&2; status=1
    fi

    # 8. A gitignored path is skipped even though absent — the CI-vs-local
    #    divergence this gate would otherwise invent. `docs/tickets/` is
    #    gitignore-local by project policy.
    printf 'See `docs/tickets/coverage-audit/wave-b/ghost.md` for the ticket.\n' \
        >"$tmp/tools/parity/demo/README.md"
    run_scratch
    if [ "$rc" -eq 0 ]; then
        echo "self-test PASS: gitignore-local citations are skipped, not failed"
    else
        echo "self-test FAIL: a gitignored citation should be skipped" >&2
        printf '%s\n' "$out" >&2; status=1
    fi

    # 9. But a NON-ignored absent path in the same doc still fails — proving
    #    case 8 passed because of the ignore rule, not because docs/ is skipped.
    printf 'See `docs/handoff/ghost-runbook.md` for the ticket.\n' \
        >"$tmp/tools/parity/demo/README.md"
    run_scratch
    if [ "$rc" -ne 0 ] && grep -q 'docs/handoff/ghost-runbook.md' <<<"$out"; then
        echo "self-test PASS: a non-ignored absent docs/ path still fails"
    else
        echo "self-test FAIL: non-ignored absent docs/ path should fail" >&2
        printf '%s\n' "$out" >&2; status=1
    fi

    # 10. `docs/handoff/*.md` is scanned too, not just parity READMEs.
    printf 'Run `bash scripts/ghost/setup.sh` on the box.\n' \
        >"$tmp/docs/handoff/runbook.md"
    printf 'ok\n' >"$tmp/tools/parity/demo/README.md"
    run_scratch
    if [ "$rc" -ne 0 ] && grep -q 'docs/handoff/runbook.md:1' <<<"$out"; then
        echo "self-test PASS: docs/handoff/*.md is scanned as well"
    else
        echo "self-test FAIL: handoff docs should be scanned" >&2
        printf '%s\n' "$out" >&2; status=1
    fi
    rm -f "$tmp/docs/handoff/runbook.md"

    # 11. Every in-tree absence phrasing is accepted, one at a time, so a
    #     vocabulary entry cannot silently stop working. Bilingual on purpose.
    local phrase
    local -a phrases=(
        'A future `crates/ghost/src/lib.rs` (not yet written).'
        'The file `crates/ghost/src/lib.rs` does not exist.'
        'There is no `crates/ghost/src/lib.rs` in the tree.'
        'runtime forward は未実装、`crates/ghost/src/lib.rs` は future wave。'
        '`crates/ghost/src/lib.rs` は存在しない。'
        '新規 `crates/ghost/src/lib.rs` を作る。'
        '- `crates/ghost/src/lib.rs` (new op) — learnable sinc conv1d'
        'Vokra 実装候補: `crates/ghost/src/lib.rs`'
        'Deferred: a `crates/ghost/src/lib.rs` internal doc.'
        'This wave intentionally did not build `crates/ghost/src/lib.rs`.'
        '`crates/ghost/src/lib.rs` scaffold 追加'
        '# MSL kernel `crates/ghost/src/lib.rs` を追加'
    )
    for phrase in "${phrases[@]}"; do
        printf '%s\n' "$phrase" >"$tmp/tools/parity/demo/README.md"
        run_scratch
        if [ "$rc" -ne 0 ]; then
            echo "self-test FAIL: in-tree phrasing rejected: $phrase" >&2
            status=1
        fi
    done
    echo "self-test PASS: all ${#phrases[@]} in-tree absence phrasings are accepted"

    # 12. Directory citations are checked, not just files — the
    #     `crates/vokra-backend-metal/src/kernels/` defect had no extension.
    printf 'Add it under `crates/vokra-backend-metal/src/kernels/` and rebuild.\n' \
        >"$tmp/tools/parity/demo/README.md"
    run_scratch
    if [ "$rc" -ne 0 ] && grep -q 'src/kernels/' <<<"$out"; then
        echo "self-test PASS: an absent directory citation is caught"
    else
        echo "self-test FAIL: directory citation should be caught" >&2
        printf '%s\n' "$out" >&2; status=1
    fi

    # 13. Parser guard: runbooks exist but contain zero anchored citations.
    printf 'Nothing to see here.\n' >"$tmp/tools/parity/demo/README.md"
    run_scratch
    if [ "$rc" -ne 0 ] && grep -q 'zero anchored path citations' <<<"$out"; then
        echo "self-test PASS: a scan finding no citations fails rather than passing vacuously"
    else
        echo "self-test FAIL: zero-citation guard did not fire" >&2
        printf '%s\n' "$out" >&2; status=1
    fi

    # 14. Parser guard: no runbooks at all.
    rm -rf "$tmp/tools" "$tmp/docs"
    run_scratch
    if [ "$rc" -ne 0 ]; then
        echo "self-test PASS: an empty tree trips the parser guard"
    else
        echo "self-test FAIL: an empty tree should trip the parser guard" >&2
        status=1
    fi

    if [ "$status" -eq 0 ]; then
        echo "check-runbook-path-citations --self-test: OK (14 cases)"
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
        if [ ! -d "$ROOT/docs/handoff" ]; then
            echo "error: required directory not found: $ROOT/docs/handoff" >&2
            exit 1
        fi
        if [ ! -d "$ROOT/tools/parity" ]; then
            echo "error: required directory not found: $ROOT/tools/parity" >&2
            exit 1
        fi
        run_check "$ROOT"
        ;;
    *)
        echo "error: unknown argument '$1'" >&2
        usage >&2
        exit 1
        ;;
esac
