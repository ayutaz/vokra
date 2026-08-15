#!/usr/bin/env bash
# check-crate-path-citations.sh — every `vokra_<crate>::<ident>` named in a
# comment, docstring or string literal under `crates/`, `integrations/` and
# `tools/` must resolve to a real, PUBLIC item of that crate, or be a declared
# exception that reads honestly.
#
# WHY THIS EXISTS (the phantom-citation class, widened)
# ---------------------------------------------------------------------------
# Round 5 landed `check-ops-path-citations.sh`, which nails the `vokra_ops::`
# namespace shut. Round 6 found the same defect class living happily in the
# OTHER crate namespaces, which nothing checked:
#
#   * `vokra_models::rnnoise_v02` — cited by `vokra-ops/src/lib.rs` as the
#     consumer of the RNNoise v0.2 op set, and by `vokra-models/src/lib.rs` as
#     a landed member of the denoise family. No such module has ever existed;
#     the only consumer outside `vokra-ops` is a test.
#   * `vokra_eval::dnsmos::{p808_score, p835_score}` — five sites (four Rust,
#     one Python sidecar) named it as the runtime side of the DNSMOS converter.
#     It landed at `vokra_models::dnsmos_p808_p835`. `vokra-eval` has no
#     `dnsmos` module and its CLI treats `dnsmos` as a fail-closed unknown.
#   * `vokra_core::restamp_provenance` — cited as the route to the
#     `vokra.whisper.*` chunks a binder needed. It lives in `vokra-convert`,
#     and its body carries every non-provenance metadata key through verbatim,
#     so it structurally CANNOT add those chunks. Wrong crate AND unusable for
#     the stated purpose.
#
# Widening also turned up a second, subtler shape the ops gate never had to
# think about: a path whose first segment is a REAL FILE but a PRIVATE module,
# so the citation names something a downstream can never write.
#
#   * `vokra_backend_cpu::dispatch::fused_log_mel_dispatch` — `dispatch.rs`
#     exists, but `lib.rs` declares `mod dispatch;` and re-exports the function
#     at the crate root. The public path is `vokra_backend_cpu::fused_log_mel_dispatch`.
#   * `vokra_vad_micro::math` — `math.rs` exists, `mod math;` is private, and
#     the module's own docstring says it is "deliberately **private**". The
#     public scalar surface is `vokra_vad_micro::scalar`.
#   * `vokra_convert::safetensors` — private `pub(crate) use` re-export of
#     `vokra_core::safetensors`, which is the only public path.
#   * `vokra_cli::bench::parse_backend` — `vokra-cli` is a BINARY-ONLY crate
#     (no `lib.rs`), so no `vokra_cli::` path is nameable by anyone, and
#     `parse_backend` is `pub(crate)` on top of that. A file path is the only
#     honest referent.
#
# Resolving against the PUBLIC surface rather than "does a file of that name
# exist" is what makes those four findable at all.
#
# DIVISION OF LABOUR WITH THE SIBLING GATE
# ---------------------------------------------------------------------------
# `vokra_ops::` is owned by `scripts/check-ops-path-citations.sh` (its own
# ledger, its own rationale block). This gate deliberately SKIPS that one
# namespace so the two never maintain contradictory ledgers for the same
# identifier. To stop that split from silently becoming a hole again, this
# script HARD-FAILS if the sibling script is missing — deleting it can no
# longer quietly un-check 1000+ `vokra_ops::` citations.
#
# WHAT IT CHECKS (and, deliberately, what it does not)
# ---------------------------------------------------------------------------
# Namespaces are DISCOVERED from the filesystem: every `vokra-*` directory
# under `crates/` or `integrations/` becomes `vokra_<snake>`. That is what
# keeps `vokra_status_t::VOKRA_OK` and friends out — those are C ABI enum
# types from `vokra-capi`, not crate namespaces, and they are ignored because
# no such directory exists (115 + 9 + 8 occurrences that must not be flagged).
#
# For every `vokra_<crate>::<seg1>`, `<seg1>` must be one of:
#   * a `pub mod` of that crate's root module, or
#   * an item re-exported / declared `pub` at that crate root.
# A trailing `*` is a wildcard and prefix-matched.
#
# A binary-only crate (`vokra-cli`: `main.rs`, no `lib.rs`) resolves NOTHING
# by construction. That is correct, not a bug — a binary crate has no
# importable namespace, so every `vokra_cli::` citation must either be
# re-expressed as a file path or be ledgered.
#
# DEPTH IS DELIBERATELY LIMITED TO THE FIRST SEGMENT, for the reason the ops
# gate documents at length: deeper segments are associated functions, enum
# variants, struct fields, trait methods and `#[cfg(test)]` modules that no
# regex can tell apart from a missing item. Every defect found in six rounds
# has been at the first segment — including the two "wrong depth" ones
# (`vokra_core::chunks`, really `vokra_core::gguf::chunks`), which this layer
# catches precisely BECAUSE the first segment is the wrong one.
#
# THE EXCEPTION LEDGER
# ---------------------------------------------------------------------------
# Naming a path that does not resolve is legitimate in exactly two shapes:
# a primitive that is proposed but unlanded, and a design that was CONSIDERED
# AND REJECTED (prose explaining why the code is not where a reader might
# look for it is worth more than silence). Such a citation must clear TWO bars:
#   1. `<namespace>::<ident>` is listed in EXPECTED_UNLANDED below, with a
#      rationale;
#   2. EVERY occurrence carries an explicit marker in its immediate context
#      ("PROPOSED", "no such module", "does not exist", "instead", ...) so the
#      prose cannot read as a claim that the path is available today.
# Bar 2 is what separates this from an allowlist: ledgering `vokra_eval::dnsmos`
# would NOT have rescued the four converter comments, because they said "the
# runtime side is" rather than "there is no such module".
#
# SCOPE: `crates/`, `integrations/`, `tools/`, over `.rs`, `.md` and `.py`.
# The `.py` reach is not decoration — two of this round's stale citations were
# in `tools/parity/*_prepare_checkpoint.py` sidecars, which a Rust-only sweep
# would have missed exactly the way the ops gate's `vokra-models`-only
# predecessor missed `vokra-convert`.
#
# `docs/` IS DELIBERATELY EXCLUDED, on the same judgement the ops gate states:
# `docs/tickets/` is planning material whose job is to name the modules a
# model WOULD need. Gating that genre would either cry wolf on every planning
# note or force a ledger so large it stops meaning anything. Run it over docs/
# manually if ever wanted:
#   VOKRA_CITE_SCAN_DIRS=docs bash scripts/check-crate-path-citations.sh
#
# Local: `bash scripts/check-crate-path-citations.sh`
#        `bash scripts/check-crate-path-citations.sh --self-test`
# CI: runs in ci.yml's `license` job beside check-ops-path-citations.sh.
#
# Exit code: 0 = every citation resolves or is a declared, honestly-marked
# exception; 1 = a phantom citation (or a self-test that failed to detect an
# injected phantom).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# --- the exception ledger ----------------------------------------------------
# `<namespace>::<ident>` pairs that may appear WITHOUT resolving. Every
# occurrence must still carry an honesty marker (see MARKERS in the Python
# core). Nothing is listed here that has not been verified by reading the
# cited crate's public surface.
#
#   vokra_bert::converter      — the original SBV2 v2 plan drew the DeBERTa
#                                converter as `vokra-bert::converter`. Landing
#                                it there would have created the dependency
#                                cycle `vokra-bert -> vokra-convert ->
#                                vokra-bert`, so the real implementation lives
#                                in `vokra-convert/src/models/deberta_v2.rs`.
#                                The rejected shape is named in prose that
#                                explains the cycle; keeping it is worth more
#                                than deleting it.
#   vokra_convert::safetensors — same paragraph, and doubly unresolvable:
#                                `vokra-convert/src/safetensors.rs` is a
#                                `pub(crate) use vokra_core::safetensors::*`
#                                re-export declared `mod safetensors;`. The
#                                public path is `vokra_core::safetensors`.
#   vokra_eval::dnsmos         — never existed. The DNSMOS binder landed at
#                                `vokra_models::dnsmos_p808_p835`; `vokra-eval`
#                                has no `dnsmos` module and its CLI rejects
#                                `dnsmos` as an unknown metric
#                                (`vokra-eval/src/main.rs`
#                                `dnsmos_is_not_a_known_metric`). The surviving
#                                citations are all NEGATIONS that say so.
#   vokra_eval::squim          — same shape: the sidecar anticipated
#                                `vokra_eval::squim::from_gguf`; it landed at
#                                `vokra_models::squim` instead, and
#                                `squim/mod.rs` carries a section explaining
#                                the `vokra-models` / `vokra-eval` layer split.
#   vokra_models::load         — PROPOSED shared GGUF -> engine dispatch helper
#                                that `vokra-capi`'s private `build_session`
#                                and `vokra-cli`'s `engine.rs` would both call.
#                                Not landed; the small match is duplicated in
#                                both places today.
EXPECTED_UNLANDED=(
    "vokra_bert::converter"
    "vokra_convert::safetensors"
    "vokra_eval::dnsmos"
    "vokra_eval::squim"
    "vokra_models::load"
)

# The sibling gate that owns the `vokra_ops::` namespace this script skips.
SIBLING_GATE="scripts/check-ops-path-citations.sh"

# --- the checker core --------------------------------------------------------
# Reads: VOKRA_CITE_SCAN_ROOT (tree to scan), VOKRA_CITE_SURFACE_ROOT (tree to
# read crate public surfaces from), VOKRA_CITE_LEDGER (space-separated
# `ns::ident` pairs). Kept in one string so the self-test can re-run it against
# a synthetic tree without duplicating the logic.
#
# NOTE ON THE ASSIGNMENT FORM (`read -r -d ''`, not `$(cat <<'EOF')`):
# macOS ships bash 3.2, whose `$( ... )` parser scans a quoted heredoc's body
# for metacharacters instead of treating it as literal. An ODD number of
# backticks inside the body makes it die with "unexpected EOF while looking for
# matching \`". The sibling ops gate uses `$(cat <<'EOF')` and survives only
# because its Python happens to contain an even number of backticks — adding
# one more would break it for every macOS developer (CI runs bash 5 on ubuntu,
# so it would not show up there). `read -r -d ''` is not a command
# substitution, so the body is genuinely literal and the hazard is gone.
# `read` returns non-zero at EOF, hence the `|| true` under `set -e`.
IFS= read -r -d '' PY_CORE <<'PYEOF' || true
import os, re, sys

root    = os.environ["VOKRA_CITE_SCAN_ROOT"]
ledger  = set(os.environ.get("VOKRA_CITE_LEDGER", "").split())
# Public surfaces are read from the REAL tree so the self-test can scan a
# synthetic one while still resolving against the true crate list.
surface_root = os.environ.get("VOKRA_CITE_SURFACE_ROOT", root)

# `vokra_ops` is owned by the sibling gate (see header). Skipping it here is
# what stops two ledgers from disagreeing about the same identifier.
SKIP_NAMESPACES = {"vokra_ops"}

# Honesty markers. At least one must appear in the citation's own
# neighbourhood for an unresolved citation to read as "proposed / rejected /
# relocated" rather than "available".
MARKERS = [
    "proposed", "no such module", "not a landed", "not landed",
    "does not exist", "do not exist", "there is no", "no shared",
    "follow-up", "followup", "separate wp", "missing", "not yet",
    "would be", "would bind", "a new ", "no public", "deferred",
    "never landed", "never existed", "was never", "never gained",
    # Negation of a specifically-named item: "No `vokra_eval::dnsmos` module
    # exists.", "that crate has no `dnsmos` module".
    "no `", "has no ",
    # The rejected-design / relocated-binder shape: "It landed here instead",
    # "as originally drawn creates a dependency cycle", "the sidecar's
    # docstring anticipates ...".
    "instead", "originally", "anticipat", "dependency cycle",
]

# Claim-of-existence phrases. These are the recurring defect itself, so they
# are a HARD FAIL next to an unresolved citation — ledger membership and a
# marker elsewhere in the paragraph do not excuse them.
#
# The first three came from round 5's `vokra_ops::qwen2` sweep. The last three
# are round 6's verbatim wordings, kept so the exact defects this round fixed
# cannot come back:
#   "Consumed by `vokra_models::rnnoise_v02`."                  -> consumed by
#   "The runtime side is `vokra_eval::dnsmos::{p808_score,...}`" -> runtime side is
#   "The runtime binder lives in the `vokra-eval` crate (...)"   -> lives in the
#
# Phrases are kept narrow and unambiguous. "lives in the" is safe next to the
# two legitimate "Why this lives in `vokra-models`, not `vokra-eval`" headings
# because those spell a crate name in backticks, not "the".
ANTI_MARKERS = [
    "already wired", "already landed", "already available", "already exists",
    "already provides", "already provided", "already in place", "already shared",
    "is wired through", "are wired through", "wired through",
    "consumed by", "runtime side is", "lives in the",
]

# Window sizes in CHARACTERS around the citation, not lines: a line window is
# too coarse for prose that wraps at ~70 columns, and was what let round 5's
# historical defect through.
ANTI_WINDOW   = 140   # tight: the claim must attach to THIS citation
MARKER_WINDOW = 320   # looser: the surrounding sentence or two

# ---- discover crates + their public surfaces --------------------------------
def crate_roots():
    """`vokra_<snake>` -> path of the crate's root module (lib.rs, else main.rs).

    Discovery is from the filesystem so C ABI enum types that merely look like
    namespaces (`vokra_status_t::`, `vokra_aec_status_t::`,
    `vokra_event_kind_t::`) are never treated as crates."""
    out = {}
    for base in ("crates", "integrations"):
        d = os.path.join(surface_root, base)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.startswith("vokra-"):
                continue
            ident = name.replace("-", "_")
            lib  = os.path.join(d, name, "src", "lib.rs")
            main = os.path.join(d, name, "src", "main.rs")
            if os.path.isfile(lib):
                out[ident] = lib
            elif os.path.isfile(main):
                # Binary-only crate: nothing is importable, so nothing resolves.
                out[ident] = main
    return out

def public_surface(path):
    """`pub mod` names + crate-root `pub use` / `pub` item names.

    Deliberately does NOT credit the path PREFIX of a `pub use a::{b, c};`:
    that re-exports `b` and `c` at the root, not the module `a`. Crediting `a`
    would have made `vokra_backend_cpu::dispatch` resolve through
    `pub use dispatch::active_isa;` and hidden this round's private-module
    findings."""
    mods, roots = set(), set()
    try:
        lib = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return mods, roots
    # `pub mod x;` and inline `pub mod x { ... }`
    for m in re.finditer(r"^\s*pub mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*[;{]", lib, re.M):
        mods.add(m.group(1))
    # `pub use` — brace aware, so multi-line re-export blocks are captured.
    i = 0
    while True:
        m = re.search(r"^[ \t]*pub use\s+", lib[i:], re.M)
        if not m:
            break
        start, depth, j = i + m.end(), 0, i + m.end()
        while j < len(lib):
            c = lib[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            elif c == ";" and depth == 0:
                break
            j += 1
        stmt, i = lib[start:j], j + 1
        if "{" in stmt and "}" in stmt:
            names = stmt[stmt.index("{") + 1: stmt.rindex("}")].split(",")
        else:
            names = [stmt]
        for part in names:
            part = part.strip()
            if not part:
                continue
            if " as " in part:
                part = part.split(" as ")[-1]
            part = part.split("::")[-1].strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part):
                roots.add(part)
    for m in re.finditer(
        r"^\s*pub\s+(?:unsafe\s+)?(?:extern\s+\"[^\"]*\"\s+)?"
        r"(?:fn|struct|enum|trait|const|static|type|union)\s+"
        r"([A-Za-z_][A-Za-z0-9_]*)", lib, re.M):
        roots.add(m.group(1))
    return mods, roots

CRATES  = crate_roots()
SURFACE = {ns: public_surface(p) for ns, p in CRATES.items()}

def resolves(ns, seg):
    mods, roots = SURFACE[ns]
    if seg.endswith("*"):                       # wildcard family
        pre = seg[:-1]
        return any(n.startswith(pre) for n in mods | roots)
    return seg in mods or seg in roots

# ---- scan -------------------------------------------------------------------
CITE = re.compile(r"\b(vokra_[a-z0-9_]+)::([A-Za-z_][A-Za-z0-9_]*)(\*?)")
# An identifier split across a doc-comment line break, e.g.
#   //! the runtime `vokra_models::dnsmos_
#   //! p808_p835` binder ...
# Rejoined so the truncated stem is neither reported nor able to hide a phantom.
WRAP = re.compile(r"_\n[ \t]*(?://[!/]?|\*|#)[ \t]*(?=[A-Za-z0-9_])")

def prose(s):
    """Strip comment leaders and collapse whitespace, so a marker phrase is
    still found when it wraps across a line ("does not\\n /// exist")."""
    s = re.sub(r"(?m)^[ \t]*(?://[!/]?|\*|#)[ \t]?", " ", s)
    return re.sub(r"\s+", " ", s).lower()

def normalize(text):
    """Glue wrapped identifiers; return (text, line_of_offset)."""
    out, line_of, i, line = [], [], 0, 1
    while i < len(text):
        m = WRAP.match(text, i)
        if m:
            out.append("_"); line_of.append(line)
            line += text.count("\n", m.start(), m.end())
            i = m.end()
            continue
        c = text[i]
        out.append(c); line_of.append(line)
        if c == "\n":
            line += 1
        i += 1
    return "".join(out), line_of

violations = []
counts     = {}
scan_dirs = os.environ.get("VOKRA_CITE_SCAN_DIRS", "crates integrations tools").split()
exts = tuple(os.environ.get("VOKRA_CITE_SCAN_EXTS", ".rs .md .py").split())
for scan_root in [os.path.join(root, d) for d in scan_dirs]:
  for dirpath, dirnames, filenames in os.walk(scan_root):
    dirnames[:] = [d for d in dirnames if d not in ("target", ".git", ".venv")]
    for fn in sorted(filenames):
        if not fn.endswith(exts):
            continue
        path = os.path.join(dirpath, fn)
        rel  = os.path.relpath(path, root)
        try:
            raw = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        if "vokra_" not in raw:
            continue
        text, line_of = normalize(raw)
        for m in CITE.finditer(text):
            ns  = m.group(1)
            seg = m.group(2) + m.group(3)
            if ns in SKIP_NAMESPACES or ns not in SURFACE:
                continue                      # sibling gate's, or not a crate
            counts[ns] = counts.get(ns, 0) + 1
            if resolves(ns, seg):
                continue
            bare = seg.rstrip("*")
            key  = f"{ns}::{bare}"
            ln   = line_of[m.start()]
            anti_ctx = prose(text[max(0, m.start() - ANTI_WINDOW):
                                  min(len(text), m.end() + ANTI_WINDOW)])
            hit = next((a for a in ANTI_MARKERS if a in anti_ctx), None)
            if hit:
                violations.append((rel, ln, key,
                    f'is claimed as "{hit}" but does not resolve to any public '
                    f"item of `{ns}` — say where the code actually lives"))
                continue
            if key not in ledger:
                mods, roots = SURFACE[ns]
                extra = ""
                if not mods and not roots:
                    extra = (f" (`{ns}` is a binary-only crate — it has no "
                             "importable namespace at all; cite a file path)")
                violations.append((rel, ln, key,
                    "does not resolve to a public item of "
                    f"`{ns}` and is not in the exception ledger{extra}"))
                continue
            marker_ctx = prose(text[max(0, m.start() - MARKER_WINDOW):
                                    min(len(text), m.end() + MARKER_WINDOW)])
            if not any(k in marker_ctx for k in MARKERS):
                violations.append((rel, ln, key,
                    "is a ledgered unlanded path but this occurrence carries "
                    "no honesty marker (say PROPOSED / no such module / instead)"))

if os.environ.get("VOKRA_CITE_REPORT_COUNTS"):
    for ns in sorted(counts):
        bad = sum(1 for v in violations if v[2].startswith(ns + "::"))
        print(f"  {ns:26s} {counts[ns]:5d} citations, {bad} unresolved")

if violations:
    print("check-crate-path-citations: FAIL", file=sys.stderr)
    print("", file=sys.stderr)
    for rel, ln, key, why in violations:
        print(f"  {rel}:{ln}: `{key}` {why}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Fix the prose to name where the code actually lives, or add the", file=sys.stderr)
    print("`ns::ident` pair to EXPECTED_UNLANDED in", file=sys.stderr)
    print("scripts/check-crate-path-citations.sh with a rationale AND an", file=sys.stderr)
    print("explicit marker at each mention.", file=sys.stderr)
    sys.exit(1)
sys.exit(0)
PYEOF

# --- self-test ---------------------------------------------------------------
# Proves the gate has teeth in BOTH directions, against a synthetic tree so the
# real sources are never mutated:
#   1. unresolved + unmarked                 -> FAIL (the recurring defect)
#   2. unresolved + marked, not in ledger    -> FAIL (ledger is mandatory)
#   3. unresolved + marked + ledgered        -> PASS (no false positive)
#   4. resolved                              -> PASS (no false positive)
#   5. ledgered but unmarked                 -> FAIL (marker is mandatory)
#   6. line-wrapped identifier, unmarked     -> FAIL (wrap must not hide it)
#   7. round-6 verbatim "Consumed by"        -> FAIL (regression lock)
#   8. the honest wording, same ident        -> PASS (no false positive)
#   9. C ABI enum type is NOT a namespace    -> PASS (115 real occurrences)
#  10. private module does not resolve       -> FAIL (this round's 2nd shape)
#  11. crate-root re-export of the same fn   -> PASS (the corrected form)
#  12. binary-only crate resolves nothing    -> FAIL, with the right advice
#  13. wrong depth (`vokra_core::chunks`)    -> FAIL (real round-6 finding)
#  14. the corrected depth                   -> PASS (the corrected form)
#  15. ledger is per-namespace, not global   -> FAIL (ns::ident, not ident)
#  16. Python sidecar is in reach            -> FAIL (two real defects lived there)
#
# Cases 7 and 13 are the ones that matter most: they are verbatim round-6
# defects, so a future refactor that weakens the gate fails here by name.
# Cases 10/11 and 13/14 come in pairs on purpose — a gate that only ever
# says "no" can be satisfied by deleting the prose, so each FAIL case is
# paired with the corrected wording that must still PASS.
#
# The case count printed at the end is COUNTED, not hard-coded, so adding a
# case here cannot leave a stale "N/N passed" claim behind.
run_self_test() {
    local tmp status fails=0 ran=0
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' RETURN

    mkdir -p "$tmp/crates/probe/src"

    check_case() {
        local name="$1" expect="$2" body="$3" ledger="$4" fn="${5:-lib.rs}"
        rm -f "$tmp/crates/probe/src/"*.rs "$tmp/crates/probe/src/"*.py
        printf '%s\n' "$body" > "$tmp/crates/probe/src/$fn"
        set +e
        VOKRA_CITE_SCAN_ROOT="$tmp" VOKRA_CITE_SURFACE_ROOT="$ROOT" \
        VOKRA_CITE_LEDGER="$ledger" \
            python3 -c "$PY_CORE" >/dev/null 2>&1
        status=$?
        set -e
        ran=$((ran + 1))
        if [ "$status" -eq "$expect" ]; then
            echo "  ok   $name (exit $status)"
        else
            echo "  FAIL $name (exit $status, expected $expect)"
            fails=$((fails + 1))
        fi
    }

    echo "self-test: proving detection power"
    check_case "unresolved + unmarked is caught" 1 \
        '//! binds the vokra_models::ghost_binder primitive.' ""
    check_case "unresolved + marked but unledgered is caught" 1 \
        '//! vokra_models::ghost_binder is PROPOSED; no such module exists.' ""
    check_case "unresolved + marked + ledgered passes" 0 \
        '//! vokra_models::ghost_binder is PROPOSED; no such module exists.' \
        "vokra_models::ghost_binder"
    check_case "resolved citation passes" 0 \
        '//! reuses the vokra_models::nsnet2 denoise binder.' ""
    check_case "ledgered but unmarked is still caught" 1 \
        '//! binds the vokra_models::ghost_binder primitive.' \
        "vokra_models::ghost_binder"
    check_case "line-wrapped identifier is rejoined" 1 \
        '//! the runtime vokra_models::ghost_
//! binder is not bound.' ""

    # REGRESSION (round 6): the verbatim `vokra_models::rnnoise_v02` wording.
    # `rnnoise_v02` is IN the ledger here and an unrelated "follow-up" sits
    # nearby — the claim-of-existence layer must fail it anyway.
    check_case "round-6 'Consumed by' claim is caught" 1 \
        '// Xiph RNNoise v0.2 primitives (Vorbis window, Bark filterbank).
// Real-weight parity is a follow-up wave gated on the upstream manifest.
// Consumed by `vokra_models::rnnoise_v02`. Runtime function set, NOT
// `OpKind` variants.' \
        "vokra_models::rnnoise_v02"

    # The honest wording for the SAME identifier must still pass, or the gate
    # would simply forbid discussing unlanded work.
    check_case "honest 'no runtime binder' wording passes" 0 \
        '// No RNNoise runtime binder exists yet: `vokra_models::rnnoise_v02`
// does not exist. Outside vokra-ops the primitive set is exercised only
// by an env-gated parity harness.' \
        "vokra_models::rnnoise_v02"

    # A C ABI enum type is not a crate. There are 115 `vokra_status_t::`
    # occurrences in vokra-capi; flagging them would be pure noise.
    check_case "C ABI enum type is not treated as a namespace" 0 \
        '// assert_eq!(st, vokra_status_t::VOKRA_OK);
// assert_eq!(st, vokra_aec_status_t::VOKRA_AEC_OK);' ""

    # Round 6, second shape: `dispatch` is a REAL FILE but `mod dispatch;` is
    # private. Resolving against the public surface is what catches it.
    check_case "private module does not resolve" 1 \
        '//! see vokra_backend_cpu::dispatch::fused_log_mel_dispatch.' ""
    # ... and the crate-root re-export of the same function does.
    check_case "crate-root re-export of the same fn resolves" 0 \
        '//! see vokra_backend_cpu::fused_log_mel_dispatch.' ""

    # `vokra-cli` is binary-only: nothing is nameable, and the message says so.
    check_case "binary-only crate resolves nothing" 1 \
        '/// Mirrors `vokra_cli::bench::parse_backend` verbatim.' ""

    # Round 6: `vokra_core::chunks` is really `vokra_core::gguf::chunks`. The
    # first segment is the wrong one, which is exactly what depth 1 catches.
    check_case "wrong-depth first segment is caught" 1 \
        '/// Kept local rather than a `vokra_core::chunks::*` re-export.' ""
    check_case "the corrected depth resolves" 0 \
        '/// Kept local rather than a `vokra_core::gguf::chunks::*` re-export.' ""

    # The ledger is keyed by `ns::ident`, so ledgering one namespace must not
    # silently excuse the same bare identifier in another.
    check_case "ledger does not leak across namespaces" 1 \
        '//! vokra_core::ghost_binder is PROPOSED; no such module exists.' \
        "vokra_models::ghost_binder"

    # Two real round-6 defects lived in `tools/parity/*.py` sidecars.
    check_case "python sidecar is in reach" 1 \
        'the future ``vokra_eval::ghost_binder`` walks these prefixes' "" \
        "sidecar.py"

    if [ "$fails" -ne 0 ]; then
        echo "self-test FAILED ($fails of $ran case(s))" >&2
        return 1
    fi
    echo "self-test passed ($ran/$ran)"
    return 0
}

# --- main --------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ]; then
    run_self_test
    exit $?
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "check-crate-path-citations: python3 not found; cannot run" >&2
    exit 1
fi

# The `vokra_ops::` namespace is skipped here because the sibling gate owns it.
# If that script is ever deleted, this one must not go on silently pretending
# 1000+ `vokra_ops::` citations are covered.
if [ ! -f "$ROOT/$SIBLING_GATE" ]; then
    echo "check-crate-path-citations: FAIL" >&2
    echo "" >&2
    echo "  $SIBLING_GATE is missing." >&2
    echo "  This gate deliberately skips the \`vokra_ops::\` namespace because" >&2
    echo "  that script owns it. With the sibling gone, nothing checks it." >&2
    echo "  Either restore the sibling, or drop 'vokra_ops' from" >&2
    echo "  SKIP_NAMESPACES here and merge its ledger into EXPECTED_UNLANDED." >&2
    exit 1
fi

VOKRA_CITE_SCAN_ROOT="$ROOT" \
VOKRA_CITE_LEDGER="${EXPECTED_UNLANDED[*]}" \
    python3 -c "$PY_CORE"

echo "check-crate-path-citations: OK — every vokra_<crate>:: citation in crates/ integrations/ tools/ resolves"
echo "  (vokra_ops:: is covered by $SIBLING_GATE)"
echo "  (declared unlanded exceptions: ${EXPECTED_UNLANDED[*]})"
