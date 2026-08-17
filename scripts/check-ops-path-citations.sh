#!/usr/bin/env bash
# check-ops-path-citations.sh — every `vokra_ops::<ident>` named in a comment
# or a string literal under `crates/` must resolve to a real public item in
# `vokra-ops`, or be a declared exception that reads honestly.
#
# WHY THIS EXISTS (the phantom-primitive class)
# ---------------------------------------------------------------------------
# Five consecutive audit rounds found prose citing `vokra_ops::` modules that
# have never existed. The failure mode is not a typo — it is a comment that
# asserts a shared primitive is "already wired through" some sibling model,
# which sends the next reader hunting for a module to bind against. Round 4
# swept for this class but covered `vokra-models` only, so four `vokra_ops::qwen2`
# citations survived in `vokra-convert` (plus two `vokra_ops::beam_search`,
# whose beam search actually lives in `vokra_core::decode::beam_search`).
#
# A per-round manual sweep has now demonstrably failed five times. This is the
# mechanical replacement: a structural tripwire in the `check-zero-deps.sh`
# mould. It does NOT test behavior, it asserts an invariant that must not
# regress — prose may not claim an op exists when it does not.
#
# WHAT IT CHECKS (and, deliberately, what it does not)
# ---------------------------------------------------------------------------
# For every `vokra_ops::<seg1>` occurrence, `<seg1>` must be one of:
#   * a `pub mod` of `vokra-ops`, or
#   * an item re-exported / declared at the `vokra-ops` crate root.
# A trailing `*` is treated as a wildcard and prefix-matched (`fused_log_mel_*`
# legitimately names the `fused_log_mel_scalar` family).
#
# DEPTH IS DELIBERATELY LIMITED TO THE FIRST SEGMENT. Deeper segments would
# need real type resolution to check without crying wolf: the corpus contains
# associated functions (`Qwen3TtsCodecConfig::qwen3_tts_12hz`), enum variants
# (`ConvSubsampleKind::Stacking`), struct fields (`ViTAttrs::patch_w`), trait
# methods (`SineGen::forward`) and `#[cfg(test)]` modules
# (`fsmn_vad::tests::state_carry_matches_single_chunk`) — none of which a regex
# can distinguish from a missing function. The recurring defect has been at the
# first segment every time (a whole module that does not exist), which is
# exactly what this layer nails shut. Extending to depth 2 was prototyped and
# rejected: it flagged the legitimate `tests::` and associated-item citations.
#
# THE EXCEPTION LEDGER
# ---------------------------------------------------------------------------
# Naming a not-yet-landed op is legitimate and useful — a scaffold should say
# which primitive it is waiting on. Such a citation must clear TWO bars:
#   1. the identifier is listed in EXPECTED_UNLANDED below, with a rationale;
#   2. EVERY occurrence carries an explicit marker in its immediate context
#      ("PROPOSED", "no such module", "does not exist", "follow-up", ...) so
#      the prose cannot read as a claim that the op is already available.
# Bar 2 is what separates this from a plain allowlist: adding `qwen2` to the
# ledger would NOT have rescued the four converter comments, because they said
# "already wired through" rather than "proposed".
#
# SCOPE: `crates/`, `integrations/` and `tools/`, over `.rs` and `.md`. The
# `.md` reach is not decoration — a `tools/parity/firered_asr_llm_l/README.md`
# carried the same "already wired through voxtral / kyutai_stt / canary_qwen"
# claim as the converters, and would have survived a Rust-only sweep exactly the
# way `vokra-convert` survived round 4's `vokra-models`-only sweep.
#
# `docs/` IS DELIBERATELY EXCLUDED, and this is a judgement call worth stating
# rather than hiding. `docs/tickets/` (gitignored, local planning material) is a
# different genre: an audit ticket's job is to enumerate the ops a model WOULD
# need — `vokra_ops::gemm`, `vokra_ops::rvq_codec`, `vokra_ops::attention`,
# `vokra_ops::posterior_encoder` — none of which exist or are meant to yet.
# Gating that genre would either cry wolf on every planning note or force a
# ledger so large it stops meaning anything. The invariant this gate protects is
# about SHIPPED prose next to shipped code, which is where a reader goes hunting
# for a module to bind against. Run it over docs/ manually if ever wanted:
#   VOKRA_OPS_SCAN_DIRS=docs bash scripts/check-ops-path-citations.sh
#
# Local: `bash scripts/check-ops-path-citations.sh`
#        `bash scripts/check-ops-path-citations.sh --self-test`
# CI: runs in ci.yml's `license` job alongside the other structural tripwires.
#
# Exit code: 0 = every citation resolves or is a declared, honestly-marked
# exception; 1 = a phantom citation (or a self-test that failed to detect an
# injected phantom).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# --- the exception ledger ----------------------------------------------------
# Identifiers that may appear as `vokra_ops::<ident>` WITHOUT resolving, because
# they name a primitive that is proposed but not yet landed. Every occurrence
# must still carry an honesty marker (see MARKERS in the Python core below).
#
#   qwen2            — proposed consolidation of the Qwen2-family forward. The
#                      only landed one is inline in
#                      `vokra-models/src/voxtral/text_decoder.rs`; `canary_qwen`
#                      reuses it, `kyutai_stt` does not. Consolidating them
#                      unlocks voxtral / canary_qwen / kyutai_stt /
#                      firered_asr_llm_l / llama_omni2 together.
#   lstm             — no generic sequence-LSTM op. The one public LSTM is
#                      `vokra_ops::hybrid_ctc_attention::LstmLmCell` (LM-shaped:
#                      token id in, one log-probability out); Silero's is a
#                      pub(crate) fixed-width cell in the `vokra-vad-micro`
#                      crate. Blocks demucs / dtln_aec / gtcrn.
#   conv2d           — no public 2-D convolution of any kind. Cited by gtcrn
#                      precisely to record that absence.
#   wav2vec2_encoder — proposed shared wav2vec 2.0 encoder body. The stem is
#                      covered by `vokra_ops::waveform_frontend` and the search
#                      by `vokra_ops::ctc_decode`; the encoder body is not.
#   speculative_decode — runtime speculative-decoding op, a separate WP
#                      (cited by the whisper_medusa_v1 converter).
EXPECTED_UNLANDED=(
    "qwen2"
    "lstm"
    "conv2d"
    "wav2vec2_encoder"
    "speculative_decode"
)

# --- self-test ---------------------------------------------------------------
# Proves the gate has teeth in BOTH directions, against a synthetic tree so the
# real sources are never mutated:
#   1. unresolved + unmarked              -> must FAIL (the recurring defect)
#   2. unresolved + marked, not in ledger -> must FAIL (ledger is mandatory)
#   3. unresolved + marked + in ledger    -> must PASS (no false positive)
#   4. resolved                           -> must PASS (no false positive)
#   5. ledgered but unmarked              -> must FAIL (marker is mandatory)
#   6. line-wrapped identifier, unmarked  -> must FAIL (wrap must not hide it)
#   7. the verbatim historical wording    -> must FAIL (regression lock)
#   8. the honest wording, same ident     -> must PASS (no false positive)
#
# Case 7 is the one that matters most. An earlier draft of this gate PASSED the
# real historical defect: `qwen2` was in the ledger, and an unrelated "follow-up
# wave" sentence a few lines above satisfied a paragraph-scoped marker search.
# That draft would have shipped as a gate that never fires. The claim-of-
# existence layer and the character-scoped windows exist because of it, and case
# 7 keeps that hole closed.
run_self_test() {
    local tmp status fails=0
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' RETURN

    mkdir -p "$tmp/crates/probe/src"

    check_case() {
        local name="$1" expect="$2" body="$3" ledger="$4"
        printf '%s\n' "$body" > "$tmp/crates/probe/src/lib.rs"
        set +e
        VOKRA_OPS_SCAN_ROOT="$tmp" VOKRA_OPS_SURFACE_ROOT="$ROOT" \
        VOKRA_OPS_LEDGER="$ledger" \
            python3 -c "$PY_CORE" >/dev/null 2>&1
        status=$?
        set -e
        if [ "$status" -eq "$expect" ]; then
            echo "  ok   $name (exit $status)"
        else
            echo "  FAIL $name (exit $status, expected $expect)"
            fails=$((fails + 1))
        fi
    }

    echo "self-test: proving detection power"
    check_case "unresolved + unmarked is caught" 1 \
        '//! binds the vokra_ops::ghost_op primitive.' ""
    check_case "unresolved + marked but unledgered is caught" 1 \
        '//! vokra_ops::ghost_op is a PROPOSED name; no such module exists.' ""
    check_case "unresolved + marked + ledgered passes" 0 \
        '//! vokra_ops::ghost_op is a PROPOSED name; no such module exists.' "ghost_op"
    check_case "resolved citation passes" 0 \
        '//! reuses the vokra_ops::conformer encoder body.' ""
    check_case "ledgered but unmarked is still caught" 1 \
        '//! binds the vokra_ops::ghost_op primitive.' "ghost_op"
    # A wrapped identifier must be rejoined before resolution, otherwise the
    # truncated stem would be reported (or silently tolerated).
    check_case "line-wrapped identifier is rejoined" 1 \
        '//! the runtime vokra_ops::speculative_
//! decode op is not bound.' ""

    # REGRESSION: the verbatim historical defect. `qwen2` is IN the ledger and an
    # unrelated "follow-up wave" sits a few lines above, which is exactly what
    # defeated the first (paragraph-scoped, marker-only) version of this gate.
    # The claim-of-existence layer must fail it regardless.
    check_case "historical 'already wired through' claim is caught" 1 \
        '//! GGUF tensor names are the upstream safetensors names verbatim.
//! Real-weight parity is a follow-up wave gated on the upstream
//! tensor-name manifest fetch + §3.1 sign-off; this converter passes
//! every float tensor through unchanged so a future
//! `LlamaOmni2Weights::from_gguf` can walk the same names (Qwen2.5
//! forward shares the `vokra_ops::qwen2` primitives already wired
//! through voxtral / kyutai_stt / canary_qwen).' \
        "qwen2 lstm conv2d wav2vec2_encoder speculative_decode"

    # The honest wording for the SAME identifier must still pass, or the gate
    # would simply forbid discussing unlanded work.
    check_case "honest 'PROPOSED, not landed' wording passes" 0 \
        '//! `vokra_ops::qwen2` is a PROPOSED consolidation, not a landed
//! module — no such module exists today. The only landed Qwen2-family
//! forward is inline in vokra-models/src/voxtral/text_decoder.rs.' \
        "qwen2 lstm conv2d wav2vec2_encoder speculative_decode"

    if [ "$fails" -ne 0 ]; then
        echo "self-test FAILED ($fails case(s))" >&2
        return 1
    fi
    echo "self-test passed (8/8)"
    return 0
}

# --- the checker core --------------------------------------------------------
# Reads: VOKRA_OPS_SCAN_ROOT (repo root to scan), VOKRA_OPS_LEDGER (space-
# separated identifiers). Kept in one string so the self-test can re-run it
# against a synthetic tree without duplicating the logic.
PY_CORE=$(cat <<'PYEOF'
import os, re, sys

root   = os.environ["VOKRA_OPS_SCAN_ROOT"]
ledger = set(os.environ.get("VOKRA_OPS_LEDGER", "").split())
# The public surface is always read from the REAL vokra-ops so the self-test can
# scan a synthetic tree while still resolving against the true module list.
surface_root = os.environ.get("VOKRA_OPS_SURFACE_ROOT", root)
ops_src = os.path.join(surface_root, "crates", "vokra-ops", "src")

# Honesty markers. At least one must appear in the citation's own neighbourhood
# for an unresolved citation to read as "proposed" rather than "available".
MARKERS = [
    "proposed", "no such module", "not a landed", "not landed",
    "does not exist", "do not exist", "there is no", "no shared",
    "follow-up", "followup", "separate wp", "missing", "not yet",
    "would be", "would bind", "a new ", "no public", "deferred",
    "never landed", "never existed", "was never",
]

# Claim-of-existence phrases. These are the recurring defect itself, so they are
# a HARD FAIL next to an unresolved citation — ledger membership and a marker
# elsewhere in the paragraph do not excuse them.
#
# This layer exists because the marker check alone was empirically insufficient:
# the historical `vokra_ops::qwen2` comments sat a few lines below an unrelated
# "Real-weight parity is a follow-up wave" sentence, whose "follow-up" satisfied
# a paragraph-scoped marker search. The gate was re-tested against the verbatim
# original wording and now fails it. Phrases are kept narrow and unambiguous —
# "reuses"/"shares" are deliberately NOT here, because a comment may legitimately
# say it reuses op A in the same breath as noting op B does not exist (the
# omniasr_ctc / wav2vec2_encoder case).
ANTI_MARKERS = [
    "already wired", "already landed", "already available", "already exists",
    "already provides", "already provided", "already in place", "already shared",
    "is wired through", "are wired through", "wired through",
]

# Window sizes in CHARACTERS around the citation, not lines: a line window is
# too coarse for prose that wraps at ~70 columns, and was what let the historical
# defect through.
ANTI_WINDOW   = 140   # tight: the claim must attach to this citation
MARKER_WINDOW = 320   # looser: the surrounding sentence or two

# ---- public surface of vokra-ops -------------------------------------------
def public_surface():
    mods, roots = set(), set()
    lib_path = os.path.join(ops_src, "lib.rs")
    if not os.path.isfile(lib_path):
        return mods, roots            # synthetic tree (self-test): nothing resolves
    lib = open(lib_path, encoding="utf-8").read()
    for m in re.finditer(r"^\s*pub mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*;", lib, re.M):
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
        names = []
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
        r"^\s*pub\s+(?:fn|struct|enum|trait|const|static|type|union)\s+"
        r"([A-Za-z_][A-Za-z0-9_]*)", lib, re.M):
        roots.add(m.group(1))
    return mods, roots

MODS, ROOTS = public_surface()

def resolves(seg):
    if seg.endswith("*"):                       # wildcard family, e.g. fused_log_mel_*
        pre = seg[:-1]
        return any(n.startswith(pre) for n in MODS | ROOTS)
    return seg in MODS or seg in ROOTS

# ---- scan -------------------------------------------------------------------
CITE = re.compile(r"vokra_ops::([A-Za-z_][A-Za-z0-9_]*)(\*?)")
# An identifier split across a doc-comment line break, e.g.
#   //! the runtime `vokra_ops::speculative_
#   //! decode` op ...
# Rejoined so the truncated stem is neither reported nor able to hide a phantom.
WRAP = re.compile(r"_\n[ \t]*(?://[!/]?|\*)[ \t]*(?=[A-Za-z0-9_])")

def prose(s):
    """Strip comment leaders and collapse whitespace, so a marker phrase is
    still found when it wraps across a line ("does not\\n /// exist")."""
    s = re.sub(r"(?m)^[ \t]*(?://[!/]?|\*)[ \t]?", " ", s)
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
scan_dirs = os.environ.get("VOKRA_OPS_SCAN_DIRS", "crates integrations tools").split()
exts = tuple(os.environ.get("VOKRA_OPS_SCAN_EXTS", ".rs .md").split())
walk_roots = [os.path.join(root, d) for d in scan_dirs]
for scan_root in walk_roots:
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
        if "vokra_ops::" not in raw:
            continue
        text, line_of = normalize(raw)
        for m in CITE.finditer(text):
            seg = m.group(1) + m.group(2)
            if resolves(seg):
                continue
            bare = seg.rstrip("*")
            ln = line_of[m.start()]
            anti_ctx = prose(text[max(0, m.start() - ANTI_WINDOW):
                                  min(len(text), m.end() + ANTI_WINDOW)])
            hit = next((a for a in ANTI_MARKERS if a in anti_ctx), None)
            if hit:
                violations.append((rel, ln, bare,
                    f'is claimed as "{hit}" but does not resolve to any public '
                    "vokra-ops item — say where the code actually lives"))
                continue
            if bare not in ledger:
                violations.append((rel, ln, bare,
                    "does not resolve to a public vokra-ops item and is not "
                    "in the exception ledger"))
                continue
            marker_ctx = prose(text[max(0, m.start() - MARKER_WINDOW):
                                    min(len(text), m.end() + MARKER_WINDOW)])
            if not any(k in marker_ctx for k in MARKERS):
                violations.append((rel, ln, bare,
                    "is a ledgered unlanded op but this occurrence carries no "
                    "honesty marker (say PROPOSED / no such module / follow-up)"))

if violations:
    print("check-ops-path-citations: FAIL", file=sys.stderr)
    print("", file=sys.stderr)
    for rel, ln, seg, why in violations:
        print(f"  {rel}:{ln}: `vokra_ops::{seg}` {why}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Fix the prose to name where the code actually lives, or add the", file=sys.stderr)
    print("identifier to EXPECTED_UNLANDED in scripts/check-ops-path-citations.sh", file=sys.stderr)
    print("with a rationale AND an explicit marker at each mention.", file=sys.stderr)
    sys.exit(1)
sys.exit(0)
PYEOF
)

# --- main --------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ]; then
    run_self_test
    exit $?
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "check-ops-path-citations: python3 not found; cannot run" >&2
    exit 1
fi

VOKRA_OPS_SCAN_ROOT="$ROOT" \
VOKRA_OPS_LEDGER="${EXPECTED_UNLANDED[*]}" \
    python3 -c "$PY_CORE"

echo "check-ops-path-citations: OK — every vokra_ops:: citation in crates/ integrations/ tools/ resolves"
echo "  (declared unlanded exceptions: ${EXPECTED_UNLANDED[*]})"
