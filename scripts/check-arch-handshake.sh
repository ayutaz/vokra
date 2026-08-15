#!/usr/bin/env bash
# check-arch-handshake.sh — the converter <-> binder arch handshake gate.
#
# WHY THIS GATE EXISTS
#   A model arch tag is a contract between two crates. `vokra-convert` STAMPS
#   `vokra.model.arch = "X"` into the GGUF it writes; `vokra-models` READS that
#   stamp back (the repo's own convention is a strict
#   `vokra.model.arch == "X"` verification in `from_gguf`, so a foreign GGUF
#   fails loudly instead of misrouting into a wrong-shape forward, FR-EX-08).
#   Either half without the other is a dead end:
#
#     - a converter with no reader  -> the tool happily produces a GGUF that
#       nothing in the workspace can load, and the stamp is decoration;
#     - a binder with no converter  -> the loader can only ever be fed a GGUF
#       this repo has no way to produce.
#
#   `scripts/check-bound-arch-coverage.sh` (its sibling) watches a DIFFERENT
#   edge: binder -> `crates/vokra-cli/src/engine.rs` BOUND_ARCHES, i.e. "does
#   the CLI tell the truth about what it binds". Neither direction of the
#   converter/binder handshake was watched by anything, and that is how two
#   live problems survived undetected:
#
#     - `voila`: a landed binder (`vokra-models/src/voila/mod.rs`) with no
#       converter anywhere. RESOLVED 2026-08-15 in the same wave that added
#       this gate — `vokra-convert/src/models/voila.rs` landed, so the
#       NO_CONVERTER ledger entry went stale and the double-sided check
#       failed until it was deleted. That is the ledger working as designed:
#       a closed gap that stays listed is as much a lie as an unlisted one;
#     - 21 converter arch tags with no reader at all.
#
#   Several of those 21 are legitimately converter-only — publish-only BF16
#   pass-throughs, an ELVIS-Act voice-clone that belongs in a separate repo, a
#   redistribution-forbidden weight, vast.ai-gated multi-shard giants. That is
#   a fine state to be in. The problem is that it was recorded NOWHERE, so
#   every audit rediscovered all 21 as "gaps" — which is exactly how a real
#   regression gets lost in the noise.
#
# THE THREE LEGS
#   (a) converter -> reader
#       Every `pub const ARCH…: &str = "X"` under
#       `crates/vokra-convert/src/models/` must be answered by a reader:
#       the literal "X" appearing in non-comment source under
#       `crates/vokra-models/src/`, OR "X" being routed / registered in
#       `crates/vokra-cli/src/engine.rs` (a routed `const ARCH_*` or a
#       BOUND_ARCHES row — both of which assert a binder exists, and the
#       sibling gate keeps that assertion honest from the other side).
#
#   (b) binder -> converter
#       Every `pub const ARCH…: &str = "X"` under
#       `crates/vokra-models/src/` must be emitted by some converter: the
#       literal "X" appearing in non-comment source under
#       `crates/vokra-convert/src/`.
#
#   Comment lines are stripped before literals are collected, deliberately: a
#   doc-comment that merely NAMES an arch is not a reader and not an emitter,
#   and must not be able to satisfy this gate.
#
#   (c) recovery command -> CLI parser            [added 2026-08-15]
#       Every `convert --model <slug>` that appears anywhere under
#       `crates/vokra-models/src/` must parse through
#       `ModelKind::from_arg` in `crates/vokra-convert/src/lib.rs`.
#
#       WHY: a binder that rejects a GGUF is supposed to hand the operator
#       an actionable next step, and the repo's convention is to print the
#       converter invocation that would produce an acceptable artifact.
#       Ten such messages across four modules (mt3, beat_this, redimnet,
#       llama_omni2) named slugs `from_arg` had never accepted — following
#       the instruction produced `unknown model`, which is strictly worse
#       than printing nothing: it teaches the operator that the error text
#       is not to be trusted. The 2026-08-15 wave that closed those four
#       also found `kokoro-82m` (4 sites) and `speaker-3d-eres2net` (3
#       sites) in the same state. This leg is what would have caught all
#       six, and what keeps a renamed slug from re-opening the hole.
#
#       Comments are NOT stripped for this leg — the opposite of (a)/(b),
#       on purpose. A doc comment that tells a reader to run a command is
#       making the same promise a runtime `format!` string is; the earlier
#       legs strip comments because a comment cannot *implement* anything,
#       whereas here the comment IS the artifact under test.
#
#       Line joining: a slug is often split across physical lines, either
#       by a Rust string continuation (`--model \` + newline) or by a
#       doc-comment wrap (`--model` at end of line, slug on the next
#       `//!` line). Both are rejoined before matching. A `convert
#       --model` left with NO slug after rejoining is a failure in its own
#       right: the message names no command at all.
#
#       Templates are skipped, and each skip is a documented shape, not a
#       catch-all: `bigvgan-*` / `sepformer-*` (glob family notation),
#       `magnet-{}` / `sepformer-{tag}` (a `format!` placeholder), `<arg>`
#       (an angle-bracketed metavariable). Brace alternations ARE expanded
#       and checked -- `moonshine-{tiny,base}` becomes two candidates,
#       `musicgen-{small|medium}` two more -- as is a bare `mimi|dac`,
#       because no accepted `--model` value contains `|`, `<`, `*`, `{` or
#       a space, so those characters can only be notation.
#
# THE LEDGERS ARE DOUBLE-SIDED
#   Known, accepted gaps live in `NO_READER` / `NO_CONVERTER` below with a real
#   reason each. Exactly like `EXPECTED_GAPS` in
#   `scripts/publish/check-catalog-reality.sh`, the gate fails BOTH ways:
#     - a gap that is NOT in the ledger        -> new drift, fail;
#     - a ledger entry that is no longer a gap -> stale ledger, fail.
#   A one-sided ledger rots: entries outlive the condition they described and
#   the file slowly becomes a list of claims nobody has checked in a year.
#
# Zero-dep: bash + python3 stdlib only (no jq, no pip, no cargo). Not a Vokra
# runtime dep.
# Exit: 0 = all three legs clean, 1 = an undeclared gap / a stale ledger entry /
# an unparseable recovery-command slug / a parser guard trip / a bad argument.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONVERT_MODELS_DEFAULT="$ROOT/crates/vokra-convert/src/models"
CONVERT_SRC_DEFAULT="$ROOT/crates/vokra-convert/src"
MODELS_DEFAULT="$ROOT/crates/vokra-models/src"
ENGINE_DEFAULT="$ROOT/crates/vokra-cli/src/engine.rs"
# Leg (c) reads `ModelKind::from_arg` out of this file. It is the CLI
# parser both front-ends (`vokra-cli convert` and the standalone
# `vokra-convert`) call, so it is the single authority on which `--model`
# spellings exist.
CONVERT_LIB_DEFAULT="$ROOT/crates/vokra-convert/src/lib.rs"

# ---------------------------------------------------------------------------
# LEDGER (a): converter arch tags with no reader.
#
# Format: 'arch|reason'. The arch is the STAMPED VALUE (the string literal on
# the right of `pub const ARCH`), not the module file name — several diverge
# (`naturalspeech3_facodec.rs` stamps `facodec`, `xtts_v2.rs` stamps `xtts`,
# `qwen2_5_omni_7b.rs` stamps `qwen2-omni`, `ultravox_v0_5_llama_3_2_1b.rs`
# stamps `ultravox`), and the gate compares stamped values.
#
# Reasons name the ACTUAL cause. "TODO" is not a reason: if nobody can say why
# the gap is acceptable, it is not an accepted gap, it is an open defect.
# ---------------------------------------------------------------------------
declare -a NO_READER=(
  'audioseal_real_weight|publish-only: weights ship as vokra/audioseal-real-weight (MIT), but the Generator+Detector runtime binder is gated on the M5-05 T04 watermark ADR, which is owner-pending (converter states this at audioseal_real_weight.rs:185).'
  'facodec|publish-only BF16 pass-through (naturalspeech3_facodec.rs). Runtime binder + real-weight parity are a post-signoff follow-up on the RMVPE / Charsiu loud-partial precedent; the redecoder variants additionally await an owner ELVIS-Act routing decision (main zoo vs voiceclone-experimental) because they enable timbre swapping.'
  'focalcodec|publish-only BF16 pass-through. Header reserves the tag for a future native FocalCodec loader; no binder module exists yet.'
  'freevc|ELVIS Act separation (CLAUDE.md design decision 8): any-to-any voice conversion belongs in the vokra-voiceclone-experimental repo, and license-audit.md:314 marks the row explicitly out of main-repo section 3.1 scope. A main-repo binder is forbidden by policy, not merely absent.'
  'granite_speech|awaiting a binder: the converter header reserves crates/vokra-models/src/granite_speech/ for it. Input is a 4.87 GB three-shard release the owner pre-merges offline, so the binder has not been started.'
  'higgs_audio_v3_tts_4b|publish-forbidden: BOSON HIGGS TTS 3 R and NC is LicenseClass::RedistributionForbidden (section II-A(c) bans redistribution, hosting and embedding). The converter exists for local owner use only; no publish and no binder follow.'
  'magpietts_v2602|publish-only BF16 pass-through (NVIDIA NeMo .nemo flattened to safetensors offline). No runtime binder; the tag is reserved for a future TTS forward.'
  'miocodec|publish-only BF16 pass-through. Header reserves the tag for a future native MioCodec runtime side; no binder module exists yet.'
  'moss_audio_tokenizer|publish-only BF16 pass-through; the codec half of the MOSS-TTS pipeline. Header reserves the tag for a future native loader; no binder module exists yet.'
  'nemotron-speech-streaming-v2603|publish-only BF16 pass-through. Header names a future vokra-models::nemotron_speech_streaming_v2603 implementation; the streaming FastConformer forward is unwritten.'
  'neucodec|publish-only BF16 pass-through (2.35 GB base plus the distill sibling). Header reserves the tag for a future native Neucodec loader.'
  'neutts-air|publish-only BF16 pass-through. Header (neutts_air.rs:119) defers the arch-tag verification and the runtime binder to the same later wave.'
  'qwen2-omni|vast.ai-gated (22.37 GB, five-shard Thinker+Talker) AND publish-blocked by the GGUF writer 5D-tensor limit that the multimodal adapter trips. No binder until that reshape-vs-extend decision lands.'
  'qwen2_audio|vast.ai-gated (~16 GB, five-shard). Owner runbook is required before a first conversion even runs, so no binder work has started.'
  'sgmse|publish-only BF16 pass-through. Header states that real-weight parity and a native Sgmse::from_gguf forward are a follow-up.'
  'ultravox|awaiting a binder: local convert is safe at ~1.83 GB, and the converter header records the runtime binder as a follow-up. Nothing blocks it but wave ordering.'
  'vibevoice_asr|vast.ai-gated (~16.5 GB, eight-shard). The sibling TTS vibevoice is published; the ASR head has neither been converted nor bound.'
  'wavtokenizer|no arch-tag binder: the M4-16 wavtokenizer_vq op in vokra-ops/src/fsq_codec.rs is a GENERIC FSQ op that never reads the vokra.model.arch stamp, so nothing dispatches on this tag. A WavTokenizer::from_gguf is still owed.'
  'xtts|T4 Research-only: the Coqui Public Model License maps to LicenseClass::NonCommercial, so publish requires --allow-noncommercial. It is also zero-shot voice cloning, which keeps it out of a main-repo binder under design decision 8.'
  'yue_upsampler|publish-only BF16 pass-through: the 145 MB Vocos plus iSTFT vocoder half of the YuE bundle. Header reserves the tag for a future native YuE loader.'
  'yue_xcodec_mini|publish-only BF16 pass-through: the 2.2 GB SoundStream RVQ codec half of the YuE bundle. Same future loader as its yue_upsampler sibling.'
)

# ---------------------------------------------------------------------------
# LEDGER (b): binder arch tags no converter emits.
# ---------------------------------------------------------------------------
declare -a NO_CONVERTER=(
)

# ---------------------------------------------------------------------------
# LEDGER (c): `convert --model X` strings under crates/vokra-models/src/ that
# are deliberately NOT real CLI slugs.
#
# Empty is the goal state and the current state. The notation shapes the
# extractor already understands (globs, `format!` placeholders, `<meta>`
# variables, brace and pipe alternations) do not belong here — they are
# handled in the parser, so an entry here means something stronger: a
# literal-looking slug that is intentionally unrunnable. If you find
# yourself adding one, check first that rewriting the message is not the
# better fix, because the whole point of leg (c) is that these strings are
# instructions an operator will actually type.
#
# Format: 'slug|reason', same as the ledgers above, and equally
# double-sided: an unparseable slug not listed here fails, and a listed
# slug that now parses fails as stale.
# ---------------------------------------------------------------------------
declare -a NOT_A_MODEL_SLUG=(
)

usage() {
    cat <<'USAGE'
check-arch-handshake.sh — converter <-> binder arch handshake gate

Usage:
  bash scripts/check-arch-handshake.sh
  bash scripts/check-arch-handshake.sh --help
  bash scripts/check-arch-handshake.sh --self-test

Leg (a): every `pub const ARCH…: &str` under crates/vokra-convert/src/models/
is answered by a reader — the arch literal in non-comment source under
crates/vokra-models/src/, or a routed constant / BOUND_ARCHES row in
crates/vokra-cli/src/engine.rs.

Leg (b): every `pub const ARCH…: &str` under crates/vokra-models/src/ is
emitted by some converter — the arch literal in non-comment source under
crates/vokra-convert/src/.

Leg (c): every `convert --model <slug>` printed anywhere under
crates/vokra-models/src/ (comments included) parses through
ModelKind::from_arg in crates/vokra-convert/src/lib.rs. A binder that tells an
operator to run a command that does not exist is worse than one that stays
silent.

Accepted gaps live in the NO_READER / NO_CONVERTER / NOT_A_MODEL_SLUG ledgers at
the top of this script, one reason each. All three are double-sided: an
undeclared gap fails, and a ledger entry whose gap has since been closed also
fails. Exit 1 on any.
USAGE
}

# The checker. Args:
#   $1 convert models dir (arch constants are declared here)
#   $2 convert src dir    (emitter literals are searched here)
#   $3 models src dir     (binder arch constants AND reader literals AND the
#                          `convert --model` recovery commands leg (c) reads)
#   $4 engine.rs path     (routed constants + BOUND_ARCHES rows)
#   $5 ledger file for leg (a), $6 ledger file for leg (b), $8 ledger file for
#      leg (c); all 'key|reason' per line, blank lines and #-comments ignored.
#   $7 convert lib.rs path (leg (c) reads `ModelKind::from_arg` out of it)
# stdlib only. Reused verbatim by the main path and --self-test.
run_check() {
    python3 - "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" <<'PY'
import os, re, sys

(conv_models, conv_src, models_dir, engine_path, ledger_a, ledger_b,
 convert_lib, ledger_c) = sys.argv[1:9]

ARCH_CONST = re.compile(
    r'^\s*pub\s+const\s+(ARCH(?:_[A-Z0-9_]+)?)\s*:\s*&(?:\'static\s+)?str\s*=\s*"([^"]+)"\s*;'
)
STRING_LIT = re.compile(r'"((?:[^"\\]|\\.)*)"')

# ---- leg (c) patterns -----------------------------------------------------
# Matches `convert --model <slug>` in ANY line (comments included — see the
# header). The slug stops at whitespace or any quoting character, so a
# trailing backtick / quote never becomes part of it.
MODEL_CMD = re.compile(r"convert\s+--model[ \t]+([^\s`'\"]+)")
# A `convert --model` that MODEL_CMD did not capture — end of line, end of
# file, or immediately followed by a closing backtick / quote. Any of those
# is a message that names no model at all.
MODEL_CMD_NO_SLUG = re.compile(r"convert\s+--model(?![ \t]+[^\s`'\"])")
# `--model` sitting at end of line: its slug is on the NEXT line, so this is
# the signal to rejoin rather than a defect (see `logical_lines`).
MODEL_CMD_AT_EOL = re.compile(r"convert\s+--model[ \t]*$")
# Leading noise on a continuation line: indentation, then an optional comment
# marker (`//`, `///`, `//!`), then optional spacing.
CONT_PREFIX = re.compile(r"^[ \t]*(?://[/!]?)?[ \t]*")
# Innermost brace group, for alternation expansion.
BRACE = re.compile(r"\{([^{}]*)\}")


def rust_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            if fn.endswith(".rs"):
                yield os.path.join(dirpath, fn)


def arch_consts(root):
    """[(arch, 'relpath:lineno', const_name)] for every `pub const ARCH…`."""
    found = []
    for path in rust_files(root):
        rel = os.path.relpath(path, root)
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                m = ARCH_CONST.match(line)
                if m:
                    found.append((m.group(2), f"{rel}:{lineno}", m.group(1)))
    return found


def literals(root):
    """Every string literal in NON-COMMENT source under `root`.

    Comment lines are dropped on purpose: a doc comment that merely names an
    arch is neither a reader nor an emitter, and must not satisfy this gate.
    """
    out = set()
    for path in rust_files(root):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.lstrip().startswith("//"):
                    continue
                for m in STRING_LIT.finditer(line):
                    out.add(m.group(1))
    return out


def logical_lines(path):
    """[(text, first_lineno)] with `--model` continuations rejoined.

    A slug is regularly split across two physical lines, in two shapes:

        "... convert --model \\
         sensevoicesmall`?)"           <- Rust string continuation
        //!   ... convert --model
        //!   canary-1b-flash` runs.   <- doc-comment wrap

    Both are rejoined so the slug is visible to `MODEL_CMD`. Joining is
    deliberately narrow — only after a line ending in a backslash, or a line
    whose last token is `--model` — so line numbers stay accurate for every
    other line and a whole doc block never collapses into one.
    """
    out = []
    with open(path, encoding="utf-8") as fh:
        raw_lines = fh.read().split("\n")

    buf = None
    first = 0
    join_with = ""
    for lineno, raw in enumerate(raw_lines, 1):
        line = raw.rstrip("\r")
        if buf is None:
            buf, first = line, lineno
        else:
            buf = buf + join_with + CONT_PREFIX.sub("", line)
        stripped = buf.rstrip()
        if stripped.endswith("\\"):
            # Rust string continuation: the backslash and the next line's
            # leading whitespace both vanish, so join with nothing.
            buf = stripped[:-1]
            join_with = ""
            continue
        if MODEL_CMD_AT_EOL.search(stripped):
            # `--model` with its slug on the following line: the newline
            # renders as a space, so join with one.
            buf = stripped
            join_with = " "
            continue
        out.append((buf, first))
        buf = None
    if buf is not None:
        out.append((buf, first))
    return out


def model_cmd_slugs(root):
    """({raw_slug: ['relpath:lineno', ...]}, [dangling sites]).

    Every `convert --model X` under `root`, comments included. See the
    header for why comments count here and nowhere else in this gate.
    """
    found = {}
    dangling = []
    for path in rust_files(root):
        rel = os.path.relpath(path, root)
        for text, lineno in logical_lines(path):
            for m in MODEL_CMD.finditer(text):
                found.setdefault(m.group(1), []).append(f"{rel}:{lineno}")
            if MODEL_CMD_NO_SLUG.search(text):
                dangling.append(f"{rel}:{lineno}")
    return found, dangling


def expand_slug(raw):
    """[candidate, ...] for a captured slug, or [] if it is notation.

    Returns the empty list for a template an operator would never type
    verbatim (glob, `format!` placeholder, `<metavariable>`); returns one
    entry per alternative for `{a,b}` / `{a|b}` / `a|b`; otherwise the
    cleaned slug alone. `{{`/`}}` are un-escaped first because these
    strings are usually `format!` arguments.
    """
    s = raw.replace("{{", "{").replace("}}", "}").strip("`'\"")
    s = s.rstrip(".,?;:)")
    if not s:
        return []
    if "*" in s:
        return []                       # glob family notation, e.g. bigvgan-*
    if s.startswith("<") and s.endswith(">"):
        return []                       # metavariable, e.g. <arg>

    cands = [s]
    while True:
        grew = False
        nxt = []
        for c in cands:
            m = BRACE.search(c)
            if not m:
                nxt.append(c)
                continue
            inner = m.group(1)
            if "," not in inner and "|" not in inner:
                # `{}` / `{tag}` — a format placeholder, not an alternation.
                return []
            for part in re.split(r"[,|]", inner):
                nxt.append(c[: m.start()] + part.strip() + c[m.end():])
            grew = True
        cands = nxt
        if not grew:
            break

    # A bare `mimi|dac` (no braces). Safe because no accepted `--model`
    # value contains `|` — see the header.
    out = []
    for c in cands:
        out.extend(p.strip() for p in c.split("|") if p.strip())
    return out


def from_arg_literals(path):
    """Every string literal in `ModelKind::from_arg`'s match arms.

    Anchored on `impl ModelKind` so the sibling `PolicyPreset::from_arg` in
    the same file is not mistaken for it, and terminated on the first
    column-4 `}` (the fn's own closing brace at impl-member indent). The
    body is nothing but `"lit" | "lit" => Some(Self::X),` arms, so every
    non-comment literal inside it is an accepted spelling.
    """
    lits = set()
    in_impl = False
    capturing = False
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if re.match(r"^impl\s+ModelKind\s*\{", line):
                in_impl = True
            elif re.match(r"^impl\s", line):
                in_impl = False
            if in_impl and re.match(r"^\s*pub fn from_arg\s*\(", line):
                capturing = True
                continue
            if capturing:
                if re.match(r"^    \}\s*$", line):
                    capturing = False
                    continue
                if line.lstrip().startswith("//"):
                    continue
                for m in STRING_LIT.finditer(line):
                    lits.add(m.group(1))
    return lits


def read_ledger(path):
    """{arch: reason} from an 'arch|reason' file."""
    entries = {}
    dupes = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            arch, sep, reason = line.partition("|")
            arch, reason = arch.strip(), reason.strip()
            if not sep or not arch or not reason:
                dupes.append(f"malformed ledger line (want 'arch|reason'): {line!r}")
                continue
            if arch in entries:
                dupes.append(f"duplicate ledger entry for `{arch}`")
            entries[arch] = reason
    return entries, dupes


# ---- engine.rs: routed constants + BOUND_ARCHES rows ----------------------
ROUTED_CONST = re.compile(
    r'^\s*(?:pub\s+)?const\s+ARCH_[A-Z0-9_]+\s*:\s*&(?:\'static\s+)?str\s*=\s*"([^"]+)"\s*;'
)
REGISTRY_START = re.compile(r'^\s*(?:pub\s+)?const\s+BOUND_ARCHES\s*:')
REGISTRY_ROW = re.compile(r'^\s*arch\s*:\s*"([^"]+)"\s*,?\s*$')

routed, registry = set(), set()
in_registry = False
registry_seen = False
with open(engine_path, encoding="utf-8") as fh:
    for line in fh:
        if not in_registry and REGISTRY_START.match(line):
            in_registry = True
            registry_seen = True
            continue
        if in_registry:
            if line.rstrip() == "];":
                in_registry = False
                continue
            m = REGISTRY_ROW.match(line)
            if m:
                registry.add(m.group(1))
            continue
        m = ROUTED_CONST.match(line)
        if m:
            routed.add(m.group(1))

converters = arch_consts(conv_models)
binders = arch_consts(models_dir)
reader_lits = literals(models_dir)
emitter_lits = literals(conv_src)
cmd_slugs, cmd_dangling = model_cmd_slugs(models_dir)
accepted_slugs = from_arg_literals(convert_lib)
ledger_no_reader, ledger_a_bad = read_ledger(ledger_a)
ledger_no_conv, ledger_b_bad = read_ledger(ledger_b)
ledger_not_slug, ledger_c_bad = read_ledger(ledger_c)

errors = list(ledger_a_bad) + list(ledger_b_bad) + list(ledger_c_bad)

# ---- parser guards --------------------------------------------------------
# A checker that silently scanned nothing would pass every run — the exact
# fabricated-pass shape this gate exists to prevent. Each guard fires only if
# the source layout moved out from under the parser.
if not converters:
    errors.append(
        f"no `pub const ARCH...: &str = \"…\"` found anywhere under {conv_models} — the "
        f"walk or the constant spelling changed; leg (a) covered nothing, so a pass "
        f"here would be vacuous."
    )
if not binders:
    errors.append(
        f"no `pub const ARCH...: &str = \"…\"` found anywhere under {models_dir} — the "
        f"walk or the constant spelling changed; leg (b) covered nothing, so a pass "
        f"here would be vacuous."
    )
if not reader_lits:
    errors.append(
        f"zero string literals scanned in non-comment source under {models_dir} — the "
        f"reader scan is broken; every converter arch would read as unanswered."
    )
if not emitter_lits:
    errors.append(
        f"zero string literals scanned in non-comment source under {conv_src} — the "
        f"emitter scan is broken; every binder arch would read as unemitted."
    )
if not registry_seen:
    errors.append(
        f"`const BOUND_ARCHES` not found in {engine_path} — the registry was renamed or "
        f"moved; leg (a) would then miss every arch whose only reader evidence is a "
        f"registry row."
    )
elif not registry:
    errors.append(
        f"`const BOUND_ARCHES` in {engine_path} parsed to ZERO rows — the row shape "
        f"changed (expected `arch: \"…\",` one per line)."
    )
if in_registry:
    errors.append(
        f"`const BOUND_ARCHES` in {engine_path} never closed on a column-0 `];` — the "
        f"registry literal was reformatted and the row scan may have run past its end."
    )
# Leg (c)'s two guards. Either one silently disabled would turn the leg into
# a rubber stamp: an empty accepted-slug set makes every slug look broken
# (noisy, so it self-reports), but an empty found-slug set makes every slug
# look fine (silent, which is the dangerous direction) — hence both.
if not accepted_slugs:
    errors.append(
        f"`ModelKind::from_arg` yielded ZERO accepted spellings from {convert_lib} — the "
        f"`impl ModelKind` anchor, the fn signature or the closing-brace terminator "
        f"changed; leg (c) would then report every recovery-command slug as unparseable."
    )
if not cmd_slugs:
    errors.append(
        f"no `convert --model <slug>` found anywhere under {models_dir} — the scan or the "
        f"command spelling changed; leg (c) covered nothing, so a pass here would be "
        f"vacuous."
    )

# ---- leg (a): converter -> reader ----------------------------------------
answered = reader_lits | routed | registry
gap_a = {}
for arch, where, const_name in converters:
    if arch not in answered:
        gap_a.setdefault(arch, f"{const_name} at vokra-convert/src/models/{where}")

for arch in sorted(gap_a):
    if arch not in ledger_no_reader:
        errors.append(
            f"[leg a] converter arch `{arch}` ({gap_a[arch]}) has NO reader: the literal "
            f"appears nowhere in non-comment source under vokra-models/src/, and it is "
            f"neither routed nor a BOUND_ARCHES row in vokra-cli/src/engine.rs. The "
            f"converter therefore stamps `vokra.model.arch = \"{arch}\"` into a GGUF "
            f"nothing in this workspace can load. Fix: land a binder that verifies the "
            f"tag — or, if converter-only is the intended state (publish-only, "
            f"license-blocked, vast.ai-gated, separate repo), add it to the NO_READER "
            f"ledger in scripts/check-arch-handshake.sh with the real reason."
        )

for arch in sorted(ledger_no_reader):
    if arch not in gap_a:
        errors.append(
            f"[leg a] STALE ledger entry `{arch}`: NO_READER says it has no reader, but "
            f"one now exists (a literal under vokra-models/src/, or a routed constant / "
            f"BOUND_ARCHES row in vokra-cli/src/engine.rs). The recorded reason is out of "
            f"date. Fix: delete the `{arch}` line from NO_READER in "
            f"scripts/check-arch-handshake.sh."
        )

# ---- leg (b): binder -> converter ----------------------------------------
gap_b = {}
for arch, where, const_name in binders:
    if arch not in emitter_lits:
        gap_b.setdefault(arch, f"{const_name} at vokra-models/src/{where}")

for arch in sorted(gap_b):
    if arch not in ledger_no_conv:
        errors.append(
            f"[leg b] binder arch `{arch}` ({gap_b[arch]}) is emitted by NO converter: the "
            f"literal appears nowhere in non-comment source under vokra-convert/src/. The "
            f"loader can only ever be fed a GGUF this repo has no way to produce. Fix: "
            f"land a converter under crates/vokra-convert/src/models/ that stamps "
            f"`vokra.model.arch = \"{arch}\"` — or, if binder-only is intended, add it to "
            f"the NO_CONVERTER ledger in scripts/check-arch-handshake.sh with the real "
            f"reason."
        )

for arch in sorted(ledger_no_conv):
    if arch not in gap_b:
        errors.append(
            f"[leg b] STALE ledger entry `{arch}`: NO_CONVERTER says nothing emits it, but "
            f"a converter now does. Fix: delete the `{arch}` line from NO_CONVERTER in "
            f"scripts/check-arch-handshake.sh."
        )

# ---- leg (c): recovery command -> CLI parser ------------------------------
# A `convert --model` whose slug never arrived is its own defect: the message
# names no command at all, so there is nothing for the operator to run.
for site in sorted(set(cmd_dangling)):
    errors.append(
        f"[leg c] `convert --model` at vokra-models/src/{site} is followed by no slug "
        f"(even after rejoining continuation lines). The message tells an operator to run "
        f"a command and then does not say which model — name the `--model` value."
    )

gap_c = {}
checked_c = 0
for raw in sorted(cmd_slugs):
    candidates = expand_slug(raw)
    if not candidates:
        continue                        # notation: glob / placeholder / <meta>
    for cand in candidates:
        checked_c += 1
        if cand not in accepted_slugs:
            gap_c.setdefault(cand, (raw, cmd_slugs[raw][0]))

for slug in sorted(gap_c):
    raw, where = gap_c[slug]
    quoted = f" (from `{raw}`)" if raw != slug else ""
    if slug not in ledger_not_slug:
        errors.append(
            f"[leg c] `vokra-cli convert --model {slug}`{quoted} is printed at "
            f"vokra-models/src/{where}, but `ModelKind::from_arg` does not accept "
            f"`{slug}` — an operator who follows that instruction gets `unknown model`. "
            f"That is worse than printing no recovery step: it teaches them the error "
            f"text is unreliable. Fix: add the spelling to `ModelKind::from_arg` in "
            f"crates/vokra-convert/src/lib.rs (wiring an existing converter is usually "
            f"all it takes), or reword the message to name something that exists — or, "
            f"if this string is deliberately not a runnable slug, add it to the "
            f"NOT_A_MODEL_SLUG ledger in scripts/check-arch-handshake.sh with the real "
            f"reason."
        )

for slug in sorted(ledger_not_slug):
    if slug not in gap_c:
        errors.append(
            f"[leg c] STALE ledger entry `{slug}`: NOT_A_MODEL_SLUG says it is not a real "
            f"`--model` value, but `ModelKind::from_arg` now accepts it (or it no longer "
            f"appears under vokra-models/src/). Fix: delete the `{slug}` line from "
            f"NOT_A_MODEL_SLUG in scripts/check-arch-handshake.sh."
        )

if errors:
    print(f"check-arch-handshake: FAIL — {len(errors)} problem(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)

conv_archs = {a for a, _, _ in converters}
bind_archs = {a for a, _, _ in binders}
print(
    f"check-arch-handshake: OK — leg (a) {len(conv_archs)} converter arch(es), "
    f"{len(conv_archs) - len(gap_a)} answered by a reader, {len(gap_a)} declared in "
    f"NO_READER; leg (b) {len(bind_archs)} binder arch(es), "
    f"{len(bind_archs) - len(gap_b)} emitted by a converter, {len(gap_b)} declared in "
    f"NO_CONVERTER; leg (c) {len(cmd_slugs)} distinct `convert --model` string(s) -> "
    f"{checked_c} slug candidate(s) checked against {len(accepted_slugs)} accepted "
    f"spelling(s), {len(gap_c)} declared in NOT_A_MODEL_SLUG."
)
PY
}

# Serialise a ledger array to a file. `${arr[@]+"${arr[@]}"}` so an EMPTY
# ledger does not trip `set -u`. An empty ledger is the goal state, not an
# error: it means both halves of every arch handshake are present.
write_ledger() {
    local out="$1"
    shift
    : >"$out"
    local e
    for e in "$@"; do
        printf '%s\n' "$e" >>"$out"
    done
}

self_test() {
    local status=0
    local tmp
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' RETURN

    mkdir -p "$tmp/conv/models" "$tmp/models/alpha" "$tmp/models/nested"

    # Converter side: three arch constants.
    #   alpha  -> answered by a literal under models/
    #   gamma  -> answered only by a BOUND_ARCHES row (registry acceptance)
    #   orphan -> answered by nothing (the leg (a) defect)
    {
        printf 'pub const ARCH: &str = "alpha";\n'
        printf 'pub const ARCH_GAMMA: &str = "gamma";\n'
        printf 'pub const ARCH_ORPHAN: &str = "orphan";\n'
        # Emits the binder tag `mbeta`, so leg (b) is satisfied for it.
        printf 'const EMITS: &str = "mbeta";\n'
        # A comment naming `mlonely` must NOT count as an emitter.
        printf '// this comment mentions "mlonely" and must not satisfy leg (b)\n'
    } >"$tmp/conv/models/convs.rs"

    # Binder side: two arch constants + a reader literal for `alpha`.
    #   mbeta   -> emitted by the converter file above
    #   mlonely -> emitted by nobody (the leg (b) defect)
    printf 'pub const ARCH: &str = "mbeta";\nconst READS: &str = "alpha";\n' \
        >"$tmp/models/alpha/mod.rs"
    printf 'pub const ARCH_LONELY: &str = "mlonely";\n' >"$tmp/models/nested/mod.rs"

    # Leg (c) fixture: a stand-in `ModelKind::from_arg` accepting three
    # spellings, plus a `PolicyPreset::from_arg` in the same file whose
    # literals must NOT leak into the accepted set (the real lib.rs has
    # exactly this shape).
    {
        printf 'impl ModelKind {\n'
        printf '    pub fn from_arg(s: &str) -> Option<Self> {\n'
        printf '        match s {\n'
        printf '            "goodslug" => Some(Self::Good),\n'
        printf '            "wrapped-slug" | "joined-slug" => Some(Self::Wrapped),\n'
        printf '            "moonshine-tiny" | "moonshine-base" => Some(Self::Moonshine),\n'
        printf '            // "commented-out-slug" must not count\n'
        printf '            _ => None,\n'
        printf '        }\n'
        printf '    }\n'
        printf '}\n\n'
        printf 'impl PolicyPreset {\n'
        printf '    pub fn from_arg(s: &str) -> Option<Self> {\n'
        printf '        match s {\n'
        printf '            "not-a-model-slug" => Some(Self::X),\n'
        printf '            _ => None,\n'
        printf '        }\n'
        printf '    }\n'
        printf '}\n'
    } >"$tmp/conv/lib.rs"

    # Recovery commands under the binder tree. Written into the `nested`
    # module so `alpha/mod.rs` stays a pure leg (a)/(b) fixture:
    #   goodslug            -> plain, accepted
    #   wrapped-slug        -> split by a Rust string continuation
    #   joined-slug         -> split by a doc-comment wrap
    #   moonshine-{tiny,base} -> brace alternation, both accepted
    #   bigvgan-*           -> glob notation, skipped
    #   <arg>               -> metavariable, skipped
    write_cmds() {
        {
            printf 'pub const ARCH_LONELY: &str = "mlonely";\n'
            printf '/// Run `vokra-cli convert --model goodslug` first.\n'
            printf '"... convert --model \\\n'
            printf '                 wrapped-slug`?)"\n'
            printf '//!   ... `vokra-cli convert --model\n'
            printf '//!   joined-slug` runs.\n'
            printf '/// `vokra-cli convert --model moonshine-{tiny,base}`.\n'
            printf '/// `vokra-cli convert --model bigvgan-*`.\n'
            printf '/// The `vokra-cli convert --model <arg>` spelling.\n'
            local extra
            for extra in "$@"; do
                printf '%s\n' "$extra"
            done
        } >"$tmp/models/nested/mod.rs"
    }
    write_cmds

    write_engine() {
        {
            printf 'const ARCH_ROUTED: &str = "some-routed-arch";\n\n'
            printf 'const BOUND_ARCHES: &[BoundArch] = &[\n'
            local a
            for a in "$@"; do
                printf '    BoundArch {\n        arch: "%s",\n        module: "vokra_models::x",\n    },\n' "$a"
            done
            printf '];\n'
        } >"$tmp/engine.rs"
    }
    write_engine gamma

    # run <ledger-a...> -- <ledger-b...> [-- <ledger-c...>]
    run() {
        local -a la=() lb=() lc=()
        local seen_sep=0 arg
        for arg in "$@"; do
            if [ "$arg" = "--" ]; then
                seen_sep=$((seen_sep + 1))
                continue
            fi
            case "$seen_sep" in
                0) la+=("$arg") ;;
                1) lb+=("$arg") ;;
                *) lc+=("$arg") ;;
            esac
        done
        write_ledger "$tmp/ledger_a" ${la[@]+"${la[@]}"}
        write_ledger "$tmp/ledger_b" ${lb[@]+"${lb[@]}"}
        write_ledger "$tmp/ledger_c" ${lc[@]+"${lc[@]}"}
        run_check "$tmp/conv/models" "$tmp/conv" "$tmp/models" "$tmp/engine.rs" \
            "$tmp/ledger_a" "$tmp/ledger_b" "$tmp/conv/lib.rs" "$tmp/ledger_c"
    }

    local out
    local ok='orphan|declared converter-only'
    local okb='mlonely|declared binder-only'

    # 1. Fully declared -> passes. Also proves registry acceptance: `gamma`
    #    has no models-side literal and is NOT in the ledger, so if the
    #    BOUND_ARCHES row did not count, this case would fail.
    if run "$ok" -- "$okb" >/dev/null 2>&1; then
        echo "self-test PASS: declared gaps pass, and a BOUND_ARCHES row answers leg (a)"
    else
        echo "self-test FAIL: a fully declared tree should pass" >&2
        status=1
    fi

    # 2. Undeclared leg (a) gap -> fails, naming the arch.
    if out="$(run -- "$okb" 2>&1)"; then
        echo "self-test FAIL: an undeclared converter-with-no-reader should fail" >&2
        status=1
    elif grep -q 'leg a.*`orphan`' <<<"$out"; then
        echo "self-test PASS: an undeclared converter-with-no-reader fails, naming it"
    else
        echo "self-test FAIL: leg (a) failure did not name \`orphan\`" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi

    # 3. Undeclared leg (b) gap -> fails, naming the arch. Doubles as proof
    #    that the comment mentioning "mlonely" did not count as an emitter.
    if out="$(run "$ok" -- 2>&1)"; then
        echo "self-test FAIL: an undeclared binder-with-no-converter should fail" >&2
        status=1
    elif grep -q 'leg b.*`mlonely`' <<<"$out"; then
        echo "self-test PASS: an undeclared binder-with-no-converter fails, naming it"
    else
        echo "self-test FAIL: leg (b) failure did not name \`mlonely\`" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi

    # 4. Stale leg (a) entry (a gap that is not a gap) -> fails.
    if out="$(run "$ok" 'alpha|stale claim' -- "$okb" 2>&1)"; then
        echo "self-test FAIL: a stale NO_READER entry should fail" >&2
        status=1
    elif grep -q 'STALE.*`alpha`' <<<"$out"; then
        echo "self-test PASS: a NO_READER entry whose gap closed fails as stale"
    else
        echo "self-test FAIL: stale leg (a) failure did not name \`alpha\`" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi

    # 5. Stale leg (b) entry -> fails.
    if out="$(run "$ok" -- "$okb" 'mbeta|stale claim' 2>&1)"; then
        echo "self-test FAIL: a stale NO_CONVERTER entry should fail" >&2
        status=1
    elif grep -q 'STALE.*`mbeta`' <<<"$out"; then
        echo "self-test PASS: a NO_CONVERTER entry whose gap closed fails as stale"
    else
        echo "self-test FAIL: stale leg (b) failure did not name \`mbeta\`" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi

    # 6. A converter tree the walk finds nothing in -> parser guard.
    write_ledger "$tmp/ledger_a" "$ok"
    write_ledger "$tmp/ledger_b" "$okb"
    write_ledger "$tmp/ledger_c"
    mkdir -p "$tmp/empty"
    if run_check "$tmp/empty" "$tmp/conv" "$tmp/models" "$tmp/engine.rs" \
        "$tmp/ledger_a" "$tmp/ledger_b" "$tmp/conv/lib.rs" "$tmp/ledger_c" >/dev/null 2>&1; then
        echo "self-test FAIL: scanning zero converter arches should fail the guard" >&2
        status=1
    else
        echo "self-test PASS: a scan that found no converter arches fails rather than passing vacuously"
    fi

    # 7. Registry renamed away -> parser guard, not a silent loss of leg (a)
    #    evidence.
    printf 'const ARCH_ROUTED: &str = "some-routed-arch";\n' >"$tmp/engine.rs"
    if run "$ok" -- "$okb" >/dev/null 2>&1; then
        echo "self-test FAIL: a missing BOUND_ARCHES literal should fail the guard" >&2
        status=1
    else
        echo "self-test PASS: a renamed/absent registry fails the parser guard"
    fi
    write_engine gamma

    # 8. Malformed ledger line -> fails rather than being silently ignored.
    if out="$(run 'orphan-with-no-pipe' -- "$okb" 2>&1)"; then
        echo "self-test FAIL: a malformed ledger line should fail" >&2
        status=1
    elif grep -q 'malformed ledger line' <<<"$out"; then
        echo "self-test PASS: a ledger line with no reason fails as malformed"
    else
        echo "self-test FAIL: malformed ledger line was not reported" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi

    # 9. Leg (c): a recovery command naming a slug `from_arg` rejects ->
    #    fails, naming the slug. This is the exact defect the leg was added
    #    for (mt3 / beat-this / redimnet / llama-omni2, 2026-08-15).
    write_cmds '/// Re-run `vokra-cli convert --model ghostslug` against the checkpoint.'
    if out="$(run "$ok" -- "$okb" 2>&1)"; then
        echo "self-test FAIL: a recovery command naming an unparseable slug should fail" >&2
        status=1
    elif grep -q 'leg c.*ghostslug' <<<"$out" && grep -q 'unknown model' <<<"$out"; then
        echo "self-test PASS: an unparseable recovery-command slug fails, naming it"
    else
        echo "self-test FAIL: leg (c) failure did not name \`ghostslug\`" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi

    # 10. The same slug, declared in NOT_A_MODEL_SLUG -> passes. Proves the
    #     ledger is honoured (and, with case 9, that it is what changed).
    if run "$ok" -- "$okb" -- 'ghostslug|deliberately not a runnable slug' >/dev/null 2>&1; then
        echo "self-test PASS: a declared non-slug passes leg (c)"
    else
        echo "self-test FAIL: a declared NOT_A_MODEL_SLUG entry should pass" >&2
        status=1
    fi
    write_cmds

    # 11. Stale leg (c) entry -> fails. `goodslug` parses, so claiming it is
    #     not a real slug is a lie the gate must catch.
    if out="$(run "$ok" -- "$okb" -- 'goodslug|stale claim' 2>&1)"; then
        echo "self-test FAIL: a stale NOT_A_MODEL_SLUG entry should fail" >&2
        status=1
    elif grep -q 'STALE.*`goodslug`' <<<"$out"; then
        echo "self-test PASS: a NOT_A_MODEL_SLUG entry whose slug now parses fails as stale"
    else
        echo "self-test FAIL: stale leg (c) failure did not name \`goodslug\`" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi

    # 12. Continuation rejoining is load-bearing, not decoration. The base
    #     fixture already contains a backslash-continued `wrapped-slug` and a
    #     doc-comment-wrapped `joined-slug`, both of which case 1 accepted.
    #     Break only the accepted-spelling side and both must now be reported
    #     — if the rejoining silently dropped them, they would never have been
    #     checked at all and this case would pass vacuously.
    local saved_lib="$tmp/lib_saved.rs"
    cp "$tmp/conv/lib.rs" "$saved_lib"
    sed 's/"wrapped-slug" | "joined-slug"/"neither-spelling"/' "$saved_lib" >"$tmp/conv/lib.rs"
    if out="$(run "$ok" -- "$okb" 2>&1)"; then
        echo "self-test FAIL: slugs split across lines should be checked, not skipped" >&2
        status=1
    elif grep -q 'wrapped-slug' <<<"$out" && grep -q 'joined-slug' <<<"$out"; then
        echo "self-test PASS: backslash- and doc-wrap-continued slugs are rejoined and checked"
    else
        echo "self-test FAIL: a continued slug was not checked (rejoining is broken)" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi
    cp "$saved_lib" "$tmp/conv/lib.rs"

    # 13. `convert --model` with no slug at all -> fails. A message that
    #     names no command is as useless as one naming a command that does
    #     not exist.
    write_cmds '/// Re-run `vokra-cli convert --model` and try again.'
    if out="$(run "$ok" -- "$okb" 2>&1)"; then
        echo "self-test FAIL: a bare \`convert --model\` with no slug should fail" >&2
        status=1
    elif grep -q 'followed by no slug' <<<"$out"; then
        echo "self-test PASS: a recovery command with no \`--model\` value fails"
    else
        echo "self-test FAIL: the dangling \`--model\` was not reported" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi
    write_cmds

    # 14. A `from_arg` the parser cannot find -> guard, not a silent pass of
    #     every slug. (Renaming the impl block is the realistic way to break
    #     it.)
    sed 's/^impl ModelKind {/impl RenamedKind {/' "$saved_lib" >"$tmp/conv/lib.rs"
    if run "$ok" -- "$okb" >/dev/null 2>&1; then
        echo "self-test FAIL: an unparseable from_arg should fail the guard" >&2
        status=1
    else
        echo "self-test PASS: a from_arg the parser cannot locate fails the guard"
    fi
    cp "$saved_lib" "$tmp/conv/lib.rs"

    if [ "$status" -eq 0 ]; then
        echo "check-arch-handshake --self-test: OK (14 cases)"
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
        for d in "$CONVERT_MODELS_DEFAULT" "$CONVERT_SRC_DEFAULT" "$MODELS_DEFAULT"; do
            if [ ! -d "$d" ]; then
                echo "error: required directory not found: $d" >&2
                exit 1
            fi
        done
        for f in "$ENGINE_DEFAULT" "$CONVERT_LIB_DEFAULT"; do
            if [ ! -f "$f" ]; then
                echo "error: required file not found: $f" >&2
                exit 1
            fi
        done
        LEDGER_A="$(mktemp)"
        LEDGER_B="$(mktemp)"
        LEDGER_C="$(mktemp)"
        trap 'rm -f "$LEDGER_A" "$LEDGER_B" "$LEDGER_C"' EXIT
        write_ledger "$LEDGER_A" ${NO_READER[@]+"${NO_READER[@]}"}
        write_ledger "$LEDGER_B" ${NO_CONVERTER[@]+"${NO_CONVERTER[@]}"}
        write_ledger "$LEDGER_C" ${NOT_A_MODEL_SLUG[@]+"${NOT_A_MODEL_SLUG[@]}"}
        run_check "$CONVERT_MODELS_DEFAULT" "$CONVERT_SRC_DEFAULT" "$MODELS_DEFAULT" \
            "$ENGINE_DEFAULT" "$LEDGER_A" "$LEDGER_B" "$CONVERT_LIB_DEFAULT" "$LEDGER_C"
        ;;
    *)
        echo "error: unknown argument '$1'" >&2
        usage >&2
        exit 1
        ;;
esac
