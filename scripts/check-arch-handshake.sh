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
# WHICH CONSTANTS COUNT AS A DECLARATION            [widened 2026-08-15]
#   `pub const ARCH`, `pub const ARCH_<SUFFIX>` (gigaam's ARCH_V3, whisper's
#   size siblings) AND `pub const EXPECTED_ARCH`. That last spelling is used by
#   29 of the 89 arch constants under vokra-models/src, and the regex did not
#   match it — so this gate and `check-bound-arch-coverage.sh` both printed a
#   confident "60 binder arches, all clean" over a population that was missing
#   a third of the binders. `charsiu`, which had no converter at the time, sat
#   inside that blind spot. A gate that is green because it did not look is
#   worse than no gate: it actively certifies the thing it failed to check.
#   `unseen_arch_spellings` now fails the run if ANY arch-shaped `&str`
#   constant on disk is outside the discovery regex, so the next new spelling
#   is a loud one-time failure rather than a silently smaller population.
#
# THE THREE LEGS
#   (a) converter -> reader
#       Every arch constant (see above) under
#       `crates/vokra-convert/src/models/` must be answered by a reader:
#       the literal "X" appearing in non-comment source under
#       `crates/vokra-models/src/`, OR "X" being routed / registered in
#       `crates/vokra-cli/src/engine.rs` (a routed `const ARCH_*` or a
#       BOUND_ARCHES row — both of which assert a binder exists, and the
#       sibling gate keeps that assertion honest from the other side).
#
#   (b) binder -> converter
#       Every arch constant (see above) under `crates/vokra-models/src/` must
#       be emitted by some converter: the literal "X" appearing in
#       non-comment source under `crates/vokra-convert/src/`.
#
#   Comment lines are stripped before literals are collected, deliberately: a
#   doc-comment that merely NAMES an arch is not a reader and not an emitter,
#   and must not be able to satisfy this gate.
#
#   TEST CODE IS STRIPPED TOO                          [added 2026-08-15]
#   Same principle, and it was the larger hole. Comments were stripped but
#   `#[cfg(test)]` blocks and `*/tests.rs` files were not, so leg (a)
#   reported "108 converter arches, 87 answered by a reader" over a
#   population where 25 of those 87 were answered ONLY by test code. The
#   clearest case: every `"yamnet"` literal under vokra-models/src lived in
#   the PANNs / ATST / MAEST test modules, and one of them was
#   `for sibling in ["yamnet", "ast", …]` inside a test whose entire purpose
#   is asserting PANNs REFUSES a YAMNet GGUF. A test proving "we do NOT bind
#   this" was being counted as proof that we do — while
#   crates/vokra-models/src/yamnet/ does not exist. That is the same
#   "green because it did not look" failure the EXPECTED_ARCH widening
#   above was written to end, recurring inside the gate meant to prevent it.
#   The ledger claimed 21 known gaps against a real 46: precisely the noise
#   it exists to remove.
#
#   The rule is applied to BOTH trees, though only leg (a) moved: measured
#   on the 2026-08-15 tree, 0 of 89 binder arches were emitted solely by
#   converter-side test code, so leg (b)'s count is unchanged. It is applied
#   symmetrically anyway because the hole is symmetric — a `#[cfg(test)]`
#   fixture under vokra-convert/src that stamps `"foo"` would otherwise
#   satisfy leg (b) for a binder with no real converter, which is the exact
#   shape that just fired on the other side.
#
#   Over-skipping is the safe direction and is chosen on purpose: dropping
#   too much source makes MORE arches read as unanswered, which fails
#   loudly. Under-skipping is what fails silently. A `#[cfg(test)]` region
#   whose braces never balance is reported as a parser guard rather than
#   swallowing the rest of the file.
#
#   WHAT STRIPPING CANNOT FIX: A COLLIDING LITERAL
#   `ast` is answered by `Self::Ast => "ast"` at
#   vokra-models/src/canary_1b_flash/mod.rs:363 — Canary's Automatic Speech
#   Translation task label, ordinary non-test source, nothing to do with the
#   Audio Spectrogram Transformer arch that vokra-convert stamps. No amount
#   of comment- or test-stripping can separate those two: a short arch tag
#   simply collided with an unrelated enum discriminant. The NOT_A_READER
#   ledger below is how that is said out loud — it names the one FILE whose
#   occurrences of a literal do not count, which is a checkable claim rather
#   than a heuristic, and it is double-sided like every other ledger here.
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
#   (d) metadata key -> converter stamp          [added 2026-08-15]
#       Every `vokra.<group>.<key>` a REQUIRED reader under
#       `crates/vokra-models/src/` looks up must be stamped by some file
#       under `crates/vokra-convert/src/`.
#
#       WHY: legs (a)-(c) compare arch literals and nothing else, and the
#       NO_READER ledger entry for `openwakeword` says out loud what that
#       cost — "What handshakes on the arch tag need not handshake on
#       anything else: the binder also requires seven
#       `vokra.openwakeword.*` metadata keys, the converter stamped none of
#       them, and so every GGUF it produced failed to load — while this
#       gate stayed green, because it only ever compared arch literals."
#       Round 8 repaired that instance; round 9 found a second
#       (`llama_omni2`, ten unstamped `vokra.llama_omni2.arch.*` keys).
#       Neither `scripts/check-bound-arch-coverage.sh` nor the in-crate
#       registry test can see this class either: an arch tag that matches
#       perfectly still describes a GGUF whose config chunk is empty.
#
#       WHAT "REQUIRED" MEANS, AND WHY THE DEFINITION IS THE WHOLE JOB
#       A naive scan — every `vokra.*` literal on the reader side must be
#       emitted on the converter side — reports 172 gaps on the 2026-08-15
#       tree, which is not a gate, it is noise with a shell script around
#       it. The overwhelming majority are deliberate, documented and
#       correct, in two shapes this repo uses everywhere:
#
#         - OPTIONAL ALL-OR-NOTHING GROUPS. `from_gguf` returns
#           `Result<Option<Self>>` and answers `Ok(None)` when NO key of
#           the group is present, but errors loudly when only SOME are
#           (nisqa, ten_vad, smart_turn, firered_vad, firered_asr_aed_l,
#           gigaam, whisper_medusa). A converter that stamps none of the
#           group is exactly what that reader is written for.
#         - CALLER DEFAULTS. `unwrap_or(default.n_fft)` over a
#           primary-source constant (panns, musicgen, jasco, audiogen,
#           demucs, sortformer, canary_1b_flash). panns says so in its own
#           comment: "the PANNs converter does NOT stamp these, so
#           `PannsConfig::from_gguf` falls back to the primary-source
#           constants per-key."
#
#       So a key is REQUIRED only when the reader has no escape for it:
#       absence is an error (`ok_or_else` / `None => Err`), or absence
#       yields a ZERO SENTINEL that a later gate rejects. The sentinel case
#       is not a stylistic quibble — it is `llama_omni2`, whose
#       `read_u32_or_zero` decays every unstamped axis to `0` and defers
#       the failure to `validate_for_forward`. A zero is not a default; it
#       is a deferred loud-partial, and a converter that never stamps the
#       axis makes that gate fire on every artifact it writes.
#
#       THE SUPPRESSIONS ARE RULES, NOT A NAME LIST
#       Five classes of literal look like a required read and are not. Each
#       is expressed as a CATEGORY, because a category generalises to the
#       next model and a list of five names does not:
#
#         S1 group-optional escape  — the enclosing fn returns
#            `Result<Option<…>>`/`Option<…>`, or its body has an early
#            `return Ok(None)`. The all-or-nothing shape above.
#         S2 caller default         — the read supplies a real fallback
#            value (`unwrap_or*`, `None => Ok(<value>)`). A zero/empty
#            fallback does NOT qualify; see the sentinel note above.
#         S3 mention, not a read    — every occurrence sits inside a string
#            literal (a diagnostic naming a key it cannot find, or an error
#            describing a future artifact) and none in code position.
#            `vokra.snac.codebook_tables` is only ever error text.
#         S4 runtime-assembled prefix — the constant is a PREFIX an index
#            is appended to (`GGUF_KEY_PATCH_GRID_PREFIX =
#            "vokra.atst.patch_grid"` stamped as `_0`/`_1`; `PREFIX_DELAY =
#            "vokra.moshi.delay."`). The assembled key never exists as a
#            literal on either side, so comparing literals is a category
#            error. Recognised by `PREFIX` in the constant NAME.
#         S5 dead-code reserved constant — the declaration carries
#            `#[allow(dead_code)]`/`#[expect(dead_code)]` and is read by
#            nothing yet (`vokra.kokoro.phase_activation`, "consumed by the
#            T18 load/forward wiring").
#
#       Every suppression is counted and printed. A run that suppressed
#       everything would say so rather than printing a confident clean line.
#
#       DECLARING A KEY IS NOT STAMPING IT  [fixed 2026-08-15]
#       For its first day this leg answered "is the key stamped?" by asking
#       whether the literal appeared ANYWHERE in non-comment, non-test
#       converter source — and a
#       `pub const KEY_N_WAKEWORDS: &str = "vokra.openwakeword.n_wakewords";`
#       declaration is such an appearance. Deleting the single
#       `b.add_u32(KEY_N_WAKEWORDS, …)` call that actually writes it left
#       the leg's output byte-identical and still saying OK; deleting six of
#       openwakeword's seven stamps did too. So the leg certified a property
#       it did not check — the round-4 lesson recurring inside the gate
#       written after learning it — and it did so for the more likely of the
#       two regressions: the founding cases (openwakeword round 8,
#       llama_omni2 round 9) had converters that never mentioned the group at
#       all, which a literal scan does catch, whereas a const block wired six
#       ways out of seven is what a refactor actually leaves behind.
#
#       `meta_stamped` now mirrors the reader half, which never had this bug:
#       a key counts as written only when it REACHES CODE — the const name in
#       code position, the key inline at the call site, or head interpolation
#       (`format!("{KEY_WORDPIECE_PREFIX}.kind")`). The declaration line
#       itself is skipped, exactly as `meta_read_keys` skips it and counts
#       the difference as S3. The suppressions were never involved: the
#       reader side classified all seven openwakeword keys REQUIRED with no
#       suppression firing, and the driver then short-circuited on
#       `key in stamped_keys`.
#
#       WHAT THIS LEG DOES NOT SEE — measured, not assumed
#       It resolves a key to its read site through a `const NAME: &str`
#       binding or an inline literal. A key assembled by `format!` from
#       parts (other than the S4 prefix shape) is invisible to it, as is a
#       key stamped by an offline Python sidecar under `tools/parity/`
#       rather than by Rust. Both were checked on the 2026-08-15 tree —
#       no `vokra.*` key is stamped by a sidecar, and no required read
#       assembles its key beyond the S4 prefixes — but a pass here is not a
#       proof that every metadata handshake in the tree is sound.
#
#       A READ-BACK IS NOT A WRITE, AND THE EMITTER SIDE CANNOT TELL
#       [open, measured 2026-08-15]
#       `meta_stamped` asks "does this key reach converter CODE", not "does
#       it reach a GGUF writer", because converters stamp through too many
#       shapes to enumerate (`b.add_u32`, `b.add_metadata`,
#       `write_u32_array(b, KEY, …)`, `[KEY_A, KEY_B]` arrays a loop walks).
#       So a converter-side READ of a key counts as a stamp. That is not
#       hypothetical: `crates/vokra-convert/src/main.rs` prints a per-model
#       summary by reading the GGUF it just wrote, and 80 required keys have
#       their only INLINE sighting in one of those `file.get("vokra.…")`
#       calls.
#
#       Consequence, measured rather than argued: deleting
#       `b.add_u32(KEY_SAMPLE_RATE, …)` from
#       `crates/vokra-convert/src/models/kyutai_stt.rs` leaves this gate
#       GREEN, and NOT because a suppression fired —
#       `vokra.kyutai_stt.sample_rate` is classified REQUIRED at
#       crates/vokra-models/src/kyutai_stt/mod.rs:503, since its
#       `read_u32_or_zero` yields the zero sentinel that self-test case 32
#       pins as "a deferred failure, not a default". It stays green only
#       because main.rs:885 reads the key back for that summary line. Mask
#       that one literal and the gate names the key correctly. Worth noting
#       that `KyutaiSttConfig::validate_for_forward` does NOT re-check
#       `sample_rate`, so unlike llama_omni2's axes the zero would not be
#       caught downstream either.
#
#       Excluding `.get(` / `contains_key(` positions from the emitter scan
#       was measured against the whole tree and costs ZERO new findings, so
#       it is available as a follow-up. It is deliberately NOT taken here:
#       it would change the verdict on a live converter, which is an owner
#       call rather than a gate-hygiene one.
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
#   BUT ONLY THE GAP IS MACHINE-CHECKED, NEVER THE REASON. The staleness leg
#   asks "is there still no reader?" and nothing more. Every word to the right
#   of the `|` is prose no gate reads, so a reason can go false while its entry
#   stays perfectly green — the gap outlives the explanation for it. A
#   2026-08-15 sweep of all 47 `NO_READER` reasons found five that had drifted
#   that way (`ecapa_tdnn`, `stable_audio_open_small`, `neutts-air`, plus loose
#   citations on `mossformer2_ss_16k` and `openwakeword`); each now carries a
#   dated CORRECTION rather than a silent rewrite, because the shape of the
#   error is the useful part. Two failure modes recurred and are worth naming:
#     - a reason that describes a MECHANISM contradicted by its own stated
#       consequence (`stable_audio_open_small` named a resolver result that,
#       had it been real, would have blocked publish outright instead of
#       gating it behind a flag);
#     - a reason that restates the GAP as its own CAUSE (`neutts-air` said a
#       later wave defers "the arch-tag verification", which is just another
#       way of saying no reader exists).
#   So: when touching an entry, re-read the file:line it cites. A citation that
#   was right when written drifts as the cited file grows.
#
# Zero-dep: bash + python3 stdlib only (no jq, no pip, no cargo). Not a Vokra
# runtime dep.
# Exit: 0 = all three legs clean, 1 = an undeclared gap / a stale ledger entry /
# an unparsable recovery-command slug / a parser guard trip / a bad argument.

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
  'ace_step|publish-only: MIT music generation (ACE-Step), a ~9.6 GB multi-file bundle needing an offline merge plus a vast.ai handoff per [[feedback-large-models-on-vast-ai]]. ace_step.rs records the prepare script as future work; no vokra-models::ace_step module exists.'
  'basic-pitch|publish-only BF16 pass-through (Spotify basic-pitch, apache-2.0; <2 GB so local convert is safe). basic_pitch.rs:47-48 defers the offline TF-SavedModel flattening script; no vokra-models::basic_pitch module exists.'
  'beats|publish-only BF16 pass-through (microsoft/unilm BEATs, MIT via the repo-root LICENSE). beats.rs:60 defers the safetensors prepare script. The tag is kept distinct precisely so a BEATs checkpoint cannot misroute into a HuBERT loader; no binder module exists.'
  'bs_roformer|publish-BLOCKED, not merely unbound: the BS-RoFormer family carries no uniform upstream license (most releases carry none at all), so the converter defaults to LicenseClass::RedistributionForbidden fail-closed and a caller must supply --license <spdx> from their own attestation. bs_roformer.rs:19 names crates/vokra-models/src/bs_roformer/ as the binder that "will read" this metadata; that directory does not exist yet.'
  'dasheng|publish-only BF16 pass-through (Xiaomi mispeech Dasheng, apache-2.0, <2 GB). dasheng.rs:45 defers the prepare script; no vokra-models::dasheng module exists.'
  'firered_asr_llm_l|awaiting a binder: ~16.6 GB / 8.3B BF16 (Conformer encoder + Qwen2 LM decoder). firered_asr_llm_l.rs:100 records that it has no runtime binder of its own and that the only landed Qwen2-family forward belongs to a different model. Its siblings firered_asr_aed_l and firered_vad each have a binder; the tag stays distinct so an AED loader cannot try to read an LLM decoder.'
  'htdemucs_multi|publish-only, variant-agnostic BF16 pass-through skeleton (facebookresearch/demucs, MIT). htdemucs_multi.rs:50 names a future vokra-models::htdemucs_multi module; real-weight parity is deferred to owner sign-off (license-audit.md section 3.1).'
  'mert|publish-only BF16 pass-through AND NonCommercial: m-a-p/MERT-v1-330M is cc-by-nc-4.0, so publish needs --allow-noncommercial and the M2-13 runtime gate refuses a commercial-mode load. mert.rs:53 defers the prepare script; no binder module exists.'
  'mossformer2_ss_16k|publish-only BF16 pass-through (ClearerVoice-Studio MossFormer2, apache-2.0). mossformer2_ss_16k.rs:52-56 defers real-weight parity against a future gated-attention forward to owner sign-off, and :60-70 is where the tag is kept distinct from the landed FsmnVad binder (related FSMN block, different task head) — those are two separate sections, and citing only the first for both claims is how a reason starts drifting off its evidence.'
  'muq|publish-BLOCKED by license posture: OpenMuQ/MuQ-large-msd-iter declared no license as of 2026-08-13, so the converter maps to LicenseClass::Unknown fail-closed and section 3.1 stays blank. Publish-only BF16 pass-through besides, with the prepare script deferred at muq.rs:50; no binder module exists.'
  'openwakeword|publish-only BF16 pass-through for the raw wake-word weights (dscripka/openWakeWord, apache-2.0). openwakeword.rs:32-34 states the runtime port is deferred — the audio-dialect kws op consumes the artifact in a future WP. NOT to be confused with the SEPARATE openwakeword_op pair: vokra-convert/src/models/openwakeword_op.rs stamps "openwakeword_op" and vokra-models/src/kws/openwakeword/mod.rs verifies it on load at :394-404 against the mirrored constant at :116. Same family, two tags, only this one is unbound. CORRECTION 2026-08-15: this entry used to call that pair "fully bound", which was wrong twice over, and the error is instructive about the limit of THIS gate. What handshakes on the arch tag need not handshake on anything else: the binder also requires seven vokra.openwakeword.* metadata keys, the converter stamped none of them, and so every GGUF it produced failed to load — while this gate stayed green, because it only ever compared arch literals. That half is now repaired (the converter stamps the group, and a --config side-car supplies the per-wake-word names, which are not derivable from the tensors) and is pinned by crates/vokra-convert/tests/openwakeword_op_roundtrip.rs plus the convert-then-bind test crates/vokra-models/tests/openwakeword_convert_bind.rs. Even so "fully bound" would overstate it: the runtime FORWARD is a loud-partial, because the frozen Google speech_embedding extractor is still untranscribed. Load: yes. Forward: no.'
  'pyannote-speaker-diarization|converter-only BY DESIGN: this GGUF is a WEIGHTLESS pipeline orchestrator (clustering thresholds plus sub-model references, no sincnet.* / lstm.* tensors at all). vokra-models/src/pyannote/mod.rs:130 documents the refusal verbatim — EXPECTED_ARCH is deliberately "pyannote-segmentation" — and verify_arch names this tag in its rejection text so an operator who hands the pipeline to the backbone binder is told "you handed me a pipeline, not a backbone" instead of hitting a confusing empty-manifest error. A pipeline-level loader is a follow-up.'
  'reazonspeech_nemo_v2|publish-only BF16 pass-through (ReazonSpeech NeMo v2, apache-2.0). reazonspeech_nemo_v2.rs:53-54 names a future vokra-models::reazonspeech_nemo_v2 module (Longformer local-attention encoder + RNN-T / CTC head) and defers the forward to owner sign-off.'
  'stable_audio_open_small|publish-gated: the Stability AI Community License is not SPDX-registered, so the converter HARD-MAPS that string and its aliases to LicenseClass::NonCommercial ahead of the SPDX resolver (stable_audio_open_small.rs:77-81 documents the map, :140-158 is the is_sacl arm), which is why publish requires --allow-noncommercial. Same shape as the CPML / xtts_v2 precedent. stable_audio_open_small.rs:38 names a future vokra-models::stable_audio_open_small binder; none exists. CORRECTION 2026-08-15: this entry used to say from_license_str "returns Unknown". That is what the helper would do if it were reached, and it is exactly what the muq entry above correctly describes for ITS tag — but here the hard-map short-circuits it, and an Unknown would make publish refuse outright rather than merely demand a flag, so the old reason named a mechanism that contradicted the consequence in its own next clause.'
  'unity-2|vast.ai-gated (~9.00 GB) AND NonCommercial: SeamlessM4T v2 Large is cc-by-nc-4.0, so publish needs --allow-noncommercial. The converter is a BF16 pass-through skeleton and seamless_m4t_v2_large.rs:29-31 keeps the tag distinct from the M4T v1 / MMS siblings so it cannot misroute the runtime binder (FR-EX-08). No binder module exists.'
  'yamnet|publish-only BF16 pass-through (YAMNet mirror, apache-2.0 with section 3.1 blank fail-closed pending owner confirmation of the mirror LICENSE). yamnet.rs:49 defers the prepare script and :24-25 keeps the tag distinct so a MobileNet checkpoint cannot route through a Cnn14 loader. Until 2026-08-15 the ONLY "yamnet" literals under vokra-models/src were the PANNs / ATST / MAEST tests asserting those binders REFUSE a YAMNet GGUF, and the gate was counting that refusal as proof of a binder.'
  'facodec|publish-only BF16 pass-through (naturalspeech3_facodec.rs). Runtime binder + real-weight parity are a post-signoff follow-up on the RMVPE / Charsiu loud-partial precedent; the redecoder variants additionally await an owner ELVIS-Act routing decision (main zoo vs voiceclone-experimental) because they enable timbre swapping.'
  'freevc|ELVIS Act separation (CLAUDE.md design decision 8): any-to-any voice conversion belongs in the vokra-voiceclone-experimental repo, and license-audit.md:314 marks the row explicitly out of main-repo section 3.1 scope. A main-repo binder is forbidden by policy, not merely absent.'
  'granite_speech|awaiting a binder: the converter header reserves crates/vokra-models/src/granite_speech/ for it. Input is a 4.87 GB three-shard release the owner pre-merges offline, so the binder has not been started.'
  'higgs_audio_v3_tts_4b|publish-forbidden: BOSON HIGGS TTS 3 R and NC is LicenseClass::RedistributionForbidden (section II-A(c) bans redistribution, hosting and embedding). The converter exists for local owner use only; no publish and no binder follow.'
  'magpietts_v2602|publish-only BF16 pass-through (NVIDIA NeMo .nemo flattened to safetensors offline). No runtime binder; the tag is reserved for a future TTS forward.'
  'moss_audio_tokenizer|publish-only BF16 pass-through; the codec half of the MOSS-TTS pipeline. Header reserves the tag for a future native loader; no binder module exists yet.'
  'nemotron-speech-streaming-v2603|publish-only BF16 pass-through. Header names a future vokra-models::nemotron_speech_streaming_v2603 implementation; the streaming FastConformer forward is unwritten.'
  'neutts-air|publish-only BF16 pass-through. neutts_air.rs:110-119 defers the tokenizer embedding, the config-side hparams (RoPE theta, KV head split, sliding-window flag) and the NeuCodec-token vocab-slot mapping to the same later wave as the runtime binder; no crates/vokra-models/src/neutts_air/ module exists. CORRECTION 2026-08-15: this entry used to say that wave also defers "the arch-tag verification". It does not. The tag is stamped and its distinctness from twenty sibling TTS arches is argued at length at :36-50 and :138-144 — what has no verifier is the reader side, which is the gap this ledger line already records, so the old reason restated the gap as its own cause.'
  'qwen2-omni|vast.ai-gated (22.37 GB, five-shard Thinker+Talker) AND publish-blocked by the GGUF writer 5D-tensor limit that the multimodal adapter trips. No binder until that reshape-vs-extend decision lands.'
  'qwen2_audio|vast.ai-gated (~16 GB, five-shard). Owner runbook is required before a first conversion even runs, so no binder work has started.'
  'sgmse|publish-only BF16 pass-through. Header states that real-weight parity and a native Sgmse::from_gguf forward are a follow-up.'
  'ultravox|awaiting a binder: local convert is safe at ~1.83 GB, and the converter header records the runtime binder as a follow-up. Nothing blocks it but wave ordering.'
  'vibevoice_asr|vast.ai-gated (~16.5 GB, eight-shard). The sibling TTS vibevoice is published; the ASR head has neither been converted nor bound.'
  'xtts|T4 Research-only: the Coqui Public Model License maps to LicenseClass::NonCommercial, so publish requires --allow-noncommercial. It is also zero-shot voice cloning, which keeps it out of a main-repo binder under design decision 8.'
  'yue_xcodec_mini|publish-only BF16 pass-through: the multi-part SoundStream RVQ + HuBERT semantic + Vocos decoder bundle still needs its own strict codec binder; the standalone yue_upsampler sibling is independently routed.'
)

# ---------------------------------------------------------------------------
# LEDGER (b): binder arch tags no converter emits.
#
# This ledger was empty until 2026-08-15 — not because there were no gaps, but
# because the discovery regex could not see the 29 binders that spell the
# constant `EXPECTED_ARCH` instead of `ARCH`. Widening it surfaced exactly one
# real gap out of those 29 (`charsiu`), which the 2026-08-21 runtime closure
# subsequently supplied. The other 28 were already emitted by a converter and
# simply had nothing checking them. The guard added in the same change
# (`unseen_arch_spellings`) is what stops the next spelling from re-opening
# this hole.
# ---------------------------------------------------------------------------
declare -a NO_CONVERTER=()

# ---------------------------------------------------------------------------
# LEDGER (a-suppress): literal occurrences that are NOT reader evidence.
#
# Stripping comments and test code removes two whole CLASSES of fake evidence.
# This ledger handles the residue neither can touch: a short arch tag that
# collided with an unrelated string in ordinary production source. Only one
# such collision exists on the 2026-08-15 tree, and it was found by asking
# which answered arches are answered solely by a literal in a module sharing
# no name token with the arch.
#
# Format: 'arch|models-relative-path|reason'. Occurrences of the literal in
# THAT FILE stop counting toward leg (a). Naming the file rather than a line
# number is deliberate — a line number rots on the next edit above it, while
# "this file's occurrences are not readers" stays true and stays checkable.
#
# Double-sided, like every ledger here, and the staleness test is what keeps
# it from becoming a blanket excuse:
#   - the file must exist                                     -> else fail;
#   - it must still contain the literal in scanned source      -> else fail,
#     because an entry that suppresses nothing is an unchecked claim.
# Suppression applies to the literal scan ONLY. A routed constant or a
# BOUND_ARCHES row in engine.rs is an explicit registration, not an
# accidental collision, and is kept honest from the other side by
# scripts/check-bound-arch-coverage.sh.
# ---------------------------------------------------------------------------
declare -a NOT_A_READER=(
  'ast|canary_1b_flash/mod.rs|`Self::Ast => "ast"` (canary_1b_flash/mod.rs:363) is Canary-1B-Flash SPELLING ITS OWN TASK LABEL — "ast" there is Automatic Speech Translation, and Canary1bFlashTask::Ast.as_str() returns it for task routing. It has nothing to do with the Audio Spectrogram Transformer arch that vokra-convert/src/models/ast.rs:39 stamps. Because the match arm is ordinary non-test, non-comment source, neither comment-stripping nor test-stripping can tell the two apart; the three-letter tag simply collides. crates/vokra-models/src/ast/ does not exist, so counting this arm as a reader let a genuinely unbound converter arch report as answered.'
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
# double-sided: an unparsable slug not listed here fails, and a listed
# slug that now parses fails as stale.
# ---------------------------------------------------------------------------
declare -a NOT_A_MODEL_SLUG=(
)

# ---------------------------------------------------------------------------
# LEDGER (d): `vokra.<group>.<key>` chunks a REQUIRED reader looks up that no
# converter stamps.
#
# Empty is the goal state, because every entry here describes a converter that
# writes a GGUF its own binder refuses. That is a strictly worse failure than
# the ones legs (a)/(b) catch: the arch tag matches, the load is attempted,
# and it dies on a missing config chunk.
#
# The five suppression CATEGORIES (S1-S5, see the header) are implemented in
# the leg itself and must not be re-litigated here — an entry in this ledger
# is a claim that the reader genuinely requires the key and the converter
# genuinely does not stamp it.
#
# Format: 'group|reason', keyed on the CHUNK GROUP rather than the individual
# key. A group is stamped or not stamped as a unit — `magnet` is ten keys from
# one `require_u32`/`require_f32` block — and per-key entries would be ten
# copies of one reason that rot together.
#
# Double-sided like every ledger here: an unlisted group fails, and a listed
# group whose converter now stamps it fails as stale.
# ---------------------------------------------------------------------------
declare -a NO_STAMP=(
  'magnet|converter is a BF16 pass-through skeleton and says so in the binder error text: `require_u32` (crates/vokra-models/src/magnet/mod.rs:332-336) refuses with "the converter does not yet emit `vokra.magnet.*` config (BF16 pass-through skeleton only). Extend the converter to stamp this key before loading the GGUF into `MagnetEngine::from_gguf`." Both converters (magnet_small_10secs.rs, magnet_medium_30secs.rs) stamp only arch / name / category / provenance; their sole `vokra.magnet.*` occurrences are `//!` doc comments naming the group as future work. Unlike openwakeword and llama_omni2 this is NOT a silent mismatch — the refusal names the missing key and the reason — but the artifact is still unloadable, so it is recorded rather than tolerated.'
  'melodyflow|same shape as its `magnet` sibling and the same AudioCraft wave: `MelodyFlowConfig::from_gguf` (crates/vokra-models/src/melodyflow/mod.rs:344-355) hard-requires twelve axes through eleven `require_u32` calls plus one `require_f32`, with no all-or-nothing escape (it returns `Result<Self>` at :319, not `Result<Option<Self>>`), while melodyflow_t24_30secs.rs stamps only arch / name / category / provenance. Its one `vokra.melodyflow.*` mention is a `//!` doc comment explaining that naming the chunk keys early "would force a rename cycle" — i.e. the gap is deliberate and dated, not overlooked.'
)

usage() {
    cat <<'USAGE'
check-arch-handshake.sh — converter <-> binder arch handshake gate

Usage:
  bash scripts/check-arch-handshake.sh
  bash scripts/check-arch-handshake.sh --help
  bash scripts/check-arch-handshake.sh --self-test

An arch constant is `pub const ARCH`, `pub const ARCH_<SUFFIX>` or
`pub const EXPECTED_ARCH` typed `&str`. A parser guard fails the run if any
arch-shaped `&str` constant on disk falls outside that set, so a discovery
regex that stops matching cannot report a smaller clean population.

Leg (a): every arch constant under crates/vokra-convert/src/models/ is
answered by a reader — the arch literal in non-comment source under
crates/vokra-models/src/, or a routed constant / BOUND_ARCHES row in
crates/vokra-cli/src/engine.rs.

Leg (b): every arch constant under crates/vokra-models/src/ is emitted by some
converter — the arch literal in non-comment source under
crates/vokra-convert/src/.

For both legs, "source" excludes comments, `#[cfg(test)]` regions and
tests.rs files: none of them can implement a reader or an emitter, and a test
asserting that a binder REFUSES an arch is evidence of the opposite. The
NOT_A_READER ledger retracts individual files for the residue neither
exclusion reaches — an arch tag colliding with an unrelated production string.

Leg (c): every `convert --model <slug>` printed anywhere under
crates/vokra-models/src/ (comments included) parses through
ModelKind::from_arg in crates/vokra-convert/src/lib.rs. A binder that tells an
operator to run a command that does not exist is worse than one that stays
silent.

Leg (d): every `vokra.<group>.<key>` chunk a REQUIRED reader under
crates/vokra-models/src looks up is stamped by some file under
crates/vokra-convert/src. Required means the reader has no escape: absence is
an error, or absence yields a zero sentinel a later gate rejects. Stamped
means the key REACHES CODE — the const name in code position, the key inline
at the call site, or head interpolation (`format!("{KEY_PFX}.kind")`);
declaring `const KEY_X: &str = "vokra.g.x"` and never using it does NOT
count, which is the shape a dropped stamp leaves behind. Five
suppression categories (group-optional `Ok(None)` escapes, caller defaults,
string-literal mentions, runtime-assembled `PREFIX` constants, dead-code
reserved constants) are implemented in the leg and counted in its output.
An arch tag that matches perfectly still describes a GGUF whose config chunk
is empty — that is the failure legs (a)-(c) cannot see.

Accepted gaps live in the NO_READER / NO_CONVERTER / NOT_A_MODEL_SLUG /
NOT_A_READER / NO_STAMP ledgers at the top of this script, one reason each.
All five are double-sided: an undeclared gap fails, and a ledger entry whose
gap has since been closed also fails. Exit 1 on any.
USAGE
}

# The checker. Args:
#   $1 convert models dir (arch constants are declared here)
#   $2 convert src dir    (emitter literals are searched here)
#   $3 models src dir     (binder arch constants AND reader literals AND the
#                          `convert --model` recovery commands leg (c) reads)
#   $4 engine.rs path     (routed constants + BOUND_ARCHES rows)
#   $5 ledger file for leg (a), $6 ledger file for leg (b), $8 ledger file for
#      leg (c), ${10} ledger file for leg (d); all 'key|reason' per line, blank
#      lines and #-comments ignored.
#   $7 convert lib.rs path (leg (c) reads `ModelKind::from_arg` out of it)
# stdlib only. Reused verbatim by the main path and --self-test.
run_check() {
    python3 - "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" <<'PY'
import os, re, sys

(conv_models, conv_src, models_dir, engine_path, ledger_a, ledger_b,
 convert_lib, ledger_c, ledger_sup, ledger_d) = sys.argv[1:11]

# BOTH binder spellings are in scope. `EXPECTED_ARCH` is not a stylistic
# variant nobody uses: it is what 29 of the 89 arch constants under
# vokra-models/src are called (charsiu, csm, moshi, silero-vad, voxtral,
# zonos, the whole chatterbox family, …). Until 2026-08-15 this regex matched
# only the `ARCH` form, so this gate and its sibling both reported a confident
# green over a population missing a third of the binders — `charsiu` among
# them, whose then-missing converter was precisely the defect leg (b) was
# meant to catch. A gate that is green because it did not look is worse than
# no gate: it certifies the thing it failed to check. LOOSE_ARCH_CONST below
# is what keeps the NEXT spelling from going invisible the same way.
ARCH_CONST = re.compile(
    r'^\s*pub\s+const\s+((?:EXPECTED_)?ARCH(?:_[A-Z0-9_]+)?)\s*:\s*&(?:\'static\s+)?str\s*=\s*"([^"]+)"\s*;'
)
# Deliberately sloppy twin of ARCH_CONST: ANY `pub const <name>: &str = "…";`
# whose name contains `ARCH`. Never used for discovery — only to prove that
# discovery saw everything on disk that looks like an arch declaration. See
# `unseen_arch_spellings`.
LOOSE_ARCH_CONST = re.compile(
    r'^\s*pub\s+const\s+([A-Z0-9_]*ARCH[A-Z0-9_]*)\s*:\s*&(?:\'static\s+)?str\s*=\s*"([^"]+)"\s*;'
)
STRING_LIT = re.compile(r'"((?:[^"\\]|\\.)*)"')

# ---- test-code exclusion --------------------------------------------------
# Any `#[cfg(...)]` whose predicate mentions a bare `test`, EXCEPT a
# `not(test)` (which marks code compiled only OUTSIDE tests — production
# source that must keep counting). All 415 such attributes on the
# 2026-08-15 tree are the plain `#[cfg(test)]`, but matching the family
# means an `#[cfg(all(test, feature = "x"))] mod tests` cannot silently
# re-open the hole the way a stricter literal match would.
#
# Quoted spans are dropped from the predicate before the token test, so
# `#[cfg(feature = "test-utils")]` is NOT read as a test gate — that one
# guards production source, and skipping it would drop real readers.
CFG_TEST_ATTR = re.compile(r'^\s*#\[\s*cfg\s*\((?P<pred>.*)\)\s*\]\s*$')
BARE_TEST = re.compile(r'\btest\b')
NOT_TEST = re.compile(r'\bnot\s*\(\s*test\s*\)')
# Char literal, so a `'{'` cannot be mistaken for a real brace.
CHAR_LIT = re.compile(r"'(?:\\.|[^'\\])'")

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

# ---- leg (d) patterns -----------------------------------------------------
# A `vokra.<group>.<key>` chunk name. Anchored: the literal must be the WHOLE
# string, so a diagnostic sentence that happens to contain a key is not
# mistaken for the key itself.
META_KEY = re.compile(r"^vokra\.[A-Za-z0-9_]+\.[A-Za-z0-9_.]+$")
# `const NAME: &str = "vokra.…";` at any visibility — `pub`, `pub(crate)` and
# private all bind keys, unlike the arch constants legs (a)/(b) discover.
META_CONST = re.compile(
    r'^\s*(?:pub(?:\([^)]*\))?\s+)?const\s+([A-Z][A-Z0-9_]*)\s*:\s*&(?:\'static\s+)?str'
    r'\s*=\s*"(vokra\.[^"]+)"\s*;'
)
FN_SIG = re.compile(r'^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([a-z_][a-z0-9_]*)\s*[(<]')
# S1: the group-optional escape. Either spelling is decisive on its own.
OPTION_RETURN = re.compile(r'->\s*(?:Result\s*<\s*)?Option\s*<')
RETURN_OK_NONE = re.compile(r'\bOk\s*\(\s*None\s*\)')
# The same escape spelled at the CALL SITE: the read yields an `Option` the
# caller destructures, so a missing key leaves the field at its default.
OPTION_BIND = re.compile(r'\b(?:if|while)\s+let\s+Some\s*\(|\.is_some\s*\(\)|\.is_none\s*\(\)')
# S2: a real caller-supplied fallback. `unwrap_or_default()` is included
# because a `Default` impl is a declared value, unlike the bare `0` below.
FALLBACK_CALL = re.compile(r'\.unwrap_or(?:_else|_default)?\s*\(')
# The `unwrap_or(<expr>)` ARGUMENT, so the same ZERO-SENTINEL test the
# `None => Ok(<expr>)` arm gets can be applied here too. Without it, the
# round-9 defect spelled `.unwrap_or(0)` instead of `None => Ok(0)` files
# itself as an S2 caller default and disappears into the suppression count —
# in the one leg that exists to stop a third recurrence of that class.
# `unwrap_or_else` / `unwrap_or_default` are deliberately NOT matched here:
# the first takes a closure (its body is a real computed default) and the
# second names a `Default` impl, which is a declared value.
FALLBACK_ARG = re.compile(r'\.unwrap_or\s*\(\s*([^,()\n]*?)\s*\)')
# `None => Ok(<expr>)`. Captured so a ZERO SENTINEL can be told from a real
# default — the distinction `llama_omni2` turns on.
NONE_ARM = re.compile(r'None\s*=>\s*Ok\s*\(\s*([^,\n]*?)\s*\)\s*,')
SENTINEL_VALUE = re.compile(
    r'^(?:0|0u\d+|0i\d+|0\.0|0\.0f\d+|""|String::new\(\)|Vec::new\(\)|Default::default\(\))$'
)
# S4: a constant whose value is a prefix an index is appended to at runtime.
# Matched as a SCREAMING_SNAKE segment, not with `\b`: the two real spellings
# are `GGUF_KEY_PATCH_GRID_PREFIX` and `PREFIX_DELAY`, and `\bPREFIX\b` finds
# neither, because `_` is a word character on both sides.
PREFIX_NAME = re.compile(r'(?:^|_)PREFIX(?:_|$)')
# S5: a reserved constant nothing reads yet.
DEAD_CODE_ATTR = re.compile(r'^\s*#\[\s*(?:allow|expect)\s*\(\s*dead_code\s*\)\s*\]')
# Emitter-side use detection (see `meta_stamped`). A SCREAMING_SNAKE name in
# code position; `\b` is enough because a path-qualified `chunks::KEY_FOO`
# still starts the name at a word boundary.
CONST_USE = re.compile(r'\b([A-Z][A-Z0-9_]*)\b')
# A string literal that IS a key being assembled from a head constant:
# `{KEY_WORDPIECE_PREFIX}.kind`, `{KEY_WAVLM_CONV_DIM_PREFIX}_{i}`. The
# no-whitespace tail is the discriminator against a diagnostic sentence.
ASSEMBLED_KEY = re.compile(r'^\{([A-Z][A-Z0-9_]*)\}(\S*)$')
# The tail of an assembled key, when it is literal enough to name exactly.
KEY_TAIL = re.compile(r'[A-Za-z0-9_.]*')
SOME_ARG = re.compile(r'^\s*Some\s*\(')


def function_call_arguments(source, name):
    """Yield balanced argument text for calls to `name` in sanitized code."""
    cursor = 0
    while True:
        start = source.find(name, cursor)
        if start < 0:
            return
        before = source[start - 1] if start else ""
        after_name = start + len(name)
        if before.isalnum() or before == "_":
            cursor = after_name
            continue
        opening = after_name
        while opening < len(source) and source[opening].isspace():
            opening += 1
        if opening >= len(source) or source[opening] != "(":
            cursor = after_name
            continue
        depth = 1
        end = opening + 1
        while end < len(source) and depth:
            if source[end] == "(":
                depth += 1
            elif source[end] == ")":
                depth -= 1
            end += 1
        if depth:
            return
        yield source[opening + 1 : end - 1]
        cursor = end


def top_level_arguments(call):
    """Split one balanced call at commas outside nested expressions."""
    out, depth, start = [], 0, 0
    for index, char in enumerate(call):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            out.append(call[start:index])
            start = index + 1
    out.append(call[start:])
    return out


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


def unseen_arch_spellings(root):
    """{const_name: 'relpath:lineno'} for arch-shaped consts DISCOVERY MISSED.

    The failure mode this catches is the one that motivated the 2026-08-15
    widening: a binder module spells its arch constant in a way ARCH_CONST
    does not match, so the gate silently scans a smaller population and
    passes cleanly over every module it never looked at. Nothing else in
    this script can notice that — every leg is phrased over the constants
    discovery *found*, so a discovery bug reads as "no gaps".

    WHY THIS SHAPE AND NOT A COUNT BAND
    A "discovered count is within N of the module count on disk" check
    needs a threshold, and the threshold is unpickable: modules
    legitimately declare zero arch constants (pure ops, helper modules) or
    several (gigaam's ARCH_V3 + ARCH_MULTILINGUAL, whisper's size
    siblings), so today's honest ratio is 89 constants across 86 declaring
    files out of far more files overall. A band loose enough to avoid
    false alarms would have happily accepted 60-of-89. Running the strict
    regex against a deliberately sloppy twin has no threshold at all, and
    instead of "the number moved" it names the exact unmatched spelling
    and the file it is in. It is also self-maintaining: a genuinely new
    spelling fails once, loudly, at the line that introduced it.

    WHAT THIS GUARD DOES *NOT* COVER — measured, not assumed [2026-08-15]
    It keys on the constant NAME while still requiring `pub `, so a
    declaration hidden by VISIBILITY rather than by spelling stays
    invisible to it. That is not hypothetical: 71 converter-side and 3
    models-side arch constants are `pub(crate)` or private and sit outside
    discovery for that reason (`pub(crate) const ARCH: &str = "whisper"` at
    vokra-convert/src/models/whisper.rs:212 is typical). Those 74 were NOT
    triaged in the change that added this guard, and at least one is a real
    defect: `ARCH_CRISPERWHISPER` ("crisper-whisper",
    vokra-convert/src/models/whisper.rs:225) is a converter arch that
    `ModelKind::from_arg` accepts and `whisper::ACCEPTED_ARCHS` loads, yet
    engine.rs matches `arch.as_str()` against `ARCH_WHISPER` alone
    (engine.rs:353) — so `convert` succeeds and `run` answers "unsupported
    model arch". Widening to `pub(crate)` means triaging ~74 constants
    first, so it is deliberately left as a separate, scoped change rather
    than smuggled in here. Do not read a pass from this guard as "every
    arch constant in the tree is covered".
    """
    unseen = {}
    for path in rust_files(root):
        rel = os.path.relpath(path, root)
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                m = LOOSE_ARCH_CONST.match(line)
                if m and not ARCH_CONST.match(line):
                    unseen.setdefault(m.group(1), f"{rel}:{lineno}")
    return unseen


def brace_source(line):
    """`line` with string literals, char literals and the `//` tail removed.

    Only ever used for brace COUNTING, so that a `"{"` inside a string or a
    `//` comment cannot shift the depth of a `#[cfg(test)]` block and make
    it end early — ending early is the silent direction, which is the one
    that matters here.
    """
    out, i, n = [], 0, len(line)
    while i < n:
        c = line[i]
        if c == '"':
            i += 1
            while i < n:
                if line[i] == "\\":
                    i += 2
                    continue
                if line[i] == '"':
                    i += 1
                    break
                i += 1
            continue
        if c == "'":
            m = CHAR_LIT.match(line, i)
            if m:
                i = m.end()
                continue
            i += 1                      # a lifetime tick, not a literal
            continue
        if c == "/" and i + 1 < n and line[i + 1] == "/":
            break
        out.append(c)
        i += 1
    return "".join(out)


def test_region_lines(path):
    """({1-based line numbers inside a #[cfg(test)] item}, unclosed_lineno).

    Walks from each test attribute to the end of the item it guards:
    brace-balanced for `mod tests { … }` / `fn … { … }`, or the single
    line for a `;`-terminated `mod tests;` / `use …;`. Extra attributes
    between the `#[cfg(test)]` and the item carry neither braces nor a
    `;`, so they are simply consumed on the way.

    `unclosed_lineno` is set when a region runs off the end of the file
    without balancing. That is reported as a parser guard rather than
    swallowed: silently skipping the rest of a file is how a scan starts
    covering less than it claims.
    """
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    skip, i, unclosed = set(), 0, None
    while i < len(lines):
        m = CFG_TEST_ATTR.match(lines[i])
        pred = STRING_LIT.sub("", m.group("pred")) if m else ""
        if not m or not BARE_TEST.search(NOT_TEST.sub("", pred)):
            i += 1
            continue
        depth, started, j = 0, False, i
        while j < len(lines):
            src = brace_source(lines[j])
            depth += src.count("{") - src.count("}")
            skip.add(j + 1)
            if not started and depth > 0:
                started = True
            if started and depth <= 0:
                break
            if not started and src.rstrip().endswith(";"):
                break                   # `mod tests;` / `use …;`
            j += 1
        if j >= len(lines):
            unclosed = i + 1
        i = j + 1
    return skip, unclosed


def literal_sites(root):
    """({literal: {relpath, …}}, [parser problems]) over REAL source.

    Three kinds of text are excluded, all for the same reason — none of
    them can implement a reader or an emitter, so none may satisfy this
    gate:

      - comment lines: a doc comment that merely names an arch;
      - `#[cfg(test)]` regions;
      - whole `tests.rs` files (the repo's file-per-module test layout).

    Sites are tracked per literal so the NOT_A_READER ledger can retract
    one file's occurrences without retracting the literal everywhere.
    """
    out, problems = {}, []
    for path in rust_files(root):
        rel = os.path.relpath(path, root)
        if os.path.basename(path) == "tests.rs":
            continue
        skip, unclosed = test_region_lines(path)
        if unclosed is not None:
            problems.append(
                f"{rel}:{unclosed}: a `#[cfg(test)]` region never closes — brace "
                f"counting ran to end of file. The scan would have skipped "
                f"everything after it, so this fails instead of quietly covering "
                f"less than it reports."
            )
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if lineno in skip or line.lstrip().startswith("//"):
                    continue
                for m in STRING_LIT.finditer(line):
                    out.setdefault(m.group(1), set()).add(rel)
    return out, problems


def answering_literals(sites, suppressed):
    """{literal} that some NON-suppressed file contains.

    `suppressed` is {literal: {relpath, …}} from NOT_A_READER. A literal
    whose every site is retracted stops answering.
    """
    return {
        lit for lit, files in sites.items()
        if files - suppressed.get(lit, set())
    }


def split_code_and_strings(path):
    """([code_line, …], {lineno: [literal, …]}) — strings lifted out of code.

    Leg (d) turns on telling a key used in CODE position (a lookup) from the
    same key MENTIONED inside a diagnostic (S3). A line-at-a-time regex
    cannot do that, because Rust strings legally span lines: the second
    line of a wrapped `format!` message looks like bare code to it, so a
    `{KEY_FOO}` interpolation inside an error reads as a lookup of
    `KEY_FOO`. That is not hypothetical — it is `vokra.gigaam.required_tensors`
    and `vokra.snac.codebook_tables`, both of which a naive scan reports as
    unstamped required reads when neither is read at all.

    Comments are blanked too, so a doc comment naming a key cannot count.
    Raw strings (`r"…"`, `r#"…"#`) are handled because converter docs use
    them for JSON samples.
    """
    text = open(path, encoding="utf-8").read()
    code, literal_values = [], {}
    line = []
    lineno = 1
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "\n":
            code.append("".join(line))
            line = []
            lineno += 1
            i += 1
            continue
        # line comment: blank to end of line
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        # block comment
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                if text[i] == "\n":
                    code.append("".join(line))
                    line = []
                    lineno += 1
                i += 1
            i += 2
            continue
        # raw string r"…" / r#"…"#
        if c == "r" and i + 1 < n and text[i + 1] in '#"':
            j = i + 1
            hashes = 0
            while j < n and text[j] == "#":
                hashes += 1
                j += 1
            if j < n and text[j] == '"':
                start_line = lineno
                j += 1
                buf = []
                close = '"' + "#" * hashes
                while j < n and not text.startswith(close, j):
                    if text[j] == "\n":
                        code.append("".join(line))
                        line = []
                        lineno += 1
                    else:
                        buf.append(text[j])
                    j += 1
                literal_values.setdefault(start_line, []).append("".join(buf))
                i = j + len(close)
                continue
        if c == '"':
            start_line = lineno
            i += 1
            buf = []
            while i < n and text[i] != '"':
                if text[i] == "\\":
                    # A backslash-newline is a Rust line continuation: the
                    # newline and the next line's indent vanish from the
                    # value but the FILE still advanced a line.
                    if i + 1 < n and text[i + 1] == "\n":
                        code.append("".join(line))
                        line = []
                        lineno += 1
                        i += 2
                        while i < n and text[i] in " \t":
                            i += 1
                        continue
                    buf.append(text[i : i + 2])
                    i += 2
                    continue
                if text[i] == "\n":
                    code.append("".join(line))
                    line = []
                    lineno += 1
                else:
                    buf.append(text[i])
                i += 1
            literal_values.setdefault(start_line, []).append("".join(buf))
            i += 1
            continue
        if c == "'":
            m = CHAR_LIT.match(text, i)
            if m:
                i = m.end()
                continue
            line.append(c)
            i += 1
            continue
        line.append(c)
        i += 1
    code.append("".join(line))
    return code, literal_values


def fn_spans(lines):
    """[(start_idx, end_idx, sig_line, body_text)] for every `fn` in a file.

    Brace-balanced over `brace_source`, so a `{` inside a string or comment
    cannot end a body early. Used for S1: whether the function enclosing a
    read has a whole-group-absent escape.
    """
    spans = []
    for i, l in enumerate(lines):
        if not FN_SIG.match(l):
            continue
        depth, started = 0, False
        j = i
        while j < len(lines):
            src = brace_source(lines[j])
            depth += src.count("{") - src.count("}")
            if not started and depth > 0:
                started = True
            if started and depth <= 0:
                break
            if not started and src.rstrip().endswith(";"):
                break               # a trait method declaration, no body
            j += 1
        spans.append((i, min(j, len(lines) - 1), l, "\n".join(lines[i : j + 1])))
    return spans


def enclosing_fn(spans, idx):
    """The innermost span containing `idx`, or None."""
    best = None
    for s in spans:
        if s[0] <= idx <= s[1] and (best is None or s[0] >= best[0]):
            best = s
    return best


def stmt_window(lines, idx):
    """The whole statement around line `idx`, as one string.

    Wide enough that a `?` on a later physical line still counts — a
    multi-line call like `read_u32_key(\\n gguf,\\n KEY,\\n )?;` must not
    read as an infallible lookup — and bounded so it never swallows a
    neighbouring statement whose fallback would then be misattributed.
    """
    start = idx
    while start > 0:
        prev = brace_source(lines[start - 1]).rstrip()
        if prev == "" or prev.endswith((";", "{", "}", ",")):
            break
        start -= 1
    end = idx
    depth = 0
    while end < len(lines):
        src = brace_source(lines[end])
        depth += src.count("(") - src.count(")")
        stripped = src.rstrip()
        if depth <= 0 and stripped.endswith((";", ",")):
            break
        if end + 1 < len(lines) and lines[end + 1].lstrip().startswith("."):
            end += 1
            continue
        if depth <= 0 and end > idx:
            break
        end += 1
    return "\n".join(lines[start : min(end + 1, len(lines))])


def absence_tolerant(expr):
    """True when a read expression tolerates the key being absent (S2).

    Four shapes, all meaning "absence is a value, not a failure":

      - an explicit fallback call (`unwrap_or`, `unwrap_or_else`,
        `unwrap_or_default`);
      - a `None => Ok(<value>)` arm whose value is a real one;
      - an OPTION-RETURNING read — the helper hands back `Option<T>` or
        `Result<Option<T>>` and the call site consumes it as an option
        (`if let Some(v) = opt_usize(file, KEY)?`). A function that answers
        `None` for a missing key is tolerating absence by construction.
        This is `canary_1b_flash`, whose per-key `opt_*` helpers leave the
        field at its primary-source default and merely stop incrementing a
        `stamped` counter — the escape lives at the READ SITE rather than
        on the enclosing `from_gguf`, so the S1 check cannot see it;
      - an INFALLIBLE read — no `?` and no `return Err`, so the expression
        has no failure path at all. That is `panns`, whose closure takes a
        `fallback: u32` parameter and whose `from_gguf` returns `Self`
        rather than `Result<Self>`.

    A ZERO SENTINEL is deliberately NOT tolerance: `None => Ok(0)` hands
    back a value no forward pass can use, which is `llama_omni2`'s
    `read_u32_or_zero` and precisely the round-9 defect. Treating it as a
    default would suppress the very thing this leg was added to catch.
    """
    if FALLBACK_CALL.search(expr):
        # A bare `.unwrap_or(0)` is the round-9 defect wearing a different
        # spelling, so it gets the same sentinel test as the `None => Ok(0)`
        # arm below rather than being waved through as a caller default.
        for m in FALLBACK_ARG.finditer(expr):
            if SENTINEL_VALUE.match(m.group(1).strip()):
                return False
        return True
    for m in NONE_ARM.finditer(expr):
        if not SENTINEL_VALUE.match(m.group(1).strip()):
            return True
    if OPTION_RETURN.search(expr) or OPTION_BIND.search(expr):
        return True
    if "?" not in expr and "return Err" not in expr and "Err(" not in expr:
        return True
    return False


def meta_read_keys(root):
    """({key: [site, …]}, {category: count}) — REQUIRED metadata reads.

    Walks every non-comment, non-test line under `root`, resolves each
    `vokra.<group>.<key>` to its read site (through a `const NAME: &str`
    binding or as an inline literal), and applies suppression categories
    S1-S5 (see the header). Every suppression is counted so a run that
    dropped everything reports it instead of printing a clean line.
    """
    out, counts = {}, {}
    def bump(cat):
        counts[cat] = counts.get(cat, 0) + 1

    for path in rust_files(root):
        rel = os.path.relpath(path, root)
        if os.path.basename(path) == "tests.rs":
            continue
        skip, _ = test_region_lines(path)
        lines = open(path, encoding="utf-8").read().split("\n")
        code, _literal_values = split_code_and_strings(path)
        while len(code) < len(lines):
            code.append("")
        spans = fn_spans(lines)

        # Const bindings, plus the inline literals that are their own key.
        binds, inline = {}, {}
        for i, line in enumerate(lines):
            if (i + 1) in skip:
                continue
            m = META_CONST.match(line)
            if m:
                name, key = m.group(1), m.group(2)
                if not META_KEY.match(key):
                    continue
                if PREFIX_NAME.search(name):
                    bump("S4 runtime-assembled prefix")
                    continue
                if i > 0 and DEAD_CODE_ATTR.match(lines[i - 1]):
                    bump("S5 dead-code reserved constant")
                    continue
                binds[name] = key
                continue
            for lit in _literal_values.get(i + 1, []):
                if META_KEY.match(lit):
                    inline.setdefault(lit, []).append(i)

        # Code-position references. `code` has strings blanked out, so a key
        # named inside a diagnostic is invisible here — that IS S3.
        seen_in_code = set()
        for name, key in binds.items():
            pat = re.compile(r"\b" + re.escape(name) + r"\b")
            for i, cline in enumerate(code):
                if (i + 1) in skip or META_CONST.match(lines[i]):
                    continue
                if not pat.search(cline):
                    continue
                seen_in_code.add(key)
                span = enclosing_fn(spans, i)
                if span is not None and (
                    OPTION_RETURN.search(span[2]) or RETURN_OK_NONE.search(span[3])
                ):
                    bump("S1 group-optional escape")
                    continue
                expr = stmt_window(lines, i)
                # Resolve a helper called on this line to its body: the
                # `None => Ok(0)` that decides sentinel-vs-default usually
                # lives there, not at the call site.
                for hm in re.finditer(r"\b([a-z_][a-z0-9_]*)\s*\(", cline):
                    for s in spans:
                        if FN_SIG.match(s[2]) and FN_SIG.match(s[2]).group(1) == hm.group(1):
                            expr = expr + "\n" + s[3]
                            break
                if absence_tolerant(expr):
                    bump("S2 caller default")
                    continue
                out.setdefault(key, []).append(f"{rel}:{i + 1}")
        for key, idxs in inline.items():
            for i in idxs:
                if not re.search(r"\bget\s*\(|\bcontains_key\s*\(", code[i]) and \
                        not re.search(r'"', lines[i].split("//")[0]):
                    continue
                seen_in_code.add(key)
                span = enclosing_fn(spans, i)
                if span is not None and (
                    OPTION_RETURN.search(span[2]) or RETURN_OK_NONE.search(span[3])
                ):
                    bump("S1 group-optional escape")
                    continue
                if absence_tolerant(stmt_window(lines, i)):
                    bump("S2 caller default")
                    continue
                out.setdefault(key, []).append(f"{rel}:{i + 1}")
        for key in set(binds.values()) - seen_in_code:
            bump("S3 mention, not a read")
    return out, counts


def meta_stamped(root):
    """{key} every `vokra.<group>.<key>` some converter writes, + prefixes.

    Comments and test code are excluded for the same reason as legs
    (a)/(b): a doc comment naming a chunk group does not stamp it, and a
    test fixture that builds a GGUF by hand is not the converter.

    Prefix constants are returned too, so a runtime-assembled
    `vokra.moshi.delay.0` is answered by the stamped `vokra.moshi.delay.`
    rather than read as missing.

    DECLARING A KEY IS NOT STAMPING IT [2026-08-15]
    Until this rewrite the whole function was "harvest every `vokra.*`
    string literal in non-comment, non-test converter source", and a
    `pub const KEY_N_WAKEWORDS: &str = "vokra.openwakeword.n_wakewords";`
    IS such a literal. So the DECLARATION of a key constant was accepted as
    proof the key gets written. Deleting the one
    `b.add_u32(KEY_N_WAKEWORDS, …)` call from openwakeword_op.rs — which
    reproduces the exact round-8 defect this leg was added to catch — left
    every number in the leg's output byte-identical, and it printed OK.
    Deleting six of the seven did too.

    The reader half never had this bug: `meta_read_keys` builds `binds`
    from the declarations and then requires the NAME to appear in CODE
    position on a non-declaration line, counting the difference as S3
    "mention, not a read". This half now asks the mirrored question, so the
    two halves are finally symmetric: a literal must REACH CODE, not merely
    be bound to a name.

    A use is one of four shapes:
      - the const NAME in code position (`b.add_u32(KEY_FOO, v)`,
        `write_u32_array(b, KEY_CONV_DIM, &CONV_DIM)`, a `[KEY_A, KEY_B]`
        array a loop later stamps);
      - the key spelled inline at the call site
        (`b.add_u32("vokra.foo.bar", v)`);
      - HEAD INTERPOLATION — a string literal that IS `{NAME}` followed by
        key-shaped text, i.e. `format!("{KEY_WORDPIECE_PREFIX}.kind")` or
        `format!("{KEY_WAVLM_CONV_DIM_PREFIX}_{i}")`. Strings are blanked
        out of `code`, so without this the prefix builders read as
        declaration-only. Requiring the interpolation at the HEAD and no
        whitespace after it is what separates key construction from a
        diagnostic that merely names a key: `"{KEY_FOO} is missing"` has a
        space, `"missing {k}"` interpolates a lowercase local. Both the
        head const's own value and the assembled key are recorded.
      - the shared `stamp_provenance` writer. Its implementation lives in
        `vokra-core`, outside this converter-tree scan: weight-license is
        unconditional, while `model_id` and `source` count only when their
        respective call arguments are visibly `Some(..)`. A `None` must not
        satisfy a required reader.

    Resolution is same-file first, then tree-wide for names that bind
    exactly one key. Ambiguity is real — `KEY_SAMPLE_RATE` is declared in
    several converters with a different `vokra.<group>.` each — so a
    cross-file use of an ambiguous name is deliberately NOT resolved.

    FAILURE DIRECTION: a bug in this scan shrinks the stamped set, which
    makes the gate FAIL loudly on keys that are in fact written. That is
    the opposite of the bug it replaces, which made the gate pass over
    keys that were not.
    """
    per_file, name_keys, files = {}, {}, []
    for path in rust_files(root):
        if os.path.basename(path) == "tests.rs":
            continue
        skip, _ = test_region_lines(path)
        lines = open(path, encoding="utf-8").read().split("\n")
        code, literal_values = split_code_and_strings(path)
        while len(code) < len(lines):
            code.append("")
        decls = {}
        for i, line in enumerate(lines):
            if (i + 1) in skip:
                continue
            m = META_CONST.match(line)
            if m:
                decls[m.group(1)] = m.group(2)
                name_keys.setdefault(m.group(1), set()).add(m.group(2))
        per_file[path] = decls
        files.append((path, skip, lines, code, literal_values))

    def resolve(name, decls):
        if name in decls:
            return decls[name]
        bound = name_keys.get(name)
        if bound and len(bound) == 1:
            return next(iter(bound))
        return None

    keys = set()
    for path, skip, lines, code, literal_values in files:
        decls = per_file[path]
        for i, line in enumerate(lines):
            # The declaration line itself is not a use — the mirror of the
            # `META_CONST.match(lines[i]): continue` guard on the reader
            # side. This single `continue` is what gives the leg teeth.
            if (i + 1) in skip or META_CONST.match(line):
                continue
            for v in literal_values.get(i + 1, []):
                if META_KEY.match(v) or v.startswith("vokra."):
                    keys.add(v)
                am = ASSEMBLED_KEY.match(v)
                if am:
                    base = resolve(am.group(1), decls)
                    if base:
                        keys.add(base)
                        if KEY_TAIL.fullmatch(am.group(2)):
                            keys.add(base + am.group(2))
            for m in CONST_USE.finditer(code[i]):
                base = resolve(m.group(1), decls)
                if base:
                    keys.add(base)

        # `vokra_core::stamp_provenance` is a real metadata writer whose
        # body is deliberately outside `vokra-convert/src`. Parse balanced
        # calls from string/comment-sanitized production code rather than
        # treating a mere import or diagnostic mention as a write.
        production = "\n".join(
            line if index + 1 not in skip else ""
            for index, line in enumerate(code)
        )
        for call in function_call_arguments(production, "stamp_provenance"):
            args = top_level_arguments(call)
            if len(args) < 5:
                continue
            keys.add("vokra.provenance.weight_license")
            if SOME_ARG.match(args[3]):
                keys.add("vokra.provenance.model_id")
            if SOME_ARG.match(args[4]):
                keys.add("vokra.provenance.source")
    return keys


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
    literal_values = set()
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
                    literal_values.add(m.group(1))
    return literal_values


def read_suppress_ledger(path):
    """({literal: {relpath, …}}, [problems]) from 'arch|path|reason' lines."""
    entries, bad = {}, []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|", 2)
            if len(parts) != 3 or not all(p.strip() for p in parts):
                bad.append(
                    f"malformed NOT_A_READER line (want 'arch|path|reason'): {line!r}"
                )
                continue
            arch, rel, _reason = (p.strip() for p in parts)
            if rel in entries.setdefault(arch, set()):
                bad.append(f"duplicate NOT_A_READER entry for `{arch}` at `{rel}`")
            entries[arch].add(rel)
    return entries, bad


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
reader_sites, reader_problems = literal_sites(models_dir)
emitter_sites, emitter_problems = literal_sites(conv_src)
cmd_slugs, cmd_dangling = model_cmd_slugs(models_dir)
accepted_slugs = from_arg_literals(convert_lib)
ledger_no_reader, ledger_a_bad = read_ledger(ledger_a)
ledger_no_conv, ledger_b_bad = read_ledger(ledger_b)
ledger_not_slug, ledger_c_bad = read_ledger(ledger_c)
suppressed, ledger_sup_bad = read_suppress_ledger(ledger_sup)
ledger_no_stamp, ledger_d_bad = read_ledger(ledger_d)
required_keys, suppress_counts = meta_read_keys(models_dir)
stamped_keys = meta_stamped(conv_src)

reader_literals = answering_literals(reader_sites, suppressed)
# Leg (b) applies the same test-code exclusion but no suppression: no
# converter-side collision exists, and an empty ledger nobody can populate
# is worse than none. Mirror NOT_A_READER here if one ever appears.
emitter_literals = answering_literals(emitter_sites, {})

errors = (list(ledger_a_bad) + list(ledger_b_bad) + list(ledger_c_bad)
          + list(ledger_sup_bad) + list(ledger_d_bad) + list(reader_problems)
          + list(emitter_problems))

# NOT_A_READER staleness: an entry must name a file that exists AND still
# contains the literal in scanned source. An entry that retracts nothing is
# an unchecked claim, which is what every ledger here exists to prevent.
for arch in sorted(suppressed):
    for rel in sorted(suppressed[arch]):
        if not os.path.isfile(os.path.join(models_dir, rel)):
            errors.append(
                f"[leg a] STALE NOT_A_READER entry `{arch}` -> `{rel}`: that file does "
                f"not exist under vokra-models/src/. Fix: update the path, or delete the "
                f"entry from NOT_A_READER in scripts/check-arch-handshake.sh."
            )
        elif rel not in reader_sites.get(arch, set()):
            errors.append(
                f"[leg a] STALE NOT_A_READER entry `{arch}` -> `{rel}`: the literal "
                f"\"{arch}\" no longer appears in scanned (non-comment, non-test) source "
                f"in that file, so the entry suppresses nothing and its claim is no longer "
                f"checked. Fix: delete the `{arch}|{rel}` line from NOT_A_READER in "
                f"scripts/check-arch-handshake.sh."
            )

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
if not reader_sites:
    errors.append(
        f"zero string literals scanned in non-comment, non-test source under "
        f"{models_dir} — the reader scan is broken; every converter arch would read as "
        f"unanswered."
    )
if not emitter_sites:
    errors.append(
        f"zero string literals scanned in non-comment, non-test source under "
        f"{conv_src} — the emitter scan is broken; every binder arch would read as "
        f"unemitted."
    )
# The discovery-coverage guard. Everything above notices a scan that found
# NOTHING; this one notices a scan that found SOME — the far more dangerous
# shape, because a partial population still prints a confident count. See
# `unseen_arch_spellings` for why this is a regex-vs-regex check rather than
# a count band.
for _root, _rootname, _leg in (
    (conv_models, "vokra-convert/src/models", "a"),
    (models_dir, "vokra-models/src", "b"),
):
    for _name, _where in sorted(unseen_arch_spellings(_root).items()):
        errors.append(
            f"[guard] `pub const {_name}` at {_rootname}/{_where} declares an arch-shaped "
            f"`&str` constant that the discovery regex does NOT match, so leg ({_leg}) never "
            f"looked at it — and every OTHER declaration spelled `{_name}` is invisible too. "
            f"This is the failure that let 29 `EXPECTED_ARCH` binders (charsiu among them) sit "
            f"outside both arch gates until 2026-08-15 while they reported a clean green. Fix: "
            f"widen ARCH_CONST in scripts/check-arch-handshake.sh AND in "
            f"scripts/check-bound-arch-coverage.sh to cover `{_name}` (they discover "
            f"independently, so both must change), then re-run both gates and expect NEW "
            f"findings — or rename the constant to an already-covered spelling."
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
        f"changed; leg (c) would then report every recovery-command slug as unparsable."
    )
if not cmd_slugs:
    errors.append(
        f"no `convert --model <slug>` found anywhere under {models_dir} — the scan or the "
        f"command spelling changed; leg (c) covered nothing, so a pass here would be "
        f"vacuous."
    )

# ---- leg (a): converter -> reader ----------------------------------------
answered = reader_literals | routed | registry
gap_a = {}
for arch, where, const_name in converters:
    if arch not in answered:
        gap_a.setdefault(arch, f"{const_name} at vokra-convert/src/models/{where}")

for arch in sorted(gap_a):
    if arch not in ledger_no_reader:
        errors.append(
            f"[leg a] converter arch `{arch}` ({gap_a[arch]}) has NO reader: the literal "
            f"appears nowhere in non-comment, NON-TEST source under vokra-models/src/, and "
            f"it is neither routed nor a BOUND_ARCHES row in vokra-cli/src/engine.rs. "
            f"(If this arch used to pass: `#[cfg(test)]` blocks and tests.rs files stopped "
            f"counting on 2026-08-15. A test that asserts some OTHER binder REFUSES this "
            f"arch is evidence of the opposite of a reader.) The "
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
    if arch not in emitter_literals:
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

# ---- leg (d): metadata key -> converter stamp ----------------------------
# Grouped by chunk group: a group is stamped or not stamped as a unit, so
# per-key findings would be N copies of one defect. The keys are listed
# inside the message, because "which of the twelve" is the first thing
# anyone fixing it needs.
gap_d = {}
for key in sorted(required_keys):
    if key in stamped_keys:
        continue
    if any(key.startswith(p) for p in stamped_keys if p.endswith((".", "_"))):
        continue                        # runtime-assembled index, S4
    group = key.split(".")[1]
    gap_d.setdefault(group, []).append(key)

for group in sorted(gap_d):
    keys = gap_d[group]
    if group not in ledger_no_stamp:
        first = required_keys[keys[0]][0]
        # A PARTIAL gap — some keys of the group stamped, some not — became
        # reportable when `meta_stamped` stopped accepting a bare `const`
        # declaration as a stamp. Saying "stamped by NO converter" about a
        # group that is six-sevenths stamped would send whoever reads this
        # looking for the wrong thing, so the sentence adapts.
        in_group = sum(1 for k in required_keys if k.split(".")[1] == group)
        scope = (
            f"stamped by NO converter: {len(keys)} key(s)"
            if len(keys) == in_group
            else f"only PARTLY stamped: {len(keys)} of its {in_group} required key(s)"
        )
        errors.append(
            f"[leg d] chunk group `vokra.{group}.*` is REQUIRED by a reader and "
            f"{scope} — {', '.join(keys[:6])}"
            f"{' …' if len(keys) > 6 else ''} — are looked up under vokra-models/src/ "
            f"(first at vokra-models/src/{first}) with no escape for their absence, and "
            f"no non-comment, non-test code under vokra-convert/src/ WRITES them. The "
            f"converter therefore writes a GGUF whose arch tag matches but whose config "
            f"chunk is missing those keys, so the binder refuses every artifact it "
            f"produces — the openwakeword (round 8) and llama_omni2 (round 9) defect, "
            f"which legs (a)-(c) cannot see because they only ever compare arch literals. "
            f"NOTE: DECLARING the key does not stamp it. A "
            f"`const KEY_…: &str = \"vokra.{group}.…\";` that no code ever passes to a "
            f"`b.add_*` call counts as unstamped here, which is the shape a dropped stamp "
            f"leaves behind and the reason this finding can name part of a group. Fix: "
            f"stamp the key(s) in the converter under crates/vokra-convert/src/models/ — "
            f"or, if the reader is meant to tolerate the group being absent, give it the "
            f"all-or-nothing escape the rest of the repo uses (`from_gguf` returning "
            f"`Result<Option<Self>>` with `Ok(None)` when no key of the group is present) "
            f"— or, if converter-side work is genuinely queued, add `{group}` to the "
            f"NO_STAMP ledger in scripts/check-arch-handshake.sh with the real reason."
        )

for group in sorted(ledger_no_stamp):
    if group not in gap_d:
        errors.append(
            f"[leg d] STALE ledger entry `{group}`: NO_STAMP says no converter stamps the "
            f"`vokra.{group}.*` chunks its binder requires, but that is no longer true — "
            f"either a converter now stamps them, or the reader stopped requiring them. "
            f"Fix: delete the `{group}` line from NO_STAMP in "
            f"scripts/check-arch-handshake.sh."
        )

# Leg (d)'s parser guards. The reader scan finding nothing is the silent
# direction — every group would read as fine — so it fails rather than
# passing vacuously, exactly like leg (c)'s pair.
if not required_keys:
    errors.append(
        f"no REQUIRED `vokra.<group>.<key>` read found anywhere under {models_dir} — the "
        f"const-binding scan, the suppression rules or the walk changed; leg (d) covered "
        f"nothing, so a pass here would be vacuous."
    )
if not stamped_keys:
    errors.append(
        f"zero `vokra.*` metadata literals scanned in non-comment, non-test source under "
        f"{conv_src} — the emitter scan is broken; every required read would read as "
        f"unstamped."
    )

if errors:
    print(f"check-arch-handshake: FAIL — {len(errors)} problem(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)

conv_archs = {a for a, _, _ in converters}
bind_archs = {a for a, _, _ in binders}
# Print the raw declaration counts, not just the distinct arch values: a
# discovery regex that stops matching shrinks these first, so they are the
# numbers to eyeball in a CI log.
bind_names = sorted({n for _, _, n in binders})
print(
    f"check-arch-handshake: discovery saw {len(converters)} converter + {len(binders)} binder "
    f"arch constant(s); binder spellings in use: {', '.join(bind_names)}."
)
n_sup = sum(len(v) for v in suppressed.values())
print(
    f"check-arch-handshake: reader/emitter scans exclude comments, `#[cfg(test)]` regions "
    f"and tests.rs files; {n_sup} literal site(s) additionally retracted by NOT_A_READER."
)
print(
    f"check-arch-handshake: OK — leg (a) {len(conv_archs)} converter arch(es), "
    f"{len(conv_archs) - len(gap_a)} answered by a reader, {len(gap_a)} declared in "
    f"NO_READER; leg (b) {len(bind_archs)} binder arch(es), "
    f"{len(bind_archs) - len(gap_b)} emitted by a converter, {len(gap_b)} declared in "
    f"NO_CONVERTER; leg (c) {len(cmd_slugs)} distinct `convert --model` string(s) -> "
    f"{checked_c} slug candidate(s) checked against {len(accepted_slugs)} accepted "
    f"spelling(s), {len(gap_c)} declared in NOT_A_MODEL_SLUG."
)
# Printed even when clean: the suppression tally is the number to eyeball if
# leg (d) ever goes suspiciously quiet. A rule that started over-matching
# would shrink `required_keys` and grow one of these instead of failing.
sup_summary = ", ".join(f"{k} ×{v}" for k, v in sorted(suppress_counts.items())) or "none"
print(
    f"check-arch-handshake: OK — leg (d) {len(required_keys)} required `vokra.*` metadata "
    f"key(s) read, {len(required_keys) - sum(len(v) for v in gap_d.values())} stamped by a "
    f"converter, {len(gap_d)} chunk group(s) declared in NO_STAMP; suppressed: "
    f"{sup_summary}."
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
        # Emits `mexpected`, the tag declared as `pub const EXPECTED_ARCH`
        # below, so the base tree is clean for it. Case 15 removes this line
        # to prove that spelling is genuinely discovered and checked.
        printf 'const EMITS_EXPECTED: &str = "mexpected";\n'
        # A comment naming `mlonely` must NOT count as an emitter.
        printf '// this comment mentions "mlonely" and must not satisfy leg (b)\n'
    } >"$tmp/conv/models/convs.rs"
    cp "$tmp/conv/models/convs.rs" "$tmp/convs_saved.rs"

    # Binder side: two arch constants + a reader literal for `alpha`.
    #   mbeta   -> emitted by the converter file above
    #   mlonely -> emitted by nobody (the leg (b) defect)
    printf 'pub const ARCH: &str = "mbeta";\nconst READS: &str = "alpha";\n' \
        >"$tmp/models/alpha/mod.rs"
    printf 'pub const ARCH_LONELY: &str = "mlonely";\n' >"$tmp/models/nested/mod.rs"

    # The `EXPECTED_ARCH` spelling — 29 real binders use it, and it was
    # invisible to this gate until 2026-08-15. Emitted by convs.rs above, so
    # it needs no ledger entry and perturbs none of the existing cases.
    mkdir -p "$tmp/models/expected"
    printf 'pub const EXPECTED_ARCH: &str = "mexpected";\n' >"$tmp/models/expected/mod.rs"

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

    # Leg (d) fixture. Kept in its own file so the leg (a)/(b)/(c) cases are
    # untouched by it. One REQUIRED read that IS stamped (`dkept`), so the
    # base tree is clean and every red case below is the planted change.
    write_meta() {
        {
            printf 'const KEY_KEPT: &str = "vokra.dfix.kept";\n'
            printf 'fn load(g: &GgufFile) -> Result<Self> {\n'
            printf '    let kept = g.get(KEY_KEPT).ok_or_else(|| e())?;\n'
            printf '    Ok(Self { kept })\n'
            printf '}\n'
            local extra
            for extra in "$@"; do
                printf '%s\n' "$extra"
            done
        } >"$tmp/models/nested/meta.rs"
    }
    write_meta

    write_stamps() {
        {
            printf 'fn stamp(b: &mut GgufBuilder) {\n'
            printf '    b.add_u32("vokra.dfix.kept", 1);\n'
            printf '}\n'
        } >"$tmp/conv/models/stamps.rs"
    }
    write_stamps

    # run <ledger-a...> -- <ledger-b...> [-- <ledger-c...> [-- <suppress...>
    #      [-- <ledger-d...>]]]
    run() {
        local -a la=() lb=() lc=() ls=() ld=()
        local seen_sep=0 arg
        for arg in "$@"; do
            if [ "$arg" = "--" ]; then
                seen_sep=$((seen_sep + 1))
                continue
            fi
            case "$seen_sep" in
                0) la+=("$arg") ;;
                1) lb+=("$arg") ;;
                2) lc+=("$arg") ;;
                3) ls+=("$arg") ;;
                *) ld+=("$arg") ;;
            esac
        done
        write_ledger "$tmp/ledger_a" ${la[@]+"${la[@]}"}
        write_ledger "$tmp/ledger_b" ${lb[@]+"${lb[@]}"}
        write_ledger "$tmp/ledger_c" ${lc[@]+"${lc[@]}"}
        write_ledger "$tmp/ledger_sup" ${ls[@]+"${ls[@]}"}
        write_ledger "$tmp/ledger_d" ${ld[@]+"${ld[@]}"}
        run_check "$tmp/conv/models" "$tmp/conv" "$tmp/models" "$tmp/engine.rs" \
            "$tmp/ledger_a" "$tmp/ledger_b" "$tmp/conv/lib.rs" "$tmp/ledger_c" \
            "$tmp/ledger_sup" "$tmp/ledger_d"
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
    write_ledger "$tmp/ledger_d"
    write_ledger "$tmp/ledger_sup"
    mkdir -p "$tmp/empty"
    if run_check "$tmp/empty" "$tmp/conv" "$tmp/models" "$tmp/engine.rs" \
        "$tmp/ledger_a" "$tmp/ledger_b" "$tmp/conv/lib.rs" "$tmp/ledger_c" \
        "$tmp/ledger_sup" "$tmp/ledger_d" >/dev/null 2>&1; then
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
        echo "self-test FAIL: a recovery command naming an unparsable slug should fail" >&2
        status=1
    elif grep -q 'leg c.*ghostslug' <<<"$out" && grep -q 'unknown model' <<<"$out"; then
        echo "self-test PASS: an unparsable recovery-command slug fails, naming it"
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
        echo "self-test FAIL: an unparsable from_arg should fail the guard" >&2
        status=1
    else
        echo "self-test PASS: a from_arg the parser cannot locate fails the guard"
    fi
    cp "$saved_lib" "$tmp/conv/lib.rs"

    # 15. The `EXPECTED_ARCH` spelling is genuinely DISCOVERED, not merely
    #     tolerated. Case 1 already passed with `mexpected` emitted; drop the
    #     emitter and leg (b) must now report it. Had the discovery regex
    #     stopped matching `EXPECTED_ARCH` — the state this gate shipped in
    #     until 2026-08-15 — this case would pass vacuously, because nothing
    #     would ever have looked at that binder at all.
    grep -v 'EMITS_EXPECTED' "$tmp/convs_saved.rs" >"$tmp/conv/models/convs.rs"
    if out="$(run "$ok" -- "$okb" 2>&1)"; then
        echo "self-test FAIL: an EXPECTED_ARCH binder with no emitter should fail" >&2
        status=1
    elif grep -q 'leg b.*`mexpected`' <<<"$out"; then
        echo "self-test PASS: a \`pub const EXPECTED_ARCH\` binder is discovered and checked"
    else
        echo "self-test FAIL: leg (b) failure did not name \`mexpected\`" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi
    cp "$tmp/convs_saved.rs" "$tmp/conv/models/convs.rs"

    # 16. The guard that would have CAUGHT the 2026-08-15 bug: an arch-shaped
    #     `&str` constant whose name the discovery regex does not match. The
    #     gate must fail loudly and name the spelling, instead of quietly
    #     scanning a smaller population and reporting it as clean.
    printf 'pub const LEGACY_ARCH: &str = "minvisible";\n' >"$tmp/models/nested/legacy.rs"
    if out="$(run "$ok" -- "$okb" 2>&1)"; then
        echo "self-test FAIL: an undiscoverable arch spelling should fail the guard" >&2
        status=1
    elif grep -q 'guard.*LEGACY_ARCH' <<<"$out"; then
        echo "self-test PASS: an arch-shaped constant outside the discovery regex fails the guard"
    else
        echo "self-test FAIL: the guard did not name \`LEGACY_ARCH\`" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi
    rm -f "$tmp/models/nested/legacy.rs"

    # 17. The same guard on the CONVERTER tree. The two roots are scanned
    #     separately, so covering only one would leave half the blind spot
    #     open — leg (a) would keep passing over converters it never saw.
    printf 'pub const MODEL_ARCH: &str = "cinvisible";\n' >"$tmp/conv/models/legacy.rs"
    if out="$(run "$ok" -- "$okb" 2>&1)"; then
        echo "self-test FAIL: an undiscoverable converter arch spelling should fail" >&2
        status=1
    elif grep -q 'guard.*MODEL_ARCH' <<<"$out"; then
        echo "self-test PASS: the discovery-coverage guard covers the converter tree too"
    else
        echo "self-test FAIL: the guard did not name \`MODEL_ARCH\`" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi
    rm -f "$tmp/conv/models/legacy.rs"

    # ---- test-code exclusion (2026-08-15) --------------------------------
    # The twin of case 16, for the OTHER way a scan can look at the wrong
    # thing. Cases 18-21 all plant the arch literal where a reader/emitter
    # cannot actually live and assert the gate STILL fails; run with an
    # EMPTY ledger so the only way to pass is for the planted literal to
    # have counted. Before this change every one of them passed — which is
    # how 25 converter arches, `yamnet` among them, were certified as
    # "answered by a reader" by test modules asserting the opposite.

    # 18. A literal inside `#[cfg(test)] mod tests { … }` under the binder
    #     tree must not answer leg (a).
    {
        printf '#[cfg(test)]\n'
        printf 'mod tests {\n'
        printf '    const PLANTED: &str = "orphan";\n'
        printf '}\n'
    } >"$tmp/models/nested/planted.rs"
    if out="$(run -- "$okb" 2>&1)"; then
        echo "self-test FAIL: a literal in a #[cfg(test)] block must not answer leg (a)" >&2
        status=1
    elif grep -q 'leg a.*`orphan`' <<<"$out"; then
        echo "self-test PASS: a #[cfg(test)] literal does not count as a reader"
    else
        echo "self-test FAIL: leg (a) did not report \`orphan\` past the planted test literal" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi
    rm -f "$tmp/models/nested/planted.rs"

    # 19. The same rule on the CONVERTER tree: a literal inside a
    #     `#[cfg(test)]` block must not emit for leg (b). Measured as a
    #     no-op on the real tree, applied anyway because the hole is
    #     symmetric.
    {
        printf '#[cfg(test)]\n'
        printf 'mod tests {\n'
        printf '    const PLANTED: &str = "mlonely";\n'
        printf '}\n'
    } >"$tmp/conv/models/planted.rs"
    if out="$(run "$ok" -- 2>&1)"; then
        echo "self-test FAIL: a literal in a #[cfg(test)] block must not emit for leg (b)" >&2
        status=1
    elif grep -q 'leg b.*`mlonely`' <<<"$out"; then
        echo "self-test PASS: a #[cfg(test)] literal does not count as an emitter"
    else
        echo "self-test FAIL: leg (b) did not report \`mlonely\` past the planted test literal" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi
    rm -f "$tmp/conv/models/planted.rs"

    # 20. A whole `tests.rs` file is excluded — the repo's file-per-module
    #     test layout, which carries no `#[cfg(test)]` of its own.
    printf 'const PLANTED: &str = "orphan";\n' >"$tmp/models/nested/tests.rs"
    if out="$(run -- "$okb" 2>&1)"; then
        echo "self-test FAIL: a literal in tests.rs must not answer leg (a)" >&2
        status=1
    elif grep -q 'leg a.*`orphan`' <<<"$out"; then
        echo "self-test PASS: a tests.rs literal does not count as a reader"
    else
        echo "self-test FAIL: leg (a) did not report \`orphan\` past the tests.rs literal" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi
    rm -f "$tmp/models/nested/tests.rs"

    # 21. A compound predicate — `#[cfg(all(test, …))]` — is still a test
    #     gate. Matching only the exact `#[cfg(test)]` spelling would let
    #     this one spelling re-open the whole hole silently.
    {
        printf '#[cfg(all(test, feature = "x"))]\n'
        printf 'mod tests {\n'
        printf '    const PLANTED: &str = "orphan";\n'
        printf '}\n'
    } >"$tmp/models/nested/planted.rs"
    if out="$(run -- "$okb" 2>&1)"; then
        echo "self-test FAIL: #[cfg(all(test, …))] must be treated as a test gate" >&2
        status=1
    elif grep -q 'leg a.*`orphan`' <<<"$out"; then
        echo "self-test PASS: a compound #[cfg(all(test, …))] region is excluded too"
    else
        echo "self-test FAIL: leg (a) did not report \`orphan\` past the compound-cfg literal" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi
    rm -f "$tmp/models/nested/planted.rs"

    # 22. …but `#[cfg(not(test))]` is PRODUCTION source and must keep
    #     counting. Over-skipping fails loudly rather than silently, so it
    #     is the safe direction — that is not a licence to skip code that
    #     only ever compiles OUTSIDE tests. `orphan` becomes answered here,
    #     so its ledger entry must now read as stale.
    {
        printf '#[cfg(not(test))]\n'
        printf 'mod prod {\n'
        printf '    const REAL_READER: &str = "orphan";\n'
        printf '}\n'
    } >"$tmp/models/nested/planted.rs"
    if out="$(run "$ok" -- "$okb" 2>&1)"; then
        echo "self-test FAIL: #[cfg(not(test))] source must still count as a reader" >&2
        status=1
    elif grep -q 'STALE.*`orphan`' <<<"$out"; then
        echo "self-test PASS: #[cfg(not(test))] is production source and still answers"
    else
        echo "self-test FAIL: a #[cfg(not(test))] reader was wrongly skipped" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi
    rm -f "$tmp/models/nested/planted.rs"

    # 23. …and neither is `#[cfg(feature = "test-utils")]`, whose predicate
    #     only mentions `test` inside a string. Quoted spans are dropped
    #     before the token test for exactly this reason.
    {
        printf '#[cfg(feature = "test-utils")]\n'
        printf 'mod helpers {\n'
        printf '    const REAL_READER: &str = "orphan";\n'
        printf '}\n'
    } >"$tmp/models/nested/planted.rs"
    if out="$(run "$ok" -- "$okb" 2>&1)"; then
        echo "self-test FAIL: a feature = \"test-utils\" gate must not read as a test gate" >&2
        status=1
    elif grep -q 'STALE.*`orphan`' <<<"$out"; then
        echo "self-test PASS: \`feature = \"test-utils\"\` is not mistaken for a test gate"
    else
        echo "self-test FAIL: a feature-gated reader was wrongly skipped as test code" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi
    rm -f "$tmp/models/nested/planted.rs"

    # 24. NOT_A_READER retracts one FILE's occurrences. The planted literal
    #     is ordinary production source — the `ast` / canary shape, which no
    #     amount of comment- or test-stripping can reach — so without the
    #     entry `orphan` reads as answered and its NO_READER line goes
    #     stale. With it, the tree is clean again.
    printf 'const COLLIDES: &str = "orphan";\n' >"$tmp/models/nested/collide.rs"
    if out="$(run "$ok" -- "$okb" 2>&1)"; then
        echo "self-test FAIL: an unsuppressed colliding literal should answer and go stale" >&2
        status=1
    elif grep -q 'STALE.*`orphan`' <<<"$out"; then
        echo "self-test PASS: a colliding literal answers leg (a) until it is retracted"
    else
        echo "self-test FAIL: the colliding literal did not make \`orphan\` stale" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi
    if run "$ok" -- "$okb" -- -- 'orphan|nested/collide.rs|an unrelated string' \
        >/dev/null 2>&1; then
        echo "self-test PASS: NOT_A_READER retracts that file and the gap returns"
    else
        echo "self-test FAIL: a NOT_A_READER entry should retract the colliding site" >&2
        status=1
    fi

    # 25. A NOT_A_READER entry that retracts nothing is an unchecked claim,
    #     so it fails as stale. Same double-sidedness as every other ledger
    #     here: this is what stops the suppression list becoming a place to
    #     quietly park inconvenient findings.
    rm -f "$tmp/models/nested/collide.rs"
    if out="$(run -- "$okb" -- -- 'orphan|nested/collide.rs|an unrelated string' 2>&1)"; then
        echo "self-test FAIL: a NOT_A_READER entry naming a vanished file should fail" >&2
        status=1
    elif grep -q 'STALE NOT_A_READER.*`orphan`' <<<"$out"; then
        echo "self-test PASS: a NOT_A_READER entry that suppresses nothing fails as stale"
    else
        echo "self-test FAIL: the stale NOT_A_READER entry was not reported" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi

    # ---- leg (d): metadata key -> converter stamp (2026-08-15) -----------
    # Cases 26-38. The base fixture has ONE required read (`vokra.dfix.kept`)
    # that IS stamped, so every red case below is the planted change and
    # every green case is a suppression rule actually firing. Suppression
    # cases are asserted GREEN with an EMPTY leg (d) ledger, so the only way
    # to pass is for the rule to have classified the planted read as
    # tolerated — a rule that stopped working would fail them, not silently
    # widen the population.

    # 26. The planted TRUE POSITIVE: a required read (`ok_or_else` on the
    #     absent branch) whose group no converter stamps. This is the
    #     openwakeword / llama_omni2 shape, and the whole reason for the leg.
    write_meta \
        'const KEY_GAP: &str = "vokra.dgap.missing";' \
        'fn load2(g: &GgufFile) -> Result<u32> {' \
        '    g.get(KEY_GAP).ok_or_else(|| e())?' \
        '}'
    if out="$(run "$ok" -- "$okb" 2>&1)"; then
        echo "self-test FAIL: a required metadata read no converter stamps should fail" >&2
        status=1
    elif grep -q 'leg d.*`vokra.dgap.\*`' <<<"$out" \
        && grep -q 'vokra.dgap.missing' <<<"$out"; then
        echo "self-test PASS: an unstamped required metadata group fails, naming group and key"
    else
        echo "self-test FAIL: leg (d) failure did not name \`vokra.dgap.*\`" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi

    # 27. The same gap, declared in NO_STAMP -> passes. Proves the ledger is
    #     honoured and (with 26) that it is what changed.
    if run "$ok" -- "$okb" -- -- -- 'dgap|converter sub-wave queued' >/dev/null 2>&1; then
        echo "self-test PASS: a declared NO_STAMP group passes leg (d)"
    else
        echo "self-test FAIL: a declared NO_STAMP entry should pass" >&2
        status=1
    fi

    # 28. Stale NO_STAMP entry -> fails. `dfix` is stamped, so claiming it is
    #     not is a lie the gate must catch — same double-sidedness as every
    #     other ledger here.
    write_meta
    if out="$(run "$ok" -- "$okb" -- -- -- 'dfix|stale claim' 2>&1)"; then
        echo "self-test FAIL: a stale NO_STAMP entry should fail" >&2
        status=1
    elif grep -q 'STALE ledger entry `dfix`' <<<"$out"; then
        echo "self-test PASS: a NO_STAMP entry whose converter now stamps it fails as stale"
    else
        echo "self-test FAIL: stale leg (d) failure did not name \`dfix\`" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi

    # 29. S1, spelled on the return type: a group whose reader answers
    #     `Result<Option<Self>>` is the repo's all-or-nothing convention
    #     (nisqa, ten_vad, smart_turn, firered_*). A converter that stamps
    #     none of it is what that reader is FOR.
    write_meta \
        'const KEY_OPT: &str = "vokra.dopt.axis";' \
        'fn load3(g: &GgufFile) -> Result<Option<Self>> {' \
        '    let v = g.get(KEY_OPT).ok_or_else(|| e())?;' \
        '    Ok(Some(v))' \
        '}'
    if run "$ok" -- "$okb" >/dev/null 2>&1; then
        echo "self-test PASS: S1 — a \`Result<Option<..>>\` reader tolerates an unstamped group"
    else
        echo "self-test FAIL: an all-or-nothing optional group must not be a leg (d) finding" >&2
        run "$ok" -- "$okb" 2>&1 >&2 || true
        status=1
    fi

    # 30. S1, spelled as an early escape: `-> Result<Self>` but the body
    #     returns `Ok(None)` when no key of the group is present. gigaam's
    #     `let Some(..) = .. else { return Ok(None) }` shape.
    write_meta \
        'const KEY_ESC: &str = "vokra.desc.axis";' \
        'fn load4(g: &GgufFile) -> Result<Self> {' \
        '    if g.is_empty() { return Ok(None); }' \
        '    let v = g.get(KEY_ESC).ok_or_else(|| e())?;' \
        '    Ok(v)' \
        '}'
    if run "$ok" -- "$okb" >/dev/null 2>&1; then
        echo "self-test PASS: S1 — an \`Ok(None)\` whole-group escape is tolerated"
    else
        echo "self-test FAIL: an \`Ok(None)\` escape must not be a leg (d) finding" >&2
        status=1
    fi

    # 31. S2, caller default: `unwrap_or` over a primary-source constant.
    #     panns / musicgen, which document the fallback in their own source.
    write_meta \
        'const KEY_DEF: &str = "vokra.ddef.axis";' \
        'fn load5(g: &GgufFile) -> Result<u32> {' \
        '    Ok(g.get(KEY_DEF).and_then(|v| v.as_u64()).unwrap_or(DEFAULT_AXIS)?)' \
        '}'
    if run "$ok" -- "$okb" >/dev/null 2>&1; then
        echo "self-test PASS: S2 — an \`unwrap_or\` caller default is tolerated"
    else
        echo "self-test FAIL: an unwrap_or fallback must not be a leg (d) finding" >&2
        status=1
    fi

    # 32. …but a ZERO SENTINEL is NOT a default, and this is the case the
    #     whole rule turns on. `read_u32_or_zero`'s `None => Ok(0)` hands
    #     back a value no forward pass can use: llama_omni2 (round 9)
    #     decayed every unstamped axis to `0` and deferred the failure to
    #     `validate_for_forward`. If this case ever goes green, the leg has
    #     stopped catching the defect it was written for.
    write_meta \
        'const KEY_SENT: &str = "vokra.dsent.axis";' \
        'fn read_u32_or_zero(g: &GgufFile, key: &str) -> Result<u32> {' \
        '    match g.get(key) {' \
        '        Some(V::U32(v)) => Ok(*v),' \
        '        None => Ok(0),' \
        '        Some(o) => Err(e(o)),' \
        '    }' \
        '}' \
        'fn load6(g: &GgufFile) -> Result<Self> {' \
        '    let axis = read_u32_or_zero(g, KEY_SENT)? as usize;' \
        '    Ok(Self { axis })' \
        '}'
    if out="$(run "$ok" -- "$okb" 2>&1)"; then
        echo "self-test FAIL: a zero-sentinel read must NOT be treated as a caller default" >&2
        status=1
    elif grep -q 'leg d.*`vokra.dsent.\*`' <<<"$out"; then
        echo "self-test PASS: S2 — \`None => Ok(0)\` is a deferred failure, not a default"
    else
        echo "self-test FAIL: leg (d) did not report the zero-sentinel group" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi

    # 32b. The SAME zero sentinel spelled `.unwrap_or(0)` instead of
    #      `None => Ok(0)`. Until 2026-08-16 the `FALLBACK_CALL` branch
    #      returned True without inspecting its argument, so this spelling
    #      filed itself as an S2 caller default and vanished into the
    #      suppression count — in the one leg that exists to stop a third
    #      recurrence of the round-8 / round-9 class. Proven at the time by
    #      deleting a real `vokra.voxtral.audio_encoder.n_ctx` stamp, which
    #      the gate missed before the fix and names after it.
    write_meta \
        'const KEY_UWZ: &str = "vokra.duwz.axis";' \
        'fn load6b(g: &GgufFile) -> Result<Self> {' \
        '    let axis = g.get(KEY_UWZ).and_then(|v| v.as_u64()).unwrap_or(0) as usize;' \
        '    Ok(Self { axis })' \
        '}'
    if out="$(run "$ok" -- "$okb" 2>&1)"; then
        echo "self-test FAIL: \`.unwrap_or(0)\` must NOT be treated as a caller default" >&2
        status=1
    elif grep -q 'leg d.*`vokra.duwz.\*`' <<<"$out"; then
        echo "self-test PASS: S2 — \`.unwrap_or(0)\` is the same deferred failure"
    else
        echo "self-test FAIL: leg (d) did not report the unwrap_or-sentinel group" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi

    # 32c. Negative control for 32b: `.unwrap_or_default()` stays tolerated.
    #      A `Default` impl is a DECLARED value, unlike a bare `0` written at
    #      the call site, so narrowing the rule must not sweep it up. Without
    #      this case, an over-strict fix would look correct.
    write_meta \
        'const KEY_UWD: &str = "vokra.duwd.axis";' \
        'fn load6c(g: &GgufFile) -> Result<Self> {' \
        '    let axis = g.get(KEY_UWD).and_then(|v| v.as_u64()).unwrap_or_default() as usize;' \
        '    Ok(Self { axis })' \
        '}'
    if run "$ok" -- "$okb" >/dev/null 2>&1; then
        echo "self-test PASS: S2 — \`unwrap_or_default()\` is a declared value"
    else
        echo "self-test FAIL: unwrap_or_default must remain a tolerated default" >&2
        status=1
    fi

    # 33. S2, option-returning helper: the escape lives at the READ SITE, not
    #     on the enclosing fn, so the S1 check cannot see it. canary_1b_flash
    #     reads 23 axes this way and leaves each at its primary-source
    #     default — it was a false positive until this rule was added.
    write_meta \
        'const KEY_OPTF: &str = "vokra.doptf.axis";' \
        'fn opt_u32(g: &GgufFile, key: &str) -> Result<Option<u32>> {' \
        '    Ok(g.get(key).and_then(|v| v.as_u64()).map(|v| v as u32))' \
        '}' \
        'fn load7(g: &GgufFile) -> Result<Self> {' \
        '    let mut cfg = Self::default();' \
        '    if let Some(v) = opt_u32(g, KEY_OPTF)? { cfg.axis = v; }' \
        '    Ok(cfg)' \
        '}'
    if run "$ok" -- "$okb" >/dev/null 2>&1; then
        echo "self-test PASS: S2 — an \`Option\`-returning per-key helper is tolerated"
    else
        echo "self-test FAIL: an opt_* read-site escape must not be a leg (d) finding" >&2
        run "$ok" -- "$okb" 2>&1 >&2 || true
        status=1
    fi

    # 34. S3, mention not a read: the key is named only inside a diagnostic.
    #     `vokra.snac.codebook_tables` describes an artifact that does not
    #     exist yet; gigaam interpolates its key into a wrapped error whose
    #     continuation lines look like bare code to a line-at-a-time scan.
    #     The multi-line string here is the load-bearing half of the case.
    write_meta \
        'const KEY_MENTION: &str = "vokra.dment.future";' \
        'fn explain() -> String {' \
        '    format!(' \
        '        "this binder will one day need `{KEY_MENTION}` \' \
        '         but no converter writes it yet"' \
        '    )' \
        '}'
    if run "$ok" -- "$okb" >/dev/null 2>&1; then
        echo "self-test PASS: S3 — a key named only inside a diagnostic is not a read"
    else
        echo "self-test FAIL: an error-text mention must not be a leg (d) finding" >&2
        run "$ok" -- "$okb" 2>&1 >&2 || true
        status=1
    fi

    # 35. S4, runtime-assembled prefix: `vokra.atst.patch_grid` is stamped as
    #     `_0` / `_1` and `vokra.moshi.delay.` as `.0` / `.1`, so the
    #     assembled key exists as a literal on NEITHER side and comparing
    #     literals is a category error. Recognised by the `PREFIX` segment in
    #     the constant name — note `\bPREFIX\b` matches neither real
    #     spelling, because `_` is a word character on both sides.
    write_meta \
        'const GGUF_KEY_AXIS_PREFIX: &str = "vokra.dpfx.axis";' \
        'fn load8(g: &GgufFile, i: usize) -> Result<u32> {' \
        '    g.get(&format!("{GGUF_KEY_AXIS_PREFIX}_{i}")).ok_or_else(|| e())?' \
        '}'
    if run "$ok" -- "$okb" >/dev/null 2>&1; then
        echo "self-test PASS: S4 — a runtime-assembled PREFIX constant is not compared literally"
    else
        echo "self-test FAIL: an indexed-prefix constant must not be a leg (d) finding" >&2
        run "$ok" -- "$okb" 2>&1 >&2 || true
        status=1
    fi

    # 36. S5, dead-code reserved constant: declared for a wiring wave that has
    #     not landed, read by nothing. `vokra.kokoro.phase_activation`,
    #     "consumed by the T18 load/forward wiring".
    write_meta \
        '#[allow(dead_code)]' \
        'const KEY_RESERVED: &str = "vokra.dres.future";'
    if run "$ok" -- "$okb" >/dev/null 2>&1; then
        echo "self-test PASS: S5 — an \`#[allow(dead_code)]\` reserved constant is not a read"
    else
        echo "self-test FAIL: a dead-code reserved constant must not be a leg (d) finding" >&2
        run "$ok" -- "$okb" 2>&1 >&2 || true
        status=1
    fi

    # 37. A required read inside `#[cfg(test)]` must not count either — the
    #     same exclusion legs (a)/(b) apply, checked on leg (d) because it
    #     discovers constants independently.
    write_meta \
        '#[cfg(test)]' \
        'mod tests {' \
        '    const KEY_T: &str = "vokra.dtest.axis";' \
        '    fn t(g: &GgufFile) -> Result<u32> { g.get(KEY_T).ok_or_else(|| e())? }' \
        '}'
    if run "$ok" -- "$okb" >/dev/null 2>&1; then
        echo "self-test PASS: a required read inside #[cfg(test)] is not a leg (d) finding"
    else
        echo "self-test FAIL: a #[cfg(test)] metadata read must not count" >&2
        status=1
    fi
    write_meta

    # 38. The leg's own parser guard: a models tree with no required read at
    #     all. Finding nothing is the SILENT direction — every group would
    #     read as fine — so it must fail rather than pass vacuously, exactly
    #     like leg (c)'s pair.
    local saved_meta="$tmp/meta_saved.rs"
    cp "$tmp/models/nested/meta.rs" "$saved_meta"
    : >"$tmp/models/nested/meta.rs"
    if out="$(run "$ok" -- "$okb" 2>&1)"; then
        echo "self-test FAIL: zero required metadata reads should fail the guard" >&2
        status=1
    elif grep -q 'no REQUIRED `vokra' <<<"$out"; then
        echo "self-test PASS: a leg (d) scan that found no required read fails the guard"
    else
        echo "self-test FAIL: the leg (d) empty-scan guard did not fire" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi
    cp "$saved_meta" "$tmp/models/nested/meta.rs"

    # 39. The EMITTER half, checked from the converter side. Every case above
    #     varies the reader; this one deletes the converter's stamp and
    #     asserts the same key becomes a finding. Without it the whole leg
    #     could be passing because the emitter scan matches everything
    #     rather than because the stamp is really there — and that is the
    #     precise shape of the bug this gate exists to prevent.
    printf 'fn stamp(b: &mut GgufBuilder) {\n}\n' >"$tmp/conv/models/stamps.rs"
    if out="$(run "$ok" -- "$okb" 2>&1)"; then
        echo "self-test FAIL: deleting the converter stamp should make the key a finding" >&2
        status=1
    elif grep -q 'leg d.*`vokra.dfix.\*`' <<<"$out"; then
        echo "self-test PASS: removing the converter's stamp turns a clean key into a finding"
    else
        echo "self-test FAIL: leg (d) did not report \`vokra.dfix.*\` once unstamped" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi
    write_stamps

    # 40. …and a stamp that lives only in a converter COMMENT does not
    #     count. `//! …deserialises `vokra.melodyflow.*`…` is exactly how
    #     both real NO_STAMP entries describe work they have not done, so a
    #     comment-blind scan would have marked them stamped and hidden both.
    printf 'fn stamp(b: &mut GgufBuilder) {\n    // b.add_u32("vokra.dfix.kept", 1);\n}\n' \
        >"$tmp/conv/models/stamps.rs"
    if out="$(run "$ok" -- "$okb" 2>&1)"; then
        echo "self-test FAIL: a commented-out stamp must not satisfy leg (d)" >&2
        status=1
    elif grep -q 'leg d.*`vokra.dfix.\*`' <<<"$out"; then
        echo "self-test PASS: a stamp that exists only in a comment does not count"
    else
        echo "self-test FAIL: leg (d) accepted a commented-out stamp" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi
    write_stamps

    # 41. THE TEETH CASE. A converter that DECLARES the key constant and
    #     never passes it to a writer. This is the shape a dropped stamp
    #     leaves behind, and for one day it was invisible: deleting
    #     `b.add_u32(KEY_N_WAKEWORDS, …)` from openwakeword_op.rs while its
    #     `pub const KEY_N_WAKEWORDS: &str = "vokra.openwakeword.n_wakewords";`
    #     stayed put reproduced the round-8 defect exactly, and leg (d)
    #     printed OK with every number unchanged — because `meta_stamped`
    #     harvested literals and a declaration IS a literal. The unrelated
    #     `dother` stamp keeps the emitter scan non-empty, so the ONLY thing
    #     that can fail this case is the declaration being mistaken for a
    #     stamp.
    {
        printf 'const KEY_KEPT: &str = "vokra.dfix.kept";\n'
        printf 'fn stamp(b: &mut GgufBuilder) {\n'
        printf '    b.add_u32("vokra.dother.axis", 1);\n'
        printf '}\n'
    } >"$tmp/conv/models/stamps.rs"
    if out="$(run "$ok" -- "$okb" 2>&1)"; then
        echo "self-test FAIL: a declared-but-never-stamped required key must fail" >&2
        status=1
    elif grep -q 'leg d.*`vokra.dfix.\*`' <<<"$out"; then
        echo "self-test PASS: declaring a key constant is not stamping it"
    else
        echo "self-test FAIL: leg (d) did not report \`vokra.dfix.*\` as unstamped" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi
    write_stamps

    # 42. The OTHER side of the same boundary, and the reason case 41's fix
    #     cannot be "treat every unstamped key as a defect": a read with a
    #     REAL caller default is not broken by a missing stamp. S2 covers 383
    #     sites on the live tree, so a fix that failed this would bury the
    #     signal it just gained. The converter here declares the constant and
    #     never stamps it, exactly as in case 41 — the ONLY difference is
    #     `unwrap_or(16)` on the read, which is precisely the distinction
    #     that should decide the verdict.
    write_meta \
        'const KEY_DEF: &str = "vokra.ddef.axis";' \
        'fn load_def(g: &GgufFile) -> Result<Self> {' \
        '    let axis = g.get(KEY_DEF).and_then(|v| v.as_u64()).unwrap_or(16);' \
        '    Ok(Self { axis })' \
        '}'
    {
        printf 'const KEY_DEF: &str = "vokra.ddef.axis";\n'
        printf 'fn stamp(b: &mut GgufBuilder) {\n'
        printf '    b.add_u32("vokra.dfix.kept", 1);\n'
        printf '}\n'
    } >"$tmp/conv/models/stamps.rs"
    if run "$ok" -- "$okb" >/dev/null 2>&1; then
        echo "self-test PASS: a DEFAULTED read with no stamp is not a leg (d) finding"
    else
        echo "self-test FAIL: an unwrap_or read must survive its key going unstamped" >&2
        run "$ok" -- "$okb" 2>&1 >&2 || true
        status=1
    fi
    write_meta
    write_stamps

    # 43. Head interpolation IS a stamp. Tightening case 41 by looking for
    #     the constant in CODE position alone would break the prefix
    #     builders, because `format!("{KEY_WORDPIECE_PREFIX}.kind")` hides
    #     the name inside a string and strings are blanked out of `code`.
    #     Found by measurement, not by guesswork: it was the single false
    #     positive the first cut of the fix produced across the whole tree
    #     (`vokra.bert.wordpiece`, stamped by bert_base.rs:616-641 and read
    #     as an inline prefix literal by sbv2/mod.rs:2708).
    {
        printf 'const KEY_PFX: &str = "vokra.dfix";\n'
        printf 'fn stamp(b: &mut GgufBuilder) {\n'
        printf '    b.add_u32(&format!("{KEY_PFX}.kept"), 1);\n'
        printf '}\n'
    } >"$tmp/conv/models/stamps.rs"
    if run "$ok" -- "$okb" >/dev/null 2>&1; then
        echo "self-test PASS: a key assembled from a head-interpolated constant counts as stamped"
    else
        echo "self-test FAIL: format!(\"{KEY_PFX}.kept\") must count as a stamp" >&2
        run "$ok" -- "$okb" 2>&1 >&2 || true
        status=1
    fi
    write_stamps

    # 44. A use that exists only inside `#[cfg(test)]` is not a stamp — the
    #     converter half of case 37. Real shape: openwakeword_op.rs names all
    #     seven keys in `all_seven_runtime_metadata_keys_are_stamped`, so a
    #     scan that forgot the test exclusion here would call a group stamped
    #     on the strength of the test asserting that it is.
    {
        printf 'const KEY_KEPT: &str = "vokra.dfix.kept";\n'
        printf 'fn stamp(b: &mut GgufBuilder) {\n'
        printf '    b.add_u32("vokra.dother.axis", 1);\n'
        printf '}\n'
        printf '#[cfg(test)]\n'
        printf 'mod tests {\n'
        printf '    fn t(b: &mut GgufBuilder) { b.add_u32(KEY_KEPT, 1); }\n'
        printf '}\n'
    } >"$tmp/conv/models/stamps.rs"
    if out="$(run "$ok" -- "$okb" 2>&1)"; then
        echo "self-test FAIL: a stamp that only exists in #[cfg(test)] must not count" >&2
        status=1
    elif grep -q 'leg d.*`vokra.dfix.\*`' <<<"$out"; then
        echo "self-test PASS: a test-only stamp does not satisfy leg (d)"
    else
        echo "self-test FAIL: leg (d) accepted a #[cfg(test)]-only stamp" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi
    write_stamps

    # 45. The shared provenance helper lives in vokra-core, outside the
    #     converter subtree scanned by this gate. A visible Some(model_id)
    #     argument is nevertheless a real write performed by that helper.
    write_meta \
        'const KEY_PROV_ID: &str = "vokra.provenance.model_id";' \
        'fn load_prov(g: &GgufFile) -> Result<Self> {' \
        '    let id = g.get(KEY_PROV_ID).ok_or_else(|| e())?;' \
        '    Ok(Self { id })' \
        '}'
    {
        printf 'fn stamp(b: &mut GgufBuilder) {\n'
        printf '    b.add_u32("vokra.dfix.kept", 1);\n'
        printf '    vokra_core::stamp_provenance(\n'
        printf '        b, LicenseClass::Permissive, "apache-2.0", Some("model"), None,\n'
        printf '    );\n'
        printf '}\n'
    } >"$tmp/conv/models/stamps.rs"
    if run "$ok" -- "$okb" >/dev/null 2>&1; then
        echo "self-test PASS: stamp_provenance with Some(model_id) counts as a real model_id stamp"
    else
        echo "self-test FAIL: stamp_provenance(Some(model_id), ..) must satisfy the provenance reader" >&2
        run "$ok" -- "$okb" 2>&1 >&2 || true
        status=1
    fi

    # 46. The same helper with None for source must not answer a required
    #     source reader. This keeps helper recognition fail-closed instead
    #     of treating every optional field as unconditionally stamped.
    write_meta \
        'const KEY_PROV_SOURCE: &str = "vokra.provenance.source";' \
        'fn load_source(g: &GgufFile) -> Result<Self> {' \
        '    let source = g.get(KEY_PROV_SOURCE).ok_or_else(|| e())?;' \
        '    Ok(Self { source })' \
        '}'
    if out="$(run "$ok" -- "$okb" 2>&1)"; then
        echo "self-test FAIL: stamp_provenance(..., None) must not count as a source stamp" >&2
        status=1
    elif grep -q 'leg d.*`vokra.provenance.\*`' <<<"$out" \
        && grep -q 'vokra.provenance.source' <<<"$out"; then
        echo "self-test PASS: stamp_provenance None does not fabricate an optional provenance stamp"
    else
        echo "self-test FAIL: the absent provenance source was not reported" >&2
        printf '%s\n' "$out" >&2
        status=1
    fi
    write_meta
    write_stamps

    if [ "$status" -eq 0 ]; then
        echo "check-arch-handshake --self-test: OK (49 cases)"
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
        LEDGER_SUP="$(mktemp)"
        LEDGER_D="$(mktemp)"
        trap 'rm -f "$LEDGER_A" "$LEDGER_B" "$LEDGER_C" "$LEDGER_SUP" "$LEDGER_D"' EXIT
        write_ledger "$LEDGER_A" ${NO_READER[@]+"${NO_READER[@]}"}
        write_ledger "$LEDGER_B" ${NO_CONVERTER[@]+"${NO_CONVERTER[@]}"}
        write_ledger "$LEDGER_C" ${NOT_A_MODEL_SLUG[@]+"${NOT_A_MODEL_SLUG[@]}"}
        write_ledger "$LEDGER_SUP" ${NOT_A_READER[@]+"${NOT_A_READER[@]}"}
        write_ledger "$LEDGER_D" ${NO_STAMP[@]+"${NO_STAMP[@]}"}
        run_check "$CONVERT_MODELS_DEFAULT" "$CONVERT_SRC_DEFAULT" "$MODELS_DEFAULT" \
            "$ENGINE_DEFAULT" "$LEDGER_A" "$LEDGER_B" "$CONVERT_LIB_DEFAULT" "$LEDGER_C" \
            "$LEDGER_SUP" "$LEDGER_D"
        ;;
    *)
        echo "error: unknown argument '$1'" >&2
        usage >&2
        exit 1
        ;;
esac
