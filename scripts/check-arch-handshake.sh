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
#   a third of the binders. `charsiu`, which has no converter at all, sat
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
  'ace_step|publish-only: MIT music generation (ACE-Step), a ~9.6 GB multi-file bundle needing an offline merge plus a vast.ai handoff per [[feedback-large-models-on-vast-ai]]. ace_step.rs records the prepare script as future work; no vokra-models::ace_step module exists.'
  'ast|publish-only BF16 pass-through (Audio Spectrogram Transformer, bsd-3-clause). ast.rs:10-11 reserves the tag for "a future vokra-models::ast::* loader"; crates/vokra-models/src/ast/ does not exist. Also listed in NOT_A_READER: the sole non-test "ast" literal under vokra-models/src is Canary1bFlashTask::Ast.as_str(), an unrelated task label, so without that suppression this gap would read as answered.'
  'basic-pitch|publish-only BF16 pass-through (Spotify basic-pitch, apache-2.0; <2 GB so local convert is safe). basic_pitch.rs:47-48 defers the offline TF-SavedModel flattening script; no vokra-models::basic_pitch module exists.'
  'beats|publish-only BF16 pass-through (microsoft/unilm BEATs, MIT via the repo-root LICENSE). beats.rs:60 defers the safetensors prepare script. The tag is kept distinct precisely so a BEATs checkpoint cannot misroute into a HuBERT loader; no binder module exists.'
  'bs_roformer|publish-BLOCKED, not merely unbound: the BS-RoFormer family carries no uniform upstream license (most releases carry none at all), so the converter defaults to LicenseClass::RedistributionForbidden fail-closed and a caller must supply --license <spdx> from their own attestation. bs_roformer.rs:19 names crates/vokra-models/src/bs_roformer/ as the binder that "will read" this metadata; that directory does not exist yet.'
  'dasheng|publish-only BF16 pass-through (Xiaomi mispeech Dasheng, apache-2.0, <2 GB). dasheng.rs:45 defers the prepare script; no vokra-models::dasheng module exists.'
  'ecapa_tdnn|not wired on EITHER side yet: ecapa_tdnn.rs "Wiring status" records that convert_ecapa_tdnn_file is `pub` but is neither re-exported at the crate root nor dispatched from convert_file_licensed, so the converter half is itself a landing pad awaiting its ModelKind arm. Real-weight parity and a runtime binder are deferred to owner sign-off (license-audit.md section 3.1).'
  'firered_asr_llm_l|awaiting a binder: ~16.6 GB / 8.3B BF16 (Conformer encoder + Qwen2 LM decoder). firered_asr_llm_l.rs:100 records that it has no runtime binder of its own and that the only landed Qwen2-family forward belongs to a different model. Its siblings firered_asr_aed_l and firered_vad each have a binder; the tag stays distinct so an AED loader cannot try to read an LLM decoder.'
  'frcrn|publish-only BF16 pass-through (alibabasglab FRCRN, apache-2.0). frcrn.rs:10 reserves the tag for a future vokra-models::frcrn::* implementation. The tag is deliberately NOT "denoise", so it cannot misroute into the DeepFilterNet3 binder.'
  'htdemucs_multi|publish-only, variant-agnostic BF16 pass-through skeleton (facebookresearch/demucs, MIT). htdemucs_multi.rs:50 names a future vokra-models::htdemucs_multi module; real-weight parity is deferred to owner sign-off (license-audit.md section 3.1).'
  'hubert|publish-only BF16 pass-through (facebook/hubert-large-ls960-ft, apache-2.0; local convert safe). The future native forward is expected to share ops with wav2vec2_ctc, but hubert_large_ls960.rs:17 keeps the tag distinct so a HuBERT checkpoint cannot misroute into the wav2vec2 loader. No binder module exists.'
  'mert|publish-only BF16 pass-through AND NonCommercial: m-a-p/MERT-v1-330M is cc-by-nc-4.0, so publish needs --allow-noncommercial and the M2-13 runtime gate refuses a commercial-mode load. mert.rs:53 defers the prepare script; no binder module exists.'
  'metricgan_plus|publish-only BF16 pass-through (SpeechBrain MetricGAN+, apache-2.0). metricgan_plus.rs:43 names a future crates/vokra-models/src/metricgan_plus/ module and :32 defers the from_gguf forward to owner sign-off.'
  'mossformer2_ss_16k|publish-only BF16 pass-through (ClearerVoice-Studio MossFormer2, apache-2.0). mossformer2_ss_16k.rs:52-56 defers real-weight parity against a future gated-attention forward to owner sign-off and keeps the tag distinct from the landed FsmnVad binder.'
  'mp_senet|publish-only BF16 pass-through (MP-SENet, MIT). mp_senet.rs:42 names a future crates/vokra-models/src/mp_senet/ module; :31 defers the forward path to owner sign-off (license-audit.md section 3.1).'
  'muq|publish-BLOCKED by license posture: OpenMuQ/MuQ-large-msd-iter declared no license as of 2026-08-13, so the converter maps to LicenseClass::Unknown fail-closed and section 3.1 stays blank. Publish-only BF16 pass-through besides, with the prepare script deferred at muq.rs:50; no binder module exists.'
  'openwakeword|publish-only BF16 pass-through for the raw wake-word weights (dscripka/openWakeWord, apache-2.0). openwakeword.rs:32-34 states the runtime port is deferred — the audio-dialect kws op consumes the artifact in a future WP. NOT to be confused with the SEPARATE and fully bound openwakeword_op pair: vokra-convert/src/models/openwakeword_op.rs:108 stamps "openwakeword_op" and vokra-models/src/kws/openwakeword/mod.rs:84 verifies it. Same family, two tags, only this one is unbound.'
  'pyannote-speaker-diarization|converter-only BY DESIGN: this GGUF is a WEIGHTLESS pipeline orchestrator (clustering thresholds plus sub-model references, no sincnet.* / lstm.* tensors at all). vokra-models/src/pyannote/mod.rs:130 documents the refusal verbatim — EXPECTED_ARCH is deliberately "pyannote-segmentation" — and verify_arch names this tag in its rejection text so an operator who hands the pipeline to the backbone binder is told "you handed me a pipeline, not a backbone" instead of hitting a confusing empty-manifest error. A pipeline-level loader is a follow-up.'
  'reazonspeech_nemo_v2|publish-only BF16 pass-through (ReazonSpeech NeMo v2, apache-2.0). reazonspeech_nemo_v2.rs:53-54 names a future vokra-models::reazonspeech_nemo_v2 module (Longformer local-attention encoder + RNN-T / CTC head) and defers the forward to owner sign-off.'
  'rnnoise|publish-only BF16 pass-through (RNNoise, permissive). rnnoise.rs:8-9 reserves the tag for a future vokra-models::rnnoise::* implementation and :70 states tensors pass through unchanged so a future RnnoiseWeights::from_gguf can walk them. The tag is deliberately NOT "denoise".'
  'stable_audio_open_small|publish-gated: the Stability AI Community License is not SPDX-registered, so LicenseClass::from_license_str returns Unknown and publish requires --allow-noncommercial (the CPML / xtts_v2 precedent). stable_audio_open_small.rs:38 names a future vokra-models::stable_audio_open_small binder; none exists.'
  'tiger_separator|publish-only BF16 pass-through (JusperLee TIGER-DnR, apache-2.0). tiger.rs:70 names a future crates/vokra-models/src/tiger/ module and :56 defers TigerSeparator::from_gguf to a follow-up on the RMVPE / Charsiu loud-partial precedent.'
  'titanet-large|publish-only BF16 pass-through (NVIDIA TitaNet-L, cc-by-4.0 so AttributionRequired). titanet.rs:59 states the runtime port is out of scope for the converter wave; :10 reserves the tag for a future native TitaNet loader.'
  'unity-2|vast.ai-gated (~9.00 GB) AND NonCommercial: SeamlessM4T v2 Large is cc-by-nc-4.0, so publish needs --allow-noncommercial. The converter is a BF16 pass-through skeleton and seamless_m4t_v2_large.rs:29-31 keeps the tag distinct from the M4T v1 / MMS siblings so it cannot misroute the runtime binder (FR-EX-08). No binder module exists.'
  'xvector|publish-only BF16 pass-through (SpeechBrain x-vector, apache-2.0). xvector.rs:12 reserves the tag for a future vokra-models::xvector::* loader; no such module exists.'
  'yamnet|publish-only BF16 pass-through (YAMNet mirror, apache-2.0 with section 3.1 blank fail-closed pending owner confirmation of the mirror LICENSE). yamnet.rs:49 defers the prepare script and :24-25 keeps the tag distinct so a MobileNet checkpoint cannot route through a Cnn14 loader. Until 2026-08-15 the ONLY "yamnet" literals under vokra-models/src were the PANNs / ATST / MAEST tests asserting those binders REFUSE a YAMNet GGUF, and the gate was counting that refusal as proof of a binder.'
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
#
# This ledger was empty until 2026-08-15 — not because there were no gaps, but
# because the discovery regex could not see the 29 binders that spell the
# constant `EXPECTED_ARCH` instead of `ARCH`. Widening it surfaced exactly one
# real gap out of those 29 (`charsiu`); the other 28 were already emitted by a
# converter and simply had nothing checking them. The guard added in the same
# change (`unseen_arch_spellings`) is what stops the next spelling from
# re-opening this hole.
# ---------------------------------------------------------------------------
declare -a NO_CONVERTER=(
  'charsiu|reader-first by design, and the binder half is NOT wired either: no crates/vokra-convert/src/models/charsiu.rs exists, and every "charsiu" occurrence under vokra-convert/src is a doc comment citing the loud-partial precedent, never a stamped literal. Charsiu::from_gguf (align/charsiu.rs:509) verifies the arch tag and then refuses with LoadError::Gguf "from_gguf is not wired yet ... the upstream tensor-name manifest binder is a follow-up wave (T29-equivalent)", so this is NOT the dangerous shape of a working loader with no producer — nothing can be mis-bound, and the module is reachable only via Charsiu::new with caller-supplied weights. The tag is fixed reader-side first so the converter has exactly one string to match when it lands (align/charsiu.rs:76-94 states this verbatim), and it is deliberately distinct from wav2vec2_ctc because Charsiu head is an IPA phoneme inventory, not a letter vocab. Registered in engine.rs BOUND_ARCHES as BoundReason::NoGgufLoader — the variant whose explain() ends "there is nothing to bind from this artifact", i.e. the registry records the same refusal this reason opens with, which is what makes the two sides agree. That row shipped as NoCliShapedOutput ("the forward works, only the presentation blocks it") and was corrected in the same 2026-08-15 change that added this ledger line, so naming the old label here would have re-asserted, in the ledger, the exact claim that change deleted from the row. `charsiu` is correspondingly not a ModelKind::from_arg slug.'
)

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

Accepted gaps live in the NO_READER / NO_CONVERTER / NOT_A_MODEL_SLUG /
NOT_A_READER ledgers at the top of this script, one reason each. All four are
double-sided: an undeclared gap fails, and a ledger entry whose gap has since
been closed also fails. Exit 1 on any.
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
    python3 - "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" <<'PY'
import os, re, sys

(conv_models, conv_src, models_dir, engine_path, ledger_a, ledger_b,
 convert_lib, ledger_c, ledger_sup) = sys.argv[1:10]

# BOTH binder spellings are in scope. `EXPECTED_ARCH` is not a stylistic
# variant nobody uses: it is what 29 of the 89 arch constants under
# vokra-models/src are called (charsiu, csm, moshi, silero-vad, voxtral,
# zonos, the whole chatterbox family, …). Until 2026-08-15 this regex matched
# only the `ARCH` form, so this gate and its sibling both reported a confident
# green over a population missing a third of the binders — `charsiu` among
# them, which has no converter at all and is precisely the defect leg (b)
# exists to catch. A gate that is green because it did not look is worse than
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

reader_lits = answering_literals(reader_sites, suppressed)
# Leg (b) applies the same test-code exclusion but no suppression: no
# converter-side collision exists, and an empty ledger nobody can populate
# is worse than none. Mirror NOT_A_READER here if one ever appears.
emitter_lits = answering_literals(emitter_sites, {})

errors = (list(ledger_a_bad) + list(ledger_b_bad) + list(ledger_c_bad)
          + list(ledger_sup_bad) + list(reader_problems) + list(emitter_problems))

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

    # run <ledger-a...> -- <ledger-b...> [-- <ledger-c...> [-- <suppress...>]]
    run() {
        local -a la=() lb=() lc=() ls=()
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
                *) ls+=("$arg") ;;
            esac
        done
        write_ledger "$tmp/ledger_a" ${la[@]+"${la[@]}"}
        write_ledger "$tmp/ledger_b" ${lb[@]+"${lb[@]}"}
        write_ledger "$tmp/ledger_c" ${lc[@]+"${lc[@]}"}
        write_ledger "$tmp/ledger_sup" ${ls[@]+"${ls[@]}"}
        run_check "$tmp/conv/models" "$tmp/conv" "$tmp/models" "$tmp/engine.rs" \
            "$tmp/ledger_a" "$tmp/ledger_b" "$tmp/conv/lib.rs" "$tmp/ledger_c" \
            "$tmp/ledger_sup"
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
    write_ledger "$tmp/ledger_sup"
    mkdir -p "$tmp/empty"
    if run_check "$tmp/empty" "$tmp/conv" "$tmp/models" "$tmp/engine.rs" \
        "$tmp/ledger_a" "$tmp/ledger_b" "$tmp/conv/lib.rs" "$tmp/ledger_c" \
        "$tmp/ledger_sup" >/dev/null 2>&1; then
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

    if [ "$status" -eq 0 ]; then
        echo "check-arch-handshake --self-test: OK (25 cases)"
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
        trap 'rm -f "$LEDGER_A" "$LEDGER_B" "$LEDGER_C" "$LEDGER_SUP"' EXIT
        write_ledger "$LEDGER_A" ${NO_READER[@]+"${NO_READER[@]}"}
        write_ledger "$LEDGER_B" ${NO_CONVERTER[@]+"${NO_CONVERTER[@]}"}
        write_ledger "$LEDGER_C" ${NOT_A_MODEL_SLUG[@]+"${NOT_A_MODEL_SLUG[@]}"}
        write_ledger "$LEDGER_SUP" ${NOT_A_READER[@]+"${NOT_A_READER[@]}"}
        run_check "$CONVERT_MODELS_DEFAULT" "$CONVERT_SRC_DEFAULT" "$MODELS_DEFAULT" \
            "$ENGINE_DEFAULT" "$LEDGER_A" "$LEDGER_B" "$CONVERT_LIB_DEFAULT" "$LEDGER_C" \
            "$LEDGER_SUP"
        ;;
    *)
        echo "error: unknown argument '$1'" >&2
        usage >&2
        exit 1
        ;;
esac
