//! smart-turn v2 (`pipecat-ai/smart-turn-v2`, **BSD-2-Clause**) — semantic
//! **turn-completion / endpointing** runtime binder (Wave B 2026-08-15,
//! loud-partial per the `nisqa` / `emotion2vec` / `panns` / `utmosv2`
//! precedent — CLAUDE.md 教訓 (a)「loud-partial は fake-complete より
//! honest」).
//!
//! # THIS IS NOT A VAD — read this before wiring anything
//!
//! `crates/vokra-convert/src/models/smart_turn.rs` stamps
//! `vokra.model.category = "vad"`, and the upstream HF card's pipeline tag
//! is literally `voice-activity-detection` (primary source: HF cardData
//! API, recorded verbatim in `docs/license-audit.md` §3.1 row
//! "Smart-Turn v2", fetched 2026-07-30). **Both of those are catalog
//! labels, not an architectural claim**, and taking either at face value
//! is the single most likely way to misuse this model.
//!
//! A VAD answers *"is there speech in this frame?"* — a per-frame,
//! streaming, acoustic question, and its runtime shape is
//! [`vokra_core::engines::VadStreamHandle::push_pcm`] returning one
//! probability **per frame**.
//!
//! smart-turn answers *"has this speaker finished their turn?"* — a
//! **semantic endpointing** question asked once about a whole
//! utterance-length segment, and its runtime shape is one
//! [`TurnPrediction`] (a single turn-completion probability) **per
//! segment**. A speaker pausing mid-sentence is unambiguously *speech
//! present* (VAD says 1.0) and *turn not complete* (smart-turn says low)
//! at the same instant; that disagreement is the entire point of the
//! model, and it is why a realtime pipeline runs a VAD **and** a turn
//! detector rather than one in place of the other.
//!
//! Consequently [`vokra_core::engines::VadEngine`] is **deliberately NOT
//! implemented** for [`SmartTurn`] — see the "Why no `VadEngine` impl"
//! section below. `CATEGORY` is mirrored here only so the model-card
//! generator and the zoo-manifest tier gate keep agreeing with the
//! converter; nothing in this module treats it as a behavioural claim.
//!
//! # Primary sources
//!
//! - **HF release**: <https://huggingface.co/pipecat-ai/smart-turn-v2>
//!   (`license: bsd-2-clause`, `pipeline: voice-activity-detection` —
//!   HF cardData API, CC-verified 2026-07-30 and recorded in
//!   `docs/license-audit.md` §3.1; that row is already signed
//!   ☑ Commercial by the owner, and this module does not touch it).
//! - **Reference repo**: <https://github.com/pipecat-ai/smart-turn>
//!   (the pipecat-ai smart-turn training / inference reference — confirm
//!   the exact file paths during the flip-the-switch tensor walk).
//! - **Backbone**: w2v-BERT 2.0, <https://huggingface.co/facebook/w2v-bert-2.0>
//!   (**MIT**). In-repo cross-anchor:
//!   `crates/vokra-convert/src/models/w2v_bert_2.rs` already converts that
//!   backbone standalone under the distinct arch tag `w2v-bert-2`.
//! - **Backbone paper**: Chung et al. 2021, *"w2v-BERT: Combining
//!   Contrastive Learning and Masked Language Modeling for Self-Supervised
//!   Speech Pre-Training"*, <https://arxiv.org/abs/2108.06209> (cited from
//!   the in-repo `w2v_bert_2.rs` converter docstring).
//! - **In-repo survey row**: `docs/handoff/hf-audio-gap-comprehensive-2026-07-30.md`
//!   §3.4 — "pipecat wav2vec2-BERT turn-detection".
//!
//! # Architecture (as far as the primary sources specify it)
//!
//! ```text
//! PCM (mono f32, 16 kHz — w2v-BERT 2.0 lineage input convention)
//!   -> log-mel filterbank front-end                     <- **loud-partial**
//!        (the w2v-BERT 2.0 lineage consumes precomputed
//!         filterbank `input_features`, NOT a raw waveform
//!         like the wav2vec 2.0 lineage; the exact bin
//!         count / normalisation live in the upstream
//!         `preprocessor_config.json`, which the converter
//!         does not transcribe into any `vokra.*` chunk.)
//!   -> w2v-BERT 2.0 Conformer encoder stack             <- **loud-partial**
//!        (see "Why the existing Conformer port is not a
//!         drop-in" below — this is NOT the NeMo-flavoured
//!         `vokra_ops::conformer` variant.)
//!   -> pooling + turn-completion classification head    <- **loud-partial**
//!        (head width / activation / class-index order are
//!         not recoverable from the converter's metadata —
//!         see blocker (3).)
//!   -> ONE turn-completion probability for the whole segment
//!      ([`TurnPrediction`]) — never a per-frame vector.
//! ```
//!
//! # Why the existing Conformer port is not a drop-in
//!
//! `vokra-ops` **does** ship a Conformer (`vokra_ops::conformer`), so the
//! honest description of the gap is not "no Conformer exists" — it is that
//! the one that exists is the wrong flavour:
//!
//! - `vokra_ops::conformer::ConformerEncoder` is a direct port of the
//!   **NeMo** implementation (`nemo/collections/asr/modules/conformer_encoder.py`
//!   + `.../parts/submodules/conformer_modules.py`), built around a
//!     `ConvSubsampleKind::Stacking` subsampling stem and NeMo's parameter
//!     layout. Its consumers are the parakeet / canary / Qwen3-ASR family.
//! - w2v-BERT 2.0's encoder is the HuggingFace `Wav2Vec2BertEncoder`
//!   variant: it consumes precomputed filterbank features through a
//!   feature-projection (there is no conv subsampling stem to configure),
//!   and it differs from the NeMo layer in relative-position scheme,
//!   normalisation placement and parameter naming.
//!
//! That mismatch is exactly the dangerous kind: the two are close enough
//! that shapes can line up and a forward can *run*, producing a plausible
//! probability that is simply wrong. FR-EX-08 forbids that silent
//! misroute, so the forward stays a loud error until a real-checkpoint
//! tensor walk plus a parity dump against the upstream reference proves
//! the adapter correct.
//!
//! # Loud-partial classification (CLAUDE.md 教訓 (a))
//!
//! - **Real (this WP)**: strict `vokra.model.arch` verification that
//!   refuses foreign GGUFs loudly with the whole `category = "vad"`
//!   sibling fleet plus the bare backbone enumerated; a non-empty tensor
//!   gate; an encoder-stack presence gate; a classification-head presence
//!   gate (so a bare `w2v-bert-2` backbone GGUF cannot silently bind as a
//!   turn detector); the optional all-or-nothing `vokra.smart_turn.*`
//!   segment group; weight-license surfacing that fail-closes to
//!   [`LicenseClass::Unknown`]; and full input validation on
//!   [`SmartTurn::predict_endpoint`] (zero / mismatched sample rate, empty
//!   segment, over-long segment) — every one a loud
//!   [`VokraError::InvalidArgument`] that fires **before** the
//!   loud-partial gate, so a caller always gets the specific diagnostic.
//!
//! - **Loud-partial (this WP)**: [`SmartTurn::predict_endpoint`] returns
//!   [`VokraError::UnsupportedOp`] naming four concrete blockers (the
//!   w2v-BERT-flavoured Conformer adapter, the un-transcribed front-end /
//!   topology metadata, the unknown head contract, the missing converter
//!   sidecar). **No fabricated turn-completion probability is ever
//!   emitted** (FR-EX-08 — no silent partial output). A fabricated one
//!   here is unusually harmful: a wrong "turn complete" makes a realtime
//!   agent interrupt the user mid-sentence, which reads as a product bug
//!   rather than a model gap.
//!
//! # Why no `VadEngine` impl
//!
//! [`vokra_core::engines::VadEngine`] hands out a
//! [`vokra_core::engines::VadStreamHandle`] whose `push_pcm` returns
//! `Vec<f32>` — **one speech probability per frame**, consumed by the
//! `stream::open_stream` glue in `vokra-core` as a frame-level gate.
//! smart-turn has no per-frame output to put in that vector: it produces
//! one utterance-level decision. Implementing the trait would force a
//! choice between broadcasting the single probability across every frame
//! (a fabricated per-frame signal) or returning a one-element vector (a
//! silently wrong frame count). Both are FR-EX-08 violations, so the trait
//! is left unimplemented and the API is [`SmartTurn::predict_endpoint`]
//! instead. This is the same posture `nisqa` / `utmosv2` take toward
//! `MosScorerEngine`.
//!
//! # Cross-crate constant duplication
//!
//! [`ARCH`] / [`NAME`] / [`CATEGORY`] / [`UPSTREAM_HF`] /
//! [`DEFAULT_LICENSE_SPDX`] are duplicated from
//! `crates/vokra-convert/src/models/smart_turn.rs` rather than imported —
//! the same rule every sibling BF16 pass-through binder follows so
//! `vokra-models` does not gain a dependency edge onto `vokra-convert`,
//! preserving the layered convention `vokra-ops → nothing GGUF-aware`,
//! `vokra-core → GGUF reader`, `vokra-models → GGUF binder`,
//! `vokra-convert → GGUF writer`. The pins are asserted in the tests, so a
//! converter-side drift fails here in the same commit.
//!
//! # No ONNX / no pickle (permanent)
//!
//! The runtime never loads an ONNX graph or a Python pickle (FR-LD-05 /
//! NFR-DS-02). Any upstream artifact that is not already safetensors must
//! be flattened by the uv-managed Python 3.12 sidecar named in
//! [`SIDECAR_PATH`] (memory `[[feedback-python-uses-uv]]` +
//! `[[feedback-python-3-12]]`), which runs offline and never enters the
//! shipped binary.

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

// ---------------------------------------------------------------------------
// Contract constants — mirror of `crates/vokra-convert/src/models/smart_turn.rs`.
// See the module docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model smart-turn`.
///
/// Deliberately distinct from every sibling `category = "vad"` arch tag
/// (`fsmn-vad`, `firered_vad`, `silero-vad`, `pyannote-segmentation`) and
/// from the bare backbone tag `w2v-bert-2`, because smart-turn answers a
/// different question than all of them (semantic turn completion, not
/// frame-level voice activity) and has a different output shape. Silently
/// aliasing arch would misroute runtime dispatch — FR-EX-08.
pub const ARCH: &str = "smart_turn";

/// Expected `vokra.model.name` value written by the converter.
///
/// The upstream family already has a v3 (`pipecat-ai/smart-turn-v3`, noted
/// in `docs/handoff/hf-audio-gap-comprehensive-2026-07-30.md` §3.4 as
/// "weight 到着後 land"). The converter hard-codes this v2 name today, so
/// gating on it would be vacuous; [`SmartTurn::model_name`] surfaces
/// whatever the GGUF actually carries so a future v3 checkpoint is
/// distinguishable at runtime rather than silently assumed to be v2.
pub const NAME: &str = "smart-turn-v2";

/// Expected `vokra.model.category` value.
///
/// **A catalog tag, not a behavioural claim.** See the module docstring:
/// smart-turn is a turn-taking / endpointing model, not a VAD. This
/// constant exists so the model-card generator and the zoo-manifest tier
/// gate keep agreeing with the converter; nothing in this module branches
/// on it.
pub const CATEGORY: &str = "vad";

/// Upstream HuggingFace slug, echoed in load / loud-partial diagnostics so
/// a reader never has to re-fetch a manifest to find the source.
pub const UPSTREAM_HF: &str = "pipecat-ai/smart-turn-v2";

/// SPDX identifier the converter stamps by default
/// (→ [`LicenseClass::Permissive`]).
///
/// Primary source: HF cardData API `license: bsd-2-clause`, CC-verified
/// 2026-07-30 and recorded in `docs/license-audit.md` §3.1. That sign-off
/// row is owner-only and already signed; this module never writes to it.
pub const DEFAULT_LICENSE_SPDX: &str = "bsd-2-clause";

/// Model-category metadata key (`vokra.model.category`) — written by the
/// converter, surfaced by [`SmartTurn::category`].
pub const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Upstream HF slug metadata key (`vokra.provenance.upstream_hf`).
pub const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

// ---------------------------------------------------------------------------
// Primary-source anchors — cited verbatim in the loud-partial message.
// ---------------------------------------------------------------------------

/// Primary-source anchor for the smart-turn v2 HF release.
pub const PRIMARY_SOURCE_HF: &str = "huggingface.co/pipecat-ai/smart-turn-v2";

/// Primary-source anchor for the pipecat-ai smart-turn reference
/// training / inference repository. Confirm the exact file paths during
/// the flip-the-switch tensor walk.
pub const PRIMARY_SOURCE_CODE: &str = "github.com/pipecat-ai/smart-turn";

/// Primary-source anchor for the w2v-BERT 2.0 backbone release (MIT).
pub const PRIMARY_SOURCE_BACKBONE_HF: &str = "huggingface.co/facebook/w2v-bert-2.0";

/// Primary-source anchor for the w2v-BERT paper (Chung et al. 2021),
/// cited from the in-repo `w2v_bert_2.rs` converter docstring.
pub const PRIMARY_SOURCE_BACKBONE_PAPER: &str = "arxiv.org/abs/2108.06209";

/// The uv-managed Python 3.12 converter sidecar that must exist before the
/// real forward can land: it has to emit the `vokra.smart_turn.*` groups
/// from the upstream `config.json` / `preprocessor_config.json` (neither of
/// which the current verbatim float pass-through converter reads).
///
/// It does **not** exist yet — this path is a target, and the loud-partial
/// message says so.
pub const SIDECAR_PATH: &str = "tools/parity/smart_turn_prepare_checkpoint.py";

// ---------------------------------------------------------------------------
// Optional `vokra.smart_turn.*` segment group (all-or-nothing).
// ---------------------------------------------------------------------------

/// Sample rate the checkpoint's front-end expects, in Hz.
pub const KEY_SMART_TURN_SAMPLE_RATE: &str = "vokra.smart_turn.sample_rate";

/// Longest audio segment the checkpoint accepts for one turn-completion
/// query, in seconds.
pub const KEY_SMART_TURN_MAX_SEGMENT_SECONDS: &str = "vokra.smart_turn.max_segment_seconds";

/// Every key of the all-or-nothing `vokra.smart_turn.*` segment group.
///
/// The current converter stamps **none** of them (it is a verbatim float
/// pass-through that writes only arch / name / category / provenance), so
/// a GGUF produced today reads back `None` and the documented fallbacks
/// apply. A *partially* stamped group is a loud error, because silently
/// defaulting the missing half would produce a wrong-shaped guard with no
/// crash (FR-EX-08).
pub const SEGMENT_SPEC_KEYS: [&str; 2] = [
    KEY_SMART_TURN_SAMPLE_RATE,
    KEY_SMART_TURN_MAX_SEGMENT_SECONDS,
];

// ---------------------------------------------------------------------------
// Binder-chosen fallbacks. These are explicitly NOT transcriptions of
// upstream hyper-parameters — see each doc comment.
// ---------------------------------------------------------------------------

/// Sample rate assumed when the GGUF does not stamp
/// [`KEY_SMART_TURN_SAMPLE_RATE`].
///
/// 16 kHz mono is the input convention of the whole w2v-BERT 2.0 /
/// wav2vec2 lineage this model is built on, and it is what a pipecat
/// realtime pipeline feeds. Assuming it and then **refusing a mismatched
/// rate loudly** is the fail-closed choice: silently accepting 8 kHz or
/// 44.1 kHz audio would shift every filterbank bin and produce a
/// confidently wrong endpoint decision. Stamp
/// [`KEY_SMART_TURN_SAMPLE_RATE`] to override.
pub const DEFAULT_SAMPLE_RATE_HZ: u32 = 16_000;

/// Upper bound this binder accepts for one turn-completion query, in
/// seconds, when the GGUF does not stamp
/// [`KEY_SMART_TURN_MAX_SEGMENT_SECONDS`].
///
/// **This is a runtime guard chosen by this binder, NOT a transcription of
/// an upstream hyper-parameter.** The real receptive window of
/// `pipecat-ai/smart-turn-v2` is not recoverable from anything the
/// converter writes, and guessing it would be exactly the kind of
/// fabricated number CLAUDE.md forbids. The guard is therefore set an
/// order of magnitude above any plausible conversational turn: its job is
/// to catch "the caller handed us a whole podcast" mistakes before a
/// forward allocates for them, not to reproduce the model's own limit.
/// A checkpoint that stamps the real value overrides it.
pub const GUARD_MAX_SEGMENT_SECONDS: f32 = 60.0;

/// Neutral decision boundary for [`TurnPrediction::is_complete`].
///
/// 0.5 is the mathematically neutral midpoint of a probability, **not** a
/// tuned upstream deployment threshold — the value pipecat actually ships
/// is a product tuning knob this binder does not know. Callers running a
/// real pipeline should pass their own.
pub const DEFAULT_COMPLETION_THRESHOLD: f32 = 0.5;

// ---------------------------------------------------------------------------
// Tensor-manifest gates.
// ---------------------------------------------------------------------------

/// Substring that must appear in at least one tensor name for the GGUF to
/// be accepted as carrying a w2v-BERT-lineage encoder stack.
///
/// A substring rather than a prefix on purpose: every plausible wrapper
/// (`wav2vec2_bert.encoder.layers.0.…`, `model.encoder.layers.0.…`, bare
/// `encoder.layers.0.…`) leaves it intact, so the gate catches a
/// head-only or otherwise truncated GGUF without over-fitting to one
/// upstream module attribute name.
pub const TENSOR_SUBSTR_ENCODER_LAYERS: &str = "encoder.layers.";

/// Substrings, any one of which marks a classification head.
///
/// The gate exists to stop the most likely misroute in this
/// neighbourhood: a **bare `facebook/w2v-bert-2.0` backbone** converted
/// with the wrong `--model` flag would otherwise satisfy the encoder gate
/// and bind as a turn detector, then loud-partial with a message about the
/// head that is not there.
///
/// Fail-closed trade-off, stated plainly: if a legitimate checkpoint names
/// its head something outside this list, the load is **refused loudly**
/// with a message naming exactly what was searched. The fix is to extend
/// this list after confirming the name against the real manifest — never
/// to delete the gate, which would restore the silent-misroute hole.
pub const TENSOR_HEAD_SUBSTRS: [&str; 3] = ["classifier.", "turn_classifier.", "score."];

// ---------------------------------------------------------------------------
// TurnPrediction.
// ---------------------------------------------------------------------------

/// The result of one turn-completion query: a single probability that the
/// speaker has **finished their turn** over the whole submitted segment.
///
/// Deliberately not a `Vec<f32>`: the type itself is the documentation
/// that smart-turn produces one utterance-level decision, never a
/// per-frame speech track (see the module docstring's "THIS IS NOT A VAD"
/// section).
///
/// The constructor validates its input, so a `TurnPrediction` that exists
/// always carries a finite probability in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TurnPrediction {
    completion_probability: f32,
}

impl TurnPrediction {
    /// Wraps a turn-completion probability, refusing anything that is not
    /// a finite value in `[0, 1]`.
    ///
    /// A NaN slipping through here would silently compare `false` against
    /// every threshold, which reads as "the user is still talking" forever
    /// — a hang rather than a crash. Refusing loudly is the fail-closed
    /// choice (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] when `p` is NaN, infinite, or
    ///   outside `[0, 1]`.
    pub fn new(p: f32) -> Result<Self> {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(VokraError::InvalidArgument(format!(
                "smart_turn: turn-completion probability {p} is not a finite value in \
                 [0, 1]. A non-finite probability compares false against every \
                 threshold, so an endpointer would report 'still speaking' forever \
                 instead of failing — refusing loudly instead (FR-EX-08)."
            )));
        }
        Ok(Self {
            completion_probability: p,
        })
    }

    /// The probability that the speaker has finished their turn.
    #[inline]
    #[must_use]
    pub const fn completion_probability(&self) -> f32 {
        self.completion_probability
    }

    /// `true` when the completion probability reaches `threshold`.
    ///
    /// The caller supplies the threshold because it is a product tuning
    /// knob (how eager the agent should be to start replying), not a model
    /// constant; [`DEFAULT_COMPLETION_THRESHOLD`] is the neutral 0.5
    /// midpoint for callers that have not tuned one.
    #[inline]
    #[must_use]
    pub fn is_complete(&self, threshold: f32) -> bool {
        self.completion_probability >= threshold
    }
}

// ---------------------------------------------------------------------------
// SmartTurnSegmentSpec — the optional `vokra.smart_turn.*` group.
// ---------------------------------------------------------------------------

/// The segment-level front-end contract, when a GGUF stamps it.
///
/// Both fields come from the all-or-nothing [`SEGMENT_SPEC_KEYS`] group.
/// No current converter writes them, so this is `None` on every GGUF
/// produced today and the [`DEFAULT_SAMPLE_RATE_HZ`] /
/// [`GUARD_MAX_SEGMENT_SECONDS`] fallbacks apply.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmartTurnSegmentSpec {
    /// Sample rate the checkpoint's front-end expects, in Hz.
    pub sample_rate: u32,
    /// Longest segment accepted for one turn-completion query, in seconds.
    pub max_segment_seconds: f32,
}

impl SmartTurnSegmentSpec {
    /// Validates the stamped group loudly (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when the sample rate is zero, or the
    ///   segment cap is not a finite positive number of seconds.
    pub fn validate(&self) -> Result<()> {
        if self.sample_rate == 0 {
            return Err(VokraError::ModelLoad(format!(
                "smart_turn: `{KEY_SMART_TURN_SAMPLE_RATE}` = 0 — a front-end sample \
                 rate must be positive (a zero rate makes every duration bound divide \
                 by zero, so the over-long guard would never fire)."
            )));
        }
        if !self.max_segment_seconds.is_finite() || self.max_segment_seconds <= 0.0 {
            return Err(VokraError::ModelLoad(format!(
                "smart_turn: `{KEY_SMART_TURN_MAX_SEGMENT_SECONDS}` = {} is not a \
                 finite positive number of seconds — a non-positive or non-finite cap \
                 disables the over-long segment guard entirely (FR-EX-08).",
                self.max_segment_seconds
            )));
        }
        Ok(())
    }

    /// Reads the group from a GGUF. Returns `Ok(None)` when no key of the
    /// group is present; loud when it is only partially stamped.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] on a partially stamped group, a wrong
    ///   value type, or a failed [`Self::validate`].
    pub fn from_gguf(gguf: &GgufFile) -> Result<Option<Self>> {
        if !group_present(gguf, &SEGMENT_SPEC_KEYS) {
            return Ok(None);
        }
        let spec = Self {
            sample_rate: read_u32_key(gguf, KEY_SMART_TURN_SAMPLE_RATE)?,
            max_segment_seconds: read_f32_key(gguf, KEY_SMART_TURN_MAX_SEGMENT_SECONDS)?,
        };
        spec.validate()?;
        Ok(Some(spec))
    }
}

// ---------------------------------------------------------------------------
// Metadata read helpers (mirror of the sibling `nisqa` binder).
// ---------------------------------------------------------------------------

/// `true` when **any** key of an all-or-nothing group is present.
fn group_present(gguf: &GgufFile, keys: &[&str]) -> bool {
    keys.iter().any(|k| gguf.get(k).is_some())
}

/// Reads a required unsigned-integer key, refusing a wrong value type
/// rather than coercing (FR-EX-08).
fn read_u32_key(gguf: &GgufFile, key: &str) -> Result<u32> {
    let raw = gguf.get(key).and_then(|v| v.as_u64()).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "smart_turn: GGUF metadata `{key}` is missing or is not an unsigned \
             integer. The `vokra.smart_turn.*` group is all-or-nothing — a partially \
             stamped group is a bug in `{SIDECAR_PATH}`, and silently defaulting the \
             missing half would produce a wrong-shaped segment guard with no crash \
             (FR-EX-08)."
        ))
    })?;
    u32::try_from(raw).map_err(|_| {
        VokraError::ModelLoad(format!(
            "smart_turn: GGUF metadata `{key}` = {raw} does not fit in u32"
        ))
    })
}

/// Reads a required float key, refusing a wrong value type rather than
/// coercing (FR-EX-08).
fn read_f32_key(gguf: &GgufFile, key: &str) -> Result<f32> {
    let raw = gguf.get(key).and_then(|v| v.as_f64()).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "smart_turn: GGUF metadata `{key}` is missing or is not a float. The \
             `vokra.smart_turn.*` group is all-or-nothing — a partially stamped group \
             is a bug in `{SIDECAR_PATH}` (FR-EX-08)."
        ))
    })?;
    Ok(raw as f32)
}

// ---------------------------------------------------------------------------
// SmartTurnConfig.
// ---------------------------------------------------------------------------

/// smart-turn runtime config.
///
/// [`Self::segment`] is `None` until a converter sidecar stamps the
/// `vokra.smart_turn.*` group; the accessors apply the documented
/// binder-chosen fallbacks in that case.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmartTurnConfig {
    /// The stamped segment contract, when present.
    pub segment: Option<SmartTurnSegmentSpec>,
}

impl SmartTurnConfig {
    /// Derives the config from a parsed GGUF.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when the `vokra.smart_turn.*` group is
    ///   partially stamped or malformed.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        Ok(Self {
            segment: SmartTurnSegmentSpec::from_gguf(gguf)?,
        })
    }

    /// The sample rate this checkpoint's front-end expects, in Hz — the
    /// stamped value when present, otherwise [`DEFAULT_SAMPLE_RATE_HZ`].
    #[must_use]
    pub fn expected_sample_rate(&self) -> u32 {
        match self.segment {
            Some(s) => s.sample_rate,
            None => DEFAULT_SAMPLE_RATE_HZ,
        }
    }

    /// `true` when [`Self::expected_sample_rate`] is the binder-chosen
    /// [`DEFAULT_SAMPLE_RATE_HZ`] assumption rather than a value read from
    /// the GGUF. Surfaced so a diagnostic can say which it is.
    #[must_use]
    pub fn sample_rate_is_assumed(&self) -> bool {
        self.segment.is_none()
    }

    /// The longest segment accepted for one turn-completion query, in
    /// seconds — the stamped value when present, otherwise the
    /// binder-chosen [`GUARD_MAX_SEGMENT_SECONDS`] guard.
    #[must_use]
    pub fn max_segment_seconds(&self) -> f32 {
        match self.segment {
            Some(s) => s.max_segment_seconds,
            None => GUARD_MAX_SEGMENT_SECONDS,
        }
    }

    /// [`Self::max_segment_seconds`] expressed in samples at `sample_rate`
    /// (floored). Returns `0` for a degenerate rate, which makes every
    /// non-empty segment over-long — fail-closed by construction.
    #[must_use]
    pub fn max_segment_samples(&self, sample_rate: u32) -> usize {
        let s = f64::from(self.max_segment_seconds()) * f64::from(sample_rate);
        if !s.is_finite() || s <= 0.0 {
            return 0;
        }
        // float → int `as` casts saturate in Rust, so an absurd stamped cap
        // clamps to usize::MAX instead of wrapping.
        s.floor() as usize
    }
}

// ---------------------------------------------------------------------------
// SmartTurnWeights — tensor manifest with the structural gates.
// ---------------------------------------------------------------------------

/// Weight tensors bound from a smart-turn v2 GGUF.
///
/// Names + GGUF-side dims only: the forward is a loud-partial, so
/// preloading every tensor into RAM would buy nothing. The follow-up wave
/// that lands the real forward picks its own caching shape.
#[derive(Debug)]
pub struct SmartTurnWeights {
    tensors: Vec<(String, Vec<usize>)>,
}

impl SmartTurnWeights {
    /// Scans `gguf` for the smart-turn `state_dict` tensors and applies
    /// three loud gates: non-empty manifest, encoder-stack present,
    /// classification head present.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors.
    /// - [`VokraError::ModelLoad`] when no tensor name contains
    ///   [`TENSOR_SUBSTR_ENCODER_LAYERS`].
    /// - [`VokraError::ModelLoad`] when no tensor name contains any of
    ///   [`TENSOR_HEAD_SUBSTRS`] (the bare-backbone misroute).
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let mut tensors: Vec<(String, Vec<usize>)> = Vec::new();
        for info in gguf.tensors() {
            let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            tensors.push((info.name.clone(), dims));
        }

        if tensors.is_empty() {
            return Err(VokraError::ModelLoad(format!(
                "smart_turn: GGUF carries zero tensors — refusing to bind an all-zero \
                 forward (FR-EX-08). A legitimate smart-turn v2 checkpoint carries the \
                 whole w2v-BERT 2.0 Conformer encoder stack plus a turn-completion \
                 head (arch={ARCH}, name={NAME}); zero tensors always signals a \
                 mis-produced GGUF. Re-run `vokra-cli convert --model smart-turn` \
                 against an upstream `{UPSTREAM_HF}` safetensors checkpoint."
            )));
        }

        if !tensors
            .iter()
            .any(|(n, _)| n.contains(TENSOR_SUBSTR_ENCODER_LAYERS))
        {
            return Err(VokraError::ModelLoad(format!(
                "smart_turn: no tensor name contains `{TENSOR_SUBSTR_ENCODER_LAYERS}` \
                 — the w2v-BERT 2.0 Conformer encoder stack is missing. Every \
                 `{UPSTREAM_HF}` checkpoint carries it (the backbone is the bulk of \
                 the model); a GGUF without it is truncated or is a head-only export. \
                 Bound {count} tensor(s). Backbone reference: \
                 {PRIMARY_SOURCE_BACKBONE_HF}. Refusing rather than binding a model \
                 with no encoder (FR-EX-08).",
                count = tensors.len()
            )));
        }

        if !tensors
            .iter()
            .any(|(n, _)| TENSOR_HEAD_SUBSTRS.iter().any(|h| n.contains(*h)))
        {
            return Err(VokraError::ModelLoad(format!(
                "smart_turn: no tensor name contains any classification-head marker \
                 {TENSOR_HEAD_SUBSTRS:?} — this GGUF has an encoder but no \
                 turn-completion head. The most likely cause is that a BARE w2v-BERT \
                 2.0 backbone ({PRIMARY_SOURCE_BACKBONE_HF}, converter arch \
                 `w2v-bert-2`) was converted with `--model smart-turn` by mistake: a \
                 bare SSL encoder cannot decide anything about turn completion, and \
                 binding it here would misroute runtime dispatch (FR-EX-08). If a \
                 legitimate `{UPSTREAM_HF}` checkpoint names its head something else, \
                 confirm the name against the real manifest and EXTEND \
                 `TENSOR_HEAD_SUBSTRS` — do not delete the gate, which is what stops \
                 the bare-backbone misroute. Bound {count} tensor(s).",
                count = tensors.len()
            )));
        }

        Ok(Self { tensors })
    }

    /// Number of tensors bound from the GGUF.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// The bound tensor names with their GGUF-side dims — a diagnostic
    /// accessor for the follow-up forward wave.
    #[must_use]
    pub fn tensors(&self) -> &[(String, Vec<usize>)] {
        &self.tensors
    }
}

// ---------------------------------------------------------------------------
// SmartTurn — the runtime binder handle.
// ---------------------------------------------------------------------------

/// smart-turn v2 (`pipecat-ai/smart-turn-v2`, BSD-2-Clause) runtime binder
/// for **semantic turn completion / endpointing**.
///
/// Bind with [`from_gguf`](Self::from_gguf) or
/// [`from_path`](Self::from_path), then call
/// [`predict_endpoint`](Self::predict_endpoint) with an utterance-length
/// mono `f32` segment to obtain one [`TurnPrediction`].
///
/// **Not a VAD** — see the module docstring. [`vokra_core::engines::VadEngine`]
/// is deliberately not implemented.
#[derive(Debug)]
pub struct SmartTurn {
    cfg: SmartTurnConfig,
    weights: SmartTurnWeights,
    weight_license: LicenseClass,
    model_name: Option<String>,
    category: Option<String>,
    upstream_hf: Option<String>,
}

impl SmartTurn {
    /// Binds a smart-turn v2 GGUF: verifies the arch tag strictly, applies
    /// the tensor-manifest gates, reads the optional `vokra.smart_turn.*`
    /// group, and surfaces the stamped provenance.
    ///
    /// Every failure is a distinct [`VokraError::ModelLoad`] naming the
    /// missing or wrong key, so a reader diagnosing a mis-produced GGUF has
    /// exactly one place to walk (FR-EX-08 — never a silent partial bind).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent or is
    ///   not [`ARCH`] (a sibling `category = "vad"` GGUF — `fsmn-vad`,
    ///   `firered_vad`, `silero-vad`, `pyannote-segmentation` — or the bare
    ///   `w2v-bert-2` backbone handed here by mistake fails with a specific
    ///   message rather than a downstream missing-tensor error).
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors, no
    ///   encoder stack, or no classification head
    ///   ([`SmartTurnWeights::from_gguf`]).
    /// - [`VokraError::ModelLoad`] when the `vokra.smart_turn.*` group is
    ///   partially stamped or malformed.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check first, always — a sibling VAD GGUF or a bare
        //    backbone handed here must fail with a clear message, not a
        //    downstream shape error.
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "smart_turn: GGUF arch is `{other}`, expected `{ARCH}` (was this \
                     GGUF produced by `vokra-cli convert --model smart-turn`? The \
                     sibling `category = \"vad\"` arch tags — `fsmn-vad` (FunASR \
                     FSMN, per-frame speech/silence), `firered_vad` (Xiaohongshu \
                     FireRedVAD, per-frame), `silero-vad` (Silero v5 1:1-preserved \
                     subgraph, per-frame), `pyannote-segmentation` (per-frame \
                     speaker-activity segmentation) — all answer the frame-level \
                     question \"is there speech right now?\", while smart-turn answers \
                     the utterance-level SEMANTIC question \"has this speaker finished \
                     their turn?\" and emits ONE probability per segment instead of a \
                     per-frame track. The bare backbone `w2v-bert-2` \
                     ({PRIMARY_SOURCE_BACKBONE_HF}) is also distinct: it is an SSL \
                     encoder with no turn-completion head at all. Silently aliasing \
                     arch would misroute runtime dispatch onto a model with a \
                     different output shape (FR-EX-08 — no silent partial load)."
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(format!(
                    "smart_turn: GGUF is missing `vokra.model.arch` — this is not a \
                     Vokra-native smart_turn GGUF (was it produced by `vokra-cli \
                     convert --model smart-turn`? that converter stamps \
                     `vokra.model.arch = {ARCH}`)."
                )));
            }
        }

        // 2. Tensor manifest with the non-empty / encoder / head gates,
        //    then the optional metadata group.
        let weights = SmartTurnWeights::from_gguf(file)?;
        let cfg = SmartTurnConfig::from_gguf(file)?;

        // 3. Provenance surfacing. The converter stamps `bsd-2-clause`,
        //    which resolves to Permissive; a GGUF with no stamp fail-closes
        //    to Unknown (memory `[[feedback-license-signoff-primary-source]]`).
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);

        let model_name = file
            .get(chunks::KEY_MODEL_NAME)
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        let category = file
            .get(KEY_MODEL_CATEGORY)
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        let upstream_hf = file
            .get(KEY_PROVENANCE_UPSTREAM_HF)
            .and_then(|v| v.as_str())
            .map(str::to_owned);

        Ok(Self {
            cfg,
            weights,
            weight_license,
            model_name,
            category,
            upstream_hf,
        })
    }

    /// Opens and binds the model from a GGUF file on disk.
    ///
    /// # Errors
    ///
    /// - Whatever [`GgufFile::open`] returns, plus every error of
    ///   [`Self::from_gguf`].
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        Self::from_gguf(&gguf)
    }

    /// The config derived from the GGUF.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &SmartTurnConfig {
        &self.cfg
    }

    /// Number of tensors bound from the GGUF.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// The bound tensor names with their GGUF-side dims.
    #[inline]
    #[must_use]
    pub fn tensors(&self) -> &[(String, Vec<usize>)] {
        self.weights.tensors()
    }

    /// The stamped `vokra.model.name`, when present.
    ///
    /// Surfaced rather than gated so a future `pipecat-ai/smart-turn-v3`
    /// checkpoint is distinguishable at runtime instead of being silently
    /// assumed to be v2 — the arch tag alone cannot tell them apart.
    #[inline]
    #[must_use]
    pub fn model_name(&self) -> Option<&str> {
        self.model_name.as_deref()
    }

    /// The stamped `vokra.model.category`, when present.
    ///
    /// Expected to be [`CATEGORY`] (`"vad"`) — a catalog tag inherited from
    /// the converter and the upstream HF pipeline tag, **not** a claim that
    /// this model is a VAD. See the module docstring.
    #[inline]
    #[must_use]
    pub fn category(&self) -> Option<&str> {
        self.category.as_deref()
    }

    /// The stamped `vokra.provenance.upstream_hf`, when present.
    #[inline]
    #[must_use]
    pub fn upstream_hf(&self) -> Option<&str> {
        self.upstream_hf.as_deref()
    }

    /// The stamped weight-license class.
    ///
    /// The converter stamps `bsd-2-clause` → [`LicenseClass::Permissive`];
    /// a GGUF without the stamp reads back as [`LicenseClass::Unknown`]
    /// (fail-closed at the M2-13 compliance gate).
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// `true` when the bound weights may only be used behind a research
    /// flag. The canonical BSD-2-Clause weights answer `false`; an
    /// unstamped GGUF answers `true` (fail-closed).
    #[inline]
    #[must_use]
    pub fn is_research_only(&self) -> bool {
        self.weight_license.requires_research_flag()
    }

    /// Predicts whether the speaker has finished their turn over an
    /// utterance-length mono `f32` segment.
    ///
    /// This is **semantic endpointing over a whole segment**, not
    /// frame-level voice activity: the result is one [`TurnPrediction`],
    /// and there is deliberately no streaming / per-frame variant (see the
    /// module docstring's "Why no `VadEngine` impl").
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] — see
    /// [`predict_endpoint_loud_partial`] for the four blockers and the
    /// flip-the-switch recipe. **No fabricated turn-completion probability
    /// is ever emitted** (FR-EX-08): a wrong "turn complete" makes a
    /// realtime agent interrupt the user mid-sentence, so a plausible
    /// fake here is worse than a loud failure.
    ///
    /// All input validation fires **before** the loud-partial gate, so a
    /// caller with a malformed request always gets the specific
    /// diagnostic rather than the generic "not implemented" message.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] when `sample_rate` is `0`.
    /// - [`VokraError::InvalidArgument`] when `sample_rate` differs from
    ///   [`SmartTurnConfig::expected_sample_rate`] — resampling silently
    ///   would shift every filterbank bin.
    /// - [`VokraError::InvalidArgument`] when `pcm` is empty — there is no
    ///   turn to judge.
    /// - [`VokraError::InvalidArgument`] when `pcm` is longer than
    ///   [`SmartTurnConfig::max_segment_seconds`].
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate.
    pub fn predict_endpoint(&self, pcm: &[f32], sample_rate: u32) -> Result<TurnPrediction> {
        if sample_rate == 0 {
            return Err(VokraError::InvalidArgument(
                "smart_turn predict_endpoint: sample_rate must be > 0 (a zero rate has \
                 no duration interpretation, so neither the segment guard nor the \
                 front-end could be applied)"
                    .to_owned(),
            ));
        }

        let expected = self.cfg.expected_sample_rate();
        if sample_rate != expected {
            let origin = if self.cfg.sample_rate_is_assumed() {
                format!(
                    "the binder-assumed w2v-BERT 2.0 lineage convention \
                     ({DEFAULT_SAMPLE_RATE_HZ} Hz) because this GGUF does not stamp \
                     `{KEY_SMART_TURN_SAMPLE_RATE}`"
                )
            } else {
                format!("the GGUF's stamped `{KEY_SMART_TURN_SAMPLE_RATE}`")
            };
            return Err(VokraError::InvalidArgument(format!(
                "smart_turn predict_endpoint: sample rate mismatch — got {sample_rate} \
                 Hz but the front-end expects {expected} Hz, from {origin}. Resampling \
                 silently would shift every filterbank bin and yield a confidently \
                 wrong endpoint decision, so this is a loud refusal (FR-EX-08). \
                 Resample upstream, or stamp `{KEY_SMART_TURN_SAMPLE_RATE}` if the \
                 checkpoint really was trained at {sample_rate} Hz."
            )));
        }

        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(format!(
                "smart_turn predict_endpoint: the audio segment is empty — there is no \
                 turn to judge. This model scores an UTTERANCE-LENGTH segment and \
                 returns one turn-completion probability; it is not a per-frame VAD \
                 that can be fed a zero-length chunk. Pass the speaker's segment (see \
                 `{PRIMARY_SOURCE_HF}`)."
            )));
        }

        let max_samples = self.cfg.max_segment_samples(sample_rate);
        if pcm.len() > max_samples {
            let cap_origin = if self.cfg.sample_rate_is_assumed() {
                format!(
                    "the binder's `GUARD_MAX_SEGMENT_SECONDS` runtime guard, which is \
                     an order-of-magnitude ceiling chosen HERE and not a transcription \
                     of the upstream receptive window; stamp \
                     `{KEY_SMART_TURN_MAX_SEGMENT_SECONDS}` with the real value to \
                     override it"
                )
            } else {
                format!("the GGUF's stamped `{KEY_SMART_TURN_MAX_SEGMENT_SECONDS}`")
            };
            return Err(VokraError::InvalidArgument(format!(
                "smart_turn predict_endpoint: segment is {len} samples \
                 ({secs:.3} s at {sample_rate} Hz), over the {cap} s cap = \
                 {max_samples} samples. The cap comes from {cap_origin}. A turn \
                 detector scores one conversational turn, not an entire recording — \
                 segment the audio (a VAD is the usual upstream stage) and query once \
                 per turn.",
                len = pcm.len(),
                secs = pcm.len() as f64 / f64::from(sample_rate),
                cap = self.cfg.max_segment_seconds(),
            )));
        }

        // The loud-partial gate fires only after every input check, and
        // before any front-end work, so a caller can never observe a
        // partial computation that looks like a real forward.
        Err(predict_endpoint_loud_partial())
    }
}

/// Constructs the loud-partial [`VokraError::UnsupportedOp`] returned by
/// [`SmartTurn::predict_endpoint`] until the real forward lands.
///
/// The message names all four blockers so a reader (or the follow-up wave)
/// knows exactly where to flip the switch:
///
/// 1. **The missing adapter** — a w2v-BERT-flavoured Conformer.
///    `vokra_ops::conformer` exists but is the NeMo variant; see the
///    module docstring for why substituting it would be silent-wrong
///    rather than a compile error.
/// 2. **The missing metadata** — the converter is a verbatim float
///    pass-through that stamps no topology or front-end axes at all, so
///    encoder depth / width / head count and the whole filterbank
///    front-end are unrecoverable from the GGUF.
/// 3. **The missing head contract** — whether the head emits one logit
///    through a sigmoid or two through a softmax, and which class index
///    means "complete", is not recoverable from the converter's output.
///    Guessing has a 50% chance of inverting the decision, which is worse
///    than useless in a realtime agent.
/// 4. **The missing sidecar** — [`SIDECAR_PATH`] does not exist; it must
///    read the upstream `config.json` / `preprocessor_config.json` and emit
///    the `vokra.smart_turn.*` group.
#[must_use]
pub fn predict_endpoint_loud_partial() -> VokraError {
    VokraError::UnsupportedOp(format!(
        "smart_turn predict_endpoint (loud-partial): the turn-completion forward is \
         deferred; four pieces must land before a real probability can be emitted. \
         (1) MISSING ADAPTER: a w2v-BERT-flavoured Conformer encoder. `vokra-ops` \
         DOES ship `vokra_ops::conformer`, but it is a direct port of the NeMo \
         implementation (conformer_encoder.py + conformer_modules.py) with a \
         Stacking subsampling stem and NeMo parameter naming, serving the \
         parakeet / canary family; w2v-BERT 2.0 uses the HuggingFace \
         `Wav2Vec2BertEncoder` variant, which projects precomputed filterbank \
         features (no conv subsampling stem) and differs in relative-position \
         scheme, normalisation placement and parameter naming. The shapes are close \
         enough that a substituted forward would RUN and return a plausible wrong \
         number instead of failing — exactly the silent misroute FR-EX-08 forbids. \
         (2) MISSING METADATA: the `vokra.smart_turn.*` group. The converter \
         (crates/vokra-convert/src/models/smart_turn.rs) is a verbatim F32/F16/BF16 \
         pass-through that stamps only arch / name / category / provenance — NO \
         topology or front-end axes — so encoder depth, hidden width, head count and \
         the whole log-mel front-end (bin count, normalisation, window) are \
         unrecoverable from the GGUF. Note the w2v-BERT 2.0 lineage consumes \
         filterbank `input_features`, NOT a raw waveform like wav2vec 2.0; \
         best-guessing that front-end from the wrong lineage would be silent-wrong. \
         (3) MISSING HEAD CONTRACT: whether the turn-completion head emits one logit \
         through a sigmoid or two through a softmax, and which class index means \
         'turn complete', is not recoverable from the converter's output. A guessed \
         index has a 50% chance of INVERTING the decision, which makes a realtime \
         agent interrupt exactly when it should wait. \
         (4) MISSING SIDECAR: `{SIDECAR_PATH}` does not exist — it must read the \
         upstream `config.json` / `preprocessor_config.json` and emit the \
         `vokra.smart_turn.*` group (uv-managed Python 3.12, offline; no pickle or \
         ONNX ever enters the runtime, FR-LD-05). \
         Output once real: ONE turn-completion probability in [0, 1] for the whole \
         submitted segment (`TurnPrediction`) — never a per-frame vector; this is a \
         semantic endpointer, not a VAD, despite the inherited \
         `vokra.model.category = \"{CATEGORY}\"` catalog tag and the upstream HF \
         `voice-activity-detection` pipeline tag. Primary sources: release {hf}, \
         reference repo {code}, backbone {backbone} ({paper}). Runtime cannot \
         fabricate a turn-completion probability (FR-EX-08 — no silent partial \
         output).",
        hf = PRIMARY_SOURCE_HF,
        code = PRIMARY_SOURCE_CODE,
        backbone = PRIMARY_SOURCE_BACKBONE_HF,
        paper = PRIMARY_SOURCE_BACKBONE_PAPER,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the smart-turn v2 runtime binder — contract-constant pins
    //! (cross-crate consistency with the converter), metadata round-trip,
    //! and negative-space round-trip on every loud gate.
    //!
    //! # What "round-trip" means here
    //!
    //! On a real checkpoint this would be `predict_endpoint(...)` returning
    //! a turn-completion probability, but the forward is a loud-partial
    //! (see the module doc). Fabricating a probability would violate
    //! CLAUDE.md 教訓 (a)「loud-partial は fake-complete より honest」, and
    //! is unusually harmful for this model: a fake "turn complete" makes a
    //! realtime agent interrupt the user.
    //!
    //! The semantics we *can* honestly test:
    //!
    //! 1. **Contract-constant pin** — `ARCH` / `NAME` / `CATEGORY` /
    //!    `UPSTREAM_HF` / `DEFAULT_LICENSE_SPDX` match the converter
    //!    exactly, so a converter drift fails here in the same commit.
    //! 2. **Metadata round-trip** — a synthetic GGUF binds, and the
    //!    provenance / config surfaces read back with the right semantics
    //!    (Permissive stamp binds, `Unknown` fallback fires when absent,
    //!    the optional segment group round-trips or falls back).
    //! 3. **Loud-error negative space** — every stated blocker (missing
    //!    arch / wrong arch / empty manifest / missing encoder / missing
    //!    head / partial metadata group / bad input / unsupported forward)
    //!    fires at its documented surface point, in the documented error
    //!    variant.
    //! 4. **Pure-function behaviour** — `TurnPrediction` validation and the
    //!    segment-cap arithmetic are real code with real assertions.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// A representative encoder tensor name (satisfies the encoder gate).
    const T_ENCODER: &str = "wav2vec2_bert.encoder.layers.0.self_attn.linear_q.weight";
    /// A representative head tensor name (satisfies the head gate). Mirrors
    /// the placeholder the converter's own test module uses.
    const T_HEAD: &str = "turn_classifier.head.weight";

    /// Builds a legitimate smart-turn GGUF: arch + name + category +
    /// upstream slug + optional weight-license stamp + optional segment
    /// group + one encoder tensor + one head tensor.
    fn smart_turn_gguf(
        weight_license_class: Option<LicenseClass>,
        segment: Option<(u32, f32)>,
    ) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
        b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        if let Some((rate, secs)) = segment {
            b.add_u32(KEY_SMART_TURN_SAMPLE_RATE, rate);
            b.add_f32(KEY_SMART_TURN_MAX_SEGMENT_SECONDS, secs);
        }
        b.add_tensor(T_ENCODER, GgmlType::F32, vec![2, 3], vec![0u8; 2 * 3 * 4])
            .expect("add encoder tensor");
        b.add_tensor(T_HEAD, GgmlType::F32, vec![2, 2], vec![0u8; 2 * 2 * 4])
            .expect("add head tensor");
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------------
    // Test 1 — Contract-constant pin (cross-crate consistency)
    // -----------------------------------------------------------------------

    #[test]
    fn contract_constants_mirror_the_converter() {
        assert_eq!(ARCH, "smart_turn", "smart-turn arch tag pin");
        assert_eq!(NAME, "smart-turn-v2", "canonical name pin");
        // CATEGORY is a CATALOG tag inherited from the converter and the
        // upstream HF `voice-activity-detection` pipeline tag. It is
        // deliberately NOT a claim that this model is a VAD — see the
        // module docstring. Pinned so a converter-side change surfaces
        // here, not so any behaviour branches on it.
        assert_eq!(CATEGORY, "vad", "catalog category pin (NOT a VAD claim)");
        assert_eq!(UPSTREAM_HF, "pipecat-ai/smart-turn-v2", "upstream slug pin");
        assert_eq!(DEFAULT_LICENSE_SPDX, "bsd-2-clause", "default SPDX pin");
        // The arch tag must stay distinct from every sibling it could be
        // confused with — a silent alias is the FR-EX-08 boundary.
        for sibling in [
            "fsmn-vad",
            "firered_vad",
            "silero-vad",
            "pyannote-segmentation",
            "w2v-bert-2",
        ] {
            assert_ne!(ARCH, sibling, "arch must stay distinct from `{sibling}`");
        }
    }

    // -----------------------------------------------------------------------
    // Test 2 — happy path: a synthetic GGUF binds and surfaces provenance
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_binds_synthetic_gguf() {
        let file = smart_turn_gguf(Some(LicenseClass::Permissive), None);
        let m = SmartTurn::from_gguf(&file).expect("valid GGUF must bind");

        assert_eq!(
            m.weight_license(),
            LicenseClass::Permissive,
            "the converter's bsd-2-clause stamp resolves to Permissive"
        );
        assert!(!m.is_research_only(), "BSD-2-Clause is not research-only");
        assert_eq!(m.tensor_count(), 2, "both fixture tensors bind");
        assert_eq!(m.model_name(), Some(NAME));
        assert_eq!(m.category(), Some(CATEGORY));
        assert_eq!(m.upstream_hf(), Some(UPSTREAM_HF));

        // No `vokra.smart_turn.*` group on a converter-produced GGUF today,
        // so the documented binder-chosen fallbacks apply.
        assert!(m.config().segment.is_none());
        assert!(m.config().sample_rate_is_assumed());
        assert_eq!(m.config().expected_sample_rate(), DEFAULT_SAMPLE_RATE_HZ);
        assert!(
            (m.config().max_segment_seconds() - GUARD_MAX_SEGMENT_SECONDS).abs() < 1e-6,
            "unstamped cap falls back to the runtime guard"
        );
        assert_eq!(
            m.config().max_segment_samples(DEFAULT_SAMPLE_RATE_HZ),
            (GUARD_MAX_SEGMENT_SECONDS as usize) * (DEFAULT_SAMPLE_RATE_HZ as usize),
        );
    }

    // -----------------------------------------------------------------------
    // Test 3 — missing arch is loud
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_arch() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_NAME, "some-other-name");
        b.add_tensor(T_ENCODER, GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        b.add_tensor(T_HEAD, GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = SmartTurn::from_gguf(&file) else {
            panic!("expected ModelLoad on missing arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`vokra.model.arch`"),
                    "message must name the missing key, got `{m}`"
                );
                assert!(
                    m.contains("not a Vokra-native smart_turn GGUF"),
                    "message must name the missing-arch surface, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 4 — wrong arch is loud and enumerates the confusable fleet
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_wrong_arch() {
        // An `fsmn-vad` GGUF handed here by mistake: same
        // `category = "vad"` catalog tag, completely different question and
        // output shape (per-frame speech track vs one turn-completion
        // probability). FR-EX-08 forbids the silent misroute.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "fsmn-vad");
        b.add_string(chunks::KEY_MODEL_NAME, "fsmn-vad-zh-cn-16k-common");
        b.add_tensor(
            "encoder.in_linear.weight",
            GgmlType::F32,
            vec![4, 4],
            vec![0u8; 64],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = SmartTurn::from_gguf(&file) else {
            panic!("expected ModelLoad on wrong arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`fsmn-vad`") && m.contains("`smart_turn`"),
                    "message must name both the got and expected arch tags, got `{m}`"
                );
                for sibling in [
                    "fsmn-vad",
                    "firered_vad",
                    "silero-vad",
                    "pyannote-segmentation",
                    "w2v-bert-2",
                ] {
                    assert!(
                        m.contains(sibling),
                        "expected sibling `{sibling}` disambiguation in error: {m}"
                    );
                }
                // The message must call out the turn-taking vs VAD
                // distinction — that is the whole reason this arch is
                // separate.
                assert!(
                    m.contains("finished their turn"),
                    "message must state the semantic turn-completion question, got `{m}`"
                );
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite the FR-EX-08 clause, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 5 — empty tensor manifest is loud
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_empty_tensor_list() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, "permissive");
        // NO tensors added.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = SmartTurn::from_gguf(&file) else {
            panic!("expected ModelLoad on empty tensor manifest");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("zero tensors"),
                    "message must name the empty-manifest gap, got `{m}`"
                );
                assert!(m.contains("FR-EX-08"), "must cite FR-EX-08, got `{m}`");
                assert!(
                    m.contains("vokra-cli convert --model smart-turn"),
                    "message must include the repro command, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 6 — missing encoder stack is loud and names the substring
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_encoder_stack() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        // Head present, encoder absent — a truncated / head-only export.
        b.add_tensor(T_HEAD, GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = SmartTurn::from_gguf(&file) else {
            panic!("expected ModelLoad on missing encoder stack");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains(TENSOR_SUBSTR_ENCODER_LAYERS),
                    "message must name the missing tensor substring \
                     `{TENSOR_SUBSTR_ENCODER_LAYERS}`, got `{m}`"
                );
                assert!(m.contains("FR-EX-08"), "must cite FR-EX-08, got `{m}`");
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 7 — a bare backbone (encoder, no head) is loud
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_bare_backbone_without_head() {
        // The classic misroute: `facebook/w2v-bert-2.0` converted with
        // `--model smart-turn`. It satisfies the encoder gate but has no
        // turn-completion head, and binding it would produce a "model" that
        // cannot decide anything.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_tensor(T_ENCODER, GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = SmartTurn::from_gguf(&file) else {
            panic!("expected ModelLoad on missing classification head");
        };
        match err {
            VokraError::ModelLoad(m) => {
                for marker in TENSOR_HEAD_SUBSTRS {
                    assert!(
                        m.contains(marker),
                        "message must name the searched head marker `{marker}`, got `{m}`"
                    );
                }
                assert!(
                    m.contains("w2v-bert-2"),
                    "message must name the bare-backbone misroute, got `{m}`"
                );
                assert!(m.contains("FR-EX-08"), "must cite FR-EX-08, got `{m}`");
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 8 — the optional segment group round-trips
    // -----------------------------------------------------------------------

    #[test]
    fn segment_group_round_trips() {
        let file = smart_turn_gguf(Some(LicenseClass::Permissive), Some((16_000, 8.0)));
        let m = SmartTurn::from_gguf(&file).expect("stamped GGUF must bind");

        let spec = m.config().segment.expect("segment group must be present");
        assert_eq!(spec.sample_rate, 16_000);
        assert!((spec.max_segment_seconds - 8.0).abs() < 1e-6);
        assert!(!m.config().sample_rate_is_assumed());
        assert_eq!(m.config().expected_sample_rate(), 16_000);
        assert_eq!(m.config().max_segment_samples(16_000), 128_000);
    }

    // -----------------------------------------------------------------------
    // Test 9 — a partially stamped group is loud (never half-defaulted)
    // -----------------------------------------------------------------------

    #[test]
    fn partially_stamped_segment_group_is_loud() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        // Only half the all-or-nothing group.
        b.add_u32(KEY_SMART_TURN_SAMPLE_RATE, 16_000);
        b.add_tensor(T_ENCODER, GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        b.add_tensor(T_HEAD, GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = SmartTurn::from_gguf(&file) else {
            panic!("expected ModelLoad on a partially stamped group");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains(KEY_SMART_TURN_MAX_SEGMENT_SECONDS),
                    "message must name the missing half of the group, got `{m}`"
                );
                assert!(
                    m.contains("all-or-nothing"),
                    "message must state the all-or-nothing rule, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 10 — a degenerate stamped group is loud
    // -----------------------------------------------------------------------

    #[test]
    fn degenerate_segment_group_is_loud() {
        // A zero cap would disable the over-long guard entirely.
        let file = smart_turn_gguf(None, Some((16_000, 0.0)));
        let Err(err) = SmartTurn::from_gguf(&file) else {
            panic!("expected ModelLoad on a zero segment cap");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains(KEY_SMART_TURN_MAX_SEGMENT_SECONDS),
                    "message must name the offending key, got `{m}`"
                );
                assert!(m.contains("FR-EX-08"), "must cite FR-EX-08, got `{m}`");
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }

        // A zero sample rate would make every duration bound divide by zero.
        let file = smart_turn_gguf(None, Some((0, 8.0)));
        let Err(err) = SmartTurn::from_gguf(&file) else {
            panic!("expected ModelLoad on a zero sample rate");
        };
        match err {
            VokraError::ModelLoad(m) => assert!(
                m.contains(KEY_SMART_TURN_SAMPLE_RATE),
                "message must name the offending key, got `{m}`"
            ),
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 11 — an empty segment is a loud InvalidArgument
    // -----------------------------------------------------------------------

    #[test]
    fn predict_endpoint_rejects_empty_segment() {
        let file = smart_turn_gguf(Some(LicenseClass::Permissive), None);
        let m = SmartTurn::from_gguf(&file).expect("bind");

        let Err(err) = m.predict_endpoint(&[], DEFAULT_SAMPLE_RATE_HZ) else {
            panic!("expected InvalidArgument on an empty segment");
        };
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("empty"),
                    "message must name the empty-segment gap, got `{msg}`"
                );
                // The diagnostic must reinforce the utterance-level posture
                // so a caller does not "fix" it by feeding VAD-sized chunks.
                assert!(
                    msg.contains("UTTERANCE-LENGTH"),
                    "message must state the utterance-level contract, got `{msg}`"
                );
                assert!(
                    msg.contains("not a per-frame VAD"),
                    "message must state that this is not a per-frame VAD, got `{msg}`"
                );
            }
            other => panic!("expected VokraError::InvalidArgument, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 12 — an over-long segment is a loud InvalidArgument
    // -----------------------------------------------------------------------

    #[test]
    fn predict_endpoint_rejects_over_long_segment() {
        // Stamp a small cap so the fixture stays cheap: 0.5 s at 16 kHz =
        // 8000 samples, so 8001 is one sample over.
        let file = smart_turn_gguf(Some(LicenseClass::Permissive), Some((16_000, 0.5)));
        let m = SmartTurn::from_gguf(&file).expect("bind");
        assert_eq!(m.config().max_segment_samples(16_000), 8_000);

        // Exactly at the cap must NOT trip the guard — it must fall through
        // to the loud-partial instead. (An off-by-one here would reject
        // legitimate maximum-length turns.)
        let at_cap = vec![0.0_f32; 8_000];
        let Err(err) = m.predict_endpoint(&at_cap, 16_000) else {
            panic!("expected an error (loud-partial) at exactly the cap");
        };
        assert!(
            matches!(err, VokraError::UnsupportedOp(_)),
            "a segment exactly at the cap must reach the loud-partial gate, got {err:?}"
        );

        // One sample over must be refused.
        let over = vec![0.0_f32; 8_001];
        let Err(err) = m.predict_endpoint(&over, 16_000) else {
            panic!("expected InvalidArgument on an over-long segment");
        };
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("8001") && msg.contains("8000"),
                    "message must report the actual length and the cap, got `{msg}`"
                );
                assert!(
                    msg.contains(KEY_SMART_TURN_MAX_SEGMENT_SECONDS),
                    "message must name the cap's source key, got `{msg}`"
                );
            }
            other => panic!("expected VokraError::InvalidArgument, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 13 — a bad sample rate is a loud InvalidArgument
    // -----------------------------------------------------------------------

    #[test]
    fn predict_endpoint_rejects_bad_sample_rate() {
        let file = smart_turn_gguf(Some(LicenseClass::Permissive), None);
        let m = SmartTurn::from_gguf(&file).expect("bind");
        let pcm = vec![0.0_f32; 1_600];

        // Zero rate has no duration interpretation at all.
        let Err(err) = m.predict_endpoint(&pcm, 0) else {
            panic!("expected InvalidArgument on a zero sample rate");
        };
        assert!(matches!(err, VokraError::InvalidArgument(_)), "{err:?}");

        // A mismatched rate must be refused rather than silently resampled.
        let Err(err) = m.predict_endpoint(&pcm, 44_100) else {
            panic!("expected InvalidArgument on a mismatched sample rate");
        };
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("44100") && msg.contains("16000"),
                    "message must report both rates, got `{msg}`"
                );
                assert!(
                    msg.contains("binder-assumed"),
                    "message must disclose that the expected rate is assumed, not \
                     stamped, got `{msg}`"
                );
                assert!(msg.contains("FR-EX-08"), "must cite FR-EX-08, got `{msg}`");
            }
            other => panic!("expected VokraError::InvalidArgument, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 14 — the forward is a loud-partial naming all four blockers
    // -----------------------------------------------------------------------

    #[test]
    fn predict_endpoint_loud_partial_returns_unsupported_op() {
        // No license stamp — the binder still binds (arch + manifest are
        // the load gates; license is a compliance surface) but fail-closes
        // the class to Unknown.
        let file = smart_turn_gguf(None, None);
        let m = SmartTurn::from_gguf(&file).expect("valid arch must bind");
        assert_eq!(
            m.weight_license(),
            LicenseClass::Unknown,
            "a missing license stamp must fail-closed to Unknown"
        );
        assert!(
            m.is_research_only(),
            "an unstamped GGUF is research-only until proven otherwise"
        );

        // A plausible one-second turn at the expected rate.
        let pcm = vec![0.0_f32; DEFAULT_SAMPLE_RATE_HZ as usize];
        let Err(err) = m.predict_endpoint(&pcm, DEFAULT_SAMPLE_RATE_HZ) else {
            panic!("predict_endpoint must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(
                    msg.contains("smart_turn predict_endpoint"),
                    "surface must be called out: {msg}"
                );
                assert!(msg.contains("loud-partial"), "posture label: {msg}");

                // All four blockers, by their labels.
                for blocker in [
                    "MISSING ADAPTER",
                    "MISSING METADATA",
                    "MISSING HEAD CONTRACT",
                    "MISSING SIDECAR",
                ] {
                    assert!(msg.contains(blocker), "expected blocker `{blocker}`: {msg}");
                }

                // Blocker (1) must be honest that a Conformer DOES exist —
                // the gap is that it is the wrong flavour, which is a
                // silent-wrong risk rather than a compile error.
                assert!(
                    msg.contains("vokra_ops::conformer") && msg.contains("NeMo"),
                    "must name the existing NeMo Conformer port and why it is not a \
                     drop-in: {msg}"
                );
                assert!(
                    msg.contains("Wav2Vec2BertEncoder"),
                    "must name the required HF encoder variant: {msg}"
                );

                // Blocker (4) must name the sidecar path.
                assert!(msg.contains(SIDECAR_PATH), "must name the sidecar: {msg}");

                // Every primary source must be cited.
                for url in [
                    PRIMARY_SOURCE_HF,
                    PRIMARY_SOURCE_CODE,
                    PRIMARY_SOURCE_BACKBONE_HF,
                    PRIMARY_SOURCE_BACKBONE_PAPER,
                ] {
                    assert!(msg.contains(url), "expected primary source `{url}`: {msg}");
                }

                // The output contract must be restated so nobody wires this
                // in as a per-frame VAD.
                assert!(
                    msg.contains("ONE turn-completion probability")
                        && msg.contains("never a per-frame vector"),
                    "must restate the utterance-level output contract: {msg}"
                );
                assert!(
                    msg.contains("FR-EX-08"),
                    "expected the FR-EX-08 rationale for no fake probability: {msg}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 15 — TurnPrediction validation (real pure-function behaviour)
    // -----------------------------------------------------------------------

    #[test]
    fn turn_prediction_validates_its_probability() {
        let p = TurnPrediction::new(0.75).expect("0.75 is a valid probability");
        assert!((p.completion_probability() - 0.75).abs() < 1e-6);
        assert!(p.is_complete(DEFAULT_COMPLETION_THRESHOLD));
        assert!(!p.is_complete(0.9));

        // The boundaries are valid.
        assert!(TurnPrediction::new(0.0).is_ok());
        assert!(TurnPrediction::new(1.0).is_ok());

        // Out-of-range and non-finite values are refused loudly — a NaN
        // would compare false against every threshold, so an endpointer
        // would silently report "still speaking" forever.
        for bad in [
            -0.001_f32,
            1.001,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            let Err(err) = TurnPrediction::new(bad) else {
                panic!("expected InvalidArgument for probability {bad}");
            };
            match err {
                VokraError::InvalidArgument(msg) => assert!(
                    msg.contains("[0, 1]"),
                    "message must state the valid range, got `{msg}`"
                ),
                other => panic!("expected VokraError::InvalidArgument, got {other:?}"),
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 16 — segment-cap arithmetic (fail-closed on a degenerate rate)
    // -----------------------------------------------------------------------

    #[test]
    fn max_segment_samples_is_fail_closed() {
        let cfg = SmartTurnConfig {
            segment: Some(SmartTurnSegmentSpec {
                sample_rate: 16_000,
                max_segment_seconds: 2.5,
            }),
        };
        assert_eq!(cfg.max_segment_samples(16_000), 40_000);
        // Fractional products floor rather than round up, so the guard can
        // never admit a sample past the cap.
        assert_eq!(cfg.max_segment_samples(3), 7, "2.5 s * 3 Hz = 7.5 -> 7");
        // A degenerate rate yields 0, which makes every non-empty segment
        // over-long — fail-closed by construction.
        assert_eq!(cfg.max_segment_samples(0), 0);
    }
}
