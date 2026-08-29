//! # SBV2 (Style-Bert-VITS2 v2) native TTS.
//!
//! Clean-room Apache-2.0 implementation of Style-Bert-VITS2 v2 inference,
//! per the design doc `docs/superpowers/specs/2026-07-26-sbv2-v2-design.md`.
//!
//! # References (permissive only)
//!
//! - VITS paper: arXiv:2106.06103 (Kim et al. 2021)
//! - jaywalnut310/vits (MIT): VITS core reference
//! - VITS2 paper: arXiv:2307.16430
//! - p0p4k/vits2_pytorch (MIT): VITS2 code reference
//! - DeBERTa v2 paper: arXiv:2006.03654
//! - DeBERTa v3 paper: arXiv:2111.09543
//! - HF transformers deberta_v2/v3 (Apache-2.0): BERT reference
//!
//! # NOT REFERENCED
//!
//! - github.com/litagin02/Style-Bert-VITS2 (AGPL-3.0)
//! - github.com/fishaudio/Bert-VITS2 (AGPL-3.0)
//! - Any AGPL derivative of the above.
//!
//! # Language coverage caveat (JP-Extra base checkpoint)
//!
//! Current base ckpt is a JP-Extra fine-tune — EN/ZH synthesis runs but
//! produces degraded audio; genuine multilingual synthesis requires switching
//! to a non-JP-Extra multilingual base. See `docs/adr/sbv2-cleanroom.md` and
//! CLAUDE.md "documented ceilings".
//!
//! # ZH code-side status (WP-21, updated 2026-08-18)
//!
//! **Forward pointer — additive, does not contradict the JP-Extra caveat
//! above (which still governs real-checkpoint audio quality).** As of the
//! Phase D SBV2 v2 wave, the ZH code path is wired end-to-end at the
//! scaffolding level: [`text_encoder::N_LANGUAGES`] = 3 (JA/EN/ZH),
//! [`Language::ZH`] and its `language_id() = 2` dispatch to
//! [`SbV2TextEncoder`]'s `language_embed` row 2 are exercised in
//! synthetic-parity tests, and the WordPiece tokenizer aimed at the
//! owner-approved ZH BERT checkpoint (`hfl/chinese-roberta-wwm-ext-large`,
//! Apache-2.0) landed as [`vokra_bert::wordpiece::BertWordpieceTokenizer`]
//! (WP-17). See [`g2p::Language`]'s "ZH scope note" for the enum-side
//! historical + owner-decision context.
//!
//! **Owner decisions (2026-08-09)**:
//!
//! - **ZH BERT PERMITTED** = `hfl/chinese-roberta-wwm-ext-large`
//!   (Apache-2.0, standard `BertForMaskedLM` — vocab 21128 / hidden 1024
//!   / 24 layers / 16 heads / **learned absolute position embedding +
//!   token_type + word embedding sum + LayerNorm**, i.e. **standard BERT
//!   NOT DeBERTa disentangled** — a `BertBaseEncoder` distinct from
//!   [`DebertaV2Encoder`] / [`DebertaV3Encoder`] is required, per
//!   `wordpiece.rs`'s target-model doc). The optional four-file loader
//!   wires this encoder into `Language::ZH`; publishing remains a separate,
//!   explicitly authorized operation.
//! - **ZH G2P = piper-plus reuse** = bridge the existing 8-language G2P
//!   via the excluded-workspace `integrations/vokra-piper-g2p` crate
//!   (which already routes `zh` through `PassthroughPhonemizer`; see that
//!   crate's `README.md`).
//!
//! **Remaining runtime gap**:
//!
//! 1. [`g2p::SbV2Phonemizer::phonemize`]'s `Language::ZH` arm still returns
//!    [`VokraError::NotImplemented`] — fail-closed placeholder for the
//!    future piper-plus reuse route (never a silent JA fallback, FR-EX-08).
//!
//! Genuine ZH synthesis quality additionally requires
//! (i) `hfl/chinese-roberta-wwm-ext-large` weights on the runtime side,
//! (ii) a production Mandarin G2P implementation rather than a parity
//! fixture, and (iii) owner §3.1 license sign-off before any HF publish.

pub mod decoder;
pub mod duration;
pub mod flow;
pub mod g2p;
pub mod parity;
pub mod rng_mode;
pub mod speaker;
pub mod spline;
pub mod style;
pub mod text_encoder;

pub use rng_mode::RngMode;
// Task 25 (SBV2 v2 plan) places the safetensors -> GGUF converter in
// `crates/vokra-convert/src/models/sbv2.rs` instead of a `mod converter`
// here -- avoids a `vokra-models <-> vokra-convert` normal-dependency
// cycle (this crate depends on `vokra-convert` only as a dev-dependency,
// for M4-04-style roundtrip tests; see that file's module doc for the full
// rationale, mirroring Task 11's identical DeBERTa converter placement).

pub use decoder::SbV2Decoder;
pub use duration::{ConvFlow, DDSConv, ElementwiseAffine, SbV2SDP, SdpLayerNorm, length_regulate};
pub use flow::{Flip, FlowLayer, SbV2Flow, SbV2TransformerCouplingLayer};
pub use g2p::{Language, OovPolicy, PhonemizeFixture, PhonemizeResult, SbV2Phonemizer};
pub use parity::{
    ATOL_DEFAULT, AtolCalibration, MEL_LOSS_ATOL, PER_TENSOR_ATOL, UTMOS_ATOL,
    atol_calibration_for, tolerance_for,
};
pub use speaker::{ExternalSpeakerProjection, SpeakerEmbedding};
pub use style::StyleVectorInjector;
pub use text_encoder::{BertBridge, N_LANGUAGES, SbV2TextEncoder};

// ---------------------------------------------------------------------------
// Task 23: SbV2Model — full pipeline integration
// ---------------------------------------------------------------------------
//
// Wires Tasks 14-22 above (this crate) plus Stage A's `vokra-bert` (DeBERTa
// v2/v3 + SentencePiece tokenizer) into one `SbV2Model::synthesize` forward
// pass, plus a `vokra_core::TtsEngine` adapter over the cross-engine
// `SynthesisRequest` shape — the same "thin adapter converts the unified
// request, then calls the model's own inherent `synthesize`" pattern
// `piper_plus::PiperPlusTts`'s own `impl TtsEngine` uses
// (`crates/vokra-models/src/piper_plus/mod.rs`). See
// `docs/superpowers/specs/2026-07-26-sbv2-v2-design.md` §7's pipeline
// diagram for the canonical forward order this mirrors.

use vokra_bert::bert_base::BertBaseEncoder;
use vokra_bert::deberta_v2::DebertaV2Encoder;
use vokra_bert::deberta_v3::DebertaV3Encoder;
use vokra_bert::tokenizer::SbertTokenizer;
use vokra_bert::wordpiece::BertWordpieceTokenizer;
use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::rng::{GaussianSplitMix64, TorchRandnStream};
use vokra_core::{BackendKind, Result, SynthesisRequest, SynthesizedAudio, TtsEngine, VokraError};
use vokra_ops::attrs::{HifiGanAttrs, ResBlockType};
use vokra_ops::hifigan::{
    GinCondition, HifiGanConfig, HifiGanWeights, MrfBranchWeights, ResBlockLayer,
    UpsampleStageWeights,
};

use crate::bert_runtime::{
    bert_base_forward_with_backend, deberta_v2_forward_with_backend,
    deberta_v3_forward_with_backend,
};
use crate::compute::{Compute, HotOp};

/// Complete learned-op set used by the SBV2 Metal path. Host-side tokenization,
/// indexing, layout transforms, residual adds and RNG are intentionally not
/// listed because they are not learned kernels.
pub(crate) const SBV2_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::Gelu,
    HotOp::Conv1d,
    HotOp::GroupedConv1d,
    HotOp::Gemv,
];

/// The `vokra.model.arch` value the **main** Style-Bert-VITS2 GGUF must
/// carry (the `bert_*` side-cars carry their own, different tags — see
/// [`SbV2Model::from_gguf`]).
///
/// Mirror of `crates/vokra-convert/src/models/sbv2.rs::ARCH` — the
/// converter owns the writer contract, this module owns the reader
/// contract (the deliberate two-copies convention [`crate::pyannote`]
/// documents; a compile-time check would need `vokra-convert` in
/// `vokra-models`'s dependency graph, which the workspace pins forbid).
///
/// The converter's own test suite pins this tag as distinct from every
/// near neighbour it could be confused with — `piper-plus-mb-istft-vits2`,
/// `vits-ja`, `deberta_v2`, `deberta_v3` (`…/sbv2.rs`
/// `arch_is_distinct_from_siblings`). Those last two matter especially
/// here: they are the arch tags of this loader's *own* `bert_ja` /
/// `bert_en` arguments, so an argument-order slip is a live failure mode.
pub const EXPECTED_ARCH: &str = "sbv2";

/// Rejects a **main** GGUF whose `vokra.model.arch` is absent or is not
/// [`EXPECTED_ARCH`].
///
/// A *loud* validation step (FR-EX-08) — see [`SbV2Model::from_gguf`].
fn verify_main_arch(main: &GgufFile) -> Result<()> {
    match main.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
        Some(a) if a == EXPECTED_ARCH => Ok(()),
        Some(other) => Err(VokraError::ModelLoad(format!(
            "SbV2Model::from_gguf: `main` GGUF arch is `{other}`, expected `{EXPECTED_ARCH}` \
             (was this GGUF produced by `vokra-cli convert --model sbv2`?). Two mis-routes \
             are common here: (a) a sibling VITS-lineage TTS GGUF — \
             `piper-plus-mb-istft-vits2`, `vits-ja` — which shares VITS ancestry but not \
             this tensor manifest; (b) an argument-order slip that passes one of this \
             loader's own BERT side-cars (`deberta_v2` for `bert_ja`, `deberta_v3` for \
             `bert_en`, `bert_base` for `bert_zh`) in the `main` slot. Either way the load \
             would bind whatever names happen to overlap (FR-EX-08 — no silent partial \
             load)."
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "SbV2Model::from_gguf: `main` GGUF is missing `{}` — this is not a Vokra-native \
             Style-Bert-VITS2 GGUF (was it produced by `vokra-cli convert --model sbv2`?)",
            chunks::KEY_MODEL_ARCH,
        ))),
    }
}

/// Both BERT encoders (+ their tokenizers) [`SbV2Model`] needs, loaded
/// together so one loaded model instance can serve either language without
/// a reload: JA text routes through [`DebertaV2Encoder`], EN through
/// [`DebertaV3Encoder`] (`docs/superpowers/specs/2026-07-26-sbv2-v2-design.md`
/// §7's "BERT Router").
///
/// # WP-19: optional ZH branch
///
/// The [`zh`](Self::zh) / [`zh_tokenizer`](Self::zh_tokenizer) pair holds
/// the ZH BERT branch introduced by WP-19 (owner decision 2026-08-09:
/// `hfl/chinese-roberta-wwm-ext-large`, `BertForMaskedLM`, Apache-2.0 —
/// **plain BERT**, not DeBERTa, so [`BertBaseEncoder`] rather than
/// `DebertaV2Encoder`/`DebertaV3Encoder`; WordPiece rather than
/// SentencePiece). Both fields default to `None` — the legacy 3-file
/// (`main` + `bert_ja` + `bert_en`) loader
/// [`SbV2Model::from_gguf`] and every synthetic `SbV2Model` constructor
/// leave them `None`, so pre-WP-19 code continues to compile and behave
/// identically. The WP-19 [`SbV2Model::from_gguf_with_zh_bert`] loader
/// populates them together (either both `Some`, both `None` — one-side
/// is a caller bug the loader refuses), matching
/// [`SbV2Model::synthesize`]'s ZH-branch dispatch which likewise refuses
/// to run with only one side of the pair wired (FR-EX-08).
pub struct SbV2BertContainer {
    /// SentencePiece tokenizer feeding [`ja`](Self::ja)'s input ids.
    pub ja_tokenizer: SbertTokenizer,
    /// SentencePiece tokenizer feeding [`en`](Self::en)'s input ids.
    pub en_tokenizer: SbertTokenizer,
    /// JA BERT encoder (DeBERTa v2).
    pub ja: DebertaV2Encoder,
    /// EN BERT encoder (DeBERTa v3).
    pub en: DebertaV3Encoder,
    /// ZH BERT encoder (plain BERT, `hfl/chinese-roberta-wwm-ext-large`)
    /// — `Some` only on 4-file
    /// [`SbV2Model::from_gguf_with_zh_bert`]-loaded models and the
    /// [`SbV2Model::synthetic_with_zh_for_test`] synthetic constructor.
    /// See the struct-level "WP-19: optional ZH branch" doc.
    pub zh: Option<BertBaseEncoder>,
    /// WordPiece tokenizer feeding [`zh`](Self::zh)'s input ids — paired
    /// with [`zh`](Self::zh) (either both `Some`, both `None`; a mixed
    /// state is a caller bug that [`SbV2Model::synthesize`]'s ZH-branch
    /// dispatch refuses to accept, per FR-EX-08).
    pub zh_tokenizer: Option<BertWordpieceTokenizer>,
}

/// Inputs to [`SbV2Model::synthesize`] — the SBV2-native request shape,
/// distinct from [`vokra_core::SynthesisRequest`] (the cross-engine unified
/// shape [`SbV2Model`]'s [`TtsEngine`] adapter converts to/from; see that
/// impl's doc for the field-by-field mapping).
#[derive(Debug, Clone)]
pub struct SbV2SynthRequest {
    /// Raw input text (JA or EN — `language` selects the G2P + BERT path).
    pub text: String,
    /// Which phonemizer / BERT path the request routes through.
    pub language: Language,
    /// Discrete speaker id, looked up in [`SpeakerEmbedding`]'s table.
    ///
    /// This is the **synthetic / legacy** speaker-selection path
    /// [`SbV2Model::synthetic_for_test`]-shaped models use — used only
    /// when both this request's [`speaker_embedding`](Self::speaker_embedding)
    /// is `None` **and** the model has no
    /// [`ExternalSpeakerProjection`] loaded
    /// (via [`SbV2Model::with_external_speaker_projection`]). For the
    /// real SBV2 v2 base ckpt (which has no per-speaker table), pass a
    /// caller-supplied external embedding via
    /// [`speaker_embedding`](Self::speaker_embedding) instead — that is
    /// the sole speaker-conditioning path the base ckpt supports (scout
    /// report §1: `emb_g` does not exist on that ckpt).
    pub speaker_id: u32,
    /// External zero-shot speaker embedding (Blocker 3) — a
    /// caller-supplied `[d_speaker]` (real ckpt: `d_speaker = 512`)
    /// continuous embedding, projected through the model's
    /// [`ExternalSpeakerProjection`] to a `[d_model]` broadcast-add
    /// contribution (see [`SbV2Model::synthesize`]'s step 5 for the
    /// full dispatch table).
    ///
    /// - `None`: the deterministic zero-shot default — mirrors the
    ///   cross-engine [`vokra_core::SynthesisRequest::speaker_embedding`]'s
    ///   documented "None uses the zero vector" contract; the projection
    ///   layer still contributes its bias to the resulting zero-input
    ///   output, so synthesis is deterministic but not silent.
    /// - `Some(vec)`: `vec.len()` **must** equal the projection's `d_in`
    ///   — a wrong-length vector is a loud
    ///   [`vokra_core::VokraError::InvalidArgument`], never silently
    ///   zero-padded or truncated (FR-EX-08).
    /// - `Some(_)` on a model with **no** projection loaded (a synthetic
    ///   `SbV2Model::synthetic_for_test` that was never handed a projection
    ///   via [`SbV2Model::with_external_speaker_projection`]) is also a
    ///   loud [`vokra_core::VokraError::InvalidArgument`]: silently
    ///   discarding caller-supplied speaker data would produce
    ///   plausible-looking-but-wrong-speaker audio, exactly the class of
    ///   silent failure FR-EX-08 forbids.
    pub speaker_embedding: Option<Vec<f32>>,
    /// Per-utterance AdaIN style conditioning (see
    /// [`StyleVectorInjector::inject`]). Length must equal the loaded
    /// voice's style width — an all-zero vector of the right length is the
    /// identity (no-op) style ([`StyleVectorInjector`]'s module doc).
    pub style_vec: Vec<f32>,
    /// Speed multiplier applied to predicted per-phoneme durations
    /// (`duration / speed`, floored at 1 frame): `1.0` = unchanged, `>
    /// 1.0` = faster speech, `< 1.0` = slower. Must be positive —
    /// [`synthesize`](SbV2Model::synthesize) rejects `speed <= 0.0`.
    pub speed: f32,
    /// Flow-latent noise scale, consumed by [`SbV2Model::synthesize`]
    /// (FLOW-NOISE-SCALE fix, 2026-08-09) as part of the VITS-family
    /// prior reparameterization: `z_p = mel_hidden + torch.randn *
    /// noise_scale` before [`SbV2Flow::inverse`]. Draws use the
    /// [`rng_mode`](Self::rng_mode) dispatch (torch-parity RNG by
    /// default) with the same `seed` the SDP consumes.
    ///
    /// `noise_scale = 0.0` short-circuits the RNG entirely — a
    /// deterministic byte-frozen pipeline regardless of `seed` /
    /// `rng_mode`. This is the fully-deterministic posture every
    /// pre-Step-10 synthetic parity test uses.
    ///
    /// Upstream default (from `docs/superpowers/specs/2026-07-26-sbv2-v2-design.md`
    /// §7) is `0.667`. The full VITS-family scheme also multiplies the
    /// scaled noise by `exp(logs_p)` — a mean/logstd split from the
    /// prior head (`enc_p.proj.*`, tracked as SBV2-INFO-01-ENC-P-PROJ)
    /// — which this scaffold treats as `logs = 0` (equivalent to
    /// `logstd = 0`, i.e. `exp(0) = 1`) pending the prior-head loader.
    pub noise_scale: f32,
    /// Stochastic-duration-predictor noise scale, forwarded to
    /// [`SbV2SDP::sample`] — `0.0` is fully deterministic (every predicted
    /// duration is `1` regardless of `seed`).
    pub noise_scale_w: f32,
    /// Seed for the duration predictor's Gaussian draws (see
    /// [`rng_mode`](Self::rng_mode) for which RNG the seed feeds).
    /// Irrelevant when `noise_scale_w == 0.0`.
    pub seed: u64,
    /// Which RNG family the SDP's Gaussian noise draws come from. Default is
    /// [`RngMode::PhiloxRngEnginePyTorchParity`], which byte-matches
    /// `torch.manual_seed(seed); torch.randn(...)` under the PhiloxRNGEngine.h
    /// path (see [`RngMode`]'s module doc for the rationale). Pre-Step-10
    /// synthetic tests preserve their byte-frozen assertions by explicitly
    /// setting `RngMode::GaussianSplitMix64Legacy` on their requests.
    pub rng_mode: RngMode,
}

/// Per-stage intermediate tensors from
/// [`SbV2Model::synthesize_with_intermediates`], each field corresponding
/// to one entry in `tools/parity/sbv2_dump_reference.py`'s manifest
/// `tensors[]` array (design doc §10). Every buffer is flat, row-major
/// — see each field's doc for its exact shape and dumper filename.
///
/// Wave-4 INTERMEDIATE-ACCESSORS (2026-08-09): added so
/// `crates/vokra-models/tests/parity_sbv2_real.rs` can diff per-stage
/// tensors against Python reference `.bin` fixtures instead of only
/// comparing the final waveform (`parity_kokoro` / `parity_whisper` have
/// per-tensor tables for exactly this reason). The Python dumper writes
/// each field as its own `reference_dump/<name>.bin` file; the
/// [`Self::to_dumper_map`] helper maps every Rust field to its dumper
/// filename for a driver-style tensor-diff loop.
///
/// The only intermediate the dumper writes but this struct does NOT
/// carry separately is `phonemize_fixture/*` (the G2P inputs — those are
/// captured in the request, not the model's output).
#[derive(Debug, Clone)]
pub struct SbV2Intermediates {
    /// `[T_text, d_model]` — **pre-scale** embedding sum
    /// `(phoneme_embed[id] + tone_embed[tone] + language_embed[lang])`,
    /// snapshot BEFORE the `sqrt(d_model)` scale multiplication. Dumper
    /// filename: `phoneme_embed.bin`. Matches the Python reference
    /// dumper's `tools/parity/sbv2_dump_reference.py:928`
    /// (`phoneme_embed = x_phon + x_tone + lang_row`, before the line-929
    /// `x = phoneme_embed * sqrt(D_MODEL)` step) — see
    /// [`SbV2TextEncoder::forward_with_embed`](super::text_encoder::SbV2TextEncoder::forward_with_embed)'s
    /// "Snapshot convention" doc for the parity-CI root cause behind this
    /// convention (2026-08-09 fix, run 31314913038).
    pub phoneme_embed: Vec<f32>,
    /// `[T_text, d_model]` — post-transformer text encoder hidden state
    /// (the value fed to the SDP at step 6 — see the Bug-4 fix comment
    /// in [`SbV2Model::synthesize_with_intermediates`]). Dumper filename:
    /// `text_hidden.bin`.
    pub text_hidden: Vec<f32>,
    /// `[T_bert_ja, D_BERT]` — DeBERTa v2 (JA) `last_hidden_state`, present
    /// on JA-language requests. Dumper filename: `bert_hidden_ja.bin`.
    /// Empty when `req.language != Language::JA`.
    pub bert_hidden_ja: Vec<f32>,
    /// `[T_bert_en, D_BERT]` — DeBERTa v3 (EN) `last_hidden_state`, present
    /// on EN-language requests. Dumper filename: `bert_hidden_en.bin`.
    /// Empty when `req.language != Language::EN`.
    pub bert_hidden_en: Vec<f32>,
    /// `[T_bert_zh, D_BERT]` — plain BERT (ZH)
    /// `last_hidden_state`, present on ZH-language requests. Dumper filename:
    /// `bert_hidden_zh.bin`. Empty when `req.language != Language::ZH`.
    ///
    /// This field is populated only when the model was loaded through the
    /// four-file ZH path and a ZH request reaches the forward pass. Keeping a
    /// separate bucket prevents the parity harness from accidentally
    /// comparing the plain-BERT output against either DeBERTa fixture.
    pub bert_hidden_zh: Vec<f32>,
    /// `[T_text, d_model]` — bert-bridge projected contribution alone,
    /// pre-addition into `text_hidden`. The dumper writes
    /// `bert_bridge_out.bin = text_hidden + bridge_projected`
    /// (matching Python `bert_bridge_out` which the length regulator
    /// consumes); this field carries the SAME sum so the accessor can be
    /// diffed against `reference_dump/bert_bridge_out.bin` directly.
    pub bert_bridge_out: Vec<f32>,
    /// `[d_speaker]` — raw speaker conditioning vector, before per-stage
    /// slice/pad reconciliation (SDP / flow / decoder each slice or pad
    /// this to their own gin_channels; the intermediate is the pre-slice
    /// source). Dumper filename: `speaker_embed.bin`. Length depends on
    /// which speaker path fired (external projection / lookup / synthetic).
    pub speaker_embed: Vec<f32>,
    /// `[d_target]` — [`StyleVectorInjector::project`] output on
    /// `req.style_vec`. Dumper filename: `style_projected.bin`. See that
    /// method's doc for why this returns the bias projection alone (the
    /// Python reference has one linear projection; Vokra's AdaIN has two).
    pub style_projected: Vec<f32>,
    /// `[T_text]` — SDP-sampled per-phoneme durations (post-speed
    /// scaling, post-OOM-clamp, discretized to i32). Dumper filename:
    /// `sdp_sample.bin`.
    pub sdp_sample: Vec<i32>,
    /// `[T_mel, d_model]` — length-regulated `hidden_for_flow` (=
    /// `text_hidden + bert_bridge_out`, expanded by `sdp_sample`).
    /// Dumper filename: `mel_hidden.bin`.
    pub mel_hidden: Vec<f32>,
    /// `[T_mel, d_z]` — normalizing-flow `inverse` output (post-noise
    /// reparameterization). Dumper filename: `z_latent.bin`.
    pub z_latent: Vec<f32>,
}

/// SBV2 (Style-Bert-VITS2 v2) native TTS model: the full inference pipeline
/// wiring [`SbV2Phonemizer`] (G2P) → [`SbV2TextEncoder`] +
/// [`SbV2BertContainer`] + [`BertBridge`] (text/BERT hidden state) →
/// [`SpeakerEmbedding`] + [`StyleVectorInjector`] (conditioning) →
/// [`SbV2SDP`] + [`length_regulate`] (duration) → [`SbV2Flow`] (acoustic
/// latent) → [`SbV2Decoder`] (HiFi-GAN → PCM). See
/// [`synthesize`](Self::synthesize) for the exact forward order and
/// `docs/superpowers/specs/2026-07-26-sbv2-v2-design.md` §7 for the design
/// pipeline this mirrors.
///
/// # Scaffold-level dimensional assumption (Task 23)
///
/// [`synthesize`](Self::synthesize)'s speaker-embedding broadcast add has no
/// projection layer to reconcile a width mismatch, so it assumes
/// `speaker_embed`'s per-speaker embedding width equals
/// `text_encoder.d_model()` (checked with a `debug_assert!` at the point of
/// use). It likewise assumes `bert_bridge`'s projected-contribution width
/// equals `text_encoder.d_model()` (also `debug_assert!`-checked). Real Task
/// 24-27 checkpoint weights that violate either will need a projection layer
/// added at that point.
pub struct SbV2Model {
    backend: BackendKind,
    phonemizer: SbV2Phonemizer,
    text_encoder: SbV2TextEncoder,
    bert: SbV2BertContainer,
    bert_bridge: BertBridge,
    speaker_embed: SpeakerEmbedding,
    /// Real-ckpt zero-shot speaker projection (Blocker 3). `Some` when a
    /// caller has bound the real SBV2 v2 base ckpt's
    /// `enc_p.encoder.spk_emb_linear.{weight,bias}` (either via
    /// [`SbV2Model::with_external_speaker_projection`] on a
    /// hand-assembled model, or via the future real-ckpt path in
    /// [`SbV2Model::from_gguf`] once the converter renames those two
    /// tensors — see the loader's own doc). `None` on
    /// [`SbV2Model::synthetic_for_test`]-shaped models (which route
    /// speaker through the legacy [`SpeakerEmbedding`] lookup instead).
    /// See [`SbV2Model::synthesize`]'s step 5 for the exact dispatch
    /// table.
    speaker_projection: Option<ExternalSpeakerProjection>,
    style_injector: StyleVectorInjector,
    sdp: SbV2SDP,
    flow: SbV2Flow,
    decoder: SbV2Decoder,
}

impl SbV2Model {
    /// Assembles a model from its nine pre-trained/pre-built components (see
    /// the struct's field list — one argument per field, same order).
    #[allow(clippy::too_many_arguments)] // one arg per model component, mirrors the struct's fields
    pub fn new(
        phonemizer: SbV2Phonemizer,
        text_encoder: SbV2TextEncoder,
        bert: SbV2BertContainer,
        bert_bridge: BertBridge,
        speaker_embed: SpeakerEmbedding,
        style_injector: StyleVectorInjector,
        sdp: SbV2SDP,
        flow: SbV2Flow,
        decoder: SbV2Decoder,
    ) -> Self {
        Self {
            backend: BackendKind::Cpu,
            phonemizer,
            text_encoder,
            bert,
            bert_bridge,
            speaker_embed,
            // Blocker 3: `new` binds the legacy [`SpeakerEmbedding`] path
            // only. A caller that has an [`ExternalSpeakerProjection`]
            // for the real ckpt attaches it via
            // [`with_external_speaker_projection`](Self::with_external_speaker_projection)
            // rather than growing this constructor to 10 arguments.
            speaker_projection: None,
            style_injector,
            sdp,
            flow,
            decoder,
        }
    }

    /// Attaches an [`ExternalSpeakerProjection`] to this model — Blocker
    /// 3's real-ckpt speaker-conditioning path.
    ///
    /// A synthetic model built via [`new`](Self::new),
    /// [`synthetic_for_test`](Self::synthetic_for_test) or
    /// [`synthetic_for_test_e2e`](Self::synthetic_for_test_e2e) has
    /// `speaker_projection = None`, so
    /// [`synthesize`](Self::synthesize) routes speaker conditioning
    /// through the legacy [`SpeakerEmbedding::lookup`] path (backward
    /// compat with every pre-Blocker-3 synthetic test — see
    /// [`synthesize`](Self::synthesize)'s step 5 dispatch table). This
    /// setter attaches the projection so subsequent `synthesize` calls
    /// route through the real-ckpt path instead. Returns `self` (moved)
    /// so callers can chain it after a constructor.
    ///
    /// Builder-style setter (not a growth of [`new`](Self::new)'s
    /// argument list) so real-ckpt use sites can opt in without
    /// disturbing every synthetic test's existing 9-argument
    /// construction.
    #[must_use]
    pub fn with_external_speaker_projection(mut self, proj: ExternalSpeakerProjection) -> Self {
        self.speaker_projection = Some(proj);
        self
    }

    /// Selects the backend for subsequent synthesis. The complete learned-op
    /// registry is preflighted at synthesis entry, so unsupported devices fail
    /// explicitly before any model operation runs.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Returns the selected synthesis backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Changes the backend for the next synthesis. Callers that need to
    /// compare CPU and Metal on one loaded checkpoint can reuse the same
    /// immutable weight bundle and switch this selector between runs.
    pub fn set_backend(&mut self, backend: BackendKind) {
        self.backend = backend;
    }

    /// Returns a read-only handle to the loaded [`ExternalSpeakerProjection`]
    /// (Blocker 3), or `None` if this model routes speaker conditioning
    /// through the legacy [`SpeakerEmbedding::lookup`] path only (see
    /// [`synthesize`](Self::synthesize)'s step-5 dispatch table).
    ///
    /// The projection is bound in one of two ways:
    /// - [`with_external_speaker_projection`](Self::with_external_speaker_projection)
    ///   attaches a caller-built one (synthetic tests, custom fine-tunes).
    /// - [`from_gguf`](Self::from_gguf) binds the pair when the loaded
    ///   `main` GGUF carries the
    ///   `sbv2.text_encoder.spk_emb_linear.{weight,bias}` tensors (the real
    ///   SBV2 v2 base ckpt's `enc_p.encoder.spk_emb_linear.*` renamed by
    ///   the Blocker 1 converter mapping table).
    ///
    /// A read-only handle (rather than exposing `speaker_projection`
    /// directly as `pub`) so callers cannot mutate the projection weights
    /// post-load — matches every other SBV2 field's encapsulation. Used
    /// by `parity_sbv2_real.rs` to assert-loudly that a real-ckpt load
    /// bound the projection (otherwise a converter regression that
    /// silently drops the pair would only surface as a subtle waveform
    /// drift; FR-EX-08 prefers loud-fail at load time).
    pub fn speaker_projection(&self) -> Option<&ExternalSpeakerProjection> {
        self.speaker_projection.as_ref()
    }

    /// Runs the real SDP conditioner only, for the VAST-generated parity
    /// fixture in `tests/fixtures/sbv2/sdp_body_seed*_T*.f32.bin`.
    ///
    /// This deliberately exposes no mutable SDP state and skips the random
    /// latent/flow-inverse portion of [`SbV2SDP::sample`]. It exists so the
    /// integration parity test can compare the deterministic body against the
    /// independently executed upstream MIT VITS implementation. It is not a
    /// general synthesis entry point.
    #[doc(hidden)]
    pub fn sdp_body_for_parity(
        &self,
        hidden_row_major: &[f32],
        text_seq_len: usize,
        speaker_embedding: &[f32],
    ) -> Vec<f32> {
        assert_eq!(
            hidden_row_major.len(),
            text_seq_len * self.sdp.d_hidden(),
            "SBV2 SDP body parity input must be [T, d_hidden] row-major"
        );
        assert_eq!(
            speaker_embedding.len(),
            self.sdp.gin(),
            "SBV2 SDP body parity speaker input must have gin elements"
        );
        self.sdp
            .body(hidden_row_major, text_seq_len, speaker_embedding)
    }

    /// Test-only constructor: assembles a full pipeline out of tiny,
    /// deterministic, "shaped-like-the-real-thing" components — this proves
    /// the Task 23 *wiring*, not any trained voice's quality (no real
    /// checkpoint is involved; that lands with the Task 24-27 converter).
    ///
    /// Dimensional choices, all deliberately tiny for fast tests:
    /// `d_model = d_z = d_speaker = 8` (shared across the text encoder, SDP,
    /// flow latent and speaker embedding — [`synthesize`](Self::synthesize)'s
    /// broadcast-add and the flow/decoder hand-off both require these to
    /// line up, see the struct doc and [`synthesize`](Self::synthesize)'s
    /// doc), `d_bert = 8`, `d_style = 4`, `n_tones = 3`, `n_speakers = 2`.
    /// `n_vocab = 256`, **not** the `100` a first cut might reach for: it
    /// must clear every id [`SbV2Phonemizer::synthetic_for_test`]'s
    /// synthetic char mapping can emit, and the EN branch alone reaches id
    /// `226` (`200 + 26`, the trailing space slot of
    /// `"abcdefghijklmnopqrstuvwxyz "`) — `100` would make
    /// [`SbV2TextEncoder::forward`]'s `phoneme_ids < n_vocab` debug-assert
    /// fail on any EN test input.
    ///
    /// The duration predictor's flow stack and the acoustic flow's coupling
    /// stack are both left empty (each type's own documented no-op
    /// precedent — see `duration.rs`'s and `flow.rs`'s module docs): with an
    /// empty SDP flow stack and `noise_scale_w == 0.0` (as every test below
    /// passes), every predicted duration is exactly `1`, so `mel_seq_len`
    /// equals the phoneme count, keeping test assertions simple and exact.
    #[doc(hidden)]
    pub fn synthetic_for_test() -> Self {
        const D_MODEL: usize = 8;
        const D_BERT: usize = 8;
        const D_STYLE: usize = 4;
        const N_VOCAB: usize = 256;
        const N_TONES: usize = 3;
        const N_SPEAKERS: usize = 2;

        let phonemizer = SbV2Phonemizer::synthetic_for_test();

        let text_encoder = SbV2TextEncoder::from_weights(
            (0..N_VOCAB * D_MODEL)
                .map(|i| ((i as f32) * 0.001).sin() * 0.05)
                .collect(),
            (0..N_TONES * D_MODEL)
                .map(|i| ((i as f32) * 0.01).cos() * 0.02)
                .collect(),
            // language_embed [N_LANGUAGES=3, D_MODEL]: all-zero identity
            // (`synthetic_for_test`'s existing convention for the removed
            // `wb_embed` — the additive contribution is the additive
            // identity, so downstream tests' exact-length /
            // byte-identical-PCM assertions carry over unchanged).
            vec![0.0; N_LANGUAGES * D_MODEL],
            Vec::new(), // empty transformer stack — SbV2TextEncoder's own exercised no-op precedent
            D_MODEL,
            N_VOCAB,
            N_TONES,
        );

        // A minimal 4-entry SentencePiece table (just the required special
        // tokens — see `SbertTokenizer::from_pieces_for_test`'s doc) is
        // sufficient: any input text that matches none of its pieces still
        // tokenizes via `encode`'s per-byte `unk_id` fallback, producing a
        // non-empty id sequence for any non-empty text. Shared by both
        // languages (this scaffold has no real per-language vocabulary yet).
        let tokenizer_pieces = vec![
            ("<pad>".to_string(), 0.0),
            ("<unk>".to_string(), 0.0),
            ("<s>".to_string(), 0.0),
            ("</s>".to_string(), 0.0),
        ];
        let ja_tokenizer = SbertTokenizer::from_pieces_for_test(tokenizer_pieces.clone());
        let en_tokenizer = SbertTokenizer::from_pieces_for_test(tokenizer_pieces);

        let bert = SbV2BertContainer {
            ja_tokenizer,
            en_tokenizer,
            // Proven-shape tuple (n_layers=2, d_model=8, n_heads=2, vocab=16,
            // n_pos_buckets=512) — `vokra-bert`'s own
            // `encoder_stack_forward_shape` test exercises this exact tuple.
            ja: DebertaV2Encoder::synthetic_for_test(2, D_BERT, 2, 16, 512),
            en: DebertaV3Encoder::synthetic_for_test(2, D_BERT, 2, 16, 512),
            // WP-19: `synthetic_for_test` leaves ZH unwired — a caller that
            // needs a ZH branch uses `synthetic_with_zh_for_test` instead.
            zh: None,
            zh_tokenizer: None,
        };

        let bert_bridge = BertBridge::from_conv(
            (0..D_MODEL * D_BERT)
                .map(|i| ((i as f32) * 0.02).sin() * 0.03)
                .collect(),
            vec![0.0; D_MODEL],
            D_BERT,
            D_MODEL, // == text_encoder.d_model(): the struct doc's dimensional assumption
        );

        let speaker_embed = SpeakerEmbedding::from_table(
            (0..N_SPEAKERS * D_MODEL)
                .map(|i| ((i as f32) * 0.05).cos() * 0.1)
                .collect(),
            N_SPEAKERS,
            D_MODEL, // == text_encoder.d_model(): the struct doc's dimensional assumption
        );

        let style_injector = StyleVectorInjector::from_projections(
            vec![0.0; D_MODEL * D_STYLE],
            vec![0.0; D_MODEL * D_STYLE],
            D_STYLE,
            D_MODEL,
        );

        // Empty-flows SDP (post-Blocker-2c primitive stack): zero-weight
        // `pre`/`proj`/`cond`/`convs`, identity `ElementwiseAffine`, and
        // `flows = vec![]` — combined with `noise_scale_w == 0.0` (as every
        // caller of this factory passes), every predicted duration is
        // exactly `1`. `gin == D_MODEL` on the synthetic path (matches
        // `d_speaker`; see `SbV2SDP::empty`'s doc).
        let sdp = SbV2SDP::empty(D_MODEL, D_MODEL);

        let flow = SbV2Flow::from_layers(Vec::new(), D_MODEL); // empty coupling stack: z = mel_hidden unchanged

        let attrs = HifiGanAttrs {
            n_mels: D_MODEL, // the flow's d_z feeds the decoder's n_mels directly — see synthesize's doc
            initial_channel: 6,
            upsample_rates: vec![2, 2],
            upsample_kernel_sizes: vec![4, 4], // kernel = 2*stride: exact upsample length (SbV2Decoder's doc)
            resblock_kernel_sizes: vec![3],
            resblock_dilation_sizes: vec![vec![1]],
            sample_rate: 44_100,
            leaky_relu_slope: 0.1,
            // Synthetic weights below build single-conv layers (no c2)
            // to keep synthesize()-shape smoke tests unchanged from
            // pre-Wave-2 behavior. A future synthesize e2e that swaps
            // in real ckpt weights should build V1 attrs — the real
            // SBV2 v2 checkpoint uses ResBlock1.
            res_block_type: ResBlockType::V2,
        };
        let weights = synthetic_hifigan_weights(&attrs);
        let sample_rate = attrs.sample_rate;
        let decoder = SbV2Decoder::new(weights, attrs, HifiGanConfig::fp32(), sample_rate);

        Self::new(
            phonemizer,
            text_encoder,
            bert,
            bert_bridge,
            speaker_embed,
            style_injector,
            sdp,
            flow,
            decoder,
        )
    }

    /// WP-19 test-only factory: like
    /// [`synthetic_for_test`](Self::synthetic_for_test) but with the
    /// [`SbV2Phonemizer`] slot swappable by the caller. The other 8
    /// components (text encoder, BERT container, bridge, speaker, style,
    /// SDP, flow, decoder) are byte-identical to
    /// [`synthetic_for_test`](Self::synthetic_for_test).
    ///
    /// Used by `tests/sbv2_model_synthetic.rs`'s WP-19 tests to swap in a
    /// [`SbV2Phonemizer::from_fixture`]-backed phonemizer that supplies
    /// pre-computed ZH phoneme ids — the production ZH G2P is a
    /// piper-plus delegation (WP-18) that lives in an excluded workspace,
    /// so this crate's tests use a `PhonemizeFixture` to reach the ZH
    /// BERT dispatch arm without wiring a real 8-language G2P.
    ///
    /// `#[doc(hidden)]` — for `tests/sbv2_*.rs` only; do not use in
    /// production code (a production caller wires
    /// [`from_gguf_with_phonemizer`](Self::from_gguf_with_phonemizer) or
    /// [`from_gguf_with_zh_bert`](Self::from_gguf_with_zh_bert)).
    #[doc(hidden)]
    pub fn synthetic_for_test_with_phonemizer(phonemizer: SbV2Phonemizer) -> Self {
        let mut model = Self::synthetic_for_test();
        model.phonemizer = phonemizer;
        model
    }

    /// WP-19 test-only factory: like
    /// [`synthetic_for_test`](Self::synthetic_for_test) but with the ZH
    /// branch of [`SbV2BertContainer`] wired (both
    /// [`SbV2BertContainer::zh`] and [`SbV2BertContainer::zh_tokenizer`]
    /// are `Some`). Uses the caller-supplied [`PhonemizeFixture`] as the
    /// G2P (WP-18 production ZH G2P is a piper-plus-side delegation the
    /// vokra-models crate cannot exercise standalone; the fixture pattern
    /// is what `parity_sbv2_real.rs` also uses for the same reason).
    ///
    /// The synthetic ZH BERT is a tiny 1-layer [`BertBaseEncoder`] and a
    /// 6-entry WordPiece vocab (`[PAD]`/`[UNK]`/`[CLS]`/`[SEP]` + two
    /// content tokens `"你"` / `"好"`) — enough that the tokenize step
    /// segments any input into at least one non-special id per
    /// `bert_input_text` (unmatched characters fall to `[UNK]`), matching
    /// the JA/EN synthetic tokenizers' "any input tokenizes to something"
    /// contract. Dimensions match [`synthetic_for_test`] byte-for-byte
    /// (`D_BERT = 8`, `d_model = 8`, `type_vocab = 2`), so the shared
    /// [`BertBridge`] (`D_BERT` → `d_model` projection) is directly
    /// reusable — a mismatch would fire the loader's `d_bert` guard on
    /// the real path.
    ///
    /// `#[doc(hidden)]` — for `tests/sbv2_*.rs` only; do not use in
    /// production code.
    #[doc(hidden)]
    pub fn synthetic_with_zh_for_test(fixture: PhonemizeFixture) -> Self {
        // Base the model on `synthetic_for_test` so every non-ZH
        // component (text encoder, JA/EN BERT, bridge, speaker, style,
        // SDP, flow, decoder) is byte-identical to the reference
        // synthetic pipeline — the only differences are the phonemizer
        // (fixture-backed for ZH) and the two ZH BERT slots
        // (`Some(BertBaseEncoder)` / `Some(BertWordpieceTokenizer)`).
        let mut model = Self::synthetic_for_test();

        // ZH BERT: tiny plain-BERT encoder shaped to match
        // `synthetic_for_test`'s `D_BERT = 8` and `type_vocab = 2` so the
        // shared `BertBridge` (`D_BERT` → `d_model` projection built by
        // `synthetic_for_test`) accepts its output without shape
        // reconciliation.
        const D_BERT: usize = 8;
        const ZH_VOCAB: usize = 6;
        let cfg = vokra_bert::bert_base::BertConfig {
            vocab_size: ZH_VOCAB,
            hidden_size: D_BERT,
            num_hidden_layers: 1,
            num_attention_heads: 2, // 8 / 2 = 4, valid head_dim
            intermediate_size: 32,
            max_position_embeddings: 32,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
        };
        let zh_enc = BertBaseEncoder::synthetic_for_test(&cfg);

        // ZH tokenizer: 6-entry vocab (`[PAD]/[UNK]/[CLS]/[SEP]` + `"你"` +
        // `"好"`). `from_vocab` defaults to `OovPolicy::Unk`, so any
        // codepoint outside `{你, 好}` maps to `[UNK]` — every input
        // produces a non-empty id sequence (`[CLS] ... [SEP]`), matching
        // the JA/EN synthetic tokenizers' contract.
        let vocab: Vec<String> = vec![
            "[PAD]".to_string(),
            "[UNK]".to_string(),
            "[CLS]".to_string(),
            "[SEP]".to_string(),
            "你".to_string(),
            "好".to_string(),
        ];
        let zh_tok = BertWordpieceTokenizer::from_vocab(
            vocab, /* unk */ 1, /* cls */ 2, /* sep */ 3, /* pad */ 0,
        )
        .expect("synthetic ZH wordpiece vocab is well-formed");

        model.bert.zh = Some(zh_enc);
        model.bert.zh_tokenizer = Some(zh_tok);
        model.phonemizer = SbV2Phonemizer::from_fixture(fixture);
        model
    }

    /// WP-23 test-only factory: a shape-preserving twin of
    /// [`synthetic_for_test`](Self::synthetic_for_test) whose
    /// [`StyleVectorInjector`] carries **nonzero** projection weights so
    /// tests can prove that
    /// `<SbV2Model as vokra_core::TtsEngine>::synthesize` threads
    /// [`vokra_core::SynthesisRequest::style_vec`] into the pipeline
    /// (via the `SbV2SynthRequest::style_vec` route). The original
    /// [`synthetic_for_test`](Self::synthetic_for_test)'s style
    /// projections are all-zero (see that factory's own body: `vec![0.0;
    /// D_MODEL * D_STYLE]` for both `proj_scale` and `proj_bias`), which
    /// makes any `style_vec` map to the identity injection — so the
    /// original factory cannot observe a `None` vs `Some(nonzero)` PCM
    /// difference. This factory swaps in projections generated by
    /// `sin(i * 0.03) * 0.05` / `cos(i * 0.04) * 0.05` (small enough to
    /// keep the pipeline numerically well-behaved, nonzero enough that
    /// AdaIN's `hidden * (1 + scale) + bias` visibly perturbs downstream
    /// mel-hidden and thus the decoder's PCM). Every other component
    /// (text encoder, BERT, bridge, speaker table, SDP, flow, decoder)
    /// is byte-identical to
    /// [`synthetic_for_test`](Self::synthetic_for_test) — the only
    /// difference is the style projections, so tests over this factory
    /// isolate the `style_vec` threading and nothing else.
    ///
    /// `#[doc(hidden)]` — this exists purely for
    /// `tests/sbv2_tts_engine_extension.rs`; do not use in production
    /// code.
    #[doc(hidden)]
    pub fn synthetic_for_test_with_nonzero_style() -> Self {
        // Dimensions must mirror `synthetic_for_test` bit-for-bit — see
        // that method's doc for why.
        const D_MODEL: usize = 8;
        const D_BERT: usize = 8;
        const D_STYLE: usize = 4;
        const N_VOCAB: usize = 256;
        const N_TONES: usize = 3;
        const N_SPEAKERS: usize = 2;

        let phonemizer = SbV2Phonemizer::synthetic_for_test();

        let text_encoder = SbV2TextEncoder::from_weights(
            (0..N_VOCAB * D_MODEL)
                .map(|i| ((i as f32) * 0.001).sin() * 0.05)
                .collect(),
            (0..N_TONES * D_MODEL)
                .map(|i| ((i as f32) * 0.01).cos() * 0.02)
                .collect(),
            vec![0.0; 2 * D_MODEL],
            Vec::new(),
            D_MODEL,
            N_VOCAB,
            N_TONES,
        );

        let tokenizer_pieces = vec![
            ("<pad>".to_string(), 0.0),
            ("<unk>".to_string(), 0.0),
            ("<s>".to_string(), 0.0),
            ("</s>".to_string(), 0.0),
        ];
        let ja_tokenizer = SbertTokenizer::from_pieces_for_test(tokenizer_pieces.clone());
        let en_tokenizer = SbertTokenizer::from_pieces_for_test(tokenizer_pieces);

        let bert = SbV2BertContainer {
            ja_tokenizer,
            en_tokenizer,
            ja: DebertaV2Encoder::synthetic_for_test(2, D_BERT, 2, 16, 512),
            en: DebertaV3Encoder::synthetic_for_test(2, D_BERT, 2, 16, 512),
            // WP-19: this synthetic constructor does not exercise the ZH
            // branch. `synthetic_with_zh_for_test` is the sibling that
            // wires ZH; every other synthetic constructor mirrors the
            // legacy 2-language shape (both `None`).
            zh: None,
            zh_tokenizer: None,
        };

        let bert_bridge = BertBridge::from_conv(
            (0..D_MODEL * D_BERT)
                .map(|i| ((i as f32) * 0.02).sin() * 0.03)
                .collect(),
            vec![0.0; D_MODEL],
            D_BERT,
            D_MODEL,
        );

        let speaker_embed = SpeakerEmbedding::from_table(
            (0..N_SPEAKERS * D_MODEL)
                .map(|i| ((i as f32) * 0.05).cos() * 0.1)
                .collect(),
            N_SPEAKERS,
            D_MODEL,
        );

        // WP-23 only-difference: nonzero style projections so `style_vec`
        // has observable PCM effect. `sin(_)*0.05` / `cos(_)*0.05` keep
        // the magnitudes small enough that the pipeline stays
        // numerically well-behaved (finite PCM, no clipping) — the
        // AdaIN update `hidden * (1 + scale) + bias` scales the hidden
        // channels by `1 + O(0.05 * ||style_vec||)` per position.
        let style_injector = StyleVectorInjector::from_projections(
            (0..D_MODEL * D_STYLE)
                .map(|i| ((i as f32) * 0.03).sin() * 0.05)
                .collect(),
            (0..D_MODEL * D_STYLE)
                .map(|i| ((i as f32) * 0.04).cos() * 0.05)
                .collect(),
            D_STYLE,
            D_MODEL,
        );

        // Use the canonical zero-weight identity SDP — same shape as
        // synthetic_for_test (line 483); WP-23's threading test only cares
        // about style-vec propagation, not SDP behavior.
        let sdp = SbV2SDP::empty(D_MODEL, D_MODEL);

        let flow = SbV2Flow::from_layers(Vec::new(), D_MODEL);

        let attrs = HifiGanAttrs {
            n_mels: D_MODEL,
            initial_channel: 6,
            upsample_rates: vec![2, 2],
            upsample_kernel_sizes: vec![4, 4],
            resblock_kernel_sizes: vec![3],
            resblock_dilation_sizes: vec![vec![1]],
            sample_rate: 44_100,
            leaky_relu_slope: 0.1,
            res_block_type: vokra_core::ir::graph::ResBlockType::V1,
        };
        let weights = synthetic_hifigan_weights(&attrs);
        let sample_rate = attrs.sample_rate;
        let decoder = SbV2Decoder::new(weights, attrs, HifiGanConfig::fp32(), sample_rate);

        Self::new(
            phonemizer,
            text_encoder,
            bert,
            bert_bridge,
            speaker_embed,
            style_injector,
            sdp,
            flow,
            decoder,
        )
    }

    /// Test-only e2e-scale constructor (Task 42,
    /// `tests/sbv2_e2e_synthetic.rs`): a **separate** factory from
    /// [`synthetic_for_test`](Self::synthetic_for_test) — that method's exact
    /// 12-sample output is a Task 23/27 contract
    /// (`tests/parity_sbv2_synthetic.rs`'s `synthetic_shape_invariants_hold`
    /// pins it precisely), so this constructor duplicates rather than
    /// mutates it, guaranteeing zero regression risk. This one instead
    /// assembles a pipeline shaped to clear "sounds like real audio" bars —
    /// more than 1 second of PCM at 44.1 kHz, every sample finite, and a
    /// non-silent peak amplitude — for both JA and EN input text.
    ///
    /// Two shape changes from [`synthetic_for_test`](Self::synthetic_for_test):
    ///
    /// 1. **Decoder upsample ladder** — SBV2 JP-Extra's real
    ///    `upsample_rates = [8, 8, 2, 2]` / `upsample_kernel_sizes = [16,
    ///    16, 4, 4]` (256x total upsample, *exact* — no rounding slack, per
    ///    `decoder.rs`'s module doc and `tests/sbv2_decoder.rs`'s
    ///    `jp_extra_attrs`), replacing `synthetic_for_test`'s toy 2-stage
    ///    `[2, 2]` ladder (4x).
    /// 2. **SDP** — an empty-flows SDP whose slot-0
    ///    [`duration::ElementwiseAffine`] has `m = [-ln(FIXED_DURATION),
    ///    0]` and `logs = [0, 0]`, replacing `synthetic_for_test`'s pure
    ///    zero-weight identity. Every request this factory is built for
    ///    uses `noise_scale_w == 0.0`, which makes `SbV2SDP::sample`'s
    ///    Gaussian latent draw exactly `0.0` at every phoneme position
    ///    regardless of `seed`/RNG state (`SbV2SDP::sample`'s doc), and
    ///    with the flow stack empty the final `flip` and the `EA.reverse`
    ///    reduce to `(0 - (-ln FD)) * exp(0) = ln FD` on channel 0 (and
    ///    `0` on channel 1). `sample`'s `duration =
    ///    logw.exp().ceil().max(1.0)` then returns exactly `FIXED_DURATION`
    ///    at every position, independent of `hidden`, `g`, or RNG — a
    ///    closed-form, weight-only way to reach e2e-scale mel lengths
    ///    without touching `SbV2Model::synthesize` or `SbV2SDP` itself.
    ///    (Pre-Blocker-2c this same trick lived on a scalar-affine
    ///    `SbV2CouplingLayer`, now removed; the EA form is architecturally
    ///    equivalent — see `duration.rs`'s post-Blocker-2c module doc.)
    ///
    ///    `FIXED_DURATION = 40.0`: JA's "こんにちは" (5 phonemes, all
    ///    present in [`SbV2Phonemizer::synthetic_for_test`]'s char map —
    ///    its table's tail literally spells "...わをんこんにちは") reaches
    ///    `5 * 40 == 200` mel frames → `200 * 256 == 51,200` samples; EN's
    ///    "hello world" (10 phonemes — [`g2p`]'s char-mapping path skips
    ///    the space, see `g2p.rs`'s `phonemize_en_char_mapping`) reaches
    ///    `10 * 40 == 400` mel frames → `400 * 256 == 102,400` samples.
    ///    Both clear the `> 44,100` (1s at 44.1 kHz) bar with margin.
    ///
    /// Every synthetic weight magnitude coefficient is also bumped from
    /// `synthetic_for_test`'s `~0.05-0.1` to `0.5` (the decoder's
    /// conv/upsample/MRF weights and biases, and the speaker embedding
    /// table) — cheap insurance against the accumulated HiFi-GAN forward
    /// pass underflowing toward silence through 4 sequential upsample/MRF
    /// stages of small-magnitude convolutions, per this task's brief.
    #[doc(hidden)]
    pub fn synthetic_for_test_e2e() -> Self {
        const D_MODEL: usize = 8;
        const D_BERT: usize = 8;
        const D_STYLE: usize = 4;
        const N_VOCAB: usize = 256;
        const N_TONES: usize = 3;
        const N_SPEAKERS: usize = 2;
        // See this fn's doc point 2 for the full derivation of why this
        // value is the exact per-phoneme duration every request produces.
        const FIXED_DURATION: f32 = 40.0;

        let phonemizer = SbV2Phonemizer::synthetic_for_test();

        let text_encoder = SbV2TextEncoder::from_weights(
            (0..N_VOCAB * D_MODEL)
                .map(|i| ((i as f32) * 0.001).sin() * 0.05)
                .collect(),
            (0..N_TONES * D_MODEL)
                .map(|i| ((i as f32) * 0.01).cos() * 0.02)
                .collect(),
            // language_embed [N_LANGUAGES=3, D_MODEL]: all-zero identity —
            // same rationale as `synthetic_for_test` above.
            vec![0.0; N_LANGUAGES * D_MODEL],
            Vec::new(), // empty transformer stack — SbV2TextEncoder's own exercised no-op precedent
            D_MODEL,
            N_VOCAB,
            N_TONES,
        );

        // Same minimal 4-entry SentencePiece table as `synthetic_for_test`
        // (see that method's doc) — any input text still tokenizes via
        // `encode`'s per-byte `unk_id` fallback.
        let tokenizer_pieces = vec![
            ("<pad>".to_string(), 0.0),
            ("<unk>".to_string(), 0.0),
            ("<s>".to_string(), 0.0),
            ("</s>".to_string(), 0.0),
        ];
        let ja_tokenizer = SbertTokenizer::from_pieces_for_test(tokenizer_pieces.clone());
        let en_tokenizer = SbertTokenizer::from_pieces_for_test(tokenizer_pieces);

        let bert = SbV2BertContainer {
            ja_tokenizer,
            en_tokenizer,
            ja: DebertaV2Encoder::synthetic_for_test(2, D_BERT, 2, 16, 512),
            en: DebertaV3Encoder::synthetic_for_test(2, D_BERT, 2, 16, 512),
            // WP-19: this synthetic constructor does not exercise the ZH
            // branch. `synthetic_with_zh_for_test` is the sibling that
            // wires ZH; every other synthetic constructor mirrors the
            // legacy 2-language shape (both `None`).
            zh: None,
            zh_tokenizer: None,
        };

        let bert_bridge = BertBridge::from_conv(
            (0..D_MODEL * D_BERT)
                .map(|i| ((i as f32) * 0.02).sin() * 0.03)
                .collect(),
            vec![0.0; D_MODEL],
            D_BERT,
            D_MODEL,
        );

        // Bumped 0.1 -> 0.5 (this fn's doc's magnitude-bump paragraph).
        let speaker_embed = SpeakerEmbedding::from_table(
            (0..N_SPEAKERS * D_MODEL)
                .map(|i| ((i as f32) * 0.05).cos() * 0.5)
                .collect(),
            N_SPEAKERS,
            D_MODEL,
        );

        let style_injector = StyleVectorInjector::from_projections(
            vec![0.0; D_MODEL * D_STYLE],
            vec![0.0; D_MODEL * D_STYLE],
            D_STYLE,
            D_MODEL,
        );

        // Post-Blocker-2c empty-flows SDP whose `ElementwiseAffine` at slot
        // 0 has `m = [-ln(FIXED_DURATION), 0]` and `logs = [0, 0]`. With
        // `noise_scale_w == 0.0` (as every caller of this factory passes),
        // the latent `z = [0, 0]` after the final `flip2` (a no-op with an
        // empty flow stack) becomes `z[0] = (0 - (-ln FD)) * exp(0) = ln
        // FD` and `z[1] = 0` after EA reverse, so every duration is
        // `exp(ln FD).ceil().max(1) = FIXED_DURATION` regardless of
        // `hidden`, `g`, or RNG state (`SbV2SDP::sample`'s doc). This
        // replaces the pre-Blocker-2c scalar-affine `SbV2CouplingLayer`
        // trick with the equivalent EA-based one that fits the real
        // primitive stack.
        let ea = duration::ElementwiseAffine::from_weights(
            vec![-FIXED_DURATION.ln(), 0.0],
            vec![0.0, 0.0],
        );
        let sdp = duration::SbV2SDP::from_weights(
            D_MODEL,
            D_MODEL,
            vec![0.0; D_MODEL * D_MODEL],
            vec![0.0; D_MODEL],
            duration::DDSConv::zero(D_MODEL, duration::DP_CONV_LAYERS, duration::DP_KERNEL),
            vec![0.0; D_MODEL * D_MODEL],
            vec![0.0; D_MODEL],
            vec![0.0; D_MODEL * D_MODEL],
            vec![0.0; D_MODEL],
            ea,
            Vec::new(),
        );

        let flow = SbV2Flow::from_layers(Vec::new(), D_MODEL); // empty coupling stack: z = mel_hidden unchanged

        let attrs = HifiGanAttrs {
            n_mels: D_MODEL, // the flow's d_z feeds the decoder's n_mels directly — see synthesize's doc
            initial_channel: 8,
            upsample_rates: vec![8, 8, 2, 2], // SBV2 JP-Extra base config (decoder.rs's module doc)
            upsample_kernel_sizes: vec![16, 16, 4, 4], // kernel = 2*stride: exact upsample length
            resblock_kernel_sizes: vec![3],
            resblock_dilation_sizes: vec![vec![1]],
            sample_rate: 44_100,
            leaky_relu_slope: 0.1,
            // See the sibling synthetic constructor above: synthetic
            // weights builder emits single-conv per layer (no c2).
            res_block_type: ResBlockType::V2,
        };
        let weights = synthetic_hifigan_weights_e2e(&attrs);
        let sample_rate = attrs.sample_rate;
        let decoder = SbV2Decoder::new(weights, attrs, HifiGanConfig::fp32(), sample_rate);

        Self::new(
            phonemizer,
            text_encoder,
            bert,
            bert_bridge,
            speaker_embed,
            style_injector,
            sdp,
            flow,
            decoder,
        )
    }

    /// Runs the full SBV2 forward pass: G2P → text encoder → BERT (+
    /// bridge) → speaker + style conditioning → stochastic duration
    /// prediction → length regulation → normalizing flow → HiFi-GAN decoder
    /// → PCM. See `docs/superpowers/specs/2026-07-26-sbv2-v2-design.md` §7's
    /// pipeline diagram for the canonical order this follows.
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::InvalidArgument`] if `req.speed` is not
    /// positive, if G2P produces no phonemes for `req.text`, if the BERT
    /// tokenizer produces no tokens for `req.text`, or if `req.speaker_id`
    /// is out of range ([`SpeakerEmbedding::lookup`]'s own error). When
    /// wired through [`from_piper_g2p`](SbV2Phonemizer::from_piper_g2p),
    /// also propagates any error the injected G2P returns.
    pub fn synthesize(&self, req: &SbV2SynthRequest) -> Result<SynthesizedAudio> {
        // Full pipeline; discards intermediates. See
        // [`Self::synthesize_with_intermediates`] for the accessor variant
        // that returns each per-stage tensor alongside the final PCM
        // (Wave-4 INTERMEDIATE-ACCESSORS).
        self.synthesize_with_intermediates(req).map(|(pcm, _)| pcm)
    }

    /// Same forward pipeline as [`Self::synthesize`], but returns every
    /// per-stage intermediate the Python reference dumper
    /// (`tools/parity/sbv2_dump_reference.py`) writes to `reference_dump/*.bin`.
    ///
    /// Added for Wave-4 `INTERMEDIATE-ACCESSORS` so
    /// `parity_sbv2_real.rs` can diff per-stage tensors against Python
    /// reference `.bin` fixtures instead of only comparing the final
    /// waveform. Every field of [`SbV2Intermediates`] corresponds to one
    /// entry in the manifest's `tensors[]` array (design doc §10, dumper's
    /// `TENSOR_SCHEMA`) — see [`SbV2Intermediates`]'s field docs for the
    /// exact shape and dumper filename each maps to.
    ///
    /// # Errors
    ///
    /// Same as [`Self::synthesize`].
    pub fn synthesize_with_intermediates(
        &self,
        req: &SbV2SynthRequest,
    ) -> Result<(SynthesizedAudio, SbV2Intermediates)> {
        if req.speed <= 0.0 {
            return Err(VokraError::InvalidArgument(format!(
                "SbV2Model::synthesize: req.speed must be positive, got {}",
                req.speed
            )));
        }

        // Preflight the entire learned-op set before G2P or tensor work. This
        // makes a requested Metal path fail closed rather than partially
        // executing and silently falling back to CPU.
        let compute = if self.backend == BackendKind::Cpu {
            None
        } else {
            Some(Compute::for_backend(self.backend, SBV2_HOT_OPS)?)
        };

        // 1. G2P
        let phon = self.phonemizer.phonemize(&req.text, req.language)?;
        if phon.phoneme_ids.is_empty() {
            return Err(VokraError::InvalidArgument(
                "SbV2Model::synthesize: text produced no phonemes".to_string(),
            ));
        }

        // 2. Text encoder — `language_id` is derived from `req.language`
        // via [`Language::language_id`], which pins the row of
        // `SbV2TextEncoder::language_embed` that gets broadcast-added to
        // every position. See `SbV2TextEncoder::forward`'s `language_id`
        // doc for the tentative JA=0/EN=1/ZH=2 row-ordering convention.
        //
        // Wave-4 INTERMEDIATE-ACCESSORS: capture both the pre-transformer
        // sum (`phoneme_embed`) and the post-transformer hidden state
        // (`text_hidden`) so parity harnesses can diff each independently
        // against the Python dumper's `phoneme_embed.bin` and
        // `text_hidden.bin`. The full pipeline consumes only `text_hidden`
        // downstream; `phoneme_embed` is a pure snapshot of the sum
        // buffer taken before the in-place transformer stack.
        let (phoneme_embed, text_hidden) = match compute.as_ref() {
            Some(compute) => self.text_encoder.forward_with_compute(
                compute,
                &phon.phoneme_ids,
                &phon.tones,
                req.language.language_id(),
            )?,
            None => self.text_encoder.forward_with_embed(
                &phon.phoneme_ids,
                &phon.tones,
                req.language.language_id(),
            ),
        };

        // 3. BERT (per-language). ZH is optional (WP-19): the ZH arm below
        // routes through [`SbV2BertContainer::zh_tokenizer`] +
        // [`SbV2BertContainer::zh`] when both are wired (`Some`), and
        // fails loudly with [`VokraError::NotImplemented`] when either is
        // `None` — never a silent fall-through to JA/EN (FR-EX-08).
        //
        // Two paths make this reachable at runtime:
        // (a) the ZH G2P is wired via WP-18's [`SbV2Phonemizer::with_zh_g2p`]
        //     (phonemize_zh succeeds instead of returning NotImplemented);
        // (b) a caller feeds a `PhonemizeFixture` that maps a ZH
        //     `(language, text)` entry (see `parity_sbv2_real.rs` for
        //     the fixture pattern used by parity/WP-19 wiring tests).
        //
        // The ZH BERT stack itself is `hfl/chinese-roberta-wwm-ext-large`
        // (`BertForMaskedLM`, NOT DeBERTa; owner decision 2026-08-09) —
        // loaded through the WP-16 `BertBaseEncoder` + WP-17 WordPiece
        // tokenizer (`crates/vokra-bert/src/wordpiece.rs`), wired into
        // [`SbV2BertContainer`] by WP-19's
        // [`SbV2Model::from_gguf_with_zh_bert`]. Callers that only need
        // ZH G2P (fixture builders, WP-19 loader tests) can still call
        // [`SbV2Phonemizer::phonemize`] directly with `Language::ZH`;
        // this fail-closed BERT gate only fires from the full
        // [`synthesize`] pipeline when the loader path did not supply
        // the ZH pair.
        let bert_ids = match req.language {
            Language::JA => self
                .bert
                .ja_tokenizer
                .encode_with_special_tokens(&phon.bert_input_text),
            Language::EN => self
                .bert
                .en_tokenizer
                .encode_with_special_tokens(&phon.bert_input_text),
            Language::ZH => {
                let zh_tok = self
                    .bert
                    .zh_tokenizer
                    .as_ref()
                    .ok_or(VokraError::NotImplemented(
                        "SbV2Model::synthesize: language ZH requested but no ZH BERT \
                         tokenizer is wired on this model (SbV2BertContainer::zh_tokenizer \
                         is None). Load the model via SbV2Model::from_gguf_with_zh_bert \
                         (WP-19) to bind the ZH branch — the pre-WP-19 3-file \
                         SbV2Model::from_gguf leaves ZH unwired on purpose (FR-EX-08 \
                         fail-closed default).",
                    ))?;
                zh_tok.encode(&phon.bert_input_text, true)?
            }
        };
        if bert_ids.is_empty() {
            return Err(VokraError::InvalidArgument(
                "SbV2Model::synthesize: BERT tokenizer produced no tokens for the input text"
                    .to_string(),
            ));
        }
        let bert_hidden = match (req.language, compute.as_ref()) {
            (Language::JA, Some(_)) => {
                deberta_v2_forward_with_backend(&self.bert.ja, &bert_ids, self.backend)?
            }
            (Language::EN, Some(_)) => {
                deberta_v3_forward_with_backend(&self.bert.en, &bert_ids, self.backend)?
            }
            (Language::JA, None) => self.bert.ja.forward(&bert_ids),
            (Language::EN, None) => self.bert.en.forward(&bert_ids),
            (Language::ZH, compute) => {
                let zh_enc = self.bert.zh.as_ref().ok_or(VokraError::NotImplemented(
                    "SbV2Model::synthesize: language ZH requested and ZH tokenizer \
                         is wired but the ZH encoder (SbV2BertContainer::zh) is None — \
                         one-sided ZH wiring is a caller bug (FR-EX-08). Use \
                         SbV2Model::from_gguf_with_zh_bert to load both sides together.",
                ))?;
                match compute {
                    Some(_) => bert_base_forward_with_backend(zh_enc, &bert_ids, self.backend)?,
                    None => zh_enc.forward(&bert_ids, None),
                }
            }
        };

        // 4. BERT bridge — build `hidden_for_flow` (matches Python
        // reference `bert_bridge_out = projected + text_hidden`, used
        // downstream by `length_regulate` at step 7). Keep `text_hidden`
        // pristine so step 6 feeds SDP the raw encoder output — matching
        // `tools/parity/sbv2_dump_reference.py::run_pipeline_body`'s step
        // 10 which calls `sdp.sample(text_hidden_transposed.unsqueeze(0),
        // ...)` unmodified. Bug 4 fix (2026-08-08 handoff:
        // docs/handoff/sbv2-sdp-debug-2026-08-08.md): the pre-fix code
        // accumulated bridge + speaker + style into a shared `hidden`
        // buffer that then fed SDP, which is architecturally wrong for
        // three reasons all matching the Python reference:
        // (a) Python feeds RAW text_hidden to SDP (not text_hidden +
        //     bridge). SDP has its own `.cond(g)` for speaker
        //     conditioning (see SbV2SDP::body — pre → +cond(g) → DDS →
        //     proj), so it does not need speaker broadcast-added at the
        //     call site.
        // (b) Python's length_regulate input is `bert_bridge_out`
        //     (text_hidden + bridge only — NO speaker, NO style). Speaker
        //     enters the flow via per-block `spk_emb_linear` internally.
        // (c) Python's dumper explicitly notes "style is dumped for the
        //     manifest slot but not otherwise mixed here" — style_projected
        //     is NOT broadcast-added into text_hidden.
        // The pre-fix Rust code produced `hidden` values in ±33 (bridge
        // ~±4.5 + spk_emb_linear bias ~±30) that saturated the SDP's
        // RQS spline softmax, producing runaway durations
        // (sum=24229/max=12036 on 8-phoneme "テスト"). Feeding SDP raw
        // text_hidden (±0.9 magnitude, bit-identical to Python
        // reference) restores sane durations (sum≤30/max≤10 range).
        let bridged = match compute.as_ref() {
            Some(compute) => self.bert_bridge.forward_with_compute(
                compute,
                &bert_hidden,
                phon.phoneme_ids.len(),
                bert_ids.len(),
            )?,
            None => self
                .bert_bridge
                .forward(&bert_hidden, phon.phoneme_ids.len(), bert_ids.len()),
        };
        debug_assert_eq!(
            text_hidden.len(),
            bridged.len(),
            "SbV2Model::synthesize: text_encoder hidden width must equal bert_bridge's \
             projected width (BertBridge's d_target must equal SbV2TextEncoder's d_model — \
             see SbV2Model's struct doc)"
        );
        let mut hidden_for_flow = text_hidden.clone();
        for (h, &b) in hidden_for_flow.iter_mut().zip(bridged.iter()) {
            *h += b;
        }
        // Wave-4 INTERMEDIATE-ACCESSORS: snapshot the summed
        // `bert_bridge_out` (dumper name) before the downstream in-place
        // length-regulate + flow. The Python dumper writes
        // `text_hidden + bridge_projected`; this snapshot IS that same
        // sum, so the parity harness can diff against
        // `reference_dump/bert_bridge_out.bin` directly.
        let bert_bridge_out_snapshot = hidden_for_flow.clone();
        // Split BERT hidden into per-language buckets. The active-language
        // path already carries the tensor; the other path is empty (this
        // matches parity-test-side "compare only the active language"
        // convention, and keeps the intermediate struct's byte cost bounded
        // for callers who only diff the active side).
        let bert_hidden_ja_snapshot = if req.language == Language::JA {
            bert_hidden.clone()
        } else {
            Vec::new()
        };
        let bert_hidden_en_snapshot = if req.language == Language::EN {
            bert_hidden.clone()
        } else {
            Vec::new()
        };
        let bert_hidden_zh_snapshot = if req.language == Language::ZH {
            bert_hidden.clone()
        } else {
            Vec::new()
        };

        // 5. Speaker + style — derive `speaker_e_flow` (the raw
        // `[d_speaker]` conditioning vector consumed by SDP's `.cond(g)`
        // at step 6 and by the flow's per-block `spk_emb_linear` at step
        // 8), but do NOT broadcast-add speaker/style projections into
        // `text_hidden` or `hidden_for_flow`. See step 4's Bug 4 fix
        // comment for the primary-source reasoning; the dispatch table
        // below is unchanged except for the omitted broadcast-add loops.
        //
        // The ExternalSpeakerProjection is still called on each real-ckpt
        // path arm so its shape validation (loud FR-EX-08 error on
        // wrong-length caller input, exercised by
        // `sbv2_speaker_external::synthesize_with_wrong_length_embedding_is_invalid_argument`)
        // still fires — the returned `[d_model]` projected vector is
        // then intentionally DISCARDED (its former use was the broadcast
        // add that this fix removes).
        //
        // # SBV2-SPK-EMB-LINEAR-DECISION (2026-08-11, RESOLVED)
        // See: docs/adr/sbv2-spk-emb-linear-decision.md (gitignored local, Task 12)
        //
        // **Decision: (c) INFERENCE NO-OP** — the projected `[d_model=192]`
        // output of `enc_p.encoder.spk_emb_linear.{weight,bias}` is
        // DISCARDED at inference; only its shape-validation side-effect
        // (loud FR-EX-08 error on wrong-length caller input, exercised by
        // `sbv2_speaker_external::synthesize_with_wrong_length_*`) survives.
        //
        // **Rationale** (ADR §3: full evidence chain; NOT REFERENCED: AGPL
        // litagin02/Style-Bert-VITS2, fishaudio/Bert-VITS2):
        // - Vendored p0p4k/vits2_pytorch (MIT) has the canonical mechanism
        //   at attentions.py::Encoder with `cond_layer_idx=2`, but SBV2's
        //   TextEncoder-level `spk_emb_linear` behavior is behind AGPL
        //   red-line — unresolvable from permissive sources alone.
        // - Reference dumper (tools/parity/sbv2_dump_reference.py) also
        //   discards (constructs jaywalnut310/vits VITS1 TextEncoder with
        //   `strict=False`, silently drops the ckpt weight).
        // - T6 baseline parity_sbv2_real: text_hidden Δ 5.51e-7 at atol
        //   0.01 = float noise floor — matches only because BOTH sides
        //   discard consistently.
        //
        // **SPEAKER conditioning IS active via:**
        // (1) the flow's per-block `spk_emb_linear` (weights live in
        //     `flow.flows.<i>.enc.spk_emb_linear`, wired in
        //     `SbV2TransformerCouplingLayer`)
        // (2) the SDP's `cond(g)` primitive (step 6 below)
        // both fed the raw `[d_speaker]` conditioning vector
        // `speaker_e_flow` — NOT the projected `[d_model]` output of
        // `enc_p.encoder.spk_emb_linear`.
        //
        // **Revisit trigger** (ADR §6): if T15's per-layer flow dumps +
        // UTMOS delta > 0.1 exposes a real missing speaker-conditioning
        // path, the ADR is revisited with owner-supplied primary-source
        // evidence for a two-side upgrade to (a).
        //
        // **CRITICAL**: Do NOT reintroduce the pre-Bug-4 broadcast-add
        // into `hidden_for_flow` (mod.rs:1215-1236 documented the OOM
        // chain: bridge+speaker ±33 → RQS softmax saturation → runaway
        // durations sum=24229/max=12036 → 46.6 GiB peak).
        //
        // | request.speaker_embedding | model.speaker_projection | path                                                            |
        // |---------------------------|--------------------------|-----------------------------------------------------------------|
        // | `Some(vec)`               | `Some(proj)`             | validate via `proj.forward(vec)`; pass `vec` to SDP.g and flow |
        // | `None`                    | `Some(proj)`             | validate via `proj.forward(zeros)`; pass zeros to SDP.g and flow |
        // | `Some(_)`                 | `None`                   | loud `VokraError::InvalidArgument` (FR-EX-08)                  |
        // | `None`                    | `None`                   | legacy: `speaker_embed.lookup(speaker_id)`, pass to SDP.g and flow |
        let d_model = self.text_encoder.d_model();
        // Post-STYLE-INJECTOR-fix (2026-08-09): the only pre-fix reader
        // of `phoneme_count` in this synthesize body was the dropped
        // `style_injector.inject(&mut hidden_for_flow, phoneme_count,
        // &req.style_vec)` call. Prefixed with `_` so a future
        // reintroduction of a phoneme-count-sized loop notices the
        // rename; kept in scope so `cargo clippy -D warnings` stays
        // green.
        let _phoneme_count = phon.phoneme_ids.len();
        let speaker_e_flow: Vec<f32> = match (
            req.speaker_embedding.as_deref(),
            self.speaker_projection.as_ref(),
        ) {
            (Some(ext), Some(proj)) => {
                // Real-ckpt path: caller-supplied external embedding.
                // `proj.forward` still loudly rejects wrong-length input
                // (FR-EX-08). The projected `[d_model]` result is
                // discarded — see step 5's Bug 4 fix comment.
                let projected = match compute.as_ref() {
                    Some(compute) => proj.forward_with_compute(compute, ext)?,
                    None => proj.forward(ext)?,
                };
                debug_assert_eq!(
                    projected.len(),
                    d_model,
                    "SbV2Model::synthesize: ExternalSpeakerProjection's d_out must equal \
                         SbV2TextEncoder's d_model — see SbV2Model's struct doc"
                );
                let _ = projected; // discard; see Bug 4 fix comment
                ext.to_vec()
            }
            (None, Some(proj)) => {
                // Deterministic zero-shot default. Projected result
                // discarded — see Bug 4 fix.
                let zeros = vec![0.0_f32; proj.d_in()];
                let projected = match compute.as_ref() {
                    Some(compute) => proj.forward_with_compute(compute, &zeros)?,
                    None => proj.forward(&zeros)?,
                };
                debug_assert_eq!(
                    projected.len(),
                    d_model,
                    "SbV2Model::synthesize: ExternalSpeakerProjection's d_out must equal \
                         SbV2TextEncoder's d_model — see SbV2Model's struct doc"
                );
                let _ = projected; // discard; see Bug 4 fix comment
                zeros
            }
            (Some(_), None) => {
                return Err(VokraError::InvalidArgument(
                    "SbV2Model::synthesize: caller-supplied speaker_embedding was provided \
                         but this model carries no ExternalSpeakerProjection — attach one via \
                         SbV2Model::with_external_speaker_projection, or pass \
                         speaker_embedding = None and use speaker_id for the legacy \
                         SpeakerEmbedding::lookup path (FR-EX-08: silent discard of \
                         caller-supplied speaker data is forbidden)"
                        .to_string(),
                ));
            }
            (None, None) => {
                // Legacy synthetic-only path: keep the lookup call so
                // `req.speaker_id` out-of-range still surfaces the
                // documented `SpeakerEmbedding::lookup` error.
                let speaker_e = self.speaker_embed.lookup(req.speaker_id)?;
                debug_assert_eq!(
                    speaker_e.len(),
                    d_model,
                    "SbV2Model::synthesize: SpeakerEmbedding's d_speaker must equal \
                         SbV2TextEncoder's d_model on the legacy lookup path — see \
                         SbV2Model's struct doc"
                );
                speaker_e.to_vec()
            }
        };
        // Wave-4 INTERMEDIATE-ACCESSORS: snapshot the raw speaker
        // conditioning vector BEFORE per-stage slice/pad reconciliation
        // (SDP / flow / decoder each slice this to their own gin_channels
        // separately). The Python dumper writes `speaker_embed.bin` as
        // the pre-conditioning vector, so this snapshot matches its shape
        // exactly on every path arm.
        let speaker_embed_snapshot = speaker_e_flow.clone();
        // Wave-4 INTERMEDIATE-ACCESSORS: snapshot the style projection
        // `[d_target]` — matches the Python dumper's `style_projected.bin`
        // slot (see `StyleVectorInjector::project` doc). Computed even
        // though it is not otherwise mixed into `hidden_for_flow` (see
        // STYLE-INJECTOR fix below) so parity harnesses can diff the
        // projection value directly.
        let style_projected_snapshot = match compute.as_ref() {
            Some(compute) => self
                .style_injector
                .project_with_compute(compute, &req.style_vec)?,
            None => self.style_injector.project(&req.style_vec),
        };
        // STYLE-INJECTOR fix (2026-08-09): the Python reference
        // (`sbv2_dump_reference.py` step 9) explicitly does NOT mix
        // style into `text_hidden` on the base-checkpoint path — "style
        // is dumped for the manifest slot but not otherwise mixed
        // here". The pre-fix code called `self.style_injector.inject(
        // &mut hidden_for_flow, ..)` unconditionally, a latent parity
        // risk that base-ckpt tests missed because base ckpt ships
        // all-zero style projections (identity injector).
        //
        // A future fine-tune SKU with real `emb_g_style.{weight,bias}`
        // tensors would silently diverge from Python without any test
        // catching it. Dropping the call matches the Python reference
        // for the base-ckpt path AND for any future fine-tune (Python
        // dumps but does not mix into the text branch either). If a
        // future variant DOES need style mixed into the text branch,
        // that is an owner-supplied ADR + real fine-tune ckpt
        // introspection under a licensing-cleared upstream — not a
        // silent Rust-side interpretation.
        //
        // `self.style_injector` remains held on `SbV2Model` so
        // introspective test harnesses and future consumers keep the
        // handle; it just no longer runs on this pipeline step.
        //
        // `req.style_vec` is still shape-validated by
        // `SbV2Model::synthesize`'s callers and manifest-dumped by
        // parity harnesses — dropping the injection call does not
        // weaken the caller-supplied-data contract. `self.style_injector`
        // is retained on the model so an introspective test harness or
        // a future consumer can still call `.d_style()` / `.inject(..)`
        // directly; only the pipeline's automatic call is dropped.

        // 6. SDP -> durations
        //
        // Blocker 9 (2026-08-06): `SbV2SDP` is constructed in two shapes:
        // (a) real ckpt path where `sdp.cond: Conv1d(d_speaker, d_hidden,
        //     1)` gives `sdp.gin() == d_speaker` (e.g. 512), and
        // (b) synthetic path (paper defaults, no `cond` layer) where
        //     `sdp.gin() == d_hidden == d_model` (e.g. 6/192).
        // The dispatch above always returns `speaker_e_flow` sized as
        // `d_speaker` (or, on the legacy `(None, None)` path, whatever
        // the lookup table's row width is). Reconcile to what SDP
        // expects: (a) use full raw vector, (b) slice to `sdp.gin()`.
        // Truncation is safe: synthetic zero-init speakers are all-zero
        // anyway; on the real ckpt we always take the full vector.
        let sdp_gin = self.sdp.gin();
        let sdp_g_owned: Vec<f32> = if speaker_e_flow.len() == sdp_gin {
            Vec::new() // reuse via slice below
        } else if speaker_e_flow.len() >= sdp_gin {
            // Truncate — synthetic case where d_speaker >= d_hidden.
            speaker_e_flow[..sdp_gin].to_vec()
        } else {
            // Pad with zeros — extremely unusual; keeps loud-fail
            // debug_assert in `SbV2SDP::sample` intact only if len
            // matches, so this path exists purely for symmetry.
            let mut v = vec![0.0_f32; sdp_gin];
            v[..speaker_e_flow.len()].copy_from_slice(&speaker_e_flow);
            v
        };
        let sdp_g: &[f32] = if sdp_g_owned.is_empty() {
            &speaker_e_flow
        } else {
            &sdp_g_owned
        };
        // Bug 4 fix (2026-08-08): feed SDP the RAW text_hidden (encoder
        // output), NOT the accumulated bridge+speaker+style buffer.
        // Matches Python `sbv2_dump_reference.py::run_pipeline_body`
        // step 10 which calls `sdp.sample(text_hidden_transposed, ...)`.
        // Empirical proof of correctness at time of fix: an env-var
        // override that replaced `hidden` with the reference dump's
        // `text_hidden.bin` (magnitude ±0.9) produced sane durations
        // (sum=28/max=10) matching Python reference (sum=26/max=8),
        // while the pre-fix `hidden` (magnitude ±33 from broadcast
        // adds) produced runaway durations (sum=24229/max=12036).
        //
        // Step 10 (torch.randn parity, 2026-08-08): dispatch the SDP
        // noise draws through `req.rng_mode` so a caller opting into
        // torch parity (the default; see `RngMode`) gets byte-exact
        // agreement with `torch.manual_seed(seed); torch.randn(...)`
        // under the PhiloxRNGEngine.h path — verified by
        // `crates/vokra-models/tests/sbv2_sdp_torch_parity.rs`. The
        // legacy path is unchanged so pre-Step-10 synthetic tests keep
        // their byte-frozen assertions when they opt into it.
        let mut durations = match req.rng_mode {
            RngMode::PhiloxRngEnginePyTorchParity => {
                let mut rng = TorchRandnStream::new(req.seed);
                match compute.as_ref() {
                    Some(compute) => self.sdp.sample_with_compute(
                        compute,
                        &text_hidden,
                        phon.phoneme_ids.len(),
                        sdp_g,
                        &mut rng,
                        req.noise_scale_w,
                    )?,
                    None => self.sdp.sample(
                        &text_hidden,
                        phon.phoneme_ids.len(),
                        sdp_g,
                        &mut rng,
                        req.noise_scale_w,
                    ),
                }
            }
            RngMode::GaussianSplitMix64Legacy => {
                let mut rng = GaussianSplitMix64::new(req.seed);
                match compute.as_ref() {
                    Some(compute) => self.sdp.sample_with_compute(
                        compute,
                        &text_hidden,
                        phon.phoneme_ids.len(),
                        sdp_g,
                        &mut rng,
                        req.noise_scale_w,
                    )?,
                    None => self.sdp.sample(
                        &text_hidden,
                        phon.phoneme_ids.len(),
                        sdp_g,
                        &mut rng,
                        req.noise_scale_w,
                    ),
                }
            }
        };
        for d in &mut durations {
            *d = ((*d as f32) / req.speed).max(1.0) as i32;
        }

        // OOM-STOPGAP-CLEANUP Phase 2 (2026-08-09, audit rank 17):
        //
        // A temporary per-phoneme duration cap (500 frames ≈ 5.8s at
        // 86 Hz) + a matching stderr-warning `eprintln!` used to sit
        // here as a safety fuse while Wave-2 SBV2-BUG4 was under
        // investigation — an upstream `text_encoder` producing hidden
        // values ~35× too large in magnitude drove the SDP's
        // `exp().ceil()` into runaway integer durations (max ≈ 26539 in
        // CI run 31197123061), which then blew the runtime up in the
        // flow-attention `[mel_seq_len × mel_seq_len × f32]` allocation
        // (~46.6 GiB). Bug 4's true root cause chain (missing PosFFN
        // `x*x_mask`, missing `enc_p.encoder.spk_emb_linear` per-block
        // gating, and adjacent BERT-bridge / flow-noise-scale / style-
        // injector issues) landed on this branch across the Wave-2 fix
        // wave (`f3b10ab hifigan per-iteration residual + convs2 chain`,
        // `15df641 test(sbv2/parity): pin text_encoder bit-exact`,
        // `b2c5c96 posffn-xmask`, `af58ba8 flow-noise-scale`, `bfaf2ac
        // style-injector`, `058030b hgan-05 dec.cond speaker`, etc.),
        // making the cap and its stderr warning dead code — audit rank
        // 17 Phase 2 remedy called for total deletion. The
        // `sbv2_oom_stopgap_removed.rs` integration test pins the
        // deletion against future re-introduction (source-string check
        // on both the removed constant identifier and the removed stderr
        // fingerprint). Historical trail preserved in
        // `docs/handoff/sbv2-sdp-debug-2026-08-08.md`.

        // Wave-4 INTERMEDIATE-ACCESSORS: snapshot post-speed-scale
        // integer durations for the `sdp_sample.bin` slot (matches
        // Python dumper's post-speed-scale integer durations).
        let sdp_sample_snapshot = durations.clone();

        // 7. Length regulate — uses `hidden_for_flow` (= text_hidden +
        // bridge, matching Python `bert_bridge_out`). Bug 4 fix
        // (2026-08-08): pre-fix code fed the accumulated `hidden` which
        // included speaker/style broadcast-adds; Python reference does
        // not add speaker/style here (they enter via flow's per-block
        // spk_emb_linear and decoder's `dec.cond` respectively).
        let mel_hidden = length_regulate(&hidden_for_flow, &durations, d_model);
        // Wave-4 INTERMEDIATE-ACCESSORS: snapshot the length-regulated
        // mel_hidden BEFORE the flow's noise reparameterization consumes
        // it — matches Python dumper's `mel_hidden.bin` shape (dumper
        // dumps mel_hidden pre-flow too).
        let mel_hidden_snapshot = mel_hidden.clone();
        let mel_seq_len = durations.iter().sum::<i32>() as usize;
        debug_assert!(
            mel_seq_len > 0,
            "SbV2Model::synthesize: mel_seq_len must be positive (every SbV2SDP::sample \
             duration is >= 1 by construction once phoneme_ids is non-empty)"
        );

        // 8. Flow inverse.
        //
        // Blocker 9 (2026-08-06): The flow's per-block `spk_emb_linear:
        // Linear(gin_channels, hidden)` is trained against the raw
        // `[d_speaker]` vector alone (`d_speaker = 512` real, or the
        // corresponding synthetic `d_model` value on synthetic paths).
        // Style is NOT concatenated into `g` — it is applied earlier
        // via `style_injector.inject` at step 5b, before the flow
        // sees `mel_hidden`.
        //
        // The pre-Blocker-9 `g_stopgap = speaker_e_flow ‖ style_vec`
        // concat guessed at a composition that the real ckpt's
        // Sbv2FlowEncoder (see `tools/parity/vendor/vits/sbv2_flow.py`)
        // does not use; its `spk_emb_linear` weight has shape
        // `[hidden=192, gin=512]`, so the input must be exactly 512-d.
        //
        // Slice or pad to `flow.gin_channels()` when caller vector
        // length differs (synthetic paths where `speaker_e_flow` len
        // may not match `flow.gin_channels()`).
        let flow_gin = self.flow.gin_channels();
        let flow_g_owned: Vec<f32> = if speaker_e_flow.len() == flow_gin {
            Vec::new()
        } else if speaker_e_flow.len() >= flow_gin {
            speaker_e_flow[..flow_gin].to_vec()
        } else {
            let mut v = vec![0.0_f32; flow_gin];
            v[..speaker_e_flow.len()].copy_from_slice(&speaker_e_flow);
            v
        };
        let flow_g: &[f32] = if flow_g_owned.is_empty() {
            &speaker_e_flow
        } else {
            &flow_g_owned
        };

        // FLOW-NOISE-SCALE fix (2026-08-09): reparameterize the flow's
        // Gaussian prior with `req.noise_scale`. The Python reference
        // draws `torch.randn_like(mel_hidden)`, scales by
        // `req.noise_scale`, and adds elementwise to `mel_hidden`
        // before `flow.inverse` — the standard VITS-family
        // reparameterization step (`z_p = mean + torch.randn * scale`),
        // simplified here to `mean = mel_hidden, logstd = 0` (real
        // prior-head mean/logstd split lands with the
        // SBV2-INFO-01-ENC-P-PROJ scaffold; until then this bridge
        // treats the length-regulated text hidden as the mean).
        //
        // Draws use `req.rng_mode` dispatch (identical to the SDP step
        // 6 above) so the torch-parity path byte-matches
        // `torch.manual_seed(seed); torch.randn(...)` under the
        // PhiloxRNGEngine.h contract that
        // `crates/vokra-core/tests/rng_torch_randn_cpu_parity.rs` pins.
        // The legacy Split-Mix64 stream is preserved so pre-Step-10
        // synthetic tests that opt into it keep byte-frozen output when
        // `noise_scale == 0.0` (skipped fill loop) or when they pass
        // `RngMode::GaussianSplitMix64Legacy`.
        //
        // Zero-noise fast path: when `noise_scale == 0.0` the pre-fix
        // behavior (feed `mel_hidden` unchanged into `flow.inverse`) is
        // byte-identical — skip the RNG draw entirely so `noise_scale =
        // 0.0` remains a byte-frozen deterministic pipeline regardless
        // of `req.seed` and `req.rng_mode` (matches every existing
        // synthetic parity test's noise_scale=0.0 posture).
        let mel_hidden_with_noise: Vec<f32> = if req.noise_scale == 0.0 {
            mel_hidden
        } else {
            let flow_noise = draw_flow_prior_noise(req.seed, req.rng_mode, mel_seq_len, d_model);
            debug_assert_eq!(
                flow_noise.len(),
                mel_hidden.len(),
                "flow prior noise buffer length must match mel_hidden"
            );
            let scale = req.noise_scale;
            let mut buf = mel_hidden;
            for (m, n) in buf.iter_mut().zip(flow_noise.iter()) {
                *m += *n * scale;
            }
            buf
        };
        let z = match compute.as_ref() {
            Some(compute) => self.flow.inverse_with_compute(
                compute,
                &mel_hidden_with_noise,
                mel_seq_len,
                flow_g,
            )?,
            None => self
                .flow
                .inverse(&mel_hidden_with_noise, mel_seq_len, flow_g),
        };
        // Wave-4 INTERMEDIATE-ACCESSORS: snapshot the flow inverse
        // output BEFORE the transpose + decoder consume it — matches
        // Python dumper's `z_latent.bin` shape.
        let z_latent_snapshot = z.clone();

        // 9. HiFi-GAN decoder — transpose SbV2Flow::inverse's time-major
        // [mel_seq_len, d_z] into SbV2Decoder::generate's channel-major
        // [n_mels, mel_seq_len] (decoder.rs's module doc: "bridging one
        // layout to the other is Task 23's integration concern, not this
        // thin wrapper's"). d_z must equal the decoder's n_mels (a Task 23
        // construction-time contract, held by `synthetic_for_test` above);
        // a mismatch surfaces as `SbV2Decoder::generate`'s own
        // `debug_assert!`.
        //
        // HGAN-05-GIN-COND (2026-08-09): when the loaded decoder carries
        // a `cond` (speaker conditioning) layer, thread the raw
        // `[d_speaker]` speaker vector through — sliced/padded to
        // `decoder.gin_channels()` — so the upstream `x = x +
        // self.cond(g)` broadcast-add fires after `conv_pre`. Multi-
        // speaker SBV2 v2 checkpoints ship `dec.cond.*` (gin_channels =
        // 512); single-speaker fixtures / synthetic tests do not, and
        // decoder.has_gin_condition() → false short-circuits to the
        // unconditioned path.
        let z_channel_major = transpose_time_major_to_channel_major(&z, mel_seq_len);
        let pcm = if self.decoder.has_gin_condition() {
            let decoder_gin = self.decoder.gin_channels();
            // Slice/pad speaker_e_flow to decoder.gin_channels() — the
            // same reconciliation pattern SDP + flow use above (see
            // steps 6 and 8).
            let decoder_g_owned: Vec<f32> = if speaker_e_flow.len() == decoder_gin {
                Vec::new()
            } else if speaker_e_flow.len() >= decoder_gin {
                speaker_e_flow[..decoder_gin].to_vec()
            } else {
                let mut v = vec![0.0_f32; decoder_gin];
                v[..speaker_e_flow.len()].copy_from_slice(&speaker_e_flow);
                v
            };
            let decoder_g: &[f32] = if decoder_g_owned.is_empty() {
                &speaker_e_flow
            } else {
                &decoder_g_owned
            };
            self.decoder.generate_conditioned_with_backend(
                &z_channel_major,
                mel_seq_len,
                Some(decoder_g),
                self.backend,
            )?
        } else {
            self.decoder.generate_conditioned_with_backend(
                &z_channel_major,
                mel_seq_len,
                None,
                self.backend,
            )?
        };

        let audio = SynthesizedAudio::new(pcm, self.decoder.sample_rate());
        let intermediates = SbV2Intermediates {
            phoneme_embed,
            text_hidden,
            bert_hidden_ja: bert_hidden_ja_snapshot,
            bert_hidden_en: bert_hidden_en_snapshot,
            bert_hidden_zh: bert_hidden_zh_snapshot,
            bert_bridge_out: bert_bridge_out_snapshot,
            speaker_embed: speaker_embed_snapshot,
            style_projected: style_projected_snapshot,
            sdp_sample: sdp_sample_snapshot,
            mel_hidden: mel_hidden_snapshot,
            z_latent: z_latent_snapshot,
        };
        Ok((audio, intermediates))
    }
}

impl SbV2Intermediates {
    /// Maps every intermediate field to its dumper filename
    /// (`reference_dump/<name>.bin`), keyed as `(dumper_name,
    /// as_f32_bytes)` for a driver-style tensor-diff loop. `sdp_sample`
    /// is emitted as f32 (matching the Python dumper's f32 dump — it
    /// writes `sdp_sample.tobytes()` after keeping durations as float
    /// tensors internally); consumers reading the returned `Vec<u8>`
    /// slice can `bytemuck::cast_slice` back to f32.
    ///
    /// Wave-4 INTERMEDIATE-ACCESSORS: added so parity harnesses can
    /// iterate every intermediate uniformly against the manifest's
    /// `tensors[]` array without switching on each field individually.
    /// Empty per-language BERT hidden buckets (`bert_hidden_ja` on an EN
    /// request; `bert_hidden_en` on a JA request) are OMITTED from the
    /// map — the parity harness should skip the corresponding manifest
    /// entry on the inactive side.
    #[must_use]
    pub fn to_dumper_map(&self) -> Vec<(&'static str, Vec<u8>)> {
        fn f32_bytes(v: &[f32]) -> Vec<u8> {
            v.iter().flat_map(|f| f.to_le_bytes()).collect()
        }
        let mut out: Vec<(&'static str, Vec<u8>)> = Vec::with_capacity(11);
        out.push(("phoneme_embed", f32_bytes(&self.phoneme_embed)));
        out.push(("text_hidden", f32_bytes(&self.text_hidden)));
        if !self.bert_hidden_ja.is_empty() {
            out.push(("bert_hidden_ja", f32_bytes(&self.bert_hidden_ja)));
        }
        if !self.bert_hidden_en.is_empty() {
            out.push(("bert_hidden_en", f32_bytes(&self.bert_hidden_en)));
        }
        if !self.bert_hidden_zh.is_empty() {
            out.push(("bert_hidden_zh", f32_bytes(&self.bert_hidden_zh)));
        }
        out.push(("bert_bridge_out", f32_bytes(&self.bert_bridge_out)));
        out.push(("speaker_embed", f32_bytes(&self.speaker_embed)));
        out.push(("style_projected", f32_bytes(&self.style_projected)));
        // `sdp_sample` matches the Python dumper's f32-cast dump.
        let sdp_f32: Vec<f32> = self.sdp_sample.iter().map(|&d| d as f32).collect();
        out.push(("sdp_sample", f32_bytes(&sdp_f32)));
        out.push(("mel_hidden", f32_bytes(&self.mel_hidden)));
        out.push(("z_latent", f32_bytes(&self.z_latent)));
        out
    }
}

impl TtsEngine for SbV2Model {
    /// Adapts the cross-engine [`SynthesisRequest`] shape to
    /// [`SbV2SynthRequest`] and calls the inherent
    /// [`SbV2Model::synthesize`] — the same "convert, then call the model's
    /// own inherent `synthesize`" pattern `piper_plus::PiperPlusTts`'s
    /// `TtsEngine` impl uses.
    ///
    /// `request.language` maps case-insensitively: any value starting with
    /// `"en"` selects [`Language::EN`]; any value starting with `"zh"`
    /// (WP-18: covers `"zh"` / `"zh-cn"` / `"zh_cn"` etc.) selects
    /// [`Language::ZH`]; anything else — including `None` — selects
    /// [`Language::JA`] (SBV2's base config is Japanese-first, per its
    /// JP-Extra heritage — see `decoder.rs`'s module doc). Note that
    /// selecting [`Language::ZH`] here reaches [`Self::synthesize`]'s
    /// fail-closed ZH gate until the ZH BERT WP lands; only the G2P side
    /// is exercisable end-to-end today (via [`SbV2Phonemizer::phonemize`]
    /// directly).
    /// `request.deterministic` zeroes both `noise_scale` and
    /// `noise_scale_w` (mirrors the piper-plus adapter's identical
    /// convention); otherwise this adapter applies
    /// `docs/superpowers/specs/2026-07-26-sbv2-v2-design.md` §7's documented
    /// SDP defaults (`noise_scale = 0.667`, `noise_scale_w = 0.8`).
    ///
    /// # WP-23: `style_vec` + `speaker_id` threading
    ///
    /// `request.style_vec` and `request.speaker_id` (both
    /// [`Option`]-typed on [`SynthesisRequest`]) flow through to the
    /// corresponding [`SbV2SynthRequest`] fields:
    ///
    /// - `style_vec = Some(v)` → forwarded verbatim; length must equal
    ///   [`self.style_injector.d_style()`](StyleVectorInjector::d_style)
    ///   or the adapter errors loudly with
    ///   [`VokraError::InvalidArgument`] (FR-EX-08 — never silently
    ///   truncate / zero-pad).
    /// - `style_vec = None` → the identity all-zero vector sized from
    ///   `self.style_injector.d_style()` (pre-WP-23 behavior — preserves
    ///   an existing caller's PCM byte-for-byte).
    /// - `speaker_id = Some(id)` → forwarded verbatim; out-of-range
    ///   propagates as [`VokraError::InvalidArgument`] from
    ///   [`SpeakerEmbedding::lookup`].
    /// - `speaker_id = None` → speaker 0 (pre-WP-23 behavior — preserves
    ///   an existing caller's PCM byte-for-byte).
    ///
    /// The capability probes [`supports_style_vec`](Self::supports_style_vec)
    /// and [`supports_multi_speaker`](Self::supports_multi_speaker) both
    /// return `true` so a cross-engine caller can safely populate either
    /// field before building the request.
    ///
    /// # Speaker conditioning (Blocker 3)
    ///
    /// `request.speaker_embedding` is forwarded verbatim to
    /// [`SbV2SynthRequest::speaker_embedding`]. The inherent
    /// [`SbV2Model::synthesize`] step 5 dispatches on
    /// `(request.speaker_embedding, self.speaker_projection)` — see that
    /// method's dispatch-table doc. In particular, a `Some(_)` embedding
    /// on a model with no [`ExternalSpeakerProjection`] loaded
    /// (e.g. a legacy `SbV2Model::synthetic_for_test`) surfaces as a
    /// [`VokraError::InvalidArgument`] from the inherent method — the
    /// same loud-error posture the pre-Blocker-3 adapter used to enforce
    /// itself, moved down into the pipeline so both entry points
    /// (inherent `synthesize` and this trait method) surface identical
    /// errors.
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::InvalidArgument`] if:
    /// - `request.prosody_features` is `Some(..)` — SBV2 derives
    ///   pitch-accent tones from its own G2P, not a caller-supplied
    ///   per-phoneme accent triple; honoring it would silently discard
    ///   caller-supplied data.
    /// - `request.style_vec = Some(v)` with `v.len() != d_style` — the
    ///   AdaIN projection cannot silently reshape a mismatched vector
    ///   (FR-EX-08).
    /// - The inherent [`synthesize`](SbV2Model::synthesize) call fails
    ///   for any reason it documents there. In particular,
    ///   `request.speaker_embedding = Some(_)` on a model with no
    ///   [`ExternalSpeakerProjection`] loaded surfaces as
    ///   [`VokraError::InvalidArgument`] from that inherent method (the
    ///   "loud-error posture" the pre-Blocker-3 adapter used to enforce
    ///   itself, moved down into the pipeline so both entry points
    ///   surface identical errors — see the "Speaker conditioning" note
    ///   above).
    ///
    /// # WP-13a (2026-08-10)
    ///
    /// Pre-Blocker-3 the adapter rejected any `request.speaker_embedding
    /// = Some(_)` upfront with a hard-coded `InvalidArgument`. Blocker 3
    /// moved the loud-error contract down into
    /// [`SbV2Model::synthesize`], which now honors a caller-supplied
    /// embedding when [`.with_external_speaker_projection(_)`] has been
    /// wired (and still surfaces the same loud error when it has not).
    /// This adapter's job is now to *thread* `speaker_embedding` through,
    /// not to reject it — the pre-Blocker-3 rejection block was orphaned
    /// dead code by the earlier refactor. `sbv2_speaker_external.rs`
    /// tests hit the loud-error path via the inherent method, not this
    /// adapter, so removing the guard here restores the intended
    /// symmetry the Blocker-3 rustdoc already advertised (see the
    /// "Speaker conditioning" paragraph above).
    fn synthesize(&self, request: &SynthesisRequest) -> Result<SynthesizedAudio> {
        if request.prosody_features.is_some() {
            return Err(VokraError::InvalidArgument(
                "SbV2Model (TtsEngine): caller-supplied prosody_features is not supported — \
                 SBV2 derives pitch-accent tones from its own G2P; call SbV2Model::synthesize \
                 directly"
                    .to_string(),
            ));
        }

        // Case-insensitive prefix match: `"en"*` -> EN, `"zh"*` -> ZH,
        // everything else (including `None`) -> JA (SBV2 v2's base config
        // is Japanese-first, per its JP-Extra heritage — see
        // `decoder.rs`'s module doc). ZH selection routes to the
        // language_embed row 2 code path but currently returns
        // NotImplemented at either the phonemizer or the BERT tokenizer
        // step below — see `Language`'s ZH scope note.
        let language = match request.language.as_deref() {
            Some(lang) if lang.to_ascii_lowercase().starts_with("en") => Language::EN,
            // WP-18: `"zh"` / `"zh-cn"` / `"zh_cn"` all select ZH — same
            // case-insensitive prefix convention EN uses. Note that the
            // full [`Self::synthesize`] pipeline still fail-closes on ZH
            // until the ZH BERT WP lands; only G2P is wired here.
            Some(lang) if lang.to_ascii_lowercase().starts_with("zh") => Language::ZH,
            _ => Language::JA,
        };
        let (noise_scale, noise_scale_w) = if request.deterministic {
            (0.0, 0.0)
        } else {
            // docs/superpowers/specs/2026-07-26-sbv2-v2-design.md §7's
            // documented SDP defaults.
            (0.667, 0.8)
        };

        // WP-23: thread `request.style_vec` / `speaker_id` through, with
        // the pre-WP-23 defaults (`None` → zero vector / speaker 0)
        // preserved so an existing caller keeps its byte-for-byte PCM.
        // The length check is explicit here — `StyleVectorInjector::inject`
        // itself uses `debug_assert!` for shape validation (see its doc:
        // hot inner-loop constructor, not a public API validation
        // boundary), so a release build without this check would happily
        // read past the buffer or panic non-descriptively. FR-EX-08
        // requires the loud, descriptive error path.
        let d_style = self.style_injector.d_style();
        let style_vec = match &request.style_vec {
            None => vec![0.0; d_style],
            Some(v) => {
                if v.len() != d_style {
                    return Err(VokraError::InvalidArgument(format!(
                        "SbV2Model (TtsEngine): style_vec length mismatch — got {}, expected \
                         {} (StyleVectorInjector::d_style())",
                        v.len(),
                        d_style,
                    )));
                }
                v.clone()
            }
        };
        let speaker_id = request.speaker_id.unwrap_or(0);

        let sbv2_request = SbV2SynthRequest {
            text: request.text.clone(),
            language,
            speaker_id,
            speaker_embedding: request.speaker_embedding.clone(),
            style_vec,
            speed: 1.0,
            noise_scale,
            noise_scale_w,
            seed: 0,
            // Cross-engine adapter callers get the torch-parity path by
            // default — the only new construction site introduced after
            // Step 10, so no byte-frozen synthetic fixture depends on
            // its RNG choice.
            rng_mode: RngMode::default(),
        };
        self.synthesize(&sbv2_request)
    }

    fn backend(&self) -> BackendKind {
        self.backend
    }

    /// SBV2 threads [`SynthesisRequest::style_vec`] into its
    /// [`StyleVectorInjector`] AdaIN pipeline — WP-23.
    fn supports_style_vec(&self) -> bool {
        true
    }

    /// SBV2 threads [`SynthesisRequest::speaker_id`] into its
    /// [`SpeakerEmbedding`] discrete-id table lookup — WP-23.
    fn supports_multi_speaker(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Task 24: SbV2Model::from_gguf — real-fixture GGUF loader
// ---------------------------------------------------------------------------
//
// Loads a full `SbV2Model` from three separately converted GGUF files: the
// Task 25 converter's own `main` output, plus `bert_ja` / `bert_en` (each
// `vokra-bert`'s Task 11 converter output). `from_gguf`'s doc is the
// canonical `vokra.sbv2.*` metadata schema and `sbv2.*` tensor hierarchy —
// Task 25's converter must emit exactly this shape.

/// A [`vokra_piper_plus::Phonemizer`] stand-in for a [`SbV2Model`] built by
/// [`SbV2Model::from_gguf`], which has no G2P wired at load time (see that
/// method's "G2P is not loaded here" doc section). Every call is a loud
/// [`VokraError::NotImplemented`] rather than a silent fallback to
/// [`SbV2Phonemizer::synthetic_for_test`]'s toy char-mapping G2P, which
/// would let a `from_gguf`-loaded model *load* successfully but
/// *synthesize* wrong-but-plausible-looking audio for any real text —
/// exactly the class of silent failure FR-EX-08 forbids.
struct UnwiredPhonemizer;

impl vokra_piper_plus::Phonemizer for UnwiredPhonemizer {
    fn phonemize(&self, _text: &str) -> Result<Vec<i64>> {
        Err(VokraError::NotImplemented(
            "SbV2Model::from_gguf loads no G2P: piper-plus text-to-phoneme is a separate \
             model with its own GGUF file, not one of the 3 files this loader's signature \
             takes (main + bert_ja + bert_en). To synthesize, rebuild the model via \
             SbV2Model::new, passing a real phonemizer built with \
             SbV2Phonemizer::from_piper_g2p (see g2p.rs's Task 15 real-G2P-wiring precedent) \
             in place of the placeholder from_gguf installs here — every other component \
             from_gguf loads (text encoder, BERT, bridge, speaker, style, SDP, flow, decoder) \
             is reusable as-is.",
        ))
    }
}

impl SbV2Model {
    /// Loads a full SBV2 (Style-Bert-VITS2 v2) model from three separately
    /// converted GGUF files.
    ///
    /// - `main` — this model's own weights: text encoder, BERT bridge,
    ///   speaker table, style injector, stochastic duration predictor,
    ///   normalizing flow, HiFi-GAN decoder (Task 25's converter output).
    /// - `bert_ja` — the JA-path [`DebertaV2Encoder`] plus its own
    ///   [`SbertTokenizer`] (`vokra-bert`'s converter output).
    /// - `bert_en` — the EN-path [`DebertaV3Encoder`] plus its own
    ///   [`SbertTokenizer`].
    ///
    /// # Metadata keys (`vokra.sbv2.*`, all read from `main`)
    ///
    /// Every key below is required; scalar dims are `u32` unless noted
    /// otherwise (`decoder.leaky_relu_slope` is `f32`). A missing or
    /// wrong-typed key is [`VokraError::ModelLoad`] naming the key, never a
    /// silent default (FR-EX-08). Rationale for why each value lives here
    /// (vs. being derivable some other way) follows each entry.
    ///
    /// - `d_model` — [`SbV2TextEncoder`]/[`SbV2SDP`]/[`SbV2Flow`]-conditioning
    ///   shared hidden width (SBV2 v2's real-world value is 192, per
    ///   `docs/superpowers/specs/2026-07-26-sbv2-v2-design.md` §7's hparam
    ///   table — this loader treats it as checkpoint-driven, never
    ///   hard-coded).
    /// - `d_bert` — the DeBERTa hidden width [`BertBridge`] projects
    ///   *from*. Cross-checked against both `bert_ja`'s and `bert_en`'s own
    ///   loaded `get_d_model()` after they load — a mismatch is
    ///   [`VokraError::ModelLoad`], since [`SbV2Model`]'s single
    ///   `bert_bridge` field (Task 23) is shared by both languages and can
    ///   only have one `d_bert`.
    /// - `d_speaker` — [`SpeakerEmbedding`]'s per-speaker embedding width.
    ///   Read independently of `d_model`: [`synthesize`](Self::synthesize)'s
    ///   broadcast-add `debug_assert!`s the two are equal (a documented
    ///   scaffold limitation on the [`SbV2Model`] struct doc since Task
    ///   23) — this loader does not paper over a real checkpoint whose
    ///   speaker embedding is a different width (e.g. an external 512-d
    ///   table) by silently reshaping; that would need a projection layer
    ///   this task does not add.
    /// - `n_speakers` — [`SpeakerEmbedding`]'s table row count.
    /// - `d_style` — [`StyleVectorInjector`]'s input style-vector width.
    /// - `d_z` — [`SbV2Flow`]'s latent channel width, and — since
    ///   [`synthesize`](Self::synthesize) feeds the flow's output straight
    ///   into the decoder (`decoder.rs`'s "bridging one layout to the
    ///   other" doc) — also the decoder's `n_mels`. Must be even and
    ///   non-zero ([`SbV2Flow::from_layers`]'s own panic contract); this
    ///   loader checks that itself and returns [`VokraError::ModelLoad`]
    ///   instead of relying on a debug-only panic.
    /// - `n_vocab` — [`SbV2TextEncoder`]'s phoneme vocabulary size.
    /// - `n_tones` — pitch-accent tone count, shared by the text encoder
    ///   and [`SbV2SDP`]'s tone conditioning.
    /// - `d_ff` — [`SbV2TransformerBlock`](text_encoder::SbV2TransformerBlock)'s
    ///   FFN inner width. A metadata key rather than shape-derived from a
    ///   tensor's own dimensions: this reader's GGUF tensor `dimensions`
    ///   are stored **innermost-first** (`GgufTensorInfo::dimensions`'s own
    ///   doc) — the opposite axis order from this crate's row-major
    ///   `[out_dim, in_dim]` weight convention — so shape-deriving here
    ///   would risk a silently-transposed read. This follows
    ///   [`DebertaV2Encoder::from_gguf`]'s sibling precedent of reading
    ///   every shape parameter from metadata instead.
    /// - `n_text_layers` / `n_flow_layers` / `n_sdp_layers` — stack depths
    ///   for the text encoder's transformer blocks, the flow's coupling
    ///   layers and the SDP's coupling layers respectively. `0` is a
    ///   legitimate, exercised empty-stack configuration for all three
    ///   (each component's own module doc), not an error.
    /// - `sample_rate` — PCM output rate (shared with [`HifiGanAttrs`] and
    ///   [`SbV2Decoder`]'s own `sample_rate` field, which
    ///   [`SbV2Decoder::new`] `debug_assert!`s agree).
    /// - `decoder.initial_channel` / `decoder.conv_pre_kernel` /
    ///   `decoder.conv_post_kernel` — [`HifiGanAttrs::initial_channel`] and
    ///   the pre/post conv kernel widths
    ///   ([`HifiGanWeights::conv_pre_kernel`] /
    ///   [`HifiGanWeights::conv_post_kernel`]).
    /// - `decoder.upsample_rates` (array) — per-stage transposed-conv
    ///   stride. **Not** shape-derivable (a `ConvTranspose1d` attribute
    ///   with no trace in a tensor's shape — the same fact
    ///   `crates/vokra-models/src/piper_plus/config.rs`'s `DEC_UP_STRIDE`
    ///   doc records for piper-plus's own decoder).
    /// - `decoder.upsample_kernel_sizes` (array) — per-stage
    ///   transposed-conv kernel width.
    /// - `decoder.upsample_out_channels` (array) — per-stage output channel
    ///   count. [`HifiGanAttrs`] has no field for this (only
    ///   [`UpsampleStageWeights::out_ch`], a per-tensor loader-populated
    ///   value) — every real HiFi-GAN preset (V1/V2/V3) has its own
    ///   channel schedule, so this loader does not assume the halving
    ///   ladder [`synthetic_hifigan_weights`]'s test-only helper uses.
    /// - `decoder.resblock_kernel_sizes` (array) — per-MRF-branch kernel
    ///   width; array length fixes `n_mrf_branches`.
    /// - `decoder.resblock_dilation_counts` (array) — per-branch dilation
    ///   *count* (branches need not share a layer count — e.g. HiFi-GAN
    ///   V1's 3-layer branches vs. V3's 1-layer branches).
    /// - `decoder.resblock_dilations_flat` (array) — every branch's
    ///   dilation list concatenated in branch order; walked back into
    ///   per-branch slices using `resblock_dilation_counts` as the stride
    ///   table. A flat array avoids relying on nested-array metadata (no
    ///   other loader in this codebase emits or reads one, so this stays
    ///   consistent with the established flat-metadata convention).
    /// - `decoder.leaky_relu_slope` (`f32`) — [`HifiGanAttrs::leaky_relu_slope`].
    ///   Historically defaulted to the universal jik876/hifi-gan `LRELU_SLOPE`
    ///   = `0.1` (`vits_ja::VITS_JA_LEAKY_RELU_SLOPE`, piper-plus's
    ///   `LRELU_SLOPE`) when the metadata key was absent; WP-13 promoted the
    ///   read to required per FR-EX-08 — silently defaulting a hparam that
    ///   the checkpoint can legitimately vary would produce audio that is
    ///   subtly wrong with no observable signal. The Vokra converter
    ///   (`write_hparams` in `crates/vokra-convert/src/models/sbv2.rs`;
    ///   `vokra-convert`'s `models` module is private, so the file is the
    ///   only referent) always emits this
    ///   key (falling back to `0.1` at config-parse time when the JSON
    ///   side-car omits it), so no Vokra-produced GGUF is affected.
    ///
    /// `n_pos_buckets` is deliberately **not** read here: it is a
    /// DeBERTa-only concept (relative-position attention bucketing) that
    /// [`DebertaV2Encoder::from_gguf`] / [`DebertaV3Encoder::from_gguf`]
    /// already read from `bert_ja` / `bert_en`'s own
    /// `vokra.bert.deberta_v{2,3}.*` metadata; [`SbV2TextEncoder`]'s own
    /// transformer block is plain full-attention with no relative-position
    /// bias at all (`text_encoder.rs`'s module doc), so `vokra.sbv2.*` has
    /// nothing to bucket.
    ///
    /// # Tensor names (`sbv2.*`, all read from `main`)
    ///
    /// - `sbv2.text_encoder.phoneme_embed` / `.tone_embed` /
    ///   `.language_embed` — the three embedding tables
    ///   ([`SbV2TextEncoder::from_weights`]'s first three parameters).
    ///   `.language_embed` replaces the M6-pre-scout `.wb_embed`; see
    ///   [`SbV2TextEncoder`]'s "`language_embed` design correction"
    ///   section for the primary-source verification (real base
    ///   checkpoint has `enc_p.language_emb.weight [3, 192]`, no
    ///   `enc_p.word_boundary_emb.weight`).
    /// - Per text-encoder layer `<i>` in `0..n_text_layers`:
    ///   `sbv2.text_encoder.layer.<i>.attn.{q,k,v,o}.weight` (bias-free,
    ///   matching
    ///   [`SbV2TransformerBlock`](text_encoder::SbV2TransformerBlock)'s
    ///   struct fields exactly), `.ln1.{gamma,beta}`,
    ///   `.ffn.w1.{weight,bias}`, `.ffn.w2.{weight,bias}`,
    ///   `.ln2.{gamma,beta}`. The `ln1`/`ln2` naming (not the task brief's
    ///   guessed `norm1`/`norm2`) matches both this struct's actual field
    ///   names (`ln1_gamma`, ...) and its sibling
    ///   [`DebertaV2Encoder::from_gguf`]'s identical `.ln1.gamma` /
    ///   `.ln2.gamma` convention.
    /// - `sbv2.bert_bridge.conv.weight` / `.conv.bias` —
    ///   [`BertBridge::from_conv`]'s projection.
    /// - `sbv2.speaker.table` — [`SpeakerEmbedding::from_table`]'s table.
    /// - `sbv2.style_injector.proj_scale` / `.proj_bias` —
    ///   [`StyleVectorInjector::from_projections`]'s two projections.
    /// - `sbv2.sdp.tone_embed` / `.tone_bias` — [`SbV2SDP`]'s tone
    ///   conditioning (`SbV2SDP::from_weights`'s `tone_proj`/`tone_bias`
    ///   parameters; named `tone_embed` in the tensor path to mirror the
    ///   text encoder's own `tone_embed` table).
    /// - Per SDP flow layer `<i>` in `0..n_sdp_layers`:
    ///   `sbv2.sdp.flow_layer.<i>.proj_weight` / `.proj_bias` — the real
    ///   [`SbV2CouplingLayer`](duration::SbV2CouplingLayer) struct has one
    ///   fused `[2, d_hidden]` `proj_weight` (row 0 = log-scale
    ///   projection, row 1 = shift projection) and a `[2]` `proj_bias`,
    ///   **not** the task brief's guessed separate
    ///   `scale_weight`/`shift_weight`/`tone_embed_delta` names — there is
    ///   no per-layer tone field at all; tone conditioning is the
    ///   SDP-level `tone_embed`/`tone_bias` above, shared across every
    ///   layer.
    /// - Per flow layer `<i>` in `0..n_flow_layers` (Blocker 2b, 2026-08-06):
    ///   the VITS2 [`SbV2TransformerCouplingLayer`](flow::SbV2TransformerCouplingLayer)
    ///   is loaded from the following per-block tensors, plus a
    ///   [`FlowLayer::Flip`](flow::FlowLayer::Flip) inserted after each
    ///   coupling (matching upstream `p0p4k/vits2_pytorch/models.
    ///   TransformerCouplingBlock`'s flat `[TCL, Flip, TCL, Flip, ...]`
    ///   layout at `n_flows = n_flow_layers`):
    ///
    ///   - `sbv2.flow.layer.<i>.pre.{weight,bias}` — 1×1 Conv1d
    ///     `[d_hidden, half_d_z]` + `[d_hidden]` bias.
    ///   - `sbv2.flow.layer.<i>.spk_emb.{weight,bias}` — per-block g
    ///     projection, `[d_hidden, gin_channels]` + `[d_hidden]` bias
    ///     (matches upstream `TransformerCouplingLayer.spk_emb_linear`).
    ///   - Per encoder-stack layer `<j>` in `0..n_flow_encoder_layers`,
    ///     the six-family per-layer tensor set (identical to the text
    ///     encoder's own per-layer scheme documented above):
    ///     `sbv2.flow.layer.<i>.enc.<j>.attn.{conv_q,conv_k,conv_v,conv_o}.{weight,bias}`,
    ///     `sbv2.flow.layer.<i>.enc.<j>.attn.rel_pos_{k,v}`,
    ///     `sbv2.flow.layer.<i>.enc.<j>.ffn.conv_{1,2}.{weight,bias}`,
    ///     `sbv2.flow.layer.<i>.enc.<j>.norm{1,2}.{gamma,beta}`.
    ///   - `sbv2.flow.layer.<i>.post.{weight,bias}` — 1×1 Conv1d
    ///     `[post_out_dim, d_hidden]` + `[post_out_dim]` bias, where
    ///     `post_out_dim = half_d_z` if `mean_only` is true (SBV2 v2
    ///     base) else `2 * half_d_z`.
    ///
    /// **SDP (Blocker 2c, 2026-08-06):**
    /// - `sbv2.sdp.{pre,proj,cond}.{weight,bias}` — the SDP body's three
    ///   1×1 convs (`pre`/`proj` shape `[d_hidden, d_hidden, 1]`, `cond`
    ///   shape `[d_hidden, d_speaker, 1]` — speaker conditioning).
    /// - `sbv2.sdp.convs.{convs_sep,convs_1x1,norms_1,norms_2}.<i>.<w>`
    ///   for `i` in `0..3` — the module-level [`DDSConv`](duration::DDSConv)
    ///   stack shared across the body (upstream field names preserved
    ///   verbatim in the tail; see the sbv2 converter's `sdp.*` rewriter
    ///   arm for the full sub-key list).
    /// - `sbv2.sdp.ea.m` / `.logs` — the slot-0
    ///   [`ElementwiseAffine`](duration::ElementwiseAffine)'s `[2]` params
    ///   each (upstream `sdp.flows.0.m|logs`, remapped to `.ea.*`).
    /// - Per SDP `ConvFlow` `<i>` in `0..n_sdp_layers` (real SBV2 v2 base =
    ///   4, densified from upstream sparse indices `{1,3,5,7}`):
    ///   `sbv2.sdp.flow.<i>.pre.{weight,bias}`,
    ///   `sbv2.sdp.flow.<i>.convs.{convs_sep,convs_1x1,norms_1,norms_2}.<j>.<w>`
    ///   for `j` in `0..3`, and `sbv2.sdp.flow.<i>.proj.{weight,bias}`
    ///   (`proj` shape `[num_bins*3-1=29, d_hidden, 1]`). See
    ///   [`ConvFlow::from_weights`](duration::ConvFlow::from_weights) for
    ///   the full shape contract.
    ///
    /// - `sbv2.decoder.conv_pre.{weight,bias}`,
    ///   `sbv2.decoder.conv_post.{weight,bias}`.
    /// - Per upsample stage `<i>` in `0..upsample_rates.len()`:
    ///   `sbv2.decoder.upsample.<i>.{weight,bias}`.
    /// - Per MRF branch `<j>` in `0..resblock_kernel_sizes.len()` of stage
    ///   `<i>`, per layer `<l>` in `0..resblock_dilation_counts[j]`:
    ///   `sbv2.decoder.mrf.<i>.<j>.layer.<l>.{weight,bias}`. The
    ///   `layer.<l>` naming (not the task brief's guessed fixed
    ///   `conv1`/`conv2`) handles the real, variable per-branch layer
    ///   count [`ResBlockLayer`] requires (HiFi-GAN V1's 3 layers vs. V3's
    ///   1).
    ///
    /// # G2P is not loaded here
    ///
    /// This loader's 3-file signature has no piper-plus G2P GGUF — that is
    /// a wholly separate model with its own loading path (see
    /// [`SbV2Phonemizer::from_piper_g2p`]'s Task 15 real-wiring precedent,
    /// which takes an already-constructed `Box<dyn Phonemizer>` the caller
    /// owns). Rather than silently falling back to
    /// [`SbV2Phonemizer::synthetic_for_test`]'s toy char-mapping G2P — which
    /// would let a `from_gguf`-loaded model *load* successfully but
    /// *synthesize* wrong-but-plausible-looking audio for any real text
    /// (exactly the class of silent failure FR-EX-08 forbids) — the
    /// returned model's phonemizer is wired to an internal
    /// [`UnwiredPhonemizer`] stand-in whose every call is a loud
    /// [`VokraError::NotImplemented`]. A caller that needs `synthesize` to
    /// work end-to-end must instead assemble a model via [`SbV2Model::new`],
    /// passing a real phonemizer (built with
    /// [`SbV2Phonemizer::from_piper_g2p`]) in place of this method's
    /// placeholder — every other component `from_gguf` loads (text
    /// encoder, BERT, bridge, speaker, style, SDP, flow, decoder) is
    /// reusable as-is.
    ///
    /// # HiFi-GAN weight loading
    ///
    /// [`vokra_ops::hifigan::HifiGanWeights`] has no `from_gguf` of its own
    /// (unlike [`DebertaV2Encoder`]/[`DebertaV3Encoder`]) — it is a plain
    /// value bundle the M3-07 op-only WP intentionally left storage-format
    /// agnostic (`hifigan.rs`'s own doc: "the M3-07 op-only WP does not
    /// describe a storage layout"). This function reads every
    /// [`HifiGanWeights`] tensor field itself, following the `decoder.*`
    /// tensor-name scheme documented above.
    ///
    /// # Arch verification (FR-EX-08)
    ///
    /// `main`'s `vokra.model.arch` is checked against [`EXPECTED_ARCH`]
    /// **before** anything else — ahead of even the Blocker 2c
    /// format-anomaly walk. Only `main` is gated: the two BERT side-cars
    /// legitimately carry *different* tags (`deberta_v2` for `bert_ja`,
    /// `deberta_v3` for `bert_en`) and are cross-checked structurally
    /// against `vokra.sbv2.d_bert` further down. Gating those tags too —
    /// which would additionally catch a JA/EN argument swap — is a
    /// follow-up.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] when `main`'s `vokra.model.arch` is absent
    /// or is not [`EXPECTED_ARCH`].
    ///
    /// [`VokraError::ModelLoad`], naming the offending metadata key or
    /// tensor name, for any of: a missing or wrong-typed `vokra.sbv2.*`
    /// metadata key on `main`; a missing `sbv2.*` tensor on `main`; a
    /// `vokra.sbv2.d_z` that is zero or odd; a decoder array-length
    /// mismatch (`upsample_kernel_sizes`/`upsample_out_channels` vs.
    /// `upsample_rates`, or `resblock_dilation_counts` vs.
    /// `resblock_kernel_sizes`, or `resblock_dilations_flat`'s total
    /// length vs. the sum of `resblock_dilation_counts`); a
    /// `vokra.sbv2.d_bert` that disagrees with `bert_ja`'s or `bert_en`'s
    /// own loaded hidden width; or any error [`DebertaV2Encoder::from_gguf`]
    /// / [`DebertaV3Encoder::from_gguf`] / [`SbertTokenizer::from_gguf`]
    /// return while loading `bert_ja` / `bert_en`.
    pub fn from_gguf(main: &GgufFile, bert_ja: &GgufFile, bert_en: &GgufFile) -> Result<Self> {
        // Backwards-compatible default: install the [`UnwiredPhonemizer`]
        // stand-in per this method's own "G2P is not loaded here" section.
        // A caller that needs [`synthesize`](Self::synthesize) to actually
        // run must use [`from_gguf_with_phonemizer`](Self::from_gguf_with_phonemizer)
        // (Task 7) instead, passing a real
        // [`SbV2Phonemizer::from_piper_g2p`] or Task 7
        // [`SbV2Phonemizer::from_fixture`] in place of this default.
        let phonemizer = SbV2Phonemizer::from_piper_g2p(
            Box::new(UnwiredPhonemizer),
            Box::new(UnwiredPhonemizer),
            std::collections::HashMap::new(),
            std::collections::HashMap::new(),
        );
        Self::from_gguf_inner(main, bert_ja, bert_en, None, phonemizer)
    }

    // The full loader body — shared by [`from_gguf`] (which passes the
    // [`UnwiredPhonemizer`]-backed default phonemizer + `bert_zh = None`),
    // Task 7's [`from_gguf_with_phonemizer`] (which passes the caller's own
    // phonemizer + `bert_zh = None`), and WP-19's
    // [`from_gguf_with_zh_bert`] (which passes the default phonemizer +
    // `bert_zh = Some(...)`). Every step besides the two shared arguments
    // is identical, so all three public entry points share this to
    // guarantee the same error surface — a WP-19 code path change cannot
    // silently degrade the pre-WP-19 3-file loader.
    fn from_gguf_inner(
        main: &GgufFile,
        bert_ja: &GgufFile,
        bert_en: &GgufFile,
        bert_zh: Option<&GgufFile>,
        phonemizer: SbV2Phonemizer,
    ) -> Result<Self> {
        // Arch gate (FR-EX-08) — the very first thing, ahead of even the
        // Blocker 2c format-anomaly walk below, so a caller who hands a
        // sibling VITS-lineage GGUF as `main` is told *which* model they
        // actually passed instead of chasing a "missing vokra.sbv2.d_model"
        // that looks like a converter bug.
        //
        // Only `main` is gated here. The `bert_*` side-cars legitimately
        // carry three *different* arch tags (`deberta_v2` for JA,
        // `deberta_v3` for EN, `bert_base` for ZH) and are already
        // cross-checked structurally against `vokra.sbv2.d_bert` further
        // down; gating their tags too (which would also catch a JA/EN
        // argument swap) is a follow-up.
        verify_main_arch(main)?;
        // Blocker 2c defensive check (2026-08-10): the converter
        // (`crates/vokra-convert/src/models/sbv2.rs::rewrite_sdp_tensor_name`)
        // maps even-index upstream `sdp.flows.<even>.*` production tensors
        // verbatim to `sbv2.sdp.flows.<even>.*` (preserving the loud-detect
        // path for anomalies). Upstream VITS-SDP architecture puts `Flip`
        // modules at even indices — Flip has zero parameters, so a healthy
        // checkpoint should have NO `sbv2.sdp.flows.<even>.*` tensors at
        // all. The load path below reads only `sbv2.sdp.flow.<dense>.*`
        // (the densified odd-index ConvFlow slots), so any surviving
        // `sbv2.sdp.flows.*` tensor would be silently dropped without
        // this check — FR-EX-08 no-silent-wrong. If real checkpoints ever
        // legitimately ship such tensors (e.g. an SDP variant that stores
        // parameters at even slots), this check must be relaxed with a
        // recorded rationale, but until then it acts as a canary for
        // converter regressions or format corruption.
        //
        // Placed up-front (before all metadata / tensor reads) so a test
        // can exercise it without constructing a fully-loadable SDP —
        // the anomaly is a pure format check with no other dependencies.
        for t in main.tensors() {
            if let Some(rest) = t.name.strip_prefix("sbv2.sdp.flows.") {
                return Err(VokraError::ModelLoad(format!(
                    "SbV2Model::from_gguf: unexpected `sbv2.sdp.flows.{rest}` tensor — upstream \
                     VITS-SDP puts `Flip` (parameter-free) modules at even flow slots, so no \
                     `sbv2.sdp.flows.*` production tensor should reach the loader. Loaded \
                     ConvFlow tensors live under `sbv2.sdp.flow.<dense>.*`. This tensor \
                     would be silently dropped without this check (FR-EX-08). Root cause \
                     is either a converter regression in `rewrite_sdp_tensor_name` \
                     (Blocker 2c) or a checkpoint format anomaly — inspect the emitting \
                     tool before disabling this check."
                )));
            }
        }

        // ---- metadata + tensor read helpers (mirrors
        // vokra_bert::deberta_v2::DebertaV2Encoder::from_gguf's established
        // closure shape) ----
        let meta_u32 =
            |key: &str| -> Option<u32> { main.get(key).and_then(|v| v.as_u64()).map(|u| u as u32) };
        let require_u32 = |key: &str| -> Result<u32> {
            meta_u32(key).ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "SbV2Model::from_gguf: missing GGUF metadata key: {key}"
                ))
            })
        };
        // `require_f32` is the `f32` sibling of `require_u32` — SBV2's only
        // scalar float hparam today is `decoder.leaky_relu_slope`, but
        // structured this way so the pattern is reusable if the schema grows
        // (mirrors the `require_u32` / `require_array_usize` shape below).
        let require_f32 = |key: &str| -> Result<f32> {
            main.get(key)
                .and_then(|v| v.as_f64())
                .map(|f| f as f32)
                .ok_or_else(|| {
                    VokraError::ModelLoad(format!(
                        "SbV2Model::from_gguf: missing GGUF metadata key: {key}"
                    ))
                })
        };
        let require_array_usize = |key: &str| -> Result<Vec<usize>> {
            let val = main.get(key).ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "SbV2Model::from_gguf: missing GGUF metadata key: {key}"
                ))
            })?;
            let arr = val.as_array().ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "SbV2Model::from_gguf: GGUF metadata key {key} is not an array"
                ))
            })?;
            arr.values
                .iter()
                .map(|v| {
                    v.as_u64().map(|u| u as usize).ok_or_else(|| {
                        VokraError::ModelLoad(format!(
                            "SbV2Model::from_gguf: an element of GGUF metadata array {key} is \
                             not an unsigned integer"
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()
        };
        let load_tensor_f32 = |name: &str| -> Result<Vec<f32>> {
            main.tensor_f32(name).map_err(|e| {
                VokraError::ModelLoad(format!("SbV2Model::from_gguf: tensor {name}: {e}"))
            })
        };

        // ---- top-level dims ----
        let d_model = require_u32("vokra.sbv2.d_model")? as usize;
        let d_bert = require_u32("vokra.sbv2.d_bert")? as usize;
        let d_speaker = require_u32("vokra.sbv2.d_speaker")? as usize;
        let n_speakers = require_u32("vokra.sbv2.n_speakers")? as usize;
        let d_style = require_u32("vokra.sbv2.d_style")? as usize;
        let d_z = require_u32("vokra.sbv2.d_z")? as usize;
        let n_vocab = require_u32("vokra.sbv2.n_vocab")? as usize;
        let n_tones = require_u32("vokra.sbv2.n_tones")? as usize;
        let d_ff = require_u32("vokra.sbv2.d_ff")? as usize;
        let n_text_layers = require_u32("vokra.sbv2.n_text_layers")? as usize;
        // Relative-position transformer block hparams for the SBV2 text
        // encoder — the M6 refactor lifted these from the design-doc §7
        // pins (`n_heads = 2`, `window = 4`, `kernel_ffn = 3`) to real
        // metadata keys because the VITS `MultiHeadAttention` /
        // `PositionWiseFFN` are architecturally parameterized by all
        // three. The converter (`vokra-convert::models::sbv2`) stamps
        // these under the same `vokra.sbv2.*` chunk group as every
        // other hparam — see `write_hparams`. Loud-fail if absent so a
        // stale GGUF (still emitting the pre-M6 12-tensor simple
        // prenorm layer set) never quietly wires up the wrong
        // architecture.
        let n_heads = require_u32("vokra.sbv2.n_heads")? as usize;
        let window_size = require_u32("vokra.sbv2.window_size")? as usize;
        let kernel_ffn = require_u32("vokra.sbv2.kernel_ffn")? as usize;
        let n_flow_layers = require_u32("vokra.sbv2.n_flow_layers")? as usize;
        // Blocker 2b (2026-08-06) — VITS2 flow's own hparams. Independent
        // from the text encoder's own values because the flow's internal
        // `SbV2TransformerCouplingLayer.encoder_stack` uses a different
        // per-block layer count and FFN kernel width than the text
        // encoder's transformer stack (real SBV2 v2 base:
        // `n_flow_encoder_layers = 6`, `kernel_ffn_flow = 5` vs. the text
        // encoder's `kernel_ffn = 3`). See flow.rs's module doc.
        //
        // Optional to preserve backward compatibility with GGUFs converted
        // before Blocker 2b landed (which have `n_flow_layers = 0` and no
        // flow tensors — the loader still parses the empty flow stack).
        // For any `n_flow_layers > 0`, all four flow hparams are read
        // from the same `vokra.sbv2.flow.*` chunk group the converter
        // stamps alongside the flow tensor stack.
        let (n_flow_encoder_layers, kernel_ffn_flow, gin_channels, mean_only_flow) =
            if n_flow_layers > 0 {
                (
                    require_u32("vokra.sbv2.flow.n_encoder_layers")? as usize,
                    require_u32("vokra.sbv2.flow.kernel_ffn")? as usize,
                    require_u32("vokra.sbv2.flow.gin_channels")? as usize,
                    main.get("vokra.sbv2.flow.mean_only")
                        .and_then(|v| v.as_bool())
                        .ok_or_else(|| {
                            VokraError::ModelLoad(
                                "SbV2Model::from_gguf: missing GGUF metadata key: \
                             vokra.sbv2.flow.mean_only (bool)"
                                    .to_string(),
                            )
                        })?,
                )
            } else {
                // Empty flow — none of these are read; supply zeros / false
                // so no ambient hparam variation leaks into the code path.
                (0, 0, 0, false)
            };
        let n_sdp_layers = require_u32("vokra.sbv2.n_sdp_layers")? as usize;
        let sample_rate = require_u32("vokra.sbv2.sample_rate")?;
        // `decoder.leaky_relu_slope` is read up front alongside the other
        // scalar dims (rather than beside the decoder-array metadata block
        // below) so all scalar hparam reads are validated before any tensor
        // load — a missing key surfaces a named-key error instead of a
        // misleading downstream "tensor not found" (WP-13, FR-EX-08). The
        // read is required (never a silent default): while `0.1` is the
        // universal jik876/hifi-gan `LRELU_SLOPE` every sibling decoder in
        // this codebase compiles in (`vits_ja::VITS_JA_LEAKY_RELU_SLOPE`,
        // piper-plus's `LRELU_SLOPE`), the SBV2 schema treats it as a
        // per-checkpoint hparam that the converter always emits — silently
        // defaulting when the GGUF omits it would produce audio that is
        // subtly wrong (leaky-ReLU negative-slope drift) with no observable
        // signal.
        let leaky_relu_slope = require_f32("vokra.sbv2.decoder.leaky_relu_slope")?;

        // Cross-check the relative-position transformer hparams against
        // `d_model` before any tensor load — `n_heads` must divide
        // `d_model` so `d_head = d_model / n_heads` is exact, and
        // `n_heads` / `window_size` / `kernel_ffn` must all be positive.
        // A malformed metadata combination fails loudly here rather than
        // panicking inside `RelPositionMHA::new`'s debug-asserts (which
        // would only fire in debug builds — this loader promises the
        // FR-EX-08 "loud on load" property in release too).
        if n_heads == 0 {
            return Err(VokraError::ModelLoad(
                "SbV2Model::from_gguf: vokra.sbv2.n_heads must be positive".to_string(),
            ));
        }
        if window_size == 0 {
            return Err(VokraError::ModelLoad(
                "SbV2Model::from_gguf: vokra.sbv2.window_size must be positive".to_string(),
            ));
        }
        if kernel_ffn == 0 {
            return Err(VokraError::ModelLoad(
                "SbV2Model::from_gguf: vokra.sbv2.kernel_ffn must be positive".to_string(),
            ));
        }
        if d_model % n_heads != 0 {
            return Err(VokraError::ModelLoad(format!(
                "SbV2Model::from_gguf: vokra.sbv2.d_model ({d_model}) must be divisible by \
                 vokra.sbv2.n_heads ({n_heads}) — VITS `MultiHeadAttention` requires d_head = \
                 d_model / n_heads to be an exact integer"
            )));
        }
        let d_head = d_model / n_heads;

        if d_z == 0 || d_z % 2 != 0 {
            return Err(VokraError::ModelLoad(format!(
                "SbV2Model::from_gguf: vokra.sbv2.d_z must be non-zero and even (VITS2 affine \
                 coupling splits the flow latent into two equal channel halves — see \
                 SbV2Flow::from_layers), got {d_z}"
            )));
        }
        let half_d_z = d_z / 2;

        // ---- text encoder ----
        let phoneme_embed = load_tensor_f32("sbv2.text_encoder.phoneme_embed")?;
        let tone_embed = load_tensor_f32("sbv2.text_encoder.tone_embed")?;
        // M6 refactor (2026-08-06): the real base checkpoint has
        // `enc_p.language_emb.weight [3, 192]` (JA/EN/ZH), never a
        // `word_boundary_emb`. The converter now emits this under
        // `sbv2.text_encoder.language_embed`; the runtime cross-checks
        // its length is exactly N_LANGUAGES * d_model so a stale
        // converter output (still emitting the old 2-row `wb_embed`
        // shape) surfaces as a clean, loud `VokraError::ModelLoad`
        // rather than a debug-only `debug_assert!` panic inside
        // `SbV2TextEncoder::from_weights`. See `SbV2TextEncoder`'s own
        // "`language_embed` design correction" section for the full
        // rationale.
        let language_embed = load_tensor_f32("sbv2.text_encoder.language_embed")?;
        let expected_language_embed_len = N_LANGUAGES * d_model;
        if language_embed.len() != expected_language_embed_len {
            return Err(VokraError::ModelLoad(format!(
                "SbV2Model::from_gguf: sbv2.text_encoder.language_embed length {} does not \
                 match N_LANGUAGES * d_model = {N_LANGUAGES} * {d_model} = \
                 {expected_language_embed_len} (a stale converter that still emits the pre-M6 \
                 wb_embed [2, d_model] shape would land here) — reconvert the checkpoint with \
                 the current vokra-convert::models::sbv2",
                language_embed.len(),
            )));
        }
        let mut transformer_layers = Vec::with_capacity(n_text_layers);
        for i in 0..n_text_layers {
            let p = format!("sbv2.text_encoder.layer.{i}");
            // Attention Q/K/V/O — 1×1 Conv1d (kernel=1) with bias, matches
            // upstream VITS `MultiHeadAttention.conv_{q,k,v,o}` naming.
            let attn = text_encoder::RelPositionMHA::new(
                load_tensor_f32(&format!("{p}.attn.conv_q.weight"))?,
                load_tensor_f32(&format!("{p}.attn.conv_q.bias"))?,
                load_tensor_f32(&format!("{p}.attn.conv_k.weight"))?,
                load_tensor_f32(&format!("{p}.attn.conv_k.bias"))?,
                load_tensor_f32(&format!("{p}.attn.conv_v.weight"))?,
                load_tensor_f32(&format!("{p}.attn.conv_v.bias"))?,
                load_tensor_f32(&format!("{p}.attn.conv_o.weight"))?,
                load_tensor_f32(&format!("{p}.attn.conv_o.bias"))?,
                // Relative-position embeddings: `heads_share=True` (the
                // upstream default and what real SBV2 v2 uses), so a
                // single `[2*window+1, d_head]` table is broadcast
                // across every head. Upstream on-disk shape is `[1,
                // 2*window+1, d_head]`; the leading singleton is dropped
                // when this tensor lands in the flat GGUF buffer.
                load_tensor_f32(&format!("{p}.attn.rel_pos_k"))?,
                load_tensor_f32(&format!("{p}.attn.rel_pos_v"))?,
                n_heads,
                d_head,
                window_size,
            );
            // Post-attn / post-FFN residual LayerNorms (channel-last,
            // matches upstream `modules.LayerNorm`).
            let norm1 = text_encoder::LayerNorm::new(
                load_tensor_f32(&format!("{p}.norm1.gamma"))?,
                load_tensor_f32(&format!("{p}.norm1.beta"))?,
                d_model,
            );
            let ffn = text_encoder::PositionWiseFFN::new(
                load_tensor_f32(&format!("{p}.ffn.conv_1.weight"))?,
                load_tensor_f32(&format!("{p}.ffn.conv_1.bias"))?,
                load_tensor_f32(&format!("{p}.ffn.conv_2.weight"))?,
                load_tensor_f32(&format!("{p}.ffn.conv_2.bias"))?,
                d_model,
                d_ff,
                kernel_ffn,
            );
            let norm2 = text_encoder::LayerNorm::new(
                load_tensor_f32(&format!("{p}.norm2.gamma"))?,
                load_tensor_f32(&format!("{p}.norm2.beta"))?,
                d_model,
            );
            transformer_layers.push(text_encoder::SbV2TransformerBlock::new(
                attn, norm1, ffn, norm2, d_model,
            ));
        }
        let text_encoder = SbV2TextEncoder::from_weights(
            phoneme_embed,
            tone_embed,
            language_embed,
            transformer_layers,
            d_model,
            n_vocab,
            n_tones,
        );

        // ---- BERT (delegated to vokra-bert's own loaders, on the
        // separate bert_ja / bert_en files) ----
        let ja = DebertaV2Encoder::from_gguf(bert_ja)?;
        let en = DebertaV3Encoder::from_gguf(bert_en)?;
        if ja.get_d_model() != d_bert {
            return Err(VokraError::ModelLoad(format!(
                "SbV2Model::from_gguf: vokra.sbv2.d_bert ({d_bert}) disagrees with bert_ja's \
                 own hidden width ({}) — main.gguf and bert_ja.gguf were not converted together",
                ja.get_d_model()
            )));
        }
        if en.get_d_model() != d_bert {
            return Err(VokraError::ModelLoad(format!(
                "SbV2Model::from_gguf: vokra.sbv2.d_bert ({d_bert}) disagrees with bert_en's \
                 own hidden width ({}) — main.gguf and bert_en.gguf were not converted together",
                en.get_d_model()
            )));
        }
        let ja_tokenizer = SbertTokenizer::from_gguf(bert_ja, "vokra.bert.tokenizer")?;
        let en_tokenizer = SbertTokenizer::from_gguf(bert_en, "vokra.bert.tokenizer")?;

        // ---- WP-19: optional ZH BERT (plain BERT, WordPiece) ----
        //
        // `bert_zh = None` (the 3-file `from_gguf` / `from_gguf_with_phonemizer`
        // entry points) leaves `zh` / `zh_tokenizer` at `None` — the pre-WP-19
        // ZH path in `synthesize` (loud NotImplemented) survives unchanged.
        //
        // `bert_zh = Some(g)` (the 4-file `from_gguf_with_zh_bert` entry point)
        // loads a [`BertBaseEncoder`] and its paired [`BertWordpieceTokenizer`]
        // from `g` — same `d_bert`-consistency check as the JA/EN branches
        // above so the three BERT files' hidden widths must agree (a
        // mismatch means the four files were not converted together, exactly
        // the class of loader mistake FR-EX-08 wants caught at load time).
        //
        // The tokenizer prefix is `vokra.bert.wordpiece` — parallel to the
        // JA/EN SentencePiece prefix `vokra.bert.tokenizer` (`from_gguf`
        // above), but with a distinct suffix because the two tokenizer
        // families read different schema keys
        // (`{prefix}.vocab`/`unk_id`/`cls_id`/`sep_id`/`pad_id`/`do_lower_case`
        // for WordPiece vs the SentencePiece piece table).
        let (zh, zh_tokenizer) = match bert_zh {
            Some(g) => {
                let zh_enc = BertBaseEncoder::from_gguf(g).map_err(|e| {
                    VokraError::ModelLoad(format!(
                        "SbV2Model::from_gguf_with_zh_bert: bert_zh: {e}"
                    ))
                })?;
                if zh_enc.d_model() != d_bert {
                    return Err(VokraError::ModelLoad(format!(
                        "SbV2Model::from_gguf_with_zh_bert: vokra.sbv2.d_bert ({d_bert}) \
                         disagrees with bert_zh's own hidden width ({}) — main.gguf and \
                         bert_zh.gguf were not converted together",
                        zh_enc.d_model()
                    )));
                }
                let zh_tok =
                    BertWordpieceTokenizer::from_gguf(g, "vokra.bert.wordpiece").map_err(|e| {
                        VokraError::ModelLoad(format!(
                            "SbV2Model::from_gguf_with_zh_bert: bert_zh tokenizer: {e}"
                        ))
                    })?;
                (Some(zh_enc), Some(zh_tok))
            }
            None => (None, None),
        };
        let bert = SbV2BertContainer {
            ja_tokenizer,
            en_tokenizer,
            ja,
            en,
            zh,
            zh_tokenizer,
        };

        // ---- BERT bridge ----
        let bert_bridge = BertBridge::from_conv(
            load_tensor_f32("sbv2.bert_bridge.conv.weight")?,
            load_tensor_f32("sbv2.bert_bridge.conv.bias")?,
            d_bert,
            d_model,
        );

        // ---- speaker ----
        //
        // Two co-existing paths (see the `speaker` module doc for the
        // full dispatch table):
        //
        // 1. Legacy synthetic path — a converter that emits
        //    `sbv2.speaker.table` (row-major `[n_speakers, d_speaker]`)
        //    binds the [`SpeakerEmbedding::lookup`] path.
        // 2. Real-ckpt path (Blocker 3) — a converter that emits
        //    `sbv2.text_encoder.spk_emb_linear.{weight,bias}`
        //    (`[d_model, d_speaker]` + `[d_model]`, the real SBV2 v2
        //    base ckpt's `enc_p.encoder.spk_emb_linear`) binds an
        //    [`ExternalSpeakerProjection`] onto the loaded model.
        //
        // Both are optional; loud-fail if neither is present so a
        // malformed converter output fails at load time rather than
        // silently producing a model whose `synthesize` step 5 also
        // loud-fails on every call (FR-EX-08 caught earlier is better
        // caught later). If **both** are present, load both — the
        // pipeline's dispatch table then picks between them per-request
        // based on `SbV2SynthRequest::speaker_embedding`, without this
        // loader having to guess which one the caller wants.
        let table_tensor = main.tensor_f32("sbv2.speaker.table").ok();
        let spk_emb_linear_weight = main
            .tensor_f32("sbv2.text_encoder.spk_emb_linear.weight")
            .ok();
        let spk_emb_linear_bias = main
            .tensor_f32("sbv2.text_encoder.spk_emb_linear.bias")
            .ok();
        // The `spk_emb_linear.{weight,bias}` pair is all-or-nothing —
        // one present without the other is a converter bug, not a
        // partial legacy fallback.
        let projection = match (spk_emb_linear_weight, spk_emb_linear_bias) {
            (Some(w), Some(b)) => {
                if w.len() != d_model * d_speaker {
                    return Err(VokraError::ModelLoad(format!(
                        "SbV2Model::from_gguf: sbv2.text_encoder.spk_emb_linear.weight has \
                         {} elements, expected d_model * d_speaker = {} * {} = {}",
                        w.len(),
                        d_model,
                        d_speaker,
                        d_model * d_speaker,
                    )));
                }
                if b.len() != d_model {
                    return Err(VokraError::ModelLoad(format!(
                        "SbV2Model::from_gguf: sbv2.text_encoder.spk_emb_linear.bias has {} \
                         elements, expected d_model = {}",
                        b.len(),
                        d_model,
                    )));
                }
                Some(ExternalSpeakerProjection::from_weights(
                    w, b, d_speaker, d_model,
                ))
            }
            (None, None) => None,
            (Some(_), None) => {
                return Err(VokraError::ModelLoad(
                    "SbV2Model::from_gguf: sbv2.text_encoder.spk_emb_linear.weight is present \
                     but its .bias is missing — a converter must emit both or neither"
                        .to_string(),
                ));
            }
            (None, Some(_)) => {
                return Err(VokraError::ModelLoad(
                    "SbV2Model::from_gguf: sbv2.text_encoder.spk_emb_linear.bias is present \
                     but its .weight is missing — a converter must emit both or neither"
                        .to_string(),
                ));
            }
        };
        // Bind the legacy `SpeakerEmbedding` if the tensor is present;
        // otherwise install a `[1, d_speaker]` all-zero placeholder that
        // is never reached (the `synthesize` step-5 dispatch routes
        // through `projection` when `Some`, never the placeholder). The
        // placeholder is required only because `SbV2Model`'s
        // `speaker_embed` field is not `Option` — see the field's own
        // doc for why (backward compat with `synthetic_for_test`'s
        // 9-argument construction).
        let speaker_embed = match (table_tensor, &projection) {
            (Some(t), _) => SpeakerEmbedding::from_table(t, n_speakers, d_speaker),
            (None, Some(_)) => {
                // Never reached at request time — kept as a shape-valid
                // placeholder so this field's type invariant holds.
                SpeakerEmbedding::from_table(vec![0.0_f32; d_speaker], 1, d_speaker)
            }
            (None, None) => {
                return Err(VokraError::ModelLoad(
                    "SbV2Model::from_gguf: neither sbv2.speaker.table (legacy lookup path) \
                     nor sbv2.text_encoder.spk_emb_linear.{weight,bias} (real-ckpt \
                     projection path) is present — a converter must emit one of the two \
                     speaker-conditioning shapes (FR-EX-08: no silent all-zero speaker \
                     fallback)"
                        .to_string(),
                ));
            }
        };

        // ---- style ----
        //
        // Blocker 6 (2026-08-06): the SBV2 v2 base checkpoint
        // (`litagin/Style-Bert-VITS2-2.0-base-JP-Extra`) ships **no**
        // `style_injector.*` tensors. The style-vector projection is
        // learned during per-speaker fine-tuning; base ckpt inference is
        // an identity injector (equivalent to `style_vec=0`). Post-
        // Blocker-6 the loader falls back to all-zero `proj_scale` +
        // `proj_bias` when both tensors are absent — [`inject`](style::
        // StyleVectorInjector::inject) then computes `h = h * (1 + 0) +
        // 0 = h`, a byte-identity no-op. A converter that emits **only
        // one** of the two projections (impossible on real upstream, but
        // possible on a hand-authored fixture) still errors loudly (that
        // would be a converter bug, not an absent-fixture default).
        let style_injector = match (
            main.get("sbv2.style_injector.proj_scale"),
            main.get("sbv2.style_injector.proj_bias"),
        ) {
            (Some(_), Some(_)) => StyleVectorInjector::from_projections(
                load_tensor_f32("sbv2.style_injector.proj_scale")?,
                load_tensor_f32("sbv2.style_injector.proj_bias")?,
                d_style,
                d_model,
            ),
            (None, None) => {
                // Zero-weight identity fallback — equivalent to per-
                // utterance `style_vec = 0` regardless of caller's input.
                StyleVectorInjector::from_projections(
                    vec![0.0_f32; d_model * d_style],
                    vec![0.0_f32; d_model * d_style],
                    d_style,
                    d_model,
                )
            }
            (Some(_), None) | (None, Some(_)) => {
                return Err(VokraError::ModelLoad(
                    "SbV2Model::from_gguf: exactly one of sbv2.style_injector.\
                     {proj_scale, proj_bias} is present — a converter must \
                     emit both or neither (FR-EX-08: partial style projection \
                     is undefined)"
                        .to_string(),
                ));
            }
        };

        // ---- stochastic duration predictor (Blocker 2c: real DDS-net +
        // rational-quadratic-spline ConvFlow shape, mirroring the 144
        // production tensors under `sdp.*` — the sibling 142 `sdp.post_*`
        // training-side inverse-flow tensors the converter skipped are
        // never read here). See `SbV2SDP::from_weights`'s doc for every
        // field's shape, and `duration.rs`'s module doc for the tensor
        // layout convention. `n_sdp_layers` is now the ConvFlow count —
        // real SBV2 v2 base = 4, empty-SDP synthetic path = 0. The tensor
        // path scheme `sbv2.sdp.flow.<i>.*` is a dense re-index of
        // upstream's sparse `sdp.flows.{1,3,5,7}` (the sparse indices
        // arise from the interleaved `Flip` layers, which carry no
        // parameters and are recreated at inference time). The
        // `ElementwiseAffine` at upstream `sdp.flows.0` maps to
        // `sbv2.sdp.ea.{m,logs}`. See the sbv2 converter's `sdp.*`
        // rewriter arm for the exact index remapping.)
        let sdp = {
            let load_dds = |prefix: &str, channels: usize| -> Result<duration::DDSConv> {
                let mut convs_sep_w = Vec::with_capacity(duration::DP_CONV_LAYERS);
                let mut convs_sep_b = Vec::with_capacity(duration::DP_CONV_LAYERS);
                let mut convs_1x1_w = Vec::with_capacity(duration::DP_CONV_LAYERS);
                let mut convs_1x1_b = Vec::with_capacity(duration::DP_CONV_LAYERS);
                let mut norms_1 = Vec::with_capacity(duration::DP_CONV_LAYERS);
                let mut norms_2 = Vec::with_capacity(duration::DP_CONV_LAYERS);
                for i in 0..duration::DP_CONV_LAYERS {
                    convs_sep_w.push(load_tensor_f32(&format!("{prefix}.convs_sep.{i}.weight"))?);
                    convs_sep_b.push(load_tensor_f32(&format!("{prefix}.convs_sep.{i}.bias"))?);
                    convs_1x1_w.push(load_tensor_f32(&format!("{prefix}.convs_1x1.{i}.weight"))?);
                    convs_1x1_b.push(load_tensor_f32(&format!("{prefix}.convs_1x1.{i}.bias"))?);
                    norms_1.push(duration::SdpLayerNorm {
                        gamma: load_tensor_f32(&format!("{prefix}.norms_1.{i}.gamma"))?,
                        beta: load_tensor_f32(&format!("{prefix}.norms_1.{i}.beta"))?,
                    });
                    norms_2.push(duration::SdpLayerNorm {
                        gamma: load_tensor_f32(&format!("{prefix}.norms_2.{i}.gamma"))?,
                        beta: load_tensor_f32(&format!("{prefix}.norms_2.{i}.beta"))?,
                    });
                }
                Ok(duration::DDSConv::from_weights(
                    channels,
                    duration::DP_CONV_LAYERS,
                    duration::DP_KERNEL,
                    convs_sep_w,
                    convs_sep_b,
                    convs_1x1_w,
                    convs_1x1_b,
                    norms_1,
                    norms_2,
                ))
            };
            let body_convs = load_dds("sbv2.sdp.convs", d_model)?;
            let ea = duration::ElementwiseAffine::from_weights(
                load_tensor_f32("sbv2.sdp.ea.m")?,
                load_tensor_f32("sbv2.sdp.ea.logs")?,
            );
            let mut flows = Vec::with_capacity(n_sdp_layers);
            for i in 0..n_sdp_layers {
                let p = format!("sbv2.sdp.flow.{i}");
                let convs = load_dds(&format!("{p}.convs"), d_model)?;
                flows.push(duration::ConvFlow::from_weights(
                    load_tensor_f32(&format!("{p}.pre.weight"))?,
                    load_tensor_f32(&format!("{p}.pre.bias"))?,
                    convs,
                    load_tensor_f32(&format!("{p}.proj.weight"))?,
                    load_tensor_f32(&format!("{p}.proj.bias"))?,
                    d_model,
                ));
            }
            duration::SbV2SDP::from_weights(
                d_model,
                d_speaker, // gin = d_speaker in real SBV2 v2 (both = 512)
                load_tensor_f32("sbv2.sdp.pre.weight")?,
                load_tensor_f32("sbv2.sdp.pre.bias")?,
                body_convs,
                load_tensor_f32("sbv2.sdp.cond.weight")?,
                load_tensor_f32("sbv2.sdp.cond.bias")?,
                load_tensor_f32("sbv2.sdp.proj.weight")?,
                load_tensor_f32("sbv2.sdp.proj.bias")?,
                ea,
                flows,
            )
        };

        // Blocker 2c defensive check (2026-08-10): the converter
        // (`crates/vokra-convert/src/models/sbv2.rs::rewrite_sdp_tensor_name`)
        // maps even-index upstream `sdp.flows.<even>.*` production tensors
        // verbatim to `sbv2.sdp.flows.<even>.*` (preserving the loud-detect
        // path for anomalies). Upstream VITS-SDP architecture puts `Flip`
        // modules at even indices — Flip has zero parameters, so a healthy
        // checkpoint should have NO `sbv2.sdp.flows.<even>.*` tensors at
        // all. The load path above reads only `sbv2.sdp.flow.<dense>.*`
        // (the densified odd-index ConvFlow slots), so any surviving
        // `sbv2.sdp.flows.*` tensor would be silently dropped without
        // this check — FR-EX-08 no-silent-wrong. If real checkpoints ever
        // legitimately ship such tensors (e.g. an SDP variant that stores
        // parameters at even slots), this check must be relaxed with a
        // recorded rationale, but until then it acts as a canary for
        // converter regressions or format corruption.
        // ---- VITS2 normalizing flow (Blocker 2b, 2026-08-06) ----
        //
        // Upstream `p0p4k/vits2_pytorch/models.TransformerCouplingBlock`
        // stores the flow as a flat `nn.ModuleList` of length `2 *
        // n_flow_layers`, alternating `TransformerCouplingLayer` (even
        // indices) and `Flip` (odd indices). We rebuild that layout here
        // by pushing one `FlowLayer::Coupling(...)` followed by one
        // `FlowLayer::Flip` per `n_flow_layers` — see the `SbV2Flow`
        // struct doc for the invariant.
        //
        // `d_hidden` is the coupling's inner transformer stack width
        // (upstream `hidden_channels`, = `d_model` on the SBV2 v2 base).
        // Bound to `d_model` here — the runtime uses the same value; a
        // future SKU shipping `d_hidden != d_model` would need a distinct
        // `vokra.sbv2.flow.d_hidden` key.
        let d_hidden_flow = d_model;
        // COSMETIC-BUNDLE (2026-08-09): the pre-fix `.checked_div(...).unwrap_or(0)`
        // silently returned `0` when `n_heads == 0` (letting a wrong-shape
        // downstream `RelPositionMHA::new` fire only under `debug_assert!` —
        // release builds silently produced garbage). `derive_flow_head_dim`'s
        // `.expect(...)` is the loud-fail replacement; the `n_heads == 0`
        // guard at the top of this fn (`vokra.sbv2.n_heads must be positive`)
        // means the panic is unreachable on any successful load — this is
        // defence in depth (FR-EX-08).
        let d_head_flow = derive_flow_head_dim(d_model, n_heads);
        let mut flow_stack: Vec<FlowLayer> = Vec::with_capacity(n_flow_layers * 2);
        for i in 0..n_flow_layers {
            let p = format!("sbv2.flow.layer.{i}");
            let pre_weight = load_tensor_f32(&format!("{p}.pre.weight"))?;
            let pre_bias = load_tensor_f32(&format!("{p}.pre.bias"))?;
            let spk_emb_weight = load_tensor_f32(&format!("{p}.spk_emb.weight"))?;
            let spk_emb_bias = load_tensor_f32(&format!("{p}.spk_emb.bias"))?;
            let post_weight = load_tensor_f32(&format!("{p}.post.weight"))?;
            let post_bias = load_tensor_f32(&format!("{p}.post.bias"))?;

            // Build the flow-encoder stack for this coupling — same block
            // type as the text encoder's own stack, but distinct hparams
            // (see `flow.rs`'s doc: `n_flow_encoder_layers`, `kernel_ffn_flow`).
            let mut encoder_stack = Vec::with_capacity(n_flow_encoder_layers);
            for j in 0..n_flow_encoder_layers {
                let ep = format!("{p}.enc.{j}");
                let attn = text_encoder::RelPositionMHA::new(
                    load_tensor_f32(&format!("{ep}.attn.conv_q.weight"))?,
                    load_tensor_f32(&format!("{ep}.attn.conv_q.bias"))?,
                    load_tensor_f32(&format!("{ep}.attn.conv_k.weight"))?,
                    load_tensor_f32(&format!("{ep}.attn.conv_k.bias"))?,
                    load_tensor_f32(&format!("{ep}.attn.conv_v.weight"))?,
                    load_tensor_f32(&format!("{ep}.attn.conv_v.bias"))?,
                    load_tensor_f32(&format!("{ep}.attn.conv_o.weight"))?,
                    load_tensor_f32(&format!("{ep}.attn.conv_o.bias"))?,
                    load_tensor_f32(&format!("{ep}.attn.rel_pos_k"))?,
                    load_tensor_f32(&format!("{ep}.attn.rel_pos_v"))?,
                    n_heads,
                    d_head_flow,
                    window_size,
                );
                let norm1 = text_encoder::LayerNorm::new(
                    load_tensor_f32(&format!("{ep}.norm1.gamma"))?,
                    load_tensor_f32(&format!("{ep}.norm1.beta"))?,
                    d_hidden_flow,
                );
                let ffn = text_encoder::PositionWiseFFN::new(
                    load_tensor_f32(&format!("{ep}.ffn.conv_1.weight"))?,
                    load_tensor_f32(&format!("{ep}.ffn.conv_1.bias"))?,
                    load_tensor_f32(&format!("{ep}.ffn.conv_2.weight"))?,
                    load_tensor_f32(&format!("{ep}.ffn.conv_2.bias"))?,
                    d_hidden_flow,
                    d_ff,
                    kernel_ffn_flow,
                );
                let norm2 = text_encoder::LayerNorm::new(
                    load_tensor_f32(&format!("{ep}.norm2.gamma"))?,
                    load_tensor_f32(&format!("{ep}.norm2.beta"))?,
                    d_hidden_flow,
                );
                encoder_stack.push(text_encoder::SbV2TransformerBlock::new(
                    attn,
                    norm1,
                    ffn,
                    norm2,
                    d_hidden_flow,
                ));
            }

            let tcl = flow::SbV2TransformerCouplingLayer::from_weights(
                pre_weight,
                pre_bias,
                spk_emb_weight,
                spk_emb_bias,
                encoder_stack,
                post_weight,
                post_bias,
                half_d_z,
                d_hidden_flow,
                gin_channels,
                mean_only_flow,
            );
            flow_stack.push(FlowLayer::Coupling(tcl));
            flow_stack.push(FlowLayer::Flip);
        }
        let flow = SbV2Flow::from_layers(flow_stack, d_z);

        // ---- HiFi-GAN decoder ----
        let initial_channel = require_u32("vokra.sbv2.decoder.initial_channel")? as usize;
        let conv_pre_kernel = require_u32("vokra.sbv2.decoder.conv_pre_kernel")? as usize;
        let conv_post_kernel = require_u32("vokra.sbv2.decoder.conv_post_kernel")? as usize;
        let upsample_rates = require_array_usize("vokra.sbv2.decoder.upsample_rates")?;
        let upsample_kernel_sizes =
            require_array_usize("vokra.sbv2.decoder.upsample_kernel_sizes")?;
        let upsample_out_channels =
            require_array_usize("vokra.sbv2.decoder.upsample_out_channels")?;
        let resblock_kernel_sizes =
            require_array_usize("vokra.sbv2.decoder.resblock_kernel_sizes")?;
        let resblock_dilation_counts =
            require_array_usize("vokra.sbv2.decoder.resblock_dilation_counts")?;
        let resblock_dilations_flat =
            require_array_usize("vokra.sbv2.decoder.resblock_dilations_flat")?;
        // `leaky_relu_slope` was read up front alongside the top-level scalar
        // dims — see that read's comment for the FR-EX-08 rationale.

        let n_upsample_stages = upsample_rates.len();
        if upsample_kernel_sizes.len() != n_upsample_stages {
            return Err(VokraError::ModelLoad(format!(
                "SbV2Model::from_gguf: vokra.sbv2.decoder.upsample_kernel_sizes.len() ({}) != \
                 vokra.sbv2.decoder.upsample_rates.len() ({n_upsample_stages})",
                upsample_kernel_sizes.len()
            )));
        }
        if upsample_out_channels.len() != n_upsample_stages {
            return Err(VokraError::ModelLoad(format!(
                "SbV2Model::from_gguf: vokra.sbv2.decoder.upsample_out_channels.len() ({}) != \
                 vokra.sbv2.decoder.upsample_rates.len() ({n_upsample_stages})",
                upsample_out_channels.len()
            )));
        }
        let n_mrf_branches = resblock_kernel_sizes.len();
        if resblock_dilation_counts.len() != n_mrf_branches {
            return Err(VokraError::ModelLoad(format!(
                "SbV2Model::from_gguf: vokra.sbv2.decoder.resblock_dilation_counts.len() ({}) \
                 != vokra.sbv2.decoder.resblock_kernel_sizes.len() ({n_mrf_branches})",
                resblock_dilation_counts.len()
            )));
        }

        let mut resblock_dilation_sizes = Vec::with_capacity(n_mrf_branches);
        let mut cursor = 0usize;
        for &count in &resblock_dilation_counts {
            let end = cursor + count;
            if end > resblock_dilations_flat.len() {
                return Err(VokraError::ModelLoad(format!(
                    "SbV2Model::from_gguf: vokra.sbv2.decoder.resblock_dilations_flat.len() \
                     ({}) is shorter than the sum of resblock_dilation_counts (needs at least \
                     {end})",
                    resblock_dilations_flat.len()
                )));
            }
            resblock_dilation_sizes.push(resblock_dilations_flat[cursor..end].to_vec());
            cursor = end;
        }
        if cursor != resblock_dilations_flat.len() {
            return Err(VokraError::ModelLoad(format!(
                "SbV2Model::from_gguf: vokra.sbv2.decoder.resblock_dilations_flat has {} \
                 trailing element(s) beyond the sum of resblock_dilation_counts ({cursor})",
                resblock_dilations_flat.len() - cursor
            )));
        }

        let attrs = HifiGanAttrs {
            n_mels: d_z,
            initial_channel,
            upsample_rates: upsample_rates.clone(),
            upsample_kernel_sizes: upsample_kernel_sizes.clone(),
            resblock_kernel_sizes: resblock_kernel_sizes.clone(),
            resblock_dilation_sizes: resblock_dilation_sizes.clone(),
            sample_rate,
            leaky_relu_slope,
            // SBV2 v2 base checkpoint uses `resblock='1'` (ResBlock1)
            // per upstream `tools/parity/vendor/vits/modules.py` +
            // `docs/superpowers/specs/2026-07-26-sbv2-v2-design.md` §7.
            // The `weight_c2 + bias_c2` loader below requires this to
            // be V1 — a mismatch would loudly fail via
            // `mrf_branch_forward`'s FR-EX-08 gate.
            res_block_type: ResBlockType::V1,
        };

        let conv_pre_weight = load_tensor_f32("sbv2.decoder.conv_pre.weight")?;
        let conv_pre_bias = load_tensor_f32("sbv2.decoder.conv_pre.bias")?;

        let mut in_ch = initial_channel;
        let mut upsample_weights = Vec::with_capacity(n_upsample_stages);
        let mut mrf_stage_weights = Vec::with_capacity(n_upsample_stages);
        for stage in 0..n_upsample_stages {
            let out_ch = upsample_out_channels[stage];
            upsample_weights.push(UpsampleStageWeights {
                weight: load_tensor_f32(&format!("sbv2.decoder.upsample.{stage}.weight"))?,
                bias: load_tensor_f32(&format!("sbv2.decoder.upsample.{stage}.bias"))?,
                in_ch,
                out_ch,
                kernel: upsample_kernel_sizes[stage],
                stride: upsample_rates[stage],
            });

            let mut branches = Vec::with_capacity(n_mrf_branches);
            for (branch, dilations) in resblock_dilation_sizes.iter().enumerate() {
                let kernel = resblock_kernel_sizes[branch];
                let mut layers = Vec::with_capacity(dilations.len());
                for (layer, &dilation) in dilations.iter().enumerate() {
                    let p = format!("sbv2.decoder.mrf.{stage}.{branch}.layer.{layer}");
                    // HGAN-01 fix (Wave 2, 2026-08-09): load convs2.*
                    // — the undilated conv chain the upstream ResBlock1
                    // forward pairs with each convs1 iteration
                    // (`tools/parity/vendor/vits/modules.py:254 for (c1,
                    // c2) in zip(self.convs1, self.convs2)`). Pre-fix
                    // the converter passed convs2 through unread and
                    // the loader had no `weight_c2` slot; half the
                    // vocoder convs were silently dropped, guaranteeing
                    // SBV2 v2 waveform parity would never converge.
                    // Loader now requires both convs2.weight and
                    // convs2.bias — a converter that emits neither is
                    // a bug (loud `VokraError::ModelLoad`, FR-EX-08).
                    layers.push(ResBlockLayer {
                        weight: load_tensor_f32(&format!("{p}.weight"))?,
                        bias: load_tensor_f32(&format!("{p}.bias"))?,
                        weight_c2: Some(load_tensor_f32(&format!("{p}.weight_c2"))?),
                        bias_c2: Some(load_tensor_f32(&format!("{p}.bias_c2"))?),
                        dilation,
                        kernel,
                        channels: out_ch,
                    });
                }
                branches.push(MrfBranchWeights { layers });
            }
            mrf_stage_weights.push(branches);
            in_ch = out_ch;
        }

        let conv_post_weight = load_tensor_f32("sbv2.decoder.conv_post.weight")?;
        // Blocker 6 (2026-08-06): `dec.conv_post.bias` is absent in the
        // real SBV2 v2 base checkpoint (upstream ships `dec.conv_post.
        // weight [1, 16, 7]` only — the bias slot is fused into the
        // preceding tanh nonlinearity or trained to zero-equivalent).
        // The Rust HiFi-GAN decoder expects a `conv_post_bias` slot in
        // its `HifiGanWeights` struct; the safe fallback is an all-zero
        // `[out_channels=1]` bias buffer, which computes `out = conv(x)
        // + 0 = conv(x)` = the identity behavior real inference expects.
        // A synthetic fixture that supplies the bias still overrides
        // this fallback.
        let conv_post_bias = if main.get("sbv2.decoder.conv_post.bias").is_some() {
            load_tensor_f32("sbv2.decoder.conv_post.bias")?
        } else {
            // `conv_post_weight` shape is `[out_channels, in_channels,
            // kernel]` — out_channels lives at index 0 of the tensor's
            // metadata shape. We assume `out_channels = 1` (HiFi-GAN
            // convention: single-channel waveform output); a mismatch
            // would surface downstream in `SbV2Decoder::generate`'s
            // dimension checks.
            vec![0.0_f32; 1]
        };

        // HGAN-05-GIN-COND (2026-08-09): load the optional
        // `sbv2.decoder.cond.{weight,bias}` pair. Upstream
        // `dec.cond` is a `Conv1d(gin_channels, initial_channel, 1)`
        // — the multi-speaker HiFi-GAN's speaker-conditioning layer.
        // Present on SBV2 v2 base ckpt (`d_speaker = 512`); absent on
        // single-speaker fixtures / synthetic tests.
        //
        // All-or-nothing: one present without the other is a
        // converter bug (loud `VokraError::ModelLoad`, FR-EX-08).
        let cond_weight = main.get("sbv2.decoder.cond.weight").is_some();
        let cond_bias = main.get("sbv2.decoder.cond.bias").is_some();
        let cond = match (cond_weight, cond_bias) {
            (true, true) => {
                let cond_weight_vec = load_tensor_f32("sbv2.decoder.cond.weight")?;
                let cond_bias_vec = load_tensor_f32("sbv2.decoder.cond.bias")?;
                // Bias length pins initial_channel; weight length
                // determines gin_channels via
                // `weight.len() = initial_channel * gin_channels`.
                if cond_bias_vec.len() != initial_channel {
                    return Err(VokraError::ModelLoad(format!(
                        "SbV2Model::from_gguf: sbv2.decoder.cond.bias has {} elements, expected \
                         initial_channel = {}",
                        cond_bias_vec.len(),
                        initial_channel,
                    )));
                }
                if cond_weight_vec.len() % initial_channel != 0 {
                    return Err(VokraError::ModelLoad(format!(
                        "SbV2Model::from_gguf: sbv2.decoder.cond.weight length {} is not a \
                         multiple of initial_channel {} — expected shape \
                         [initial_channel, gin_channels, 1]",
                        cond_weight_vec.len(),
                        initial_channel,
                    )));
                }
                let gin_channels = cond_weight_vec.len() / initial_channel;
                if gin_channels == 0 {
                    return Err(VokraError::ModelLoad(
                        "SbV2Model::from_gguf: sbv2.decoder.cond.weight implies gin_channels = 0 \
                         — a zero-input cond layer is upstream's way of representing \
                         no-cond-layer; a converter that observed dec.cond.* must not emit an \
                         empty weight"
                            .to_owned(),
                    ));
                }
                Some(GinCondition {
                    weight: cond_weight_vec,
                    bias: cond_bias_vec,
                    gin_channels,
                })
            }
            (false, false) => None,
            (true, false) => {
                return Err(VokraError::ModelLoad(
                    "SbV2Model::from_gguf: sbv2.decoder.cond.weight is present but its .bias \
                     is missing — a converter must emit both or neither"
                        .to_owned(),
                ));
            }
            (false, true) => {
                return Err(VokraError::ModelLoad(
                    "SbV2Model::from_gguf: sbv2.decoder.cond.bias is present but its .weight \
                     is missing — a converter must emit both or neither"
                        .to_owned(),
                ));
            }
        };

        let weights = HifiGanWeights {
            conv_pre_weight,
            conv_pre_bias,
            conv_pre_kernel,
            upsample_weights,
            mrf_stage_weights,
            conv_post_weight,
            conv_post_bias,
            conv_post_kernel,
            cond,
        };
        let decoder = SbV2Decoder::new(weights, attrs, HifiGanConfig::fp32(), sample_rate);

        // ---- phonemizer — supplied by the caller (see this fn's own doc's
        // "G2P is not loaded here" section for why loading a G2P here isn't
        // this signature's job). `SbV2Model::from_gguf` passes an
        // `UnwiredPhonemizer`-backed placeholder; `from_gguf_with_phonemizer`
        // passes the caller's own `SbV2Phonemizer` (real piper-plus G2P via
        // `from_piper_g2p`, or a Task 7 parity `PhonemizeFixture` via
        // `from_fixture`).

        let mut model = Self::new(
            phonemizer,
            text_encoder,
            bert,
            bert_bridge,
            speaker_embed,
            style_injector,
            sdp,
            flow,
            decoder,
        );
        // Blocker 3: attach the real-ckpt speaker projection if the
        // loader found the `sbv2.text_encoder.spk_emb_linear.*` pair;
        // otherwise the model stays on the legacy `SpeakerEmbedding`
        // lookup path (see the `speaker` module doc's dispatch table).
        if let Some(proj) = projection {
            model = model.with_external_speaker_projection(proj);
        }
        Ok(model)
    }

    /// Task 7 sibling of [`from_gguf`](Self::from_gguf) that lets the caller
    /// substitute the [`SbV2Phonemizer`] the loaded model carries, instead
    /// of accepting the [`from_gguf`](Self::from_gguf) default that
    /// [`synthesize`](Self::synthesize) refuses to run.
    ///
    /// # Why this exists
    ///
    /// [`from_gguf`](Self::from_gguf)'s 3-file signature (`main` +
    /// `bert_ja` + `bert_en`) has no piper-plus G2P GGUF — that is a wholly
    /// separate model with its own loading path (see
    /// [`SbV2Phonemizer::from_piper_g2p`]'s Task 15 precedent). The default
    /// loader installs an internal `UnwiredPhonemizer` whose every call is a
    /// loud [`VokraError::NotImplemented`] (never a silent fall-through to
    /// [`SbV2Phonemizer::synthetic_for_test`]'s toy char-mapping —
    /// exactly the class of silent failure FR-EX-08 forbids). That default
    /// is safe but blocks [`synthesize`](Self::synthesize) for callers that
    /// have a real G2P.
    ///
    /// This constructor lets those callers pass their own
    /// [`SbV2Phonemizer`] in place of the placeholder — every other
    /// component ([`from_gguf`](Self::from_gguf) already loads (text
    /// encoder, BERT, bridge, speaker, style, SDP, flow, decoder) is
    /// reusable as-is.
    ///
    /// # Two supported call sites (per [`SbV2Phonemizer`]'s three construction paths)
    ///
    /// 1. **Production**: pass a real
    ///    [`SbV2Phonemizer::from_piper_g2p`]-built phonemizer whose
    ///    `Box<dyn Phonemizer>` implementations come from the
    ///    excluded-workspace 8-language `integrations/vokra-piper-g2p` crate
    ///    (see that integration's own docs for how to construct one; it
    ///    lives outside this crate to preserve zero-dependency NFR-DS-02).
    /// 2. **Parity fixture (Task 28)**: pass a Task 7
    ///    [`SbV2Phonemizer::from_fixture`]-built phonemizer whose
    ///    [`PhonemizeFixture`] holds the exact ids the permissive Python
    ///    reference dumper (Task 30's
    ///    `tools/parity/sbv2_dump_reference.py`) fed the reference forward
    ///    pass for a fixed set of test sentences. This is what
    ///    `crates/vokra-models/tests/parity_sbv2_real.rs` uses to unblock
    ///    the end-to-end numeric-parity assertion without needing a real
    ///    8-language G2P available in-workspace.
    ///
    /// # Errors
    ///
    /// Every error [`from_gguf`](Self::from_gguf) itself returns
    /// (`VokraError::ModelLoad` for a missing or wrong-typed
    /// `vokra.sbv2.*` metadata key, a missing `sbv2.*` tensor, a
    /// `d_z`-consistency failure, a decoder array-length mismatch, a
    /// `d_bert` mismatch across the three files, or any error the delegated
    /// [`DebertaV2Encoder::from_gguf`] / [`DebertaV3Encoder::from_gguf`] /
    /// [`SbertTokenizer::from_gguf`] return). This constructor and
    /// [`from_gguf`](Self::from_gguf) share their entire loader body via a
    /// private `from_gguf_inner`, so the error surfaces of the two are
    /// identical.
    pub fn from_gguf_with_phonemizer(
        main: &GgufFile,
        bert_ja: &GgufFile,
        bert_en: &GgufFile,
        phonemizer: SbV2Phonemizer,
    ) -> Result<Self> {
        Self::from_gguf_inner(main, bert_ja, bert_en, None, phonemizer)
    }

    /// WP-19 4-file loader — [`from_gguf`](Self::from_gguf)'s sibling that
    /// also loads a ZH BERT branch (plain BERT +
    /// [`BertWordpieceTokenizer`], per owner decision 2026-08-09:
    /// `hfl/chinese-roberta-wwm-ext-large`, Apache-2.0). The 3-file
    /// [`from_gguf`](Self::from_gguf) signature is preserved unchanged for
    /// backward compatibility — a caller with only JA/EN GGUFs keeps
    /// working; a caller with a ZH GGUF calls this instead.
    ///
    /// # Arguments
    ///
    /// - `main` — the SBV2 pipeline GGUF (text encoder, BERT bridge,
    ///   speaker, style, SDP, flow, decoder).
    /// - `bert_ja` — JA BERT (DeBERTa v2) — same file the 3-file loader
    ///   takes.
    /// - `bert_en` — EN BERT (DeBERTa v3) — same file the 3-file loader
    ///   takes.
    /// - `bert_zh` — ZH BERT (plain BERT — [`BertBaseEncoder`] +
    ///   [`BertWordpieceTokenizer`]), populating
    ///   [`SbV2BertContainer::zh`] and [`SbV2BertContainer::zh_tokenizer`].
    ///
    /// # G2P
    ///
    /// This loader installs the same [`UnwiredPhonemizer`]-backed
    /// placeholder [`from_gguf`](Self::from_gguf) does. A caller that
    /// already owns a production or fixture-driven G2P uses
    /// [`from_gguf_with_zh_bert_and_phonemizer`](Self::from_gguf_with_zh_bert_and_phonemizer)
    /// instead. The two entry points share the same private loader body;
    /// they differ only in which phonemizer is attached.
    ///
    /// # Errors
    ///
    /// Every error [`from_gguf`](Self::from_gguf) itself returns, plus
    /// [`VokraError::ModelLoad`] on any of the following ZH-specific
    /// mistakes:
    ///
    /// - `bert_zh` fails [`BertBaseEncoder::from_gguf`]'s own load (a
    ///   missing `vokra.bert_base.*` metadata key, a shape mismatch, or
    ///   any error the delegated loader surfaces).
    /// - `bert_zh` fails [`BertWordpieceTokenizer::from_gguf`]'s own load
    ///   under the `vokra.bert.wordpiece` prefix.
    /// - `bert_zh`'s hidden width disagrees with
    ///   `main`'s `vokra.sbv2.d_bert` — same class of mistake the JA/EN
    ///   branches catch, extended to the ZH branch.
    pub fn from_gguf_with_zh_bert(
        main: &GgufFile,
        bert_ja: &GgufFile,
        bert_en: &GgufFile,
        bert_zh: &GgufFile,
    ) -> Result<Self> {
        let phonemizer = SbV2Phonemizer::from_piper_g2p(
            Box::new(UnwiredPhonemizer),
            Box::new(UnwiredPhonemizer),
            std::collections::HashMap::new(),
            std::collections::HashMap::new(),
        );
        Self::from_gguf_with_zh_bert_and_phonemizer(main, bert_ja, bert_en, bert_zh, phonemizer)
    }

    /// Four-file ZH loader with an explicitly supplied phonemizer.
    ///
    /// This is the composition of
    /// [`from_gguf_with_zh_bert`](Self::from_gguf_with_zh_bert) and
    /// [`from_gguf_with_phonemizer`](Self::from_gguf_with_phonemizer): it
    /// loads the SBV2 main checkpoint plus JA, EN, and ZH BERT GGUFs, while
    /// attaching the caller's production or fixture-backed
    /// [`SbV2Phonemizer`]. It exists so the real ZH parity harness can run a
    /// complete `Language::ZH` request without weakening either fail-closed
    /// default:
    ///
    /// - the legacy three-file loader still has no implicit ZH BERT;
    /// - the four-file convenience loader still has no implicit G2P.
    ///
    /// # Errors
    ///
    /// Returns every error documented by
    /// [`from_gguf_with_zh_bert`](Self::from_gguf_with_zh_bert), including
    /// ZH encoder/tokenizer load failures and hidden-width disagreement.
    pub fn from_gguf_with_zh_bert_and_phonemizer(
        main: &GgufFile,
        bert_ja: &GgufFile,
        bert_en: &GgufFile,
        bert_zh: &GgufFile,
        phonemizer: SbV2Phonemizer,
    ) -> Result<Self> {
        Self::from_gguf_inner(main, bert_ja, bert_en, Some(bert_zh), phonemizer)
    }
}

/// Deterministic, small HiFi-GAN weight bundle for
/// [`SbV2Model::synthetic_for_test`] — the same smooth-sinusoidal, bounded,
/// nonzero shape convention `tests/sbv2_decoder.rs`'s `jp_extra_weights`
/// helper (Task 22) uses, generalized over whatever `attrs` the caller
/// passes (that helper already loops over `attrs.n_upsample_stages()` /
/// `attrs.n_mrf_branches()` rather than hard-coding JP-Extra's `[8, 8, 2,
/// 2]` ladder, so this is the same algorithm, not a fork of it).
fn synthetic_hifigan_weights(attrs: &HifiGanAttrs) -> HifiGanWeights {
    let conv_pre_kernel = 7;
    let conv_post_kernel = 7;
    let mut w = HifiGanWeights {
        conv_pre_weight: Vec::new(),
        conv_pre_bias: Vec::new(),
        conv_pre_kernel,
        upsample_weights: Vec::new(),
        mrf_stage_weights: Vec::new(),
        conv_post_weight: Vec::new(),
        conv_post_bias: Vec::new(),
        conv_post_kernel,
        // Synthetic tests use the unconditioned generator path
        // (piper-plus / VITS-JA parity). `SbV2Model::synthetic_for_test`
        // and `_e2e` construct these weights and pass a `None` `g` at
        // decode time (see the `SbV2Decoder::generate` doc for the
        // pre-HGAN-05 default).
        cond: None,
    };
    for oc in 0..attrs.initial_channel {
        for ic in 0..attrs.n_mels {
            for k in 0..conv_pre_kernel {
                w.conv_pre_weight
                    .push(((oc + ic + k) as f32 * 0.017).sin() * 0.05);
            }
        }
    }
    w.conv_pre_bias = (0..attrs.initial_channel)
        .map(|i| (i as f32 * 0.05).cos() * 0.01)
        .collect();

    let mut in_ch = attrs.initial_channel;
    for stage in 0..attrs.n_upsample_stages() {
        let out_ch = (in_ch / 2).max(3);
        let kernel = attrs.upsample_kernel_sizes[stage];
        let stride = attrs.upsample_rates[stage];
        let mut weight = Vec::new();
        for ic in 0..in_ch {
            for oc in 0..out_ch {
                for k in 0..kernel {
                    weight.push(((ic + oc + k + stage) as f32 * 0.023).sin() * 0.05);
                }
            }
        }
        let bias: Vec<f32> = (0..out_ch)
            .map(|i| ((i + stage) as f32 * 0.07).cos() * 0.01)
            .collect();
        w.upsample_weights.push(UpsampleStageWeights {
            weight,
            bias,
            in_ch,
            out_ch,
            kernel,
            stride,
        });

        let mut branches = Vec::new();
        for b in 0..attrs.n_mrf_branches() {
            let layers = attrs.resblock_dilation_sizes[b]
                .iter()
                .map(|dilation| {
                    let kernel = attrs.resblock_kernel_sizes[b];
                    let mut weight = Vec::new();
                    for oc in 0..out_ch {
                        for ic in 0..out_ch {
                            for k in 0..kernel {
                                weight.push(((oc + ic + k + dilation) as f32 * 0.031).sin() * 0.05);
                            }
                        }
                    }
                    let bias: Vec<f32> = (0..out_ch)
                        .map(|i| ((i + *dilation + b) as f32 * 0.11).cos() * 0.01)
                        .collect();
                    ResBlockLayer {
                        weight,
                        bias,
                        // Synthetic V2 fixture: no convs2 chain — the
                        // paired synthetic HifiGanAttrs sets
                        // `res_block_type = V2`, and mrf_branch_forward's
                        // FR-EX-08 gate rejects any layer that mixes
                        // V2 topology with populated c2 weights.
                        weight_c2: None,
                        bias_c2: None,
                        dilation: *dilation,
                        kernel,
                        channels: out_ch,
                    }
                })
                .collect();
            branches.push(MrfBranchWeights { layers });
        }
        w.mrf_stage_weights.push(branches);
        in_ch = out_ch;
    }
    for ic in 0..in_ch {
        for k in 0..conv_post_kernel {
            w.conv_post_weight
                .push(((ic + k) as f32 * 0.019).sin() * 0.05);
        }
    }
    w.conv_post_bias = vec![0.0];
    w
}

/// Deterministic, small HiFi-GAN weight bundle for
/// [`SbV2Model::synthetic_for_test_e2e`] (Task 42) — the same
/// smooth-sinusoidal, bounded, nonzero shape convention as
/// [`synthetic_hifigan_weights`] above, but with every magnitude
/// coefficient bumped `0.05 -> 0.5` (weight) / `0.01 -> 0.1` (bias) —
/// cheap insurance against the accumulated forward pass underflowing
/// toward silence through the JP-Extra ladder's 4 sequential upsample/MRF
/// stages (`synthetic_for_test_e2e`'s doc, magnitude-bump paragraph). Kept
/// as its own function rather than parameterizing
/// [`synthetic_hifigan_weights`] with a magnitude scale, so
/// `synthetic_for_test`'s existing PCM values stay byte-for-byte
/// unchanged — this crate's own precedent for two near-identical weight
/// builders kept deliberately separate (`tests/sbv2_decoder.rs`'s
/// `jp_extra_weights` doc explains the same choice for the same reason:
/// no shared `HiFiGanGenerator` type to hang a parameterized helper off
/// of).
fn synthetic_hifigan_weights_e2e(attrs: &HifiGanAttrs) -> HifiGanWeights {
    let conv_pre_kernel = 7;
    let conv_post_kernel = 7;
    let mut w = HifiGanWeights {
        conv_pre_weight: Vec::new(),
        conv_pre_bias: Vec::new(),
        conv_pre_kernel,
        upsample_weights: Vec::new(),
        mrf_stage_weights: Vec::new(),
        conv_post_weight: Vec::new(),
        conv_post_bias: Vec::new(),
        conv_post_kernel,
        // Synthetic e2e path is unconditioned — see
        // `synthetic_hifigan_weights`'s comment on the same field.
        cond: None,
    };
    for oc in 0..attrs.initial_channel {
        for ic in 0..attrs.n_mels {
            for k in 0..conv_pre_kernel {
                w.conv_pre_weight
                    .push(((oc + ic + k) as f32 * 0.017).sin() * 0.5);
            }
        }
    }
    w.conv_pre_bias = (0..attrs.initial_channel)
        .map(|i| (i as f32 * 0.05).cos() * 0.1)
        .collect();

    let mut in_ch = attrs.initial_channel;
    for stage in 0..attrs.n_upsample_stages() {
        let out_ch = (in_ch / 2).max(3);
        let kernel = attrs.upsample_kernel_sizes[stage];
        let stride = attrs.upsample_rates[stage];
        let mut weight = Vec::new();
        for ic in 0..in_ch {
            for oc in 0..out_ch {
                for k in 0..kernel {
                    weight.push(((ic + oc + k + stage) as f32 * 0.023).sin() * 0.5);
                }
            }
        }
        let bias: Vec<f32> = (0..out_ch)
            .map(|i| ((i + stage) as f32 * 0.07).cos() * 0.1)
            .collect();
        w.upsample_weights.push(UpsampleStageWeights {
            weight,
            bias,
            in_ch,
            out_ch,
            kernel,
            stride,
        });

        let mut branches = Vec::new();
        for b in 0..attrs.n_mrf_branches() {
            let layers = attrs.resblock_dilation_sizes[b]
                .iter()
                .map(|dilation| {
                    let kernel = attrs.resblock_kernel_sizes[b];
                    let mut weight = Vec::new();
                    for oc in 0..out_ch {
                        for ic in 0..out_ch {
                            for k in 0..kernel {
                                weight.push(((oc + ic + k + dilation) as f32 * 0.031).sin() * 0.5);
                            }
                        }
                    }
                    let bias: Vec<f32> = (0..out_ch)
                        .map(|i| ((i + *dilation + b) as f32 * 0.11).cos() * 0.1)
                        .collect();
                    ResBlockLayer {
                        weight,
                        bias,
                        // Synthetic V2 e2e fixture — no convs2 chain.
                        // See the sibling synthetic_hifigan_weights
                        // builder for the same rationale.
                        weight_c2: None,
                        bias_c2: None,
                        dilation: *dilation,
                        kernel,
                        channels: out_ch,
                    }
                })
                .collect();
            branches.push(MrfBranchWeights { layers });
        }
        w.mrf_stage_weights.push(branches);
        in_ch = out_ch;
    }
    for ic in 0..in_ch {
        for k in 0..conv_post_kernel {
            w.conv_post_weight
                .push(((ic + k) as f32 * 0.019).sin() * 0.5);
        }
    }
    w.conv_post_bias = vec![0.1];
    w
}

/// Transposes a `[rows, cols]` row-major buffer into a `[cols, rows]`
/// row-major buffer — bridges [`SbV2Flow::inverse`]'s time-major
/// `[mel_seq_len, d_z]` output into [`SbV2Decoder::generate`]'s
/// channel-major `[n_mels, mel_seq_len]` input convention (`decoder.rs`'s
/// module doc: "bridging one layout to the other is Task 23's integration
/// concern, not this thin wrapper's"). `cols` is derived from `buf.len() /
/// rows` rather than taken as a parameter, since every caller already knows
/// `rows` (`mel_seq_len`) and the buffer's own length pins `cols`.
///
/// # Panics
///
/// Panics (via `debug_assert!`, so only in debug builds — matches every
/// other `sbv2` module's shape-check convention) if `rows == 0` or
/// `buf.len()` is not a multiple of `rows`.
/// FLOW-NOISE-SCALE helper (2026-08-09): draws `n` standard-normal
/// `f32` deviates from either the torch-parity `TorchRandnStream` or
/// the legacy `GaussianSplitMix64` per `rng_mode`, seeded with `seed`.
///
/// Callers scale + add elementwise to `mel_hidden` before
/// `SbV2Flow::inverse` so `req.noise_scale = 0.667` (docs/superpowers/
/// specs/2026-07-26-sbv2-v2-design.md §7 default) reproduces VITS's
/// standard prior reparameterization `z_p = mel_hidden + torch.randn
/// * noise_scale`. Delegates to `NormalSource::fill` so both streams
/// use the per-stream contract (`TorchRandnStream::fill` overrides for
/// the `>= 16` normal_fill dispatch that byte-matches `torch.randn`).
fn draw_flow_prior_noise(
    seed: u64,
    rng_mode: RngMode,
    mel_seq_len: usize,
    d_model: usize,
) -> Vec<f32> {
    use vokra_core::rng::NormalSource;
    let n = mel_seq_len * d_model;
    match rng_mode {
        RngMode::PhiloxRngEnginePyTorchParity => {
            // RNG PARITY (2026-08-09, PR27-parity-bisect): per-sample streaming
            // (`next_normal()` in a loop) — NOT the batch `fill()` fast path.
            //
            // The Python reference dumper's `run_flow` draws the flow prior
            // noise via `torch.empty(B, D, T+1)[..., :T].normal_(0, 1)` on a
            // NON-CONTIGUOUS strided view. Two consequences:
            //
            // 1. `.normal_()` on a non-contig view forces torch's
            //    `normal_kernel` off the `normal_fill` fast path onto the
            //    scalar streaming `at::normal_distribution<double>` path
            //    (with pair caching). `TorchRandnStream::next_normal` is a
            //    bit-exact port of exactly that streaming path (verified by
            //    `rng_torch_randn_cpu_parity.rs` k=4 anchor).
            //
            // 2. torch's tensor iterator visits the visible elements in
            //    memory-linear order over the underlying `[B, D, T+1]`
            //    contiguous storage. For a `[..., :T]` view that skips
            //    index T in the last dim, the visit order is:
            //    `(b=0, c=0, t=0..T-1), (b=0, c=1, t=0..T-1), …` — i.e.
            //    channel-major (channel varies slowest, time varies
            //    fastest). This CHANNEL-MAJOR ordering is critical:
            //    Rust adds `flow_noise[i]` to `mel_hidden[i]` where
            //    `mel_hidden` is `[T, D]` **position-major** row-major
            //    (see `length_regulate`). To honor Python's channel-major
            //    fill while landing samples on the correct positions in
            //    `mel_hidden`, we fill a `[D, T]` scratch and transpose
            //    into `[T, D]` on the way out.
            //
            // Batch `fill()` was also considered — it takes Rust down
            // `normal_fill_16_scalar`, which byte-matches torch's SCALAR
            // fast path (M1, non-AVX2 x86_64) but NOT torch's streaming
            // path: the two consume RNG in fundamentally different orders
            // (see `TorchRandnStream::fill`'s doc). On AVX2 CI hosts torch
            // takes `normal_fill_AVX2` for contig-fast-path tensors, which
            // is a THIRD ordering (vectorized `avx_mathfun`). Per-sample
            // streaming is the only ordering that Python can honestly
            // reproduce on every CPU host (via the non-contig trick above),
            // so that is the ordering Rust commits to here.
            let mut rng = TorchRandnStream::new(seed);
            let mut ct = vec![0.0_f32; n]; // channel-major scratch [D, T]
            for v in &mut ct {
                *v = rng.next_normal();
            }
            // Transpose `[D, T]` channel-major → `[T, D]` position-major.
            let mut buf = vec![0.0_f32; n];
            for c in 0..d_model {
                for t in 0..mel_seq_len {
                    buf[t * d_model + c] = ct[c * mel_seq_len + t];
                }
            }
            buf
        }
        RngMode::GaussianSplitMix64Legacy => {
            // Legacy synthetic path — no cross-parity claim, no transpose.
            let mut buf = vec![0.0_f32; n];
            let mut rng = GaussianSplitMix64::new(seed);
            rng.fill(&mut buf);
            buf
        }
    }
}

fn transpose_time_major_to_channel_major(buf: &[f32], rows: usize) -> Vec<f32> {
    debug_assert!(
        rows > 0,
        "transpose_time_major_to_channel_major: rows must be positive"
    );
    debug_assert_eq!(
        buf.len() % rows,
        0,
        "transpose_time_major_to_channel_major: buf.len() must be a multiple of rows"
    );
    let cols = buf.len() / rows;
    let mut out = vec![0.0_f32; buf.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = buf[r * cols + c];
        }
    }
    out
}

/// Derive the per-head channel width for the SBV2 flow encoder stack from
/// the model-wide `d_model` and `n_heads` metadata (COSMETIC-BUNDLE audit,
/// 2026-08-09).
///
/// The upstream `n_heads == 0` guard inside
/// [`SbV2Model::from_gguf`]'s early metadata validation loop (`if n_heads
/// == 0 { return Err(...) }`) ensures this division is safe on every
/// reachable call site. This helper's `.expect()` converts a hypothetical
/// future guard regression from **silent-wrong** — the pre-fix
/// `d_model.checked_div(n_heads).unwrap_or(0)` returned `0` on
/// `n_heads == 0`, letting a wrong-shape [`text_encoder::RelPositionMHA`]
/// downstream mis-parse tensors in release builds — into a **loud panic**
/// naming the exact site (FR-EX-08 defence in depth).
///
/// The sibling text-encoder d_head derivation at
/// [`SbV2Model::from_gguf`]'s line ~1934 uses direct `d_model / n_heads`
/// (idiomatic Rust panic on divide-by-zero); this helper's `.expect()`
/// carries a richer message for the flow chain's specific failure mode.
fn derive_flow_head_dim(d_model: usize, n_heads: usize) -> usize {
    d_model.checked_div(n_heads).expect(
        "derive_flow_head_dim: n_heads must be positive (enforced by \
         SbV2Model::from_gguf's `vokra.sbv2.n_heads must be positive` guard)",
    )
}

#[cfg(test)]
mod tests {
    use super::{SBV2_HOT_OPS, derive_flow_head_dim};
    use crate::compute::HotOp;

    /// COSMETIC-BUNDLE (d_head divide-by-zero, 2026-08-09): the pre-fix
    /// `.unwrap_or(0)` at the `d_head_flow` site silently returned `0` on
    /// `n_heads == 0`, letting a wrong-shape downstream
    /// `RelPositionMHA::new` fire only under `debug_assert!` (i.e. release
    /// builds silently produced garbage). This helper's `.expect()` is the
    /// loud-fail replacement — this test pins the panic behaviour so a
    /// future refactor that reintroduces a silent fallback trips
    /// `#[should_panic]`.
    #[test]
    #[should_panic(expected = "n_heads must be positive")]
    fn derive_flow_head_dim_panics_on_zero_n_heads() {
        let _ = derive_flow_head_dim(192, 0);
    }

    /// Positive-path sanity: with the SBV2 v2 base's typical
    /// (`d_model = 192`, `n_heads = 2`) the helper returns `96`
    /// exactly — matches the sibling text-encoder d_head derivation
    /// (`d_head = d_model / n_heads` at line ~1934), preventing a
    /// future refactor from accidentally shifting the flow chain's
    /// head-width off the text-encoder's.
    #[test]
    fn derive_flow_head_dim_matches_direct_division() {
        assert_eq!(derive_flow_head_dim(192, 2), 96);
        assert_eq!(derive_flow_head_dim(768, 12), 64);
        assert_eq!(derive_flow_head_dim(4, 1), 4);
    }

    #[test]
    fn metal_preflight_registry_covers_all_learned_sbv2_seams() {
        // This is the complete set consumed by text attention/FFN, BERT
        // bridges, speaker/style projections, SDP/flow convolutions, and the
        // conditioned HiFi-GAN adapter. A missing entry would let a newly
        // selected backend reach a learned operation without its capability
        // check and is therefore a fail-closed regression.
        for required in [
            HotOp::Gemm,
            HotOp::Softmax,
            HotOp::LayerNorm,
            HotOp::Gelu,
            HotOp::Conv1d,
            HotOp::GroupedConv1d,
            HotOp::Gemv,
        ] {
            assert!(
                SBV2_HOT_OPS.contains(&required),
                "SBV2 backend preflight omitted {required:?}"
            );
        }
    }
}
