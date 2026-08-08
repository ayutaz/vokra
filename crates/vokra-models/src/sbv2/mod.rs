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

pub mod decoder;
pub mod duration;
pub mod flow;
pub mod g2p;
pub mod parity;
pub mod rng_mode;
pub mod speaker;
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
pub use g2p::{Language, PhonemizeFixture, PhonemizeResult, SbV2Phonemizer};
pub use parity::{ATOL_DEFAULT, PER_TENSOR_ATOL, tolerance_for};
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

use vokra_bert::deberta_v2::DebertaV2Encoder;
use vokra_bert::deberta_v3::DebertaV3Encoder;
use vokra_bert::tokenizer::SbertTokenizer;
use vokra_core::gguf::GgufFile;
use vokra_core::rng::{GaussianSplitMix64, TorchRandnStream};
use vokra_core::{Result, SynthesisRequest, SynthesizedAudio, TtsEngine, VokraError};
use vokra_ops::attrs::HifiGanAttrs;
use vokra_ops::hifigan::{
    HifiGanConfig, HifiGanWeights, MrfBranchWeights, ResBlockLayer, UpsampleStageWeights,
};

/// Both BERT encoders (+ their tokenizers) [`SbV2Model`] needs, loaded
/// together so one loaded model instance can serve either language without
/// a reload: JA text routes through [`DebertaV2Encoder`], EN through
/// [`DebertaV3Encoder`] (`docs/superpowers/specs/2026-07-26-sbv2-v2-design.md`
/// §7's "BERT Router").
pub struct SbV2BertContainer {
    /// SentencePiece tokenizer feeding [`ja`](Self::ja)'s input ids.
    pub ja_tokenizer: SbertTokenizer,
    /// SentencePiece tokenizer feeding [`en`](Self::en)'s input ids.
    pub en_tokenizer: SbertTokenizer,
    /// JA BERT encoder (DeBERTa v2).
    pub ja: DebertaV2Encoder,
    /// EN BERT encoder (DeBERTa v3).
    pub en: DebertaV3Encoder,
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
    /// Flow-latent noise scale. **Not yet consumed** by
    /// [`SbV2Model::synthesize`]'s scaffold pipeline — a full VITS-family
    /// inverse reparameterizes the flow's Gaussian prior with this scale
    /// (`z ~ N(mel_hidden_mean, exp(mel_hidden_logstd) * noise_scale)`);
    /// this scaffold instead feeds `mel_hidden` straight into
    /// [`SbV2Flow::inverse`] with no added noise, since the prior-head
    /// weights that would produce a mean/logstd split do not exist until
    /// Task 24-27 loads a real checkpoint. Kept on the request now so this
    /// public shape does not need to grow later.
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
        if req.speed <= 0.0 {
            return Err(VokraError::InvalidArgument(format!(
                "SbV2Model::synthesize: req.speed must be positive, got {}",
                req.speed
            )));
        }

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
        let text_hidden =
            self.text_encoder
                .forward(&phon.phoneme_ids, &phon.tones, req.language.language_id());

        // 3. BERT (per-language). ZH has no in-crate BERT encoder yet
        // (BERT bridge is JA/EN only in the M6 land — see the ZH scope
        // note on `Language`); reaching ZH here would be a bug (the
        // phonemizer's `phonemize_zh` already returned NotImplemented and
        // step 1 propagated it), but if a caller has bypassed the
        // phonemizer via `PhonemizeFixture` for ZH the tokenizer step
        // becomes reachable and must fail loudly rather than silently
        // routing to JA/EN — FR-EX-08.
        let bert_ids = match req.language {
            Language::JA => self.bert.ja_tokenizer.encode(&phon.bert_input_text),
            Language::EN => self.bert.en_tokenizer.encode(&phon.bert_input_text),
            Language::ZH => {
                return Err(VokraError::NotImplemented(
                    "SbV2Model::synthesize: language ZH has no BERT tokenizer wired in this \
                     crate (SbV2BertContainer holds only ja/en). The text encoder's \
                     language_embed row 2 is reachable, but the BERT bridge path is not — \
                     Vokra ZH BERT + G2P are out of scope for the M6 SBV2 v2 land (FR-EX-08).",
                ));
            }
        };
        if bert_ids.is_empty() {
            return Err(VokraError::InvalidArgument(
                "SbV2Model::synthesize: BERT tokenizer produced no tokens for the input text"
                    .to_string(),
            ));
        }
        let bert_hidden = match req.language {
            Language::JA => self.bert.ja.forward(&bert_ids),
            Language::EN => self.bert.en.forward(&bert_ids),
            // Unreachable: the tokenizer arm above already returned for ZH.
            // Kept as a loud panic (not silent fall-through) so a future
            // ZH BERT wiring can't accidentally leave this arm speaking
            // for JA — FR-EX-08.
            Language::ZH => unreachable!(
                "SbV2Model::synthesize: ZH tokenizer arm above must have returned \
                 NotImplemented before reaching here"
            ),
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
        let bridged =
            self.bert_bridge
                .forward(&bert_hidden, phon.phoneme_ids.len(), bert_ids.len());
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
        // | request.speaker_embedding | model.speaker_projection | path                                                            |
        // |---------------------------|--------------------------|-----------------------------------------------------------------|
        // | `Some(vec)`               | `Some(proj)`             | validate via `proj.forward(vec)`; pass `vec` to SDP.g and flow |
        // | `None`                    | `Some(proj)`             | validate via `proj.forward(zeros)`; pass zeros to SDP.g and flow |
        // | `Some(_)`                 | `None`                   | loud `VokraError::InvalidArgument` (FR-EX-08)                  |
        // | `None`                    | `None`                   | legacy: `speaker_embed.lookup(speaker_id)`, pass to SDP.g and flow |
        let d_model = self.text_encoder.d_model();
        let phoneme_count = phon.phoneme_ids.len();
        let speaker_e_flow: Vec<f32> = match (
            req.speaker_embedding.as_deref(),
            self.speaker_projection.as_ref(),
        ) {
            (Some(ext), Some(proj)) => {
                // Real-ckpt path: caller-supplied external embedding.
                // `proj.forward` still loudly rejects wrong-length input
                // (FR-EX-08). The projected `[d_model]` result is
                // discarded — see step 5's Bug 4 fix comment.
                let projected = proj.forward(ext)?;
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
                let projected = proj.forward(&zeros)?;
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
        // Style injector: Python reference explicitly does NOT mix style
        // into text_hidden ("style is dumped for the manifest slot but
        // not otherwise mixed here" — sbv2_dump_reference.py step 9). We
        // still call `inject` on `hidden_for_flow` because on synthetic
        // paths `inject` is a no-op (all-zero weights) and on real-ckpt
        // paths (once wired) the style might legitimately mix into the
        // flow input via `emb_g_style`. Base ckpt has no style tensors
        // so `inject` is bias-only there. This matches the pre-fix
        // behavior for the flow-input path; the SDP-input path was the
        // one that had to change. TODO(sbv2-follow-up): validate this
        // matches upstream once fine-tune ckpts with real
        // `emb_g_style.*` tensors are available.
        self.style_injector
            .inject(&mut hidden_for_flow, phoneme_count, &req.style_vec);

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
                self.sdp.sample(
                    &text_hidden,
                    phon.phoneme_ids.len(),
                    sdp_g,
                    &mut rng,
                    req.noise_scale_w,
                )
            }
            RngMode::GaussianSplitMix64Legacy => {
                let mut rng = GaussianSplitMix64::new(req.seed);
                self.sdp.sample(
                    &text_hidden,
                    phon.phoneme_ids.len(),
                    sdp_g,
                    &mut rng,
                    req.noise_scale_w,
                )
            }
        };
        for d in &mut durations {
            *d = ((*d as f32) / req.speed).max(1.0) as i32;
        }

        // OOM stopgap (2026-08-08, follow-up to run 31197123061 46.6 GiB
        // alloc = flow attention [mel_seq_len × mel_seq_len × f32]):
        // clamp each phoneme's duration to a per-phoneme sanity ceiling
        // so an upstream-scale-inflated text_hidden cannot blow the
        // runtime up.
        //
        // # True root cause (audit 2026-08-08, Bug 4 in SBV2-BUG4 spec)
        //
        // The SDP's flow-inverse math is correct — `SbV2SDP` was rewritten
        // post-Blocker-2c to a real DDS + ConvFlow StochasticDurationPredictor
        // (the earlier "scalar-affine simplification" is gone; the module
        // doc still tracks the pre-Blocker-2c history but that comment does
        // NOT describe the current implementation). What CI observed with
        // max=26539 durations was the SDP being fed a text_hidden ~35× too
        // large in magnitude by the text_encoder (Wave-2 SBV2-BUG4 gap):
        // the SDP correctly amplifies its input through `exp().ceil()` and
        // a 35× input becomes an exponentially large duration.
        //
        // The `VOKRA_SBV2_SDP_HIDDEN_OVERRIDE` experiment (see
        // `docs/handoff/sbv2-sdp-debug-2026-08-08.md` §Bug 4) proved that
        // feeding the SDP the Python reference `text_hidden.bin` bytes
        // produces sum=28 vs the Python reference sum=26 — the SDP itself
        // is not the bug. The 3 candidate root causes tracked upstream are
        // (a) missing `x*x_mask` scaling in PositionWiseFFN, (b) missing
        // `enc_p.encoder.spk_emb_linear` per-block gating, (c) wrong
        // Conv1d weight layout in `conv1d_same_padded`.
        //
        // # Why the cap stays until Wave 2 lands SBV2-BUG4
        //
        // Cap here is only a safety fuse; a parity assertion still fires
        // on the numeric delta downstream — this stopgap turns an
        // unbounded panic into a bounded parity red so CI can actually
        // report the text_encoder scale gap instead of OOM-ing before the
        // assertion runs. Once SBV2-BUG4 lands, both the cap and the
        // warning below become dead code and should be deleted (Phase 2
        // of the OOM-STOPGAP-CLEANUP audit gap).
        //
        // Ceiling 500 = ~5.8s at 86 Hz frame rate per phoneme (way above
        // real speech's ~10-30-frame span), so it never truncates a real
        // duration a working forward pass would produce — only the
        // runaway values from a scale-inflated text_hidden.
        const PER_PHONEME_DURATION_CEILING: i32 = 500;
        let capped_any = durations.iter().any(|&d| d > PER_PHONEME_DURATION_CEILING);
        if capped_any {
            let original_sum: i32 = durations.iter().copied().sum();
            let original_max: i32 = durations.iter().copied().max().unwrap_or(0);
            for d in &mut durations {
                *d = (*d).min(PER_PHONEME_DURATION_CEILING);
            }
            eprintln!(
                "[sbv2-synth-warn] SbV2SDP produced runaway durations \
                 (max={original_max}, sum={original_sum}) — clamped to \
                 per-phoneme ceiling {PER_PHONEME_DURATION_CEILING}. True cause is \
                 upstream text_encoder emitting hidden values ~35× too large \
                 (SBV2-BUG4 in the Wave-2 spec, docs/handoff/sbv2-sdp-debug-2026-08-08.md); \
                 the SDP forward itself is correct. Downstream parity WILL fail — \
                 this cap only prevents OOM."
            );
        }

        // 7. Length regulate — uses `hidden_for_flow` (= text_hidden +
        // bridge, matching Python `bert_bridge_out`). Bug 4 fix
        // (2026-08-08): pre-fix code fed the accumulated `hidden` which
        // included speaker/style broadcast-adds; Python reference does
        // not add speaker/style here (they enter via flow's per-block
        // spk_emb_linear and decoder's `dec.cond` respectively).
        let mel_hidden = length_regulate(&hidden_for_flow, &durations, d_model);
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
        let z = self.flow.inverse(&mel_hidden, mel_seq_len, flow_g);

        // 9. HiFi-GAN decoder — transpose SbV2Flow::inverse's time-major
        // [mel_seq_len, d_z] into SbV2Decoder::generate's channel-major
        // [n_mels, mel_seq_len] (decoder.rs's module doc: "bridging one
        // layout to the other is Task 23's integration concern, not this
        // thin wrapper's"). d_z must equal the decoder's n_mels (a Task 23
        // construction-time contract, held by `synthetic_for_test` above);
        // a mismatch surfaces as `SbV2Decoder::generate`'s own
        // `debug_assert!`.
        let z_channel_major = transpose_time_major_to_channel_major(&z, mel_seq_len);
        let pcm = self.decoder.generate(&z_channel_major, mel_seq_len);

        Ok(SynthesizedAudio::new(pcm, self.decoder.sample_rate()))
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
    /// `"en"` selects [`Language::EN`]; anything else — including `None` —
    /// selects [`Language::JA`] (SBV2's base config is Japanese-first, per
    /// its JP-Extra heritage — see `decoder.rs`'s module doc).
    /// `request.deterministic` zeroes both `noise_scale` and
    /// `noise_scale_w` (mirrors the piper-plus adapter's identical
    /// convention); otherwise this adapter applies
    /// `docs/superpowers/specs/2026-07-26-sbv2-v2-design.md` §7's documented
    /// SDP defaults (`noise_scale = 0.667`, `noise_scale_w = 0.8`).
    /// `style_vec` defaults to the identity all-zero vector (sized from
    /// `self.style_injector.d_style()`) and `speaker_id` defaults to `0` —
    /// [`SynthesisRequest`] carries neither a style-vector nor a
    /// discrete-speaker-id field.
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
    /// Returns [`VokraError::InvalidArgument`] if
    /// `request.prosody_features` is `Some(..)`: SBV2 derives
    /// pitch-accent tones from its own G2P, not a caller-supplied
    /// per-phoneme accent triple — honoring it would mean silently
    /// discarding caller-supplied data. Also propagates any error the
    /// inherent [`synthesize`](SbV2Model::synthesize) call returns
    /// (including Blocker 3's speaker-conditioning errors — see the
    /// section above).
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

        let sbv2_request = SbV2SynthRequest {
            text: request.text.clone(),
            language,
            speaker_id: 0,
            speaker_embedding: request.speaker_embedding.clone(),
            style_vec: vec![0.0; self.style_injector.d_style()],
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
    /// Every key below is a required `u32` unless noted otherwise — a
    /// missing or wrong-typed key is [`VokraError::ModelLoad`] naming the
    /// key, never a silent default (FR-EX-08). Rationale for why each value
    /// lives here (vs. being derivable some other way) follows each entry.
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
    /// - `decoder.leaky_relu_slope` (`f32`, **optional**, default `0.1`) —
    ///   [`HifiGanAttrs::leaky_relu_slope`]. Defaults to the universal
    ///   jik876/hifi-gan `LRELU_SLOPE` every sibling decoder in this
    ///   codebase uses (`vits_ja::VITS_JA_LEAKY_RELU_SLOPE`, piper-plus's
    ///   `LRELU_SLOPE`), so a converter need not emit it for a stock
    ///   config.
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
    /// # Errors
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
        Self::from_gguf_inner(main, bert_ja, bert_en, phonemizer)
    }

    // The full loader body — shared by [`from_gguf`] (which passes the
    // [`UnwiredPhonemizer`]-backed default phonemizer) and Task 7's
    // [`from_gguf_with_phonemizer`] (which passes the caller's own). Every
    // step besides the `phonemizer` argument is identical, so both public
    // entry points share this to guarantee the same error surface (Task 7
    // must not silently change the load path).
    fn from_gguf_inner(
        main: &GgufFile,
        bert_ja: &GgufFile,
        bert_en: &GgufFile,
        phonemizer: SbV2Phonemizer,
    ) -> Result<Self> {
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
        let meta_f32 =
            |key: &str| -> Option<f32> { main.get(key).and_then(|v| v.as_f64()).map(|f| f as f32) };
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
        let bert = SbV2BertContainer {
            ja_tokenizer,
            en_tokenizer,
            ja,
            en,
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
        let d_head_flow = d_model.checked_div(n_heads).unwrap_or(0);
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
        let leaky_relu_slope = meta_f32("vokra.sbv2.decoder.leaky_relu_slope").unwrap_or(0.1);

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
                    layers.push(ResBlockLayer {
                        weight: load_tensor_f32(&format!("{p}.weight"))?,
                        bias: load_tensor_f32(&format!("{p}.bias"))?,
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

        let weights = HifiGanWeights {
            conv_pre_weight,
            conv_pre_bias,
            conv_pre_kernel,
            upsample_weights,
            mrf_stage_weights,
            conv_post_weight,
            conv_post_bias,
            conv_post_kernel,
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
        Self::from_gguf_inner(main, bert_ja, bert_en, phonemizer)
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
