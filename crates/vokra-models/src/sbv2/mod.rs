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
pub mod speaker;
pub mod style;
pub mod text_encoder;
// Later tasks add: mod converter; mod parity;

pub use decoder::SbV2Decoder;
pub use duration::{SbV2SDP, length_regulate};
pub use flow::SbV2Flow;
pub use g2p::{Language, PhonemizeResult, SbV2Phonemizer};
pub use speaker::SpeakerEmbedding;
pub use style::StyleVectorInjector;
pub use text_encoder::{BertBridge, SbV2TextEncoder};

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
use vokra_core::rng::GaussianSplitMix64;
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
    pub speaker_id: u32,
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
    /// Seed for the duration predictor's Gaussian draws
    /// ([`GaussianSplitMix64`]). Irrelevant when `noise_scale_w == 0.0`.
    pub seed: u64,
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
            style_injector,
            sdp,
            flow,
            decoder,
        }
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
            vec![0.0; 2 * D_MODEL],
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

        let sdp = SbV2SDP::from_weights(
            Vec::new(), // empty flow stack — see this fn's doc
            vec![0.0; N_TONES * D_MODEL],
            vec![0.0; D_MODEL],
            D_MODEL,
            N_TONES,
        );

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

        // 2. Text encoder
        let text_hidden =
            self.text_encoder
                .forward(&phon.phoneme_ids, &phon.tones, &phon.word_boundaries);

        // 3. BERT (per-language)
        let bert_ids = match req.language {
            Language::JA => self.bert.ja_tokenizer.encode(&phon.bert_input_text),
            Language::EN => self.bert.en_tokenizer.encode(&phon.bert_input_text),
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
        };

        // 4. BERT bridge
        let bridged =
            self.bert_bridge
                .forward(&bert_hidden, phon.phoneme_ids.len(), bert_ids.len());
        let mut hidden = text_hidden;
        debug_assert_eq!(
            hidden.len(),
            bridged.len(),
            "SbV2Model::synthesize: text_encoder hidden width must equal bert_bridge's \
             projected width (BertBridge's d_target must equal SbV2TextEncoder's d_model — \
             see SbV2Model's struct doc)"
        );
        for (h, &b) in hidden.iter_mut().zip(bridged.iter()) {
            *h += b;
        }

        // 5. Speaker + style
        let speaker_e = self.speaker_embed.lookup(req.speaker_id)?;
        let d_model = self.text_encoder.d_model();
        debug_assert_eq!(
            speaker_e.len(),
            d_model,
            "SbV2Model::synthesize: SpeakerEmbedding's d_speaker must equal SbV2TextEncoder's \
             d_model for the broadcast add below — see SbV2Model's struct doc"
        );
        for i in 0..phon.phoneme_ids.len() {
            for (d, &s) in speaker_e.iter().enumerate() {
                hidden[i * d_model + d] += s;
            }
        }
        self.style_injector
            .inject(&mut hidden, phon.phoneme_ids.len(), &req.style_vec);

        // 6. SDP -> durations
        let mut rng = GaussianSplitMix64::new(req.seed);
        let mut durations = self.sdp.sample(
            &hidden,
            &phon.tones,
            phon.phoneme_ids.len(),
            &mut rng,
            req.noise_scale_w,
        );
        for d in &mut durations {
            *d = ((*d as f32) / req.speed).max(1.0) as i32;
        }

        // 7. Length regulate
        let mel_hidden = length_regulate(&hidden, &durations, d_model);
        let mel_seq_len = durations.iter().sum::<i32>() as usize;
        debug_assert!(
            mel_seq_len > 0,
            "SbV2Model::synthesize: mel_seq_len must be positive (every SbV2SDP::sample \
             duration is >= 1 by construction once phoneme_ids is non-empty)"
        );

        // 8. Flow inverse (scaffold: mel_hidden feeds the flow directly —
        // see SbV2SynthRequest::noise_scale's doc for the real
        // reparameterization this stands in for).
        let z = self
            .flow
            .inverse(&mel_hidden, mel_seq_len, &req.style_vec, speaker_e);

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
    /// # Errors
    ///
    /// Returns [`VokraError::InvalidArgument`] if `request.speaker_embedding`
    /// or `request.prosody_features` is `Some(..)`: SBV2 selects speakers
    /// through [`SpeakerEmbedding`]'s discrete id lookup, not a
    /// caller-supplied continuous embedding, and derives pitch-accent tones
    /// from its own G2P, not a caller-supplied per-phoneme accent triple —
    /// honoring either would mean silently discarding caller-supplied data,
    /// so this adapter errors loudly instead (this codebase's established
    /// FR-EX-08 convention for a request field a specific engine cannot
    /// honor — e.g. the Whisper `no_repeat_ngram_size` gate in
    /// `integrations/vokra-server`). Also propagates any error the inherent
    /// [`synthesize`](SbV2Model::synthesize) call returns.
    fn synthesize(&self, request: &SynthesisRequest) -> Result<SynthesizedAudio> {
        if request.speaker_embedding.is_some() {
            return Err(VokraError::InvalidArgument(
                "SbV2Model (TtsEngine): caller-supplied speaker_embedding is not supported — \
                 SBV2 selects speakers via a discrete id (SbV2SynthRequest::speaker_id); call \
                 SbV2Model::synthesize directly for speaker-id control"
                    .to_string(),
            ));
        }
        if request.prosody_features.is_some() {
            return Err(VokraError::InvalidArgument(
                "SbV2Model (TtsEngine): caller-supplied prosody_features is not supported — \
                 SBV2 derives pitch-accent tones from its own G2P; call SbV2Model::synthesize \
                 directly"
                    .to_string(),
            ));
        }

        let language = match request.language.as_deref() {
            Some(lang) if lang.to_ascii_lowercase().starts_with("en") => Language::EN,
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
            style_vec: vec![0.0; self.style_injector.d_style()],
            speed: 1.0,
            noise_scale,
            noise_scale_w,
            seed: 0,
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
    /// - `sbv2.text_encoder.phoneme_embed` / `.tone_embed` / `.wb_embed` —
    ///   the three embedding tables ([`SbV2TextEncoder::from_weights`]'s
    ///   first three parameters).
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
    /// - Per flow layer `<i>` in `0..n_flow_layers`:
    ///   `sbv2.flow.layer.<i>.scale_weight` / `.shift_weight` /
    ///   `.style_proj` / `.speaker_proj` — matches
    ///   [`SbV2AffineCouplingLayer`](flow::SbV2AffineCouplingLayer)'s four
    ///   fields 1:1.
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
        let n_flow_layers = require_u32("vokra.sbv2.n_flow_layers")? as usize;
        let n_sdp_layers = require_u32("vokra.sbv2.n_sdp_layers")? as usize;
        let sample_rate = require_u32("vokra.sbv2.sample_rate")?;

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
        let wb_embed = load_tensor_f32("sbv2.text_encoder.wb_embed")?;
        let mut transformer_layers = Vec::with_capacity(n_text_layers);
        for i in 0..n_text_layers {
            let p = format!("sbv2.text_encoder.layer.{i}");
            transformer_layers.push(text_encoder::SbV2TransformerBlock::new(
                load_tensor_f32(&format!("{p}.attn.q.weight"))?,
                load_tensor_f32(&format!("{p}.attn.k.weight"))?,
                load_tensor_f32(&format!("{p}.attn.v.weight"))?,
                load_tensor_f32(&format!("{p}.attn.o.weight"))?,
                load_tensor_f32(&format!("{p}.ln1.gamma"))?,
                load_tensor_f32(&format!("{p}.ln1.beta"))?,
                load_tensor_f32(&format!("{p}.ffn.w1.weight"))?,
                load_tensor_f32(&format!("{p}.ffn.w1.bias"))?,
                load_tensor_f32(&format!("{p}.ffn.w2.weight"))?,
                load_tensor_f32(&format!("{p}.ffn.w2.bias"))?,
                load_tensor_f32(&format!("{p}.ln2.gamma"))?,
                load_tensor_f32(&format!("{p}.ln2.beta"))?,
                d_model,
                d_ff,
            ));
        }
        let text_encoder = SbV2TextEncoder::from_weights(
            phoneme_embed,
            tone_embed,
            wb_embed,
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
        let speaker_embed = SpeakerEmbedding::from_table(
            load_tensor_f32("sbv2.speaker.table")?,
            n_speakers,
            d_speaker,
        );

        // ---- style ----
        let style_injector = StyleVectorInjector::from_projections(
            load_tensor_f32("sbv2.style_injector.proj_scale")?,
            load_tensor_f32("sbv2.style_injector.proj_bias")?,
            d_style,
            d_model,
        );

        // ---- stochastic duration predictor ----
        let mut sdp_layers = Vec::with_capacity(n_sdp_layers);
        for i in 0..n_sdp_layers {
            let p = format!("sbv2.sdp.flow_layer.{i}");
            sdp_layers.push(duration::SbV2CouplingLayer::new(
                load_tensor_f32(&format!("{p}.proj_weight"))?,
                load_tensor_f32(&format!("{p}.proj_bias"))?,
                d_model,
            ));
        }
        let sdp = SbV2SDP::from_weights(
            sdp_layers,
            load_tensor_f32("sbv2.sdp.tone_embed")?,
            load_tensor_f32("sbv2.sdp.tone_bias")?,
            d_model,
            n_tones,
        );

        // ---- normalizing flow ----
        let mut flow_layers = Vec::with_capacity(n_flow_layers);
        for i in 0..n_flow_layers {
            let p = format!("sbv2.flow.layer.{i}");
            flow_layers.push(flow::SbV2AffineCouplingLayer::new(
                load_tensor_f32(&format!("{p}.scale_weight"))?,
                load_tensor_f32(&format!("{p}.shift_weight"))?,
                load_tensor_f32(&format!("{p}.style_proj"))?,
                load_tensor_f32(&format!("{p}.speaker_proj"))?,
                half_d_z,
                d_style,
                d_speaker,
            ));
        }
        let flow = SbV2Flow::from_layers(flow_layers, d_z);

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
        let conv_post_bias = load_tensor_f32("sbv2.decoder.conv_post.bias")?;

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

        // ---- phonemizer — deliberately NOT loaded here, see "G2P is not
        // loaded here" above ----
        let phonemizer = SbV2Phonemizer::from_piper_g2p(
            Box::new(UnwiredPhonemizer),
            Box::new(UnwiredPhonemizer),
            std::collections::HashMap::new(),
            std::collections::HashMap::new(),
        );

        Ok(Self::new(
            phonemizer,
            text_encoder,
            bert,
            bert_bridge,
            speaker_embed,
            style_injector,
            sdp,
            flow,
            decoder,
        ))
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
