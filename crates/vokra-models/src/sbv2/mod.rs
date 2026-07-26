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
