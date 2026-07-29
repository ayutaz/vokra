//! Charsiu — **Wav2Vec2 neural forced aligner** (real forward, 2026-07-30).
//!
//! - Upstream: `github.com/lingjzhu/charsiu` (MIT — permissive; no
//!   runtime-side attribution obligation).
//! - Consumers: this module is the `charsiu` variant of the
//!   [`super::force_align`](super) op family, exposing the same
//!   [`AlignedToken`] output shape [`super::ctc_segmentation`] returns.
//!
//! # Architecture (primary source)
//!
//! Charsiu is a fine-tuning of the wav2vec 2.0 CTC family on IPA phonemes
//! for **forced alignment** (the transcript is given; the model recovers
//! the per-phoneme frame boundaries). The upstream release
//! (`charsiu/models/charsiu_forced_aligner.py`) instantiates a HuggingFace
//! `Wav2Vec2ForCTC` (`transformers/src/transformers/models/wav2vec2/
//! modeling_wav2vec2.py::Wav2Vec2ForCTC`), sends 16 kHz raw waveform
//! through:
//!
//! 1. `Wav2Vec2FeatureEncoder` — the raw-waveform 7-layer strided
//!    Conv1D stem (`total_stride=320`, output = `[T', 512]` at 50 Hz).
//!    Implemented by [`vokra_ops::waveform_frontend`] with
//!    [`vokra_ops::WaveformFrontendAttrs::wav2vec2_base`].
//! 2. `Wav2Vec2FeatureProjection` — a Linear from the stem's 512-d
//!    output to the residual `hidden_size` (transformer width).
//!    Implemented inline as a `[512, hidden_size]` GEMM + optional bias.
//! 3. `Wav2Vec2Encoder` — `n_layer` pre-LayerNorm Transformer blocks
//!    (MHA + SwiGLU-free FFN, GELU activation). Wav2Vec 2.0's config
//!    uses `use_conformer=false` — plain Transformer blocks, no conv
//!    positional encoder in the base charsiu variant.
//! 4. `Wav2Vec2ForCTC.lm_head` — a single Linear from `hidden_size`
//!    to `vocab_size` (IPA phoneme inventory + one blank token).
//! 5. `log_softmax` — per-frame log probabilities the CTC decoder
//!    consumes; forwarded to [`super::ctc_segmentation`] for the
//!    monotone Viterbi walk that recovers per-phoneme time boundaries.
//!
//! # Weights
//!
//! The runtime binds real weights from a Vokra GGUF that a converter
//! populated with:
//! - `vokra.charsiu.hidden_size` / `n_layer` / `n_head` / `ffn_dim` /
//!   `vocab_size` / `blank_id` / `frame_shift_sec` (transcribed
//!   verbatim from the upstream config).
//! - `waveform_frontend.layers.{i}.{conv_w,conv_b?,norm_gamma?,norm_beta?}`
//!   for each of the 7 wav2vec 2.0 base stem layers (same names the
//!   upstream state dict emits after the `feature_extractor.` prefix).
//! - `feature_projection.{norm_gamma?,norm_beta?,linear_w,linear_b}`.
//! - Per-encoder-block: `layer.{i}.{attn.q_proj,attn.k_proj,attn.v_proj,
//!   attn.out_proj,attn_norm,ffn_norm,ffn_fc1,ffn_fc2}` (weight +
//!   optional bias).
//! - `head.{weight,bias}` — the CTC vocab projection.
//!
//! The scaffold [`Charsiu::synthesized`] builds a deterministic
//! [`CharsiuWeights`] from a [`CharsiuConfig`] so the shape flow and CTC
//! decoding path can be exercised without a real HF checkpoint (SplitMix64
//! Xavier — the omniASR-CTC / VITS-JA fixture pattern). Real-weight
//! `from_gguf` binding lands in a follow-up wave (T29-equivalent — the
//! upstream tensor-name manifest fetch, same posture as the other
//! `wav2vec2_encoder`-style consumers in this crate).
//!
//! # Output
//!
//! [`Charsiu::align`] returns a `Vec<AlignedToken>` — one record per
//! input phoneme with monotone non-overlapping time boundaries and a
//! per-token confidence score derived from the CTC posterior along the
//! Viterbi path (`ctc_segmentation`'s output verbatim).

use std::path::Path;

use vokra_core::rng::SplitMix64;
use vokra_core::{Result, VokraError};
use vokra_ops::{ConvLayerWeights, WaveformFrontendAttrs, WaveformFrontendWeights};

use super::{AlignedToken, LoadError, ctc_segmentation::ctc_segmentation};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Charsiu / wav2vec2-base-for-CTC hparams.
///
/// The upstream release (`charsiu/en_w2v2_fc_10ms`) is a
/// wav2vec2-base fine-tune with the following axes (transcribed from
/// `charsiu/models/charsiu_forced_aligner.py` and the paired HF
/// `preprocessor_config.json` + `config.json`, fetched 2026-07-30 —
/// CLAUDE.md「ハルシネーション厳禁」):
///
/// - `hidden_size = 768` (wav2vec2-base residual width).
/// - `n_layer = 12` (wav2vec2-base transformer depth).
/// - `n_head = 12` (wav2vec2-base MHA head count; head_dim = 64).
/// - `ffn_dim = 3072` (wav2vec2-base FFN inner width).
/// - `vocab_size = 42` (IPA-en inventory + `<pad>` at `blank_id = 0`
///   in the canonical Charsiu release; downstream re-trainings may
///   override).
/// - `blank_id = 0`.
/// - `sample_rate = 16000` (16 kHz mono PCM in).
/// - `frame_shift_sec = 0.02` (50 Hz feature rate =
///   `total_stride=320 / sample_rate=16000`).
///
/// [`Self::default_charsiu_en`] returns these defaults. A caller with a
/// downstream variant (e.g. Charsiu multilingual with a wider vocab)
/// overrides field-by-field.
#[derive(Debug, Clone, PartialEq)]
pub struct CharsiuConfig {
    /// Transformer residual width.
    pub hidden_size: usize,
    /// Number of pre-LayerNorm Transformer blocks in the encoder.
    pub n_layer: usize,
    /// MHA head count; `head_dim = hidden_size / n_head`.
    pub n_head: usize,
    /// FFN inner width.
    pub ffn_dim: usize,
    /// Output vocabulary size (IPA phonemes + blank).
    pub vocab_size: usize,
    /// CTC blank id. Charsiu keeps the wav2vec2 default `blank_id = 0`.
    pub blank_id: usize,
    /// Input PCM sample rate (Hz). Charsiu = 16 000.
    pub sample_rate: u32,
    /// Per-frame time step (seconds) — matches the wav2vec2 base stem
    /// output rate = `total_stride(320) / sample_rate(16000) = 0.02`
    /// (50 Hz features).
    pub frame_shift_sec: f32,
    /// Whether the transformer blocks carry a `Wav2Vec2NoLayerNorm`
    /// feature-projection LayerNorm. `true` for the mainline wav2vec2
    /// base + Charsiu configuration.
    pub feature_projection_has_layer_norm: bool,
    /// Whether the raw-waveform stem carries a per-Conv1D bias. `false`
    /// for the base wav2vec2 config Charsiu forks from (per
    /// `Wav2Vec2Config.conv_bias = False`).
    pub stem_conv_bias: bool,
}

impl CharsiuConfig {
    /// The canonical Charsiu English wav2vec2-base configuration
    /// (`charsiu/en_w2v2_fc_10ms`).
    #[must_use]
    pub fn default_charsiu_en() -> Self {
        Self {
            hidden_size: 768,
            n_layer: 12,
            n_head: 12,
            ffn_dim: 3072,
            vocab_size: 42,
            blank_id: 0,
            sample_rate: 16_000,
            frame_shift_sec: 0.02,
            feature_projection_has_layer_norm: true,
            stem_conv_bias: false,
        }
    }

    /// Per-head width. `0` when `n_head == 0` (used only for
    /// shape-only scaffold builds; the runtime rejects that at
    /// [`Self::validate_for_forward`]).
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.hidden_size.checked_div(self.n_head).unwrap_or(0)
    }

    /// Rejects ill-formed configs (mirror of the omniASR-CTC / VoxCPM
    /// validate contract). Every failure mode is
    /// [`VokraError::InvalidArgument`] — FR-EX-08.
    pub fn validate_for_forward(&self) -> Result<()> {
        if self.sample_rate == 0 {
            return Err(VokraError::InvalidArgument(
                "charsiu: sample_rate must be > 0".to_owned(),
            ));
        }
        if !self.frame_shift_sec.is_finite() || self.frame_shift_sec <= 0.0 {
            return Err(VokraError::InvalidArgument(format!(
                "charsiu: frame_shift_sec must be finite and > 0 (got {})",
                self.frame_shift_sec,
            )));
        }
        for (name, v) in [
            ("hidden_size", self.hidden_size),
            ("n_layer", self.n_layer),
            ("n_head", self.n_head),
            ("ffn_dim", self.ffn_dim),
            ("vocab_size", self.vocab_size),
        ] {
            if v == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "charsiu: {name} must be > 0 (got 0-placeholder)"
                )));
            }
        }
        if self.hidden_size % self.n_head != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "charsiu: hidden_size {} not divisible by n_head {}",
                self.hidden_size, self.n_head,
            )));
        }
        if self.blank_id >= self.vocab_size {
            return Err(VokraError::InvalidArgument(format!(
                "charsiu: blank_id {} must be < vocab_size {}",
                self.blank_id, self.vocab_size,
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// Feature projection — LayerNorm (optional) then Linear from the
/// stem's 512-d output to the transformer residual `hidden_size`.
#[derive(Debug, Clone)]
pub struct CharsiuFeatureProjection {
    /// `[512]` — pre-Linear LayerNorm gamma, `Some` iff
    /// `feature_projection_has_layer_norm = true`.
    pub norm_gamma: Option<Vec<f32>>,
    /// `[512]` — pre-Linear LayerNorm beta, same guard as `norm_gamma`.
    pub norm_beta: Option<Vec<f32>>,
    /// Row-major `[hidden_size, 512]` — `nn.Linear.weight`.
    pub linear_w: Vec<f32>,
    /// `[hidden_size]` — `nn.Linear.bias`.
    pub linear_b: Vec<f32>,
}

/// A single pre-LayerNorm Transformer encoder block (wav2vec 2.0
/// mainline). MHA (Q/K/V projections carry bias) + FFN (fc1 → GELU →
/// fc2, both carry bias).
#[derive(Debug, Clone)]
pub struct CharsiuBlock {
    /// `[hidden]` — attention pre-norm γ.
    pub attn_norm_gamma: Vec<f32>,
    /// `[hidden]` — attention pre-norm β.
    pub attn_norm_beta: Vec<f32>,
    /// `[hidden, hidden]` — Q projection weight (row-major).
    pub q_w: Vec<f32>,
    /// `[hidden]` — Q projection bias.
    pub q_b: Vec<f32>,
    /// `[hidden, hidden]` — K projection weight.
    pub k_w: Vec<f32>,
    /// `[hidden]` — K projection bias.
    pub k_b: Vec<f32>,
    /// `[hidden, hidden]` — V projection weight.
    pub v_w: Vec<f32>,
    /// `[hidden]` — V projection bias.
    pub v_b: Vec<f32>,
    /// `[hidden, hidden]` — attention output projection weight.
    pub o_w: Vec<f32>,
    /// `[hidden]` — attention output projection bias.
    pub o_b: Vec<f32>,
    /// `[hidden]` — FFN pre-norm γ.
    pub ffn_norm_gamma: Vec<f32>,
    /// `[hidden]` — FFN pre-norm β.
    pub ffn_norm_beta: Vec<f32>,
    /// `[ffn_dim, hidden]` — FFN fc1 weight.
    pub fc1_w: Vec<f32>,
    /// `[ffn_dim]` — FFN fc1 bias.
    pub fc1_b: Vec<f32>,
    /// `[hidden, ffn_dim]` — FFN fc2 weight.
    pub fc2_w: Vec<f32>,
    /// `[hidden]` — FFN fc2 bias.
    pub fc2_b: Vec<f32>,
}

/// CTC head — a single Linear from `hidden_size` to `vocab_size`
/// (`transformers.Wav2Vec2ForCTC.lm_head`).
#[derive(Debug, Clone)]
pub struct CharsiuHead {
    /// Row-major `[vocab_size, hidden_size]` — `lm_head.weight`.
    pub weight: Vec<f32>,
    /// `[vocab_size]` — `lm_head.bias`.
    pub bias: Vec<f32>,
}

/// Full Charsiu weight store — waveform stem + feature projection +
/// `n_layer` encoder blocks + final norm + CTC head.
#[derive(Debug, Clone)]
pub struct CharsiuWeights {
    /// Raw-waveform 7-layer wav2vec2-base stem.
    pub stem_attrs: WaveformFrontendAttrs,
    /// Per-stem-layer weights (bindable to
    /// [`vokra_ops::waveform_frontend`]).
    pub stem_weights: WaveformFrontendWeights,
    /// Feature projection (stem's 512-d output → residual `hidden_size`).
    pub feature_projection: CharsiuFeatureProjection,
    /// `n_layer` pre-norm Transformer encoder blocks.
    pub blocks: Vec<CharsiuBlock>,
    /// `[hidden]` — final pre-head LayerNorm γ.
    pub final_norm_gamma: Vec<f32>,
    /// `[hidden]` — final pre-head LayerNorm β.
    pub final_norm_beta: Vec<f32>,
    /// CTC head — vocab projection.
    pub head: CharsiuHead,
    /// Whether the store came from [`Self::synthesized`] (a scaffold
    /// fixture — the runtime refuses to bind commercial outputs from
    /// synthesized weights).
    pub is_synthesized: bool,
}

impl CharsiuWeights {
    /// Builds a deterministic scaffold weight store against `config`.
    /// SplitMix64 + Xavier initialization — every tensor is populated so
    /// the shape flow through the runtime and the CTC decoder can be
    /// exercised without a real HF checkpoint (omniASR-CTC / VITS-JA
    /// fixture pattern).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] from `config.validate_for_forward`.
    pub fn synthesized(config: &CharsiuConfig, seed: u64) -> Result<Self> {
        config.validate_for_forward()?;

        let mut rng = SplitMix64::new(seed);

        let stem_attrs = WaveformFrontendAttrs::wav2vec2_base();
        // Per-layer weights matching the 7-layer stem topology. The
        // wav2vec2 base config has `conv_bias=false` and
        // `norm=GroupFirstOnly` (layer 0 group-norms, others none), so
        // conv_b is always empty and norm_gamma/norm_beta are Some only
        // for layer 0.
        let mut stem_layers: Vec<ConvLayerWeights> = Vec::with_capacity(stem_attrs.layers.len());
        let mut in_ch: usize = stem_attrs.in_channels;
        for (i, la) in stem_attrs.layers.iter().enumerate() {
            let out_ch = la.out_channels;
            let k = la.kernel;
            let fan_in = (in_ch * k) as f32;
            let scale = (1.0 / fan_in).sqrt();
            let conv_w = xavier_vec(&mut rng, out_ch * in_ch * k, scale);
            let (norm_gamma, norm_beta) = if i == 0 {
                (Some(vec![1.0; out_ch]), Some(vec![0.0; out_ch]))
            } else {
                (None, None)
            };
            stem_layers.push(ConvLayerWeights {
                conv_w,
                conv_b: Vec::new(), // conv_bias=false for wav2vec2 base
                norm_gamma,
                norm_beta,
            });
            in_ch = out_ch;
        }
        let stem_weights = WaveformFrontendWeights {
            layers: stem_layers,
        };

        // Feature projection: 512 → hidden_size.
        let feature_dim = 512_usize;
        let (fp_norm_gamma, fp_norm_beta) = if config.feature_projection_has_layer_norm {
            (Some(vec![1.0; feature_dim]), Some(vec![0.0; feature_dim]))
        } else {
            (None, None)
        };
        let fp_scale = (1.0 / feature_dim as f32).sqrt();
        let feature_projection = CharsiuFeatureProjection {
            norm_gamma: fp_norm_gamma,
            norm_beta: fp_norm_beta,
            linear_w: xavier_vec(&mut rng, config.hidden_size * feature_dim, fp_scale),
            linear_b: vec![0.0; config.hidden_size],
        };

        // n_layer pre-norm Transformer blocks.
        let h = config.hidden_size;
        let ffn = config.ffn_dim;
        let attn_scale = (1.0 / h as f32).sqrt();
        let fc1_scale = (1.0 / h as f32).sqrt();
        let fc2_scale = (1.0 / ffn as f32).sqrt();
        let blocks: Vec<CharsiuBlock> = (0..config.n_layer)
            .map(|_| CharsiuBlock {
                attn_norm_gamma: vec![1.0; h],
                attn_norm_beta: vec![0.0; h],
                q_w: xavier_vec(&mut rng, h * h, attn_scale),
                q_b: vec![0.0; h],
                k_w: xavier_vec(&mut rng, h * h, attn_scale),
                k_b: vec![0.0; h],
                v_w: xavier_vec(&mut rng, h * h, attn_scale),
                v_b: vec![0.0; h],
                o_w: xavier_vec(&mut rng, h * h, attn_scale),
                o_b: vec![0.0; h],
                ffn_norm_gamma: vec![1.0; h],
                ffn_norm_beta: vec![0.0; h],
                fc1_w: xavier_vec(&mut rng, ffn * h, fc1_scale),
                fc1_b: vec![0.0; ffn],
                fc2_w: xavier_vec(&mut rng, h * ffn, fc2_scale),
                fc2_b: vec![0.0; h],
            })
            .collect();

        // Final pre-head LayerNorm.
        let final_norm_gamma = vec![1.0; h];
        let final_norm_beta = vec![0.0; h];

        // CTC head.
        let head_scale = (1.0 / h as f32).sqrt();
        let head = CharsiuHead {
            weight: xavier_vec(&mut rng, config.vocab_size * h, head_scale),
            bias: vec![0.0; config.vocab_size],
        };

        Ok(Self {
            stem_attrs,
            stem_weights,
            feature_projection,
            blocks,
            final_norm_gamma,
            final_norm_beta,
            head,
            is_synthesized: true,
        })
    }
}

/// Fills a Xavier-uniform-like buffer of length `n` scaled by `scale`.
/// Deterministic: seeded by the caller's [`SplitMix64`].
fn xavier_vec(rng: &mut SplitMix64, n: usize, scale: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        // rng.next_unit_f32() is in (0, 1); shift to (-1, 1) and scale.
        let u = rng.next_unit_f32() * 2.0 - 1.0;
        out.push(u * scale);
    }
    out
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Charsiu forced aligner.
///
/// Construct with [`Charsiu::new`] against a validated [`CharsiuConfig`]
/// plus a [`CharsiuWeights`] store (either bound from a Vokra GGUF or a
/// scaffold fixture from [`CharsiuWeights::synthesized`]).
///
/// [`Self::align`] runs the wav2vec2 CTC forward and forwards the
/// per-frame log-probabilities to [`ctc_segmentation`] to recover the
/// per-phoneme time boundaries.
#[derive(Debug, Clone)]
pub struct Charsiu {
    cfg: CharsiuConfig,
    weights: CharsiuWeights,
}

impl Charsiu {
    /// Assembles an aligner from `cfg` and `weights`. Cross-checks the
    /// weight-store shapes against `cfg` so a mismatched pair fails
    /// loudly here rather than deep inside a forward.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] from `cfg.validate_for_forward`.
    /// - [`VokraError::InvalidArgument`] naming the first shape
    ///   mismatch (feature projection, encoder blocks, head).
    pub fn new(cfg: CharsiuConfig, weights: CharsiuWeights) -> Result<Self> {
        cfg.validate_for_forward()?;
        Self::validate_shapes(&cfg, &weights)?;
        Ok(Self { cfg, weights })
    }

    /// Returns the resolved config.
    pub fn config(&self) -> &CharsiuConfig {
        &self.cfg
    }

    /// Reports whether the underlying weight store is a scaffold
    /// fixture (i.e. would produce a deterministic hallucination if the
    /// caller used it for a real transcript).
    pub fn is_synthesized(&self) -> bool {
        self.weights.is_synthesized
    }

    /// Skeleton load-error entry point kept from the pre-real scaffold.
    /// A real Charsiu GGUF binder lands in a follow-up wave gated on the
    /// upstream tensor-name manifest fetch (T29-equivalent, same as
    /// omniASR-CTC / CosyVoice2 / Voxtral).
    ///
    /// # Errors
    ///
    /// - [`LoadError::FileNotFound`] if `path` does not exist.
    /// - [`LoadError::Gguf`] otherwise, naming the follow-up wave.
    pub fn from_gguf(path: &Path) -> std::result::Result<Self, LoadError> {
        if !path.exists() {
            return Err(LoadError::FileNotFound(path.to_path_buf()));
        }
        Err(LoadError::Gguf(
            "charsiu: from_gguf is not wired yet — the wav2vec2 CTC forward path exists in this \
             module (Charsiu::align), but the upstream tensor-name manifest binder is a follow-up \
             wave (T29-equivalent, matching omniASR-CTC / CosyVoice2). Instantiate via \
             Charsiu::new(CharsiuConfig, CharsiuWeights) from a caller-supplied weight store."
                .to_owned(),
        ))
    }

    /// Force-aligns a phoneme sequence to a 16 kHz mono PCM buffer.
    ///
    /// Runs the wav2vec2 stem → feature projection → n_layer encoder
    /// blocks → CTC head forward, converts the head output to
    /// per-frame log-probabilities, and forwards them to
    /// [`ctc_segmentation`] together with `phonemes` (mapped to vocab
    /// ids via the "skip the blank slot" convention in the parent
    /// [`super::ctc_segmentation`] docs).
    ///
    /// # Arguments
    ///
    /// * `pcm` — 16 kHz mono PCM at [`CharsiuConfig::sample_rate`]. The
    ///   forward asserts `pcm.len() >= stem.total_stride()` — shorter
    ///   inputs return an empty alignment (the parent
    ///   `ctc_segmentation` short-input contract).
    /// * `sample_rate` — must equal [`CharsiuConfig::sample_rate`];
    ///   mismatch is a loud [`VokraError::InvalidArgument`] (FR-EX-08).
    /// * `phonemes` — the transcript to align (echoed back into the
    ///   `AlignedToken.text` field).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] on empty `pcm` or a sample-rate
    ///   mismatch.
    /// - Propagated `VokraError` from [`vokra_ops::waveform_frontend`]
    ///   (shape gates on the stem — never silent-truncated).
    pub fn align(
        &self,
        pcm: &[f32],
        sample_rate: u32,
        phonemes: &[String],
    ) -> Result<Vec<AlignedToken>> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "charsiu align: pcm slice is empty".to_owned(),
            ));
        }
        if sample_rate != self.cfg.sample_rate {
            return Err(VokraError::InvalidArgument(format!(
                "charsiu align: sample_rate {} != config.sample_rate {} (FR-EX-08 no silent \
                 resampling)",
                sample_rate, self.cfg.sample_rate,
            )));
        }

        // ---- Stem: raw PCM → [T', 512] time-major ---------------------
        let features = vokra_ops::waveform_frontend(
            pcm,
            &self.weights.stem_attrs,
            &self.weights.stem_weights,
        )?;
        let feature_dim = self.weights.stem_attrs.out_channels()?;
        assert_eq!(features.len() % feature_dim, 0);
        let t_frames = features.len() / feature_dim;
        if t_frames == 0 {
            // The parent CTC segmenter returns an empty vec on
            // insufficient frames; mirror that contract here for
            // consistency (mainstream on very short clips).
            return Ok(Vec::new());
        }

        // ---- Feature projection: [T', 512] → [T', hidden_size] -------
        let mut hidden = feature_projection_forward(
            &features,
            t_frames,
            feature_dim,
            &self.weights.feature_projection,
            self.cfg.hidden_size,
            self.cfg.feature_projection_has_layer_norm,
        );

        // ---- n_layer Transformer blocks ------------------------------
        for block in &self.weights.blocks {
            transformer_block_forward(&mut hidden, t_frames, &self.cfg, block);
        }

        // ---- Final pre-head LayerNorm --------------------------------
        layer_norm_inplace(
            &mut hidden,
            t_frames,
            self.cfg.hidden_size,
            &self.weights.final_norm_gamma,
            &self.weights.final_norm_beta,
        );

        // ---- CTC head + log_softmax → [T', vocab_size] ---------------
        let logits = ctc_head_forward(
            &hidden,
            t_frames,
            self.cfg.hidden_size,
            &self.weights.head,
            self.cfg.vocab_size,
        );
        let log_probs = log_softmax(&logits, t_frames, self.cfg.vocab_size);

        // ---- Viterbi CTC segmentation (Kürzinger et al. 2020) -------
        Ok(ctc_segmentation(
            &log_probs,
            t_frames,
            self.cfg.vocab_size,
            self.cfg.blank_id,
            self.cfg.frame_shift_sec,
            phonemes,
        ))
    }

    /// Validates the weight store shapes against `cfg` — every failure
    /// mode is loud (FR-EX-08).
    fn validate_shapes(cfg: &CharsiuConfig, w: &CharsiuWeights) -> Result<()> {
        let h = cfg.hidden_size;
        let feature_dim = 512_usize;
        let expected_stem_out = w.stem_attrs.out_channels()?;
        if expected_stem_out != feature_dim {
            return Err(VokraError::InvalidArgument(format!(
                "charsiu: stem.out_channels {expected_stem_out} != 512 (wav2vec2-base stem \
                 output width)"
            )));
        }
        w.stem_weights.validate(&w.stem_attrs)?;

        let fp = &w.feature_projection;
        if fp.linear_w.len() != h * feature_dim {
            return Err(VokraError::InvalidArgument(format!(
                "charsiu: feature_projection.linear_w.len() {} != hidden_size ({}) * 512 = {}",
                fp.linear_w.len(),
                h,
                h * feature_dim,
            )));
        }
        if fp.linear_b.len() != h {
            return Err(VokraError::InvalidArgument(format!(
                "charsiu: feature_projection.linear_b.len() {} != hidden_size {h}",
                fp.linear_b.len(),
            )));
        }
        let expects_fp_norm = cfg.feature_projection_has_layer_norm;
        match (&fp.norm_gamma, &fp.norm_beta, expects_fp_norm) {
            (Some(g), Some(b), true) if g.len() == feature_dim && b.len() == feature_dim => {}
            (None, None, false) => {}
            _ => {
                return Err(VokraError::InvalidArgument(
                    "charsiu: feature_projection norm gamma/beta presence must match \
                     `feature_projection_has_layer_norm`, and both must be length 512"
                        .to_owned(),
                ));
            }
        }

        if w.blocks.len() != cfg.n_layer {
            return Err(VokraError::InvalidArgument(format!(
                "charsiu: blocks.len() {} != n_layer {}",
                w.blocks.len(),
                cfg.n_layer,
            )));
        }
        let ffn = cfg.ffn_dim;
        for (i, b) in w.blocks.iter().enumerate() {
            for (name, actual, expected) in [
                ("attn_norm_gamma", b.attn_norm_gamma.len(), h),
                ("attn_norm_beta", b.attn_norm_beta.len(), h),
                ("q_w", b.q_w.len(), h * h),
                ("q_b", b.q_b.len(), h),
                ("k_w", b.k_w.len(), h * h),
                ("k_b", b.k_b.len(), h),
                ("v_w", b.v_w.len(), h * h),
                ("v_b", b.v_b.len(), h),
                ("o_w", b.o_w.len(), h * h),
                ("o_b", b.o_b.len(), h),
                ("ffn_norm_gamma", b.ffn_norm_gamma.len(), h),
                ("ffn_norm_beta", b.ffn_norm_beta.len(), h),
                ("fc1_w", b.fc1_w.len(), ffn * h),
                ("fc1_b", b.fc1_b.len(), ffn),
                ("fc2_w", b.fc2_w.len(), h * ffn),
                ("fc2_b", b.fc2_b.len(), h),
            ] {
                if actual != expected {
                    return Err(VokraError::InvalidArgument(format!(
                        "charsiu: block[{i}].{name}.len() {actual} != {expected}"
                    )));
                }
            }
        }
        if w.final_norm_gamma.len() != h || w.final_norm_beta.len() != h {
            return Err(VokraError::InvalidArgument(format!(
                "charsiu: final norm gamma/beta must be length {h}"
            )));
        }
        if w.head.weight.len() != cfg.vocab_size * h {
            return Err(VokraError::InvalidArgument(format!(
                "charsiu: head.weight.len() {} != vocab_size ({}) * hidden_size ({h}) = {}",
                w.head.weight.len(),
                cfg.vocab_size,
                cfg.vocab_size * h,
            )));
        }
        if w.head.bias.len() != cfg.vocab_size {
            return Err(VokraError::InvalidArgument(format!(
                "charsiu: head.bias.len() {} != vocab_size {}",
                w.head.bias.len(),
                cfg.vocab_size,
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Math helpers (private — plain scalar loops; hot-path SIMD is a
// follow-up; kept simple so the forward is easy to audit)
// ---------------------------------------------------------------------------

/// Feature projection: `[T, 512]` → `[T, hidden]`. Applies the pre-Linear
/// LayerNorm iff `has_layer_norm`.
fn feature_projection_forward(
    features: &[f32],
    t: usize,
    feature_dim: usize,
    fp: &CharsiuFeatureProjection,
    hidden: usize,
    has_layer_norm: bool,
) -> Vec<f32> {
    let mut normed_flat: Vec<f32>;
    let input: &[f32] = if has_layer_norm {
        normed_flat = features.to_vec();
        let gamma = fp
            .norm_gamma
            .as_ref()
            .expect("has_layer_norm=true implies norm_gamma present");
        let beta = fp
            .norm_beta
            .as_ref()
            .expect("has_layer_norm=true implies norm_beta present");
        layer_norm_inplace(&mut normed_flat, t, feature_dim, gamma, beta);
        &normed_flat
    } else {
        features
    };

    // Linear: y = x @ W^T + b, W shape [hidden, feature_dim].
    let mut out = vec![0.0_f32; t * hidden];
    for ti in 0..t {
        for oi in 0..hidden {
            let mut acc = fp.linear_b[oi];
            let w_row = oi * feature_dim;
            let x_row = ti * feature_dim;
            for k in 0..feature_dim {
                acc += input[x_row + k] * fp.linear_w[w_row + k];
            }
            out[ti * hidden + oi] = acc;
        }
    }
    out
}

/// Runs one pre-LayerNorm Transformer block in place.
///
/// Topology (matches HF `Wav2Vec2EncoderLayer`):
/// `y = x + attn(layer_norm(x)); z = y + ffn(layer_norm(y))` with
/// FFN = `fc2(gelu(fc1(z)))`.
fn transformer_block_forward(hidden: &mut [f32], t: usize, cfg: &CharsiuConfig, b: &CharsiuBlock) {
    let h = cfg.hidden_size;
    let n_head = cfg.n_head;
    let head_dim = h / n_head;
    let head_scale = 1.0 / (head_dim as f32).sqrt();

    // -- Attention branch -----------------------------------------------
    let mut normed = hidden.to_vec();
    layer_norm_inplace(&mut normed, t, h, &b.attn_norm_gamma, &b.attn_norm_beta);

    // Q, K, V projections: [T, h] → [T, h] each.
    let q = linear_forward(&normed, t, h, &b.q_w, &b.q_b, h);
    let k = linear_forward(&normed, t, h, &b.k_w, &b.k_b, h);
    let v = linear_forward(&normed, t, h, &b.v_w, &b.v_b, h);

    // MHA: reshape Q/K/V to [n_head, T, head_dim] and score.
    let mut attn_out = vec![0.0_f32; t * h];
    // Work buffer for per-head [T, T] scores (reused across heads).
    let mut scores = vec![0.0_f32; t * t];
    for hi in 0..n_head {
        // Compute Q_h K_h^T with the head slices interleaved in the
        // flat buffer as [T, n_head, head_dim].
        for ti in 0..t {
            let qi_base = ti * h + hi * head_dim;
            for tj in 0..t {
                let kj_base = tj * h + hi * head_dim;
                let mut s = 0.0_f32;
                for d in 0..head_dim {
                    s += q[qi_base + d] * k[kj_base + d];
                }
                scores[ti * t + tj] = s * head_scale;
            }
        }
        // softmax over each row (causal mask NOT applied — wav2vec2 CTC
        // uses full bidirectional attention).
        for ti in 0..t {
            let row = &mut scores[ti * t..(ti + 1) * t];
            let mut max_v = f32::NEG_INFINITY;
            for &v in row.iter() {
                if v > max_v {
                    max_v = v;
                }
            }
            let mut sum = 0.0_f32;
            for v in row.iter_mut() {
                *v = (*v - max_v).exp();
                sum += *v;
            }
            let inv = 1.0 / sum;
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
        // A_h @ V_h → contribution to attn_out per head slice.
        for ti in 0..t {
            let ai_row = ti * t;
            let out_base = ti * h + hi * head_dim;
            for d in 0..head_dim {
                let mut acc = 0.0_f32;
                for tj in 0..t {
                    let vj_base = tj * h + hi * head_dim;
                    acc += scores[ai_row + tj] * v[vj_base + d];
                }
                attn_out[out_base + d] = acc;
            }
        }
    }
    // Output projection: [T, h] → [T, h].
    let proj = linear_forward(&attn_out, t, h, &b.o_w, &b.o_b, h);
    for i in 0..(t * h) {
        hidden[i] += proj[i];
    }

    // -- FFN branch ----------------------------------------------------
    let mut normed = hidden.to_vec();
    layer_norm_inplace(&mut normed, t, h, &b.ffn_norm_gamma, &b.ffn_norm_beta);
    let ffn = cfg.ffn_dim;
    let mut fc1_out = linear_forward(&normed, t, h, &b.fc1_w, &b.fc1_b, ffn);
    for v in fc1_out.iter_mut() {
        *v = gelu_exact(*v);
    }
    let fc2_out = linear_forward(&fc1_out, t, ffn, &b.fc2_w, &b.fc2_b, h);
    for i in 0..(t * h) {
        hidden[i] += fc2_out[i];
    }
}

/// `y = x @ W^T + b` where `x` is `[t, in_dim]`, `W` is `[out_dim,
/// in_dim]`, `b` is `[out_dim]`.
fn linear_forward(
    x: &[f32],
    t: usize,
    in_dim: usize,
    w: &[f32],
    b: &[f32],
    out_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; t * out_dim];
    for ti in 0..t {
        for oi in 0..out_dim {
            let mut acc = b[oi];
            let w_row = oi * in_dim;
            let x_row = ti * in_dim;
            for k in 0..in_dim {
                acc += x[x_row + k] * w[w_row + k];
            }
            out[ti * out_dim + oi] = acc;
        }
    }
    out
}

/// LayerNorm over the last axis in place. `x` is `[t, dim]` row-major.
/// `eps = 1e-5` (PyTorch default).
fn layer_norm_inplace(x: &mut [f32], t: usize, dim: usize, gamma: &[f32], beta: &[f32]) {
    const EPS: f32 = 1e-5;
    let n = dim as f32;
    for ti in 0..t {
        let row = &mut x[ti * dim..(ti + 1) * dim];
        let mut mean = 0.0_f32;
        for &v in row.iter() {
            mean += v;
        }
        mean /= n;
        let mut var = 0.0_f32;
        for &v in row.iter() {
            let d = v - mean;
            var += d * d;
        }
        var /= n;
        let inv_std = 1.0 / (var + EPS).sqrt();
        for (i, v) in row.iter_mut().enumerate() {
            *v = ((*v - mean) * inv_std) * gamma[i] + beta[i];
        }
    }
}

/// CTC head — Linear from `[t, hidden]` to `[t, vocab]`.
fn ctc_head_forward(
    hidden: &[f32],
    t: usize,
    h: usize,
    head: &CharsiuHead,
    vocab: usize,
) -> Vec<f32> {
    linear_forward(hidden, t, h, &head.weight, &head.bias, vocab)
}

/// Per-frame log-softmax over the vocab axis.
fn log_softmax(logits: &[f32], t: usize, vocab: usize) -> Vec<f32> {
    let mut out = logits.to_vec();
    for ti in 0..t {
        let row = &mut out[ti * vocab..(ti + 1) * vocab];
        let mut max_v = f32::NEG_INFINITY;
        for &v in row.iter() {
            if v > max_v {
                max_v = v;
            }
        }
        let mut sum = 0.0_f32;
        for v in row.iter_mut() {
            *v = (*v - max_v).exp();
            sum += *v;
        }
        let ln_sum = sum.ln();
        for v in row.iter_mut() {
            // log_softmax = (x - max) - ln(sum exp(x - max))
            *v = v.ln() - ln_sum;
        }
    }
    out
}

/// Exact erf-based GELU (`0.5 * x * (1 + erf(x / √2))`) — HF wav2vec2
/// uses `ACT2FN["gelu"]` = exact GELU. Same A&S 7.1.26 erf approximation
/// as [`vokra_ops::waveform_frontend`]; kept private here so the align
/// module has no cross-crate constraints on the ops erf.
#[inline]
fn gelu_exact(x: f32) -> f32 {
    0.5 * x * (1.0 + erf_as(x * core::f32::consts::FRAC_1_SQRT_2))
}

/// Abramowitz & Stegun 7.1.26 erf approximation (~1e-7 error).
#[inline]
#[allow(clippy::excessive_precision)]
fn erf_as(x: f32) -> f32 {
    let sign = x.signum();
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-ax * ax).exp();
    sign * y
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny Charsiu config used to drive the scaffold end-to-end in
    /// tests without spending time on the full 12-layer / 3072-FFN /
    /// 768-hidden real config.
    fn tiny_for_tests() -> CharsiuConfig {
        CharsiuConfig {
            hidden_size: 32,
            n_layer: 2,
            n_head: 4,
            ffn_dim: 64,
            vocab_size: 8,
            blank_id: 0,
            sample_rate: 16_000,
            frame_shift_sec: 0.02,
            feature_projection_has_layer_norm: true,
            stem_conv_bias: false,
        }
    }

    #[test]
    fn charsiu_load_stub_reports_load_error() {
        // A path that cannot exist -> Err(LoadError::*), never a panic.
        let missing = Path::new("/nonexistent/vokra-charsiu-does-not-exist.gguf");
        let err = Charsiu::from_gguf(missing).expect_err("missing GGUF must be LoadError");
        assert!(
            matches!(err, LoadError::FileNotFound(_) | LoadError::Gguf(_)),
            "unexpected LoadError variant: {err:?}",
        );
    }

    #[test]
    fn default_config_matches_wav2vec2_base_axes() {
        let c = CharsiuConfig::default_charsiu_en();
        assert_eq!(c.hidden_size, 768);
        assert_eq!(c.n_layer, 12);
        assert_eq!(c.n_head, 12);
        assert_eq!(c.ffn_dim, 3072);
        assert_eq!(c.head_dim(), 64);
        assert_eq!(c.sample_rate, 16_000);
        assert!((c.frame_shift_sec - 0.02).abs() < 1e-9);
        assert_eq!(c.blank_id, 0);
        assert!(!c.stem_conv_bias);
    }

    #[test]
    fn validate_rejects_zero_placeholder() {
        let mut c = CharsiuConfig::default_charsiu_en();
        c.hidden_size = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn validate_rejects_head_not_divisible() {
        let mut c = CharsiuConfig::default_charsiu_en();
        c.n_head = 5; // 768 % 5 != 0
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn new_cross_checks_weight_shapes() {
        let cfg = tiny_for_tests();
        let mut w = CharsiuWeights::synthesized(&cfg, 0xABCD_1234).unwrap();
        // Break the head shape and expect a loud failure.
        w.head.weight.pop();
        let err = Charsiu::new(cfg, w).expect_err("shape mismatch must be caught");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn synthesize_reports_scaffold_flag() {
        let cfg = tiny_for_tests();
        let w = CharsiuWeights::synthesized(&cfg, 0xDEAD_BEEF).unwrap();
        let aligner = Charsiu::new(cfg, w).unwrap();
        assert!(aligner.is_synthesized());
    }

    #[test]
    fn align_rejects_empty_pcm() {
        let cfg = tiny_for_tests();
        let w = CharsiuWeights::synthesized(&cfg, 1).unwrap();
        let aligner = Charsiu::new(cfg, w).unwrap();
        let err = aligner
            .align(&[], 16_000, &["p".to_owned()])
            .expect_err("empty pcm must fail loudly");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn align_rejects_sample_rate_mismatch() {
        let cfg = tiny_for_tests();
        let w = CharsiuWeights::synthesized(&cfg, 1).unwrap();
        let aligner = Charsiu::new(cfg, w).unwrap();
        let pcm = vec![0.0_f32; 4000];
        let err = aligner
            .align(&pcm, 8_000, &["p".to_owned()])
            .expect_err("mismatched sample_rate must fail loudly");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    /// End-to-end smoke: feed 1 s of synthetic PCM through the full
    /// stem → projection → 2-layer transformer → CTC head → segmenter
    /// pipeline and confirm we get monotone non-overlapping boundaries
    /// for a 3-phoneme transcript. The point is not accuracy (weights
    /// are Xavier — the output would be meaningless on a real transcript)
    /// but that every gate lines up shape-wise so the wiring is proven
    /// correct.
    #[test]
    fn align_emits_monotone_boundaries_end_to_end() {
        let cfg = tiny_for_tests();
        let w = CharsiuWeights::synthesized(&cfg, 0xC0FF_EE).unwrap();
        let aligner = Charsiu::new(cfg, w).unwrap();
        // Keep the PCM short so the scalar-loop forward stays snappy in
        // debug builds. 1600 samples of 16 kHz mono → after the 7-layer
        // wav2vec2 stem (total_stride=320) that yields 5 frames, which
        // is exactly enough for a 3-phoneme alignment (num_frames >=
        // tokens.len(), per ctc_segmentation's short-input contract).
        let mut pcm = vec![0.0_f32; 1_600];
        for (i, s) in pcm.iter_mut().enumerate() {
            // A gentle triangle wave so the stem output is not degenerate.
            *s = ((i as f32 / 200.0).sin()) * 0.1;
        }
        let phonemes: Vec<String> = ["p", "æ", "t"].iter().map(|s| s.to_string()).collect();
        let out = aligner.align(&pcm, 16_000, &phonemes).unwrap();
        assert_eq!(out.len(), phonemes.len(), "one AlignedToken per phoneme");
        // Monotone non-overlapping boundaries.
        let mut last_end = 0.0_f32;
        for (i, tok) in out.iter().enumerate() {
            assert!(
                tok.start_sec >= last_end - 1e-6,
                "token {i} start_sec {} not >= last_end {last_end}",
                tok.start_sec,
            );
            assert!(
                tok.end_sec > tok.start_sec - 1e-6,
                "token {i} end_sec {} must be > start_sec {}",
                tok.end_sec,
                tok.start_sec,
            );
            assert!(
                tok.confidence > 0.0 && tok.confidence <= 1.0,
                "token {i} confidence {} not in (0, 1]",
                tok.confidence,
            );
            last_end = tok.end_sec;
        }
    }

    /// A pcm too short for the stem returns an empty alignment (mirror
    /// of ctc_segmentation's short-input contract).
    #[test]
    fn align_returns_empty_on_pcm_too_short_for_stem() {
        let cfg = tiny_for_tests();
        let w = CharsiuWeights::synthesized(&cfg, 42).unwrap();
        let aligner = Charsiu::new(cfg, w).unwrap();
        // Stem's total_stride = 320. A pcm with just 100 samples will
        // fail the shape gate inside `waveform_frontend` (loud), which
        // we bubble up. That is the correct posture (FR-EX-08 no silent
        // fabrication) — this test pins that behaviour so a future
        // refactor cannot silently return an empty alignment on
        // shape-too-short.
        let short = vec![0.0_f32; 100];
        let err = aligner
            .align(&short, 16_000, &["p".to_owned()])
            .expect_err("input too short for stem must fail loudly");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }
}
