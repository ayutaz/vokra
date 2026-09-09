//! Charsiu English 10 ms neural forced aligner.
//!
//! The implementation is pinned to `charsiu/en_w2v2_fc_10ms` revision
//! `e9bf8dd314313fc57f6e4d0b5425bde4bbeac80f`. Unlike a stock wav2vec2-base
//! CTC model, this checkpoint's last feature-convolution stride is **1**, so
//! the total stride is 160 samples (10 ms at 16 kHz). Its encoder includes
//! the grouped 128-tap positional convolution and uses Hugging Face's
//! `do_stable_layer_norm=false` **post-norm** block topology.
//!
//! [`Charsiu::from_gguf`] binds the exact writer contract emitted by
//! `vokra-convert --model charsiu`: the official 42-entry phone inventory,
//! all stem/projection/encoder/head tensors, and the converter-folded
//! positional-convolution weight norm. [`Charsiu::align`] reproduces the
//! upstream silence-mask plus monotone DTW forced alignment. It returns one
//! [`AlignedToken`] per caller-supplied non-silence phone; long model-predicted
//! silence runs remain gaps between returned phone intervals.

use std::path::Path;

use vokra_backend_cpu::kernels;
use vokra_core::gguf::{GgufFile, GgufMetadataValue, GgufValueType, chunks};
use vokra_core::rng::SplitMix64;
use vokra_core::{Result, VokraError};
use vokra_ops::{
    ConvLayerAttrs, ConvLayerWeights, Norm, WaveformFrontendAttrs, WaveformFrontendWeights,
};

use crate::compute::Compute;

use super::{AlignedToken, LoadError};

/// The `vokra.model.arch` value a Charsiu GGUF must carry.
///
/// This is the **reader** half of the writer contract in
/// `crates/vokra-convert/src/models/charsiu.rs`. The tag is deliberately
/// duplicated across the writer and reader so each side fails closed if the
/// other changes. It is **not** transcribed from any upstream artifact
/// (upstream Charsiu ships HF `Wav2Vec2ForCTC` weights, which carry no Vokra
/// arch tag at all).
///
/// Deliberately distinct from `wav2vec2_ctc` (the generic Meta
/// wav2vec2 + CTC ASR head, which emits characters/letters) even though
/// the two share a topology: Charsiu's head is an **IPA phoneme**
/// inventory used for forced alignment, so aliasing the tags would let a
/// letter-vocab checkpoint silently produce nonsense phoneme boundaries.
pub const EXPECTED_ARCH: &str = "charsiu";
const EXPECTED_REVISION: &str = "e9bf8dd314313fc57f6e4d0b5425bde4bbeac80f";
const EXPECTED_CHECKPOINT_SHA256: &str =
    "6dc8a18422db7c22e951d5f72dc2afc267b942eb0b8459ac6dcc0cf412536de1";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Config for the canonical Charsiu English frame classifier.
#[derive(Debug, Clone, PartialEq)]
pub struct CharsiuConfig {
    /// Transformer residual width.
    pub hidden_size: usize,
    /// Number of post-LayerNorm Transformer blocks in the encoder.
    pub n_layer: usize,
    /// MHA head count; `head_dim = hidden_size / n_head`.
    pub n_head: usize,
    /// FFN inner width.
    pub ffn_dim: usize,
    /// Output vocabulary size (phones + `[SIL]` / `[UNK]` / `[PAD]`).
    pub vocab_size: usize,
    /// Model class used to identify silence runs (`[SIL]`, canonical id 0).
    pub silence_id: usize,
    /// Padding label id (`[PAD]`, canonical id 41). It is not a CTC blank.
    pub pad_id: usize,
    /// Input PCM sample rate (Hz). Charsiu = 16 000.
    pub sample_rate: u32,
    /// Per-frame time step (canonical total stride 160 / 16 kHz = 0.01 s).
    pub frame_shift_sec: f32,
    /// LayerNorm epsilon used by every Wav2Vec2 norm.
    pub layer_norm_eps: f32,
    /// Positional convolution kernel width.
    pub pos_conv_kernel: usize,
    /// Positional convolution group count.
    pub pos_conv_groups: usize,
    /// Minimum consecutive `[SIL]` argmax frames considered real silence.
    pub silence_threshold: usize,
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
            silence_id: 0,
            pad_id: 41,
            sample_rate: 16_000,
            frame_shift_sec: 0.01,
            layer_norm_eps: 1e-5,
            pos_conv_kernel: 128,
            pos_conv_groups: 16,
            silence_threshold: 4,
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
        if !self.layer_norm_eps.is_finite() || self.layer_norm_eps <= 0.0 {
            return Err(VokraError::InvalidArgument(format!(
                "charsiu: layer_norm_eps must be finite and > 0 (got {})",
                self.layer_norm_eps,
            )));
        }
        for (name, v) in [
            ("hidden_size", self.hidden_size),
            ("n_layer", self.n_layer),
            ("n_head", self.n_head),
            ("ffn_dim", self.ffn_dim),
            ("vocab_size", self.vocab_size),
            ("pos_conv_kernel", self.pos_conv_kernel),
            ("pos_conv_groups", self.pos_conv_groups),
            ("silence_threshold", self.silence_threshold),
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
        if self.silence_id >= self.vocab_size || self.pad_id >= self.vocab_size {
            return Err(VokraError::InvalidArgument(format!(
                "charsiu: silence_id {} and pad_id {} must both be < vocab_size {}",
                self.silence_id, self.pad_id, self.vocab_size,
            )));
        }
        if self.hidden_size % self.pos_conv_groups != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "charsiu: hidden_size {} not divisible by pos_conv_groups {}",
                self.hidden_size, self.pos_conv_groups,
            )));
        }
        Ok(())
    }
}

fn charsiu_stem_attrs() -> WaveformFrontendAttrs {
    WaveformFrontendAttrs {
        in_channels: 1,
        layers: vec![
            ConvLayerAttrs {
                out_channels: 512,
                kernel: 10,
                stride: 5,
            },
            ConvLayerAttrs {
                out_channels: 512,
                kernel: 3,
                stride: 2,
            },
            ConvLayerAttrs {
                out_channels: 512,
                kernel: 3,
                stride: 2,
            },
            ConvLayerAttrs {
                out_channels: 512,
                kernel: 3,
                stride: 2,
            },
            ConvLayerAttrs {
                out_channels: 512,
                kernel: 3,
                stride: 2,
            },
            ConvLayerAttrs {
                out_channels: 512,
                kernel: 2,
                stride: 2,
            },
            ConvLayerAttrs {
                out_channels: 512,
                kernel: 2,
                stride: 1,
            },
        ],
        norm: Norm::GroupFirstOnly,
        conv_bias: false,
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

/// Weight-norm-folded grouped positional convolution.
#[derive(Debug, Clone)]
pub struct CharsiuPosConv {
    /// `[hidden, hidden / groups, kernel]` in PyTorch Conv1D layout.
    pub weight: Vec<f32>,
    /// `[hidden]`.
    pub bias: Vec<f32>,
}

/// A single post-LayerNorm Wav2Vec2 encoder block.
#[derive(Debug, Clone)]
pub struct CharsiuBlock {
    /// `[hidden]` — norm after the attention residual, γ.
    pub attn_norm_gamma: Vec<f32>,
    /// `[hidden]` — norm after the attention residual, β.
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
    /// `[hidden]` — norm after the FFN residual, γ.
    pub ffn_norm_gamma: Vec<f32>,
    /// `[hidden]` — norm after the FFN residual, β.
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

/// Full Charsiu weight store.
#[derive(Debug, Clone)]
pub struct CharsiuWeights {
    /// Raw-waveform 7-layer wav2vec2-base stem.
    pub stem_attrs: WaveformFrontendAttrs,
    /// Per-stem-layer weights (bindable to
    /// [`vokra_ops::waveform_frontend()`]).
    pub stem_weights: WaveformFrontendWeights,
    /// Feature projection (stem's 512-d output → residual `hidden_size`).
    pub feature_projection: CharsiuFeatureProjection,
    /// Grouped positional convolution added to projected features.
    pub pos_conv: CharsiuPosConv,
    /// Encoder input LayerNorm, applied after the positional residual.
    pub encoder_input_norm_gamma: Vec<f32>,
    /// Encoder input LayerNorm beta.
    pub encoder_input_norm_beta: Vec<f32>,
    /// `n_layer` post-norm Transformer encoder blocks.
    pub blocks: Vec<CharsiuBlock>,
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

        let stem_attrs = charsiu_stem_attrs();
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

        let pos_conv = CharsiuPosConv {
            weight: xavier_vec(
                &mut rng,
                config.hidden_size
                    * (config.hidden_size / config.pos_conv_groups)
                    * config.pos_conv_kernel,
                (config.pos_conv_groups as f32
                    / (config.hidden_size * config.pos_conv_kernel) as f32)
                    .sqrt(),
            ),
            bias: vec![0.0; config.hidden_size],
        };
        let encoder_input_norm_gamma = vec![1.0; config.hidden_size];
        let encoder_input_norm_beta = vec![0.0; config.hidden_size];

        // n_layer post-norm Transformer blocks.
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
            pos_conv,
            encoder_input_norm_gamma,
            encoder_input_norm_beta,
            blocks,
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
/// [`Self::align`] runs the canonical frame classifier, silence mask, and
/// monotone DTW to recover per-phoneme time boundaries.
#[derive(Debug, Clone)]
pub struct Charsiu {
    cfg: CharsiuConfig,
    weights: CharsiuWeights,
    vocab: Vec<String>,
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
    pub fn new(cfg: CharsiuConfig, weights: CharsiuWeights, vocab: Vec<String>) -> Result<Self> {
        cfg.validate_for_forward()?;
        Self::validate_shapes(&cfg, &weights)?;
        validate_vocab(&cfg, &vocab)?;
        Ok(Self {
            cfg,
            weights,
            vocab,
        })
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

    /// Model label inventory in id order.
    pub fn vocabulary(&self) -> &[String] {
        &self.vocab
    }

    /// Opens and binds a Charsiu GGUF from disk.
    pub fn from_gguf(path: &Path) -> std::result::Result<Self, LoadError> {
        if !path.exists() {
            return Err(LoadError::FileNotFound(path.to_path_buf()));
        }
        let file = GgufFile::open(path)
            .map_err(|e| LoadError::Gguf(format!("charsiu: opening {}: {e}", path.display())))?;
        Self::from_file(&file).map_err(|e| LoadError::Gguf(e.to_string()))
    }

    /// Binds from an already-parsed GGUF, validating every metadata axis and
    /// tensor shape before returning a usable aligner.
    pub fn from_file(file: &GgufFile) -> Result<Self> {
        verify_arch(file)?;
        require_meta_string(file, "vokra.charsiu.revision", EXPECTED_REVISION)?;
        require_meta_string(
            file,
            "vokra.charsiu.checkpoint_sha256",
            EXPECTED_CHECKPOINT_SHA256,
        )?;
        let cfg = CharsiuConfig {
            hidden_size: meta_u32(file, "vokra.charsiu.hidden_size")? as usize,
            ffn_dim: meta_u32(file, "vokra.charsiu.ffn_dim")? as usize,
            n_layer: meta_u32(file, "vokra.charsiu.n_layer")? as usize,
            n_head: meta_u32(file, "vokra.charsiu.n_head")? as usize,
            vocab_size: meta_u32(file, "vokra.charsiu.vocab_size")? as usize,
            silence_id: meta_u32(file, "vokra.charsiu.silence_id")? as usize,
            pad_id: meta_u32(file, "vokra.charsiu.pad_id")? as usize,
            sample_rate: meta_u32(file, "vokra.charsiu.sample_rate")?,
            frame_shift_sec: meta_f32(file, "vokra.charsiu.frame_shift_sec")?,
            layer_norm_eps: meta_f32(file, "vokra.charsiu.layer_norm_eps")?,
            pos_conv_kernel: meta_u32(file, "vokra.charsiu.pos_conv_kernel")? as usize,
            pos_conv_groups: meta_u32(file, "vokra.charsiu.pos_conv_groups")? as usize,
            silence_threshold: meta_u32(file, "vokra.charsiu.silence_threshold")? as usize,
            feature_projection_has_layer_norm: true,
            stem_conv_bias: false,
        };
        cfg.validate_for_forward()?;
        let canonical = CharsiuConfig::default_charsiu_en();
        if cfg != canonical {
            return Err(VokraError::ModelLoad(format!(
                "charsiu GGUF config {cfg:?} does not match the only implemented canonical \
                 en_w2v2_fc_10ms config {canonical:?}"
            )));
        }
        let vocab = read_string_array(file, "vokra.charsiu.vocab")?;

        let stem_attrs = charsiu_stem_attrs();
        let mut stem_layers = Vec::with_capacity(stem_attrs.layers.len());
        let mut in_ch = 1usize;
        for (i, attrs) in stem_attrs.layers.iter().enumerate() {
            let prefix = format!("wav2vec2.feature_extractor.conv_layers.{i}");
            let conv_w = tensor_shaped(
                file,
                &format!("{prefix}.conv.weight"),
                &[attrs.out_channels, in_ch, attrs.kernel],
            )?;
            let (norm_gamma, norm_beta) = if i == 0 {
                (
                    Some(tensor_shaped(
                        file,
                        &format!("{prefix}.layer_norm.weight"),
                        &[attrs.out_channels],
                    )?),
                    Some(tensor_shaped(
                        file,
                        &format!("{prefix}.layer_norm.bias"),
                        &[attrs.out_channels],
                    )?),
                )
            } else {
                (None, None)
            };
            stem_layers.push(ConvLayerWeights {
                conv_w,
                conv_b: Vec::new(),
                norm_gamma,
                norm_beta,
            });
            in_ch = attrs.out_channels;
        }
        let feature_projection = CharsiuFeatureProjection {
            norm_gamma: Some(tensor_shaped(
                file,
                "wav2vec2.feature_projection.layer_norm.weight",
                &[512],
            )?),
            norm_beta: Some(tensor_shaped(
                file,
                "wav2vec2.feature_projection.layer_norm.bias",
                &[512],
            )?),
            linear_w: tensor_shaped(
                file,
                "wav2vec2.feature_projection.projection.weight",
                &[cfg.hidden_size, 512],
            )?,
            linear_b: tensor_shaped(
                file,
                "wav2vec2.feature_projection.projection.bias",
                &[cfg.hidden_size],
            )?,
        };
        let pos_conv = CharsiuPosConv {
            weight: tensor_shaped(
                file,
                "charsiu.pos_conv.weight",
                &[
                    cfg.hidden_size,
                    cfg.hidden_size / cfg.pos_conv_groups,
                    cfg.pos_conv_kernel,
                ],
            )?,
            bias: tensor_shaped(file, "charsiu.pos_conv.bias", &[cfg.hidden_size])?,
        };
        let encoder_input_norm_gamma = tensor_shaped(
            file,
            "wav2vec2.encoder.layer_norm.weight",
            &[cfg.hidden_size],
        )?;
        let encoder_input_norm_beta =
            tensor_shaped(file, "wav2vec2.encoder.layer_norm.bias", &[cfg.hidden_size])?;
        let mut blocks = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = format!("wav2vec2.encoder.layers.{i}");
            let lin = |name: &str, dims: &[usize]| tensor_shaped(file, name, dims);
            blocks.push(CharsiuBlock {
                attn_norm_gamma: lin(&format!("{p}.layer_norm.weight"), &[cfg.hidden_size])?,
                attn_norm_beta: lin(&format!("{p}.layer_norm.bias"), &[cfg.hidden_size])?,
                q_w: lin(
                    &format!("{p}.attention.q_proj.weight"),
                    &[cfg.hidden_size, cfg.hidden_size],
                )?,
                q_b: lin(&format!("{p}.attention.q_proj.bias"), &[cfg.hidden_size])?,
                k_w: lin(
                    &format!("{p}.attention.k_proj.weight"),
                    &[cfg.hidden_size, cfg.hidden_size],
                )?,
                k_b: lin(&format!("{p}.attention.k_proj.bias"), &[cfg.hidden_size])?,
                v_w: lin(
                    &format!("{p}.attention.v_proj.weight"),
                    &[cfg.hidden_size, cfg.hidden_size],
                )?,
                v_b: lin(&format!("{p}.attention.v_proj.bias"), &[cfg.hidden_size])?,
                o_w: lin(
                    &format!("{p}.attention.out_proj.weight"),
                    &[cfg.hidden_size, cfg.hidden_size],
                )?,
                o_b: lin(&format!("{p}.attention.out_proj.bias"), &[cfg.hidden_size])?,
                ffn_norm_gamma: lin(&format!("{p}.final_layer_norm.weight"), &[cfg.hidden_size])?,
                ffn_norm_beta: lin(&format!("{p}.final_layer_norm.bias"), &[cfg.hidden_size])?,
                fc1_w: lin(
                    &format!("{p}.feed_forward.intermediate_dense.weight"),
                    &[cfg.ffn_dim, cfg.hidden_size],
                )?,
                fc1_b: lin(
                    &format!("{p}.feed_forward.intermediate_dense.bias"),
                    &[cfg.ffn_dim],
                )?,
                fc2_w: lin(
                    &format!("{p}.feed_forward.output_dense.weight"),
                    &[cfg.hidden_size, cfg.ffn_dim],
                )?,
                fc2_b: lin(
                    &format!("{p}.feed_forward.output_dense.bias"),
                    &[cfg.hidden_size],
                )?,
            });
        }
        let weights = CharsiuWeights {
            stem_attrs,
            stem_weights: WaveformFrontendWeights {
                layers: stem_layers,
            },
            feature_projection,
            pos_conv,
            encoder_input_norm_gamma,
            encoder_input_norm_beta,
            blocks,
            head: CharsiuHead {
                weight: tensor_shaped(file, "lm_head.weight", &[cfg.vocab_size, cfg.hidden_size])?,
                bias: tensor_shaped(file, "lm_head.bias", &[cfg.vocab_size])?,
            },
            is_synthesized: false,
        };
        Self::new(cfg, weights, vocab)
    }

    /// Force-aligns a phoneme sequence to a 16 kHz mono PCM buffer.
    ///
    /// Runs the canonical Charsiu silence mask and monotone DTW alignment.
    ///
    /// # Arguments
    ///
    /// * `pcm` — 16 kHz mono PCM at [`CharsiuConfig::sample_rate`]. The
    ///   forward asserts `pcm.len() >= stem.total_stride()` — shorter
    ///   inputs fail loudly in the waveform frontend.
    /// * `sample_rate` — must equal [`CharsiuConfig::sample_rate`];
    ///   mismatch is a loud [`VokraError::InvalidArgument`] (FR-EX-08).
    /// * `phonemes` — the transcript to align (echoed back into the
    ///   `AlignedToken.text` field).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] on empty `pcm` or a sample-rate
    ///   mismatch.
    /// - Propagated `VokraError` from [`vokra_ops::waveform_frontend()`]
    ///   (shape gates on the stem — never silent-truncated).
    pub fn align(
        &self,
        pcm: &[f32],
        sample_rate: u32,
        phonemes: &[String],
    ) -> Result<Vec<AlignedToken>> {
        if phonemes.is_empty() {
            return Err(VokraError::InvalidArgument(
                "charsiu align: phoneme sequence is empty".to_owned(),
            ));
        }
        let mut phone_ids = Vec::with_capacity(phonemes.len());
        for phone in phonemes {
            let id = self.vocab.iter().position(|p| p == phone).ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "charsiu align: phoneme {phone:?} is not in the embedded vocabulary"
                ))
            })?;
            if id == self.cfg.silence_id || id == self.cfg.pad_id {
                return Err(VokraError::InvalidArgument(format!(
                    "charsiu align: transcript phoneme {phone:?} uses reserved model id {id}; \
                     pass spoken phones only (silence is inferred from audio)"
                )));
            }
            phone_ids.push(id);
        }
        let (logits, frames) = self.logits(pcm, sample_rate)?;
        let probabilities = softmax(&logits, frames, self.cfg.vocab_size);
        charsiu_forced_align(
            &probabilities,
            frames,
            self.cfg.vocab_size,
            &phone_ids,
            self.cfg.silence_id,
            self.cfg.silence_threshold,
            self.cfg.frame_shift_sec,
            phonemes,
        )
    }

    /// Returns frame-classification logits in `[time, vocab]` order.
    /// Exposed for independent real-checkpoint parity consumers.
    pub fn logits(&self, pcm: &[f32], sample_rate: u32) -> Result<(Vec<f32>, usize)> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "charsiu logits: pcm slice is empty".to_owned(),
            ));
        }
        if sample_rate != self.cfg.sample_rate {
            return Err(VokraError::InvalidArgument(format!(
                "charsiu logits: sample_rate {} != config.sample_rate {} (FR-EX-08 no silent \
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
        // ---- Feature projection + grouped positional conv ------------
        let mut hidden = feature_projection_forward(
            &features,
            t_frames,
            feature_dim,
            &self.weights.feature_projection,
            self.cfg.hidden_size,
            self.cfg.feature_projection_has_layer_norm,
            self.cfg.layer_norm_eps,
        );
        let position =
            positional_conv_forward(&hidden, t_frames, &self.cfg, &self.weights.pos_conv)?;
        for (value, pos) in hidden.iter_mut().zip(position) {
            *value += pos;
        }
        layer_norm_inplace(
            &mut hidden,
            t_frames,
            self.cfg.hidden_size,
            &self.weights.encoder_input_norm_gamma,
            &self.weights.encoder_input_norm_beta,
            self.cfg.layer_norm_eps,
        );

        // ---- post-norm Transformer blocks -----------------------------
        for block in &self.weights.blocks {
            transformer_block_forward(&mut hidden, t_frames, &self.cfg, block);
        }
        let logits = ctc_head_forward(
            &hidden,
            t_frames,
            self.cfg.hidden_size,
            &self.weights.head,
            self.cfg.vocab_size,
        );
        Ok((logits, t_frames))
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

        let want_pos = h * (h / cfg.pos_conv_groups) * cfg.pos_conv_kernel;
        if w.pos_conv.weight.len() != want_pos || w.pos_conv.bias.len() != h {
            return Err(VokraError::InvalidArgument(format!(
                "charsiu: positional conv weight/bias lengths {} / {} != {want_pos} / {h}",
                w.pos_conv.weight.len(),
                w.pos_conv.bias.len(),
            )));
        }
        if w.encoder_input_norm_gamma.len() != h || w.encoder_input_norm_beta.len() != h {
            return Err(VokraError::InvalidArgument(format!(
                "charsiu: encoder input norm gamma/beta must be length {h}"
            )));
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
pub(crate) fn feature_projection_forward(
    features: &[f32],
    t: usize,
    feature_dim: usize,
    fp: &CharsiuFeatureProjection,
    hidden: usize,
    has_layer_norm: bool,
    eps: f32,
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
        layer_norm_inplace(&mut normed_flat, t, feature_dim, gamma, beta, eps);
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

/// Backend-dispatched sibling of [`feature_projection_forward`].
/// LayerNorm and the learned projection both execute on `compute`; no
/// per-operation CPU fallback is available through this entry point.
#[allow(clippy::too_many_arguments)]
pub(crate) fn feature_projection_forward_with_compute(
    features: &[f32],
    t: usize,
    feature_dim: usize,
    fp: &CharsiuFeatureProjection,
    hidden: usize,
    has_layer_norm: bool,
    eps: f32,
    compute: &Compute,
) -> Result<Vec<f32>> {
    let mut normed = vec![0.0f32; features.len()];
    let input = if has_layer_norm {
        let gamma = fp
            .norm_gamma
            .as_ref()
            .expect("has_layer_norm=true implies norm_gamma present");
        let beta = fp
            .norm_beta
            .as_ref()
            .expect("has_layer_norm=true implies norm_beta present");
        compute.layer_norm_f32(features, &mut normed, t, feature_dim, gamma, beta, eps)?;
        normed.as_slice()
    } else {
        features
    };
    linear_forward_with_compute(
        input,
        t,
        feature_dim,
        &fp.linear_w,
        &fp.linear_b,
        hidden,
        compute,
    )
}

/// Grouped positional Conv1D + SamePad + exact GELU. The input/output are
/// frame-major; the backend convolution is channel-major.
pub(crate) fn positional_conv_forward(
    hidden: &[f32],
    t: usize,
    cfg: &CharsiuConfig,
    pos: &CharsiuPosConv,
) -> Result<Vec<f32>> {
    let h = cfg.hidden_size;
    let mut channel_major = vec![0.0f32; h * t];
    for tt in 0..t {
        for c in 0..h {
            channel_major[c * t + tt] = hidden[tt * h + c];
        }
    }
    let padding = cfg.pos_conv_kernel / 2;
    let raw_len = t + 2 * padding - cfg.pos_conv_kernel + 1;
    let mut raw = vec![0.0f32; h * raw_len];
    kernels::grouped_conv1d_f32(
        &channel_major,
        h,
        t,
        &pos.weight,
        h,
        cfg.pos_conv_kernel,
        Some(&pos.bias),
        1,
        padding,
        cfg.pos_conv_groups,
        &mut raw,
    )?;
    let keep = if cfg.pos_conv_kernel % 2 == 0 {
        raw_len - 1
    } else {
        raw_len
    };
    if keep != t {
        return Err(VokraError::InvalidArgument(format!(
            "charsiu positional conv produced {keep} frame(s) for {t} input frame(s)"
        )));
    }
    let mut out = vec![0.0f32; t * h];
    for c in 0..h {
        for tt in 0..t {
            out[tt * h + c] = gelu_exact(raw[c * raw_len + tt]);
        }
    }
    Ok(out)
}

/// Backend-dispatched grouped positional convolution plus exact GELU.
pub(crate) fn positional_conv_forward_with_compute(
    hidden: &[f32],
    t: usize,
    cfg: &CharsiuConfig,
    pos: &CharsiuPosConv,
    compute: &Compute,
) -> Result<Vec<f32>> {
    let h = cfg.hidden_size;
    let mut channel_major = vec![0.0f32; h * t];
    for tt in 0..t {
        for c in 0..h {
            channel_major[c * t + tt] = hidden[tt * h + c];
        }
    }
    let padding = cfg.pos_conv_kernel / 2;
    let raw_len = t + 2 * padding - cfg.pos_conv_kernel + 1;
    let mut raw = vec![0.0f32; h * raw_len];
    compute.grouped_conv1d_f32(
        &channel_major,
        h,
        t,
        &pos.weight,
        h,
        cfg.pos_conv_kernel,
        Some(&pos.bias),
        1,
        padding,
        cfg.pos_conv_groups,
        &mut raw,
    )?;
    let keep = if cfg.pos_conv_kernel % 2 == 0 {
        raw_len - 1
    } else {
        raw_len
    };
    if keep != t {
        return Err(VokraError::InvalidArgument(format!(
            "charsiu positional conv produced {keep} frame(s) for {t} input frame(s)"
        )));
    }
    let mut activated = vec![0.0f32; raw.len()];
    compute.gelu_f32(&raw, &mut activated)?;
    let mut out = vec![0.0f32; t * h];
    for c in 0..h {
        for tt in 0..t {
            out[tt * h + c] = activated[c * raw_len + tt];
        }
    }
    Ok(out)
}

/// Runs one post-LayerNorm Wav2Vec2 encoder block in place.
///
/// `y = LN1(x + attn(x)); z = LN2(y + ffn(y))`.
fn transformer_block_forward(hidden: &mut [f32], t: usize, cfg: &CharsiuConfig, b: &CharsiuBlock) {
    transformer_block_forward_with_valid_keys(hidden, t, t, cfg, b);
}

/// Wav2Vec2 post-LayerNorm block with an explicit number of valid attention
/// keys. SmartTurn v2 right-pads raw audio to a fixed 16-second window: the
/// padded feature rows remain queries in the upstream encoder but must never
/// be visible as keys. Charsiu passes `valid_keys == t` through the wrapper
/// above, so its established unmasked behaviour is unchanged.
pub(crate) fn transformer_block_forward_with_valid_keys(
    hidden: &mut [f32],
    t: usize,
    valid_keys: usize,
    cfg: &CharsiuConfig,
    b: &CharsiuBlock,
) {
    debug_assert!(valid_keys > 0 && valid_keys <= t);
    let h = cfg.hidden_size;
    let n_head = cfg.n_head;
    let head_dim = h / n_head;
    let head_scale = 1.0 / (head_dim as f32).sqrt();

    // -- Attention branch -----------------------------------------------
    // Q, K, V projections: [T, h] → [T, h] each.
    let q = linear_forward(hidden, t, h, &b.q_w, &b.q_b, h);
    let k = linear_forward(hidden, t, h, &b.k_w, &b.k_b, h);
    let v = linear_forward(hidden, t, h, &b.v_w, &b.v_b, h);

    // MHA: reshape Q/K/V to [n_head, T, head_dim] and score.
    let mut attn_out = vec![0.0_f32; t * h];
    // Work buffer for per-head [T query, valid_keys] scores (reused across
    // heads). Padded SmartTurn rows are queries only; the upstream additive
    // attention mask removes them from the key axis.
    let mut scores = vec![0.0_f32; t * valid_keys];
    for hi in 0..n_head {
        // Compute Q_h K_h^T with the head slices interleaved in the
        // flat buffer as [T, n_head, head_dim].
        for ti in 0..t {
            let qi_base = ti * h + hi * head_dim;
            for tj in 0..valid_keys {
                let kj_base = tj * h + hi * head_dim;
                let mut s = 0.0_f32;
                for d in 0..head_dim {
                    s += q[qi_base + d] * k[kj_base + d];
                }
                scores[ti * valid_keys + tj] = s * head_scale;
            }
        }
        // softmax over each row (causal mask NOT applied — wav2vec2 CTC
        // uses full bidirectional attention).
        for ti in 0..t {
            let row = &mut scores[ti * valid_keys..(ti + 1) * valid_keys];
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
            let ai_row = ti * valid_keys;
            let out_base = ti * h + hi * head_dim;
            for d in 0..head_dim {
                let mut acc = 0.0_f32;
                for tj in 0..valid_keys {
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
    layer_norm_inplace(
        hidden,
        t,
        h,
        &b.attn_norm_gamma,
        &b.attn_norm_beta,
        cfg.layer_norm_eps,
    );

    // -- FFN branch ----------------------------------------------------
    let ffn = cfg.ffn_dim;
    let mut fc1_out = linear_forward(hidden, t, h, &b.fc1_w, &b.fc1_b, ffn);
    for v in fc1_out.iter_mut() {
        *v = gelu_exact(*v);
    }
    let fc2_out = linear_forward(&fc1_out, t, ffn, &b.fc2_w, &b.fc2_b, h);
    for i in 0..(t * h) {
        hidden[i] += fc2_out[i];
    }
    layer_norm_inplace(
        hidden,
        t,
        h,
        &b.ffn_norm_gamma,
        &b.ffn_norm_beta,
        cfg.layer_norm_eps,
    );
}

/// Backend-dispatched Wav2Vec2 post-LayerNorm block with an explicit valid-key
/// prefix. Learned projections, both attention matrix products, softmax,
/// GELU and affine normalisation all execute through one `Compute` instance.
pub(crate) fn transformer_block_forward_with_valid_keys_and_compute(
    hidden: &mut [f32],
    t: usize,
    valid_keys: usize,
    cfg: &CharsiuConfig,
    b: &CharsiuBlock,
    compute: &Compute,
) -> Result<()> {
    debug_assert!(valid_keys > 0 && valid_keys <= t);
    let h = cfg.hidden_size;
    let n_head = cfg.n_head;
    let head_dim = h / n_head;
    let head_scale = 1.0 / (head_dim as f32).sqrt();

    let q = linear_forward_with_compute(hidden, t, h, &b.q_w, &b.q_b, h, compute)?;
    let k = linear_forward_with_compute(hidden, t, h, &b.k_w, &b.k_b, h, compute)?;
    let v = linear_forward_with_compute(hidden, t, h, &b.v_w, &b.v_b, h, compute)?;

    let mut attn_out = vec![0.0f32; t * h];
    let mut q_head = vec![0.0f32; t * head_dim];
    let mut k_head_t = vec![0.0f32; head_dim * valid_keys];
    let mut v_head = vec![0.0f32; valid_keys * head_dim];
    let mut scores = vec![0.0f32; t * valid_keys];
    let mut probabilities = vec![0.0f32; t * valid_keys];
    let mut head_out = vec![0.0f32; t * head_dim];
    for head in 0..n_head {
        for frame in 0..t {
            let src = frame * h + head * head_dim;
            let dst = frame * head_dim;
            q_head[dst..dst + head_dim].copy_from_slice(&q[src..src + head_dim]);
        }
        for frame in 0..valid_keys {
            let src = frame * h + head * head_dim;
            let v_dst = frame * head_dim;
            v_head[v_dst..v_dst + head_dim].copy_from_slice(&v[src..src + head_dim]);
            for dim in 0..head_dim {
                k_head_t[dim * valid_keys + frame] = k[src + dim];
            }
        }
        compute.gemm_f32(
            t,
            valid_keys,
            head_dim,
            &q_head,
            &k_head_t,
            None,
            &mut scores,
        )?;
        for score in &mut scores {
            *score *= head_scale;
        }
        compute.softmax_f32(&scores, &mut probabilities, t, valid_keys)?;
        compute.gemm_f32(
            t,
            head_dim,
            valid_keys,
            &probabilities,
            &v_head,
            None,
            &mut head_out,
        )?;
        for frame in 0..t {
            let src = frame * head_dim;
            let dst = frame * h + head * head_dim;
            attn_out[dst..dst + head_dim].copy_from_slice(&head_out[src..src + head_dim]);
        }
    }

    let projected = linear_forward_with_compute(&attn_out, t, h, &b.o_w, &b.o_b, h, compute)?;
    for (value, residual) in hidden.iter_mut().zip(projected) {
        *value += residual;
    }
    layer_norm_with_compute_inplace(
        hidden,
        t,
        h,
        &b.attn_norm_gamma,
        &b.attn_norm_beta,
        cfg.layer_norm_eps,
        compute,
    )?;

    let mut fc1 =
        linear_forward_with_compute(hidden, t, h, &b.fc1_w, &b.fc1_b, cfg.ffn_dim, compute)?;
    let mut activated = vec![0.0f32; fc1.len()];
    compute.gelu_f32(&fc1, &mut activated)?;
    fc1.clear();
    let fc2 =
        linear_forward_with_compute(&activated, t, cfg.ffn_dim, &b.fc2_w, &b.fc2_b, h, compute)?;
    for (value, residual) in hidden.iter_mut().zip(fc2) {
        *value += residual;
    }
    layer_norm_with_compute_inplace(
        hidden,
        t,
        h,
        &b.ffn_norm_gamma,
        &b.ffn_norm_beta,
        cfg.layer_norm_eps,
        compute,
    )
}

/// `y = x @ W^T + b` where `x` is `[t, in_dim]`, `W` is `[out_dim,
/// in_dim]`, `b` is `[out_dim]`.
pub(crate) fn linear_forward(
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

/// Backend-dispatched `y = x @ W^T + b` for output-major linear weights.
pub(crate) fn linear_forward_with_compute(
    x: &[f32],
    t: usize,
    in_dim: usize,
    w: &[f32],
    b: &[f32],
    out_dim: usize,
    compute: &Compute,
) -> Result<Vec<f32>> {
    let mut w_t = vec![0.0f32; w.len()];
    for out in 0..out_dim {
        for input in 0..in_dim {
            w_t[input * out_dim + out] = w[out * in_dim + input];
        }
    }
    let mut output = vec![0.0f32; t * out_dim];
    compute.gemm_f32(t, out_dim, in_dim, x, &w_t, Some(b), &mut output)?;
    Ok(output)
}

/// LayerNorm over the last axis in place. `x` is `[t, dim]` row-major.
pub(crate) fn layer_norm_inplace(
    x: &mut [f32],
    t: usize,
    dim: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) {
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
        let inv_std = 1.0 / (var + eps).sqrt();
        for (i, v) in row.iter_mut().enumerate() {
            *v = ((*v - mean) * inv_std) * gamma[i] + beta[i];
        }
    }
}

/// Backend-dispatched affine LayerNorm, copied back in place only after the
/// selected backend completes successfully.
#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_norm_with_compute_inplace(
    x: &mut [f32],
    t: usize,
    dim: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    compute: &Compute,
) -> Result<()> {
    let mut output = vec![0.0f32; x.len()];
    compute.layer_norm_f32(x, &mut output, t, dim, gamma, beta, eps)?;
    x.copy_from_slice(&output);
    Ok(())
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

/// Per-frame softmax over the vocab axis.
fn softmax(logits: &[f32], t: usize, vocab: usize) -> Vec<f32> {
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
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn charsiu_forced_align(
    probabilities: &[f32],
    frames: usize,
    vocab: usize,
    phone_ids: &[usize],
    silence_id: usize,
    silence_threshold: usize,
    frame_shift_sec: f32,
    phones: &[String],
) -> Result<Vec<AlignedToken>> {
    let mut predicted = Vec::with_capacity(frames);
    for frame in probabilities.chunks_exact(vocab) {
        let mut best = 0usize;
        for i in 1..vocab {
            if frame[i] > frame[best] {
                best = i;
            }
        }
        predicted.push(best);
    }

    // Upstream `_get_sil_mask`: a silence argmax run shorter than the
    // threshold is treated as speech; longer runs are removed before DTW.
    let mut is_silence = vec![false; frames];
    let mut start = 0usize;
    while start < frames {
        let value = predicted[start];
        let mut end = start + 1;
        while end < frames && predicted[end] == value {
            end += 1;
        }
        if value == silence_id && end - start >= silence_threshold {
            is_silence[start..end].fill(true);
        }
        start = end;
    }
    let nonsil: Vec<usize> = (0..frames).filter(|&i| !is_silence[i]).collect();
    if nonsil.is_empty() {
        return Err(VokraError::InvalidArgument(
            "charsiu align: no speech frames remain after the canonical silence mask".to_owned(),
        ));
    }
    if nonsil.len() < phone_ids.len() {
        return Err(VokraError::InvalidArgument(format!(
            "charsiu align: {} non-silence frame(s) cannot align {} phoneme(s)",
            nonsil.len(),
            phone_ids.len()
        )));
    }

    // `librosa.sequence.dtw(C=-cost[:, phone_ids], step_sizes_sigma=
    // [[1,1],[1,0]])`: maximize the summed phone posterior while consuming
    // every non-silence frame and advancing by at most one target phone.
    let n = nonsil.len();
    let m = phone_ids.len();
    let mut prev = vec![f32::NEG_INFINITY; m];
    let mut back = vec![false; n * m]; // true = predecessor was diagonal
    prev[0] = probabilities[nonsil[0] * vocab + phone_ids[0]];
    for i in 1..n {
        let mut cur = vec![f32::NEG_INFINITY; m];
        let max_j = i.min(m - 1);
        for j in 0..=max_j {
            let stay = prev[j];
            let diagonal = if j > 0 {
                prev[j - 1]
            } else {
                f32::NEG_INFINITY
            };
            if stay.is_finite() || diagonal.is_finite() {
                let use_diagonal = diagonal >= stay;
                let predecessor = if use_diagonal { diagonal } else { stay };
                cur[j] = predecessor + probabilities[nonsil[i] * vocab + phone_ids[j]];
                back[i * m + j] = use_diagonal;
            }
        }
        prev = cur;
    }
    if !prev[m - 1].is_finite() {
        return Err(VokraError::InvalidArgument(
            "charsiu align: monotone DTW found no complete path".to_owned(),
        ));
    }
    let mut assignments = vec![0usize; n];
    let mut j = m - 1;
    for i in (1..n).rev() {
        assignments[i] = j;
        if back[i * m + j] {
            j -= 1;
        }
    }
    assignments[0] = 0;
    if j != 0 {
        return Err(VokraError::InvalidArgument(
            "charsiu align: DTW backtrack did not reach the first phone".to_owned(),
        ));
    }

    let mut out = Vec::with_capacity(m);
    for target in 0..m {
        let first = assignments
            .iter()
            .position(|&x| x == target)
            .expect("complete DTW path visits every target");
        let last = assignments
            .iter()
            .rposition(|&x| x == target)
            .expect("complete DTW path visits every target");
        let mut confidence = 0.0f32;
        let mut count = 0usize;
        for i in first..=last {
            if assignments[i] == target {
                confidence += probabilities[nonsil[i] * vocab + phone_ids[target]];
                count += 1;
            }
        }
        out.push(AlignedToken {
            text: phones[target].clone(),
            start_sec: nonsil[first] as f32 * frame_shift_sec,
            end_sec: (nonsil[last] + 1) as f32 * frame_shift_sec,
            confidence: confidence / count as f32,
        });
    }
    Ok(out)
}

/// Exact erf-based GELU (`0.5 * x * (1 + erf(x / √2))`) — HF wav2vec2
/// uses `ACT2FN["gelu"]` = exact GELU. Same A&S 7.1.26 erf approximation
/// as [`vokra_ops::waveform_frontend()`]; kept private here so the align
/// module has no cross-crate constraints on the ops erf.
#[inline]
pub(crate) fn gelu_exact(x: f32) -> f32 {
    0.5 * x * (1.0 + erf_as(x * core::f32::consts::FRAC_1_SQRT_2))
}

/// Rejects a GGUF whose model architecture is absent or incompatible.
fn verify_arch(file: &GgufFile) -> Result<()> {
    match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
        Some(a) if a == EXPECTED_ARCH => Ok(()),
        Some(other) => Err(VokraError::ModelLoad(format!(
            "charsiu: GGUF arch is `{other}`, expected `{EXPECTED_ARCH}`. Sibling \
             wav2vec2-lineage arch tags share this exact topology but have incompatible \
             output heads — `wav2vec2_ctc` (Meta wav2vec2 + CTC ASR head, character / \
             letter vocabulary), `hubert` (HuBERT SSL encoder, no fixed downstream head), \
             `emotion2vec` (fixed 9-way emotion classifier), `wavlm_sv` (XVector speaker \
             verification head). Charsiu's head is an IPA *phoneme* inventory used for \
             forced alignment; binding a letter-vocab or classifier checkpoint here would \
             emit confident, meaningless phoneme boundaries (FR-EX-08 — no silent partial \
             load)."
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "charsiu: GGUF is missing `{}` — this is not a Vokra-native Charsiu GGUF.",
            chunks::KEY_MODEL_ARCH,
        ))),
    }
}

fn require_meta_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file
        .get(key)
        .and_then(GgufMetadataValue::as_str)
        .ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "charsiu GGUF missing required string metadata `{key}`"
            ))
        })?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "charsiu GGUF `{key}` is {actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn meta_u32(file: &GgufFile, key: &str) -> Result<u32> {
    let value = file
        .get(key)
        .and_then(GgufMetadataValue::as_u64)
        .ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "charsiu GGUF missing required u32 metadata `{key}`"
            ))
        })?;
    u32::try_from(value).map_err(|_| {
        VokraError::ModelLoad(format!(
            "charsiu GGUF metadata `{key}` = {value} overflows u32"
        ))
    })
}

fn meta_f32(file: &GgufFile, key: &str) -> Result<f32> {
    match file.get(key) {
        Some(GgufMetadataValue::F32(value)) => Ok(*value),
        _ => Err(VokraError::ModelLoad(format!(
            "charsiu GGUF missing required f32 metadata `{key}`"
        ))),
    }
}

fn read_string_array(file: &GgufFile, key: &str) -> Result<Vec<String>> {
    let array = file
        .get(key)
        .and_then(GgufMetadataValue::as_array)
        .ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "charsiu GGUF missing required Array<String> metadata `{key}`"
            ))
        })?;
    if array.element_type != GgufValueType::String {
        return Err(VokraError::ModelLoad(format!(
            "charsiu GGUF `{key}` has element type {:?}, expected String",
            array.element_type
        )));
    }
    array
        .values
        .iter()
        .enumerate()
        .map(|(i, value)| match value {
            GgufMetadataValue::String(value) => Ok(value.clone()),
            _ => Err(VokraError::ModelLoad(format!(
                "charsiu GGUF `{key}[{i}]` is not a string"
            ))),
        })
        .collect()
}

fn validate_vocab(cfg: &CharsiuConfig, vocab: &[String]) -> Result<()> {
    if vocab.len() != cfg.vocab_size {
        return Err(VokraError::InvalidArgument(format!(
            "charsiu: vocabulary length {} != vocab_size {}",
            vocab.len(),
            cfg.vocab_size
        )));
    }
    if vocab[cfg.silence_id] != "[SIL]" || vocab[cfg.pad_id] != "[PAD]" {
        return Err(VokraError::InvalidArgument(format!(
            "charsiu: vocabulary special ids do not match: id {} = {:?}, id {} = {:?}",
            cfg.silence_id, vocab[cfg.silence_id], cfg.pad_id, vocab[cfg.pad_id]
        )));
    }
    let mut unique = std::collections::BTreeSet::new();
    for phone in vocab {
        if !unique.insert(phone) {
            return Err(VokraError::InvalidArgument(format!(
                "charsiu: duplicate vocabulary entry {phone:?}"
            )));
        }
    }
    Ok(())
}

fn tensor_shaped(file: &GgufFile, name: &str, dims: &[usize]) -> Result<Vec<f32>> {
    let info = file
        .tensor_info(name)
        .ok_or_else(|| VokraError::ModelLoad(format!("charsiu GGUF missing tensor `{name}`")))?;
    let expected: Vec<u64> = dims.iter().map(|&d| d as u64).collect();
    if info.dimensions != expected {
        return Err(VokraError::ModelLoad(format!(
            "charsiu GGUF tensor `{name}` has dims {:?}, expected {expected:?}",
            info.dimensions
        )));
    }
    file.tensor_f32(name)
        .map_err(|e| VokraError::ModelLoad(format!("charsiu GGUF reading `{name}`: {e}")))
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
            silence_id: 0,
            pad_id: 7,
            sample_rate: 16_000,
            frame_shift_sec: 0.01,
            layer_norm_eps: 1e-5,
            pos_conv_kernel: 4,
            pos_conv_groups: 4,
            silence_threshold: 4,
            feature_projection_has_layer_norm: true,
            stem_conv_bias: false,
        }
    }

    fn tiny_vocab() -> Vec<String> {
        ["[SIL]", "P", "AE", "T", "K", "S", "[UNK]", "[PAD]"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect()
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

    /// Writes `bytes` to a unique scratch path and returns it.
    fn scratch_gguf(tag: &str, bytes: Vec<u8>) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-charsiu-arch-{}-{}-{}.gguf",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default(),
        ));
        std::fs::write(&p, bytes).expect("write scratch gguf");
        p
    }

    #[test]
    fn from_gguf_rejects_gguf_without_arch_stamp() {
        // FR-EX-08: an unstamped GGUF is not a Vokra-native Charsiu
        // artifact. The refusal must name the missing key rather than
        // sending the caller to the (unrelated) follow-up-wave note.
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_u32("vokra.charsiu.hidden_size", 768);
        let path = scratch_gguf("no-arch", b.to_bytes().expect("serialize"));

        let err = Charsiu::from_gguf(&path).expect_err("missing arch must fail");
        match &err {
            LoadError::Gguf(msg) => {
                assert!(
                    msg.contains(chunks::KEY_MODEL_ARCH),
                    "message must name the missing key: {msg}",
                );
                assert!(
                    msg.contains("charsiu"),
                    "message must name the expected model: {msg}",
                );
            }
            other => panic!("expected LoadError::Gguf, got {other:?}"),
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn from_gguf_rejects_foreign_arch_naming_expected_and_actual() {
        // `wav2vec2_ctc` is the most dangerous mis-route: identical
        // topology, identical tensor names, incompatible output vocabulary
        // (letters vs IPA phonemes). It must be named alongside the
        // expectation so the caller sees exactly what they passed.
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "wav2vec2_ctc");
        let path = scratch_gguf("foreign-arch", b.to_bytes().expect("serialize"));

        let err = Charsiu::from_gguf(&path).expect_err("foreign arch must fail");
        match &err {
            LoadError::Gguf(msg) => {
                assert!(
                    msg.contains("wav2vec2_ctc"),
                    "message must name the actual arch: {msg}",
                );
                assert!(
                    msg.contains(EXPECTED_ARCH),
                    "message must name the expected arch: {msg}",
                );
            }
            other => panic!("expected LoadError::Gguf, got {other:?}"),
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn from_gguf_with_correct_arch_requires_pinned_revision() {
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, EXPECTED_ARCH);
        let path = scratch_gguf("real-arch", b.to_bytes().expect("serialize"));

        let err = Charsiu::from_gguf(&path).expect_err("metadata-only GGUF must fail");
        match &err {
            LoadError::Gguf(msg) => assert!(
                msg.contains("vokra.charsiu.revision"),
                "binder must name the first missing writer-contract key: {msg}",
            ),
            other => panic!("expected LoadError::Gguf, got {other:?}"),
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn default_config_matches_canonical_charsiu_axes() {
        let c = CharsiuConfig::default_charsiu_en();
        assert_eq!(c.hidden_size, 768);
        assert_eq!(c.n_layer, 12);
        assert_eq!(c.n_head, 12);
        assert_eq!(c.ffn_dim, 3072);
        assert_eq!(c.head_dim(), 64);
        assert_eq!(c.sample_rate, 16_000);
        assert!((c.frame_shift_sec - 0.01).abs() < 1e-9);
        assert_eq!(c.silence_id, 0);
        assert_eq!(c.pad_id, 41);
        assert_eq!(charsiu_stem_attrs().total_stride().unwrap(), 160);
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
        let err = Charsiu::new(cfg, w, tiny_vocab()).expect_err("shape mismatch must be caught");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn synthesize_reports_scaffold_flag() {
        let cfg = tiny_for_tests();
        let w = CharsiuWeights::synthesized(&cfg, 0xDEAD_BEEF).unwrap();
        let aligner = Charsiu::new(cfg, w, tiny_vocab()).unwrap();
        assert!(aligner.is_synthesized());
    }

    #[test]
    fn align_rejects_empty_pcm() {
        let cfg = tiny_for_tests();
        let w = CharsiuWeights::synthesized(&cfg, 1).unwrap();
        let aligner = Charsiu::new(cfg, w, tiny_vocab()).unwrap();
        let err = aligner
            .align(&[], 16_000, &["P".to_owned()])
            .expect_err("empty pcm must fail loudly");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn align_rejects_sample_rate_mismatch() {
        let cfg = tiny_for_tests();
        let w = CharsiuWeights::synthesized(&cfg, 1).unwrap();
        let aligner = Charsiu::new(cfg, w, tiny_vocab()).unwrap();
        let pcm = vec![0.0_f32; 4000];
        let err = aligner
            .align(&pcm, 8_000, &["P".to_owned()])
            .expect_err("mismatched sample_rate must fail loudly");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn logits_runs_canonical_stage_order_end_to_end() {
        let cfg = tiny_for_tests();
        let w = CharsiuWeights::synthesized(&cfg, 0x00C0_FFEE).unwrap();
        let aligner = Charsiu::new(cfg, w, tiny_vocab()).unwrap();
        let mut pcm = vec![0.0_f32; 1_600];
        for (i, s) in pcm.iter_mut().enumerate() {
            *s = ((i as f32 / 200.0).sin()) * 0.1;
        }
        let (logits, frames) = aligner.logits(&pcm, 16_000).unwrap();
        assert!(frames >= 3);
        assert_eq!(logits.len(), frames * 8);
        assert!(logits.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn forced_alignment_masks_long_silence_and_returns_phone_intervals() {
        // Six frames, vocabulary [SIL, P, AE, T]. Frames 2..=5 would be a
        // long silence only if all four were SIL; here frames 2..=3 are a
        // short run and therefore remain in the DTW input.
        let probabilities = [
            0.05, 0.90, 0.03, 0.02, // P
            0.05, 0.80, 0.10, 0.05, // P
            0.80, 0.05, 0.10, 0.05, // short SIL
            0.80, 0.05, 0.10, 0.05, // short SIL
            0.05, 0.05, 0.80, 0.10, // AE
            0.05, 0.05, 0.10, 0.80, // T
        ];
        let phones = vec!["P".to_owned(), "AE".to_owned(), "T".to_owned()];
        let out =
            charsiu_forced_align(&probabilities, 6, 4, &[1, 2, 3], 0, 4, 0.01, &phones).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].text, "P");
        assert!(out.windows(2).all(|w| w[0].end_sec <= w[1].start_sec));
    }

    /// PCM too short for the stem fails loudly.
    #[test]
    fn align_returns_empty_on_pcm_too_short_for_stem() {
        let cfg = tiny_for_tests();
        let w = CharsiuWeights::synthesized(&cfg, 42).unwrap();
        let aligner = Charsiu::new(cfg, w, tiny_vocab()).unwrap();
        // Stem's total_stride = 160. A pcm with just 100 samples will
        // fail the shape gate inside `waveform_frontend` (loud), which
        // we bubble up. That is the correct posture (FR-EX-08 no silent
        // fabrication) — this test pins that behaviour so a future
        // refactor cannot silently return an empty alignment on
        // shape-too-short.
        let short = vec![0.0_f32; 100];
        let err = aligner
            .align(&short, 16_000, &["P".to_owned()])
            .expect_err("input too short for stem must fail loudly");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }
}
