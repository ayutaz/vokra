//! Zipformer encoder (SoTA plan Phase JA JA-ASR-5 primitive).
//!
//! Direct Rust port of the k2-fsa/icefall Zipformer implementation
//! (`icefall/egs/librispeech/ASR/zipformer/zipformer.py`, Apache-2.0). The
//! primary consumer in Vokra's SoTA plan is the reazonspeech-k2 CTC family
//! (`reazon-research/reazonspeech-k2-v2`, Apache-2.0), a modern JA-first
//! Zipformer-CTC ASR built on top of icefall.
//!
//! # What Zipformer is (vs. Conformer / FastConformer)
//!
//! [`ConformerEncoder`](crate::conformer::ConformerEncoder) runs a single
//! resolution stack: subsample → N identical Conformer layers → out. Zipformer
//! generalises that to a **multi-resolution pyramid**:
//!
//! 1. an encoder embed that produces a low-rate hidden stream at
//!    `d_model = downsampling_factor(0) · d_model_stage(0)` (per stage);
//! 2. a series of [`ZipformerStack`]s where each stack is
//!    `DownSampling → k Conformer-family layers → UpSampling → bypass`. The
//!    downsampling drops the sample rate before the layers, the upsampling
//!    restores it, and the bypass fuses the pre-stack signal into the
//!    up-sampled output via a learnable scalar (upstream
//!    `SimpleUpsample.bypass_scale`, initialised near 1.0);
//! 3. **attention weight sharing** — within a stack, every layer's
//!    multi-head-attention `Q·K^T / sqrt(d)` matrix is computed **once** for
//!    the first layer and reused by every subsequent layer at that scale
//!    (upstream `AttentionSqueeze` in `zipformer.py`); the value projection
//!    and output projection are per-layer. This is the key compute win over
//!    stacking N independent Conformer layers.
//!
//! # Scope of this primitive
//!
//! The upstream Zipformer has ~20 more attributes than we surface (four
//! separate stack widths, per-stack head counts, per-stack ff-expansion
//! factors, SkipConnect scales, bypass warm-up schedules, feedforward
//! probes). This primitive collapses those to the axes a caller actually
//! changes for the reazonspeech-k2 CTC family:
//!
//! - a **single** hidden dim `d_model` throughout the encoder (upstream
//!   allows per-stack widths; the released CTC checkpoint uses a single
//!   width);
//! - a per-stack **downsampling factor** (matches `downsampling_factor` in
//!   `zipformer.py`);
//! - a per-stack **layer count** `k`;
//! - a per-stack **kernel size** for the depthwise conv;
//! - a single **head count**, **ff-expansion**, and **RoPE overlay** shared
//!   across stacks.
//!
//! The rest of the axes are pinned to their released-checkpoint defaults;
//! adding a per-stack override is a mechanical follow-up.
//!
//! # Attention weight sharing (why it's not free)
//!
//! In pure Conformer, each layer holds independent `Q`, `K`, `V`, `Wo`. In
//! Zipformer, within a stack only `V` and `Wo` (and the FF branches) are
//! per-layer; `Q` and `K` are **shared** — one attention-score matrix
//! `A = softmax((Q_shared·K_shared^T) / sqrt(d) + rel_pos)` is computed once
//! per stack and applied to each layer's own `V`. This module keeps that
//! wiring explicit: [`ZipformerLayerWeights`] carries only `wv` / `wo` +
//! FF + Conv + LayerNorm gains; the *stack* owns [`SharedMhaQkWeights`].
//!
//! # No silent CPU fallback (FR-EX-08)
//!
//! Every degenerate input — mismatched hidden lengths, empty layer stack,
//! downsampling that would drop every frame, `d_model % n_heads != 0`,
//! even kernel size — becomes a loud [`VokraError::InvalidArgument`].
//! Silent truncation to zero-frame output is banned (matches
//! [`crate::conformer::ConformerEncoder`] posture).
//!
//! # Zero-dependency posture (NFR-DS-02)
//!
//! No BLAS, no `serde`, no third-party crate. Scalar Rust with `unsafe`
//! deliberately absent (SIMD kernels can land later without changing the
//! public surface).
//!
//! # Upstream primary sources
//!
//! - [`k2-fsa/icefall/egs/librispeech/ASR/zipformer/zipformer.py`][icefall-zipformer]
//!   — the reference implementation used across every icefall recipe. Class
//!   `Zipformer2` (~L52) owns the outer pyramid; `Zipformer2EncoderLayer`
//!   owns the per-layer wiring; `AttentionSqueeze` (~L1400) is the shared-QK
//!   attention path.
//! - [`k2-fsa/icefall/egs/librispeech/ASR/zipformer/subsampling.py`][icefall-subsample]
//!   — `SimpleDownsample` / `SimpleUpsample`. The bypass fuse is
//!   `output = bypass_scale · pre + (1 - bypass_scale) · upsampled_post`.
//! - [`reazon-research/reazonspeech`][reazon-repo] — Apache-2.0
//!   Zipformer-CTC recipe on ReazonSpeech (JA-first); the released
//!   checkpoints (`reazonspeech-k2-v2`) are the concrete tensors this
//!   primitive is dimensioned to accept.
//!
//! [icefall-zipformer]: https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/zipformer.py
//! [icefall-subsample]: https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/subsampling.py
//! [reazon-repo]: https://github.com/reazon-research/reazonspeech

use vokra_core::{Result, VokraError};

use crate::conformer::{
    ConformerConvWeights, ConvSubsampleKind, FeedForwardWeights, PositionEncoding,
};

// ---------------------------------------------------------------------------
// Public config
// ---------------------------------------------------------------------------

/// Per-stack description of the Zipformer pyramid: how much we downsample,
/// how many layers run at that rate, and their conv kernel size.
///
/// The **layer count** covers the per-layer FF + Conv + LN stack;
/// **attention** across those layers reuses the stack-level shared-QK
/// projection (see [`SharedMhaQkWeights`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZipformerStackDesc {
    /// Downsampling factor applied to the time axis on entry to the stack.
    /// `1` means "no rate change" (an identity down/up-sample pair, useful
    /// for the top-most stack). `factor > 1` collapses `factor` adjacent
    /// frames by averaging (upstream `SimpleDownsample.forward` mean over a
    /// blocked window). Must be `> 0`.
    pub downsampling_factor: u32,
    /// Number of Conformer-family layers stacked at this resolution.
    /// Attention across these layers reuses the stack's `SharedMhaQkWeights`
    /// so the QK score matrix is computed exactly once per stack per
    /// forward pass.
    pub n_layers: u32,
    /// Depthwise convolution kernel size for the layers in this stack. Must
    /// be odd for symmetric same-padding.
    pub kernel_size: u32,
}

/// Encoder hyperparameters.
///
/// The fields are the axes we actually change for the reazonspeech-k2
/// checkpoint. Per-stack width / head-count overrides are deliberately
/// absent; adding them is a mechanical follow-up that keeps the layout
/// contract intact.
#[derive(Debug, Clone)]
pub struct ZipformerConfig {
    /// Feature dim on the mel input (upstream `feat_in`, e.g. 80 log-mel
    /// bins after the SpecAugment / global-mean-var-norm front-end).
    pub in_dim: u32,
    /// Hidden dim used throughout the encoder (same width in every stack).
    pub d_model: u32,
    /// Number of attention heads. `d_model % n_heads == 0` is checked at
    /// construction time.
    pub n_heads: u32,
    /// FeedForward hidden width (upstream `d_ff = d_model *
    /// ff_expansion_factor`; caller supplies the final value).
    pub ffn_dim: u32,
    /// Stem subsampling variant (upstream Zipformer feeds the encoder embed
    /// via a Conv-based subsample; we accept the same
    /// [`ConvSubsampleKind`] enum the Conformer primitive uses so callers
    /// can share the stem wiring).
    pub stem: ConvSubsampleKind,
    /// Positional encoding overlay for the shared-QK path (upstream
    /// Zipformer uses RoPE by default; we accept the same enum
    /// [`crate::conformer::PositionEncoding`] to keep the surface small).
    pub position_encoding: PositionEncoding,
    /// The per-stack pyramid. Must be non-empty. Each entry describes one
    /// down/up-sample slab with `n_layers` Conformer-family layers inside.
    pub stacks: Vec<ZipformerStackDesc>,
}

impl ZipformerConfig {
    /// Per-head attention dimension (`d_model / n_heads`).
    pub fn head_dim(&self) -> usize {
        (self.d_model / self.n_heads) as usize
    }
}

// ---------------------------------------------------------------------------
// Weight structs
// ---------------------------------------------------------------------------

/// Stem-subsample weights — same layout as
/// [`crate::conformer::ConformerSubsampleWeights`]. Kept as a distinct type
/// so a future divergence in the stem does not force a Zipformer-side
/// rewrite of every caller.
#[derive(Debug, Clone)]
pub struct ZipformerStemWeights {
    /// Row-major `[d_model, projection_in_dim]` linear weight —
    /// `projection_in_dim = in_dim` for [`ConvSubsampleKind::Linear`] and
    /// `factor * in_dim` for the stacking variants.
    pub linear_w: Vec<f32>,
    /// `[d_model]` linear bias.
    pub linear_b: Vec<f32>,
    /// `[d_model]` LayerNorm gain — required iff `stem.has_norm()`.
    pub norm_gamma: Option<Vec<f32>>,
    /// `[d_model]` LayerNorm bias — required iff `stem.has_norm()`.
    pub norm_beta: Option<Vec<f32>>,
}

impl ZipformerStemWeights {
    fn validate(&self, cfg: &ZipformerConfig) -> Result<()> {
        let d_model = cfg.d_model as usize;
        let in_dim = cfg.in_dim as usize;
        let proj_in = cfg.stem.projection_in_dim(in_dim);
        let expected_w = d_model * proj_in;
        if self.linear_w.len() != expected_w {
            return Err(VokraError::InvalidArgument(format!(
                "Zipformer stem linear_w must be length {expected_w} \
                 (d_model={d_model} × projection_in_dim={proj_in}), got {}",
                self.linear_w.len(),
            )));
        }
        if self.linear_b.len() != d_model {
            return Err(VokraError::InvalidArgument(format!(
                "Zipformer stem linear_b must be length {d_model}, got {}",
                self.linear_b.len(),
            )));
        }
        let need_norm = cfg.stem.has_norm();
        match (&self.norm_gamma, &self.norm_beta, need_norm) {
            (Some(g), Some(b), true) => {
                if g.len() != d_model || b.len() != d_model {
                    return Err(VokraError::InvalidArgument(format!(
                        "Zipformer stem norm gamma/beta must be length {d_model}, \
                         got gamma={} beta={}",
                        g.len(),
                        b.len(),
                    )));
                }
            }
            (None, None, false) => {}
            _ => {
                return Err(VokraError::InvalidArgument(
                    "Zipformer stem norm gamma/beta presence must match stem.has_norm()".to_owned(),
                ));
            }
        }
        Ok(())
    }
}

/// Stack-level attention QK weights, **shared across every layer in the
/// stack** (upstream `AttentionSqueeze`).
///
/// Only `Q` and `K` are shared — `V` and the output projection are
/// per-layer (see [`ZipformerLayerWeights::wv`] / `wo`). This mirrors the
/// upstream `SharedQK` split.
#[derive(Debug, Clone)]
pub struct SharedMhaQkWeights {
    /// Row-major `[d_model, d_model]`.
    pub wq: Vec<f32>,
    /// `[d_model]`.
    pub bq: Vec<f32>,
    /// Row-major `[d_model, d_model]`.
    pub wk: Vec<f32>,
    /// `[d_model]`.
    pub bk: Vec<f32>,
}

impl SharedMhaQkWeights {
    fn validate(&self, d_model: usize, tag: &str) -> Result<()> {
        let dd = d_model * d_model;
        if self.wq.len() != dd || self.wk.len() != dd {
            return Err(VokraError::InvalidArgument(format!(
                "{tag}: shared wq/wk must be length {dd} (d_model²), got wq={} wk={}",
                self.wq.len(),
                self.wk.len(),
            )));
        }
        if self.bq.len() != d_model || self.bk.len() != d_model {
            return Err(VokraError::InvalidArgument(format!(
                "{tag}: shared bq/bk must be length {d_model}, got bq={} bk={}",
                self.bq.len(),
                self.bk.len(),
            )));
        }
        Ok(())
    }
}

/// Per-layer weights. `V` + output projection + LayerNorm gains + FF + Conv;
/// `Q` and `K` are supplied by the enclosing stack (see
/// [`SharedMhaQkWeights`]).
#[derive(Debug, Clone)]
pub struct ZipformerLayerWeights {
    /// `[d_model]` γ for the FF1 pre-norm.
    pub ln1_gamma: Vec<f32>,
    /// `[d_model]` β for the FF1 pre-norm.
    pub ln1_beta: Vec<f32>,
    /// FF1 weights.
    pub ff1: FeedForwardWeights,
    /// `[d_model]` γ for the MHA pre-norm.
    pub ln2_gamma: Vec<f32>,
    /// `[d_model]` β for the MHA pre-norm.
    pub ln2_beta: Vec<f32>,
    /// Row-major `[d_model, d_model]` — per-layer value projection.
    pub wv: Vec<f32>,
    /// `[d_model]` per-layer value bias.
    pub bv: Vec<f32>,
    /// Row-major `[d_model, d_model]` — per-layer output projection.
    pub wo: Vec<f32>,
    /// `[d_model]` per-layer output bias.
    pub bo: Vec<f32>,
    /// `[d_model]` γ for the Conv pre-norm.
    pub ln3_gamma: Vec<f32>,
    /// `[d_model]` β for the Conv pre-norm.
    pub ln3_beta: Vec<f32>,
    /// Conv-module weights (reuses the Conformer primitive's layout — same
    /// pointwise-GLU-depthwise-LN-Swish-pointwise chain).
    pub conv: ConformerConvWeights,
    /// `[d_model]` γ for the FF2 pre-norm.
    pub ln4_gamma: Vec<f32>,
    /// `[d_model]` β for the FF2 pre-norm.
    pub ln4_beta: Vec<f32>,
    /// FF2 weights.
    pub ff2: FeedForwardWeights,
    /// `[d_model]` γ for the final per-layer norm.
    pub ln_out_gamma: Vec<f32>,
    /// `[d_model]` β for the final per-layer norm.
    pub ln_out_beta: Vec<f32>,
}

impl ZipformerLayerWeights {
    fn validate(&self, d_model: usize, ffn_dim: usize, kernel: usize, tag: &str) -> Result<()> {
        for (name, v) in [
            ("ln1_gamma", &self.ln1_gamma),
            ("ln1_beta", &self.ln1_beta),
            ("ln2_gamma", &self.ln2_gamma),
            ("ln2_beta", &self.ln2_beta),
            ("ln3_gamma", &self.ln3_gamma),
            ("ln3_beta", &self.ln3_beta),
            ("ln4_gamma", &self.ln4_gamma),
            ("ln4_beta", &self.ln4_beta),
            ("ln_out_gamma", &self.ln_out_gamma),
            ("ln_out_beta", &self.ln_out_beta),
            ("bv", &self.bv),
            ("bo", &self.bo),
        ] {
            if v.len() != d_model {
                return Err(VokraError::InvalidArgument(format!(
                    "{tag}: {name} must be length {d_model}, got {}",
                    v.len(),
                )));
            }
        }
        let dd = d_model * d_model;
        for (name, w) in [("wv", &self.wv), ("wo", &self.wo)] {
            if w.len() != dd {
                return Err(VokraError::InvalidArgument(format!(
                    "{tag}: {name} must be length {dd} (d_model²), got {}",
                    w.len(),
                )));
            }
        }
        self.ff1
            .validate_ext(d_model, ffn_dim, &format!("{tag} FF1"))?;
        self.ff2
            .validate_ext(d_model, ffn_dim, &format!("{tag} FF2"))?;
        self.conv
            .validate_ext(d_model, kernel, &format!("{tag} Conv"))?;
        Ok(())
    }
}

/// Weights for one stack: the shared-QK attention pair + per-layer bodies
/// + the up-sample bypass scalar.
#[derive(Debug, Clone)]
pub struct ZipformerStackWeights {
    /// Shared attention QK across every layer of the stack.
    pub shared_qk: SharedMhaQkWeights,
    /// Per-layer bodies (`length == desc.n_layers`).
    pub layers: Vec<ZipformerLayerWeights>,
    /// Learnable scalar the up-sample step uses to fuse the pre-stack
    /// signal into the up-sampled post-stack signal:
    /// `output[t] = bypass_scale · pre[t] + (1 - bypass_scale) · post[t]`.
    /// Upstream initialises this to ~1.0 and slowly anneals; here it is a
    /// static per-stack scalar the checkpoint pins.
    pub bypass_scale: f32,
}

impl ZipformerStackWeights {
    fn validate(
        &self,
        cfg: &ZipformerConfig,
        desc: &ZipformerStackDesc,
        stack_idx: usize,
    ) -> Result<()> {
        let d_model = cfg.d_model as usize;
        let ffn_dim = cfg.ffn_dim as usize;
        let kernel = desc.kernel_size as usize;
        let tag = format!("Zipformer stack {stack_idx}");
        self.shared_qk.validate(d_model, &tag)?;
        if self.layers.len() != desc.n_layers as usize {
            return Err(VokraError::InvalidArgument(format!(
                "{tag}: expected {} layers, got {}",
                desc.n_layers,
                self.layers.len(),
            )));
        }
        for (idx, layer) in self.layers.iter().enumerate() {
            layer.validate(d_model, ffn_dim, kernel, &format!("{tag} layer {idx}"))?;
        }
        if !self.bypass_scale.is_finite() {
            return Err(VokraError::InvalidArgument(format!(
                "{tag}: bypass_scale must be finite, got {}",
                self.bypass_scale,
            )));
        }
        Ok(())
    }
}

/// All learned parameters an encoder owns.
#[derive(Debug, Clone)]
pub struct ZipformerWeights {
    /// Stem subsample.
    pub stem: ZipformerStemWeights,
    /// One entry per stack (must match `cfg.stacks.len()`).
    pub stacks: Vec<ZipformerStackWeights>,
}

impl ZipformerWeights {
    fn validate(&self, cfg: &ZipformerConfig) -> Result<()> {
        self.stem.validate(cfg)?;
        if cfg.stacks.is_empty() {
            return Err(VokraError::InvalidArgument(
                "ZipformerConfig: stacks must be non-empty".to_owned(),
            ));
        }
        if self.stacks.len() != cfg.stacks.len() {
            return Err(VokraError::InvalidArgument(format!(
                "Zipformer weights: expected {} stacks, got {}",
                cfg.stacks.len(),
                self.stacks.len(),
            )));
        }
        for (idx, (desc, stack)) in cfg.stacks.iter().zip(self.stacks.iter()).enumerate() {
            stack.validate(cfg, desc, idx)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Main encoder
// ---------------------------------------------------------------------------

/// Zipformer encoder — multi-resolution stack with shared-QK attention.
///
/// See the module docstring for the full block sequence. Use [`Self::new`]
/// to shape-check weights and [`Self::forward`] to run the encoder.
#[derive(Debug, Clone)]
pub struct ZipformerEncoder {
    cfg: ZipformerConfig,
    weights: ZipformerWeights,
}

impl ZipformerEncoder {
    /// Build an encoder from its config + weights.
    ///
    /// Fails loudly on any shape mismatch, on `d_model % n_heads != 0`, on
    /// an empty stack list, on any `n_layers == 0`, on any even
    /// `kernel_size`, on any `downsampling_factor == 0`, or on a non-finite
    /// `bypass_scale`.
    pub fn new(cfg: ZipformerConfig, weights: ZipformerWeights) -> Result<Self> {
        if cfg.d_model == 0 {
            return Err(VokraError::InvalidArgument(
                "ZipformerConfig: d_model must be > 0".to_owned(),
            ));
        }
        if cfg.n_heads == 0 {
            return Err(VokraError::InvalidArgument(
                "ZipformerConfig: n_heads must be > 0".to_owned(),
            ));
        }
        if cfg.d_model % cfg.n_heads != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "ZipformerConfig: d_model ({}) must be divisible by n_heads ({})",
                cfg.d_model, cfg.n_heads,
            )));
        }
        if cfg.ffn_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "ZipformerConfig: ffn_dim must be > 0".to_owned(),
            ));
        }
        if cfg.in_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "ZipformerConfig: in_dim must be > 0".to_owned(),
            ));
        }
        for (idx, desc) in cfg.stacks.iter().enumerate() {
            if desc.n_layers == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "ZipformerConfig stack {idx}: n_layers must be > 0",
                )));
            }
            if desc.kernel_size == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "ZipformerConfig stack {idx}: kernel_size must be > 0",
                )));
            }
            if desc.kernel_size % 2 == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "ZipformerConfig stack {idx}: kernel_size must be odd for symmetric \
                     same-padding, got {}",
                    desc.kernel_size,
                )));
            }
            if desc.downsampling_factor == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "ZipformerConfig stack {idx}: downsampling_factor must be > 0",
                )));
            }
        }
        if let ConvSubsampleKind::Stacking { factor } | ConvSubsampleKind::StackingNorm { factor } =
            cfg.stem
        {
            if factor == 0 {
                return Err(VokraError::InvalidArgument(
                    "ZipformerConfig: stem stacking factor must be > 0".to_owned(),
                ));
            }
        }
        weights.validate(&cfg)?;
        Ok(Self { cfg, weights })
    }

    /// Immutable access to the [`ZipformerConfig`] the encoder was built
    /// with.
    pub fn config(&self) -> &ZipformerConfig {
        &self.cfg
    }

    /// Full forward pass — mel → encoded hidden state.
    ///
    /// `mel` is a row-major `[mel_frames, in_dim]` buffer. Returns
    /// `(hidden, T_out)` where `T_out` is the time count after the stem
    /// downsampling (subsequent stacks down/up-sample internally and restore
    /// the same length via the bypass fuse; the top-level output rate
    /// therefore equals the stem-subsampled rate).
    pub fn forward(&self, mel: &[f32], mel_frames: usize) -> Result<(Vec<f32>, usize)> {
        let in_dim = self.cfg.in_dim as usize;
        if mel_frames == 0 {
            return Err(VokraError::InvalidArgument(
                "ZipformerEncoder::forward: mel_frames must be > 0".to_owned(),
            ));
        }
        let expected_len = mel_frames * in_dim;
        if mel.len() != expected_len {
            return Err(VokraError::InvalidArgument(format!(
                "ZipformerEncoder::forward: mel length {} does not match \
                 mel_frames×in_dim = {mel_frames}×{in_dim} = {expected_len}",
                mel.len(),
            )));
        }

        // Stem: mel → hidden [T_stem, d_model].
        let (mut hidden, t_stem) = self.stem_forward(mel, mel_frames)?;
        if t_stem == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "ZipformerEncoder::forward: stem-subsampled sequence is empty \
                 (mel_frames={mel_frames}, stem_factor={})",
                self.cfg.stem.factor(),
            )));
        }

        // Per-stack pyramid.
        for (idx, (desc, stack)) in self
            .cfg
            .stacks
            .iter()
            .zip(self.weights.stacks.iter())
            .enumerate()
        {
            hidden = self.stack_forward(&hidden, t_stem, desc, stack, idx)?;
        }

        Ok((hidden, t_stem))
    }

    // -----------------------------------------------------------------------
    // Stem subsample
    // -----------------------------------------------------------------------

    fn stem_forward(&self, mel: &[f32], mel_frames: usize) -> Result<(Vec<f32>, usize)> {
        let in_dim = self.cfg.in_dim as usize;
        let d_model = self.cfg.d_model as usize;
        let stem_w = &self.weights.stem;
        match self.cfg.stem {
            ConvSubsampleKind::Linear => {
                let mut out = vec![0.0f32; mel_frames * d_model];
                for t in 0..mel_frames {
                    linear_row(
                        &mel[t * in_dim..(t + 1) * in_dim],
                        &stem_w.linear_w,
                        &stem_w.linear_b,
                        d_model,
                        in_dim,
                        &mut out[t * d_model..(t + 1) * d_model],
                    );
                }
                Ok((out, mel_frames))
            }
            ConvSubsampleKind::Stacking { factor } | ConvSubsampleKind::StackingNorm { factor } => {
                let factor = factor as usize;
                let t_out = mel_frames / factor;
                let proj_in = factor * in_dim;
                let mut out = vec![0.0f32; t_out * d_model];
                for t in 0..t_out {
                    let src = t * factor * in_dim;
                    linear_row(
                        &mel[src..src + proj_in],
                        &stem_w.linear_w,
                        &stem_w.linear_b,
                        d_model,
                        proj_in,
                        &mut out[t * d_model..(t + 1) * d_model],
                    );
                }
                if let (Some(gamma), Some(beta)) = (&stem_w.norm_gamma, &stem_w.norm_beta) {
                    for t in 0..t_out {
                        let row = &mut out[t * d_model..(t + 1) * d_model];
                        layer_norm_inplace(row, gamma, beta);
                    }
                }
                Ok((out, t_out))
            }
        }
    }

    // -----------------------------------------------------------------------
    // Per-stack forward: down → layers with shared QK → up → bypass fuse
    // -----------------------------------------------------------------------

    fn stack_forward(
        &self,
        input: &[f32],
        t_in: usize,
        desc: &ZipformerStackDesc,
        stack: &ZipformerStackWeights,
        stack_idx: usize,
    ) -> Result<Vec<f32>> {
        let d_model = self.cfg.d_model as usize;
        let ds = desc.downsampling_factor as usize;

        // 1. Downsample: mean over `ds` adjacent frames (upstream
        //    `SimpleDownsample`). Trailing partial block is dropped
        //    (matches upstream stride semantics).
        let (down, t_down) = downsample_mean(input, t_in, d_model, ds);
        if t_down == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "ZipformerEncoder stack {stack_idx}: downsampling to zero frames \
                 (t_in={t_in}, downsampling_factor={ds})",
            )));
        }

        // 2. Shared attention over the down-sampled sequence: Q and K are
        //    the stack-level shared projections; every layer reuses the
        //    resulting softmax output when combining its own V.
        let n_heads = self.cfg.n_heads as usize;
        let head_dim = self.cfg.head_dim();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut q = vec![0.0f32; t_down * d_model];
        let mut k = vec![0.0f32; t_down * d_model];
        for ti in 0..t_down {
            let src = &down[ti * d_model..(ti + 1) * d_model];
            linear_row(
                src,
                &stack.shared_qk.wq,
                &stack.shared_qk.bq,
                d_model,
                d_model,
                &mut q[ti * d_model..(ti + 1) * d_model],
            );
            linear_row(
                src,
                &stack.shared_qk.wk,
                &stack.shared_qk.bk,
                d_model,
                d_model,
                &mut k[ti * d_model..(ti + 1) * d_model],
            );
        }

        // Optional RoPE overlay on shared Q / K (upstream default).
        if let PositionEncoding::Rope { theta } = self.cfg.position_encoding {
            apply_rope(&mut q, t_down, n_heads, head_dim, theta);
            apply_rope(&mut k, t_down, n_heads, head_dim, theta);
        }

        // Compute per-head softmax probs once for the whole stack.
        // probs_by_head[h] is a row-major [t_down, t_down] matrix.
        let mut probs_by_head: Vec<Vec<f32>> = Vec::with_capacity(n_heads);
        let mut scratch_scores = vec![0.0f32; t_down * t_down];
        for h in 0..n_heads {
            let head_off = h * head_dim;
            for i in 0..t_down {
                let q_row = &q[i * d_model + head_off..i * d_model + head_off + head_dim];
                for j in 0..t_down {
                    let k_row = &k[j * d_model + head_off..j * d_model + head_off + head_dim];
                    let mut acc = 0.0f32;
                    for d in 0..head_dim {
                        acc += q_row[d] * k_row[d];
                    }
                    scratch_scores[i * t_down + j] = acc * scale;
                }
            }
            let mut probs = vec![0.0f32; t_down * t_down];
            for i in 0..t_down {
                softmax_row(
                    &scratch_scores[i * t_down..(i + 1) * t_down],
                    &mut probs[i * t_down..(i + 1) * t_down],
                );
            }
            probs_by_head.push(probs);
        }

        // 3. Per-layer stack over the down-sampled sequence.
        let mut hidden = down;
        for layer in &stack.layers {
            hidden = self.zipformer_layer(&hidden, t_down, layer, &probs_by_head, desc)?;
        }

        // 4. Upsample: repeat every down-sampled frame `ds` times. Trailing
        //    frames after the block cutoff are filled from the last block
        //    (upstream `SimpleUpsample` uses transposed conv w/ zero-hold on
        //    the tail; block-repeat is the numerically-equivalent
        //    identity-init fallback).
        let mut up = upsample_repeat(&hidden, t_down, d_model, ds, t_in);

        // 5. Bypass fuse: output = bypass · pre + (1 - bypass) · up.
        let bypass = stack.bypass_scale;
        let one_minus = 1.0 - bypass;
        for i in 0..(t_in * d_model) {
            up[i] = bypass * input[i] + one_minus * up[i];
        }

        Ok(up)
    }

    fn zipformer_layer(
        &self,
        input: &[f32],
        t: usize,
        w: &ZipformerLayerWeights,
        probs_by_head: &[Vec<f32>],
        _desc: &ZipformerStackDesc,
    ) -> Result<Vec<f32>> {
        let d_model = self.cfg.d_model as usize;
        let ffn_dim = self.cfg.ffn_dim as usize;
        let n_heads = self.cfg.n_heads as usize;
        let head_dim = self.cfg.head_dim();

        // ---- FF1: residual += 0.5 * FF1(LN1(x)) ---------------------------
        let mut residual = input.to_vec();
        let mut buf = residual.clone();
        for row_off in (0..residual.len()).step_by(d_model) {
            layer_norm_inplace(
                &mut buf[row_off..row_off + d_model],
                &w.ln1_gamma,
                &w.ln1_beta,
            );
        }
        let ff1_out = feed_forward(&buf, t, d_model, ffn_dim, &w.ff1);
        add_scaled_inplace(&mut residual, &ff1_out, 0.5);

        // ---- Shared-QK attention: residual += Wo·(P·V) --------------------
        // Compute per-layer V from LN2(residual), combine with the shared
        // probs (once per head), then per-layer output projection.
        buf.copy_from_slice(&residual);
        for row_off in (0..buf.len()).step_by(d_model) {
            layer_norm_inplace(
                &mut buf[row_off..row_off + d_model],
                &w.ln2_gamma,
                &w.ln2_beta,
            );
        }
        let mut v = vec![0.0f32; t * d_model];
        for ti in 0..t {
            let src = &buf[ti * d_model..(ti + 1) * d_model];
            linear_row(
                src,
                &w.wv,
                &w.bv,
                d_model,
                d_model,
                &mut v[ti * d_model..(ti + 1) * d_model],
            );
        }
        let mut context = vec![0.0f32; t * d_model];
        for (h, probs) in probs_by_head.iter().enumerate().take(n_heads) {
            let head_off = h * head_dim;
            for i in 0..t {
                for j in 0..t {
                    let p = probs[i * t + j];
                    if p == 0.0 {
                        continue;
                    }
                    let v_row = &v[j * d_model + head_off..j * d_model + head_off + head_dim];
                    let ctx_row =
                        &mut context[i * d_model + head_off..i * d_model + head_off + head_dim];
                    for d in 0..head_dim {
                        ctx_row[d] += p * v_row[d];
                    }
                }
            }
        }
        let mut attn_out = vec![0.0f32; t * d_model];
        for i in 0..t {
            linear_row(
                &context[i * d_model..(i + 1) * d_model],
                &w.wo,
                &w.bo,
                d_model,
                d_model,
                &mut attn_out[i * d_model..(i + 1) * d_model],
            );
        }
        add_inplace(&mut residual, &attn_out);

        // ---- Conv branch: residual += Conv(LN3(residual)) ------------------
        buf.copy_from_slice(&residual);
        for row_off in (0..buf.len()).step_by(d_model) {
            layer_norm_inplace(
                &mut buf[row_off..row_off + d_model],
                &w.ln3_gamma,
                &w.ln3_beta,
            );
        }
        // Convolution reuses the Conformer primitive's chain via a
        // module-level helper to keep this file self-contained (see
        // `zipformer_conv` below).
        let conv_out = zipformer_conv(&buf, t, d_model, _desc.kernel_size as usize, &w.conv)?;
        add_inplace(&mut residual, &conv_out);

        // ---- FF2: residual += 0.5 * FF2(LN4(residual)) --------------------
        buf.copy_from_slice(&residual);
        for row_off in (0..buf.len()).step_by(d_model) {
            layer_norm_inplace(
                &mut buf[row_off..row_off + d_model],
                &w.ln4_gamma,
                &w.ln4_beta,
            );
        }
        let ff2_out = feed_forward(&buf, t, d_model, ffn_dim, &w.ff2);
        add_scaled_inplace(&mut residual, &ff2_out, 0.5);

        // ---- Final per-layer norm ------------------------------------------
        for row_off in (0..residual.len()).step_by(d_model) {
            layer_norm_inplace(
                &mut residual[row_off..row_off + d_model],
                &w.ln_out_gamma,
                &w.ln_out_beta,
            );
        }
        Ok(residual)
    }
}

// ---------------------------------------------------------------------------
// Local helpers — copies of the Conformer helpers so we do not force pub
// exports from that module. Kept in a single spot for zero-dep clarity.
// ---------------------------------------------------------------------------

/// Extension trait pattern via inherent-method sugar — we cannot add
/// methods to `FeedForwardWeights` / `ConformerConvWeights` directly
/// without touching the conformer module, so we shadow the validators here
/// using a private trait.
trait ValidateExt {
    fn validate_ext(&self, d_model: usize, other: usize, tag: &str) -> Result<()>;
}

impl ValidateExt for FeedForwardWeights {
    fn validate_ext(&self, d_model: usize, ffn_dim: usize, tag: &str) -> Result<()> {
        if self.w1.len() != ffn_dim * d_model {
            return Err(VokraError::InvalidArgument(format!(
                "{tag}: w1 must be length {}×{}={}, got {}",
                ffn_dim,
                d_model,
                ffn_dim * d_model,
                self.w1.len(),
            )));
        }
        if self.b1.len() != ffn_dim {
            return Err(VokraError::InvalidArgument(format!(
                "{tag}: b1 must be length {ffn_dim}, got {}",
                self.b1.len(),
            )));
        }
        if self.w2.len() != d_model * ffn_dim {
            return Err(VokraError::InvalidArgument(format!(
                "{tag}: w2 must be length {}×{}={}, got {}",
                d_model,
                ffn_dim,
                d_model * ffn_dim,
                self.w2.len(),
            )));
        }
        if self.b2.len() != d_model {
            return Err(VokraError::InvalidArgument(format!(
                "{tag}: b2 must be length {d_model}, got {}",
                self.b2.len(),
            )));
        }
        Ok(())
    }
}

impl ValidateExt for ConformerConvWeights {
    fn validate_ext(&self, d_model: usize, kernel_size: usize, tag: &str) -> Result<()> {
        let two_d = 2 * d_model;
        if self.pointwise1_w.len() != two_d * d_model {
            return Err(VokraError::InvalidArgument(format!(
                "{tag}: pointwise1_w must be length {}×{}={}, got {}",
                two_d,
                d_model,
                two_d * d_model,
                self.pointwise1_w.len(),
            )));
        }
        if self.pointwise1_b.len() != two_d {
            return Err(VokraError::InvalidArgument(format!(
                "{tag}: pointwise1_b must be length {two_d}, got {}",
                self.pointwise1_b.len(),
            )));
        }
        if self.depthwise_w.len() != d_model * kernel_size {
            return Err(VokraError::InvalidArgument(format!(
                "{tag}: depthwise_w must be length {}×{}={}, got {}",
                d_model,
                kernel_size,
                d_model * kernel_size,
                self.depthwise_w.len(),
            )));
        }
        if self.depthwise_b.len() != d_model {
            return Err(VokraError::InvalidArgument(format!(
                "{tag}: depthwise_b must be length {d_model}, got {}",
                self.depthwise_b.len(),
            )));
        }
        if self.norm_gamma.len() != d_model || self.norm_beta.len() != d_model {
            return Err(VokraError::InvalidArgument(format!(
                "{tag}: norm gamma/beta must be length {d_model}, \
                 got gamma={} beta={}",
                self.norm_gamma.len(),
                self.norm_beta.len(),
            )));
        }
        if self.pointwise2_w.len() != d_model * d_model {
            return Err(VokraError::InvalidArgument(format!(
                "{tag}: pointwise2_w must be length {}×{}={}, got {}",
                d_model,
                d_model,
                d_model * d_model,
                self.pointwise2_w.len(),
            )));
        }
        if self.pointwise2_b.len() != d_model {
            return Err(VokraError::InvalidArgument(format!(
                "{tag}: pointwise2_b must be length {d_model}, got {}",
                self.pointwise2_b.len(),
            )));
        }
        Ok(())
    }
}

fn feed_forward(
    x: &[f32],
    t: usize,
    d_model: usize,
    ffn_dim: usize,
    w: &FeedForwardWeights,
) -> Vec<f32> {
    let mut hidden = vec![0.0f32; t * ffn_dim];
    let mut out = vec![0.0f32; t * d_model];
    for ti in 0..t {
        let src = &x[ti * d_model..(ti + 1) * d_model];
        let mid = &mut hidden[ti * ffn_dim..(ti + 1) * ffn_dim];
        linear_row(src, &w.w1, &w.b1, ffn_dim, d_model, mid);
        for v in mid.iter_mut() {
            *v = swish(*v);
        }
        linear_row(
            mid,
            &w.w2,
            &w.b2,
            d_model,
            ffn_dim,
            &mut out[ti * d_model..(ti + 1) * d_model],
        );
    }
    out
}

fn zipformer_conv(
    x: &[f32],
    t: usize,
    d_model: usize,
    kernel_size: usize,
    w: &ConformerConvWeights,
) -> Result<Vec<f32>> {
    let two_d = 2 * d_model;
    let mut expanded = vec![0.0f32; t * two_d];
    for ti in 0..t {
        linear_row(
            &x[ti * d_model..(ti + 1) * d_model],
            &w.pointwise1_w,
            &w.pointwise1_b,
            two_d,
            d_model,
            &mut expanded[ti * two_d..(ti + 1) * two_d],
        );
    }
    let mut glued = vec![0.0f32; t * d_model];
    for ti in 0..t {
        let row = &expanded[ti * two_d..(ti + 1) * two_d];
        let out_row = &mut glued[ti * d_model..(ti + 1) * d_model];
        for c in 0..d_model {
            out_row[c] = row[c] * sigmoid(row[d_model + c]);
        }
    }
    let mut ct = vec![0.0f32; d_model * t];
    for ti in 0..t {
        for c in 0..d_model {
            ct[c * t + ti] = glued[ti * d_model + c];
        }
    }
    if kernel_size == 0 {
        return Err(VokraError::InvalidArgument(
            "zipformer_conv: kernel_size must be > 0".to_owned(),
        ));
    }
    let padding = kernel_size / 2;
    let mut conv_out_ct = vec![0.0f32; d_model * t];
    let t_i = t as isize;
    let pad_i = padding as isize;
    for c in 0..d_model {
        let filter = &w.depthwise_w[c * kernel_size..(c + 1) * kernel_size];
        let bias = w.depthwise_b[c];
        let src_row = &ct[c * t..(c + 1) * t];
        let dst_row = &mut conv_out_ct[c * t..(c + 1) * t];
        for (ti, dst_slot) in dst_row.iter_mut().enumerate() {
            let mut acc = bias;
            for (k, &tap) in filter.iter().enumerate() {
                let src = ti as isize + k as isize - pad_i;
                if src < 0 || src >= t_i {
                    continue;
                }
                acc += src_row[src as usize] * tap;
            }
            *dst_slot = acc;
        }
    }
    let mut normed = vec![0.0f32; t * d_model];
    for c in 0..d_model {
        for ti in 0..t {
            normed[ti * d_model + c] = conv_out_ct[c * t + ti];
        }
    }
    for ti in 0..t {
        let row = &mut normed[ti * d_model..(ti + 1) * d_model];
        layer_norm_inplace(row, &w.norm_gamma, &w.norm_beta);
        for v in row.iter_mut() {
            *v = swish(*v);
        }
    }
    let mut out = vec![0.0f32; t * d_model];
    for ti in 0..t {
        linear_row(
            &normed[ti * d_model..(ti + 1) * d_model],
            &w.pointwise2_w,
            &w.pointwise2_b,
            d_model,
            d_model,
            &mut out[ti * d_model..(ti + 1) * d_model],
        );
    }
    Ok(out)
}

/// Downsample `x` `[t_in, d]` by mean-pooling `ds` adjacent frames.
/// Trailing partial block is dropped (upstream `SimpleDownsample` mean
/// over stride-`ds` blocks).
fn downsample_mean(x: &[f32], t_in: usize, d: usize, ds: usize) -> (Vec<f32>, usize) {
    if ds == 0 {
        // Guarded by caller; return empty rather than divide-by-zero.
        return (Vec::new(), 0);
    }
    if ds == 1 {
        return (x.to_vec(), t_in);
    }
    let t_out = t_in / ds;
    let mut out = vec![0.0f32; t_out * d];
    let scale = 1.0 / (ds as f32);
    for ti in 0..t_out {
        for j in 0..ds {
            let src = (ti * ds + j) * d;
            for c in 0..d {
                out[ti * d + c] += x[src + c];
            }
        }
        for c in 0..d {
            out[ti * d + c] *= scale;
        }
    }
    (out, t_out)
}

/// Upsample `x` `[t_in, d]` back to `t_out` frames by block-repeating each
/// input frame `ds` times, then padding any tail with the last block.
fn upsample_repeat(x: &[f32], t_in: usize, d: usize, ds: usize, t_out: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; t_out * d];
    for ti in 0..t_in {
        for j in 0..ds {
            let dst_idx = ti * ds + j;
            if dst_idx >= t_out {
                break;
            }
            let dst = dst_idx * d;
            let src = ti * d;
            out[dst..dst + d].copy_from_slice(&x[src..src + d]);
        }
    }
    // Fill any trailing frames from the last upsample block.
    if t_in > 0 {
        let last_src = (t_in - 1) * d;
        for dst_idx in (t_in * ds)..t_out {
            let dst = dst_idx * d;
            out[dst..dst + d].copy_from_slice(&x[last_src..last_src + d]);
        }
    }
    out
}

#[inline]
fn linear_row(src: &[f32], w: &[f32], b: &[f32], out_dim: usize, in_dim: usize, out: &mut [f32]) {
    for (o, &bias) in b.iter().enumerate().take(out_dim) {
        let mut acc = bias;
        let w_row = &w[o * in_dim..(o + 1) * in_dim];
        for i in 0..in_dim {
            acc += w_row[i] * src[i];
        }
        out[o] = acc;
    }
}

#[inline]
fn layer_norm_inplace(row: &mut [f32], gamma: &[f32], beta: &[f32]) {
    let n = row.len() as f32;
    let mean = row.iter().sum::<f32>() / n;
    let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let inv = 1.0 / (var + 1e-5).sqrt();
    for i in 0..row.len() {
        row[i] = (row[i] - mean) * inv * gamma[i] + beta[i];
    }
}

#[inline]
fn softmax_row(src: &[f32], dst: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &v in src {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        let e = (s - max).exp();
        *d = e;
        sum += e;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for d in dst.iter_mut() {
            *d *= inv;
        }
    }
}

#[inline]
fn add_inplace(dst: &mut [f32], src: &[f32]) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += s;
    }
}

#[inline]
fn add_scaled_inplace(dst: &mut [f32], src: &[f32], scale: f32) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += scale * s;
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline]
fn swish(x: f32) -> f32 {
    x * sigmoid(x)
}

fn apply_rope(x: &mut [f32], t: usize, n_heads: usize, head_dim: usize, theta: f32) {
    if head_dim < 2 {
        return;
    }
    let half = head_dim / 2;
    let d_model = n_heads * head_dim;
    let mut inv_freqs = vec![0.0f32; half];
    for (k, slot) in inv_freqs.iter_mut().enumerate() {
        *slot = theta.powf(-(2.0 * k as f32) / head_dim as f32);
    }
    for ti in 0..t {
        for h in 0..n_heads {
            let base = ti * d_model + h * head_dim;
            for (k, &inv_freq) in inv_freqs.iter().enumerate() {
                let angle = ti as f32 * inv_freq;
                let (s, c) = angle.sin_cos();
                let a = x[base + 2 * k];
                let b = x[base + 2 * k + 1];
                x[base + 2 * k] = a * c - b * s;
                x[base + 2 * k + 1] = a * s + b * c;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_uniform(state: &mut u64, scale: f32) -> f32 {
        let bits = splitmix64(state) >> 40;
        (bits as f32) * (1.0 / (1u32 << 24) as f32) * 2.0 * scale - scale
    }

    fn synth_vec(state: &mut u64, len: usize, scale: f32) -> Vec<f32> {
        (0..len).map(|_| next_uniform(state, scale)).collect()
    }

    fn synth_ff(state: &mut u64, d_model: usize, ffn_dim: usize) -> FeedForwardWeights {
        FeedForwardWeights {
            w1: synth_vec(state, ffn_dim * d_model, 0.1),
            b1: synth_vec(state, ffn_dim, 0.1),
            w2: synth_vec(state, d_model * ffn_dim, 0.1),
            b2: synth_vec(state, d_model, 0.1),
        }
    }

    fn synth_conv(state: &mut u64, d_model: usize, kernel: usize) -> ConformerConvWeights {
        ConformerConvWeights {
            pointwise1_w: synth_vec(state, 2 * d_model * d_model, 0.1),
            pointwise1_b: synth_vec(state, 2 * d_model, 0.1),
            depthwise_w: synth_vec(state, d_model * kernel, 0.1),
            depthwise_b: synth_vec(state, d_model, 0.1),
            norm_gamma: vec![1.0; d_model],
            norm_beta: vec![0.0; d_model],
            pointwise2_w: synth_vec(state, d_model * d_model, 0.1),
            pointwise2_b: synth_vec(state, d_model, 0.1),
        }
    }

    fn synth_shared_qk(state: &mut u64, d_model: usize) -> SharedMhaQkWeights {
        let dd = d_model * d_model;
        SharedMhaQkWeights {
            wq: synth_vec(state, dd, 0.1),
            bq: synth_vec(state, d_model, 0.1),
            wk: synth_vec(state, dd, 0.1),
            bk: synth_vec(state, d_model, 0.1),
        }
    }

    fn synth_layer(
        state: &mut u64,
        d_model: usize,
        ffn_dim: usize,
        kernel: usize,
    ) -> ZipformerLayerWeights {
        let dd = d_model * d_model;
        ZipformerLayerWeights {
            ln1_gamma: vec![1.0; d_model],
            ln1_beta: vec![0.0; d_model],
            ff1: synth_ff(state, d_model, ffn_dim),
            ln2_gamma: vec![1.0; d_model],
            ln2_beta: vec![0.0; d_model],
            wv: synth_vec(state, dd, 0.1),
            bv: synth_vec(state, d_model, 0.1),
            wo: synth_vec(state, dd, 0.1),
            bo: synth_vec(state, d_model, 0.1),
            ln3_gamma: vec![1.0; d_model],
            ln3_beta: vec![0.0; d_model],
            conv: synth_conv(state, d_model, kernel),
            ln4_gamma: vec![1.0; d_model],
            ln4_beta: vec![0.0; d_model],
            ff2: synth_ff(state, d_model, ffn_dim),
            ln_out_gamma: vec![1.0; d_model],
            ln_out_beta: vec![0.0; d_model],
        }
    }

    fn synth_stack(
        state: &mut u64,
        d_model: usize,
        ffn_dim: usize,
        desc: &ZipformerStackDesc,
    ) -> ZipformerStackWeights {
        let kernel = desc.kernel_size as usize;
        let layers = (0..desc.n_layers)
            .map(|_| synth_layer(state, d_model, ffn_dim, kernel))
            .collect();
        ZipformerStackWeights {
            shared_qk: synth_shared_qk(state, d_model),
            layers,
            bypass_scale: 0.8,
        }
    }

    fn synth_weights(cfg: &ZipformerConfig, state: &mut u64) -> ZipformerWeights {
        let d_model = cfg.d_model as usize;
        let ffn_dim = cfg.ffn_dim as usize;
        let in_dim = cfg.in_dim as usize;
        let proj_in = cfg.stem.projection_in_dim(in_dim);
        let stem = ZipformerStemWeights {
            linear_w: synth_vec(state, d_model * proj_in, 0.1),
            linear_b: synth_vec(state, d_model, 0.1),
            norm_gamma: if cfg.stem.has_norm() {
                Some(vec![1.0; d_model])
            } else {
                None
            },
            norm_beta: if cfg.stem.has_norm() {
                Some(vec![0.0; d_model])
            } else {
                None
            },
        };
        let stacks = cfg
            .stacks
            .iter()
            .map(|desc| synth_stack(state, d_model, ffn_dim, desc))
            .collect();
        ZipformerWeights { stem, stacks }
    }

    fn small_cfg() -> ZipformerConfig {
        ZipformerConfig {
            in_dim: 4,
            d_model: 8,
            n_heads: 2,
            ffn_dim: 16,
            stem: ConvSubsampleKind::Linear,
            position_encoding: PositionEncoding::None,
            stacks: vec![
                ZipformerStackDesc {
                    downsampling_factor: 1,
                    n_layers: 1,
                    kernel_size: 3,
                },
                ZipformerStackDesc {
                    downsampling_factor: 2,
                    n_layers: 1,
                    kernel_size: 3,
                },
            ],
        }
    }

    // ---- Happy path ------------------------------------------------------

    #[test]
    fn forward_produces_expected_output_shape_linear_stem() {
        let cfg = small_cfg();
        let mut state = 1u64;
        let weights = synth_weights(&cfg, &mut state);
        let encoder = ZipformerEncoder::new(cfg.clone(), weights).unwrap();
        let t = 12;
        let mel = synth_vec(&mut state, t * cfg.in_dim as usize, 1.0);
        let (out, t_out) = encoder.forward(&mel, t).unwrap();
        assert_eq!(t_out, t, "Linear stem must not change T");
        assert_eq!(out.len(), t_out * cfg.d_model as usize);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_stacking_stem_downsamples() {
        let mut cfg = small_cfg();
        cfg.stem = ConvSubsampleKind::Stacking { factor: 4 };
        let mut state = 2u64;
        let weights = synth_weights(&cfg, &mut state);
        let encoder = ZipformerEncoder::new(cfg.clone(), weights).unwrap();
        // 16 frames → 16 / 4 = 4 stem-output frames.
        let mel = synth_vec(&mut state, 16 * cfg.in_dim as usize, 1.0);
        let (out, t_out) = encoder.forward(&mel, 16).unwrap();
        assert_eq!(t_out, 4);
        assert_eq!(out.len(), 4 * cfg.d_model as usize);
    }

    #[test]
    fn per_stack_downsample_and_upsample_restore_time_axis() {
        // Even with a per-stack downsampling factor > 1, the stack's up-sample
        // must restore the top-level T so subsequent stacks operate at the
        // stem-subsampled rate.
        let cfg = ZipformerConfig {
            in_dim: 4,
            d_model: 8,
            n_heads: 2,
            ffn_dim: 16,
            stem: ConvSubsampleKind::Linear,
            position_encoding: PositionEncoding::None,
            stacks: vec![
                ZipformerStackDesc {
                    downsampling_factor: 4,
                    n_layers: 2,
                    kernel_size: 3,
                },
                ZipformerStackDesc {
                    downsampling_factor: 2,
                    n_layers: 1,
                    kernel_size: 3,
                },
            ],
        };
        let mut state = 3u64;
        let weights = synth_weights(&cfg, &mut state);
        let encoder = ZipformerEncoder::new(cfg.clone(), weights).unwrap();
        let t = 16;
        let mel = synth_vec(&mut state, t * cfg.in_dim as usize, 1.0);
        let (out, t_out) = encoder.forward(&mel, t).unwrap();
        assert_eq!(t_out, t, "T_out must equal stem T");
        assert_eq!(out.len(), t * cfg.d_model as usize);
    }

    #[test]
    fn shared_qk_reused_across_multiple_layers() {
        // A stack with 3 layers must run without exploding on synthetic
        // weights (LN keeps activations bounded); the shared-QK path is
        // exercised implicitly.
        let cfg = ZipformerConfig {
            in_dim: 4,
            d_model: 8,
            n_heads: 2,
            ffn_dim: 16,
            stem: ConvSubsampleKind::Linear,
            position_encoding: PositionEncoding::None,
            stacks: vec![ZipformerStackDesc {
                downsampling_factor: 2,
                n_layers: 3,
                kernel_size: 3,
            }],
        };
        let mut state = 4u64;
        let weights = synth_weights(&cfg, &mut state);
        let encoder = ZipformerEncoder::new(cfg.clone(), weights).unwrap();
        let mel = synth_vec(&mut state, 8 * cfg.in_dim as usize, 1.0);
        let (out, t_out) = encoder.forward(&mel, 8).unwrap();
        assert_eq!(t_out, 8);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ---- RoPE overlay ---------------------------------------------------

    #[test]
    fn rope_overlay_changes_shared_qk_output() {
        let mut state = 5u64;
        let mut cfg = small_cfg();
        let weights = synth_weights(&cfg, &mut state);
        let enc_none = ZipformerEncoder::new(cfg.clone(), weights.clone()).unwrap();
        cfg.position_encoding = PositionEncoding::Rope { theta: 10_000.0 };
        let enc_rope = ZipformerEncoder::new(cfg.clone(), weights).unwrap();
        let mel = synth_vec(&mut state, 8 * cfg.in_dim as usize, 1.0);
        let (out_none, _) = enc_none.forward(&mel, 8).unwrap();
        let (out_rope, _) = enc_rope.forward(&mel, 8).unwrap();
        let any_diff = out_none
            .iter()
            .zip(out_rope.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(any_diff, "RoPE overlay must change the shared-QK output");
    }

    // ---- Bypass scalar --------------------------------------------------

    #[test]
    fn bypass_one_equals_input_when_conv_ff_are_bounded() {
        // With bypass_scale == 1.0 the stack output should equal the input
        // (the up-sampled post signal is discarded).
        let cfg = small_cfg();
        let mut state = 6u64;
        let mut weights = synth_weights(&cfg, &mut state);
        for stack in weights.stacks.iter_mut() {
            stack.bypass_scale = 1.0;
        }
        let encoder = ZipformerEncoder::new(cfg.clone(), weights).unwrap();
        let mel = synth_vec(&mut state, 8 * cfg.in_dim as usize, 1.0);
        let (out, _) = encoder.forward(&mel, 8).unwrap();
        // With bypass_scale=1.0 the input propagates through every stack
        // unchanged; the output must equal the stem projection (identity
        // through every stack).
        let (expected, _) = encoder.stem_forward(&mel, 8).unwrap();
        for (o, e) in out.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-6, "got {o}, expected {e}");
        }
    }

    #[test]
    fn bypass_zero_uses_only_processed_signal() {
        // With bypass_scale == 0.0 the pre-stack signal is discarded — the
        // output is the up-sampled processed signal. Must run and stay
        // finite.
        let cfg = small_cfg();
        let mut state = 7u64;
        let mut weights = synth_weights(&cfg, &mut state);
        for stack in weights.stacks.iter_mut() {
            stack.bypass_scale = 0.0;
        }
        let encoder = ZipformerEncoder::new(cfg.clone(), weights).unwrap();
        let mel = synth_vec(&mut state, 8 * cfg.in_dim as usize, 1.0);
        let (out, _) = encoder.forward(&mel, 8).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ---- Determinism ----------------------------------------------------

    #[test]
    fn forward_is_deterministic() {
        let cfg = small_cfg();
        let mut state = 8u64;
        let weights = synth_weights(&cfg, &mut state);
        let encoder = ZipformerEncoder::new(cfg.clone(), weights).unwrap();
        let mel = synth_vec(&mut state, 8 * cfg.in_dim as usize, 1.0);
        let (a, _) = encoder.forward(&mel, 8).unwrap();
        let (b, _) = encoder.forward(&mel, 8).unwrap();
        assert_eq!(a, b);
    }

    // ---- Shape validation errors ----------------------------------------

    #[test]
    fn new_rejects_empty_stacks() {
        let mut cfg = small_cfg();
        cfg.stacks.clear();
        let mut state = 9u64;
        // Provide any valid stem weight — the empty-stacks check should
        // fire from within `weights.validate`.
        let weights = synth_weights(&small_cfg(), &mut state);
        // Rebuild weights for the empty-stack case:
        let empty_weights = ZipformerWeights {
            stem: weights.stem,
            stacks: Vec::new(),
        };
        let err = ZipformerEncoder::new(cfg, empty_weights).unwrap_err();
        assert!(
            err.to_string().contains("stacks must be non-empty"),
            "got {err}"
        );
    }

    #[test]
    fn new_rejects_d_model_not_divisible_by_n_heads() {
        let mut cfg = small_cfg();
        cfg.d_model = 9;
        cfg.n_heads = 2;
        let dummy = ZipformerWeights {
            stem: ZipformerStemWeights {
                linear_w: vec![],
                linear_b: vec![],
                norm_gamma: None,
                norm_beta: None,
            },
            stacks: Vec::new(),
        };
        let err = ZipformerEncoder::new(cfg, dummy).unwrap_err();
        assert!(
            err.to_string().contains("divisible by n_heads"),
            "got {err}"
        );
    }

    #[test]
    fn new_rejects_even_kernel_size() {
        let mut cfg = small_cfg();
        cfg.stacks[0].kernel_size = 4;
        let dummy = ZipformerWeights {
            stem: ZipformerStemWeights {
                linear_w: vec![],
                linear_b: vec![],
                norm_gamma: None,
                norm_beta: None,
            },
            stacks: Vec::new(),
        };
        let err = ZipformerEncoder::new(cfg, dummy).unwrap_err();
        assert!(
            err.to_string().contains("kernel_size must be odd"),
            "got {err}"
        );
    }

    #[test]
    fn new_rejects_zero_downsampling_factor() {
        let mut cfg = small_cfg();
        cfg.stacks[0].downsampling_factor = 0;
        let dummy = ZipformerWeights {
            stem: ZipformerStemWeights {
                linear_w: vec![],
                linear_b: vec![],
                norm_gamma: None,
                norm_beta: None,
            },
            stacks: Vec::new(),
        };
        let err = ZipformerEncoder::new(cfg, dummy).unwrap_err();
        assert!(
            err.to_string().contains("downsampling_factor must be > 0"),
            "got {err}"
        );
    }

    #[test]
    fn new_rejects_zero_n_layers() {
        let mut cfg = small_cfg();
        cfg.stacks[0].n_layers = 0;
        let dummy = ZipformerWeights {
            stem: ZipformerStemWeights {
                linear_w: vec![],
                linear_b: vec![],
                norm_gamma: None,
                norm_beta: None,
            },
            stacks: Vec::new(),
        };
        let err = ZipformerEncoder::new(cfg, dummy).unwrap_err();
        assert!(
            err.to_string().contains("n_layers must be > 0"),
            "got {err}"
        );
    }

    #[test]
    fn new_rejects_non_finite_bypass_scale() {
        let cfg = small_cfg();
        let mut state = 10u64;
        let mut weights = synth_weights(&cfg, &mut state);
        weights.stacks[0].bypass_scale = f32::NAN;
        let err = ZipformerEncoder::new(cfg, weights).unwrap_err();
        assert!(
            err.to_string().contains("bypass_scale must be finite"),
            "got {err}"
        );
    }

    #[test]
    fn new_rejects_wrong_stack_count() {
        let cfg = small_cfg();
        let mut state = 11u64;
        let mut weights = synth_weights(&cfg, &mut state);
        weights.stacks.truncate(1);
        let err = ZipformerEncoder::new(cfg, weights).unwrap_err();
        assert!(err.to_string().contains("expected 2 stacks"), "got {err}");
    }

    #[test]
    fn new_rejects_wrong_layer_count_in_stack() {
        let cfg = small_cfg();
        let mut state = 12u64;
        let mut weights = synth_weights(&cfg, &mut state);
        weights.stacks[0].layers.pop();
        // Stack 0 has n_layers=1 in small_cfg, so removing one → 0 layers
        // → mismatch.
        let err = ZipformerEncoder::new(cfg, weights).unwrap_err();
        assert!(err.to_string().contains("expected 1 layers"), "got {err}");
    }

    #[test]
    fn new_rejects_wrong_shared_qk_shape() {
        let cfg = small_cfg();
        let mut state = 13u64;
        let mut weights = synth_weights(&cfg, &mut state);
        weights.stacks[0].shared_qk.wq.truncate(4);
        let err = ZipformerEncoder::new(cfg, weights).unwrap_err();
        assert!(err.to_string().contains("shared wq/wk"), "got {err}");
    }

    #[test]
    fn forward_rejects_mismatched_mel_length() {
        let cfg = small_cfg();
        let mut state = 14u64;
        let weights = synth_weights(&cfg, &mut state);
        let encoder = ZipformerEncoder::new(cfg, weights).unwrap();
        let mel = vec![0.0f32; 5]; // Wrong length.
        let err = encoder.forward(&mel, 6).unwrap_err();
        assert!(
            err.to_string().contains("does not match mel_frames"),
            "got {err}"
        );
    }

    #[test]
    fn forward_rejects_zero_mel_frames() {
        let cfg = small_cfg();
        let mut state = 15u64;
        let weights = synth_weights(&cfg, &mut state);
        let encoder = ZipformerEncoder::new(cfg, weights).unwrap();
        let err = encoder.forward(&[], 0).unwrap_err();
        assert!(
            err.to_string().contains("mel_frames must be > 0"),
            "got {err}"
        );
    }

    #[test]
    fn forward_rejects_stack_downsample_to_zero_frames() {
        // Small stem output but a large stack downsample → 0 frames.
        let cfg = ZipformerConfig {
            in_dim: 4,
            d_model: 8,
            n_heads: 2,
            ffn_dim: 16,
            stem: ConvSubsampleKind::Linear,
            position_encoding: PositionEncoding::None,
            stacks: vec![ZipformerStackDesc {
                downsampling_factor: 16,
                n_layers: 1,
                kernel_size: 3,
            }],
        };
        let mut state = 16u64;
        let weights = synth_weights(&cfg, &mut state);
        let encoder = ZipformerEncoder::new(cfg.clone(), weights).unwrap();
        // 3 stem-output frames but downsampling factor 16 → 0 down frames.
        let mel = synth_vec(&mut state, 3 * cfg.in_dim as usize, 1.0);
        let err = encoder.forward(&mel, 3).unwrap_err();
        assert!(
            err.to_string().contains("downsampling to zero frames"),
            "got {err}"
        );
    }

    // ---- Small helper pins ----------------------------------------------

    #[test]
    fn downsample_mean_of_stride_two_is_pairwise_mean() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (out, t_out) = downsample_mean(&x, 4, 2, 2);
        assert_eq!(t_out, 2);
        // rows [1,2],[3,4] → mean [2,3]; rows [5,6],[7,8] → mean [6,7].
        assert_eq!(out, vec![2.0, 3.0, 6.0, 7.0]);
    }

    #[test]
    fn downsample_mean_drops_trailing_partial_block() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (out, t_out) = downsample_mean(&x, 3, 2, 2);
        // 3 rows / factor 2 → 1 output row; trailing single row dropped.
        assert_eq!(t_out, 1);
        // Row 0 = [1,2], row 1 = [3,4] → mean = [2,3].
        assert_eq!(out, vec![2.0, 3.0]);
    }

    #[test]
    fn downsample_mean_factor_one_is_identity() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (out, t_out) = downsample_mean(&x, 3, 2, 1);
        assert_eq!(t_out, 3);
        assert_eq!(out, x);
    }

    #[test]
    fn upsample_repeat_expands_block_repeat() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let out = upsample_repeat(&x, 2, 2, 2, 4);
        // Each input row repeated twice: [[1,2],[1,2],[3,4],[3,4]]
        assert_eq!(out, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    }

    #[test]
    fn upsample_repeat_fills_tail_from_last_row() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        // 2 input rows, factor 2, t_out = 5 (tail slot after 2*2 = 4).
        let out = upsample_repeat(&x, 2, 2, 2, 5);
        assert_eq!(out.len(), 10);
        // Last slot must be copied from the last input row.
        assert_eq!(&out[8..10], &[3.0, 4.0]);
    }

    #[test]
    fn config_head_dim_is_d_model_over_n_heads() {
        let cfg = small_cfg();
        assert_eq!(cfg.head_dim(), 4); // 8 / 2 = 4
    }
}
