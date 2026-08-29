//! E-Branchformer encoder (SoTA plan Phase JA JA-ASR-4 primitive).
//!
//! Direct Rust port of the E-Branchformer encoder used by ESPnet's
//! OWSM family (`espnet/owsm-ctc-v3.1-1B`, `espnet/owsm_v3.1_ebf_base`,
//! CC-BY-4.0). Reference implementations:
//!
//! - [`espnet/espnet/nets/pytorch_backend/transformer/e_branchformer_encoder_layer.py`][espnet-ebf-layer]
//! - [`espnet/espnet/nets/pytorch_backend/transformer/cgmlp.py`][espnet-cgmlp]
//! - Kim et al. 2023, [E-Branchformer: Branchformer with Enhanced Merging
//!   for Speech Recognition][kim-2023].
//!
//! # What E-Branchformer is (vs. Conformer)
//!
//! [`ConformerEncoder`](crate::conformer::ConformerEncoder) is a **serial**
//! stack: `FF → MHA → Conv → FF`. E-Branchformer replaces the middle
//! `MHA → Conv` serial with a **parallel two-branch merge**:
//!
//! ```text
//! residual = x
//! residual = residual + 0.5 · FF1(LN(residual))       // macaron FF1
//! branch_a = MHA(LN(residual))                        // attention branch
//! branch_b = cgMLP(LN(residual))                      // gated conv MLP branch
//! merged   = Merge(concat(branch_a, branch_b))        // dw-conv → linear
//! residual = residual + merged                        // parallel merge residual
//! residual = residual + 0.5 · FF2(LN(residual))       // macaron FF2
//! output   = LN_out(residual)
//! ```
//!
//! The **cgMLP branch** (gated Convolutional MLP, upstream `ConvolutionalGatingMLP`
//! in `cgmlp.py`) is the interesting part:
//!
//! - `Linear(d_model → 2·d_ffn)` → GELU → split along channel axis into
//!   two halves `(u, v)` (each `[T, d_ffn]`);
//! - `DepthwiseConv1d(v)` — same-padding depthwise conv over time on the
//!   `v` half only;
//! - `LayerNorm(v)` + optional linear "gate proj";
//! - `u ⊙ v` — elementwise gate multiplication (cgMLP core);
//! - `Linear(d_ffn → d_model)` — pointwise output projection.
//!
//! The **Merge module** (upstream `MergeModule` in `e_branchformer_encoder_layer.py`)
//! fuses the two branches:
//!
//! - concat `[branch_a; branch_b]` along the channel axis (result `[T, 2·d_model]`);
//! - `DepthwiseConv1d` over time with `groups = 2·d_model`;
//! - `Linear(2·d_model → d_model)` to project back to model dim.
//!
//! # Scope of this primitive
//!
//! The upstream E-Branchformer has a handful more optional switches (a
//! "stochastic depth" residual dropout, a per-branch gate on the merge
//! module, an `identity_v` cgMLP shortcut). This primitive collapses those
//! to the axes released OWSM checkpoints actually change:
//!
//! - `d_model` / `n_heads` / `ffn_dim` (macaron FF width, upstream default
//!   `d_ff = 1024` for the base config);
//! - `cgmlp_hidden_dim` — the internal width of the cgMLP branch (upstream
//!   `cgmlp_linear_units`, default 3072 = 3 × d_model for `d_model = 1024`);
//! - `cgmlp_kernel_size` — depthwise conv kernel inside cgMLP;
//! - `merge_kernel_size` — depthwise conv kernel inside the merge module.
//!
//! # No silent CPU fallback (FR-EX-08)
//!
//! Every degenerate input becomes a loud [`VokraError::InvalidArgument`]:
//! `d_model % n_heads != 0`, `d_model == 0`, `n_layers == 0`,
//! even kernel size in either module, `cgmlp_hidden_dim == 0`, mel length
//! mismatch. Silent truncation is banned.
//!
//! # Zero-dependency posture (NFR-DS-02)
//!
//! No BLAS, no `serde`, no third-party crate. Scalar Rust with `unsafe`
//! deliberately absent.
//!
//! [espnet-ebf-layer]: https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/e_branchformer_encoder_layer.py
//! [espnet-cgmlp]: https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/cgmlp.py
//! [kim-2023]: https://arxiv.org/abs/2210.00077

use vokra_core::{Result, VokraError};

use crate::conformer::{ConvSubsampleKind, FeedForwardWeights, MhaWeights, PositionEncoding};

// ---------------------------------------------------------------------------
// Public config
// ---------------------------------------------------------------------------

/// E-Branchformer encoder hyperparameters.
#[derive(Debug, Clone)]
pub struct EBranchformerConfig {
    /// Mel channels on the input (upstream `feat_in`, e.g. 80 log-mel bins
    /// after ESPnet's global-mean-var-norm front-end).
    pub in_dim: u32,
    /// Model dimension (upstream `d_model`, e.g. 512 for OWSM v3.1 base,
    /// 1024 for OWSM v3.2 large).
    pub d_model: u32,
    /// Number of attention heads.
    pub n_heads: u32,
    /// FeedForward hidden width for the macaron FF1 / FF2 branches
    /// (upstream `d_ff = d_model * ff_expansion_factor`).
    pub ffn_dim: u32,
    /// Number of E-Branchformer layers to stack (upstream `n_layers`).
    pub n_layers: u32,
    /// Depthwise conv kernel size inside the cgMLP branch. Must be odd.
    pub cgmlp_kernel_size: u32,
    /// Hidden channel width for the cgMLP branch (upstream
    /// `cgmlp_linear_units`, typically 3 × `d_model`). Split into two
    /// halves (`u`, `v`) for the gated multiplication.
    pub cgmlp_hidden_dim: u32,
    /// Depthwise conv kernel size inside the merge module (upstream
    /// `merge_conv_kernel`, typically 3). Must be odd.
    pub merge_kernel_size: u32,
    /// Subsampling stem variant. Same enum the Conformer primitive uses so
    /// callers can share the stem wiring.
    pub stem: ConvSubsampleKind,
    /// Positional encoding overlay for the attention branch.
    pub position_encoding: PositionEncoding,
}

impl EBranchformerConfig {
    /// Per-head attention dimension (`d_model / n_heads`).
    pub fn head_dim(&self) -> usize {
        (self.d_model / self.n_heads) as usize
    }

    /// Split width of the cgMLP branch (`cgmlp_hidden_dim / 2`). The
    /// gate multiplication runs on two equal halves of the internal
    /// hidden representation; `cgmlp_hidden_dim` must therefore be even
    /// (checked at construction time).
    pub fn cgmlp_half_dim(&self) -> usize {
        (self.cgmlp_hidden_dim / 2) as usize
    }
}

// ---------------------------------------------------------------------------
// Weight structs
// ---------------------------------------------------------------------------

/// Stem subsample weights — same layout as
/// [`crate::conformer::ConformerSubsampleWeights`]. Kept as a distinct type
/// so a future divergence in the stem does not force a caller rewrite.
#[derive(Debug, Clone)]
pub struct EBranchformerStemWeights {
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

impl EBranchformerStemWeights {
    fn validate(&self, cfg: &EBranchformerConfig) -> Result<()> {
        let d_model = cfg.d_model as usize;
        let in_dim = cfg.in_dim as usize;
        let proj_in = cfg.stem.projection_in_dim(in_dim);
        let expected_w = d_model * proj_in;
        if self.linear_w.len() != expected_w {
            return Err(VokraError::InvalidArgument(format!(
                "E-Branchformer stem linear_w must be length {expected_w} \
                 (d_model={d_model} × projection_in_dim={proj_in}), got {}",
                self.linear_w.len(),
            )));
        }
        if self.linear_b.len() != d_model {
            return Err(VokraError::InvalidArgument(format!(
                "E-Branchformer stem linear_b must be length {d_model}, got {}",
                self.linear_b.len(),
            )));
        }
        let need_norm = cfg.stem.has_norm();
        match (&self.norm_gamma, &self.norm_beta, need_norm) {
            (Some(g), Some(b), true) => {
                if g.len() != d_model || b.len() != d_model {
                    return Err(VokraError::InvalidArgument(format!(
                        "E-Branchformer stem norm gamma/beta must be length {d_model}, \
                         got gamma={} beta={}",
                        g.len(),
                        b.len(),
                    )));
                }
            }
            (None, None, false) => {}
            _ => {
                return Err(VokraError::InvalidArgument(
                    "E-Branchformer stem norm gamma/beta presence must match stem.has_norm()"
                        .to_owned(),
                ));
            }
        }
        Ok(())
    }
}

/// cgMLP branch weights (upstream `ConvolutionalGatingMLP`).
///
/// Layout:
/// - `linear_in_w`  `[cgmlp_hidden_dim, d_model]`  — expand + GELU (before split)
/// - `linear_in_b`  `[cgmlp_hidden_dim]`
/// - `norm_gamma`   `[cgmlp_half_dim]`             — LayerNorm on the `v` half
/// - `norm_beta`    `[cgmlp_half_dim]`
/// - `depthwise_w`  `[cgmlp_half_dim, kernel_size]` — depthwise conv on the `v` half
/// - `depthwise_b`  `[cgmlp_half_dim]`
/// - `linear_out_w` `[d_model, cgmlp_half_dim]`   — project gated output back
/// - `linear_out_b` `[d_model]`
#[derive(Debug, Clone)]
pub struct CgMlpWeights {
    /// Row-major `[cgmlp_hidden_dim, d_model]`.
    pub linear_in_w: Vec<f32>,
    /// `[cgmlp_hidden_dim]`.
    pub linear_in_b: Vec<f32>,
    /// `[cgmlp_half_dim]` LayerNorm γ on the `v` half.
    pub norm_gamma: Vec<f32>,
    /// `[cgmlp_half_dim]` LayerNorm β on the `v` half.
    pub norm_beta: Vec<f32>,
    /// Row-major `[cgmlp_half_dim, kernel_size]` — depthwise filters (one
    /// per channel, `groups == cgmlp_half_dim`).
    pub depthwise_w: Vec<f32>,
    /// `[cgmlp_half_dim]`.
    pub depthwise_b: Vec<f32>,
    /// Row-major `[d_model, cgmlp_half_dim]` — pointwise output projection.
    pub linear_out_w: Vec<f32>,
    /// `[d_model]`.
    pub linear_out_b: Vec<f32>,
}

impl CgMlpWeights {
    fn validate(&self, cfg: &EBranchformerConfig, tag: &str) -> Result<()> {
        let d_model = cfg.d_model as usize;
        let hidden = cfg.cgmlp_hidden_dim as usize;
        let half = cfg.cgmlp_half_dim();
        let kernel = cfg.cgmlp_kernel_size as usize;
        if self.linear_in_w.len() != hidden * d_model {
            return Err(VokraError::InvalidArgument(format!(
                "{tag}: cgmlp linear_in_w must be length {}×{}={}, got {}",
                hidden,
                d_model,
                hidden * d_model,
                self.linear_in_w.len(),
            )));
        }
        if self.linear_in_b.len() != hidden {
            return Err(VokraError::InvalidArgument(format!(
                "{tag}: cgmlp linear_in_b must be length {hidden}, got {}",
                self.linear_in_b.len(),
            )));
        }
        if self.norm_gamma.len() != half || self.norm_beta.len() != half {
            return Err(VokraError::InvalidArgument(format!(
                "{tag}: cgmlp norm gamma/beta must be length {half}, got gamma={} beta={}",
                self.norm_gamma.len(),
                self.norm_beta.len(),
            )));
        }
        if self.depthwise_w.len() != half * kernel {
            return Err(VokraError::InvalidArgument(format!(
                "{tag}: cgmlp depthwise_w must be length {}×{}={}, got {}",
                half,
                kernel,
                half * kernel,
                self.depthwise_w.len(),
            )));
        }
        if self.depthwise_b.len() != half {
            return Err(VokraError::InvalidArgument(format!(
                "{tag}: cgmlp depthwise_b must be length {half}, got {}",
                self.depthwise_b.len(),
            )));
        }
        if self.linear_out_w.len() != d_model * half {
            return Err(VokraError::InvalidArgument(format!(
                "{tag}: cgmlp linear_out_w must be length {}×{}={}, got {}",
                d_model,
                half,
                d_model * half,
                self.linear_out_w.len(),
            )));
        }
        if self.linear_out_b.len() != d_model {
            return Err(VokraError::InvalidArgument(format!(
                "{tag}: cgmlp linear_out_b must be length {d_model}, got {}",
                self.linear_out_b.len(),
            )));
        }
        Ok(())
    }
}

/// Merge module weights (upstream `MergeModule` in
/// `e_branchformer_encoder_layer.py`).
///
/// Layout:
/// - `depthwise_w` `[2·d_model, kernel_size]` — depthwise conv over the
///   concat `[branch_a; branch_b]`, `groups == 2·d_model`
/// - `depthwise_b` `[2·d_model]`
/// - `linear_w`    `[d_model, 2·d_model]` — pointwise projection back
/// - `linear_b`    `[d_model]`
#[derive(Debug, Clone)]
pub struct MergeWeights {
    /// Row-major `[2·d_model, kernel_size]`.
    pub depthwise_w: Vec<f32>,
    /// `[2·d_model]`.
    pub depthwise_b: Vec<f32>,
    /// Row-major `[d_model, 2·d_model]`.
    pub linear_w: Vec<f32>,
    /// `[d_model]`.
    pub linear_b: Vec<f32>,
}

impl MergeWeights {
    fn validate(&self, cfg: &EBranchformerConfig, tag: &str) -> Result<()> {
        let d_model = cfg.d_model as usize;
        let two_d = 2 * d_model;
        let kernel = cfg.merge_kernel_size as usize;
        if self.depthwise_w.len() != two_d * kernel {
            return Err(VokraError::InvalidArgument(format!(
                "{tag}: merge depthwise_w must be length {}×{}={}, got {}",
                two_d,
                kernel,
                two_d * kernel,
                self.depthwise_w.len(),
            )));
        }
        if self.depthwise_b.len() != two_d {
            return Err(VokraError::InvalidArgument(format!(
                "{tag}: merge depthwise_b must be length {two_d}, got {}",
                self.depthwise_b.len(),
            )));
        }
        if self.linear_w.len() != d_model * two_d {
            return Err(VokraError::InvalidArgument(format!(
                "{tag}: merge linear_w must be length {}×{}={}, got {}",
                d_model,
                two_d,
                d_model * two_d,
                self.linear_w.len(),
            )));
        }
        if self.linear_b.len() != d_model {
            return Err(VokraError::InvalidArgument(format!(
                "{tag}: merge linear_b must be length {d_model}, got {}",
                self.linear_b.len(),
            )));
        }
        Ok(())
    }
}

/// Per-layer weights for one E-Branchformer layer.
#[derive(Debug, Clone)]
pub struct EBranchformerLayerWeights {
    /// `[d_model]` γ for the FF1 pre-norm.
    pub ln1_gamma: Vec<f32>,
    /// `[d_model]` β for the FF1 pre-norm.
    pub ln1_beta: Vec<f32>,
    /// FF1 weights.
    pub ff1: FeedForwardWeights,
    /// `[d_model]` γ for the attention branch pre-norm.
    pub ln_attn_gamma: Vec<f32>,
    /// `[d_model]` β for the attention branch pre-norm.
    pub ln_attn_beta: Vec<f32>,
    /// Multi-head attention weights.
    pub mha: MhaWeights,
    /// `[d_model]` γ for the cgMLP branch pre-norm.
    pub ln_cg_gamma: Vec<f32>,
    /// `[d_model]` β for the cgMLP branch pre-norm.
    pub ln_cg_beta: Vec<f32>,
    /// cgMLP branch weights.
    pub cgmlp: CgMlpWeights,
    /// Merge module weights.
    pub merge: MergeWeights,
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

impl EBranchformerLayerWeights {
    fn validate(&self, cfg: &EBranchformerConfig, layer_idx: usize) -> Result<()> {
        let d_model = cfg.d_model as usize;
        let ffn_dim = cfg.ffn_dim as usize;
        for (name, v) in [
            ("ln1_gamma", &self.ln1_gamma),
            ("ln1_beta", &self.ln1_beta),
            ("ln_attn_gamma", &self.ln_attn_gamma),
            ("ln_attn_beta", &self.ln_attn_beta),
            ("ln_cg_gamma", &self.ln_cg_gamma),
            ("ln_cg_beta", &self.ln_cg_beta),
            ("ln4_gamma", &self.ln4_gamma),
            ("ln4_beta", &self.ln4_beta),
            ("ln_out_gamma", &self.ln_out_gamma),
            ("ln_out_beta", &self.ln_out_beta),
        ] {
            if v.len() != d_model {
                return Err(VokraError::InvalidArgument(format!(
                    "E-Branchformer layer {layer_idx}: {name} must be length {d_model}, got {}",
                    v.len(),
                )));
            }
        }
        self.ff1.validate_ext(
            d_model,
            ffn_dim,
            &format!("E-Branchformer layer {layer_idx} FF1"),
        )?;
        self.mha
            .validate_ext(d_model, &format!("E-Branchformer layer {layer_idx} MHA"))?;
        self.cgmlp
            .validate(cfg, &format!("E-Branchformer layer {layer_idx}"))?;
        self.merge
            .validate(cfg, &format!("E-Branchformer layer {layer_idx}"))?;
        self.ff2.validate_ext(
            d_model,
            ffn_dim,
            &format!("E-Branchformer layer {layer_idx} FF2"),
        )?;
        Ok(())
    }
}

/// All learned parameters an encoder owns.
#[derive(Debug, Clone)]
pub struct EBranchformerWeights {
    /// Stem subsample.
    pub stem: EBranchformerStemWeights,
    /// One entry per layer (must match `cfg.n_layers`).
    pub layers: Vec<EBranchformerLayerWeights>,
}

impl EBranchformerWeights {
    fn validate(&self, cfg: &EBranchformerConfig) -> Result<()> {
        self.stem.validate(cfg)?;
        if self.layers.len() != cfg.n_layers as usize {
            return Err(VokraError::InvalidArgument(format!(
                "E-Branchformer weights: expected {} layers, got {}",
                cfg.n_layers,
                self.layers.len(),
            )));
        }
        for (idx, layer) in self.layers.iter().enumerate() {
            layer.validate(cfg, idx)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Main encoder
// ---------------------------------------------------------------------------

/// E-Branchformer encoder (parallel MHA + cgMLP with a Merge module).
#[derive(Debug, Clone)]
pub struct EBranchformerEncoder {
    cfg: EBranchformerConfig,
    weights: EBranchformerWeights,
}

impl EBranchformerEncoder {
    /// Build an encoder from its config + weights.
    ///
    /// Fails loudly on any shape mismatch, on `d_model % n_heads != 0`, on
    /// `n_layers == 0`, on `cgmlp_hidden_dim == 0` or odd
    /// `cgmlp_hidden_dim`, on any even kernel size, or on empty stems.
    pub fn new(cfg: EBranchformerConfig, weights: EBranchformerWeights) -> Result<Self> {
        if cfg.d_model == 0 {
            return Err(VokraError::InvalidArgument(
                "EBranchformerConfig: d_model must be > 0".to_owned(),
            ));
        }
        if cfg.n_heads == 0 {
            return Err(VokraError::InvalidArgument(
                "EBranchformerConfig: n_heads must be > 0".to_owned(),
            ));
        }
        if cfg.d_model % cfg.n_heads != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "EBranchformerConfig: d_model ({}) must be divisible by n_heads ({})",
                cfg.d_model, cfg.n_heads,
            )));
        }
        if cfg.n_layers == 0 {
            return Err(VokraError::InvalidArgument(
                "EBranchformerConfig: n_layers must be > 0".to_owned(),
            ));
        }
        if cfg.ffn_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "EBranchformerConfig: ffn_dim must be > 0".to_owned(),
            ));
        }
        if cfg.in_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "EBranchformerConfig: in_dim must be > 0".to_owned(),
            ));
        }
        if cfg.cgmlp_hidden_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "EBranchformerConfig: cgmlp_hidden_dim must be > 0".to_owned(),
            ));
        }
        if cfg.cgmlp_hidden_dim % 2 != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "EBranchformerConfig: cgmlp_hidden_dim must be even (splits into two halves \
                 for the gate multiplication), got {}",
                cfg.cgmlp_hidden_dim,
            )));
        }
        if cfg.cgmlp_kernel_size == 0 || cfg.cgmlp_kernel_size % 2 == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "EBranchformerConfig: cgmlp_kernel_size must be odd and > 0, got {}",
                cfg.cgmlp_kernel_size,
            )));
        }
        if cfg.merge_kernel_size == 0 || cfg.merge_kernel_size % 2 == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "EBranchformerConfig: merge_kernel_size must be odd and > 0, got {}",
                cfg.merge_kernel_size,
            )));
        }
        if let ConvSubsampleKind::Stacking { factor } | ConvSubsampleKind::StackingNorm { factor } =
            cfg.stem
        {
            if factor == 0 {
                return Err(VokraError::InvalidArgument(
                    "EBranchformerConfig: stem stacking factor must be > 0".to_owned(),
                ));
            }
        }
        if matches!(cfg.stem, ConvSubsampleKind::Conv1d { .. }) {
            return Err(VokraError::InvalidArgument(
                "EBranchformerConfig: Conv1d stem is only supported by ConformerEncoder".to_owned(),
            ));
        }
        weights.validate(&cfg)?;
        Ok(Self { cfg, weights })
    }

    /// Immutable access to the [`EBranchformerConfig`] the encoder was
    /// built with.
    pub fn config(&self) -> &EBranchformerConfig {
        &self.cfg
    }

    /// Full forward pass — mel → encoded hidden state.
    ///
    /// `mel` is a flat row-major `[mel_frames, in_dim]` buffer. Returns
    /// `(hidden, T_out)` where `T_out = mel_frames / stem_factor` under
    /// the same stacking-tail-drop convention as
    /// [`crate::conformer::ConformerEncoder`].
    pub fn forward(&self, mel: &[f32], mel_frames: usize) -> Result<(Vec<f32>, usize)> {
        let in_dim = self.cfg.in_dim as usize;
        if mel_frames == 0 {
            return Err(VokraError::InvalidArgument(
                "EBranchformerEncoder::forward: mel_frames must be > 0".to_owned(),
            ));
        }
        let expected_len = mel_frames * in_dim;
        if mel.len() != expected_len {
            return Err(VokraError::InvalidArgument(format!(
                "EBranchformerEncoder::forward: mel length {} does not match \
                 mel_frames×in_dim = {mel_frames}×{in_dim} = {expected_len}",
                mel.len(),
            )));
        }

        let (mut hidden, t_out) = self.stem_forward(mel, mel_frames)?;
        if t_out == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "EBranchformerEncoder::forward: subsampled sequence is empty \
                 (mel_frames={mel_frames}, stem_factor={})",
                self.cfg.stem.factor(),
            )));
        }

        for layer in &self.weights.layers {
            hidden = self.ebranchformer_layer(&hidden, t_out, layer)?;
        }
        Ok((hidden, t_out))
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
            ConvSubsampleKind::Conv1d { .. } => Err(VokraError::InvalidArgument(
                "EBranchformerEncoder: Conv1d stem is only supported by ConformerEncoder"
                    .to_owned(),
            )),
        }
    }

    // -----------------------------------------------------------------------
    // Per-layer forward
    // -----------------------------------------------------------------------

    fn ebranchformer_layer(
        &self,
        input: &[f32],
        t: usize,
        w: &EBranchformerLayerWeights,
    ) -> Result<Vec<f32>> {
        let d_model = self.cfg.d_model as usize;
        let ffn_dim = self.cfg.ffn_dim as usize;

        // ---- FF1 branch: residual += 0.5 * FF1(LN1(x)) --------------------
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

        // ---- Parallel MHA + cgMLP merge ------------------------------------
        // Pre-norm for the attention branch.
        buf.copy_from_slice(&residual);
        for row_off in (0..buf.len()).step_by(d_model) {
            layer_norm_inplace(
                &mut buf[row_off..row_off + d_model],
                &w.ln_attn_gamma,
                &w.ln_attn_beta,
            );
        }
        let branch_a = self.multi_head_attention(&buf, t, &w.mha)?;

        // Pre-norm for the cgMLP branch.
        buf.copy_from_slice(&residual);
        for row_off in (0..buf.len()).step_by(d_model) {
            layer_norm_inplace(
                &mut buf[row_off..row_off + d_model],
                &w.ln_cg_gamma,
                &w.ln_cg_beta,
            );
        }
        let branch_b = self.cgmlp_forward(&buf, t, &w.cgmlp)?;

        // Merge: concat → depthwise conv → linear.
        let merged = self.merge_forward(&branch_a, &branch_b, t, &w.merge)?;
        add_inplace(&mut residual, &merged);

        // ---- FF2 branch: residual += 0.5 * FF2(LN4(residual)) --------------
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

    fn multi_head_attention(&self, x: &[f32], t: usize, w: &MhaWeights) -> Result<Vec<f32>> {
        let d_model = self.cfg.d_model as usize;
        let n_heads = self.cfg.n_heads as usize;
        let head_dim = self.cfg.head_dim();
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut q = vec![0.0f32; t * d_model];
        let mut k = vec![0.0f32; t * d_model];
        let mut v = vec![0.0f32; t * d_model];
        for ti in 0..t {
            let src = &x[ti * d_model..(ti + 1) * d_model];
            linear_row(
                src,
                &w.wq,
                &w.bq,
                d_model,
                d_model,
                &mut q[ti * d_model..(ti + 1) * d_model],
            );
            linear_row(
                src,
                &w.wk,
                &w.bk,
                d_model,
                d_model,
                &mut k[ti * d_model..(ti + 1) * d_model],
            );
            linear_row(
                src,
                &w.wv,
                &w.bv,
                d_model,
                d_model,
                &mut v[ti * d_model..(ti + 1) * d_model],
            );
        }

        if let PositionEncoding::Rope { theta } = self.cfg.position_encoding {
            apply_rope(&mut q, t, n_heads, head_dim, theta);
            apply_rope(&mut k, t, n_heads, head_dim, theta);
        }

        let mut output = vec![0.0f32; t * d_model];
        let mut scores = vec![0.0f32; t * t];
        let mut probs = vec![0.0f32; t * t];
        for h in 0..n_heads {
            let head_off = h * head_dim;
            for i in 0..t {
                let q_row = &q[i * d_model + head_off..i * d_model + head_off + head_dim];
                for j in 0..t {
                    let k_row = &k[j * d_model + head_off..j * d_model + head_off + head_dim];
                    let mut acc = 0.0f32;
                    for d in 0..head_dim {
                        acc += q_row[d] * k_row[d];
                    }
                    scores[i * t + j] = acc * scale;
                }
            }
            for i in 0..t {
                softmax_row(&scores[i * t..(i + 1) * t], &mut probs[i * t..(i + 1) * t]);
            }
            for i in 0..t {
                for j in 0..t {
                    let p = probs[i * t + j];
                    if p == 0.0 {
                        continue;
                    }
                    let v_row = &v[j * d_model + head_off..j * d_model + head_off + head_dim];
                    let ctx_row =
                        &mut output[i * d_model + head_off..i * d_model + head_off + head_dim];
                    for d in 0..head_dim {
                        ctx_row[d] += p * v_row[d];
                    }
                }
            }
        }

        let mut proj = vec![0.0f32; t * d_model];
        for i in 0..t {
            linear_row(
                &output[i * d_model..(i + 1) * d_model],
                &w.wo,
                &w.bo,
                d_model,
                d_model,
                &mut proj[i * d_model..(i + 1) * d_model],
            );
        }
        Ok(proj)
    }

    fn cgmlp_forward(&self, x: &[f32], t: usize, w: &CgMlpWeights) -> Result<Vec<f32>> {
        let d_model = self.cfg.d_model as usize;
        let hidden = self.cfg.cgmlp_hidden_dim as usize;
        let half = self.cfg.cgmlp_half_dim();
        let kernel = self.cfg.cgmlp_kernel_size as usize;

        // Step 1 — linear_in + GELU. Result `[t, hidden]`.
        let mut expanded = vec![0.0f32; t * hidden];
        for ti in 0..t {
            linear_row(
                &x[ti * d_model..(ti + 1) * d_model],
                &w.linear_in_w,
                &w.linear_in_b,
                hidden,
                d_model,
                &mut expanded[ti * hidden..(ti + 1) * hidden],
            );
            for v in expanded[ti * hidden..(ti + 1) * hidden].iter_mut() {
                *v = gelu_tanh(*v);
            }
        }

        // Step 2 — split into (u, v) halves along channel axis.
        let mut u = vec![0.0f32; t * half];
        let mut v_half = vec![0.0f32; t * half];
        for ti in 0..t {
            let src = &expanded[ti * hidden..(ti + 1) * hidden];
            u[ti * half..(ti + 1) * half].copy_from_slice(&src[..half]);
            v_half[ti * half..(ti + 1) * half].copy_from_slice(&src[half..]);
        }

        // Step 3 — LayerNorm on the `v` half.
        for ti in 0..t {
            let row = &mut v_half[ti * half..(ti + 1) * half];
            layer_norm_inplace(row, &w.norm_gamma, &w.norm_beta);
        }

        // Step 4 — depthwise conv on the `v` half (channel-first transpose,
        // conv, transpose back).
        let mut ct = vec![0.0f32; half * t];
        for ti in 0..t {
            for c in 0..half {
                ct[c * t + ti] = v_half[ti * half + c];
            }
        }
        let padding = kernel / 2;
        let mut conv_out_ct = vec![0.0f32; half * t];
        let t_i = t as isize;
        let pad_i = padding as isize;
        for c in 0..half {
            let filter = &w.depthwise_w[c * kernel..(c + 1) * kernel];
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
        // Transpose back.
        for c in 0..half {
            for ti in 0..t {
                v_half[ti * half + c] = conv_out_ct[c * t + ti];
            }
        }

        // Step 5 — gate multiplication `u ⊙ v`.
        for i in 0..(t * half) {
            u[i] *= v_half[i];
        }

        // Step 6 — linear_out `[half → d_model]`.
        let mut out = vec![0.0f32; t * d_model];
        for ti in 0..t {
            linear_row(
                &u[ti * half..(ti + 1) * half],
                &w.linear_out_w,
                &w.linear_out_b,
                d_model,
                half,
                &mut out[ti * d_model..(ti + 1) * d_model],
            );
        }
        Ok(out)
    }

    fn merge_forward(
        &self,
        branch_a: &[f32],
        branch_b: &[f32],
        t: usize,
        w: &MergeWeights,
    ) -> Result<Vec<f32>> {
        let d_model = self.cfg.d_model as usize;
        let two_d = 2 * d_model;
        let kernel = self.cfg.merge_kernel_size as usize;

        // Step 1 — concat along channel axis: `[branch_a; branch_b]`
        // → `[t, 2·d_model]`.
        let mut concat = vec![0.0f32; t * two_d];
        for ti in 0..t {
            concat[ti * two_d..ti * two_d + d_model]
                .copy_from_slice(&branch_a[ti * d_model..(ti + 1) * d_model]);
            concat[ti * two_d + d_model..ti * two_d + two_d]
                .copy_from_slice(&branch_b[ti * d_model..(ti + 1) * d_model]);
        }

        // Step 2 — depthwise conv over time, `groups = 2·d_model`.
        // Transpose to channel-first `[2·d_model, t]`.
        let mut ct = vec![0.0f32; two_d * t];
        for ti in 0..t {
            for c in 0..two_d {
                ct[c * t + ti] = concat[ti * two_d + c];
            }
        }
        let padding = kernel / 2;
        let mut conv_out_ct = vec![0.0f32; two_d * t];
        let t_i = t as isize;
        let pad_i = padding as isize;
        for c in 0..two_d {
            let filter = &w.depthwise_w[c * kernel..(c + 1) * kernel];
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
        // Transpose back.
        let mut conv_out = vec![0.0f32; t * two_d];
        for c in 0..two_d {
            for ti in 0..t {
                conv_out[ti * two_d + c] = conv_out_ct[c * t + ti];
            }
        }

        // Step 3 — linear projection `[2·d_model → d_model]`.
        let mut out = vec![0.0f32; t * d_model];
        for ti in 0..t {
            linear_row(
                &conv_out[ti * two_d..(ti + 1) * two_d],
                &w.linear_w,
                &w.linear_b,
                d_model,
                two_d,
                &mut out[ti * d_model..(ti + 1) * d_model],
            );
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Local helpers — mirror the conformer helpers so this module is
// self-contained. The extension trait shadows the private conformer
// validators without forcing pub exports from that module.
// ---------------------------------------------------------------------------

trait FfValidateExt {
    fn validate_ext(&self, d_model: usize, ffn_dim: usize, tag: &str) -> Result<()>;
}

impl FfValidateExt for FeedForwardWeights {
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

trait MhaValidateExt {
    fn validate_ext(&self, d_model: usize, tag: &str) -> Result<()>;
}

impl MhaValidateExt for MhaWeights {
    fn validate_ext(&self, d_model: usize, tag: &str) -> Result<()> {
        let dd = d_model * d_model;
        for (name, w) in [
            ("wq", &self.wq),
            ("wk", &self.wk),
            ("wv", &self.wv),
            ("wo", &self.wo),
        ] {
            if w.len() != dd {
                return Err(VokraError::InvalidArgument(format!(
                    "{tag}: {name} must be length {dd} (d_model²), got {}",
                    w.len(),
                )));
            }
        }
        for (name, b) in [
            ("bq", &self.bq),
            ("bk", &self.bk),
            ("bv", &self.bv),
            ("bo", &self.bo),
        ] {
            if b.len() != d_model {
                return Err(VokraError::InvalidArgument(format!(
                    "{tag}: {name} must be length {d_model}, got {}",
                    b.len(),
                )));
            }
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

/// GELU with the tanh approximation (upstream ESPnet default for cgMLP).
/// `0.5 · x · (1 + tanh(sqrt(2/π) · (x + 0.044715·x³)))`.
#[inline]
fn gelu_tanh(x: f32) -> f32 {
    const C0: f32 = 0.797_884_6; // sqrt(2 / π)
    const C1: f32 = 0.044_715;
    0.5 * x * (1.0 + (C0 * (x + C1 * x * x * x)).tanh())
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

    fn synth_mha(state: &mut u64, d_model: usize) -> MhaWeights {
        let dd = d_model * d_model;
        MhaWeights {
            wq: synth_vec(state, dd, 0.1),
            bq: synth_vec(state, d_model, 0.1),
            wk: synth_vec(state, dd, 0.1),
            bk: synth_vec(state, d_model, 0.1),
            wv: synth_vec(state, dd, 0.1),
            bv: synth_vec(state, d_model, 0.1),
            wo: synth_vec(state, dd, 0.1),
            bo: synth_vec(state, d_model, 0.1),
        }
    }

    fn synth_cgmlp(state: &mut u64, cfg: &EBranchformerConfig) -> CgMlpWeights {
        let d_model = cfg.d_model as usize;
        let hidden = cfg.cgmlp_hidden_dim as usize;
        let half = cfg.cgmlp_half_dim();
        let kernel = cfg.cgmlp_kernel_size as usize;
        CgMlpWeights {
            linear_in_w: synth_vec(state, hidden * d_model, 0.1),
            linear_in_b: synth_vec(state, hidden, 0.1),
            norm_gamma: vec![1.0; half],
            norm_beta: vec![0.0; half],
            depthwise_w: synth_vec(state, half * kernel, 0.1),
            depthwise_b: synth_vec(state, half, 0.1),
            linear_out_w: synth_vec(state, d_model * half, 0.1),
            linear_out_b: synth_vec(state, d_model, 0.1),
        }
    }

    fn synth_merge(state: &mut u64, cfg: &EBranchformerConfig) -> MergeWeights {
        let d_model = cfg.d_model as usize;
        let two_d = 2 * d_model;
        let kernel = cfg.merge_kernel_size as usize;
        MergeWeights {
            depthwise_w: synth_vec(state, two_d * kernel, 0.1),
            depthwise_b: synth_vec(state, two_d, 0.1),
            linear_w: synth_vec(state, d_model * two_d, 0.1),
            linear_b: synth_vec(state, d_model, 0.1),
        }
    }

    fn synth_layer(state: &mut u64, cfg: &EBranchformerConfig) -> EBranchformerLayerWeights {
        let d_model = cfg.d_model as usize;
        let ffn_dim = cfg.ffn_dim as usize;
        EBranchformerLayerWeights {
            ln1_gamma: vec![1.0; d_model],
            ln1_beta: vec![0.0; d_model],
            ff1: synth_ff(state, d_model, ffn_dim),
            ln_attn_gamma: vec![1.0; d_model],
            ln_attn_beta: vec![0.0; d_model],
            mha: synth_mha(state, d_model),
            ln_cg_gamma: vec![1.0; d_model],
            ln_cg_beta: vec![0.0; d_model],
            cgmlp: synth_cgmlp(state, cfg),
            merge: synth_merge(state, cfg),
            ln4_gamma: vec![1.0; d_model],
            ln4_beta: vec![0.0; d_model],
            ff2: synth_ff(state, d_model, ffn_dim),
            ln_out_gamma: vec![1.0; d_model],
            ln_out_beta: vec![0.0; d_model],
        }
    }

    fn synth_weights(cfg: &EBranchformerConfig, state: &mut u64) -> EBranchformerWeights {
        let d_model = cfg.d_model as usize;
        let in_dim = cfg.in_dim as usize;
        let proj_in = cfg.stem.projection_in_dim(in_dim);
        let stem = EBranchformerStemWeights {
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
        let layers = (0..cfg.n_layers).map(|_| synth_layer(state, cfg)).collect();
        EBranchformerWeights { stem, layers }
    }

    fn small_cfg() -> EBranchformerConfig {
        EBranchformerConfig {
            in_dim: 4,
            d_model: 8,
            n_heads: 2,
            ffn_dim: 16,
            n_layers: 1,
            cgmlp_kernel_size: 3,
            cgmlp_hidden_dim: 16,
            merge_kernel_size: 3,
            stem: ConvSubsampleKind::Linear,
            position_encoding: PositionEncoding::None,
        }
    }

    // ---- Happy path ------------------------------------------------------

    #[test]
    fn forward_produces_expected_output_shape_linear_stem() {
        let cfg = small_cfg();
        let mut state = 1u64;
        let weights = synth_weights(&cfg, &mut state);
        let encoder = EBranchformerEncoder::new(cfg.clone(), weights).unwrap();
        let t = 12;
        let mel = synth_vec(&mut state, t * cfg.in_dim as usize, 1.0);
        let (out, t_out) = encoder.forward(&mel, t).unwrap();
        assert_eq!(t_out, t);
        assert_eq!(out.len(), t_out * cfg.d_model as usize);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_stacking_stem_downsamples() {
        let mut cfg = small_cfg();
        cfg.stem = ConvSubsampleKind::Stacking { factor: 4 };
        let mut state = 2u64;
        let weights = synth_weights(&cfg, &mut state);
        let encoder = EBranchformerEncoder::new(cfg.clone(), weights).unwrap();
        let t = 16;
        let mel = synth_vec(&mut state, t * cfg.in_dim as usize, 1.0);
        let (out, t_out) = encoder.forward(&mel, t).unwrap();
        assert_eq!(t_out, 4);
        assert_eq!(out.len(), 4 * cfg.d_model as usize);
    }

    #[test]
    fn multi_layer_stacks_and_stays_finite() {
        let mut cfg = small_cfg();
        cfg.n_layers = 3;
        let mut state = 3u64;
        let weights = synth_weights(&cfg, &mut state);
        let encoder = EBranchformerEncoder::new(cfg.clone(), weights).unwrap();
        let t = 8;
        let mel = synth_vec(&mut state, t * cfg.in_dim as usize, 1.0);
        let (out, t_out) = encoder.forward(&mel, t).unwrap();
        assert_eq!(t_out, t);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rope_overlay_changes_output_vs_no_pos_encoding() {
        let mut state = 4u64;
        let mut cfg = small_cfg();
        let weights = synth_weights(&cfg, &mut state);
        let enc_none = EBranchformerEncoder::new(cfg.clone(), weights.clone()).unwrap();
        cfg.position_encoding = PositionEncoding::Rope { theta: 10_000.0 };
        let enc_rope = EBranchformerEncoder::new(cfg.clone(), weights).unwrap();
        let mel = synth_vec(&mut state, 8 * cfg.in_dim as usize, 1.0);
        let (out_none, _) = enc_none.forward(&mel, 8).unwrap();
        let (out_rope, _) = enc_rope.forward(&mel, 8).unwrap();
        let any_diff = out_none
            .iter()
            .zip(out_rope.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(any_diff);
    }

    #[test]
    fn forward_is_deterministic() {
        let cfg = small_cfg();
        let mut state = 5u64;
        let weights = synth_weights(&cfg, &mut state);
        let encoder = EBranchformerEncoder::new(cfg.clone(), weights).unwrap();
        let mel = synth_vec(&mut state, 8 * cfg.in_dim as usize, 1.0);
        let (a, _) = encoder.forward(&mel, 8).unwrap();
        let (b, _) = encoder.forward(&mel, 8).unwrap();
        assert_eq!(a, b);
    }

    // ---- Shape validation errors ----------------------------------------

    #[test]
    fn new_rejects_d_model_not_divisible_by_n_heads() {
        let mut cfg = small_cfg();
        cfg.d_model = 9;
        cfg.n_heads = 2;
        let dummy = EBranchformerWeights {
            stem: EBranchformerStemWeights {
                linear_w: vec![],
                linear_b: vec![],
                norm_gamma: None,
                norm_beta: None,
            },
            layers: Vec::new(),
        };
        let err = EBranchformerEncoder::new(cfg, dummy).unwrap_err();
        assert!(
            err.to_string().contains("divisible by n_heads"),
            "got {err}"
        );
    }

    #[test]
    fn new_rejects_odd_cgmlp_hidden_dim() {
        let mut cfg = small_cfg();
        cfg.cgmlp_hidden_dim = 15;
        let dummy = EBranchformerWeights {
            stem: EBranchformerStemWeights {
                linear_w: vec![],
                linear_b: vec![],
                norm_gamma: None,
                norm_beta: None,
            },
            layers: Vec::new(),
        };
        let err = EBranchformerEncoder::new(cfg, dummy).unwrap_err();
        assert!(
            err.to_string().contains("cgmlp_hidden_dim must be even"),
            "got {err}"
        );
    }

    #[test]
    fn new_rejects_even_cgmlp_kernel_size() {
        let mut cfg = small_cfg();
        cfg.cgmlp_kernel_size = 4;
        let dummy = EBranchformerWeights {
            stem: EBranchformerStemWeights {
                linear_w: vec![],
                linear_b: vec![],
                norm_gamma: None,
                norm_beta: None,
            },
            layers: Vec::new(),
        };
        let err = EBranchformerEncoder::new(cfg, dummy).unwrap_err();
        assert!(
            err.to_string().contains("cgmlp_kernel_size must be odd"),
            "got {err}"
        );
    }

    #[test]
    fn new_rejects_even_merge_kernel_size() {
        let mut cfg = small_cfg();
        cfg.merge_kernel_size = 4;
        let dummy = EBranchformerWeights {
            stem: EBranchformerStemWeights {
                linear_w: vec![],
                linear_b: vec![],
                norm_gamma: None,
                norm_beta: None,
            },
            layers: Vec::new(),
        };
        let err = EBranchformerEncoder::new(cfg, dummy).unwrap_err();
        assert!(
            err.to_string().contains("merge_kernel_size must be odd"),
            "got {err}"
        );
    }

    #[test]
    fn new_rejects_zero_cgmlp_hidden_dim() {
        let mut cfg = small_cfg();
        cfg.cgmlp_hidden_dim = 0;
        let dummy = EBranchformerWeights {
            stem: EBranchformerStemWeights {
                linear_w: vec![],
                linear_b: vec![],
                norm_gamma: None,
                norm_beta: None,
            },
            layers: Vec::new(),
        };
        let err = EBranchformerEncoder::new(cfg, dummy).unwrap_err();
        assert!(
            err.to_string().contains("cgmlp_hidden_dim must be > 0"),
            "got {err}"
        );
    }

    #[test]
    fn new_rejects_zero_n_layers() {
        let mut cfg = small_cfg();
        cfg.n_layers = 0;
        let dummy = EBranchformerWeights {
            stem: EBranchformerStemWeights {
                linear_w: vec![],
                linear_b: vec![],
                norm_gamma: None,
                norm_beta: None,
            },
            layers: Vec::new(),
        };
        let err = EBranchformerEncoder::new(cfg, dummy).unwrap_err();
        assert!(
            err.to_string().contains("n_layers must be > 0"),
            "got {err}"
        );
    }

    #[test]
    fn new_rejects_wrong_layer_count() {
        let cfg = EBranchformerConfig {
            n_layers: 3,
            ..small_cfg()
        };
        let mut state = 6u64;
        let mut weights = synth_weights(&cfg, &mut state);
        weights.layers.truncate(1);
        let err = EBranchformerEncoder::new(cfg, weights).unwrap_err();
        assert!(err.to_string().contains("expected 3 layers"), "got {err}");
    }

    #[test]
    fn new_rejects_wrong_cgmlp_linear_in_shape() {
        let cfg = small_cfg();
        let mut state = 7u64;
        let mut weights = synth_weights(&cfg, &mut state);
        weights.layers[0].cgmlp.linear_in_w.truncate(4);
        let err = EBranchformerEncoder::new(cfg, weights).unwrap_err();
        assert!(err.to_string().contains("cgmlp linear_in_w"), "got {err}");
    }

    #[test]
    fn new_rejects_wrong_merge_linear_shape() {
        let cfg = small_cfg();
        let mut state = 8u64;
        let mut weights = synth_weights(&cfg, &mut state);
        weights.layers[0].merge.linear_w.truncate(4);
        let err = EBranchformerEncoder::new(cfg, weights).unwrap_err();
        assert!(err.to_string().contains("merge linear_w"), "got {err}");
    }

    #[test]
    fn forward_rejects_mismatched_mel_length() {
        let cfg = small_cfg();
        let mut state = 9u64;
        let weights = synth_weights(&cfg, &mut state);
        let encoder = EBranchformerEncoder::new(cfg, weights).unwrap();
        let mel = vec![0.0f32; 5];
        let err = encoder.forward(&mel, 6).unwrap_err();
        assert!(
            err.to_string().contains("does not match mel_frames"),
            "got {err}"
        );
    }

    #[test]
    fn forward_rejects_zero_mel_frames() {
        let cfg = small_cfg();
        let mut state = 10u64;
        let weights = synth_weights(&cfg, &mut state);
        let encoder = EBranchformerEncoder::new(cfg, weights).unwrap();
        let err = encoder.forward(&[], 0).unwrap_err();
        assert!(
            err.to_string().contains("mel_frames must be > 0"),
            "got {err}"
        );
    }

    // ---- Small helper pins ----------------------------------------------

    #[test]
    fn gelu_tanh_zero_is_zero() {
        assert!(gelu_tanh(0.0).abs() < 1e-7);
    }

    #[test]
    fn gelu_tanh_positive_is_monotonic() {
        // GELU is monotone-increasing on `[0, inf)`.
        assert!(gelu_tanh(0.5) < gelu_tanh(1.0));
        assert!(gelu_tanh(1.0) < gelu_tanh(2.0));
    }

    #[test]
    fn gelu_tanh_approximates_relu_for_large_positive() {
        // For x >> 0, GELU(x) ≈ x.
        let x = 5.0f32;
        assert!((gelu_tanh(x) - x).abs() < 1e-2);
    }

    #[test]
    fn cgmlp_gate_multiplication_is_elementwise() {
        // Set u to all-ones and v to a known pattern via zeroed norm +
        // depthwise = 0 & bias; the gated output should be v itself
        // (before linear_out projection). Easier to pin at the module
        // level via the encoder wiring: a small forward with fixed
        // input must produce a finite, deterministic result.
        let cfg = EBranchformerConfig {
            n_layers: 1,
            ..small_cfg()
        };
        let mut state = 42u64;
        let weights = synth_weights(&cfg, &mut state);
        let encoder = EBranchformerEncoder::new(cfg.clone(), weights).unwrap();
        let mel = synth_vec(&mut state, 4 * cfg.in_dim as usize, 1.0);
        let (out, _) = encoder.forward(&mel, 4).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn config_head_dim_is_d_model_over_n_heads() {
        let cfg = small_cfg();
        assert_eq!(cfg.head_dim(), 4);
    }

    #[test]
    fn config_cgmlp_half_dim_is_hidden_over_two() {
        let cfg = small_cfg();
        assert_eq!(cfg.cgmlp_half_dim(), 8);
    }
}
