//! ViT audio encoder — 2-D patch embedding + plain pre-norm Transformer.
//!
//! This is the **shared** encoder primitive behind the SSL audio-embedding
//! fleet. Five binders in `vokra-models` (`atst`, `eat`, `m2d`, `maest`, and
//! the `ast` / `beats` / `dasheng` converters behind them) are loud-partial
//! for the *same* single reason, and their [`VokraError::UnsupportedOp`]
//! messages independently name it: `vokra-ops` already has the log-mel front
//! end ([`crate::mel`], [`crate::fused_logmel`], [`crate::kaldi_fbank`]) but
//! had **no 2-D patch embedding + plain pre-norm Transformer encoder**.
//! [`crate::conformer`], [`crate::ebranchformer`] and [`crate::zipformer`]
//! are conv-augmented **ASR** encoders over a 1-D frame sequence — a
//! genuinely different architecture, not a substitute. One primitive here
//! closes that blocker for all five rather than five unrelated follow-ups.
//!
//! # There is no "upstream default" in this module
//!
//! Every axis — embedding width, depth, head count, MLP ratio, patch size,
//! patch stride, prepended-token count, LayerNorm epsilon, GELU flavour,
//! positional-embedding policy — is **caller-supplied**. The five consumers
//! use five different axis sets, so a default here would be a fabricated
//! number that binds shape-valid garbage without failing (FR-EX-08; CLAUDE.md
//! 教訓 (a) 「loud-partial は fake-complete より honest」). [`ViTAttrs`]
//! deliberately implements neither `Default` nor any `*_base()` constructor.
//!
//! # Pre-norm, NOT post-norm
//!
//! Each block is
//!
//! ```text
//! h = x + MHSA(LayerNorm(x))          // norm BEFORE the branch
//! y = h + MLP(LayerNorm(h))           // norm BEFORE the branch
//! ```
//!
//! followed by a single final LayerNorm after the whole stack. This is the
//! ViT / DeiT ordering. The post-norm ordering (`x + Branch(x)` then norm)
//! is a *different* function whose outputs are shape-valid and numerically
//! wrong — that failure is silent, so it is called out here explicitly.
//!
//! # Layout conventions
//!
//! - The input plane is `[n_mels, n_frames]` **row-major**: index
//!   `m * n_frames + f`. `patch_h` / `stride_h` walk the mel-bin axis and
//!   `patch_w` / `stride_w` walk the frame axis.
//! - Upstream models disagree about which of those two axes they call "H".
//!   This primitive does **not** guess: if an upstream `Conv2d` is defined
//!   over the transposed plane, the caller must transpose before calling.
//! - Patches are flattened **row-major within the patch** (mel-bin major):
//!   element `(i, j)` of a patch lands at `i * patch_w + j`. That matches a
//!   `Conv2d` weight `[out_ch, 1, kH, kW]` flattened over its trailing dims.
//! - Tokens are emitted in **grid row-major order** (mel-bin major, then
//!   frame): patch `(gy, gx)` is token index `gy * grid_w + gx`. The
//!   positional-embedding table must use the same order.
//! - Every hidden state is `[n_tokens, embed_dim]` row-major (token-major),
//!   which is the natural layout for the token-parallel attention pass —
//!   same convention as [`crate::conformer`].
//! - Prepended tokens (CLS / distillation) occupy indices
//!   `[0, n_prepended_tokens)`; patch tokens follow.
//!
//! # Prepended tokens
//!
//! The count is configurable rather than fixed at 1, because both
//! conventions occur in the wild: a single class token, and a class token
//! plus a distillation token (the DeiT-style pair). This module does not
//! record which consumer uses which count — that has to be read off each
//! model's own checkpoint, and guessing it would silently shift every patch
//! token's positional embedding by one row.
//!
//! # Zero-dependency posture (NFR-DS-02)
//!
//! No BLAS, no `serde`, no third-party crate. All math is safe Rust; there
//! is no `unsafe` in this module. The error-function used by the exact GELU
//! is the Abramowitz & Stegun 7.1.26 rational-times-Gaussian approximation
//! (`|ε| ≤ 1.5e-7`), evaluated in `f64` and rounded once to `f32`.

use vokra_core::{Result, VokraError};

// ---------------------------------------------------------------------------
// Configuration enums
// ---------------------------------------------------------------------------

/// Which GELU formulation the MLP uses.
///
/// The two differ by up to ~1e-3 in absolute value. Picking the wrong one is
/// *silently* slightly wrong — outputs stay finite and correctly shaped — so
/// the choice is an explicit axis rather than a default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeluKind {
    /// Exact GELU, `0.5 · x · (1 + erf(x / √2))`.
    ///
    /// This is what a bare `torch.nn.GELU()` computes.
    Erf,
    /// The tanh approximation,
    /// `0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))`.
    ///
    /// This is what `torch.nn.GELU(approximate="tanh")` computes.
    Tanh,
}

/// What to do when the positional-embedding table does not have exactly one
/// row per token.
///
/// ViT-audio checkpoints are trained at one input length and then applied at
/// another, so upstream inference code resizes the table. Doing that
/// silently would change the numbers under the caller's feet, so the policy
/// is an explicit axis with no default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PosEmbedPolicy {
    /// Require `pos_embed` to hold exactly `n_prepended_tokens + n_patches`
    /// rows. Anything else is a loud [`VokraError::InvalidArgument`].
    ///
    /// This is the parity-safe choice: resize the table offline with the
    /// upstream resizer and hand this primitive the already-correct table.
    RequireExact,
    /// Resize the patch-grid part of the table from `table_grid_h ×
    /// table_grid_w` to the runtime grid by **bilinear** interpolation
    /// (half-pixel centres, i.e. the `align_corners=False` convention),
    /// copying the prepended rows through unchanged.
    ///
    /// **This is not bit-exact with upstream.** ViT-audio implementations
    /// generally resize positional tables *bicubically*; bilinear is a
    /// different filter and will differ in the third decimal or so. Use it
    /// when an approximate embedding is acceptable, and use
    /// [`PosEmbedPolicy::RequireExact`] with an offline-resized table when
    /// numerical parity against upstream matters.
    InterpolateGridBilinear {
        /// Mel-bin-axis grid height the table was trained at.
        table_grid_h: usize,
        /// Frame-axis grid width the table was trained at.
        table_grid_w: usize,
    },
}

/// How to collapse the per-token hidden states into a single embedding.
///
/// Both conventions genuinely occur, and they produce different vectors from
/// the same encoder, so there is no default: the caller states which one its
/// model was trained with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViTPooling {
    /// Take one token out of the prepended block (index `0` is the usual
    /// class token; index `1` is the usual distillation token when a model
    /// prepends two).
    PrependedToken {
        /// Index within the prepended block, i.e. `0 <= index <
        /// n_prepended_tokens`.
        index: usize,
    },
    /// Mean over the **patch** tokens only, skipping the prepended block.
    MeanPatchTokens,
}

// ---------------------------------------------------------------------------
// Attributes
// ---------------------------------------------------------------------------

/// Caller-supplied axis set for a ViT audio encoder.
///
/// Nothing in this struct has a default — see the module docs. Construct it
/// from axes that a checkpoint or a transcribed primary source actually
/// carries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViTAttrs {
    /// Token width `D` carried through the whole encoder.
    pub embed_dim: usize,
    /// Number of stacked pre-norm Transformer blocks.
    pub depth: usize,
    /// Number of attention heads; must divide [`ViTAttrs::embed_dim`].
    pub n_heads: usize,
    /// MLP hidden width as a multiple of [`ViTAttrs::embed_dim`]; the
    /// resolved width is [`ViTAttrs::mlp_dim`].
    pub mlp_ratio: f32,
    /// Patch extent along the mel-bin axis.
    pub patch_h: usize,
    /// Patch extent along the frame axis.
    pub patch_w: usize,
    /// Patch stride along the mel-bin axis. Equal to [`ViTAttrs::patch_h`]
    /// for non-overlapping patches, smaller for overlapping ones.
    pub stride_h: usize,
    /// Patch stride along the frame axis. Equal to [`ViTAttrs::patch_w`] for
    /// non-overlapping patches, smaller for overlapping ones.
    pub stride_w: usize,
    /// How many learned tokens are prepended ahead of the patch tokens
    /// (CLS / distillation). May be `0`.
    pub n_prepended_tokens: usize,
    /// LayerNorm epsilon used by every norm in the encoder.
    pub layer_norm_eps: f32,
    /// Which GELU formulation the MLP uses.
    pub gelu: GeluKind,
    /// What to do when the positional-embedding table length does not match
    /// the runtime token count.
    pub pos_embed_policy: PosEmbedPolicy,
}

impl ViTAttrs {
    /// Per-head width, `embed_dim / n_heads`.
    ///
    /// Only meaningful once [`ViTAttrs::validate`] has passed; it uses an
    /// integer division that would silently floor otherwise.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.n_heads
    }

    /// Resolved MLP hidden width, `round(embed_dim · mlp_ratio)`.
    ///
    /// Rounding (rather than truncation) is used because ratios such as
    /// `8/3` are common and truncation would silently lose a unit.
    #[must_use]
    pub fn mlp_dim(&self) -> usize {
        let scaled = self.embed_dim as f32 * self.mlp_ratio;
        if scaled <= 0.0 {
            return 0;
        }
        scaled.round() as usize
    }

    /// Validate the axis set.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] when any axis is zero, when
    /// `embed_dim` is not divisible by `n_heads`, when `mlp_ratio` /
    /// `layer_norm_eps` are not finite and positive, or when `mlp_ratio`
    /// rounds the hidden width down to zero.
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("embed_dim", self.embed_dim),
            ("depth", self.depth),
            ("n_heads", self.n_heads),
            ("patch_h", self.patch_h),
            ("patch_w", self.patch_w),
            ("stride_h", self.stride_h),
            ("stride_w", self.stride_w),
        ] {
            if value == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "ViTAttrs::validate: {name} must be > 0, got 0"
                )));
            }
        }
        if self.embed_dim % self.n_heads != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "ViTAttrs::validate: embed_dim ({}) must be divisible by n_heads ({})",
                self.embed_dim, self.n_heads
            )));
        }
        if !self.mlp_ratio.is_finite() || self.mlp_ratio <= 0.0 {
            return Err(VokraError::InvalidArgument(format!(
                "ViTAttrs::validate: mlp_ratio must be finite and > 0, got {}",
                self.mlp_ratio
            )));
        }
        if self.mlp_dim() == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "ViTAttrs::validate: mlp_ratio {} rounds the MLP hidden width to 0 \
                 for embed_dim {}",
                self.mlp_ratio, self.embed_dim
            )));
        }
        if !self.layer_norm_eps.is_finite() || self.layer_norm_eps <= 0.0 {
            return Err(VokraError::InvalidArgument(format!(
                "ViTAttrs::validate: layer_norm_eps must be finite and > 0, got {}",
                self.layer_norm_eps
            )));
        }
        if let PosEmbedPolicy::InterpolateGridBilinear {
            table_grid_h,
            table_grid_w,
        } = self.pos_embed_policy
        {
            if table_grid_h == 0 || table_grid_w == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "ViTAttrs::validate: InterpolateGridBilinear needs a non-empty \
                     table grid, got {table_grid_h}×{table_grid_w}"
                )));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Patch grid
// ---------------------------------------------------------------------------

/// The patch grid a `[n_mels, n_frames]` plane produces under a given patch
/// size and stride.
///
/// A caller cannot reconstruct this from the token count alone (many
/// `grid_h × grid_w` factorisations give the same product), so
/// [`ViTEncoder::forward`] returns it alongside the tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PatchGrid {
    /// Patch count along the mel-bin axis.
    pub grid_h: usize,
    /// Patch count along the frame axis.
    pub grid_w: usize,
    /// `grid_h · grid_w`.
    pub n_patches: usize,
    /// Mel bins at the tail of the plane that no patch covers.
    ///
    /// Non-zero whenever `(n_mels - patch_h)` is not a multiple of
    /// `stride_h`. Reported rather than hidden: a silently discarded tail is
    /// a real information loss and the caller may want to pad instead.
    pub dropped_rows: usize,
    /// Frames at the tail of the plane that no patch covers, same rationale
    /// as [`PatchGrid::dropped_rows`].
    pub dropped_cols: usize,
}

impl PatchGrid {
    /// Total sequence length once `n_prepended_tokens` learned tokens sit in
    /// front of the patch tokens.
    #[must_use]
    pub fn n_tokens(&self, n_prepended_tokens: usize) -> usize {
        n_prepended_tokens + self.n_patches
    }
}

/// Compute the patch grid for a `[n_mels, n_frames]` plane.
///
/// The arithmetic is the no-padding, no-dilation `Conv2d` rule:
///
/// ```text
/// grid_h = (n_mels   - patch_h) / stride_h + 1      (integer division)
/// grid_w = (n_frames - patch_w) / stride_w + 1
/// ```
///
/// The integer division floors, so a **ragged final patch** — a tail too
/// short to fill a whole patch — is dropped. That is the upstream behaviour,
/// and the size of the dropped tail is reported in
/// [`PatchGrid::dropped_rows`] / [`PatchGrid::dropped_cols`] rather than
/// being swallowed.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] when the attributes are invalid, or when
/// the plane is smaller than one patch along either axis (no patch could be
/// formed, and returning an empty grid would hand the encoder a zero-length
/// sequence instead of reporting the real problem).
pub fn patch_grid(n_mels: usize, n_frames: usize, attrs: &ViTAttrs) -> Result<PatchGrid> {
    attrs.validate()?;
    if n_mels < attrs.patch_h {
        return Err(VokraError::InvalidArgument(format!(
            "vit::patch_grid: n_mels ({n_mels}) is smaller than patch_h ({}), \
             so no patch fits along the mel-bin axis",
            attrs.patch_h
        )));
    }
    if n_frames < attrs.patch_w {
        return Err(VokraError::InvalidArgument(format!(
            "vit::patch_grid: n_frames ({n_frames}) is smaller than patch_w ({}), \
             so no patch fits along the frame axis",
            attrs.patch_w
        )));
    }
    let grid_h = (n_mels - attrs.patch_h) / attrs.stride_h + 1;
    let grid_w = (n_frames - attrs.patch_w) / attrs.stride_w + 1;
    let covered_h = (grid_h - 1) * attrs.stride_h + attrs.patch_h;
    let covered_w = (grid_w - 1) * attrs.stride_w + attrs.patch_w;
    Ok(PatchGrid {
        grid_h,
        grid_w,
        n_patches: grid_h * grid_w,
        dropped_rows: n_mels - covered_h,
        dropped_cols: n_frames - covered_w,
    })
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// Patch-embedding projection: one learned linear from a flattened patch to
/// a token.
#[derive(Debug, Clone)]
pub struct PatchEmbedWeights {
    /// Row-major `[embed_dim, patch_h · patch_w]`.
    ///
    /// A `Conv2d(1, embed_dim, (patch_h, patch_w))` weight
    /// `[embed_dim, 1, patch_h, patch_w]` flattens directly into this
    /// layout.
    pub proj_w: Vec<f32>,
    /// Optional `[embed_dim]` bias. `None` for a bias-free projection.
    pub proj_b: Option<Vec<f32>>,
}

/// Multi-head self-attention weights, all biases optional (both
/// `qkv_bias=True` and `qkv_bias=False` configurations occur).
#[derive(Debug, Clone)]
pub struct ViTAttnWeights {
    /// Row-major `[embed_dim, embed_dim]` query projection.
    pub wq: Vec<f32>,
    /// Optional `[embed_dim]` query bias.
    pub bq: Option<Vec<f32>>,
    /// Row-major `[embed_dim, embed_dim]` key projection.
    pub wk: Vec<f32>,
    /// Optional `[embed_dim]` key bias.
    pub bk: Option<Vec<f32>>,
    /// Row-major `[embed_dim, embed_dim]` value projection.
    pub wv: Vec<f32>,
    /// Optional `[embed_dim]` value bias.
    pub bv: Option<Vec<f32>>,
    /// Row-major `[embed_dim, embed_dim]` output projection.
    pub wo: Vec<f32>,
    /// Optional `[embed_dim]` output bias.
    pub bo: Option<Vec<f32>>,
}

/// Two-layer MLP weights: `Linear(D → H) → GELU → Linear(H → D)`.
#[derive(Debug, Clone)]
pub struct ViTMlpWeights {
    /// Row-major `[mlp_dim, embed_dim]`.
    pub w1: Vec<f32>,
    /// Optional `[mlp_dim]` bias.
    pub b1: Option<Vec<f32>>,
    /// Row-major `[embed_dim, mlp_dim]`.
    pub w2: Vec<f32>,
    /// Optional `[embed_dim]` bias.
    pub b2: Option<Vec<f32>>,
}

/// One pre-norm Transformer block.
///
/// `ln1` normalises the attention branch's input and `ln2` the MLP branch's
/// input — both **before** their branch, never after the residual add.
#[derive(Debug, Clone)]
pub struct ViTBlockWeights {
    /// `[embed_dim]` gain of the pre-attention LayerNorm.
    pub ln1_gamma: Vec<f32>,
    /// `[embed_dim]` bias of the pre-attention LayerNorm.
    pub ln1_beta: Vec<f32>,
    /// Self-attention weights.
    pub attn: ViTAttnWeights,
    /// `[embed_dim]` gain of the pre-MLP LayerNorm.
    pub ln2_gamma: Vec<f32>,
    /// `[embed_dim]` bias of the pre-MLP LayerNorm.
    pub ln2_beta: Vec<f32>,
    /// MLP weights.
    pub mlp: ViTMlpWeights,
}

/// Full encoder weight set.
#[derive(Debug, Clone)]
pub struct ViTWeights {
    /// Patch projection.
    pub patch_embed: PatchEmbedWeights,
    /// Row-major `[n_prepended_tokens, embed_dim]` learned prepended tokens.
    /// Empty when `n_prepended_tokens == 0`.
    pub prepended_tokens: Vec<f32>,
    /// Row-major `[pos_rows, embed_dim]` additive positional table, ordered
    /// prepended-rows-first then patch rows in grid row-major order. How
    /// `pos_rows` may differ from the runtime token count is governed by
    /// [`ViTAttrs::pos_embed_policy`].
    pub pos_embed: Vec<f32>,
    /// One entry per block; length must equal [`ViTAttrs::depth`].
    pub blocks: Vec<ViTBlockWeights>,
    /// `[embed_dim]` gain of the norm applied after the whole stack.
    pub final_ln_gamma: Vec<f32>,
    /// `[embed_dim]` bias of the norm applied after the whole stack.
    pub final_ln_beta: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Backend seam
// ---------------------------------------------------------------------------

/// Backend-dispatched learned operations used by [`ViTEncoder`].
///
/// The public trait lives in `vokra-ops` so higher layers can route the same
/// validated ViT topology through CPU or GPU kernels without adding a reverse
/// dependency from `vokra-ops` onto a model crate. Tensor gathering,
/// transposes, positional addition and residual addition remain host-side
/// layout glue; every learned reduction is represented here.
pub trait ViTBackendOps {
    /// PyTorch-style linear: `weight` is row-major `[out_dim, in_dim]`.
    #[allow(clippy::too_many_arguments)]
    fn linear_f32(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        rows: usize,
        in_dim: usize,
        out_dim: usize,
        output: &mut [f32],
    ) -> Result<()>;

    /// Row-major matrix multiplication: `[m, k] * [k, n] -> [m, n]`.
    #[allow(clippy::too_many_arguments)]
    fn matmul_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        left: &[f32],
        right: &[f32],
        output: &mut [f32],
    ) -> Result<()>;

    /// Row-wise softmax over a `[rows, cols]` matrix.
    fn softmax_f32(
        &self,
        input: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
    ) -> Result<()>;

    /// Affine LayerNorm over the innermost dimension.
    #[allow(clippy::too_many_arguments)]
    fn layer_norm_f32(
        &self,
        input: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Result<()>;

    /// Element-wise GELU in the explicitly selected formulation.
    fn gelu_f32(&self, kind: GeluKind, input: &[f32], output: &mut [f32]) -> Result<()>;
}

/// Scalar reference backend used by the original CPU entry points.
#[derive(Debug, Clone, Copy)]
struct ScalarViTBackend;

impl ViTBackendOps for ScalarViTBackend {
    fn linear_f32(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        rows: usize,
        in_dim: usize,
        out_dim: usize,
        output: &mut [f32],
    ) -> Result<()> {
        for row in 0..rows {
            linear_row(
                &input[row * in_dim..(row + 1) * in_dim],
                weight,
                bias,
                out_dim,
                in_dim,
                &mut output[row * out_dim..(row + 1) * out_dim],
            );
        }
        Ok(())
    }

    fn matmul_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        left: &[f32],
        right: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for inner in 0..k {
                    sum += left[row * k + inner] * right[inner * n + col];
                }
                output[row * n + col] = sum;
            }
        }
        Ok(())
    }

    fn softmax_f32(
        &self,
        input: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        for row in 0..rows {
            softmax_row(
                &input[row * cols..(row + 1) * cols],
                &mut output[row * cols..(row + 1) * cols],
            );
        }
        Ok(())
    }

    fn layer_norm_f32(
        &self,
        input: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Result<()> {
        output.copy_from_slice(input);
        for row in output.chunks_mut(cols).take(rows) {
            layer_norm_row(row, gamma, beta, eps);
        }
        Ok(())
    }

    fn gelu_f32(&self, kind: GeluKind, input: &[f32], output: &mut [f32]) -> Result<()> {
        for (slot, &value) in output.iter_mut().zip(input) {
            *slot = gelu(value, kind);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// A validated ViT audio encoder: patch embedding, prepended tokens,
/// positional embedding, a pre-norm Transformer stack and a final norm.
#[derive(Debug, Clone)]
pub struct ViTEncoder {
    attrs: ViTAttrs,
    weights: ViTWeights,
}

impl ViTEncoder {
    /// Validate the axes and every weight buffer, then build the encoder.
    ///
    /// All shape and finiteness checking happens here so the forward path
    /// can index without re-checking, and so a mis-bound checkpoint fails at
    /// load time rather than producing shape-valid garbage (FR-EX-08).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] when [`ViTAttrs::validate`] fails,
    /// when any weight buffer has the wrong length, when the block count
    /// differs from `depth`, when the positional table is not a whole number
    /// of rows (or, under
    /// [`PosEmbedPolicy::InterpolateGridBilinear`], not the number of rows
    /// the declared table grid implies), or when any weight is non-finite.
    pub fn new(attrs: ViTAttrs, weights: ViTWeights) -> Result<Self> {
        attrs.validate()?;
        let d = attrs.embed_dim;
        let patch_len = attrs.patch_h * attrs.patch_w;

        check_len(
            &weights.patch_embed.proj_w,
            d * patch_len,
            "ViTEncoder::new: patch_embed.proj_w",
        )?;
        if let Some(bias) = &weights.patch_embed.proj_b {
            check_len(bias, d, "ViTEncoder::new: patch_embed.proj_b")?;
        }
        check_len(
            &weights.prepended_tokens,
            attrs.n_prepended_tokens * d,
            "ViTEncoder::new: prepended_tokens",
        )?;

        if weights.pos_embed.is_empty() || weights.pos_embed.len() % d != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "ViTEncoder::new: pos_embed length {} is not a non-zero multiple of \
                 embed_dim {d}",
                weights.pos_embed.len()
            )));
        }
        let pos_rows = weights.pos_embed.len() / d;
        if let PosEmbedPolicy::InterpolateGridBilinear {
            table_grid_h,
            table_grid_w,
        } = attrs.pos_embed_policy
        {
            let expected = attrs.n_prepended_tokens + table_grid_h * table_grid_w;
            if pos_rows != expected {
                return Err(VokraError::InvalidArgument(format!(
                    "ViTEncoder::new: pos_embed holds {pos_rows} rows but the declared \
                     table grid {table_grid_h}×{table_grid_w} plus \
                     {} prepended token(s) implies {expected}",
                    attrs.n_prepended_tokens
                )));
            }
        }

        if weights.blocks.len() != attrs.depth {
            return Err(VokraError::InvalidArgument(format!(
                "ViTEncoder::new: depth is {} but {} block(s) were supplied",
                attrs.depth,
                weights.blocks.len()
            )));
        }
        let mlp_dim = attrs.mlp_dim();
        for (i, block) in weights.blocks.iter().enumerate() {
            validate_block(block, d, mlp_dim, i)?;
        }
        check_len(
            &weights.final_ln_gamma,
            d,
            "ViTEncoder::new: final_ln_gamma",
        )?;
        check_len(&weights.final_ln_beta, d, "ViTEncoder::new: final_ln_beta")?;

        require_finite(&weights.patch_embed.proj_w, "ViTEncoder::new: proj_w")?;
        if let Some(bias) = &weights.patch_embed.proj_b {
            require_finite(bias, "ViTEncoder::new: proj_b")?;
        }
        require_finite(
            &weights.prepended_tokens,
            "ViTEncoder::new: prepended_tokens",
        )?;
        require_finite(&weights.pos_embed, "ViTEncoder::new: pos_embed")?;
        require_finite(&weights.final_ln_gamma, "ViTEncoder::new: final_ln_gamma")?;
        require_finite(&weights.final_ln_beta, "ViTEncoder::new: final_ln_beta")?;
        for (i, block) in weights.blocks.iter().enumerate() {
            require_finite_block(block, i)?;
        }

        Ok(Self { attrs, weights })
    }

    /// The axis set this encoder was built with.
    #[must_use]
    pub fn attrs(&self) -> &ViTAttrs {
        &self.attrs
    }

    /// The weight set this encoder was built with.
    #[must_use]
    pub fn weights(&self) -> &ViTWeights {
        &self.weights
    }

    /// Patch grid for a plane of the given shape, under this encoder's patch
    /// size and stride.
    ///
    /// # Errors
    ///
    /// Propagates [`patch_grid`].
    pub fn patch_grid(&self, n_mels: usize, n_frames: usize) -> Result<PatchGrid> {
        patch_grid(n_mels, n_frames, &self.attrs)
    }

    /// Full forward: patch-embed the plane, prepend the learned tokens, add
    /// the positional table, run the pre-norm stack and the final norm.
    ///
    /// Returns the `[n_tokens, embed_dim]` row-major hidden states together
    /// with the [`PatchGrid`], because the grid dimensions cannot be
    /// recovered from the token count.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] when `mel.len() != n_mels · n_frames`,
    /// when the plane is non-finite, when no patch fits, or when the
    /// positional table cannot be applied under the configured policy.
    pub fn forward(
        &self,
        mel: &[f32],
        n_mels: usize,
        n_frames: usize,
    ) -> Result<(Vec<f32>, PatchGrid)> {
        self.forward_with_backend(mel, n_mels, n_frames, &ScalarViTBackend)
    }

    /// Backend-selectable twin of [`Self::forward`].
    ///
    /// All learned projections, attention matrix products, softmaxes,
    /// normalizations and GELUs are delegated to `backend`. Host work is
    /// limited to shape/layout transforms, positional addition and residuals.
    pub fn forward_with_backend(
        &self,
        mel: &[f32],
        n_mels: usize,
        n_frames: usize,
        backend: &dyn ViTBackendOps,
    ) -> Result<(Vec<f32>, PatchGrid)> {
        let (patch_tokens, grid) = vit_patch_embed_with_backend(
            mel,
            n_mels,
            n_frames,
            &self.attrs,
            &self.weights.patch_embed,
            backend,
        )?;
        let d = self.attrs.embed_dim;
        let n_prepended = self.attrs.n_prepended_tokens;
        let n_tokens = grid.n_tokens(n_prepended);

        let mut tokens = Vec::with_capacity(n_tokens * d);
        tokens.extend_from_slice(&self.weights.prepended_tokens);
        tokens.extend_from_slice(&patch_tokens);

        vit_add_pos_embed(
            &mut tokens,
            &grid,
            &self.weights.pos_embed,
            d,
            n_prepended,
            self.attrs.pos_embed_policy,
        )?;

        let hidden = self.encode_tokens_with_backend(&tokens, n_tokens, backend)?;
        Ok((hidden, grid))
    }

    /// Run the pre-norm block stack and the final norm over an
    /// already-assembled `[n_tokens, embed_dim]` token sequence.
    ///
    /// This entry point adds **no** positional embedding and prepends
    /// nothing — it is the encoder body alone, for callers that build their
    /// own token sequence. Because self-attention without a positional
    /// signal is permutation-equivariant, this function is too; that is
    /// precisely why [`ViTEncoder::forward`] adds the positional table
    /// before calling it.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] when `n_tokens` is zero, when
    /// `tokens.len() != n_tokens · embed_dim`, or when any input value is
    /// non-finite.
    pub fn encode_tokens(&self, tokens: &[f32], n_tokens: usize) -> Result<Vec<f32>> {
        self.encode_tokens_with_backend(tokens, n_tokens, &ScalarViTBackend)
    }

    /// Backend-selectable twin of [`Self::encode_tokens`].
    pub fn encode_tokens_with_backend(
        &self,
        tokens: &[f32],
        n_tokens: usize,
        backend: &dyn ViTBackendOps,
    ) -> Result<Vec<f32>> {
        let d = self.attrs.embed_dim;
        if n_tokens == 0 {
            return Err(VokraError::InvalidArgument(
                "ViTEncoder::encode_tokens: n_tokens must be > 0".to_owned(),
            ));
        }
        check_len(tokens, n_tokens * d, "ViTEncoder::encode_tokens: tokens")?;
        require_finite(tokens, "ViTEncoder::encode_tokens: tokens")?;

        let mut hidden = tokens.to_vec();
        for block in &self.weights.blocks {
            hidden = self.block_forward(&hidden, n_tokens, block, backend)?;
        }
        let mut output = vec![0.0f32; hidden.len()];
        backend.layer_norm_f32(
            &hidden,
            &mut output,
            n_tokens,
            d,
            &self.weights.final_ln_gamma,
            &self.weights.final_ln_beta,
            self.attrs.layer_norm_eps,
        )?;
        Ok(output)
    }

    /// Add the positional table to an assembled token sequence in place,
    /// under this encoder's [`PosEmbedPolicy`].
    ///
    /// # Errors
    ///
    /// Propagates [`vit_add_pos_embed`].
    pub fn add_pos_embed(&self, tokens: &mut [f32], grid: &PatchGrid) -> Result<()> {
        vit_add_pos_embed(
            tokens,
            grid,
            &self.weights.pos_embed,
            self.attrs.embed_dim,
            self.attrs.n_prepended_tokens,
            self.attrs.pos_embed_policy,
        )
    }

    /// Collapse per-token hidden states into a single `[embed_dim]` vector.
    ///
    /// # Errors
    ///
    /// Propagates [`vit_pool`].
    pub fn pool(&self, tokens: &[f32], n_tokens: usize, pooling: ViTPooling) -> Result<Vec<f32>> {
        vit_pool(
            tokens,
            n_tokens,
            self.attrs.embed_dim,
            self.attrs.n_prepended_tokens,
            pooling,
        )
    }

    // -----------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------

    /// One pre-norm block: `h = x + MHSA(LN1(x))`, `y = h + MLP(LN2(h))`.
    fn block_forward(
        &self,
        x: &[f32],
        n_tokens: usize,
        w: &ViTBlockWeights,
        backend: &dyn ViTBackendOps,
    ) -> Result<Vec<f32>> {
        let d = self.attrs.embed_dim;
        let eps = self.attrs.layer_norm_eps;

        let mut residual = x.to_vec();

        let mut normed = vec![0.0f32; residual.len()];
        backend.layer_norm_f32(
            &residual,
            &mut normed,
            n_tokens,
            d,
            &w.ln1_gamma,
            &w.ln1_beta,
            eps,
        )?;
        let attn_out = self.attention(&normed, n_tokens, &w.attn, backend)?;
        for (slot, delta) in residual.iter_mut().zip(attn_out.iter()) {
            *slot += delta;
        }

        let mut normed2 = vec![0.0f32; residual.len()];
        backend.layer_norm_f32(
            &residual,
            &mut normed2,
            n_tokens,
            d,
            &w.ln2_gamma,
            &w.ln2_beta,
            eps,
        )?;
        let mlp_out = self.mlp_inner(&normed2, n_tokens, &w.mlp, backend)?;
        for (slot, delta) in residual.iter_mut().zip(mlp_out.iter()) {
            *slot += delta;
        }

        Ok(residual)
    }

    /// Standard scaled-dot-product multi-head self-attention, no mask.
    ///
    /// A ViT encoder over a patch grid is fully bidirectional — there is no
    /// causal structure to mask, unlike the decoder-side attention in
    /// [`crate::conformer`]'s consumers.
    fn attention(
        &self,
        x: &[f32],
        n_tokens: usize,
        w: &ViTAttnWeights,
        backend: &dyn ViTBackendOps,
    ) -> Result<Vec<f32>> {
        let d = self.attrs.embed_dim;
        let n_heads = self.attrs.n_heads;
        let head_dim = self.attrs.head_dim();
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut q = vec![0.0f32; n_tokens * d];
        let mut k = vec![0.0f32; n_tokens * d];
        let mut v = vec![0.0f32; n_tokens * d];
        backend.linear_f32(x, &w.wq, w.bq.as_deref(), n_tokens, d, d, &mut q)?;
        backend.linear_f32(x, &w.wk, w.bk.as_deref(), n_tokens, d, d, &mut k)?;
        backend.linear_f32(x, &w.wv, w.bv.as_deref(), n_tokens, d, d, &mut v)?;

        let mut context = vec![0.0f32; n_tokens * d];
        for h in 0..n_heads {
            let off = h * head_dim;
            let mut q_head = vec![0.0f32; n_tokens * head_dim];
            let mut k_transposed = vec![0.0f32; head_dim * n_tokens];
            let mut v_head = vec![0.0f32; n_tokens * head_dim];
            for token in 0..n_tokens {
                for inner in 0..head_dim {
                    q_head[token * head_dim + inner] = q[token * d + off + inner];
                    k_transposed[inner * n_tokens + token] = k[token * d + off + inner];
                    v_head[token * head_dim + inner] = v[token * d + off + inner];
                }
            }
            let mut scores = vec![0.0f32; n_tokens * n_tokens];
            backend.matmul_f32(
                n_tokens,
                n_tokens,
                head_dim,
                &q_head,
                &k_transposed,
                &mut scores,
            )?;
            for score in &mut scores {
                *score *= scale;
            }
            let mut probs = vec![0.0f32; scores.len()];
            backend.softmax_f32(&scores, &mut probs, n_tokens, n_tokens)?;
            let mut head_context = vec![0.0f32; n_tokens * head_dim];
            backend.matmul_f32(
                n_tokens,
                head_dim,
                n_tokens,
                &probs,
                &v_head,
                &mut head_context,
            )?;
            for token in 0..n_tokens {
                context[token * d + off..token * d + off + head_dim]
                    .copy_from_slice(&head_context[token * head_dim..(token + 1) * head_dim]);
            }
        }

        let mut out = vec![0.0f32; n_tokens * d];
        backend.linear_f32(&context, &w.wo, w.bo.as_deref(), n_tokens, d, d, &mut out)?;
        Ok(out)
    }

    /// `Linear(D → H) → GELU → Linear(H → D)`, applied per token.
    fn mlp_inner(
        &self,
        x: &[f32],
        n_tokens: usize,
        w: &ViTMlpWeights,
        backend: &dyn ViTBackendOps,
    ) -> Result<Vec<f32>> {
        let d = self.attrs.embed_dim;
        let h = self.attrs.mlp_dim();
        let kind = self.attrs.gelu;
        let mut out = vec![0.0f32; n_tokens * d];
        let mut hidden = vec![0.0f32; n_tokens * h];
        backend.linear_f32(x, &w.w1, w.b1.as_deref(), n_tokens, d, h, &mut hidden)?;
        let mut activated = vec![0.0f32; hidden.len()];
        backend.gelu_f32(kind, &hidden, &mut activated)?;
        backend.linear_f32(&activated, &w.w2, w.b2.as_deref(), n_tokens, h, d, &mut out)?;
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Patch-embed a `[n_mels, n_frames]` row-major plane into
/// `[n_patches, embed_dim]` tokens.
///
/// Patches are gathered row-major within the patch (mel-bin major) and
/// emitted in grid row-major order — see the module docs, since a
/// transposed convention here is silently wrong rather than loud.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] when the attributes are invalid, when the
/// plane is smaller than one patch, when `mel.len() != n_mels · n_frames`,
/// when the plane holds a non-finite value, or when `proj_w` / `proj_b` have
/// the wrong length.
pub fn vit_patch_embed(
    mel: &[f32],
    n_mels: usize,
    n_frames: usize,
    attrs: &ViTAttrs,
    weights: &PatchEmbedWeights,
) -> Result<(Vec<f32>, PatchGrid)> {
    vit_patch_embed_with_backend(mel, n_mels, n_frames, attrs, weights, &ScalarViTBackend)
}

fn vit_patch_embed_with_backend(
    mel: &[f32],
    n_mels: usize,
    n_frames: usize,
    attrs: &ViTAttrs,
    weights: &PatchEmbedWeights,
    backend: &dyn ViTBackendOps,
) -> Result<(Vec<f32>, PatchGrid)> {
    let grid = patch_grid(n_mels, n_frames, attrs)?;
    check_len(mel, n_mels * n_frames, "vit_patch_embed: mel")?;
    require_finite(mel, "vit_patch_embed: mel")?;

    let d = attrs.embed_dim;
    let patch_len = attrs.patch_h * attrs.patch_w;
    check_len(&weights.proj_w, d * patch_len, "vit_patch_embed: proj_w")?;
    if let Some(bias) = &weights.proj_b {
        check_len(bias, d, "vit_patch_embed: proj_b")?;
    }

    let mut patches = vec![0.0f32; grid.n_patches * patch_len];
    for gy in 0..grid.grid_h {
        let row0 = gy * attrs.stride_h;
        for gx in 0..grid.grid_w {
            let col0 = gx * attrs.stride_w;
            let token = gy * grid.grid_w + gx;
            let patch = &mut patches[token * patch_len..(token + 1) * patch_len];
            for (pi, chunk) in patch.chunks_mut(attrs.patch_w).enumerate() {
                let src = (row0 + pi) * n_frames + col0;
                chunk.copy_from_slice(&mel[src..src + attrs.patch_w]);
            }
        }
    }
    let mut tokens = vec![0.0f32; grid.n_patches * d];
    backend.linear_f32(
        &patches,
        &weights.proj_w,
        weights.proj_b.as_deref(),
        grid.n_patches,
        patch_len,
        d,
        &mut tokens,
    )?;
    Ok((tokens, grid))
}

/// Add a positional table to an assembled token sequence, in place.
///
/// `tokens` must already hold `n_prepended_tokens + grid.n_patches` rows of
/// `embed_dim` each, prepended rows first.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] when `embed_dim` is zero, when `tokens`
/// or `pos_embed` have the wrong length, or when the table row count does
/// not match the token count under [`PosEmbedPolicy::RequireExact`] (the
/// error names both counts and points at the interpolating policy, rather
/// than silently truncating the longer of the two).
pub fn vit_add_pos_embed(
    tokens: &mut [f32],
    grid: &PatchGrid,
    pos_embed: &[f32],
    embed_dim: usize,
    n_prepended_tokens: usize,
    policy: PosEmbedPolicy,
) -> Result<()> {
    if embed_dim == 0 {
        return Err(VokraError::InvalidArgument(
            "vit_add_pos_embed: embed_dim must be > 0".to_owned(),
        ));
    }
    let n_tokens = grid.n_tokens(n_prepended_tokens);
    check_len(tokens, n_tokens * embed_dim, "vit_add_pos_embed: tokens")?;
    if pos_embed.is_empty() || pos_embed.len() % embed_dim != 0 {
        return Err(VokraError::InvalidArgument(format!(
            "vit_add_pos_embed: pos_embed length {} is not a non-zero multiple of \
             embed_dim {embed_dim}",
            pos_embed.len()
        )));
    }
    let pos_rows = pos_embed.len() / embed_dim;

    match policy {
        PosEmbedPolicy::RequireExact => {
            if pos_rows != n_tokens {
                return Err(VokraError::InvalidArgument(format!(
                    "vit_add_pos_embed: PosEmbedPolicy::RequireExact needs one table row \
                     per token, but the table has {pos_rows} row(s) and the sequence has \
                     {n_tokens} token(s) ({n_prepended_tokens} prepended + \
                     {} patch, grid {}×{}). Either resize the table offline with the \
                     upstream resizer, or select \
                     PosEmbedPolicy::InterpolateGridBilinear and accept that it is a \
                     bilinear approximation of upstream's bicubic resize.",
                    grid.n_patches, grid.grid_h, grid.grid_w
                )));
            }
            for (slot, delta) in tokens.iter_mut().zip(pos_embed.iter()) {
                *slot += delta;
            }
        }
        PosEmbedPolicy::InterpolateGridBilinear {
            table_grid_h,
            table_grid_w,
        } => {
            if table_grid_h == 0 || table_grid_w == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "vit_add_pos_embed: InterpolateGridBilinear needs a non-empty table \
                     grid, got {table_grid_h}×{table_grid_w}"
                )));
            }
            let expected = n_prepended_tokens + table_grid_h * table_grid_w;
            if pos_rows != expected {
                return Err(VokraError::InvalidArgument(format!(
                    "vit_add_pos_embed: pos_embed holds {pos_rows} row(s) but the declared \
                     table grid {table_grid_h}×{table_grid_w} plus {n_prepended_tokens} \
                     prepended token(s) implies {expected}"
                )));
            }
            let prepended_len = n_prepended_tokens * embed_dim;
            for (slot, delta) in tokens.iter_mut().zip(pos_embed[..prepended_len].iter()) {
                *slot += delta;
            }
            let resized = bilinear_resize_grid(
                &pos_embed[prepended_len..],
                table_grid_h,
                table_grid_w,
                grid.grid_h,
                grid.grid_w,
                embed_dim,
            );
            for (slot, delta) in tokens[prepended_len..].iter_mut().zip(resized.iter()) {
                *slot += delta;
            }
        }
    }
    Ok(())
}

/// Collapse `[n_tokens, embed_dim]` hidden states into one `[embed_dim]`
/// vector under an explicitly chosen convention.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] when `embed_dim` is zero, when `tokens`
/// has the wrong length, when the prepended block is larger than the
/// sequence, when [`ViTPooling::PrependedToken`] names an index outside the
/// prepended block, or when [`ViTPooling::MeanPatchTokens`] is asked to
/// average an empty patch-token range.
pub fn vit_pool(
    tokens: &[f32],
    n_tokens: usize,
    embed_dim: usize,
    n_prepended_tokens: usize,
    pooling: ViTPooling,
) -> Result<Vec<f32>> {
    if embed_dim == 0 {
        return Err(VokraError::InvalidArgument(
            "vit_pool: embed_dim must be > 0".to_owned(),
        ));
    }
    check_len(tokens, n_tokens * embed_dim, "vit_pool: tokens")?;
    if n_prepended_tokens > n_tokens {
        return Err(VokraError::InvalidArgument(format!(
            "vit_pool: n_prepended_tokens ({n_prepended_tokens}) exceeds the sequence \
             length ({n_tokens})"
        )));
    }
    match pooling {
        ViTPooling::PrependedToken { index } => {
            if index >= n_prepended_tokens {
                return Err(VokraError::InvalidArgument(format!(
                    "vit_pool: ViTPooling::PrependedToken {{ index: {index} }} is outside \
                     the prepended block, which holds {n_prepended_tokens} token(s)"
                )));
            }
            Ok(tokens[index * embed_dim..(index + 1) * embed_dim].to_vec())
        }
        ViTPooling::MeanPatchTokens => {
            let n_patch = n_tokens - n_prepended_tokens;
            if n_patch == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "vit_pool: ViTPooling::MeanPatchTokens has no patch tokens to average \
                     ({n_tokens} token(s), all {n_prepended_tokens} prepended)"
                )));
            }
            let mut acc = vec![0.0f32; embed_dim];
            for row in tokens[n_prepended_tokens * embed_dim..].chunks(embed_dim) {
                for (slot, &value) in acc.iter_mut().zip(row.iter()) {
                    *slot += value;
                }
            }
            let inv = 1.0 / n_patch as f32;
            for slot in acc.iter_mut() {
                *slot *= inv;
            }
            Ok(acc)
        }
    }
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

/// `out = W · src + b` for one row; `w` is row-major `[out_dim, in_dim]`.
///
/// `b` is `None` for a bias-free linear. All lengths are the caller's
/// responsibility — every public entry point validates up front.
fn linear_row(
    src: &[f32],
    w: &[f32],
    b: Option<&[f32]>,
    out_dim: usize,
    in_dim: usize,
    out: &mut [f32],
) {
    for (o, slot) in out.iter_mut().enumerate() {
        let w_row = &w[o * in_dim..(o + 1) * in_dim];
        let dot: f32 = w_row.iter().zip(src.iter()).map(|(a, c)| a * c).sum();
        *slot = match b {
            Some(bias) => bias[o] + dot,
            None => dot,
        };
    }
    debug_assert_eq!(out.len(), out_dim);
}

/// In-place LayerNorm `y = (x - mean) / sqrt(var + eps) · γ + β` over one
/// token row, with a caller-supplied epsilon.
fn layer_norm_row(row: &mut [f32], gamma: &[f32], beta: &[f32], eps: f32) {
    let n = row.len() as f32;
    let mean = row.iter().sum::<f32>() / n;
    let var = row
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f32>()
        / n;
    let inv = 1.0 / (var + eps).sqrt();
    for ((slot, &g), &b) in row.iter_mut().zip(gamma.iter()).zip(beta.iter()) {
        *slot = (*slot - mean) * inv * g + b;
    }
}

/// Numerically stable row-wise softmax.
fn softmax_row(src: &[f32], dst: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &v in src {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for (slot, &s) in dst.iter_mut().zip(src.iter()) {
        let e = (s - max).exp();
        *slot = e;
        sum += e;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for slot in dst.iter_mut() {
            *slot *= inv;
        }
    }
}

/// Error function, Abramowitz & Stegun 7.1.26 (`|ε| ≤ 1.5e-7`).
///
/// Evaluated in `f64` so the published coefficients keep their full
/// precision; the caller rounds once to `f32`.
fn erf_f64(x: f64) -> f64 {
    const P: f64 = 0.327_591_1;
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + P * ax);
    let poly = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5))));
    sign * (1.0 - poly * (-ax * ax).exp())
}

/// GELU in the requested formulation. `gelu(0) == 0` for both kinds.
fn gelu(x: f32, kind: GeluKind) -> f32 {
    let xd = f64::from(x);
    let y = match kind {
        GeluKind::Erf => 0.5 * xd * (1.0 + erf_f64(xd / std::f64::consts::SQRT_2)),
        GeluKind::Tanh => {
            const C: f64 = 0.044_715;
            let inner = (2.0 / std::f64::consts::PI).sqrt() * (xd + C * xd * xd * xd);
            0.5 * xd * (1.0 + inner.tanh())
        }
    };
    y as f32
}

/// Bilinear resize of a `[src_h, src_w, channels]` row-major grid to
/// `[dst_h, dst_w, channels]`, with half-pixel sample centres (the
/// `align_corners=False` convention). Exact identity when the sizes match.
fn bilinear_resize_grid(
    src: &[f32],
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
    channels: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; dst_h * dst_w * channels];
    let scale_y = src_h as f32 / dst_h as f32;
    let scale_x = src_w as f32 / dst_w as f32;
    for y in 0..dst_h {
        let fy = (y as f32 + 0.5) * scale_y - 0.5;
        let fy0 = fy.floor();
        let wy = fy - fy0;
        let y0 = clamp_index(fy0, src_h);
        let y1 = clamp_index(fy0 + 1.0, src_h);
        for x in 0..dst_w {
            let fx = (x as f32 + 0.5) * scale_x - 0.5;
            let fx0 = fx.floor();
            let wx = fx - fx0;
            let x0 = clamp_index(fx0, src_w);
            let x1 = clamp_index(fx0 + 1.0, src_w);
            let dst_off = (y * dst_w + x) * channels;
            let off00 = (y0 * src_w + x0) * channels;
            let off01 = (y0 * src_w + x1) * channels;
            let off10 = (y1 * src_w + x0) * channels;
            let off11 = (y1 * src_w + x1) * channels;
            for ch in 0..channels {
                let top = (1.0 - wx) * src[off00 + ch] + wx * src[off01 + ch];
                let bottom = (1.0 - wx) * src[off10 + ch] + wx * src[off11 + ch];
                out[dst_off + ch] = (1.0 - wy) * top + wy * bottom;
            }
        }
    }
    out
}

/// Clamp an integral-valued `f32` coordinate into `[0, limit)`.
fn clamp_index(v: f32, limit: usize) -> usize {
    if v <= 0.0 {
        return 0;
    }
    let idx = v as usize;
    if idx >= limit { limit - 1 } else { idx }
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// Loud length check.
fn check_len(values: &[f32], expected: usize, tag: &str) -> Result<()> {
    if values.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "{tag}: expected length {expected}, got {}",
            values.len()
        )));
    }
    Ok(())
}

/// Loud finiteness check — reports the first offending index and value.
fn require_finite(values: &[f32], tag: &str) -> Result<()> {
    for (i, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(VokraError::InvalidArgument(format!(
                "{tag}: element {i} is not finite ({value})"
            )));
        }
    }
    Ok(())
}

/// Shape-validate one block against the resolved widths.
fn validate_block(block: &ViTBlockWeights, d: usize, mlp_dim: usize, index: usize) -> Result<()> {
    let dd = d * d;
    check_len(&block.ln1_gamma, d, &format!("block {index}: ln1_gamma"))?;
    check_len(&block.ln1_beta, d, &format!("block {index}: ln1_beta"))?;
    check_len(&block.ln2_gamma, d, &format!("block {index}: ln2_gamma"))?;
    check_len(&block.ln2_beta, d, &format!("block {index}: ln2_beta"))?;
    for (name, w) in [
        ("wq", &block.attn.wq),
        ("wk", &block.attn.wk),
        ("wv", &block.attn.wv),
        ("wo", &block.attn.wo),
    ] {
        check_len(w, dd, &format!("block {index}: attn.{name}"))?;
    }
    for (name, b) in [
        ("bq", &block.attn.bq),
        ("bk", &block.attn.bk),
        ("bv", &block.attn.bv),
        ("bo", &block.attn.bo),
    ] {
        if let Some(bias) = b {
            check_len(bias, d, &format!("block {index}: attn.{name}"))?;
        }
    }
    check_len(
        &block.mlp.w1,
        mlp_dim * d,
        &format!("block {index}: mlp.w1"),
    )?;
    check_len(
        &block.mlp.w2,
        d * mlp_dim,
        &format!("block {index}: mlp.w2"),
    )?;
    if let Some(bias) = &block.mlp.b1 {
        check_len(bias, mlp_dim, &format!("block {index}: mlp.b1"))?;
    }
    if let Some(bias) = &block.mlp.b2 {
        check_len(bias, d, &format!("block {index}: mlp.b2"))?;
    }
    Ok(())
}

/// Finiteness-validate every buffer in one block.
fn require_finite_block(block: &ViTBlockWeights, index: usize) -> Result<()> {
    let required: [(&str, &Vec<f32>); 8] = [
        ("ln1_gamma", &block.ln1_gamma),
        ("ln1_beta", &block.ln1_beta),
        ("ln2_gamma", &block.ln2_gamma),
        ("ln2_beta", &block.ln2_beta),
        ("attn.wq", &block.attn.wq),
        ("attn.wk", &block.attn.wk),
        ("attn.wv", &block.attn.wv),
        ("attn.wo", &block.attn.wo),
    ];
    for (name, values) in required {
        require_finite(values, &format!("block {index}: {name}"))?;
    }
    require_finite(&block.mlp.w1, &format!("block {index}: mlp.w1"))?;
    require_finite(&block.mlp.w2, &format!("block {index}: mlp.w2"))?;
    let optional: [(&str, &Option<Vec<f32>>); 6] = [
        ("attn.bq", &block.attn.bq),
        ("attn.bk", &block.attn.bk),
        ("attn.bv", &block.attn.bv),
        ("attn.bo", &block.attn.bo),
        ("mlp.b1", &block.mlp.b1),
        ("mlp.b2", &block.mlp.b2),
    ];
    for (name, maybe) in optional {
        if let Some(values) = maybe {
            require_finite(values, &format!("block {index}: {name}"))?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic uniform-ish source in `[-1, 1)`. No committed fixtures:
    /// every signal below is generated in-test from a fixed-seed LCG (Knuth /
    /// MMIX multiplier), so the tests reproduce on every platform.
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = (self.0 >> 33) as f64 / (1u64 << 30) as f64; // [0, 2)
            (u - 1.0) as f32
        }
    }

    fn noise(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = Lcg::new(seed);
        (0..n).map(|_| rng.next_f32()).collect()
    }

    fn attrs(embed_dim: usize, depth: usize, n_heads: usize, n_prepended: usize) -> ViTAttrs {
        ViTAttrs {
            embed_dim,
            depth,
            n_heads,
            mlp_ratio: 2.0,
            patch_h: 2,
            patch_w: 2,
            stride_h: 2,
            stride_w: 2,
            n_prepended_tokens: n_prepended,
            layer_norm_eps: 1e-6,
            gelu: GeluKind::Erf,
            pos_embed_policy: PosEmbedPolicy::RequireExact,
        }
    }

    /// Identity linear: row-major `[out, in]` with a unit diagonal. Requires
    /// `out == inp` to be a true identity.
    fn eye(n: usize) -> Vec<f32> {
        let mut w = vec![0.0f32; n * n];
        for i in 0..n {
            w[i * n + i] = 1.0;
        }
        w
    }

    /// A block whose two branches both contribute exactly zero, so the block
    /// is the identity on its input: the attention output projection and the
    /// MLP second projection are all-zero, which kills each branch after the
    /// residual add regardless of what the branch computed.
    fn zero_branch_block(a: &ViTAttrs) -> ViTBlockWeights {
        let d = a.embed_dim;
        let h = a.mlp_dim();
        ViTBlockWeights {
            ln1_gamma: vec![1.0; d],
            ln1_beta: vec![0.0; d],
            attn: ViTAttnWeights {
                wq: eye(d),
                bq: None,
                wk: eye(d),
                bk: None,
                wv: eye(d),
                bv: None,
                wo: vec![0.0; d * d],
                bo: None,
            },
            ln2_gamma: vec![1.0; d],
            ln2_beta: vec![0.0; d],
            mlp: ViTMlpWeights {
                w1: vec![0.0; h * d],
                b1: None,
                w2: vec![0.0; d * h],
                b2: None,
            },
        }
    }

    /// A block with random weights, for tests that need real mixing.
    fn random_block(a: &ViTAttrs, seed: u64) -> ViTBlockWeights {
        let d = a.embed_dim;
        let h = a.mlp_dim();
        let s = |k: u64, n: usize| -> Vec<f32> {
            noise(n, seed.wrapping_add(k))
                .iter()
                .map(|v| v * 0.2)
                .collect()
        };
        ViTBlockWeights {
            ln1_gamma: vec![1.0; d],
            ln1_beta: vec![0.0; d],
            attn: ViTAttnWeights {
                wq: s(1, d * d),
                bq: None,
                wk: s(2, d * d),
                bk: None,
                wv: s(3, d * d),
                bv: None,
                wo: s(4, d * d),
                bo: None,
            },
            ln2_gamma: vec![1.0; d],
            ln2_beta: vec![0.0; d],
            mlp: ViTMlpWeights {
                w1: s(5, h * d),
                b1: None,
                w2: s(6, d * h),
                b2: None,
            },
        }
    }

    fn weights(a: &ViTAttrs, n_pos_rows: usize, blocks: Vec<ViTBlockWeights>) -> ViTWeights {
        let d = a.embed_dim;
        ViTWeights {
            patch_embed: PatchEmbedWeights {
                proj_w: noise(d * a.patch_h * a.patch_w, 0x9A7C),
                proj_b: None,
            },
            prepended_tokens: vec![0.25; a.n_prepended_tokens * d],
            pos_embed: vec![0.0; n_pos_rows * d],
            blocks,
            final_ln_gamma: vec![1.0; d],
            final_ln_beta: vec![0.0; d],
        }
    }

    // ---- patch grid arithmetic ------------------------------------------

    #[test]
    fn patch_grid_non_overlapping_is_exact_division() {
        let a = attrs(8, 1, 2, 1);
        // 8 mels / patch 2 stride 2 -> 4 rows; 10 frames -> 5 cols.
        let g = patch_grid(8, 10, &a).expect("grid");
        assert_eq!((g.grid_h, g.grid_w, g.n_patches), (4, 5, 20));
        assert_eq!((g.dropped_rows, g.dropped_cols), (0, 0));
        assert_eq!(g.n_tokens(1), 21);
    }

    #[test]
    fn patch_grid_overlapping_stride_produces_more_patches_than_the_tiling() {
        let mut a = attrs(8, 1, 2, 0);
        a.stride_h = 1;
        a.stride_w = 1;
        // (8 - 2)/1 + 1 = 7 rows, (10 - 2)/1 + 1 = 9 cols.
        let g = patch_grid(8, 10, &a).expect("grid");
        assert_eq!((g.grid_h, g.grid_w, g.n_patches), (7, 9, 63));
        // Overlapping must yield strictly more patches than the disjoint
        // tiling of the same plane (4 x 5 = 20) — this is the property that
        // makes a transposed stride/patch mix-up visible.
        assert!(g.n_patches > 20);
        assert_eq!((g.dropped_rows, g.dropped_cols), (0, 0));
    }

    #[test]
    fn patch_grid_reports_the_ragged_tail_instead_of_swallowing_it() {
        let a = attrs(8, 1, 2, 0);
        // 9 mels with patch 2 / stride 2: (9 - 2)/2 + 1 = 4 rows covering
        // rows 0..8, so exactly 1 mel bin is uncovered. Same for 11 frames.
        let g = patch_grid(9, 11, &a).expect("grid");
        assert_eq!((g.grid_h, g.grid_w), (4, 5));
        assert_eq!(
            (g.dropped_rows, g.dropped_cols),
            (1, 1),
            "a dropped tail must be reported, not hidden"
        );
    }

    #[test]
    fn patch_grid_refuses_a_plane_smaller_than_one_patch() {
        let a = attrs(8, 1, 2, 0);
        let Err(err) = patch_grid(1, 10, &a) else {
            panic!("a 1-mel plane cannot hold a 2-mel patch and must be refused");
        };
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument, got {err:?}"
        );
    }

    // ---- attribute validation -------------------------------------------

    #[test]
    fn attrs_reject_a_head_count_that_does_not_divide_the_width() {
        let mut a = attrs(8, 1, 3, 0); // 8 / 3 does not divide
        a.n_heads = 3;
        let Err(err) = a.validate() else {
            panic!("n_heads must divide embed_dim");
        };
        assert!(matches!(err, VokraError::InvalidArgument(_)), "{err:?}");
    }

    #[test]
    fn attrs_reject_a_zero_axis() {
        for mutate in [
            (|a: &mut ViTAttrs| a.embed_dim = 0) as fn(&mut ViTAttrs),
            |a: &mut ViTAttrs| a.depth = 0,
            |a: &mut ViTAttrs| a.n_heads = 0,
            |a: &mut ViTAttrs| a.patch_h = 0,
            |a: &mut ViTAttrs| a.patch_w = 0,
            |a: &mut ViTAttrs| a.stride_h = 0,
            |a: &mut ViTAttrs| a.stride_w = 0,
        ] {
            let mut a = attrs(8, 1, 2, 0);
            mutate(&mut a);
            assert!(a.validate().is_err(), "a zero axis must be refused: {a:?}");
        }
    }

    // ---- encoder construction -------------------------------------------

    #[test]
    fn encoder_rejects_a_block_count_that_disagrees_with_depth() {
        let a = attrs(8, 3, 2, 1);
        let w = weights(&a, 1 + 20, vec![zero_branch_block(&a)]); // 1 block, depth 3
        let Err(err) = ViTEncoder::new(a, w) else {
            panic!("depth 3 with 1 block must be refused");
        };
        assert!(matches!(err, VokraError::InvalidArgument(_)), "{err:?}");
    }

    // ---- positional embedding -------------------------------------------

    #[test]
    fn require_exact_pos_embed_length_mismatch_is_loud() {
        let a = attrs(8, 1, 2, 1);
        // Correct table would be 1 prepended + 20 patches = 21 rows.
        let w = weights(&a, 5, vec![zero_branch_block(&a)]);
        let enc = ViTEncoder::new(a, w).expect("encoder");
        let mel = noise(8 * 10, 7);
        let Err(err) = enc.forward(&mel, 8, 10) else {
            panic!("a 5-row table against 21 tokens must be refused under RequireExact");
        };
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument, got {err:?}"
        );
    }

    #[test]
    fn correct_pos_embed_length_is_accepted() {
        let a = attrs(8, 1, 2, 1);
        let w = weights(&a, 21, vec![zero_branch_block(&a)]);
        let enc = ViTEncoder::new(a, w).expect("encoder");
        let mel = noise(8 * 10, 7);
        let (hidden, grid) = enc.forward(&mel, 8, 10).expect("forward");
        assert_eq!(grid.n_patches, 20);
        assert_eq!(hidden.len(), 21 * 8);
    }

    // ---- block semantics -------------------------------------------------

    #[test]
    fn a_zero_branch_block_is_the_identity_up_to_the_final_norm() {
        // Both branches are annihilated by their output projection, so the
        // block leaves its input untouched and only the final LayerNorm acts.
        // Bound is derived, not tuned: the sole arithmetic is one LayerNorm
        // over 8 f32 accumulations, so agreement with an independently
        // computed LayerNorm should sit at f32 rounding, ~1e-6 relative on
        // unit-scale values. 1e-5 is ~10x that.
        let a = attrs(8, 1, 2, 0);
        let w = weights(&a, 1, vec![zero_branch_block(&a)]);
        let enc = ViTEncoder::new(a, w).expect("encoder");

        let n_tokens = 4;
        let tokens = noise(n_tokens * 8, 11);
        let out = enc.encode_tokens(&tokens, n_tokens).expect("encode");

        for t in 0..n_tokens {
            let row = &tokens[t * 8..(t + 1) * 8];
            let mean = row.iter().sum::<f32>() / 8.0;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 8.0;
            let inv = 1.0 / (var + 1e-6).sqrt();
            for i in 0..8 {
                let want = (row[i] - mean) * inv;
                let got = out[t * 8 + i];
                assert!(
                    (want - got).abs() < 1e-5,
                    "token {t} dim {i}: want {want}, got {got}"
                );
            }
        }
    }

    #[test]
    fn the_mlp_branch_actually_contributes() {
        // Regression guard with teeth: an earlier revision of this module
        // shipped an `mlp()` stub returning all zeros, which compiled and
        // silently removed the whole MLP branch. Zeroing ONLY the attention
        // output projection leaves the MLP as the sole branch, so if the MLP
        // is inert the block collapses to the identity — exactly the bug.
        let a = attrs(8, 1, 2, 0);
        let mut block = random_block(&a, 0xBEEF);
        block.attn.wo = vec![0.0; 8 * 8];
        let w = weights(&a, 1, vec![block]);
        let enc = ViTEncoder::new(a, w).expect("encoder");

        let n_tokens = 4;
        let tokens = noise(n_tokens * 8, 13);
        let out = enc.encode_tokens(&tokens, n_tokens).expect("encode");

        // Compare against the same encoder with the MLP also annihilated.
        let a2 = attrs(8, 1, 2, 0);
        let mut inert = random_block(&a2, 0xBEEF);
        inert.attn.wo = vec![0.0; 8 * 8];
        inert.mlp.w2 = vec![0.0; 8 * a2.mlp_dim()];
        let w2 = weights(&a2, 1, vec![inert]);
        let enc2 = ViTEncoder::new(a2, w2).expect("encoder");
        let out2 = enc2.encode_tokens(&tokens, n_tokens).expect("encode");

        let delta: f32 = out
            .iter()
            .zip(out2.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        assert!(
            delta > 1e-3,
            "the MLP branch must change the output; max |delta| was {delta}"
        );
    }

    #[test]
    fn attention_mixes_across_tokens() {
        // Without positional embedding the encoder body is permutation
        // EQUIVARIANT, so permuting the input tokens must permute the output
        // rows identically. That property holds only if attention genuinely
        // couples tokens through a shared softmax — a per-token-independent
        // implementation would pass this too, so the companion assertion
        // below checks that a token's output actually depends on its
        // neighbours.
        let a = attrs(8, 1, 2, 0);
        let w = weights(&a, 1, vec![random_block(&a, 0xC0FFEE)]);
        let enc = ViTEncoder::new(a, w).expect("encoder");

        let n_tokens = 3;
        let tokens = noise(n_tokens * 8, 17);
        let out = enc.encode_tokens(&tokens, n_tokens).expect("encode");

        // Swap tokens 0 and 2.
        let mut swapped = tokens.clone();
        for i in 0..8 {
            swapped.swap(i, 2 * 8 + i);
        }
        let out_swapped = enc.encode_tokens(&swapped, n_tokens).expect("encode");
        for i in 0..8 {
            assert!(
                (out[i] - out_swapped[2 * 8 + i]).abs() < 1e-5,
                "equivariance broken at dim {i}"
            );
        }

        // Now perturb ONLY token 1 and require token 0's output to move:
        // that is coupling, which a per-token map could not produce.
        //
        // The perturbation touches a SINGLE dimension. Adding a constant to
        // every dimension of a row would be erased by the pre-norm
        // LayerNorm's mean subtraction (LayerNorm is shift- and
        // scale-invariant), so such a "perturbation" reaches attention as a
        // no-op and this assertion would fail against a perfectly correct
        // encoder.
        let mut perturbed = tokens.clone();
        perturbed[8] += 2.0;
        let out_perturbed = enc.encode_tokens(&perturbed, n_tokens).expect("encode");
        let moved: f32 = (0..8)
            .map(|i| (out[i] - out_perturbed[i]).abs())
            .fold(0.0, f32::max);
        assert!(
            moved > 1e-4,
            "token 0 must depend on token 1 through attention; max |delta| was {moved}"
        );
    }

    // ---- pooling ---------------------------------------------------------

    #[test]
    fn pooling_conventions_differ_and_both_are_reachable() {
        let a = attrs(8, 1, 2, 2);
        let w = weights(&a, 1, vec![random_block(&a, 0xF00D)]);
        let enc = ViTEncoder::new(a, w).expect("encoder");

        let n_tokens = 6; // 2 prepended + 4 patch
        let tokens = noise(n_tokens * 8, 19);
        let hidden = enc.encode_tokens(&tokens, n_tokens).expect("encode");

        let cls = enc
            .pool(&hidden, n_tokens, ViTPooling::PrependedToken { index: 0 })
            .expect("cls pool");
        let dist = enc
            .pool(&hidden, n_tokens, ViTPooling::PrependedToken { index: 1 })
            .expect("dist pool");
        let mean = enc
            .pool(&hidden, n_tokens, ViTPooling::MeanPatchTokens)
            .expect("mean pool");

        assert_eq!(cls.len(), 8);
        assert_eq!(dist.len(), 8);
        assert_eq!(mean.len(), 8);
        assert_eq!(
            cls,
            hidden[0..8].to_vec(),
            "index 0 is the first prepended row"
        );
        assert_ne!(
            cls, dist,
            "the two prepended tokens must be distinguishable"
        );
        assert_ne!(
            cls, mean,
            "CLS and mean-over-patches are different conventions, not aliases"
        );
    }

    #[test]
    fn pooling_rejects_a_prepended_index_that_does_not_exist() {
        let a = attrs(8, 1, 2, 1);
        let w = weights(&a, 1, vec![zero_branch_block(&a)]);
        let enc = ViTEncoder::new(a, w).expect("encoder");
        let hidden = noise(3 * 8, 23);
        let Err(err) = enc.pool(&hidden, 3, ViTPooling::PrependedToken { index: 4 }) else {
            panic!("index 4 with 1 prepended token must be refused");
        };
        assert!(matches!(err, VokraError::InvalidArgument(_)), "{err:?}");
    }

    // ---- input validation ------------------------------------------------

    #[test]
    fn shape_and_finiteness_violations_are_loud() {
        let a = attrs(8, 1, 2, 0);
        let w = weights(&a, 20, vec![zero_branch_block(&a)]);
        let enc = ViTEncoder::new(a, w).expect("encoder");

        // Wrong plane length.
        assert!(enc.forward(&noise(8 * 10 - 1, 29), 8, 10).is_err());
        // Non-finite sample.
        let mut mel = noise(8 * 10, 31);
        mel[5] = f32::NAN;
        assert!(enc.forward(&mel, 8, 10).is_err());
        // Zero token count.
        assert!(enc.encode_tokens(&[], 0).is_err());
        // Token buffer inconsistent with n_tokens.
        assert!(enc.encode_tokens(&noise(3 * 8 - 1, 37), 3).is_err());
    }

    // ---- determinism -----------------------------------------------------

    #[test]
    fn same_input_twice_is_bit_identical() {
        let a = attrs(8, 2, 2, 1);
        let w = weights(&a, 21, vec![random_block(&a, 0xA1), random_block(&a, 0xA2)]);
        let enc = ViTEncoder::new(a, w).expect("encoder");
        let mel = noise(8 * 10, 41);
        let (first, g1) = enc.forward(&mel, 8, 10).expect("forward");
        let (second, g2) = enc.forward(&mel, 8, 10).expect("forward");
        assert_eq!(g1, g2);
        assert_eq!(first, second, "the encoder must be deterministic");
    }
}
