//! Plain BERT encoder — clean-room per Devlin et al. 2018
//! (arXiv:1810.04805, "BERT: Pre-training of Deep Bidirectional
//! Transformers for Language Understanding").
//!
//! This is intentionally **arch-different** from
//! [`crate::deberta_v2`] / [`crate::deberta_v3`]:
//!
//! - **Embeddings**: sum of three learned tables (word + learned
//!   absolute position + token_type) followed by LayerNorm.
//!   DeBERTa v2/v3 use disentangled relative position instead.
//! - **Self-attention**: standard `softmax(Q·K^T / sqrt(d_head)) · V`.
//!   No disentangled attention, no relative position bucket, no P2C/C2P
//!   terms.
//! - **Layer order**: **post-norm** — LayerNorm is applied *after* the
//!   residual add (`LN(x + sublayer(x))`), matching HF `BertSelfOutput`
//!   and `BertOutput`. DeBERTa v2/v3 in this crate use pre-norm.
//! - **Activation**: exact GELU (erf-based), matching HF BERT's default
//!   `hidden_act = "gelu"` which is `torch.nn.functional.gelu(approximate="none")`.
//!
//! Consumer today: `hfl/chinese-roberta-wwm-ext-large`
//! (`BertForMaskedLM`, Apache-2.0) as the ZH BERT branch of SBV2 v2.
//! The module is arch-independent from its consumer — WP-19 wires it
//! into `SbV2Model`.
//!
//! # References (permissive only)
//!
//! - Devlin, Chang, Lee, Toutanova 2018 (arXiv:1810.04805)
//! - google-research/bert (Apache-2.0): reference implementation
//! - HuggingFace transformers `modeling_bert.py` (Apache-2.0)
//! - Vaswani et al. 2017 (arXiv:1706.03762) for the underlying
//!   scaled dot-product attention
//! - Abramowitz & Stegun 1964 §7.1.26 for the erf approximation used
//!   in exact GELU
//!
//! # NOT REFERENCED
//!
//! - github.com/litagin02/Style-Bert-VITS2 (AGPL-3.0)
//! - github.com/fishaudio/Bert-VITS2 (AGPL-3.0)
//! - Any AGPL derivative of the above.

use crate::backend::{gather_head, linear_with_backend, transpose_rows, BertBackendOps};
use crate::deberta_v2::LayerNorm;
use vokra_core::gguf::GgufFile;
use vokra_core::{Result as VokraResult, VokraError};

/// Static hyper-parameters for a plain BERT encoder.
///
/// Names mirror HuggingFace `BertConfig` for legibility. All fields are
/// required at construction time — there are no runtime defaults that
/// silently paper over a missing checkpoint value (FR-EX-08).
#[derive(Debug, Clone)]
pub struct BertConfig {
    /// Size of the word-piece / character vocabulary.
    pub vocab_size: usize,
    /// Model dimension `d_model` (a.k.a. `hidden_size` in HF).
    pub hidden_size: usize,
    /// Number of stacked [`BertLayer`] blocks.
    pub num_hidden_layers: usize,
    /// Number of attention heads per layer. `hidden_size % num_attention_heads == 0`.
    pub num_attention_heads: usize,
    /// FFN inner dimension (`intermediate_size` in HF; 4·hidden for the
    /// canonical BERT config, but the loader honors whatever the
    /// checkpoint actually stores).
    pub intermediate_size: usize,
    /// Length of the learned absolute position table.
    pub max_position_embeddings: usize,
    /// Length of the learned token-type ("segment") table. Standard BERT
    /// uses 2 (segment A / B). NSP-free models set 1.
    pub type_vocab_size: usize,
    /// LayerNorm `epsilon`. HF BERT default is `1e-12`.
    pub layer_norm_eps: f32,
}

/// Exact GELU: `x · 0.5 · (1 + erf(x / sqrt(2)))`, matching PyTorch's
/// `nn.functional.gelu(approximate="none")` — which is what
/// HuggingFace BERT (`hidden_act = "gelu"`) invokes.
///
/// `erf` is computed with the Abramowitz & Stegun §7.1.26 rational
/// approximation (max absolute error ≈ 1.5·10⁻⁷), well below the atol
/// bounds this crate operates at.
#[inline]
pub(crate) fn gelu_exact(x: f32) -> f32 {
    x * 0.5 * (1.0 + erf_approx(x * core::f32::consts::FRAC_1_SQRT_2))
}

/// Abramowitz & Stegun §7.1.26 erf approximation.
///
/// `erf(x) ≈ sign(x) · [1 - (a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵) · exp(-x²)]`
/// with `t = 1 / (1 + p·|x|)`. Max absolute error ≈ 1.5·10⁻⁷.
#[inline]
fn erf_approx(x: f32) -> f32 {
    // Coefficients from Abramowitz & Stegun 1964 (public-domain reference),
    // truncated to `f32` mantissa precision (clippy `excessive_precision`).
    const A1: f32 = 0.254_829_6;
    const A2: f32 = -0.284_496_74;
    const A3: f32 = 1.421_413_7;
    const A4: f32 = -1.453_152;
    const A5: f32 = 1.061_405_4;
    const P: f32 = 0.327_591_1;

    let sign = if x < 0.0 { -1.0_f32 } else { 1.0_f32 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + P * ax);
    // WP-10 (2026-08-10): route erf's exp through vokra_math for cross-plat
    // determinism within Vokra (BertBaseEncoder GELU, per-hidden-dim hot site).
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * vokra_math::exp(-ax * ax);
    sign * y
}

/// Row-major `y[i,o] = sum_d x[i,d] * w[o,d] + b[o]`.
///
/// `x`: [n_rows × d_in], `w`: [d_out × d_in], `b`: [d_out]. Returns
/// [n_rows × d_out]. Mirrors the naive triple loop used in
/// [`crate::deberta_v2`] — correctness first, blocking / SIMD is a
/// Stage B follow-up if profiling shows a hot path.
fn matmul_bias_rm(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    n_rows: usize,
    d_in: usize,
    d_out: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), n_rows * d_in, "matmul_bias_rm: x shape mismatch");
    assert_eq!(w.len(), d_out * d_in, "matmul_bias_rm: w shape mismatch");
    assert_eq!(b.len(), d_out, "matmul_bias_rm: b shape mismatch");
    let mut y = vec![0.0_f32; n_rows * d_out];
    for i in 0..n_rows {
        for o in 0..d_out {
            let mut acc = b[o];
            for d in 0..d_in {
                acc += x[i * d_in + d] * w[o * d_in + d];
            }
            y[i * d_out + o] = acc;
        }
    }
    y
}

/// BERT input embeddings.
///
/// `emb_i = LN(word_embed[id_i] + position_embed[i] + token_type_embed[type_i])`
///
/// Matches HF `BertEmbeddings.forward` (sum-then-norm order). All three
/// tables are `[N × hidden_size]` row-major.
pub struct BertEmbeddings {
    token_embed: Vec<f32>,      // [vocab_size, hidden_size]
    position_embed: Vec<f32>,   // [max_position_embeddings, hidden_size]
    token_type_embed: Vec<f32>, // [type_vocab_size, hidden_size]
    layer_norm: LayerNorm,
    hidden_size: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    type_vocab_size: usize,
}

impl BertEmbeddings {
    /// Constructor. Shape-checks the three tables against the declared
    /// dimensions; a mismatch panics (FR-EX-08 loud-fail).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        token_embed: Vec<f32>,
        position_embed: Vec<f32>,
        token_type_embed: Vec<f32>,
        layer_norm: LayerNorm,
        hidden_size: usize,
        vocab_size: usize,
        max_position_embeddings: usize,
        type_vocab_size: usize,
    ) -> Self {
        assert_eq!(
            token_embed.len(),
            vocab_size * hidden_size,
            "token_embed shape [{vocab_size} × {hidden_size}] mismatch"
        );
        assert_eq!(
            position_embed.len(),
            max_position_embeddings * hidden_size,
            "position_embed shape [{max_position_embeddings} × {hidden_size}] mismatch"
        );
        assert_eq!(
            token_type_embed.len(),
            type_vocab_size * hidden_size,
            "token_type_embed shape [{type_vocab_size} × {hidden_size}] mismatch"
        );
        Self {
            token_embed,
            position_embed,
            token_type_embed,
            layer_norm,
            hidden_size,
            vocab_size,
            max_position_embeddings,
            type_vocab_size,
        }
    }

    /// `token_ids` → `[seq_len × hidden_size]` row-major hidden states.
    ///
    /// `token_type_ids = None` is treated as all-zero (single-segment),
    /// matching HF's `token_type_ids or torch.zeros(...)` default.
    ///
    /// Panics loud on: token id ≥ `vocab_size`, token_type_id ≥
    /// `type_vocab_size`, seq_len > `max_position_embeddings`, or a
    /// `token_type_ids` slice whose length ≠ token_ids' length
    /// (FR-EX-08).
    pub fn forward(&self, token_ids: &[u32], token_type_ids: Option<&[u32]>) -> Vec<f32> {
        let seq_len = token_ids.len();
        assert!(
            seq_len <= self.max_position_embeddings,
            "seq_len {seq_len} > max_position_embeddings {}",
            self.max_position_embeddings
        );
        if let Some(t) = token_type_ids {
            assert_eq!(
                t.len(),
                seq_len,
                "token_type_ids length {} != token_ids length {seq_len}",
                t.len()
            );
        }
        let d = self.hidden_size;
        let mut hidden = vec![0.0_f32; seq_len * d];
        for (i, &tok_id) in token_ids.iter().enumerate() {
            let tok = tok_id as usize;
            assert!(
                tok < self.vocab_size,
                "token id {tok} out of vocab {}",
                self.vocab_size
            );
            let tt = token_type_ids.map(|t| t[i] as usize).unwrap_or(0);
            assert!(
                tt < self.type_vocab_size,
                "token_type_id {tt} out of type_vocab {}",
                self.type_vocab_size
            );
            let dst_row = i * d;
            let tok_row = tok * d;
            let pos_row = i * d;
            let tt_row = tt * d;
            for k in 0..d {
                hidden[dst_row + k] = self.token_embed[tok_row + k]
                    + self.position_embed[pos_row + k]
                    + self.token_type_embed[tt_row + k];
            }
        }
        self.layer_norm.forward(&hidden, seq_len, d)
    }

    fn forward_with_backend(
        &self,
        backend: &dyn BertBackendOps,
        token_ids: &[u32],
        token_type_ids: Option<&[u32]>,
    ) -> VokraResult<Vec<f32>> {
        let seq_len = token_ids.len();
        assert!(seq_len <= self.max_position_embeddings);
        if let Some(types) = token_type_ids {
            assert_eq!(types.len(), seq_len);
        }
        let d = self.hidden_size;
        let mut hidden = vec![0.0; seq_len * d];
        for (position, &token_id) in token_ids.iter().enumerate() {
            let token = token_id as usize;
            assert!(token < self.vocab_size);
            let token_type = token_type_ids
                .map(|types| types[position] as usize)
                .unwrap_or(0);
            assert!(token_type < self.type_vocab_size);
            for channel in 0..d {
                hidden[position * d + channel] = self.token_embed[token * d + channel]
                    + self.position_embed[position * d + channel]
                    + self.token_type_embed[token_type * d + channel];
            }
        }
        self.layer_norm
            .forward_with_backend(backend, &hidden, seq_len, d)
    }
}

/// Standard multi-head self-attention.
///
/// `Attention(Q, K, V) = softmax(Q·K^T / sqrt(head_dim)) · V`
///
/// Q/K/V projections are `[hidden_size × hidden_size]` row-major, split
/// into `num_heads` heads of dimension `head_dim = hidden_size / num_heads`.
/// The output projection is applied by [`BertSelfOutput`], **not** here
/// — this struct returns the raw concatenated multi-head result, so
/// that `BertSelfOutput` can perform `LN(dense(attn) + residual)` in
/// one place.
pub struct BertSelfAttention {
    wq: Vec<f32>,
    bq: Vec<f32>,
    wk: Vec<f32>,
    bk: Vec<f32>,
    wv: Vec<f32>,
    bv: Vec<f32>,
    hidden_size: usize,
    num_heads: usize,
    head_dim: usize,
}

impl BertSelfAttention {
    /// Constructor. Panics loud (FR-EX-08) when weight shapes or
    /// `hidden_size % num_heads` are wrong.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        wq: Vec<f32>,
        bq: Vec<f32>,
        wk: Vec<f32>,
        bk: Vec<f32>,
        wv: Vec<f32>,
        bv: Vec<f32>,
        hidden_size: usize,
        num_heads: usize,
    ) -> Self {
        assert!(num_heads > 0, "num_heads must be > 0");
        assert_eq!(
            hidden_size % num_heads,
            0,
            "hidden_size ({hidden_size}) must divide num_heads ({num_heads})"
        );
        let head_dim = hidden_size / num_heads;
        for (name, w) in [("wq", &wq), ("wk", &wk), ("wv", &wv)] {
            assert_eq!(
                w.len(),
                hidden_size * hidden_size,
                "{name} shape [{hidden_size} × {hidden_size}] mismatch"
            );
        }
        for (name, b) in [("bq", &bq), ("bk", &bk), ("bv", &bv)] {
            assert_eq!(
                b.len(),
                hidden_size,
                "{name} shape [{hidden_size}] mismatch"
            );
        }
        Self {
            wq,
            bq,
            wk,
            bk,
            wv,
            bv,
            hidden_size,
            num_heads,
            head_dim,
        }
    }

    /// `hidden`: `[seq_len × hidden_size]` row-major.
    /// Returns `[seq_len × hidden_size]` row-major.
    pub fn forward(&self, hidden: &[f32], seq_len: usize) -> Vec<f32> {
        let d = self.hidden_size;
        assert_eq!(hidden.len(), seq_len * d, "hidden shape mismatch");
        // 1. Project to Q, K, V.
        let q = matmul_bias_rm(hidden, &self.wq, &self.bq, seq_len, d, d);
        let k = matmul_bias_rm(hidden, &self.wk, &self.bk, seq_len, d, d);
        let v = matmul_bias_rm(hidden, &self.wv, &self.bv, seq_len, d, d);

        // 2. Multi-head attention.
        // WP-10 (2026-08-10): attention scale sqrt through vokra_math for
        // cross-plat determinism within Vokra.
        let scale = 1.0 / vokra_math::sqrt(self.head_dim as f32);
        let mut out = vec![0.0_f32; seq_len * d];
        let mut scores = vec![0.0_f32; seq_len * seq_len];

        for head in 0..self.num_heads {
            let head_off = head * self.head_dim;

            // 2a. Compute attention scores per (q_i, k_j).
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut s = 0.0_f32;
                    for h in 0..self.head_dim {
                        s += q[i * d + head_off + h] * k[j * d + head_off + h];
                    }
                    scores[i * seq_len + j] = s * scale;
                }
            }

            // 2b. Row-wise softmax.
            for i in 0..seq_len {
                let row_start = i * seq_len;
                let row_end = row_start + seq_len;
                let max_v = scores[row_start..row_end]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0_f32;
                for j in 0..seq_len {
                    // WP-10 (2026-08-10): softmax exp through vokra_math
                    // for cross-plat determinism within Vokra
                    // (BertBaseEncoder, plain BERT variant for ZH path).
                    let e = vokra_math::exp(scores[row_start + j] - max_v);
                    scores[row_start + j] = e;
                    sum += e;
                }
                let inv = 1.0 / sum;
                for j in 0..seq_len {
                    scores[row_start + j] *= inv;
                }
            }

            // 2c. Weighted V sum → per-head output slice.
            for i in 0..seq_len {
                for h in 0..self.head_dim {
                    let mut acc = 0.0_f32;
                    for j in 0..seq_len {
                        acc += scores[i * seq_len + j] * v[j * d + head_off + h];
                    }
                    out[i * d + head_off + h] = acc;
                }
            }
        }

        out
    }

    fn forward_with_backend(
        &self,
        backend: &dyn BertBackendOps,
        hidden: &[f32],
        seq_len: usize,
    ) -> VokraResult<Vec<f32>> {
        let d = self.hidden_size;
        assert_eq!(hidden.len(), seq_len * d);
        let q = linear_with_backend(backend, hidden, &self.wq, Some(&self.bq), seq_len, d, d)?;
        let k = linear_with_backend(backend, hidden, &self.wk, Some(&self.bk), seq_len, d, d)?;
        let v = linear_with_backend(backend, hidden, &self.wv, Some(&self.bv), seq_len, d, d)?;
        let scale = 1.0 / vokra_math::sqrt(self.head_dim as f32);
        let mut output = vec![0.0; seq_len * d];

        for head in 0..self.num_heads {
            let head_offset = head * self.head_dim;
            let q_head = gather_head(&q, seq_len, d, head_offset, self.head_dim);
            let k_head = gather_head(&k, seq_len, d, head_offset, self.head_dim);
            let v_head = gather_head(&v, seq_len, d, head_offset, self.head_dim);
            let mut scores = linear_with_backend(
                backend,
                &q_head,
                &k_head,
                None,
                seq_len,
                self.head_dim,
                seq_len,
            )?;
            for score in &mut scores {
                *score *= scale;
            }
            let mut probabilities = vec![0.0; scores.len()];
            backend.softmax_f32(&scores, &mut probabilities, seq_len, seq_len)?;
            let value_out_in = transpose_rows(&v_head, seq_len, self.head_dim);
            let context = linear_with_backend(
                backend,
                &probabilities,
                &value_out_in,
                None,
                seq_len,
                seq_len,
                self.head_dim,
            )?;
            for position in 0..seq_len {
                output[position * d + head_offset..position * d + head_offset + self.head_dim]
                    .copy_from_slice(
                        &context[position * self.head_dim..(position + 1) * self.head_dim],
                    );
            }
        }
        Ok(output)
    }
}

/// Attention output projection + post-norm residual, matching HF
/// `BertSelfOutput.forward`:
///
/// `LN(dense(attn_out) + residual)`
///
/// Dropout is elided (inference-only). Dense weights `[hidden × hidden]`.
pub struct BertSelfOutput {
    dense: Vec<f32>,
    bias: Vec<f32>,
    layer_norm: LayerNorm,
    hidden_size: usize,
}

impl BertSelfOutput {
    /// Constructor. Panics loud (FR-EX-08) on shape mismatch.
    pub fn new(dense: Vec<f32>, bias: Vec<f32>, layer_norm: LayerNorm, hidden_size: usize) -> Self {
        assert_eq!(
            dense.len(),
            hidden_size * hidden_size,
            "dense shape [{hidden_size} × {hidden_size}] mismatch"
        );
        assert_eq!(
            bias.len(),
            hidden_size,
            "bias shape [{hidden_size}] mismatch"
        );
        Self {
            dense,
            bias,
            layer_norm,
            hidden_size,
        }
    }

    /// `attn_out`, `residual`: `[seq_len × hidden_size]` row-major.
    /// Returns `[seq_len × hidden_size]` row-major.
    pub fn forward(&self, attn_out: &[f32], residual: &[f32], seq_len: usize) -> Vec<f32> {
        let d = self.hidden_size;
        assert_eq!(attn_out.len(), seq_len * d, "attn_out shape mismatch");
        assert_eq!(residual.len(), seq_len * d, "residual shape mismatch");
        let dense_out = matmul_bias_rm(attn_out, &self.dense, &self.bias, seq_len, d, d);
        let mut added = vec![0.0_f32; seq_len * d];
        for i in 0..(seq_len * d) {
            added[i] = dense_out[i] + residual[i];
        }
        self.layer_norm.forward(&added, seq_len, d)
    }

    fn forward_with_backend(
        &self,
        backend: &dyn BertBackendOps,
        attn_out: &[f32],
        residual: &[f32],
        seq_len: usize,
    ) -> VokraResult<Vec<f32>> {
        let d = self.hidden_size;
        let dense = linear_with_backend(
            backend,
            attn_out,
            &self.dense,
            Some(&self.bias),
            seq_len,
            d,
            d,
        )?;
        let added: Vec<f32> = dense
            .iter()
            .zip(residual)
            .map(|(value, skip)| value + skip)
            .collect();
        self.layer_norm
            .forward_with_backend(backend, &added, seq_len, d)
    }
}

/// FFN inner linear + GELU. Analogous to HF `BertIntermediate`.
///
/// Dense weight `[intermediate_size × hidden_size]`.
pub struct BertIntermediate {
    dense: Vec<f32>,
    bias: Vec<f32>,
    hidden_size: usize,
    intermediate_size: usize,
}

impl BertIntermediate {
    /// Constructor. Panics loud (FR-EX-08) on shape mismatch.
    pub fn new(
        dense: Vec<f32>,
        bias: Vec<f32>,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Self {
        assert_eq!(
            dense.len(),
            intermediate_size * hidden_size,
            "dense shape [{intermediate_size} × {hidden_size}] mismatch"
        );
        assert_eq!(
            bias.len(),
            intermediate_size,
            "bias shape [{intermediate_size}] mismatch"
        );
        Self {
            dense,
            bias,
            hidden_size,
            intermediate_size,
        }
    }

    /// `hidden`: `[seq_len × hidden_size]`. Returns `[seq_len × intermediate_size]`.
    pub fn forward(&self, hidden: &[f32], seq_len: usize) -> Vec<f32> {
        let d_in = self.hidden_size;
        let d_out = self.intermediate_size;
        assert_eq!(hidden.len(), seq_len * d_in, "hidden shape mismatch");
        // Fused linear + GELU to save an alloc.
        let mut y = vec![0.0_f32; seq_len * d_out];
        for i in 0..seq_len {
            for o in 0..d_out {
                let mut acc = self.bias[o];
                for k in 0..d_in {
                    acc += hidden[i * d_in + k] * self.dense[o * d_in + k];
                }
                y[i * d_out + o] = gelu_exact(acc);
            }
        }
        y
    }

    fn forward_with_backend(
        &self,
        backend: &dyn BertBackendOps,
        hidden: &[f32],
        seq_len: usize,
    ) -> VokraResult<Vec<f32>> {
        let projected = linear_with_backend(
            backend,
            hidden,
            &self.dense,
            Some(&self.bias),
            seq_len,
            self.hidden_size,
            self.intermediate_size,
        )?;
        let mut activated = vec![0.0; projected.len()];
        backend.gelu_f32(&projected, &mut activated)?;
        Ok(activated)
    }
}

/// FFN outer linear + post-norm residual, matching HF `BertOutput.forward`:
///
/// `LN(dense(ffn_inner) + residual)`
///
/// Dense weight `[hidden × intermediate_size]`.
pub struct BertOutput {
    dense: Vec<f32>,
    bias: Vec<f32>,
    layer_norm: LayerNorm,
    hidden_size: usize,
    intermediate_size: usize,
}

impl BertOutput {
    /// Constructor. Panics loud (FR-EX-08) on shape mismatch.
    pub fn new(
        dense: Vec<f32>,
        bias: Vec<f32>,
        layer_norm: LayerNorm,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Self {
        assert_eq!(
            dense.len(),
            hidden_size * intermediate_size,
            "dense shape [{hidden_size} × {intermediate_size}] mismatch"
        );
        assert_eq!(
            bias.len(),
            hidden_size,
            "bias shape [{hidden_size}] mismatch"
        );
        Self {
            dense,
            bias,
            layer_norm,
            hidden_size,
            intermediate_size,
        }
    }

    /// `ffn_out`: `[seq_len × intermediate_size]`.
    /// `residual`: `[seq_len × hidden_size]`.
    /// Returns `[seq_len × hidden_size]`.
    pub fn forward(&self, ffn_out: &[f32], residual: &[f32], seq_len: usize) -> Vec<f32> {
        let d_in = self.intermediate_size;
        let d_out = self.hidden_size;
        assert_eq!(ffn_out.len(), seq_len * d_in, "ffn_out shape mismatch");
        assert_eq!(residual.len(), seq_len * d_out, "residual shape mismatch");
        let dense_out = matmul_bias_rm(ffn_out, &self.dense, &self.bias, seq_len, d_in, d_out);
        let mut added = vec![0.0_f32; seq_len * d_out];
        for i in 0..(seq_len * d_out) {
            added[i] = dense_out[i] + residual[i];
        }
        self.layer_norm.forward(&added, seq_len, d_out)
    }

    fn forward_with_backend(
        &self,
        backend: &dyn BertBackendOps,
        ffn_out: &[f32],
        residual: &[f32],
        seq_len: usize,
    ) -> VokraResult<Vec<f32>> {
        let projected = linear_with_backend(
            backend,
            ffn_out,
            &self.dense,
            Some(&self.bias),
            seq_len,
            self.intermediate_size,
            self.hidden_size,
        )?;
        let added: Vec<f32> = projected
            .iter()
            .zip(residual)
            .map(|(value, skip)| value + skip)
            .collect();
        self.layer_norm
            .forward_with_backend(backend, &added, seq_len, self.hidden_size)
    }
}

/// One transformer block: attention + FFN with post-norm residuals.
///
/// The forward pass (matching HF `BertLayer.forward` less dropout) is:
///
/// ```text
/// attn_out    = attention(hidden)
/// hidden      = self_output(attn_out, hidden)          # LN(dense(attn) + hidden)
/// ffn_inner   = intermediate(hidden)                    # linear + GELU
/// hidden      = output(ffn_inner, hidden)              # LN(dense(ffn) + hidden)
/// ```
pub struct BertLayer {
    pub attention: BertSelfAttention,
    pub self_output: BertSelfOutput,
    pub intermediate: BertIntermediate,
    pub output: BertOutput,
}

impl BertLayer {
    /// One transformer forward pass, `[seq_len × hidden_size]` in / out.
    pub fn forward(&self, hidden: &[f32], seq_len: usize) -> Vec<f32> {
        let attn = self.attention.forward(hidden, seq_len);
        let h_attn = self.self_output.forward(&attn, hidden, seq_len);
        let ff = self.intermediate.forward(&h_attn, seq_len);
        self.output.forward(&ff, &h_attn, seq_len)
    }

    fn forward_with_backend(
        &self,
        backend: &dyn BertBackendOps,
        hidden: &[f32],
        seq_len: usize,
    ) -> VokraResult<Vec<f32>> {
        let attention = self
            .attention
            .forward_with_backend(backend, hidden, seq_len)?;
        let attention_hidden = self
            .self_output
            .forward_with_backend(backend, &attention, hidden, seq_len)?;
        let intermediate =
            self.intermediate
                .forward_with_backend(backend, &attention_hidden, seq_len)?;
        self.output
            .forward_with_backend(backend, &intermediate, &attention_hidden, seq_len)
    }

    /// Deterministic synthetic-weight layer for structure tests only.
    /// Weights are varied per-index (see [`seeded_weights`]) so that
    /// LayerNorm produces non-degenerate output — a constant row would
    /// collapse to zero variance and hide real routing bugs.
    #[doc(hidden)]
    pub fn synthetic_for_test(cfg: &BertConfig) -> Self {
        Self::synthetic_for_test_with_seed(cfg, 0)
    }

    /// Same as [`Self::synthetic_for_test`] but with a caller-controlled
    /// seed, so multiple layers in a stack each get their own weights.
    #[doc(hidden)]
    pub fn synthetic_for_test_with_seed(cfg: &BertConfig, seed: u32) -> Self {
        let d = cfg.hidden_size;
        let ff = cfg.intermediate_size;
        Self {
            attention: BertSelfAttention::new(
                seeded_weights(seed.wrapping_add(1), d * d),
                vec![0.0_f32; d],
                seeded_weights(seed.wrapping_add(2), d * d),
                vec![0.0_f32; d],
                seeded_weights(seed.wrapping_add(3), d * d),
                vec![0.0_f32; d],
                d,
                cfg.num_attention_heads,
            ),
            self_output: BertSelfOutput::new(
                seeded_weights(seed.wrapping_add(4), d * d),
                vec![0.0_f32; d],
                LayerNorm::new(vec![1.0_f32; d], vec![0.0_f32; d], cfg.layer_norm_eps),
                d,
            ),
            intermediate: BertIntermediate::new(
                seeded_weights(seed.wrapping_add(5), ff * d),
                vec![0.0_f32; ff],
                d,
                ff,
            ),
            output: BertOutput::new(
                seeded_weights(seed.wrapping_add(6), d * ff),
                vec![0.0_f32; d],
                LayerNorm::new(vec![1.0_f32; d], vec![0.0_f32; d], cfg.layer_norm_eps),
                d,
                ff,
            ),
        }
    }
}

/// Deterministic bounded PRNG for synthetic-test weights.
///
/// `sin((i * 0.017) + (seed * 0.023)) * 0.05` — bounded in
/// `[-0.05, 0.05]`, varies per-index (so LayerNorm produces
/// non-degenerate rows) and per-seed (so each layer / each embed table
/// sees a different pattern and downstream tests can distinguish them).
/// Not a cryptographic PRNG — the only requirement is determinism plus
/// enough spatial variation to defeat LN row-collapse.
#[doc(hidden)]
fn seeded_weights(seed: u32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let phase = ((i as f32) * 0.017 + (seed as f32) * 0.023).sin();
            phase * 0.05
        })
        .collect()
}

/// Full plain-BERT encoder: [`BertEmbeddings`] → N × [`BertLayer`].
///
/// The last hidden state (i.e. the output of the final layer) is
/// returned as a flat `[seq_len × hidden_size]` row-major `Vec<f32>`.
/// Downstream consumers (SBV2 v2's ZH branch, etc.) slice or reshape
/// as needed.
///
/// This struct is deliberately **stand-alone** — it does not touch
/// `SbV2Model`. WP-19 wires it into the ZH branch of SBV2.
pub struct BertBaseEncoder {
    embeddings: BertEmbeddings,
    layers: Vec<BertLayer>,
    hidden_size: usize,
}

impl BertBaseEncoder {
    /// Constructor. The caller is responsible for having supplied a
    /// consistent set of tables and layers; individual sub-components
    /// have already shape-checked themselves.
    pub fn new(embeddings: BertEmbeddings, layers: Vec<BertLayer>, hidden_size: usize) -> Self {
        Self {
            embeddings,
            layers,
            hidden_size,
        }
    }

    /// Forward pass.
    ///
    /// - `token_ids`: `[seq_len]` u32 token ids.
    /// - `token_type_ids`: `Some([seq_len])` u32 segment ids, or `None`
    ///   for all-zero (single-segment, matching HF's default).
    ///
    /// Returns the last hidden state as `[seq_len × hidden_size]` flat
    /// row-major.
    pub fn forward(&self, token_ids: &[u32], token_type_ids: Option<&[u32]>) -> Vec<f32> {
        let seq_len = token_ids.len();
        let mut hidden = self.embeddings.forward(token_ids, token_type_ids);
        for layer in &self.layers {
            hidden = layer.forward(&hidden, seq_len);
        }
        hidden
    }

    /// Backend-dispatched sibling of [`Self::forward`].
    ///
    /// Embedding lookup and residual additions remain host layout/control
    /// work. Every learned projection, attention reduction, softmax, GELU and
    /// LayerNorm is delegated to the supplied single-backend implementation.
    pub fn forward_with_backend(
        &self,
        backend: &dyn BertBackendOps,
        token_ids: &[u32],
        token_type_ids: Option<&[u32]>,
    ) -> VokraResult<Vec<f32>> {
        let seq_len = token_ids.len();
        let mut hidden =
            self.embeddings
                .forward_with_backend(backend, token_ids, token_type_ids)?;
        for layer in &self.layers {
            hidden = layer.forward_with_backend(backend, &hidden, seq_len)?;
        }
        Ok(hidden)
    }

    /// Model dimension (`hidden_size` in the config).
    pub fn d_model(&self) -> usize {
        self.hidden_size
    }

    /// Builds a `BertBaseEncoder` with deterministic synthetic weights.
    /// Structure / shape / determinism tests only — no real checkpoint
    /// is involved. Embed tables are varied per-row (per-id / per-pos /
    /// per-type) so downstream tests can distinguish e.g. `type_id = 0`
    /// from `type_id = 1` and LN does not collapse constant rows.
    #[doc(hidden)]
    pub fn synthetic_for_test(cfg: &BertConfig) -> Self {
        let d = cfg.hidden_size;
        let emb = BertEmbeddings::new(
            seeded_weights(100, cfg.vocab_size * d),
            seeded_weights(200, cfg.max_position_embeddings * d),
            seeded_weights(300, cfg.type_vocab_size * d),
            LayerNorm::new(vec![1.0_f32; d], vec![0.0_f32; d], cfg.layer_norm_eps),
            d,
            cfg.vocab_size,
            cfg.max_position_embeddings,
            cfg.type_vocab_size,
        );
        let layers = (0..cfg.num_hidden_layers)
            // Distinct seed per layer so stacks are non-degenerate.
            .map(|i| BertLayer::synthetic_for_test_with_seed(cfg, 1_000 + i as u32 * 10))
            .collect();
        Self::new(emb, layers, d)
    }

    /// Loads a `BertBaseEncoder` from a GGUF file written by the
    /// SBV2-v2 converter (or any Vokra converter that follows the
    /// same schema).
    ///
    /// # Metadata keys (`vokra.bert_base.*`)
    ///
    /// All keys are **required** — this loader intentionally has no
    /// silent defaults (FR-EX-08). A missing hparam surfaces as
    /// `VokraError::ModelLoad` so the caller cannot end up running with
    /// a mismatched shape without noticing.
    ///
    /// - `vokra.bert_base.n_layers` (u32)
    /// - `vokra.bert_base.hidden` (u32)
    /// - `vokra.bert_base.heads` (u32)
    /// - `vokra.bert_base.ffn` (u32) — intermediate_size
    /// - `vokra.bert_base.vocab` (u32)
    /// - `vokra.bert_base.max_pos` (u32) — max_position_embeddings
    /// - `vokra.bert_base.type_vocab` (u32)
    /// - `vokra.bert_base.layer_norm_eps` (f32, optional; default `1e-12`)
    ///
    /// # Tensor names
    ///
    /// - `bert_base.embeddings.word_embed`
    /// - `bert_base.embeddings.position_embed`
    /// - `bert_base.embeddings.token_type_embed`
    /// - `bert_base.embeddings.layernorm.gamma`
    /// - `bert_base.embeddings.layernorm.beta`
    /// - Per layer `i` (0..n_layers):
    ///   - `bert_base.encoder.layer.<i>.attention.query.{weight,bias}`
    ///   - `bert_base.encoder.layer.<i>.attention.key.{weight,bias}`
    ///   - `bert_base.encoder.layer.<i>.attention.value.{weight,bias}`
    ///   - `bert_base.encoder.layer.<i>.attention.output.dense.{weight,bias}`
    ///   - `bert_base.encoder.layer.<i>.attention.output.layernorm.{gamma,beta}`
    ///   - `bert_base.encoder.layer.<i>.intermediate.dense.{weight,bias}`
    ///   - `bert_base.encoder.layer.<i>.output.dense.{weight,bias}`
    ///   - `bert_base.encoder.layer.<i>.output.layernorm.{gamma,beta}`
    pub fn from_gguf(g: &GgufFile) -> Result<Self, VokraError> {
        let require_u32 = |key: &str| -> Result<u32, VokraError> {
            g.get(key)
                .and_then(|v| v.as_u64())
                .map(|u| u as u32)
                .ok_or_else(|| VokraError::ModelLoad(format!("missing GGUF metadata key: {key}")))
        };

        let n_layers = require_u32("vokra.bert_base.n_layers")? as usize;
        let hidden_size = require_u32("vokra.bert_base.hidden")? as usize;
        let num_heads = require_u32("vokra.bert_base.heads")? as usize;
        let intermediate_size = require_u32("vokra.bert_base.ffn")? as usize;
        let vocab_size = require_u32("vokra.bert_base.vocab")? as usize;
        let max_position_embeddings = require_u32("vokra.bert_base.max_pos")? as usize;
        let type_vocab_size = require_u32("vokra.bert_base.type_vocab")? as usize;
        // eps is optional — HF BERT's default 1e-12 is the historically
        // stable value and is safe when the checkpoint omits it.
        let layer_norm_eps = g
            .get("vokra.bert_base.layer_norm_eps")
            .and_then(|v| v.as_f64())
            .map(|f| f as f32)
            .unwrap_or(1e-12);

        if num_heads == 0 || hidden_size % num_heads != 0 {
            return Err(VokraError::ModelLoad(format!(
                "vokra.bert_base: hidden ({hidden_size}) not divisible by heads ({num_heads})"
            )));
        }

        let load = |name: &str| -> Result<Vec<f32>, VokraError> {
            g.tensor_f32(name)
                .map_err(|e| VokraError::ModelLoad(format!("{name}: {e}")))
        };

        // Embeddings.
        let word_embed = load("bert_base.embeddings.word_embed")?;
        let position_embed = load("bert_base.embeddings.position_embed")?;
        let token_type_embed = load("bert_base.embeddings.token_type_embed")?;
        let emb_ln = LayerNorm::new(
            load("bert_base.embeddings.layernorm.gamma")?,
            load("bert_base.embeddings.layernorm.beta")?,
            layer_norm_eps,
        );
        let embeddings = BertEmbeddings::new(
            word_embed,
            position_embed,
            token_type_embed,
            emb_ln,
            hidden_size,
            vocab_size,
            max_position_embeddings,
            type_vocab_size,
        );

        // Layers.
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("bert_base.encoder.layer.{i}");
            let attention = BertSelfAttention::new(
                load(&format!("{p}.attention.query.weight"))?,
                load(&format!("{p}.attention.query.bias"))?,
                load(&format!("{p}.attention.key.weight"))?,
                load(&format!("{p}.attention.key.bias"))?,
                load(&format!("{p}.attention.value.weight"))?,
                load(&format!("{p}.attention.value.bias"))?,
                hidden_size,
                num_heads,
            );
            let self_output = BertSelfOutput::new(
                load(&format!("{p}.attention.output.dense.weight"))?,
                load(&format!("{p}.attention.output.dense.bias"))?,
                LayerNorm::new(
                    load(&format!("{p}.attention.output.layernorm.gamma"))?,
                    load(&format!("{p}.attention.output.layernorm.beta"))?,
                    layer_norm_eps,
                ),
                hidden_size,
            );
            let intermediate = BertIntermediate::new(
                load(&format!("{p}.intermediate.dense.weight"))?,
                load(&format!("{p}.intermediate.dense.bias"))?,
                hidden_size,
                intermediate_size,
            );
            let output = BertOutput::new(
                load(&format!("{p}.output.dense.weight"))?,
                load(&format!("{p}.output.dense.bias"))?,
                LayerNorm::new(
                    load(&format!("{p}.output.layernorm.gamma"))?,
                    load(&format!("{p}.output.layernorm.beta"))?,
                    layer_norm_eps,
                ),
                hidden_size,
                intermediate_size,
            );
            layers.push(BertLayer {
                attention,
                self_output,
                intermediate,
                output,
            });
        }

        Ok(Self::new(embeddings, layers, hidden_size))
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn erf_approx_matches_known_values() {
        // Reference values (double-precision):
        //   erf(0) = 0, erf(1) ≈ 0.84270079, erf(-1) ≈ -0.84270079.
        // A&S §7.1.26 max abs error ≈ 1.5e-7 — well below our 1e-4 assertion.
        assert!(erf_approx(0.0_f32).abs() < 1e-6);
        assert!((erf_approx(1.0_f32) - 0.842_700_8).abs() < 1e-5);
        assert!((erf_approx(-1.0_f32) + 0.842_700_8).abs() < 1e-5);
    }

    #[test]
    fn gelu_exact_at_zero_is_zero() {
        assert!(gelu_exact(0.0_f32).abs() < 1e-6);
    }

    #[test]
    fn gelu_exact_saturates_for_large_positive() {
        // gelu(x) → x as x → +∞; at x=4 we're within 1% of x.
        let x = 4.0_f32;
        assert!((gelu_exact(x) - x).abs() < 0.01);
    }

    #[test]
    fn gelu_exact_vanishes_for_large_negative() {
        // gelu(x) → 0 as x → -∞. At x = -4 the true value is
        // -4 · 0.5 · (1 + erf(-2√2)) ≈ -1.27·10⁻⁴, so we assert
        // |gelu(-4)| < 1·10⁻³ (three orders of magnitude below input).
        let g = gelu_exact(-4.0_f32);
        assert!(g.abs() < 1e-3, "gelu(-4) = {g}");
    }

    #[test]
    fn matmul_bias_rm_matches_hand_computation() {
        // x = [[1, 2]], w = [[10, 20], [30, 40]], b = [1, 2]
        // Expected y = [[1*10 + 2*20 + 1, 1*30 + 2*40 + 2]] = [[51, 112]]
        let y = matmul_bias_rm(&[1.0, 2.0], &[10.0, 20.0, 30.0, 40.0], &[1.0, 2.0], 1, 2, 2);
        assert_eq!(y, vec![51.0, 112.0]);
    }
}
