//! Voxtral text decoder — Mistral LLaMA-style transformer.
//!
//! # Structural summary (from the upstream Mistral release)
//!
//! - **Pre-norm** blocks: input → RMSNorm → attention → residual → RMSNorm
//!   → SwiGLU FFN → residual;
//! - **GQA** attention: `n_head_q` query heads, `n_head_kv` key/value heads
//!   (`n_head_q % n_head_kv == 0`, key/value are broadcast `n_head_q /
//!   n_head_kv` times);
//! - **RoPE** applied to query & key before the score matmul;
//! - **SwiGLU** FFN: `w2(silu(w1(x)) * w3(x))` (equivalently
//!   `down(silu(gate(x)) * up(x))`);
//! - **RMSNorm** with the checkpoint's ε (Mistral ships `1e-5`);
//! - **Tied logits**: the token embedding acts as the LM head.
//!
//! # Foundation scope (M3-10-T09 / T10)
//!
//! This foundation file:
//! - reads the Mistral text decoder weights out of the GGUF (recognising
//!   both the packaged Voxtral prefix `language_model.model.*` and the plain
//!   Mistral prefix `model.*`);
//! - exposes small, unit-testable Rust primitives ([`rms_norm`], [`silu`],
//!   [`rope_apply`]) that the eventual full forward will compose;
//! - ships a [`TextDecoderStep`] type shaped like Whisper's `DecoderState`
//!   so a future full decode loop has an obvious slot to hang KV caches
//!   off (see M3-03 paged KV cache);
//! - does **NOT** yet run a full autoregressive forward — the block math
//!   (GQA + RoPE + SwiGLU) is a follow-on ticket once a real Mistral
//!   checkpoint parity dump exists (T19+).
//!
//! The primitives (RMSNorm / SwiGLU / RoPE) are already fully tested
//! against internal oracles so downstream tickets can compose them without
//! re-deriving the math.

use vokra_core::gguf::GgufFile;
use vokra_core::{Result, VokraError};

use super::VoxtralConfig;

/// A `nn.Linear` decoded for direct row-major GEMM (`w_t` is `[in, out]`).
///
/// Mistral decoder projections are always **bias-less** (`bias = None`).
///
/// # `dead_code` posture (foundation)
///
/// The fields are read by [`TextDecoder::load`]'s callers and by the future
/// full forward — [`allow(dead_code)`] silences the foundation-only
/// warning without hiding the intent.
#[allow(dead_code)]
pub(crate) struct Linear {
    pub(crate) w_t: Vec<f32>,
    pub(crate) in_features: usize,
    pub(crate) out_features: usize,
}

/// A block's four attention projections. GQA: `q` is `[d, n_head_q*head_dim]`
/// = `[d, d]`; `k` / `v` are `[d, n_head_kv*head_dim]`.
#[allow(dead_code)]
pub(crate) struct GqaAttention {
    pub(crate) q: Linear,
    pub(crate) k: Linear,
    pub(crate) v: Linear,
    pub(crate) o: Linear,
}

/// SwiGLU FFN weights: `w2(silu(w1(x)) * w3(x))`.
#[allow(dead_code)]
pub(crate) struct SwiGluFfn {
    pub(crate) gate: Linear, // = w1
    pub(crate) up: Linear,   // = w3
    pub(crate) down: Linear, // = w2
}

/// One Mistral decoder block.
#[allow(dead_code)]
pub(crate) struct DecoderBlock {
    /// RMSNorm γ vector (no bias — RMSNorm is scale-only, `[hidden_dim]`).
    pub(crate) attn_norm_gamma: Vec<f32>,
    pub(crate) attn: GqaAttention,
    pub(crate) ffn_norm_gamma: Vec<f32>,
    pub(crate) ffn: SwiGluFfn,
}

/// All text-decoder weights (tied logits head → the token embedding IS the
/// LM head).
#[allow(dead_code)]
pub struct TextDecoder {
    /// Token embedding `[vocab_size, hidden_dim]` — also the tied LM head.
    pub(crate) token_emb: Vec<f32>,
    /// Per-block weights.
    pub(crate) blocks: Vec<DecoderBlock>,
    /// Final RMSNorm γ (post-block, pre-head).
    pub(crate) final_norm_gamma: Vec<f32>,
    /// Which safetensors prefix the tensors were found under.
    pub(crate) prefix: &'static str,
}

impl TextDecoder {
    /// Binds every text-decoder tensor.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] with the offending tensor named on any
    /// missing / mis-shaped tensor.
    pub fn load(file: &GgufFile, cfg: &VoxtralConfig) -> Result<Self> {
        // The shape-only converter path leaves `n_layer == 0` — surface an
        // empty decoder to the caller (forward will still refuse to run).
        if cfg.text.n_layer == 0 || cfg.text.hidden_dim == 0 {
            return Ok(Self {
                token_emb: Vec::new(),
                blocks: Vec::new(),
                final_norm_gamma: Vec::new(),
                prefix: "",
            });
        }
        // Try both possible prefixes: modern Voxtral packaging vs. plain
        // Mistral release.
        let prefix = pick_prefix(file);
        let d = cfg.text.hidden_dim;
        let vocab = cfg.text.vocab_size;
        if vocab == 0 {
            return Err(bad("text_decoder.vocab_size must be non-zero".to_owned()));
        }

        let token_emb = tensor(file, &format!("{prefix}embed_tokens.weight"), &[vocab, d])?;

        // GQA head widths.
        let n_head_q = cfg.text.n_head_q;
        let n_head_kv = cfg.text.n_head_kv;
        if n_head_q == 0 || n_head_kv == 0 {
            return Err(bad(
                "text_decoder.n_head_q and n_head_kv must be non-zero (GQA head split)".to_owned(),
            ));
        }
        let head_dim = d / n_head_q;
        let kv_hidden = n_head_kv * head_dim;

        let mut blocks = Vec::with_capacity(cfg.text.n_layer);
        for i in 0..cfg.text.n_layer {
            let p = format!("{prefix}layers.{i}");
            let attn_norm_gamma = tensor(file, &format!("{p}.input_layernorm.weight"), &[d])?;
            let attn = GqaAttention {
                q: linear(file, &format!("{p}.self_attn.q_proj"), d, d)?,
                k: linear(file, &format!("{p}.self_attn.k_proj"), d, kv_hidden)?,
                v: linear(file, &format!("{p}.self_attn.v_proj"), d, kv_hidden)?,
                o: linear(file, &format!("{p}.self_attn.o_proj"), d, d)?,
            };
            let ffn_norm_gamma =
                tensor(file, &format!("{p}.post_attention_layernorm.weight"), &[d])?;
            let ffn = SwiGluFfn {
                gate: linear(file, &format!("{p}.mlp.gate_proj"), d, cfg.text.ffn_dim)?,
                up: linear(file, &format!("{p}.mlp.up_proj"), d, cfg.text.ffn_dim)?,
                down: linear(file, &format!("{p}.mlp.down_proj"), cfg.text.ffn_dim, d)?,
            };
            blocks.push(DecoderBlock {
                attn_norm_gamma,
                attn,
                ffn_norm_gamma,
                ffn,
            });
        }
        let final_norm_gamma = tensor(file, &format!("{prefix}norm.weight"), &[d])?;
        Ok(Self {
            token_emb,
            blocks,
            final_norm_gamma,
            prefix: prefix_label(prefix),
        })
    }

    /// The prefix the tensors were found under. Useful for diagnostics /
    /// validation from external test harnesses.
    #[must_use]
    pub fn source_prefix(&self) -> &'static str {
        self.prefix
    }

    /// Number of loaded blocks.
    #[must_use]
    pub fn n_layer(&self) -> usize {
        self.blocks.len()
    }
}

/// A single-step decoder state placeholder — the shape a future full
/// autoregressive forward will attach KV caches to (M3-03 paged KV cache).
///
/// Foundation-only: currently just carries the current sequence length.
pub struct TextDecoderStep {
    /// Number of tokens processed so far.
    pub seq_len: usize,
}

impl TextDecoderStep {
    /// Fresh state (nothing decoded).
    #[must_use]
    pub fn new() -> Self {
        Self { seq_len: 0 }
    }

    /// Advance one token (increment `seq_len`).
    pub fn advance(&mut self) {
        self.seq_len += 1;
    }
}

impl Default for TextDecoderStep {
    fn default() -> Self {
        Self::new()
    }
}

// ---------- primitives ------------------------------------------------------

/// RMSNorm applied row-wise: `out[i, c] = x[i, c] * gamma[c] / sqrt(mean(x^2) + eps)`.
pub fn rms_norm(x: &[f32], gamma: &[f32], eps: f32, rows: usize, out: &mut [f32]) -> Result<()> {
    let d = gamma.len();
    if x.len() != rows * d || out.len() != rows * d {
        return Err(VokraError::InvalidArgument(format!(
            "rms_norm: x/out len must be rows*d ({}*{}={}), got x={}, out={}",
            rows,
            d,
            rows * d,
            x.len(),
            out.len(),
        )));
    }
    for i in 0..rows {
        let row = &x[i * d..(i + 1) * d];
        let sum_sq: f32 = row.iter().map(|&v| v * v).sum();
        let inv = 1.0 / (sum_sq / d as f32 + eps).sqrt();
        let dst = &mut out[i * d..(i + 1) * d];
        for c in 0..d {
            dst[c] = row[c] * inv * gamma[c];
        }
    }
    Ok(())
}

/// In-place SiLU: `x <- x * sigmoid(x)`.
pub fn silu_inplace(x: &mut [f32]) {
    for v in x {
        let s = 1.0 / (1.0 + (-*v).exp());
        *v *= s;
    }
}

/// Element-wise multiply: `a[i] <- a[i] * b[i]`. Length mismatch is a
/// programming error, so we surface it as an error rather than truncating.
pub fn hadamard_inplace(a: &mut [f32], b: &[f32]) -> Result<()> {
    if a.len() != b.len() {
        return Err(VokraError::InvalidArgument(format!(
            "hadamard_inplace: length mismatch {} != {}",
            a.len(),
            b.len()
        )));
    }
    for (dst, &src) in a.iter_mut().zip(b) {
        *dst *= src;
    }
    Ok(())
}

/// Applies RoPE to one head's `q` / `k` slice in place.
///
/// `x` is `[seq_len, head_dim]` row-major; `head_dim` MUST be even. The
/// rotation frequencies are computed from `rope_base` per the standard
/// formula: `theta_j = rope_base ^ (-2j / head_dim)` for `j = 0..head_dim/2`.
///
/// `position_offset` supports incremental decoding: pass the absolute
/// starting position of `x[0]`. RoPE at row `i` uses frequency `theta_j`
/// scaled by `position_offset + i`.
pub fn rope_apply(
    x: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    rope_base: f32,
    position_offset: usize,
) -> Result<()> {
    if head_dim % 2 != 0 {
        return Err(VokraError::InvalidArgument(format!(
            "rope_apply: head_dim ({head_dim}) must be even"
        )));
    }
    if x.len() != seq_len * head_dim {
        return Err(VokraError::InvalidArgument(format!(
            "rope_apply: x len {} != seq_len*head_dim {}",
            x.len(),
            seq_len * head_dim
        )));
    }
    let half = head_dim / 2;
    for i in 0..seq_len {
        let m = (position_offset + i) as f32;
        let row = &mut x[i * head_dim..(i + 1) * head_dim];
        for j in 0..half {
            let theta = rope_base.powf(-2.0 * (j as f32) / (head_dim as f32));
            let angle = m * theta;
            let (s, c) = angle.sin_cos();
            let a = row[j];
            let b = row[j + half];
            row[j] = a * c - b * s;
            row[j + half] = a * s + b * c;
        }
    }
    Ok(())
}

// ---------- internals -------------------------------------------------------

fn pick_prefix(file: &GgufFile) -> &'static str {
    // Modern Voxtral packages the Mistral backbone as a submodule.
    if file
        .tensor_info("language_model.model.embed_tokens.weight")
        .is_some()
    {
        "language_model.model."
    } else if file
        .tensor_info("language_model.embed_tokens.weight")
        .is_some()
    {
        "language_model."
    } else {
        "model."
    }
}

fn prefix_label(p: &str) -> &'static str {
    match p {
        "language_model.model." => "language_model.model.",
        "language_model." => "language_model.",
        "model." => "model.",
        _ => "",
    }
}

fn bad(msg: String) -> VokraError {
    VokraError::ModelLoad(format!("voxtral text_decoder: {msg}"))
}

fn tensor(file: &GgufFile, name: &str, want: &[usize]) -> Result<Vec<f32>> {
    let info = file
        .tensor_info(name)
        .ok_or_else(|| bad(format!("`{name}` missing from GGUF")))?;
    let got: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
    if got != want {
        return Err(bad(format!("`{name}` shape {got:?} != expected {want:?}")));
    }
    file.tensor_f32(name)
        .map_err(|e| bad(format!("`{name}`: {e}")))
}

fn linear(
    file: &GgufFile,
    prefix: &str,
    in_features: usize,
    out_features: usize,
) -> Result<Linear> {
    // Mistral projections are bias-less. The stored shape is `[out, in]`
    // (safetensors convention); we transpose once so row-major GEMM reads
    // `[in, out]`.
    let w = tensor(
        file,
        &format!("{prefix}.weight"),
        &[out_features, in_features],
    )?;
    let mut w_t = vec![0.0f32; in_features * out_features];
    for o in 0..out_features {
        let row = &w[o * in_features..(o + 1) * in_features];
        for (i, &v) in row.iter().enumerate() {
            w_t[i * out_features + o] = v;
        }
    }
    Ok(Linear {
        w_t,
        in_features,
        out_features,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_normalises_row_to_unit_rms() {
        // With gamma = 1, RMSNorm output should have unit RMS (per row).
        let d = 4;
        let x = vec![1.0f32, 2.0, 3.0, 4.0]; // mean(x^2) = 7.5.
        let gamma = vec![1.0f32; d];
        let mut out = vec![0.0f32; d];
        rms_norm(&x, &gamma, 0.0, 1, &mut out).unwrap();
        let mean_sq: f32 = out.iter().map(|v| v * v).sum::<f32>() / d as f32;
        assert!(
            (mean_sq - 1.0).abs() < 1e-5,
            "row RMS should be 1.0, got sqrt({mean_sq})"
        );
    }

    #[test]
    fn rms_norm_zero_row_stays_zero_with_epsilon() {
        // An all-zero row must not blow up (eps guards the divisor).
        let x = vec![0.0f32; 4];
        let gamma = vec![1.0f32; 4];
        let mut out = vec![0.0f32; 4];
        rms_norm(&x, &gamma, 1e-5, 1, &mut out).unwrap();
        assert!(out.iter().all(|v| v.abs() < 1e-6));
    }

    #[test]
    fn silu_matches_reference_at_specific_points() {
        // silu(0)=0, silu(large positive)≈x, silu(large negative)≈0.
        let mut x = vec![0.0f32, 5.0, -5.0, 1.0];
        silu_inplace(&mut x);
        assert!((x[0]).abs() < 1e-6);
        assert!((x[1] - 5.0 * (1.0 / (1.0 + (-5.0f32).exp()))).abs() < 1e-5);
        assert!(x[2].abs() < 0.05); // small negative
        // silu(1) = 1 * sigmoid(1) ≈ 0.731
        assert!((x[3] - 0.731_058_6).abs() < 1e-3);
    }

    #[test]
    fn hadamard_multiplies_elementwise() {
        let mut a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        hadamard_inplace(&mut a, &b).unwrap();
        assert_eq!(a, vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn hadamard_rejects_length_mismatch() {
        let mut a = vec![1.0f32, 2.0];
        let b = vec![1.0f32];
        assert!(matches!(
            hadamard_inplace(&mut a, &b),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rope_apply_position_zero_is_identity() {
        // At m=0, all angles are 0 → cos=1, sin=0 → unchanged.
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        let orig = x.clone();
        rope_apply(&mut x, 1, 4, 10_000.0, 0).unwrap();
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn rope_apply_rotation_preserves_norm() {
        // RoPE is a rotation, so it preserves the vector norm per row.
        let mut x = vec![1.0f32, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let orig_norms: Vec<f32> = x
            .chunks(4)
            .map(|c| c.iter().map(|v| v * v).sum::<f32>().sqrt())
            .collect();
        rope_apply(&mut x, 2, 4, 10_000.0, 3).unwrap();
        let new_norms: Vec<f32> = x
            .chunks(4)
            .map(|c| c.iter().map(|v| v * v).sum::<f32>().sqrt())
            .collect();
        for (a, b) in orig_norms.iter().zip(new_norms.iter()) {
            assert!((a - b).abs() < 1e-4, "norm changed: {a} -> {b}");
        }
    }

    #[test]
    fn rope_apply_rejects_odd_head_dim() {
        let mut x = vec![1.0f32, 2.0, 3.0];
        assert!(matches!(
            rope_apply(&mut x, 1, 3, 10_000.0, 0),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn text_decoder_step_advances_seq_len() {
        let mut s = TextDecoderStep::new();
        assert_eq!(s.seq_len, 0);
        s.advance();
        s.advance();
        assert_eq!(s.seq_len, 2);
    }
}
